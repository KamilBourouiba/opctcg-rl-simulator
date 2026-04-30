#!/usr/bin/env python3
"""
Entraînement PPO avec :
  - Masquage des actions illégales
  - GAE-λ
  - Checkpoints automatiques
  - Génération de vidéo MP4 toutes les N mises à jour (ou steps / parties)

Usage rapide :
  python scripts/train_ppo.py \\
      --deck "~/Library/Application Support/com.Batsu.OPTCGSim/NAMI BY.txt" \\
      --steps 100000 \\
      --video-every 50 \\
      --mp4 runs/training_replay

Entraînement contre le pool gauntlet (validation comme ``deck_gauntlet_web.py``) :
  paires **diversifiées** (fichiers distincts ; leaders différents si le pool le permet) à chaque
  reset ; sous-dossier filtré par ``training.gauntlet_deck_glob`` (ex. ``*.txt`` = tous les .txt valides).
  python scripts/train_ppo.py \\
      --config config.yaml \\
      --gauntlet-decks-dir decks/tournament_op15 \\
      --steps 500000

Exemple avec options avancées :
  python scripts/train_ppo.py \\
      --deck "NAMI BY.txt" --deck1 "OTHER.txt" \\
      --csv data/cards_tcgcsv.csv \\
      --steps 200000 --rollout 1024 \\
      --video-every 25 --mp4 runs/replay \\
      --checkpoint-every 10 --out checkpoints/model.pt
"""
from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import torch

from opctcg_text_sim.env import OPTextSimEnv
from opctcg_text_sim.checkpoint_io import checkpoint_dict, load_policy_checkpoint
from opctcg_text_sim.obs_rms import RunningMeanStd
from opctcg_text_sim.ppo import ActorCritic, EpisodeStats, format_wall_elapsed, train_loop
from opctcg_text_sim.vec_env import make_subproc_vec_env
from opctcg_text_sim.coreml_policy import CoreMLPolicy, AVAILABLE as COREML_AVAILABLE
from opctcg_text_sim.shmem_vec_env import SharedMemoryVecEnv

# Aligné sur ``SubprocVecEnv`` / ``SharedMemoryVecEnv`` (tous deux utilisent ``get_context("spawn")``).
# Un ``multiprocessing.Value`` créé via le contexte par défaut (souvent « fork » sous Linux)
# n’est pas garanti compatible avec des enfants « spawn » : curriculum + event_shaping faisait
# tomber les workers (EOFError) sans traceback.
_MP_SPAWN = mp.get_context("spawn")

from opctcg_text_sim.training_boot import (
    TrainingDashboard,
    _EnvFactory,
    _ResamplingGauntletEnv,
    _ResamplingGauntletFactory,
    collect_gauntlet_valid_paths,
    build_gauntlet_leader_by_path,
    pick_gauntlet_pair,
    _deep_merge_dict,
    _cfg_opt_int,
)


# ──────────────────────────────────────────────────────────────────────────────
# Génération de vidéo (une partie déterministe avec l'agent courant)
# ──────────────────────────────────────────────────────────────────────────────

def _run_greedy_episode(
    net: ActorCritic,
    env: OPTextSimEnv,
    device: torch.device,
    max_steps: int = 300,
    *,
    obs_rms: RunningMeanStd | None = None,
) -> tuple[float, bool]:
    """
    Joue une partie en mode greedy (argmax des logits) et retourne (total_reward, won).
    """
    net.eval()
    o, _ = env.reset()
    total_r = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            m_np = env.legal_actions_mask()
            if not m_np.any():
                m_np[:] = True
            o_feed = o if obs_rms is None else obs_rms.normalize_eval(o)
            x = torch.as_tensor(o_feed, dtype=torch.float32, device=device).unsqueeze(0)
            m_t = torch.as_tensor(m_np, dtype=torch.bool, device=device).unsqueeze(0)
            logits, _ = net(x, m_t)
            # Greedy : action avec le logit le plus élevé parmi les légales
            a = int(logits.argmax(dim=-1).item())
            o, r, term, trunc, _ = env.step(a)
            total_r += float(r)
            if term or trunc:
                break
    return total_r, total_r > 0


def _resolve_mp4_prefix(prefix: Path) -> Path:
    p = Path(prefix).expanduser()
    if not p.is_absolute():
        p = (ROOT / p).resolve()
    return p


def _next_video_index_from_existing(mp4_prefix: Path) -> int:
    """
    Retourne l'index max déjà présent pour <prefix>_epXXXXX.mp4 (0 si aucun).
    """
    pref = _resolve_mp4_prefix(mp4_prefix)
    parent = pref.parent
    stem = pref.name
    if not parent.is_dir():
        return 0
    pat = re.compile(rf"^{re.escape(stem)}_ep(\d{{5}})\.mp4$", re.IGNORECASE)
    best = 0
    for f in parent.iterdir():
        if not f.is_file():
            continue
        m = pat.match(f.name)
        if not m:
            continue
        try:
            best = max(best, int(m.group(1)))
        except ValueError:
            continue
    return best


def generate_video(
    net: ActorCritic,
    env_kwargs: dict,
    mp4_prefix: Path,
    video_idx: int,
    device: torch.device,
    fps: float = 1.5,
    frame_size: tuple[int, int] | None = None,
    thumb_size: tuple[int, int] | None = None,
    *,
    verbose: bool = True,
    cleanup_jsonl: bool = True,
    obs_rms: RunningMeanStd | None = None,
) -> Path | None:
    """
    Lance une partie déterministe avec l'agent actuel et génère un MP4.
    Retourne le chemin du fichier MP4 ou None en cas d'erreur.
    """
    mp4_prefix = _resolve_mp4_prefix(mp4_prefix)
    mp4_path = Path(f"{mp4_prefix}_ep{video_idx:05d}.mp4")
    jsonl_path = mp4_path.with_suffix(".jsonl")
    manifest_path = mp4_path.with_name(mp4_path.stem + ".manifest.json")

    # Créer un env avec les logs activés
    try:
        vid_env = OPTextSimEnv(
            **env_kwargs,
            animation_log_path=jsonl_path,
        )
    except TypeError:
        # Compatibilité si animation_log_path n'est pas dans **kwargs
        vid_env = OPTextSimEnv(
            env_kwargs["deck0_path"],
            env_kwargs["deck1_path"],
            env_kwargs["cards_csv"],
            env_kwargs["column_map"],
            env_kwargs.get("rules_corpus_path"),
            env_kwargs["sim_cfg"],
            obs_dim=env_kwargs.get("obs_dim", 96),
            seed=env_kwargs.get("seed", 0) + video_idx,
            animation_log_path=jsonl_path,
        )

    try:
        _run_greedy_episode(net, vid_env, device, max_steps=400, obs_rms=obs_rms)
        vid_env.close()

        if not jsonl_path.is_file() or jsonl_path.stat().st_size == 0:
            return None

        from opctcg_text_sim.replay_render_images import jsonl_to_mp4_with_images

        jsonl_to_mp4_with_images(
            jsonl_path,
            mp4_path,
            manifest_path,
            mp4_path.parent / "card_image_cache",
            fps=fps,
            frame_size=frame_size,
            thumb_size=thumb_size,
        )
        if cleanup_jsonl:
            try:
                jsonl_path.unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)
            except OSError:
                pass
        if verbose:
            print(f"  📹 Vidéo → {mp4_path.resolve()}", flush=True)
        return mp4_path
    except Exception as exc:
        if verbose:
            print(f"  ⚠️  Génération vidéo échouée : {exc}", flush=True)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="Entraîne un agent PPO sur le simulateur OPTCG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Decks
    p.add_argument(
        "--deck",
        type=Path,
        default=ROOT / "decks" / "NAMIBY.txt",
        help="Deck de l'agent (P0)",
    )
    p.add_argument(
        "--deck1",
        type=Path,
        default=None,
        help="Deck de l'adversaire IA (P1) — défaut : même que --deck",
    )
    p.add_argument(
        "--gauntlet-decks-dir",
        type=Path,
        default=None,
        dest="gauntlet_decks_dir",
        help="Dossier de .txt (ex. decks/tournament_op15) : ignore --deck/--deck1 ; "
        "chaque worker tire une paire valide comme deck_gauntlet_web.py. "
        "Défaut : training.gauntlet_decks_dir dans config.yaml (vide = mode --deck).",
    )
    # Données
    p.add_argument("--csv",   type=Path, default=None, help="Fichier CSV des cartes")
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml", help="config.yaml")
    p.add_argument(
        "--config-overlay",
        type=Path,
        action="append",
        default=None,
        dest="config_overlay",
        help="YAML fusionné en profondeur après --config (répétable, ordre conservé). "
        "Voir configs/ab/*.yaml pour des profils A/B (n_envs, minibatch, threads PyTorch).",
    )
    # Entraînement
    p.add_argument("--steps",    type=int,   default=None, help="Nombre total de steps (écrase config)")
    p.add_argument("--rollout",  type=int,   default=None, help="Longueur d'un rollout")
    p.add_argument("--lr",       type=float, default=None, help="Learning rate")
    p.add_argument("--hidden",   type=int,   default=None, help="Taille couche cachée")
    p.add_argument("--seed",     type=int,   default=None, help="Graine aléatoire")
    p.add_argument("--device",   type=str,   default="auto",
                   help="cpu | cuda | mps | auto")
    p.add_argument(
        "--n-envs",
        type=int,
        default=0,
        dest="n_envs",
        help="Nombre d'envs parallèles (SubprocVecEnv). 0 = auto (6 sur M1 Pro)",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        dest="no_compile",
        help="Désactiver torch.compile",
    )
    # Sorties
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "checkpoints" / "policy.pt",
        help="Chemin de sauvegarde du modèle final",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        dest="ckpt_every",
        help="Sauvegarder un checkpoint toutes les N mises à jour (défaut : training.checkpoint_every du YAML)",
    )
    # Vidéo
    p.add_argument(
        "--mp4",
        type=Path,
        default=ROOT / "runs" / "training_replay",
        help="Préfixe des vidéos générées (ex. runs/replay → runs/replay_ep00050.mp4)",
    )
    p.add_argument(
        "--video-every",
        type=int,
        default=None,
        dest="video_every",
        help="Générer une vidéo toutes les N mises à jour (défaut : training.video_every du YAML)",
    )
    p.add_argument(
        "--video-every-steps",
        type=int,
        default=None,
        dest="video_every_steps",
        help="Vidéo toutes les N steps d'environnement (cumulés). Ignoré si --video-every-episodes > 0.",
    )
    p.add_argument(
        "--video-every-episodes",
        type=int,
        default=None,
        dest="video_every_episodes",
        help="Vidéo toutes les N parties terminées (matchs, cumul sur les rollouts). "
        "Prioritaire sur --video-every-steps et --video-every.",
    )
    p.add_argument(
        "--no-final-video",
        action="store_true",
        dest="no_final_video",
        help="Ne pas générer la MP4 finale (économise CPU / disque en fin de run).",
    )
    p.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=None,
        dest="checkpoint_every_steps",
        help="Checkpoint .pt toutes les N steps d'environnement (défaut possible : "
        "training.checkpoint_every_steps du YAML). Si > 0, remplace --checkpoint-every.",
    )
    p.add_argument("--fps",    type=float, default=1.5, help="FPS de la vidéo")
    p.add_argument(
        "--frame",
        nargs=2, type=int, metavar=("W", "H"), default=None,
        help="Taille frame vidéo (ex. 2400 1400)",
    )
    p.add_argument(
        "--thumb",
        nargs=2, type=int, metavar=("W", "H"), default=None,
        help="Taille vignette carte (ex. 106 148)",
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Reprendre l'entraînement depuis un checkpoint .pt",
    )
    p.add_argument(
        "--effects-snapshot",
        type=Path,
        default=None,
        help="JSON effets préparsés (ex. data/card_effects_sim.json). "
        "Génération : python scripts/export_card_effects_json.py — surcharge sim.card_effects_snapshot.",
    )
    p.add_argument(
        "--max-resources",
        action="store_true",
        help="Saturation CPU : n_envs ≈ 1 worker / cœur (plafond training.n_envs_max_cap), "
        "limite FD élevée, threads BLAS=1 dans les workers.",
    )
    p.add_argument(
        "--until-stop",
        action="store_true",
        dest="until_stop",
        help="Boucle PPO jusqu’à arrêt : taper « stop » + Entrée (TTY) ou créer le fichier "
        "STOP (voir training.stop_trigger_path) si stdin n’est pas un terminal (nohup, SSH sans -t).",
    )
    args = p.parse_args()

    # ── Config ──
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        print("Config YAML : document racine attendu (mapping).", file=sys.stderr)
        return 1
    for rel_ov in args.config_overlay or []:
        ov_path = Path(rel_ov).expanduser()
        if not ov_path.is_absolute():
            ov_path = (ROOT / ov_path).resolve()
        else:
            ov_path = ov_path.resolve()
        if not ov_path.is_file():
            print(f"--config-overlay introuvable : {ov_path}", file=sys.stderr)
            return 1
        patch = yaml.safe_load(ov_path.read_text(encoding="utf-8"))
        if isinstance(patch, dict):
            _deep_merge_dict(raw, patch)
    paths = raw.get("paths", {})
    sim_cfg = dict(raw.get("sim") or {})
    train_cfg: dict = dict(raw.get("training", {}))

    # Surcharge CLI
    if args.steps   is not None: train_cfg["total_steps"]  = args.steps
    if args.rollout is not None: train_cfg["rollout_len"]   = args.rollout
    if args.lr      is not None: train_cfg["lr"]            = args.lr
    if args.hidden  is not None: train_cfg["hidden"]        = args.hidden
    if args.seed    is not None: train_cfg["seed"]          = args.seed
    if args.no_compile:          train_cfg["torch_compile"] = False
    if args.until_stop:
        train_cfg["train_until_stop"] = True

    if args.ckpt_every is None:
        args.ckpt_every = int(train_cfg.get("checkpoint_every", 20))
    if args.video_every is None:
        # 0 = pas de MP4 (aligné sur config.yaml ; évite le vieux défaut 50 si la clé manque)
        args.video_every = int(train_cfg.get("video_every", 0))

    if args.checkpoint_every_steps is None:
        cy = train_cfg.get("checkpoint_every_steps")
        if cy is not None and str(cy).strip() != "":
            args.checkpoint_every_steps = int(cy)

    if args.video_every_steps is None:
        _vs = train_cfg.get("video_every_steps")
        if _vs is not None and str(_vs).strip() != "":
            args.video_every_steps = int(_vs)
    if args.video_every_episodes is None:
        _ve = train_cfg.get("video_every_episodes")
        if _ve is not None and str(_ve).strip() != "":
            args.video_every_episodes = int(_ve)

    ckpt_every_steps = int(args.checkpoint_every_steps or 0)
    video_every_steps = int(args.video_every_steps or 0)
    if ckpt_every_steps < 0:
        ckpt_every_steps = 0
    if video_every_steps < 0:
        video_every_steps = 0

    video_every_episodes = int(args.video_every_episodes or 0)
    if video_every_episodes < 0:
        video_every_episodes = 0
    if video_every_episodes > 0:
        video_every_steps = 0

    seed = int(train_cfg.get("seed", 42))

    # ── CSV cartes (avant les decks : validation gauntlet) ──
    csv_path = args.csv
    if csv_path is None:
        rel = paths.get("cards_csv", "data/cards_stub.csv")
        pc  = Path(rel).expanduser()
        csv_path = (pc if pc.is_absolute() else ROOT / pc).resolve()
        if not csv_path.is_file():
            csv_path = (ROOT / "data" / "cards_stub.csv").resolve()
    if not csv_path.is_file():
        print(f"CSV introuvable : {csv_path}", file=sys.stderr)
        return 1

    col_map = {str(k): str(v) for k, v in raw.get("card_csv", {}).items()}
    rules_p = ROOT / paths.get("rules_corpus_out", "data/rules_corpus.txt")
    rules_p = rules_p if rules_p.is_file() else None

    # ── Decks : gauntlet (dossier) ou paire fixe --deck / --deck1 ──
    gauntlet_raw = None
    if args.gauntlet_decks_dir is not None:
        gauntlet_raw = args.gauntlet_decks_dir
    else:
        gy = train_cfg.get("gauntlet_decks_dir")
        if gy is not None and str(gy).strip():
            gauntlet_raw = gy

    gauntlet_valid_paths: list[Path] | None = None
    gauntlet_leader_map: dict[str, str] | None = None
    deck_glob = str(train_cfg.get("gauntlet_deck_glob") or "*.txt").strip() or "*.txt"
    if gauntlet_raw:
        g = Path(str(gauntlet_raw)).expanduser()
        gdir = g.resolve() if g.is_absolute() else (ROOT / g).resolve()
        gauntlet_valid_paths = collect_gauntlet_valid_paths(
            gdir, csv_path, col_map, deck_glob=deck_glob,
        )
        if len(gauntlet_valid_paths) < 1:
            print(f"Gauntlet : aucun deck valide dans {gdir} (glob={deck_glob!r})", file=sys.stderr)
            return 1
        print(
            f"Mode gauntlet : {len(gauntlet_valid_paths)} deck(s) valide(s) dans {gdir} (glob={deck_glob!r})",
            flush=True,
        )
        print(
            "  Matchups : paires diversifiées (fichiers distincts ; leaders différents si le pool le permet)",
            flush=True,
        )
        gauntlet_leader_map = build_gauntlet_leader_by_path(gauntlet_valid_paths, csv_path, col_map)
        deck0 = deck1 = None  # réservé au mode paire fixe ci-dessous
    else:
        deck0 = args.deck.expanduser()
        deck1 = (args.deck1 or args.deck).expanduser()
        if not deck0.is_file():
            print(f"Deck introuvable : {deck0}", file=sys.stderr)
            return 1
        if not deck1.is_file():
            print(f"Deck1 introuvable : {deck1}", file=sys.stderr)
            return 1

    if args.effects_snapshot is not None:
        snap = args.effects_snapshot.expanduser()
        if not snap.is_file():
            snap = (ROOT / snap).resolve()
        if not snap.is_file():
            print(f"Snapshot effets introuvable : {args.effects_snapshot}", file=sys.stderr)
            return 1
        sim_cfg["card_effects_snapshot"] = str(snap.resolve())

    # ── Device ──
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Détection plateforme / cœurs
    import platform
    import subprocess as _sp

    try:
        _chip = _sp.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except Exception:
        _chip = ""
    if not _chip:
        _chip = platform.processor() or platform.machine() or "unknown"
    _n_cpu = __import__("os").cpu_count() or 4
    is_apple_silicon = "Apple" in _chip

    from opctcg_text_sim.runtime_resources import (
        configure_main_process_torch,
        raise_process_file_limit,
        suggested_parallel_envs,
    )

    configure_main_process_torch(
        _n_cpu,
        num_threads=_cfg_opt_int(train_cfg, "main_torch_num_threads"),
        num_interop_threads=_cfg_opt_int(train_cfg, "main_torch_interop_threads"),
    )

    resource_profile = (
        "max"
        if args.max_resources
        else str(train_cfg.get("resource_profile", "balanced")).lower()
    )

    # Nombre d'envs parallèles
    n_envs = args.n_envs
    if n_envs == 0:
        cfg_n_envs = int(train_cfg.get("n_envs", 0))
        if args.max_resources or (
            resource_profile == "max" and cfg_n_envs == 0
        ):
            raise_process_file_limit(max(16384, _n_cpu * 256))
            cap = int(train_cfg.get("n_envs_max_cap", 64))
            n_envs = suggested_parallel_envs(_n_cpu, cap=cap, profile="max")
        elif cfg_n_envs > 0:
            n_envs = cfg_n_envs
        elif resource_profile == "conservative":
            n_envs = suggested_parallel_envs(_n_cpu, profile="conservative")
        elif resource_profile == "balanced" and not is_apple_silicon:
            n_envs = suggested_parallel_envs(_n_cpu, profile="balanced")
        elif is_apple_silicon:
            n_envs = _n_cpu
        else:
            n_envs = max(1, _n_cpu - 1)
    if n_envs > 1:
        # Beaucoup de workers (pipes + shared memory) : sous macOS, une limite trop
        # basse déclenche OSError 24 (EMFILE) avant même le premier rollout.
        fd_target = max(4096, n_envs * 64)
        if n_envs >= 24:
            fd_target = max(fd_target, 16384)
        raise_process_file_limit(fd_target)
    # Rollout via processus dédiés + shared memory si plusieurs envs, ou 1 worker si Core ML (ANE).
    use_vec_rollout = n_envs > 1 or (COREML_AVAILABLE and n_envs == 1)

    if n_envs > 1:
        print(
            f"Mode vectorisé : {n_envs} envs parallèles ({_chip})  "
            f"[profil={resource_profile}]",
            flush=True,
        )
    elif use_vec_rollout:
        print(
            f"Mode Core ML : 1 env déporté (shared memory) pour inférence ANE ({_chip})",
            flush=True,
        )

    print(f"Device : {device}  |  Chip : {_chip}  |  CPU cores : {_n_cpu}")
    if platform.system() == "Darwin" and not COREML_AVAILABLE:
        print(
            "Astuce : pip install coremltools → rollout sur Neural Engine (souvent >> MPS seul).",
            flush=True,
        )

    # Rollout / updates (curriculum event_shaping + dashboard ; avant création des envs)
    total_steps = int(train_cfg.get("total_steps", 50_000))
    rollout_len = int(train_cfg.get("rollout_len", 512))
    steps_per_env = max(1, rollout_len // max(1, n_envs))
    actual_rollout = steps_per_env * max(1, n_envs)
    total_updates = max(1, total_steps // actual_rollout)
    _es_cfg = sim_cfg.get("event_shaping") or {}
    shaping_progress_ref = (
        _MP_SPAWN.Value("d", 0.0) if bool(_es_cfg.get("enabled", False)) else None
    )

    # ── Environnement principal ──
    obs_dim = int(train_cfg.get("obs_dim", 96))
    if gauntlet_valid_paths is not None:
        env = _ResamplingGauntletFactory(
            valid_paths=gauntlet_valid_paths,
            cards_csv=csv_path,
            column_map=col_map,
            rules_corpus_path=rules_p,
            sim_cfg=sim_cfg,
            obs_dim=obs_dim,
            leader_by_resolved_str=gauntlet_leader_map,
            shaping_progress_ref=shaping_progress_ref,
        )(seed)
        env.reset(seed=seed)
    else:
        assert deck0 is not None and deck1 is not None
        env = OPTextSimEnv(
            deck0, deck1, csv_path, col_map, rules_p, sim_cfg,
            obs_dim=obs_dim,
            seed=seed,
            shaping_progress_ref=shaping_progress_ref,
        )

    # Kwargs pour les envs vidéo (paire fixe ou tirage à la génération si gauntlet)
    if gauntlet_valid_paths is not None:
        v0, v1 = pick_gauntlet_pair(
            gauntlet_valid_paths,
            seed,
            77_777,
            leader_by_resolved_str=gauntlet_leader_map,
        )
        env_kwargs = dict(
            deck0_path=v0,
            deck1_path=v1,
            cards_csv=csv_path,
            column_map=col_map,
            rules_corpus_path=rules_p,
            sim_cfg=sim_cfg,
            obs_dim=obs_dim,
            seed=seed,
        )
    else:
        assert deck0 is not None and deck1 is not None
        env_kwargs = dict(
            deck0_path=deck0,
            deck1_path=deck1,
            cards_csv=csv_path,
            column_map=col_map,
            rules_corpus_path=rules_p,
            sim_cfg=sim_cfg,
            obs_dim=obs_dim,
            seed=seed,
        )

    # ── Réseau + normalisation des observations (running mean/std) ──
    hidden  = int(train_cfg.get("hidden", 256))
    n_res   = int(train_cfg.get("n_res",    2))
    n_act   = int(env.action_space.n)
    _obs_norm = bool(train_cfg.get("obs_normalize", False))
    _obs_clip = float(train_cfg.get("obs_normalize_clip", 10.0))
    _rew_norm = bool(train_cfg.get("reward_normalize", False))
    _rew_eps = float(train_cfg.get("reward_normalize_epsilon", 1e-4))
    obs_rms = (
        RunningMeanStd((obs_dim,), clip_obs=_obs_clip)
        if _obs_norm
        else None
    )
    reward_rms = (
        RunningMeanStd((1,), epsilon=_rew_eps, clip_obs=0.0)
        if _rew_norm
        else None
    )
    net = ActorCritic(obs_dim, n_act, hidden, n_res).to(device)

    if args.resume is not None and args.resume.is_file():
        load_policy_checkpoint(
            args.resume, net, device, obs_rms=obs_rms, reward_rms=reward_rms,
        )
        print(f"Reprise depuis : {args.resume}")

    # ── Dossiers de sortie ──
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.mp4 = Path(args.mp4).expanduser()
    if not args.mp4.is_absolute():
        args.mp4 = (ROOT / args.mp4).resolve()
    else:
        args.mp4 = args.mp4.resolve()
    args.mp4.parent.mkdir(parents=True, exist_ok=True)

    # ── VecEnv (envs parallèles, ou 1× shm sur macOS pour Core ML) ──
    vec_env = None
    # Shared memory : moins d’IPC que les pipes Subproc (Linux + macOS).
    # Sur macOS avec coremltools : même n_envs==1 utilise 1 worker shm + ANE (évite MPS↔CPU à chaque step).
    use_shmem = use_vec_rollout
    n_act = int(env.action_space.n)

    if use_vec_rollout:
        if gauntlet_valid_paths is not None:
            env_factory = _ResamplingGauntletFactory(
                valid_paths=gauntlet_valid_paths,
                cards_csv=csv_path,
                column_map=col_map,
                rules_corpus_path=rules_p,
                sim_cfg=sim_cfg,
                obs_dim=obs_dim,
                leader_by_resolved_str=gauntlet_leader_map,
                shaping_progress_ref=shaping_progress_ref,
            )
        else:
            env_factory = _EnvFactory(
                deck0=deck0, deck1=deck1,
                cards_csv=csv_path, column_map=col_map,
                rules_corpus_path=rules_p, sim_cfg=sim_cfg,
                obs_dim=obs_dim,
                shaping_progress_ref=shaping_progress_ref,
            )
        if use_shmem:
            _shmem_label = (
                f"{n_envs}× SharedMemoryVecEnv + CoreML (ANE)"
                if COREML_AVAILABLE
                else f"{n_envs}× SharedMemoryVecEnv + PyTorch (rollout)"
            )
            print(f"Mode : {_shmem_label}…", flush=True)
            vec_env = SharedMemoryVecEnv(
                env_factory, n_envs=n_envs,
                obs_dim=obs_dim, n_act=n_act,
                base_seed=seed,
            )
        else:
            print(f"Lancement de {n_envs} processus workers (Subproc)…", flush=True)
            vec_env = make_subproc_vec_env(env_factory, n_envs=n_envs, base_seed=seed)
        print("Workers prêts.")
        # Libère l’env probe : même ``shaping_progress_ref`` / gros caches ; le laisser ouvert
        # en parallèle des workers a contribué aux crashes Linux (EOFError) avec event_shaping.
        try:
            env.close()
        except Exception:
            pass
        env = None  # type: ignore[assignment]

    # ── CoreMLPolicy (ANE) : dès qu’on a un VecEnv (y compris 1× shm sur macOS) ──
    coreml_policy = None
    if COREML_AVAILABLE and vec_env is not None:
        net_cpu = ActorCritic(obs_dim, n_act, hidden, n_res)
        coreml_policy = CoreMLPolicy(net_cpu, obs_dim=obs_dim, n_act=n_act)
        # Copier les poids initiaux vers CoreML
        net_orig = net._orig_mod if hasattr(net, "_orig_mod") else net
        coreml_policy.sync(net_orig)

    # ── Paramètres de la boucle ──
    train_until_stop = bool(train_cfg.get("train_until_stop", False))
    # Démarre après le dernier index existant pour éviter toute collision de noms.
    video_idx = _next_video_index_from_existing(args.mp4)

    stop_trigger_path: Path | None = None
    if train_until_stop:
        stop_trigger_path = Path(str(train_cfg.get("stop_trigger_path", "STOP"))).expanduser()
        if not stop_trigger_path.is_absolute():
            stop_trigger_path = Path.cwd() / stop_trigger_path
        stop_trigger_path = stop_trigger_path.resolve()

    dash = TrainingDashboard(total_updates, actual_rollout, infinite=train_until_stop)

    video_schedule_episodes = video_every_episodes
    video_schedule_steps = video_every_steps
    video_schedule_updates = int(args.video_every)

    compile_status = "ON" if bool(train_cfg.get("torch_compile", False)) else "OFF"
    print(f"\n{'='*70}")
    print(f"  OPTCG PPO Training  —  {_chip}")
    print(f"  Device     : {device}  |  torch.compile : {compile_status}")
    try:
        print(
            f"  PyTorch CPU (process principal) : intra={torch.get_num_threads()}  "
            f"interop={torch.get_num_interop_threads()}",
            flush=True,
        )
    except Exception:
        pass
    print(f"  Envs       : {n_envs}x parallèles  |  Rollout/env: {steps_per_env}  |  Total/update: {actual_rollout}")
    if train_until_stop:
        assert stop_trigger_path is not None
        print("  Mode       : jusqu'à arrêt (PPO réel, boucle jusqu'à « stop » ou fichier STOP)")
        print(f"  Arrêt      : ligne « stop » + Entrée (terminal) et/ou : touch {stop_trigger_path}")
    else:
        print(f"  Steps      : {total_steps:,}   →  {total_updates} updates")
    print(f"  Minibatch  : {train_cfg.get('minibatch', 128)}  |  Epochs: {train_cfg.get('epochs', 6)}")
    print(f"  Réseau     : obs={obs_dim} → hidden={hidden} → act={n_act}")
    if obs_rms is not None:
        print(f"  Obs norm   : running mean/std (clip={obs_rms.clip_obs})")
    if reward_rms is not None:
        print(f"  Reward σ   : running scale (ε={reward_rms.epsilon})")
    _tkl = train_cfg.get("target_kl")
    if _tkl is not None and float(_tkl) > 0:
        print(f"  PPO KL cap : arrêt intra-update si KL≈ > {float(_tkl)}")
    _ce0, _ce1 = float(train_cfg.get("clip_eps", 0.2)), float(train_cfg.get("clip_eps_end", train_cfg.get("clip_eps", 0.2)))
    if _ce1 != _ce0:
        print(f"  Clip ε     : {_ce0} → {_ce1} (linéaire sur la progression)")
    if video_schedule_episodes > 0:
        print(f"  Video every: {video_schedule_episodes:,} parties (épisodes)  ({args.fps} fps)")
        print(
            f"  → Une MP4 est générée quand le **nombre cumulé** de parties terminées "
            f"(tous les envs) atteint {video_schedule_episodes:,}. "
            f"Le « ep= » dans les logs PPO = parties finies **sur ce rollout seulement**, pas le cumul.",
            flush=True,
        )
    elif video_schedule_steps > 0:
        print(f"  Video every: {video_schedule_steps:,} env steps  ({args.fps} fps)")
    else:
        _v1 = bool(train_cfg.get("video_on_first_update", True))
        if _v1 and video_schedule_updates > 1:
            _hint = f" — dès update 1 puis toutes les {video_schedule_updates} updates"
        elif not _v1 and video_schedule_updates > 0:
            _hint = f" — 1re MP4 à l’update {video_schedule_updates}"
        else:
            _hint = ""
        print(f"  Video every: {video_schedule_updates} updates  ({args.fps} fps){_hint}")
    if ckpt_every_steps > 0:
        print(f"  Checkpoint : every {ckpt_every_steps:,} env steps → {args.out.parent}/")
    else:
        print(f"  Checkpoint : every {args.ckpt_every} updates → {args.out.parent}/")
    if video_schedule_episodes > 0 or video_schedule_steps > 0 or video_schedule_updates > 0:
        print(f"  MP4 dossier : {args.mp4.parent}/  (vidéos CPU, file d’attente 1 thread)")
    print(f"{'='*70}\n")

    # Seuils steps (cumul env) ; listes pour muter depuis le callback sans nonlocal verbeux
    next_ckpt_step = [ckpt_every_steps] if ckpt_every_steps > 0 else [0]
    next_video_step = [video_schedule_steps] if video_schedule_steps > 0 else [0]
    episodes_done = [0]
    next_video_ep = [video_schedule_episodes] if video_schedule_episodes > 0 else [0]

    video_executor: ThreadPoolExecutor | None = None
    if video_schedule_episodes > 0 or video_schedule_steps > 0 or video_schedule_updates > 0:
        video_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="opctcg_vid",
        )

    def _state_dict_cpu_clone(mod: ActorCritic) -> dict[str, torch.Tensor]:
        src = mod._orig_mod if hasattr(mod, "_orig_mod") else mod
        return {k: v.detach().cpu().clone() for k, v in src.state_dict().items()}

    def _background_video_job(
        sd_cpu: dict[str, torch.Tensor],
        vidx: int,
        fps_: float,
        frame_sz: tuple[int, int] | None,
        thumb_sz: tuple[int, int] | None,
        gauntlet_paths: list[Path] | None,
        leader_map_arg: dict[str, str] | None,
        rms_sd: dict | None,
        rms_clip: float,
    ) -> None:
        vid_net: ActorCritic | None = None
        try:
            vid_net = ActorCritic(obs_dim, n_act, hidden, n_res)
            vid_net.load_state_dict(sd_cpu)
            vid_net.eval()
            vid_rms: RunningMeanStd | None = None
            if rms_sd is not None:
                vid_rms = RunningMeanStd((obs_dim,), clip_obs=rms_clip)
                vid_rms.load_state_dict(rms_sd)
            ek = dict(env_kwargs)
            if gauntlet_paths is not None:
                g0, g1 = pick_gauntlet_pair(
                    gauntlet_paths,
                    int(time.time() * 1_000_000) % (2**31),
                    vidx * 97_981,
                    leader_by_resolved_str=leader_map_arg,
                )
                ek["deck0_path"] = g0
                ek["deck1_path"] = g1
            out = generate_video(
                vid_net,
                ek,
                args.mp4,
                vidx,
                torch.device("cpu"),
                fps=fps_,
                frame_size=frame_sz,
                thumb_size=thumb_sz,
                verbose=False,
                cleanup_jsonl=True,
                obs_rms=vid_rms,
            )
            if out:
                print(f"  Vidéo prête → {out.resolve()}", flush=True)
        except Exception as exc:
            print(f"  ⚠️ Vidéo (arrière-plan) : {exc}", flush=True)
        finally:
            del sd_cpu
            if vid_net is not None:
                del vid_net

    def schedule_training_video(net_: ActorCritic, vidx: int) -> None:
        if video_executor is None:
            return
        print(
            f"  Vidéo MP4 mise en file d’attente (indice {vidx}, encodage CPU en arrière-plan)…",
            flush=True,
        )
        sd = _state_dict_cpu_clone(net_)
        _rms_snap = obs_rms.state_dict() if obs_rms is not None else None
        video_executor.submit(
            _background_video_job,
            sd,
            vidx,
            args.fps,
            tuple(args.frame) if args.frame else None,
            tuple(args.thumb) if args.thumb else None,
            gauntlet_valid_paths,
            gauntlet_leader_map,
            _rms_snap,
            _obs_clip,
        )

    # ── Callback ──
    def on_update(it: int, net_: ActorCritic, stats: EpisodeStats, metrics: dict) -> None:
        nonlocal video_idx
        update_num = it + 1
        extra = ""
        total_env_steps = update_num * actual_rollout

        if shaping_progress_ref is not None:
            _an = int(_es_cfg.get("anneal_updates", 0))
            _den = float(_an) if _an > 0 else float(max(1, total_updates))
            _p = min(1.0, float(update_num) / _den)
            with shaping_progress_ref.get_lock():
                shaping_progress_ref.value = _p

        # Checkpoint (par steps ou par updates)
        if ckpt_every_steps > 0:
            while next_ckpt_step[0] > 0 and total_env_steps >= next_ckpt_step[0]:
                marker = next_ckpt_step[0]
                ckpt = args.out.parent / f"policy_step{marker:012d}.pt"
                torch.save(checkpoint_dict(net_, obs_rms=obs_rms, reward_rms=reward_rms), ckpt)
                extra += f"💾{ckpt.name} "
                next_ckpt_step[0] += ckpt_every_steps
        elif args.ckpt_every > 0 and update_num % args.ckpt_every == 0:
            ckpt = args.out.parent / f"policy_step{(update_num * actual_rollout):08d}.pt"
            torch.save(checkpoint_dict(net_, obs_rms=obs_rms, reward_rms=reward_rms), ckpt)
            extra += f"💾{ckpt.name} "

        # MP4 en arrière-plan (1 worker : file d’attente naturelle dans le ThreadPoolExecutor).
        if video_schedule_episodes > 0:
            episodes_done[0] += int(stats.n_episodes)
            _ep_bar = next_video_ep[0]
            while next_video_ep[0] > 0 and episodes_done[0] >= next_video_ep[0]:
                video_idx += 1
                schedule_training_video(net_, video_idx)
                next_video_ep[0] += video_schedule_episodes
            extra += f"epΣ{episodes_done[0]}/{_ep_bar} "
        elif video_schedule_steps > 0:
            while next_video_step[0] > 0 and total_env_steps >= next_video_step[0]:
                video_idx += 1
                schedule_training_video(net_, video_idx)
                next_video_step[0] += video_schedule_steps
        elif video_schedule_updates > 0:
            _first_vid = bool(train_cfg.get("video_on_first_update", True))
            _periodic = update_num % video_schedule_updates == 0
            _early = _first_vid and update_num == 1
            if _early or _periodic:
                video_idx += 1
                schedule_training_video(net_, video_idx)

        dash.update(it, stats, metrics, extra.strip())

    stop_event: threading.Event | None = None
    if train_until_stop:
        assert stop_trigger_path is not None
        stop_event = threading.Event()

        def _stdin_stop_watcher() -> None:
            try:
                for line in sys.stdin:
                    if line.strip().lower() == "stop":
                        stop_event.set()
                        return
            except (EOFError, OSError):
                return

        def _file_stop_watcher() -> None:
            p = stop_trigger_path
            while not stop_event.is_set():
                try:
                    if p.is_file():
                        stop_event.set()
                        try:
                            p.unlink()
                        except OSError:
                            pass
                        return
                except OSError:
                    pass
                time.sleep(1.5)

        threading.Thread(target=_file_stop_watcher, daemon=True).start()
        if sys.stdin.isatty():
            threading.Thread(target=_stdin_stop_watcher, daemon=True).start()

    # Métriques JSONL : chemins relatifs = racine du dépôt (aligné avec ``ppo.py``).
    _mj = train_cfg.get("metrics_jsonl")
    if _mj:
        _mp = Path(str(_mj)).expanduser()
        if not _mp.is_absolute():
            train_cfg = dict(train_cfg)
            train_cfg["metrics_jsonl"] = str((ROOT / _mp).resolve())

    # ── Entraînement ──
    try:
        net = train_loop(
            env,
            train_cfg,
            device=device,
            vec_env=vec_env,
            coreml_policy=coreml_policy,
            on_update_callback=on_update,
            stop_event=stop_event,
            train_until_stop=train_until_stop,
            net=net,
            obs_rms=obs_rms,
            reward_rms=reward_rms,
        )
    finally:
        if video_executor is not None:
            video_executor.shutdown(wait=True)
    dash.newline()

    if vec_env is not None:
        vec_env.close()

    # ── Sauvegarde finale ──
    torch.save(checkpoint_dict(net, obs_rms=obs_rms, reward_rms=reward_rms), args.out)
    print(f"\nModèle final → {args.out}")

    # ── Vidéo finale (MP4 ; jsonl temporaire nettoyé si cleanup_jsonl) ──
    ek_final = dict(env_kwargs)
    if gauntlet_valid_paths is not None:
        f0, f1 = pick_gauntlet_pair(
            gauntlet_valid_paths,
            seed,
            888_888,
            leader_by_resolved_str=gauntlet_leader_map,
        )
        ek_final["deck0_path"] = f0
        ek_final["deck1_path"] = f1
    if not args.no_final_video:
        print("Génération de la vidéo finale…", flush=True)
        final_idx = max(video_idx + 1, _next_video_index_from_existing(args.mp4) + 1)
        generate_video(
            net,
            ek_final,
            args.mp4,
            final_idx,
            device,
            fps=args.fps,
            frame_size=tuple(args.frame) if args.frame else None,
            thumb_size=tuple(args.thumb) if args.thumb else None,
            obs_rms=obs_rms,
        )
    else:
        print("Vidéo finale ignorée (--no-final-video).", flush=True)

    if env is not None:
        env.close()

    # ── Résumé ──
    _wall_done = time.time() - dash.start_time
    print(f"\n{'='*70}")
    print("  Entraînement terminé !")
    print(f"  Durée totale (mur) : {format_wall_elapsed(_wall_done)}")
    print(f"  Modèle : {args.out}")
    print(f"  Vidéos : {args.mp4.parent}/ (préfixe : {args.mp4.name})")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
