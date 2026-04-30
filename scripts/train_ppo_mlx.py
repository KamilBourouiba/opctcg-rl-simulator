#!/usr/bin/env python3
"""
Entraînement PPO **MLX** sur Apple Silicon (Metal) — pas de PyTorch sur le chemin critique.

Usage (aligné sur ``train_ppo.py``) :
  python scripts/train_ppo_mlx.py --config config.yaml --steps 100000 \\
      --out checkpoints/policy_mlx.pt

Checkpoints : pickle (``save_mlx_checkpoint``) avec ``mlx_flat`` + ``obs_rms`` / ``reward_rms`` optionnels.
Les anciens fichiers ``torch.save`` se rechargent si PyTorch est installé.
Les poids ``ActorCritic`` PyTorch (.pt classiques) ne sont pas chargés tels quels — utiliser ce format MLX.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import platform
import subprocess as _sp
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opctcg_text_sim.env import OPTextSimEnv
from opctcg_text_sim.mlx_ppo import (
    ActorCriticMLX,
    load_mlx_checkpoint,
    mlx_checkpoint_dict,
    save_mlx_checkpoint,
    train_loop_mlx,
)
from opctcg_text_sim.obs_rms import RunningMeanStd
from opctcg_text_sim.shmem_vec_env import SharedMemoryVecEnv
from opctcg_text_sim.training_boot import (
    TrainingDashboard,
    _EnvFactory,
    _ResamplingGauntletFactory,
    _cfg_opt_int,
    _deep_merge_dict,
    build_gauntlet_leader_by_path,
    collect_gauntlet_valid_paths,
)

from opctcg_text_sim.runtime_resources import (
    configure_main_process_torch,
    raise_process_file_limit,
    suggested_parallel_envs,
)

_MP_SPAWN = mp.get_context("spawn")


def main() -> int:
    p = argparse.ArgumentParser(
        description="PPO MLX (Apple Silicon) — simulateur OPTCG texte",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--deck", type=Path, default=ROOT / "decks" / "NAMIBY.txt")
    p.add_argument("--deck1", type=Path, default=None)
    p.add_argument("--gauntlet-decks-dir", type=Path, default=None, dest="gauntlet_decks_dir")
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--config-overlay", type=Path, action="append", default=None, dest="config_overlay")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--rollout", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-envs", type=int, default=0, dest="n_envs")
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "policy_mlx.pt")
    p.add_argument("--checkpoint-every", type=int, default=None, dest="ckpt_every")
    p.add_argument("--checkpoint-every-steps", type=int, default=None, dest="checkpoint_every_steps")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--effects-snapshot", type=Path, default=None, dest="effects_snapshot")
    p.add_argument("--max-resources", action="store_true", dest="max_resources")
    args = p.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        print("Config YAML : racine mapping attendue.", file=sys.stderr)
        return 1
    for rel_ov in args.config_overlay or []:
        ov_path = Path(rel_ov).expanduser()
        ov_path = ov_path.resolve() if ov_path.is_absolute() else (ROOT / ov_path).resolve()
        patch = yaml.safe_load(ov_path.read_text(encoding="utf-8"))
        if isinstance(patch, dict):
            _deep_merge_dict(raw, patch)

    paths = raw.get("paths", {})
    sim_cfg = dict(raw.get("sim") or {})
    train_cfg: dict = dict(raw.get("training", {}))

    if args.steps is not None:
        train_cfg["total_steps"] = args.steps
    if args.rollout is not None:
        train_cfg["rollout_len"] = args.rollout
    if args.lr is not None:
        train_cfg["lr"] = args.lr
    if args.hidden is not None:
        train_cfg["hidden"] = args.hidden
    if args.seed is not None:
        train_cfg["seed"] = args.seed

    ckpt_every_steps = int(args.checkpoint_every_steps or train_cfg.get("checkpoint_every_steps") or 0)
    if ckpt_every_steps < 0:
        ckpt_every_steps = 0
    args.ckpt_every = int(args.ckpt_every if args.ckpt_every is not None else train_cfg.get("checkpoint_every", 20))

    csv_path = args.csv
    if csv_path is None:
        rel = paths.get("cards_csv", "data/cards_stub.csv")
        pc = Path(rel).expanduser()
        csv_path = (pc if pc.is_absolute() else ROOT / pc).resolve()
        if not csv_path.is_file():
            csv_path = (ROOT / "data" / "cards_stub.csv").resolve()
    if not csv_path.is_file():
        print(f"CSV introuvable : {csv_path}", file=sys.stderr)
        return 1

    col_map = {str(k): str(v) for k, v in raw.get("card_csv", {}).items()}
    rules_p = ROOT / paths.get("rules_corpus_out", "data/rules_corpus.txt")
    rules_p = rules_p if rules_p.is_file() else None

    if args.effects_snapshot is not None:
        snap = args.effects_snapshot.expanduser()
        snap = snap.resolve() if snap.is_absolute() else (ROOT / snap).resolve()
        if not snap.is_file():
            print(f"Snapshot effets introuvable : {args.effects_snapshot}", file=sys.stderr)
            return 1
        sim_cfg["card_effects_snapshot"] = str(snap)

    gauntlet_raw = args.gauntlet_decks_dir or train_cfg.get("gauntlet_decks_dir")
    gauntlet_valid_paths: list[Path] | None = None
    gauntlet_leader_map: dict[str, str] | None = None
    deck_glob = str(train_cfg.get("gauntlet_deck_glob") or "*.txt").strip() or "*.txt"
    deck0 = deck1 = None

    if gauntlet_raw and str(gauntlet_raw).strip():
        gdir = Path(str(gauntlet_raw)).expanduser()
        gdir = gdir.resolve() if gdir.is_absolute() else (ROOT / gdir).resolve()
        gauntlet_valid_paths = collect_gauntlet_valid_paths(gdir, csv_path, col_map, deck_glob=deck_glob)
        if len(gauntlet_valid_paths) < 1:
            print(f"Gauntlet : aucun deck valide dans {gdir}", file=sys.stderr)
            return 1
        gauntlet_leader_map = build_gauntlet_leader_by_path(gauntlet_valid_paths, csv_path, col_map)
        print(f"Mode gauntlet MLX : {len(gauntlet_valid_paths)} deck(s) dans {gdir}", flush=True)
    else:
        deck0 = args.deck.expanduser()
        deck1 = (args.deck1 or args.deck).expanduser()
        if not deck0.is_file():
            print(f"Deck introuvable : {deck0}", file=sys.stderr)
            return 1
        if not deck1.is_file():
            print(f"Deck1 introuvable : {deck1}", file=sys.stderr)
            return 1

    seed = int(train_cfg.get("seed", 42))
    obs_dim = int(train_cfg.get("obs_dim", 96))

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

    configure_main_process_torch(
        _n_cpu,
        num_threads=_cfg_opt_int(train_cfg, "main_torch_num_threads"),
        num_interop_threads=_cfg_opt_int(train_cfg, "main_torch_interop_threads"),
    )

    resource_profile = "max" if args.max_resources else str(train_cfg.get("resource_profile", "balanced")).lower()
    n_envs = args.n_envs
    if n_envs == 0:
        cfg_n_envs = int(train_cfg.get("n_envs", 0))
        if args.max_resources or (resource_profile == "max" and cfg_n_envs == 0):
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
        fd_target = max(4096, n_envs * 64)
        if n_envs >= 24:
            fd_target = max(fd_target, 16384)
        raise_process_file_limit(fd_target)

    total_steps = int(train_cfg.get("total_steps", 50_000))
    rollout_len = int(train_cfg.get("rollout_len", 512))
    steps_per_env = max(1, rollout_len // max(1, n_envs))
    actual_rollout = steps_per_env * max(1, n_envs)
    total_updates = max(1, total_steps // actual_rollout)

    _es_cfg = sim_cfg.get("event_shaping") or {}
    shaping_progress_ref = _MP_SPAWN.Value("d", 0.0) if bool(_es_cfg.get("enabled", False)) else None

    # Probe env
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

    # Probe : lecture de ``n_act`` ; fermé après instanciation du VecEnv.

    hidden = int(train_cfg.get("hidden", 256))
    n_res = int(train_cfg.get("n_res", 2))
    n_act = int(env.action_space.n)

    _obs_norm = bool(train_cfg.get("obs_normalize", False))
    _obs_clip = float(train_cfg.get("obs_normalize_clip", 10.0))
    _rew_norm = bool(train_cfg.get("reward_normalize", False))
    _rew_eps = float(train_cfg.get("reward_normalize_epsilon", 1e-4))
    obs_rms = RunningMeanStd((obs_dim,), clip_obs=_obs_clip) if _obs_norm else None
    reward_rms = RunningMeanStd((1,), epsilon=_rew_eps, clip_obs=0.0) if _rew_norm else None

    net = ActorCriticMLX(obs_dim, n_act, hidden, n_res)

    if args.resume is not None and args.resume.is_file():
        load_mlx_checkpoint(args.resume, net, obs_rms=obs_rms, reward_rms=reward_rms)
        print(f"Reprise MLX depuis : {args.resume}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # MLX : rollout vectorisé dès n_envs ≥ 1 via SharedMemoryVecEnv (batch Metal).
    use_vec_rollout = n_envs >= 1
    vec_env = None
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
                deck0=deck0,
                deck1=deck1,
                cards_csv=csv_path,
                column_map=col_map,
                rules_corpus_path=rules_p,
                sim_cfg=sim_cfg,
                obs_dim=obs_dim,
                shaping_progress_ref=shaping_progress_ref,
            )
        print(f"Mode MLX : {n_envs}× SharedMemoryVecEnv + batch MLX ({_chip})", flush=True)
        vec_env = SharedMemoryVecEnv(
            env_factory,
            n_envs=n_envs,
            obs_dim=obs_dim,
            n_act=n_act,
            base_seed=seed,
        )
        print("Workers prêts.")
        try:
            env.close()
        except Exception:
            pass
        env = None

    train_until_stop = bool(train_cfg.get("train_until_stop", False))
    dash = TrainingDashboard(total_updates, actual_rollout, infinite=train_until_stop)

    next_ckpt_step = [ckpt_every_steps] if ckpt_every_steps > 0 else [0]

    def on_update(it: int, net_: ActorCriticMLX, stats, metrics: dict) -> None:
        update_num = it + 1
        extra = ""
        total_env_steps = update_num * actual_rollout

        if shaping_progress_ref is not None:
            _an = int(_es_cfg.get("anneal_updates", 0))
            _den = float(_an) if _an > 0 else float(max(1, total_updates))
            _p = min(1.0, float(update_num) / _den)
            with shaping_progress_ref.get_lock():
                shaping_progress_ref.value = _p

        if ckpt_every_steps > 0:
            while next_ckpt_step[0] > 0 and total_env_steps >= next_ckpt_step[0]:
                marker = next_ckpt_step[0]
                ckpt = args.out.parent / f"policy_mlx_step{marker:012d}.pt"
                save_mlx_checkpoint(ckpt, mlx_checkpoint_dict(net_, obs_rms=obs_rms, reward_rms=reward_rms))
                extra += f"💾{ckpt.name} "
                next_ckpt_step[0] += ckpt_every_steps
        elif args.ckpt_every > 0 and update_num % args.ckpt_every == 0:
            ckpt = args.out.parent / f"policy_mlx_step{(update_num * actual_rollout):08d}.pt"
            save_mlx_checkpoint(ckpt, mlx_checkpoint_dict(net_, obs_rms=obs_rms, reward_rms=reward_rms))
            extra += f"💾{ckpt.name} "

        dash.update(it, stats, metrics, extra.strip())

    stop_event: threading.Event | None = None
    if train_until_stop:
        stop_event = threading.Event()
        stop_trigger_path = Path(str(train_cfg.get("stop_trigger_path", "STOP"))).expanduser()
        if not stop_trigger_path.is_absolute():
            stop_trigger_path = Path.cwd() / stop_trigger_path
        stop_trigger_path = stop_trigger_path.resolve()

        def _file_stop_watcher() -> None:
            while not stop_event.is_set():
                try:
                    if stop_trigger_path.is_file():
                        stop_event.set()
                        try:
                            stop_trigger_path.unlink()
                        except OSError:
                            pass
                        return
                except OSError:
                    pass
                time.sleep(1.5)

        threading.Thread(target=_file_stop_watcher, daemon=True).start()

    _mj = train_cfg.get("metrics_jsonl")
    if _mj:
        _mp = Path(str(_mj)).expanduser()
        if not _mp.is_absolute():
            train_cfg = dict(train_cfg)
            train_cfg["metrics_jsonl"] = str((ROOT / _mp).resolve())

    print(f"\n{'='*70}\n  OPTCG PPO — backend **MLX** (Metal)\n  Chip : {_chip}  |  envs : {n_envs}\n{'='*70}\n")

    try:
        train_loop_mlx(
            env,
            train_cfg,
            vec_env=vec_env,
            net=net,
            obs_rms=obs_rms,
            reward_rms=reward_rms,
            on_update_callback=on_update,
            stop_event=stop_event,
            train_until_stop=train_until_stop,
        )
    finally:
        if vec_env is not None:
            vec_env.close()

    save_mlx_checkpoint(args.out, mlx_checkpoint_dict(net, obs_rms=obs_rms, reward_rms=reward_rms))
    print(f"\nOK — politique MLX finale → {args.out.resolve()}")
    dash.newline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
