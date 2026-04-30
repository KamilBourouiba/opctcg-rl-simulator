#!/usr/bin/env python3
"""
Génère une MP4 à partir d’un checkpoint (sans ré-entraînement).
Réutilise la même logique que la vidéo finale de ``train_ppo.py`` (greedy + FFmpeg).

Exemple :
  python scripts/checkpoint_to_video.py \\
      --checkpoint checkpoints/ablation_seed42.pt \\
      --config config.yaml --mp4 runs/training_replay
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opctcg_text_sim.checkpoint_io import load_policy_checkpoint, torch_load_train_checkpoint
from opctcg_text_sim.obs_rms import RunningMeanStd
from opctcg_text_sim.ppo import ActorCritic


def _pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _load_generate_video():
    spec = importlib.util.spec_from_file_location(
        "_opctcg_train_ppo", ROOT / "scripts" / "train_ppo.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Impossible de charger train_ppo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_video


def main() -> int:
    p = argparse.ArgumentParser(description="Checkpoint → MP4 (replay greedy)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--deck", type=Path, default=ROOT / "decks" / "NAMIBY.txt")
    p.add_argument("--deck1", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument(
        "--mp4",
        type=Path,
        default=ROOT / "runs" / "training_replay",
        help="Préfixe des MP4 (comme --mp4 de train_ppo)",
    )
    p.add_argument("--video-idx", type=int, default=9999)
    p.add_argument("--fps", type=float, default=1.5)
    p.add_argument("--frame", nargs=2, type=int, metavar=("W", "H"), default=None)
    p.add_argument("--thumb", nargs=2, type=int, metavar=("W", "H"), default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths = raw.get("paths") or {}
    train_cfg = raw.get("training") or {}
    sim_cfg = dict(raw.get("sim") or {})

    csv_path = args.csv
    if csv_path is None:
        rel = paths.get("cards_csv", "data/cards_stub.csv")
        pc = Path(rel).expanduser()
        csv_path = (pc if pc.is_absolute() else ROOT / pc).resolve()
    if not csv_path.is_file():
        print(f"CSV introuvable : {csv_path}", file=sys.stderr)
        return 1

    col_map = {str(k): str(v) for k, v in raw.get("card_csv", {}).items()}
    rules_p = ROOT / paths.get("rules_corpus_out", "data/rules_corpus.txt")
    rules_p = rules_p if rules_p.is_file() else None

    deck0 = args.deck.expanduser()
    deck1 = (args.deck1 or args.deck).expanduser()
    if not deck0.is_file() or not deck1.is_file():
        print("Deck introuvable.", file=sys.stderr)
        return 1

    obs_dim = int(train_cfg.get("obs_dim", 96))
    hidden = int(train_cfg.get("hidden", 256))
    n_res = int(train_cfg.get("n_res", 2))
    device = _pick_device(args.device)

    env_kwargs = dict(
        deck0_path=deck0,
        deck1_path=deck1,
        cards_csv=csv_path,
        column_map=col_map,
        rules_corpus_path=rules_p,
        sim_cfg=sim_cfg,
        obs_dim=obs_dim,
        seed=args.seed,
    )

    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.is_file():
        tried = ckpt_path.resolve()
        alt = (ROOT / ckpt_path).resolve() if not ckpt_path.is_absolute() else None
        print(
            f"Checkpoint introuvable : {tried}",
            file=sys.stderr,
        )
        if alt is not None and alt.is_file():
            print(f"  → fichier présent à la racine du dépôt : {alt}", file=sys.stderr)
        else:
            print(
                "  Lance d’abord un entraînement (train_ppo) ou passe un .pt existant, "
                f"ex. : {ROOT / 'checkpoints' / 'ablation_seed42.pt'}",
                file=sys.stderr,
            )
        return 1

    payload = torch_load_train_checkpoint(ckpt_path, device)
    n_act = None
    if isinstance(payload, dict) and "policy_state_dict" in payload:
        sd = payload["policy_state_dict"]
        if "pi_head.weight" in sd:
            n_act = int(sd["pi_head.weight"].shape[0])
    if n_act is None:
        from opctcg_text_sim.env import OPTextSimEnv

        _e = OPTextSimEnv(
            deck0, deck1, csv_path, col_map, rules_p, sim_cfg,
            obs_dim=obs_dim, seed=args.seed,
        )
        n_act = int(_e.action_space.n)
        _e.close()

    net = ActorCritic(obs_dim, n_act, hidden, n_res).to(device)
    _obs_clip = float(train_cfg.get("obs_normalize_clip", 10.0))
    obs_rms: RunningMeanStd | None = None
    if isinstance(payload, dict) and payload.get("obs_rms") is not None:
        obs_rms = RunningMeanStd((obs_dim,), clip_obs=_obs_clip)
    load_policy_checkpoint(
        ckpt_path, net, device, obs_rms=obs_rms, reward_rms=None, data=payload,
    )

    generate_video = _load_generate_video()
    mp4_prefix = Path(args.mp4).expanduser()
    if not mp4_prefix.is_absolute():
        mp4_prefix = (ROOT / mp4_prefix).resolve()
    mp4_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"Génération vidéo depuis {ckpt_path.resolve()}…", flush=True)
    out = generate_video(
        net,
        env_kwargs,
        mp4_prefix,
        int(args.video_idx),
        device,
        fps=float(args.fps),
        frame_size=tuple(args.frame) if args.frame else None,
        thumb_size=tuple(args.thumb) if args.thumb else None,
        obs_rms=obs_rms,
        verbose=True,
        cleanup_jsonl=True,
    )
    if out is None:
        return 1
    print(f"OK → {out.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
