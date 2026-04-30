#!/usr/bin/env python3
"""
Deux fois le deck « NAMI BY » + journal d’une partie (une ligne par action).

Usage :
  python scripts/run_two_nami_logged.py
  python scripts/run_two_nami_logged.py --steps 80 --log runs/action_log.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opctcg_text_sim.env import OPTextSimEnv


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--deck",
        type=Path,
        default=Path.home()
        / "Library/Application Support/com.Batsu.OPTCGSim/NAMI BY.txt",
        help="Deck utilisé pour les deux joueurs",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Base cartes CSV (défaut : paths.cards_csv dans config.yaml, sinon data/cards_stub.csv)",
    )
    p.add_argument(
        "--log",
        type=Path,
        default=ROOT / "runs" / "two_nami_actions.log",
    )
    p.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Replay structuré pour l’animation (défaut : même dossier que --log, .jsonl)",
    )
    p.add_argument(
        "--gif",
        type=Path,
        default=None,
        help="GIF matplotlib (texte) après la partie (ex. runs/replay.gif)",
    )
    p.add_argument(
        "--gif-images",
        type=Path,
        default=None,
        dest="gif_images",
        help="GIF avec vignettes cartes (URLs du CSV / tcgcsv) ; utilise le .manifest.json",
    )
    p.add_argument(
        "--image-cache",
        type=Path,
        default=ROOT / "runs" / "card_image_cache",
        help="Dossier cache téléchargements (mode --gif-images / --mp4)",
    )
    p.add_argument(
        "--mp4",
        type=Path,
        default=None,
        dest="mp4",
        help="Vidéo MP4 (H.264) avec le même rendu que --gif-images ; requiert imageio + imageio-ffmpeg",
    )
    p.add_argument("--fps", type=float, default=1.5, help="Images / seconde (GIF ou MP4)")
    p.add_argument(
        "--mp4-frame",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=None,
        help="Avec --mp4 : largeur et hauteur des frames (ex. 2560 1440). Défaut : cadre large.",
    )
    p.add_argument(
        "--mp4-thumb",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=None,
        help="Avec --mp4 : taille des vignettes cartes en pixels.",
    )
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.deck.is_file():
        print("Deck introuvable :", args.deck, file=sys.stderr)
        return 1

    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    if args.csv is None:
        rel = raw.get("paths", {}).get("cards_csv", "data/cards_stub.csv")
        pc = Path(rel).expanduser()
        args.csv = (pc if pc.is_absolute() else ROOT / pc).resolve()
        if not args.csv.is_file():
            args.csv = (ROOT / "data" / "cards_stub.csv").resolve()

    if not args.csv.is_file():
        print("CSV introuvable :", args.csv, file=sys.stderr)
        print("  python scripts/fetch_opcg_tcgcsv.py -o data/cards_tcgcsv.csv", file=sys.stderr)
        print("  ou python scripts/build_stub_cards_from_decks.py", args.deck, args.deck, "-o", args.csv, file=sys.stderr)
        return 1

    sim = raw.get("sim", {})
    train = raw.get("training", {})
    col = {str(k): str(v) for k, v in raw.get("card_csv", {}).items()}

    jsonl_path = args.jsonl
    if jsonl_path is None:
        jsonl_path = args.log.with_suffix(".jsonl")

    env = OPTextSimEnv(
        args.deck,
        args.deck,
        args.csv,
        col,
        None,
        sim,
        obs_dim=int(train.get("obs_dim", 96)),
        seed=args.seed,
        action_log_path=args.log,
        animation_log_path=jsonl_path,
    )
    env.reset(seed=args.seed)
    rng = __import__("numpy").random.default_rng(args.seed + 1)
    for t in range(args.steps):
        a = int(rng.integers(0, env.action_space.n))
        _, _, done, _, _ = env.step(a)
        if done:
            break
    env.close()
    print("Journal texte :", args.log.resolve())
    print("Journal animation (JSONL) :", jsonl_path.resolve())
    print(
        "Manifeste (images) :",
        jsonl_path.with_name(jsonl_path.stem + ".manifest.json").resolve(),
    )
    if args.gif is not None:
        from opctcg_text_sim.replay_render import jsonl_to_gif

        jsonl_to_gif(jsonl_path, args.gif, fps=args.fps)
        print("GIF animé (matplotlib) :", args.gif.resolve())
    if args.gif_images is not None:
        from opctcg_text_sim.replay_render_images import jsonl_to_gif_with_images

        man = jsonl_path.with_name(jsonl_path.stem + ".manifest.json")
        jsonl_to_gif_with_images(
            jsonl_path,
            args.gif_images,
            man,
            args.image_cache,
            fps=args.fps,
        )
        print("GIF avec images :", args.gif_images.resolve())
    if args.mp4 is not None:
        from opctcg_text_sim.replay_render_images import jsonl_to_mp4_with_images

        man = jsonl_path.with_name(jsonl_path.stem + ".manifest.json")
        jsonl_to_mp4_with_images(
            jsonl_path,
            args.mp4,
            man,
            args.image_cache,
            fps=args.fps,
            frame_size=tuple(args.mp4_frame) if args.mp4_frame else None,
            thumb_size=tuple(args.mp4_thumb) if args.mp4_thumb else None,
        )
        print("Vidéo MP4 :", args.mp4.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
