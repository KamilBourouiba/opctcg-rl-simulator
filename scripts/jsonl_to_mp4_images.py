#!/usr/bin/env python3
"""Replay JSONL + manifeste → vidéo MP4 (vignettes cartes, même rendu que le GIF images)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opctcg_text_sim.replay_render_images import jsonl_to_mp4_with_images


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("runs/replay_images.mp4"))
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Défaut : <jsonl stem>.manifest.json",
    )
    p.add_argument("--cache", type=Path, default=Path("runs/card_image_cache"))
    p.add_argument("--fps", type=float, default=1.5)
    p.add_argument(
        "--frame",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=None,
        help="Résolution de chaque image (ex. 2560 1440). Défaut : cadre large du module.",
    )
    p.add_argument(
        "--thumb",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=None,
        help="Taille des vignettes cartes (ex. 120 168).",
    )
    args = p.parse_args()
    if not args.jsonl.is_file():
        print("Fichier introuvable :", args.jsonl, file=sys.stderr)
        return 1
    man = args.manifest or args.jsonl.with_name(args.jsonl.stem + ".manifest.json")
    if not man.is_file():
        print("Manifeste introuvable :", man, file=sys.stderr)
        return 1
    try:
        jsonl_to_mp4_with_images(
            args.jsonl,
            args.out,
            man,
            args.cache,
            fps=args.fps,
            frame_size=tuple(args.frame) if args.frame else None,
            thumb_size=tuple(args.thumb) if args.thumb else None,
        )
    except ImportError as e:
        print(e, file=sys.stderr)
        return 1
    print("MP4 écrit :", args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
