#!/usr/bin/env python3
"""Replay JSONL + manifeste cartes → GIF animé avec vignettes (URLs tcgcsv / TCGplayer)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opctcg_text_sim.replay_render_images import jsonl_to_gif_with_images


def main() -> int:
    p = argparse.ArgumentParser(
        description="Lit un .jsonl de replay et un .manifest.json (noms + image_url), "
        "télécharge les images dans --cache, écrit un GIF."
    )
    p.add_argument("jsonl", type=Path)
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("runs/replay_images.gif"),
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Défaut : même nom que le jsonl avec extension .manifest.json",
    )
    p.add_argument(
        "--cache",
        type=Path,
        default=Path("runs/card_image_cache"),
        help="Répertoire de cache des images téléchargées",
    )
    p.add_argument("--fps", type=float, default=1.5)
    p.add_argument(
        "--frame",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=None,
        help="Résolution de chaque image (ex. 2560 1440).",
    )
    p.add_argument(
        "--thumb",
        nargs=2,
        type=int,
        metavar=("W", "H"),
        default=None,
        help="Taille des vignettes cartes.",
    )
    args = p.parse_args()
    if not args.jsonl.is_file():
        print("Fichier introuvable :", args.jsonl, file=sys.stderr)
        print(
            "Astuce : avec run_two_nami_logged.py --log runs/actions.txt, "
            "le replay est runs/actions.jsonl (même nom que le .log, extension .jsonl).",
            file=sys.stderr,
        )
        return 1
    man = args.manifest
    if man is None:
        man = args.jsonl.with_name(args.jsonl.stem + ".manifest.json")
    if not man.is_file():
        print(
            "Manifeste introuvable :",
            man,
            "(lancez une partie avec animation_log_path ou passez --manifest)",
            file=sys.stderr,
        )
        return 1
    jsonl_to_gif_with_images(
        args.jsonl,
        args.out,
        man,
        args.cache,
        fps=args.fps,
        frame_size=tuple(args.frame) if args.frame else None,
        thumb_size=tuple(args.thumb) if args.thumb else None,
    )
    print("GIF écrit :", args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
