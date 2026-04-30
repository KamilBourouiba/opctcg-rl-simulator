#!/usr/bin/env python3
"""Convertit un replay JSONL en GIF animé."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opctcg_text_sim.replay_render import jsonl_to_gif


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("runs/replay.gif"))
    p.add_argument("--fps", type=float, default=1.5)
    args = p.parse_args()
    if not args.jsonl.is_file():
        print("Fichier introuvable :", args.jsonl, file=sys.stderr)
        return 1
    jsonl_to_gif(args.jsonl, args.out, fps=args.fps)
    print("GIF écrit :", args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
