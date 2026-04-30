#!/usr/bin/env python3
"""Génère data/cards_stub.csv à partir de deux fichiers deck (NxID)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opctcg_text_sim.deck_parser import deck_to_multiset, parse_deck_file


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("deck_a", type=Path)
    p.add_argument("deck_b", type=Path)
    p.add_argument("-o", type=Path, default=ROOT / "data" / "cards_stub.csv")
    args = p.parse_args()
    ids = set(deck_to_multiset(parse_deck_file(args.deck_a))) | set(
        deck_to_multiset(parse_deck_file(args.deck_b))
    )
    rows = []
    for cid in sorted(ids):
        h = abs(hash(cid))
        rows.append(
            {
                "card_id": cid,
                "name": f"Card_{cid}",
                "cost": 1 + (h % 5),
                "power": 3000 + (h % 8) * 500,
                "counter": (h % 3) * 1000,
                "color": ["R", "G", "B", "Y", "P"][h % 5],
                "image_url": "",
                "card_text": "",
            }
        )
    args.o.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.o, index=False)
    print("Écrit :", args.o, "—", len(rows), "cartes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
