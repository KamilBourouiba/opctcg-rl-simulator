#!/usr/bin/env python3
"""
Télécharge toutes les cartes One Piece Card Game depuis tcgcsv.com (JSON TCGplayer),
puis écrit un CSV compatible avec ``load_card_csv`` (id, stats, ``imageUrl``, texte d’effet).

  python scripts/fetch_opcg_tcgcsv.py -o data/cards_tcgcsv.csv
  python scripts/fetch_opcg_tcgcsv.py --groups 23589,24241 -o data/op09_op11.csv
  python scripts/fetch_opcg_tcgcsv.py --list-groups
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opctcg_text_sim.tcgcsv_fetch import (
    DEFAULT_OP_CATEGORY_ID,
    fetch_all_card_rows,
    find_one_piece_category_id,
    list_groups,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=ROOT / "data" / "cards_tcgcsv.csv",
        help="CSV fusionné (défaut : data/cards_tcgcsv.csv)",
    )
    p.add_argument(
        "--category-id",
        type=int,
        default=None,
        help=f"ID catégorie TCGplayer (défaut : détection « One Piece » ou {DEFAULT_OP_CATEGORY_ID})",
    )
    p.add_argument(
        "--groups",
        type=str,
        default="",
        help="Liste d’IDs de groupes séparés par des virgules ; vide = tous les sets OP",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.08,
        help="Pause entre requêtes (secondes)",
    )
    p.add_argument(
        "--list-groups",
        action="store_true",
        help="Affiche les groupId / noms puis quitte",
    )
    args = p.parse_args()

    cat = args.category_id
    if cat is None:
        cat = find_one_piece_category_id()

    groups_meta = list_groups(cat)
    if args.list_groups:
        for g in sorted(groups_meta, key=lambda x: str(x.get("name") or "")):
            print(g.get("groupId"), g.get("abbreviation", ""), g.get("name", ""))
        print(f"# category_id={cat} — {len(groups_meta)} groupes")
        return 0

    gids = None
    if args.groups.strip():
        gids = [int(x.strip()) for x in args.groups.split(",") if x.strip()]

    def _on(gid: int, gname: str, total: int) -> None:
        print(f"  [{gid}] {gname[:56]}… ({total} groupes)", flush=True)

    print(f"Catégorie {cat} — récupération des produits…")
    merged = fetch_all_card_rows(cat, group_ids=gids, sleep_s=args.sleep, on_group=_on)
    if not merged:
        print("Aucune carte (vérifiez --groups ou la connexion).", file=sys.stderr)
        return 1

    rows = list(merged.values())
    df = pd.DataFrame(rows)
    cols = [
        "card_id",
        "name",
        "cost",
        "power",
        "counter",
        "color",
        "image_url",
        "card_text",
        "rarity",
        "card_type",
        "life",
        "product_id",
        "group_name",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols].sort_values("card_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Écrit {len(df)} cartes uniques → {args.output.resolve()}")
    print("Utilisez ce fichier avec :  --csv", args.output)
    print("Ou définissez paths.cards_csv dans config.yaml sur ce chemin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
