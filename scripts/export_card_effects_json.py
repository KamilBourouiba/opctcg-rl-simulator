#!/usr/bin/env python3
"""
Exporte un unique JSON des effets parsés (vue simulateur : ParsedCard).

Usage :
  python scripts/export_card_effects_json.py
  python scripts/export_card_effects_json.py --csv data/cards_tcgcsv.csv --out data/card_effects_sim.json

Ensuite dans ``config.yaml`` → ``sim.card_effects_snapshot: "data/card_effects_sim.json"``
pour charger ce fichier au lieu de reparser tout le CSV au démarrage.

Le même schéma peut être stocké en colonne **jsonb** (PostgreSQL) côté infra :
un export fichier reste pratique pour le dev offline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    import yaml

    from opctcg_text_sim.card_db import load_card_csv
    from opctcg_text_sim.effect_snapshot import SNAPSHOT_VERSION, export_effect_snapshot_json

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "data" / "cards_tcgcsv.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "card_effects_sim.json")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    col = {str(k): str(v) for k, v in cfg.get("card_csv", {}).items()}

    cards = load_card_csv(args.csv, col)
    if not cards:
        print("Aucune carte chargée.", file=sys.stderr)
        return 1

    n = export_effect_snapshot_json(cards, args.out)
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(f"Export OK : {n} cartes (texte non vide) → {args.out}")
    print(f"  version snapshot : {SNAPSHOT_VERSION}")
    print(f"  taille fichier     : {size_mb:.2f} MiB")
    print(f'  Activer dans config.yaml : sim.card_effects_snapshot: "{args.out.relative_to(ROOT)}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
