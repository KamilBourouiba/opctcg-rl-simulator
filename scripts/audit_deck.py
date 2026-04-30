#!/usr/bin/env python3
"""
Parcourt un fichier deck (.txt) : chaque ID de carte est résolu dans le CSV,
avec un résumé utile pour le simulateur ([Activate: Main], timings non simulés, etc.).

Usage :
  python scripts/audit_deck.py path/to/deck.txt
  python scripts/audit_deck.py --config config.yaml decks/tournament_op15/p0016_MerryTCG.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit carte par carte d’un deck vs le CSV.")
    ap.add_argument("deck", type=Path, help="Fichier deck .txt")
    ap.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config.yaml",
        help="YAML pour card_csv et chemin CSV (défaut : config.yaml)",
    )
    args = ap.parse_args()

    import yaml

    from opctcg_text_sim.card_db import load_card_csv
    from opctcg_text_sim.deck_parser import deck_to_multiset, parse_deck_file

    deck_path = args.deck.expanduser()
    if not deck_path.is_file():
        print(f"Fichier introuvable : {deck_path}", file=sys.stderr)
        return 1

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths = raw.get("paths") or {}
    csv_rel = paths.get("cards_csv", "data/cards_tcgcsv.csv")
    csv_path = Path(csv_rel)
    if not csv_path.is_absolute():
        csv_path = (ROOT / csv_path).resolve()
    col_map = {str(k): str(v) for k, v in raw.get("card_csv", {}).items()}

    entries = parse_deck_file(deck_path)
    m = deck_to_multiset(entries)
    db = load_card_csv(csv_path, col_map)
    if not db:
        print(f"CSV vide ou illisible : {csv_path}", file=sys.stderr)
        return 1

    print(f"Deck : {deck_path.name}\nCSV  : {csv_path}\n")

    # Ordre stable : d’abord les lignes du fichier (une entrée par ligne utile)
    seen: set[str] = set()
    rows: list[tuple[str, int]] = []
    for cid, n in m.items():
        rows.append((cid, n))

    leader_id: str | None = None
    for cid, _n in rows:
        cd = db.get(cid)
        if cd and "leader" in (cd.card_type or "").lower():
            leader_id = cid
            break

    for cid, count in sorted(rows, key=lambda x: (x[0] != leader_id, x[0])):
        cd = db.get(cid)
        if not cd:
            print(f"  {count}× {cid}  — ABSENT DU CSV")
            continue
        raw_txt = cd.card_text or ""
        low = raw_txt.replace("\u2019", "'").lower()
        flags: list[str] = []
        if "[activate: main]" in low:
            flags.append("[Activate: Main]")
        if "[opponent's turn]" in low:
            flags.append("[Opponent's Turn] (timing souvent partiel / absent du sim)")
        if "[trigger]" in low:
            flags.append("[Trigger]")
        if not flags:
            flags.append("(pas de marqueur Activate:Main / Opp.Turn détecté)")
        extra = ""
        if "leader" in (cd.card_type or "").lower():
            extra = f"  | vie={cd.life}  don_deck_size={cd.don_deck_size}"
            if cid == "OP05-098":
                extra += (
                    "\n     → Pas d’[Activate: Main] ; effet « vie → 0 » en tour adverse : "
                    "implémenté dans le sim (dégât vie, pas un bouton Main)."
                )
            if cid == "OP15-058":
                extra += "\n     → [Activate: Main] géré ; 2e tour+ requis ; DON deck 6 (CSV)."
        txt_preview = (cd.card_text or "").replace("\n", " ")[:140]
        print(f"  {count}× {cid}  {cd.name}")
        print(f"      type={cd.card_type!r}  color={cd.color!r}{extra}")
        print(f"      { ' ; '.join(flags) }")
        if txt_preview:
            print(f"      « {txt_preview}… »")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
