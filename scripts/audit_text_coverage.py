#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opctcg_text_sim.card_db import load_card_csv
from opctcg_text_sim.deck_parser import parse_deck_file


@dataclass(frozen=True)
class Rule:
    key: str
    label: str
    status: str
    regex: re.Pattern[str]


def load_csv_db(config_path: Path):
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = raw.get("paths") or {}
    csv_rel = paths.get("cards_csv", "data/cards_tcgcsv.csv")
    csv_path = Path(csv_rel)
    if not csv_path.is_absolute():
        csv_path = (ROOT / csv_path).resolve()
    col_map = {str(k): str(v) for k, v in (raw.get("card_csv") or {}).items()}
    db = load_card_csv(csv_path, col_map)
    return csv_path, db


def load_rules(path: Path) -> list[Rule]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: list[Rule] = []
    for item in raw.get("rules", []):
        status = str(item.get("status", "")).strip().lower()
        if status not in {"supported", "partial", "unsupported"}:
            raise ValueError(f"Invalid status for rule {item.get('key')}: {status}")
        out.append(
            Rule(
                key=str(item.get("key", "")).strip(),
                label=str(item.get("label", "")).strip(),
                status=status,
                regex=re.compile(str(item.get("regex", "")), re.IGNORECASE),
            )
        )
    return out


def iter_deck_ids(decks_dir: Path) -> dict[str, set[str]]:
    id_to_decks: dict[str, set[str]] = defaultdict(set)
    for p in sorted(decks_dir.rglob("*.txt")):
        if p.name.startswith("."):
            continue
        for cid, _n in parse_deck_file(p):
            id_to_decks[cid].add(str(p.relative_to(ROOT)))
    return id_to_decks


def norm_text(s: str) -> str:
    return (
        str(s or "")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .lower()
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit global de couverture des effets texte pour tous les decks."
    )
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--decks-dir", type=Path, default=ROOT / "decks")
    ap.add_argument(
        "--rules",
        type=Path,
        default=ROOT / "scripts" / "text_effect_coverage_rules.yaml",
    )
    ap.add_argument("--show-supported", action="store_true", help="Afficher aussi les cartes 100% supportées")
    args = ap.parse_args()

    if not args.decks_dir.is_dir():
        print(f"Dossier decks introuvable: {args.decks_dir}", file=sys.stderr)
        return 1
    if not args.rules.is_file():
        print(f"Fichier de règles introuvable: {args.rules}", file=sys.stderr)
        return 1

    csv_path, db = load_csv_db(args.config)
    if not db:
        print(f"CSV vide/illisible: {csv_path}", file=sys.stderr)
        return 1
    rules = load_rules(args.rules)
    if not rules:
        print("Aucune règle chargée.", file=sys.stderr)
        return 1

    id_to_decks = iter_deck_ids(args.decks_dir)
    ids = sorted(id_to_decks)

    status_per_card: dict[str, str] = {}
    matches_per_card: dict[str, list[Rule]] = {}
    missing_csv: list[str] = []
    global_counter = Counter()

    for cid in ids:
        cd = db.get(cid)
        if not cd:
            missing_csv.append(cid)
            continue
        text = norm_text(cd.card_text)
        matches = [r for r in rules if r.regex.search(text)]
        matches_per_card[cid] = matches
        for m in matches:
            global_counter[m.status] += 1

        if any(m.status == "unsupported" for m in matches):
            status_per_card[cid] = "unsupported"
        elif any(m.status == "partial" for m in matches):
            status_per_card[cid] = "partial"
        else:
            status_per_card[cid] = "supported"

    counts = Counter(status_per_card.values())
    total_known = len(status_per_card)
    print(f"Decks : {args.decks_dir}")
    print(f"CSV   : {csv_path}")
    print(f"Rules : {args.rules}")
    print(f"Cartes uniques deck : {len(ids)}")
    print(f"Cartes trouvées CSV : {total_known}")
    print(f"Cartes absentes CSV : {len(missing_csv)}")
    print("")
    print("### Couverture (par carte)")
    print(f"- supported  : {counts.get('supported', 0)}")
    print(f"- partial    : {counts.get('partial', 0)}")
    print(f"- unsupported: {counts.get('unsupported', 0)}")
    print("")

    print("### Détails cartes à revoir")
    for cid in ids:
        if cid not in status_per_card:
            continue
        st = status_per_card[cid]
        if st == "supported" and not args.show_supported:
            continue
        cd = db[cid]
        mats = matches_per_card[cid]
        tags = ", ".join(sorted({f"{m.key}:{m.status}" for m in mats})) or "no_rule_match"
        loc = f"{len(id_to_decks[cid])} decks"
        print(f"- {cid} | {cd.name[:48]} | {st} | {tags} | {loc}")

    if missing_csv:
        print("")
        print("### IDs absents du CSV")
        for cid in missing_csv:
            print(f"- {cid} ({len(id_to_decks[cid])} decks)")

    print("")
    print("### Répartition des matches de règles")
    print(
        f"- matched supported tags  : {global_counter.get('supported', 0)}\n"
        f"- matched partial tags    : {global_counter.get('partial', 0)}\n"
        f"- matched unsupported tags: {global_counter.get('unsupported', 0)}"
    )

    # 2 = présence de cartes "unsupported" (utile en CI)
    if counts.get("unsupported", 0) > 0:
        return 2
    if missing_csv:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
