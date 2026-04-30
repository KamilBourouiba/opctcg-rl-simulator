#!/usr/bin/env python3
"""
Extrait les lignes de métriques PPO depuis un fichier log (stdout/nohup).

Les lignes ressemblent à :
  [  4096 / 5000000] upd    1/1220  rew=-4.80  len=38  win=12%  ep=72  loss=3.4965  ent=1.332  clip=0.00  lr=6.94e-06

Usage :
  python scripts/parse_training_log.py runs/train_5e9.log
  python scripts/parse_training_log.py runs/train_5e9.log --last 20
  python scripts/parse_training_log.py runs/train_5e9.log --jsonl > metrics.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Ligne PPO (avec ou sans total steps / total updates après « / »)
LINE_RE = re.compile(
    r"\[\s*(\d+)\s*(?:/\s*(\d+))?\]\s+"
    r"upd\s+(\d+)\s*(?:/\s*(\d+))?\s+"
    r"rew=([+-]?\d+(?:\.\d+)?)\s+"
    r"len=(\d+)\s+"
    r"win=(\d+)%\s+"
    r"ep=(\d+)\s+"
    r"loss=([0-9.eE+-]+)\s+"
    r"ent=([0-9.eE+-]+)\s+"
    r"clip=([0-9.eE+-]+)\s+"
    r"lr=([0-9.eE+-]+)",
)


def parse_file(path: Path) -> list[dict]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for m in LINE_RE.finditer(raw):
        steps_total = m.group(2)
        upd_total = m.group(4)
        rows.append(
            {
                "steps": int(m.group(1)),
                "steps_total": int(steps_total) if steps_total else None,
                "update": int(m.group(3)),
                "updates_total": int(upd_total) if upd_total else None,
                "rew": float(m.group(5)),
                "len": int(m.group(6)),
                "win_pct": int(m.group(7)),
                "episodes": int(m.group(8)),
                "loss": float(m.group(9)),
                "ent": float(m.group(10)),
                "clip": float(m.group(11)),
                "lr": float(m.group(12)),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Extrait les métriques PPO d'un fichier .log")
    ap.add_argument("log_file", type=Path, help="Fichier log (ex. runs/train_5e9.log)")
    ap.add_argument("--last", type=int, default=0, help="N'afficher que les N dernières lignes (0 = toutes)")
    ap.add_argument("--jsonl", action="store_true", help="Sortie JSONL au lieu du tableau texte")
    args = ap.parse_args()

    if not args.log_file.is_file():
        print(f"Fichier introuvable : {args.log_file}", file=sys.stderr)
        return 1

    rows = parse_file(args.log_file)
    if args.last > 0:
        rows = rows[-args.last :]

    if args.jsonl:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
        return 0

    if not rows:
        print("Aucune ligne de métriques PPO reconnue (format [ steps ] upd …).", file=sys.stderr)
        return 1

    print(f"{'steps':>12} {'upd':>6} {'rew':>8} {'win%':>5} {'loss':>8} {'lr':>10}")
    print("-" * 56)
    for r in rows:
        print(
            f"{r['steps']:>12} {r['update']:>6} {r['rew']:>8.2f} {r['win_pct']:>5d} "
            f"{r['loss']:>8.4f} {r['lr']:>10.2e}"
        )
    print(f"\n{len(rows)} point(s) extrait(s) depuis {args.log_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
