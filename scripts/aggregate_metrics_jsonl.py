#!/usr/bin/env python3
"""
Agrège des fichiers metrics JSONL (ex. plusieurs graines) : médiane / min / max
sur la **dernière** ligne enregistrée de chaque fichier (état final de chaque run).

Usage :
  python scripts/aggregate_metrics_jsonl.py runs/metrics_s42.jsonl runs/metrics_s43.jsonl
  python scripts/aggregate_metrics_jsonl.py runs/metrics_*.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from glob import glob
from pathlib import Path
from typing import Any


def _last_row(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def main() -> int:
    p = argparse.ArgumentParser(description="Agrège des metrics.jsonl (dernière ligne par fichier)")
    p.add_argument(
        "paths",
        nargs="*",
        help="Fichiers .jsonl ou glob (shell expand ou passer des littéraux)",
    )
    args = p.parse_args()
    files: list[Path] = []
    for pat in args.paths:
        if any(c in pat for c in "*?["):
            files.extend(Path(p) for p in glob(pat))
        else:
            files.append(Path(pat))

    files = sorted({p.resolve() for p in files if str(p)})
    if not files:
        print("Aucun fichier.", file=sys.stderr)
        return 1

    keys_float = (
        "rew", "len", "win_pct", "loss", "ent", "clip", "kl", "lr", "stp_s",
        "steps", "update",
    )
    collected: dict[str, list[float]] = {k: [] for k in keys_float}
    meta: list[dict[str, Any]] = []

    for f in files:
        row = _last_row(f)
        if row is None:
            print(f"  (vide ou illisible : {f})", file=sys.stderr)
            continue
        meta.append({"file": str(f), "update": row.get("update"), "steps": row.get("steps")})
        for k in keys_float:
            if k in row and row[k] is not None:
                try:
                    collected[k].append(float(row[k]))
                except (TypeError, ValueError):
                    pass

    out: dict[str, Any] = {"n_files": len(meta), "runs": meta}
    for k, vals in collected.items():
        if not vals:
            continue
        out[k] = {
            "median": float(statistics.median(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "mean": float(statistics.mean(vals)),
        }

    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
