#!/usr/bin/env python3
"""
Affiche la dernière ligne de metrics.jsonl (PPO) et un débit approximatif entre deux lignes.

Usage :
  python scripts/monitor_ppo_metrics.py
  python scripts/monitor_ppo_metrics.py runs/metrics.jsonl --watch 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _tail_last_lines(path: Path, n: int = 3) -> list[str]:
    if not path.is_file():
        return []
    data = path.read_bytes()
    if not data:
        return []
    text = data.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    lines = text.splitlines()
    return lines[-n:]


def main() -> int:
    ap = argparse.ArgumentParser(description="Surveille runs/metrics.jsonl (PPO)")
    ap.add_argument(
        "jsonl",
        type=Path,
        nargs="?",
        default=Path("runs/metrics.jsonl"),
        help="Fichier metrics JSONL (défaut : runs/metrics.jsonl depuis le cwd)",
    )
    ap.add_argument("--watch", type=float, default=0.0, help="Rafraîchir toutes les N secondes (0 = une fois)")
    args = ap.parse_args()
    p = args.jsonl.expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()

    def once() -> None:
        lines = _tail_last_lines(p, 4)
        if not lines:
            print(f"(vide ou absent) {p}", flush=True)
            return
        rows = []
        for ln in lines:
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        if not rows:
            print("(aucune ligne JSON valide)", flush=True)
            return
        last = rows[-1]
        stp = last.get("stp_s")
        parts = [
            f"steps={last.get('steps')}",
            f"upd={last.get('update')}",
            f"rew={last.get('rew'):+.2f}" if last.get("rew") is not None else "",
            f"win={last.get('win_pct'):.0f}%" if last.get("win_pct") is not None else "",
        ]
        if stp is not None:
            parts.append(f"stp/s≈{float(stp):.0f}")
        if len(rows) >= 2:
            a, b = rows[-2], rows[-1]
            ta, tb = a.get("t"), b.get("t")
            sa, sb = a.get("steps"), b.get("steps")
            if isinstance(ta, (int, float)) and isinstance(tb, (int, float)) and sa is not None and sb is not None:
                dt = float(tb) - float(ta)
                ds = int(sb) - int(sa)
                if dt > 0.01 and ds >= 0:
                    parts.append(f"Δstp/s≈{ds/dt:.0f}")
        print(f"{p.name}:  " + "  ".join(x for x in parts if x), flush=True)

    if args.watch <= 0:
        once()
        return 0
    print(f"Surveillance {p} (Ctrl+C pour quitter)\n", flush=True)
    try:
        while True:
            once()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
