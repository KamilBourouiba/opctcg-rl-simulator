#!/usr/bin/env python3
"""
Enchaîne plusieurs entraînements ``train_ppo.py`` avec des graines différentes
(statistiques de robustesse, comparaison de checkpoints).

Les arguments après ``--`` sont transmis tel quel à ``train_ppo.py``.

Exemple :
  python scripts/train_ppo_multi_seed.py --runs 3 --seed-base 40 --out-prefix checkpoints/ablation \\
      -- --config config.yaml --steps 50000 --no-final-video
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Multi-graines : appelle train_ppo.py plusieurs fois",
    )
    p.add_argument("--runs", type=int, default=3, help="Nombre d’entraînements séquentiels")
    p.add_argument("--seed-base", type=int, default=42, help="Première graine ; suivantes +1, +2, …")
    p.add_argument(
        "--out-prefix",
        type=Path,
        default=ROOT / "checkpoints" / "policy",
        help="Préfixe : fichiers policy_seed{S}.pt et résumé JSON",
    )
    p.add_argument(
        "train_ppo_args",
        nargs=argparse.REMAINDER,
        help="Préfixer avec ``--`` puis les options de train_ppo.py",
    )
    args = p.parse_args()
    rest = args.train_ppo_args
    if rest and rest[0] == "--":
        rest = rest[1:]

    if args.runs < 1:
        print("--runs doit être >= 1", file=sys.stderr)
        return 1

    exe = sys.executable
    script = ROOT / "scripts" / "train_ppo.py"
    summaries: list[dict] = []
    prefix = Path(args.out_prefix).expanduser()
    if not prefix.is_absolute():
        prefix = (ROOT / prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    for i in range(args.runs):
        seed = int(args.seed_base) + i
        out_pt = prefix.parent / f"{prefix.name}_seed{seed}.pt"
        cmd = [exe, str(script), "--seed", str(seed), "--out", str(out_pt), *rest]
        print(f"\n=== Run {i + 1}/{args.runs}  seed={seed} ===\n→ {' '.join(cmd)}\n", flush=True)
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"Échec seed={seed} (code {r.returncode})", file=sys.stderr)
            return int(r.returncode)
        summaries.append({"seed": seed, "checkpoint": str(out_pt.resolve())})

    agg = prefix.parent / f"{prefix.name}_multi_seed_summary.json"
    agg.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nRésumé multi-graines → {agg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
