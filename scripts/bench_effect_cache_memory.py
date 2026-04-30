#!/usr/bin/env python3
"""
Benchmark mémoire / dédup des effets parsés (vs cache classique).

Compare :
  - ``CardEffectCache()`` : une entrée ParsedCard complète par carte ;
  - ``CardEffectCache(dedupe_identical_rule_text=True)`` : mêmes listes
    ``effects`` / ``keywords`` / ``timing_segments`` partagées quand l’empreinte
    ``rule_text_fingerprint`` coïncide.

Simule une boucle type « train » : plusieurs ``precompute`` + accès ``get`` répétés
pour mesurer RSS (psutil si dispo, sinon ``resource``).

Usage :
  python scripts/bench_effect_cache_memory.py
  python scripts/bench_effect_cache_memory.py --csv data/cards_tcgcsv.csv --rounds 5
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _rss_bytes() -> int:
    try:
        import psutil  # type: ignore[import-untyped]

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _load_cards(csv_path: Path, column_map: dict) -> dict:
    from opctcg_text_sim.card_db import load_card_csv

    return load_card_csv(csv_path, column_map=column_map)


def _train_loop(
    cards: dict,
    *,
    dedupe: bool,
    rounds: int,
    accesses_per_round: int,
) -> tuple[object, int, int]:
    """Instancie un cache, précalcule ``rounds`` fois (GC entre les runs), puis get aléatoire."""
    from opctcg_text_sim.engine.effect_resolver import CardEffectCache

    ids = [cid for cid, c in cards.items() if (c.card_text or "").strip()]
    if not ids:
        raise SystemExit("Aucune carte avec card_text dans le CSV.")

    cache = CardEffectCache(dedupe_identical_rule_text=dedupe)
    for _ in range(rounds):
        cache.precompute(cards)
    # accès répétés (comme résolution d’effets en partie)
    for i in range(accesses_per_round):
        cid = ids[i % len(ids)]
        cache.get(cards[cid])
    n_cache = len(cache._cache)  # type: ignore[attr-defined]
    n_shared = cache.shared_fingerprint_entries
    return cache, n_cache, n_shared


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark cache effets + empreintes uniques.")
    ap.add_argument("--csv", type=Path, default=ROOT / "data" / "cards_tcgcsv.csv")
    ap.add_argument("--rounds", type=int, default=4, help="Passes precompute par scénario")
    ap.add_argument("--accesses", type=int, default=8000, help="get() après precompute")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    col = {str(k): str(v) for k, v in cfg.get("card_csv", {}).items()}

    cards = _load_cards(args.csv, col)
    if not cards:
        raise SystemExit(f"CSV vide ou introuvable : {args.csv}")

    from opctcg_text_sim.effect_unique import group_card_ids_by_fingerprint, unique_fingerprint_stats

    n_text, n_fp, ratio = unique_fingerprint_stats(cards)
    groups = group_card_ids_by_fingerprint(cards)
    multi = sum(1 for g in groups.values() if len(g) > 1)

    print("=== Empreintes (texte de règle + type) ===")
    print(f"  Cartes avec texte     : {n_text}")
    print(f"  Empreintes uniques    : {n_fp}")
    print(f"  Part théorique dédup  : {ratio * 100:.1f}%")
    print(f"  Empreintes partagées (≥2 cartes) : {multi}")

    scenarios = [
        ("cache standard (sans dédup ParsedCard interne)", False),
        ("cache + dédup listes parsées identiques", True),
    ]

    for label, dedupe in scenarios:
        gc.collect()
        t0 = time.perf_counter()
        rss0 = _rss_bytes()
        tracemalloc.start()
        _, n_ent, n_sh = _train_loop(
            cards,
            dedupe=dedupe,
            rounds=args.rounds,
            accesses_per_round=args.accesses,
        )
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss1 = _rss_bytes()
        dt = time.perf_counter() - t0
        print(f"\n=== {label} ===")
        print(f"  temps        : {dt:.2f}s")
        print(
            f"  entrées cache: {n_ent}  |  templates d’effet (empreintes distinctes): {n_sh}"
        )
        print(f"  tracemalloc  : current={cur / 1e6:.2f} MB  peak={peak / 1e6:.2f} MB")
        print(f"  RSS (approx) : delta {(rss1 - rss0) / 1e6:.2f} MB (fiable surtout avec psutil)")

    print(
        "\nNote : un fichier Python par effet de carte ne scale pas (milliers de cartes,"
        "\n      révisions TCG) ; l’empreinte + cache partagé donne le gain mémoire"
        "\n      sans perdre le moteur texte. Pour production, ``dedupe_identical_rule_text``"
        "\n      reste désactivé par défaut (mutations futures sur ``effects`` interdites)."
    )


if __name__ == "__main__":
    main()
