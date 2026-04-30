#!/usr/bin/env python3
"""
Parcourt un dossier de decklists (.txt), valide chaque deck, simule des matchs
aléatoires (mirror autorisé) et écrit un JSON de session (stats / extraits de log).

Avant la première utilisation, importer les decks Egman (voir ``scripts/fetch_egman_tournament_decks.py``).

Usage :
  python scripts/deck_gauntlet_web.py \\
    --decks-dir decks/tournament_op15 \\
    --out-json runs/session.json \\
    --games 30 --steps-per-match 120 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opctcg_text_sim.card_db import ensure_cards_for_deck, load_card_csv
from opctcg_text_sim.deck_parser import deck_to_multiset, parse_deck_file
from opctcg_text_sim.deck_validate import validate_deck_file
from opctcg_text_sim.env import OPTextSimEnv


def _list_decks(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.txt") if p.is_file() and p.name.upper() != "README.TXT")


def main() -> int:
    ap = argparse.ArgumentParser(description="Gauntlet decks → JSON de session")
    ap.add_argument("--decks-dir", type=Path, default=ROOT / "decks" / "tournament_op15")
    ap.add_argument("--out-json", type=Path, default=ROOT / "runs" / "session.json")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--games", type=int, default=24, help="Nombre de matchs aléatoires")
    ap.add_argument("--steps-per-match", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--log-excerpt-lines",
        type=int,
        default=100,
        help="Nombre de lignes de log d’action conservées par match (replay dans le JSON)",
    )
    args = ap.parse_args()

    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    if args.csv is None:
        rel = raw.get("paths", {}).get("cards_csv", "data/cards_tcgcsv.csv")
        pc = Path(rel).expanduser()
        args.csv = (pc if pc.is_absolute() else ROOT / pc).resolve()
    if not args.csv.is_file():
        print("CSV introuvable :", args.csv, file=sys.stderr)
        return 1

    col = {str(k): str(v) for k, v in raw.get("card_csv", {}).items()}
    sim_cfg = dict(raw.get("sim", {}))
    train_cfg = raw.get("training", {})
    obs_dim = int(train_cfg.get("obs_dim", 96))

    db = load_card_csv(args.csv, col)
    deck_paths = _list_decks(args.decks_dir.resolve())
    if len(deck_paths) < 1:
        print("Aucun .txt dans", args.decks_dir, file=sys.stderr)
        return 1

    # Précharge toutes les cartes
    all_ids: set[str] = set()
    for p in deck_paths:
        all_ids |= set(deck_to_multiset(parse_deck_file(p)))
    cards = ensure_cards_for_deck(all_ids, db)

    deck_entries: list[dict] = []
    valid_paths: list[Path] = []
    for p in deck_paths:
        vr = validate_deck_file(p, cards)
        lid = vr.leader_id
        leader_name: str | None = None
        leader_image_url: str | None = None
        if lid and lid in cards:
            lc = cards[lid]
            leader_name = lc.name or None
            leader_image_url = lc.image_url or None
        deck_entries.append(
            {
                "file": p.name,
                "path": str(p.relative_to(ROOT)) if ROOT in p.parents else p.name,
                "valid": vr.ok,
                "errors": vr.errors,
                "leader_id": lid,
                "leader_name": leader_name,
                "leader_image_url": leader_image_url,
                "main_size": vr.main_deck_size,
            }
        )
        if vr.ok:
            valid_paths.append(p)

    if len(valid_paths) < 1:
        print("Aucun deck valide.", file=sys.stderr)
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "decks": deck_entries,
                    "error": "no_valid_decks",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return 1

    rng_py = random.Random(args.seed)
    rng_np = np.random.default_rng(args.seed)

    wins: dict[str, int] = {p.name: 0 for p in valid_paths}
    games_ct: dict[str, int] = {p.name: 0 for p in valid_paths}
    h2h: dict[str, dict[str, int]] = {}

    last_match: dict | None = None
    matches: list[dict] = []

    for g in range(args.games):
        a, b = rng_py.sample(valid_paths, 2) if len(valid_paths) >= 2 else (valid_paths[0], valid_paths[0])
        key = f"{a.name} vs {b.name}"
        h2h.setdefault(key, {"games": 0, "wins_a": 0, "wins_b": 0, "draws": 0})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as tf:
            log_path = Path(tf.name)
        try:
            env = OPTextSimEnv(
                a,
                b,
                args.csv,
                col,
                None,
                sim_cfg,
                obs_dim=obs_dim,
                seed=int(rng_np.integers(0, 2**31 - 1)),
                action_log_path=log_path,
            )
            obs, _ = env.reset(seed=int(rng_np.integers(0, 2**31 - 1)))
            winner = None
            steps = 0
            for _ in range(args.steps_per_match):
                mask = env.legal_actions_mask()
                idx = np.flatnonzero(mask)
                if len(idx) == 0:
                    break
                act = int(rng_py.choice(idx))
                obs, _r, term, trunc, info = env.step(act)
                steps += 1
                if term or trunc:
                    winner = info.get("winner")
                    break
            env.close()
            excerpt = ""
            if log_path.is_file():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                ntail = max(20, int(args.log_excerpt_lines))
                excerpt = "\n".join(lines[-ntail:])
        finally:
            if log_path.is_file():
                log_path.unlink(missing_ok=True)

        games_ct[a.name] += 1
        games_ct[b.name] += 1
        h2h[key]["games"] += 1
        if winner == 0:
            wins[a.name] += 1
            h2h[key]["wins_a"] += 1
        elif winner == 1:
            wins[b.name] += 1
            h2h[key]["wins_b"] += 1
        else:
            h2h[key]["draws"] += 1

        last_match = {
            "deck0": a.name,
            "deck1": b.name,
            "steps": steps,
            "winner": "p0" if winner == 0 else ("p1" if winner == 1 else None),
            "log_excerpt": excerpt,
        }
        matches.append(
            {
                "deck0": a.name,
                "deck1": b.name,
                "steps": steps,
                "winner": last_match["winner"],
                "log_excerpt": excerpt,
            }
        )

    deck_stats = {}
    for p in valid_paths:
        n = games_ct[p.name]
        deck_stats[p.name] = {
            "games": n,
            "wins": wins[p.name],
            "losses": n - wins[p.name],
            "win_rate": round(wins[p.name] / n, 4) if n else 0.0,
        }

    valid_n = len(valid_paths)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decks_dir": str(args.decks_dir.resolve().relative_to(ROOT)) if ROOT in args.decks_dir.resolve().parents else str(args.decks_dir),
        "games_run": args.games,
        "steps_per_match": args.steps_per_match,
        "seed": args.seed,
        "decks_listed": len(deck_entries),
        "decks_valid": valid_n,
        "log_excerpt_lines": int(args.log_excerpt_lines),
        "decks": deck_entries,
        "deck_stats": deck_stats,
        "head_to_head": h2h,
        "matches": matches,
        "last_match": last_match,
        "note": "Les matchs utilisent des actions légales aléatoires (pas d’agent entraîné).",
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Écrit :", args.out_json.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
