#!/usr/bin/env python3
"""
validate_cards.py — Boucle de validation interactive des effets de cartes.

Workflow :
  1. Charge les cartes depuis le CSV (cards_tcgcsv.csv par défaut)
  2. Tire une carte au hasard
  3. Passe la carte dans le pipeline : KeywordModel → TimingModel → EffectClassifier
  4. Affiche les attributs et les effets parsés
  5. L'utilisateur tape Entrée (OK) ou décrit ce qui est faux
  6. Sauvegarde la validation dans data/validated_effects.jsonl

Usage :
  python scripts/validate_cards.py
  python scripts/validate_cards.py --csv data/cards_stub.csv --out data/my_validated.jsonl
  python scripts/validate_cards.py --filter-type Character --filter-color Red
  python scripts/validate_cards.py --card OP01-001   # forcer une carte précise
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# ── Chemin racine du projet ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import pandas as pd

from opctcg_text_sim.models import (
    EffectClassifier,
    KeywordModel,
    ParsedCard,
    TimingModel,
)

# ── Couleurs ANSI ──────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RED    = "\033[91m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_WHITE  = "\033[97m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{_RESET}"


def _sep(char: str = "─", width: int = 60) -> str:
    return char * width


# ── Chargement du CSV ──────────────────────────────────────────────────────

def load_cards(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    df = df.fillna("")
    return df


def df_row_to_parsed_card(row: pd.Series) -> ParsedCard:
    def _int(val: str, default: int = 0) -> int:
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

    return ParsedCard(
        card_id   = row.get("card_id", ""),
        name      = row.get("name", ""),
        cost      = _int(row.get("cost", "0")),
        power     = _int(row.get("power", "0")),
        counter   = _int(row.get("counter", "0")),
        color     = row.get("color", ""),
        rarity    = row.get("rarity", ""),
        card_type = row.get("card_type", ""),
        life      = _int(row.get("life", "0")),
        card_text = row.get("card_text", ""),
        group_name= row.get("group_name", ""),
    )


# ── Pipeline ───────────────────────────────────────────────────────────────

_KEYWORD_MODEL   = KeywordModel()
_TIMING_MODEL    = TimingModel()
_EFFECT_CLASSIFIER = EffectClassifier()


def run_pipeline(card: ParsedCard) -> ParsedCard:
    _KEYWORD_MODEL.parse(card)
    _TIMING_MODEL.parse(card)
    _EFFECT_CLASSIFIER.parse(card)
    return card


# ── Affichage ──────────────────────────────────────────────────────────────

_COLOR_MAP = {
    "Red": "\033[91m", "Blue": "\033[94m", "Green": "\033[92m",
    "Yellow": "\033[93m", "Purple": "\033[95m", "Black": "\033[90m",
}


def _color_badge(color_str: str) -> str:
    parts = [c.strip() for c in color_str.split(";")]
    colored = []
    for c in parts:
        ansi = _COLOR_MAP.get(c, _WHITE)
        colored.append(f"{ansi}●{_RESET} {c}")
    return "  ".join(colored)


def display_card(card: ParsedCard) -> None:
    w = 62
    print()
    print(_c(_BOLD + _CYAN, _sep("═", w)))

    # Nom + ID
    id_str  = _c(_DIM, f"[{card.card_id}]")
    name_str = _c(_BOLD + _WHITE, card.name)
    print(f"  {name_str}  {id_str}")
    print(_c(_DIM, f"  {card.group_name}"))
    print(_c(_CYAN, _sep("─", w)))

    # Attributs
    type_line  = f"  Type     : {_c(_BOLD, card.card_type)}"
    color_line = f"  Couleur  : {_color_badge(card.color)}"
    if card.card_type == "Leader":
        stat_line = (
            f"  Vie      : {_c(_BOLD, str(card.life))}   "
            f"Puissance : {_c(_BOLD, str(card.power))}"
        )
    else:
        counter_str = str(card.counter) if card.counter > 0 else "—"
        stat_line = (
            f"  Coût     : {_c(_BOLD, str(card.cost))}   "
            f"Puissance : {_c(_BOLD, str(card.power))}   "
            f"Counter : {_c(_BOLD, counter_str)}"
        )
    rarity_line = f"  Rareté   : {_c(_BOLD, card.rarity)}"

    print(type_line)
    print(color_line)
    print(stat_line)
    print(rarity_line)

    # Texte brut
    print(_c(_CYAN, _sep("─", w)))
    print(_c(_DIM, "  TEXTE ORIGINAL :"))
    if card.card_text.strip():
        for line in card.card_text.strip().splitlines():
            print(f"    {line}")
    else:
        print(_c(_DIM, "    (aucun texte)"))

    # ── Modèle 1 : Keywords ────────────────────────────────────────────
    print(_c(_CYAN, _sep("─", w)))
    print(_c(_BOLD + _YELLOW, "  ▸ MODÈLE 1 — KeywordModel"))
    if card.keywords:
        kw_str = "  ".join(_c(_GREEN, f"[{k}]") for k in card.keywords)
        print(f"    Mots-clés : {kw_str}")
    else:
        print(_c(_DIM, "    Aucun mot-clé détecté."))

    # ── Modèle 2 : Timing segments ─────────────────────────────────────
    print(_c(_CYAN, _sep("─", w)))
    print(_c(_BOLD + _YELLOW, "  ▸ MODÈLE 2 — TimingModel"))
    if card.timing_segments:
        for timing, clause in card.timing_segments.items():
            print(f"    {_c(_BOLD, '[' + timing + ']')}")
            for sub in clause.split(" | "):
                print(f"      → {sub}")
    else:
        print(_c(_DIM, "    Aucun segment temporel détecté."))

    # ── Modèle 3 : Effects ─────────────────────────────────────────────
    print(_c(_CYAN, _sep("─", w)))
    print(_c(_BOLD + _YELLOW, "  ▸ MODÈLE 3 — EffectClassifier"))
    classifier_effects = [
        e for e in card.effects
        if not e.effect_type.value.startswith("keyword_")
        and e.effect_type.value != "condition"
    ]
    if classifier_effects:
        for eff in classifier_effects:
            effect_color = _GREEN if eff.confidence >= 0.9 else (
                _YELLOW if eff.confidence >= 0.7 else _RED
            )
            conf_str = ""
            if eff.confidence < 1.0:
                stars = "★" * round(eff.confidence * 3) + "☆" * (3 - round(eff.confidence * 3))
                conf_str = _c(_DIM, f" {stars} {eff.confidence:.0%}")

            # Ligne principale : timing + type + params
            params_parts = [f"{k}={v}" for k, v in eff.params.items()
                            if k not in ("clause", "description") and v not in ("", None, False)]
            params_str = _c(_DIM, "  (" + ", ".join(params_parts) + ")") if params_parts else ""
            print(
                f"    {_c(_DIM, '[' + eff.timing + ']')}"
                f" {_c(effect_color, eff.effect_type.value)}"
                f"{params_str}{conf_str}"
            )

            # Cible structurée
            tgt = eff.target.pretty()
            if tgt:
                print(f"       {_c(_DIM, 'target:')} {tgt}")

            # Condition if …
            if eff.condition:
                print(f"       {_c(_DIM, 'if:')} {_c(_YELLOW, eff.condition)}")

            # Durée
            if eff.duration not in ("unknown", "permanent"):
                print(f"       {_c(_DIM, 'dur:')} {eff.duration}")

            # Optionnel
            if eff.optional:
                print(f"       {_c(_DIM, '[optional]')}")

            # Coût d'activation
            if eff.activation_cost:
                print(f"       {_c(_DIM, 'cost:')} {eff.activation_cost}")
    else:
        print(_c(_DIM, "    Aucun effet classifié."))

    print(_c(_BOLD + _CYAN, _sep("═", w)))


# ── Sauvegarde ─────────────────────────────────────────────────────────────

def save_validation(
    card: ParsedCard,
    status: str,
    correction: str,
    out_path: Path,
) -> None:
    record = card.to_dict()
    record["validation"] = {
        "status": status,
        "correction": correction,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Stats de session ───────────────────────────────────────────────────────

class SessionStats:
    def __init__(self):
        self.total   = 0
        self.ok      = 0
        self.wrong   = 0
        self.skipped = 0

    def display(self) -> None:
        print()
        print(_c(_BOLD, "  Session terminée :"))
        print(f"    Cartes vues     : {self.total}")
        print(f"    Validées (OK)   : {_c(_GREEN, str(self.ok))}")
        print(f"    Corrigées       : {_c(_YELLOW, str(self.wrong))}")
        print(f"    Ignorées (skip) : {_c(_DIM, str(self.skipped))}")
        print()


# ── Boucle principale ──────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(_c(_BOLD + _CYAN, "\n  OPTCG — Validation des effets de cartes"))
    print(_c(_DIM,          "  Entrée = OK  |  'q' = quitter  |  'skip' = passer\n"))

    df = load_cards(csv_path)

    # Filtres optionnels
    if args.filter_type:
        df = df[df["card_type"].str.lower() == args.filter_type.lower()]
    if args.filter_color:
        df = df[df["color"].str.contains(args.filter_color, case=False, na=False)]
    if args.filter_has_text:
        df = df[df["card_text"].str.strip() != ""]

    if df.empty:
        print(_c(_RED, "  Aucune carte ne correspond aux filtres."))
        sys.exit(1)

    total_cards = len(df)
    print(f"  {_c(_BOLD, str(total_cards))} cartes disponibles.")

    stats = SessionStats()

    already_seen: set[str] = set()

    # Charge les cartes déjà validées pour éviter les doublons
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    already_seen.add(rec.get("card_id", ""))
                except json.JSONDecodeError:
                    pass
        if already_seen:
            print(f"  {_c(_DIM, str(len(already_seen)))} cartes déjà validées (ignorées).")

    remaining = df[~df["card_id"].isin(already_seen)]
    if remaining.empty:
        print(_c(_GREEN, "\n  Toutes les cartes ont déjà été validées !"))
        return

    # Si une carte précise est demandée
    if args.card:
        sub = remaining[remaining["card_id"] == args.card]
        if sub.empty:
            print(_c(_RED, f"  Carte {args.card} introuvable ou déjà validée."))
            sys.exit(1)
        rows = [sub.iloc[0]]
        random_mode = False
    else:
        rows = None
        random_mode = True

    try:
        while True:
            if random_mode:
                if remaining.empty:
                    print(_c(_GREEN, "\n  Toutes les cartes ont été validées !"))
                    break
                row = remaining.sample(1).iloc[0]
            else:
                if not rows:
                    break
                row = rows.pop(0)

            card = df_row_to_parsed_card(row)
            card = run_pipeline(card)

            display_card(card)
            stats.total += 1

            prompt = _c(_BOLD, "\n  Valider ? ") + _c(_DIM, "[Entrée=OK / décris l'erreur / 'skip' / 'q'] ") + ": "
            try:
                user_input = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if user_input.lower() == "q":
                break

            if user_input.lower() == "skip":
                stats.skipped += 1
                print(_c(_DIM, "  ↷ Ignoré.\n"))
                continue

            if user_input == "":
                status = "ok"
                correction = ""
                stats.ok += 1
                print(_c(_GREEN, "  ✓ Validé.\n"))
            else:
                status = "wrong"
                correction = user_input
                stats.wrong += 1
                print(_c(_YELLOW, f"  ✎ Correction enregistrée.\n"))

            save_validation(card, status, correction, out_path)

            # Retire la carte des restantes
            remaining = remaining[remaining["card_id"] != card.card_id]

    except KeyboardInterrupt:
        pass

    stats.display()
    print(f"  Résultats sauvegardés dans : {_c(_BOLD, str(out_path))}")


# ── Entrée ─────────────────────────────────────────────────────────────────

def main() -> None:
    default_csv = str(_ROOT / "data" / "cards_tcgcsv.csv")
    default_out = str(_ROOT / "data" / "validated_effects.jsonl")

    parser = argparse.ArgumentParser(
        description="Validation interactive des effets de cartes OPTCG.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--csv",  default=default_csv, help="Chemin vers le CSV de cartes")
    parser.add_argument("--out",  default=default_out, help="Fichier JSONL de sortie")
    parser.add_argument("--card", default=None,        help="ID d'une carte spécifique")
    parser.add_argument("--filter-type",  default=None, help="Filtrer par type (Character, Event, ...)")
    parser.add_argument("--filter-color", default=None, help="Filtrer par couleur (Red, Blue, ...)")
    parser.add_argument("--filter-has-text", action="store_true",
                        help="Ignorer les cartes sans texte d'effet")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
