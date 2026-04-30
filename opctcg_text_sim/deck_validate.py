"""
Validation légère des decklists (format OPTCGSim) pour le simulateur.

Règles appliquées (configurables) :
  - Deck principal = 50 cartes après retrait du Leader (5-1-2-3).
  - Max 4 exemplaires d’un même numéro de carte (5-1-2-3).
  - Couleurs : chaque carte du deck doit partager une couleur avec le Leader (CSV).
  - Stages : jusqu’à **4 cartes Stage** au total dans le deck 50 (même limite que les autres types
    par id : max 4× le même numéro). Sur le terrain, **une seule** Stage en zone Stage à la fois
    (remplacement : l’ancienne part au trash) — appliqué dans le simulateur, pas ici.
"""
from __future__ import annotations

from dataclasses import dataclass

from .card_db import CardDef
from .deck_parser import deck_to_multiset, extract_leader_from_multiset, parse_deck_file


@dataclass
class DeckValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    leader_id: str | None
    main_deck_size: int


def validate_deck_file(
    path,
    cards: dict[str, CardDef],
    *,
    max_stage_cards: int = 4,
    official_main_size: int = 50,
) -> DeckValidationResult:
    """
    ``path`` : pathlib.Path ou str.
    ``max_stage_cards`` : nombre max de cartes dont le type CSV est « Stage » dans le deck 50.
    """
    from pathlib import Path

    p = Path(path)
    errors: list[str] = []
    warnings: list[str] = []

    entries = parse_deck_file(p)
    if not entries:
        return DeckValidationResult(False, ["Deck vide ou illisible."], [], None, 0)

    m = deck_to_multiset(entries)
    m2, leader = extract_leader_from_multiset(
        m,
        cards,
        explicit_leader_id=None,
        preferred_cid_order=[cid for cid, _ in entries],
    )
    if leader is None:
        errors.append("Aucune carte Leader identifiable dans le deck (vérifie le CSV / la 1ère ligne).")

    def _color_tokens(s: str) -> set[str]:
        raw = (s or "").strip().lower()
        if not raw:
            return set()
        out: set[str] = set()
        for part in raw.replace(";", "/").replace(",", "/").split("/"):
            tok = part.strip()
            if tok:
                out.add(tok)
        return out

    leader_colors: set[str] = set()
    if leader:
        leader_colors = _color_tokens(cards[leader].color or "")

    stage_count = 0
    for cid, n in m2.items():
        cd = cards.get(cid)
        if cd is None:
            errors.append(f"Carte inconnue dans le CSV : {cid}")
            continue
        ct = (cd.card_type or "").strip().lower()
        if n > 4:
            errors.append(f"Plus de 4 exemplaires : {cid} ×{n}")
        if "stage" in ct:
            stage_count += n
        card_cols = _color_tokens(cd.color or "")
        if leader_colors and not (card_cols & leader_colors):
            errors.append(
                f"Couleur incompatible avec le Leader : {cid} ({cd.color}) vs leader {leader_colors}"
            )

    if stage_count > max_stage_cards:
        errors.append(
            f"Trop de cartes Stage dans le deck : {stage_count} (max autorisé : {max_stage_cards})."
        )

    size = sum(m2.values())
    if size != official_main_size:
        errors.append(f"Taille du deck principal : {size} (attendu {official_main_size} après retrait du Leader).")

    ok = not errors
    return DeckValidationResult(ok, errors, warnings, leader, size)
