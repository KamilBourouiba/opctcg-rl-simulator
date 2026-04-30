"""
État des restrictions par personnage (et parallèle pour le Leader).

Sépare :
  - restrictions « ce tour » (effets adverses, Perona, …) → effacées au Refresh ;
  - restrictions « imprimées » sur la carte tant qu’elle est en jeu.

La légalité d’attaque / ciblage / K.O. lit cet état plutôt que des booléens épars.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..card_db import CardDef

    from .keyword_grant_state import KeywordGrantState


@dataclass
class RestrictionState:
    """Restrictions modifiables par les effets (perso ou Leader)."""

    # ── Ce tour (effacé par clear_turn_scoped au Refresh) ───────────────────
    cannot_attack_this_turn: bool = False
    cannot_become_active_this_turn: bool = False
    cannot_be_ko_this_turn: bool = False
    cannot_be_targeted_this_turn: bool = False

    # ── Tant que la carte reste sur le terrain (pose / texte statique) ─────
    cannot_attack_printed: bool = False
    cannot_be_ko_printed: bool = False
    cannot_be_targeted_printed: bool = False

    def clear_turn_scoped(self) -> None:
        self.cannot_attack_this_turn = False
        self.cannot_become_active_this_turn = False
        self.cannot_be_ko_this_turn = False
        self.cannot_be_targeted_this_turn = False

    def blocks_attack_declaration(self) -> bool:
        return self.cannot_attack_this_turn or self.cannot_attack_printed

    def blocks_becoming_active_after_refresh(self, was_rested_before_refresh: bool) -> bool:
        """Si vrai et le perso était reposé, il reste reposé après le Refresh."""
        if not was_rested_before_refresh:
            return False
        return self.cannot_become_active_this_turn

    def blocks_ko(self) -> bool:
        return self.cannot_be_ko_this_turn or self.cannot_be_ko_printed

    def blocks_opponent_targeting(self) -> bool:
        return self.cannot_be_targeted_this_turn or self.cannot_be_targeted_printed


_RE_THIS_CHAR_CANNOT_ATTACK = re.compile(
    r"this\s+(?:character|card|leader)\s+cannot\s+attack\b", re.IGNORECASE
)
_RE_THIS_CHAR_CANNOT_KO = re.compile(
    r"this\s+(?:character|card|leader)\s+cannot\s+be\s+k\.?o\.?'?d?\b", re.IGNORECASE
)
_RE_THIS_CHAR_CANNOT_TARGETED = re.compile(
    r"this\s+(?:character|card|leader)\s+cannot\s+be\s+targeted\b", re.IGNORECASE
)


def bootstrap_printed_restrictions_from_card(cd: CardDef | None) -> RestrictionState:
    """
    Remplit les drapeaux « printed » à partir du texte (pose du perso).
    Évite les phrases du type « your opponent's Characters cannot attack »
    en exigeant « this Character » / « this card ».
    """
    r = RestrictionState()
    if cd is None:
        return r
    blob = cd.card_text or ""
    if _RE_THIS_CHAR_CANNOT_ATTACK.search(blob):
        r.cannot_attack_printed = True
    if _RE_THIS_CHAR_CANNOT_KO.search(blob):
        r.cannot_be_ko_printed = True
    if _RE_THIS_CHAR_CANNOT_TARGETED.search(blob):
        r.cannot_be_targeted_printed = True
    return r


def character_may_declare_attack(
    *,
    rested: bool,
    just_played: bool,
    has_rush: bool,
    has_rush_char: bool,
    restrictions: RestrictionState,
    keyword_grants: KeywordGrantState | None = None,
) -> bool:
    """Un personnage actif peut-il déclarer une attaque (hors règle Rush:Character → Leader) ?"""
    from .keyword_grant_state import KeywordGrantState

    kg = keyword_grants or KeywordGrantState()
    eff_rush = has_rush or kg.rush
    eff_rush_char = has_rush_char or kg.rush_character
    if rested:
        return False
    if restrictions.blocks_attack_declaration():
        return False
    if just_played and not eff_rush and not eff_rush_char:
        return False
    return True


def leader_may_declare_attack(*, leader_rested: bool, restrictions: RestrictionState) -> bool:
    if leader_rested:
        return False
    if restrictions.blocks_attack_declaration():
        return False
    return True
