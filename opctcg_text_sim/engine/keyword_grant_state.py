"""
Buffs de mots-clés « ce tour » (effets type « gains [Rush] », « gains [Double Attack] », …).

Les capacités imprimées sur la carte restent dans ``BoardChar.has_*`` ; les effets
temporaires posent des drapeaux ici puis ``clear_turn_scoped()`` au Refresh.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KeywordGrantState:
    double_attack: bool = False
    banish: bool = False
    unblockable: bool = False
    rush: bool = False
    rush_character: bool = False
    """Peut cibler des personnages adverses actifs (non reposés) en attaque."""
    can_attack_active_characters: bool = False

    def clear_turn_scoped(self) -> None:
        self.double_attack = False
        self.banish = False
        self.unblockable = False
        self.rush = False
        self.rush_character = False
        self.can_attack_active_characters = False
