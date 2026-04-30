"""
Modificateurs de puissance « ce tour » (boost / malus) pour persos et Leader.

Les valeurs imprimées sur la carte restent dans ``BoardChar.power`` / ``PlayerState.leader_power``.
Les effets du type +2000 / −3000 « during this turn » passent par ce module et sont
effacés au Refresh (comme ``RestrictionState``).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PowerModifierState:
    """Bonus et pénalités de puissance cumulables jusqu’au prochain Refresh."""

    bonus_turn: int = 0   # +N (effets alliés / OOA, etc.)
    penalty_turn: int = 0  # pénalité positive à soustraire (effets adverses « −N power »)

    def clear_turn_scoped(self) -> None:
        self.bonus_turn = 0
        self.penalty_turn = 0

    def net_bonus(self) -> int:
        """Bonus algebrique appliqué sur la puissance imprimée."""
        return self.bonus_turn - self.penalty_turn
