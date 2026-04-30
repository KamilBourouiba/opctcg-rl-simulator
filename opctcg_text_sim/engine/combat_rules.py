"""
Règles de combat (OP-TCG CR v1.2) : puissance effective, Counter, Blocker.

Step Counter (CR 7-1-3) :
  - [(Symbol)Counter] sur un CHARACTER card : trash depuis la main → +N power sur
    le Leader ou un perso défenseur (CR 7-1-3-2-1).
  - [Counter] sur un EVENT card : payer le coût (reposer des DON!!) + trash → effet
    Counter de l'Event (CR 7-1-3-2-2 / CR 10-2-4).
  Le défenseur peut utiliser autant de Counter qu'il le souhaite (CR 7-1-3-2).

Puissance effective (CR 6-5-5-2) :
  - Character : puissance imprimée + ``PowerModifierState`` (boost/malus ce tour)
    + optionnellement +1000 × DON!! attachés.
  - Leader : idem via ``PlayerState.leader_power`` + ``leader_power_modifiers`` +
    ``effective_leader_power()`` pour le DON!! attaché.
  - En défense pendant le tour adverse : pas de +1000 des DON!! attachés (simulator) ;
    les modificateurs de puissance d’effet (ex. +2000 OOA) restent pris en compte via
    ``board_character_power(..., include_attached_don_bonus=False)``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..card_db import CardDef

if TYPE_CHECKING:
    from ..simulator import BoardChar, PlayerState


def board_character_power(
    b: "BoardChar",
    cards: dict[str, CardDef] | None = None,
    *,
    include_attached_don_bonus: bool = True,
) -> int:
    """
    Puissance de combat d'un personnage : imprimée + modificateurs ce tour
    (``power_modifiers``) + optionnellement DON!! attachés.
    """
    n = int(b.power) + int(b.power_modifiers.net_bonus())
    if include_attached_don_bonus:
        n += 1000 * int(b.attached_don)
    return max(0, n)


def effective_power(b: "BoardChar", cards: dict[str, CardDef]) -> int:
    """Alias : puissance en attaque (DON!! attachés inclus). ``cards`` réservé pour extensions."""
    _ = cards
    return board_character_power(b, cards, include_attached_don_bonus=True)


def has_blocker_keyword(cd: CardDef) -> bool:
    blob = " ".join(cd.keywords).lower() + " " + (cd.card_text or "").lower()
    return "[blocker]" in blob


def is_counter_event(cd: CardDef) -> bool:
    """Event card utilisable comme Counter (CR 10-2-4)."""
    ct = (cd.card_type or "").lower()
    if "event" not in ct:
        return False
    t = (cd.card_text or "").lower()
    return "[counter]" in t


def is_character_counter(cd: CardDef) -> bool:
    """
    Character card avec une valeur [(Symbol)Counter] > 0 (CR 7-1-3-2-1).
    La valeur ``counter`` sur la fiche représente le bonus de puissance.
    """
    ct = (cd.card_type or "").lower()
    if "character" not in ct:
        return False
    return int(cd.counter or 0) > 0


def best_counter_card_index(
    hand: list[str],
    cards: dict[str, CardDef],
    active_don: int,
) -> tuple[int | None, int]:
    """
    Cherche la meilleure carte Counter dans la main (Character ou Event).
    Priorité : valeur counter la plus élevée.

    - Character Counter : pas de coût DON!! → priorité naturelle.
    - Event Counter : nécessite de reposer un nombre de DON!! égal au coût.

    Retourne (index, counter_value) ou (None, 0).
    """
    best_i: int | None = None
    best_v = -1
    for i, cid in enumerate(hand):
        cd = cards.get(cid)
        if cd is None:
            continue
        v = int(cd.counter or 0)
        if v <= 0:
            continue
        if is_character_counter(cd):
            # Pas de coût DON!! pour le character counter
            if v > best_v:
                best_v = v
                best_i = i
        elif is_counter_event(cd):
            # Coût DON!! requis
            cost = int(cd.cost or 0)
            if active_don >= cost and v > best_v:
                best_v = v
                best_i = i
    return best_i, best_v


def apply_counter_stack_until_safe(
    defender: "PlayerState",
    cards: dict[str, CardDef],
    incoming: int,
    base_defense: int,
    *,
    max_counters: int = 6,
) -> int:
    """
    CR 7-1-3-2 : le défenseur utilise autant de Counter qu'il veut (Character ou Event)
    jusqu'à ce que sa défense dépasse l'attaque, ou qu'il n'ait plus de Counter.

    - Character Counter (CR 7-1-3-2-1) : trash depuis la main, aucun coût DON!!.
    - Event Counter (CR 7-1-3-2-2) : reposer le coût en DON!! + trash.

    Retourne la défense totale après Counter.
    """
    defense = int(base_defense)
    used = 0
    while used < max_counters and incoming >= defense:
        active_don = int(getattr(defender, "don_active", 0))
        idx, cval = best_counter_card_index(defender.hand, cards, active_don)
        if idx is None:
            break
        cid = defender.hand.pop(idx)
        cd = cards[cid]
        defender.trash.append(cid)

        # Payer le coût DON!! pour les Events Counter
        if is_counter_event(cd):
            cost = int(cd.cost or 0)
            paid = 0
            # Reposer les DON!! actifs pour payer le coût
            don_cost_to_pay = min(cost, active_don)
            defender.don_active = max(0, active_don - don_cost_to_pay)
            defender.don_rested = getattr(defender, "don_rested", 0) + don_cost_to_pay

        defense += cval
        used += 1
    return defense
