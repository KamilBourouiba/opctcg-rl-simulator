from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from .card_db import CardDef


class Phase(Enum):
    MULLIGAN = auto()  # Choix keep/relancer avant le 1er tour
    MAIN = auto()
    BATTLE = auto()
    BLOCKER = auto()   # P0 choisit un Blocker pendant l'attaque de P1


@dataclass
class PlayabilityContext:
    phase: Phase
    don_available: int
    hand_size: int


def can_play_character(card: CardDef, ctx: PlayabilityContext) -> bool:
    """Personnage / Stage jouable en Main Phase si coût payable."""
    if ctx.phase != Phase.MAIN:
        return False
    if ctx.hand_size <= 0:
        return False
    ct = (card.card_type or "").lower()
    if "event" in ct:
        return False   # les Events ont leur propre fonction
    return ctx.don_available >= card.cost


def can_play_event(card: CardDef, ctx: PlayabilityContext) -> bool:
    """
    Event jouable en Main Phase (CR 10-2-1) :
    phase MAIN, coût en DON!! payable, le type doit contenir « event ».
    """
    if ctx.phase != Phase.MAIN:
        return False
    if ctx.hand_size <= 0:
        return False
    ct = (card.card_type or "").lower()
    if "event" not in ct:
        return False
    return ctx.don_available >= card.cost


def can_play_card(card: CardDef, ctx: PlayabilityContext) -> bool:
    """
    Peut-on jouer cette carte en Main Phase ?
    Couvre les Characters, Stages et Events.
    """
    ct = (card.card_type or "").lower()
    if "event" in ct:
        return can_play_event(card, ctx)
    return can_play_character(card, ctx)
