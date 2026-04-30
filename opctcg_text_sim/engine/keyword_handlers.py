"""
Gestionnaires par mot-clé / timing (OP Comprehensive Rules, section 8–10).

Chaque timing a maintenant sa fonction câblée :
  - [On Play] et [Main] : on_play_resolver (regex historique, robuste)
  - [On K.O.], [When Attacking], [Trigger], [End of Your Turn], etc. :
    effect_resolver (pipeline models/ → ParsedEffect → mutation GameState)
  - Mots-clés passifs ([Rush], [Blocker], etc.) : gérés dans simulator._has_*
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..card_db import CardDef
    from ..simulator import PlayerState, SimplifiedOPSim


@dataclass
class EffectContext:
    sim: "SimplifiedOPSim"
    owner: "PlayerState"
    owner_idx: int          # 0 ou 1
    card: "CardDef"


# ── Helpers d'accès au cache ──────────────────────────────────────────────────

def _cache(ctx: EffectContext):
    """Accède au CardEffectCache stocké sur le simulateur."""
    return ctx.sim._effect_cache


def _resolve(timing: str, ctx: EffectContext) -> bool:
    """Raccourci : résout les effets structurés pour un timing donné."""
    from .effect_resolver import resolve_effects
    return resolve_effects(timing, ctx, _cache(ctx))


# ── Mots-clés passifs (10-1) — déjà gérés dans simulator._has_* ─────────────

def kw_rush(ctx: EffectContext) -> None:
    pass  # BoardChar.has_rush est défini à la pose du personnage


def kw_double_attack(ctx: EffectContext) -> None:
    pass  # BoardChar.has_double_attack est défini à la pose


def kw_banish(ctx: EffectContext) -> None:
    pass  # géré dans simulator._has_banish + _resolve_attack


def kw_blocker(ctx: EffectContext) -> None:
    pass  # BoardChar.has_blocker défini à la pose ; phase Blocker dans sim


def kw_unblockable(ctx: EffectContext) -> None:
    pass  # BoardChar.has_unblockable défini à la pose


def kw_rush_character(ctx: EffectContext) -> None:
    pass


# ── Timing : [On Play] ────────────────────────────────────────────────────────

def kw_on_play(ctx: EffectContext) -> None:
    """[On Play] — double résolution : regex (on_play_resolver) + structurée."""
    from .on_play_resolver import apply_on_play_effects
    apply_on_play_effects(ctx)
    # Le résolveur structuré couvre les effets non captés par les regex
    _resolve("On Play", ctx)


# ── Timing : [Main] ──────────────────────────────────────────────────────────

def kw_main_event(ctx: EffectContext) -> None:
    """[Main] — résolution event (regex) + structurée."""
    from .on_play_resolver import apply_main_text_effects
    apply_main_text_effects(ctx)
    _resolve("Main", ctx)


# ── Timing : [Trigger] ───────────────────────────────────────────────────────

def kw_trigger(ctx: EffectContext) -> None:
    """[Trigger] — activé quand la carte sort de la pile Vie (CR 10-1-5)."""
    # on_play_resolver ne gère pas Trigger ; on utilise effect_resolver
    if not _resolve("Trigger", ctx):
        # Fallback : si la carte est un personnage, on la joue sur le terrain
        from .on_play_resolver import apply_on_play_effects
        apply_on_play_effects(ctx)


# ── Timing : [When Attacking] ────────────────────────────────────────────────

def kw_when_attacking(ctx: EffectContext) -> None:
    """[When Attacking] — déclenché à la déclaration d'attaque du personnage."""
    _resolve("When Attacking", ctx)


# ── Timing : [On K.O.] ───────────────────────────────────────────────────────

def kw_on_ko(ctx: EffectContext) -> None:
    """[On K.O.] — déclenché quand le personnage est mis K.O."""
    from .on_play_resolver import apply_on_ko_effects
    apply_on_ko_effects(ctx)
    _resolve("On K.O.", ctx)


# ── Timing : [Activate:Main] ─────────────────────────────────────────────────

def kw_activate_main(ctx: EffectContext) -> None:
    """[Activate:Main] — coût de repose + effet pendant la Main Phase."""
    from .on_play_resolver import apply_activate_main_effects
    apply_activate_main_effects(ctx)
    _resolve("Main", ctx)


# ── Timing : [On Your Opponent's Attack] ─────────────────────────────────────

def kw_on_opponent_attack(ctx: EffectContext) -> None:
    """[On Your Opponent's Attack] — réaction défensive."""
    _resolve("On Your Opponent's Attack", ctx)


# ── Timing : [On Block] ──────────────────────────────────────────────────────

def kw_on_block(ctx: EffectContext) -> None:
    """[On Block] — déclenché quand le Blocker intercepte."""
    _resolve("On Block", ctx)


# ── Timing : [End of Your Turn] ──────────────────────────────────────────────

def kw_end_your_turn(ctx: EffectContext) -> None:
    """[End of Your Turn] — résolu en fin de tour du propriétaire."""
    _resolve("End of Your Turn", ctx)


def kw_end_opponent_turn(ctx: EffectContext) -> None:
    """[End of Your Opponent's Turn] — résolu en fin de tour adverse."""
    _resolve("End of Your Opponent's Turn", ctx)


# ── Timing : [Your Turn] / [Opponent's Turn] (effets permanents) ─────────────

def kw_your_turn(ctx: EffectContext) -> None:
    """[Your Turn] — effet permanent actif pendant le tour du joueur."""
    _resolve("Your Turn", ctx)


def kw_opponents_turn(ctx: EffectContext) -> None:
    """[Opponent's Turn] — effet permanent actif pendant le tour adverse."""
    _resolve("Opponent's Turn", ctx)


# ── Modificateurs de timing ──────────────────────────────────────────────────

def kw_once_per_turn(ctx: EffectContext) -> None:
    pass  # géré via flags `*_once_per_turn_*_used` dans PlayerState


def kw_don_x_condition(ctx: EffectContext) -> None:
    """DON!! x N : débloque un effet si N DON!! ou plus sont attachés."""
    _resolve("DON!! xN", ctx)


def kw_trash_instruction(ctx: EffectContext) -> None:
    pass


def kw_counter_event(ctx: EffectContext) -> None:
    """[Counter] sur Event — phase Counter (stub : puissance bonus non sim)."""
    pass


# ── Table de dispatch ─────────────────────────────────────────────────────────
# Ordre : sous-chaînes les plus longues en premier (évite les collisions).

_KEYWORD_DISPATCH: list[tuple[str, Callable[[EffectContext], None]]] = [
    ("[rush: character]",              kw_rush_character),
    ("[double attack]",                kw_double_attack),
    ("[once per turn]",                kw_once_per_turn),
    ("[end of your opponent's turn]",  kw_end_opponent_turn),
    ("[end of your turn]",             kw_end_your_turn),
    ("[on your opponent's attack]",    kw_on_opponent_attack),
    ("[when attacking]",               kw_when_attacking),
    ("[activate: main]",               kw_activate_main),
    ("[opponent's turn]",              kw_opponents_turn),
    ("[your turn]",                    kw_your_turn),
    ("[unblockable]",                  kw_unblockable),
    ("[blocker]",                      kw_blocker),
    ("[trigger]",                      kw_trigger),
    ("[counter]",                      kw_counter_event),
    ("[banish]",                       kw_banish),
    ("[on k.o.]",                      kw_on_ko),
    ("[on ko]",                        kw_on_ko),
    ("[on block]",                     kw_on_block),
    ("[on play]",                      kw_on_play),
    ("[rush]",                         kw_rush),
    ("[main]",                         kw_main_event),
]

# À la pose uniquement : ne pas scanner le texte entier avec _KEYWORD_DISPATCH, sinon
# toute carte « … [On K.O.] … » déclencherait kw_on_ko au moment du [On Play] (bug vie / dégâts).
_KEYWORD_DISPATCH_ON_PLAY: list[tuple[str, Callable[[EffectContext], None]]] = [
    ("[rush: character]",              kw_rush_character),
    ("[double attack]",                kw_double_attack),
    ("[once per turn]",                kw_once_per_turn),
    ("[unblockable]",                  kw_unblockable),
    ("[blocker]",                      kw_blocker),
    ("[banish]",                       kw_banish),
    ("[counter]",                      kw_counter_event),
    ("[rush]",                         kw_rush),
    ("[main]",                         kw_main_event),
    ("[on play]",                      kw_on_play),
]


def dispatch_on_play(ctx: EffectContext) -> None:
    """
    Appelle les handlers pertinents quand une carte entre en jeu (perso / Stage).
    Ne déclenche pas [On K.O.], [When Attacking], [Trigger], fins de tour, etc.
    """
    parts = [(ctx.card.card_text or "").lower()]
    for k in ctx.card.keywords:
        parts.append(k.lower())
    blob = " ".join(parts)

    fired: set[str] = set()
    for needle, fn in _KEYWORD_DISPATCH_ON_PLAY:
        if needle in blob and fn.__name__ not in fired:
            fired.add(fn.__name__)
            fn(ctx)

    # Texte « On Play: » sans crochets (extract_timing_segment le reconnaît) mais absent du blob « [on play] »
    if kw_on_play.__name__ not in fired:
        from .on_play_resolver import extract_on_play_segment

        if extract_on_play_segment(ctx.card.card_text or "") is not None:
            fired.add(kw_on_play.__name__)
            kw_on_play(ctx)


# ── Dispatch par timing explicite (pour phases en dehors du jeu de carte) ────

def dispatch_timing(timing: str, ctx: EffectContext) -> None:
    """
    Résout les effets d'un timing spécifique sur une carte déjà en jeu
    (ex. fin de tour, attaque, etc.), en passant par le résolveur structuré.
    """
    from .effect_resolver import resolve_effects
    resolve_effects(timing, ctx, ctx.sim._effect_cache)



