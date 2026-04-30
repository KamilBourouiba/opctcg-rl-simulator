"""
Effets texte (anglais type TCGplayer/tcgcsv) : [On Play], [Main], etc.

Inclut notamment look / search deck, trash depuis le deck, pioche + défausse main,
cartes depuis la défausse, DON!!, placement sous le deck — via regex sur
``_RULES`` (motifs les plus longs en premier).

Les cartes avec textes [On Play] / [Trigger] très longs ou plusieurs blocs sur
une même ligne : ``extract_timing_segment`` coupe avant le prochain timing connu
ou tout ``[...]`` en début de ligne (ex. ``[Counter]``), puis applique une borne
``MAX_TIMING_SEGMENT_CHARS`` pour limiter le coût regex et la profondeur des boucles.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable, Pattern

from .keyword_handlers import EffectContext
from .restriction_state import bootstrap_printed_restrictions_from_card

from ..card_db import CardDef
from .combat_rules import effective_power

if TYPE_CHECKING:
    from ..simulator import PlayerState, SimplifiedOPSim


def _norm(s: str) -> str:
    return (
        str(s)
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


# Borne dure sur un segment d'effet (évite chemins pathologiques sur cartes « roman »).
MAX_TIMING_SEGMENT_CHARS = 16384

# Prochains blocs de timing en début de ligne (prioritaire sur une coupure générique).
_RE_TIMING_STRICT_NEWLINE = re.compile(
    r"\n\s*\[(?:"
    r"trigger|main|on\s*play|on\s*k\.?o\.?|when\s+attacking|"
    r"activate\s*:\s*main|activate\s*:\s*counter|activate\s*:\s*action|"
    r"end\s+of\s+your\s+turn|end\s+of\s+your\s+opponent'?s?\s+turn|"
    r"drip|once\s+per\s+turn|on\s+your\s+opponent'?s?\s+attack|"
    r"on\s+block"
    r")[^\]\n]*\]",
    re.IGNORECASE,
)

# Même ligne : « … effet. [Trigger] … » sans saut de ligne entre les blocs.
_RE_TIMING_INLINE = re.compile(
    r"(?<=[\w\)\]\.\"'])\s+\[(?:"
    r"trigger|main|on\s*play|on\s*k\.?o\.?|when\s+attacking|"
    r"activate\s*:\s*main|activate\s*:\s*counter|"
    r"end\s+of\s+your\s+turn|drip|once\s+per\s+turn|"
    r"on\s+your\s+opponent'?s?\s+attack"
    r")[^\]\n]*\]",
    re.IGNORECASE,
)

# Secours : tout nouveau ``[...]`` en début de ligne (Counter, mots-clés, etc.).
_RE_ANY_BRACKET_NEWLINE = re.compile(r"\n\s*\[[^\n]{1,200}\]")


def _clip_timing_segment_body(rest: str, *, max_chars: int = MAX_TIMING_SEGMENT_CHARS) -> str:
    """
    Limite le texte résolu pour un timing au seul paragraphe concerné :
    coupe avant le prochain bloc reconnu, puis tronque à ``max_chars``.
    """
    rest = (rest or "").strip()
    if not rest:
        return rest
    m = _RE_TIMING_STRICT_NEWLINE.search(rest)
    if m:
        rest = rest[: m.start()]
    else:
        m_any = _RE_ANY_BRACKET_NEWLINE.search(rest)
        if m_any:
            rest = rest[: m_any.start()]
    m_inline = _RE_TIMING_INLINE.search(rest)
    if m_inline:
        rest = rest[: m_inline.start()]
    if len(rest) > max_chars:
        rest = rest[:max_chars]
    return rest.strip()


def extract_timing_segment(card_text: str, timing: str) -> str | None:
    """
    Extrait le passage après ``[On Play]``, ``[Main]``, ``[Trigger]``, etc.
    ``timing`` : ``"on_play"``, ``"main"``, ou ``"trigger"``.
    """
    if not card_text or not str(card_text).strip():
        return None
    t = _norm(card_text)
    tl = timing.lower()
    if tl == "main":
        m = re.search(r"\[main\]\s*:?\s*", t, flags=re.IGNORECASE)
    elif tl == "trigger":
        m = re.search(r"\[trigger\]\s*:?\s*", t, flags=re.IGNORECASE)
    else:
        m = re.search(r"\[on\s*play\]\s*:?\s*", t, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"(?:^|\n)\s*on\s*play\s*:\s*", t, flags=re.IGNORECASE)
    if not m:
        return None
    rest = _clip_timing_segment_body(t[m.end() :])
    return rest or None


def extract_on_play_segment(card_text: str) -> str | None:
    """Clause ``[On Play]`` (alias)."""
    return extract_timing_segment(card_text, "on_play")


def extract_on_ko_segment(card_text: str) -> str | None:
    """Extrait le passage après ``[On K.O.]`` (ou ``[On KO]``)."""
    if not card_text or not str(card_text).strip():
        return None
    t = _norm(card_text)
    m = re.search(r"\[on\s*k\.?o\.?\]\s*:?\s*", t, flags=re.IGNORECASE)
    if not m:
        return None
    rest = _clip_timing_segment_body(t[m.end():])
    return rest or None


def _opponent(ctx: EffectContext) -> "PlayerState":
    return ctx.sim.p1 if ctx.owner_idx == 0 else ctx.sim.p0


def _draw_n(ctx: EffectContext, n: int) -> int:
    """
    Pioche n cartes via le simulateur (qui gère le deck vide = défaite, CR 9-2-1-2).
    Toute pioche — y compris via effet de carte — peut déclencher la perte si le deck est vide.
    """
    c = 0
    for _ in range(max(0, n)):
        if ctx.sim.done:
            break
        if not ctx.sim._draw(ctx.owner):
            break   # deck vide → done=True dans _draw
        c += 1
    return c


def _discard_n_random(ctx: EffectContext, pl: "PlayerState", n: int) -> int:
    sim = ctx.sim
    c = 0
    for _ in range(max(0, n)):
        if not pl.hand:
            break
        i = int(sim.rng.integers(0, len(pl.hand)))
        cid = pl.hand.pop(i)
        pl.trash.append(cid)
        c += 1
    return c


def _add_rested_don_from_deck(ctx: EffectContext, n: int) -> int:
    """« Add up to N rested DON!! … to your cost area » (approximation)."""
    pl = ctx.owner
    sim = ctx.sim
    if n <= 0 or pl.don_deck <= 0:
        return 0
    in_cost = pl.don_active + pl.don_rested
    room = max(0, sim.max_don - in_cost)
    gain = min(n, pl.don_deck, room)
    pl.don_deck -= gain
    pl.don_rested += gain
    return gain


def _add_active_don_from_deck(ctx: EffectContext, n: int) -> int:
    """« Add … DON!! from your DON!! deck … set as active » (zone de coût : jetons actifs)."""
    pl = ctx.owner
    sim = ctx.sim
    if n <= 0 or pl.don_deck <= 0:
        return 0
    in_cost = pl.don_active + pl.don_rested
    room = max(0, sim.max_don - in_cost)
    gain = min(n, pl.don_deck, room)
    pl.don_deck -= gain
    pl.don_active += gain
    return gain


def _gain_life(ctx: EffectContext, n: int) -> None:
    """
    Ajoute n cartes du dessus du deck au SOMMET de la pile vie (index 0).
    CR : la plupart des effets de gain de vie ajoutent au sommet.
    """
    pl = ctx.owner
    for _ in range(max(0, n)):
        if len(pl.life_cards) >= 99 or not pl.deck:
            break
        pl.life_cards.insert(0, pl.deck.pop(0))


def _trash_life_top(ctx: EffectContext, n: int) -> None:
    """Envoie n cartes du dessus de la pile vie au trash (coût de certains effets)."""
    pl = ctx.owner
    for _ in range(min(n, len(pl.life_cards))):
        pl.trash.append(pl.life_cards.pop(0))


def _shuffle_deck(ctx: EffectContext) -> None:
    d = ctx.owner.deck
    if len(d) > 1:
        ctx.sim.rng.shuffle(d)


def _type_blob(cd: CardDef) -> str:
    return " ".join(
        (
            (cd.card_text or ""),
            cd.name,
            " ".join(cd.keywords),
            cd.card_type or "",
        ),
    ).lower()


def _type_matches(cd: CardDef, fragment: str) -> bool:
    frag = fragment.strip().strip('"').lower()
    if not frag:
        return True
    b = _type_blob(cd)
    if frag in b or frag.replace(" ", "") in b.replace(" ", ""):
        return True
    ct = cd.card_text or ""
    for m in re.finditer(r'[""]{1,2}([^""]{2,100})[""]{1,2}\s*type', ct, flags=re.IGNORECASE):
        if frag in m.group(1).lower().replace(" ", ""):
            return True
    return False


def _pick_from_sequence(
    ctx: EffectContext,
    ids: list[str],
    *,
    type_sub: str | None,
    cost_max: int | None,
    cost_min: int | None,
    exclude: set[str],
    want_char: bool,
    want_event: bool,
    max_add: int,
    sim: "SimplifiedOPSim",
    pl: "PlayerState",
    strict_type: bool,
) -> tuple[list[str], list[str]]:
    """Retourne (cartes prises en main, cartes restantes dans l’ordre d’origine)."""
    taken: list[str] = []
    rest: list[str] = []

    def ok(cid: str, cd: CardDef | None, *, relax_type: bool) -> bool:
        if _excludes_card(cid, cd, exclude):
            return False
        if cd is None:
            return False
        ct = (cd.card_type or "").lower()
        if want_char and ct and "character" not in ct:
            return False
        if want_event and ct and "event" not in ct:
            return False
        if cost_max is not None and cd.cost > cost_max:
            return False
        if cost_min is not None and cd.cost < cost_min:
            return False
        if type_sub and not relax_type and not _type_matches(cd, type_sub):
            return False
        return True

    for cid in ids:
        cap = int(getattr(sim, "max_hand", 0) or 0)
        if len(taken) >= max_add or (cap > 0 and len(pl.hand) + len(taken) >= cap):
            rest.append(cid)
            continue
        cd = sim.cards.get(cid)
        if ok(cid, cd, relax_type=False):
            taken.append(cid)
            continue
        if strict_type and type_sub and ok(cid, cd, relax_type=True):
            taken.append(cid)
            continue
        rest.append(cid)
    return taken, rest


def _apply_look_at_top(ctx: EffectContext, n: int, window: str) -> None:
    """Regarde les N premières cartes : en ajoute jusqu’à max_add à la main, le reste sous le deck."""
    pl = ctx.owner
    sim = ctx.sim
    wl = window.lower()
    max_add = 1
    mm = re.search(r"reveal\s+up\s+to\s+(\d+)", wl)
    if mm:
        max_add = min(7, int(mm.group(1)))
    cost_max = _parse_cost_max(wl)
    cost_min = _parse_cost_min(wl)
    type_sub = _parse_type_quote(window) or _parse_bracket_type(window)
    exclude = _parse_exclude_ids(window)
    want_char = "character card" in wl or "character cards" in wl
    want_event = re.search(r"\bevent\s+cards?\b", wl) is not None and not want_char

    npeek = min(max(0, n), len(pl.deck))
    top = [pl.deck.pop(0) for _ in range(npeek)]
    taken, rest = _pick_from_sequence(
        ctx,
        top,
        type_sub=type_sub,
        cost_max=cost_max,
        cost_min=cost_min,
        exclude=exclude,
        want_char=want_char,
        want_event=want_event,
        max_add=max_add,
        sim=sim,
        pl=pl,
        strict_type=True,
    )
    for cid in taken:
        pl.hand.append(cid)
    pl.deck.extend(rest)
    if "shuffle your deck" in wl:
        _shuffle_deck(ctx)


def _search_deck_add_hand(ctx: EffectContext, max_take: int, window: str) -> None:
    """Cherche dans tout le deck (filtres texte), ajoute à la main, mélange."""
    pl = ctx.owner
    sim = ctx.sim
    wl = window.lower()
    cost_max = _parse_cost_max(wl)
    cost_min = _parse_cost_min(wl)
    type_sub = _parse_type_quote(window) or _parse_bracket_type(window)
    exclude = _parse_exclude_ids(window)
    want_char = "character card" in wl or "character cards" in wl
    want_event = re.search(r"\bevent\s+cards?\b", wl) is not None and not want_char

    order = list(pl.deck)
    taken, rest = _pick_from_sequence(
        ctx,
        order,
        type_sub=type_sub,
        cost_max=cost_max,
        cost_min=cost_min,
        exclude=exclude,
        want_char=want_char,
        want_event=want_event,
        max_add=max_take,
        sim=sim,
        pl=pl,
        strict_type=True,
    )
    pl.deck.clear()
    pl.deck.extend(rest)
    cap = int(getattr(sim, "max_hand", 0) or 0)
    for cid in taken:
        if cap <= 0 or len(pl.hand) < cap:
            pl.hand.append(cid)
    _shuffle_deck(ctx)


def _parse_cost_max(wl: str) -> int | None:
    for pat in (
        r"cost\s+of\s+(\d+)\s+or\s+less",
        r"with\s+a\s+cost\s+of\s+(\d+)\s+or\s+less",
        r"cost\s+(\d+)\s+or\s+less",
    ):
        m = re.search(pat, wl, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _parse_cost_min(wl: str) -> int | None:
    for pat in (
        r"cost\s+of\s+(\d+)\s+or\s+more",
        r"with\s+a\s+cost\s+of\s+(\d+)\s+or\s+more",
    ):
        m = re.search(pat, wl, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def _parse_type_quote(window: str) -> str | None:
    """Types entre guillemets (souvent ``""Straw Hat Crew"" type`` dans tcgcsv)."""
    m = re.search(r'"+([^"]{2,120})"+\s*type', window, re.IGNORECASE)
    return m.group(1) if m else None


def _parse_bracket_type(window: str) -> str | None:
    m = re.search(
        r"reveal\s+up\s+to\s+\d+\s+\[([^\]]{1,80})\]\s*(?:and\s+)?add",
        window,
        re.IGNORECASE,
    )
    if m:
        return m.group(1)
    m2 = re.search(r"\[([^\]]{1,80})\]\s*type", window, re.IGNORECASE)
    return m2.group(1) if m2 else None


def _parse_exclude_ids(window: str) -> set[str]:
    return {x.strip().upper() for x in re.findall(r"other\s+than\s+\[([^\]]+)\]", window, re.IGNORECASE)}


def _excludes_card(cid: str, cd: CardDef | None, exclude: set[str]) -> bool:
    """``other than [Name]`` : compare à l’id et au nom de la carte."""
    cux = cid.strip().upper()
    if cux in exclude:
        return True
    if not cd or not exclude:
        return False
    nm = re.sub(r"[^A-Z0-9]", "", (cd.name or "").upper())
    for ex in exclude:
        e = re.sub(r"[^A-Z0-9]", "", ex)
        if not e:
            continue
        if nm.startswith(e) or (len(e) >= 4 and e in nm):
            return True
    return False


def _trash_deck_top_n(ctx: EffectContext, n: int) -> None:
    pl = ctx.owner
    for _ in range(min(n, len(pl.deck))):
        pl.trash.append(pl.deck.pop(0))


def _add_from_trash_to_hand(ctx: EffectContext, n: int, window: str) -> None:
    wl = window.lower()
    cost_max = _parse_cost_max(wl)
    want_char = "character" in wl and (
        "from your trash" in wl or "from your trash." in wl or "trash to your hand" in wl
    )
    sim = ctx.sim
    pl = ctx.owner
    cap = int(getattr(sim, "max_hand", 0) or 0)
    trash_list = list(pl.trash)
    pl.trash.clear()
    sim.rng.shuffle(trash_list)
    taken = 0
    for cid in trash_list:
        if taken >= n or (cap > 0 and len(pl.hand) >= cap):
            pl.trash.append(cid)
            continue
        cd = sim.cards.get(cid)
        if cd is None:
            pl.trash.append(cid)
            continue
        if cost_max is not None and cd.cost > cost_max:
            pl.trash.append(cid)
            continue
        ct = (cd.card_type or "").lower()
        if want_char and ct and "character" not in ct:
            pl.trash.append(cid)
            continue
        pl.hand.append(cid)
        taken += 1


def _place_n_from_hand_bottom(ctx: EffectContext, n: int) -> None:
    pl = ctx.owner
    sim = ctx.sim
    for _ in range(min(n, len(pl.hand))):
        if not pl.hand:
            break
        i = int(sim.rng.integers(0, len(pl.hand)))
        cid = pl.hand.pop(i)
        pl.deck.append(cid)


def _give_rested_don_to_board(ctx: EffectContext, n: int) -> None:
    """Attache n DON!! reposés à un perso (priorité : la carte source de l'effet [On Play], sinon board[0])."""
    pl = ctx.owner
    if not pl.board or pl.don_rested <= 0:
        return
    target_i = 0
    src_id = getattr(ctx.card, "card_id", None) or ""
    if src_id:
        for i, bc in enumerate(pl.board):
            if bc.card_id == src_id:
                target_i = i
                break
    for _ in range(min(n, pl.don_rested)):
        if pl.don_rested <= 0:
            break
        pl.don_rested -= 1
        pl.board[target_i].attached_don += 1


def _give_rested_don_to_leader(ctx: EffectContext, n: int) -> None:
    """
    Nami character EB03-053 : attache n DON!! reposés directement au Leader.
    (CR : les DON!! reposés passent de la zone reposée → attachés au Leader)
    """
    pl = ctx.owner
    if pl.don_rested <= 0:
        return
    give = min(n, pl.don_rested)
    pl.don_rested -= give
    pl.leader_attached_don += give


def _opp_life_to_opp_hand(ctx: EffectContext) -> None:
    """
    Prend 1 carte du DESSUS de la pile vie de l'adversaire et la met dans SA main.
    Non-combat : pas de Trigger possible.  Déclenche l'auto-draw du Leader si applicable.
    Était conditionnel (min_opp_life) ; la condition est désormais gérée par
    _preprocess_conditions() au niveau du texte.
    """
    opp = ctx.sim.p1 if ctx.owner_idx == 0 else ctx.sim.p0
    if opp.life_cards:
        card = opp.life_cards.pop(0)
        opp.hand.append(card)
        ctx.sim._trigger_leader_auto_draw()


# ── Moteur d'évaluation de conditions génériques ─────────────────────────────
# Permet de lire « if you have N or less cards in your hand » directement
# depuis le texte de la carte, sans hardcoder chaque cas.

def _opponent_state(ctx: EffectContext) -> "PlayerState":
    return ctx.sim.p1 if ctx.owner_idx == 0 else ctx.sim.p0


def _leader_has_type_check(ctx: EffectContext, type_str: str) -> bool:
    """Vérifie si le Leader du joueur possède le type/sous-type indiqué."""
    leader_cd = ctx.sim.cards.get(ctx.owner.leader_id or "")
    if not leader_cd:
        return False
    blob = " ".join([
        leader_cd.card_text or "", leader_cd.name or "", leader_cd.card_type or "",
    ]).lower()
    return type_str.strip().lower() in blob


def _player_turn_number(ctx: EffectContext) -> int:
    """1 = premier tour de ce joueur, 2 = deuxième, … (aligné sur ``SimplifiedOPSim.player_turn_number``)."""
    return int(ctx.sim.player_turn_number(ctx.owner))


# (regex, évaluateur) → True = condition remplie
_COND_RULES: list[tuple[Pattern, Callable[[EffectContext, re.Match], bool]]] = [
    # 2e tour du joueur (ex. Enel OP15-058 [Activate: Main])
    (
        re.compile(
            r"if\s+(?:it\s+is|it''s)\s+your\s+second\s+turn\s+or\s+later",
            re.IGNORECASE,
        ),
        lambda ctx, m: _player_turn_number(ctx) >= 2,
    ),
    # Main ≤ N
    (
        re.compile(r"if\s+you\s+have\s+(\d+)\s+or\s+less\s+cards?\s+in\s+your\s+hand", re.IGNORECASE),
        lambda ctx, m: len(ctx.owner.hand) <= int(m.group(1)),
    ),
    # Main ≥ N
    (
        re.compile(r"if\s+you\s+have\s+(\d+)\s+or\s+more\s+cards?\s+in\s+your\s+hand", re.IGNORECASE),
        lambda ctx, m: len(ctx.owner.hand) >= int(m.group(1)),
    ),
    # Adversaire vie ≤ N
    (
        re.compile(r"if\s+your\s+opponent\s+has\s+(\d+)\s+or\s+less\s+Life\s+cards?", re.IGNORECASE),
        lambda ctx, m: len(_opponent_state(ctx).life_cards) <= int(m.group(1)),
    ),
    # Adversaire vie ≥ N
    (
        re.compile(r"if\s+your\s+opponent\s+has\s+(\d+)\s+or\s+more\s+Life\s+cards?", re.IGNORECASE),
        lambda ctx, m: len(_opponent_state(ctx).life_cards) >= int(m.group(1)),
    ),
    # Moi vie ≤ N
    (
        re.compile(r"if\s+you\s+have\s+(\d+)\s+or\s+less\s+Life\s+cards?", re.IGNORECASE),
        lambda ctx, m: len(ctx.owner.life_cards) <= int(m.group(1)),
    ),
    # Moi vie ≥ N
    (
        re.compile(r"if\s+you\s+have\s+(\d+)\s+or\s+more\s+Life\s+cards?", re.IGNORECASE),
        lambda ctx, m: len(ctx.owner.life_cards) >= int(m.group(1)),
    ),
    # Leader a le type X  (e.g., {Straw Hat Crew} type)
    (
        re.compile(
            r"if\s+your\s+(?:leader|Leader)\s+has\s+the\s+[{\"\[]?([\w][\w\s\-]*?)[\}\"\]]?\s+type",
            re.IGNORECASE,
        ),
        lambda ctx, m: _leader_has_type_check(ctx, m.group(1)),
    ),
    # Perso board ≥ N
    (
        re.compile(
            r"if\s+you\s+have\s+(\d+)\s+or\s+more\s+(?:character|Character)s?"
            r"\s+(?:in\s+play|on\s+(?:your\s+)?(?:field|area))",
            re.IGNORECASE,
        ),
        lambda ctx, m: len(ctx.owner.board) >= int(m.group(1)),
    ),
    # Perso board ≤ N
    (
        re.compile(
            r"if\s+you\s+have\s+(\d+)\s+or\s+less\s+(?:character|Character)s?"
            r"\s+(?:in\s+play|on\s+(?:your\s+)?(?:field|area))",
            re.IGNORECASE,
        ),
        lambda ctx, m: len(ctx.owner.board) <= int(m.group(1)),
    ),
    # DON!! actifs ≥ N
    (
        re.compile(r"if\s+you\s+have\s+(\d+)\s+or\s+more\s+(?:active\s+)?DON!!", re.IGNORECASE),
        lambda ctx, m: ctx.owner.don_active >= int(m.group(1)),
    ),
    # Adversaire perso board ≤ N
    (
        re.compile(
            r"if\s+your\s+opponent\s+has\s+(\d+)\s+or\s+less\s+(?:character|Character)s?"
            r"\s+(?:in\s+play|on\s+(?:their\s+)?(?:field|area))",
            re.IGNORECASE,
        ),
        lambda ctx, m: len(_opponent_state(ctx).board) <= int(m.group(1)),
    ),
    # Adversaire perso board ≥ N
    (
        re.compile(
            r"if\s+your\s+opponent\s+has\s+(\d+)\s+or\s+more\s+(?:character|Character)s?"
            r"\s+(?:in\s+play|on\s+(?:their\s+)?(?:field|area))",
            re.IGNORECASE,
        ),
        lambda ctx, m: len(_opponent_state(ctx).board) >= int(m.group(1)),
    ),
]


def evaluate_conditions(ctx: EffectContext, text: str) -> bool:
    """
    Évalue toutes les conditions « if … » présentes dans *text*.
    Retourne True si TOUTES sont remplies (ou si aucune condition trouvée).
    """
    t = _norm(text)
    for pattern, evaluator in _COND_RULES:
        m = pattern.search(t)
        if m and not evaluator(ctx, m):
            return False
    return True


# Regex pour découper les clauses conditionnelles « if [cond], [effet][.] »
# Capture optionnellement le point final pour ne pas laisser de résidus.
_IF_CLAUSE_RX = re.compile(
    r"\bif\b\s+([^,.:;\n]{3,250}?)(?:,|:)\s*([^.;\n]+)\.?",
    re.IGNORECASE,
)


def _preprocess_conditions(ctx: EffectContext, work: str) -> str:
    """
    Parcourt le texte à la recherche de « if [condition], [effet] ».
    - Condition vraie  → remplace par l'effet seul.
    - Condition fausse → efface la clause entière (y compris le point terminal).
    Répète jusqu'à stabilisation pour gérer les conditions imbriquées.
    """
    def _replace(m: re.Match) -> str:
        cond_text = "if " + m.group(1)
        effect_text = m.group(2).strip()
        return effect_text if evaluate_conditions(ctx, cond_text) else " "

    max_if_passes = min(32, max(6, len(work) // 500 + 6))
    for _ in range(max_if_passes):
        new_work = _IF_CLAUSE_RX.sub(_replace, work)
        if new_work == work:
            break
        work = new_work
    return work


RuleFn = Callable[[EffectContext, re.Match], None]


def _rule_look_add_bottom(ctx: EffectContext, m: re.Match) -> None:
    n = int(m.group(1))
    _apply_look_at_top(ctx, n, m.group(0))


def _rule_search_shuffle(ctx: EffectContext, m: re.Match) -> None:
    n = int(m.group(1))
    _search_deck_add_hand(ctx, n, m.group(0))


def _rule_search_deck_only(ctx: EffectContext, m: re.Match) -> None:
    n = int(m.group(1))
    w = m.group(0) + " shuffle your deck"
    _search_deck_add_hand(ctx, n, w)


def _rule_trash_deck_top(ctx: EffectContext, m: re.Match) -> None:
    _trash_deck_top_n(ctx, int(m.group(1)))


def _rule_add_from_trash(ctx: EffectContext, m: re.Match) -> None:
    n = int(m.group(1))
    _add_from_trash_to_hand(ctx, n, m.group(0))


def _rule_draw_trash_hand(ctx: EffectContext, m: re.Match) -> None:
    d = int(m.group(1))
    t = int(m.group(2))
    _draw_n(ctx, d)
    _discard_n_random(ctx, ctx.owner, t)


def _rule_draw_trash_one_hand(ctx: EffectContext, m: re.Match) -> None:
    d = int(m.group(1))
    _draw_n(ctx, d)
    _discard_n_random(ctx, ctx.owner, 1)


def _rule_draw_place_hand_bottom(ctx: EffectContext, m: re.Match) -> None:
    d = int(m.group(1))
    b = int(m.group(2))
    _draw_n(ctx, d)
    _place_n_from_hand_bottom(ctx, b)


def _rule_place_hand_bottom(ctx: EffectContext, m: re.Match) -> None:
    _place_n_from_hand_bottom(ctx, int(m.group(1)))


def _rule_give_rested_don_leader(ctx: EffectContext, m: re.Match) -> None:
    """Give up to N rested DON!! to your Leader (Nami char EB03-053)."""
    _give_rested_don_to_leader(ctx, int(m.group(1)))


def _rule_give_rested_don(ctx: EffectContext, m: re.Match) -> None:
    _give_rested_don_to_board(ctx, int(m.group(1)))


def _rule_opp_life_to_opp_hand(ctx: EffectContext, m: re.Match) -> None:
    """Top de la pile vie de l'adversaire → sa main (condition déjà évaluée par _preprocess_conditions)."""
    _opp_life_to_opp_hand(ctx)


def _rule_play_look_bottom(ctx: EffectContext, m: re.Match) -> None:
    """Look at N puis « play up to 1 … rested » : approximation = ajout main comme look."""
    n = int(m.group(1))
    _apply_look_at_top(ctx, n, m.group(0))


def _rule_draw_then_discard(ctx: EffectContext, m: re.Match) -> None:
    d = int(m.group(1))
    x = int(m.group(2))
    _draw_n(ctx, d)
    _discard_n_random(ctx, ctx.owner, x)


def _rule_discard_then_draw(ctx: EffectContext, m: re.Match) -> None:
    x = int(m.group(1))
    d = int(m.group(2))
    _discard_n_random(ctx, ctx.owner, x)
    _draw_n(ctx, d)


def _rule_discard_a_draw_a(ctx: EffectContext, m: re.Match) -> None:
    _discard_n_random(ctx, ctx.owner, 1)
    _draw_n(ctx, 1)


def _rule_opponent_trash_one(ctx: EffectContext, m: re.Match) -> None:
    _discard_n_random(ctx, _opponent(ctx), 1)


def _rule_draw_a_and_discard_a(ctx: EffectContext, m: re.Match) -> None:
    _draw_n(ctx, 1)
    _discard_n_random(ctx, ctx.owner, 1)


def _rule_draw_n(ctx: EffectContext, m: re.Match) -> None:
    _draw_n(ctx, int(m.group(1)))


def _rule_draw_a_card(ctx: EffectContext, m: re.Match) -> None:
    _draw_n(ctx, 1)


def _rule_discard_n(ctx: EffectContext, m: re.Match) -> None:
    _discard_n_random(ctx, ctx.owner, int(m.group(1)))


def _rule_discard_a_card(ctx: EffectContext, m: re.Match) -> None:
    _discard_n_random(ctx, ctx.owner, 1)


def _rule_opponent_trash_n(ctx: EffectContext, m: re.Match) -> None:
    opp = _opponent(ctx)
    _discard_n_random(ctx, opp, int(m.group(1)))


def _rule_add_rested_don(ctx: EffectContext, m: re.Match) -> None:
    _add_rested_don_from_deck(ctx, int(m.group(1)))


def _rule_add_active_don_from_deck(ctx: EffectContext, m: re.Match) -> None:
    _add_active_don_from_deck(ctx, int(m.group(1)))


def _rule_gain_life(ctx: EffectContext, m: re.Match) -> None:
    _gain_life(ctx, int(m.group(1)))


def _rule_trash_life_then_gain(ctx: EffectContext, m: re.Match) -> None:
    """Robin EB03-055 : trash N du dessus de la pile vie, puis ajouter M du deck au sommet."""
    n_trash = int(m.group(1))
    n_gain = int(m.group(2))
    if ctx.owner.life_cards:
        _trash_life_top(ctx, n_trash)
        _gain_life(ctx, n_gain)


def _rule_add_deck_to_life(ctx: EffectContext, m: re.Match) -> None:
    """Ajoute N cartes du sommet du deck au sommet de la pile vie (Borsalino, Kikunojo, …)."""
    _gain_life(ctx, int(m.group(1)))


def _rule_trash_opponent_life_n(ctx: EffectContext, m: re.Match) -> None:
    """
    Kaido ST04-001 [Activate: Main] DON!! -7 :
    'Trash up to N of your opponent's Life cards.'
    La carte est envoyée au trash adverse (sans déclencher de Trigger).
    """
    n = int(m.group(1)) if m.lastindex and m.group(1) else 1
    opp = _opponent_state(ctx)
    for _ in range(min(n, len(opp.life_cards))):
        opp.trash.append(opp.life_cards.pop(0))


def _rule_deal_damage_to_opponent(ctx: EffectContext, m: re.Match) -> None:
    """Robin [On K.O.] : deal 1 damage to opponent → _take_life_damage de l'adversaire."""
    n = int(m.group(1)) if m.lastindex and m.group(1) else 1
    opp = ctx.sim.p1 if ctx.owner_idx == 0 else ctx.sim.p0
    for _ in range(n):
        if ctx.sim.done:
            break
        ctx.sim._take_life_damage(opp)


def _rule_cond_gain_life_if_opp_low(ctx: EffectContext, m: re.Match) -> None:
    """Kikunojo [On K.O.] : gain 1 vie si l'adversaire a ≤ threshold vie."""
    threshold = int(m.group(1))
    n_gain = int(m.group(2)) if m.lastindex and m.lastindex >= 2 and m.group(2) else 1
    opp = ctx.sim.p1 if ctx.owner_idx == 0 else ctx.sim.p0
    if len(opp.life_cards) <= threshold:
        _gain_life(ctx, n_gain)


def _rule_play_char_from_trash(ctx: EffectContext, m: re.Match) -> None:
    """Dr. Hogback [On K.O.] : joue 1 personnage (coût ≤ N) depuis le trash sur le board."""
    cost_max = int(m.group(1))
    sim = ctx.sim
    pl = ctx.owner
    exclude_name = m.group(0).lower()
    for i, cid in enumerate(pl.trash):
        cd = sim.cards.get(cid)
        if cd is None:
            continue
        ct = (cd.card_type or "").lower()
        if "character" not in ct:
            continue
        if cd.cost > cost_max:
            continue
        if cd.name and cd.name.lower() in exclude_name:
            continue
        pl.trash.pop(i)
        # Utilise _play_char_to_board pour gérer le remplacement si board plein
        # joué depuis trash → reposé (CR 3-4-1)
        from ..simulator import BoardChar
        bc = BoardChar(
            card_id=cid, power=cd.power, rested=True,
            has_rush=sim._has_rush(cd),
            has_rush_char=sim._has_rush_char(cd),
            has_blocker=sim._has_blocker(cd),
            has_double_attack=sim._has_double_attack(cd),
            has_unblockable=sim._has_unblockable(cd),
            has_attack_active=sim._has_attack_active_chars(cd),
            just_played=True,
            restrictions=bootstrap_printed_restrictions_from_card(cd),
        )
        if len(pl.board) >= sim.max_board:
            weakest_idx = min(
                range(len(pl.board)),
                key=lambda i: effective_power(pl.board[i], sim.cards),
            )
            replaced = pl.board.pop(weakest_idx)
            if replaced.attached_don:
                pl.don_rested += replaced.attached_don
            pl.trash.append(replaced.card_id)
        pl.board.append(bc)
        break


def _rule_lock_opponent_char(ctx: EffectContext, m: re.Match) -> None:
    """Perona [On K.O.] : un perso adversaire (coût ≤ N) ne peut pas attaquer ce tour."""
    cost_max = int(m.group(1))
    opp = ctx.sim.p1 if ctx.owner_idx == 0 else ctx.sim.p0
    for b in opp.board:
        cd = ctx.sim.cards.get(b.card_id)
        if cd and cd.cost <= cost_max:
            b.restrictions.cannot_attack_this_turn = True
            break


def _rule_shuffle_deck(ctx: EffectContext, m: re.Match) -> None:
    _shuffle_deck(ctx)


# (pattern, handler) — ordre : composés puis atomiques (les plus longs d’abord)
_RULES: list[tuple[Pattern, RuleFn]] = [
    (
        re.compile(
            r"look\s+at\s+(\d+)\s+cards?\s+from\s+the\s+top\s+of\s+your\s+deck"
            r".*?play\s+up\s+to"
            r".*?place\s+the\s+rest\s+at\s+the\s+bottom\s+of\s+your\s+deck(?:\s+in\s+any\s+order)?",
            re.IGNORECASE | re.DOTALL,
        ),
        _rule_play_look_bottom,
    ),
    (
        re.compile(
            r"look\s+at\s+(\d+)\s+cards?\s+from\s+the\s+top\s+of\s+your\s+deck"
            r".*?add\s+(?:it|the\s+revealed\s+card|up\s+to\s+\d+\s+cards?)\s+to\s+your\s+hand"
            r".*?place\s+the\s+rest\s+at\s+the\s+bottom\s+of\s+your\s+deck(?:\s+in\s+any\s+order)?",
            re.IGNORECASE | re.DOTALL,
        ),
        _rule_look_add_bottom,
    ),
    (
        re.compile(
            r"search\s+your\s+deck\s+for\s+up\s+to\s+(\d+)"
            r"[\s\S]{0,450}?"
            r"shuffle\s+your\s+deck",
            re.IGNORECASE | re.DOTALL,
        ),
        _rule_search_shuffle,
    ),
    (
        re.compile(
            r"search\s+your\s+deck\s+for\s+up\s+to\s+(\d+)\b",
            re.IGNORECASE,
        ),
        _rule_search_deck_only,
    ),
    (
        re.compile(
            r"draw\s+(\d+)\s+cards?\s+and\s+trash\s+(\d+)\s+cards?\s+from\s+your\s+hand",
            re.IGNORECASE,
        ),
        _rule_draw_trash_hand,
    ),
    (
        re.compile(
            r"draw\s+(\d+)\s+cards?\s+and\s+trash\s+(?:a|1|one)\s+card\s+from\s+your\s+hand",
            re.IGNORECASE,
        ),
        _rule_draw_trash_one_hand,
    ),
    (
        re.compile(
            r"draw\s+(\d+)\s+cards?\s+and\s+place\s+(\d+)\s+cards?\s+from\s+your\s+hand\s+at\s+the\s+bottom",
            re.IGNORECASE,
        ),
        _rule_draw_place_hand_bottom,
    ),
    (
        re.compile(
            r"add\s+up\s+to\s+(\d+)\s+[^.;]{0,200}?from\s+your\s+trash(?:\s+to\s+your\s+hand)?",
            re.IGNORECASE,
        ),
        _rule_add_from_trash,
    ),
    (
        re.compile(
            r"trash\s+(\d+)\s+cards?\s+from\s+the\s+top\s+of\s+your\s+deck",
            re.IGNORECASE,
        ),
        _rule_trash_deck_top,
    ),
    (
        re.compile(
            r"give\s+up\s+to\s+(\d+)\s+rested\s+don(?:!!)?\s+card\s+to\s+your\s+leader",
            re.IGNORECASE,
        ),
        _rule_give_rested_don_leader,
    ),
    (
        re.compile(
            r"give\s+up\s+to\s+(\d+)\s+rested\s+don(?:!!)?(?:\s+cards?)?\s+to\s+1\s+of\s+your\s+characters?",
            re.IGNORECASE,
        ),
        _rule_give_rested_don,
    ),
    (
        re.compile(
            r"give\s+up\s+to\s+(\d+)\s+rested\s+don(?:!!)?(?:\s+cards?)?",
            re.IGNORECASE,
        ),
        _rule_give_rested_don,
    ),
    (
        # Nami char EB03-053 (condition déjà résolue par _preprocess_conditions)
        re.compile(
            r"add\s+up\s+to\s+\d+\s+cards?\s+from\s+the\s+top\s+of\s+your\s+opponent.?s\s+Life\s+cards?"
            r".*?to\s+(?:the\s+owner.?s\s+hand|your\s+opponent.?s\s+hand|their\s+hand)",
            re.IGNORECASE | re.DOTALL,
        ),
        _rule_opp_life_to_opp_hand,
    ),
    (
        re.compile(
            r"place\s+(\d+)\s+cards?\s+from\s+your\s+hand\s+at\s+the\s+bottom",
            re.IGNORECASE,
        ),
        _rule_place_hand_bottom,
    ),
    (
        re.compile(
            r"draw\s+(\d+)\s+cards?\s*(?:,|;|\.|\s+then\s+|\s+and\s+)\s*discard\s+(\d+)\s+cards?",
            re.IGNORECASE,
        ),
        _rule_draw_then_discard,
    ),
    (
        re.compile(
            r"draw\s+(\d+)\s+cards?\s*(?:,|;|\.|\s+then\s+|\s+and\s+)\s*discard\s+(\d+)\s+card\b",
            re.IGNORECASE,
        ),
        _rule_draw_then_discard,
    ),
    (
        re.compile(
            r"draw\s+a\s+card\s*(?:,|;|\.|\s+then\s+|\s+and\s+)\s*discard\s+(?:a|1|one)\s+card",
            re.IGNORECASE,
        ),
        _rule_draw_a_and_discard_a,
    ),
    (
        re.compile(
            r"draw\s+1\s+card\s*(?:,|;|\.|\s+then\s+|\s+and\s+)\s*discard\s+1\s+card",
            re.IGNORECASE,
        ),
        _rule_draw_a_and_discard_a,
    ),
    (
        re.compile(
            r"discard\s+(\d+)\s+cards?\s*(?:,|;|\.|\s+then\s+|\s+and\s+)\s*draw\s+(\d+)\s+cards?",
            re.IGNORECASE,
        ),
        _rule_discard_then_draw,
    ),
    (
        re.compile(
            r"discard\s+(?:a|1|one)\s+card\s*(?:,|;|\.|\s+then\s+|\s+and\s+)\s*draw\s+(?:a|1|one)\s+card",
            re.IGNORECASE,
        ),
        _rule_discard_a_draw_a,
    ),
    (
        re.compile(
            r"discard\s+(\d+)\s+cards?\s*(?:,|;|\.|\s+then\s+|\s+and\s+)\s*draw\s+(\d+)\s+card\b",
            re.IGNORECASE,
        ),
        _rule_discard_then_draw,
    ),
    (
        re.compile(
            r"your\s+opponent\s+trashes\s+(\d+)\s+cards?",
            re.IGNORECASE,
        ),
        _rule_opponent_trash_n,
    ),
    (
        re.compile(
            r"your\s+opponent\s+trashes\s+(?:a|1|one)\s+card",
            re.IGNORECASE,
        ),
        _rule_opponent_trash_one,
    ),
    (
        re.compile(
            r"add\s+up\s+to\s+(\d+)\s+don(?:!!)?\s+cards?\s+from\s+your\s+don(?:!!)?\s+deck\s+"
            r"and\s+set\s+(?:it|them)\s+as\s+active",
            re.IGNORECASE,
        ),
        _rule_add_active_don_from_deck,
    ),
    (
        re.compile(
            r"add\s+up\s+to\s+(\d+)\s+additional\s+don(?:!!)?\s+cards?\s+and\s+rest\s+them",
            re.IGNORECASE,
        ),
        _rule_add_rested_don,
    ),
    (
        re.compile(
            r"add\s+up\s+to\s+(\d+)\s+rested\s+don(?:!!)?",
            re.IGNORECASE,
        ),
        _rule_add_rested_don,
    ),
    # Note: l'ancienne règle conditionnelle _rule_cond_gain_life_if_opp_low est supprimée.
    # La condition « if opponent has N or less Life cards » est désormais évaluée
    # dynamiquement par _preprocess_conditions(), puis _rule_add_deck_to_life s'applique.
    (
        re.compile(
            r"up\s+to\s+1\s+of\s+your\s+opponent.?s\s+characters?\s+with\s+a\s+cost\s+of\s+(\d+)\s+or\s+less"
            r".*?cannot\s+attack",
            re.IGNORECASE | re.DOTALL,
        ),
        _rule_lock_opponent_char,
    ),
    (
        re.compile(
            r"play\s+up\s+to\s+1\s+character\s+card"
            r".*?cost\s+of\s+(\d+)\s+or\s+less"
            r".*?from\s+your\s+trash",
            re.IGNORECASE | re.DOTALL,
        ),
        _rule_play_char_from_trash,
    ),
    (
        re.compile(
            r"deal\s+(\d+)\s+damage\s+to\s+your\s+opponent",
            re.IGNORECASE,
        ),
        _rule_deal_damage_to_opponent,
    ),
    (
        re.compile(
            r"deal\s+1\s+damage\s+to\s+your\s+opponent",
            re.IGNORECASE,
        ),
        _rule_deal_damage_to_opponent,
    ),
    (
        re.compile(
            r"trash\s+up\s+to\s+(\d+)\s+of\s+your\s+opponent.?s\s+Life\s+cards?",
            re.IGNORECASE,
        ),
        _rule_trash_opponent_life_n,
    ),
    (
        re.compile(
            r"trash\s+(\d+)\s+cards?\s+from\s+the\s+top\s+of\s+your\s+Life\s+cards?"
            r".*?add\s+up\s+to\s+(\d+)\s+cards?\s+from\s+the\s+top\s+of\s+your\s+deck"
            r".*?to\s+the\s+top\s+of\s+your\s+Life\s+cards?",
            re.IGNORECASE | re.DOTALL,
        ),
        _rule_trash_life_then_gain,
    ),
    (
        re.compile(
            r"add\s+up\s+to\s+(\d+)\s+cards?\s+from\s+the\s+top\s+of\s+your\s+deck"
            r".*?to\s+the\s+top\s+of\s+your\s+Life\s+cards?",
            re.IGNORECASE | re.DOTALL,
        ),
        _rule_add_deck_to_life,
    ),
    (
        re.compile(
            r"gain\s+(\d+)\s+life",
            re.IGNORECASE,
        ),
        _rule_gain_life,
    ),
    (
        re.compile(
            r"add\s+(\d+)\s+life",
            re.IGNORECASE,
        ),
        _rule_gain_life,
    ),
    (
        re.compile(
            r"shuffle\s+your\s+deck",
            re.IGNORECASE,
        ),
        _rule_shuffle_deck,
    ),
    (
        re.compile(
            r"draw\s+up\s+to\s+(\d+)\s+cards?",
            re.IGNORECASE,
        ),
        _rule_draw_n,
    ),
    (
        re.compile(
            r"draw\s+(\d+)\s+cards?",
            re.IGNORECASE,
        ),
        _rule_draw_n,
    ),
    (
        re.compile(
            r"draw\s+(?:a|1|one)\s+card",
            re.IGNORECASE,
        ),
        _rule_draw_a_card,
    ),
    (
        re.compile(
            r"discard\s+(\d+)\s+cards?",
            re.IGNORECASE,
        ),
        _rule_discard_n,
    ),
    (
        re.compile(
            r"discard\s+(?:a|1|one)\s+card",
            re.IGNORECASE,
        ),
        _rule_discard_a_card,
    ),
]


def _preprocess_second_turn_if_clause(ctx: EffectContext, work: str) -> str:
    """
    « If it is your second turn or later, » (Enel OP15-058) : effet multi-phrases.
    Le découpeur générique ``_IF_CLAUSE_RX`` s’arrête au 1er « . » et laissait « Then, give… »
    actif au mauvais tour — on retire ou on ne garde que le corps d’effet ici.
    """
    if "second turn or later" not in work.lower():
        return work
    if _player_turn_number(ctx) < 2:
        return re.sub(
            r"if\s+(?:it\s+is|it''s)\s+your\s+second\s+turn\s+or\s+later\s*,[\s\S]*?characters?\s*\.?",
            " ",
            work,
            flags=re.IGNORECASE,
        )
    return re.sub(
        r"if\s+(?:it\s+is|it''s)\s+your\s+second\s+turn\s+or\s+later\s*,\s*",
        " ",
        work,
        count=1,
        flags=re.IGNORECASE,
    )


def _apply_rules_to_text(ctx: EffectContext, text: str) -> None:
    if not text.strip():
        return
    work = _norm(text)
    # "You may" → l'IA accepte toujours les effets optionnels
    work = re.sub(r"\byou\s+may\s+", "", work, flags=re.IGNORECASE)
    work = _preprocess_second_turn_if_clause(ctx, work)
    # Évaluer et résoudre les clauses « if [condition], [effet] » depuis le texte
    work = _preprocess_conditions(ctx, work)
    # Texte long = plus de motifs disjoints à retirer successivement.
    max_rule_passes = min(512, max(40, len(work) // 10 + 32))
    for _ in range(max_rule_passes):
        matched = False
        for rx, fn in _RULES:
            m = rx.search(work)
            if not m:
                continue
            fn(ctx, m)
            work = work[: m.start()] + " " + work[m.end() :]
            matched = True
            break
        if not matched:
            break


def apply_on_play_effects(ctx: EffectContext) -> None:
    """
    Applique les effets reconnus dans la clause [On Play] ; si elle est absente,
    analyse tout le ``card_text`` (comportement hérité pour textes sans marqueur).
    """
    segment = extract_on_play_segment(ctx.card.card_text or "")
    full = ctx.card.card_text or ""
    blob = segment if segment else _clip_timing_segment_body(full)
    _apply_rules_to_text(ctx, blob)


def apply_main_text_effects(ctx: EffectContext) -> None:
    """Effets [Main] (Events, etc.) — même moteur de motifs que [On Play]."""
    full = ctx.card.card_text or ""
    segment = extract_timing_segment(full, "main")
    blob = segment if segment else _clip_timing_segment_body(full)
    _apply_rules_to_text(ctx, blob)


def apply_activate_main_effects(ctx: EffectContext) -> None:
    """
    Effets [Activate: Main] (ex. Kaido ST04-001).
    Extrait le segment d'effet après le coût DON!! -N :
      '[Activate: Main] [Once Per Turn] DON!! -N (...) : <effet>'
    """
    text = ctx.card.card_text or ""
    t = _norm(text)
    m = re.search(r"\[activate\s*:\s*main\]", t, re.IGNORECASE)
    if not m:
        return
    rest = t[m.end():]
    # Chercher le ':' qui sépare le coût de l'effet (après DON!! -N ou [Once Per Turn])
    colon = re.search(r"\)\s*:", rest)
    if colon:
        effect_text = rest[colon.end():].strip()
    else:
        colon2 = re.search(r"(?:don!!\s*-\s*\d+|once\s+per\s+turn)[^:]*:", rest, re.IGNORECASE)
        if colon2:
            effect_text = rest[colon2.end():].strip()
        else:
            # Pas de coût explicite — tout le reste est l'effet
            effect_text = rest.strip()
    effect_text = _clip_timing_segment_body(effect_text.strip())
    if effect_text:
        _apply_rules_to_text(ctx, effect_text)


def apply_on_ko_effects(ctx: EffectContext) -> None:
    """
    Applique les effets ``[On K.O.]`` de la carte lorsqu'elle est mise au K.O.
    Cherche d'abord la clause ``[On K.O.]``; si absent, n'applique rien.
    Pour Robin : le qualificatif ``[Opponent's Turn]`` est ignoré (le K.O. se
    produit toujours pendant le tour de l'adversaire dans ce simulateur).
    """
    text = ctx.card.card_text or ""
    segment = extract_on_ko_segment(text)
    if segment:
        _apply_rules_to_text(ctx, segment)
