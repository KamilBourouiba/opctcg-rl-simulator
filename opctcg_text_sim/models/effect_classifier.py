"""
EffectClassifier — Modèle 3 : classification exhaustive des effets.

Pour chaque segment textuel fourni par TimingModel, le classifieur :
  1. Détecte le(s) type(s) d'effets présents
  2. Extrait les paramètres spécifiques (n, amount, source…)
  3. Extrait la cible structurée (owner, type, cost filtre…)
  4. Extrait la condition "if …"
  5. Extrait la durée et le caractère optionnel
  6. Détecte les coûts d'activation "You may [cost] : [effect]"
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .effect_models import EffectType, ParsedCard, ParsedEffect, Target

# Ré-import explicite pour le nouveau type
_SET_ACTIVE = EffectType.SET_ACTIVE


# ══════════════════════════════════════════════════════════════════════════
# Helpers d'extraction globaux
# ══════════════════════════════════════════════════════════════════════════

def _int(s: str | None, default: int = 1) -> int:
    if s is None:
        return default
    try:
        return int(s.replace(",", ""))
    except (ValueError, TypeError):
        return default


_WORD_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def _word_to_int(w: str | None) -> int:
    if w is None:
        return 1
    return _WORD_NUM.get(w.lower(), _int(w))


def _norm(s: str) -> str:
    return (
        str(s)
        .replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
    )


# ── Cible ─────────────────────────────────────────────────────────────────

_RE_COUNT    = re.compile(r"up\s+to\s+(\d+)\s+of\b", re.IGNORECASE)
_RE_TYPE_TAG = re.compile(r'\[([^\]\n]{1,50})\]\s+type', re.IGNORECASE)
_RE_COST_LEQ = re.compile(r'cost\s+of\s+(\d+)\s+or\s+less', re.IGNORECASE)
_RE_COST_GEQ = re.compile(r'cost\s+of\s+(\d+)\s+or\s+more', re.IGNORECASE)
_RE_POWER_LEQ= re.compile(r'(?:power|Power)\s+of\s+(\d[\d,]*)\s+or\s+less', re.IGNORECASE)
_RE_POWER_GEQ= re.compile(r'(?:power|Power)\s+of\s+(\d[\d,]*)\s+or\s+more', re.IGNORECASE)
_RE_COLOR    = re.compile(r'\b(Red|Blue|Green|Yellow|Purple|Black)\b\s+(?:type\s+)?(?:card|Character|Leader)', re.IGNORECASE)
_RE_NAME     = re.compile(r'with\s+"([^"]{1,50})"\s+in\s+(?:its\s+)?name', re.IGNORECASE)


def _extract_target(text: str, hint_owner: str = "") -> Target:
    t = Target()

    # ── Owner ──────────────────────────────────────────────────────────────
    opp = bool(re.search(r"your\s+opponent'?s?", text, re.IGNORECASE))
    self_ref = bool(re.search(r"\bthis\s+(?:Leader|Character|card|Stage)\b", text, re.IGNORECASE))
    if hint_owner:
        t.owner = hint_owner
    elif opp:
        t.owner = "opponent"
    elif self_ref:
        t.owner = "self"
    else:
        t.owner = "you"

    # ── Card type ──────────────────────────────────────────────────────────
    if re.search(r"\bLeader\b\s+or\s+(?:\d+\s+of\s+)?(?:your\s+)?(?:\d+\s+)?Characters?", text, re.IGNORECASE):
        t.card_type = "Leader or Character"
    elif re.search(r"\bLeader\b", text, re.IGNORECASE):
        t.card_type = "Leader"
    elif re.search(r"\bCharacters?\b", text, re.IGNORECASE):
        t.card_type = "Character"
    elif re.search(r"\bEvents?\b", text, re.IGNORECASE):
        t.card_type = "Event"
    elif re.search(r"\bStages?\b", text, re.IGNORECASE):
        t.card_type = "Stage"

    # ── Count ──────────────────────────────────────────────────────────────
    m_count = _RE_COUNT.search(text)
    if m_count:
        t.count = int(m_count.group(1))
        t.up_to = True
    elif re.search(r"\ball\b", text, re.IGNORECASE):
        t.count = 0  # 0 = "all"
        t.up_to = False
    else:
        # Compte explicite sans "up to"
        m_exact = re.search(r"\b(\d+)\s+of\s+(?:your|your\s+opponent)", text, re.IGNORECASE)
        t.count  = int(m_exact.group(1)) if m_exact else 1
        t.up_to  = False

    # ── Filters ───────────────────────────────────────────────────────────
    m_tag = _RE_TYPE_TAG.search(text)
    if m_tag:
        t.type_tag = m_tag.group(1)

    m_cmax = _RE_COST_LEQ.search(text)
    if m_cmax:
        t.cost_max = int(m_cmax.group(1))

    m_cmin = _RE_COST_GEQ.search(text)
    if m_cmin:
        t.cost_min = int(m_cmin.group(1))

    m_pmax = _RE_POWER_LEQ.search(text)
    if m_pmax:
        t.power_max = int(m_pmax.group(1).replace(",", ""))

    m_pmin = _RE_POWER_GEQ.search(text)
    if m_pmin:
        t.power_min = int(m_pmin.group(1).replace(",", ""))

    m_color = _RE_COLOR.search(text)
    if m_color:
        t.color = m_color.group(1)

    m_name = _RE_NAME.search(text)
    if m_name:
        t.name_contains = m_name.group(1)

    t.rested = bool(re.search(r"\brested\b", text, re.IGNORECASE))

    return t


# ── Condition ─────────────────────────────────────────────────────────────

# Verbes qui débutent l'effet principal (terminent la clause "if …")
_ACTION_ALTS = (
    r'K\.O\.|give|draw|search|add|play|trash|rest|set|activate|return|'
    r'look|reveal|place|select|your\s+opponent\s+(?:places?|chooses?)|'
    r'this\s+(?:Character|Leader|card)\s+gains|you\s+cannot'
)

_RE_IF = re.compile(
    r'\bif\b\s+(?!this\s+card\s+(?:deals|gains|is))'
    r'(.{5,120}?)'
    r'(?=\.|,\s*(?:' + _ACTION_ALTS + r')(?:\s|\Z)|\Z)',
    re.IGNORECASE | re.DOTALL,
)

_RE_IF_LIFE = re.compile(
    r'if\s+(?:your\s+)?(?:opponent\s+has|you\s+have)\s+(\d+)\s+or\s+(?:fewer|less|more)\s+Life',
    re.IGNORECASE,
)
_RE_IF_HAND_COUNT = re.compile(
    r'if\s+you\s+have\s+(\d+)\s+or\s+(?:fewer|less|more)\s+cards?\s+in\s+your\s+hand',
    re.IGNORECASE,
)
_RE_IF_OPP_HAND = re.compile(
    r'if\s+your\s+opponent\s+has\s+(\d+)\s+or\s+(?:fewer|more)\s+cards?\s+in\s+their\s+hand',
    re.IGNORECASE,
)
_RE_IF_TYPE = re.compile(
    r'if\s+(?:you\s+have\s+a?\s*|your\s+(?:Leader\s+has(?:\s+the)?|Leader\'?s?\s+type\s+includes?)\s*)'
    r'(?:"([^"]+)"|\'([^\']+)\'|\[([^\]]+)\]|\{([^}]+)\})(?:\s+type)?',
    re.IGNORECASE,
)
_RE_IF_FIELD = re.compile(
    r'if\s+you\s+have\s+(\d+)\s+or\s+more\s+(?:\[[^\]]+\]\s+type\s+)?Characters?',
    re.IGNORECASE,
)
_RE_IF_KO_PROTECT = re.compile(
    r'(?:if\s+)?(?:this|your\s+\S+\s+type)\s+(?:card|Character|Leader)\s+would\s+be\s+K\.?O\.?\'?d',
    re.IGNORECASE,
)
_RE_IF_TRASH_COUNT = re.compile(
    r'if\s+(?:there\s+are|you\s+have)\s+(\d+)\s+or\s+more\s+cards?\s+in\s+your\s+trash',
    re.IGNORECASE,
)
_RE_IF_DECK_EMPTY = re.compile(
    r'if\s+(?:your\s+)?deck\s+(?:has|becomes?)\s+0\s+cards?',
    re.IGNORECASE,
)
_RE_IF_DON_FIELD = re.compile(
    r'if\s+(?:the\s+number\s+of\s+DON!!\s+cards?\s+on\s+your\s+field\s+is|'
    r'you\s+have)\s+(\d+)\s+or\s+(?:more|fewer|less)\s+(?:active\s+|rested\s+)?DON!!\s+cards?'
    r'(?:\s+on\s+your\s+field|\s+in\s+play)?',
    re.IGNORECASE,
)
_RE_IF_DON_COMPARE = re.compile(
    r'(?:the\s+number\s+of\s+DON!!\s+cards?\s+on\s+your\s+field\s+is\s+)'
    r'(?:equal\s+to\s+or\s+less\s+than|less\s+than\s+or\s+equal\s+to)'
    r'\s+(?:the\s+number\s+on\s+)?your\s+opponent\'?s',
    re.IGNORECASE,
)
_RE_IF_LEADER_IS = re.compile(
    r'if\s+your\s+Leader\s+is\s+(?:multicolored|active|rested|\[([^\]]+)\]|\{([^}]+)\})',
    re.IGNORECASE,
)
_RE_IF_TYPE_COMPLEX = re.compile(
    r'if\s+your\s+Leader\s+has\s+the\s+'
    r'(?:\[[^\]]+\]|\{[^}]+\}|"[^"]+"|\'[^\']+\')'
    r'\s+type\s+or\s+a\s+type\s+including',
    re.IGNORECASE,
)
_RE_IF_NO_OTHER = re.compile(
    r'if\s+you\s+have\s+no\s+other\s+(?:\[[^\]]+\]\s+)?(?:Characters?|copies)',
    re.IGNORECASE,
)


def _extract_condition(text: str) -> str:
    """Retourne la clause conditionnelle si présente, chaîne vide sinon."""
    for pat in (
        _RE_IF_TYPE_COMPLEX, _RE_IF_DON_FIELD, _RE_IF_DON_COMPARE,
        _RE_IF_LEADER_IS, _RE_IF_NO_OTHER,
        _RE_IF_LIFE, _RE_IF_HAND_COUNT, _RE_IF_OPP_HAND,
        _RE_IF_TYPE, _RE_IF_FIELD, _RE_IF_KO_PROTECT,
        _RE_IF_TRASH_COUNT, _RE_IF_DECK_EMPTY,
    ):
        m = pat.search(text)
        if m:
            return m.group(0).strip()
    m = _RE_IF.search(text)
    if m:
        return "if " + m.group(1).strip()
    return ""


# ── Durée ─────────────────────────────────────────────────────────────────

def _extract_duration(text: str) -> str:
    if re.search(r"until\s+the\s+start\s+of\s+your\s+next\s+turn", text, re.IGNORECASE):
        return "until_next_turn"
    if re.search(r"during\s+this\s+(?:turn|attack)", text, re.IGNORECASE):
        return "this_turn"
    if re.search(r"for\s+the\s+(?:duration|rest)\s+of\s+(?:the\s+)?battle", text, re.IGNORECASE):
        return "battle"
    if re.search(r"for\s+the\s+(?:rest|remainder)\s+of\s+(?:this\s+)?(?:turn|game)", text, re.IGNORECASE):
        return "this_turn"
    if re.search(r"permanently|for\s+the\s+rest\s+of\s+the\s+game", text, re.IGNORECASE):
        return "permanent"
    return "unknown"


# ── Optionnel / Coût ──────────────────────────────────────────────────────

def _is_optional(text: str) -> bool:
    return bool(re.search(r"\byou\s+may\b", text, re.IGNORECASE))


_RE_ACTIVATION_COST = re.compile(
    r'(?:you\s+may\s+)?'
    r'((?:trash|rest|return|discard|remove|place|add)\s+[^:]{3,200}?)'
    r'\s*:\s+',
    re.IGNORECASE,
)


def _extract_activation_cost(text: str) -> str:
    m = _RE_ACTIVATION_COST.search(text)
    if m:
        return m.group(1).strip()
    return ""


# ══════════════════════════════════════════════════════════════════════════
# Règles d'effets
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class _Rule:
    effect_type: EffectType
    pattern: re.Pattern
    extractor: Callable[[re.Match], dict]
    target_hint: str = ""    # pré-remplit target.owner si connu
    confidence: float = 1.0


# ── Pioche ────────────────────────────────────────────────────────────────
_RULES_DRAW: list[_Rule] = [
    _Rule(
        EffectType.DRAW,
        re.compile(r"draw\s+(?:up\s+to\s+)?(\d+|one|two|three|four|five)\s+cards?", re.IGNORECASE),
        lambda m: {"n": _word_to_int(m.group(1))},
    ),
    _Rule(
        EffectType.DRAW,
        re.compile(r"draw\s+(?:a|an)\s+card", re.IGNORECASE),
        lambda m: {"n": 1},
    ),
    _Rule(
        EffectType.DRAW,
        re.compile(r"place\s+it\s+in\s+your\s+hand", re.IGNORECASE),
        lambda m: {"n": 1, "source": "deck"},
        confidence=0.8,
    ),
]

# ── Look / reveal ──────────────────────────────────────────────────────────
_RULES_LOOK: list[_Rule] = [
    _Rule(
        EffectType.LOOK_TOP,
        re.compile(
            r"look\s+at\s+(?:the\s+)?(?:top\s+)?(\d+|one|two|three|four|five)\s+cards?",
            re.IGNORECASE,
        ),
        lambda m: {"n": _word_to_int(m.group(1))},
    ),
    _Rule(
        EffectType.REVEAL_TOP,
        re.compile(
            r"reveal\s+(?:the\s+)?(?:top\s+)?(\d+|one|two|three)?\s*cards?\s+(?:from\s+the\s+top\s+of\s+)?(?:your\s+)?deck",
            re.IGNORECASE,
        ),
        lambda m: {"n": _word_to_int(m.group(1)) if m.group(1) else 1},
        confidence=0.9,
    ),
]

# ── Recherche ──────────────────────────────────────────────────────────────
_RULES_SEARCH: list[_Rule] = [
    # search deck for up to N [type] type cards with cost X or less
    _Rule(
        EffectType.SEARCH_DECK,
        re.compile(
            r"search\s+your\s+deck\s+for\s+(?:up\s+to\s+)?(\d+|one|two|three|a)?\s*"
            r"(?:\[[^\]]+\]\s+type\s+)?"
            r"(?:Character|Event|Stage|Leader|card)s?",
            re.IGNORECASE,
        ),
        lambda m: {
            "n": _word_to_int(m.group(1)) if m.group(1) not in (None, "a") else 1,
        },
        confidence=0.95,
    ),
]

# ── Défausse / Trash ───────────────────────────────────────────────────────
_RULES_TRASH: list[_Rule] = [
    # trash from hand
    _Rule(
        EffectType.TRASH_FROM_HAND,
        re.compile(
            r"(?:you\s+may\s+)?trash\s+(?:up\s+to\s+)?(\d+|one|two|three|a)?\s*"
            r"(?:Event\s+or\s+Stage\s+|Character\s+|Event\s+|Stage\s+)?card(?:s)?"
            r"\s+from\s+your\s+hand",
            re.IGNORECASE,
        ),
        lambda m: {
            "n": _word_to_int(m.group(1)) if m.group(1) not in (None, "a") else 1,
        },
        target_hint="you",
    ),
    # discard (alias)
    _Rule(
        EffectType.TRASH_FROM_HAND,
        re.compile(r"discard\s+(?:up\s+to\s+)?(\d+|one|two|three)\s+cards?", re.IGNORECASE),
        lambda m: {"n": _word_to_int(m.group(1))},
        confidence=0.85,
    ),
    # trash from top of deck
    _Rule(
        EffectType.TRASH_FROM_DECK,
        re.compile(
            r"trash\s+(?:the\s+)?(?:top\s+)?(\d+|one|two|three)?\s*cards?(?:\s+from\s+the\s+top\s+of)?\s+"
            r"(?:your\s+)?deck",
            re.IGNORECASE,
        ),
        lambda m: {"n": _word_to_int(m.group(1)) if m.group(1) else 1},
        confidence=0.85,
    ),
    # trash N of your Life cards
    _Rule(
        EffectType.TRASH_FROM_LIFE,
        re.compile(
            r"trash\s+(?:up\s+to\s+)?(\d+)?\s*(?:of\s+(?:your|your\s+opponent'?s?)\s+)?Life\s+cards?",
            re.IGNORECASE,
        ),
        lambda m: {"n": _int(m.group(1))},
        confidence=0.9,
    ),
    # trash from trash zone (to activate)
    _Rule(
        EffectType.TRASH_FROM_HAND,
        re.compile(
            r"trash\s+(\d+|one|two|three|a)\s+(?:\[[^\]]+\]\s+type\s+)?(?:Character|Event|Stage)\s+card"
            r"\s+from\s+your\s+hand",
            re.IGNORECASE,
        ),
        lambda m: {
            "n": _word_to_int(m.group(1)) if m.group(1) != "a" else 1,
        },
        target_hint="you",
    ),
]

# ── Jouer des cartes ───────────────────────────────────────────────────────
_RULES_PLAY: list[_Rule] = [
    # play from trash (handles "Characters", power/name filters, [CardName])
    _Rule(
        EffectType.PLAY_FROM_TRASH,
        re.compile(
            r"play\s+(?:up\s+to\s+)?(\d+|one|two|three|a)?\s*"
            r"(?:\[[^\]]+\](?:\s+type)?\s+|\"[^\"]+\"\s+type\s+)?"
            r"(?:Character|Event|Stage)?s?\s*(?:cards?\s*)?"
            r"[^.]{0,120}?\s+from\s+your\s+(?:trash|discard)",
            re.IGNORECASE,
        ),
        lambda m: {
            "n": _word_to_int(m.group(1)) if m.group(1) not in (None, "a") else 1,
            "free": bool(re.search(r"for\s+free", m.string, re.IGNORECASE)),
        },
        confidence=0.9,
    ),
    # play this card (trigger)
    _Rule(
        EffectType.PLAY_FROM_HAND,
        re.compile(r"play\s+this\s+card", re.IGNORECASE),
        lambda m: {"source": "trigger", "free": True},
    ),
    # play up to 1 card from deck (after look)
    _Rule(
        EffectType.PLAY_FROM_DECK,
        re.compile(
            r"play\s+(?:up\s+to\s+)?(\d+|one|two|three|a)?\s*"
            r"(?:\[[^\]]+\]\s+type\s+)?(?:Character|Event|Stage)?\s*cards?"
            r"(?:\s+with\s+[^.]+)?"
            r"\s+(?:from\s+(?:your\s+deck|among\s+them))",
            re.IGNORECASE,
        ),
        lambda m: {
            "n": _word_to_int(m.group(1)) if m.group(1) not in (None, "a") else 1,
        },
        confidence=0.85,
    ),
]

# ── Retour en main ────────────────────────────────────────────────────────
_RULES_HAND: list[_Rule] = [
    # add from own Life (top or bottom) to hand — AVANT la règle générique « add … to hand »
    # (sinon Zeus OP11-106 matche la générique sans source:life → pioche depuis le deck par erreur)
    _Rule(
        EffectType.ADD_TO_HAND,
        re.compile(
            r"add\s+(?:up\s+to\s+)?(\d+|one|a)?\s*cards?\s+"
            r"from\s+(?:the\s+top\s+or\s+bottom\s+of\s+)?your\s+Life\s+cards?"
            r"\s+to\s+(?:your|their)\s+hand",
            re.IGNORECASE,
        ),
        lambda m: {"n": _word_to_int(m.group(1)) if m.group(1) not in (None, "a") else 1, "source": "life"},
        confidence=0.95,
    ),
    # return from field to hand (handles "Characters", "of your opponent's Characters")
    _Rule(
        EffectType.RETURN_TO_HAND,
        re.compile(
            r"return\s+(?:up\s+to\s+)?(\d+|one|two|three|all)?\s*"
            r"(?:of\s+(?:your|their|the\s+owner'?s?)\s+)?(?:opponent'?s?\s+)?"
            r"(?:rested\s+)?(?:\[[^\]]+\]\s+type\s+)?(?:Character|Event|Stage|Leader)?s?\s*"
            r"(?:cards?\s*)?"
            r"[^.]{0,120}?(?:to|into)\s+(?:(?:its?|the)\s+owner'?s?\s+|their\s+|your\s+)?hand",
            re.IGNORECASE,
        ),
        lambda m: {"n": _word_to_int(m.group(1)) if m.group(1) not in (None, "all") else 1},
        confidence=0.9,
    ),
    # add N cards from trash/deck to hand
    _Rule(
        EffectType.ADD_TO_HAND,
        re.compile(
            r"add\s+(?:up\s+to\s+)?(\d+|one|two|three|a)?\s*"
            r"(?:\[[^\]]+\](?:\s+type)?\s+|\"[^\"]+\"\s+type\s+)?"
            r"(?:Character|Event|Stage|Leader)?s?\s*(?:cards?\s*)?"
            r"[^.]{0,120}?to\s+(?:your|their)\s+hand",
            re.IGNORECASE,
        ),
        lambda m: {"n": _word_to_int(m.group(1)) if m.group(1) not in (None, "a") else 1},
        confidence=0.85,
    ),
    # put at bottom of deck (the rest)
    _Rule(
        EffectType.ADD_TO_HAND,
        re.compile(r"place\s+(?:the\s+rest|them)?\s*at\s+the\s+bottom\s+of\s+your\s+deck", re.IGNORECASE),
        lambda m: {"destination": "deck_bottom"},
        confidence=0.9,
    ),
]

# ── DON!! ─────────────────────────────────────────────────────────────────
_RULES_DON: list[_Rule] = [
    # give rested DON!! to leader/character (direct)
    _Rule(
        EffectType.DON_ATTACH,
        re.compile(
            r"give\s+(?:up\s+to\s+)?(\d+)\s+rested\s+DON!!",
            re.IGNORECASE,
        ),
        lambda m: {"n": int(m.group(1))},
    ),
    # give your [target] up to N rested DON!! card (distributed)
    _Rule(
        EffectType.DON_ATTACH,
        re.compile(
            r"give\s+(?:your\s+)?(?:Leader|Character).{0,60}?up\s+to\s+(\d+)\s+rested\s+DON!!",
            re.IGNORECASE,
        ),
        lambda m: {"n": int(m.group(1)), "distribution": "each"},
        confidence=0.9,
    ),
    # return DON!! card to deck
    _Rule(
        EffectType.DON_RETURN,
        re.compile(
            r"return\s+(\d+)\s+DON!!\s+card(?:s)?\s+from\s+your\s+field\s+to\s+your\s+DON!!\s+deck",
            re.IGNORECASE,
        ),
        lambda m: {"n": int(m.group(1))},
    ),
    # rest a DON!! card
    _Rule(
        EffectType.DON_REST,
        re.compile(r"rest\s+(\d+)?\s*(?:of\s+your\s+)?(?:active\s+)?DON!!", re.IGNORECASE),
        lambda m: {"n": _int(m.group(1))},
        confidence=0.8,
    ),
    # activate a DON!! card
    _Rule(
        EffectType.DON_ACTIVATE,
        re.compile(r"activate\s+(?:up\s+to\s+)?(\d+)?\s*(?:of\s+your\s+)?DON!!", re.IGNORECASE),
        lambda m: {"n": _int(m.group(1))},
        confidence=0.8,
    ),
]

# ── Boost de puissance ────────────────────────────────────────────────────
_RULES_BOOST: list[_Rule] = [
    # gains +N power
    _Rule(
        EffectType.POWER_BOOST,
        re.compile(r"gains?\s+\+(\d[\d,]*)\s*power", re.IGNORECASE),
        lambda m: {"amount": _int(m.group(1))},
    ),
    # give your Leader/Character(s) +N power — inclut "Leader and N Characters"
    _Rule(
        EffectType.POWER_BOOST,
        re.compile(
            r"give\s+(?:up\s+to\s+\d+\s+of\s+)?(?:your|this)\s+"
            r"(?:Leader(?:\s+and\s+(?:up\s+to\s+)?\d+\s+Characters?)?|Characters?)"
            r"[^.]*?\+(\d[\d,]*)\s*power",
            re.IGNORECASE,
        ),
        lambda m: {"amount": _int(m.group(1))},
        target_hint="you",
        confidence=0.9,
    ),
    # +N power (for every X / per X in trash)
    _Rule(
        EffectType.POWER_BOOST,
        re.compile(
            r"\+(\d[\d,]*)\s*power\s+for\s+every\s+(\d+)\s+(.{3,40})\s+in\s+your\s+(trash|hand|field)",
            re.IGNORECASE,
        ),
        lambda m: {
            "amount_per":  _int(m.group(1)),
            "per_n":       _int(m.group(2)),
            "per_source":  m.group(3).strip(),
            "zone":        m.group(4),
        },
        confidence=0.8,
    ),
]

# ── Réduction de puissance ─────────────────────────────────────────────────
_RULES_REDUCE: list[_Rule] = [
    # give opponent's Leader/Character -N power
    _Rule(
        EffectType.POWER_REDUCE,
        re.compile(
            r"give\s+(?:up\s+to\s+)?(?:\d+\s+of\s+)?your\s+opponent'?s?\s+"
            r"(?:Leader|Characters?|Leader\s+or\s+Characters?)"
            r"[^.]*?-(\d[\d,]*)\s*power",
            re.IGNORECASE,
        ),
        lambda m: {"amount": _int(m.group(1))},
        target_hint="opponent",
    ),
    # -N power (general)
    _Rule(
        EffectType.POWER_REDUCE,
        re.compile(r"-(\d[\d,]*)\s*power", re.IGNORECASE),
        lambda m: {"amount": _int(m.group(1))},
        confidence=0.75,
    ),
    # give N Characters N cost (unsigned cost reduction e.g. "2 cost during this turn")
    _Rule(
        EffectType.COST_REDUCE,
        re.compile(
            r"give\s+(?:up\s+to\s+)?(\d+)?\s*(?:of\s+)?your\s+opponent'?s?\s+"
            r"(?:Leader|Characters?|Leader\s+or\s+Characters?|cards?)"
            r"(?:\s+cards?)?\s+(\d+)\s+cost",
            re.IGNORECASE,
        ),
        lambda m: {"n": _int(m.group(1)), "amount": _int(m.group(2))},
        target_hint="opponent",
        confidence=0.8,
    ),
]

# ── K.O. ──────────────────────────────────────────────────────────────────
_RULES_KO: list[_Rule] = [
    # K.O. up to N of your opponent's [rested] Characters (with filters)
    _Rule(
        EffectType.KO,
        re.compile(
            r"K\.?O\.?\s+(?:up\s+to\s+)?(\d+)?\s*(?:of\s+)?your\s+opponent'?s?\s+"
            r"(?:rested\s+)?(?:\[[^\]]+\]\s+type\s+)?Characters?",
            re.IGNORECASE,
        ),
        lambda m: {"n": _int(m.group(1))},
        target_hint="opponent",
    ),
    # K.O. this Character
    _Rule(
        EffectType.KO,
        re.compile(r"K\.?O\.?\s+this\s+(?:Character|card)", re.IGNORECASE),
        lambda m: {"n": 1},
        target_hint="self",
    ),
    # would be K.O.'d → protect
    _Rule(
        EffectType.KO,
        re.compile(r"would\s+be\s+K\.?O\.?'?d\s+by\s+(?:an?\s+)?(?:effect|attack)", re.IGNORECASE),
        lambda m: {"trigger": "ko_protection"},
        confidence=0.8,
    ),
]

# ── Bannissement ──────────────────────────────────────────────────────────
_RULES_BANISH: list[_Rule] = [
    _Rule(
        EffectType.BANISH,
        re.compile(r"card\s+is\s+trashed\s+without\s+activating\s+its\s+Trigger", re.IGNORECASE),
        lambda m: {"via": "banish_rule"},
    ),
    _Rule(
        EffectType.GIVE_KEYWORD,
        re.compile(r"gains?\s+\[Banish\]\s+during\s+this\s+turn", re.IGNORECASE),
        lambda m: {"keyword": "Banish", "duration": "this_turn"},
    ),
]

# ── Mettre au repos / activer ──────────────────────────────────────────────
_RULES_REST: list[_Rule] = [
    # rest an opponent's card/character/leader
    _Rule(
        EffectType.REST_TARGET,
        re.compile(
            r"rest\s+(?:up\s+to\s+)?(\d+)?\s*(?:of\s+)?your\s+opponent'?s?\s+"
            r"(?:active\s+)?(?:\[[^\]]+\]\s+type\s+)?(?:Leader|Characters?|Leader\s+or\s+Characters?|cards?)",
            re.IGNORECASE,
        ),
        lambda m: {"n": _int(m.group(1))},
        target_hint="opponent",
    ),
    # rest this Character (activation cost)
    _Rule(
        EffectType.REST_TARGET,
        re.compile(r"(?:you\s+may\s+)?rest\s+this\s+(?:Character|card|Leader)", re.IGNORECASE),
        lambda m: {"n": 1},
        target_hint="self",
        confidence=0.9,
    ),
]

# ── Vie ────────────────────────────────────────────────────────────────────
_RULES_LIFE: list[_Rule] = [
    # add N card(s) / Characters from deck/hand to Life (all variants)
    _Rule(
        EffectType.LIFE_ADD,
        re.compile(
            r"add\s+(?:up\s+to\s+)?(\d+|one|two|three|a)?\s*"
            r"(?:of\s+(?:your|their)\s+)?(?:\[[^\]]+\](?:\s+type)?\s+)?(?:Characters?|cards?)?"
            r"(?:\s+with\s+[^.]{0,60}?)?"
            r"\s*(?:from\s+[^,.]{0,60}?)?"
            r"to\s+(?:the\s+top\s+of\s+)?(?:your|their|the\s+owner'?s?)\s+Life",
            re.IGNORECASE,
        ),
        lambda m: {
            "n": _word_to_int(m.group(1)) if m.group(1) not in (None, "a") else 1,
            "source": "hand" if re.search(r"from\s+your\s+hand", m.string, re.IGNORECASE) else "deck_top",
        },
    ),
    # trash N of your Life cards
    _Rule(
        EffectType.LIFE_REMOVE,
        re.compile(
            r"trash\s+(?:up\s+to\s+)?(\d+)?\s*(?:of\s+)?(?:your|your\s+opponent'?s?)\s+Life\s+cards?",
            re.IGNORECASE,
        ),
        lambda m: {"n": _int(m.group(1))},
    ),
    # send X life to top/bottom of deck
    _Rule(
        EffectType.LIFE_REMOVE,
        re.compile(
            r"place\s+(?:up\s+to\s+)?(\d+)?\s*(?:of\s+)?(?:your|your\s+opponent'?s?)\s+Life\s+cards?"
            r"[^.]*?(?:on|at)\s+the\s+(?:top|bottom)\s+of\s+(?:your|their)\s+deck",
            re.IGNORECASE,
        ),
        lambda m: {"n": _int(m.group(1)), "destination": "deck"},
        confidence=0.85,
    ),
]

# ── Jouer depuis la main (toutes variantes) ───────────────────────────────
_RULES_PLAY_HAND: list[_Rule] = [
    # Type filter : [X] / "X" / 'X' suivi de "type"
    # play from hand (standard) — accepte [X] ou "X" ou 'X' comme filtre de type
    _Rule(
        EffectType.PLAY_FROM_HAND,
        re.compile(
            r"play\s+(?:up\s+to\s+)?(\d+|one|two|three|a)?\s*"
            r'(?:(?:\[[^\]]+\]|"[^"]+"|\'[^\']+\')\s+(?:or\s+(?:\[[^\]]+\]|"[^"]+"|\'[^\']+\')\s+)?(?:type|type\s+including)\s+)?'
            r"(?:Character|Event|Stage)?\s*cards?"
            r"(?:.{0,100}?(?:cost|name).{0,80}?)?"  # filtres optionnels
            r"\s+(?:from\s+your\s+hand|for\s+free)",
            re.IGNORECASE,
        ),
        lambda m: {
            "n": _word_to_int(m.group(1)) if m.group(1) not in (None, "a") else 1,
            "source": "hand",
        },
        confidence=0.9,
    ),
    # play from hand OR trash (noms de cartes avec points OK car on utilise .{0,150}?)
    _Rule(
        EffectType.PLAY_FROM_HAND,
        re.compile(
            r"play\s+(?:up\s+to\s+)?(\d+|one|two|three|a)?\s*"
            r'(?:(?:\[[^\]]+\]|"[^"]+"|\'[^\']+\')\s+(?:type\s+)?)?'
            r"(?:Character|Event|Stage)?\s*cards?"
            r".{0,150}?from\s+your\s+hand\s+or\s+trash",
            re.IGNORECASE,
        ),
        lambda m: {
            "n": _word_to_int(m.group(1)) if m.group(1) not in (None, "a") else 1,
            "source": "hand_or_trash",
        },
        confidence=0.9,
    ),
    # play up to N [CardName] from hand (nom de carte sans "cards?")
    _Rule(
        EffectType.PLAY_FROM_HAND,
        re.compile(
            r"play\s+(?:up\s+to\s+)?(\d+|one|a)?\s*\[[^\]]+\][^.]{0,80}?from\s+your\s+hand",
            re.IGNORECASE,
        ),
        lambda m: {
            "n": _word_to_int(m.group(1)) if m.group(1) not in (None, "a") else 1,
            "source": "hand",
        },
        confidence=0.85,
    ),
]

# ── DON!! ajouter depuis le deck DON!! ────────────────────────────────────
_RULES_DON_ADD: list[_Rule] = [
    _Rule(
        EffectType.DON_ADD,
        re.compile(
            # "Ad" = typo CSV courant (OCR artifact)
            r"add?\s+(?:up\s+to\s+)?(\d+)\s+DON!!\s+card(?:s)?\s+from\s+your\s+DON!!\s+deck",
            re.IGNORECASE,
        ),
        lambda m: {"n": int(m.group(1))},
    ),
]

# ── Puissance / coût étendus ──────────────────────────────────────────────
_RULES_POWER_EXT: list[_Rule] = [
    # base power becomes N
    _Rule(
        EffectType.SET_POWER,
        re.compile(
            r"(?:base\s+)?power\s+becomes?\s+(\d[\d,]*)",
            re.IGNORECASE,
        ),
        lambda m: {"power": _int(m.group(1))},
    ),
    # give opponent's Characters -N cost
    _Rule(
        EffectType.COST_REDUCE,
        re.compile(
            r"give\s+(?:all\s+of\s+)?(?:your\s+)?(?:opponent'?s?\s+)?(?:Characters?\s+)?-(\d+)\s+cost",
            re.IGNORECASE,
        ),
        lambda m: {"amount": _int(m.group(1))},
        target_hint="opponent",
        confidence=0.85,
    ),
    # gains [keyword] during this turn
    _Rule(
        EffectType.GIVE_KEYWORD,
        re.compile(
            r"(?:this\s+(?:Character|Leader|card)\s+)?gains?\s+"
            r"\[(Rush|Blocker|Double\s*Attack|Banish|Unblockable|Infiltrate)\]"
            r"(?:\s+during\s+this\s+turn)?",
            re.IGNORECASE,
        ),
        lambda m: {"keyword": m.group(1)},
        confidence=0.9,
    ),
]

# ── Protection / état ─────────────────────────────────────────────────────
_RULES_PROTECT: list[_Rule] = [
    # cannot be K.O.'d during this battle
    _Rule(
        EffectType.CANNOT_KO,
        re.compile(r"cannot\s+be\s+K\.?O\.?'?d\s+during\s+this\s+(?:turn|battle|game)", re.IGNORECASE),
        lambda m: {"scope": "battle"},
        confidence=0.95,
    ),
    # cannot be K.O.'d by an effect
    _Rule(
        EffectType.CANNOT_KO,
        re.compile(r"cannot\s+be\s+K\.?O\.?'?d\s+by\s+(?:an?\s+)?effect", re.IGNORECASE),
        lambda m: {"scope": "effect"},
        confidence=0.95,
    ),
    # you may trash [card] instead (sacrifice protection)
    _Rule(
        EffectType.SACRIFICE_PROTECT,
        re.compile(
            r"you\s+may\s+trash\s+(?:this\s+)?(?:\[[^\]]+\]\s+type\s+)?(?:Character|card)?"
            r"[^.]*?instead",
            re.IGNORECASE,
        ),
        lambda m: {"mechanism": "trash_instead"},
        confidence=0.85,
    ),
]

# ── Placer au bas/haut du deck ────────────────────────────────────────────
_RULES_DECK_PLACE: list[_Rule] = [
    # place N Characters (yours/opponent's) at deck bottom
    _Rule(
        EffectType.PLACE_TO_DECK,
        re.compile(
            r"place\s+(?:up\s+to\s+)?(\d+|one|two|three|all|a)?\s*"
            r"(?:of\s+)?(?:your|their|your\s+opponent'?s?)\s+"
            r"(?:rested\s+)?(?:\[[^\]]+\](?:\s+type)?\s+)?(?:Character|card|Leader)s?"
            r"[^.]*?(?:at|on|to)\s+the\s+(?:top|bottom)\s+of\s+(?:the\s+)?(?:owner'?s?\s+)?deck",
            re.IGNORECASE,
        ),
        lambda m: {
            "n": _word_to_int(m.group(1)) if m.group(1) not in (None, "a") else 1,
        },
        confidence=0.85,
    ),
    # opponent places card from hand to deck bottom
    _Rule(
        EffectType.PLACE_TO_DECK,
        re.compile(
            r"(?:your\s+opponent\s+places?\s+|places?\s+)(\d+|one|two|three|a)?\s*card(?:s)?"
            r"\s+from\s+their\s+hand\s+at\s+the\s+bottom\s+of\s+(?:their|the)\s+deck",
            re.IGNORECASE,
        ),
        lambda m: {
            "n": _word_to_int(m.group(1)) if m.group(1) not in (None, "a") else 1,
            "target_owner": "opponent",
            "source": "hand",
        },
        target_hint="opponent",
        confidence=0.9,
    ),
]

# ── Auto-déclenchement et texte de règle spéciale ─────────────────────────
_RULES_SPECIAL: list[_Rule] = [
    # Activate this card's [Main] effect (Trigger → re-use Main)
    _Rule(
        EffectType.ACTIVATE_SELF_EFFECT,
        re.compile(
            r"activate\s+this\s+card'?s?\s+(?:\[\w+\]|<<[^>]+>>)\s+effect",
            re.IGNORECASE,
        ),
        lambda m: {"source": "self_main"},
        confidence=0.95,
    ),
    # Under the rules of this game… (texte de règle spéciale)
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(
            r"under\s+the\s+rules\s+of\s+this\s+game|"
            r"according\s+to\s+the\s+rules|"
            r"you\s+do\s+not\s+lose\s+when",
            re.IGNORECASE,
        ),
        lambda m: {"type": "game_rule"},
        confidence=0.95,
    ),
    # [Your Turn] passive effect marker
    _Rule(
        EffectType.CONDITION,
        re.compile(r"\[Your\s+Turn\]", re.IGNORECASE),
        lambda m: {"scope": "your_turn"},
        confidence=0.9,
    ),
]

# ── KO étendu (Stages, power filter) ──────────────────────────────────────
_RULES_KO_EXT: list[_Rule] = [
    # K.O. Stages
    _Rule(
        EffectType.KO,
        re.compile(
            r"K\.?O\.?\s+(?:up\s+to\s+)?(\d+)?\s*(?:of\s+)?your\s+opponent'?s?\s+Stages?",
            re.IGNORECASE,
        ),
        lambda m: {"n": _int(m.group(1)), "target_type": "Stage"},
        target_hint="opponent",
        confidence=0.9,
    ),
    # K.O. with power filter "with X power or less"
    _Rule(
        EffectType.KO,
        re.compile(
            r"K\.?O\.?\s+(?:up\s+to\s+)?(\d+)?\s*(?:of\s+)?your\s+opponent'?s?\s+"
            r"(?:Leader\s+or\s+)?Characters?\s+with\s+(?:\d[\d,]*)\s+power\s+or\s+less",
            re.IGNORECASE,
        ),
        lambda m: {"n": _int(m.group(1))},
        target_hint="opponent",
        confidence=0.9,
    ),
]

# ── DON!! set active / add+active ────────────────────────────────────────
_RULES_DON_ACTIVE: list[_Rule] = [
    # set N of your DON!! cards as active
    _Rule(
        EffectType.DON_ACTIVATE,
        re.compile(
            r"set\s+(?:up\s+to\s+)?(\d+)?\s*(?:of\s+your\s+)?(?:rested\s+)?DON!!\s+cards?\s+as\s+active",
            re.IGNORECASE,
        ),
        lambda m: {"n": _int(m.group(1)), "method": "set_active"},
    ),
    # add N DON!! and set it as active (combo)
    _Rule(
        EffectType.DON_ADD,
        re.compile(
            r"add\s+(?:up\s+to\s+)?(\d+)\s+DON!!\s+card(?:s)?[^.]*?set\s+it\s+as\s+active",
            re.IGNORECASE,
        ),
        lambda m: {"n": int(m.group(1)), "then_set_active": True},
    ),
    # you cannot set DON!! as active (restriction)
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(
            r"you\s+cannot\s+set\s+DON!!\s+cards?\s+as\s+active",
            re.IGNORECASE,
        ),
        lambda m: {"type": "restriction", "action": "set_don_active"},
        confidence=0.9,
    ),
]

# ── Effets sur tout le champ ──────────────────────────────────────────────
_RULES_FIELD: list[_Rule] = [
    # give all of your [type] Characters +N power
    _Rule(
        EffectType.POWER_BOOST,
        re.compile(
            r"give\s+all\s+(?:of\s+)?(?:your\s+)?(?:\[[^\]]+\]\s+type\s+)?Characters?\s*\+(\d[\d,]*)\s*power",
            re.IGNORECASE,
        ),
        lambda m: {"amount": _int(m.group(1)), "scope": "all"},
        target_hint="you",
    ),
    # give all of your opponent's Characters -N power
    _Rule(
        EffectType.POWER_REDUCE,
        re.compile(
            r"give\s+all\s+(?:of\s+)?your\s+opponent'?s?\s+Characters?\s*-(\d[\d,]*)\s*power",
            re.IGNORECASE,
        ),
        lambda m: {"amount": _int(m.group(1)), "scope": "all"},
        target_hint="opponent",
    ),
    # give all of your opponent's Characters -N cost
    _Rule(
        EffectType.COST_REDUCE,
        re.compile(
            r"give\s+all\s+(?:of\s+)?your\s+opponent'?s?\s+Characters?\s*-(\d+)\s*cost",
            re.IGNORECASE,
        ),
        lambda m: {"amount": _int(m.group(1)), "scope": "all"},
        target_hint="opponent",
        confidence=0.9,
    ),
    # this Character's base power becomes the same as target
    _Rule(
        EffectType.SET_POWER,
        re.compile(
            r"this\s+(?:Character|Leader)'?s?\s+(?:base\s+)?power\s+becomes?\s+the\s+(?:same\s+as|equal\s+to)",
            re.IGNORECASE,
        ),
        lambda m: {"power": "mimic", "source": "target"},
        confidence=0.85,
    ),
    # you cannot [action] during this turn (restriction)
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(
            r"you\s+cannot\s+(?:play\s+Character\s+cards?|attack|block|use\s+effects?|activate)[^.]{0,60}",
            re.IGNORECASE,
        ),
        lambda m: {"type": "restriction", "clause": m.group(0)[:80]},
        confidence=0.85,
    ),
    # your opponent chooses one: (branch effect)
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(r"your\s+opponent\s+chooses?\s+one\s*:", re.IGNORECASE),
        lambda m: {"type": "opponent_choice"},
        confidence=0.9,
    ),
    # choose one: (branch effect — après début de ligne, ., ;, ou ,)
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(r"(?:^|[;.,])\s*choose\s+one\s*:", re.IGNORECASE),
        lambda m: {"type": "player_choice"},
        confidence=0.9,
    ),
    # select up to N of your opponent's Characters [and apply effect]
    _Rule(
        EffectType.REST_TARGET,
        re.compile(
            r"select\s+(?:up\s+to\s+)?(\d+)?\s*(?:of\s+)?your\s+opponent'?s?\s+"
            r"(?:active\s+)?(?:\[[^\]]+\]\s+type\s+)?Characters?",
            re.IGNORECASE,
        ),
        lambda m: {"n": _int(m.group(1)), "method": "select"},
        target_hint="opponent",
        confidence=0.75,
    ),
    # gains +N cost (cost increase on self)
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(r"(?:this\s+(?:Character|Leader)\s+)?gains?\s+\+(\d+)\s+cost", re.IGNORECASE),
        lambda m: {"type": "cost_increase", "amount": _int(m.group(1))},
        confidence=0.8,
    ),
    # give N of your Characters +N power (specific count, not "all")
    _Rule(
        EffectType.POWER_BOOST,
        re.compile(
            r"give\s+(?:up\s+to\s+)?(\d+)\s+of\s+your\s+(?:\[[^\]]+\]\s+type\s+)?Characters?\s*\+(\d[\d,]*)\s*power",
            re.IGNORECASE,
        ),
        lambda m: {"n_targets": _int(m.group(1)), "amount": _int(m.group(2))},
        target_hint="you",
        confidence=0.9,
    ),
    # gains an additional +N power (fragment matching e.g. Counter effects)
    _Rule(
        EffectType.POWER_BOOST,
        re.compile(r"gains?\s+(?:an\s+additional\s+)?\+(\d[\d,]*)\s*power", re.IGNORECASE),
        lambda m: {"amount": _int(m.group(1)), "fragment": True},
        confidence=0.8,
    ),
    # give opponent card N power (without explicit sign — implicit buff, ex Counter)
    _Rule(
        EffectType.POWER_BOOST,
        re.compile(
            r"give\s+(?:up\s+to\s+)?(\d+)?\s*(?:of\s+)?your\s+opponent'?s?\s+"
            r"(?:Leader|Characters?|Leader\s+or\s+Characters?|cards?)"
            r"(?:\s+cards?)?\s+(\d[\d,]*)\s*power",
            re.IGNORECASE,
        ),
        lambda m: {"n": _int(m.group(1)), "amount": _int(m.group(2)), "unsigned": True},
        target_hint="opponent",
        confidence=0.75,
    ),
    # rest this [Character/Stage/card] (activer sa propre carte comme coût/effet)
    _Rule(
        EffectType.REST_TARGET,
        re.compile(r"rest\s+this\s+(?:Character|Stage|Leader|card)", re.IGNORECASE),
        lambda m: {"target": "self"},
        target_hint="self",
        confidence=0.85,
    ),
    # this Character cannot attack (restriction absolue)
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(r"this\s+(?:Character|Leader|card)\s+cannot\s+attack", re.IGNORECASE),
        lambda m: {"type": "restriction", "action": "cannot_attack"},
        confidence=0.9,
    ),
    # all Characters with cost N or less can also attack / gain [keyword]
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(r"all\s+Characters?\s+with\s+a\s+cost\s+of\s+(\d+)\s+or\s+less\s+can", re.IGNORECASE),
        lambda m: {"type": "aura", "cost_max": int(m.group(1))},
        confidence=0.85,
    ),
    # set N of your / this [type] rested? Character(s) (cards?) as active
    _Rule(
        EffectType.SET_ACTIVE,
        re.compile(
            r"set\s+(?:up\s+to\s+)?(\d+)?\s*"
            r"(?:of\s+your\s+|this\s+)?"
            r"(?:\"[^\"]+\"\s+type\s+|'[^']+'\s+type\s+|\[[^\]]+\](?:\s+type)?\s+|\{[^}]+\}\s+type\s+)?"
            r"(?:rested\s+)?(?:Leaders?|Characters?)"
            r"(?:\s+cards?)?"
            r"[^.]{0,80}?as\s+active",
            re.IGNORECASE,
        ),
        lambda m: {"n": _int(m.group(1)), "method": "set_active"},
        target_hint="you",
        confidence=0.9,
    ),
    # can also attack active Characters (Rush-like)
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(r"can\s+(?:also\s+)?attack\s+active\s+Characters?", re.IGNORECASE),
        lambda m: {"type": "attack_active"},
        confidence=0.9,
    ),
    # cannot be rested (restriction on target)
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(r"cannot\s+be\s+rested", re.IGNORECASE),
        lambda m: {"type": "restriction", "action": "cannot_rest"},
        confidence=0.9,
    ),
    # cannot attack unless [condition] (self-restriction)
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(r"cannot\s+attack\s+unless", re.IGNORECASE),
        lambda m: {"type": "restriction", "action": "attack_requires_condition"},
        confidence=0.85,
    ),
    # (add Life → main : voir _RULES_HAND en tête de liste — ordre de priorité global)
    # place all Characters with cost N or less at bottom of deck
    _Rule(
        EffectType.PLACE_TO_DECK,
        re.compile(
            r"place\s+all\s+Characters?\s+with\s+a\s+cost\s+of\s+(\d+)\s+or\s+less"
            r"\s+at\s+the\s+bottom\s+of\s+(?:the\s+)?(?:owner'?s?\s+)?deck",
            re.IGNORECASE,
        ),
        lambda m: {"scope": "all", "dest": "deck_bottom", "cost_max": int(m.group(1))},
        confidence=0.9,
    ),
    # When this Character becomes rested, [effect] (self-trigger condition)
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(r"when\s+this\s+(?:Character|card)\s+becomes?\s+rested", re.IGNORECASE),
        lambda m: {"type": "trigger", "event": "self_rest"},
        confidence=0.85,
    ),
]

# ── Restrictions / états spéciaux ─────────────────────────────────────────
_RULES_RESTRICTION: list[_Rule] = [
    # DON!! -N cost (coût normalisé par TimingModel en [DON COST -N])
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(r"\[DON\s+COST\s*-(\d+)\]", re.IGNORECASE),
        lambda m: {"type": "don_cost", "n": int(m.group(1))},
        confidence=0.95,
    ),
    # DON!! N : effect (syntaxe sans tiret, ex: "DON!! 1: ...")
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(r"DON!!\s+(\d+)\s*:", re.IGNORECASE),
        lambda m: {"type": "don_cost_add", "n": int(m.group(1))},
        confidence=0.9,
    ),
    # your opponent cannot attack any card other than X
    _Rule(
        EffectType.SPECIAL_RULE,
        re.compile(r"your\s+opponent\s+cannot\s+attack\s+any\s+card\s+other\s+than", re.IGNORECASE),
        lambda m: {"type": "restriction", "action": "force_target"},
        confidence=0.9,
    ),
    # When N or more DON!! cards are returned to your deck
    _Rule(
        EffectType.DON_RETURN,
        re.compile(
            r"when\s+(\d+)\s+or\s+more\s+DON!!\s+cards?\s+(?:on\s+your\s+field\s+)?are\s+returned",
            re.IGNORECASE,
        ),
        lambda m: {"n": int(m.group(1)), "trigger": "on_don_return"},
        confidence=0.9,
    ),
]

# ── Tous les groupes de règles dans l'ordre de priorité ───────────────────
_ALL_RULES: list[_Rule] = (
    _RULES_SPECIAL
    + _RULES_RESTRICTION
    + _RULES_DRAW
    + _RULES_LOOK
    + _RULES_SEARCH
    + _RULES_TRASH
    + _RULES_PLAY
    + _RULES_PLAY_HAND
    + _RULES_HAND
    + _RULES_DON
    + _RULES_DON_ADD
    + _RULES_DON_ACTIVE
    + _RULES_BOOST
    + _RULES_POWER_EXT
    + _RULES_FIELD
    + _RULES_REDUCE
    + _RULES_KO_EXT
    + _RULES_KO
    + _RULES_BANISH
    + _RULES_REST
    + _RULES_LIFE
    + _RULES_PROTECT
    + _RULES_DECK_PLACE
)


# ══════════════════════════════════════════════════════════════════════════
# Extraction des sous-clauses "You may [cost] : [effect]"
# ══════════════════════════════════════════════════════════════════════════

# Coût avant « : » puis effet. Inclut « add » (Zeus : vie → main), pas seulement trash/rest/…
_RE_COST_COLON = re.compile(
    r'(?P<cost>(?:you\s+may\s+)?(?:trash|rest|return|discard|remove|place|add)\s[^:]{1,240}?)\s*:\s+'
    r'(?P<effect>.+?)(?=$|\n|\[(?:On|Activate|When|Trigger|Counter|Main))',
    re.IGNORECASE | re.DOTALL,
)


# ── Découpe des clauses composées ─────────────────────────────────────────

_RE_BULLET       = re.compile(r'[•·]\s*')
_RE_THEN         = re.compile(r'(?<=\.)\s+Then,?\s+', re.IGNORECASE)
_RE_PREAMBLE     = re.compile(
    r'^(?:Apply\s+each\s+of\s+the\s+following|'
    r'Choose\s+one|Your\s+opponent\s+chooses?\s+one)'
    r'[^:•]*:?\s*',
    re.IGNORECASE,
)
_RE_SEMICOLON    = re.compile(r'\s*;\s+(?=[A-Z])')  # ; puis majuscule


def _split_compound_text(text: str) -> list[str]:
    """
    Découpe un texte OPTCG complexe en clauses atomiques :

    1. Bullet points (•) avec préambule optionnel → préambule préservé comme
       contexte de condition attaché à chaque bullet
    2. "Then," → effets séquentiels distincts
    3. ";" entre clauses distinctes

    Retourne la liste des clauses atomiques, ou [text] si non applicable.
    """
    # ── Bullet points ──────────────────────────────────────────────────────
    if _RE_BULLET.search(text):
        preamble_m = _RE_PREAMBLE.match(text)
        preamble   = preamble_m.group(0).rstrip(": ") if preamble_m else ""
        rest       = text[preamble_m.end():] if preamble_m else text
        parts      = [p.strip() for p in _RE_BULLET.split(rest) if p.strip()]

        clauses = []
        for part in parts:
            # Attache le préambule comme contexte si utile
            if preamble and not re.search(r'\bif\b', part, re.IGNORECASE):
                clauses.append(preamble + ": " + part)
            else:
                clauses.append(part)
        return clauses if clauses else [text]

    # ── "Then," — effets séquentiels ──────────────────────────────────────
    then_parts = _RE_THEN.split(text)
    if len(then_parts) > 1:
        return [p.strip() for p in then_parts if p.strip()]

    # ── Semicolon entre clauses ────────────────────────────────────────────
    semi_parts = _RE_SEMICOLON.split(text)
    if len(semi_parts) > 1:
        return [p.strip() for p in semi_parts if p.strip()]

    return [text]


def _split_cost_effect(text: str) -> list[tuple[str, str]]:
    """
    Découpe une clause "you may [cost] : [effect]" en paires (cost, effect_text).
    Retourne [(cost, effect)] ou [] si pas de coût.
    """
    results = []
    for m in _RE_COST_COLON.finditer(text):
        results.append((m.group("cost").strip(), m.group("effect").strip()))
    return results


# ══════════════════════════════════════════════════════════════════════════
# Classifieur principal
# ══════════════════════════════════════════════════════════════════════════

class EffectClassifier:
    """
    Modèle 3 — Classification exhaustive des effets.

    Requiert que TimingModel ait déjà rempli card.timing_segments.
    Ajoute des ParsedEffect dans card.effects.
    """

    name = "EffectClassifier"

    def parse(self, card: ParsedCard) -> None:
        for timing_label, clause_text in card.timing_segments.items():
            for raw_sub in clause_text.split(" | "):
                raw_sub = raw_sub.strip()
                if not raw_sub:
                    continue
                # Expand les clauses composées (bullets, Then, semicolons)
                for atomic in _split_compound_text(raw_sub):
                    if atomic.strip():
                        self._classify_clause(timing_label, atomic.strip(), card)

    def _classify_clause(self, timing: str, text: str, card: ParsedCard) -> None:
        text = _norm(text)

        # ── Détection des coûts d'activation ──────────────────────────────
        cost_pairs = _split_cost_effect(text)
        activation_cost = ""
        if cost_pairs:
            activation_cost = cost_pairs[0][0]
            # L'effet réel est la partie après ":"
            effect_text = cost_pairs[0][1]
        else:
            effect_text = text

        # ── Extraction des champs globaux ──────────────────────────────────
        condition = _extract_condition(text)
        duration  = _extract_duration(effect_text)
        optional  = _is_optional(text)

        matched_any = False

        # Filtre les modificateurs purs qui ne constituent pas un effet
        _MODIFIER_ONLY = re.compile(
            r'^\s*(?:\[Once\s+Per\s+Turn\]|'
            r'\[Your\s+Turn\]|'
            r'\[Opponent\'?s?\s+Turn\]|'
            r'\[DON!!\s*x\d+\])\s*$',
            re.IGNORECASE,
        )
        if _MODIFIER_ONLY.match(effect_text):
            return

        for rule in _ALL_RULES:
            m = rule.pattern.search(effect_text)
            if not m:
                continue

            params  = rule.extractor(m)
            target  = _extract_target(effect_text, hint_owner=rule.target_hint)

            # Déduplication : évite de créer deux effets identiques
            # (ex: power_reduce matchant 2 règles sur le même texte)
            # Une règle a matché → on considère le segment comme classifié
            # même si l'effet est un doublon (pour ne pas générer UNKNOWN)
            matched_any = True

            is_dup = any(
                e.timing == timing
                and e.effect_type == rule.effect_type
                and e.params.get("amount") == params.get("amount")
                and e.params.get("n") == params.get("n")
                for e in card.effects
                if e.raw_text == text[:200]
            )
            if is_dup:
                continue

            card.effects.append(ParsedEffect(
                timing=timing,
                effect_type=rule.effect_type,
                raw_text=text[:200],
                params=params,
                target=target,
                condition=condition,
                duration=duration,
                optional=optional,
                activation_cost=activation_cost,
                confidence=rule.confidence,
            ))

        if not matched_any and effect_text.strip():
            card.effects.append(ParsedEffect(
                timing=timing,
                effect_type=EffectType.UNKNOWN,
                raw_text=text[:200],
                condition=condition,
                duration=duration,
                optional=optional,
                activation_cost=activation_cost,
                confidence=0.0,
            ))
