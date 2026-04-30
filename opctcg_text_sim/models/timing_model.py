"""
TimingModel — Modèle 2 : extraction et segmentation des timings.

Découpe le texte de la carte en segments nommés selon leur balise :
  [On Play], [Main], [Trigger], [When Attacking], [On K.O.],
  [Activate:Main], [Counter], [On Your Opponent's Attack],
  [Your Turn], [End of Turn], [End of Your Turn], [Start of Turn],
  [On Play]/[When Attacking] (dual-timing → deux segments distincts)

Gère aussi les coûts DON!! -N : "DON!! -N (explanation) : effect"
"""
from __future__ import annotations

import re

from .effect_models import ParsedCard


# ── Timings officiels — du plus long au plus court ──────────────────────────
_TIMING_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("On Your Opponent's Attack",
        re.compile(r"\[On\s+Your\s+Opponent'?s?\s+Attack\]\s*:?\s*", re.IGNORECASE)),
    ("Activate:Main",
        re.compile(r"\[Activate\s*:\s*Main\]\s*:?\s*", re.IGNORECASE)),
    ("When Attacking",
        re.compile(r"\[When\s+Attacking\]\s*:?\s*", re.IGNORECASE)),
    ("End of Your Turn",
        re.compile(r"\[End\s+of\s+Your\s+Turn\]\s*:?\s*", re.IGNORECASE)),
    ("End of Turn",
        re.compile(r"\[End\s+of\s+(?:the\s+)?Turn\]\s*:?\s*", re.IGNORECASE)),
    ("Start of Your Turn",
        re.compile(r"\[Start\s+of\s+Your\s+Turn\]\s*:?\s*", re.IGNORECASE)),
    ("Start of Turn",
        re.compile(r"\[Start\s+of\s+Turn\]\s*:?\s*", re.IGNORECASE)),
    ("On K.O.",
        re.compile(r"\[On\s*K\.?O\.?\]\s*:?\s*", re.IGNORECASE)),
    ("On Play",
        re.compile(r"\[On\s*Play\]\s*:?\s*", re.IGNORECASE)),
    ("Trigger",
        re.compile(r"\[Trigger\]\s*:?\s*", re.IGNORECASE)),
    ("Counter",
        re.compile(r"(?:^|\n)\s*\[Counter\]\s*:?\s*", re.IGNORECASE)),
    ("Blocker",
        re.compile(r"\[Blocker\]\s*\([^)]*\)\s*", re.IGNORECASE)),
    ("Main",
        re.compile(r"\[Main\]\s*:?\s*", re.IGNORECASE)),
    ("Your Turn",
        re.compile(r"\[Your\s+Turn\]\s*:?\s*", re.IGNORECASE)),
    ("Opponent's Turn",
        re.compile(r"\[Opponent'?s?\s+Turn\]\s*:?\s*", re.IGNORECASE)),
    ("When DON!! Returned",
        re.compile(r"When\s+\d+\s+or\s+more\s+DON!!\s+cards?\s+(?:on\s+your\s+field\s+)?are\s+returned",
                   re.IGNORECASE)),
]

# Regex pour les dual-timings "[A]/[B]" ou "[A] [B]"
_DUAL_TIMING = re.compile(
    r'\[On\s*Play\]\s*/?\s*\[When\s+Attacking\]'
    r'|\[When\s+Attacking\]\s*/?\s*\[On\s+Play\]',
    re.IGNORECASE,
)

# Coût DON!! -N — "(explanation)" optionnel puis ":"
_DON_COST = re.compile(
    r'DON!!\s*-\s*(\d+)'
    r'(?:\s*\([^)]{0,200}?\))?'  # description optionnelle entre parenthèses
    r'\s*:\s*',
    re.IGNORECASE,
)

# Protège "Activate this card's [X] effect"
_SELF_TRIGGER = re.compile(
    r"Activate\s+this\s+card'?s?\s+\[[^\]]+\]\s+effect\.?",
    re.IGNORECASE,
)

# Protège "[Keyword]" quand il référence une capacité de carte (pas un timing)
# ex: "a card with a [Trigger]" / "and a [Rush] ability"
_EMBEDDED_KW_REF = re.compile(
    r'(?<=\s)((?:with|and|having)\s+a\s*)(\[(?:Trigger|Rush|Double\s*Attack|Blocker|Banish|Infiltrate|Unblockable)\])',
    re.IGNORECASE,
)

# [DON!! xN] — géré par KeywordModel
_DON_TAGS = re.compile(r'\s*\[DON!!\s*x\d+\]\s*', re.IGNORECASE)

# Descriptions officielles des mots-clés (à supprimer des segments)
_KEYWORD_DESCRIPTIONS = re.compile(
    r'\s*\((?:'
    r'This card can attack on the turn[^)]*'
    r'|After your opponent declares an attack[^)]*'
    r'|When this card deals damage, the target card is trashed[^)]*'
    r'|Characters cannot become the target[^)]*'
    r'|This card can attack your opponent\'?s? Life cards?[^)]*'
    r'|If your opponent has the fewest[^)]*'
    r'|You may return the specified number of DON!![^)]*'
    r')\)',
    re.IGNORECASE,
)

# Tags passifs en début de segment seulement
_PASSIVE_KW_TAGS = re.compile(
    r'^(?:\s*\[(?:Rush|Blocker|Double\s*Attack|Banish|Unblockable|Infiltrate)\]\s*)+',
    re.IGNORECASE | re.MULTILINE,
)


def _norm(s: str) -> str:
    return (
        str(s)
        .replace("\u2019", "'").replace("\u2018", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
    )


def _split_into_segments(text: str) -> list[tuple[str, str]]:
    """
    Retourne une liste de (timing_label, clause_text).
    Gère les dual-timings [A]/[B] en créant deux entrées avec le même texte.
    """
    # Remplace les dual-timings par le premier timing (le second sera ajouté ci-après)
    dual_found: list[tuple[int, int, list[str]]] = []
    for m in _DUAL_TIMING.finditer(text):
        labels = ["On Play", "When Attacking"]
        dual_found.append((m.start(), m.end(), labels))

    segments: list[tuple[str, str]] = []
    found: list[tuple[int, int, str]] = []

    for start, end, labels in dual_found:
        found.append((start, end, labels[0]))  # le premier timing

    for label, pat in _TIMING_PATTERNS:
        for m in pat.finditer(text):
            # Évite les overlaps avec dual timings
            overlap = any(s <= m.start() < e for s, e, _ in dual_found)
            if not overlap:
                found.append((m.start(), m.end(), label))

    if not found:
        stripped = text.strip()
        if stripped:
            segments.append(("Passive", stripped))
        return segments

    found.sort(key=lambda x: x[0])

    pre = text[: found[0][0]].strip()
    if pre:
        segments.append(("Passive", pre))

    for i, (start, end, label) in enumerate(found):
        next_start = found[i + 1][0] if i + 1 < len(found) else len(text)
        clause = text[end:next_start].strip()
        segments.append((label, clause))

        # Dual timing : ajoute le second label avec le même texte
        for ds, de, dlabels in dual_found:
            if ds == start and len(dlabels) > 1:
                segments.append((dlabels[1], clause))

    return segments


def _clean_clause(clause: str) -> str:
    """Nettoie un segment de ses balises parasites."""
    clause = _KEYWORD_DESCRIPTIONS.sub("", clause)
    clause = _PASSIVE_KW_TAGS.sub("", clause)
    clause = _DON_TAGS.sub("", clause)
    # Normalise les coûts DON!! -N en ajoutant un marqueur clair
    clause = _DON_COST.sub(lambda m: f"[DON COST -{m.group(1)}] ", clause)
    clause = clause.replace("<<", "[").replace(">>", "]").strip()
    return clause


class TimingModel:
    """
    Modèle 2 — Segmentation du texte par timing.

    - Gère les dual-timings [A]/[B] → deux segments distincts
    - Normalise les coûts DON!! -N en balise [DON COST -N]
    - Supprime les descriptions de mots-clés officiels
    """

    name = "TimingModel"

    def parse(self, card: ParsedCard) -> None:
        text = card.card_text or ""
        if not text.strip():
            return

        text = _norm(text)
        # Protège les références de mots-clés embarqués AVANT la segmentation
        text = _EMBEDDED_KW_REF.sub(
            lambda m: m.group(1) + m.group(2).replace("[", "<<").replace("]", ">>"), text
        )
        text = _SELF_TRIGGER.sub(
            lambda m: m.group(0).replace("[", "<<").replace("]", ">>"), text
        )

        for label, clause in _split_into_segments(text):
            clause = _clean_clause(clause)
            if not clause:
                continue
            if label in card.timing_segments:
                card.timing_segments[label] += " | " + clause
            else:
                card.timing_segments[label] = clause
