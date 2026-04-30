"""
KeywordModel — Modèle 1 : détection exhaustive des mots-clés OPTCG.

Détecte :
  - Mots-clés passifs (Rush, Blocker, Double Attack, Banish, Unblockable, Infiltrate)
  - Balises de timing ([On Play], [Trigger], [Activate:Main], …)
  - Conditions DON!! (DON!! x1, DON!! x2, …)
  - Clauses spéciales ([Once Per Turn], [Counter], [When Attacking], …)
"""
from __future__ import annotations

import re

from .effect_models import EffectType, ParsedCard, ParsedEffect, Target, Timing


# ── Mots-clés et balises officiels ────────────────────────────────────────

_KEYWORDS: list[tuple[str, re.Pattern]] = [
    # Mots-clés passifs
    ("Rush",                re.compile(r"\[Rush\]", re.IGNORECASE)),
    ("Blocker",             re.compile(r"\[Blocker\]", re.IGNORECASE)),
    ("Double Attack",       re.compile(r"\[Double\s*Attack\]", re.IGNORECASE)),
    ("Banish",              re.compile(r"\[Banish\]", re.IGNORECASE)),
    ("Unblockable",         re.compile(r"\[Unblockable\]", re.IGNORECASE)),
    ("Infiltrate",          re.compile(r"\[Infiltrate\]", re.IGNORECASE)),

    # Balises de timing
    ("On Play",             re.compile(r"\[On\s*Play\]", re.IGNORECASE)),
    ("Trigger",             re.compile(r"\[Trigger\]", re.IGNORECASE)),
    ("On K.O.",             re.compile(r"\[On\s*K\.?O\.?\]", re.IGNORECASE)),
    ("Activate:Main",       re.compile(r"\[Activate\s*:\s*Main\]", re.IGNORECASE)),
    ("When Attacking",      re.compile(r"\[When\s*Attacking\]", re.IGNORECASE)),
    ("Counter",             re.compile(r"\[Counter\]", re.IGNORECASE)),
    ("Once Per Turn",       re.compile(r"\[Once\s*Per\s*Turn\]", re.IGNORECASE)),
    ("On Your Opponent's Attack",
                            re.compile(r"\[On\s*Your\s*Opponent'?s?\s*Attack\]", re.IGNORECASE)),

    # Mots-clés donnés à une autre carte
    ("Gains Rush",          re.compile(r"gains?\s+\[Rush\]", re.IGNORECASE)),
    ("Gains Blocker",       re.compile(r"gains?\s+\[Blocker\]", re.IGNORECASE)),
    ("Gains Double Attack", re.compile(r"gains?\s+\[Double\s*Attack\]", re.IGNORECASE)),
    ("Gains Banish",        re.compile(r"gains?\s+\[Banish\]", re.IGNORECASE)),
    ("Gains Unblockable",   re.compile(r"gains?\s+\[Unblockable\]", re.IGNORECASE)),
    ("Gains Infiltrate",    re.compile(r"gains?\s+\[Infiltrate\]", re.IGNORECASE)),

    # Clauses de protection / état
    ("K.O. Protection",     re.compile(r"would\s+be\s+K\.?O\.?'?d", re.IGNORECASE)),
    ("Cannot Attack",       re.compile(r"cannot\s+attack", re.IGNORECASE)),
    ("Cannot Block",        re.compile(r"cannot\s+(?:be\s+)?block(?:ed)?", re.IGNORECASE)),
    ("Cannot Target",       re.compile(r"cannot\s+(?:be\s+)?(?:the\s+)?target", re.IGNORECASE)),
    ("Cannot Use Effects",  re.compile(r"cannot\s+activate\s+(?:its\s+)?effects?", re.IGNORECASE)),
]

# Mots-clés passifs → EffectType
_KW_TO_EFFECT: dict[str, EffectType] = {
    "Rush":          EffectType.KEYWORD_RUSH,
    "Blocker":       EffectType.KEYWORD_BLOCKER,
    "Double Attack": EffectType.KEYWORD_DOUBLE_ATTACK,
    "Banish":        EffectType.KEYWORD_BANISH,
    "Unblockable":   EffectType.KEYWORD_UNBLOCKABLE,
    "Infiltrate":    EffectType.KEYWORD_INFILTRATE,
}

# DON!! x<n>
_RE_DON_COND = re.compile(r"\[DON!!\s*x(\d+)\]", re.IGNORECASE)

# Texte de description standard des mots-clés (parenthèses officielles)
_KW_DESCRIPTIONS: dict[str, str] = {
    "Rush":          "(This card can attack on the turn it is played.)",
    "Blocker":       "(When your opponent attacks, you may rest this card to redirect the attack.)",
    "Double Attack": "(This card deals 2 damage when it attacks.)",
    "Banish":        "(Damage dealt by this card sends the target to trash without triggering Trigger.)",
    "Unblockable":   "(Characters cannot become the target of this card's attacks.)",
    "Infiltrate":    "(This card can attack your opponent's Life cards directly.)",
}


class KeywordModel:
    """
    Modèle 1 — Extraction exhaustive des mots-clés et balises.

    - Remplit card.keywords avec tous les mots-clés détectés
    - Ajoute des ParsedEffect pour les mots-clés passifs
    - Ajoute des ParsedEffect pour les conditions DON!!
    """

    name = "KeywordModel"

    def parse(self, card: ParsedCard) -> None:
        text = card.card_text or ""

        # ── Détection des mots-clés ────────────────────────────────────────
        keywords_found: list[str] = []
        for kw_name, pattern in _KEYWORDS:
            if pattern.search(text):
                keywords_found.append(kw_name)
        card.keywords = keywords_found

        # ── Effets pour mots-clés passifs ──────────────────────────────────
        for kw in keywords_found:
            if kw in _KW_TO_EFFECT:
                card.effects.append(ParsedEffect(
                    timing=Timing.PASSIVE.value,
                    effect_type=_KW_TO_EFFECT[kw],
                    raw_text=f"[{kw}]",
                    params={"description": _KW_DESCRIPTIONS.get(kw, "")},
                    confidence=1.0,
                ))

        # ── Effets pour mots-clés "gagne [X]" ────────────────────────────
        _GAINS_MAP: dict[str, EffectType] = {
            "Gains Rush":          EffectType.KEYWORD_RUSH,
            "Gains Blocker":       EffectType.KEYWORD_BLOCKER,
            "Gains Double Attack": EffectType.KEYWORD_DOUBLE_ATTACK,
            "Gains Banish":        EffectType.KEYWORD_BANISH,
            "Gains Unblockable":   EffectType.KEYWORD_UNBLOCKABLE,
            "Gains Infiltrate":    EffectType.KEYWORD_INFILTRATE,
        }
        for kw in keywords_found:
            if kw in _GAINS_MAP:
                card.effects.append(ParsedEffect(
                    timing=Timing.UNKNOWN.value,
                    effect_type=EffectType.GIVE_KEYWORD,
                    raw_text=kw,
                    params={"keyword": kw.replace("Gains ", "")},
                    confidence=0.95,
                ))

        # ── Conditions DON!! ───────────────────────────────────────────────
        for m in _RE_DON_COND.finditer(text):
            n = int(m.group(1))
            rest = text[m.end():].strip()
            # Trouve la fin de la clause (prochain "[" ou fin)
            next_bracket = re.search(r"\[", rest)
            clause = (rest[:next_bracket.start()].strip()
                      if next_bracket else rest[:150].strip())
            card.effects.append(ParsedEffect(
                timing=Timing.DON_CONDITION.value,
                effect_type=EffectType.CONDITION,
                raw_text=m.group(0) + " " + clause,
                params={"don_required": n, "clause": clause},
                confidence=0.95,
            ))
