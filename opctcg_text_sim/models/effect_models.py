"""
Dataclasses représentant les effets de cartes OPTCG de façon structurée.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Timing(str, Enum):
    ON_PLAY          = "On Play"
    ACTIVATE_MAIN    = "Activate:Main"
    WHEN_ATTACKING   = "When Attacking"
    TRIGGER          = "Trigger"
    ON_KO            = "On K.O."
    COUNTER          = "Counter"
    BLOCKER          = "Blocker"
    OPPONENT_ATTACK  = "On Your Opponent's Attack"
    ONCE_PER_TURN    = "Once Per Turn"
    DON_CONDITION    = "DON!! Condition"
    PASSIVE          = "Passive"
    UNKNOWN          = "Unknown"


class EffectType(str, Enum):
    # ── Pioche / deck ──────────────────────────────────────────────────────
    DRAW             = "draw"
    LOOK_TOP         = "look_top"
    REVEAL_TOP       = "reveal_top"

    # ── Recherche ──────────────────────────────────────────────────────────
    SEARCH_DECK      = "search_deck"

    # ── Défausse / trash ───────────────────────────────────────────────────
    TRASH_FROM_HAND  = "trash_from_hand"
    TRASH_FROM_DECK  = "trash_from_deck"
    TRASH_FROM_LIFE  = "trash_from_life"

    # ── Jouer des cartes ───────────────────────────────────────────────────
    PLAY_FROM_TRASH  = "play_from_trash"
    PLAY_FROM_DECK   = "play_from_deck"
    PLAY_FROM_HAND   = "play_from_hand"

    # ── Main / retour en main ──────────────────────────────────────────────
    RETURN_TO_HAND   = "return_to_hand"
    ADD_TO_HAND      = "add_to_hand"

    # ── DON!! ──────────────────────────────────────────────────────────────
    DON_ATTACH       = "don_attach"
    DON_ACTIVATE     = "don_activate"
    DON_REST         = "don_rest"
    DON_RETURN       = "don_return"

    # ── Puissance ──────────────────────────────────────────────────────────
    POWER_BOOST      = "power_boost"
    POWER_REDUCE     = "power_reduce"

    # ── Vie ────────────────────────────────────────────────────────────────
    LIFE_ADD         = "life_add"
    LIFE_REMOVE      = "life_remove"

    # ── K.O. / état ────────────────────────────────────────────────────────
    KO               = "ko"
    BANISH           = "banish"
    REST_TARGET      = "rest_target"
    ACTIVATE_TARGET  = "activate_target"

    # ── Mots-clés passifs ──────────────────────────────────────────────────
    KEYWORD_RUSH          = "keyword_rush"
    KEYWORD_BLOCKER       = "keyword_blocker"
    KEYWORD_DOUBLE_ATTACK = "keyword_double_attack"
    KEYWORD_BANISH        = "keyword_banish"
    KEYWORD_UNBLOCKABLE   = "keyword_unblockable"
    KEYWORD_INFILTRATE    = "keyword_infiltrate"

    # ── DON!! étendu ───────────────────────────────────────────────────────
    DON_ADD          = "don_add"        # ajouter un DON!! depuis le deck DON!!

    # ── Puissance / coût ───────────────────────────────────────────────────
    SET_POWER        = "set_power"      # puissance de base devient N
    COST_REDUCE      = "cost_reduce"    # -N coût sur une carte adverse

    # ── Protection / état ──────────────────────────────────────────────────
    CANNOT_KO        = "cannot_ko"      # ne peut pas être K.O. ce combat
    SACRIFICE_PROTECT= "sacrifice_protect"  # trash self instead of KO

    # ── Placement dans le deck ─────────────────────────────────────────────
    PLACE_TO_DECK    = "place_to_deck"  # placer au bas / haut du deck

    # ── Auto-déclenchement ─────────────────────────────────────────────────
    ACTIVATE_SELF_EFFECT = "activate_self_effect"  # "Activate this card's [Main]"

    # ── Activer une carte perso ────────────────────────────────────────────
    SET_ACTIVE       = "set_active"   # "set a Character card as active"

    # ── Texte de règle spéciale ────────────────────────────────────────────
    SPECIAL_RULE     = "special_rule"

    # ── Divers ─────────────────────────────────────────────────────────────
    GIVE_KEYWORD     = "give_keyword"
    CONDITION        = "condition"
    UNKNOWN          = "unknown"


@dataclass
class Target:
    """Description structurée d'une cible d'effet."""
    owner: str = ""              # "you", "opponent", "self", "both"
    card_type: str = ""          # "Leader", "Character", "Event", "Stage",
                                 # "Leader or Character", "any"
    count: int = 1               # nombre de cibles (0 = "all")
    up_to: bool = False          # "up to N" vs exactement N
    type_tag: str = ""           # "[Straw Hat Crew]" type
    cost_max: int = -1           # -1 = pas de filtre
    cost_min: int = -1
    power_max: int = -1
    power_min: int = -1
    color: str = ""
    name_contains: str = ""      # filtre sur le nom de la carte
    rested: bool = False         # cible doit être au repos

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()
             if v not in ("", -1, False, 1) or k in ("owner", "card_type")}
        return d

    def is_empty(self) -> bool:
        """True si aucun champ significatif n'a été rempli."""
        return not any([self.owner, self.card_type, self.type_tag, self.name_contains,
                        self.cost_max >= 0, self.cost_min >= 0, self.power_max >= 0,
                        self.power_min >= 0, self.color, self.rested, self.count != 1])

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Target:
        if not d:
            return cls()

        def _i(key: str, default: int = -1) -> int:
            if key not in d:
                return default
            try:
                return int(d[key])
            except (TypeError, ValueError):
                return default

        return cls(
            owner=str(d.get("owner", "") or ""),
            card_type=str(d.get("card_type", "") or ""),
            count=int(d.get("count", 1) or 1),
            up_to=bool(d.get("up_to", False)),
            type_tag=str(d.get("type_tag", "") or ""),
            cost_max=_i("cost_max"),
            cost_min=_i("cost_min"),
            power_max=_i("power_max"),
            power_min=_i("power_min"),
            color=str(d.get("color", "") or ""),
            name_contains=str(d.get("name_contains", "") or ""),
            rested=bool(d.get("rested", False)),
        )

    def pretty(self) -> str:
        if self.is_empty():
            return ""
        parts = []
        if self.owner:
            parts.append(self.owner)
        count_str = f"up to {self.count}" if self.up_to else str(self.count)
        parts.append(count_str)
        if self.type_tag:
            parts.append(f"[{self.type_tag}]")
        if self.card_type:
            parts.append(self.card_type)
        if self.cost_max >= 0:
            parts.append(f"cost≤{self.cost_max}")
        if self.cost_min >= 0:
            parts.append(f"cost≥{self.cost_min}")
        if self.power_max >= 0:
            parts.append(f"pow≤{self.power_max}")
        if self.color:
            parts.append(f"color={self.color}")
        if self.rested:
            parts.append("rested")
        return " ".join(parts)


@dataclass
class ParsedEffect:
    """Un effet atomique parsé depuis le texte d'une carte."""

    timing: str
    effect_type: EffectType
    raw_text: str

    # Paramètres spécifiques à l'effet (n, amount, source, …)
    params: dict[str, Any] = field(default_factory=dict)

    # Cible structurée
    target: Target = field(default_factory=Target)

    # Clause conditionnelle : "if your opponent has 2 or fewer Life cards"
    condition: str = ""

    # Durée : "this_turn", "until_next_turn", "battle", "permanent", "unknown"
    duration: str = "unknown"

    # Effet optionnel précédé de "You may"
    optional: bool = False

    # Coût d'activation : "trash 1 card from hand", "rest this Character"
    activation_cost: str = ""

    # Confiance du modèle [0..1]
    confidence: float = 1.0

    # ── Sérialisation ──────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "timing":          self.timing,
            "effect_type":     self.effect_type.value,
            "raw_text":        self.raw_text,
            "params":          self.params,
            "target":          self.target.to_dict(),
            "condition":       self.condition,
            "duration":        self.duration,
            "optional":        self.optional,
            "activation_cost": self.activation_cost,
            "confidence":      round(self.confidence, 3),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParsedEffect:
        et_raw = d.get("effect_type", "unknown")
        try:
            et = EffectType(et_raw)
        except ValueError:
            et = EffectType.UNKNOWN
        return cls(
            timing=str(d.get("timing", "")),
            effect_type=et,
            raw_text=str(d.get("raw_text", "")),
            params=dict(d.get("params") or {}),
            target=Target.from_dict(d.get("target") or {}),
            condition=str(d.get("condition", "")),
            duration=str(d.get("duration", "unknown")),
            optional=bool(d.get("optional", False)),
            activation_cost=str(d.get("activation_cost", "")),
            confidence=float(d.get("confidence", 1.0)),
        )

    def pretty(self) -> str:
        parts = [f"  [{self.timing}] → {self.effect_type.value}"]

        # Paramètres principaux
        p_parts = []
        for k, v in self.params.items():
            p_parts.append(f"{k}={v}")
        if p_parts:
            parts.append(f"({', '.join(p_parts)})")

        # Cible
        tgt = self.target.pretty()
        if tgt:
            parts.append(f"  target: {tgt}")

        # Condition
        if self.condition:
            parts.append(f"  if: {self.condition}")

        # Durée
        if self.duration not in ("unknown", "permanent"):
            parts.append(f"  dur: {self.duration}")

        # Optionnel
        if self.optional:
            parts.append("  [optional]")

        # Coût d'activation
        if self.activation_cost:
            parts.append(f"  cost: {self.activation_cost}")

        # Confiance
        if self.confidence < 1.0:
            stars = "★" * round(self.confidence * 3) + "☆" * (3 - round(self.confidence * 3))
            parts.append(f"  [{stars} {self.confidence:.0%}]")

        return "  ".join(parts)


@dataclass
class ParsedCard:
    """Représentation structurée d'une carte et de ses effets."""

    card_id:    str
    name:       str
    cost:       int
    power:      int
    counter:    int
    color:      str
    rarity:     str
    card_type:  str
    life:       int
    card_text:  str
    group_name: str = ""

    # Résultats des modèles
    keywords:        list[str]               = field(default_factory=list)
    timing_segments: dict[str, str]          = field(default_factory=dict)
    effects:         list[ParsedEffect]      = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "card_id":         self.card_id,
            "name":            self.name,
            "cost":            self.cost,
            "power":           self.power,
            "counter":         self.counter,
            "color":           self.color,
            "rarity":          self.rarity,
            "card_type":       self.card_type,
            "life":            self.life,
            "card_text":       self.card_text,
            "group_name":      self.group_name,
            "keywords":        self.keywords,
            "timing_segments": self.timing_segments,
            "effects":         [e.to_dict() for e in self.effects],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ParsedCard:
        return cls(
            card_id=str(d.get("card_id", "")),
            name=str(d.get("name", "")),
            cost=int(d.get("cost", 0) or 0),
            power=int(d.get("power", 0) or 0),
            counter=int(d.get("counter", 0) or 0),
            color=str(d.get("color", "")),
            rarity=str(d.get("rarity", "")),
            card_type=str(d.get("card_type", "")),
            life=int(d.get("life", 0) or 0),
            card_text=str(d.get("card_text", "")),
            group_name=str(d.get("group_name", "")),
            keywords=list(d.get("keywords") or []),
            timing_segments=dict(d.get("timing_segments") or {}),
            effects=[ParsedEffect.from_dict(x) for x in (d.get("effects") or [])],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
