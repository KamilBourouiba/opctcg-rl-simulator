"""
EffectResolver — traduit les ParsedEffect (pipeline models/) en mutations de GameState.

Ce module est le pont entre le parsing structuré (models/) et l'exécution simulée.
Il complète on_play_resolver.py (regex brute) en gérant :
  - Tous les timings ([On Play], [On K.O.], [When Attacking], [Trigger], etc.)
  - Les effets structurés produits par EffectClassifier
  - Les conditions évaluées contre l'état de jeu courant
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..effect_unique import rule_text_fingerprint
from ..models import EffectClassifier, KeywordModel, ParsedCard, TimingModel
from ..models.effect_models import EffectType, ParsedEffect

if TYPE_CHECKING:
    from ..card_db import CardDef
    from ..simulator import BoardChar, PlayerState, SimplifiedOPSim
    from .keyword_handlers import EffectContext


@dataclass
class _SharedParsedSlice:
    """Sous-partie d'un ParsedCard mutualisée entre cartes au même ``rule_text_fingerprint``."""

    keywords: list[str]
    timing_segments: dict[str, str]
    effects: list["ParsedEffect"]


def _return_attached_don_to_cost(pl: "PlayerState", bc: "BoardChar") -> None:
    """CR 3-4-2 : DON!! attachés reviennent en zone de coût (reposés) quand le perso quitte le terrain."""
    if bc.attached_don:
        pl.don_rested += bc.attached_don
        bc.attached_don = 0


# ══════════════════════════════════════════════════════════════════════════════
# Cache de ParsedCard
# ══════════════════════════════════════════════════════════════════════════════

class CardEffectCache:
    """
    Pré-calcule et met en cache les ParsedCard (effets structurés) pour chaque CardDef.
    Initialiser une seule fois par simulateur ; le parsing est CPU-léger mais non nul.

    Si ``dedupe_identical_rule_text`` est vrai, les listes ``effects`` / ``keywords`` /
    ``timing_segments`` identiques pour un même texte de règle + type de carte sont
    partagées entre plusieurs ``ParsedCard`` (métadonnées ``card_id`` / coût restent propres).

    Si ``snapshot_path`` pointe vers un JSON produit par ``export_effect_snapshot_json``,
    les entrées présentes dans le fichier évitent le reparsing (métadonnées CSV recollées
    à la volée pour rester cohérentes avec le deck).
    """

    def __init__(
        self,
        *,
        dedupe_identical_rule_text: bool = False,
        snapshot_path: str | Path | None = None,
    ) -> None:
        self._dedupe = dedupe_identical_rule_text
        self._snapshot_path = Path(snapshot_path) if snapshot_path else None
        self._snapshot_index: dict[str, dict[str, Any]] | None = None
        self._cache: dict[str, ParsedCard] = {}
        self._shared_by_fp: dict[str, _SharedParsedSlice] = {}
        self._kw = KeywordModel()
        self._tm = TimingModel()
        self._ec = EffectClassifier()

    def _ensure_snapshot_loaded(self) -> None:
        if self._snapshot_path is None or self._snapshot_index is not None:
            return
        from ..effect_snapshot import load_snapshot_index

        self._snapshot_index = load_snapshot_index(self._snapshot_path)

    @property
    def shared_fingerprint_entries(self) -> int:
        """Nombre d'empreintes distinctes réutilisées (0 si dédup désactivée)."""
        return len(self._shared_by_fp)

    def get(self, card: "CardDef") -> ParsedCard:
        if card.card_id in self._cache:
            return self._cache[card.card_id]
        self._ensure_snapshot_loaded()
        if self._snapshot_index and card.card_id in self._snapshot_index:
            from ..effect_snapshot import overlay_parsed_card_from_live_csv

            pc = ParsedCard.from_dict(self._snapshot_index[card.card_id])
            overlay_parsed_card_from_live_csv(pc, card)
            self._cache[card.card_id] = pc
            return pc
        fp = rule_text_fingerprint(card)
        if self._dedupe and fp in self._shared_by_fp:
            sh = self._shared_by_fp[fp]
            pc = ParsedCard(
                card_id=card.card_id,
                name=card.name,
                cost=card.cost,
                power=card.power,
                counter=card.counter,
                color=card.color,
                rarity="",
                card_type=card.card_type,
                life=card.life,
                card_text=card.card_text or "",
            )
            pc.keywords = sh.keywords
            pc.timing_segments = sh.timing_segments
            pc.effects = sh.effects
            self._cache[card.card_id] = pc
            return pc

        pc = ParsedCard(
            card_id=card.card_id,
            name=card.name,
            cost=card.cost,
            power=card.power,
            counter=card.counter,
            color=card.color,
            rarity="",
            card_type=card.card_type,
            life=card.life,
            card_text=card.card_text or "",
        )
        self._kw.parse(pc)
        self._tm.parse(pc)
        self._ec.parse(pc)
        if self._dedupe:
            self._shared_by_fp[fp] = _SharedParsedSlice(
                keywords=pc.keywords,
                timing_segments=pc.timing_segments,
                effects=pc.effects,
            )
        self._cache[card.card_id] = pc
        return pc

    def precompute(self, cards: dict[str, "CardDef"]) -> None:
        """Pré-calcule les effets de toutes les cartes (appelé au démarrage)."""
        for card in cards.values():
            if card.card_text:
                self.get(card)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers d'accès à l'état
# ══════════════════════════════════════════════════════════════════════════════

def _opp(ctx: "EffectContext") -> "PlayerState":
    return ctx.sim.p1 if ctx.owner_idx == 0 else ctx.sim.p0


def _draw_n(ctx: "EffectContext", n: int) -> int:
    """Pioche n cartes (via le helper existant dans on_play_resolver)."""
    from .on_play_resolver import _draw_n as _draw
    return _draw(ctx, n)


def _discard_n(ctx: "EffectContext", pl: "PlayerState", n: int) -> int:
    """Défausse n cartes aléatoires d'un joueur."""
    from .on_play_resolver import _discard_n_random
    return _discard_n_random(ctx, pl, n)


def _add_don(ctx: "EffectContext", n: int) -> int:
    """Ajoute n DON!! depuis le deck DON!!."""
    from .on_play_resolver import _add_rested_don_from_deck
    return _add_rested_don_from_deck(ctx, n)


# ══════════════════════════════════════════════════════════════════════════════
# Coûts d'activation (« You may [coût] : [effet] » — texte dans ParsedEffect.activation_cost)
# ══════════════════════════════════════════════════════════════════════════════


def _apply_activation_subcost(sub: str, ctx: "EffectContext") -> bool:
    """
    Applique un fragment de coût (sans « and » entre coûts composés).
    Retourne False si le paiement est impossible (main / vie / board insuffisant).
    """
    sub = re.sub(r"^(?:you\s+may\s+)", "", (sub or "").strip(), flags=re.I).strip()
    if not sub:
        return True
    pl = ctx.owner
    sim = ctx.sim
    my_cid = ctx.card.card_id

    def _idx_this_char() -> int:
        for i, b in enumerate(pl.board):
            if b.card_id == my_cid:
                return i
        return -1

    # trash this Character / card / Leader
    if re.search(r"^trash\s+this\s+(?:character|card|leader)\b", sub, re.I):
        idx = _idx_this_char()
        if idx < 0:
            return False
        bc = pl.board.pop(idx)
        if bc.attached_don:
            pl.don_rested += bc.attached_don
        pl.trash.append(bc.card_id)
        return True

    # rest this Character / Leader / card
    if re.search(r"^rest\s+this\s+(?:character|leader|card)\b", sub, re.I):
        if (pl.leader_id or "") == my_cid:
            pl.leader_rested = True
            return True
        idx = _idx_this_char()
        if idx < 0:
            return False
        pl.board[idx].rested = True
        return True

    # rest N of your DON!! cards (zone de coût : actif → reposé)
    m = re.search(r"^rest\s+(?:up\s+to\s+)?(\d+)\s+of\s+your\s+don", sub, re.I)
    if m:
        n = int(m.group(1))
        if pl.don_active < n:
            return False
        for _ in range(n):
            pl.don_active -= 1
            pl.don_rested += 1
        return True

    # trash N cards from your hand | trash a card from your hand
    m = re.search(r"^trash\s+(?:up\s+to\s+)?(\d+)\s+cards?\s+from\s+your\s+hand", sub, re.I)
    if m:
        n = int(m.group(1))
        if len(pl.hand) < n:
            return False
        _discard_n(ctx, pl, n)
        return True
    if re.search(r"^trash\s+(?:a|1|one)\s+card\s+from\s+your\s+hand", sub, re.I):
        if not pl.hand:
            return False
        _discard_n(ctx, pl, 1)
        return True
    m_disc = re.search(
        r"^discard\s+(?:up\s+to\s+)?(\d+)\s+cards?\s+from\s+your\s+hand", sub, re.I
    )
    if m_disc:
        n = int(m_disc.group(1))
        if len(pl.hand) < n:
            return False
        _discard_n(ctx, pl, n)
        return True
    if re.search(r"^discard\s+(?:a|1|one)\s+card\s+from\s+your\s+hand", sub, re.I):
        if not pl.hand:
            return False
        _discard_n(ctx, pl, 1)
        return True

    # trash N from top of your Life cards
    m = re.search(
        r"^trash\s+(?:up\s+to\s+)?(\d+|a|one)\s+cards?\s+from\s+the\s+top\s+of\s+your\s+life",
        sub, re.I,
    )
    if m:
        raw_n = m.group(1).lower()
        n = 1 if raw_n in ("a", "one") else int(raw_n)
        if len(pl.life_cards) < n:
            return False
        from .on_play_resolver import _trash_life_top
        _trash_life_top(ctx, n)
        return True

    # trash N from top of your deck
    m = re.search(
        r"^trash\s+(?:up\s+to\s+)?(\d+)\s+cards?\s+from\s+the\s+top\s+of\s+your\s+deck",
        sub, re.I,
    )
    if m:
        n = int(m.group(1))
        if len(pl.deck) < n:
            return False
        for _ in range(n):
            pl.trash.append(pl.deck.pop(0))
        return True

    # add 1 card from (the top or bottom of) your Life cards to your hand (Zeus, …)
    if re.search(
        r"^add\s+(?:up\s+to\s+)?(?:1|one|a)\s+card\s+from\s+(?:the\s+top\s+or\s+bottom\s+of\s+)?"
        r"your\s+life\s+cards?\s+to\s+your\s+hand",
        sub, re.I,
    ):
        if not pl.life_cards:
            return False
        if int(sim.rng.integers(0, 2)) == 0:
            pl.hand.append(pl.life_cards.pop(0))
        else:
            pl.hand.append(pl.life_cards.pop(-1))
        if hasattr(sim, "_trigger_leader_auto_draw") and hasattr(sim, "_active_player_idx"):
            sim._trigger_leader_auto_draw()
        return True

    # return N DON!! … (simplifié : N actifs → deck DON!!)
    m = re.search(
        r"^return\s+(?:up\s+to\s+)?(\d+)\s+don(?:!!)?\s+cards?\s+from\s+your\s+field\s+to\s+your\s+don",
        sub, re.I,
    )
    if m:
        n = int(m.group(1))
        need = n
        take = min(need, pl.don_active)
        pl.don_active -= take
        pl.don_deck += take
        need -= take
        if need and pl.leader_attached_don:
            t = min(need, pl.leader_attached_don)
            pl.leader_attached_don -= t
            pl.don_deck += t
            need -= t
        if need:
            for bc in list(pl.board):
                if need <= 0:
                    break
                t = min(need, bc.attached_don)
                bc.attached_don -= t
                pl.don_deck += t
                need -= t
        return need == 0

    # Coût non reconnu mais clairement un verbe de coût → ne pas appliquer l'effet gratuitement
    if re.match(r"^(?:trash|rest|return|discard|remove|place|add)\b", sub, re.I):
        return False
    return True


def _apply_activation_cost_string(cost: str, ctx: "EffectContext", *, optional: bool) -> bool:
    """Applique tous les sous-coûts séparés par « and ». False → effet annulé (You may)."""
    if not (cost or "").strip():
        return True
    parts = re.split(r"\s+and\s+", cost.strip(), flags=re.I)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not _apply_activation_subcost(p, ctx):
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Évaluation des conditions
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_condition(eff: ParsedEffect, ctx: "EffectContext") -> bool:
    """
    Évalue la condition d'un ParsedEffect contre l'état de jeu courant.
    Retourne True si la condition est vérifiée, ou True par défaut (optimiste)
    pour les conditions inconnues/non évaluables.
    """
    cond = eff.condition.lower().strip()
    if not cond:
        return True

    owner = ctx.owner
    opp = _opp(ctx)

    def _extract_threshold(text: str) -> tuple[int, str] | None:
        m = re.search(r'(\d+)\s+or\s+(fewer|more|less)', text)
        if m:
            return int(m.group(1)), m.group(2)
        return None

    # ── Vie (nombre de cartes dans la pile Life) ─────────────────────────────
    # Ne pas utiliser « "your" in cond » : « if you have 2 or less Life cards »
    # contient « you » mais pas « your » → l'ancienne heuristique lisait la vie adverse.
    # Le classifier peut aussi tronquer en « … or less Life » sans le mot « cards ».
    res_thr = _extract_threshold(cond)
    if res_thr and "life" in cond and "character" not in cond and "hand" not in cond:
        n_thr, direction = res_thr
        is_life_pile = bool(
            re.search(r"\blife\s+cards?\b", cond)
            or re.search(r"have\s+\d+\s+or\s+(less|more|fewer)\s+life\b", cond)
            or re.search(r"or\s+(less|more|fewer)\s+life\b", cond)
        )
        if is_life_pile:
            if "opponent" in cond:
                target_life = opp.life
            else:
                target_life = owner.life
            return target_life <= n_thr if direction in ("fewer", "less") else target_life >= n_thr

    # ── Main (hand) ───────────────────────────────────────────────────────────
    if "hand" in cond and "card" in cond:
        res = _extract_threshold(cond)
        if res:
            n, direction = res
            target = owner.hand if "your" in cond else opp.hand
            return len(target) <= n if direction in ("fewer", "less") else len(target) >= n

    # ── DON!! sur le terrain ──────────────────────────────────────────────────
    if "don" in cond and ("field" in cond or "play" in cond):
        res = _extract_threshold(cond)
        if res:
            n, direction = res
            total = owner.don_active + owner.don_rested
            return total <= n if direction in ("fewer", "less") else total >= n

    # ── Personnages sur le terrain ────────────────────────────────────────────
    if "character" in cond and "field" in cond:
        res = _extract_threshold(cond)
        if res:
            n, direction = res
            count = len(owner.board)
            return count <= n if direction in ("fewer", "less") else count >= n

    # ── Trash ────────────────────────────────────────────────────────────────
    if "trash" in cond:
        res = _extract_threshold(cond)
        if res:
            n, direction = res
            count = len(owner.trash)
            return count <= n if direction in ("fewer", "less") else count >= n

    # ── Deck vide ────────────────────────────────────────────────────────────
    if "deck" in cond and "0" in cond:
        return len(owner.deck) == 0

    # ── Type / Leader → optimiste (on n'a pas l'arbre de types complet ici) ──
    if "leader" in cond or "type" in cond or "multicolored" in cond:
        return True

    # Optimiste par défaut
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Handlers d'effets individuels
# ══════════════════════════════════════════════════════════════════════════════

def _h_draw(eff: ParsedEffect, ctx: "EffectContext") -> None:
    n = eff.params.get("n", 1)
    _draw_n(ctx, int(n))


def _h_search(eff: ParsedEffect, ctx: "EffectContext") -> None:
    """Search → prend la première carte du deck (simplifié)."""
    owner = ctx.owner
    if owner.deck:
        cid = owner.deck.pop(0)
        owner.hand.append(cid)


def _h_look_top(eff: ParsedEffect, ctx: "EffectContext") -> None:
    """Look top N + reveal → simplifié : ajoute 1 carte en main."""
    _h_search(eff, ctx)


def _h_ko(eff: ParsedEffect, ctx: "EffectContext") -> None:
    """K.O. d'un personnage adverse (le moins coûteux / le moins puissant)."""
    opp = _opp(ctx)
    tgt = eff.target
    cost_max = tgt.cost_max if tgt else 99
    power_max = tgt.power_max if tgt else 99_999

    # Filtre les cibles éligibles (coût / puissance + « cannot be targeted »)
    from ..engine.combat_rules import effective_power

    eligible: list[int] = [
        i for i, b in enumerate(opp.board)
        if (cost_max >= 99 or ctx.sim.cards.get(b.card_id, _stub_card()).cost <= cost_max)
        and effective_power(b, ctx.sim.cards) <= power_max
        and not b.restrictions.blocks_opponent_targeting()
    ]
    if not eligible and opp.board:
        eligible = [
            i for i in range(len(opp.board))
            if not opp.board[i].restrictions.blocks_opponent_targeting()
        ]
    if not eligible:
        return

    idx = min(eligible, key=lambda i: ctx.sim.cards.get(opp.board[i].card_id, _stub_card()).cost)
    bc = opp.board.pop(idx)
    _return_attached_don_to_cost(opp, bc)
    opp.trash.append(bc.card_id)

    ko_card = ctx.sim.cards.get(bc.card_id)
    if ko_card:
        from .keyword_handlers import EffectContext as EC
        ko_ctx = EC(sim=ctx.sim, owner=opp, owner_idx=1 - ctx.owner_idx, card=ko_card)
        try:
            from .on_play_resolver import apply_on_ko_effects
            apply_on_ko_effects(ko_ctx)
        except Exception:
            pass


def _h_power_boost(eff: ParsedEffect, ctx: "EffectContext") -> None:
    amount = int(eff.params.get("amount", 0))
    scope = eff.params.get("scope", "")
    tgt = eff.target
    owner_is_target = (not tgt) or tgt.owner in ("you", "self", "")

    pl = ctx.owner if owner_is_target else _opp(ctx)
    if scope == "all":
        for bc in pl.board:
            bc.power_modifiers.bonus_turn += amount
    elif pl.board:
        from ..engine.combat_rules import effective_power

        idx = max(
            range(len(pl.board)),
            key=lambda i: effective_power(pl.board[i], ctx.sim.cards),
        )
        pl.board[idx].power_modifiers.bonus_turn += amount


def _h_power_reduce(eff: ParsedEffect, ctx: "EffectContext") -> None:
    amount = int(eff.params.get("amount", 0))
    scope = eff.params.get("scope", "")
    opp = _opp(ctx)

    if scope == "all":
        for bc in opp.board:
            bc.power_modifiers.penalty_turn += amount
    elif opp.board:
        from ..engine.combat_rules import effective_power

        cand = [
            i for i in range(len(opp.board))
            if not opp.board[i].restrictions.blocks_opponent_targeting()
        ]
        if not cand:
            return
        idx = max(cand, key=lambda i: effective_power(opp.board[i], ctx.sim.cards))
        opp.board[idx].power_modifiers.penalty_turn += amount


def _h_return_to_hand(eff: ParsedEffect, ctx: "EffectContext") -> None:
    n = int(eff.params.get("n", 1))
    tgt = eff.target
    cost_max = tgt.cost_max if tgt else 99

    is_opp_target = tgt and tgt.owner == "opponent"
    pl = _opp(ctx) if is_opp_target else ctx.owner

    for _ in range(n):
        eligible = [
            i for i, b in enumerate(pl.board)
            if ctx.sim.cards.get(b.card_id, _stub_card()).cost <= cost_max
            and (not is_opp_target or not b.restrictions.blocks_opponent_targeting())
        ]
        if not eligible and pl.board:
            pool = [
                i for i in range(len(pl.board))
                if ctx.sim.cards.get(pl.board[i].card_id, _stub_card()).cost <= cost_max
                and (not is_opp_target or not pl.board[i].restrictions.blocks_opponent_targeting())
            ]
            eligible = [min(pool, key=lambda i: ctx.sim.cards.get(pl.board[i].card_id, _stub_card()).cost)] if pool else []
        if not eligible:
            break
        idx = eligible[0]
        bc = pl.board.pop(idx)
        _return_attached_don_to_cost(pl, bc)
        pl.hand.append(bc.card_id)


def _h_life_add(eff: ParsedEffect, ctx: "EffectContext") -> None:
    n = int(eff.params.get("n", 1))
    source = eff.params.get("source", "deck_top")
    owner = ctx.owner

    for _ in range(n):
        if source == "hand" and owner.hand:
            cid = owner.hand.pop()
        elif owner.deck:
            cid = owner.deck.pop(0)
        else:
            break
        owner.life_cards.insert(0, cid)


def _h_life_remove(eff: ParsedEffect, ctx: "EffectContext") -> None:
    n = int(eff.params.get("n", 1))
    owner = ctx.owner
    for _ in range(n):
        if not owner.life_cards:
            break
        cid = owner.life_cards.pop(0)
        owner.trash.append(cid)


def _h_don_add(eff: ParsedEffect, ctx: "EffectContext") -> None:
    n = int(eff.params.get("n", 1))
    _add_don(ctx, n)
    if eff.params.get("then_set_active"):
        owner = ctx.owner
        if owner.don_rested >= 1:
            owner.don_rested -= 1
            owner.don_active += 1


def _h_don_activate(eff: ParsedEffect, ctx: "EffectContext") -> None:
    """Active des cartes DON!! reposées."""
    n = int(eff.params.get("n", 1))
    owner = ctx.owner
    activated = min(n, owner.don_rested)
    owner.don_rested -= activated
    owner.don_active += activated


def _h_set_active(eff: ParsedEffect, ctx: "EffectContext") -> None:
    """Active des personnages ou DON!! reposés."""
    n = int(eff.params.get("n", 1))
    owner = ctx.owner
    # Tente d'abord les DON!!
    if owner.don_rested >= 1:
        activated = min(n, owner.don_rested)
        owner.don_rested -= activated
        owner.don_active += activated
    else:
        # Active des personnages reposés
        rested = [i for i, b in enumerate(owner.board) if b.rested]
        for i in rested[:n]:
            owner.board[i].rested = False


def _h_trash_hand(eff: ParsedEffect, ctx: "EffectContext") -> None:
    n = int(eff.params.get("n", 1))
    _discard_n(ctx, ctx.owner, n)


def _h_trash_deck(eff: ParsedEffect, ctx: "EffectContext") -> None:
    n = int(eff.params.get("n", 1))
    owner = ctx.owner
    for _ in range(n):
        if not owner.deck:
            break
        owner.trash.append(owner.deck.pop(0))


def _h_add_to_hand(eff: ParsedEffect, ctx: "EffectContext") -> None:
    n = int(eff.params.get("n", 1))
    source = eff.params.get("source", "")
    owner = ctx.owner

    for _ in range(n):
        if source == "life" and owner.life_cards:
            owner.hand.append(owner.life_cards.pop(0))
        elif source in ("trash", "discard") and owner.trash:
            owner.hand.append(owner.trash.pop())
        elif owner.deck:
            owner.hand.append(owner.deck.pop(0))


def _h_rest_target(eff: ParsedEffect, ctx: "EffectContext") -> None:
    target_self = eff.params.get("target") == "self"
    method = eff.params.get("method", "")
    n = int(eff.params.get("n", 1))

    if target_self:
        # Repose sa propre carte (souvent un coût)
        if ctx.owner.board:
            ctx.owner.board[0].rested = True
        return

    from ..engine.combat_rules import effective_power

    opp = _opp(ctx)
    active = [i for i, b in enumerate(opp.board) if not b.rested]
    for i in sorted(active, key=lambda i: -effective_power(opp.board[i], ctx.sim.cards))[:n]:
        opp.board[i].rested = True


def _h_give_keyword(eff: ParsedEffect, ctx: "EffectContext") -> None:
    kw = eff.params.get("keyword", "").lower()
    if not ctx.owner.board:
        return
    bc = ctx.owner.board[0]
    g = bc.keyword_grants
    if "rush" in kw:
        if "character" in kw:
            g.rush_character = True
        else:
            g.rush = True
    elif "blocker" in kw:
        bc.has_blocker = True
    elif "double" in kw:
        g.double_attack = True
    elif "unblockable" in kw:
        g.unblockable = True
    elif "banish" in kw:
        g.banish = True


def _h_play_from_hand(eff: ParsedEffect, ctx: "EffectContext") -> None:
    """Joue une carte depuis la main gratuitement (simplifié)."""
    owner = ctx.owner
    for i, cid in enumerate(owner.hand):
        cd = ctx.sim.cards.get(cid)
        if cd and "character" in (cd.card_type or "").lower():
            if len(owner.board) < ctx.sim.max_board:
                owner.hand.pop(i)
                from ..engine.restriction_state import bootstrap_printed_restrictions_from_card
                from ..simulator import BoardChar
                owner.board.append(BoardChar(
                    card_id=cid, power=cd.power,
                    has_rush=ctx.sim._has_rush(cd),
                    has_rush_char=ctx.sim._has_rush_char(cd),
                    has_blocker=ctx.sim._has_blocker(cd),
                    has_double_attack=ctx.sim._has_double_attack(cd),
                    has_unblockable=ctx.sim._has_unblockable(cd),
                    has_attack_active=ctx.sim._has_attack_active_chars(cd),
                    just_played=True,
                    restrictions=bootstrap_printed_restrictions_from_card(cd),
                ))
                from .keyword_handlers import EffectContext as EC
                play_ctx = EC(sim=ctx.sim, owner=owner, owner_idx=ctx.owner_idx, card=cd)
                from .on_play_resolver import apply_on_play_effects
                apply_on_play_effects(play_ctx)
            break


def _h_play_from_trash(eff: ParsedEffect, ctx: "EffectContext") -> None:
    """Joue un personnage depuis le trash gratuitement."""
    owner = ctx.owner
    for i in reversed(range(len(owner.trash))):
        cid = owner.trash[i]
        cd = ctx.sim.cards.get(cid)
        if cd and "character" in (cd.card_type or "").lower():
            if len(owner.board) < ctx.sim.max_board:
                owner.trash.pop(i)
                from ..engine.restriction_state import bootstrap_printed_restrictions_from_card
                from ..simulator import BoardChar
                owner.board.append(BoardChar(
                    card_id=cid, power=cd.power,
                    has_rush=ctx.sim._has_rush(cd),
                    has_rush_char=ctx.sim._has_rush_char(cd),
                    has_blocker=ctx.sim._has_blocker(cd),
                    has_double_attack=ctx.sim._has_double_attack(cd),
                    has_unblockable=ctx.sim._has_unblockable(cd),
                    has_attack_active=ctx.sim._has_attack_active_chars(cd),
                    just_played=True,
                    restrictions=bootstrap_printed_restrictions_from_card(cd),
                ))
            break


def _h_set_power(eff: ParsedEffect, ctx: "EffectContext") -> None:
    power = eff.params.get("power")
    if isinstance(power, int) and ctx.owner.board:
        bc = ctx.owner.board[0]
        bc.power = int(power)
        bc.power_modifiers.clear_turn_scoped()


def _h_place_to_deck(eff: ParsedEffect, ctx: "EffectContext") -> None:
    """Place un personnage (adverse ou propre) au fond du deck."""
    n = int(eff.params.get("n", 1))
    scope = eff.params.get("scope", "")
    dest = eff.params.get("dest", "deck_bottom")
    tgt = eff.target
    is_opp = tgt and tgt.owner == "opponent"
    pl = _opp(ctx) if is_opp else ctx.owner

    if scope == "all":
        cost_max = eff.params.get("cost_max", 99)
        eligible = [
            i for i, b in enumerate(pl.board)
            if ctx.sim.cards.get(b.card_id, _stub_card()).cost <= cost_max
        ]
        for i in sorted(eligible, reverse=True):
            bc = pl.board.pop(i)
            _return_attached_don_to_cost(pl, bc)
            pl.deck.append(bc.card_id)
    else:
        for _ in range(n):
            if not pl.board:
                break
            bc = pl.board.pop(0)
            _return_attached_don_to_cost(pl, bc)
            if dest == "deck_bottom":
                pl.deck.append(bc.card_id)
            else:
                pl.deck.insert(0, bc.card_id)


def _h_special_rule(eff: ParsedEffect, ctx: "EffectContext") -> None:
    rtype = eff.params.get("type", "")
    if rtype == "don_cost":
        n = int(eff.params.get("n", 1))
        owner = ctx.owner
        used = min(n, owner.don_active)
        owner.don_active -= used
        owner.don_deck += used
    elif rtype == "cost_reduce":
        pass  # pas de coût dynamique dans le sim simplifié
    elif rtype == "attack_active":
        if ctx.owner.board:
            b = ctx.owner.board[0]
            b.keyword_grants.rush = True
            b.keyword_grants.can_attack_active_characters = True


def _h_noop(eff: ParsedEffect, ctx: "EffectContext") -> None:
    pass


# Stub card pour éviter KeyError
class _StubCard:
    cost = 0
    card_type = "Character"

def _stub_card() -> _StubCard:
    return _StubCard()


def _h_don_attach(eff: ParsedEffect, ctx: "EffectContext") -> None:
    """
    DON!! attaché : +1000 en combat uniquement via BoardChar.attached_don (comme _attach_don).
    Source : DON!! actifs ou reposés selon le texte (« rested »).
    Cible : le personnage source de l'effet s'il est sur le terrain, sinon board[0].
    """
    n = int(eff.params.get("n", 1))
    owner = ctx.owner
    rested = "rested" in (eff.raw_text or "").lower()
    target_i = 0
    src_id = getattr(ctx.card, "card_id", None) or ""
    if src_id and owner.board:
        for i, bc in enumerate(owner.board):
            if bc.card_id == src_id:
                target_i = i
                break
    for _ in range(n):
        if rested:
            if owner.don_rested <= 0:
                break
            owner.don_rested -= 1
        else:
            if owner.don_active <= 0:
                break
            owner.don_active -= 1
        if owner.board:
            owner.board[target_i].attached_don += 1
        else:
            owner.leader_attached_don += 1


def _h_trash_from_life(eff: ParsedEffect, ctx: "EffectContext") -> None:
    """Trash depuis le deck Vie."""
    n = int(eff.params.get("n", 1))
    owner = ctx.owner
    for _ in range(n):
        if not owner.life_cards:
            break
        owner.trash.append(owner.life_cards.pop(0))


# ══════════════════════════════════════════════════════════════════════════════
# Table de dispatch
# ══════════════════════════════════════════════════════════════════════════════

_HANDLERS = {
    EffectType.DRAW:              _h_draw,
    EffectType.SEARCH_DECK:       _h_search,
    EffectType.LOOK_TOP:          _h_look_top,
    EffectType.REVEAL_TOP:        _h_look_top,
    EffectType.KO:                _h_ko,
    EffectType.POWER_BOOST:       _h_power_boost,
    EffectType.POWER_REDUCE:      _h_power_reduce,
    EffectType.RETURN_TO_HAND:    _h_return_to_hand,
    EffectType.LIFE_ADD:          _h_life_add,
    EffectType.LIFE_REMOVE:       _h_life_remove,
    EffectType.DON_ADD:           _h_don_add,
    EffectType.DON_ATTACH:        _h_don_attach,
    EffectType.DON_ACTIVATE:      _h_don_activate,
    EffectType.DON_REST:          _h_noop,
    EffectType.DON_RETURN:        lambda eff, ctx: ctx.sim.return_don(ctx.owner, int(eff.params.get("n", 1))),
    EffectType.SET_ACTIVE:        _h_set_active,
    EffectType.TRASH_FROM_HAND:   _h_trash_hand,
    EffectType.TRASH_FROM_DECK:   _h_trash_deck,
    EffectType.TRASH_FROM_LIFE:   _h_trash_from_life,
    EffectType.ADD_TO_HAND:       _h_add_to_hand,
    EffectType.REST_TARGET:       _h_rest_target,
    EffectType.ACTIVATE_TARGET:   _h_noop,
    EffectType.GIVE_KEYWORD:      _h_give_keyword,
    EffectType.PLAY_FROM_HAND:    _h_play_from_hand,
    EffectType.PLAY_FROM_TRASH:   _h_play_from_trash,
    EffectType.PLAY_FROM_DECK:    _h_noop,
    EffectType.COST_REDUCE:       _h_noop,
    EffectType.SET_POWER:         _h_set_power,
    EffectType.CANNOT_KO:         _h_noop,
    EffectType.SACRIFICE_PROTECT: _h_noop,
    EffectType.PLACE_TO_DECK:     _h_place_to_deck,
    EffectType.ACTIVATE_SELF_EFFECT: _h_noop,
    EffectType.SPECIAL_RULE:      _h_special_rule,
    EffectType.BANISH:            _h_noop,
    EffectType.CONDITION:         _h_noop,
    EffectType.UNKNOWN:           _h_noop,
}

# Types à ignorer (mots-clés passifs déjà gérés dans simulator._has_*)
_SKIP_TYPES = frozenset({
    EffectType.KEYWORD_RUSH, EffectType.KEYWORD_BLOCKER,
    EffectType.KEYWORD_DOUBLE_ATTACK, EffectType.KEYWORD_BANISH,
    EffectType.KEYWORD_UNBLOCKABLE, EffectType.KEYWORD_INFILTRATE,
    EffectType.CONDITION, EffectType.UNKNOWN,
})


# ══════════════════════════════════════════════════════════════════════════════
# Point d'entrée principal
# ══════════════════════════════════════════════════════════════════════════════

def resolve_effects(
    timing: str,
    ctx: "EffectContext",
    cache: CardEffectCache,
) -> bool:
    """
    Résout tous les ParsedEffect pour le timing donné sur ctx.card.

    - Ignore les effets UNKNOWN et les mots-clés passifs (déjà gérés ailleurs)
    - Évalue les conditions contre l'état de jeu courant
    - Retourne True si au moins un effet a été appliqué

    Args:
        timing: label de timing ("On Play", "On K.O.", "Trigger", etc.)
        ctx: contexte d'effet (carte + propriétaire + simulateur)
        cache: cache de ParsedCard partagé pour le simulateur
    """
    pc = cache.get(ctx.card)
    fired = False
    cost_paid_ok: set[tuple[str, str, str]] = set()
    cost_pay_failed: set[tuple[str, str, str]] = set()

    for eff in pc.effects:
        if eff.timing != timing:
            continue
        if eff.effect_type in _SKIP_TYPES:
            continue

        ckey = (eff.timing, eff.raw_text, (eff.activation_cost or "").strip())
        if ckey[2]:
            if ckey in cost_pay_failed:
                continue
            if ckey not in cost_paid_ok:
                if not _apply_activation_cost_string(
                    eff.activation_cost, ctx, optional=eff.optional
                ):
                    cost_pay_failed.add(ckey)
                    continue
                cost_paid_ok.add(ckey)

        if not _resolve_condition(eff, ctx):
            continue
        handler = _HANDLERS.get(eff.effect_type, _h_noop)
        try:
            handler(eff, ctx)
            fired = True
        except Exception:
            pass  # Effets non fatals — le sim continue
    return fired
