"""
Simulateur OP-TCG — logique de jeu fidèle aux règles officielles.

Phases par tour (CR 6.0) :
  Refresh → Draw → DON!! → Main (inclut les attaques) → End (défausse excédent)

Actions (voir ``ACTION_SPACE_SIZE`` dans env) :
  MAIN  : 0–6 jouer carte main[slot] ; 7 = fin de tour
          45–104 attacher 1..10 DON!! : slot=(a-45)//10, quantité=(a-45)%10+1
          puis [Activate: Main] perso / Leader
          (bonus step configurable : ``sim.reward_activate_main_leader`` / ``reward_activate_main_character``)
  BATTLE: 8 fin phase ; 9–44 attaques (code = action-9, 6×6) ; 45–104 attache-DON

Règles implémentées :
  - Le Leader est une carte séparée du deck principal (hors pile, hors pioche, hors vie).
  - Sa puissance est lue dans le CSV si disponible (sinon `leader_block` du config).
  - La carte vie prise en combat est ajoutée à la MAIN (règle officielle).
  - L'attaquant se repose à la déclaration ; il n'est jamais K.O. par un combat.
  - Seul le défenseur peut jouer des Events Counter.
  - Le Blocker redirige automatiquement une attaque sur le Leader.
  - DON!! : gain en début de tour, zone actif / reposé, refresh en début de son tour.
  - Pas de dégâts sur le Leader le premier cycle de tours (tours_started < 3).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .card_db import CardDef
from .engine.combat_rules import (
    apply_counter_stack_until_safe,
    best_counter_card_index,
    board_character_power,
    effective_power,
    has_blocker_keyword,
    is_counter_event,
)
from .engine.keyword_grant_state import KeywordGrantState
from .engine.power_modifier_state import PowerModifierState
from .engine.keyword_handlers import EffectContext, dispatch_on_play
from .engine.restriction_state import (
    RestrictionState,
    bootstrap_printed_restrictions_from_card,
    character_may_declare_attack,
    leader_may_declare_attack,
)
from .playability import Phase, PlayabilityContext, can_play_card, can_play_character, can_play_event

# ──────────────────────────────────────────────────────────────────────────────
# Constantes d'actions
# ──────────────────────────────────────────────────────────────────────────────
MAIN_END_ACTION = 7          # Passer en BATTLE puis fin de tour
BATTLE_PASS_ACTION = 8       # Fin de la phase de combat → fin de tour
BATTLE_ATTACK_BASE = 9       # Premier code d'attaque
N_ATTACKERS = 6              # 0=Leader, 1–5=perso board[0–4]  (CR 7-1-1 : Leader ou perso actif)
N_TARGETS = 6                # 0=Leader, 1–5=perso board[0–4]  (CR 7-1-1-2 : Leader ou perso reposé)
N_BATTLE_ATTACK_CODES = N_ATTACKERS * N_TARGETS  # 36
# Actions Main / Battle — attacher N DON!! actifs à un slot (CR 6-5-5), N = 1..MAIN_ATTACH_DON_MAX
# Encodage : rel = action - MAIN_ATTACH_DON_BASE → slot = rel // MAX, N = rel % MAX + 1
# slot 0 = Leader, 1–5 = board[0–4]
MAIN_ATTACH_DON_BASE  = BATTLE_ATTACK_BASE + N_BATTLE_ATTACK_CODES  # 9 + 36 = 45
MAIN_ATTACH_DON_SLOTS = 6    # Leader + 5 persos (cibles)
MAIN_ATTACH_DON_MAX   = 10   # jusqu'à 10 DON!! par action (un pas d'env au lieu de N)
MAIN_ATTACH_DON_ACTIONS = MAIN_ATTACH_DON_SLOTS * MAIN_ATTACH_DON_MAX  # 60
# Actions Main Phase — activer [Activate: Main] sur un perso (repos + effet, CR 10-2-2)
MAIN_ACTIVATE_MAIN_BASE  = MAIN_ATTACH_DON_BASE + MAIN_ATTACH_DON_ACTIONS  # 105
MAIN_ACTIVATE_MAIN_SLOTS = 5  # board[0–4]
# Action suivant les 5 slots perso = [Activate: Main] Leader
MAIN_ACTIVATE_MAIN_LEADER = MAIN_ACTIVATE_MAIN_BASE + MAIN_ACTIVATE_MAIN_SLOTS  # 110
ACTION_SPACE_SIZE = MAIN_ACTIVATE_MAIN_LEADER + 1  # 111


def decode_attach_don_action(action: int) -> tuple[int, int] | None:
    """
    Si ``action`` est une action attache-DON, retourne ``(slot, count)`` avec count 1..MAX.
    Sinon ``None``.
    """
    if MAIN_ATTACH_DON_BASE <= action < MAIN_ATTACH_DON_BASE + MAIN_ATTACH_DON_ACTIONS:
        rel = action - MAIN_ATTACH_DON_BASE
        slot = rel // MAIN_ATTACH_DON_MAX
        count = rel % MAIN_ATTACH_DON_MAX + 1
        return slot, count
    return None

# ── Phase MULLIGAN (réutilise les slots 0-1) ──────────────────────────────────
MULLIGAN_KEEP     = 0    # Garder la main initiale
MULLIGAN_TAKE     = 1    # Relancer : mélanger la main dans le deck + repiocher 5

# ── Phase BLOCKER (réutilise les slots 0-5) ───────────────────────────────────
BLOCKER_PASS      = 0    # Pas de Blocker → le Leader prend le dégât
BLOCKER_SLOT_BASE = 1    # 1..5 = board[0..4] comme Blocker (se repose)
BLOCKER_N_SLOTS   = 6    # 0 = passer, 1–5 = board[0–4]


# ──────────────────────────────────────────────────────────────────────────────
# Structures de données
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BoardChar:
    """Un personnage sur le terrain."""
    card_id: str
    power: int
    rested: bool = False
    has_rush: bool = False            # CR 10-1-1 : peut attaquer le tour où il est joué (Leader ou perso)
    has_rush_char: bool = False       # CR 10-1-6 : peut attaquer les persos adverses le tour où il est joué (pas le Leader)
    has_blocker: bool = False
    has_double_attack: bool = False   # CR 10-1-2 : 2 dégâts vie au lieu de 1
    has_unblockable: bool = False     # CR 10-1-7 : ignore [Blocker]
    has_attack_active: bool = False   # imprimé : peut cibler des persos adverses actifs
    just_played: bool = True          # True ce tour → pas d'attaque sans Rush / Rush:Character
    attached_don: int = 0             # Nombre de DON!! attachés (+1000 puissance chacun)
    restrictions: RestrictionState = field(default_factory=RestrictionState)
    power_modifiers: PowerModifierState = field(default_factory=PowerModifierState)
    keyword_grants: KeywordGrantState = field(default_factory=KeywordGrantState)
    activate_main_used: bool = False  # [Activate:Main] déjà utilisé ce tour (CR 10-2-13 Once Per Turn)


@dataclass
class PlayerState:
    """État d'un joueur."""
    deck: list[str] = field(default_factory=list)
    leader_id: str | None = None      # Carte Leader (hors deck principal)
    leader_power: int = 5000          # Puissance réelle du Leader (lue depuis le CSV)
    leader_rested: bool = False       # Leader reposé (après avoir attaqué)
    leader_just_played: bool = False  # Toujours False (le Leader est là dès le début)
    leader_attached_don: int = 0      # DON!! attachés au Leader
    leader_power_modifiers: PowerModifierState = field(default_factory=PowerModifierState)
    leader_once_per_turn_draw_used: bool = False   # [Once Per Turn] pioche (ex. Nami)
    leader_once_per_turn_def_used: bool = False    # [Once Per Turn] défense (ex. Nami)
    leader_opponent_turn_life_zero_used: bool = (
        False  # [Opponent's Turn][Once Per Turn] vie → 0 (ex. Enel OP05-098)
    )
    leader_activate_main_used: bool = False        # [Activate:Main] Leader déjà utilisé ce tour
    leader_restrictions: RestrictionState = field(default_factory=RestrictionState)
    hand: list[str] = field(default_factory=list)
    board: list[BoardChar] = field(default_factory=list)  # max_board personnages
    stage_area: list[str] = field(default_factory=list)   # zone Stage : au plus 1 carte à la fois (CR 3-8-5)
    trash: list[str] = field(default_factory=list)
    life_cards: list[str] = field(default_factory=list)   # index 0 = dessus de pile
    don_active: int = 0               # DON!! payables ce tour
    don_rested: int = 0               # DON!! déjà utilisés / attachés
    don_deck: int = 10                # Cartes restantes dans le deck DON!!
    don_deck_initial: int = 10        # Taille initiale du deck DON!! (leader ; replay / affichage)
    # Shaping env : tour Main terminé sans action (hors fin de tour) → bonus sur la prochaine Main
    last_main_finished_idle: bool = False
    boost_next_main_activity: bool = False

    @property
    def don(self) -> int:
        return self.don_active

    @property
    def life(self) -> int:
        return len(self.life_cards)

    def effective_leader_power(self, *, include_attached_don_bonus: bool = True) -> int:
        """
        Puissance du Leader : base + bonus temporaire (effets) + optionnellement
        +1000 par DON!! attaché (CR 6-5-5-2 : ce bonus ne compte que pendant ton tour).
        En défense pendant le tour adverse, utiliser include_attached_don_bonus=False.
        Les DON!! restent attachés pour [DON!! xN] et coûts ; seul le +1000 combat est retiré.
        """
        p = self.leader_power + self.leader_power_modifiers.net_bonus()
        if include_attached_don_bonus:
            p += 1000 * self.leader_attached_don
        return p


# ──────────────────────────────────────────────────────────────────────────────
# Simulateur principal
# ──────────────────────────────────────────────────────────────────────────────

class SimplifiedOPSim:
    """
    Simulateur OP-TCG simplifié mais fidèle aux règles de base :
    - Leader séparé du deck, puissance depuis le CSV
    - Vie → main (pas trash)
    - Pas de KO attaquant en combat
    - Leader peut attaquer
    - Structure de tour : Refresh → Draw → DON!! → Main → End
    """

    def __init__(
        self,
        cards: dict[str, CardDef],
        deck0: list[str],
        deck1: list[str],
        *,
        starting_hand: int = 5,
        starting_life: int = 5,
        don_per_turn: int = 2,
        official_first_player_don: int = 1,
        max_don: int = 10,
        draw_per_turn: int = 1,
        skip_first_draw_first_player: bool = True,
        max_board: int = 5,          # CR 3-7-6 : maximum 5 cartes dans la Character area
        max_hand: int = 0,           # 0 = pas de limite (CR : pas de taille max de main sauf effet de carte). >0 = plafond outil (tests / shaping).
        leader_block: int = 5000,    # Fallback si leader_id absent du CSV
        leader_defense: int | None = None,
        leader0_id: str | None = None,
        leader1_id: str | None = None,
        effect_snapshot_path: str | None = None,
        effect_cache_dedupe: bool = False,
        rng: np.random.Generator | None = None,
        opponent_action_policy: str = "random",
        self_play: bool = False,
        stall_no_card_after_turns_started: int = 0,
        reward_activate_main_leader: float = 0.05,
        reward_activate_main_leader_fail: float = -0.05,
        reward_activate_main_character: float = 0.03,
        reward_activate_main_character_fail: float = -0.05,
        reward_leader_opponent_turn_life_zero: float = 0.45,
        shuffle_decks: bool = True,
    ):
        self.cards = cards
        self.deck0_init = list(deck0)
        self.deck1_init = list(deck1)
        self._leader0_id = leader0_id
        self._leader1_id = leader1_id
        self.starting_hand = starting_hand
        self.starting_life = starting_life
        self.don_per_turn = don_per_turn
        self.official_first_player_don = int(official_first_player_don)
        self.max_don = max_don
        self.draw_per_turn = draw_per_turn
        self.skip_first_draw_first_player = skip_first_draw_first_player
        self.max_board = max_board
        self.max_hand = max_hand
        self._leader_block_fallback = int(leader_block)
        self._leader_defense_override = int(leader_defense) if leader_defense is not None else None
        self.rng = rng or np.random.default_rng()
        pol = (opponent_action_policy or "random").strip().lower()
        self._opponent_action_policy = pol if pol in ("random", "deterministic_min") else "random"
        self._self_play_external = bool(self_play)
        self._stall_no_card_after_ts = max(0, int(stall_no_card_after_turns_started))
        self._total_cards_played_from_hand: int = 0
        self._reward_activate_main_leader_ok = float(reward_activate_main_leader)
        self._reward_activate_main_leader_bad = float(reward_activate_main_leader_fail)
        self._reward_activate_main_char_ok = float(reward_activate_main_character)
        self._reward_activate_main_char_bad = float(reward_activate_main_character_fail)
        self._reward_leader_opp_turn_life0 = float(reward_leader_opponent_turn_life_zero)
        self._opponent_battle_extra_reward: float = 0.0
        self._shuffle_decks = bool(shuffle_decks)

        # Cache d'effets structurés (pipeline models/) — pré-calculé au démarrage
        from .engine.effect_resolver import CardEffectCache
        self._effect_cache = CardEffectCache(
            snapshot_path=effect_snapshot_path,
            dedupe_identical_rule_text=effect_cache_dedupe,
        )
        self._effect_cache.precompute(cards)

        self.p0 = PlayerState()
        self.p1 = PlayerState()
        self.phase = Phase.MAIN
        self.done = False
        self._winner: int | None = None
        self._timeout_forced: bool = False
        self._turn_cap_forced: bool = False
        self._stall_no_card_forced: bool = False
        self.turns_started: int = 0
        self._p0_don_phases: int = 0
        # Self-play : P1 piloté par la même politique (env incrémental)
        self._mulligan_p0_done: bool = False
        self._mulligan_need_p1: bool = False
        self._policy_controls_p1: bool = False

    def perspective_p1(self) -> bool:
        """True si l’observation / les actions légales sont du point de vue de P1 (pas en BLOCKER)."""
        if not self._self_play_external:
            return False
        if self.phase == Phase.BLOCKER:
            return False
        return self._mulligan_need_p1 or self._policy_controls_p1

    # ── helpers ───────────────────────────────────────────────────────────────

    def _leader_power_for(self, leader_id: str | None) -> int:
        if leader_id and leader_id in self.cards:
            p = self.cards[leader_id].power
            if p > 0:
                return p
        return self._leader_block_fallback

    def _leader_don_deck_size(self, leader_id: str | None) -> int:
        """Nombre de cartes dans le deck DON!! du Leader (10 par défaut ; ex. Enel OP15 : 6)."""
        if not leader_id or leader_id not in self.cards:
            return 10
        cd = self.cards[leader_id]
        if "leader" not in (cd.card_type or "").lower():
            return 10
        return max(1, min(20, int(cd.don_deck_size)))

    def player_turn_number(self, pl: PlayerState) -> int:
        """
        Numéro du tour courant pour ce joueur (1 = son 1er tour, 2 = 2e, …).
        ``turns_started`` alterne P0 puis P1 : 1→P0, 2→P1, 3→P0…
        """
        t = int(self.turns_started)
        if pl is self.p0:
            return (t + 1) // 2
        return t // 2

    def turn_has_meaningful_activity(self) -> bool:
        """Au moins une action du tour : Main (carte / DON!! / Activate), attache combat, ou attaque."""
        return (
            self._main_nonidle_actions > 0
            or self._attacks_made_turn > 0
            or self._don_attached_turn > 0
        )

    def any_legal_main_commit_for_player(self, pl: PlayerState) -> bool:
        """
        Il existe une action Main autre que la fin de phase (aligné sur ``env.legal_actions_mask``).
        """
        if self.phase != Phase.MAIN:
            return False
        ctx = PlayabilityContext(Phase.MAIN, pl.don, len(pl.hand))
        for slot in range(min(len(pl.hand), 7)):
            cid = pl.hand[slot]
            cd = self.cards.get(cid)
            if cd is None:
                continue
            if can_play_card(cd, ctx):
                return True
        if pl.don_active > 0:
            d = pl.don_active
            for rel in range(MAIN_ATTACH_DON_ACTIONS):
                slot = rel // MAIN_ATTACH_DON_MAX
                k = rel % MAIN_ATTACH_DON_MAX + 1
                if k > d:
                    continue
                if slot > 0 and (slot - 1) >= len(pl.board):
                    continue
                return True
        for i, b in enumerate(pl.board):
            if i >= MAIN_ACTIVATE_MAIN_SLOTS:
                break
            if b.rested or b.restrictions.blocks_attack_declaration() or b.activate_main_used:
                continue
            cd_b = self.cards.get(b.card_id)
            if cd_b and cd_b.has_activate_main:
                return True
        if not pl.leader_activate_main_used:
            leader_cd = self.cards.get(pl.leader_id or "")
            if leader_cd and leader_cd.has_activate_main:
                ltxt = (leader_cd.card_text or "").lower()
                if "second turn or later" in ltxt and self.player_turn_number(pl) < 2:
                    pass
                else:
                    don_cost = int(leader_cd.activate_main_don_minus)
                    don_total = pl.don_active + pl.leader_attached_don + sum(
                        b.attached_don for b in pl.board
                    )
                    if don_total >= don_cost:
                        return True
        return False

    def any_legal_battle_commit_for_pair(
        self,
        atk_pl: PlayerState,
        def_pl: PlayerState,
    ) -> bool:
        """Attaque légalisable ou attachement DON!! en phase combat pour l'attaquant."""
        for code in range(N_BATTLE_ATTACK_CODES):
            atk_slot = code // N_TARGETS
            tgt_slot = code % N_TARGETS
            if self.attack_is_legal(atk_pl, def_pl, atk_slot, tgt_slot):
                return True
        if atk_pl.don_active > 0:
            d = atk_pl.don_active
            for rel in range(MAIN_ATTACH_DON_ACTIONS):
                slot = rel // MAIN_ATTACH_DON_MAX
                k = rel % MAIN_ATTACH_DON_MAX + 1
                if k > d:
                    continue
                if slot > 0 and (slot - 1) >= len(atk_pl.board):
                    continue
                return True
        return False

    def _card(self, cid: str) -> CardDef:
        return self.cards[cid]

    def _opp_choice_index(self, n: int) -> int:
        """Index parmi n options légales pour l’IA adverse (P1 / combat IA)."""
        if n <= 0:
            return 0
        if self._opponent_action_policy == "deterministic_min":
            return 0
        return int(self.rng.integers(0, n))

    def _has_rush(self, cd: CardDef) -> bool:
        """CR 10-1-1 : peut attaquer le tour de pose, toutes cibles valides."""
        blob = " ".join(cd.keywords).lower() + " " + (cd.card_text or "").lower()
        return "[rush]" in blob

    def _has_rush_char(self, cd: CardDef) -> bool:
        """CR 10-1-6 : peut attaquer les Characters adverses le tour de pose (pas le Leader)."""
        blob = " ".join(cd.keywords).lower() + " " + (cd.card_text or "").lower()
        return "[rush: character]" in blob

    def _has_blocker(self, cd: CardDef) -> bool:
        return has_blocker_keyword(cd)

    def _has_double_attack(self, cd: CardDef) -> bool:
        """CR 10-1-2 : 2 dégâts vie au lieu de 1."""
        blob = " ".join(cd.keywords).lower() + " " + (cd.card_text or "").lower()
        return "[double attack]" in blob

    def _has_banish(self, cd: CardDef) -> bool:
        """CR 10-1-3 : la carte vie va au trash (pas en main)."""
        blob = " ".join(cd.keywords).lower() + " " + (cd.card_text or "").lower()
        return "[banish]" in blob

    def _has_unblockable(self, cd: CardDef) -> bool:
        """CR 10-1-7 : ne peut pas être bloqué."""
        blob = " ".join(cd.keywords).lower() + " " + (cd.card_text or "").lower()
        return "[unblockable]" in blob

    def _has_attack_active_chars(self, cd: CardDef) -> bool:
        """Imprimé sur la carte : peut attaquer des personnages adverses actifs (non reposés)."""
        blob = " ".join(cd.keywords).lower() + " " + (cd.card_text or "").lower()
        return bool(
            re.search(
                r"\bcan\s+(?:also\s+)?attack\s+active\s+characters?\b",
                blob,
                re.IGNORECASE,
            )
        )

    def _eff_rush(self, bc: BoardChar) -> bool:
        return bc.has_rush or bc.keyword_grants.rush

    def _eff_rush_char(self, bc: BoardChar) -> bool:
        return bc.has_rush_char or bc.keyword_grants.rush_character

    def _can_attack_active_opponent_chars(self, bc: BoardChar) -> bool:
        return bc.has_attack_active or bc.keyword_grants.can_attack_active_characters

    def _char_double_attack(self, bc: BoardChar, atk_cd: CardDef | None) -> bool:
        return bc.has_double_attack or bc.keyword_grants.double_attack

    def _char_banish(self, bc: BoardChar, atk_cd: CardDef | None) -> bool:
        return bc.keyword_grants.banish or bool(atk_cd and self._has_banish(atk_cd))

    def _char_unblockable(self, bc: BoardChar, atk_cd: CardDef | None) -> bool:
        return bc.has_unblockable or bc.keyword_grants.unblockable

    def _has_trigger(self, cd: CardDef) -> bool:
        """CR 10-1-5 : effet [Trigger] activable quand la carte sort de la pile vie."""
        blob = " ".join(cd.keywords).lower() + " " + (cd.card_text or "").lower()
        return "[trigger]" in blob

    def _trigger_plays_self(self, cd: CardDef) -> bool:
        """
        True si l'effet [Trigger] dit « Play this card » (la carte va sur le terrain, pas au trash).
        CR 10-1-5-2 : la carte est jouée reposée sur le terrain lors de l'activation du Trigger.
        """
        from .engine.on_play_resolver import extract_timing_segment
        seg = extract_timing_segment(cd.card_text or "", "trigger") or ""
        return bool(re.search(r"\bplay\s+this\s+card\b", seg, re.IGNORECASE))

    def _should_use_trigger(self, pl: PlayerState, cd: CardDef) -> bool:
        """
        Heuristique IA : faut-il activer l'effet [Trigger] ?

        Règle officielle (CR 10-1-5-1) : le joueur CHOISIT d'activer ou non.
        - OUI → l'effet se résout, puis la carte va au trash (ou sur le terrain si « Play this card »).
        - NON → la carte va en main normalement.

        Heuristique IA :
          YES si : draw, deal damage, gain life, add DON!!, search deck,
                   « play this card » (si personnage et board non plein),
                   ou toute clause conditionnelle (le moteur évalue lui-même).
          NO  si : pur coût de défausse sur soi-même sans gain visible.
          Défaut : YES si un texte d'effet est détecté.
        """
        from .engine.on_play_resolver import extract_timing_segment
        text = cd.card_text or ""
        seg = (extract_timing_segment(text, "trigger") or "").lower().strip()
        if not seg:
            return False  # aucun texte → rien à activer

        # Effets toujours bénéfiques → YES
        if re.search(r"\bdraw\b", seg):
            return True
        if re.search(r"\bdeal\s+\d+\s+damage\b", seg):
            return True
        if re.search(r"\bgain\s+\d+\s+life\b|\badd.*?life\s+cards?\b|\blife\s+cards?\b.*\bdeck\b", seg):
            return True
        if re.search(r"\badd.*?don!!\b|\bdon!!\b.*\bactive\b", seg):
            return True
        if re.search(r"\bsearch\s+your\s+deck\b|\blook\s+at.*\btop\b", seg):
            return True

        # « Play this card » → YES si c'est un personnage (le board peut être plein,
        # _play_char_to_board gère le remplacement du plus faible)
        if re.search(r"\bplay\s+this\s+card\b", seg):
            ct = (cd.card_type or "").lower()
            return "character" in ct or "stage" in ct

        # Effets conditionnels → YES (le moteur de conditions évalue lui-même)
        if re.search(r"\bif\b", seg):
            return True

        # Pur coût de défausse sur soi-même → NO (inutile d'activer pour se blesser)
        if re.search(r"\btrash\s+(?:\d+|a)\s+cards?\s+from\s+your\s+hand\b", seg):
            if not re.search(r"\byour\s+opponent\b|\bdraw\b|\bgain\b|\bdeal\b", seg):
                return False

        # Tout autre effet non nul → YES par défaut
        return bool(seg)

    # ── initialisation ────────────────────────────────────────────────────────

    def reset(self) -> None:
        d0 = list(self.deck0_init)
        d1 = list(self.deck1_init)
        if self._shuffle_decks:
            self.rng.shuffle(d0)
            self.rng.shuffle(d1)

        lp0 = self._leader_power_for(self._leader0_id)
        lp1 = self._leader_power_for(self._leader1_id)
        ld0 = self._leader_defense_override if self._leader_defense_override else lp0
        ld1 = self._leader_defense_override if self._leader_defense_override else lp1
        self._leader_def0 = ld0
        self._leader_def1 = ld1

        dd0 = self._leader_don_deck_size(self._leader0_id)
        dd1 = self._leader_don_deck_size(self._leader1_id)
        self.p0 = PlayerState(
            deck=d0,
            don_deck=dd0,
            don_deck_initial=dd0,
            leader_id=self._leader0_id,
            leader_power=lp0,
        )
        self.p1 = PlayerState(
            deck=d1,
            don_deck=dd1,
            don_deck_initial=dd1,
            leader_id=self._leader1_id,
            leader_power=lp1,
        )
        self.p0.leader_restrictions = bootstrap_printed_restrictions_from_card(
            self.cards.get(self._leader0_id or "")
        )
        self.p1.leader_restrictions = bootstrap_printed_restrictions_from_card(
            self.cards.get(self._leader1_id or "")
        )
        self._active_player_idx: int = 0

        # Distribution de la main de départ (avant Mulligan)
        for _ in range(self.starting_hand):
            self._draw(self.p0)
            self._draw(self.p1)

        # Cartes vie
        self._fill_life(self.p0)
        self._fill_life(self.p1)

        self.phase = Phase.MULLIGAN   # La partie commence par la phase Mulligan
        self.done = False
        self._winner = None
        self._timeout_forced = False
        self._turn_cap_forced = False
        self._stall_no_card_forced = False
        self._total_cards_played_from_hand = 0
        self.turns_started = 0
        self._p0_don_phases = 0

        # État Blocker
        self._blocker_pending: bool = False
        self._blocker_atk_power: int = 0
        self._blocker_atk_double: bool = False
        self._blocker_atk_banish: bool = False
        self._opp_attacks_done: int = 0
        self._mulligan_p0_done = False
        self._mulligan_need_p1 = False
        self._policy_controls_p1 = False

        # ── Compteurs de séquence du tour du joueur actif ─────────────────────
        self._turn_steps: int = 0        # actions depuis le début du tour courant
        self._cards_played_turn: int = 0 # cartes jouées (perso + events) ce tour
        self._don_attached_turn: int = 0 # DON!! attachés ce tour
        self._main_nonidle_actions: int = 0  # actions Main « réelles » (hors passer en combat)
        self._attacks_made_turn: int = 0 # attaques déclarées ce tour
        self._last_action: int = -1      # dernière action choisie par P0 (-1 = aucune)
        # Dernière attaque du joueur piloté par la politique : 0=Leader, 1–5=board[0–4], -1=pass / autre
        self._last_agent_attack_slot: int = -1

        # Note: _start_turn(p0) est appelé dans step_mulligan, PAS ici.
        # Le 1er tour commence seulement après que P0 ait décidé du Mulligan.

    def force_end_by_timeout(self) -> None:
        """
        Termine la partie sans condition CR : tie-break par vie (plus de cartes vie = gagnant) ;
        égalité → tirage aléatoire. Utilisé quand ``max_episode_steps`` est atteint côté env.
        """
        if self.done:
            return
        self.done = True
        self._timeout_forced = True
        if self.p0.life > self.p1.life:
            self._winner = 0
        elif self.p1.life > self.p0.life:
            self._winner = 1
        else:
            self._winner = int(self.rng.integers(0, 2))

    def force_end_by_turn_cap(self) -> None:
        """
        Même tie-break que ``force_end_by_timeout`` : fin forcée quand ``turns_started``
        dépasse la limite (parties trop longues vs ~10–15 tours par joueur).
        """
        if self.done:
            return
        self.done = True
        self._turn_cap_forced = True
        if self.p0.life > self.p1.life:
            self._winner = 0
        elif self.p1.life > self.p0.life:
            self._winner = 1
        else:
            self._winner = int(self.rng.integers(0, 2))

    def force_end_by_stall_no_card(self) -> None:
        """
        Fin forcée : aucune carte jouée depuis la main (perso / event / stage) avant le seuil
        de ``turns_started`` (tie-break vie comme timeout / turn cap).
        """
        if self.done:
            return
        self.done = True
        self._stall_no_card_forced = True
        if self.p0.life > self.p1.life:
            self._winner = 0
        elif self.p1.life > self.p0.life:
            self._winner = 1
        else:
            self._winner = int(self.rng.integers(0, 2))

    # ── Mulligan ──────────────────────────────────────────────────────────────

    def step_mulligan(self, action: int) -> None:
        """
        CR 5.1 — Mulligan optionnel (une seule fois, avant le 1er tour).

        action == MULLIGAN_KEEP (0) : garder la main initiale.
        action == MULLIGAN_TAKE (1) : mélanger la main dans le deck et repiocher 5 cartes.

        Mode classique : P0 puis heuristique P1 en un appel, puis 1er tour P0.

        Mode self_play : premier appel = mulligan P0 uniquement ; second appel = mulligan P1
        (même politique), puis 1er tour P0.
        """
        if self.phase != Phase.MULLIGAN:
            return

        if not self._self_play_external:
            # ── P0 ──
            if action == MULLIGAN_TAKE:
                self.p0.deck.extend(self.p0.hand)
                self.p0.hand.clear()
                if self._shuffle_decks:
                    self.rng.shuffle(self.p0.deck)
                for _ in range(self.starting_hand):
                    if self.p0.deck:
                        self.p0.hand.append(self.p0.deck.pop(0))

            # ── P1 IA : mulligan si main avec < 2 cartes coût ≤ 3 ──
            p1_low_cost = sum(
                1 for cid in self.p1.hand
                if (c := self.cards.get(cid)) and c.cost <= 3
            )
            if p1_low_cost < 2:
                self.p1.deck.extend(self.p1.hand)
                self.p1.hand.clear()
                if self._shuffle_decks:
                    self.rng.shuffle(self.p1.deck)
                for _ in range(self.starting_hand):
                    if self.p1.deck:
                        self.p1.hand.append(self.p1.deck.pop(0))

            self.phase = Phase.MAIN
            self._start_turn(self.p0)
            return

        # ── Self-play : deux appels successifs ───────────────────────────────
        if not self._mulligan_p0_done:
            if action == MULLIGAN_TAKE:
                self.p0.deck.extend(self.p0.hand)
                self.p0.hand.clear()
                if self._shuffle_decks:
                    self.rng.shuffle(self.p0.deck)
                for _ in range(self.starting_hand):
                    if self.p0.deck:
                        self.p0.hand.append(self.p0.deck.pop(0))
            self._mulligan_p0_done = True
            self._mulligan_need_p1 = True
            return

        # Mulligan P1 (politique externe)
        if action == MULLIGAN_TAKE:
            self.p1.deck.extend(self.p1.hand)
            self.p1.hand.clear()
            if self._shuffle_decks:
                self.rng.shuffle(self.p1.deck)
            for _ in range(self.starting_hand):
                if self.p1.deck:
                    self.p1.hand.append(self.p1.deck.pop(0))

        self._mulligan_p0_done = False
        self._mulligan_need_p1 = False
        self.phase = Phase.MAIN
        self._start_turn(self.p0)

    # ── gestion du deck / pioche ──────────────────────────────────────────────

    def _draw(self, pl: PlayerState, *, force: bool = False) -> bool:
        """
        Pioche une carte (CR 4-5-1).
        Si le deck est vide au moment de la pioche → défaite immédiate (CR 9-2-1-2).
        Aucune limite de main officielle dans OPTCG ; ``max_hand`` ne borne pas la pioche ici.
        """
        if not pl.deck:
            self.done = True
            self._winner = 1 if pl is self.p0 else 0
            return False
        pl.hand.append(pl.deck.pop(0))
        return True

    def _leader_life_for(self, leader_id: str | None) -> int:
        """Renvoie la vraie valeur de vie du Leader depuis le CSV, ou starting_life par défaut."""
        if leader_id and leader_id in self.cards:
            life_val = self.cards[leader_id].life
            if life_val > 0:
                return life_val
        return self.starting_life

    def _fill_life(self, pl: PlayerState) -> None:
        """Place la valeur de vie du Leader (depuis le CSV) dans la pile vie."""
        n = self._leader_life_for(pl.leader_id)
        pl.life_cards.clear()
        for _ in range(n):
            if not pl.deck:
                break
            pl.life_cards.insert(0, pl.deck.pop(0))

    # ── vie ───────────────────────────────────────────────────────────────────

    def _take_life_damage(
        self,
        pl: PlayerState,
        *,
        has_banish: bool = False,
    ) -> float:
        """
        CR 4-6-2 + 1-2-1-1-1 : dégât sur le Leader (1 dégât).

        - Si 0 vie au moment du dégât → défaite (CR 9-2-1-1).
        - Sinon : carte du dessus vers main, sauf :
            [Banish]  (CR 10-1-3) → carte au trash, [Trigger] non activé
            [Trigger] (CR 10-1-5) → l'IA active l'effet (au lieu d'aller en main)

        Retourne un bonus RL si l'effet passif type Enel OP05-098 a été résolu (sinon 0).
        """
        if not pl.life_cards:
            self.done = True
            self._winner = 1 if pl is self.p0 else 0
            return 0.0
        card = pl.life_cards.pop(0)

        if has_banish:
            pl.trash.append(card)
            # Nami draw s'active même sur [Banish] (une carte est retirée de la pile vie)
            self._trigger_leader_auto_draw()
            return 0.0

        # Vérifier [Trigger] sur la carte vie révélée
        # CR 10-1-5-1 : le joueur CHOISIT d'activer ou non.
        #   YES → effet résolu, carte au trash (ou terrain si « Play this card »).
        #   NO  → carte en main.
        cd = self.cards.get(card)
        if cd and self._has_trigger(cd):
            if self._should_use_trigger(pl, cd):
                self._activate_trigger(pl, card, cd)   # YES : effet + trash/terrain
            else:
                pl.hand.append(card)                   # NO  : carte en main
        else:
            pl.hand.append(card)

        # Nami Leader [Your Turn] [Once Per Turn] : draw quand une vie est retirée
        self._trigger_leader_auto_draw()

        # Enel OP05-098 : [Opponent's Turn] quand la pile vie atteint 0 après ce dégât
        if not pl.life_cards:
            return self._maybe_opponent_turn_life_zero(pl)
        return 0.0

    def _maybe_opponent_turn_life_zero(self, pl: PlayerState) -> float:
        """
        Leader type Enel OP05-098 :
        « [Opponent's Turn][Once Per Turn] When your number of Life cards becomes 0,
        add 1 card from the top of your deck to the top of your Life cards.
        Then, trash 1 card from your hand. »
        Uniquement si ce n'est pas le tour du joueur ``pl`` (dégâts subis pendant le tour adverse).
        """
        if pl.life_cards:
            return 0.0
        if self._active_player_idx == (0 if pl is self.p0 else 1):
            return 0.0
        if pl.leader_opponent_turn_life_zero_used:
            return 0.0
        if not pl.leader_id:
            return 0.0
        leader_cd = self.cards.get(pl.leader_id)
        if not leader_cd:
            return 0.0
        blob = (leader_cd.card_text or "").replace("\u2019", "'").lower()
        if "[opponent's turn]" not in blob:
            return 0.0
        if not re.search(
            r"when\s+your\s+number\s+of\s+life\s+cards?\s+becomes?\s+0",
            blob,
            re.IGNORECASE,
        ):
            return 0.0

        pl.leader_opponent_turn_life_zero_used = True

        if pl.deck:
            pl.life_cards.insert(0, pl.deck.pop(0))
        if pl.hand:
            di = self._opp_choice_index(len(pl.hand))
            pl.trash.append(pl.hand.pop(di))
        return self._reward_leader_opp_turn_life0

    def _trigger_leader_auto_draw(self) -> None:
        """
        Leader [Your Turn] [Once Per Turn] — déclenchement quand une carte quitte une pile vie.
        Au lieu d'un check hardcodé, lit le texte du Leader pour trouver le segment d'effet,
        puis le passe au moteur générique (évaluation des conditions + application des effets).
        Exemple : Nami EB03-052 — « If you have 7 or less cards in your hand, draw 1 card. »
        """
        active_pl = self.p0 if self._active_player_idx == 0 else self.p1
        if active_pl.leader_once_per_turn_draw_used:
            return
        if not active_pl.leader_id:
            return
        leader_cd = self.cards.get(active_pl.leader_id)
        if not leader_cd:
            return
        text = leader_cd.card_text or ""
        # Détecter la présence du déclencheur [Your Turn] ... when a card is removed from ... Life
        if not re.search(
            r"\[Your\s+Turn\].*?when\s+a\s+card\s+is\s+removed\s+from.*?Life",
            text, re.IGNORECASE | re.DOTALL,
        ):
            return
        # Extraire la clause d'effet après le déclencheur (jusqu'au prochain marqueur ou fin)
        m_seg = re.search(
            r"when\s+a\s+card\s+is\s+removed\s+from[^.]+?\.\s*(.+?)(?:\s*\[|\Z)",
            text, re.IGNORECASE | re.DOTALL,
        )
        if not m_seg:
            return
        segment = m_seg.group(1).strip()
        if not segment:
            return
        # Utiliser le moteur générique : conditions + effets lus depuis le texte
        from .engine.on_play_resolver import _apply_rules_to_text
        from .engine.keyword_handlers import EffectContext
        owner_idx = 0 if active_pl is self.p0 else 1
        ctx = EffectContext(sim=self, owner=active_pl, owner_idx=owner_idx, card=leader_cd)
        _apply_rules_to_text(ctx, segment)
        active_pl.leader_once_per_turn_draw_used = True

    def _try_leader_defensive_effect(self, def_pl: PlayerState) -> None:
        """
        Nami Leader [DON!! x1] [On Your Opponent's Attack] [Once Per Turn] :
        Condition (CR [DON!! xN]) : le leader doit avoir >= N DON!! attachés.
        Coût     : défausser 1 carte de la main.
        Effet    : ce Leader gagne +2000 puissance ce tour.
        """
        if def_pl.leader_once_per_turn_def_used:
            return
        if not def_pl.leader_id:
            return
        leader_cd = self.cards.get(def_pl.leader_id)
        if not leader_cd:
            return
        text = leader_cd.card_text or ""
        m = re.search(
            r"\[On\s+Your\s+Opponent.?s\s+Attack\].*?\+(\d+)\s+power",
            text, re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return
        bonus = int(m.group(1))
        # [DON!! xN] = CONDITION : le leader doit avoir >= N DON!! attachés (pas un coût)
        don_req_m = re.search(r"\[DON!!\s*x(\d+)\]", text, re.IGNORECASE)
        don_required = int(don_req_m.group(1)) if don_req_m else 1
        if def_pl.leader_attached_don < don_required:
            return   # condition non remplie : pas assez de DON!! attachés au leader
        if not def_pl.hand:
            return   # coût : il faut pouvoir défausser 1 carte
        discard_idx = self._opp_choice_index(len(def_pl.hand))
        def_pl.trash.append(def_pl.hand.pop(discard_idx))
        def_pl.leader_power_modifiers.bonus_turn += bonus
        def_pl.leader_once_per_turn_def_used = True

    def _activate_trigger(self, pl: PlayerState, card_id: str, cd: "CardDef") -> None:
        """
        CR 10-1-5 : active l'effet [Trigger] (le joueur a déjà dit OUI via _should_use_trigger).

        Deux cas :
          Cas A — « Play this card » (CR 10-1-5-2) :
              La carte est jouée sur le terrain (reposée, comme toute carte jouée en cours de
              combat). Son effet [On Play] se déclenche normalement.
              La carte NE va PAS au trash.

          Cas B — Tout autre effet (draw, damage, life, DON!!, etc.) :
              L'effet se résout via le moteur de règles.
              La carte va ensuite au TRASH (CR 10-1-5-3).
        """
        from .engine.keyword_handlers import EffectContext
        from .engine.on_play_resolver import (
            _apply_rules_to_text,
            _clip_timing_segment_body,
            extract_timing_segment,
        )

        owner_idx = 0 if pl is self.p0 else 1
        text = cd.card_text or ""
        segment = extract_timing_segment(text, "trigger") or _clip_timing_segment_body(text)

        # ── Cas A : "Play this card" ──────────────────────────────────────────
        if self._trigger_plays_self(cd):
            ct = (cd.card_type or "").lower()
            if "character" in ct:
                # Jouer rested sur le terrain ; [On Play] se déclenche via _play_char_to_board
                # La carte est jouée comme si c'était un coût 0 pendant la phase adverse
                self._play_char_to_board(pl, owner_idx, card_id, cd)
                # La carte est sur le terrain — elle ne va ni en main ni au trash
                return
            if "stage" in ct:
                self._play_stage(pl, owner_idx, card_id, cd)
                return
            # Autres types : tomber sur le cas B (trash) en fallback

        # ── Cas B : effet ordinaire → résoudre puis trash ─────────────────────
        if segment:
            ctx = EffectContext(sim=self, owner=pl, owner_idx=owner_idx, card=cd)
            _apply_rules_to_text(ctx, segment)

        pl.trash.append(card_id)  # CR 10-1-5-3

    # ── DON!! ─────────────────────────────────────────────────────────────────

    def _don_phase(self, pl: PlayerState, *, is_p0: bool) -> None:
        """Gain de DON!! depuis le deck : 1 pour P0 au 1er tour, 2 sinon."""
        if pl.don_deck <= 0:
            return
        if is_p0 and self._p0_don_phases == 0:
            want = min(self.official_first_player_don, pl.don_deck)
        else:
            want = min(self.don_per_turn, pl.don_deck)
        in_cost = pl.don_active + pl.don_rested
        gain = min(want, max(0, self.max_don - in_cost))
        pl.don_deck -= gain
        pl.don_active += gain
        if is_p0:
            self._p0_don_phases += 1

    def _pay_don(self, pl: PlayerState, cost: int) -> bool:
        """Repose `cost` DON!! actifs (paiement d'un coût)."""
        if cost < 0:
            return False
        if cost == 0:
            return True
        if pl.don_active < cost:
            return False
        pl.don_active -= cost
        pl.don_rested += cost
        return True

    # ── refresh (début de tour) ───────────────────────────────────────────────

    def _refresh_phase(self, pl: PlayerState) -> None:
        """
        Phase Refresh (CR 6.1) :
        1. Les DON!! attachés aux personnages reviennent en zone de coût (reposés).
        2. Tous les DON!! de la zone de coût deviennent actifs.
        3. Tous les personnages ET le Leader se déreposent.
        """
        # Récupérer DON!! attachés aux persos
        for b in pl.board:
            if b.attached_don:
                pl.don_rested += b.attached_don
                b.attached_don = 0
        # Récupérer DON!! attaché au Leader
        if pl.leader_attached_don:
            pl.don_rested += pl.leader_attached_don
            pl.leader_attached_don = 0
        # Activer tous les DON!! de la zone de coût
        pl.don_active += pl.don_rested
        pl.don_rested = 0
        # Dereposer persos et Leader (CR « cannot become Active » : reste reposé)
        for b in pl.board:
            was_rested = b.rested
            must_stay_rested = b.restrictions.blocks_becoming_active_after_refresh(was_rested)
            b.restrictions.clear_turn_scoped()
            b.power_modifiers.clear_turn_scoped()
            b.keyword_grants.clear_turn_scoped()
            b.just_played = False
            if must_stay_rested:
                b.rested = True
            else:
                b.rested = False
        pl.leader_restrictions.clear_turn_scoped()
        pl.leader_rested = False
        # Réinitialiser les bonus et flags [Once Per Turn] du leader
        pl.leader_power_modifiers.clear_turn_scoped()
        pl.leader_once_per_turn_draw_used = False
        pl.leader_once_per_turn_def_used  = False
        pl.leader_opponent_turn_life_zero_used = False
        pl.leader_activate_main_used      = False

    # ── structure de tour ─────────────────────────────────────────────────────

    def _start_turn(self, pl: PlayerState) -> None:
        """
        Exécute les phases automatiques de début de tour :
        Refresh → (Draw) → DON!! → (prêt pour Main Phase).
        """
        self.turns_started += 1
        is_p0 = pl is self.p0
        self._active_player_idx = 0 if is_p0 else 1

        # Partie « idle » : après N débuts de tour (ex. 7 = fin du 6e tour alterné = 3 tours par joueur),
        # si personne n’a joué de carte depuis la main → fin + pénalité côté env.
        if (
            self._stall_no_card_after_ts > 0
            and self.turns_started >= self._stall_no_card_after_ts
            and self._total_cards_played_from_hand == 0
        ):
            self.force_end_by_stall_no_card()
            return

        # Réinitialiser les compteurs de séquence au début de chaque tour (P0 ou P1)
        self._turn_steps = 0
        self._cards_played_turn = 0
        self._don_attached_turn = 0
        self._main_nonidle_actions = 0
        self._attacks_made_turn = 0
        # _last_action est conservé d'un tour sur l'autre (info de contexte)

        if pl.last_main_finished_idle and self.player_turn_number(pl) >= 2:
            pl.boost_next_main_activity = True
        pl.last_main_finished_idle = False

        # Refresh
        self._refresh_phase(pl)

        # Draw (P0 ne pioche pas à son tout premier tour)
        first_turn_first_player = is_p0 and self.turns_started == 1
        if not (self.skip_first_draw_first_player and first_turn_first_player):
            for _ in range(self.draw_per_turn):
                if not self._draw(pl):
                    return  # défaite par deck vide

        # DON!!
        self._don_phase(pl, is_p0=is_p0)

    def _end_turn(self, pl: PlayerState) -> None:
        """
        Phase End (CR 6-6) : pas de défausse à la fin du tour dans le jeu officiel.
        Déclenche les effets [End of Your Turn] sur tous les personnages et le Leader.
        """
        from .engine.keyword_handlers import EffectContext, dispatch_timing
        owner_idx = 0 if pl is self.p0 else 1

        # Effets [End of Your Turn] sur les personnages du board
        for bc in list(pl.board):
            cd = self.cards.get(bc.card_id)
            if cd and cd.card_text:
                blob = cd.card_text.lower()
                if "[end of your turn]" in blob:
                    ctx = EffectContext(sim=self, owner=pl, owner_idx=owner_idx, card=cd)
                    dispatch_timing("End of Your Turn", ctx)

        # Effets [End of Your Turn] sur le Leader
        if pl.leader_id:
            leader_cd = self.cards.get(pl.leader_id)
            if leader_cd and leader_cd.card_text:
                blob = leader_cd.card_text.lower()
                if "[end of your turn]" in blob:
                    ctx = EffectContext(sim=self, owner=pl, owner_idx=owner_idx, card=leader_cd)
                    dispatch_timing("End of Your Turn", ctx)

    # ── vérification de victoire ──────────────────────────────────────────────

    def _check_win(self) -> None:
        """
        CR 9-2 : vérification des conditions de défaite.
        La défaite n'arrive PAS automatiquement quand vie = 0 ;
        elle arrive quand le Leader prend un dégât alors que la vie est déjà à 0.
        C'est géré dans `_take_life_damage`.
        La seule vérification ici : deck vide (CR 9-2-1-2).
        """
        if self.done:
            return
        if not self.p0.deck and not self.p0.life_cards and not self.p0.hand:
            pass  # deck vide est vérifié dans _draw
        if not self.p1.deck and not self.p1.life_cards and not self.p1.hand:
            pass

    def _check_win_lives(self) -> None:
        """Alias pour compatibilité (plus utilisé directement, géré dans _take_life_damage)."""
        pass

    # ── blockers ──────────────────────────────────────────────────────────────

    def _active_blockers(self, pl: PlayerState) -> list[tuple[int, BoardChar]]:
        """Personnages actifs (non reposés) avec [Blocker], sans blocage d’attaque (Perona, etc.)."""
        return [
            (i, b)
            for i, b in enumerate(pl.board)
            if not b.rested
            and b.has_blocker
            and not b.restrictions.blocks_attack_declaration()
        ]

    def _resolve_blocker(self, def_pl: PlayerState, declared_target: int) -> int:
        """
        CR 10-1-4 : [Blocker] peut intercepter TOUTE attaque (Leader OU Character adverse).
        L'IA défenseur choisit le Blocker le plus fort.
        Retourne le slot réel final (0=Leader ou 1–5=board slot du Blocker).
        """
        blockers = self._active_blockers(def_pl)
        if not blockers:
            return declared_target
        # Choisit le Blocker avec la puissance la plus élevée (heuristique IA)
        bi, bb = max(blockers, key=lambda t: effective_power(t[1], self.cards))
        bb.rested = True  # CR 10-1-4-2 : le Blocker se repose en intervenant
        # Déclencher [On Block] du Blocker (CR 10-2-15)
        bb_cd = self.cards.get(bb.card_id)
        if bb_cd:
            blob = (bb_cd.card_text or "").lower()
            if "[on block]" in blob:
                from .engine.keyword_handlers import EffectContext, dispatch_timing
                owner_idx = 0 if def_pl is self.p0 else 1
                ctx = EffectContext(sim=self, owner=def_pl, owner_idx=owner_idx, card=bb_cd)
                dispatch_timing("On Block", ctx)
        return bi + 1  # code 1–5 (board slot du Blocker)

    # ── légalité d'une attaque ────────────────────────────────────────────────

    def _attacker_ok(self, atk_pl: PlayerState, atk_slot: int) -> bool:
        """slot 0 = Leader, 1–5 = board[slot-1]."""
        if atk_slot == 0:
            return leader_may_declare_attack(
                leader_rested=atk_pl.leader_rested,
                restrictions=atk_pl.leader_restrictions,
            )
        idx = atk_slot - 1
        if idx < 0 or idx >= len(atk_pl.board):
            return False
        b = atk_pl.board[idx]
        return character_may_declare_attack(
            rested=b.rested,
            just_played=b.just_played,
            has_rush=b.has_rush,
            has_rush_char=b.has_rush_char,
            restrictions=b.restrictions,
            keyword_grants=b.keyword_grants,
        )

    def _target_ok(
        self,
        atk_pl: PlayerState,
        atk_slot: int,
        def_pl: PlayerState,
        tgt_slot: int,
    ) -> bool:
        """
        CR 7-1-1-2 : cible = Leader ou personnage adverse REPOSÉ,
        sauf texte / buff « attack active Character(s) » → perso actif autorisé.
        """
        if tgt_slot == 0:
            return True
        idx = tgt_slot - 1
        if not (0 <= idx < len(def_pl.board)):
            return False
        tgt = def_pl.board[idx]
        if tgt.rested:
            return True
        if atk_slot == 0:
            atk_cd = self.cards.get(atk_pl.leader_id or "")
            return bool(atk_cd and self._has_attack_active_chars(atk_cd))
        bc = atk_pl.board[atk_slot - 1]
        return self._can_attack_active_opponent_chars(bc)

    def attack_is_legal(self, atk_pl: PlayerState, def_pl: PlayerState,
                        atk_slot: int, tgt_slot: int) -> bool:
        """
        CR 6-5-6-1 : aucune attaque au premier tour de chaque joueur.
        CR 10-1-6   : [Rush: Character] → le tour de pose, uniquement les persos (tgt_slot ≥ 1).
        """
        # Aucune attaque au premier tour du joueur (CR 6-5-6-1)
        if atk_pl is self.p0 and self.turns_started <= 1:
            return False
        if atk_pl is self.p1 and self.turns_started <= 2:
            return False
        if not self._attacker_ok(atk_pl, atk_slot) or not self._target_ok(
            atk_pl, atk_slot, def_pl, tgt_slot
        ):
            return False
        # [Rush: Character] (+ grants) : le tour de pose, pas le Leader adverse (CR 10-1-6-1)
        if atk_slot >= 1:
            bc = atk_pl.board[atk_slot - 1]
            if self._eff_rush_char(bc) and bc.just_played and tgt_slot == 0:
                return False
        return True

    # ── résolution d'une attaque ──────────────────────────────────────────────

    def _attacker_power(self, atk_pl: PlayerState, atk_slot: int) -> int:
        if atk_slot == 0:
            return atk_pl.effective_leader_power()
        return effective_power(atk_pl.board[atk_slot - 1], self.cards)

    def _rest_attacker(self, atk_pl: PlayerState, atk_slot: int) -> None:
        if atk_slot == 0:
            atk_pl.leader_rested = True
        else:
            atk_pl.board[atk_slot - 1].rested = True

    def _resolve_attack(
        self,
        atk_pl: PlayerState,
        def_pl: PlayerState,
        atk_slot: int,      # 0=Leader, 1–3=board[slot-1]
        tgt_slot: int,      # 0=Leader, 1–3=board[slot-1]
        *,
        agent_is_attacker: bool,
    ) -> float:
        """
        Résolution complète d'une attaque (CR 7-1) :

        1. Légalité (CR 7-1-1-1)
        2. Attack Step  : repos de l'attaquant (CR 7-1-1-1)
        3. Block Step   : Blocker redirige sauf [Unblockable] (CR 7-1-2 / 10-1-7)
        4. Counter Step : défenseur joue ses Counter (CR 7-1-3)
        5. Damage Step  :
           - Cible Leader : si atk ≥ def → dégâts vie (CR 7-1-4-1-1)
               [Double Attack] (CR 10-1-2) → 2 dégâts
               [Banish]        (CR 10-1-3) → carte vie au trash
           - Cible perso  : si atk ≥ def → K.O. (trash) (CR 7-1-4-1-2)
        NB : l'attaquant n'est JAMAIS K.O. par un combat.
        """
        reward = 0.0

        if not self.attack_is_legal(atk_pl, def_pl, atk_slot, tgt_slot):
            return -0.05

        if agent_is_attacker:
            self._last_agent_attack_slot = int(atk_slot)

        # ── Lire les mots-clés de l'attaquant ──
        if atk_slot == 0:
            atk_cd = self.cards.get(atk_pl.leader_id or "")
            atk_double  = False   # Le Leader ne peut pas avoir [Double Attack] via BoardChar
            atk_banish  = False
            atk_unblock = False
            if atk_cd:
                atk_double  = self._has_double_attack(atk_cd)
                atk_banish  = self._has_banish(atk_cd)
                atk_unblock = self._has_unblockable(atk_cd)
        else:
            bc = atk_pl.board[atk_slot - 1]
            atk_cd = self.cards.get(bc.card_id)
            atk_double = self._char_double_attack(bc, atk_cd)
            atk_banish = self._char_banish(bc, atk_cd)
            atk_unblock = self._char_unblockable(bc, atk_cd)

        # ── Étape 2 : Déclaration → repos de l'attaquant ──
        self._rest_attacker(atk_pl, atk_slot)
        self._attacks_made_turn += 1
        atk_pow = self._attacker_power(atk_pl, atk_slot)

        # Effets [When Attacking] de l'attaquant (CR 10-2-5)
        if atk_cd:
            blob = (atk_cd.card_text or "").lower()
            if "[when attacking]" in blob:
                from .engine.keyword_handlers import EffectContext, dispatch_timing
                wa_idx = 0 if atk_pl is self.p0 else 1
                wa_ctx = EffectContext(sim=self, owner=atk_pl, owner_idx=wa_idx, card=atk_cd)
                dispatch_timing("When Attacking", wa_ctx)

        # [On Your Opponent's Attack] sur les cartes du défenseur (CR 10-2-16)
        # Se déclenche après les effets [When Attacking] de l'attaquant
        from .engine.keyword_handlers import EffectContext, dispatch_timing
        def_idx = 0 if def_pl is self.p0 else 1
        _ooa_blob = "[on your opponent's attack]"
        # Leader défenseur
        _ldr_cd_def = self.cards.get(def_pl.leader_id or "")
        if _ldr_cd_def and _ooa_blob in ((_ldr_cd_def.card_text or "").lower()):
            _ooa_ctx = EffectContext(sim=self, owner=def_pl, owner_idx=def_idx, card=_ldr_cd_def)
            dispatch_timing("On Your Opponent's Attack", _ooa_ctx)
        # Characters défenseurs
        for _bc_def in list(def_pl.board):
            _bc_def_cd = self.cards.get(_bc_def.card_id)
            if _bc_def_cd and _ooa_blob in ((_bc_def_cd.card_text or "").lower()):
                _ooa_ctx = EffectContext(sim=self, owner=def_pl, owner_idx=def_idx, card=_bc_def_cd)
                dispatch_timing("On Your Opponent's Attack", _ooa_ctx)

        # ── Étape 3 : Block Step (CR 7-1-2) ──
        # [Unblockable] (CR 10-1-7) : l'adversaire ne peut pas activer [Blocker]
        if atk_unblock:
            real_tgt = tgt_slot
        else:
            real_tgt = self._resolve_blocker(def_pl, tgt_slot)

        # ── Étape 4 : Counter Step ──
        if real_tgt == 0:
            # Attaque sur le Leader
            self._try_leader_defensive_effect(def_pl)
            # CR 6-5-5-2 : pas de +1000 par DON!! attaché hors du tour du propriétaire (combat).
            leader_def = def_pl.effective_leader_power(include_attached_don_bonus=False)
            defense = apply_counter_stack_until_safe(def_pl, self.cards, atk_pow, leader_def)

            # ── Étape 5 : Damage Step ──
            if atk_pow >= defense:
                n_dmg = 2 if atk_double else 1   # [Double Attack] (CR 10-1-2)
                enel_life0 = 0.0
                for _ in range(n_dmg):
                    if self.done:
                        break
                    enel_life0 += self._take_life_damage(def_pl, has_banish=atk_banish)
                if agent_is_attacker:
                    reward += 0.6 * n_dmg
                    # Enel jaune (vie→0 tour adverse) : avantage défenseur → pénalise l’attaquant agent
                    reward -= enel_life0
                else:
                    # Tour adverse simulé : avantage P0 si son Enel s’active
                    reward += enel_life0
                self._check_win_lives()
        else:
            # Attaque sur un personnage (CR 7-1-4-1-2)
            ci = real_tgt - 1
            if ci >= len(def_pl.board):
                return -0.05  # La cible a disparu (Blocker invalide)
            defc = def_pl.board[ci]
            # CR 6-5-5-2 : DON!! +1000 seulement pendant le tour du propriétaire.
            # En défense (tour adverse), on n'ajoute pas le bonus DON!! du défenseur.
            # Puissance défense : imprimée + boost/malus d’effet, sans DON!! attachés (tour adverse)
            base_def = board_character_power(
                defc, self.cards, include_attached_don_bonus=False
            )
            defense = apply_counter_stack_until_safe(def_pl, self.cards, atk_pow, base_def)
            if atk_pow >= defense:
                if defc.restrictions.blocks_ko():
                    pass  # ne peut pas être K.O. par cette attaque (texte / effet ce tour)
                else:
                    killed = def_pl.board.pop(ci)
                    # DON!! attaché revient IMMÉDIATEMENT dans la zone reposée (CR 3-4-2)
                    if killed.attached_don:
                        def_pl.don_rested += killed.attached_don
                    def_pl.trash.append(killed.card_id)
                    if agent_is_attacker:
                        reward += 0.4
                    # Déclencher l'effet [On K.O.] de la carte tuée
                    killed_cd = self.cards.get(killed.card_id)
                    if killed_cd:
                        from .engine.keyword_handlers import EffectContext, kw_on_ko
                        owner_idx = 1 if def_pl is self.p1 else 0
                        ko_ctx = EffectContext(
                            sim=self,
                            owner=def_pl,
                            owner_idx=owner_idx,
                            card=killed_cd,
                        )
                        kw_on_ko(ko_ctx)

        return reward

    # ── Tour adversaire avec pause Blocker ────────────────────────────────────

    def _finish_opponent_turn(self) -> None:
        """Fin du tour de P1 → début du tour de P0 (phase MAIN)."""
        self._end_turn(self.p1)
        if not self.done:
            self._start_turn(self.p0)
            self.phase = Phase.MAIN

    def _process_one_opponent_attack(self) -> bool:
        """
        Sélectionne et traite UNE attaque de P1.
        Retourne True si mis en pause (phase = BLOCKER) pour que P0 choisisse un Blocker.
        Retourne False si l'attaque a été résolue sans pause, ou si plus d'attaques légales.
        """
        if self._opp_attacks_done >= 6:
            return False

        legal: list[tuple[int, int]] = [
            (a, t)
            for a in range(N_ATTACKERS)
            for t in range(N_TARGETS)
            if self.attack_is_legal(self.p1, self.p0, a, t)
        ]
        if not legal:
            return False

        # Exploration pure : attaque aléatoire parmi les légales
        atk_s, tgt_s = legal[self._opp_choice_index(len(legal))]

        # ── Vérifier si une décision Blocker est nécessaire pour P0 ──────────
        # CR 10-1-4 : [Blocker] peut intercepter TOUTE attaque (Leader OU perso)
        if atk_s == 0:
            atk_cd = self.cards.get(self.p1.leader_id or "")
            atk_pow = self.p1.effective_leader_power()
            atk_unblockable = bool(atk_cd and self._has_unblockable(atk_cd))
            atk_double = bool(atk_cd and self._has_double_attack(atk_cd))
            atk_banish = bool(atk_cd and self._has_banish(atk_cd))
        else:
            bc = self.p1.board[atk_s - 1]
            atk_cd = self.cards.get(bc.card_id)
            atk_pow = effective_power(bc, self.cards)
            atk_unblockable = self._char_unblockable(bc, atk_cd)
            atk_double = self._char_double_attack(bc, atk_cd)
            atk_banish = self._char_banish(bc, atk_cd)
        eligible_blockers = [
            b for b in self.p0.board
            if b.has_blocker
            and not b.rested
            and not b.restrictions.blocks_attack_declaration()
        ]

        if eligible_blockers and not atk_unblockable:
            # Étape 2 (CR 7-1-1-1) : repos de l'attaquant AVANT la pause
            self._rest_attacker(self.p1, atk_s)
            self._attacks_made_turn += 1
            # Stocker l'état pour step_blocker
            self._blocker_pending = True
            self._blocker_atk_power = atk_pow
            self._blocker_atk_target = tgt_s         # 0=Leader, 1–5=perso
            self._blocker_atk_double = atk_double
            self._blocker_atk_banish = atk_banish
            self.phase = Phase.BLOCKER
            return True   # Pause — step_blocker sera appelé

        # Pas de pause → résoudre directement
        r_atk = self._resolve_attack(
            self.p1, self.p0, atk_s, tgt_s, agent_is_attacker=False,
        )
        self._opponent_battle_extra_reward += r_atk
        self._opp_attacks_done += 1
        return False

    def _run_opponent_battle(self) -> bool:
        """
        Traite les attaques restantes de P1 (jusqu'à 6 - _opp_attacks_done).
        Retourne True si mis en pause pour Blocker (phase = BLOCKER).
        Retourne False quand toutes les attaques sont traitées.
        """
        remaining = 6 - self._opp_attacks_done
        for _ in range(max(0, remaining)):
            if self.done:
                return False
            if self._opp_attacks_done >= 6:
                return False
            # Vérifier s'il reste des attaques légales avant de tenter
            legal_exist = any(
                self.attack_is_legal(self.p1, self.p0, a, t)
                for a in range(N_ATTACKERS)
                for t in range(N_TARGETS)
            )
            if not legal_exist:
                return False
            paused = self._process_one_opponent_attack()
            if paused:
                return True
        return False

    def step_blocker(self, blocker_action: int) -> float:
        """
        P0 répond à une attaque adverse avec un [Blocker] (CR 10-1-4, CR 7-1-2).

        blocker_action == BLOCKER_PASS (0)       : pas de Blocker, la cible initiale prend les dégâts.
        blocker_action == BLOCKER_SLOT_BASE + i  : utiliser board[i] comme Blocker.
            → Le Blocker se repose ; si atk ≥ def → K.O. du Blocker (cible initiale indemne).
            → Si atk < def → Blocker survit reposé, cible indemne.

        Après résolution, continue les attaques restantes de P1.
        Si une autre attaque exige encore un Blocker → retourne sans terminer le tour P1.
        """
        if self.phase != Phase.BLOCKER or not self._blocker_pending:
            return -0.05

        self._turn_steps += 1
        self._last_action = blocker_action
        self._blocker_pending = False
        reward = 0.0

        atk_pow    = self._blocker_atk_power
        atk_double = self._blocker_atk_double
        atk_banish = self._blocker_atk_banish
        orig_tgt   = getattr(self, "_blocker_atk_target", 0)  # cible originale (0=Leader, 1–5=perso)

        use_slot = blocker_action - BLOCKER_SLOT_BASE  # index board (≥0 si valide)
        valid_blocker = (
            0 <= use_slot < len(self.p0.board)
            and self.p0.board[use_slot].has_blocker
            and not self.p0.board[use_slot].rested
            and not self.p0.board[use_slot].restrictions.blocks_attack_declaration()
        )

        if blocker_action == BLOCKER_PASS or not valid_blocker:
            # ── Pas de Blocker → la cible originale absorbe le dégât ──────
            if orig_tgt == 0:
                # Cible = Leader de P0
                self._try_leader_defensive_effect(self.p0)
                leader_def = self.p0.effective_leader_power(include_attached_don_bonus=False)
                defense = apply_counter_stack_until_safe(
                    self.p0, self.cards, atk_pow, leader_def
                )
                if atk_pow >= defense:
                    n_dmg = 2 if atk_double else 1
                    enel_life0 = 0.0
                    for _ in range(n_dmg):
                        if self.done:
                            break
                        enel_life0 += self._take_life_damage(self.p0, has_banish=atk_banish)
                    reward -= 0.3 * n_dmg
                    reward += enel_life0
                self._check_win_lives()
            else:
                # Cible = perso de P0 — le perso peut être K.O. (CR 7-1-4-1-2)
                ci = orig_tgt - 1
                if ci < len(self.p0.board):
                    defc = self.p0.board[ci]
                    defense = apply_counter_stack_until_safe(
                        self.p0,
                        self.cards,
                        atk_pow,
                        board_character_power(defc, self.cards, include_attached_don_bonus=False),
                    )
                    if atk_pow >= defense:
                        if defc.restrictions.blocks_ko():
                            pass
                        else:
                            killed = self.p0.board.pop(ci)
                            if killed.attached_don:
                                self.p0.don_rested += killed.attached_don
                            self.p0.trash.append(killed.card_id)
                            reward -= 0.2
                            killed_cd = self.cards.get(killed.card_id)
                            if killed_cd:
                                from .engine.keyword_handlers import EffectContext, kw_on_ko
                                ko_ctx = EffectContext(
                                    sim=self, owner=self.p0, owner_idx=0, card=killed_cd
                                )
                                kw_on_ko(ko_ctx)
        else:
            # ── Blocker choisi par P0 (CR 10-1-6) ─────────────────────────
            blk = self.p0.board[use_slot]
            blk.rested = True        # le Blocker se repose en intervenant
            # Pas de bonus DON!! pour le défenseur pendant le tour adverse (CR 6-5-5-2)
            def_pow = board_character_power(blk, self.cards, include_attached_don_bonus=False)
            defense = apply_counter_stack_until_safe(
                self.p0, self.cards, atk_pow, def_pow
            )
            if atk_pow >= defense:
                if blk.restrictions.blocks_ko():
                    pass
                else:
                    killed = self.p0.board.pop(use_slot)
                    if killed.attached_don:
                        self.p0.don_rested += killed.attached_don
                    self.p0.trash.append(killed.card_id)
                    reward -= 0.2
                    killed_cd = self.cards.get(killed.card_id)
                    if killed_cd:
                        from .engine.keyword_handlers import EffectContext, kw_on_ko
                        ko_ctx = EffectContext(
                            sim=self, owner=self.p0, owner_idx=0, card=killed_cd
                        )
                        kw_on_ko(ko_ctx)
            # Si Blocker survit : reste reposé, Leader indemne

        self._opp_attacks_done += 1

        if self.done:
            self.phase = Phase.MAIN
            reward += self._opponent_battle_extra_reward
            self._opponent_battle_extra_reward = 0.0
            return reward

        # Reprendre les attaques restantes de P1
        self.phase = Phase.MAIN  # _run_opponent_battle peut repasser à BLOCKER si besoin
        paused = self._run_opponent_battle()
        reward += self._opponent_battle_extra_reward
        self._opponent_battle_extra_reward = 0.0
        if paused:
            # Une autre attaque nécessite un Blocker → rester en pause
            return reward

        # Toutes les attaques de P1 terminées → fin du tour P1 + début du tour P0
        if not self.done:
            self._finish_opponent_turn()

        return reward

    # ── IA adversaire ─────────────────────────────────────────────────────────

    def _run_ai_battle(self, atk_pl: PlayerState, def_pl: PlayerState,
                       *, agent_is_attacker: bool) -> None:
        """
        Combat IA — exploration aléatoire pure.
        Sélectionne au hasard parmi les attaques légales pour créer de la
        diversité dans les situations d'entraînement de P0.
        """
        for _ in range(6):
            if self.done:
                return
            legal: list[tuple[int, int]] = [
                (a, t)
                for a in range(N_ATTACKERS)
                for t in range(N_TARGETS)
                if self.attack_is_legal(atk_pl, def_pl, a, t)
            ]
            if not legal:
                break
            atk_s, tgt_s = legal[self._opp_choice_index(len(legal))]
            self._resolve_attack(atk_pl, def_pl, atk_s, tgt_s, agent_is_attacker=agent_is_attacker)
            if self.done:
                return

    # ── tour adversaire complet ───────────────────────────────────────────────

    def _simulate_opponent_turn(self) -> None:
        """
        Tour de P1 — exploration aléatoire pure.

        P1 ne suit aucune stratégie fixe : il choisit aléatoirement parmi
        toutes ses actions légales.  Cela génère une grande diversité de
        situations d'entraînement et force P0 à apprendre à gérer n'importe
        quel adversaire, pas seulement une heuristique prédéfinie.
        """
        self._opponent_battle_extra_reward = 0.0
        self._opp_attacks_done = 0

        self._start_turn(self.p1)
        if self.done:
            return

        # ── Phase Main : jouer et/ou attacher DON!! dans un ordre aléatoire ──
        # On mélange les options à chaque pas pour éviter tout biais.
        max_actions = (self.max_board + 1) * 2 + 4  # borne haute
        for _ in range(max_actions):
            if self.done:
                break

            ctx = PlayabilityContext(Phase.MAIN, self.p1.don, len(self.p1.hand))

            # Construire la liste de toutes les micro-actions légales disponibles
            options: list[tuple[str, int]] = []   # (type, index)

            # Cartes jouables (Characters, Stages, Events)
            for i, cid in enumerate(self.p1.hand):
                cd = self._card(cid)
                if can_play_card(cd, ctx):
                    options.append(("card", i))

            # Attacher DON!! au Leader ou à un perso
            if self.p1.don_active > 0:
                options.append(("don", 0))   # Leader
                for i in range(len(self.p1.board)):
                    options.append(("don", i + 1))

            if not options:
                break  # Plus rien à faire en Main Phase

            # Choisir une action au hasard (exploration pure)
            kind, idx = options[self._opp_choice_index(len(options))]

            if kind == "card":
                cid = self.p1.hand[idx]
                cd = self._card(cid)
                if not self._pay_don(self.p1, cd.cost):
                    break
                self.p1.hand.pop(idx)
                card_type = (cd.card_type or "").lower()
                if "event" in card_type:
                    self._play_event(self.p1, 1, cid, cd)
                elif "stage" in card_type:
                    self._play_stage(self.p1, 1, cid, cd)
                else:
                    self._play_char_to_board(self.p1, 1, cid, cd)
            else:  # "don"
                self._attach_don(self.p1, idx, 1)

        if self.done:
            return

        # ── Phase Battle : traiter les attaques une par une ────────────────────
        paused = self._run_opponent_battle()
        if paused:
            return   # P0 doit choisir un Blocker → ne pas finir le tour ici

        if not self.done:
            self._finish_opponent_turn()

    # ── interface publique (env.py) ───────────────────────────────────────────

    def battle_damage_allowed(self) -> bool:
        """
        CR 6-5-6-1 : le combat est interdit au premier tour de chaque joueur.
        La légalité de l'attaque est gérée par attack_is_legal() ; les dégâts
        sont toujours autorisés dès qu'une attaque est légale.
        """
        return True

    def _play_char_to_board(
        self,
        pl: PlayerState,
        owner_idx: int,
        cid: str,
        cd: "CardDef",
        *,
        just_played: bool = True,
    ) -> None:
        """
        Place un personnage sur le board (CR 3-4-1).
        Si le board est plein (max_board) : le joueur CHOISIT un perso à remplacer
        (envoyé au trash, pas de [On K.O.], DON!! attaché revient reposé — CR 3-4-3).
        Dans cette IA, on remplace toujours le perso avec la plus faible puissance effective.
        """
        if len(pl.board) >= self.max_board:
            # Remplacement : choisir le perso le moins puissant (heuristique IA)
            weakest_idx = min(
                range(len(pl.board)),
                key=lambda i: effective_power(pl.board[i], self.cards),
            )
            replaced = pl.board.pop(weakest_idx)
            # DON!! attaché revient reposé immédiatement (CR 3-4-3)
            if replaced.attached_don:
                pl.don_rested += replaced.attached_don
            pl.trash.append(replaced.card_id)
            # Pas de [On K.O.] — le remplacement n'est pas un K.O. (CR 3-4-3)

        bc = BoardChar(
            cid,
            cd.power,
            has_rush=self._has_rush(cd),
            has_rush_char=self._has_rush_char(cd),
            has_blocker=self._has_blocker(cd),
            has_double_attack=self._has_double_attack(cd),
            has_unblockable=self._has_unblockable(cd),
            has_attack_active=self._has_attack_active_chars(cd),
            just_played=just_played,
            restrictions=bootstrap_printed_restrictions_from_card(cd),
        )
        pl.board.append(bc)
        dispatch_on_play(EffectContext(sim=self, owner=pl, owner_idx=owner_idx, card=cd))

    def _play_event(
        self,
        pl: PlayerState,
        owner_idx: int,
        cid: str,
        cd: "CardDef",
    ) -> float:
        """
        Résout une carte Event jouée en Main Phase (CR 10-2) :
          1. Extraire et appliquer l'effet [Main] (ou le texte entier si pas de timing).
          2. La carte va au trash (CR 10-2-3).
        Retourne une récompense positive si l'effet est utile.
        """
        from .engine.keyword_handlers import EffectContext
        from .engine.on_play_resolver import (
            _apply_rules_to_text,
            _clip_timing_segment_body,
            extract_timing_segment,
        )

        text = cd.card_text or ""
        segment = extract_timing_segment(text, "main") or _clip_timing_segment_body(text)
        reward = 0.0
        if segment:
            ctx = EffectContext(sim=self, owner=pl, owner_idx=owner_idx, card=cd)
            _apply_rules_to_text(ctx, segment)
            reward = 0.05   # récompense légèrement supérieure à jouer un perso
        pl.trash.append(cid)
        return reward

    def return_don(self, pl: PlayerState, n: int) -> int:
        """
        CR 8-3-1-6 / CR 10-2-10 : DON!! −N.
        Renvoie N DON!! vers le deck DON!! en les prélevant dans cet ordre :
          1. DON!! actifs (zone de coût)
          2. DON!! reposés (zone de coût)
          3. DON!! attachés au Leader
          4. DON!! attachés aux Characters (du plus faible DON!! au plus grand)
        Retourne le nombre de DON!! effectivement renvoyés (peut être < N si manquants).
        """
        returned = 0
        remaining = n
        # 1. DON!! actifs
        take = min(remaining, pl.don_active)
        pl.don_active -= take; pl.don_deck += take; returned += take; remaining -= take
        # 2. DON!! reposés
        if remaining:
            take = min(remaining, pl.don_rested)
            pl.don_rested -= take; pl.don_deck += take; returned += take; remaining -= take
        # 3. DON!! sur le Leader
        if remaining and pl.leader_attached_don:
            take = min(remaining, pl.leader_attached_don)
            pl.leader_attached_don -= take; pl.don_deck += take; returned += take; remaining -= take
        # 4. DON!! sur les Characters (du plus petit attachement)
        if remaining:
            for bc in sorted(pl.board, key=lambda b: b.attached_don):
                if not remaining:
                    break
                take = min(remaining, bc.attached_don)
                bc.attached_don -= take; pl.don_deck += take; returned += take; remaining -= take
        return returned

    def _attach_don(self, pl: PlayerState, slot: int, n: int = 1) -> int:
        """
        Attache jusqu'à ``n`` DON!! actifs au slot (CR 6-5-5) :
          slot 0 = Leader, slot 1–5 = board[slot-1].
        Retourne le nombre de DON!! effectivement attachés (0 si impossible).
        """
        if n < 1 or pl.don_active <= 0:
            return 0
        take = min(n, pl.don_active)
        if slot == 0:
            pl.don_active -= take
            pl.leader_attached_don += take
            return take
        idx = slot - 1
        if idx < 0 or idx >= len(pl.board):
            return 0
        pl.don_active -= take
        pl.board[idx].attached_don += take
        return take

    def _play_stage(
        self,
        pl: PlayerState,
        owner_idx: int,
        cid: str,
        cd: "CardDef",
    ) -> None:
        """
        Pose une carte Stage en zone Stage (CR 3-8).
        Si une Stage est déjà présente → elle va au trash avant (CR 3-8-5-1).
        Déclenche [On Play] (CR 10-2-6).
        """
        if pl.stage_area:
            pl.trash.append(pl.stage_area.pop())
        pl.stage_area.append(cid)
        dispatch_on_play(EffectContext(sim=self, owner=pl, owner_idx=owner_idx, card=cd))

    def _activate_main_effect(self, pl: PlayerState, owner_idx: int, slot_idx: int) -> float:
        """
        Résout l'effet [Activate: Main] du perso en board[slot_idx] (CR 10-2-2).
        Repose le perso, marque l'activation, résout l'effet.
        Retourne une récompense.
        """
        bad = self._reward_activate_main_char_bad
        if slot_idx < 0 or slot_idx >= len(pl.board):
            return bad
        bc = pl.board[slot_idx]
        if bc.rested or bc.restrictions.blocks_attack_declaration() or bc.activate_main_used:
            return bad
        cd = self.cards.get(bc.card_id)
        if not cd:
            return bad
        blob = (cd.card_text or "").lower()
        if "[activate: main]" not in blob:
            return bad
        bc.rested = True
        bc.activate_main_used = True
        from .engine.keyword_handlers import EffectContext
        from .engine.on_play_resolver import apply_activate_main_effects
        ctx = EffectContext(sim=self, owner=pl, owner_idx=owner_idx, card=cd)
        apply_activate_main_effects(ctx)
        return self._reward_activate_main_char_ok

    def _activate_main_leader_effect(self, pl: PlayerState, owner_idx: int) -> float:
        """
        Résout l'effet [Activate: Main] du Leader (CR 10-2-2).

        Différences vs perso :
        - Le Leader ne se repose PAS
        - Le coût DON!! -N est payé via retour au deck DON!!
        - [Once Per Turn] suivi via leader_activate_main_used

        Ex. Kaido ST04-001 : [Activate: Main] [Once Per Turn] DON!! -7 :
            Trash up to 1 of your opponent's Life cards.
        """
        bad = self._reward_activate_main_leader_bad
        if pl.leader_activate_main_used:
            return bad
        leader_cd = self.cards.get(pl.leader_id or "")
        if not leader_cd:
            return bad
        text = leader_cd.card_text or ""
        blob = text.lower()
        if "[activate: main]" not in blob:
            return bad

        # Enel OP15-058 : « If it is your second turn or later, … » (pas d’effet avant le 2e tour)
        if "second turn or later" in blob and self.player_turn_number(pl) < 2:
            return bad

        # Lire le coût DON!! -N
        don_cost = 0
        m_cost = re.search(r"don!!\s*-\s*(\d+)", blob)
        if m_cost:
            don_cost = int(m_cost.group(1))

        # Calculer le DON!! total sur le terrain (actif + attaché au leader + board)
        don_total_field = pl.don_active + pl.leader_attached_don
        for bc in pl.board:
            don_total_field += bc.attached_don
        if don_total_field < don_cost:
            return bad  # pas assez de DON!! sur le terrain

        # Payer le coût DON!! -N : retirer en priorité don_active, puis attaché
        remaining_cost = don_cost
        take_active = min(remaining_cost, pl.don_active)
        pl.don_active -= take_active
        pl.don_deck   += take_active
        remaining_cost -= take_active
        if remaining_cost and pl.leader_attached_don:
            take = min(remaining_cost, pl.leader_attached_don)
            pl.leader_attached_don -= take
            pl.don_deck += take
            remaining_cost -= take
        if remaining_cost:
            for bc in pl.board:
                if remaining_cost <= 0:
                    break
                take = min(remaining_cost, bc.attached_don)
                bc.attached_don -= take
                pl.don_deck += take
                remaining_cost -= take

        pl.leader_activate_main_used = True

        from .engine.keyword_handlers import EffectContext
        from .engine.on_play_resolver import apply_activate_main_effects
        ctx = EffectContext(sim=self, owner=pl, owner_idx=owner_idx, card=leader_cd)
        apply_activate_main_effects(ctx)
        return self._reward_activate_main_leader_ok

    def _begin_external_p1_turn(self) -> None:
        """Après la phase combat de P0 : démarre le tour de P1 pour self-play (politique externe)."""
        self._opp_attacks_done = 0
        self._policy_controls_p1 = True
        self._start_turn(self.p1)
        if not self.done:
            self.phase = Phase.MAIN

    def _step_main_for(self, pl: PlayerState, owner_idx: int, action: int) -> float:
        """
        MAIN phase pour le joueur ``pl`` (P0 ou P1) — même encodage d'actions que pour P0.
        """
        if self.phase != Phase.MAIN:
            return -0.05

        self._turn_steps += 1
        self._last_action = action

        if 0 <= action <= 6:
            if action >= len(pl.hand):
                return -0.02
            cid = pl.hand[action]
            cd = self._card(cid)
            ctx = PlayabilityContext(Phase.MAIN, pl.don, len(pl.hand))
            if not can_play_card(cd, ctx):
                return -0.05
            if not self._pay_don(pl, cd.cost):
                return -0.05
            pl.hand.pop(action)
            self._cards_played_turn += 1
            self._total_cards_played_from_hand += 1
            self._main_nonidle_actions += 1
            card_type = (cd.card_type or "").lower()
            if "event" in card_type:
                return self._play_event(pl, owner_idx, cid, cd)
            if "stage" in card_type:
                self._play_stage(pl, owner_idx, cid, cd)
                return 0.02
            self._play_char_to_board(pl, owner_idx, cid, cd)
            return 0.02

        if action == MAIN_END_ACTION:
            if self.turn_has_meaningful_activity() or not self.any_legal_main_commit_for_player(pl):
                pl.last_main_finished_idle = self._main_nonidle_actions == 0
                self.phase = Phase.BATTLE
                return 0.0
            return -0.08

        dec = decode_attach_don_action(action)
        if dec is not None:
            slot, k = dec
            attached = self._attach_don(pl, slot, k)
            if attached:
                self._main_nonidle_actions += 1
                self._don_attached_turn += attached
                return 0.01
            return -0.02

        if MAIN_ACTIVATE_MAIN_BASE <= action < MAIN_ACTIVATE_MAIN_BASE + MAIN_ACTIVATE_MAIN_SLOTS:
            slot_idx = action - MAIN_ACTIVATE_MAIN_BASE
            rr = self._activate_main_effect(pl, owner_idx, slot_idx)
            if rr >= 0.0:
                self._main_nonidle_actions += 1
            return rr

        if action == MAIN_ACTIVATE_MAIN_LEADER:
            rr = self._activate_main_leader_effect(pl, owner_idx)
            if rr >= 0.0:
                self._main_nonidle_actions += 1
            return rr

        return -0.02

    def step_main(self, action: int) -> float:
        """
        MAIN phase :
          0–6    = jouer carte main[slot]
          7      = fin de tour → phase de combat
          45–104 = attacher 1..10 DON!! (slot = (a-45)//10, quantité = (a-45)%10+1)
          105–109 = activer [Activate: Main] sur board[0-4]
          110    = activer [Activate: Main] du Leader
        """
        if self.phase != Phase.MAIN:
            return -0.05
        if self._policy_controls_p1:
            return self._step_main_for(self.p1, 1, action)
        return self._step_main_for(self.p0, 0, action)

    def _step_battle_p1(self, battle_action: int) -> float:
        """Phase combat du tour P1 (self-play) : attaquant = P1, défenseur = P0."""
        self._turn_steps += 1
        self._last_action = battle_action

        if battle_action == BATTLE_PASS_ACTION:
            atk, defe = self.p1, self.p0
            if self.turn_has_meaningful_activity() or not self.any_legal_battle_commit_for_pair(
                atk, defe
            ):
                self._end_turn(self.p1)
                self._policy_controls_p1 = False
                if not self.done:
                    self._start_turn(self.p0)
                    self.phase = Phase.MAIN
                return 0.0
            return -0.08

        if BATTLE_ATTACK_BASE <= battle_action < BATTLE_ATTACK_BASE + N_BATTLE_ATTACK_CODES:
            code = battle_action - BATTLE_ATTACK_BASE
            atk_slot = code // N_TARGETS
            tgt_slot = code % N_TARGETS
            # Perspective récompense P0 : P1 attaque → même branche Enel que le tour adverse simulé
            r = self._resolve_attack(
                self.p1, self.p0, atk_slot, tgt_slot, agent_is_attacker=False,
            )
            return r

        dec = decode_attach_don_action(battle_action)
        if dec is not None:
            slot, k = dec
            attached = self._attach_don(self.p1, slot, k)
            if attached:
                self._don_attached_turn += attached
                return 0.01
            return -0.02

        return -0.02

    def step_battle(self, battle_action: int) -> float:
        """
        BATTLE phase :
          8      = fin de la phase de combat → tour adverse puis notre tour
          9–44   = attaque (code = action-9 ; attaquant=code//6 ; cible=code%6)
          45–104 = attacher 1..10 DON!! avant déclaration (CR 6-5-5)
        """
        if self.phase != Phase.BATTLE:
            return -0.05

        self._last_agent_attack_slot = -1

        if self._policy_controls_p1:
            return self._step_battle_p1(battle_action)

        self._turn_steps += 1
        self._last_action = battle_action

        if battle_action == BATTLE_PASS_ACTION:
            atk, defe = self.p0, self.p1
            if self.turn_has_meaningful_activity() or not self.any_legal_battle_commit_for_pair(
                atk, defe
            ):
                self._end_turn(self.p0)
                if not self.done:
                    if self._self_play_external:
                        self._begin_external_p1_turn()
                    else:
                        self._simulate_opponent_turn()
                extra = self._opponent_battle_extra_reward
                self._opponent_battle_extra_reward = 0.0
                return float(extra)
            return -0.08

        if BATTLE_ATTACK_BASE <= battle_action < BATTLE_ATTACK_BASE + N_BATTLE_ATTACK_CODES:
            code = battle_action - BATTLE_ATTACK_BASE
            atk_slot = code // N_TARGETS
            tgt_slot = code % N_TARGETS
            r = self._resolve_attack(self.p0, self.p1, atk_slot, tgt_slot, agent_is_attacker=True)
            return r

        dec = decode_attach_don_action(battle_action)
        if dec is not None:
            slot, k = dec
            attached = self._attach_don(self.p0, slot, k)
            if attached:
                self._don_attached_turn += attached
                return 0.01
            return -0.02

        return -0.02

    def suggest_first_battle_action(self) -> int | None:
        """Première attaque légale parmi les codes disponibles (fallback env)."""
        if self._policy_controls_p1:
            atk, deff = self.p1, self.p0
        else:
            atk, deff = self.p0, self.p1
        for code in range(N_BATTLE_ATTACK_CODES):
            atk_slot = code // N_TARGETS
            tgt_slot = code % N_TARGETS
            if self.attack_is_legal(atk, deff, atk_slot, tgt_slot):
                return BATTLE_ATTACK_BASE + code
        return None

    def legal_battle_actions(self) -> list[int]:
        """Liste complète des actions légales en phase de combat (pour l'agent)."""
        if self._policy_controls_p1:
            atk, deff = self.p1, self.p0
        else:
            atk, deff = self.p0, self.p1
        out: list[int] = []
        if self.turn_has_meaningful_activity() or not self.any_legal_battle_commit_for_pair(atk, deff):
            out.append(BATTLE_PASS_ACTION)
        for code in range(N_BATTLE_ATTACK_CODES):
            atk_slot = code // N_TARGETS
            tgt_slot = code % N_TARGETS
            if self.attack_is_legal(atk, deff, atk_slot, tgt_slot):
                out.append(BATTLE_ATTACK_BASE + code)
        return out

    @property
    def winner(self) -> int | None:
        return self._winner
