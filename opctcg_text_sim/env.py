from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .card_db import CardDef, ensure_cards_for_deck, load_card_csv
from .deck_parser import (
    deck_to_multiset,
    entry_list_cid_order,
    extract_leader_from_multiset,
    multiset_to_deck_list,
    parse_deck_file,
    read_deck_leader_directive,
)
from .playability import Phase, PlayabilityContext, can_play_card, can_play_character
from .rules_simple import combined_rules_scalar, load_rules_corpus, log_session_preamble
from .engine.combat_rules import effective_power
from .engine.restriction_state import character_may_declare_attack
from .simulator import (
    ACTION_SPACE_SIZE,
    BATTLE_ATTACK_BASE,
    BATTLE_PASS_ACTION,
    BLOCKER_N_SLOTS,
    BLOCKER_PASS,
    BLOCKER_SLOT_BASE,
    MAIN_ATTACH_DON_ACTIONS,
    MAIN_ATTACH_DON_BASE,
    MAIN_ATTACH_DON_MAX,
    MAIN_ACTIVATE_MAIN_BASE,
    MAIN_ACTIVATE_MAIN_LEADER,
    MAIN_ACTIVATE_MAIN_SLOTS,
    MAIN_END_ACTION,
    MULLIGAN_KEEP,
    MULLIGAN_TAKE,
    N_BATTLE_ATTACK_CODES,
    N_TARGETS,
    SimplifiedOPSim,
    decode_attach_don_action,
)


def _coerce_max_hand(raw: object) -> int:
    """
    Taille max de la main côté simulateur : 0 ou absent = illimité (règles CR par défaut).
    Un entier > 0 impose un plafond (optionnel, ex. stress-test ou shaping).
    """
    if raw is None:
        return 0
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(0, n)


def _is_productive_main_action(action: int) -> bool:
    """MAIN : jouer, attacher ou [Activate: Main] — pas seulement fin de tour (7)."""
    a = int(action)
    if 0 <= a <= 6:
        return True
    if MAIN_ATTACH_DON_BASE <= a < MAIN_ATTACH_DON_BASE + MAIN_ATTACH_DON_ACTIONS:
        return True
    if MAIN_ACTIVATE_MAIN_BASE <= a <= MAIN_ACTIVATE_MAIN_LEADER:
        return True
    return False


@dataclass(frozen=True)
class _EventShapingSnapshot:
    """État « moi vs adversaire » pour le shaping par événements (perspective agent)."""

    opp_life: int
    opp_board: int
    me_deck: int
    me_hand: int
    me_board: int


def _fill_hand_segment(
    dest: np.ndarray,
    hand: list[str],
    emb_vecs: dict[str, np.ndarray],
    slots: int,
    emb: int,
) -> None:
    """Écrit l'encodage main dans ``dest[: slots * emb]`` (``dest`` déjà mis à zéro au préalable)."""
    for i in range(min(slots, len(hand))):
        cid = hand[i]
        e = emb_vecs.get(cid)
        if e is not None:
            dest[i * emb : (i + 1) * emb] = e


class OPTextSimEnv(gym.Env):
    """
    Environnement Gymnasium wrappant SimplifiedOPSim.

    Action space : Discrete(ACTION_SPACE_SIZE) — attache-DON : 60 codes (6 slots × 1..10)
      MAIN  : 0–6   jouer carte main[slot]   7 = fin de tour → phase combat
              45–104 = attacher 1..10 DON!! (slot=(a-45)//10, N=(a-45)%10+1)
              105–109 = [Activate: Main] char[0–4] ; 110 = [Activate: Main] Leader
      BATTLE: 8     fin phase de combat       9–44 = attaque (code = action-9)
                attack_code = action - 9
                attaquant   = code // 6  (0=Leader, 1=char[0], …, 5=char[4])
                cible       = code  % 6  (0=Leader, 1=char[0], …, 5=char[4])
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        deck0_path: Path,
        deck1_path: Path,
        cards_csv: Path,
        column_map: dict[str, str],
        rules_corpus_path: Path | None,
        sim_cfg: dict[str, Any],
        obs_dim: int = 96,
        seed: int = 0,
        action_log_path: Path | None = None,
        animation_log_path: Path | None = None,
        shaping_progress_ref: Any | None = None,
    ):
        super().__init__()
        self._action_log_path = action_log_path
        self._animation_log_path = animation_log_path
        self._log_fp = None
        self._anim_fp = None
        self._anim_frame_idx = 0
        self._step_idx = 0
        self._obs_dim = obs_dim
        self._emb = 8
        self._hand_slots = 7

        self._deck0_path = Path(deck0_path)
        self._deck1_path = Path(deck1_path)

        e0 = parse_deck_file(deck0_path)
        e1 = parse_deck_file(deck1_path)
        m0, m1 = deck_to_multiset(e0), deck_to_multiset(e1)
        ids = set(m0) | set(m1)
        db = load_card_csv(cards_csv, column_map)
        self._cards = ensure_cards_for_deck(ids, db)
        m0, leader0 = extract_leader_from_multiset(
            m0, self._cards,
            explicit_leader_id=read_deck_leader_directive(deck0_path),
            preferred_cid_order=entry_list_cid_order(e0),
        )
        m1, leader1 = extract_leader_from_multiset(
            m1, self._cards,
            explicit_leader_id=read_deck_leader_directive(deck1_path),
            preferred_cid_order=entry_list_cid_order(e1),
        )
        self._deck0 = multiset_to_deck_list(m0)
        self._deck1 = multiset_to_deck_list(m1)

        self._rules_corpus = load_rules_corpus(rules_corpus_path) if rules_corpus_path else ""
        self._project_root = Path(__file__).resolve().parents[1]
        self._rules_scalar = combined_rules_scalar(self._project_root, self._rules_corpus)

        snap_raw = sim_cfg.get("card_effects_snapshot")
        effect_snap: str | None = None
        if snap_raw:
            from pathlib import Path as _Path

            cand = _Path(str(snap_raw))
            snap_path = cand if cand.is_file() else (self._project_root / cand)
            if snap_path.is_file():
                effect_snap = str(snap_path.resolve())

        self._win_reward = float(sim_cfg.get("win_reward", 10.0))
        self._loss_reward = float(sim_cfg.get("loss_reward", -10.0))
        raw_max_ep = sim_cfg.get("max_episode_steps")
        self._max_episode_steps = int(raw_max_ep) if raw_max_ep is not None else 0
        if self._max_episode_steps < 0:
            self._max_episode_steps = 0
        self._timeout_both_penalty = float(sim_cfg.get("timeout_both_penalty", 0.0))
        raw_max_turns = sim_cfg.get("max_turns_started")
        self._max_turns_started = int(raw_max_turns) if raw_max_turns is not None else 0
        if self._max_turns_started < 0:
            self._max_turns_started = 0
        self._turn_cap_penalty = float(sim_cfg.get("turn_cap_penalty", 0.0))
        raw_stall_ts = sim_cfg.get("stall_no_card_after_turns_started")
        self._stall_no_card_after_turns_started = (
            int(raw_stall_ts) if raw_stall_ts is not None else 0
        )
        if self._stall_no_card_after_turns_started < 0:
            self._stall_no_card_after_turns_started = 0
        self._stall_no_card_penalty = float(sim_cfg.get("stall_no_card_penalty", 0.0))
        self._activity_after_idle_reward = float(sim_cfg.get("activity_after_idle_reward", 0.0))

        es = sim_cfg.get("event_shaping") or {}
        self._event_shaping_enabled = bool(es.get("enabled", False))
        self._es_w_opp_life = float(es.get("w_opp_life", 0.28))
        self._es_w_opp_ko = float(es.get("w_opp_ko", 0.09))
        self._es_w_draw = float(es.get("w_draw", 0.035))
        self._es_max_per_step = float(es.get("max_per_step", 0.65))
        self._es_max_opp_life_hits = int(es.get("max_opp_life_hits_per_step", 3))
        self._es_life_m0 = float(es.get("life_mult_start", 0.65))
        self._es_life_m1 = float(es.get("life_mult_end", 1.25))
        self._es_ko_m0 = float(es.get("ko_mult_start", 0.55))
        self._es_ko_m1 = float(es.get("ko_mult_end", 1.12))
        self._es_draw_m0 = float(es.get("draw_mult_start", 1.05))
        self._es_draw_m1 = float(es.get("draw_mult_end", 0.28))
        self._es_w_play_character = float(es.get("w_play_character", 0.07))
        self._es_w_char_attack = float(es.get("w_char_attack", 0.14))
        self._es_pc_m0 = float(es.get("play_char_mult_start", 1.0))
        self._es_pc_m1 = float(es.get("play_char_mult_end", 1.12))
        self._es_ca_m0 = float(es.get("char_attack_mult_start", 1.0))
        self._es_ca_m1 = float(es.get("char_attack_mult_end", 1.22))
        self._es_max_play_char_step = max(0, int(es.get("max_play_character_per_step", 2)))
        self._es_episode_fallback_horizon = max(
            1, int(es.get("episode_fallback_horizon", 420))
        )
        self._shaping_progress_ref = shaping_progress_ref
        # Si False : pas de chaînes d’action / snapshot par step (gain perf massif hors debug).
        self._step_trace = bool(sim_cfg.get("step_trace", False))

        self._sim = SimplifiedOPSim(
            self._cards,
            self._deck0,
            self._deck1,
            starting_hand=int(sim_cfg.get("starting_hand", 5)),
            starting_life=int(sim_cfg.get("starting_life", 5)),
            don_per_turn=int(sim_cfg.get("don_per_turn", 2)),
            official_first_player_don=int(sim_cfg.get("official_first_player_don", 1)),
            max_don=int(sim_cfg.get("max_don", 10)),
            draw_per_turn=int(sim_cfg.get("draw_per_turn", 1)),
            skip_first_draw_first_player=bool(sim_cfg.get("skip_first_draw_first_player", True)),
            max_board=int(sim_cfg.get("max_board", 3)),
            max_hand=_coerce_max_hand(sim_cfg.get("max_hand")),
            leader_block=int(sim_cfg.get("leader_power", 5000)),
            leader_defense=sim_cfg.get("leader_defense"),
            leader0_id=leader0,
            leader1_id=leader1,
            effect_snapshot_path=effect_snap,
            effect_cache_dedupe=bool(sim_cfg.get("effect_cache_dedupe", True)),
            rng=np.random.default_rng(seed),
            opponent_action_policy=str(sim_cfg.get("opponent_action_policy", "random")),
            self_play=bool(sim_cfg.get("self_play", False)),
            stall_no_card_after_turns_started=self._stall_no_card_after_turns_started,
            reward_activate_main_leader=float(sim_cfg.get("reward_activate_main_leader", 0.05)),
            reward_activate_main_leader_fail=float(
                sim_cfg.get("reward_activate_main_leader_fail", -0.05)
            ),
            reward_activate_main_character=float(
                sim_cfg.get("reward_activate_main_character", 0.03)
            ),
            reward_activate_main_character_fail=float(
                sim_cfg.get("reward_activate_main_character_fail", -0.05)
            ),
            reward_leader_opponent_turn_life_zero=float(
                sim_cfg.get("reward_leader_opponent_turn_life_zero", 0.45)
            ),
            shuffle_decks=bool(sim_cfg.get("shuffle_decks", True)),
        )

        # Un vecteur d'embedding figé par carte (hot path : plus d'alloc à chaque obs)
        self._hand_emb_vecs: dict[str, np.ndarray] = {
            cid: cd.embedding(self._emb) for cid, cd in self._cards.items()
        }
        # Buffers réutilisés à chaque step / masque (évite np.zeros + GC)
        self._obs_vec = np.zeros(obs_dim, dtype=np.float32)
        self._legal_mask_buf = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._rng = np.random.default_rng(seed)

    def _persp_me_opp(self):
        """(joueur « moi » pour obs/masque, adversaire) — P0 sauf self-play vue P1."""
        s = self._sim
        if s.perspective_p1():
            return s.p1, s.p0
        return s.p0, s.p1

    def _activity_after_idle_shaping_reward(
        self, phase_before: Phase, action: int, main_step_reward: float
    ) -> float:
        """
        Après un tour Main où le joueur n'a rien fait de la phase Main (hors draw auto),
        active ``boost_next_main_activity``. Dès son 2e tour au moins, la première action
        productive réussie dans une Main donne une bonification (encourage à ne pas enchaîner
        des tours « rien que piocher puis passer »).
        """
        w = self._activity_after_idle_reward
        if w == 0.0:
            return 0.0
        if phase_before != Phase.MAIN:
            return 0.0
        if not _is_productive_main_action(action):
            return 0.0
        # Échec de jeu / attache illégale : pas de bonus, conserve le boost pour un autre essai
        if float(main_step_reward) < -0.01:
            return 0.0
        s = self._sim
        me, _ = self._persp_me_opp()
        if not getattr(me, "boost_next_main_activity", False):
            return 0.0
        if s.player_turn_number(me) < 2:
            return 0.0
        me.boost_next_main_activity = False
        return w

    def _event_shaping_progress(self) -> float:
        """Progression 0→1 : ref partagée (PPO) ou repli sur la longueur d’épisode."""
        ref = self._shaping_progress_ref
        if ref is not None:
            with ref.get_lock():
                p = float(ref.value)
            return max(0.0, min(1.0, p))
        h = float(self._es_episode_fallback_horizon)
        return min(1.0, float(self._episode_step_idx) / max(1.0, h))

    def _infer_draws_from_deck(self, before: _EventShapingSnapshot, me: Any) -> int:
        """Heuristique : cartes piochées depuis le deck ce pas (évite le spam via plafond step)."""
        deck_lost = before.me_deck - len(me.deck)
        if deck_lost <= 0:
            return 0
        net_hand = len(me.hand) - before.me_hand
        played = max(0, before.me_hand - len(me.hand))
        inferred = min(deck_lost, max(0, net_hand) + played)
        return max(0, min(inferred, 5))

    def _compute_event_shaping(self, before: _EventShapingSnapshot | None) -> float:
        if not self._event_shaping_enabled or before is None:
            return 0.0
        me, op = self._persp_me_opp()
        p = self._event_shaping_progress()

        def _curve(a0: float, a1: float) -> float:
            return a0 + p * (a1 - a0)

        w_life = self._es_w_opp_life * _curve(self._es_life_m0, self._es_life_m1)
        w_ko = self._es_w_opp_ko * _curve(self._es_ko_m0, self._es_ko_m1)
        w_draw = self._es_w_draw * _curve(self._es_draw_m0, self._es_draw_m1)
        w_pc = self._es_w_play_character * _curve(self._es_pc_m0, self._es_pc_m1)
        w_ca = self._es_w_char_attack * _curve(self._es_ca_m0, self._es_ca_m1)

        dl = max(0, before.opp_life - op.life)
        cap_hits = self._es_max_opp_life_hits
        if cap_hits > 0:
            dl = min(dl, cap_hits)
        r_life = w_life * float(dl)

        dko = max(0, before.opp_board - len(op.board))
        r_ko = w_ko * float(dko)

        draws = self._infer_draws_from_deck(before, me)
        r_draw = w_draw * float(draws)

        added_chars = max(0, len(me.board) - before.me_board)
        cap_pc = self._es_max_play_char_step
        if cap_pc > 0:
            added_chars = min(added_chars, cap_pc)
        r_pc = w_pc * float(added_chars)

        atk_slot = int(getattr(self._sim, "_last_agent_attack_slot", -1))
        r_ca = w_ca * (1.0 if atk_slot >= 1 else 0.0)

        total = r_life + r_ko + r_draw + r_pc + r_ca
        cap = self._es_max_per_step
        if cap > 0.0 and total > cap:
            return float(cap)
        return float(total)

    # ── logs ──────────────────────────────────────────────────────────────────

    def _close_log(self) -> None:
        for attr in ("_log_fp", "_anim_fp"):
            fp = getattr(self, attr, None)
            if fp is not None:
                try:
                    fp.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def _log_line(self, text: str) -> None:
        if self._log_fp is not None:
            self._log_fp.write(text.rstrip() + "\n")
            self._log_fp.flush()

    # ── snapshot / animation ──────────────────────────────────────────────────

    def _snapshot_dict(self) -> dict[str, Any]:
        s = self._sim

        def _player(pl_idx: int) -> dict[str, Any]:
            pl = s.p0 if pl_idx == 0 else s.p1
            return {
                "leader_id": pl.leader_id,
                "leader_power": pl.leader_power,
                "leader_power_effective": pl.effective_leader_power(),
                "leader_power_mod_bonus": pl.leader_power_modifiers.bonus_turn,
                "leader_power_mod_penalty": pl.leader_power_modifiers.penalty_turn,
                "leader_rested": pl.leader_rested,
                "leader_attached_don": pl.leader_attached_don,
                "life": pl.life,
                "life_cards": list(pl.life_cards),
                "don": pl.don_active,
                "don_active": pl.don_active,
                "don_rested": pl.don_rested,
                "don_deck": pl.don_deck,
                "official_don_deck": pl.don_deck_initial,
                "trash_count": len(pl.trash),
                "trash": list(pl.trash[-32:]),
                "hand": list(pl.hand),
                "hand_count": len(pl.hand),
                "deck_remaining": len(pl.deck),
                "board": [
                    {
                        "id": b.card_id,
                        "power": b.power,
                        "power_effective": effective_power(b, self._cards),
                        "power_mod_bonus": b.power_modifiers.bonus_turn,
                        "power_mod_penalty": b.power_modifiers.penalty_turn,
                        "rested": b.rested,
                        "has_rush": b.has_rush,
                        "has_rush_char": b.has_rush_char,
                        "has_blocker": b.has_blocker,
                        "just_played": b.just_played,
                        "attached_don": b.attached_don,
                        "cannot_attack": b.restrictions.blocks_attack_declaration(),
                        "cannot_be_ko": b.restrictions.blocks_ko(),
                        "cannot_be_targeted": b.restrictions.blocks_opponent_targeting(),
                    }
                    for b in pl.board
                ],
                "stage_area": list(pl.stage_area),
            }

        ph_str = {
            Phase.MULLIGAN: "MULLIGAN",
            Phase.MAIN: "MAIN",
            Phase.BATTLE: "BATTLE",
            Phase.BLOCKER: "BLOCKER",
        }.get(s.phase, "MAIN")
        return {
            "phase": ph_str,
            "p0": _player(0),
            "p1": _player(1),
            "done": s.done,
            "winner": s.winner,
            "sim_meta": {
                "rules_version": "OP-TCG CR v2.0 — simulateur",
                "rules_doc": "data/OPTCG_GameRules_Reference.md",
                "don_per_turn": s.don_per_turn,
                "official_first_player_don": s.official_first_player_don,
                "max_don": s.max_don,
                "max_hand": s.max_hand,
                "max_board": s.max_board,
                "official_deck": 50,
                "official_don_deck": 10,
                "turns_started": s.turns_started,
                "battle_damage_allowed": s.battle_damage_allowed(),
                "action_space_size": ACTION_SPACE_SIZE,
                "turn_hint": (
                    "BATTLE — attaques : 9-44 (code = action-9 ; att=code//6 [0=Leader,1-5=perso] ; cible=code%6) ; "
                    "8 = fin de combat ; 45-50 = DON!! attach. L'attaquant se repose dès la déclaration."
                    if ph_str == "BATTLE"
                    else f"{ph_str} — jouer carte : 0-6 (slot main) ; 7 = fin de tour ; 45-50 = DON!! ; 51-55 = [Activate: Main]."
                ),
            },
        }

    def _training_snapshot_compact(self) -> dict[str, Any]:
        """Résumé JSON léger de l'état courant (fin de partie — monitoring web / PPO)."""
        snap = self._snapshot_dict()

        def _summ(p: dict[str, Any]) -> dict[str, Any]:
            brd = p.get("board") or []
            return {
                "leader_id": p.get("leader_id"),
                "life": p.get("life"),
                "hand_n": p.get("hand_count"),
                "board_n": len(brd),
                "don_active": p.get("don_active"),
            }

        meta = snap.get("sim_meta") or {}
        return {
            "phase": snap.get("phase"),
            "winner": snap.get("winner"),
            "turns_started": meta.get("turns_started"),
            "p0": _summ(snap["p0"]),
            "p1": _summ(snap["p1"]),
        }

    def _write_anim_frame(self, kind: str, **extra: Any) -> None:
        if self._anim_fp is None:
            return
        payload: dict[str, Any] = {
            "frame": self._anim_frame_idx,
            "kind": kind,
            **self._snapshot_dict(),
            **extra,
        }
        self._anim_fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._anim_fp.flush()
        self._anim_frame_idx += 1

    # ── description d'action lisible ──────────────────────────────────────────

    def _describe_action(self, action: int) -> str:
        s = self._sim
        if s.phase == Phase.MAIN:
            if 0 <= action <= 6:
                if action < len(s.p0.hand):
                    cid = s.p0.hand[action]
                    cd = self._cards[cid]
                    return (
                        f"MAIN_PLAY slot={action} {cid} «{cd.name}» "
                        f"coût={cd.cost} puiss={cd.power} type={cd.card_type}"
                    )
                return f"MAIN_PLAY slot={action} (invalide, main={len(s.p0.hand)} cartes)"
            if action == MAIN_END_ACTION:
                return "MAIN_END_TURN → phase de combat"
            dec = decode_attach_don_action(action)
            if dec is not None:
                slot, k = dec
                target = "Leader" if slot == 0 else f"board[{slot - 1}]"
                don = s.p0.don_active
                return (
                    f"MAIN_ATTACH_DON x{k} → {target} "
                    f"(don_active={don}→{max(0, don - min(k, don))})"
                )
            if MAIN_ACTIVATE_MAIN_BASE <= action < MAIN_ACTIVATE_MAIN_BASE + MAIN_ACTIVATE_MAIN_SLOTS:
                idx = action - MAIN_ACTIVATE_MAIN_BASE
                if idx < len(s.p0.board):
                    b = s.p0.board[idx]
                    return f"MAIN_ACTIVATE_MAIN board[{idx}] {b.card_id}"
                return f"MAIN_ACTIVATE_MAIN board[{idx}] (invalide)"
            return f"MAIN action={action} (hors plage, traité comme fin de tour)"

        # Phase BATTLE
        if action == BATTLE_PASS_ACTION:
            return "BATTLE_END_PHASE → fin de tour, tour adverse, nouveau tour"
        if BATTLE_ATTACK_BASE <= action < BATTLE_ATTACK_BASE + N_BATTLE_ATTACK_CODES:
            code = action - BATTLE_ATTACK_BASE
            atk_slot = code // N_TARGETS
            tgt_slot = code % N_TARGETS
            atk_s = "Leader" if atk_slot == 0 else f"perso[{atk_slot - 1}]"
            tgt_s = "Leader" if tgt_slot == 0 else f"perso[{tgt_slot - 1}]"
            if atk_slot == 0:
                ap = s.p0.effective_leader_power()
            elif atk_slot - 1 < len(s.p0.board):
                b = s.p0.board[atk_slot - 1]
                ap = effective_power(b, self._cards)
            else:
                return f"BATTLE_ATTACK code={code} (attaquant slot={atk_slot} invalide)"
            legal = s.attack_is_legal(s.p0, s.p1, atk_slot, tgt_slot)
            return (
                f"BATTLE_ATTACK {atk_s} ({ap}) → {tgt_s} "
                f"{'[LEGAL]' if legal else '[ILLÉGAL]'}"
            )
        dec = decode_attach_don_action(action)
        if dec is not None:
            slot, k = dec
            target = "Leader" if slot == 0 else f"board[{slot - 1}]"
            don = s.p0.don_active
            return (
                f"BATTLE_ATTACH_DON x{k} → {target} "
                f"(don_active={don}→{max(0, don - min(k, don))})"
            )
        return f"BATTLE action={action} (hors plage, traité comme fin de phase)"

    def _state_snapshot(self) -> str:
        s = self._sim
        def _fmt_board(board):
            parts = []
            for b in board:
                don_s = f"+{b.attached_don}D" if b.attached_don else ""
                mod = b.power_modifiers.net_bonus()
                mod_s = f"({mod:+d})" if mod else ""
                pe = effective_power(b, self._cards)
                parts.append(f"{b.card_id}:{pe}{don_s}{mod_s}{'R' if b.rested else 'A'}")
            return "/".join(parts)

        b0 = _fmt_board(s.p0.board)
        b1 = _fmt_board(s.p1.board)
        ldr0_don = f"+{s.p0.leader_attached_don}D" if s.p0.leader_attached_don else ""
        ldr1_don = f"+{s.p1.leader_attached_don}D" if s.p1.leader_attached_don else ""
        ph = {
            Phase.MULLIGAN: "MULLIGAN",
            Phase.MAIN: "MAIN",
            Phase.BATTLE: "BATTLE",
            Phase.BLOCKER: "BLOCKER",
        }.get(s.phase, "MAIN")
        return (
            f"phase={ph} "
            f"p0 life={s.p0.life} don={s.p0.don_active}a+{s.p0.don_rested}r "
            f"hand={len(s.p0.hand)} deck={len(s.p0.deck)} board=[{b0}] "
            f"leader={'R' if s.p0.leader_rested else 'A'}{ldr0_don} "
            f"| p1 life={s.p1.life} don={s.p1.don_active}a+{s.p1.don_rested}r "
            f"hand={len(s.p1.hand)} deck={len(s.p1.deck)} board=[{b1}] "
            f"leader={'R' if s.p1.leader_rested else 'A'}{ldr1_don} "
            f"done={s.done}"
        )

    # ── masque d'actions légales ──────────────────────────────────────────────

    def legal_actions_mask(self) -> np.ndarray:
        """
        Retourne un tableau bool de taille ACTION_SPACE_SIZE.
        True = action légale / pertinente pour cette phase.

        Phase MULLIGAN : 0=keep, 1=take — toujours les deux disponibles.
        Phase MAIN     : 0–6 jouer carte, 7 fin de tour (toujours dispo),
                         45.. attacher 1..10 DON!! par cible, puis [Activate: Main].
        Phase BATTLE   : 8 passer (toujours dispo), 9–44 attaques, 45.. DON!!.
        Phase BLOCKER  : 0 passer (toujours dispo), 1–5 board[0–4] (si eligible).

        Principe : PASS en Main / combat n'est autorisé que si le joueur a déjà fait
        une action « réelle » ce tour (jouer, attacher, Activate, attaque, DON!! en combat)
        ou s'il n'existe aucune action de ce type légalisable (évite les blocages).

        Retour : même buffer réutilisé — ``np.copy(mask)`` si stockage multi-pas.
        """
        mask = self._legal_mask_buf
        mask.fill(False)
        s = self._sim

        if s.phase == Phase.MULLIGAN:
            mask[MULLIGAN_KEEP] = True
            mask[MULLIGAN_TAKE] = True

        elif s.phase == Phase.BLOCKER:
            mask[BLOCKER_PASS] = True
            for i, b in enumerate(s.p0.board):
                if i < BLOCKER_N_SLOTS - 1 and b.has_blocker and not b.rested and not b.restrictions.blocks_attack_declaration():
                    mask[BLOCKER_SLOT_BASE + i] = True

        elif s.phase == Phase.MAIN:
            me, op = self._persp_me_opp()
            ctx = PlayabilityContext(Phase.MAIN, me.don, len(me.hand))
            for slot in range(min(len(me.hand), 7)):
                cid = me.hand[slot]
                cd = self._cards.get(cid)
                if cd is None:
                    continue
                if can_play_card(cd, ctx):
                    mask[slot] = True
            mask[MAIN_END_ACTION] = (
                s.turn_has_meaningful_activity()
                or not s.any_legal_main_commit_for_player(me)
            )
            if me.don_active > 0:
                d = me.don_active
                for rel in range(MAIN_ATTACH_DON_ACTIONS):
                    slot = rel // MAIN_ATTACH_DON_MAX
                    k = rel % MAIN_ATTACH_DON_MAX + 1
                    if k > d:
                        continue
                    if slot > 0 and (slot - 1) >= len(me.board):
                        continue
                    mask[MAIN_ATTACH_DON_BASE + rel] = True
            for i, b in enumerate(me.board):
                if i >= MAIN_ACTIVATE_MAIN_SLOTS:
                    break
                if b.rested or b.restrictions.blocks_attack_declaration() or b.activate_main_used:
                    continue
                cd_b = self._cards.get(b.card_id)
                if cd_b and cd_b.has_activate_main:
                    mask[MAIN_ACTIVATE_MAIN_BASE + i] = True
            if not me.leader_activate_main_used:
                leader_cd = self._cards.get(me.leader_id or "")
                if leader_cd and leader_cd.has_activate_main:
                    ltxt = (leader_cd.card_text or "").lower()
                    if "second turn or later" in ltxt and s.player_turn_number(me) < 2:
                        pass
                    else:
                        don_cost = int(leader_cd.activate_main_don_minus)
                        don_total = (me.don_active + me.leader_attached_don
                                     + sum(b.attached_don for b in me.board))
                        if don_total >= don_cost:
                            mask[MAIN_ACTIVATE_MAIN_LEADER] = True

        else:  # BATTLE
            me, op = self._persp_me_opp()
            mask[BATTLE_PASS_ACTION] = (
                s.turn_has_meaningful_activity()
                or not s.any_legal_battle_commit_for_pair(me, op)
            )
            for code in range(N_BATTLE_ATTACK_CODES):
                atk_slot = code // N_TARGETS
                tgt_slot = code % N_TARGETS
                if s.attack_is_legal(me, op, atk_slot, tgt_slot):
                    mask[BATTLE_ATTACK_BASE + code] = True
            if me.don_active > 0:
                d = me.don_active
                for rel in range(MAIN_ATTACH_DON_ACTIONS):
                    slot = rel // MAIN_ATTACH_DON_MAX
                    k = rel % MAIN_ATTACH_DON_MAX + 1
                    if k > d:
                        continue
                    if slot > 0 and (slot - 1) >= len(me.board):
                        continue
                    mask[MAIN_ATTACH_DON_BASE + rel] = True

        return mask

    # ── observation ───────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        """
        Observation courante (même buffer réutilisé : copier avec ``np.copy`` si vous
        stockez plusieurs pas dans une liste — voir ``ppo.collect_rollout``).
        """
        s = self._sim
        me, op = self._persp_me_opp()
        vec = self._obs_vec
        vec.fill(0.0)
        hs = self._hand_slots * self._emb
        _fill_hand_segment(vec, me.hand, self._hand_emb_vecs, self._hand_slots, self._emb)
        i0 = hs
        vec[i0 + 0] = len(me.board) / 4.0
        vec[i0 + 1] = len(op.board) / 4.0
        vec[i0 + 2] = me.don_active / 10.0
        vec[i0 + 3] = op.don_active / 10.0
        vec[i0 + 4] = me.life / 10.0
        vec[i0 + 5] = op.life / 10.0
        vec[i0 + 6]  = 1.0 if s.phase == Phase.MAIN else 0.0
        vec[i0 + 7]  = 1.0 if s.phase == Phase.BATTLE else 0.0
        vec[i0 + 8]  = self._rules_scalar
        vec[i0 + 9]  = min(
            1.0, sum(effective_power(b, self._cards) for b in me.board) / 20_000.0
        )
        vec[i0 + 10] = me.don_rested / 10.0
        vec[i0 + 11] = op.don_rested / 10.0
        vec[i0 + 12] = 1.0 if me.leader_rested else 0.0
        vec[i0 + 13] = 1.0 if op.leader_rested else 0.0
        vec[i0 + 14] = me.effective_leader_power() / 10_000.0
        vec[i0 + 15] = op.effective_leader_power() / 10_000.0
        vec[i0 + 16] = 1.0 if s.phase == Phase.MULLIGAN else 0.0
        vec[i0 + 17] = 1.0 if s.phase == Phase.BLOCKER else 0.0
        vec[i0 + 18] = getattr(s, "_blocker_atk_power", 0) / 10_000.0
        # Blockers du défenseur (toujours le plateau P0 en phase BLOCKER)
        blk_pl = s.p0 if s.phase == Phase.BLOCKER else me
        vec[i0 + 19] = sum(
            1 for b in blk_pl.board
            if b.has_blocker
            and not b.rested
            and not b.restrictions.blocks_attack_declaration()
        ) / 5.0

        vec[i0 + 20] = min(1.0, getattr(s, "_turn_steps", 0) / 20.0)
        vec[i0 + 21] = min(1.0, getattr(s, "_cards_played_turn", 0) / 5.0)
        vec[i0 + 22] = min(1.0, getattr(s, "_don_attached_turn", 0) / 10.0)
        vec[i0 + 23] = min(1.0, getattr(s, "_attacks_made_turn", 0) / 6.0)
        last_a = getattr(s, "_last_action", -1)
        vec[i0 + 24] = (last_a + 1) / (ACTION_SPACE_SIZE + 1)
        vec[i0 + 25] = sum(
            1 for b in me.board
            if character_may_declare_attack(
                rested=b.rested,
                just_played=b.just_played,
                has_rush=b.has_rush,
                has_rush_char=b.has_rush_char,
                restrictions=b.restrictions,
                keyword_grants=b.keyword_grants,
            )
        ) / 5.0
        vec[i0 + 26] = me.don_active / 10.0
        total_life = max(1, me.life + op.life)
        vec[i0 + 27] = me.life / total_life
        vec[i0 + 28] = 1.0 if me.stage_area else 0.0
        vec[i0 + 29] = 1.0 if op.stage_area else 0.0
        vec[i0 + 30] = 0.0 if me.leader_activate_main_used else 1.0
        vec[i0 + 31] = 0.0 if op.leader_activate_main_used else 1.0
        vec[i0 + 32] = (me.don_active + me.leader_attached_don
                        + sum(b.attached_don for b in me.board)) / 10.0
        vec[i0 + 33] = (op.don_active + op.leader_attached_don
                        + sum(b.attached_don for b in op.board)) / 10.0
        return vec

    # ── reward shaping ────────────────────────────────────────────────────────

    def _state_value(self) -> float:
        """
        Heuristique de valeur d'état normalisée.
        Utilisée pour calculer un reward intrinsèque (delta entre deux états).
        Pondération volontairement plus forte sur les **personnages** (plateau + puissance)
        pour que le signal GAE ne se résume pas à « leader qui tape seul ».
        En self-play vue P1, l'avantage est inversé (même politique, siège adverse).
        """
        s = self._sim
        if s.done:
            return 0.0
        val = 0.0
        val += s.p0.life * 0.38
        val -= s.p1.life * 0.38
        val += len(s.p0.board) * 0.26
        val -= len(s.p1.board) * 0.26
        p0_pow = sum(effective_power(b, self._cards) for b in s.p0.board)
        p1_pow = sum(effective_power(b, self._cards) for b in s.p1.board)
        val += (p0_pow - p1_pow) / 62_000.0
        val += (s.p0.don_active - s.p1.don_active) * 0.03
        total0 = max(1, len(s.p0.deck) + len(s.p0.hand) + s.p0.life)
        total1 = max(1, len(s.p1.deck) + len(s.p1.hand) + s.p1.life)
        val += (total0 - total1) / 100.0
        if getattr(s, "_self_play_external", False) and s.perspective_p1():
            return -val
        return val

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._sim.rng = np.random.default_rng(seed)
        self._close_log()
        self._step_idx = 0
        self._anim_frame_idx = 0

        if self._action_log_path is not None:
            self._action_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fp = self._action_log_path.open("w", encoding="utf-8")
            for line in log_session_preamble():
                self._log_line(line)
            self._log_line(
                f"# RESET seed={seed} deck0={len(self._deck0)} deck1={len(self._deck1)}"
            )

        if self._animation_log_path is not None:
            self._animation_log_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path = self._animation_log_path.with_name(
                self._animation_log_path.stem + ".manifest.json"
            )
            manifest: dict[str, dict[str, Any]] = {
                cid: {
                    "name": c.name,
                    "image_url": c.image_url,
                    "card_text": (c.card_text or "")[:800],
                    "keywords": list(c.keywords)[:48],
                    "power": c.power,
                    "cost": c.cost,
                    "card_type": c.card_type,
                }
                for cid, c in self._cards.items()
            }
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            self._anim_fp = self._animation_log_path.open("w", encoding="utf-8")

        self._sim.reset()
        self._episode_step_idx = 0
        self._prev_state_val: float = self._state_value()
        obs = self._obs()
        if self._log_fp is not None:
            self._log_line(f"# OBS_DIM={self._obs_dim} ACTION_SPACE={ACTION_SPACE_SIZE}")
            self._log_line("STATE " + self._state_snapshot())
        if self._anim_fp is not None:
            self._write_anim_frame("reset", seed=seed)
        return obs, {}

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: int):
        if self._sim.done:
            if self._log_fp is not None:
                self._log_line("STEP skip (partie déjà terminée)")
            return self._obs(), 0.0, True, False, {}

        s = self._sim
        if getattr(s, "_self_play_external", False):
            if s.phase == Phase.BLOCKER:
                self._step_reward_perspective = 0
            elif s.perspective_p1():
                self._step_reward_perspective = 1
            else:
                self._step_reward_perspective = 0
        else:
            self._step_reward_perspective = 0

        shaping_before: _EventShapingSnapshot | None = None
        if self._event_shaping_enabled and s.phase != Phase.MULLIGAN:
            me, op = self._persp_me_opp()
            shaping_before = _EventShapingSnapshot(
                opp_life=op.life,
                opp_board=len(op.board),
                me_deck=len(me.deck),
                me_hand=len(me.hand),
                me_board=len(me.board),
            )

        self._step_idx += 1
        self._episode_step_idx += 1

        phase_before = {
            Phase.MULLIGAN: "MULLIGAN",
            Phase.MAIN:     "MAIN",
            Phase.BATTLE:   "BATTLE",
            Phase.BLOCKER:  "BLOCKER",
        }.get(s.phase, "MAIN")
        action = int(action)

        need_trace = self._step_trace or self._log_fp is not None or self._anim_fp is not None
        desc = self._describe_action(action) if need_trace else ""

        r = 0.0
        if s.phase == Phase.MULLIGAN:
            # action 0=keep, 1=take ; tout autre → keep
            s.step_mulligan(action if action in (MULLIGAN_KEEP, MULLIGAN_TAKE) else MULLIGAN_KEEP)

        elif s.phase == Phase.BLOCKER:
            # action 0=pass, 1-5=board[0-4] ; tout hors plage → pass
            blocker_act = action if 0 <= action < BLOCKER_N_SLOTS else BLOCKER_PASS
            r = s.step_blocker(blocker_act)

        elif s.phase == Phase.MAIN:
            # Clamp : actions 8–44 (codes bataille hors MAIN valides) → fin de tour
            if MAIN_END_ACTION < action < MAIN_ATTACH_DON_BASE:
                action = MAIN_END_ACTION
            r = s.step_main(action)
            r += self._activity_after_idle_shaping_reward(phase_before, action, r)

        else:  # BATTLE
            # Clamp : actions < 8 hors attach DON!! → suggestion ou pass
            if action < BATTLE_PASS_ACTION:
                sug = s.suggest_first_battle_action()
                action = sug if sug is not None else BATTLE_PASS_ACTION
            r = s.step_battle(action)

        if shaping_before is not None:
            r += self._compute_event_shaping(shaping_before)

        if not s.done and self._max_turns_started > 0 and s.turns_started > self._max_turns_started:
            s.force_end_by_turn_cap()

        if (
            not s.done
            and self._max_episode_steps > 0
            and self._episode_step_idx >= self._max_episode_steps
        ):
            s.force_end_by_timeout()

        truncated = bool(
            getattr(s, "_timeout_forced", False)
            or getattr(s, "_turn_cap_forced", False)
            or getattr(s, "_stall_no_card_forced", False)
        )
        terminated = s.done and not truncated

        if s.winner is not None and (terminated or truncated):
            if getattr(s, "_self_play_external", False):
                r += (
                    self._win_reward
                    if s.winner == self._step_reward_perspective
                    else self._loss_reward
                )
            else:
                r += self._win_reward if s.winner == 0 else self._loss_reward
            if truncated:
                if getattr(s, "_turn_cap_forced", False):
                    r += self._turn_cap_penalty
                elif getattr(s, "_stall_no_card_forced", False):
                    r += self._stall_no_card_penalty
                elif getattr(s, "_timeout_forced", False):
                    r += self._timeout_both_penalty

        # ── Reward intrinsèque : delta de valeur d'état ───────────────────────
        if not terminated and not truncated:
            new_val = self._state_value()
            r += (new_val - self._prev_state_val) * 0.05
            self._prev_state_val = new_val

        episode_done = terminated or truncated
        if self._log_fp is not None:
            self._log_line(
                f"STEP {self._step_idx} {phase_before} action={action} "
                f"{desc} r={float(r):.4f} term={terminated} trunc={truncated} | {self._state_snapshot()}"
            )
            if episode_done and s.winner is not None:
                if getattr(s, "_turn_cap_forced", False):
                    tag = "TURN_CAP "
                elif getattr(s, "_stall_no_card_forced", False):
                    tag = "STALL_NO_CARD "
                elif truncated:
                    tag = "TIMEOUT "
                else:
                    tag = ""
                self._log_line(f"RESULT {tag}winner=P{s.winner} (0=agent)")

        if self._anim_fp is not None:
            self._write_anim_frame(
                "step",
                step=self._step_idx,
                phase_before=phase_before,
                action=action,
                action_desc=desc,
                reward=float(r),
                terminated=episode_done,
            )

        info: dict[str, Any] = {}
        if episode_done:
            info["winner"] = s.winner
            if truncated:
                if getattr(s, "_turn_cap_forced", False):
                    info["truncation_reason"] = "turn_cap"
                elif getattr(s, "_stall_no_card_forced", False):
                    info["truncation_reason"] = "stall_no_card"
                elif getattr(s, "_timeout_forced", False):
                    info["truncation_reason"] = "step_limit"
                    info["timeout_forced"] = True
            win_s = "p0" if s.winner == 0 else ("p1" if s.winner == 1 else None)
            info["training_row"] = {
                "deck0": self._deck0_path.name,
                "deck1": self._deck1_path.name,
                "steps": self._step_idx,
                "winner": win_s,
                "snapshot": self._training_snapshot_compact(),
                "state_line": self._state_snapshot(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        return self._obs(), float(r), terminated, truncated, info

    def close(self):
        self._close_log()
        try:
            super().close()
        except AttributeError:
            pass
