"""
Bootstrap environnement pour PPO / VecEnv (factories, gauntlet, dashboard).

Sans dépendance PyTorch — utilisé par ``scripts/train_ppo_mlx.py`` pour un flux **MLX-only**
sur Apple Silicon (M1/M2/M3…).
"""
from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from opctcg_text_sim.env import OPTextSimEnv
from opctcg_text_sim.ppo import EpisodeStats, format_wall_elapsed


class _EnvFactory:
    """
    Callable picklable pour SubprocVecEnv (mp.spawn sur macOS exige pickle).
    Les lambdas et fonctions locales ne sont pas picklables.
    """

    def __init__(
        self,
        *,
        deck0,
        deck1,
        cards_csv,
        column_map,
        rules_corpus_path,
        sim_cfg,
        obs_dim,
        shaping_progress_ref=None,
    ):
        self.deck0 = deck0
        self.deck1 = deck1
        self.cards_csv = cards_csv
        self.column_map = column_map
        self.rules_corpus_path = rules_corpus_path
        self.sim_cfg = sim_cfg
        self.obs_dim = obs_dim
        self.shaping_progress_ref = shaping_progress_ref

    def __call__(self, seed: int):
        return OPTextSimEnv(
            self.deck0,
            self.deck1,
            self.cards_csv,
            self.column_map,
            self.rules_corpus_path,
            self.sim_cfg,
            obs_dim=self.obs_dim,
            seed=seed,
            shaping_progress_ref=self.shaping_progress_ref,
        )


def _iter_gauntlet_deck_txts(decks_dir: Path, deck_glob: str = "*.txt") -> list[Path]:
    if not decks_dir.is_dir():
        return []
    pat = deck_glob.strip() or "*.txt"
    return sorted(
        p
        for p in decks_dir.glob(pat)
        if p.is_file() and p.name.upper() != "README.TXT"
    )


def collect_gauntlet_valid_paths(
    decks_dir: Path,
    csv_path: Path,
    col_map: dict,
    deck_glob: str = "*.txt",
) -> list[Path]:
    """Même critère de validité que ``deck_gauntlet_web.py``."""
    from opctcg_text_sim.card_db import ensure_cards_for_deck, load_card_csv
    from opctcg_text_sim.deck_parser import deck_to_multiset, parse_deck_file
    from opctcg_text_sim.deck_validate import validate_deck_file

    deck_paths = _iter_gauntlet_deck_txts(decks_dir, deck_glob)
    if len(deck_paths) < 1:
        return []
    db = load_card_csv(csv_path, col_map)
    all_ids: set[str] = set()
    for p in deck_paths:
        all_ids |= set(deck_to_multiset(parse_deck_file(p)))
    cards = ensure_cards_for_deck(all_ids, db)
    out: list[Path] = []
    for p in deck_paths:
        if validate_deck_file(p, cards).ok:
            out.append(p)
    return out


def build_gauntlet_leader_by_path(
    paths: list[Path],
    csv_path: Path,
    col_map: dict,
) -> dict[str, str]:
    """``resolved_path_str -> leader_id`` pour diversifier les matchups gauntlet."""
    from opctcg_text_sim.card_db import ensure_cards_for_deck, load_card_csv
    from opctcg_text_sim.deck_parser import deck_to_multiset, infer_deck_leader_id, parse_deck_file

    db = load_card_csv(csv_path, col_map)
    all_ids: set[str] = set()
    for p in paths:
        all_ids |= set(deck_to_multiset(parse_deck_file(p)))
    cards = ensure_cards_for_deck(all_ids, db)
    out: dict[str, str] = {}
    for p in paths:
        lid = infer_deck_leader_id(p, cards) or ""
        out[str(p.resolve())] = lid
    return out


def pick_gauntlet_pair(
    paths: list[Path],
    worker_seed: int,
    seq: int,
    *,
    leader_by_resolved_str: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """
    Paire (deck0, deck1) pour une partie.

    - Toujours **deux fichiers distincts** si ``len(paths) >= 2``.
    - Si ``leader_by_resolved_str`` est fourni : **leaders différents** dès que le pool le permet
      (évite 31× Enel vs Enel quand un seul deck non-Enel existe dans le dossier).
    """
    import random

    rng = random.Random(int(worker_seed) ^ (int(seq) * 1_041_287_917))
    n = len(paths)
    if n <= 0:
        raise ValueError("gauntlet : liste de decks vide")
    if n == 1:
        p = paths[0]
        return p, p

    def _key(p: Path) -> str:
        return str(p.resolve())

    def _lid(p: Path) -> str:
        if not leader_by_resolved_str:
            return ""
        return leader_by_resolved_str.get(_key(p), "") or ""

    i0 = rng.randrange(n)
    d0 = paths[i0]
    ld0 = _lid(d0)

    cand: list[Path] = []
    for p in paths:
        if _key(p) == _key(d0):
            continue
        if ld0 and _lid(p) == ld0:
            continue
        cand.append(p)
    if not cand:
        cand = [p for p in paths if _key(p) != _key(d0)]
    if not cand:
        cand = list(paths)

    d1 = cand[rng.randrange(len(cand))]
    return d0, d1


def _deep_merge_dict(base: dict, patch: dict) -> None:
    """Fusion récursive in-place (``patch`` écrase / complète ``base``)."""
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge_dict(base[k], v)
        else:
            base[k] = v


def _cfg_opt_int(cfg: dict, key: str) -> int | None:
    v = cfg.get(key)
    if v is None or (isinstance(v, str) and not str(v).strip()):
        return None
    return int(v)


class _ResamplingGauntletEnv(gym.Env):
    """
    À chaque ``reset()`` : nouvelle paire (deck0, deck1) tirée au hasard dans ``valid_paths``.
    Utilisé par les workers VecEnv (chaque partie terminée → reset → nouveaux decks).
    """

    metadata = OPTextSimEnv.metadata

    def __init__(
        self,
        *,
        valid_paths: list[Path],
        cards_csv: Path,
        column_map: dict,
        rules_corpus_path: Path | None,
        sim_cfg: dict,
        obs_dim: int,
        worker_seed: int,
        leader_by_resolved_str: dict[str, str] | None = None,
        shaping_progress_ref=None,
    ) -> None:
        super().__init__()
        if not valid_paths:
            raise ValueError("gauntlet : valid_paths vide")
        self._paths = tuple(valid_paths)
        self._cards_csv = cards_csv
        self._column_map = column_map
        self._rules_corpus_path = rules_corpus_path
        self._sim_cfg = sim_cfg
        self._obs_dim = obs_dim
        self._worker_seed = int(worker_seed)
        self._leader_by_str = leader_by_resolved_str
        self._shaping_progress_ref = shaping_progress_ref
        self._seq = 0
        self._inner: OPTextSimEnv | None = None
        p0, p1 = self._paths[0], self._paths[min(1, len(self._paths) - 1)]
        probe = OPTextSimEnv(
            p0,
            p1,
            cards_csv,
            column_map,
            rules_corpus_path,
            sim_cfg,
            obs_dim=obs_dim,
            seed=0,
            shaping_progress_ref=shaping_progress_ref,
        )
        self.observation_space = probe.observation_space
        self.action_space = probe.action_space
        probe.close()

    def _rebuild_inner(self, reset_seed: int | None) -> None:
        d0, d1 = pick_gauntlet_pair(
            list(self._paths),
            self._worker_seed,
            self._seq,
            leader_by_resolved_str=self._leader_by_str,
        )
        self._seq += 1
        if self._inner is not None:
            self._inner.close()
        rs = 0 if reset_seed is None else int(reset_seed)
        self._inner = OPTextSimEnv(
            d0,
            d1,
            self._cards_csv,
            self._column_map,
            self._rules_corpus_path,
            self._sim_cfg,
            obs_dim=self._obs_dim,
            seed=rs,
            shaping_progress_ref=self._shaping_progress_ref,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._rebuild_inner(seed)
        assert self._inner is not None
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        assert self._inner is not None
        return self._inner.step(action)

    def legal_actions_mask(self):
        assert self._inner is not None
        return self._inner.legal_actions_mask()

    def close(self):
        if self._inner is not None:
            self._inner.close()
            self._inner = None


class _ResamplingGauntletFactory:
    """Callable picklable ``(seed) -> _ResamplingGauntletEnv`` pour SubprocVecEnv."""

    __slots__ = (
        "valid_paths",
        "cards_csv",
        "column_map",
        "rules_corpus_path",
        "sim_cfg",
        "obs_dim",
        "leader_by_resolved_str",
        "shaping_progress_ref",
    )

    def __init__(
        self,
        *,
        valid_paths: list[Path],
        cards_csv: Path,
        column_map: dict,
        rules_corpus_path: Path | None,
        sim_cfg: dict,
        obs_dim: int,
        leader_by_resolved_str: dict[str, str] | None = None,
        shaping_progress_ref=None,
    ):
        if not valid_paths:
            raise ValueError("gauntlet : valid_paths vide")
        self.valid_paths = list(valid_paths)
        self.cards_csv = cards_csv
        self.column_map = column_map
        self.rules_corpus_path = rules_corpus_path
        self.sim_cfg = sim_cfg
        self.obs_dim = obs_dim
        self.leader_by_resolved_str = leader_by_resolved_str
        self.shaping_progress_ref = shaping_progress_ref

    def __call__(self, seed: int) -> _ResamplingGauntletEnv:
        return _ResamplingGauntletEnv(
            valid_paths=self.valid_paths,
            cards_csv=self.cards_csv,
            column_map=self.column_map,
            rules_corpus_path=self.rules_corpus_path,
            sim_cfg=self.sim_cfg,
            obs_dim=self.obs_dim,
            worker_seed=seed,
            leader_by_resolved_str=self.leader_by_resolved_str,
            shaping_progress_ref=self.shaping_progress_ref,
        )


class TrainingDashboard:
    """Affiche un résumé glissant de l'entraînement."""

    def __init__(self, total_updates: int, rollout_len: int, *, infinite: bool = False):
        self.total_updates = total_updates
        self.rollout_len = rollout_len
        self.infinite = infinite
        self.start_time = time.time()
        self.history_rewards: list[float] = []
        self.history_wins: list[float] = []

    def update(
        self,
        it: int,
        stats: EpisodeStats,
        metrics: dict[str, float],
        extra: str = "",
    ) -> None:
        if stats.n_episodes > 0:
            self.history_rewards.append(stats.mean_reward)
            self.history_wins.append(stats.win_rate * 100)

        elapsed = time.time() - self.start_time
        steps_done = (it + 1) * self.rollout_len
        steps_per_s = steps_done / max(1.0, elapsed)

        recent_rew = float(np.mean(self.history_rewards[-10:])) if self.history_rewards else 0.0
        recent_win = float(np.mean(self.history_wins[-10:])) if self.history_wins else 0.0

        if self.infinite:
            bar = "█" * 8 + "░" * 22
            print(
                f"\r[{bar}] stp={steps_done:>10}  upd={it + 1}  "
                f"rew={stats.mean_reward:+.2f} (avg:{recent_rew:+.2f})  "
                f"win={stats.win_rate*100:.0f}% (avg:{recent_win:.0f}%)  "
                f"ep={stats.n_episodes}  len={stats.mean_length:.0f}  "
                f"loss={metrics['loss']:.4f}  ent={metrics['ent']:.3f}  "
                f"clip={metrics['clip']:.2f}  "
                f"{steps_per_s:.0f}stp/s  t={format_wall_elapsed(elapsed)}  (stop ou fichier STOP)"
                + (f"  {extra}" if extra else ""),
                end="",
                flush=True,
            )
            return

        total_steps = self.total_updates * self.rollout_len
        eta_s = max(0, (total_steps - steps_done) / max(1.0, steps_per_s))

        bar_len = 30
        filled = int(bar_len * steps_done / total_steps)
        bar = "█" * filled + "░" * (bar_len - filled)

        print(
            f"\r[{bar}] {steps_done:>7}/{total_steps}  "
            f"rew={stats.mean_reward:+.2f} (avg:{recent_rew:+.2f})  "
            f"win={stats.win_rate*100:.0f}% (avg:{recent_win:.0f}%)  "
            f"ep={stats.n_episodes}  len={stats.mean_length:.0f}  "
            f"loss={metrics['loss']:.4f}  ent={metrics['ent']:.3f}  "
            f"clip={metrics['clip']:.2f}  "
            f"{steps_per_s:.0f}stp/s  t={format_wall_elapsed(elapsed)}  "
            f"ETA:{int(eta_s // 60)}m{int(eta_s % 60):02d}s"
            + (f"  {extra}" if extra else ""),
            end="",
            flush=True,
        )

    def newline(self) -> None:
        print(flush=True)
