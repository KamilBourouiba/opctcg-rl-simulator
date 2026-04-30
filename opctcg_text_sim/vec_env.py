"""
SubprocVecEnv — environnements parallèles via multiprocessing.
Chaque worker tourne dans son propre processus Python (contourne le GIL).
Optimisé pour Apple Silicon (M1 Pro : 8 cœurs CPU).

Usage :
    from opctcg_text_sim.vec_env import make_subproc_vec_env
    vec = make_subproc_vec_env(env_factory, n_envs=6)
    obs = vec.reset()               # (n_envs, obs_dim)
    masks = vec.legal_actions_masks()  # (n_envs, n_act)
    obs, rews, dones, infos, masks = vec.step(actions)  # masques pour l’état suivant
    vec.close()
"""
from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Any, Callable

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Commandes worker
# ──────────────────────────────────────────────────────────────────────────────
_CMD_RESET  = "reset"
_CMD_STEP   = "step"
_CMD_MASK   = "mask"
_CMD_CLOSE  = "close"
_CMD_ATTR   = "attr"    # lire un attribut de l'env


def _worker(
    conn: mp.connection.Connection,
    env_fn: Callable,
    seed: int,
) -> None:
    """Process worker : attend des commandes et renvoie les résultats."""
    from .runtime_resources import configure_worker_blas_threads

    configure_worker_blas_threads()
    try:
        env = env_fn(seed)
        while True:
            cmd, data = conn.recv()
            if cmd == _CMD_RESET:
                obs, info = env.reset(seed=data)
                if hasattr(env, "legal_actions_mask"):
                    mask = np.asarray(env.legal_actions_mask(), dtype=bool)
                else:
                    mask = np.ones(env.action_space.n, dtype=bool)
                conn.send(("ok", (obs, mask)))
            elif cmd == _CMD_STEP:
                obs, r, term, trunc, info = env.step(data)
                done = term or trunc
                if done:
                    obs, _ = env.reset()
                if hasattr(env, "legal_actions_mask"):
                    mask = np.asarray(env.legal_actions_mask(), dtype=bool)
                else:
                    mask = np.ones(env.action_space.n, dtype=bool)
                conn.send(("ok", (obs, float(r), done, info, mask)))
            elif cmd == _CMD_MASK:
                if hasattr(env, "legal_actions_mask"):
                    mask = env.legal_actions_mask()
                else:
                    n = env.action_space.n
                    mask = np.ones(n, dtype=bool)
                conn.send(("ok", mask))
            elif cmd == _CMD_ATTR:
                conn.send(("ok", getattr(env, data, None)))
            elif cmd == _CMD_CLOSE:
                env.close()
                conn.send(("ok", None))
                break
            else:
                conn.send(("err", f"Commande inconnue : {cmd}"))
    except Exception:
        conn.send(("err", traceback.format_exc()))


class SubprocVecEnv:
    """
    Vectorised environment : N copies tournant dans N processus séparés.

    Paramètres
    ----------
    env_fns : liste de callables ``fn(seed) -> gym.Env``
    seeds   : graines par env (défaut : 0, 1, 2, …)
    """

    def __init__(
        self,
        env_fns: list[Callable],
        seeds: list[int] | None = None,
    ):
        from .runtime_resources import raise_process_file_limit

        self.n_envs = len(env_fns)
        if seeds is None:
            seeds = list(range(self.n_envs))

        raise_process_file_limit(max(8192, self.n_envs * 24))

        ctx = mp.get_context("spawn")   # spawn requis sur macOS
        self._parents: list[mp.connection.Connection] = []
        self._procs:   list[mp.Process] = []

        for i, fn in enumerate(env_fns):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(child_conn, fn, seeds[i]),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._parents.append(parent_conn)
            self._procs.append(proc)

        # Récupérer action_space et observation_space depuis env 0
        self._parents[0].send((_CMD_ATTR, "action_space"))
        _, self.action_space = self._parents[0].recv()
        self._parents[0].send((_CMD_ATTR, "observation_space"))
        _, self.observation_space = self._parents[0].recv()
        self._last_obs: np.ndarray | None = None
        self._last_masks: np.ndarray | None = None

    # ── interface publique ──────────────────────────────────────────────────

    def reset(self, *, seeds: list[int] | None = None) -> np.ndarray:
        """Remet à zéro tous les envs. Retourne obs (n_envs, obs_dim)."""
        obs, _ = self.reset_with_masks(seeds=seeds)
        return obs

    def reset_with_masks(
        self, *, seeds: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reset + masques légaux initiaux (un aller-retour groupé par worker)."""
        if seeds is None:
            seeds = [None] * self.n_envs
        for conn, s in zip(self._parents, seeds):
            conn.send((_CMD_RESET, s))
        obs_list: list[np.ndarray] = []
        mask_list: list[np.ndarray] = []
        for conn in self._parents:
            status, data = conn.recv()
            if status == "err":
                raise RuntimeError(f"Worker error on reset:\n{data}")
            obs, mask = data
            obs_list.append(obs)
            mask_list.append(mask)
        self._last_obs = np.stack(obs_list)
        self._last_masks = np.stack(mask_list)
        return self._last_obs, self._last_masks

    def step(
        self, actions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], np.ndarray]:
        """
        Envoie les actions à tous les envs en parallèle.
        Si un épisode se termine, le worker reset automatiquement.
        Retourne (obs, rewards, dones, infos, legal_masks) — masques pour l'état suivant.
        """
        for conn, a in zip(self._parents, actions):
            conn.send((_CMD_STEP, int(a)))
        obs_list: list[np.ndarray] = []
        rew_list: list[float] = []
        done_list: list[bool] = []
        info_list: list[dict] = []
        mask_list: list[np.ndarray] = []
        for conn in self._parents:
            status, data = conn.recv()
            if status == "err":
                raise RuntimeError(f"Worker error on step:\n{data}")
            obs, r, done, info, mask = data
            obs_list.append(obs)
            rew_list.append(r)
            done_list.append(done)
            info_list.append(info or {})
            mask_list.append(mask)
        self._last_obs = np.stack(obs_list)
        self._last_masks = np.stack(mask_list)
        return (
            self._last_obs,
            np.array(rew_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
            info_list,
            self._last_masks,
        )

    def legal_actions_masks(self) -> np.ndarray:
        """Retourne le masque légal de chaque env (n_envs, n_act)."""
        if self._last_masks is not None:
            return self._last_masks
        for conn in self._parents:
            conn.send((_CMD_MASK, None))
        masks = []
        for conn in self._parents:
            status, data = conn.recv()
            if status == "err":
                raise RuntimeError(f"Worker error on mask:\n{data}")
            masks.append(data)
        return np.stack(masks)

    def close(self) -> None:
        for conn in self._parents:
            try:
                conn.send((_CMD_CLOSE, None))
                conn.recv()
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Factory helper
# ──────────────────────────────────────────────────────────────────────────────

def make_subproc_vec_env(
    env_factory: Callable[[int], Any],
    n_envs: int,
    base_seed: int = 0,
) -> SubprocVecEnv:
    """
    Crée un SubprocVecEnv à partir d'une factory ``fn(seed) -> gym.Env``.

    Exemple :
        def make_env(seed):
            return OPTextSimEnv(deck0, deck1, csv, col, None, sim_cfg, seed=seed)

        vec = make_subproc_vec_env(make_env, n_envs=6, base_seed=42)
    """
    fns = [env_factory for _ in range(n_envs)]
    seeds = [base_seed + i for i in range(n_envs)]
    return SubprocVecEnv(fns, seeds=seeds)
