"""
Running mean / variance pour normaliser les observations (usage RL type VecNormalize).
Mise à jour en ligne (fusion de batches) — stable pour des milliers de steps.
"""
from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Normalise (x - mean) / sqrt(var + eps) ; mise à jour avec les obs brutes du simulateur."""

    __slots__ = ("mean", "var", "count", "epsilon", "clip_obs")

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        epsilon: float = 1e-4,
        clip_obs: float = 10.0,
    ) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)
        self.epsilon = float(epsilon)
        self.clip_obs = float(clip_obs)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[-1] != self.mean.shape[0]:
            raise ValueError(f"obs_rms shape mismatch: x {x.shape} vs mean {self.mean.shape}")
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = float(x.shape[0])
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: float,
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * (batch_count / tot_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalise avec les moments courants (après ``update`` si entraînement)."""
        x = np.asarray(x, dtype=np.float32)
        mean = self.mean.astype(np.float32, copy=False)
        std = np.sqrt(self.var.astype(np.float32, copy=False) + self.epsilon)
        y = (x - mean) / std
        if self.clip_obs > 0.0:
            np.clip(y, -self.clip_obs, self.clip_obs, out=y)
        return y

    def update_then_normalize(self, x: np.ndarray) -> np.ndarray:
        """Met à jour sur l’obs brute puis renvoie la version normalisée (rollout)."""
        self.update(x)
        return self.normalize(x)

    def normalize_eval(self, x: np.ndarray) -> np.ndarray:
        """Inférence / vidéo : pas de mise à jour des moments."""
        return self.normalize(x)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
            "epsilon": self.epsilon,
            "clip_obs": self.clip_obs,
        }

    def load_state_dict(self, d: dict) -> None:
        m = np.asarray(d["mean"], dtype=np.float64)
        v = np.asarray(d["var"], dtype=np.float64)
        if m.shape != self.mean.shape or v.shape != self.var.shape:
            raise ValueError(
                f"obs_rms : forme checkpoint {m.shape} incompatible avec le modèle {self.mean.shape}"
            )
        self.mean = m.copy()
        self.var = v.copy()
        self.count = float(d["count"])
        if "epsilon" in d:
            self.epsilon = float(d["epsilon"])
        if "clip_obs" in d:
            self.clip_obs = float(d["clip_obs"])
