"""
PPO (Proximal Policy Optimization) avec :
  - Masquage des actions illégales (logit = -1e9 pour les actions interdites)
  - GAE-λ (Generalized Advantage Estimation) pour des estimations plus stables
  - Entropy bonus avec décroissance linéaire
  - Collecte de statistiques d'épisodes (win rate, récompense moyenne, longueur)
  - Normalisation optionnelle des observations et des rewards (running σ)
  - PPO : clipping ε (annealing optionnel), arrêt anticipé si KL dépasse un seuil
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from .obs_rms import RunningMeanStd

_NEG_INF = float("-inf")  # Masque actions illégales — compatible fp16 et fp32


def _rms_obs_train(raw: np.ndarray, rms: RunningMeanStd | None) -> np.ndarray:
    """Obs brute → normalisée (mise à jour des moments) pour le rollout."""
    if rms is None:
        return raw
    return rms.update_then_normalize(raw)


def _rms_reward_train(rewards: np.ndarray, rms: RunningMeanStd | None) -> np.ndarray:
    """
    Rewards bruts du simulateur → mis à jour RMS scalaire puis r / σ (sans centrage).
    Les stats d'épisode (mean_reward, win) restent calculées sur les rewards **bruts** dans la collecte.
    """
    if rms is None:
        return np.asarray(rewards, dtype=np.float32)
    r64 = np.asarray(rewards, dtype=np.float64).reshape(-1, 1)
    rms.update(r64)
    std = float(np.sqrt(rms.var[0] + rms.epsilon))
    inv = 1.0 / max(std, 1e-8)
    return (np.asarray(rewards, dtype=np.float32) * inv).astype(np.float32, copy=False)


def format_wall_elapsed(seconds: float) -> str:
    """Durée humaine pour les logs (ex. ``1h02m05s``, ``12m30s``, ``8s``)."""
    s = max(0.0, float(seconds))
    total = int(round(s))
    h, r = divmod(total, 3600)
    m, sec = divmod(r, 60)
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


# ──────────────────────────────────────────────────────────────────────────────
# Réseau Actor-Critic
# ──────────────────────────────────────────────────────────────────────────────

class _ResBlock(nn.Module):
    """Bloc résiduel : Linear → LN → LeakyReLU → Linear → LN + skip."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.LeakyReLU(0.1)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ActorCritic(nn.Module):
    """
    Réseau partagé avec blocs résiduels pour une meilleure stabilité du gradient.
    Architecture : projection → 3 ResBlocks → tête politique + tête valeur.
    """
    def __init__(self, obs_dim: int, n_act: int, hidden: int = 256, n_res: int = 2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1),
        )
        self.blocks = nn.Sequential(*[_ResBlock(hidden) for _ in range(n_res)])
        self.pi_head = nn.Linear(hidden, n_act)
        self.v_head  = nn.Linear(hidden, 1)

        # Initialisation orthogonale (standard PPO)
        for m in [*self.proj.modules(), *self.blocks.modules()]:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi_head.weight, gain=0.01)
        nn.init.zeros_(self.pi_head.bias)
        nn.init.zeros_(self.v_head.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x    : (B, obs_dim) float32
        mask : (B, n_act) bool  — True = action légale
        Retourne (logits, values) après application éventuelle du masque.
        """
        h = self.blocks(self.proj(x))
        logits = self.pi_head(h)
        if mask is not None:
            logits = logits.masked_fill(~mask, _NEG_INF)
        return logits, self.v_head(h).squeeze(-1)

    def act(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Échantillonne une action ; retourne (action, log_prob, value)."""
        logits, val = self.forward(x, mask)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), val


# ──────────────────────────────────────────────────────────────────────────────
# Structures de données
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RolloutBatch:
    obs:    torch.Tensor
    act:    torch.Tensor
    logp:   torch.Tensor
    ret:    torch.Tensor   # GAE returns (cible pour la value)
    adv:    torch.Tensor   # Avantages normalisés
    mask:   torch.Tensor   # bool mask d'actions légales


class EpisodeStats(NamedTuple):
    mean_reward:  float
    mean_length:  float
    win_rate:     float     # fréquence de victoire de l'agent (P0)
    n_episodes:   int


# ──────────────────────────────────────────────────────────────────────────────
# GAE-λ
# ──────────────────────────────────────────────────────────────────────────────

def _gae(
    rewards: list[float],
    values:  list[float],
    dones:   list[bool],
    last_val: float,
    gamma: float,
    lam: float,
) -> tuple[list[float], list[float]]:
    """
    Calcul GAE-λ (Schulman 2016).
    Retourne (advantages, returns_for_value_loss).
    """
    n = len(rewards)
    adv   = [0.0] * n
    ret   = [0.0] * n
    gae   = 0.0
    v_next = last_val
    for t in reversed(range(n)):
        mask_t  = 0.0 if dones[t] else 1.0
        delta   = rewards[t] + gamma * v_next * mask_t - values[t]
        gae     = delta + gamma * lam * mask_t * gae
        adv[t]  = gae
        ret[t]  = gae + values[t]
        v_next  = values[t]
    return adv, ret


def _gae_np_batched(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_vals: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GAE-λ pour ``n`` trajectoires en parallèle — shapes (n, T).
    ``last_vals`` : V(s_fin) par env ; mis à 0 où l’épisode se termine au dernier pas du rollout.
    """
    n_env, T = rewards.shape
    adv = np.empty((n_env, T), dtype=np.float32)
    ret = np.empty((n_env, T), dtype=np.float32)
    gae = np.zeros(n_env, dtype=np.float32)
    v_next = last_vals.astype(np.float32, copy=False)
    if dones.shape[1] > 0:
        v_next = np.where(dones[:, -1], 0.0, v_next)
    for t in range(T - 1, -1, -1):
        m = (~dones[:, t]).astype(np.float32)
        delta = rewards[:, t] + gamma * v_next * m - values[:, t]
        gae = delta + gamma * lam * m * gae
        adv[:, t] = gae
        ret[:, t] = gae + values[:, t]
        v_next = values[:, t]
    return adv, ret


def _adamw_for_device(
    params,
    lr: float,
    weight_decay: float,
    device: torch.device,
) -> optim.Optimizer:
    """AdamW : ``fused`` sur CUDA si dispo ; ``foreach`` sur CPU/MPS quand supporté (PyTorch 2+)."""
    base_kw: dict = dict(lr=lr, eps=1e-5, weight_decay=weight_decay)
    if device.type == "cuda":
        try:
            return optim.AdamW(params, **base_kw, fused=True)
        except (TypeError, ValueError):
            pass
    if device.type in ("cpu", "mps"):
        try:
            return optim.AdamW(params, **base_kw, foreach=True)
        except TypeError:
            pass
    return optim.AdamW(params, **base_kw)


# ──────────────────────────────────────────────────────────────────────────────
# Collecte du rollout — env unique
# ──────────────────────────────────────────────────────────────────────────────

def collect_rollout(
    env,
    net: ActorCritic,
    steps: int,
    gamma: float,
    lam: float,
    device: torch.device,
    obs_rms: RunningMeanStd | None = None,
    reward_rms: RunningMeanStd | None = None,
) -> tuple[RolloutBatch, EpisodeStats]:
    """
    Collecte `steps` transitions depuis un seul env.
    Utilise env.legal_actions_mask() si disponible.
    """
    obs_l:  list[np.ndarray]  = []
    act_l:  list[int]         = []
    rew_l:  list[float]       = []
    logp_l: list[float]       = []
    done_l: list[bool]        = []
    val_l:  list[float]       = []
    mask_l: list[np.ndarray]  = []

    ep_rewards:  list[float] = []
    ep_lengths:  list[int]   = []
    ep_wins:     list[bool]  = []
    cur_reward   = 0.0
    cur_len      = 0
    n_act        = env.action_space.n

    o, _ = env.reset()
    net.eval()
    with torch.inference_mode():
        for _ in range(steps):
            has_mask = hasattr(env, "legal_actions_mask")
            m_np = env.legal_actions_mask() if has_mask else np.ones(n_act, dtype=bool)
            if not m_np.any():
                m_np[:] = True

            o_feed = _rms_obs_train(o, obs_rms)
            x   = torch.as_tensor(o_feed, dtype=torch.float32, device=device).unsqueeze(0)
            m_t = torch.as_tensor(m_np, dtype=torch.bool,    device=device).unsqueeze(0)

            a, logp, val = net.act(x, m_t)
            a_int = int(a.item())

            o2, r, terminated, truncated, info = env.step(a_int)
            done = terminated or truncated
            r_raw = float(r)
            r_train = float(_rms_reward_train(np.array([r_raw], dtype=np.float32), reward_rms)[0])

            # Copie obs / masque : l'env peut réutiliser les mêmes buffers numpy à chaque step.
            obs_l.append(np.copy(o_feed))
            act_l.append(a_int)
            rew_l.append(r_train)
            logp_l.append(float(logp.item()))
            done_l.append(done)
            val_l.append(float(val.item()))
            mask_l.append(np.copy(m_np))

            cur_reward += r_raw;  cur_len += 1
            o = o2

            if done:
                ep_rewards.append(cur_reward);  ep_lengths.append(cur_len)
                ep_wins.append(cur_reward > 0.0)
                cur_reward = 0.0;  cur_len = 0
                o, _ = env.reset()

        o_last = _rms_obs_train(o, obs_rms)
        x_last  = torch.as_tensor(o_last, dtype=torch.float32, device=device).unsqueeze(0)
        _, lv   = net.forward(x_last)
        last_val = float(lv.item()) if not done_l[-1] else 0.0

    adv_l, ret_l = _gae(rew_l, val_l, done_l, last_val, gamma, lam)
    return _build_batch(obs_l, act_l, logp_l, ret_l, adv_l, mask_l, device), EpisodeStats(
        mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        mean_length = float(np.mean(ep_lengths)) if ep_lengths else float(steps),
        win_rate    = float(np.mean(ep_wins))    if ep_wins    else 0.0,
        n_episodes  = len(ep_rewards),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Collecte vectorisée (SubprocVecEnv — N envs en parallèle)
# ──────────────────────────────────────────────────────────────────────────────

def collect_rollout_vec(
    vec_env,
    net: ActorCritic,
    steps_per_env: int,
    gamma: float,
    lam: float,
    device: torch.device,
    obs_rms: RunningMeanStd | None = None,
    reward_rms: RunningMeanStd | None = None,
) -> tuple[RolloutBatch, EpisodeStats]:
    """
    Collecte ``steps_per_env × n_envs`` transitions en parallèle.
    Optimisé : tampons numpy, GAE vectorisé par env, pas d’IPC masque par pas si
    ``SubprocVecEnv`` renvoie les masques dans ``step`` / ``reset_with_masks``.
    """
    n = vec_env.n_envs
    n_act = vec_env.action_space.n
    spe = steps_per_env
    obs_dim = int(np.prod(vec_env.observation_space.shape))

    buf_obs = np.empty((spe, n, obs_dim), dtype=np.float32)
    buf_act = np.empty((spe, n), dtype=np.int64)
    buf_rew = np.empty((spe, n), dtype=np.float32)
    buf_logp = np.empty((spe, n), dtype=np.float32)
    buf_done = np.empty((spe, n), dtype=bool)
    buf_val = np.empty((spe, n), dtype=np.float32)
    buf_mask = np.empty((spe, n, n_act), dtype=bool)

    ep_rewards_all: list[list[float]] = [[] for _ in range(n)]
    ep_lengths_all: list[list[int]] = [[] for _ in range(n)]
    ep_wins_all: list[list[bool]] = [[] for _ in range(n)]
    cur_rew = np.zeros(n, dtype=np.float32)
    cur_len = np.zeros(n, dtype=np.int32)

    if hasattr(vec_env, "reset_with_masks"):
        obs_np, masks_np = vec_env.reset_with_masks()
    else:
        obs_np = vec_env.reset()
        masks_np = vec_env.legal_actions_masks()

    net.eval()
    with torch.inference_mode():
        for t in range(spe):
            no_legal = ~masks_np.any(axis=1)
            masks_np[no_legal] = True

            obs_feed = _rms_obs_train(obs_np, obs_rms)
            if obs_feed.dtype == np.float32 and obs_feed.flags["C_CONTIGUOUS"]:
                x = torch.from_numpy(obs_feed).to(device=device, dtype=torch.float32)
            else:
                x = torch.as_tensor(obs_feed, dtype=torch.float32, device=device)
            if masks_np.flags["C_CONTIGUOUS"]:
                m_t = torch.from_numpy(masks_np).to(device=device)
            else:
                m_t = torch.as_tensor(masks_np, dtype=torch.bool, device=device)

            logits, vals = net(x, m_t)
            dist = Categorical(logits=logits)
            acts = dist.sample()
            logps = dist.log_prob(acts)

            # Un seul passage CPU←device pour logp + value (moins de sync MPS/CUDA).
            _lv = torch.stack((logps, vals.float()), dim=1).detach().cpu().numpy()
            buf_logp[t] = _lv[:, 0]
            buf_val[t] = _lv[:, 1]
            acts_np = acts.detach().cpu().numpy()
            step_out = vec_env.step(acts_np)
            if len(step_out) == 5:
                obs2_np, rews_np, dones_np, _infos, masks_np = step_out
            elif len(step_out) == 4:
                # SharedMemoryVecEnv : (obs, rews, dones, masks) — pas de ``infos``.
                fourth = step_out[3]
                if isinstance(fourth, dict):
                    obs2_np, rews_np, dones_np, _infos = step_out
                    masks_np = vec_env.legal_actions_masks()
                else:
                    obs2_np, rews_np, dones_np, masks_np = step_out
                    _infos = {}
            else:
                raise ValueError(f"vec_env.step a renvoyé {len(step_out)} valeurs (attendu 4 ou 5)")

            buf_obs[t] = obs_feed
            buf_act[t] = acts_np.astype(np.int64, copy=False)
            buf_rew[t] = _rms_reward_train(rews_np.astype(np.float32, copy=False), reward_rms)
            buf_done[t] = dones_np
            buf_mask[t] = masks_np

            cur_rew += rews_np.astype(np.float32, copy=False)
            cur_len += 1
            for i in np.flatnonzero(dones_np):
                ep_rewards_all[int(i)].append(float(cur_rew[i]))
                ep_lengths_all[int(i)].append(int(cur_len[i]))
                ep_wins_all[int(i)].append(bool(cur_rew[i] > 0))
                cur_rew[i] = 0.0
                cur_len[i] = 0

            obs_np = obs2_np

        obs_last = _rms_obs_train(obs_np, obs_rms)
        if obs_last.dtype == np.float32 and obs_last.flags["C_CONTIGUOUS"]:
            x_last = torch.from_numpy(obs_last).to(device=device, dtype=torch.float32)
        else:
            x_last = torch.as_tensor(obs_last, dtype=torch.float32, device=device)
        _, lv = net(x_last)
        last_vals = lv.detach().cpu().numpy()

    rews_T = buf_rew.T
    vals_T = buf_val.T
    dones_T = buf_done.T
    adv_buf, ret_buf = _gae_np_batched(rews_T, vals_T, dones_T, last_vals, gamma, lam)

    total = n * spe
    obs_flat = buf_obs.transpose(1, 0, 2).reshape(total, -1)
    act_flat = buf_act.T.reshape(total)
    logp_flat = buf_logp.T.reshape(total)
    mask_flat = buf_mask.transpose(1, 0, 2).reshape(total, -1)
    adv_flat = adv_buf.reshape(total)
    ret_flat = ret_buf.reshape(total)

    obs_t = torch.as_tensor(np.ascontiguousarray(obs_flat), dtype=torch.float32, device=device)
    act_t = torch.as_tensor(np.ascontiguousarray(act_flat), dtype=torch.int64, device=device)
    logp_t = torch.as_tensor(np.ascontiguousarray(logp_flat), dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(np.ascontiguousarray(ret_flat), dtype=torch.float32, device=device)
    adv_t = torch.as_tensor(np.ascontiguousarray(adv_flat), dtype=torch.float32, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    mask_t = torch.as_tensor(np.ascontiguousarray(mask_flat), dtype=torch.bool, device=device)
    batch = RolloutBatch(obs_t, act_t, logp_t, ret_t, adv_t, mask_t)

    all_rews = [r for lst in ep_rewards_all for r in lst]
    all_lens = [ln for lst in ep_lengths_all for ln in lst]
    all_wins = [w for lst in ep_wins_all for w in lst]

    stats = EpisodeStats(
        mean_reward=float(np.mean(all_rews)) if all_rews else 0.0,
        mean_length=float(np.mean(all_lens)) if all_lens else float(spe),
        win_rate=float(np.mean(all_wins)) if all_wins else 0.0,
        n_episodes=len(all_rews),
    )
    return batch, stats


# ──────────────────────────────────────────────────────────────────────────────
# Collecte avec CoreMLPolicy (ANE) + SharedMemoryVecEnv
# ──────────────────────────────────────────────────────────────────────────────

def collect_rollout_coreml(
    shmem_env,
    policy,         # CoreMLPolicy
    net: ActorCritic,
    steps_per_env: int,
    gamma: float,
    lam: float,
    device: torch.device,
    *,
    init_obs:   "np.ndarray | None" = None,
    init_masks: "np.ndarray | None" = None,
    obs_rms: RunningMeanStd | None = None,
    reward_rms: RunningMeanStd | None = None,
) -> "tuple[RolloutBatch, EpisodeStats, np.ndarray, np.ndarray]":
    """
    Collecte avec CoreMLPolicy (inférence ANE) + SharedMemoryVecEnv (shm IPC).
    Utilise des tableaux numpy pré-alloués (pas de listes Python) pour minimiser
    les copies et la surcharge de conversion → MPS.
    """
    n     = shmem_env.n_envs
    n_act = shmem_env.n_act
    spe   = steps_per_env

    # Tableaux pré-alloués (spe, n, dim) — zéro allocation dans la boucle
    buf_obs  = np.empty((spe, n, shmem_env.obs_dim), dtype=np.float32)
    buf_act  = np.empty((spe, n),                    dtype=np.int64)
    buf_rew  = np.empty((spe, n),                    dtype=np.float32)
    buf_logp = np.empty((spe, n),                    dtype=np.float32)
    buf_done = np.empty((spe, n),                    dtype=bool)
    buf_val  = np.empty((spe, n),                    dtype=np.float32)
    buf_mask = np.empty((spe, n, n_act),             dtype=bool)

    ep_rewards_all: list[list[float]] = [[] for _ in range(n)]
    ep_lengths_all: list[list[int]]   = [[] for _ in range(n)]
    ep_wins_all:    list[list[bool]]  = [[] for _ in range(n)]
    cur_rew = np.zeros(n, dtype=np.float32)
    cur_len = np.zeros(n, dtype=np.int32)

    # Réutiliser obs/masks du rollout précédent si disponibles (évite reset_all inutile)
    if init_obs is not None and init_masks is not None:
        obs_np   = init_obs
        masks_np = init_masks
    else:
        obs_np   = shmem_env.reset_all()
        masks_np = shmem_env.legal_actions_masks()

    _rng_shape = (n, n_act)
    for t in range(spe):
        no_legal = ~masks_np.any(axis=1)
        masks_np[no_legal] = True

        obs_feed = _rms_obs_train(obs_np, obs_rms)
        logits_np, vals_np = policy.infer(obs_feed, masks_np)

        # Gumbel-max trick (vectorisé, zéro loop Python)
        probs    = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
        probs   /= probs.sum(axis=1, keepdims=True)
        u        = np.random.uniform(size=_rng_shape).astype(np.float32)
        acts_np  = (np.log(probs + 1e-20) - np.log(-np.log(u + 1e-20))).argmax(axis=1).astype(np.int64)
        logps_np = np.log(probs[np.arange(n), acts_np] + 1e-10)

        obs2_np, rews_np, dones_np, masks_np = shmem_env.step(acts_np)

        buf_obs[t]  = obs_feed
        buf_act[t]  = acts_np
        buf_rew[t]  = _rms_reward_train(rews_np.astype(np.float32, copy=False), reward_rms)
        buf_logp[t] = logps_np
        buf_done[t] = dones_np
        buf_val[t]  = vals_np
        buf_mask[t] = masks_np

        cur_rew += rews_np.astype(np.float32, copy=False);  cur_len += 1
        done_idx = np.where(dones_np)[0]
        for i in done_idx:
            ep_rewards_all[i].append(float(cur_rew[i]))
            ep_lengths_all[i].append(int(cur_len[i]))
            ep_wins_all[i].append(bool(cur_rew[i] > 0))
            cur_rew[i] = 0.0;  cur_len[i] = 0

        obs_np = obs2_np

    # Bootstrap valeur finale
    obs_last = _rms_obs_train(obs_np, obs_rms)
    _, last_vals = policy.infer(obs_last)   # (n,) float32

    # GAE vectorisé : traiter chaque env indépendamment via numpy (transpose)
    # buf_*  shape : (spe, n, ...) → transposer en (n, spe, ...) pour env-major
    rews_T  = buf_rew.T    # (n, spe)
    vals_T  = buf_val.T    # (n, spe)
    dones_T = buf_done.T   # (n, spe)

    adv_buf, ret_buf = _gae_np_batched(rews_T, vals_T, dones_T, last_vals, gamma, lam)

    # Aplatir (n, spe, dim) → (n×spe, dim) par reshape contiguous (zéro copie)
    total = n * spe
    obs_flat  = buf_obs.transpose(1, 0, 2).reshape(total, -1)  # (n×spe, obs_dim)
    act_flat  = buf_act.T.reshape(total)
    logp_flat = buf_logp.T.reshape(total)
    mask_flat = buf_mask.transpose(1, 0, 2).reshape(total, -1)
    adv_flat  = adv_buf.reshape(total)
    ret_flat  = ret_buf.reshape(total)

    # Construire le batch directement depuis des arrays contiguus (rapide)
    obs_t  = torch.as_tensor(np.ascontiguousarray(obs_flat),  dtype=torch.float32, device=device)
    act_t  = torch.as_tensor(np.ascontiguousarray(act_flat),  dtype=torch.int64,   device=device)
    logp_t = torch.as_tensor(np.ascontiguousarray(logp_flat), dtype=torch.float32, device=device)
    ret_t  = torch.as_tensor(np.ascontiguousarray(ret_flat),  dtype=torch.float32, device=device)
    adv_t  = torch.as_tensor(np.ascontiguousarray(adv_flat),  dtype=torch.float32, device=device)
    adv_t  = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    mask_t = torch.as_tensor(np.ascontiguousarray(mask_flat), dtype=torch.bool,    device=device)
    batch  = RolloutBatch(obs_t, act_t, logp_t, ret_t, adv_t, mask_t)

    all_rews = [r for lst in ep_rewards_all for r in lst]
    all_lens = [l for lst in ep_lengths_all for l in lst]
    all_wins = [w for lst in ep_wins_all    for w in lst]

    stats = EpisodeStats(
        mean_reward = float(np.mean(all_rews)) if all_rews else 0.0,
        mean_length = float(np.mean(all_lens)) if all_lens else float(spe),
        win_rate    = float(np.mean(all_wins)) if all_wins else 0.0,
        n_episodes  = len(all_rews),
    )
    return batch, stats, obs_np, masks_np


# ──────────────────────────────────────────────────────────────────────────────
# Helper interne
# ──────────────────────────────────────────────────────────────────────────────

def _build_batch(
    obs_l:  list,
    act_l:  list,
    logp_l: list,
    ret_l:  list,
    adv_l:  list,
    mask_l: list,
    device: torch.device,
) -> RolloutBatch:
    obs_t  = torch.as_tensor(np.array(obs_l),  dtype=torch.float32, device=device)
    act_t  = torch.as_tensor(act_l,            dtype=torch.int64,   device=device)
    logp_t = torch.as_tensor(logp_l,           dtype=torch.float32, device=device)
    ret_t  = torch.as_tensor(ret_l,            dtype=torch.float32, device=device)
    adv_t  = torch.as_tensor(adv_l,            dtype=torch.float32, device=device)
    mask_t = torch.as_tensor(np.array(mask_l), dtype=torch.bool,    device=device)
    adv_t  = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    return RolloutBatch(obs_t, act_t, logp_t, ret_t, adv_t, mask_t)


# ──────────────────────────────────────────────────────────────────────────────
# Mise à jour PPO
# ──────────────────────────────────────────────────────────────────────────────

def ppo_update(
    net: ActorCritic,
    opt: optim.Optimizer,
    batch: RolloutBatch,
    *,
    clip_eps:   float,
    epochs:     int,
    minibatch:  int,
    ent_coef:   float = 0.01,
    vf_coef:    float = 0.5,
    max_grad:   float = 0.5,
    use_amp:    bool  = False,   # fp16 autocast (MPS / CUDA)
    scaler: "torch.GradScaler | None" = None,
    target_kl: float | None = None,
) -> dict[str, float]:
    """
    Effectue `epochs` passes PPO sur le batch.
    - Value loss clippée (standard implémentation OpenAI)
    - fp16 autocast optionnel pour MPS/CUDA (2-4× speedup)
    Retourne un dict de métriques (loss, pi_loss, v_loss, entropy, clip_frac, kl approx., epochs effectifs).
    """
    n = batch.obs.shape[0]

    # Accumuler les métriques SANS sync CPU↔GPU (float/item) dans la boucle interne.
    # On lit une seule fois à la fin avec torch.stack → 5 syncs au total vs 80+.
    m_loss: list[torch.Tensor] = []
    m_pi:   list[torch.Tensor] = []
    m_v:    list[torch.Tensor] = []
    m_ent:  list[torch.Tensor] = []
    m_clip: list[torch.Tensor] = []
    m_kl:   list[torch.Tensor] = []

    # Valeurs de référence pour le clipping de value
    old_vals = batch.ret - batch.adv  # approximation des values au moment du rollout

    dev = batch.obs.device

    net.train()
    epochs_done = 0
    for _e in range(epochs):
        # Pré-shuffle via torch.randperm sur device → contiguous slices ensuite
        # (évite les random gathers numpy→MPS par mini-batch)
        perm     = torch.randperm(n, device=dev)
        obs_e    = batch.obs[perm]
        act_e    = batch.act[perm]
        logp_e   = batch.logp[perm]
        ret_e    = batch.ret[perm]
        adv_e    = batch.adv[perm]
        mask_e   = batch.mask[perm]
        oldv_e   = old_vals[perm]

        kl_stop = False
        for s in range(0, n, minibatch):
            e        = slice(s, s + minibatch)
            obs      = obs_e[e]
            act      = act_e[e]
            old_logp = logp_e[e]
            ret      = ret_e[e]
            adv      = adv_e[e]
            m        = mask_e[e]
            old_v    = oldv_e[e]

            if use_amp:
                with torch.autocast(device_type=dev.type, dtype=torch.float16):
                    logits, val = net(obs, m)
            else:
                logits, val = net(obs, m)

            dist  = Categorical(logits=logits.float())
            logp  = dist.log_prob(act)
            ratio = torch.exp(logp - old_logp.float())

            # KL approximée (moyenne sur le batch) — critère d’arrêt style SB3 / OpenAI
            with torch.no_grad():
                approx_kl = (old_logp.float() - logp.float()).mean()

            # Clipped policy objective
            surr1   = ratio * adv
            surr2   = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            pi_loss = -torch.min(surr1, surr2).mean()

            # Value loss clippée (PPO2 OpenAI)
            val_clipped = old_v + torch.clamp(val.float() - old_v, -clip_eps * 10, clip_eps * 10)
            v_loss = 0.5 * torch.max(
                (ret - val.float()).pow(2),
                (ret - val_clipped).pow(2),
            ).mean()

            # Entropy bonus
            ent  = dist.entropy().mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent

            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(net.parameters(), max_grad)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad)
                opt.step()

            # Accumuler comme tenseurs GPU — zéro sync ici
            with torch.no_grad():
                m_loss.append(loss.detach())
                m_pi.append(pi_loss.detach())
                m_v.append(v_loss.detach())
                m_ent.append(ent.detach())
                m_clip.append(((ratio.detach() - 1.0).abs() > clip_eps).float().mean())
                m_kl.append(approx_kl.detach())
            if target_kl is not None and float(approx_kl) > target_kl:
                kl_stop = True
                break
        epochs_done = _e + 1
        if kl_stop:
            break

    # Une seule lecture CPU (5 syncs au lieu de 80+)
    return {
        "loss": float(torch.stack(m_loss).mean()),
        "pi":   float(torch.stack(m_pi).mean()),
        "v":    float(torch.stack(m_v).mean()),
        "ent":  float(torch.stack(m_ent).mean()),
        "clip": float(torch.stack(m_clip).mean()),
        "kl":   float(torch.stack(m_kl).mean()) if m_kl else 0.0,
        "ppo_epochs_done": float(epochs_done),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Boucle d'entraînement principale
# ──────────────────────────────────────────────────────────────────────────────

def train_loop(
    env,
    cfg: dict,
    *,
    device: torch.device | None = None,
    vec_env=None,
    coreml_policy=None,   # CoreMLPolicy optionnel — inférence sur ANE
    on_update_callback=None,
    stop_event=None,
    train_until_stop: bool = False,
    net: ActorCritic | None = None,
    obs_rms: RunningMeanStd | None = None,
    reward_rms: RunningMeanStd | None = None,
) -> ActorCritic:
    """
    Boucle PPO.

    cfg (training section de config.yaml) :
      total_steps, rollout_len, gamma, lam, lr, clip_eps, epochs, minibatch,
      obs_dim, hidden, ent_coef_start, ent_coef_end, seed,
      coreml_sync_every (optionnel, défaut 4) : n’exporter Core ML qu’une fois sur N updates.

    vec_env  : SubprocVecEnv optionnel — si fourni, collecte vectorisée (plus rapide).
    net      : réseau déjà instancié (ex. poids chargés depuis un checkpoint) ; sinon créé ici.
    obs_rms    : ``RunningMeanStd`` optionnel ; si None et ``cfg['obs_normalize']``, créé automatiquement.
    reward_rms : idem pour ``cfg['reward_normalize']`` (échelle σ des rewards vers le GAE).
    on_update_callback(update_idx, net, stats, metrics) → checkpoint / vidéo.
    stop_event : threading.Event optionnel — si défini et set, sortie propre de la boucle.
    train_until_stop : si True, ``updates`` devient un plafond très haut ; la boucle s’arrête
        surtout via ``stop_event`` (ex. stdin « stop »). LR / entropy utilisent des
        références fixes (cfg ``until_stop_warmup_updates``, ``ent_anneal_updates``).
    """
    import math as _math

    device = device or torch.device("cpu")
    dev_type = device.type  # "mps", "cuda", or "cpu"

    obs_dim  = int(cfg.get("obs_dim",      96))
    hidden   = int(cfg.get("hidden",       256))
    n_res    = int(cfg.get("n_res",          2))   # blocs résiduels
    if vec_env is not None:
        n_act = int(vec_env.action_space.n)
    else:
        if env is None:
            raise ValueError("train_loop : fournir ``env`` lorsque ``vec_env`` est None.")
        n_act = int(env.action_space.n)
    if net is None:
        net = ActorCritic(obs_dim, n_act, hidden, n_res).to(device)
    else:
        net = net.to(device)

    if obs_rms is None and bool(cfg.get("obs_normalize", False)):
        obs_rms = RunningMeanStd(
            (obs_dim,),
            epsilon=float(cfg.get("obs_normalize_epsilon", 1e-4)),
            clip_obs=float(cfg.get("obs_normalize_clip", 10.0)),
        )
    if obs_rms is not None:
        print(
            f"Observations : running mean/std (ε={obs_rms.epsilon}, clip={obs_rms.clip_obs})",
            flush=True,
        )

    if reward_rms is None and bool(cfg.get("reward_normalize", False)):
        reward_rms = RunningMeanStd(
            (1,),
            epsilon=float(cfg.get("reward_normalize_epsilon", 1e-4)),
            clip_obs=0.0,
        )
    if reward_rms is not None:
        print(
            f"Rewards      : échelle running σ (ε={reward_rms.epsilon})",
            flush=True,
        )

    # AdamW (fused CUDA / foreach CPU-MPS si disponible)
    lr          = float(cfg.get("lr",           3e-4))
    weight_decay= float(cfg.get("weight_decay", 1e-4))
    opt = _adamw_for_device(net.parameters(), lr, weight_decay, device)

    # torch.compile : défaut off (souvent contre-productif sur MPS / petits réseaux)
    use_compile = bool(cfg.get("torch_compile", False))
    if use_compile and hasattr(torch, "compile"):
        try:
            net = torch.compile(net, mode="reduce-overhead")
            print("torch.compile activé (reduce-overhead)")
        except Exception as e:
            print(f"torch.compile indisponible : {e}")

    # fp16 autocast — disponible sur CUDA et MPS (PyTorch ≥ 2.2)
    use_amp = bool(cfg.get("use_amp", True)) and dev_type in ("cuda", "mps")
    scaler  = None
    if use_amp and dev_type == "cuda":
        scaler = torch.GradScaler()   # GradScaler utile seulement sur CUDA
    elif use_amp and dev_type == "mps":
        pass  # autocast MPS sans GradScaler (pas besoin)
    print(f"fp16 autocast : {'activé' if use_amp else 'désactivé'} ({dev_type})")

    total    = int(cfg.get("total_steps",  50_000))
    roll     = int(cfg.get("rollout_len",  512))
    gamma    = float(cfg.get("gamma",      0.99))
    lam      = float(cfg.get("lam",        0.95))
    clip_eps = float(cfg.get("clip_eps",   0.2))
    clip_eps_end = float(cfg.get("clip_eps_end", clip_eps))
    target_kl_cfg = cfg.get("target_kl", None)
    target_kl = float(target_kl_cfg) if target_kl_cfg is not None and float(target_kl_cfg) > 0 else None
    epochs   = int(cfg.get("epochs",       6))
    mini     = int(cfg.get("minibatch",    128))

    ent_start  = float(cfg.get("ent_coef_start", 0.05))
    ent_end    = float(cfg.get("ent_coef_end",   0.005))
    warmup_frac= float(cfg.get("warmup_frac",    0.03))  # 3% du training en warmup LR

    # Mode vectorisé : steps_per_env × n_envs = roll total par update
    n_envs      = vec_env.n_envs if vec_env is not None else 1
    steps_p_env = max(1, roll // n_envs)
    total_per_update = steps_p_env * n_envs

    train_until_stop = bool(train_until_stop)
    ent_ref = max(1, int(cfg.get("ent_anneal_updates", 20_000)))
    if train_until_stop:
        updates = int(cfg.get("train_until_stop_max_updates", 1_000_000_000))
        warmup_iters = max(1, int(cfg.get("until_stop_warmup_updates", 400)))
        log_every = max(1, int(cfg.get("until_stop_log_every_updates", 20)))
    else:
        updates = max(1, total // total_per_update)
        warmup_iters = max(1, int(updates * warmup_frac))
        # Sans plafond, total_steps énorme → log_every ≈ updates/40 (ex. 30k+) : métriques « figées ».
        _sparse = max(1, updates // 40)
        _cap = max(1, int(cfg.get("metrics_log_every_updates_max", 100)))
        log_every = max(1, min(_sparse, _cap))
    # Core ML : re-export coûteux (MIL/tqdm) — ne pas le lancer à chaque update si N>1
    coreml_sync_every = max(1, int(cfg.get("coreml_sync_every", 4)))

    _last_obs:   "np.ndarray | None" = None   # obs/masks réutilisés entre rollouts
    _last_masks: "np.ndarray | None" = None   # (évite reset_all à chaque update)

    _last_log_steps = 0
    _last_log_t = time.perf_counter()
    _train_wall_start = time.time()

    it = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            print("\n>>> Arrêt : « stop » (ou fichier STOP) détecté.", flush=True)
            break
        if it >= updates:
            break

        if train_until_stop:
            if it < warmup_iters:
                lr_mult = (it + 1) / warmup_iters
            else:
                lr_mult = 1.0
            frac = min(1.0, it / float(ent_ref))
        else:
            frac = it / max(1, updates - 1)
            if it < warmup_iters:
                lr_mult = (it + 1) / warmup_iters
            else:
                progress = (it - warmup_iters) / max(1, updates - warmup_iters)
                lr_mult = 0.5 * (1.0 + _math.cos(_math.pi * progress))
        for pg in opt.param_groups:
            pg["lr"] = lr * lr_mult

        # ── Entropy : décroissance linéaire ────────────────────────────────
        ent_coef = ent_start + frac * (ent_end - ent_start)

        clip_now = clip_eps + frac * (clip_eps_end - clip_eps)

        if coreml_policy is not None and vec_env is not None:
            batch, stats, _last_obs, _last_masks = collect_rollout_coreml(
                vec_env, coreml_policy, net, steps_p_env, gamma, lam, device,
                init_obs=_last_obs, init_masks=_last_masks,
                obs_rms=obs_rms,
                reward_rms=reward_rms,
            )
        elif vec_env is not None:
            batch, stats = collect_rollout_vec(
                vec_env, net, steps_p_env, gamma, lam, device,
                obs_rms=obs_rms, reward_rms=reward_rms,
            )
        else:
            batch, stats = collect_rollout(
                env, net, roll, gamma, lam, device,
                obs_rms=obs_rms, reward_rms=reward_rms,
            )

        metrics = ppo_update(
            net, opt, batch,
            clip_eps=clip_now,
            epochs=epochs,
            minibatch=mini,
            ent_coef=ent_coef,
            use_amp=use_amp,
            scaler=scaler,
            target_kl=target_kl,
        )

        # Re-synchro Core ML en arrière-plan (non-bloquant), throttling N updates
        # (spam MIL + charge CPU ; rollout avec politique légèrement « stale » — OK en PPO)
        if coreml_policy is not None:
            last_up = (not train_until_stop and it >= updates - 1) or (
                stop_event is not None and stop_event.is_set()
            )
            if coreml_sync_every <= 1 or (it + 1) % coreml_sync_every == 0 or last_up:
                coreml_policy.sync_async(net)

        if on_update_callback is not None:
            on_update_callback(it, net, stats, metrics)

        if (it + 1) % log_every == 0 or it == 0:
            steps_done = (it + 1) * total_per_update
            cur_lr = opt.param_groups[0]["lr"]
            win = stats.win_rate * 100
            rwd = stats.mean_reward
            llen = stats.mean_length
            ep = stats.n_episodes
            now_m = time.perf_counter()
            dsteps = int(steps_done - _last_log_steps)
            dt_s = max(1e-6, now_m - _last_log_t)
            stp_s = float(dsteps / dt_s)
            _last_log_steps = int(steps_done)
            _last_log_t = now_m
            wall_unix = time.time()
            prog = f"{steps_done:>10}" + ("" if train_until_stop else f" / {total}")
            upprog = f"upd {it + 1:>6}" + ("" if train_until_stop else f"/{updates}")
            _elapsed_wall = time.time() - _train_wall_start
            print(
                f"[{prog}] {upprog}  "
                f"rew={rwd:+.2f}  len={llen:.0f}  "
                f"win={win:.0f}%  ep={ep}  "
                f"loss={metrics['loss']:.4f}  "
                f"ent={metrics['ent']:.3f}  "
                f"clip={metrics['clip']:.2f}  "
                f"kl={metrics['kl']:.4f}  "
                f"lr={cur_lr:.2e}  "
                f"stp/s≈{stp_s:.0f}  "
                f"t={format_wall_elapsed(_elapsed_wall)}",
                flush=True,
            )
            mpath = cfg.get("metrics_jsonl")
            if mpath:
                p = Path(str(mpath)).expanduser()
                if not p.is_absolute():
                    # Relatif : racine du dépôt (``opctcg_text_sim/`` parent), pas le cwd.
                    _root = Path(__file__).resolve().parents[1]
                    p = (_root / p).resolve()
                p.parent.mkdir(parents=True, exist_ok=True)
                row = {
                    "t": float(wall_unix),
                    "elapsed_wall_s": float(_elapsed_wall),
                    "steps": int(steps_done),
                    "update": int(it + 1),
                    "updates_total": int(updates) if not train_until_stop else None,
                    "stp_s": float(stp_s),
                    "rew": float(rwd),
                    "len": float(llen),
                    "win_pct": float(win),
                    "episodes": int(ep),
                    "loss": float(metrics["loss"]),
                    "ent": float(metrics["ent"]),
                    "clip": float(metrics["clip"]),
                    "kl": float(metrics["kl"]),
                    "ppo_epochs_done": float(metrics.get("ppo_epochs_done", epochs)),
                    "lr": float(cur_lr),
                }
                with p.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        it += 1

    # Dernier export Core ML + attente du thread (aligne ANE sur les poids PyTorch finaux)
    if coreml_policy is not None:
        coreml_policy.sync_async(net)
        coreml_policy.wait_sync()

    if obs_rms is not None:
        net.obs_rms = obs_rms
    if reward_rms is not None:
        net.reward_rms = reward_rms
    return net
