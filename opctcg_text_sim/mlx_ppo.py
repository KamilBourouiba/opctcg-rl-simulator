"""
PPO sur Apple Silicon via **MLX** (MLX-LM stack : graphes lazy + Metal).

- Inférence / optimiseur entièrement sur le backend MLX (mémoire unifiée avec les buffers numpy du simulateur).
- Architecture alignée sur ``ActorCritic`` (PyTorch) : projection → blocs résiduels → têtes π / V.
- Pas de Core ML / pas de MPS : une seule pile pour le rollout + la mise à jour.

Checkpoint : dict pickle (``.pt`` ou ``.pkl``) pour ``obs_rms`` / ``reward_rms`` ; poids sous la clé
``mlx_flat`` (nom → ndarray float32). Les anciens fichiers ``torch.save`` restent chargeables si PyTorch est installé.
"""
from __future__ import annotations

import io
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.optimizers import AdamW
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from .obs_rms import RunningMeanStd
from .ppo import (
    EpisodeStats,
    _gae_np_batched,
    _rms_obs_train,
    _rms_reward_train,
    format_wall_elapsed,
)

_NEG_INF = -1e9


def _orthogonal_init(w: mx.array, gain: float = 1.0) -> None:
    import mlx.nn.init as mlx_init

    mlx_init.orthogonal(gain=gain)(w)


def _zeros_init(w: mx.array) -> None:
    import mlx.nn.init as mlx_init

    mlx_init.constant(0.0)(w)


class _ResBlockMLX(nn.Module):
    """Linear → LN → LeakyReLU → Linear → LN + résidu."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        for lin in (self.lin1, self.lin2):
            _orthogonal_init(lin.weight, gain=math.sqrt(2))
            _zeros_init(lin.bias)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.lin1(x)
        y = self.ln1(y)
        y = nn.leaky_relu(y, negative_slope=0.1)
        y = self.lin2(y)
        y = self.ln2(y)
        return nn.leaky_relu(x + y, negative_slope=0.1)


class ActorCriticMLX(nn.Module):
    """Même structure que ``ppo.ActorCritic`` (torch)."""

    def __init__(self, obs_dim: int, n_act: int, hidden: int = 256, n_res: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_act = n_act
        self.hidden = hidden

        self.proj = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1),
        )
        self.blocks = nn.Sequential(*[_ResBlockMLX(hidden) for _ in range(n_res)])
        self.pi_head = nn.Linear(hidden, n_act)
        self.v_head = nn.Linear(hidden, 1)

        for ly in (*self.proj.layers, *self.blocks.layers):
            if isinstance(ly, nn.Linear):
                _orthogonal_init(ly.weight, gain=math.sqrt(2))
                _zeros_init(ly.bias)
        _orthogonal_init(self.pi_head.weight, gain=0.01)
        _zeros_init(self.pi_head.bias)
        _zeros_init(self.v_head.bias)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        h = self.blocks(self.proj(x))
        logits = self.pi_head(h)
        if mask is not None:
            neg = mx.full(logits.shape, _NEG_INF, dtype=logits.dtype)
            logits = mx.where(mask, logits, neg)
        values = self.v_head(h).squeeze(-1)
        return logits, values


def _categorical_log_prob(logits: mx.array, actions: mx.array) -> mx.array:
    log_sm = nn.log_softmax(logits, axis=-1)
    idx = mx.expand_dims(actions.astype(mx.int32), axis=-1)
    lp = mx.take_along_axis(log_sm, idx, axis=-1).squeeze(-1)
    return lp


def _categorical_entropy(logits: mx.array) -> mx.array:
    log_sm = nn.log_softmax(logits, axis=-1)
    p = mx.softmax(logits, axis=-1)
    return -mx.sum(p * log_sm, axis=-1)


def _sample_actions(logits: mx.array) -> mx.array:
    """Échantillon sans remplacement catégorique (une ligne par batch)."""
    return mx.random.categorical(logits).astype(mx.int32)


def _clip_grad_global(grads: Any, max_norm: float) -> Any:
    flat = tree_flatten(grads)
    sq = mx.array(0.0, dtype=mx.float32)
    for _, g in flat:
        sq = sq + mx.sum(g.astype(mx.float32) * g.astype(mx.float32))
    norm_f = float(mx.sqrt(sq)) + 1e-8
    coef = min(1.0, max_norm / norm_f)
    return tree_map(lambda g: g * coef, grads)


@dataclass
class MLXRolloutBatch:
    obs: mx.array
    act: mx.array
    logp: mx.array
    ret: mx.array
    adv: mx.array
    mask: mx.array


def collect_rollout_mlx(
    env,
    net: ActorCriticMLX,
    steps: int,
    gamma: float,
    lam: float,
    obs_rms: RunningMeanStd | None = None,
    reward_rms: RunningMeanStd | None = None,
) -> tuple[MLXRolloutBatch, EpisodeStats]:
    """Un seul environnement — MLX pour π(a|s), numpy pour le simulateur."""
    obs_l: list[np.ndarray] = []
    act_l: list[int] = []
    rew_l: list[float] = []
    logp_l: list[float] = []
    done_l: list[bool] = []
    val_l: list[float] = []
    mask_l: list[np.ndarray] = []

    ep_rewards: list[float] = []
    ep_lengths: list[int] = []
    ep_wins: list[bool] = []
    cur_reward = 0.0
    cur_len = 0
    n_act = env.action_space.n

    o, _ = env.reset()

    for _ in range(steps):
        has_mask = hasattr(env, "legal_actions_mask")
        m_np = env.legal_actions_mask() if has_mask else np.ones(n_act, dtype=bool)
        if not m_np.any():
            m_np[:] = True

        o_feed = _rms_obs_train(o, obs_rms)
        x = mx.array(np.asarray(o_feed, dtype=np.float32)[None, :])
        m_mx = mx.array(np.asarray(m_np, dtype=bool)[None, :])

        logits, val = net(x, m_mx)
        acts = _sample_actions(logits)
        lp = _categorical_log_prob(logits, acts)
        mx.eval(logits, val, lp, acts)

        a_int = int(np.asarray(acts).squeeze())
        logp_l.append(float(np.asarray(lp).squeeze()))
        val_l.append(float(np.asarray(val).squeeze()))

        o2, r, terminated, truncated, info = env.step(a_int)
        done = terminated or truncated
        r_raw = float(r)
        r_train = float(
            _rms_reward_train(np.array([r_raw], dtype=np.float32), reward_rms)[0]
        )

        obs_l.append(np.copy(o_feed))
        act_l.append(a_int)
        rew_l.append(r_train)
        done_l.append(done)
        mask_l.append(np.copy(m_np))

        cur_reward += r_raw
        cur_len += 1
        o = o2

        if done:
            ep_rewards.append(cur_reward)
            ep_lengths.append(cur_len)
            ep_wins.append(cur_reward > 0.0)
            cur_reward = 0.0
            cur_len = 0
            o, _ = env.reset()

    o_last = _rms_obs_train(o, obs_rms)
    x_last = mx.array(np.asarray(o_last, dtype=np.float32)[None, :])
    _, lv = net(x_last)
    mx.eval(lv)
    last_val = float(np.asarray(lv).squeeze()) if not done_l[-1] else 0.0

    from .ppo import _gae

    adv_l, ret_l = _gae(rew_l, val_l, done_l, last_val, gamma, lam)

    n = len(obs_l)
    obs_mx = mx.array(np.stack(obs_l, axis=0).astype(np.float32))
    act_mx = mx.array(np.asarray(act_l, dtype=np.int32))
    logp_mx = mx.array(np.asarray(logp_l, dtype=np.float32))
    ret_mx = mx.array(np.asarray(ret_l, dtype=np.float32))
    adv_mx = mx.array(np.asarray(adv_l, dtype=np.float32))
    adv_mx = (adv_mx - mx.mean(adv_mx)) / (mx.std(adv_mx) + 1e-8)
    mask_mx = mx.array(np.stack(mask_l, axis=0))

    batch = MLXRolloutBatch(
        obs=obs_mx,
        act=act_mx,
        logp=logp_mx,
        ret=ret_mx,
        adv=adv_mx,
        mask=mask_mx,
    )
    stats = EpisodeStats(
        mean_reward=float(np.mean(ep_rewards)) if ep_rewards else 0.0,
        mean_length=float(np.mean(ep_lengths)) if ep_lengths else float(steps),
        win_rate=float(np.mean(ep_wins)) if ep_wins else 0.0,
        n_episodes=len(ep_rewards),
    )
    return batch, stats


def collect_rollout_vec_mlx(
    vec_env,
    net: ActorCriticMLX,
    steps_per_env: int,
    gamma: float,
    lam: float,
    obs_rms: RunningMeanStd | None = None,
    reward_rms: RunningMeanStd | None = None,
) -> tuple[MLXRolloutBatch, EpisodeStats]:
    """Vectorisé — batch MLX ``(n_envs,)`` à chaque pas (Metal-friendly)."""
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

    for t in range(spe):
        no_legal = ~masks_np.any(axis=1)
        masks_np[no_legal] = True

        obs_feed = _rms_obs_train(obs_np, obs_rms)
        x = mx.array(np.ascontiguousarray(obs_feed, dtype=np.float32))
        m_mx = mx.array(np.ascontiguousarray(masks_np, dtype=bool))

        logits, vals = net(x, m_mx)
        acts = _sample_actions(logits)
        logps = _categorical_log_prob(logits, acts)

        mx.eval(logits, vals, acts, logps)

        buf_logp[t] = np.asarray(logps)
        buf_val[t] = np.asarray(vals)
        acts_np = np.asarray(acts).astype(np.int64, copy=False)

        step_out = vec_env.step(acts_np)
        if len(step_out) == 5:
            obs2_np, rews_np, dones_np, _infos, masks_np = step_out
        elif len(step_out) == 4:
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
        buf_act[t] = acts_np
        buf_rew[t] = _rms_reward_train(
            rews_np.astype(np.float32, copy=False), reward_rms
        )
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
    x_last = mx.array(np.ascontiguousarray(obs_last, dtype=np.float32))
    _, lv = net(x_last)
    mx.eval(lv)
    last_vals = np.asarray(lv)

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

    obs_mx = mx.array(np.ascontiguousarray(obs_flat, dtype=np.float32))
    act_mx = mx.array(np.ascontiguousarray(act_flat, dtype=np.int32))
    logp_mx = mx.array(np.ascontiguousarray(logp_flat, dtype=np.float32))
    ret_mx = mx.array(np.ascontiguousarray(ret_flat, dtype=np.float32))
    adv_mx = mx.array(np.ascontiguousarray(adv_flat, dtype=np.float32))
    adv_mx = (adv_mx - mx.mean(adv_mx)) / (mx.std(adv_mx) + 1e-8)
    mask_mx = mx.array(np.ascontiguousarray(mask_flat, dtype=bool))

    batch = MLXRolloutBatch(
        obs=obs_mx,
        act=act_mx,
        logp=logp_mx,
        ret=ret_mx,
        adv=adv_mx,
        mask=mask_mx,
    )

    all_rews = [r for lst in ep_rewards_all for r in lst]
    all_lens = [l for lst in ep_lengths_all for l in lst]
    all_wins = [w for lst in ep_wins_all for w in lst]
    stats = EpisodeStats(
        mean_reward=float(np.mean(all_rews)) if all_rews else 0.0,
        mean_length=float(np.mean(all_lens)) if all_lens else float(spe),
        win_rate=float(np.mean(all_wins)) if all_wins else 0.0,
        n_episodes=len(all_rews),
    )
    return batch, stats


def ppo_update_mlx(
    net: ActorCriticMLX,
    optimizer: AdamW,
    batch: MLXRolloutBatch,
    *,
    clip_eps: float,
    epochs: int,
    minibatch: int,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad: float = 0.5,
    target_kl: float | None = None,
) -> dict[str, float]:
    """PPO — optimiseur MLX ; une sous-passe par mini-batch (compile lazy stable)."""
    n = int(batch.obs.shape[0])
    old_vals = batch.ret - batch.adv

    losses: list[float] = []
    pi_losses: list[float] = []
    v_losses: list[float] = []
    ents: list[float] = []
    clips: list[float] = []
    kls: list[float] = []

    epochs_done = 0
    for _e in range(epochs):
        perm = mx.random.permutation(mx.arange(n, dtype=mx.int32))
        obs_e = batch.obs[perm]
        act_e = batch.act[perm]
        logp_e = batch.logp[perm]
        ret_e = batch.ret[perm]
        adv_e = batch.adv[perm]
        mask_e = batch.mask[perm]
        oldv_e = old_vals[perm]

        kl_stop = False
        for s in range(0, n, minibatch):
            sl = slice(s, min(s + minibatch, n))
            diag: dict[str, mx.array] = {}

            def loss_fn(model: ActorCriticMLX) -> mx.array:
                logits, val = model(obs_e[sl], mask_e[sl])
                logp = _categorical_log_prob(logits, act_e[sl])
                ratio = mx.exp(logp - logp_e[sl])

                diag["kl"] = mx.mean(logp_e[sl] - logp)
                surr1 = ratio * adv_e[sl]
                surr2 = mx.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_e[sl]
                pi_loss = -mx.mean(mx.minimum(surr1, surr2))

                val_clipped = oldv_e[sl] + mx.clip(
                    val - oldv_e[sl], -clip_eps * 10, clip_eps * 10
                )
                v_loss = 0.5 * mx.mean(
                    mx.maximum(
                        mx.square(ret_e[sl] - val),
                        mx.square(ret_e[sl] - val_clipped),
                    )
                )

                ent = mx.mean(_categorical_entropy(logits))
                diag["pi"] = pi_loss
                diag["v"] = v_loss
                diag["ent"] = ent
                diag["clip"] = mx.mean(
                    (mx.abs(ratio - 1.0) > clip_eps).astype(mx.float32)
                )
                return pi_loss + vf_coef * v_loss - ent_coef * ent

            loss_and_grad = nn.value_and_grad(net, loss_fn)
            loss, grads = loss_and_grad(net)
            grads = _clip_grad_global(grads, max_grad)
            optimizer.update(net, grads)
            mx.eval(net.parameters(), optimizer.state, loss, *diag.values())

            losses.append(float(np.asarray(loss)))
            approx_kl_f = float(np.asarray(diag["kl"]))
            pi_losses.append(float(np.asarray(diag["pi"])))
            v_losses.append(float(np.asarray(diag["v"])))
            ents.append(float(np.asarray(diag["ent"])))
            clips.append(float(np.asarray(diag["clip"])))
            kls.append(approx_kl_f)

            if target_kl is not None and approx_kl_f > target_kl:
                kl_stop = True
                break

        epochs_done = _e + 1
        if kl_stop:
            break

    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "pi": float(np.mean(pi_losses)) if pi_losses else 0.0,
        "v": float(np.mean(v_losses)) if v_losses else 0.0,
        "ent": float(np.mean(ents)) if ents else 0.0,
        "clip": float(np.mean(clips)) if clips else 0.0,
        "kl": float(np.mean(kls)) if kls else 0.0,
        "ppo_epochs_done": float(epochs_done),
    }


def mlx_checkpoint_dict(
    net: ActorCriticMLX,
    *,
    obs_rms: RunningMeanStd | None = None,
    reward_rms: RunningMeanStd | None = None,
) -> dict[str, Any]:
    flat = tree_flatten(net.parameters())
    mlx_flat = {k: np.asarray(v, dtype=np.float32) for k, v in flat}
    out: dict[str, Any] = {"mlx_flat": mlx_flat}
    if obs_rms is not None:
        out["obs_rms"] = obs_rms.state_dict()
    if reward_rms is not None:
        out["reward_rms"] = reward_rms.state_dict()
    return out


def save_mlx_checkpoint(path: Path | str, payload: dict[str, Any]) -> None:
    """Écrit un checkpoint MLX sans PyTorch (pickle pur)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_checkpoint_payload(path: Path | str) -> Any:
    """Charge pickle natif ; sinon ``torch.load`` si PyTorch est disponible (anciens exports)."""
    path = Path(path)
    raw = path.read_bytes()
    bio = io.BytesIO(raw)
    try:
        return pickle.load(bio)
    except Exception:
        pass
    try:
        import torch

        bio.seek(0)
        return torch.load(bio, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(
            f"Checkpoint MLX illisible : {path} (pickle attendu, ou torch.save avec PyTorch installé)."
        ) from exc


def load_mlx_checkpoint(
    path: Path | str,
    net: ActorCriticMLX,
    *,
    obs_rms: RunningMeanStd | None = None,
    reward_rms: RunningMeanStd | None = None,
    data: Any | None = None,
) -> None:
    if data is None:
        data = _load_checkpoint_payload(path)
    if isinstance(data, dict) and "mlx_flat" in data:
        flat_list = [(k, mx.array(v)) for k, v in data["mlx_flat"].items()]
        net.update(tree_unflatten(flat_list))
        mx.eval(net.parameters())
        if obs_rms is not None and data.get("obs_rms") is not None:
            obs_rms.load_state_dict(data["obs_rms"])
        if reward_rms is not None and data.get("reward_rms") is not None:
            reward_rms.load_state_dict(data["reward_rms"])
        return
    raise ValueError(
        "Checkpoint MLX attendu : dict avec clé 'mlx_flat' (weights nommés)."
    )


def train_loop_mlx(
    env,
    cfg: dict,
    *,
    vec_env=None,
    net: ActorCriticMLX | None = None,
    obs_rms: RunningMeanStd | None = None,
    reward_rms: RunningMeanStd | None = None,
    on_update_callback=None,
    stop_event=None,
    train_until_stop: bool = False,
) -> ActorCriticMLX:
    """Boucle PPO MLX — mêmes hyperparamètres clés que ``ppo.train_loop``."""
    obs_dim = int(cfg.get("obs_dim", 96))
    hidden = int(cfg.get("hidden", 256))
    n_res = int(cfg.get("n_res", 2))
    if vec_env is not None:
        n_act = int(vec_env.action_space.n)
    else:
        if env is None:
            raise ValueError("train_loop_mlx : fournir env ou vec_env.")
        n_act = int(env.action_space.n)

    if net is None:
        net = ActorCriticMLX(obs_dim, n_act, hidden, n_res)
    mx.eval(net.parameters())

    if obs_rms is None and bool(cfg.get("obs_normalize", False)):
        obs_rms = RunningMeanStd(
            (obs_dim,),
            epsilon=float(cfg.get("obs_normalize_epsilon", 1e-4)),
            clip_obs=float(cfg.get("obs_normalize_clip", 10.0)),
        )
    if reward_rms is None and bool(cfg.get("reward_normalize", False)):
        reward_rms = RunningMeanStd(
            (1,),
            epsilon=float(cfg.get("reward_normalize_epsilon", 1e-4)),
            clip_obs=0.0,
        )

    lr = float(cfg.get("lr", 3e-4))
    wd = float(cfg.get("weight_decay", 1e-4))
    optimizer = AdamW(learning_rate=lr, weight_decay=wd)

    total = int(cfg.get("total_steps", 50_000))
    roll = int(cfg.get("rollout_len", 512))
    gamma = float(cfg.get("gamma", 0.99))
    lam = float(cfg.get("lam", 0.95))
    clip_eps = float(cfg.get("clip_eps", 0.2))
    clip_eps_end = float(cfg.get("clip_eps_end", clip_eps))
    target_kl_cfg = cfg.get("target_kl", None)
    target_kl = (
        float(target_kl_cfg)
        if target_kl_cfg is not None and float(target_kl_cfg) > 0
        else None
    )
    epochs = int(cfg.get("epochs", 6))
    mini = int(cfg.get("minibatch", 128))

    ent_start = float(cfg.get("ent_coef_start", 0.05))
    ent_end = float(cfg.get("ent_coef_end", 0.005))
    warmup_frac = float(cfg.get("warmup_frac", 0.03))

    n_envs = vec_env.n_envs if vec_env is not None else 1
    steps_p_env = max(1, roll // max(1, n_envs))
    total_per_update = steps_p_env * max(1, n_envs)

    train_until_stop = bool(train_until_stop)
    ent_ref = max(1, int(cfg.get("ent_anneal_updates", 20_000)))
    if train_until_stop:
        updates = int(cfg.get("train_until_stop_max_updates", 1_000_000_000))
        warmup_iters = max(1, int(cfg.get("until_stop_warmup_updates", 400)))
        log_every = max(1, int(cfg.get("until_stop_log_every_updates", 20)))
    else:
        updates = max(1, total // total_per_update)
        warmup_iters = max(1, int(updates * warmup_frac))
        _sparse = max(1, updates // 40)
        _cap = max(1, int(cfg.get("metrics_log_every_updates_max", 100)))
        log_every = max(1, min(_sparse, _cap))

    _last_log_steps = 0
    _last_log_t = time.perf_counter()
    _train_wall_start = time.time()

    it = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            print("\n>>> Arrêt demandé (MLX).", flush=True)
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
                lr_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
        optimizer.learning_rate = lr * lr_mult

        ent_coef = ent_start + frac * (ent_end - ent_start)
        clip_now = clip_eps + frac * (clip_eps_end - clip_eps)

        if vec_env is not None:
            batch, stats = collect_rollout_vec_mlx(
                vec_env,
                net,
                steps_p_env,
                gamma,
                lam,
                obs_rms=obs_rms,
                reward_rms=reward_rms,
            )
        else:
            batch, stats = collect_rollout_mlx(
                env,
                net,
                roll,
                gamma,
                lam,
                obs_rms=obs_rms,
                reward_rms=reward_rms,
            )

        metrics = ppo_update_mlx(
            net,
            optimizer,
            batch,
            clip_eps=clip_now,
            epochs=epochs,
            minibatch=mini,
            ent_coef=ent_coef,
            target_kl=target_kl,
        )

        if on_update_callback is not None:
            on_update_callback(it, net, stats, metrics)

        if (it + 1) % log_every == 0 or it == 0:
            steps_done = (it + 1) * total_per_update
            cur_lr = optimizer.learning_rate
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
                f"[MLX {prog}] {upprog}  "
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
                    "backend": "mlx",
                }
                with p.open("a", encoding="utf-8") as fp:
                    fp.write(json.dumps(row, ensure_ascii=False) + "\n")

        it += 1

    return net
