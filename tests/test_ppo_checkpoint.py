"""Tests légers : checkpoints PPO + RunningMeanStd."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from opctcg_text_sim.checkpoint_io import checkpoint_dict, load_policy_checkpoint, torch_load_train_checkpoint
from opctcg_text_sim.obs_rms import RunningMeanStd
from opctcg_text_sim.ppo import ActorCritic, _rms_reward_train


def test_checkpoint_roundtrip_obs_reward_rms():
    obs_dim, n_act, hidden, n_res = 8, 3, 16, 1
    net = ActorCritic(obs_dim, n_act, hidden, n_res)
    obs_rms = RunningMeanStd((obs_dim,), clip_obs=5.0)
    reward_rms = RunningMeanStd((1,), epsilon=1e-4, clip_obs=0.0)
    obs_rms.update(np.random.randn(4, obs_dim).astype(np.float64))
    reward_rms.update(np.random.randn(5, 1).astype(np.float64))

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ck.pt"
        torch.save(checkpoint_dict(net, obs_rms=obs_rms, reward_rms=reward_rms), p)

        net2 = ActorCritic(obs_dim, n_act, hidden, n_res)
        o2 = RunningMeanStd((obs_dim,), clip_obs=5.0)
        r2 = RunningMeanStd((1,), epsilon=1e-4, clip_obs=0.0)
        load_policy_checkpoint(p, net2, torch.device("cpu"), obs_rms=o2, reward_rms=r2)

        for a, b in zip(net.parameters(), net2.parameters(), strict=True):
            assert torch.allclose(a, b)
        assert np.allclose(obs_rms.mean, o2.mean)
        assert np.allclose(obs_rms.var, o2.var)
        assert np.allclose(reward_rms.mean, r2.mean)


def test_torch_load_weights_only_kw():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = Path(f.name)
    try:
        torch.save({"x": 1}, path)
        d = torch_load_train_checkpoint(path, "cpu")
        assert d["x"] == 1
    finally:
        path.unlink(missing_ok=True)


def test_rms_reward_scale_finite():
    rms = RunningMeanStd((1,), epsilon=1e-4, clip_obs=0.0)
    raw = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    out = _rms_reward_train(raw, rms)
    assert out.shape == raw.shape
    assert np.all(np.isfinite(out))
