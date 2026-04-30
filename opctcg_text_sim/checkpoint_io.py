"""
Chargement / sérialisation des checkpoints PPO (politique + RunningMeanStd).
``weights_only=False`` : requis pour les dict avec états numpy (obs_rms, reward_rms).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .obs_rms import RunningMeanStd
    from .ppo import ActorCritic


def torch_load_train_checkpoint(path: Path | str, map_location: torch.device | str) -> Any:
    """``torch.load`` compatible PyTorch 2.6+ (checkpoints non strictement « weights only »)."""
    p = str(Path(path).expanduser())
    kw: dict = dict(map_location=map_location)
    try:
        kw["weights_only"] = False
    except TypeError:
        pass
    return torch.load(p, **kw)


def load_policy_checkpoint(
    path: Path | str,
    net: "ActorCritic",
    device: torch.device,
    *,
    obs_rms: "RunningMeanStd | None" = None,
    reward_rms: "RunningMeanStd | None" = None,
    data: Any | None = None,
) -> None:
    """Charge un .pt plat ou ``{policy_state_dict, obs_rms?, reward_rms?}``."""
    if data is None:
        data = torch_load_train_checkpoint(path, device)
    if isinstance(data, dict) and "policy_state_dict" in data:
        net.load_state_dict(data["policy_state_dict"], strict=True)
        if obs_rms is not None:
            rms = data.get("obs_rms")
            if rms is not None:
                obs_rms.load_state_dict(rms)
            else:
                print(
                    "  (checkpoint sans obs_rms — stats obs réinitialisées)",
                    flush=True,
                )
        if reward_rms is not None:
            rr = data.get("reward_rms")
            if rr is not None:
                reward_rms.load_state_dict(rr)
            else:
                print(
                    "  (checkpoint sans reward_rms — stats reward réinitialisées)",
                    flush=True,
                )
    else:
        net.load_state_dict(data, strict=True)
        if obs_rms is not None:
            print("  (ancien .pt — stats obs réinitialisées)", flush=True)
        if reward_rms is not None:
            print("  (ancien .pt — stats reward réinitialisées)", flush=True)


def checkpoint_dict(
    net: "ActorCritic",
    *,
    obs_rms: "RunningMeanStd | None" = None,
    reward_rms: "RunningMeanStd | None" = None,
) -> Any:
    """Dict à passer à ``torch.save`` ; format plat si aucun RMS."""
    src = net._orig_mod if hasattr(net, "_orig_mod") else net
    sd: Any = src.state_dict()
    if obs_rms is None and reward_rms is None:
        return sd
    out: dict[str, Any] = {"policy_state_dict": sd}
    if obs_rms is not None:
        out["obs_rms"] = obs_rms.state_dict()
    if reward_rms is not None:
        out["reward_rms"] = reward_rms.state_dict()
    return out
