#!/usr/bin/env python3
"""
Évaluation greedy d’un checkpoint PPO (obs normalisées comme à l’entraînement si présentes).

Exemple :
  python scripts/infer_policy.py --checkpoint checkpoints/policy.pt \\
      --config config.yaml --deck decks/NAMIBY.txt --episodes 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opctcg_text_sim.checkpoint_io import load_policy_checkpoint, torch_load_train_checkpoint
from opctcg_text_sim.env import OPTextSimEnv
from opctcg_text_sim.obs_rms import RunningMeanStd
from opctcg_text_sim.ppo import ActorCritic


def _pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def greedy_episode(
    net: ActorCritic,
    env: OPTextSimEnv,
    device: torch.device,
    *,
    obs_rms: RunningMeanStd | None,
    max_steps: int,
) -> tuple[float, bool]:
    net.eval()
    o, _ = env.reset()
    total = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            m_np = env.legal_actions_mask()
            if not m_np.any():
                m_np[:] = True
            o_f = o if obs_rms is None else obs_rms.normalize_eval(o)
            x = torch.as_tensor(o_f, dtype=torch.float32, device=device).unsqueeze(0)
            m_t = torch.as_tensor(m_np, dtype=torch.bool, device=device).unsqueeze(0)
            logits, _ = net(x, m_t)
            a = int(logits.argmax(dim=-1).item())
            o, r, term, trunc, _ = env.step(a)
            total += float(r)
            if term or trunc:
                break
    return total, total > 0.0


def main() -> int:
    p = argparse.ArgumentParser(description="Inférence greedy (PPO)")
    p.add_argument("--checkpoint", type=Path, required=True, help="policy.pt")
    p.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    p.add_argument("--deck", type=Path, default=ROOT / "decks" / "NAMIBY.txt")
    p.add_argument("--deck1", type=Path, default=None)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=512)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths = raw.get("paths") or {}
    train_cfg = raw.get("training") or {}
    sim_cfg = dict(raw.get("sim") or {})

    csv_path = args.csv
    if csv_path is None:
        rel = paths.get("cards_csv", "data/cards_stub.csv")
        pc = Path(rel).expanduser()
        csv_path = (pc if pc.is_absolute() else ROOT / pc).resolve()
    if not csv_path.is_file():
        print(f"CSV introuvable : {csv_path}", file=sys.stderr)
        return 1

    col_map = {str(k): str(v) for k, v in raw.get("card_csv", {}).items()}
    rules_p = ROOT / paths.get("rules_corpus_out", "data/rules_corpus.txt")
    rules_p = rules_p if rules_p.is_file() else None

    deck0 = args.deck.expanduser()
    deck1 = (args.deck1 or args.deck).expanduser()
    if not deck0.is_file() or not deck1.is_file():
        print("Deck introuvable.", file=sys.stderr)
        return 1

    obs_dim = int(train_cfg.get("obs_dim", 96))
    hidden = int(train_cfg.get("hidden", 256))
    n_res = int(train_cfg.get("n_res", 2))
    device = _pick_device(args.device)

    env = OPTextSimEnv(
        deck0, deck1, csv_path, col_map, rules_p, sim_cfg,
        obs_dim=obs_dim, seed=args.seed,
    )
    n_act = int(env.action_space.n)
    net = ActorCritic(obs_dim, n_act, hidden, n_res).to(device)

    _obs_clip = float(train_cfg.get("obs_normalize_clip", 10.0))
    payload = torch_load_train_checkpoint(args.checkpoint, device)
    obs_rms: RunningMeanStd | None = None
    if isinstance(payload, dict) and payload.get("obs_rms") is not None:
        obs_rms = RunningMeanStd((obs_dim,), clip_obs=_obs_clip)
    load_policy_checkpoint(
        args.checkpoint,
        net,
        device,
        obs_rms=obs_rms,
        reward_rms=None,
        data=payload,
    )

    wins = 0
    rews: list[float] = []
    for ep in range(int(args.episodes)):
        env.reset(seed=args.seed + ep)
        r, w = greedy_episode(net, env, device, obs_rms=obs_rms, max_steps=int(args.max_steps))
        rews.append(r)
        wins += int(w)
        print(f"  ep {ep + 1}/{args.episodes}  reward={r:+.2f}  win={w}", flush=True)

    env.close()
    print(
        f"Résumé : win_rate={100.0 * wins / max(1, args.episodes):.1f}%  "
        f"mean_rew={sum(rews) / len(rews):+.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
