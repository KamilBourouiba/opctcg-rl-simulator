#!/usr/bin/env python3
"""
Grille courte n_envs × minibatch (rollout 4096 fixe) pour choisir les défauts training.
Utilise SharedMemoryVecEnv + collect_rollout_coreml si CoreML dispo, sinon collect_rollout_vec (PyTorch).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from opctcg_text_sim.env import OPTextSimEnv
from opctcg_text_sim.ppo import ActorCritic, collect_rollout_coreml, collect_rollout_vec, ppo_update
from opctcg_text_sim.coreml_policy import CoreMLPolicy, AVAILABLE as COREML_AVAILABLE
from opctcg_text_sim.shmem_vec_env import SharedMemoryVecEnv
from opctcg_text_sim.simulator import ACTION_SPACE_SIZE as N_ACT
from scripts.train_ppo import _EnvFactory

ROLLOUT = 4096
GAMMA, LAM = 0.995, 0.95
EPOCHS = 8
SEED = 42
DECK0 = ROOT / "decks" / "NAMIBY.txt"
DECK1 = ROOT / "decks" / "KAIDO.txt"
CSV = ROOT / "data" / "cards_tcgcsv.csv"


def _load_cfg() -> tuple[dict, dict, dict]:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    col = {str(k): str(v) for k, v in raw.get("card_csv", {}).items()}
    sim = dict(raw.get("sim") or {})
    train = dict(raw.get("training") or {})
    return col, sim, train


def bench_once(
    n_envs: int,
    minibatch: int,
    col: dict,
    sim_cfg: dict,
    train: dict,
) -> float:
    obs_dim = int(train.get("obs_dim", 96))
    hidden = int(train.get("hidden", 1024))
    n_res = int(train.get("n_res", 3))
    steps_per_env = max(1, ROLLOUT // n_envs)
    total = n_envs * steps_per_env

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net_mps = ActorCritic(obs_dim, N_ACT, hidden, n_res).to(device)
    net_cpu = ActorCritic(obs_dim, N_ACT, hidden, n_res)

    factory = _EnvFactory(
        deck0=DECK0,
        deck1=DECK1,
        cards_csv=CSV,
        column_map=col,
        rules_corpus_path=None,
        sim_cfg=sim_cfg,
        obs_dim=obs_dim,
    )
    policy: CoreMLPolicy | None = None
    if COREML_AVAILABLE:
        vec = SharedMemoryVecEnv(
            factory, n_envs=n_envs, obs_dim=obs_dim, n_act=N_ACT, base_seed=SEED
        )
        policy = CoreMLPolicy(net_cpu, obs_dim=obs_dim, n_act=N_ACT)
        net_orig = net_mps._orig_mod if hasattr(net_mps, "_orig_mod") else net_mps
        policy.sync(net_orig)
    else:
        vec = SharedMemoryVecEnv(
            factory, n_envs=n_envs, obs_dim=obs_dim, n_act=N_ACT, base_seed=SEED
        )

    try:
        # warmup
        if policy is not None:
            collect_rollout_coreml(
                vec, policy, net_mps, steps_per_env, GAMMA, LAM, device
            )
        else:
            collect_rollout_vec(vec, net_mps, steps_per_env, GAMMA, LAM, device)

        t0 = time.perf_counter()
        for _ in range(3):
            if policy is not None:
                batch, _, _, _ = collect_rollout_coreml(
                    vec, policy, net_mps, steps_per_env, GAMMA, LAM, device
                )
            else:
                batch, _ = collect_rollout_vec(
                    vec, net_mps, steps_per_env, GAMMA, LAM, device
                )
        dt_roll = (time.perf_counter() - t0) / 3.0

        opt = torch.optim.AdamW(net_mps.parameters(), lr=2.5e-4, weight_decay=1e-4)
        ppo_update(
            net_mps, opt, batch, clip_eps=0.2, epochs=1, minibatch=512, ent_coef=0.05
        )
        t0 = time.perf_counter()
        for _ in range(3):
            ppo_update(
                net_mps,
                opt,
                batch,
                clip_eps=0.2,
                epochs=EPOCHS,
                minibatch=minibatch,
                ent_coef=0.05,
            )
        if device.type == "mps":
            torch.mps.synchronize()
        dt_upd = (time.perf_counter() - t0) / 3.0

        return float(total / (dt_roll + dt_upd))
    finally:
        vec.close()


def main() -> int:
    col, sim_cfg, train = _load_cfg()
    if not DECK0.is_file() or not DECK1.is_file() or not CSV.is_file():
        print("Decks ou CSV manquants (NAMIBY, KAIDO, cards_tcgcsv).", file=sys.stderr)
        return 1

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(
        f"bench_throughput_grid  rollout={ROLLOUT}  epochs={EPOCHS}  "
        f"device={device}  CoreML={COREML_AVAILABLE}",
        flush=True,
    )

    best_eff = -1.0
    best: tuple[int, int] | None = None
    results: list[tuple[int, int, float]] = []

    n_list = [8, 12, 16, 20]
    mini_list = [1024, 2048, 4096]

    for n in n_list:
        for mini in mini_list:
            if mini > ROLLOUT:
                continue
            try:
                eff = bench_once(n, mini, col, sim_cfg, train)
            except Exception as e:
                print(f"n_envs={n} minibatch={mini}  ERREUR: {e}", flush=True)
                continue
            results.append((n, mini, eff))
            print(f"n_envs={n:2d}  minibatch={mini:4d}  →  {eff:,.0f} steps/s (rollout+update)", flush=True)
            if eff > best_eff:
                best_eff = eff
                best = (n, mini)

    if best is None:
        print("Aucun résultat.", file=sys.stderr)
        return 1

    print(f"\nMeilleur couple : n_envs={best[0]}  minibatch={best[1]}  ({best_eff:,.0f} steps/s)")
    out = ROOT / "runs" / "bench_throughput_winner.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"n_envs={best[0]}\nminibatch={best[1]}\neffective_steps_per_s={best_eff:.1f}\n"
        f"CoreML={COREML_AVAILABLE}\ndevice={device}\n",
        encoding="utf-8",
    )
    print(f"Écrit : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
