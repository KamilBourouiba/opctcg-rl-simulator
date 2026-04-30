#!/usr/bin/env python3
"""
Benchmark du pipeline d'entraînement complet.
Mesure : rollout (env + inférence), PPO update, et throughput total.
"""
from __future__ import annotations
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from opctcg_text_sim.env import OPTextSimEnv
from opctcg_text_sim.ppo import (
    ActorCritic,
    collect_rollout_coreml,
    collect_rollout_vec,
    ppo_update,
)
from opctcg_text_sim.coreml_policy import CoreMLPolicy, AVAILABLE as COREML_AVAILABLE
from opctcg_text_sim.shmem_vec_env import SharedMemoryVecEnv
from scripts.train_ppo import _EnvFactory  # fabrique picklable

cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
col = {str(k): str(v) for k, v in cfg.get("card_csv", {}).items()}
sim_cfg = cfg.get("sim", {})
train = cfg.get("training", {})

DECK0 = ROOT / "decks" / "NAMIBY.txt"
DECK1 = ROOT / "decks" / "KAIDO.txt"
CSV   = ROOT / "data" / "cards_tcgcsv.csv"
OBS   = 96
from opctcg_text_sim.simulator import ACTION_SPACE_SIZE as N_ACT
SEED  = 42

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def run_bench(n_envs: int = 12, steps_per_env: int = 342, epochs: int = 8, mini: int = 2048):
    """Lance un benchmark complet avec n_envs workers."""
    import os
    hidden = int(train.get("hidden", 1024))
    n_res  = int(train.get("n_res",    3))

    net_mps = ActorCritic(OBS, N_ACT, hidden, n_res).to(device)
    net_cpu = ActorCritic(OBS, N_ACT, hidden, n_res)

    factory = _EnvFactory(
        deck0=DECK0, deck1=DECK1, cards_csv=CSV,
        column_map=col, rules_corpus_path=None,
        sim_cfg=sim_cfg, obs_dim=OBS,
    )

    print(f"\n{'='*60}")
    print(f"n_envs={n_envs}  steps_per_env={steps_per_env}  "
          f"total={n_envs*steps_per_env}  epochs={epochs}  mini={mini}")
    print(f"Réseau : obs={OBS} → hidden={hidden} × n_res={n_res} → act={N_ACT} "
          f"({sum(p.numel() for p in net_mps.parameters()):,} params)")

    # Créer les workers (shm+CoreML si dispo, sinon Subproc + MPS comme train_ppo sans ANE)
    t_init = time.perf_counter()
    policy = None
    if COREML_AVAILABLE:
        vec = SharedMemoryVecEnv(factory, n_envs=n_envs, obs_dim=OBS, n_act=N_ACT, base_seed=SEED)
        policy = CoreMLPolicy(net_cpu, obs_dim=OBS, n_act=N_ACT)
        net_orig = net_mps._orig_mod if hasattr(net_mps, "_orig_mod") else net_mps
        policy.sync(net_orig)
        print("CoreML (ANE) actif pour l'inférence du rollout")
    else:
        vec = SharedMemoryVecEnv(factory, n_envs=n_envs, obs_dim=OBS, n_act=N_ACT, base_seed=SEED)
        print("SharedMemoryVecEnv + inférence PyTorch (rollout)")
    print(f"Workers prêts en {time.perf_counter()-t_init:.1f}s")

    # ── Benchmark rollout ──────────────────────────────────────────────────
    print("Benchmark rollout (warmup 1 iter)...", flush=True)
    if policy is not None:
        collect_rollout_coreml(vec, policy, net_mps, steps_per_env, 0.995, 0.95, device)
    else:
        collect_rollout_vec(vec, net_mps, steps_per_env, 0.995, 0.95, device)
    print("Rollout...", flush=True)
    t0 = time.perf_counter()
    for _ in range(3):
        if policy is not None:
            batch, stats, _, _ = collect_rollout_coreml(
                vec, policy, net_mps, steps_per_env, 0.995, 0.95, device
            )
        else:
            batch, stats = collect_rollout_vec(
                vec, net_mps, steps_per_env, 0.995, 0.95, device
            )
    dt_roll = (time.perf_counter() - t0) / 3
    total = n_envs * steps_per_env
    print(f"  Rollout : {total/dt_roll:>7,.0f} steps/s  ({dt_roll:.2f}s/{total} steps)")
    print(f"  Episodes: {stats.n_episodes}  win_rate={stats.win_rate:.0%}")

    # ── Benchmark update ───────────────────────────────────────────────────
    opt = torch.optim.AdamW(net_mps.parameters(), lr=2.5e-4, weight_decay=1e-4)
    print("PPO update warmup...", flush=True)
    ppo_update(net_mps, opt, batch, clip_eps=0.2, epochs=1, minibatch=512, ent_coef=0.05)

    print("PPO update...", flush=True)
    t0 = time.perf_counter()
    for _ in range(3):
        m = ppo_update(net_mps, opt, batch, clip_eps=0.2, epochs=epochs,
                       minibatch=mini, ent_coef=0.05)
    if device.type == "mps":
        torch.mps.synchronize()
    dt_upd = (time.perf_counter() - t0) / 3
    print(f"  Update  : {dt_upd:.2f}s/update (epochs={epochs}, mini={mini})")
    print(f"  Métriques: loss={m['loss']:.4f}  ent={m['ent']:.3f}  clip={m['clip']:.2f}")

    # ── Throughput total ───────────────────────────────────────────────────
    eff = total / (dt_roll + dt_upd)
    print(f"  → Throughput effectif : {eff:>7,.0f} steps/s  (rollout+update)")

    vec.close()
    return dt_roll, dt_upd, eff


if __name__ == "__main__":
    print(f"Device : {device}  |  CoreML : {COREML_AVAILABLE}")
    print(f"torch={torch.__version__}")

    # Configs à tester
    for n in [8, 12, 16]:
        try:
            run_bench(n_envs=n, steps_per_env=max(256, 4096 // n), epochs=8, mini=2048)
        except Exception as e:
            print(f"n_envs={n} ERREUR: {e}")
