#!/usr/bin/env python3
"""
Exporte une policy **PyTorch** ``ActorCritic`` vers des paquets Core ML pour inférence rapide
(Swift / Apple Neural Engine ou Python ``CoreMLPolicy``).

Écrit dans ``-o dir`` :
  PolicyPi.mlpackage   — logits (tronc + tête π)
  PolicyVal.mlpackage — valeur (tronc + tête V)
  coreml_manifest.json

Prérequis : macOS + ``pip install coremltools torch``

Exemple :
  python scripts/export_policy_coreml.py checkpoints/policy.pt -o coreml_export/

Voir aussi ``export_policy_swift_bundle.py`` (inférence Swift CPU float) si Core ML indisponible.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from opctcg_text_sim.checkpoint_io import load_policy_checkpoint, torch_load_train_checkpoint
from opctcg_text_sim.coreml_policy import AVAILABLE, export_actor_critic_mlpackages
from opctcg_text_sim.ppo import ActorCritic


def main() -> int:
    if not AVAILABLE:
        print(
            "Core ML indisponible : installez coremltools sur macOS "
            "(voir requirements.txt).",
            file=sys.stderr,
        )
        return 1

    ap = argparse.ArgumentParser(description="Export ActorCritic PyTorch → Core ML (.mlpackage)")
    ap.add_argument("checkpoint", type=Path, help="policy.pt (policy_state_dict ou état plat)")
    ap.add_argument("-o", "--out", type=Path, required=True, help="dossier sortie")
    ap.add_argument("--obs-dim", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--n-act", type=int, default=None)
    ap.add_argument("--n-res", type=int, default=None)
    args = ap.parse_args()

    raw = torch_load_train_checkpoint(args.checkpoint, map_location="cpu")
    if isinstance(raw, dict) and "policy_state_dict" in raw:
        sd = raw["policy_state_dict"]
    else:
        sd = raw
    if not isinstance(sd, dict):
        print("Checkpoint invalide.", file=sys.stderr)
        return 1

    od = args.obs_dim or int(sd["proj.0.weight"].shape[1])
    hd = args.hidden or int(sd["proj.0.weight"].shape[0])
    na = args.n_act or int(sd["pi_head.weight"].shape[0])
    if args.n_res is not None:
        nr = args.n_res
    else:
        bids = {
            int(k.split(".")[1])
            for k in sd
            if k.startswith("blocks.") and k.endswith(".net.0.weight")
        }
        nr = max(bids) + 1 if bids else 2

    net = ActorCritic(od, na, hidden=hd, n_res=nr)
    load_policy_checkpoint(args.checkpoint, net, torch.device("cpu"), data=raw)

    path_pi, path_val = export_actor_critic_mlpackages(net, args.out, obs_dim=od)
    print(f"OK Core ML → {args.out.resolve()}")
    print(f"  {path_pi.name}")
    print(f"  {path_val.name}")
    print(f"  coreml_manifest.json  (obs_dim={od}, n_act={na})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
