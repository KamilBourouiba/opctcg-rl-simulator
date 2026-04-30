#!/usr/bin/env python3
"""
Exporte une policy PyTorch (``.pt``) ou MLX (``mlx_flat``) vers un dossier consommable par le paquet Swift :

Sur **Apple Silicon**, préférez ``scripts/export_policy_coreml.py`` pour l’inférence Swift/Core ML
(ANE, plus rapide) ; ce script fournit un fallback **pur Swift float** sans dépendance Core ML.

  swift_bundle/
    manifest.json    — métadonnées + liste des segments float32
    weights.bin      — concaténation little-endian float32

Usage :
  python scripts/export_policy_swift_bundle.py checkpoints/policy.pt -o swift_export/
  python scripts/export_policy_swift_bundle.py checkpoints/policy_mlx.pt --mlx -o swift_export/

Vérification (optionnel) :
  SWIFT_BIN=.build/debug/op-policy-infer  # après swift build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from opctcg_text_sim.checkpoint_io import torch_load_train_checkpoint

NEG_INF = -1e9


def _load_flat_torch(sd: dict[str, torch.Tensor], obs_dim: int, hidden: int, n_act: int, n_res: int) -> list[tuple[str, np.ndarray]]:
    """Ordre canonique aligné sur ``swift/Sources/OPCTCGPolicy/PolicyNetwork.swift``."""

    def conv(name: str, t: torch.Tensor) -> tuple[str, np.ndarray]:
        arr = t.detach().cpu().numpy().astype(np.float32, copy=False)
        return name, np.ascontiguousarray(arr)

    out: list[tuple[str, np.ndarray]] = []

    out.append(conv("proj.linear.weight", sd["proj.0.weight"]))
    out.append(conv("proj.linear.bias", sd["proj.0.bias"]))
    out.append(conv("proj.norm.weight", sd["proj.1.weight"]))
    out.append(conv("proj.norm.bias", sd["proj.1.bias"]))

    for i in range(n_res):
        prefix = f"blocks.{i}"
        out.append(conv(f"res{i}.lin1.weight", sd[f"{prefix}.net.0.weight"]))
        out.append(conv(f"res{i}.lin1.bias", sd[f"{prefix}.net.0.bias"]))
        out.append(conv(f"res{i}.ln1.weight", sd[f"{prefix}.net.1.weight"]))
        out.append(conv(f"res{i}.ln1.bias", sd[f"{prefix}.net.1.bias"]))
        out.append(conv(f"res{i}.lin2.weight", sd[f"{prefix}.net.3.weight"]))
        out.append(conv(f"res{i}.lin2.bias", sd[f"{prefix}.net.3.bias"]))
        out.append(conv(f"res{i}.ln2.weight", sd[f"{prefix}.net.4.weight"]))
        out.append(conv(f"res{i}.ln2.bias", sd[f"{prefix}.net.4.bias"]))

    out.append(conv("pi.weight", sd["pi_head.weight"]))
    out.append(conv("pi.bias", sd["pi_head.bias"]))
    out.append(conv("v.weight", sd["v_head.weight"]))
    out.append(conv("v.bias", sd["v_head.bias"]))

    # Sanity shapes
    assert out[0][1].shape == (hidden, obs_dim), out[0]
    assert out[-2][1].shape == (1, hidden), out[-2]
    assert out[-4][1].shape == (n_act, hidden), out[-4]

    return out


def _load_flat_mlx(flat: dict[str, np.ndarray], obs_dim: int, hidden: int, n_act: int, n_res: int) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []

    def get(k: str) -> np.ndarray:
        a = flat[k]
        return np.ascontiguousarray(np.asarray(a, dtype=np.float32))

    out.append(("proj.linear.weight", get("proj.layers.0.weight")))
    out.append(("proj.linear.bias", get("proj.layers.0.bias")))
    out.append(("proj.norm.weight", get("proj.layers.1.weight")))
    out.append(("proj.norm.bias", get("proj.layers.1.bias")))

    for i in range(n_res):
        out.append((f"res{i}.lin1.weight", get(f"blocks.layers.{i}.lin1.weight")))
        out.append((f"res{i}.lin1.bias", get(f"blocks.layers.{i}.lin1.bias")))
        out.append((f"res{i}.ln1.weight", get(f"blocks.layers.{i}.ln1.weight")))
        out.append((f"res{i}.ln1.bias", get(f"blocks.layers.{i}.ln1.bias")))
        out.append((f"res{i}.lin2.weight", get(f"blocks.layers.{i}.lin2.weight")))
        out.append((f"res{i}.lin2.bias", get(f"blocks.layers.{i}.lin2.bias")))
        out.append((f"res{i}.ln2.weight", get(f"blocks.layers.{i}.ln2.weight")))
        out.append((f"res{i}.ln2.bias", get(f"blocks.layers.{i}.ln2.bias")))

    out.append(("pi.weight", get("pi_head.weight")))
    out.append(("pi.bias", get("pi_head.bias")))
    out.append(("v.weight", get("v_head.weight")))
    out.append(("v.bias", get("v_head.bias")))
    return out


def export_bundle(
    tensors: list[tuple[str, np.ndarray]],
    *,
    obs_dim: int,
    hidden: int,
    n_act: int,
    n_res: int,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    segments: list[dict[str, Any]] = []
    offset = 0
    for name, arr in tensors:
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        raw = flat.astype("<f4", copy=False).tobytes()
        blob.extend(raw)
        n = flat.size
        segments.append(
            {
                "name": name,
                "byte_offset": offset,
                "byte_length": n * 4,
                "shape": list(arr.shape),
            }
        )
        offset += n * 4

    manifest = {
        "schema_version": 1,
        "neg_inf_mask": NEG_INF,
        "obs_dim": obs_dim,
        "hidden_dim": hidden,
        "n_act": n_act,
        "n_res_blocks": n_res,
        "dtype": "float32",
        "endian": "little",
        "segments": segments,
        "weights_file": "weights.bin",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "weights.bin").write_bytes(blob)


def main() -> int:
    ap = argparse.ArgumentParser(description="Export policy → Swift bundle (manifest.json + weights.bin)")
    ap.add_argument("checkpoint", type=Path, help="policy.pt (torch) ou MLX")
    ap.add_argument("-o", "--out", type=Path, required=True, help="dossier sortie")
    ap.add_argument("--mlx", action="store_true", help="checkpoint MLX (clé mlx_flat)")
    ap.add_argument("--obs-dim", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--n-act", type=int, default=None)
    ap.add_argument("--n-res", type=int, default=None)
    args = ap.parse_args()

    data = torch_load_train_checkpoint(args.checkpoint, map_location="cpu")

    if args.mlx:
        if not isinstance(data, dict) or "mlx_flat" not in data:
            print("Format MLX attendu : dict avec cle 'mlx_flat'", flush=True)
            return 1
        flat_mlx = {k: np.asarray(v, dtype=np.float32) for k, v in data["mlx_flat"].items()}
        od = args.obs_dim or int(flat_mlx["proj.layers.0.weight"].shape[1])
        hd = args.hidden or int(flat_mlx["proj.layers.0.weight"].shape[0])
        na = args.n_act or int(flat_mlx["pi_head.weight"].shape[0])
        if args.n_res is not None:
            nr = args.n_res
        else:
            idxs = {
                int(k.split(".")[2])
                for k in flat_mlx
                if k.startswith("blocks.layers.") and k.endswith(".lin1.weight")
            }
            nr = max(idxs) + 1 if idxs else 2
        tensors = _load_flat_mlx(flat_mlx, od, hd, na, nr)
    else:
        if isinstance(data, dict) and "policy_state_dict" in data:
            sd = data["policy_state_dict"]
        else:
            sd = data
        if not isinstance(sd, dict):
            print("Checkpoint torch invalide.", flush=True)
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
        tensors = _load_flat_torch(sd, od, hd, na, nr)

    export_bundle(tensors, obs_dim=od, hidden=hd, n_act=na, n_res=nr, out_dir=args.out.resolve())
    print(f"OK → {args.out.resolve()}  (manifest.json + weights.bin)")
    print(f"  obs_dim={od} hidden={hd} n_act={na} n_res={nr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
