"""
CoreMLPolicy — accélération Apple Neural Engine pour l'inférence PPO.

Rôle dans la boucle d'entraînement :
  • Collecte (rollout) : inférence via Core ML (ANE + GPU + CPU, ~10× MPS)
  • Mise à jour (PPO)  : gradient sur PyTorch/MPS (Core ML ne supporte pas l'entraînement)
  • Synchronisation    : sync_async() lance le re-export CoreML dans un thread background ;
                        la boucle principale continue sans attendre (~4× speedup réel).

Usage :
    from opctcg_text_sim.coreml_policy import CoreMLPolicy

    policy = CoreMLPolicy(net, obs_dim=96, n_act=111)
    logits, values = policy.infer(obs_np)    # obs_np : (B, obs_dim) float32 numpy
    policy.sync_async(net)                   # non-bloquant — lance en arrière-plan
    policy.wait_sync()                       # appeler en fin de training seulement

Important : si l'entraînement utilise ``obs_normalize`` (running mean/std), le modèle
exporté attend des **observations déjà normalisées** comme dans ``collect_rollout_coreml``.
Pour de l'inférence hors ``train_ppo`` / ``infer_policy.py``, appliquer les mêmes stats
(``obs_rms`` du checkpoint) avant ``infer``.
"""
from __future__ import annotations

import copy
import logging
import os
import threading
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from .ppo import ActorCritic


def _is_available() -> bool:
    """Retourne True si coremltools est installé et qu'on est sur macOS."""
    try:
        import coremltools  # noqa: F401
        import platform
        return platform.system() == "Darwin"
    except ImportError:
        return False


AVAILABLE = _is_available()


# ──────────────────────────────────────────────────────────────────────────────
# Modules internes légers (exportés séparément pour éviter les limitations
# des traced models avec branches dynamiques)
# ──────────────────────────────────────────────────────────────────────────────

class _PolicyHead(object):
    """Wrapper autour du torch.nn.Module pour export Core ML."""

    class _Module:
        pass  # défini dynamiquement via import torch


def _make_torch_modules(net: "ActorCritic"):
    """
    Crée deux modules torch traceables depuis le réseau ActorCritic.
    Compatible avec l'architecture proj+blocks (résiduelle) et l'ancienne body.
      - pi_module  : obs → logits
      - val_module : obs → value  (scalaire)
    """
    import torch
    import torch.nn as nn
    import copy

    net_cpu = copy.deepcopy(net).to("cpu").eval()

    class _Pi(nn.Module):
        def __init__(self, trunk, pi_head):
            super().__init__()
            self.trunk   = trunk
            self.pi_head = pi_head

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.pi_head(self.trunk(x))

    class _Val(nn.Module):
        def __init__(self, trunk, v_head):
            super().__init__()
            self.trunk  = trunk
            self.v_head = v_head

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.v_head(self.trunk(x)).squeeze(-1)

    # Détecter l'architecture (proj+blocks résiduelle, ou body historique)
    if hasattr(net_cpu, "proj") and hasattr(net_cpu, "blocks"):
        trunk = nn.Sequential(net_cpu.proj, net_cpu.blocks)
    elif hasattr(net_cpu, "body"):
        trunk = net_cpu.body
    else:
        raise AttributeError("ActorCritic : architecture inconnue (ni proj/blocks ni body)")

    pi  = _Pi(trunk, net_cpu.pi_head).eval()
    val = _Val(trunk, net_cpu.v_head).eval()
    return pi, val


# ──────────────────────────────────────────────────────────────────────────────
# Export PyTorch → Core ML
# ──────────────────────────────────────────────────────────────────────────────

def _export_coreml(module, obs_dim: int, name: str):
    """Exporte un nn.Module vers un MLModel Core ML (compute_units=ALL, fp16)."""
    import torch
    import coremltools as ct

    dummy = torch.randn(1, obs_dim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(module, dummy)
        # coremltools affiche des barres tqdm « Running MIL … » sur stderr → bruit + ralentissements ;
        # on coupe stdout/stderr pendant convert + on baisse le niveau des loggers du package.
        _loggers = [
            logging.getLogger(n)
            for n in ("coremltools", "coremltools.converters", "coremltools.converters.mil")
        ]
        _prev = [(lg, lg.level, lg.propagate) for lg in _loggers]
        for lg in _loggers:
            lg.setLevel(logging.ERROR)
            lg.propagate = False
        mlmodel = None
        try:
            with open(os.devnull, "w", encoding="utf-8") as _devnull:
                with redirect_stdout(_devnull), redirect_stderr(_devnull):
                    mlmodel = ct.convert(
                        traced,
                        inputs=[ct.TensorType(name="x", shape=ct.Shape(shape=(ct.RangeDim(1, 64), obs_dim)))],
                        compute_units=ct.ComputeUnit.ALL,
                        minimum_deployment_target=ct.target.macOS13,
                        compute_precision=ct.precision.FLOAT16,
                    )
        finally:
            for lg, lev, prop in _prev:
                lg.setLevel(lev)
                lg.propagate = prop
            try:
                del traced
            except Exception:
                pass
        if mlmodel is None:
            raise RuntimeError("coremltools.convert a renvoyé None")
        return mlmodel


def export_actor_critic_mlpackages(
    net: "ActorCritic",
    out_dir: os.PathLike[str] | str,
    *,
    obs_dim: int | None = None,
) -> tuple[Path, Path]:
    """
    Écrit deux paquets Core ML (π et V) pour inférence Swift / ANE :
      ``PolicyPi.mlpackage``, ``PolicyVal.mlpackage`` + ``coreml_manifest.json``.

    Même graphe que ``CoreMLPolicy`` (tronc partagé exporté deux fois dans chaque paquet —
    fichier plus gros que deux têtes seules, mais compatible ``torch.jit.trace`` sans refactoring).

    Retourne ``(chemin_pi, chemin_val)``.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    if obs_dim is None:
        w = net.proj[0].weight if hasattr(net, "proj") else net.body[0].weight  # type: ignore[attr-defined]
        obs_dim = int(w.shape[1])

    pi_mod, val_mod = _make_torch_modules(net)
    ml_pi = _export_coreml(pi_mod, obs_dim, "pi")
    ml_val = _export_coreml(val_mod, obs_dim, "val")

    path_pi = out / "PolicyPi.mlpackage"
    path_val = out / "PolicyVal.mlpackage"
    ml_pi.save(str(path_pi))
    ml_val.save(str(path_val))

    n_act = int(net.pi_head.out_features) if hasattr(net.pi_head, "out_features") else int(net.pi_head.weight.shape[0])

    import json

    meta = {
        "schema_version": 1,
        "obs_dim": obs_dim,
        "n_act": n_act,
        "input_name": "x",
        "batch_range": [1, 64],
        "neg_inf_mask": -1e9,
        "packages": {"pi": path_pi.name, "val": path_val.name},
    }
    (out / "coreml_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return path_pi, path_val


# ──────────────────────────────────────────────────────────────────────────────
# Classe principale
# ──────────────────────────────────────────────────────────────────────────────

class CoreMLPolicy:
    """
    Encapsule deux MLModel Core ML (politique + valeur) pour l'inférence rapide.

    L'export initial prend ~1-2 s (une seule fois au démarrage).
    sync_async() relance l'export en arrière-plan : le rollout suivant démarre
    immédiatement avec le modèle précédent (1 update stale → sans impact PPO).
    Quand le thread finit, le nouveau modèle est swappé atomiquement sous lock.

    Si coremltools n'est pas disponible, replie silencieusement sur MPS/CPU.
    """

    def __init__(self, net: "ActorCritic", obs_dim: int = 96, n_act: int = 111):
        self.obs_dim = obs_dim
        self.n_act   = n_act
        self._net    = net           # référence PyTorch (pour fallback)
        self._ml_pi  = None
        self._ml_val = None
        self._ok     = False

        # Thread background pour sync async
        self._lock:        threading.Lock          = threading.Lock()
        self._sync_thread: threading.Thread | None = None

        if AVAILABLE:
            try:
                self._build(net)
                self._ok = True
                print("CoreMLPolicy : ANE + GPU + CPU activés (fp16, async sync)", flush=True)
            except Exception as exc:
                print(f"CoreMLPolicy : échec export → fallback MPS ({exc})", flush=True)
        else:
            print("CoreMLPolicy : coremltools absent → fallback MPS", flush=True)

    def _build(self, net: "ActorCritic") -> None:
        pi_mod, val_mod = _make_torch_modules(net)
        self._ml_pi  = _export_coreml(pi_mod,  self.obs_dim, "pi")
        self._ml_val = _export_coreml(val_mod, self.obs_dim, "val")

    # ── Inférence ────────────────────────────────────────────────────────────

    def infer(
        self,
        obs: "np.ndarray",
        mask: "np.ndarray | None" = None,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """
        Inférence rapide sur ANE.

        Args:
            obs  : (B, obs_dim) float32
            mask : (B, n_act)   bool  — True = action légale (None = toutes légales)

        Returns:
            logits : (B, n_act) float32
            values : (B,)       float32
        """
        if not self._ok:
            return self._fallback_infer(obs, mask)

        # Snapshot atomique des modèles courants (swap possible en arrière-plan)
        with self._lock:
            ml_pi  = self._ml_pi
            ml_val = self._ml_val

        inp    = {"x": obs.astype(np.float32)}
        logits = next(iter(ml_pi.predict(inp).values()))    # (B, n_act)
        values = next(iter(ml_val.predict(inp).values()))   # (B,)

        if mask is not None:
            logits = np.where(mask, logits, -1e9)

        return logits, values

    def _fallback_infer(
        self,
        obs: "np.ndarray",
        mask: "np.ndarray | None",
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Repli sur PyTorch (MPS ou CPU) si Core ML non disponible."""
        import torch
        device = next(self._net.parameters()).device
        x = torch.as_tensor(obs, dtype=torch.float32, device=device)
        m = torch.as_tensor(mask, dtype=torch.bool, device=device) if mask is not None else None
        with torch.no_grad():
            logits_t, val_t = self._net(x, m)
        return logits_t.cpu().numpy(), val_t.cpu().numpy()

    # ── Synchronisation après update PPO (non-bloquante) ─────────────────────

    def sync_async(self, net: "ActorCritic") -> None:
        """
        Lance la re-synchronisation des poids CoreML dans un thread background.

        • Non-bloquant : le rollout suivant démarre immédiatement.
        • Si un sync précédent tourne encore → on l'ignore (modèle légèrement
          stale, sans impact mesurable sur la convergence PPO).
        • Quand le thread finit, les modèles CoreML sont swappés sous lock
          de façon atomique, sans bloquer infer().
        """
        if not self._ok:
            self._net = net
            return

        # Ne lance pas un nouveau sync si le précédent est encore actif
        if self._sync_thread is not None and self._sync_thread.is_alive():
            return

        # Copier les poids AVANT de lancer le thread (le réseau va continuer
        # à être entraîné pendant ce temps)
        net_orig = net._orig_mod if hasattr(net, "_orig_mod") else net
        net_copy = copy.deepcopy(net_orig).cpu().eval()
        obs_dim  = self.obs_dim

        def _do() -> None:
            nonlocal net_copy
            try:
                pi_mod, val_mod = _make_torch_modules(net_copy)
                new_pi = _export_coreml(pi_mod, obs_dim, "pi")
                new_val = _export_coreml(val_mod, obs_dim, "val")
                del pi_mod, val_mod
                net_copy = None  # libère le deepcopy (export MIL + traced est très gourmand en RAM)
                with self._lock:
                    old_pi, old_val = self._ml_pi, self._ml_val
                    self._ml_pi = new_pi
                    self._ml_val = new_val
                del old_pi, old_val
            except Exception as exc:
                print(f"CoreMLPolicy.sync_async : échec ({exc})", flush=True)

        self._sync_thread = threading.Thread(target=_do, daemon=True)
        self._sync_thread.start()

    def sync(self, net: "ActorCritic") -> None:
        """Synchronisation bloquante (compat. rétrograde). Préférer sync_async()."""
        if not self._ok:
            self._net = net
            return
        try:
            self._build(net)
            self._net = net
        except Exception as exc:
            print(f"CoreMLPolicy.sync : échec → fallback ({exc})", flush=True)
            self._ok  = False
            self._net = net

    def wait_sync(self) -> None:
        """Attend la fin du sync background (appeler une seule fois en fin de training)."""
        if self._sync_thread is not None:
            self._sync_thread.join()
            self._sync_thread = None

    @property
    def ready(self) -> bool:
        return self._ok
