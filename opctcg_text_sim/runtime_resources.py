"""
Réglages processus pour entraînement vectorisé : limites de fichiers, threads BLAS,
threads PyTorch (process principal vs workers).

Les workers fixent OMP/MKL/OpenBLAS à 1 pour éviter la sur-souscription quand
plusieurs processus tournent en parallèle.
"""
from __future__ import annotations

import os
import resource


def raise_process_file_limit(target: int = 8192) -> None:
    """
    Augmente la limite soft de descripteurs de fichiers (macOS / Linux).
    Nécessaire pour de nombreux workers + pipes / shared memory.
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < target:
            new_soft = min(target, hard) if hard > 0 else target
            resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except Exception:
        pass


def configure_worker_blas_threads() -> None:
    """À appeler au début de chaque worker Subproc / SharedMemory (spawn)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def configure_main_process_torch(
    cpu_count: int | None = None,
    *,
    num_threads: int | None = None,
    num_interop_threads: int | None = None,
) -> None:
    """
    Process principal : threads PyTorch pour les ops CPU (mise à jour PPO, etc.).

    ``num_threads`` / ``num_interop_threads`` : plafonds optionnels (YAML
    ``training.main_torch_num_threads``) pour A/B « moins de CPU / laisser la marge ».
    """
    try:
        import torch

        n_cpu = max(1, cpu_count or os.cpu_count() or 4)
        intra = n_cpu if num_threads is None else max(1, min(n_cpu, int(num_threads)))
        if num_interop_threads is None:
            inter = max(1, min(8, intra // 2))
        else:
            inter = max(1, min(32, int(num_interop_threads)))
        torch.set_num_threads(intra)
        torch.set_num_interop_threads(inter)
    except Exception:
        pass


def suggested_parallel_envs(
    cpu_count: int,
    *,
    cap: int = 64,
    profile: str = "balanced",
) -> int:
    """
    Nombre d’envs parallèles suggéré.

    - ``max`` : un env par cœur (plafonné à ``cap``).
    - ``balanced`` : plafond 24 pour rester raisonnable hors Apple Silicon.
    - ``conservative`` : la moitié des cœurs, max 8.
    """
    cpu_count = max(1, cpu_count)
    p = (profile or "balanced").strip().lower()
    if p == "max":
        return max(2, min(cap, cpu_count))
    if p == "conservative":
        return max(1, min(8, cpu_count // 2))
    return max(1, min(24, cpu_count))
