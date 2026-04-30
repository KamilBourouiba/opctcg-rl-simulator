#!/usr/bin/env python3
"""Alias vers ``train_ppo_mlx.py`` — entrée documentée « pile M1 native » (MLX + pickle)."""
from __future__ import annotations

import runpy
from pathlib import Path

_scripts = Path(__file__).resolve().parent
runpy.run_path(str(_scripts / "train_ppo_mlx.py"), run_name="__main__")
