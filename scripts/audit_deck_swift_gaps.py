#!/usr/bin/env python3
"""
Compatibilité: ancien point d'entrée.
Utiliser de préférence: scripts/audit_text_coverage.py
"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).with_name("audit_text_coverage.py")
    runpy.run_path(str(target), run_name="__main__")
