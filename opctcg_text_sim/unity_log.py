"""
Lecture des exports JSON OPTCGSim (Unity) : normalisation des champs hétérogènes.

Le schéma varie parfois (ex. ``life`` en liste de cartes vs entier) ; ces helpers
servent à comparer ou rejouer des snapshots sans planter.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def life_count(life: Any) -> int:
    if isinstance(life, list):
        return len(life)
    try:
        return int(life)
    except (TypeError, ValueError):
        return 0


def _as_int(x: Any, default: int = 0) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str) and x.isdigit():
        return int(x)
    return default


def normalize_player_snapshot(pl: dict[str, Any]) -> dict[str, Any]:
    """Retourne une vue avec ``life_n``, ``activeDon``, ``restedDon`` entiers."""
    out = dict(pl)
    out["life_n"] = life_count(pl.get("life"))
    out["activeDon"] = _as_int(pl.get("activeDon"), 0)
    out["restedDon"] = _as_int(pl.get("restedDon"), 0)
    out["givenDon"] = _as_int(pl.get("givenDon"), 0)
    return out


def load_snapshots_json(path: str | Path) -> list[dict[str, Any]]:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Le fichier doit être une liste de snapshots")
    return data


def board_card_rested(entry: dict[str, Any]) -> bool:
    """``bTapped`` dans les logs Unity = reposé (horizontal)."""
    card = entry.get("card") or {}
    return bool(card.get("bTapped"))


def board_attached_don(entry: dict[str, Any]) -> int:
    return _as_int(entry.get("attachedDon"), 0)
