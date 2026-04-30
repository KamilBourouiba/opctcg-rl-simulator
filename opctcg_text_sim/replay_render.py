"""
Rendu d’animation (GIF) à partir d’un journal JSONL produit par ``OPTextSimEnv``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from .card_keywords import extract_keywords_from_text, format_keywords_line


def load_jsonl_frames(path: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(json.loads(line))
    return frames


def _fmt_board(board: list[dict[str, Any]]) -> str:
    if not board:
        return "(vide)"
    parts = []
    for b in board:
        st = "[R]" if b.get("rested") else "[A]"
        parts.append(f"{b.get('id','?')} {b.get('power',0)}{st}")
    return " | ".join(parts)


def _kw_from_manifest(manifest: dict[str, Any], cid: str) -> list[str]:
    meta = manifest.get(cid) or manifest.get(str(cid).strip().upper()) or {}
    raw = meta.get("keywords")
    if raw:
        return [str(x) for x in raw]
    return list(extract_keywords_from_text(str(meta.get("card_text") or "")))


def _aggregate_keywords_matplotlib(fr: dict[str, Any], manifest: dict[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    p0, p1 = fr.get("p0") or {}, fr.get("p1") or {}

    def add(cid: str) -> None:
        if not cid or cid == "?":
            return
        for kw in _kw_from_manifest(manifest, str(cid)):
            if kw not in seen:
                seen.add(kw)
                out.append(kw)

    for cid in p0.get("hand") or []:
        add(str(cid))
    for cid in p1.get("hand") or []:
        add(str(cid))
    for b in p0.get("board") or []:
        add(str(b.get("id", "")))
    for b in p1.get("board") or []:
        add(str(b.get("id", "")))
    return tuple(out[:36])


def _draw_frame(ax: plt.Axes, fr: dict[str, Any], manifest: dict[str, Any] | None = None) -> None:
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    p0 = fr.get("p0") or {}
    p1 = fr.get("p1") or {}
    phase = fr.get("phase", "?")
    title = f"Frame {fr.get('frame', '?')} — {fr.get('kind', '?')} — phase {phase}"
    if fr.get("kind") == "step":
        title += f"\nAction {fr.get('action')} — {fr.get('action_desc', '')}"
        title += f"\nreward={fr.get('reward', 0):.3f} terminated={fr.get('terminated', False)}"
    if fr.get("winner") is not None:
        title += f"\nGagnant : joueur {fr['winner']} (0 = agent)"

    ax.text(0.5, 0.97, title, ha="center", va="top", fontsize=10, wrap=True)

    # Agent (bas)
    y0 = 0.52
    ax.add_patch(
        plt.Rectangle((0.04, y0), 0.42, 0.42, fill=True, facecolor="#1a3a5c", alpha=0.35, zorder=0)
    )
    ax.text(0.05, y0 + 0.38, "AGENT (P0)", fontsize=11, fontweight="bold", color="#cce")
    ax.text(
        0.05,
        y0 + 0.30,
        f"Vie : {p0.get('life', 0)}  |  DON actif {p0.get('don_active', p0.get('don', 0))} · reposé {p0.get('don_rested', 0)}  |  deck : {p0.get('deck_remaining', '?')}",
        fontsize=10,
        family="monospace",
        color="white",
    )
    hand0 = ", ".join(p0.get("hand") or [])
    ax.text(
        0.05,
        y0 + 0.18,
        f"Main : {hand0[:120]}{'…' if len(hand0) > 120 else ''}",
        fontsize=8,
        family="monospace",
        color="#eee",
    )
    ax.text(
        0.05,
        y0 + 0.06,
        f"Terrain : {_fmt_board(p0.get('board') or [])}",
        fontsize=8,
        family="monospace",
        color="#9cf",
    )

    # Adversaire (haut)
    y1 = 0.06
    ax.add_patch(
        plt.Rectangle((0.54, y1), 0.42, 0.42, fill=True, facecolor="#5c1a1a", alpha=0.35, zorder=0)
    )
    ax.text(0.55, y1 + 0.38, "ADVERSAIRE (P1)", fontsize=11, fontweight="bold", color="#fcc")
    h1 = p1.get("hand") or []
    h1s = ", ".join(h1) if h1 else f"({p1.get('hand_count', 0)} c., replay sans liste)"
    ax.text(
        0.55,
        y1 + 0.30,
        f"Vie : {p1.get('life', 0)}  |  DON actif {p1.get('don_active', p1.get('don', 0))} · reposé {p1.get('don_rested', 0)}  |  deck : {p1.get('deck_remaining', '?')}",
        fontsize=10,
        family="monospace",
        color="white",
    )
    ax.text(
        0.55,
        y1 + 0.22,
        f"Main P1 : {h1s[:100]}{'…' if len(h1s) > 100 else ''}",
        fontsize=8,
        family="monospace",
        color="#fdd",
    )
    ax.text(
        0.55,
        y1 + 0.12,
        f"Terrain : {_fmt_board(p1.get('board') or [])}",
        fontsize=8,
        family="monospace",
        color="#f99",
    )

    # Barres de vie
    max_life = max(p0.get("life", 0), p1.get("life", 0), 5)
    ax.barh(
        [0.96],
        [p0.get("life", 0) / max_life * 0.45],
        height=0.02,
        left=0.04,
        color="#4488ff",
        label="P0 life",
    )
    ax.barh(
        [0.96],
        [p1.get("life", 0) / max_life * 0.45],
        height=0.02,
        left=0.54,
        color="#ff6644",
        label="P1 life",
    )

    raw_list = list(_aggregate_keywords_matplotlib(fr, manifest or {}))
    if not raw_list:
        raw_list = list(extract_keywords_from_text(str(fr.get("action_desc", ""))))
    seen2: set[str] = set()
    uniq: list[str] = []
    for k in raw_list:
        if k not in seen2:
            seen2.add(k)
            uniq.append(k)
    if uniq:
        ax.text(
            0.5,
            0.01,
            "Marqueurs : " + format_keywords_line(tuple(uniq), max_len=160),
            ha="center",
            va="bottom",
            fontsize=7,
            family="monospace",
            color="#cca",
        )


def jsonl_to_gif(
    jsonl_path: Path,
    gif_path: Path,
    *,
    fps: float = 1.5,
    dpi: int = 100,
    manifest_path: Path | None = None,
) -> None:
    frames = load_jsonl_frames(jsonl_path)
    if not frames:
        raise ValueError(f"Aucune frame dans {jsonl_path}")

    manifest: dict[str, Any] = {}
    if manifest_path is None:
        cand = jsonl_path.with_name(jsonl_path.stem + ".manifest.json")
        if cand.is_file():
            manifest_path = cand
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(gif_path), dpi=dpi):
        for fr in frames:
            _draw_frame(ax, fr, manifest)
            writer.grab_frame()
    plt.close(fig)
