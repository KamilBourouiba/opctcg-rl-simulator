"""
GIF animé à partir du JSONL + manifeste cartes (URLs tcgcsv / TCGplayer).
Disposition type plateau OP : deck, deck DON!!, zone de coût, vie, leader,
mains des deux joueurs (cartes visibles en replay), terrain Characters.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .card_images import load_or_placeholder
from .card_keywords import extract_keywords_from_text, format_keywords_line
from .replay_render import load_jsonl_frames

# Résolution de référence (anciennes valeurs) pour adapter le placement au format choisi.
_REF_W, _REF_H = 1680, 1040
# Par défaut : cadre plus grand + vignettes plus lisibles (MP4 / GIF images).
DEFAULT_FRAME_SIZE: tuple[int, int] = (2400, 1400)
DEFAULT_THUMB: tuple[int, int] = (106, 148)


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _keywords_for_card(manifest: dict[str, dict[str, Any]], cid: str) -> list[str]:
    if not cid or cid == "?":
        return []
    meta = manifest.get(cid) or manifest.get(str(cid).strip().upper())
    if not meta:
        return []
    raw = meta.get("keywords")
    if raw:
        return [str(x) for x in raw]
    return list(extract_keywords_from_text(str(meta.get("card_text") or "")))


def _aggregate_visible_keywords(
    fr: dict[str, Any],
    manifest: dict[str, dict[str, Any]],
    *,
    max_tokens: int = 42,
) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    p0, p1 = fr.get("p0") or {}, fr.get("p1") or {}

    def _consume(cid: str) -> None:
        for kw in _keywords_for_card(manifest, cid):
            if kw not in seen:
                seen.add(kw)
                out.append(kw)

    for lid in (p0.get("leader_id"), p1.get("leader_id")):
        if lid:
            _consume(str(lid))
    for cid in list(p0.get("hand") or []):
        _consume(str(cid))
    for cid in list(p1.get("hand") or []):
        _consume(str(cid))
    for b in p0.get("board") or []:
        _consume(str(b.get("id", "")))
    for b in p1.get("board") or []:
        _consume(str(b.get("id", "")))
    return tuple(out[:max_tokens])


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _paste_row(
    canvas: Image.Image,
    y: int,
    x0: int,
    card_ids: list[str],
    manifest: dict[str, dict[str, Any]],
    cache_dir: Path,
    thumb: tuple[int, int],
    *,
    face_down: bool = False,
    max_cards: int = 14,
) -> None:
    x = x0
    gap = 6
    for cid in card_ids[:max_cards]:
        if face_down:
            im = load_or_placeholder("?", None, cache_dir, thumb)
            dr = ImageDraw.Draw(im)
            dr.rectangle((4, 4, thumb[0] - 5, thumb[1] - 5), outline=(90, 90, 110), width=2)
        else:
            meta = manifest.get(cid, {})
            url = meta.get("image_url") or None
            im = load_or_placeholder(cid, url, cache_dir, thumb)
        canvas.paste(im, (x, y), im if im.mode == "RGBA" else None)
        x += thumb[0] + gap


def _paste_board_row(
    canvas: Image.Image,
    y: int,
    x0: int,
    board: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    cache_dir: Path,
    thumb: tuple[int, int],
) -> None:
    x = x0
    gap = 6
    for b in board:
        cid = str(b.get("id", "?"))
        meta = manifest.get(cid, {})
        url = meta.get("image_url") or None
        im = load_or_placeholder(cid, url, cache_dir, thumb)
        drim = ImageDraw.Draw(im)
        if b.get("rested"):
            drim.line([(4, 4), (thumb[0] - 4, thumb[1] - 4)], fill=(255, 80, 80), width=3)
        if b.get("has_rush"):
            drim.rounded_rectangle((thumb[0] - 22, 4, thumb[0] - 4, 20), radius=3, fill=(180, 40, 40))
            drim.text((thumb[0] - 18, 5), "R", fill=(255, 255, 255), font=_font(11))
        if b.get("has_blocker"):
            drim.rounded_rectangle((4, 4, 22, 20), radius=3, fill=(40, 90, 200))
            drim.text((7, 5), "B", fill=(255, 255, 255), font=_font(11))
        canvas.paste(im, (x, y), im if im.mode == "RGBA" else None)
        pw = int(b.get("power_effective", b.get("power", 0)))
        dr2 = ImageDraw.Draw(canvas)
        dr2.text((x + 2, y + thumb[1] - 14), str(pw), fill=(255, 255, 200), font=_font(10))
        x += thumb[0] + gap


def _draw_life_pile(
    dr: ImageDraw.ImageDraw,
    x: int,
    y: int,
    life: int,
    *,
    max_show: int = 8,
    w: int = 24,
    h: int = 16,
    fill: tuple[int, int, int] = (52, 120, 82),
    outline: tuple[int, int, int] = (140, 200, 160),
) -> None:
    """Repli si le JSONL n’a pas ``life_cards`` (anciens replays)."""
    dr.text((x, y - 22), "VIE", fill=(220, 255, 220), font=_font(12))
    dr.text((x + 2, y - 6), str(max(0, life)), fill=(255, 255, 255), font=_font(22))
    n = min(max(0, life), max_show)
    base_y = y + 18
    for i in range(n):
        yy = base_y - i * 6
        dr.rounded_rectangle((x, yy, x + w, yy + h), radius=3, fill=fill, outline=outline, width=1)
    if life > max_show:
        dr.text((x, base_y + h + 4), f"+{life - max_show}", fill=(200, 220, 200), font=_font(10))


def _paste_card_row(
    canvas: Image.Image,
    y: int,
    x0: int,
    card_ids: list[str],
    manifest: dict[str, dict[str, Any]],
    cache_dir: Path,
    thumb: tuple[int, int],
    *,
    max_cards: int = 12,
    gap: int = 4,
) -> int:
    """Colle une rangée de cartes ; retourne la largeur utilisée (pour alignement)."""
    x = x0
    for cid in card_ids[:max_cards]:
        meta = manifest.get(cid, {})
        url = meta.get("image_url") or None
        im = load_or_placeholder(cid, url, cache_dir, thumb)
        canvas.paste(im, (x, y), im if im.mode == "RGBA" else None)
        x += thumb[0] + gap
    return x - x0 - gap if x > x0 else 0


def _draw_deck_stack(
    dr: ImageDraw.ImageDraw,
    x: int,
    y: int,
    remaining: int,
    label: str,
) -> None:
    dr.rounded_rectangle((x, y, x + 52, y + 72), radius=4, fill=(32, 38, 52), outline=(100, 110, 140), width=2)
    dr.line((x + 8, y + 12, x + 44, y + 60), fill=(70, 80, 100), width=2)
    dr.text((x + 6, y + 76), label, fill=(180, 190, 210), font=_font(11))
    dr.text((x + 10, y + 90), str(remaining), fill=(255, 255, 240), font=_font(14))


def _draw_don_deck_indicator(
    dr: ImageDraw.ImageDraw,
    x: int,
    y: int,
    official: int = 10,
    remaining: int | None = None,
) -> None:
    rem = official if remaining is None else max(0, min(remaining, official))
    dr.text((x, y), "DON!! deck", fill=(240, 210, 120), font=_font(11))
    dr.text((x + 92, y), f"reste {rem}", fill=(255, 235, 160), font=_font(12))
    yy = y + 18
    chip_w, chip_h, gap = 12, 16, 3
    for i in range(official):
        still = i < rem
        dr.rounded_rectangle(
            (x + i * (chip_w + gap), yy, x + i * (chip_w + gap) + chip_w, yy + chip_h),
            radius=2,
            fill=(55, 52, 38) if still else (24, 26, 32),
            outline=(140, 120, 50) if still else (55, 58, 70),
            width=1,
        )


def _draw_don_cost_area(
    dr: ImageDraw.ImageDraw,
    x: int,
    y: int,
    don_active: int,
    don_rested: int,
    don_cap: int,
) -> None:
    dr.text(
        (x, y),
        "DON!! zone de coût — jetons A = actifs (payables), R = reposés (coût payé)",
        fill=(255, 230, 150),
        font=_font(10),
    )
    yy = y + 18
    chip_w, chip_h, gap = 14, 18, 4
    cap = max(1, min(don_cap, 12))
    total_in_cost = min(don_active + don_rested, cap)
    f_chip = _font(9)
    for i in range(cap):
        cx = x + i * (chip_w + gap)
        if i < min(don_active, cap):
            fill = (218, 175, 55)
            out = (255, 210, 80)
            mark = "A"
            mcol = (40, 30, 10)
        elif i < total_in_cost:
            fill = (72, 68, 88)
            out = (130, 120, 160)
            mark = "R"
            mcol = (230, 225, 250)
        else:
            fill = (45, 48, 58)
            out = (80, 85, 100)
            mark = ""
            mcol = (120, 125, 140)
        dr.rounded_rectangle(
            (cx, yy, cx + chip_w, yy + chip_h),
            radius=3,
            fill=fill,
            outline=out,
            width=1,
        )
        if mark:
            dr.text((cx + 3, yy + 3), mark, fill=mcol, font=f_chip)
    dr.text(
        (x, yy + chip_h + 6),
        f"compte : {don_active} actifs · {don_rested} reposés · plafond {don_cap}",
        fill=(220, 220, 235),
        font=_font(10),
    )


def _draw_leader_zone(
    im: Image.Image,
    dr: ImageDraw.ImageDraw,
    x: int,
    y: int,
    title: str,
    life_val: int,
    accent: tuple[int, int, int],
    *,
    leader_id: str | None = None,
    leader_power: int = 0,
    leader_display_power: int | None = None,
    leader_rested: bool = False,
    leader_attached_don: int = 0,
    manifest: dict[str, dict[str, Any]] | None = None,
    cache_dir: Path | None = None,
) -> None:
    # Fond : rouge tamisé si reposé, normal sinon
    bg = (52, 28, 28) if leader_rested else (28, 34, 48)
    bdr = (200, 80, 80) if leader_rested else accent
    dr.rounded_rectangle((x, y, x + 120, y + 130), radius=6, fill=bg, outline=bdr, width=2)
    lbl = "LEADER (REPOSÉ)" if leader_rested else "LEADER"
    dr.text((x + 6, y + 8), lbl, fill=bdr, font=_font(9))
    eff_pow = (
        int(leader_display_power)
        if leader_display_power is not None
        else leader_power + 1000 * leader_attached_don
    )
    don_txt = f" (+{leader_attached_don}DON)" if leader_attached_don else ""
    dr.text((x + 6, y + 22), f"{eff_pow:,}{don_txt}", fill=(255, 230, 140), font=_font(11))
    dr.text((x + 6, y + 40), title[:14], fill=(230, 235, 245), font=_font(11))
    dr.text((x + 6, y + 56), f"Vie {life_val}", fill=(180, 220, 255), font=_font(11))
    # Vignette carte Leader
    if leader_id and manifest is not None and cache_dir is not None:
        lid = str(leader_id).strip().upper()
        meta = manifest.get(lid, {})
        url = meta.get("image_url")
        tw, th = 44, 62
        lim = load_or_placeholder(lid, url, cache_dir, (tw, th))
        px, py = x + 72, y + 62
        if px + tw <= x + 118 and py + th <= y + 128:
            # Superposition d'un filtre sombre si reposé
            if leader_rested:
                from PIL import ImageEnhance
                lim = ImageEnhance.Brightness(lim.convert("RGB")).enhance(0.55).convert("RGBA")
            im.paste(lim, (px, py), lim if lim.mode == "RGBA" else None)


def _render_player_half(
    im: Image.Image,
    dr: ImageDraw.ImageDraw,
    *,
    y0: int,
    y1: int,
    label: str,
    accent: tuple[int, int, int],
    panel_bg: tuple[int, int, int],
    p: dict[str, Any],
    manifest: dict[str, dict[str, Any]],
    cache_dir: Path,
    thumb: tuple[int, int],
    meta: dict[str, Any],
    max_don: int,
) -> None:
    dr.rounded_rectangle((10, y0, im.width - 10, y1), radius=10, fill=panel_bg, outline=accent, width=1)
    dr.text((24, y0 + 8), label, fill=accent, font=_font(16))

    deck_n = int(p.get("deck_remaining", 0))
    life_n = int(p.get("life", 0))
    life_cards: list[str] = [str(x) for x in (p.get("life_cards") or []) if x]
    trash_ids: list[str] = [str(x) for x in (p.get("trash") or []) if x]
    don = int(p.get("don_active", p.get("don", 0)))
    don_rested = int(p.get("don_rested", 0))
    don_deck = int(p.get("don_deck", meta.get("official_don_deck", 10)))
    hand: list[str] = list(p.get("hand") or [])
    if not hand and int(p.get("hand_count") or 0) > 0:
        hand = ["?"] * min(int(p.get("hand_count")), 14)

    x_col, x_ld, x_play = _player_half_x_layout(im.width, thumb[0])
    _draw_deck_stack(dr, x_col, y0 + 36, deck_n, "Deck")
    _draw_don_deck_indicator(
        dr,
        x_col,
        y0 + 138,
        official=int(p.get("official_don_deck", meta.get("official_don_deck", 10))),
        remaining=don_deck,
    )

    lid = p.get("leader_id")
    leader_id = str(lid).strip().upper() if lid else None
    leader_name = ""
    if leader_id and manifest:
        leader_name = manifest.get(leader_id, {}).get("name", leader_id[:10])
    _draw_leader_zone(
        im,
        dr,
        x_ld,
        y0 + 36,
        leader_name or label.replace("(", "").replace(")", "")[:14],
        life_n,
        accent,
        leader_id=leader_id,
        leader_power=int(p.get("leader_power", 5000)),
        leader_display_power=(
            int(p["leader_power_effective"])
            if p.get("leader_power_effective") is not None
            else None
        ),
        leader_rested=bool(p.get("leader_rested", False)),
        leader_attached_don=int(p.get("leader_attached_don", 0)),
        manifest=manifest,
        cache_dir=cache_dir,
    )

    don_cap = int(meta.get("max_don", max_don))
    _draw_don_cost_area(dr, x_play, y0 + 36, don, don_rested, don_cap)

    life_thumb = (min(52, thumb[0]), min(72, thumb[1]))
    y_zone = y0 + 168
    dr.text(
        (x_play, y_zone - 20),
        "Vie — dessus à gauche (face visible)",
        fill=(160, 220, 180),
        font=_font(11),
    )
    if life_cards:
        _paste_card_row(
            im,
            y_zone,
            x_play,
            life_cards,
            manifest,
            cache_dir,
            life_thumb,
            max_cards=10,
        )
    else:
        _draw_life_pile(dr, x_play, y_zone + 8, life_n)

    y_tr = y_zone + life_thumb[1] + 14
    dr.text((x_play, y_tr - 18), "Défausse (dernières cartes)", fill=(220, 170, 170), font=_font(11))
    if trash_ids:
        _paste_card_row(
            im,
            y_tr,
            x_play,
            trash_ids,
            manifest,
            cache_dir,
            life_thumb,
            max_cards=14,
        )
    else:
        dr.text((x_play, y_tr + 8), "(vide)", fill=(130, 135, 155), font=_font(11))

    y_hand = y_tr + life_thumb[1] + 26
    dr.text((x_play, y_hand - 18), "Main", fill=(200, 210, 230), font=_font(12))
    _paste_row(im, y_hand, x_play, hand, manifest, cache_dir, thumb, face_down=False, max_cards=14)

    y_board = y_hand + thumb[1] + 22
    dr.text((x_play, y_board - 18), "Terrain (Characters)", fill=(200, 210, 230), font=_font(12))
    _paste_board_row(im, y_board, x_play, p.get("board") or [], manifest, cache_dir, thumb)


def _player_half_x_layout(im_w: int, thumb_w: int) -> tuple[int, int, int]:
    """Colonnes deck / leader / zone de jeu (main, terrain) selon la largeur totale."""
    gap_cards = 6
    hand_block = 14 * (thumb_w + gap_cards)
    margin_right = 24
    x_col = max(16, min(28, int(im_w * 0.014)))
    x_ld = max(118, int(im_w * 0.065))
    leader_guard = x_ld + 118
    room_right = im_w - hand_block - margin_right
    upper = max(leader_guard, room_right)
    pref = max(300, int(im_w * 0.195))
    x_play = min(upper, max(leader_guard, pref))
    return x_col, x_ld, x_play


def render_frame_image(
    fr: dict[str, Any],
    manifest: dict[str, dict[str, Any]],
    cache_dir: Path,
    *,
    size: tuple[int, int] | None = None,
    thumb: tuple[int, int] | None = None,
) -> Image.Image:
    if size is None:
        size = DEFAULT_FRAME_SIZE
    if thumb is None:
        thumb = DEFAULT_THUMB
    w, h = size
    bg_top = (22, 26, 38)
    bg_bot = (20, 28, 42)
    im = Image.new("RGB", (w, h), (14, 16, 22))
    dr = ImageDraw.Draw(im)
    dr.rectangle((0, 0, w, h // 2), fill=bg_top)
    dr.rectangle((0, h // 2, w, h), fill=bg_bot)

    scale = max(1.0, min(1.35, ((w / _REF_W) * (h / _REF_H)) ** 0.25))
    font = _font(int(round(17 * scale)))
    font_s = _font(int(round(13 * scale)))
    p0 = fr.get("p0") or {}
    p1 = fr.get("p1") or {}
    phase = fr.get("phase", "?")
    meta = fr.get("sim_meta") or {}
    max_don = int(meta.get("max_don", 10))

    title = f"Frame {fr.get('frame')}  |  {fr.get('kind')}  |  Phase sim : {phase}"
    dr.text((24, 10), title, fill=(235, 238, 248), font=font)
    sub = meta.get("rules_version", "Comprehensive Rules — sim réduite")
    dr.text((24, 34), sub, fill=(160, 175, 210), font=font_s)
    turn_hint = meta.get("turn_hint")
    if isinstance(turn_hint, str) and turn_hint.strip():
        wrap = max(60, int(w * 0.055))
        dr.text(
            (24, 52),
            turn_hint.strip()[:wrap] + ("…" if len(turn_hint.strip()) > wrap else ""),
            fill=(175, 190, 225),
            font=_font(int(round(10 * scale))),
        )

    if fr.get("kind") == "step":
        line = (
            f"Action {fr.get('action')} — {str(fr.get('action_desc', ''))[:180]}  |  "
            f"r={float(fr.get('reward', 0)):.3f}"
        )
        dr.text((max(420, int(w * 0.22)), 34), line, fill=(190, 205, 255), font=font_s)
    if fr.get("winner") is not None:
        y_win = 72 if isinstance(turn_hint, str) and turn_hint.strip() else 58
        dr.text(
            (w // 2 - 120, y_win),
            f"Gagnant joueur {fr['winner']} (0 = vous)",
            fill=(255, 220, 120),
            font=_font(16),
        )

    # Moitié haute = adversaire (comme en face sur un tapis)
    y_split = 118
    y_mid = h // 2 - 24
    y_low = h - 24

    _render_player_half(
        im,
        dr,
        y0=y_split,
        y1=y_mid - 8,
        label="ADVERSAIRE (P1)",
        accent=(255, 140, 120),
        panel_bg=(30, 26, 34),
        p=p1,
        manifest=manifest,
        cache_dir=cache_dir,
        thumb=thumb,
        meta=meta,
        max_don=max_don,
    )

    dr.rectangle((8, y_mid - 4, w - 8, y_mid + 28), fill=(12, 14, 20), outline=(70, 90, 130), width=1)
    phases = "Officiel : Refresh → Draw → DON!! → Main → End"
    dr.text((20, y_mid + 2), phases, fill=(150, 170, 210), font=_font(int(round(11 * scale))))
    bd = meta.get("battle_damage_allowed")
    if bd is not None:
        dr.text(
            (max(700, int(w * 0.38)), y_mid + 2),
            f"Dégâts combat : {'OK' if bd else 'interdits (1er cycle)'}",
            fill=(255, 200, 120) if not bd else (140, 255, 160),
            font=_font(int(round(11 * scale))),
        )

    _render_player_half(
        im,
        dr,
        y0=y_mid + 36,
        y1=y_low - 8,
        label="VOUS (P0)",
        accent=(120, 190, 255),
        panel_bg=(22, 30, 44),
        p=p0,
        manifest=manifest,
        cache_dir=cache_dir,
        thumb=thumb,
        meta=meta,
        max_don=max_don,
    )

    kws = _aggregate_visible_keywords(fr, manifest)
    if kws:
        kw_max = max(320, int(w * 0.42))
        kw_wrap = max(96, int(w // 19))
        kw_line = format_keywords_line(kws, max_len=kw_max)
        dr.text((24, h - 58), "Marqueurs visibles (mains + terrain) :", fill=(210, 200, 140), font=_font(10))
        for i, chunk in enumerate([kw_line[j : j + kw_wrap] for j in range(0, len(kw_line), kw_wrap)][:4]):
            dr.text((24, h - 42 + i * 14), chunk, fill=(230, 225, 190), font=_font(9))

    doc = meta.get("rules_doc", "data/OPTCG_GameRules_Reference.md")
    kwdoc = "data/OPTCG_Keywords_Reference.md"
    dr.text(
        (24, h - 14),
        f"PDF rule_comprehensive.pdf  ·  {doc}  ·  {kwdoc}  ·  DON +{meta.get('don_per_turn', 2)}/tour (max {max_don})",
        fill=(120, 130, 155),
        font=_font(9),
    )
    return im


def jsonl_to_gif_with_images(
    jsonl_path: Path,
    gif_path: Path,
    manifest_path: Path,
    cache_dir: Path,
    *,
    fps: float = 1.5,
    frame_size: tuple[int, int] | None = None,
    thumb_size: tuple[int, int] | None = None,
) -> None:
    frames = load_jsonl_frames(jsonl_path)
    manifest = load_manifest(manifest_path)
    if not frames:
        raise ValueError(f"Aucune frame : {jsonl_path}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    pil_frames: list[Image.Image] = []
    for fr in frames:
        pil_frames.append(
            render_frame_image(fr, manifest, cache_dir, size=frame_size, thumb=thumb_size)
        )

    duration_ms = int(1000 / max(fps, 0.1))
    pil_frames[0].save(
        gif_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def jsonl_to_mp4_with_images(
    jsonl_path: Path,
    mp4_path: Path,
    manifest_path: Path,
    cache_dir: Path,
    *,
    fps: float = 1.5,
    frame_size: tuple[int, int] | None = None,
    thumb_size: tuple[int, int] | None = None,
) -> None:
    """Encode une vidéo H.264 (MP4) ; nécessite ``imageio`` + ``imageio-ffmpeg``.

    Les frames sont encodées au fil de l’eau (pas de liste de toutes les images en RAM).
    """
    try:
        import imageio.v2 as imageio
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Installez : pip install imageio imageio-ffmpeg"
        ) from e

    frames = load_jsonl_frames(jsonl_path)
    manifest = load_manifest(manifest_path)
    if not frames:
        raise ValueError(f"Aucune frame : {jsonl_path}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    def _even_rgb(im: Image.Image) -> np.ndarray:
        w, h = im.size
        if w % 2:
            im = im.crop((0, 0, w - 1, h))
        if im.size[1] % 2:
            im = im.crop((0, 0, im.size[0], im.size[1] - 1))
        return np.ascontiguousarray(np.asarray(im.convert("RGB")))

    writer = imageio.get_writer(
        str(mp4_path),
        format="FFMPEG",
        mode="I",
        fps=max(fps, 0.25),
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_log_level="error",
        input_params=["-probesize", "32M", "-analyzeduration", "10M"],
    )
    _min_mp4_frames = 24
    last_arr: np.ndarray | None = None
    n_out = 0
    try:
        for fr in frames:
            im = render_frame_image(fr, manifest, cache_dir, size=frame_size, thumb=thumb_size)
            last_arr = _even_rgb(im)
            writer.append_data(last_arr)
            n_out += 1
        # FFmpeg : flux très courts
        if last_arr is not None:
            pad = np.ascontiguousarray(last_arr.copy())
            while n_out < _min_mp4_frames:
                writer.append_data(np.ascontiguousarray(pad.copy()))
                n_out += 1
    finally:
        writer.close()
