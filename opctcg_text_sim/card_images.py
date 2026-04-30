"""
Téléchargement / cache des visuels de cartes (URLs type export TCGplayer / tcgcsv).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": "opctcg_text_sim/0.1 (+https://github.com) image-cache",
        "Accept": "image/*,*/*;q=0.8",
    }
)


def _safe_filename(url: str) -> str:
    h = hashlib.sha256(url.encode()).hexdigest()[:24]
    suf = Path(urlparse(url).path).suffix.lower()
    if suf not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        suf = ".img"
    return f"{h}{suf}"


def fetch_image_to_file(url: str, dest: Path, timeout: float = 25.0) -> bool:
    try:
        r = _SESSION.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return True
    except OSError:
        return False
    except requests.RequestException:
        return False


def ensure_cached_image(
    card_id: str,
    url: str | None,
    cache_dir: Path,
) -> Path | None:
    """
    Retourne le chemin d’une image locale pour ``card_id``.
    Si ``url`` est vide ou téléchargement impossible, retourne None (placeholder PIL).
    """
    if not url or not str(url).strip().startswith("http"):
        return None
    u = str(url).strip()
    dest = cache_dir / _safe_filename(u)
    if dest.is_file() and dest.stat().st_size > 32:
        return dest
    if fetch_image_to_file(u, dest):
        return dest
    return None


def placeholder_card_image(card_id: str, size: tuple[int, int] = (140, 196)) -> Any:
    """Image RGB sans URL (dépendance Pillow lazy)."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = size
    im = Image.new("RGB", (w, h), (40, 44, 58))
    dr = ImageDraw.Draw(im)
    dr.rectangle((2, 2, w - 3, h - 3), outline=(120, 130, 160), width=2)
    txt = card_id[:14]
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    tw, th = dr.textbbox((0, 0), txt, font=font)[2:4]
    dr.text(((w - tw) // 2, (h - th) // 2), txt, fill=(200, 210, 230), font=font)
    return im


def load_or_placeholder(card_id: str, url: str | None, cache_dir: Path, thumb: tuple[int, int]):
    from PIL import Image

    p = ensure_cached_image(card_id, url, cache_dir)
    if p is not None:
        try:
            im = Image.open(p).convert("RGBA")
        except OSError:
            im = placeholder_card_image(card_id, thumb).convert("RGBA")
    else:
        im = placeholder_card_image(card_id, thumb).convert("RGBA")
    im.thumbnail(thumb, Image.Resampling.LANCZOS)
    return im
