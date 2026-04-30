"""
Téléchargement des métadonnées One Piece Card Game depuis tcgcsv.com (miroir JSON TCGplayer).

API documentée : https://tcgcsv.com/docs — catégorie OP = ``category_id`` 68 (vérifiable via ``/tcgplayer/categories``).
"""
from __future__ import annotations

import html
import json
import re
import time
from typing import Any, Iterable

import requests

TCGCSV_BASE = "https://tcgcsv.com/tcgplayer"
DEFAULT_OP_CATEGORY_ID = 68
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": "opctcg_text_sim/0.1 (+https://github.com) tcgcsv-fetch",
        "Accept": "application/json",
    }
)


def strip_html(s: str) -> str:
    if not s:
        return ""
    t = html.unescape(s)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _get_json(url: str, timeout: float = 60.0) -> dict[str, Any]:
    r = _SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return json.loads(r.text)


def list_categories() -> list[dict[str, Any]]:
    d = _get_json(f"{TCGCSV_BASE}/categories")
    return list(d.get("results") or [])


def find_one_piece_category_id() -> int:
    for c in list_categories():
        name = (c.get("name") or "") + " " + (c.get("displayName") or "")
        if "one piece" in name.lower():
            return int(c["categoryId"])
    return DEFAULT_OP_CATEGORY_ID


def list_groups(category_id: int) -> list[dict[str, Any]]:
    d = _get_json(f"{TCGCSV_BASE}/{category_id}/groups")
    return list(d.get("results") or [])


def list_products(category_id: int, group_id: int) -> list[dict[str, Any]]:
    d = _get_json(f"{TCGCSV_BASE}/{category_id}/{group_id}/products")
    return list(d.get("results") or [])


def extended_dict(product: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in product.get("extendedData") or []:
        k = item.get("name")
        if k:
            out[str(k)] = str(item.get("value") or "")
    return out


def _parse_int(s: str | None) -> int:
    if not s:
        return 0
    s = str(s).strip().replace(",", "")
    if not s or s.upper() == "NAN":
        return 0
    try:
        return int(float(s))
    except ValueError:
        m = re.search(r"-?\d+", s)
        return int(m.group(0)) if m else 0


def product_to_row(
    product: dict[str, Any],
    *,
    group_name: str = "",
) -> dict[str, Any] | None:
    """
    Convertit un produit TCGplayer (JSON tcgcsv) en ligne CSV sim / ``None`` si ce n’est pas une carte (pas de Number).
    """
    ed = extended_dict(product)
    num = (ed.get("Number") or "").strip().upper()
    if not num:
        return None

    desc = strip_html(ed.get("Description") or "")
    cost = _parse_int(ed.get("Cost"))
    power = _parse_int(ed.get("Power"))
    counter = _parse_int(ed.get("Counterplus"))

    img = (product.get("imageUrl") or "").strip()
    if img and not img.lower().startswith("http"):
        img = ""

    return {
        "card_id": num,
        "name": (product.get("name") or ed.get("Number") or "").strip(),
        "cost": cost,
        "power": power,
        "counter": counter,
        "color": (ed.get("Color") or "").strip(),
        "image_url": img,
        "card_text": desc,
        "rarity": (ed.get("Rarity") or "").strip(),
        "card_type": (ed.get("CardType") or "").strip(),
        "life": _parse_int(ed.get("Life")),
        "product_id": product.get("productId", ""),
        "group_name": group_name,
    }


def fetch_all_card_rows(
    category_id: int = DEFAULT_OP_CATEGORY_ID,
    *,
    group_ids: Iterable[int] | None = None,
    sleep_s: float = 0.08,
    on_group: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Parcourt les groupes (extensions) et fusionne les cartes par ``card_id`` (Number).
    En cas de doublon, la dernière occurrence l’emporte.
    """
    groups = list_groups(category_id)
    if group_ids is not None:
        wanted = set(int(x) for x in group_ids)
        groups = [g for g in groups if int(g["groupId"]) in wanted]

    merged: dict[str, dict[str, Any]] = {}
    for g in groups:
        gid = int(g["groupId"])
        gname = str(g.get("name") or "")
        if on_group:
            on_group(gid, gname, len(groups))
        try:
            products = list_products(category_id, gid)
        except requests.RequestException:
            continue
        for p in products:
            row = product_to_row(p, group_name=gname)
            if row:
                merged[row["card_id"]] = row
        if sleep_s > 0:
            time.sleep(sleep_s)
    return merged
