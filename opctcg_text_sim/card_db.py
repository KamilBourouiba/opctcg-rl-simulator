from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .card_keywords import extract_keywords_from_text


def infer_leader_don_deck_size(card_text: str) -> int:
    """
    Taille du deck DON!! pour un Leader (10 par défaut ; ex. Enel OP15 : 6).
    Dérivé du texte de règle officiel (« DON!! deck consists of N cards »).
    """
    if not card_text:
        return 10
    t = card_text.lower()
    for pat in (
        r"don!!\s*deck\s+consists\s+of\s+(\d+)\s+cards?",
        r"don!!\s*deck\s+has\s+(\d+)\s+cards?",
        r"don!!\s*deck\s+of\s+(\d+)\s+cards?",
    ):
        m = re.search(pat, t)
        if m:
            return max(1, min(20, int(m.group(1))))
    return 10


@dataclass
class CardDef:
    card_id: str
    name: str
    cost: int
    power: int
    counter: int
    color: str
    raw_text: str = ""
    # Texte de règle / effet (export tcgcsv : Description)
    card_text: str = ""
    # URL visuel (export tcgcsv / TCGplayer : colonne imageUrl, photoUrl, …)
    image_url: str | None = None
    # Marqueurs extraits du texte : [On Play], [Counter], {Type}, <Attribute>, …
    keywords: tuple[str, ...] = ()
    # tcgcsv : Character, Event, Stage, Leader (vide si colonne absente)
    card_type: str = ""
    # Valeur de vie du Leader (CR 2-9-1) — 0 pour les cartes non-Leader
    life: int = 0
    # Leader : nombre de cartes dans le deck DON!! (10 par défaut ; certains 6, etc.)
    don_deck_size: int = 10
    # Pré-calcul pour le masque d’actions (évite scans texte à chaque step)
    has_activate_main: bool = False
    # Coût « DON!! -N » associé à [Activate: Main] si présent dans le texte (0 sinon)
    activate_main_don_minus: int = 0

    def embedding(self, dim: int = 8) -> np.ndarray:
        """Petit vecteur déterministe à partir de l’id (pour l’observation réseau)."""
        v = np.zeros(dim, dtype=np.float32)
        h = abs(hash(self.card_id)) % (256 * dim)
        for i in range(dim):
            v[i] = ((h >> (i * 3)) & 255) / 255.0
        return v


def _pick_col(df: pd.DataFrame, logical: str, column_map: dict[str, str], fallbacks: list[str]) -> str | None:
    if logical in column_map and column_map[logical] in df.columns:
        return column_map[logical]
    for f in fallbacks:
        if f in df.columns:
            return f
    return None


def _to_int(x: object, default: int = 0) -> int:
    s = str(x).strip()
    if not s or s.upper() == "NAN":
        return default
    try:
        return int(float(s.replace(",", ".")))
    except ValueError:
        return default


def load_card_csv(path: Path, column_map: dict[str, str] | None = None) -> dict[str, CardDef]:
    """Charge un CSV ; colonnes flexibles (export manuel tcgcsv / custom)."""
    column_map = column_map or {}
    if not path.is_file():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if df.empty:
        return {}

    id_col = _pick_col(df, "card_id", column_map, ["card_id", "Number", "number", "productId", "id"])
    if id_col is None:
        return {}
    name_col = _pick_col(df, "name", column_map, ["name", "Name", "cleanName"]) or id_col
    cost_col = _pick_col(df, "cost", column_map, ["cost", "Cost"]) or None
    pow_col = _pick_col(df, "power", column_map, ["power", "Power"]) or None
    ctr_col = _pick_col(df, "counter", column_map, ["counter", "Counter"]) or None
    color_col = _pick_col(df, "color", column_map, ["color", "Color"]) or None
    img_col = _pick_col(
        df,
        "image_url",
        column_map,
        [
            "image_url",
            "imageUrl",
            "ImageUrl",
            "photoUrl",
            "PhotoUrl",
            "image",
            "Image",
            "url",
        ],
    )
    txt_col = _pick_col(
        df,
        "card_text",
        column_map,
        [
            "card_text",
            "Description",
            "description",
            "CardText",
            "cardText",
            "text",
            "rules_text",
        ],
    )
    type_col = _pick_col(
        df,
        "card_type",
        column_map,
        ["card_type", "Type", "type", "CardType"],
    )
    life_col = _pick_col(df, "life", column_map, ["life", "Life"])

    out: dict[str, CardDef] = {}
    for _, row in df.iterrows():
        cid = str(row[id_col]).strip().upper()
        if not cid or cid == "NAN":
            continue
        cost = _to_int(row[cost_col], 0) if cost_col else 0
        power = _to_int(row[pow_col], 0) if pow_col else 0
        counter = _to_int(row[ctr_col], 0) if ctr_col else 0
        name = str(row[name_col]) if name_col in row.index else cid
        color = str(row[color_col]) if color_col and color_col in row.index else ""
        img_u: str | None = None
        if img_col and img_col in row.index:
            s = str(row[img_col]).strip()
            if s and s.upper() != "NAN" and s.lower().startswith("http"):
                img_u = s
        ctext = ""
        if txt_col and txt_col in row.index:
            ctext = str(row[txt_col]).strip()
            if ctext.upper() == "NAN":
                ctext = ""
        raw_cols = [c for c in df.columns if c != txt_col]
        raw = "|".join(str(row[c]) for c in raw_cols)
        kws = extract_keywords_from_text(ctext)
        ctype = ""
        if type_col and type_col in row.index:
            ctype = str(row[type_col]).strip()
            if ctype.upper() == "NAN":
                ctype = ""
        life = _to_int(row[life_col], 0) if life_col else 0
        low = ctext.lower()
        has_am = "[activate: main]" in low
        don_minus = 0
        if has_am:
            mm = re.search(r"don!!\s*-\s*(\d+)", low)
            if mm:
                don_minus = int(mm.group(1))
        dd_size = 10
        if "leader" in (ctype or "").lower():
            dd_size = infer_leader_don_deck_size(ctext)
        out[cid] = CardDef(
            card_id=cid,
            name=name,
            cost=max(0, cost),
            power=max(0, power),
            counter=max(0, counter),
            color=color,
            raw_text=raw[:500],
            card_text=ctext[:4000],
            image_url=img_u,
            keywords=kws,
            card_type=ctype,
            life=max(0, life),
            don_deck_size=dd_size,
            has_activate_main=has_am,
            activate_main_don_minus=don_minus,
        )
    return out


def ensure_cards_for_deck(
    deck_ids: set[str],
    db: dict[str, CardDef],
) -> dict[str, CardDef]:
    """Complète les IDs manquants avec des stubs."""
    full = dict(db)
    for cid in deck_ids:
        if cid not in full:
            h = abs(hash(cid))
            full[cid] = CardDef(
                card_id=cid,
                name=f"Stub_{cid}",
                cost=1 + (h % 5),
                power=3000 + (h % 8) * 500,
                counter=(h % 3) * 1000,
                color="",
                raw_text="",
                card_text="",
                image_url=None,
                keywords=(),
                card_type="Character",
                don_deck_size=10,
                has_activate_main=False,
                activate_main_don_minus=0,
            )
    return full
