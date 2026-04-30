"""
Empreintes « layout d’effet » pour détecter des cartes au texte de règle équivalent.

Sert au cache partagé (``CardEffectCache``) et aux benchmarks mémoire : un fichier
par carte serait lourd à maintenir ; la dédup par empreinte conserve le moteur
script/regex tout en mutualisant les structures parsées identiques.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .card_db import CardDef

_WS = re.compile(r"\s+")


def normalize_rule_blob(card_text: str) -> str:
    """Normalisation stable pour comparaison (pas pour affichage)."""
    t = (card_text or "").strip().lower()
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = _WS.sub(" ", t)
    return t


def rule_text_fingerprint(card: "CardDef") -> str:
    """
    Empreinte du contenu pertinent pour KeywordModel / Timing / EffectClassifier.

    Inclut le type de carte : deux types avec le même paragraphe marketing restent
    séparés si le classifieur un jour diverge.
    """
    ct = (card.card_type or "").strip().lower()
    blob = normalize_rule_blob(card.card_text or "")
    raw = f"{ct}\x1e{blob}".encode("utf-8", errors="replace")
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def group_card_ids_by_fingerprint(cards: dict[str, "CardDef"]) -> dict[str, list[str]]:
    """``fingerprint -> [card_id, …]`` (ordre non garanti)."""
    out: dict[str, list[str]] = defaultdict(list)
    for cid, cd in cards.items():
        if not (cd.card_text or "").strip():
            continue
        out[rule_text_fingerprint(cd)].append(cid)
    return dict(out)


def unique_fingerprint_stats(cards: dict[str, "CardDef"]) -> tuple[int, int, float]:
    """
    Retourne ``(n_cartes_avec_texte, n_empreintes_uniques, ratio_dedup)``
    où ``ratio_dedup = 1 - n_empreintes / n_cartes`` (0 = tout unique).
    """
    groups = group_card_ids_by_fingerprint(cards)
    n_text = sum(len(v) for v in groups.values())
    n_fp = len(groups)
    ratio = 0.0 if n_text <= 0 else 1.0 - (n_fp / n_text)
    return n_text, n_fp, ratio
