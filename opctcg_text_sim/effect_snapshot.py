"""
Snapshot JSON des effets parsés (``ParsedCard``) pour le simulateur.

- **Export** : un seul fichier ``.json`` avec toutes les cartes ayant un ``card_text``,
  telles que les produit ``KeywordModel`` / ``TimingModel`` / ``EffectClassifier``.
- **Chargement** : ``CardEffectCache(snapshot_path=...)`` hydrate depuis le JSON et
  recolle nom / stats / texte depuis le CSV courant (le snapshot porte surtout sur
  ``keywords``, ``timing_segments``, ``effects``).

Pour une base SQL, le même objet peut être stocké en **jsonb** (PostgreSQL) : même
schéma, pas de format propriétaire supplémentaire.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .card_db import CardDef
from .models.effect_models import ParsedCard

SNAPSHOT_VERSION = 1


def overlay_parsed_card_from_live_csv(pc: ParsedCard, card: CardDef) -> None:
    """Aligne métadonnées gameplay sur le ``CardDef`` courant (CSV / deck)."""
    pc.card_id = card.card_id
    pc.name = card.name or pc.name
    pc.cost = int(card.cost)
    pc.power = int(card.power)
    pc.counter = int(card.counter)
    pc.color = card.color or pc.color
    pc.card_type = card.card_type or pc.card_type
    pc.life = int(card.life)
    pc.card_text = card.card_text or ""


def load_snapshot_index(path: Path | str) -> dict[str, dict[str, Any]]:
    """Lit le fichier snapshot et retourne la carte ``card_id -> dict``."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Snapshot effets introuvable : {p.resolve()}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    ver = int(raw.get("version", 0))
    if ver != SNAPSHOT_VERSION:
        raise ValueError(
            f"Snapshot version {ver} != attendue {SNAPSHOT_VERSION} — régénérer l’export."
        )
    cards = raw.get("cards")
    if not isinstance(cards, dict):
        raise ValueError("Snapshot JSON invalide : clé top-level 'cards' (objet) requise.")
    return cards


def export_effect_snapshot_json(
    cards: dict[str, CardDef],
    out_path: Path | str,
    *,
    cache: Any | None = None,
) -> int:
    """
    Écrit ``out_path`` au format snapshot. Retourne le nombre de cartes exportées
    (celles avec ``card_text`` non vide).
    """
    from .engine.effect_resolver import CardEffectCache

    c = cache or CardEffectCache()
    c.precompute(cards)
    payload: dict[str, Any] = {
        "version": SNAPSHOT_VERSION,
        "format": "opctcg_text_sim.parsed_effects",
        "cards": {},
    }
    n = 0
    for cid, card in cards.items():
        if not (card.card_text or "").strip():
            continue
        pc = c.get(card)
        payload["cards"][cid] = pc.to_dict()
        n += 1
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return n
