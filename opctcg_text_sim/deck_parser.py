from __future__ import annotations

import re
from pathlib import Path

from .card_db import CardDef


_LINE_RE = re.compile(r"^\s*(\d+)\s*[xX]\s*([A-Za-z0-9\-]+)\s*$")
# Ligne optionnelle : « # leader OP05-041 » ou « Leader: OP05-041 » (hors deck, zone Leader).
_LEADER_DIRECTIVE = re.compile(
    r"^\s*(?:#\s*)?(?:leader|LEADER)\s*[:\s]+\s*([A-Za-z0-9\-]+)\s*$",
    re.IGNORECASE,
)


def parse_deck_file(path: Path) -> list[tuple[str, int]]:
    """
    Deck OPTCGSim : lignes ``4xOP01-001`` ou ``1 x OP11-041``.
    Retourne liste (card_id, count) dans l’ordre du fichier.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    out: list[tuple[str, int]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        n, cid = int(m.group(1)), m.group(2).strip().upper()
        out.append((cid, n))
    return out


def deck_to_multiset(entries: list[tuple[str, int]]) -> dict[str, int]:
    m: dict[str, int] = {}
    for cid, n in entries:
        m[cid] = m.get(cid, 0) + n
    return m


def multiset_to_deck_list(m: dict[str, int]) -> list[str]:
    """Liste plate pour pioche (répétitions)."""
    deck: list[str] = []
    for cid, n in sorted(m.items()):
        deck.extend([cid] * n)
    return deck


def read_deck_leader_directive(path: Path) -> str | None:
    """Si le fichier deck contient une ligne ``# leader OP05-041`` (ou équivalent), retourne l’id."""
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        m = _LEADER_DIRECTIVE.match(raw.strip())
        if m:
            return m.group(1).strip().upper()
    return None


def entry_list_cid_order(entries: list[tuple[str, int]]) -> list[str]:
    """Ordre de première apparition des ids dans le fichier (pour choisir le Leader typé)."""
    order: list[str] = []
    for cid, _ in entries:
        if cid not in order:
            order.append(cid)
    return order


def _is_leader_card(cards: dict[str, CardDef], cid: str) -> bool:
    c = cards.get(cid)
    if c is None:
        return False
    return (c.card_type or "").strip().lower() == "leader"


def extract_leader_from_multiset(
    m: dict[str, int],
    cards: dict[str, CardDef],
    *,
    explicit_leader_id: str | None = None,
    preferred_cid_order: list[str] | None = None,
) -> tuple[dict[str, int], str | None]:
    """
    Retire du pool « main deck » toutes les copies de la carte Leader.

    - Si ``explicit_leader_id`` est présent et dans le multiset : ce Leader.
    - Sinon : première id du fichier parmi ``preferred_cid_order`` typée Leader dans ``cards``.
    - Sinon : première id (tri) du multiset typée Leader.
    Le Leader n’est plus dans le deck (règle officielle : deck 50 sans le Leader).
    """
    m = dict(m)
    chosen: str | None = None
    if explicit_leader_id:
        eid = explicit_leader_id.strip().upper()
        if m.get(eid, 0) >= 1:
            chosen = eid
    if chosen is None and preferred_cid_order:
        for cid in preferred_cid_order:
            if m.get(cid, 0) >= 1 and _is_leader_card(cards, cid):
                chosen = cid
                break
    if chosen is None:
        for cid in sorted(m.keys()):
            if m[cid] >= 1 and _is_leader_card(cards, cid):
                chosen = cid
                break
    if chosen is None or m.get(chosen, 0) < 1:
        return m, None
    m.pop(chosen, None)
    return m, chosen


def infer_deck_leader_id(path: Path, cards: dict[str, CardDef]) -> str | None:
    """
    Identifiant du Leader pour un fichier deck (même logique que la validation :
    directive ``# leader …``, sinon première carte Leader dans l’ordre du fichier).
    """
    entries = parse_deck_file(path)
    if not entries:
        return None
    m = deck_to_multiset(entries)
    explicit = read_deck_leader_directive(path)
    _, lid = extract_leader_from_multiset(
        m,
        cards,
        explicit_leader_id=explicit,
        preferred_cid_order=[cid for cid, _ in entries],
    )
    return lid
