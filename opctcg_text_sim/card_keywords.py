"""
Extraction des marqueurs de règles depuis le texte de carte (effets TCGplayer / tcgcsv).

Repère notamment :
  - tout texte entre **crochets** ``[ ... ]`` (ex. ``[On Play]``, ``[When Attacking]``, ``[Counter]``) ;
  - conditions **``[DON!! x2]``** avec espaces variables ;
  - **types** ``{ ... }`` et **attributs** ``< ... >`` souvent présents sur les cartes.
"""
from __future__ import annotations

import re
from typing import Final

_DON_X_FULL: Final = re.compile(r"^\[\s*DON!!\s*x\s*(\d+)\s*\]$", re.IGNORECASE)


def _normalize_bracket_token(tok: str) -> str:
    tok = tok.strip()
    m = _DON_X_FULL.match(tok)
    if m:
        return f"[DON!! x{m.group(1)}]"
    return tok


def extract_keywords_from_text(text: str) -> tuple[str, ...]:
    """
    Retourne une tuple **dédupliquée**, ordre d’apparition dans ``text``.

    Inclut chaque ``[ ... ]`` (1–120 caractères intérieurs), les ``{Type}`` et ``<Attribute>``.
    """
    if not text or not str(text).strip():
        return ()

    t = str(text).replace("\u2019", "'").replace("\u2018", "'")
    seen: set[str] = set()
    out: list[str] = []

    def _push(raw: str) -> None:
        s = _normalize_bracket_token(raw) if raw.startswith("[") else raw.strip()
        if len(s) < 2 or s in seen:
            return
        seen.add(s)
        out.append(s)

    for m in re.finditer(r"\[[^\]]{1,120}\]", t):
        _push(m.group(0))

    for m in re.finditer(r"\{[^\}]{1,80}\}", t):
        _push(m.group(0))

    for m in re.finditer(r"<[^>]{1,40}>", t):
        _push(m.group(0))

    return tuple(out)


def format_keywords_line(keywords: tuple[str, ...], *, max_len: int = 220) -> str:
    if not keywords:
        return ""
    s = " · ".join(keywords)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"
