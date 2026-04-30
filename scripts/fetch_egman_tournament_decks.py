#!/usr/bin/env python3
"""
Télécharge les listes du tournoi Egman Deckbuilder via l’API Supabase (PostgREST).

L’URL du type
  https://deckbuilder.egmanevents.com/optcg/tournaments?format=OP15&t=<UUID>
utilise ``<UUID>`` comme ``tournament_id``.

Chaque ligne ``tournament_results`` fournit ``deck_list_url`` (paramètre ``deck=CODE:qty,...``)
converti au format simulateur (``4xOP01-001``).

Variables d’environnement (optionnelles) :
  EGMAN_SUPABASE_URL   (défaut : URL publique du bundle web)
  EGMAN_SUPABASE_KEY   (défaut : clé publishable du bundle web — déjà exposée côté client)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse, urlsplit

try:
    import requests
except ImportError:
    print("Installe requests : pip install requests", file=sys.stderr)
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SUPABASE_URL = "https://resgvirjzcpamfumrygh.supabase.co"
DEFAULT_SUPABASE_KEY = "sb_publishable_bdDgor6ifmOvryEuZKWniw_RBzb3vuh"


def _slug(s: str, max_len: int = 72) -> str:
    s = (s or "unknown").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("._")
    return (s or "deck")[:max_len]


def _parse_deck_query(deck_param: str) -> list[tuple[str, int]]:
    """``EB03-055:4,OP11-041:1`` → [('EB03-055', 4), ('OP11-041', 1)] en conservant l’ordre (fusion des doublons)."""
    out: list[tuple[str, int]] = []
    seen: dict[str, int] = {}
    for part in deck_param.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        code, _, qty_s = part.partition(":")
        code = code.strip().upper()
        if not code:
            continue
        try:
            qty = int(qty_s.strip())
        except ValueError:
            continue
        if qty <= 0:
            continue
        if code in seen:
            i = seen[code]
            oc, oq = out[i]
            out[i] = (oc, oq + qty)
        else:
            seen[code] = len(out)
            out.append((code, qty))
    return out


def _deck_url_to_lines(deck_url: str) -> list[str]:
    q = urlparse(deck_url)
    qs = parse_qs(q.query)
    raw = qs.get("deck", [""])[0]
    raw = unquote(raw)
    entries = _parse_deck_query(raw)
    return [f"{n}x{cid}" for cid, n in entries]


def fetch_results(session: requests.Session, base: str, key: str, tournament_id: str) -> list[dict]:
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    url = f"{base}/rest/v1/tournament_results"
    params = {
        "tournament_id": f"eq.{tournament_id}",
        "select": "placement,player_name,deck_type,deck_list_url,result_order",
        "order": "result_order.asc",
    }
    r = session.get(url, headers=headers, params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()
    if not isinstance(rows, list):
        raise RuntimeError(f"Réponse inattendue : {rows!r}")
    return rows


def _tournament_id_from_url(url: str) -> str:
    q = urlsplit(url).query
    for pair in q.split("&"):
        if pair.startswith("t="):
            return unquote(pair[2:].strip())
    raise ValueError("Paramètre t= introuvable dans l’URL")


def main() -> int:
    ap = argparse.ArgumentParser(description="Import decklists Egman (Supabase) → dossier .txt")
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--tournament-id",
        default=None,
        help="UUID du paramètre ``t=`` dans l’URL Egman",
    )
    g.add_argument(
        "--tournament-url",
        default=None,
        help="URL complète du tournoi (extrait ``t=`` automatiquement)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "decks" / "tournament_op15",
        help="Dossier cible pour les .txt",
    )
    ap.add_argument("--dry-run", action="store_true", help="Affiche seulement ce qui serait écrit")
    args = ap.parse_args()

    if args.tournament_url:
        tournament_id = _tournament_id_from_url(args.tournament_url)
    elif args.tournament_id:
        tournament_id = args.tournament_id
    else:
        tournament_id = "009ab10b-49d1-4b3e-9342-5b6eb808c15a"

    base = (os.environ.get("EGMAN_SUPABASE_URL") or DEFAULT_SUPABASE_URL).rstrip("/")
    key = os.environ.get("EGMAN_SUPABASE_KEY") or DEFAULT_SUPABASE_KEY

    session = requests.Session()
    rows = fetch_results(session, base, key, tournament_id)
    if not rows:
        print("Aucun résultat pour ce tournoi.", file=sys.stderr)
        return 1

    meta_url = args.tournament_url or (
        f"https://deckbuilder.egmanevents.com/optcg/tournaments?"
        f"format=OP15&t={tournament_id}"
    )
    out_dir: Path = args.out_dir.resolve()
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob("*.txt"):
            if old.name.upper() != "README.TXT":
                old.unlink()

    written = 0
    for row in rows:
        url = (row.get("deck_list_url") or "").strip()
        if not url or "deck=" not in url:
            continue
        name = row.get("player_name") or "unknown"
        placement = row.get("placement")
        dtype = row.get("deck_type") or ""
        try:
            lines = _deck_url_to_lines(url)
        except Exception as e:
            print(f"Skip {name!r} : {e}", file=sys.stderr)
            continue
        if not lines:
            print(f"Skip {name!r} : deck vide", file=sys.stderr)
            continue
        ro = row.get("result_order")
        try:
            pi = int(ro if ro is not None else placement)
        except (TypeError, ValueError):
            pi = written + 1
        fname = _slug(f"p{pi:04d}_{name}")
        body = [
            f"# Egman tournament {tournament_id}",
            f"# Page : {meta_url}",
            f"# Joueur : {name} | Classement affiché : {placement} | deck_type : {dtype}",
            f"# Liste : {url}",
            "",
            *lines,
            "",
        ]
        text = "\n".join(body)
        if args.dry_run:
            print(f"→ {fname}.txt ({len(lines)} lignes)")
        else:
            (out_dir / f"{fname}.txt").write_text(text, encoding="utf-8")
        written += 1

    print(json.dumps({"tournament_id": tournament_id, "decks_written": written, "out_dir": str(out_dir)}, indent=2))
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
