#!/usr/bin/env python3
"""Extrait le texte d’un PDF de règles vers data/rules_corpus.txt (pypdf)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("pdf", type=Path, help="Chemin vers rule_comprehensive.pdf")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=ROOT / "data" / "rules_corpus.txt",
    )
    args = p.parse_args()
    try:
        from pypdf import PdfReader
    except ImportError:
        print("Installez pypdf : pip install pypdf", file=sys.stderr)
        return 1
    if not args.pdf.is_file():
        print("PDF introuvable :", args.pdf, file=sys.stderr)
        return 1
    reader = PdfReader(str(args.pdf))
    parts: list[str] = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        parts.append(t)
    text = "\n\n".join(parts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print("Écrit :", args.out, "—", len(text), "caractères,", len(reader.pages), "pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
