"""
Règles et documentation liées au PDF « Comprehensive Rules ».

- ``data/rules_corpus.txt`` : extrait brut (pypdf) pour recherche / RAG.
- ``data/OPTCG_GameRules_Reference.md`` : synthèse structurée + cartographie simulateur.

Le moteur ``SimplifiedOPSim`` reste volontairement réduit ; ce module fournit
les constantes officielles de référence et les en-têtes de journal.
"""
from __future__ import annotations

from pathlib import Path

# Référence officielle (table des matières du PDF rule_comprehensive.pdf, v1.2.0)
RULES_VERSION_LABEL = "ONE PIECE CARD GAME Comprehensive Rules v1.2.0"

# Construction deck / mise en place (section 5 du PDF)
OFFICIAL_DECK_SIZE = 50
OFFICIAL_DON_DECK_SIZE = 10
OFFICIAL_OPENING_HAND = 5
OFFICIAL_MAX_SAME_NUMBER = 4

# Phases d’un tour complet (règle 6-1-1)
OFFICIAL_PHASE_ORDER = (
    "Refresh Phase",
    "Draw Phase",
    "DON!! Phase",
    "Main Phase",
    "End Phase",
)

# Simulateur texte (aligné sur config.yaml sim.* par défaut)
SIM_DEFAULT_MAX_HAND = 7
SIM_DEFAULT_MAX_BOARD = 3


def game_rules_reference_path(project_root: Path) -> Path:
    return project_root / "data" / "OPTCG_GameRules_Reference.md"


def load_rules_corpus(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:200_000]


def load_game_rules_reference(project_root: Path) -> str:
    p = game_rules_reference_path(project_root)
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")[:120_000]


def rules_corpus_hash(corpus: str) -> float:
    """Scalaire dérivé du corpus PDF (conditionnement optionnel du réseau)."""
    if not corpus:
        return 0.0
    return (sum(ord(c) for c in corpus[:5000]) % 10_000) / 10_000.0


def rules_reference_hash(reference_md: str) -> float:
    if not reference_md:
        return 0.0
    return (sum(ord(c) for c in reference_md[:8000]) % 9_997) / 9_997.0


def combined_rules_scalar(project_root: Path, corpus: str) -> float:
    """Combine corpus PDF + fiche markdown de référence (0..~2)."""
    ref = load_game_rules_reference(project_root)
    return min(1.0, rules_corpus_hash(corpus) + 0.35 * rules_reference_hash(ref))


def log_session_preamble() -> list[str]:
    """Lignes à écrire en tête d’un journal d’actions (règles + avertissement sim)."""
    return [
        f"# {RULES_VERSION_LABEL}",
        "# Référence projet : data/OPTCG_GameRules_Reference.md",
        "# Mots-clés / timings / effets : data/OPTCG_Keywords_Reference.md",
        "# Phases officielles tour : Refresh → Draw → DON!! → Main → End (règle 6-1-1).",
        f"# Construction officielle : deck {OFFICIAL_DECK_SIZE} cartes, deck DON!! "
        f"{OFFICIAL_DON_DECK_SIZE} cartes, main initiale {OFFICIAL_OPENING_HAND} "
        f"(puis mulligan, règles 5-1-2, 5-2-1-6).",
        "# SIMULATEUR : deck DON!! 10 ; 1re phase DON du joueur qui commence = 1 carte (6-4-1), "
        "puis +don/tour ; 1er Draw du 1er joueur ignoré (6-3-1) si activé ; dégâts de combat "
        "après le 1er cycle P0→P1 (6-5-6-1).",
        "# Pour extraire le texte brut du PDF : python scripts/extract_rules_pdf.py "
        "<chemin/rule_comprehensive.pdf> -o data/rules_corpus.txt",
    ]
