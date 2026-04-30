#!/usr/bin/env python3
"""
Entrée principale : option gauntlet multi-decks (config.yaml) ou paire de decks fixe,
PDF + CSV optionnels, puis PPO sur le simulateur texte.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opctcg_text_sim.checkpoint_io import checkpoint_dict
from opctcg_text_sim.env import OPTextSimEnv
from opctcg_text_sim.ppo import format_wall_elapsed, train_loop


def _input_path(prompt: str, default: str | None = None) -> Path:
    s = input(f"{prompt}" + (f" [{default}] " if default else " ")).strip()
    if not s and default:
        s = default
    return Path(s).expanduser()


def main() -> int:
    print("=== opctcg_text_sim — simulateur + entraînement ===\n")
    cfg_path = ROOT / "config.yaml"
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    paths = raw.get("paths", {})
    sim = raw.get("sim", {})
    train_cfg = raw.get("training", {})

    pdf_def = str(paths.get("rules_pdf", "~/Downloads/rule_comprehensive.pdf"))
    do_pdf = input(f"Extraire le PDF des règles ? (o/N) [défaut PDF: {pdf_def}] ").strip().lower()
    corpus_path = ROOT / Path(paths.get("rules_corpus_out", "data/rules_corpus.txt"))
    if do_pdf in ("o", "oui", "y", "yes"):
        pdf = _input_path("Chemin du PDF", pdf_def)
        if pdf.is_file():
            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "extract_rules_pdf.py"),
                    str(pdf),
                    "-o",
                    str(corpus_path),
                ],
                check=False,
            )
        else:
            print("PDF absent, on continue sans corpus.")

    csv_def = str(paths.get("cards_csv", "data/cards_stub.csv"))
    csv_path = _input_path("CSV cartes", csv_def)
    if not csv_path.is_file():
        print(f"CSV introuvable : {csv_path}", file=sys.stderr)
        return 1

    col_map = {str(k): str(v) for k, v in raw.get("card_csv", {}).items()}
    rules_p = corpus_path if corpus_path.is_file() else None
    seed = int(train_cfg.get("seed", 42))
    obs_dim = int(train_cfg.get("obs_dim", 96))

    gauntlet_dir = train_cfg.get("gauntlet_decks_dir")
    use_gauntlet = False
    if gauntlet_dir and str(gauntlet_dir).strip():
        gdir_cfg = str(gauntlet_dir).strip()
        print(f"Config : gauntlet_decks_dir = {gdir_cfg!r}")
        mode = input(
            "Mode [1] Gauntlet — deck aléatoire vs deck aléatoire à chaque partie (recommandé)  "
            "[2] Paire fixe (deux fichiers .txt)  [1/2] : ",
        ).strip()
        use_gauntlet = mode in ("", "1", "o", "O")

    if use_gauntlet:
        from scripts.train_ppo import (
            _ResamplingGauntletFactory,
            build_gauntlet_leader_by_path,
            collect_gauntlet_valid_paths,
        )

        g = Path(gdir_cfg).expanduser()
        gdir = g.resolve() if g.is_absolute() else (ROOT / g).resolve()
        deck_glob = str(train_cfg.get("gauntlet_deck_glob") or "*.txt").strip() or "*.txt"
        valid = collect_gauntlet_valid_paths(
            gdir, csv_path.resolve(), col_map, deck_glob=deck_glob,
        )
        if len(valid) < 1:
            print(f"Aucun deck valide dans {gdir} (glob={deck_glob!r}).", file=sys.stderr)
            return 1
        leader_map = build_gauntlet_leader_by_path(valid, csv_path.resolve(), col_map)
        print(f"→ {len(valid)} deck(s) valide(s) ; paires diversifiées à chaque reset.\n")
        fac = _ResamplingGauntletFactory(
            valid_paths=valid,
            cards_csv=csv_path.resolve(),
            column_map=col_map,
            rules_corpus_path=rules_p,
            sim_cfg=sim,
            obs_dim=obs_dim,
            leader_by_resolved_str=leader_map,
        )
        env = fac(seed)
        env.reset(seed=seed)
    else:
        d0 = _input_path("Deck joueur (agent) .txt", "")
        d1 = _input_path("Deck adversaire .txt", "")
        if not d0.is_file() or not d1.is_file():
            print("Deck introuvable.", file=sys.stderr)
            return 1
        if not csv_path.is_file():
            print("Génération de cards_stub.csv depuis les deux decks…")
            import subprocess

            out = ROOT / "data" / "cards_stub.csv"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_stub_cards_from_decks.py"),
                    str(d0),
                    str(d1),
                    "-o",
                    str(out),
                ],
                check=True,
            )
            csv_path = out

        env = OPTextSimEnv(
            d0,
            d1,
            csv_path.resolve(),
            col_map,
            rules_p,
            sim,
            obs_dim=obs_dim,
            seed=seed,
        )

    _ve = int(train_cfg.get("video_every") or 0)
    _vs = int(train_cfg.get("video_every_steps") or 0)
    _v_ep = int(train_cfg.get("video_every_episodes") or 0)
    if _ve > 0 or _vs > 0 or _v_ep > 0:
        print(
            "\nNote : la config active des vidéos (video_every / steps / episodes), mais ce script "
            "« train.py » n’appelle que train_loop : aucune MP4 n’est générée ici.\n"
            f"   Pour des MP4 pendant l’entraînement : {sys.executable} "
            f"{ROOT / 'scripts' / 'train_ppo.py'} --config {ROOT / 'config.yaml'}\n",
            flush=True,
        )

    print("\nDémarrage PPO (simulateur simplifié)…")
    _t0 = time.time()
    net = train_loop(env, train_cfg)
    print(f"Durée totale (mur) : {format_wall_elapsed(time.time() - _t0)}", flush=True)
    out_m = ROOT / "checkpoints" / "policy_text_sim.pt"
    out_m.parent.mkdir(parents=True, exist_ok=True)
    import torch

    _o = getattr(net, "obs_rms", None)
    _r = getattr(net, "reward_rms", None)
    torch.save(checkpoint_dict(net, obs_rms=_o, reward_rms=_r), out_m)
    print("Politique sauvegardée :", out_m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
