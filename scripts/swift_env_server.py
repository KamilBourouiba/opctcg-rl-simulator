#!/usr/bin/env python3
"""
Serveur JSON lignes (stdin → stdout) pour piloter ``OPTextSimEnv`` depuis Swift.

Protocole (une ligne JSON par message, UTF-8, flush après chaque réponse) :

  {"cmd":"create","repo_root":"/abs/chemin/opctcg_text_sim","deck0":"decks/....txt","deck1":"..."}
      → charge ``config.yaml`` du dépôt, instancie l’env (chemins decks relatifs à repo_root).

  {"cmd":"reset","seed":42}
      → obs[], mask[] (111 ints 0/1), reward 0, done false

  {"cmd":"step","action":7}
      → obs[], mask[], reward, done, truncated

  {"cmd":"close"}
      → arrêt propre.

Les tableaux ``obs`` et ``mask`` sont des listes JSON (float / 0-1).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT_PKG = Path(__file__).resolve().parents[1]
if str(ROOT_PKG) not in sys.path:
    sys.path.insert(0, str(ROOT_PKG))

import yaml


def _reply(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _fail(msg: str) -> None:
    _reply({"ok": False, "error": msg})


def main() -> int:
    env = None
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"json: {exc}")
            continue

        cmd = msg.get("cmd")
        if cmd == "close":
            _reply({"ok": True})
            return 0

        if cmd == "create":
            try:
                repo = Path(str(msg["repo_root"])).resolve()
                cfg_path = repo / "config.yaml"
                if not cfg_path.is_file():
                    _fail(f"config.yaml introuvable : {cfg_path}")
                    continue
                raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
                if not isinstance(raw_cfg, dict):
                    _fail("config.yaml : racine mapping attendue")
                    continue

                from opctcg_text_sim.env import OPTextSimEnv

                paths = raw_cfg.get("paths") or {}
                sim_cfg = dict(raw_cfg.get("sim") or {})
                if bool(msg.get("parity_no_shuffle", False)):
                    sim_cfg["shuffle_decks"] = False
                    sim_cfg["self_play"] = False
                train_cfg = dict(raw_cfg.get("training") or {})

                deck0 = repo / str(msg.get("deck0", "decks/NAMIBY.txt"))
                deck1 = repo / str(msg.get("deck1", msg.get("deck0", "decks/NAMIBY.txt")))
                rel_csv = paths.get("cards_csv", "data/cards_tcgcsv.csv")
                pc = Path(rel_csv).expanduser()
                cards_csv = pc.resolve() if pc.is_absolute() else (repo / pc).resolve()

                rules_rel = paths.get("rules_corpus_out", "data/rules_corpus.txt")
                rp = Path(rules_rel).expanduser()
                rules_p = rp.resolve() if rp.is_absolute() else (repo / rp).resolve()
                rules_corpus_path = rules_p if rules_p.is_file() else None

                col_map = {str(k): str(v) for k, v in (raw_cfg.get("card_csv") or {}).items()}
                obs_dim = int(msg.get("obs_dim") or train_cfg.get("obs_dim", 96))
                seed = int(msg.get("seed", train_cfg.get("seed", 42)))

                env = OPTextSimEnv(
                    deck0,
                    deck1,
                    cards_csv,
                    col_map,
                    rules_corpus_path,
                    sim_cfg,
                    obs_dim=obs_dim,
                    seed=seed,
                )
                _reply({"ok": True, "action_space": int(env.action_space.n), "obs_dim": obs_dim})
            except Exception as exc:
                _fail(f"create: {exc!r}")
            continue

        if env is None:
            _fail("env non créé : envoyez cmd=create d'abord")
            continue

        if cmd == "reset":
            try:
                seed = msg.get("seed")
                obs, _ = env.reset(seed=int(seed) if seed is not None else None)
                mask = env.legal_actions_mask()
                _reply(
                    {
                        "ok": True,
                        "obs": obs.astype(float).tolist(),
                        "mask": [bool(x) for x in mask],
                        "reward": 0.0,
                        "done": False,
                        "truncated": False,
                    }
                )
            except Exception as exc:
                _fail(f"reset: {exc!r}")
            continue

        if cmd == "step":
            try:
                action = int(msg["action"])
                obs, reward, done, truncated, _info = env.step(action)
                mask = env.legal_actions_mask()
                _reply(
                    {
                        "ok": True,
                        "obs": obs.astype(float).tolist(),
                        "mask": [bool(x) for x in mask],
                        "reward": float(reward),
                        "done": bool(done),
                        "truncated": bool(truncated),
                    }
                )
            except Exception as exc:
                _fail(f"step: {exc!r}")
            continue

        if cmd == "debug_state":
            try:
                s = getattr(env, "_sim", None)
                if s is None:
                    _fail("debug_state: simulateur interne indisponible")
                    continue
                cards = getattr(s, "cards", {}) or {}

                def _card_cost(cid: str) -> int:
                    c = cards.get(cid)
                    return int(getattr(c, "cost", 0)) if c is not None else 0

                def _card_type(cid: str) -> str:
                    c = cards.get(cid)
                    return str(getattr(c, "card_type", "")) if c is not None else ""

                p0 = s.p0
                payload = {
                    "ok": True,
                    "phase": str(getattr(s, "phase", "UNKNOWN")),
                    "turns_started": int(getattr(s, "turns_started", 0)),
                    "p0_don_active": int(getattr(p0, "don_active", 0)),
                    "p0_hand": list(getattr(p0, "hand", [])),
                    "p0_hand_costs": [_card_cost(cid) for cid in getattr(p0, "hand", [])],
                    "p0_hand_types": [_card_type(cid) for cid in getattr(p0, "hand", [])],
                }
                _reply(payload)
            except Exception as exc:
                _fail(f"debug_state: {exc!r}")
            continue

        _fail(f"cmd inconnue : {cmd!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
