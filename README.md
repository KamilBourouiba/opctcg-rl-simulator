# opctcg_text_sim — simulateur + apprentissage (sans client Unity)

Projet **autonome** : charge deux decklists (format `4xOP01-001` comme OPTCGSim), une base cartes CSV, et entraîne une politique sur un **modèle de jeu volontairement simplifié** (don, coût, puissance, vie, pose / attaque / passer).

## Limites (honnêtes)

- Ce n’est **pas** une reproduction fidèle des règles OP-01…OP-12 : le PDF `rule_comprehensive.pdf` est indexé en **`data/rules_corpus.txt`** (script `scripts/extract_rules_pdf.py`) et synthétisé dans **`data/OPTCG_GameRules_Reference.md`** (phases officielles, zones, mise en place, combat, cartographie sim). Les journaux d’actions citent la **Comprehensive Rules v1.2** ; le GIF replay reprend une disposition type plateau (decks, DON!!, vie, mains des deux joueurs).
- **TCGCSV** : le dépôt inclut `scripts/fetch_opcg_tcgcsv.py`, qui interroge l’API JSON publique [tcgcsv.com](https://tcgcsv.com) (catégorie One Piece = `68`) et produit `data/cards_tcgcsv.csv` avec **imageUrl**, **Description** (texte d’effet), Number, Cost, Power, Counterplus, Color, etc. Sans cela, utilisez un export manuel ou `python scripts/build_stub_cards_from_decks.py` pour un stub local.

## Installation

```bash
cd ~/Documents/opctcg_text_sim
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

```bash
python train.py
```

### Données réelles (cartes, images, textes) via tcgcsv

```bash
python scripts/fetch_opcg_tcgcsv.py -o data/cards_tcgcsv.csv
# optionnel : seulement certains sets — python scripts/fetch_opcg_tcgcsv.py --list-groups
# python scripts/fetch_opcg_tcgcsv.py --groups 23589 -o data/op09_only.csv
```

Puis pointez votre CSV dans `config.yaml` (`paths.cards_csv`) ou passez `--csv data/cards_tcgcsv.csv` à `run_two_nami_logged.py`. Les GIF avec images utiliseront les URLs TCGplayer du fichier.

### Deux fois le même deck + journal ligne à ligne des actions

```bash
python scripts/run_two_nami_logged.py --steps 80 --log runs/two_nami_actions.log
```

Le fichier de log contient une ligne par pas : phase, indice d’action, libellé (carte jouée ou BATTLE), récompense, état (mains, board, don, vies).

### Animation (GIF) à partir du replay JSONL

Chaque pas produit aussi une ligne **JSON** (même base que `--log`, extension `.jsonl` par défaut).

```bash
python scripts/run_two_nami_logged.py --steps 50 --log runs/actions.txt --gif runs/replay.gif
```

Conversion seule d’un JSONL existant :

```bash
python scripts/jsonl_to_gif.py runs/two_nami_actions.jsonl -o runs/out.gif --fps 2
```

Si un fichier `*.manifest.json` porte le même nom d’étape que le `.jsonl`, les **mots-clés** des cartes visibles sont affichés en bas du GIF matplotlib.

### Vidéo MP4 (même rendu que le GIF images)

```bash
pip install imageio imageio-ffmpeg
python scripts/run_two_nami_logged.py --log runs/actions.txt --mp4 runs/replay.mp4
python scripts/jsonl_to_mp4_images.py runs/actions.jsonl -o runs/out.mp4
```

Les replays très courts sont **allongés** (dernière image répétée) pour éviter l’avertissement FFmpeg `rawvideo … not enough frames`, et l’encodage passe en journal FFmpeg `error` (moins de bruit sur stderr).

### GIF avec visuels des cartes (données tcgcsv / imageUrl)

À chaque `reset` avec `animation_log_path`, l’environnement écrit aussi un fichier **`*.manifest.json`** (à côté du `.jsonl`) : pour chaque `card_id` connu, `name` et `image_url` issus du CSV. Ajoutez une colonne d’URL dans votre export (souvent `imageUrl` ou `photoUrl`) : la détection est automatique dans `card_db.py`, ou forcez le mapping dans `config.yaml` sous `card_csv` avec la clé logique `image_url` (ex. `image_url: "imageUrl"`).

```bash
python scripts/run_two_nami_logged.py --steps 50 --log runs/actions.txt --gif-images runs/replay_img.gif
```

Sur un replay déjà enregistré, utilisez le **même nom de base** que `--log` (ex. `--log runs/actions.txt` → `runs/actions.jsonl` + `runs/actions.manifest.json`) :

```bash
python scripts/jsonl_to_gif_images.py runs/actions.jsonl -o runs/out_img.gif --cache runs/card_image_cache
```

Le script demande les chemins des deux decklists, optionnellement le PDF des règles et un CSV cartes, puis lance l’entraînement PPO (PyTorch pur, léger).

## Fichiers

| Chemin | Rôle |
|--------|------|
| `opctcg_text_sim/deck_parser.py` | Parse `NxID` |
| `opctcg_text_sim/card_db.py` | Chargement CSV + cartes manquantes (stub) |
| `opctcg_text_sim/playability.py` | Heuristiques « jouable » (coût, phase, main) |
| `opctcg_text_sim/simulator.py` | Moteur tour par tour simplifié |
| `opctcg_text_sim/env.py` | Gymnasium `Discrete` |
| `opctcg_text_sim/ppo.py` | PPO minimal |
| `scripts/extract_rules_pdf.py` | PDF → `data/rules_corpus.txt` |
| `scripts/build_stub_cards_from_decks.py` | Génère `data/cards_stub.csv` depuis decks |
| `scripts/fetch_opcg_tcgcsv.py` | Télécharge l’OPCG depuis tcgcsv.com → CSV (images + texte) |
| `opctcg_text_sim/tcgcsv_fetch.py` | Client JSON catégories / groupes / produits |
| `data/OPTCG_GameRules_Reference.md` | Règles synthétisées depuis le PDF + lien avec le sim |
| `data/OPTCG_Keywords_Reference.md` | Mots-clés / effets à mots-clés (sec. 10 + timings + Counter) |
| `opctcg_text_sim/card_keywords.py` | Extraction `[ … ]`, `{ … }`, `< … >` depuis le texte de carte |
| `data/rules_corpus.txt` | Texte brut extrait du PDF (recherche / RAG) |
| `opctcg_text_sim/replay_render_images.py` | JSONL + manifeste → GIF Pillow (vignettes) |
| `scripts/jsonl_to_gif_images.py` | CLI conversion JSONL → GIF avec images |

## Étendre

1. Enrichir `rules_simple.py` (effets déclencheurs, counter, DON!!, etc.).
2. Remplacer `cards_stub.csv` par un export réel et adapter `column_map` dans `config.yaml`.
3. Augmenter `obs_dim` / graphe d’actions dans `env.py` si vous ajoutez des choix (cibles, DON attaché, …).

## Showcase

- Latest generated demo video: `docs/media/native_replay_20260430_043005.mp4`
