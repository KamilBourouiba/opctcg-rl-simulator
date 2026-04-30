# OPTCG — pile Swift (workflow complet)

Ce dossier regroupe **tout le flux « modèle + règles + entraînement »** côté Swift, avec deux niveaux :

## 1. Démo CPU (`optrain`) — 100 % Swift

Régression linéaire synthétique + `TrainingRules` JSON (voir `Sources/OPTrainingCore/Resources/default_training.json`). Aucun Python.

```bash
cd swift_training
swift run -c release optrain
```

## 2. Entraînement `optcg_train` — backend Python **ou** Swift natif

Deux backends sont disponibles :

- `--backend python` (défaut) : simulateur OPTCG de référence (`simulator.py` + `env.py`).
- `--backend native` : simulateur Swift V1 (`OPTCGSimulator`) sans runtime Python.

En backend Python, Swift pilote une boucle **REINFORCE + baseline** (politique **linéaire**) via un **serveur JSON lignes** :

- `scripts/swift_env_server.py` — stdin/stdout, une ligne JSON par commande.
- `Sources/OPTCGBridge/PythonEnvSession.swift` — client Process Swift.

### Exemple backend Python

Installer les deps du dépôt (`gymnasium`, `numpy`, …), par ex. :

```bash
cd ..   # racine opctcg_text_sim
pip install -r requirements.txt
```

Utiliser **le même interpréteur** que celui où les modules sont installés :

```bash
cd swift_training
swift run -c release optcg_train \
  --repo .. \
  --python "$(which python3)" \
  --episodes 20 \
  --max-steps 2048
```

Options utiles :

| Option | Rôle |
|--------|------|
| `--repo` | Racine du dépôt (doit contenir `config.yaml`, `scripts/swift_env_server.py`, decks). |
| `--python` | Interpréteur Python avec `opctcg_text_sim` importable. |
| `--backend` | `python` (défaut) ou `native`. |
| `--cards-csv` | CSV cartes (backend `native`, défaut `data/cards_tcgcsv.csv`). |
| `--deck0` / `--deck1` | Chemins **relatifs au repo** (défaut `decks/NAMIBY.txt`). |
| `--episodes`, `--max-steps` | Volume d’entraînement. |
| `--lr-policy`, `--lr-value`, `--gamma` | Hyperparamètres REINFORCE. |
| `--save-model` | Sauvegarde le modèle Swift entraîné en JSON. |
| `--replay-jsonl` | Exporte un replay greedy (backend `native` uniquement). |
| `--video-mp4` | Rend une vidéo MP4 depuis le replay (backend `native`, nécessite deps Python de rendu). |

### Exemple backend natif (sans Python)

```bash
cd swift_training
swift run -c release optcg_train \
  --repo .. \
  --backend native \
  --cards-csv data/cards_tcgcsv.csv \
  --episodes 20 \
  --max-steps 1024
```

Sortie finale : une ligne JSON `{"ok":true,"mean_reward":...}` sur stdout.

Exemple avec artefacts :

```bash
swift run optcg_train \
  --repo .. \
  --backend native \
  --episodes 20 \
  --save-model runs/native_model.json \
  --replay-jsonl runs/native_replay.jsonl \
  --video-mp4 runs/native_replay.mp4
```

### Écarts connus backend natif V1

Le backend Swift V1 est jouable pour l’entraînement de smoke-test, mais il ne couvre pas encore toute la fidélité du moteur Python :

- pas de moteur d’effets complet (`[On Play]`, chaînes, timings avancés) ;
- pas de phase `blocker/counter` complète ;
- heuristique adverse simplifiée ;
- observation/masque alignés en taille (`obs_dim`, `action_space`) mais sémantique simplifiée.

Pour les runs de recherche/benchmark fidèles règles officielles, conserver `--backend python`.

## Modules SPM

| Module | Rôle |
|--------|------|
| `OPTrainingCore` | `TrainingRules`, RNG, démo `SwiftCPUDemoTrainer`, constantes `GameConstants`. |
| `OPTCGBridge` | `PythonEnvSession` (JSON lignes). |
| `OPTCGRL` | `LinearActorCritic`, softmax masqué, échantillonnage d’actions. |
| `OPTCGSimulator` | Simulateur natif V1 + `NativeTextSimSession`. |
| `optcg_train` | Exécutable : boucle REINFORCE + backend Python ou natif. |
| `optrain` | Exécutable : démo CPU uniquement. |

## MLX / Metal

Une extension possible : ajouter la dépendance SPM `mlx-swift` et déplacer la mise à jour de politique vers MLX ; sur certaines machines `swift run` seul peut échouer au chargement des métallibs — **ouvrir le paquet dans Xcode** est souvent plus fiable pour MLX.
