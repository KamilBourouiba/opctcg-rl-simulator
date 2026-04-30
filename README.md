# OPTCG RL Simulator

End-to-end One Piece TCG simulation and training stack:
- Python PPO training pipeline
- Swift native simulator/training path
- Replay rendering (JSONL -> GIF/MP4)
- Card/effect data pipeline from TCGCSV

## Showcase

- Demo video: [native_replay_20260430_043005.mp4](docs/media/native_replay_20260430_043005.mp4)

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

## Core Commands

Fetch card data:
```bash
python scripts/fetch_opcg_tcgcsv.py -o data/cards_tcgcsv.csv
```

Run logged simulation:
```bash
python scripts/run_two_nami_logged.py --steps 80 --log runs/two_nami_actions.log
```

Render replay MP4:
```bash
python scripts/jsonl_to_mp4_images.py runs/actions.jsonl -o runs/out.mp4
```

## Architecture

- `opctcg_text_sim/simulator.py`: main Python game loop
- `opctcg_text_sim/env.py`: Gymnasium environment wrapper
- `opctcg_text_sim/ppo.py`: PPO trainer
- `opctcg_text_sim/engine/`: text effect resolution pipeline
- `swift_training/Sources/OPTCGSimulator/`: native Swift simulator
- `scripts/`: data fetch, conversion, training, and audit tooling

## Notes

- This is a practical simulator/training project, not a full official-rules engine.
- Local runtime artifacts (`runs/`, `checkpoints/`, caches) are excluded from version control.
