#!/usr/bin/env bash
# Supprime les sorties d’entraînement locales (runs, checkpoints, wandb) avant un nouveau run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

shopt -s nullglob
for d in runs wandb; do
  if [[ -d "$d" ]]; then
    rm -rf "${d:?}"/*
    mkdir -p "$d"
  fi
done

if [[ -d checkpoints ]]; then
  rm -f checkpoints/*.pt
fi
mkdir -p checkpoints runs

# Cache images replay éventuel
if [[ -d runs/card_image_cache ]]; then
  rm -rf runs/card_image_cache
fi

echo "Nettoyé : runs/*, checkpoints/*.pt, wandb/* (dossiers conservés)."
