#!/usr/bin/env bash
# Deux runs A/B courts (mêmes steps) avec overlays différents, puis affiche le dernier stp_s
# des metrics JSONL. Réglages : AB_STEPS (défaut 400000), AB_GAUNTLET (défaut decks/tournament_op15).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi
STEPS="${AB_STEPS:-400000}"
GAUNTLET="${AB_GAUNTLET:-decks/tournament_op15}"

run_one_seq() {
  local label="$1"
  local overlay="$2"
  echo ""
  echo "========== A/B : ${label} =========="
  "$PY" "$ROOT/scripts/train_ppo.py" \
    --config "$ROOT/config.yaml" \
    --config-overlay "$ROOT/${overlay}" \
    --gauntlet-decks-dir "$ROOT/${GAUNTLET}" \
    --steps "$STEPS" \
    --n-envs 0 \
    --video-every 0
}

run_one_seq "few_workers + gros minibatch + threads bas" "configs/ab/a_few_workers_more_ram.yaml"
run_one_seq "plus d'envs + minibatch 2048" "configs/ab/b_many_workers_default_mini.yaml"

echo ""
echo "========== Dernier enregistrement metrics (stp_s) =========="
for f in runs/ab_a_few_workers.jsonl runs/ab_b_many_workers.jsonl; do
  if [[ ! -f "$f" ]]; then
    echo "(absent) $f"
    continue
  fi
  echo "--- $f ---"
  tail -n 1 "$f" | "$PY" -c "import json,sys; d=json.loads(sys.stdin.read()); print('stp_s=', d.get('stp_s'), ' update=', d.get('update'), sep='')"
done
