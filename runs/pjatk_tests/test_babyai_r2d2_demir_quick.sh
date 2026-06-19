#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../../"

PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python3}"
THREADS="${THREADS:-2}"
SEED="${SEED:-2}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

ENV_IDS=(
  "BabyAI-GoToObj-v0"
  "BabyAI-KeyCorridorS4R3-v0"
  "BabyAI-UnlockPickup-v0"
)

echo "=== QUICK SSH TEST: BabyAI (R2D2_DEMIR) ==="

for env_id in "${ENV_IDS[@]}"; do
  echo "[babyai-quick] variant=r2d2_demir env=${env_id}"
  "$PYTHON_BIN" -u fine_tuned/minigrid/r2d2_run.py \
    --variant "r2d2_demir" \
    --env-id "${env_id}" \
    --seed "${SEED}" \
    --total-steps 4000 \
    --warmup-steps 500 \
    --train-every 4 \
    --batch-size 16 \
    --replay-capacity-sequences 2000 \
    --burn-in 8 \
    --unroll-len 16 \
    --n-step 3 \
    --num-threads "${THREADS}" \
    --log-interval 1000 \
    --eval-interval 4000 \
    --eval-episodes 1 \
    --checkpoint-interval 1000
  sleep 1
done

echo "=== DONE: BabyAI quick SSH tests ==="
