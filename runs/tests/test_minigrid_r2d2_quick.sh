#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
THREADS="${THREADS:-2}"

VARIANTS=("r2d2" "r2d2_rnd" "r2d2_ngu" "r2d2_demir")

ENV_IDS=(
  "BabyAI-GoToObj-v0"
  "BabyAI-KeyCorridorS4R3-v0"
  "BabyAI-UnlockPickup-v0"
)

for env_id in "${ENV_IDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    echo "[quick-test] ${variant} on ${env_id}"
    "${PYTHON_BIN}" -u fine_tuned/minigrid/r2d2_run.py \
      --variant "${variant}" \
      --env-id "${env_id}" \
      --seed 2 \
      --total-steps 8000 \
      --warmup-steps 1000 \
      --train-every 4 \
      --batch-size 16 \
      --replay-capacity-sequences 2000 \
      --burn-in 8 \
      --unroll-len 16 \
      --n-step 3 \
      --num-threads "${THREADS}" \
      --log-interval 1000 \
      --eval-interval 4000 \
      --eval-episodes 2 \
      --checkpoint-interval 0
    sleep 1
  done
done

echo "BabyAI R2D2 quick tests finished."
