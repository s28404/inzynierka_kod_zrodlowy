#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
THREADS="${THREADS:-4}"
SEED="${SEED:-1}"
TOTAL_STEPS="${TOTAL_STEPS:-300000}"

# BabyAI hierarchical tasks for DEMIR evaluation
ENV_IDS=(
  "BabyAI-GoToObj-v0"
  "BabyAI-KeyCorridorS4R3-v0"
  "BabyAI-UnlockPickup-v0"
)

VARIANTS=(
  "r2d2"
  "r2d2_rnd"
  "r2d2_ngu"
  "r2d2_demir"
)

for env_id in "${ENV_IDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    echo "==========================================================="
    echo "RUN: ${variant} on ${env_id} (seed=${SEED}, steps=${TOTAL_STEPS})"
    echo "==========================================================="

    "${PYTHON_BIN}" -u fine_tuned/minigrid/r2d2_run.py \
      --variant "${variant}" \
      --env-id "${env_id}" \
      --seed "${SEED}" \
      --total-steps "${TOTAL_STEPS}" \
      --num-threads "${THREADS}" \
      --log-interval 2000 \
      --eval-interval 20000 \
      --eval-episodes 5 \
      --checkpoint-interval 50000

    sleep 1
  done
done

echo "All R2D2 BabyAI runs finished."
