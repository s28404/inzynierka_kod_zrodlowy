#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../../"

ENV_ID="BabyAI-GoToObj-v0"
VARIANTS=("r2d2" "r2d2_demir" "r2d2_rnd")
SEEDS=(100 101 102)

LOG_DIR="./logs/runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

ENV_SLUG="${ENV_ID//-/_}"

for variant in "${VARIANTS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        LOG_FILE="$LOG_DIR/${variant}_${ENV_SLUG}_seed${seed}.log"

        python3 -u fine_tuned/minigrid/r2d2_run.py \
                --variant "$variant" \
                --env-id "$ENV_ID" \
                --seed "$seed" \
                2>&1 | tee "$LOG_FILE"

        sleep 90
    done
done
