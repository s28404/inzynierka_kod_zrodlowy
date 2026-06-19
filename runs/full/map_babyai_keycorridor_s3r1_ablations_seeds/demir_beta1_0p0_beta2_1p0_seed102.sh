#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../"

python3 -u fine_tuned/minigrid/r2d2_run.py \
    --variant "r2d2_demir" \
    --env-id "BabyAI-KeyCorridorS3R1-v0" \
    --seed 102 \
    --demir-beta1 0.0 \
    --demir-beta2 1.0

sleep 90
