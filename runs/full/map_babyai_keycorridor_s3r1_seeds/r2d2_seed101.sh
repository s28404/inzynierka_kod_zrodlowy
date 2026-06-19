#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../"

python3 -u fine_tuned/minigrid/r2d2_run.py \
    --variant "r2d2" \
    --env-id "BabyAI-KeyCorridorS3R1-v0" \
    --seed 101

sleep 90
