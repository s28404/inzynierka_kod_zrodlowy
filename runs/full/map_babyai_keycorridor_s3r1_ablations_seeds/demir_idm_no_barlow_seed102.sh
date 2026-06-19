#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../../"

python3 -u fine_tuned/minigrid/r2d2_run.py \
    --variant "r2d2_demir" \
    --env-id "BabyAI-KeyCorridorS3R1-v0" \
    --seed 102 \
    --demir-encoder-type idm_no_barlow

sleep 90
