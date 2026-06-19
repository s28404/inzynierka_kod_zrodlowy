#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../../"

ENV_ID="BabyAI-KeyCorridorS4R3-v0"
SEEDS=(100 101 102)

LOG_DIR="./logs/runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

run_demir_ablation() {
    local label="$1"
    shift

    for seed in "${SEEDS[@]}"; do
        local log_file="$LOG_DIR/${label}_${ENV_ID//-/_}_seed${seed}.log"

        python3 -u fine_tuned/minigrid/r2d2_run.py \
                --variant "r2d2_demir" \
                --env-id "$ENV_ID" \
                --seed "$seed" \
                "$@" \
                2>&1 | tee "$log_file"

        sleep 90
    done
}

# DEMIR ablations
run_demir_ablation "demir_beta1_0p0_beta2_1p0" --demir-beta1 0.0 --demir-beta2 1.0
run_demir_ablation "demir_beta1_1p0_beta2_0p0" --demir-beta1 1.0 --demir-beta2 0.0
run_demir_ablation "demir_mlp_encoder" --demir-encoder-type mlp
run_demir_ablation "demir_idm_no_barlow" --demir-encoder-type idm_no_barlow
