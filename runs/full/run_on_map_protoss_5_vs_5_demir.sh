#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../../"

TASKS=("smacv2/protoss_5_vs_5")
ALGOS=("demir")
SEEDS=(100 101 102)

LOG_DIR="./logs/runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

for task in "${TASKS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            LOG_FILE="$LOG_DIR/${algo}_${task//\//_}_seed${seed}.log"
            
            python3 -u fine_tuned/smacv2/smacv2_run.py \
                    task="$task" \
                    algorithm="$algo" \
                    seed="$seed" \
                    2>&1 | tee "$LOG_FILE"
            
            sleep 90
        done
    done
done
