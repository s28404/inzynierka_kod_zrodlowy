#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../../"

LOG_DIR="./logs/runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/demir_scale_1_0_sync_factory_seed101.log"

python3 -u fine_tuned/logic_env/logic_env_run.py \
    task="logic_env/synchronized" \
    algorithm="demir" \
    algorithm.demir_scale="1.0" \
    seed="101" \
    2>&1 | tee "$LOG_FILE"
