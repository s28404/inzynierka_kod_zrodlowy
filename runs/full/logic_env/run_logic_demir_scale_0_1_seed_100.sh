#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../../../"

LOG_DIR="./logs/runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/demir_scale_0_1_sync_factory_seed100.log"

python3 -u fine_tuned/logic_env/logic_env_run.py \
    task="logic_env/synchronized" \
    algorithm="demir" \
    algorithm.demir_scale="0.1" \
    seed="100" \
    2>&1 | tee "$LOG_FILE"
