#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../../../"

LOG_DIR="./logs/runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/rnd_smacv2_protoss_5_vs_5_seed100.log"

python3 -u fine_tuned/smacv2/smacv2_run.py \
        task="smacv2/protoss_5_vs_5" \
        algorithm="rnd" \
        seed="100" \
        2>&1 | tee "$LOG_FILE"