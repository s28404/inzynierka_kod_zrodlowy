#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../../../"

LOG_DIR="./logs/runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/demir_beta1_1p0_beta2_0p0_smacv2_protoss_5_vs_5_seed100.log"

python3 -u fine_tuned/smacv2/smacv2_run.py \
        task="smacv2/protoss_5_vs_5" \
        algorithm="demir" \
        seed="100" \
        algorithm.beta1=1.0 \
        algorithm.beta2=0.0 \
        2>&1 | tee "$LOG_FILE"