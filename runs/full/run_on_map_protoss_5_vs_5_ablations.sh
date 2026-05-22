#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../../"

TASK="smacv2/protoss_5_vs_5"
SEEDS=(100 101 102)

LOG_DIR="./logs/runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

run_demir_ablation() {
	local label="$1"
	shift

	for seed in "${SEEDS[@]}"; do
		local log_file="$LOG_DIR/${label}_${TASK//\//_}_seed${seed}.log"

		python3 -u fine_tuned/smacv2/smacv2_run.py \
				task="$TASK" \
				algorithm="demir" \
				seed="$seed" \
				"$@" \
				2>&1 | tee "$log_file"

		sleep 90
	done
}

# DEMIR ablations
run_demir_ablation "demir_beta1_0p0_beta2_1p0" algorithm.beta1=0.0 algorithm.beta2=1.0
run_demir_ablation "demir_beta1_1p0_beta2_0p0" algorithm.beta1=1.0 algorithm.beta2=0.0
run_demir_ablation "demir_mlp_encoder" algorithm.encoder_type=mlp
run_demir_ablation "demir_idm_no_barlow" algorithm.encoder_type=idm_no_barlow

