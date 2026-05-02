#!/usr/bin/env bash
set -euo pipefail

# Collects all artifacts from a training run into a ZIP file for easy download
# Usage: ./collect_results.sh <experiment_id> [output_dir]
# Example: ./collect_results.sh "qmix_smacv2_protoss_5_vs_5_seed1" /tmp/results

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <experiment_id_pattern> [output_dir]"
  echo "  <experiment_id_pattern>: Part of experiment name to search for (e.g., 'protoss_5_vs_5_qmix_seed100')"
  echo "  [output_dir]: Where to save the ZIP file (default: current dir)"
  echo ""
  echo "Example:"
  echo "  $0 'protoss_5_vs_5_qmix_seed100' /tmp/results"
  echo "  $0 'protoss_5_vs_5_qmix_seed100'  # saves to current directory"
  exit 1
fi

PATTERN="$1"
OUTPUT_DIR="${2:-.}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Find experiment directory in outputs/
# Typically: outputs/YYYY-MM-DD/HH-MM-SS/experiment_name_with_timestamp/
FOUND_DIRS=()
while IFS= read -r -d '' dir; do
  if [[ "$dir" == *"$PATTERN"* ]]; then
    FOUND_DIRS+=("$dir")
  fi
done < <(find outputs -type d -name "*$PATTERN*" -print0 2>/dev/null)

if [[ ${#FOUND_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: No experiment directories found matching pattern: $PATTERN"
  echo "Searched in: $(pwd)/outputs/"
  exit 2
fi

if [[ ${#FOUND_DIRS[@]} -gt 1 ]]; then
  echo "WARNING: Found ${#FOUND_DIRS[@]} matching directories. Using the first one:"
  echo "  ${FOUND_DIRS[0]}"
fi

EXPERIMENT_DIR="${FOUND_DIRS[0]}"
EXPERIMENT_NAME=$(basename "$EXPERIMENT_DIR")

# Create a temporary working directory
TEMP_COLLECT_DIR=$(mktemp -d)
trap "rm -rf $TEMP_COLLECT_DIR" EXIT

COLLECT_DIR="$TEMP_COLLECT_DIR/$EXPERIMENT_NAME"
mkdir -p "$COLLECT_DIR"

echo "Collecting artifacts from: $EXPERIMENT_DIR"
echo "Experiment name: $EXPERIMENT_NAME"

# 1. Copy experiment directory (contains checkpoints, config, JSON)
if [[ -d "$EXPERIMENT_DIR" ]]; then
  echo "  → Copying experiment outputs..."
  cp -r "$EXPERIMENT_DIR" "$COLLECT_DIR/outputs"
fi

# 2. Find and copy relevant CSV files from logs_thesis/
if [[ -d "logs_thesis" ]]; then
  mkdir -p "$COLLECT_DIR/logs_thesis"
  # Match CSV files related to this experiment (loose matching by task/algo/seed)
  while IFS= read -r csv_file; do
    if [[ -f "$csv_file" ]]; then
      cp "$csv_file" "$COLLECT_DIR/logs_thesis/"
      echo "  → Copied CSV: $(basename "$csv_file")"
    fi
  done < <(find logs_thesis -maxdepth 1 -type f -name "*.csv" 2>/dev/null | head -20)
fi

# 3. Copy JSONL logs if they exist
if [[ -d "logs_thesis" ]]; then
  while IFS= read -r jsonl_file; do
    if [[ -f "$jsonl_file" ]]; then
      cp "$jsonl_file" "$COLLECT_DIR/logs_thesis/"
      echo "  → Copied JSONL: $(basename "$jsonl_file")"
    fi
  done < <(find logs_thesis -maxdepth 1 -type f -name "*.jsonl" 2>/dev/null | head -20)
fi

# 4. Copy run log if exists
if [[ -f "outputs/$(dirname "$EXPERIMENT_DIR" | xargs basename)/smacv2_run.log" ]]; then
  cp "outputs/$(dirname "$EXPERIMENT_DIR" | xargs basename)/smacv2_run.log" "$COLLECT_DIR/"
  echo "  → Copied run log"
fi

# 5. Create a README with metadata
cat > "$COLLECT_DIR/README.txt" << 'EOF'
EXPERIMENT RESULTS PACKAGE
==========================

This package contains all artifacts from a BenchMARL training run.

CONTENTS:
---------
outputs/           - Main experiment output (checkpoints, config, W&B logs, JSON metrics)
logs_thesis/       - CSV and JSONL training metrics for thesis analysis
smacv2_run.log     - Training run log
README.txt         - This file

KEY FILES:
----------
outputs/<exp_name>/checkpoints/checkpoint_*.pt
  → Training checkpoints saved at regular intervals

outputs/<exp_name>/<exp_name>.json
  → Experiment metadata and evaluation results in JSON format

outputs/<exp_name>/config.pkl
  → Full experiment configuration (Hydra config + algorithm config)

outputs/<exp_name>/wandb/
  → Weights & Biases logs (if enabled)

logs_thesis/*.csv
  → Training metrics in CSV format for easy plotting and analysis

IMPORTANT METRICS IN CSV:
------------------------
- frame: Total frames processed
- step: Training iteration number
- train_return_mean: Mean return during training
- train_intrinsic_reward_mean: Mean intrinsic reward (if applicable)
- eval_return_mean: Mean return during evaluation
- eval_return_std: Standard deviation of evaluation return
- eval_win_rate: Win rate on SMACv2 tasks
- current_beta1, current_beta2: DEMIR hyperparameter values (if applicable)
- loss_idm: IDM auxiliary loss (DEMIR only)
- loss_barlow_twins: Barlow Twins decorrelation loss (DEMIR only)

USAGE:
------
1. Download this ZIP to your machine
2. Extract it
3. Open Python/Jupyter and load:
   - checkpoints for inference or fine-tuning
   - CSV files for plotting/analysis
   - JSON for detailed results
   - W&B folder for online dashboard

QUESTIONS:
----------
For questions about the experiment setup, check the config.pkl file.
For questions about metrics, see the CSV headers and the README.

Generated: $(date)
EOF

echo "  → Created README"

# Create ZIP file
ZIP_PATH="$OUTPUT_DIR/${EXPERIMENT_NAME}.zip"
cd "$TEMP_COLLECT_DIR"
zip -r "$ZIP_PATH" "$EXPERIMENT_NAME" > /dev/null 2>&1

echo ""
echo "✓ COLLECTION COMPLETE"
echo "  ZIP file: $ZIP_PATH"
echo "  Size: $(du -h "$ZIP_PATH" | cut -f1)"
echo ""
echo "To download this file from the cluster:"
echo "  scp user@host:$ZIP_PATH /local/path/"
