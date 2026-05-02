#!/usr/bin/env bash
set -euo pipefail

# Mock training test script — runs a quick 1-minute training, then collects results
# Tests: GPU detection, auto-tuning, training loop, logging, collection, and ZIP creation
# Usage: ./test_quick_train.sh

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "=================================="
echo "MOCK QUICK TRAINING TEST"
echo "=================================="
echo ""

# Configuration for quick test
TASK="smacv2/protoss_5_vs_5"
ALGO="qmix"
SEED="999"  # Use 999 to mark as test
MAX_FRAMES="10000"  # Very short for quick testing
EVAL_INTERVAL="5000"  # Evaluate halfway through

echo "[1/3] Running quick training (${MAX_FRAMES} frames, should take ~1 min)..."
echo "  Task: $TASK"
echo "  Algorithm: $ALGO"
echo "  Seed: $SEED"
echo ""

# Run the training
PYTHON_BIN="${PYTHON_BIN:-uv run python}"
eval "$PYTHON_BIN" -u fine_tuned/smacv2/smacv2_run.py \
  task="$TASK" \
  algorithm="$ALGO" \
  seed="$SEED" \
  experiment.sampling_device=cpu \
  experiment.train_device=cuda \
  experiment.buffer_device=cpu \
  experiment.max_n_frames="$MAX_FRAMES" \
  experiment.evaluation_interval=2000 \
  experiment.evaluation_episodes=5 \
  experiment.checkpoint_interval=0 \
  experiment.parallel_collection=false \
  experiment.off_policy_n_envs_per_worker=4 \
  experiment.off_policy_collected_frames_per_batch=2000 \
  2>&1 | tail -50

echo ""
echo "[2/3] Collecting results into ZIP..."

# Find the experiment directory (should be the most recent)
LATEST_EXPERIMENT=$(find outputs -maxdepth 3 -type d -name "*seed${SEED}*" 2>/dev/null | sort -V | tail -1)

if [[ -z "$LATEST_EXPERIMENT" ]]; then
  echo "ERROR: Could not find experiment directory for seed $SEED"
  exit 1
fi

EXPERIMENT_NAME=$(basename "$LATEST_EXPERIMENT")
echo "  Found: $EXPERIMENT_NAME"

# Use collect_results script
if ! ./runs/smacv2_21/collect_results.sh "$EXPERIMENT_NAME" /tmp 2>&1 | grep -E "Collection|ZIP|Size"; then
  echo "WARNING: Collection may have failed, checking manually..."
fi

# Check what was created
ZIP_FILE="/tmp/${EXPERIMENT_NAME}.zip"
if [[ -f "$ZIP_FILE" ]]; then
  echo ""
  echo "[3/3] Verifying ZIP contents..."
  echo "  ZIP file: $ZIP_FILE"
  echo "  Size: $(du -h "$ZIP_FILE" | cut -f1)"
  echo ""
  echo "  Contents:"
  unzip -l "$ZIP_FILE" | head -40
  echo ""
  echo "✅ SUCCESS: All artifacts collected!"
else
  echo "ERROR: ZIP file not created"
  exit 2
fi

# Show what files are inside
echo ""
echo "Test Summary:"
echo "============="
echo ""

# Extract and check structure
TEMP_CHECK=$(mktemp -d)
unzip -q "$ZIP_FILE" -d "$TEMP_CHECK"

EXTRACTED_DIR=$(ls "$TEMP_CHECK")
echo "✓ Experiment directory: $EXTRACTED_DIR"

if [[ -d "$TEMP_CHECK/$EXTRACTED_DIR/outputs" ]]; then
  echo "✓ Outputs directory: EXISTS"
  
  # List what's in outputs
  ls -lh "$TEMP_CHECK/$EXTRACTED_DIR/outputs/" | grep -v "^d" | awk '{print "    " $9 " (" $5 ")"}'
fi

if [[ -d "$TEMP_CHECK/$EXTRACTED_DIR/logs_thesis" ]]; then
  echo "✓ Logs (CSV/JSONL): EXISTS"
  ls -1 "$TEMP_CHECK/$EXTRACTED_DIR/logs_thesis/" | head -5 | awk '{print "    " $0}'
else
  echo "⚠ Logs (CSV/JSONL): NOT FOUND (may be expected if logging not configured)"
fi

if [[ -f "$TEMP_CHECK/$EXTRACTED_DIR/README.txt" ]]; then
  echo "✓ README: EXISTS"
fi

rm -rf "$TEMP_CHECK"

echo ""
echo "=================================="
echo "✅ QUICK TEST COMPLETE!"
echo "=================================="
echo ""
echo "ZIP file ready for download:"
echo "  $ZIP_FILE"
echo ""
echo "Next: Download this ZIP, extract it, and verify:"
echo "  - Checkpoints in outputs/*/checkpoints/"
echo "  - JSON metrics in outputs/*/*.json"
echo "  - W&B logs in outputs/*/wandb/"
echo "  - CSV metrics in logs_thesis/ (if configured)"
