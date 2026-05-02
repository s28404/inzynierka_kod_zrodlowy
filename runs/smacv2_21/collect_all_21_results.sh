#!/usr/bin/env bash
set -euo pipefail

# Batch collector for all 21 experiments
# Collects results from each experiment into separate ZIP files
# Usage: ./collect_all_21_results.sh [output_dir]

OUTPUT_DIR="${1:-.}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

COLLECT_SCRIPT="runs/smacv2_21/collect_results.sh"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Collecting all 21 SMACv2 experiment results"
echo "=========================================="
echo ""

# List of all 21 experiment patterns
EXPERIMENTS=(
  "protoss_5_vs_5_qmix_seed100"
  "protoss_5_vs_5_qmix_seed101"
  "protoss_5_vs_5_qmix_seed102"
  "protoss_5_vs_5_demir_seed100"
  "protoss_5_vs_5_demir_seed101"
  "protoss_5_vs_5_demir_seed102"
  "protoss_5_vs_5_ngu_seed100"
  "protoss_5_vs_5_ngu_seed101"
  "protoss_5_vs_5_ngu_seed102"
  "protoss_5_vs_5_rnd_seed100"
  "protoss_5_vs_5_rnd_seed101"
  "protoss_5_vs_5_rnd_seed102"
  "protoss_10_vs_11_qmix_seed100"
  "protoss_10_vs_11_qmix_seed101"
  "protoss_10_vs_11_qmix_seed102"
  "protoss_10_vs_11_demir_seed100"
  "protoss_10_vs_11_demir_seed101"
  "protoss_10_vs_11_demir_seed102"
  "protoss_10_vs_11_rnd_seed100"
  "protoss_10_vs_11_rnd_seed101"
  "protoss_10_vs_11_rnd_seed102"
)

SUCCESS_COUNT=0
FAIL_COUNT=0

for i in "${!EXPERIMENTS[@]}"; do
  EXP="${EXPERIMENTS[$i]}"
  NUM=$((i + 1))
  
  echo "[$NUM/21] Collecting: $EXP"
  
  if "$COLLECT_SCRIPT" "$EXP" "$OUTPUT_DIR" 2>/dev/null; then
    ((SUCCESS_COUNT++))
    echo "        ✓ Success"
  else
    echo "        ✗ Not found or collection failed"
    ((FAIL_COUNT++))
  fi
done

echo ""
echo "=========================================="
echo "Collection Summary:"
echo "  Successful: $SUCCESS_COUNT / 21"
echo "  Failed/Not Found: $FAIL_COUNT / 21"
echo ""
echo "ZIP files saved to: $OUTPUT_DIR"
echo "Total size: $(du -sh "$OUTPUT_DIR" | cut -f1)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Download all ZIPs from $OUTPUT_DIR"
echo "  2. Extract each ZIP locally"
echo "  3. Process CSV files for plotting"
echo "  4. Load checkpoints for inference/fine-tuning if needed"
