#!/bin/bash

# Quick command to test everything in one line
# Run: bash runs/smacv2_21/test_oneliner.sh

cd ~/projects/inzynierka_kod_zrodlowy

echo "🧪 TESTING: Quick 10K frame training + collection"
echo ""

# Train for 10K frames
echo "Step 1: Training..."
timeout 120 uv run python -u fine_tuned/smacv2/smacv2_run.py \
  task=smacv2/protoss_5_vs_5 \
  algorithm=qmix \
  seed=999 \
  experiment.sampling_device=cpu \
  experiment.train_device=cuda \
  experiment.buffer_device=cpu \
  experiment.max_n_frames=10000 \
  experiment.evaluation_interval=2000 \
  experiment.evaluation_episodes=5 \
  experiment.parallel_collection=false \
  experiment.off_policy_n_envs_per_worker=4 \
  experiment.off_policy_collected_frames_per_batch=2000 \
  2>&1 | tail -30

echo ""
echo "Step 2: Collecting results..."

# Find latest experiment with seed 999
EXP=$(find outputs -maxdepth 3 -type d -name "*seed999*" 2>/dev/null | sort -V | tail -1)
if [[ -n "$EXP" ]]; then
  EXP_NAME=$(basename "$EXP")
  echo "Found: $EXP_NAME"
  
  # Collect it
  ./runs/smacv2_21/collect_results.sh "$EXP_NAME" /tmp 2>&1 | grep -E "Collection|ZIP|Size|Success"
  
  # Show what's in the ZIP
  ZIP="/tmp/${EXP_NAME}.zip"
  if [[ -f "$ZIP" ]]; then
    echo ""
    echo "✅ ZIP created: $(du -h $ZIP | cut -f1)"
    echo ""
    echo "Contents:"
    unzip -l "$ZIP" | grep -E "\.(json|pt|csv|txt|log)$" | head -20
  fi
else
  echo "❌ No experiment found with seed 999"
fi
