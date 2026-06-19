#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python3}"
SEED="${SEED:-2}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

ALGOS=("demir")
MAPS=(
  "protoss_5_vs_5"
  "protoss_10_vs_11"
  "terran_10_vs_10"
)

echo "=== QUICK SSH TEST: SMACv2 (DEMIR) ==="

for map_name in "${MAPS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    echo "[smacv2-quick] task=smacv2/${map_name} algorithm=${algo}"
    "$PYTHON_BIN" -u fine_tuned/smacv2/smacv2_run.py \
      task="smacv2/${map_name}" \
      algorithm="${algo}" \
      experiment.off_policy_n_envs_per_worker=1 \
      experiment.off_policy_collected_frames_per_batch=1000 \
      experiment.off_policy_train_batch_size=64 \
      experiment.parallel_collection=false \
      experiment.render=false \
      experiment.buffer_device=cpu \
      experiment.off_policy_memory_size=50000 \
      experiment.max_n_frames=1000 \
      experiment.evaluation_interval=1000 \
      experiment.evaluation_episodes=1 \
      experiment.checkpoint_interval=1000 \
      experiment.checkpoint_at_end=true \
      experiment.keep_checkpoints_num=3 \
      seed="${SEED}"
    sleep 1
  done
done

echo "=== DONE: SMACv2 quick SSH tests ==="
