#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
SEED="${SEED:-1}"

TASKS=(
  "smacv2/protoss_5_vs_5"
  "smacv2/terran_10_vs_10"
  "smacv2/protoss_10_vs_11"
)

ALGOS=(
  "qmix"
  "rnd"
  "ngu"
  "demir"
)

for task in "${TASKS[@]}"; do
  for algo in "${ALGOS[@]}"; do
    echo "==========================================================="
    echo "RUN: algo=${algo} task=${task} seed=${SEED} (CPU-only)"
    echo "==========================================================="

    "${PYTHON_BIN}" -u fine_tuned/smacv2/smacv2_run.py \
      task="${task}" \
      algorithm="${algo}" \
      seed="${SEED}" \
      experiment.sampling_device=cpu \
      experiment.train_device=cpu \
      experiment.buffer_device=cpu \
      experiment.parallel_collection=false \
      experiment.render=false \
      experiment.off_policy_n_envs_per_worker=1 \
      experiment.off_policy_collected_frames_per_batch=1000 \
      experiment.off_policy_train_batch_size=128 \
      experiment.off_policy_memory_size=500000 \
      experiment.evaluation_interval=100000 \
      experiment.evaluation_episodes=20 \
      experiment.max_n_frames=2000000

    sleep 1
  done
done

echo "All SMACv2 QMIX intrinsic comparison runs finished."
