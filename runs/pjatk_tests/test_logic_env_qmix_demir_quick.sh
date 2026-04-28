#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../../"

PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python3}"
SEED="${SEED:-2}"
LOGIC_TASK="${LOGIC_TASK:-logic_env/synchronized}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

ALGOS=("demir")

echo "=== QUICK SSH TEST: Logic Env (QMIX + DEMIR) ==="

for algo in "${ALGOS[@]}"; do
  echo "[logic-quick] task=${LOGIC_TASK} algorithm=${algo}"
  "$PYTHON_BIN" -u fine_tuned/logic_env/logic_env_run.py \
    task="${LOGIC_TASK}" \
    algorithm="${algo}" \
    experiment.max_n_frames=6000 \
    experiment.evaluation_interval=6000 \
    experiment.evaluation_episodes=1 \
    experiment.checkpoint_interval=6000 \
    experiment.checkpoint_at_end=true \
    experiment.keep_checkpoints_num=3 \
    seed="${SEED}"
  sleep 1
done

echo "=== DONE: Logic Env quick SSH tests ==="
