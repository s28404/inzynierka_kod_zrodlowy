#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <algo> <seed> [hydra_overrides...]"
  exit 2
fi

ALGO="$1"
SEED="$2"
shift 2

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"

exec ${PYTHON_BIN} -u fine_tuned/logic_env/logic_env_run.py \
  algorithm="${ALGO}" \
  task="logic_env/synchronized" \
  seed="${SEED}" \
  "$@"
