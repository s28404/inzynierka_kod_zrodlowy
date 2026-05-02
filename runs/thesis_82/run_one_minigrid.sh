#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <env_id> <variant> <seed> [extra_cli_args...]"
  exit 2
fi

ENV_ID="$1"
VARIANT="$2"
SEED="$3"
shift 3

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"

exec ${PYTHON_BIN} -u fine_tuned/minigrid/r2d2_run.py \
  --env-id "$ENV_ID" \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --device auto \
  "$@"
