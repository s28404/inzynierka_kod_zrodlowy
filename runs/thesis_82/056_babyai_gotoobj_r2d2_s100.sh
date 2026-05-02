#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/run_one_minigrid.sh" "BabyAI-GoToObj-v0" "r2d2" "100"
