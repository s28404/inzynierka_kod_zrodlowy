#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/run_one_minigrid.sh" "BabyAI-KeyCorridorS4R3-v0" "r2d2_demir" "102" --demir-beta1 1.0 --demir-beta2 0.0 --demir-encoder-type idm
