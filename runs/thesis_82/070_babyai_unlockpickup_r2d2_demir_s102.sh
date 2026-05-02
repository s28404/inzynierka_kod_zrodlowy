#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/run_one_minigrid.sh" "BabyAI-UnlockPickup-v0" "r2d2_demir" "102"
