#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/run_one_smac.sh" "smacv2/protoss_10_vs_11" "rnd" "101"
