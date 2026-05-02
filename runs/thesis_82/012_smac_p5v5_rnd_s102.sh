#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/run_one_smac.sh" "smacv2/protoss_5_vs_5" "rnd" "102"
