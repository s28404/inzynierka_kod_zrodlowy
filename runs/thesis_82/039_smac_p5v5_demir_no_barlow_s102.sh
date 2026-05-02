#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/run_one_smac.sh" "smacv2/protoss_5_vs_5" "demir" "102" algorithm.beta1=0.7 algorithm.beta2=0.3 algorithm.encoder_type=idm algorithm.barlow_scale=0.0
