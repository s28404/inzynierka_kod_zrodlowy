#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$DIR/run_one_logic.sh" "demir" "100" algorithm.beta1=0.7 algorithm.beta2=0.3 algorithm.encoder_type=idm
