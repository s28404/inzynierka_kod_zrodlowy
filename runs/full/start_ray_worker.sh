#!/usr/bin/env bash
set -euo pipefail

# Start a Ray worker and connect it to a head. Usage:
# ./start_ray_worker.sh <HEAD_IP>

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <HEAD_IP>"
    exit 2
fi
HEAD_IP=$1

cd "$(dirname "$0")/../../"

echo "Installing Ray (user install)..."
python3 -m pip install --user "ray[default]"

echo "Stopping any existing Ray instance..."
ray stop -f || true

echo "Starting Ray worker and connecting to ${HEAD_IP}:6379"
ray start --address=${HEAD_IP}:6379 \
    --node-manager-port=8077 \
    --object-manager-port=8076 \
    --min-worker-port=11000 \
    --max-worker-port=11050

echo "Ray worker started and connected to head at ${HEAD_IP}:6379"
