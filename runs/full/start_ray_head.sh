#!/usr/bin/env bash
set -euo pipefail

# Start a Ray head node on this machine. Run inside WSL or a Linux shell.
cd "$(dirname "$0")/../../"

echo "Installing Ray (user install)..."
python3 -m pip install --user "ray[default]"

echo "Stopping any existing Ray instance..."
ray stop -f || true

echo "Starting Ray head..."
ray start --head \
    --port=6379 \
    --dashboard-port=8265 \
    --node-manager-port=8077 \
    --object-manager-port=8076 \
    --min-worker-port=11000 \
    --max-worker-port=11050

echo "Ray head started. Dashboard: http://localhost:8265 (or use SSH tunnel/VPN to access remotely)"
