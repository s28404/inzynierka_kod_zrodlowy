#!/usr/bin/env bash
set -euo pipefail

# Start a Ray head node on this machine. Run inside WSL or a Linux shell.

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <YOUR_IP>"
    echo "For example, if your machine's IP is 172.17.107.3, run:"
    echo "  $0 172.17.107.3"
    exit 2
fi
HEAD_IP=$1

cd "$(dirname "$0")/../../"

export PATH="$HOME/.local/bin:$PATH"

if command -v uv >/dev/null 2>&1; then
    echo "Using uv to sync environment (preferred)..."
    uv sync || true
    RAY_CMD="uv run ray"
else
    echo "Installing Ray (user install)..."
    python3 -m pip install --user "ray[default]"
    RAY_CMD="ray"
fi

echo "Stopping any existing Ray instance..."
$RAY_CMD stop -f || true

echo "Starting Ray head..."
$RAY_CMD start --head \
    --node-ip-address=${HEAD_IP} \
    --port=6379 \
    --dashboard-host=0.0.0.0

echo "Ray head started. Dashboard: http://${HEAD_IP}:8265"
