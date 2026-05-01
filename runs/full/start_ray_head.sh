#!/usr/bin/env bash
set -euo pipefail

# Start a Ray head node on this machine. Run inside WSL or a Linux shell.
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
    --port=6379 \
    --dashboard-port=8265 \
    --node-manager-port=8077 \
    --object-manager-port=8076 \
    --min-worker-port=11000 \
    --max-worker-port=11050

echo "Ray head started. Dashboard: http://localhost:8265 (or use SSH tunnel/VPN to access remotely)"
