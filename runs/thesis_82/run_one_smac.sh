#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <task> <algo> <seed> [hydra_overrides...]"
  exit 2
fi

TASK="$1"
ALGO="$2"
SEED="$3"
shift 3

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-uv run python}"
MIN_RAM_MB="${MIN_RAM_MB:-10000}"
MIN_GPU_FREE_MB="${MIN_GPU_FREE_MB:-5000}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"

available_ram_mb="$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo || echo 0)"
if (( available_ram_mb < MIN_RAM_MB )); then
  echo "ERROR: Available RAM ${available_ram_mb} MB < MIN_RAM_MB=${MIN_RAM_MB}."
  exit 3
fi

gpu_ok=0
gpu_free_mb=0
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ' || echo 0)"
  if [[ "$gpu_free_mb" =~ ^[0-9]+$ ]] && (( gpu_free_mb >= MIN_GPU_FREE_MB )); then
    gpu_ok=1
  fi
fi

if (( gpu_ok == 1 )); then
  train_device="cuda"
  sampling_device="cpu"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
else
  if [[ "$REQUIRE_GPU" == "1" ]]; then
    echo "ERROR: GPU not available or too little free memory (${gpu_free_mb} MB)."
    exit 4
  fi
  train_device="cpu"
  sampling_device="cpu"
fi

buffer_device="${BUFFER_DEVICE:-cpu}"
AUTO_TUNE="${AUTO_TUNE:-1}"

off_policy_n_envs_per_worker="${OFF_POLICY_N_ENVS_PER_WORKER:-}"
off_policy_collected_frames_per_batch="${OFF_POLICY_COLLECTED_FRAMES_PER_BATCH:-}"
off_policy_train_batch_size="${OFF_POLICY_TRAIN_BATCH_SIZE:-}"
off_policy_memory_size="${OFF_POLICY_MEMORY_SIZE:-}"
extra_args=()

if [[ "$AUTO_TUNE" == "1" ]]; then
  cpu_cores="$(nproc || echo 1)"

  if [[ -z "$off_policy_n_envs_per_worker" ]]; then
    if (( cpu_cores >= 16 )); then
      off_policy_n_envs_per_worker=8
    elif (( cpu_cores >= 8 )); then
      off_policy_n_envs_per_worker=6
    else
      off_policy_n_envs_per_worker=4
    fi
  fi

  if [[ -z "$off_policy_collected_frames_per_batch" ]]; then
    if (( off_policy_n_envs_per_worker >= 8 )); then
      off_policy_collected_frames_per_batch=1000
    elif (( off_policy_n_envs_per_worker >= 6 )); then
      off_policy_collected_frames_per_batch=750
    else
      off_policy_collected_frames_per_batch=500
    fi
  fi

  if [[ -z "$off_policy_train_batch_size" ]]; then
    if [[ "$train_device" == "cuda" ]]; then
      if (( gpu_free_mb >= 7000 )); then
        off_policy_train_batch_size=128
      elif (( gpu_free_mb >= 5500 )); then
        off_policy_train_batch_size=96
      else
        off_policy_train_batch_size=64
      fi
    else
      off_policy_train_batch_size=64
    fi
  fi

  if [[ -z "$off_policy_memory_size" ]]; then
    if (( available_ram_mb >= 14000 )); then
      off_policy_memory_size=150000
    elif (( available_ram_mb >= 11000 )); then
      off_policy_memory_size=120000
    elif (( available_ram_mb >= 9000 )); then
      off_policy_memory_size=80000
    else
      off_policy_memory_size=40000
    fi
  fi
fi

[[ -n "$off_policy_n_envs_per_worker" ]] && extra_args+=("experiment.off_policy_n_envs_per_worker=${off_policy_n_envs_per_worker}")
[[ -n "$off_policy_collected_frames_per_batch" ]] && extra_args+=("experiment.off_policy_collected_frames_per_batch=${off_policy_collected_frames_per_batch}")
[[ -n "$off_policy_train_batch_size" ]] && extra_args+=("experiment.off_policy_train_batch_size=${off_policy_train_batch_size}")
[[ -n "$off_policy_memory_size" ]] && extra_args+=("experiment.off_policy_memory_size=${off_policy_memory_size}")

echo "RUN: algo=${ALGO} task=${TASK} seed=${SEED}"
echo "Devices: sampling=${sampling_device}, train=${train_device}, buffer=${buffer_device}"
[[ "$AUTO_TUNE" == "1" ]] && echo "Auto-tune: n_envs=${off_policy_n_envs_per_worker}, frames_per_batch=${off_policy_collected_frames_per_batch}, train_batch=${off_policy_train_batch_size}, memory_size=${off_policy_memory_size}"

exec ${PYTHON_BIN} -u fine_tuned/smacv2/smacv2_run.py \
  task="${TASK}" \
  algorithm="${ALGO}" \
  seed="${SEED}" \
  experiment.sampling_device="${sampling_device}" \
  experiment.train_device="${train_device}" \
  experiment.buffer_device="${buffer_device}" \
  "${extra_args[@]}" \
  "$@"
