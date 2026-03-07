#!/bin/bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export CUDA_VISIBLE_DEVICES=0
DEVICE="cuda"

MAPS=("simple_spread" "simple_tag" "simple_crypto")
SEEDS=(1 2 3)

echo "START EKSPERYMENTÓW MPE/VMAS x4 warianty"

run_experiment() {
    local ALGO=$1
    local MAP=$2
    local SEED=$3
    echo "--- $ALGO na $MAP | seed=$SEED ---"
    python fine_tuned/vmas/vmas_run.py \
        algorithm=$ALGO task=vmas/$MAP \
        experiment.max_n_frames=2000000 \
        experiment.train_device=$DEVICE \
        experiment.buffer_device=cpu \
        experiment.render=false \
        experiment.parallel_collection=true \
        experiment.off_policy_n_envs_per_worker=16 \
        experiment.off_policy_memory_size=50000 \
        experiment.checkpoint_interval=480000 \
        experiment.evaluation_interval=120000 \
        seed=$SEED
}

for MAP in "${MAPS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        run_experiment "qmix"  $MAP $SEED
        run_experiment "rnd"   $MAP $SEED
        run_experiment "ngu"   $MAP $SEED
        run_experiment "demir" $MAP $SEED
    done
done

echo "KONIEC. Sprawdź wyniki na WandB."