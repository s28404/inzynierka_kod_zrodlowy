#!/bin/bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
DEVICE="cuda"

# Mapy SMACv2 i seedy
MAPS=("corridor" "six_h_vs_8z" "three_m")
SEEDS=(1 2 3)

echo "START EKSPERYMENTÓW QMIX x4 warianty (Elsevier)"

# Funkcja pomocnicza
run_experiment() {
    local ALGO=$1
    local MAP=$2
    local SEED=$3
    [[ "$MAP" == "three_m" ]] && FRAMES=5000000 || FRAMES=20000000
    echo "--- $ALGO na $MAP | seed=$SEED ---"
    python fine_tuned/smacv2/smacv2_run.py \
        algorithm=$ALGO task=smacv2/$MAP \
        experiment.max_n_frames=$FRAMES \
        experiment.train_device=$DEVICE \
        experiment.buffer_device=cpu \
        experiment.off_policy_memory_size=100000 \
        experiment.off_policy_n_envs_per_worker=2 \
        experiment.checkpoint_interval=5000000 \
        seed=$SEED
}

# 4 warianty x 3 mapy x 3 seedy = 36 runów
for MAP in "${MAPS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        run_experiment "qmix"  $MAP $SEED   # Baseline (brak intrinsic)
        run_experiment "rnd"   $MAP $SEED   # Rywal 1: Random Network Distillation
        run_experiment "ngu"   $MAP $SEED   # Rywal 2: Never Give Up
        run_experiment "demir" $MAP $SEED   # Nasz: DEMIR
    done
done

echo "KONIEC KOLEJKI. Sprawdź wyniki na WandB."