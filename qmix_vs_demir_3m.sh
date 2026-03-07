#!/bin/bash
# Autor: Kajetan Frąckowiak, s28404 (2026) — praca inżynierska
# Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
# Opis: Test porównawczy QMIX vs DEMIR na mapie three_m (seed=1).
#  

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
PYTHON="$(dirname "$0")/.venv/bin/python"
DEVICE="cuda"
MAP="three_m"
SEED=1
FRAMES=600000

echo "=== QMIX vs DEMIR | mapa: $MAP | seed=$SEED | frames=$FRAMES ==="

echo "--- [1/2] QMIX (baseline) ---"
"$PYTHON" fine_tuned/smacv2/smacv2_run.py \
    algorithm=qmix task=smacv2/$MAP \
    experiment.max_n_frames=$FRAMES \
    experiment.train_device=$DEVICE \
    experiment.buffer_device=cpu \
    experiment.off_policy_memory_size=100000 \
    experiment.off_policy_n_envs_per_worker=2 \
    experiment.checkpoint_at_end=true \
    seed=$SEED

echo "--- [2/2] DEMIR ---"
"$PYTHON" fine_tuned/smacv2/smacv2_run.py \
    algorithm=demir task=smacv2/$MAP \
    experiment.max_n_frames=$FRAMES \
    experiment.train_device=$DEVICE \
    experiment.buffer_device=cpu \
    experiment.off_policy_memory_size=100000 \
    experiment.off_policy_n_envs_per_worker=2 \
    experiment.checkpoint_at_end=true \
    seed=$SEED

echo "=== GOTOWE. Sprawdź wyniki na WandB: projekt kod_zrodlowy_demir ==="
