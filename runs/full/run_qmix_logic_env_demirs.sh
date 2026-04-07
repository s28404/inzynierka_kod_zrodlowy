#!/bin/bash
#
# Copyright (c) 2026 Kajetan Frąckowiak, s28404
#
# Projekt: Algorytm DEMIR dla SMACv2 i Custom Logic Environment
# Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
# Praca Inżynierska (2026)
#
# Opis: Skrypt treningowy dla QMIX algorithm na Logic Environment (SynchronizedFactory).
#

set -e # Zatrzymanie skryptu w razie błędu

# Zmiana katalogu roboczego na główny katalog projektu
cd "$(dirname "$0")/../.."

### Run DEMIR algorithm

SCALES=(0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for scale in "${SCALES[@]}"; do
    echo "==========================================================="
    echo "Running with scale: $scale"
    echo "==========================================================="

    /home/kajetan/Documents/inzynierka_kod_zrodlowy/.venv/bin/python3 -u fine_tuned/logic_env/logic_env_run.py \
        algorithm="demir" \
        task=logic_env/synchronized \
        experiment.off_policy_n_envs_per_worker=4 \
        experiment.off_policy_collected_frames_per_batch=1000 \
        experiment.parallel_collection=false \
        experiment.render=false \
        experiment.buffer_device=cpu \
        experiment.off_policy_memory_size=200000 \
        experiment.max_n_frames=10000000 \
        algorithm.demir_scale=$scale \
        seed=1

    # brief pause between runs
    sleep 1
done