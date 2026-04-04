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

#ALGOS=("demir" "qmix" "ngu" "rnd")
ALGOS=("demir")

for algo in "${ALGOS[@]}"; do
	echo "==========================================================="
	echo "Running algorithm: $algo"
	echo "==========================================================="

	python fine_tuned/logic_env/logic_env_run.py \
		algorithm=$algo \
		task=logic_env/synchronized \
		experiment.off_policy_n_envs_per_worker=4 \
		experiment.off_policy_collected_frames_per_batch=1000 \
		experiment.parallel_collection=false \
		experiment.render=false \
		experiment.buffer_device=cpu \
		experiment.off_policy_memory_size=200000 \
		experiment.max_n_frames=10000000 \
		seed=1

	# brief pause between runs
	sleep 1
done


