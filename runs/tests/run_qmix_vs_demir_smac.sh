#!/bin/bash
#
# Skrypt: run_qmix_vs_demir_protoss10.sh
# Autor: Kajetan Frąckowiak (modyfikacja AI)
# Cel: Uruchomienie eksperymentów QMIX i DEMIR na mapie Protoss 10 vs 10
#
set -e

# Przejdź do katalogu głównego projektu
cd "$(dirname "$0")/../.."

ALGOS=("qmix" "demir")

for algo in "${ALGOS[@]}"; do
	echo "==========================================================="
	echo "Uruchamiam algorytm: $algo na smacv2/terran_10_vs_11"
	echo "==========================================================="


	# Parametry treningu dobrane dla Terran 10 vs 11 (mapa mniejsza - zwiększone parametry)
	python3 -u fine_tuned/smacv2/smacv2_run.py \
		task=smacv2/terran_10_vs_11 \
		algorithm=$algo \
		experiment.off_policy_n_envs_per_worker=1 \
		experiment.off_policy_collected_frames_per_batch=100 \
		experiment.parallel_collection=false \
		experiment.render=false \
		experiment.buffer_device=cpu \
		experiment.off_policy_memory_size=25000 \
		experiment.max_n_frames=500000 \
		seed=1

	# Krótkie opóźnienie między uruchomieniami
	sleep 1
done

echo "Wszystkie uruchomienia zakończone."