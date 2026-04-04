#!/bin/bash
#
# Skrypt: run_qmix_vs_demir_protoss10.sh
# Autor: Kajetan Frąckowiak (modyfikacja AI)
# Cel: Uruchomienie eksperymentów QMIX i DEMIR na mapie Protoss 10 vs 10
#
set -e

# Przejdź do katalogu głównego projektu
cd "$(dirname "$0")/../.."

#ALGOS=("qmix" "demir")
ALGOS=("qmix")

for algo in "${ALGOS[@]}"; do
	echo "==========================================================="
	echo "Uruchamiam algorytm: $algo na smacv2/protoss_10_vs_10"
	echo "==========================================================="


	# Parametry treningu dobrane z myślą o starszym sprzęcie (GTX1080, i7-4770, DDR3 16GB)
	python3 -u fine_tuned/smacv2/smacv2_run.py \
		task=smacv2/protoss_10_vs_10 \
		algorithm=$algo \
		experiment.off_policy_n_envs_per_worker=2 \
		experiment.off_policy_collected_frames_per_batch=1000 \
		experiment.parallel_collection=false \
		experiment.render=false \
		experiment.buffer_device=cpu \
		experiment.off_policy_memory_size=200000 \
		experiment.max_n_frames=5000000 \
		seed=1 \
		$DEVICE_ARG

	# Krótkie opóźnienie między uruchomieniami
	sleep 1
done

echo "Wszystkie uruchomienia zakończone."