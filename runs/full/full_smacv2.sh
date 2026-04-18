#!/bin/bash

cd "$(dirname "$0")/../../"

# Mapy wybrane do Direct Science (Prosta, Asymetryczna, Inna Frakcja)
TASKS=("smacv2/protoss_5_vs_5" "smacv2/protoss_10_vs_11" "smacv2/terran_10_vs_10")
# Algorytmy do porównania
ALGOS=("qmix" "rnd" "ngu" "demir")
# Domyślnie wszystkie eksperymentują z GRU (zamiast MLP-u)
# Dla wiekszego sprzetu mozna zwikeszyc off_policy_memory_size
for task in "${TASKS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        echo ""
        echo "-----------------------------------------------------------"
        echo "URUCHAMIAM: $algo NA $task"
        echo "-----------------------------------------------------------"

        # nice -n 19 nadaje najniższy priorytet procesora dla treningu, 
        # dzięki czemu system i przeglądarka pozostaną responsywne.
        
        python3 -u fine_tuned/smacv2/smacv2_run.py \
                task=$task \
                algorithm=$algo \
                seed=1 \
            experiment.sampling_device=cpu \
            experiment.train_device=cpu \
                experiment.lr=0.00005 \
                experiment.max_n_frames=10000000 \
                experiment.off_policy_memory_size=150000 \
                experiment.off_policy_collected_frames_per_batch=500 \
                experiment.off_policy_train_batch_size=64 \
                experiment.evaluation_interval=250000 \
                experiment.evaluation_episodes=32 \
                experiment.buffer_device=cpu \
                experiment.checkpoint_interval=1000000 \
                experiment.clip_grad_val=10.0 \
                experiment.parallel_collection=false \
                experiment.render=false

        echo "-----------------------------------------------------------"
        echo "ZAKOŃCZONO: $algo na $task."
        echo "Chłodzenie systemu przez 90 sekund..."
        echo "-----------------------------------------------------------"
        
        # Dłuższa przerwa na schłodzenie starego procesora i7-4770
        sleep 90
    done
done

echo "==========================================================="
echo "WSZYSTKIE EKSPERYMENTY ZAKOŃCZONE POMYŚLNIE"
echo "Sprawdź wyniki w serwisie WandB lub w folderze outputs/"
echo "==========================================================="