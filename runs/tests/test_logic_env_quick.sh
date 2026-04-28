#!/bin/bash
# --- MODYFIKACJE / MODIFICATIONS ---
# Autor: Kajetan Frąckowiak, s28404
# Opis: Skrypt sanity check dla autorskiego środowiska SynchronizedFactory.
# Obniża parametry do dosłownie jednej iteracji walidacyjnej, zmuszając 
# thesis_logger do wczesnego wyplucia nagromadzonych metryk Reward Machine.
# ---

set -e # Zatrzymanie skryptu w razie błędu

# Zmiana katalogu roboczego na główny katalog projektu
cd "$(dirname "$0")/../.."

ALGOS=("qmix" "demir" "ngu" "rnd")

echo "=== ROZPOCZĘCIE SZYBKICH TESTÓW LOGIC ENV (SynchronizedFactory) ==="

mkdir -p logs_thesis

for algo in "${ALGOS[@]}"; do
    echo "==========================================================="
    echo "🚀 Uruchamianie testu 1-epokowego środowiska: synchronized (Factory) | algorytm: $algo"
    echo "==========================================================="
    
    # Dopasowanie evaluation_interval do wielokrotności collected_frames_per_batch (domyślnie 6000 dla bazowych configów TorchRL)
    /home/kajetan/Documents/inzynierka_kod_zrodlowy/.venv/bin/python3 fine_tuned/logic_env/logic_env_run.py \
        task=logic_env/synchronized \
        algorithm=$algo \
        experiment.max_n_frames=12000 \
        experiment.evaluation_interval=6000 \
        experiment.evaluation_episodes=1 \
        seed=2
        
        
    echo "✅ Sukces uruchomieniowy: test z algorytmem $algo zapisał JSON."
    sleep 1
done

echo "=== ZAKOŃCZONO SZYBKIE TESTY LOGIC ENV ==="
echo "Skuteczność logowania specyficznych autorskich metryk środowiskowych (rm_state itp)"
echo "można potwierdzić upewniając się, że pliki znajdują się w folderze './logs_thesis/'."
