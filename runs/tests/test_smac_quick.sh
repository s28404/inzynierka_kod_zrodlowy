#!/bin/bash
# --- MODYFIKACJE / MODIFICATIONS ---
# Autor: Kajetan Frąckowiak, s28404 (2026) — praca inżynierska
# Opis: Skrypt do szybkiego testowania zapisu logów i działania środowiska SMAC z BenchMARL.
# Zawiera 3 zadania (corridor, six_h_vs_8z, protoss_10_vs_10) i 4 algorytmy na zaledwie kilku krokach.
# Pętla uruchamia bardzo krótki trening (2000 ramek), aby wymusić przynajmniej jedną ewaluację
# i upewnić się, że plik JSON w 'logs_thesis' poprawnie operuje danymi wandb.
# ---

set -e # Zatrzymanie skryptu w razie krytycznego błędu

# Zmiana katalogu roboczego na główny katalog projektu
cd "$(dirname "$0")/../.."

MAPS=("corridor" "six_h_vs_8z" "protoss_10_vs_10")
ALGOS=("qmix" "demir" "ngu" "rnd")

echo "=== ROZPOCZĘCIE SZYBKICH TESTÓW (Sanity Checks) SMAC ==="

# Tworzymy log w root folderze jako backup
mkdir -p logs_thesis

for map in "${MAPS[@]}"; do
    for algo in "${ALGOS[@]}"; do
        echo "==========================================================="
        echo "🚀 Szybki test uruchomieniowy: mapa=$map | algorytm=$algo"
        echo "==========================================================="
        
        # Ograniczenie ewaluacji na potrzeby unit testu loggera JSON z uwzględnieniem collected_frames_per_batch
        python fine_tuned/smacv2/smacv2_run.py \
            task=smacv2/$map \
            algorithm=$algo \
            experiment.max_n_frames=12000 \
            experiment.evaluation_interval=6000 \
            experiment.evaluation_episodes=1
            
        echo "✅ Sukces uruchomieniowy: test $map z algorytmem $algo przeszedł pomyślnie."
        sleep 1
    done
done

echo "=== WYKONANO WSZYSTKIE SZYBKIE TESTY SMAC ==="
echo "Sprawdź pliki formatu JSON wygenerowane w katalogu './logs_thesis/'!"
