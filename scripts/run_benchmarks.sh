#!/bin/bash
# Skrypt do pełnego benchmarkowania aplikacji i generowania wykresów.

cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
RESULTS_DIR="$PROJECT_DIR/results"
mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR"/*.txt

SIZES=(500 1000 2000)
ITERS=1000
STENCIL=5

echo "=== ROZPOCZĘCIE PEŁNEGO BENCHMARKU ==="

for SIZE in "${SIZES[@]}"; do
    echo "--- Testowanie dla siatki ${SIZE}x${SIZE} ---"

    # 1. Sekwencyjnie
    echo "Uruchamianie Sequential (Rozmiar: $SIZE)"
    ./build/bin/diffusion_seq --size $SIZE --iterations $ITERS --stencil $STENCIL --benchmark > "$RESULTS_DIR/seq_${SIZE}.txt"

    # 2. OpenMP (wątki: 1, 2, 4, 8)
    for THREADS in 1 2 4 8; do
        echo "Uruchamianie OpenMP (Rozmiar: $SIZE, Wątki: $THREADS)"
        OMP_NUM_THREADS=$THREADS ./build/bin/diffusion_omp --size $SIZE --iterations $ITERS --stencil $STENCIL --threads $THREADS --benchmark >> "$RESULTS_DIR/omp_${SIZE}.txt"
    done

    # 3. MPI (procesy: 1, 2, 4, 8)
    for PROCS in 1 2 4 8; do
        echo "Uruchamianie MPI (Rozmiar: $SIZE, Procesy: $PROCS)"
        mpirun --allow-run-as-root -np $PROCS --oversubscribe ./build/bin/diffusion_mpi --size $SIZE --iterations $ITERS --stencil $STENCIL --benchmark >> "$RESULTS_DIR/mpi_${SIZE}.txt"
    done

    # 4. CUDA (blok: 8, 16, 32)
    for BLOCK in 8 16 32; do
        echo "Uruchamianie CUDA (Rozmiar: $SIZE, Block size: $BLOCK)"
        ./build/bin/diffusion_cuda --size $SIZE --iterations $ITERS --stencil $STENCIL --block-size $BLOCK --benchmark >> "$RESULTS_DIR/cuda_${SIZE}.txt"
    done

done

echo "=== BENCHMARK ZAKOŃCZONY ==="
echo "Generowanie wykresów za pomocą python3 visualization/benchmark_plots.py..."

# Tworzenie wykresów
mkdir -p "$RESULTS_DIR/wykresy"
python3 visualization/benchmark_plots.py "$RESULTS_DIR/" --output "$RESULTS_DIR/wykresy/"

echo "Wykresy zostały wygenerowane pomyślnie w katalogu $RESULTS_DIR/wykresy/"
