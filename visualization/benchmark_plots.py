#!/usr/bin/env python3
"""
Generowanie wykresów wydajności: przyspieszenie, efektywność, porównanie.

Użycie:
    python benchmark_plots.py results/ --output wykresy/

Format pliku CSV (results.csv):
    typ,rozmiar,iteracje,stencil,watki/procesy,czas_s
    seq,1000,500,5,1,12.345
    omp,1000,500,5,2,6.789
    omp,1000,500,5,4,3.567
    ...
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import sys
import argparse
from collections import defaultdict


def _open_safe(filepath):
    """Otwiera plik z automatycznym wykryciem kodowania (UTF-8/UTF-16/Latin-1)."""
    for enc in ('utf-8', 'utf-16', 'latin-1'):
        try:
            with open(filepath, 'r', encoding=enc) as f:
                content = f.read()
            return content
        except (UnicodeDecodeError, UnicodeError):
            continue
    return ''


def load_results(results_dir):
    """Ładuje wyniki z plików CSV w katalogu."""
    results = []
    for fname in os.listdir(results_dir):
        if fname.endswith('.csv'):
            content = _open_safe(os.path.join(results_dir, fname))
            for row in csv.reader(content.splitlines()):
                if len(row) >= 6:
                    try:
                        results.append({
                            'type': row[0].strip(),
                            'size': int(row[1]),
                            'iterations': int(row[2]),
                            'stencil': int(row[3]),
                            'workers': int(row[4]),
                            'time': float(row[5])
                        })
                    except (ValueError, IndexError):
                        continue
    return results


def parse_stdout_results(results_dir):
    """Parsuje wyniki z plików tekstowych zawierających linie CSV:."""
    results = []
    for fname in os.listdir(results_dir):
        if fname.endswith('.txt') or fname.endswith('.log'):
            content = _open_safe(os.path.join(results_dir, fname))
            for line in content.splitlines():
                line = line.strip()
                if line.startswith('CSV:'):
                    parts = line[4:].strip().split(',')
                    if len(parts) >= 6:
                        try:
                            results.append({
                                'type': parts[0].strip(),
                                'size': int(parts[1]),
                                'iterations': int(parts[2]),
                                'stencil': int(parts[3]),
                                'workers': int(parts[4]),
                                'time': float(parts[5])
                            })
                        except (ValueError, IndexError):
                            continue
    return results


def plot_speedup_per_technology(results, output_dir, grid_size=1000):
    """Wykres przyspieszenia dla każdej technologii osobno."""
    # Filtruj wyniki dla danego rozmiaru
    filtered = [r for r in results if r['size'] == grid_size]
    if not filtered:
        print(f"Brak wyników dla rozmiaru {grid_size}")
        return

    # Czas sekwencyjny
    seq_results = [r for r in filtered if r['type'] == 'seq']
    if not seq_results:
        print("Brak wyników sekwencyjnych!")
        return
    seq_time = seq_results[0]['time']

    for tech in ['omp', 'mpi', 'cuda']:
        tech_results = [r for r in filtered if r['type'] == tech]
        if not tech_results:
            continue

        workers = sorted(set(r['workers'] for r in tech_results))
        speedups = []
        for w in workers:
            t = min(r['time'] for r in tech_results if r['workers'] == w)
            speedups.append(seq_time / t)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        tech_names = {'omp': 'OpenMP', 'mpi': 'MPI', 'cuda': 'CUDA'}
        color = {'omp': '#2196F3', 'mpi': '#4CAF50', 'cuda': '#FF5722'}

        # Wykres przyspieszenia
        ax1.plot(workers, speedups, 'o-', color=color[tech], linewidth=2,
                 markersize=8, label=tech_names[tech])
        ax1.plot(workers, workers, '--', color='gray', alpha=0.5, label='Idealne')
        ax1.set_xlabel('Liczba wątków/procesów', fontsize=12)
        ax1.set_ylabel('Przyspieszenie (S)', fontsize=12)
        ax1.set_title(f'Przyspieszenie - {tech_names[tech]} (siatka {grid_size}×{grid_size})',
                      fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(workers)

        # Wykres efektywności
        efficiencies = [s/w for s, w in zip(speedups, workers)]
        ax2.bar(range(len(workers)), efficiencies, color=color[tech], alpha=0.8)
        ax2.set_xticks(range(len(workers)))
        ax2.set_xticklabels(workers)
        ax2.set_xlabel('Liczba wątków/procesów', fontsize=12)
        ax2.set_ylabel('Efektywność (E)', fontsize=12)
        ax2.set_title(f'Efektywność - {tech_names[tech]} (siatka {grid_size}×{grid_size})',
                      fontsize=14, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fname = os.path.join(output_dir, f'speedup_efficiency_{tech}_{grid_size}.png')
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Zapisano: {fname}")
        plt.close()


def plot_comparison(results, output_dir, grid_size=1000):
    """Wykres porównawczy najlepszych wyników z każdej technologii."""
    filtered = [r for r in results if r['size'] == grid_size]
    if not filtered:
        return

    seq_results = [r for r in filtered if r['type'] == 'seq']
    if not seq_results:
        return
    seq_time = seq_results[0]['time']

    technologies = []
    best_times = []
    best_speedups = []
    colors = []
    color_map = {'seq': '#9E9E9E', 'omp': '#2196F3', 'mpi': '#4CAF50', 'cuda': '#FF5722'}
    name_map = {'seq': 'Sekwencyjny', 'omp': 'OpenMP', 'mpi': 'MPI', 'cuda': 'CUDA'}

    for tech in ['seq', 'omp', 'mpi', 'cuda']:
        tech_results = [r for r in filtered if r['type'] == tech]
        if tech_results:
            best_time = min(r['time'] for r in tech_results)
            technologies.append(name_map[tech])
            best_times.append(best_time)
            best_speedups.append(seq_time / best_time)
            colors.append(color_map[tech])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Porównanie czasów
    bars1 = ax1.bar(technologies, best_times, color=colors, alpha=0.85, edgecolor='white')
    ax1.set_ylabel('Czas [s]', fontsize=12)
    ax1.set_title(f'Porównanie czasów - najlepsze wyniki\n(siatka {grid_size}×{grid_size})',
                  fontsize=14, fontweight='bold')
    for bar, t in zip(bars1, best_times):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01*max(best_times),
                 f'{t:.3f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Porównanie przyspieszenia
    bars2 = ax2.bar(technologies, best_speedups, color=colors, alpha=0.85, edgecolor='white')
    ax2.set_ylabel('Przyspieszenie', fontsize=12)
    ax2.set_title(f'Porównanie przyspieszenia\n(siatka {grid_size}×{grid_size})',
                  fontsize=14, fontweight='bold')
    for bar, s in zip(bars2, best_speedups):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01*max(best_speedups),
                 f'{s:.2f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = os.path.join(output_dir, f'comparison_{grid_size}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Zapisano: {fname}")
    plt.close()


def plot_size_scaling(results, output_dir):
    """Wykres przyspieszenia dla różnych rozmiarów siatki."""
    sizes = sorted(set(r['size'] for r in results))
    if len(sizes) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = {'omp': '#2196F3', 'mpi': '#4CAF50', 'cuda': '#FF5722'}
    name_map = {'omp': 'OpenMP', 'mpi': 'MPI', 'cuda': 'CUDA'}

    for tech in ['omp', 'mpi', 'cuda']:
        speedups = []
        valid_sizes = []
        for size in sizes:
            seq = [r for r in results if r['type']=='seq' and r['size']==size]
            tech_r = [r for r in results if r['type']==tech and r['size']==size]
            if seq and tech_r:
                best = min(r['time'] for r in tech_r)
                speedups.append(seq[0]['time'] / best)
                valid_sizes.append(size)

        if valid_sizes:
            ax.plot(valid_sizes, speedups, 'o-', color=color_map[tech],
                    linewidth=2, markersize=8, label=name_map[tech])

    ax.set_xlabel('Rozmiar siatki (N×N)', fontsize=12)
    ax.set_ylabel('Najlepsze przyspieszenie', fontsize=12)
    ax.set_title('Przyspieszenie vs rozmiar siatki', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = os.path.join(output_dir, 'scaling_by_size.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"Zapisano: {fname}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Wykresy wydajności dyfuzji koloru")
    parser.add_argument("results_dir", help="Katalog z wynikami (.csv lub .txt)")
    parser.add_argument("--output", "-o", default="wykresy", help="Katalog na wykresy")
    parser.add_argument("--sizes", nargs='+', type=int, default=None,
                        help="Rozmiary siatki do analizy")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Ładowanie wyników
    results = load_results(args.results_dir)
    results += parse_stdout_results(args.results_dir)

    if not results:
        print("Nie znaleziono wyników!")
        sys.exit(1)

    print(f"Załadowano {len(results)} wyników")

    sizes = args.sizes or sorted(set(r['size'] for r in results))
    for size in sizes:
        plot_speedup_per_technology(results, args.output, size)
        plot_comparison(results, args.output, size)

    plot_size_scaling(results, args.output)
    print(f"\nWykresy zapisane w: {args.output}/")


if __name__ == "__main__":
    main()
