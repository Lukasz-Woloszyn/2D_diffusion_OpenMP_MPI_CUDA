#ifndef ARGS_H
#define ARGS_H

#include <string>
#include <cstdlib>
#include <cstring>
#include <iostream>

/**
 * Struktura przechowująca argumenty wiersza poleceń.
 */
struct Args {
    int size = 1000;           // Rozmiar siatki NxN
    int iterations = 500;     // Liczba iteracji symulacji
    int stencil = 5;          // Szablon sąsiedztwa (5 lub 9)
    int threads = 4;          // Liczba wątków (OpenMP)
    int blockSize = 16;       // Rozmiar bloku CUDA
    std::string output = "output"; // Katalog wyjściowy
    int saveInterval = 50;    // Co ile iteracji zapisywać klatkę
    bool benchmark = false;   // Tryb benchmarku
    bool help = false;
    
    static Args parse(int argc, char* argv[]) {
        Args args;
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
                args.size = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
                args.iterations = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--stencil") == 0 && i + 1 < argc) {
                args.stencil = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
                args.threads = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--block-size") == 0 && i + 1 < argc) {
                args.blockSize = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
                args.output = argv[++i];
            } else if (strcmp(argv[i], "--save-interval") == 0 && i + 1 < argc) {
                args.saveInterval = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--benchmark") == 0) {
                args.benchmark = true;
            } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                args.help = true;
            }
        }
        return args;
    }
    
    static void printHelp(const char* programName) {
        std::cout << "Użycie: " << programName << " [opcje]\n"
                  << "\nOpcje:\n"
                  << "  --size N          Rozmiar siatki NxN (domyślnie: 1000)\n"
                  << "  --iterations N    Liczba iteracji (domyślnie: 500)\n"
                  << "  --stencil N       Szablon sąsiedztwa: 5 lub 9 (domyślnie: 5)\n"
                  << "  --threads N       Liczba wątków OpenMP (domyślnie: 4)\n"
                  << "  --block-size N    Rozmiar bloku CUDA (domyślnie: 16)\n"
                  << "  --output DIR      Katalog wyjściowy (domyślnie: output)\n"
                  << "  --save-interval N Co ile iteracji zapisać klatkę (domyślnie: 50)\n"
                  << "  --benchmark       Tryb benchmarku (bez zapisu klatek)\n"
                  << "  --help, -h        Wyświetl pomoc\n";
    }
    
    void print() const {
        std::cout << "=== Konfiguracja ===\n"
                  << "  Rozmiar siatki: " << size << " x " << size << "\n"
                  << "  Iteracje:       " << iterations << "\n"
                  << "  Szablon:        " << stencil << "-punktowy\n"
                  << "  Katalog wyj.:   " << output << "\n"
                  << "  Zapis co:       " << saveInterval << " iteracji\n"
                  << "  Benchmark:      " << (benchmark ? "TAK" : "NIE") << "\n"
                  << "===================\n";
    }
};

#endif // ARGS_H
