/**
 * Rozprzestrzenianie koloru - Implementacja SEKWENCYJNA (bazowa)
 * 
 * Prosta iteracyjna dyfuzja na siatce 2D.
 * Każda komórka jest aktualizowana jako średnia z jej sąsiadów.
 * Służy jako punkt odniesienia do pomiarów przyspieszenia.
 */

#include "grid.h"
#include "args.h"
#include "io.h"

#include <iostream>
#include <chrono>
#include <string>

int main(int argc, char* argv[]) {
    Args args = Args::parse(argc, argv);
    if (args.help) {
        Args::printHelp(argv[0]);
        return 0;
    }
    
    std::cout << "=== Dyfuzja Koloru - Wersja Sekwencyjna ===\n";
    args.print();
    
    // Inicjalizacja siatek (podwójne buforowanie)
    Grid gridA(args.size, args.size);
    Grid gridB(args.size, args.size);
    
    // Generowanie źródeł koloru
    auto sources = generateDefaultSources(args.size);
    gridA.initSources(sources);
    
    std::cout << "Źródła koloru: " << sources.size() << "\n";
    
    // Tworzenie katalogu wyjściowego
    if (!args.benchmark) {
        createOutputDir(args.output);
        saveGridPPM(gridA, frameFilename(args.output, 0, ".ppm"));
        saveGridBinary(gridA, frameFilename(args.output, 0, ".bin"));
    }
    
    // Główna pętla symulacji
    auto startTime = std::chrono::high_resolution_clock::now();
    
    Grid* current = &gridA;
    Grid* next = &gridB;
    
    for (int iter = 1; iter <= args.iterations; iter++) {
        // Krok dyfuzji
        if (args.stencil == 9) {
            diffuseStep9(*current, *next);
        } else {
            diffuseStep5(*current, *next);
        }
        
        // Przywrócenie źródeł
        next->applySources(sources);
        
        // Zamiana buforów
        std::swap(current, next);
        
        // Zapis klatki
        if (!args.benchmark && (iter % args.saveInterval == 0 || iter == args.iterations)) {
            saveGridPPM(*current, frameFilename(args.output, iter, ".ppm"));
            saveGridBinary(*current, frameFilename(args.output, iter, ".bin"));
            std::cout << "  Zapisano klatkę: iteracja " << iter << "\n";
        }
        
        // Postęp
        if (iter % 100 == 0) {
            std::cout << "  Iteracja " << iter << "/" << args.iterations << "\n";
        }
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();
    
    std::cout << "\n=== Wyniki ===\n";
    std::cout << "  Czas obliczenia: " << elapsed << " s\n";
    std::cout << "  Iteracje:        " << args.iterations << "\n";
    std::cout << "  Rozmiar siatki:  " << args.size << " x " << args.size << "\n";
    std::cout << "  Czas/iterację:   " << (elapsed / args.iterations * 1000.0) << " ms\n";
    
    // Zapis ostatecznego wyniku
    if (!args.benchmark) {
        saveGridPPM(*current, args.output + "/frame_final.ppm");
        saveGridBinary(*current, args.output + "/frame_final.bin");
        std::cout << "  Wynik zapisany w: " << args.output << "/\n";
    }
    
    // Wypisanie wyniku w formacie CSV (dla benchmarków)
    std::cout << "\nCSV: seq," << args.size << "," << args.iterations << ","
              << args.stencil << ",1," << elapsed << "\n";
    
    return 0;
}
