#ifndef IO_H
#define IO_H

#include "grid.h"
#include <string>

/**
 * Tworzy katalog wyjściowy jeśli nie istnieje.
 */
void createOutputDir(const std::string& path);

/**
 * Zapisuje stan siatki do pliku binarnego.
 * Format: [width:int32][height:int32][data:float32 * width * height * 3]
 */
void saveGridBinary(const Grid& grid, const std::string& filename);

/**
 * Zapisuje stan siatki do pliku PPM (prosty format obrazu).
 */
void saveGridPPM(const Grid& grid, const std::string& filename);

/**
 * Generuje nazwę pliku klatki.
 */
std::string frameFilename(const std::string& dir, int iteration, const std::string& ext = ".bin");

#endif // IO_H
