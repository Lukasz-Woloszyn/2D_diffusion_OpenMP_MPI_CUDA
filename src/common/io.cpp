#include "io.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir(path, 0755)
#endif

void createOutputDir(const std::string &path) { MKDIR(path.c_str()); }

void saveGridBinary(const Grid &grid, const std::string &filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Błąd: nie można otworzyć pliku " << filename << "\n";
    return;
  }

  int32_t w = grid.width;
  int32_t h = grid.height;
  file.write(reinterpret_cast<const char *>(&w), sizeof(int32_t));
  file.write(reinterpret_cast<const char *>(&h), sizeof(int32_t));
  file.write(reinterpret_cast<const char *>(grid.data.data()),
             grid.data.size() * sizeof(float));
  file.close();
}

void saveGridPPM(const Grid &grid, const std::string &filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Błąd: nie można otworzyć pliku " << filename << "\n";
    return;
  }

  file << "P6\n" << grid.width << " " << grid.height << "\n255\n";

  for (int y = 0; y < grid.height; y++) {
    for (int x = 0; x < grid.width; x++) {
      int idx = (y * grid.width + x) * 3;
      uint8_t r = static_cast<uint8_t>(
          std::clamp(grid.data[idx + 0], 0.0f, 1.0f) * 255.0f);
      uint8_t g = static_cast<uint8_t>(
          std::clamp(grid.data[idx + 1], 0.0f, 1.0f) * 255.0f);
      uint8_t b = static_cast<uint8_t>(
          std::clamp(grid.data[idx + 2], 0.0f, 1.0f) * 255.0f);
      file.put(r);
      file.put(g);
      file.put(b);
    }
  }
  file.close();
}

std::string frameFilename(const std::string &dir, int iteration,
                          const std::string &ext) {
  char buf[256];
  snprintf(buf, sizeof(buf), "%s/frame_%05d%s", dir.c_str(), iteration,
           ext.c_str());
  return std::string(buf);
}
