#include "grid.h"
#include <cstring>
#include <cmath>
#include <algorithm>

// ============================================================
// Grid - implementacja
// ============================================================

Grid::Grid() : width(0), height(0) {}

Grid::Grid(int width, int height)
    : width(width), height(height), data(width * height * 3, 0.0f) {}

void Grid::set(int x, int y, const Color& c) {
    if (x < 0 || x >= width || y < 0 || y >= height) return;
    int idx = (y * width + x) * 3;
    data[idx + 0] = c.r;
    data[idx + 1] = c.g;
    data[idx + 2] = c.b;
}

Color Grid::get(int x, int y) const {
    if (x < 0 || x >= width || y < 0 || y >= height) return Color();
    int idx = (y * width + x) * 3;
    return Color(data[idx + 0], data[idx + 1], data[idx + 2]);
}

void Grid::initSources(const std::vector<ColorSource>& sources) {
    for (const auto& src : sources) {
        // Wypełnienie okrągłego obszaru wokół źródła
        for (int dy = -src.radius; dy <= src.radius; dy++) {
            for (int dx = -src.radius; dx <= src.radius; dx++) {
                int nx = src.x + dx;
                int ny = src.y + dy;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    float dist = std::sqrt(float(dx * dx + dy * dy));
                    if (dist <= src.radius) {
                        // Intensywność maleje z odległością od centrum
                        float factor = 1.0f - (dist / (src.radius + 1.0f));
                        Color c = src.color * factor;
                        set(nx, ny, c);
                    }
                }
            }
        }
    }
}

void Grid::applySources(const std::vector<ColorSource>& sources) {
    for (const auto& src : sources) {
        // Przywrócenie wartości w centralnym punkcie źródła
        set(src.x, src.y, src.color);
    }
}

void Grid::copyFrom(const Grid& other) {
    width = other.width;
    height = other.height;
    data = other.data;
}

void Grid::clear() {
    std::fill(data.begin(), data.end(), 0.0f);
}

// ============================================================
// Domyślne źródła koloru
// ============================================================

std::vector<ColorSource> generateDefaultSources(int gridSize) {
    std::vector<ColorSource> sources;
    int radius = std::max(3, gridSize / 50);
    
    // Czerwony - lewy górny
    sources.emplace_back(gridSize / 4, gridSize / 4,
                         Color(1.0f, 0.0f, 0.0f), radius);
    
    // Zielony - prawy górny
    sources.emplace_back(3 * gridSize / 4, gridSize / 4,
                         Color(0.0f, 1.0f, 0.0f), radius);
    
    // Niebieski - dolny środek
    sources.emplace_back(gridSize / 2, 3 * gridSize / 4,
                         Color(0.0f, 0.0f, 1.0f), radius);
    
    // Żółty - centrum
    sources.emplace_back(gridSize / 2, gridSize / 2,
                         Color(1.0f, 1.0f, 0.0f), radius);
    
    // Magenta - lewy dolny
    sources.emplace_back(gridSize / 4, 3 * gridSize / 4,
                         Color(1.0f, 0.0f, 1.0f), radius);
    
    // Cyjan - prawy dolny
    sources.emplace_back(3 * gridSize / 4, 3 * gridSize / 4,
                         Color(0.0f, 1.0f, 1.0f), radius);
    
    return sources;
}

// ============================================================
// Dyfuzja sekwencyjna - szablon 5-punktowy
// ============================================================

void diffuseStep5(const Grid& input, Grid& output) {
    int W = input.width;
    int H = input.height;
    
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
            int count = 0;
            
            // Centrum
            int idx = (y * W + x) * 3;
            sumR += input.data[idx + 0];
            sumG += input.data[idx + 1];
            sumB += input.data[idx + 2];
            count++;
            
            // Góra
            if (y > 0) {
                idx = ((y - 1) * W + x) * 3;
                sumR += input.data[idx + 0];
                sumG += input.data[idx + 1];
                sumB += input.data[idx + 2];
                count++;
            }
            // Dół
            if (y < H - 1) {
                idx = ((y + 1) * W + x) * 3;
                sumR += input.data[idx + 0];
                sumG += input.data[idx + 1];
                sumB += input.data[idx + 2];
                count++;
            }
            // Lewo
            if (x > 0) {
                idx = (y * W + (x - 1)) * 3;
                sumR += input.data[idx + 0];
                sumG += input.data[idx + 1];
                sumB += input.data[idx + 2];
                count++;
            }
            // Prawo
            if (x < W - 1) {
                idx = (y * W + (x + 1)) * 3;
                sumR += input.data[idx + 0];
                sumG += input.data[idx + 1];
                sumB += input.data[idx + 2];
                count++;
            }
            
            int outIdx = (y * W + x) * 3;
            output.data[outIdx + 0] = sumR / count;
            output.data[outIdx + 1] = sumG / count;
            output.data[outIdx + 2] = sumB / count;
        }
    }
}

// ============================================================
// Dyfuzja sekwencyjna - szablon 9-punktowy
// ============================================================

void diffuseStep9(const Grid& input, Grid& output) {
    int W = input.width;
    int H = input.height;
    
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
            int count = 0;
            
            // Przejście przez sąsiedztwo 3x3
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx >= 0 && nx < W && ny >= 0 && ny < H) {
                        int idx = (ny * W + nx) * 3;
                        sumR += input.data[idx + 0];
                        sumG += input.data[idx + 1];
                        sumB += input.data[idx + 2];
                        count++;
                    }
                }
            }
            
            int outIdx = (y * W + x) * 3;
            output.data[outIdx + 0] = sumR / count;
            output.data[outIdx + 1] = sumG / count;
            output.data[outIdx + 2] = sumB / count;
        }
    }
}
