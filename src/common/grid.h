#ifndef GRID_H
#define GRID_H

#include <vector>
#include <cstdint>
#include <string>

/**
 * Struktura reprezentująca kolor RGB jako trzy wartości zmiennoprzecinkowe [0.0, 1.0].
 */
struct Color {
    float r, g, b;
    
    Color() : r(0.0f), g(0.0f), b(0.0f) {}
    Color(float r, float g, float b) : r(r), g(g), b(b) {}
    
    Color operator+(const Color& other) const {
        return Color(r + other.r, g + other.g, b + other.b);
    }
    
    Color operator*(float scalar) const {
        return Color(r * scalar, g * scalar, b * scalar);
    }
    
    Color& operator+=(const Color& other) {
        r += other.r; g += other.g; b += other.b;
        return *this;
    }
};

/**
 * Struktura opisująca źródło koloru na siatce.
 */
struct ColorSource {
    int x, y;       // Pozycja na siatce
    Color color;     // Kolor źródła
    int radius;      // Promień początkowy źródła
    
    ColorSource() : x(0), y(0), color(), radius(1) {}
    ColorSource(int x, int y, Color c, int radius = 3)
        : x(x), y(y), color(c), radius(radius) {}
};

/**
 * Klasa reprezentująca dwuwymiarową siatkę kolorów.
 */
class Grid {
public:
    int width, height;
    std::vector<float> data; // Dane RGB: [height * width * 3]
    
    Grid();
    Grid(int width, int height);
    
    // Dostęp do piksela
    void set(int x, int y, const Color& c);
    Color get(int x, int y) const;
    
    // Dostęp do surowych danych
    float* raw() { return data.data(); }
    const float* raw() const { return data.data(); }
    size_t dataSize() const { return data.size() * sizeof(float); }
    
    // Inicjalizacja źródeł koloru
    void initSources(const std::vector<ColorSource>& sources);
    
    // Aplikacja źródeł (przywrócenie wartości w punktach źródłowych)
    void applySources(const std::vector<ColorSource>& sources);
    
    // Kopiowanie danych
    void copyFrom(const Grid& other);
    
    // Czyszczenie siatki
    void clear();
};

/**
 * Generuje domyślne źródła koloru rozmieszczone na siatce.
 */
std::vector<ColorSource> generateDefaultSources(int gridSize);

/**
 * Wykonuje jedną iterację dyfuzji sekwencyjnie (szablon 5-punktowy).
 */
void diffuseStep5(const Grid& input, Grid& output);

/**
 * Wykonuje jedną iterację dyfuzji sekwencyjnie (szablon 9-punktowy).
 */
void diffuseStep9(const Grid& input, Grid& output);

#endif // GRID_H
