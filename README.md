# Rozprzestrzenianie Koloru - Symulacja Dyfuzji na Siatce 2D

## Opis Projektu

Projekt przedstawia symulację rozlewania się koloru po dwuwymiarowej siatce. 
Na początku w kilku punktach siatki znajdują się źródła koloru (czerwony, zielony, niebieski), które z każdą iteracją rozlewają się na sąsiednie pola. Wartość koloru w każdej komórce jest obliczana jako średnia z jej sąsiadów (szablon 5-punktowy lub 9-punktowy). 

Głównym centrum dowodzenia projektu jest **interfejs graficzny (GUI)**, który umożliwia podgląd animacji na żywo oraz generowanie wykresów w oparciu o zautomatyzowane benchmarki!

## Technologie

- **Sekwencyjna** - implementacja bazowa (CPU, jednowątkowa)
- **OpenMP** - równoległa aktualizacja komórek siatki na CPU
- **MPI** - podział siatki na bloki i wymiana danych brzegowych między procesami
- **CUDA** - obliczenia na GPU z wykorzystaniem pamięci współdzielonej

## Wymagania

### System i Kompilacja (Backend C++/CUDA)
- System operacyjny: Linux (bezpośrednio lub poprzez WSL2 na Windows)
- GCC/G++ z obsługą OpenMP (>= 9.0)
- MPICH lub OpenMPI
- NVIDIA CUDA Toolkit (>= 11.0)
- CMake (>= 3.18)

### Interfejs Graficzny (Python)
- Python 3.8+
- Pakiety: `Pillow`, `matplotlib`, `numpy` (instalacja komendą: `pip install -r requirements.txt`)
- W systemie Windows z WSL, aplikacja `gui.py` domyślnie komunikuje się z instancją Linuksa w celu zlecania obliczeń, można ją więc uruchomić normalnie w konsoli Windows lub wewnątrz WSL.

## Instrukcja Uruchomienia Krok po Kroku (Od zera)

**1. Przejdź do folderu z projektem:**
```bash
cd prir_projekt
```

**2. Zainstaluj wymagane pakiety Pythona dla GUI:**
```bash
pip install -r requirements.txt
```

**3. Kompilacja programów (w środowisku Linux / WSL2):**
Zbudowanie silników obliczeniowych w C++ i CUDA:
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..
```

**4. Uruchomienie aplikacji:**
Całość obsługiwana jest z poziomu interfejsu graficznego.
Uruchom po prostu plik `gui.py`:
```bash
# Otwórz terminal (np. PowerShell) w głównym folderze projektu i wpisz:
wsl -d Ubuntu-24.04 -e bash -c "python3 gui.py"

# Ewentualnie, będąc już natywnie wewnątrz środowiska Linux / WSL2:
python3 gui.py
```

Po uruchomieniu aplikacji, po lewej stronie możesz wybrać interesującą Cię technologię (np. CUDA) i kliknąć **"▶ Uruchom"**, aby rozpocząć symulację i na żywo na ekranie obserwować rozchodzenie się kolorów.

Aby sprawdzić pełną wydajność i zestawienie wszystkich metod, przejdź do zakładki **"Benchmark / Wykresy"** i naciśnij przycisk **"Benchmark"**. Pamiętaj: wywoła to obszerny skrypt testowy w tle i zajmie chwilę. Po wszystkim ujrzysz piękne, dogłębne wykresy!

## Struktura Projektu

Projekt został zoptymalizowany pod obsługę graficzną. Z nieużywanych plików zachowano jedynie to, co niezbędne:

```text
prir_projekt/
├── CMakeLists.txt
├── README.md               # Ten plik
├── requirements.txt        # Zależności Python
├── sprawozdanie.md         # Dokumentacja / Sprawozdanie z działania
├── gui.py                  # Aplikacja GUI (Główny punkt wejścia)
├── src/                    # Kody źródłowe C++/CUDA
│   ├── common/             # Wspólne narzędzia, definicje siatki, I/O
│   ├── sequential/         # Wersja CPU (sekwencyjna)
│   ├── openmp/             # Wersja CPU (wielowątkowa OpenMP)
│   ├── mpi/                # Wersja MPI (wieloprocesowa)
│   └── cuda/               # Wersja GPU (CUDA)
├── scripts/
│   └── run_benchmarks.sh   # Bash skrypt pełnego zestawu benchmarków
├── visualization/
│   └── benchmark_plots.py  # Moduł Pythona do rysowania wykresów analitycznych
├── output_gui/             # (Generowane automatycznie) Klatki symulacji dla GUI
└── results/                # (Generowane automatycznie) Dane tekstowe i folder wykresy/
```
