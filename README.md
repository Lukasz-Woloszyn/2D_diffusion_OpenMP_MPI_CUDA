# Rozprzestrzenianie Koloru - Symulacja Dyfuzji na Siatce 2D

## Opis Projektu

Projekt przedstawia symulację rozlewania się koloru po dwuwymiarowej siatce. 
Na początku w kilku punktach siatki znajdują się źródła koloru (czerwony, zielony, niebieski), które z każdą iteracją rozlewają się na sąsiednie pola. Wartość koloru w każdej komórce jest obliczana jako średnia z jej sąsiadów (szablon 5-punktowy lub 9-punktowy). 

**Interfejs graficzny (GUI)** umożliwia podgląd animacji na żywo oraz generowanie wykresów w oparciu o zautomatyzowane benchmarki.

## Technologie

- **Sekwencyjna** - implementacja bazowa (CPU, jednowątkowa)
- **OpenMP** - równoległa aktualizacja komórek siatki na CPU
- **MPI** - podział siatki na bloki i wymiana danych brzegowych między procesami
- **CUDA** - obliczenia na GPU z wykorzystaniem pamięci współdzielonej

## Wymagania

### System i Kompilacja (Backend C++/CUDA)
- System operacyjny: Linux (bezpośrednio lub poprzez WSL2 na Windows)
- Aby zainstalować wszystkie wymagane narzędzia w systemie Ubuntu / WSL2, wykonaj w terminalu polecenie:
  ```bash
  sudo apt update
  sudo apt install build-essential cmake openmpi-bin libopenmpi-dev nvidia-cuda-toolkit python3-pip python3-tk
  ```

### Interfejs Graficzny (Python)
- Python 3.8+
- Pakiety systemowe w Ubuntu/WSL: `sudo apt install python3-tk` (wymagane do poprawnego wyświetlania okna interfejsu graficznego)
- Pakiety Pythona: `Pillow`, `matplotlib`, `numpy` (instalacja komendą: `pip install -r requirements.txt`)
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
Uruchom plik `gui.py`:
```bash
# Otwórz terminal (np. PowerShell) w głównym folderze projektu i wpisz:
wsl -d Ubuntu-24.04 -e bash -c "python3 gui.py"

# Ewentualnie, będąc już natywnie wewnątrz środowiska Linux / WSL2:
python3 gui.py
```

Po uruchomieniu aplikacji, po lewej stronie można wybrać technologię (np. CUDA) i kliknąć **"▶ Uruchom"**, aby rozpocząć symulację i na żywo na ekranie obserwować rozchodzenie się kolorów.

Aby sprawdzić pełną wydajność i zestawienie wszystkich metod, przejdź do zakładki **"Benchmark / Wykresy"** i naciśnij przycisk **"Benchmark"**. Wywoła to skrypt testowy w tle, który zajmie chwilę. Po wszystkim wygenerowane zostaną wykresy.

## Struktura Projektu

```text
prir_projekt/
├── CMakeLists.txt
├── README.md               # Ten plik
├── requirements.txt        # Zależności Python
├── Sprawozdanie.pdf        # Dokumentacja / Sprawozdanie z działania
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
