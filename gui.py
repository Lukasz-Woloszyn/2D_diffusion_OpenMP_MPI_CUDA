#!/usr/bin/env python3
"""
GUI do symulacji rozprzestrzeniania koloru (dyfuzja 2D).
Pozwala na uruchamianie symulacji z roznymi technologiami (Sekwencyjna, OpenMP, MPI, CUDA),
wizualizacje wynikow i porownanie wydajnosci.

Uruchomienie z WSL:
    python3 gui.py

Uruchomienie z PowerShell:
    wsl -d Ubuntu-24.04 -e bash -c "cd /mnt/c/Users/Lukas/vscode/prir_projekt && python3 gui.py"
"""

import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import os
import sys
import platform
import time
import re
import glob
import shutil
from pathlib import Path

try:
    from PIL import Image, ImageTk, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import matplotlib
    import matplotlib.pyplot
    matplotlib.use('TkAgg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ============================================================
# Konfiguracja sciezek
# ============================================================
IS_WINDOWS = platform.system() == 'Windows'
PROJECT_DIR = Path(__file__).parent.resolve()

if IS_WINDOWS:
    drive = str(PROJECT_DIR)[0].lower()
    WSL_PROJECT = '/mnt/' + drive + str(PROJECT_DIR)[2:].replace('\\', '/')
else:
    WSL_PROJECT = str(PROJECT_DIR)

BIN_DIR = PROJECT_DIR / 'build' / 'bin'
OUTPUT_DIR = PROJECT_DIR / 'output_gui'
CHARTS_DIR = PROJECT_DIR / 'results' / 'wykresy'

# ============================================================
# Czcionki (cross-platform)
# ============================================================
_SANS = 'Helvetica'
_MONO = 'Courier'

# ============================================================
# Kolory motywu
# ============================================================
class C:
    BG       = '#0f0f1a'
    SURFACE  = '#1a1a2e'
    CARD     = '#22223b'
    ACCENT   = '#6c63ff'
    ACCENT_H = '#8b83ff'
    GREEN    = '#00c896'
    YELLOW   = '#ffc857'
    RED      = '#e63946'
    ORANGE   = '#ff6b35'
    TEXT     = '#eaeaea'
    DIM      = '#8892a0'
    BORDER   = '#2a2a4a'
    SEQ = '#9ca3af'; OMP = '#3b82f6'; MPI = '#22c55e'; CUDA = '#f97316'

TECH_NAMES  = {'seq': 'Sekwencyjna', 'omp': 'OpenMP', 'mpi': 'MPI', 'cuda': 'CUDA'}
TECH_COLORS = {'seq': C.SEQ, 'omp': C.OMP, 'mpi': C.MPI, 'cuda': C.CUDA}

# ============================================================
# Budowanie komendy
# ============================================================
def build_cmd(tech, size, iterations, stencil, threads, block_size,
              output_dir, save_interval, benchmark=False):
    binary = {
        'seq': 'diffusion_seq', 'omp': 'diffusion_omp',
        'mpi': 'diffusion_mpi', 'cuda': 'diffusion_cuda'
    }[tech]

    args = f'--size {size} --iterations {iterations} --stencil {stencil}'
    args += f' --output {output_dir} --save-interval {save_interval}'
    if benchmark:
        args += ' --benchmark'

    if tech == 'omp':
        args += f' --threads {threads}'
        run = f'OMP_NUM_THREADS={threads} ./build/bin/{binary} {args}'
    elif tech == 'mpi':
        run = (f'mpirun --allow-run-as-root -np {threads} '
               f'--oversubscribe ./build/bin/{binary} {args}')
    elif tech == 'cuda':
        args += f' --block-size {block_size}'
        run = f'./build/bin/{binary} {args}'
    else:
        run = f'./build/bin/{binary} {args}'

    if IS_WINDOWS:
        return f'wsl -d Ubuntu-24.04 -e bash -c "cd {WSL_PROJECT} && {run} 2>&1"'
    else:
        return f'bash -c "cd {WSL_PROJECT} && {run} 2>&1"'

# ============================================================
# Ladowanie obrazu PPM
# ============================================================
def load_ppm(path, max_size=480):
    if not os.path.exists(path):
        return None
    try:
        if HAS_PIL:
            img = Image.open(path)
            ratio = min(max_size / img.width, max_size / img.height)
            new_w, new_h = int(img.width * ratio), int(img.height * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        else:
            return tk.PhotoImage(file=path)
    except Exception as e:
        print(f"Blad ladowania {path}: {e}")
        return None


def load_png(path, max_w=800, max_h=500):
    """Laduje PNG i skaluje. Zwraca ImageTk.PhotoImage."""
    if not HAS_PIL or not os.path.exists(path):
        return None
    try:
        img = Image.open(path)
        ratio = min(max_w / img.width, max_h / img.height, 1.0)
        if ratio < 1.0:
            new_w, new_h = int(img.width * ratio), int(img.height * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None

def sort_frames(files):
    """Sortuje pliki ramek numerycznie zamiast leksykograficznie."""
    def extract_num(f):
        m = re.search(r'frame_(\d+)', f)
        return int(m.group(1)) if m else float('inf')
    return sorted(files, key=extract_num)

# ============================================================
# Glowna aplikacja
# ============================================================
class DiffusionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rozprzestrzenianie Koloru  -  Symulacja Dyfuzji 2D")
        self.geometry("1280x820")
        self.configure(bg=C.BG)
        self.minsize(1100, 700)

        # Stan
        self.process = None
        self.running = False
        self.bench_running = False
        self.results = {}
        self.current_photo = None
        self.anim_frames = []
        self.anim_idx = 0
        self.anim_playing = False
        self.chart_files = []
        self.chart_idx = 0
        self.chart_photo = None

        # Zmienne
        self.tech_var    = tk.StringVar(value='omp')
        self.size_var    = tk.IntVar(value=500)
        self.iter_var    = tk.IntVar(value=1000)
        self.stencil_var = tk.IntVar(value=5)
        self.threads_var = tk.IntVar(value=4)
        self.block_var   = tk.IntVar(value=16)
        self.status_var  = tk.StringVar(value='Gotowy do uruchomienia')

        OUTPUT_DIR.mkdir(exist_ok=True)
        CHARTS_DIR.mkdir(exist_ok=True)
        self._build_ui()

    # --------------------------------------------------------
    # Budowanie interfejsu
    # --------------------------------------------------------
    def _build_ui(self):
        # --- Naglowek ---
        header = tk.Frame(self, bg=C.ACCENT, height=48)
        header.pack(fill='x')
        header.pack_propagate(False)
        tk.Label(header, text="  Rozprzestrzenianie Koloru  --  Symulacja Dyfuzji 2D",
                 bg=C.ACCENT, fg='white',
                 font=(_SANS, 14, 'bold')).pack(side='left', padx=12)
        tk.Label(header, text="OpenMP  |  MPI  |  CUDA",
                 bg=C.ACCENT, fg='#c7d2fe',
                 font=(_SANS, 10)).pack(side='right', padx=16)

        body = tk.Frame(self, bg=C.BG)
        body.pack(fill='both', expand=True, padx=8, pady=6)
        self._build_sidebar(body)
        self._build_main(body)

    def _build_sidebar(self, parent):
        sb = tk.Frame(parent, bg=C.SURFACE, width=270, relief='flat',
                      highlightbackground=C.BORDER, highlightthickness=1)
        sb.pack(side='left', fill='y', padx=(0, 6))
        sb.pack_propagate(False)

        def section(text):
            f = tk.Frame(sb, bg=C.SURFACE)
            f.pack(fill='x', padx=10, pady=(8, 2))
            tk.Label(f, text=text, bg=C.SURFACE, fg=C.ACCENT,
                     font=(_SANS, 9, 'bold')).pack(anchor='w')
            tk.Frame(f, bg=C.BORDER, height=1).pack(fill='x', pady=(2, 0))

        def param_row(label, variable, width=8):
            f = tk.Frame(sb, bg=C.SURFACE)
            f.pack(fill='x', padx=10, pady=2)
            tk.Label(f, text=label, bg=C.SURFACE, fg=C.TEXT,
                     font=(_SANS, 9), width=14, anchor='w').pack(side='left')
            tk.Entry(f, textvariable=variable, bg=C.CARD, fg=C.TEXT,
                     insertbackground=C.TEXT, font=(_MONO, 9),
                     width=width, relief='flat',
                     highlightbackground=C.BORDER,
                     highlightthickness=1).pack(side='right')

        # --- Technologia ---
        section("TECHNOLOGIA")
        tf = tk.Frame(sb, bg=C.SURFACE)
        tf.pack(fill='x', padx=10, pady=2)
        for tech, name in TECH_NAMES.items():
            color = TECH_COLORS[tech]
            tk.Radiobutton(tf, text=f'  {name}', variable=self.tech_var,
                           value=tech, bg=C.SURFACE, fg=color,
                           selectcolor=C.CARD, activebackground=C.SURFACE,
                           activeforeground=color, font=(_SANS, 9, 'bold'),
                           indicatoron=True, anchor='w').pack(fill='x')

        # --- Parametry ---
        section("PARAMETRY")
        param_row("Rozmiar siatki:", self.size_var)
        param_row("Iteracje:", self.iter_var)

        sf = tk.Frame(sb, bg=C.SURFACE)
        sf.pack(fill='x', padx=10, pady=2)
        tk.Label(sf, text="Szablon:", bg=C.SURFACE, fg=C.TEXT,
                 font=(_SANS, 9), width=14, anchor='w').pack(side='left')
        for val in [5, 9]:
            tk.Radiobutton(sf, text=f'{val}-pkt', variable=self.stencil_var,
                           value=val, bg=C.SURFACE, fg=C.TEXT,
                           selectcolor=C.CARD, activebackground=C.SURFACE,
                           activeforeground=C.ACCENT,
                           font=(_SANS, 8)).pack(side='left', padx=3)

        param_row("Watki/Procesy:", self.threads_var)
        param_row("CUDA block:", self.block_var)

        # --- Przyciski ---
        section("AKCJE")
        bf = tk.Frame(sb, bg=C.SURFACE)
        bf.pack(fill='x', padx=10, pady=2)
        bf.columnconfigure(0, weight=1)
        bf.columnconfigure(1, weight=1)

        def btn(text, cmd, color, row, col=0, cs=1):
            b = tk.Button(bf, text=text, command=cmd, bg=color, fg='white',
                          font=(_SANS, 9, 'bold'), relief='flat',
                          cursor='hand2', activebackground=color, pady=4)
            b.grid(row=row, column=col, columnspan=cs, sticky='ew', padx=2, pady=2)
            return b

        btn("▶ Uruchom",    self.start_simulation, C.GREEN,  0, 0, 2)
        btn("■ Zatrzymaj",  self.stop_simulation,  C.RED,    1, 0, 2)
        btn("Benchmark",    self.start_benchmark,  C.ACCENT, 2, 0, 2)
        btn("Animacja",     self.play_animation,   C.ORANGE, 3, 0, 1)
        btn("Wyczysc",      self.clear_output,     C.CARD,   3, 1, 1)

        # --- Status ---
        section("STATUS")
        tk.Label(sb, textvariable=self.status_var, bg=C.SURFACE, fg=C.YELLOW,
                 font=(_SANS, 8), wraplength=245, justify='left').pack(
                     fill='x', padx=10, pady=2)

        # --- Wyniki ---
        section("WYNIKI")
        self.results_frame = tk.Frame(sb, bg=C.SURFACE)
        self.results_frame.pack(fill='x', padx=10, pady=2)
        self._update_results_panel()

    def _build_main(self, parent):
        main = tk.Frame(parent, bg=C.BG)
        main.pack(side='right', fill='both', expand=True)

        style = ttk.Style()
        style.theme_use('default')
        style.configure('Dark.TNotebook', background=C.BG, borderwidth=0)
        style.configure('Dark.TNotebook.Tab', background=C.CARD,
                        foreground=C.TEXT, padding=[14, 6],
                        font=(_SANS, 9, 'bold'))
        style.map('Dark.TNotebook.Tab',
                  background=[('selected', C.ACCENT)],
                  foreground=[('selected', 'white')])

        nb = ttk.Notebook(main, style='Dark.TNotebook')
        nb.pack(fill='both', expand=True)

        self._build_sim_tab(nb)
        self._build_bench_tab(nb)
        self._build_log_tab(nb)

    def _build_sim_tab(self, nb):
        tab = tk.Frame(nb, bg=C.BG)
        nb.add(tab, text='  Symulacja  ')

        canvas_frame = tk.Frame(tab, bg=C.CARD,
                                highlightbackground=C.BORDER,
                                highlightthickness=1)
        canvas_frame.pack(fill='both', expand=True, padx=6, pady=6)
        self.canvas = tk.Canvas(canvas_frame, bg='#000000', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=2, pady=2)

        info = tk.Frame(tab, bg=C.SURFACE, height=44)
        info.pack(fill='x', padx=6, pady=(0, 6))
        info.pack_propagate(False)

        self.info_tech = tk.Label(info, text="Technologia: -",
                                  bg=C.SURFACE, fg=C.TEXT,
                                  font=(_SANS, 10, 'bold'))
        self.info_tech.pack(side='left', padx=12)
        self.info_time = tk.Label(info, text="Czas: -",
                                  bg=C.SURFACE, fg=C.GREEN,
                                  font=(_MONO, 10, 'bold'))
        self.info_time.pack(side='left', padx=12)
        self.info_speedup = tk.Label(info, text="",
                                     bg=C.SURFACE, fg=C.YELLOW,
                                     font=(_MONO, 10, 'bold'))
        self.info_speedup.pack(side='left', padx=12)
        self.info_size = tk.Label(info, text="", bg=C.SURFACE, fg=C.DIM,
                                  font=(_SANS, 8))
        self.info_size.pack(side='right', padx=12)

    def _build_bench_tab(self, nb):
        tab = tk.Frame(nb, bg=C.BG)
        nb.add(tab, text='  Benchmark / Wykresy  ')

        # Obszar wykresu
        self.chart_container = tk.Frame(tab, bg=C.BG)
        self.chart_container.pack(fill='both', expand=True, padx=6, pady=(6, 0))

        self.chart_image_label = tk.Label(self.chart_container, bg=C.BG,
                                          text="Kliknij [Benchmark] aby wygenerować wykresy\nlub wybierz z wyników poniżej",
                                          fg=C.DIM, font=(_SANS, 14))
        self.chart_image_label.pack(fill='both', expand=True)

        # Galeria zapisanych wykresow
        self.gallery_frame = tk.Frame(tab, bg=C.SURFACE, height=42)
        self.gallery_frame.pack(fill='x', padx=6, pady=6)
        self.gallery_frame.pack_propagate(False)

        tk.Button(self.gallery_frame, text="<  Poprzedni",
                  command=self._chart_prev, bg=C.CARD, fg=C.TEXT,
                  font=(_SANS, 9), relief='flat',
                  cursor='hand2').pack(side='left', padx=8)

        self.chart_label = tk.Label(self.gallery_frame, text="Brak wykresow",
                                    bg=C.SURFACE, fg=C.DIM,
                                    font=(_SANS, 9))
        self.chart_label.pack(side='left', expand=True)

        tk.Button(self.gallery_frame, text="Nastepny  >",
                  command=self._chart_next, bg=C.CARD, fg=C.TEXT,
                  font=(_SANS, 9), relief='flat',
                  cursor='hand2').pack(side='right', padx=8)

    def _build_log_tab(self, nb):
        tab = tk.Frame(nb, bg=C.BG)
        nb.add(tab, text='  Logi  ')

        self.log_text = tk.Text(tab, bg='#0a0a14', fg='#b0b8c8',
                                font=(_MONO, 9), relief='flat', wrap='word',
                                insertbackground=C.TEXT)
        scroll = tk.Scrollbar(tab, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y', padx=(0, 6), pady=6)
        self.log_text.pack(fill='both', expand=True, padx=6, pady=6)

    # --------------------------------------------------------
    # Pomocnicze
    # --------------------------------------------------------
    def log(self, msg):
        self.log_text.insert('end', msg + '\n')
        self.log_text.see('end')

    def set_status(self, msg):
        self.status_var.set(msg)
        self.update_idletasks()

    # --------------------------------------------------------
    # Symulacja
    # --------------------------------------------------------
    def start_simulation(self):
        if self.running or self.bench_running:
            return
        tech = self.tech_var.get()
        size = self.size_var.get()
        iters = self.iter_var.get()
        stencil = self.stencil_var.get()
        threads = self.threads_var.get()
        block = self.block_var.get()

        wsl_out = (WSL_PROJECT + '/output_gui') if IS_WINDOWS else str(OUTPUT_DIR)
        save_interval = max(1, iters // 20)

        cmd = build_cmd(tech, size, iters, stencil, threads, block,
                        wsl_out, save_interval, benchmark=False)

        self.running = True
        self.set_status(f'[▶] Uruchamiam {TECH_NAMES[tech]}...')
        self.info_tech.config(text=f"Technologia: {TECH_NAMES[tech]}",
                              fg=TECH_COLORS[tech])
        self.info_time.config(text="Czas: obliczam...")
        self.info_speedup.config(text="")
        self.info_size.config(
            text=f"{size}x{size} | {iters} iter | stencil {stencil}")
        self.log(f"\n{'='*60}")
        self.log(f"[▶] {TECH_NAMES[tech]} | {size}x{size} | {iters} iteracji")
        self.log(f"CMD: {cmd}")
        self.log('='*60)

        thread = threading.Thread(target=self._run_process,
                                  args=(tech, cmd), daemon=True)
        thread.start()
        self.after(1000, self._poll_frames)

    def stop_simulation(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.kill()
            except Exception:
                pass
        self.running = False
        self.set_status('[■] Zatrzymano')
        self.log("[■] Symulacja zatrzymana")

    def _run_process(self, tech, cmd):
        try:
            self.process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)

            output_lines = []
            for line in self.process.stdout:
                line = line.rstrip()
                output_lines.append(line)
                self.after(0, self.log, line)
                m = re.search(r'Iteracja\s+(\d+)/(\d+)', line)
                if m:
                    cur, total = int(m.group(1)), int(m.group(2))
                    pct = cur * 100 // total
                    self.after(0, self.set_status,
                               f'[▶] {TECH_NAMES[tech]}: {pct}% ({cur}/{total})')

            self.process.wait()

            elapsed = None
            workers = 1
            for line in output_lines:
                if line.strip().startswith('CSV:'):
                    parts = line.strip()[4:].split(',')
                    if len(parts) >= 6:
                        try:
                            workers = int(parts[4])
                            elapsed = float(parts[5])
                        except ValueError:
                            pass

            self.after(0, self._on_sim_done, tech, elapsed, workers)
        except Exception as e:
            self.after(0, self.log, f"BLAD: {e}")
            self.after(0, self.set_status, f'[!] Blad: {e}')
        finally:
            self.running = False
            self.process = None

    def _on_sim_done(self, tech, elapsed, workers):
        if elapsed is not None:
            self.results[tech] = {'time': elapsed, 'workers': workers}
            self.info_time.config(text=f"Czas: {elapsed:.4f} s")
            self.log(f"\n[OK] {TECH_NAMES[tech]} zakonczone: {elapsed:.4f} s")

            if 'seq' in self.results and tech != 'seq':
                speedup = self.results['seq']['time'] / elapsed
                self.info_speedup.config(text=f"Przyspieszenie: {speedup:.2f}x")
            elif tech == 'seq':
                self.info_speedup.config(text="(bazowa)")
        else:
            self.log("[!] Nie udalo sie odczytac czasu")

        self.set_status(f'[OK] Zakonczone: {TECH_NAMES[tech]}')
        self._update_results_panel()
        self._show_final_frame()
        if HAS_MPL and len(self.results) > 1:
            self._draw_benchmark_chart()

    def _poll_frames(self):
        if not self.running:
            return
        ppm_files = sort_frames(glob.glob(str(OUTPUT_DIR / 'frame_*.ppm')))
        if ppm_files:
            self._display_image(ppm_files[-1])
        self.after(800, self._poll_frames)

    def _show_final_frame(self):
        final = OUTPUT_DIR / 'frame_final.ppm'
        if final.exists():
            self._display_image(str(final))
        else:
            ppm_files = sort_frames(glob.glob(str(OUTPUT_DIR / 'frame_*.ppm')))
            if ppm_files:
                self._display_image(ppm_files[-1])

    def _display_image(self, path):
        self.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        max_dim = min(cw, ch) - 20 if min(cw, ch) > 100 else 480
        photo = load_ppm(path, max_dim)
        if photo:
            self.current_photo = photo
            self.canvas.delete('all')
            self.canvas.create_image(cw // 2, ch // 2, image=photo,
                                     anchor='center')

    # --------------------------------------------------------
    # Animacja
    # --------------------------------------------------------
    def play_animation(self):
        ppm_files = sort_frames(glob.glob(str(OUTPUT_DIR / 'frame_*.ppm')))
        ppm_files = [f for f in ppm_files
                     if 'final' not in os.path.basename(f)]
        if not ppm_files:
            self.set_status("[!] Brak klatek. Uruchom symulacje.")
            return
        if self.anim_playing:
            self.anim_playing = False
            self.set_status("Animacja zatrzymana")
            return

        self.set_status(f"Animacja: {len(ppm_files)} klatek")
        self.log(f"[>] Odtwarzanie animacji: {len(ppm_files)} klatek")
        self.anim_frames = ppm_files
        self.anim_idx = 0
        self.anim_playing = True
        self._animate_step()

    def _animate_step(self):
        if not self.anim_playing or self.anim_idx >= len(self.anim_frames):
            self.anim_playing = False
            self.set_status("[OK] Animacja zakonczona")
            self._show_final_frame()
            return
        self._display_image(self.anim_frames[self.anim_idx])
        self.set_status(
            f"Klatka {self.anim_idx + 1}/{len(self.anim_frames)}")
        self.anim_idx += 1
        self.after(150, self._animate_step)

    # --------------------------------------------------------
    # Czyszczenie
    # --------------------------------------------------------
    def clear_output(self):
        if OUTPUT_DIR.exists():
            shutil.rmtree(OUTPUT_DIR)
            OUTPUT_DIR.mkdir(exist_ok=True)
        self.results.clear()
        self.canvas.delete('all')
        self.current_photo = None
        self.info_time.config(text="Czas: -")
        self.info_speedup.config(text="")
        self.info_tech.config(text="Technologia: -", fg=C.TEXT)
        self._update_results_panel()
        self.set_status("Wyczyszczono")
        self.log("[X] Wyczyszczono wyniki")

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------
    def start_benchmark(self):
        if self.running or self.bench_running:
            self.set_status("[!] Poczekaj na zakonczenie")
            return

        self.results.clear()
        self._update_results_panel()
        self.bench_running = True
        self.set_status("[B] Uruchamiam benchmark...")
        self.log(f"\n{'='*60}")
        self.log("[B] BENCHMARK - pełny pakiet testowy")
        self.log("    Siatki:   500, 1000, 2000")
        self.log("    Iteracje: 1000")
        self.log("    Szablon:  5-pkt")
        self.log('='*60)

        thread = threading.Thread(target=self._run_benchmark, daemon=True)
        thread.start()

    def _run_benchmark(self):
        if IS_WINDOWS:
            cmd = f'wsl -d Ubuntu-24.04 -e bash -c "cd {WSL_PROJECT} && bash scripts/run_benchmarks.sh 2>&1"'
        else:
            cmd = f'bash -c "cd {WSL_PROJECT} && bash scripts/run_benchmarks.sh 2>&1"'
        
        try:
            self.process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)

            target_size = str(self.size_var.get())

            for line in self.process.stdout:
                line = line.rstrip()
                self.after(0, self.log, line)
                if line.startswith("---") or line.startswith("==="):
                    self.after(0, self.set_status, f'[B] {line}')
                elif line.startswith("CSV:"):
                    parts = line[4:].strip().split(',')
                    if len(parts) >= 6:
                        tech = parts[0].strip()
                        size = parts[1].strip()
                        if size == target_size:
                            try:
                                elapsed = float(parts[5].strip())
                                workers = int(parts[4].strip())
                                if tech not in self.results or elapsed < self.results[tech]['time']:
                                    self.results[tech] = {'time': elapsed, 'workers': workers}
                            except ValueError:
                                pass

            self.process.wait()
        except Exception as e:
            self.after(0, self.log, f"  [!] Blad: {e}")

        self.bench_running = False
        self.after(0, self._on_benchmark_done)

    def _on_benchmark_done(self):
        self.set_status("[OK] Benchmark zakonczony!")
        self._update_results_panel()
        self._refresh_chart_gallery()
        self.log(f"\n{'='*60}")
        self.log("[B] PODSUMOWANIE BENCHMARKU ZAKOŃCZONE")
        self.log(f"Wykresy wczytane z: {CHARTS_DIR}/")
        self.log('='*60)

    # --------------------------------------------------------
    # Panel wynikow
    # --------------------------------------------------------
    def _update_results_panel(self):
        for w in self.results_frame.winfo_children():
            w.destroy()

        if not self.results:
            tk.Label(self.results_frame, text="Brak wynikow", bg=C.SURFACE,
                     fg=C.DIM, font=(_SANS, 8)).pack(anchor='w')
            return

        seq_time = self.results.get('seq', {}).get('time', None)

        for tech in ['seq', 'omp', 'mpi', 'cuda']:
            if tech not in self.results:
                continue
            data = self.results[tech]
            row = tk.Frame(self.results_frame, bg=C.SURFACE)
            row.pack(fill='x', pady=0)

            tk.Label(row, text=TECH_NAMES[tech], bg=C.SURFACE,
                     fg=TECH_COLORS[tech], font=(_SANS, 8, 'bold'),
                     width=12, anchor='w').pack(side='left')

            tk.Label(row, text=f"{data['time']:.4f}s", bg=C.SURFACE,
                     fg=C.TEXT, font=(_MONO, 8)).pack(side='left')

            if seq_time and tech != 'seq':
                speedup = seq_time / data['time']
                tk.Label(row, text=f"  {speedup:.1f}x", bg=C.SURFACE,
                         fg=C.GREEN,
                         font=(_MONO, 8, 'bold')).pack(side='right')

    # --------------------------------------------------------
    # Galeria wykresow
    # --------------------------------------------------------
    def _refresh_chart_gallery(self):
        def extract_key(f):
            name = os.path.basename(f)
            m = re.search(r'_(\d+)\.png$', name)
            if m:
                return (int(m.group(1)), name)
            return (float('inf'), name)
            
        self.chart_files = sorted(glob.glob(str(CHARTS_DIR / '*.png')), key=extract_key)
        if self.chart_files:
            self.chart_idx = 0
            self._show_chart(self.chart_idx)
        else:
            self.chart_label.config(text="Brak wykresow")

    def _show_chart(self, idx):
        if not self.chart_files:
            return
        idx = idx % len(self.chart_files)
        self.chart_idx = idx
        path = self.chart_files[idx]
        name = os.path.basename(path)
        self.chart_label.config(
            text=f"{name}  ({idx + 1}/{len(self.chart_files)})",
            fg=C.TEXT)

        # Pokaz wykres w tkinter label
        photo = load_png(path, max_w=1000, max_h=600)
        if photo:
            self.chart_photo = photo
            self.chart_image_label.config(image=photo, text="")
        else:
            self.log(f"[!] Blad ladowania wykresu: {path}")

    def _chart_prev(self):
        if self.chart_files:
            self._show_chart(self.chart_idx - 1)

    def _chart_next(self):
        if self.chart_files:
            self._show_chart(self.chart_idx + 1)


# ============================================================
# Punkt wejscia
# ============================================================
if __name__ == '__main__':
    app = DiffusionApp()
    app.mainloop()
