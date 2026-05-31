/**
 * Rozprzestrzenianie koloru - Implementacja MPI
 * Podział siatki na bloki wierszy (dekompozycja 1D).
 * Wymiana danych brzegowych (halo exchange) między procesami.
 */

#include "../common/args.h"
#include "../common/grid.h"
#include "../common/io.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <mpi.h>
#include <vector>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int rank, numProcs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

  Args args = Args::parse(argc, argv);
  if (args.help) {
    if (rank == 0)
      Args::printHelp(argv[0]);
    MPI_Finalize();
    return 0;
  }

  int N = args.size;
  // Podział wierszy na procesy
  int baseRows = N / numProcs;
  int remainder = N % numProcs;
  int localRows = baseRows + (rank < remainder ? 1 : 0);
  int startRow = rank * baseRows + std::min(rank, remainder);

  // Lokalna siatka z wierszami halo (góra i dół)
  int totalLocalRows = localRows + 2; // +1 halo góra, +1 halo dół
  int localSize = totalLocalRows * N * 3;

  std::vector<float> localA(localSize, 0.0f);
  std::vector<float> localB(localSize, 0.0f);

  if (rank == 0) {
    std::cout << "=== Dyfuzja Koloru - MPI ===\n";
    std::cout << "  Procesy: " << numProcs << "\n";
    args.print();
  }

  // Inicjalizacja - proces 0 tworzy pełną siatkę i rozprowadza
  auto sources = generateDefaultSources(N);
  if (rank == 0) {
    Grid fullGrid(N, N);
    fullGrid.initSources(sources);

    // Kopiowanie danych procesu 0 (wiersze 0..localRows-1 -> wiersz halo+1)
    int r0rows = baseRows + (0 < remainder ? 1 : 0);
    memcpy(localA.data() + N * 3, fullGrid.data.data(),
           r0rows * N * 3 * sizeof(float));

    // Wysyłanie do innych procesów
    for (int p = 1; p < numProcs; p++) {
      int pRows = baseRows + (p < remainder ? 1 : 0);
      int pStart = p * baseRows + std::min(p, remainder);
      MPI_Send(fullGrid.data.data() + pStart * N * 3, pRows * N * 3, MPI_FLOAT,
               p, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(localA.data() + N * 3, localRows * N * 3, MPI_FLOAT, 0, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  auto startTime = std::chrono::high_resolution_clock::now();

  float *curBuf = localA.data();
  float *nxtBuf = localB.data();

  for (int iter = 1; iter <= args.iterations; iter++) {
    // === Wymiana halo ===
    int prevRank = rank - 1;
    int nextRank = rank + 1;

    // Wysyłanie górnego wiersza do poprzedniego procesu
    if (prevRank >= 0) {
      MPI_Sendrecv(curBuf + N * 3, N * 3, MPI_FLOAT, prevRank, 0, curBuf, N * 3,
                   MPI_FLOAT, prevRank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    // Wysyłanie dolnego wiersza do następnego procesu
    if (nextRank < numProcs) {
      MPI_Sendrecv(curBuf + localRows * N * 3, N * 3, MPI_FLOAT, nextRank, 1,
                   curBuf + (localRows + 1) * N * 3, N * 3, MPI_FLOAT, nextRank,
                   0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // === Krok dyfuzji na lokalnym fragmencie ===
    for (int ly = 1; ly <= localRows; ly++) {
      int globalY = startRow + (ly - 1);
      for (int x = 0; x < N; x++) {
        float sR = 0, sG = 0, sB = 0;
        int cnt = 0;
        // Centrum
        int idx = (ly * N + x) * 3;
        sR += curBuf[idx];
        sG += curBuf[idx + 1];
        sB += curBuf[idx + 2];
        cnt++;

        if (args.stencil == 9) {
          // Szablon 9-punktowy
          for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
              if (dy == 0 && dx == 0)
                continue;
              int ny = ly + dy, nx = x + dx;
              if (nx < 0 || nx >= N)
                continue;
              if (globalY + dy < 0 || globalY + dy >= N)
                continue;
              idx = (ny * N + nx) * 3;
              sR += curBuf[idx];
              sG += curBuf[idx + 1];
              sB += curBuf[idx + 2];
              cnt++;
            }
          }
        } else {
          // Szablon 5-punktowy
          if (globalY > 0) {
            idx = ((ly - 1) * N + x) * 3;
            sR += curBuf[idx];
            sG += curBuf[idx + 1];
            sB += curBuf[idx + 2];
            cnt++;
          }
          if (globalY < N - 1) {
            idx = ((ly + 1) * N + x) * 3;
            sR += curBuf[idx];
            sG += curBuf[idx + 1];
            sB += curBuf[idx + 2];
            cnt++;
          }
          if (x > 0) {
            idx = (ly * N + (x - 1)) * 3;
            sR += curBuf[idx];
            sG += curBuf[idx + 1];
            sB += curBuf[idx + 2];
            cnt++;
          }
          if (x < N - 1) {
            idx = (ly * N + (x + 1)) * 3;
            sR += curBuf[idx];
            sG += curBuf[idx + 1];
            sB += curBuf[idx + 2];
            cnt++;
          }
        }

        int oi = (ly * N + x) * 3;
        float inv = 1.0f / cnt;
        nxtBuf[oi] = sR * inv;
        nxtBuf[oi + 1] = sG * inv;
        nxtBuf[oi + 2] = sB * inv;
      }
    }

    // Przywrócenie źródeł
    for (const auto &src : sources) {
      int ly = src.y - startRow + 1; // +1 dla halo
      if (ly >= 1 && ly <= localRows && src.x >= 0 && src.x < N) {
        int idx = (ly * N + src.x) * 3;
        nxtBuf[idx] = src.color.r;
        nxtBuf[idx + 1] = src.color.g;
        nxtBuf[idx + 2] = src.color.b;
      }
    }

    std::swap(curBuf, nxtBuf);

    // Zapis klatki (zbieranie na rank 0)
    if (!args.benchmark &&
        (iter % args.saveInterval == 0 || iter == args.iterations)) {
      if (rank == 0) {
        Grid fullGrid(N, N);
        int r0rows = baseRows + (0 < remainder ? 1 : 0);
        memcpy(fullGrid.data.data(), curBuf + N * 3,
               r0rows * N * 3 * sizeof(float));

        for (int p = 1; p < numProcs; p++) {
          int pRows = baseRows + (p < remainder ? 1 : 0);
          int pStart = p * baseRows + std::min(p, remainder);
          MPI_Recv(fullGrid.data.data() + pStart * N * 3, pRows * N * 3,
                   MPI_FLOAT, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        saveGridPPM(fullGrid, frameFilename(args.output, iter, ".ppm"));
        saveGridBinary(fullGrid, frameFilename(args.output, iter, ".bin"));
        std::cout << "  Klatka: " << iter << "\n";
      } else {
        MPI_Send(curBuf + N * 3, localRows * N * 3, MPI_FLOAT, 0, 2,
                 MPI_COMM_WORLD);
      }
    }

    if (rank == 0 && iter % 100 == 0)
      std::cout << "  Iteracja " << iter << "/" << args.iterations << "\n";
  }

  MPI_Barrier(MPI_COMM_WORLD);
  auto endTime = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(endTime - startTime).count();

  if (rank == 0) {
    std::cout << "\n=== Wyniki ===\n"
              << "  Procesy: " << numProcs << "\n"
              << "  Czas: " << elapsed << " s\n"
              << "  Czas/iter: " << (elapsed / args.iterations * 1000)
              << " ms\n";

    if (!args.benchmark) {
      // Zebranie finalnej siatki
      Grid fullGrid(N, N);
      int r0rows = baseRows + (0 < remainder ? 1 : 0);
      memcpy(fullGrid.data.data(), curBuf + N * 3,
             r0rows * N * 3 * sizeof(float));
      for (int p = 1; p < numProcs; p++) {
        int pRows = baseRows + (p < remainder ? 1 : 0);
        int pStart = p * baseRows + std::min(p, remainder);
        MPI_Recv(fullGrid.data.data() + pStart * N * 3, pRows * N * 3,
                 MPI_FLOAT, p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      createOutputDir(args.output);
      saveGridPPM(fullGrid, args.output + "/frame_final.ppm");
      saveGridBinary(fullGrid, args.output + "/frame_final.bin");
    }
    std::cout << "\nCSV: mpi," << N << "," << args.iterations << ","
              << args.stencil << "," << numProcs << "," << elapsed << "\n";
  } else if (!args.benchmark) {
    MPI_Send(curBuf + N * 3, localRows * N * 3, MPI_FLOAT, 0, 3,
             MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}
