/**
 * Rozprzestrzenianie koloru - Implementacja OpenMP
 * Równoległa aktualizacja komórek siatki z dyrektywami OpenMP.
 */

#include "../common/args.h"
#include "../common/grid.h"
#include "../common/io.h"
#include <chrono>
#include <iostream>
#include <omp.h>

void diffuseStep5_OMP(const Grid &input, Grid &output, int numThreads) {
  int W = input.width;
  int H = input.height;
  const float *in = input.data.data();
  float *out = output.data.data();

#pragma omp parallel for schedule(static) num_threads(numThreads)
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      float sumR = 0, sumG = 0, sumB = 0;
      int count = 0;
      int idx = (y * W + x) * 3;
      sumR += in[idx];
      sumG += in[idx + 1];
      sumB += in[idx + 2];
      count++;
      if (y > 0) {
        idx = ((y - 1) * W + x) * 3;
        sumR += in[idx];
        sumG += in[idx + 1];
        sumB += in[idx + 2];
        count++;
      }
      if (y < H - 1) {
        idx = ((y + 1) * W + x) * 3;
        sumR += in[idx];
        sumG += in[idx + 1];
        sumB += in[idx + 2];
        count++;
      }
      if (x > 0) {
        idx = (y * W + (x - 1)) * 3;
        sumR += in[idx];
        sumG += in[idx + 1];
        sumB += in[idx + 2];
        count++;
      }
      if (x < W - 1) {
        idx = (y * W + (x + 1)) * 3;
        sumR += in[idx];
        sumG += in[idx + 1];
        sumB += in[idx + 2];
        count++;
      }
      int oi = (y * W + x) * 3;
      float inv = 1.0f / count;
      out[oi] = sumR * inv;
      out[oi + 1] = sumG * inv;
      out[oi + 2] = sumB * inv;
    }
  }
}

void diffuseStep9_OMP(const Grid &input, Grid &output, int numThreads) {
  int W = input.width, H = input.height;
  const float *in = input.data.data();
  float *out = output.data.data();

#pragma omp parallel for schedule(static) num_threads(numThreads)
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      float sumR = 0, sumG = 0, sumB = 0;
      int count = 0;
      for (int dy = -1; dy <= 1; dy++) {
        int ny = y + dy;
        if (ny < 0 || ny >= H)
          continue;
        for (int dx = -1; dx <= 1; dx++) {
          int nx = x + dx;
          if (nx < 0 || nx >= W)
            continue;
          int idx = (ny * W + nx) * 3;
          sumR += in[idx];
          sumG += in[idx + 1];
          sumB += in[idx + 2];
          count++;
        }
      }
      int oi = (y * W + x) * 3;
      float inv = 1.0f / count;
      out[oi] = sumR * inv;
      out[oi + 1] = sumG * inv;
      out[oi + 2] = sumB * inv;
    }
  }
}

int main(int argc, char *argv[]) {
  Args args = Args::parse(argc, argv);
  if (args.help) {
    Args::printHelp(argv[0]);
    return 0;
  }

  omp_set_num_threads(args.threads);
  std::cout << "=== Dyfuzja Koloru - OpenMP ===\n";
  std::cout << "  Watki: " << args.threads << "\n";
  args.print();

  Grid gridA(args.size, args.size), gridB(args.size, args.size);
  auto sources = generateDefaultSources(args.size);
  gridA.initSources(sources);

  if (!args.benchmark) {
    createOutputDir(args.output);
    saveGridPPM(gridA, frameFilename(args.output, 0, ".ppm"));
    saveGridBinary(gridA, frameFilename(args.output, 0, ".bin"));
  }

  // Rozgrzewka
  diffuseStep5_OMP(gridA, gridB, args.threads);
  gridB.copyFrom(gridA);

  auto startTime = std::chrono::high_resolution_clock::now();
  Grid *cur = &gridA;
  Grid *nxt = &gridB;

  for (int iter = 1; iter <= args.iterations; iter++) {
    if (args.stencil == 9)
      diffuseStep9_OMP(*cur, *nxt, args.threads);
    else
      diffuseStep5_OMP(*cur, *nxt, args.threads);
    nxt->applySources(sources);
    std::swap(cur, nxt);
    if (!args.benchmark &&
        (iter % args.saveInterval == 0 || iter == args.iterations)) {
      saveGridPPM(*cur, frameFilename(args.output, iter, ".ppm"));
      saveGridBinary(*cur, frameFilename(args.output, iter, ".bin"));
      std::cout << "  Klatka: " << iter << "\n";
    }
    if (iter % 100 == 0)
      std::cout << "  Iteracja " << iter << "/" << args.iterations << "\n";
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(endTime - startTime).count();

  std::cout << "\n=== Wyniki ===\n"
            << "  Watki: " << args.threads << "\n"
            << "  Czas: " << elapsed << " s\n"
            << "  Czas/iter: " << (elapsed / args.iterations * 1000) << " ms\n";

  if (!args.benchmark) {
    saveGridPPM(*cur, args.output + "/frame_final.ppm");
    saveGridBinary(*cur, args.output + "/frame_final.bin");
  }

  std::cout << "\nCSV: omp," << args.size << "," << args.iterations << ","
            << args.stencil << "," << args.threads << "," << elapsed << "\n";
  return 0;
}
