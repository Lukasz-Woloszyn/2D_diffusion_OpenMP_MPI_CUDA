/**
 * Rozprzestrzenianie koloru - Implementacja CUDA
 * Obliczenia na GPU z wykorzystaniem pamięci współdzielonej.
 * Każdy blok wątków ładuje fragment siatki + halo do shared memory.
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir(path, 0755)
#endif

// ============================================================
// Argumenty
// ============================================================
struct Args {
    int size = 1000;
    int iterations = 500;
    int stencil = 5;
    int blockSize = 16;
    std::string output = "output";
    int saveInterval = 50;
    bool benchmark = false;
    bool help = false;

    static Args parse(int argc, char* argv[]) {
        Args a;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i],"--size") && i+1<argc) a.size=atoi(argv[++i]);
            else if (!strcmp(argv[i],"--iterations") && i+1<argc) a.iterations=atoi(argv[++i]);
            else if (!strcmp(argv[i],"--stencil") && i+1<argc) a.stencil=atoi(argv[++i]);
            else if (!strcmp(argv[i],"--block-size") && i+1<argc) a.blockSize=atoi(argv[++i]);
            else if (!strcmp(argv[i],"--output") && i+1<argc) a.output=argv[++i];
            else if (!strcmp(argv[i],"--save-interval") && i+1<argc) a.saveInterval=atoi(argv[++i]);
            else if (!strcmp(argv[i],"--benchmark")) a.benchmark=true;
            else if (!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")) a.help=true;
        }
        return a;
    }
    void print() const {
        printf("  Siatka: %dx%d, Iter: %d, Stencil: %d, Block: %d\n",
               size, size, iterations, stencil, blockSize);
    }
};

// ============================================================
// Struktury kolorów i źródeł
// ============================================================
struct ColorSource {
    int x, y;
    float r, g, b;
};

// ============================================================
// CUDA kernele
// ============================================================

/**
 * Kernel dyfuzji 5-punktowej z pamięcią współdzieloną.
 * Każdy blok wątków ładuje kafelek + 1-pikselowe halo do shared memory,
 * co minimalizuje dostępy do pamięci globalnej.
 */
__global__ void diffuseKernel5_shared(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int W, int H) {
    extern __shared__ float tile[];
    // Szerokość kafelka w shared memory (block + 2*halo)
    int tileW = blockDim.x + 2;

    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int lx = threadIdx.x + 1; // pozycja w tile (z halo offset)
    int ly = threadIdx.y + 1;

    // Ładowanie centralnej części do shared memory
    if (gx < W && gy < H) {
        int gidx = (gy * W + gx) * 3;
        int lidx = (ly * tileW + lx) * 3;
        tile[lidx] = input[gidx];
        tile[lidx+1] = input[gidx+1];
        tile[lidx+2] = input[gidx+2];
    }

    // Ładowanie halo - górny wiersz
    if (threadIdx.y == 0) {
        int sy = gy - 1;
        int lidx = (0 * tileW + lx) * 3;
        if (sy >= 0 && gx < W) {
            int gidx = (sy * W + gx) * 3;
            tile[lidx] = input[gidx]; tile[lidx+1] = input[gidx+1]; tile[lidx+2] = input[gidx+2];
        } else {
            tile[lidx] = 0; tile[lidx+1] = 0; tile[lidx+2] = 0;
        }
    }
    // Dolny wiersz
    if (threadIdx.y == blockDim.y - 1 || gy == H - 1) {
        int sy = gy + 1;
        int lidx = ((ly+1) * tileW + lx) * 3;
        if (sy < H && gx < W) {
            int gidx = (sy * W + gx) * 3;
            tile[lidx] = input[gidx]; tile[lidx+1] = input[gidx+1]; tile[lidx+2] = input[gidx+2];
        } else {
            tile[lidx] = 0; tile[lidx+1] = 0; tile[lidx+2] = 0;
        }
    }
    // Lewa kolumna
    if (threadIdx.x == 0) {
        int sx = gx - 1;
        int lidx = (ly * tileW + 0) * 3;
        if (sx >= 0 && gy < H) {
            int gidx = (gy * W + sx) * 3;
            tile[lidx] = input[gidx]; tile[lidx+1] = input[gidx+1]; tile[lidx+2] = input[gidx+2];
        } else {
            tile[lidx] = 0; tile[lidx+1] = 0; tile[lidx+2] = 0;
        }
    }
    // Prawa kolumna
    if (threadIdx.x == blockDim.x - 1 || gx == W - 1) {
        int sx = gx + 1;
        int lidx = (ly * tileW + (lx+1)) * 3;
        if (sx < W && gy < H) {
            int gidx = (gy * W + sx) * 3;
            tile[lidx] = input[gidx]; tile[lidx+1] = input[gidx+1]; tile[lidx+2] = input[gidx+2];
        } else {
            tile[lidx] = 0; tile[lidx+1] = 0; tile[lidx+2] = 0;
        }
    }

    __syncthreads();

    if (gx >= W || gy >= H) return;

    float sR=0, sG=0, sB=0;
    int cnt=0;
    int ci = (ly*tileW+lx)*3;
    sR+=tile[ci]; sG+=tile[ci+1]; sB+=tile[ci+2]; cnt++;

    if (gy > 0)   { int i=((ly-1)*tileW+lx)*3; sR+=tile[i]; sG+=tile[i+1]; sB+=tile[i+2]; cnt++; }
    if (gy < H-1) { int i=((ly+1)*tileW+lx)*3; sR+=tile[i]; sG+=tile[i+1]; sB+=tile[i+2]; cnt++; }
    if (gx > 0)   { int i=(ly*tileW+(lx-1))*3; sR+=tile[i]; sG+=tile[i+1]; sB+=tile[i+2]; cnt++; }
    if (gx < W-1) { int i=(ly*tileW+(lx+1))*3; sR+=tile[i]; sG+=tile[i+1]; sB+=tile[i+2]; cnt++; }

    int oi = (gy*W+gx)*3;
    float inv = 1.0f/cnt;
    output[oi]=sR*inv; output[oi+1]=sG*inv; output[oi+2]=sB*inv;
}

/**
 * Kernel dyfuzji 9-punktowej z pamięcią współdzieloną.
 */
__global__ void diffuseKernel9_shared(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int W, int H) {
    extern __shared__ float tile[];
    int tileW = blockDim.x + 2;

    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    // Ładowanie do shared (podobnie jak wyżej)
    if (gx < W && gy < H) {
        int gi=(gy*W+gx)*3, li=(ly*tileW+lx)*3;
        tile[li]=input[gi]; tile[li+1]=input[gi+1]; tile[li+2]=input[gi+2];
    }
    if (threadIdx.y==0) {
        int sy=gy-1, li=(0*tileW+lx)*3;
        if(sy>=0&&gx<W){int gi=(sy*W+gx)*3;tile[li]=input[gi];tile[li+1]=input[gi+1];tile[li+2]=input[gi+2];}
        else{tile[li]=0;tile[li+1]=0;tile[li+2]=0;}
    }
    if (threadIdx.y==blockDim.y-1||gy==H-1) {
        int sy=gy+1, li=((ly+1)*tileW+lx)*3;
        if(sy<H&&gx<W){int gi=(sy*W+gx)*3;tile[li]=input[gi];tile[li+1]=input[gi+1];tile[li+2]=input[gi+2];}
        else{tile[li]=0;tile[li+1]=0;tile[li+2]=0;}
    }
    if (threadIdx.x==0) {
        int sx=gx-1, li=(ly*tileW+0)*3;
        if(sx>=0&&gy<H){int gi=(gy*W+sx)*3;tile[li]=input[gi];tile[li+1]=input[gi+1];tile[li+2]=input[gi+2];}
        else{tile[li]=0;tile[li+1]=0;tile[li+2]=0;}
    }
    if (threadIdx.x==blockDim.x-1||gx==W-1) {
        int sx=gx+1, li=(ly*tileW+(lx+1))*3;
        if(sx<W&&gy<H){int gi=(gy*W+sx)*3;tile[li]=input[gi];tile[li+1]=input[gi+1];tile[li+2]=input[gi+2];}
        else{tile[li]=0;tile[li+1]=0;tile[li+2]=0;}
    }
    // Rogi
    if (threadIdx.x==0&&threadIdx.y==0) {
        int sx=gx-1,sy=gy-1,li=0;
        if(sx>=0&&sy>=0){int gi=(sy*W+sx)*3;tile[li]=input[gi];tile[li+1]=input[gi+1];tile[li+2]=input[gi+2];}
        else{tile[li]=0;tile[li+1]=0;tile[li+2]=0;}
    }
    if ((threadIdx.x==blockDim.x-1||gx==W-1)&&threadIdx.y==0) {
        int sx=gx+1,sy=gy-1,li=(0*tileW+(lx+1))*3;
        if(sx<W&&sy>=0){int gi=(sy*W+sx)*3;tile[li]=input[gi];tile[li+1]=input[gi+1];tile[li+2]=input[gi+2];}
        else{tile[li]=0;tile[li+1]=0;tile[li+2]=0;}
    }
    if (threadIdx.x==0&&(threadIdx.y==blockDim.y-1||gy==H-1)) {
        int sx=gx-1,sy=gy+1,li=((ly+1)*tileW+0)*3;
        if(sx>=0&&sy<H){int gi=(sy*W+sx)*3;tile[li]=input[gi];tile[li+1]=input[gi+1];tile[li+2]=input[gi+2];}
        else{tile[li]=0;tile[li+1]=0;tile[li+2]=0;}
    }
    if ((threadIdx.x==blockDim.x-1||gx==W-1)&&(threadIdx.y==blockDim.y-1||gy==H-1)) {
        int sx=gx+1,sy=gy+1,li=((ly+1)*tileW+(lx+1))*3;
        if(sx<W&&sy<H){int gi=(sy*W+sx)*3;tile[li]=input[gi];tile[li+1]=input[gi+1];tile[li+2]=input[gi+2];}
        else{tile[li]=0;tile[li+1]=0;tile[li+2]=0;}
    }

    __syncthreads();
    if (gx>=W||gy>=H) return;

    float sR=0,sG=0,sB=0; int cnt=0;
    for(int dy=-1;dy<=1;dy++){
        for(int dx=-1;dx<=1;dx++){
            int ny=gy+dy,nx=gx+dx;
            if(nx<0||nx>=W||ny<0||ny>=H) continue;
            int i=((ly+dy)*tileW+(lx+dx))*3;
            sR+=tile[i]; sG+=tile[i+1]; sB+=tile[i+2]; cnt++;
        }
    }
    int oi=(gy*W+gx)*3; float inv=1.0f/cnt;
    output[oi]=sR*inv; output[oi+1]=sG*inv; output[oi+2]=sB*inv;
}

/**
 * Kernel do przywracania źródeł koloru na GPU.
 */
__global__ void applySourcesKernel(float* grid, const ColorSource* sources,
                                    int numSources, int W) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSources) return;
    int idx = (sources[i].y * W + sources[i].x) * 3;
    grid[idx] = sources[i].r;
    grid[idx+1] = sources[i].g;
    grid[idx+2] = sources[i].b;
}

// ============================================================
// Funkcje pomocnicze (standalone, nie linkujemy common)
// ============================================================

void createDir(const std::string& p) { MKDIR(p.c_str()); }

void savePPM(const float* data, int W, int H, const std::string& filename) {
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) return;
    f << "P6\n" << W << " " << H << "\n255\n";
    for (int i = 0; i < W*H; i++) {
        uint8_t r = (uint8_t)(std::min(std::max(data[i*3],0.0f),1.0f)*255);
        uint8_t g = (uint8_t)(std::min(std::max(data[i*3+1],0.0f),1.0f)*255);
        uint8_t b = (uint8_t)(std::min(std::max(data[i*3+2],0.0f),1.0f)*255);
        f.put(r); f.put(g); f.put(b);
    }
}

void saveBIN(const float* data, int W, int H, const std::string& filename) {
    std::ofstream f(filename, std::ios::binary);
    if (!f.is_open()) return;
    int32_t w=W, h=H;
    f.write((char*)&w,4); f.write((char*)&h,4);
    f.write((char*)data, W*H*3*sizeof(float));
}

std::string frameName(const std::string& dir, int iter, const std::string& ext) {
    char buf[256]; snprintf(buf,256,"%s/frame_%05d%s",dir.c_str(),iter,ext.c_str());
    return buf;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {
    Args args = Args::parse(argc, argv);
    if (args.help) { args.print(); return 0; }

    int N = args.size;
    int dataSize = N * N * 3;
    int BS = args.blockSize;

    printf("=== Dyfuzja Koloru - CUDA ===\n");
    args.print();

    // Informacja o GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("  GPU: %s (SM: %d.%d, %d MB)\n", prop.name,
           prop.major, prop.minor, (int)(prop.totalGlobalMem/1024/1024));

    // Przygotowanie źródeł koloru
    int radius = std::max(3, N/50);
    std::vector<ColorSource> srcVec;
    auto addSrc = [&](int x, int y, float r, float g, float b) {
        srcVec.push_back({x,y,r,g,b});
    };
    addSrc(N/4, N/4, 1,0,0);
    addSrc(3*N/4, N/4, 0,1,0);
    addSrc(N/2, 3*N/4, 0,0,1);
    addSrc(N/2, N/2, 1,1,0);
    addSrc(N/4, 3*N/4, 1,0,1);
    addSrc(3*N/4, 3*N/4, 0,1,1);

    // Inicjalizacja siatki na CPU
    std::vector<float> hostData(dataSize, 0.0f);
    for (auto& s : srcVec) {
        for (int dy=-radius; dy<=radius; dy++) {
            for (int dx=-radius; dx<=radius; dx++) {
                int nx=s.x+dx, ny=s.y+dy;
                if(nx<0||nx>=N||ny<0||ny>=N) continue;
                float dist = sqrtf(float(dx*dx+dy*dy));
                if (dist <= radius) {
                    float fac = 1.0f - dist/(radius+1.0f);
                    int idx=(ny*N+nx)*3;
                    hostData[idx]=s.r*fac; hostData[idx+1]=s.g*fac; hostData[idx+2]=s.b*fac;
                }
            }
        }
    }

    // Alokacja pamięci GPU
    float *d_gridA, *d_gridB;
    ColorSource *d_sources;
    cudaMalloc(&d_gridA, dataSize * sizeof(float));
    cudaMalloc(&d_gridB, dataSize * sizeof(float));
    cudaMalloc(&d_sources, srcVec.size() * sizeof(ColorSource));

    cudaMemcpy(d_gridA, hostData.data(), dataSize*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sources, srcVec.data(), srcVec.size()*sizeof(ColorSource), cudaMemcpyHostToDevice);

    dim3 block(BS, BS);
    dim3 grid((N+BS-1)/BS, (N+BS-1)/BS);
    int sharedSize = (BS+2) * (BS+2) * 3 * sizeof(float);

    if (!args.benchmark) createDir(args.output);

    // Zapis klatki 0
    if (!args.benchmark) {
        savePPM(hostData.data(), N, N, frameName(args.output, 0, ".ppm"));
        saveBIN(hostData.data(), N, N, frameName(args.output, 0, ".bin"));
    }

    // Rozgrzewka GPU
    if (args.stencil == 9)
        diffuseKernel9_shared<<<grid, block, sharedSize>>>(d_gridA, d_gridB, N, N);
    else
        diffuseKernel5_shared<<<grid, block, sharedSize>>>(d_gridA, d_gridB, N, N);
    cudaDeviceSynchronize();
    cudaMemcpy(d_gridB, d_gridA, dataSize*sizeof(float), cudaMemcpyDeviceToDevice);

    // Pomiar czasu GPU
    cudaEvent_t startEv, stopEv;
    cudaEventCreate(&startEv);
    cudaEventCreate(&stopEv);
    cudaEventRecord(startEv);

    float *curDev = d_gridA, *nxtDev = d_gridB;
    int numSrc = (int)srcVec.size();

    for (int iter = 1; iter <= args.iterations; iter++) {
        if (args.stencil == 9)
            diffuseKernel9_shared<<<grid, block, sharedSize>>>(curDev, nxtDev, N, N);
        else
            diffuseKernel5_shared<<<grid, block, sharedSize>>>(curDev, nxtDev, N, N);

        applySourcesKernel<<<(numSrc+255)/256, 256>>>(nxtDev, d_sources, numSrc, N);
        std::swap(curDev, nxtDev);

        if (!args.benchmark && (iter % args.saveInterval == 0 || iter == args.iterations)) {
            cudaMemcpy(hostData.data(), curDev, dataSize*sizeof(float), cudaMemcpyDeviceToHost);
            savePPM(hostData.data(), N, N, frameName(args.output, iter, ".ppm"));
            saveBIN(hostData.data(), N, N, frameName(args.output, iter, ".bin"));
            printf("  Klatka: %d\n", iter);
        }
        if (iter % 100 == 0) printf("  Iteracja %d/%d\n", iter, args.iterations);
    }

    cudaEventRecord(stopEv);
    cudaEventSynchronize(stopEv);
    float elapsedMs;
    cudaEventElapsedTime(&elapsedMs, startEv, stopEv);
    double elapsed = elapsedMs / 1000.0;

    printf("\n=== Wyniki ===\n");
    printf("  GPU: %s\n", prop.name);
    printf("  Block: %dx%d\n", BS, BS);
    printf("  Czas: %.4f s\n", elapsed);
    printf("  Czas/iter: %.4f ms\n", elapsedMs / args.iterations);

    if (!args.benchmark) {
        cudaMemcpy(hostData.data(), curDev, dataSize*sizeof(float), cudaMemcpyDeviceToHost);
        savePPM(hostData.data(), N, N, (args.output + "/frame_final.ppm"));
        saveBIN(hostData.data(), N, N, (args.output + "/frame_final.bin"));
    }

    printf("\nCSV: cuda,%d,%d,%d,%d,%.6f\n", N, args.iterations, args.stencil, BS, elapsed);

    cudaFree(d_gridA); cudaFree(d_gridB); cudaFree(d_sources);
    cudaEventDestroy(startEv); cudaEventDestroy(stopEv);
    return 0;
}
