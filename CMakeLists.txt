cmake_minimum_required(VERSION 3.18)
project(ColorDiffusion LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ============================================================
# Common library
# ============================================================
add_library(common STATIC
    src/common/grid.cpp
    src/common/io.cpp
)
target_include_directories(common PUBLIC src/common)

# ============================================================
# Sequential version
# ============================================================
add_executable(diffusion_seq src/sequential/main.cpp)
target_link_libraries(diffusion_seq PRIVATE common)

# ============================================================
# OpenMP version
# ============================================================
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_executable(diffusion_omp src/openmp/main.cpp)
    target_link_libraries(diffusion_omp PRIVATE common OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found - building OpenMP version")
else()
    message(WARNING "OpenMP not found - skipping OpenMP version")
endif()

# ============================================================
# MPI version
# ============================================================
find_package(MPI)
if(MPI_CXX_FOUND)
    add_executable(diffusion_mpi src/mpi/main.cpp)
    target_link_libraries(diffusion_mpi PRIVATE common MPI::MPI_CXX)
    message(STATUS "MPI found - building MPI version")
else()
    message(WARNING "MPI not found - skipping MPI version")
endif()

# ============================================================
# CUDA version
# ============================================================
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Enable shared memory optimizations
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
    
    add_executable(diffusion_cuda src/cuda/main.cu)
    target_link_libraries(diffusion_cuda PRIVATE common)
    set_target_properties(diffusion_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    message(STATUS "CUDA found - building CUDA version")
else()
    message(WARNING "CUDA not found - skipping CUDA version")
endif()
