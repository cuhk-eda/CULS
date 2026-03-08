cmake_minimum_required(VERSION 3.18)
project(gpuls C CXX CUDA)

include(FetchContent)

# ── CCCL (Thrust, CUB, libcu++) ───────────────────────────────────────────────
FetchContent_Declare(
    CCCL
    GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
    GIT_TAG        main   # replace with a version tag, e.g. v2.8.0, for reproducibility
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(CCCL)

# ── Language standards ────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD  17)
set(CMAKE_CUDA_STANDARD 17)   # replaces the per-flag -std=c++17

# ── Sources ───────────────────────────────────────────────────────────────────
# CONFIGURE_DEPENDS makes CMake re-scan when you add/remove files
file(GLOB_RECURSE GPULS_SOURCES CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.cu"
)

# ── Target ────────────────────────────────────────────────────────────────────
add_executable(gpuls ${GPULS_SOURCES})

target_include_directories(gpuls PRIVATE src src/hash_table include)

# sm_120 = Blackwell (RTX 500 Pro / RTX 50-series)  |  CUDA 13.1  |  GCC 13.3  |  Ubuntu 24.04
target_compile_options(gpuls PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_120 -O3 --extended-lambda>
)

target_link_libraries(gpuls PRIVATE CCCL::CCCL)