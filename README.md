# cuda-matrix-engine

A performance-first CUDA C++ library for single-precision matrix multiplication:

```
C = alpha * A * B + beta * C
```

- Row-major storage, FP32 only, synchronous public API.
- Algorithms: CPU reference, naive CUDA, shared-memory tiled CUDA, and an optimized CUDA kernel
  with a register micro-tile.
- Target hardware: NVIDIA GeForce RTX 4070 Ti SUPER (compute capability 8.9, sm_89). Linux is the
  primary platform; Windows 11 is secondary.

Author: Nirmal. License: Apache-2.0 (see [LICENSE](LICENSE)).

## Status

Batch-level status lives in [TASK_BOARD.md](TASK_BOARD.md). Batches 0.1–0.3 (environment and
control documents) and 7.1–7.3 (profiling and final benchmark) are not yet verified.

## Requirements

- CMake 3.24 or newer.
- A C++17 host compiler (g++ 13.3.0 was used during development).
- CUDA Toolkit 12.0 (used during development). The GPU architecture defaults to 89; set
  `CME_CUDA_ARCHITECTURES` to target another GPU.
- Catch2 v3.9.1 is fetched automatically by CMake when tests are enabled.
- For sanitizer runs: `compute-sanitizer`. Point `CME_COMPUTE_SANITIZER` at it if it is not on
  `PATH`.

## Build

```sh
cmake --preset linux-release
cmake --build --preset linux-release
```

Use `linux-debug` instead of `linux-release` for a debug build with tests enabled.

Options:

| Option | Default | Meaning |
|---|---|---|
| `CME_BUILD_TESTS` | ON | Build the Catch2 test suite (`cme_tests`). |
| `CME_BUILD_BENCHMARKS` | ON | Build the benchmark driver (`cme_benchmark`). |
| `CME_BUILD_EXAMPLES` | ON | Build the three examples. |
| `CME_WITH_CUBLAS` | OFF | Enable the optional cuBLAS reference row. Not measured in the current results. |
| `CME_CUDA_ARCHITECTURES` | 89 | CUDA architecture(s) to compile for. |

## Test

```sh
./scripts/test.sh
```

The script configures and builds the `linux-debug` preset, runs `ctest`, runs `memcheck` on the full
test binary, and runs `racecheck` on the `[racecheck]` tests. It exits non-zero on any failure.
GPU-dependent tests are meant to skip with a message when no CUDA device is present
(see [QUALITY_STANDARDS.md](QUALITY_STANDARDS.md)).

## Benchmarks

```sh
./scripts/benchmark.sh           # full run: sizes 256–4096, 30 measured iterations
./scripts/benchmark.sh --quick   # smoke test: sizes 256 and 512 only
```

Each run writes a CSV to `results/` and an environment snapshot beside it. Column order and
methodology are described in [results/README.md](results/README.md) and
[docs/BENCHMARKING.md](docs/BENCHMARKING.md).

## Usage

From [examples/basic_gemm.cpp](examples/basic_gemm.cpp):

```cpp
#include <iostream>

#include "cme/gemm.hpp"
#include "cme/matrix.hpp"
#include "cme/status.hpp"

int main() {
    constexpr int M = 128;
    constexpr int K = 64;
    constexpr int N = 32;

    cme::Matrix A(M, K);
    cme::Matrix B(K, N);
    cme::Matrix C(M, N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            A(i, j) = 0.01f * static_cast<float>(i + 1) + 0.001f * static_cast<float>(j + 1);
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            B(i, j) = 0.02f * static_cast<float>(i + 1) - 0.003f * static_cast<float>(j + 1);
        }
    }

    cme::GemmConfig config;
    const cme::Status status = cme::gemm(A.view(), B.view(), C.view(), 1.0f, 0.0f, config);
    if (!status.is_ok()) {
        std::cerr << "gemm failed: " << status.message() << '\n';
        return 1;
    }

    std::cout << "C(0,0)..C(0,3):";
    for (int c = 0; c < 4; ++c) {
        std::cout << ' ' << C(0, c);
    }
    std::cout << '\n';
    return 0;
}
```

The default `GemmConfig` selects `CUDA_TILED` with `tile_size = 16`. That default is frozen by the
interface, and it is not the fastest measured configuration. To use the fastest measured kernel,
set `config.algorithm = cme::GemmAlgorithm::CUDA_OPTIMIZED` and `config.tile_size = 32`.

## Headline results

From [docs/PERFORMANCE_REPORT.md](docs/PERFORMANCE_REPORT.md), using the final run
[results/gemm_20261103T062611Z.csv](results/gemm_20261103T062611Z.csv) on the target GPU, FP32,
square 2048×2048×2048:

| Configuration | Kernel-only median (GFLOP/s) | End-to-end median (ms) |
|---|---:|---:|
| `CUDA_TILED`, tile 16 (public default) | 3737.16 | 11.24 |
| `CUDA_OPTIMIZED`, tile 32 (fastest measured) | 14075.40 | 7.95 |

Kernel-only excludes host-device copies and reuses device buffers. End-to-end includes validation,
allocation, copies, launch, and synchronization. cuBLAS was not measured.

## Known limitations

- FP32 only, row-major, no transposed inputs, no batched GEMM, and no Tensor Core kernels.
- The public API is synchronous and copies A, B, and C to and from the device on every call.
- The CPU reference is a scalar implementation. Its timings appear only for sizes 256 and 512.
- cuBLAS is not measured, so no comparison against it is claimed.
- The public default algorithm is not the fastest measured configuration (see above).
- CI runs on hosted runners without a GPU. It builds the project and runs the CPU-side tests. It
  does not verify GPU correctness.

## Documentation

- [HANDOFF.md](HANDOFF.md): frozen architecture, interface contract, and batch plan.
- [INTERFACES.md](INTERFACES.md), [ARCHITECTURE.md](ARCHITECTURE.md), and
  [QUALITY_STANDARDS.md](QUALITY_STANDARDS.md): binding specifications.
- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md): recorded hardware and toolchain.
- [docs/ALGORITHMS.md](docs/ALGORITHMS.md), [docs/BENCHMARKING.md](docs/BENCHMARKING.md),
  [docs/PROFILING.md](docs/PROFILING.md), [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md),
  and [docs/PERFORMANCE_REPORT.md](docs/PERFORMANCE_REPORT.md).
- [handoffs/](handoffs/): one note per batch.
