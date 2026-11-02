# CUDA Matrix Engine — Master Handoff

**Project:** `cuda-matrix-engine` (Project 01 of the CUDA Portfolio)
**Owner:** Nirmal
**Status of this document:** Architecture frozen. Implementation not started.
**Copy location:** Place this file at the repository root as `HANDOFF.md` once the repo exists. Until then it lives beside the project plan.

---

## 0. Read this first (for every agent)

You are one agent in a chain. Agents may run out of context, time, or quota at any moment, and the next agent will pick up where you stopped. This file is designed so that **any agent can resume the project from any point** without asking the owner to re-explain anything.

Follow this loop every session:

1. **Read this file** sections 0, 1, 2, 3 and 4 in full. Do not skip the interface contracts in section 4. They are binding.
2. **Read `TASK_BOARD.md`** (created in Batch 0.3). Find the first batch whose status is not `VERIFIED`.
3. **Read the batch** in section 6 that matches that ID. Check its dependencies are `VERIFIED`. If a dependency is not verified, stop and mark your batch `BLOCKED` with the reason.
4. **Mark the batch `IN_PROGRESS`** on `TASK_BOARD.md` before writing any code. Record your agent name and the date.
5. **Do only the files listed for that batch.** If you need a change in a file owned by another batch, do not make it. Write the request in the batch's handoff note and mark yourself `BLOCKED`.
6. **Run the batch's verification command.** Paste the actual output summary into the handoff note.
7. **Write a handoff note** to `handoffs/<BATCH_ID>.md` using the template in section 8.
8. **Update `TASK_BOARD.md`** to `UNDER_REVIEW` (you finished it) or `BLOCKED`. Only the verifier role moves a batch to `VERIFIED`.
9. **Stop.** Do not start the next batch in the same session unless your context clearly has room and the next batch is small. Finishing one batch cleanly is worth more than starting two and leaving a mess.

### Non-negotiable rules

- **Interfaces are frozen.** Section 4 defines every public signature. Do not rename, reorder, or change types. If you believe an interface is wrong, mark `BLOCKED` and explain. Do not silently change it.
- **Public headers in `include/cme/` must not include any CUDA header.** Host applications must compile without CUDA headers. All CUDA-dependent code lives in `src/cuda/`.
- **A file's status is not a batch's status.** "It compiles" is not done. A batch is done only when its verification command passes and its acceptance criteria are all checked.
- **Never claim performance you have not measured.** Never claim the project beats cuBLAS unless a benchmark in `results/` shows it, with hardware recorded.
- **Never make up numbers, versions, or profiler output.** If you do not know a value, run the command that produces it. If you cannot run it, write `NOT MEASURED` and say why.
- **Do not add features outside scope.** Scope is FP32 GEMM, row-major, no transposes, no Tensor Cores, synchronous public API. Anything else goes in `docs/DESIGN_DECISIONS.md` under "Deferred", not into code.
- **Keep your session output small.** Do not paste whole source files into notes. Reference file paths and line counts.

### If your context is running out

Write a checkpoint immediately, before you run out:

- Update `TASK_BOARD.md` to `IN_PROGRESS` with a one-line "next step" for the batch.
- Write `handoffs/<BATCH_ID>.md` with what is done, what is not, and the exact next command to run.
- Leave the repository in a building state if at all possible. If it cannot build, say so in the handoff note at the top.

---

## 1. Project goal

Build a **performance-first CUDA C++ library** that computes single-precision matrix multiplication:

```
C = alpha * A * B + beta * C
```

Where A is M×K, B is K×N, C is M×N. The library must demonstrate, with evidence:

- Correct design of a GPU compute library (RAII memory, error handling, strided views, clean API boundary).
- Correct GPU kernels at three levels: naive, shared-memory tiled, and further optimized.
- Correctness against a CPU reference, including edge cases.
- Honest, reproducible benchmarks with separate kernel-only and end-to-end timings.
- Profiler evidence (Nsight Systems and Nsight Compute) for every optimization claim.
- Complete documentation so an outside engineer can build, test, and reproduce results.

**What success looks like:** a clean repository that builds with one command, passes all tests under CTest and `compute-sanitizer`, produces machine-readable benchmark CSVs, and has a performance report where every number is traceable to a run.

**What this project is not:** a tutorial, a cuBLAS replacement, or a Tensor Core project. The differentiator is engineering evidence, not kernel count.

---

## 2. Scope (frozen)

| In scope | Out of scope (for this milestone) |
|---|---|
| FP32 (`float`) only | FP16, BF16, FP64, INT8 |
| Row-major storage | Column-major, transposed inputs (`op(A)`) |
| Strided matrix views | Batched GEMM, strided-batched GEMM |
| Synchronous public API | Async / streams in public API |
| CPU reference, naive, tiled, optimized CUDA | Tensor Core (WMMA / MMA) kernels |
| Single GPU, device 0 | Multi-GPU |
| Linux primary, Windows 11 secondary testing | macOS, ROCm, other vendors |
| Catch2 v3 (pinned) + CTest | Google Test, other frameworks |
| cuBLAS as an optional external benchmark reference | Shipping cuBLAS as a dependency of the library |

**Target hardware (owner's machine):** Intel i9-14900K, NVIDIA RTX 4070 Ti SUPER (16 GB VRAM, Ada Lovelace, compute capability 8.9), 64 GB system RAM, Linux primary, Windows 11 secondary.

Phase 0 must verify all of these with real command output. Do not trust this table without checking.

---

## 3. Repository layout (frozen, with amendments)

The layout below is the frozen target. Section 3.2 lists the amendments that were required when this document was finalized. Amendments are binding; treat them as part of the frozen layout.

### 3.1 Target tree

```
cuda-matrix-engine/
├── CMakeLists.txt
├── CMakePresets.json
├── README.md
├── LICENSE                       (Batch 0.2, owner decision — see section 9)
├── .gitignore
├── .clang-format
├── HANDOFF.md                    (copy of this file)
├── MASTER_PLAN.md                (Batch 0.3)
├── AGENTS.md                     (Batch 0.3)
├── ARCHITECTURE.md               (Batch 0.3)
├── INTERFACES.md                 (Batch 0.3, mirrors section 4 of this file)
├── QUALITY_STANDARDS.md          (Batch 0.3)
├── TASK_BOARD.md                 (Batch 0.3)
├── docs/
│   ├── ENVIRONMENT.md            (Batch 0.1)
│   ├── ALGORITHMS.md
│   ├── BENCHMARKING.md
│   ├── PROFILING.md
│   ├── PERFORMANCE_REPORT.md
│   └── DESIGN_DECISIONS.md
├── include/cme/                  (PUBLIC — no CUDA headers allowed)
│   ├── matrix.hpp
│   ├── matrix_view.hpp
│   ├── gemm.hpp
│   ├── gemm_config.hpp
│   ├── status.hpp
│   └── version.hpp
├── src/
│   ├── core/
│   │   ├── matrix.cpp
│   │   ├── status.cpp
│   │   ├── gemm_reference.hpp    (AMENDED — internal)
│   │   └── gemm_reference.cpp    (AMENDED — CPU GEMM)
│   ├── cuda/                     (INTERNAL — CUDA allowed here only)
│   │   ├── cuda_utils.hpp        (MOVED from include/cme/)
│   │   ├── cuda_utils.cu
│   │   ├── device_buffer.hpp     (MOVED from include/cme/)
│   │   ├── device_buffer.cu
│   │   ├── gemm_dispatch.hpp     (AMENDED — internal)
│   │   ├── gemm_dispatch.cu
│   │   └── kernels/
│   │       ├── gemm_kernels.hpp  (AMENDED — internal launch declarations)
│   │       ├── gemm_naive.cu
│   │       ├── gemm_tiled.cu
│   │       └── gemm_optimized.cu
│   └── api/
│       └── gemm.cpp              (validation + host orchestration)
├── tests/
│   ├── CMakeLists.txt
│   ├── test_matrix.cpp
│   ├── test_gemm_correctness.cpp
│   ├── test_gemm_edge_cases.cpp
│   ├── test_gemm_algorithms.cpp
│   ├── test_cuda_errors.cpp
│   └── test_helpers.hpp          (AMENDED — shared random fill + tolerance)
├── benchmarks/
│   ├── CMakeLists.txt
│   ├── benchmark_main.cpp
│   ├── benchmark_gemm.cu
│   ├── benchmark_transfer.cu
│   ├── benchmark_config.hpp
│   ├── csv_writer.hpp            (AMENDED)
│   └── csv_writer.cpp
├── examples/
│   ├── CMakeLists.txt            (AMENDED)
│   ├── basic_gemm.cpp
│   ├── algorithm_comparison.cpp
│   └── custom_configuration.cpp
├── scripts/
│   ├── build.sh
│   ├── test.sh
│   ├── benchmark.sh
│   ├── profile.sh
│   └── environment.sh
├── results/
│   ├── .gitkeep
│   └── README.md
├── .github/workflows/build.yml
└── handoffs/                     (one note per batch, created as work happens)
```

### 3.2 Frozen-tree amendments (binding)

1. **`src/core/gemm_reference.{hpp,cpp}`** — the CPU reference implementation. It must not live inside the matrix abstraction.
2. **`src/cuda/gemm_dispatch.hpp`** — internal header declaring the device-pointer entry point used by both the public API and the benchmarks.
3. **`src/cuda/kernels/gemm_kernels.hpp`** — internal header declaring the three kernel launch wrappers. Lets kernels be tested before dispatch exists.
4. **`device_buffer` and `cuda_utils` moved to `src/cuda/`** — they include CUDA headers, so they cannot be public. Public headers must stay CUDA-free.
5. **No `gemm_config.cpp`** — config validation (`validate_config`) is defined in `src/api/gemm.cpp`.
6. **`tests/test_helpers.hpp`** — shared random-fill and tolerance utilities so every test uses the same tolerance rule.
7. **`benchmarks/csv_writer.hpp`** — header for the CSV writer.
8. **`examples/CMakeLists.txt`** — examples need their own target definitions.

If any of these are wrong when you reach them, record it in `docs/DESIGN_DECISIONS.md` and mark `BLOCKED`. Do not improvise a new layout.

---

## 4. Interface contracts (BINDING)

Agents must not invent interfaces. These are the signatures. Implementation details inside a `.cpp` or `.cu` file are free; the signatures below are not.

### 4.1 `include/cme/status.hpp` (PUBLIC)

```cpp
#pragma once
#include <string>

namespace cme {

enum class StatusCode {
    Ok,
    InvalidArgument,          // null data, non-positive dimension, bad enum value
    DimensionMismatch,        // A, B, C shapes incompatible
    UnsupportedConfiguration, // tile_size not in {16, 32}, aliasing, empty matrices
    OutOfMemory,              // device allocation failed
    CudaError                 // any other CUDA runtime or launch error
};

class [[nodiscard]] Status {
public:
    static Status ok();
    static Status error(StatusCode code, std::string message);

    bool is_ok() const noexcept;
    StatusCode code() const noexcept;
    const std::string& message() const noexcept;

private:
    Status(StatusCode code, std::string message);
    StatusCode code_;
    std::string message_;
};

}  // namespace cme
```

### 4.2 `include/cme/matrix_view.hpp` (PUBLIC)

```cpp
#pragma once
#include <cstddef>

namespace cme {

// Non-owning, row-major view. Element (r, c) lives at data[r * stride + c].
// Requirements for a valid view: data != nullptr, rows > 0, cols > 0, stride >= cols.
template <typename T>
struct MatrixView {
    T* data = nullptr;
    int rows = 0;
    int cols = 0;
    int stride = 0;

    T& operator()(int r, int c) const {
        return data[static_cast<std::size_t>(r) * static_cast<std::size_t>(stride) +
                    static_cast<std::size_t>(c)];
    }

    bool is_valid() const noexcept {
        return data != nullptr && rows > 0 && cols > 0 && stride >= cols;
    }
};

}  // namespace cme
```

Required: `MatrixView<float>` must implicitly convert to `MatrixView<const float>`. Add a converting constructor or conversion operator. Verify with a compile-time test in `test_matrix.cpp`.

### 4.3 `include/cme/matrix.hpp` (PUBLIC — host-side owning matrix)

```cpp
#pragma once
#include <cstddef>
#include <vector>
#include "cme/matrix_view.hpp"

namespace cme {

// Owning, row-major, tightly packed host matrix (stride == cols).
// Constructor throws std::invalid_argument if rows <= 0 or cols <= 0.
// Elements are zero-initialized.
class Matrix {
public:
    Matrix(int rows, int cols);

    int rows() const noexcept;
    int cols() const noexcept;
    int stride() const noexcept;   // always == cols

    float& operator()(int r, int c);
    const float& operator()(int r, int c) const;

    MatrixView<float> view() noexcept;
    MatrixView<const float> view() const noexcept;

    float* data() noexcept;
    const float* data() const noexcept;

private:
    int rows_;
    int cols_;
    std::vector<float> storage_;
};

}  // namespace cme
```

Design note: the constructor throws because `Matrix` is a host container, not an API boundary. The GEMM API itself returns `Status`. Record this rationale in `docs/DESIGN_DECISIONS.md`.

### 4.4 `include/cme/gemm_config.hpp` (PUBLIC)

```cpp
#pragma once

namespace cme {

enum class GemmAlgorithm {
    CPU_REFERENCE,
    CUDA_NAIVE,
    CUDA_TILED,
    CUDA_OPTIMIZED
};

struct GemmConfig {
    GemmAlgorithm algorithm = GemmAlgorithm::CUDA_TILED;
    int tile_size = 16;   // Must be 16 or 32 for CUDA_TILED and CUDA_OPTIMIZED. Ignored otherwise.
};

}  // namespace cme
```

### 4.5 `include/cme/gemm.hpp` (PUBLIC)

```cpp
#pragma once
#include "cme/gemm_config.hpp"
#include "cme/matrix_view.hpp"
#include "cme/status.hpp"

namespace cme {

// Computes C = alpha * A * B + beta * C.
//
// Shape rules:
//   A is M x K, B is K x N, C is M x N.
//   A.rows == C.rows, B.cols == C.cols, A.cols == B.rows.
//
// Semantics:
//   - When beta == 0, C is NOT read. Its prior contents (including NaN/Inf) are ignored.
//   - When alpha == 0, the result is C = beta * C (A and B are still validated).
//   - A, B, and C must not overlap in memory. Overlap is reported as UnsupportedConfiguration.
//   - Empty matrices (any dimension 0) are not supported and return InvalidArgument.
//
// Errors are returned via Status, never thrown.
Status gemm(MatrixView<const float> A,
            MatrixView<const float> B,
            MatrixView<float> C,
            float alpha,
            float beta,
            const GemmConfig& config);

// Validates a configuration without running anything.
// Returns InvalidArgument for an unknown algorithm value,
// UnsupportedConfiguration for a tile_size not in {16, 32} when the algorithm uses tiles.
Status validate_config(const GemmConfig& config);

}  // namespace cme
```

### 4.6 `include/cme/version.hpp` (PUBLIC)

```cpp
#pragma once
namespace cme {
inline constexpr int kVersionMajor = 0;
inline constexpr int kVersionMinor = 1;
inline constexpr int kVersionPatch = 0;
}  // namespace cme
```

### 4.7 `src/core/gemm_reference.hpp` (INTERNAL)

```cpp
#pragma once
#include "cme/matrix_view.hpp"
#include "cme/status.hpp"

namespace cme::detail {

// CPU reference. Accumulates each dot product in double, then casts to float.
// Assumes arguments already validated by the public API.
void gemm_cpu_reference(MatrixView<const float> A,
                        MatrixView<const float> B,
                        MatrixView<float> C,
                        float alpha,
                        float beta);

}  // namespace cme::detail
```

### 4.8 `src/cuda/cuda_utils.hpp` (INTERNAL, CUDA allowed)

```cpp
#pragma once
#include <cuda_runtime.h>
#include "cme/status.hpp"

namespace cme::detail {

// Converts a cudaError_t to a Status. Returns Ok on cudaSuccess.
// The message includes cudaGetErrorName and cudaGetErrorString.
Status to_status(cudaError_t err, const char* context);

// Checks the last error after a kernel launch. Returns CudaError on failure.
Status check_last_launch(const char* context);

}  // namespace cme::detail

// Use inside functions that return cme::Status:
//   CME_RETURN_IF_CUDA_ERROR(cudaMalloc(&p, n), "allocate A");
#define CME_RETURN_IF_CUDA_ERROR(expr, context)                  \
    do {                                                         \
        ::cme::Status _cme_st = ::cme::detail::to_status((expr), (context)); \
        if (!_cme_st.is_ok()) return _cme_st;                    \
    } while (0)
```

### 4.9 `src/cuda/device_buffer.hpp` (INTERNAL, CUDA allowed)

```cpp
#pragma once
#include <cstddef>
#include "cme/status.hpp"

namespace cme::detail {

// RAII owner of a float device allocation. Move-only. Never copyable.
// Destructor frees memory and ignores errors (cudaFree failures are not recoverable here).
class DeviceBuffer {
public:
    DeviceBuffer() noexcept;
    ~DeviceBuffer();
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Allocates `count` floats. count == 0 returns InvalidArgument.
    // On failure, *out is left unchanged and an error Status is returned.
    static Status create(std::size_t count, DeviceBuffer* out);

    Status copy_from_host(const float* host, std::size_t count);  // count <= size()
    Status copy_to_host(float* host, std::size_t count) const;    // count <= size()

    float* data() noexcept;
    const float* data() const noexcept;
    std::size_t size() const noexcept;

private:
    float* ptr_;
    std::size_t size_;
};

}  // namespace cme::detail
```

### 4.10 `src/cuda/kernels/gemm_kernels.hpp` (INTERNAL)

```cpp
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace cme::detail {

// Device-pointer arguments. All matrices row-major. Pointers are device pointers.
// Arguments are assumed valid. Validation happens in the caller.
struct DeviceGemmArgs {
    int M = 0;
    int N = 0;
    int K = 0;
    const float* A = nullptr;  // M x K
    int lda = 0;               // >= K
    const float* B = nullptr;  // K x N
    int ldb = 0;               // >= N
    float* C = nullptr;        // M x N
    int ldc = 0;               // >= N
    float alpha = 1.0f;
    float beta = 0.0f;
    int tile_size = 16;        // used only by tiled and optimized kernels
};

// Each launcher enqueues one kernel on `stream` and returns the launch status.
// It does NOT synchronize. The caller synchronizes and checks errors.
// Semantics must be identical across all three kernels.
cudaError_t launch_gemm_naive(const DeviceGemmArgs& args, cudaStream_t stream);
cudaError_t launch_gemm_tiled(const DeviceGemmArgs& args, cudaStream_t stream);
cudaError_t launch_gemm_optimized(const DeviceGemmArgs& args, cudaStream_t stream);

}  // namespace cme::detail
```

### 4.11 `src/cuda/gemm_dispatch.hpp` (INTERNAL)

```cpp
#pragma once
#include <cuda_runtime.h>
#include "cme/gemm_config.hpp"
#include "cme/status.hpp"
#include "src/cuda/kernels/gemm_kernels.hpp"

namespace cme::detail {

// Selects the kernel for `algorithm` and enqueues it on `stream`.
// Does NOT synchronize. Returns CudaError on launch failure.
// Returns UnsupportedConfiguration if algorithm == CPU_REFERENCE (caller must not route here).
Status gemm_device(GemmAlgorithm algorithm,
                   const DeviceGemmArgs& args,
                   cudaStream_t stream);

}  // namespace cme::detail
```

Used by: the public host API (`src/api/gemm.cpp`) and the kernel-only benchmarks. This is the reason it exists as a separate internal entry point.

### 4.12 `tests/test_helpers.hpp` (TEST-ONLY)

```cpp
#pragma once
#include <cstdint>
#include "cme/matrix.hpp"

namespace cme::test {

// Fills with uniform values in [-1, 1] from a deterministic seed.
void fill_random(Matrix& m, std::uint32_t seed);

// Tolerance rule used by ALL correctness tests (see QUALITY_STANDARDS.md):
//   |got - ref| <= atol_for(K) + rtol * |ref|
//   rtol = 1e-4,  atol_for(K) = 1e-5 * K
bool close_enough(float got, float ref, int K);

}  // namespace cme::test
```

### 4.13 Benchmark interfaces (INTERNAL)

```cpp
// benchmarks/benchmark_config.hpp
struct BenchmarkConfig {
    int warmup_iterations = 5;
    int measured_iterations = 30;
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096};  // square M = N = K
};

// benchmarks/csv_writer.hpp
class CsvWriter {
public:
    explicit CsvWriter(const std::string& path);   // writes header on construction
    void write_row(const std::vector<std::string>& fields);
};
```

Benchmark CSV columns (exact order, do not change):

```
timestamp_utc,algorithm,M,N,K,tile_size,measurement,iterations,median_ms,min_ms,max_ms,iqr_ms,gflops_median,gpu_name,driver_version,cuda_runtime_version,sm_clock_mhz_median,temperature_c_median,power_w_median
```

`measurement` is one of: `kernel_only`, `end_to_end`, `h2d_transfer`, `d2h_transfer`.

---

## 5. Architecture and dependency rules

### 5.1 Dependency graph (binding)

```
examples  ─┐
benchmarks ┼──► public API (include/cme/gemm.hpp, matrix.hpp, ...)
tests      ─┘          │
                       ▼
             src/api/gemm.cpp  (validation + host orchestration)
                 │             │
                 ▼             ▼
   src/core/gemm_reference   src/cuda/gemm_dispatch  ──► kernels/*.cu
                                   │
                                   ▼
                       src/cuda/device_buffer, cuda_utils
```

### 5.2 Rules

- **Examples, benchmarks, and tests may include only `include/cme/*` headers** — with one exception: the kernel-only benchmark may include `src/cuda/gemm_dispatch.hpp` and `src/cuda/device_buffer.hpp` to measure device-only time. Record this exception in `docs/BENCHMARKING.md`.
- **Kernels never call the public API.** They are called only through `gemm_dispatch`.
- **Only `src/cuda/` and `benchmarks/*.cu` may include CUDA headers.** `include/cme/` must never.
- **`src/api/gemm.cpp` is the only place that performs host-to-device orchestration** for the public API.

### 5.3 Public `gemm()` execution flow

For `algorithm == CPU_REFERENCE`:
1. Validate arguments (shapes, validity, overlap, config).
2. Call `cme::detail::gemm_cpu_reference`.
3. Return Ok.

For any CUDA algorithm:
1. Validate arguments.
2. Allocate device buffers for A, B, and C (`DeviceBuffer::create`).
3. Copy A and B host-to-device. Copy C host-to-device only if `beta != 0`.
4. Build `DeviceGemmArgs` with leading dimensions equal to the views' strides.
5. Call `gemm_device(algorithm, args, stream = 0)`.
6. Synchronize the device and check for errors.
7. Copy C device-to-host.
8. Return Ok.

Note: the host matrix layout uses `stride`, but device buffers are allocated tightly packed. Copy with care: copy each row individually (`cudaMemcpy2D`) when `stride != cols`. Record the chosen approach in `docs/DESIGN_DECISIONS.md`. Prefer `cudaMemcpy2D` in `DeviceBuffer` helpers; document the choice.

---

## 6. Phased execution plan

Each phase has batches. A batch is the unit of work for one agent session. Batch IDs are permanent. Do not renumber them.

**Roles** (any agent can take any role, but record which role you used in the handoff note):

| Role | Responsibility |
|---|---|
| **ARCH** | Interface validation, control docs, integration oversight |
| **CORE** | Matrix, views, status, CPU reference |
| **RUNTIME** | CUDA utilities, device buffer |
| **KERNEL** | GEMM kernels and kernel performance |
| **API** | Public API, dispatch, examples |
| **TEST** | Correctness, edge cases, sanitizer runs |
| **BENCH** | Benchmarks, CSV, reports |
| **OPT** | Profiling and optimization (strongest model recommended) |
| **INTEG** | CI, final review, publication |
| **VERIFY** | Independent verification. Moves batches to `VERIFIED`. Must not be the agent that wrote the batch. |

A batch is `VERIFIED` only when a **different agent** (VERIFY role) has run the verification command and confirmed the acceptance criteria. If no separate verifier is available, the owner verifies manually.

---

### PHASE 0 — Environment and control plane

**Goal:** a repository that configures cleanly and records the real environment.

#### Batch 0.1 — Record the environment (ARCH)

**Depends on:** nothing.

**Files to create:** `docs/ENVIRONMENT.md`

**Steps:**
1. Run and record the output of each command (summaries are fine, but include versions):
   - `uname -a`
   - `nvidia-smi` (driver version, GPU name, CUDA version reported by driver)
   - `nvcc --version` (toolkit version)
   - `cmake --version`
   - `g++ --version` or `clang++ --version`
   - `ncu --version` and `nsys --version` (Nsight Compute and Systems), or `NOT FOUND`
   - `python3 --version` (for scripts)
2. Record the GPU compute capability. Expected value for RTX 4070 Ti SUPER is **8.9 (sm_89)**. Verify with `nvidia-smi --query-gpu=compute_cap --format=csv` or a deviceQuery-style program. Do not assume.
3. Note explicitly that the **driver's supported CUDA version and the installed `nvcc` toolkit version are different things.** Record both separately.
4. Record the Windows 11 environment separately (if available) in a clearly labeled section. If not available, write `NOT AVAILABLE`.

**Acceptance criteria:**
- Every tool version is recorded from real output, or marked `NOT FOUND`.
- Compute capability is recorded from a command, not assumed.
- Driver and toolkit are listed as separate lines.

**Verification:** `cat docs/ENVIRONMENT.md` shows all sections filled.

**Done when:** the file exists with no empty fields.

---

#### Batch 0.2 — Build system and repo skeleton (ARCH)

**Depends on:** 0.1.

**Files to create:**
- `CMakeLists.txt`
- `CMakePresets.json`
- `.gitignore`
- `.clang-format`
- `README.md` (placeholder: title, one-paragraph description, "under construction")
- `LICENSE` — **do not create until the owner chooses a license (see section 9, open decision D1).** Mark the batch with this item open if not decided.
- All empty directories with `.gitkeep` where needed: `include/cme/`, `src/core/`, `src/cuda/kernels/`, `src/api/`, `tests/`, `benchmarks/`, `examples/`, `scripts/`, `results/`, `docs/`, `handoffs/`, `.github/workflows/`

**Requirements for `CMakeLists.txt`:**
- `cmake_minimum_required(VERSION 3.24)` (CUDA language support and `CMAKE_CUDA_ARCHITECTURES` handling)
- `project(cuda-matrix-engine VERSION 0.1.0 LANGUAGES CXX CUDA)`
- `set(CMAKE_CXX_STANDARD 17)`, `CMAKE_CXX_STANDARD_REQUIRED ON`
- `set(CMAKE_CUDA_STANDARD 17)`
- Options:
  - `CME_BUILD_TESTS` (default ON)
  - `CME_BUILD_BENCHMARKS` (default ON)
  - `CME_BUILD_EXAMPLES` (default ON)
  - `CME_CUDA_ARCHITECTURES` (cache string, default `89`)
  - `CME_WITH_CUBLAS` (default OFF — used only by benchmarks)
- Set `CMAKE_CUDA_ARCHITECTURES` from `CME_CUDA_ARCHITECTURES`.
- Define a static library target `cme` with sources from `src/core/`, `src/cuda/`, `src/cuda/kernels/`, `src/api/`. **Initially, list only the files that exist.** Add files as batches create them.
- Public include directory for `cme`: `include/`. Private: `src/`.
- `CUDAToolkit` found via `find_package(CUDAToolkit REQUIRED)`.
- Subdirectories added conditionally on the options (`tests/`, `benchmarks/`, `examples/`). Create stub `CMakeLists.txt` in each that is empty or contains a comment until their batches run.
- Catch2 is **not** added in this batch. It is added in Batch 1.4 (see below).

**Requirements for `CMakePresets.json`:**
- Version 3 schema.
- Configure presets: `linux-debug` (Debug, tests on), `linux-release` (Release), `windows-release` (Release, Visual Studio generator or Ninja — record which).
- Each preset sets `CME_CUDA_ARCHITECTURES` from the cache default.
- Each preset has a matching build preset and test preset (`ctest`).

**Requirements for `.clang-format`:**
- Based on LLVM style, 4-space indent, column limit 100. Record in `docs/DESIGN_DECISIONS.md`.

**Acceptance criteria:**
- `cmake --preset linux-debug` configures without errors.
- `cmake --build --preset linux-debug` succeeds with an empty library (or with only the sources that exist).
- No file references a source that does not exist.

**Verification:**
```
cmake --preset linux-debug
cmake --build --preset linux-debug
```

**Done when:** both commands exit 0.

---

#### Batch 0.3 — Control documents (ARCH)

**Depends on:** 0.2.

**Files to create:**
- `MASTER_PLAN.md` — copy the phase list from section 6 of this file, with batch IDs.
- `AGENTS.md` — the rules from section 0 of this file, verbatim, plus: "If you are a free-tier or limited-context agent, do one batch per session."
- `ARCHITECTURE.md` — section 5 of this file, verbatim.
- `INTERFACES.md` — section 4 of this file, verbatim.
- `QUALITY_STANDARDS.md` — section 11 (Definition of Done) and the tolerance rule from section 4.12, plus the benchmark rules from section 7.
- `TASK_BOARD.md` — table of every batch ID in section 6, with columns: `ID | Phase | Role | Status | Agent | Date | Depends on | Notes`. All statuses start as `NOT_STARTED` except 0.1, which is `NOT_STARTED` too.
- `docs/ALGORITHMS.md`, `docs/BENCHMARKING.md`, `docs/PROFILING.md`, `docs/PERFORMANCE_REPORT.md`, `docs/DESIGN_DECISIONS.md` — stubs with a title and "TBD by batch X.Y".
- `handoffs/README.md` — explains the handoff note template in section 8.

**Status values for TASK_BOARD (use exactly these):**
`NOT_STARTED` → `IN_PROGRESS` → `UNDER_REVIEW` → `VERIFIED`
Side states: `BLOCKED` (with reason in Notes).

**Acceptance criteria:**
- All seven control files exist.
- `TASK_BOARD.md` lists every batch from section 6, in order.
- `INTERFACES.md` matches section 4 exactly. Diff the two to confirm.

**Verification:** `grep -c "^Batch\|^| [0-9]" TASK_BOARD.md` equals the number of batches in section 6 (see section 6 summary table).

**Done when:** the verification passes and the files are committed.

**Phase 0 gate:** 0.1, 0.2, 0.3 are `VERIFIED`. Then Phase 1 may start.

---

### PHASE 1 — Core foundation (host only, no CUDA)

**Goal:** a host-side matrix abstraction, status type, and CPU reference, all tested. **No CUDA headers in this phase.**

#### Batch 1.1 — Status (CORE)

**Depends on:** 0.3.

**Files:** `include/cme/status.hpp`, `src/core/status.cpp`.

**Requirements:**
- Exactly the interface in section 4.1.
- `Status::ok()` returns an `Ok` status with empty message.
- `Status::error` must not accept `StatusCode::Ok` — if passed, return an `InvalidArgument` status with message `"Status::error called with Ok"`.

**Acceptance criteria:** all methods implemented; `[[nodiscard]]` present.

**Verification:** `cmake --build --preset linux-debug` succeeds.

**Done when:** build passes. (Tests for status are in Batch 1.4.)

---

#### Batch 1.2 — Matrix view and matrix (CORE)

**Depends on:** 1.1.

**Files:** `include/cme/matrix_view.hpp`, `include/cme/matrix.hpp`, `src/core/matrix.cpp`, `include/cme/version.hpp`.

**Requirements:**
- Exactly the interfaces in sections 4.2, 4.3, 4.6.
- `Matrix` constructor throws `std::invalid_argument` for non-positive dimensions. Message names the dimensions.
- `Matrix` storage is `std::vector<float>` of size `rows*cols`, zero-initialized.
- `view()` returns a view with `stride == cols`.
- Implicit conversion `MatrixView<float>` → `MatrixView<const float>` works.

**Acceptance criteria:** all of the above; no CUDA header included.

**Verification:** `grep -r "cuda" include/ src/core/` returns no matches.

**Done when:** build passes and the grep is empty.

---

#### Batch 1.3 — Configuration types (CORE)

**Depends on:** 1.1.

**Files:** `include/cme/gemm_config.hpp`.

**Requirements:** exactly section 4.4. Default algorithm `CUDA_TILED`, default tile_size `16`.

**Done when:** build passes.

---

#### Batch 1.4 — CPU reference and test harness (CORE → TEST)

**Depends on:** 1.2.

**Files:**
- `src/core/gemm_reference.hpp`, `src/core/gemm_reference.cpp`
- `tests/CMakeLists.txt` (adds Catch2 — see requirements)
- `tests/test_helpers.hpp`
- `tests/test_matrix.cpp`
- `tests/test_status.cpp` (new file, not in original tree — record in DESIGN_DECISIONS)
- `tests/test_gemm_reference.cpp` (new file, not in original tree — record in DESIGN_DECISIONS)

**Requirements for `gemm_cpu_reference`:**
- Implements `C = alpha*A*B + beta*C` exactly per section 4.7.
- Accumulates each dot product in `double`, then computes `alpha * static_cast<float>(acc) + beta * C(i,j)`.
- When `beta == 0`, does not read C.
- Triple loop is fine. Do not optimize.

**Requirements for `tests/CMakeLists.txt`:**
- Add Catch2 v3 via `FetchContent`, pinned to a **specific tag**. Record the tag in `docs/DESIGN_DECISIONS.md` and `QUALITY_STANDARDS.md`. Pick the most recent v3 release available at the time of the batch; do not use a branch name.
- Use `catch_discover_tests` with CTest.
- Each test source is a separate executable or one executable per file — choose one and document it. Recommended: one executable `cme_tests` with all test files.

**Tests to write in this batch:**
- `test_status`: `ok()` is ok; `error(Ok, ...)` returns InvalidArgument; message retained.
- `test_matrix`: construction, zero-init, `operator()` indexing row-major, `view()` stride equals cols, conversion to const view, throws on 0 rows, throws on negative cols.
- `test_gemm_reference`: 2x2 hand-computed case; `beta == 0` ignores NaN in C; `alpha == 0` yields `beta*C`; a 3x2 times 2x4 case computed by hand.

**Acceptance criteria:** all tests above pass; Catch2 version pinned to a tag.

**Verification:**
```
cmake --build --preset linux-debug
ctest --preset linux-debug --output-on-failure
```

**Done when:** ctest reports all tests passed, 0 failed.

**Phase 1 gate:** 1.1–1.4 are `VERIFIED`. Public API headers contain no CUDA. The CPU reference is tested.

---

### PHASE 2 — CUDA runtime

**Goal:** reliable device memory and CUDA error handling. Depends on CORE (status type) only.

#### Batch 2.1 — CUDA utilities (RUNTIME)

**Depends on:** 1.1, 0.2 (CUDA must be enabled in CMake).

**Files:** `src/cuda/cuda_utils.hpp`, `src/cuda/cuda_utils.cu`.

**Requirements:** exactly section 4.8.

- `to_status(cudaSuccess, ...)` returns Ok.
- Any other error maps to `CudaError` with message: `"<context>: <cudaGetErrorName> (<cudaGetErrorString>)"`.
- `check_last_launch` calls `cudaGetLastError()` and maps via `to_status`. Do not call `cudaDeviceSynchronize` here — that is the caller's job.
- Add `cme` target sources for these two files.

**Done when:** build passes.

---

#### Batch 2.2 — Device buffer (RUNTIME)

**Depends on:** 2.1.

**Files:** `src/cuda/device_buffer.hpp`, `src/cuda/device_buffer.cu`.

**Requirements:** exactly section 4.9.

- `create(count, out)`:
  - `count == 0` → `InvalidArgument`.
  - `cudaMalloc` failure with `cudaErrorMemoryAllocation` → `OutOfMemory`; other errors → `CudaError`.
  - On success, assigns `*out` (moving the new buffer in) and leaves `*out` unchanged on failure.
- `copy_from_host` and `copy_to_host`:
  - `count > size()` → `InvalidArgument`.
  - `count == 0` → Ok (no-op).
  - Use `cudaMemcpy` (synchronous). Record in DESIGN_DECISIONS.
- Destructor calls `cudaFree` if `ptr_ != nullptr`; ignores the return value.
- Move constructor and move assignment null out the source.
- Add a helper `copy_from_host_2d(const float* host, int rows, int cols, int host_stride)` **only if** the API batch needs it. Do not add it speculatively. Decide in Batch 4.2 and amend this batch if needed.

**Tests:** Write `tests/test_cuda_errors.cpp` with:
- Round-trip copy: allocate 1024 floats, copy from a host vector, copy back, compare exactly.
- `create(0, ...)` returns InvalidArgument.
- Move semantics: moved-from buffer has size 0 and null data.
- `copy_from_host` with count > size returns InvalidArgument.

These tests need a GPU. If no CUDA device is present, the test must **skip** with a clear message rather than fail. Use Catch2 `SKIP`. Record this policy in QUALITY_STANDARDS.md.

**Verification:**
```
cmake --build --preset linux-debug
ctest --preset linux-debug --output-on-failure
```

**Done when:** tests pass on the owner's GPU (not skipped), and the build is clean.

**Phase 2 gate:** 2.1 and 2.2 are `VERIFIED` and the device tests actually ran (not skipped) on real hardware. Record the test count from ctest.

---

### PHASE 3 — GEMM kernels

**Goal:** three kernels that all produce correct results. Each kernel is tested directly through its launcher (`gemm_kernels.hpp`), not through the public API.

Kernel semantics (identical across all three, binding):
- `C[i][j] = alpha * sum_k(A[i][k] * B[k][j]) + beta * C[i][j]`
- When `beta == 0`, do not read C.
- Accumulate in `float`. (The CPU reference uses `double`; tolerance rule in section 4.12 accounts for this.)
- Row-major with leading dimensions `lda`, `ldb`, `ldc`.

#### Batch 3.1 — Naive kernel (KERNEL)

**Depends on:** 2.2.

**Files:** `src/cuda/kernels/gemm_kernels.hpp`, `src/cuda/kernels/gemm_naive.cu`.

**Requirements:**
- One thread computes one element of C.
- 2D grid and block. Block size 16×16 is acceptable. Record the choice.
- Bounds check `row < M && col < N`.
- Read A and B directly from global memory. No shared memory.
- Launch wrapper computes grid dims with ceil-division and returns `cudaGetLastError()`-style launch status. Does not synchronize.

**Tests:** Write `tests/test_gemm_algorithms.cpp` section for naive only, for now. Compare against `gemm_cpu_reference` on shapes: (1,1,1), (2,2,2), (17,33,5), (64,64,64).

Test harness for kernels: allocate host matrices with `cme::Matrix`, copy to `DeviceBuffer`, call launcher with a `DeviceGemmArgs`, synchronize, check `cudaGetLastError`, copy C back, compare with `close_enough`. Put this harness in `tests/test_helpers.hpp` or a new `tests/kernel_test_util.hpp` — record which.

**Done when:** naive kernel passes all shapes with `compute-sanitizer --tool memcheck` clean on at least the 17×33×5 case.

---

#### Batch 3.2 — Tiled kernel (KERNEL)

**Depends on:** 3.1 (for the test harness).

**Files:** `src/cuda/kernels/gemm_tiled.cu`.

**Requirements:**
- Shared-memory tiling. Supported tile sizes: **16 and 32** only. Implement as a template on `TILE` and dispatch from the launcher with a `switch` on `args.tile_size`. Any other value returns `cudaErrorInvalidValue` from the launcher (the public API will have rejected it earlier).
- Both A and B tiles loaded into shared memory with bounds checks; out-of-range elements loaded as `0.0f`.
- `__syncthreads()` placed correctly before and after the inner product.
- Block dimensions equal `TILE × TILE`.
- Apply `alpha` and `beta` in the final write. Respect `beta == 0` (do not read C).

**Tests:** Add tiled cases to `test_gemm_algorithms.cpp`: for tile sizes 16 and 32, shapes (1,1,1), (17,33,5), (64,64,64), (127,129,131), (256,256,256).

**Done when:** all pass; memcheck clean on (127,129,131) with tile 16 and tile 32; racecheck clean on (64,64,64) tile 16 (`compute-sanitizer --tool racecheck`).

---

#### Batch 3.3 — Optimized kernel, first candidate (KERNEL)

**Depends on:** 3.2.

**Files:** `src/cuda/kernels/gemm_optimized.cu`.

**Important:** Phase 7 of this file requires that the optimized kernel **earn its place through measurement**. This batch only creates a correct candidate. Whether it stays in the default dispatch is decided in Phase 7.

**Requirements (candidate design — KERNEL may choose a better one, but must document it in DESIGN_DECISIONS):**
- Start from the tiled kernel. Add one or more of: each thread computes a small register micro-tile (for example 2×2 or 4×4 outputs), loads of A and B into registers before the inner loop, vectorized `float4` global loads where alignment and bounds allow, `#pragma unroll` on the inner loop.
- Must support `tile_size` 16 and 32 under the same rules as tiled.
- Must produce identical semantics and pass the same correctness tests.
- **Do not claim it is faster in this batch.** Speed is not measured yet.

**Done when:** correctness tests pass for all shapes and tile sizes, memcheck clean on one non-multiple-of-tile shape, and the design is written in `docs/ALGORITHMS.md` under "Optimized kernel — candidate design" with the reason each technique was chosen.

**Phase 3 gate:** 3.1, 3.2, 3.3 are `VERIFIED`. All three kernels agree with the CPU reference on every test shape.

---

### PHASE 4 — Public API and dispatch

**Goal:** `cme::gemm()` works end-to-end for every algorithm. Applications can call it without knowing kernel internals.

#### Batch 4.1 — Device dispatch (API)

**Depends on:** 3.3.

**Files:** `src/cuda/gemm_dispatch.hpp`, `src/cuda/gemm_dispatch.cu`.

**Requirements:** exactly section 4.11.

- `CPU_REFERENCE` → `UnsupportedConfiguration` with message "CPU_REFERENCE is not a device algorithm".
- `CUDA_NAIVE` → `launch_gemm_naive`. `CUDA_TILED` → `launch_gemm_tiled`. `CUDA_OPTIMIZED` → `launch_gemm_optimized`.
- Launch failure is converted with `to_status` and `check_last_launch` semantics. Does not synchronize.

**Done when:** build passes; a test calls `gemm_device` directly for each device algorithm and gets `Ok` on valid input.

---

#### Batch 4.2 — Public host API (API)

**Depends on:** 4.1, 2.2, 1.4.

**Files:** `src/api/gemm.cpp`, `include/cme/gemm.hpp`.

**Requirements:**
- Implement section 4.5 exactly.
- **Validation order (binding):**
  1. Views valid (`is_valid()` for A, B, C). Failure → `InvalidArgument`.
  2. Shapes compatible. Failure → `DimensionMismatch`.
  3. Config valid (`validate_config`). Failure → as returned.
  4. Overlap check. Compute address ranges of A, B, C (using `stride`, rows, cols, element size). If any two overlap → `UnsupportedConfiguration`.
- `validate_config`:
  - Unknown enum value → `InvalidArgument`.
  - For `CUDA_TILED` and `CUDA_OPTIMIZED`: `tile_size` not in {16, 32} → `UnsupportedConfiguration`.
  - For other algorithms: `tile_size` ignored.
- `CPU_REFERENCE`: call `gemm_cpu_reference`, return Ok.
- CUDA algorithms: follow the flow in section 5.3.
  - Host-to-device copy of a strided host matrix: use **row-by-row** copy into a tightly packed staging vector, or `cudaMemcpy2D`. Record the choice in DESIGN_DECISIONS. Prefer `cudaMemcpy2D` via a new `DeviceBuffer` helper, which requires amending Batch 2.2 — do so and note it in the handoff.
  - Copy C device-to-host with the same strided approach, so the caller's `C.stride` is respected.
- Every CUDA error maps to a `Status` without throwing.

**Tests:** Add `tests/test_gemm_edge_cases.cpp` and `tests/test_gemm_correctness.cpp` in Phase 5. In this batch, write only a smoke test: one call per algorithm on a 2×2 case.

**Done when:** build passes and smoke test passes for all four algorithms.

---

#### Batch 4.3 — Examples (API)

**Depends on:** 4.2.

**Files:** `examples/CMakeLists.txt`, `examples/basic_gemm.cpp`, `examples/algorithm_comparison.cpp`, `examples/custom_configuration.cpp`.

**Requirements:**
- Examples link only to `cme` and include only `cme/*.hpp`.
- `basic_gemm`: 128×64 times 64×32, prints the first 4 elements of the result.
- `algorithm_comparison`: runs the same input through all four algorithms and prints the max absolute difference from `CPU_REFERENCE`.
- `custom_configuration`: demonstrates a strided view (a sub-matrix of a larger `Matrix`) and a non-default `tile_size`.
- Each example returns non-zero on any `Status` error.

**Verification:** each example runs and exits 0.

**Done when:** all three examples run and print results on the owner's machine.

**Phase 4 gate:** 4.1–4.3 are `VERIFIED`. Public API works for all algorithms. `include/cme/` still contains no CUDA headers (re-run the grep from Batch 1.2).

---

### PHASE 5 — Testing

**Goal:** reproducible correctness evidence, including edge cases and memory safety.

#### Batch 5.1 — Correctness matrix (TEST)

**Depends on:** 4.2.

**Files:** `tests/test_gemm_correctness.cpp`.

**Requirements:** for each algorithm in {CUDA_NAIVE, CUDA_TILED (16), CUDA_TILED (32), CUDA_OPTIMIZED (16), CUDA_OPTIMIZED (32)}, compare the public `gemm()` output against `CPU_REFERENCE` for shapes:
- (1,1,1), (1,7,3), (7,1,3), (17,33,5), (64,64,64), (127,129,131), (256,256,256), (513,255,257).
- With `(alpha, beta)` in {(1,0), (1,1), (2.5,-0.75), (0,1), (-1,0.5)} — exactly these five pairs.
- Random inputs from `fill_random` with seeds 1, 2, 3.

Compare with `close_enough(got, ref, K)`.

**Done when:** all combinations pass, with output of the count of comparisons logged.

---

#### Batch 5.2 — Edge cases and invalid input (TEST)

**Depends on:** 5.1.

**Files:** `tests/test_gemm_edge_cases.cpp`, `tests/test_cuda_errors.cpp` (extend).

**Requirements — each must be a test case:**
- Dimension mismatch (A.cols != B.rows) → `DimensionMismatch`.
- Output shape mismatch (C.rows != A.rows) → `DimensionMismatch`.
- Invalid view (null data) → `InvalidArgument`.
- Zero-dimension view → `InvalidArgument`.
- `beta == 0` with NaN prefilled in C → result is finite and correct. (The NaN must not propagate.)
- `alpha == 0` → C equals `beta * C_original`.
- Non-contiguous stride: A, B, C are sub-views of larger matrices with `stride > cols`. Result matches reference.
- Overlapping C and A → `UnsupportedConfiguration`.
- `tile_size = 8` with CUDA_TILED → `UnsupportedConfiguration`.
- `tile_size = 33` with CUDA_OPTIMIZED → `UnsupportedConfiguration`.
- Invalid enum cast (`static_cast<GemmAlgorithm>(99)`) → `InvalidArgument`.
- Large K (e.g. K = 4096, M = N = 1) — tolerance rule still applies.

**Done when:** every case passes; no test is skipped except when no GPU exists.

---

#### Batch 5.3 — Sanitizer and CI-style test script (TEST)

**Depends on:** 5.2.

**Files:** `scripts/test.sh`, `scripts/environment.sh`.

**Requirements:**
- `scripts/environment.sh`: prints the environment in the same format as `docs/ENVIRONMENT.md`. Must not fail if a tool is missing; print `NOT FOUND`.
- `scripts/test.sh`:
  1. Configure and build with `linux-debug`.
  2. Run `ctest --preset linux-debug`.
  3. Run `compute-sanitizer --tool memcheck` on the test binary, with `--error-exitcode 1`.
  4. Run `compute-sanitizer --tool racecheck` on the test binary (restricted to kernel tests if the whole suite is too slow — document the filter).
  5. Exit non-zero on any failure.

**Done when:** `scripts/test.sh` exits 0 on the owner's machine. Record the full output summary in `docs/PERFORMANCE_REPORT.md` under "Correctness evidence" (it is evidence, not performance).

**Phase 5 gate:** 5.1–5.3 are `VERIFIED`. The test script exits 0 on real hardware, with memcheck and racecheck clean.

---

### PHASE 6 — Benchmarking

**Goal:** reproducible, honest timing with CSV output. Separate kernel-only, end-to-end, and transfer measurements.

**Methodology (binding — copy into `docs/BENCHMARKING.md` in Batch 6.4):**
- Warm-up: 5 iterations, not measured.
- Measured: 30 iterations per configuration.
- Kernel time: CUDA events around the launch only, on the same stream, with a `cudaEventSynchronize` before reading elapsed time.
- End-to-end time: host wall clock (`std::chrono::steady_clock`) around the full public `gemm()` call.
- Transfer time: CUDA events around `cudaMemcpy`/`cudaMemcpy2D` only.
- Report **median, min, max, and IQR**. Do not report only the mean.
- GFLOP/s = `2.0 * M * N * K / (median_ms * 1e-3) / 1e9`. Record the formula in the docs.
- Sizes: square 256, 512, 1024, 2048, 4096.
- Record GPU clocks, temperature, and power **before and after** each configuration using `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits`. Report the median of samples taken during the run (sample on a background thread or between iterations; document the choice).
- Record GPU name, driver version, CUDA runtime version in every CSV row.

#### Batch 6.1 — Benchmark utilities (BENCH)

**Depends on:** 4.2.

**Files:** `benchmarks/benchmark_config.hpp`, `benchmarks/csv_writer.hpp`, `benchmarks/csv_writer.cpp`.

**Requirements:** exactly sections 4.13 (`BenchmarkConfig`, `CsvWriter`). CSV columns exactly as specified. `CsvWriter` escapes fields containing commas or quotes per RFC 4180.

**Done when:** a small test program writes a CSV with one row and the header matches section 4.13 byte-for-byte.

---

#### Batch 6.2 — Kernel-only benchmark (BENCH)

**Depends on:** 6.1, 4.1.

**Files:** `benchmarks/benchmark_gemm.cu` (kernel-only function first).

**Requirements:**
- Uses `gemm_device` and `DeviceBuffer` directly (this is the permitted exception in section 5.2). Allocates device buffers once per size, outside the timed region.
- Times only the kernel launch with CUDA events.
- Covers algorithms: CUDA_NAIVE, CUDA_TILED (16, 32), CUDA_OPTIMIZED (16, 32).
- Writes rows with `measurement = kernel_only`.

**Done when:** a run produces a CSV with the expected number of rows (algorithms × sizes).

---

#### Batch 6.3 — End-to-end and transfer benchmarks (BENCH)

**Depends on:** 6.2.

**Files:** `benchmarks/benchmark_gemm.cu` (end-to-end section), `benchmarks/benchmark_transfer.cu`.

**Requirements:**
- End-to-end: wall clock around the public `gemm()` call, for CPU_REFERENCE (for reference only, with a note that it is not a device measurement) and each CUDA algorithm.
- Transfer: H2D of A and B, D2H of C, at each size. Use `measurement = h2d_transfer` and `d2h_transfer`.

**Done when:** CSV contains all four measurement types.

---

#### Batch 6.4 — Benchmark driver and reproducibility (BENCH)

**Depends on:** 6.3.

**Files:** `benchmarks/benchmark_main.cpp`, `benchmarks/CMakeLists.txt`, `scripts/benchmark.sh`, `results/README.md`, `docs/BENCHMARKING.md`.

**Requirements:**
- `benchmark_main` accepts `--output <path>` (default `results/gemm_<timestamp>.csv`) and `--quick` (fewer iterations, sizes 256 and 512 only, for smoke testing).
- `scripts/benchmark.sh` runs the full benchmark in `linux-release` and writes the CSV to `results/`. Also writes a `results/<file>.env.txt` containing the output of `scripts/environment.sh`.
- `results/README.md` explains file naming and how to reproduce.
- `docs/BENCHMARKING.md` contains the methodology above verbatim, plus the GFLOP/s formula and the exception in section 5.2.
- Optional, behind `CME_WITH_CUBLAS`: a cuBLAS `cublasSgemm` row with `algorithm = CUBLAS_REFERENCE`. If the owner has not enabled it, skip it and note in the handoff.

**Done when:** `scripts/benchmark.sh --quick` produces a valid CSV on the owner's machine, and `scripts/benchmark.sh` (full) completes and produces the full CSV.

**Phase 6 gate:** 6.1–6.4 are `VERIFIED`. A full benchmark CSV exists in `results/` from a real run. Record the file name in `docs/PERFORMANCE_REPORT.md`.

---

### PHASE 7 — Optimization and profiling

**Goal:** every optimization claim is backed by a profiler capture and a before/after measurement. Negative results are recorded, not hidden.

**Rule (binding):** an optimization is kept in the default dispatch only if the median kernel-only GFLOP/s improves by **at least 5%** at the sizes where it is claimed, measured across the 30 iterations, and the improvement is larger than the IQR overlap. Otherwise the variant stays in the code, is removed from default dispatch, and the negative result is documented.

#### Batch 7.1 — Baseline profiling (OPT)

**Depends on:** 6.4.

**Files:** `scripts/profile.sh`, `docs/PROFILING.md`.

**Steps:**
1. Run Nsight Systems on the end-to-end benchmark for size 2048 and save the report under `results/profiles/`. Record the command.
2. Run Nsight Compute on each CUDA kernel at size 2048 for the baseline configuration. Save reports. Record key metrics: achieved occupancy, memory throughput, compute throughput, warp stall reasons, shared-memory bank conflicts (if reported).
3. Record each metric in `docs/PROFILING.md` with the exact value from the report.

**Acceptance criteria:** no metric in the document is estimated; every value cites the report file.

**Done when:** the profiles exist and the document has the baseline metrics.

---

#### Batch 7.2 — Optimization iterations (OPT)

**Depends on:** 7.1.

**Files:** `src/cuda/kernels/gemm_optimized.cu` (may be revised), `docs/PROFILING.md`, `docs/ALGORITHMS.md`, `docs/DESIGN_DECISIONS.md`.

**Process — repeat for each hypothesis:**
1. State the hypothesis from the profile evidence (for example "tiled kernel is limited by global-memory load latency per warp, stall reason X is dominant").
2. Make **one** change.
3. Re-run correctness tests (must pass).
4. Re-run kernel-only benchmark for affected sizes.
5. Re-run Nsight Compute on the changed kernel.
6. Record: hypothesis, change, before metrics, after metrics, decision (keep / revert / not in default dispatch), with report file names.
7. Revert if the change does not meet the 5% rule.

**Candidate hypotheses to test (not mandatory — use the profile to decide):**
- Register micro-tiling size (2×2 vs 4×4 vs 8×8).
- Vectorized loads.
- Shared-memory padding to remove bank conflicts.
- Double-buffering shared-memory tiles.
- Occupancy vs register pressure trade-off (check `--maxrregcount` effects by recording, not guessing).

**Acceptance criteria:** every kept change has a full record. No change is kept without a measured improvement.

**Done when:** the optimized kernel either meets the 5% rule at the claimed sizes, or the document records that it does not and it is excluded from the default dispatch.

---

#### Batch 7.3 — Dispatch decision and final benchmark (OPT → VERIFY)

**Depends on:** 7.2.

**Files:** `src/cuda/gemm_dispatch.cu` (only if the optimized kernel is excluded or its selection changes), `results/` (new CSV), `docs/PERFORMANCE_REPORT.md`.

**Steps:**
1. Make the final dispatch decision.
2. Run the full benchmark again (`scripts/benchmark.sh`).
3. Run the full test script (`scripts/test.sh`).
4. Write `docs/PERFORMANCE_REPORT.md`:
   - Hardware and software (from `docs/ENVIRONMENT.md`).
   - Methodology (reference `docs/BENCHMARKING.md`).
   - Table of median kernel-only GFLOP/s per algorithm per size.
   - Table of end-to-end median ms.
   - Table of transfer median ms.
   - Profiler summary (reference `docs/PROFILING.md`).
   - **Comparison against cuBLAS only if the cuBLAS row exists in the CSV.** Otherwise state "not measured".
   - Known limitations.
   - Reproduction command.

**Done when:** the report's numbers match the final CSV exactly. The VERIFY role spot-checks three numbers against the CSV.

**Phase 7 gate:** 7.1–7.3 are `VERIFIED`.

---

### PHASE 8 — Publication

**Goal:** a clean, documented, CI-checked repository that a stranger can build and understand.

#### Batch 8.1 — Documentation (INTEG)

**Depends on:** 7.3.

**Files:** `README.md`, `docs/ARCHITECTURE.md` (if separate from root `ARCHITECTURE.md` — keep one copy only, at the root), `docs/ALGORITHMS.md`, `docs/DESIGN_DECISIONS.md`, `docs/PERFORMANCE_REPORT.md` (final pass).

**Requirements for `README.md`:**
- One-paragraph description.
- Build instructions (`cmake --preset linux-release`).
- Test instructions (`scripts/test.sh`).
- Benchmark instructions (`scripts/benchmark.sh`).
- API usage example (copy from `examples/basic_gemm.cpp`, verified to compile).
- Hardware and environment summary, linked to `docs/ENVIRONMENT.md`.
- Headline result with a link to `docs/PERFORMANCE_REPORT.md`. Do not state a number that is not in that report.
- Known limitations (FP32 only, row-major, no Tensor Cores, synchronous API).

**Done when:** a reader can follow the README from clone to running example.

---

#### Batch 8.2 — Continuous integration (INTEG)

**Depends on:** 8.1.

**Files:** `.github/workflows/build.yml`.

**Requirements:**
- Hosted GitHub runners generally have **no GPU**. CI therefore builds the library, examples, and tests, and runs only the CPU-side tests. GPU tests must skip cleanly when no device is present (established in Batch 2.2).
- Use a CUDA toolkit installation action or container. Pin the version. Record the pin in `docs/ENVIRONMENT.md`.
- CI does **not** claim GPU correctness. The README must say so.

**Done when:** the workflow file is valid YAML, and the build step succeeds locally in a dry run or on a push if the owner has a remote.

---

#### Batch 8.3 — Final review and definition of done (INTEG → VERIFY)

**Depends on:** 8.2.

**Steps:**
1. Walk through the Definition of Done table in `QUALITY_STANDARDS.md` (section 11 of this file). Mark each row with evidence (file path or command output).
2. Confirm every batch in `TASK_BOARD.md` is `VERIFIED`.
3. Confirm the frozen-tree amendments in section 3.2 are all either applied or recorded in `DESIGN_DECISIONS.md`.
4. Confirm no CUDA header is in `include/cme/`.
5. Confirm `INTERFACES.md` still matches section 4 of this file.
6. Confirm no performance claim in any document lacks a CSV or profile reference.

**Done when:** the checklist is complete with evidence. The owner reviews the summary.

**Phase 8 gate:** 8.1–8.3 are `VERIFIED`. The project is complete.

---

## 6a. Batch summary (for TASK_BOARD)

| ID | Phase | Role | Depends on |
|---|---|---|---|
| 0.1 | 0 | ARCH | — |
| 0.2 | 0 | ARCH | 0.1 |
| 0.3 | 0 | ARCH | 0.2 |
| 1.1 | 1 | CORE | 0.3 |
| 1.2 | 1 | CORE | 1.1 |
| 1.3 | 1 | CORE | 1.1 |
| 1.4 | 1 | CORE/TEST | 1.2 |
| 2.1 | 2 | RUNTIME | 1.1, 0.2 |
| 2.2 | 2 | RUNTIME | 2.1 |
| 3.1 | 3 | KERNEL | 2.2 |
| 3.2 | 3 | KERNEL | 3.1 |
| 3.3 | 3 | KERNEL | 3.2 |
| 4.1 | 4 | API | 3.3 |
| 4.2 | 4 | API | 4.1, 2.2, 1.4 |
| 4.3 | 4 | API | 4.2 |
| 5.1 | 5 | TEST | 4.2 |
| 5.2 | 5 | TEST | 5.1 |
| 5.3 | 5 | TEST | 5.2 |
| 6.1 | 6 | BENCH | 4.2 |
| 6.2 | 6 | BENCH | 6.1, 4.1 |
| 6.3 | 6 | BENCH | 6.2 |
| 6.4 | 6 | BENCH | 6.3 |
| 7.1 | 7 | OPT | 6.4 |
| 7.2 | 7 | OPT | 7.1 |
| 7.3 | 7 | OPT/VERIFY | 7.2 |
| 8.1 | 8 | INTEG | 7.3 |
| 8.2 | 8 | INTEG | 8.1 |
| 8.3 | 8 | INTEG/VERIFY | 8.2 |

Total: 28 batches across 9 phases (0–8).

Parallelism note: 1.3 and 1.2 can run in parallel. 2.x can start as soon as 1.1 is verified, in parallel with 1.2–1.4. 6.1 can start once 4.2 is verified, in parallel with 5.x. Use `TASK_BOARD.md` to avoid two agents editing the same file.

---

## 7. Benchmark and measurement rules (binding)

1. **Warm-up before measuring.** Five unmeasured iterations per configuration.
2. **Repeat.** Thirty measured iterations per configuration. No single-run numbers in any document.
3. **Report distribution.** Median, min, max, IQR. Never report only the best run or only the mean.
4. **Separate what you time.** Kernel-only, end-to-end, and transfer are three different numbers. Never mix them in one column.
5. **Synchronize correctly.** Before reading CUDA event elapsed time, synchronize the event. Before reading host wall clock after GPU work, synchronize the device.
6. **Record the hardware.** Every CSV row carries GPU name, driver version, and CUDA runtime version. Every report links to `docs/ENVIRONMENT.md`.
7. **Record the conditions.** Clocks, temperature, and power, as specified in section 6 intro.
8. **Fixed dimensions.** Benchmarks use the sizes in `BenchmarkConfig`. Do not change sizes to make a result look better.
9. **No cherry-picking.** A benchmark result is reported whether or not it is favorable.
10. **No unmeasured comparisons.** Do not say "faster than cuBLAS", "competitive with cuBLAS", or any percentage against cuBLAS unless the cuBLAS row is in a CSV in `results/` and the report references it.
11. **Tensor Cores are out of scope** for this milestone. Do not mention them as a performance path in claims.

---

## 8. Handoff note template

Every batch ends with a file `handoffs/<BATCH_ID>.md` (for example `handoffs/3.2.md`). Use this exact template:

```markdown
# Handoff — Batch <ID>: <title>

- **Agent / session:** <name or identifier>
- **Role used:** <ARCH | CORE | RUNTIME | KERNEL | API | TEST | BENCH | OPT | INTEG | VERIFY>
- **Date:** <YYYY-MM-DD>
- **Status at end of session:** <UNDER_REVIEW | BLOCKED | IN_PROGRESS (checkpoint)>

## Done
- <bullet list of completed items, mapped to the batch's acceptance criteria>

## Not done
- <bullet list, or "none">

## Files changed
- <path> — created / modified — <one-line reason>

## Decisions made (and where recorded)
- <decision> — recorded in <docs/DESIGN_DECISIONS.md section or "not recorded: reason">

## Verification
- Command: `<exact command>`
- Result: <pass/fail, test counts, key output line(s)>

## Blockers / questions for owner
- <or "none">

## Next step (exact command or action for the next agent)
- <one concrete action>
```

Do not paste source code into this note.

---

## 9. Open decisions (owner must answer before the listed batch)

| ID | Decision | Needed before | Default if owner does not answer |
|---|---|---|---|
| D1 | License for the repository (MIT, Apache-2.0, or other) | 0.2 | Do not create LICENSE. Mark 0.2 `BLOCKED` on D1 only if all other 0.2 items are done; otherwise proceed without LICENSE. |
| D2 | Author name and contact for README and license headers | 0.2 | Use "Nirmal" |
| D3 | Whether to enable the cuBLAS reference row | 6.4 | Off (`CME_WITH_CUBLAS=OFF`). Results will say "cuBLAS not measured". |
| D4 | Installed CUDA toolkit version to pin in CI | 8.2 | Use the version recorded in `docs/ENVIRONMENT.md` from Batch 0.1 |
| D5 | Whether the Windows 11 build is required to pass before Phase 5 is complete | 5.3 | Not required. Linux is the gate. Windows result is recorded as `NOT VERIFIED` if not run. |
| D6 | Whether to publish the repository publicly | 8.3 | Keep private. Do not add public links anywhere. |

Agents encountering an open decision must not guess silently. Record it in the handoff note under "Blockers / questions for owner", apply the default if one is listed and the batch is blocked on it, and note that the default was applied.

---

## 10. Known risks and how to handle them

| Risk | Early sign | Response |
|---|---|---|
| Free-tier agent runs out mid-batch | Handoff note missing or partial | Next agent reads `TASK_BOARD.md` and the last `handoffs/` file. If the batch's files are partially written, the next agent completes or reverts them, then verifies from scratch. Never trust a partial batch without running its verification. |
| Two agents edit the same file | Merge conflicts, build breaks in unrelated code | Check `TASK_BOARD.md` before touching a file. Owners are listed per batch. Do not edit files owned by another batch. |
| Interface drift | Compile errors between modules | Diff against section 4. Fix the implementation, not the interface. If the interface is wrong, `BLOCKED`. |
| CUDA version mismatch | Build fails on `cuda_runtime.h` or architecture flags | Check `docs/ENVIRONMENT.md`. Confirm `CME_CUDA_ARCHITECTURES` matches the GPU (89 for RTX 4070 Ti SUPER). Do not change the architecture value to silence an error without recording why. |
| Kernel hangs or wrong results | Test timeouts, mismatched output | Run `compute-sanitizer` first. Check `__syncthreads()` placement and bounds checks. Do not increase tolerance to hide a bug. |
| Tolerance too tight for FP32 | Correct-looking kernel fails only at large K | Check the error grows with K as expected. Adjust `atol_for(K)` only with a documented analysis in `docs/ALGORITHMS.md`, and apply the change to all tests at once. |
| Benchmark results too noisy | IQR large relative to median | Check clocks and temperature in the CSV. Re-run after GPU cooldown. Report the noise honestly. Do not drop outliers without a written rule. |
| Optimization makes things worse | Profiler shows regression | Revert. Record the negative result. This is an acceptable outcome. |
| Context overflow in a single agent | Agent truncates or loses track | Use the checkpoint procedure in section 0. Keep batches small. Read only the files listed for your batch. |

---

## 11. Definition of done (project level)

Copy into `QUALITY_STANDARDS.md`. The project is complete only when every row has evidence.

| Area | Acceptance criteria | Evidence required |
|---|---|---|
| Build | Clean CMake configure and build on Linux release preset | Command output summary |
| Correctness | All supported algorithms agree with CPU reference within the tolerance rule | `ctest` summary; test count |
| Memory | No CUDA memory errors | `compute-sanitizer --tool memcheck` clean (in `scripts/test.sh` output) |
| Races | No shared-memory races in kernels | `compute-sanitizer --tool racecheck` clean on kernel tests |
| Edge cases | Odd dimensions, non-square, K=1, M=1, strided views, beta=0 with NaN, alpha=0 tested | `test_gemm_edge_cases` passes |
| API | Stable documented public headers with no CUDA includes | grep result; `INTERFACES.md` matches section 4 |
| Performance | Reproducible benchmark CSV from a full run | File in `results/` and name in `PERFORMANCE_REPORT.md` |
| Profiling | Nsight evidence for every optimization claim | Profile files in `results/profiles/`; `PROFILING.md` entries |
| Documentation | Build, usage, algorithms, benchmarking, and performance documented | Files exist and contain no `TBD` placeholders |
| Integration | All targets build together including examples and benchmarks | Build command with all options ON |
| Publication | Clean repository; examples compile and run; CI workflow valid | Example outputs; workflow file |

Additional decisions (binding):
- Catch2 v3 pinned to a specific tag, recorded.
- CTest for test execution.
- Explicit CUDA architecture configuration (default 89, recorded).
- Benchmark results are machine-readable CSV.
- No hardware-specific performance claim without the hardware recorded in the same document.
- No Tensor Core implementation in this milestone.

---

## 12. Quick start for the very first agent

If you are the first agent reading this and the repository does not exist yet:

1. Confirm with the owner that open decisions D1 and D2 are answered, or apply their defaults.
2. Execute Batch 0.1. Do not skip it.
3. Execute Batch 0.2 and 0.3.
4. Stop. Write your handoff. The next agent continues from Phase 1.

If you are the first agent and the repository **already exists**: skip to `TASK_BOARD.md`, find the first batch not `VERIFIED`, and follow section 0.

---

*End of handoff. This document is the single source of truth for the project plan. If it conflicts with a chat message, this document wins unless the owner explicitly overrides a specific rule, in which case record the override in `docs/DESIGN_DECISIONS.md`.*
