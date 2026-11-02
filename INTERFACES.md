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
