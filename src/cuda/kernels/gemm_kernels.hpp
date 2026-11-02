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
