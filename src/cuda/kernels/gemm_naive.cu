#include "cuda/kernels/gemm_kernels.hpp"

#include <cuda_runtime.h>

namespace cme::detail {
namespace {

constexpr int kBlockSize = 16;

__global__ void gemm_naive_kernel(DeviceGemmArgs args) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= args.M || col >= args.N) {
        return;
    }

    float acc = 0.0f;
    for (int k = 0; k < args.K; ++k) {
        acc += args.A[row * args.lda + k] * args.B[k * args.ldb + col];
    }

    float value = args.alpha * acc;
    if (args.beta != 0.0f) {
        value += args.beta * args.C[row * args.ldc + col];
    }
    args.C[row * args.ldc + col] = value;
}

}  // namespace

cudaError_t launch_gemm_naive(const DeviceGemmArgs& args, cudaStream_t stream) {
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((args.N + kBlockSize - 1) / kBlockSize,
                    (args.M + kBlockSize - 1) / kBlockSize);
    gemm_naive_kernel<<<grid, block, 0, stream>>>(args);
    return cudaGetLastError();
}

}  // namespace cme::detail
