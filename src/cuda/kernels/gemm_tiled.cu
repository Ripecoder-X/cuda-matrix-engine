#include "cuda/kernels/gemm_kernels.hpp"

#include <cuda_runtime.h>

namespace cme::detail {
namespace {

template <int TILE>
__global__ void gemm_tiled_kernel(DeviceGemmArgs args) {
    __shared__ float a_tile[TILE][TILE];
    __shared__ float b_tile[TILE][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty;
    const int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int k0 = 0; k0 < args.K; k0 += TILE) {
        const int a_row = row;
        const int a_col = k0 + tx;
        a_tile[ty][tx] =
            (a_row < args.M && a_col < args.K) ? args.A[a_row * args.lda + a_col] : 0.0f;

        const int b_row = k0 + ty;
        const int b_col = col;
        b_tile[ty][tx] =
            (b_row < args.K && b_col < args.N) ? args.B[b_row * args.ldb + b_col] : 0.0f;

        __syncthreads();
#pragma unroll
        for (int p = 0; p < TILE; ++p) {
            acc += a_tile[ty][p] * b_tile[p][tx];
        }
        __syncthreads();
    }

    if (row < args.M && col < args.N) {
        float value = args.alpha * acc;
        if (args.beta != 0.0f) {
            value += args.beta * args.C[row * args.ldc + col];
        }
        args.C[row * args.ldc + col] = value;
    }
}

template <int TILE>
cudaError_t launch_tiled(const DeviceGemmArgs& args, cudaStream_t stream) {
    const dim3 block(TILE, TILE);
    const dim3 grid((args.N + TILE - 1) / TILE, (args.M + TILE - 1) / TILE);
    gemm_tiled_kernel<TILE><<<grid, block, 0, stream>>>(args);
    return cudaGetLastError();
}

}  // namespace

cudaError_t launch_gemm_tiled(const DeviceGemmArgs& args, cudaStream_t stream) {
    switch (args.tile_size) {
        case 16:
            return launch_tiled<16>(args, stream);
        case 32:
            return launch_tiled<32>(args, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace cme::detail
