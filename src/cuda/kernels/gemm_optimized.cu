#include "cuda/kernels/gemm_kernels.hpp"

#include <cuda_runtime.h>

namespace cme::detail {
namespace {

// One thread computes a 2x2 register micro-tile. The shared-memory tiles are padded by
// one column to break the stride that otherwise maps consecutive rows to the same bank.
template <int TILE>
__global__ void gemm_optimized_kernel(DeviceGemmArgs args) {
    __shared__ float a_tile[TILE][TILE + 1];
    __shared__ float b_tile[TILE][TILE + 1];

    constexpr int kThreadsPerDim = TILE / 2;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kThreadsPerDim + tx;
    constexpr int kThreadCount = kThreadsPerDim * kThreadsPerDim;

    const int row0 = blockIdx.y * TILE + ty * 2;
    const int col0 = blockIdx.x * TILE + tx * 2;

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;

    for (int k0 = 0; k0 < args.K; k0 += TILE) {
        for (int index = tid; index < TILE * TILE; index += kThreadCount) {
            const int tile_row = index / TILE;
            const int tile_col = index % TILE;

            const int a_row = blockIdx.y * TILE + tile_row;
            const int a_col = k0 + tile_col;
            a_tile[tile_row][tile_col] =
                (a_row < args.M && a_col < args.K) ? args.A[a_row * args.lda + a_col] : 0.0f;

            const int b_row = k0 + tile_row;
            const int b_col = blockIdx.x * TILE + tile_col;
            b_tile[tile_row][tile_col] =
                (b_row < args.K && b_col < args.N) ? args.B[b_row * args.ldb + b_col] : 0.0f;
        }
        __syncthreads();

        const int local_row = ty * 2;
        const int local_col = tx * 2;
#pragma unroll
        for (int p = 0; p < TILE; ++p) {
            const float a0 = a_tile[local_row][p];
            const float a1 = a_tile[local_row + 1][p];
            const float b0 = b_tile[p][local_col];
            const float b1 = b_tile[p][local_col + 1];
            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }
        __syncthreads();
    }

    if (row0 < args.M && col0 < args.N) {
        float value = args.alpha * acc00;
        if (args.beta != 0.0f) {
            value += args.beta * args.C[row0 * args.ldc + col0];
        }
        args.C[row0 * args.ldc + col0] = value;
    }
    if (row0 < args.M && col0 + 1 < args.N) {
        float value = args.alpha * acc01;
        if (args.beta != 0.0f) {
            value += args.beta * args.C[row0 * args.ldc + col0 + 1];
        }
        args.C[row0 * args.ldc + col0 + 1] = value;
    }
    if (row0 + 1 < args.M && col0 < args.N) {
        float value = args.alpha * acc10;
        if (args.beta != 0.0f) {
            value += args.beta * args.C[(row0 + 1) * args.ldc + col0];
        }
        args.C[(row0 + 1) * args.ldc + col0] = value;
    }
    if (row0 + 1 < args.M && col0 + 1 < args.N) {
        float value = args.alpha * acc11;
        if (args.beta != 0.0f) {
            value += args.beta * args.C[(row0 + 1) * args.ldc + col0 + 1];
        }
        args.C[(row0 + 1) * args.ldc + col0 + 1] = value;
    }
}

template <int TILE>
cudaError_t launch_optimized(const DeviceGemmArgs& args, cudaStream_t stream) {
    constexpr int kThreadsPerDim = TILE / 2;
    const dim3 block(kThreadsPerDim, kThreadsPerDim);
    const dim3 grid((args.N + TILE - 1) / TILE, (args.M + TILE - 1) / TILE);
    gemm_optimized_kernel<TILE><<<grid, block, 0, stream>>>(args);
    return cudaGetLastError();
}

}  // namespace

cudaError_t launch_gemm_optimized(const DeviceGemmArgs& args, cudaStream_t stream) {
    switch (args.tile_size) {
        case 16:
            return launch_optimized<16>(args, stream);
        case 32:
            return launch_optimized<32>(args, stream);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace cme::detail
