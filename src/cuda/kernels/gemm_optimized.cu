#include "cuda/kernels/gemm_kernels.hpp"

#include <cuda_runtime.h>

namespace cme::detail {
namespace {

// Each thread computes a kMicro x kMicro register micro-tile. TILE=32 uses a
// 4x4 micro-tile to reduce shared-memory load pressure per multiply-add. The
// shared-memory tiles are padded by one column to break the stride that
// otherwise maps consecutive rows to the same bank.
template <int TILE>
__global__ void gemm_optimized_kernel(DeviceGemmArgs args) {
    constexpr int kMicro = (TILE == 32) ? 4 : 2;
    constexpr int kThreadsPerDim = TILE / kMicro;
    constexpr int kThreadCount = kThreadsPerDim * kThreadsPerDim;

    __shared__ float a_tile[TILE][TILE + 1];
    __shared__ float b_tile[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * kThreadsPerDim + tx;

    const int row0 = blockIdx.y * TILE + ty * kMicro;
    const int col0 = blockIdx.x * TILE + tx * kMicro;

    float acc[kMicro][kMicro] = {};

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

        const int local_row = ty * kMicro;
        const int local_col = tx * kMicro;
#pragma unroll
        for (int p = 0; p < TILE; ++p) {
            float a[kMicro];
            float b[kMicro];
#pragma unroll
            for (int i = 0; i < kMicro; ++i) {
                a[i] = a_tile[local_row + i][p];
            }
#pragma unroll
            for (int j = 0; j < kMicro; ++j) {
                b[j] = b_tile[p][local_col + j];
            }
#pragma unroll
            for (int i = 0; i < kMicro; ++i) {
#pragma unroll
                for (int j = 0; j < kMicro; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kMicro; ++i) {
#pragma unroll
        for (int j = 0; j < kMicro; ++j) {
            if (row0 + i < args.M && col0 + j < args.N) {
                float value = args.alpha * acc[i][j];
                if (args.beta != 0.0f) {
                    value += args.beta * args.C[(row0 + i) * args.ldc + col0 + j];
                }
                args.C[(row0 + i) * args.ldc + col0 + j] = value;
            }
        }
    }
}

template <int TILE>
cudaError_t launch_optimized(const DeviceGemmArgs& args, cudaStream_t stream) {
    constexpr int kMicro = (TILE == 32) ? 4 : 2;
    constexpr int kThreadsPerDim = TILE / kMicro;
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
