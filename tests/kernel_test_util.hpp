#pragma once
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include "cme/matrix.hpp"
#include "core/gemm_reference.hpp"
#include "cuda/device_buffer.hpp"
#include "cuda/kernels/gemm_kernels.hpp"
#include "test_helpers.hpp"

namespace cme::test {

using KernelLauncher = cudaError_t (*)(const detail::DeviceGemmArgs&, cudaStream_t);

inline void require_cuda_device() {
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        cudaGetLastError();
        SKIP("No CUDA device available");
    }
}

// Runs one kernel and compares every output element with the CPU reference.
inline void run_kernel_case(KernelLauncher launcher,
                            int M,
                            int N,
                            int K,
                            float alpha,
                            float beta,
                            int tile_size,
                            std::uint32_t seed) {
    require_cuda_device();

    Matrix A(M, K);
    Matrix B(K, N);
    Matrix C(M, N);
    Matrix reference(M, N);
    fill_random(A, seed);
    fill_random(B, seed + 1);
    fill_random(C, seed + 2);
    reference = C;

    detail::gemm_cpu_reference(A.view(), B.view(), reference.view(), alpha, beta);

    const std::size_t a_count = static_cast<std::size_t>(M) * static_cast<std::size_t>(K);
    const std::size_t b_count = static_cast<std::size_t>(K) * static_cast<std::size_t>(N);
    const std::size_t c_count = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);

    detail::DeviceBuffer device_a;
    detail::DeviceBuffer device_b;
    detail::DeviceBuffer device_c;
    REQUIRE(detail::DeviceBuffer::create(a_count, &device_a).is_ok());
    REQUIRE(detail::DeviceBuffer::create(b_count, &device_b).is_ok());
    REQUIRE(detail::DeviceBuffer::create(c_count, &device_c).is_ok());

    REQUIRE(device_a.copy_from_host(A.data(), a_count).is_ok());
    REQUIRE(device_b.copy_from_host(B.data(), b_count).is_ok());
    REQUIRE(device_c.copy_from_host(C.data(), c_count).is_ok());

    detail::DeviceGemmArgs args;
    args.M = M;
    args.N = N;
    args.K = K;
    args.A = device_a.data();
    args.lda = K;
    args.B = device_b.data();
    args.ldb = N;
    args.C = device_c.data();
    args.ldc = N;
    args.alpha = alpha;
    args.beta = beta;
    args.tile_size = tile_size;

    REQUIRE(launcher(args, nullptr) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    Matrix result(M, N);
    REQUIRE(device_c.copy_to_host(result.data(), c_count).is_ok());

    int mismatches = 0;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            const float got = result(r, c);
            const float expected = reference(r, c);
            if (!close_enough(got, expected, K)) {
                ++mismatches;
                if (mismatches <= 10) {
                    INFO("M=" << M << " N=" << N << " K=" << K << " alpha=" << alpha
                              << " beta=" << beta << " tile_size=" << tile_size << " at ("
                              << r << ", " << c << "): got " << got << ", expected "
                              << expected);
                    CHECK(mismatches < 0);
                }
            }
        }
    }
    CHECK(mismatches == 0);
}

}  // namespace cme::test
