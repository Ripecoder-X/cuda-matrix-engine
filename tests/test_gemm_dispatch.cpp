#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include "cme/gemm_config.hpp"
#include "cme/matrix.hpp"
#include "cme/status.hpp"
#include "core/gemm_reference.hpp"
#include "cuda/device_buffer.hpp"
#include "cuda/gemm_dispatch.hpp"
#include "kernel_test_util.hpp"
#include "test_helpers.hpp"

namespace {

using cme::GemmAlgorithm;

void run_dispatch_case(GemmAlgorithm algorithm, int tile_size) {
    cme::test::require_cuda_device();

    constexpr int M = 17;
    constexpr int N = 33;
    constexpr int K = 5;

    cme::Matrix A(M, K);
    cme::Matrix B(K, N);
    cme::Matrix C(M, N);
    cme::Matrix reference(M, N);
    cme::test::fill_random(A, 700);
    cme::test::fill_random(B, 701);
    cme::test::fill_random(C, 702);
    reference = C;

    cme::detail::gemm_cpu_reference(A.view(), B.view(), reference.view(), 1.25f, -0.5f);

    const std::size_t a_count = static_cast<std::size_t>(M) * static_cast<std::size_t>(K);
    const std::size_t b_count = static_cast<std::size_t>(K) * static_cast<std::size_t>(N);
    const std::size_t c_count = static_cast<std::size_t>(M) * static_cast<std::size_t>(N);

    cme::detail::DeviceBuffer device_a;
    cme::detail::DeviceBuffer device_b;
    cme::detail::DeviceBuffer device_c;
    REQUIRE(cme::detail::DeviceBuffer::create(a_count, &device_a).is_ok());
    REQUIRE(cme::detail::DeviceBuffer::create(b_count, &device_b).is_ok());
    REQUIRE(cme::detail::DeviceBuffer::create(c_count, &device_c).is_ok());
    REQUIRE(device_a.copy_from_host(A.data(), a_count).is_ok());
    REQUIRE(device_b.copy_from_host(B.data(), b_count).is_ok());
    REQUIRE(device_c.copy_from_host(C.data(), c_count).is_ok());

    cme::detail::DeviceGemmArgs args;
    args.M = M;
    args.N = N;
    args.K = K;
    args.A = device_a.data();
    args.lda = K;
    args.B = device_b.data();
    args.ldb = N;
    args.C = device_c.data();
    args.ldc = N;
    args.alpha = 1.25f;
    args.beta = -0.5f;
    args.tile_size = tile_size;

    const cme::Status status = cme::detail::gemm_device(algorithm, args, nullptr);
    REQUIRE(status.is_ok());
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    cme::Matrix result(M, N);
    REQUIRE(device_c.copy_to_host(result.data(), c_count).is_ok());

    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            CHECK(cme::test::close_enough(result(r, c), reference(r, c), K));
        }
    }
}

}  // namespace

TEST_CASE("Device dispatch runs every CUDA algorithm", "[gemm][dispatch][cuda]") {
    run_dispatch_case(GemmAlgorithm::CUDA_NAIVE, 16);
    run_dispatch_case(GemmAlgorithm::CUDA_TILED, 16);
    run_dispatch_case(GemmAlgorithm::CUDA_TILED, 32);
    run_dispatch_case(GemmAlgorithm::CUDA_OPTIMIZED, 16);
    run_dispatch_case(GemmAlgorithm::CUDA_OPTIMIZED, 32);
}

TEST_CASE("Device dispatch rejects CPU reference and unknown algorithms",
          "[gemm][dispatch][cuda]") {
    cme::detail::DeviceGemmArgs args;
    const cme::Status cpu = cme::detail::gemm_device(GemmAlgorithm::CPU_REFERENCE, args, nullptr);
    CHECK(cpu.code() == cme::StatusCode::UnsupportedConfiguration);

    const cme::Status unknown = cme::detail::gemm_device(static_cast<GemmAlgorithm>(99), args,
                                                         nullptr);
    CHECK(unknown.code() == cme::StatusCode::InvalidArgument);
}
