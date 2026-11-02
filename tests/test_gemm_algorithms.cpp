#include <array>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include "kernel_test_util.hpp"

namespace {

struct Shape {
    int M;
    int N;
    int K;
};

const std::vector<Shape> kNaiveShapes = {
    {1, 1, 1},
    {2, 2, 2},
    {17, 33, 5},
    {64, 64, 64},
};

const std::vector<Shape> kTiledShapes = {
    {1, 1, 1},
    {17, 33, 5},
    {64, 64, 64},
    {127, 129, 131},
    {256, 256, 256},
};

const std::array<int, 2> kTileSizes = {16, 32};

}  // namespace

TEST_CASE("Naive GEMM matches CPU reference", "[gemm][naive][cuda]") {
    std::uint32_t seed = 100;
    for (const Shape& shape : kNaiveShapes) {
        cme::test::run_kernel_case(cme::detail::launch_gemm_naive, shape.M, shape.N, shape.K,
                                   1.0f, 0.0f, 16, seed++);
    }
}

TEST_CASE("Naive GEMM honors alpha and beta", "[gemm][naive][cuda]") {
    cme::test::run_kernel_case(cme::detail::launch_gemm_naive, 17, 33, 5, 2.5f, -0.75f, 16, 200);
    cme::test::run_kernel_case(cme::detail::launch_gemm_naive, 17, 33, 5, 0.0f, 1.0f, 16, 201);
}

TEST_CASE("Tiled GEMM matches CPU reference", "[gemm][tiled][cuda]") {
    std::uint32_t seed = 300;
    for (const int tile_size : kTileSizes) {
        for (const Shape& shape : kTiledShapes) {
            cme::test::run_kernel_case(cme::detail::launch_gemm_tiled, shape.M, shape.N, shape.K,
                                       1.0f, 0.0f, tile_size, seed++);
        }
    }
}

TEST_CASE("Tiled GEMM honors alpha and beta", "[gemm][tiled][cuda]") {
    cme::test::run_kernel_case(cme::detail::launch_gemm_tiled, 17, 33, 5, 2.5f, -0.75f, 16, 400);
    cme::test::run_kernel_case(cme::detail::launch_gemm_tiled, 17, 33, 5, 0.0f, 1.0f, 32, 401);
}

TEST_CASE("Tiled GEMM rejects unsupported tile size", "[gemm][tiled][cuda]") {
    cme::test::require_cuda_device();
    cme::detail::DeviceGemmArgs args;
    args.tile_size = 8;
    REQUIRE(cme::detail::launch_gemm_tiled(args, nullptr) == cudaErrorInvalidValue);
}

TEST_CASE("Tiled GEMM racecheck case", "[gemm][tiled][racecheck][cuda]") {
    cme::test::run_kernel_case(cme::detail::launch_gemm_tiled, 64, 64, 64, 1.0f, 0.0f, 16, 402);
}

TEST_CASE("Optimized GEMM matches CPU reference", "[gemm][optimized][cuda]") {
    std::uint32_t seed = 500;
    for (const int tile_size : kTileSizes) {
        for (const Shape& shape : kTiledShapes) {
            cme::test::run_kernel_case(cme::detail::launch_gemm_optimized, shape.M, shape.N,
                                       shape.K, 1.0f, 0.0f, tile_size, seed++);
        }
    }
}

TEST_CASE("Optimized GEMM honors alpha and beta", "[gemm][optimized][cuda]") {
    cme::test::run_kernel_case(cme::detail::launch_gemm_optimized, 17, 33, 5, 2.5f, -0.75f, 16,
                               600);
    cme::test::run_kernel_case(cme::detail::launch_gemm_optimized, 17, 33, 5, 0.0f, 1.0f, 32, 601);
}

TEST_CASE("Optimized GEMM rejects unsupported tile size", "[gemm][optimized][cuda]") {
    cme::test::require_cuda_device();
    cme::detail::DeviceGemmArgs args;
    args.tile_size = 33;
    REQUIRE(cme::detail::launch_gemm_optimized(args, nullptr) == cudaErrorInvalidValue);
}

TEST_CASE("Optimized GEMM memcheck non-multiple case",
          "[gemm][optimized][memcheck][cuda]") {
    cme::test::run_kernel_case(cme::detail::launch_gemm_optimized, 127, 129, 131, 1.0f, 0.0f, 16,
                               602);
}
