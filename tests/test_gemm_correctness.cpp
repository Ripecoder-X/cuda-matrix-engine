#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "cme/gemm.hpp"
#include "cme/matrix.hpp"
#include "cme/status.hpp"
#include "kernel_test_util.hpp"
#include "test_helpers.hpp"

namespace {

struct Shape {
    int M;
    int N;
    int K;
};

struct Coefficients {
    float alpha;
    float beta;
};

struct AlgorithmCase {
    cme::GemmAlgorithm algorithm;
    int tile_size;
    const char* name;
};

const std::vector<Shape> kShapes = {
    {1, 1, 1},     {1, 7, 3},       {7, 1, 3},       {17, 33, 5},
    {64, 64, 64},  {127, 129, 131}, {256, 256, 256}, {513, 255, 257},
};

const std::array<Coefficients, 5> kCoefficients = {{
    {1.0f, 0.0f},
    {1.0f, 1.0f},
    {2.5f, -0.75f},
    {0.0f, 1.0f},
    {-1.0f, 0.5f},
}};

const std::array<AlgorithmCase, 5> kAlgorithms = {{
    {cme::GemmAlgorithm::CUDA_NAIVE, 16, "CUDA_NAIVE"},
    {cme::GemmAlgorithm::CUDA_TILED, 16, "CUDA_TILED_16"},
    {cme::GemmAlgorithm::CUDA_TILED, 32, "CUDA_TILED_32"},
    {cme::GemmAlgorithm::CUDA_OPTIMIZED, 16, "CUDA_OPTIMIZED_16"},
    {cme::GemmAlgorithm::CUDA_OPTIMIZED, 32, "CUDA_OPTIMIZED_32"},
}};

void check_matrix(const cme::Matrix& got,
                  const cme::Matrix& expected,
                  int K,
                  const char* algorithm_name,
                  const Shape& shape,
                  const Coefficients& coefficients,
                  std::uint32_t seed) {
    int mismatches = 0;
    int first_row = -1;
    int first_col = -1;
    float first_got = 0.0f;
    float first_expected = 0.0f;

    for (int r = 0; r < got.rows(); ++r) {
        for (int c = 0; c < got.cols(); ++c) {
            if (!cme::test::close_enough(got(r, c), expected(r, c), K)) {
                if (mismatches == 0) {
                    first_row = r;
                    first_col = c;
                    first_got = got(r, c);
                    first_expected = expected(r, c);
                }
                ++mismatches;
            }
        }
    }

    if (mismatches != 0) {
        INFO("algorithm=" << algorithm_name << " M=" << shape.M << " N=" << shape.N
                          << " K=" << shape.K << " alpha=" << coefficients.alpha
                          << " beta=" << coefficients.beta << " seed=" << seed << " at ("
                          << first_row << ", " << first_col << "): got " << first_got
                          << ", expected " << first_expected);
    }
    CHECK(mismatches == 0);
}

}  // namespace

TEST_CASE("GEMM correctness matrix matches CPU reference", "[gemm][correctness][cuda]") {
    cme::test::require_cuda_device();

    const std::array<std::uint32_t, 3> seeds = {1, 2, 3};
    std::size_t algorithm_runs = 0;
    std::size_t element_comparisons = 0;

    for (const Shape& shape : kShapes) {
        for (const std::uint32_t seed : seeds) {
            cme::Matrix A(shape.M, shape.K);
            cme::Matrix B(shape.K, shape.N);
            cme::Matrix initial_c(shape.M, shape.N);
            cme::test::fill_random(A, seed);
            cme::test::fill_random(B, seed + 1);
            cme::test::fill_random(initial_c, seed + 2);

            for (const Coefficients& coefficients : kCoefficients) {
                cme::Matrix reference = initial_c;
                cme::GemmConfig reference_config;
                reference_config.algorithm = cme::GemmAlgorithm::CPU_REFERENCE;
                const cme::Status reference_status =
                    cme::gemm(A.view(), B.view(), reference.view(), coefficients.alpha,
                              coefficients.beta, reference_config);
                REQUIRE(reference_status.is_ok());

                for (const AlgorithmCase& algorithm : kAlgorithms) {
                    cme::Matrix result = initial_c;
                    cme::GemmConfig config;
                    config.algorithm = algorithm.algorithm;
                    config.tile_size = algorithm.tile_size;

                    const cme::Status status =
                        cme::gemm(A.view(), B.view(), result.view(), coefficients.alpha,
                                  coefficients.beta, config);
                    REQUIRE(status.is_ok());

                    check_matrix(result, reference, shape.K, algorithm.name, shape, coefficients,
                                 seed);
                    ++algorithm_runs;
                    element_comparisons +=
                        static_cast<std::size_t>(shape.M) * static_cast<std::size_t>(shape.N);
                }
            }
        }
    }

    std::cout << "GEMM correctness matrix: " << algorithm_runs << " algorithm runs, "
              << element_comparisons << " element comparisons\n";
    WARN("GEMM correctness matrix: " << algorithm_runs << " algorithm runs, "
                                     << element_comparisons << " element comparisons");
}
