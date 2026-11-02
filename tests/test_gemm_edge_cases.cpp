#include <algorithm>
#include <cmath>
#include <limits>

#include <catch2/catch_test_macros.hpp>

#include "cme/gemm.hpp"
#include "cme/matrix.hpp"
#include "cme/status.hpp"
#include "kernel_test_util.hpp"
#include "test_helpers.hpp"

namespace {

constexpr float kExpected[2][2] = {{19.0f, 22.0f}, {43.0f, 50.0f}};

void check_smoke_result(const cme::Matrix& C) {
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 2; ++c) {
            CHECK(cme::test::close_enough(C(r, c), kExpected[r][c], 2));
        }
    }
}

void run_smoke_case(cme::GemmAlgorithm algorithm, int tile_size) {
    cme::Matrix A(2, 2);
    cme::Matrix B(2, 2);
    cme::Matrix C(2, 2);
    A(0, 0) = 1.0f;
    A(0, 1) = 2.0f;
    A(1, 0) = 3.0f;
    A(1, 1) = 4.0f;
    B(0, 0) = 5.0f;
    B(0, 1) = 6.0f;
    B(1, 0) = 7.0f;
    B(1, 1) = 8.0f;

    cme::GemmConfig config;
    config.algorithm = algorithm;
    config.tile_size = tile_size;

    const cme::Status status = cme::gemm(A.view(), B.view(), C.view(), 1.0f, 0.0f, config);
    REQUIRE(status.is_ok());
    check_smoke_result(C);
}

}  // namespace

TEST_CASE("Public gemm CPU reference smoke test", "[gemm][api][smoke]") {
    run_smoke_case(cme::GemmAlgorithm::CPU_REFERENCE, 16);
}

TEST_CASE("Public gemm CUDA smoke test", "[gemm][api][smoke][cuda]") {
    cme::test::require_cuda_device();
    run_smoke_case(cme::GemmAlgorithm::CUDA_NAIVE, 16);
    run_smoke_case(cme::GemmAlgorithm::CUDA_TILED, 16);
    run_smoke_case(cme::GemmAlgorithm::CUDA_TILED, 32);
    run_smoke_case(cme::GemmAlgorithm::CUDA_OPTIMIZED, 16);
    run_smoke_case(cme::GemmAlgorithm::CUDA_OPTIMIZED, 32);
}

TEST_CASE("GEMM rejects incompatible dimensions", "[gemm][edge]") {
    cme::Matrix A(2, 3);
    cme::Matrix B(2, 3);
    cme::Matrix C(2, 3);
    cme::GemmConfig config;
    config.algorithm = cme::GemmAlgorithm::CUDA_NAIVE;

    const cme::Status status = cme::gemm(A.view(), B.view(), C.view(), 1.0f, 0.0f, config);
    REQUIRE(status.code() == cme::StatusCode::DimensionMismatch);

    cme::Matrix compatible_b(3, 4);
    cme::Matrix tall_c(3, 4);
    const cme::Status shape_status =
        cme::gemm(A.view(), compatible_b.view(), tall_c.view(), 1.0f, 0.0f, config);
    REQUIRE(shape_status.code() == cme::StatusCode::DimensionMismatch);
}

TEST_CASE("GEMM rejects invalid and empty views", "[gemm][edge]") {
    cme::Matrix A(2, 3);
    cme::Matrix B(3, 2);
    cme::Matrix C(2, 2);
    cme::GemmConfig config;
    config.algorithm = cme::GemmAlgorithm::CUDA_NAIVE;

    cme::MatrixView<const float> null_view{};
    const cme::Status null_status =
        cme::gemm(null_view, B.view(), C.view(), 1.0f, 0.0f, config);
    REQUIRE(null_status.code() == cme::StatusCode::InvalidArgument);

    cme::MatrixView<const float> empty_view{A.data(), 0, A.cols(), A.stride()};
    const cme::Status empty_status =
        cme::gemm(empty_view, B.view(), C.view(), 1.0f, 0.0f, config);
    REQUIRE(empty_status.code() == cme::StatusCode::InvalidArgument);
}

TEST_CASE("GEMM rejects overlapping matrices", "[gemm][edge]") {
    cme::Matrix A(4, 4);
    cme::Matrix B(4, 4);
    cme::GemmConfig config;
    config.algorithm = cme::GemmAlgorithm::CUDA_TILED;
    config.tile_size = 16;

    cme::MatrixView<float> c_over_a{A.data(), A.rows(), A.cols(), A.stride()};
    const cme::Status status =
        cme::gemm(A.view(), B.view(), c_over_a, 1.0f, 0.0f, config);
    REQUIRE(status.code() == cme::StatusCode::UnsupportedConfiguration);
}

TEST_CASE("GEMM validates tile size and algorithm values", "[gemm][edge]") {
    cme::Matrix A(4, 4);
    cme::Matrix B(4, 4);
    cme::Matrix C(4, 4);

    cme::GemmConfig tiled_config;
    tiled_config.algorithm = cme::GemmAlgorithm::CUDA_TILED;
    tiled_config.tile_size = 8;
    REQUIRE(cme::gemm(A.view(), B.view(), C.view(), 1.0f, 0.0f, tiled_config).code() ==
            cme::StatusCode::UnsupportedConfiguration);

    cme::GemmConfig optimized_config;
    optimized_config.algorithm = cme::GemmAlgorithm::CUDA_OPTIMIZED;
    optimized_config.tile_size = 33;
    REQUIRE(cme::gemm(A.view(), B.view(), C.view(), 1.0f, 0.0f, optimized_config).code() ==
            cme::StatusCode::UnsupportedConfiguration);

    cme::GemmConfig invalid_config;
    invalid_config.algorithm = static_cast<cme::GemmAlgorithm>(99);
    REQUIRE(cme::gemm(A.view(), B.view(), C.view(), 1.0f, 0.0f, invalid_config).code() ==
            cme::StatusCode::InvalidArgument);
}

TEST_CASE("GEMM beta zero ignores NaN contents of C", "[gemm][edge][cuda]") {
    cme::test::require_cuda_device();

    constexpr int size = 17;
    cme::Matrix A(size, size);
    cme::Matrix B(size, size);
    cme::Matrix C(size, size);
    cme::test::fill_random(A, 700);
    cme::test::fill_random(B, 701);
    std::fill(C.data(), C.data() + static_cast<std::size_t>(size) * size,
              std::numeric_limits<float>::quiet_NaN());

    cme::Matrix expected(size, size);
    cme::GemmConfig reference_config;
    reference_config.algorithm = cme::GemmAlgorithm::CPU_REFERENCE;
    REQUIRE(cme::gemm(A.view(), B.view(), expected.view(), 1.0f, 0.0f, reference_config).is_ok());

    cme::GemmConfig config;
    config.algorithm = cme::GemmAlgorithm::CUDA_TILED;
    config.tile_size = 16;
    REQUIRE(cme::gemm(A.view(), B.view(), C.view(), 1.0f, 0.0f, config).is_ok());

    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            REQUIRE(std::isfinite(C(r, c)));
            CHECK(cme::test::close_enough(C(r, c), expected(r, c), size));
        }
    }
}

TEST_CASE("GEMM alpha zero returns beta times C", "[gemm][edge][cuda]") {
    cme::test::require_cuda_device();

    cme::Matrix A(9, 7);
    cme::Matrix B(7, 11);
    cme::Matrix initial_c(9, 11);
    cme::test::fill_random(A, 710);
    cme::test::fill_random(B, 711);
    cme::test::fill_random(initial_c, 712);

    cme::Matrix expected = initial_c;
    cme::GemmConfig reference_config;
    reference_config.algorithm = cme::GemmAlgorithm::CPU_REFERENCE;
    REQUIRE(cme::gemm(A.view(), B.view(), expected.view(), 0.0f, -0.75f, reference_config).is_ok());

    cme::Matrix result = initial_c;
    cme::GemmConfig config;
    config.algorithm = cme::GemmAlgorithm::CUDA_OPTIMIZED;
    config.tile_size = 16;
    REQUIRE(cme::gemm(A.view(), B.view(), result.view(), 0.0f, -0.75f, config).is_ok());

    for (int r = 0; r < result.rows(); ++r) {
        for (int c = 0; c < result.cols(); ++c) {
            CHECK(cme::test::close_enough(result(r, c), expected(r, c), A.cols()));
        }
    }
}

TEST_CASE("GEMM supports strided sub-views", "[gemm][edge][cuda]") {
    cme::test::require_cuda_device();

    cme::Matrix a_storage(7, 8);
    cme::Matrix b_storage(5, 9);
    cme::Matrix c_storage(6, 10);
    cme::test::fill_random(a_storage, 720);
    cme::test::fill_random(b_storage, 721);
    cme::test::fill_random(c_storage, 722);

    cme::MatrixView<const float> A{a_storage.data() + 1 * a_storage.stride() + 2, 4, 3,
                                   a_storage.stride()};
    cme::MatrixView<const float> B{b_storage.data() + 0 * b_storage.stride() + 1, 3, 5,
                                   b_storage.stride()};
    cme::MatrixView<float> C{c_storage.data() + 1 * c_storage.stride() + 3, 4, 5,
                             c_storage.stride()};

    auto pack = [](cme::MatrixView<const float> view) {
        cme::Matrix packed(view.rows, view.cols);
        for (int r = 0; r < view.rows; ++r) {
            for (int c = 0; c < view.cols; ++c) {
                packed(r, c) = view(r, c);
            }
        }
        return packed;
    };

    cme::Matrix packed_a = pack(A);
    cme::Matrix packed_b = pack(B);
    cme::Matrix expected = pack(C);
    cme::GemmConfig reference_config;
    reference_config.algorithm = cme::GemmAlgorithm::CPU_REFERENCE;
    REQUIRE(cme::gemm(packed_a.view(), packed_b.view(), expected.view(), 2.5f, -0.75f,
                      reference_config)
                .is_ok());

    cme::GemmConfig config;
    config.algorithm = cme::GemmAlgorithm::CUDA_TILED;
    config.tile_size = 32;
    REQUIRE(cme::gemm(A, B, C, 2.5f, -0.75f, config).is_ok());

    for (int r = 0; r < C.rows; ++r) {
        for (int c = 0; c < C.cols; ++c) {
            CHECK(cme::test::close_enough(C(r, c), expected(r, c), A.cols));
        }
    }
}

TEST_CASE("GEMM large K remains within tolerance", "[gemm][edge][cuda]") {
    cme::test::require_cuda_device();

    constexpr int K = 4096;
    cme::Matrix A(1, K);
    cme::Matrix B(K, 1);
    cme::Matrix C(1, 1);
    cme::test::fill_random(A, 730);
    cme::test::fill_random(B, 731);

    cme::Matrix expected(1, 1);
    cme::GemmConfig reference_config;
    reference_config.algorithm = cme::GemmAlgorithm::CPU_REFERENCE;
    REQUIRE(cme::gemm(A.view(), B.view(), expected.view(), 1.0f, 0.0f, reference_config).is_ok());

    cme::GemmConfig config;
    config.algorithm = cme::GemmAlgorithm::CUDA_NAIVE;
    REQUIRE(cme::gemm(A.view(), B.view(), C.view(), 1.0f, 0.0f, config).is_ok());

    REQUIRE(std::isfinite(C(0, 0)));
    CHECK(cme::test::close_enough(C(0, 0), expected(0, 0), K));
}
