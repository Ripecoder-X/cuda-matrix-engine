#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <initializer_list>
#include <limits>
#include "cme/matrix.hpp"
#include "core/gemm_reference.hpp"

namespace {

void fill(cme::Matrix& m, std::initializer_list<float> values) {
    REQUIRE(static_cast<std::size_t>(m.rows()) * static_cast<std::size_t>(m.cols()) ==
            values.size());
    int index = 0;
    for (float value : values) {
        m(index / m.cols(), index % m.cols()) = value;
        ++index;
    }
}

void check_matrix(const cme::Matrix& m, std::initializer_list<float> expected) {
    REQUIRE(static_cast<std::size_t>(m.rows()) * static_cast<std::size_t>(m.cols()) ==
            expected.size());
    int index = 0;
    for (float value : expected) {
        CHECK(m(index / m.cols(), index % m.cols()) == value);
        ++index;
    }
}

}  // namespace

TEST_CASE("CPU reference computes a hand-checked 2x2 product") {
    cme::Matrix A(2, 2);
    cme::Matrix B(2, 2);
    cme::Matrix C(2, 2);
    fill(A, {1.0f, 2.0f, 3.0f, 4.0f});
    fill(B, {5.0f, 6.0f, 7.0f, 8.0f});

    cme::detail::gemm_cpu_reference(A.view(), B.view(), C.view(), 1.0f, 0.0f);

    check_matrix(C, {19.0f, 22.0f, 43.0f, 50.0f});
}

TEST_CASE("CPU reference computes a hand-checked 3x2 by 2x4 product") {
    cme::Matrix A(3, 2);
    cme::Matrix B(2, 4);
    cme::Matrix C(3, 4);
    fill(A, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    fill(B, {1.0f, 0.0f, 2.0f, -1.0f, 0.0f, 1.0f, 3.0f, 2.0f});

    cme::detail::gemm_cpu_reference(A.view(), B.view(), C.view(), 1.0f, 0.0f);

    check_matrix(C, {1.0f, 2.0f, 8.0f, 3.0f, 3.0f, 4.0f, 18.0f, 5.0f, 5.0f, 6.0f, 28.0f, 7.0f});
}

TEST_CASE("CPU reference does not read C when beta is zero") {
    cme::Matrix A(1, 1);
    cme::Matrix B(1, 1);
    cme::Matrix C(1, 1);
    fill(A, {2.0f});
    fill(B, {3.0f});
    C(0, 0) = std::numeric_limits<float>::quiet_NaN();

    cme::detail::gemm_cpu_reference(A.view(), B.view(), C.view(), 1.0f, 0.0f);

    CHECK(std::isfinite(C(0, 0)));
    CHECK(C(0, 0) == 6.0f);
}

TEST_CASE("CPU reference returns beta times C when alpha is zero") {
    cme::Matrix A(2, 2);
    cme::Matrix B(2, 2);
    cme::Matrix C(2, 2);
    fill(A, {1.0f, 2.0f, 3.0f, 4.0f});
    fill(B, {5.0f, 6.0f, 7.0f, 8.0f});
    fill(C, {4.0f, 5.0f, 6.0f, 7.0f});

    cme::detail::gemm_cpu_reference(A.view(), B.view(), C.view(), 0.0f, 2.5f);

    check_matrix(C, {10.0f, 12.5f, 15.0f, 17.5f});
}
