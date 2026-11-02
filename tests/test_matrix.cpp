#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "cme/matrix.hpp"
#include "cme/matrix_view.hpp"

static_assert(
    std::is_convertible_v<cme::MatrixView<float>, cme::MatrixView<const float>>,
    "MatrixView<float> must convert to MatrixView<const float>");
static_assert(
    !std::is_convertible_v<cme::MatrixView<const float>, cme::MatrixView<float>>,
    "MatrixView<const float> must not convert to MatrixView<float>");

TEST_CASE("Matrix reports dimensions and zero-initializes") {
    cme::Matrix m(2, 3);
    CHECK(m.rows() == 2);
    CHECK(m.cols() == 3);
    CHECK(m.stride() == 3);
    for (int r = 0; r < m.rows(); ++r) {
        for (int c = 0; c < m.cols(); ++c) {
            CHECK(m(r, c) == 0.0f);
        }
    }
}

TEST_CASE("Matrix indexing is row-major") {
    cme::Matrix m(2, 3);
    m(0, 0) = 1.0f;
    m(0, 2) = 3.0f;
    m(1, 1) = 5.0f;

    CHECK(m.data()[0] == 1.0f);
    CHECK(m.data()[2] == 3.0f);
    CHECK(m.data()[4] == 5.0f);
    CHECK(m(1, 1) == 5.0f);

    const cme::Matrix& cm = m;
    CHECK(cm(0, 2) == 3.0f);
}

TEST_CASE("Matrix::view has stride equal to cols and shares storage") {
    cme::Matrix m(3, 4);
    m(1, 2) = 7.5f;

    cme::MatrixView<float> view = m.view();
    CHECK(view.data == m.data());
    CHECK(view.rows == 3);
    CHECK(view.cols == 4);
    CHECK(view.stride == m.cols());
    CHECK(view.is_valid());
    CHECK(view(1, 2) == 7.5f);

    view(0, 0) = 2.0f;
    CHECK(m(0, 0) == 2.0f);
}

TEST_CASE("Matrix converts to a const view") {
    cme::Matrix m(2, 2);
    m(0, 1) = 4.0f;

    cme::MatrixView<const float> const_view = m.view();
    CHECK(const_view.data == m.data());
    CHECK(const_view(0, 1) == 4.0f);
}

TEST_CASE("Matrix rejects non-positive dimensions") {
    REQUIRE_THROWS_AS(cme::Matrix(0, 3), std::invalid_argument);
    REQUIRE_THROWS_AS(cme::Matrix(3, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(cme::Matrix(-1, 2), std::invalid_argument);
    REQUIRE_THROWS_AS(cme::Matrix(2, -1), std::invalid_argument);

    try {
        cme::Matrix bad(0, 3);
        FAIL("expected std::invalid_argument");
    } catch (const std::invalid_argument& e) {
        const std::string message = e.what();
        CHECK(message.find("rows=0") != std::string::npos);
        CHECK(message.find("cols=3") != std::string::npos);
    }
}

TEST_CASE("MatrixView validity rules") {
    cme::Matrix m(2, 2);
    cme::MatrixView<float> view = m.view();
    CHECK(view.is_valid());

    cme::MatrixView<float> null_data = view;
    null_data.data = nullptr;
    CHECK_FALSE(null_data.is_valid());

    cme::MatrixView<float> zero_rows = view;
    zero_rows.rows = 0;
    CHECK_FALSE(zero_rows.is_valid());

    cme::MatrixView<float> short_stride = view;
    short_stride.stride = short_stride.cols - 1;
    CHECK_FALSE(short_stride.is_valid());
}
