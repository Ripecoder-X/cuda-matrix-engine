#pragma once
#include <cstddef>
#include <vector>
#include "cme/matrix_view.hpp"

namespace cme {

// Owning, row-major, tightly packed host matrix (stride == cols).
// Constructor throws std::invalid_argument if rows <= 0 or cols <= 0.
// Elements are zero-initialized.
class Matrix {
public:
    Matrix(int rows, int cols);

    int rows() const noexcept;
    int cols() const noexcept;
    int stride() const noexcept;   // always == cols

    float& operator()(int r, int c);
    const float& operator()(int r, int c) const;

    MatrixView<float> view() noexcept;
    MatrixView<const float> view() const noexcept;

    float* data() noexcept;
    const float* data() const noexcept;

private:
    int rows_;
    int cols_;
    std::vector<float> storage_;
};

}  // namespace cme
