#include "cme/matrix.hpp"

#include <stdexcept>
#include <string>

namespace cme {

Matrix::Matrix(int rows, int cols) : rows_(rows), cols_(cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument("Matrix dimensions must be positive (rows=" +
                                    std::to_string(rows) + ", cols=" + std::to_string(cols) + ")");
    }
    storage_.resize(static_cast<std::size_t>(rows_) * static_cast<std::size_t>(cols_), 0.0f);
}

int Matrix::rows() const noexcept { return rows_; }

int Matrix::cols() const noexcept { return cols_; }

int Matrix::stride() const noexcept { return cols_; }

float& Matrix::operator()(int r, int c) {
    return storage_[static_cast<std::size_t>(r) * static_cast<std::size_t>(stride()) +
                    static_cast<std::size_t>(c)];
}

const float& Matrix::operator()(int r, int c) const {
    return storage_[static_cast<std::size_t>(r) * static_cast<std::size_t>(stride()) +
                    static_cast<std::size_t>(c)];
}

MatrixView<float> Matrix::view() noexcept { return MatrixView<float>{data(), rows_, cols_, cols_}; }

MatrixView<const float> Matrix::view() const noexcept {
    return MatrixView<const float>{data(), rows_, cols_, cols_};
}

float* Matrix::data() noexcept { return storage_.data(); }

const float* Matrix::data() const noexcept { return storage_.data(); }

}  // namespace cme
