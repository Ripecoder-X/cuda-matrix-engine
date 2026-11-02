#pragma once
#include <cstddef>
#include <type_traits>

namespace cme {

// Non-owning, row-major view. Element (r, c) lives at data[r * stride + c].
// Requirements for a valid view: data != nullptr, rows > 0, cols > 0, stride >= cols.
template <typename T>
struct MatrixView {
    T* data = nullptr;
    int rows = 0;
    int cols = 0;
    int stride = 0;

    // Implicit conversion from a mutable view to a read-only view.
    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<T*, U*> && !std::is_same_v<T, U>>>
    operator MatrixView<U>() const noexcept {
        return MatrixView<U>{data, rows, cols, stride};
    }

    T& operator()(int r, int c) const {
        return data[static_cast<std::size_t>(r) * static_cast<std::size_t>(stride) +
                    static_cast<std::size_t>(c)];
    }

    bool is_valid() const noexcept {
        return data != nullptr && rows > 0 && cols > 0 && stride >= cols;
    }
};

}  // namespace cme
