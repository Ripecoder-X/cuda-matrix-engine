#pragma once
#include <cmath>
#include <cstdint>
#include "cme/matrix.hpp"

namespace cme::test {

// Fills with uniform values in [-1, 1] from a deterministic seed.
inline void fill_random(Matrix& m, std::uint32_t seed) {
    std::uint32_t state = seed;
    auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return state;
    };
    for (int r = 0; r < m.rows(); ++r) {
        for (int c = 0; c < m.cols(); ++c) {
            const std::uint32_t bits = next() & 0x00FFFFFFu;
            const float unit = static_cast<float>(bits) / static_cast<float>(0x00FFFFFFu);
            m(r, c) = unit * 2.0f - 1.0f;
        }
    }
}

// Tolerance rule used by ALL correctness tests (see QUALITY_STANDARDS.md):
//   |got - ref| <= atol_for(K) + rtol * |ref|
//   rtol = 1e-4,  atol_for(K) = 1e-5 * K
inline bool close_enough(float got, float ref, int K) {
    constexpr float rtol = 1e-4f;
    const float atol = 1e-5f * static_cast<float>(K);
    return std::fabs(got - ref) <= atol + rtol * std::fabs(ref);
}

}  // namespace cme::test
