#include <cmath>
#include <cstdint>
#include <iostream>

#include "cme/gemm.hpp"
#include "cme/matrix.hpp"
#include "cme/status.hpp"

namespace {

struct AlgorithmChoice {
    const char* name;
    cme::GemmAlgorithm algorithm;
    int tile_size;
};

const AlgorithmChoice kChoices[] = {
    {"CPU_REFERENCE", cme::GemmAlgorithm::CPU_REFERENCE, 16},
    {"CUDA_NAIVE", cme::GemmAlgorithm::CUDA_NAIVE, 16},
    {"CUDA_TILED(16)", cme::GemmAlgorithm::CUDA_TILED, 16},
    {"CUDA_TILED(32)", cme::GemmAlgorithm::CUDA_TILED, 32},
    {"CUDA_OPTIMIZED(16)", cme::GemmAlgorithm::CUDA_OPTIMIZED, 16},
    {"CUDA_OPTIMIZED(32)", cme::GemmAlgorithm::CUDA_OPTIMIZED, 32},
};

void fill_random(cme::Matrix& m, std::uint32_t seed) {
    std::uint32_t state = seed;
    for (int r = 0; r < m.rows(); ++r) {
        for (int c = 0; c < m.cols(); ++c) {
            state = state * 1664525U + 1013904223U;
            const float unit = static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
            m(r, c) = 2.0f * unit - 1.0f;
        }
    }
}

float max_abs_difference(const cme::Matrix& lhs, const cme::Matrix& rhs) {
    float maximum = 0.0f;
    for (int r = 0; r < lhs.rows(); ++r) {
        for (int c = 0; c < lhs.cols(); ++c) {
            maximum = std::fmax(maximum, std::fabs(lhs(r, c) - rhs(r, c)));
        }
    }
    return maximum;
}

}  // namespace

int main() {
    constexpr int M = 64;
    constexpr int K = 48;
    constexpr int N = 32;

    cme::Matrix A(M, K);
    cme::Matrix B(K, N);
    cme::Matrix reference(M, N);
    fill_random(A, 11);
    fill_random(B, 12);

    cme::GemmConfig reference_config;
    reference_config.algorithm = cme::GemmAlgorithm::CPU_REFERENCE;
    const cme::Status reference_status =
        cme::gemm(A.view(), B.view(), reference.view(), 1.0f, 0.0f, reference_config);
    if (!reference_status.is_ok()) {
        std::cerr << "reference gemm failed: " << reference_status.message() << '\n';
        return 1;
    }

    for (const AlgorithmChoice& choice : kChoices) {
        cme::Matrix C(M, N);
        cme::GemmConfig config;
        config.algorithm = choice.algorithm;
        config.tile_size = choice.tile_size;

        const cme::Status status = cme::gemm(A.view(), B.view(), C.view(), 1.0f, 0.0f, config);
        if (!status.is_ok()) {
            std::cerr << choice.name << " failed: " << status.message() << '\n';
            return 1;
        }
        std::cout << choice.name << ": max_abs_diff_from_CPU = "
                  << max_abs_difference(C, reference) << '\n';
    }
    return 0;
}
