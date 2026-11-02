#include <cmath>
#include <iostream>

#include "cme/gemm.hpp"
#include "cme/matrix.hpp"
#include "cme/matrix_view.hpp"
#include "cme/status.hpp"

namespace {

void fill_view(cme::MatrixView<float> view) {
    for (int r = 0; r < view.rows; ++r) {
        for (int c = 0; c < view.cols; ++c) {
            view(r, c) = 0.05f * static_cast<float>(r + 1) - 0.01f * static_cast<float>(c + 1);
        }
    }
}

}  // namespace

int main() {
    constexpr int M = 17;
    constexpr int K = 9;
    constexpr int N = 33;

    cme::Matrix a_storage(24, 40);
    cme::Matrix b_storage(16, 48);
    cme::Matrix c_storage(24, 48);

    cme::MatrixView<float> A{a_storage.data() + 3 * 40 + 2, M, K, 40};
    cme::MatrixView<float> B{b_storage.data() + 4 * 48 + 5, K, N, 48};
    cme::MatrixView<float> C{c_storage.data() + 3 * 48 + 2, M, N, 48};
    fill_view(A);
    fill_view(B);
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            C(r, c) = 0.0f;
        }
    }

    cme::GemmConfig config;
    config.algorithm = cme::GemmAlgorithm::CUDA_TILED;
    config.tile_size = 32;

    const cme::Status status = cme::gemm(A, B, C, 1.0f, 0.0f, config);
    if (!status.is_ok()) {
        std::cerr << "strided gemm failed: " << status.message() << '\n';
        return 1;
    }

    cme::Matrix a_packed(M, K);
    cme::Matrix b_packed(K, N);
    cme::Matrix reference(M, N);
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < K; ++c) {
            a_packed(r, c) = A(r, c);
        }
    }
    for (int r = 0; r < K; ++r) {
        for (int c = 0; c < N; ++c) {
            b_packed(r, c) = B(r, c);
        }
    }

    cme::GemmConfig reference_config;
    reference_config.algorithm = cme::GemmAlgorithm::CPU_REFERENCE;
    const cme::Status reference_status = cme::gemm(a_packed.view(), b_packed.view(),
                                                   reference.view(), 1.0f, 0.0f,
                                                   reference_config);
    if (!reference_status.is_ok()) {
        std::cerr << "reference gemm failed: " << reference_status.message() << '\n';
        return 1;
    }

    float maximum = 0.0f;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            maximum = std::fmax(maximum, std::fabs(C(r, c) - reference(r, c)));
        }
    }

    std::cout << "strided CUDA_TILED(32) max_abs_diff_from_CPU = " << maximum << '\n';
    return maximum <= 1e-3f ? 0 : 2;
}
