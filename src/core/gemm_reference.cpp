#include "core/gemm_reference.hpp"

namespace cme::detail {

void gemm_cpu_reference(MatrixView<const float> A,
                        MatrixView<const float> B,
                        MatrixView<float> C,
                        float alpha,
                        float beta) {
    const int M = C.rows;
    const int N = C.cols;
    const int K = A.cols;

    if (alpha == 0.0f) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C(i, j) = (beta == 0.0f) ? 0.0f : beta * C(i, j);
            }
        }
        return;
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<double>(A(i, k)) * static_cast<double>(B(k, j));
            }
            const float scaled = alpha * static_cast<float>(acc);
            C(i, j) = (beta == 0.0f) ? scaled : scaled + beta * C(i, j);
        }
    }
}

}  // namespace cme::detail
