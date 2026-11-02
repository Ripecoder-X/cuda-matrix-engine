#include <iostream>

#include "cme/gemm.hpp"
#include "cme/matrix.hpp"
#include "cme/status.hpp"

int main() {
    constexpr int M = 128;
    constexpr int K = 64;
    constexpr int N = 32;

    cme::Matrix A(M, K);
    cme::Matrix B(K, N);
    cme::Matrix C(M, N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            A(i, j) = 0.01f * static_cast<float>(i + 1) + 0.001f * static_cast<float>(j + 1);
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            B(i, j) = 0.02f * static_cast<float>(i + 1) - 0.003f * static_cast<float>(j + 1);
        }
    }

    cme::GemmConfig config;
    const cme::Status status = cme::gemm(A.view(), B.view(), C.view(), 1.0f, 0.0f, config);
    if (!status.is_ok()) {
        std::cerr << "gemm failed: " << status.message() << '\n';
        return 1;
    }

    std::cout << "C(0,0)..C(0,3):";
    for (int c = 0; c < 4; ++c) {
        std::cout << ' ' << C(0, c);
    }
    std::cout << '\n';
    return 0;
}
