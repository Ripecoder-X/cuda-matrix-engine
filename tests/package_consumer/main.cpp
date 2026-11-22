#include <iostream>

#include "cme/gemm.hpp"
#include "cme/gemm_config.hpp"
#include "cme/matrix.hpp"
#include "cme/status.hpp"

int main() {
    cme::Matrix a(2, 2);
    cme::Matrix b(2, 2);
    cme::Matrix c(2, 2);

    a(0, 0) = 1.0f;
    a(0, 1) = 2.0f;
    a(1, 0) = 3.0f;
    a(1, 1) = 4.0f;

    b(0, 0) = 5.0f;
    b(0, 1) = 6.0f;
    b(1, 0) = 7.0f;
    b(1, 1) = 8.0f;

    cme::GemmConfig config;
    config.algorithm = cme::GemmAlgorithm::CPU_REFERENCE;

    const cme::Status status = cme::gemm(a.view(), b.view(), c.view(), 1.0f, 0.0f, config);
    if (!status.is_ok()) {
        std::cerr << "gemm failed: " << status.message() << '\n';
        return 1;
    }

    if (c(0, 0) != 19.0f || c(0, 1) != 22.0f || c(1, 0) != 43.0f || c(1, 1) != 50.0f) {
        std::cerr << "unexpected result: " << c(0, 0) << ' ' << c(0, 1) << ' '
                  << c(1, 0) << ' ' << c(1, 1) << '\n';
        return 2;
    }

    std::cout << "external consumer ok\n";
    return 0;
}
