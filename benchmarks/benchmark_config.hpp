#pragma once

#include <vector>

struct BenchmarkConfig {
    int warmup_iterations = 5;
    int measured_iterations = 30;
    std::vector<int> sizes = {256, 512, 1024, 2048, 4096};
};
