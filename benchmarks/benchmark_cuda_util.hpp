#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark_common.hpp"

namespace cme::bench {

inline void require_cuda(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
}

inline void fill_values(float* values, std::size_t count, std::uint32_t seed) {
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < count; ++i) {
        state = state * 1664525U + 1013904223U;
        values[i] = static_cast<float>((state >> 8) & 0xFFFFU) / 65536.0f - 0.5f;
    }
}

inline std::vector<GpuReading> start_gpu_readings() {
    std::vector<GpuReading> readings;
    GpuReading reading;
    query_gpu_reading(&reading);
    readings.push_back(reading);
    return readings;
}

inline void append_gpu_reading(std::vector<GpuReading>* readings) {
    GpuReading reading;
    query_gpu_reading(&reading);
    readings->push_back(reading);
}

class DeviceAllocation {
public:
    DeviceAllocation() = default;
    ~DeviceAllocation() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }
    DeviceAllocation(const DeviceAllocation&) = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    cudaError_t allocate(std::size_t count) {
        return cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(float));
    }

    float* data() { return ptr_; }

private:
    float* ptr_ = nullptr;
};

}  // namespace cme::bench
