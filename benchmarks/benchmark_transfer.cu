#include "benchmark_config.hpp"
#include "csv_writer.hpp"
#include "cuda/device_buffer.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct RunStats {
    float median_ms = 0.0f;
    float min_ms = 0.0f;
    float max_ms = 0.0f;
    float iqr_ms = 0.0f;
    double gflops_median = 0.0;
};

RunStats compute_stats(std::vector<float>& samples) {
    if (samples.empty()) {
        return RunStats{};
    }

    std::sort(samples.begin(), samples.end());
    const std::size_t n = samples.size();
    float min_ms = samples.front();
    float max_ms = samples.back();

    auto median_of = [](const float* arr, std::size_t len) -> float {
        if (len == 0) return 0.0f;
        if (len % 2 == 1) return arr[len / 2];
        return 0.5f * (arr[len / 2 - 1] + arr[len / 2]);
    };

    float median_ms = median_of(samples.data(), n);

    float q1 = 0.0f;
    float q3 = 0.0f;
    if (n >= 4) {
        std::size_t half = n / 2;
        q1 = median_of(samples.data(), half);
        if (n % 2 == 0) {
            q3 = median_of(samples.data() + half, half);
        } else {
            q3 = median_of(samples.data() + half + 1, half);
        }
    }
    float iqr_ms = q3 - q1;

    return RunStats{median_ms, min_ms, max_ms, iqr_ms, 0.0};
}

std::string get_current_utc_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm gm{};
#if defined(_WIN32)
    gmtime_s(&gm, &tt);
#else
    gmtime_r(&tt, &gm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &gm);
    return std::string(buf);
}

struct TelemetrySample {
    std::string driver_version = "NOT MEASURED";
    float sm_clock_mhz = -1.0f;
    float temperature_c = -1.0f;
    float power_w = -1.0f;
    bool valid = false;
};

TelemetrySample query_telemetry() {
    TelemetrySample ts;
#if defined(_WIN32)
    FILE* pipe = _popen("nvidia-smi --query-gpu=driver_version,clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits", "r");
#else
    FILE* pipe = popen("nvidia-smi --query-gpu=driver_version,clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits", "r");
#endif
    if (!pipe) {
        return ts;
    }
    char buffer[256];
    if (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        std::stringstream ss(buffer);
        std::string d_ver, clk_str, temp_str, pwr_str;
        if (std::getline(ss, d_ver, ',') &&
            std::getline(ss, clk_str, ',') &&
            std::getline(ss, temp_str, ',') &&
            std::getline(ss, pwr_str)) {
            auto trim = [](std::string& s) {
                while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
                while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) s.pop_back();
            };
            trim(d_ver);
            trim(clk_str);
            trim(temp_str);
            trim(pwr_str);
            try {
                ts.driver_version = d_ver;
                ts.sm_clock_mhz = std::stof(clk_str);
                ts.temperature_c = std::stof(temp_str);
                ts.power_w = std::stof(pwr_str);
                ts.valid = true;
            } catch (...) {}
        }
    }
#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return ts;
}

std::string get_gpu_name() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        return std::string(prop.name);
    }
    return "Unknown GPU";
}

std::string get_cuda_runtime_version() {
    int rt_ver = 0;
    if (cudaRuntimeGetVersion(&rt_ver) == cudaSuccess) {
        return std::to_string(rt_ver / 1000) + "." + std::to_string((rt_ver % 1000) / 10);
    }
    return "Unknown";
}

std::string format_float(double val, int precision = 4) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(precision) << val;
    return ss.str();
}

}  // namespace

void run_transfer_benchmarks(const BenchmarkConfig& config, CsvWriter& writer) {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    const std::string gpu_name = get_gpu_name();
    const std::string cuda_rt_ver = get_cuda_runtime_version();

    for (int size : config.sizes) {
        const int M = size;
        const int N = size;
        const int K = size;

        const std::size_t count_a = static_cast<std::size_t>(M) * K;
        const std::size_t count_b = static_cast<std::size_t>(K) * N;
        const std::size_t count_c = static_cast<std::size_t>(M) * N;

        const std::size_t bytes_a = count_a * sizeof(float);
        const std::size_t bytes_b = count_b * sizeof(float);
        const std::size_t bytes_c = count_c * sizeof(float);

        cme::detail::DeviceBuffer buf_a;
        cme::detail::DeviceBuffer buf_b;
        cme::detail::DeviceBuffer buf_c;

        cme::Status sa = cme::detail::DeviceBuffer::create(count_a, &buf_a);
        cme::Status sb = cme::detail::DeviceBuffer::create(count_b, &buf_b);
        cme::Status sc = cme::detail::DeviceBuffer::create(count_c, &buf_c);

        if (!sa.is_ok() || !sb.is_ok() || !sc.is_ok()) {
            std::cerr << "Transfer benchmark: Failed to allocate device buffers for size " << size << std::endl;
            continue;
        }

        std::vector<float> host_a(count_a, 1.0f);
        std::vector<float> host_b(count_b, 0.5f);
        std::vector<float> host_c(count_c, 0.0f);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // 1. H2D Transfer of A and B
        {
            // Warmup
            for (int w = 0; w < config.warmup_iterations; ++w) {
                cudaMemcpyAsync(buf_a.data(), host_a.data(), bytes_a, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(buf_b.data(), host_b.data(), bytes_b, cudaMemcpyHostToDevice, stream);
            }
            cudaStreamSynchronize(stream);

            TelemetrySample t_before = query_telemetry();

            std::vector<float> samples;
            samples.reserve(config.measured_iterations);

            for (int i = 0; i < config.measured_iterations; ++i) {
                cudaEventRecord(start, stream);
                cudaMemcpyAsync(buf_a.data(), host_a.data(), bytes_a, cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(buf_b.data(), host_b.data(), bytes_b, cudaMemcpyHostToDevice, stream);
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);

                float elapsed_ms = 0.0f;
                cudaEventElapsedTime(&elapsed_ms, start, stop);
                samples.push_back(elapsed_ms);
            }

            TelemetrySample t_after = query_telemetry();

            RunStats stats = compute_stats(samples);

            std::string driver_ver = "NOT MEASURED";
            if (t_before.valid && !t_before.driver_version.empty()) {
                driver_ver = t_before.driver_version;
            } else if (t_after.valid && !t_after.driver_version.empty()) {
                driver_ver = t_after.driver_version;
            }

            std::string sm_clock = "NOT MEASURED";
            std::string temp = "NOT MEASURED";
            std::string power = "NOT MEASURED";

            if (t_before.valid && t_after.valid) {
                sm_clock = std::to_string(static_cast<int>(std::round(0.5f * (t_before.sm_clock_mhz + t_after.sm_clock_mhz))));
                temp = std::to_string(static_cast<int>(std::round(0.5f * (t_before.temperature_c + t_after.temperature_c))));
                power = format_float(0.5f * (t_before.power_w + t_after.power_w), 2);
            } else if (t_before.valid) {
                sm_clock = std::to_string(static_cast<int>(t_before.sm_clock_mhz));
                temp = std::to_string(static_cast<int>(t_before.temperature_c));
                power = format_float(t_before.power_w, 2);
            } else if (t_after.valid) {
                sm_clock = std::to_string(static_cast<int>(t_after.sm_clock_mhz));
                temp = std::to_string(static_cast<int>(t_after.temperature_c));
                power = format_float(t_after.power_w, 2);
            }

            const std::string timestamp = get_current_utc_timestamp();

            std::vector<std::string> row = {
                timestamp,
                "CUDA_MEMCPY",
                std::to_string(M),
                std::to_string(N),
                std::to_string(K),
                "0",
                "h2d_transfer",
                std::to_string(config.measured_iterations),
                format_float(stats.median_ms, 4),
                format_float(stats.min_ms, 4),
                format_float(stats.max_ms, 4),
                format_float(stats.iqr_ms, 4),
                "0.00",
                gpu_name,
                driver_ver,
                cuda_rt_ver,
                sm_clock,
                temp,
                power
            };
            writer.write_row(row);

            const double total_gb = static_cast<double>(bytes_a + bytes_b) / 1e9;
            const double bw_gbps = (stats.median_ms > 0.0f) ? (total_gb / (stats.median_ms * 1e-3)) : 0.0;

            std::cout << "[h2d_transfer] M=N=K=" << size << ": "
                      << "median = " << std::setw(8) << format_float(stats.median_ms, 4) << " ms, "
                      << "IQR = " << std::setw(6) << format_float(stats.iqr_ms, 4) << " ms, "
                      << "bandwidth = " << std::setw(6) << format_float(bw_gbps, 2) << " GB/s"
                      << std::endl;
        }

        // 2. D2H Transfer of C
        {
            // Warmup
            for (int w = 0; w < config.warmup_iterations; ++w) {
                cudaMemcpyAsync(host_c.data(), buf_c.data(), bytes_c, cudaMemcpyDeviceToHost, stream);
            }
            cudaStreamSynchronize(stream);

            TelemetrySample t_before = query_telemetry();

            std::vector<float> samples;
            samples.reserve(config.measured_iterations);

            for (int i = 0; i < config.measured_iterations; ++i) {
                cudaEventRecord(start, stream);
                cudaMemcpyAsync(host_c.data(), buf_c.data(), bytes_c, cudaMemcpyDeviceToHost, stream);
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);

                float elapsed_ms = 0.0f;
                cudaEventElapsedTime(&elapsed_ms, start, stop);
                samples.push_back(elapsed_ms);
            }

            TelemetrySample t_after = query_telemetry();

            RunStats stats = compute_stats(samples);

            std::string driver_ver = "NOT MEASURED";
            if (t_before.valid && !t_before.driver_version.empty()) {
                driver_ver = t_before.driver_version;
            } else if (t_after.valid && !t_after.driver_version.empty()) {
                driver_ver = t_after.driver_version;
            }

            std::string sm_clock = "NOT MEASURED";
            std::string temp = "NOT MEASURED";
            std::string power = "NOT MEASURED";

            if (t_before.valid && t_after.valid) {
                sm_clock = std::to_string(static_cast<int>(std::round(0.5f * (t_before.sm_clock_mhz + t_after.sm_clock_mhz))));
                temp = std::to_string(static_cast<int>(std::round(0.5f * (t_before.temperature_c + t_after.temperature_c))));
                power = format_float(0.5f * (t_before.power_w + t_after.power_w), 2);
            } else if (t_before.valid) {
                sm_clock = std::to_string(static_cast<int>(t_before.sm_clock_mhz));
                temp = std::to_string(static_cast<int>(t_before.temperature_c));
                power = format_float(t_before.power_w, 2);
            } else if (t_after.valid) {
                sm_clock = std::to_string(static_cast<int>(t_after.sm_clock_mhz));
                temp = std::to_string(static_cast<int>(t_after.temperature_c));
                power = format_float(t_after.power_w, 2);
            }

            const std::string timestamp = get_current_utc_timestamp();

            std::vector<std::string> row = {
                timestamp,
                "CUDA_MEMCPY",
                std::to_string(M),
                std::to_string(N),
                std::to_string(K),
                "0",
                "d2h_transfer",
                std::to_string(config.measured_iterations),
                format_float(stats.median_ms, 4),
                format_float(stats.min_ms, 4),
                format_float(stats.max_ms, 4),
                format_float(stats.iqr_ms, 4),
                "0.00",
                gpu_name,
                driver_ver,
                cuda_rt_ver,
                sm_clock,
                temp,
                power
            };
            writer.write_row(row);

            const double total_gb = static_cast<double>(bytes_c) / 1e9;
            const double bw_gbps = (stats.median_ms > 0.0f) ? (total_gb / (stats.median_ms * 1e-3)) : 0.0;

            std::cout << "[d2h_transfer] M=N=K=" << size << ": "
                      << "median = " << std::setw(8) << format_float(stats.median_ms, 4) << " ms, "
                      << "IQR = " << std::setw(6) << format_float(stats.iqr_ms, 4) << " ms, "
                      << "bandwidth = " << std::setw(6) << format_float(bw_gbps, 2) << " GB/s"
                      << std::endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    cudaStreamDestroy(stream);
}
