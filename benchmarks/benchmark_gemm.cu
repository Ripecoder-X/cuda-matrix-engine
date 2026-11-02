#include "benchmark_config.hpp"
#include "csv_writer.hpp"
#include "cme/gemm.hpp"
#include "cme/gemm_config.hpp"
#include "cme/matrix.hpp"
#include "cme/status.hpp"
#include "cuda/device_buffer.hpp"
#include "cuda/gemm_dispatch.hpp"

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

struct KernelConfig {
    const char* name;
    cme::GemmAlgorithm algorithm;
    int tile_size;
};

const KernelConfig kKernelConfigs[] = {
    {"CUDA_NAIVE", cme::GemmAlgorithm::CUDA_NAIVE, 0},
    {"CUDA_TILED", cme::GemmAlgorithm::CUDA_TILED, 16},
    {"CUDA_TILED", cme::GemmAlgorithm::CUDA_TILED, 32},
    {"CUDA_OPTIMIZED", cme::GemmAlgorithm::CUDA_OPTIMIZED, 16},
    {"CUDA_OPTIMIZED", cme::GemmAlgorithm::CUDA_OPTIMIZED, 32},
};

struct RunStats {
    float median_ms = 0.0f;
    float min_ms = 0.0f;
    float max_ms = 0.0f;
    float iqr_ms = 0.0f;
    double gflops_median = 0.0;
};

RunStats compute_stats(std::vector<float>& samples, int M, int N, int K) {
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

    // GFLOP/s = 2.0 * M * N * K / (median_ms * 1e-3) / 1e9
    double gflops = 0.0;
    if (median_ms > 0.0f) {
        double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        double seconds = static_cast<double>(median_ms) * 1e-3;
        gflops = (flops / seconds) / 1e9;
    }

    return RunStats{median_ms, min_ms, max_ms, iqr_ms, gflops};
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

void run_kernel_benchmarks(const BenchmarkConfig& config, CsvWriter& writer) {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    const std::string gpu_name = get_gpu_name();
    const std::string cuda_rt_ver = get_cuda_runtime_version();

    for (int size : config.sizes) {
        const int M = size;
        const int N = size;
        const int K = size;
        const int lda = K;
        const int ldb = N;
        const int ldc = N;

        cme::detail::DeviceBuffer buf_a;
        cme::detail::DeviceBuffer buf_b;
        cme::detail::DeviceBuffer buf_c;

        cme::Status sa = cme::detail::DeviceBuffer::create(static_cast<std::size_t>(M) * K, &buf_a);
        cme::Status sb = cme::detail::DeviceBuffer::create(static_cast<std::size_t>(K) * N, &buf_b);
        cme::Status sc = cme::detail::DeviceBuffer::create(static_cast<std::size_t>(M) * N, &buf_c);

        if (!sa.is_ok() || !sb.is_ok() || !sc.is_ok()) {
            std::cerr << "Failed to allocate device buffers for size " << size << std::endl;
            continue;
        }

        std::vector<float> host_a(static_cast<std::size_t>(M) * K, 1.0f);
        std::vector<float> host_b(static_cast<std::size_t>(K) * N, 0.5f);
        std::vector<float> host_c(static_cast<std::size_t>(M) * N, 0.0f);

        (void)buf_a.copy_from_host(host_a.data(), host_a.size());
        (void)buf_b.copy_from_host(host_b.data(), host_b.size());
        (void)buf_c.copy_from_host(host_c.data(), host_c.size());

        for (const auto& kc : kKernelConfigs) {
            cme::detail::DeviceGemmArgs args;
            args.M = M;
            args.N = N;
            args.K = K;
            args.A = buf_a.data();
            args.lda = lda;
            args.B = buf_b.data();
            args.ldb = ldb;
            args.C = buf_c.data();
            args.ldc = ldc;
            args.alpha = 1.0f;
            args.beta = 0.0f;
            args.tile_size = kc.tile_size;

            // Warmup iterations
            for (int w = 0; w < config.warmup_iterations; ++w) {
                cme::Status st = cme::detail::gemm_device(kc.algorithm, args, stream);
                if (!st.is_ok()) {
                    std::cerr << "Warmup launch failed for " << kc.name << ": " << st.message() << std::endl;
                    break;
                }
            }
            cudaStreamSynchronize(stream);

            TelemetrySample t_before = query_telemetry();

            // Timing with CUDA events
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            std::vector<float> samples;
            samples.reserve(config.measured_iterations);

            for (int i = 0; i < config.measured_iterations; ++i) {
                cudaEventRecord(start, stream);
                cme::Status st = cme::detail::gemm_device(kc.algorithm, args, stream);
                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);

                if (!st.is_ok()) {
                    std::cerr << "Measurement launch failed for " << kc.name << ": " << st.message() << std::endl;
                    break;
                }

                float elapsed_ms = 0.0f;
                cudaEventElapsedTime(&elapsed_ms, start, stop);
                samples.push_back(elapsed_ms);
            }

            TelemetrySample t_after = query_telemetry();

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            RunStats stats = compute_stats(samples, M, N, K);

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
                kc.name,
                std::to_string(M),
                std::to_string(N),
                std::to_string(K),
                std::to_string(kc.tile_size),
                "kernel_only",
                std::to_string(config.measured_iterations),
                format_float(stats.median_ms, 4),
                format_float(stats.min_ms, 4),
                format_float(stats.max_ms, 4),
                format_float(stats.iqr_ms, 4),
                format_float(stats.gflops_median, 2),
                gpu_name,
                driver_ver,
                cuda_rt_ver,
                sm_clock,
                temp,
                power
            };
            writer.write_row(row);

            std::cout << "[kernel_only] M=N=K=" << size << " | "
                      << std::setw(14) << kc.name
                      << " (tile=" << std::setw(2) << kc.tile_size << "): "
                      << "median = " << std::setw(8) << format_float(stats.median_ms, 4) << " ms, "
                      << "IQR = " << std::setw(6) << format_float(stats.iqr_ms, 4) << " ms, "
                      << "GFLOP/s = " << std::setw(8) << format_float(stats.gflops_median, 2)
                      << std::endl;
        }
    }

    cudaStreamDestroy(stream);
}

struct EndToEndConfig {
    const char* name;
    cme::GemmAlgorithm algorithm;
    int tile_size;
};

const EndToEndConfig kEndToEndConfigs[] = {
    {"CPU_REFERENCE", cme::GemmAlgorithm::CPU_REFERENCE, 0},
    {"CUDA_NAIVE", cme::GemmAlgorithm::CUDA_NAIVE, 0},
    {"CUDA_TILED", cme::GemmAlgorithm::CUDA_TILED, 16},
    {"CUDA_TILED", cme::GemmAlgorithm::CUDA_TILED, 32},
    {"CUDA_OPTIMIZED", cme::GemmAlgorithm::CUDA_OPTIMIZED, 16},
    {"CUDA_OPTIMIZED", cme::GemmAlgorithm::CUDA_OPTIMIZED, 32},
};

void run_end_to_end_benchmarks(const BenchmarkConfig& config, CsvWriter& writer) {
    const std::string gpu_name = get_gpu_name();
    const std::string cuda_rt_ver = get_cuda_runtime_version();

    for (int size : config.sizes) {
        const int M = size;
        const int N = size;
        const int K = size;

        cme::Matrix A(M, K);
        cme::Matrix B(K, N);
        cme::Matrix C(M, N);

        for (int i = 0; i < M * K; ++i) A.data()[i] = 1.0f;
        for (int i = 0; i < K * N; ++i) B.data()[i] = 0.5f;

        for (const auto& ec : kEndToEndConfigs) {
            cme::GemmConfig gemm_cfg;
            gemm_cfg.algorithm = ec.algorithm;
            gemm_cfg.tile_size = ec.tile_size;

            int warmup_iters = config.warmup_iterations;
            int measured_iters = config.measured_iterations;

            if (ec.algorithm == cme::GemmAlgorithm::CPU_REFERENCE) {
                if (size >= 2048) {
                    warmup_iters = 0;
                    measured_iters = 1;
                } else if (size >= 1024) {
                    warmup_iters = 1;
                    measured_iters = 3;
                }
            }

            for (int w = 0; w < warmup_iters; ++w) {
                (void)cme::gemm(A.view(), B.view(), C.view(), 1.0f, 0.0f, gemm_cfg);
                cudaDeviceSynchronize();
            }

            TelemetrySample t_before = query_telemetry();

            std::vector<float> samples;
            samples.reserve(measured_iters);

            for (int i = 0; i < measured_iters; ++i) {
                cudaDeviceSynchronize();
                auto start = std::chrono::steady_clock::now();
                cme::Status st = cme::gemm(A.view(), B.view(), C.view(), 1.0f, 0.0f, gemm_cfg);
                cudaDeviceSynchronize();
                auto stop = std::chrono::steady_clock::now();

                if (!st.is_ok()) {
                    std::cerr << "End-to-end launch failed for " << ec.name << ": " << st.message() << std::endl;
                    break;
                }

                float elapsed_ms = std::chrono::duration<float, std::milli>(stop - start).count();
                samples.push_back(elapsed_ms);
            }

            TelemetrySample t_after = query_telemetry();

            RunStats stats = compute_stats(samples, M, N, K);

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
                ec.name,
                std::to_string(M),
                std::to_string(N),
                std::to_string(K),
                std::to_string(ec.tile_size),
                "end_to_end",
                std::to_string(measured_iters),
                format_float(stats.median_ms, 4),
                format_float(stats.min_ms, 4),
                format_float(stats.max_ms, 4),
                format_float(stats.iqr_ms, 4),
                format_float(stats.gflops_median, 2),
                gpu_name,
                driver_ver,
                cuda_rt_ver,
                sm_clock,
                temp,
                power
            };
            writer.write_row(row);

            std::cout << "[end_to_end]  M=N=K=" << size << " | "
                      << std::setw(14) << ec.name
                      << " (tile=" << std::setw(2) << ec.tile_size << "): "
                      << "median = " << std::setw(8) << format_float(stats.median_ms, 4) << " ms, "
                      << "IQR = " << std::setw(6) << format_float(stats.iqr_ms, 4) << " ms, "
                      << "GFLOP/s = " << std::setw(8) << format_float(stats.gflops_median, 2)
                      << (ec.algorithm == cme::GemmAlgorithm::CPU_REFERENCE ? " [host reference]" : "")
                      << std::endl;
        }
    }
}

// Forward declaration from benchmark_transfer.cu
void run_transfer_benchmarks(const BenchmarkConfig& config, CsvWriter& writer);

