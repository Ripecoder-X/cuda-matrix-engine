#include "benchmark_gemm.hpp"

#include <cuda_runtime.h>

#if defined(CME_WITH_CUBLAS)
#include <cublas_v2.h>
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "cme/gemm.hpp"
#include "cme/gemm_config.hpp"
#include "cme/matrix.hpp"
#include "cme/matrix_view.hpp"
#include "cuda/device_buffer.hpp"
#include "cuda/gemm_dispatch.hpp"

#include "benchmark_cuda_util.hpp"

using cme::bench::append_gpu_reading;
using cme::bench::fill_values;
using cme::bench::require_cuda;
using cme::bench::start_gpu_readings;

namespace {

struct AlgorithmSpec {
    const char* name;
    cme::GemmAlgorithm algorithm;
    int tile_size;
};

const AlgorithmSpec kDeviceAlgorithms[] = {
    {"CUDA_NAIVE", cme::GemmAlgorithm::CUDA_NAIVE, 0},
    {"CUDA_TILED", cme::GemmAlgorithm::CUDA_TILED, 16},
    {"CUDA_TILED", cme::GemmAlgorithm::CUDA_TILED, 32},
    {"CUDA_OPTIMIZED", cme::GemmAlgorithm::CUDA_OPTIMIZED, 16},
    {"CUDA_OPTIMIZED", cme::GemmAlgorithm::CUDA_OPTIMIZED, 32},
};

void require_status(const cme::Status& status, const char* what) {
    if (!status.is_ok()) {
        throw std::runtime_error(std::string(what) + ": " + status.message());
    }
}

#if defined(CME_WITH_CUBLAS)

void require_cublas(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": " + cublasGetStatusString(status));
    }
}

class CublasHandle {
public:
    CublasHandle() {
        require_cublas(cublasCreate(&handle_), "cublasCreate");
    }

    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }

    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;

    cublasHandle_t get() const { return handle_; }

private:
    cublasHandle_t handle_ = nullptr;
};

std::vector<double> measure_cublas_launches(cublasHandle_t handle,
                                            int m,
                                            int n,
                                            int k,
                                            const float* a,
                                            const float* b,
                                            float* c,
                                            int warmup_iterations,
                                            int measured_iterations) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int i = 0; i < warmup_iterations; ++i) {
        require_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k,
                                   &beta, c, n),
                       "cuBLAS warm-up launch");
    }
    require_cuda(cudaDeviceSynchronize(), "cuBLAS warm-up synchronize");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    require_cuda(cudaEventCreate(&start), "cuBLAS cudaEventCreate start");
    require_cuda(cudaEventCreate(&stop), "cuBLAS cudaEventCreate stop");

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(measured_iterations));
    for (int i = 0; i < measured_iterations; ++i) {
        require_cuda(cudaEventRecord(start, nullptr), "cuBLAS cudaEventRecord start");
        require_cublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k,
                                   &beta, c, n),
                       "cuBLAS measured launch");
        require_cuda(cudaEventRecord(stop, nullptr), "cuBLAS cudaEventRecord stop");
        require_cuda(cudaEventSynchronize(stop), "cuBLAS cudaEventSynchronize");
        float elapsed_ms = 0.0f;
        require_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cuBLAS cudaEventElapsedTime");
        samples.push_back(static_cast<double>(elapsed_ms));
    }

    require_cuda(cudaEventDestroy(start), "cuBLAS cudaEventDestroy start");
    require_cuda(cudaEventDestroy(stop), "cuBLAS cudaEventDestroy stop");
    return samples;
}

#endif

std::vector<double> measure_kernel_launches(const cme::detail::DeviceGemmArgs& args,
                                            cme::GemmAlgorithm algorithm,
                                            int warmup_iterations,
                                            int measured_iterations) {
    for (int i = 0; i < warmup_iterations; ++i) {
        require_status(cme::detail::gemm_device(algorithm, args, nullptr), "warm-up launch");
    }
    require_cuda(cudaDeviceSynchronize(), "warm-up synchronize");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    require_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    require_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(measured_iterations));
    for (int i = 0; i < measured_iterations; ++i) {
        require_cuda(cudaEventRecord(start, nullptr), "cudaEventRecord start");
        require_status(cme::detail::gemm_device(algorithm, args, nullptr), "measured launch");
        require_cuda(cudaEventRecord(stop, nullptr), "cudaEventRecord stop");
        require_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize");
        float elapsed_ms = 0.0f;
        require_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
        samples.push_back(static_cast<double>(elapsed_ms));
    }

    require_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    require_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    return samples;
}

}  // namespace

std::string cuda_runtime_version_string() {
    int version = 0;
    if (cudaRuntimeGetVersion(&version) != cudaSuccess) {
        return "NOT MEASURED";
    }
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    return std::to_string(major) + "." + std::to_string(minor);
}

void run_kernel_only_benchmarks(const BenchmarkConfig& config,
                                CsvWriter& writer,
                                const RowContext& context) {
#if defined(CME_WITH_CUBLAS)
    CublasHandle cublas_handle;
    require_cublas(cublasSetStream(cublas_handle.get(), nullptr), "cublasSetStream");
#endif

    for (int size : config.sizes) {
        const int M = size;
        const int N = size;
        const int K = size;
        const std::size_t a_count =
            static_cast<std::size_t>(M) * static_cast<std::size_t>(K);
        const std::size_t b_count =
            static_cast<std::size_t>(K) * static_cast<std::size_t>(N);
        const std::size_t c_count =
            static_cast<std::size_t>(M) * static_cast<std::size_t>(N);

        std::vector<float> host_a(a_count);
        std::vector<float> host_b(b_count);
        std::vector<float> host_c(c_count, 0.0f);
        fill_values(host_a.data(), a_count, 1U);
        fill_values(host_b.data(), b_count, 2U);

        cme::detail::DeviceBuffer device_a;
        cme::detail::DeviceBuffer device_b;
        cme::detail::DeviceBuffer device_c;
        require_status(cme::detail::DeviceBuffer::create(a_count, &device_a),
                       "allocate device A");
        require_status(cme::detail::DeviceBuffer::create(b_count, &device_b),
                       "allocate device B");
        require_status(cme::detail::DeviceBuffer::create(c_count, &device_c),
                       "allocate device C");
        require_status(device_a.copy_from_host(host_a.data(), a_count), "copy A to device");
        require_status(device_b.copy_from_host(host_b.data(), b_count), "copy B to device");
        require_status(device_c.copy_from_host(host_c.data(), c_count), "copy C to device");

        for (const AlgorithmSpec& spec : kDeviceAlgorithms) {
            cme::detail::DeviceGemmArgs args;
            args.M = M;
            args.N = N;
            args.K = K;
            args.A = device_a.data();
            args.lda = K;
            args.B = device_b.data();
            args.ldb = N;
            args.C = device_c.data();
            args.ldc = N;
            args.alpha = 1.0f;
            args.beta = 0.0f;
            args.tile_size = spec.tile_size;

            std::vector<GpuReading> readings = start_gpu_readings();
            const std::vector<double> samples = measure_kernel_launches(
                args, spec.algorithm, config.warmup_iterations, config.measured_iterations);
            append_gpu_reading(&readings);

            const SampleStats stats = summarize_samples(samples);
            write_measurement_row(writer, context, spec.name, M, N, K, spec.tile_size,
                                  "kernel_only", config.measured_iterations, stats,
                                  summarize_gpu_readings(readings));

            std::fprintf(stderr, "kernel_only %s tile=%d %dx%dx%d median=%.3f ms\n",
                         spec.name, spec.tile_size, M, N, K, stats.median);
        }

#if defined(CME_WITH_CUBLAS)
        std::vector<GpuReading> cublas_readings = start_gpu_readings();
        const std::vector<double> cublas_samples = measure_cublas_launches(
            cublas_handle.get(), M, N, K, device_a.data(), device_b.data(), device_c.data(),
            config.warmup_iterations, config.measured_iterations);
        append_gpu_reading(&cublas_readings);

        const SampleStats cublas_stats = summarize_samples(cublas_samples);
        write_measurement_row(writer, context, "CUBLAS_REFERENCE", M, N, K, 0, "kernel_only",
                              config.measured_iterations, cublas_stats,
                              summarize_gpu_readings(cublas_readings));

        std::fprintf(stderr, "kernel_only CUBLAS_REFERENCE tile=0 %dx%dx%d median=%.3f ms\n",
                     M, N, K, cublas_stats.median);
#endif
    }
}

void run_end_to_end_benchmarks(const BenchmarkConfig& config,
                               CsvWriter& writer,
                               const RowContext& context) {
    for (int size : config.sizes) {
        const int M = size;
        const int N = size;
        const int K = size;

        cme::Matrix a(M, K);
        cme::Matrix b(K, N);
        cme::Matrix c(M, N);
        fill_values(a.data(), static_cast<std::size_t>(M) * static_cast<std::size_t>(K), 3U);
        fill_values(b.data(), static_cast<std::size_t>(K) * static_cast<std::size_t>(N), 4U);

        struct EndToEndSpec {
            const char* name;
            cme::GemmConfig config;
        };

        std::vector<EndToEndSpec> specs;
        if (size <= 512) {
            specs.push_back({"CPU_REFERENCE", {cme::GemmAlgorithm::CPU_REFERENCE, 0}});
        }
        for (const AlgorithmSpec& spec : kDeviceAlgorithms) {
            specs.push_back({spec.name, {spec.algorithm, spec.tile_size}});
        }

        for (const EndToEndSpec& spec : specs) {
            for (int i = 0; i < config.warmup_iterations; ++i) {
                require_status(cme::gemm(a.view(), b.view(), c.view(), 1.0f, 0.0f, spec.config),
                               "end-to-end warm-up");
            }

            std::vector<GpuReading> readings = start_gpu_readings();
            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(config.measured_iterations));
            for (int i = 0; i < config.measured_iterations; ++i) {
                const auto start = std::chrono::steady_clock::now();
                require_status(cme::gemm(a.view(), b.view(), c.view(), 1.0f, 0.0f, spec.config),
                               "end-to-end measured");
                const auto stop = std::chrono::steady_clock::now();
                const std::chrono::duration<double, std::milli> elapsed = stop - start;
                samples.push_back(elapsed.count());
            }
            append_gpu_reading(&readings);

            const SampleStats stats = summarize_samples(samples);
            write_measurement_row(writer, context, spec.name, M, N, K, spec.config.tile_size,
                                  "end_to_end", config.measured_iterations, stats,
                                  summarize_gpu_readings(readings));

            std::fprintf(stderr, "end_to_end %s tile=%d %dx%dx%d median=%.3f ms\n",
                         spec.name, spec.config.tile_size, M, N, K, stats.median);
        }
    }
}
