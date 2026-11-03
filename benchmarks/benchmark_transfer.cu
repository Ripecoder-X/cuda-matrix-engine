#include "benchmark_gemm.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <vector>

#include "benchmark_cuda_util.hpp"

using cme::bench::DeviceAllocation;
using cme::bench::append_gpu_reading;
using cme::bench::fill_values;
using cme::bench::require_cuda;
using cme::bench::start_gpu_readings;

void run_transfer_benchmarks(const BenchmarkConfig& config,
                             CsvWriter& writer,
                             const RowContext& context) {
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
        fill_values(host_a.data(), a_count, 5U);
        fill_values(host_b.data(), b_count, 6U);

        DeviceAllocation device_a;
        DeviceAllocation device_b;
        DeviceAllocation device_c;
        require_cuda(device_a.allocate(a_count), "allocate transfer A");
        require_cuda(device_b.allocate(b_count), "allocate transfer B");
        require_cuda(device_c.allocate(c_count), "allocate transfer C");

        for (int i = 0; i < config.warmup_iterations; ++i) {
            require_cuda(cudaMemcpy(device_a.data(), host_a.data(), a_count * sizeof(float),
                                    cudaMemcpyHostToDevice),
                         "warm-up H2D A");
            require_cuda(cudaMemcpy(device_b.data(), host_b.data(), b_count * sizeof(float),
                                    cudaMemcpyHostToDevice),
                         "warm-up H2D B");
            require_cuda(cudaMemcpy(host_c.data(), device_c.data(), c_count * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "warm-up D2H C");
        }
        require_cuda(cudaDeviceSynchronize(), "transfer warm-up synchronize");

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        require_cuda(cudaEventCreate(&start), "cudaEventCreate start");
        require_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

        std::vector<GpuReading> h2d_readings = start_gpu_readings();
        std::vector<double> h2d_samples;
        h2d_samples.reserve(static_cast<std::size_t>(config.measured_iterations));
        for (int i = 0; i < config.measured_iterations; ++i) {
            require_cuda(cudaEventRecord(start, nullptr), "H2D start event");
            require_cuda(cudaMemcpy(device_a.data(), host_a.data(), a_count * sizeof(float),
                                    cudaMemcpyHostToDevice),
                         "measured H2D A");
            require_cuda(cudaMemcpy(device_b.data(), host_b.data(), b_count * sizeof(float),
                                    cudaMemcpyHostToDevice),
                         "measured H2D B");
            require_cuda(cudaEventRecord(stop, nullptr), "H2D stop event");
            require_cuda(cudaEventSynchronize(stop), "H2D event synchronize");
            float elapsed_ms = 0.0f;
            require_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "H2D elapsed");
            h2d_samples.push_back(static_cast<double>(elapsed_ms));
        }
        append_gpu_reading(&h2d_readings);

        std::vector<GpuReading> d2h_readings = start_gpu_readings();
        std::vector<double> d2h_samples;
        d2h_samples.reserve(static_cast<std::size_t>(config.measured_iterations));
        for (int i = 0; i < config.measured_iterations; ++i) {
            require_cuda(cudaEventRecord(start, nullptr), "D2H start event");
            require_cuda(cudaMemcpy(host_c.data(), device_c.data(), c_count * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "measured D2H C");
            require_cuda(cudaEventRecord(stop, nullptr), "D2H stop event");
            require_cuda(cudaEventSynchronize(stop), "D2H event synchronize");
            float elapsed_ms = 0.0f;
            require_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "D2H elapsed");
            d2h_samples.push_back(static_cast<double>(elapsed_ms));
        }
        append_gpu_reading(&d2h_readings);

        require_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
        require_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");

        const SampleStats h2d_stats = summarize_samples(h2d_samples);
        write_measurement_row(writer, context, "TRANSFER", M, N, K, 0, "h2d_transfer",
                              config.measured_iterations, h2d_stats,
                              summarize_gpu_readings(h2d_readings));

        const SampleStats d2h_stats = summarize_samples(d2h_samples);
        write_measurement_row(writer, context, "TRANSFER", M, N, K, 0, "d2h_transfer",
                              config.measured_iterations, d2h_stats,
                              summarize_gpu_readings(d2h_readings));

        std::fprintf(stderr, "transfer %dx%dx%d h2d=%.3f ms d2h=%.3f ms\n", M, N, K,
                     h2d_stats.median, d2h_stats.median);
    }
}
