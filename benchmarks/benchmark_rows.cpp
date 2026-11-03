#include "benchmark_rows.hpp"

#include <iomanip>
#include <sstream>
#include <vector>

namespace {

std::string fixed(double value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string optional_metric(bool valid, double value, int precision) {
    return valid ? fixed(value, precision) : "NOT MEASURED";
}

}  // namespace

RowContext make_row_context() {
    RowContext context;
    context.gpu_name = query_gpu_name();
    context.driver_version = query_driver_version();
    context.cuda_runtime_version = cuda_runtime_version_string();
    return context;
}

void write_measurement_row(CsvWriter& writer,
                           const RowContext& context,
                           const std::string& algorithm,
                           int M,
                           int N,
                           int K,
                           int tile_size,
                           const std::string& measurement,
                           int iterations,
                           const SampleStats& stats,
                           const GpuSummary& gpu) {
    const bool is_transfer = measurement == "h2d_transfer" || measurement == "d2h_transfer";
    const double operations = is_transfer
                                  ? 0.0
                                  : 2.0 * static_cast<double>(M) * static_cast<double>(N) *
                                        static_cast<double>(K);
    const double gflops =
        stats.median > 0.0 ? operations / (stats.median * 1.0e-3) / 1.0e9 : 0.0;

    std::vector<std::string> fields;
    fields.reserve(19);
    fields.push_back(utc_timestamp());
    fields.push_back(algorithm);
    fields.push_back(std::to_string(M));
    fields.push_back(std::to_string(N));
    fields.push_back(std::to_string(K));
    fields.push_back(std::to_string(tile_size));
    fields.push_back(measurement);
    fields.push_back(std::to_string(iterations));
    fields.push_back(fixed(stats.median, 6));
    fields.push_back(fixed(stats.min, 6));
    fields.push_back(fixed(stats.max, 6));
    fields.push_back(fixed(stats.iqr, 6));
    fields.push_back(fixed(gflops, 6));
    fields.push_back(context.gpu_name);
    fields.push_back(context.driver_version);
    fields.push_back(context.cuda_runtime_version);
    fields.push_back(optional_metric(gpu.valid, gpu.clock_mhz, 3));
    fields.push_back(optional_metric(gpu.valid, gpu.temperature_c, 3));
    fields.push_back(optional_metric(gpu.valid, gpu.power_w, 3));
    writer.write_row(fields);
}
