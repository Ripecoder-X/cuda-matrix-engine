#pragma once

#include <string>

#include "benchmark_common.hpp"
#include "csv_writer.hpp"

std::string cuda_runtime_version_string();

struct RowContext {
    std::string gpu_name;
    std::string driver_version;
    std::string cuda_runtime_version;
};

RowContext make_row_context();

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
                           const GpuSummary& gpu);
