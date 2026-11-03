#pragma once

#include "benchmark_config.hpp"
#include "benchmark_rows.hpp"
#include "csv_writer.hpp"

void run_kernel_only_benchmarks(const BenchmarkConfig& config,
                                CsvWriter& writer,
                                const RowContext& context);

void run_end_to_end_benchmarks(const BenchmarkConfig& config,
                               CsvWriter& writer,
                               const RowContext& context);

void run_transfer_benchmarks(const BenchmarkConfig& config,
                             CsvWriter& writer,
                             const RowContext& context);
