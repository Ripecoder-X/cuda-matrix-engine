#pragma once

#include <string>
#include <vector>

struct SampleStats {
    double median = 0.0;
    double min = 0.0;
    double max = 0.0;
    double iqr = 0.0;
};

// Sorts the samples and reports the p50, minimum, maximum, and p75 - p25.
// Percentiles use linear interpolation between the closest ranks.
SampleStats summarize_samples(std::vector<double> samples);

std::string utc_timestamp();

struct GpuReading {
    bool valid = false;
    double clock_mhz = 0.0;
    double temperature_c = 0.0;
    double power_w = 0.0;
};

// Runs the nvidia-smi query required by the benchmark methodology.
// Returns false when the query fails or a field is not a number.
bool query_gpu_reading(GpuReading* out);

struct GpuSummary {
    bool valid = false;
    double clock_mhz = 0.0;
    double temperature_c = 0.0;
    double power_w = 0.0;
};

// Reports the median of the valid readings for each field.
GpuSummary summarize_gpu_readings(const std::vector<GpuReading>& readings);

std::string query_gpu_name();
std::string query_driver_version();
