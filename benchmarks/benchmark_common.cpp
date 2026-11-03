#include "benchmark_common.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <ctime>
#include <sstream>

namespace {

double percentile(const std::vector<double>& sorted, double fraction) {
    if (sorted.empty()) {
        return 0.0;
    }
    if (sorted.size() == 1) {
        return sorted.front();
    }
    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
    const double weight = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

std::string run_command(const std::string& command) {
    std::string output;
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return output;
    }
    std::array<char, 512> buffer{};
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
    pclose(pipe);
    return output;
}

std::string first_line(const std::string& text) {
    const std::size_t end = text.find('\n');
    return end == std::string::npos ? text : text.substr(0, end);
}

}  // namespace

SampleStats summarize_samples(std::vector<double> samples) {
    SampleStats stats;
    if (samples.empty()) {
        return stats;
    }
    std::sort(samples.begin(), samples.end());
    stats.median = percentile(samples, 0.50);
    stats.min = samples.front();
    stats.max = samples.back();
    stats.iqr = percentile(samples, 0.75) - percentile(samples, 0.25);
    return stats;
}

std::string utc_timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return buffer;
}

bool query_gpu_reading(GpuReading* out) {
    const std::string line = first_line(run_command(
        "nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw "
        "--format=csv,noheader,nounits 2>/dev/null"));
    if (line.empty()) {
        return false;
    }

    double clock = 0.0;
    double temperature = 0.0;
    double power = 0.0;
    if (std::sscanf(line.c_str(), "%lf, %lf, %lf", &clock, &temperature, &power) != 3) {
        return false;
    }

    out->valid = true;
    out->clock_mhz = clock;
    out->temperature_c = temperature;
    out->power_w = power;
    return true;
}

GpuSummary summarize_gpu_readings(const std::vector<GpuReading>& readings) {
    std::vector<double> clocks;
    std::vector<double> temperatures;
    std::vector<double> powers;
    for (const GpuReading& reading : readings) {
        if (reading.valid) {
            clocks.push_back(reading.clock_mhz);
            temperatures.push_back(reading.temperature_c);
            powers.push_back(reading.power_w);
        }
    }

    GpuSummary summary;
    if (clocks.empty()) {
        return summary;
    }
    summary.valid = true;
    summary.clock_mhz = summarize_samples(clocks).median;
    summary.temperature_c = summarize_samples(temperatures).median;
    summary.power_w = summarize_samples(powers).median;
    return summary;
}

std::string query_gpu_name() {
    const std::string line = first_line(run_command(
        "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null"));
    const std::size_t comma = line.rfind(',');
    if (comma == std::string::npos) {
        return line.empty() ? "NOT MEASURED" : line;
    }
    return line.substr(0, comma);
}

std::string query_driver_version() {
    const std::string line = first_line(run_command(
        "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null"));
    const std::size_t comma = line.rfind(',');
    if (comma == std::string::npos) {
        return "NOT MEASURED";
    }
    std::string version = line.substr(comma + 1);
    while (!version.empty() && version.front() == ' ') {
        version.erase(version.begin());
    }
    return version.empty() ? "NOT MEASURED" : version;
}
