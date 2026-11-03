#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "csv_writer.hpp"

namespace {

const char* const kExpectedHeader =
    "timestamp_utc,algorithm,M,N,K,tile_size,measurement,iterations,median_ms,min_ms,"
    "max_ms,iqr_ms,gflops_median,gpu_name,driver_version,cuda_runtime_version,"
    "sm_clock_mhz_median,temperature_c_median,power_w_median";

bool read_file(const std::string& path, std::string* contents) {
    std::ifstream input(path, std::ios::in | std::ios::binary);
    if (!input) {
        return false;
    }
    contents->assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <output.csv>\n";
        return 2;
    }

    const std::string path = argv[1];
    {
        CsvWriter writer(path);
        writer.write_row({"2026-11-03T00:00:00Z", "CUDA_TILED", "256", "256", "256", "16",
                          "kernel_only", "30", "1.234500", "1.200000", "1.300000", "0.050000",
                          "27.180000", "test gpu, with comma", "595.84", "12.0", "210",
                          "43", "9.33"});
    }

    std::string contents;
    if (!read_file(path, &contents)) {
        std::cerr << "failed to read " << path << '\n';
        return 1;
    }

    const std::size_t newline = contents.find('\n');
    if (newline == std::string::npos || contents.substr(0, newline) != kExpectedHeader) {
        std::cerr << "CSV header does not match the frozen column order\n";
        return 1;
    }

    const std::size_t row_start = newline + 1;
    const std::string row = contents.substr(row_start);
    if (row.find("\"test gpu, with comma\"") == std::string::npos) {
        std::cerr << "CSV did not escape a field containing a comma\n";
        return 1;
    }
    if (row.find("timestamp_utc") != std::string::npos) {
        std::cerr << "unexpected extra header row\n";
        return 1;
    }

    std::cout << "CSV header and escaping verified\n";
    return 0;
}
