#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark_config.hpp"
#include "benchmark_gemm.hpp"
#include "benchmark_rows.hpp"
#include "csv_writer.hpp"

namespace {

struct Options {
    BenchmarkConfig config;
    std::string output;
    std::set<std::string> measurements = {
        "kernel_only", "end_to_end", "h2d_transfer", "d2h_transfer"};
    bool quick = false;
    bool sizes_set = false;
    bool warmup_set = false;
    bool iterations_set = false;
};

std::vector<int> parse_sizes(const std::string& text) {
    std::vector<int> sizes;
    std::istringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (item.empty()) {
            continue;
        }
        const int size = std::stoi(item);
        if (size <= 0) {
            throw std::invalid_argument("sizes must be positive");
        }
        sizes.push_back(size);
    }
    if (sizes.empty()) {
        throw std::invalid_argument("no sizes supplied");
    }
    return sizes;
}

std::set<std::string> parse_measurements(const std::string& text) {
    std::set<std::string> measurements;
    std::istringstream input(text);
    std::string item;
    while (std::getline(input, item, ',')) {
        if (item.empty()) {
            continue;
        }
        if (item == "transfer") {
            measurements.insert("h2d_transfer");
            measurements.insert("d2h_transfer");
        } else if (item == "kernel_only" || item == "end_to_end" || item == "h2d_transfer" ||
                   item == "d2h_transfer") {
            measurements.insert(item);
        } else {
            throw std::invalid_argument("unknown measurement: " + item);
        }
    }
    if (measurements.empty()) {
        throw std::invalid_argument("no measurements supplied");
    }
    return measurements;
}

void usage(const char* program) {
    std::cout << "usage: " << program << " [--output <path>] [--quick] [--sizes 256,512]\n"
              << "       [--measurements kernel_only,end_to_end,h2d_transfer,d2h_transfer]\n"
              << "       [--warmup <n>] [--iterations <n>]\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--output") {
            if (++i >= argc) {
                throw std::invalid_argument("--output requires a path");
            }
            options.output = argv[i];
        } else if (arg == "--quick") {
            options.quick = true;
        } else if (arg == "--sizes") {
            if (++i >= argc) {
                throw std::invalid_argument("--sizes requires a value");
            }
            options.config.sizes = parse_sizes(argv[i]);
            options.sizes_set = true;
        } else if (arg == "--measurements") {
            if (++i >= argc) {
                throw std::invalid_argument("--measurements requires a value");
            }
            options.measurements = parse_measurements(argv[i]);
        } else if (arg == "--warmup") {
            if (++i >= argc) {
                throw std::invalid_argument("--warmup requires a value");
            }
            options.config.warmup_iterations = std::stoi(argv[i]);
            options.warmup_set = true;
        } else if (arg == "--iterations") {
            if (++i >= argc) {
                throw std::invalid_argument("--iterations requires a value");
            }
            options.config.measured_iterations = std::stoi(argv[i]);
            options.iterations_set = true;
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (options.quick) {
        if (!options.sizes_set) {
            options.config.sizes = {256, 512};
        }
        if (!options.warmup_set) {
            options.config.warmup_iterations = 2;
        }
        if (!options.iterations_set) {
            options.config.measured_iterations = 5;
        }
    }

    if (options.config.warmup_iterations < 0 || options.config.measured_iterations <= 0) {
        throw std::invalid_argument("iteration counts must be positive");
    }
    if (options.output.empty()) {
        options.output = "results/gemm_" + utc_timestamp() + ".csv";
    }
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::filesystem::path output_path(options.output);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }

        const RowContext context = make_row_context();
        CsvWriter writer(options.output);

        if (options.measurements.count("kernel_only") != 0) {
            run_kernel_only_benchmarks(options.config, writer, context);
        }
        if (options.measurements.count("end_to_end") != 0) {
            run_end_to_end_benchmarks(options.config, writer, context);
        }
        if (options.measurements.count("h2d_transfer") != 0 ||
            options.measurements.count("d2h_transfer") != 0) {
            run_transfer_benchmarks(options.config, writer, context);
        }

        std::cout << "wrote " << options.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
