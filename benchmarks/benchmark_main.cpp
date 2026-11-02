#include "benchmark_config.hpp"
#include "csv_writer.hpp"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void run_kernel_benchmarks(const BenchmarkConfig& config, CsvWriter& writer);
void run_end_to_end_benchmarks(const BenchmarkConfig& config, CsvWriter& writer);
void run_transfer_benchmarks(const BenchmarkConfig& config, CsvWriter& writer);

namespace {

std::string get_default_output_path() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm gm{};
#if defined(_WIN32)
    gmtime_s(&gm, &tt);
#else
    gmtime_r(&tt, &gm);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &gm);
    return std::string("results/gemm_") + buf + ".csv";
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --output <path>    Specify output CSV path (default: results/gemm_<timestamp>.csv)\n"
              << "  --quick            Run fewer iterations and sizes 256 and 512 only for smoke testing\n"
              << "  --size <N>         Run benchmark for a single matrix size (e.g. 2048)\n"
              << "  --warmup <N>       Number of warmup iterations\n"
              << "  --iterations <N>   Number of measured iterations\n"
              << "  --kernel-only      Run only kernel-level benchmarks\n"
              << "  --end-to-end       Run only end-to-end benchmarks\n"
              << "  --transfer-only    Run only transfer benchmarks\n"
              << "  -h, --help         Show this help message\n";
}

}  // namespace

int main(int argc, char** argv) {
    BenchmarkConfig config;
    std::string output_path = "";
    bool quick_mode = false;
    bool do_kernel = true;
    bool do_end_to_end = true;
    bool do_transfer = true;
    bool phase_filtered = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quick") {
            quick_mode = true;
            config.sizes = {256, 512};
            config.warmup_iterations = 2;
            config.measured_iterations = 10;
        } else if (arg == "--size" && i + 1 < argc) {
            config.sizes = {std::stoi(argv[++i])};
        } else if (arg == "--warmup" && i + 1 < argc) {
            config.warmup_iterations = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.measured_iterations = std::stoi(argv[++i]);
        } else if (arg == "--kernel-only") {
            if (!phase_filtered) { do_kernel = false; do_end_to_end = false; do_transfer = false; phase_filtered = true; }
            do_kernel = true;
        } else if (arg == "--end-to-end") {
            if (!phase_filtered) { do_kernel = false; do_end_to_end = false; do_transfer = false; phase_filtered = true; }
            do_end_to_end = true;
        } else if (arg == "--transfer-only") {
            if (!phase_filtered) { do_kernel = false; do_end_to_end = false; do_transfer = false; phase_filtered = true; }
            do_transfer = true;
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if (output_path.empty()) {
        output_path = get_default_output_path();
    }

    // Ensure output directory exists
    try {
        std::filesystem::path p(output_path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
    } catch (const std::exception& e) {
        std::cerr << "Warning: could not create parent directory for " << output_path << ": " << e.what() << std::endl;
    }

    std::cout << "=========================================================\n"
              << " CUDA Matrix Engine - Benchmark Driver\n"
              << " Mode:   " << (quick_mode ? "QUICK (smoke test)" : "FULL (standard methodology)") << "\n"
              << " Output: " << output_path << "\n"
              << " Sizes:  ";
    for (std::size_t i = 0; i < config.sizes.size(); ++i) {
        if (i != 0) std::cout << ", ";
        std::cout << config.sizes[i];
    }
    std::cout << "\n Warmup: " << config.warmup_iterations << " iterations\n"
              << " Timing: " << config.measured_iterations << " iterations\n"
              << "=========================================================\n" << std::endl;

    try {
        CsvWriter writer(output_path);

        if (do_kernel) {
            std::cout << ">>> Phase 1: Kernel-Only Benchmarks (direct device dispatch)" << std::endl;
            run_kernel_benchmarks(config, writer);
        }

        if (do_end_to_end) {
            std::cout << "\n>>> Phase 2: End-To-End Benchmarks (public gemm API)" << std::endl;
            run_end_to_end_benchmarks(config, writer);
        }

        if (do_transfer) {
            std::cout << "\n>>> Phase 3: Transfer Benchmarks (host <-> device copies)" << std::endl;
            run_transfer_benchmarks(config, writer);
        }

        std::cout << "\n[SUCCESS] Benchmark run finished. Results saved to " << output_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Benchmark execution failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
