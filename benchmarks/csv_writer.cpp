#include "csv_writer.hpp"

#include <stdexcept>

namespace {

const char* const kHeader[] = {
    "timestamp_utc",          "algorithm",         "M",
    "N",                      "K",                 "tile_size",
    "measurement",            "iterations",        "median_ms",
    "min_ms",                 "max_ms",            "iqr_ms",
    "gflops_median",          "gpu_name",          "driver_version",
    "cuda_runtime_version",   "sm_clock_mhz_median", "temperature_c_median",
    "power_w_median",
};

}  // namespace

CsvWriter::CsvWriter(const std::string& path) {
    out_.open(path, std::ios::out | std::ios::trunc);
    if (!out_) {
        throw std::runtime_error("CsvWriter: failed to open " + path);
    }

    std::vector<std::string> header;
    for (const char* field : kHeader) {
        header.emplace_back(field);
    }
    write_row(header);
}

void CsvWriter::write_row(const std::vector<std::string>& fields) {
    for (std::size_t i = 0; i < fields.size(); ++i) {
        if (i != 0) {
            out_ << ',';
        }
        out_ << escape(fields[i]);
    }
    out_ << '\n';
    if (!out_) {
        throw std::runtime_error("CsvWriter: failed while writing a row");
    }
}

std::string CsvWriter::escape(const std::string& field) {
    const bool needs_quotes = field.find_first_of(",\"\r\n") != std::string::npos;
    if (!needs_quotes) {
        return field;
    }

    std::string escaped;
    escaped.reserve(field.size() + 2);
    escaped.push_back('"');
    for (char c : field) {
        if (c == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(c);
    }
    escaped.push_back('"');
    return escaped;
}
