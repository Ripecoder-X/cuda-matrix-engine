#pragma once

#include <fstream>
#include <string>
#include <vector>

class CsvWriter {
public:
    explicit CsvWriter(const std::string& path);

    void write_row(const std::vector<std::string>& fields);

private:
    static std::string escape(const std::string& field);

    std::ofstream out_;
};
