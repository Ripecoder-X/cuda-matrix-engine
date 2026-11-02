#pragma once
#include <string>

namespace cme {

enum class StatusCode {
    Ok,
    InvalidArgument,          // null data, non-positive dimension, bad enum value
    DimensionMismatch,        // A, B, C shapes incompatible
    UnsupportedConfiguration, // tile_size not in {16, 32}, aliasing, empty matrices
    OutOfMemory,              // device allocation failed
    CudaError                 // any other CUDA runtime or launch error
};

class [[nodiscard]] Status {
public:
    static Status ok();
    static Status error(StatusCode code, std::string message);

    bool is_ok() const noexcept;
    StatusCode code() const noexcept;
    const std::string& message() const noexcept;

private:
    Status(StatusCode code, std::string message);
    StatusCode code_;
    std::string message_;
};

}  // namespace cme
