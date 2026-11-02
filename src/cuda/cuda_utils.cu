#include "cuda/cuda_utils.hpp"

#include <string>

namespace cme::detail {

Status to_status(cudaError_t err, const char* context) {
    if (err == cudaSuccess) {
        return Status::ok();
    }

    const char* name = cudaGetErrorName(err);
    const char* description = cudaGetErrorString(err);
    std::string message = context != nullptr ? context : "CUDA error";
    message += ": ";
    message += name != nullptr ? name : "unknown error";
    message += " (";
    message += description != nullptr ? description : "unknown description";
    message += ")";
    return Status::error(StatusCode::CudaError, std::move(message));
}

Status check_last_launch(const char* context) {
    return to_status(cudaGetLastError(), context);
}

}  // namespace cme::detail
