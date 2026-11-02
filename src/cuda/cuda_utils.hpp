#pragma once
#include <cuda_runtime.h>
#include "cme/status.hpp"

namespace cme::detail {

// Converts a cudaError_t to a Status. Returns Ok on cudaSuccess.
// The message includes cudaGetErrorName and cudaGetErrorString.
Status to_status(cudaError_t err, const char* context);

// Checks the last error after a kernel launch. Returns CudaError on failure.
Status check_last_launch(const char* context);

}  // namespace cme::detail

// Use inside functions that return cme::Status:
//   CME_RETURN_IF_CUDA_ERROR(cudaMalloc(&p, n), "allocate A");
#define CME_RETURN_IF_CUDA_ERROR(expr, context)                  \
    do {                                                         \
        ::cme::Status _cme_st = ::cme::detail::to_status((expr), (context)); \
        if (!_cme_st.is_ok()) return _cme_st;                    \
    } while (0)
