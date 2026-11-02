#pragma once
#include <cuda_runtime.h>
#include "cme/gemm_config.hpp"
#include "cme/status.hpp"
#include "cuda/kernels/gemm_kernels.hpp"

namespace cme::detail {

// Selects the kernel for `algorithm` and enqueues it on `stream`.
// Does NOT synchronize. Returns CudaError on launch failure.
// Returns UnsupportedConfiguration if algorithm == CPU_REFERENCE (caller must not route here).
Status gemm_device(GemmAlgorithm algorithm,
                   const DeviceGemmArgs& args,
                   cudaStream_t stream);

}  // namespace cme::detail
