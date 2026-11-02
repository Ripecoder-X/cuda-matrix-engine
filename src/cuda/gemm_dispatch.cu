#include "cuda/gemm_dispatch.hpp"

#include "cuda/cuda_utils.hpp"

namespace cme::detail {

Status gemm_device(GemmAlgorithm algorithm, const DeviceGemmArgs& args, cudaStream_t stream) {
    cudaError_t err = cudaSuccess;
    const char* context = "gemm_device launch";

    switch (algorithm) {
        case GemmAlgorithm::CPU_REFERENCE:
            return Status::error(StatusCode::UnsupportedConfiguration,
                                 "CPU_REFERENCE is not a device algorithm");
        case GemmAlgorithm::CUDA_NAIVE:
            context = "gemm_device launch CUDA_NAIVE";
            err = launch_gemm_naive(args, stream);
            break;
        case GemmAlgorithm::CUDA_TILED:
            context = "gemm_device launch CUDA_TILED";
            err = launch_gemm_tiled(args, stream);
            break;
        case GemmAlgorithm::CUDA_OPTIMIZED:
            context = "gemm_device launch CUDA_OPTIMIZED";
            err = launch_gemm_optimized(args, stream);
            break;
        default:
            return Status::error(StatusCode::InvalidArgument, "unknown GEMM algorithm");
    }

    if (err != cudaSuccess) {
        return to_status(err, context);
    }
    return Status::ok();
}

}  // namespace cme::detail
