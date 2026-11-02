#include "cme/gemm.hpp"

#include <cstddef>
#include <cstdint>

#include "core/gemm_reference.hpp"
#include "cuda/cuda_utils.hpp"
#include "cuda/device_buffer.hpp"
#include "cuda/gemm_dispatch.hpp"

namespace cme {
namespace {

struct HostRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

HostRange host_range(const float* data, int rows, int cols, int stride) {
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(data);
    const std::uintptr_t elements =
        (static_cast<std::uintptr_t>(rows) - 1U) * static_cast<std::uintptr_t>(stride) +
        static_cast<std::uintptr_t>(cols);
    return {begin, begin + elements * sizeof(float)};
}

bool ranges_overlap(const HostRange& lhs, const HostRange& rhs) {
    return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

}  // namespace

Status validate_config(const GemmConfig& config) {
    switch (config.algorithm) {
        case GemmAlgorithm::CPU_REFERENCE:
        case GemmAlgorithm::CUDA_NAIVE:
            return Status::ok();
        case GemmAlgorithm::CUDA_TILED:
        case GemmAlgorithm::CUDA_OPTIMIZED:
            if (config.tile_size != 16 && config.tile_size != 32) {
                return Status::error(
                    StatusCode::UnsupportedConfiguration,
                    "tile_size must be 16 or 32 for CUDA_TILED and CUDA_OPTIMIZED");
            }
            return Status::ok();
    }
    return Status::error(StatusCode::InvalidArgument, "unknown GEMM algorithm");
}

Status gemm(MatrixView<const float> A,
            MatrixView<const float> B,
            MatrixView<float> C,
            float alpha,
            float beta,
            const GemmConfig& config) {
    if (!A.is_valid()) {
        return Status::error(StatusCode::InvalidArgument, "A must be a valid matrix view");
    }
    if (!B.is_valid()) {
        return Status::error(StatusCode::InvalidArgument, "B must be a valid matrix view");
    }
    if (!C.is_valid()) {
        return Status::error(StatusCode::InvalidArgument, "C must be a valid matrix view");
    }

    if (A.rows != C.rows || A.cols != B.rows || B.cols != C.cols) {
        return Status::error(StatusCode::DimensionMismatch,
                             "A, B, and C dimensions are incompatible");
    }

    const Status config_status = validate_config(config);
    if (!config_status.is_ok()) {
        return config_status;
    }

    const HostRange range_a = host_range(A.data, A.rows, A.cols, A.stride);
    const HostRange range_b = host_range(B.data, B.rows, B.cols, B.stride);
    const HostRange range_c = host_range(C.data, C.rows, C.cols, C.stride);
    if (ranges_overlap(range_a, range_b) || ranges_overlap(range_a, range_c) ||
        ranges_overlap(range_b, range_c)) {
        return Status::error(StatusCode::UnsupportedConfiguration,
                             "A, B, and C must not overlap in memory");
    }

    if (config.algorithm == GemmAlgorithm::CPU_REFERENCE) {
        detail::gemm_cpu_reference(A, B, C, alpha, beta);
        return Status::ok();
    }

    const std::size_t a_count =
        static_cast<std::size_t>(A.rows) * static_cast<std::size_t>(A.cols);
    const std::size_t b_count =
        static_cast<std::size_t>(B.rows) * static_cast<std::size_t>(B.cols);
    const std::size_t c_count =
        static_cast<std::size_t>(C.rows) * static_cast<std::size_t>(C.cols);

    detail::DeviceBuffer device_a;
    detail::DeviceBuffer device_b;
    detail::DeviceBuffer device_c;

    Status status = detail::DeviceBuffer::create(a_count, &device_a);
    if (!status.is_ok()) {
        return status;
    }
    status = detail::DeviceBuffer::create(b_count, &device_b);
    if (!status.is_ok()) {
        return status;
    }
    status = detail::DeviceBuffer::create(c_count, &device_c);
    if (!status.is_ok()) {
        return status;
    }

    status = device_a.copy_from_host_2d(A.data, A.rows, A.cols, A.stride);
    if (!status.is_ok()) {
        return status;
    }
    status = device_b.copy_from_host_2d(B.data, B.rows, B.cols, B.stride);
    if (!status.is_ok()) {
        return status;
    }
    if (beta != 0.0f) {
        status = device_c.copy_from_host_2d(C.data, C.rows, C.cols, C.stride);
        if (!status.is_ok()) {
            return status;
        }
    }

    detail::DeviceGemmArgs args;
    args.M = A.rows;
    args.N = B.cols;
    args.K = A.cols;
    args.A = device_a.data();
    args.lda = A.cols;
    args.B = device_b.data();
    args.ldb = B.cols;
    args.C = device_c.data();
    args.ldc = C.cols;
    args.alpha = alpha;
    args.beta = beta;
    args.tile_size = config.tile_size;

    status = detail::gemm_device(config.algorithm, args, nullptr);
    if (!status.is_ok()) {
        return status;
    }

    CME_RETURN_IF_CUDA_ERROR(cudaDeviceSynchronize(), "gemm synchronize");

    return device_c.copy_to_host_2d(C.data, C.rows, C.cols, C.stride);
}

}  // namespace cme
