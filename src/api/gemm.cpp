#include "cme/gemm.hpp"

#include <cstddef>
#include <cstdint>

#include "core/gemm_reference.hpp"
#include "cuda/cuda_utils.hpp"
#include "cuda/device_buffer.hpp"
#include "cuda/gemm_dispatch.hpp"

namespace cme {
namespace {

// Returns true if two row-major strided views share any element address.
// Each row is a contiguous block of cols * sizeof(float) bytes at
// data + row * stride.  Rows within a view never overlap because
// is_valid() requires stride >= cols.
bool views_overlap(const float* data_a, int rows_a, int cols_a, int stride_a,
                   const float* data_b, int rows_b, int cols_b, int stride_b) {
    const std::size_t elem_bytes = sizeof(float);
    for (int r = 0; r < rows_a; ++r) {
        const std::uintptr_t a_start =
            reinterpret_cast<std::uintptr_t>(data_a) +
            static_cast<std::uintptr_t>(r) * static_cast<std::uintptr_t>(stride_a) * elem_bytes;
        const std::uintptr_t a_end =
            a_start + static_cast<std::uintptr_t>(cols_a) * elem_bytes;
        for (int s = 0; s < rows_b; ++s) {
            const std::uintptr_t b_start =
                reinterpret_cast<std::uintptr_t>(data_b) +
                static_cast<std::uintptr_t>(s) * static_cast<std::uintptr_t>(stride_b) * elem_bytes;
            const std::uintptr_t b_end =
                b_start + static_cast<std::uintptr_t>(cols_b) * elem_bytes;
            if (a_start < b_end && b_start < a_end) {
                return true;
            }
        }
    }
    return false;
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

    if (views_overlap(A.data, A.rows, A.cols, A.stride,
                      B.data, B.rows, B.cols, B.stride) ||
        views_overlap(A.data, A.rows, A.cols, A.stride,
                      C.data, C.rows, C.cols, C.stride) ||
        views_overlap(B.data, B.rows, B.cols, B.stride,
                      C.data, C.rows, C.cols, C.stride)) {
        return Status::error(StatusCode::UnsupportedConfiguration,
                             "A, B, and C must not overlap in memory");
    }

    if (config.algorithm == GemmAlgorithm::CPU_REFERENCE) {
        detail::gemm_cpu_reference(A, B, C, alpha, beta);
        return Status::ok();
    }

    // When alpha == 0, A and B are validated but not read, matching the CPU reference.
    // Compute C = beta * C directly without launching a GEMM kernel.
    if (alpha == 0.0f) {
        for (int i = 0; i < C.rows; ++i) {
            for (int j = 0; j < C.cols; ++j) {
                C(i, j) = (beta == 0.0f) ? 0.0f : beta * C(i, j);
            }
        }
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
