#include "cuda/device_buffer.hpp"

#include <cuda_runtime.h>
#include <string>

#include "cuda/cuda_utils.hpp"

namespace cme::detail {

DeviceBuffer::DeviceBuffer() noexcept : ptr_(nullptr), size_(0) {}

DeviceBuffer::~DeviceBuffer() {
    if (ptr_ != nullptr) {
        cudaFree(ptr_);
    }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

Status DeviceBuffer::create(std::size_t count, DeviceBuffer* out) {
    if (count == 0) {
        return Status::error(StatusCode::InvalidArgument,
                             "DeviceBuffer::create called with count 0");
    }

    float* ptr = nullptr;
    const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(float));
    if (err != cudaSuccess) {
        const StatusCode code = err == cudaErrorMemoryAllocation ? StatusCode::OutOfMemory
                                                                 : StatusCode::CudaError;
        Status status = to_status(err, "DeviceBuffer::create (cudaMalloc)");
        return Status::error(code, status.message());
    }

    DeviceBuffer buffer;
    buffer.ptr_ = ptr;
    buffer.size_ = count;
    *out = std::move(buffer);
    return Status::ok();
}

Status DeviceBuffer::copy_from_host(const float* host, std::size_t count) {
    if (count > size_) {
        return Status::error(StatusCode::InvalidArgument,
                             "DeviceBuffer::copy_from_host count exceeds buffer size");
    }
    if (count == 0) {
        return Status::ok();
    }

    CME_RETURN_IF_CUDA_ERROR(
        cudaMemcpy(ptr_, host, count * sizeof(float), cudaMemcpyHostToDevice),
        "DeviceBuffer::copy_from_host");
    return Status::ok();
}

Status DeviceBuffer::copy_to_host(float* host, std::size_t count) const {
    if (count > size_) {
        return Status::error(StatusCode::InvalidArgument,
                             "DeviceBuffer::copy_to_host count exceeds buffer size");
    }
    if (count == 0) {
        return Status::ok();
    }

    CME_RETURN_IF_CUDA_ERROR(cudaMemcpy(host, ptr_, count * sizeof(float), cudaMemcpyDeviceToHost),
                             "DeviceBuffer::copy_to_host");
    return Status::ok();
}

Status DeviceBuffer::copy_from_host_2d(const float* host, int rows, int cols, int host_stride) {
    if (host == nullptr || rows <= 0 || cols <= 0 || host_stride < cols) {
        return Status::error(StatusCode::InvalidArgument,
                             "DeviceBuffer::copy_from_host_2d invalid dimensions");
    }
    const std::size_t required =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (required > size_) {
        return Status::error(StatusCode::InvalidArgument,
                             "DeviceBuffer::copy_from_host_2d exceeds buffer size");
    }

    CME_RETURN_IF_CUDA_ERROR(
        cudaMemcpy2D(ptr_, static_cast<std::size_t>(cols) * sizeof(float), host,
                     static_cast<std::size_t>(host_stride) * sizeof(float),
                     static_cast<std::size_t>(cols) * sizeof(float),
                     static_cast<std::size_t>(rows), cudaMemcpyHostToDevice),
        "DeviceBuffer::copy_from_host_2d");
    return Status::ok();
}

Status DeviceBuffer::copy_to_host_2d(float* host, int rows, int cols, int host_stride) const {
    if (host == nullptr || rows <= 0 || cols <= 0 || host_stride < cols) {
        return Status::error(StatusCode::InvalidArgument,
                             "DeviceBuffer::copy_to_host_2d invalid dimensions");
    }
    const std::size_t required =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    if (required > size_) {
        return Status::error(StatusCode::InvalidArgument,
                             "DeviceBuffer::copy_to_host_2d exceeds buffer size");
    }

    CME_RETURN_IF_CUDA_ERROR(
        cudaMemcpy2D(host, static_cast<std::size_t>(host_stride) * sizeof(float), ptr_,
                     static_cast<std::size_t>(cols) * sizeof(float),
                     static_cast<std::size_t>(cols) * sizeof(float),
                     static_cast<std::size_t>(rows), cudaMemcpyDeviceToHost),
        "DeviceBuffer::copy_to_host_2d");
    return Status::ok();
}

float* DeviceBuffer::data() noexcept {
    return ptr_;
}

const float* DeviceBuffer::data() const noexcept {
    return ptr_;
}

std::size_t DeviceBuffer::size() const noexcept {
    return size_;
}

}  // namespace cme::detail
