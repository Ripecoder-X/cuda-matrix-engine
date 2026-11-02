#pragma once
#include <cstddef>
#include "cme/status.hpp"

namespace cme::detail {

// RAII owner of a float device allocation. Move-only. Never copyable.
// Destructor frees memory and ignores errors (cudaFree failures are not recoverable here).
class DeviceBuffer {
public:
    DeviceBuffer() noexcept;
    ~DeviceBuffer();
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Allocates `count` floats. count == 0 returns InvalidArgument.
    // On failure, *out is left unchanged and an error Status is returned.
    static Status create(std::size_t count, DeviceBuffer* out);

    Status copy_from_host(const float* host, std::size_t count);  // count <= size()
    Status copy_to_host(float* host, std::size_t count) const;    // count <= size()

    // Copies a row-major block with a host stride into this tightly packed buffer.
    // Requires rows > 0, cols > 0, host_stride >= cols, and rows * cols <= size().
    Status copy_from_host_2d(const float* host, int rows, int cols, int host_stride);

    // Copies this tightly packed buffer into a row-major block with a host stride.
    // Requires the same dimensions and bounds as copy_from_host_2d.
    Status copy_to_host_2d(float* host, int rows, int cols, int host_stride) const;

    float* data() noexcept;
    const float* data() const noexcept;
    std::size_t size() const noexcept;

private:
    float* ptr_;
    std::size_t size_;
};

}  // namespace cme::detail
