#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "cuda/cuda_utils.hpp"
#include "cuda/device_buffer.hpp"

namespace {

bool require_cuda_device() {
    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        cudaGetLastError();
        SKIP("No CUDA device available");
    }
    return true;
}

}  // namespace

TEST_CASE("DeviceBuffer round-trip copy", "[cuda][device_buffer]") {
    require_cuda_device();

    constexpr std::size_t count = 1024;
    std::vector<float> host_in(count);
    for (std::size_t i = 0; i < count; ++i) {
        host_in[i] = static_cast<float>(i) * 0.5f - 3.0f;
    }

    cme::detail::DeviceBuffer buffer;
    REQUIRE(cme::detail::DeviceBuffer::create(count, &buffer).is_ok());
    REQUIRE(buffer.size() == count);
    REQUIRE(buffer.data() != nullptr);

    REQUIRE(buffer.copy_from_host(host_in.data(), count).is_ok());

    std::vector<float> host_out(count, 0.0f);
    REQUIRE(buffer.copy_to_host(host_out.data(), count).is_ok());
    REQUIRE(host_in == host_out);
}

TEST_CASE("DeviceBuffer create rejects zero count", "[cuda][device_buffer]") {
    require_cuda_device();

    cme::detail::DeviceBuffer buffer;
    const cme::Status status = cme::detail::DeviceBuffer::create(0, &buffer);
    REQUIRE_FALSE(status.is_ok());
    REQUIRE(status.code() == cme::StatusCode::InvalidArgument);
    REQUIRE(buffer.size() == 0);
    REQUIRE(buffer.data() == nullptr);
}

TEST_CASE("DeviceBuffer move semantics", "[cuda][device_buffer]") {
    require_cuda_device();

    cme::detail::DeviceBuffer source;
    REQUIRE(cme::detail::DeviceBuffer::create(64, &source).is_ok());

    cme::detail::DeviceBuffer moved(std::move(source));
    REQUIRE(moved.size() == 64);
    REQUIRE(moved.data() != nullptr);
    REQUIRE(source.size() == 0);
    REQUIRE(source.data() == nullptr);

    cme::detail::DeviceBuffer assigned;
    assigned = std::move(moved);
    REQUIRE(assigned.size() == 64);
    REQUIRE(assigned.data() != nullptr);
    REQUIRE(moved.size() == 0);
    REQUIRE(moved.data() == nullptr);
}

TEST_CASE("DeviceBuffer 2D copy honors a strided host layout", "[cuda][device_buffer]") {
    require_cuda_device();

    constexpr int rows = 3;
    constexpr int cols = 4;
    constexpr int stride = 6;
    std::vector<float> host_in(static_cast<std::size_t>(rows) * stride, -1.0f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            host_in[static_cast<std::size_t>(r) * stride + c] =
                static_cast<float>(r * 10 + c);
        }
    }

    cme::detail::DeviceBuffer buffer;
    REQUIRE(cme::detail::DeviceBuffer::create(rows * cols, &buffer).is_ok());
    REQUIRE(buffer.copy_from_host_2d(host_in.data(), rows, cols, stride).is_ok());

    std::vector<float> host_out(static_cast<std::size_t>(rows) * stride, 99.0f);
    REQUIRE(buffer.copy_to_host_2d(host_out.data(), rows, cols, stride).is_ok());
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const std::size_t index = static_cast<std::size_t>(r) * stride + c;
            REQUIRE(host_out[index] == host_in[index]);
        }
        for (int c = cols; c < stride; ++c) {
            REQUIRE(host_out[static_cast<std::size_t>(r) * stride + c] == 99.0f);
        }
    }

    REQUIRE_FALSE(buffer.copy_from_host_2d(host_in.data(), rows, cols, cols - 1).is_ok());
}

TEST_CASE("DeviceBuffer rejects copy count larger than size", "[cuda][device_buffer]") {
    require_cuda_device();

    cme::detail::DeviceBuffer buffer;
    REQUIRE(cme::detail::DeviceBuffer::create(8, &buffer).is_ok());

    std::vector<float> host(16, 1.0f);
    const cme::Status from_status = buffer.copy_from_host(host.data(), host.size());
    REQUIRE_FALSE(from_status.is_ok());
    REQUIRE(from_status.code() == cme::StatusCode::InvalidArgument);

    const cme::Status to_status = buffer.copy_to_host(host.data(), host.size());
    REQUIRE_FALSE(to_status.is_ok());
    REQUIRE(to_status.code() == cme::StatusCode::InvalidArgument);
}

TEST_CASE("to_status maps success and CUDA errors", "[cuda][status]") {
    const cme::Status ok = cme::detail::to_status(cudaSuccess, "test context");
    REQUIRE(ok.is_ok());

    const cme::Status error =
        cme::detail::to_status(cudaErrorInvalidValue, "test context");
    REQUIRE_FALSE(error.is_ok());
    REQUIRE(error.code() == cme::StatusCode::CudaError);
    REQUIRE(error.message().find("test context") != std::string::npos);
    REQUIRE(error.message().find(cudaGetErrorName(cudaErrorInvalidValue)) !=
            std::string::npos);
    REQUIRE(error.message().find(cudaGetErrorString(cudaErrorInvalidValue)) !=
            std::string::npos);
}

TEST_CASE("check_last_launch reports a clean error state", "[cuda][status]") {
    static_cast<void>(cudaGetLastError());
    const cme::Status status = cme::detail::check_last_launch("clean launch");
    REQUIRE(status.is_ok());
}
