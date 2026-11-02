cmake_minimum_required(VERSION 3.24)

project(cuda-matrix-engine VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(CME_BUILD_TESTS "Build the test suite" ON)
option(CME_BUILD_BENCHMARKS "Build the benchmarks" ON)
option(CME_BUILD_EXAMPLES "Build the examples" ON)
option(CME_WITH_CUBLAS "Enable the optional cuBLAS benchmark reference" OFF)

include(CTest)

set(CME_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA architectures for the GPU (default: 89)")
set(CMAKE_CUDA_ARCHITECTURES "${CME_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)

set(CME_SOURCES
    src/core/matrix.cpp
    src/core/status.cpp
    src/core/gemm_reference.cpp
    src/cuda/cuda_utils.cu
    src/cuda/device_buffer.cu
    src/cuda/gemm_dispatch.cu
    src/cuda/kernels/gemm_naive.cu
    src/cuda/kernels/gemm_tiled.cu
    src/cuda/kernels/gemm_optimized.cu
    src/api/gemm.cpp
)

set(CME_EXISTING_SOURCES)
foreach(_cme_source IN LISTS CME_SOURCES)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_cme_source}")
        list(APPEND CME_EXISTING_SOURCES "${_cme_source}")
    endif()
endforeach()

if(NOT CME_EXISTING_SOURCES)
    set(_cme_stub "${CMAKE_CURRENT_BINARY_DIR}/cme_empty.cpp")
    file(WRITE "${_cme_stub}" "")
    set(CME_EXISTING_SOURCES "${_cme_stub}")
    message(STATUS "No library sources exist yet; using a generated empty translation unit.")
endif()

add_library(cme STATIC ${CME_EXISTING_SOURCES})
add_library(cme::cme ALIAS cme)

target_include_directories(cme
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(cme PRIVATE CUDA::cudart)

if(CME_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(CME_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(CME_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
