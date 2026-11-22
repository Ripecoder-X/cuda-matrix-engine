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

foreach(_cme_source IN LISTS CME_SOURCES)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_cme_source}")
        message(FATAL_ERROR "Required source file missing: ${_cme_source}")
    endif()
endforeach()

set(CME_EXISTING_SOURCES)
foreach(_cme_source IN LISTS CME_SOURCES)
    list(APPEND CME_EXISTING_SOURCES "${_cme_source}")
endforeach()

if(NOT CME_EXISTING_SOURCES)
    set(_cme_stub "${CMAKE_CURRENT_BINARY_DIR}/cme_empty.cpp")
    file(WRITE "${_cme_stub}" "")
    set(CME_EXISTING_SOURCES "${_cme_stub}")
    message(STATUS "No library sources exist yet; using a generated empty translation unit.")
endif()

add_library(cme STATIC ${CME_EXISTING_SOURCES})
add_library(cme::cme ALIAS cme)
target_compile_features(cme PUBLIC cxx_std_17)

include(GNUInstallDirs)

target_include_directories(cme
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_link_libraries(cme PRIVATE CUDA::cudart)

include(CMakePackageConfigHelpers)

set(CME_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/cme")

install(TARGETS cme
    EXPORT cmeTargets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(EXPORT cmeTargets
    FILE cmeTargets.cmake
    NAMESPACE cme::
    DESTINATION "${CME_INSTALL_CMAKEDIR}"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmeConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmeConfig.cmake"
    INSTALL_DESTINATION "${CME_INSTALL_CMAKEDIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cmeConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cmeConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmeConfigVersion.cmake"
    DESTINATION "${CME_INSTALL_CMAKEDIR}"
)

if(CME_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(CME_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(CME_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
