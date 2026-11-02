include(FetchContent)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.9.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(Catch2)

add_executable(cme_tests
    test_status.cpp
    test_matrix.cpp
    test_gemm_reference.cpp
    test_cuda_errors.cpp
    test_gemm_algorithms.cpp
    test_gemm_dispatch.cpp
    test_gemm_edge_cases.cpp
    test_gemm_correctness.cpp
)

target_link_libraries(cme_tests PRIVATE cme Catch2::Catch2WithMain CUDA::cudart)
target_include_directories(cme_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

include(Catch)
catch_discover_tests(cme_tests)
