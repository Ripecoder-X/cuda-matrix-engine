add_library(cme_benchmark_common STATIC
    csv_writer.cpp
    benchmark_common.cpp
)
target_link_libraries(cme_benchmark_common PUBLIC cme)
target_include_directories(cme_benchmark_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(cme_csv_writer_test csv_writer_test.cpp)
target_link_libraries(cme_csv_writer_test PRIVATE cme_benchmark_common)

add_executable(cme_benchmark
    benchmark_main.cpp
    benchmark_rows.cpp
    benchmark_gemm.cu
    benchmark_transfer.cu
)
target_include_directories(cme_benchmark PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(cme_benchmark PRIVATE cme cme_benchmark_common CUDA::cudart)

add_test(
    NAME benchmark_csv_writer
    COMMAND cme_csv_writer_test "${CMAKE_CURRENT_BINARY_DIR}/csv_writer_test.csv"
)
