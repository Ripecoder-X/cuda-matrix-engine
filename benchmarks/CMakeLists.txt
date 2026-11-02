add_executable(benchmark_main
    benchmark_main.cpp
    benchmark_gemm.cu
    benchmark_transfer.cu
    csv_writer.cpp
)

target_include_directories(benchmark_main
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_link_libraries(benchmark_main
    PRIVATE
        cme
        CUDA::cudart
)
