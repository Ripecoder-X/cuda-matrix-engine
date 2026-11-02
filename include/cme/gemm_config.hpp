#pragma once

namespace cme {

enum class GemmAlgorithm {
    CPU_REFERENCE,
    CUDA_NAIVE,
    CUDA_TILED,
    CUDA_OPTIMIZED
};

struct GemmConfig {
    GemmAlgorithm algorithm = GemmAlgorithm::CUDA_TILED;
    int tile_size = 16;   // Must be 16 or 32 for CUDA_TILED and CUDA_OPTIMIZED. Ignored otherwise.
};

}  // namespace cme
