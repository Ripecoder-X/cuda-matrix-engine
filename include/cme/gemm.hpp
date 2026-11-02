#pragma once
#include "cme/gemm_config.hpp"
#include "cme/matrix_view.hpp"
#include "cme/status.hpp"

namespace cme {

// Computes C = alpha * A * B + beta * C.
//
// Shape rules:
//   A is M x K, B is K x N, C is M x N.
//   A.rows == C.rows, B.cols == C.cols, A.cols == B.rows.
//
// Semantics:
//   - When beta == 0, C is NOT read. Its prior contents (including NaN/Inf) are ignored.
//   - When alpha == 0, the result is C = beta * C (A and B are still validated).
//   - A, B, and C must not overlap in memory. Overlap is reported as UnsupportedConfiguration.
//   - Empty matrices (any dimension 0) are not supported and return InvalidArgument.
//
// Errors are returned via Status, never thrown.
Status gemm(MatrixView<const float> A,
            MatrixView<const float> B,
            MatrixView<float> C,
            float alpha,
            float beta,
            const GemmConfig& config);

// Validates a configuration without running anything.
// Returns InvalidArgument for an unknown algorithm value,
// UnsupportedConfiguration for a tile_size not in {16, 32} when the algorithm uses tiles.
Status validate_config(const GemmConfig& config);

}  // namespace cme
