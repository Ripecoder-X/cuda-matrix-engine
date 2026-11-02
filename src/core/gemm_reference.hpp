#pragma once
#include "cme/matrix_view.hpp"
#include "cme/status.hpp"

namespace cme::detail {

// CPU reference. Accumulates each dot product in double, then casts to float.
// Assumes arguments already validated by the public API.
void gemm_cpu_reference(MatrixView<const float> A,
                        MatrixView<const float> B,
                        MatrixView<float> C,
                        float alpha,
                        float beta);

}  // namespace cme::detail
