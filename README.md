# cuda-matrix-engine

A performance-first CUDA C++ library for single-precision matrix multiplication
(`C = alpha * A * B + beta * C`). Row-major, FP32 only, synchronous public API.

Target: NVIDIA RTX 4070 Ti SUPER (compute capability 8.9), Linux primary.

**Under construction.** The architecture is frozen in `HANDOFF.md`; implementation
is in progress. Author: Nirmal. License: Apache-2.0.
