# cuda-matrix-engine

A performance-first CUDA C++ library for single-precision matrix multiplication
(`C = alpha * A * B + beta * C`). Row-major, FP32 only, synchronous public API.

Target: NVIDIA RTX 4070 Ti SUPER (compute capability 8.9), Linux primary.

**Under construction.** The architecture is frozen in `HANDOFF.md`; implementation
is in progress. Author: Nirmal. License: Apache-2.0.

## Key Features

- High-performance single-precision matrix multiplication
- Row-major memory layout support
- Synchronous public API
- CUDA-based implementation for NVIDIA GPUs
- Memory-efficient design

## Architecture Overview

The project follows a modular architecture:
- `include/cme/` - Public headers for library consumers
- `src/core/` - CPU-side algorithms and utilities
- `src/cuda/` - GPU-specific CUDA implementations
- `tests/` - Unit tests and validation
- `benchmarks/` - Performance testing suite
- `examples/` - Usage examples

## Status

This project is currently under development. Implementation details and performance benchmarks will be documented in the handoffs directory.

## License

This project is licensed under Apache License, Version 2.0.