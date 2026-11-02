# Algorithms

All kernels compute `C = alpha * A * B + beta * C` for row-major FP32 matrices with leading
dimensions `lda`, `ldb`, and `ldc`. Each kernel accumulates in `float`; the CPU reference accumulates
each dot product in `double`. The tolerance rule in `QUALITY_STANDARDS.md` accounts for this
difference. No kernel reads `C` when `beta == 0`.

## Naive kernel

`src/cuda/kernels/gemm_naive.cu` assigns one thread to one element of `C`. The block is 16x16 threads
and the grid covers `M` and `N` with ceiling division. Both operands are read directly from global
memory. Threads outside the matrix return before the dot product.

This kernel exists as the correctness baseline and as a low-optimization reference for later
measurements. It has no shared-memory usage and no special alignment requirements.

## Tiled kernel

`src/cuda/kernels/gemm_tiled.cu` uses square shared-memory tiles. `TILE` is a compile-time template
parameter and the launcher accepts only `tile_size` 16 or 32; any other value returns
`cudaErrorInvalidValue`. Block dimensions equal `TILE x TILE`.

For each step along `K`, every thread loads one element of an A tile and one element of a B tile into
shared memory. Out-of-range elements are loaded as `0.0f`, so partial edge tiles contribute zero to
the dot product. `__syncthreads()` separates the tile load from the inner product and the inner
product from the next load. The final write applies `alpha` and `beta` and is bounds-checked.

## Optimized kernel — candidate design

`src/cuda/kernels/gemm_optimized.cu` is a correctness candidate, not yet a measured improvement over
the tiled kernel. It supports the same `tile_size` values and the same semantics.

Techniques used, and why:

- **2x2 register micro-tile per thread.** Each thread accumulates four outputs, which reuses each
  loaded A and B value across two multiply-adds and halves the number of threads that would otherwise
  each load the same operands from shared memory.
- **Block dimensions `TILE/2 x TILE/2`.** The smaller block still covers the full `TILE x TILE`
  output tile because each thread owns a 2x2 region. The tile load is a strided loop over
  `TILE * TILE` elements, so every shared-memory element is still initialized once.
- **Padded shared-memory rows (`TILE + 1`).** The extra column changes the row stride so consecutive
  rows do not map to the same shared-memory bank for the access patterns used here. The step is
  intended to reduce bank conflicts; it is not yet backed by profiler evidence.
- **Compile-time unrolled inner loop.** `TILE` is known at compile time, so `#pragma unroll` removes
  loop overhead and exposes the shared-memory loads to the compiler for scheduling.

No speed claim is made for this kernel in this phase. Batch 7 decides whether it remains in the
default dispatch using measured kernel-only results and the 5% rule.

## Tolerance analysis

The tolerance rule was not changed in Phase 3. Kernel results agree with the CPU reference for every
tested shape using the existing `|got - ref| <= 1e-5 * K + 1e-4 * |ref|` bound.
