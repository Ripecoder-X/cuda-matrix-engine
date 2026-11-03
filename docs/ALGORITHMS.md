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

## Optimized kernel

`src/cuda/kernels/gemm_optimized.cu` supports the same `tile_size` values and semantics as the tiled
kernel. Its register micro-tile size depends on the block tile:

- **TILE=16: 2x2 register micro-tile per thread.** Each thread accumulates four outputs. Block
  dimensions are `TILE/2 x TILE/2` (8x8), so the block still covers a 16x16 output tile.
- **TILE=32: 4x4 register micro-tile per thread.** Each thread accumulates sixteen outputs. Block
  dimensions are `TILE/4 x TILE/4` (8x8), so the block still covers a 32x32 output tile. This
  reduces shared-memory loads per multiply-add from 1.0 to 0.5.

Other techniques used by both variants:

- **Padded shared-memory rows (`TILE + 1`).** The extra column changes the row stride so consecutive
  rows do not map to the same shared-memory bank for the access patterns used here.
- **Compile-time unrolled loops.** `TILE` and the micro-tile size are compile-time constants, so
  `#pragma unroll` removes loop overhead and exposes the shared-memory loads to the compiler.

The tile load is a strided loop over `TILE * TILE` elements, so every shared-memory element is
initialized once. Out-of-range elements are loaded as `0.0f`; the final write applies `alpha` and
`beta` with bounds checks.

### Measured result

Batch 7.2 measured the 4x4 TILE=32 variant against the previous 2x2 candidate with the kernel-only
benchmark (30 measured iterations, sizes 256 to 4096). At sizes 512 and larger the median GFLOP/s
improved by more than 5%; the improvement at 4096 was 5,487.1 to 12,277.9 GFLOP/s. At 256 the 4x4
variant was slower and is not claimed for that size. The full before/after table and profiler values
are in `docs/PROFILING.md`. The 2x2 TILE=16 variant was unchanged in that iteration.

CUDA_OPTIMIZED remains an explicit algorithm choice. The public default is still CUDA_TILED with
tile_size 16 as frozen in section 4.4; changing the public default is outside the Phase 7 file list.

## Tolerance analysis

The tolerance rule was not changed in Phase 3. Kernel results agree with the CPU reference for every
tested shape using the existing `|got - ref| <= 1e-5 * K + 1e-4 * |ref|` bound.
