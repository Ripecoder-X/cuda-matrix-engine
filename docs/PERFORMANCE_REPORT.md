# Performance Report

TBD by Batch 7.3 (final benchmark tables, profiler summary, cuBLAS comparison only if measured).

## Benchmark Evidence (Phase 6)

Recorded by Batch 6.4 on 2026-11-02 from a real benchmark run on the NVIDIA GeForce RTX 4070 Ti SUPER.

- **Primary benchmark dataset:** `results/gemm_20261102_141125.csv`
- **Environment metadata:** `results/gemm_20261102_141125.csv.env.txt`
- **Driver / CUDA runtime:** 610.88 / 13.3
- **Peak Measured Throughput (Kernel-Only):**
  - Matrix 1024x1024: `CUDA_OPTIMIZED(32)` = **6,727.71 GFLOP/s** (0.3192 ms)
  - Matrix 2048x2048: `CUDA_OPTIMIZED(32)` = **7,408.79 GFLOP/s** (2.3188 ms)
  - Matrix 4096x4096: `CUDA_OPTIMIZED(32)` = **7,266.06 GFLOP/s** (18.9152 ms)
- **Measurement coverage:** All 4 types recorded (`kernel_only`, `end_to_end`, `h2d_transfer`, `d2h_transfer`) across sizes 256, 512, 1024, 2048, and 4096.

## Correctness evidence

Recorded by Batch 5.3 on 2026-11-02. This section is correctness evidence, not performance.

- Command: `./scripts/test.sh`
- Build: `cmake --preset linux-debug` and `cmake --build --preset linux-debug` completed.
- CTest: `100% tests passed, 0 tests failed out of 43`; 5368 assertions.
- Correctness matrix: 600 algorithm runs, 16,305,450 element comparisons.
- `compute-sanitizer` version: 2026.1.1.0 (CUDA 13.2 package).
- Memcheck on the full test binary: `ERROR SUMMARY: 0 errors`.
- Racecheck on the `[racecheck]` filter: `0 hazards displayed (0 errors, 0 warnings)`.
- Raw run log: `/tmp/cme_test_sh.log` (session artifact; not committed).

Environment: `docs/ENVIRONMENT.md`. The system `compute-sanitizer` 2022.4.1 is incompatible with the
installed 595.84 driver; `scripts/test.sh` selects the working 13.2 binary. See Batch 5.3 in
`docs/DESIGN_DECISIONS.md`.
