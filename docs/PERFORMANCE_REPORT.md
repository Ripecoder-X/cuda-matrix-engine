# Performance Report

Final report verified during closeout on 2026-11-22. Main tables are copied from the final benchmark
CSV `results/gemm_20261122T073654Z.csv` (63 lines: one frozen header plus 62 data rows). The optional
cuBLAS baseline was measured later in `results/gemm_20261122T102414Z_linux-release-cublas.csv`
(68 lines: one header plus 67 data rows). Profiler evidence is in `results/profiles/` and summarized
in `docs/PROFILING.md`.

## Hardware and software

Source: `docs/ENVIRONMENT.md`.

- CPU: Intel Core i9-14900K, 32 logical CPUs, 62 GiB system RAM.
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB, compute capability 8.9 (sm_89).
- GPU driver: 595.84; CUDA version reported by the driver: 13.2.
- Installed CUDA toolkit: 12.0 (`nvcc` V12.0.140). The benchmark records the CUDA runtime version
  reported by `cudaRuntimeGetVersion()`: 12.0.
- OS: Pop!_OS 24.04 LTS, kernel 7.0.11-76070011-generic.
- Build: CMake 3.28.3, g++ 13.3.0, Release build (`linux-release`), `CME_CUDA_ARCHITECTURES=89`.
- cuBLAS: main CSV used `CME_WITH_CUBLAS=OFF` (owner default D3). The optional baseline CSV used the
  `linux-release-cublas` preset (`CME_WITH_CUBLAS=ON`) and direct `cublasSgemm` from the CUDA 12.0
  toolkit. No Tensor Core result is claimed.

Each CSV row records the GPU name, driver version, and CUDA runtime version. The measured GPU state
(median of samples taken before and after each configuration) is also in every row.

## Methodology

The binding methodology is in `docs/BENCHMARKING.md` and follows HANDOFF §6.4:

- 5 unmeasured warm-up iterations and 30 measured iterations per configuration.
- Kernel-only time: CUDA events recorded immediately before and after the kernel launch; the stop
  event is synchronized before elapsed time is read.
- End-to-end time: host `std::chrono::steady_clock` around the full public `cme::gemm()` call,
  including host validation, device allocation/copies, launch, and synchronization.
- Transfer time: CUDA events around the copy only (H2D aggregates A and B copies; D2H covers C).
- Reported statistics: median, minimum, maximum, and IQR.
- GFLOP/s = 2.0 * M * N * K / (median_ms * 1e-3) / 1e9. Transfer rows report 0 GFLOP/s because no
  arithmetic is performed.
- Fixed sizes: 256, 512, 1024, 2048, 4096. No sizes or rows were removed after measurement.
- GPU clocks, temperature, and power were sampled with
  `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits`
  before and after each configuration; the CSV stores the median sample.
- CPU reference rows are reported only for 256 and 512 because the scalar reference is slow. They
  are host measurements, not device-kernel measurements.

## Kernel-only median GFLOP/s

Square matrices, M = N = K, alpha = 1, beta = 0. Algorithm/tile specifications are the same ones the
dispatcher exposes. Values are `gflops_median` from the final CSV.

| Algorithm | tile | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|---:|
| CUDA_NAIVE | 0 | 1927.529378 | 2520.615413 | 2700.829618 | 2763.053151 | 2264.934096 |
| CUDA_TILED | 16 | 2348.434518 | 3086.884302 | 3477.864053 | 3735.845543 | 3350.459438 |
| CUDA_TILED | 32 | 2548.179891 | 3160.741579 | 3334.104841 | 3584.109363 | 3176.535999 |
| CUDA_OPTIMIZED | 16 | 2034.095068 | 4766.254508 | 5656.035843 | 6116.374924 | 5593.337515 |
| CUDA_OPTIMIZED | 32 | 2164.243581 | 7906.321992 | 12633.445936 | 14122.236099 | 12210.370568 |

Highest measured project-kernel kernel-only result in the main CSV: **14122.236099 GFLOP/s** for
CUDA_OPTIMIZED tile 32 at 2048x2048x2048. At 4096 the same configuration measured 12210.370568
GFLOP/s with an IQR of 0.555928 ms.

Kernel-only medians are not directly comparable to end-to-end numbers: the kernel-only harness
keeps device buffers allocated and copied once, outside the timed region.

## Optional cuBLAS baseline

Source: `results/gemm_20261122T102414Z_linux-release-cublas.csv`. This optional run used the same
5 warm-up / 30 measured iteration methodology and CUDA-event kernel-only timing. `CUBLAS_REFERENCE`
calls `cublasSgemm` directly on the same preallocated device buffers. Row-major inputs are passed by
swapping operands; no public interface changed. No end-to-end cuBLAS row is produced.

| Size | CUBLAS median ms | CUBLAS IQR ms | CUBLAS GFLOP/s | CUDA_OPTIMIZED tile 32 GFLOP/s | CUBLAS / optimized |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.009808 | 0.001024 | 3421.128813 | 2184.533389 | 1.57x |
| 512 | 0.018608 | 0.000800 | 14425.809014 | 8456.258179 | 1.71x |
| 1024 | 0.100656 | 0.001728 | 21334.879874 | 12633.445936 | 1.69x |
| 2048 | 0.523264 | 0.002048 | 32832.125817 | 14069.865882 | 2.33x |
| 4096 | 4.490208 | 0.035432 | 30608.594571 | 12241.675154 | 2.50x |

The optimized-row values above are from the same optional CSV, not the earlier main CSV. In this run
`CUBLAS_REFERENCE` was faster than every project kernel at all five sizes. The baselines compare
rows produced by one command on the same GPU; min, max, and full GPU state are in the CSV.

## End-to-end median milliseconds

Full public `cme::gemm()`, alpha = 1, beta = 0. Values are `median_ms` from the final CSV. CPU
reference appears only for 256 and 512.

| Algorithm | tile | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|---:|
| CPU_REFERENCE | 0 | 6.607452 | 55.452782 | not measured | not measured | not measured |
| CUDA_NAIVE | 0 | 0.832710 | 1.255162 | 4.212067 | 18.493276 | 108.183267 |
| CUDA_TILED | 16 | 0.641063 | 1.246135 | 4.011719 | 16.839740 | 87.956173 |
| CUDA_TILED | 32 | 0.494945 | 1.231997 | 4.060462 | 17.295091 | 90.934148 |
| CUDA_OPTIMIZED | 16 | 0.393352 | 1.230471 | 3.791351 | 15.043261 | 70.465666 |
| CUDA_OPTIMIZED | 32 | 0.629763 | 1.175209 | 3.560707 | 13.428698 | 57.728051 |

At 2048x2048x2048 the fastest measured end-to-end configuration is CUDA_OPTIMIZED tile 32 at
13.428698 ms, versus 16.839740 ms for the frozen public default CUDA_TILED tile 16. At 4096 the
same comparison is 57.728051 ms versus 87.956173 ms.

## Transfer median milliseconds

Values are `median_ms` from the final CSV. H2D is the combined copy of A and B; D2H is the copy of C.

| Size | h2d_transfer | d2h_transfer |
|---:|---:|---:|
| 256 | 0.052240 | 0.059952 |
| 512 | 0.202640 | 0.209152 |
| 1024 | 0.926176 | 0.562080 |
| 2048 | 3.194608 | 1.924288 |
| 4096 | 14.973728 | 7.563456 |

## Profiler summary

Full commands, workarounds, and exact report contents are in `docs/PROFILING.md`. Reports are under
`results/profiles/`.

- Nsight Systems, end-to-end size 2048 (`nsys_end_to_end_2048.nsys-rep`): baseline kernel durations
  were naive 6.222 ms, tiled 16 5.061 ms, tiled 32 5.186 ms, optimized 16 2.870 ms, and optimized
  32 (2x2 micro-tile) 2.422 ms. After the Batch 7.2 change, the final closeout benchmark measured
  optimized tile 32 at 1.216512 ms kernel-only and 13.428698 ms end-to-end at 2048.
- Nsight Compute baseline at 2048: compute and L1 throughput were near the ceiling for every
  kernel (99.21% naive, 96.46% tiled 16, 84.71% tiled 32, 95.99% optimized 16, 96.70% optimized 32).
  DRAM throughput was low (1.16% to 2.98%), confirming the workload is limited by on-chip work and
  memory-pipeline issue rate rather than DRAM bandwidth.
- Baseline stall evidence identified MIO throttle: 20.99 cycles per issue for tiled 16 and 21.26
  for tiled 32; optimized 32 (2x2) spent 13.84 cycles on MIO throttle, 52.8% of its 26.23 issue
  cycles.
- Batch 7.2 changed only the CUDA_OPTIMIZED tile-32 micro-tile from 2x2 to 4x4 (block 8x8). After
  the change (`ncu_kernel_2048_optimized32_4x4.ncu-rep`), duration fell from 2.73 ms to 1.42 ms,
  MIO throttle fell from 13.84 to 0.05 cycles, and achieved occupancy fell from 96.30% to 39.57%
  without hurting throughput.
- Bank conflicts baseline (load/store): naive 0/0, tiled 16 0/0, tiled 32 0/5,748,813,
  optimized 16 122,966/35,550,923, optimized 32 (2x2) 0/0. After the 4x4 change, optimized 32
  measured 70,173/2,065,317.
- The 4x4 change was kept because kernel-only median GFLOP/s improved by 52.7% at 512, 87.8% at
  1024, 88.7% at 2048, and 78.2% at 4096, each larger than the IQR overlap and above the 5% binding
  threshold. At 256 it was 21.4% slower, so no improvement is claimed for 256.

## Final dispatch decision

`src/cuda/gemm_dispatch.cu` is unchanged. It maps CUDA_NAIVE to the naive kernel, CUDA_TILED to the
tiled kernel, and CUDA_OPTIMIZED to the optimized kernel. The public default remains
`CUDA_TILED` with `tile_size = 16` because that interface is frozen by HANDOFF §4.4 and Batch 1.3.
CUDA_OPTIMIZED tile 32 is the measured fastest choice and must be selected explicitly by the caller.
The frozen default is a known limitation, not an unnoticed result.

## Known limitations

- The public default algorithm is still CUDA_TILED tile 16, so callers who do not set
  `GemmConfig::algorithm` do not get the fastest measured configuration.
- CUDA_OPTIMIZED tile 32 is slower than CUDA_TILED at 256 (0.015504 ms kernel-only versus 0.014288
  ms) and slightly faster than CUDA_OPTIMIZED tile 16 (0.016496 ms). The 4x4 micro-tile is only
  claimed for sizes 512 and above.
- End-to-end time is dominated by allocation and host/device copies at small sizes. At 256 the best
  end-to-end result is 0.393352 ms (CUDA_OPTIMIZED tile 16) while transfer alone is 0.052240 ms plus
  0.059952 ms.
- CUDA_OPTIMIZED tile 32 at 4096 showed clock variation (kernel-only SM clock median 2550.0 MHz,
  lower than the 2745.0 MHz to 2760.0 MHz seen for other kernel-only 4096 rows) and an IQR of
  0.555928 ms. No rows were removed.
- The scalar CPU reference is reported only for 256 and 512 and is not a device measurement.
- The main benchmark CSV has no cuBLAS row (`CME_WITH_CUBLAS=OFF`). The optional
  `CUBLAS_REFERENCE` baseline is in a separate CSV and is faster than the project kernels at every
  measured size.
- Tensor Core behavior is not claimed or characterized. Transposes and batched GEMM remain out of
  scope for this project.
- Nsight Compute required a privileged Docker wrapper because
  `/proc/driver/nvidia/params` has `RmProfilingAdminOnly: 1`. Nsight Systems required the manual
  `QdstrmImporter` step. Both are documented in `docs/PROFILING.md`.

## Reproduction

```bash
cd /home/Ripecoder/work/cuda-matrix-engine
./scripts/benchmark.sh
```

This builds `linux-release`, runs the full benchmark, and writes
`results/gemm_<UTC timestamp>.csv` plus the matching `.env.txt` environment file. For a reduced run:

```bash
./scripts/benchmark.sh --quick
```

The optional cuBLAS baseline is reproduced with:

```bash
CME_BENCHMARK_PRESET=linux-release-cublas ./scripts/benchmark.sh
```

It builds `linux-release-cublas` and writes `results/gemm_<UTC>_linux-release-cublas.csv`.
The main CSV used for this report is `results/gemm_20261122T073654Z.csv`; the cuBLAS CSV is
`results/gemm_20261122T102414Z_linux-release-cublas.csv`.

## Correctness evidence

Recorded by Batch 5.3 on 2026-11-02, re-run during Batch 7.3 on 2026-11-03, and re-run during
closeout on 2026-11-22 after four Phase 8 edge-case tests were corrected.

- Command: `CME_COMPUTE_SANITIZER=/tmp/opencode/cuda-sanitizer-13-2/usr/local/cuda-13.2/bin/compute-sanitizer ./scripts/test.sh`
- Build: `cmake --preset linux-debug` and `cmake --build --preset linux-debug` completed.
- CTest: `100% tests passed, 0 tests failed out of 48`; 47 Catch2 test cases and 5423 assertions.
- Correctness matrix: 600 algorithm runs, 16,305,450 element comparisons.
- `compute-sanitizer` version: `2026.1.1.0` (build 38000597), CUDA 13.2 package.
- Memcheck on the full test binary: `ERROR SUMMARY: 0 errors`.
- Racecheck on the `[racecheck]` filter: `0 hazards displayed (0 errors, 0 warnings)`.
- Raw run log: `/tmp/cme_verify_test.log` (session artifact; not committed).

Environment: `docs/ENVIRONMENT.md`. The system `compute-sanitizer` 2022.4.1 is incompatible with the
installed 595.84 driver; `scripts/test.sh` selects the working 13.2 binary. See Batch 5.3 in
`docs/DESIGN_DECISIONS.md`.
