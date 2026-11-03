# Performance Report

Final report produced by Batch 7.3 on 2026-11-03. All numbers below are copied from the final
benchmark CSV `results/gemm_20261103T062611Z.csv` (63 lines: one frozen header plus 62 data rows).
Profiler evidence is in `results/profiles/` and summarized in `docs/PROFILING.md`.

## Hardware and software

Source: `docs/ENVIRONMENT.md`.

- CPU: Intel Core i9-14900K, 32 logical CPUs, 62 GiB system RAM.
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB, compute capability 8.9 (sm_89).
- GPU driver: 595.84; CUDA version reported by the driver: 13.2.
- Installed CUDA toolkit: 12.0 (`nvcc` V12.0.140). The benchmark records the CUDA runtime version
  reported by `cudaRuntimeGetVersion()`: 12.0.
- OS: Pop!_OS 24.04 LTS, kernel 7.0.11-76070011-generic.
- Build: CMake 3.28.3, g++ 13.3.0, Release build (`linux-release`), `CME_CUDA_ARCHITECTURES=89`.
- cuBLAS: `CME_WITH_CUBLAS=OFF` (owner default D3). No cuBLAS row exists in the CSV, so cuBLAS is
  **not measured**.

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
| CUDA_NAIVE | 0 | 1936.428473 | 2508.930103 | 2702.515563 | 2762.591182 | 2276.460497 |
| CUDA_TILED | 16 | 2312.185278 | 3196.878101 | 3489.346993 | 3737.159001 | 3378.671564 |
| CUDA_TILED | 32 | 2545.087416 | 3158.361484 | 3334.104841 | 3594.511858 | 3132.970473 |
| CUDA_OPTIMIZED | 16 | 1882.542217 | 4599.017455 | 5602.209334 | 6126.600404 | 5533.724808 |
| CUDA_OPTIMIZED | 32 | 2286.970562 | 8456.258179 | 12409.182971 | 14075.398984 | 12077.491619 |

Highest measured kernel-only result: **14075.398984 GFLOP/s** for CUDA_OPTIMIZED tile 32 at
2048x2048x2048. At 4096 the same configuration measured 12077.491619 GFLOP/s with an IQR of
0.924616 ms.

Kernel-only medians are not directly comparable to end-to-end numbers: the kernel-only harness
keeps device buffers allocated and copied once, outside the timed region.

## End-to-end median milliseconds

Full public `cme::gemm()`, alpha = 1, beta = 0. Values are `median_ms` from the final CSV. CPU
reference appears only for 256 and 512.

| Algorithm | tile | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|---:|
| CPU_REFERENCE | 0 | 7.102941 | 60.035730 | not measured | not measured | not measured |
| CUDA_NAIVE | 0 | 0.313442 | 1.166422 | 2.761772 | 12.739729 | 86.625762 |
| CUDA_TILED | 16 | 0.301804 | 0.876799 | 2.658532 | 11.238789 | 66.882886 |
| CUDA_TILED | 32 | 0.292778 | 0.905292 | 2.636362 | 11.537349 | 69.944105 |
| CUDA_OPTIMIZED | 16 | 0.495186 | 0.876290 | 2.325485 | 9.389532 | 49.191942 |
| CUDA_OPTIMIZED | 32 | 0.333952 | 0.852131 | 2.151030 | 7.952398 | 36.556368 |

At 2048x2048x2048 the fastest measured end-to-end configuration is CUDA_OPTIMIZED tile 32 at
7.952398 ms, versus 11.238789 ms for the frozen public default CUDA_TILED tile 16. At 4096 the
same comparison is 36.556368 ms versus 66.882886 ms.

## Transfer median milliseconds

Values are `median_ms` from the final CSV. H2D is the combined copy of A and B; D2H is the copy of C.

| Size | h2d_transfer | d2h_transfer |
|---:|---:|---:|
| 256 | 0.052432 | 0.037520 |
| 512 | 0.240416 | 0.175936 |
| 1024 | 0.801792 | 0.576816 |
| 2048 | 3.294688 | 1.947344 |
| 4096 | 15.146880 | 7.515744 |

## Profiler summary

Full commands, workarounds, and exact report contents are in `docs/PROFILING.md`. Reports are under
`results/profiles/`.

- Nsight Systems, end-to-end size 2048 (`nsys_end_to_end_2048.nsys-rep`): baseline kernel durations
  were naive 6.222 ms, tiled 16 5.061 ms, tiled 32 5.186 ms, optimized 16 2.870 ms, and optimized
  32 (2x2 micro-tile) 2.422 ms. After the Batch 7.2 change, the full benchmark measured optimized
  tile 32 at 1.220560 ms kernel-only and 7.952398 ms end-to-end at 2048.
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
- CUDA_OPTIMIZED tile 32 is slower than CUDA_OPTIMIZED tile 16 and CUDA_TILED at 256 (0.014672 ms
  kernel-only versus 0.017824 ms and 0.013184 ms respectively). The 4x4 micro-tile is only claimed
  for sizes 512 and above.
- End-to-end time is dominated by allocation and host/device copies at small sizes. At 256 the best
  end-to-end result is 0.292778 ms while transfer alone is 0.052432 ms plus 0.037520 ms.
- CUDA_OPTIMIZED tile 32 at 4096 showed clock variation (SM clock median 2482.5 MHz, lower than the
  2760 MHz seen for several other 4096 rows) and a wider IQR of 0.924616 ms. No rows were removed.
- The scalar CPU reference is reported only for 256 and 512 and is not a device measurement.
- cuBLAS was not measured (`CME_WITH_CUBLAS=OFF`).
- Tensor Cores, transposes, and batched GEMM are out of scope for this project.
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

The final CSV used for this report is `results/gemm_20261103T062611Z.csv`.

## Correctness evidence

Recorded by Batch 5.3 on 2026-11-02 and re-run during Batch 7.3 on 2026-11-03.

- Command: `./scripts/test.sh`
- Build: `cmake --preset linux-debug` and `cmake --build --preset linux-debug` completed.
- CTest: `100% tests passed, 0 tests failed out of 44`; 5368 assertions.
- Correctness matrix: 600 algorithm runs, 16,305,450 element comparisons.
- `compute-sanitizer` version: 2026.1.1.0 (CUDA 13.2 package).
- Memcheck on the full test binary: `ERROR SUMMARY: 0 errors`.
- Racecheck on the `[racecheck]` filter: `0 hazards displayed (0 errors, 0 warnings)`.
- Raw run log: `/tmp/cme_test_sh.log` (session artifact; not committed).

Environment: `docs/ENVIRONMENT.md`. The system `compute-sanitizer` 2022.4.1 is incompatible with the
installed 595.84 driver; `scripts/test.sh` selects the working 13.2 binary. See Batch 5.3 in
`docs/DESIGN_DECISIONS.md`.
