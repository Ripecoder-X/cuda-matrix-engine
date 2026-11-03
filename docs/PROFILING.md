# Profiling

Baseline profiles for Batch 7.1 and the measured optimization iteration from Batch 7.2.
Every profiler value below is copied from the saved report named in the same row or
section. No value is estimated.

## Environment and setup

- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, compute capability 8.9.
- Driver: 595.84.
- CUDA toolkit used to build: 12.0.140.
- Nsight Systems: 2022.4.2.
- Nsight Compute: 2022.4.1.0 (build 32308335).
- Profile size: 2048 x 2048 x 2048, FP32, row-major, `alpha = 1.0`, `beta = 0.0`.
- Build preset: `linux-release`.
- Nsight Compute reports the SM frequency as `2.34 cycle/nsecond` for all
  captured kernels.

Nsight Systems captured the full end-to-end benchmark (all device algorithms,
two calls each: one warm-up and one measured). Nsight Compute captured the
kernel-only benchmark with `--warmup 0 --iterations 1`, so each report contains
the baseline launch of the selected kernel after one warm-up was removed from
the timed path.

Reproduce the captures with:

```bash
./scripts/profile.sh
```

The script writes reports to `results/profiles/`.

## Nsight Systems: end-to-end, size 2048

Report: `results/profiles/nsys_end_to_end_2048.nsys-rep`

Command:

```bash
nsys profile \
    --force-overwrite true \
    --trace=cuda,osrt \
    --sample=none \
    --cpuctxsw=none \
    --output /tmp/nsys_end_to_end_2048 \
    ./build/linux-release/benchmarks/cme_benchmark \
    --sizes 2048 --measurements end_to_end --warmup 1 --iterations 1 \
    --output /tmp/nsys_run.csv
```

On this host `nsys` produced a `.qdstrm` file and failed to import it. The
report was converted with:

```bash
/usr/lib/nsight-systems/host-linux-x64/QdstrmImporter \
    -i /tmp/nsys_end_to_end_2048.qdstrm \
    -o results/profiles/nsys_end_to_end_2048.nsys-rep -f
```

Kernel summary from
`results/profiles/nsys_end_to_end_2048_gpukernsum_gpukernsum.csv`
(`nsys stats -r gpukernsum`):

| Kernel | Instances | Avg (ns) | Med (ns) | Min (ns) | Max (ns) | Grid | Block |
|---|---:|---:|---:|---:|---:|---|---|
| `gemm_naive_kernel` | 2 | 6,222,185.0 | 6,222,185.0 | 6,221,785 | 6,222,585 | 128x128x1 | 16x16x1 |
| `gemm_tiled_kernel<(int)32>` | 2 | 5,186,432.0 | 5,186,432.0 | 5,026,748 | 5,346,116 | 64x64x1 | 32x32x1 |
| `gemm_tiled_kernel<(int)16>` | 2 | 5,060,908.0 | 5,060,908.0 | 4,803,542 | 5,318,274 | 128x128x1 | 16x16x1 |
| `gemm_optimized_kernel<(int)16>` | 2 | 2,870,358.5 | 2,870,358.5 | 2,870,215 | 2,870,502 | 128x128x1 | 8x8x1 |
| `gemm_optimized_kernel<(int)32>` | 2 | 2,422,060.0 | 2,422,060.0 | 2,422,044 | 2,422,076 | 64x64x1 | 16x16x1 |

These durations cover the kernel launches inside the end-to-end calls. The
end-to-end measurement also includes host validation, allocation, and copies,
so its wall-clock values are larger than the kernel durations above.

## Nsight Compute: baseline kernels, size 2048

Each report was captured with:

```bash
ncu --target-processes all --kernel-name-base demangled \
    --kernel-name 'regex:.*<kernel>.*' \
    --launch-skip 0 --launch-count 1 \
    --section SpeedOfLight --section Occupancy --section WarpStateStats \
    --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables \
    -o results/profiles/ncu_kernel_2048_<name> \
    ./build/linux-release/benchmarks/cme_benchmark \
    --sizes 2048 --measurements kernel_only --warmup 0 --iterations 1 \
    --output /tmp/ncu_<name>.csv
```

`--launch-skip 1` was used for the second tile-size variant of the tiled and
optimized kernels. On this host direct `ncu` fails with `ERR_NVGPUCTRPERM`
because `/proc/driver/nvidia/params` has `RmProfilingAdminOnly: 1`; the script
runs the same command inside a privileged `ubuntu:24.04` container with
`chroot /host` (see `scripts/profile.sh`). Report paths below are relative to
`results/profiles/`.

Read a report with:

```bash
ncu --import results/profiles/ncu_kernel_2048_optimized32.ncu-rep --page details
ncu --import results/profiles/ncu_kernel_2048_optimized32.ncu-rep --page raw
```

### Throughput and occupancy

| Metric | naive | tiled16 | tiled32 | optimized16 | optimized32 |
|---|---:|---:|---:|---:|---:|
| Report | `ncu_kernel_2048_naive.ncu-rep` | `ncu_kernel_2048_tiled16.ncu-rep` | `ncu_kernel_2048_tiled32.ncu-rep` | `ncu_kernel_2048_optimized16.ncu-rep` | `ncu_kernel_2048_optimized32.ncu-rep` |
| Duration (ms) | 7.01 | 5.41 | 5.65 | 3.24 | 2.73 |
| Memory Throughput (%) | 99.21 | 96.46 | 84.71 | 95.99 | 96.70 |
| DRAM Throughput (%) | 1.16 | 0.98 | 1.49 | 2.50 | 2.98 |
| L1/TEX Cache Throughput (%) | 99.53 | 96.73 | 85.47 | 96.36 | 97.96 |
| L2 Cache Throughput (%) | 30.51 | 26.13 | 12.60 | 39.48 | 26.06 |
| Compute (SM) Throughput (%) | 99.21 | 96.46 | 84.71 | 95.99 | 96.70 |
| Theoretical Occupancy (%) | 100 | 100 | 66.67 | 100 | 100 |
| Achieved Occupancy (%) | 99.02 | 99.04 | 66.65 | 96.45 | 96.30 |
| Warp Cycles Per Issued Instruction | 43.46 | 39.85 | 34.53 | 22.41 | 26.23 |

The tiled32 kernel reports three simultaneous theoretical occupancy limits:
6 blocks per SM by registers, 6 blocks per SM by shared memory, and 6 blocks
per SM by warps. The other kernels report a theoretical occupancy of 100%.
The `Memory Throughput (%)` row is the GPU Speed Of Light value; the Memory
Workload Analysis section reports the corresponding absolute values as
7.58 GB/s (naive), 6.43 GB/s (tiled16), 9.78 GB/s (tiled32),
16.39 GB/s (optimized16), and 19.54 GB/s (optimized32).

### Shared-memory bank conflicts

Values are from the raw reports:

| Metric | naive | tiled16 | tiled32 | optimized16 | optimized32 |
|---|---:|---:|---:|---:|---:|
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | 0 | 0 | 0 | 122,966 | 0 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` | 0 | 0 | 5,748,813 | 35,550,923 | 0 |

### Warp stall reasons

Top stall reasons from the raw reports. Values are
`smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio` in cycles.

| Kernel | Reason 1 | Reason 2 | Reason 3 | Total cycles per issued instruction |
|---|---|---:|---:|---:|
| naive | `lg_throttle` 27.80 | `long_scoreboard` 6.04 | `not_selected` 4.70 | 43.46 |
| tiled16 | `mio_throttle` 20.99 | `barrier` 6.13 | `long_scoreboard` 5.21 | 39.85 |
| tiled32 | `mio_throttle` 21.26 | `not_selected` 3.69 | `long_scoreboard` 3.60 | 34.53 |
| optimized16 | `mio_throttle` 9.67 | `not_selected` 3.22 | `long_scoreboard` 2.26 | 22.41 |
| optimized32 | `mio_throttle` 13.84 | `not_selected` 3.65 | `short_scoreboard` 1.66 | 26.23 |

Nsight Compute reports the tiled32 and optimized32 `mio_throttle` values as
21.3 cycles (61.6% of 34.5) and 13.8 cycles (52.8% of 26.2) respectively. For
naive, the dominant stall is `lg_throttle` at 27.80 cycles of 43.46. For
tiled16 and optimized16 the dominant stall is `mio_throttle` at 20.99 of 39.85
and 9.67 of 22.41 cycles.

The report warnings also state that global stores in optimized16 and
optimized32 access `5.6` and `5.9` bytes per thread per request but transfer
`8.0` sectors (256.0 bytes) per request. The tiled16, tiled32, and optimized16
loads from L1TEX to L2 access `2.0` of the possible `4` sectors per cache line;
the naive loads access `1.3` of `4`, and naive stores access `2.0` of `4`.


## Batch 7.2 optimization iteration

### Hypothesis

The baseline optimized32 kernel is limited by shared-memory instruction throughput:
Nsight Compute reports `mio_throttle` at 13.84 cycles per issued instruction
(52.8% of 26.23 total cycles), zero shared-load/store bank conflicts, and 96.70%
compute and memory throughput. Increasing the register micro-tile from 2x2 to 4x4
should halve the shared-memory loads per multiply-add and reduce that pressure.

### Change

Only `src/cuda/kernels/gemm_optimized.cu` changed. TILE=32 now uses a 4x4 register
micro-tile with an 8x8 block; TILE=16 remains at the previous 2x2 micro-tile with an
8x8 block. The shared tile is still padded by one float and the inner loops remain
compile-time unrolled.

### Correctness

`ctest --preset linux-release --output-on-failure` passed 44/44 tests after the change.

### Kernel-only benchmark

Baseline CSV: `results/gemm_20261103T062036Z.csv`.
Iteration CSV: `results/profiles/kernel_only_optimized32_4x4.csv`.
Both used 30 measured iterations with the release preset.

| Size | Baseline median ms | Baseline GFLOP/s | 4x4 median ms | 4x4 GFLOP/s | Change in GFLOP/s | Baseline IQR ms | 4x4 IQR ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.011264 | 2,978.909068 | 0.014336 | 2,340.571396 | -21.4% | 0.000128 | 0.000224 |
| 512 | 0.046896 | 5,724.058749 | 0.030720 | 8,738.133554 | +52.7% | 0.000864 | 0.000344 |
| 1024 | 0.319408 | 6,723.324561 | 0.170080 | 12,626.315056 | +87.8% | 0.001016 | 0.000760 |
| 2048 | 2.317200 | 7,414.064210 | 1.227792 | 13,992.491269 | +88.7% | 0.004760 | 0.046496 |
| 4096 | 19.949984 | 6,889.176264 | 11.194032 | 12,277.877275 | +78.2% | 0.574968 | 0.955073 |

The 5% rule is met at sizes 512, 1024, 2048, and 4096. At 256 the change is slower,
so the 4x4 TILE=32 variant is not claimed for that size. The 2x2 TILE=16 variant is
unchanged and keeps its baseline values.

### Nsight Compute after the change

Report: `results/profiles/ncu_kernel_2048_optimized32_4x4.ncu-rep`.
Captured with the same command and sections as the baseline, at size 2048.

| Metric | Baseline 2x2 | 4x4 |
|---|---:|---:|
| Duration (ms) | 2.73 | 1.42 |
| Memory Throughput (%) | 96.70 | 73.70 |
| DRAM Throughput (%) | 2.98 | 5.69 |
| L1/TEX Cache Throughput (%) | 97.96 | 75.13 |
| L2 Cache Throughput (%) | 26.06 | 46.74 |
| Compute (SM) Throughput (%) | 96.70 | 73.70 |
| Theoretical Occupancy (%) | 100 | 41.67 |
| Achieved Occupancy (%) | 96.30 | 39.57 |
| Warp Cycles Per Issued Instruction | 26.23 | 6.96 |
| `mio_throttle` cycles per issued instruction | 13.84 | 0.05 |
| `not_selected` cycles per issued instruction | 3.65 | 1.88 |
| `short_scoreboard` cycles per issued instruction | 1.66 | 1.04 |
| Shared-load bank conflicts | 0 | 70,173 |
| Shared-store bank conflicts | 0 | 2,065,317 |

The 4x4 kernel is no longer MIO-throttle limited. Its lower occupancy is not
occupancy-limited by warps: the theoretical limit is 41.67% because shared memory
allows 10 blocks per SM and each block has two warps. The profile shows the change is
faster despite lower occupancy, so the measured kernel-only result decides the
iteration.

### Decision

Keep the 4x4 TILE=32 variant. It meets the 5% rule at sizes 512 to 4096, with the
improvement far larger than the IQR overlap at those sizes. Revert is not needed.
`CUDA_OPTIMIZED` remains an explicit algorithm choice; the public default
`CUDA_TILED`/16 is unchanged because section 4.4 freezes it.
