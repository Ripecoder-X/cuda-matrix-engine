# Benchmark Results

This directory holds benchmark CSV files produced by `scripts/benchmark.sh` and profiler captures
produced by `scripts/profile.sh`.

## Files

- `gemm_20261103T062611Z.csv` (+ `.env.txt`): the final full benchmark run, 62 data rows. The
  performance report uses this file.
- `gemm_20261103T062036Z.csv` (+ `.env.txt`): an earlier full run from the same day, kept as recorded.
- `gemm_20261103T062031Z_quick.csv` (+ `.env.txt`): a quick smoke-test run with sizes 256 and 512 only.
  Do not cite quick results as full-run evidence.
- `profiles/`: Nsight Systems (`nsys_*`) and Nsight Compute (`ncu_*`) reports at size 2048, plus a
  kernel-only CSV for the optimized tile-32 kernel and a GPU kernel summary CSV.

## File naming

- `gemm_<timestamp>.csv`: full run. The timestamp is `YYYYMMDDTHHMMSSZ` in UTC.
- `gemm_<timestamp>_quick.csv`: output of `scripts/benchmark.sh --quick`.
- `gemm_<timestamp>.env.txt`: output of `scripts/environment.sh` captured beside the CSV.

## Reproduce

```sh
./scripts/benchmark.sh
```

For a fast smoke test:

```sh
./scripts/benchmark.sh --quick
```

## Reading the CSV

The column order is frozen in `HANDOFF.md` section 4.13. One row is one algorithm, size, tile size,
and measurement type.

- `measurement` is one of `kernel_only`, `end_to_end`, `h2d_transfer`, or `d2h_transfer`.
- `algorithm` is `TRANSFER` for transfer rows, and `CPU_REFERENCE` for the host reference.
- `tile_size` is `0` where it does not apply (naive, CPU reference, transfers).
- Times are milliseconds. `gflops_median` is zero for transfer rows, which perform no GEMM
  arithmetic.
- Every row records the GPU name, driver version, CUDA runtime version, and the median GPU clock,
  temperature, and power sample for that configuration. `NOT MEASURED` means the value could not be
  read from `nvidia-smi`; it is never estimated.

Do not remove outliers or select a favorable file. The methodology in `docs/BENCHMARKING.md` fixes
the warm-up count, measured count, sizes, and summary statistics.
