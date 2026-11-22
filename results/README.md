# Benchmark Results

This directory holds benchmark CSV files produced by `scripts/benchmark.sh` and profiler captures
produced by `scripts/profile.sh`.

## File naming

- `gemm_<timestamp>.csv` — full benchmark run. Timestamp format is `YYYYMMDDTHHMMSSZ` in UTC.
- `gemm_<timestamp>_quick.csv` — smoke-test run produced by `scripts/benchmark.sh --quick`. It uses
  sizes 256 and 512 only and fewer iterations. Do not cite quick results as full-run evidence.
- `gemm_<timestamp>_<preset>.csv` — full run with a non-default preset. Example:
  `gemm_<timestamp>_linux-release-cublas.csv` contains `CUBLAS_REFERENCE` rows.
- `gemm_<timestamp>_<preset>_quick.csv` — smoke run with a non-default preset.
- `gemm_<timestamp>.env.txt` — output of `scripts/environment.sh` captured beside the CSV.
- `profiles/` — Nsight Systems and Nsight Compute reports. File names identify the kernel and size.

## Reproduce

```sh
./scripts/benchmark.sh
```

For a fast smoke test:

```sh
./scripts/benchmark.sh --quick
```

For the optional cuBLAS baseline:

```sh
CME_BENCHMARK_PRESET=linux-release-cublas ./scripts/benchmark.sh
```

## Reading the CSV

The column order is frozen in `docs/development/HANDOFF.md` section 4.13. One row is one algorithm, size, and
measurement type. `measurement` is one of `kernel_only`, `end_to_end`, `h2d_transfer`, or
`d2h_transfer`. Times are milliseconds. `gflops_median` is zero for transfer rows because a transfer
does not perform GEMM arithmetic.

Every row records the GPU name, driver version, CUDA runtime version, and the median GPU clock,
temperature, and power sample for that configuration. `NOT MEASURED` means the value could not be
read from `nvidia-smi`; it is never estimated.

Do not remove outliers or select a favorable file. The methodology in `docs/BENCHMARKING.md` fixes
the warm-up count, measured count, sizes, and summary statistics.
