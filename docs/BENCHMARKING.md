# Benchmarking

The benchmark writes machine-readable CSV files. Every number in this document is a methodology
rule, not a measured result. Measured results live in `results/` and are summarized in
`docs/PERFORMANCE_REPORT.md`.

## Methodology (binding)

- Warm-up: 5 iterations, not measured.
- Measured: 30 iterations per configuration.
- Kernel time: CUDA events around the launch only, on the same stream, with a `cudaEventSynchronize`
  before reading elapsed time.
- End-to-end time: host wall clock (`std::chrono::steady_clock`) around the full public `gemm()` call.
- Transfer time: CUDA events around `cudaMemcpy`/`cudaMemcpy2D` only.
- Report **median, min, max, and IQR**. Do not report only the mean.
- GFLOP/s = `2.0 * M * N * K / (median_ms * 1e-3) / 1e9`. Record the formula in the docs.
- Sizes: square 256, 512, 1024, 2048, 4096.
- Record GPU clocks, temperature, and power **before and after** each configuration using
  `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits`.
  Report the median of samples taken during the run (sample on a background thread or between
  iterations; document the choice).
- Record GPU name, driver version, CUDA runtime version in every CSV row.

## GFLOP/s formula

For every GEMM row:

```
gflops_median = 2.0 * M * N * K / (median_ms * 1.0e-3) / 1.0e9
```

Transfer rows report `gflops_median = 0.000000` because they move bytes and do not perform GEMM
arithmetic.

## Kernel-only benchmark exception

HANDOFF.md section 5.2 normally limits benchmarks to `include/cme/*` headers. The kernel-only
benchmark is the one permitted exception: `benchmarks/benchmark_gemm.cu` includes
`src/cuda/gemm_dispatch.hpp` and `src/cuda/device_buffer.hpp` so it can call `gemm_device` and
allocate device buffers directly. This isolates kernel launch time from host allocation, host copies,
and API validation. The end-to-end benchmark uses only the public API.

## Implementation notes

- **Percentiles.** Samples are sorted ascending. The median is p50 and IQR is p75 - p25. Percentiles
  use linear interpolation between the closest ranks, the same convention as NumPy's default
  `linear` method.
- **Kernel-only loop.** Device buffers for each size are allocated once, outside the timed region.
  Each measured iteration records a start event, calls `gemm_device`, records a stop event, then
  synchronizes the stop event before reading elapsed time.
- **End-to-end loop.** One `std::chrono::steady_clock` pair surrounds each full public `gemm()` call.
  The public API is synchronous, so each sample includes validation, allocation, host-to-device
  copies, kernel launch, device synchronization, and device-to-host copy.
- **CPU reference scope.** `CPU_REFERENCE` end-to-end rows are produced for sizes 256 and 512 only.
  The reference is a scalar host loop and is included for context, not as a device measurement. Larger
  CPU sizes would make the full run impractically long. No CUDA algorithm is limited.
- **Transfer aggregation.** `h2d_transfer` times the host-to-device copy of A followed by B in one
  event interval. `d2h_transfer` times the device-to-host copy of C. Warm-up copies run before the
  measured loop.
- **GPU state sampling.** Each configuration takes one `nvidia-smi` sample before the warm-up and one
  after the measured loop. The CSV reports the median of the valid samples. This satisfies the
  before-and-after requirement and reports a median without perturbing the timed region.
- **Quick mode.** `--quick` sets sizes to 256 and 512, warm-up to 2, and measured iterations to 5.
  Quick output is a smoke test and is not evidence for performance claims.
- **Driver and toolkit.** The installed `nvcc` is CUDA 12.0. The CUDA runtime version reported in each
  row comes from `cudaRuntimeGetVersion`; the driver version comes from `nvidia-smi`. They are separate
  values.
- **cuBLAS.** Optional `CUBLAS_REFERENCE` rows are off by default (`CME_WITH_CUBLAS=OFF`, owner
  decision D3). No cuBLAS row is produced or claimed unless the option is enabled and a non-optimized
  `cublasSgemm` row appears in the CSV.

## Running

```sh
./scripts/benchmark.sh          # full run, linux-release, results/gemm_<timestamp>.csv
./scripts/benchmark.sh --quick  # smoke run, sizes 256 and 512
```

The benchmark executable also accepts `--output <path>`, `--sizes <list>`, `--measurements <list>`,
`--warmup <n>`, and `--iterations <n>`. `scripts/profile.sh` uses the narrower options to isolate
one size and measurement type.
