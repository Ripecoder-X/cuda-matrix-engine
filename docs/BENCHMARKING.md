# Benchmarking Methodology

This document defines the methodology, measurement protocols, and reproducibility standards for `cuda-matrix-engine`.

## Methodology (Binding)

- **Warm-up:** 5 iterations, not measured.
- **Measured:** 30 iterations per configuration.
- **Kernel time:** CUDA events around the launch only, on the same stream, with a `cudaEventSynchronize` before reading elapsed time.
- **End-to-end time:** Host wall clock (`std::chrono::steady_clock`) around the full public `gemm()` call.
- **Transfer time:** CUDA events around `cudaMemcpy`/`cudaMemcpy2D` only.
- **Distribution metrics:** Report **median, min, max, and IQR** (Interquartile Range, $Q_3 - Q_1$). Do not report only the mean.
- **GFLOP/s formula:**
  $$\text{GFLOP/s} = \frac{2.0 \times M \times N \times K}{\text{median\_ms} \times 10^{-3} \times 10^9}$$
- **Sizes:** Square matrices $M = N = K \in \{256, 512, 1024, 2048, 4096\}$.
- **Telemetry:** Record GPU clocks, temperature, and power **before and after** each configuration using:
  ```bash
  nvidia-smi --query-gpu=driver_version,clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits
  ```
  The median or average of before and after readings is reported in the telemetry columns.
- **Hardware tracking:** Every CSV row records the GPU name, driver version, and CUDA runtime version.

## Architecture Exception (Section 5.2)

To isolate pure GPU kernel compute performance from host-side memory allocations and PCIe transfers, benchmark drivers are permitted to bypass the public host API and directly call internal device primitives:
- `cme::detail::gemm_device` (kernel launch dispatch)
- `cme::detail::DeviceBuffer` (device memory allocation)

This exception applies strictly to the `kernel_only` measurement type. All `end_to_end` benchmarks use the public `cme::gemm()` interface with host-managed `cme::Matrix` containers.

## Reproducing Benchmarks

### Quick Verification (Smoke test)
To run a fast smoke test across sizes 256 and 512:
```bash
./scripts/benchmark.sh --quick
# On Windows PowerShell:
.\scripts\benchmark.ps1 -Quick
```

### Full Benchmark
To execute the complete benchmark suite across all sizes {256, 512, 1024, 2048, 4096} with 30 measured iterations:
```bash
./scripts/benchmark.sh
# On Windows PowerShell:
.\scripts\benchmark.ps1
```
Results and machine environment logs will be saved to `results/gemm_<YYYYMMDD_HHMMSS>.csv` and `results/gemm_<YYYYMMDD_HHMMSS>.csv.env.txt`.
