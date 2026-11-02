# Benchmark Results

This directory stores reproducible GEMM benchmark runs and associated environment reports.

## File Naming Conventions

Each run produces two corresponding files:
1. **`gemm_<YYYYMMDD_HHMMSS>.csv`**:
   The primary benchmark dataset containing 19 standardized RFC 4180 columns:
   ```
   timestamp_utc,algorithm,M,N,K,tile_size,measurement,iterations,median_ms,min_ms,max_ms,iqr_ms,gflops_median,gpu_name,driver_version,cuda_runtime_version,sm_clock_mhz_median,temperature_c_median,power_w_median
   ```
   - `measurement` identifies the timing scope:
     - `kernel_only`: Isolated GPU kernel duration timed with CUDA events.
     - `end_to_end`: Public `cme::gemm()` execution timed with host wall clock (`std::chrono::steady_clock`).
     - `h2d_transfer`: Host-to-device memory copy duration for input matrices A and B.
     - `d2h_transfer`: Device-to-host memory copy duration for output matrix C.

2. **`gemm_<YYYYMMDD_HHMMSS>.csv.env.txt`**:
   The corresponding environment snapshot generated at run time, recording GPU state, driver version, compiler toolchain, and OS information.

## How to Reproduce

### Full Benchmark
To run the full benchmark suite across all target dimensions ($M = N = K \in \{256, 512, 1024, 2048, 4096\}$):

**Linux / Bash:**
```bash
./scripts/benchmark.sh
```

**Windows PowerShell:**
```powershell
.\scripts\benchmark.ps1
```

### Quick Verification
To run a fast smoke test across sizes 256 and 512:

**Linux / Bash:**
```bash
./scripts/benchmark.sh --quick
```

**Windows PowerShell:**
```powershell
.\scripts\benchmark.ps1 -Quick
```

### Direct Binary Execution
You can also run the built benchmark executable directly:
```bash
./build/<preset>/benchmarks/benchmark_main --output results/my_run.csv
```
Options:
- `--output <path>`: Destination CSV path.
- `--quick`: Quick smoke test (dimensions 256 and 512, reduced iterations).
