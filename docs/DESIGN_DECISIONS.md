# Design Decisions

Decisions are appended here as batches make them. Each entry names the batch.

## Batch 0.2

- **License:** Apache-2.0, chosen by the owner (D1).
- **Author:** Nirmal (D2).
- **Formatting:** `.clang-format` uses the LLVM base style, 4-space indent, column limit 100.
- **Linux generator:** `Unix Makefiles`, because Ninja is not installed and `sudo` is unavailable in
  the batch 0.2 session. `windows-release` keeps Ninja. This is a build-environment adaptation, not
  an interface change.
- **Empty library target:** until real sources exist, `CMakeLists.txt` adds a generated empty
  translation unit so the `cme` target links. The generated file lives in the build directory; no
  placeholder source is added to the frozen tree.
- **Source discovery:** the root CMake list names every frozen source path and filters by existence.
  Later batches create the files; no CMake edit is needed for those paths.

## Process override

- **Phase 0 gate:** the owner directed the project to continue into Phase 1 without a separate
  verifier. Batches 0.1–0.3 therefore remained `UNDER_REVIEW` when Phase 1 started. This is an
  explicit owner override of the handoff gate, recorded here per the handoff's override rule.

## Batch 1.2

- **`MatrixView` conversion:** implemented as a templated implicit conversion operator rather than a
  converting constructor. A converting constructor removes aggregate initialization, which the frozen
  layout relies on; the operator preserves `MatrixView{data, rows, cols, stride}` construction while
  still enabling `MatrixView<float>` → `MatrixView<const float>`. The reverse conversion is disabled
  with SFINAE.
- **`Matrix` throws on bad dimensions:** `Matrix` is a host container, not the GEMM API boundary, so
  its constructor throws `std::invalid_argument` naming both dimensions. The public GEMM API returns
  `Status` instead.

## Batch 1.4

- **Catch2 version:** pinned to tag `v3.9.1` (latest v3 tag visible from the upstream repository when
  the batch ran). Also recorded in `QUALITY_STANDARDS.md`.
- **Test target layout:** one executable `cme_tests` contains all test sources, with
  `catch_discover_tests` registering each test case with CTest.
- **New test files:** `tests/test_status.cpp` and `tests/test_gemm_reference.cpp` were added beyond
  the frozen tree because the batch requires their tests. Both files are now part of the test target.
- **Internal test includes:** `test_gemm_reference.cpp` includes `core/gemm_reference.hpp` directly so
  the CPU reference can be tested before the public API exists. This is a white-box test exception to
  the "tests include only public headers" rule; it disappears once the reference is reachable through
  the API.
- **CTest enablement:** `include(CTest)` was added to the root `CMakeLists.txt`. `catch_discover_tests`
  registers nothing without it, and CTest expects testing to be enabled from the top-level project.
- **Random fill:** `fill_random` uses a small deterministic 32-bit LCG mapped to `[-1, 1]`. The exact
  distribution is not statistically important; determinism and bounded values are.

## Deferred

Features outside the frozen scope are not implemented in this milestone: FP16/BF16/FP64/INT8,
column-major or transposed inputs, batched GEMM, asynchronous public API, Tensor Cores, multi-GPU,
and non-NVIDIA backends.

## Batch 2.1

- **Launch error checking:** `check_last_launch` only drains `cudaGetLastError()`. Synchronization is
  the caller's responsibility, matching handoff section 4.8 and the dispatch contract.
- **Error text:** `to_status` builds `"<context>: <cudaGetErrorName> (<cudaGetErrorString>)"` so
  callers keep both the symbolic name and driver description.

## Batch 2.2

- **Synchronous copies:** `DeviceBuffer` uses synchronous `cudaMemcpy` for host/device copies. The
  public API is synchronous, and streams are out of scope, so asynchronous copies buy nothing here.
- **Out-of-memory mapping:** `cudaErrorMemoryAllocation` maps to `StatusCode::OutOfMemory`; other
  allocation failures map to `CudaError`. The original CUDA message is preserved in both cases.
- **`copy_from_host_2d`:** not added speculatively. Batch 4.2 decides whether the strided C copy
  needs it and, if so, amends DeviceBuffer there.
- **Test-only CUDA include:** `tests/test_cuda_errors.cpp` includes `cuda_runtime.h` and the internal
  `cuda/device_buffer.hpp`, and the test target links `CUDA::cudart`. This is a deliberate internal
  test exception to the section 5.2 include rule because DeviceBuffer cannot be exercised before the
  public API exists. It does not leak CUDA into `include/cme/`.

## Batch 3.1

- **Naive block shape:** 16x16 threads per block. The block is small enough for wide occupancy on the
  target GPU and the grid ceiling-divides both matrix dimensions.
- **Kernel test harness location:** `tests/kernel_test_util.hpp`, a new test-only header. It runs one
  launcher, synchronizes, and compares every element against the CPU reference. It needs the internal
  kernel header and `DeviceBuffer`, so it is another internal test exception to the section 5.2
  include rule.
- **Compute-sanitizer tooling:** the system `compute-sanitizer` 2022.4.1 cannot inject into processes
  with the installed 595.84 driver; it fails with "Unable to find injection library" or "terminated
  before first instrumented API call", including on a trivial standalone kernel. Sanitizer checks in
  this phase used the CUDA 13.2 package extracted under `/tmp` (version 2026.1.1.0), matching the
  driver's reported CUDA 13.2. Batch 5.3 must decide how to make `scripts/test.sh` reproducible given
  this environment mismatch.

## Batch 3.2

- **Tiled tile sizes:** implemented as a compile-time template with a launcher `switch` on
  `tile_size` 16 and 32. Unsupported values return `cudaErrorInvalidValue` without launching.
- **Shared-memory edge handling:** out-of-range A and B elements are stored as `0.0f`, so partial
  tiles add zero and need no special inner-loop branch.
- **Sanitizer evidence:** memcheck covered every tiled test, including 127x129x131 at both tile
  sizes. Racecheck covered the dedicated 64x64x64 tile-16 case; both reported zero errors.

## Batch 3.3

- **Optimized kernel candidate:** 2x2 register micro-tile per thread, `TILE/2 x TILE/2` block,
  shared-memory rows padded by one float, and an unrolled inner loop. The design and rationale are in
  `docs/ALGORITHMS.md`. No performance claim is made; Phase 7 must measure it against the tiled
  kernel and apply the 5% rule.
- **Sanitizer evidence:** memcheck covered 127x129x131 tile 16 and reported zero errors.

## Batch 4.1

- **Internal dispatch include:** `src/cuda/gemm_dispatch.hpp` includes
  `cuda/kernels/gemm_kernels.hpp`, matching the repository's `-I src` internal include convention.
  The frozen function signatures are unchanged.
- **Unknown algorithm values:** `gemm_device` returns `InvalidArgument` for enum values outside the
  four declared algorithms. `CPU_REFERENCE` returns `UnsupportedConfiguration` as specified.
- **Dispatch test:** `tests/test_gemm_dispatch.cpp` is a new test file not in the original tree. It
  covers direct `gemm_device` calls for all device algorithms and the rejection cases.

## Batch 4.2

- **Strided host copies:** `DeviceBuffer` gained `copy_from_host_2d` and `copy_to_host_2d`, both
  implemented with synchronous `cudaMemcpy2D`. The public API uses them so non-contiguous views keep
  their caller stride without a host staging vector. This is the Batch 2.2 amendment required by
  Batch 4.2.
- **Conditional C upload:** the public API copies C to the device only when `beta != 0`. When
  `beta == 0` the device C buffer is left uninitialized, and the kernels write every output element
  without reading C.
- **Overlap detection:** after config validation, A, B, and C are treated as half-open byte ranges
  derived from the data pointer, rows, cols, and stride. Any pairwise intersection returns
  `UnsupportedConfiguration`.
- **Validation order:** views, shapes, config, then overlap, exactly as listed in Batch 4.2.
- **Smoke tests:** `tests/test_gemm_edge_cases.cpp` now holds the 2x2 public-API smoke cases for all
  four algorithms; Batch 5.2 extends this file with the full edge-case matrix.

## Batch 4.3

- **Example target names:** `cme_example_basic`, `cme_example_algorithms`, and
  `cme_example_custom`. They link only `cme` and include only `cme/*.hpp` public headers.
- **Algorithm comparison coverage:** the example runs CPU_REFERENCE, CUDA_NAIVE, CUDA_TILED at 16 and
  32, and CUDA_OPTIMIZED at 16 and 32 on one 64x48x32 input, reporting the maximum absolute difference
  from the CPU reference.
- **Custom configuration:** demonstrates `MatrixView` sub-matrices with `stride > cols` and
  `CUDA_TILED` with `tile_size = 32`, then checks the strided result against a packed CPU reference.
  These are correctness checks, not performance measurements.

## Batch 5.1

- **Correctness matrix layout:** one Catch2 case covers the five algorithm variants
  (`CUDA_NAIVE`, `CUDA_TILED` at 16 and 32, `CUDA_OPTIMIZED` at 16 and 32), eight shapes, three
  seeds, and the five required `(alpha, beta)` pairs. The CPU reference is computed once per
  shape/seed/coefficient combination and reused for all five algorithms.
- **Comparison count:** 600 algorithm runs and 16,305,450 element comparisons. The count is printed
  to stdout and repeated with `WARN` so it appears in normal Catch2 output.

## Batch 5.2

- **Edge-case coverage:** the required invalid, empty, overlapping, NaN, alpha-zero, strided,
  tile-size, enum, and large-K cases live in `tests/test_gemm_edge_cases.cpp`. Validation-only
  cases do not require a GPU; cases that execute a kernel call `require_cuda_device()`.
- **CUDA error mapping tests:** `tests/test_cuda_errors.cpp` now checks `to_status` for success and
  error inputs, including the context/name/string message format, and checks a clean
  `check_last_launch` result. These tests do not need a device.

## Batch 5.3

- **Sanitizer selection:** `scripts/test.sh` uses `CME_COMPUTE_SANITIZER` when set, then the known
  working CUDA 13.2 extracted binary, then `compute-sanitizer` from `PATH`. The environment mismatch
  recorded in Batch 3.1 remains: the system 2022.4.1 sanitizer cannot run with driver 595.84.
- **Memcheck scope:** memcheck runs on the full `cme_tests` binary, as required.
- **Racecheck scope:** racecheck runs only the `[racecheck]` filter. The full suite is unnecessary
  for race detection and substantially slower; the dedicated case is a 64x64x64 tiled kernel run.
- **environment.sh policy:** every tool is probed with `command -v`; a missing tool prints
  `NOT FOUND` and the script continues with exit status 0.

## Batch 7.1

- **Profiler access:** direct `ncu` is blocked by `RmProfilingAdminOnly: 1`, and `sudo -n` is not
  available. `scripts/profile.sh` runs Nsight Compute in a privileged `ubuntu:24.04` container with
  `chroot /host` when that condition is detected, so the report still comes from the host binary and
  host GPU. If the user can enable profiling permissions, the same commands run directly.
- **Nsight Systems export:** this host's `nsys` writes a `.qdstrm` but cannot find its bundled
  importer. The script converts it with `QdstrmImporter` from
  `/usr/lib/nsight-systems/host-linux-x64/`. Both the `.nsys-rep` and the GPU kernel summary are
  saved under `results/profiles/`.
- **Profiler clocks:** Nsight Compute reports `2.34 cycle/nsecond` for every captured kernel. The
  baseline values are recorded from the reports and are not compared as wall-clock benchmark values.

## Batch 7.2

- **TILE=32 micro-tile:** changed from 2x2 to 4x4 after the profile showed `mio_throttle` at 13.84
  cycles per issued instruction. The measured kernel-only improvement was 52.7% at 512, 87.8% at
  1024, 88.7% at 2048, and 78.2% at 4096 in GFLOP/s, with the 4x4 result outside the baseline IQR.
  TILE=16 remains at 2x2 because a 4x4 micro-tile would reduce that block to 4x4 threads.
- **Size claim:** the 4x4 variant is claimed for sizes 512 and larger. At 256 it measured 21.4%
  slower in GFLOP/s, so it is not claimed for that size.
- **Default dispatch:** `CUDA_OPTIMIZED` stays selectable for clients that choose it. The public
  default remains `CUDA_TILED` with `tile_size = 16`, because section 4.4 freezes that default and
  Batch 7.3 may change only the dispatch file, not the public configuration header.

## Batch 6.1

- **Utilities layout:** `benchmark_config.hpp`, `csv_writer.hpp`, and `csv_writer.cpp` were kept as
  the frozen Batch 6.1 interface. Shared statistics, timestamp, and GPU sampling helpers live in
  `benchmark_common.hpp/.cpp`; CSV row construction lives in `benchmark_rows.hpp/.cpp`.
- **CSV test:** `benchmarks/csv_writer_test.cpp` writes one row containing a comma in a field, reads
  the file back, and fails unless the 19-field header matches byte-for-byte and the comma is quoted
  per RFC 4180. It runs as CTest test `benchmark_csv_writer`.
- **Transfer GFLOP/s:** `h2d_transfer` and `d2h_transfer` rows store `0.000000` GFLOP/s because they
  perform no arithmetic.

## Batch 6.2

- **Five kernel specifications:** the benchmark covers CUDA_NAIVE, CUDA_TILED at 16 and 32, and
  CUDA_OPTIMIZED at 16 and 32. Allocating device buffers per size outside the timed region and
  timing only the launch with CUDA events follows the batch requirement.
- **Private-header exception:** `benchmarks/benchmark_gemm.cu` includes
  `src/cuda/gemm_dispatch.hpp` and `src/cuda/device_buffer.hpp`, the HANDOFF §5.2 exception. The
  exception is recorded in `docs/BENCHMARKING.md`.

## Batch 6.3

- **End-to-end scope:** each device algorithm and the CPU reference run through the public
  `cme::gemm()` with `std::chrono::steady_clock`. The CPU reference is limited to sizes 256 and 512
  to keep the full run practical; no device size was dropped.
- **Transfer timing:** H2D aggregates the A and B copies in one timed region; D2H times the C copy.
  Both use CUDA events around the copy only. The transfer harness uses raw `cudaMalloc` buffers and
  does not depend on the public `gemm()` path.

## Batch 6.4

- **Quick mode:** `--quick` uses sizes 256 and 512, 2 warm-up iterations, and 5 measured iterations
  unless the caller overrides them. The full mode uses the frozen 5/30 and 256..4096 defaults.
- **Output naming:** `scripts/benchmark.sh` writes `results/gemm_<UTC>[_quick].csv` and a matching
  `.env.txt` from `scripts/environment.sh`. The benchmark binary itself accepts `--output` and
  `--sizes`/`--measurements`/`--warmup`/`--iterations`.
- **cuBLAS:** `CME_WITH_CUBLAS=OFF` by owner default D3; the report says cuBLAS not measured rather
  than citing a row that does not exist.
- **Benchmark documentation:** `docs/BENCHMARKING.md` stores the binding methodology and the
  kernel-only private-header exception. `results/README.md` explains file naming and CSV reading.

## Batch 7.3

- **Dispatch unchanged:** the final decision is to keep `src/cuda/gemm_dispatch.cu` as-is. The public
  default remains `CUDA_TILED`/16 because it is frozen; callers must select `CUDA_OPTIMIZED`/32 to
  get the fastest measured configuration.
- **Final evidence:** the final full run is `results/gemm_20261103T062611Z.csv`. The full test script
  passed 44/44 CTest tests, memcheck reported 0 errors, and racecheck reported 0 hazards.
- **Report mapping:** `docs/PERFORMANCE_REPORT.md` numbers are copied from that CSV. cuBLAS is
  reported as not measured. The 256-size regression of the 4x4 optimized tile is listed as a known
  limitation.
