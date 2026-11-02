# Quality Standards

## Tolerance rule (binding for all correctness tests)

```
|got - ref| <= atol_for(K) + rtol * |ref|
rtol = 1e-4,  atol_for(K) = 1e-5 * K
```

Implemented by `cme::test::close_enough(float got, float ref, int K)` in `tests/test_helpers.hpp`.
Do not tighten or loosen this rule without a documented analysis in `docs/ALGORITHMS.md`, applied to
all tests at once.

## Benchmark and measurement rules (binding)

1. **Warm-up before measuring.** Five unmeasured iterations per configuration.
2. **Repeat.** Thirty measured iterations per configuration. No single-run numbers in any document.
3. **Report distribution.** Median, min, max, IQR. Never report only the best run or only the mean.
4. **Separate what you time.** Kernel-only, end-to-end, and transfer are three different numbers.
   Never mix them in one column.
5. **Synchronize correctly.** Before reading CUDA event elapsed time, synchronize the event. Before
   reading host wall clock after GPU work, synchronize the device.
6. **Record the hardware.** Every CSV row carries GPU name, driver version, and CUDA runtime
   version. Every report links to `docs/ENVIRONMENT.md`.
7. **Record the conditions.** Clocks, temperature, and power, as specified in the handoff section 6
   intro.
8. **Fixed dimensions.** Benchmarks use the sizes in `BenchmarkConfig`. Do not change sizes to make
   a result look better.
9. **No cherry-picking.** A benchmark result is reported whether or not it is favorable.
10. **No unmeasured comparisons.** Do not say "faster than cuBLAS", "competitive with cuBLAS", or
    any percentage against cuBLAS unless the cuBLAS row is in a CSV in `results/` and the report
    references it.
11. **Tensor Cores are out of scope** for this milestone. Do not mention them as a performance path
    in claims.

## Definition of done (project level)

The project is complete only when every row has evidence.

| Area | Acceptance criteria | Evidence required |
|---|---|---|
| Build | Clean CMake configure and build on Linux release preset | Command output summary |
| Correctness | All supported algorithms agree with CPU reference within the tolerance rule | `ctest` summary; test count |
| Memory | No CUDA memory errors | `compute-sanitizer --tool memcheck` clean (in `scripts/test.sh` output) |
| Races | No shared-memory races in kernels | `compute-sanitizer --tool racecheck` clean on kernel tests |
| Edge cases | Odd dimensions, non-square, K=1, M=1, strided views, beta=0 with NaN, alpha=0 tested | `test_gemm_edge_cases` passes |
| API | Stable documented public headers with no CUDA includes | grep result; `INTERFACES.md` matches section 4 |
| Performance | Reproducible benchmark CSV from a full run | File in `results/` and name in `PERFORMANCE_REPORT.md` |
| Profiling | Nsight evidence for every optimization claim | Profile files in `results/profiles/`; `PROFILING.md` entries |
| Documentation | Build, usage, algorithms, benchmarking, and performance documented | Files exist and contain no `TBD` placeholders |
| Integration | All targets build together including examples and benchmarks | Build command with all options ON |
| Publication | Clean repository; examples compile and run; CI workflow valid | Example outputs; workflow file |

## Additional binding decisions

- Catch2 v3 pinned to tag `v3.9.1` (recorded in `docs/DESIGN_DECISIONS.md`, Batch 1.4).
- CTest for test execution.
- Explicit CUDA architecture configuration (default 89, recorded).
- Benchmark results are machine-readable CSV.
- No hardware-specific performance claim without the hardware recorded in the same document.
- No Tensor Core implementation in this milestone.
- GPU-dependent tests must `SKIP` cleanly when no CUDA device is present, with a clear message.
