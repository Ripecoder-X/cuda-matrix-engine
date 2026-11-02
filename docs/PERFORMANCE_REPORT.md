# Performance Report

TBD by Batch 7.3 (final benchmark tables, profiler summary, cuBLAS comparison only if measured).

## Correctness evidence

Recorded by Batch 5.3 on 2026-11-02. This section is correctness evidence, not performance.

- Command: `./scripts/test.sh`
- Build: `cmake --preset linux-debug` and `cmake --build --preset linux-debug` completed.
- CTest: `100% tests passed, 0 tests failed out of 43`; 5368 assertions.
- Correctness matrix: 600 algorithm runs, 16,305,450 element comparisons.
- `compute-sanitizer` version: 2026.1.1.0 (CUDA 13.2 package).
- Memcheck on the full test binary: `ERROR SUMMARY: 0 errors`.
- Racecheck on the `[racecheck]` filter: `0 hazards displayed (0 errors, 0 warnings)`.
- Raw run log: `/tmp/cme_test_sh.log` (session artifact; not committed).

Environment: `docs/ENVIRONMENT.md`. The system `compute-sanitizer` 2022.4.1 is incompatible with the
installed 595.84 driver; `scripts/test.sh` selects the working 13.2 binary. See Batch 5.3 in
`docs/DESIGN_DECISIONS.md`.
