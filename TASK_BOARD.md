# Task Board

Status values: `NOT_STARTED` → `IN_PROGRESS` → `UNDER_REVIEW` → `VERIFIED`.
Side state: `BLOCKED` (reason in Notes). Only the VERIFY role moves a batch to `VERIFIED`.

| ID | Phase | Role | Status | Agent | Date | Depends on | Notes |
|---|---|---|---|---|---|---|---|
| 0.1 | 0 | ARCH | UNDER_REVIEW | opencode | 2026-11-02 | — | Environment recorded; awaiting independent verification |
| 0.2 | 0 | ARCH | UNDER_REVIEW | opencode | 2026-11-02 | 0.1 | Build + skeleton; Apache-2.0 chosen; awaiting verification |
| 0.3 | 0 | ARCH | UNDER_REVIEW | opencode | 2026-11-02 | 0.2 | Control docs created; awaiting verification |
| 1.1 | 1 | CORE | UNDER_REVIEW | opencode | 2026-11-02 | 0.3 | Build passes; Phase 0 gate overridden by owner |
| 1.2 | 1 | CORE | UNDER_REVIEW | opencode | 2026-11-02 | 1.1 | Build passes; no CUDA includes |
| 1.3 | 1 | CORE | UNDER_REVIEW | opencode | 2026-11-02 | 1.1 | Build passes |
| 1.4 | 1 | CORE/TEST | UNDER_REVIEW | opencode | 2026-11-02 | 1.2 | Catch2 v3.9.1; 13/13 tests pass |
| 2.1 | 2 | RUNTIME | UNDER_REVIEW | opencode | 2026-11-02 | 1.1, 0.2 | cuda_utils implemented; build passes |
| 2.2 | 2 | RUNTIME | UNDER_REVIEW | opencode | 2026-11-02 | 2.1 | DeviceBuffer + tests; 17/17 ctest, GPU tests ran |
| 3.1 | 3 | KERNEL | UNDER_REVIEW | opencode | 2026-11-02 | 2.2 | Naive kernel; 27/27 ctest; memcheck 0 errors |
| 3.2 | 3 | KERNEL | UNDER_REVIEW | opencode | 2026-11-02 | 3.1 | Tiled 16/32; memcheck + racecheck 0 errors |
| 3.3 | 3 | KERNEL | UNDER_REVIEW | opencode | 2026-11-02 | 3.2 | Optimized candidate; correctness + memcheck clean |
| 4.1 | 4 | API | UNDER_REVIEW | opencode | 2026-11-02 | 3.3 | Dispatch + direct dispatch test; 29/29 ctest |
| 4.2 | 4 | API | UNDER_REVIEW | opencode | 2026-11-02 | 4.1, 2.2, 1.4 | Public API + cudaMemcpy2D copies; 32/32 ctest |
| 4.3 | 4 | API | UNDER_REVIEW | opencode | 2026-11-02 | 4.2 | Examples built and all exit 0 |
| 5.1 | 5 | TEST | UNDER_REVIEW | opencode | 2026-11-02 | 4.2 | Correctness matrix: 600 runs, 16,305,450 comparisons |
| 5.2 | 5 | TEST | UNDER_REVIEW | opencode | 2026-11-02 | 5.1 | 8 edge cases, 715 assertions; CUDA error mapping |
| 5.3 | 5 | TEST | UNDER_REVIEW | opencode | 2026-11-02 | 5.2 | Scripts pass: 43/43 ctest; memcheck/racecheck 0 errors |
| 6.1 | 6 | BENCH | UNDER_REVIEW | antigravity | 2026-11-02 | 4.2 | BenchmarkConfig + CsvWriter; header verified byte-for-byte |
| 6.2 | 6 | BENCH | UNDER_REVIEW | antigravity | 2026-11-02 | 6.1, 4.1 | Kernel-only benchmark implemented; 10/10 configs verified |
| 6.3 | 6 | BENCH | UNDER_REVIEW | antigravity | 2026-11-02 | 6.2 | End-to-end and transfer benchmarks; all 4 measurement types verified |
| 6.4 | 6 | BENCH | UNDER_REVIEW | antigravity | 2026-11-02 | 6.3 | benchmark_main + scripts; full benchmark captured in results/ |
| 7.1 | 7 | OPT | UNDER_REVIEW | opencode | 2026-11-03 | 6.4 | Performance analysis complete, profiling plan created |
| 7.2 | 7 | OPT | UNDER_REVIEW | opencode | 2026-11-03 | 7.1 | Profiling commands documented, ready for execution |
| 7.3 | 7 | OPT/VERIFY | NOT_STARTED | — | — | 7.2 | — |
| 8.1 | 8 | INTEG | NOT_STARTED | — | — | 7.3 | — |
| 8.2 | 8 | INTEG | NOT_STARTED | — | — | 8.1 | — |
| 8.3 | 8 | INTEG/VERIFY | NOT_STARTED | — | — | 8.2 | — |

Total: 28 batches. Phase 0 gate requires 0.1, 0.2, and 0.3 to be `VERIFIED` by an agent other than
the one that wrote them (or by the owner).
