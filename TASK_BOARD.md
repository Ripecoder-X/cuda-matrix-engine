# Task Board

Status values: `NOT_STARTED` → `IN_PROGRESS` → `UNDER_REVIEW` → `VERIFIED`.
Side state: `BLOCKED` (reason in Notes). Only the VERIFY role moves a batch to `VERIFIED`.

| ID | Phase | Role | Status | Agent | Date | Depends on | Notes |
|---|---|---|---|---|---|---|---|
| 0.1 | 0 | ARCH | UNDER_REVIEW | opencode | 2026-11-02 | — | Environment recorded; awaiting independent verification |
| 0.2 | 0 | ARCH | UNDER_REVIEW | opencode | 2026-11-02 | 0.1 | Build + skeleton; Apache-2.0 chosen; awaiting verification |
| 0.3 | 0 | ARCH | UNDER_REVIEW | opencode | 2026-11-02 | 0.2 | Control docs created; awaiting verification |
| 1.1 | 1 | CORE | VERIFIED | owner | 2026-11-06 | 0.3 | Build passes; Phase 0 gate overridden by owner Verified per owner instruction 2026-11-06. |
| 1.2 | 1 | CORE | VERIFIED | owner | 2026-11-06 | 1.1 | Build passes; no CUDA includes Verified per owner instruction 2026-11-06. |
| 1.3 | 1 | CORE | VERIFIED | owner | 2026-11-06 | 1.1 | Build passes Verified per owner instruction 2026-11-06. |
| 1.4 | 1 | CORE/TEST | VERIFIED | owner | 2026-11-06 | 1.2 | Catch2 v3.9.1; 13/13 tests pass Verified per owner instruction 2026-11-06. |
| 2.1 | 2 | RUNTIME | VERIFIED | owner | 2026-11-06 | 1.1, 0.2 | cuda_utils implemented; build passes Verified per owner instruction 2026-11-06. |
| 2.2 | 2 | RUNTIME | VERIFIED | owner | 2026-11-06 | 2.1 | DeviceBuffer + tests; 17/17 ctest, GPU tests ran Verified per owner instruction 2026-11-06. |
| 3.1 | 3 | KERNEL | VERIFIED | owner | 2026-11-06 | 2.2 | Naive kernel; 27/27 ctest; memcheck 0 errors Verified per owner instruction 2026-11-06. |
| 3.2 | 3 | KERNEL | VERIFIED | owner | 2026-11-06 | 3.1 | Tiled 16/32; memcheck + racecheck 0 errors Verified per owner instruction 2026-11-06. |
| 3.3 | 3 | KERNEL | VERIFIED | owner | 2026-11-06 | 3.2 | Optimized candidate; correctness + memcheck clean Verified per owner instruction 2026-11-06. |
| 4.1 | 4 | API | VERIFIED | owner | 2026-11-06 | 3.3 | Dispatch + direct dispatch test; 29/29 ctest Verified per owner instruction 2026-11-06. |
| 4.2 | 4 | API | VERIFIED | owner | 2026-11-06 | 4.1, 2.2, 1.4 | Public API + cudaMemcpy2D copies; 32/32 ctest Verified per owner instruction 2026-11-06. |
| 4.3 | 4 | API | VERIFIED | owner | 2026-11-06 | 4.2 | Examples built and all exit 0 Verified per owner instruction 2026-11-06. |
| 5.1 | 5 | TEST | VERIFIED | owner | 2026-11-06 | 4.2 | Correctness matrix: 600 runs, 16,305,450 comparisons Verified per owner instruction 2026-11-06. |
| 5.2 | 5 | TEST | VERIFIED | owner | 2026-11-06 | 5.1 | 8 edge cases, 715 assertions; CUDA error mapping Verified per owner instruction 2026-11-06. |
| 5.3 | 5 | TEST | VERIFIED | owner | 2026-11-06 | 5.2 | Scripts pass: 43/43 ctest; memcheck/racecheck 0 errors Verified per owner instruction 2026-11-06. |
| 6.1 | 6 | BENCH | VERIFIED | owner | 2026-11-06 | 4.2 | Benchmark config/CSV/common rows; `benchmark_csv_writer` passes; handoff 6.1 Verified per owner instruction 2026-11-06. |
| 6.2 | 6 | BENCH | VERIFIED | owner | 2026-11-06 | 6.1, 4.1 | Kernel-only events benchmark; 25 rows; handoff 6.2 Verified per owner instruction 2026-11-06. |
| 6.3 | 6 | BENCH | VERIFIED | owner | 2026-11-06 | 6.2 | End-to-end + H2D/D2H; all four measurement types; handoff 6.3 Verified per owner instruction 2026-11-06. |
| 6.4 | 6 | BENCH | VERIFIED | owner | 2026-11-06 | 6.3 | Driver/script/docs; full CSV `results/gemm_20261103T062611Z.csv` (62 rows); handoff 6.4 Verified per owner instruction 2026-11-06. |
| 7.1 | 7 | OPT | UNDER_REVIEW | opencode | 2026-11-03 | 6.4 | Nsight Systems/Compute baseline reports in `results/profiles/`; handoff 7.1 |
| 7.2 | 7 | OPT | UNDER_REVIEW | opencode | 2026-11-03 | 7.1 | Tile-32 4x4 micro-tile kept; +52.7% to +88.7% kernel-only GFLOP/s at 512-4096; handoff 7.2 |
| 7.3 | 7 | OPT/VERIFY | UNDER_REVIEW | opencode | 2026-11-03 | 7.2 | Dispatch unchanged; final CSV + report; 44/44 tests, sanitizers clean; handoff 7.3 Owner noted: source has 47 TEST_CASEs vs 44 documented; re-run needed. |
| 8.1 | 8 | INTEG | VERIFIED | owner | 2026-11-06 | 7.3 | Verified per owner instruction 2026-11-06. |
| 8.2 | 8 | INTEG | VERIFIED | owner | 2026-11-06 | 8.1 | Verified per owner instruction 2026-11-06. |
| 8.3 | 8 | INTEG/VERIFY | VERIFIED | owner | 2026-11-06 | 8.2 | Verified per owner instruction 2026-11-06. |

Total: 28 batches. Phase 0 gate requires 0.1, 0.2, and 0.3 to be `VERIFIED` by an agent other than
the one that wrote them (or by the owner).
