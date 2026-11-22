# Task Board

Status values: `NOT_STARTED` → `IN_PROGRESS` → `UNDER_REVIEW` → `VERIFIED`.
Side state: `BLOCKED` (reason in Notes). Only the VERIFY role moves a batch to `VERIFIED`.

| ID | Phase | Role | Status | Agent | Date | Depends on | Notes |
|---|---|---|---|---|---|---|---|
| 0.1 | 0 | ARCH | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | — | Environment recorded; VERIFY evidence in `docs/development/handoffs/VERIFY-closeout.md` |
| 0.2 | 0 | ARCH | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 0.1 | Build + skeleton; Apache-2.0; VERIFY evidence in `docs/development/handoffs/VERIFY-closeout.md` |
| 0.3 | 0 | ARCH | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 0.2 | Control docs created; VERIFY evidence in `docs/development/handoffs/VERIFY-closeout.md` |
| 1.1 | 1 | CORE | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 0.3 | Core matrix/status builds; owner Phase 0 waiver recorded |
| 1.2 | 1 | CORE | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 1.1 | MatrixView/Matrix built; no CUDA includes |
| 1.3 | 1 | CORE | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 1.1 | Config headers frozen; `docs/development/INTERFACES.md` matches HANDOFF §4 |
| 1.4 | 1 | CORE/TEST | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 1.2 | Catch2 v3.9.1; 48/48 CTest final |
| 2.1 | 2 | RUNTIME | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 1.1, 0.2 | cuda_utils implemented; build and status tests pass |
| 2.2 | 2 | RUNTIME | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 2.1 | DeviceBuffer + strided 2D copies verified |
| 3.1 | 3 | KERNEL | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 2.2 | Naive kernel; correctness + memcheck clean |
| 3.2 | 3 | KERNEL | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 3.1 | Tiled 16/32; memcheck + racecheck clean |
| 3.3 | 3 | KERNEL | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 3.2 | Optimized kernel; correctness + sanitizers clean |
| 4.1 | 4 | API | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 3.3 | Dispatch + direct dispatch test pass |
| 4.2 | 4 | API | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 4.1, 2.2, 1.4 | Public API + `cudaMemcpy2D` copies; edge cases pass |
| 4.3 | 4 | API | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 4.2 | All three examples built and exited 0 |
| 5.1 | 5 | TEST | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 4.2 | 600 runs, 16,305,450 comparisons in final CTest |
| 5.2 | 5 | TEST | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 5.1 | Edge cases + CUDA error mapping pass |
| 5.3 | 5 | TEST | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 5.2 | `./scripts/test.sh`: 48/48, memcheck 0, racecheck 0 |
| 6.1 | 6 | BENCH | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 4.2 | Config/CSV/common rows; CSV writer test passes |
| 6.2 | 6 | BENCH | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 6.1, 4.1 | Kernel-only rows in final CSV |
| 6.3 | 6 | BENCH | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 6.2 | End-to-end + H2D/D2H rows in final CSV |
| 6.4 | 6 | BENCH | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 6.3 | Final CSV `results/gemm_20261122T073654Z.csv` (62 rows) |
| 7.1 | 7 | OPT | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 6.4 | Nsight reports in `results/profiles/`; `docs/PROFILING.md` |
| 7.2 | 7 | OPT | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 7.1 | Tile-32 4x4 micro-tile kept; profile evidence linked |
| 7.3 | 7 | OPT/VERIFY | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 7.2 | Dispatch unchanged; final report refreshed from final CSV |
| 8.1 | 8 | INTEG | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 7.3 | `docs/development/handoffs/8.1.md`; README and final docs complete |
| 8.2 | 8 | INTEG | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 8.1 | `docs/development/handoffs/8.2.md`; pinned CI valid; GPU tests skip cleanly |
| 8.3 | 8 | INTEG/VERIFY | VERIFIED | opencode/owner-authorized VERIFY | 2026-11-22 | 8.2 | `docs/development/handoffs/8.3.md`; DoD evidence complete |

Total: 28 batches. Phase 0 gate requires 0.1, 0.2, and 0.3 to be `VERIFIED` by an agent other than
the one that wrote them (or by the owner). The owner explicitly authorized this VERIFY pass.
