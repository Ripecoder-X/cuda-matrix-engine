# Master Plan

Execution plan for `cuda-matrix-engine`. A batch is the unit of work for one agent session.
Batch IDs are permanent. Dependency and acceptance details live in `HANDOFF.md` section 6.

## Phase 0 — Environment and control plane

| Batch | Title | Role | Depends on |
|---|---|---|---|
| 0.1 | Record the environment | ARCH | — |
| 0.2 | Build system and repo skeleton | ARCH | 0.1 |
| 0.3 | Control documents | ARCH | 0.2 |

## Phase 1 — Core foundation (host only, no CUDA)

| Batch | Title | Role | Depends on |
|---|---|---|---|
| 1.1 | Status | CORE | 0.3 |
| 1.2 | Matrix view and matrix | CORE | 1.1 |
| 1.3 | Configuration types | CORE | 1.1 |
| 1.4 | CPU reference and test harness | CORE/TEST | 1.2 |

## Phase 2 — CUDA runtime

| Batch | Title | Role | Depends on |
|---|---|---|---|
| 2.1 | CUDA utilities | RUNTIME | 1.1, 0.2 |
| 2.2 | Device buffer | RUNTIME | 2.1 |

## Phase 3 — GEMM kernels

| Batch | Title | Role | Depends on |
|---|---|---|---|
| 3.1 | Naive kernel | KERNEL | 2.2 |
| 3.2 | Tiled kernel | KERNEL | 3.1 |
| 3.3 | Optimized kernel, first candidate | KERNEL | 3.2 |

## Phase 4 — Public API and dispatch

| Batch | Title | Role | Depends on |
|---|---|---|---|
| 4.1 | Device dispatch | API | 3.3 |
| 4.2 | Public host API | API | 4.1, 2.2, 1.4 |
| 4.3 | Examples | API | 4.2 |

## Phase 5 — Testing

| Batch | Title | Role | Depends on |
|---|---|---|---|
| 5.1 | Correctness matrix | TEST | 4.2 |
| 5.2 | Edge cases and invalid input | TEST | 5.1 |
| 5.3 | Sanitizer and CI-style test script | TEST | 5.2 |

## Phase 6 — Benchmarking

| Batch | Title | Role | Depends on |
|---|---|---|---|
| 6.1 | Benchmark utilities | BENCH | 4.2 |
| 6.2 | Kernel-only benchmark | BENCH | 6.1, 4.1 |
| 6.3 | End-to-end and transfer benchmarks | BENCH | 6.2 |
| 6.4 | Benchmark driver and reproducibility | BENCH | 6.3 |

## Phase 7 — Optimization and profiling

| Batch | Title | Role | Depends on |
|---|---|---|---|
| 7.1 | Baseline profiling | OPT | 6.4 |
| 7.2 | Optimization iterations | OPT | 7.1 |
| 7.3 | Dispatch decision and final benchmark | OPT/VERIFY | 7.2 |

## Phase 8 — Publication

| Batch | Title | Role | Depends on |
|---|---|---|---|
| 8.1 | Documentation | INTEG | 7.3 |
| 8.2 | Continuous integration | INTEG | 8.1 |
| 8.3 | Final review and definition of done | INTEG/VERIFY | 8.2 |

Total: 28 batches across 9 phases (0–8).
