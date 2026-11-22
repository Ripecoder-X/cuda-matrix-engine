## 5. Architecture and dependency rules

### 5.1 Dependency graph (binding)

```
examples  ─┐
benchmarks ┼──► public API (include/cme/gemm.hpp, matrix.hpp, ...)
tests      ─┘          │
                       ▼
             src/api/gemm.cpp  (validation + host orchestration)
                 │             │
                 ▼             ▼
   src/core/gemm_reference   src/cuda/gemm_dispatch  ──► kernels/*.cu
                                   │
                                   ▼
                       src/cuda/device_buffer, cuda_utils
```

### 5.2 Rules

- **Examples, benchmarks, and tests may include only `include/cme/*` headers** — with one exception: the kernel-only benchmark may include `src/cuda/gemm_dispatch.hpp` and `src/cuda/device_buffer.hpp` to measure device-only time. Record this exception in `docs/BENCHMARKING.md`.
- **Kernels never call the public API.** They are called only through `gemm_dispatch`.
- **Only `src/cuda/` and `benchmarks/*.cu` may include CUDA headers.** `include/cme/` must never.
- **`src/api/gemm.cpp` is the only place that performs host-to-device orchestration** for the public API.

### 5.3 Public `gemm()` execution flow

For `algorithm == CPU_REFERENCE`:
1. Validate arguments (shapes, validity, overlap, config).
2. Call `cme::detail::gemm_cpu_reference`.
3. Return Ok.

For any CUDA algorithm:
1. Validate arguments.
2. Allocate device buffers for A, B, and C (`DeviceBuffer::create`).
3. Copy A and B host-to-device. Copy C host-to-device only if `beta != 0`.
4. Build `DeviceGemmArgs` with leading dimensions equal to the views' strides.
5. Call `gemm_device(algorithm, args, stream = 0)`.
6. Synchronize the device and check for errors.
7. Copy C device-to-host.
8. Return Ok.

Note: the host matrix layout uses `stride`, but device buffers are allocated tightly packed. Copy with care: copy each row individually (`cudaMemcpy2D`) when `stride != cols`. Record the chosen approach in `docs/DESIGN_DECISIONS.md`. Prefer `cudaMemcpy2D` in `DeviceBuffer` helpers; document the choice.

---
