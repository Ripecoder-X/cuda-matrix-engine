#!/usr/bin/env bash
# Batch 7.1: baseline Nsight Systems (end-to-end, size 2048) and Nsight Compute
# (each GEMM kernel, size 2048). Reports land in results/profiles/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PRESET="${CME_PRESET:-}"
if [[ -z "${PRESET}" ]]; then
    if [[ "${OSTYPE:-}" == "msys" || "${OSTYPE:-}" == "win32" || -n "${WINDIR:-}" ]]; then
        PRESET="windows-release"
    else
        PRESET="linux-release"
    fi
fi

find_tool() {
    local name="$1"
    shift
    if command -v "${name}" >/dev/null 2>&1; then
        command -v "${name}"
        return 0
    fi
    local candidate
    for candidate in "$@"; do
        if [[ -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}

NSYS="$(find_tool nsys \
    "/usr/local/cuda/bin/nsys" \
    "/opt/nvidia/nsight-systems/bin/nsys" \
    "/Program Files/NVIDIA Corporation/Nsight Systems 2026.3.1/target-windows-x64/nsys.exe" \
    "/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe" \
    || true)"
NCU="$(find_tool ncu \
    "/usr/local/cuda/bin/ncu" \
    "/usr/local/NVIDIA-Nsight-Compute/ncu" \
    "/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/ncu.bat" \
    || true)"

if [[ -z "${NSYS}" ]]; then
    printf 'nsys not found. Set PATH or CME_NSYS.\n' >&2
    exit 1
fi
if [[ -z "${NCU}" ]]; then
    printf 'ncu not found. Set PATH or CME_NCU.\n' >&2
    exit 1
fi
NSYS="${CME_NSYS:-${NSYS}}"
NCU="${CME_NCU:-${NCU}}"

printf 'Configuring and building preset: %s\n' "${PRESET}"
cmake --preset "${PRESET}"
cmake --build --preset "${PRESET}"

BENCH_BINARY="${ROOT_DIR}/build/${PRESET}/benchmarks/benchmark_main"
if [[ ! -f "${BENCH_BINARY}" && -f "${BENCH_BINARY}.exe" ]]; then
    BENCH_BINARY="${BENCH_BINARY}.exe"
fi
if [[ ! -f "${BENCH_BINARY}" ]]; then
    printf 'Benchmark binary not found at %s\n' "${BENCH_BINARY}" >&2
    exit 1
fi

PROFILE_DIR="${ROOT_DIR}/results/profiles"
mkdir -p "${PROFILE_DIR}"

printf '\n=== Nsight Systems: end-to-end size 2048 ===\n'
printf 'nsys: %s\n' "${NSYS}"
"${NSYS}" --version
NSYS_OUT="${PROFILE_DIR}/nsys_end_to_end_2048"
"${NSYS}" profile \
    --force-overwrite=true \
    --stats=true \
    --trace=cuda,nvtx,osrt \
    -o "${NSYS_OUT}" \
    "${BENCH_BINARY}" \
    --size 2048 --end-to-end --warmup 1 --iterations 3 \
    --output "${PROFILE_DIR}/temp_e2e.csv"

if [[ -f "${NSYS_OUT}.nsys-rep" ]]; then
    "${NSYS}" stats --report cuda_gpu_kern_sum --format csv \
        -o "${PROFILE_DIR}/nsys_end_to_end_2048_kern" \
        "${NSYS_OUT}.nsys-rep" || true
    "${NSYS}" stats --report cuda_gpu_mem_time_sum --format csv \
        -o "${PROFILE_DIR}/nsys_end_to_end_2048_mem" \
        "${NSYS_OUT}.nsys-rep" || true
fi

# Occupancy, SOL (compute + memory), warp stalls, shared-memory bank conflicts.
NCU_METRICS="sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
smsp__average_warps_issue_stalled_long_scoreboard.pct,\
smsp__average_warps_issue_stalled_short_scoreboard.pct,\
smsp__average_warps_issue_stalled_barrier.pct,\
smsp__average_warps_issue_stalled_membar.pct,\
smsp__average_warps_issue_stalled_mio_throttle.pct,\
smsp__average_warps_issue_stalled_lg_throttle.pct,\
smsp__average_warps_issue_stalled_misc.pct,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct,\
launch__registers_per_thread,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_warps,\
launch__grid_size,\
launch__block_size,\
launch__shared_mem_per_block_allocated"

profile_kernel() {
    local label="$1"
    local regex="$2"
    local out="${PROFILE_DIR}/ncu_${label}_2048"
    printf '\n=== Nsight Compute: %s (size 2048) ===\n' "${label}"
    printf 'ncu: %s\n' "${NCU}"
    "${NCU}" \
        --force-overwrite \
        --target-processes all \
        --kernel-name-base demangled \
        --kernel-name "regex:${regex}" \
        --launch-count 1 \
        --metrics "${NCU_METRICS}" \
        --csv \
        --log-file "${out}.log" \
        -o "${out}" \
        "${BENCH_BINARY}" \
        --size 2048 --kernel-only --warmup 0 --iterations 1 \
        --output "${PROFILE_DIR}/temp_kernel_${label}.csv"
    if [[ -f "${out}.ncu-rep" ]]; then
        "${NCU}" --import "${out}.ncu-rep" --csv --page raw > "${out}.csv" || true
    fi
}

printf '\nncu: %s\n' "${NCU}"
"${NCU}" --version

profile_kernel "naive" "gemm_naive_kernel"
profile_kernel "tiled16" "gemm_tiled_kernel<16>"
profile_kernel "tiled32" "gemm_tiled_kernel<32>"
profile_kernel "optimized16" "gemm_optimized_kernel<16>"
profile_kernel "optimized32" "gemm_optimized_kernel<32>"

printf '\nProfiling complete. Reports in %s\n' "${PROFILE_DIR}"
printf 'Nsight Systems: %s.nsys-rep\n' "${NSYS_OUT}"
printf 'Nsight Compute: ncu_{naive,tiled16,tiled32,optimized16,optimized32}_2048.ncu-rep\n'
