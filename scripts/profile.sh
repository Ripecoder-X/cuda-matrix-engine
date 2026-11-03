#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

SIZE="${CME_PROFILE_SIZE:-2048}"
BENCH="${ROOT}/build/linux-release/benchmarks/cme_benchmark"
PROFILE_DIR="${ROOT}/results/profiles"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cme-profile.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

cmake --preset linux-release
cmake --build --preset linux-release --parallel "$(nproc)"
mkdir -p "${PROFILE_DIR}"

if [[ ! -x "${BENCH}" ]]; then
    printf 'Benchmark binary not found: %s\n' "${BENCH}" >&2
    exit 1
fi

printf '\n== Nsight Systems: end-to-end, size %s ==\n' "${SIZE}"
NSYS_BASE="${TMP_DIR}/nsys_end_to_end_${SIZE}"
nsys profile \
    --force-overwrite true \
    --trace=cuda,osrt \
    --sample=none \
    --cpuctxsw=none \
    --output "${NSYS_BASE}" \
    "${BENCH}" \
    --sizes "${SIZE}" \
    --measurements end_to_end \
    --warmup 1 \
    --iterations 1 \
    --output "${TMP_DIR}/nsys_run.csv"

NSYS_REPORT="${PROFILE_DIR}/nsys_end_to_end_${SIZE}.nsys-rep"
if [[ -f "${NSYS_BASE}.nsys-rep" ]]; then
    cp "${NSYS_BASE}.nsys-rep" "${NSYS_REPORT}"
elif [[ -f "${NSYS_BASE}.qdstrm" ]]; then
    IMPORTER="$(command -v QdstrmImporter || true)"
    if [[ -z "${IMPORTER}" ]]; then
        for candidate in \
            /usr/lib/nsight-systems/host-linux-x64/QdstrmImporter \
            /opt/nvidia/nsight-systems/*/host-linux-x64/QdstrmImporter; do
            if [[ -x "${candidate}" ]]; then
                IMPORTER="${candidate}"
                break
            fi
        done
    fi
    if [[ -z "${IMPORTER}" ]]; then
        printf 'QdstrmImporter not found; nsys produced only %s\n' "${NSYS_BASE}.qdstrm" >&2
        exit 1
    fi
    "${IMPORTER}" -i "${NSYS_BASE}.qdstrm" -o "${NSYS_REPORT}" -f
else
    printf 'Nsight Systems did not produce a report\n' >&2
    exit 1
fi

nsys stats -r gpukernsum -o "${PROFILE_DIR}/nsys_end_to_end_${SIZE}_gpukernsum" \
    "${NSYS_REPORT}" >/dev/null 2>&1 || true

printf 'Nsight Systems report: %s\n' "${NSYS_REPORT}"

NCU_SECTIONS=(
    --section SpeedOfLight
    --section Occupancy
    --section WarpStateStats
    --section MemoryWorkloadAnalysis
    --section MemoryWorkloadAnalysis_Tables
)

USE_DOCKER=0
if [[ "${EUID}" -ne 0 ]] && command -v docker >/dev/null 2>&1 &&
    grep -q '^RmProfilingAdminOnly: 1$' /proc/driver/nvidia/params 2>/dev/null; then
    USE_DOCKER=1
fi

run_ncu() {
    local name="$1"
    local kernel_regex="$2"
    local launch_skip="$3"
    local report="${PROFILE_DIR}/ncu_kernel_${SIZE}_${name}"
    local csv="${TMP_DIR}/ncu_${name}.csv"

    printf '\n== Nsight Compute: %s ==\n' "${name}"
    if [[ "${USE_DOCKER}" -eq 1 ]]; then
        docker run --rm --privileged --pid=host --ipc=host --network=host \
            -v /:/host -v /proc:/host/proc -v /sys:/host/sys -v /dev:/host/dev \
            ubuntu:24.04 chroot /host /bin/bash -lc \
            "cd '${ROOT}' && ncu --target-processes all --kernel-name-base demangled \
                --kernel-name 'regex:.*${kernel_regex}.*' \
                --launch-skip '${launch_skip}' --launch-count 1 \
                ${NCU_SECTIONS[*]} \
                --force-overwrite -o '${report}' \
                '${BENCH}' --sizes '${SIZE}' --measurements kernel_only \
                --warmup 0 --iterations 1 --output '${csv}'"
    else
        ncu --target-processes all --kernel-name-base demangled \
            --kernel-name "regex:.*${kernel_regex}.*" \
            --launch-skip "${launch_skip}" --launch-count 1 \
            "${NCU_SECTIONS[@]}" \
            --force-overwrite -o "${report}" \
            "${BENCH}" --sizes "${SIZE}" --measurements kernel_only \
            --warmup 0 --iterations 1 --output "${csv}"
    fi

    if [[ ! -f "${report}.ncu-rep" ]]; then
        printf 'Nsight Compute did not produce %s.ncu-rep\n' "${report}" >&2
        exit 1
    fi
    printf 'Nsight Compute report: %s.ncu-rep\n' "${report}"
}

run_ncu naive gemm_naive_kernel 0
run_ncu tiled16 gemm_tiled_kernel 0
run_ncu tiled32 gemm_tiled_kernel 1
run_ncu optimized16 gemm_optimized_kernel 0
run_ncu optimized32 gemm_optimized_kernel 1

printf '\nProfiles written under %s\n' "${PROFILE_DIR}"
