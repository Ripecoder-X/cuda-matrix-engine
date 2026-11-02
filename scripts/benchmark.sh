#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

QUICK=0
for arg in "$@"; do
    if [[ "${arg}" == "--quick" ]]; then
        QUICK=1
    fi
done

PRESET="${CME_PRESET:-}"
if [[ -z "${PRESET}" ]]; then
    if [[ "${OSTYPE:-}" == "msys" || "${OSTYPE:-}" == "win32" || -n "${WINDIR:-}" ]]; then
        PRESET="windows-release"
    else
        PRESET="linux-release"
    fi
fi

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

TIMESTAMP="$(date -u '+%Y%m%d_%H%M%S')"
mkdir -p "${ROOT_DIR}/results"
CSV_FILE="gemm_${TIMESTAMP}.csv"
CSV_OUT="${ROOT_DIR}/results/${CSV_FILE}"
ENV_OUT="${ROOT_DIR}/results/${CSV_FILE}.env.txt"

printf 'Capturing environment to %s\n' "${ENV_OUT}"
if [[ -f "${SCRIPT_DIR}/environment.sh" ]]; then
    bash "${SCRIPT_DIR}/environment.sh" > "${ENV_OUT}" 2>&1 || true
else
    printf 'scripts/environment.sh not found\n' > "${ENV_OUT}"
fi

ARGS=("--output" "${CSV_OUT}")
if [[ "${QUICK}" -eq 1 ]]; then
    ARGS+=("--quick")
fi

printf 'Running benchmark: %s %s\n\n' "${BENCH_BINARY}" "${ARGS[*]}"
"${BENCH_BINARY}" "${ARGS[@]}"

printf '\nBenchmark run complete.\n'
printf 'CSV Output: %s\n' "${CSV_OUT}"
printf 'Environment: %s\n' "${ENV_OUT}"
