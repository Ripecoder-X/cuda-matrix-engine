#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PRESET="${CME_BENCHMARK_PRESET:-linux-release}"

QUICK=0
for arg in "$@"; do
    if [[ "${arg}" == "--quick" ]]; then
        QUICK=1
    fi
done

cmake --preset "${PRESET}"
cmake --build --preset "${PRESET}" -j "$(nproc)"

mkdir -p results
STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
SUFFIX=""
if [[ "${PRESET}" != "linux-release" ]]; then
    SUFFIX="_${PRESET}"
fi
if [[ "${QUICK}" -eq 1 ]]; then
    OUTPUT="results/gemm_${STAMP}${SUFFIX}_quick.csv"
else
    OUTPUT="results/gemm_${STAMP}${SUFFIX}.csv"
fi

"./build/${PRESET}/benchmarks/cme_benchmark" --output "${OUTPUT}" "$@"
./scripts/environment.sh > "${OUTPUT%.csv}.env.txt"

printf 'benchmark CSV: %s\n' "${OUTPUT}"
printf 'environment:  %s\n' "${OUTPUT%.csv}.env.txt"
