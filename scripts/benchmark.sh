#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

QUICK=0
for arg in "$@"; do
    if [[ "${arg}" == "--quick" ]]; then
        QUICK=1
    fi
done

cmake --preset linux-release
cmake --build --preset linux-release -j "$(nproc)"

mkdir -p results
STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
if [[ "${QUICK}" -eq 1 ]]; then
    OUTPUT="results/gemm_${STAMP}_quick.csv"
else
    OUTPUT="results/gemm_${STAMP}.csv"
fi

./build/linux-release/benchmarks/cme_benchmark --output "${OUTPUT}" "$@"
./scripts/environment.sh > "${OUTPUT%.csv}.env.txt"

printf 'benchmark CSV: %s\n' "${OUTPUT}"
printf 'environment:  %s\n' "${OUTPUT%.csv}.env.txt"
