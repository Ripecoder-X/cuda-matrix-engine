#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

cmake --preset linux-debug
cmake --build --preset linux-debug --parallel "$(nproc)"
ctest --preset linux-debug --output-on-failure

TEST_BINARY="${ROOT_DIR}/build/linux-debug/tests/cme_tests"
if [[ ! -x "${TEST_BINARY}" ]]; then
    printf 'Test binary not found: %s\n' "${TEST_BINARY}" >&2
    exit 1
fi

SANITIZER="${CME_COMPUTE_SANITIZER:-}"
if [[ -z "${SANITIZER}" && -x /tmp/cuda-sanitizer-13-2/usr/local/cuda-13.2/bin/compute-sanitizer ]]; then
    SANITIZER=/tmp/cuda-sanitizer-13-2/usr/local/cuda-13.2/bin/compute-sanitizer
fi
if [[ -z "${SANITIZER}" ]] && command -v compute-sanitizer >/dev/null 2>&1; then
    SANITIZER="$(command -v compute-sanitizer)"
fi
if [[ -z "${SANITIZER}" || ! -x "${SANITIZER}" ]]; then
    printf 'compute-sanitizer not found. Set CME_COMPUTE_SANITIZER to its path.\n' >&2
    exit 1
fi

printf '\nUsing compute-sanitizer: %s\n' "${SANITIZER}"
"${SANITIZER}" --version

printf '\nRunning memcheck on the full test binary\n'
"${SANITIZER}" --tool memcheck --error-exitcode 1 "${TEST_BINARY}"

printf '\nRunning racecheck on the [racecheck] kernel test filter\n'
"${SANITIZER}" --tool racecheck --error-exitcode 1 "${TEST_BINARY}" "[racecheck]"

printf '\nAll tests and sanitizer checks passed.\n'
