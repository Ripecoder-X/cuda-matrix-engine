#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

cmake --preset linux-release
cmake --build --preset linux-release --parallel "$(nproc)"
