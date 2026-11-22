#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${repo_root}/build/linux-release}"
prefix="${2:-${repo_root}/build/package-prefix}"
consumer_dir="${repo_root}/build/package-consumer"

if [[ "${build_dir}" != /* ]]; then
    build_dir="${repo_root}/${build_dir}"
fi

if [[ "${prefix}" != /* ]]; then
    prefix="${repo_root}/${prefix}"
fi

cmake --install "${build_dir}" --prefix "${prefix}"
rm -rf "${consumer_dir}"
cmake -S "${repo_root}/tests/package_consumer" -B "${consumer_dir}" -DCMAKE_PREFIX_PATH="${prefix}"
cmake --build "${consumer_dir}" --parallel
"${consumer_dir}/cme_package_consumer"
