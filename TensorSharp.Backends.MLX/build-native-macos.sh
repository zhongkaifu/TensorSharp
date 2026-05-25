#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${script_dir}/Native"
build_dir="${source_dir}/build"
install_dir="${source_dir}/dist"
configuration="${CONFIGURATION:-Release}"
jobs="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
arch="${CMAKE_OSX_ARCHITECTURES:-arm64}"

cmake -S "${source_dir}" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE="${configuration}" \
  -DCMAKE_INSTALL_PREFIX="${install_dir}" \
  -DCMAKE_OSX_ARCHITECTURES="${arch}"

cmake --build "${build_dir}" --config "${configuration}" --target install --parallel "${jobs}"
