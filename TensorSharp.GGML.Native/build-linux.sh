#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
ENABLE_CUDA="${TENSORSHARP_GGML_NATIVE_ENABLE_CUDA:-OFF}"
BUILD_TESTS="${TENSORSHARP_GGML_NATIVE_BUILD_TESTS:-OFF}"
EXTRA_CMAKE_ARGS=()

while (($# > 0)); do
    case "$1" in
        --cuda)
            ENABLE_CUDA=ON
            ;;
        --no-cuda)
            ENABLE_CUDA=OFF
            ;;
        --tests)
            BUILD_TESTS=ON
            ;;
        *)
            EXTRA_CMAKE_ARGS+=("$1")
            ;;
    esac
    shift
done

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA="${ENABLE_CUDA}" \
    -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS="${BUILD_TESTS}" \
    "${EXTRA_CMAKE_ARGS[@]}"

if [[ "${BUILD_TESTS}" == "ON" ]]; then
    cmake --build "${BUILD_DIR}" --config Release
else
    cmake --build "${BUILD_DIR}" --config Release --target GgmlOps
fi
