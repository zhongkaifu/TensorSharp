#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Release}"
PLATFORM="${PLATFORM:-$(uname -m)}"
ENABLE_CUDA="${TSG_ENABLE_CUDA:-OFF}"
BUILD_DIR="${SCRIPT_DIR}/build/${PLATFORM}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --platform)
            PLATFORM="${2:?missing platform value}"
            BUILD_DIR="${SCRIPT_DIR}/build/${PLATFORM}"
            shift 2
            ;;
        --cuda)
            ENABLE_CUDA="${2:?missing cuda value}"
            shift 2
            ;;
        --build-type)
            BUILD_TYPE="${2:?missing build type value}"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

mkdir -p "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DTSG_ENABLE_CUDA="${ENABLE_CUDA}"
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" --target GgmlOps

# Keep a stable path for managed probing/copy steps.
cp -f "${BUILD_DIR}/libGgmlOps.so" "${SCRIPT_DIR}/build/libGgmlOps.so"
