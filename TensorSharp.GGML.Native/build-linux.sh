#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Release}"
PLATFORM="${PLATFORM:-$(uname -m)}"
ENABLE_CUDA="${TSG_ENABLE_CUDA:-auto}"
BUILD_DIR="${SCRIPT_DIR}/build/${PLATFORM}"

has_cuda_toolchain() {
    if command -v nvcc >/dev/null 2>&1; then
        return 0
    fi

    if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
        return 0
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi

    if command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q "libcudart\\.so"; then
        return 0
    fi

    return 1
}

normalize_cuda_toggle() {
    local value="${1,,}"
    case "${value}" in
        1|on|true|yes) echo "ON" ;;
        0|off|false|no) echo "OFF" ;;
        auto)
            if has_cuda_toolchain; then
                echo "ON"
            else
                echo "OFF"
            fi
            ;;
        *)
            echo "Invalid CUDA toggle '${1}'. Use ON/OFF/auto." >&2
            exit 1
            ;;
    esac
}

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

ENABLE_CUDA="$(normalize_cuda_toggle "${ENABLE_CUDA}")"
echo "Building GgmlOps for platform='${PLATFORM}', buildType='${BUILD_TYPE}', cuda='${ENABLE_CUDA}'."

mkdir -p "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DTSG_ENABLE_CUDA="${ENABLE_CUDA}"
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" --target GgmlOps

# Keep a stable path for managed probing/copy steps.
cp -f "${BUILD_DIR}/libGgmlOps.so" "${SCRIPT_DIR}/build/libGgmlOps.so"
