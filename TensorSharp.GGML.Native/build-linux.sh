#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
ENABLE_CUDA="${TENSORSHARP_GGML_NATIVE_ENABLE_CUDA:-}"
BUILD_TESTS="${TENSORSHARP_GGML_NATIVE_BUILD_TESTS:-OFF}"
EXTRA_CMAKE_ARGS=()

normalize_bool() {
    local value="${1:-}"
    case "${value}" in
        ON|on|On|TRUE|true|True|YES|yes|Yes|1)
            echo "ON"
            ;;
        OFF|off|Off|FALSE|false|False|NO|no|No|0)
            echo "OFF"
            ;;
        *)
            echo ""
            ;;
    esac
}

has_cuda_toolkit() {
    if command -v nvcc >/dev/null 2>&1; then
        return 0
    fi

    local cuda_home="${CUDA_HOME:-${CUDA_PATH:-}}"
    if [[ -n "${cuda_home}" && -x "${cuda_home}/bin/nvcc" ]]; then
        return 0
    fi

    return 1
}

read_cached_cuda_setting() {
    local cache_file="${BUILD_DIR}/CMakeCache.txt"
    if [[ ! -f "${cache_file}" ]]; then
        echo ""
        return
    fi

    local cached
    cached="$(awk -F= '/^TENSORSHARP_GGML_NATIVE_ENABLE_CUDA:BOOL=/{print $2; exit}' "${cache_file}")"
    normalize_bool "${cached}"
}

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

ENABLE_CUDA="$(normalize_bool "${ENABLE_CUDA}")"
if [[ -z "${ENABLE_CUDA}" ]]; then
    ENABLE_CUDA="$(read_cached_cuda_setting)"
fi
if [[ -z "${ENABLE_CUDA}" ]] && has_cuda_toolkit; then
    ENABLE_CUDA="ON"
fi
if [[ -z "${ENABLE_CUDA}" ]]; then
    ENABLE_CUDA="OFF"
fi

echo "Configuring TensorSharp.GGML.Native (CUDA=${ENABLE_CUDA}, TESTS=${BUILD_TESTS})"

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
