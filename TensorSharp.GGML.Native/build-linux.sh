#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
ENABLE_CUDA="${TENSORSHARP_GGML_NATIVE_ENABLE_CUDA:-}"
BUILD_TESTS="${TENSORSHARP_GGML_NATIVE_BUILD_TESTS:-OFF}"
CUDA_ARCHITECTURES="${TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES:-}"
BUILD_PARALLEL_LEVEL="${TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL:-${CMAKE_BUILD_PARALLEL_LEVEL:-}}"
EXTRA_CMAKE_ARGS=()
USER_SET_CMAKE_CUDA_ARCHITECTURES=OFF

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

detect_local_cuda_architectures() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 1
    fi

    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | \
        awk '
            BEGIN {
                first = 1
            }
            {
                gsub(/[[:space:]]/, "", $0)
                if ($0 ~ /^[0-9]+\.[0-9]+$/) {
                    split($0, cap, ".")
                    arch = cap[1] cap[2] "-real"
                    if (!(arch in seen)) {
                        seen[arch] = 1
                        if (!first) {
                            printf(";")
                        }
                        printf("%s", arch)
                        first = 0
                    }
                }
            }
            END {
                if (first) {
                    exit 1
                }
            }
        '
}

detect_default_build_parallel_level() {
    local cpu_count=1
    if command -v nproc >/dev/null 2>&1; then
        cpu_count="$(nproc)"
    fi

    local jobs="${cpu_count}"
    if [[ -r /proc/meminfo ]]; then
        local mem_kb
        mem_kb="$(awk '/^MemTotal:/ { print $2; exit }' /proc/meminfo)"
        if [[ "${mem_kb}" =~ ^[0-9]+$ ]]; then
            local memory_limited_jobs=$(( mem_kb / (3 * 1024 * 1024) ))
            if (( memory_limited_jobs < 1 )); then
                memory_limited_jobs=1
            fi
            if (( memory_limited_jobs < jobs )); then
                jobs="${memory_limited_jobs}"
            fi
        fi
    fi

    if (( jobs > 4 )); then
        jobs=4
    fi

    echo "${jobs}"
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
        --cuda-arch=*)
            CUDA_ARCHITECTURES="${1#*=}"
            ;;
        --cuda-arch)
            shift
            if (($# == 0)); then
                echo "Missing value for --cuda-arch" >&2
                exit 1
            fi
            CUDA_ARCHITECTURES="$1"
            ;;
        -DCMAKE_CUDA_ARCHITECTURES=*)
            USER_SET_CMAKE_CUDA_ARCHITECTURES=ON
            EXTRA_CMAKE_ARGS+=("$1")
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

if [[ "${ENABLE_CUDA}" == "ON" && -z "${CUDA_ARCHITECTURES}" && "${USER_SET_CMAKE_CUDA_ARCHITECTURES}" != "ON" ]]; then
    if detected_cuda_architectures="$(detect_local_cuda_architectures)"; then
        CUDA_ARCHITECTURES="${detected_cuda_architectures}"
    fi
fi

CUDA_ARCH_SUMMARY="n/a"
if [[ "${ENABLE_CUDA}" == "ON" ]]; then
    if [[ "${USER_SET_CMAKE_CUDA_ARCHITECTURES}" == "ON" ]]; then
        CUDA_ARCH_SUMMARY="custom via CMAKE_CUDA_ARCHITECTURES"
    elif [[ -n "${CUDA_ARCHITECTURES}" ]]; then
        CUDA_ARCH_SUMMARY="${CUDA_ARCHITECTURES}"
    else
        CUDA_ARCH_SUMMARY="ggml default"
    fi
fi

BUILD_PARALLEL_ARGS=()
if [[ -z "${BUILD_PARALLEL_LEVEL}" ]]; then
    BUILD_PARALLEL_LEVEL="$(detect_default_build_parallel_level)"
fi
if [[ ! "${BUILD_PARALLEL_LEVEL}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid build parallel level: ${BUILD_PARALLEL_LEVEL}" >&2
    exit 1
fi
BUILD_PARALLEL_ARGS=(--parallel "${BUILD_PARALLEL_LEVEL}")

echo "Configuring TensorSharp.GGML.Native (CUDA=${ENABLE_CUDA}, CUDA_ARCHITECTURES=${CUDA_ARCH_SUMMARY}, TESTS=${BUILD_TESTS}, PARALLEL=${BUILD_PARALLEL_LEVEL})"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA="${ENABLE_CUDA}"
    -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS="${BUILD_TESTS}"
)

if [[ "${ENABLE_CUDA}" == "ON" && -n "${CUDA_ARCHITECTURES}" && "${USER_SET_CMAKE_CUDA_ARCHITECTURES}" != "ON" ]]; then
    CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}")
fi

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    "${CMAKE_ARGS[@]}" \
    "${EXTRA_CMAKE_ARGS[@]}"

if [[ "${BUILD_TESTS}" == "ON" ]]; then
    cmake --build "${BUILD_DIR}" --config Release "${BUILD_PARALLEL_ARGS[@]}"
else
    cmake --build "${BUILD_DIR}" --config Release "${BUILD_PARALLEL_ARGS[@]}" --target GgmlOps
fi
