#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
ENABLE_CUDA="${TENSORSHARP_GGML_NATIVE_ENABLE_CUDA:-}"
ENABLE_VULKAN="${TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN:-}"
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

# Prints the system Vulkan loader library path (preferring the dev symlink
# libvulkan.so over the runtime soname libvulkan.so.1 — CMake accepts a full
# path in Vulkan_LIBRARY either way). Fails when no loader is installed, which
# is the "this machine does not support Vulkan" signal for auto-detection.
find_vulkan_loader_library() {
    local candidates=()
    if command -v ldconfig >/dev/null 2>&1; then
        while IFS= read -r line; do
            candidates+=("${line}")
        done < <(ldconfig -p 2>/dev/null | awk '$1 == "libvulkan.so" || $1 == "libvulkan.so.1" { print $NF }')
    fi
    local dir name
    for dir in /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/lib64 /usr/lib /usr/local/lib; do
        for name in libvulkan.so libvulkan.so.1; do
            if [[ -e "${dir}/${name}" ]]; then
                candidates+=("${dir}/${name}")
            fi
        done
    done

    local fallback=""
    local candidate
    for candidate in ${candidates[@]+"${candidates[@]}"}; do
        if [[ "${candidate}" == *libvulkan.so ]]; then
            echo "${candidate}"
            return 0
        fi
        if [[ -z "${fallback}" ]]; then
            fallback="${candidate}"
        fi
    done
    if [[ -n "${fallback}" ]]; then
        echo "${fallback}"
        return 0
    fi
    return 1
}

# Ensures the Vulkan build toolchain (headers, glslc, SPIRV-Headers) exists,
# downloading the portable pieces via eng/fetch-vulkan-toolchain.sh when the
# system does not already provide them, and fills VULKAN_CMAKE_ARGS with
# explicit FindVulkan hints. Returns non-zero when the toolchain cannot be
# provisioned (no loader, no network, ...).
VULKAN_CMAKE_ARGS=()
prepare_vulkan_toolchain() {
    VULKAN_CMAKE_ARGS=()

    # Full LunarG SDK: CMake's FindVulkan discovers everything via VULKAN_SDK.
    if [[ -n "${VULKAN_SDK:-}" && -f "${VULKAN_SDK}/include/vulkan/vulkan.h" && -x "${VULKAN_SDK}/bin/glslc" ]]; then
        return 0
    fi

    local loader
    if ! loader="$(find_vulkan_loader_library)"; then
        echo "warning: no Vulkan loader (libvulkan.so / libvulkan.so.1) found on this system." >&2
        return 1
    fi

    if ! bash "${SCRIPT_DIR}/../eng/fetch-vulkan-toolchain.sh"; then
        return 1
    fi

    local toolchain_dir
    toolchain_dir="$(cd "${SCRIPT_DIR}/.." && pwd)/ExternalProjects/vulkan-toolchain"

    local glslc_path
    if command -v glslc >/dev/null 2>&1; then
        glslc_path="$(command -v glslc)"
    elif [[ -x "${toolchain_dir}/shaderc-linux/bin/glslc" ]]; then
        glslc_path="${toolchain_dir}/shaderc-linux/bin/glslc"
    else
        echo "warning: glslc not found after provisioning the Vulkan toolchain." >&2
        return 1
    fi

    VULKAN_CMAKE_ARGS=(
        "-DVulkan_INCLUDE_DIR=${toolchain_dir}/Vulkan-Headers/include"
        "-DVulkan_LIBRARY=${loader}"
        "-DVulkan_GLSLC_EXECUTABLE=${glslc_path}"
        "-DSPIRV-Headers_DIR=${toolchain_dir}/spirv-headers-install/share/cmake/SPIRV-Headers"
    )
    return 0
}

read_cached_backend_setting() {
    local cache_variable="$1"
    local cache_file="${BUILD_DIR}/CMakeCache.txt"
    if [[ ! -f "${cache_file}" ]]; then
        echo ""
        return
    fi

    local cached
    cached="$(awk -F= -v var="^${cache_variable}:BOOL=" '$0 ~ var {print $2; exit}' "${cache_file}")"
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

    # Use all CPUs, bounded only by RAM. nvcc is memory-hungry, so allow ~3 GB per
    # parallel job (matches llama.cpp/ggml's own heuristic). The previous hard cap
    # of 4 jobs throttled the CUDA compile to a fraction of available cores even on
    # large machines with plenty of RAM; the memory bound below already prevents
    # OOM. Override with TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL when needed.
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
        --vulkan)
            ENABLE_VULKAN=ON
            ;;
        --no-vulkan)
            ENABLE_VULKAN=OFF
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
    ENABLE_CUDA="$(read_cached_backend_setting TENSORSHARP_GGML_NATIVE_ENABLE_CUDA)"
fi
if [[ -z "${ENABLE_CUDA}" ]] && has_cuda_toolkit; then
    ENABLE_CUDA="ON"
fi
if [[ -z "${ENABLE_CUDA}" ]]; then
    ENABLE_CUDA="OFF"
fi

# Vulkan: an explicit choice (--vulkan/--no-vulkan or
# TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN) wins and sticks via the CMake cache;
# TENSORSHARP_GGML_NATIVE_VULKAN_EXPLICIT records that the cached value was
# explicit, so an auto-detected default never becomes sticky. Otherwise the
# backend is auto-enabled when the machine supports Vulkan (a loader is
# installed) and the build toolchain (headers, glslc, SPIRV-Headers) is
# downloaded automatically by eng/fetch-vulkan-toolchain.sh when missing. An
# auto-enable whose toolchain cannot be provisioned (e.g. no network) degrades
# to a warning and builds without Vulkan; an explicit --vulkan makes it fatal.
VULKAN_EXPLICIT=OFF
ENABLE_VULKAN="$(normalize_bool "${ENABLE_VULKAN}")"
if [[ -n "${ENABLE_VULKAN}" ]]; then
    VULKAN_EXPLICIT=ON
elif [[ "$(read_cached_backend_setting TENSORSHARP_GGML_NATIVE_VULKAN_EXPLICIT)" == "ON" ]]; then
    ENABLE_VULKAN="$(read_cached_backend_setting TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN)"
    if [[ -n "${ENABLE_VULKAN}" ]]; then
        VULKAN_EXPLICIT=ON
    fi
fi
if [[ -z "${ENABLE_VULKAN}" ]]; then
    if [[ -n "${VULKAN_SDK:-}" ]] || find_vulkan_loader_library >/dev/null; then
        ENABLE_VULKAN=ON
    else
        ENABLE_VULKAN=OFF
    fi
fi
if [[ "${ENABLE_VULKAN}" == "ON" ]] && ! prepare_vulkan_toolchain; then
    if [[ "${VULKAN_EXPLICIT}" == "ON" ]]; then
        echo "error: the ggml-vulkan backend was requested explicitly but its build toolchain could not be provisioned." >&2
        echo "       Install the Vulkan SDK (e.g. apt install libvulkan-dev glslc spirv-headers) or set VULKAN_SDK." >&2
        exit 1
    fi
    echo "warning: this machine supports Vulkan but the ggml-vulkan build toolchain could not be provisioned;" >&2
    echo "         building without the ggml-vulkan backend. Pass --vulkan to make this an error." >&2
    ENABLE_VULKAN=OFF
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

echo "Configuring TensorSharp.GGML.Native (CUDA=${ENABLE_CUDA}, CUDA_ARCHITECTURES=${CUDA_ARCH_SUMMARY}, VULKAN=${ENABLE_VULKAN}, TESTS=${BUILD_TESTS}, PARALLEL=${BUILD_PARALLEL_LEVEL})"

# Ensure the ggml sources are present (cloned from ggml-org/ggml at build time).
bash "${SCRIPT_DIR}/../eng/fetch-ggml.sh"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA="${ENABLE_CUDA}"
    -DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN="${ENABLE_VULKAN}"
    -DTENSORSHARP_GGML_NATIVE_VULKAN_EXPLICIT="${VULKAN_EXPLICIT}"
    -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS="${BUILD_TESTS}"
    ${VULKAN_CMAKE_ARGS[@]+"${VULKAN_CMAKE_ARGS[@]}"}
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
