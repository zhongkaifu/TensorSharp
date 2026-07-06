#!/usr/bin/env bash
# Provision the Vulkan build toolchain needed by the ggml-vulkan backend
# (TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON) into ExternalProjects/vulkan-toolchain.
#
# Linux counterpart of eng/fetch-vulkan-toolchain.ps1. ggml-vulkan needs three
# things at build time that a machine with only a Vulkan *runtime*
# (libvulkan.so.1 from the GPU driver stack) does not have:
#   1. Vulkan headers (vulkan_core.h / vulkan.hpp)  -> KhronosGroup/Vulkan-Headers
#   2. The glslc GLSL->SPIR-V compiler              -> Google shaderc CI prebuilt
#      (skipped when glslc is already on PATH)
#   3. The SPIRV-Headers CMake package (find_package(SPIRV-Headers) in ggml)
# The loader itself is NOT provisioned: build-linux.sh links against the system
# libvulkan.so / libvulkan.so.1 it finds via ldconfig.
#
# The toolchain directory is shared with the Windows script: Vulkan-Headers and
# the SPIRV-Headers install are OS-neutral (headers only); the Linux glslc lives
# in its own shaderc-linux/ subdirectory next to the Windows shaderc/.
#
# When a LunarG Vulkan SDK is installed (VULKAN_SDK env var) everything above is
# already available and this script exits without doing anything; build-linux.sh
# then lets CMake's FindVulkan discover the SDK on its own.
#
# Environment overrides:
#   TENSORSHARP_GGML_NO_UPDATE  if set to 1/ON/true and the toolchain is already
#                               populated, skip all network access and reuse it
#                               (same contract as eng/fetch-ggml.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLCHAIN_DIR="${REPO_ROOT}/ExternalProjects/vulkan-toolchain"

is_truthy() {
    case "${1:-}" in
        1|ON|on|On|TRUE|true|True|YES|yes|Yes) return 0 ;;
        *) return 1 ;;
    esac
}

vulkan_sdk_complete() {
    local sdk="${VULKAN_SDK:-}"
    [[ -n "${sdk}" ]] || return 1
    [[ -f "${sdk}/include/vulkan/vulkan.h" && -x "${sdk}/bin/glslc" ]] || return 1
    [[ -e "${sdk}/lib/libvulkan.so" || -e "${sdk}/lib/libvulkan.so.1" ]] || return 1
}

have_glslc() {
    command -v glslc >/dev/null 2>&1 || [[ -x "${TOOLCHAIN_DIR}/shaderc-linux/bin/glslc" ]]
}

toolchain_complete() {
    [[ -f "${TOOLCHAIN_DIR}/Vulkan-Headers/include/vulkan/vulkan.h" ]] &&
        [[ -f "${TOOLCHAIN_DIR}/spirv-headers-install/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake" ]] &&
        have_glslc
}

download() {
    local url="$1" out="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${url}" -o "${out}"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "${url}" -O "${out}"
    else
        echo "vulkan-toolchain: neither curl nor wget is available to download ${url}" >&2
        return 1
    fi
}

if vulkan_sdk_complete; then
    echo "vulkan-toolchain: using installed Vulkan SDK at ${VULKAN_SDK}"
    exit 0
fi

if toolchain_complete; then
    if is_truthy "${TENSORSHARP_GGML_NO_UPDATE:-}"; then
        echo "vulkan-toolchain: TENSORSHARP_GGML_NO_UPDATE set; using existing toolchain at ${TOOLCHAIN_DIR}"
    else
        echo "vulkan-toolchain: already populated at ${TOOLCHAIN_DIR}"
    fi
    exit 0
fi

echo "vulkan-toolchain: provisioning portable Vulkan toolchain in ${TOOLCHAIN_DIR}"
mkdir -p "${TOOLCHAIN_DIR}"

# --- 1. Vulkan headers -------------------------------------------------------
HEADERS_DIR="${TOOLCHAIN_DIR}/Vulkan-Headers"
if [[ ! -f "${HEADERS_DIR}/include/vulkan/vulkan.h" ]]; then
    rm -rf "${HEADERS_DIR}"
    git clone --depth 1 https://github.com/KhronosGroup/Vulkan-Headers.git "${HEADERS_DIR}"
fi

# --- 2. SPIRV-Headers CMake package -----------------------------------------
SPIRV_SRC_DIR="${TOOLCHAIN_DIR}/SPIRV-Headers"
SPIRV_INSTALL_DIR="${TOOLCHAIN_DIR}/spirv-headers-install"
if [[ ! -f "${SPIRV_INSTALL_DIR}/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake" ]]; then
    if [[ ! -f "${SPIRV_SRC_DIR}/CMakeLists.txt" ]]; then
        rm -rf "${SPIRV_SRC_DIR}"
        git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers.git "${SPIRV_SRC_DIR}"
    fi
    # Separate binary dir: the Windows script configures build/ with an MSVC
    # generator in the same shared checkout, which a Linux cmake cannot reuse.
    cmake -S "${SPIRV_SRC_DIR}" -B "${SPIRV_SRC_DIR}/build-linux" \
        -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=${SPIRV_INSTALL_DIR}"
    cmake --install "${SPIRV_SRC_DIR}/build-linux" --config Release
fi

# --- 3. glslc (Google shaderc CI prebuilt; skipped when already on PATH) -----
SHADERC_DIR="${TOOLCHAIN_DIR}/shaderc-linux"
if ! command -v glslc >/dev/null 2>&1 && [[ ! -x "${SHADERC_DIR}/bin/glslc" ]]; then
    # The build-link badge HTML that the Windows script uses points at a
    # since-pruned bucket layout for Linux, so resolve the newest continuous
    # build's install.tgz through the public GCS object-listing API instead.
    LIST_URL="https://storage.googleapis.com/storage/v1/b/shaderc/o?prefix=artifacts/prod/graphics_shader_compiler/shaderc/linux/continuous_clang_release/&fields=items(name)"
    LIST_PATH="${TOOLCHAIN_DIR}/shaderc-listing.json"
    download "${LIST_URL}" "${LIST_PATH}"
    TGZ_NAME="$(grep -oE '"name"[[:space:]]*:[[:space:]]*"[^"]+/install\.tgz"' "${LIST_PATH}" \
        | sed -E 's/"name"[[:space:]]*:[[:space:]]*"([^"]+)"/\1/' | sort -V | tail -n 1)"
    rm -f "${LIST_PATH}"
    if [[ -z "${TGZ_NAME}" ]]; then
        echo "vulkan-toolchain: could not resolve a shaderc prebuilt install.tgz from ${LIST_URL}" >&2
        exit 1
    fi
    TGZ_URL="https://storage.googleapis.com/shaderc/${TGZ_NAME}"
    TGZ_PATH="${TOOLCHAIN_DIR}/shaderc-install.tgz"
    echo "vulkan-toolchain: downloading glslc from ${TGZ_URL}"
    download "${TGZ_URL}" "${TGZ_PATH}"
    EXTRACT_DIR="${TOOLCHAIN_DIR}/shaderc-extract"
    rm -rf "${EXTRACT_DIR}" "${SHADERC_DIR}"
    mkdir -p "${EXTRACT_DIR}"
    tar -xzf "${TGZ_PATH}" -C "${EXTRACT_DIR}"
    # The tarball contains a single top-level "install" folder with bin/lib/include.
    mv "${EXTRACT_DIR}/install" "${SHADERC_DIR}"
    rm -rf "${EXTRACT_DIR}" "${TGZ_PATH}"
    chmod +x "${SHADERC_DIR}/bin/glslc"
fi

if ! toolchain_complete; then
    echo "vulkan-toolchain: provisioning finished but the toolchain is still incomplete at ${TOOLCHAIN_DIR}" >&2
    exit 1
fi
echo "vulkan-toolchain: ready at ${TOOLCHAIN_DIR}"
