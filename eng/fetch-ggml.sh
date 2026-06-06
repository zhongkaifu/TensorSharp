#!/usr/bin/env bash
# Clone the upstream ggml-org/ggml repository into ExternalProjects/ggml when
# the sources are missing.
#
# TensorSharp builds ggml from source: both the GGML native ops library
# (TensorSharp.GGML.Native/CMakeLists.txt) and the CUDA PTX kernels
# (TensorSharp.Backends.Cuda/native/kernels/tensorsharp_kernels.cu) consume the
# sources at ExternalProjects/ggml. The directory is not committed; it is fetched
# here on first use, then reused for faster and offline-friendly rebuilds.
#
# Environment overrides:
#   TENSORSHARP_GGML_GIT_URL   git URL                (default: ggml-org/ggml)
#   TENSORSHARP_GGML_GIT_REF   branch/tag/commit      (default: master, the ggml default branch)
#   TENSORSHARP_GGML_UPDATE    if set to 1/ON/true and a checkout already exists,
#                              fetch and reset it to the requested ref.
#   TENSORSHARP_GGML_NO_UPDATE legacy override; if truthy, always use what is on disk.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GGML_DIR="${REPO_ROOT}/ExternalProjects/ggml"
GGML_REQUIRED_HEADER="${GGML_DIR}/src/ggml-common.h"

GIT_URL="${TENSORSHARP_GGML_GIT_URL:-https://github.com/ggml-org/ggml.git}"
GIT_REF="${TENSORSHARP_GGML_GIT_REF:-master}"
UPDATE_RAW="${TENSORSHARP_GGML_UPDATE:-}"
NO_UPDATE_RAW="${TENSORSHARP_GGML_NO_UPDATE:-}"

is_truthy() {
    case "${1:-}" in
        1|ON|on|On|TRUE|true|True|YES|yes|Yes) return 0 ;;
        *) return 1 ;;
    esac
}

if [[ -d "${GGML_DIR}/.git" ]]; then
    if is_truthy "${NO_UPDATE_RAW}"; then
        echo "ggml: TENSORSHARP_GGML_NO_UPDATE set; using existing checkout at ${GGML_DIR}"
        exit 0
    fi

    if [[ -f "${GGML_REQUIRED_HEADER}" ]] && ! is_truthy "${UPDATE_RAW}"; then
        echo "ggml: using existing checkout at ${GGML_DIR} (set TENSORSHARP_GGML_UPDATE=1 to fetch updates)"
        exit 0
    fi

    if [[ -f "${GGML_REQUIRED_HEADER}" ]]; then
        echo "ggml: updating existing checkout to ${GIT_REF} (${GIT_URL})"
    else
        echo "ggml: existing checkout is missing src/ggml-common.h; fetching ${GIT_REF} (${GIT_URL})"
    fi
    git -C "${GGML_DIR}" remote set-url origin "${GIT_URL}" 2>/dev/null || true
    if git -C "${GGML_DIR}" fetch --depth 1 origin "${GIT_REF}" 2>/dev/null; then
        if git -C "${GGML_DIR}" reset --hard FETCH_HEAD; then
            git -C "${GGML_DIR}" rev-parse --short HEAD | sed 's/^/ggml: now at /'
        elif [[ -f "${GGML_REQUIRED_HEADER}" ]]; then
            echo "ggml: WARNING - fetched ${GIT_REF}, but could not reset checkout; using existing sources" >&2
        else
            echo "ggml: ERROR - fetched ${GIT_REF}, but could not reset checkout and no usable sources exist" >&2
            exit 128
        fi
    else
        if [[ -f "${GGML_REQUIRED_HEADER}" ]]; then
            echo "ggml: WARNING - could not fetch ${GIT_REF} (offline?); using existing checkout" >&2
        else
            echo "ggml: ERROR - could not fetch ${GIT_REF}, and no usable ggml sources exist at ${GGML_DIR}" >&2
            exit 128
        fi
    fi
    exit 0
fi

if [[ -f "${GGML_REQUIRED_HEADER}" ]] && ! is_truthy "${UPDATE_RAW}"; then
    echo "ggml: using existing source tree at ${GGML_DIR} (not a git checkout)"
    exit 0
fi

# No checkout yet: a partial/empty directory left by a failed clone would make
# 'git clone' fail, so clear it first.
if [[ -e "${GGML_DIR}" ]]; then
    rm -rf "${GGML_DIR}"
fi

echo "ggml: cloning ${GIT_URL} (${GIT_REF}) into ${GGML_DIR}"
if git clone --depth 1 --branch "${GIT_REF}" "${GIT_URL}" "${GGML_DIR}" 2>/dev/null; then
    :
else
    # --branch only accepts a branch or tag; fall back to fetching an explicit
    # commit ref shallowly.
    rm -rf "${GGML_DIR}"
    git init -q "${GGML_DIR}"
    git -C "${GGML_DIR}" remote add origin "${GIT_URL}"
    git -C "${GGML_DIR}" fetch --depth 1 origin "${GIT_REF}"
    git -C "${GGML_DIR}" checkout -q FETCH_HEAD
fi
git -C "${GGML_DIR}" rev-parse --short HEAD | sed 's/^/ggml: cloned at /'
