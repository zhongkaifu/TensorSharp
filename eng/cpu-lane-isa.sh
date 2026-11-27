#!/usr/bin/env bash
# Run the PR CPU correctness lane (pr-unit-tests.yml, "Run environment-independent tests") under the
# instruction sets the CI runners take, on one x64 Linux host.
#
# A test that compares one computation run in steps of different row counts (a prompt in pieces
# against whole, a lane beside others against alone) sees the kernels' rounding by row count, and
# which kernels a run meets depends on the host's ISA (DEVELOPMENT.md, "Rounding by row count"). A
# bound measured only on an AVX-512 host has failed again and again on the runners that lack it.
# This runs the lane as each of them would:
#
#   native    this host's widest kernels (an AVX-512 host's managed kernels, ggml as built)
#   avx2      the x64 CI job: managed kernels and the JIT without AVX-512
#             (TS_CPU_DISABLE_AVX512=1, DOTNET_EnableAVX512=0). The job also builds ggml for the
#             x86-64-v3 baseline (-DGGML_NATIVE=OFF): --ggml-baseline rebuilds this checkout's native
#             library the same way, and it stays so until build-linux.sh -DGGML_NATIVE=ON.
#   portable  the managed portable 128-bit Vector<T> paths ARM64 takes (DOTNET_EnableAVX2=0). The
#             ARM64-only kernels (AdvSimd host attention, the ARM quantized GEMM) run only on an ARM64
#             host; the ARM64 CI job is the reference for them.
#
# Usage: eng/cpu-lane-isa.sh [--no-build] [--ggml-baseline] [--filter <extra filter>] [profile...]
#   profiles default to: native avx2 portable
#   --filter "FullyQualifiedName~Gemma4RuntimeTests" narrows the lane (ANDed with the lane's filter).
# Logs and TRX files go to artifacts/cpu-lane-isa/<profile>/ (ignored by Git).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${REPO_ROOT}/InferenceWeb.Tests/InferenceWeb.Tests.csproj"
OUT_ROOT="${REPO_ROOT}/artifacts/cpu-lane-isa"
LANE_FILTER="Category!=Bench&Requires!=Cuda&Requires!=Mlx&Requires!=Models"

build=1
ggml_baseline=0
extra_filter=""
profiles=()
while (($# > 0)); do
    case "$1" in
        --no-build) build=0 ;;
        --ggml-baseline) ggml_baseline=1 ;;
        --filter)
            shift
            (($# > 0)) || { echo "--filter needs a value" >&2; exit 2; }
            extra_filter="$1"
            ;;
        native|avx2|portable) profiles+=("$1") ;;
        -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
    shift
done
((${#profiles[@]} > 0)) || profiles=(native avx2 portable)

arch="$(uname -m)"
for p in "${profiles[@]}"; do
    if [[ "${p}" != native && "${arch}" != x86_64 ]]; then
        echo "profile '${p}' emulates an x64 runner and needs an x64 host (this one is ${arch})" >&2
        exit 2
    fi
done

# The same environment the CI job sets (GPU backends off, the CPU GGML backend pinned).
export TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=OFF TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF TENSORSHARP_CUDNN=OFF
export TS_TEST_GGML_BACKEND=cpu DOTNET_NOLOGO=true

if ((ggml_baseline)); then
    [[ "${arch}" == x86_64 ]] || { echo "--ggml-baseline is for x64 hosts" >&2; exit 2; }
    bash "${REPO_ROOT}/TensorSharp.GGML.Native/build-linux.sh" -DGGML_NATIVE=OFF
fi
if ((build)); then
    dotnet build "${PROJECT}"
fi

filter="${LANE_FILTER}"
[[ -z "${extra_filter}" ]] || filter="${filter}&${extra_filter}"

if [[ -f "${REPO_ROOT}/TensorSharp.GGML.Native/build/CMakeCache.txt" ]]; then
    ggml_native="$(awk -F= '/^GGML_NATIVE:/ { print $2; exit }' "${REPO_ROOT}/TensorSharp.GGML.Native/build/CMakeCache.txt")"
else
    ggml_native="unknown"
fi
ggml_rev="$(git -C "${REPO_ROOT}/ExternalProjects/ggml" rev-parse --short HEAD 2>/dev/null || echo unknown)"
cpu="$(awk -F': ' '/^model name/ { print $2; exit }' /proc/cpuinfo 2>/dev/null || true)"
echo "host: ${cpu:-${arch}}; ggml ${ggml_rev}, GGML_NATIVE=${ggml_native}"

declare -A result
status=0
for p in "${profiles[@]}"; do
    env_args=()
    case "${p}" in
        avx2) env_args=(TS_CPU_DISABLE_AVX512=1 DOTNET_EnableAVX512=0) ;;
        portable) env_args=(DOTNET_EnableAVX2=0) ;;
    esac
    out="${OUT_ROOT}/${p}"
    mkdir -p "${out}"
    echo "=== ${p} ${env_args[*]:-}"
    set +e
    env ${env_args[@]+"${env_args[@]}"} dotnet test "${PROJECT}" --no-build --filter "${filter}" \
        --results-directory "${out}" --logger "trx;LogFileName=${p}.trx" 2>&1 | tee "${out}/${p}.log"
    rc=${PIPESTATUS[0]}
    set -e
    result["${p}"]="$(grep -E '^(Passed|Failed)!' "${out}/${p}.log" | tail -1 || true)"
    ((rc == 0)) || status=1
done

echo
for p in "${profiles[@]}"; do
    echo "${p}: ${result[${p}]:-no summary (see ${OUT_ROOT}/${p}/${p}.log)}"
done
exit "${status}"
