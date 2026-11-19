#!/usr/bin/env bash
# The LFM3 single-sequence matrix on one Linux GPU: TensorSharp's ggml_cuda, direct cuda and ggml_vulkan lanes against
# llama-bench's CUDA and Vulkan builds (flash attention on and off), every arm interleaved (ab.py), results under
# artifacts/lfm3/bench/<stamp>/ (one markdown table per quant, raw logs beside it, conditions.txt).
#
# usage: gpu-bench.sh [QUANT ...]                       (default: Q4_K_M Q8_0 QAD-Q4_0 BF16 NVFP4)
#   MODELS     GGUF directory (default /root/models/lfm3)
#   LLAMA_CPP  llama.cpp checkout with build-cuda/ and build-vk/ (default /root/llama.cpp)
#   ARMS       arms to run (default "ts-ggml_cuda ts-cuda ts-ggml_vulkan llama-cuda-fa1 llama-cuda-fa0 llama-vk-fa1 llama-vk-fa0")
#   PP         prompt lengths (default 512,4096,16384,32768); TG decode tokens (default 128);
#   DEPTHS     decode depths (default 0,4096,16384,32768); ROUNDS interleaved rounds (default 3); REPS repeats a run (default 3)
# Build first (Release): TENSORSHARP_GGML_NO_UPDATE=1 dotnet build eng/lfm3/Lfm3Probe -c Release (with the CUDA and Vulkan
# native backends). The GPU must be idle (the script checks) and stay so: no nvidia-smi -l, no other jobs.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
M=${MODELS:-/root/models/lfm3}
L=${LLAMA_CPP:-/root/llama.cpp}
ARMS=${ARMS:-"ts-ggml_cuda ts-cuda ts-ggml_vulkan llama-cuda-fa1 llama-cuda-fa0 llama-vk-fa1 llama-vk-fa0"}
PP=${PP:-512,4096,16384,32768}
TG=${TG:-128}
DEPTHS=${DEPTHS:-0,4096,16384,32768}
ROUNDS=${ROUNDS:-3}
REPS=${REPS:-3}
quants=${*:-Q4_K_M Q8_0 QAD-Q4_0 BF16 NVFP4}
deepest=${DEPTHS##*,}; longest=${PP##*,}
ctx=$(( (deepest > longest ? deepest : longest) + TG + 1024 ))
stamp=$(date +%Y%m%d-%H%M%S)
out=$root/artifacts/lfm3/bench/gpu-$stamp
mkdir -p "$out"
if [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
  echo "the GPU is busy:" >&2; nvidia-smi --query-compute-apps=pid,name --format=csv,noheader >&2; exit 3
fi
{ uptime; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; git -C "$root" log --oneline -1;
  git -C "$L" log --oneline -1 2>/dev/null; echo "PP=$PP TG=$TG DEPTHS=$DEPTHS ROUNDS=$ROUNDS REPS=$REPS ARMS=$ARMS"; } > "$out/conditions.txt"
echo "results: $out" >&2

arm() {   # NAME QUANT -> "NAME=COMMAND" for ab.py
  local m=$M/LFM3-1.7B-$2.gguf
  case $1 in
    ts-*) echo "$1=MAX_CONTEXT=$ctx $here/ts-bench.sh $m ${1#ts-} $PP $TG $DEPTHS --reps $REPS" ;;
    llama-cuda-fa*) echo "$1=LLAMA_CPP_DIR=$L LLAMA_BUILD=build-cuda FA=${1##*fa} $here/llama-bench-result.sh $m $PP $TG $DEPTHS $REPS" ;;
    llama-vk-fa*) echo "$1=LLAMA_CPP_DIR=$L LLAMA_BUILD=build-vk FA=${1##*fa} $here/llama-bench-result.sh $m $PP $TG $DEPTHS $REPS" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

for q in $quants; do
  [[ -f $M/LFM3-1.7B-$q.gguf ]] || { echo "skip $q: $M/LFM3-1.7B-$q.gguf missing" >&2; continue; }
  args=()
  for a in $ARMS; do args+=("$(arm "$a" "$q")"); done
  python3 "$here/ab.py" --rounds "$ROUNDS" --log "$out/$q" "${args[@]}" | tee "$out/$q.md"
done
uptime >> "$out/conditions.txt"
echo "results: $out" >&2
