#!/usr/bin/env bash
# TensorSharp ggml_cpu against llama.cpp's CPU builds for one quant, every arm interleaved (ab.py --quiet), the layout of
# the CPU baseline: prompts at TPP threads, decode at TTG threads (both engines peak there on an M5 Pro: 18 and 12),
# flash attention on. Results under artifacts/lfm3/cpu/<stamp>-<quant>-<kind>/ (ab.json, ab.md, raw logs).
#
# usage: cpu-matrix.sh QUANT pp|tg [ROUNDS] [DEPTHS]
#   pp: PROMPTS (default 128,512,2048) for TS ggml_cpu, llama.cpp noblas, noblas --repack 0 (llama.cpp with its
#       weights in their GGUF layout, as TensorSharp places them with TS_GGML_CPU_REPACK=0) and, with BLAS_ARM=1, the BLAS build; REPS (default 3) a round
#   tg: tg128 at DEPTHS (default 0,4096) for TS, llama.cpp noblas and noblas --repack 0; 2 reps a round (1 past 8K)
#   MODELS (default ~/work/models/lfm3), TPP / TTG (default 18 / 12), QUIET (ab.py --quiet, default 12),
#   LLAMA_CPP_DIR (build-cpu and build-cpu-noblas under it), PROBE (an A/B arm's Lfm3Probe.dll, see ts-bench.sh),
#   TS_ARMS="name=ENV..." to add TensorSharp arms (e.g. 'ts-new=PROBE=/path/to/bin/Lfm3Probe.dll').
# On a shared machine run it under the CPU lock; a round of pp at 2048 takes ~5 min with four arms.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
Q=${1:?quant}; K=${2:?pp or tg}; ROUNDS=${3:-3}; D=${4:-${DEPTHS:-0,4096}}
M=${MODELS:-$HOME/work/models/lfm3}/LFM3-1.7B-$Q.gguf
TPP=${TPP:-18}; TTG=${TTG:-12}; PP=${PROMPTS:-128,512,2048}; REPS=${REPS:-3}
out=$root/artifacts/lfm3/cpu/$(date +%Y%m%d-%H%M%S)-$Q-$K
mkdir -p "$out"
{ uptime; sysctl -n machdep.cpu.brand_string; git -C "$root" log --oneline -1; } > "$out/conditions.txt"
arms=()
if [[ $K == pp ]]; then
  ctx=$(( ${PP##*,} + 2048 ))
  arms+=("ts-t$TPP=MAX_CONTEXT=$ctx $here/ts-bench.sh $M ggml_cpu $PP 0 0 --reps $REPS --threads $TPP"
         "l-noblas-t$TPP=LLAMA_BUILD=build-cpu-noblas $here/llama-bench-result.sh $M $PP 0 0 $REPS -- -ngl 0 -t $TPP"
         "l-norepack-t$TPP=LLAMA_BUILD=build-cpu-noblas $here/llama-bench-result.sh $M $PP 0 0 $REPS -- -ngl 0 -t $TPP --repack 0")
  [[ ${BLAS_ARM:-0} == 1 ]] && arms+=("l-blas-t$TPP=LLAMA_BUILD=build-cpu $here/llama-bench-result.sh $M $PP 0 0 $REPS -- -ngl 0 -t $TPP")
  threads=$TPP; extra="$PP 0 0 --reps $REPS"
elif [[ $K == tg ]]; then
  ctx=$(( ${D##*,} + 1024 )); reps=2; (( ${D##*,} > 8192 )) && reps=1
  arms+=("ts-t$TTG=MAX_CONTEXT=$ctx $here/ts-bench.sh $M ggml_cpu - 128 $D --reps $reps --threads $TTG"
         "l-noblas-t$TTG=LLAMA_BUILD=build-cpu-noblas $here/llama-bench-result.sh $M - 128 $D $reps -- -ngl 0 -t $TTG"
         "l-norepack-t$TTG=LLAMA_BUILD=build-cpu-noblas $here/llama-bench-result.sh $M - 128 $D $reps -- -ngl 0 -t $TTG --repack 0")
  threads=$TTG; extra="- 128 $D --reps $reps"
else
  echo "kind must be pp or tg" >&2; exit 2
fi
for a in ${TS_ARMS:-}; do
  name=${a%%=*}; env=${a#*=}
  arms+=("$name=$env MAX_CONTEXT=$ctx $here/ts-bench.sh $M ggml_cpu $extra --threads $threads")
done
python3 "$here/ab.py" --rounds "$ROUNDS" --quiet "${QUIET:-12}" --quiet-wait 120 --log "$out/logs" --json "$out/ab.json" \
  "${arms[@]}" | tee "$out/ab.md"
uptime >> "$out/conditions.txt"
echo "results: $out" >&2
