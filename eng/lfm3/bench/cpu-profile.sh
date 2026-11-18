#!/usr/bin/env bash
# Where the CPU time goes, TensorSharp ggml_cpu vs llama.cpp (build-cpu-noblas), for one quant at T threads:
#   1. per-op time of a 512-row prompt step and of decode steps at depth 0 and 4096: TS_GGML_NODE_PROFILE=shapes on
#      Lfm3Probe bench, llamacpp-op-profile (built here on first use) on llama.cpp;
#   2. macOS `sample` profiles of unmodified runs (tg2048, pp8192), grouped by sample-top.py.
# Results under artifacts/lfm3/cpu/<stamp>-profile-<quant>-t<T>/ with the reduced tables in summary.txt.
#
# usage: cpu-profile.sh [QUANT] [T]      (default Q4_K_M 6); MODELS, LLAMA_CPP_DIR, PROBE as for ts-bench.sh
# CPU-heavy for ~4 minutes: on a shared machine run it under the CPU lock.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
Q=${1:-Q4_K_M}; T=${2:-6}
M=${MODELS:-$HOME/work/models/lfm3}/LFM3-1.7B-$Q.gguf
L=${LLAMA_CPP_DIR:-$HOME/work/llama.cpp-lfm3}
out=$root/artifacts/lfm3/cpu/$(date +%Y%m%d-%H%M%S)-profile-$Q-t$T
mkdir -p "$out"
prof=$root/artifacts/lfm3/cpu/llamacpp-op-profile-noblas
if [[ ! -x $prof ]]; then
  c++ -std=c++17 -O2 -Wno-deprecated-declarations "$here/llamacpp-op-profile.cpp" -I"$L/include" -I"$L/ggml/include" \
    -L"$L/build-cpu-noblas/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$L/build-cpu-noblas/bin" -o "$prof"
fi
{ uptime; git -C "$root" log --oneline -1; } > "$out/conditions.txt"

# 1. per-op
TS_GGML_NODE_PROFILE=shapes TS_GGML_NODE_PROFILE_TAG=prefill TS_GGML_NODE_PROFILE_EVERY=1 MAX_CONTEXT=8192 \
  "$here/ts-bench.sh" "$M" ggml_cpu 512 0 0 --reps 4 --threads "$T" > "$out/ts-prefill.log" 2>&1
for d in 0 4096; do
  TS_GGML_NODE_PROFILE=shapes TS_GGML_NODE_PROFILE_TAG=decode TS_GGML_NODE_PROFILE_EVERY=8 MAX_CONTEXT=8192 \
    "$here/ts-bench.sh" "$M" ggml_cpu - 64 "$d" --reps 1 --threads "$T" > "$out/ts-decode-d$d.log" 2>&1
done
"$prof" "$M" "$T" 1 512 128 > "$out/llama-p512-d128.log" 2>&1
"$prof" "$M" "$T" 1 4096 64 > "$out/llama-p4096-d64.log" 2>&1

# 2. sample (the dotnet host runs the app in its own process, so $! is the process to sample)
sample_run() {  # name delay seconds command...
  local name=$1 delay=$2 secs=$3; shift 3
  "$@" > "$out/$name.run.log" 2>&1 &
  local pid=$!
  sleep "$delay"
  sample "$pid" "$secs" 1 -mayDie -file "$out/$name.sample.txt" > /dev/null 2>&1 || true
  wait "$pid" || true
}
sample_run ts-decode 7 6 env MAX_CONTEXT=4096 "$here/ts-bench.sh" "$M" ggml_cpu - 2048 0 --reps 1 --threads "$T"
sample_run llama-decode 3 6 "$L/build-cpu-noblas/bin/llama-bench" -m "$M" -ngl 0 -t "$T" -fa 1 -p 0 -n 2048 -r 1
sample_run ts-prompt 8 6 env MAX_CONTEXT=10240 "$here/ts-bench.sh" "$M" ggml_cpu 8192 0 0 --reps 1 --threads "$T"
sample_run llama-prompt 4 6 "$L/build-cpu-noblas/bin/llama-bench" -m "$M" -ngl 0 -t "$T" -fa 1 -p 8192 -n 0 -r 1

{
  for f in ts-prefill ts-decode-d0 ts-decode-d4096; do python3 "$here/node-profile-diff.py" "$out/$f.log" || true; done
  for f in llama-p512-d128 llama-p4096-d64; do LC_ALL=C grep -a -E '^unprofiled|^\[op-profile\]|^    ' "$out/$f.log" || true; done
  for f in ts-decode llama-decode ts-prompt llama-prompt; do python3 "$here/sample-top.py" "$out/$f.sample.txt" --top 12 || true; done
} > "$out/summary.txt" 2>&1
uptime >> "$out/conditions.txt"
echo "results: $out (summary.txt)" >&2
