#!/usr/bin/env bash
# llama-bench with "RESULT name median min max" lines (the names Lfm3Probe bench prints: ppN, tgN@dD), so ab.py can
# interleave it with TensorSharp arms. An "INFO" line per test records the build's backends, threads and flash attention.
#
# usage: llama-bench-result.sh MODEL [PP,..|-] [TG] [DEPTH,..] [REPS] [-- extra llama-bench args]
#   LLAMA_CPP_DIR (default ~/work/llama.cpp-lfm3) and LLAMA_BUILD (its build directory, default build: Metal + BLAS;
#   build-cpu: CPU + Accelerate BLAS; build-cpu-noblas: ggml-cpu alone; build-cuda / build-vk on a Linux GPU machine)
#   select the binary; FA (default 1) is -fa.
#   Extra args e.g. "-ngl 0 -t 8" for the CPU (one value each: the RESULT names carry no thread count).
set -euo pipefail
model=$1; pp=${2:-512,4096}; tg=${3:-128}; depths=${4:-0}; reps=${5:-3}
shift $(( $# < 5 ? $# : 5 )); [[ ${1:-} == "--" ]] && shift
bench=${LLAMA_CPP_DIR:-$HOME/work/llama.cpp-lfm3}/${LLAMA_BUILD:-build}/bin/llama-bench
[[ -x $bench ]] || { echo "no llama-bench at $bench" >&2; exit 2; }
args=(-m "$model" -r "$reps" -fa "${FA:-1}" -o jsonl "$@")
{
  if [[ $pp != "-" ]]; then "$bench" "${args[@]}" -p "$pp" -n 0; fi
  if [[ $tg != "0" ]]; then "$bench" "${args[@]}" -p 0 -n "$tg" -d "$depths"; fi
} 2>/dev/null | python3 -c '
import json, statistics, sys
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("{"): continue
    d = json.loads(line)
    s = d["samples_ts"]
    name = "pp%d" % d["n_prompt"] if d["n_gen"] == 0 else "tg%d@d%d" % (d["n_gen"], d["n_depth"])
    info = " ".join("%s=%s" % (k, d.get(k)) for k in ("backends", "n_threads", "flash_attn", "n_batch", "n_ubatch", "type_k"))
    print(f"INFO {name} {info} samples={s}")
    print(f"RESULT {name} {statistics.median(s):.2f} {min(s):.2f} {max(s):.2f}")
'
