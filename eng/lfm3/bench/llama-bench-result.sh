#!/usr/bin/env bash
# llama-bench with "RESULT name median min max" lines (the names Lfm3Probe bench prints: ppN, tgN@dD), so ab.py can
# interleave it with TensorSharp arms.
#
# usage: llama-bench-result.sh MODEL [PP,..|-] [TG] [DEPTH,..] [REPS] [-- extra llama-bench args]
#   LLAMA_CPP_DIR (default ~/work/llama.cpp-lfm3) selects the build; extra args e.g. "-ngl 0 -t 8" for the CPU.
set -euo pipefail
model=$1; pp=${2:-512,4096}; tg=${3:-128}; depths=${4:-0}; reps=${5:-3}
shift $(( $# < 5 ? $# : 5 )); [[ ${1:-} == "--" ]] && shift
bench=${LLAMA_CPP_DIR:-$HOME/work/llama.cpp-lfm3}/build/bin/llama-bench
args=(-m "$model" -r "$reps" -fa 1 -o jsonl "$@")
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
    print(f"RESULT {name} {statistics.median(s):.2f} {min(s):.2f} {max(s):.2f}")
'
