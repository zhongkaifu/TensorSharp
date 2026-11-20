#!/usr/bin/env bash
# TensorSharp's side of llama-bench: Lfm3Probe bench (built Release from this checkout), "RESULT name median min max".
#
# usage: ts-bench.sh MODEL BACKEND [PP,..|-] [TG] [DEPTH,..] [--reps N] [--greedy] [--threads N]
#   MAX_CONTEXT must cover the deepest test (depth + TG); MLX needs TENSORSHARP_MLX_LIBRARY_DIR.
#   --threads N runs the CPU lanes on N threads, as llama-bench -t N does: TS_GGML_CPU_THREADS (ggml_cpu's prompt and
#   decode pools; defaults every physical core for prompts and, on a hybrid Mac, every physical core but one for prompts
#   and every physical core but two for steps of at most 4 rows), TS_CPU_THREADS (the managed worker pool; default every core on Apple silicon, half the logical
#   processors above 8 elsewhere; on a hybrid Mac every core but two for steps of at most 4 rows, TS_CPU_DECODE_THREADS,
#   which follows TS_CPU_THREADS when only that is set) and DOTNET_PROCESSOR_COUNT (Environment.ProcessorCount: the .NET
#   thread pool and the managed kernels that size their loops from it) all become N. Without it each lane keeps its own
#   defaults.
#   PROBE overrides the Lfm3Probe.dll (an A/B arm's copied bin directory).
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
probe=${PROBE:-$root/eng/lfm3/Lfm3Probe/bin/Release/net10.0/Lfm3Probe.dll}
[[ -f $probe ]] || { echo "build it first: TENSORSHARP_GGML_NO_UPDATE=1 dotnet build $root/eng/lfm3/Lfm3Probe -c Release" >&2; exit 2; }
args=()
while (( $# )); do
  if [[ $1 == --threads ]]; then
    export TS_GGML_CPU_THREADS=$2 TS_CPU_THREADS=$2 DOTNET_PROCESSOR_COUNT=$2
    shift 2
  else
    args+=("$1"); shift
  fi
done
echo "INFO threads=${TS_GGML_CPU_THREADS:-default} probe=$probe" >&2
exec dotnet "$probe" bench "${args[@]}"
