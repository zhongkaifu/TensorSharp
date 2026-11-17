#!/usr/bin/env bash
# TensorSharp's side of llama-bench: Lfm3Probe bench (built Release from this checkout), "RESULT name median min max".
#
# usage: ts-bench.sh MODEL BACKEND [PP,..|-] [TG] [DEPTH,..] [--reps N] [--greedy]
#   MAX_CONTEXT must cover the deepest test (depth + TG); MLX needs TENSORSHARP_MLX_LIBRARY_DIR.
set -euo pipefail
root=$(cd "$(dirname "$0")/../../.." && pwd)
probe=$root/eng/lfm3/Lfm3Probe/bin/Release/net10.0/Lfm3Probe.dll
[[ -f $probe ]] || { echo "build it first: TENSORSHARP_GGML_NO_UPDATE=1 dotnet build $root/eng/lfm3/Lfm3Probe -c Release" >&2; exit 2; }
exec dotnet "$probe" bench "$@"
