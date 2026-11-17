#!/usr/bin/env bash
# The LFM3 performance matrix: TensorSharp vs llama.cpp on one Mac, every arm interleaved (ab.py), results under
# artifacts/lfm3/bench/<stamp>/ (one markdown table per section, raw logs beside it).
#
# usage: run-matrix.sh [metal] [mlx] [cpu] [serve]      (default: all four)
#   MODELS     model directory (default ~/work/models/lfm3)
#   ROUNDS     interleaved rounds per section (default 3); REPS repeats inside a run (default 3)
#   QUANTS     GPU quants (default "Q4_K_M Q8_0 QAD-Q4_0 BF16"); CPU_QUANTS (default "Q4_K_M Q8_0")
#   DEPTHS     decode depths (default 0,4096,16384,32768); CPU_THREADS (default 6)
# Build first (Release): TENSORSHARP_GGML_NO_UPDATE=1 dotnet build eng/lfm3/Lfm3Probe -c Release and
# dotnet build TensorSharp.Server.Host -c Release; MLX needs TENSORSHARP_MLX_LIBRARY_DIR (defaults to the test output).
# The machine must be quiet: check `uptime` and that no other GPU job runs; the tables carry each arm's min-max spread.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
M=${MODELS:-$HOME/work/models/lfm3}
ROUNDS=${ROUNDS:-3}
REPS=${REPS:-3}
QUANTS=${QUANTS:-"Q4_K_M Q8_0 QAD-Q4_0 BF16"}
CPU_QUANTS=${CPU_QUANTS:-"Q4_K_M Q8_0"}
DEPTHS=${DEPTHS:-0,4096,16384,32768}
CPU_THREADS=${CPU_THREADS:-6}
export TENSORSHARP_MLX_LIBRARY_DIR=${TENSORSHARP_MLX_LIBRARY_DIR:-$root/InferenceWeb.Tests/bin/Release/net10.0}
stamp=$(date +%Y%m%d-%H%M%S)
out=$root/artifacts/lfm3/bench/$stamp
mkdir -p "$out"
sections=${*:-metal mlx cpu serve}
deepest=${DEPTHS##*,}
ctx=$(( deepest + 1024 ))
echo "results: $out" >&2
{ uptime; sysctl -n machdep.cpu.brand_string; git -C "$root" log --oneline -1; } > "$out/conditions.txt"

for s in $sections; do
  case $s in
    metal)
      for q in $QUANTS; do
        python3 "$here/ab.py" --rounds "$ROUNDS" --log "$out/metal-$q" \
          "ts=MAX_CONTEXT=$ctx $here/ts-bench.sh $M/LFM3-1.7B-$q.gguf ggml_metal 512,4096 128 $DEPTHS --reps $REPS" \
          "llama=$here/llama-bench-result.sh $M/LFM3-1.7B-$q.gguf 512,4096 128 $DEPTHS $REPS" | tee "$out/metal-$q.md"
      done ;;
    mlx)
      for q in $QUANTS; do
        python3 "$here/ab.py" --rounds "$ROUNDS" --log "$out/mlx-$q" \
          "ts-mlx=MAX_CONTEXT=8192 $here/ts-bench.sh $M/LFM3-1.7B-$q.gguf mlx 512,4096 128 0,4096 --reps $REPS" \
          "ts-metal=MAX_CONTEXT=8192 $here/ts-bench.sh $M/LFM3-1.7B-$q.gguf ggml_metal 512,4096 128 0,4096 --reps $REPS" \
          "llama=$here/llama-bench-result.sh $M/LFM3-1.7B-$q.gguf 512,4096 128 0,4096 $REPS" | tee "$out/mlx-$q.md"
      done ;;
    cpu)
      for q in $CPU_QUANTS; do
        python3 "$here/ab.py" --rounds "$ROUNDS" --log "$out/cpu-$q" \
          "ts-ggml_cpu=TS_GGML_CPU_THREADS=$CPU_THREADS MAX_CONTEXT=4096 $here/ts-bench.sh $M/LFM3-1.7B-$q.gguf ggml_cpu 512 64 0,1024 --reps $REPS" \
          "ts-managed=MAX_CONTEXT=4096 $here/ts-bench.sh $M/LFM3-1.7B-$q.gguf cpu 512 64 0,1024 --reps $REPS" \
          "llama-cpu=$here/llama-bench-result.sh $M/LFM3-1.7B-$q.gguf 512 64 0,1024 $REPS -- -ngl 0 -t $CPU_THREADS" | tee "$out/cpu-$q.md"
      done ;;
    serve)
      for e in llama ts; do
        python3 "$here/serve-bench.py" --engine "$e" --model "$M/LFM3-1.7B-Q4_K_M.gguf" --config "$root/config/lfm3-1.7b.json" \
          --concurrency 1,4,16 --prompts 128,1024,4096 --gen 256 --out "$out/serve" | tee "$out/serve-$e.md"
      done ;;
    *) echo "unknown section $s" >&2; exit 2 ;;
  esac
done
uptime >> "$out/conditions.txt"
echo "results: $out" >&2
