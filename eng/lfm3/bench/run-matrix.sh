#!/usr/bin/env bash
# The LFM3 performance matrix: TensorSharp vs llama.cpp on one Mac, every arm interleaved (ab.py), results under
# artifacts/lfm3/bench/<stamp>/ (one markdown table per section, raw logs beside it).
#
# usage: run-matrix.sh [metal] [mlx] [cpu] [serve] [ttft]      (default: all five)
#   MODELS     model directory (default ~/work/models/lfm3)
#   ROUNDS     interleaved rounds per section (default 3); REPS repeats inside a run (default 3)
#   QUANTS     GPU quants (default "Q4_K_M Q8_0 QAD-Q4_0 BF16"); CPU_QUANTS (default "Q4_K_M Q8_0")
#   PP         Metal prompt lengths (default 512,4096,16384,32768); MLX_PP (default 512,4096)
#   DEPTHS     decode depths (default 0,4096,16384,32768); CPU_THREADS (default 6)
#   TTFT_LENGTHS  server time-to-first-token prompts (default 16,128,512,2048,4096,16384,32768)
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
PP=${PP:-512,4096,16384,32768}
MLX_PP=${MLX_PP:-512,4096}
DEPTHS=${DEPTHS:-0,4096,16384,32768}
CPU_THREADS=${CPU_THREADS:-6}
TTFT_LENGTHS=${TTFT_LENGTHS:-16,128,512,2048,4096,16384,32768}
export TENSORSHARP_MLX_LIBRARY_DIR=${TENSORSHARP_MLX_LIBRARY_DIR:-$root/InferenceWeb.Tests/bin/Release/net10.0}
stamp=$(date +%Y%m%d-%H%M%S)
out=$root/artifacts/lfm3/bench/$stamp
mkdir -p "$out"
sections=${*:-metal mlx cpu serve ttft}
deepest=${DEPTHS##*,}
longest=${PP##*,}
ctx=$(( (deepest > longest ? deepest : longest) + 1024 ))
echo "results: $out" >&2
{ uptime; sysctl -n machdep.cpu.brand_string; git -C "$root" log --oneline -1; } > "$out/conditions.txt"

for s in $sections; do
  case $s in
    metal)
      for q in $QUANTS; do
        python3 "$here/ab.py" --rounds "$ROUNDS" --log "$out/metal-$q" \
          "ts=MAX_CONTEXT=$ctx $here/ts-bench.sh $M/LFM3-1.7B-$q.gguf ggml_metal $PP 128 $DEPTHS --reps $REPS" \
          "llama=$here/llama-bench-result.sh $M/LFM3-1.7B-$q.gguf $PP 128 $DEPTHS $REPS" | tee "$out/metal-$q.md"
      done ;;
    mlx)
      for q in $QUANTS; do
        python3 "$here/ab.py" --rounds "$ROUNDS" --log "$out/mlx-$q" \
          "ts-mlx=MAX_CONTEXT=8192 $here/ts-bench.sh $M/LFM3-1.7B-$q.gguf mlx $MLX_PP 128 0,4096 --reps $REPS" \
          "ts-metal=MAX_CONTEXT=8192 $here/ts-bench.sh $M/LFM3-1.7B-$q.gguf ggml_metal $MLX_PP 128 0,4096 --reps $REPS" \
          "llama=$here/llama-bench-result.sh $M/LFM3-1.7B-$q.gguf $MLX_PP 128 0,4096 $REPS" | tee "$out/mlx-$q.md"
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
    ttft)
      # Both servers up at once, requests interleaved (ttft.py): one per prompt length and server per round.
      longest_ttft=${TTFT_LENGTHS##*,}
      tctx=$(( longest_ttft + 2048 ))
      tsbin=$out/ttft-server-bin
      rm -rf "$tsbin"; cp -R "$root/TensorSharp.Server.Host/bin" "$tsbin"
      ( cd "$root" && TS_SERVER_BIN="$tsbin" MAX_CONTEXT=$tctx TENSORSHARP_MODELS="$(dirname "$M")" \
          "$root/eng/lfm3/e2e/ts-server.sh" 18091 "$out/ttft-ts.log" --model "$M/LFM3-1.7B-Q4_K_M.gguf" --no-webui \
          --sampling-precedence request ) &
      tspid=$!
      "${LLAMA_CPP_DIR:-$HOME/work/llama.cpp-lfm3}/build/bin/llama-server" -m "$M/LFM3-1.7B-Q4_K_M.gguf" --port 18092 -np 1 \
          -c $tctx -fa on --jinja -ngl 99 --no-webui -b 2048 -ub 512 > "$out/ttft-llama.log" 2>&1 &
      llpid=$!
      for _ in $(seq 1 300); do
        curl -sf -o /dev/null http://127.0.0.1:18091/v1/models && curl -sf -o /dev/null http://127.0.0.1:18092/v1/models && break
        sleep 1
      done
      python3 "$here/ttft.py" --server ts=http://127.0.0.1:18091 --server llama=http://127.0.0.1:18092 \
          --tokenizer-url http://127.0.0.1:18092 --lengths "$TTFT_LENGTHS" --reps "$ROUNDS" --out "$out/ttft.jsonl" \
          2>/dev/null | tee "$out/ttft.md" || true
      pkill -P $tspid 2>/dev/null || true; kill $tspid $llpid 2>/dev/null || true; wait 2>/dev/null || true
      rm -rf "$tsbin" ;;
    *) echo "unknown section $s" >&2; exit 2 ;;
  esac
done
uptime >> "$out/conditions.txt"
echo "results: $out" >&2
