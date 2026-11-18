#!/usr/bin/env bash
# The LFM3 performance matrix: TensorSharp vs llama.cpp on one Mac, every arm interleaved (ab.py), results under
# artifacts/lfm3/bench/<stamp>/ (one markdown table per section, raw logs beside it).
#
# usage: run-matrix.sh [metal] [mlx] [cpu] [serve] [ttft] [cpu-serve] [cpu-omni]      (default: metal mlx cpu serve ttft)
#   MODELS     model directory (default ~/work/models/lfm3)
#   ROUNDS     interleaved rounds per section (default 3); REPS repeats inside a run (default 3)
#   QUANTS     GPU quants (default "Q4_K_M Q8_0 QAD-Q4_0 BF16"); DEPTHS GPU decode depths (default 0,4096,16384,32768)
#   PP         Metal prompt lengths (default 512,4096,16384,32768); MLX_PP (default 512,4096)
#   TTFT_LENGTHS  ttft: server time-to-first-token prompts on Metal (default 16,128,512,2048,4096,16384,32768)
#   cpu: both TensorSharp CPU lanes (ggml_cpu, the managed cpu) against llama-bench -ngl 0 from a CPU-only llama.cpp
#     tree, with flash attention on and ab.py --quiet 12 (each arm waits for a quiet machine):
#     CPU_QUANTS        (default "Q4_K_M Q8_0 QAD-Q4_0 NVFP4 BF16")
#     CPU_PP            prompt sizes (default 128,512,2048,8192; BF16 CPU_PP_BF16 default 128,512,2048)
#     CPU_PROMPT_THREADS  every engine's prompt threads (default hw.physicalcpu: both TensorSharp lanes' prompt default
#                       and llama.cpp's best prompt count on an M5 Pro)
#     CPU_DEPTHS        decode depths (default 0,4096; BF16 0); CPU_DEEP_QUANTS / CPU_DEEP_DEPTH one more decode run
#                       per listed quant at that depth (default Q4_K_M / 16384, one rep)
#     CPU_TG_REPS       decode repeats inside a run (default 5: odd, so each side's median is one of its runs; with 2,
#                       the median is the mean of two, and a run that is slow for its own reasons moves it by half)
#     CPU_THREADS       an equal-thread decode count (default 12, the CPU baseline's): TensorSharp decodes at its
#                       defaults and at CPU_THREADS, llama.cpp at CPU_THREADS and at CPU_WIDE_THREADS (default
#                       hw.physicalcpu - 2: ggml_cpu's decode-pool default on a hybrid Mac, and llama.cpp's best decode
#                       count on an M5 Pro: 16-17 threads beat 12 by 4-12 % (BF16 and the 16K depth by 27 %) and 14
#                       by 2-3 %, and 18 is about 8 % slower than 16)
#     LLAMA_CPU_BUILD   llama.cpp's Accelerate BLAS CPU build (default build-cpu), LLAMA_CPU_NOBLAS its ggml-cpu-alone
#                       build (default build-cpu-noblas; never the Metal build, which can still run ops on the GPU)
#     BASE_PROBE        optional: another build's Lfm3Probe.dll, added as before arms (prompt pp512,pp2048 at
#                       CPU_PROMPT_THREADS; decode at CPU_THREADS; the managed lane one rep of pp512 / tg128@d0)
#   cpu-serve: TTFT and streams on the CPU (Q4_K_M, 512-token prompts, 128 out, concurrency 1 and 4): llama-server
#     -t CPU_WIDE_THREADS -tb CPU_PROMPT_THREADS, then TensorSharp.Server.Host on ggml_cpu and on cpu at their defaults.
#   cpu-omni: LFM3-Omni-2B Q8_0 + mmproj Q8_0 on the CPU: vision encode (OmniInProbe encode vs llama-mtmd-cli at
#     CPU_THREADS and CPU_PROMPT_THREADS; IMAGE, default a 512 x 512 resize of a test fixture: the encode time depends
#     on the size only), audio encode (Lfm3AudioProbe compare against LFM3_AUDIO_REF = lfm3-audio-ref built on the CPU
#     tree, eng/lfm3/audio-in/README.md; AUDIO_CLIPS .f32 files) and streaming speech (eng/lfm3/speech-bench.py, three
#     servers; llama-server -t CPU_WIDE_THREADS -tb CPU_PROMPT_THREADS).
# Build first (Release): TENSORSHARP_GGML_NO_UPDATE=1 dotnet build eng/lfm3/Lfm3Probe -c Release and
# dotnet build TensorSharp.Server.Host -c Release (cpu-omni: also eng/lfm3/omni-in/OmniInProbe and
# eng/lfm3/audio-in/Lfm3AudioProbe); MLX needs TENSORSHARP_MLX_LIBRARY_DIR (defaults to the test output).
# The machine must be quiet: check `uptime` and that no other GPU job or CPU-heavy process runs; the tables carry each
# arm's min-max spread.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
M=${MODELS:-$HOME/work/models/lfm3}
ROUNDS=${ROUNDS:-3}
REPS=${REPS:-3}
QUANTS=${QUANTS:-"Q4_K_M Q8_0 QAD-Q4_0 BF16"}
PP=${PP:-512,4096,16384,32768}
MLX_PP=${MLX_PP:-512,4096}
DEPTHS=${DEPTHS:-0,4096,16384,32768}
TTFT_LENGTHS=${TTFT_LENGTHS:-16,128,512,2048,4096,16384,32768}
CPU_QUANTS=${CPU_QUANTS:-"Q4_K_M Q8_0 QAD-Q4_0 NVFP4 BF16"}
CPU_PP=${CPU_PP:-128,512,2048,8192}
CPU_PP_BF16=${CPU_PP_BF16:-128,512,2048}
CPU_DEPTHS=${CPU_DEPTHS:-0,4096}
CPU_DEEP_QUANTS=${CPU_DEEP_QUANTS-Q4_K_M}
CPU_DEEP_DEPTH=${CPU_DEEP_DEPTH:-16384}
CPU_TG_REPS=${CPU_TG_REPS:-5}
physical=$(sysctl -n hw.physicalcpu 2>/dev/null || nproc)
CPU_PROMPT_THREADS=${CPU_PROMPT_THREADS:-$physical}
CPU_THREADS=${CPU_THREADS:-12}
CPU_WIDE_THREADS=${CPU_WIDE_THREADS:-$(( physical > 4 ? physical - 2 : physical ))}
LLAMA_CPU_BUILD=${LLAMA_CPU_BUILD:-build-cpu}
LLAMA_CPU_NOBLAS=${LLAMA_CPU_NOBLAS:-build-cpu-noblas}
export TENSORSHARP_MLX_LIBRARY_DIR=${TENSORSHARP_MLX_LIBRARY_DIR:-$root/InferenceWeb.Tests/bin/Release/net10.0}
stamp=$(date +%Y%m%d-%H%M%S)
out=$root/artifacts/lfm3/bench/$stamp
mkdir -p "$out"
sections=${*:-metal mlx cpu serve ttft}
deepest=${DEPTHS##*,}
longest=${PP##*,}
ctx=$(( (deepest > longest ? deepest : longest) + 1024 ))
echo "results: $out" >&2
{ uptime; sysctl -n machdep.cpu.brand_string; git -C "$root" log --oneline -1; ps -Ao pcpu,command | sort -rn | head -5 | cut -c1-160 || true; } \
  > "$out/conditions.txt"

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
        g=$M/LFM3-1.7B-$q.gguf
        pp=$CPU_PP; d=$CPU_DEPTHS; [[ $q == BF16 ]] && { pp=$CPU_PP_BF16; d=0; }
        pt=$CPU_PROMPT_THREADS; t=$CPU_THREADS
        arms=("ts-ggml_cpu=MAX_CONTEXT=$(( ${pp##*,} + 512 )) $here/ts-bench.sh $g ggml_cpu $pp 0 0 --reps $REPS"
              "ts-cpu=MAX_CONTEXT=$(( ${pp##*,} + 512 )) $here/ts-bench.sh $g cpu $pp 0 0 --reps $REPS"
              "llama-blas=LLAMA_BUILD=$LLAMA_CPU_BUILD $here/llama-bench-result.sh $g $pp 0 0 $REPS -- -ngl 0 -t $pt")
        # ggml-cpu's BF16 product is a scalar loop on ARM (minutes a prompt); the BLAS build is llama.cpp's BF16 path.
        [[ $q != BF16 ]] && arms+=("llama-noblas=LLAMA_BUILD=$LLAMA_CPU_NOBLAS $here/llama-bench-result.sh $g $pp 0 0 $REPS -- -ngl 0 -t $pt")
        # (a BF16 before arm on ggml_cpu would be ggml-cpu's scalar product again: minutes a round)
        [[ -n ${BASE_PROBE:-} && $q != BF16 ]] && arms+=(
          "ts-ggml_cpu-base=PROBE=$BASE_PROBE MAX_CONTEXT=2560 $here/ts-bench.sh $g ggml_cpu 512,2048 0 0 --reps 2 --threads $pt")
        [[ -n ${BASE_PROBE:-} ]] && arms+=(
          "ts-cpu-base=PROBE=$BASE_PROBE MAX_CONTEXT=1024 $here/ts-bench.sh $g cpu 512 0 0 --reps 1 --threads $pt")
        python3 "$here/ab.py" --rounds "$ROUNDS" --quiet 12 --log "$out/cpu-$q-pp" --json "$out/cpu-$q-pp.json" "${arms[@]}" \
          | tee "$out/cpu-$q-pp.md"
        runs=("$d:$CPU_TG_REPS"); [[ " $CPU_DEEP_QUANTS " == *" $q "* ]] && runs+=("$CPU_DEEP_DEPTH:1")
        for run in "${runs[@]}"; do
          dd=${run%%:*}; reps=${run##*:}; c=$(( ${dd##*,} + 1024 ))
          arms=("ts-ggml_cpu=MAX_CONTEXT=$c $here/ts-bench.sh $g ggml_cpu - 128 $dd --reps $reps"
                "ts-ggml_cpu-t$t=MAX_CONTEXT=$c $here/ts-bench.sh $g ggml_cpu - 128 $dd --reps $reps --threads $t"
                "ts-cpu=MAX_CONTEXT=$c $here/ts-bench.sh $g cpu - 128 $dd --reps $reps"
                "ts-cpu-t$t=MAX_CONTEXT=$c $here/ts-bench.sh $g cpu - 128 $dd --reps $reps --threads $t"
                "llama-t$t=LLAMA_BUILD=$LLAMA_CPU_NOBLAS $here/llama-bench-result.sh $g - 128 $dd $reps -- -ngl 0 -t $t"
                "llama-t$CPU_WIDE_THREADS=LLAMA_BUILD=$LLAMA_CPU_NOBLAS $here/llama-bench-result.sh $g - 128 $dd $reps -- -ngl 0 -t $CPU_WIDE_THREADS")
          [[ -n ${BASE_PROBE:-} && $dd == "$d" ]] && arms+=(
            "ts-ggml_cpu-base=PROBE=$BASE_PROBE MAX_CONTEXT=$c $here/ts-bench.sh $g ggml_cpu - 128 $dd --reps $reps --threads $t"
            "ts-cpu-base=PROBE=$BASE_PROBE MAX_CONTEXT=1024 $here/ts-bench.sh $g cpu - 128 0 --reps 1 --threads $pt")
          python3 "$here/ab.py" --rounds "$ROUNDS" --quiet 12 --log "$out/cpu-$q-tg-d${dd//,/-}" --json "$out/cpu-$q-tg-d${dd//,/-}.json" \
            "${arms[@]}" | tee "$out/cpu-$q-tg-d${dd//,/-}.md"
        done
      done ;;
    cpu-serve)
      for e in llama ts-ggml_cpu ts-cpu; do
        case $e in
          llama) a=(--engine llama --backend cpu --threads "$CPU_WIDE_THREADS" --threads-batch "$CPU_PROMPT_THREADS") ;;
          *) a=(--engine ts --backend "${e#ts-}") ;;
        esac
        LLAMA_BUILD=$LLAMA_CPU_BUILD python3 "$here/serve-bench.py" "${a[@]}" --tag "$e" --model "$M/LFM3-1.7B-Q4_K_M.gguf" \
          --config "$root/config/lfm3-1.7b.json" --concurrency 1,4 --prompts 512 --gen 128 --out "$out/cpu-serve" \
          | tee "$out/cpu-serve-$e.md"
      done ;;
    cpu-omni)
      lm=$M/LFM3-Omni-2B-Q8_0.gguf; mm=$M/mmproj-LFM3-Omni-2B-Q8_0.gguf
      lb=${LLAMA_CPP_DIR:-$HOME/work/llama.cpp-lfm3}/$LLAMA_CPU_BUILD/bin
      img=${IMAGE:-}
      if [[ -z $img ]]; then
        img=$out/omni-512.png
        sips -z 512 512 "$root/InferenceWeb.Tests/Fixtures/MuseSpark/vision-chart-a.png" --out "$img" >/dev/null
      fi
      probe=$root/eng/lfm3/omni-in/OmniInProbe/bin/Release/net10.0/OmniInProbe.dll
      for lane in ggml_cpu cpu; do
        for tt in default $CPU_THREADS; do
          e=(); [[ $tt != default ]] && e=(TS_GGML_CPU_THREADS=$tt TS_CPU_THREADS=$tt DOTNET_PROCESSOR_COUNT=$tt)
          echo "vision ts-$lane-$tt: $(env ${e[@]+"${e[@]}"} dotnet "$probe" encode "$lm" "$mm" $lane "$img" 5 2>&1 | grep patches | tail -1)"
        done
      done | tee "$out/cpu-omni-vision.txt"
      for tt in $CPU_THREADS $CPU_PROMPT_THREADS; do
        for rep in 1 2 3 4 5; do
          echo "vision llama-t$tt rep $rep: $("$lb/llama-mtmd-cli" -m "$lm" --mmproj "$mm" --image "$img" -p Describe. -n 1 --temp 0 -t "$tt" 2>&1 \
            | grep -o -E 'batch encoding done in [0-9]+ ms')"
        done
      done | tee -a "$out/cpu-omni-vision.txt"
      if [[ -n ${LFM3_AUDIO_REF:-} && -n ${AUDIO_CLIPS:-} ]]; then
        aprobe=$root/eng/lfm3/audio-in/Lfm3AudioProbe/bin/Release/net10.0/Lfm3AudioProbe.dll
        for tt in $CPU_THREADS $CPU_PROMPT_THREADS; do
          LFM3_REF_THREADS=$tt "$LFM3_AUDIO_REF" dump "$lm" "$mm" "$out/audio-ref-t$tt" 5 $AUDIO_CLIPS > "$out/audio-ref-t$tt.log" 2>&1
        done
        for lane in ggml_cpu cpu; do
          echo "audio ts-$lane (llama.cpp at $CPU_THREADS threads):"
          (cd "$root" && dotnet "$aprobe" compare "$mm" "$out/audio-ref-t$CPU_THREADS" $lane 5 "$out/audio-cmp-$lane.json" 2>&1 | grep -E "time:")
        done | tee "$out/cpu-omni-audio.txt"
        grep -H -E 'samples, .* encode' "$out"/audio-ref-t*.log | tee -a "$out/cpu-omni-audio.txt"
      else
        echo "cpu-omni: audio skipped (set LFM3_AUDIO_REF and AUDIO_CLIPS)" >&2
      fi
      pids=()
      "$lb/llama-server" -m "$lm" -mm "$mm" --port 18391 -ngl 0 --jinja -c 8192 -np 1 -t "$CPU_WIDE_THREADS" -tb "$CPU_PROMPT_THREADS" \
        > "$out/tts-llama.log" 2>&1 & pids+=($!)
      sdll=${TS_SERVER_DLL:-$root/TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll}
      for p in "18392 ggml_cpu" "18393 cpu"; do set -- $p
        mkdir -p "$out/studio-$2"
        TS_STUDIO_DIRECTORY=$out/studio-$2 MAX_CONTEXT=8192 dotnet "$sdll" --model "$lm" --mmproj "$mm" --backend $2 --port $1 \
          --no-skills --no-multi-agent --no-webui > "$out/tts-ts-$2.log" 2>&1 & pids+=($!)
      done
      if python3 "$root/eng/lfm3/omni-in/wait_ready.py" http://127.0.0.1:18391/health http://127.0.0.1:18392/v1/models \
          http://127.0.0.1:18393/v1/models; then
        python3 -I "$root/eng/lfm3/speech-bench.py" "$out/cpu-omni-speech.json" 2 ts-ggml_cpu=http://127.0.0.1:18392 \
          llama-cpu=http://127.0.0.1:18391 ts-cpu=http://127.0.0.1:18393 | tee "$out/cpu-omni-speech.txt"
      else
        echo "cpu-omni: a speech server did not start (logs in $out)" >&2
      fi
      kill "${pids[@]}" 2>/dev/null || true; wait 2>/dev/null || true ;;
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
