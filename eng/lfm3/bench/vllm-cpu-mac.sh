#!/usr/bin/env bash
# LFM3-1.7B on an Apple-silicon Mac's CPU: vLLM's CPU backend vs TensorSharp (ggml_cpu and the managed cpu lane) vs
# llama.cpp's CPU build, with one client for every engine (`vllm bench serve --backend openai-chat`).
#
#   eng/lfm3/bench/vllm-cpu-mac.sh [check] [correct] [sweep] [bench] [stock] [summary]      (no argument: all six)
#
#   check    the venv's vLLM imports, vllm._C carries its CPU ops, the model files and binaries exist; conditions.txt
#   correct  greedy (temperature 0) generations of five chat prompts (HF chat template, thinking off, 64 tokens):
#            vLLM offline (each of VLLM_DTYPES), HF transformers float32 (generate(), and the reference: greedy by
#            full forwards without a cache, --recompute), and the servers (vLLM,
#            TensorSharp ggml_cpu and cpu, llama.cpp) on the BF16 weights; token identity / first divergence ->
#            $OUT/correct/compare.md, and the reference's logit gap at each first divergence ->
#            $OUT/correct/margins.md (near tie or bug). Three lock holds (offline, servers, margins).
#   sweep    vLLM's thread count: SWEEP_ARM at each of SWEEP_THREADS (+ SWEEP_EXTRA arms) over the concurrency-1 cells,
#            one lock hold; sets VLLM_THREADS (best mean output tok/s) unless it is given
#   bench    ROUNDS rounds of ARMS; each arm starts its server per concurrency (slots = concurrency), warms it, runs
#            its cells (ISLxOSL@concurrency) with the one client, stops it. ARMS is a list of groups (spaces); a
#            group's arms (joined by +) share one lock hold; the group order and the order inside a group rotate
#            each round
#   stock    stock vLLM (vllm-fp16, no workaround), one reduced round: STOCK_CELLS with STOCK_REQUESTS
#   summary  medians over rounds with min-max spreads -> $OUT/bench/summary.md
#
# Arms (all: prefix/prompt caching off, max context CTX per sequence, greedy with no repetition penalty, ignore_eos,
# thinking off):
#   vllm-{fp16,bf16,fp32}[fix][-tN]  vllm serve HF --dtype .. --max-model-len CTX --no-enable-prefix-caching
#          --max-num-seqs C; VLLM_CPU_OMP_THREADS_BIND=0-(N-1) (the CPU worker's OpenMP width; macOS has no
#          affinity, only the count matters; N defaults to VLLM_THREADS), VLLM_CPU_KVCACHE_SPACE=2 (GiB).
#          `fix` loads vllm-mac-shortconv/sitecustomize.py: vLLM 0.34's Arm short-conv path is a grouped F.conv1d
#          that PyTorch's macOS wheels (no oneDNN) run 2048 groups at a time (~1.5 s per decode step); the
#          workaround routes it to vLLM's own causal_conv1d_update_cpu_vec kernel (bit-identical outputs).
#   ts-{ggml,cpu}-{bf16,q4km}[-tN]  TensorSharp.Server.Host (this checkout's Release build) --backend ggml_cpu|cpu
#          --no-prefix-cache, TS_SCHED_MAX_RUNNING_SEQS=C; TS_GGML_CPU_THREADS=TS_GGML_T / TS_CPU_THREADS=TS_CPU_T (or N)
#          when that count is above 0, else the lane's own thread defaults
#   llama-{bf16,q4km}  llama-server -ngl 0 -np C -cb -fa on -t LLAMA_T -tb LLAMA_TB --no-cache-prompt -cram 0
#
# Environment (defaults in brackets):
#   OUT [artifacts/lfm3/cpu/vllm/<stamp>]  results (never commit them); reuse one OUT across steps
#   HEAVY [none]  a lock wrapper run as `$HEAVY <label> <command...>` around every CPU-heavy job on a shared machine
#   VLLM_VENV [~/work/venvs/vllm-cpu]  vLLM built from source for macOS CPU (uv venv -p 3.12; uv pip install -r
#                 requirements/cpu.txt; uv pip install -e .)
#   HF [~/work/models/lfm3/hf-1.7b-full]  LiquidAI/LFM3-1.7B (model.safetensors, tokenizer, chat_template.jinja)
#   GGUF_DIR [~/work/models/lfm3]  LFM3-1.7B-BF16.gguf, LFM3-1.7B-Q4_K_M.gguf
#   LLAMA_BIN [~/work/llama.cpp-lfm3/build-cpu/bin]
#   ROUNDS [3]  CELLS [512x128@1,128x256@1,512x128@4,128x256@4]  CTX [4096]
#   ARMS [vllm-fp16fix+vllm-fp32fix+ts-ggml-q4km+llama-q4km llama-bf16+ts-ggml-bf16 ts-cpu-q4km@1+ts-cpu-bf16@1
#         ts-cpu-q4km@4+ts-cpu-bf16@4+ts-ggml-q4km-t18]   (ARM@C: only the cells at concurrency C)
#   STOCK_CELLS [512x128@1,512x128@4]  STOCK_REQUESTS [1:2,4:4]
#   VLLM_DTYPES [float16 bfloat16 float32]  (correct; FIX= runs them without the workaround)
#   VLLM_BENCH_DTYPE [float16]  (the vLLM server in `correct`)
#   SWEEP_ARM [vllm-fp16fix]  SWEEP_THREADS [6 12 18]  SWEEP_EXTRA [vllm-bf16fix vllm-fp32fix]  VLLM_THREADS [sweep]
#   TS_GGML_T [0] TS_CPU_T [0] LLAMA_T [16] LLAMA_TB [18]  threads. 0 keeps TensorSharp's own defaults, which already
#                 split prompts from decode (ggml_cpu: prompts on every physical core, decode on every physical core but
#                 two on a hybrid Mac; the managed cpu lane: every core); N pins every step of the lane to N
#                 (TS_GGML_CPU_THREADS / TS_CPU_THREADS, as ts-bench.sh --threads does). llama.cpp takes -t and -tb: on
#                 a 6P+12E M5 Pro it decodes best at 16 and prompts best at 18 (bench/run-matrix.sh cpu)
#   REQUESTS [1:4,4:8]  requests per cell by concurrency
#   SLOW_ARMS [llama-bf16 vllm-fp16]  SLOW_REQUESTS [1:3,4:4]  SLOW_CELLS [$CELLS]
#                 (llama.cpp's BF16 decode, a scalar dot on ARM, and stock vLLM; e.g. SLOW_CELLS=512x128@1,128x256@1
#                 when a group would hold the lock past ~20 min)
#   HOLD_GAP [0]  seconds to wait between lock holds, so other users of a shared lock get their turn
set -euo pipefail
# Run from a private copy: bash reads a script as it goes, so editing the checkout's copy would derail a long run.
if [ -z "${VLLM_CPU_MAC_ROOT:-}" ]; then
  VLLM_CPU_MAC_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
  copy=$(mktemp -t vllm-cpu-mac.XXXXXX)
  cp "$0" "$copy"
  export VLLM_CPU_MAC_ROOT
  exec bash "$copy" "$@"
fi
here=$VLLM_CPU_MAC_ROOT
self=$0
case "${1:-}" in _*) ;; *) trap 'rm -f "$self"' EXIT ;; esac
cd "$here"
export VLLM_VENV="${VLLM_VENV:-$HOME/work/venvs/vllm-cpu}"
export HF="${HF:-$HOME/work/models/lfm3/hf-1.7b-full}"
export GGUF_DIR="${GGUF_DIR:-$HOME/work/models/lfm3}"
export LLAMA_BIN="${LLAMA_BIN:-$HOME/work/llama.cpp-lfm3/build-cpu/bin}"
export OUT="${OUT:-$here/artifacts/lfm3/cpu/vllm/$(date +%Y%m%d-%H%M)}"
ROUNDS="${ROUNDS:-3}"
CELLS="${CELLS:-512x128@1,128x256@1,512x128@4,128x256@4}"
CTX="${CTX:-4096}"
ARMS="${ARMS:-vllm-fp16fix+vllm-fp32fix+ts-ggml-q4km+llama-q4km llama-bf16+ts-ggml-bf16 ts-cpu-q4km@1+ts-cpu-bf16@1 ts-cpu-q4km@4+ts-cpu-bf16@4+ts-ggml-q4km-t18}"
VLLM_DTYPES="${VLLM_DTYPES:-float16 bfloat16 float32}"
VLLM_BENCH_DTYPE="${VLLM_BENCH_DTYPE:-float16}"
SWEEP_ARM="${SWEEP_ARM:-vllm-fp16fix}" SWEEP_THREADS="${SWEEP_THREADS:-6 12 18}"
SWEEP_EXTRA="${SWEEP_EXTRA-vllm-bf16fix vllm-fp32fix}"
TS_GGML_T="${TS_GGML_T:-0}" TS_CPU_T="${TS_CPU_T:-0}" LLAMA_T="${LLAMA_T:-16}" LLAMA_TB="${LLAMA_TB:-18}"
REQUESTS="${REQUESTS:-1:4,4:8}" SLOW_REQUESTS="${SLOW_REQUESTS:-1:3,4:4}"
SLOW_ARMS="${SLOW_ARMS:-llama-bf16 vllm-fp16}"
SLOW_CELLS="${SLOW_CELLS:-$CELLS}"
PY="$VLLM_VENV/bin/python"
TOOL="$here/eng/lfm3/bench/vllm-cpu-mac.py"
BF16="$GGUF_DIR/LFM3-1.7B-BF16.gguf" Q4KM="$GGUF_DIR/LFM3-1.7B-Q4_K_M.gguf"
heavy() { if [ -n "${HEAVY:-}" ]; then "$HEAVY" "$@"; else shift; "$@"; fi; }
mkdir -p "$OUT/correct" "$OUT/bench" "$OUT/sweep"

conditions() {
  {
    echo "date: $(date '+%F %T')  host: $(sysctl -n machdep.cpu.brand_string) P=$(sysctl -n hw.perflevel0.physicalcpu) E=$(sysctl -n hw.perflevel1.physicalcpu) mem=$(( $(sysctl -n hw.memsize) >> 30 )) GB"
    echo "macOS: $(sw_vers -productVersion)  loadavg: $(sysctl -n vm.loadavg)"
    echo "TensorSharp: $(git rev-parse --short HEAD) $(git rev-parse --abbrev-ref HEAD)  ggml: $(git -C ExternalProjects/ggml rev-parse --short HEAD 2>/dev/null || echo '?')"
    echo "llama.cpp: $(git -C "$LLAMA_BIN/../.." rev-parse --short HEAD 2>/dev/null || echo '?') ($LLAMA_BIN)"
    echo "vLLM: $("$PY" -c 'import vllm, torch; print(vllm.__version__, "torch", torch.__version__)' 2>/dev/null)  ($(git -C "$("$PY" -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))' 2>/dev/null)" rev-parse --short HEAD 2>/dev/null || echo wheel))"
    echo "dotnet: $(dotnet --version)"
    echo "threads: VLLM_THREADS=${VLLM_THREADS:-sweep} TS_GGML_T=$TS_GGML_T TS_CPU_T=$TS_CPU_T (0: TensorSharp's defaults) LLAMA_T=$LLAMA_T LLAMA_TB=$LLAMA_TB"
    echo "cells: $CELLS, requests $REQUESTS (slow arms $SLOW_ARMS: $SLOW_CELLS, $SLOW_REQUESTS)  ctx: $CTX  rounds: $ROUNDS"
  } | tee -a "$OUT/conditions.txt"
}

arm_args() {   # arm name -> engine arguments
  case "$1" in
    vllm-*)
      local d=${1#vllm-} t=${VLLM_THREADS:-12} fix=
      case $d in *-t*) t=${d##*-t}; d=${d%-t*} ;; esac
      case $d in *fix) fix=--shortconv-fix; d=${d%fix} ;; esac
      case $d in fp16) d=float16 ;; bf16) d=bfloat16 ;; fp32) d=float32 ;; *) echo "bad arm $1" >&2; return 1 ;; esac
      echo "--engine vllm --dtype $d --threads $t $fix" ;;
    ts-*-t*)      echo "$(arm_args "${1%-t*}" | sed 's/--threads [0-9]*/--threads '"${1##*-t}"'/')" ;;
    ts-ggml-bf16) echo "--engine ts --backend ggml_cpu --gguf $BF16 --threads $TS_GGML_T" ;;
    ts-ggml-q4km) echo "--engine ts --backend ggml_cpu --gguf $Q4KM --threads $TS_GGML_T" ;;
    ts-cpu-bf16)  echo "--engine ts --backend cpu --gguf $BF16 --threads $TS_CPU_T" ;;
    ts-cpu-q4km)  echo "--engine ts --backend cpu --gguf $Q4KM --threads $TS_CPU_T" ;;
    llama-bf16)   echo "--engine llama --gguf $BF16 --threads $LLAMA_T --threads-batch $LLAMA_TB" ;;
    llama-q4km)   echo "--engine llama --gguf $Q4KM --threads $LLAMA_T --threads-batch $LLAMA_TB" ;;
    *) echo "unknown arm $1" >&2; return 1 ;;
  esac
}

is_slow() { case " $SLOW_ARMS " in *" ${1%%-t[0-9]*} "*) return 0 ;; esac; return 1; }

run_arm() {   # run_arm ARM[@C] ROUND OUTDIR [CELLS REQUESTS]: one arm-round (servers up, cells, servers down);
              # ARM@C keeps only the cells at concurrency C
  local arm=${1%@*} r=$2 out=$3 cells=${4:-$CELLS} req=${5:-$REQUESTS} args
  if [ -z "${4:-}" ] && is_slow "$arm"; then cells=$SLOW_CELLS req=$SLOW_REQUESTS; fi
  case $1 in *@*) cells=$(echo "$cells" | tr ',' '\n' | grep "@${1##*@}\$" | paste -sd, -) ;; esac
  args=$(arm_args "$arm")
  # shellcheck disable=SC2086
  "$PY" "$TOOL" arm $args --name "$arm" --round "$r" --hf "$HF" --ctx "$CTX" --cells "$cells" --requests "$req" \
    --out "$out" > "$out/${1/@/.c}.r$r.log" 2>&1 || echo "arm $1 round $r failed (see $out/${1/@/.c}.r$r.log)"
}

# Internal steps, each run inside one lock hold by the public steps below.
case "${1:-}" in
  _group)   # _group ROUND A+B+C: the group's arms, the order rotated by the round
    IFS=+ read -r -a members <<< "$3"
    n=${#members[@]}
    for ((k = 0; k < n; k++)); do run_arm "${members[$(((k + $2 - 1) % n))]}" "$2" "$OUT/bench"; done
    exit 0 ;;
  _correct_offline)   # vLLM offline per dtype (with the workaround unless FIX is set empty) + HF float32
    for d in $VLLM_DTYPES; do
      # shellcheck disable=SC2086
      "$PY" "$TOOL" gen-vllm --hf "$HF" --dtype "$d" --threads "${VLLM_THREADS:-12}" --out "$OUT/correct" ${FIX---shortconv-fix} \
        > "$OUT/correct/gen-vllm-offline-$d${FIX-fix}.log" 2>&1 || echo "gen-vllm $d failed"
    done
    "$PY" "$TOOL" gen-hf --hf "$HF" --dtype float32 --threads 12 --out "$OUT/correct" > "$OUT/correct/gen-hf.log" 2>&1 \
      || echo "gen-hf failed"
    # the reference: full forwards, no cache (generate()'s cached path is checked against it, not trusted)
    "$PY" "$TOOL" gen-hf --recompute --hf "$HF" --dtype float32 --threads 12 --out "$OUT/correct" \
      > "$OUT/correct/gen-hf-recompute.log" 2>&1 || echo "gen-hf --recompute failed"
    exit 0 ;;
  _correct_servers)   # each server engine on the BF16 weights (vLLM: VLLM_BENCH_DTYPE with the workaround)
    vb=$(echo "$VLLM_BENCH_DTYPE" | sed 's/^bfloat16$/bf16/; s/^float/fp/')
    for a in "vllm-server-${vb}fix|vllm-${vb}fix" "ts-ggml_cpu-bf16|ts-ggml-bf16" "ts-cpu-bf16|ts-cpu-bf16" \
             "llama-bf16|llama-bf16"; do
      args=$(arm_args "${a#*|}")
      # shellcheck disable=SC2086
      "$PY" "$TOOL" gen-server $args --name "${a%%|*}" --hf "$HF" --out "$OUT/correct" \
        > "$OUT/correct/gen-${a%%|*}.log" 2>&1 || echo "gen-server ${a%%|*} failed"
    done
    exit 0 ;;
  _stock)
    run_arm vllm-fp16 1 "$OUT/bench" "${STOCK_CELLS:-512x128@1,512x128@4}" "${STOCK_REQUESTS:-1:2,4:4}"
    exit 0 ;;
  _sweep)
    c1=$(echo "$CELLS" | tr ',' '\n' | grep '@1$' | paste -sd, -)
    for t in $SWEEP_THREADS; do run_arm "$SWEEP_ARM-t$t" 1 "$OUT/sweep" "$c1" "$REQUESTS"; done
    for a in $SWEEP_EXTRA; do run_arm "$a" 1 "$OUT/sweep" "$c1" "$REQUESTS"; done
    exit 0 ;;
esac

steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(check correct sweep bench stock summary)
for step in "${steps[@]}"; do
  case "$step" in
    check)
      conditions
      "$PY" "$TOOL" check --hf "$HF" | tee "$OUT/check.txt" ;;
    correct)
      "$PY" "$TOOL" prompts --hf "$HF" --out "$OUT/correct" > "$OUT/correct/prompts.log" 2>&1
      heavy vllm-correct-offline bash "$self" _correct_offline
      heavy vllm-correct-servers bash "$self" _correct_servers
      "$PY" "$TOOL" compare --out "$OUT/correct" --ref hf-float32-recompute
      # near tie or bug: the HF float32 logits of the two tokens at each first divergence
      heavy vllm-correct-margins "$PY" "$TOOL" margins --hf "$HF" --out "$OUT/correct" --ref hf-float32-recompute --threads 12 \
        > "$OUT/correct/margins.log" 2>&1 || echo "margins failed"
      cat "$OUT/correct/margins.md" 2>/dev/null || true ;;
    sweep)
      heavy vllm-sweep bash "$self" _sweep
      "$PY" "$TOOL" summary --out "$OUT/sweep"
      if [ -z "${VLLM_THREADS:-}" ]; then
        VLLM_THREADS=$("$PY" -c '
import glob, json, statistics, sys, collections
tput = collections.defaultdict(list)
for f in glob.glob(sys.argv[1] + "/*.json"):
    r = json.load(open(f))
    if r.get("arm", "").startswith(sys.argv[2] + "-t"):
        tput[r["threads"]].append(r["output_throughput"])
print(max(tput, key=lambda t: statistics.mean(tput[t])) if tput else 12)' "$OUT/sweep" "$SWEEP_ARM")
        export VLLM_THREADS
        echo "vLLM threads: $VLLM_THREADS (best mean output tok/s over the concurrency-1 cells)" | tee -a "$OUT/conditions.txt"
      fi ;;
    bench)
      read -r -a groups <<< "$ARMS"
      n=${#groups[@]}
      for ((r = 1; r <= ROUNDS; r++)); do
        for ((k = 0; k < n; k++)); do
          g=${groups[$(((k + r - 1) % n))]}
          heavy "vllm-bench-r$r-$g" bash "$self" _group "$r" "$g"
          sleep "${HOLD_GAP:-0}"
        done
      done ;;
    stock)
      heavy vllm-bench-stock bash "$self" _stock ;;
    summary)
      "$PY" "$TOOL" summary --out "$OUT/bench" \
        --arms "$( (echo "$ARMS" | tr ' +' '\n\n' | sed 's/@.*//'; echo vllm-fp16) | awk '!seen[$0]++' | paste -sd, -)" ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
echo "results: $OUT"
