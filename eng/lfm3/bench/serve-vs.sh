#!/usr/bin/env bash
# LFM3 serving on one Linux GPU: TensorSharp.Server.Host against vLLM, SGLang and llama-server, all measured by one client
# (`vllm bench serve --backend openai-chat`: random token prompts, ignore_eos, temperature 0, thinking off, streamed).
#
# usage: serve-vs.sh OUTDIR ENGINE [ENGINE ...]
#   ENGINE: vllm | sglang | ts-cuda | ts-ggml_cuda | ts-ggml_vulkan | llama-cuda | llama-vk
#           (ts-* and llama-* serve GGUF=..., vllm/sglang serve HF_MODEL=...)
# Each engine is started, warmed (WARMUP requests), run over every SHAPES x CONC cell and stopped before the next one
# starts; one JSON per cell (vllm bench serve --save-result) plus the GPU memory in use after the cell (nvidia-smi, read
# between cells only: a poller during a run perturbs short decode steps). serve-vs-summary.py tabulates OUTDIR.
#
# Environment (defaults are the A40 VM's layout):
#   SHAPES        "ISL,OSL ..."           (default "128,512 1024,128 1024,1024 4096,256")
#   CONC          concurrencies           (default "1 4 16 64")
#   NPROMPTS      requests per cell: "x4" = 4 x C with a floor of MIN_PROMPTS (16), or a number (default x4)
#   PREFIX_CACHE  0 (default: off on every engine) or 1
#   RANDOM_PREFIX tokens of a prefix every request of a cell shares (vllm bench's --random-prefix-len, in front of ISL;
#                 default 0): with PREFIX_CACHE=1, the prefix-reuse arm
#   MAX_CONTEXT   context length every engine is given (default 32768)
#   MAX_SEQS      running-sequence limit (default 64; TS_SCHED_MAX_RUNNING_SEQS, vLLM --max-num-seqs, SGLang
#                 --max-running-requests, llama-server -np)
#   BATCH_TOKENS  per-step token budget (default 8192: TS_SCHED_MAX_BATCHED_TOKENS, vLLM --max-num-batched-tokens,
#                 SGLang --chunked-prefill-size)
#   KV_POOL       sized (default): TensorSharp's KV block pool holds MAX_SEQS x MAX_CONTEXT tokens (TS_SCHED_NUM_BLOCKS),
#                 as vLLM and SGLang size theirs from free memory, so no running request waits or is preempted for blocks
#                 (LFM3's block pool is bookkeeping: its sequences' caches grow with what they hold); default: the
#                 server's own pool (256 blocks of 256 tokens, or one context): past 65,536 tokens of running requests
#                 they wait, or are preempted and re-prefilled
#   HF_MODEL (/root/models/hf/LFM3-1.7B), GGUF (/root/models/lfm3/LFM3-1.7B-BF16.gguf), TS_ROOT (/root/ts),
#   TS_SERVER_BIN (TS_ROOT/TensorSharp.Server.Host/bin), VLLM_VENV (/root/venv-vllm), SGLANG_VENV (/root/venv-sglang),
#   LLAMA_CPP (/root/llama.cpp: build-cuda, build-vk), PORT (18431), SEED (1)
set -uo pipefail
out=$1; shift
mkdir -p "$out"
SHAPES=${SHAPES:-"128,512 1024,128 1024,1024 4096,256"}
CONC=${CONC:-"1 4 16 64"}
NPROMPTS=${NPROMPTS:-x4}
MIN_PROMPTS=${MIN_PROMPTS:-16}
PREFIX_CACHE=${PREFIX_CACHE:-0}
RANDOM_PREFIX=${RANDOM_PREFIX:-0}
MAX_CONTEXT=${MAX_CONTEXT:-32768}
MAX_SEQS=${MAX_SEQS:-64}
BATCH_TOKENS=${BATCH_TOKENS:-8192}
KV_POOL=${KV_POOL:-sized}
HF_MODEL=${HF_MODEL:-/root/models/hf/LFM3-1.7B}
GGUF=${GGUF:-/root/models/lfm3/LFM3-1.7B-BF16.gguf}
TS_ROOT=${TS_ROOT:-/root/ts}
TS_SERVER_BIN=${TS_SERVER_BIN:-$TS_ROOT/TensorSharp.Server.Host/bin}
VLLM_VENV=${VLLM_VENV:-/root/venv-vllm}
SGLANG_VENV=${SGLANG_VENV:-/root/venv-sglang}
LLAMA_CPP=${LLAMA_CPP:-/root/llama.cpp}
PORT=${PORT:-18431}
SEED=${SEED:-1}
WARMUP=${WARMUP:-8}
NAME=lfm3

gpu_mem() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
gpu_idle() { [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; }

start() {   # ENGINE LOG -> sets SERVER_PID
  local e=$1 log=$2
  case $e in
    vllm)
      local pc=--no-enable-prefix-caching; [[ $PREFIX_CACHE == 1 ]] && pc=--enable-prefix-caching
      PATH="$VLLM_VENV/bin:$PATH" setsid "$VLLM_VENV/bin/vllm" serve "$HF_MODEL" --served-model-name $NAME --dtype bfloat16 --max-model-len "$MAX_CONTEXT" \
        --gpu-memory-utilization 0.90 --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens "$BATCH_TOKENS" \
        --generation-config vllm $pc --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 < /dev/null & ;;
    sglang)
      local pc=--disable-radix-cache; [[ $PREFIX_CACHE == 1 ]] && pc=
      # flashinfer builds its kernels with the venv's ninja on the first run
      PATH="$SGLANG_VENV/bin:$PATH" setsid "$SGLANG_VENV/bin/python" -m sglang.launch_server --model-path "$HF_MODEL" --trust-remote-code --served-model-name $NAME \
        --context-length "$MAX_CONTEXT" --mem-fraction-static 0.85 --max-running-requests "$MAX_SEQS" \
        --chunked-prefill-size "$BATCH_TOKENS" --sampling-defaults openai $pc --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 < /dev/null & ;;
    ts-*)
      local pc=; [[ $PREFIX_CACHE == 1 ]] || pc=--no-prefix-cache
      local pool=; [[ $KV_POOL == sized ]] && pool="TS_SCHED_NUM_BLOCKS=$(( (MAX_SEQS * MAX_CONTEXT + 255) / 256 ))"
      ( cd "$TS_ROOT" && MAX_CONTEXT=$MAX_CONTEXT TS_SCHED_MAX_RUNNING_SEQS=$MAX_SEQS TS_SCHED_MAX_BATCHED_TOKENS=$BATCH_TOKENS \
        exec setsid env $pool dotnet "$TS_SERVER_BIN/TensorSharp.Server.Host.dll" --model "$GGUF" --backend "${e#ts-}" \
        --host 127.0.0.1 --port "$PORT" --no-multi-agent --no-skills --no-webui $pc \
        --sampling-precedence request --repeat-penalty 1.0 --max-tokens 32768 > "$log" 2>&1 < /dev/null ) & ;;
    llama-cuda|llama-vk)
      local pc=--no-cache-prompt; [[ $PREFIX_CACHE == 1 ]] && pc=
      local np=$MAX_SEQS
      setsid "$LLAMA_CPP/build-${e#llama-}/bin/llama-server" -m "$GGUF" -ngl 99 -fa on -c $((MAX_CONTEXT * np)) -np $np \
        --cont-batching $pc --jinja --host 127.0.0.1 --port "$PORT" > "$log" 2>&1 < /dev/null & ;;
    *) echo "unknown engine $e" >&2; return 2 ;;
  esac
  SERVER_PID=$!
  for _ in $(seq 1 600); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1 && return 0
    kill -0 $SERVER_PID 2>/dev/null || { echo "$e exited during startup (see $log)" >&2; return 1; }
    sleep 2
  done
  echo "$e not ready after 20 min" >&2; return 1
}

stop() {   # the server's whole session (setsid): vLLM / SGLang spawn workers
  [[ -n ${SERVER_PID:-} ]] || return 0
  kill -TERM -- -"$SERVER_PID" 2>/dev/null; kill -TERM "$SERVER_PID" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -- -"$SERVER_PID" 2>/dev/null
  for _ in $(seq 1 60); do gpu_idle && break; sleep 1; done
  SERVER_PID=
}
trap stop EXIT

bench() {   # ISL OSL C N LABEL [extra]
  local isl=$1 osl=$2 c=$3 n=$4 label=$5; shift 5
  "$VLLM_VENV/bin/vllm" bench serve --backend openai-chat --base-url "http://127.0.0.1:$PORT" --endpoint /v1/chat/completions \
    --model "$NAME" --tokenizer "$HF_MODEL" --dataset-name random --random-input-len "$isl" --random-output-len "$osl" \
    --random-range-ratio 0 --random-prefix-len "$RANDOM_PREFIX" --num-prompts "$n" --max-concurrency "$c" --request-rate inf --ignore-eos --temperature 0 \
    --chat-template-kwargs '{"enable_thinking": false}' --seed "$SEED" --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 --disable-tqdm "$@" --save-result --result-dir "$out" --result-filename "$label.json" \
    --label "$label" < /dev/null
}

for e in "$@"; do
  gpu_idle || { echo "GPU busy before $e: $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader)" >&2; exit 3; }
  echo "=== $e $(date -u +%H:%M:%S) idle GPU memory $(gpu_mem) MiB" | tee -a "$out/serve.log"
  NAME=lfm3
  start "$e" "$out/server-$e.log" || { stop; continue; }
  # The name the server hosts (TensorSharp and llama-server serve the GGUF's own; vLLM and SGLang the one given).
  NAME=$(curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
  echo "$e ready $(date -u +%H:%M:%S) as $NAME, GPU memory $(gpu_mem) MiB" | tee -a "$out/serve.log"
  bench 256 64 4 "$WARMUP" "warmup-$e" > "$out/warmup-$e.log" 2>&1
  for shape in $SHAPES; do
    isl=${shape%,*}; osl=${shape#*,}
    for c in $CONC; do
      if [[ $NPROMPTS == x* ]]; then n=$(( c * ${NPROMPTS#x} )); (( n < MIN_PROMPTS )) && n=$MIN_PROMPTS; else n=$NPROMPTS; fi
      label="$e-i$isl-o$osl-c$c"
      bench "$isl" "$osl" "$c" "$n" "$label" > "$out/$label.log" 2>&1
      echo "$label exit $? $(date -u +%H:%M:%S) gpu_mem_mib $(gpu_mem)" | tee -a "$out/serve.log"
    done
  done
  stop
done
echo SERVE-VS-DONE | tee -a "$out/serve.log"
