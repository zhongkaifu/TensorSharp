#!/usr/bin/env bash
# Start, stop and benchmark the six Qwen3.8-27B servers compared by eng/qwen38-engine-bench.py, each
# with the settings the 2026-10-27 RTX 5090 comparison used. One engine runs at a time.
#
#   eng/qwen38-engine-servers.sh start  <engine> <plain|dflash2> [label]   detached; returns when /v1/models answers
#   eng/qwen38-engine-servers.sh stop                                       stops every engine, waits for a free GPU
#   eng/qwen38-engine-servers.sh bench  <engine> <plain|dflash2> <label> [scenarios] [repeats]
#                                                                           start, run the bench, stop
#   eng/qwen38-engine-servers.sh matrix                                     every engine plain, then DFlash2, then a
#                                                                           TensorSharp plain control run (drift check)
#
# Engines: tensorsharp llamacpp vllm sglang velox ninfer. Every server gets a 40960-token context per
# sequence and 4 lanes (SGLang+DFlash2: 2, the most that fit in 32 GB), vision on, thinking off.
# Paths (defaults are the VM layout of the comparison):
#   MODELS        GGUFs: Qwen3.8-27B-NVFP4-MTP-N4_0.gguf, mmproj-BF16.gguf, Qwen3.8-27B-DFlash2-Q4_K_M.gguf
#   HF            Qwen3.8-27B-NVFP4-nvidia/, Qwen3.8-27B-DFlash2/ (safetensors), ninfer/qwen3_8_27b_nvfp4.ninfer
#   TS_BIN        TensorSharp.Server.Host/bin       LLAMA_BIN    llama.cpp build/bin
#   VLLM_VENV     vLLM 0.32 virtualenv              SGLANG_VENV  SGLang 0.5.22 virtualenv
#   VELOX         velox binary (rtx-5090-32g)       NINFER_SERVE ninfer-serve binary
#   OUT           logs/ and results/ go here        PY           python with `tokenizers` (completion counts)
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODELS=${MODELS:-/workspace/models}; HF=${HF:-/workspace/hf}
TS_BIN=${TS_BIN:-$ROOT/TensorSharp.Server.Host/bin}; LLAMA_BIN=${LLAMA_BIN:-/root/llama.cpp/build/bin}
VLLM_VENV=${VLLM_VENV:-/root/venvs/vllm}; SGLANG_VENV=${SGLANG_VENV:-/root/venvs/sglang}
VELOX=${VELOX:-/root/velox-inferer/target/release/velox}; NINFER_SERVE=${NINFER_SERVE:-/root/ninfer/build/apps/ninfer-serve}
OUT=${OUT:-$ROOT/artifacts/qwen38-engines}; PY=${PY:-$VLLM_VENV/bin/python}
CTX=40960; LANES=4
ALL=short_chat,long_prompt_4k,long_prompt_16k,long_prompt_32k,json_object,json_schema,tool_call_single,tool_call_parallel,agentic_loop,multi_turn,image,long_generation,parallel_2,parallel_4

die() { echo "qwen38-engine-servers: $*" >&2; exit 2; }
port_of() { case $1 in tensorsharp) echo 5088;; llamacpp) echo 8082;; vllm) echo 8000;; sglang) echo 30000;;
                       velox) echo 8010;; ninfer) echo 8090;; *) die "unknown engine $1";; esac; }

start() {
  local engine=$1 mode=$2 label=${3:-main} port log spec=()
  [ "$mode" = plain ] || [ "$mode" = dflash2 ] || die "mode must be plain or dflash2"
  port=$(port_of "$engine"); mkdir -p "$OUT/logs"; log=$OUT/logs/$engine-$mode-$label.log
  case $engine in
  tensorsharp)
    # Tuned: 7 drafts, gate off, 4096-row prefill batches (see the comparison's tuning sweep).
    [ "$mode" = dflash2 ] && spec=(--draft-model "$MODELS/Qwen3.8-27B-DFlash2-Q4_K_M.gguf" --spec-draft 7 --spec-pmin 0)
    (cd "$TS_BIN" && MAX_CONTEXT=$CTX TS_Q35_UBATCH=4096 exec dotnet TensorSharp.Server.Host.dll \
      --model "$MODELS/Qwen3.8-27B-NVFP4-MTP-N4_0.gguf" --mmproj "$MODELS/mmproj-BF16.gguf" --backend cuda \
      --host 127.0.0.1 --port "$port" --no-webui --no-multi-agent --no-skills --temperature 0 --repeat-penalty 1 \
      "${spec[@]}") > "$log" 2>&1 < /dev/null & ;;
  llamacpp)
    # Context checkpoints cost ~200 ms per prompt and a unified KV slows the next long prompt: both off.
    [ "$mode" = dflash2 ] && spec=(-md "$MODELS/Qwen3.8-27B-DFlash2-Q4_K_M.gguf" --spec-type draft-dflash --spec-draft-n-max 7 -ngld 99)
    "$LLAMA_BIN/llama-server" -m "$MODELS/Qwen3.8-27B-NVFP4-MTP-N4_0.gguf" --mmproj "$MODELS/mmproj-BF16.gguf" \
      -ngl 99 -fa on -c $((CTX * LANES)) -np $LANES -no-kvu -b 2048 -ub 2048 -ctxcp 0 --cache-ram 0 -ctk q8_0 -ctv q8_0 \
      --jinja --metrics --host 127.0.0.1 --port "$port" "${spec[@]}" > "$log" 2>&1 < /dev/null & ;;
  vllm)
    # The BF16 drafter leaves too little KV for a 40960-token sequence: quantize it and the KV to FP8.
    local util=0.95 graphs=64
    if [ "$mode" = dflash2 ]; then
      util=0.97; graphs=32
      spec=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$HF/Qwen3.8-27B-DFlash2\",\"num_speculative_tokens\":7,\"quantization\":\"fp8\"}")
    fi
    "$VLLM_VENV/bin/vllm" serve "$HF/Qwen3.8-27B-NVFP4-nvidia" --served-model-name qwen3.8-27b --host 127.0.0.1 --port "$port" \
      --max-model-len $CTX --max-num-seqs $LANES --gpu-memory-utilization $util --kv-cache-dtype fp8 --enable-prefix-caching \
      --max-cudagraph-capture-size $graphs --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      "${spec[@]}" > "$log" 2>&1 < /dev/null & ;;
  sglang)
    # With DFlash2 the per-draft-token recurrent-state cache does not fit 4 lanes: 2 lanes, BF16 state.
    local lanes=$LANES frac=0.85
    if [ "$mode" = dflash2 ]; then
      lanes=2; frac=0.88
      spec=(--speculative-algorithm DFLASH --speculative-draft-model-path "$HF/Qwen3.8-27B-DFlash2" --speculative-num-draft-tokens 8
            --speculative-draft-model-quantization fp8 --max-mamba-cache-size 10 --mamba-ssm-dtype bfloat16 --cuda-graph-max-bs-prefill 512)
    fi
    "$SGLANG_VENV/bin/python" -m sglang.launch_server --model-path "$HF/Qwen3.8-27B-NVFP4-nvidia" --served-model-name qwen3.8-27b \
      --host 127.0.0.1 --port "$port" --context-length $CTX --max-running-requests $lanes --cuda-graph-max-bs-decode $lanes \
      --mem-fraction-static $frac --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 2048 --reasoning-parser qwen3 \
      --tool-call-parser qwen3_coder --enable-metrics "${spec[@]}" > "$log" 2>&1 < /dev/null & ;;
  velox)
    spec=(--dflash-drafts 0 --mtp-drafts 0); [ "$mode" = dflash2 ] && spec=(--dflash-drafts 7 --mtp-drafts 0)
    "$VELOX" serve --artifact "$HF/ninfer/qwen3_8_27b_nvfp4.ninfer" --host 127.0.0.1 --port "$port" --model-id qwen3.8-27b \
      --max-context $CTX --max-concurrency $LANES --kv-format nvfp4 --prefill-chunk 2048 --host-context-mib 16384 \
      --enable-thinking false "${spec[@]}" > "$log" 2>&1 < /dev/null & ;;
  ninfer)
    [ "$mode" = dflash2 ] && spec=(--spec dflash2 --draft-tokens 7)
    "$NINFER_SERVE" "$HF/ninfer/qwen3_8_27b_nvfp4.ninfer" --host 127.0.0.1 --port "$port" --model-id qwen3.8-27b \
      --max-context $CTX --kv-capacity auto --max-concurrency $LANES --kv-dtype fp8 --vision --no-thinking \
      --request-log-jsonl "$OUT/logs/ninfer-$mode-$label.requests.jsonl" "${spec[@]}" > "$log" 2>&1 < /dev/null & ;;
  *) die "unknown engine $engine" ;;
  esac
  local pid=$!
  for _ in $(seq 1 400); do
    kill -0 $pid 2>/dev/null || { echo "DIED $engine $mode (log $log)"; tail -5 "$log"; return 1; }
    curl -sf -m 3 "127.0.0.1:$port/v1/models" > /dev/null 2>&1 && { echo "READY $engine $mode port=$port log=$log"; return 0; }
    sleep 3
  done
  echo "TIMEOUT $engine $mode (log $log)"; return 1
}

stop() {
  # Anchored on the executables so a shell whose command line mentions an engine is never matched.
  local pats=("^dotnet TensorSharp.Server.Host.dll" "^$LLAMA_BIN/" "^$VLLM_VENV/bin/" "^VLLM::" "^$SGLANG_VENV/bin/"
              "^sglang::" "^$VELOX" "^$NINFER_SERVE") p used
  for p in "${pats[@]}"; do pkill -f "$p"; done
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$used" -lt 1000 ] && return 0
    sleep 2
  done
  for p in "${pats[@]}"; do pkill -9 -f "$p"; done
}

bench() {
  local engine=$1 mode=$2 label=$3 scenarios=${4:-$ALL} repeats=${5:-3} log
  stop
  start "$engine" "$mode" "$label" || { stop; return 1; }
  log=$OUT/logs/$engine-$mode-$label.log
  mkdir -p "$OUT/results"
  "$PY" "$ROOT/eng/qwen38-engine-bench.py" --engine "$engine" --mode "$mode" --label "$label" \
    --url "http://127.0.0.1:$(port_of "$engine")" --out "$OUT/results/$engine-$mode-$label.jsonl" --overwrite \
    --scenarios "$scenarios" --repeats "$repeats" --warmup 3 --tokenizer "$HF/Qwen3.8-27B-NVFP4-nvidia/tokenizer.json" \
    --server-log "$log"
  stop
}

case ${1:-} in
start) shift; [ $# -ge 2 ] || die "start <engine> <plain|dflash2> [label]"; start "$@" ;;
stop) stop ;;
bench) shift; [ $# -ge 3 ] || die "bench <engine> <plain|dflash2> <label> [scenarios] [repeats]"; bench "$@" ;;
matrix)
  for mode in plain dflash2; do
    for engine in tensorsharp llamacpp vllm sglang velox ninfer; do bench $engine $mode main; done
  done
  bench tensorsharp plain control short_chat,long_prompt_16k,multi_turn,long_generation ;;
*) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
