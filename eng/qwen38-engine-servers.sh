#!/usr/bin/env bash
# Reproducible, isolated Qwen3.8-27B serving and benchmark orchestration (Linux).
#
# start <engine> <plain|dflash2> [label]
# stop                          stop only process groups registered under OUT/pids
# bench <engine> <mode> <label> [scenarios] [repeats] [-- bench arguments...]
# matrix [-- bench arguments...]
#
# Engines: tensorsharp llamacpp vllm sglang velox ninfer infernix ignis.
# Optional: eole (single active lane, text-only DFlash2), focus-llama (GGUF fork).
# CTX and LANES are per-sequence context and active lanes. For the long suite use
# CTX=262144 LANES=1 SCENARIOS=long_prompt_128k,long_prompt_250k (250K=256000).
# STARTUP_TIMEOUT_SECONDS defaults to 3600 for first-use vendor JIT. Optional
# START_ATTEMPTS additionally limits readiness probes (0 means no count limit).
# All logs, manifests, PID records and results go below ignored artifacts/ by default.
# Model formats differ across engines: retain each manifest and quality results.
set -uo pipefail
shopt -s nullglob

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODELS=${MODELS:-/workspace/models}; HF=${HF:-/workspace/hf}
TS_BIN=${TS_BIN:-$ROOT/TensorSharp.Server.Host/bin}; TS_BACKEND=${TS_BACKEND:-cuda}
TS_SOURCE=${TENSORSHARP_REPO:-${TS_REPO:-$ROOT}}
LLAMA_BIN=${LLAMA_BIN:-/root/llama.cpp/build/bin}
FOCUS_LLAMA_BIN=${FOCUS_LLAMA_BIN:-/root/focus-llama/build/bin}
VLLM_VENV=${VLLM_VENV:-/root/venvs/vllm}; SGLANG_VENV=${SGLANG_VENV:-/root/venvs/sglang}
VELOX=${VELOX:-/root/velox-inferer/target/release/velox}
NINFER_SERVE=${NINFER_SERVE:-/root/ninfer/build/rtx5090-release/apps/ninfer-serve}
INFERNIX_SERVE=${INFERNIX_SERVE:-/root/Infernix/build/apps/infernix-serve}
IGNIS_SERVE=${IGNIS_SERVE:-/root/ignis/ignis-server}
EOLE_VENV=${EOLE_VENV:-/root/venvs/eole}; EOLE_SERVE=${EOLE_SERVE:-$EOLE_VENV/bin/eole}
OUT=${OUT:-$ROOT/artifacts/qwen38-engines}; CONTROL_PY=${CONTROL_PY:-python3}
PY=${PY:-$VLLM_VENV/bin/python}; [ -x "$PY" ] || PY=python3
PROCESS=$ROOT/eng/qwen38-engine-process.py
CTX=${CTX:-40960}; LANES=${LANES:-4}; SPEC_DRAFT=${SPEC_DRAFT:-7}; SPEC_PMIN=${SPEC_PMIN:-0}
WARMUP=${WARMUP:-3}; NONCE_SEED=${NONCE_SEED:-20261101}
TOKENIZER=${TOKENIZER:-$HF/Qwen3.8-27B-NVFP4-nvidia/tokenizer.json}
TS_MODEL=${TS_MODEL:-$MODELS/Qwen3.8-27B-NVFP4-MTP-N4_0.gguf}
TS_DRAFT=${TS_DRAFT:-$MODELS/Qwen3.8-27B-DFlash2-Q4_K_M.gguf}
MMPROJ=${MMPROJ:-$MODELS/mmproj-BF16.gguf}
LLAMA_MODEL=${LLAMA_MODEL:-$TS_MODEL}; LLAMA_DRAFT=${LLAMA_DRAFT:-$TS_DRAFT}
HF_MODEL=${HF_MODEL:-$HF/Qwen3.8-27B-NVFP4-nvidia}
HF_DRAFT=${HF_DRAFT:-$HF/Qwen3.8-27B-DFlash2}
NINFER_ARTIFACT=${NINFER_ARTIFACT:-$HF/ninfer/qwen3_8_27b_nvfp4.ninfer}
VELOX_ARTIFACT=${VELOX_ARTIFACT:-$NINFER_ARTIFACT}
INFERNIX_ARTIFACT=${INFERNIX_ARTIFACT:-$NINFER_ARTIFACT}
IGNIS_ARTIFACT=${IGNIS_ARTIFACT:-$HF/ignis/qwen3_8_27b_nvfp4full-v2.ninfer}
EOLE_MODEL=${EOLE_MODEL:-$HF/Qwen3.8-27B-INT4-eole}
BASE_SCENARIOS=short_chat,long_prompt_4k,long_prompt_16k,long_prompt_32k,json_object,json_schema,tool_call_single,tool_call_parallel,agentic_loop,multi_turn,image,long_generation,parallel_2,parallel_4,quality_tasks,quality_gsm8k
SCENARIOS=${SCENARIOS:-$BASE_SCENARIOS}
CAPTURE_ENV=CUDA_VISIBLE_DEVICES,CUDA_MODULE_LOADING,LD_LIBRARY_PATH,OMP_NUM_THREADS,TS_Q35_UBATCH,TS_CUDA_Q35_PREFILL,TS_CUDA_NVFP4,TS_CUDA_NVFP4_GEMM,TS_CUDA_NVFP4_GEMV,TS_CUDA_GRAPH,TS_CUDA_TF32,TS_CUDA_QUANT_MATMUL,TS_CUDA_GRAPH_DISABLE
CAPTURE_ENV=$CAPTURE_ENV,${MANIFEST_ENV_ALLOWLIST:-}
CAPTURE_ENV=$CAPTURE_ENV,TS_Q35_CUDA_GRAPHS,TS_Q35_VERIFY_GRAPHS,TS_Q35_TOPK_PARTITIONED,TS_Q35_GDN_PREPARED_PREFILL,TS_Q35_GREEDY_VERIFY,TS_Q35_DRAFT_SPLITK,TS_Q35_GEMM_SPLITK,TS_Q35_ATTN_PREFILL,TS_Q35_NATIVE,TS_PREFIX_CACHE_ISOLATION,GGML_CUDA_PDL
CAPTURE_ENV=$CAPTURE_ENV,EOLE_TORCH_COMPILE,EOLE_COMPILE_MODE,TORCH_CUDA_ARCH_LIST
CAPTURE_ENV=$CAPTURE_ENV,TMPDIR,MAX_JOBS,FLASHINFER_CUDA_ARCH_LIST,FLASHINFER_WORKSPACE_BASE,VLLM_CPU_OFFLOAD_GB
CAPTURE_ENV=$CAPTURE_ENV,TS_SCHED_MAX_RUNNING_SEQS

die() { echo "qwen38-engine-servers: $*" >&2; return 2; }
port_of() {
  case $1 in
  tensorsharp) echo "${TS_PORT:-5088}";; llamacpp) echo "${LLAMA_PORT:-8082}";;
  vllm) echo "${VLLM_PORT:-8000}";; sglang) echo "${SGLANG_PORT:-30000}";;
  velox) echo "${VELOX_PORT:-8010}";; ninfer) echo "${NINFER_PORT:-8090}";;
  infernix) echo "${INFERNIX_PORT:-8012}";; ignis) echo "${IGNIS_PORT:-8013}";;
  eole) echo "${EOLE_PORT:-8014}";; focus-llama) echo "${FOCUS_LLAMA_PORT:-8015}";;
  *) die "unknown engine $1";; esac
}
manifest_of() { echo "$OUT/manifests/$1-$2-$3.json"; }

start() {
  local engine=$1 mode=$2 label=${3:-main} port log record manifest cwd=$ROOT repo='' pid i actual_lanes=$LANES python_executable=''
  local startup_budget=${STARTUP_TIMEOUT_SECONDS:-3600} startup_attempt_limit=${START_ATTEMPTS:-0} startup_clock
  local -a argv=() spec=() env_args=() launch=() input_args=()
  [[ $label =~ ^[a-zA-Z0-9._-]+$ ]] || { die "label must contain only letters, numbers, dots, underscores or hyphens"; return 2; }
  [ "$mode" = plain ] || [ "$mode" = dflash2 ] || { die "mode must be plain or dflash2"; return 2; }
  [[ $CTX =~ ^[1-9][0-9]*$ && $LANES =~ ^[1-9][0-9]*$ ]] || { die "CTX and LANES must be positive integers"; return 2; }
  [[ $startup_budget =~ ^[1-9][0-9]*$ && $startup_attempt_limit =~ ^[0-9]+$ ]] || { die "STARTUP_TIMEOUT_SECONDS must be positive and START_ATTEMPTS nonnegative integers"; return 2; }
  port=$(port_of "$engine") || return $?
  mkdir -p "$OUT/logs" "$OUT/pids" "$OUT/manifests" || return $?
  log=$OUT/logs/$engine-$mode-$label.log; record=$OUT/pids/$engine-$port.json
  manifest=$(manifest_of "$engine" "$mode" "$label")
  "$CONTROL_PY" - "$port" <<'PORTCHECK' || return 1
import socket, sys
with socket.socket() as sock:
    if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0:
        print(f"Refusing occupied port {sys.argv[1]}; stop its owner first.", file=sys.stderr)
        raise SystemExit(1)
PORTCHECK
  case $engine in
  tensorsharp)
    cwd=$TS_BIN; repo=$TS_SOURCE
    [ "$mode" = dflash2 ] && spec=(--draft-model "$TS_DRAFT" --spec-draft "$SPEC_DRAFT" --spec-pmin "$SPEC_PMIN")
    env_args=(--env "MAX_CONTEXT=$CTX" --env "TS_Q35_UBATCH=${TS_Q35_UBATCH:-4096}"
      --env "TS_SCHED_MAX_RUNNING_SEQS=$LANES")
    argv=(dotnet TensorSharp.Server.Host.dll --model "$TS_MODEL" --mmproj "$MMPROJ" --backend "$TS_BACKEND"
      --host 127.0.0.1 --port "$port" --no-webui --no-multi-agent --no-skills --temperature 0 --repeat-penalty 1 "${spec[@]}") ;;
  llamacpp|focus-llama)
    local llama_bin=$LLAMA_BIN
    if [ "$engine" = focus-llama ]; then
      llama_bin=$FOCUS_LLAMA_BIN; repo=${FOCUS_LLAMA_REPO:-/root/focus-llama}
    else
      repo=${LLAMA_REPO:-$(dirname "$(dirname "$LLAMA_BIN")")}
    fi
    [ "$mode" = dflash2 ] && spec=(-md "$LLAMA_DRAFT" --spec-type draft-dflash --spec-draft-n-max "$SPEC_DRAFT" -ngld 99)
    argv=("$llama_bin/llama-server" -m "$LLAMA_MODEL" --mmproj "$MMPROJ" -ngl 99 -fa on
      -c "$((CTX * LANES))" -np "$LANES" -no-kvu -b "${LLAMA_BATCH:-2048}" -ub "${LLAMA_UBATCH:-2048}"
      -ctxcp "${LLAMA_CHECKPOINTS:-0}" --cache-ram "${LLAMA_CACHE_RAM:-0}" -ctk "${LLAMA_KV_K:-q8_0}" -ctv "${LLAMA_KV_V:-q8_0}"
      --jinja --metrics --host 127.0.0.1 --port "$port" "${spec[@]}") ;;
  vllm)
    local util=${VLLM_MEMORY_UTIL:-0.95} graphs=${VLLM_GRAPHS:-64}
    repo=${VLLM_REPO:-/root/vllm}
    python_executable=$VLLM_VENV/bin/python
    if [ "$mode" = dflash2 ]; then
      util=${VLLM_MEMORY_UTIL:-0.97}; graphs=${VLLM_GRAPHS:-32}
      spec=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$HF_DRAFT\",\"num_speculative_tokens\":$SPEC_DRAFT,\"quantization\":\"fp8\"}")
    fi
    argv=("$VLLM_VENV/bin/vllm" serve "$HF_MODEL" --served-model-name qwen3.8-27b --host 127.0.0.1 --port "$port"
      --max-model-len "$CTX" --max-num-seqs "$LANES" --gpu-memory-utilization "$util" --kv-cache-dtype "${VLLM_KV:-fp8}"
      --cpu-offload-gb "${VLLM_CPU_OFFLOAD_GB:-0}"
      --enable-prefix-caching --max-cudagraph-capture-size "$graphs" --reasoning-parser qwen3
      --enable-auto-tool-choice --tool-call-parser qwen3_coder "${spec[@]}") ;;
  sglang)
    local lanes=$LANES frac=${SGLANG_MEMORY_FRAC:-0.85}
    repo=${SGLANG_REPO:-/root/sglang}
    python_executable=$SGLANG_VENV/bin/python
    if [ "$mode" = dflash2 ]; then
      lanes=${SGLANG_SPEC_LANES:-$((LANES < 2 ? LANES : 2))}; frac=${SGLANG_MEMORY_FRAC:-0.88}
      spec=(--speculative-algorithm DFLASH --speculative-draft-model-path "$HF_DRAFT"
        --speculative-num-draft-tokens "$((SPEC_DRAFT + 1))" --speculative-draft-model-quantization fp8
        --max-mamba-cache-size "${SGLANG_MAMBA_CACHE:-10}" --mamba-ssm-dtype "${SGLANG_SSM_DTYPE:-bfloat16}" --cuda-graph-max-bs-prefill 512)
    fi
    argv=("$SGLANG_VENV/bin/python" -m sglang.launch_server --model-path "$HF_MODEL" --served-model-name qwen3.8-27b
      --host 127.0.0.1 --port "$port" --context-length "$CTX" --max-running-requests "$lanes" --cuda-graph-max-bs-decode "$lanes"
      --mem-fraction-static "$frac" --kv-cache-dtype "${SGLANG_KV:-fp8_e4m3}" --chunked-prefill-size "${SGLANG_PREFILL_CHUNK:-2048}"
      --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-metrics "${spec[@]}")
    actual_lanes=$lanes ;;
  velox)
    repo=${VELOX_REPO:-/root/velox-inferer}
    spec=(--dflash-drafts 0 --mtp-drafts 0)
    [ "$mode" = dflash2 ] && spec=(--dflash-drafts "$SPEC_DRAFT" --mtp-drafts 0)
    argv=("$VELOX" serve --artifact "$VELOX_ARTIFACT" --host 127.0.0.1 --port "$port" --model-id qwen3.8-27b
      --max-context "$CTX" --max-concurrency "$LANES" --kv-format "${VELOX_KV:-nvfp4}"
      --prefill-chunk "${VELOX_PREFILL_CHUNK:-2048}" --host-context-mib "${VELOX_HOST_MIB:-16384}"
      --enable-thinking false "${spec[@]}") ;;
  ninfer)
    repo=${NINFER_REPO:-/root/ninfer}
    spec=(--spec none)
    [ "$mode" = dflash2 ] && spec=(--spec dflash2 --draft-tokens "$SPEC_DRAFT")
    argv=("$NINFER_SERVE" "$NINFER_ARTIFACT" --host 127.0.0.1 --port "$port" --model-id qwen3.8-27b
      --max-context "$CTX" --context-capacity auto --max-concurrency "$LANES" --kv-dtype "${NINFER_KV:-fp8}"
      --proposal-head "${NINFER_PROPOSAL_HEAD:-full}"
      --prefill-chunk "${NINFER_PREFILL_CHUNK:-1024}" --vision --no-thinking
      --request-log-jsonl "$OUT/logs/ninfer-$mode-$label.requests.jsonl" "${spec[@]}") ;;
  infernix)
    repo=${INFERNIX_REPO:-/root/Infernix}
    spec=(--spec off)
    [ "$mode" = dflash2 ] && spec=(--spec dflash2 --draft-tokens "${INFERNIX_DRAFT_TOKENS:-$SPEC_DRAFT}" --lm-head-draft)
    argv=("$INFERNIX_SERVE" "$INFERNIX_ARTIFACT" --host 127.0.0.1 --port "$port" --model-id qwen3.8-27b
      --max-context "$CTX" --kv-capacity auto --max-concurrency "$LANES" --kv-dtype "${INFERNIX_KV:-int8}"
      --prefill-chunk "${INFERNIX_PREFILL_CHUNK:-4096}" --host-context-mib "${INFERNIX_HOST_MIB:-16384}"
      --vision --no-thinking "${spec[@]}") ;;
  ignis)
    repo=${IGNIS_REPO:-/root/ignis}
    spec=(--spec-backend off)
    [ "$mode" = dflash2 ] && spec=(--spec-backend dflash2 --spec-draft-tokens "$SPEC_DRAFT")
    argv=("$IGNIS_SERVE" --profile none --model-artifact "$IGNIS_ARTIFACT" --server-bind "127.0.0.1:$port"
      --model-max-context "$CTX" --spec-decode-lanes "$LANES" --model-kv-format "${IGNIS_KV:-hq-e8-2b}"
      --model-rope-scaling none --model-prefill-chunk "${IGNIS_PREFILL_CHUNK:-2048}"
      --model-prefill-attention "${IGNIS_PREFILL_ATTENTION:-fp8}"
      --model-exception-weights bf16 --model-enable-thinking false --vision-enabled --download-enabled false "${spec[@]}") ;;
  eole)
    repo=${EOLE_REPO:-/root/eole}; python_executable=$EOLE_VENV/bin/python
    actual_lanes=1
    local config=${EOLE_CONFIG:-}
    if [ -z "$config" ]; then
      mkdir -p "$OUT/configs" || return $?
      config=$OUT/configs/eole-$mode-$label.json
      # JSON is valid YAML. No target-head requantization: preserve the converted
      # checkpoint's native head, and record the exact serving config by hash.
      "$CONTROL_PY" - "$config" "$EOLE_MODEL" "$HF_DRAFT" "$CTX" "$mode" "${EOLE_PREFILL_CHUNK:-2048}" <<'EOLE' || return 2
import json, pathlib, sys
path, model, drafter, context, mode, chunk = sys.argv[1:]
config = dict(compute_dtype='bf16', self_attn_backend='flash', beam_size=1, n_best=1,
              top_k=1, top_p=0.0, temperature=1.0, min_length=0, max_length=2048,
              max_length_ratio=0, context_length=int(context), batch_size=1, gpu_ranks=[0],
              self_speculative_decoding=False, report_time=True, fuse_kvq=True, fuse_gate=True,
              prefill_cache_size=0, prefill_chunk_size=int(chunk), vocabulary_quantization='none')
if mode == 'dflash2':
    config.update(dflash2_model=drafter, dflash2_quantization='none')
value = {'models_root': str(pathlib.Path(model).parent), 'models': [
    {'id': 'qwen3.8-27b', 'path': model, 'preload': True, 'warmup': True, 'config': config}]}
pathlib.Path(path).write_text(json.dumps(value, indent=2) + '\n')
EOLE
    fi
    [ -f "$config" ] || { die "EOLE_CONFIG does not exist: $config"; return 2; }
    input_args=(--input-file "$config")
    env_args=(--env "EOLE_TORCH_COMPILE=${EOLE_TORCH_COMPILE:-1}" --env "EOLE_COMPILE_MODE=${EOLE_COMPILE_MODE:-0}")
    cwd=$repo
    argv=("$EOLE_SERVE" serve -c "$config" --host 127.0.0.1 --port "$port") ;;
  *) die "unknown engine $engine"; return 2 ;;
  esac
  # JSON argv additions avoid shell evaluation, word splitting and embedded quoting.
  if [ -n "${ENGINE_EXTRA_ARGS_JSON:-}" ]; then
    local extra_file=$OUT/manifests/.extra-args-$$
    "$CONTROL_PY" - "$ENGINE_EXTRA_ARGS_JSON" > "$extra_file" <<'EXTRA' || { rm -f "$extra_file"; return 2; }
import json, sys
value = json.loads(sys.argv[1])
if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
    raise ValueError("ENGINE_EXTRA_ARGS_JSON must be an array of strings")
for item in value:
    if "\0" in item:
        raise ValueError("NUL in argv")
    sys.stdout.buffer.write(item.encode() + b"\0")
EXTRA
    while IFS= read -r -d '' i; do argv+=("$i"); done < "$extra_file"
    rm -f "$extra_file"
  fi
  launch=("$CONTROL_PY" "$PROCESS" start --record "$record" --manifest "$manifest" --engine "$engine" --mode "$mode"
    --label "$label" --context "$CTX" --lanes "$actual_lanes" --cwd "$cwd" --log "$log"
    --root "$TS_SOURCE" --harness-root "$ROOT" --engine-repo "$repo" --capture-env "$CAPTURE_ENV"
    --model-manifest-json "${MODEL_MANIFEST_JSON:-}" --python-executable "$python_executable"
    --startup-timeout-seconds "$startup_budget" --startup-attempt-limit "$startup_attempt_limit" --readiness-poll-seconds "${POLL_INTERVAL:-3}"
    "${input_args[@]}" "${env_args[@]}" -- "${argv[@]}")
  pid=$("${launch[@]}") || return $?
  startup_clock=$SECONDS
  for ((i=0; startup_attempt_limit == 0 || i < startup_attempt_limit; i++)); do
    if ((SECONDS - startup_clock >= startup_budget)); then break; fi
    if ! "$CONTROL_PY" "$PROCESS" check --record "$record"; then
      "$CONTROL_PY" "$PROCESS" mark --manifest "$manifest" --status launch_failed
      echo "DIED $engine $mode (log $log)" >&2; tail -10 "$log" >&2
      "$CONTROL_PY" "$PROCESS" stop --record "$record" --timeout "${STOP_TIMEOUT:-30}"
      return 1
    fi
    if curl -sf -m 3 "http://127.0.0.1:$port/v1/models" > /dev/null 2>&1; then
      "$CONTROL_PY" "$PROCESS" mark --manifest "$manifest" --status ready || return $?
      echo "READY $engine $mode pid=$pid port=$port manifest=$manifest log=$log"
      return 0
    fi
    sleep "${POLL_INTERVAL:-3}"
  done
  "$CONTROL_PY" "$PROCESS" mark --manifest "$manifest" --status readiness_timeout
  echo "TIMEOUT $engine $mode (log $log)" >&2
  "$CONTROL_PY" "$PROCESS" stop --record "$record" --timeout "${STOP_TIMEOUT:-30}"
  return 1
}

stop() {
  local record status=0
  for record in "$OUT"/pids/*.json; do
    "$CONTROL_PY" "$PROCESS" stop --record "$record" --timeout "${STOP_TIMEOUT:-30}" || status=1
  done
  return "$status"
}

prepare_bench_command() {
  local engine=$1 mode=$2 label=$3 scenarios=$SCENARIOS repeats=${REPEATS:-3} log
  shift 3
  if [ $# -gt 0 ] && [ "$1" != -- ]; then scenarios=$1; shift; fi
  if [ $# -gt 0 ] && [ "$1" != -- ]; then repeats=$1; shift; fi
  [ "${1:-}" != -- ] || shift
  local -a extra=("$@")
  [ -z "${HAYSTACK:-}" ] || extra+=(--haystack "$HAYSTACK")
  [ -z "${QUALITY_BASELINE:-}" ] || extra+=(--quality-baseline "$QUALITY_BASELINE")
  # argparse accepts repeated options with the last value winning. Validate the
  # effective job, including ordinary benchmark options forwarded after `--`.
  local option_index option
  for ((option_index=0; option_index<${#extra[@]}; option_index++)); do
    option=${extra[option_index]}
    case $option in
      --scenarios) option_index=$((option_index + 1)); scenarios=${extra[option_index]:-} ;;
      --scenarios=*) scenarios=${option#*=} ;;
      --repeats) option_index=$((option_index + 1)); repeats=${extra[option_index]:-} ;;
      --repeats=*) repeats=${option#*=} ;;
    esac
  done
  # Warmups alone must never look like a completed evaluation, and invalid
  # configuration must not disturb an existing server.
  [[ $scenarios =~ [^,[:space:]] ]] || { die "scenarios must contain at least one nonempty scenario"; return 2; }
  [[ $repeats =~ ^[1-9][0-9]*$ ]] || { die "repeats must be a positive integer"; return 2; }
  log=$OUT/logs/$engine-$mode-$label.log
  # This command is shared by ordinary bench and callers with an explicit
  # readiness/headroom gate. Keep all fixture and sampling arguments together.
  BENCH_COMMAND=("$PY" "$ROOT/eng/qwen38-engine-bench.py" --engine "$engine" --mode "$mode" --label "$label"
    --url "http://127.0.0.1:$(port_of "$engine")" --out "$OUT/results/$engine-$mode-$label.jsonl" --overwrite
    --scenarios "$scenarios" --repeats "$repeats" --warmup "$WARMUP" --nonce-seed "$NONCE_SEED" --tokenizer "$TOKENIZER"
    --server-log "$log" --manifest "$(manifest_of "$engine" "$mode" "$label")" "${extra[@]}")
}

bench_command() {
  prepare_bench_command "$@" || return $?
  "$CONTROL_PY" - "${BENCH_COMMAND[@]}" <<'COMMAND'
import json, sys
print(json.dumps(sys.argv[1:]))
COMMAND
}

bench() {
  local engine=$1 mode=$2 label=$3 status
  prepare_bench_command "$@" || return $?
  stop || return $?
  start "$engine" "$mode" "$label" || return $?
  mkdir -p "$OUT/results" || { stop; return 1; }
  # Any benchmark failure is retained as a failure even when cleanup succeeds.
  "${BENCH_COMMAND[@]}"
  status=$?
  stop || status=1
  return "$status"
}

case ${1:-} in
start) shift; [ $# -ge 2 ] || { die "start <engine> <plain|dflash2> [label]"; exit 2; }; start "$@" ;;
stop) stop ;;
bench) shift; [ $# -ge 3 ] || { die "bench <engine> <plain|dflash2> <label> [scenarios] [repeats] [-- arguments...]"; exit 2; }; bench "$@" ;;
bench-command) shift; [ $# -ge 3 ] || { die "bench-command <engine> <plain|dflash2> <label> [scenarios] [repeats] [-- arguments...]"; exit 2; }; bench_command "$@" ;;
matrix)
  shift; [ "${1:-}" != -- ] || shift
  failures=0
  for mode in plain dflash2; do
    for engine in ${ENGINES:-tensorsharp llamacpp vllm sglang velox ninfer infernix ignis}; do
      bench "$engine" "$mode" "${LABEL:-main}" -- "$@" || failures=$((failures + 1))
    done
  done
  bench tensorsharp plain control "${CONTROL_SCENARIOS:-short_chat,long_prompt_16k,multi_turn,long_generation}" -- "$@" || failures=$((failures + 1))
  [ "$failures" -eq 0 ] || { echo "$failures benchmark configurations failed; inspect logs/results (not passing validation)." >&2; exit 1; } ;;
*) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
