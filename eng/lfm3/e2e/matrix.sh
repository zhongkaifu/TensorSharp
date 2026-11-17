#!/usr/bin/env bash
# One model/backend cell of the LFM3 end-to-end matrix: start TensorSharp.Server.Host on MODEL and
# BACKEND, run the functional scenarios (e2e.py) and the IFEval-style and function-calling sets,
# optionally the same two sets on llama-server with the same GGUF (REF=1), then stop the servers.
#
#   eng/lfm3/e2e/matrix.sh LABEL MODEL BACKEND OUT_DIR [PORT] [REF_PORT]
#
# TS_SERVER_BIN (a copy of TensorSharp.Server.Host/bin, one per concurrently running server) and
# REF=1 / LLAMA_CPP_BIN are read from the environment. SCENARIOS narrows e2e.py (comma list).
set -uo pipefail
label="$1"; model="$2"; backend="$3"; out="$4"; port="${5:-5737}"; ref_port="${6:-5738}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
wait_up() { for _ in $(seq 1 300); do curl -s -m 2 "$1" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

"$here/ts-server.sh" "$port" "$out/server-$label.log" --sampling-precedence request --model "$model" --backend "$backend" &
ts_pid=$!
trap 'kill $ts_pid ${ref_pid:-} 2>/dev/null' EXIT
if ! wait_up "http://127.0.0.1:$port/api/tags"; then echo "$label: server did not start"; kill $ts_pid; exit 1; fi
ref_args=()
if [ "${REF:-0}" = "1" ]; then
  "$here/llama-server.sh" "$ref_port" "$out/llama-$label.log" "$model" &
  ref_pid=$!
  wait_up "http://127.0.0.1:$ref_port/health" && ref_args=(--ref "http://127.0.0.1:$ref_port")
fi
python3 -I "$here/e2e.py" --url "http://127.0.0.1:$port" ${ref_args[@]+"${ref_args[@]}"} --label "$label" \
  --out "$out/e2e-$label.jsonl" ${SCENARIOS:+--scenarios "$SCENARIOS"} > "$out/e2e-$label.log" 2>&1
python3 -I "$here/ifeval.py" --url "http://127.0.0.1:$port" --out "$out/ifeval-$label-ts.jsonl" > "$out/ifeval-$label-ts.log" 2>&1
python3 -I "$here/toolcall_eval.py" --url "http://127.0.0.1:$port" --out "$out/tools-$label-ts.jsonl" > "$out/tools-$label-ts.log" 2>&1
if [ ${#ref_args[@]} -gt 0 ]; then
  python3 -I "$here/ifeval.py" --url "http://127.0.0.1:$ref_port" --out "$out/ifeval-$label-llama.jsonl" > "$out/ifeval-$label-llama.log" 2>&1
  python3 -I "$here/toolcall_eval.py" --url "http://127.0.0.1:$ref_port" --out "$out/tools-$label-llama.jsonl" > "$out/tools-$label-llama.log" 2>&1
fi
kill $ts_pid 2>/dev/null; pkill -P $ts_pid 2>/dev/null
[ -n "${ref_pid:-}" ] && { kill $ref_pid 2>/dev/null; pkill -P $ref_pid 2>/dev/null; }
wait 2>/dev/null
echo "== $label ($backend, $(basename "$model"))"
grep -E "PASS|FAIL" "$out/e2e-$label.log" | awk '{print $1, $2}' | tr '\n' ' '; echo
echo "ifeval ts: $(tail -1 "$out/ifeval-$label-ts.log")"
echo "tools  ts: $(tail -1 "$out/tools-$label-ts.log")"
if [ ${#ref_args[@]} -gt 0 ]; then
  echo "ifeval llama: $(tail -1 "$out/ifeval-$label-llama.log")"
  echo "tools  llama: $(tail -1 "$out/tools-$label-llama.log")"
fi
