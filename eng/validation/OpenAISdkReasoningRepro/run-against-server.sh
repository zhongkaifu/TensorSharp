#!/usr/bin/env bash
# Start TensorSharp.Server.Host on a model, run the OpenAI SDK reasoning repro against it,
# and stop the server. Evidence (server log, repro output, wire traffic) goes to OUT.
#
#   eng/validation/OpenAISdkReasoningRepro/run-against-server.sh MODEL.gguf OUT [repro args...]
#
# Env: PORT (5386), BACKEND (ggml_metal), MAX_CONTEXT (16384), SERVER_ARGS (extra server
# flags), REPRO_DLL (a prebuilt repro; default: build and run it with dotnet run -c Release),
# KEEP_PREFIX_CACHE (1 keeps the server's prefix-cache files).
set -euo pipefail
model=${1:?model .gguf}
out=${2:?output directory}
shift 2
root=$(cd "$(dirname "$0")/../../.." && pwd)
port=${PORT:-5386}
backend=${BACKEND:-ggml_metal}
mkdir -p "$out"
out=$(cd "$out" && pwd)
name=$(basename "$model" .gguf)

if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "port $port is already in use" >&2
  exit 2
fi

# The server works in OUT. Its persisted prefix-cache checkpoints (about 0.6 GB for a 9B
# model) go there too and are deleted with the server, unless KEEP_PREFIX_CACHE=1.
export TENSORSHARP_PREFIX_CACHE_DIR="$out/prefix-cache"
export MAX_CONTEXT=${MAX_CONTEXT:-16384}
(cd "$out" && exec dotnet "$root/TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll" \
    --model "$model" --mmproj none --backend "$backend" --host 127.0.0.1 --port "$port" \
    ${SERVER_ARGS:-}) >"$out/server.log" 2>&1 &
server=$!
stop() {
  kill $server 2>/dev/null; wait $server 2>/dev/null || true
  [ "${KEEP_PREFIX_CACHE:-0}" = 1 ] || rm -rf "$out/prefix-cache"
}
trap stop EXIT

for _ in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1; then break; fi
  if ! kill -0 $server 2>/dev/null; then echo "server exited; see $out/server.log" >&2; exit 3; fi
  sleep 1
done
curl -sf "http://127.0.0.1:$port/v1/models" >"$out/models.json"
echo "server up: $(cat "$out/models.json")"

status=0
if [ -n "${REPRO_DLL:-}" ]; then
  dotnet "$REPRO_DLL" --endpoint "http://127.0.0.1:$port/v1" --model "$name" --wire-log "$out/wire" "$@" \
    2>&1 | tee "$out/repro.txt" || status=$?
else
  dotnet run -c Release --project "$root/eng/validation/OpenAISdkReasoningRepro" -- \
    --endpoint "http://127.0.0.1:$port/v1" --model "$name" --wire-log "$out/wire" "$@" \
    2>&1 | tee "$out/repro.txt" || status=$?
fi
exit $status
