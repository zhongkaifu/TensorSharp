#!/usr/bin/env bash
# Start one chat server, send the repetition-study prompt set through one or more surfaces, stop the server.
#
#   eng/lfm3/loop/serve-and-run.sh OUT_DIR ENGINE "SERVER ARGS" RUN1 [RUN2 ...]
#
# ENGINE: ts      TensorSharp.Server.Host (TS_SERVER_BIN, default TensorSharp.Server.Host/bin), SERVER ARGS appended
#                 (e.g. "--config config/lfm3-omni-2b.json"); agent features stay on unless the args turn them off,
#                 as a user's server has them;
#         llama   llama-server from LLAMA_CPP_BIN (default ~/work/llama.cpp-lfm3/build/bin) with --jinja, -ngl 99,
#                 the GGUF (and mmproj) named in SERVER ARGS ("-m MODEL --mmproj MMPROJ ...");
#         tagent  eng/validation/TensorAgentHost (TA_HOST_DLL), SERVER ARGS are its options (--weights, --mmproj ...).
# Each RUN is "NAME|run.py args" (without --base/--out): the rows go to OUT_DIR/NAME.jsonl, the log to NAME.log.
# "NAME|!command" runs the command instead (bash -c, with BASE = the server URL), its output to NAME.log.
# PORT defaults to 5191. MEDIA (default artifacts/loop/media) is passed to run.py.
set -uo pipefail
here="$(cd "$(dirname "$0")/../../.." && pwd)"
out="$1"; engine="$2"; sargs="$3"; shift 3
mkdir -p "$out"
port="${PORT:-5191}"
media="${MEDIA:-$here/artifacts/loop/media}"
cd "$here"
case "$engine" in
  ts)
    bin="${TS_SERVER_BIN:-$here/TensorSharp.Server.Host/bin}"
    # shellcheck disable=SC2086
    MAX_CONTEXT="${MAX_CONTEXT:-32768}" dotnet "$bin/TensorSharp.Server.Host.dll" --host 127.0.0.1 --port "$port" $sargs \
      > "$out/server.log" 2>&1 &
    base="http://127.0.0.1:$port"; ready="$base/api/tags" ;;
  llama)
    lb="${LLAMA_CPP_BIN:-$HOME/work/llama.cpp-lfm3/build/bin}"
    # shellcheck disable=SC2086
    "$lb/llama-server" --host 127.0.0.1 --port "$port" --jinja -ngl 99 -c "${CTX:-32768}" -np 1 $sargs \
      > "$out/server.log" 2>&1 &
    base="http://127.0.0.1:$port"; ready="$base/v1/models" ;;
  tagent)
    rm -f "$out/connection.json" "$out/stop"
    # shellcheck disable=SC2086
    dotnet "${TA_HOST_DLL:?TA_HOST_DLL names TensorAgentHost.dll}" --root "$out/host" --skills "${TA_SKILLS:-$here/TensorAgent/skills}" \
      --stop-file "$out/stop" $sargs > "$out/server.log" 2>&1 &
    base=""; ready="" ;;
  *) echo "unknown engine $engine" >&2; exit 64 ;;
esac
spid=$!
t0=$(date +%s)
while :; do
  if ! kill -0 "$spid" 2>/dev/null; then echo "server exited"; tail -40 "$out/server.log"; exit 1; fi
  if [ "$engine" = tagent ]; then
    [ -s "$out/host/connection.json" ] && break
  else
    curl -sf "$ready" >/dev/null 2>&1 && break
  fi
  [ $(( $(date +%s) - t0 )) -gt 600 ] && { echo "server not ready in 600 s"; kill "$spid"; exit 1; }
  sleep 1
done
echo "ready after $(( $(date +%s) - t0 )) s"
rc=0
for run in "$@"; do
  name="${run%%|*}"; rargs="${run#*|}"
  if [ "${rargs:0:1}" = "!" ]; then
    BASE="$base" bash -c "${rargs:1}" > "$out/$name.log" 2>&1 || rc=$?
    echo "$name: $(tail -1 "$out/$name.log")"
    continue
  fi
  if [ "$engine" = tagent ]; then target=(--connection "$out/host/connection.json"); else target=(--base "$base"); fi
  # shellcheck disable=SC2086
  python3 "$here/eng/lfm3/loop/run.py" "${target[@]}" --media "$media" --setting "$name" --out "$out/$name.jsonl" $rargs \
    > "$out/$name.log" 2>&1 || rc=$?
  echo "$name: $(grep -c . "$out/$name.jsonl" 2>/dev/null) rows, $(grep -c '"loop": true' "$out/$name.jsonl" 2>/dev/null) loops"
done
if [ "$engine" = tagent ]; then touch "$out/stop"; sleep 3; fi
kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
exit $rc
