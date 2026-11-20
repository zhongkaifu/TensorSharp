#!/usr/bin/env bash
# Start TensorSharp.Server.Host for the LFM3 end-to-end suite (eng/lfm3/e2e/e2e.py).
#
#   eng/lfm3/e2e/ts-server.sh PORT LOGFILE [extra server args...]
#
# Uses config/lfm3-1.7b.json (Q4_K_M, ggml_metal, the card's sampling) unless the extra
# arguments override it (--model, --backend, --temperature ...; the command line wins over
# the config). Agent features are off (--no-multi-agent --no-skills) so requests exercise
# the engine and the chat protocol only. MAX_CONTEXT defaults to 32768; export it to test
# longer prompts. TS_SERVER_BIN runs a copy of the build output (cp -R
# TensorSharp.Server.Host/bin somewhere) so a rebuild cannot replace a running server's
# assemblies; servers running at the same time need a copy each (the host locks its Studio
# storage beside the binaries). Runs in the foreground; the caller backgrounds and kills it.
set -euo pipefail
here="$(cd "$(dirname "$0")/../../.." && pwd)"
port="$1"; log="$2"; shift 2
export TENSORSHARP_MODELS="${TENSORSHARP_MODELS:-$here/../models}"
export MAX_CONTEXT="${MAX_CONTEXT:-32768}"
bin="${TS_SERVER_BIN:-$here/TensorSharp.Server.Host/bin}"
cd "$here"
exec dotnet "$bin/TensorSharp.Server.Host.dll" \
  --config config/lfm3-1.7b.json --host 127.0.0.1 --port "$port" \
  --no-multi-agent --no-skills "$@" >"$log" 2>&1
