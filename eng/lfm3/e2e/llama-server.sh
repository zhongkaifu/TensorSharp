#!/usr/bin/env bash
# Start llama-server as the LFM3 reference for eng/lfm3/e2e (same GGUF, OpenAI API).
#
#   eng/lfm3/e2e/llama-server.sh PORT LOGFILE MODEL [extra llama-server args...]
#
# LLAMA_CPP_BIN names the llama.cpp build/bin directory (default: ../llama.cpp/build/bin
# beside the repository). --jinja renders the GGUF template; --reasoning-format deepseek
# returns the reasoning as reasoning_content (as TensorSharp does); --no-reasoning-preserve
# (when the build has it) keeps past turns' reasoning out of the prompt, as HF renders the
# template. Context defaults to 32768 per slot (CTX); llama-server picks its slot count
# (4, unified KV) unless -np is given. One request at a time decodes alone either way.
set -euo pipefail
port="$1"; log="$2"; model="$3"; shift 3
bin="${LLAMA_CPP_BIN:-$(cd "$(dirname "$0")/../../.." && pwd)/../llama.cpp/build/bin}"
extra=()
if "$bin/llama-server" --help 2>&1 | grep -q -- "--no-reasoning-preserve"; then
  extra+=(--no-reasoning-preserve)
fi
exec "$bin/llama-server" -m "$model" --host 127.0.0.1 --port "$port" --jinja \
  --reasoning-format deepseek -c "${CTX:-32768}" -ngl 99 -fa on "${extra[@]}" "$@" >"$log" 2>&1
