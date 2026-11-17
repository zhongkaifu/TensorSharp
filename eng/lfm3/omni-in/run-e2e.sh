#!/usr/bin/env bash
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
#
# LFM3-Omni end to end against llama.cpp: start llama-server (Metal) and TensorSharp.Server.Host on <backend> with the
# same GGUF + mmproj, run compare.py (TOOL=ttft: ttft.py, cold TTFT or --concurrent), stop both.
#
#   eng/lfm3/omni-in/run-e2e.sh <ts-backend> <out-dir> [compare.py args...]
#   e.g. eng/lfm3/omni-in/run-e2e.sh ggml_metal artifacts/lfm3/omni-in/e2e/metal
#        eng/lfm3/omni-in/run-e2e.sh mlx artifacts/lfm3/omni-in/e2e/mlx --reuse-llama artifacts/lfm3/omni-in/e2e/metal/llama.json
#
# Environment: LLAMA (llama.cpp checkout, default ~/work/llama.cpp-lfm3), MODELS (default ~/work/models/lfm3), MODEL
# (default LFM3-Omni-2B-Q4_K_M.gguf), MMPROJ (default mmproj-LFM3-Omni-2B-Q8_0.gguf), MEDIA (manifest, default
# artifacts/lfm3/omni-in/media/manifest.json), LPORT / TPORT (8181 / 5181), LLAMA_NP (llama-server slots, 1). llama-server always runs: a --reuse-llama
# run still tokenizes with it. Build first with TENSORSHARP_GGML_NO_UPDATE=1.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
backend="${1:?ts backend}"; out="${2:?out dir}"; shift 2
LLAMA="${LLAMA:-$HOME/work/llama.cpp-lfm3}"
MODELS="${MODELS:-$HOME/work/models/lfm3}"
MODEL="${MODEL:-LFM3-Omni-2B-Q4_K_M.gguf}"
MMPROJ="${MMPROJ:-mmproj-LFM3-Omni-2B-Q8_0.gguf}"
MEDIA="${MEDIA:-$repo/artifacts/lfm3/omni-in/media/manifest.json}"
LPORT="${LPORT:-8181}"; TPORT="${TPORT:-5181}"
mkdir -p "$out"
pids=()
cleanup() { for p in "${pids[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT

"$LLAMA/build/bin/llama-server" -m "$MODELS/$MODEL" -mm "$MODELS/$MMPROJ" --port "$LPORT" -ngl 99 --jinja -c 32768 -np "${LLAMA_NP:-1}" \
  > "$out/llama-server.log" 2>&1 &
pids+=($!)
MAX_CONTEXT="${MAX_CONTEXT:-32768}" dotnet "$repo/TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll" \
  --model "$MODELS/$MODEL" --mmproj "$MODELS/$MMPROJ" --backend "$backend" --port "$TPORT" --no-skills --no-multi-agent \
  > "$out/ts-server.log" 2>&1 &
pids+=($!)

python3 "$here/wait_ready.py" "http://127.0.0.1:$LPORT/health" "http://127.0.0.1:$TPORT/v1/models" \
  || { echo "a server did not start; see $out/*.log"; exit 1; }

tool="${TOOL:-compare}"
python3 "$here/$tool.py" --media "$MEDIA" --llama "http://127.0.0.1:$LPORT" --ts "http://127.0.0.1:$TPORT" --out "$out" "$@" \
  2>&1 | tee "$out/$tool.log"
