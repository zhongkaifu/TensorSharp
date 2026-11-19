#!/usr/bin/env bash
# Copyright (c) Zhongkai Fu. All rights reserved.
# Licensed under the BSD-3-Clause license in the repository root.
#
# llama.cpp references for TensorSharp's LFM3-Omni speech output (see speech-ref.cpp):
#   eng/lfm3/speech-ref.sh LLAMA_DIR LM_GGUF MMPROJ OUT_DIR [SEED] [TEXT]
# builds speech-ref against LLAMA_DIR's build tree (sources unchanged), runs it (fixture.json, PCM, timings) and runs
# llama-tts on the same text and seed for the end-to-end reference audio (llama-tts.wav, llama-tts.log).
# The test fixture is InferenceWeb.Tests/Fixtures/Lfm3Speech/llamacpp-reference.json (copy OUT_DIR/fixture.json there).
# LLAMA_BUILD names the build tree under LLAMA_DIR (default build, the Metal build on the Mac; build-cpu for a CPU-only
# reference, which also runs llama-tts on the CPU; build-cuda on a Linux GPU machine).
set -euo pipefail
LLAMA=${1:?llama.cpp checkout with a build/ tree}
LM=${2:?LFM3-Omni LM GGUF}
MMPROJ=${3:?LFM3-Omni mmproj GGUF}
OUT=${4:?output directory}
SEED=${5:-1234}
TEXT=${6:-"Hello! This is a short test of speech output, eighty milliseconds at a time."}
HERE=$(cd "$(dirname "$0")" && pwd)
BUILD="$LLAMA/${LLAMA_BUILD:-build}"
mkdir -p "$OUT"
BIN="$OUT/speech-ref"
if [[ ! -x "$BIN" || "$HERE/speech-ref.cpp" -nt "$BIN" ]]; then
  c++ -std=c++17 -O2 "$HERE/speech-ref.cpp" -I"$LLAMA/include" -I"$LLAMA/ggml/include" -I"$LLAMA/tools/mtmd" \
    -I"$LLAMA/common" -I"$LLAMA/vendor" -L"$BUILD/bin" -lllama -lmtmd -lllama-common -lggml -lggml-base \
    -Wl,-rpath,"$BUILD/bin" -o "$BIN"
fi
echo "llama.cpp $(git -C "$LLAMA" rev-parse --short HEAD) ${LLAMA_BUILD:-build}" > "$OUT/versions.txt"
"$BIN" "$LM" "$MMPROJ" "$OUT" --text "$TEXT" --seed "$SEED" 2> "$OUT/speech-ref.log" || { cat "$OUT/speech-ref.log"; exit 1; }
tail -n 6 "$OUT/speech-ref.log"
"$BUILD/bin/llama-tts" -m "$LM" -mm "$MMPROJ" -p "$TEXT" --seed "$SEED" -ngl 99 --output "$OUT/llama-tts.wav" \
  > "$OUT/llama-tts.log" 2>&1 || { tail -n 20 "$OUT/llama-tts.log"; exit 1; }
grep -E "timings|output audio|generated" "$OUT/llama-tts.log" || true
# llama-tts writes the pipeline's WAV16: the same bytes as speech-ref's pipeline.wav for the same seed and text
if cmp -s "$OUT/llama-tts.wav" "$OUT/pipeline.wav"; then echo "llama-tts.wav == pipeline.wav (byte for byte)"; else echo "llama-tts.wav differs from pipeline.wav"; fi
