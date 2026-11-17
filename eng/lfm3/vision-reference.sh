#!/usr/bin/env bash
# Build eng/lfm3/lfm3-vision-ref.cpp against a llama.cpp build (libmtmd + libllama, sources unchanged) in a
# scratch directory, then run it: llama.cpp's LFM3-Omni vision preprocessing, layout and encode for the given
# cases (see the .cpp header for the case syntax and the files it writes).
#
# usage: eng/lfm3/vision-reference.sh <llama.cpp tree> <LM gguf> <mmproj gguf> <out dir> [--cpu] [--reps N] case...
#   e.g. eng/lfm3/vision-reference.sh ~/work/llama.cpp-lfm3 ~/work/models/lfm3/LFM3-Omni-2B-Q8_0.gguf \
#          ~/work/models/lfm3/mmproj-LFM3-Omni-2B-Q8_0.gguf artifacts/lfm3/vision/ref-q8 synth:wide:1000:250:2
# The binary is built once per llama.cpp tree into <out dir>/../.build (LFM3_VISION_REF_BUILD overrides).
set -euo pipefail
tree=${1:?llama.cpp tree}; model=${2:?LM gguf}; mmproj=${3:?mmproj gguf}; out=${4:?out dir}; shift 4
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
build=${LFM3_VISION_REF_BUILD:-$(cd "$out/.." && pwd)/.build}
mkdir -p "$build"
bin="$build/lfm3-vision-ref"
if [ ! -x "$bin" ] || [ "$here/lfm3-vision-ref.cpp" -nt "$bin" ]; then
    c++ -std=c++17 -O2 -o "$bin" "$here/lfm3-vision-ref.cpp" \
        -I"$tree/include" -I"$tree/ggml/include" -I"$tree/tools/mtmd" \
        -L"$tree/build/bin" -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,"$tree/build/bin"
fi
"$bin" "$model" "$mmproj" "$out" "$@"
