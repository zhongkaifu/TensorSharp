#!/usr/bin/env bash
# LFM3-1.7B's text decoder against llama.cpp on one Linux GPU machine, on TensorSharp's three GPU lanes: teacher-forced
# logits and perplexity, each lane against llama.cpp's own run on the same GPU (ggml_cuda and direct cuda against
# llama.cpp's CUDA build, ggml_vulkan against its Vulkan build). Outputs go to OUT (keep it under the ignored
# artifacts/lfm3/).
#
#   eng/lfm3/gpu-matrix.sh OUT MODELS_DIR CASES_DIR CORPUS [MODEL_FILE ...]
#
# MODELS_DIR holds the GGUFs (default list: the five LFM3-1.7B quantizations); CASES_DIR the forced-logit prompts (UTF-8
# .txt, special tokens parsed; the same five as the Mac runs); CORPUS the perplexity text. LLAMA (default ../llama.cpp
# beside the repository) holds build-cuda/ and build-vk/ trees of an unchanged llama.cpp. PHASES (default
# "build ref ts ppl") and LANES (default "ggml_cuda cuda ggml_vulkan") narrow a run; STEPS (64), PPL_CTX (1024) and
# PPL_CHUNKS (32) are the Mac runs' settings.
#   ref-<model>-<cuda|vk>/   llamacpp-forced: STEPS greedy steps per case, every step's logits (NGL 99, F16 KV)
#   ts-<model>-<lane>/       ForcedLogitProbe --reference: TensorSharp through the same histories (metrics.json)
#   ppl-<model>-<side>.log   llama-perplexity and Lfm3Probe ppl (no BOS, second halves scored)
#   summary.txt              eng/lfm3/forced-summary.py over every ts-* run, and every perplexity's final estimate
set -uo pipefail
out="$1"; models="$2"; cases="$3"; corpus="$4"; shift 4
here="$(cd "$(dirname "$0")/../.." && pwd)"
llama="${LLAMA:-$here/../llama.cpp}"
phases="${PHASES:-build ref ts ppl}"
lanes="${LANES:-ggml_cuda cuda ggml_vulkan}"
steps="${STEPS:-64}"; ctx="${PPL_CTX:-1024}"; chunks="${PPL_CHUNKS:-32}"
list=("$@")
[ ${#list[@]} -eq 0 ] && list=(LFM3-1.7B-Q4_K_M.gguf LFM3-1.7B-Q8_0.gguf LFM3-1.7B-BF16.gguf LFM3-1.7B-QAD-Q4_0.gguf LFM3-1.7B-NVFP4.gguf)
mkdir -p "$out"
probe="$here/eng/ForcedLogitProbe/bin/Release/net10.0/ForcedLogitProbe.dll"
lfm3probe="$here/eng/lfm3/Lfm3Probe/bin/Release/net10.0/Lfm3Probe.dll"
side_of() { case "$1" in ggml_vulkan) echo vk ;; *) echo cuda ;; esac; }
has() { [[ " $phases " == *" $1 "* ]]; }

if has build; then
  export TENSORSHARP_GGML_NO_UPDATE=1
  dotnet build "$here/eng/ForcedLogitProbe" -c Release -nologo > "$out/build-probe.log" 2>&1 || { echo "ForcedLogitProbe build failed"; exit 1; }
  dotnet build "$here/eng/lfm3/Lfm3Probe" -c Release -nologo > "$out/build-lfm3probe.log" 2>&1 || { echo "Lfm3Probe build failed"; exit 1; }
  for side in cuda vk; do
    lib="$llama/build-$side/bin"
    [ -d "$lib" ] || continue
    c++ -std=c++17 -O2 "$here/eng/lfm3/llamacpp-forced.cpp" -I"$llama/include" -I"$llama/ggml/include" \
      -L"$lib" -lllama -lggml -lggml-base -Wl,-rpath,"$lib" -o "$out/llamacpp-forced-$side" || { echo "llamacpp-forced ($side) build failed"; exit 1; }
  done
fi

for m in "${list[@]}"; do
  name="${m%.gguf}"; name="${name#LFM3-1.7B-}"
  if has ref; then
    for side in cuda vk; do
      [ -x "$out/llamacpp-forced-$side" ] || continue
      mkdir -p "$out/ref-$name-$side"
      "$out/llamacpp-forced-$side" "$models/$m" "$out/ref-$name-$side" "$steps" 99 "$cases"/*.txt > "$out/ref-$name-$side.log" 2>&1
      echo "ref $name $side exit $?"
    done
  fi
  if has ts; then
    for lane in $lanes; do
      side=$(side_of "$lane")
      [ -d "$out/ref-$name-$side" ] || { echo "no reference $out/ref-$name-$side"; continue; }
      dotnet "$probe" "$models/$m" "$out/ts-$name-$lane" "$lane" --reference "$out/ref-$name-$side" --steps "$steps" > "$out/ts-$name-$lane.log" 2>&1
      echo "ts $name $lane exit $?"
    done
  fi
  if has ppl; then
    for side in cuda vk; do
      [ -x "$llama/build-$side/bin/llama-perplexity" ] || continue
      "$llama/build-$side/bin/llama-perplexity" -m "$models/$m" -f "$corpus" -c "$ctx" --chunks "$chunks" -ngl 99 > "$out/ppl-$name-llama-$side.log" 2>&1
      echo "ppl $name llama-$side exit $?"
    done
    for lane in $lanes; do
      dotnet "$lfm3probe" ppl "$models/$m" "$lane" "$corpus" "$ctx" "$chunks" > "$out/ppl-$name-ts-$lane.log" 2>&1
      echo "ppl $name ts-$lane exit $?"
    done
  fi
done

{
  python3 -I "$here/eng/lfm3/forced-summary.py" "$out"/ts-*/ 2>/dev/null
  for f in "$out"/ppl-*.log; do
    [ -f "$f" ] && echo "$(basename "$f" .log): $(grep -h "Final estimate" "$f" | tail -1)"
  done
} > "$out/summary.txt"
cat "$out/summary.txt"
