#!/usr/bin/env bash
# llama.cpp's LFM3-Omni vision encode time for square images of given sizes: the reference
# NetworkGraphRealWeightsTests.MetalVisionTowerSpeed is compared against (a 512 x 512 image is a
# 32 x 32 patch grid, 1024 x 1024 a 64 x 64 one).
#
# usage: eng/lfm3/mtmd-encode-timing.sh <llama.cpp bin dir> <LM gguf> <mmproj gguf> <source image> [reps] [sizes...]
#   e.g. eng/lfm3/mtmd-encode-timing.sh ~/work/llama.cpp-lfm3/build/bin \
#          ~/work/models/lfm3/LFM3-Omni-2B-Q8_0.gguf ~/work/models/lfm3/mmproj-LFM3-Omni-2B-F16.gguf imgs/assistant_logo_2.png 3 512 1024
# Resized images go to a temporary directory (macOS sips); prints "size rep: N ms" per run.
set -euo pipefail
bin=${1:?llama.cpp bin dir}; model=${2:?LM gguf}; mmproj=${3:?mmproj gguf}; source=${4:?source image}
reps=${5:-3}; shift $(( $# < 5 ? $# : 5 ))
sizes=("$@"); [ ${#sizes[@]} -eq 0 ] && sizes=(512 1024)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
for size in "${sizes[@]}"; do
    sips -z "$size" "$size" "$source" --out "$work/img$size.png" >/dev/null
    for rep in $(seq 1 "$reps"); do
        ms=$("$bin/llama-mtmd-cli" -m "$model" --mmproj "$mmproj" --image "$work/img$size.png" -p "Describe." -n 1 --temp 0 2>&1 \
            | sed -n 's/.*mtmd batch encoding done in \([0-9]*\) ms.*/\1/p')
        echo "${size}x${size} rep $rep: ${ms:-?} ms"
    done
done
