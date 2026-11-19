#!/usr/bin/env bash
# LFM3-Omni vision: TensorSharp against llama.cpp, end to end. For each Omni mmproj it runs llama.cpp's mtmd
# (eng/lfm3/vision-reference.sh) and TensorSharp's Lfm3VisionProbe over the same cases, then diffs them
# (eng/lfm3/vision-compare.py): layouts, graph inputs, every embedding row, encode times.
#
# usage: eng/lfm3/vision-validate.sh <llama.cpp tree> <model dir> <out dir> [--backend ggml_metal|ggml_cpu|mlx|cpu] [--reps N]
#   <model dir> holds LFM3-Omni-2B-Q8_0.gguf and mmproj-LFM3-Omni-2B-{Q8_0,F16}.gguf; MMPROJ_TYPES="Q8_0 F16" overrides.
# The cases: the six synthetic pictures and the synthetic clip of Fixtures/Lfm3Vision, the alpha fixture PNG,
# llama.cpp's tools/mtmd/test-1.jpeg and test-3.mp4 (decoded by each side), and test-3.mp4's frames as TensorSharp
# decoded and resized them, encoded by llama.cpp (frames:, so both encode the same pixels).
# Fixtures: eng/lfm3/vision-fixtures.py packs <out dir>/ref-<type> into InferenceWeb.Tests/Fixtures/Lfm3Vision.
set -euo pipefail
tree=${1:?llama.cpp tree}; models=${2:?model dir}; out=${3:?out dir}; shift 3
backend=ggml_metal; reps=5
while [ $# -gt 0 ]; do
    case "$1" in
        --backend) backend=$2; shift 2 ;;
        --reps) reps=$2; shift 2 ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
repo=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$out"
out=$(cd "$out" && pwd)
cases=(synth:small:100:80:1 synth:wide:1000:250:2 synth:tall:360:1280:3 synth:square:512:512:4 synth:hd:1920:1080:5
    synth:cap:3000:4000:6 synthvideo:svideo:320:256:75:30 "file:alpha:$repo/InferenceWeb.Tests/Fixtures/DeepSeek41Vision/alpha_ignored.png"
    "file:photo:$tree/tools/mtmd/test-1.jpeg" "video:mp4:$tree/tools/mtmd/test-3.mp4")
TENSORSHARP_GGML_NO_UPDATE=1 dotnet build "$repo/eng/lfm3/Lfm3VisionProbe" -c Release -v q -nologo >/dev/null
probe="$repo/eng/lfm3/Lfm3VisionProbe/bin/Release/net10.0/Lfm3VisionProbe"
for type in ${MMPROJ_TYPES:-Q8_0 F16}; do
    mmproj="$models/mmproj-LFM3-Omni-2B-$type.gguf"
    "$probe" "$mmproj" "$out/ts-$type" --backend "$backend" --reps "$reps" "${cases[@]}" > "$out/ts-$type.log" 2>&1
    "$repo/eng/lfm3/vision-reference.sh" "$tree" "$models/LFM3-Omni-2B-Q8_0.gguf" "$mmproj" "$out/ref-$type" --reps "$reps" \
        "${cases[@]}" "frames:frames:$out/ts-$type/mp4.frames" > "$out/ref-$type.log" 2>&1
    echo "## $type mmproj, TensorSharp $backend vs llama.cpp (GPU)"
    python3 "$repo/eng/lfm3/vision-compare.py" "$out/ref-$type" "$out/ts-$type" small wide tall square hd cap svideo alpha photo mp4 mp4:frames
done
