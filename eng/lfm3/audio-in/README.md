# LFM3-Omni audio input: llama.cpp comparison tools

Tools that check TensorSharp's LFM3-Omni audio input (`Lfm3AudioPreprocessor`, `Lfm3AudioEncoder`) against llama.cpp's
mtmd on the same mmproj. Outputs belong in the ignored `artifacts/lfm3/audio-in/`.

- `lfm3-audio-ref.cpp` uses only the public libmtmd / libllama API, so the llama.cpp sources stay unchanged.
  - `dump` writes each clip's encoder input (the log-mel), the downsampler output, every layer's operator norm, the final
    norm and the LM rows, read through mtmd's eval callback. It also records llama.cpp's encode time, with and without the
    mel computation.
  - `asr` transcribes greedily with the LM. Given a `ROWS.f32`, it decodes TensorSharp's rows instead of llama.cpp's own.
- `Lfm3AudioProbe` runs on the TensorSharp side. It links the test clips from `InferenceWeb.Tests/Lfm3AudioSignals.cs`
  and has these commands:
  - `signals`: writes the clips as raw F32 PCM.
  - `compare`: compares every dumped value and times both implementations.
  - `rows`: writes TensorSharp's rows for `asr`.
  - `stream`: compares live-stream rows with the whole clip's rows.
  - `shapes`: measures the cost of a new clip length.
  - `frontend`: times the front end.
  - `fixture`: regenerates `InferenceWeb.Tests/Fixtures/Lfm3Audio/llamacpp-reference.json`.

```sh
export TENSORSHARP_GGML_NO_UPDATE=1
LLAMA=/path/to/llama.cpp          # a build tree with build/bin/libmtmd.dylib (Metal)
M=/path/to/models/lfm3            # LFM3-Omni-2B-Q8_0.gguf, mmproj-LFM3-Omni-2B-{Q8_0,F16}.gguf
A=artifacts/lfm3/audio-in
mkdir -p $A/bin
c++ -std=c++17 -O2 eng/lfm3/audio-in/lfm3-audio-ref.cpp -I$LLAMA/include -I$LLAMA/ggml/include -I$LLAMA/tools/mtmd \
    -L$LLAMA/build/bin -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,$LLAMA/build/bin -o $A/bin/lfm3-audio-ref
dotnet build eng/lfm3/audio-in/Lfm3AudioProbe -c Release -o $A/probe
dotnet $A/probe/Lfm3AudioProbe.dll signals $A/clips

# llama.cpp reference (the LM file is needed for its vocabulary only)
$A/bin/lfm3-audio-ref dump $M/LFM3-Omni-2B-Q8_0.gguf $M/mmproj-LFM3-Omni-2B-Q8_0.gguf $A/ref-q8 5 $A/clips/*.f32
$A/bin/lfm3-audio-ref dump $M/LFM3-Omni-2B-Q8_0.gguf $M/mmproj-LFM3-Omni-2B-F16.gguf $A/ref-f16 5 $A/clips/*.f32

# TensorSharp against it: ggml_metal | ggml_cpu | cpu (managed) | mlx
dotnet $A/probe/Lfm3AudioProbe.dll compare $M/mmproj-LFM3-Omni-2B-Q8_0.gguf $A/ref-q8 ggml_metal 5 $A/cmp-ggml_metal-q8.json

# end to end: llama.cpp's LM transcribes TensorSharp's rows
dotnet $A/probe/Lfm3AudioProbe.dll rows $M/mmproj-LFM3-Omni-2B-Q8_0.gguf speech_6s ggml_metal $A/ts-rows.f32
$A/bin/lfm3-audio-ref asr $M/LFM3-Omni-2B-Q8_0.gguf $M/mmproj-LFM3-Omni-2B-Q8_0.gguf $A/clips/speech_6s.f32 en 200 $A/ts-rows.f32

# the test fixture (after both dumps)
dotnet $A/probe/Lfm3AudioProbe.dll fixture InferenceWeb.Tests/Fixtures/Lfm3Audio/llamacpp-reference.json \
    mmproj-LFM3-Omni-2B-Q8_0.gguf=$A/ref-q8 mmproj-LFM3-Omni-2B-F16.gguf=$A/ref-f16
```

The tests read the fixture:

```sh
TS_TEST_GGML_BACKEND=metal TS_LFM3_MMPROJ=$M/mmproj-LFM3-Omni-2B-Q8_0.gguf \
    dotnet test InferenceWeb.Tests -c Release --filter "FullyQualifiedName~Lfm3Audio"
```

- With `TS_TEST_GGML_BACKEND=metal`, the run covers the GGML metal, managed CPU and MLX lanes.
- With `=cpu`, it covers the GGML cpu lane.
- The front-end tests need no model.
- `FfmpegDecodesFlacM4aAndAVideosSoundtrack` needs ffmpeg on PATH or `TS_FFMPEG`.
