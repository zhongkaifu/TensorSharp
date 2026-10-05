# Qwen3.8 parallel image checks

`qwen38-parallel-vision.py` checks two different four-digit image cards and
rectangle colors, image attachment order, and follow-up questions about images
retained in conversation history. Parallel requests begin together and use the
same exact prompt as their sequential controls. The report checks the request
hash, completed answer, and exact content of every turn against its control.
A missing, failed, or incomplete control fails the comparison.
Every concurrency must cover all control requests exactly once; image follow-up
cases must contain both required turns.

Install the shared media runner's Python dependencies into an ignored environment:

```sh
python3 -m venv artifacts/qwen38-validation-venv
artifacts/qwen38-validation-venv/bin/python -m pip install requests Pillow
artifacts/qwen38-validation-venv/bin/python eng/validation/qwen38-parallel-vision.py \
  --prepare --fixtures docs/validation/qwen38-parallel/vision-fixtures
```

The image-only fixture preparation needs no ffmpeg. Optional video scenarios
reuse the existing video fixture generator and require ffmpeg. Generated images,
manifests, reports, and dependency environments belong in ignored
`docs/validation/` or `artifacts/`.

Start a server using the selected model's first shard and the corresponding
projector, with `--backend ggml_metal --mmproj <projector> --no-multi-agent`.
The server requires the explicit projector argument. Run, for example:

```sh
artifacts/qwen38-validation-venv/bin/python eng/validation/qwen38-parallel-vision.py \
  --url http://127.0.0.1:5001 \
  --model Qwen3.8-Flash-Next-Uncensored-IQ2_XXS-00001-of-00002 \
  --weights-id /absolute/path/to/Qwen3.8-Flash-Next-Uncensored-IQ2_XXS-00001-of-00002.gguf \
  --companion-sha256 4fee3582716fe691461576402f178882963e428ca9cbcc84dfeac9c53ebc1ae3 \
  --fixtures docs/validation/qwen38-parallel/vision-fixtures \
  --concurrency 1,2 --requests-per-wave 2 \
  --output docs/validation/qwen38-parallel/uncensored-vision.json
```

Use `--concurrency 1 --output <serial.json>` followed by
`--concurrency 2 --baseline <serial.json> --output <parallel.json>` to compare
separate server runs. The model, projector, and fixture identities must match.
Both requests per wave are retained at C=1 so that each distinct image has a
sequential control. `--repeats` adds separate deterministic prompt markers.

The Qwen4Exp loader preserves the supplied GGUF path's existing tanh patch-merger
activation. The [published Transformers v5.16.1 merger](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1705)
uses GELU(erf); `TS_Q4E_VISION_MERGER_ERF=1` before server startup selects that
activation without changing the transformer blocks' tanh GELU. Record this setting
with each report. The delivered base-model default passed 12/12 image cases and
reproduced all 16 answers from the earlier tanh run. The full erf trial passed
10/12. In the subsequent same-build C=1 OCR control, tanh passed 2/2 and erf
passed 1/2, reading blue `9364` as `9334`. These scoped observations support
retaining tanh as the compatibility default; they do not explain the recognition
difference or establish broader activation quality. Future activation comparisons
must match model, projector, fixtures, sampling, cache settings and build identity.

## Projector sources

The supplied base quantization is published by
[Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66).
Its [F16 projector](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/blob/38bb39ee97821de2c9009abb7e93950eec396e66/mmproj-F16.gguf)
has SHA256 `1f7b7f0b984cf065c604360c29c8098362ed61b290db0ff12c6f360bb1a8a980`
and 904,004,000 bytes.

The matching uncensored
[F16 projector](https://huggingface.co/mradermacher/Qwen3.8-Flash-Next-Uncensored-GGUF/blob/61f739cd47b26ba67764deb28c99c92501892e26/Qwen3.8-Flash-Next-Uncensored.mmproj-f16.gguf)
is published by mradermacher at revision
`61f739cd47b26ba67764deb28c99c92501892e26`; it has SHA256
`4fee3582716fe691461576402f178882963e428ca9cbcc84dfeac9c53ebc1ae3`
and 904,004,352 bytes. This public conversion is used because the original
OrcaRouter projector download requires gated repository access.

Downloaded file size and SHA256 must match the pinned repository's LFS metadata.
These checks establish file identity; the end-to-end image tests establish the
projector's actual compatibility with the served quantization. The
[official Qwen model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/README.md)
describes the model's image input and 2560-wide language hidden states.

## Scope and limits

Both downloaded projectors have `qwen3vl_merger` metadata, 16-pixel patches,
spatial merge factor 2, and projection dimension 2560. The current image path
fits a still image to 65,536–2,097,152 pixels and rounds dimensions to multiples
of 32. A 640×480 card contributes 1,200 patches and 300 image tokens; the upper
pixel limit permits at most 8,192 patches and 2,048 image tokens per image.

The exact checks cover synthetic OCR, color recognition, image order, and
retained image history. They are not a broad visual-quality evaluation. Reported
throughput is all completion tokens divided by each wave's complete wall time,
including media preparation, queueing, prompt evaluation, and follow-up turns.
The sequential arm can warm the media cache for a subsequent parallel arm;
use fresh processes and report startup/cache settings when measuring cold image
latency. The supplied language weights exceed a 48 GiB machine's RAM, and F16
projector weights are promoted to F32 in memory, so memory pressure and disk
paging are material limits on local throughput.

The comparison uses assembled assistant content and rendered request hashes.
The shared SSE client does not require a `[DONE]` frame or verify that completion
IDs stay constant across chunks; these reports do not retain raw wire frames.
Passing this runner does not qualify those aspects of SSE protocol integrity.

`python3 eng/tests/test_qwen38_parallel_vision.py` exercises the comparison
validator's rejection of missing or corrupt evidence. Those tests do not count
as model or device validation.
