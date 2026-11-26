# DiffusionGemma

[← back to model index](README.md) | [中文](diffusiongemma_zh-cn.md)

## Status snapshot

| Field | Status |
|---|---|
| GGUF architecture keys | `diffusion-gemma`, `diffusion_gemma` |
| Source class | [`DiffusionGemmaModel`](../../TensorSharp.Models/Models/DiffusionGemma/DiffusionGemmaModel.cs) |
| Sampler | [`DiffusionGemmaSampler`](../../TensorSharp.Models/Models/DiffusionGemma/DiffusionGemmaSampler.cs) |
| Modalities | Native text + **image**; [Jev](jev.md) adds document extraction, sampled video frames and speech via a configured ASR companion |
| Thinking / tools | Thinking is not prompted: the prompt always renders with thinking off. A thought block the model writes anyway is parsed out and returned as reasoning only on `"think": true`; a canvas that holds nothing but a thought block is returned as the answer (§6). Tools/tool_choice refused with HTTP 400 |
| Generation mode | Block text diffusion, not autoregressive token decode |
| CLI support | `TensorSharp.Cli` detects `DiffusionGemmaModel` and uses diffusion run mode |
| Server support | Web UI chat stream with live denoising previews; Ollama/OpenAI compatibility endpoints use append-oriented response shapes and return the final text only (no denoising previews) |
| Continuous batching | Dedicated [`DiffusionBatchScheduler`](../../TensorSharp.Chat/DiffusionBatchScheduler.cs), admitted at block boundaries |

## Downloads

Verified GGUF pointers:

| Model | HF repo | Recommended file | Notes |
|---|---|---|---|
| diffusiongemma-26B-A4B-it | [unsloth/diffusiongemma-26B-A4B-it-GGUF](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | `diffusiongemma-26B-A4B-it-Q4_K_M.gguf` (16.807 GB); also `Q5_K_M`, `Q6_K`, `Q8_0`, `BF16` | GGUF `general.architecture` = `diffusion-gemma`. Official upstream weights: [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) |

`Q4_K_M` is the smallest quant in the unsloth repo. For tighter VRAM,
`config/diffusiongemma-26b-a4b-q3.json` pins DevQuasar's `Q3_K_M` (~13.3 GB;
unsloth publishes no Q3_K_M). Its metadata matches the unsloth file apart from
`diffusion.eb_*` sampler hints, which TensorSharp does not read.

**For image input you also need the vision tower, and it is NOT in any GGUF.**
Every published GGUF of this checkpoint is text-only — the conversion drops the
vision tower, and no mmproj was ever released. The tower does exist upstream:
all 356 of its tensors live in a single 2.8 GB shard of the 11-shard BF16
checkpoint, and TensorSharp loads that shard directly (no conversion step):

| File | HF repo | Size |
|---|---|---|
| `model-00011-of-00011.safetensors` | [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) | 2.84 GB |

```bash
hf download google/diffusiongemma-26B-A4B-it model-00011-of-00011.safetensors --local-dir models
```

Or let the config fetch it on first use — `config/diffusiongemma-26b-a4b-q4.json`,
`config/diffusiongemma-26b-a4b-q3.json` and `config/jev-diffusiongemma-q4.json`
declare it under `mmproj` with a SHA-256, so it downloads once and is reused
afterwards. To run text-only without the shard, add `--mmproj none` on the command
line: a command-line `--mmproj` drops the configuration's entry before it is
resolved, so the shard is never fetched, and `none` loads no projector — on the
server and the CLI alike (text-only requests work; image requests are refused).

To get an ordinary mmproj GGUF instead,
[`eng/diffusiongemma-mmproj.py`](../../eng/diffusiongemma-mmproj.py) converts the
shard with numpy alone (no torch or gguf package) into a single-file `gemma4v`
projector whose 2-D matmul weights are F16 (`--dtype f32` keeps them F32):

```bash
python3 eng/diffusiongemma-mmproj.py --src models/model-00011-of-00011.safetensors --out models/mmproj-diffusiongemma-26B-A4B-it-F16.gguf
```

Pass the result to `--mmproj` like any projector.
`Gemma4VisionOracleTests.MmprojTower_AgreesWithSafetensorsTower` checks that it
encodes like the shard; it needs a local fixture directory
(`TS_DIFFUSIONGEMMA_VISION_DIR`) and skips without one. The script also targets
llama.cpp's clip loader, but no llama.cpp run of its output is recorded.

Native audio is **not** supported and no projector can add it: the upstream config has
no `audio_config` and the weights contain no audio tower, so the `<|audio|>`
tokens the tokenizer inherits from Gemma 4 have nothing behind them.

Command-line download (one line per file; requires `pip install -U huggingface_hub`):

```bash
python -m pip install -U huggingface_hub
hf download unsloth/diffusiongemma-26B-A4B-it-GGUF diffusiongemma-26B-A4B-it-Q4_K_M.gguf --local-dir models
```

CLI diffusion mode (auto-dispatched from the model's architecture — no mode
flag; the prompt comes from a file via `--input`):

```bash
dotnet run --project TensorSharp.Cli -c Release -- --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt \
  --backend ggml_cuda --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0 --diffusion-blocks 1
```

To ask about a picture, add `--mmproj models/model-00011-of-00011.safetensors`
and `--image photo.png` (repeat `--image` for several pictures, in order). The
CLI does not look for this shard beside the model, and without a loaded tower it
refuses an image rather than answering without it.

Server (the Web UI at `http://localhost:5000/index.html` streams live denoising previews —
each step repaints the whole message via `replace` SSE frames; the Ollama/OpenAI
compatibility endpoints return the final text only):

```bash
dotnet run --project TensorSharp.Server.Host -c Release -- --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggml_cuda
```

Add `--mmproj models/model-00011-of-00011.safetensors` to accept images in chat
requests; the server never auto-detects a projector.

## 1. Origin and intent

For typed decisions (`noul`, `choice`, `score`) use the native
[`/v1/systemone` Jev endpoint](jev.md). It reads label probabilities from a seeded
canvas in one denoising step, with a sparse output projection and no generated
JSON parsing. Its state supports uploaded text/documents, images, sampled video
frames through the vision tower, and audio transcripts from a configured ASR
companion. These preprocessing paths are described in the Jev guide. Start with
[`jev-diffusiongemma-q4.json`](../../config/jev-diffusiongemma-q4.json).

DiffusionGemma is a block text-diffusion language model built on a Gemma-4-style
Mixture-of-Experts backbone. It is not the same runtime contract as the
autoregressive `gemma4` model:

- `Forward(int[] tokens)` intentionally throws. Generation must go through
  `DiffusionGemmaSampler`.
- Each denoising step runs over a concatenated `[prompt | canvas]` sequence.
- The prompt side is causal and never attends to the canvas.
- The canvas side is bidirectional over the prompt and canvas.
- The emitted block is the current deterministic argmax canvas, refined over
  multiple denoising steps.

The GGUF file must report `general.architecture=diffusion-gemma` or
`diffusion_gemma`; `ModelBase.Create()` routes those keys to
`DiffusionGemmaModel`.

## 2. Forward graph

DiffusionGemma runs on the shared decoder runtime: `DiffusionGemmaModel` derives from
[`Gemma4DecoderModel`](../../TensorSharp.Models/Models/Gemma4/Gemma4DecoderModel.cs), the
Gemma 4 26B-A4B recipe (per-layer head widths, V = K global layers, factored RoPE, sandwich
norms, the dense GeGLU branch beside the 128-expert routed one), so its caches, placements,
expert offload and graph executors are the runtime's on every backend.

A block runs as a prompt prefill and then one canvas step per denoising step:

1. `PrefillPrompt(promptTokens)` writes the prompt into the runtime's cache as a causal
   prompt (sliding-window layers clip it; an image's soft tokens attend each other on those
   layers only, each image apart). Each layer's output takes its `enc_layer_output_scale`.
   The next block's prompt extends this one, so only its new rows are prefilled.
2. `DecodeCanvas(canvasTokens, scBuffer, scUse, prevTempInv)` (or `DecodeCanvasSampled`)
   runs the canvas as one graph of block rows after the prompt: every canvas row reads every
   other canvas row on every layer, and the prompt rows from the first canvas row's window on
   (all of them on a global layer); each layer's output takes its `layer_output_scale`.
   Nothing is committed: the canvas's keys and values go to cells the next prompt rows
   overwrite, so every step of a block reads the same prompt. The steps of a block replay one
   recorded graph.

```text
canvas tokens
  -> rms_norm(embed * sqrt(n_embd) + selfcond(previous step's top-32))   (no weight)
  -> 30 Gemma 4 layers as block rows over the prompt's cache
  -> output norm -> tied lm-head -> final logit softcap
  -> canvas logits, or (on-device sampling) each position's argmax / entropy / draw / top-32
```

`ForwardCanvas(tokens, promptLen)` takes the prompt and the canvas in one call and runs the
same two steps (the prompt is causal and never reads the canvas).

## 3. Sampler contract

`DiffusionEbParams` controls generation:

| Parameter | Default | Meaning |
|---|---:|---|
| `MaxDenoisingSteps` | 48 | Maximum refinement steps per canvas block |
| `TMin` / `TMax` | 0.4 / 0.8 | Temperature schedule from late to early denoising |
| `EntropyBound` | 0.1 | Cumulative mutual-information bound for accepted positions |
| `StabilityThreshold` | 1 | How many stable argmax steps are required before early stop |
| `ConfidenceThreshold` | 0.005 | Mean entropy threshold for early stop |
| `Seed` | 0 | Deterministic sampler seed |
| `MaxBlocks` | 1 | Number of block-autoregressive canvas blocks |

The CLI maps this through:

```bash
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --input prompt.txt --backend ggml_metal \
  --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0 --diffusion-blocks 1
```

When `--diffusion-blocks` is `0`, the CLI derives the number of blocks from
`--max-tokens` and `diffusion.canvas_length`.

## 4. Architecture details

DiffusionGemma reuses many Gemma-4 backbone choices:

- NeoX RoPE with separate local/global dimensions.
- Five local sliding-window layers followed by one global layer pattern.
- Per-head Q/K RMSNorm and unweighted V RMSNorm.
- Global layers can omit `attn_v.weight`, using raw K as V.
- Dense gated-GELU MLP plus 128-expert top-8 MoE.
- Tied embeddings / lm-head and final logit softcapping.

Diffusion-specific metadata includes:

| Key | Meaning |
|---|---|
| `diffusion.canvas_length` | Number of canvas positions denoised per block, default 256 |
| `tokenizer.ggml.mask_token_id` | Mask token id used by warmup and fallback paths |
| `<arch>.attention.sliding_window_pattern` | Local/global layer pattern |
| `<arch>.attention.head_count_kv` | Per-layer KV head counts |
| `<arch>.expert_count` / `<arch>.expert_used_count` | MoE expert count and active top-k |

## 5. Acceleration status

- The prompt cache, the canvas steps and the batched canvases run on every backend
  (`ggml_cuda`, `ggml_metal`, `ggml_vulkan`, `ggml_cpu`, `cuda`, `mlx`, `cpu`), each step one
  graph on the backend's executor, recorded once per block and replayed.
- A block's prompt reuses the rows the previous block cached; only its new rows prefill.
- On-device sampling on `ggml_cuda`: the head's `[vocab, canvas]` logits stay on the device
  and the graph returns each position's argmax, entropy, multinomial draw and top-32 (the next
  step's self-conditioning). Elsewhere the logits come back and the host samples.
- Self-conditioning (on when the checkpoint has the `self_cond_*` weights): the previous step's
  top-32 soft embedding is summed on the host from the token table and runs through the
  gated-GELU MLP in the step's graph.
- Routed experts that do not fit the device run from host memory (the runtime's placement, or
  `--n-cpu-moe N` / `--cpu-moe`), the fused gate|up GeGLU stack on every executor.
- Block-boundary continuous batching in `TensorSharp.Server` through `DiffusionBatchScheduler`;
  several requests' canvases can run in one step (`DecodeCanvasBatched`).

Measured on one A40 (`diffusiongemma-26B-A4B-it-Q4_K_M.gguf`, ggml d9fe7bff,
`eng/validation/DiffusionGemmaAB`), against the legacy model this runtime replaced:

| `ggml_cuda` | legacy | runtime |
|---|---:|---:|
| 512-token prompt prefill (warm) | 2.47–2.65 s | 0.14–0.15 s |
| canvas step, host sampling (256 rows, self-conditioning) | 325–333 ms | 145–160 ms |
| canvas step, on-device sampling | 218 ms | 114–128 ms |
| generation, three prompts (ms per denoising step) | 268–353 | 129–153 |

Teacher-forced on the legacy model's own answers (the same 256-token canvases), the two
agree on every position's argmax and the mean log-probability of the canvas tokens matches
(-0.0000 / -0.0000 and -0.0003 / -0.0003 with self-conditioning); the generated answers to
the same prompts and seed say the same things.

Important toggles:

| Variable | Effect |
|---|---|
| `DIFFUSION_STEPS` | Server-side denoising steps per block, default 48 |
| `DIFFUSION_MAX_BATCH` | Server diffusion scheduler max active requests, default 2 |
| `TS_VRAM_HEADROOM_MB` | Device memory the runtime's expert placement keeps free (default 1/16 of the device); the routed experts that do not fit run from host memory |

The legacy model's switches (`DIFFUSION_NO_PKV`, `DIFFUSION_NO_SC`, `DIFFUSION_SC_TOPK`,
`DIFFUSION_BATCHED_FORWARD`, `DIFFUSION_NO_FUSED_DECODE`, `DIFFUSION_NO_FUSED_LMHEAD_TAIL`,
`DIFFUSION_LMHEAD_BATCH_CAP_MB`, `DIFFUSION_VRAM_HEADROOM_MB`,
`DIFFUSION_DEVICE_COPY_BUDGET_MB`, `DIFFUSION_SEGMENTED_DECODE`, `DIFFUSION_PIN_STREAMED`,
`DIFFUSION_FUSED_PREFILL_ATTN`, `DIFFUSION_NO_DEVICE_SAMPLE`, `DIFFUSION_DEVICE_SAMPLE_FORCE`,
`DIFFUSION_IMAGE_BIDIRECTIONAL`, `DIFFUSION_ASYNC_COMPUTE`, `DIFFUSION_CPU_ATTN_FAST`,
`DIFFUSION_CPU_MOE_CHUNK`, `DIFFUSION_CPU_LEGACY*`) were removed with it; setting one
prints what replaced it.

## 6. Server behavior

When the Web UI hosts a DiffusionGemma GGUF:

- `/api/chat` takes the diffusion path.
- The stream emits `replace` events rather than token append events, because
  every denoising step refines the whole current canvas.
- A final replacement is emitted before the `done` event.
- Concurrent requests share one background diffusion scheduler and are admitted
  between blocks.
- A Jev structured read, or the image encode of a new turn, does not wait for a
  chat's whole block: the scheduler hands the model over before its next forward,
  so it waits only for the forward in progress. The block's output is unchanged.
- Every request keeps its prompt in its own cache of the runtime (prefilled once per
  block, extended by the next block) and each step runs its canvas over it.
- On `cpu` and `ggml_cpu` the server skips its startup shared-prompt warm-up for
  this model. DiffusionGemma keeps nothing across requests, and the warm-up was a
  full 48-step denoise of a 256-token canvas, once per thinking mode, before the
  port opened (the first one took 252.8 s on `cpu` before the `cpu` work above).
  The GPU backends keep it: it records their prompt and canvas graphs before the
  first request.
- Image turns need the vision tower loaded with `--mmproj`. Each image is
  expanded into its soft-token span before the context check. The encoded spans
  belong to the sequence in the scheduler and are written over their placeholder
  rows whenever its cache prefills them, so image and text requests batch together. Ordinary chat refuses
  audio. It has no native video path: an OpenAI `video_url` part is refused, and a video uploaded
  in the Web UI reaches the model only as its extracted frames, each rendered as
  a plain `<|image>` (no `<|video>` marker and no frame timestamps). The separate
  [Jev endpoint](jev.md#files-documents-video-and-audio) accepts uploaded video
  through bounded frame sampling and speech through an ASR companion.

The Ollama and OpenAI compatibility adapters still use append-oriented response
shapes through `ChatStreamWithMetricsAsync`. They can surface the final
DiffusionGemma text, but the live denoising previews and `replace` frames are
Web UI-only.

The model writes Gemma 4's channel syntax: a canvas may open with the
`<|channel>thought\n` primer, or close a thought block the prompt opened with a
bare `<channel|>`, before the answer. The architecture is registered as the
`diffusion-gemma` chat protocol (`Gemma4OutputParser`, always required, the
GGUF template still renders the prompt), and every preview and the final text
go through that parser: the thought block is dropped unless the request asks for
reasoning (`"think": true` returns it as `reasoning_content`), and the channel
markers never reach a client. Before this the raw canvas was delivered verbatim
and OpenAI answers began with the literal `<|channel>thought` marker.

This checkpoint often answers inside the thought block and never writes the
closing `<channel|>`. A finished canvas is not a truncated thought, so when it
parses to no answer but a thought block, that text is returned as the answer
(and not repeated as reasoning). Without this, 3 of 6 one-line questions on the
Q4_K_M file came back empty. Likewise, when the only thing on the canvas is a
tool call the model wrote anyway, that call is shown as text rather than
returning an empty answer; next to other answer text it is dropped.

Tool calling is refused up front: `/v1/chat/completions` answers HTTP 400
(`{"error": ...}`, `invalid_request_error`) to any request that carries `tools`
or a `tool_choice` other than `"none"` while a DiffusionGemma model is loaded,
because a block-diffusion turn has no tool loop to feed a result back into.
`/v1/responses` and the Ollama endpoint `/api/chat/ollama` refuse `tools` the
same way. The built-in skills / code-execution tools and the sub-agent
coordination tools are never offered to this family either (the protocol entry
declares `RendersToolDeclarations = false`), so `--code-exec`, skills discovery
and sub-agent delegation leave a diffusion request exactly as it was before.

## 7. Test coverage

[`DiffusionGemmaRuntimeTests`](../../InferenceWeb.Tests/DiffusionGemmaRuntimeTests.cs) needs
no weights: a synthetic diffusion-gemma file on the 26B-A4B backbone (fused gate|up experts,
V = K global layers, a 32-token window, a 16-row canvas, the self-conditioning MLP and the
encoder scales) pins the canvas step's attention (every canvas row reads every other; a prompt
token below the first canvas row's window does not reach the canvas, the first one inside does),
steps that commit nothing, a block's prompt that extends the cached one against a fresh prefill,
the unified call, the pinned GGML backend against the managed one, the graph's sampler against
the host's arithmetic, generation on either sampler, batched canvases against each alone, the
scheduler's batched block, routed experts on the host and structured reads against their
full-vocabulary reference. It is the family's dedicated suite in
`SharedRuntimeConformanceTests`.

[`DiffusionGemmaTests`](../../InferenceWeb.Tests/DiffusionGemmaTests.cs) and
[`JevStructuredReadTests`](../../InferenceWeb.Tests/JevStructuredReadTests.cs) are opt-in on
the real GGUF via `TS_TEST_MODEL_DIR` (`TS_TEST_BACKEND` picks the backend): finite logits,
end-to-end generation, prompt-cache equivalence, repeated-token and memory-retention guards,
batched decode, and the selected-label read against the full vocabulary.
[`DiffusionGemmaInterleaveTests`](../../InferenceWeb.Tests/DiffusionGemmaInterleaveTests.cs)
checks that Jev reads between a block's forwards change neither the block nor the reads.

[`DiffusionGemmaProtocolTests`](../../InferenceWeb.Tests/DiffusionGemmaProtocolTests.cs)
needs no weights and pins the channel parsing: a thought block is dropped unless
requested, an unterminated one is the answer, and a spontaneous tool call
surfaces as text.

Image input has two more suites.
[`DiffusionGemmaVisionConcurrencyTests`](../../InferenceWeb.Tests/DiffusionGemmaVisionConcurrencyTests.cs)
needs no weights and pins that image spans belong to a sequence, so two image
requests in flight cannot overwrite each other.
[`Gemma4VisionOracleTests`](../../InferenceWeb.Tests/Gemma4VisionOracleTests.cs)
(opt-in via `TS_DIFFUSIONGEMMA_VISION_DIR`) compares the tower loaded from the
shard against a NumPy transcription of the Hugging Face reference
(`eng/diffusiongemma-vision-oracle.py`).

[`eng/validation/DiffusionGemmaAB`](../../eng/validation/DiffusionGemmaAB/Program.cs) compares
two builds on the real checkpoint (teacher-forced logits, generation, timing, batched canvases,
structured reads); its reports belong under the ignored `artifacts/`.

## 8. Remaining work

- Add dedicated API examples once Ollama/OpenAI adapters grow a diffusion-aware
  compatibility surface.
- On-device sampling on `ggml_metal` and `ggml_vulkan` (their graphs carry the same sampler
  ops; not yet validated there), and in one step for several requests' canvases.
- Fold more diffusion scheduler metrics into `/api/queue/status` if operators
  need per-diffusion-batch visibility.
