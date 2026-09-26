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

The model exposes two execution regimes.

The unified correctness path is `ForwardCanvas(tokens, promptLen)`:

```text
[prompt tokens | canvas tokens]
  -> region-aware embedding scale
  -> prompt/canvas attention masks
  -> N Gemma-style transformer layers
       - local/global QK-norm attention
       - dense gated-GELU MLP
       - top-k MoE experts
       - prompt encoder scale / canvas decoder scale
  -> output norm
  -> tied lm-head
  -> final logit softcap
  -> canvas logits
```

The optimized GPU path splits each block into a prompt prefill plus repeated
canvas decodes:

1. `PrefillPrompt(promptTokens)` computes the prompt K/V once.
2. `DecodeCanvas(canvasTokens, scBuffer, scUse, prevTempInv)` reuses prompt K/V
   for every denoising step.
3. The sampler accepts low-entropy positions, re-noises the rest, and repeats.

Prompt-KV caching is enabled on the device-glue backends (`ggml_metal`,
`ggml_cuda`, `mlx`, `cuda`); on `cpu`, `ggml_cpu` and `ggml_vulkan` every step
runs the unified `[prefix|canvas]` forward instead.

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

Current optimized paths include:

- Prompt-KV cache on `ggml_metal`, `ggml_cuda`, `mlx` and `cuda` (not
  `ggml_vulkan`).
- Self-conditioning enabled by default; disable with `DIFFUSION_NO_SC=1`.
- GGML fused decode layer, fused whole-model decode, and fused lm-head tail.
- CUDA VRAM residency planning: when the model is larger than VRAM, weights are
  preloaded device-side in priority order (lm_head/embedding, per-layer
  attention/dense, then MoE expert stacks) up to free-VRAM-minus-headroom, the
  device-copy cache is capped, and decode switches to the SEGMENTED per-layer
  fused path so the non-resident remainder streams through one bounded staging
  buffer instead of oversubscribing VRAM (which makes Windows WDDM page the
  working set every submission — measured ~4x slower than streaming).
- Step-invariant decode masks are cached host-side and bound cacheable (one
  device upload per block geometry instead of a rebuild+upload per layer/step).
- SIMD-vectorized host paths (`TensorPrimitives`): per-position
  argmax/entropy/multinomial sampling and the final-logit softcap; the fused
  lm-head logits land in one pooled pinned buffer instead of a fresh 268 MB
  allocation per step.
- MLX K-quant affine repacking for DiffusionGemma's multi-row canvas workload.
- Block-boundary continuous batching in `TensorSharp.Server` through
  `DiffusionBatchScheduler`.

Important toggles:

| Variable | Effect |
|---|---|
| `DIFFUSION_STEPS` | Server-side denoising steps per block, default 48 |
| `DIFFUSION_MAX_BATCH` | Server diffusion scheduler max active requests, default 2 |
| `DIFFUSION_NO_PKV=1` | Disable prompt-KV caching on device-glue backends |
| `DIFFUSION_NO_SC=1` | Disable self-conditioning |
| `DIFFUSION_SC_TOPK` | Experimental self-conditioning top-K cutoff, default 32 |
| `DIFFUSION_BATCHED_FORWARD=1` | Use true batched canvas decode instead of time-sliced fused single-canvas decode |
| `DIFFUSION_NO_FUSED_DECODE=1` | Disable GGML fused whole-model diffusion decode |
| `DIFFUSION_NO_FUSED_LMHEAD_TAIL=1` | Disable fused output-norm + lm-head + softcap tail |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | Cap transient batched lm-head logits memory, default 300 MB |
| `DIFFUSION_VRAM_HEADROOM_MB` | ggml_cuda: VRAM kept free of preloaded weights, default 2048 |
| `DIFFUSION_DEVICE_COPY_BUDGET_MB` | ggml_cuda: device-copy cache cap when the model spills VRAM, default 768 |
| `DIFFUSION_SEGMENTED_DECODE` | ggml_cuda: force per-layer fused decode `1`/`0` (auto when the model spills VRAM) |
| `DIFFUSION_PIN_STREAMED=1` | ggml_cuda: page-locked copies of streamed weights for DMA uploads (costs RAM) |
| `DIFFUSION_FUSED_PREFILL_ATTN` | Fused GGML prompt attention: on by default for `ggml_cuda` (`0` restores the per-op reference), `1` opts in on other GGML backends without implying they were validated; always off for image prompts, which need the bidirectional image-span mask |
| `DIFFUSION_NO_DEVICE_SAMPLE=1` | ggml_cuda: disable on-device argmax / entropy / sampling / self-conditioning top-K (default on while the model fits in VRAM) |
| `DIFFUSION_DEVICE_SAMPLE_FORCE=1` | ggml_cuda: keep device sampling even when the model spills VRAM (segmented decode), where the host sampler is normally used; for experiments |
| `DIFFUSION_IMAGE_BIDIRECTIONAL=0` | Make image soft-token spans causal; by default attention inside a span is bidirectional on sliding layers |
| `DIFFUSION_ASYNC_COMPUTE=1` | ggml_metal: keep Metal's lazy-sync async compute on (GGML enables it only on Metal). The model turns it off at load because its per-op paths write tensors from the CPU (MoE expert inputs, embeddings, masks, self-conditioning, the re-noised canvas) and Metal has no host-write barrier, so the prompt K/V cached for the whole turn could be corrupted and answers came back fluent but off-topic. Unsafe; only for measuring the cost of the synchronization |

## 6. Server behavior

When the Web UI hosts a DiffusionGemma GGUF:

- `/api/chat` takes the diffusion path.
- The stream emits `replace` events rather than token append events, because
  every denoising step refines the whole current canvas.
- A final replacement is emitted before the `done` event.
- Concurrent requests share one background diffusion scheduler and are admitted
  between blocks.
- On backends without prompt-KV caching (`cpu`, `ggml_cpu`, `ggml_vulkan`) the scheduler runs
  each sequence's step through the unified `[prefix|canvas]` forward instead of
  prefill + canvas decode; behavior and output are identical.
- Image turns need the vision tower loaded with `--mmproj`. Each image is
  expanded into its soft-token span before the context check. The encoded spans
  belong to the sequence in the scheduler and are re-applied whenever its prompt
  is prefilled (once per block, or at every step on backends without prompt-KV
  caching), so image and text requests batch together. Ordinary chat refuses
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

[`DiffusionGemmaTests`](../../InferenceWeb.Tests/DiffusionGemmaTests.cs) is
opt-in on real GGUFs via `TS_TEST_MODEL_DIR`. It covers:

- `ForwardCanvas` finite-logit correctness.
- End-to-end EntropyBound generation.
- Prompt-KV equivalence and speed probes.
- Regression guards for repeated-token output and device-memory retention.
- Batched decode equivalence and two-request generation through the scheduler
  style used by the server.

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

## 8. Remaining work

- Add dedicated API examples once Ollama/OpenAI adapters grow a diffusion-aware
  compatibility surface.
- Promote true batched canvas decode only if it wins on target GPUs; today the
  fused single-canvas path can be faster when one canvas already saturates the
  GPU.
- Fold more diffusion scheduler metrics into `/api/queue/status` if operators
  need per-diffusion-batch visibility.
