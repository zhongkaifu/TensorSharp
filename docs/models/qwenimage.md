# Qwen-Image-Edit

[← back to model index](README.md)

## Status snapshot

| Field | Status |
|---|---|
| GGUF architecture key | `qwen_image` (the MMDiT diffusion transformer) |
| Source class | [`QwenImageModel`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs) (+ internal [`QwenImagePipeline`](../../TensorSharp.Models/Models/QwenImage/QwenImagePipeline.cs)) |
| Task | Image editing — prompt + input image → edited image |
| Modalities | Image in + text in → image out |
| Thinking / tools | Not applicable |
| Generation mode | FlowMatch-Euler diffusion denoise (true-CFG), **not** autoregressive token decode |
| CLI support | `TensorSharp.Cli` detects `QwenImageModel` and runs image-edit mode (`--image`, `--prompt`, `--output`, `--cfg`, `--diffusion-steps`, `--diffusion-seed`) |
| Server support | Web UI image-edit flow: `POST /api/image-edit` and `POST /api/image-edit/stream` (SSE with live denoising previews) |
| Continuous batching | None — image edits are serialized (the diffusion nets are not thread-safe); concurrent requests run one at a time |

## 1. Origin and intent

Qwen-Image-Edit is an instruction-driven image editor. Unlike the autoregressive
LLMs, the loaded `qwen_image` GGUF is **only** the MMDiT (multimodal diffusion
transformer); image editing additionally needs two networks the DiT GGUF does
not contain:

- the **Qwen-Image VAE** — image ↔ 16-channel latent (8× spatial downsample), and
- the **Qwen2.5-VL-7B text encoder** — prompt → 3584-dim conditioning, with an
  optional `mmproj` vision tower for image-grounded conditioning.

`ModelBase.Create()` routes `general.architecture = qwen_image` to
`QwenImageModel`. The companion GGUFs are resolved from the DiT GGUF's directory,
or via the `TS_QWEN_IMAGE_VAE` / `TS_QWEN_IMAGE_TE` / `TS_QWEN_IMAGE_MMPROJ`
environment variables (the VAE may be the original `.safetensors`, loaded
directly with BF16→F32 upcast). `QwenImageModel` is not an
`IModelArchitecture` text generator: `Forward()` throws and editing is driven
through [`EditImage()`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs).

## 2. Pipeline (the denoise loop)

[`QwenImagePipeline.Edit`](../../TensorSharp.Models/Models/QwenImage/QwenImagePipeline.cs)
orchestrates the full edit:

```text
input image
  -> preprocess (aspect-preserving; dims multiple of 16; area clamped to VRAM budget, up to native ~1 MP)
  -> VAE encode -> normalize (diffusers latents mean/std) -> pack [refSeq, 64]
  -> text conditioning: Qwen2.5-VL encode(prompt [+ vision-grounded image tokens])
       (+ negative prompt when CfgScale > 1, for true-CFG)
  -> free text + vision encoders (reclaim VRAM for the DiT)
  -> noise latents (seeded Gaussian) packed to [genSeq, 64]
  -> concatenate [generated | reference] tokens (modulateIndex marks the ref half)
  -> FlowMatch-Euler scheduler (timestep == sigma), First-Block-Cache reset
  -> for each step:
       DiT velocity prediction (cond) [+ neg -> true-CFG combine + per-row renorm]
       scheduler.Step(latents, v, step)
       optional decoded RGB preview (downsampled) via OnStep callback
  -> unpack -> denormalize -> VAE decode -> output image
```

True-CFG combines the conditional and unconditional velocity per packed token
row (`comb = neg + scale·(cond − neg)`) and renormalizes each row back to the
conditional norm. With `CfgScale <= 1` the negative pass is skipped (one DiT
forward per step instead of two).

## 3. MMDiT architecture constants

The `qwen_image` GGUF carries no hyperparameters; they are pinned in
[`QwenImageModel`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs):

| Constant | Value |
|---|---|
| Hidden size | 3072 |
| Layers (double-stream blocks) | 60 |
| Attention heads | 24 (head dim 128) |
| DiT input channels | 64 (16 latent channels × 2×2 patch) |
| Text conditioning dim | 3584 (Qwen2.5-VL hidden) |
| VAE latent channels | 16 |
| VAE scale factor | 8 |
| RoPE axes dims | {16, 56, 56} (sum = head dim 128) |
| Eps | 1e-6 |

Each MMDiT block jointly attends image and text tokens with multimodal RoPE;
timestep/text modulation is applied via the per-token `modulateIndex`
(0 = generated half, 1 = reference half). The reference latent is concatenated
into the token sequence so the edit is grounded on the original image.

## 4. TensorSharp implementation

| Component | File |
|---|---|
| Model entry / companion resolution | [`QwenImageModel.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageModel.cs) |
| Denoise orchestration | [`QwenImagePipeline.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImagePipeline.cs) |
| Generation params | [`QwenImageParams.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageParams.cs) |
| MMDiT (graph, cache, CFG-batch, native, whole-model) | `QwenImageDiT*.cs` |
| FlowMatch-Euler scheduler | [`QwenImageScheduler.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageScheduler.cs) |
| VAE (+ reference math) | [`QwenImageVae.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageVae.cs) |
| Qwen2.5-VL text encoder | [`QwenImageTextEncoder.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageTextEncoder.cs) |
| Vision tower (grounding) | [`QwenImageVisionEncoder.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImageVisionEncoder.cs) |
| Image decode / encode (PNG) | [`ImageIO.cs`](../../TensorSharp.Models/Models/QwenImage/ImageIO.cs) |
| Fused native kernels | [`ggml_ops_qwen_image.cpp`](../../TensorSharp.GGML.Native/ggml_ops_qwen_image.cpp) |

## 5. Acceleration status

- **CUDA-graph-captured whole-DiT forward** (`TSGgml_QwenImageForward`): the
  weights are made resident once and the whole 60-block forward is captured into
  a single replayable graph with a dedicated `gallocr`. This turned the denoise
  from launch-bound (~40% GPU) into compute-bound (~100% GPU) — the per-forward
  cost dropped ~2.9× and the 8-step denoise fell from ~153 s to ~63 s on the
  measured CUDA box. Toggle off with `TS_QWEN_DIT_WHOLE_CAPTURE=0`.
- **Flash attention by default** on the ggml-cuda path (`TS_QWEN_DIT_FLASH=0`
  forces the explicit-scores path): O(n) attention memory, so higher resolutions
  fit before the VAE decode's im2col scratch becomes the limit.
- **CFG-batching**: both guidance branches run in one launch-amortized fused
  pass when the combined token block fits VRAM (`TS_QWEN_DIT_CFG_BATCH_MAXTOK`);
  larger images fall back to two forwards per step.
- **First-Block-Cache** across denoise steps (reset per generation).
- **Fused vision encoder** (`TS_QWEN35_VENC_FUSED`): the Qwen2.5-VL vision tower
  runs as a single graph (~2.1× over per-block), though it is only a small slice
  of the edit's wall time.
- **VRAM budgeting**: the text + vision encoders are freed after conditioning so
  the DiT (and its attention scratch) own the working set; the target output
  area is auto-clamped to fit device VRAM unless `Width`/`Height` are pinned, and
  the persistent reuse `gallocr` is handed back before the final VAE decode.
- **Band-tiled VAE conv** (`VaeReferenceMath.TryGpuConv2dMaybeTiled`): `ggml_conv_2d`
  materializes an F16 im2col of `~IC·KH·KW·OH·OW·2` bytes, which is several GB at
  high resolution and used to spill into shared VRAM (≈3× slower VAE) or OOM — it
  was the real ceiling that forced the output down to ~0.55 MP. The conv is now
  split into horizontal output bands whose im2col fits a budget
  (`TS_QWEN_VAE_CONV_TILE_BYTES`, default 1 GiB); each band re-runs the same conv
  on a manually-padded input slice so the result is **bit-identical** (no tile
  seams — only the transient im2col is bounded). This lets the area clamp target
  the model's **native ~1 MP**, which materially improves face/fine detail.

Important toggles:

| Variable | Effect |
|---|---|
| `TS_QWEN_IMAGE_VAE` / `TS_QWEN_IMAGE_TE` / `TS_QWEN_IMAGE_MMPROJ` | Override the resolved companion GGUFs (the CLI exposes these as `--qwen-image-vae` / `--qwen-image-vl` / `--qwen-image-mmproj`) |
| `TS_QWEN_IMAGE_NO_VISION=1` | Skip vision grounding (faster, ungrounded text-only conditioning) |
| `TS_QWEN_IMAGE_MAX_AREA` | Override the Metal default target-area clamp |
| `TS_QWEN_DIT_WHOLE_CAPTURE=0` | Disable the CUDA-graph-captured whole-DiT forward |
| `TS_QWEN_DIT_FLASH=0` | Force the explicit-scores attention path (tighter quadratic VRAM budget) |
| `TS_QWEN_DIT_CFG_BATCH_MAXTOK` | Token budget under which CFG-batching stays enabled |
| `TS_QIMG_DEBUG=1` | Per-step velocity / latent statistics |

## 6. Generation parameters (`QwenImageParams`)

| Property | Default | Meaning |
|---|---:|---|
| `Steps` | 30 | FlowMatch-Euler denoising steps |
| `CfgScale` | 2.5 | True-CFG guidance scale; `<= 1` disables the negative pass. 2.5 follows the Qwen-Image-Edit-2511 recommendation — 4.0 over-guides ("CFG burn": distorted faces, over-saturated color). Raise toward 3.5–4 for stronger stylization at the cost of face fidelity |
| `NegativePrompt` | `" "` | Negative prompt for the CFG pass (used only when `CfgScale > 1`) |
| `Seed` | 0 | Deterministic initial-noise seed |
| `TargetArea` | 1024×1024 | Output area in pixels (aspect follows the input; dims snapped to /16) |
| `Width` / `Height` | 0 | Explicit output size (bypasses the VRAM area clamp) |
| `OnStep` | null | Per-step callback `(step, totalSteps, preview)` for live UI feedback |
| `PreviewCount` | 0 | Number of decoded RGB previews emitted across the loop |

## 7. Server behavior

When the Web UI hosts a `qwen_image` DiT GGUF, the upload + edit endpoints take
over:

- `POST /api/image-edit` runs a single edit and returns the output image.
- `POST /api/image-edit/stream` streams SSE frames: progress ticks plus
  throttled decoded previews (`{ imageEdit: true, step, total, image, width,
  height }`), then a final full-resolution image before `done`.
- Edits are serialized behind a shared lock (the diffusion nets are not
  thread-safe), so concurrent requests run one at a time.
- The Ollama / OpenAI chat adapters are autoregressive and do not expose image
  editing.

## 8. Remaining work

- Concurrent edits are serialized; batched / pipelined editing across requests
  is not implemented.
- Vision-grounded conditioning is accurate but heavy; the fastest practical
  full-quality edits are on the CUDA path with the whole-DiT capture.
