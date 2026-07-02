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
- **Lightning distillation LoRA (4/8-step editing)** — `--qwen-image-lora
  <lora.safetensors>` (CLI + server) / `TS_QWEN_IMAGE_LORA` applies a DiT LoRA as
  a **runtime side-path** next to each targeted projection:
  `y = W_quant·x + b + (alpha/rank)·up·(down·x)` in F32, with the quantized base
  weights untouched. (Merging into the weights — the stable-diffusion.cpp
  dequantize→add→requantize path — is only sound for F16/Q8_0 storage: the
  Lightning deltas are ~1e-4 RMS, far below a Q2_K quantization step, so a merge
  replaces them with requantization noise.) The factors are resident like the
  weights, capture-safe, and cost a few % extra per forward + ~1.6 GB VRAM. With
  a lightx2v
  [Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)
  checkpoint the sampling defaults switch automatically (parsed from the
  filename): its trained step count (4 or 8), cfg 1.0 (single forward per step),
  and the fixed timestep shift 3 the distillation was trained with — cutting the
  default 60 DiT forwards to 4–8. Explicit `steps`/`cfg` still win;
  `TS_QWEN_IMAGE_LORA_SCALE` adjusts the LoRA multiplier (default 1.0). The
  side-path is implemented in the whole-model forward (the default CUDA path); a
  fallback to the per-block/managed paths logs a warning that the LoRA is not
  applied there.
- **Whole-step denoise cache (EasyCache)** — the default cache mode, ported from
  stable-diffusion.cpp's `easycache.hpp`. It predicts each step's output change
  from the measured input-latent change (times an empirically tracked
  input→output transformation rate) and, while the accumulated prediction stays
  below a threshold, skips the **entire DiT forward for both CFG branches**,
  reconstructing each branch's velocity as `input + cached(output − input)`.
  The first ~15% and last ~5% of steps always compute. Typically skips 40–55%
  of steps at the default threshold with a perceptually equivalent result.
  Knobs: `TS_QWEN_DIT_CACHE_MODE` (`easycache`/`fbc`/`both`/`off`),
  `TS_QWEN_DIT_EASYCACHE_THRESHOLD` (default 0.2 — lower = closer to no-cache),
  `TS_QWEN_DIT_EASYCACHE_START`/`_END` (window fractions, 0.15/0.95),
  `TS_QWEN_DIT_CACHE_DEBUG=1` (per-step decision trace).
- **First-Block-Cache** across denoise steps (reset per generation); the previous
  default, now selected with `TS_QWEN_DIT_CACHE_MODE=fbc`. It always computes
  block 0 per branch and skips blocks 1..59 on low-change steps — strictly less
  saving than the whole-step cache, but its decision uses the actual block-0
  residual rather than a prediction.
- **Fused conditioning-encoder trunks** (`TSGgml_QwenTeTrunk`, default on): the
  Qwen2.5-VL text-encoder LLM (28 layers, GQA, causal) and its vision tower
  (32 layers, MHA, window/full attention as an additive mask) each run their whole
  layer stack as ONE device graph — the per-op path paid ~10 device⇄host round
  trips per layer (M-RoPE, bias adds, SiLU, block-mask softmax as host loops).
  RoPE cos/sin tables are host-precomputed; weights bind resident by GGUF mmap
  pointer. At the default edit: vision 4.7 s → **1.6 s**, LLM prefill 2.8 s →
  **0.8 s** (text-conditioning phase 11.2 s → 2.5 s). Verified vs the numpy
  oracles (`te-verify` cosine 0.9989, `vis-verify` 0.9994 — both ≥ the per-op
  path's own scores). Falls back to the per-op path when the backend can't run
  it; `TS_QWEN_TE_FUSED=0` disables both.
- **VRAM budgeting**: the text + vision encoders are freed after conditioning so
  the DiT (and its attention scratch) own the working set; the target output
  area is auto-clamped to fit device VRAM unless `Width`/`Height` are pinned, and
  the persistent reuse `gallocr` is handed back before the final VAE decode.
- **Fused whole-VAE graph** (`TSGgml_QwenVaeRun`, default on): the entire VAE
  encode/decode runs as ONE device-resident ggml graph — the C# side emits a flat
  op list mirroring the verified `VaeReferenceMath` topology, features stay on the
  GPU end-to-end, weights bind resident from stable buffers (uploaded once), and
  each conv picks im2col+GEMM (tensor cores) while its transient F16 im2col fits a
  budget (`TS_QWEN_VAE_FUSED_IM2COL_BUDGET`, default 2 GiB; the gallocr reuses the
  scratch so the peak is one conv's im2col) or `ggml_conv_2d_direct` above it.
  Replaces the per-conv path's GBs of PCIe round-trips + CPU SiLU/norm loops:
  at 928×688 encode 19.5 s → **0.95 s**, decode 22.8 s → **1.35 s**, bit-equivalent
  to the diffusers oracle (PSNR 99 dB, same as the legacy path). Falls back to the
  per-conv path when the backend can't run it; `TS_QWEN_VAE_FUSED=0` disables.
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
| `TS_QWEN_IMAGE_LORA` | DiT LoRA safetensors applied as a runtime F32 side-path (CLI/server: `--qwen-image-lora`); a Lightning checkpoint also switches the sampling defaults (its steps, cfg 1.0, fixed shift 3) |
| `TS_QWEN_IMAGE_LORA_SCALE` | LoRA multiplier (default 1.0; 0 = structurally on but zero effect) |
| `TS_QWEN_IMAGE_MAX_AREA` | Override the Metal default target-area clamp |
| `TS_QWEN_DIT_WHOLE_CAPTURE=0` | Disable the CUDA-graph-captured whole-DiT forward |
| `TS_QWEN_DIT_FLASH=0` | Force the explicit-scores attention path (tighter quadratic VRAM budget) |
| `TS_QWEN_DIT_CFG_BATCH_MAXTOK` | Token budget under which CFG-batching stays enabled |
| `TS_QWEN_DIT_CACHE_MODE` | Denoise cache: `easycache` (default, whole-step skip), `fbc` (First-Block-Cache), `both`, `off` |
| `TS_QWEN_DIT_EASYCACHE_THRESHOLD` | EasyCache accumulated-change threshold (default 0.2; lower = closer to no-cache, higher = more skips) |
| `TS_QWEN_DIT_CACHE=0` | Legacy master switch — disables all denoise caching |
| `TS_QIMG_DEBUG=1` | Per-step velocity / latent statistics |

## 6. Generation parameters (`QwenImageParams`)

| Property | Default | Meaning |
|---|---:|---|
| `Steps` | 0 (auto) | FlowMatch-Euler denoising steps. Auto = 30, or the step count of a loaded Lightning LoRA (4/8) |
| `CfgScale` | 0 (auto) | True-CFG guidance scale; `<= 1` disables the negative pass. Auto = 2.5 (the Qwen-Image-Edit-2511 recommendation — 4.0 over-guides: distorted faces, over-saturated color), or 1.0 with a Lightning LoRA. Raise toward 3.5–4 for stronger stylization at the cost of face fidelity |
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
