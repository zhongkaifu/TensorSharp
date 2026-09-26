# Wan Video Generation (2.1 Text-to-Video · 2.2 Text/Image-to-Video)

TensorSharp runs the [Wan 2.1](https://github.com/Wan-Video/Wan2.1) and
[Wan 2.2](https://github.com/Wan-Video/Wan2.2) video diffusion models natively:
prompt (plus an optional first-frame image on the Wan 2.2 models) in, H.264 MP4
out, from both `TensorSharp.Cli` and `TensorSharp.Server.Host` (OpenAI-style API +
the bundled Web UI chat).

> **Wan is the video-only family.** The newer video model in TensorSharp is
> [MiniMax-H3](minimax-h3.md), which generates video **and native 32 kHz stereo
> audio together in one packed latent** — text-to-video, image-to-video,
> first/last frame and reference-to-video, CFG-free at 4–8 steps. Wan stays the
> family to reach for when you want video alone, and it is the one with the
> step-distilled checkpoints that cut the 100-DiT-pass recipe to 4.

Supported checkpoint families (auto-detected from the DiT GGUF):

| Family | Latent | VAE | Modes | Notes |
|---|---|---|---|---|
| Wan 2.1 T2V (1.3B / 14B) | 16 ch, 8×8×4 | wan_2.1_vae | T2V | single DiT |
| Wan 2.2 TI2V-5B | 48 ch, 16×16×4 | Wan2.2_VAE | T2V + **I2V** | dense 5B, 24 fps, 720p-capable |
| Wan 2.2 A14B (T2V / I2V) | 16 ch (36 ch I2V input) | wan_2.1_vae | T2V + **I2V** | two 14B experts switched at a timestep boundary |

For image-to-video the uploaded image becomes the **first frame** of the video
and the text prompt drives motion, camera work, atmosphere and scene changes.

The networks cooperate per generation (all resolved automatically when placed
in one folder — subfolders like `VAE/`, `HighNoise/`, `LowNoise/` included):

| Piece | File | Source |
|---|---|---|
| DiT (the `--model` GGUF, `general.architecture = wan`) | e.g. `Wan2.2-TI2V-5B-Q8_0.gguf` | [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF), [QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF), [city96/Wan2.1-T2V-14B-gguf](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf) |
| second A14B expert (A14B only) | the matching `…HighNoise…`/`…LowNoise…` GGUF | same repo (`TS_WAN_DIT2` overrides; auto-found by name in the same/sibling folder) |
| UMT5-XXL text encoder | `umt5-xxl-encoder-Q8_0.gguf` | [city96/umt5-xxl-encoder-gguf](https://huggingface.co/city96/umt5-xxl-encoder-gguf) (`--video-text-encoder` / `TS_WAN_TE`) |
| Wan 2.1 video VAE (2.1 + A14B) | `wan_2.1_vae.safetensors` | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors) (`--video-vae` / `TS_WAN_VAE`) |
| Wan 2.2 video VAE (TI2V-5B) | `Wan2.2_VAE.safetensors` | bundled in [QuantStack/Wan2.2-TI2V-5B-GGUF](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF/tree/main/VAE) |

The text encoder is released from VRAM before the denoise starts, the VAE
encoder (I2V) before the DiT loads, and the DiT before the VAE decode, so peak
VRAM is roughly `max(TE, DiT + attention, VAE)` — TI2V-5B Q8_0 generates
81-frame 480p image-to-video on a 16 GB GPU in under 8 minutes, and both A14B
Q4_K_M experts run sequentially on the same card.

### Quickest way to get all of them: a config file

Because several files have to line up, the ready-made configs in
[`config/`](../../config/README.md#video-generation-video-only-wan) name every network and
download whatever is missing on the first run:

```bash
# Text-to-video AND image-to-video, 4-step distilled (~12.9 GB on first run).
TensorSharp.Server.Host --config config/wan-video-ti2v-5b-turbo.json

TensorSharp.Cli --config config/wan-video-ti2v-5b-turbo.json \
  --prompt "a red fox trotting through falling snow" --output fox.mp4

# Image-to-video: the image becomes frame 0, the prompt drives the motion.
TensorSharp.Cli --config config/wan-video-ti2v-5b-turbo.json \
  --image first_frame.png --prompt "she turns and smiles" --output clip.mp4
```

| Config | DiT | VAE | Text encoder | Modes | First-run download |
|---|---|---|---|---|---|
| `wan-video-ti2v-5b-turbo.json` | TI2V-5B Turbo (4-step) | Wan 2.2 | UMT5-XXL | T2V + I2V | ~12.9 GB (DiT 5.4 GB, UMT5 6.0 GB, Wan 2.2 VAE 1.4 GB) |
| `wan-video-ti2v-5b.json` | TI2V-5B (50-step) | Wan 2.2 | UMT5-XXL | T2V + I2V | ~12.9 GB |
| `wan-video-i2v-a14b.json` | A14B high **+** low noise | Wan 2.1 | UMT5-XXL | I2V only | ~25.6 GB |

The models are stored wherever `TENSORSHARP_MODELS` points, or in the
repository's own `models/` folder when it is unset (the configs resolve
`../models` relative to `config/`, giving `<repository>/models`) — the configs
contain no absolute paths, so the same file works on Windows, Linux and macOS.

**The two VAEs are not interchangeable**: TI2V-5B needs the Wan 2.2 VAE
(48-channel latent, 16×16×4), while A14B and the Wan 2.1 models need the Wan 2.1
VAE (16-channel, 8×8×4). Loading the wrong one fails at startup rather than
producing bad video. UMT5-XXL is shared by all of them, so it downloads once.

To point at files you already have, use `--video-vae`, `--video-text-encoder` and (for A14B)
`--video-dit2`, or drop everything in one folder next to the DiT and let the
same-directory scan find it.

## Backends

Video generation runs on every TensorSharp backend except MLX (`--backend mlx`
is rejected at load):

| `--backend` | Path | Notes |
|---|---|---|
| `ggml_cuda` | GGML whole-graph kernels | fastest; persistent DiT graph + CUDA-graph capture |
| `ggml_metal` | GGML whole-graph kernels | Apple Silicon; the VAE convolutions go through MPSGraph |
| `ggml_vulkan` | GGML whole-graph kernels | AMD/Intel/NVIDIA via Vulkan; the VAE stays on the banded conv path (Vulkan drivers reject multi-GB arenas) |
| `ggml_cpu` | GGML whole-graph kernels | CPU-only machines; slow but exact |
| `cuda` | direct CUDA (`WanDirect*`) | ggml-independent: quantized weights resident via TensorSharp's own MMQ/dp4a/cuBLAS routing, streaming online-softmax attention kernels, channels-last im2col VAE |
| `cpu` | direct pure-C# (`WanDirect*`) | no native dependencies beyond the managed runtime; parallel SIMD GEMM/attention kernels |

All backends produce numerically equivalent videos (verified against each other
and against diffusers' reference implementation on identical seeds/embeddings —
final-latent cosine ≥ 0.999 between backends, ≈ 0.994 vs the bf16 diffusers
reference with quantized checkpoints). The `cpu` / `ggml_cpu` backends are for
functional use — a 480p multi-second video takes tens of minutes on a
high-core-count machine.

Reference timings (RTX 2000 Ada 16 GB, Wan2.1-1.3B F16, 832×480, 33 frames,
30 UniPC steps — the official 480p recipe): `ggml_cuda` 12.0 s/step (445 s
total), `ggml_vulkan` 17.2 s/step (625 s), direct `cuda` 19.3 s/step (700 s).
On shorter sequences the direct backend matches `ggml_cuda` (TI2V-5B
480×480×9f: 1.1 s/step on both); at the 14k-token recipe its F32 attention
path trails ggml's F16 flash attention. Long-sequence attention routes through
chunked cuBLAS GEMM+softmax automatically (`TS_WAN_DIRECT_FUSED_ATTN=0` forces
it everywhere).

## CLI

Text-to-video (any family):

```bash
TensorSharp.Cli --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda \
  --prompt "a cute fluffy orange cat walking through a sunny garden with flowers" \
  --output cat.mp4 --width 832 --height 480 --video-frames 49 --diffusion-seed 7
```

Image-to-video (Wan 2.2 models — pass `--image`):

```bash
TensorSharp.Cli --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda \
  --prompt "the cat runs toward the camera, dynamic tracking shot, golden hour" \
  --image first_frame.png --output cat_run.mp4 --video-frames 81
```

```bash
# A14B: point --model at either expert; the sibling is found automatically
TensorSharp.Cli --model HighNoise/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf \
  --backend ggml_cuda --prompt "the ship sails into the storm, waves crashing" \
  --image ship.jpg --output ship.mp4
```

- `--image <file>` — first-frame conditioning (PNG/JPEG/WebP/…). EXIF
  orientation is honored. When no explicit `--width/--height` is given, the
  output resolution follows the image's aspect ratio at the model's native
  area (TI2V-5B: 1280×704 = the official 720p recipe; others: 832×480).
- `--video-frames N` — frames, snapped to `4k+1`; `--video-frames 1` produces a
  still image (use `--output out.png`).
- `--sampler unipc|euler` — UniPC (default) is the official Wan sampler,
  verified numerically against diffusers' `UniPCMultistepScheduler`.
- `--cfg` / `--flow-shift` / `--diffusion-steps` default to each model's
  official recipe: TI2V-5B cfg 5.0, shift 5.0, 50 steps, 24 fps; A14B I2V
  cfg 3.5 (both experts), shift 5.0, 40 steps; A14B T2V cfg 4.0/3.0, shift
  12.0; Wan 2.1 cfg 6.0, shift 8.0 (1.3B video) or 3.0/5.0, 30 steps, 16 fps.
- `--negative-prompt` defaults to the official Wan negative prompt.
- **Step-distilled checkpoints are auto-detected** and are by far the biggest
  speed lever — 100 DiT passes become 4, for the same video. No flag: it is an
  ordinary `--model` GGUF. See [the fast lane](#the-fast-lane-step-distilled-checkpoints)
  below.
- `--cfg-cache-stride N` (default off) — approximate speedup. A guided step
  is `v = v_cond + (cfg-1)·d` with `d = v_cond - v_uncond`; the guidance
  direction `d` changes much more slowly across the schedule than `v` does, so
  the unconditional pass can run on one step in `N` and the cached `d` cover the
  rest. At 50 steps, `2` runs 77 of the 100 passes (1.30× faster) and `3` runs
  70 (1.43×). The first three steps and the last always recompute `d`. Leave it
  off when matching a reference sample matters. Server: `"cfgCacheStride": 2`.
- MP4 writing prefers `ffmpeg` on `PATH` (or `TS_FFMPEG=<path>`, or an
  `ffmpeg` folder next to the executable) — near-lossless CRF 17 H.264.
  Without it the OS codec via OpenCV is used at its default bitrate, which
  is visibly worse; the CLI warns when that happens.

### The fast lane: step-distilled checkpoints

This is the difference between a 5-second 720p video costing three and a half
hours and costing seventeen minutes, and it is a one-word change to a file path.

TensorSharp reads the DiT GGUF's **file name**. A name carrying `Turbo` /
`distill` / `Lightning` / `lightx2v` / `FastWan` / `-dmd`, or an explicit step
count (`…-4steps-…`, `…8step…`, accepted for 1–16), switches the sampling
defaults to that step count with **guidance off** — which is what those
checkpoints are trained for. On load the console confirms it:

```
step-distilled checkpoint detected -> 4 steps, guidance off (--diffusion-steps / --cfg override)
```

The official 50-step + CFG recipe costs 50 × 2 = 100 DiT passes; a 4-step
guidance-free checkpoint costs **4** — 1/25th of the denoising work. Everything
else (companion files, flags, endpoints, the Web UI) is unchanged, and
`--diffusion-steps` / `--cfg` still override the detected values.

Drop-in GGUFs. The distilled repos ship the DiT only, so the VAE and the text
encoder still come from the base repos:

```bash
# TI2V-5B, 4 steps — note the Wan2_2 underscore in the file name
hf download hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --local-dir models
hf download QuantStack/Wan2.2-TI2V-5B-GGUF VAE/Wan2.2_VAE.safetensors --local-dir models
hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q8_0.gguf --local-dir models

TensorSharp.Cli --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --backend ggml_metal \
  --prompt "the cat runs toward the camera, cinematic" --image first_frame.png \
  --output cat.mp4 --video-frames 121
```

```bash
# I2V-A14B, 4 steps — both experts; --model either one, the sibling is found by name
hf download jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF high_noise/wan2.2_i2v_A14b_high_noise_lightx2v_4step-Q4_K_M.gguf --local-dir models
hf download jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF low_noise/wan2.2_i2v_A14b_low_noise_lightx2v_4step-Q4_K_M.gguf --local-dir models
hf download QuantStack/Wan2.2-I2V-A14B-GGUF VAE/Wan2.1_VAE.safetensors --local-dir models
```

[hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF)
also offers Q6_K (4.22 GB) through Q2_K (1.86 GB);
[jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF)
carries the Lightning distillation already merged into both experts, Q8_0
(15.4 GB/expert) through Q2_K (5.31 GB).
[Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF](https://huggingface.co/Green-Sky/FastWan2.2-TI2V-5B-FullAttn-GGUF)
is a second TI2V-5B option. Note that
[lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning)
publishes LoRA `.safetensors` only, and TensorSharp has no Wan LoRA option — use
the pre-merged GGUFs above. (`--lora` applies to Qwen-Image-2.1 only: the CLI
refuses it with a Wan model, and the server logs a warning and loads the model
without it.)

### Getting good quality

- **Resolution is the biggest lever.** Wan is trained at 480p (832×480 area)
  and 720p (1280×704); TI2V-5B is natively a 720p model. Well below the 480p
  area the DiT is out of distribution and produces soft, unstable, color-drifting
  video no matter how many steps you spend — the pipeline warns below ~0.3 MP.
  Generate at a supported resolution and downscale afterwards instead of
  generating small (e.g. use `--width 480 --height 704` rather than 320×480).
- **Describe the shot.** Wan's prompt adherence rewards detailed prompts —
  subject, action, camera, lighting, style — over 3-4 word fragments; the
  official demos run prompts through an LLM "prompt extend" step first. Short
  prompts leave the model underconstrained and drift-prone.
- More `--diffusion-steps` beyond the recipe (50) buys little; prefer fixing
  resolution and prompt first.

## Server

```bash
TensorSharp.Server.Host --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda \
  --video-frames 121 --fps 24
```

`--video-frames` and `--fps` set server-wide defaults for the Web UI and all
three video endpoints below. The Web UI does not send either field, so the
example produces 121 frames at 24 fps (about 5.04 seconds of playback). An API
request that supplies `frames` or `fps` overrides the corresponding startup
default independently; these are defaults, not caps. If the flags and request
fields are both omitted, the model recipes apply: 49 frames at 24 fps for
TI2V-5B, and 33 frames at 16 fps otherwise. Frame counts are snapped to `4k+1`.
Keep the model's native FPS and adjust the frame count when changing duration;
changing only FPS changes playback speed. `--video-width` / `--video-height`
(also accepted as `--width` / `--height`) and `--video-steps` are server-wide
defaults in the same way: a request's `width` / `height` (or `size`) and `steps`
override them, and without them the model recipe applies.

The step-distilled fast lane is a property of the `--model` GGUF, not of the
request, so it applies to the Web UI and to every endpoint with no client-side
change at all — point `--model` at a Turbo/Lightning checkpoint and nothing else
moves:

```bash
TensorSharp.Server.Host --model Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --backend ggml_metal \
  --video-frames 121 --fps 24
```

This matters most from the browser, where the request is a prompt and an image
and the cost is invisible until the ETA appears. The same Web UI request —
a 1024×768 first frame, 121 frames at 24 fps, which resolves to 1088×832 and
27 404 tokens — takes **≈ 3 h 30 m** on the base `Wan2.2-TI2V-5B-Q8_0.gguf` and
**17 m 30 s** on `Wan2_2-TI2V-5B-Turbo-Q8_0.gguf` (M5 Pro, `ggml_metal`; see
[Performance](#performance)). If a run's ETA looks impossible, check the startup
log for the `step-distilled checkpoint detected` line before changing anything
else.

- **Web UI** (`http://localhost:5000`): type a prompt in the chat — attach an
  image to get image-to-video (the image becomes the first frame); the reply is
  the generated video with live denoising progress.
- **API**:

```bash
curl http://localhost:5000/v1/videos/generations -H "Content-Type: application/json" -d '{
  "prompt": "the cat runs toward the camera, cinematic",
  "image": "data:image/png;base64,....",
  "size": "832x480", "frames": 81, "steps": 50, "seed": 7
}'
# -> { "created": ..., "data": [ { "url": "/uploads/video-....mp4" } ], "frames": 81, ... }
```

  `image` accepts base64 (with or without a `data:` URL prefix); omit it for
  text-to-video. Also `POST /api/video-generate` (same JSON, `{ ok, url, ... }`
  envelope, plus `imagePath` referencing a previously uploaded file) and
  `POST /api/video-generate/stream` (SSE progress, used by the Web UI). A14B
  accepts `cfg2` (low-noise-expert guidance) on every endpoint.

## Implementation notes

The networks each run as ONE resident-weight ggml graph per invocation
(`TensorSharp.GGML.Native/ggml_ops_wan.cpp`):

- `TSGgml_WanT5Encode` — 24-layer UMT5 encoder with per-layer relative attention
  bias; tokens in, hidden states out (conditioning is zero-padded to 512 tokens,
  Wan semantics).
- `TSGgml_WanDitForward` — one denoising-step velocity prediction; persistent
  per shape on CUDA so ggml's CUDA-graph capture removes the launch overhead;
  flash attention over the 3D-RoPE self-attention. For TI2V-5B image-to-video
  the graph modulates two token segments at different timesteps (the
  conditioning frame's tokens at t=0, diffusers `expand_timesteps` semantics)
  — attention still runs over the full joint sequence.
- `TSGgml_WanVaeDecode` — the causal 3D video VAE decoder (both generations),
  iterating temporal chunks in-graph with the causal feature cache carried
  between chunks; convs run as banded im2col+GEMM under a scratch budget so
  peak VRAM stays bounded (`TS_WAN_VAE_GEMM_MAX_MB`; 384 MB on CUDA — sizing it
  from free memory measured **2.2x slower**, because the scratch then eats the
  headroom the rest of the graph needs and WDDM pages the overflow — free/8
  clamped to 384 MB–8 GB on Metal, which has no PCIe cliff, and 384 MB on the
  other GGML backends; `0` forces direct convolution). On Metal the budget
  applies only with `TS_WAN_VAE_MPS_CONV=0`, because by default MPSGraph runs
  each convolution whole; while the Metal 4 tensor API is active (M5-class
  devices running a 14B-class model) it defaults to `0`. The kd temporal taps
  of a causal conv share ONE im2col instead of lowering the same pixels once
  each, and bands are split evenly rather than walked at a fixed height; both
  are bit-identical. The cross-chunk feature caches are stored in F16. The
  Wan 2.2 decoder adds the weight-free DupUp3D residual shortcuts and the final
  2×2 pixel unpatchify. Above a per-band pixel budget (on the GGML backends it
  scales with free device memory: about 0.64 MP at 16 GB free, never below
  0.16 MP; the direct `cuda`/`cpu` backends use 0.3 MP), the decode is
  additionally tiled into full-width horizontal
  bands blended over an 8-latent-row overlap (the diffusers `enable_tiling`
  approach, ~59 dB vs the untiled decode): a 720p plane's activations plus
  causal caches are otherwise ~12 GB device-resident, which pushes 16 GB
  WDDM cards into shared-memory paging (33 min → ~3 min for 69-frame 720p).
  `TS_WAN_VAE_TILE=0` disables the tiling.
- `TSGgml_WanVaeEncode` — the causal 3D VAE encoder (both generations) behind
  image-to-video conditioning: 1+4k pixel-frame chunks, stride-2 spatial and
  temporal downsampling with the causal caches, AvgDown3D shortcuts and 2×2
  pixel patchify for Wan 2.2.

I2V conditioning follows the reference pipelines exactly: TI2V-5B replaces the
first latent frame with the VAE-encoded image every step and modulates its
tokens at timestep 0; A14B concatenates a 4-channel first-frame mask plus the
VAE-encoded `[image, zeros…]` clip behind the 16 noise channels (36-channel DiT
input) and switches from the high-noise to the low-noise expert at
`t < boundary·1000` (0.9 I2V / 0.875 T2V), releasing the first expert's VRAM at
the switch.

Numerics are verified against the reference implementations by guarded tests
(`InferenceWeb.Tests/WanVideoOracleTests.cs`, fixtures generated by
`InferenceWeb.Tests/tools/wan22_oracle.py`): the TI2V-5B DiT (Q8_0) matches
diffusers' `WanTransformer3DModel`
at cosine > 0.995 in both uniform and per-token-timestep modes, both VAE
encoders match `AutoencoderKLWan` at cosine > 0.999, the Wan 2.2 VAE decode at
> 35 dB PSNR (Wan 2.1 decode: 59.9 dB), and the tokenizer reproduces the
HuggingFace T5 ids exactly on English and CJK prompts.

Environment knobs: `TS_WAN_DIT_CAPTURE=0` (disable the persistent captured DiT
graph), `TS_WAN_DIT_FLASH=0` (materialized attention), `TS_WAN_DIT_KV_F16=0`
(F32 attention keys/values — the old, ~2x slower default), `TS_WAN_HEARTBEAT_S`
(progress tick interval, default 30; `0` silences the ticks),
`TS_WAN_VAE_GEMM_MAX_MB` (im2col budget), `TS_WAN_DIT_FFN_CHUNK_MB` (DiT
feed-forward token-chunk budget, default 256 MB; `0` disables chunking),
`TS_WAN_VAE_TAP_SHARE=0` (one im2col per temporal tap instead of one shared
across them — A/B only; the shared lowering is bit-identical),
`TS_VAE_CUDNN_CONV=1` (opt in to cuDNN for the VAE convolutions on CUDA; OFF by
default because it wins ~1.2x on a short clip and loses ~4.5x on a 121-frame one
-- each convolution runs outside the graph, so the host round trips scale with
the temporal chunk count), `TS_CUDNN_DIR` (cuDNN install to use),
`TS_WAN_VAE_MPS_CONV=0`
(ggml conv lowering instead of MPSGraph on Metal),
`TS_WAN_METAL_TENSOR_API=1|0` (force the Metal 4 tensor API on or off; default on
for 14B-class DiTs, off for smaller ones — see
[below](#the-metal-4-tensor-api-and-how-not-to-test-it)),
`TS_WAN_VAE`/`TS_WAN_TE`/`TS_WAN_DIT2` (companion paths), `TS_FFMPEG` (ffmpeg
path for MP4 export), `TS_WAN_DIT_TRACE=<file>` (per-stage activation stats for
debugging).

## Performance

**The headline: use a step-distilled checkpoint.** Every other lever on this
page is worth tens of percent; that one is worth 25×. The same 5-second 720p
image-to-video request runs in **3 h 30 m** on a base checkpoint and
**17 m 30 s** on a Turbo one, on the same machine, with the same flags — see
[the fast lane](#the-fast-lane-step-distilled-checkpoints) for where to get one
and the M5 Pro table below for the breakdown.

RTX 3080 Laptop 16 GB (Windows/WDDM), ggml_cuda:

| Model / workload | Text enc | Image enc | Denoise | VAE decode | Total |
|---|---|---|---|---|---|
| **Wan2_2-TI2V-5B-Turbo Q4_0** I2V 1088×832×121f, 4 steps (720p, 5 s) | 5.9 s | 6.1 s | 27.8 s/pass (27 404 tokens) | 305 s | **429 s** |
| **Wan2.2-TI2V-5B Q8_0** I2V 832×480×81f, 30 steps | 3.5 s | 8.5 s | 11.5 s/step (CFG ×2, 8190 tokens) | 103 s | 464 s |
| Wan2.2-TI2V-5B Q8_0 T2V 640×384×25f, 20 steps | 3.7 s | — | 1.5 s/step | 23 s | 65 s |
| Wan2.2-I2V-A14B Q4_K_M 480×480×9f, 6 steps (smoke) | 3.4 s | 5 s | 7–8 s/step + 2 expert loads | 7.4 s | 89 s |
| Wan2.1-T2V-1.3B F16 832×480×33f, 30 steps | 3.1 s | — | 9.4 s/step (CFG ×2) | 51 s | 337 s |

That 720p/121-frame row was **1 143 s** before the 2026-08 pass: the DiT graph
needed 11.4 GiB at 27 404 tokens and the card has 16 GB, so the whole denoise ran
against a working set that did not fit. Chunking the feed-forward over tokens
(`TS_WAN_DIT_FFN_CHUNK_MB`) took the 1.5 GiB intermediate out of the peak and the
per-pass time went 110.4 s -> 26.1 s; capping the VAE im2col scratch and sharing
one lowering across the causal taps took the decode 712 s -> 305 s. Both are
exact: the DiT still matches diffusers at cosine 0.99997 and the decode at
80.2 dB PSNR, unchanged from before.

The TI2V-5B model's 16×16 spatial compression gives it ~2.7× fewer DiT tokens
than Wan 2.1 at the same resolution — it is both the fastest and the
highest-quality option for consumer GPUs, and the only 720p-24fps one.

### Long sequences: cost is quadratic in the token count

Token count is `latent_frames × (h/2) × (w/2)`, and DiT self-attention costs
`O(tokens²)`. Past a few thousand tokens attention — not the weight matmuls —
is where the time goes, so the frame count and the frame area both matter far
more than the step count:

| Request (TI2V-5B) | Tokens | Attention work |
|---|---|---|
| 640×384×25f | 1 200 | 1× |
| 832×480×81f | 8 190 | 47× |
| 1088×832×121f (5 s at 24 fps, 720p class) | 27 404 | 520× |

The full 5-second 720p recipe is a genuinely large job: 50 steps × 2 CFG passes
= 100 DiT passes over 27 k tokens. The pipeline prints the token count, a
per-pass timing and a running ETA, and heartbeats every 30 s while a pass is in
flight, so a long run is visibly progressing rather than apparently hung:

```
Wan 2.2-TI2V I2V: 1088x832x121f (31x26x34 = 27404 tokens), 50 steps, unipc, cfg 5, shift 5, seed ...
  [wan] large request: 100 DiT passes over 27404 tokens. Self-attention costs O(tokens^2) ...
  step 1/50 t=999.0 (249.0s: cond 124.6s + uncond 124.4s) — ~3h 24m left
  [wan] …denoise step 2/50 (cond pass) 60s in this pass, 315s total, ~3h 24m left
```

To make such a request cheaper, in order of effect:

1. **Use a step-distilled checkpoint.** This dwarfs everything else: 100 DiT
   passes become 4. See [the fast lane](#the-fast-lane-step-distilled-checkpoints)
   — TensorSharp detects them by name and switches the recipe automatically.
2. **Fewer frames.** 121 → 61 frames roughly quarters the attention work and
   halves the VAE decode; the video is 2.5 s instead of 5 s at 24 fps.
3. **Smaller frame area**, but not below Wan's training resolutions — under
   ~0.3 MP the model is out of distribution and the result gets *worse*, not
   just cheaper. `--width 480 --height 704` is a good portrait target.
4. **Fewer steps** (base checkpoints only). 50 is the official TI2V recipe;
   30 is visibly close and 1.7× cheaper.
5. **`--cfg-cache-stride 2` or `3`** — 1.30× / 1.43× by reusing the guidance
   direction between steps (approximate; see the CLI section). Pointless on a
   distilled checkpoint, which already runs guidance-free.

M5 Pro (20-core GPU, 48 GB unified), `ggml_metal`, Wan2.2-TI2V-5B Q8_0,
1088×832×121f = 27 404 tokens, image-to-video — the request this was all
measured against (5 s at 24 fps, 720p class):

| | Base checkpoint, before | Base checkpoint, now | **Turbo checkpoint, now** |
|---|---|---|---|
| DiT passes | 100 (50 steps × CFG) | 100 | **4** (4 steps, guidance-free) |
| per pass | 206.2 s | **120.2 s** | **120.2 s** |
| denoise total | 20 615 s | 12 020 s | **481 s** |
| VAE decode, 121 frames | 863 s | **563 s** | **563 s** |
| **end to end** | **≈ 5 h 58 m** | ≈ 3 h 30 m | **17 m 30 s** |

The two levers are independent: ~1.7× per pass from the attention and VAE work
below, and 25× fewer passes from running a step-distilled checkpoint. Once
distilled, the **VAE decode is the bottleneck** (~55% of the run), not the DiT.

Frame count and frame area then set the rest, and the VAE scales linearly with
output pixels while attention scales with the square of the token count. All
measured on the same machine with the same Turbo checkpoint and image:

| Output | Tokens | Denoise | VAE decode | **Total** |
|---|---|---|---|---|
| 736×544 × 81f (3.4 s, 480p class) | 8 211 | 84 s | 159 s | **4 m 09 s** |
| 736×544 × 121f (5 s, 480p class) | 12 121 | 137 s | 237 s | **6 m 19 s** |
| 1088×832 × 121f (5 s, 720p class) | 27 404 | 481 s | 563 s | **17 m 30 s** |

480p (≈0.4 MP) is a resolution Wan is *trained* at, so the middle rows are
in-distribution, not a degraded mode — that is the setting to reach for when a
few minutes matters. Going below ~0.3 MP is where quality actually falls off.

Three changes account for the per-pass part.

**F16 attention keys and values.** Every backend's flash-attention kernel is
built around an F16 KV cache: ggml-metal instantiates the F32 variant with
`simdgroup_float8x8` accumulators where the F16 one uses `simdgroup_half8x8`,
and the F32 tiles also cost twice the bandwidth in a kernel that re-streams K
and V once per 8-query threadgroup. The permute into flash-attention layout and
the narrowing happen in one `ggml_cpy`, so the F16 path also writes half the
bytes the old `ggml_cont` did. `TS_WAN_DIT_KV_F16=0` restores F32.

**VAE decode band layout.** Tiling picks the band *count* from the memory budget
and then splits the plane evenly, instead of walking a fixed band height and
letting the last band land wherever the stride puts it. At 52 latent rows that
was three 24-row bands — 72 rows decoded to produce 52 — and is now two 30-row
bands (60 rows, one seam instead of two) within the same per-band budget.

**VAE convolutions on MPSGraph (Metal).** ggml lowers `conv2d` to im2col +
`mul_mat`, and a decode profile (`WanVideoBench`, 640×480×9f) puts **44.4% of
the graph in MUL_MAT and a further 30.2% in IM2COL** — 74.6% in the convolution,
with im2col moving 865 MB per node at ~57 GB/s. The lowering is pure overhead: a
3×3 conv materialises 9× its input before the GEMM starts. Apple's tuned
convolution runs the same shapes far faster:

| conv shape | ggml im2col+GEMM | MPSGraph | |
|---|---|---|---|
| 512→512 k3 320×240 t9 | 666 ms | 108 ms | 6.2× |
| 256→256 k3 640×480 t9 | 947 ms | 110 ms | 8.6× |
| 160→160 k3 640×480 t9 | 491 ms | 47 ms | 10.4× |
| 512→512 k1 320×240 t9 | 184 ms | 14 ms | 13.9× |

MPS reaches ~30 TFLOP/s where ggml's Metal GEMM gets ~4.9 — and it does so
*without* ggml's `mul_mm`, the kernel whose Metal 4 tensor path corrupts this
exact graph. It buys matrix-unit throughput while stepping around that defect.

Layout costs nothing: ggml's activation `[W,H,C,T]` and kernel `[KW,KH,IC,OC]`
are byte-for-byte MPS NCHW and OIHW, and NCHW measured as fast as NHWC here, so
no transpose is inserted. On this path the VAE emits un-lowered `CONV_2D` nodes,
which also skips the horizontal banding and its leading-pad shim entirely; the
decode graph is then run in segments with each `CONV_2D` handed to MPSGraph
(`ggml_graph_view`, the same mechanism the node profiler uses, so it is safe
against gallocr reuse). Measured end to end: **VAE decode 159 s → 80 s (1.99×)**
at 736×544×81f. Numerics are unchanged — ggml already convolves in F16 with F32
accumulation and so does this, agreeing to **93.9 dB PSNR / max Δ 0.128 of 255**
over a whole decode. `TS_WAN_VAE_MPS_CONV=0` restores the ggml lowering.

**VAE conv im2col budget, and the tiling threshold itself, sized from device
memory.** Both were pinned at a 16 GB card's budget on every non-CUDA backend.
Metal now sizes the im2col budget from free device memory (free/8, capped at
8 GB); CUDA keeps a fixed 384 MB, because sizing it from free memory measured
2.2x slower there, and Vulkan (like ggml_cpu) keeps a fixed 384 MB because its
drivers reject multi-GB arenas. The tiling threshold now scales with free device memory on
every GGML backend. A bigger im2col budget turns many
small banded GEMMs into few large ones: 56.4 s → 49.7 s for a 1088×832×9f
decode. And tiling is not free — the bands overlap, so the decoder runs more
latent rows than the plane has, and the band buffer lives alongside the canvas.
Decoding 1088×832×121f **whole** measured both faster *and* lighter than two
bands: **565 s vs 655 s, peak RSS 4.85 vs 5.37 GB**. Small cards still tile.

#### Alternatives measured and rejected

One self-attention at this shape (seq 27 404, 24 heads, head dim 128), same run:

| KV type | time | vs F32 |
|---|---|---|
| F32 (before) | 4993 ms | 1.00× |
| **F16 (now)** | **2467 ms** | **2.02×** |
| Q8_0 | 2652 ms | 1.88× |
| Q4_0 | 2395 ms | 2.08× |

Q8_0 keys/values are *slower* than F16 — the dequantization outweighs the
bandwidth saved — and Q4_0 buys 3% for a real precision cost, so F16 is the
sweet spot rather than a compromise. Materialized chunked attention
(`mul_mat` + `soft_max_ext`, which unlike flash attention *can* use the Metal 4
tensor API) came in at 1.08×: at this size attention is bound by score traffic,
not by the GEMM rate. Dequantizing the DiT weights from Q8_0 to F16 moved the
matmuls by <5% (25.2 → 26.4 ms for an attention projection).

#### The Metal 4 tensor API, and how not to test it

The most tempting knob here, and it stays **off** for TI2V-5B. It measures 1.20×
on the DiT (121.4 → 100.8 s/pass) and 1.66× on the VAE decode, because it routes
`mul_mm` through Metal 4 tensor operations — but on M5 / macOS 26.6 it also
miscomputes the VAE's conv GEMMs and the video comes out uniformly **black**.

For A14B and other 14B-class DiTs (patch-embedding width ≥ 5120) it is **on** by
default on M5-class Macs, because the DiT gain is larger there: A14B I2V measured
17.1 vs 30.2 s/step (1.77×) at 480×480×9f on an M5 Pro
([Wan video and the tensor API](../../DEVELOPMENT.md#wan-video-and-the-tensor-api)).
The VAE's convolutions are kept off the tensor-API `mul_mm` in that case: they run
on MPSGraph (the default), or on ggml's direct convolution instead of im2col+GEMM
when `TS_WAN_VAE_MPS_CONV=0`. `TS_WAN_METAL_TENSOR_API=1` / `0` forces the tensor
API on or off for the process, overriding the model-class default.

The trap is that this does not reproduce in isolation. `WanVideoBench
vae-decode` was run at five latent shapes, including the 32×32 layout recorded
as the original all-NaN repro, dumping raw pixels with the tensor API on and
off: they agreed to **91–93 dB PSNR, max delta 0.18/255, zero NaN**. On that
evidence the workaround was removed — and the very next full 1088×832×121f
generation produced 121 black frames. The defect follows the process's
allocation history (the DiT is loaded and released before the decode), so a
fresh-process decode probe reproduces the wrong buffer layout.

It was then re-tested a second time with a *much* more faithful harness —
`WanVideoBench pipeline`, which runs VAE-encode → DiT forward → release → VAE
decode in one process at the exact production shapes, including the full
121-frame decode. That came back clean too, and 1.50× faster (374 s vs 563 s).
The real generation still produced flat frames. The only remaining difference is
the UMT5-XXL text encoder that loads and releases before everything else, which
is the largest allocation in the pipeline.

So: re-enabling it requires a full end-to-end video, and nothing less has ever
been sufficient. The pipeline now refuses to return a uniformly flat video and
names this cause (`AssertFramesAreNotDegenerate`), which is what caught the
second attempt automatically instead of costing another eyeballed PNG.

#### Quality

F16 keys and values change the DiT's output by less than the sampler's own
sensitivity to floating-point reassociation:

- One DiT forward against the diffusers `WanTransformer3DModel` reference
  (`WanVideoOracleTests`): cosine **0.999964** with F16 K/V and **0.999964**
  with F32 K/V — the per-pass accuracy is unchanged.
- Full 25-frame generations from an identical seed: F16 vs F32 K/V differ by
  39.08 dB mean PSNR. The control — F32 K/V with flash attention *off*, i.e. the
  same arithmetic in a different summation order — differs by 39.40 dB. The two
  are the same magnitude, so the pixel difference is an 8-step diffusion
  trajectory diverging from a rounding difference, not a loss of accuracy.

The speedup only appears where attention dominates: at 2 310 tokens
(480×704×25f) F16 and F32 K/V both run 5.0 s per pass.

Wan 2.1 vs stable-diffusion.cpp (master-769, identical GGUFs/settings,
33-frame 480p): sampling is near parity (281 s vs 258 s) but sd.cpp's VAE
decode materializes ≈8 GB of 3D im2col and oversubscribes a 16 GB card into
WDDM paging (1762 s vs **51 s**) — TensorSharp is 6.0× faster end to end.
