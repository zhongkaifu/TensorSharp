# Qwen-Image-2.1

Qwen-Image-2.1 uses the `QwenImageModel` image pipeline for text-to-image generation
and image editing. It requires a Qwen3-VL-8B text encoder and the dedicated 2.1
VAE. Native RGBA input and PNG output preserve transparency; the Qwen3-VL
conditioning branch composites the reference over white while the VAE keeps alpha.
Earlier Qwen-Image / Qwen-Image-Edit checkpoints (such as Qwen-Image-Edit-2511) are
no longer supported and are refused at load (exit code 2). The `--qwen-image-lora`
option was replaced by `--lora` ([LoRA plug-ins](#lora-plug-ins)), and
`--offload-cpu` was removed; either flag, on the command line or as a config-file
key, now stops the CLI or server with a configuration error that says what to use
instead. The old `TS_QWEN_IMAGE_LORA` environment variable is refused at load
(exit code 2) with the same advice: pass the LoRA with `--lora` and unset the
variable.

The download configuration is [`config/qwen-image-2.1.json`](../../config/qwen-image-2.1.json).
It pins repository revisions and SHA-256 checksums for new downloads; existing
cached files are reused. The four files total
11,051,668,216 bytes (about 10.29 GiB); this is download size, not peak inference
memory. Runtime memory also includes activations, decoding buffers and working
weights. Larger images and multiple references increase that requirement.

| Component | File | Source |
|---|---|---|
| Diffusion transformer | `qwen_image_2.1_Q4_K_M.gguf` | [Abiray/Qwen-Image-2.1-GGUF](https://huggingface.co/Abiray/Qwen-Image-2.1-GGUF) |
| Dedicated VAE | `qwen_image_2.1_vae_bf16.safetensors` | [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/vae) |
| Text encoder | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) |
| Vision encoder for editing | `mmproj-Qwen3VL-8B-Instruct-F16.gguf` | Same Qwen3-VL repository |

## Launch the CLI

Run these commands from the TensorSharp repository root. The configuration
selects `ggml_metal` for Apple Silicon. On an NVIDIA machine with the CUDA backend
built, append `--backend ggml_cuda`; the native CPU backend is `ggml_cpu`.
Missing models download automatically at startup. Set `TENSORSHARP_MODELS` to an
absolute directory to choose where they are stored; files go in its
`qwen-image-2.1` subdirectory. Without this variable, the configuration resolves
`../models` relative to `config/`, giving `<repository>/models/qwen-image-2.1/`.

For the models downloaded in this workspace, set:

```bash
export TENSORSHARP_MODELS="$PWD/../models"
```

Build:

```bash
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release
```

Text-to-image:

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --prompt 'A small orange cat beside a blue ceramic vase, soft daylight, detailed photograph' \
  --width 2048 --height 2048 --diffusion-steps 40 --cfg 1 \
  --diffusion-seed 42 --output generated.png
```

Image editing:

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --image generated.png \
  --prompt 'Change the blue vase to a red vase. Preserve the cat, lighting and composition.' \
  --width 2048 --height 2048 --diffusion-steps 40 --cfg 1 \
  --diffusion-seed 42 --output edited.png
```

No `--image` selects generation; one or more `--image` arguments select editing.
Repeat `--image first.png --image second.png` for multiple references in that order.
Each reference is tagged `<image1>`, `<image2>`, … in command-line order ahead of the
prompt, so the prompt can name a picture by its tag.
`--input prompt.txt` can supply the prompt instead. Omitted sampling settings
select **40 Euler steps and CFG 1.0**, following
[Qwen's recommended unguided sampling](https://github.com/huggingface/diffusers/blob/main/docs/source/en/api/pipelines/qwenimage21.md).
CFG 1 runs one transformer prediction per step; the previous CFG 6 default ran
both positive and negative predictions. Explicit CFG above 1 still enables the
second prediction and conditions it on `--negative-prompt` (for example
`--negative-prompt 'blur, low detail'`; without one, the negative branch uses an
empty prompt). Negative prompts have no effect at CFG 1.

Omitting dimensions selects **2048×2048 for generation**, or approximately the
same pixel area with the first reference's aspect ratio for editing. Set width
and height together, in multiples of 32, to override this. The model supports
[native 2K aspect ratios](https://github.com/QwenLM/Qwen-Image-2.1#supported-aspect-ratios).
Reference images are conditioned at approximately 1 megapixel each, or the
output area if smaller; increasing the output to 2K does not also quadruple each
reference's VAE, vision-encoder and transformer workload.

The schedule now follows the
[official scheduler configuration](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json):
exponential dynamic shifting with the 256/0.5 and 8192/0.9 sequence-length/shift
anchors, followed by terminal stretching to 0.02 and a final Euler step to zero.
This replaces the earlier Flux-derived 4096/1.15 schedule, so existing seeds can
produce different images after this correction.

For faster drafts, specify `--width 1024 --height 1024`. A 2K square has four
times the latent image tokens and more attention work than a 1K square.
`--diffusion-steps 25 --cfg 1` is an optional faster profile used by the official
ComfyUI [generation](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json)
and [editing](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json)
workflows. Forty steps remains the default; fewer steps are a quality/speed
tradeoff, not a claim of equivalent image quality.

The CFG 1 speed improvement uses the released 2.1 checkpoint directly.
Step-distillation LoRA plug-ins reduce the step count further, to 4–8 transformer
passes; see [LoRA plug-ins](#lora-plug-ins).

For a quick executable smoke test use 256×256 and one step. Such a run verifies
loading and the end-to-end data path; it does not demonstrate image quality or
performance at the default 2048×2048/40-step settings.

## Launch TensorSharp.Server.Host

```bash
dotnet run --project TensorSharp.Server.Host -c Release --no-build -- \
  --config config/qwen-image-2.1.json --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. A prompt without an attachment generates an image;
attach one or more images to edit. The page displays denoising progress and the
output download link. Image operations are serialized because the diffusion
pipeline shares mutable working state.

Existing Unsloth downloads can be passed directly after rebuilding the host:

```bash
TensorSharp.Server.Host/bin/TensorSharp.Server.Host \
  --model ~/work/models/qwen-image-2.1-unsloth/qwen-image-2.1-Q8_0.gguf \
  --qwen-image-vae ~/work/models/qwen-image-2.1-unsloth/qwen_image_2.1_vae_bf16.safetensors \
  --qwen-image-vl ~/work/models/qwen-image-2.1-unsloth/Qwen3-VL-8B-Instruct-Q4_K_M.gguf \
  --qwen-image-mmproj ~/work/models/qwen-image-2.1-unsloth/mmproj-BF16.gguf \
  --backend ggml_metal
```

The loader recognizes metadata-free diffusion GGUFs by their tensor layout,
including the `model.diffusion_model.` prefix. The dedicated 2.1 VAE accepts
both original Wan names and Diffusers names with spatial convolution kernels;
the adapter preserves the stored weights. No model-file conversion is required.

Text-to-image JSON API:

```bash
curl --fail-with-body http://127.0.0.1:5000/api/image-generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"An orange cat beside a blue ceramic vase, soft daylight","width":2048,"height":2048,"steps":40,"cfg":1,"seed":42}'
```

The response is `{ "ok": true, "url": "...", "width": 2048, "height": 2048,
"elapsedSeconds": ... }`. Download the returned URL from the same server.

Multipart image editing:

```bash
curl --fail-with-body http://127.0.0.1:5000/api/image-edit \
  -F 'image=@generated.png' \
  -F 'prompt=Change the blue vase to red. Preserve the cat and composition.' \
  -F 'width=2048' -F 'height=2048' -F 'steps=40' -F 'cfg=1' -F 'seed=42'
```

Repeat the `image` part for multiple references. Alternatively upload files to
`/api/upload` and send JSON containing `imagePaths` (or legacy `imagePath`) to
`/api/image-edit`. Both image endpoints accept `negativePrompt`, `targetArea`,
`width`, `height`, `steps`, `cfg` and `seed`. `targetArea` controls automatic
geometry; explicit dimensions take precedence. Omitting `width`, `height`,
`targetArea`, `steps` and `cfg` selects the model defaults above. `targetArea: 1048576`
selects approximately 1K output while retaining automatic aspect-ratio selection.

Starting the server with `--width` and `--height` changes that default size. The
host publishes them as `TS_QWEN_IMAGE_WIDTH` / `TS_QWEN_IMAGE_HEIGHT`, and every
image request that sets neither `width`/`height` nor an explicit `targetArea` then
uses that size, including Web UI requests, which send no size; an edit then no
longer follows the first reference's aspect ratio. A request that sets its own
`targetArea` keeps its own geometry. The default needs both flags. A value that is
not a multiple of 32 is rounded down to one (never below 32), with a one-time
`[qwen-image] WARNING: … render at WxH instead. Reported once.`; with only one of
the two set, or an unparsable or negative value, the default is ignored with a
one-time warning and the automatic size stays. A Qwen-Image server also warns at
startup in either case, and nothing is refused. A `width` / `height` set in the
request itself must still be a positive multiple of 32. On the server the same two
flags are also the aliases of `--video-width` / `--video-height`.

For progress, use the JSON routes `/api/image-generate/stream` and
`/api/image-edit/stream` with `curl -N`. They emit SSE `data:` frames with
`imageGenerate: true` or `imageEdit: true`, `step` and `total`, optionally a
preview `image` data URL. The terminal frame contains `done: true` and the final
`url`, dimensions and elapsed seconds, or `error`. Chat-completion routes do not
run this diffusion model. Existing `/api/image-edit` requests still require
at least one reference; generation has its own endpoint.
Previews decode the estimated clean latent from the current flow prediction.

The 2.1 diffusion transformer runs a complete GGML graph with resident quantized
weights; there is no CPU weight-streaming mode. Start with smaller dimensions if
available memory is insufficient. CUDA and Vulkan were exercised on NVIDIA A40s; the
measurements below record where.

## LoRA plug-ins

TensorSharp applies LoRA adapters to the 2.1 diffusion transformer at run time:
style and editing LoRAs, DoRA, and step-distillation adapters that replace the
40-step default with 4–8 transformer passes. Pass them with `--lora` on the CLI or
the server, and repeat it to stack several. `--lora-scale` sets the strength of the
preceding `--lora`, and `--lora-config` names its companion config. The plug-ins in
[`config/lora/`](../../config/lora/) download and hash-check their weights on first
use and carry the adapter's strength and sampling recipe:

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --lora config/lora/qwen-image-2.1-viggle-turbo.json \
  --prompt 'A small orange cat beside a blue ceramic vase, soft daylight, detailed photograph' \
  --width 1024 --height 1024 --diffusion-seed 42 --output turbo.png
```

The flag reference, the plug-in config format and the table of the twelve shipped
plug-ins are in [USAGE.md](../../USAGE.md#qwen-image-21-lora-plug-ins).

### Sampling recipes

A step-distillation LoRA is trained for one schedule, so its plug-in records that
schedule. The shipped recipes follow the adapters' model cards:

- **Viggle Turbo** (6 steps by default, 4–8 supported, CFG 1). The raw nodes, for
  example `[1, 0.9375, 0.875, 0.75, 0.5, 0.25]` at 6 steps, pass through the
  checkpoint's resolution-dependent exponential shift (the 256/0.5 and 8192/0.9
  anchors) but not through the base scheduler's terminal stretch, followed by a
  final 0 (`"shift": "dynamic"`).
- **Pruna 8-step and 5-step** (CFG 1). The card's sigmas are used verbatim, with no
  shift, followed by a final 0 (`"shift": "none"`).
- **Fun-Acc 4-step** (CFG 1). The PDD bundle's trained grid from `pdd_config.json`
  is used verbatim at every resolution. The timestep is rounded through bf16, as the
  reference hook does, and step *i* uses output head *i*.

Explicit settings win over a recipe, and a recipe wins over the model defaults (40
steps, CFG 1): `--diffusion-steps` / `--cfg` on the CLI, and a request's `steps` /
`cfg` on the server (`0` or omitted selects the recipe). A step count the recipe
has no schedule for is refused, and the error lists the supported counts; a PDD
bundle, which has one output head per trained step, runs only its trained count.
Two plug-ins that both carry a recipe cannot be stacked. Style and editing
plug-ins carry no recipe and keep the checkpoint's own schedule. The run logs the
resolved recipe and its sigmas before denoising.

### Supported formats

- **Tensor names** from diffusers / PEFT (`transformer.` prefix, `lora_A` /
  `lora_B`, adapter slot names such as `lora_A.default.weight`), ComfyUI and
  ai-toolkit (`diffusion_model.`), DiffSynth / ModelScope (no prefix), kohya
  (`lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight`), and `lora_down` /
  `lora_up` with or without `.weight`.
- **Alpha** from a per-module `.alpha` tensor, the PEFT config a diffusers saver
  embeds in the safetensors metadata (`lora_adapter_metadata`), an
  `adapter_config.json` (`lora_alpha`, `alpha_pattern`, `use_rslora`); otherwise alpha
  equals the rank. kohya's `ss_network_alpha` is training metadata and is ignored, as
  in ComfyUI and diffusers (a converter that dropped the per-module `.alpha` tensors
  folded alpha into the factors). An explicit config wins over the file's own
  metadata, and a PEFT folder's `adapter_config.json` still supplies alpha beside a
  recipe-only config. The Pruna files record alpha 128 at rank 64, so their
  applied scale is 2; a loader that assumes alpha = rank applies half the adapter.
- **DoRA** `dora_scale` magnitudes, with ComfyUI's semantics on the output axis:
  the magnitude is divided by the row norms of the checkpoint's own weight,
  dequantized from the GGUF.
- **VideoX-Fun PDD bundles**: per-step output heads that replace `proj_out`,
  replaced norm gains, and the low-rank deltas. The bundle's `pdd_config.json` is
  read from beside the weights or from `--lora-config`; only `pdd_block_size` 1 (one
  head per step) is supported.
- **1-D `.diff` tensors** on the norm gains (`txt_in.text_norm`, `attn.norm_q`,
  `attn.norm_k`).
- **Split MLP projections.** Diffusers' separate `img_mlp.gate_layer` and
  `img_mlp.proj` are the gate and up halves of the checkpoint's fused
  `img_mlp.gate_up` (gate first), and the loader places them there.
- **PEFT folders.** An `adapter_model.safetensors` with an `adapter_config.json`
  beside it picks up that config without `--lora-config`.

Refused, with a message that names the tensor: LoKr, LoHa and LoCon mid factors;
text-encoder LoRAs; bias terms and `diff_b` (the transformer has no biases); 2-D
full-weight diffs, and full `.weight` values outside a PDD bundle; convolution
LoRAs; a file holding several PEFT adapters; and LoRAs made for other models, such
as the dual-stream 20B Qwen-Image (`add_q_proj`, `txt_mlp` or `img_mod` modules),
or any factor whose shape does not match the 2.1 projection. Every tensor in a file
is applied or the load fails; nothing is skipped silently, because a partly applied
LoRA would not be the adapter that was asked for.

### How the update is applied

The base weights stay quantized as stored, and each adapted projection computes
`y = W x + B (A x)`. A step-distillation LoRA moves the weights by about 0.1–0.5%,
which is the size of Q8_0's own rounding step, so dequantizing, adding the delta
and requantizing loses most of it. On the Pruna adapter's block 0, the delta that
survived such a merge had a cosine similarity of 0.07 with the intended delta.

- The strength, alpha / rank and any DoRA magnitude are folded into `B` in F32.
  Each rank component is then rebalanced so that its `A` row and `B` column have
  equal norms, which leaves the product unchanged, and the factors are stored in
  F16. A group in which some value would overflow F16 stays in F32.
- Several LoRAs on one projection are concatenated along the rank, so the graph
  runs one shrink and one expand per projection whatever the number of plug-ins.
- Ranks are zero-padded to a multiple of 64 on Metal, whose simdgroup matrix
  kernel needs K ≥ 64, and to a multiple of 16 elsewhere.
- Q, K and V read the same input, so their `A` factors are stacked and one shrink
  serves all three; the gate and up halves share theirs the same way.
- Adapted projections are the image and text inputs, the timestep embedding, the
  modulation, `norm_out`, `proj_out`, and in each of the 32 blocks Q, K, V, the
  attention output, gate, up and down.

This runs on every backend the model runs on: `ggml_metal`, `ggml_cuda`,
`ggml_vulkan` and `ggml_cpu`. The load logs the plug-in count and the size of the
packed factors (`Qwen-Image-2.1 LoRA: N plug-in(s), applied unmerged (... MiB of
factors, ...)`), then one line per file with its update count, ranks and scales.

**Prefix KV cache.** The cache stays on. The first step runs the whole sequence
through the adapted transformer, so the stored text and reference keys and values
include the LoRA. Retained graphs and stored prefixes are keyed on the adapter's
factor buffers, so one built with other factors, or none, is never reused.

**Tensor parallelism.** Under `--tp N`, a column-parallel projection (Q, K, V, gate,
up) gives each GPU the rows of `B` (and of any DoRA row scale) for its own output
slice. A row-parallel projection (`to_out`, `img_mlp.out`) gives each GPU the
columns of `A` for its input slice, and each GPU adds its partial LoRA term before
the all-reduce, which sums the partial terms to the full update. Replicated
projections keep the full factors.

### LoRA performance

TensorSharp against stable-diffusion.cpp `19bbbca` on 2026-09-25. Both builds used
unchanged ggml `353b63b`. Every run was a 1024×1024 text-to-image with the teapot
prompt, seed 42, CFG 1 and Euler, each engine in a fresh process with a cooldown
between runs. Both engines received the same F32 sigma vector. Each configuration
ran twice, once with each engine going first, and the tables show the medians.
Seconds per step is the steady-state step, excluding the first. PSNR compares the
two engines' images. Without a step-distillation LoRA, 6 or 8 steps give a blurry
image by design (the model's default is 40 steps), so the "No LoRA" and Film Stills
(a style LoRA) rows measure speed, not quality.

Apple M5 Pro, 48 GB, `ggml_metal`:

| Configuration | Steps | s/step TensorSharp | s/step sd.cpp | Denoise speedup | Wall time TensorSharp / sd.cpp | Wall speedup | PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| No LoRA | 6 | 7.52 | 8.59 | 1.17× | 51.8 / 61.7 s | 1.19× | 63.1 dB |
| Viggle Turbo r128 | 6 | 7.94 | 9.40 | 1.22× | 54.5 / 67.0 s | 1.23× | 60.4 dB |
| Viggle Turbo r256 | 6 | 8.03 | 9.52 | 1.23× | 55.8 / 68.7 s | 1.23× | 57.5 dB |
| Pruna 8-step | 8 | 7.91 | 9.39 | 1.21× | 69.7 / 85.5 s | 1.23× | 53.4 dB |
| Film Stills, strength 0.7 | 8 | 7.92 | 9.43 | 1.21× | 69.8 / 85.8 s | 1.23× | 63.9 dB |

NVIDIA RTX 4000 Ada, 20 GB, driver 580, `ggml_cuda`:

| Configuration | Steps | s/step TensorSharp | s/step sd.cpp | Denoise speedup | Wall time TensorSharp / sd.cpp | Wall speedup | PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| No LoRA | 6 | 1.57 | 1.85 | 1.16× | 20.0 / 26.5 s | 1.33× | 46.2 dB |
| Viggle Turbo r128 | 6 | 1.94 | 2.53 | 1.28× | 23.3 / 32.6 s | 1.40× | 35.2 dB |
| Viggle Turbo r256 | 6 | 2.00 | 2.55 | 1.25× | 24.5 / 32.7 s | 1.33× | 36.8 dB |
| Pruna 8-step | 8 | 1.94 | 2.53 | 1.29× | 26.6 / 37.6 s | 1.41× | 35.3 dB |
| Film Stills, strength 0.7 | 8 | 1.92 | 2.51 | 1.30× | 26.0 / 37.3 s | 1.43× | 47.5 dB |

- **Cost per step.** LoRA adds 5–7% per step on Metal (sd.cpp: 9–11%) and 22–27% on
  CUDA (sd.cpp: 36–38%). The cost barely depends on the rank: rank 64, 128 and 256 are
  within 3% of each other. It comes from the extra pass over each adapted
  projection's output: the low-rank product writes an F32 tensor the size of that
  output, and the add reads it back. A faster GPU finishes the base step sooner, so
  the pass is a larger share there. ggml has no matrix product that accumulates
  into its destination, so this pass cannot be folded into the base projection
  without changing ggml. On CUDA, forcing cuBLAS to F16 compute
  (`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16`) made steps 5% slower, so the F32 products are
  not what limits it.
- **Loading.** The factors are read, scaled and packed in parallel at startup. This
  takes 0.1–0.8 s on the M5 Pro and 0.2–1.3 s on the CUDA machine; the output is
  bit-identical to a serial load.
- **VAE decoding.** The whole-VAE graph is now the default on Metal as well as CUDA.
  It decodes 1024×1024 in 5.0 s on the M5 Pro, where the per-convolution path took
  13.0 s and sd.cpp takes 7.1 s. At 2048×2048 it takes 24.9 s instead of 85.3 s, with
  a 45 GB peak memory footprint instead of 64 GB. It never runs on Vulkan, where
  F16 cooperative-matrix operands overflow in the decoder; setting
  `TS_QWEN21_VAE_FUSED=1` there prints a warning and decodes per convolution.
  `TS_QWEN21_VAE_FUSED=0` selects the per-convolution path on any backend.
- **Why the CUDA images differ more.** On the CUDA machine, sd.cpp keeps the
  transformer on the GPU and runs out of memory for a whole-image decode, then
  retries with 256×256 tiles (11.5–13.2 s). Its wall times include that retry.
  TensorSharp frees the transformer's weights first and decodes in 4.0–4.1 s. Both
  engines also run F32 matrix products as TF32 on CUDA. For both reasons, the two
  engines' images agree less closely on CUDA than on Metal. The images were
  checked visually and match.

The runs used [`eng/validation/qwen-image21-bench.py`](../../eng/validation/qwen-image21-bench.py)
with `--lora`, `--lora-config` and `--sigma-nodes`/`--sigma-shift` for the recipe
schedules. sd.cpp was given Pruna at multiplier 2 (`--sd-lora-multiplier 2`)
because it ignores the alpha stored in `lora_adapter_metadata`.

### Server and C# API

The server loads its `--lora` set at startup and applies it to every generation and
edit request; per-request LoRA selection is not implemented. A request's `steps`
and `cfg` still override a plug-in's recipe. In process,
`QwenImageModel.SetLoras(IReadOnlyList<LoraSpec>)` replaces the set for later
requests (an empty list removes it). The new set is validated against the
transformer immediately, and a failure leaves the previous set in place.

### Limitations

- The plug-ins apply to Qwen-Image-2.1 only; the CLI refuses `--lora` with any
  other model, and the server logs a warning and loads the other model without
  them.
- Only one plug-in per run can carry a sampling recipe, and a recipe with sigmas
  runs only the step counts it defines.
- The Qwen-Image-2.1-Fix author's workflow also uses APG, FreSca and the `seeds_2`
  sampler at CFG 3, which TensorSharp does not implement; the DoRA itself is
  applied exactly.
- The Pruna adapters were trained at 1K. Fun-Acc was trained at 2048×2048, and its
  card notes that small dense text and some edits are weaker than the 40-step
  teacher.

## Prefix KV cache

Qwen-Image-2.1 modulates the text and reference-image tokens with the `t = 0`
row, and its block-causal attention never lets them attend to the image being
generated (the checkpoint's `causal_condition`). Their hidden states, and so every
block's keys and values, are the same at every denoising step. TensorSharp
implements the official
[prefix KV cache](https://github.com/QwenLM/Qwen-Image-2.1#prefix-kv-cache): the
first step runs the whole sequence and stores each block's post-RoPE keys and
values for the prefix on the device. Every later step runs only the target
image's tokens and attends over the stored prefix followed by the target. A CFG
run keeps one cache per branch. The caches are released when denoising ends,
before VAE decoding. Each step's log line ends with `prefix=extract` or
`prefix=cached` (`prefix=declined` when the cache did not fit; see below).

The cache is on by default; `TS_QWEN21_PREFIX_CACHE=0` turns it off. By default
it stores exactly what the attention kernel reads: F16 for Metal and CUDA flash
attention, F32 otherwise. Cached steps therefore reproduce the uncached
computation. On Metal, one seed gives a byte-identical PNG with the cache on, with
it off, and from the build before the cache existed. This was checked for
generation, one- and two-reference editing, and CFG 4 with two caches. The
diffusers `use_kv_cache` documentation notes that its PyTorch implementation does
not reproduce images bit for bit across the two settings; TensorSharp's graphs do.

The cost is memory: 512 KiB per prefix token per CFG branch in F16 (32 blocks, K
and V, 4096 values of 2 bytes). A prompt is tens to a few hundred tokens, and
each reference at about 1 megapixel adds 4,096 tokens, about 2 GiB. A cache may
use at most half of the memory the device reports free, and
`TS_QWEN21_PREFIX_CACHE_MAX_MIB` caps it further. A cache that does not fit is
declined with a warning on stderr, and that request recomputes the prefix every
step, as before.

`TS_QWEN21_PREFIX_CACHE_TYPE` selects the storage type:

| Value | K / V storage | Bytes per prefix token and branch | Output |
|---|---|---:|---|
| `auto` (default) | What attention reads (F16 on Metal/CUDA flash, F32 otherwise) | 512 KiB (F16) | Identical to uncached |
| `f16`, `f32` | As named | 512 KiB / 1 MiB | Identical where attention reads that type |
| `q8_0` | Q8_0 / Q8_0 | 272 KiB | Rounds the stored prefix |
| `q8_0_v` | Attention type / Q8_0 | 392 KiB | Rounds the stored prefix values |

The 8-bit settings correspond to vLLM-Omni's `fp8` and `fp8_v` prefix caches.
They use ggml Q8_0 blocks, which have one scale per 32 values, instead of FP8
E4M3 with one scale per token and head. The stored prefix is converted back to
the attention type every step, so the target's own keys and values stay exact.
Their measured effect on output quality is below.

### Measured effect

Each row compares the build before the cache with this build, in fresh processes
with the Q4_K_M files, seed 42 and CFG 1 unless stated. Step time is the mean of
steps 2 onward, because step 1 stores the prefix and costs the same as an uncached
step. "Identical" means the output PNGs have the same SHA-256. Wall time is the
whole process, including model load, encoders and VAE. References are about
1 megapixel, or 4,096 prefix tokens each.

Apple M5 Pro, 48 GiB, `ggml_metal`:

| Workload | Uncached step | Cached step | Per step | Wall time | Output |
|---|---:|---:|---:|---|---|
| Generate 1024², 40 steps (2 rounds) | 7.91 s | 7.84 s | 1.01× | 330.5 → 327.6 s | Identical |
| Edit, 1 reference, 1024², 40 steps (2 runs) | 17.97–18.00 s | 9.65–10.38 s | 1.73–1.86× | 747.7 → 422.4 s; 744.9 → 452.5 s | Identical |
| Edit, 2 references, 1024², 40 steps | 33.32 s | 11.46 s | 2.91× | 1,369.7 → 512.3 s | Identical |
| Edit, 1 reference, CFG 4, 1024², 10 steps | 36.30 s | 19.29 s | 1.88× | 390.2 → 238.0 s | Identical |
| Edit, 1 reference, 2048², 10 steps | 73.41 s | 64.84 s | 1.13× | 813.3 → 739.0 s | Identical |

The two single-reference edit runs differ because the second ran with other
desktop applications using the GPU. Peak process RSS was unchanged within
run-to-run variation (19.1 → 18.9 GiB for the first edit pair).

NVIDIA A40, 46 GB, `ggml_cuda` on one GPU:

| Workload | Uncached step | Cached step | Per step | Wall time | Output |
|---|---:|---:|---:|---|---|
| Generate 1024², 40 steps (2 rounds) | 1.151 s | 1.113 s | 1.03× | 57.6 → 54.5 s | Identical |
| Edit, 1 reference, 1024², 40 steps (2 rounds) | 2.547 s | 1.328 s | 1.92× | 123.8 → 74.8 s | Identical |
| Edit, 2 references, 1024², 40 steps | 4.106 s | 1.538 s | 2.67× | 188.4 → 87.5 s | Identical |
| Edit, 1 reference, CFG 4, 1024², 20 steps | 5.074 s | 2.652 s | 1.91× | 122.7 → 75.8 s | Identical |
| Edit, 1 reference, 2048², 20 steps | 8.929 s | 7.608 s | 1.17× | 205.7 → 181.3 s | Identical |
| Generate 2048², 20 steps | 6.973 s | 6.821 s | 1.02× | 155.2 → 152.2 s | Identical |

NVIDIA A40, `ggml_vulkan` on one GPU. The build before this change decoded on
the CPU under Vulkan (see "Vulkan VAE" below), so its wall times are not
comparable. The cache comparison therefore uses this build with
`TS_QWEN21_PREFIX_CACHE=0` as the uncached side:

| Workload | Uncached step | Cached step | Per step | Output |
|---|---:|---:|---:|---|
| Edit, 1 reference, 1024², 20 steps (2 rounds) | 4.93 s | 2.67 s | 1.85× | Identical |
| Generate 1024², 20 steps | 2.07 s | 2.03 s | 1.02× | Identical |
| Edit, 1 reference, 512², 20 steps | 0.91 s | 0.48 s | 1.90× | Identical |

At 512², the build before this change took 683.5 s end to end, 557 s of it in
the CPU VAE decode; this build took 48.7 s. Its PNG differs from the previous
build's by 54.4 dB PSNR, from the device VAE's F16 rounding.

Sampled peak GPU memory for the 1024² single-reference edit rose from 6.2 to
8.2 GB, which is the 2.0 GiB cache. At 2048², where the target dominates, both
peaked at 16.3 GB. The cache helps in proportion to the prefix's share of the
sequence: generation, whose prefix is only the prompt, gains 1–3%; editing gains
the most at 1024², and less at 2048² where the target is four times larger.

8-bit storage, same 1024² single-reference edit:

| Storage | Prefix cache size | Metal step | Metal PSNR | CUDA step | CUDA PSNR |
|---|---:|---:|---:|---:|---:|
| `auto` (F16) | 2,066 MiB | 10.38 s | Identical | 1.328 s | Identical |
| `q8_0` | 1,098 MiB | 10.95 s | 58.9 dB | 1.499 s | 53.0 dB |
| `q8_0_v` | 1,582 MiB | 10.90 s | 60.5 dB | 1.414 s | 52.8 dB |

PSNR compares the output PNG with the default setting's (RGBA, 0–255). The 8-bit
types save memory but are 5–13% slower per step, because the stored prefix is
dequantized every step; CUDA dequantizes Q8_0 through F32. Use them only when a
cache would otherwise be declined.

## CUDA graphs and tensor parallelism

**CUDA graphs.** Upstream ggml-cuda captures a graph as a CUDA graph once two
consecutive executions leave it unchanged, then replays it. The cached step graph
is retained across steps (`TS_QWEN21_GRAPH_REUSE=1`, the default) with fixed
input and cache buffers. From a request's third step onward, each denoising step
is therefore one graph replay. This is the effect of vLLM-Omni's CUDA-graph
decode, without separate capture code. As in vLLM-Omni, the first step, which
stores the prefix, runs uncaptured. Under tensor parallelism, each rank's
segments between reductions are captured separately; vLLM-Omni disables graphs
under TP.

On an A40, counting CUDA runtime calls confirmed this: a 10-step request made 1
capture and 8 graph launches (20 steps: 18 launches), and a two-GPU request made
130 captures (65 segments on each GPU) and 1,040 launches. Output PNGs were
identical with graphs on and with `GGML_CUDA_DISABLE_GRAPHS=1`. The speed benefit
is small because each step's kernels are large: 1.5% per step at 256×256 on one
GPU (0.0662 s against 0.0672 s), 2.6% on two, and nothing measurable at 1024²
(1.112 s both ways).

**Tensor parallelism.** With `--tp N` on `ggml_cuda` or `ggml_vulkan`, the
diffusion transformer is sharded Megatron-style over N GPUs, following
vLLM-Omni's layout for 2.1:

- Each GPU holds 32/N whole attention heads: its Q, K and V rows and the matching
  `to_out` input columns.
- Each GPU holds 12,288/N MLP columns: gate and up rows and the matching
  `img_mlp.out` input columns.
- The input, time, modulation and output projections and all norms are
  replicated.
- The two row-parallel products in each block are summed across GPUs. ggml-cuda's
  collective (NCCL or P2P) does this on the devices when available; otherwise it
  goes through host memory.
- Each GPU caches the prefix of its own heads, so the cache is split N ways.
- N must divide the 32 attention heads — 2, 4, 8 or 16 on one machine, given
  ggml's 16-device limit — and every weight's quantized blocks must stay whole when
  it is sharded, which is checked per weight type at load. Only 2 GPUs have been
  measured.
- The text encoder, vision encoder and VAE stay on the first GPU. Multi-node
  groups are refused.

Measured on two NVIDIA A40s in different CPU sockets (46 GB each). Peer access is
not functional between them, so NCCL used its shared-memory transport. The prefix
cache was on unless stated; step times exclude step 1:

| Workload | 1 GPU step | 2 GPUs step | Speedup | Wall time | PSNR vs 1 GPU |
|---|---:|---:|---:|---|---:|
| Generate 1024², 40 steps (2 rounds) | 1.107 s | 0.823 s | 1.34× | 53.3 → 44.5 s | 41.5 dB |
| Edit, 1 reference, 1024², 40 steps (2 rounds) | 1.326 s | 0.934 s | 1.42× | 73.7 → 60.1 s | 44.9 dB |
| Generate 2048², 20 steps | 6.817 s | 4.436 s | 1.54× | 155.5 → 107.2 s | 34.2 dB |
| Edit, 1 reference, 2048², 20 steps | 7.612 s | 4.849 s | 1.57× | 181.5 → 128.9 s | 51.5 dB |
| `ggml_vulkan` edit, 1024², 20 steps (host reduction) | 2.670 s | 3.092 s | 0.86× | 144.8 → 156.1 s | 53.9 dB |

The sharded prefix cache compounds with TP: the two-GPU 1024² edit took 1.836 s
per step with the cache off. On one GPU, peak memory for that edit was 8.2 GB. On
two GPUs, the first GPU peaked at 8.3 GB and the second at 4.6 GB; the first GPU
also runs the encoders and VAE. At 2048², the VAE decode keeps the first GPU's
peak at 16.3 GB. On Vulkan, the host round trip per reduction outweighs the split,
so `--tp` is slower there on this hardware.

TP changes the order in which each block's partial sums are added, so outputs are
not bit-identical to one GPU. That rounding difference compounds over the
denoising trajectory. Images keep the same composition and quality, but details
can differ: at 2048²/20 steps, the shop sign sat in a different place. The PSNR
column quantifies this. Forcing an exact F32 all-reduce
(`GGML_CUDA_AR_BF16_THRESHOLD=0`) produced byte-identical PNGs to the default,
so reduced-precision reduction is not the cause.

The sharded graphs were also checked against the unsharded graph on single-GPU
machines, with a loopback group of backend instances on one device reducing
through host memory. Across 146 synthetic sharded forwards, the maximum
normalized error was 9.4e-5 on Metal and 1.7e-7 on CPU. With the real Q4_K_M
weights on Metal, relative L2 error was 0.07–0.10% of one prediction, with and
without the prefix cache. The native test runs the same 146 forwards on a real
two-GPU group for CUDA (4.5e-5, NCCL) and Vulkan (9.3e-5, host reduction).
vLLM-Omni lists TP for 2.1 as unverified and publishes no 2.1 scaling numbers.

**Vulkan VAE.** ggml-vulkan multiplies F32 matrices through F16
cooperative-matrix operands on GPUs that have them, and the VAE feeds some
convolutions activations above 65,504; on the NVIDIA A40 used for testing, one
decoder shortcut convolution saw inputs up to about 288,000. The VAE therefore
used to run its convolutions on the CPU under `ggml_vulkan`, and a 512×512
decode took over 8 minutes. The native F32 convolution now scales each Vulkan
input by an exact power of two until its magnitude is at most 32,768, then
scales the F32 result back before the bias. The VAE runs on the Vulkan device:
a 256×256 decode took 3.3 s and matched the F32 CPU reference to 0.14% relative
L2 (encode: 0.22%). The remaining difference is the F16 rounding of Vulkan's
matrix operands.

**FP8 weights.** vLLM-Omni's FP8 option quantizes a BF16 checkpoint's in-block
linear layers to FP8 W8A8 at load time. Its recipe reports this as a memory
saving, not a speedup, on GB200. TensorSharp loads block-quantized GGUF weights
(Q4_K_M or Q8_0). ggml has no FP8 E4M3 tensor type, and its Metal and CUDA
backends ignore activation-precision hints, so there is no FP8 GEMM to select.
The 8-bit weight configuration is the Q8_0 GGUF. On NVIDIA Turing and newer,
ggml-cuda's quantized matrix kernels also quantize the activations to 8 bits, so
this runs as int8 W8A8 on tensor cores. The part of vLLM-Omni's FP8 work that
applies here is 8-bit prefix storage, described above.

## Current Unsloth Q8_0 validation

The metadata-free Unsloth Q8_0 diffusion model and the three companion files in
the direct launch command above passed generation and editing on Apple M5 Pro,
48 GiB unified memory, macOS 27.0. Both engines used Metal, Euler, CFG 1, seed 42,
matching F32 sigma vectors and Philox noise, in serial fresh processes.

| Workload | TensorSharp wall time | stable-diffusion.cpp wall time |
|---|---:|---:|
| 1024×1024 generation, 40 steps | 337.654 s | 383.080 s |
| 512×512 color-change edit, 25 steps | 101.664 s | 114.186 s |

TensorSharp used 11.9% and 11.0% less wall time respectively in these single-run
comparisons. The generation images were visually close (48.38 dB RGB PSNR after
white compositing); both editing outputs changed the teapot to blue. This does
not establish broad quality or performance superiority. TensorSharp's 1K VAE
decode remained slower (13.319 s versus 7.270 s), and peak process RSS was higher
(16.50 GiB versus 13.29 GiB). File-cache and thermal state were uncontrolled.
The whole-VAE graph has since become the Metal default, and the 1K decode now
takes 5.0 s; see [LoRA performance](#lora-performance).

The exact files passed 17/17 real-server HTTP cases, including previews,
multi-reference editing, cancellation and recovery. The managed suite passed
265 tests with two explicit skips for other unavailable model fixtures; the
native suite passed all five CTests, with zero skips. A 2048×2048 single-step
execution check completed in 136.092 s with 34.83 GiB peak process RSS and
48.45 GiB macOS peak memory footprint. It is not a full-quality 2K measurement.
Neither memory metric is dedicated GPU allocation.

Both benchmark builds used unchanged ggml
`179b60f27b1019d42da01ac532cabdb8f73ba8b7`; stable-diffusion.cpp was
`c92d73c408515c94beef32161bb5960764fde7a0`. Commands, model/binary hashes,
images, numerical checks and limitations are recorded in ignored
`docs/validation/qwen-image21-unsloth/REPORT.md`. CUDA and Vulkan were not tested.

## Earlier Q4_K_M full-model performance

On this Apple M5 Pro/48 GiB machine, the Q4_K_M model generated the same bookstore
prompt at 1024×1024, 40 steps and seed 42 in **353.120 seconds** (353.734 seconds
process wall time), compared with the historical **751.156 seconds** below.
This is a **2.13× inference speedup for this workload**. Denoising took 334.695
seconds, VAE decoding 18.117 seconds, and peak process RSS was 16.69 GiB.

Both runs used the same model files, resolution, prompt and step count on Metal,
but the current run uses the recommended CFG 1 and corrected Qwen scheduler;
the historical run used CFG 6 and the Flux-derived schedule. This measures the
combined change in defaults and implementation, not isolated kernel speed or
equivalent output pixels. Each result is one fresh process with warm OS file
caches; thermal state was uncontrolled. Visual inspection of the new PNG found
correct “TENSORSHARP” lettering, detailed shelves, warm lighting and wet-pavement
reflections. One image is not a general quality score.

That run's commands, model hashes, dependency revisions, phase timings,
memory measurement and RGBA PNG are in ignored
`docs/validation/qwen-image-2.1/performance/quality-1024-40/`.

A real **2048×2048, 25-step, CFG 1** run with the same prompt and seed also
completed, producing a native-resolution RGBA PNG. It took **1548.719 seconds**
for inference (**1551.161 seconds / 25m 51s process wall time**): 1438.921 seconds
denoising and 109.483 seconds decoding. Peak process RSS was **32.30 GiB**;
macOS also reported a 57.33 GiB peak memory footprint. Neither measure is a
dedicated GPU-allocation measurement. This run demonstrates that native 2K
works here, while also showing its substantial time and memory cost.

Visual inspection at full resolution found correct main-sign lettering,
detailed brickwork and window frames, warm interior lighting and wet-pavement
reflections. Small interior details remain synthesized; this single prompt does
not establish general quality superiority over the 1K/40-step image. The PNG,
log and benchmark manifest are in
`docs/validation/qwen-image-2.1/performance/quality-2048-25/`.
The default 2K/40-step run, full-resolution editing, and CUDA/Vulkan generation
were not run in this validation and are not counted as passing scenarios.

That earlier Release build and focused suite passed **169 tests, zero skipped**,
including real companion-file metadata, automatic/explicit output geometry,
reference geometry, official 1K/2K sigma golden vectors, CPU VAE primitives,
RGBA handling, request parsing, Web UI service and upload-confinement regressions.
A legacy Qwen-Image DiT GPU-forward test was outside this focused run. Evidence
is `docs/validation/qwen-image-2.1/performance/final-managed.trx`; native
operator coverage is detailed below.

## Native optimization validation

CUDA and Metal retain up to two complete DiT graphs for the positive/negative
CFG layouts. `TS_QWEN21_GRAPH_REUSE=1` is the default on both backends; `0`
rebuilds the graph for each prediction. Reuse retains graph metadata and scratch
allocations while refreshing all dynamic inputs. Weight invalidation, cache
clearing and scratch release retire the graphs. `TS_QWEN21_GRAPH_TRACE=1` logs
builds and replay counts. Scratch is released before final VAE decoding.

The Metal VAE uses F32 MPS convolutions, with direct F32 ggml convolution for
unsupported vendor shapes. This preserves activations above the FP16 range:
upstream Metal matrix multiplication can narrow F32 operands internally even
when accumulation is F32. `TS_VAE_MPS_CONV=0` selects the direct convolution
baseline. TensorSharp releases its MPS graphs and staging buffers on explicit
scratch release and backend shutdown. Upstream ggml sources remain unchanged.

Metal reuse passed **159 native forwards** against the CPU explicit-attention
reference and **16 independent NumPy cases / 80 forwards**, including multiple
reference images and flash-attention tile boundaries. The maximum normalized
native error was 0.00054533; the maximum NumPy absolute error was 0.00140832,
within the existing 0.002 tile-boundary tolerance. These are synthetic numerical
checks, not downloaded-model quality or performance measurements. The unchanged
ggml revision for these checks was `179b60f27b1019d42da01ac532cabdb8f73ba8b7`;
evidence is in ignored `docs/validation/qwen21-native-metal-report.md`.

CPU, Metal and CUDA image attention now uses each segment's exact key/value
length without a dense padding mask; on CUDA, `TS_QWEN21_PAD_MASK=1` restores the
padded mask as a comparison diagnostic (see
[`docs/perf/qwen-image21-cuda.md`](../perf/qwen-image21-cuda.md)). Causal text
masks are retained. Metal casts the strided key/value tensors directly to F16, and
supported backends use upstream fused SwiGLU to avoid intermediate feed-forward
copies. These operations preserve the mathematical computation, with possible
floating-point rounding differences; `ggml_vulkan` still builds padded image
masks.
The earlier attention optimization measurements below used unchanged ggml at
`456172ec733a135778adcd32d00e576a58232e45`.

The independent NumPy transformer oracle passed **16 cases on CPU and 16 on
Metal**, each with five forwards. It covers generation, editing, multiple
references, fused/separate MLP weights, flash/explicit attention, changed latent
inputs, shape shrink/restore, and attention tile boundaries. Maximum absolute
error was 0.000005282 on CPU and 0.001409 on Metal. The larger boundary fixture
uses a 0.002 Metal tolerance because the unchanged baseline already has 0.001333
error from its half-precision matmuls; the existing small-case tolerances remain
0.001 on Metal and 0.0001 on CPU.

On Apple M5 Pro/Metal, a synthetic two-layer F32 transformer with 16,384 target
tokens, 64 prefix tokens, hidden size 256 and two 128-wide attention heads had
median warm forward latency **194.34 ms before → 167.81 ms after** (13.65% lower).
Each version ran in a separate process with five measurements after the first
forward. This geometry eliminates a 520 MiB image-mask allocation per prediction;
that is the calculated allocation size, not a measured reduction in peak RSS.
These timings exclude the full model, conditioning, scheduler and VAE and do not
establish full-resolution generation speed or visual quality. CUDA and Vulkan
were unavailable and are not counted as passing validation.

Run the oracle and optional synthetic benchmark with:

```bash
python3 eng/tests/qwen-image21-dit.py --backend cpu
python3 eng/tests/qwen-image21-dit.py --backend metal
python3 eng/tests/qwen-image21-dit.py --backend metal --benchmark-target-tokens 16384
```

The benchmark mode reports timings separately and does not count as an oracle
test. Logs, the before/after measurements and dependency metadata are under
ignored `docs/validation/qwen-image-2.1/performance-native-20260920/`.

The 2.1 VAE also routes spatial attention through native matrix operations,
tiling queries while every query still attends to every key. Each score tile is
bounded to 16 MiB, supporting the VAE's 768/1152-channel attention heads without
requiring a flash-attention kernel for those widths. The managed implementation
remains the fallback; `TS_QWEN21_VAE_ATTN=0` selects it for comparison.

Large VAE CPU normalization passes now visit contiguous spatial tiles, and SiLU
uses parallel ranges. Twelve scalar-oracle tests pass with bit-exact outputs,
including real channel counts, tile/chunk boundaries and extreme inputs. Resident
DiT weights are released before final VAE decoding to reduce memory pressure at
2K. These changes also retain the small-array path for previews.

[`eng/tests/qwen-image21-vae-attention.py`](../../eng/tests/qwen-image21-vae-attention.py)
passed **16 numerical cases and seven invalid-argument cases on each of CPU and
Metal**, including actual head widths, partial query tiles, uniform attention,
large logits, input/shape reuse and output guard regions. Maximum absolute error
was 0.000006323 on CPU and 0.002377 on Metal; maximum relative L2 error was
0.000000705 and 0.000891 respectively. Metal uses an explicit 0.003 absolute and
0.002 relative-L2 tolerance for upstream half-precision matrix operands with F32
accumulation. These are operator checks, not a VAE image-quality or speed
benchmark. Evidence is under ignored
`docs/validation/qwen-image-2.1/vae-attention-{cpu,metal}.{json,log}`.

## Historical validation and comparison

The results below predate the current 2K/CFG 1 defaults, official scheduler
correction and native optimizations. Image-generation measurements used CFG 6
and the earlier Flux-derived schedule. They describe those historical outputs
and workloads, not current-default performance or quality.

The earlier focused suite passed **317 tests**, with **one explicit skip** for a
legacy Qwen-Image DiT full-weight test because its older checkpoint was unavailable.
The skipped scenario is not counted as validation. Coverage includes request and
configuration handling, sampling/layout math, companion compatibility checks
against the downloaded files, RGBA round trips, and regressions in the shared
text/vision code. Evidence: ignored
`docs/validation/qwen-image-2.1/final-focused.trx`.

The real HTTP harness passed **16/16 cases** at 128×128: JSON generation, JSON and
multipart edits, two-reference editing, SSE progress/previews, RGBA uploads and
outputs, repeated-seed determinism, request refusals, and cancellation followed
by a successful generation. `http/report.json` records the results. These short
runs cover endpoint execution, not visual quality or full-resolution performance.

Run the 2.1 request/math/media regressions and downloaded companion metadata checks:

```bash
TENSORSHARP_QWEN21_DIT="$TENSORSHARP_MODELS/qwen-image-2.1/qwen_image_2.1_Q4_K_M.gguf" \
  dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj \
  --filter 'FullyQualifiedName~QwenImage21|FullyQualifiedName~QwenImageAlphaTests|FullyQualifiedName~QwenImageRequestTests|FullyQualifiedName~WebUiChatServiceTests|FullyQualifiedName~UploadRootConfinementTests'
```

The real-companion metadata test reports a skip if its model environment variable
is absent. The tests above do not replace the image-generation benchmarks.

The independent NumPy/native transformer oracle passed **12 cases on CPU and
12 on Metal**, covering text-only, one-reference and multiple-reference layouts,
fused/unfused MLP projections, and flash-attention on/off. This checks small
synthetic graphs, not full-model CPU inference. See
[`eng/tests/qwen-image21-dit.py`](../../eng/tests/qwen-image21-dit.py).

Matched generation and editing runs on an Apple M5 Pro with 48 GiB unified
memory used the configuration's Q4_K_M diffusion/text weights, 512×512 pixels,
20 Euler steps, CFG 6 and seed 42. Both engines ran on Metal. Each row is a
single fresh process; OS file caches were warm and shader compilation was
excluded. These are observations for these workloads.

| Task | Engine | Inference | Process wall time | Process peak RSS |
|---|---|---:|---:|---:|
| Text-to-image | TensorSharp | 84.714 s | 85.22 s | 6.17 GiB |
| Text-to-image | stable-diffusion.cpp | 113.43 s | 113.72 s | 9.81 GiB |
| Image edit | TensorSharp | 175.645 s | 176.00 s | 7.53 GiB |
| Image edit | stable-diffusion.cpp | 212.80 s | 213.06 s | 10.71 GiB |

TensorSharp inference was 1.34× as fast for generation and 1.21× as fast for
editing in these runs. The generated teapot images were visually close;
comparing RGB after compositing over white gave MAE 0.2606 and RMSE 0.5838 on
the 0–255 scale.

The editing run used the same reference file, `sd-t2i-512.png`, in both engines
and this instruction: “Change the red teapot to cobalt blue. Keep its shape,
lighting, the wooden table and background unchanged.” Both outputs showed a
blue teapot with the scene preserved. Raw RGBA comparison gave MAE 0.620672
and RMSE 1.12041; white-composited RGB gave MAE 0.825444 and RMSE 1.292416,
on the 0–255 scale. TensorSharp spent 1.664 s encoding the reference VAE,
6.599 s encoding text/vision, 162.564 s denoising and 4.817 s decoding.

Pixel agreement is not a general prompt-adherence score. Logs, PNGs and comparison
images are under ignored `docs/validation/qwen-image-2.1/` with the stems
`ts-t2i-512`, `sd-t2i-512`, `ts-edit-512` and `sd-edit-512`. These 512-pixel runs
do not establish quality or speed at the official 2048-pixel example, across
arbitrary prompts, or on other devices.

A separate TensorSharp quality run completed at **1024×1024, 40 Euler steps,
CFG 6, seed 42**, producing an RGBA PNG. Its prompt described a rainy-evening
bookstore with warm windows and a sign reading “TENSORSHARP.” Visual inspection
confirmed the main sign's lettering, detailed shelves and architecture, warm
lighting and reflections on wet pavement. Inference took **751.156 s** (751.94 s
process wall time) with **17.864 GiB** peak RSS; text encoding took 0.443 s,
denoising 715.220 s and VAE decoding 35.493 s. The output and log are
`docs/validation/qwen-image-2.1/ts-t2i-1024-40.png` and
`docs/validation/qwen-image-2.1/ts-t2i-1024-40.log`. This is one prompt/run;
no matched 1024×1024 stable-diffusion.cpp benchmark was performed.

Dependency revisions for this local comparison:

- TensorSharp's unchanged ggml: `456172ec733a135778adcd32d00e576a58232e45`.
- stable-diffusion.cpp: `c678dfe704a2230342376b46add9c8ca736a653d`.
- llama.cpp source studied: `ce8caa6e60a03093351d6016a818720e0d46f0fb`.
- ComfyUI-GGUF source studied: `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`.

The stable-diffusion.cpp reference used its supported `SD_USE_UPSTREAM_GGML=ON`
configuration with TensorSharp's unchanged ggml checkout. Its default local ggml
submodule did not match the current sd.cpp sources and failed to build; no
reference-tree files were changed. Upstream mode disables tensorwise INT8 and
convrot paths. The benchmark uses Q4_K_M/BF16 weights, so it does not compare
every available stable-diffusion.cpp build or optimization.

ComfyUI-GGUF's loading code was inspected; full ComfyUI was unavailable and no
ComfyUI end-to-end benchmark was run. llama.cpp supplied conditioning/GGUF
reference behavior, not a standalone image-generation benchmark.

For reproducible performance comparisons, record hardware, backend and dependency
revisions; model hashes; output dimensions; reference images; seed; sampler;
step count; CFG; cache settings; and whether weight loading and first-run shader
compilation are included. Compare both cold end-to-end latency and repeated
inference with the same loaded process. Identical seeds across engines do not
guarantee identical initial noise; compare images as well as timing.

## Reproduce comparisons with the current implementation

The reusable [benchmark runner](../../eng/validation/qwen-image21-bench.py)
launches both engines serially with matched request settings and records commands,
model hashes, revisions, phase timings, peak RSS and image diagnostics. The
command below uses the current sampling settings; reproducing the historical
images above requires their earlier implementation and schedule. Older sd.cpp
revisions use the Flux-derived schedule, so matching command-line settings alone
does not establish matching sigma schedules or image quality:

```bash
python3 eng/validation/qwen-image21-bench.py \
  --models-dir "$TENSORSHARP_MODELS/qwen-image-2.1" \
  --width 1024 --height 1024 --steps 40 --cfg 1 --seed 42 \
  --prompt 'A red ceramic teapot on a wooden table, soft daylight, product photograph.'
```

Add `--mode edit --image reference.png` and an editing prompt for an edit,
`--engine tensorsharp` to run only TensorSharp without a reference binary,
`--repeat 3` for repeated fresh-process measurements, or `--dry-run` to
inspect commands. The default reference binary is
`artifacts/qwen-image-2.1/sd-build/bin/sd-cli`; override `--sd-cli` when needed.
Image diagnostics use NumPy and Pillow.

The reference implementations are
[stable-diffusion.cpp's 2.1 guide](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md),
[llama.cpp](https://github.com/ggml-org/llama.cpp) for Qwen3-VL/GGUF, and
[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) for quantized tensor loading.
The model card's ComfyUI workflows also provide generation and editing recipes.
Generated logs, benchmark records and output images belong under ignored
`docs/validation/` or `artifacts/`. A unit test, low-resolution smoke image or
unavailable device scenario is not evidence of full-resolution quality or speed
parity; no such parity claim follows from the commands above.
