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

Qwen-Image-2.1-Turbo, Qwen's 8-step distillation of the same model, uses these
companions with its own transformer and configuration; see
[Qwen-Image-2.1-Turbo](#qwen-image-21-turbo).

## Launch the CLI

Run these commands from the TensorSharp repository root. The configuration
selects `ggml_metal` for Apple Silicon. On an NVIDIA machine with the CUDA backend
built, append `--backend ggml_cuda`; the native CPU backend is `ggml_cpu`, and
`--backend cpu` runs the whole pipeline in pure C#
([Pure C# CPU backend](#pure-c-cpu-backend---backend-cpu)).
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
same pixel area with the first reference's aspect ratio for editing. On the
pure-C# `cpu` backend the automatic area is 1 MP instead (1024×1024), because a
2048×2048 step takes about 5x as long there; `ggml_cpu` and the GPU backends keep
2048×2048. Set width and height together, in multiples of 32, to override this. The model supports
[native 2K aspect ratios](https://github.com/QwenLM/Qwen-Image-2.1#supported-aspect-ratios).
Reference images are conditioned at approximately 1 megapixel each, or the
output area if smaller; increasing the output to 2K does not also quadruple each
reference's VAE, vision-encoder and transformer workload.

An edit sized from an area returns a picture larger than that area smaller, and
one smaller than it larger. To edit a picture at its own size — and so edit an
edit without it shrinking — pass `--keep-source-size` (`keepSourceSize: true` in
an API request): the output is the first image's exact width and height. It is
sampled at about that image's own area -- at least 1 megapixel, where references
are conditioned, and at most the area the edit would otherwise use -- at about its
aspect ratio (the 32-pixel grid rounds both), then resized to the source when that
differs from it. It cannot be combined with an explicit width and
height, and needs an input image. A masked edit (below) keeps the source size
without it. The server Web UI and TensorAgent send it with every edit. The initial
noise is drawn for the size the edit samples at ([seeds and edits](#seeds-and-edits)).

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

### Seeds and edits

`--diffusion-seed` (`seed` in the API; default 0) is the key of the Philox
generator that draws the initial noise. Text-to-image noise depends on the seed
and the output size alone, as in stable-diffusion.cpp with `--rng cuda`, so the
same request gives the same picture.

An edit's noise also depends on its reference images. TensorSharp hashes the
pictures as they were passed (SHA-256 over their 8-bit RGB values in order, plus
alpha when any pixel is not opaque) and draws the noise on a separate Philox
stream chosen by that hash; the pipeline's `Qwen-Image-2.1: …` log line shows it
as `(edit noise stream <id>)`. Without this, editing a picture at the seed and size it
was drawn with starts from the very noise that became that picture, and
Qwen-Image-2.1 can then retrace the picture instead of following the instruction.
Every host's defaults lead there: the seed is 0 unless a request names one, and an
edit samples at its source's size — an automatic size follows the source's aspect
at the same area, and `keepSourceSize` (sent by the Web UI and TensorAgent) samples
a 1024×1024 picture at 1024×1024. A fox drawn at seed 0 (1024×1024, CFG 1) and asked at
seed 0 to "change the background to a sandy beach with the ocean behind the fox"
came back over-sharpened and still in the snow, at 12 and 40 steps and with a
speed LoRA; a fox drawn and edited at seed 5 did the same, while the same fox
edited at another seed got the beach. That retrace is likely rather than certain:
a teapot recolored at CFG 6 from its own seed's noise did change
([historical comparison](#historical-validation-and-comparison)).

- The same pictures, prompt, seed and settings give the same edit, and another
  seed gives another composition, as before. Rewording the prompt keeps the noise.
- Qwen-Image-2.1-Turbo draws its edits the same way: the stream depends on the
  pictures, not on the checkpoint.
- The noise is drawn for the size the edit samples at. With `keepSourceSize` that
  is the sampling size [above](#launch-the-cli), not the source's own size when the
  two differ; the result is resized to the source afterwards.
- Editing a result again at the same seed starts from new noise, because its
  picture changed.
- A masked edit takes the stream of the whole source picture, not of the selection
  or its crop, so every selection on one picture shares that picture's stream.
- The stream follows the decoded pixels, and a picture kept in memory hashes like
  its saved PNG. Every PNG TensorSharp writes, and any PNG without colour
  information, decodes to the same bytes on every host. The TensorAgent apps convert
  a PNG to sRGB when it embeds a colour profile, as macOS and iPhone screenshots do,
  and when it has no alpha channel and its `gAMA` or `cHRM` chunk describes another
  space; the CLI and server keep the stored values
  (`eng/validation/apple-png-decode-check.py` lists which). Such a PNG, like a JPEG
  or HEIC photo, which can decode a few levels differently in the apps, gets another
  stream in the apps than on the CLI and server, so another composition. Compare
  edits of such pictures across hosts with `TS_QWEN21_EDIT_NOISE=seed`.
- `TS_QWEN21_EDIT_NOISE=seed` draws edits from the seed's text-to-image noise, as
  stable-diffusion.cpp and diffusers do: use it for matched-noise comparisons with
  those engines and to reproduce edits made before this change. Each such edit
  prints `TS_QWEN21_EDIT_NOISE=seed: this edit starts from the seed's text-to-image
  noise …`. `references` is the default; any other value fails the request.
- Hashing is the only cost. On an M5 Pro with CoreCLR it took 2.6–3.9 ms for a
  1-megapixel reference and 29–44 ms for a 12-megapixel photo in a Release build
  (the high end when a translucent alpha plane is hashed too), and 16–26 ms and
  177–298 ms in a Debug build, over two runs of
  `eng/validation/QwenImage21EditNoiseCost` (its `-p:ModelsDir` measures a
  TensorSharp.Models built elsewhere, such as the CLI's). Text-to-image hashes
  nothing. The Apple apps run on Mono, where a per-pixel loop is slower still; that
  was not measured.

#### Measured effect

Apple M5 Pro (48 GiB, macOS 27.0), `ggml_metal`, unchanged ggml
`ffa4e8b80930029a35991f94e7c8a93cd67730ab`. The Q4_K_M checkpoint at 12 steps and
Turbo AD-Q4_K at its 8, both at 1024×1024 and CFG 1, one fresh CLI process per
picture. "Before" is the same build without reference-keyed edit noise.

- Text-to-image did not change: the teapot prompt below at seeds 0 and 1, and "A
  red fox sitting in deep snow in a winter forest, photograph." drawn by Turbo at
  seed 0, gave PNGs with the same SHA-256 before and after.
- The retrace is gone. The fox drawn at seed 0 and edited at seed 0 to "change the
  background to a sandy beach with the ocean behind the fox" came back before as an
  over-sharpened copy, still in the snowy forest; now the fox sits on a sandy beach
  with the sea behind it. With `--keep-source-size` in place of `--width 1024
  --height 1024` the edit samples at 1024×1024 as well and gave the same PNG, and so
  did the server's `/api/image-edit` given the picture as a multipart upload with
  `keepSourceSize` and no `seed` or size, as the Web UI sends it; both printed the
  same noise stream. With `TS_QWEN21_EDIT_NOISE=seed` the edit's PNG had the same
  SHA-256 as before, and so did a masked edit (the seed-0 recolor below).
- Turbo behaves the same. Its seed-0 fox, edited at seed 0 with the beach
  instruction and `--keep-source-size`, came back before as an over-sharpened copy,
  the forest still behind the fox and the snow turned to cracked earth; now the fox
  sits on a beach with waves behind it. Seed mode again gave the PNG from before.
- A chained edit follows its instruction: each beach picture, edited again at seed 0
  to "put a red knitted scarf around the fox's neck", got the scarf and kept the
  beach, on both checkpoints.
- Local edits keep more of the scene. "A red ceramic teapot on a wooden table, soft
  daylight, product photograph." was drawn at seeds 0, 1 and 2, and each picture was
  recolored at its own seed with "Change the red teapot to cobalt blue. Keep its
  shape, lighting, the wooden table and background unchanged.", once without and
  once with a selection around the teapot. Both noises turned the teapot blue (the
  same share of blue pixels within 0.003). From the source's own noise the teapot
  came out a gritty navy and the table and background over-sharpened; from the
  reference-keyed noise they stayed close to the source. PSNR against the source,
  leaving out the teapot and 16 pixels around it
  (`eng/validation/qwen-image21-edit-fidelity.py`):

  | Recolor | Seed 0 | Seed 1 | Seed 2 |
  |---|---:|---:|---:|
  | Whole picture, reference-keyed noise | 25.4 dB | 29.5 dB | 30.9 dB |
  | Whole picture, `TS_QWEN21_EDIT_NOISE=seed` | 20.2 dB | 17.0 dB | 16.9 dB |
  | Inside the selection, reference-keyed noise | 29.8 dB | 26.4 dB | 29.0 dB |
  | Inside the selection, `TS_QWEN21_EDIT_NOISE=seed` | 21.8 dB | 19.3 dB | 19.4 dB |

  Pixels outside the selection were exact in every masked run.
- Wall time was the same within run-to-run noise: 135–139 s per edit in both modes
  and before, 98 s per Turbo edit.

Images, hashes and the analysis are in ignored `artifacts/editnoise4/` and
`artifacts/editnoise5/` (local validation evidence, not committed). Not measured:
CUDA, Vulkan and `--tp`, the `cpu` backend, the TensorAgent apps (they pass the same
pictures to the same pipeline, but run on Mono; their PNG decoding was checked by a
Swift copy of it on macOS, not on iOS), masked and multi-reference edits on Turbo,
and JPEG or HEIC sources.

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
`width`, `height`, `steps`, `cfg` and `seed` (default 0; an edit's noise also
depends on its references, see [Seeds and edits](#seeds-and-edits)). `targetArea`
controls automatic geometry; explicit dimensions take precedence. Omitting `width`, `height`,
`targetArea`, `steps` and `cfg` selects the model defaults above. `targetArea: 1048576`
selects approximately 1K output while retaining automatic aspect-ratio selection.
The edit endpoints also accept `keepSourceSize` (JSON `true`, or `-F 'keepSourceSize=true'`
in multipart): the result keeps the first image's exact size, sampled within
`targetArea` (or the default area) as described above. Sent with `width`/`height`,
or to `/api/image-generate`, it is rejected with a 400.

Starting the server with `--width` and `--height` changes that default size. The
host publishes them as `TS_QWEN_IMAGE_WIDTH` / `TS_QWEN_IMAGE_HEIGHT`, and every
image request that sets neither `width`/`height` nor an explicit `targetArea` then
uses that size, including Web UI requests, which send no size; an edit then no
longer follows the first reference's aspect ratio, except one with `keepSourceSize`
(every Web UI edit), which spends only that size's area, at the source's aspect
ratio, and keeps the source's size. A request that sets its own
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

## Qwen-Image-2.1-Turbo

[Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) is Qwen's
accelerated checkpoint of the same 7B transformer, distilled to **8 Euler steps at
CFG 1** on one fixed schedule. Its VAE, Qwen3-VL-8B text encoder and vision projector
are the 2.1 files above, byte for byte. TensorSharp runs the
[AtomicChat GGUFs](https://huggingface.co/AtomicChat/Qwen-Image-2.1-Turbo-GGUF)
(revision `bb25d06`), which carry no metadata:

| File | Size | LPIPS against BF16 (the card's) | Use |
|---|---:|---:|---|
| `Qwen-Image-2.1-Turbo-AD-Q4_K.gguf` | 4,201,694,944 bytes | 0.147 | The fast default of [`config/qwen-image-2.1-turbo.json`](../../config/qwen-image-2.1-turbo.json) |
| `Qwen-Image-2.1-Turbo-Q8_0.gguf` | 7,591,554,784 bytes | 0.037 | The transformer closest to full precision |

The card's other files (AD-Q6_K, AD-Q5_K, AD-Q3_K, AD-Q2_K, BF16) have the same tensor
names; only the two above were run here. The card also measures the text encoder: a
Q4_K_M encoder moves the pictures about as far from a BF16 encoder (LPIPS about 0.17) as
AD-Q4_K moves them from the BF16 transformer, and a Q8_0 encoder by 0.037. The
configuration keeps the Q4_K_M encoder of the 2.1 layout above, under its published file
name, so one folder holding both models downloads it once.

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1-turbo.json \
  --prompt 'a neon sign that reads "OPEN LATE", rainy night' \
  --width 1024 --height 1024 --diffusion-seed 42 --output turbo.png
dotnet run --project TensorSharp.Server.Host -c Release --no-build -- \
  --config config/qwen-image-2.1-turbo.json --host 127.0.0.1 --port 5000
```

### Declaring the checkpoint

Turbo's tensors have the base checkpoint's names and shapes, and the GGUFs have no
metadata, so nothing in the file tells the two apart. The host declares it:

- `--qwen-image-variant turbo` (or `base`) on the CLI and the server, or
  `"qwen-image-variant": "turbo"` in a config file; the Turbo config sets it. The hosts
  hand it to the model as `TS_QWEN_IMAGE_VARIANT`, which an in-process caller can set
  before constructing `QwenImageModel` (its `Variant` property reports the result). An
  unknown flag value is a configuration error at startup (exit code 1), and an unknown
  `TS_QWEN_IMAGE_VARIANT` refuses the load (exit code 2).
- TensorAgent's Turbo catalog entries declare it themselves.

The load prints `variant = turbo (declared)`. Without a declaration it reads the file
name: a name containing the word `turbo` is assumed to be Turbo, the way Wan's
step-distilled checkpoints are recognised, and the load prints
`variant = turbo, ASSUMED from the word "turbo" in the file name: ...`. That is a guess.
Merges of the Viggle step-distillation LoRA into the base checkpoint are published under
the same names (Abiray/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-GGUF ships
`qwen_image_2.1_turbo_Q4_K_M.gguf`, exactly as Abiray/Qwen-Image-2.1-Turbo-GGUF does), and
such a merge needs Viggle's schedule, not Turbo's; declare `base` for it. Any other name
is the base checkpoint. The server applies its declaration to every Qwen-Image model it
loads, like the companion paths. The declaration means nothing to another model: the CLI
refuses `--qwen-image-variant` with one (exit code 1), and the server loads it without the
declaration and logs a warning, as it does for `--lora`.

### Sampling

Turbo samples `sample_sigmas` from its
[`model_index.json`](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/model_index.json),
followed by a final 0:

```text
[1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0]
```

Its scheduler has shift 1.0 and no dynamic shifting, so the values are used as they are
at every resolution: that is how diffusers loads them, and what the AtomicChat card passes
to stable-diffusion.cpp as `--sigmas`. The run logs it before denoising:

```text
Qwen-Image-2.1 Turbo: 1024x1024, 8 steps, CFG 1, seed 42, 0 reference(s)
  [qwen21] Turbo schedule (model_index.json sample_sigmas): 8 steps on fixed sigmas [1, 0.97845, 0.95418, 0.92663, 0.89508, 0.84515, 0.70453, 0.41457, 0]
```

- The default is 8 steps at CFG 1. An explicit `--diffusion-steps 8` (`steps: 8` in a
  request) is accepted, and any other count is refused with a message that names the
  supported one. Qwen's card says that setting the step count alone does not override the
  saved schedule and that other schedules have not been evaluated, so TensorSharp does not
  resample the eight values into another count.
- An explicit `--cfg` above 1 (`cfg` in a request) still adds the negative pass, as on the
  base checkpoint; Turbo was distilled for CFG 1.
- The timestep reaches the transformer in F32, as for the base checkpoint and in
  stable-diffusion.cpp.
- Editing, `--keep-source-size` and the prefix KV cache work as on the base checkpoint
  (validated below, and by TensorAgent's planner test on Turbo); masks and several
  references take the same code path but were not run on Turbo. An edit's noise
  follows its pictures here too ([seeds and edits](#seeds-and-edits)).

### LoRA plug-ins on Turbo

Turbo is step-distilled already. A plug-in that brings a sampling recipe (Viggle Turbo,
Pruna 8-step and 5-step, Fun-Acc's PDD bundle, or any config with `"sampling"`) is refused
before its tensors are read: `LoRA plug-in refused: LoRA '...' carries a sampling recipe
(...): it is a step-distillation plug-in, and this transformer is Qwen-Image-2.1-Turbo
(declared), which is step-distilled already and samples on its own 8-step schedule.`
The CLI and the server refuse the load (exit code 2), and the message says how Turbo was
identified. A plug-in without a recipe loads. A bare LoRA `.safetensors`
passed without its plug-in config carries no recipe either, so the engine cannot tell a
speed adapter from a style one; pass the `config/lora/` plug-in.

Validated on Turbo with real pictures: Film Stills and Grainscape through their
`config/lora/` plug-ins at their strength 0.7, each against the same prompt and seed
without it, at seeds 42 and 7 (1024×1024, AD-Q4_K; `--suite style` of
[`qwen-image21-turbo.py`](../../eng/validation/qwen-image21-turbo.py) reproduces them, and
its seed-42 pictures were byte-identical to an earlier run's). Both plug-ins apply their
look, and both redraw part of the scene, Film Stills more than Grainscape:

- Film Stills turned a sharp night scene at a tram stop into a softer, grainier, warmer
  film still. At seed 42 it also dropped the tram and the bench, and the woman sits on an
  indistinct dark shape; at seed 7 she stays on the shelter's bench, but the street behind
  her is redrawn, with a tree and shopfronts where the tracks were.
- Grainscape turned a clear mountain lake at dawn into a hazy, muted film look. The lake,
  the boat and the mountains stay in about the same places; the rock faces and the skyline
  are redrawn, and at seed 42 the boat gained a pair of oars.
- Redrawing the scene is the plug-in's, not Turbo's: on the base checkpoint at 40 steps
  (Q4_K_M, seed 42), Film Stills replaced the same tram-stop scene entirely, with no tram,
  a lower viewpoint and a tree-lined street.
- Each adds about 5.4% per step (8.28–8.31 s against 7.86–7.88 s).

TensorAgent offers these two for its Turbo entries and no other plug-in. The
editing plug-ins and Quality Fix have not been tried on Turbo, and Object Remover works
only at the base model's 40 steps.

### Parity with stable-diffusion.cpp

TensorSharp against stable-diffusion.cpp `f89d9b1` (its master on 2026-10-09, which
includes the `c150a6b` custom-sigma fix; built with Metal and its own ggml submodule
`d25b121`), on 2026-10-09. TensorSharp used unchanged upstream ggml `ffa4e8b`. Apple M5
Pro, 48 GB, `ggml_metal`. Both engines got the same transformer GGUF, the Q4_K_M text
encoder, the BF16 VAE and the F16 mmproj; the same prompt, seed 42 and Philox noise
(`--rng cuda` on sd.cpp); Euler at CFG 1; and the eight published sigmas (sd.cpp through
`--sigmas`). The edit's reference was the Turbo card's yacht sketch, composited over white
and resized beforehand to its 1728×608 conditioning size, so neither engine resized it.
PSNR compares the two engines' PNGs, and every pair was also compared by eye. Seconds per
step is the steady state (step 2 onwards). The edit ran before TensorSharp keyed edit
noise to the references, so both engines started it from the seed's noise; repeating it
takes `TS_QWEN21_EDIT_NOISE=seed`, which the bench sets when both engines run.

| Prompt | Transformer | Size | PSNR | s/step TensorSharp / sd.cpp | Wall time TensorSharp / sd.cpp |
|---|---|---|---:|---:|---:|
| The card's `a neon sign that reads "OPEN LATE", rainy night` | AD-Q4_K | 1024×1024 | 57.4 dB | 7.79 / 8.76 | 70.3 / 92.8 s |
| A tea-house sign with the brush-written name 清风茶社 (prompt in Chinese) | AD-Q4_K | 1024×1024 | 44.7 dB | 7.78 / 8.77 | 69.4 / 79.9 s |
| A close-up portrait photograph of an elderly fisherman | AD-Q4_K | 1024×1024 | 59.6 dB | 7.79 / 8.76 | 69.5 / 79.4 s |
| Edit: the card's yacht sketch into a photograph (its full prompt) | AD-Q4_K | 1728×608 | 39.3 dB | 7.97 / 11.16 | 87.2 / 114.3 s |
| The neon sign | Q8_0 | 1024×1024 | 59.5 dB | 7.62 / 8.63 | 69.0 / 79.1 s |

- Each pair is the same picture. The neon sign reads OPEN in red and LATE in blue; the
  tea-house sign has the four characters, both engines writing 风 in its traditional form
  風; the portrait matches down to the skin texture; and the edited yacht keeps the
  sketch's mast, wheelhouse, four midships windows, two lifeboats, yellow funnel and the
  rows of five and nine portholes.
- The remaining differences are rounding: the two engines order their sums differently.
  The edit differs most, because the reference also goes through both engines' vision and
  VAE encoders. sd.cpp's first run (the neon sign) includes reading its files from a cold
  page cache.
- The 8-bit and 4-bit transformers draw the same scene with different details: the two
  TensorSharp neon signs are 19.1 dB apart, in line with the card's 20.8 dB between
  AD-Q4_K and BF16.
- TensorSharp's output is deterministic on Metal: repeated runs gave byte-identical PNGs.

The card's sample grid uses `--rng cpu` and a BF16 text encoder, so it cannot be matched
pixel for pixel here; the comparison above uses sd.cpp itself as the reference.

### Performance

Apple M5 Pro, 48 GB, `ggml_metal`, 1024×1024, the neon-sign prompt, seed 42, the Q4_K_M
text encoder. Each configuration ran in two fresh processes, in A-B-C-D-D-C-B-A order with
20 s between runs, and the table shows the medians (the two runs of each were within 0.4%).
Seconds per step is the steady state; the first step also stores the prefix KV cache and
takes about 0.2 s longer. The VAE decode took 5.1–5.3 s and the text encoder 0.3 s in every
configuration.

| Configuration | Steps | s/step | Denoise | Wall time | Against 40 steps |
|---|---:|---:|---:|---:|---:|
| Qwen-Image-2.1 Q4_K_M | 40 | 7.83 | 313.3 s | 319.5 s | 1.0× |
| Qwen-Image-2.1 Q4_K_M + Viggle Turbo r128 | 6 | 8.30 | 50.0 s | 56.3 s | 5.7× |
| Turbo AD-Q4_K | 8 | 7.86 | 63.1 s | 69.3 s | 4.6× |
| Turbo Q8_0 | 8 | 7.62 | 61.2 s | 67.6 s | 4.7× |

- A Turbo step costs what a base step costs: the transformer is the same, and AD-Q4_K's
  mix (Q5_K attention in the first and last four blocks and Q4_K elsewhere, Q8_0 modulation,
  BF16 time embedding, output head and text input) runs within 0.4% of Q4_K_M's (Q6_K
  attention values). The BF16 tensors work on two rows (the time embedding) or the prompt's
  tokens (`txt_in`), not the 4,096 image tokens, so their type does not show.
- Q8_0 is 3% faster per step than AD-Q4_K: at 4,096 image tokens the products are
  compute-bound, and a Q8_0 block is cheaper to unpack in the matrix kernels than a K-quant
  block. The higher-quality file is also the faster one; it costs 3.4 GB more weights.
- Viggle Turbo's 6 steps finish first: each of its steps costs 6% more (the unmerged
  adapter), but there are two fewer. Turbo is Qwen's own distillation and takes no adapter.
- The peak memory footprint of an edit at 1248×832 (`/usr/bin/time -l`) was 14.35–14.37 GB
  with every one of the three transformers: the transformer is mapped from its file, and the
  peak comes from the stages around it. TensorAgent's tiers add the weights to it.

Reproduce these with [`eng/validation/qwen-image21-turbo.py`](../../eng/validation/qwen-image21-turbo.py)
(`--suite parity,perf,footprint`), which drives
[`qwen-image21-bench.py`](../../eng/validation/qwen-image21-bench.py) `--variant turbo`.

### Turbo limitations

- Only 8 steps run on Turbo. The card's measurements, and these, are at about 1 megapixel;
  Qwen recommends about 2 megapixels (the 2048×2048 default).
- The variant is a declaration. An undeclared file named `...turbo...` is assumed to be
  Turbo, which is wrong for a step-distillation merge published under the same name.
- Turbo's other quantizations were not run here.

## Precise local editing with a mask

Supply a mask at the first input image's exact dimensions to edit a selected
region. **White edits; black preserves** by default. Gray values blend the edit
with the source. Transparent masks are also supported with `maskMode: "alpha"`
or `--mask-mode alpha`: transparent pixels edit, opaque pixels preserve. The
source image's transparency remains separate from the selection mask.

The output retains the first input's original dimensions and its decoded RGB and
alpha values at every unselected pixel. The pipeline constrains protected latents during
denoising and composites the generated region onto the original source at the
end. Additional input images remain references. An empty selection returns the
source without running diffusion. A full selection edits the entire canvas.

```bash
dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json \
  --image photo.png --mask selection.png --mask-mode grayscale \
  --prompt 'Change the selected vase to red ceramic' \
  --mask-feather 8 --mask-crop --mask-crop-padding 96 \
  --width 1024 --height 1024 --diffusion-seed 42 --output edited.png
```

`--mask-feather` softens edges inward in source pixels (0–1024, default 0), so
unselected pixels stay protected. `--mask-invert` reverses the selection.
`--mask-crop` processes the selected region with surrounding context, then places
it back on the original canvas. `--mask-crop-padding` sets that context in source
pixels (0–16384, default 64). Cropping is optional; small regions can use a smaller
internal `--width`/`--height` to reduce inference work. These set the nominal
full-canvas sampling resolution; crop mode scales it to the selected crop's share
of the canvas and rounds up to the 32-pixel grid. The saved image still has the
source dimensions. The quality and speed tradeoff depends on the selection,
context, internal resolution and sampling steps.

Multipart API:

```bash
curl --fail-with-body http://127.0.0.1:5000/api/image-edit \
  -F 'image=@photo.png' -F 'mask=@selection.png' \
  -F 'maskMode=grayscale' -F 'maskFeather=8' \
  -F 'maskCrop=true' -F 'maskCropPadding=96' \
  -F 'prompt=Change the selected vase to red ceramic' \
  -F 'width=1024' -F 'height=1024' -F 'seed=42'
```

For JSON and SSE, first upload both files through `/api/upload`, then send their
returned server filenames:

```json
{
  "imagePaths": ["uploaded-photo.png"],
  "maskPath": "uploaded-selection.png",
  "maskMode": "grayscale",
  "maskInvert": false,
  "maskFeather": 8,
  "maskCrop": true,
  "maskCropPadding": 96,
  "prompt": "Change the selected vase to red ceramic",
  "width": 1024,
  "height": 1024,
  "seed": 42
}
```

The same fields reach `/api/image-edit` and `/api/image-edit/stream`, including
TensorAgent's shared image-edit service. Mask references must stay within the
upload directory. A mask requires an input image and Qwen-Image-2.1; mismatched
dimensions, unsupported mask modes, malformed numeric fields and multiple
multipart masks are rejected before inference. Mask options without a mask are
also rejected. Streaming reports request errors in its terminal `{done,error}`
frame, as for other image-edit failures.

In Server Chat and TensorAgent (desktop and mobile), attach one or more photos
and choose **Select area** on any photo. Paint or erase the region, zoom and pan
for details, then choose **Use selection** and describe the change. Saving a
selection makes that photo the **Editing target**, moves it to the first image
position, and keeps the other photos as references in their existing order. Each
photo retains its saved selection, but only the editing target's selection is
sent for the current edit. Canceling or a failed selection upload leaves the
previous target unchanged. Each turn produces one edited image.
Removing the editing target leaves the remaining photos' selections saved;
reopen and save one to activate it for another local edit.

Undo/redo and inversion operate on the selection. **Edit again** restores the
original photos, their saved selections, the editing target and prompt; the
comparison button switches between the target's original image and result.
TensorAgent also saves the selection with the conversation. Selection masks are
uploaded separately from reference images. Every decoded SSE preview includes
the protected source pixels, just like the final image.

HEIC/HEIF uploads retain a small thumbnail and a separate full-resolution PNG
for painting and reopening selections. The original photo remains the model's
source. The browser editor has no fixed megapixel or per-side limit and exports
selections at the source dimensions. Available browser memory and canvas support
determine the practical maximum image size.

The mask is enforced by TensorSharp's sampling and compositing code; it does not
add an annotation image or a dedicated mask channel to Qwen's conditioning.
Precise preservation outside the selection does not guarantee that the model
will follow every instruction inside it. Crop mode can help isolate one object,
but also removes surrounding context. Keep it disabled when that context matters.

Reusable validation tools:

- `eng/validation/QwenImageMaskBench`: synthetic preparation, latent reinjection
  and compositing timings, allocation checks and scalar parity, without weights.
- `eng/validation/qwen-image21-mask-bench.py`: real CLI or HTTP/SSE runs with
  pixel preservation checks, crop comparisons and explicit measurement limits.
- `eng/validation/validate-image-mask-editor.py`: desktop and touch-emulated
  browser interaction, exported mask geometry and exact undo/redo regression checks.
- `eng/validation/validate-image-mask-live.py`: real Server Chat upload, selection,
  inference, result and reuse, plus multipart/error handling checks.
- `eng/validation/tensoragent-mask-bench.py`: real TensorAgent host chat workflow.

Run each tool with `--help` (the C# benchmark documents its arguments in
`Program.cs`). Reports, logs and screenshots belong in ignored `docs/validation/`
or `artifacts/`. Browser touch emulation does not establish native phone behavior;
CPU mask microbenchmarks do not measure full model inference or GPU speedups.

On the GGML backends the 2.1 diffusion transformer runs a complete GGML graph with
resident quantized weights; on `cpu` it runs a managed forward over the same
file-mapped weights. There is no weight-streaming mode on either. Start with smaller
dimensions if available memory is insufficient. CUDA and Vulkan were exercised on
NVIDIA A40s; the measurements below record where.

## Pure C# CPU backend (`--backend cpu`)

`--backend cpu` runs the whole pipeline in managed C#: the diffusion transformer,
the Qwen3-VL text encoder, the vision encoder used for editing, the VAE, LoRA
plug-ins and the prefix KV cache. No GGML graph is built and the pipeline makes no
call into the native GgmlOps library; the CLI also skips its GGML teardown at exit on
this backend unless something else in the process loaded the library. Weights are
read from the memory-mapped GGUF and safetensors files in their stored types.
Previously the model refused `cpu` and needed a GGML backend. Outside the model
compute two native pieces remain, as before: image files are read and written
through Magick.NET on desktop, and the server probes the GGML and CUDA backends
at startup.

```bash
TENSORSHARP_MODELS="$PWD/models" dotnet run --project TensorSharp.Cli -c Release --no-build -- \
  --config config/qwen-image-2.1.json --backend cpu \
  --lora config/lora/qwen-image-2.1-pruna-5step.json \
  --prompt 'A small orange cat beside a blue ceramic vase, soft daylight, detailed photograph' \
  --width 512 --height 512 --diffusion-seed 42 --output cat-cpu.png
```

The server takes the same `--backend cpu`. On this backend a request that names no
size renders at the 1 MP automatic area (1024×1024, or that area at the first
reference's aspect ratio for an edit) rather than 2048×2048, whose transformer
steps take about 5x as long; explicit sizes, an explicit `targetArea` and the
server's `--width` / `--height` are unaffected, and the run prints the size it
chose. `ggml_cpu` keeps the native 2048×2048 default.

What runs where:

- **Transformer** (`QwenImage21ManagedDiT`): the operations of the native graph,
  in the same order and from the same weight descriptors. The quantized
  projections quantize their activations to Q8_K / Q8_0, as ggml-cpu does, and run
  the managed multi-row integer GEMM; attention is a tiled flash attention; LoRA
  factors (stacked shrinks, DoRA row scales) are applied unmerged as on the GGML
  backends. `TS_QWEN21_CPU_MATMUL=f32`, the default before, multiplies F32
  activations by dequantized weight tiles instead: slower, but numerically steadier.
- **Prefix KV cache**: a managed cache in host memory with the same types
  (`auto`/`f32` store what attention reads, so cached steps reproduce uncached
  ones bit for bit; `f16`, `q8_0`, `q8_0_v`) and the same rule: at most half of the
  free physical memory, and `TS_QWEN21_PREFIX_CACHE_MAX_MIB`.
- **Text encoder**: the projections run the same multi-row integer GEMM (8-bit
  activations; `TS_QWEN_TE_CPU_MATMUL=f32` selects a packed F32 GEMM on
  dequantized Q4_K / Q6_K tiles), with a managed causal grouped-query attention.
- **Vision encoder** (editing): packed-GEMM linear layers and a managed multi-head
  attention. Each linear weight's F32 copy is released once it is packed, so the
  tower holds one copy of its weights: encoding a 512×512 reference peaked at
  3.2 GB of commit in the stage benchmark, against 5.4 GB with both copies, with
  bit-identical output.
- **VAE**: every convolution is an implicit-im2col packed SGEMM against weights
  packed once per layer, the decoder's 2x upsample is folded into the convolution
  that reads it, and it runs on a pool with one thread per logical CPU
  (`TS_CPU_GEMM_THREADS`; `Parallel.For` at that width under `TS_CPU_POOL=0`).

All of these have AVX-512 and AVX2 kernels, chosen by one instruction-set decision
for the whole backend (`TS_CPU_DISABLE_AVX512=1` selects the AVX2 ones), and a
portable fallback; the environment variable matrix lists
[every switch](../env_var_feature_matrix.md#out-of-matrix-qwen-image-21-knobs),
including the `0` / `scalar` settings that restore each previous stage.

### Measured on an 8-core laptop

i7-11800H (8 cores / 16 threads, AVX-512), 32 GB, Windows. DiT
`qwen_image_2.1_Q4_K_M.gguf`, text encoder `Qwen3VL-8B-Instruct-Q4_K_M.gguf`,
VAE BF16; text-to-image, CFG 1, seed 42, with the default integer (Q8) projections.
`ggml_cpu` runs native code these changes did not touch. PSNR / SSIM compare the
`cpu` image with the `ggml_cpu` one.

| Run | Stage | `cpu` | `ggml_cpu` |
|---|---|---:|---:|
| 256×256, 2 steps (two runs) | text and vision encode | 3.1 s | 4.1–4.2 s |
| | denoise, 2 steps | 11.0–11.1 s | 20.8–21.4 s |
| | VAE decode | 2.8–2.9 s | 10.8–11.2 s |
| | total | **16.9–17.1 s** | 36.1–36.4 s |
| | PSNR / SSIM | 37.7 dB / 0.984 | reference |
| 512×512, Pruna 5-step LoRA | steady step (prefix cached) | 17.6–21.7 s | 43.9–61.4 s |
| | total | **118 s** | 295.7 s |
| | PSNR / SSIM | 32.9 dB / 0.972 | reference |

With `TS_QWEN21_CPU_MATMUL=f32` (F32 activations against dequantized weight
tiles, the default before) the same 256×256 run took 20.4 s at 42.0 dB / 0.984,
and the 512×512 one 169.6 s (steps of 30.7–31.6 s) at 31.4 dB / 0.96. The text
encoder alone takes 0.76–0.88 s for the 37-token default prompt with the integer
projections, against 1.8–1.9 s with `TS_QWEN_TE_CPU_MATMUL=f32` and about 1.4 s on
`ggml_cpu`. Two larger runs were measured with the F32 transformer, before the
integer route became the default: a 1024×1024 Pruna 5-step generation took 756 s
on `cpu` against 1101 s on `ggml_cpu`, and a 512×512 edit 224 s against 353 s.

The two backends do not produce the same pixels, and neither is the reference:
the managed transformer quantizes the activations as ggml-cpu does but sums in
another order and keeps F32 where ggml-cpu uses its F16 GELU table and
BF16-rounded inputs (`TS_QWEN21_CPU_GELU_FP16` / `TS_QWEN21_CPU_ROUND_ACTIVATIONS`
reproduce those), and each re-quantization can flip a rounding. On single
forwards (`benchmarks/QwenImageDiTBench`, 256×256) the managed velocity has cosine
0.99993 to ggml-cpu at sigma 1 and 0.99938 at sigma 0.02 (0.99994 and 0.9978 with
the F32 route), while a 1e-4 relative change of the timestep alone moves
ggml-cpu's own velocity by 4.7e-2 (cosine 0.9989). Editing on `cpu` is also covered by
unit tests of the managed transformer (edit layouts, several references) and by
stage benchmarks (`benchmarks/QwenImageStagesBench`). In the stage benchmark a
1024×1024 reference image took 12–13 s through the vision encoder, and a 256×256
one 0.72 s against 5.0 s on `ggml_cpu`.

### Limitations on `cpu`

- **Tensor parallelism is GPU-only.** `--tp N` with `--backend cpu` is refused at
  load (exit code 2) with a message naming `ggml_cuda` / `ggml_vulkan`; the managed
  pipeline runs in one process.
- **Memory.** The DiT and text-encoder weights are file-mapped (4.2 GB for the
  Q4_K_M DiT, 5.0 GB for the Q4_K_M text encoder), but activations, the prefix
  cache and the VAE feature maps are ordinary process memory. The VAE releases
  each feature map after its last read: a 1024×1024 Pruna 5-step generation
  peaked at 3.7 GiB of commit (8.0 GiB of working set), and a 2048×2048 VAE encode
  plus decode at 10.5 GiB of commit. Before any work, a size whose estimated peak
  exceeds the machine's memory is refused with the largest square size that fits
  (`TS_QWEN_IMAGE_CPU_MEMORY_CHECK=0` overrides), and one that exceeds the memory
  free right now gets a warning.
- **Speed.** At 512×512 one step takes 17.6–21.7 s on the 8-core laptop above,
  and a 2K square has 16 times the image tokens of 512×512, with attention growing
  faster than that; that is why the automatic size on this backend is 1024×1024.
  Use a step-distillation LoRA and a small size for anything interactive.
- The GGML-only switches (`TS_QWEN21_GRAPH_REUSE`, `TS_QWEN21_FLASH`,
  `TS_QWEN21_PAD_MASK`, `TS_QWEN21_VAE_FUSED`, `TS_QWEN21_VISION_FUSED`) have no
  effect on `cpu`.

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
Two plug-ins that both carry a recipe cannot be stacked, and on
[Qwen-Image-2.1-Turbo](#lora-plug-ins-on-turbo), whose checkpoint has a schedule of its
own, a plug-in that carries one is refused. Style and editing
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
`ggml_vulkan`, `ggml_cpu` and the pure-C# `cpu`. The load logs the plug-in count
and the size of the packed factors (`Qwen-Image-2.1 LoRA: N plug-in(s), applied unmerged (... MiB of
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
edit request; its HTTP requests cannot choose plug-ins. A request's `steps`
and `cfg` still override a plug-in's recipe. In process,
`QwenImageModel.SetLoras(IReadOnlyList<LoraSpec>)` replaces the set for later
requests (an empty list removes it). The new set is validated against the
transformer immediately, and a failure leaves the previous set in place.

A host that chooses plug-ins per picture passes them to `WebUiChatService`'s
`ImageGenerateStreamAsync`, `ImageEditStreamAsync` or `ImageEditAsync(body, loras, ct)`.
The set is swapped in under the same lock as the run, so a picture that waited behind
another is made with the set it asked for. An unchanged set costs nothing; pass
absolute paths, which is how the model records the set. The TensorAgent Mac app works this
way: it offers the twelve plug-ins of [USAGE.md's table](../../USAGE.md#qwen-image-21-lora-plug-ins)
from its own pinned catalog and applies the user's choice to each picture, its edit routes
included (see [TensorAgent's README](../../TensorAgent/README.md)).

### Limitations

- The plug-ins apply to Qwen-Image-2.1 only; the CLI refuses `--lora` with any
  other model, and the server logs a warning and loads the other model without
  them.
- Only one plug-in per run can carry a sampling recipe, and a recipe with sigmas
  runs only the step counts it defines. Qwen-Image-2.1-Turbo takes none.
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

**Tensor parallelism.** With `--tp N` on `ggml_cuda` or `ggml_vulkan` (`cpu`
refuses it at load), the diffusion transformer is sharded Megatron-style over N
GPUs, following vLLM-Omni's layout for 2.1:

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
comparisons. The edit ran before TensorSharp keyed edit noise to the reference
images; matching stable-diffusion.cpp's edit noise now takes
`TS_QWEN21_EDIT_NOISE=seed` (the benchmark's `--edit-noise seed`, its default when
both engines run). The generation images were visually close (48.38 dB RGB PSNR after
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
blue teapot with the scene preserved. Both started from the noise that had drawn
`sd-t2i-512.png` (seed 42, 512×512), and the CFG 6 recolor still followed the
instruction; TensorSharp now draws an edit's noise from a stream keyed to its
references, so this comparison needs `TS_QWEN21_EDIT_NOISE=seed` to be repeated.
Raw RGBA comparison gave MAE 0.620672
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
guarantee identical initial noise; compare images as well as timing. TensorSharp's
edits draw their noise from a stream keyed to the reference images
([Seeds and edits](#seeds-and-edits)), so an edit matches stable-diffusion.cpp's
noise only with `TS_QWEN21_EDIT_NOISE=seed`.

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

Add `--mode edit --image reference.png` and an editing prompt for an edit
(`--edit-noise seed`, the default when both engines run, gives TensorSharp
stable-diffusion.cpp's seed-only edit noise; `--edit-noise references`, the
default with `--engine tensorsharp`, measures TensorSharp's own; the report
records which),
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
