# Configuration files

Both `TensorSharp.Cli` and the server, `TensorSharp.Server.Host`, can read their
startup options from a JSON file passed with `--config`, in addition to the
command line:

```bash
TensorSharp.Server.Host --config config/server-basic.json
TensorSharp.Cli         --config config/cli-basic.json
```

These are the executables in the release archives (`tensorsharp-server-*` and
`tensorsharp-cli-*`). From a source build, run
`dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll` and
`dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll` in their place; the examples
below use the short names.

**Command-line option values always win.** When the command line sets a single-valued
option itself, that option's entry in the file is dropped before it is resolved:
it is not emitted, its `${variables}` are not substituted and its download is never
attempted — so one config file can be reused across machines while you override
just what differs:

```bash
# Use the file, but force the CPU backend for this run:
TensorSharp.Server.Host --config config/server-basic.json --backend ggml_cpu
```

Each override prints one line to standard error:

```text
[config] --X is set on the command line, so the value from '<file>' is ignored and the command-line value is used instead[; its download entry is skipped, so nothing is fetched for it]
```

So passing `--model` or `--mmproj` yourself skips the file's own
[download entry](#auto-download) for that option, and `--mmproj none` runs a
multimodal config text-only without fetching its projector.

You can pass `--config` more than once; a later file's entry replaces an earlier
one's the same way (and the command line still wins over all of them).
`--skills-dir` follows this precedence for its full list of roots: command-line
roots replace all configured roots, or the last file that sets `skills-dir`
replaces earlier files' roots. Repeated command-line flags or an array within
the winning config file retain all their values in order. The other
repeatable options — `--stop`, `--skill`, `--lora`,
`--lora-scale`, `--lora-config`, `--image`, `--ref-image`, `--ref-video`,
`--ref-audio` and `--ref-video-audio` — keep every file's values and add the
command line's after them instead, and a download entry under one of them still
runs.

## File format

The file is a JSON object whose keys are the same long option names each host
already accepts (listed in `--help`), with or without the leading `--`. Comments
(`//`, `/* */`) and trailing commas are allowed.

| JSON value          | Becomes                          | Example |
|---------------------|----------------------------------|---------|
| string / number     | `--key value`                    | `"max-tokens": 4096` → `--max-tokens 4096` |
| `true`              | the bare switch `--key`          | `"continuous-batching": true` → `--continuous-batching` |
| `false` / `null`    | nothing (omit the option)        | to turn something off, use its negation key, e.g. `"no-continuous-batching": true` |
| array               | a repeated flag                  | `"stop": ["</s>", "<\|eot\|>"]` → `--stop </s> --stop <\|eot\|>` |
| object              | a downloadable file (see below)  | `{ "path": "...", "urls": ["..."] }` |

`"variables"` (see below), its alias `"vars"`, `"$schema"`, and `"$unavailable"` are
reserved and never become options. A key that names a removed option is refused with a message
saying what to use instead, and `lora-scale` / `lora-config` cannot be arrays,
because each binds to the one `--lora` before it.

`"$unavailable": "reason"` marks a preset whose required implementation is missing.
Both hosts stop with that reason before expanding options or downloading any
files from any supplied config. The reason must be a nonempty string. Command-line
overrides and later configs cannot bypass this marker; use a supported preset to
run inference. `"$schema"` remains an informational schema hint.

**The two hosts treat an unknown key differently.** The server refuses to start
on an option it does not recognise (suggesting a near match when there is one); the CLI
silently ignores it. A file meant for both hosts should use only keys both
accept. The differences that come up in practice are server-only options such as
`host`, `port`, `embeddings`, `embedding-threads` and `no-webui`. The
repeat-penalty window is `repeat-last-n` on both hosts; the CLI's former
`penalty-last-n` is a removed option, refused as a key like any other.

## Variables

Define shared values once under `"variables"` and reference them with `${name}`
in any string value. A `${name}` that is not defined there falls back to an
environment variable of the same name (an empty one counts as unset), and variables may reference other
variables. `${name:-fallback}` supplies a value for when the name is defined
nowhere. See [`variables.json`](variables.json).

```json
{
  "variables": { "modelRoot": "${TENSORSHARP_MODELS:-../models}" },
  "model": "${modelRoot}/gemma-4-E4B-it-Q8_0.gguf",
  "mmproj": "${modelRoot}/mmproj-gemma-4-E4B-it-Q8_0.gguf"
}
```

### Where the models go (and why no config here has a drive letter)

Every config in this folder resolves its model root the same way:

```json
"modelRoot": "${TENSORSHARP_MODELS:-../models}"
```

- Set **`TENSORSHARP_MODELS`** to keep your models wherever you like — a Windows
  path, `/mnt/data/models`, `~/models`. One variable covers every config here.
- Leave it unset and the models land in the `models/` folder at the repository
  root (gitignored): a relative `path` is resolved against the **config file**, not
  the working directory, so `../models` from `config/` is `<repo>/models`.

This is not a style preference. A hard-coded `"C:/models/x.gguf"` is **not an
absolute path on Linux or macOS** — .NET's `Path.IsPathRooted` knows nothing about
drive letters there — so it would be treated as *relative* and silently glued onto
the config's own directory. Written the way above, one file works unmodified on
Windows, Linux and macOS.

The [`agent-*.json`](#agent-configs) files are the one deliberate exception: their
fallbacks are absolute POSIX paths on a specific Mac, because they also pin a
`skills-dir` that has to exist or the host refuses to start. `TENSORSHARP_MODELS`
and `TENSORSHARP_SKILLS` still override both, which is how you move them.

You can declare **as many variables (and root paths) as you need** — models that
live in different folders each get their own root:

```json
{
  "variables": {
    "ditRoot": "${TENSORSHARP_MODELS:-../models}/qwen-image-2.1",
    "encoderRoot": "${TENSORSHARP_MODELS:-../models}/qwen3-vl"
  },
  "model": { "path": "${ditRoot}/qwen_image_2.1_Q4_K_M.gguf", "urls": ["..."] },
  "qwen-image-vae": "${ditRoot}/qwen_image_2.1_vae_bf16.safetensors",
  "qwen-image-vl": "${encoderRoot}/Qwen3VL-8B-Instruct-Q4_K_M.gguf"
}
```

## Auto-download

Any file option can be an object with a local `path` and one or more `urls`
instead of a plain string. On startup:

- if `path` already exists, it is used as-is (no network access);
- otherwise the file is downloaded from the first working URL, saved to `path`,
  and reused on every later run.

List several mirrors under `urls` for automatic fallback — if the first URL
fails, the next is tried. A relative `path` resolves next to the config file. An
optional `sha256` (lowercase hex) verifies a freshly downloaded file. Download
progress is printed to standard error so you can see what is being fetched and
how far along it is. See [`auto-download.json`](auto-download.json).

```json
{
  "model": {
    "path": "C:/models/Qwen3.5-9B-Q8_0.gguf",
    "urls": [
      "https://primary-mirror.example.com/Qwen3.5-9B-Q8_0.gguf",
      "https://backup-mirror.example.com/Qwen3.5-9B-Q8_0.gguf"
    ],
    "sha256": "0123...optional..."
  }
}
```

`url` (singular) is accepted as a shorthand for a single-entry `urls`.

Split GGUF models need every shard beside the first. Add the remaining shards
under the primary download object's `files` array, each with its own `path`,
`urls` and `sha256`, as in [`qwen3.8-flash-next.json`](qwen3.8-flash-next.json).
The hosts download every missing shard and emit just one `--model` argument
pointing to the primary `path`. Cached shards are reused independently. Paths
inside `files` are relative to the config file, just like the primary path;
`--model` on the command line skips the entire configured download group.
Use separate `mmproj` and `draft-model` options for optional companions so they
can be selected independently of the required shards.

## Examples in this folder

The runnable examples use **real, public, ungated** Hugging Face URLs, so the files
auto-download to their local `path` on the first run and are reused afterward.
Place the MiniMax-H3 tokenizer (`vocab.json`, `merges.txt` and
`tokenizer_config.json`) by hand; see [below](#video-generation-with-sound-minimax-h3).
If you already have a file at that `path`, it is used as-is (no download).
The experimental LTX 2.5 preset described below includes gated
official companions; it does not download or start inference.

| File | Model(s) | Shows |
|------|----------|-------|
| [`cli-basic.json`](cli-basic.json) | Qwen3.5-9B (~8.9 GB) | Minimal CLI config, auto-download |
| [`server-basic.json`](server-basic.json) | Gemma-4 E4B model + vision projector | Multimodal server, sampling defaults, auto-download |
| [`variables.json`](variables.json) | Gemma-4 26B-A4B: model + mmproj + MTP draft | One shared root/repo reused across three related files |
| [`auto-download.json`](auto-download.json) | Qwen3.5-9B (~8.9 GB) | Auto-download demo using a public GGUF |
| [`qwen-image-2.1.json`](qwen-image-2.1.json) | Qwen-Image-2.1 Q4_K_M + dedicated VAE + Qwen3-VL-8B + projector | Text-to-image and editing; pinned, checksum-verified downloads |
| [`qwen-image-2.1-turbo.json`](qwen-image-2.1-turbo.json) | Qwen-Image-2.1-Turbo AD-Q4_K + the same VAE, Qwen3-VL-8B and projector | Text-to-image and editing in 8 steps; declares `"qwen-image-variant": "turbo"`; the companions keep their publishers' file names, so a base-model config pointed at the same folder and names downloads them once for both |
| [`lora/qwen-image-2.1-*.json`](lora/) | Fourteen Qwen-Image-2.1 LoRA plug-ins (each downloads its weights; Fun-Acc also its `pdd_config.json`) | `--lora` plug-ins for Qwen-Image-2.1; Film Stills, Grainscape, Casual Phone Photo and Voxel Style also work with the official 8-step Turbo checkpoint; see [below](#qwen-image-21-lora-plug-ins-lora) |
| [`lora/minimax-h3-fl2v-turbo-*.json`](lora/) | Two MiniMax-H3 FL2VA LoRA plug-ins (each downloads its 1.38 GB of weights) | `--lora` step-distillation plug-ins for the FL2VA checkpoint: LightX2V Turbo, 8 or 4 steps on a locked recipe; see [below](#minimax-h3-turbo-plug-ins) |
| [`minimax-h3-fl2va.json`](minimax-h3-fl2va.json) | MiniMax-H3 FL2VA: DiT + Qwen3-VL-32B + video VAE + audio VAE (~35.5 GB) | **Video and 32 kHz stereo audio in one packed latent**; text-to-video, image-to-video, first/last frame |
| [`minimax-h3-ref2va.json`](minimax-h3-ref2va.json) | MiniMax-H3 Ref2VA: DiT + Qwen3-VL-32B + video VAE + audio VAE (~35.4 GB) | The same four networks, reference checkpoint: up to nine stills, clips and soundtracks |
| [`wan-video-ti2v-5b-turbo.json`](wan-video-ti2v-5b-turbo.json) | Wan 2.2 TI2V-5B Turbo: DiT + video VAE + UMT5 (~12.9 GB) | Video only, 4-step distilled, text- **and** image-to-video |
| [`wan-video-ti2v-5b.json`](wan-video-ti2v-5b.json) | Wan 2.2 TI2V-5B: DiT + video VAE + UMT5 (~12.9 GB) | Video only, undistilled 50-step reference recipe |
| [`wan-video-i2v-a14b.json`](wan-video-i2v-a14b.json) | Wan 2.2 I2V-A14B: **two** expert DiTs + Wan 2.1 VAE + UMT5 (~25.6 GB) | Image-to-video, two-expert schedule |
| [`embedding-snowflake.json`](embedding-snowflake.json) / [`embedding-minilm.json`](embedding-minilm.json) | Snowflake Arctic Embed L v2.0 / all-MiniLM-L6-v2 (Q8_0) | Embedding server; pinned, checksum-verified downloads |

`server-basic.json` uses the standard `gemma-4-E4B-it` build — point its `path` at
your own file to host a different variant.

The two embedding configs are server configs: `embeddings`, `embedding-threads`,
`host`, `port` and `no-webui` are server options. They default to native CPU
execution (`ggml_cpu`) and port 5000, with no Web UI:

```bash
TensorSharp.Server.Host --config config/embedding-snowflake.json --backend ggml_metal
```

See the [embedding guide](../docs/embeddings.md) for API examples and backend support.

## Ready-made configs, one per runnable model

One config per runnable model, with its companions (vision projector, MTP draft
head) already wired in. Every file names each model file's Hugging Face source, so
a missing file downloads on first run and an existing one is used as-is; every one
pins a commit and a SHA-256. Every key in every file is one the server accepts, and
a test enforces that. The CLI accepts all of them too except in
`jev-diffusiongemma-q4.json`, which is a server config (`host`, `port` and the
`/v1/systemone` endpoint); `qwen3.5-9b-uncensored-q8.json`'s `repeat-last-n` now
applies on both hosts:

```bash
TensorSharp.Cli         --config config/qwen3.5-9b-q8.json --input prompt.txt
TensorSharp.Server.Host --config config/gemma-4-26b-a4b.json
```

| File | Model | Kind |
|------|-------|------|
| [`qwen3.5-9b-q8.json`](qwen3.5-9b-q8.json) | Qwen3.5-9B (Q8_0) | Text LLM |
| [`qwen3.5-9b-iq4_xs.json`](qwen3.5-9b-iq4_xs.json) | Qwen3.5-9B (IQ4_XS) | Text LLM, smaller quant |
| [`qwen3.5-9b-uncensored-q8.json`](qwen3.5-9b-uncensored-q8.json) | Qwen3.5-9B Uncensored (Q8_0) | Text LLM |
| [`qwen3.6-27b.json`](qwen3.6-27b.json) | Qwen3.6-27B + vision | Multimodal LLM |
| [`qwen3.6-35b-a3b.json`](qwen3.6-35b-a3b.json) | Qwen3.6-35B-A3B (MoE) + vision | Multimodal MoE LLM |
| [`qwen3.8-flash-next.json`](qwen3.8-flash-next.json) | Qwen3.8 Flash Next (UD-Q2_K_XL, three shards) + vision | Multimodal MoE LLM; optional shared MTP head |
| [`gemma-4-e4b.json`](gemma-4-e4b.json) | Gemma-4 E4B **uncensored** (TrevorJS's community build, Q8_0) + stock unsloth vision projector + AtomicChat MTP draft | Multimodal LLM |
| [`gemma-4-12b.json`](gemma-4-12b.json) | Gemma-4 12B (QAT) + vision + MTP draft | Multimodal LLM |
| [`gemma-4-26b-a4b.json`](gemma-4-26b-a4b.json) | Gemma-4 26B-A4B (MoE) + vision + MTP draft | Multimodal MoE LLM |
| [`gpt-oss-20b.json`](gpt-oss-20b.json) | gpt-oss-20b (Q8_0) | Text reasoning LLM |
| [`jev-diffusiongemma-q4.json`](jev-diffusiongemma-q4.json) | DiffusionGemma 26B-A4B (Q4_K_M) + vision | [Jev typed decisions](../docs/models/jev.md) over text, images, documents, sampled video and configured ASR transcripts, native `/v1/systemone` server |
| [`diffusiongemma-26b-a4b-q4.json`](diffusiongemma-26b-a4b-q4.json) | DiffusionGemma 26B-A4B (Q4_K_M) | Text diffusion + image input (CLI/server); auto-downloads the vision shard |
| [`diffusiongemma-26b-a4b-q3.json`](diffusiongemma-26b-a4b-q3.json) | DiffusionGemma 26B-A4B (Q3_K_M) | Text diffusion + image input, smaller; shares the Q4 vision shard |

For the agent-enabled counterparts of three of these models, plus Qwen3.8-27B — skills,
code execution, network and package installs switched on — see
[Agent configs](#agent-configs) below.

Notes:

- **Multimodal** configs load a vision projector, so add `--image photo.png` to ask
  about a picture.
- **Qwen3.8 Flash Next** downloads about 78.9 GB of weights across three shards
  and a 0.9 GB BF16 projector. It defaults to `ggml_cuda` with F16 KV; pass
  `--backend ggml_metal` on Apple Silicon. Expert and PLE paging can keep the
  resident working set below the download size, with device- and SSD-dependent
  performance. Pass `--mmproj none` to skip vision. The commented `draft-model`
  entry enables the separate 2.8 GB shared Q8_0 MTP head on GGML backends;
  uncomment it to download and use that head. Speculation applies to solo
  requests prefilling from position zero; retained-prefix and concurrent
  requests decode plainly. See the [model guide](../docs/models/qwen38-flash-next.md)
  for supported execution paths and measured limits.
- **Speculative decoding** is lossless and works on **both** hosts — a key here
  becomes the matching flag, and `TensorSharp.Cli` honours every one of them.
  There is one spelling per option: `"spec"`, `"spec-type"`, `"spec-draft"`,
  `"spec-pmin"` and `"draft-model"` (the old `"mtp-*"` / `"spec-draft-model"`
  keys now fail with a message naming the replacement). Any drafter that ships
  as its own GGUF — Gemma's assistant head, a DFlash/DSpark block drafter — is
  named by `"draft-model"`, and naming it is the request: no `"spec": true`
  needed beside it. Qwen3.6, Qwen3.8-27B, GLM 5.2 and GLM-5.3 embed theirs in the
  trunk, so `"spec": true` is all they need. `"spec-type"`, `"spec-draft"` and
  `"spec-pmin"` only tune speculation and never turn it on. `"spec-type": "ngram"` needs no drafter at
  all, but it still needs a model that can act as a speculative target: it does
  not run on `gpt-oss-20b.json` (GPT-OSS is not one), and the diffusion, image,
  video and embedding configs do not decode token by token at all.
- **The three `gemma-4-*.json` files speculate by default, on both hosts.**
  Naming their MTP `"draft-model"` turns speculation on, with no `"spec"` key
  beside it. Pass `--no-spec` (or add `"no-spec": true`) to decode without it;
  the draft file is still fetched.
- **Qwen-Image-2.1** ([`qwen-image-2.1.json`](qwen-image-2.1.json)): `--prompt "…"
  --output out.png` generates an image, and adding `--image in.png` edits it instead.
  Add `--mask selection.png` to edit the selected region of the first image while
  preserving its dimensions and unselected decoded RGBA pixels; `--mask-feather`
  softens edges inward and `--mask-crop` reduces work for a small selection.
  Server JSON requests use `maskPath`, `maskMode`, `maskInvert`, `maskFeather`,
  `maskCrop` and `maskCropPadding`; Server Chat and TensorAgent provide a painted
  selection editor. See [the mask guide](../docs/models/qwenimage21.md#precise-local-editing-with-a-mask).
  `--diffusion-steps` (default 40, or the recipe of a step-distillation plug-in),
  `--cfg` (default 1), `--diffusion-seed` and `--width` / `--height` (multiples of
  32) are CLI flags. On the server a request carries its own `steps`, `cfg`,
  `seed`, `width` and `height`; `--width` and `--height` given together at startup
  set the default size for requests that name neither a size nor an area (a value
  off the 32-pixel grid is rounded down with a one-time warning, and one of the two
  alone is ignored with a warning). Add a LoRA plug-in from
  [`lora/`](#qwen-image-21-lora-plug-ins-lora) with `--lora`. See
  [the Qwen-Image-2.1 guide](../docs/models/qwenimage21.md).
  [`qwen-image-2.1-turbo.json`](qwen-image-2.1-turbo.json) runs the 8-step Turbo
  checkpoint the same way. Its GGUF has no metadata, so the config declares it with
  `"qwen-image-variant": "turbo"` (`--qwen-image-variant` on both hosts). Turbo then
  defaults to its published 8-step schedule at CFG 1, refuses other step counts, and
  refuses the step-distillation plug-ins of `lora/`. See
  [the Turbo section](../docs/models/qwenimage21.md#qwen-image-21-turbo).
- **DiffusionGemma** runs its iterative denoising path on both hosts; on the CLI,
  tune it with `--diffusion-steps` / `--diffusion-seed`. Its `mmproj` entry is the
  Gemma-4 vision tower, loaded straight from the upstream Hugging Face shard
  `model-00011-of-00011.safetensors` (2.84 GB), which enables `--image` on the CLI
  and image input on the server. Ordinary chat refuses audio and has no native video path: an
  OpenAI `video_url` part is refused, and a video uploaded in the Web UI reaches
  the model only as extracted frames, treated as plain images. To run text-only
  without that download, pass `--mmproj none`: a command-line `--mmproj` drops the
  file's entry before it is resolved, so the shard is never fetched (this works on
  the CLI too). The [Jev endpoint](../docs/models/jev.md#files-documents-video-and-audio)
  additionally accepts uploaded documents, sampled video frames and audio through
  an explicitly configured transcription service.
- **Every download entry in every config in this folder is pinned** to a full
  commit and a SHA-256, not to `main`, so every machine gets the same bytes; a
  test enforces it. That covers the basic examples, the Wan and MiniMax-H3
  configs, and the DiffusionGemma model files and vision shard alike. Each pin is
  recorded for those configs on 2026-09-25 (LTX 2.5 on 2026-10-13); to take a
  later upload, change the commit in the URL and the `sha256` together. The checksum
  applies only to a new download: a
  file already at `path` is used as-is and never re-checked, so an older local copy
  keeps being used until you delete it.

## Experimental LTX 2.5 preset

[`ltx-2.5-distilled.json`](ltx-2.5-distilled.json) records the verified Q4_K_M
distilled denoiser, the dedicated Gemma 4 text encoder with LTX projections, the
convolutional video VAE, the x2 spatial latent upscaler and the audio VAE. All five
files are pinned by commit and SHA-256, with a combined size of 44.76 GB. Paths use
`${TENSORSHARP_MODELS:-../models}/ltx25`, including the upstream `text_encoders/`,
`vae/` and `latent_upscale_models/` subfolders. The preset's output is 1024 × 1024,
17 frames at 24 fps.
The native generation model runs the dedicated Gemma 4 encoder and AV connectors,
then the publisher's two-stage distilled recipe: the complete 48-block denoiser
samples the published eight sigmas (ancestral Euler, CFG 1) at half the output size,
the latent upscaler doubles it, and three refinement steps run at full size. The
convolutional video decoder and the 48 kHz stereo audio decoder/vocoder finish.
TensorAgent offers **Download** and **Use** for **LTX 2.5 Distilled** (Q4_K_M).
The experimental catalog tier is 48 GB; this is not a measured minimum for every
resolution or duration. Width and height must be multiples of 64 (each pass keeps
the 32-pixel latent grid), and frame counts must be `8k+1`. First, last and
multiple timed image keyframes are supported. Optional Ingredients IC-LoRA adds
identity/appearance references with `videoMode: "ref"`; install it under `loras/`
beside the model or set `TS_LTX25_INGREDIENTS_LORA`. Its separate single-pass
recipe uses a trained 768×448×121 preview. See [LTX 2.5](../docs/models/ltx-2.5.md)
for setup, the optional diffusion video decoder, measured coverage and limits.

```sh
dotnet run --project TensorSharp.Cli -- --config config/ltx-2.5-distilled.json \
  --backend mlx --prompt "A red toy car drives slowly along a quiet road." --output ltx25.mp4
```

Use `--backend cuda` or `ggml_cuda` on NVIDIA GPUs. `mlx` / `ggml_metal`
route to MLX, and `cpu` / `ggml_cpu` to direct CPU; Vulkan and multi-GPU are
unsupported. On Apple Silicon, launch with
`MLX_ENABLE_TF32=0` to preserve Float32 denoiser/VAE precision, including when
switching from another MLX model in an existing process. The LTX model sets that
default before its own first MLX operation; MLX caches the setting process-wide.
The desktop CLI/server require FFmpeg on `PATH` (or `TS_FFMPEG`) to include audio
in the MP4; the CLI also saves the original WAV. The Mac app uses AVFoundation.

The official Lightricks companion files are gated. Accept the publisher's license
and provide `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, or a Hugging Face CLI token cache
(`HF_TOKEN_PATH`, `HF_HOME/token`, or the standard cache). Both downloaders attach
credentials only to HTTPS `huggingface.co` requests; credentials are not sent to
download mirrors or CDN redirects. The ordinary Gemma 4 model is not
interchangeable with the dedicated LTX text encoder, which embeds its tokenizer.

<a id="qwen-image-21-lora-plug-ins-lora"></a>

## LoRA plug-ins (`lora/`)

The files in [`lora/`](lora/) are **not** host configs: they are LoRA plug-ins that
you add to a run with `--lora`, on either host. The fourteen `qwen-image-2.1-*` files
are for Qwen-Image-2.1, and the two `minimax-h3-fl2v-turbo-*` files are for the
MiniMax-H3 FL2VA checkpoint ([below](#minimax-h3-turbo-plug-ins)). A plug-in the
startup model does not take is refused at startup, before the plug-in downloads; a model
a `--config` still has to download is checked for its family right after that download.

Each Qwen-Image-2.1 plug-in names its
weights with a pinned URL and SHA-256 (the Fun-Acc bundle pins the `pdd_config.json`
it forwards the same way), downloaded on first use to
`$TENSORSHARP_MODELS/qwen-image-2.1/loras` (or `models/qwen-image-2.1/loras` in the
repository when the variable is unset), and may carry a default strength and a
sampling recipe:

```bash
TensorSharp.Cli --config config/qwen-image-2.1.json \
  --lora config/lora/qwen-image-2.1-viggle-turbo.json --prompt "…" --width 1024 --height 1024

TensorSharp.Server.Host --config config/qwen-image-2.1.json --lora config/lora/qwen-image-2.1-pruna-8step.json
```

| File | Plug-in |
|------|---------|
| [`qwen-image-2.1-viggle-turbo.json`](lora/qwen-image-2.1-viggle-turbo.json) | Viggle Turbo step distillation: 6 steps by default (4–8 supported), CFG 1 |
| [`qwen-image-2.1-pruna-8step.json`](lora/qwen-image-2.1-pruna-8step.json) | Pruna DMD step distillation: 8 steps on fixed sigmas, CFG 1 |
| [`qwen-image-2.1-pruna-5step.json`](lora/qwen-image-2.1-pruna-5step.json) | Pruna DMD step distillation: 5 steps on fixed sigmas, CFG 1; faster, lower quality than 8 |
| [`qwen-image-2.1-fun-acc-4step.json`](lora/qwen-image-2.1-fun-acc-4step.json) | Alibaba PAI Fun-Acc PDD bundle: 4 steps with per-step output heads; forwards its `pdd_config.json` |
| [`qwen-image-2.1-film-stills.json`](lora/qwen-image-2.1-film-stills.json) | Danrisi Film Stills: cinematic 35 mm film-still style, strength 0.7 |
| [`qwen-image-2.1-grainscape.json`](lora/qwen-image-2.1-grainscape.json) | Danrisi Grainscape: grainy 35 mm colour-negative look, strength 0.7 |
| [`qwen-image-2.1-casual-phone.json`](lora/qwen-image-2.1-casual-phone.json) | Casual Phone Photo: candid phone-photo style, strength 0.8; include `lbxstyle` and a clear subject and setting |
| [`qwen-image-2.1-voxel-style.json`](lora/qwen-image-2.1-voxel-style.json) | Voxel Style (editing): attach a photo and request `Transform the image into a voxel style`, strength 1.0; experimental, backgrounds can change |
| [`qwen-image-2.1-fix.json`](lora/qwen-image-2.1-fix.json) | e-n-v-y Qwen-Image-2.1-Fix: a DoRA quality fix |
| [`qwen-image-2.1-detail-enhancer.json`](lora/qwen-image-2.1-detail-enhancer.json) | elusarca Detail Enhancer (editing): detail, upscaling and restoration |
| [`qwen-image-2.1-natural-exposure.json`](lora/qwen-image-2.1-natural-exposure.json) | prithivMLmods Natural Exposure (editing): balanced, neutral exposure |
| [`qwen-image-2.1-anime-consistency.json`](lora/qwen-image-2.1-anime-consistency.json) | WarmBloodAban Anime Consistency: consistent anime characters across edits, strength 0.7 |
| [`qwen-image-2.1-object-remover.json`](lora/qwen-image-2.1-object-remover.json) | prithivMLmods Object Remover Bbox (editing): removes red-boxed objects |
| [`qwen-image-2.1-object-mover.json`](lora/qwen-image-2.1-object-mover.json) | prithivMLmods Object Mover Bbox (editing): moves an object from one red box to the other |

Notes:

- **Official Turbo keeps its own 8 steps and CFG 1.** The four compatible style/edit
  adapters above add no sampling recipe. Use the Turbo host config, for example:

  ```bash
  TensorSharp.Cli --config config/qwen-image-2.1-turbo.json \
    --lora config/lora/qwen-image-2.1-casual-phone.json \
    --prompt "lbxstyle, an orange cat beside a blue mug on a kitchen counter" \
    --width 1024 --height 1024 --output casual-phone.png
  ```

  Do not add Viggle, Pruna or Fun-Acc speed adapters to the official Turbo model.
  See the [source review and validation scope](../docs/models/qwenimage21.md#lora-plug-ins-on-turbo)
  before treating a style change as a quality improvement.
- **Explicit settings win.** `--diffusion-steps` / `--cfg` (or a server request's
  `steps` / `cfg`) override a plug-in's recipe; a step count the recipe has no
  schedule for is refused, naming the supported ones.
- **Stack by repeating `--lora`**, but only one plug-in per run may carry a
  sampling recipe. `--lora-scale` after a `--lora` overrides its strength.
- **Check each license.** Every file's comments cite the model card, any trigger
  phrase and the license; several are under the Qwen Research License
  (non-commercial).
- The plug-in format (`"type": "qwen-image-2.1-lora"`, `weights`, `scale`,
  `sampling`, or a forwarded `config`) is described in
  [USAGE.md](../USAGE.md#qwen-image-21-lora-plug-ins).

### MiniMax-H3 Turbo plug-ins

The two `minimax-h3-fl2v-turbo-*` files are LightX2V's step-distillation LoRAs for the
MiniMax-H3 **FL2VA** checkpoint (text-to-video, image-to-video and first/last frame),
from [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/a29e57d195f80ead07c89f89e55ecf2911e893c5)
under Apache-2.0. Each downloads its 1.38 GB `.safetensors` on first use, pinned to
revision `a29e57d` and checked against its SHA-256, to
`$TENSORSHARP_MODELS/minimax-h3/loras` (or `models/minimax-h3/loras` in the repository
when the variable is unset):

```bash
TensorSharp.Cli --config config/minimax-h3-fl2va.json \
  --lora config/lora/minimax-h3-fl2v-turbo-8step.json --prompt "…" --output clip.mp4

TensorSharp.Server.Host --config config/minimax-h3-fl2va.json --lora config/lora/minimax-h3-fl2v-turbo-8step.json
```

| File | Plug-in |
|------|---------|
| [`minimax-h3-fl2v-turbo-8step.json`](lora/minimax-h3-fl2v-turbo-8step.json) | LightX2V Turbo 8-step (FL2V v1.0, 768p): 8 Euler steps on the publisher's recipe, video shift 6 / audio shift 3, CFG 1 |
| [`minimax-h3-fl2v-turbo-4step.json`](lora/minimax-h3-fl2v-turbo-4step.json) | LightX2V Turbo 4-step (FL2V v1.2, 768p): 4 Euler steps at video shift 6 / audio shift 3, CFG 1. The publisher states no recipe for v1.2, so this one is **inferred** from its 4-step v1.0 row until it is validated |

Notes:

- **The recipe is locked, not a default.** Steps, flow shift, sampler and CFG come from
  the plug-in. Leave `--diffusion-steps`, `--flow-shift`, `--sampler` and `--cfg` (CLI),
  `--video-steps` (server) and a request's `steps`, `flowShift` and `sampler` unset, or
  repeat the plug-in's values; anything else is refused, never overridden. Both hosts
  check this before anything downloads, the files of a `--config` included.
- **FL2VA only.** `minimax-h3-ref2va.json` refuses both files: Ref2VA Turbo is not
  supported.
- **One plug-in, as its config.** MiniMax-H3 takes one of these files, not the bare
  `.safetensors`, because the config declares the checkpoint partition and the
  sampling contract. `--lora-scale` must be ≥ 0: the schedule was distilled at
  strength 1, and 0 keeps the adapter's terms with zero up factors.
- **The format is strict** (`"type": "minimax-h3-lora"`, `weights`, `partition`,
  `key_format`, `alpha`, `scale`, `sampling`): an unknown key, the Ref2VA partition,
  or another schedule, sampler, audio shift or CFG is refused. See
  [the MiniMax-H3 card](../docs/models/minimax-h3.md#turbo-plug-ins).
- **Do not name a plug-in with a relative `"lora"` string in a preset.** A string
  value is passed to `--lora` as written, so a relative path resolves against the
  working directory, not against the preset's folder the way its file entries do:
  `"lora": "lora/minimax-h3-fl2v-turbo-8step.json"` works only when the host starts
  in `config/`. Pass `--lora` on the command line, as above, or write the entry as an
  object, `"lora": { "path": "lora/minimax-h3-fl2v-turbo-8step.json" }`, which resolves
  against the preset's folder. This holds for every plug-in in `lora/`.

## Agent configs

The `agent-*.json` files turn a model into an **agent** rather than a chat
endpoint: Agent Skills and their bundled scripts, network access for both, the
`shell` / `read_file` / `write_file` / `apply_patch` tool loop, and
host-performed `pip` / `npm` installs are all switched on in one file. They point
at local paths on an Apple Silicon Mac (`ggml_metal`), so change `backend` and the
`variables` block to run them elsewhere. A missing model file downloads from its
upstream repo, pinned by commit and SHA-256.

```bash
TensorSharp.Cli         --config config/agent-gemma-4-12b.json --chat
TensorSharp.Server.Host --config config/agent-qwen3.8-27b.json
```

| File | Model | Speculative decoding |
|------|-------|----------------------|
| [`agent-qwen3.8-27b.json`](agent-qwen3.8-27b.json) | Qwen3.8-27B (UD-Q4_K_XL, 16.4 GB) | Off — the embedded NextN head measured only ~1.04× |
| [`agent-qwen3.6-35b-a3b.json`](agent-qwen3.6-35b-a3b.json) | Qwen3.6-35B-A3B MoE (UD-IQ2_XXS, 11 GB) | Off — MTP measured **2× slower** on this MoE |
| [`agent-gemma-4-12b.json`](agent-gemma-4-12b.json) | Gemma-4 12B (QAT UD-Q4_K_XL, 6.3 GB) | **On** — MTP draft head, auto window 7 |
| [`agent-gemma-4-26b-a4b.json`](agent-gemma-4-26b-a4b.json) | Gemma-4 26B-A4B MoE (QAT UD-Q4_K_XL, 13.3 GB) | Off — no Metal MoE measurement exists yet |

Notes (the files repeat most of these as comments):

- **`--code-exec-allow-network` is the broadest permission here.** Every command
  the model writes gets unrestricted host IP networking, including LAN and
  loopback. Both hosts print a warning at startup. Do not run these configs on a
  server reachable by people you do not trust.
- **Skill scripts and generated commands have separate network switches** —
  `"skills-allow-network"` and `"code-exec-allow-network"`, neither implying the
  other. Both are set, because the `research` and `market-data` skills need the
  first and generated code needs the second.
- **`"skills-dir"` is not optional.** Without it (and without the `TS_SKILLS_DIR`
  environment variable) the host looks only in every `.agents/skills` folder from
  the working directory up to the Git root and then in the `skills/` folder beside
  the binary; this repository has no `.agents/skills` and the `skills/` folder
  beside the binary is empty (the host creates it empty), so a config without this
  key yields an agent with zero skills. These files point at `TensorAgent/skills`: 12 skills, seven of
  which bundle runnable scripts — four Python (`documents`, `research`,
  `slack-gif-creator`, `market-data`), two shell (`web-artifacts-builder` and the
  `playwright` browser-automation wrapper, which needs Node.js/`npx` on the host;
  see [the Playwright guide](../docs/playwright_agent.md)) and one JS template
  (`algorithmic-art`). A path that does not exist is a fatal startup error.
  Command-line `--skills-dir` values replace the configured roots; repeat the flag
  to supply multiple roots. `TENSORSHARP_SKILLS` can also move the configured root.
- **`"temperature"` must be pinned, or the two hosts disagree.** These GGUFs carry
  `general.sampling.temp = 1.0`, which the CLI's chat path overlays onto any field
  left unset while the server ignores it and falls to its built-in 0.8.
  `--code-exec-temperature` cannot bridge that either: it rewrites a temperature
  only while it still sits at that built-in 0.8. The same applies to
  `"repeat-penalty"`, pinned to 1.0 here so every turn on both hosts matches.
- **`--max-tokens` is the generation cap, never the context window**, and it also
  sizes the up-front KV reservation. The window is environment-only: export
  `MAX_CONTEXT` before launching. Overshooting on Metal is not a slowdown — past
  the working set, ggml latches a sticky error and every later graph fails.
- **The agent works in its own workspace, not in your checkout.** Writes are
  confined to the session workspace and the whole home directory is unreadable, so
  the model has to copy anything it needs in. `"code-exec-unconfined"` is left off
  deliberately; on macOS it downgrades confinement to best-effort.
- **`"skills-max-rounds"` is one shared budget** for skill lookups and shell rounds.
  Naming it at all cancels the automatic raise to 24 that `--code-exec` applies, so
  it is set to 32 — never set it below 24.
- **Gemma-4 cannot use a quantized KV cache.** `Gemma4Model` declines block
  quantization because its circular sliding-window cache helpers are float-only, so
  an explicit `"kv-cache-dtype": "q8_0"` is downgraded to f16 at load with a note on
  stderr. The two Qwen configs do use it, halving a cache Metal charges twice.
- Two options must be a **single comma-separated string** if you add them:
  `"code-exec-packages"` and `"code-exec-install-domains"`. A JSON array becomes a
  repeated flag whose parse is last-one-wins, so only the final entry survives.

One more behaviour the files do not mention: **on the server these models can also
delegate to sub-agents.** Delegation is on by default on the server's chat endpoints
for every family that renders tool declarations, which includes all four models here.
The model decides whether to spawn a helper, and helpers are read-only (only a
`worker` helper gets write tools, and only with `--agents-allow-worker-tools`); add
`"no-multi-agent": true` to turn delegation off. The CLI has no sub-agents, and no
latency or quality numbers are published for delegation. See
[Multiple agents](../docs/multi_agent.md).

## Video generation with sound (MiniMax-H3)

MiniMax-H3 denoises video **and 32 kHz stereo audio in one packed latent**, so a run
writes an `.mp4` and a matching `.wav`. Four networks cooperate — denoiser, text
encoder, video VAE, audio VAE — which makes a config file the practical way to run it:

```bash
TensorSharp.Server.Host --config config/minimax-h3-fl2va.json

TensorSharp.Cli --config config/minimax-h3-fl2va.json \
  --prompt "a red fox trotting through falling snow, cinematic" --output fox.mp4

# Image-to-video, and first-and-last-frame:
TensorSharp.Cli --config config/minimax-h3-fl2va.json \
  --image start.png --end-image end.png \
  --prompt "a slow cinematic push-in" --output morph.mp4

# References, on the OTHER checkpoint: up to nine, in any mix of stills, clips
# and soundtracks. The clip does not reproduce them; it borrows the identity and
# appearance they carry into a brand-new scene.
TensorSharp.Cli --config config/minimax-h3-ref2va.json \
  --ref-image person.png --ref-image jacket.png --ref-image street.png \
  --prompt "she walks through a night market, neon reflections" --output out.mp4
```

| Config | Denoiser | VAEs | Text encoder | Modes |
|---|---|---|---|---|
| [`minimax-h3-fl2va.json`](minimax-h3-fl2va.json) | FL2VA pruned (Q4_K) | H3 video **+ H3 audio** | Qwen3-VL-32B | T2V + I2V + first/last frame |
| [`minimax-h3-ref2va.json`](minimax-h3-ref2va.json) | Ref2VA pruned (Q4_K) | H3 video **+ H3 audio** | Qwen3-VL-32B | T2V + references (stills, clips, soundtracks) |

Notes:

- **The first run downloads ~35.5 GB** (~35.4 GB for Ref2VA) — denoiser 10.64 GiB, text encoder 16.97 GiB,
  video VAE 5.21 GB, audio VAE 0.61 GB. All four come from unsloth, with Comfy-Org
  listed as a second source for the two VAEs. Each network is loaded and released in
  turn, so peak VRAM is `max(...)` and not the sum.
- **Drop the `audio-vae` entry and you still get video**, just silent — that entry is
  what decodes the audio half of the latent.
- **The text encoder ships no tokenizer**, and auto-download cannot fill that gap: it
  only resolves options that are flags, and the tokenizer is not one. Put `vocab.json`,
  `merges.txt` and `tokenizer_config.json` from
  [MiniMaxAI/MiniMax-H3/processor](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/processor)
  (the config comments give pinned `curl` lines)
  next to the encoder GGUF in `${modelRoot}`, or point `TS_VIDEO_TOKENIZER` at the
  folder holding them. Without the first two the run stops when the encoder loads.
  Without `tokenizer_config.json` text-to-video still runs, and so do references that
  are all sound recordings, but a keyframe, or a photo or clip reference, is refused
  before the vision tower runs: that file alone defines `<|vision_start|>`,
  `<|image_pad|>` and `<|vision_end|>`, and without them the picture used to be placed
  over the wrong prompt positions and quietly ignored.
- **Steps and guidance are host-specific, so no shipped config sets them.** The server
  takes `--video-steps N` and has no `--cfg` at all; the CLI takes `--diffusion-steps N`
  and `--cfg`. A flag the server does not know is not ignored — it refuses to start —
  so a config carrying a CLI-only key would break `TensorSharp.Server.Host --config`. H3 is
  CFG-distilled and enforces cfg 1.0 itself; its default is 20 steps. For fewer steps,
  add a [Turbo plug-in](#minimax-h3-turbo-plug-ins) to `minimax-h3-fl2va.json`
  (`--lora config/lora/minimax-h3-fl2v-turbo-8step.json`, or the 4-step one), which
  samples exactly the 8 or 4 steps it was distilled for.
- **`video-frames` snaps to the `17k+5` grid** (5, 22, 39, 56, 73, 90 …), not Wan's
  `4k+1`, and fps is pinned to 24.
- **`minimax-h3-fl2va.json` leaves the size to the model.** Text-to-video runs at
  640×384; `--image` and `--end-image` take the canvas from the first keyframe by
  MiniMax's released 768p recipe (1152×768 for a 3:2 photo, 1344×768 for 16:9), which
  keeps a keyframe's faces from smearing at about 2.4-4.2× the video tokens per frame,
  so those runs take several times longer. `minimax-h3-ref2va.json` pins 640×384, the
  size references run at. `--width` and `--height` together (`--video-width` and
  `--video-height` on the server) pin one size for every mode, keyframes included.
- **References need the other checkpoint**, which now has its own file. FL2VA and Ref2VA
  are separate checkpoints rather than a setting: `minimax-h3-fl2va.json` refuses
  `--ref-image` and `minimax-h3-ref2va.json` refuses `--image` as a keyframe, and each
  error names the other file. Only the `model` entry and the size pin differ between the
  two configs, so the text encoder and both VAEs are shared and will not download twice.
- **Ref2VA takes up to nine references in any mix** — `--ref-image` (a still),
  `--ref-video` (a clip, or a folder of frames), `--ref-audio` (a soundtrack), and
  `--ref-video-audio` to give the i-th `--ref-video` its own sound. Each takes its own
  stretch of the shared timeline before the generated clip and keeps its own aspect
  ratio, so every reference you add lengthens the packed sequence the DiT attends over.

## Video generation, video-only (Wan)

Wan generates **video alone**, from three cooperating networks instead of H3's four,
and a config file is still the easiest way to run it — it names them all and
downloads whatever is missing:

```bash
# Text-to-video and image-to-video, 4-step distilled. Start here.
TensorSharp.Server.Host --config config/wan-video-ti2v-5b-turbo.json

TensorSharp.Cli --config config/wan-video-ti2v-5b-turbo.json \
  --prompt "a red fox trotting through falling snow" --output fox.mp4

# Image-to-video: the image becomes frame 0 and the prompt drives the motion.
TensorSharp.Cli --config config/wan-video-ti2v-5b-turbo.json \
  --image first_frame.png --prompt "she turns and smiles" --output clip.mp4
```

| Config | DiT | VAE | Text encoder | Modes |
|---|---|---|---|---|
| [`wan-video-ti2v-5b-turbo.json`](wan-video-ti2v-5b-turbo.json) | TI2V-5B Turbo (4-step) | Wan 2.2 | UMT5-XXL | T2V + I2V |
| [`wan-video-ti2v-5b.json`](wan-video-ti2v-5b.json) | TI2V-5B (50-step) | Wan 2.2 | UMT5-XXL | T2V + I2V |
| [`wan-video-i2v-a14b.json`](wan-video-i2v-a14b.json) | A14B high **+** low noise | Wan 2.1 | UMT5-XXL | I2V only |

Notes:

- **The VAEs are not interchangeable.** TI2V-5B uses the Wan **2.2** VAE (48-channel
  latent, 16×16×4 compression); A14B and the Wan 2.1 models use the Wan **2.1** VAE
  (16-channel, 8×8×4). Each config names the right one.
- **UMT5-XXL is shared** by every Wan model, so the ~6 GB encoder downloads once and
  the other Wan configs reuse it from `${modelRoot}`.
- **A14B needs both experts.** They are auto-paired by their `high_noise`/`low_noise`
  names when they sit in the same folder; `video-dit2` in the config is what lets the
  second one auto-download.
- **Turbo/Lightning/distilled checkpoints are detected from the file name** and
  switch to 4 steps with guidance off — around 25× fewer DiT passes. The startup log
  prints `step-distilled checkpoint detected`; `--diffusion-steps` / `--cfg` override
  it.
- Each network is loaded and released in turn, so peak VRAM is
  `max(text encoder, DiT, VAE)` rather than their sum.
- `video-frames` is snapped to `4k+1`. Halving it is the cheapest quality-neutral
  saving: self-attention is O(tokens²), so 121 → 61 frames is roughly 4× less
  attention work and half the VAE decode.
