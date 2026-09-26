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

**Command-line options always win.** When the command line sets a single-valued
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
one's the same way (and the command line still wins over all of them). The
repeatable options — `--stop`, `--skills-dir`, `--skill`, `--lora`,
`--lora-scale`, `--lora-config`, `--image`, `--ref-image`, `--ref-video`,
`--ref-audio` and `--ref-video-audio` — keep every file's values and add the
command line's after them instead, and a download entry under one of them still
runs. Legacy spellings count as the same option in both directions: `--wan-vae`
and `--video-vae`; `--wan-te`, `--video-te` and `--video-text-encoder`;
`--wan-dit2` and `--video-dit2`.

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

`"variables"` (see below), its alias `"vars"`, and `"$schema"` are reserved and
never become options. A key that names a removed option is refused with a message
saying what to use instead, and `lora-scale` / `lora-config` cannot be arrays,
because each binds to the one `--lora` before it.

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

## Examples in this folder

Every example uses **real, public, ungated** Hugging Face URLs, so the files
auto-download to their local `path` on the first run and are reused afterward.
The one exception is the MiniMax-H3 tokenizer (`vocab.json` + `merges.txt`),
which you place by hand; see [below](#video-generation-with-sound-minimax-h3).
If you already have a file at that `path`, it is used as-is (no download).

| File | Model(s) | Shows |
|------|----------|-------|
| [`cli-basic.json`](cli-basic.json) | Qwen3.5-9B (~8.9 GB) | Minimal CLI config, auto-download |
| [`server-basic.json`](server-basic.json) | Gemma-4 E4B model + vision projector | Multimodal server, sampling defaults, auto-download |
| [`variables.json`](variables.json) | Gemma-4 26B-A4B: model + mmproj + MTP draft | One shared root/repo reused across three related files |
| [`auto-download.json`](auto-download.json) | Qwen3.5-9B (~8.9 GB) | Auto-download demo using a public GGUF |
| [`qwen-image-2.1.json`](qwen-image-2.1.json) | Qwen-Image-2.1 Q4_K_M + dedicated VAE + Qwen3-VL-8B + projector | Text-to-image and editing; pinned, checksum-verified downloads |
| [`lora/qwen-image-2.1-*.json`](lora/) | Twelve Qwen-Image-2.1 LoRA plug-ins (each downloads its weights; Fun-Acc also its `pdd_config.json`) | `--lora` plug-ins on top of `qwen-image-2.1.json`: step-distilled 4–8-step recipes, styles and editing skills; see [below](#qwen-image-21-lora-plug-ins-lora) |
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
| [`gemma-4-e4b.json`](gemma-4-e4b.json) | Gemma-4 E4B **uncensored** (TrevorJS's community build, Q8_0) + stock unsloth vision projector + AtomicChat MTP draft | Multimodal LLM |
| [`gemma-4-12b.json`](gemma-4-12b.json) | Gemma-4 12B (QAT) + vision + MTP draft | Multimodal LLM |
| [`gemma-4-26b-a4b.json`](gemma-4-26b-a4b.json) | Gemma-4 26B-A4B (MoE) + vision + MTP draft | Multimodal MoE LLM |
| [`gpt-oss-20b.json`](gpt-oss-20b.json) | gpt-oss-20b (Q8_0) | Text reasoning LLM |
| [`jev-diffusiongemma-q4.json`](jev-diffusiongemma-q4.json) | DiffusionGemma 26B-A4B (Q4_K_M) + vision | [Jev typed decisions](../docs/models/jev.md) over text or image state, native `/v1/systemone` server |
| [`diffusiongemma-26b-a4b-q4.json`](diffusiongemma-26b-a4b-q4.json) | DiffusionGemma 26B-A4B (Q4_K_M) | Text diffusion + image input (CLI/server); auto-downloads the vision shard |
| [`diffusiongemma-26b-a4b-q3.json`](diffusiongemma-26b-a4b-q3.json) | DiffusionGemma 26B-A4B (Q3_K_M) | Text diffusion + image input, smaller; shares the Q4 vision shard |

For the agent-enabled counterparts of three of these models, plus Qwen3.8-27B — skills,
code execution, network and package installs switched on — see
[Agent configs](#agent-configs) below.

Notes:

- **Multimodal** configs load a vision projector, so add `--image photo.png` to ask
  about a picture.
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
  `--diffusion-steps` (default 40, or the recipe of a step-distillation plug-in),
  `--cfg` (default 1), `--diffusion-seed` and `--width` / `--height` (multiples of
  32) are CLI flags. On the server a request carries its own `steps`, `cfg`,
  `seed`, `width` and `height`; `--width` and `--height` given together at startup
  set the default size for requests that name neither a size nor an area (a value
  off the 32-pixel grid is rounded down with a one-time warning, and one of the two
  alone is ignored with a warning). Add a LoRA plug-in from
  [`lora/`](#qwen-image-21-lora-plug-ins-lora) with `--lora`. See
  [the Qwen-Image-2.1 guide](../docs/models/qwenimage21.md).
- **DiffusionGemma** runs its iterative denoising path on both hosts; on the CLI,
  tune it with `--diffusion-steps` / `--diffusion-seed`. Its `mmproj` entry is the
  Gemma-4 vision tower, loaded straight from the upstream Hugging Face shard
  `model-00011-of-00011.safetensors` (2.84 GB), which enables `--image` on the CLI
  and image input on the server. Audio is refused, and there is no video path: an
  OpenAI `video_url` part is refused, and a video uploaded in the Web UI reaches
  the model only as extracted frames, treated as plain images. To run text-only
  without that download, pass `--mmproj none`: a command-line `--mmproj` drops the
  file's entry before it is resolved, so the shard is never fetched (this works on
  the CLI too).
- **Every download entry in every config in this folder is pinned** to a full
  commit and a SHA-256, not to `main`, so every machine gets the same bytes; a
  test enforces it. That covers the basic examples, the Wan and MiniMax-H3
  configs, and the DiffusionGemma model files and vision shard alike. Each pin is
  the upstream head as of 2026-09-25; to take a later upload, change the commit in
  the URL and the `sha256` together. The checksum applies only to a new download: a
  file already at `path` is used as-is and never re-checked, so an older local copy
  keeps being used until you delete it.

## Qwen-Image-2.1 LoRA plug-ins (`lora/`)

The files in [`lora/`](lora/) are **not** host configs: they are LoRA plug-ins that
you add to a Qwen-Image-2.1 run with `--lora`, on either host. Each one names its
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
| [`qwen-image-2.1-fix.json`](lora/qwen-image-2.1-fix.json) | e-n-v-y Qwen-Image-2.1-Fix: a DoRA quality fix |
| [`qwen-image-2.1-detail-enhancer.json`](lora/qwen-image-2.1-detail-enhancer.json) | elusarca Detail Enhancer (editing): detail, upscaling and restoration |
| [`qwen-image-2.1-natural-exposure.json`](lora/qwen-image-2.1-natural-exposure.json) | prithivMLmods Natural Exposure (editing): balanced, neutral exposure |
| [`qwen-image-2.1-anime-consistency.json`](lora/qwen-image-2.1-anime-consistency.json) | WarmBloodAban Anime Consistency: consistent anime characters across edits, strength 0.7 |
| [`qwen-image-2.1-object-remover.json`](lora/qwen-image-2.1-object-remover.json) | prithivMLmods Object Remover Bbox (editing): removes red-boxed objects |
| [`qwen-image-2.1-object-mover.json`](lora/qwen-image-2.1-object-mover.json) | prithivMLmods Object Mover Bbox (editing): moves an object from one red box to the other |

Notes:

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
  `--skills-dir` is repeatable, so one given on the command line adds a root
  rather than replacing this one; move this one with `TENSORSHARP_SKILLS`.
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
  only resolves options that are flags, and the tokenizer is not one. Put `vocab.json`
  and `merges.txt` from
  [MiniMaxAI/MiniMax-H3/processor](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe/processor)
  (the config comments give pinned `curl` lines)
  next to the encoder GGUF in `${modelRoot}`, or point `TS_VIDEO_TOKENIZER` at the
  folder holding them. Without them the run stops when the encoder loads.
- **Steps and guidance are host-specific, so no shipped config sets them.** The server
  takes `--video-steps N` and has no `--cfg` at all; the CLI takes `--diffusion-steps N`
  and `--cfg`. A flag the server does not know is not ignored — it refuses to start —
  so a config carrying a CLI-only key would break `TensorSharp.Server.Host --config`. H3 is
  CFG-distilled and enforces cfg 1.0 itself; its default is 20 steps, and 4-8 is the
  fast operating point.
- **`video-frames` snaps to the `17k+5` grid** (5, 22, 39, 56, 73, 90 …), not Wan's
  `4k+1`, and fps is pinned to 24.
- **References need the other checkpoint**, which now has its own file. FL2VA and Ref2VA
  are separate checkpoints rather than a setting: `minimax-h3-fl2va.json` refuses
  `--ref-image` and `minimax-h3-ref2va.json` refuses `--image` as a keyframe, and each
  error names the other file. Only the `model` entry differs between the two configs, so
  the text encoder and both VAEs are shared and will not download twice.
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
