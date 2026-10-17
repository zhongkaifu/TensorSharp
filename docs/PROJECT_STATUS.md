# TensorSharp project status

This page describes the current source as of 2026-10-15. The latest local release tag is `v2026.10.11`, which includes knowledge collections and active-turn steering. LTX 2.5, EmbeddingGemma 2, community skill discovery/installation and the latest MiniMax-H3 precision/performance work were added after that tag; use a build containing those changes. Release packaging configuration alone does not prove an asset was published or tested.

## Current direction

TensorSharp is a native .NET 10 inference engine for GGUF models. The current source includes CLI, server/Web UI, compatible HTTP APIs, AgentHost, and the TensorAgent application for iOS/iPadOS, with Mac and Windows desktop builds from the same project. AgentHost ships inside the CLI and server archives and as the `TensorSharp.AgentHost` NuGet package. The Release Binaries workflow also packages TensorAgent Desktop for Apple Silicon Mac (DMG/PKG/ZIP) and Windows x64 CPU/CUDA (MSI/ZIP); historical releases may lack Desktop assets, and iPhone/iPad still require source builds. See the [Desktop download, installation and first-chat guide](tensoragent_desktop.md). Packages and archives reflect their own tags, rather than every feature on the current branch. The [release notes](#release-notes-compatibility-and-recent-changes) separate current additions from older compatibility changes; configured packaging jobs do not establish that a particular release asset was published or tested.

### TensorAgent: chat, agents and media in one app

TensorAgent runs text and multimodal chat, code and skill workflows, personal knowledge retrieval, and model-specific image/video generation from the same conversation interface. Its searchable catalog groups entries by model family and filters them by the device memory budget. It includes Qwen-Image-2.1 and the explicitly declared eight-step Turbo variant for text-to-image and reference-image editing, with selectable LoRA plug-ins and a **Select area** editor that preserves pixels outside the selection, plus MiniMax-H3 (whose keyframes entry offers two Turbo speed plug-ins) and experimental LTX 2.5 Distilled for video with jointly generated soundtracks. LTX has a conservative 48 GB catalog tier and a five-file, 44.76 GB pinned download; its default preview is 1024×1024×17 at 24 fps. Catalog availability is distinct from validation: existing Mac app media measurements do not establish coverage of every new catalog entry, and iOS and Windows media generation remain unverified.

The interface now supports English, Simplified and Traditional Chinese, Japanese, Korean, Spanish, French and German. Settings can follow the system language or select one explicitly; the model still answers according to the user's prompt. Text turns show saved token count, elapsed time, token rate and available KV reuse counters. **Enter** sends a prompt, **Shift+Enter** inserts a newline, and Enter used to finish IME composition does not submit it. During a text response, a new message can steer the active turn: the host cancels the current generation, retains its partial answer and accepted guidance, then resumes from that conversation. The standalone browser chat and interactive CLI also support steering; it does not undo tools that have already run. See [usage](../USAGE.md) for the interface-specific controls.

`dotnet build TensorSharp.slnx -c Release` also builds TensorAgent.Core, TensorAgent.Sharing and TensorAgent.Tests. On macOS and Windows it attempts the relevant MAUI app heads; missing workloads or staged native prerequisites produce explicit skip warnings. A skipped head is not a validated app build. See the [TensorAgent guide](../TensorAgent/README.md) and [development instructions](../DEVELOPMENT.md#building).

### Personal knowledge collections

`TensorSharp.Knowledge` is an embedded, single-writer local store that retains originals, extraction status, source locations and versioned references. TensorAgent's **Menu → Knowledge** supports collection creation and renaming, file and nested-folder uploads, saved-chat snapshots with available attachments and produced files, browsing, search, original previews/downloads, and reindexing. Importing a source does not select it for inference: explicitly choose **Use {collection} in this chat**. Saved chats restore their selection; new chats start unselected. Cited image/video sources can show thumbnails and video sample timestamps; **Retrieved sources** contains only evidence actually cited in the answer.

BM25 retrieval works without an embedding model. Optional GGUF embeddings add semantic search, and chat hosts can apply one bounded model relevance pass with retrieval-order fallback. Text, records, PDFs and Office files have format-specific extraction limits; compatible loaded media models derive descriptions or transcripts from images, audio and sampled video. These generated representations remain partial, and unsupported sources remain visibly metadata-only. PDF OCR uses Apple Vision on supported Apple hosts; TensorAgent core hosts on Windows/Linux can provision Poppler/Tesseract on first use. This does not add a Linux MAUI application head or establish Windows/device runtime coverage.

The standalone server opts in with `TS_KNOWLEDGE_DIRECTORY` and exposes `/api/knowledge`. Supported chat/Responses/Ollama requests select scope using `knowledgeCollections`; management search/read use `collectionIds`. Knowledge tools share that scope with subagents and can copy complete originals into their private workspaces for calculations. There is no automatic knowledge selection for the CLI and no multi-user permissions layer. See the [knowledge guide](knowledge-system.md) for extraction coverage, endpoints, citations and validation tools; configured workflows and unavailable model/device scenarios are not passing validation.

### Embedding models and serving

Current source supports BERT/XLM-R sentence encoders (Snowflake Arctic Embed L v2.0 and MiniLM) and **EmbeddingGemma 2** (`gemma-embedding2`). EmbeddingGemma 2 has an 8192-token window, normalized 768-dimensional output and renormalized 512/256/128-dimensional Matryoshka output. Its text/code path and optional `--mmproj` image/audio/sampled-video path run through `cpu`, `mlx`, `cuda`, `ggml_cpu`, `ggml_metal` and `ggml_cuda`; Vulkan is unavailable. `--embeddings` starts the dedicated OpenAI `/v1/embeddings` / Ollama `/api/embed` service; [`config/embeddinggemma2.json`](../config/embeddinggemma2.json) is the checksum-pinned native CPU text preset.

Independent full-model text and seven-case media checks are distinct from timing and retrieval quality. CPU text and GPU multimodal latency parity with llama.cpp remain unachieved; controlled CPU media timing passes do not turn failed cross-engine numerical comparisons into passes. The [embedding guide](embeddings.md) records current independent-oracle checks, codec differences, actual backend coverage and reproducible comparisons.

### Current media-generation models

**LTX 2.5 Distilled** uses the publisher's two-stage recipe: eight ancestral Euler steps at half output size, required x2 latent upscaling, then three full-size refinement steps, all at CFG 1. It supports text, first/last/timed image keyframes, and optional Ingredients IC-LoRA identity sheets (a separate single-pass reference recipe). Output is video with 48 kHz stereo audio. CUDA and MLX use owned executors, including the `ggml_cuda` / `ggml_metal` aliases; CPU uses the direct implementation. Vulkan, multi-GPU and reference video/audio are unsupported. The default convolutional video VAE has an optional diffusion-decoder alternative on CPU/MLX. See the [model card](models/ltx-2.5.md) for downloads, quality limits, application coverage and the corrected recipe; older single-pass timings do not benchmark today's default.

**MiniMax-H3** has distinct FL2VA (text/keyframes) and Ref2VA (reference-media) checkpoints, 32 kHz stereo audio, CFG 1 and a 20-step default. All six CPU/MLX/CUDA direct/native routes have component coverage; current corrected CPU whole-generation pairs remain pending. Latest four-GPU-route preservation checks use unchanged ggml `3612c0af93f2cfc8f048295300a61187723b12ae`. Two-round full-process before/after checks preserve decoded RGB/audio exactly, with 5.0% lower native Metal latency, 12.5% lower MLX latency and 19.0% lower equal-precision direct CUDA latency; the 1.3% native CUDA difference does not establish a meaningful gain. Loading/codecs and uncontrolled caches are included. BF16-weight CUDA projections now default to pedantic F32; an experimental batched-attention model path failed full-generation quality checks and was removed. See the [model card](models/minimax-h3.md) and [validation workflow](../eng/validation/README-minimax-h3.md) for independent numerical gates and benchmark limits.

### Bundled skills and community discovery

The shared bundle has 17 skill packs, including `find-skills`, `grill-me`, `grill-with-docs`, `tdd` and `handoff`. iOS admits 15 by static runtime checks, excluding the desktop `playwright` and `web-artifacts-builder` packs; admission is not physical-device validation. Native `skills_search`, `skills_inspect` and `skills_install` discover public skills.sh/GitHub packages without Node or subprocesses. Network permission and reachable `find-skills` are required; installation also needs the server/TensorAgent writable registry and an immutable revision inspected in the same request. CLI defaults allow search/inspection, with no writable install registry. Script and code permissions remain separate. The real-model marketplace sample passed three of four full tasks despite successful transport in all cases; broader marketplace, GUI, Windows and physical-iOS coverage is unestablished. See [Agent Skills](agent_skills.md).

### Newest architectures

These architecture families carry limits worth knowing before you plan around them; their original support is in the `v2026.09.01` tag, while the cards track later changes.

- **DeepSeek V4.1 Flash (`deepseek41`)** — a dedicated native V4.1 graph with an
  optional vision companion. `ggml_cuda` is the serving backend; `ggml_cpu`
  runs the same graph on scalar fallbacks and `cpu` runs a pure-C# V4.1 executor,
  both as correctness and portability paths. `cuda` runs V4.1 through the
  direct-CUDA engine's own kernels; targeted expert-kernel and sequence-state tests do not establish a full-checkpoint numerical gate.
  `ggml_vulkan` and `ggml_metal` need `TS_DSV41_ALLOW_NON_CUDA_GPU=1`; `mlx`
  refuses the checkpoint. New downloads should use the repaired ten-shard
  Q2_K/Q5_K package pinned in the [model card](models/deepseek41.md#download-the-repaired-q2_kq5_k-checkpoint):
  it preserves the sensitive mHC matrices and Engram gates and includes Engram
  weights and hash constants, which TensorSharp reads directly without a sidecar.
  Bounded plain/DSpark text and image probes passed under both two-GPU layer split
  and routed-expert TP; they do not establish broad quality or performance.
  Historical Q2_K and Q4_K_M tests are recorded separately from those checks; at Q4_K_M the
  two Engram tables are 51.5 GiB each. On 8x46 GB they stay host mappings, and
  the checkpoint needs routed-expert CPU offload (see the
  [Q4_K_M throughput report](perf/deepseek41-q4km-throughput.md); the
  quantization report, `docs/validation/deepseek41-quants/README.md`, is local
  validation evidence, not committed).
  `--layer-split N` selects local whole-layer placement; experimental routed-MoE
  tensor parallelism uses `--tp N`. Historical
  seven-shard measurements were slower than layer split; they do not qualify the
  repaired ten-shard checkpoint. Concurrent requests have isolated slots; the fused `ggml_cuda` executor
  batches decode tokens into one graph, while CPU, DSpark and the serial opt-out
  use per-slot forwards. Batched and solo GEMMs can choose different near-tied
  tokens; the card records throughput and isolation checks separately. V4.1 DSpark is experimental: `--draft-model`
  loads a `deepseek41-dspark` drafter on `ggml_cuda` or `ggml_cpu` only. Initial
  text/image HTTP probes with trained weights passed using two-GPU layer split
  on `ggml_cuda`; broad quality and throughput remain unqualified. A multi-turn chat
  reuses its KV prefix: the reasoning drop in ordinary chat makes the render
  diverge one token after the previous turn's assistant header, and the native
  executor rewinds to that point instead of re-prefilling the conversation, which
  needs a per-slot checkpoint of the raw sliding-window ring because generating an
  answer wraps it. What is measured, and
  what is explicitly not, is tracked in the
  [validation report](deepseek41_validation.md) beside the
  [model card](models/deepseek41.md).
- **Hunyuan Dense (`hunyuan-dense`)** — Tencent's dense Hunyuan decoders, added
  so the official Hy-MT2 GGUFs load instead of failing on an unregistered
  architecture. First cut: text only, single device, generic per-op path, no
  tools and no thinking. See the [model card](models/hunyuan-dense.md).

GLM-5.3 is not on that list because it needed no new architecture: the non-Flash
release is the same `glm-dsa` block shape as GLM-5.2 — 79 blocks (78 trunk plus
one NextN), 256 routed experts at top-8 with one shared expert, MLA with the
lightning indexer, rope base 8e6 — so it loads on the GLM-5.2 path with no new
code and no new flag. It is text only ([unsloth/GLM-5.3-GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF)
publishes no mmproj at any quant, and `LoadVisionEncoder` warns and ignores an
`--mmproj` on `glm-dsa` rather than failing the run), `--spec` engages on the
layer-split mode rather than under `--tp` because the `blk.78` NextN block
ships no LM head of its own and borrows the trunk LM head, which `--tp` splits
column-wise, and UD-Q2_K_XL is 236.4 GiB across seven shards. Measured against
llama.cpp on 8x A40 46 GB — 10,531-token prompt, 300 decode tokens, median of 3,
whole-layer placement — it is a decode tie (20.48 vs 20.28 t/s) with a 2.9×
faster load of that 236.4 GiB checkpoint (264 s vs 753 s) and a slower TTFT
(41.9 s vs 29.0 s); see the [GLM card](models/glm.md#glm-53-glm-dsa). The full
cross-engine report, `docs/validation/cross-engine-2026-09/README.md`, is local
validation evidence, not committed.

Bonsai2 27B needed no new architecture string either. Its
`Ternary-Bonsai-2-27B-PQ2_0.gguf` and `-PTQ1_0.gguf` files declare `qwen35` and
are recognised by their PRISM `prism.hadamard.*` metadata and PQ2_0 / PTQ1_0
tensors, which TensorSharp transcodes to GGML Q2_0 at load and wraps in its own
signed Hadamard transforms. It needs a single-device GGML backend (`cpu`, `cuda`,
`mlx` and `--tp` are refused) and takes images through a companion projector.
Validation covers functional smokes on Metal (M5 Pro) plus a short PQ2 check on
the GGML CPU backend; CUDA, Vulkan and iOS were not validated. A CUDA-only
Hadamard defect was fixed afterwards and a CUDA comparison harness exists, but no
CUDA results are committed. See the [Bonsai2 card](models/bonsai2.md).

<a id="release-notes-behaviour-changes-since-the-last-tag"></a>

### Release notes: compatibility and recent changes

The following changes span several release tags. LTX 2.5, EmbeddingGemma 2, native skill discovery and the latest MiniMax-H3 work are newer than `v2026.10.11`, as described above. Knowledge collections and active-turn steering are included in that tag; the entries below retain compatibility and measured-performance history from earlier work.

- **Knowledge collections are now part of TensorAgent and opt-in server requests.** Select collections explicitly for each chat/request, and reindex metadata-only media after loading a compatible model. Imports retain originals; replacement and reindexing invalidate old versioned citations. See [personal knowledge](knowledge-system.md).
- **Text conversations can be steered while generating.** TensorAgent, standalone browser chat and the interactive CLI preserve accepted guidance and a partial response before restarting generation. Completed tool side effects remain. See [usage](../USAGE.md).
- **Qwen-Image-2.1 Turbo and a broader TensorAgent catalog.** The Turbo configuration and catalog declare its variant and use its eight-step defaults; family search and memory filtering help browse the expanded catalog. See the [image model card](models/qwenimage21.md#qwen-image-21-turbo) and [TensorAgent catalog](../TensorAgent/README.md).

- **Qwen 3.8 Flash Next runs larger than memory on Apple Silicon.** On `ggml_metal` its
  28.8 GB n-gram table is never uploaded or copied (a token's 16 rows are read on demand
  from the GGUF mapping), and the engine plans how many of the first layers' routed experts
  run on the host from that mapping (33 of 48 on a 48 GB M5 Pro; `--n-cpu-moe N` /
  `--cpu-moe` override the plan), decodes them on TensorSharp's own host kernel
  (`TS_HOST_MOE_DECODE=0` restores the ggml graph path) and, for a prefill of 128 tokens or
  more, streams the used ones to the GPU after faulting their pages in on 16 threads.
  Against llama.cpp on that Mac it prefills real text 3.5-5x faster and decodes it on par,
  and on random tokens prefills about 3x faster and decodes 6% behind. See the
  [Qwen 3.8 Flash Next card](models/qwen38-flash-next.md#larger-than-memory).
- **TensorAgent offers Qwen3.8 Flash Next on 48 GB Macs.** The text-only UD-Q2_K_XL entry
  (78.9 GB in three shards) is the catalog's first split GGUF and its first entry larger
  than the memory it is offered on: 18.3 GB stays resident and 60.5 GB, the n-gram table
  and 33 layers' experts, is read from the SSD on demand. It counts as installed only when
  every shard is complete, and the app refuses to load an incomplete split
  ("Qwen3.8 Flash Next is not completely downloaded yet.") instead of letting the engine
  fail on a file the user never saw.
  See [the Mac's own models](../TensorAgent/README.md#the-macs-own-models).
- **The direct `cuda` engine runs Qwen 3.8 Flash Next's UD-Q2_K_XL.** It used to refuse the
  file's IQ2_XS/IQ3_XXS experts at load ("unsupported expert quant type 17"); it now
  decodes them with kernels ported from ggml's dot products, in a decode step still
  captured as a CUDA graph, and its tensor-core and register-staged grouped expert kernels
  take both types for prefill. On 2x A40 (`--layer-split 2`) its prefill went from about
  500 tok/s to 1,612 (warm), and on real text it decodes at 95-96% of llama.cpp's speed.
  Multi-GPU for this quant is `--layer-split N`: `--tp N` still refuses it (exit code 2). See
  the [card](models/qwen38-flash-next.md#larger-than-memory).
- **Streamed-expert fixes on CUDA and Metal.** On CUDA, a copy of offloaded experts that
  spanned two page-locked registrations crashed (`cudaMemcpyAsync` rejects it) and is now
  split at the boundary, and unloading a model unregisters its ranges, so a second model
  loaded in the same process no longer crashes on stale ones. On Metal with async compute,
  a streamed batch's result was only queued for download when the seam read it, so the seam
  could pass on the previous offloaded layer's expert output; it now drains the queue before
  reading.
- **Qwen 3.8 Flash Next turns a `q8_0` / `q4_0` KV cache request into `f16`, and refuses
  `--backend mlx` up front.** Its span reads only F16/F32 K/V, so the request is changed
  at load; the MLX refusal names what MLX has no kernels for (the sparse-attention indexer,
  the hyper-connections, the n-gram table and the IQ2_XS/IQ3_XXS experts) and points to
  `ggml_metal`.

- **MiniMax-H3 refuses a photo, or a photo or clip reference, without `tokenizer_config.json`.**
  Image, first-and-last-frame and picture reference conditioning need it beside `vocab.json` and
  `merges.txt` (`MiniMaxAI/MiniMax-H3`, `processor/`); it alone defines `<|vision_start|>`,
  `<|image_pad|>` and `<|vision_end|>`. Without it those were split into ordinary pieces,
  the vision features landed on the wrong prompt positions and the clip quietly ignored
  the picture; such a request is now refused before the vision tower runs, naming the
  file. Text-to-video never needed it. See the [MiniMax-H3 card](models/minimax-h3.md).
- **On Metal, MiniMax-H3 hands back the kept networks before the next clip's text
  encoder.** The denoiser and both VAEs stayed device-resident between clips, so from the
  second clip on the text encoder ran with about 16 GB of finished networks still wired.
  Measured in the TensorAgent Mac app on an M5 Pro (two identical clips from words per
  arm), releasing them cut peak wired memory from 33.3 GB to 19.1 GB for 1.3 s more text
  conditioning on the second clip (2.8 s against 1.5 s), with the clip totals about 1%
  apart. `TS_H3_KEEP_RESIDENT=1` keeps the old behaviour; discrete GPUs are unchanged.
- **MiniMax-H3's soundtrack follows the 24 fps clip whatever fps is requested.** Frames
  are always written at 24 fps, but the audio timeline was sized from the requested rate
  (fps 16 on 39 frames asked for 98 audio latents instead of 65), which laid the
  soundtrack out on a different timeline from the frames it is trimmed to. A request at
  24 fps, the default, is unchanged.
- **MP4 soundtrack export reports whether audio is muxed.** TensorAgent's Apple provider writes MiniMax-H3's soundtrack as AAC (2 channels, 32 kHz, 128 kbit/s) and retains the WAV sidecar. The desktop CLI/server provider now also muxes when FFmpeg is available; its OpenCV/OS fallback writes video plus WAV and reports `audioMuxed: false`. `/api/video-generate` and the final SSE event expose the actual mux result. LTX similarly requires a mux-capable codec; its audio is 48 kHz.
- **TensorAgent offers MiniMax-H3 on the Mac.** Two 32 GB entries, "MiniMax-H3"
  (keyframes) and "MiniMax-H3 References", make a clip with its soundtrack from an
  ordinary chat turn. Along the way the Mac app stopped aborting on quit after GPU work (a
  `GGML_ASSERT` in ggml-metal's static destructor), the app began reading a transparent PNG
  without a colour profile as stored, where (nearly) transparent pixels used to come out
  black, a file another installed entry holds is now hard-linked instead of downloaded
  again, and the loopback server answers Range and HEAD.
  See [the Mac's own models](../TensorAgent/README.md#the-macs-own-models).

- **TensorAgent builds for the Mac and Windows as well as the iPhone.** The app's
  project now targets `net10.0-ios;net10.0-maccatalyst` (and
  `net10.0-windows10.0.19041.0` on a Windows machine), so building the iOS app also
  needs the `maui-maccatalyst` workload, and the build switch `TensorSharpIosTargets`
  is renamed `TensorSharpAppleTargets` (it now also builds `TensorSharp.Models`'
  Mac Catalyst slice). The app's launcher icon and splash draw their "TA" as outlines
  instead of SVG text, because MAUI 10.0.110's image tooling throws on any `<text>`.
  See [TensorAgent on the desktop](../TensorAgent/README.md#on-the-desktop-macos-and-windows).

- **Qwen-Image-2.1 edits on Metal encode the photo with the fused VAE graph.** Upstream
  ggml-metal cannot pad the start of a dimension, and the encoder's one front pad had
  refused the whole graph, so Metal edits encoded their reference image per convolution.
  The fused encode is about twice as fast (3.6 to 1.9 s for a 1248x832 reference on an
  M5 Pro) and moves an edit's pixels slightly (63 dB PSNR on that edit);
  `TS_QWEN21_VAE_FUSED=0` keeps the per-convolution path. Text-to-image is unchanged.

- **Qwen-Image: only Qwen-Image-2.1 loads.** Earlier Qwen-Image and Qwen-Image-Edit
  checkpoints such as Qwen-Image-Edit-2511 are refused at load (exit code 2).
  `--qwen-image-lora` and `--offload-cpu` are now hard errors that say what to do
  instead (`--lora` for LoRA plug-ins; a smaller `--width` / `--height` when
  memory is short), `TS_QWEN_IMAGE_LORA` is refused, and LoRA plug-ins load through
  `--lora` / `--lora-scale` / `--lora-config`. See the
  [Qwen-Image-2.1 card](models/qwenimage21.md).
- **Sub-agents are on by default on the server chat paths.** On
  `/v1/chat/completions`, `/v1/responses`, `/api/chat/ollama` and the Web UI
  `/api/chat`, a tool-capable model is offered five coordination tools
  (`spawn_agent`, `wait_agent`, `send_input`, `close_agent`, `list_agents`), even
  without skills or `--code-exec`. `--no-multi-agent`, `TS_NO_MULTI_AGENT`, or a
  request's `"multi_agent": false` turns delegation off. See
  [sub-agents](multi_agent.md).
- **Code tools: `apply_patch` edits, `write_file` creates.** `edit_file` is no
  longer advertised to the model, and neither is `write_file`'s `overwrite`
  option, so `write_file` refuses a path that already exists and points the model
  to `apply_patch` (an `overwrite: true` that a model sends anyway is still
  honoured).
- **DeepSeek V4.1: Engram comes from the GGUF.** The tokenizer-derived
  `deepseek41.engram.bin` sidecar, and `eng/dsv41-prepare.py` that wrote it, are
  gone. TensorSharp reads the Engram hash constants embedded in the
  vcruz305 release (revision `58d8ac86298fdf85a2440defee08b1abcad32e45`) and
  refuses a GGUF that lacks them ("Missing or invalid DeepSeek V4.1 GGUF
  metadata"). The `v2026.09.01` recipe used revision
  `8e0c4de3cb6519bfc11ed69dc87184b457a57bb5` plus the sidecar, so an older
  download has to be upgraded. For new downloads, the repaired ten-shard
  Q2_K/Q5_K checkpoint supersedes the original seven-shard Q2_K recommendation. See the
  [V4.1 card](models/deepseek41.md).
- **Qwen 3.8 Flash Next tool calls are parsed.** `qwen4exp` now returns
  structured tool calls, so it is offered skills, code tools and sub-agents
  instead of receiving selected skill bodies inline. See the
  [Qwen 3.8 Flash Next card](models/qwen38-flash-next.md#tool-calling-and-agent-workflows).
- **Shipped model configs download missing files.** The Gemma 4, Qwen 3.5 / 3.6,
  Qwen 3.8-27B agent, GPT OSS and DiffusionGemma configs under `config/` now give
  each model file as a download entry with a Hugging Face URL and a SHA-256, and
  every download entry in `config/*.json` pins a full commit. A single-valued
  option set on the command line (or in a later `--config` file) drops the file's
  entry before it is resolved, so passing your own `--model`, or `--mmproj none`,
  skips that download, with one `[config]` line on stderr. `--skills-dir` follows
  the same precedence: command-line roots replace all configured roots, and a
  later config file's roots replace those from earlier files; repeated roots or
  an array within the winning source are preserved. Other repeatable options such
  as `--stop` and `--lora` add to the file's values instead. On the
  server this also makes a command-line `--gpu-device` or `--kv-cache-dtype` win
  over the file, which it did not before. See the
  [config guide](../config/README.md#auto-download).
- **CLI spellings match the server.** `--penalty-last-n` is removed: the CLI's
  penalty window is `--repeat-last-n`, the server's and the `repeat_last_n` request
  field's spelling, so a config's `repeat-last-n` key (which the CLI used to drop)
  now applies there, and `--penalty-last-n` on either host or as a config key exits
  1 with `Configuration error: --penalty-last-n was removed: …`. Documented CLI options now also accept any case
  and `--option=value`, a switch given a value or a value option with nothing after
  it is a configuration error, and the CLI accepts `--mmproj none`.
- **Startup warnings for flags that do nothing.** `--spec-type`, `--spec-draft` or
  `--spec-pmin` without `--spec` or `--draft-model` (and with `TS_SPEC` unset) log
  a warning that speculation stays off, on both hosts. DiffusionGemma, Wan and MiniMax-H3 print a
  configuration error for unsupported `--tp N` / `--layer-split N`, and refuse a distributed group at load
  (exit code 2).
- **Default skill roots.** Without `--skills-dir` / `TS_SKILLS_DIR`, skills are
  discovered from every existing `.agents/skills` directory from the working
  directory up to the Git root (nearest first), then from the `skills` folder
  beside the binary. On the server that folder is also the `POST /api/skills`
  upload directory, which is always scanned first — ahead of the `.agents/skills`
  roots, and kept even with `--skills-dir` / `TS_SKILLS_DIR` — so it wins a name
  clash; the CLI has no upload directory. `.sh` skill scripts run with `bash`.
- **`--spec` keeps speculating after prefix reuse on Qwen 3.6 / Qwen 3.8-27B.**
  The embedded NextN head used to decline to arm on a sequence that adopted a
  reused prefix (the next turn of a chat, a warmed or disk-restored shared
  prefix, a Radix hit), so under `--spec` speculation ran on the first turn only.
  It now restarts just its own draft cache after that gap and re-arms on every
  turn (for a solo request on the default fused-verify route), while the trunk
  keeps the reused prefix and prefill takes the normal path. `--spec` leaves the Radix cache on; `--no-prefix-cache` still turns it
  off. See [speculative decoding](speculative_decoding.md).
- **Legacy modes and their options are removed.** Setting one of these is now a
  startup error on both hosts (also as a config-file key) that says what to use
  instead:
  - the standalone paged KV cache (RAM / SSD / Redis block tiers, TurboQuant block
    codec) with its `--paged-kv*` and `--paged-bench*` options and the
    `TS_KV_PAGED_CACHE`, `TS_KV_BLOCK_SIZE`, `TS_KV_CACHE_*` and
    `TS_KV_PAGED_QUANT_BITS` variables. Nothing on the serving path used it; prompt
    reuse across requests is the radix prefix cache;
  - `--paged-batching` / `--no-paged-batching` (use `--continuous-batching` /
    `--no-continuous-batching`);
  - `--wan-vae`, `--wan-te`, `--wan-dit2` and `--video-te` (use `--video-vae`,
    `--video-text-encoder` and `--video-dit2`), and the `TS_WAN_VAE`, `TS_WAN_TE`
    and `TS_WAN_DIT2` variables: every video model reads `TS_VIDEO_VAE`,
    `TS_VIDEO_TEXT_ENCODER` and `TS_VIDEO_DIT2`;
  - the `TS_MTP_SPEC`, `TS_MTP_DRAFT`, `TS_MTP_PMIN` and `TS_MTP_DRAFT_MODEL`
    spellings of `TS_SPEC`, `TS_SPEC_DRAFT`, `TS_SPEC_PMIN` and `TS_SPEC_DRAFT_MODEL`;
  - `TS_DSV41_RETAINED_CACHE` (DeepSeek V4.1 always retains a finished
    conversation's native slot; `TS_DSV41_RETAINED_CACHE_MB` sets the budget),
    `TS_MTP_FOLD_CATCHUP` and `TS_MTP_FUSED_DRAFT`;
  - `TS_PREFIX_CACHE_MODE`: the block-hash prefix cache is gone and the radix tree
    is the only prefix cache;
  - the switches that restored the pure-C# `cpu` backend's previous kernels:
    `TS_CPU_QGEMM`, `TS_CPU_FGEMM`, `TS_CPU_SGEMM`, `TS_CPU_SIMD_ELEMENTWISE`,
    `DIFFUSION_CPU_LEGACY` (and its `_MOE` / `_PROJ` / `_ATTN` / `_ROUTER` forms),
    `TS_QWEN_VAE_CPU`, `TS_QWEN_TE_CPU_GEMM`, `TS_QWEN_TE_CPU_ATTN`,
    `TS_QWEN35_VENC_CPU_GEMM`, `TS_QWEN35_VENC_CPU_ATTN` and
    `TS_DIRECT_QUANT_WEIGHTS`. Hosts without AVX2 (ARM64 included) still take the
    per-row quantized matmul;
  - the per-model `TS_QWEN35_BATCHED`, `TS_GEMMA4_BATCHED`, `TS_GPTOSS_BATCHED`,
    `TS_NEMOTRON_BATCHED`, `TS_HUNYUAN_BATCHED` and `TS_QWEN35_MIGRATE` switches:
    `--no-continuous-batching` (`TS_SCHED_DISABLE_BATCHED=1`) is the one switch for
    the per-sequence path;
  - the MLX backend's switches back to earlier kernels or into measured-slower
    experiments: `TS_MLX_DISABLE_COMPILE`, the `TS_MLX_FUSED_*` family,
    `TS_MLX_GDN_BLOCKED` / `TS_MLX_GDN_NATIVE` / `TS_MLX_DISABLE_GDN_T1`,
    `TS_MLX_QWEN35_GDN_PACKED_*`, `TS_MLX_KQUANT_AFFINE`, the `TS_MLX_Q5K_*`,
    `TS_MLX_Q6K_*` and `TS_MLX_IQ*` kernel switches (the `*_MATVEC_MAX_ROWS`
    thresholds stay), `TS_MLX_PIPELINED_DECODE`, `TS_MLX_DEVICE_*`,
    `TS_MLX_BASELINE_*` and the rest listed in the startup error. MLX Q4_K / Q5_K
    weights always repack to MLX affine, which was already the default. The Gemma 4
    and Muse-Glimmer spellings of the MLX eval and materialize intervals are gone:
    every model reads `TS_MLX_EVAL_EVERY_N_LAYERS`, `TS_MLX_LOCAL_KV_MATERIALIZE_INTERVAL`
    and `TS_MLX_KV_MATERIALIZE_INTERVAL`;
  - the switches that turned a retained whole-model graph or its pieces off
    (`TS_QWEN35_FD_PERSIST`, `TS_GEMMA4_FD_PERSIST`, `TS_GPTOSS_FD_PERSIST`,
    `TS_MUSE_GLIMMER_PERSIST`, `TS_DFLASH_PERSIST`, the `*_BATCHED_ARENA` switches,
    `TS_QWEN35_METAL_*`, `TS_Q35_VERIFY_PERSIST` and friends), GPT OSS's decode and
    tensor-parallel switches (`TS_GPTOSS_MODEL_DECODE`, `TS_GPTOSS_FUSED_DECODE`,
    `TS_GPTOSS_TP_*`, `TS_GPTOSS_MLX_MOE_GQMM`, `TS_GPTOSS_PAGED_ATTN_MANAGED`), and the
    Qwen 3.5 and Gemma 4 switches that turned a default path off
    (`TS_QWEN35_FULL_DECODE`, `TS_QWEN35_FUSED_VERIFY`, `TS_QWEN35_TP_FUSED*`,
    `TS_G4_*`, `TS_GMTP_NO_*` and the rest). `TS_QWEN35_HOST_MOE_VERIFY` is now
    `TS_HOST_MOE_VERIFY`. `TS_Q35_VERIFY_SNAPSHOTS=0` stays: it is the workaround
    for the open wide-verify divergence on `ggml_cuda`;
  - two never-default experiments: Qwen 3.5's device-resident verify state
    (`TS_QWEN35_VERIFY_RESIDENT`, known wrong) and Gemma 4's batched paged
    speculative trunk (`TS_GMTP_BATCHED_TRUNK`, about half the speed of the fused
    linear verify). Speculation always runs on the model's linear cache;
  - `TS_BATCHED_N1_FAST_PATH` (a lone request always takes the model's fused
    single-sequence decode where the model supports it), `TS_FUSED_LAYER_PREFILL`
    and `TS_GEMMA4_FORCE_UNFUSED` (the fused layer prefill always runs where the
    backend supports it), `TS_PAGED_ATTN_KERNEL` (Mistral 3 and Hunyuan Dense run the
    native paged attention on the GGML backends and the managed one elsewhere) and
    `TS_STRUCTURED_STREAM_BUFFER` (`json_object` streams; only `json_schema` buffers
    the whole response);
  - `TS_RETAINED_FUSED_CACHE` and `TS_PREFIX_CHECKPOINTS`, second spellings of
    `TS_RETAINED_FUSED_CACHE_MAX=0` and `TS_PREFIX_CHECKPOINTS_MAX=0`, which turn
    end-state retention and shared-prefix checkpoints off;
  - `TS_GEMMA4_BATCHED_CAPS`: Gemma 4's token-batched decode always covers
    per-layer embeddings, shared-KV layers and a wrapped sliding window, through one
    native entry point (`TSGgml_Gemma4ModelDecodeBatched`); `TS_BATCHED_FUSED_DECODE=0`
    still decodes concurrent requests round-robin;
  - the direct `cuda` backend's switches back to an earlier kernel or off its CUDA
    graphs: `TS_CUDA_QMM_BATCHED` / `_VEC` / `_F16GEMM`, the `TS_CUDA_*_DP4A` family,
    `TS_CUDA_Q80_VEC` / `_MMQ` / `_MMQ2` / `_F16_DEQUANT`, `TS_CUDA_Q81_WARP`,
    `TS_CUDA_IQ2_VEC`, `TS_CUDA_BF16_MATVEC` (and its `TS_DSV4_BF16_MATVEC` spelling),
    `TS_CUDA_GQA_*`, `TS_CUDA_FLASH_PREFILL` / `TS_CUDA_FLASH2`,
    `TS_CUDA_GDN_PREFILL_SPLIT`, `TS_CUDA_PREFILL_GRAPH` / `TS_CUDA_DECODE_GRAPH`,
    `TS_CUDA_MOE_ONDEVICE`, `TS_CUDA_MOE_PREFILL_GROUPED`, `TS_CUDA_QWEN35_GDN_NATIVE`
    and `TS_TP_MOE_PREFILL_ONDEVICE`. The thresholds (`TS_CUDA_QMM_F16GEMM_MIN_ROWS` /
    `_MAX_MB`, `TS_CUDA_Q80_MMQ_MAX_ROWS`, `TS_CUDA_PREFILL_GRAPH_MAX`) and the
    diagnostics stay;
  - four more never-default experiments: the native batched GatedDeltaNet step
    (`TS_QWEN35_BATCHED_GDN_NATIVE`, never validated), the wmma Q8_0 GEMM
    (`TS_CUDA_Q8_MMA`, superseded by the MMQ kernel), on-device MoE prefill
    (`TS_CUDA_MOE_PREFILL_ONDEVICE`, 0.58x the grouped prefill) and Gemma 4's
    GEMM global attention (`TS_CUDA_GEMMA4_GLOBAL_GEMM_ATTN`);
  - model switches that turned a default path off: `TS_Q4E_FUSED_ATTN` / `_FFN` /
    `_GDN`, `TS_NEMOTRON_FLASH_DECODE`, `TS_NEMOTRON_LINEAR_RESIDUAL_FUSED`,
    `TS_NEMOTRON_MAMBA2_NATIVE_DECODE` / `_PREFILL`, `TS_NEMOTRON_MOE_PREFILL_BATCHED`,
    `TS_MUSE_GLIMMER_FUSED` / `_FUSED_CPU` / `_TP_FUSED` / `_VENC_FUSED`,
    `TS_DFLASH_FUSED`, `TS_DSV4_MMA_EXPERTS` / `_STAGED_EXPERTS`,
    `TS_DSV4_DSPARK_CAPTURE`, and the hyper-connection overrides `TS_DSV4_HC_NATIVE` /
    `TS_GLM_HC_NATIVE` (the backend probe decides); Muse-Glimmer's
    `TS_MUSE_GLIMMER_INGRAPH_EMBED` / `_VENC_F32` / `_GELU_TANH` overrides; two
    never-default Nemotron experiments (`TS_NEMOTRON_MOE_PREFILL_FUSED`,
    `TS_NEMOTRON_LINEAR_RESIDUAL_FUSED_PREFILL`); `TS_GLM_BATCHED_DECODE` (a second
    spelling of `TS_BATCHED_FUSED_DECODE=0`); and `TS_GLM_MTP` (the loader pages the
    NextN block in exactly when `--spec` / `TS_SPEC` will use it);
  - the placement spellings beside `--tp` / `--layer-split`: `TS_DSV41_TP` (`--tp N`
    alone enables DeepSeek V4.1's routed-MoE tensor parallelism), and `TS_DSV4_NGPU` /
    `TS_GLM_NGPU` (`--layer-split N` sets the GPU count; there is no all-visible-GPUs
    shorthand, and with no placement flag these executors use one device);
  - `TS_Q4E_RETAINED_CACHE`: Qwen 3.8 Flash Next always retains finished conversations
    and shared-prefix checkpoints, and `TS_Q4E_RETAINED_CACHE_MB` sizes that memory
    (`0` retains nothing);
  - more switches that turned a default path off or reached a concluded experiment:
    Qwen 3.8 Flash Next's `TS_Q4E_TOKEN_GRAPH`, `TS_Q4E_SPAN_ATTN`, `TS_Q4E_FLASH_ATTN`,
    `TS_Q4E_GRAPH_UID`, `TS_Q4E_SPAN_STATE`, `TS_Q4E_SPAN_REBUILD`, `TS_Q4E_SPAN_FA_MAX`,
    `TS_Q4E_GDN_MAX_LAYERS` and `TS_Q4E_RES_RESIDENT` (known wrong); GLM 5.x's
    `TS_GLM_FA`, `TS_GLM_FUSED_LID`, `TS_GLM_TOPK`, `TS_GLM_TP_FUSED` and
    `TS_GLM_VENC_FUSED` (the backend probe decides flash attention and the fused
    lightning indexer); `TS_GGML_REUSE_COMPUTE_BUF`, `TSG_USE_FLASH_ATTN_PREFILL`,
    `TS_DISABLE_FUSED_DENSE_FFN`, `TS_ENCODER_YIELD`, `TS_GEMMA4V_FUSED`,
    `TS_GGML_FUSED_NORM_ADD`, `TS_GGML_MOE_FUSED_DECODE`, `TS_QWEN_VAE_POOL` /
    `TS_QWEN_VAE_POOL_TRIM`, `TENSORSHARP_CUDA_POOL` (`TENSORSHARP_CUDA_POOL_MAX_MB` /
    `_LARGE_MB` size the cache) and the `TS_EMBEDDING_Q8_F32` experiment;
    `TS_VRAM_HEADROOM_MB=0` now means no headroom rather than switching the VRAM
    budget policy off;
  - `TS_JSON_GRAMMAR` and `TS_JSON_FORCE_OPEN`: `json_object` / `json_schema` always
    decode under the JSON grammar, and a schema the grammar cannot express still falls
    back to the first-token constraint;
  - the per-family drafter paths `TS_DSV4_DSPARK`, `TS_QWEN35_DFLASH`,
    `TS_MUSE_GLIMMER_DFLASH` and `TS_NEMOTRON_DFLASH`: `--draft-model` (or
    `TS_SPEC_DRAFT_MODEL`) names every drafter, and the server no longer copies it into
    them.
- **Rebuild the native library with the managed code.** The managed code no longer
  probes for exports an older `GgmlOps` lacked, and several native entry points were
  merged: `TSGgml_Dsv4LoadModel` now takes the DSpark drafter path and the routed-MoE
  tensor-parallel degree itself (the `Dsv4LoadModelDspark` / `Dsv4LoadModelParallel`
  variants are gone), and `TSGgml_Qwen4ExpTokenSpan` replaces its `Ex` / `Qsa`
  variants. `TSGgml_Dsv4Reset` is gone (`TSGgml_Dsv4ResetChecked` reports a reset the
  slot refused), as are the owner-0 `TSGgml_Qwen35ModelVerify`,
  `TSGgml_Qwen35CommitStateSnapshot`, `TSGgml_Qwen35FetchStateSnapshot` and
  `TSGgml_Qwen35DrainDeviceState` exports (their `*Owned` forms remain). A `GgmlOps`
  built from an earlier revision fails to load a model with a missing-entry-point error. Scripts that call the DeepSeek loader through ctypes pass
  the two new arguments.
- **Qwen 3.5 / 3.6: a request joined mid-decode keeps its recurrent state.** A request
  that was decoding alone when a second one arrived moved to the batched decode without
  the recurrent state of its last solo step, and decoded the rest of its reply 1.4-2.8
  max |Δlogit| away from decoding alone (Qwen3.5-9B, Metal): fluent text, different
  tokens. Such replies now match decoding the request alone.
- **The server reports every option mistake as a configuration error.** A malformed
  `--kv-cache-dtype`, `--spec-draft`, `--n-cpu-moe`, `--gpu-device` or
  `--prefill-chunk-size` value, a missing `--draft-model` file, or a removed
  speculation variable aborted the server with a stack trace (exit code 134). It now
  prints one `Configuration error:` line and exits 1, as the CLI already did.
- **Queue-position stream events are gone.** Streaming responses no longer carry the
  `queue_position` / `queue_pending` chunks, which a no-op queue could never emit
  anyway; `/api/queue/status` reports the engine's live load.
- **Ollama's deprecated `/api/embeddings` is no longer served** (it answers 404).
  Embedding servers keep OpenAI `/v1/embeddings` and Ollama `/api/embed`, which
  takes `input` (a string or an array) where the old route took one `prompt`.

Already in `v2026.09.01`; relevant when upgrading from `v3.4.0.0` or earlier:

- **Radix prefix cache on by default.** Prompt reuse goes through the engine's
  Radix tree; `--no-prefix-cache` / `TS_SCHED_PREFIX_CACHE=0` turns reuse off.
  The earlier block-hash sharing has since been removed (see above).
- **Qwen 3.5/3.6: outputs after an image change.** Tokens after an image are now
  generated at Qwen-VL's compressed M-RoPE positions (KV index plus the per-sequence
  rope delta, as HF and SGLang do) instead of at the absolute KV index, so replies to
  image prompts differ from earlier builds and match a correct implementation. Follow-up
  turns after an image reuse the cache again (98% of the prompt, 0.13 s to the first token
  on Metal instead of about 1.1 s). Saved Qwen 3.5 shared-prefix checkpoint files move to
  format version 2; a version-1 file is ignored with a warning and rewritten. The native
  library must be rebuilt with the managed code. Details in the
  [Qwen 3.5 card](models/qwen35.md#positions-after-an-image-the-m-rope-delta).

<a id="tensoragent-and-ios"></a>

### TensorAgent: mobile and desktop

TensorAgent is a .NET MAUI application for iOS/iPadOS, macOS and Windows that runs the TensorSharp engine locally. It links the native GGML library as an iOS `.xcframework`, uses `ggml_metal` on physical devices, and shares the host-neutral chat pipeline (`TensorSharp.Chat`) with the CLI and server. The Apple targets (the iOS app and the Mac app) are enabled with `TensorSharpAppleTargets=true`; it is not a separate numerical backend or a remote inference service.

The app includes on-device model downloads, saved conversations, attachments, dictation, Agent Skills, bounded in-process agent tools, and an "Ask TensorAgent" share extension that turns text, links, web pages, images, movies, audio, PDFs, and documents shared from other apps into an unsent chat draft. iOS does not support ASP.NET Core runtime hosting or child processes, so TensorAgent uses an in-process loopback server and runtime-backed shell/Python/JavaScript integrations. What it shares with the desktop is the API, not the page: the app ships its own phone-shaped UI bound to the same `WebUiChatService` and `SkillsService` routes.

Three phone-specific constraints shape the current implementation:

- **The screen goes away mid-answer.** Generation is owned by a host-side manager rather than the WebView, because WebKit suspends a page whose view has left the window. A turn keeps generating while the user moves between the app's own screens, and it survives an app switch: iOS forbids background GPU work, so generation pauses while the app is not frontmost and resumes when it returns, and the page re-attaches to the turn.
- **The first message of a launch would otherwise pay for the whole shared prefix.** The model state at the end of the prompt every chat shares is checkpointed to disk per model (`IPrefixCheckpointStore` / `PrefixCheckpointFileStore`) and restored at admission, so a launch's first message costs a restore rather than a full prefill.
- **iOS jetsam kills without a stack or a message.** `EngineMemoryPolicy` sizes what the engine holds against what jetsam actually charges — wired file pages count against the device, not the process — and a memory warning releases what only serves the next request's speed.

Other current behaviour:

- **Devices.** iPhone and iPad, arm64 only, iOS 17.0 or later. The app adopts the UIKit scene lifecycle with a single window, which fixed a launch crash when it is built with the iOS 27 SDK; the recorded physical-device run is on iOS 26.6.1.
- **Speculative decoding is on by default** in the app, unlike the desktop hosts: the Gemma 4 E4B and 12B entries use their draft head when it has been downloaded, and otherwise the app selects n-gram speculation. A change to the setting applies to the running engine from the next reply; a `TS_SPEC` / `TS_SPEC_TYPE` set in the launch environment still wins.
- **Sub-agents have a switch.** Tool-capable chats are offered the delegation tools while Settings > Sandbox > "Sub-agents" is on (the default, with the host's default limits); off withholds the tools and the coordination prompt from the next message, as `--no-multi-agent` does on the server. The phone page shows no agent panel, and no on-device measurements are published.
- **Desktop hosting.** When `TensorAgent.Core` runs on macOS, Linux, or Windows, `AgentExecutionMode.Auto` runs code as native processes under the OS sandbox instead of the embedded interpreters, and refuses a network-enabled launch while a `networkHosts` allow-list is set, because a process sandbox cannot enforce it.
- **The desktop app.** The same project builds TensorAgent for the Mac (Mac Catalyst) and for Windows (WinUI, with bounded Debug/Release validation on Windows x64 on 2026-10-01). The Mac app ships the desktop engine library, runs the model's code as Seatbelt-confined processes, keeps the engine's desktop memory defaults, and holds off App Nap while the model works. Measured on an M5 Pro with Gemma 4 E2B, its decode rate is within 3% of the same host code on CoreCLR, and a turn's first token comes about 60 ms later, because .NET runs Mac Catalyst apps on Mono.
- **Windows coverage.** The 2026-10-01 Debug/Release check used an i7-11800H, 32 GiB RAM, RTX 3080 Laptop GPU (16 GiB) and .NET SDK 10.0.204 against unchanged upstream ggml `353b63b439f27ab2cc19dac97ab1681ba6d2d084`. It covered WebView uploads, saved/recovered conversations, cancellation, PowerShell, Python artifacts and synthetic vision with Gemma 4 12B QAT UD-Q4_K_XL on CUDA. The managed suite passed 895 tests with 177 explicit skips; the skipped scenarios are not passes. Catalog downloads, native dialogs/camera, audio/video generation, other models/backends and a long soak were not verified. The [app guide](../TensorAgent/README.md#on-windows) records the detailed checks and measurements.
- **CI.** PR CI checks the app's project, Info.plist, entitlements, share extension, and native export manifest through `InferenceWeb.Tests`; it does not run `TensorAgent.Tests` or build MAUI app heads. The release workflow packages desktop apps, and the separate on-demand `knowledge-pdf-portable.yml` exercises managed OCR on Windows/Linux. Workflow configuration is not evidence of a successful run; unavailable models and devices remain explicit validation gaps.

See the [TensorAgent README](../TensorAgent/README.md) for build, simulator, device, packaging, and test instructions, including the measured numbers behind those three points.

## Status matrix

Where every area actually stands, with the per-architecture exceptions the
README's summary leaves out.

| Area | Status |
|---|---|
| Personal knowledge | Local originals and versioned sources; TensorAgent file/folder/chat-snapshot import, previews and per-chat selection; opt-in standalone `/api/knowledge` and scoped inference requests. BM25, optional embeddings and bounded model ranking; explicit media/PDF/Office extraction limits. See the [guide](knowledge-system.md). |
| Active-turn steering | Text generation in TensorAgent, standalone browser chat and the interactive CLI accepts new user guidance, preserving partial output before resuming; completed tool actions remain. See [usage](../USAGE.md). |
| Embedding models | BERT/XLM-R sentence encoders plus EmbeddingGemma 2 text/code and optional media projector; 8192-token window and 768/512/256/128 dimensions for EmbeddingGemma 2. Six direct/native CPU/MLX/CUDA paths, dedicated `--embeddings` service, OpenAI/Ollama APIs. Numerical and performance limits: [guide](embeddings.md). |
| Model families | DeepSeek V4 Flash (`deepseek4`), DeepSeek V4.1 Flash (`deepseek41`), GLM 5.x (`glm-dsa`, `glm_dsa`, `glm5next`), Gemma 4, DiffusionGemma, Qwen 3.5/3.6-family (`qwen35`, `qwen35moe`, `qwen3next`), Qwen 3.8 Flash Next (`qwen4exp`), Bonsai2 27B (`qwen35` files carrying PRISM `prism.hadamard.*` metadata and PQ2_0 / PTQ1_0 tensors), GPT OSS, Nemotron-H (incl. Nemotron 3 Nano Omni and Nemotron 3.5 Lightning; `nemotron_h`, `nemotron_h_moe`, `nemotron_h_omni`), Mistral 3 (`mistral3`, plus `llama`-labelled Mistral Small 3.x files), Hunyuan Dense (`hunyuan-dense`), Muse-Glimmer (`muse-glimmer`, `muse_glimmer`). Text-to-image and image editing via Qwen-Image-2.1 (`qwen_image`, `qwen-image`); joint video-and-audio generation via MiniMax-H3 (`minimax-h3`, `minimax_h3`) and experimental LTX 2.5 Distilled (`ltx-2.5` and aliases) and video-only generation via Wan 2.1 / 2.2 (`wan`, `wan2.1`, `wan2.2`). |
| Inference hosts | CLI, interactive REPL, ASP.NET Core web UI, Ollama-style API, OpenAI Chat Completions-style API, OpenAI Responses-style API, and the Jev typed-decision API (`POST /v1/systemone`, served from a loaded DiffusionGemma model, with text, images, uploaded documents, sampled video frames and configured ASR transcripts as state; see [Jev](models/jev.md)). |
| TensorAgent app | TensorAgent targets iOS/iPadOS, and the same project builds a Mac app (Mac Catalyst: the desktop engine library, Seatbelt-confined processes, measured) and a Windows app (bounded Debug/Release chat, synthetic vision and tool checks on Windows x64; image/audio/video generation unverified). On the phone it links GGML as an iOS `.xcframework`, and uses `ggml_metal` on physical devices. It shares the host-neutral chat pipeline (`TensorSharp.Chat`) but serves its own phone-shaped page from an in-process loopback host, because iOS has neither an ASP.NET Core runtime pack nor child processes. It runs on iPhone and iPad (arm64, iOS 17.0 or later). A turn survives the app leaving the screen (generation pauses while the app is not frontmost and resumes when it returns), speculative decoding and sub-agent delegation are on by default, the shared-prompt prefix checkpoint is persisted per model so the first message of a launch costs a restore instead of a full prefill (measured on iPhone 17 Pro Max with Qwen3.5 9B: a 54 s cold first message becomes a 1.2 s warm-up and a ~0.6 s first message), and the engine's memory policy is sized against what iOS jetsam actually charges. See [TensorAgent](../TensorAgent/README.md). |
| Backends | Pure C# CPU, direct CUDA/cuBLAS (`cuda`), MLX Metal (`mlx`), GGML CPU, GGML Metal, GGML CUDA, GGML Vulkan. DeepSeek V4 additionally has three whole-model executors of its own — direct-CUDA, native ggml, and a pure-C# CPU one — with GPU executors placing whole layers through `--layer-split N` (`--layer-split N` selects the count). DeepSeek V4.1 serves on `ggml_cuda`; `ggml_cpu` runs the same native graph on scalar fallbacks and `cpu` runs a pure-C# V4.1 executor, both as correctness and portability paths rather than serving ones. `cuda` runs V4.1 through the direct-CUDA engine's own kernels (no ggml), and has targeted synthetic kernel/slot tests rather than a full-checkpoint numerical gate. `ggml_vulkan` / `ggml_metal` need `TS_DSV41_ALLOW_NON_CUDA_GPU=1`; `mlx` refuses the checkpoint rather than loading V4.1 weights into a graph that does not implement it. Wan runs on the GGML backends and direct `cuda` / pure-C# `cpu`, but not MLX. MiniMax-H3 has six native/direct CPU, CUDA and MLX routes. LTX maps `ggml_cuda` / `cuda` to its owned CUDA path, `ggml_metal` / `mlx` to MLX, and `ggml_cpu` / `cpu` to direct CPU; it refuses Vulkan and multi-GPU. Qwen-Image-2.1 runs on the GGML backends and a complete pure-C# `cpu` pipeline (single process; no tensor parallelism), and Bonsai2 needs a single-device GGML backend. Qwen 3.8 Flash Next refuses `mlx` up front, naming `ggml_metal` instead. |
| Release builds & CI | Tagged releases build self-contained CLI/server archives for Windows x64 (CPU/CUDA), Linux x64 (CPU/CUDA), and macOS arm64, and package TensorAgent Mac (DMG/PKG/ZIP) and Windows x64 CPU/CUDA (MSI/ZIP). The experimental `linux-arm64-cuda13-GB10` archive builds on a hosted ARM64 runner without a GPU; historical GB10 hardware numbers do not certify the current code ([build guide](../DEVELOPMENT.md#gb10--dgx-spark-build-container-experimental)). PR CI runs `InferenceWeb.Tests` CPU correctness, the GB10 container gate, and an engine-comparison smoke on self-hosted CUDA; it does not run `TensorAgent.Tests` or build MAUI heads. A separate on-demand workflow checks managed PDF OCR on Windows/Linux. NuGet packaging follows `eng/verify-packages.ps1`, which now includes `TensorSharp.Knowledge`. Configured workflows are not passing validation; assets and packages depend on their specific tags. |
| Multimodal | Gemma 4 image/video/audio; Qwen 3.5-family (including Bonsai2 with its companion projector), Qwen 3.8 Flash Next, GLM-5.3-Flash, Mistral 3, Nemotron-H Omni, Muse-Glimmer, DiffusionGemma image input; Qwen 3.8 Flash Next `video_url` video; DeepSeek V4.1 image and video through its vision companion; Nemotron-H audio only with a companion GGUF carrying the Parakeet tower (the public GGUFs ship none); DiffusionGemma refuses audio and `video_url` video (a Web UI video upload reaches it only as plain image frames); PDF documents (CLI `--pdf` + Web UI). Media *out*: Qwen-Image-2.1 (text-to-image, reference-image and masked local editing, with LoRA plug-ins via `--lora` and a default-on prefix KV cache for the text and reference-image tokens), MiniMax-H3 (H.264 MP4 **plus a 32 kHz stereo `.wav` sidecar**, generated together in one packed latent; MP4 soundtrack muxing uses FFmpeg on desktop or AVFoundation in the Apple app, with `audioMuxed` reporting the result), LTX 2.5 Distilled (text/image keyframes and optional Ingredients identity references → video plus 48 kHz stereo audio; experimental), and Wan 2.1 / 2.2 (H.264 MP4 video only, text→video and image→video). |
| Continuous batching | vLLM-style paged KV cache, a Radix prefix cache that is on by default (`--no-prefix-cache` / `TS_SCHED_PREFIX_CACHE=0` turns off all reuse; `--spec` leaves it on) on the Qwen 3.5-family, Gemma 4, GLM 5.x, Qwen 3.8 Flash Next, DeepSeek V4 / V4.1, GPT OSS, Mistral 3, Hunyuan Dense, Muse-Glimmer and Nemotron-H but not DiffusionGemma or the media models, shared-prefix checkpoints (the state at the end of the prompt every conversation shares is cloned into each new chat, so a new chat re-prefills only its own message; Gemma 4, Qwen 3.5/3.6 and Qwen 3.8 Flash Next on GGML, and a host can persist a Gemma 4 or Qwen 3.5-family one across process restarts via `IPrefixCheckpointStore`), iteration-level scheduler (default on; opt-out `--no-continuous-batching`). The paged pool is host-resident, so it buys memory efficiency and prefix reuse rather than throughput that scales with concurrency. DeepSeek V4 / V4.1 and GLM 5.x serve through their own native per-sequence slots on the same engine — a compressed MLA cache row per token has no paged layout to page — and GLM adds a default-on batched fused decode (set `TS_BATCHED_FUSED_DECODE=0` to use serial fused decode; 1.81x aggregate at 4 concurrent requests). Qwen 3.8 Flash Next uses per-sequence state holders for the same reason — its GatedDeltaNet, PLE and indexer state has no paged layout either. |
| Speculative decoding | MTP / NextN draft heads on Qwen 3.6, Qwen 3.8-27B, GLM 5.2 and GLM-5.3 (all embedded in the checkpoint — GLM-5.3's `blk.78` NextN block is complete but ships no LM head of its own, so speculation engages on single-device or explicit `--layer-split N` placement, without active tensor parallelism), Gemma 4 (separate draft GGUF, loaded via `--draft-model`) and Qwen 3.8 Flash Next (a separate shared MTP head GGUF via `--draft-model`, GGML backends only); DSpark block drafting on DeepSeek V4 (`cuda` / `ggml_cuda` only), an experimental `deepseek41-dspark` drafter on DeepSeek V4.1 (`ggml_cuda` / `ggml_cpu` only; initial text/image HTTP probes with trained weights passed using two-GPU layer split on `ggml_cuda`; broad quality and throughput remain unqualified), DFlash / DFlash2 block drafting on Muse-Glimmer and Qwen 3.8-27B (Nemotron-H refuses speculation: its verify and decode kernels disagree, so a speculative stream would differ from plain decoding) — all loading a separate drafter GGUF via `--draft-model`; plus a weight-free n-gram (prompt-lookup) speculator that needs no drafter, selected with `--spec-type ngram`, on the Qwen 3.5 family, Gemma 4, GLM 5.x and Qwen 3.8 Flash Next. It does not run on GPT OSS, Mistral 3 or Hunyuan Dense, which have no speculative path, nor on DeepSeek V4 / V4.1 or Muse-Glimmer without their drafter. Every emitted token is drawn from a trunk row with the run's own sampler, so the emitted stream is the one plain decoding would have produced. Off by default on the CLI and server (TensorAgent turns it on); opt in with `--spec` on either host for the embedded heads, while passing `--draft-model` enables speculation by itself for any drafter shipping as its own GGUF. |
| Tensor parallelism and layer split | **Tensor parallelism, layer split & distributed inference** — `--tp N` / `TENSORSHARP_TP_DEGREE=N` selects tensor parallelism only: it shards weights inside layers across N local GPUs. `--layer-split N` / `TENSORSHARP_LAYER_SPLIT_DEGREE=N` selects whole-layer placement on Qwen 3.8 Flash Next, DeepSeek V4 / V4.1, and GLM 5.x. The options are mutually exclusive; unsupported requests fail at startup instead of changing modes or using one GPU. Whole-layer placement is local to one node and mainly increases capacity. Supported tensor-parallel architectures can use `--tp-node-id` / `--tp-peers` for multi-node inference; GLM, Qwen 3.8 Flash Next and Qwen-Image TP remain local-only. Qwen 3.8 Flash Next shards routed/shared FFNs on `ggml_cuda` and replicates attention/GDN/PLE; its UD-Q2_K_XL quant requires layer split instead. TP measurements do not establish a speedup; see its [card](models/qwen38-flash-next.md#multi-gpu). The default is one device. DeepSeek V4.1's experimental routed-MoE TP (`--tp N`) does not implement attention TP or multi-node execution. |
| Agent Skills | 17 bundled packs; 15 statically admitted on iOS. Native `find-skills` search/inspect/install with network permission and same-request immutable inspection; install requires a writable registry. Marketplace model samples passed 3/4 full tasks, not blanket compatibility. See [guide](agent_skills.md). Skill directories come from `--skills-dir` / `TS_SKILLS_DIR`; without either, from every existing `.agents/skills` directory from the working directory up to the Git root (nearest first), then the `skills` folder beside the binary. On the server that folder is also the `POST /api/skills` runtime-install directory: it is always scanned first (ahead of the `.agents/skills` roots, and kept even with `--skills-dir` / `TS_SKILLS_DIR`), so it wins a name clash; the CLI has no install directory. Selected per request with `"skills": [...]` on `/v1/chat/completions`, `/v1/responses`, `/api/chat/ollama` (Ollama) and `/api/chat` (Web UI), or with `--skill` on the CLI. Tool-capable families, including Qwen 3.8 Flash Next (`qwen4exp`), receive metadata and activate instructions through built-in `skills_list` / `skills_read` calls answered in process; the caller's own tools are still returned to it. Script execution (`skills_run`) is off unless `--skills-allow-exec` is passed; `.sh` scripts run with `bash`. Mistral 3 and families without a parsable tool protocol instead receive selected skill bodies inline and are not offered skill/code tools. The Playwright browser skill (`TensorAgent/skills/playwright`) drives a browser through `skills_run` rather than a screen or mouse tool; it needs Node.js/npm (the host's install, or one the agent installs into the session's `$HOME/.local/bin` with network access) plus the explicit script, network and install opt-ins, has model-driven workflow validation on macOS arm64 and a separate Windows native-runtime probe for synthetic login/session and headed-window behavior; the Windows probe does not establish model-driven task completion. The Playwright skill is left out of TensorAgent's iOS app bundle, which cannot run it ([guide](playwright_agent.md)). |
| Agentic code work | Optional `--code-exec` offers `read_file`, atomic `apply_patch` (every change to an existing file), `write_file` for new files (it refuses an existing path unless the model sends the undeclared `overwrite: true`), and `shell` inside the same bounded model-to-tool loop. Web/CLI chats keep a session workspace; each OpenAI/Ollama request gets a private workspace across internal repair rounds and loses it after the response. Artifacts are captured for download. The loop runs in process on the one loaded model, while on the CLI and server shell commands and skill scripts run as child processes (confinement is in the Sandboxing row); sub-agents (next row) are the only delegation, and there is no per-command approval workflow. |
| Sub-agents | Default-on, bounded, model-selected delegation on `/v1/chat/completions`, `/v1/responses`, `/api/chat/ollama` and the Web UI `/api/chat`, and in TensorAgent (a "Sub-agents" switch in Settings, on by default; no agent panel on the phone page); the CLI has none. Eligible families are those that render tool declarations and have a tool parser — not Mistral 3, Hunyuan Dense or DiffusionGemma — with no model-size gate. Children run on the same loaded model and are read-only by default (`explorer` and `reviewer` always; `worker` gets the parent's mutable tools only with `--agents-allow-worker-tools`); host tool calls within one request tree run one at a time, and each child copies rather than shares KV state, restoring the parent's checkpointed shared prefix. `--no-multi-agent`, `TS_NO_MULTI_AGENT` or a request's `"multi_agent": false` turns it off, and `--agents-max-*` / `--agents-timeout` bound it. No latency, quality or delegation-rate measurements are published. [Guide](multi_agent.md). |
| Sandboxing & permissions | Code execution and skill scripts are off by default. macOS uses Seatbelt and Linux requires `bwrap` 0.12.0+; required mode refuses when confinement is unavailable. Windows code execution requires explicit `--code-exec-unconfined`, while Windows skill scripts require explicit `--skills-sandbox preferred` to accept job-object-only containment. Script network, code network, and host-performed package installation are separate opt-ins. |
| Server model scope | One explicitly hosted GGUF via `--model`; optional explicit projector via `--mmproj`; no directory scanning. |
| Observability | Structured per-turn logs, queue status, and KV-cache reuse metrics across Web UI, Ollama, and OpenAI shapes. |

## Make It Fast

The short version is:

1. Pick a step-distilled checkpoint for Wan. For eight-step Qwen-Image generation, use the explicitly declared [Turbo variant](models/qwenimage21.md#qwen-image-21-turbo). For base Qwen-Image-2.1 drafts, use `--width 1024 --height 1024` or `--diffusion-steps 25`, or a step-distillation LoRA plug-in from `config/lora/` (4–8 steps).
2. Match the backend to the hardware: `ggml_cuda` for NVIDIA, `ggml_metal` for Apple Silicon and iOS, and `ggml_cpu` for native CPU.
3. Reduce resolution or frame count before changing advanced flags. MiniMax-H3 requires `--cfg 1.0` and defaults to 20 steps; fewer base steps trade quality for speed and are not a distilled recipe. For 8 or 4 steps on FL2VA, add a LightX2V [Turbo plug-in](models/minimax-h3.md#turbo-plug-ins) (`--lora config/lora/minimax-h3-fl2v-turbo-8step.json`), which fixes the step count to its recipe. LTX keeps its fixed two-stage schedule; lowering output size also lowers the first pass and can hurt face/hand detail.
4. For text workloads, try speculative decoding, CPU MoE offload, or `--tp N` only when the model and workload benefit.

For measurements and caveats, read the [engine comparison report](engine_comparison_report.md), [ggml_metal vs llama.cpp on Apple Silicon](perf/metal-vs-llama-cpp.md), [model cards](models/README.md), [feature guide](../FEATURES.md), and [environment-variable matrix](env_var_feature_matrix.md).

## Where details live

- [Getting started](../README.md#quick-start) — first run and backend selection.
- [Compute backends](../USAGE.md#compute-backends) — capabilities, build requirements, and fallbacks.
- [Agent Skills and agentic work](agent_skills.md) — skills, tools, workspaces, and security.
- [Sub-agents](multi_agent.md) — delegation tools, roles, limits, and KV prefix sharing.
- [Personal knowledge](knowledge-system.md) — collections, chat imports, extraction coverage, citations, HTTP API and validation limits.
- [TensorAgent](../TensorAgent/README.md) — iOS application architecture and verification.
- [Development guide](../DEVELOPMENT.md) — project layering and native builds.
