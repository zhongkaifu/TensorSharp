# TensorSharp project status

This page keeps repository-level status and longer explanations that do not belong in the README.

## Current direction

TensorSharp is a native .NET 10 inference engine for GGUF models. The current source includes CLI, server/Web UI, compatible HTTP APIs, AgentHost, and the TensorAgent iOS/iPadOS application. AgentHost ships inside the CLI and server archives and as the `TensorSharp.AgentHost` NuGet package; TensorAgent is source-only, because no release workflow builds the iOS app. Work merged after the `v2026.09.01` tag — among it Qwen-Image-2.1, Bonsai2, DiffusionGemma image input and the Jev API, sub-agents, the Playwright browser skill and the GB10 release archive — exists only in source builds until the next tag; the post-tag changes that alter an existing setup are listed under [release notes](#release-notes-behaviour-changes-since-the-last-tag).

### Embedding models and serving

Current source adds GGUF BERT/XLM-R sentence encoders, covering Snowflake Arctic Embed L v2.0 Q8_0 and all-MiniLM-L6-v2 Q8_0. `--embeddings` starts a dedicated resident encoder with OpenAI `/v1/embeddings`, Ollama `/api/embed`, and legacy `/api/embeddings`. Backends are 100% pure C# CPU (`cpu`) and native GGML CPU (`ggml_cpu`), Metal, and CUDA; see the [embedding guide](embeddings.md) for validation scope and reproducible comparisons. Older release archives may not include this feature.

### Newest architectures

The two newest architecture families, both in the `v2026.09.01` tag, carry
limits worth knowing before you plan around them.

- **DeepSeek V4.1 Flash (`deepseek41`)** — a dedicated native V4.1 graph with an
  optional vision companion. `ggml_cuda` is the serving backend; `ggml_cpu`
  runs the same graph on scalar fallbacks and `cpu` runs a pure-C# V4.1 executor,
  both as correctness and portability paths. `cuda` runs V4.1 through the
  direct-CUDA engine's own kernels and is not yet held to a numerical gate.
  `ggml_vulkan` and `ggml_metal` need `TS_DSV41_ALLOW_NON_CUDA_GPU=1`; `mlx`
  refuses the checkpoint. The current vcruz305 GGUF release includes Engram
  weights and hash constants; TensorSharp reads them directly, without Engram
  generation or a separate Engram file (a change made after the tag; see the
  [release notes](#release-notes-behaviour-changes-since-the-last-tag)). Historical Q2_K and Q4_K_M tests are
  recorded separately from validation of the current files; at Q4_K_M the
  two Engram tables are 51.5 GiB each. On 8x46 GB they stay host mappings, and
  the checkpoint needs routed-expert CPU offload (see the
  [Q4_K_M throughput report](perf/deepseek41-q4km-throughput.md); the
  quantization report, `docs/validation/deepseek41-quants/README.md`, is local
  validation evidence, not committed).
  Multi-GPU means a layer split; routed-MoE tensor parallelism exists behind
  `TS_DSV41_TP` and has measured slower than the split. Concurrent requests get
  isolated slots but fall back to per-slot forwards, so concurrency is not
  batched GPU throughput yet. V4.1 DSpark is experimental: `--draft-model`
  loads a `deepseek41-dspark` drafter on `ggml_cuda` or `ggml_cpu` only, it is
  validated only on synthetic fixtures, and no trained drafter has been
  measured. A multi-turn chat
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
default layer split rather than under `--tp` because the `blk.78` NextN block
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

### Release notes: behaviour changes since the last tag

Changes merged after the `v2026.09.01` tag that alter what an existing setup does:

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
  gone. TensorSharp reads the Engram hash constants embedded in the current
  vcruz305 release (revision `58d8ac86298fdf85a2440defee08b1abcad32e45`) and
  refuses a GGUF that lacks them ("Missing or invalid DeepSeek V4.1 GGUF
  metadata"). The `v2026.09.01` recipe used revision
  `8e0c4de3cb6519bfc11ed69dc87184b457a57bb5` plus the sidecar, so an older
  download has to be upgraded to the current release. See the
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
  skips that download, with one `[config]` line on stderr; repeatable options such
  as `--stop`, `--lora` and `--skills-dir` add to the file's values instead. On the
  server this also makes a command-line `--gpu-device` or `--kv-cache-dtype` win
  over the file, which it did not before. See the
  [config guide](../config/README.md#auto-download).
- **CLI spellings match the server.** `--penalty-last-n` is removed: the CLI's
  penalty window is `--repeat-last-n`, the server's and the `repeat_last_n` request
  field's spelling, so a config's `repeat-last-n` key (which the CLI used to drop)
  now applies there, and `--penalty-last-n` on either host or as a config key exits
  1 with `Configuration error: --penalty-last-n was removed: …`. The aliases
  `--paged-kv-cache` / `--no-paged-kv-cache` are removed the same way (use
  `--paged-kv` / `--no-paged-kv`). Documented CLI options now also accept any case
  and `--option=value`, a switch given a value or a value option with nothing after
  it is a configuration error, and the CLI accepts `--mmproj none`.
- **Startup warnings for flags that do nothing.** `--spec-type`, `--spec-draft` or
  `--spec-pmin` without `--spec` or `--draft-model` (and with `TS_SPEC` unset) log
  a warning that speculation stays off, on both hosts. On the server, the
  standalone paged-KV flags (`--paged-kv*`, `--paged-kv-redis-*`) log one warning
  that they have no effect there. DiffusionGemma, Wan and MiniMax-H3 print a
  warning for `--tp N` and run on one GPU, and refuse a distributed group at load
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

Already in `v2026.09.01`; relevant when upgrading from `v3.4.0.0` or earlier:

- **Radix prefix cache on by default.** Prompt reuse goes through the engine's
  Radix tree (`TS_PREFIX_CACHE_MODE=tree`); `TS_PREFIX_CACHE_MODE=legacy`
  restores the earlier block-hash sharing, and `--no-prefix-cache` /
  `TS_SCHED_PREFIX_CACHE=0` turns reuse off.
- **Qwen 3.5/3.6: outputs after an image change.** Tokens after an image are now
  generated at Qwen-VL's compressed M-RoPE positions (KV index plus the per-sequence
  rope delta, as HF and SGLang do) instead of at the absolute KV index, so replies to
  image prompts differ from earlier builds and match a correct implementation. Follow-up
  turns after an image reuse the cache again (98% of the prompt, 0.13 s to the first token
  on Metal instead of about 1.1 s). Saved Qwen 3.5 shared-prefix checkpoint files move to
  format version 2; a version-1 file is ignored with a warning and rewritten. The native
  library must be rebuilt with the managed code. Details in the
  [Qwen 3.5 card](models/qwen35.md#positions-after-an-image-the-m-rope-delta).

### TensorAgent and iOS

TensorAgent is a .NET MAUI iOS/iPadOS application that runs the TensorSharp engine locally. It links the native GGML library as an iOS `.xcframework`, uses `ggml_metal` on physical devices, and shares the host-neutral chat pipeline (`TensorSharp.Chat`) with the CLI and server. The iOS target is enabled with `TensorSharpIosTargets=true`; it is not a separate numerical backend or a remote inference service.

The app includes on-device model downloads, saved conversations, attachments, dictation, Agent Skills, bounded in-process agent tools, and an "Ask TensorAgent" share extension that turns text, links, web pages, images, movies, audio, PDFs, and documents shared from other apps into an unsent chat draft. iOS does not support ASP.NET Core runtime hosting or child processes, so TensorAgent uses an in-process loopback server and runtime-backed shell/Python/JavaScript integrations. What it shares with the desktop is the API, not the page: the app ships its own phone-shaped UI bound to the same `WebUiChatService` and `SkillsService` routes.

Three phone-specific constraints shape the current implementation:

- **The screen goes away mid-answer.** Generation is owned by a host-side manager rather than the WebView, because WebKit suspends a page whose view has left the window. A turn keeps generating while the user moves between the app's own screens, and it survives an app switch: iOS forbids background GPU work, so generation pauses while the app is not frontmost and resumes when it returns, and the page re-attaches to the turn.
- **The first message of a launch would otherwise pay for the whole shared prefix.** The model state at the end of the prompt every chat shares is checkpointed to disk per model (`IPrefixCheckpointStore` / `PrefixCheckpointFileStore`) and restored at admission, so a launch's first message costs a restore rather than a full prefill.
- **iOS jetsam kills without a stack or a message.** `EngineMemoryPolicy` sizes what the engine holds against what jetsam actually charges — wired file pages count against the device, not the process — and a memory warning releases what only serves the next request's speed.

Other current behaviour:

- **Devices.** iPhone and iPad, arm64 only, iOS 17.0 or later. The app adopts the UIKit scene lifecycle with a single window, which fixed a launch crash when it is built with the iOS 27 SDK; the recorded physical-device run is on iOS 26.6.1.
- **Speculative decoding is on by default** in the app, unlike the desktop hosts: the Gemma 4 E4B and 12B entries use their draft head when it has been downloaded, and otherwise the app selects n-gram speculation (Bonsai 8B, a `qwen3` model, has no speculative path). A change to the setting applies to the running engine from the next reply; a `TS_SPEC` / `TS_SPEC_TYPE` set in the launch environment still wins.
- **Sub-agents have a switch.** Tool-capable chats are offered the delegation tools while Settings > Sandbox > "Sub-agents" is on (the default, with the host's default limits); off withholds the tools and the coordination prompt from the next message, as `--no-multi-agent` does on the server. The phone page shows no agent panel, and no on-device measurements are published.
- **Desktop hosting.** When `TensorAgent.Core` runs on macOS, Linux, or Windows, `AgentExecutionMode.Auto` runs code as native processes under the OS sandbox instead of the embedded interpreters, and refuses a network-enabled launch while a `networkHosts` allow-list is set, because a process sandbox cannot enforce it.
- **CI.** PR CI checks the app's project, Info.plist, entitlements, share extension, and native export manifest through `InferenceWeb.Tests`, but it does not run `TensorAgent.Tests` and no workflow builds the iOS app.

See the [TensorAgent README](../TensorAgent/README.md) for build, simulator, device, packaging, and test instructions, including the measured numbers behind those three points.

## Status matrix

Where every area actually stands, with the per-architecture exceptions the
README's summary leaves out.

| Area | Status |
|---|---|
| Embedding models | GGUF BERT/XLM-R: Snowflake Arctic Embed L v2.0 and all-MiniLM-L6-v2; pure C# CPU and native GGML CPU/Metal/CUDA, dedicated `--embeddings` service, OpenAI/Ollama single and batched APIs. Context, tokenization, quality, and measurement scope: [guide](embeddings.md). |
| Model families | DeepSeek V4 Flash (`deepseek4`), DeepSeek V4.1 Flash (`deepseek41`), GLM 5.x (`glm-dsa`, `glm_dsa`, `glm5next`), Gemma 4, DiffusionGemma, Qwen 3.5/3.6-family (`qwen35`, `qwen35moe`, `qwen3next`), Qwen 3.8 Flash Next (`qwen4exp`), the Qwen 3 / Qwen 2 plug-in (`qwen3`, `qwen2`, `qwen2vl`, `qwen2_vl`; text only), Bonsai Q1_0 (8B on `qwen3`, 27B on `qwen35`) and Bonsai2 27B (`qwen35` files carrying PRISM `prism.hadamard.*` metadata and PQ2_0 / PTQ1_0 tensors), GPT OSS, Nemotron-H (incl. Nemotron 3 Nano Omni and Nemotron 3.5 Lightning; `nemotron_h`, `nemotron_h_moe`, `nemotron_h_omni`), Mistral 3 (`mistral3`, plus `llama`-labelled Mistral Small 3.x files), Hunyuan Dense (`hunyuan-dense`), Muse-Glimmer (`muse-glimmer`, `muse_glimmer`). Text-to-image and image editing via Qwen-Image-2.1 (`qwen_image`, `qwen-image`); joint video-and-audio generation via MiniMax-H3 (`minimax-h3`, `minimax_h3`) and video-only generation via Wan 2.1 / 2.2 (`wan`, `wan2.1`, `wan2.2`). |
| Inference hosts | CLI, interactive REPL, ASP.NET Core web UI, Ollama-style API, OpenAI Chat Completions-style API, OpenAI Responses-style API, and the Jev typed-decision API (`POST /v1/systemone`, served from a loaded DiffusionGemma model, with text, images, uploaded documents, sampled video frames and configured ASR transcripts as state; see [Jev](models/jev.md)). |
| iOS application | TensorAgent targets iOS/iPadOS, links GGML as an iOS `.xcframework`, and uses `ggml_metal` on physical devices. It shares the host-neutral chat pipeline (`TensorSharp.Chat`) but serves its own phone-shaped page from an in-process loopback host, because iOS has neither an ASP.NET Core runtime pack nor child processes. It runs on iPhone and iPad (arm64, iOS 17.0 or later). A turn survives the app leaving the screen (generation pauses while the app is not frontmost and resumes when it returns), speculative decoding and sub-agent delegation are on by default, the shared-prompt prefix checkpoint is persisted per model so the first message of a launch costs a restore instead of a full prefill (measured on iPhone 17 Pro Max with Qwen3.5 9B: a 54 s cold first message becomes a 1.2 s warm-up and a ~0.6 s first message), and the engine's memory policy is sized against what iOS jetsam actually charges. See [TensorAgent](../TensorAgent/README.md). |
| Backends | Pure C# CPU, direct CUDA/cuBLAS (`cuda`), MLX Metal (`mlx`), GGML CPU, GGML Metal, GGML CUDA, GGML Vulkan. DeepSeek V4 additionally has three whole-model executors of its own — direct-CUDA, native ggml, and a pure-C# CPU one — each layer-splitting the weights across every visible GPU (`--tp N` / `TS_DSV4_NGPU` caps the count). DeepSeek V4.1 serves on `ggml_cuda`; `ggml_cpu` runs the same native graph on scalar fallbacks and `cpu` runs a pure-C# V4.1 executor, both as correctness and portability paths rather than serving ones. `cuda` runs V4.1 through the direct-CUDA engine's own kernels (no ggml), and is not yet held to a numerical gate. `ggml_vulkan` / `ggml_metal` need `TS_DSV41_ALLOW_NON_CUDA_GPU=1`; `mlx` refuses the checkpoint rather than loading V4.1 weights into a graph that does not implement it. Among the video families, Wan is the one that restricts its backends: it runs on the GGML backends and on the direct `cuda` / pure-C# `cpu` ones, but not on MLX. Qwen-Image-2.1 runs only on the GGML backends, and Bonsai2 needs a single-device GGML backend. |
| Release builds & CI | Tagged releases build self-contained CLI and server archives for Windows x64 (CPU/CUDA), Linux x64 (CPU/CUDA), and macOS arm64. After `v2026.09.01`, the release workflow gained an experimental `linux-arm64-cuda13-GB10` archive for NVIDIA GB10 / DGX Spark (CUDA 13, SM121a), built in Docker on a hosted ARM64 runner without a GPU; the next tag will be the first release to carry it. The recorded GB10 hardware smoke numbers are historical and predate the upstream reintegration, so they do not certify the current code ([GB10 build](../DEVELOPMENT.md#gb10--dgx-spark-build-container-experimental)). PR CI runs the `InferenceWeb.Tests` CPU correctness lane and the GB10 container gate check on x64 and ARM64 Linux; pull requests to `main` also run an engine-comparison smoke (Gemma 4 12B, TensorSharp against llama.cpp, `test-matrix.yml`) on a self-hosted CUDA runner. PR CI does not run `TensorAgent.Tests` or build the iOS app. Tagged releases also publish the NuGet packages listed in `eng/verify-packages.ps1`, including `TensorSharp.AgentHost`. |
| Multimodal | Gemma 4 image/video/audio; Qwen 3.5-family (including Bonsai2 with its companion projector), Qwen 3.8 Flash Next, GLM-5.3-Flash, Mistral 3, Nemotron-H Omni, Muse-Glimmer, DiffusionGemma image input; Qwen 3.8 Flash Next `video_url` video; DeepSeek V4.1 image and video through its vision companion; Nemotron-H audio only with a companion GGUF carrying the Parakeet tower (the public GGUFs ship none); DiffusionGemma refuses audio and `video_url` video (a Web UI video upload reaches it only as plain image frames); PDF documents (CLI `--pdf` + Web UI). Media *out*: Qwen-Image-2.1 (image, with LoRA plug-ins via `--lora` and a default-on prefix KV cache for the text and reference-image tokens), MiniMax-H3 (H.264 MP4 **plus a 32 kHz stereo `.wav` sidecar**, generated together in one packed latent), and Wan 2.1 / 2.2 (H.264 MP4 video only, text→video and image→video). |
| Continuous batching | vLLM-style paged KV cache, a Radix prefix cache that is on by default (`TS_PREFIX_CACHE_MODE=tree`; `legacy` selects the older block-hash prefix sharing, and `--no-prefix-cache` / `TS_SCHED_PREFIX_CACHE=0` turns off all reuse; `--spec` leaves it on) on the Qwen 3.5-family, Gemma 4, GLM 5.x, Qwen 3.8 Flash Next, DeepSeek V4 / V4.1, Qwen 3, GPT OSS, Mistral 3, Hunyuan Dense, Muse-Glimmer and Nemotron-H but not DiffusionGemma or the media models, shared-prefix checkpoints (the state at the end of the prompt every conversation shares is cloned into each new chat, so a new chat re-prefills only its own message; Gemma 4, Qwen 3.5/3.6 and Qwen 3.8 Flash Next on GGML, and a host can persist a Gemma 4 or Qwen 3.5-family one across process restarts via `IPrefixCheckpointStore`), iteration-level scheduler (default on; opt-out `--no-continuous-batching`). The paged pool is host-resident, so it buys memory efficiency and prefix reuse rather than throughput that scales with concurrency. DeepSeek V4 and GLM 5.x serve through their own native per-sequence slots on the same engine — a compressed MLA cache row per token has no paged layout to page — and GLM adds a default-on batched fused decode (set `TS_BATCHED_FUSED_DECODE=0` to use serial fused decode; 1.81x aggregate at 4 concurrent requests). Qwen 3.8 Flash Next uses per-sequence state holders for the same reason — its GatedDeltaNet, PLE and indexer state has no paged layout either. |
| Speculative decoding | MTP / NextN draft heads on Qwen 3.6, Qwen 3.8-27B, GLM 5.2 and GLM-5.3 (all embedded in the checkpoint — GLM-5.3's `blk.78` NextN block is complete but ships no LM head of its own, so speculation engages on the default layer split, without `--tp`), Gemma 4 (separate draft GGUF, loaded via `--draft-model`) and Qwen 3.8 Flash Next (a separate shared MTP head GGUF via `--draft-model`, GGML backends only); DSpark block drafting on DeepSeek V4 (`cuda` / `ggml_cuda` only), an experimental `deepseek41-dspark` drafter on DeepSeek V4.1 (`ggml_cuda` / `ggml_cpu` only, validated only on synthetic fixtures, no trained drafter measured), DFlash / DFlash2 block drafting on Muse-Glimmer and Qwen 3.8-27B (Nemotron-H refuses speculation: its verify and decode kernels disagree, so a speculative stream would differ from plain decoding) — all loading a separate drafter GGUF via `--draft-model`; plus a weight-free n-gram (prompt-lookup) speculator that needs no drafter, selected with `--spec-type ngram`, on the Qwen 3.5 family, Gemma 4, GLM 5.x and Qwen 3.8 Flash Next. It does not run on GPT OSS, Mistral 3, Qwen 3 / Qwen 2 (including Bonsai 8B) or Hunyuan Dense, which have no speculative path, nor on DeepSeek V4 / V4.1 or Muse-Glimmer without their drafter. Every emitted token is drawn from a trunk row with the run's own sampler, so the emitted stream is the one plain decoding would have produced. Off by default on the CLI and server (TensorAgent turns it on); opt in with `--spec` on either host for the embedded heads, while passing `--draft-model` enables speculation by itself for any drafter shipping as its own GGUF. |
| Tensor parallelism | Megatron-LM column/row-parallel TP on the direct `cuda` backend and on GGML CUDA / Vulkan (`--tp N` / `TENSORSHARP_TP_DEGREE`, CLI and server); distributed multi-node TP via peer-to-peer TCP (`--tp-node-id` / `--tp-peers`), with hierarchical AllReduce and automatic host-staging fallback when CUDA P2P is unavailable. Most autoregressive architectures (Hunyuan Dense is single-device, Bonsai2 refuses `--tp` because its PRISM transforms need a single-device GGML backend, and DiffusionGemma, Wan and MiniMax-H3 load on one device and warn that `--tp` is ignored); MoE expert parallelism and fused per-rank decode/prefill graphs for Gemma 4 and Qwen 3.5/3.6 on GGML. GLM 5.x uses a layer split by default, but `--tp N` selects its native local/single-process TP path on GGML GPU backends for GLM 5.2, GLM-5.3 and GLM-5.3-Flash (including KDA/MLA head and routed-expert-row sharding on GLM-5.3-Flash); the whole GLM family hard-refuses `--tp-node-id` / `--tp-peers` before the model is built, and on GLM-5.3 `--tp N` is an accepted mode rather than a validated one — it replicates the MLA and indexer caches per rank, so the footprint multiplies and the fitted context shrinks, and nothing above `--tp 1` has been run on that checkpoint (`--tp 8` works out to 41.7 GiB per rank, which does not fit a 46 GB card). Architectures that shard no weights take the same `--tp N` as a layer split — a contiguous run of whole layers per GPU, as with DeepSeek V4 and V4.1 (on V4.1, `TS_DSV41_TP=N` additionally enables experimental routed-MoE tensor parallelism over those same GPUs, which has measured slower than the layer split so far); on Qwen 3.8 Flash Next (`qwen4exp`) `TS_Q4E_LAYER_SPLIT=20,28` overrides the automatic balance and throws rather than ignoring a split it cannot honour. Startup prints which mode ran and the per-GPU layer/byte split, and an architecture that supports neither mode says so on stderr and runs on one GPU. Qwen-Image-2.1 shards only its diffusion transformer with `--tp` on `ggml_cuda` / `ggml_vulkan` (a GPU count that divides its 32 heads, one machine only); the text encoder, vision encoder and VAE stay on GPU 0, and Vulkan TP has measured slower than one GPU. `--redis-url` backs the Responses API store; the server accepts the Redis KV tier (the KV half of `--redis-url`, `--paged-kv-redis-url` / `--paged-kv-redis-ttl`) and the `--paged-kv*` flags but only logs them, because the tiered RAM/SSD/Redis `PagedKvCacheManager` is used only by the CLI `--paged-bench`. |
| Agent Skills | Skill directories come from `--skills-dir` / `TS_SKILLS_DIR`; without either, from every existing `.agents/skills` directory from the working directory up to the Git root (nearest first), then the `skills` folder beside the binary. On the server that folder is also the `POST /api/skills` runtime-install directory: it is always scanned first (ahead of the `.agents/skills` roots, and kept even with `--skills-dir` / `TS_SKILLS_DIR`), so it wins a name clash; the CLI has no install directory. Selected per request with `"skills": [...]` on `/v1/chat/completions`, `/v1/responses`, `/api/chat/ollama` (Ollama) and `/api/chat` (Web UI), or with `--skill` on the CLI. Tool-capable families, including Qwen 3.8 Flash Next (`qwen4exp`), receive metadata and activate instructions through built-in `skills_list` / `skills_read` calls answered in process; the caller's own tools are still returned to it. Script execution (`skills_run`) is off unless `--skills-allow-exec` is passed; `.sh` scripts run with `bash`. Mistral 3 and families without a parsable tool protocol instead receive selected skill bodies inline and are not offered skill/code tools. The Playwright browser skill (`TensorAgent/skills/playwright`) drives a browser through `skills_run` rather than a screen or mouse tool; it needs Node.js/npm (the host's install, or one the agent installs into the session's `$HOME/.local/bin` with network access) plus the explicit script, network and install opt-ins, is validated on macOS arm64 only, and is left out of TensorAgent's iOS app bundle, which cannot run it ([guide](playwright_agent.md)). |
| Agentic code work | Optional `--code-exec` offers `read_file`, atomic `apply_patch` (every change to an existing file), `write_file` for new files (it refuses an existing path unless the model sends the undeclared `overwrite: true`), and `shell` inside the same bounded model-to-tool loop. Web/CLI chats keep a session workspace; each OpenAI/Ollama request gets a private workspace across internal repair rounds and loses it after the response. Artifacts are captured for download. The loop runs in process on the one loaded model, while on the CLI and server shell commands and skill scripts run as child processes (confinement is in the Sandboxing row); sub-agents (next row) are the only delegation, and there is no per-command approval workflow. |
| Sub-agents | Default-on, bounded, model-selected delegation on `/v1/chat/completions`, `/v1/responses`, `/api/chat/ollama` and the Web UI `/api/chat`, and in TensorAgent (a "Sub-agents" switch in Settings, on by default; no agent panel on the phone page); the CLI has none. Eligible families are those that render tool declarations and have a tool parser — not Mistral 3, Hunyuan Dense or DiffusionGemma — with no model-size gate. Children run on the same loaded model and are read-only by default (`explorer` and `reviewer` always; `worker` gets the parent's mutable tools only with `--agents-allow-worker-tools`); host tool calls within one request tree run one at a time, and each child copies rather than shares KV state, restoring the parent's checkpointed shared prefix. `--no-multi-agent`, `TS_NO_MULTI_AGENT` or a request's `"multi_agent": false` turns it off, and `--agents-max-*` / `--agents-timeout` bound it. No latency, quality or delegation-rate measurements are published. [Guide](multi_agent.md). |
| Sandboxing & permissions | Code execution and skill scripts are off by default. macOS uses Seatbelt and Linux requires `bwrap` 0.12.0+; required mode refuses when confinement is unavailable. Windows code execution requires explicit `--code-exec-unconfined`, while Windows skill scripts require explicit `--skills-sandbox preferred` to accept job-object-only containment. Script network, code network, and host-performed package installation are separate opt-ins. |
| Server model scope | One explicitly hosted GGUF via `--model`; optional explicit projector via `--mmproj`; no directory scanning. |
| Observability | Structured per-turn logs, queue status, and KV-cache reuse metrics across Web UI, Ollama, and OpenAI shapes. |

## Make It Fast

The short version is:

1. Pick a step-distilled checkpoint for Wan. For Qwen-Image-2.1 drafts, use `--width 1024 --height 1024` or `--diffusion-steps 25`, or a step-distillation LoRA plug-in from `config/lora/` (4–8 steps).
2. Match the backend to the hardware: `ggml_cuda` for NVIDIA, `ggml_metal` for Apple Silicon and iOS, and `ggml_cpu` for native CPU.
3. Reduce resolution, frame count, or diffusion steps before changing advanced flags. MiniMax-H3's practical point is `--cfg 1.0` with 4–8 steps.
4. For text workloads, try speculative decoding, CPU MoE offload, or `--tp N` only when the model and workload benefit.

For measurements and caveats, read the [engine comparison report](engine_comparison_report.md), [ggml_metal vs llama.cpp on Apple Silicon](perf/metal-vs-llama-cpp.md), [model cards](models/README.md), [feature guide](../FEATURES.md), and [environment-variable matrix](env_var_feature_matrix.md).

## Where details live

- [Getting started](../README.md#quick-start) — first run and backend selection.
- [Compute backends](../USAGE.md#compute-backends) — capabilities, build requirements, and fallbacks.
- [Agent Skills and agentic work](agent_skills.md) — skills, tools, workspaces, and security.
- [Sub-agents](multi_agent.md) — delegation tools, roles, limits, and KV prefix sharing.
- [TensorAgent](../TensorAgent/README.md) — iOS application architecture and verification.
- [Development guide](../DEVELOPMENT.md) — project layering and native builds.
