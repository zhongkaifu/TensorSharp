# TensorSharp project status

This page keeps repository-level status and longer explanations that do not belong in the README.

## Current direction

TensorSharp is a native .NET 10 inference engine for GGUF models. The current source includes CLI, server/Web UI, compatible HTTP APIs, AgentHost, and the TensorAgent iOS/iPadOS application. AgentHost and TensorAgent are source-first capabilities: the latest tagged release may not contain them yet.

### Newest architectures

Two families landed after the last release tag, and both carry limits worth
knowing before you plan around them.

- **DeepSeek V4.1 Flash (`deepseek41`)** — a dedicated native V4.1 graph with an
  optional vision companion. `ggml_cuda` is the serving backend; `ggml_cpu`
  runs the same graph on scalar fallbacks and `cpu` runs a pure-C# V4.1 executor,
  both as correctness and portability paths. `cuda` runs V4.1 through the
  direct-CUDA engine's own kernels and is not yet held to a numerical gate.
  `ggml_vulkan` and `ggml_metal` need `TS_DSV41_ALLOW_NON_CUDA_GPU=1`; `mlx`
  refuses the checkpoint. Every release needs a prepared Engram sidecar before it
  will run at all — the community GGUF repositories do not ship one, so generate
  it with `eng/dsv41-prepare.py`. Q2_K and Q4_K_M are both tested; at Q4_K_M the
  two Engram tables are 51.5 GiB each and stay host mappings, so the checkpoint
  needs routed-expert CPU offload on 8x46 GB (see the
  [quantization report](validation/deepseek41-quants/README.md)).
  Multi-GPU means a layer split; routed-MoE tensor parallelism exists behind
  `TS_DSV41_TP` and has measured slower than the split. Concurrent requests get
  isolated slots but fall back to per-slot forwards, so concurrency is not
  batched GPU throughput yet, and there is no V4.1 DSpark. A multi-turn chat
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
(41.9 s vs 29.0 s); see the [cross-engine report](validation/cross-engine-2026-09/README.md)
beside the [GLM card](models/glm.md#glm-53-glm-dsa).

### TensorAgent and iOS

TensorAgent is a .NET MAUI iOS/iPadOS application that runs the TensorSharp engine locally. It links the native GGML library as an iOS `.xcframework`, uses `ggml_metal` on physical devices, and shares the host-neutral chat pipeline (`TensorSharp.Chat`) with the CLI and server. The iOS target is enabled with `TensorSharpIosTargets=true`; it is not a separate numerical backend or a remote inference service.

The app includes on-device model downloads, saved conversations, attachments, dictation, Agent Skills, and bounded in-process agent tools. iOS does not support ASP.NET Core runtime hosting or child processes, so TensorAgent uses an in-process loopback server and runtime-backed shell/Python/JavaScript integrations. What it shares with the desktop is the API, not the page: the app ships its own phone-shaped UI bound to the same `WebUiChatService` and `SkillsService` routes.

Three phone-specific constraints shape the current implementation:

- **The screen goes away mid-answer.** Generation is owned by a host-side manager rather than the WebView, because WebKit suspends a page whose view has left the window. A turn continues across an app switch and the page re-attaches to it.
- **The first message of a launch would otherwise pay for the whole shared prefix.** The model state at the end of the prompt every chat shares is checkpointed to disk per model (`IPrefixCheckpointStore` / `PrefixCheckpointFileStore`) and restored at admission, so a launch's first message costs a restore rather than a full prefill.
- **iOS jetsam kills without a stack or a message.** `EngineMemoryPolicy` sizes what the engine holds against what jetsam actually charges — wired file pages count against the device, not the process — and a memory warning releases what only serves the next request's speed.

See the [TensorAgent README](../TensorAgent/README.md) for build, simulator, device, packaging, and test instructions, including the measured numbers behind those three points.

## Status matrix

Where every area actually stands, with the per-architecture exceptions the
README's summary leaves out.

| Area | Status |
|---|---|
| Model families | DeepSeek V4 Flash (`deepseek4`), DeepSeek V4.1 Flash (`deepseek41`), GLM 5.x (`glm-dsa`, `glm5next`), Gemma 4, DiffusionGemma, Qwen 3.5/3.6-family (`qwen35`, `qwen35moe`, `qwen3next`), Qwen 3.8 Flash Next (`qwen4exp`), GPT OSS, Nemotron-H (incl. Nemotron 3 Nano Omni and Nemotron 3.5 Lightning, `nemotron_h_moe`), Mistral 3, Hunyuan Dense (`hunyuan-dense`), Muse-Glimmer (`muse-glimmer`, `muse_glimmer`). Image editing via Qwen-Image-Edit (`qwen_image`, `qwen-image` MMDiT); joint video-and-audio generation via MiniMax-H3 (`minimax-h3`, `minimax_h3`) and video-only generation via Wan 2.1 / 2.2 (`wan`, `wan2.1`, `wan2.2`). |
| Inference hosts | CLI, interactive REPL, ASP.NET Core web UI, Ollama-style API, OpenAI Chat Completions-style API, and OpenAI Responses-style API. |
| iOS application | TensorAgent targets iOS/iPadOS, links GGML as an iOS `.xcframework`, and uses `ggml_metal` on physical devices. It shares the host-neutral chat pipeline (`TensorSharp.Chat`) but serves its own phone-shaped page from an in-process loopback host, because iOS has neither an ASP.NET Core runtime pack nor child processes. Generation survives the app leaving the screen, the shared-prompt prefix checkpoint is persisted per model so the first message of a launch costs a restore instead of a full prefill (measured on iPhone 17 Pro Max with Qwen3.5 9B: a 54 s cold first message becomes a 1.2 s warm-up and a ~0.6 s first message), and the engine's memory policy is sized against what iOS jetsam actually charges. See [TensorAgent](../TensorAgent/README.md). |
| Backends | Pure C# CPU, direct CUDA/cuBLAS (`cuda`), MLX Metal (`mlx`), GGML CPU, GGML Metal, GGML CUDA, GGML Vulkan. DeepSeek V4 additionally has three whole-model executors of its own — direct-CUDA, native ggml, and a pure-C# CPU one — each layer-splitting the weights across every visible GPU (`--tp N` / `TS_DSV4_NGPU` caps the count). DeepSeek V4.1 serves on `ggml_cuda`; `ggml_cpu` runs the same native graph on scalar fallbacks and `cpu` runs a pure-C# V4.1 executor, both as correctness and portability paths rather than serving ones. `cuda` runs V4.1 through the direct-CUDA engine's own kernels (no ggml), and is not yet held to a numerical gate. `ggml_vulkan` / `ggml_metal` need `TS_DSV41_ALLOW_NON_CUDA_GPU=1`; `mlx` refuses the checkpoint rather than loading V4.1 weights into a graph that does not implement it. Among the video families, Wan is the one that restricts its backends: it runs on the GGML backends and on the direct `cuda` / pure-C# `cpu` ones, but not on MLX. |
| Multimodal | Gemma 4 image/video/audio; Qwen 3.5-family, Qwen 3.8 Flash Next, GLM-5.3-Flash, Mistral 3, Nemotron-H Omni, Muse-Glimmer image input; PDF documents (CLI `--pdf` + Web UI). Media *out*: Qwen-Image-Edit (image), MiniMax-H3 (H.264 MP4 **plus a 32 kHz stereo `.wav` sidecar**, generated together in one packed latent), and Wan 2.1 / 2.2 (H.264 MP4 video only, text→video and image→video). |
| Continuous batching | vLLM-style paged KV cache, block-hash prefix sharing, shared-prefix checkpoints (the state at the end of the prompt every conversation shares is cloned into each new chat, so a new chat re-prefills only its own message; Gemma 4 and Qwen 3.5/3.6 on GGML, and a host can persist one across process restarts via `IPrefixCheckpointStore`), iteration-level scheduler (default on; opt-out `--no-continuous-batching`). The paged pool is host-resident, so it buys memory efficiency and prefix reuse rather than throughput that scales with concurrency. DeepSeek V4 and GLM 5.x serve through their own native per-sequence slots on the same engine — a compressed MLA cache row per token has no paged layout to page — and GLM adds a default-on batched fused decode (set `TS_BATCHED_FUSED_DECODE=0` to use serial fused decode; 1.81x aggregate at 4 concurrent requests). Qwen 3.8 Flash Next uses per-sequence state holders for the same reason — its GatedDeltaNet, PLE and indexer state has no paged layout either. |
| Speculative decoding | MTP / NextN draft heads on Qwen 3.6, GLM 5.2 and GLM-5.3 (all embedded in the checkpoint — GLM-5.3's `blk.78` NextN block is complete but ships no LM head of its own, so speculation engages on the default layer split, without `--tp`) and Gemma 4 (separate draft GGUF, loaded via `--draft-model`); DSpark block drafting on DeepSeek V4 (`cuda` / `ggml_cuda` only), DFlash / DFlash2 block drafting on Muse-Glimmer and Qwen 3.8, and DSpark Markov-head block drafting on Nemotron 3.5 Lightning — the first recurrent (Mamba-2) trunk to take a block drafter — all loading a separate drafter GGUF via `--draft-model`; plus a weight-free n-gram (prompt-lookup) speculator that needs no drafter at all and therefore works on every checkpoint, selected with `--spec-type ngram`. Every emitted token is drawn from a trunk row with the run's own sampler, so the emitted stream is the one plain decoding would have produced. Off by default; opt in with `--spec` on either host for the embedded heads, while passing `--draft-model` enables speculation by itself for any drafter shipping as its own GGUF. |
| Tensor parallelism | Megatron-LM column/row-parallel TP on the direct `cuda` backend and on GGML CUDA / Vulkan (`--tp N` / `TENSORSHARP_TP_DEGREE`, CLI and server); distributed multi-node TP via peer-to-peer TCP (`--tp-node-id` / `--tp-peers`), with hierarchical AllReduce and automatic host-staging fallback when CUDA P2P is unavailable. All autoregressive architectures; MoE expert parallelism and fused per-rank decode/prefill graphs for Gemma 4 and Qwen 3.5/3.6 on GGML. GLM 5.x uses a layer split by default, but `--tp N` selects its native local/single-process TP path on GGML GPU backends for GLM 5.2, GLM-5.3 and GLM-5.3-Flash (including KDA/MLA head and routed-expert-row sharding on GLM-5.3-Flash); the whole GLM family hard-refuses `--tp-node-id` / `--tp-peers` before the model is built, and on GLM-5.3 `--tp N` is an accepted mode rather than a validated one — it replicates the MLA and indexer caches per rank, so the footprint multiplies and the fitted context shrinks, and nothing above `--tp 1` has been run on that checkpoint (`--tp 8` works out to 41.7 GiB per rank, which does not fit a 46 GB card). Architectures that shard no weights take the same `--tp N` as a layer split — a contiguous run of whole layers per GPU, as with DeepSeek V4 and V4.1 (on V4.1, `TS_DSV41_TP=N` additionally enables experimental routed-MoE tensor parallelism over those same GPUs, which has measured slower than the layer split so far); on Qwen 3.8 Flash Next (`qwen4exp`) `TS_Q4E_LAYER_SPLIT=20,28` overrides the automatic balance and throws rather than ignoring a split it cannot honour. Startup prints which mode ran and the per-GPU layer/byte split, and an architecture that supports neither mode says so on stderr and runs on one GPU. Optional Redis-backed KV cache and Responses API store. |
| Agent Skills | Skill directories discovered from `--skills-dir` (or a `skills` folder beside the binary) and installable at runtime through `POST /api/skills`. Selected per request with `"skills": [...]` on `/v1/chat/completions`, `/v1/responses`, `/api/chat/ollama` (Ollama) and `/api/chat` (Web UI), or with `--skill` on the CLI. Tool-capable families receive metadata and activate instructions through built-in `skills_list` / `skills_read` calls answered in process; the caller's own tools are still returned to it. Script execution (`skills_run`) is off unless `--skills-allow-exec` is passed. Mistral 3 and families without a parsable tool protocol, including `qwen4exp`, instead receive selected skill bodies inline and are not offered skill/code tools. |
| Agentic code work | Optional `--code-exec` offers `read_file`, `edit_file`, `write_file`, `shell`, and atomic `apply_patch` inside the same bounded model-to-tool loop. Web/CLI chats keep a session workspace; each OpenAI/Ollama request gets a private workspace across internal repair rounds and loses it after the response. Artifacts are captured for download. This is a single-model in-process loop: TensorSharp does not provide multi-agent delegation or a per-command approval workflow. |
| Sandboxing & permissions | Code execution and skill scripts are off by default. macOS uses Seatbelt and Linux requires `bwrap` 0.12.0+; required mode refuses when confinement is unavailable. Windows code execution requires explicit `--code-exec-unconfined`, while Windows skill scripts require explicit `--skills-sandbox preferred` to accept job-object-only containment. Script network, code network, and host-performed package installation are separate opt-ins. |
| Server model scope | One explicitly hosted GGUF via `--model`; optional explicit projector via `--mmproj`; no directory scanning. |
| Observability | Structured per-turn logs, queue status, and KV-cache reuse metrics across Web UI, Ollama, and OpenAI shapes. |

## Make It Fast

The short version is:

1. Pick a step-distilled checkpoint for Wan, or the Lightning LoRA for Qwen-Image-Edit.
2. Match the backend to the hardware: `ggml_cuda` for NVIDIA, `ggml_metal` for Apple Silicon and iOS, and `ggml_cpu` for native CPU.
3. Reduce resolution, frame count, or diffusion steps before changing advanced flags. MiniMax-H3's practical point is `--cfg 1.0` with 4–8 steps.
4. For text workloads, try speculative decoding, CPU MoE offload, or `--tp N` only when the model and workload benefit.

For measurements and caveats, read the [engine comparison report](engine_comparison_report.md), [ggml_metal vs llama.cpp on Apple Silicon](perf/metal-vs-llama-cpp.md), [model cards](models/README.md), [feature guide](../FEATURES.md), and [environment-variable matrix](env_var_feature_matrix.md).

## Where details live

- [Getting started](../README.md#quick-start) — first run and backend selection.
- [Compute backends](../USAGE.md#compute-backends) — capabilities, build requirements, and fallbacks.
- [Agent Skills and agentic work](agent_skills.md) — skills, tools, workspaces, and security.
- [TensorAgent](../TensorAgent/README.md) — iOS application architecture and verification.
- [Development guide](../DEVELOPMENT.md) — project layering and native builds.
