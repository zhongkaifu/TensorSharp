# Features
[English](FEATURES.md) | [中文](FEATURES_zh-cn.md)

> Part of the [TensorSharp](README.md) documentation.


- **Multi-architecture support** -- DeepSeek V4 Flash, GLM 5.x (GLM-5.2 `glm-dsa` and GLM-5.3-Flash `glm5next`), Gemma 4, DiffusionGemma, Qwen 3.5/3.6-family, Qwen 3.8 Flash Next (`qwen4exp`), GPT OSS, Nemotron-H, Mistral 3, Muse-Glimmer, Qwen-Image-Edit (image editing), MiniMax-H3 (video with native 32 kHz stereo audio), and Wan 2.1/2.2 (video only)
- **Multimodal inference** -- image, video, and audio inputs (Gemma 4); images for Qwen 3.5/3.6-family / Qwen 3.8 Flash Next / GLM-5.3-Flash / Mistral 3 / Muse-Glimmer / Nemotron-H Omni, each through its own `mmproj` tower. Audio input is Gemma 4 only. `--pdf` is architecture-agnostic: a born-digital PDF's text layer is inlined into the prompt for any model, and only scanned PDFs fall back to page images (which then need a vision model). Generated media is its own axis: Qwen-Image-Edit emits an image, Wan 2.1/2.2 emit an H.264 MP4, and MiniMax-H3 is the one family whose output is **audio as well as video** — a 32 kHz stereo track denoised jointly with the picture and written as a sidecar `.wav` beside the MP4
- **Thinking / reasoning mode** -- structured chain-of-thought output with `<think>` / `<|channel>thought` / `<|channel>analysis` / `to=self` tags (Qwen 3.5/3.6-family, Qwen 3.8 Flash Next, Gemma 4, GPT OSS, Nemotron-H, Muse-Glimmer, DeepSeek V4, GLM 5.x)
- **Tool calling / function calling** -- models can invoke user-defined tools; multi-turn tool-call conversations supported across all three API styles
- **Agent Skills** -- folders of model-facing instructions (`SKILL.md` + scripts / references / assets) that load only when a task needs them. Selected per request with `"skills": ["pdf"]` on every chat API or `--skill` on the CLI; the model pulls the rest through built-in `skills_list` / `skills_read` tools that TensorSharp answers in process, so an ordinary OpenAI client still just receives a finished completion. → [Agent Skills](#agent-skills)
- **Code execution** -- with `--code-exec` the model drives a real shell in a sandboxed workspace, typing a command line and reading back the exit code and everything it printed. Web/CLI chats retain one workspace for their session; each OpenAI/Ollama request retains a private workspace across its internal agent rounds and deletes it afterward, so separate HTTP requests remain stateless while create → run → diagnose → edit → verify can use the same file. When code execution is enabled, skill scripts share that workspace; script-only execution retains per-call scratch. File work goes through a dedicated surface rather than through the shell: `read_file` shows bounded current bytes with line numbers, `edit_file` replaces one exact string in one file, `write_file` creates or intentionally replaces a file, and `apply_patch` changes several files at once, all or nothing. When a diagnostic identifies a bounded regular workspace source file, a code failure returns that bounded source region and directs the model to edit the smallest faulty span before rerunning the check. That is the shape both references use — Claude Code's `Read`/`Edit`/`Write` for the common case and Codex's `apply_patch` envelope for the atomic multi-file one — and it exists because a heredoc re-emits the *whole* file, so a one-line fix costs every line that was already right and re-rolls each of them. The host places the bytes, from text it either finds exactly or refuses to guess at. Off by default, and it needs a real OS sandbox (`sandbox-exec` on macOS, `bwrap` 0.12.0+ on Linux) — without one the tool refuses rather than running unconfined. Generated commands also start with Internet/IP access denied; `--code-exec-allow-network` / `TS_CODE_EXEC_ALLOW_NETWORK` explicitly grants unrestricted host IP-network access without removing the macOS/Linux workspace-write or home-read boundaries (subject to the documented macOS shared-temp exception). Linux additionally bounds descendants with a PID namespace. On macOS, children inherit Seatbelt and ordinary process groups are stopped, but a deliberately detached child can outlive the request; every result reports that gap. That includes LAN/loopback services and IP listening sockets and can expose other host-readable data. macOS denies common `/private/tmp/com.apple.launchd*` pathname sockets while permitting runtime-required Mach lookup and the exact mDNSResponder pathname socket required for DNS, and Linux hides common `/run` endpoints, but local Unix IPC is not a complete isolation boundary: macOS retains shared-temporary-directory Unix IPC for compatibility, and Linux's host network namespace may expose abstract sockets and pathname sockets outside `/run`. The switch is independent of `--skills-allow-network`. Host-installer package/domain allow-lists cannot constrain direct downloads in this unrestricted mode. Windows still needs the explicit `--code-exec-unconfined` escape hatch.
- **Quantized model support** -- loads GGUF files with Q4_K_M, Q8_0, F16, MXFP4, and other quantization formats; performs native quantized matmul without dequantizing to FP32, including memory-efficient pure C# CPU loading for large GGUFs
- **GPU-accelerated** -- GGML Metal on macOS, GGML CUDA on Windows/Linux with NVIDIA GPUs, GGML Vulkan on Windows/Linux with AMD/Intel/NVIDIA GPUs, a direct CUDA/cuBLAS backend with PTX kernels, and an MLX backend for Apple Silicon (mlx-c / Metal), all with CPU fallbacks for unsupported ops
- **Optimized pure C# CPU backend** -- managed GEMM fast paths plus fused SIMD kernels for RMSNorm, RoPE, softmax, fused activations, and other inference hot paths, with the managed matmuls dispatched through a persistent spin-then-park worker pool instead of a `Parallel.For` per matmul -- worth ~+15% prefill and ~2.8x decode on a 122-CPU host. → [Pure C# CPU Backend](#pure-c-cpu-backend)
- **Continuous batching & paged KV cache** -- vLLM-style block-paged KV pool with block-hash prefix sharing across requests, iteration-level scheduler that admits / preempts sequences mid-batch, optional SSD-backed tier for very large KV working sets, and a native fused paged-attention kernel (`TSGgml_PagedAttentionForward`) that drives `ggml_flash_attn_ext` on Metal/CUDA/Vulkan. Enabled by default in `TensorSharp.Server`; opt-out with `--no-continuous-batching`. See [docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md). Note what this does and does not buy: the paged KV cache is **host-resident**, so it delivers admission control and cross-request prefix reuse rather than throughput that scales with concurrency: the `BatchedPaged` route saturates at roughly **69 tok/s** aggregate no matter how many sequences are in flight, because the pool it scatters into lives in host memory. (That ceiling is measured on the PAGED route. GLM 5.x has a separate, default-on, non-paged batched fused decode — see the Batched / parallel inference bullet below — which is reported at 1.81x aggregate decode at 4 concurrent; that is a different mechanism and is not evidence that the paged path scales.) (A device-resident paged KV pool exists in the tree — `TensorSharp.GGML.Native/ggml_ops_paged_kv_pool.cpp` and `TensorSharp.Models/Paged/DevicePagedKvCache.cs` — but it is not wired into any model and is not a shipped feature.) GLM 5.x is the exception: MLA with weight absorption (one 576-wide cache row per token per layer) and the DSA lightning indexer have no paged layout, so concurrency is served by native per-sequence **slots** instead — each request owns its MLA and indexer caches and its own `n_past`, and binding a request switches the active slot without moving KV bytes. Qwen 3.8 Flash Next (`qwen4exp`) is the same shape for the same reason — its GatedDeltaNet, PLE and QSA-indexer state have no paged layout either — and is served through per-sequence **state holders**: each in-flight request owns its attention KV and indexer caches, its GDN conv + delta-net state and its PLE history, the native kernel keys its device-resident recurrent state by the holder, so switching requests is a reference swap, and the engine round-robins sequences through their own captured fused decode graphs.
- **Speculative decoding** -- a pluggable algorithm layer (`--spec-type`: `auto` / `draft-head` / `block` / `ngram`) over a shared draft-verify-rollback runtime; the weight-free `ngram` speculator works on every model, learned draft heads accelerate solo (non-concurrent) decode. Qwen 3.6 and GLM 5.2 ship their NextN block fused into the trunk GGUF; Gemma 4 loads a separate EAGLE-style `gemma4-assistant` draft GGUF via `--draft-model` whose draft layers attend the target's own KV cache. The draft proposes up to `--spec-draft` tokens per step (kept while draft confidence ≥ `--spec-pmin`) and the trunk verifies them in a single batched forward; the request's own sampler — penalties included — drives both drafting and verification, so output is identical to standard decode. Opt in with `--spec` on either host (off by default). On `TensorSharp.Cli` it engages on every single-sequence path — `--input`, `--multi-turn-jsonl` and `--interactive`. On ggml backends fused multi-token-verify / draft-step kernels make it a clear win; the direct `cuda` backend runs a fully GPU-resident per-op verify/draft and is also a win. CPU / GGML CPU / MLX stay on standard decode. Env: `TS_SPEC_*` (shared; the legacy `TS_MTP_*` spellings still work) and `TS_GMTP_*` (Gemma 4 tuning).
- **Batched / parallel inference** -- `IBatchedPagedModel.ForwardBatch` implementations for Mistral 3, Gemma 4, GPT OSS, Qwen 3.5/3.6-family, and Nemotron-H all run by default and pack N sequences into a single forward pass with paged K/V scatter and per-sequence attention via the native kernel. Gemma 4, Qwen 3.5/3.6, GPT OSS, and Nemotron-H expose a per-family `TS_<FAMILY>_BATCHED=0` escape hatch (`TS_GEMMA4_BATCHED=0`, `TS_QWEN35_BATCHED=0`, `TS_GPTOSS_BATCHED=0`, `TS_NEMOTRON_BATCHED=0`) to fall back to the per-sequence KV-swap path for A/B comparison or regression isolation; Mistral 3 has no per-family switch — use the global `TS_SCHED_DISABLE_BATCHED=1`. GLM 5.x has no paged `ForwardBatch`; instead its default-on batched fused decode runs one graph with one token per sequence so the weights are read once — 1.81x aggregate decode at 4 concurrent requests. Set `TS_BATCHED_FUSED_DECODE=0` to use serial fused decode for A/B or regression isolation; batching changes GEMM shapes, and a 2-bit MoE can amplify that into different expert picks.
- **Tensor parallelism & distributed inference** -- split a model across multiple GPUs (Megatron-LM column/row-parallel pattern) with `--tp N` on both `TensorSharp.Cli` and `TensorSharp.Server` (or `TENSORSHARP_TP_DEGREE`), and extend across machines with peer-to-peer TCP clustering (`--tp-node-id` / `--tp-peers`). Hierarchical AllReduce minimizes inter-node traffic. Runs on the direct `cuda` backend and on the GGML CUDA / Vulkan backends, where each rank owns a ggml backend, weight shards, and KV cache on its own GPU. Supports the autoregressive architectures Mistral 3, Gemma 4, Qwen 3.5/3.6-family, GPT OSS, Nemotron-H, GLM 5.x (native local/single-process TP for both GLM-5.2 `glm-dsa` and GLM-5.3-Flash `glm5next` on GGML GPU backends), and Muse-Glimmer (`--tp 2` max there, 2 KV heads), with architecture-specific strategies for MoE expert parallelism / expert slicing, GatedDeltaNet per-rank V-head ownership, and Mamba2 replication. Fused per-rank graphs make `--tp 2` decode faster than a single GPU (Gemma 4 E4B 51.7 vs 37.3 tok/s) and run models that do not fit one card. Note that TP is not the only way a model reaches several GPUs — there are two multi-GPU modes and they are not the same thing. **Tensor parallelism** shards the weights *inside* every layer and pays a collective per layer to reconverge, so it can buy latency as well as capacity. A **layer split** gives each GPU a contiguous run of *whole* layers: nothing is sharded, no collective is issued, and it is a **capacity** feature — it is how a model that does not fit one card runs at all, not a way to make it faster. DeepSeek V4 and GLM 5.x **layer-split across every visible GPU by default, with no flag** (their whole-model executors bin-pack whole layers against each device's free VRAM); `--tp` selects native local/single-process TP for both GLM variants: GLM-5.2 uses Megatron sharding within each layer, while GLM-5.3-Flash (`glm5next`) uses KDA/MLA-head and routed-expert-row sharding on GGML GPU backends, and on DeepSeek V4 `--tp` only caps how many GPUs the layer split uses (the same thing `TS_DSV4_NGPU` sets). On Qwen 3.8 Flash Next (`qwen4exp`), `--tp N` *is* a layer split — the architecture shards no weights, and this is the same (and only) multi-GPU mode llama.cpp offers it, since `-sm row` refuses to load it. Measured on 2x A100-80GB with Qwen3.8-Flash-Next-UD-Q2_K_XL (73.4 GiB): greedy output **byte-identical** between the 1-GPU and 2-GPU runs (same SHA-256), VRAM 24.2 GB + 26.2 GB instead of all of it on one card, and throughput unchanged (prefill ~1520-1550 t/s, decode ~56 t/s either way). Startup prints which mode ran and the per-GPU layer/byte split. Every other architecture uses a single GPU unless `--tp` is passed, and one that supports neither tensor parallelism nor a layer split now says so on stderr and runs on one GPU instead of silently leaving the extra GPUs idle. Optional Redis-backed KV cache and Responses API store for shared state. → [Tensor Parallelism](USAGE.md#tensor-parallelism--distributed-inference)
- **Ollama & OpenAI API compatibility** -- drop-in replacement endpoints for existing tooling
- **Configurable sampling** -- temperature, top-k, top-p, min-p, repetition/presence/frequency penalties, seed, stop sequences
- **Structured outputs** -- the OpenAI `response_format` JSON schema is compiled to a grammar and enforced by grammar-constrained decoding: any token that would break the schema is removed from the distribution before sampling, so the response is structurally valid by construction rather than repaired afterwards. Supported: `type`, `enum`, `const`, `properties`, `required`, `additionalProperties`, `items`, `prefixItems`, `min/maxItems`, `anyOf`, `oneOf`, `allOf`, `$ref`/`$defs` (recursive included), `min/maxLength`, `pattern`, the date/time/date-time/uuid formats, and integer `minimum`/`maximum`. Keywords a CFG cannot express (`not`, `if`/`then`/`else`, `dependentSchemas`, `dependentRequired`, `multipleOf`, `patternProperties`) are refused up front. `TS_JSON_GRAMMAR=0` falls back to prompt-and-repair.
- **Chat templates** -- auto-loaded from GGUF metadata (Jinja2), with hardcoded fallbacks per architecture
- **Inference engine** -- the new `InferenceEngine` (worker-thread scheduler + paged block pool) replaces the legacy single-request FIFO queue inside `TensorSharp.Server`. The old queue object is now a compatibility shim for status/event shapes; the engine itself handles concurrency.
- **Batch processing** -- JSONL input support in the console application, plus a built-in inference benchmark for prefill/decode throughput
- **Streaming** -- token-by-token output via SSE (web) or stdout (console), with abort/stop support for in-flight generations
- **Text-diffusion generation** -- DiffusionGemma uses an iterative EntropyBound denoising sampler instead of autoregressive `Forward()`. The CLI exposes `--diffusion-steps`, `--diffusion-seed`, and `--diffusion-blocks`; the Web UI streams whole-message `replace` events for live denoising previews and batches concurrent diffusion requests through `DiffusionBatchScheduler`.
- **Image editing (Qwen-Image-Edit)** -- a prompt plus an input image produces an edited image. The loaded `qwen_image` GGUF is the MMDiT diffusion transformer; TensorSharp resolves two companion GGUFs alongside it — the Qwen-Image VAE (image ↔ 16-channel latent) and the Qwen2.5-VL-7B text encoder (prompt → 3584-dim conditioning, optional vision grounding via an `mmproj`). The pipeline VAE-encodes the reference, builds text (and optional image) conditioning, runs a FlowMatch-Euler true-CFG denoise loop with reference-latent concatenation, then VAE-decodes back to pixels. The whole 60-block DiT forward is CUDA-graph-captured (`TSGgml_QwenImageForward`), flash-attention is on by default, and the target area is auto-clamped to the device VRAM budget. An optional Lightning distillation LoRA (`--qwen-image-lora` / `TS_QWEN_IMAGE_LORA`, `.safetensors`) cuts the denoise from the base 30 steps at CFG 2.5 to the LoRA's own step count (e.g. 4 or 8, parsed from its file name) at CFG 1.0 with no negative pass -- 60 DiT forwards become 4-8. It is applied as a runtime F32 side-path next to each targeted projection (`y = W_quant*x + b + (alpha/rank)*up*(down*x)`) with the quantized base weights left untouched, **not** merged into them: the Lightning deltas are ~1e-4 RMS, far below a Q2_K quantization step, and a measured merge changed the velocity by 24% relL2 of pure requantization noise. The side-path costs ~4% extra FLOPs, is CUDA-graph-capture-safe, and requires the whole-model or fused per-block CUDA forward -- on a path that cannot host it the model throws rather than emitting noise. The whole-step denoise caches (`TS_QWEN_DIT_CACHE_MODE`: `easycache` skips 40-55% of steps, `fbc` is First-Block-Cache) stay **off** by default because they measurably soften faces on edit workloads. Measured against stable-diffusion.cpp on the project's CUDA `image_edit` scenario (Q2_K DiT + 4-step Lightning LoRA, 544x1184, identical inputs and seed): 40.44 s vs 48.16 s warm. Driven from C# via `QwenImageModel.EditImage(prompt, RgbImage, QwenImageParams)`, from the CLI image-edit mode (`--image`, `--prompt`, `--cfg`, `--diffusion-steps`, `--diffusion-seed`), and from the Web UI with live denoising previews. → [Qwen-Image-Edit card](docs/models/qwenimage.md)
- **Video generation with audio (MiniMax-H3)** -- a prompt produces an H.264 MP4 **and a native 32 kHz stereo soundtrack, generated together**: one diffusion transformer denoises a packed video+audio latent in a single token sequence, so the track is part of the model output rather than something added afterwards. Up to 15 s at 24 fps. TensorSharp runs it as native whole-network ggml graphs — one graph per network, weights bound resident straight from the GGUF/safetensors mmap: the Qwen3-VL-32B text encoder (`TSGgml_MiniMaxH3TextEncode`), its vision tower (`TSGgml_MiniMaxH3VisionEncode`), the DiT step (`TSGgml_MiniMaxH3DitForward`), and video/audio VAE encode and decode; on `--backend cpu` those same networks run instead as managed `MiniMaxH3Direct*` implementations (see [Pure C# CPU Backend](#pure-c-cpu-backend)). The DiT is 50 blocks and ~19.3 B parameters, single-stream with **no cross-attention** — text, conditioning frames, target audio and target video are ONE sequence under full bidirectional attention (hidden 5376, 56 heads × 128 = 7168 inner, patch (t, h, w) = (1, 2, 2)); AdaLN interpolates a learned `[8, 1025]` curve table instead of running a timestep MLP, and 3-axis RoPE over **continuous float** positions puts both streams on one timeline measured in audio-latent units (1/40 s). Two checkpoints, not two settings: `minimax_h3_fl2va_pruned-Q4_K.gguf` takes text and keyframes (`--video-mode t2v`, `i2v` — the image IS the first frame and gets animated — or `fl2v`, first *and* last frame), `minimax_h3_ref2va_pruned-Q4_K.gguf` takes identity/appearance references for a *new* scene (`--video-mode ref`: up to nine `--ref-image`, plus `--ref-video` / `--ref-video-audio` / `--ref-audio`, addressed positionally in the prompt as `<Picture 1>`, `<Video 1>`, `<Audio 1>`), and asking one checkpoint for the other's mode fails with a message naming the file to load instead of quietly dropping the input. It ships **CFG-distilled**, so `--cfg 1.0` is required (TensorSharp refuses anything higher, and `--negative-prompt` consequently does nothing — no unconditional pass runs) and 4-8 `--diffusion-steps` is the fast operating point against a 20-step default. `--video-frames` snaps up onto a `17k+5` grid (5, 22, 39, 56, 73, 90 …), `--width`/`--height` round up to a multiple of 32, and fps is pinned to 24 whatever is asked for. The video VAE decodes 5 latent frames at a time with a 2-frame look-ahead and cross-faded seams, and tiles at 256 px — both **correctness** requirements rather than optimizations, because its decoder is a pure 36-layer transformer whose RoPE coordinates are length-normalized over whatever extent it is handed. Long clips needed one more fix: ggml's flash-attention kernels keep the softmax numerator in FP16 with three bits of headroom, and H3 attends bidirectionally over the whole clip (2364 packed tokens at 22 frames, 8646 at 107), so `h3_attend` pre-scales V by a power of two derived from the key count and undoes it on the output — exact, because attention is linear in V — which is what turned a 107-frame clip that came back pure black into a correct one; the sampler also refuses to save a diverged sample, failing the request at the step where the latent went non-finite rather than writing a black file (`TS_H3_TRACE=1` prints the per-step magnitudes). Measured on an M5 Pro (`ggml_metal`, 22 frames, 8 steps, identical seed) against stable-diffusion.cpp at its best-performing configuration: **20.9 s vs 49.3 s at 256×256 (2.4× faster)** and **63.1 s vs 108.5 s at 640×384 (1.7×)**; on a 16 GB RTX 3080 Laptop (`ggml_cuda`, same workload, stable-diffusion.cpp at `--auto-fit --stream-layers --diffusion-fa --rng cpu` because its default `--offload-to-cpu` path cannot run this model there) the end-to-end result reverses — 43.6 s vs 37.8 s at 256×256 and 63.7 s vs 59.8 s at 640×384, stable-diffusion.cpp 1.15× / 1.07× — while the *per denoise step* cost stays TensorSharp's (3.325 s vs 3.338 s by the 8-vs-16-step slope): the difference is fixed setup on a machine with 16 GB of VRAM and 31.7 GB of RAM against a 33.5 GB model set, and roughly 3 s of the 3.9 s at 640×384 is H.264 encoding against sd.cpp's MJPEG+PCM AVI plus .NET process startup against a native binary, not inference. On a card that size the pipeline manages residency rather than assuming the weights fit: the finished denoiser's device copy is handed back before the video VAE loads when the two would not fit together (peak VRAM during decode 16 041 → ~5 600 MiB, worth 22 s at 640×384 — on Windows/WDDM the oversized allocation does not fail, it is silently backed by host memory and the decode runs at PCIe speed), and the denoiser GGUF is sequentially prefaulted as soon as the text trunk produces its hidden states, pipelined with its own upload rather than joined before it, because weights bound as pointers into the mmap otherwise fault every page in from *inside* the host-to-device copy — 0.91 GB/s, against 5.97 GB/s once the pages are resident. Output is byte-identical with the prefault on or off; the first denoise step goes 14.87 s → ~10.2 s and the whole run 89.0 s → **63.7 s** at 640×384 (67.2 → **43.6 s** at 256×256) on that card. `TS_H3_PREFAULT` picks the mode (`0` off, `1` serial, `2` overlapped with text conditioning, `3` pipelined with the upload — the default; mode `2` loses because the encoder streams its own 17 GB through the same page cache and evicts what was just placed) and `TS_H3_PREFAULT_THREADS` the reader count (default `1`; 4 and 16 streams both measured slower, since this read competes with the teardown and the upload it is warming). `TS_H3_PHASE=1` prints the per-stage breakdown — encoder open / trunk / teardown, prefault, every denoise step, VAE open / decode — and `TS_H3_TE_GROUP=<n>` runs the 50-layer text-encoder trunk in groups of `n` layers, releasing each group's device copy; it is **off** by default because it removes the encoder's own spill (peak 16 041 → 12 981 MiB) bit-identically and is still 3 s slower, a one-shot prefill over a ten-token prompt reading each weight exactly once, so the overflowed ~1.3 GB costs a single PCIe crossing while grouping still moves all 17 GB. Every network is checked against the reference implementation rather than against itself — text encoder cos 0.999999, DiT step video cos 0.9983 / audio cos 0.9998, video VAE encode and decode cos 1.000000, audio VAE decode cos 0.999995. Driven from C# via `MiniMaxH3Model.GenerateVideo(prompt, VideoGenerationParams)`, from the CLI (`--prompt`, `--image`, `--end-image`, `--ref-image`, `--video-mode`, `--width`/`--height`, `--video-frames`, `--diffusion-steps`, `--cfg`, `--audio-vae`, `--no-audio`), and from the server API — `/api/video-generate[/stream]` and `/v1/videos/generations` accept `videoMode`, `endImage`, `referenceImages` / `referenceVideos` / `referenceAudios` / `referenceVideoAudios` and `generateAudio`, and hand back `audioUrl` / `audio_url` beside the MP4, while `GET /api/models` reports what the loaded checkpoint accepts (`video.family` = `minimax-h3`, `supportsAudio`, `supportsEndImageConditioning`, `supportsReferenceConditioning`, `maxReferenceImages`) so the Web UI offers exactly the attachment controls that will work. The soundtrack is written as a **sidecar `.wav`** next to the MP4 rather than muxed in, because muxing needs an encoder that cannot be assumed present — `ffmpeg -i fox.mp4 -i fox.wav -c:v copy -c:a aac fox_with_audio.mp4`. → [MiniMax-H3 card](docs/models/minimax-h3.md)
- **Video generation, video-only (Wan 2.1 text-to-video, Wan 2.2 text/image-to-video)** -- a prompt (plus an optional first-frame image on the Wan 2.2 models) produces an H.264 MP4 with no audio track. The loaded `wan` GGUF is the Wan DiT — Wan 2.1 T2V, Wan 2.2 TI2V-5B (48-channel 16×16×4 latent, 24 fps) and Wan 2.2 A14B (two 14B experts switched at a timestep boundary, second GGUF auto-resolved) are auto-detected; TensorSharp resolves the companions alongside it — the UMT5-XXL text encoder GGUF (prompt → 512×4096 conditioning, exact unigram-Viterbi SentencePiece tokenization) and the matching causal 3D video VAE (`wan_2.1_vae.safetensors` / `Wan2.2_VAE.safetensors`). The FlowMatch CFG denoise (UniPC or Euler) runs the whole DiT (self-attention with 3D RoPE + flash attention over F16 keys/values, cross-attention, AdaLN time modulation — per-token-timestep for TI2V image-to-video) as ONE resident-weight ggml graph per step, CUDA-graph-captured per shape (`TSGgml_WanDitForward`); the video VAE decodes all temporal chunks in one graph with the causal feature cache carried in-graph (`TSGgml_WanVaeDecode`) -- convs go through MPSGraph on Metal (a 736x544x81f decode: 159 s -> 80 s, 1.99x, numerics unchanged at 93.9 dB PSNR; `TS_WAN_VAE_MPS_CONV=0` restores ggml's im2col+GEMM lowering) and through a banded im2col+GEMM path elsewhere, with the im2col budget and the tiling threshold now sized from free device memory instead of a fixed 16 GB card's budget, so large-memory devices decode a 720p plane whole (565 s vs 655 s banded, peak RSS 4.85 vs 5.37 GB) while small cards still tile; and image-to-video conditioning encodes the first frame through the causal VAE encoder in one graph (`TSGgml_WanVaeEncode`). Each stage releases its VRAM before the next, so TI2V-5B 81-frame 480p image-to-video and both A14B Q4_K_M experts fit a 16 GB GPU. Generation runs on every backend except MLX: the GGML paths (`ggml_cuda`, `ggml_metal`, `ggml_vulkan`, `ggml_cpu`) share the whole-graph kernels, while `--backend cuda` and `--backend cpu` run a ggml-independent direct implementation (`WanDirect*`: resident-quantized linears on TensorSharp's MMQ/dp4a/cuBLAS routing with streaming online-softmax attention kernels on CUDA, parallel SIMD GEMM/attention on CPU, and a channels-last banded-im2col causal video VAE shared by both). **Step-distilled checkpoints are auto-detected from the DiT file name** (`Turbo`, `distill`, `Lightning`, `lightx2v`, `FastWan`, `-dmd`, or an explicit `…-4steps-…`) and are by far the biggest speed lever: the official 50-step x CFG recipe costs 100 DiT passes, a 4-step distilled checkpoint costs 4, and the pipeline switches to that step count with guidance off automatically (`--diffusion-steps` / `--cfg` override). Measured on an M5 Pro at 1088x832x121f = 27 404 tokens, `ggml_metal`, Wan2.2-TI2V-5B Q8_0: the base checkpoint runs 100 passes at 120.2 s for ~3 h 30 m end to end, and the identical request on a Turbo checkpoint runs 4 passes for **17 m 30 s** -- only the `--model` path differs. On base checkpoints `--cfg-cache-stride 2` / `3` reuses the guidance direction between steps for a further 1.30x / 1.43x. Numerics verified against diffusers (DiT cosine > 0.995, VAE encoders > 0.999, decoders 59.9 dB / >35 dB PSNR) and across backends (final-latent cosine ≥ 0.999 on identical seeds); the F16 attention keys/values that make one 27 k-token self-attention 2.02x faster (~1.7x per DiT pass together with the VAE work) score the same 0.999964 DiT cosine as F32. Driven from C# via `WanVideoModel.GenerateVideo(prompt, WanVideoParams)`, the CLI (`--prompt`, `--image`, `--video-frames`, `--fps`, `--flow-shift`, `--negative-prompt`), the server API (`/v1/videos/generations` with base64 `image`, `/api/video-generate[/stream]` with `imagePath`), and the Web UI chat (type a prompt — with an attached image for image-to-video — and get the video with live progress: per-pass timings, a running ETA and a 30 s heartbeat, since one pass over a 5-second 720p latent is minutes of GPU work). → [Wan card](docs/models/wan.md)
- **Hybrid SSM-Transformer** -- Nemotron-H mixes Mamba2 SSM layers, attention-only layers, and MoE FFN layers in a single model. The Mamba2 step has both a per-sequence native kernel and a batched native kernel (`TSGgml_NemotronMamba2BatchedStepF32`, NEON SIMD + GCD parallelism) used by the batched path. On GGML backends the attention layers decode through the device-side flash-attention kernel against the resident KV cache (`TS_NEMOTRON_FLASH_DECODE=0` restores the host path), so decode no longer degrades with context length.
- **Hybrid Attention-Recurrent** -- Qwen 3.5/3.6-family models mix full-attention layers with GatedDeltaNet recurrent layers; the batched path keeps recurrent running state in a per-slot recurrent-state pool
- **Mixture of Experts** -- Gemma 4 MoE variants (e.g. gemma-4-26B-A4B), GPT OSS MoE (e.g. gpt-oss-20b), Qwen 3.5/3.6-family MoE (`qwen35moe` / `qwen3next` variants such as Qwen3.5-35B-A3B), Nemotron-H MoE FFN layers, and GLM 5.2 (744B-A40B: 256 routed experts at top-8 plus one shared expert, sigmoid-gated routing with a selection-only bias and a x2.5 routed scale, after 3 leading dense SwiGLU layers), GLM-5.3-Flash (320B: 288 routed experts at top-8 plus one shared expert and the same x2.5 routed scale, with a SwiGLU clamp limit of 10 on every FFN), and Qwen 3.8 Flash Next (512 experts, 10 used per token, interleaved with GatedDeltaNet recurrent layers)
- **MoE CPU offload** -- `--n-cpu-moe N` / `--cpu-moe` (llama.cpp's `-ncmoe` / `-cmoe` equivalent) keeps the routed expert weights of the first N layers in system RAM and multiplies them on the host, leaving attention, the norms, the router and the always-active shared expert on the accelerator. The offloaded layers stay inside the fused whole-model graph on every architecture that has one (Qwen 3.5/3.6, Gemma 4 MoE, GPT OSS, DiffusionGemma) — the accelerator pauses after each offloaded layer's router, the host multiplies the selected experts straight out of the GGUF mmap, and the result is handed back before the next segment — so only ~8 KB of activation crosses the bus per layer at decode. Gemma 4 MoE and Qwen 3.5/3.6 segment their prefill graphs the same way, where the host side becomes a real GEMM over the whole prompt chunk. It also composes with tensor parallelism: under `--tp N` the seams merge into the ranks' own AllReduce segment schedule, so the fused multi-rank graph is kept and each offloaded layer is evaluated once on the host over the unsharded expert stack (Qwen3.5-35B-A3B `--tp 2`: 17.4 GB of resident weights across 2 GPUs falls to 3.2 GB; gemma-4-26B-A4B: 12.9 GB falls to 2.4 GB, byte-identical output). Measured on a 16 GB RTX 3080 Laptop: Qwen3.6-35B-A3B 13.4 -> 4.6 GB at `--cpu-moe`; gemma-4-26B-A4B 16.1 -> 4.8 GB (decode 39.7 -> 17.7 tok/s, and 38.6 tok/s at `--n-cpu-moe 8` for 3 GB back); gpt-oss-20b 16.2 -> 2.9 GB, which takes it off the WDDM spill cliff and turns 0.3 tok/s into 25.4 at `--n-cpu-moe 12`. That is what makes these models fit beside a long-context KV cache on a 12-16 GB GPU. DeepSeek V4 Flash uses the same seam on the GPU backends: it is 91% routed-expert bytes, and the loader sizes its layer split against each device's *actual* free VRAM. Offload stays opt-in there too -- a checkpoint that does not fit is refused at load with the exact `--n-cpu-moe N` that would make it fit, rather than silently trading away decode throughput -- and with that flag 3x48 GB RTX A6000 hosts the UD-Q8_K_XL checkpoint at 10 tok/s decode / 126 tok/s prefill. GLM 5.x offloads the same way (92% of that checkpoint is routed-expert bytes) and its host-resident experts are served straight from the GGUF mapping rather than copied. Note that offload is for *fitting*, not speed: on 3x RTX PRO 6000 where GLM-5.2 already fits, `--n-cpu-moe 30` costs pp2048 915.9 -> 94.7 and tg64 43.9 -> 16.4 tok/s. On GLM 5.2 the host-resident experts are multiplied straight out of the GGUF mapping with no private copy, and offload composes with `--tp N` — host-resident layers keep their experts whole and rank 0 evaluates them. It also nearly doubles the context the loader can size there, 342,272 -> 646,400 tokens. -> [MoE CPU offload](USAGE.md#mixture-of-experts-cpu-offload---n-cpu-moe)
- **Batched GPU MoE** -- a single fused GGML graph dispatch handles all selected experts (plus the optional shared expert and residual add) for Qwen 3.5/3.6-family and Nemotron-H decode, eliminating per-expert round-trips
- **Whole-model fused decode graphs** -- Gemma 4 (dense and MoE), Qwen 3.5/3.6 and GPT OSS run an entire decode token — every layer, the MoE router and experts, the final norm and the LM head — as ONE GGML graph dispatch instead of one submission per layer, so the GPU is never left waiting on the host between layers. On CUDA/Vulkan the graph is built once with stable tensor addresses and replayed (`ggml_set_rows` KV write with the row as an I64 input, a stride-padded attention window with an F16 mask input), which is what lets ggml-cuda capture it as a CUDA graph. GPT OSS decode: 24 → 154 tok/s on an A40, and flat in context length (133 tok/s at 16K) where the per-layer path collapsed to 2.3. Disable per model with `TS_GPTOSS_MODEL_DECODE=0` / `TS_GEMMA4_FD_PERSIST=0` / `TS_QWEN35_FD_PERSIST=0`.
- **KV cache codecs** -- pluggable codec interface (`IKvBlockCodec`) with a built-in TurboQuant (2-bit affine / Q4 / Q8) compressed codec for paged blocks. The CLI accepts all four `--paged-kv-quant-bits 0|2|4|8` values; the server's legacy standalone flag accepts `0|4|8`, while `TS_KV_PAGED_QUANT_BITS=2` selects the 2-bit codec directly. The 2-bit tier reaches ~10x compression on fp32 blocks for very long contexts.
- **KV cache precision** -- `--kv-cache-dtype <f32|f16|q8_0|q4_0>` (CLI and server, env `KV_CACHE_DTYPE`; default auto — the backend/model pick) trades a small numerical drift for memory. `q4_0` (~0.56 bytes/element, ~1/7 of f32) is the most aggressive tier and is aimed at the very long (128K–256K) contexts where the KV cache dominates memory; the block-quantized tiers (`q8_0`/`q4_0`) require the native GGML flash path.
- **Message editing** -- edit or delete previous messages in the web chat UI and regenerate from that point
- **Text/Image/Audio/Video/PDF uploads** -- the web UI accepts file uploads up to 500 MB and preserves text content in full. Born-digital PDFs have their complete text layer extracted and inlined into the prompt (cap pages explicitly with `TS_PDF_MAX_PAGES`); scanned PDFs are rendered to page images for vision-capable models. The final prompt is checked against the model's actual context window instead of an arbitrary upload budget. The CLI accepts a PDF in one-shot mode via `--pdf <file>`
- **Per-turn observability** -- structured logs capture the full user input and the full raw assistant output (both `<think>` reasoning and the final result) plus the KV cache hit ratio. The same cache-hit stats are surfaced through every API: `prompt_cache_hit_tokens` / `prompt_cache_hit_ratio` (Ollama), `usage.prompt_tokens_details.cached_tokens` (OpenAI), and `promptTokens` / `kvReusedTokens` / `kvReusePercent` in the Web UI SSE `done` event


## Thinking / Reasoning Mode

Models that support thinking mode (Qwen 3.5/3.6-family, Qwen 3.8 Flash Next, Gemma 4, GPT OSS, Nemotron-H, Muse-Glimmer, DeepSeek V4, GLM 5.x) can produce structured chain-of-thought reasoning before generating the final answer. The thinking content is separated from the main response and can be displayed or hidden by the client.

- **Qwen 3.5/3.6-family / Nemotron-H:** uses `<think>...</think>` tags
- **Gemma 4:** uses `<|channel>thought\n...<channel|>` tags
- **GPT OSS:** uses Harmony format with `<|channel|>analysis` for thinking and `<|channel|>final` for the response
- **DeepSeek V4:** uses `<think>...</think>` tags; the chat template closes the block for you unless `--think` is passed, so reasoning is opt-in
- **GLM 5.x:** uses `<think>...</think>` tags, opt-in like the rest -- `--think` adds the `Reasoning Effort: Max` system line and leaves the generation prompt's block open for the model to close; without it the prompt emits an empty `<think></think>` so the model answers directly. Past turns' reasoning is always dropped from the prompt, matching the template's `clear_thinking` default

Enable via `--think` (console), `"think": true` (Ollama API), or the thinking toggle in the web UI.

## DSpark Block Speculative Decoding (DeepSeek V4)

DeepSeek V4 ships **DSpark** ("Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation") as a support module in the checkpoint: three DSV4 blocks that read the trunk's hidden states and propose a whole BLOCK of tokens per step instead of one, a Markov head that conditions each block position on the token before it, and a confidence head that predicts each position's acceptance probability. TensorSharp loads it as a separate drafter GGUF (`--draft-model`, built with `eng/dsv4-dspark-to-gguf.py`) and runs it on both GPU engines (`--backend cuda` and `--backend ggml_cuda`) for greedy single-sequence generation — on ggml the drafter is three extra graph layers whose key ring the trunk graph commits itself, so speculation costs no host round trips; the trunk verifies each block in one batched forward and keeps only the prefix it would have produced anyway. Measured **1.3-1.4x decode** on 4xA40 with the drafter's cumulative-confidence gate at its default; the gate matters because each extra verify row pulls a fresh set of MoE experts through VRAM. See the [DeepSeek V4 card](docs/models/deepseek4.md#dspark-speculative-decoding).

## DFlash / DFlash2 / DSpark Block Speculative Decoding (Muse-Glimmer, Qwen 3.8, Nemotron 3.5 Lightning)

Muse-Glimmer and Qwen 3.8 both have a block drafter, **DFlash**: a separate 5-layer GGUF (`general.architecture = dflash`) that proposes the whole speculative window in one forward. It is architecture-agnostic on the TensorSharp side - a target model gets it by tapping the per-layer residuals the drafter's encoder reads, and nothing else. It borrows the target's token embedding and LM head, keeps its own sliding-window KV ring, and runs three passes per step — *encode* the trunk's per-layer input residuals at `dflash.target_layers` into one wide row, *inject* that row as the K/V of every draft layer, then *draft* `[anchor, MASK x (block-1)]` through the five blocks and score it with the target's LM head. The trunk verifies the block in one batched forward and keeps only the prefix it would have produced anyway, so **the emitted token stream is the plain-greedy stream**.

**DFlash2** is the same backbone plus two additions, both keyed off the GGUF so one code path serves either generation: a grouped dynamic depthwise convolution around every attention and every FFN sublayer (which gives a block-diffusion draft a local left-to-right signal without a second forward), and a candidate selector that scores the top-K candidates of adjacent positions pairwise through two low-rank `[vocab, r]` codebooks and reads the block off as a walk through that lattice - so position *i+1* is no longer chosen without knowing what *i* chose. Attach either with `--draft-model`; the file says which it is. See [speculative_decoding.md](docs/speculative_decoding.md#dflash-and-dflash2).

Both halves are fused native graphs that CUDA-graph-capture and replay, and the draft block finishes with an on-device `argmax` (or, for DFlash2, a ~7 KB lattice), so the 202048-wide probability block never crosses PCIe. A runtime cost governor measures speculation against plain decoding and parks the drafter while it is measurably slower — speculation can therefore only help, but it needs a few hundred generated tokens to settle. Load it with `--draft-model` (CLI) or `TS_MUSE_GLIMMER_DFLASH`/`TS_QWEN35_DFLASH`. Sampling composes — verification draws each token from a trunk row with the run's own sampler — but note that a block drafter proposes its whole block in one pass, so penalties are not applied to the proposal and acceptance falls as a penalized history grows. Measured against llama.cpp's own DFlash on one RTX PRO 6000 Blackwell (Q8_0, greedy, 60-token prompt): 50.9 tok/s vs llama.cpp's 45.5 and 35.0 plain. See the [Muse-Glimmer card](docs/models/muse-glimmer.md#3-dflash-speculative-decoding).

**DSpark on Nemotron 3.5 Lightning** is the same block machinery with the DSpark module NVIDIA ships for it: the official `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark` checkpoint is a 6-layer Qwen3DSparkModel-style drafter (SWA-1024, per-head attention-sink biases, encoder reading the trunk residuals at layers 2/6/20/30/42/52, rank-512 **Markov head** that conditions every draft position on the one before it and an optional bonus-anchor slot). The GGUF (`dflash` architecture; `eng/nemotron-dspark-to-gguf.py` rebuilds it from the safetensors, bit-verified against llama.cpp's export) is loaded with `--draft-model`; the drafter's file advertises the Markov head and sinks and the block pass then drafts the whole block width with the chain — no DFlash2 conv/selector. The Nemotron trunk is the first recurrent (Mamba-2) block draft target: `SpecForward` taps the target layers, the executor snapshots and rolls back the conv/SSM state on partial acceptance, and the DSpark window defaults to 3 drafts.

## Speculative Decoding

A drafter proposes several future tokens cheaply, the trunk verifies all of them in one batched forward, and accepted tokens are committed in a single step. Because the request's own sampler — temperature, top-k/p, and repetition/presence/frequency penalties — drives both the draft and the verify, the output is identical to standard decode; speculation only changes how many forward passes it takes to produce it. Engages for solo (non-concurrent) sequences.

**Multi-turn chat.** Speculation used to arm only on a turn that started from an empty KV cache, so a Web UI conversation sped up on its first turn and then silently never again — which is what made DFlash2 look useless from turn 2 onward. Whether an algorithm can start drafting on top of a reused KV prefix is now the algorithm's own call (`ISpeculator.CanArmAfterPrefixReuse`), and the block-draft and n-gram speculators opt in. Measured **1.02x → 1.85x** in the server chat path.

**The design is three independent layers** — *model architecture* ≠ *speculation algorithm* ≠ *speculator weights* — so a new model reuses every algorithm and a new algorithm reuses every model. A model implements `ISpeculativeTarget` (a multi-row forward with per-row logits, plus the KV rollback trio); an algorithm implements `ISpeculator` (`Propose` / `Commit`) and is registered by name; a checkpoint's trained drafter sits behind `IDraftHead`, a thin model-specific adapter, because those weights are bound to one target model and do not transfer. Adding EAGLE, Medusa, PARD or a tree-drafting variant is a new class plus one `SpeculatorRegistry.Register` call — no model, executor or scheduler code changes. See [Speculative Decoding in TensorSharp](docs/speculative_decoding.md).

Four algorithms ship today, selected with `--spec-type`:

| `--spec-type` | Drafts by | Needs trained weights? |
|---|---|---|
| `auto` *(default)* | whatever drafter the checkpoint carries | — |
| `draft-head` | one token per pass through a NextN/MTP head, chaining its own hidden state (Qwen 3.6, GLM 5.2, Gemma 4's separate assistant GGUF) | yes, per target model |
| `block` | a whole block per pass with a confidence head (DeepSeek V4 DSpark, DFlash / DFlash2 on Muse-Glimmer and Qwen 3.8, DSpark Markov head on Nemotron 3.5 Lightning) | yes, per target model |
| `ngram` | suffix match over the sequence's own tokens — where did these last few tokens occur before, and what followed? | **no** |

`ngram` is the model-agnostic one: it works on **every** checkpoint, including those that ship no speculator at all, and is strong wherever the answer quotes its input — summarizing, editing, translating or answering about a document, repetitive structured output, code with repeated identifiers, agentic tool loops. Measured on Qwen3.5-9B (Q8_0, ggml_metal, M5 Pro), which ships no draft head: **45.2 tok/s vs 31.4 plain (1.44x)** on a reproduce-this-config prompt, with byte-identical output. On free-form prose it finds nothing, every step degrades to a plain decode, and the runtime cost governor keeps that cheap.

Speculative decoding is **off by default**. `--spec` (env `TS_SPEC=1`), on the server or on `TensorSharp.Cli`, is the explicit opt-in for drafters embedded in the trunk checkpoint (Qwen 3.6's and GLM 5.2's NextN blocks), because loading them pages extra weights into VRAM; a drafter that ships as its own GGUF is enabled by `--draft-model` alone, with an explicit `--no-spec` as the veto. Env vars are still published under both `TS_SPEC_*` and `TS_MTP_*` — the glm-dsa native loader reads `TS_MTP_SPEC` / `TS_MTP_DRAFT` from C++ at load time, so those names are a cross-language contract:

```bash
# Qwen 3.6 — use the -MTP- repository GGUF so the embedded NextN block is retained
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda \
    --spec --spec-draft 8 --spec-pmin 0.75

# Gemma 4 — load the separate gemma4-assistant draft GGUF that matches the target
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda \
    --draft-model models/gemma-4-E4B-it-assistant.Q8_0.gguf
```

**Three draft-head shapes:**

- **Qwen 3.6 (embedded NextN)** — the GGUF carries one extra decoder block past the main stack (`{arch}.nextn_predict_layers`) plus the NextN projection/norm tensors. No separate file is required; `--draft-model` is ignored. The recurrent trunk state (GatedDeltaNet) is snapshotted so a partially-rejected verify batch can be rolled back.
- **GLM 5.2 (embedded NextN)** — same shape, and the stock [unsloth/GLM-5.2-GGUF](https://huggingface.co/unsloth/GLM-5.2-GGUF) already carries it (`blk.78.nextn.*` plus a full MLA + 256-expert decoder block). Nothing extra to download: `--spec` is the whole configuration, on the CLI (`--input`, `--multi-turn-jsonl`, `--interactive`) as well as the server. The block is only paged in when that flag is set, because it is a whole extra decoder layer (~3 GiB at IQ2_XXS) competing with the KV cache for the same VRAM the loader sizes the context against. glm-dsa has no recurrent state, so a partially-rejected verify keeps the accepted prefix's KV and only rewinds the position — no re-forward. See the [GLM card](docs/models/glm.md#nextn--mtp-speculative-decoding).
- **Gemma 4 (separate `gemma4-assistant` GGUF)** — an EAGLE-style recurrent drafter loaded with `--draft-model`, which enables speculation by itself. It holds no K/V of its own: every draft layer queries the **target model's** existing per-layer KV cache (last local + last global layer), so the drafter is stateless given `(token, hidden)`. The draft's hidden size must match the target — pair the 12B target with its 12B draft, not the 26B-A4B draft. A mismatched, missing, or incomplete draft GGUF **fails fast at startup** with a remediation hint instead of silently disabling speculation.

**Where it's profitable** (engaged automatically; otherwise the engine serves standard decode):

| Backend | Qwen 3.6 | GLM 5.2 | Gemma 4 |
|---|---|---|---|
| GGML CUDA / GGML Metal | ✅ fused multi-token-verify + draft-step kernels | ✅ one graph per verify window in the native whole-model executor | ✅ fused multi-token-verify + draft-step kernels |
| Direct CUDA (`cuda`, driver-API/cuBLAS) | ✅ GPU-resident per-op verify/draft | — (GLM runs the per-op path only on `cpu`) | ✅ GPU-resident per-op verify/draft |
| CPU / GGML CPU / MLX | standard decode (verify can't keep up) | per-op reference path (correct, not fast) | standard decode |

Tuning: `--spec-draft` (default `8`) bounds tokens drafted per step; `--spec-pmin` is the confidence gate (`0` = never gate), and drafting stops at the first token below it. What that number *means* is the algorithm's business, so each brings its own default rather than sharing one: `0.15` for a per-token head (top-1 probability over its top-10 logits), `0.35` cumulative for a block drafter, `0` for n-gram (where it scales the required match length instead). The two knobs interact — a wide window occasionally forms a long chain that is mostly rejected, and those verify rows are paid for either way — so they are worth sweeping together on a new model/host pair. On GLM 5.2, `--spec-draft 4 --spec-pmin 0.55` was best or tied-best in every measured run, by ~4% over the defaults. Gemma 4 draft-path A/B switches are the `TS_GMTP_*` env vars (see the **MTP / speculative-decoding tunables** table under [Web Application](USAGE.md#web-application)). Per-architecture mechanics are in the [Qwen 3.5/3.6 card](docs/models/qwen35.md), the [GLM card](docs/models/glm.md#nextn--mtp-speculative-decoding) and the [Gemma 4 card](docs/models/gemma4.md).

**Greedy output and floating point.** Verification draws every emitted token from a trunk row, so speculation cannot change *which distribution* a token comes from. It does change the *arithmetic*: a K+1-row verify runs the trunk's matmuls at a different batch size than a 1-row decode, which selects different kernels and reduction orders. On a dense model that is invisible; on GLM-5.2 — 2-bit weights, 256 experts at top-8 — a last-bit difference in a router logit changes which experts run, and 78 layers amplify it. Measured over 140 verify rows against per-token decode: the top token differs on **2.9%** of rows, so a long greedy run eventually takes a different (equally valid) branch. Runs with drafting suppressed reproduce greedy exactly, which is what locates the effect in the batch size rather than in the speculation.

## Pure C# CPU Backend

`--backend cpu` is TensorSharp's 100% pure C# path. Its managed matmuls now run on
a persistent spin-then-park worker pool (`TensorSharp.Models/CpuWorkerPool.cs`)
instead of a `Parallel.For` per matmul. Two problems were fixed together: the
work-item count used to scale with the *thread* count -- 1024 tiny tasks per matmul
at 122 threads, which stopped scaling past 8 -- and every matmul paid a ThreadPool
fork/join.

Measured on gemma-4-E4B-it-Q8_0 with a 122-CPU allocation, `--backend cpu`, with the
pool-off baseline A/B-ed inside one binary via `TS_CPU_POOL` and the widths set by
`TS_CPU_THREADS` (each cell is the two interleaved runs, tok/s):

| pool width | prefill | decode |
|---|---|---|
| off (before) | 21.7 / 21.0 | 2.0 / 2.4 |
| 32 | 24.9 / 24.1 | 4.9 / 5.0 |
| 48 | 25.6 / 28.5 | 5.4 / 6.0 |
| 61 (default = cores/2) | 24.2 / 24.9 | 6.3 / 5.9 |
| 122 (every core) | 13.5 | 4.8 |

So prefill gains ~15% and decode ~2.8x. The default width is deliberately **half**
the usable cores rather than all of them: pool workers spin between jobs while the
rest of the CPU path still uses the ThreadPool, so a pool that owns every core
starves the work it is waiting on. At 122 threads that shows up as a PREFILL
regression (13.5 against 21.7 / 21.0 with the pool off) while decode still beats
the pool-off baseline (4.8 against 2.0 / 2.4). The 61-wide default is simply the
best row overall: it beats the 122-wide one on both axes, and gives up a little
prefill against the 48-wide one for equal-or-better decode. Tune with `TS_CPU_THREADS`, `TS_CPU_POOL`,
`TS_CPU_SPIN`, `TS_CPU_TASK_BYTES` and `TS_CPU_TASKS_PER_WORKER`
([env var matrix](docs/env_var_feature_matrix.md)).

**Zero-copy quantized weights.** `BackendType.Cpu` was the only backend missing
from `CanUseFileMappedQuantizedWeights`, so it alone copied every quantized
tensor into fresh anonymous memory at load instead of binding it straight from
the GGUF mapping, as every GGML backend already did. It now binds them zero-copy
-- `ManagedQuantizedOps` reads a weight through a raw pointer and never writes to
it -- and the loader reports the split, e.g. for GLM-5.3-Flash UD-Q2_K_XL:

    Quantized: 103255 MB (103255 MB file-backed), F32: 983 MB

Any model whose weights used to be copied benefits and the effect is largest on
big quantized checkpoints: that one went from a load that never completed
(resident set 412 GB and still climbing) to **~48 s**, most of which is the
page-cache prefault.

**`IQ2_XS` / `IQ4_XS`, and direct i-quant dots.** `ManagedQuantizedOps` gained
managed dequantizers for `IQ2_XS` and `IQ4_XS` (verified against ggml's own
`dequantize_row_*`) plus entries in the CPU quantized-storage matrix, so weights
of those types are kept quantized instead of expanded to F32 at load -- an
expansion that came to 765 GB for GLM-5.3-Flash UD-Q2_K_XL, which is why that
load grew rather than failing. It also gained direct `IQ2_XS x Q8_K` and
`IQ3_XXS x Q8_K` dot kernels with AVX2 paths (`VecDotIq2XsQ8KAvx2`,
`VecDotIq3XxsQ8KAvx2`); previously both types fell to the generic
dequantize-the-row-into-scratch path. This is backend-wide, not GLM-specific: it
applies to any model on `--backend cpu`.

One trap worth recording, since it does not announce itself: ggml folds a
constant into the **result** of some i-quant dots -- 0.125 for `IQ2_XS`, 0.25 for
`IQ3_XXS`, 1.0 for `IQ3_S` -- rather than into each per-block scale. Omitting it
is an 8x error that produces fluent-looking garbage rather than a crash.

**GLM-5.3-Flash on `--backend cpu`.** `glm5next` first could not load and then
could not run on the pure C# path; four fixes, three of them silent rather than a
clean error, make it work for text. NoPE MLA (GLM-5.3 sets
`rope.dimension_count = 0`, so there is no rope half anywhere and the compressed
latent IS the whole cache row, where the GLM-5.2 path narrowed a zero-width
slice), an MLA-absorption check that demanded `attn_k_b` / `attn_v_b` on all 45
trunk layers when `glm5next` carries them on only the 12 full-attention ones
starting at layer 3, and the two loader problems above. Measured on the 122-CPU
box, 22-token prompt, 16 tokens out (tok/s):

| | prefill | decode |
|---|---|---|
| `cpu`, scalar i-quant dots | 0.9 | 0.4 |
| `cpu`, AVX2 i-quant dots | 3.1 | 1.6 |
| `ggml_cpu`, same file and box | 17.7 | 3.9 |

So ~5.7x off `ggml_cpu` on prefill and ~2.4x on decode. About 89% of the time is
the MoE expert path -- 8 experts x 3 matrices x 45 layers of small matmuls per
token -- and it sits OUTSIDE the `Linear` timing bucket, so the built-in
breakdown reports it as "Other"; that is not unaccounted overhead.

**This route is not claimed as parity.** The prefill-logit cosine against
`ggml_cpu` is **0.9567** over the 154880-wide vocabulary (compared with
`TS_DUMP_LOGITS`, which writes the first *real* forward's logits and skips the
warm-up ones). The greedy text differs because native's top two logits are 0.11
apart and the managed path ranks them the other way, putting native's pick at
managed rank 2. That is consistent with 2-bit expert-pick sensitivity -- the same
effect that makes `TS_BATCHED_FUSED_DECODE=0` useful for an exact serial-path A/B -- but it is **not**
proven to be only that: 0.96 is lower than the ~0.999 a higher-precision
checkpoint would be expected to give, and no higher-precision GLM-5.3 GGUF was
available as a control. Treat the managed path as a reference implementation to
A/B against, not as bit-parity. → [GLM card](docs/models/glm.md#glm-53-flash-on---backend-cpu)

**Shared direct primitives.** The non-ggml execution primitives behind
`BackendType.Cuda` and `BackendType.Cpu` were family-neutral, so
`WanDirect{Context,Linear,Ops}` moved to `TensorSharp.Models/Direct/DirectOps.cs`
as `Direct{Context,Linear,Ops}` and are now shared by the Wan video networks and
MiniMax-H3. `DirectOps`' row loops go through the same worker pool, so the Wan and
Qwen-Image CPU paths benefit from it too. `DirectLinear` on CPU also stopped
expanding a quantized weight to F32 at load: it keeps the GGUF storage type and
calls `ManagedQuantizedOps.AddmmQuantizedToFloat32`. On Wan (256x160x5f, 1 step,
`--backend cpu`) that is **80.9 s vs 121.4 s** at **4x less** weight memory, and
against the native `ggml_cpu` render it scores 43.51 dB where the old path scored
43.39 dB -- marginally *closer* to native, not a quality regression. F16/BF16/F32
weights keep the plain GEMM; `TS_DIRECT_QUANT_WEIGHTS=0` restores the old
behaviour.

**MiniMax-H3 on `--backend cpu`.** MiniMax-H3 was the only generation model without
a pure C# path -- every stage was an unconditional whole-model native ggml call. It
has one now: `MiniMaxH3Direct{DiT, TextEncoder, VideoVae, AudioVae, VisionEncoder,
VideoVaeEncoder3D, AudioVaeEncoder}`, selected by one predicate
(`MiniMaxH3Model.UsesDirectBackend`). t2v, i2v, fl2v and reference conditioning
(images, clips, soundtracks) all work there. Wan, Qwen-Image and DiffusionGemma
already had pure C# CPU paths and did not change in coverage.

Parity against the GGML path on identical inputs with a fixed `--diffusion-seed`
(256x160, 5 frames, 1 step). The *control* column is GGML measured against
**itself** -- its own flash kernel against its explicit-softmax fallback
(`TS_H3_NO_FLASH=1`) -- which is the yardstick for how far two correct
implementations may legitimately drift:

| route | managed vs GGML (cosine) | control (cosine) | render PSNR |
|---|---|---|---|
| text encoder alone (64 layers, 32B Q4_K_M) | 0.99999899 | -- | -- |
| t2v | 0.998740 | 0.997032 | 31.95 dB (control 28.17 dB) |
| i2v (3-D VAE encode) | 0.999897 | -- | 34.87 dB |
| ref-audio (audio VAE encode) | 0.999410 | -- | 35.27 dB |
| ref-image (vision tower) | 0.998554 | 0.999275 | 28.80 dB (control 30.80 dB) |
| vision tower output alone | 0.999919 | 0.999952 | -- |

t2v, i2v and ref-audio agree with GGML more closely than GGML's own two attention
kernels agree with each other. The **vision tower is the exception**: its residual
is ~1.4x the control rather than below it. At cosine 0.9999 over 737k elements it
is structurally correct, but the residual is unexplained -- disabling flash made
agreement slightly *worse*, so the F16 K/V cast is not the cause. That route is not
claimed as parity.

The DiT is not bit-identical to GGML and cannot be: truncating the trunk on both
paths with `TS_H3_DIT_LAYERS` shows 1-cosine ~1e-5 through 25 layers and then
*non-monotonic* amplification (1.15e-3 at depth 40, 1.5e-4 at 44, 1.26e-3 at 50) --
and non-monotonic is what rules out a bug.

Speed is mixed rather than uniformly slower, `--backend cpu` against `ggml_cpu` on
the same clip: t2v 69 s vs 14 s, i2v 70 s vs 176 s, ref-audio 63 s vs 71 s,
ref-image 112 s vs 20 s. The vision tower is the expensive managed stage.
Diagnostics: `TS_H3_DUMP_TE`, `TS_H3_DUMP_VEL_V`, `TS_H3_DUMP_VEL_A`,
`TS_H3_DUMP_VIS` and `TS_H3_DIT_LAYERS`, plus the pre-existing `TS_H3_NO_FLASH`.

## Tensor Parallelism & Distributed Inference

Tensor parallelism (TP) splits a single model across multiple GPUs using the
Megatron-LM column/row-parallel pattern. Each transformer block runs
column-parallel projections (QKV, gate/up) that split output heads or
intermediate dimensions across GPUs, independent per-GPU attention or activation
computation, and row-parallel projections (output, down) followed by an AllReduce
that reconverges the hidden state. Norms, embeddings, and the LM head are
replicated.

**Two multi-GPU modes, and they are not the same thing.** Tensor parallelism
shards the weights *inside* each layer and pays a collective (one or two
AllReduces per layer) to reconverge, so every rank does part of every layer's
work and the mode can buy latency as well as capacity. A **layer split**
instead gives each GPU a contiguous run of *whole* layers: nothing is sharded,
no collective is issued, and the GPUs take a token in turn rather than working
on it together. A layer split is therefore a **capacity** feature — it is how a
model that does not fit one card runs at all — and should not be expected to
raise throughput. Which mode each architecture uses:

| Architecture | Multi-GPU mode |
|---|---|
| Mistral 3, Gemma 4, Qwen 3.5/3.6-family, GPT OSS, Nemotron-H, Muse-Glimmer (`--tp 2` max) | Tensor parallelism, opt in with `--tp N` |
| GLM-5.2 (`glm-dsa`) | Layer split across every visible GPU by default; `--tp N` switches it to native local/single-process tensor parallelism |
| GLM-5.3-Flash (`glm5next`) | Layer split across every visible GPU by default; `--tp N` selects native local tensor parallelism on GGML GPU backends |
| DeepSeek V4 Flash | Layer split across every visible GPU by default; `--tp N` only caps how many GPUs the split uses (as `TS_DSV4_NGPU` does) |
| Qwen 3.8 Flash Next (`qwen4exp`) | Layer split, opt in with `--tp N` (GGML CUDA / Vulkan) |

Startup prints which mode actually ran, so it never has to be inferred from
`nvidia-smi`.

**Local TP** runs within a single process. On the direct `cuda` backend one
thread issues commands to all GPUs and CUDA streams provide the parallelism; on
the GGML backends a rank worker pool drives the GPUs concurrently, because a
GGML op submits *and* synchronizes in one call. Enable with `--tp N` on either
`TensorSharp.Cli` or `TensorSharp.Server` (or `TENSORSHARP_TP_DEGREE=N`);
`TENSORSHARP_TP_DEVICES=0,2` picks which physical GPUs the ranks map to.

**Distributed TP** extends across machines via a peer-to-peer TCP mesh. Each node
runs its own process with its own local GPUs; AllReduce is hierarchical — local
P2P within each node, TCP across node representatives, then broadcast back — so
only `1/tp_local` of the data crosses the network. Enable with `--tp-node-id` and
`--tp-peers` (or `TENSORSHARP_TP_NODE_ID` / `TENSORSHARP_TP_PEERS`). The server
can join such a cluster as node `0` — the driver that owns sampling and serves
HTTP — with every other node running a `TensorSharp.Cli` worker.

Architecture-specific strategies handle heterogeneous layers:

| Architecture | Strategy |
|---|---|
| Dense transformer (Mistral 3) | Standard column/row-parallel QKV + FFN |
| MoE (GPT OSS, Nemotron-H) | Expert slicing — each GPU holds `1/tp` of every expert's weights; router is replicated |
| MoE on GGML (Qwen 3.5/3.6) | Expert parallelism — whole experts partition across GPUs (128 of 256 per rank), so each rank keeps the single batched `ggml_mul_mat_id` dispatch per projection; the shared expert stays Megatron-split |
| MoE on GGML (Gemma 4) | Megatron split *inside* each expert (gate/up column-parallel, down row-parallel) so the fused whole-model MoE trunk kernel keeps working with global expert ids; the expert sum becomes a third row-parallel AllReduce per layer. `TS_GEMMA4_TP_FUSED_MOE=0` falls back to the whole-expert per-op path |
| GatedDeltaNet SSM (Qwen 3.5/3.6) | Block-cyclic V-head assignment — each rank runs its own packed GDN kernel on its V-head subset with independent delta/conv state, resident on its GPU; no cross-rank communication for the recurrent path |
| Mamba2 SSM (Nemotron-H) | Replicated on rank 0, result broadcast to all ranks |
| MLA / KDA + sparse-attention MoE on GGML (GLM 5.x; native TP is local/single-process) | GLM-5.2 shards MLA heads and the hidden rows inside every routed expert. GLM-5.3 additionally head-shards KDA and its per-rank recurrent state, while MLA heads and routed-expert hidden rows are sharded the same way. Attention partials reduce before the nonlinear Sinkhorn hyper-connections. On GLM-5.3's eligible segmented fast path, routed-MoE partials reduce first, then every rank computes and adds the replicated shared expert locally; hyper-connections, pooled indexer, router, norms, dense layers and embedding also remain unsharded and execute per rank, while output norm / LM head stay on rank 0. CPU MoE, tracing, partial `TS_GLM_TP_SHARD`, oversubscription, or missing native hyper-connection kernels select the combined scheduler fallback, where the shared expert runs once on rank 0; `TS_GLM_TP_FUSED=0` forces that diagnostic fallback |

TP runs on the `cuda` backend and on the GGML CUDA / Vulkan backends
(`ggml_cuda`, `ggml_vulkan`); MLX is single-device. On the GGML backends each
rank owns a ggml backend on its own GPU with its own weight shards and KV
cache, and cross-GPU AllReduce goes through ggml-cuda's collective (NCCL when
available) or a host reduction for small payloads. CUDA **graph capture stays on
under TP** — a tensor-parallel token is dozens of small per-rank submissions, and
replaying them is worth ~45% of decode throughput (4×A40: Qwen 3.5-9B `--tp 4`
88 → 128.5 tok/s, Qwen 3.5-35B-A3B `--tp 2` 71.3 → 104.1, the latter being the
difference between TP losing and winning against a single GPU). Disable with
`TS_GGML_TP_CUDA_GRAPHS=0`. The collective is chosen by
measurement, not by capability flags: at startup the group verifies that peer
copies between the advertised device pairs actually deliver their bytes and that
a real NCCL AllReduce completes, and it picks the fastest transport that passes.
Hosts that advertise peer access which never arrives (common on virtualized
cloud instances) keep the NCCL collective with peer transport disabled rather
than losing it — which matters past two GPUs, where the pinned-host pipeline
does not apply and the alternative is reducing through host RAM at every layer
boundary (measured on 4×A40: 53.5 → 75.1 tok/s decode on Qwen 3.5-9B Q8_0). GGML TP delivers both
capacity and latency: fused per-rank block graphs (attention, dense FFN, MoE
trunk, GatedDeltaNet) replaced the op-at-a-time forward, so on 2× RTX 2000 Ada
`--tp 2` decodes **1.39×** a single GPU on Gemma 4 E4B Q8_0 (51.7 vs 37.3 tok/s)
and **1.06×** on Qwen 3.5-9B Q8_0, with Gemma 4 output byte-identical to the
single-GPU run. Models that do not fit one card run only under TP:
Qwen 3.5-35B-A3B IQ4_XS (16.6 GB) splits across two 16 GB cards at 184 tok/s
prefill / 18 tok/s decode. Full measurements: `TENSOR_PARALLELISM_PLAN.md`
(Stages 1b and 1c).

How far that generalizes depends on the interconnect and on how much of a layer
can be split at all. On hosts without NVLink the two AllReduces per layer
dominate: GLM-5.2 UD-IQ2_XXS on 3× RTX PRO 6000 (PCIe) runs pp2048 505.6 /
tg64 17.6 tok/s under `--tp 3` against 915.9 / 43.9 on the plain single-node
layer split, and every rank holds a full-length cache, which drops the fitted
context from 342,272 to 91,136 tokens. TP there is a capacity feature
rather than a latency one — and because
it changes the reduction order, a 2-bit MoE reproduces 3 of the 6 recorded
llama.cpp goldens where the layer split reproduces 5 of 6. (Against llama.cpp
running on the same backend, the layer split is 6/6.)

**Layer split on Qwen 3.8 Flash Next.** `--tp N` on `qwen4exp` runs a layer
split, not tensor parallelism: none of
its weights are sharded, its decode is one persisted single-device GGML graph
per token, and its GDN/PLE recurrent state lives in device buffers owned by a
single backend. It is also the same (and only) multi-GPU mode llama.cpp offers
this architecture — `-sm row` refuses to load it.

Measured on 2x A100-80GB, Qwen3.8-Flash-Next-UD-Q2_K_XL (73.4 GiB):

- greedy output is **byte-identical** between the 1-GPU and the 2-GPU run
  (same SHA-256);
- VRAM 24.2 GB + 26.2 GB — roughly half the model on each card instead of all
  of it on one;
- throughput unchanged: prefill ~1520-1550 t/s and decode ~56 t/s either way.
  For reference, llama.cpp on the same box: 1 GPU pp1536 1094 / tg128 61.2;
  2 GPUs `-sm layer` 1200 / 61.5 — so llama.cpp also gains ~10% prefill and
  ~0 decode from the second card.

`TS_Q4E_LAYER_SPLIT=20,28` overrides the automatic balance with explicit layer
counts per GPU (llama.cpp's `--tensor-split` in spirit) and throws rather than
silently ignoring a value it cannot honour — useful because the automatic
balance prices weights and cannot see the vision tower, which loads later and
lands on GPU 0. Details: [Qwen 3.8 Flash Next card](docs/models/qwen38-flash-next.md).

Architectures that support neither tensor parallelism nor a layer split now say
so on stderr and run on one GPU, instead of accepting `--tp N`, printing a
tensor-parallelism banner and leaving the extra GPUs holding a CUDA context and
NCCL buffers while idle.

Batched/continuous-batching forward under TP is implemented for Mistral 3;
MoE models fall back to per-sequence forward under TP.

Local collectives prefer CUDA peer-to-peer DMA, but the group self-tests every
peer-capable device pair at startup and permanently demotes any pair whose
round-trip comes back corrupt (seen on some L4 PCIe topologies), so hosts
without working P2P — A16 vGPU profiles, most consumer cards — fall back to host
staging automatically. Diagnostic overrides: `TENSORSHARP_TP_DISABLE_P2P=1`
(stage every cross-GPU copy through host memory) and
`TENSORSHARP_TP_HOST_ALLREDUCE=1` (run the local AllReduce on the CPU).
Multi-node connect and receive windows are tuned with
`TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS` (default 120 s) and
`TENSORSHARP_TP_RECV_TIMEOUT_SECONDS` (default 300 s).

The server also supports optional **Redis-backed shared state**: a shared KV
cache tier (`--redis-url` / `TS_KV_CACHE_REDIS_URL`) for cross-session KV reuse,
and a Redis-backed Responses API store (`TS_RESPONSES_STORE_REDIS_URL`) for
durable response storage.

Full configuration reference and examples: [Usage → Tensor Parallelism & Distributed Inference](USAGE.md#tensor-parallelism--distributed-inference).

## Tool Calling / Function Calling

Models can invoke user-defined tools and participate in multi-turn tool-call conversations. Define tools as JSON and pass them via `--tools` (console) or the `tools` parameter in the API.

Each architecture uses its own wire format for tool calls:

- **Nemotron-H:** `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **Qwen 3.5/3.6-family:** the same `<tool_call>` block, but with an XML body — `<function=NAME><parameter=key>value</parameter></function>` (the JSON form is still accepted)
- **Gemma 4:** `<|tool_call>call:function_name{args}<tool_call|>`
- **Muse-Glimmer:** ATEM XML — `<atem:function_calls><atem:invoke name="NAME"><atem:parameter name="key">value</atem:parameter></atem:invoke></atem:function_calls>`, routed on the assistant's `to=` recipient header
- **GPT OSS (Harmony):** tools are declared as a TypeScript namespace in the developer message, and calls are emitted on the commentary channel as `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>`
- **DeepSeek V4:** DSML markup — the system prompt teaches the syntax and carries one JSON schema per function, and the model answers with `<｜DSML｜tool_calls><｜DSML｜invoke name="NAME"><｜DSML｜parameter name="key" string="true|false">value</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>`. `string="false"` marks a JSON-typed argument
- **GLM 5.x:** XML with per-argument tags — `<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>`; the function name follows the opening tag directly as bare text, and every argument is its own `<arg_key>` / `<arg_value>` pair (values the template rendered with `tojson` are parsed back into numbers, arrays and objects)

The output parser (`OutputParser.cs`) automatically extracts tool calls from the model's raw output regardless of architecture.

## Agent Skills

A skill is a folder holding a `SKILL.md` — YAML frontmatter plus Markdown instructions written for the model — together with the scripts, reference documents and assets those instructions refer to. TensorSharp scans one or more skill directories (`--skills-dir`, or a `skills` folder next to the binary), advertises each skill's one-line description to the model, and loads the rest **only when the model asks for it**.

That on-demand part is served by built-in tools that TensorSharp executes itself, in process:

- `skills_list()` — every skill reachable in this conversation, with its description and its bundled file paths
- `skills_read(skill, path, offset)` — one page of one file; `path="SKILL.md"` is the skill's own instructions
- `skills_run(skill, path, args)` — run a bundled script. **Off by default**; `--skills-allow-exec` enables it, and it then runs **sandboxed or not at all** (`--skills-sandbox required` is the default)

Answering those calls inside the engine is what makes the feature work for clients that know nothing about skills: an ordinary OpenAI client sends `"skills": ["pdf"]` and gets back a finished completion, never a tool call it has no implementation for. The caller's *own* tools are never executed — they are returned to the caller as usual.

**Progressive disclosure.** On families that can render tool declarations and parse tool output, the prompt contains metadata only — name and description — even for explicitly selected skills. Selection scopes reach and preference; the model activates a skill by calling `skills_read(skill, "SKILL.md")`, at which point the file index is returned too. The metadata budget is about 2% of context, clamped to approximately 1,024–10,000 tokens. Bundled files are never inlined; contents come from `skills_read` in 48 KB pages. The older body budget (one quarter of context, clamped to approximately 1,024–48,000 tokens) is used only by the fallback for a family that cannot complete a tool round trip.

**Prompt shape.** The block is merged into the leading `system`/`developer` message rather than appended as a second one, which is the only injection point every chat template in the repository handles. Its bytes are a pure function of the sorted skill selection — no timestamps, paths or counters — so a conversation re-hashes identically turn to turn and the KV prefix cache keeps matching from block 0.

**Confinement.** Every path the model names is resolved through `SkillPathGuard`, which closes lexical (`..`, absolute, `~`, UNC, drive-qualified), canonical and **symlink** escapes, and confines each skill to its own directory. ZIP installs run every entry through the same guard (zip-slip), enforce size on the decompressed stream, and cap per-entry (64 MB), per-archive (256 MB), entry-count (4096) and compression ratio (200×).

**Script sandbox.** When script execution is enabled, the child runs under `sandbox-exec` on macOS and `bwrap` 0.12.0+ on Linux — network denied by default, the user's home directory unreadable, writes confined to the working directory — plus, on every platform, an interpreter allow-list, no shell, a scrubbed environment that withholds host credentials, a time limit and an output cap. That directory is the shared request/chat workspace with `--code-exec`, otherwise per-call scratch; the skill itself stays read-only. `--skills-allow-network` is the separate opt-in for these bundled scripts; it neither enables nor is enabled by `--code-exec-allow-network`. Windows bounds the process tree through a job object but cannot confine the filesystem or network, and says so: every result names what was *not* enforced. `--skills-sandbox required` (the default) refuses to run scripts on a host that cannot confine them; Windows therefore needs an explicit `--skills-sandbox preferred` opt-in for its weaker isolation.

**Model families.** Mistral 3 carries no tool declarations, while `qwen4exp` and unknown families have no parser for a structured tool round trip. TensorSharp therefore withholds skill/code tools, inlines selected skill bodies, and drops the discovery catalog for these families. Mistral 3 also drops `role: "tool"` messages, so any loop result is fed back as a user turn. Tool support requires both a declaration renderer and an output parser; model reasoning support alone is not enough.

**Structured output.** A request using JSON mode or a JSON schema suppresses the built-in tool loop and inlines selected skill instructions so that schema-constrained output is not interrupted by an internal tool call.

Selecting skills:

```bash
# CLI
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-4-E4B-it-Q8_0.gguf \
    --backend ggml_metal --skills-dir ~/skills --skill pdf --input prompt.txt

# Any chat API — /v1/chat/completions, /v1/responses, /api/chat/ollama (Ollama), /api/chat (Web UI)
curl -X POST http://localhost:5000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf",
       "messages": [{"role": "user", "content": "Pull the totals table out of this statement."}],
       "skills": ["pdf"], "skills_discovery": false}'
```

The server also exposes the registry itself — `GET /v1/skills`, `GET /api/skills`, `POST /api/skills` (upload a `.zip`), `DELETE /api/skills/{name}` — and `/api/models` reports a `skills` block (`enabled`, `installable`, `allowScripts`, `count`) so a UI knows whether to offer the control and whether script execution is available.

Full reference, including the frontmatter fields, the budget, the security model and the C# `SkillsChatClient` API: [Agent Skills in TensorSharp](docs/agent_skills.md). Open-source skills to start from: <https://github.com/anthropics/skills>.

## Multimodal Support

### Gemma 4

Gemma 4 models support image, video, and audio inputs. For the E4B example above, pass the repository's `mmproj-gemma-4-E4B-it-Q8_0.gguf` explicitly with `--mmproj` (use the projector matching other target sizes).

- **Images:** PNG, JPEG, HEIC/HEIF
- **Video:** MP4 (time-based extraction at 1 fps using OpenCV; tune with `VIDEO_SAMPLE_FPS` / `VIDEO_MAX_FRAMES`)
- **Audio:** WAV (16kHz mono), MP3, OGG Vorbis

### Qwen 3.5 / 3.6 family

All Qwen 3.5/3.6-family variants (`qwen35`, `qwen35moe`, and `qwen3next`) load through the same `Qwen35Model` implementation. Image inputs are supported via the dynamic-resolution `Qwen35VisionEncoder`; pass the selected repository's projector explicitly (for the 9B and Qwen 3.6 examples, `mmproj-F16.gguf`). The MoE variants (e.g. Qwen3.5-35B-A3B and Qwen3.6-35B-A3B GGUFs that report the same architecture keys) additionally enable a fused `MoEExpertsSwiGLUResidual` GGML kernel during decode that runs all selected experts, the optional shared expert, and the residual add in a single GPU graph dispatch.

### Qwen 3.8 Flash Next

Qwen3.8-Flash-Next (`qwen4exp`) supports image input through the Qwen3.5-VL
vision tower with (T, H, W) IMRoPE positions; put the repository's
`mmproj-BF16.gguf` beside the model to enable it. Multi-image prompts and
multi-turn image sessions both work, with KV reuse across turns — extend-only,
because the GatedDeltaNet recurrence cannot rewind, so a cached prefix is
reused only when the new prompt extends it exactly.

- **Images:** PNG, JPEG, HEIC/HEIF

### GLM-5.3-Flash

GLM-5.3-Flash (`glm5next`) supports image input through the GLM-OCR ViT in the
repository's `mmproj-BF16.gguf` — RMS norms, fused QKV, per-head q/k RMS norms,
2D vision RoPE, a SwiGLU-clamp MLP and a 2x2 conv merger, with all 24 blocks
running as one device-resident GGML graph. The projected embeddings override
the `<|image|>` placeholder rows inside the native executor; the text tower is
NoPE, so image tokens need no MRoPE bookkeeping. `--image`, multi-image prompts
and multi-turn image sessions are all supported. GLM-5.2 (`glm-dsa`) is
text-only.

- **Images:** PNG, JPEG, HEIC/HEIF

### Mistral 3

Mistral 3 supports image inputs via the Pixtral vision encoder. The example repository uses `mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf`; pass it explicitly with `--mmproj`.

- **Images:** PNG, JPEG, HEIC/HEIF

### Muse-Glimmer

Muse-Glimmer-30B supports image inputs through a 50-layer sparse-window ViT with
2D RoPE and a 2x2 pixel shuffle. Pass the companion projector explicitly with
`--mmproj` (e.g. `mmproj-Muse-Glimmer-30B-Q8_0.gguf`). The chat template renders
an image content part as a single `<|patch|>`, which the multimodal injector
expands to `<|image_start|>` + N merged-patch rows + `<|image_end|>` — up to 4096
merged tokens per image, chosen by an aspect-preserving stretch (no tiling, no
padding).

- **Images:** PNG, JPEG, HEIC/HEIF

### Nemotron-H (Omni distribution)

The Nemotron Omni distribution adds a RADIO / v2_vl ViT image encoder. Pass the matching multimodal projector with `--mmproj` (e.g. `nvidia_Nemotron-H-Omni-mmproj.gguf`); the language-model GGUF stays the same. Image tokens are inserted at `<image>` placeholders and expanded into `<img>` + N tile tokens + `</img>` automatically by the multimodal injector.

- **Images:** PNG, JPEG, HEIC/HEIF
- **Audio:** the chat template emits `<so_embedding>` per uploaded audio file and the CLI runs the Parakeet-style log-mel preprocessor for verification, but actual audio inference requires a Parakeet audio mmproj that the public GGUFs do not currently ship.
