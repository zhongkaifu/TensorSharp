# TensorSharp

<p align="center">
  <img src="imgs/banner_1.png" alt="TensorSharp logo" width="320">
</p>

[English](README.md) | [中文](README_zh-cn.md)

Native .NET LLM inference engine for GGUF models, including autoregressive LLMs and DiffusionGemma-style text-diffusion models. TensorSharp provides a console application, a web-based chatbot interface, and Ollama/OpenAI-compatible HTTP APIs for programmatic access.

## Quick Start

Zero to a streaming reply in about 30 seconds (after the model download).

**1. Prerequisites** — [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0), `git`, and (optionally) a GPU toolchain: NVIDIA → CUDA Toolkit 12.x; Apple Silicon → Xcode command-line tools (Metal is built in). Full list in [Prerequisites](#prerequisites).

**2. Clone & build** — the native GGML library is compiled automatically on the first build.

```bash
git clone https://github.com/zhongkaifu/TensorSharp.git
cd TensorSharp
dotnet build TensorSharp.slnx -c Release
```

**3. Download a model** — a small, well-tested starting point is Gemma-4-E4B (Q8_0) from [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF). More options in [Verified Models](#verified-models).

**4. Run it** — choose the `--backend` for your hardware (see [Pick a Backend](#pick-a-backend)):

```bash
# One-shot generation
echo "Explain mixture-of-experts in one sentence." > prompt.txt

./TensorSharp.Cli --model gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --backend ggml_metal   # macOS
./TensorSharp.Cli --model gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --backend ggml_cuda    # Windows/Linux + NVIDIA
./TensorSharp.Cli --model gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --backend cpu          # portable / debugging

# Interactive chat (REPL)
./TensorSharp.Cli --model gemma-4-E4B-it-Q8_0.gguf -i --backend ggml_metal
```

Prefer a browser UI plus HTTP APIs? Start the server instead:

```bash
./TensorSharp.Server --model gemma-4-E4B-it-Q8_0.gguf --backend ggml_metal
# open http://localhost:5000 — also serves Ollama- and OpenAI-compatible endpoints
```

The CLI binary lands in `TensorSharp.Cli/bin/...` and the server in `TensorSharp.Server/bin/...` after the build. Full options: [CLI usage](#console-application) · [Server usage](#web-application).

## Pick a Backend

Not sure which backend to use? Start here. Every backend falls back to CPU for any op it does not yet implement, so output stays correct on all of them.

| Your hardware | Recommended backend | Flag | Notes |
|---|---|---|---|
| **Apple Silicon (Mac)** | GGML Metal | `--backend ggml_metal` | Default on macOS. `--backend mlx` is an alternative Apple-Silicon GPU path. |
| **Windows / Linux + NVIDIA GPU** | GGML CUDA | `--backend ggml_cuda` | Most-tested NVIDIA path. `--backend cuda` is the direct PTX/cuBLAS backend for experimentation. |
| **No GPU / portability / debugging** | Pure C# CPU | `--backend cpu` | No native dependencies. For faster CPU inference use `--backend ggml_cpu` (native kernels). |

See [Compute Backends](#compute-backends) for the full description of every backend and what each one accelerates.

## Verified Models

These architectures are implemented and exercised by the test/benchmark matrix. Pick a quantization that fits your hardware (e.g. Q4_K_M for low memory, Q8_0 for higher quality). More sizes and multimodal projector files are in [Model Downloads](#model-downloads-gguf).

| Family | Example model (GGUF) | Image / Video / Audio | Thinking | Tools | Card |
|---|---|---|---|---|---|
| Gemma 4 | [gemma-4-E4B-it](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) (also 31B, 26B-A4B MoE) | ✅ / ✅ / ✅ | ✅ | ✅ | [gemma4.md](docs/models/gemma4.md) |
| Qwen 3.5 / 3.6 | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) (also 35B-A3B MoE) | ✅ / — / — | ✅ | ✅ | [qwen35.md](docs/models/qwen35.md) |
| Qwen 3 | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | — / — / — | ✅ | ✅ | [qwen3.md](docs/models/qwen3.md) |
| GPT OSS | [gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) (MoE) | — / — / — | ✅ | ✅ | [gptoss.md](docs/models/gptoss.md) |
| Nemotron-H | [Nemotron-H-8B](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) (also 47B, Omni) | ✅ (Omni) / — / — | ✅ | ✅ | [nemotron.md](docs/models/nemotron.md) |
| Mistral 3 | [Mistral-Small-3.1-24B](https://huggingface.co/bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF) | ✅ / — / — | — | — | [mistral3.md](docs/models/mistral3.md) |
| Gemma 3 | [gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf) | ✅ / — / — | — | — | [gemma3.md](docs/models/gemma3.md) |
| DiffusionGemma | diffusion-gemma GGUFs | — / — / — | — | — | [diffusiongemma.md](docs/models/diffusiongemma.md) |

## Highlights

- **Faster prefill & time-to-first-token than llama.cpp** — a pure-.NET engine that *out-prefills* the hand-tuned C++ `llama.cpp` on **every** tested model, on identical GGUF files and the same GPU (prefill geomean **1.18×–1.88×**), with decode held at parity. The Gemma 4 26B-A4B MoE leads at **1.88× prefill / 1.69× TTFT**; structured-output (JSON) prefill hits **5.89×** and multi-turn chat lands first tokens **up to 1.93× sooner**. → [Head-to-head vs llama.cpp](#head-to-head-vs-llamacpp-engine-comparison)
- **Continuous batching & paged KV cache** — vLLM-style paged KV pool with block-hash prefix sharing and an iteration-level scheduler, on by default in the server. → [deep dive](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md)
- **MTP / NextN speculative decoding** — multi-token-prediction draft heads accelerate solo decode on Qwen 3.6 (NextN block embedded in the trunk GGUF) and Gemma 4 (separate `gemma4-assistant` draft GGUF). The draft proposes several tokens per step and the trunk verifies them in one batched forward, with the request's own sampler driving both. Opt in with `--mtp-spec` (+ `--mtp-draft-model` for Gemma 4). → [Speculative decoding](#mtp--nextn-speculative-decoding)
- **DiffusionGemma text diffusion** — block-wise EntropyBound denoising over a Gemma-4-derived MoE backbone, with CLI generation flags and a Web UI denoising preview stream. → [DiffusionGemma card](docs/models/diffusiongemma.md)
- **Multimodal** — image / video / audio inputs (Gemma 4); image inputs for Gemma 3, Qwen 3.5-family, Mistral 3, and Nemotron-H Omni. → [Multimodal Support](#multimodal-support)
- **Tool calling / function calling** — multi-turn tool calls across all three API styles, with architecture-agnostic output parsing. → [Tool Calling](#tool-calling--function-calling)
- **Thinking / reasoning mode** — structured chain-of-thought for Qwen 3, Qwen 3.5/3.6-family, Gemma 4, GPT OSS, and Nemotron-H. → [Thinking Mode](#thinking--reasoning-mode)
- **Ollama- & OpenAI-compatible APIs** — drop-in endpoints for existing tooling, plus a browser chat UI. → [HTTP APIs](#http-apis)
- **Native quantized compute** — Q4_K_M / Q8_0 / MXFP4 / IQ2_XXS and more run in matmul without dequantizing to FP32.

---

Everything below is detailed reference. New here? The five sections above are all you need to get running.

## Documentation Map

| Start here | Use this when you want to... |
|---|---|
| [Quick build and usage](#building) | Build the solution, compile the native GGML bridge, and run the CLI or server |
| [Supported model architectures](#supported-model-architectures) | Check which GGUF architecture keys, modalities, thinking mode, and tool calling paths are implemented |
| [Compute backends](#compute-backends) | Choose between pure C# CPU, direct CUDA/cuBLAS, MLX Metal, GGML CPU, GGML Metal, and GGML CUDA |
| [HTTP APIs](#http-apis) | Use the Ollama-compatible, OpenAI-compatible, or Web UI SSE endpoints |
| [Per-model architecture cards](docs/models/README.md) | Read end-to-end documentation of one architecture (origin, forward graph, components, parameters, and how TensorSharp implements / optimizes prefill and decode) |
| [Paged attention & continuous batching](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md) | Understand the vLLM-style paged KV cache, prefix sharing, and iteration-level scheduler |
| [Environment variable feature matrix](docs/env_var_feature_matrix.md) | See which high-impact runtime flags affect which models, backends, and prompt types |
| [Test/benchmark matrix runner](TensorSharp.TestMatrix/README.md) | Sweep model × backend × feature × env-var cells and generate regression reports |
| [Server API examples](TensorSharp.Server/API_EXAMPLES.md) | Copy complete curl and Python examples for the server surface |
| [Server integration tests](TensorSharp.Server/testdata/README.md) | Exercise the public API contract against a running server |

## Current Status

| Area | Status |
|---|---|
| Model families | Gemma 3/4, DiffusionGemma, Qwen 3, Qwen 3.5/3.6-family GGUFs (`qwen35`, `qwen35moe`, `qwen3next`), GPT OSS, Nemotron-H (incl. Nemotron 3 Nano Omni), and Mistral 3 |
| Inference hosts | CLI, interactive REPL, ASP.NET Core web UI, Ollama-style API, OpenAI Chat Completions-style API. DiffusionGemma currently uses the CLI diffusion run mode and Web UI denoising stream. |
| Backends | Pure C# CPU, direct CUDA/cuBLAS (`cuda`), MLX Metal (`mlx`), GGML CPU, GGML Metal, GGML CUDA |
| Multimodal | Gemma 4 image/video/audio; Gemma 3, Qwen 3.5-family, Mistral 3, and Nemotron-H Omni image input |
| Continuous batching | vLLM-style paged KV cache, block-hash prefix sharing across requests, iteration-level scheduler (enabled by default; opt-out via `--no-continuous-batching`) |
| Speculative decoding | MTP / NextN draft heads for solo decode on Qwen 3.6 (embedded NextN) and Gemma 4 (separate `gemma4-assistant` draft GGUF); off by default, opt-in via `--mtp-spec` (+ `--mtp-draft-model` for Gemma 4). Profitable on ggml backends and the pure-C# `cuda` backend; CPU / MLX stay on standard decode. |
| Server model scope | One explicitly hosted GGUF via `--model`; optional explicit projector via `--mmproj`; no directory scanning |
| Observability | Structured per-turn logs, queue status, and KV-cache reuse metrics across Web UI, Ollama, and OpenAI response shapes |
| Test/eval harness | `TensorSharp.TestMatrix` sweeps supported hosts across model, backend, feature, and env-var cells, then compares against per-host baselines |

## Features

- **Multi-architecture support** -- Gemma 4, Gemma 3, DiffusionGemma, Qwen 3, Qwen 3.5/3.6-family, GPT OSS, Nemotron-H, Mistral 3
- **Multimodal inference** -- image, video, and audio inputs (Gemma 4); images for Gemma 3 / Qwen 3.5-family / Mistral 3 / Nemotron-H Omni
- **Thinking / reasoning mode** -- structured chain-of-thought output with `<think>` / `<|channel>thought` / `<|channel>analysis` tags (Qwen 3, Qwen 3.5/3.6-family, Gemma 4, GPT OSS, Nemotron-H)
- **Tool calling / function calling** -- models can invoke user-defined tools; multi-turn tool-call conversations supported across all three API styles
- **Quantized model support** -- loads GGUF files with Q4_K_M, Q8_0, F16, MXFP4, and other quantization formats; performs native quantized matmul without dequantizing to FP32, including memory-efficient pure C# CPU loading for large GGUFs
- **GPU-accelerated** -- GGML Metal on macOS, GGML CUDA on Windows/Linux with NVIDIA GPUs, a direct CUDA/cuBLAS backend with PTX kernels, and an MLX backend for Apple Silicon (mlx-c / Metal), all with CPU fallbacks for unsupported ops
- **Optimized pure C# CPU backend** -- managed GEMM fast paths plus fused SIMD kernels for RMSNorm, RoPE, softmax, fused activations, and other inference hot paths
- **Continuous batching & paged KV cache** -- vLLM-style block-paged KV pool with block-hash prefix sharing across requests, iteration-level scheduler that admits / preempts sequences mid-batch, optional SSD-backed tier for very large KV working sets, and a native fused paged-attention kernel (`TSGgml_PagedAttentionForward`) that drives `ggml_flash_attn_ext` on Metal/CUDA. Enabled by default in `TensorSharp.Server`; opt-out with `--no-continuous-batching`. See [docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md).
- **MTP / NextN speculative decoding** -- multi-token-prediction draft heads accelerate solo (non-concurrent) decode. Qwen 3.6 ships its NextN block fused into the trunk GGUF; Gemma 4 loads a separate EAGLE-style `gemma4-assistant` draft GGUF via `--mtp-draft-model` whose draft layers attend the target's own KV cache. The draft proposes up to `--mtp-draft` tokens per step (kept while draft confidence ≥ `--mtp-pmin`) and the trunk verifies them in a single batched forward; the request's own sampler — penalties included — drives both drafting and verification, so output is identical to standard decode. Opt in with `--mtp-spec` (off by default). On ggml backends fused multi-token-verify / draft-step kernels make it a clear win; the pure-C# `cuda` backend runs a fully GPU-resident per-op verify/draft and is also a win. CPU / MLX stay on standard decode. Env: `TS_MTP_*` (shared) and `TS_GMTP_*` (Gemma 4 tuning).
- **Batched / parallel inference** -- `IBatchedPagedModel.ForwardBatch` implementations for Mistral 3, Gemma 4, GPT OSS, Qwen 3, Qwen 3.5/3.6-family, and Nemotron-H all run by default and pack N sequences into a single forward pass with paged K/V scatter and per-sequence attention via the native kernel. Each model exposes a `TS_<FAMILY>_BATCHED=0` escape hatch (e.g. `TS_GEMMA4_BATCHED=0`, `TS_QWEN35_BATCHED=0`, `TS_GPTOSS_BATCHED=0`, `TS_NEMOTRON_BATCHED=0`) to fall back to the per-sequence KV-swap path for A/B comparison or regression isolation.
- **Ollama & OpenAI API compatibility** -- drop-in replacement endpoints for existing tooling
- **Configurable sampling** -- temperature, top-k, top-p, min-p, repetition/presence/frequency penalties, seed, stop sequences
- **Chat templates** -- auto-loaded from GGUF metadata (Jinja2), with hardcoded fallbacks per architecture
- **Inference engine** -- the new `InferenceEngine` (worker-thread scheduler + paged block pool) replaces the legacy single-request FIFO queue inside `TensorSharp.Server`. The old queue object is now a compatibility shim for status/event shapes; the engine itself handles concurrency.
- **Batch processing** -- JSONL input support in the console application, plus a built-in inference benchmark for prefill/decode throughput
- **Streaming** -- token-by-token output via SSE (web) or stdout (console), with abort/stop support for in-flight generations
- **Text-diffusion generation** -- DiffusionGemma uses an iterative EntropyBound denoising sampler instead of autoregressive `Forward()`. The CLI exposes `--diffusion-steps`, `--diffusion-seed`, and `--diffusion-blocks`; the Web UI streams whole-message `replace` events for live denoising previews and batches concurrent diffusion requests through `DiffusionBatchScheduler`.
- **Hybrid SSM-Transformer** -- Nemotron-H mixes Mamba2 SSM layers, attention-only layers, and MoE FFN layers in a single model. The Mamba2 step has both a per-sequence native kernel and a batched native kernel (`TSGgml_NemotronMamba2BatchedStepF32`, NEON SIMD + GCD parallelism) used by the batched path.
- **Hybrid Attention-Recurrent** -- Qwen 3.5/3.6-family models mix full-attention layers with GatedDeltaNet recurrent layers; the batched path keeps recurrent running state in a per-slot recurrent-state pool
- **Mixture of Experts** -- Gemma 4 MoE variants (e.g. gemma-4-26B-A4B), GPT OSS MoE (e.g. gpt-oss-20b), Qwen 3.5/3.6-family MoE (`qwen35moe` / `qwen3next` variants such as Qwen3.5-35B-A3B), and Nemotron-H MoE FFN layers
- **Batched GPU MoE** -- a single fused GGML graph dispatch handles all selected experts (plus the optional shared expert and residual add) for Qwen 3.5/3.6-family and Nemotron-H decode, eliminating per-expert round-trips
- **KV cache codecs** -- pluggable codec interface (`IKvBlockCodec`) with a built-in TurboQuant (Q4 / Q8) compressed codec for paged blocks, configurable via `--paged-kv-quant-bits`
- **Message editing** -- edit or delete previous messages in the web chat UI and regenerate from that point
- **Text/Image/Audio/Video uploads** -- the web UI accepts file uploads up to 500 MB, with automatic token-budget-aware truncation for large text files
- **Per-turn observability** -- structured logs capture the full user input and the full raw assistant output (both `<think>` reasoning and the final result) plus the KV cache hit ratio. The same cache-hit stats are surfaced through every API: `prompt_cache_hit_tokens` / `prompt_cache_hit_ratio` (Ollama), `usage.prompt_tokens_details.cached_tokens` (OpenAI), and `promptTokens` / `kvReusedTokens` / `kvReusePercent` in the Web UI SSE `done` event

## Supported Model Architectures

| Architecture | GGUF arch keys | Example Models | Multimodal | Thinking | Tool Calling | MTP spec | Card |
|---|---|---|---|---|---|---|---|
| Gemma 4 | `gemma4` | gemma-4-E4B, gemma-4-31B, gemma-4-26B-A4B (MoE) | Image, Video, Audio | Yes | Yes | Yes (separate `gemma4-assistant` draft GGUF) | [gemma4.md](docs/models/gemma4.md) |
| Gemma 3 | `gemma3` | gemma-3-4b | Image | No | No | — | [gemma3.md](docs/models/gemma3.md) |
| Qwen 3 | `qwen3` | Qwen3-4B | Text only | Yes | Yes | — | [qwen3.md](docs/models/qwen3.md) |
| Qwen 3.5 / 3.6 family | `qwen35`, `qwen35moe`, `qwen3next` | Qwen3.5-9B (hybrid Attn+Recurrent), Qwen3.5-35B-A3B / Qwen3.6-35B-A3B (MoE) | Image | Yes | Yes | Yes on Qwen 3.6 (embedded NextN block) | [qwen35.md](docs/models/qwen35.md) |
| GPT OSS | `gptoss`, `gpt-oss` | gpt-oss-20b (MoE) | Text only | Yes (always) | Yes | — | [gptoss.md](docs/models/gptoss.md) |
| Nemotron-H | `nemotron_h`, `nemotron_h_moe` | Nemotron-H-8B, Nemotron-H-47B (Hybrid SSM-Transformer, MoE), Nemotron 3 Nano Omni (image) | Image (Omni) | Yes | Yes | — | [nemotron.md](docs/models/nemotron.md) |
| Mistral 3 | `mistral3` | Mistral-Small-3.1-24B-Instruct | Image | No | No | — | [mistral3.md](docs/models/mistral3.md) |
| DiffusionGemma | `diffusion-gemma`, `diffusion_gemma` | diffusion-gemma text-diffusion GGUFs | Text only | No | No | — | [diffusiongemma.md](docs/models/diffusiongemma.md) |

See the [per-model architecture cards](docs/models/README.md) for end-to-end documentation of each architecture (origin, forward graph, components, parameters, weight naming, and how TensorSharp implements / optimizes prefill and decode).

## Model Downloads (GGUF)

TensorSharp loads models in GGUF format. Below are Hugging Face links where you can download GGUF files for each supported architecture. Pick a quantization that fits your hardware (Q4_K_M for low memory, Q8_0 for higher quality, etc.).

| Architecture | Model | GGUF Download |
|---|---|---|
| Gemma 4 | gemma-4-E4B-it | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) |
| Gemma 4 | gemma-4-31B-it | [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) |
| Gemma 4 | gemma-4-26B-A4B-it (MoE) | [ggml-org/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) |
| Gemma 4 | gemma-4-mmproj (multimodal projector) | Included in the GGUF repos above |
| Gemma 3 | gemma-3-4b-it | [google/gemma-3-4b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf) |
| Qwen 3 | Qwen3-4B | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) |
| Qwen 3.5 / 3.6 family | Qwen3.5-9B | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) |
| Qwen 3.5 / 3.6 family | Qwen3.5-35B-A3B | [ggml-org/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF) |
| GPT OSS | gpt-oss-20b (MoE) | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) |
| Nemotron-H | Nemotron-H-8B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron-H-47B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) |
| Mistral 3 | Mistral-Small-3.1-24B-Instruct | [bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF) |
| Mistral 3 | mistral3-mmproj (Pixtral vision projector) | [bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF) |
| DiffusionGemma | diffusion-gemma text-diffusion GGUFs | Use GGUF files whose `general.architecture` is `diffusion-gemma` or `diffusion_gemma` |

## Compute Backends

| Backend | Flag | Best fit | Description |
|---|---|---|---|
| Direct CUDA/cuBLAS | `--backend cuda` | NVIDIA inference and experimentation | Uses the CUDA Driver API, cuBLAS GEMM, PTX kernels for common float32 ops (fill, unary, binary, ternary, activations, RMSNorm, softmax, RoPE/RoPEEx, SDPA, GQA prefill/decode, causal mask, gather/concat), and native quantized matmul/get-rows for supported GGUF quant types. Unsupported ops route through CPU fallbacks while preserving tensor semantics. |
| MLX Metal | `--backend mlx` | Apple Silicon (alternative to GGML Metal) | GPU-accelerated path built on [mlx-c](https://github.com/ml-explore/mlx-c). Implements quantized ops (Q4_K_M, Q8_0, Q5_K, Q6_K, IQ2_XXS, IQ4_XS, IQ4_NL, MXFP4, etc.) without dequantizing to FP32, fused decode/prefill Metal kernels (fused QKV preprocess, fused gate+up+SiLUMul MoE, fused multi-dim KV write), compiled-graph kernels, async worker dispatch with periodic `async_eval` to overlap GPU/CPU work, batched MoE decode with stacked expert weight slabs, MoE expert offload, GGUF mmap pinned in physical RAM via `mlock(2)`, host-derived allocator caps (`TS_MLX_MEMORY_LIMIT_MB` / `TS_MLX_CACHE_LIMIT_MB` / `TS_MLX_WIRED_LIMIT_MB`), and a CPU fallback for ops that aren't yet wired up. Requires `libmlxc` (built locally by `TensorSharp.Backends.MLX/build-native-macos.sh` or located via `TENSORSHARP_MLX_LIBRARY` / `TENSORSHARP_MLX_LIBRARY_DIR`). |
| GGML Metal | `--backend ggml_metal` | Apple Silicon (default on macOS) | GPU-accelerated via Apple Metal. Quantized weights are mapped zero-copy from the GGUF file into Metal command buffers via host-pointer buffers, so the resident set stays close to the on-disk model size. |
| GGML CUDA | `--backend ggml_cuda` | NVIDIA inference through ggml | GPU-accelerated via GGML CUDA on Windows or Linux. Quantized weights are uploaded to device memory once at load time and the host copy is released afterwards. |
| GGML CPU | `--backend ggml_cpu` | Native CPU kernels | CPU inference using native GGML with optimized kernels. Quantized weights are mapped zero-copy from the GGUF file. |
| Pure C# CPU | `--backend cpu` | Portability and debugging | Portable CPU inference with no native dependencies. |

## Project Structure

```
TensorSharp/
├── TensorSharp.Core/            # Core tensor library (Tensor, Ops, memory, device abstraction, CPU SIMD/managed quantized kernels)
├── TensorSharp.Runtime/         # GGUF, tokenizers, templates, sampling, protocol parsing
│   ├── Paged/                   # Paged KV cache primitives (BlockPool, BlockTable, KvBlock, BlockHashIndex, PagedKvStorage, PagedKvBatchOps, ManagedPagedAttention)
│   ├── Scheduling/              # Continuous batching engine (InferenceEngine, BatchExecutor, ContinuousBatchScheduler, SequenceState, SchedulerConfig/Output, InferenceRequestHandle) + MTP speculative decode core (MtpSpeculativeExecution)
│   ├── PagedKvCacheManager.cs   # Per-session paged KV manager (block allocation, prefix reuse)
│   ├── PagedKvBlockStore.cs     # On-disk / RAM-tiered paged block storage with optional SSD spillover
│   ├── SsdKvBlockTier.cs        # SSD-backed cold tier for paged blocks
│   ├── TurboQuantKvCodec.cs     # Quantized KV block codec (Q4 / Q8) implementing IKvBlockCodec
│   ├── PrefillChunking.cs       # Chunked-prefill helper used by SWA / very long prompts
│   ├── KvBlockHash.cs           # Content-addressed block hash for prefix-cache sharing
│   └── Logging/                 # JSON-line file logger + per-turn telemetry
├── TensorSharp.Models/          # Model architectures and multimodal encoders/injectors
│   ├── Models/<Family>/         # One folder per architecture (DiffusionGemma, Gemma3, Gemma4, GptOss, Mistral3, Nemotron, Qwen3, Qwen35)
│   │   ├── <Family>Model.cs                # Legacy per-sequence ModelBase implementation
│   │   └── <Family>Model.BatchedForward.cs # IBatchedPagedModel.ForwardBatch — batched/paged path (Mistral3, Gemma4, GptOss, Qwen35, Nemotron, Qwen3)
│   ├── Paged/                   # Tensor-side paged-attention helpers (TensorPagedAttention)
│   ├── KvBlockTransfer.cs       # Helpers for extract/inject of KV blocks across sequences
│   ├── MtpSpeculativeDecoder.cs # MTP/NextN draft-verify-rollback driver shared by Qwen 3.6 and Gemma 4
│   └── ModelMultimodalInjector.cs # Vision / audio / video embedding injection
├── TensorSharp.Backends.GGML/   # GGML backend bindings (Metal/CUDA/CPU via native library)
├── TensorSharp.Backends.Cuda/   # Direct CUDA backend using CUDA Driver API, cuBLAS, and PTX kernels
├── TensorSharp.Backends.MLX/    # Apple Silicon MLX backend (mlx-c / Metal). Native bridge is built via `build-native-macos.sh`.
├── TensorSharp.GGML.Native/     # Native C++ bridge to ggml (builds libGgmlOps, split into focused source files)
│   ├── ggml_ops_core.cpp                  # Element-wise, reductions, basic shape ops
│   ├── ggml_ops_elementwise.cpp           # Element-wise / activation fusions
│   ├── ggml_ops_matmul.cpp                # GEMM / quantized matmul
│   ├── ggml_ops_fused.cpp                 # Cross-cutting fused per-layer kernels
│   ├── ggml_ops_norm_attn.cpp             # Norm + attention fusions
│   ├── ggml_ops_transformer.cpp           # Full-layer fused transformer kernels (decode + prefill)
│   ├── ggml_ops_moe.cpp                   # Mixture-of-Experts forward / fused router
│   ├── ggml_ops_gated_delta_net.cpp       # Qwen 3.5/3.6 GatedDeltaNet kernels (per-seq + batched)
│   ├── ggml_ops_mamba2.cpp                # Nemotron Mamba2 kernels (per-seq + batched SIMD)
│   ├── ggml_ops_paged_attention.cpp       # Paged-attention native kernel (drives ggml_flash_attn_ext + sinks variant)
│   ├── ggml_ops_diffusion.cpp             # DiffusionGemma fused decode-layer / whole-model / lm-head kernels
│   ├── ggml_ops_training.cpp              # Training-only kernels (unused at runtime)
│   └── tests/                              # Native unit + smoke tests
├── TensorSharp.Server/          # Web chatbot + API server (ASP.NET Core)
│   ├── Program.cs               # Slim bootstrap: DI wiring, middleware, endpoint mapping, paged-KV + continuous-batching CLI translation
│   ├── ModelService.cs          # Facade that keeps the public server inference API stable; owns the InferenceEngineHost
│   ├── ModelLifecycleService.cs # Model load/dispose and backend selection (CPU / CUDA / MLX / GGML CPU/Metal/CUDA)
│   ├── InferenceEngineHost.cs   # DI-registered per-model InferenceEngine singleton (continuous batching entry point)
│   ├── ChatGenerationPipeline.cs # Prompt rendering, submits to InferenceEngine, streams tokens, stop handling
│   ├── InferenceTelemetry.cs    # Prompt/eval timing, TTFT, tokens/sec, full input/output logs
│   ├── ChatHistoryPreparer.cs   # History normalization, raw-token splice helpers, multimodal order helpers
│   ├── ChatSession.cs           # Per-conversation tracked history + raw assistant tokens
│   ├── SessionManager.cs        # Thread-safe session registry (default + per-tab sessions)
│   ├── InferenceQueue.cs        # Backward-compatible queue-status surface (engine itself handles concurrency)
│   ├── BackendCatalog.cs        # Discovery of available compute backends (CPU / CUDA / MLX / GGML*)
│   ├── TextUploadHelper.cs      # Token-budget-aware text-file truncation
│   ├── WebUiChatPolicy.cs       # Web UI chat request validation
│   ├── OpenAIResponseFormatParser.cs  # OpenAI response_format (json_object / json_schema) parsing
│   ├── Hosting/                 # Startup-time concerns: options builder (ServerOptionsBuilder), backend resolution, logging, web root, paged-KV / continuous-batching CLI translation
│   ├── RequestParsers/          # JSON request parsing (sampling, chat messages, tool functions)
│   ├── ResponseSerializers/     # Per-protocol response shape factories (Ollama, OpenAI, Web UI)
│   ├── StreamingWriters/        # SSE + NDJSON wire-format helpers
│   ├── ProtocolAdapters/        # Per-protocol request handlers (WebUiAdapter, OllamaAdapter, OpenAIChatAdapter)
│   ├── Endpoints/               # ASP.NET Core endpoint mapping (one extension method per protocol)
│   ├── Logging/                 # Request logging middleware + low-noise path support
│   ├── wwwroot/index.html       # Chat UI
│   ├── testdata/                # Integration test suites (bash + Python)
│   └── API_EXAMPLES.md          # Detailed API documentation
├── TensorSharp.Cli/             # CLI application (one-shot generation, interactive REPL, batch JSONL, benchmarks)
├── TensorSharp.TestMatrix/      # Test / benchmark matrix runner, default prompts, env-var sweeps, and per-host baselines
├── InferenceWeb.Tests/          # xUnit unit tests covering ops, KV cache, paged scheduler, batched-model correctness, web/server helpers
├── AdvUtils/                    # Utility library (logger)
├── docs/                        # Developer reference
│   ├── models/                  # Per-model architecture cards (one .md per model, EN + 中文)
│   ├── PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md  # Paged KV cache, prefix sharing, scheduler, per-model batched-forward status
│   └── env_var_feature_matrix.md  # Runtime flag × model/backend/feature coverage for TestMatrix
├── benchmarks/                  # Reproducible benchmark harnesses
└── ExternalProjects/            # ggml/ is cloned from github.com/ggml-org/ggml at build time (not committed)
```

## NuGet Packages

The repository is now split along package boundaries so consumers can depend on only the layers they actually need.

| Project | NuGet package | Public namespace | Responsibility |
|---|---|---|---|
| `TensorSharp.Core` | `TensorSharp.Core` | `TensorSharp` | Tensor primitives, ops, allocators, storage, and device abstraction |
| `TensorSharp.Runtime` | `TensorSharp.Runtime` | `TensorSharp.Runtime` | GGUF parsing, tokenizers, prompt rendering, sampling, output protocol parsing, paged KV cache, continuous-batching scheduler |
| `TensorSharp.Models` | `TensorSharp.Models` | `TensorSharp.Models` | `ModelBase`, architecture implementations, multimodal encoders, batched / paged forward passes, and model-side execution helpers |
| `TensorSharp.Backends.GGML` | `TensorSharp.Backends.GGML` | `TensorSharp.GGML` | GGML-backed execution and native interop |
| `TensorSharp.Backends.Cuda` | `TensorSharp.Backends.Cuda` | `TensorSharp.Cuda` | Direct CUDA allocator, storage, cuBLAS GEMM, PTX kernels, and quantized CUDA ops |
| `TensorSharp.Backends.MLX` | `TensorSharp.Backends.MLX` | `TensorSharp.MLX` | Apple Silicon MLX backend (mlx-c / Metal) with quantized / fused / compiled kernels and MoE expert offload |
| `TensorSharp.Server` | `TensorSharp.Server` | `TensorSharp.Server` | ASP.NET Core server, OpenAI/Ollama adapters, inference engine host, web UI |
| `TensorSharp.Cli` | `TensorSharp.Cli` | `TensorSharp.Cli` | Console host and debugging / batch tooling |

This split keeps engine users off the web stack, keeps API-layer changes from leaking into core/runtime packages, and makes future benchmark or eval-harness projects easier to publish independently.

Validate package metadata and README dependency boundaries before publishing:

```powershell
pwsh ./eng/verify-packages.ps1
```

The verifier runs `dotnet pack` for the public packages above and fails if an internal dependency such as `AdvUtils` leaks into the `.nuspec`, or if a TensorSharp package depends on a layer outside this table.

### Publishing a release

Publishing is automated by the [`Publish NuGet`](.github/workflows/publish-nuget.yml) GitHub Actions workflow. Push a version tag and the workflow packs all of the public packages above and pushes them to **NuGet.org** and the repository's **GitHub Packages** feed:

```bash
git tag v2.8.6      # tag drives the package version (v2.8.6 -> 2.8.6)
git push origin v2.8.6
```

- The tag (with the leading `v` stripped) overrides `TensorSharpVersion` for every package, so all packages ship with a single coordinated version. You do not need to edit `Directory.Build.props` first.
- Packing is managed-only — the native GGML/CUDA/MLX libraries are not embedded in the packages — so the workflow runs on a stock runner with `eng/verify-packages.ps1 -SkipNativeBuild` (which also sets `TensorSharpSkipGgmlNative=true` / `TensorSharpSkipMlxNative=true`).
- Configure a `NUGET_API_KEY` repository secret for the NuGet.org push. If it is missing, the NuGet.org step is skipped with a warning and only the GitHub Packages push (which uses the built-in `GITHUB_TOKEN`) runs.
- To rehearse without publishing, run the workflow manually (`workflow_dispatch`) with a `version` input and `dry_run` checked — it packs, verifies, and uploads the `.nupkg` files as a build artifact without pushing.

### Platform binary releases

Alongside the managed NuGet packages, the [`Release Binaries`](.github/workflows/release-binaries.yml) workflow builds self-contained, ready-to-run archives of **TensorSharp.Server** and **TensorSharp.Cli** for each platform and attaches them to the GitHub Release for the tag. Each archive bundles the .NET 10 runtime and the platform's native libraries, so it runs without a separate .NET install or native build:

| Archive suffix | Native backend(s) bundled | Format |
|---|---|---|
| `win-x64-cpu` | GGML CPU | `.zip` |
| `win-x64-cuda` | GGML CUDA + pure-C# CUDA (PTX) + CUDA 12.x runtime | `.zip` |
| `linux-x64-cpu` | GGML CPU | `.tar.gz` |
| `linux-x64-cuda` | GGML CUDA + pure-C# CUDA (PTX) + CUDA 12.x runtime | `.tar.gz` |
| `osx-arm64` | GGML Metal + MLX | `.tar.gz` |

- Pushing a `v*` tag builds the archives and publishes the Release automatically — both this and the NuGet workflow trigger on the same tag.
- The `-cuda` archives bundle the CUDA runtime libraries (`cudart` / `cublas` / `cublasLt`) but still require an NVIDIA GPU and a compatible driver at run time; the `-cpu` archives run anywhere. The macOS archive requires Apple Silicon.
- To rehearse, run the workflow manually (`workflow_dispatch`) with a `version` input — it builds every platform and creates a **draft** Release. Override the target GPU architectures for the CUDA build with the `cuda_arch` input.

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- **`git` and network access:** the GGML/CUDA native builds clone the ggml sources from [github.com/ggml-org/ggml](https://github.com/ggml-org/ggml) into `ExternalProjects/ggml/` on first build (see `eng/fetch-ggml.sh` / `eng/fetch-ggml.ps1`). The clone tracks ggml's default branch (`master`); pin a different ref with `TENSORSHARP_GGML_GIT_REF`, or set `TENSORSHARP_GGML_NO_UPDATE=1` to skip the network update once cloned (offline rebuilds)
- **macOS (Metal backend):** CMake 3.20+ and Xcode command-line tools for building the native GGML library; the MLX backend additionally builds `libmlxc` from `TensorSharp.Backends.MLX/Native/` via `bash TensorSharp.Backends.MLX/build-native-macos.sh`
- **Windows (GGML CPU / CUDA backends):** CMake 3.20+ and Visual Studio 2022 C++ build tools; for `ggml_cuda` or `cuda`, install an NVIDIA driver plus CUDA Toolkit 12.x or another compatible CUDA toolkit with cuBLAS
- **Linux (GGML CPU / CUDA backends):** CMake 3.20+; for `ggml_cuda` or `cuda`, install an NVIDIA driver plus CUDA Toolkit 12.x or another compatible CUDA toolkit with cuBLAS
- GGUF model files (e.g., from [Hugging Face](https://huggingface.co))

## Building

### Build the entire solution

```bash
dotnet build TensorSharp.slnx
```

### Build individual applications

```bash
# Console application
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj

# Web application
dotnet build TensorSharp.Server/TensorSharp.Server.csproj
```

### Build the native GGML library

The native library is built automatically during the first `dotnet build` if it doesn't exist. To build it manually:

```bash
cd TensorSharp.GGML.Native
```

macOS:

```bash
bash build-macos.sh
```

Linux (CPU-only):

```bash
bash build-linux.sh
```

Linux (GGML_CUDA enabled):

```bash
bash build-linux.sh --cuda
```

Windows (CPU-only):

```powershell
.\build-windows.ps1 --no-cuda
```

Windows (GGML_CUDA enabled):

```powershell
.\build-windows.ps1 --cuda
```

On Windows and Linux, the native build script auto-detects the visible NVIDIA GPU compute capability and passes a narrow `CMAKE_CUDA_ARCHITECTURES` value to ggml-cuda (for example `86-real` on an RTX 3080), which cuts CUDA build time substantially. The native build also runs in parallel by default with a conservative job cap so `nvcc` does not overwhelm typical developer machines.

If you want to override the auto-detected architecture list or the default build parallelism, use either environment variables or explicit build flags:

```bash
TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES='86-real;89-real' bash build-linux.sh --cuda
bash build-linux.sh --cuda --cuda-arch='86-real;89-real'
TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL=2 bash build-linux.sh --cuda
```

```powershell
$env:TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES='86-real;89-real'; .\build-windows.ps1 --cuda
.\build-windows.ps1 --cuda --cuda-arch='86-real;89-real'
$env:TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL=2; .\build-windows.ps1 --cuda
```

You can also request a CUDA-enabled native build from `dotnet build`:

```bash
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
```

```powershell
$env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA='ON'; dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
```

On macOS this compiles `libGgmlOps.dylib` with Metal GPU support. On Windows and Linux, the native scripts preserve an existing CUDA-enabled build and auto-enable GGML_CUDA when a CUDA toolchain is detected; `build-windows.ps1 --cuda`, `build-linux.sh --cuda`, and `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` force CUDA explicitly. The build output is automatically copied to the application's output directory.

The direct `cuda` backend is built as managed C# plus PTX kernels. During `dotnet build`, `TensorSharp.Backends.Cuda` compiles `native/kernels/*.cu` to `native/ptx/*.ptx` when `nvcc` is available; if `nvcc` is missing, the build continues and PTX-backed ops use CPU fallbacks. cuBLAS-backed GEMM still requires the CUDA runtime libraries to be discoverable at run time.

### Build the native MLX library (macOS only)

The MLX backend depends on `libmlxc` (the C bindings for [MLX](https://github.com/ml-explore/mlx)). The repository pins a known-good tag of `mlx-c` in `TensorSharp.Backends.MLX/Native/MLX_C_VERSION` and a helper script fetches and builds it:

```bash
bash TensorSharp.Backends.MLX/build-native-macos.sh
```

The script writes the resulting libraries (`libmlxc.dylib`, `libmlx.dylib`, and any backend deps) into `TensorSharp.Backends.MLX/Native/dist/`. At run time the backend probes the application directory first; you can also point it to a custom install with `TENSORSHARP_MLX_LIBRARY=<path-to-libmlxc.dylib>` or `TENSORSHARP_MLX_LIBRARY_DIR=<dir-with-libmlxc>`. If the library cannot be located the backend reports unavailable and `--backend mlx` is rejected at startup.

## Usage

### Console Application

```bash
cd TensorSharp.Cli/bin

# Text inference
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_metal

# Text inference on Windows/Linux + NVIDIA GPU
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_cuda

# Interactive turn-by-turn chat (REPL) with KV cache reuse and slash commands
./TensorSharp.Cli --model <model.gguf> --backend ggml_metal --interactive
./TensorSharp.Cli --model <model.gguf> --backend ggml_metal -i \
    --system "You are a terse assistant." --temperature 0.7 --top-p 0.9 --think

# Image inference (Gemma 3/4, Qwen 3.5-family)
./TensorSharp.Cli --model <model.gguf> --image photo.png --backend ggml_metal

# Video inference (Gemma 4)
./TensorSharp.Cli --model <model.gguf> --video clip.mp4 --backend ggml_metal

# Audio inference (Gemma 4)
./TensorSharp.Cli --model <model.gguf> --audio speech.wav --backend ggml_metal

# DiffusionGemma text-diffusion generation
./TensorSharp.Cli --model <diffusion-gemma.gguf> --input prompt.txt --backend ggml_metal \
    --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0

# Thinking / reasoning mode
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --backend ggml_metal --think

# Tool calling
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --tools tools.json

# With sampling parameters
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.2 --seed 42

# Batch processing (JSONL)
./TensorSharp.Cli --model <model.gguf> --input-jsonl requests.jsonl \
    --output results.txt --backend ggml_metal

# Multi-turn chat simulation with KV-cache reuse (mirrors the web UI behavior)
./TensorSharp.Cli --model <model.gguf> --multi-turn-jsonl chat.jsonl \
    --backend ggml_metal --max-tokens 200

# Throughput benchmark: best-of-N prefill and decode timing
./TensorSharp.Cli --model <model.gguf> --backend ggml_metal \
    --benchmark --bench-prefill 256 --bench-decode 128 --bench-runs 3

# KV-cache reuse benchmark: measure prefill speedup across multiple chat turns
# (compares with-cache vs forced-reset prefill latency for an 8-turn conversation)
./TensorSharp.Cli --model <model.gguf> --backend ggml_metal \
    --bench-kvcache --bench-kv-turns 4 --max-tokens 64

# Inspect the rendered prompt and tokenization without running inference
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --dump-prompt

# Compare hardcoded fallback templates against GGUF Jinja2 templates for every
# *.gguf file in a directory (useful when adding new architectures)
./TensorSharp.Cli --test-templates ~/models
```

**Command-line options:**

| Option | Description |
|---|---|
| `--model <path>` | Path to a GGUF model file (required) |
| `--input <path>` | Text file containing the user prompt |
| `--input-jsonl <path>` | JSONL file with batch requests (one JSON per line) |
| `--multi-turn-jsonl <path>` | JSONL file for multi-turn chat simulation with KV cache reuse |
| `--output <path>` | Write generated text to this file |
| `--image <path>` | Image file for vision inference |
| `--video <path>` | Video file for video inference |
| `--audio <path>` | Audio file (WAV, MP3, OGG) for audio inference |
| `--mmproj <path>` | Path to the multimodal projector GGUF file |
| `--max-tokens <N>` | Maximum tokens to generate (default: 100) |
| `--backend <type>` | Compute backend: `cpu`, `cuda`, `mlx`, `ggml_cpu`, `ggml_metal`, or `ggml_cuda` |
| `--kv-cache-dtype <type>` | KV cache precision: `f32` (default), `f16`, or `q8_0`. Quantized / half-precision KV caches reduce memory at the cost of small numerical drift. |
| `--interactive` / `-i` | Start an interactive REPL chat session (turn-by-turn input/output) with KV cache reuse, slash commands, hot-swappable model/backend/projector, file attachments (image, audio, video, text) and live sampling tuning. See the **Interactive REPL commands** section below for the full list. |
| `--system <text>` | System prompt to seed the interactive session (overridden inside the REPL by `/system`) |
| `--system-file <path>` | Read the initial system prompt from a UTF-8 text file (alternative to `--system`) |
| `--think` | Enable thinking/reasoning mode (chain-of-thought) |
| `--tools <path>` | JSON file with tool/function definitions |
| `--temperature <f>` | Sampling temperature (0 = greedy) |
| `--top-k <N>` | Top-K filtering (0 = disabled) |
| `--top-p <f>` | Nucleus sampling threshold (1.0 = disabled) |
| `--min-p <f>` | Minimum probability filtering (0 = disabled) |
| `--repeat-penalty <f>` | Repetition penalty (1.0 = none) |
| `--presence-penalty <f>` | Presence penalty (0 = disabled) |
| `--frequency-penalty <f>` | Frequency penalty (0 = disabled) |
| `--seed <N>` | Random seed (-1 = non-deterministic) |
| `--stop <string>` | Stop sequence (can be repeated) |
| `--dump-prompt` | Render the prompt + tokenization and exit (no generation) |
| `--benchmark` | Run a synthetic prefill/decode throughput benchmark |
| `--bench-prefill <N>` | Synthetic prefill length in tokens (default: 32) |
| `--bench-decode <N>` | Synthetic decode length in tokens (default: 64) |
| `--bench-runs <N>` | Number of benchmark runs; reports best and average (default: 1) |
| `--bench-kvcache` | Run a multi-turn KV-cache reuse benchmark (with-cache vs forced-reset prefill) |
| `--bench-kv-turns <N>` | Number of conversation turns for `--bench-kvcache` (default: 4, max: 8) |
| `--bench-chunked` | Run a chunked-prefill micro-benchmark (Gemma 4) |
| `--warmup-runs <N>` | Number of throw-away forward passes before timing real text / multimodal prompts (default: 0) |
| `--test-chunked-prefill` | Run the chunked-prefill correctness check (compares chunked vs non-chunked logits) |
| `--correct-prefill <N>` | Prompt length used by `--test-chunked-prefill` |
| `--correct-decode <N>` | Decode length used by `--test-chunked-prefill` |
| `--diffusion-steps <N>` | DiffusionGemma denoising steps per block (default: 48) |
| `--diffusion-seed <N>` | DiffusionGemma deterministic sampler seed (default: 0) |
| `--diffusion-blocks <N>` | DiffusionGemma block-autoregressive canvas count. `0` derives the count from `--max-tokens` and the model canvas length. |
| `--test` | Run built-in tokenizer + Qwen3 chat-template + ollama-comparison tests |
| `--test-templates <dir>` | Validate hardcoded chat templates against GGUF Jinja2 templates for every *.gguf in `<dir>` |
| `--log-level <lvl>` | Console + file logger level: `trace`, `debug`, `info`, `warning`, `error`, `critical`, `off` |
| `--log-dir <path>` | Directory for the JSON-line file logger (default: `<binDir>/logs`) |
| `--log-file <0\|1>` | Disable (`0`) or enable (`1`) the file logger (default: enabled) |
| `--log-console <0\|1>` | Disable (`0`) or enable (`1`) the console logger (default: enabled) |

The multimodal projector file is auto-detected if placed alongside the model file with a recognized name (e.g., `gemma-4-mmproj-F16.gguf`).

**JSONL input format:**

Each line is a JSON object with `messages`, optional `prompt`, and optional sampling parameters:

```json
{"id": "q1", "messages": [{"role": "user", "content": "What is 2+3?"}], "max_tokens": 50}
{"id": "q2", "messages": [{"role": "user", "content": "Write a haiku."}], "max_tokens": 100, "temperature": 0.8}
```

**Interactive REPL commands:**

Once the CLI is launched with `--interactive` / `-i`, you can drive the running session with slash commands. Type `/help` (or `/?`) inside the REPL for the same list. Anything that does not start with `/` is treated as a user turn.

The prompt header summarizes the current state on every turn — model, backend, architecture, context length, projector, conversation depth, and any attachments queued for the next turn (e.g. `[turn 3 (2 attachments pending)]> `). Press Ctrl+C while generating to interrupt the current reply; press Ctrl+C at the prompt to exit.

Conversation:

| Command | Description |
|---|---|
| `/help`, `/?` | Show all interactive commands |
| `/exit`, `/quit` | Leave the session |
| `/reset`, `/new` | Clear conversation history and KV cache |
| `/history` | Print the conversation history |
| `/save <file>` | Append the current transcript to a UTF-8 file |
| `/system <text>` | Set the system prompt (empty argument clears it). Resets KV cache. |
| `/think on\|off` | Toggle thinking/reasoning mode for supported models |
| `/multiline on\|off` | Toggle multi-line input (terminate the message with a single `.` on its own line) |

Model and runtime:

| Command | Description |
|---|---|
| `/info`, `/status` | Show the loaded model, backend, architecture, context/vocab size, projector, conversation depth, and pending attachments |
| `/model <path>` | Load a different `.gguf` model on the current backend (resets the session) |
| `/backend <name>` | Reload the current model on a different backend: `cpu`, `cuda`, `mlx`, `ggml_cpu`, `ggml_metal`, or `ggml_cuda` |
| `/mmproj <path>` | Load (or replace) the multimodal projector for the current model. Aliases: `/projector` |

Sampling (live, persists across turns):

| Command | Description |
|---|---|
| `/sampling`, `/show` | Print the current sampling configuration |
| `/max <N>` | Maximum reply length in tokens |
| `/temp <float>` | Sampling temperature (0 = greedy) |
| `/topk <int>` | Top-K filtering (0 = disabled) |
| `/topp <float>` | Top-P / nucleus threshold (1.0 = disabled) |
| `/minp <float>` | Min-P filtering (0 = disabled) |
| `/repeat <float>` | Repetition penalty (1.0 = none) |
| `/presence <float>` | Presence penalty |
| `/frequency <float>` | Frequency penalty |
| `/seed <int>` | Random seed (-1 = non-deterministic) |
| `/stop <text>` | Add a stop sequence |
| `/clearstop` | Remove all stop sequences |

Uploads (queued for the next user turn, then auto-cleared after the turn):

| Command | Description |
|---|---|
| `/image <path>`, `/img <path>` | Attach an image (vision-capable models only) |
| `/audio <path>` | Attach an audio file (Gemma 4) |
| `/video <path>`, `/vid <path>` | Attach a video; frames are extracted automatically (Gemma 4) |
| `/text <path>`, `/file <path>`, `/txt <path>` | Inline a UTF-8 text/markdown/csv/code file into the next prompt (large files are token-budget truncated) |
| `/clearattach` | Drop any pending image/audio/video/text attachments without sending a turn |

Quoted paths (single or double quotes) are accepted, so drag-and-drop from a file manager works on macOS. Multimodal commands require a multimodal projector to be loaded — pass `--mmproj` at startup or use `/mmproj <path>` from the REPL.

### Web Application

```bash
cd TensorSharp.Server/bin

# Start the server with the exact hosted model
./TensorSharp.Server --model ./models/model.gguf --backend ggml_metal

# Linux + NVIDIA GPU
./TensorSharp.Server --model ./models/model.gguf --backend ggml_cuda

# Multimodal models: host an explicit projector too
./TensorSharp.Server --model ./models/model.gguf --mmproj ./models/mmproj.gguf --backend ggml_cuda

# Configure server-wide default sampling parameters
# (used whenever a request does not override the value itself)
./TensorSharp.Server --model ./models/model.gguf --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 \
    --presence-penalty 0.0 --frequency-penalty 0.0 --seed 42 \
    --stop "</s>" --stop "<|endoftext|>"
```

Open `http://localhost:5000` in your browser. The web interface supports:

- Multi-turn chat conversations
- Per-tab chat sessions: each browser tab owns its own tracked conversation history; KV blocks are owned by the inference engine
- A single hosted GGUF selected explicitly with `--model`
- An explicit hosted multimodal projector via `--mmproj` when needed
- Image, video, and audio uploads for multimodal inference (up to 500 MB)
- Thinking/reasoning mode toggle
- Tool calling with function definitions
- Streaming token generation via Server-Sent Events
- DiffusionGemma denoising previews when a `diffusion-gemma` GGUF is hosted (the UI replaces the whole assistant message on each denoising step, then emits the final answer)
- Backward-compatible queue-status events (the engine itself handles concurrency)
- Message editing and deletion with regeneration from any point in the conversation
- Free scrolling: scroll up to read earlier replies while new tokens stream in; the chat auto-scrolls again as soon as the user scrolls back to the bottom

Use `--model` to choose the hosted GGUF file and `--mmproj` to choose the hosted projector. `TensorSharp.Server` no longer scans a `MODEL_DIR`.

**Server command-line options:**

| Option | Description |
|---|---|
| `--model <path>` | GGUF file to host (required for inference; if omitted, the server starts but `/api/models/load` will report no hosted model) |
| `--mmproj <path>` | Multimodal projector GGUF (resolved relative to the model directory when only a filename is given; pass `none` to disable). Requires `--model`. |
| `--backend <type>` | Default compute backend: `cpu`, `cuda`, `mlx`, `ggml_cpu`, `ggml_metal`, or `ggml_cuda` |
| `--max-tokens <N>` | Default maximum tokens to generate when a request omits the limit (default: `20000`) |
| `--temperature <f>` | Default sampling temperature when a request does not provide one (`0` = greedy) |
| `--top-k <N>` | Default top-K filtering when a request does not provide one (`0` = disabled) |
| `--top-p <f>` | Default nucleus sampling threshold when a request does not provide one (`1.0` = disabled) |
| `--min-p <f>` | Default min-p filtering when a request does not provide one (`0` = disabled) |
| `--repeat-penalty <f>` | Default repetition penalty when a request does not provide one (`1.0` = none) |
| `--presence-penalty <f>` | Default presence penalty when a request does not provide one (`0` = disabled) |
| `--frequency-penalty <f>` | Default frequency penalty when a request does not provide one (`0` = disabled) |
| `--seed <N>` | Default random seed when a request does not provide one (`-1` = non-deterministic) |
| `--stop <string>` | Default stop sequence (can be repeated). Per-request `stop`/`stop_sequences` fully replace the default list rather than merge with it. |
| `--continuous-batching` / `--no-continuous-batching` | Enable (default) or disable iteration-level paged-batching. When enabled the server admits / preempts sequences mid-batch and packs them into one forward pass on models that implement `IBatchedPagedModel`. `--no-continuous-batching` falls back to per-sequence KV-swap for every model. Alias: `--paged-batching` / `--no-paged-batching`. |
| `--mtp-spec` / `--no-mtp-spec` | Enable NextN/MTP speculative decoding (default off) on models that ship a multi-token-prediction draft head (Qwen 3.6's embedded NextN block, or a Gemma 4 `gemma4-assistant` draft loaded via `--mtp-draft-model`). Engages for solo (non-concurrent) sequences: the draft head proposes up to `--mtp-draft` tokens per step and the trunk verifies them in one batched forward, with the request's own sampler (penalties included) driving both drafting and verification, so output matches standard decode. Engaged automatically only where profitable (ggml backends and the pure-C# `cuda` backend); CPU / MLX serve standard decode. Env: `TS_MTP_SPEC`. |
| `--mtp-draft <N>` | Maximum tokens drafted per speculative step (default `8`). Env: `TS_MTP_DRAFT`. |
| `--mtp-pmin <f>` | Minimum draft-head confidence in `(0, 1]` for a drafted token to be kept; drafting stops at the first low-confidence token (default `0.75`). Env: `TS_MTP_PMIN`. |
| `--mtp-draft-model <path>` | Path to a separate MTP draft GGUF for architectures whose draft head ships as its own file (Gemma 4's `gemma4-assistant`). The draft's hidden size must match the target (e.g. pair the 12B target with its 12B draft, not the 26B-A4B draft); a mismatched or incomplete draft fails fast at startup with a remediation hint. Ignored for Qwen 3.6, which embeds its NextN block in the trunk GGUF. Env: `TS_MTP_DRAFT_MODEL`. |
| `--paged-kv` / `--no-paged-kv` | Legacy compatibility flags for the removed per-session paged-KV manager. Current server KV state is engine-owned; use continuous-batching / `TS_SCHED_*` knobs for the engine. Aliases: `--paged-kv-cache` / `--no-paged-kv-cache`. |
| `--paged-kv-block-size <N>` | Legacy standalone paged-KV block size. The current server engine uses `TS_SCHED_BLOCK_SIZE`. |
| `--paged-kv-ram-mb <N>` | Legacy standalone paged-KV RAM-tier cap. |
| `--paged-kv-ssd-dir <dir>` | Legacy standalone paged-KV SSD cold-tier directory. |
| `--paged-kv-ssd-mb <N>` | Legacy standalone paged-KV SSD cap. |
| `--paged-kv-quant-bits <0\|4\|8>` | Legacy standalone paged-KV block quantization (TurboQuantKvCodec). |

Per-request fields in the chat / generate JSON payloads (e.g. `temperature`,
`top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`,
`frequency_penalty`, `seed`, `stop`/`stop_sequences`) always win over these
server-wide defaults; the defaults only fill in fields the client omits.

**Runtime environment variables:**

| Variable | Description |
|---|---|
| `BACKEND` | Default compute backend (`cpu`, `cuda`, `mlx`, `ggml_cpu`, `ggml_metal`, or `ggml_cuda`), used when `--backend` is not passed (default: `ggml_metal` on macOS, `ggml_cpu` elsewhere) |
| `MAX_TOKENS` | Default maximum generation length when neither `--max-tokens` nor a request-level limit is set (default: `20000`) |
| `MAX_TEXT_FILE_CHARS` | Character cap used to truncate plain-text uploads when no tokenizer is available (default: `8000`) |
| `VIDEO_SAMPLE_FPS` | Frames sampled per second of video for video prompts; time-based extraction (default: `1`) |
| `VIDEO_MAX_FRAMES` | Optional upper bound on extracted video frames (evenly down-sampled); unset/`0` means no cap (default: no cap) |
| `PORT` / `ASPNETCORE_URLS` | Currently overridden by the fixed `http://0.0.0.0:5000` listener in `Program.cs`; Docker Space images rewrite that constant with `APP_PORT` at build time. |
| `TENSORSHARP_TEMPERATURE` | Default sampling temperature when neither `--temperature` nor the request body sets one |
| `TENSORSHARP_TOP_K` | Default top-K when neither `--top-k` nor the request body sets one |
| `TENSORSHARP_TOP_P` | Default top-P when neither `--top-p` nor the request body sets one |
| `TENSORSHARP_MIN_P` | Default min-P when neither `--min-p` nor the request body sets one |
| `TENSORSHARP_REPEAT_PENALTY` | Default repetition penalty when neither `--repeat-penalty` nor the request body sets one |
| `TENSORSHARP_PRESENCE_PENALTY` | Default presence penalty when neither `--presence-penalty` nor the request body sets one |
| `TENSORSHARP_FREQUENCY_PENALTY` | Default frequency penalty when neither `--frequency-penalty` nor the request body sets one |
| `TENSORSHARP_SEED` | Default random seed when neither `--seed` nor the request body sets one |
| `TENSORSHARP_LOG_LEVEL` | Minimum log level for both console and file loggers: `Trace`, `Debug`, `Information`, `Warning`, `Error`, `Critical` (default: `Information`). Also honored by `TensorSharp.Cli`. |
| `TENSORSHARP_LOG_DIR` | Directory the JSON-line file logger writes to (default: `<binDir>/logs`). Also honored by `TensorSharp.Cli`. |
| `TENSORSHARP_LOG_FILE` | Set to `0` to disable the file logger and keep only the console output (default: enabled). Also honored by `TensorSharp.Cli`. |
| `DIFFUSION_STEPS` | Server-side DiffusionGemma denoising steps per block (default: `48`; CLI equivalent is `--diffusion-steps`) |
| `DIFFUSION_MAX_BATCH` | Maximum concurrent DiffusionGemma requests batched by the Web UI diffusion scheduler (default: `2`) |

**Paged KV cache & continuous-batching tunables (read at process / model start)**

These can be set with either the `--paged-kv*` / `--continuous-batching` CLI flags (which translate to the env vars below) or directly via the environment:

| Variable | Description |
|---|---|
| `TS_KV_PAGED_CACHE` | Legacy compatibility switch for the standalone `PagedKvCacheManager`; current `TensorSharp.Server` request KV state is engine-owned. The CLI shortcuts are `--paged-kv` / `--no-paged-kv`. |
| `TS_KV_BLOCK_SIZE` | Legacy standalone paged-KV block size. The engine uses `TS_SCHED_BLOCK_SIZE`. |
| `TS_KV_CACHE_MAX_RAM_MB` | Legacy standalone paged-KV RAM-tier cap. |
| `TS_KV_CACHE_SSD_DIR` | Legacy standalone paged-KV SSD cold-tier directory. |
| `TS_KV_CACHE_MAX_SSD_MB` | Legacy standalone paged-KV SSD cap. |
| `TS_KV_PAGED_QUANT_BITS` | Legacy standalone paged-KV block quantization bits (`0` = passthrough, `4`, or `8`). |
| `TS_SCHED_DISABLE_BATCHED` | `1` forces the per-sequence KV-swap fallback even when a model implements `IBatchedPagedModel`. The CLI shortcut is `--no-continuous-batching`. |
| `TS_SCHED_MAX_BATCHED_TOKENS` | Scheduler per-step token budget (default: `4096`). |
| `TS_SCHED_MAX_RUNNING_SEQS` | Maximum in-flight sequences (default: `16`). |
| `TS_SCHED_PREFILL_CHUNK` | Maximum prefill tokens per step (default: `1024`). |
| `TS_SCHED_NUM_BLOCKS` | Physical blocks in the engine block pool (default: `256`). |
| `TS_SCHED_BLOCK_SIZE` | Tokens per block on the engine side (default: `256`). |
| `TS_SCHED_PREFIX_CACHE` | `0` disables block-hash prefix sharing across requests. |
| `TS_SCHED_DECODE_QUANTUM` | Tokens before a sequence-switch is allowed (default: block size). |
| `TS_QWEN35_BATCHED` | Set to `0` to force the Qwen 3.5/3.6 family onto the legacy per-sequence KV-swap path (default: batched/paged). Also implicitly disabled by `--no-continuous-batching`. |
| `TS_QWEN35_BATCHED_GDN_NATIVE` | Use the native batched GatedDeltaNet kernel inside Qwen 3.5/3.6 batched path. |
| `TS_GEMMA4_BATCHED` | Set to `0` to force Gemma 4 onto the legacy per-sequence KV-swap path (default: batched/paged). |
| `TS_GPTOSS_BATCHED` | Set to `0` to force GPT OSS onto the legacy per-sequence KV-swap path (default: batched/paged). |
| `TS_GPTOSS_PAGED_ATTN_MANAGED` | Use the managed (C#) paged-attention-with-sinks kernel inside GPT OSS batched path. |
| `TS_NEMOTRON_BATCHED` | Set to `0` to force Nemotron-H onto the legacy per-sequence KV-swap path (default: batched/paged). |
| `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE` | Use the native Mamba2 batched step kernel inside Nemotron-H batched path. |
| `TS_PAGED_ATTN_KERNEL` | Paged-attention dispatch kernel for `Mistral3Model.BatchedForward`: `native` (default), `tensor` (C# Tensor-based), or `managed` (pure C# scalar). |
| `TS_MLX_PIPELINED_DECODE` | `1` (default) enables pipelined greedy decode on the MLX backend when the request is greedy, has no stop sequences, and the model supports device-side argmax / next-embedding lookup. Set to `0` to disable. CLI only. |
| `TS_MLX_MLOCK_GGUF` | `1` (default) pins the GGUF mmap region in physical RAM via `mlock(2)` so model weights stay resident between forward passes. Set to `0` to skip (use if the process `memlock` rlimit is too low or you want the OS to manage paging). MLX backend only. |
| `TS_MLX_FUSED_KV_WRITE` | `1` (default) uses a single multi-dim `slice_update` to write the per-token KV block. Set to `0` to revert to the per-head loop (A/B testing / regression isolation). |
| `TS_MLX_BATCHED_MOE_DECODE` | `1` (default) collapses K per-expert decode dispatches to one batched dispatch per (gate/up/down) kind for Qwen 3.5/3.6 MoE. Set to `0` on memory-constrained machines (saves ~weight-doubling overhead from the stacked weight slabs). |
| `TS_MLX_MOE_FUSED_GATE_UP_SILU` | `1` (default) fuses gate matmul + up matmul + SiLUMul into one Metal kernel for batched MoE decode. Set to `0` to A/B against the legacy 3-dispatch path. |
| `TS_MLX_DEVICE_ROUTER` | `1` (default) keeps MoE router top-K + softmax on device to skip ~60 host syncs/token on Qwen 3.6-35B-A3B. Set to `0` to disable; the code also falls back automatically when prerequisites are missing. |
| `TS_MLX_LOG_MEMORY_POLICY` | `1` (default) prints once-per-load MLX memory-policy lines (wired limit, GGUF mlock status, allocator caps). Set to `0` to silence. |
| `TS_MLX_MEMORY_LIMIT_MB` / `TS_MLX_CACHE_LIMIT_MB` / `TS_MLX_WIRED_LIMIT_MB` | Override the MLX allocator hard cap / unused-buffer cache cap / wired-buffer residency cap (megabytes). Defaults are derived from the host's unified-memory capacity. |
| `TS_MLX_EVAL_EVERY_N_LAYERS` / `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS` | Periodic `mlx_async_eval` cadence during decode to overlap GPU work with host queueing. Gemma 4 defaults to every 4 layers via `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS`; Qwen 3 / Qwen 3.5 / Nemotron-H default to every 16 layers via `TS_MLX_EVAL_EVERY_N_LAYERS`. Set to `0` to disable where supported. |
| `TENSORSHARP_MLX_LIBRARY` / `TENSORSHARP_MLX_LIBRARY_DIR` | Override the search path for `libmlxc` when using `--backend mlx`. |

**MTP / speculative-decoding tunables**

These gate the optional multi-token-prediction speculative decode path (see [MTP / NextN speculative decoding](#mtp--nextn-speculative-decoding)). `TS_MTP_*` are the shared knobs (also set by the `--mtp-*` CLI flags); `TS_GMTP_*` are Gemma 4 draft-path A/B switches.

| Variable | Description |
|---|---|
| `TS_MTP_SPEC` | `1` enables MTP/NextN speculative decoding for solo sequences (default `0`). CLI: `--mtp-spec` / `--no-mtp-spec`. |
| `TS_MTP_DRAFT` | Maximum tokens drafted per speculative step (default `8`). CLI: `--mtp-draft`. |
| `TS_MTP_PMIN` | Minimum draft-head confidence in `(0, 1]` to keep a drafted token (default `0.75`). CLI: `--mtp-pmin`. |
| `TS_MTP_DRAFT_MODEL` | Path to the separate Gemma 4 `gemma4-assistant` draft GGUF. CLI: `--mtp-draft-model`. Ignored by Qwen 3.6 (embedded NextN). |
| `TS_GMTP_NO_FUSED` | `1` disables the Gemma 4 fused multi-token-verify / draft-step GGML kernels and falls back to the per-op path (A/B testing on ggml backends). |
| `TS_GMTP_NO_FAST_ROLLBACK` | `1` restores the kept-prefix rollback path instead of the dense exact-match fast rollback used on partial draft acceptance. |
| `TS_GMTP_BATCHED_TRUNK` | `1` opts the Gemma 4 verify trunk back into the batched paged path; the default runs the faster linear trunk for solo speculation. |

**DiffusionGemma-specific tunables**

| Variable | Description |
|---|---|
| `DIFFUSION_NO_SC` | Set to `1` to disable self-conditioning. Enabled by default. |
| `DIFFUSION_SC_TOPK` | Experimental self-conditioning top-K cutoff (default: `32`). |
| `DIFFUSION_NO_PKV` | Set to `1` to disable prompt-KV caching on device-glue backends. Enabled by default where supported. |
| `DIFFUSION_NO_FUSED_DECODE` | Set to `1` to disable the GGML fused model decode path and fall back to per-op / per-layer diffusion decode. |
| `DIFFUSION_NO_FUSED_LMHEAD_TAIL` | Set to `1` to disable the fused output-norm + lm-head + softcap tail. |
| `DIFFUSION_BATCHED_FORWARD` | Set to `1` to use true batched `DecodeCanvasBatched` for active diffusion canvases; default time-slices the faster fused single-canvas path. |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | Memory cap for batched diffusion lm-head logits before falling back to per-sequence lm-head (default: `300`). |
| `DIFFUSION_PROFILE` / `DIFFUSION_STEPTIME` / `DIFFUSION_FUSED_DEBUG` | Development diagnostics for per-section diffusion timing and fused-kernel debug logging. |

Sampling parameter precedence (highest wins):

1. Per-request JSON fields in the API call (e.g. `temperature`, `top_p`, `stop`).
2. Server-wide CLI flags (e.g. `--temperature`, `--top-p`, `--stop`).
3. `TENSORSHARP_*` environment variables listed above.
4. Built-in `SamplingConfig` defaults (`temperature=1.0`, `top_k=0`, `top_p=1.0`, `min_p=0`, `repeat_penalty=1.0`, presence/frequency penalties `0`, `seed=-1`, no stop sequences).

### Feature × environment variable matrix

Quick reference for which environment variables (and matching CLI flags) gate each major feature. Variables in **bold** are required to turn the feature on; everything else is a tunable for a feature that's already enabled by default.

#### Continuous batching & paged KV cache

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Continuous-batching engine (`InferenceEngine` + scheduler) | ON in `TensorSharp.Server` | `TS_SCHED_DISABLE_BATCHED=1` to force per-seq fallback | `--no-continuous-batching` / `--continuous-batching` |
| Legacy per-session paged-KV manager | removed from Server request path | `TS_KV_PAGED_CACHE` (`0` / `1`), `TS_KV_BLOCK_SIZE` retained for compatibility / standalone tests | `--paged-kv` / `--no-paged-kv`, `--paged-kv-block-size N` |
| Legacy paged-KV SSD spillover (standalone manager) | OFF | `TS_KV_CACHE_MAX_RAM_MB`, `TS_KV_CACHE_SSD_DIR`, `TS_KV_CACHE_MAX_SSD_MB` | `--paged-kv-ram-mb`, `--paged-kv-ssd-dir`, `--paged-kv-ssd-mb` |
| Legacy paged-KV block quantization (standalone manager) | OFF (`0` = passthrough) | `TS_KV_PAGED_QUANT_BITS` (`0` / `4` / `8`) | `--paged-kv-quant-bits` |
| Block-hash prefix sharing across requests | ON | `TS_SCHED_PREFIX_CACHE=0` to disable | — |
| Scheduler tunables (per-step token budget, max in-flight seqs, prefill chunk, block pool size, decode quantum) | engine defaults | `TS_SCHED_MAX_BATCHED_TOKENS`, `TS_SCHED_MAX_RUNNING_SEQS`, `TS_SCHED_PREFILL_CHUNK`, `TS_SCHED_NUM_BLOCKS`, `TS_SCHED_BLOCK_SIZE`, `TS_SCHED_DECODE_QUANTUM` | — |

#### Per-model batched / paged forward (`IBatchedPagedModel.ForwardBatch`)

| Model | Default state | Env var to flip default | Native-kernel sub-toggle |
|---|---|---|---|
| Mistral 3 | ON | — | `TS_PAGED_ATTN_KERNEL` = `native` (default) / `tensor` / `managed` |
| Gemma 4 | ON | `TS_GEMMA4_BATCHED=0` to force legacy per-seq | — |
| Qwen 3 | ON (reference port) | — | — |
| Qwen 3.5 / 3.6 family | ON | `TS_QWEN35_BATCHED=0` to force legacy per-seq (or `--no-continuous-batching`) | `TS_QWEN35_BATCHED_GDN_NATIVE=1` enables native batched GDN kernel; `FUSED_ATTN_LAYER_MIN_SEQ_LEN=N` overrides fused-attention engage threshold (default 4096) |
| GPT OSS | ON | `TS_GPTOSS_BATCHED=0` to force legacy per-seq | `TS_GPTOSS_PAGED_ATTN_MANAGED=1` forces the managed (C#) sinks softmax instead of the native paged-attention-with-sinks kernel |
| Nemotron-H | ON | `TS_NEMOTRON_BATCHED=0` to force legacy per-seq | `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` enables the native batched Mamba2 step (NEON SIMD + GCD parallelism) |
| Gemma 3 | not implemented (per-seq fallback) | — | — |
| DiffusionGemma | Separate diffusion scheduler in the Web UI path; not an `IBatchedPagedModel` autoregressive path | `DIFFUSION_MAX_BATCH`, `DIFFUSION_STEPS` | `DIFFUSION_BATCHED_FORWARD=1` enables true batched canvas decode; fused GGML decode is on by default unless disabled with `DIFFUSION_NO_FUSED_DECODE=1` |

#### MTP / NextN speculative decoding

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Speculative decode engine (solo sequences) | OFF | **`TS_MTP_SPEC=1`** | `--mtp-spec` / `--no-mtp-spec` |
| Max tokens drafted per step | `8` | `TS_MTP_DRAFT` | `--mtp-draft N` |
| Min draft-head confidence to keep a token | `0.75` | `TS_MTP_PMIN` | `--mtp-pmin X` |
| Gemma 4 separate draft GGUF (`gemma4-assistant`) | none | `TS_MTP_DRAFT_MODEL` | `--mtp-draft-model <path>` |
| Gemma 4 fused verify / draft kernels (ggml) | ON | `TS_GMTP_NO_FUSED=1` falls back to per-op | — |
| Gemma 4 dense fast rollback on partial accept | ON | `TS_GMTP_NO_FAST_ROLLBACK=1` restores kept-prefix rollback | — |
| Gemma 4 verify trunk path | linear (solo) | `TS_GMTP_BATCHED_TRUNK=1` runs the batched paged trunk | — |

#### Backends

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Default compute backend | `ggml_metal` (macOS), `ggml_cpu` (Windows/Linux) | `BACKEND` | `--backend` |
| MLX backend library lookup | probe app dir | `TENSORSHARP_MLX_LIBRARY` (full path to `libmlxc`), `TENSORSHARP_MLX_LIBRARY_DIR` (directory) | — |
| MLX pipelined greedy decode (CLI only) | ON when eligible | `TS_MLX_PIPELINED_DECODE=0` disables | — |
| MLX `mlock(2)` of GGUF mmap so weights stay resident | ON | `TS_MLX_MLOCK_GGUF=0` to disable | — |
| MLX fused multi-dim KV write (single `slice_update` per cache block) | ON | `TS_MLX_FUSED_KV_WRITE=0` to revert to per-head loop | — |
| MLX batched MoE decode (Qwen 3.5/3.6 MoE) | ON | `TS_MLX_BATCHED_MOE_DECODE=0` for legacy per-expert path | — |
| MLX fused MoE gate+up+SiLUMul Metal kernel | ON | `TS_MLX_MOE_FUSED_GATE_UP_SILU=0` for legacy 3-dispatch | — |
| MLX on-device MoE router top-K + softmax | ON when prerequisites are met | `TS_MLX_DEVICE_ROUTER=0` disables | — |
| MLX layer-boundary `async_eval` cadence | Gemma 4: every 4 layers; Qwen / Nemotron: every 16 layers | `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS=N` or `TS_MLX_EVAL_EVERY_N_LAYERS=N` (`0` = disabled where supported) | — |
| MLX allocator caps (memory / cache / wired buffer) | host-derived | `TS_MLX_MEMORY_LIMIT_MB`, `TS_MLX_CACHE_LIMIT_MB`, `TS_MLX_WIRED_LIMIT_MB` | — |
| MLX one-line memory-policy banners at load | ON | `TS_MLX_LOG_MEMORY_POLICY=0` to silence | — |

#### Sampling defaults (server-only)

These fill in fields the request body omits; per-request JSON always wins, CLI flags win over env vars.

| Sampling field | Env var | CLI equivalent |
|---|---|---|
| `temperature` | `TENSORSHARP_TEMPERATURE` | `--temperature` |
| `top_k` | `TENSORSHARP_TOP_K` | `--top-k` |
| `top_p` | `TENSORSHARP_TOP_P` | `--top-p` |
| `min_p` | `TENSORSHARP_MIN_P` | `--min-p` |
| `repeat_penalty` | `TENSORSHARP_REPEAT_PENALTY` | `--repeat-penalty` |
| `presence_penalty` | `TENSORSHARP_PRESENCE_PENALTY` | `--presence-penalty` |
| `frequency_penalty` | `TENSORSHARP_FREQUENCY_PENALTY` | `--frequency-penalty` |
| `seed` | `TENSORSHARP_SEED` | `--seed` |
| max tokens | `MAX_TOKENS` | `--max-tokens` |
| stop sequences | — (CLI / per-request only) | `--stop` (repeatable) |

#### Hosting & uploads (server-only)

| Feature | Default | Env vars |
|---|---|---|
| ASP.NET Core listener | `http://0.0.0.0:5000` | Fixed in `Program.cs`; Docker Space images rewrite it with the `APP_PORT` build arg |
| Plain-text upload character cap (when no tokenizer available) | 8000 chars | `MAX_TEXT_FILE_CHARS` |
| Video-frame extraction | 1 fps (time-based, no cap) | `VIDEO_SAMPLE_FPS`, `VIDEO_MAX_FRAMES` |
| DiffusionGemma Web UI denoising | 48 steps, max batch 2 | `DIFFUSION_STEPS`, `DIFFUSION_MAX_BATCH` |

#### Logging (server + CLI)

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Console + file log minimum level | `Information` | `TENSORSHARP_LOG_LEVEL` | `--log-level` |
| File logger output directory | `<binDir>/logs` | `TENSORSHARP_LOG_DIR` | `--log-dir` |
| File logger enabled | ON | `TENSORSHARP_LOG_FILE=0` to disable | `--log-file 0\|1` |
| Console logger enabled | ON | — | `--log-console 0\|1` (CLI only) |

#### Native build (compile-time only)

These are read by `build-linux.sh` / `build-windows.ps1` / the auto-build during `dotnet build` for `TensorSharp.GGML.Native`, not at run time.

| Feature | Default | Env vars | Build-script flag |
|---|---|---|---|
| Enable GGML CUDA in the native build | auto-detected from toolchain | `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` | `--cuda` / `--no-cuda` |
| Narrow `CMAKE_CUDA_ARCHITECTURES` list | auto-detected from visible GPU | `TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES` | `--cuda-arch='86-real;89-real'` |
| Native build parallelism cap | conservative auto-cap | `TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL` | — |

### Server Logging

The server emits one structured Information-level entry at the start and end of
every chat / generate turn, so a single grep over the log file reproduces the
full request-response audit trail without replaying any traffic.

| Event id | Emitted on | Carries |
|---|---|---|
| `ChatStarted` (1500) | `chat.start`, `generate.start`, plus per-protocol request banners | sampling config, message + attachment counts, `userInput=` (full latest user message), `fullInput=` (JSON-encoded array of EVERY message in the request: system prompts + all prior user/assistant turns + the new user message, with attachment counts), or the full prompt for `/api/generate` |
| `ChatCompleted` (1502) | `chat.complete`, `generate.complete` | token counts, KV cache reuse (`kvReused`, `kvReusePercent`), TTFT, elapsed, throughput, finish reason, full raw assistant output (reasoning + result) |
| `ChatAborted` (1503) | client disconnected mid-stream | partial output, KV reuse fraction at the time of abort |
| `KvCacheReusePlan` (1510) | per-prefix-reuse decision | `Debug`-level fine-grained breakdown (exact match / partial / full reset) |
| `HttpRequestStarted/Completed` (1100/1101) | every HTTP request | method, path, remote IP, status, duration; `/api/queue/status` is demoted to `Debug` so high-frequency UI polling does not drown out the per-turn entries |

The raw assistant output captures `<think>...</think>`, `<|channel|>analysis`,
and any other inline framing the model emits, so the log line for a single turn
contains both reasoning and the user-visible result. Combined with the
`fullInput=` field on `chat.start`, every turn is fully reproducible from the
log file alone (request inputs + raw model output). Long uploads or long
reasoning traces can produce multi-kilobyte log lines; raise the log level
(`TENSORSHARP_LOG_LEVEL=Warning`) to suppress them while still keeping the start
banner and error logs.

Sample `fullInput` payload (formatted for readability; it is emitted as a
single line in the actual log):

```json
[
  {"role":"system","content":"You are a helpful assistant."},
  {"role":"user","content":"What is the tallest mountain?"},
  {"role":"assistant","content":"Mount Everest."},
  {"role":"user","content":"How tall is it?","images":1}
]
```

The same per-turn KV cache reuse stats are surfaced through every API:

- **Web UI SSE** (`POST /api/chat`) - the `done` event carries `promptTokens`, `kvReusedTokens`, and `kvReusePercent`.
- **Ollama NDJSON** (`POST /api/generate`, `POST /api/chat/ollama`) - the final chunk and the non-streaming response carry `prompt_cache_hit_tokens` (int) and `prompt_cache_hit_ratio` (0..1).
- **OpenAI** (`POST /v1/chat/completions`) - the `usage` block carries `prompt_tokens_details.cached_tokens`, matching the OpenAI extension that existing SDKs already understand.

The Web UI footer line under each assistant message also surfaces the cache hit
inline (e.g. `187 tokens · 2.1s · 87.2 tok/s · KV 420/512 (82%)`).

### HTTP APIs

TensorSharp.Server exposes three API styles. See [API_EXAMPLES.md](TensorSharp.Server/API_EXAMPLES.md) for full documentation with curl and Python examples.

**Ollama-compatible API:**

```bash
# List models
curl http://localhost:5000/api/tags

# Generate text
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "prompt": "Hello!", "stream": false}'

# Chat
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "stream": false}'

# Chat with thinking mode
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Solve 17*23"}], "think": true, "stream": false}'

# Chat with tool calling
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "What is the weather?"}], "tools": [{"function": {"name": "get_weather", "description": "Get current weather", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"]}}}], "stream": false}'
```

**OpenAI-compatible API:**

```bash
# Chat completions
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'

# Structured outputs (OpenAI response_format)
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-Q8_0.gguf",
    "messages": [{"role": "user", "content": "Extract the city and country from: Paris, France."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "location_extraction",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "confidence": {"type": ["string", "null"]}
          },
          "required": ["city", "country", "confidence"],
          "additionalProperties": false
        }
      }
    }
  }'
```

**OpenAI Python SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="gemma-4-E4B-it-Q8_0.gguf",
    messages=[{"role": "user", "content": "What is 2+3?"}],
    max_tokens=50
)
print(response.choices[0].message.content)
```

**Queue status:**

```bash
curl http://localhost:5000/api/queue/status
# {"busy":false,"pending_requests":0,"total_processed":42}
```

## Thinking / Reasoning Mode

Models that support thinking mode (Qwen 3, Qwen 3.5/3.6-family, Gemma 4, GPT OSS, Nemotron-H) can produce structured chain-of-thought reasoning before generating the final answer. The thinking content is separated from the main response and can be displayed or hidden by the client.

- **Qwen 3 / Qwen 3.5/3.6-family / Nemotron-H:** uses `<think>...</think>` tags
- **Gemma 4:** uses `<|channel>thought\n...<channel|>` tags
- **GPT OSS:** uses Harmony format with `<|channel|>analysis` for thinking and `<|channel|>final` for the response

Enable via `--think` (console), `"think": true` (Ollama API), or the thinking toggle in the web UI.

## MTP / NextN Speculative Decoding

Some architectures ship a **multi-token-prediction (MTP / NextN) draft head** that lets `TensorSharp.Server` run lossless speculative decoding for solo (non-concurrent) sequences. The draft proposes several future tokens cheaply, the trunk verifies all of them in one batched forward, and accepted tokens are committed in a single step. Because the request's own sampler — temperature, top-k/p, and repetition/presence/frequency penalties — drives both the draft and the verify, the output is identical to standard decode; speculation only changes how many forward passes it takes to produce it.

Speculative decoding is **off by default**. Enable it on the server with `--mtp-spec` (env `TS_MTP_SPEC=1`):

```bash
# Qwen 3.6 — the NextN block is embedded in the trunk GGUF, no extra file needed
./TensorSharp.Server --model Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --backend ggml_cuda \
    --mtp-spec --mtp-draft 8 --mtp-pmin 0.75

# Gemma 4 — load the separate gemma4-assistant draft GGUF that matches the target
./TensorSharp.Server --model gemma-4-12B-it-Q4_K_M.gguf --backend ggml_cuda \
    --mtp-spec --mtp-draft-model gemma-4-12B-assistant-Q8_0.gguf
```

**Two draft-head shapes:**

- **Qwen 3.6 (embedded NextN)** — the GGUF carries one extra decoder block past the main stack (`{arch}.nextn_predict_layers`) plus the NextN projection/norm tensors. No separate file is required; `--mtp-draft-model` is ignored. The recurrent trunk state (GatedDeltaNet) is snapshotted so a partially-rejected verify batch can be rolled back.
- **Gemma 4 (separate `gemma4-assistant` GGUF)** — an EAGLE-style recurrent drafter loaded with `--mtp-draft-model`. It holds no K/V of its own: every draft layer queries the **target model's** existing per-layer KV cache (last local + last global layer), so the drafter is stateless given `(token, hidden)`. The draft's hidden size must match the target — pair the 12B target with its 12B draft, not the 26B-A4B draft. A mismatched, missing, or incomplete draft GGUF **fails fast at startup** with a remediation hint instead of silently disabling speculation.

**Where it's profitable** (engaged automatically; otherwise the engine serves standard decode):

| Backend | Qwen 3.6 | Gemma 4 |
|---|---|---|
| GGML CUDA / GGML Metal | ✅ fused multi-token-verify + draft-step kernels | ✅ fused multi-token-verify + draft-step kernels |
| Direct CUDA (`cuda`, pure C#) | ✅ GPU-resident per-op verify/draft | ✅ GPU-resident per-op verify/draft |
| CPU / GGML CPU / MLX | standard decode (verify can't keep up) | standard decode |

Tuning: `--mtp-draft` (default `8`) bounds tokens drafted per step; `--mtp-pmin` (default `0.75`) is the minimum draft-head confidence to keep a token (drafting stops at the first low-confidence token). Gemma 4 draft-path A/B switches are the `TS_GMTP_*` env vars (see the **MTP / speculative-decoding tunables** table under [Web Application](#web-application)). Per-architecture mechanics are in the [Qwen 3.5/3.6 card](docs/models/qwen35.md) and the [Gemma 4 card](docs/models/gemma4.md).

## Tool Calling / Function Calling

Models can invoke user-defined tools and participate in multi-turn tool-call conversations. Define tools as JSON and pass them via `--tools` (console) or the `tools` parameter in the API.

Each architecture uses its own wire format for tool calls:

- **Qwen 3 / Qwen 3.5/3.6-family / Nemotron-H:** `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **Gemma 4:** `<|tool_call>call:function_name{args}<tool_call|>`
- **GPT OSS (Harmony):** tools are declared as a TypeScript namespace in the developer message, and calls are emitted on the commentary channel as `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>`

The output parser (`OutputParser.cs`) automatically extracts tool calls from the model's raw output regardless of architecture.

## Multimodal Support

### Gemma 4

Gemma 4 models support image, video, and audio inputs. Place the multimodal projector (`gemma-4-mmproj-F16.gguf`) in the same directory as the model file for automatic loading.

- **Images:** PNG, JPEG, HEIC/HEIF
- **Video:** MP4 (time-based extraction at 1 fps using OpenCV; tune with `VIDEO_SAMPLE_FPS` / `VIDEO_MAX_FRAMES`)
- **Audio:** WAV (16kHz mono), MP3, OGG Vorbis

### Gemma 3

Gemma 3 supports PNG, JPEG, and HEIC/HEIF image inputs. Place its multimodal projector (`mmproj-gemma3-4b-f16.gguf`) next to the model file for automatic loading.

### Qwen 3.5 / 3.6 family

All Qwen 3.5/3.6-family variants (`qwen35`, `qwen35moe`, and `qwen3next`) load through the same `Qwen35Model` implementation. Image inputs are supported via the dynamic-resolution `Qwen35VisionEncoder`; place the projector (`Qwen3.5-mmproj-F16.gguf`) next to the model GGUF for automatic loading. The MoE variants (e.g. Qwen3.5-35B-A3B and Qwen3.6-35B-A3B GGUFs that report the same architecture keys) additionally enable a fused `MoEExpertsSwiGLUResidual` GGML kernel during decode that runs all selected experts, the optional shared expert, and the residual add in a single GPU graph dispatch.

### Mistral 3

Mistral 3 supports image inputs via the Pixtral vision encoder. Place the multimodal projector (`mistral3-mmproj.gguf`) in the same directory as the model file for automatic loading.

- **Images:** PNG, JPEG, HEIC/HEIF

### Nemotron-H (Omni distribution)

The Nemotron Omni distribution adds a RADIO / v2_vl ViT image encoder. Pass the matching multimodal projector with `--mmproj` (e.g. `nvidia_Nemotron-H-Omni-mmproj.gguf`); the language-model GGUF stays the same. Image tokens are inserted at `<image>` placeholders and expanded into `<img>` + N tile tokens + `</img>` automatically by the multimodal injector.

- **Images:** PNG, JPEG, HEIC/HEIF
- **Audio:** the chat template emits `<so_embedding>` per uploaded audio file and the CLI runs the Parakeet-style log-mel preprocessor for verification, but actual audio inference requires a Parakeet audio mmproj that the public GGUFs do not currently ship.

## Architecture

TensorSharp is structured as a layered system:

1. **TensorSharp.Core** provides the core `Tensor` type, storage abstraction, and the extensible operation registry (`Ops`). CPU implementations use `System.Numerics.Vectors` for SIMD acceleration.

2. **TensorSharp.Runtime** owns runtime-facing contracts and services: GGUF parsing, tokenization (SentencePiece / BPE), chat template rendering, configurable token sampling, output parsing, paged KV cache (`Runtime/Paged/*`), the continuous-batching scheduler / engine (`Runtime/Scheduling/*`), the `IKvBlockCodec` interface plus the `TurboQuantKvCodec` Q4/Q8 implementation, and reusable contracts such as `IModelArchitecture`, `IBatchedPagedModel`, `IPromptRenderer`, `IOutputProtocolParser`, `IMultimodalInjector`, `IKVCachePolicy`, and `IBackendExecutionPlan`.

3. **TensorSharp.Models** implements `ModelBase` plus the concrete architectures and multimodal helpers (Gemma 3/4, DiffusionGemma, Qwen 3/3.5, GPT OSS, Nemotron-H, Mistral 3). Autoregressive architectures ship the legacy per-sequence forward, and most also expose an `IBatchedPagedModel.ForwardBatch` implementation (`<Family>Model.BatchedForward.cs`) for continuous batching. DiffusionGemma is intentionally different: `Forward()` is unsupported, and generation goes through `DiffusionGemmaSampler` over fixed-length denoising canvases. Models are loaded via `ModelBase.Create()` which auto-detects the architecture from GGUF metadata.

4. **TensorSharp.Backends.GGML** registers accelerated implementations of the same operations via a native C++ bridge (`libGgmlOps` / `GgmlOps.dll`) that links against [ggml](https://github.com/ggml-org/ggml). On macOS this provides Metal GPU compute, and on Windows/Linux it can expose GGML CUDA for NVIDIA GPUs. Operations include native quantized matmul (Q4_K_M, Q8_0, etc.) without dequantizing to FP32, plus paged-attention (`TSGgml_PagedAttentionForward`, with and without attention sinks) and architecture-specific batched kernels (Mamba2, GatedDeltaNet).

5. **TensorSharp.Backends.Cuda** is the direct CUDA path. It uses the CUDA Driver API for device/context/storage management, cuBLAS for float32 GEMM, PTX kernels for hot scalar and transformer helper ops, and CPU fallbacks where native kernels are not implemented yet.

6. **TensorSharp.Backends.MLX** is the Apple Silicon MLX path. It wraps [mlx-c](https://github.com/ml-explore/mlx-c) (`libmlxc`) with allocator, storage, async worker dispatch, quantized + fused + compiled kernels, MoE expert offload, and a CPU fallback layer for ops that aren't yet wired up.

7. **TensorSharp.Server** is the HTTP/application layer. It provides Ollama-compatible and OpenAI-compatible REST APIs, the browser-based chat UI, upload handling, an `InferenceEngineHost` that owns the per-model continuous-batching engine for autoregressive models, a `DiffusionBatchScheduler` for DiffusionGemma Web UI turns, and a thin queue-status surface for backward compatibility.

8. **TensorSharp.Cli** is the console/application layer for local prompts, multimodal experiments, prompt inspection, JSONL batch workflows, the interactive REPL, and the built-in prefill / decode benchmarks.

### Performance Optimizations

The list below is the cross-architecture summary; each per-model card under
[`docs/models/`](docs/models/README.md) walks through the same kernels in
context, with the exact GGML graph dispatched and the conditions under which
the fused path engages.

- **Fused GPU decode** (Gemma 4): all transformer layers are executed in a single GGML compute graph dispatch on Metal, reducing CPU-GPU round-trips from hundreds per token to one. This achieves ~2.6x speedup over per-operation dispatch.
- **Fused GPU prefill** (Gemma 4): for dense (non-MoE, non-shared, non-PLE/multimodal) layers, `Gemma4LayerPrefill` runs the entire transformer block (RMSNorm + QKV + QK-norm + RoPE + attention + output projection + post-attn norm + GeGLU FFN + post-FFN norm + residual + layer scalar) as a single GGML graph dispatch per layer during prefill, extending the fused approach from decode to multi-token prefill.
- **Chunked prefill** (Gemma 4): long prompts are split into bounded chunks (2x sliding window, max 2048 tokens) to avoid O(n^2) attention score tensors for SWA layers. Chunking is applied automatically when text-only (no multimodal embeddings) and keeps each chunk within the SWA window budget.
- **Native whole-model decode** (Qwen 3): all transformer layers run in one native call (`TransformerModelDecode`) with pre-resolved per-layer weight pointers cached at load time, removing managed-loop overhead from the decode hot path.
- **Fused Qwen 3.5/3.6-family attention layer decode**: a single GGML graph performs RMSNorm + fused QKV + Q/gate deinterleave + per-head QK norm + RoPE + KV cache append + flash attention + sigmoid-gated mix + output projection + residual add for each FullAttention layer. Replaces ~2 standalone GGML calls and ~6 small CPU/GPU sync points per attention layer. Engages once the cached sequence length exceeds 4096 tokens (override with `FUSED_ATTN_LAYER_MIN_SEQ_LEN=N`).
- **Fused prefill attention** (Qwen 3.5/3.6-family): `FusedPrefillAttention` combines Q*K^T, causal mask, softmax, and *V into a single GGML graph dispatch during multi-token prefill, eliminating ~5 separate C#-to-GGML round-trips per attention layer. Handles both initial prefill and continuation with existing KV cache entries.
- **Fused output-projection + FFN** (Qwen 3.5/3.6-family): for both FullAttention and GatedDeltaNet layers with dense FFN, `FusedOutProjFFN` merges the output projection, residual add, post-attention RMSNorm, and the full SwiGLU FFN (gate_up matmul + SiLU + down matmul + residual) into a single GGML graph dispatch, reducing two GPU round-trips to one per layer.
- **Fused output-projection + norm + router** (Qwen 3.5/3.6-family MoE): `FusedOutProjNormRouter` merges the GatedDeltaNet output projection, residual add, post-attention RMSNorm, and MoE router projection into one dispatch. The pre-computed router logits are then consumed directly by the batched MoE kernel, eliminating a separate router dispatch per MoE layer.
- **Fused vision encoder** (Qwen 3.5/3.6-family): `FusedVisionAttention` merges LayerNorm + QKV + bias + 2D RoPE + scaled dot-product attention + output projection + bias + residual into one GGML graph dispatch (~8 ops → 1). `FusedVisionMLP` merges LayerNorm + up + bias + GELU + down + bias + residual into one dispatch (7 ops → 1). Combined, these cut the per-block GPU round-trips from ~15 to 2.
- **Fused weight projections**: Q/K/V projections are fused into a single QKV matmul; gate and up projections are fused into a single gate_up matmul.
- **Native quantized compute**: quantized weights (Q4_K_M, Q6_K, Q8_0, IQ2_XXS, MXFP4, etc.) are used directly in matmul without expanding to FP32, saving memory and bandwidth. A batched `AddmmQuantBatch` kernel handles multiple sub-weight matmuls against a single quantized blob in one dispatch.
- **Direct CUDA kernels**: the `cuda` backend accelerates fill/copy, unary ops, activation fusions, RMSNorm, softmax, index select, causal masking, RoPE/RoPEEx, cuBLAS GEMM, and supported quantized matmul/get-rows while safely falling back for incomplete op coverage.
- **Batched GPU MoE**: `MoEExpertsSwiGLUResidual` (Qwen 3.5/3.6-family) and `MoEExpertsForward` (Nemotron-H) collapse all selected experts -- and, for Qwen 3.5/3.6-family, the optional shared expert and the residual add -- into a single GGML graph dispatch per MoE layer.
- **GEMM-based vision patch embedding** (Qwen 3.5/3.6-family): the patch embedding step is reformulated as parallel im2col + matrix multiplication, replacing a single-threaded scalar quintuple-nested loop with a GPU-accelerated matmul.
- **Parallelized Q/gate deinterleave** (Qwen 3.5/3.6-family): the Q + sigmoid-gate deinterleave in FullAttention prefill is parallelized across tokens, scaling linearly with CPU core count for long prompts.
- **Optimized pure C# CPU path**: managed GEMM fast paths and contiguous float32 kernels accelerate decode, softmax, RMSNorm, RoPE, fused activations, and other hot paths while keeping quantized GGUF weights compressed during CPU loading.
- **Circular KV cache**: sliding-window attention layers use a fixed-size circular buffer, bounding memory usage regardless of sequence length.
- **KV-cache prefix reuse**: multi-turn conversations reuse the longest matching token prefix across turns. Truncation is automatically backed off by the sliding-window size for SWA models so the suffix can rebuild the SWA context.
- **Paged KV cache & block-hash prefix sharing**: the continuous-batching engine partitions KV into fixed-size blocks, content-hashes each full block, and shares them across concurrent and sequential requests. Models that have not implemented `IBatchedPagedModel` still use the engine's isolated per-sequence KV-swap fallback.
- **Native paged-attention kernel**: `TSGgml_PagedAttentionForward` (and the `WithSinks` variant for GPT OSS) does a C++ gather of K/V from the paged buffer, builds a small GGML graph per sequence, and dispatches `ggml_flash_attn_ext` — the same fused Metal/CUDA flash-attention kernel the legacy single-sequence path uses. On Ministral-3-14B long-context (4×~800 tokens) it is **~21 % faster than the legacy per-sequence GGML path**.
- **Batched / paged forward passes**: Mistral 3, Gemma 4, GPT OSS, Qwen 3.5/3.6 (incl. GatedDeltaNet recurrent state pool), and Nemotron-H (incl. Mamba2 recurrent state pool + native batched Mamba2 kernel) pack N sequences into a single `ForwardBatch` call with one batched linear-projection matmul per layer, paged K/V scatter via `slotMapping`, and per-sequence attention via the native kernel. Gemma 4 batched path reaches **1.5×** legacy throughput at batch=8 short prompts and **1.6×** at 4×800-token prompts; Nemotron-H Mamba2 batched reaches **3.95×** at batch=3 on Apple M4 Pro. See [docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md).
- **MTP / NextN speculative decoding**: solo sequences can run a multi-token-prediction draft head (Qwen 3.6 embedded NextN block; Gemma 4 separate `gemma4-assistant` draft GGUF). The draft proposes up to `--mtp-draft` tokens and the trunk verifies them in one batched forward, with the request's own sampler driving both, so decode is accelerated without changing the output. On ggml backends, fused single-graph multi-token-verify and draft-step kernels (`NativeGemma4ModelVerify` / `TryFusedMoEModelVerify` / `NativeGemma4DraftStep`, plus the Qwen 3.6 NextN graph) amortize the verify; the Gemma 4 path also adds gallocr verify scratch and a dense fast-rollback that avoids re-running the kept prefix on partial acceptance. The pure-C# `cuda` backend runs a fully GPU-resident per-op verify/draft (donor-cache attention, GQA decode kernel, GPU RoPE) so the verify layer loop issues zero host-sync stalls. Off by default; `--mtp-spec`.
- **DiffusionGemma prompt-KV caching and fused denoising**: on GPU backends, the prompt side of `[prompt | canvas]` is prefetched once per block and reused across denoising steps; GGML backends default to fused whole-model diffusion decode plus a fused lm-head tail. The Web UI batches concurrent diffusion requests at block boundaries through `DiffusionBatchScheduler`.
- **Kernel warmup**: both CLI and Server run a tiny forward pass at startup to pre-compile GPU kernels (Metal pipeline states, CUDA JIT) and warm the memory pool, avoiding cold-start latency on the first real inference request.
- **Prefill caching** (Gemma 4, Qwen 3.5/3.6-family): per-forward-pass SWA mask cache (Gemma 4), NeoX RoPE cos/sin lookup table cache across global layers (Gemma 4), and RoPE position tensor cache across layers (Gemma 4, Qwen 3.5/3.6-family) eliminate redundant recomputation during prefill.
- **In-place QK RMSNorm** (Qwen 3.5/3.6-family): per-head QK normalization is performed in-place using a `View`, avoiding one tensor allocation and copy per Q/K per layer.

### Memory Optimizations

- **Zero-copy file-mapped quantized weights** (direct CUDA, GGML CUDA, GGML Metal, GGML CPU): the GGUF model file is memory-mapped and quantized tensors are bound directly into native ops via host-pointer buffers. This removes the per-tensor copy from disk into a freshly-allocated native heap buffer that previously roughly doubled the resident set on Apple Silicon for large quantized models. For example, `Qwen3.5-35B-A3B-IQ2_XXS` (~10 GB GGUF) now runs with ~7 GB peak working memory under Metal instead of ~17 GB. The OS keeps the mapped file in its page cache and pages it out under memory pressure without any inference penalty on Apple Silicon (unified memory).
- **Best-fit memory pool**: the GGML host allocator uses a best-fit search across pooled blocks instead of first-fit, which avoids handing out a large scratch block to satisfy a tiny intermediate-tensor request and keeps the working-set tightly bounded across long-running inference.
- **Bounded pool retention**: the integrated-GPU / CPU memory pool now caps individual retained blocks at 64 MB and the total pool at 32 blocks. Combined with mmap-backed weights, this keeps short-lived intermediate tensors recycled fast while bounding the peak resident set.
- **Memory-efficient model loading**: large tensors are streamed directly to native memory without intermediate managed allocations. F32 weights and norms still load on demand; quantized weights are mmap-backed when supported by the backend.
- **Paged KV block pool with optional SSD spillover**: paged KV blocks live in a per-engine `BlockPool` with LRU eviction; the `PagedKvBlockStore` keeps a configurable RAM cap (`TS_KV_CACHE_MAX_RAM_MB`) and spills cold blocks into an SSD tier (`TS_KV_CACHE_SSD_DIR`) up to `TS_KV_CACHE_MAX_SSD_MB`. Block content-hashes are kept in a global index so prefix matches are reused across sessions and requests without rematerialising the K/V.
- **KV block codecs**: blocks can be optionally compressed in-place with `TurboQuantKvCodec` (Q4 or Q8) via `--paged-kv-quant-bits`, trading a small accuracy cost for half / quarter the per-block bandwidth and memory footprint. Recurrent-state models fall back to passthrough automatically.

## Benchmarks

### Head-to-head vs llama.cpp (engine comparison)

A pure-.NET engine going toe-to-toe with the hand-tuned C++ `llama.cpp` on **identical GGUF files, the same NVIDIA RTX 3080 Laptop GPU (16 GB), and one uniform OpenAI `/v1/chat/completions` surface** — across text, multi-turn, and structured-output (JSON) scenarios. The numbers below are the **geomean speedup of TensorSharp over llama.cpp on the same GGML CUDA backend** (single-stream, greedy, MTP off). **> 1.0× means TensorSharp is faster** (decode / prefill) or lower-latency (TTFT). The full per-scenario tables are in [`docs/engine_comparison_report.md`](docs/engine_comparison_report.md).

| Model | decode | prefill | TTFT |
|---|---:|---:|---:|
| Gemma 4 26B-A4B it (QAT UD-Q4_K_XL, **MoE**) | 0.92× | **1.88×** | **1.69×** |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | 0.93× | **1.23×** | **1.10×** |
| Gemma 4 E4B it (Q8_0, dense multimodal) | 0.95× | **1.21×** | **1.07×** |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | 0.94× | **1.18×** | 0.95× |

Where TensorSharp pulls clearly ahead:

- **Prefill wins on every model.** TensorSharp out-prefills llama.cpp across the board — geomean **1.18× (Qwen 3.6) → 1.88× (26B-A4B MoE)** — on the strength of verify-based whole-model prefill plus fused FFN / attention kernels. Time-to-first-token follows, dropping on three of four models (up to **1.69×** lower on the MoE flagship).
- **The MoE flagship leads.** On Gemma 4 26B-A4B the persistent captured-CUDA-graph MoE decode plus verify-based whole-model prefill deliver **1.88× prefill / 1.69× TTFT** geomean with decode at parity — wins stacked on top of competitive token generation.
- **Structured output (JSON mode) is a standout.** Constrained-JSON prefill runs **5.89×** faster on 26B-A4B (354.7 vs 60.2 tok/s) with first token **3.34× sooner** (234 ms vs 781 ms); 12B prefill is **1.89×** and E4B **1.52×**. JSON-mode decode even wins on E4B (**1.10×**) and Qwen 3.6 (**1.07×**).
- **Multi-turn chat — the real chatbot workload.** Content-hashed paged-KV prefix reuse makes follow-up turns land far sooner: prefill / TTFT of **1.87× / 1.93×** on 26B-A4B and **1.55× / 1.60×** on 12B versus llama.cpp.
- **Snappy first token on short prompts.** E4B returns a short prompt's first token **1.75× sooner** (125 ms vs 219 ms) and prefills it **1.62×** faster; Qwen 3.6 prefills short prompts **1.59×** faster (123.2 vs 77.4 tok/s).
- **Qwen 3.6 IQ2_XXS is competitive now.** The aggressively quantized 35B-A3B MoE reaches near-parity decode (0.94×) and a prefill win (1.18×) — recovered from being the weak spot it once was.

Decode sits at parity across both dense and MoE models (0.92–0.95×), so the prefill, TTFT, and structured-output wins come on top of competitive token generation. The remaining sub-1.0× cells — chiefly long-context dense prefill — are active optimization targets rather than finished results. Every cell, wins and gaps alike, is in the [full report](docs/engine_comparison_report.md).

## Testing

### Unit tests (xUnit)

`InferenceWeb.Tests` exercises in-process behavior that doesn't require a running server: managed quantized ops, direct CUDA backend kernels when a CUDA device is available, MLX backend kernels when MLX is available, paged KV cache scheduling (`ContinuousBatchSchedulerTests`, `PagedKvCacheTests`, `PagedKvCacheCodecTests`), batched executor correctness (`BatchedExecutorTests`), per-model batched-forward correctness against the legacy path (`Qwen35BatchedCorrectnessTests`, `Mistral3BatchedForwardTests`, `Gemma4BatchedForwardTests`, `GptOssBatchedCorrectnessTests`, `NemotronBatchedCorrectnessTests`), MTP / NextN speculative-decoding correctness and opt-in end-to-end probes (`MtpSpeculativeExecutionTests`, `Qwen36MtpTests`, `Gemma4MtpTests`), DiffusionGemma denoising / prompt-KV / batched-generation probes (`DiffusionGemmaTests`), per-model batched perf microbenchmarks (`*BatchedPerfBench.cs`), `TurboQuantKvCodec` codec round-trips, prefill chunking, KV cache policies, KV-cache prompt rendering / multi-turn integration, chat-session and session-manager isolation, model service history plumbing, request-logging middleware and file-logger provider, image preprocessing, media helpers, structured-output validation, text-upload helpers, model-service upload logging, web UI chat policy, model context length parsing, backend catalog resolution, and the server CLI options builder (`ServerOptionsBuilderTests`).

```bash
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj
```

### Server integration tests

Integration tests for TensorSharp.Server are in `TensorSharp.Server/testdata/`. They cover all three API styles (Web UI SSE, Ollama, OpenAI), multi-turn conversations, thinking mode, tool calling, structured outputs, queue-status compatibility, concurrent requests, and abort support. Architecture-specific features (thinking, tool calling) are auto-detected and skipped when the active model does not support them.

```bash
# Start TensorSharp.Server, then run:
python3 TensorSharp.Server/testdata/test_multiturn.py
# or
bash TensorSharp.Server/testdata/test_multiturn.sh
```

See [TensorSharp.Server/testdata/README.md](TensorSharp.Server/testdata/README.md) for the full test matrix.

### Inference matrix runner

`TensorSharp.TestMatrix` is the broader CLI-driven harness for long-running model/backend coverage. It discovers GGUF files, filters unavailable backends and unsupported prompt types, runs baseline plus env-var sweep cells, writes one JSON result per cell, emits an aggregate Markdown report, and compares results with per-host baselines when requested.

```bash
dotnet build TensorSharp.TestMatrix/TensorSharp.TestMatrix.csproj -c Release
dotnet run --project TensorSharp.TestMatrix -c Release -- --dry-run
```

See [TensorSharp.TestMatrix/README.md](TensorSharp.TestMatrix/README.md) and [docs/env_var_feature_matrix.md](docs/env_var_feature_matrix.md) for the current runner contract.

## Author

Zhongkai Fu

## License

See [LICENSE](LICENSE) for details.
