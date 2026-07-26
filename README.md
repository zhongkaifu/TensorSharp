# TensorSharp

<p align="center">
  <img src="imgs/banner_1.png" alt="TensorSharp logo" width="320">
</p>

[English](README.md) | [中文](README_zh-cn.md)

**Native .NET LLM inference engine for GGUF models** — autoregressive LLMs *and* DiffusionGemma-style text-diffusion, plus Qwen-Image-Edit image editing. Ships a console app, a browser chat UI, and Ollama/OpenAI-compatible HTTP APIs. A pure-.NET engine that trades wins with the hand-tuned C++ `llama.cpp` on identical GGUF files and the same GPU.

## From Tensors to Tokens — the TensorSharp book

<p align="center">
  <a href="https://www.amazon.com/dp/B0H9P44QZZ">
    <img src="website/assets/from-tensors-to-tokens-cover.jpg" alt="From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B" width="220">
  </a>
</p>

**[From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B](https://www.amazon.com/dp/B0H9P44QZZ)** by Zhongkai Fu turns this repository into a guided, end-to-end learning journey. It uses Gemma 4 E4B to connect tensor fundamentals, model execution, multimodal inputs, and the application surfaces of a working LLM inference engine.

**[Explore the book and its repository reading path](docs/BOOK.md)** · **[Buy the paperback on Amazon](https://www.amazon.com/dp/B0H9P44QZZ)**

## Highlights

- **⚡ Trades wins with llama.cpp — from pure .NET.** On identical GGUF files and the same GPU, TensorSharp matches or beats `llama.cpp` on the workloads that matter: Gemma 4 E4B and 2-bit Qwen 3.6 35B-A3B MoE prefill **1.28×** faster on CUDA with first tokens **1.27×** sooner (multi-turn up to **1.49×**); Gemma 4 12B decodes **1.21×** faster on Vulkan (up to **1.32×** on long context). → [Benchmarks](#benchmarks)
- **🚀 Continuous batching & paged KV cache.** vLLM-style paged KV pool with block-hash prefix sharing and an iteration-level scheduler, on by default in the server. → [deep dive](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md)
- **🔮 MTP / NextN speculative decoding.** Multi-token-prediction draft heads accelerate solo decode on Qwen 3.6 (embedded NextN block) and Gemma 4 (separate `gemma4-assistant` draft GGUF) — the draft proposes, the trunk verifies in one batched forward, output identical to standard decode. → [Speculative decoding](FEATURES.md#mtp--nextn-speculative-decoding)
- **🎨 Qwen-Image-Edit image editing.** Prompt + input image → edited image, driving a 60-block MMDiT with a Qwen-Image VAE and Qwen2.5-VL-7B text encoder. CUDA-graph-captured DiT, FlowMatch-Euler true-CFG denoise, live Web UI previews, and Lightning-LoRA fast paths. Beat `stable-diffusion.cpp` **1.19×** on a warm 4-step edit. → [Qwen-Image-Edit card](docs/models/qwenimage.md)
- **🌫️ DiffusionGemma text diffusion.** Block-wise EntropyBound denoising over a Gemma-4-derived MoE backbone, with CLI flags and a Web UI denoising preview stream. → [DiffusionGemma card](docs/models/diffusiongemma.md)
- **🖼️ Multimodal.** Image / video / audio (Gemma 4); image input for Gemma 3, Qwen 3.5-family, Mistral 3, and Nemotron-H Omni; PDF documents via CLI and Web UI. → [Multimodal](FEATURES.md#multimodal-support)
- **🛠️ Tool calling & thinking mode.** Multi-turn tool calls and structured chain-of-thought across Qwen 3, Qwen 3.5/3.6-family, Gemma 4, GPT OSS, and Nemotron-H. → [Features](FEATURES.md)
- **🔌 Ollama- & OpenAI-compatible APIs** plus a browser chat UI — drop-in for existing tooling. → [HTTP APIs](USAGE.md#http-apis)
- **📄 Config files with auto-download.** Put CLI/Server options in a reusable JSON file with `${variables}` and `{ "path", "urls" }` entries that fetch the model on first run. → [config/README.md](config/README.md)
- **🧮 Native quantized compute.** Q4_K_M / Q8_0 / MXFP4 / IQ2_XXS and more run in matmul without dequantizing to FP32. Runs on GGML Metal / CUDA / Vulkan, a direct CUDA/cuBLAS backend, MLX (Apple Silicon), and a pure-C# CPU path — all with CPU fallbacks. → [Backends](USAGE.md#compute-backends)

## Quick Start

TensorSharp targets .NET 10. On a new machine, install the full **.NET 10 SDK**—the .NET Runtime alone cannot build TensorSharp:

| Platform | Install the SDK |
|---|---|
| **Windows** | In PowerShell, run `winget install Microsoft.DotNet.SDK.10`, or use Microsoft's [.NET installation guide for Windows](https://learn.microsoft.com/en-us/dotnet/core/install/windows). |
| **macOS** | Use the [.NET 10 SDK installer](https://dotnet.microsoft.com/en-us/download/dotnet/10.0): choose **Arm64** for Apple silicon or **x64** for an Intel Mac. See Microsoft's [macOS instructions](https://learn.microsoft.com/en-us/dotnet/core/install/macos). |
| **Linux** | Follow Microsoft's [Linux distribution guide](https://learn.microsoft.com/en-us/dotnet/core/install/linux) to configure the correct package source for your distro and install its .NET 10 SDK package (commonly `dotnet-sdk-10.0`). |

Open a new terminal and verify that a `10.0.x` SDK is listed:

```bash
dotnet --list-sdks
```

See the [cross-platform .NET install overview](https://learn.microsoft.com/en-us/dotnet/core/install/) or [Development → Prerequisites](DEVELOPMENT.md#prerequisites) for more detail.

Then get running in ~30 seconds on the verified native GGML fast path — Gemma 4 E4B. The other prerequisites are `git`, `curl`, and the toolchain for your GPU backend (see [Development → Prerequisites](DEVELOPMENT.md#prerequisites)). The recommended public file is [`gemma-4-E4B-it-Q8_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q8_0.gguf) (7.48 GiB); text-only inference needs no projector.

**Windows + NVIDIA (PowerShell)**

```powershell
git clone https://github.com/zhongkaifu/TensorSharp.git; Set-Location TensorSharp
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -L --fail "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" -o models\gemma-4-E4B-it-Q8_0.gguf
'Answer in one short sentence: what is TensorSharp?' | Set-Content prompt.txt
$env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA = 'ON'
dotnet run --project TensorSharp.Cli -c Release -p:TensorSharpSkipMlxNative=true -- --model models\gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --max-tokens 128 --backend ggml_cuda
```

**macOS (Apple Silicon)** — drop the CUDA env var and use `--backend ggml_metal`.
**Linux + NVIDIA** — prefix the `dotnet run` with `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` and use `--backend ggml_cuda`.
**AMD / Intel / NVIDIA Vulkan** — set `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON` and use `--backend ggml_vulkan`.

Host the same model as a server (browser UI at <http://localhost:5000/index.html>, plus Ollama/OpenAI APIs):

```bash
dotnet run --project TensorSharp.Server -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --max-tokens 512
```

> The server binds `0.0.0.0:5000` with no built-in auth or TLS — keep it behind a firewall or an authenticated HTTPS reverse proxy. For image/video/audio add the companion [`mmproj-gemma-4-E4B-it-Q8_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/mmproj-gemma-4-E4B-it-Q8_0.gguf) with `--mmproj`.

Full command reference: **[CLI](USAGE.md#console-application)** · **[Server](USAGE.md#web-application)** · more models to download: **[Model Downloads](MODEL_DOWNLOADS.md)** · prefer a config file? **[config/](config/README.md)**.

## Pick a Backend

Every backend falls back to CPU for any op it does not implement, so output stays correct on all of them.

| Your hardware | Recommended backend | Flag | Notes |
|---|---|---|---|
| **Apple Silicon (Mac)** | GGML Metal | `--backend ggml_metal` | Default on macOS. `--backend mlx` is an alternative Apple-Silicon GPU path. |
| **Windows / Linux + NVIDIA GPU** | GGML CUDA | `--backend ggml_cuda` | Most-tested NVIDIA path. `--backend cuda` is the direct PTX/cuBLAS backend for experimentation. |
| **Windows / Linux + AMD / Intel / NVIDIA GPU** | GGML Vulkan | `--backend ggml_vulkan` | Vendor-neutral GPU path via ggml-vulkan. Built automatically when a Vulkan runtime is present; `--no-vulkan` opts out. |
| **No GPU / portability / debugging** | Pure C# CPU | `--backend cpu` | No native dependencies. For faster CPU inference use `--backend ggml_cpu` (native kernels). |

Full per-backend description: [Usage → Compute Backends](USAGE.md#compute-backends).

## Verified Models

Implemented and exercised by the test/benchmark matrix. Pick a quantization that fits your hardware (Q4_K_M for low memory, Q8_0 for higher quality). More sizes and projector files: [Model Downloads](MODEL_DOWNLOADS.md).

| Family | Example model (GGUF) | Image / Video / Audio | Thinking | Tools | Card |
|---|---|---|---|---|---|
| Gemma 4 | [gemma-4-E4B-it](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) (also 31B, 26B-A4B MoE) | ✅ / ✅ / ✅ | ✅ | ✅ | [gemma4.md](docs/models/gemma4.md) |
| Qwen 3.5 / 3.6 | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) (also 35B-A3B MoE) | ✅ / — / — | ✅ | ✅ | [qwen35.md](docs/models/qwen35.md) |
| Qwen 3 | [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | — / — / — | ✅ | ✅ | [qwen3.md](docs/models/qwen3.md) |
| GPT OSS | [gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) (MoE) | — / — / — | ✅ | ✅ | [gptoss.md](docs/models/gptoss.md) |
| Nemotron-H | [Nemotron-H-8B](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) (also 47B, Omni) | ✅ (Omni) / — / — | ✅ | ✅ | [nemotron.md](docs/models/nemotron.md) |
| Mistral 3 | [Mistral-Small-3.1-24B](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | ✅ / — / — | — | — | [mistral3.md](docs/models/mistral3.md) |
| Gemma 3 | [gemma-3-4b-it](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF) | ✅ / — / — | — | — | [gemma3.md](docs/models/gemma3.md) |
| DiffusionGemma | [diffusiongemma-26B-A4B-it](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | — / — / — | — | — | [diffusiongemma.md](docs/models/diffusiongemma.md) |
| Qwen-Image-Edit | [Qwen-Image-Edit-2511](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) (MMDiT + VAE + Qwen2.5-VL) | 🖼️ image→image | — | — | [qwenimage.md](docs/models/qwenimage.md) |

## Supported Model Architectures

| Architecture | GGUF arch keys | Example Models | Multimodal | Thinking | Tools | MTP spec | Card |
|---|---|---|---|---|---|---|---|
| Gemma 4 | `gemma4` | gemma-4-E4B, gemma-4-31B, gemma-4-26B-A4B (MoE) | Image, Video, Audio | Yes | Yes | Yes (separate draft GGUF) | [gemma4.md](docs/models/gemma4.md) |
| Gemma 3 | `gemma3` | gemma-3-4b | Image | No | No | — | [gemma3.md](docs/models/gemma3.md) |
| Qwen 3 | `qwen3` | Qwen3-4B | Text only | Yes | Yes | — | [qwen3.md](docs/models/qwen3.md) |
| Qwen 3.5 / 3.6 family | `qwen35`, `qwen35moe`, `qwen3next` | Qwen3.5-9B (hybrid Attn+Recurrent), Qwen3.5/3.6-35B-A3B (MoE) | Image | Yes | Yes | Yes on Qwen 3.6 (embedded NextN) | [qwen35.md](docs/models/qwen35.md) |
| GPT OSS | `gptoss`, `gpt-oss` | gpt-oss-20b (MoE) | Text only | Yes (always) | Yes | — | [gptoss.md](docs/models/gptoss.md) |
| Nemotron-H | `nemotron_h`, `nemotron_h_moe` | Nemotron-H-8B/47B (Hybrid SSM-Transformer, MoE), Nemotron 3 Nano Omni | Image (Omni) | Yes | Yes | — | [nemotron.md](docs/models/nemotron.md) |
| Mistral 3 | `mistral3` | Mistral-Small-3.1-24B-Instruct | Image | No | No | — | [mistral3.md](docs/models/mistral3.md) |
| DiffusionGemma | `diffusion-gemma` | diffusion-gemma text-diffusion GGUFs | Text only | No | No | — | [diffusiongemma.md](docs/models/diffusiongemma.md) |
| Qwen-Image-Edit | `qwen_image` | qwen-image-edit MMDiT GGUFs (+ VAE & Qwen2.5-VL) | Image edit (image+text → image) | No | No | — | [qwenimage.md](docs/models/qwenimage.md) |

End-to-end per-model documentation (origin, forward graph, components, parameters, prefill/decode optimizations): [architecture cards](docs/models/README.md).

## Benchmarks

### Head-to-head vs llama.cpp (engine comparison)

A pure-.NET engine going toe-to-toe with the hand-tuned C++ `llama.cpp` on **identical GGUF files, the same NVIDIA RTX 3080 Laptop GPU (16 GB), and one uniform OpenAI `/v1/chat/completions` surface** — with **both engines measured on their GGML CUDA and Vulkan builds**. Numbers are the **geomean speedup of TensorSharp over llama.cpp on the same backend** (single-stream, greedy, MTP off); **> 1.0× means TensorSharp is faster / lower-latency**. Full per-scenario tables: [`docs/engine_comparison_report.md`](docs/engine_comparison_report.md).

| Model | Backend | decode | prefill | TTFT |
|---|---|---:|---:|---:|
| Gemma 4 E4B it (Q8_0, dense multimodal) | CUDA | 1.02× | **1.28×** | **1.27×** |
| Gemma 4 E4B it (Q8_0, dense multimodal) | Vulkan | 1.00× | 1.05× | 1.03× |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | CUDA | 1.04× | **1.17×** | **1.16×** |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | Vulkan | **1.21×** | 1.04× | 1.03× |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | CUDA | 0.98× | **1.28×** | **1.27×** |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | Vulkan | 0.87× | 1.04× | 1.03× |
| Qwen 3.6 27B (UD-IQ2_XXS, dense) | CUDA | **1.07×** | 0.96× | 0.95× |
| Qwen 3.6 27B (UD-IQ2_XXS, dense) | Vulkan | 1.02× | 0.85× | 0.84× |

TensorSharp pulls clearly ahead on CUDA prefill / first-token latency (multi-turn prefill wins on **every** model, up to **1.49×**), holds decode parity-or-better on CUDA, and wins Vulkan decode on the dense 12B (up to **1.32×** on long context) — even at 2-bit IQ2_XXS quantization. The remaining sub-1.0× cells are active optimization targets. The harness also covers tool-calling, structured-output, image-edit (vs `stable-diffusion.cpp`), MTP on/off, and parallel-request scenarios you can run yourself via [`benchmarks/engine_comparison`](benchmarks/engine_comparison). Every cell is in the [full report](docs/engine_comparison_report.md).

## Documentation

New here? The sections above are all you need to get running. Everything else is detailed reference:

| Doc | What's inside |
|---|---|
| [Book guide: From Tensors to Tokens](docs/BOOK.md) | A guided path from tensor fundamentals to a multimodal Gemma 4 E4B inference engine, with publication details and links into the companion repository |
| [Model Downloads](MODEL_DOWNLOADS.md) | Per-model `huggingface-cli` download + run quick reference (quant tiers, projectors, companions) |
| [Usage](USAGE.md) | Full CLI reference (options, interactive REPL, JSONL batch), server hosting, logging, HTTP API examples, backends, and the env-var matrix |
| [Features](FEATURES.md) | Deep dives on continuous batching, MTP speculative decoding, tool calling, thinking mode, multimodal, MoE, KV codecs, and more |
| [Configuration files](config/README.md) | Put options in a reusable JSON file with `${variables}` and auto-downloading models |
| [Development](DEVELOPMENT.md) | Prerequisites, building the native GGML/MLX libraries, repository layout, package boundaries, internal architecture, and the test harness |
| [Per-model architecture cards](docs/models/README.md) | End-to-end docs of each architecture (forward graph, components, parameters, prefill/decode optimizations) |
| [Paged attention & continuous batching](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md) | The vLLM-style paged KV cache, prefix sharing, and iteration-level scheduler |
| [Environment variable feature matrix](docs/env_var_feature_matrix.md) | Which high-impact runtime flags affect which models, backends, and prompt types |
| [Engine comparison report](docs/engine_comparison_report.md) | Full per-scenario TensorSharp vs llama.cpp / stable-diffusion.cpp tables |
| [Test/benchmark matrix runner](TensorSharp.TestMatrix/README.md) | Sweep model × backend × feature × env-var cells and generate regression reports |
| [Server API examples](TensorSharp.Server/API_EXAMPLES.md) | Complete curl and Python examples for the server surface |

## Current Status

| Area | Status |
|---|---|
| Model families | Gemma 3/4, DiffusionGemma, Qwen 3, Qwen 3.5/3.6-family (`qwen35`, `qwen35moe`, `qwen3next`), GPT OSS, Nemotron-H (incl. Nemotron 3 Nano Omni), Mistral 3. Image editing via Qwen-Image-Edit (`qwen_image` MMDiT). |
| Inference hosts | CLI, interactive REPL, ASP.NET Core web UI, Ollama-style API, OpenAI Chat Completions-style API. |
| Backends | Pure C# CPU, direct CUDA/cuBLAS (`cuda`), MLX Metal (`mlx`), GGML CPU, GGML Metal, GGML CUDA, GGML Vulkan. |
| Multimodal | Gemma 4 image/video/audio; Gemma 3, Qwen 3.5-family, Mistral 3, Nemotron-H Omni image input; PDF documents (CLI `--pdf` + Web UI). |
| Continuous batching | vLLM-style paged KV cache, block-hash prefix sharing, iteration-level scheduler (default on; opt-out `--no-continuous-batching`). |
| Speculative decoding | MTP / NextN draft heads on Qwen 3.6 (embedded) and Gemma 4 (separate draft GGUF); off by default, opt-in via the server's `--mtp-spec`. |
| Server model scope | One explicitly hosted GGUF via `--model`; optional explicit projector via `--mmproj`; no directory scanning. |
| Observability | Structured per-turn logs, queue status, and KV-cache reuse metrics across Web UI, Ollama, and OpenAI shapes. |

## Author

Zhongkai Fu

## License

See [LICENSE](LICENSE) for details.
