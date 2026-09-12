# TensorSharp

<p align="center">
  <img src="imgs/banner_1.png" alt="TensorSharp logo" width="320">
</p>

[English](README.md) | [中文](README_zh-cn.md)

**Native .NET LLM inference engine for GGUF models** — autoregressive LLMs *and* DiffusionGemma-style text-diffusion, plus Qwen-Image-Edit image editing and MiniMax-H3 video with native 32 kHz stereo audio (and Wan 2.1/2.2 for video alone). Ships a console app, a browser chat UI, and Ollama/OpenAI-compatible HTTP APIs. A pure-.NET engine that trades wins with the hand-tuned C++ `llama.cpp` on identical GGUF files and the same GPU. The optional `TensorSharp.AgentHost` layer adds Agent Skills and a bounded, in-process model-to-tool loop for sandboxed file and shell work.

## From Tensors to Tokens — the TensorSharp book

<p align="center">
  <a href="https://www.amazon.com/dp/B0H9P44QZZ">
    <img src="website/assets/from-tensors-to-tokens-cover.jpg" alt="From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B" width="220">
  </a>
</p>

**[From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B](https://www.amazon.com/dp/B0H9P44QZZ)** by Zhongkai Fu turns this repository into a guided, end-to-end learning journey. It uses Gemma 4 E4B to connect tensor fundamentals, model execution, multimodal inputs, and the application surfaces of a working LLM inference engine.

**[Explore the book and its repository reading path](docs/BOOK.md)** · **[Buy the paperback on Amazon](https://www.amazon.com/dp/B0H9P44QZZ)**

## Highlights

- **Local, native .NET inference.** Run GGUF text and multimodal models from the CLI, browser UI, or Ollama/OpenAI-compatible APIs.
- **Broad model and media support.** Current source covers modern text models, vision/audio input, PDF, image editing, and video generation; see the [model cards](docs/models/README.md).
- **Fast where it matters.** TensorSharp trades wins with `llama.cpp` on identical models and hardware, with native GGML, CUDA, Vulkan, Metal, MLX, and managed CPU paths. See the [benchmark report](docs/engine_comparison_report.md).
- **Agentic work, including iOS.** `TensorSharp.AgentHost` adds bounded Agent Skills and code tools. [TensorAgent](TensorAgent/README.md) brings the same local chat and agent experience to iPhone and iPad using the iOS `ggml_metal` backend.
- **Production-friendly building blocks.** Continuous batching, paged/prefix-shared KV cache, speculative decoding, tensor parallelism, and configurable security boundaries are available when you need them. See [Features](FEATURES.md), [Usage](USAGE.md), and the [current project status](docs/PROJECT_STATUS.md).

The detailed implementation notes and historical benchmark claims have moved to the linked documentation so this page stays useful as a starting point.

## Quick Start

Prefer a prebuilt application? The [Releases page](https://github.com/zhongkaifu/TensorSharp/releases) provides self-contained CLI and Server archives for Windows x64 (CPU/CUDA), Linux x64 (CPU/CUDA), and macOS arm64.

Source builds target .NET 10. On a new development machine, install the full **.NET 10 SDK**—the .NET Runtime alone cannot build TensorSharp:

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

Then get running in ~30 seconds on the verified native GGML fast path — Gemma 4 E4B. The other prerequisites are `git`, `curl`, [CMake](https://cmake.org/download/) 3.20+ (the native GGML library is configured and built with it — on Windows, Visual Studio's "C++ CMake tools for Windows" component ships one and the build will find it), and the toolchain for your GPU backend (see [Development → Prerequisites](DEVELOPMENT.md#prerequisites)). The recommended public file is [`gemma-4-E4B-it-Q8_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q8_0.gguf) (7.48 GiB); text-only inference needs no projector.

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

**Linux (Ubuntu) + multiple NVIDIA GPUs — tensor parallelism**

Tensor parallelism splits one model across N GPUs. It runs on the direct
`cuda` backend and on the GGML CUDA / Vulkan backends (`--backend ggml_cuda`,
`ggml_vulkan`). Qwen 3.8 Flash Next and DeepSeek V4 use the same flag for a
layer split instead: one contiguous run of whole layers per GPU. GLM 5.x also
layer-splits by default when the flag is omitted, while `--tp N` selects its
native local tensor-parallel path on the GGML GPU backends. Install the CUDA
toolkit first, then:

```bash
# On RunPod's Ubuntu 24.04 images, point the loader at the CUDA compat libraries first:
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH
# On older Ubuntu releases the .NET 10 SDK comes from the backports PPA:
add-apt-repository ppa:dotnet/backports

apt update && apt install dotnet-sdk-10.0
git clone https://github.com/zhongkaifu/TensorSharp.git
cd TensorSharp
mkdir models
wget "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" -O models/gemma-4-E4B-it-Q8_0.gguf
bash TensorSharp.GGML.Native/build-linux.sh
dotnet build -c Release

# 2 GPUs in one process
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf \
    --backend cuda --interactive --max-tokens 20000 --tp 2

# Same thing on the GGML CUDA backend (add TENSORSHARP_TP_DEVICES=0,2 to pick GPUs)
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf \
    --backend ggml_cuda --interactive --max-tokens 20000 --tp 2
```

Scale the same model across machines by adding a node ID and the shared peer
list — 2 nodes × 2 GPUs gives a global TP degree of 4:

```bash
# Node 0
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf --backend cuda --tp 2 \
    --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
# Node 1 (same peer list, different node ID)
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf --backend cuda --tp 2 \
    --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
```

`TensorSharp.Server.Host` takes the same `--tp`, `--tp-node-id`, and `--tp-peers`
flags (or the `TENSORSHARP_TP_*` environment variables); in a multi-node
cluster the server is node `0` — the driver that serves HTTP — and every other
node runs a `TensorSharp.Cli` worker. Full reference:
**[Tensor Parallelism & Distributed Inference](USAGE.md#tensor-parallelism--distributed-inference)**.


Host the same model as a server (browser UI at <http://localhost:5000>, plus Ollama/OpenAI APIs):

```bash
dotnet run --project TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --max-tokens 512
```

> The server binds `0.0.0.0:5000` by default (change it with `--port` / `--host`, or the `PORT` / `HOST` environment variables; on macOS port 5000 is taken by the AirPlay Receiver) with no built-in auth or TLS — keep it behind a firewall or an authenticated HTTPS reverse proxy. For image/video/audio add the companion [`mmproj-gemma-4-E4B-it-Q8_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/mmproj-gemma-4-E4B-it-Q8_0.gguf) with `--mmproj`.

Both executables print their full option reference — description, default, range, and an example per flag — when started with no arguments or with `--help`:

```bash
dotnet run --project TensorSharp.Cli -c Release -- --help
dotnet run --project TensorSharp.Server.Host -c Release -- --help
```

Full command reference: **[CLI](USAGE.md#console-application)** · **[Server](USAGE.md#web-application)** · more models to download: **[Model Downloads](MODEL_DOWNLOADS.md)** · prefer a config file? **[config/](config/README.md)**.

## Pick a Backend

Every backend falls back to CPU for any op it does not implement, so output stays correct on all of them.

| Your hardware | Recommended backend | Flag | Notes |
|---|---|---|---|
| **Apple Silicon (Mac)** | GGML Metal | `--backend ggml_metal` | Default on macOS. `--backend mlx` is an alternative Apple-Silicon GPU path. |
| **Windows / Linux + NVIDIA GPU** | GGML CUDA | `--backend ggml_cuda` | Most-tested NVIDIA path. `--backend cuda` is the direct PTX/cuBLAS backend for experimentation. |
| **Windows / Linux + AMD / Intel / NVIDIA GPU** | GGML Vulkan | `--backend ggml_vulkan` | Vendor-neutral GPU path via ggml-vulkan. Built automatically when a Vulkan runtime is present; `--no-vulkan` opts out. |
| **No GPU / portability / debugging** | Pure C# CPU | `--backend cpu` | No native dependencies; matmuls run on a multi-core worker pool. Even DeepSeek V4.1 Flash has a whole-model executor here — it runs on the pure-C# `DeepSeek4CpuExecutor` with no ggml and no GPU, held to the PyTorch oracle `eng/dsv41-reference.py` at `atol=rtol=2e-5` on a five-layer F32 fixture (architectural agreement with the oracle, not parity on the real Q2_K weights), as a correctness and portability path rather than a serving one. For faster CPU inference use `--backend ggml_cpu` (native kernels). |

Full per-backend description: [Usage → Compute Backends](USAGE.md#compute-backends).

## Verified Models

Implemented and exercised by the test/benchmark matrix. Pick a quantization that fits your hardware (Q4_K_M for low memory, Q8_0 for higher quality). More sizes and projector files: [Model Downloads](MODEL_DOWNLOADS.md).

| Family | Example model (GGUF) | Image / Video / Audio | Thinking | Tools | Card |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | [DeepSeek-V4.1-Flash](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/8e0c4de3cb6519bfc11ed69dc87184b457a57bb5) (Q2_K or Q4_K_M shards + prepared Engram sidecar; `ggml_cuda` serving path, with `ggml_cpu` a correctness and portability path that still takes the vision companion, and `cuda` and the pure-C# `cpu` executor text-only ones) | ✅ (vision companion) / ✅ (vision companion) / — | ✅ | ✅ | [deepseek41.md](docs/models/deepseek41.md) |
| DeepSeek V4 Flash | [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) (284B MoE, split GGUF) | — / — / — | ✅ | ✅ | [deepseek4.md](docs/models/deepseek4.md) |
| GLM 5.x | [GLM-5.2](https://huggingface.co/unsloth/GLM-5.2-GGUF) (744B-A40B MoE, split GGUF), [GLM-5.3](https://huggingface.co/unsloth/GLM-5.3-GGUF) (256 routed experts, text only; one subdirectory per quant, UD-Q2_K_XL is seven shards / 236.4 GiB — point `--model` at the `-00001-of-00007` shard), [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (320B MoE, split GGUF, + mmproj) | ✅ (5.3-Flash only; 5.2 and 5.3 are text only) / — / — | ✅ | ✅ | [glm.md](docs/models/glm.md) |
| Qwen 3.8 Flash Next | [Qwen3.8-Flash-Next](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (hybrid GDN + attention MoE, 512 experts, split GGUF, + mmproj) | ✅ / — / — | ✅ | No (no parser) | [qwen38-flash-next.md](docs/models/qwen38-flash-next.md) |
| Gemma 4 | [gemma-4-E4B-it](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) (also 31B, 26B-A4B MoE) | ✅ / ✅ / ✅ | ✅ | ✅ | [gemma4.md](docs/models/gemma4.md) |
| Qwen 3.5 / 3.6 | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) (also 35B-A3B MoE) | ✅ / — / — | ✅ | ✅ | [qwen35.md](docs/models/qwen35.md) |
| Bonsai Q1_0 | Local hash-pinned `Bonsai-8B-Q1_0.gguf` (dense Qwen 3) and `Bonsai-27B-Q1_0.gguf` (dense Qwen 3.5 hybrid); the supplied GGUFs declare no publisher URL or license | — / — / — | 8B: No (fixed empty block); 27B: ✅ | ✅ | [bonsai.md](docs/models/bonsai.md) |
| GPT OSS | [gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) (MoE) | — / — / — | ✅ | ✅ | [gptoss.md](docs/models/gptoss.md) |
| Nemotron-H | [Nemotron-H-8B](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) (also 47B, Omni) | ✅ (Omni) / — / — | ✅ | ✅ | [nemotron.md](docs/models/nemotron.md) |
| Mistral 3 | [Mistral-Small-3.1-24B](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | ✅ / — / — | — | — | [mistral3.md](docs/models/mistral3.md) |
| Hunyuan Dense | Tencent dense Hunyuan GGUFs (`hunyuan-dense`), e.g. the Hy-MT2 releases | — / — / — | — | — | [hunyuan-dense.md](docs/models/hunyuan-dense.md) |
| Muse-Glimmer | [Muse-Glimmer-30B](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) (+ mmproj) | ✅ / — / — | ✅ | ✅ | [muse-glimmer.md](docs/models/muse-glimmer.md) |
| DiffusionGemma | [diffusiongemma-26B-A4B-it](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | — / — / — | — | — | [diffusiongemma.md](docs/models/diffusiongemma.md) |
| Qwen-Image-Edit | [Qwen-Image-Edit-2511](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF) (MMDiT + VAE + Qwen2.5-VL) · fast lane: [Lightning 4-step LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) | 🖼️ image→image | — | — | [qwenimage.md](docs/models/qwenimage.md) |
| MiniMax-H3 audio+video | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF) (denoiser + Qwen3-VL-32B encoder) + [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (video + audio VAE) | 🎬🔊 text→video, image→video, first/last frame, reference→video (image/clip/audio), **with stereo audio** | — | — | [minimax-h3.md](docs/models/minimax-h3.md) |
| Wan 2.1 / 2.2 video | [Wan2.2-TI2V-5B](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF) (also [T2V-A14B](https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF), [I2V-A14B](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF), [Wan2.1-T2V-14B](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf)) + UMT5-XXL + video VAE · fast lane: [TI2V-5B-Turbo](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF) (4-step, 25× fewer DiT passes) | 🎬 text→video, image→video | — | — | [wan.md](docs/models/wan.md) |

## Make It Fast

Start with these choices, in order:

1. **Choose the right checkpoint.** For Wan video, use a Turbo/Lightning/4-step distilled GGUF. For Qwen-Image-Edit, use the Lightning LoRA.
2. **Use the matching backend.** NVIDIA: `ggml_cuda`; Apple Silicon and iOS: `ggml_metal`; CPU: `ggml_cpu` (use managed `cpu` for portability).
3. **Reduce work before tuning flags.** For H3 use `--cfg 1.0` and 4–8 steps; for media, lower resolution, frame count, or steps.
4. **Then scale or speculate.** Try `--draft-model` / `--spec`, `--n-cpu-moe`, or `--tp N` when the model or workload calls for it.

See the [performance guide and detailed fast lanes](docs/PROJECT_STATUS.md#make-it-fast), the [model cards](docs/models/README.md), and the [environment-variable matrix](docs/env_var_feature_matrix.md) for trade-offs and measurements.

## Supported Model Architectures

| Architecture | GGUF arch keys | Example Models | Multimodal | Thinking | Tools | MTP spec | Card |
|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | `deepseek41` | DeepSeek-V4.1-Flash (40 layers, 384 routed experts at top-6 plus one shared expert, four residual streams with delayed hyper-connection mixing, Engram n-gram features, 1M declared context) | Text; image and video with the prepared vision companion (`--mmproj`), audio refused | Yes | Yes (spaced DSML, grammar-constrained) | No (V4 drafters are rejected) | [deepseek41.md](docs/models/deepseek41.md) |
| DeepSeek V4 Flash | `deepseek4` | DeepSeek-V4-Flash (284B MoE, 256 experts, compressed sparse attention, 1M context) | Text only | Yes | Yes (DSML) | Yes (DSpark block drafter, separate GGUF) | [deepseek4.md](docs/models/deepseek4.md) |
| GLM 5.x | `glm-dsa`, `glm5next` | GLM-5.2 (744B-A40B MoE, 256 experts, MLA + DeepSeek Sparse Attention, 1M context), [GLM-5.3](docs/models/glm.md#glm-53-glm-dsa) (the same 79-block `glm-dsa` shape as 5.2 — 78 trunk blocks plus one NextN, 256 routed experts at top-8 with one shared expert, MLA with the lightning indexer, rope base 8e6 — so it loads on the GLM-5.2 path with no new code and no new flag; text only), GLM-5.3-Flash (320B MoE, 288 experts, KDA linear attention + NoPE MLA with a pooled indexer) | Text only (5.2 and 5.3), Image (5.3-Flash) | Yes | Yes (XML tool calls) | Yes on GLM-5.2 and GLM-5.3 (embedded NextN block; on 5.3 speculation engages on the default layer split, no `--tp`) | [glm.md](docs/models/glm.md) |
| Qwen 3.8 Flash Next | `qwen4exp` | Qwen3.8-Flash-Next (hybrid MoE, 512 experts / 10 used, GatedDeltaNet on 36 of 48 layers interleaved with QSA-indexed full attention, PLE n-gram block, ×4 hyper-connections) | Image | Yes | No (no structured tool-output parser) | — | [qwen38-flash-next.md](docs/models/qwen38-flash-next.md) |
| Gemma 4 | `gemma4` | gemma-4-E4B, gemma-4-31B, gemma-4-26B-A4B (MoE) | Image, Video, Audio | Yes | Yes | Yes (separate draft GGUF) | [gemma4.md](docs/models/gemma4.md) |
| Qwen 3.5 / 3.6 family | `qwen35`, `qwen35moe`, `qwen3next` | Qwen3.5-9B (hybrid Attn+Recurrent), Qwen3.5/3.6-35B-A3B (MoE) | Image | Yes | Yes | Yes on Qwen 3.6 (embedded NextN) | [qwen35.md](docs/models/qwen35.md) |
| Bonsai (Qwen family) | `qwen3` (8B), `qwen35` (27B) | Bonsai-8B (36-layer dense GQA), Bonsai-27B (48 GatedDeltaNet + 16 full-attention layers), both Q1_0 | Text only | 27B yes; 8B template emits a fixed empty think block | Yes | — | [bonsai.md](docs/models/bonsai.md) |
| GPT OSS | `gptoss`, `gpt-oss` | gpt-oss-20b (MoE) | Text only | Yes (always) | Yes | — | [gptoss.md](docs/models/gptoss.md) |
| Nemotron-H | `nemotron_h`, `nemotron_h_moe` | Nemotron-H-8B/47B (Hybrid SSM-Transformer, MoE), Nemotron 3 Nano Omni, Nemotron 3.5 Lightning 30B-A3B (23 Mamba-2 + 23 MoE + 6 attention) | Image (Omni) | Yes | Yes | Nemotron 3.5 Lightning: DSpark block drafting (separate drafter GGUF) | [nemotron.md](docs/models/nemotron.md) |
| Mistral 3 | `mistral3` | Mistral-Small-3.1-24B-Instruct | Image | No | No | — | [mistral3.md](docs/models/mistral3.md) |
| Hunyuan Dense | `hunyuan-dense` | Tencent dense Hunyuan decoders, e.g. Hy-MT2 (GQA with per-head QK-norm applied *after* NeoX RoPE, SwiGLU) | Text only | No | No | — | [hunyuan-dense.md](docs/models/hunyuan-dense.md) |
| Muse-Glimmer | `muse-glimmer`, `muse_glimmer` | Muse-Glimmer-30B (interleaved SWA + NoPE full layers, attention output gate) | Image | Yes | Yes (ATEM) | Yes (DFlash block drafter, separate GGUF) | [muse-glimmer.md](docs/models/muse-glimmer.md) |
| DiffusionGemma | `diffusion-gemma`, `diffusion_gemma` | diffusion-gemma text-diffusion GGUFs | Text only | No | No | — | [diffusiongemma.md](docs/models/diffusiongemma.md) |
| Qwen-Image-Edit | `qwen_image`, `qwen-image` | qwen-image-edit MMDiT GGUFs (+ VAE & Qwen2.5-VL) | Image edit (image+text → image) | No | No | — | [qwenimage.md](docs/models/qwenimage.md) |
| MiniMax-H3 | `minimax-h3`, `minimax_h3` (the published GGUFs carry no metadata at all, so they are detected from their tensors) | MiniMax-H3 FL2VA / Ref2VA (19.3B packed audio-video DiT + Qwen3-VL-32B text encoder, video VAE, audio VAE) | Video **+ 32 kHz stereo audio** out (text→video, image→video, first/last frame, reference→video) | No | No | — | [minimax-h3.md](docs/models/minimax-h3.md) |
| Wan video | `wan`, `wan2.1`, `wan2.2` | Wan 2.1 T2V 1.3B/14B, Wan 2.2 TI2V-5B, Wan 2.2 A14B T2V/I2V (two experts) | Video out (text→video, image→video) | No | No | — | [wan.md](docs/models/wan.md) |

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

Models too large for that 16 GB rig carry their own head-to-head in their card, measured the same way (both engines, same GGUF, same machine, back to back): [GLM-5.2 744B-A40B on 3x RTX PRO 6000](docs/models/glm.md#performance) — TensorSharp leads prefill from ~1k prompt tokens up (pp2048 **1.20×**, pp4096 **1.21×**) and decode by 1.04×, with llama.cpp a few percent ahead on short prefills. The non-Flash [GLM-5.3](docs/models/glm.md#glm-53-glm-dsa) has its own, on 8× A40 46 GB without NVLink (UD-Q2_K_XL, 10,531-token prompt, 300 decode tokens, median of 3, whole-layer placement): decode is a tie at **20.48** tok/s against llama.cpp's 20.28, TensorSharp prefills at 251.6 tok/s and loads the 236.4 GiB checkpoint **2.9× faster** (264 s against 753 s), and the honest gap is time to first token — 41.9 s against 29.0 s, about **1.4× slower**. llama.cpp's prefill tok/s was not recorded for that cell. Full method and per-repeat numbers: [`docs/validation/cross-engine-2026-09/README.md`](docs/validation/cross-engine-2026-09/README.md). llama.cpp is a valid reference engine for `glm-dsa`, but not for `glm5next` (GLM-5.3-Flash).

## Documentation

New here? The sections above are all you need to get running. Everything else is detailed reference:

| Doc | What's inside |
|---|---|
| [Book guide: From Tensors to Tokens](docs/BOOK.md) | A guided path from tensor fundamentals to a multimodal Gemma 4 E4B inference engine, with publication details and links into the companion repository |
| [Model Downloads](MODEL_DOWNLOADS.md) | Per-model `huggingface-cli` download + run quick reference (quant tiers, projectors, companions) |
| [Usage](USAGE.md) | Full CLI reference (options, interactive REPL, JSONL batch), server hosting, logging, HTTP API examples, backends, and the env-var matrix |
| [Features](FEATURES.md) | Deep dives on continuous batching, speculative decoding, tool calling, thinking mode, multimodal, MoE, KV codecs, and more |
| [Configuration files](config/README.md) | Put options in a reusable JSON file with `${variables}` and auto-downloading models |
| [Development](DEVELOPMENT.md) | Prerequisites, building the native GGML/MLX libraries, repository layout, package boundaries, internal architecture, and the test harness |
| [Per-model architecture cards](docs/models/README.md) | End-to-end docs of each architecture (forward graph, components, parameters, prefill/decode optimizations) |
| [Paged attention & continuous batching](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md) | The vLLM-style paged KV cache, prefix sharing, and iteration-level scheduler |
| [Agent Skills & agentic work](docs/agent_skills.md) | The `SKILL.md` format, progressive disclosure and its budget, the in-process tool loop, sandboxed code execution, workspaces and artifacts, the path/ZIP/exec security model, and the HTTP + C# surfaces |
| [Speculative decoding](docs/speculative_decoding.md) | The three-layer design (model adapter / algorithm / speculator weights), the shipped `auto` / `draft-head` / `block` / `ngram` algorithms, and what to write to add a new one |
| [Environment variable feature matrix](docs/env_var_feature_matrix.md) | Which high-impact runtime flags affect which models, backends, and prompt types |
| [Engine comparison report](docs/engine_comparison_report.md) | Full per-scenario TensorSharp vs llama.cpp / stable-diffusion.cpp tables |
| [ggml_metal vs llama.cpp](docs/perf/metal-vs-llama-cpp.md) | Head-to-head prefill/decode on Apple Silicon, the four graph-construction gaps it found, and what each was worth |
| [Test/benchmark matrix runner](TensorSharp.TestMatrix/README.md) | Sweep model × backend × feature × env-var cells and generate regression reports |
| [Server API examples](TensorSharp.Server.Host/API_EXAMPLES.md) | Complete curl and Python examples for the server surface |

## Current Status

| Area | Status |
|---|---|
| Model families | DeepSeek V4 Flash (`deepseek4`), DeepSeek V4.1 Flash (`deepseek41`), GLM 5.x (`glm-dsa`, `glm5next`), Gemma 4, DiffusionGemma, Qwen 3.5/3.6-family (`qwen35`, `qwen35moe`, `qwen3next`), Qwen 3.8 Flash Next (`qwen4exp`), GPT OSS, Nemotron-H (incl. Nemotron 3 Nano Omni and Nemotron 3.5 Lightning, `nemotron_h_moe`), Mistral 3, Hunyuan Dense (`hunyuan-dense`), Muse-Glimmer (`muse-glimmer`, `muse_glimmer`). Image editing via Qwen-Image-Edit (`qwen_image`, `qwen-image` MMDiT); joint video-and-audio generation via MiniMax-H3 (`minimax-h3`, `minimax_h3`) and video-only generation via Wan 2.1 / 2.2 (`wan`, `wan2.1`, `wan2.2`). |
| Inference hosts | CLI, interactive REPL, ASP.NET Core web UI, Ollama-style API, OpenAI Chat Completions-style API, and OpenAI Responses-style API. |
| iOS application | TensorAgent targets iOS/iPadOS, links GGML as an iOS `.xcframework`, and uses `ggml_metal` on physical devices. It shares the host-neutral chat pipeline (`TensorSharp.Chat`) but serves its own phone-shaped page from an in-process loopback host, because iOS has neither an ASP.NET Core runtime pack nor child processes. Generation survives the app leaving the screen, the shared-prompt prefix checkpoint is persisted per model so the first message of a launch costs a restore instead of a full prefill (measured on iPhone 17 Pro Max with Qwen3.5 9B: a 54 s cold first message becomes a 1.2 s warm-up and a ~0.6 s first message), and the engine's memory policy is sized against what iOS jetsam actually charges. See [TensorAgent](TensorAgent/README.md). |
| Backends | Pure C# CPU, direct CUDA/cuBLAS (`cuda`), MLX Metal (`mlx`), GGML CPU, GGML Metal, GGML CUDA, GGML Vulkan. DeepSeek V4 additionally has three whole-model executors of its own — direct-CUDA, native ggml, and a pure-C# CPU one — each layer-splitting the weights across every visible GPU (`--tp N` / `TS_DSV4_NGPU` caps the count). DeepSeek V4.1 serves on `ggml_cuda`; `ggml_cpu` runs the same native graph on scalar fallbacks and `cpu` runs a pure-C# V4.1 executor, both as correctness and portability paths rather than serving ones. `cuda` runs V4.1 through the direct-CUDA engine's own kernels (no ggml), and is not yet held to a numerical gate. `ggml_vulkan` / `ggml_metal` need `TS_DSV41_ALLOW_NON_CUDA_GPU=1`; `mlx` refuses the checkpoint rather than loading V4.1 weights into a graph that does not implement it. Among the video families, Wan is the one that restricts its backends: it runs on the GGML backends and on the direct `cuda` / pure-C# `cpu` ones, but not on MLX. |
| Multimodal | Gemma 4 image/video/audio; Qwen 3.5-family, Qwen 3.8 Flash Next, GLM-5.3-Flash, Mistral 3, Nemotron-H Omni, Muse-Glimmer image input; PDF documents (CLI `--pdf` + Web UI). Media *out*: Qwen-Image-Edit (image), MiniMax-H3 (H.264 MP4 **plus a 32 kHz stereo `.wav` sidecar**, generated together in one packed latent), and Wan 2.1 / 2.2 (H.264 MP4 video only, text→video and image→video). |
| Continuous batching | vLLM-style paged KV cache, block-hash prefix sharing, shared-prefix checkpoints (the state at the end of the prompt every conversation shares is cloned into each new chat, so a new chat re-prefills only its own message; Gemma 4 and Qwen 3.5/3.6 on GGML, and a host can persist one across process restarts via `IPrefixCheckpointStore`), iteration-level scheduler (default on; opt-out `--no-continuous-batching`). The paged pool is host-resident, so it buys memory efficiency and prefix reuse rather than throughput that scales with concurrency. DeepSeek V4 and GLM 5.x serve through their own native per-sequence slots on the same engine — a compressed MLA cache row per token has no paged layout to page — and GLM adds a default-on batched fused decode (set `TS_BATCHED_FUSED_DECODE=0` to use serial fused decode; 1.81x aggregate at 4 concurrent requests). Qwen 3.8 Flash Next uses per-sequence state holders for the same reason — its GatedDeltaNet, PLE and indexer state has no paged layout either. |
| Speculative decoding | MTP / NextN draft heads on Qwen 3.6, GLM 5.2 and GLM-5.3 (all embedded in the checkpoint — GLM-5.3's `blk.78` NextN block is complete but ships no LM head of its own, so speculation engages on the default layer split, without `--tp`) and Gemma 4 (separate draft GGUF, loaded via `--draft-model`); DSpark block drafting on DeepSeek V4 (`cuda` / `ggml_cuda` only), DFlash / DFlash2 block drafting on Muse-Glimmer and Qwen 3.8, and DSpark Markov-head block drafting on Nemotron 3.5 Lightning — the first recurrent (Mamba-2) trunk to take a block drafter — all loading a separate drafter GGUF via `--draft-model`; plus a weight-free n-gram (prompt-lookup) speculator that needs no drafter at all and therefore works on every checkpoint, selected with `--spec-type ngram`. Every emitted token is drawn from a trunk row with the run's own sampler, so the emitted stream is the one plain decoding would have produced. Off by default; opt in with `--spec` on either host for the embedded heads, while passing `--draft-model` enables speculation by itself for any drafter shipping as its own GGUF. |
| Tensor parallelism | Megatron-LM column/row-parallel TP on the direct `cuda` backend and on GGML CUDA / Vulkan (`--tp N` / `TENSORSHARP_TP_DEGREE`, CLI and server); distributed multi-node TP via peer-to-peer TCP (`--tp-node-id` / `--tp-peers`), with hierarchical AllReduce and automatic host-staging fallback when CUDA P2P is unavailable. All autoregressive architectures; MoE expert parallelism and fused per-rank decode/prefill graphs for Gemma 4 and Qwen 3.5/3.6 on GGML. GLM 5.x uses a layer split by default, but `--tp N` selects its native local/single-process TP path on GGML GPU backends for GLM 5.2, GLM-5.3 and GLM-5.3-Flash (including KDA/MLA head and routed-expert-row sharding on GLM-5.3-Flash); the whole GLM family hard-refuses `--tp-node-id` / `--tp-peers` before the model is built, and on GLM-5.3 `--tp N` is an accepted mode rather than a validated one — it replicates the MLA and indexer caches per rank, so the footprint multiplies and the fitted context shrinks, and nothing above `--tp 1` has been run on that checkpoint (`--tp 8` works out to 41.7 GiB per rank, which does not fit a 46 GB card). Architectures that shard no weights take the same `--tp N` as a layer split — a contiguous run of whole layers per GPU, as with DeepSeek V4 and V4.1 (on V4.1, `TS_DSV41_TP=N` additionally enables experimental routed-MoE tensor parallelism over those same GPUs, which has measured slower than the layer split so far); on Qwen 3.8 Flash Next (`qwen4exp`) `TS_Q4E_LAYER_SPLIT=20,28` overrides the automatic balance and throws rather than ignoring a split it cannot honour. Startup prints which mode ran and the per-GPU layer/byte split, and an architecture that supports neither mode says so on stderr and runs on one GPU. Optional Redis-backed KV cache and Responses API store. |
| Agent Skills | Skill directories discovered from `--skills-dir` (or a `skills` folder beside the binary) and installable at runtime through `POST /api/skills`. Selected per request with `"skills": [...]` on `/v1/chat/completions`, `/v1/responses`, `/api/chat/ollama` (Ollama) and `/api/chat` (Web UI), or with `--skill` on the CLI. Tool-capable families receive metadata and activate instructions through built-in `skills_list` / `skills_read` calls answered in process; the caller's own tools are still returned to it. Script execution (`skills_run`) is off unless `--skills-allow-exec` is passed. Mistral 3 and families without a parsable tool protocol, including `qwen4exp`, instead receive selected skill bodies inline and are not offered skill/code tools. |
| Agentic code work | Optional `--code-exec` offers `read_file`, `edit_file`, `write_file`, `shell`, and atomic `apply_patch` inside the same bounded model-to-tool loop. Web/CLI chats keep a session workspace; each OpenAI/Ollama request gets a private workspace across internal repair rounds and loses it after the response. Artifacts are captured for download. This is a single-model in-process loop: TensorSharp does not provide multi-agent delegation or a per-command approval workflow. |
| Sandboxing & permissions | Code execution and skill scripts are off by default. macOS uses Seatbelt and Linux requires `bwrap` 0.12.0+; required mode refuses when confinement is unavailable. Windows code execution requires explicit `--code-exec-unconfined`, while Windows skill scripts require explicit `--skills-sandbox preferred` to accept job-object-only containment. Script network, code network, and host-performed package installation are separate opt-ins. |
| Distribution (verified 2026-09-08) | GitHub release v3.3.0.0 provides ten self-contained CLI/Server archives across Windows x64 CPU/CUDA, Linux x64 CPU/CUDA, and macOS arm64. The source tree's publish set is now **thirteen** NuGet packages (`eng/verify-packages.ps1` is the authoritative list and gates the release workflow). NuGet.org still carries only eight of them — Tensors, Runtime, Models, GGML/CUDA/MLX backends, Server, and CLI at **3.1.2** (July 2026) — which lag the source and the binary release; the published `TensorSharp.Server` predates the logging and chat splits. Runtime.Logging, AgentHost, Chat, Server.Host, and Distributed are packable and verified but not yet pushed, so those layers still need project references from a source checkout. |
| Server model scope | One explicitly hosted GGUF via `--model`; optional explicit projector via `--mmproj`; no directory scanning. |
| Observability | Structured per-turn logs, queue status, and KV-cache reuse metrics across Web UI, Ollama, and OpenAI shapes. |

## Author

Zhongkai Fu

## License

See [LICENSE](LICENSE) for details.
