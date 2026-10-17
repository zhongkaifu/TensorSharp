# Getting started

[English](getting_started.md) | [中文](getting_started_zh-cn.md)

> Part of the [TensorSharp](../README.md) documentation. The README's [Quick Start](../README.md#quick-start) is the short version of this page.

## Install and first run

**For the TensorAgent Desktop app:** open the [latest release](https://github.com/zhongkaifu/TensorSharp/releases/latest), expand **Assets**, and choose `tensoragent-desktop-<version>-osx-arm64.dmg` (Apple Silicon, macOS 14+) or `tensoragent-desktop-<version>-win-x64-cpu.msi` / `win-x64-cuda.msi` (Windows x64). PKG and ZIP alternatives are available. Desktop packages include the .NET runtime; Windows also needs WebView2 if absent. Model weights download in **☰ → Models → Download → Use**, and the built-in catalog starts at 12 GB system RAM. Follow the [Desktop user guide](tensoragent_desktop.md) for installation, package verification, first chat, tools and troubleshooting. Older releases may lack Desktop assets until a release uses the updated Release Binaries workflow.

**For the TensorSharp CLI/server:** the [Releases page](https://github.com/zhongkaifu/TensorSharp/releases) provides self-contained archives for Windows x64 (CPU/CUDA), Linux x64 (CPU/CUDA), and macOS arm64. The SDK and build steps below are for source builds.

**NVIDIA DGX Spark / GB10:** use the separate experimental **CUDA 13, Linux ARM64**
[Docker build and archive instructions](../DEVELOPMENT.md#gb10--dgx-spark-build-container-experimental).
Its archives end in `linux-arm64-cuda13-GB10`; they target a single GB10, not
generic ARM64 GPUs. Tagged releases also publish them, built in Docker on a hosted
ARM64 runner without a GPU. A historical real-hardware check of CLI and server
text inference predates the upstream reintegration and does not certify the
current code; hosted CI rechecks only the CPU and archive paths. The existing x64
CUDA archives are not suitable for the Spark.

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

See the [cross-platform .NET install overview](https://learn.microsoft.com/en-us/dotnet/core/install/) or [Development → Prerequisites](../DEVELOPMENT.md#prerequisites) for more detail.

Then get running in ~30 seconds on the verified native GGML fast path — Gemma 4 E4B. The other prerequisites are `git`, `curl`, [CMake](https://cmake.org/download/) 3.20+ (the native GGML library is configured and built with it — on Windows, Visual Studio's "C++ CMake tools for Windows" component ships one and the build will find it), and the toolchain for your GPU backend (see [Development → Prerequisites](../DEVELOPMENT.md#prerequisites)). The recommended public file is [`gemma-4-E4B-it-Q8_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q8_0.gguf) (7.48 GiB); text-only inference needs no projector.

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
`ggml_vulkan`). Use `--tp N` only for tensor parallelism. Use the separate
`--layer-split N` option for whole-layer placement on Qwen 3.8 Flash Next,
DeepSeek V4 / V4.1, and GLM 5.x: one contiguous run of whole layers per GPU.
The modes are mutually exclusive, and unsupported requests fail at startup.
Layer splitting is local to one node; it cannot use `--tp-node-id` / `--tp-peers`.
Existing layer-split commands must replace `--tp N` with `--layer-split N`
(or `TENSORSHARP_LAYER_SPLIT_DEGREE=N`). With neither mode configured, inference uses one device. GLM 5.x accepts
`--layer-split N` for whole-layer placement and `--tp N` for its native local
tensor-parallel path on the GGML GPU backends. For Qwen-Image-2.1,
`--tp N` shards only the diffusion transformer; its text/vision encoders and VAE
stay on the first GPU. Install the CUDA toolkit first, then:

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
**[Tensor Parallelism & Distributed Inference](../USAGE.md#tensor-parallelism--distributed-inference)**.


Host the same model as a server (browser UI at <http://localhost:5000>, plus Ollama/OpenAI APIs):

```bash
dotnet run --project TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --max-tokens 512
```

> The server binds `0.0.0.0:5000` by default (change it with `--port` / `--host`, or the `PORT` / `HOST` environment variables; on macOS port 5000 is taken by the AirPlay Receiver) with no built-in auth or TLS — keep it behind a firewall or an authenticated HTTPS reverse proxy. For image/video/audio add the companion [`mmproj-gemma-4-E4B-it-Q8_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/mmproj-gemma-4-E4B-it-Q8_0.gguf) with `--mmproj`.

TensorSharp.Server.Host, TensorSharp.Cli, and TensorAgent use the shared engine's Radix
KV prefix cache by default for every autoregressive family in the [supported models](supported_models.md) tables
(not DiffusionGemma or the image/video models). It reuses public prompt prefixes
and each conversation's private state, respecting model and media boundaries;
speculative decoding (`--spec`) keeps it on.
Set `TS_SCHED_PREFIX_CACHE=0` to disable runtime prefix reuse.
Server and CLI `--no-prefix-cache` also disable prefix reuse and startup warmup;
on the server it also turns off the on-disk prefix checkpoints.

Both executables print their full option reference — description, default, range, and an example per flag — when started with no arguments or with `--help`:

```bash
dotnet run --project TensorSharp.Cli -c Release -- --help
dotnet run --project TensorSharp.Server.Host -c Release -- --help
```

Full command reference: **[CLI](../USAGE.md#console-application)** · **[Server](../USAGE.md#web-application)** · more models to download: **[Model Downloads](../MODEL_DOWNLOADS.md)** · prefer a config file? **[config/](../config/README.md)**.

## Build TensorAgent from the same solution

`dotnet build TensorSharp.slnx -c Release` builds the engine, CLI/server and TensorAgent.Core, TensorAgent.Sharing and TensorAgent.Tests. On macOS it also attempts the Mac app and, on Apple Silicon, the iOS simulator app; on Windows it attempts the `win-x64` app. App builds need the matching MAUI workloads and native prerequisites. Missing prerequisites skip a head with a warning; other build failures fail the solution build. Linux builds the shared projects without a MAUI app head. To leave the app out explicitly, use `-p:TensorSharpSkipTensorAgentApp=true`.

Building the test project does not run its tests. The Windows app has bounded Debug/Release chat, synthetic vision and tool validation; media generation and broader device coverage remain unverified. Consult the [TensorAgent source/validation guide](../TensorAgent/README.md) for current coverage and app-specific scripts, or the [Desktop user guide](tensoragent_desktop.md) to install a Mac/Windows release package. iPhone/iPad builds still require the source-build workflow.

## Text and code embeddings

Current source supports **Snowflake Arctic Embed L v2.0** and **all-MiniLM-L6-v2** GGUF encoders, serving normalized vectors through OpenAI `/v1/embeddings` and Ollama `/api/embed`. After the source build above, start the small MiniLM service:

```bash
curl --create-dirs -fL -o models/embeddings/all-MiniLM-L6-v2-Q8_0.gguf \
  https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/544f204f2eaa2d71361ffc74d6df7170285b286a/all-MiniLM-L6-v2-Q8_0.gguf
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model models/embeddings/all-MiniLM-L6-v2-Q8_0.gguf \
  --embeddings --backend cpu --host 127.0.0.1 --port 5001 --no-webui
```

```bash
curl http://127.0.0.1:5001/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"all-MiniLM-L6-v2-Q8_0","input":["read a file","open a document"]}'
```

Use `cpu` for 100% pure C# execution without native inference libraries, or `ggml_cpu`, `ggml_metal`, and `ggml_cuda` for native GGML execution; run chat and embedding services separately. See the [embedding guide](embeddings.md) for Snowflake downloads, batching, dimensions, the C# API, tokenization, and performance validation.

## New video, embedding and skill workflows

- [LTX 2.5 Distilled](models/ltx-2.5.md): the five-file preset downloads 44.76 GB and needs approved publisher access. Its experimental 48 GB tier starts with 1024×1024×17 frames and an 8+3-step two-stage recipe, with 48 kHz stereo audio.
- [MiniMax-H3](models/minimax-h3.md): choose FL2VA for first-frame animation or Ref2VA for identity references. Both produce 32 kHz stereo audio; their current precision and benchmark notes supersede earlier speed claims.
- [EmbeddingGemma 2](embeddings.md): use `--embeddings`, optionally `--mmproj` for media. Six backends are implemented; task prompts, dimensions and test limits are model-specific.
- [Agent Skills](agent_skills.md): desktop CLI/server builds bundle 17 skills. With `find-skills` reachable and `--skills-allow-network`, native tools search, inspect and install a pinned skill for same-request use. Running scripts still requires execution permissions and dependencies.

## Reuse your files as a knowledge library

The `v2026.10.11` source includes personal knowledge collections and response
steering. LTX 2.5, EmbeddingGemma 2 and native skill discovery require newer source;
see [project status](PROJECT_STATUS.md) before choosing a release.
In TensorAgent, open **☰ → Knowledge**, create a collection and upload files or
folders, or use **Add chats** to index snapshots of saved chats with available
attachments and generated files. Inspect each source's extraction status, then
choose **Use in this chat** before asking questions. Uploading alone does not
select a collection. Cited answer links open the retained original and excerpts.

The library supports text, code, datasets, partial Office extraction and local
PDF OCR. Images and sampled video need a compatible loaded vision model; audio can use
a loaded audio tower or a configured `TS_JEV_TRANSCRIPTION_URL` companion.
**Reindex** processes the original again after capabilities change.
Retrieval defaults to BM25, with optional GGUF embeddings for semantic search.
The [knowledge guide](knowledge-system.md) explains coverage, limits and citations.

The standalone server enables the same store through `TS_KNOWLEDGE_DIRECTORY`:

```bash
export TS_KNOWLEDGE_DIRECTORY="$HOME/TensorSharpKnowledge"
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_metal \
  --host 127.0.0.1 --port 5001
```

Use a backend available on your host. Manage sources with `/api/knowledge` and
send their collection IDs as `knowledgeCollections` on a supported chat request;
see the [API examples](../TensorSharp.Server.Host/API_EXAMPLES.md#personal-knowledge-library).
The standalone browser UI provides response steering; TensorAgent provides the
collection-management interface. While the CLI, TensorAgent or server browser UI
is reasoning or answering, submit a new instruction to interrupt and continue
with that guidance. **Stop**, or Ctrl+C in the CLI, interrupts without a new
message. Redirected CLI input remains sequential.

## Pick a Backend

Backend support depends on the model architecture. Embedding models support pure C# `cpu` and native `ggml_cpu`, `ggml_metal`, and `ggml_cuda`; see the [status matrix](PROJECT_STATUS.md#status-matrix) for other model-specific limits.

| Your hardware | Recommended backend | Flag | Notes |
|---|---|---|---|
| **Apple Silicon (Mac)** | GGML Metal | `--backend ggml_metal` | The server's default on macOS; the CLI defaults to `ggml_cpu` on every OS, so pass the flag there. `--backend mlx` is an alternative Apple-Silicon GPU path. |
| **Windows / Linux + NVIDIA GPU** | GGML CUDA | `--backend ggml_cuda` | Most-tested NVIDIA path. `--backend cuda` is the direct PTX/cuBLAS backend for experimentation. |
| **Windows / Linux + AMD / Intel / NVIDIA GPU** | GGML Vulkan | `--backend ggml_vulkan` | Vendor-neutral GPU path via ggml-vulkan. Built automatically when a Vulkan runtime is present; `--no-vulkan` opts out. |
| **No GPU / portability / debugging** | Pure C# CPU | `--backend cpu` | No native dependencies; matmuls run on a multi-core worker pool. Even DeepSeek V4.1 Flash has a whole-model executor here — it runs on the pure-C# `DeepSeek4CpuExecutor` with no ggml and no GPU, held to the PyTorch oracle `eng/dsv41-reference.py` at `atol=rtol=2e-5` on a five-layer F32 fixture (architectural agreement with the oracle, not parity on the real Q2_K weights), as a correctness and portability path rather than a serving one. For faster CPU inference use `--backend ggml_cpu` (native kernels). |

Full per-backend description: [Usage → Compute Backends](../USAGE.md#compute-backends).

## Make It Fast

Start with these choices, in order:

1. **Choose the right checkpoint.** For Wan video, use a Turbo/Lightning/4-step distilled GGUF. For Qwen-Image-2.1, add a step-distillation [LoRA plug-in](models/qwenimage21.md#lora-plug-ins) from `config/lora/` (4–8 steps instead of the 40-step default).
2. **Use the matching backend.** NVIDIA: `ggml_cuda`; Apple Silicon and iOS: `ggml_metal`; CPU: `ggml_cpu` (use managed `cpu` for portability).
3. **Reduce work before tuning flags.** For H3 use `--cfg 1.0`, and for fewer steps add a [Turbo plug-in](models/minimax-h3.md#turbo-plug-ins) on FL2VA (8 or 4 steps) rather than lowering the base step count; for media, lower resolution, frame count, or steps.
4. **Then scale or speculate.** Try `--draft-model` / `--spec`, `--n-cpu-moe`, or `--tp N` when the model or workload calls for it.

See the [performance guide and detailed fast lanes](PROJECT_STATUS.md#make-it-fast), the [model cards](models/README.md), and the [environment-variable matrix](env_var_feature_matrix.md) for trade-offs and measurements.
