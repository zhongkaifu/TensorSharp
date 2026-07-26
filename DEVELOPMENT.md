# Development
[English](DEVELOPMENT.md) | [中文](DEVELOPMENT_zh-cn.md)

> Part of the [TensorSharp](README.md) documentation: how to build TensorSharp, the repository layout, package boundaries, internal architecture, and the test harness.

## Prerequisites

### Install the .NET 10 SDK

Every TensorSharp project targets `net10.0`. To build from source, install the full **.NET 10 SDK**; a runtime-only installation is not sufficient. Installing the SDK also installs the .NET and ASP.NET Core runtimes needed by the CLI and server.

| Platform | Recommended installation |
|---|---|
| **Windows** | Open PowerShell and run `winget install Microsoft.DotNet.SDK.10`. Alternatively, choose the .NET 10 **SDK** installer for your architecture in Microsoft's [Windows installation guide](https://learn.microsoft.com/en-us/dotnet/core/install/windows). |
| **macOS** | Download and run the [.NET 10 SDK installer](https://dotnet.microsoft.com/en-us/download/dotnet/10.0). Choose **Arm64** for Apple silicon or **x64** for an Intel Mac; see Microsoft's [macOS installation guide](https://learn.microsoft.com/en-us/dotnet/core/install/macos). |
| **Linux** | Use Microsoft's [Linux installation guide](https://learn.microsoft.com/en-us/dotnet/core/install/linux) to select your distribution and release, configure its package source, and install the .NET 10 SDK package (commonly `dotnet-sdk-10.0`). Package feeds and supported architectures vary by distribution, so follow the linked distro-specific steps. |

After installation, open a new terminal and verify that a `10.0.x` SDK appears:

```bash
dotnet --list-sdks
```

See Microsoft's [cross-platform .NET installation overview](https://learn.microsoft.com/en-us/dotnet/core/install/) for installer, package-manager, manual, and non-admin options.

### Other build prerequisites

- **`git` and network access:** the GGML/CUDA native builds clone the ggml sources from [github.com/ggml-org/ggml](https://github.com/ggml-org/ggml) into `ExternalProjects/ggml/` on first build (see `eng/fetch-ggml.sh` / `eng/fetch-ggml.ps1`). The clone tracks ggml's default branch (`master`); pin a different ref with `TENSORSHARP_GGML_GIT_REF`, or set `TENSORSHARP_GGML_NO_UPDATE=1` to skip the network update once cloned (offline rebuilds)
- **macOS (Metal backend):** CMake 3.20+ and Xcode command-line tools for building the native GGML library; the MLX backend additionally builds `libmlxc` from `TensorSharp.Backends.MLX/Native/` via `bash TensorSharp.Backends.MLX/build-native-macos.sh`
- **Windows (GGML CPU / CUDA backends):** CMake 3.20+ and Visual Studio 2022 C++ build tools; for `ggml_cuda` or `cuda`, install an NVIDIA driver plus CUDA Toolkit 12.x or another compatible CUDA toolkit with cuBLAS
- **Linux (GGML CPU / CUDA backends):** CMake 3.20+; for `ggml_cuda` or `cuda`, install an NVIDIA driver plus CUDA Toolkit 12.x or another compatible CUDA toolkit with cuBLAS
- **Windows (GGML Vulkan backend):** enabled automatically when the machine has a Vulkan runtime (`System32\vulkan-1.dll`, shipped by every recent GPU driver). With a [LunarG Vulkan SDK](https://vulkan.lunarg.com/) installed it is used directly; without one the build auto-provisions a portable toolchain (Vulkan-Headers, a vulkan-1 import library generated from the system loader, glslc, SPIRV-Headers) into `ExternalProjects/vulkan-toolchain/` via `eng/fetch-vulkan-toolchain.ps1`. Opt out with `build-windows.ps1 --no-vulkan` or `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF`. A GPU driver with Vulkan 1.3 support is required at runtime
- **Linux (GGML Vulkan backend):** enabled automatically when a Vulkan loader (`libvulkan.so.1`) is installed. Distro dev packages are used when present (`apt install libvulkan-dev glslc spirv-headers`); otherwise the build auto-provisions the missing pieces (Vulkan-Headers, glslc from the shaderc CI prebuilts, SPIRV-Headers) into `ExternalProjects/vulkan-toolchain/` via `eng/fetch-vulkan-toolchain.sh`. Opt out with `build-linux.sh --no-vulkan` or `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF`
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

On macOS this compiles `libGgmlOps.dylib` with Metal GPU support. On Windows and Linux, the native scripts preserve an existing CUDA-enabled build and auto-enable GGML_CUDA when a CUDA toolchain is detected; `build-windows.ps1 --cuda`, `build-linux.sh --cuda`, and `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` force CUDA explicitly. The GGML Vulkan backend is auto-enabled the same way when the machine has a Vulkan runtime, downloading its build toolchain on first use; `--vulkan` / `--no-vulkan` or `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON/OFF` force the choice explicitly, and an explicit choice sticks across rebuilds (see [Prerequisites](#prerequisites) for the Vulkan toolchain the build auto-provisions). The build output is automatically copied to the application's output directory.

The direct `cuda` backend is built as managed C# plus PTX kernels. During `dotnet build`, `TensorSharp.Backends.Cuda` compiles `native/kernels/*.cu` to `native/ptx/*.ptx` when `nvcc` is available; if `nvcc` is missing, the build continues and PTX-backed ops use CPU fallbacks. cuBLAS-backed GEMM still requires the CUDA runtime libraries to be discoverable at run time.

### Build the native MLX library (macOS only)

The MLX backend depends on `libmlxc` (the C bindings for [MLX](https://github.com/ml-explore/mlx)). The repository pins a known-good tag of `mlx-c` in `TensorSharp.Backends.MLX/Native/MLX_C_VERSION` and a helper script fetches and builds it:

```bash
bash TensorSharp.Backends.MLX/build-native-macos.sh
```

The script writes the resulting libraries (`libmlxc.dylib`, `libmlx.dylib`, and any backend deps) into `TensorSharp.Backends.MLX/Native/dist/`. At run time the backend probes the application directory first; you can also point it to a custom install with `TENSORSHARP_MLX_LIBRARY=<path-to-libmlxc.dylib>` or `TENSORSHARP_MLX_LIBRARY_DIR=<dir-with-libmlxc>`. If the library cannot be located the backend reports unavailable and `--backend mlx` is rejected at startup.


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
│   ├── TurboQuantKvCodec.cs     # Quantized KV block codec (2-bit / Q4 / Q8) implementing IKvBlockCodec
│   ├── PrefillChunking.cs       # Chunked-prefill helper used by SWA / very long prompts
│   ├── KvBlockHash.cs           # Content-addressed block hash for prefix-cache sharing
│   └── Logging/                 # JSON-line file logger + per-turn telemetry
├── TensorSharp.Models/          # Model architectures and multimodal encoders/injectors
│   ├── Models/<Family>/         # One folder per architecture (DiffusionGemma, Gemma3, Gemma4, GptOss, Mistral3, Nemotron, Qwen3, Qwen35, QwenImage)
│   │   ├── <Family>Model.cs                # Legacy per-sequence ModelBase implementation
│   │   └── <Family>Model.BatchedForward.cs # IBatchedPagedModel.ForwardBatch — batched/paged path (Mistral3, Gemma4, GptOss, Qwen35, Nemotron, Qwen3)
│   ├── Paged/                   # Tensor-side paged-attention helpers (TensorPagedAttention)
│   ├── KvBlockTransfer.cs       # Helpers for extract/inject of KV blocks across sequences
│   ├── MtpSpeculativeDecoder.cs # MTP/NextN draft-verify-rollback driver shared by Qwen 3.6 and Gemma 4
│   └── ModelMultimodalInjector.cs # Vision / audio / video embedding injection
├── TensorSharp.Backends.GGML/   # GGML backend bindings (Metal/CUDA/Vulkan/CPU via native library)
├── TensorSharp.Backends.Cuda/   # Direct CUDA backend using CUDA Driver API, cuBLAS, and PTX kernels
├── TensorSharp.Backends.MLX/    # Apple Silicon MLX backend (mlx-c / Metal). Native bridge is built via `build-native-macos.sh`.
├── TensorSharp.GGML.Native/     # Native C++ bridge to ggml (builds libGgmlOps, split into focused source files)
│   ├── ggml_ops_core.cpp                  # Element-wise, reductions, basic shape ops
│   ├── ggml_ops_elementwise.cpp           # Element-wise / activation fusions
│   ├── ggml_ops_matmul.cpp                # GEMM / quantized matmul
│   ├── ggml_ops_fused.cpp                 # Cross-cutting fused per-layer kernels
│   ├── ggml_ops_norm_attn.cpp             # Norm + attention fusions
│   ├── ggml_ops_transformer.cpp           # Generic fused transformer layer/model decode + flash-attn decode
│   ├── ggml_ops_transformer_common.h      # Shared transformer helpers + C# layer-descriptor structs
│   ├── ggml_ops_transformer_prefill.cpp   # Fused layer prefill (Gemma 4, GPT-OSS, Qwen 3.5)
│   ├── ggml_ops_qwen35_decode.cpp         # Qwen 3.5/3.6 fused decode (layer, whole-model, batched)
│   ├── ggml_ops_qwen35_verify.cpp         # Qwen 3.5/3.6 fused multi-token verify
│   ├── ggml_ops_gemma4_decode.cpp         # Gemma 4 dense whole-model decode (CUDA-graph persisted)
│   ├── ggml_ops_gemma4_batched.cpp        # Gemma 4 dense + MoE token-batched decode
│   ├── ggml_ops_gemma4_verify.cpp         # Gemma 4 dense verify + MTP draft step
│   ├── ggml_ops_gemma4_moe.cpp            # Gemma 4 MoE layer/whole-model decode + verify
│   ├── ggml_ops_moe.cpp                   # Mixture-of-Experts forward / fused router
│   ├── ggml_ops_gated_delta_net.cpp       # Qwen 3.5/3.6 GatedDeltaNet kernels (per-seq + batched)
│   ├── ggml_ops_mamba2.cpp                # Nemotron Mamba2 kernels (per-seq + batched SIMD)
│   ├── ggml_ops_paged_attention.cpp       # Paged-attention native kernel (drives ggml_flash_attn_ext + sinks variant)
│   ├── ggml_ops_diffusion.cpp             # DiffusionGemma fused decode-layer / whole-model / lm-head kernels
│   ├── ggml_ops_qwen_image.cpp            # Qwen-Image-Edit MMDiT whole-model forward (CUDA-graph-captured) + CFG-batched kernels
│   ├── ggml_ops_training.cpp              # Training-only kernels (unused at runtime)
│   └── tests/                              # Native unit + smoke tests
├── TensorSharp.Server/          # Web chatbot + API server (ASP.NET Core)
│   ├── Program.cs               # Slim bootstrap: DI wiring, middleware, endpoint mapping, paged-KV + continuous-batching CLI translation
│   ├── ModelService.cs          # Facade that keeps the public server inference API stable; owns the InferenceEngineHost
│   ├── ModelLifecycleService.cs # Model load/dispose and backend selection (CPU / CUDA / MLX / GGML CPU/Metal/CUDA/Vulkan)
│   ├── InferenceEngineHost.cs   # DI-registered per-model InferenceEngine singleton (continuous batching entry point)
│   ├── ChatGenerationPipeline.cs # Prompt rendering, submits to InferenceEngine, streams tokens, stop handling
│   ├── InferenceTelemetry.cs    # Prompt/eval timing, TTFT, tokens/sec, bounded input summaries + output logs
│   ├── ChatHistoryPreparer.cs   # History normalization, raw-token splice helpers, multimodal order helpers
│   ├── ChatSession.cs           # Per-conversation tracked history + raw assistant tokens
│   ├── SessionManager.cs        # Thread-safe session registry (default + per-tab sessions)
│   ├── InferenceQueue.cs        # Backward-compatible queue-status surface (engine itself handles concurrency)
│   ├── BackendCatalog.cs        # Discovery of available compute backends (CPU / CUDA / MLX / GGML*)
│   ├── TextUploadHelper.cs      # Lossless text-upload normalization
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

## Project / NuGet Package Boundaries

The repository is split along package boundaries so consumers can depend on only the layers they actually need. These are buildable package projects and IDs, but the current Runtime/Models/Backends/CLI/Server packages are **not published on NuGet.org**. Use project references from a source checkout for now; do not copy `dotnet add package TensorSharp.Models` examples until a matching version appears on [NuGet.org](https://www.nuget.org/profiles/TensorSharp).

| Project | NuGet package | Public namespace | Responsibility |
|---|---|---|---|
| `TensorSharp.Core` | `TensorSharp.Tensors` | `TensorSharp` | Tensor primitives, ops, allocators, storage, and device abstraction |
| `TensorSharp.Runtime` | `TensorSharp.Runtime` | `TensorSharp.Runtime` | GGUF parsing, tokenizers, prompt rendering, sampling, output protocol parsing, paged KV cache, continuous-batching scheduler |
| `TensorSharp.Models` | `TensorSharp.Models` | `TensorSharp.Models` | `ModelBase`, architecture implementations, multimodal encoders, batched / paged forward passes, and model-side execution helpers |
| `TensorSharp.Backends.GGML` | `TensorSharp.Backends.GGML` | `TensorSharp.GGML` | GGML-backed execution and native interop |
| `TensorSharp.Backends.Cuda` | `TensorSharp.Backends.Cuda` | `TensorSharp.Cuda` | Direct CUDA allocator, storage, cuBLAS GEMM, PTX kernels, and quantized CUDA ops |
| `TensorSharp.Backends.MLX` | `TensorSharp.Backends.MLX` | `TensorSharp.MLX` | Apple Silicon MLX backend (mlx-c / Metal) with quantized / fused / compiled kernels and MoE expert offload |
| `TensorSharp.Server` | `TensorSharp.Server` | `TensorSharp.Server` | ASP.NET Core server, OpenAI/Ollama adapters, inference engine host, web UI |
| `TensorSharp.Cli` | `TensorSharp.Cli` | `TensorSharp.Cli` | Console host and debugging / batch tooling |

This split keeps engine users off the web stack, keeps API-layer changes from leaking into core/runtime packages, and makes future benchmark or eval-harness projects easier to publish independently.

> **Note:** the core layer ships as **`TensorSharp.Tensors`**, not `TensorSharp.Core`. The `TensorSharp.Core` id on NuGet.org is registered to an unrelated, abandoned project (all versions unlisted, source repo deleted), so pushing to it returns 403. Only the NuGet id differs — the project, assembly, and `TensorSharp` namespace are unchanged, so `using` statements are unaffected and only the `dotnet add package` line differs.

Validate package metadata and README dependency boundaries before publishing:

```powershell
pwsh ./eng/verify-packages.ps1
```

The verifier runs `dotnet pack` for the public packages above and fails if an internal dependency such as `AdvUtils` leaks into the `.nuspec`, or if a TensorSharp package depends on a layer outside this table.

### Publishing a package release (maintainers)

The [`Publish NuGet`](.github/workflows/publish-nuget.yml) workflow packs the public projects above on a version tag and pushes them to NuGet.org and GitHub Packages. This describes the release process, not current package availability:

```bash
git tag vX.Y.Z.W      # the tag drives package version X.Y.Z.W
git push origin vX.Y.Z.W
```

- The tag (with the leading `v` stripped) overrides `TensorSharpVersion` for every package, so all packages ship with a single coordinated version. You do not need to edit `Directory.Build.props` first.
- Packing is managed-only — the native GGML/CUDA/MLX libraries are not embedded in the packages — so the workflow runs on a stock runner with `eng/verify-packages.ps1 -SkipNativeBuild` (which also sets `TensorSharpSkipGgmlNative=true` / `TensorSharpSkipMlxNative=true`).
- NuGet.org publishing uses [Trusted Publishing](https://learn.microsoft.com/en-us/nuget/nuget-org/trusted-publishing) (OIDC) — there is no API key secret to manage. `NuGet/login@v1` exchanges the job's GitHub OIDC token for a key valid for one hour. The policy on nuget.org pins the repository owner, repository, workflow **file name** (`publish-nuget.yml`), and the `production` environment, so renaming this workflow file, or removing `environment: production` from the job, breaks publishing until the policy is updated to match.
- Packages are pushed individually rather than by glob: a package id owned by a different NuGet.org account returns 403, which `--skip-duplicate` does *not* absorb, so the run reports exactly which packages were rejected instead of stopping at the first one.
- Packages ship with Source Link and a companion `.snupkg` symbol package. `ContinuousIntegrationBuild` is set only under GitHub Actions, so local packs keep their normal source paths.
- To rehearse without publishing, run the workflow manually (`workflow_dispatch`) with a `version` input and `dry_run` checked — it packs, verifies, and uploads the `.nupkg` files as a build artifact without pushing.

### Platform binary release status

The [`Release Binaries`](.github/workflows/release-binaries.yml) workflow is intended to build self-contained archives of **TensorSharp.Server** and **TensorSharp.Cli** with the .NET 10 runtime and native libraries. However, the current latest release, [v3.0.5.0](https://github.com/zhongkaifu/TensorSharp/releases/tag/v3.0.5.0), has **no uploaded application archives** (only GitHub's automatic source downloads), so users must currently build from source. Do not construct an archive URL from the names below without first confirming that the file is listed on the [Releases page](https://github.com/zhongkaifu/TensorSharp/releases).

When a release workflow completes successfully, its intended archive matrix is:

| Archive suffix | Native backend(s) bundled | Format |
|---|---|---|
| `win-x64-cpu` | GGML CPU | `.zip` |
| `win-x64-cuda` | GGML CUDA + pure-C# CUDA (PTX) + CUDA 12.x runtime | `.zip` |
| `linux-x64-cpu` | GGML CPU | `.tar.gz` |
| `linux-x64-cuda` | GGML CUDA + pure-C# CUDA (PTX) + CUDA 12.x runtime | `.tar.gz` |
| `osx-arm64` | GGML Metal + MLX | `.tar.gz` |

- Pushing a `v*` tag triggers the archive and NuGet workflows; publication is conditional on every required job succeeding.
- The `-cuda` archives bundle the CUDA runtime libraries (`cudart` / `cublas` / `cublasLt`) but still require an NVIDIA GPU and a compatible driver at run time; the `-cpu` archives run anywhere. The macOS archive requires Apple Silicon.
- To rehearse, run the workflow manually (`workflow_dispatch`) with a `version` input — it builds every platform and creates a **draft** Release. Override the target GPU architectures for the CUDA build with the `cuda_arch` input.


## Architecture

TensorSharp is structured as a layered system:

1. **TensorSharp.Core** provides the core `Tensor` type, storage abstraction, and the extensible operation registry (`Ops`). CPU implementations use `System.Numerics.Vectors` for SIMD acceleration.

2. **TensorSharp.Runtime** owns runtime-facing contracts and services: GGUF parsing, tokenization (SentencePiece / BPE), chat template rendering, configurable token sampling, output parsing, paged KV cache (`Runtime/Paged/*`), the continuous-batching scheduler / engine (`Runtime/Scheduling/*`), the `IKvBlockCodec` interface plus the `TurboQuantKvCodec` 2-bit / Q4 / Q8 implementation, and reusable contracts such as `IModelArchitecture`, `IBatchedPagedModel`, `IPromptRenderer`, `IOutputProtocolParser`, `IMultimodalInjector`, `IKVCachePolicy`, and `IBackendExecutionPlan`.

3. **TensorSharp.Models** implements `ModelBase` plus the concrete architectures and multimodal helpers (Gemma 3/4, DiffusionGemma, Qwen 3/3.5, GPT OSS, Nemotron-H, Mistral 3, Qwen-Image-Edit). Autoregressive architectures ship the legacy per-sequence forward, and most also expose an `IBatchedPagedModel.ForwardBatch` implementation (`<Family>Model.BatchedForward.cs`) for continuous batching. DiffusionGemma is intentionally different: `Forward()` is unsupported, and generation goes through `DiffusionGemmaSampler` over fixed-length denoising canvases. Qwen-Image-Edit (`QwenImageModel`) is likewise not autoregressive — `Forward()` throws and image editing runs through `EditImage()`, which orchestrates the MMDiT diffusion transformer, the Qwen-Image VAE, and the Qwen2.5-VL text encoder. Models are loaded via `ModelBase.Create()` which auto-detects the architecture from GGUF metadata.

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
- **Native paged-attention kernel**: `TSGgml_PagedAttentionForward` (and the `WithSinks` variant for GPT OSS) does a C++ gather of K/V from the paged buffer, builds a small GGML graph per sequence, and dispatches `ggml_flash_attn_ext` — the same fused GPU flash-attention kernel (Metal/CUDA/Vulkan) the legacy single-sequence path uses. On Ministral-3-14B long-context (4×~800 tokens) it is **~21 % faster than the legacy per-sequence GGML path**.
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
- **KV block codecs**: blocks can be optionally compressed in-place with `TurboQuantKvCodec` (2-bit affine, Q4, or Q8) via `--paged-kv-quant-bits`, trading accuracy for a smaller per-block bandwidth and memory footprint — roughly half (Q8), a quarter (Q4), or a tenth (2-bit, fp32 blocks). The 2-bit tier uses an affine per-group min+scale (the block-min idea behind llama.cpp's Q2_K) so its four codes span the group's actual range; it is intended for long-context far-prefix reuse where attention weights dwarf the quantization noise. Recurrent-state models fall back to passthrough automatically.


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
