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
- **macOS (Metal backend):** CMake 3.20+ and the Xcode command-line tools for building the native GGML library — it embeds its Metal kernels as source and compiles them at run time, so it needs no Metal compiler at build time. The MLX backend additionally builds `libmlxc` from `TensorSharp.Backends.MLX/Native/` via `bash TensorSharp.Backends.MLX/build-native-macos.sh`, and that build *does* compile Metal shaders, so it needs a **full Xcode plus the Metal toolchain** — the command-line tools alone are not enough. `eng/ensure-metal-toolchain.sh` provisions this automatically on first build; see [Build the native MLX library](#build-the-native-mlx-library-macos-only)
- **Windows (GGML CPU / CUDA backends):** CMake 3.20+ and Visual Studio 2022 or 2026 C++ build tools; for `ggml_cuda` or `cuda`, install an NVIDIA driver plus CUDA Toolkit 12.x or another compatible CUDA toolkit with cuBLAS. With several toolkits installed, `CUDACXX` (or `-DCMAKE_CUDA_COMPILER=`) selects which `nvcc` to build with; `build-windows.ps1` honours it ahead of `PATH`/`CUDA_PATH`, prints the choice in the `Configuring ...` line, and discards a build tree that was configured against a different one (CMake caches the CUDA compiler on the first configure and otherwise ignores `CUDACXX` from then on). Note CMake reads `CUDACXX` only for non-Visual-Studio generators; under a `Visual Studio NN` generator use `-T cuda=<version-or-path>` instead. With Visual Studio 2026, whose MSVC 14.5x toolset is newer than current CUDA toolkits officially accept as a host compiler, the build passes `-allow-unsupported-compiler` to `nvcc` automatically; include the "C++ CMake tools for Windows" component so the build can use the Ninja generator (the Visual Studio generator additionally needs a CUDA toolkit that ships MSBuild integration for your VS version). **cuDNN is provisioned automatically** by `eng/fetch-cudnn.ps1` into `ExternalProjects/cudnn/` on the first CUDA build (~1.8 GB download, ~1.1 GB on disk; only `include/` and `bin/` are kept) so the Wan / Qwen-Image VAE convolutions can use it instead of ggml's im2col+GEMM lowering. An existing install (`TS_CUDNN_DIR`, `CUDNN_DIR`, `CUDA_PATH`) is used ahead of the download, `TENSORSHARP_CUDNN=OFF` skips it, and nothing is linked against it — the runtime resolves it with `LoadLibrary`, so the build and the resulting binaries work with or without it
- **Linux (GGML CPU / CUDA backends):** CMake 3.20+; for `ggml_cuda` or `cuda`, install an NVIDIA driver plus CUDA Toolkit 12.x or another compatible CUDA toolkit with cuBLAS. **cuDNN is provisioned automatically**: `eng/fetch-cudnn.sh` downloads the pinned redistributable from NVIDIA's public redist channel into `ExternalProjects/cudnn/` on the first CUDA build (no account or licence click needed), and the Wan / Qwen-Image VAE then runs its convolutions through cuDNN instead of ggml's im2col+GEMM lowering. An already-installed cuDNN (`libcudnn9-dev-cuda-12`, `CUDA_HOME`, `TS_CUDNN_DIR`) is used ahead of the download. It is strictly optional: the fetch never fails the build, only the headers are needed to compile, and the library itself is resolved with `dlopen` at run time, so a binary built with cuDNN still runs on a machine without it. `TENSORSHARP_CUDNN=OFF` skips it entirely; the configure step prints which path applies
- **Windows (GGML Vulkan backend):** CMake 3.20+ and the Visual Studio 2022 or 2026 C++ build tools are required here too — the Vulkan toolchain is provisioned *and* the backend is compiled with CMake, so a machine without it fails early in `eng/fetch-vulkan-toolchain.ps1`. Visual Studio's "C++ CMake tools for Windows" component ships both `cmake.exe` and `ninja.exe`, and `build-windows.ps1` falls back to that copy when neither is on `PATH`; otherwise install CMake from [cmake.org/download](https://cmake.org/download/). The native build is **x64 only** — `build-windows.ps1` imports the `vcvars64` environment itself, including over an already-active *x86* one (a plain "Developer PowerShell for VS" defaults to x86, and ggml's Vulkan backend does not compile 32-bit). The backend is enabled automatically when the machine has a Vulkan runtime (`System32\vulkan-1.dll`, shipped by every recent GPU driver). With a [LunarG Vulkan SDK](https://vulkan.lunarg.com/) installed it is used directly; without one the build auto-provisions a portable toolchain (Vulkan-Headers, a vulkan-1 import library generated from the system loader, glslc, SPIRV-Headers) into `ExternalProjects/vulkan-toolchain/` via `eng/fetch-vulkan-toolchain.ps1`. Opt out with `build-windows.ps1 --no-vulkan` or `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF`. A GPU driver with Vulkan 1.3 support is required at runtime
- **Linux (GGML Vulkan backend):** enabled automatically when a Vulkan loader (`libvulkan.so.1`) is installed. Distro dev packages are used when present (`apt install libvulkan-dev glslc spirv-headers`); otherwise the build auto-provisions the missing pieces (Vulkan-Headers, glslc from the shaderc CI prebuilts, SPIRV-Headers) into `ExternalProjects/vulkan-toolchain/` via `eng/fetch-vulkan-toolchain.sh`. Opt out with `build-linux.sh --no-vulkan` or `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF`
- GGUF model files (e.g., from [Hugging Face](https://huggingface.co))

## Building

### Build the entire solution

```bash
dotnet build TensorSharp.slnx
```

The solution build defaults to the `Any CPU` platform (`Directory.Solution.props`), so it also works from Visual Studio developer prompts, which export `Platform=x64` into the environment and would otherwise steer the build to a nonexistent `Release|x64` solution configuration. An explicit `-p:Platform=...` still takes precedence.

### Build individual applications

```bash
# Console application
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj

# Web application
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj
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

On Windows and Linux, the native build script auto-detects the visible NVIDIA GPU compute capability and passes a narrow `CMAKE_CUDA_ARCHITECTURES` value to ggml-cuda (for example `86-real` on an RTX 3080), which cuts CUDA build time substantially. The native build also runs in parallel by default, with the job count bounded by RAM (`nvcc` peaks around 3 GB per translation unit) so it does not overwhelm typical developer machines.

On Windows, `build-windows.ps1` prefers the **Ninja** generator, falling back to the `Visual Studio NN` generator and finally to whatever CMake picks. This matters for build time: Ninja parallelises across every translation unit at once, while the Visual Studio generator only parallelises across CMake projects, so ggml-cuda's ~190 `nvcc` compilations run one at a time. The script finds `ninja.exe` on `PATH` or in the Visual Studio installation (the "C++ CMake tools for Windows" component ships one) and imports the MSVC `vcvars64` environment itself, so no "x64 Native Tools" prompt is needed. It also imports `vcvars64` **over an already-active x86 environment**: the plain "Developer PowerShell for VS" and "Developer Command Prompt" shortcuts default to the x86 toolset, and a 32-bit build fails deep inside ggml's Vulkan backend with errors that never mention 32-bit. An already-active *x64* environment is left alone, so a pinned toolset (`vcvarsall.bat x64 -vcvars_ver=...`) survives. `cmake.exe` is resolved the same way as `ninja.exe` — `PATH` first, then the VS "C++ CMake tools for Windows" copy — and a missing CMake is reported up front rather than as "the term 'cmake' is not recognized". The generator and the effective job count are printed in the `Configuring TensorSharp.GGML.Native (...)` line; if the script warns that it may fall back to the serial `NMake Makefiles` generator, install that VS component or put `ninja.exe` on `PATH`.

Visual Studio is located with `eng/vs-locate.ps1`, which tolerates an installation the VS installer has flagged incomplete (`vswhere -latest` silently reports *no* installation for those, which is what made CMake fall back to the serial generator). Set `TENSORSHARP_VS_INSTALL_DIR` to override the detected installation directory. To force a generator explicitly, set `CMAKE_GENERATOR` or pass `-G` through to the script.

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

The direct `cuda` backend is built as managed C# plus PTX kernels. During `dotnet build`, `TensorSharp.Backends.Cuda` compiles `native/kernels/*.cu` to PTX in its intermediate directory (`obj/cuda_ptx/ptx/`) when `nvcc` is available, and that locally compiled PTX is what lands in every output's `cuda_kernels/` folder — building never modifies the git-tracked files under `native/ptx/`. If `nvcc` is missing, the committed PTX baseline in `native/ptx/` is used instead; if that also fails to load, PTX-backed ops use CPU fallbacks. cuBLAS-backed GEMM still requires the CUDA runtime libraries to be discoverable at run time.

After editing a `.cu` kernel, refresh the committed PTX baseline explicitly and commit the diff — machines without `nvcc` run the committed PTX, so an unrefreshed kernel change ships silently-stale kernels to them:

```powershell
dotnet build TensorSharp.Backends.Cuda/TensorSharp.Backends.Cuda.csproj -p:TensorSharpUpdateCommittedPtx=true
```

#### Metal 4 tensor API on Apple silicon

On M5 and newer GPUs ggml can route matmuls through the Metal 4 tensor API, which measured **2.6× prefill throughput** on an M5 Pro (807 → 2107 tok/s, Gemma 4 E4B Q8_0, 2048-token prefill). Decode is unchanged, as expected — single-sequence decode is memory-bandwidth bound.

ggml enables it only if a probe kernel including `<metal_tensor>` compiles at run time, and it compiles that probe at the *default* Metal Shading Language version. Metal derives that default from the SDK recorded in the **main executable**, not from `libGgmlOps.dylib`. Microsoft ships the .NET apphost — and the `dotnet` muxer — prebuilt against the macOS 15.5 SDK, so a .NET process defaults to MSL 3.2 where `<metal_tensor>` declares nothing, and ggml disables the tensor API on perfectly capable hardware:

```
ggml_metal_device_init: - the tensor API is not supported in this environment - disabling
ggml_metal_device_init: has tensor            = false
```

`tsg_metal_msl_default.m` corrects the default from inside our own library, which is where the fix has to live: `ExternalProjects/ggml` is git-ignored and `eng/fetch-ggml.sh` hard-resets it to upstream on every build, so an edit there would be erased by the next build. It raises the process-wide default to MSL 4.0 — the value a natively linked host already gets — only when the GPU advertises the Metal 4 family and only when the inherited default is older. Code that sets `languageVersion` itself keeps its own choice, so the MLX backend is unaffected (MLX always sets it explicitly). When it acts it logs one line, and `has tensor` becomes `true`.

| Environment variable | Effect |
|---|---|
| `TENSORSHARP_METAL_MSL_DEFAULT=off` | Leave the host's stale default in place (restores `has tensor = false`) |
| `TENSORSHARP_METAL_MSL_DEFAULT=<major>.<minor>` | Force a specific default MSL version, for example `4.0` |

ggml's own switches still apply on top: `GGML_METAL_TENSOR_DISABLE=1` turns the tensor API off, and `GGML_METAL_TENSOR_ENABLE=1` bypasses ggml's allowlist that restricts it to M5/M6/A19/A20 devices.

##### Wan video and the tensor API

ggml's tensor-API `mul_mm` intermittently misreads operand columns inside the Wan VAE's conv GEMMs on M5 — first pass over a graph only, buffer-layout dependent (historically the "32×32 latent decodes to a black frame while 33×33 works" bug), immune to every runtime kill switch, while the same GEMMs are bit-correct in isolation and LLM/DiT-style graphs never corrupt. It is an upstream ggml-metal/driver defect, and `has_tensor` is fixed at device init, so the kernel choice cannot be scoped per op.

When the tensor API is on, the VAE stays **correct** by routing its convolutions through `ggml_conv_2d_direct` instead of im2col+`mul_mat` on tensor-API devices (`wan_vae_gemm_budget` in `ggml_ops_wan.cpp`), which decodes within F16 rounding of the CPU backend but slower — a **fixed** per-video cost, while the tensor API's DiT speedup **scales** with step count and model size. Measured at 480×480×9f / 6 steps on an M5 Pro:

| | DiT step (tensor on / off) | VAE enc+dec (direct / GEMM) | break-even |
|---|---|---|---|
| A14B I2V Q4_K_M | **17.1s** / 30.2s (1.77×) | 135s / 19s | ~9 steps → the 40-step recipe is **~33% faster** with the tensor API (~13.7 vs ~20.5 min) |
| TI2V-5B Q8_0 | **1.6s** / 2.9s (1.8×) | 179s / 13s | ~128 steps → never wins |

The default therefore follows the DiT class (`WanVideoArchitecture.ApplyNativeTunables`): tensor API **enabled** for A14B/14B-class models (`patch_embedding` output dim ≥ 5120), **disabled** for smaller ones. When upstream fixes the tensor-API `mul_mm`, enable it everywhere and drop the direct-conv carve-out.

| Environment variable | Effect |
|---|---|
| `TS_WAN_METAL_TENSOR_API=1` / `=0` | Force the tensor API on/off for the Wan process, overriding the model-class default |
| `TS_WAN_VAE_GEMM_MAX_MB=<n>` | Force the im2col+GEMM VAE path with an `n` MB im2col budget (0 forces direct conv) — overrides the automatic choice in both directions |

#### MiniMax-H3 and the FP16 flash-attention numerator

Unlike the note above, this one is not Apple-specific: it applies to every backend whose flash-attention kernel keeps the softmax numerator in FP16, which is all of the ones built here.

MiniMax-H3 attends bidirectionally over **one** packed sequence with no mask — text, conditioning frames, target audio and target video are all in it — so the key count *is* the clip: 2364 packed tokens at 22 frames, 8646 at 107. Every flash-attention kernel in the vendored ggml accumulates `sum_j exp(s_j - max) * V_j` in FP16 registers, and `ggml_flash_attn_ext_set_prec(GGML_PREC_F32)` is inert because nothing under `ggml-cuda/` or `ggml-metal/` reads `op_params` for `GGML_OP_FLASH_ATTN_EXT`. What the kernel does grant the accumulator is three bits of headroom (`FATTN_KQ_MAX_OFFSET` inflates the running maximum by log(8), capping every softmax weight at 1/8), so a row of N keys reaches N/8 × |V| and overflows to Inf once N × |V| > 8 × 65504. H3 walks into that ceiling and no other model here does, because `q_norm` and `k_norm` exist in the checkpoint and `v_norm` does not — the value stream is the one carrying unbounded magnitudes. Measured at 640×384: 73 frames rendered correctly, 107 frames came back with every pixel black and every audio sample clamped, from a single Inf on the **first** denoise step. Video and audio share one trunk, so the overflow takes the soundtrack down with the picture.

`h3_attend` (`ggml_ops_minimax_h3.cpp`) keeps the accumulator in range by pre-scaling V by the smallest power of two that brings the key count down to `kH3FlashKeyBudget`, and undoing that scale on the output. Attention is linear in V, so the correction is **exact** — and a power of two is exact through the F16 cast as well, being nothing but an exponent shift, which is why short sequences (every oracle fixture in the test suite among them) stay bit-identical. `h3_mm` does the same for the two quantized matmuls whose activations are unbounded (the attention output feeding `o_proj`, the SwiGLU hidden state feeding `down_proj`), where the ceiling is q8_1's per-block FP16 sum instead.

| Environment variable | Effect |
|---|---|
| `TS_H3_TRACE=1` | Print latent and velocity magnitudes for every denoising step — a diverging sample shows in the latent's absmax several steps before it reaches infinity |
| `TS_H3_NO_FLASH=1` | Force the explicit softmax path, to separate a flash-attention kernel problem from a modelling one |

The sampler no longer trusts the result either. A non-finite velocity fails the request naming the step it appeared at (`RequireFinite` in `MiniMaxH3Pipeline.cs`) rather than writing out a file of the right length, frame rate and soundtrack duration that is uniformly black — the failure mode is silent by construction, because the RGB clamp pins a NaN pixel at 0 and the WAV writer clamps a NaN sample to -1.

### Build the native MLX library (macOS only)

The MLX backend depends on `libmlxc` (the C bindings for [MLX](https://github.com/ml-explore/mlx)). The repository pins a known-good tag of `mlx-c` in `TensorSharp.Backends.MLX/Native/MLX_C_VERSION` and a helper script fetches and builds it:

```bash
bash TensorSharp.Backends.MLX/build-native-macos.sh
```

The script writes the resulting libraries (`libmlxc.dylib`, `libmlx.dylib`, and any backend deps) into `TensorSharp.Backends.MLX/Native/dist/`, and the build copies them to the output directory alongside `mlx.metallib`. That metallib holds MLX's precompiled Metal kernels and is large (~150 MB) but not optional: MLX locates it by `dladdr` on its own code, so it must sit **in the same directory as `libmlx.dylib`**. Its only fallback is a path baked in at compile time that points into the build tree, so a deployment that omits it loads fine and then throws `Failed to load the default metallib` on the first GPU operation. Keep it next to the dylibs in any hand-rolled packaging. At run time the backend probes the application directory first; you can also point it to a custom install with `TENSORSHARP_MLX_LIBRARY=<path-to-libmlxc.dylib>` or `TENSORSHARP_MLX_LIBRARY_DIR=<dir-with-libmlxc>`. If the library cannot be located the backend reports unavailable and `--backend mlx` is rejected at startup.

#### Metal toolchain (provisioned automatically)

MLX compiles Metal shaders during the build, so `xcrun metal` has to work. Two things commonly stop that, and `build-native-macos.sh` repairs both by calling `eng/ensure-metal-toolchain.sh` before configuring:

1. **The active developer directory is the command-line tools.** `/Library/Developer/CommandLineTools` ships no Metal compiler at all, so `xcode-select -p` pointing there gives `xcrun: error: unable to find utility "metal", not a developer tool or in PATH`. The script locates an installed `Xcode.app` and builds with `DEVELOPER_DIR` set to it. It does **not** run `sudo xcode-select -s` — the override applies to the build only. Run that command yourself if you want it machine-wide.
2. **Xcode 16 and later do not bundle the Metal compiler.** It is a separately downloadable ~700 MB component; without it `metal` exists but refuses to run (`cannot execute tool 'metal' due to missing Metal Toolchain`). The script downloads it with `xcodebuild -downloadComponent MetalToolchain`, which needs no `sudo` and installs into the system asset store, so it is shared with every other project and survives Xcode updates.

Either problem surfaces from MLX as the much less obvious `error Metal compiler header resolution failed for .../reduce_utils.h`.

Xcode itself cannot be fetched unattended (the App Store and developer.apple.com both require a signed-in Apple ID), so if no `Xcode.app` is present the script stops with install instructions. Relevant knobs:

| Variable | Effect |
| --- | --- |
| `TENSORSHARP_XCODE_DEVELOPER_DIR` | Use this `<Xcode.app>/Contents/Developer` instead of autodetecting (useful for side-by-side or beta Xcode installs) |
| `TENSORSHARP_MLX_SKIP_METAL_SETUP` | `1`/`true` — skip the toolchain check entirely, for machines provisioned out of band |
| `TENSORSHARP_MLX_NATIVE_SKIP` | `true` — skip the MLX native build altogether, to build the rest of TensorSharp without a Metal toolchain |

Switching developer directories invalidates the CMake cache (the previous SDK is frozen into `CMakeCache.txt`), so the script discards the stale build tree and reconfigures. The fetched `_deps/*-src` checkouts are preserved, so this costs a reconfigure rather than a re-clone of MLX.


## Project Structure

```
TensorSharp/
├── TensorSharp.Core/            # Core tensor library (Tensor, Ops, memory, device abstraction, CPU SIMD/managed quantized kernels)
├── TensorSharp.Runtime/         # GGUF, tokenizers, templates, sampling, protocol parsing
│   ├── Paged/                   # Paged KV cache primitives (BlockPool, BlockTable, KvBlock, BlockHashIndex, PagedKvStorage, PagedKvBatchOps, ManagedPagedAttention)
│   ├── Scheduling/              # Continuous batching engine (InferenceEngine, BatchExecutor, ContinuousBatchScheduler, SequenceState, SchedulerConfig/Output, InferenceRequestHandle)
│   ├── Speculative/             # Speculative decoding: the draft/verify/rollback core (SpeculativeExecution), the ISpeculator algorithms (DraftHeadSpeculator, BlockDraftSpeculator, NGramSpeculator) + SpeculatorRegistry, the model-side contracts (ISpecTrunk, SpeculativeModelContracts), shared flag parsing (SpeculativeCliFlags, SpeculationOptions) and the runtime cost governor
│   ├── PagedKvCacheManager.cs   # Per-session paged KV manager (block allocation, prefix reuse)
│   ├── PagedKvBlockStore.cs     # On-disk / RAM-tiered paged block storage with optional SSD spillover
│   ├── SsdKvBlockTier.cs        # SSD-backed cold tier for paged blocks
│   ├── TurboQuantKvCodec.cs     # Quantized KV block codec (2-bit / Q4 / Q8) implementing IKvBlockCodec
│   ├── PrefillChunking.cs       # Chunked-prefill helper used by SWA / very long prompts
│   ├── KvBlockHash.cs           # Content-addressed block hash for prefix-cache sharing
│   └── Logging/                 # JSON-line file logger + per-turn telemetry
├── TensorSharp.AgentHost/       # Agentic layer on top of the runtime: Agent Skills and code execution
│   ├── Skills/                  # Agent Skills: SKILL.md frontmatter parsing (YamlFrontmatter, SkillManifest), discovery / install / lookup (SkillRegistry, SkillArchive), the containment boundary (SkillPathGuard), prompt planning (SkillPrompt), the built-in skills_list / skills_read / skills_run tools and the in-process disclosure loop (SkillTools, SkillAgentLoop, SkillScriptRunner), the OS sandboxes (SkillSandbox, SkillSandboxWindows) and their violation monitor, per-session workspaces (SessionWorkspace), shared flag parsing (SkillHostOptions) and the public client (SkillsChatClient)
│   └── CodeExec/                # The file surface (read_file, edit_file, write_file), the shell and apply_patch: the engine (ShellRunner), the two declarations and the lenient argument reading a small model needs (ShellTools), reading the command argument — splitting a line into its simple commands, classifying which of them are package installs (that is what decides whether the line gets a socket at all) and intercepting an apply_patch heredoc before the shell ever sees it (ShellCommand), the per-session working directory and exported environment, persisted through files because there is no long-lived shell process (ShellSession), shell discovery and dialect (ShellProgram), the patch envelope parser (CodePatch) and its matcher, a line-for-line port of the reference V4A applier (V4ADiff), turning a failure into the next command to type (CodeDiagnostics), reading the real API out of an installed package when a run failed because the model guessed one (ApiProbe), checking that a file a command wrote or a patch changed still parses (SyntaxCheck), rewriting the host's absolute paths out of everything a command printed (OutputPaths), Claude Code's string-replacement editor and its tolerance ladder (FileEdit), the one numbered-listing renderer every tool and every refusal shows a file through (NumberedListing), the six editing rules injected into the system prompt (CodePrompt), noticing a whole file re-typed to change two lines of it (RewriteWatch), host-initiated installs for a skill script whose import failed (PackageInstaller), the host's terms (CodeExecOptions), the confined launch, which can also start a process and not wait for it — that is how background jobs work (ConfinedProcess), interpreter discovery (CodeEnvironment), produced-file capture (CodeArtifactStore), the install-time registry proxy (EgressProxy), the result record (CodeExecResult) and the ICodeRunner seam the skills layer sees (CodeRunnerAdapter)
├── TensorSharp.Models/          # Model architectures and multimodal encoders/injectors
│   ├── Models/<Family>/         # One folder per architecture (DeepSeek4, DiffusionGemma, Gemma4, GlmDsa, GptOss, MiniMaxH3, Mistral3, MuseGlimmer, Nemotron, Qwen35, Qwen4Exp, QwenImage, WanVideo)
│   │   ├── <Family>Model.cs                # Legacy per-sequence ModelBase implementation
│   │   └── <Family>Model.BatchedForward.cs # IBatchedPagedModel.ForwardBatch — batched/paged path (Mistral3, Gemma4, GptOss, Qwen35, Nemotron)
│   ├── Models/DeepSeek4/        # DeepSeek V4 Flash: whole-model executors instead of a per-op forward
│   │   ├── DeepSeek4Model.cs               # GGUF metadata, tokenizer, chat template, executor selection
│   │   ├── DeepSeek4CudaExecutor.cs        # Bridge to the direct-CUDA whole-model engine
│   │   ├── DeepSeek4CpuExecutor*.cs        # 100% pure-C# whole-model executor (no native dependencies)
│   │   ├── DeepSeek4Model.Dspark.cs        # DSpark block drafter (draft / confidence / Markov heads)
│   │   └── DeepSeek4Model.PerSeqCache.cs   # Native per-sequence slots that make the model servable
│   ├── Models/GlmDsa/           # GLM 5.x: native-executor driver, MLA + DSA indexer per-op reference, sequence slots, NextN/MTP draft head
│   ├── Models/MuseGlimmer/      # Muse-Glimmer: fused whole-model forward, vision encoder, tensor-parallel variants, DFlash block drafter
│   ├── Models/MiniMaxH3/        # MiniMax-H3 video + joint 32 kHz stereo audio: packed-sequence DiT, Qwen3-VL text encoder + vision tower, video and audio VAEs, flow-match scheduler, pipeline
│   ├── Models/WanVideo/         # Wan 2.1/2.2, video only: DiT, UMT5-XXL text encoder, causal 3D VAE, UniPC scheduler, plus the ggml-independent WanDirect* `cuda`/`cpu` path
│   ├── Models/Video/            # The seam both video families implement: IVideoGenerationModel, VideoGenerationParams/Progress, GeneratedVideoAudio, WAV writing
│   ├── Paged/                   # Tensor-side paged-attention helpers (TensorPagedAttention)
│   ├── KvBlockTransfer.cs       # Helpers for extract/inject of KV blocks across sequences
│   ├── SpeculativeDecoder.cs    # Model-side draft-verify-rollback driver shared by Qwen 3.6, GLM 5.2 and Gemma 4
│   ├── SpeculativeDraftHeadLoader.cs # Loads a separate drafter GGUF (Gemma 4 gemma4-assistant, DSpark, DFlash) and binds it to the trunk
│   └── ModelMultimodalInjector.cs # Vision / audio / video embedding injection
├── TensorSharp.Backends.GGML/   # GGML backend bindings (Metal/CUDA/Vulkan/CPU via native library)
├── TensorSharp.Backends.Cuda/   # Direct CUDA backend using CUDA Driver API, cuBLAS, and PTX kernels
│   └── Dsv4/                    # DeepSeek V4 direct-CUDA whole-model engine (ggml-independent): streaming GGUF→VRAM loader, per-device weight arenas, layer split, DSpark drafter
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
│   ├── ggml_ops_qwen35_gdn_tp.cpp         # Qwen 3.5/3.6 per-rank packed GatedDeltaNet kernel (tensor parallel)
│   ├── ggml_ops_qwen35_recurrent_prefill.cpp # Qwen 3.5/3.6 recurrent-layer prefill
│   ├── ggml_ops_gptoss_decode.cpp         # GPT OSS whole-model decode graph (one dispatch per token, shared KV window)
│   ├── ggml_ops_gptoss_prefill.cpp        # GPT OSS whole-model prefill: N tokens through every attention + MoE layer plus the folded final norm and LM head as one graph
│   ├── ggml_ops_deepseek4.cpp             # DeepSeek V4 native whole-model executor (layer split, compressed KV caches, graph cache)
│   ├── ggml_ops_glm_dsa.cpp               # GLM 5.x native whole-model executor (MLA + DSA indexer, tensor parallelism, sequence slots, NextN/MTP verify + draft graphs)
│   ├── ggml_ops_muse_glimmer.cpp          # Muse-Glimmer whole-model forward: persistent decode graph (so ggml-cuda captures it) + transient prefill graph + tensor-parallel graphs
│   ├── ggml_ops_muse_glimmer_vision.cpp   # Muse-Glimmer ViT block on-device (a max-size image is 16,224 patches, so per-op dispatch synchronizes through the host)
│   ├── ggml_ops_dflash.cpp                # Muse-Glimmer DFlash block drafter fused into one graph per speculative step (draft blocks + the trunk's borrowed LM head)
│   ├── ggml_ops_dsv4_fused.cu / _cpu.cpp  # DeepSeek V4 fused custom ops on ggml-cuda's stream (and their CPU counterparts)
│   ├── ggml_ops_gemma4_decode.cpp         # Gemma 4 dense whole-model decode (CUDA-graph persisted)
│   ├── ggml_ops_gemma4_batched.cpp        # Gemma 4 dense + MoE token-batched decode
│   ├── ggml_ops_gemma4_verify.cpp         # Gemma 4 dense verify + MTP draft step
│   ├── ggml_ops_gemma4_moe.cpp            # Gemma 4 MoE layer/whole-model decode + verify
│   ├── ggml_ops_moe.cpp                   # Mixture-of-Experts forward / fused router
│   ├── ggml_ops_gated_delta_net.cpp       # Qwen 3.5/3.6 GatedDeltaNet kernels (per-seq + batched)
│   ├── ggml_ops_mamba2.cpp                # Nemotron Mamba2 kernels (per-seq + batched SIMD)
│   ├── ggml_ops_paged_attention.cpp       # Paged-attention native kernel (drives ggml_flash_attn_ext + sinks variant)
│   ├── ggml_ops_tensor_parallel.cpp       # Multi-rank TP group, segmented fused graph execution, collectives
│   ├── ggml_ops_tp_probe.cu               # Pre-flight peer-copy / NCCL AllReduce probe that picks the TP transport
│   ├── ggml_ops_diffusion.cpp             # DiffusionGemma fused decode-layer / whole-model / lm-head kernels
│   ├── ggml_ops_qwen_image.cpp            # Qwen-Image-Edit MMDiT whole-model forward (CUDA-graph-captured) + CFG-batched kernels
│   ├── ggml_ops_minimax_h3.cpp            # MiniMax-H3 whole-network graphs: the packed audio+video DiT, the Qwen3-VL text encoder and vision tower, and the video / audio VAE encode + decode (seven entry points, weights bound resident from the caller's mmap)
│   ├── ggml_ops_wan.cpp                   # Wan 2.1/2.2 whole-graph entry points: UMT5-XXL text encoder, per-step DiT velocity prediction (persistent per shape for CUDA-graph capture), causal 3D video VAE encode + decode
│   ├── ggml_ops_training.cpp              # Training-only kernels (unused at runtime)
│   └── tests/                              # Native unit + smoke tests
├── TensorSharp.Chat/            # Host-neutral chat pipeline shared by the Server, the CLI and the iOS app (no ASP.NET Core, no Distributed)
│   ├── ModelService.cs          # Facade over model load/unload, the InferenceEngineHost and the generation pipeline (TensorParallelGroupFactory, SchedulerConfigOverride, UnloadModel)
│   ├── ModelLifecycleService.cs # Model load/dispose and backend selection; the tensor-parallel group is handed in by the host
│   ├── InferenceEngineHost.cs   # Per-model InferenceEngine singleton (continuous batching entry point)
│   ├── ChatGenerationPipeline.cs # Prompt rendering, submits to InferenceEngine, streams tokens, stop handling
│   ├── ChatHistoryPreparer.cs   # History normalization, raw-token splice helpers, multimodal order helpers
│   ├── ChatSession.cs / SessionManager.cs # Per-conversation tracked history and the thread-safe session registry
│   ├── WebUiChatService.cs      # The Web UI request/stream contract (sessions, models, upload, image edit, video, chat frames) with the transport taken out
│   ├── SkillsService.cs         # The /api/skills and /v1/skills management surface, transport-free
│   ├── BackendCatalog.cs        # Backend vocabulary and canonical names; the availability probes stay in the Server
│   ├── Skills/                  # SkillRequestPlan, SkillChatLoop, RequestWorkspaceLease: the in-process disclosure loop over the pipeline
│   ├── Hosting/                 # ServerHostingOptions, sampling defaults, upload storage/content policy, hosted-model guard, startup loader
│   ├── RequestParsers/          # JSON request parsing (chat messages, sampling, tool functions, skills, video params, upload references)
│   ├── ResponseSerializers/     # WebUiSseEvents (the Web UI frame shapes) + shared JSON options
│   └── ProtocolAdapters/        # ChatStreamCollector + FinishReasonMapper (pipeline vocabulary -> protocol vocabulary)
├── TensorSharp.Server/          # Web chatbot + API server (ASP.NET Core)
│   ├── Program.cs               # Slim bootstrap: DI wiring, middleware, endpoint mapping, paged-KV + continuous-batching CLI translation
│   ├── BackendCatalogProbes.cs  # CUDA / MLX / GGML availability probes behind TensorSharp.Chat's BackendCatalog
│   ├── OpenAIResponseFormatParser.cs  # OpenAI response_format (json_object / json_schema) parsing
│   ├── Hosting/                 # Startup-time concerns: options builder (ServerOptionsBuilder), startup banner, logging, web root, /uploads static files, the multi-node tensor-parallel factory, paged-KV / continuous-batching CLI translation
│   ├── ResponseSerializers/     # Per-protocol response shape factories (Ollama, OpenAI)
│   ├── StreamingWriters/        # SSE + NDJSON wire-format helpers
│   ├── ProtocolAdapters/        # Per-protocol request handlers (OllamaAdapter, OpenAIChatAdapter, OpenAIResponsesAdapter; WebUiAdapter and SkillsAdapter are thin HTTP shells over TensorSharp.Chat)
│   ├── Endpoints/               # ASP.NET Core endpoint mapping (one extension method per protocol)
│   ├── Logging/                 # Request logging middleware + low-noise path support
│   ├── wwwroot/index.html       # Chat UI
│   ├── testdata/                # Integration test suites (bash + Python)
│   └── API_EXAMPLES.md          # Detailed API documentation
├── TensorSharp.Cli/             # CLI application (one-shot generation, interactive REPL, batch JSONL, benchmarks)
├── TensorSharp.TestMatrix/      # Test / benchmark matrix runner, default prompts, env-var sweeps, and per-host baselines
├── TensorAgent/                 # iPhone / iPad app: the Server's Web UI chat, running entirely on the device
│   ├── src/TensorAgent.Core/    # Platform-neutral: the model catalog and store, resumable downloads, saved conversations, settings, the loopback server and its route table, and AgentAppHost, which assembles all of it (built here rather than in the iOS head so it can be started, driven over HTTP and torn down by a test)
│   │   ├── Shell/               # An in-process POSIX shell -- pipelines, redirections, heredocs, globs, functions, and the coreutils a coding model reaches for, awk included -- plus the IShellBackend that lets the agent host run code where Process.Start is unsupported
│   │   ├── Sandbox/             # ExecutionPolicy and ConfinedPaths: the one set of rules the shell, Python and JavaScript all enforce, since there is no OS sandbox to lean on
│   │   ├── Python/              # CPython 3.13 embedded by P/Invoke, its audit-hook sandbox, and a pure-wheel installer
│   │   ├── JavaScript/          # JavaScriptCore over its C API, with Node-shaped console/process/require/fs/timers
│   │   └── WebUi/               # The one script appended to the Server's index.html, so the page itself is never forked
│   ├── src/TensorAgent.Maui/    # The net10.0-ios head: WebView + attachments + dictation, the models / chats / settings pages, and where the files live on this device
│   ├── skills/                  # The skills that were verified to work here, with verdicts.json recording why each one is in or out
│   └── scripts/                 # build / run / verify for the simulator, prepare-python.sh, build-lxml-ios.sh, verify-skills.py
├── InferenceWeb.Tests/          # xUnit unit tests covering ops, KV cache, paged scheduler, batched-model correctness, web/server helpers
├── AdvUtils/                    # Utility library (logger)
├── docs/                        # Developer reference
│   ├── models/                  # Per-model architecture cards (one .md per model, EN + 中文)
│   ├── PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md  # Paged KV cache, prefix sharing, scheduler, per-model batched-forward status
│   ├── speculative_decoding.md  # Draft-and-verify design: the ISpeculativeTarget / ISpeculator / IDraftHead layering and the draft-head, block and ngram algorithms
│   ├── agent_skills.md          # Agent Skills: the SKILL.md format, progressive disclosure and its budget, the in-process tool loop, the path / ZIP / script-execution security model, and the HTTP + C# surfaces
│   └── env_var_feature_matrix.md  # Runtime flag × model/backend/feature coverage for TestMatrix
├── benchmarks/                  # Reproducible benchmark harnesses
└── ExternalProjects/            # ggml/ is cloned from github.com/ggml-org/ggml at build time (not committed)
```

## Project / NuGet Package Boundaries

The repository is split along package boundaries so consumers can depend on only the layers they actually need.

**Status verified 2026-09-08.** The publish set is **thirteen** packages — every row of the table below. `eng/verify-packages.ps1` is the authoritative list, and it gates the publish workflow, so adding a `ProjectReference` between two packable projects without updating that script fails the release.

What is on [NuGet.org](https://www.nuget.org/profiles/TensorSharp) today is a subset: **eight** ids at version **3.1.2**, published 2026-07-21 — `TensorSharp.Tensors`, `TensorSharp.Runtime`, `TensorSharp.Models`, `TensorSharp.Backends.GGML`, `TensorSharp.Backends.Cuda`, `TensorSharp.Backends.MLX`, `TensorSharp.Server`, and `TensorSharp.Cli`. Those packages lag the current source and the v3.3.0.0 application release; in particular the published `TensorSharp.Server` predates the logging and chat splits, so it does not match the layering described here.

The remaining five — `TensorSharp.Runtime.Logging`, `TensorSharp.AgentHost`, `TensorSharp.Chat`, `TensorSharp.Server.Host`, and `TensorSharp.Distributed` — are packable and verified but have never been pushed; they ship with the next version tag. Until then, consumers of those layers need project references from a source checkout.

| Project | NuGet package | Public namespace | Responsibility |
|---|---|---|---|
| `TensorSharp.Core` | `TensorSharp.Tensors` | `TensorSharp` | Tensor primitives, ops, allocators, storage, and device abstraction |
| `TensorSharp.Runtime.Logging` | `TensorSharp.Runtime.Logging` | `TensorSharp.Runtime.Logging` | Logging abstractions and sinks shared by every host, factored out so the engine layers do not carry a host's logging stack |
| `TensorSharp.Runtime` | `TensorSharp.Runtime` | `TensorSharp.Runtime` | GGUF parsing, tokenizers, prompt rendering, sampling, output protocol parsing, paged KV cache, continuous-batching scheduler |
| `TensorSharp.AgentHost` | `TensorSharp.AgentHost` | `TensorSharp.AgentHost` | Agent Skills and code execution (`read_file` + `edit_file` + `write_file` + `shell` + `apply_patch`) with OS sandboxing, per-Web/CLI-session and per-HTTP-request workspaces, and host-classified package installs — built on `TensorSharp.Runtime` |
| `TensorSharp.Models` | `TensorSharp.Models` | `TensorSharp.Models` | `ModelBase`, architecture implementations, multimodal encoders, batched / paged forward passes, and model-side execution helpers |
| `TensorSharp.Backends.GGML` | `TensorSharp.Backends.GGML` | `TensorSharp.GGML` | GGML-backed execution and native interop |
| `TensorSharp.Backends.Cuda` | `TensorSharp.Backends.Cuda` | `TensorSharp.Cuda` | Direct CUDA allocator, storage, cuBLAS GEMM, PTX kernels, and quantized CUDA ops |
| `TensorSharp.Backends.MLX` | `TensorSharp.Backends.MLX` | `TensorSharp.MLX` | Apple Silicon MLX backend (mlx-c / Metal) with quantized / fused / compiled kernels and MoE expert offload |
| `TensorSharp.Distributed` | `TensorSharp.Distributed` | `TensorSharp.Distributed` | Peer-to-peer TCP coordination for multi-node tensor parallelism |
| `TensorSharp.Chat` | `TensorSharp.Chat` | `TensorSharp.Chat` (new types); the moved pipeline keeps its `TensorSharp.Server.*` namespaces | Host-neutral chat pipeline: `ModelService`, sessions, generation, the skills loop and the Web UI request/stream contract (`WebUiChatService`, `SkillsService`) — no ASP.NET Core, no `TensorSharp.Distributed`; shared by the Server, the CLI and the iOS app |
| `TensorSharp.Server` | `TensorSharp.Server` | `TensorSharp.Server` | ASP.NET Core server, OpenAI/Ollama adapters, HTTP transport over TensorSharp.Chat, web UI |
| `TensorSharp.Server.Host` | `TensorSharp.Server.Host` | `TensorSharp.Server.Host` | The **runnable** web application: `Program.cs`, host wiring, `wwwroot/`, and the command line. `TensorSharp.Server` is the library it builds on — building or running `TensorSharp.Server` alone produces no executable |
| `TensorSharp.Cli` | `TensorSharp.Cli` | `TensorSharp.Cli` | Console host and debugging / batch tooling |

This split keeps engine users off the web stack, keeps API-layer changes from leaking into core/runtime packages, and makes future benchmark or eval-harness projects easier to publish independently.

`TensorSharp.Chat` is the seam between the pipeline and its hosts. Everything that used to sit inside `TensorSharp.Server` without touching ASP.NET Core — `ModelService`, `ModelLifecycleService`, `InferenceEngineHost`, `ChatGenerationPipeline`, sessions, the skills loop, the request parsers and the Web UI frame builders — moved there verbatim (same namespaces), and the Web UI handlers became `WebUiChatService` / `SkillsService`: JSON payload objects in and out, refusals thrown as `WebUiRequestRejectedException(StatusCode, Payload)`, streams as `IAsyncEnumerable<object>` of frames. The Server's `WebUiAdapter` and `SkillsAdapter` are now a few lines per route (read the `HttpContext`, call the service, write status + JSON or `data:` events), so an in-process loopback server or a native view model drives the identical contract. Two things the library deliberately does not reference: `TensorSharp.Distributed` (it references the CUDA backend, which an iOS build cannot link — the Server hands multi-node tensor parallelism in through `ModelService.TensorParallelGroupFactory`) and the CUDA/MLX backend probes (`BackendCatalogProbes` stays in the Server). Hosts without a shell size the KV pool through `ModelService.SchedulerConfigOverride` and free a model without disposing the service through `ModelService.UnloadModel()`. `ChatLayeringTests` fails if any of that direction inverts.

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

**Status verified 2026-09-01:** [v3.3.0.0](https://github.com/zhongkaifu/TensorSharp/releases/tag/v3.3.0.0) publishes ten self-contained application archives of **TensorSharp.Server** and **TensorSharp.Cli**, including the .NET 10 runtime and native libraries. There is one Server archive and one CLI archive for each of the five platform/backend suffixes below. Check the [Releases page](https://github.com/zhongkaifu/TensorSharp/releases) for newer versions instead of assuming that this stamped version remains latest.

The published v3.3.0.0 archive matrix is:

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

2. **TensorSharp.Runtime** owns runtime-facing contracts and services: GGUF parsing, tokenization (SentencePiece / BPE), chat template rendering, configurable token sampling, output parsing, paged KV cache (`Runtime/Paged/*`), the continuous-batching scheduler / engine (`Runtime/Scheduling/*`), the `IKvBlockCodec` interface plus the `TurboQuantKvCodec` 2-bit / Q4 / Q8 implementation, and reusable contracts such as `IModelArchitecture`, `IBatchedPagedModel`, `IPromptRenderer`, `IOutputProtocolParser`, `IMultimodalInjector`, `IKVCachePolicy`, and `IBackendExecutionPlan`. It deliberately carries none of the agentic layer: skills and code execution live in **TensorSharp.AgentHost**, which references the runtime and is never referenced back, so a host that serves plain OpenAI/Ollama chat completions takes the runtime alone and ships no skill registry, no sandbox and no code runner. (`AgentHostLayeringTests` fails if that direction ever inverts.)

3. **TensorSharp.Models** implements `ModelBase` plus thirteen concrete architectures and their multimodal helpers — ten text families (DeepSeek V4 Flash, GLM 5.x, Gemma 4, DiffusionGemma, Qwen 3.5/3.6-family, Qwen 3.8 Flash Next, GPT OSS, Nemotron-H, Mistral 3, Muse-Glimmer) and three media-out families (Qwen-Image-Edit, MiniMax-H3, Wan 2.1/2.2). Autoregressive architectures ship the legacy per-sequence forward, and most also expose an `IBatchedPagedModel.ForwardBatch` implementation (`<Family>Model.BatchedForward.cs`) for continuous batching. DiffusionGemma is intentionally different: `Forward()` is unsupported, and generation goes through `DiffusionGemmaSampler` over fixed-length denoising canvases. Qwen-Image-Edit (`QwenImageModel`) is likewise not autoregressive — `Forward()` throws and image editing runs through `EditImage()`, which orchestrates the MMDiT diffusion transformer, the Qwen-Image VAE, and the Qwen2.5-VL text encoder. The video families go one step further out: `MiniMaxH3Model` and `WanVideoModel` both throw from `ForwardCore()` and generate through `GenerateVideo(prompt, VideoGenerationParams)` behind the shared `IVideoGenerationModel` seam in `Models/Video/`, so the CLI and the server drive either one — and anything added later — through a single path instead of type-testing the concrete model. MiniMax-H3 denoises video and 32 kHz stereo audio *together* in one packed latent and drives seven native whole-network graphs (DiT, Qwen3-VL text encoder, vision tower, video and audio VAE encode + decode); Wan 2.1/2.2 is the video-only family and runs its DiT, UMT5-XXL encoder and causal 3D VAE the same way. Models are loaded via `ModelBase.Create()` which auto-detects the architecture from GGUF metadata — except MiniMax-H3, whose published GGUFs carry no metadata at all and which is therefore detected from its tensors (`LooksLikeMiniMaxH3`, wired through `MiniMaxH3Architecture.DetectFromTensors`).

4. **TensorSharp.Backends.GGML** registers accelerated implementations of the same operations via a native C++ bridge (`libGgmlOps` / `GgmlOps.dll`) that links against [ggml](https://github.com/ggml-org/ggml). On macOS this provides Metal GPU compute, and on Windows/Linux it can expose GGML CUDA for NVIDIA GPUs. Operations include native quantized matmul (Q4_K_M, Q8_0, etc.) without dequantizing to FP32, plus paged-attention (`TSGgml_PagedAttentionForward`, with and without attention sinks) and architecture-specific batched kernels (Mamba2, GatedDeltaNet).

5. **TensorSharp.Backends.Cuda** is the direct CUDA path. It uses the CUDA Driver API for device/context/storage management, cuBLAS for float32 GEMM, PTX kernels for hot scalar and transformer helper ops, and CPU fallbacks where native kernels are not implemented yet.

6. **TensorSharp.Backends.MLX** is the Apple Silicon MLX path. It wraps [mlx-c](https://github.com/ml-explore/mlx-c) (`libmlxc`) with allocator, storage, async worker dispatch, quantized + fused + compiled kernels, MoE expert offload, and a CPU fallback layer for ops that aren't yet wired up.

7. **TensorSharp.Server** is the HTTP/application layer: Ollama-compatible and OpenAI-compatible REST APIs, the browser-based chat UI, multipart upload handling, middleware and SSE writers. The pipeline it serves lives in **TensorSharp.Chat** — the `ModelService`, the `InferenceEngineHost` that owns the per-model continuous-batching engine for autoregressive models, the `DiffusionBatchScheduler` for DiffusionGemma Web UI turns, sessions, the skills loop and the Web UI request/stream contract (`WebUiChatService`, `SkillsService`) — so the CLI and the iOS app drive the same code without ASP.NET Core; a thin queue-status surface is kept for backward compatibility.

8. **TensorSharp.Cli** is the console/application layer for local prompts, multimodal experiments, prompt inspection, JSONL batch workflows, the interactive REPL, and the built-in prefill / decode benchmarks.

### Adding a model, a modality, or a chat format

Everything an architecture needs to declare about itself lives in three tables,
so adding a family touches its own directory plus one line per table -- and
nothing else in the loader, planner, CLI or server.

**1. The architecture plug-in.** Write `Models/<Family>/<Family>Architecture.cs`
next to the model:

```csharp
internal static class MyFamilyArchitecture
{
    public static ModelArchitectureDescriptor Descriptor { get; } = new()
    {
        Id = "myfamily",
        DisplayName = "My Family",
        Aliases = new[] { "myfamily", "myfamily_moe" },   // general.architecture values
        Factory = c => new MyFamilyModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup),
        // Optional, all defaulted:
        //   MultiGpu / MultiGpuLimitation   how it uses more than one GPU, and why not
        //   ProjectorFileHints              mmproj companion names to auto-discover
        //   DetectFromTensors               for GGUFs that declare no architecture
        //   ApplyNativeTunables             process-wide ggml switches to set before load
    };
}
```

Then add one line to `Architecture/BuiltInArchitectures.cs`. `ModelBase.Create`
resolves through `ModelArchitectureRegistry`; there is no switch to extend, and
`ModelArchitectureDescriptor.Validate()` refuses a descriptor that declares a
degraded multi-GPU mode without saying why.

**2. Modalities are capability interfaces, not type tests.** A model that can see
implements `IVisionCapableModel` (load the tower, receive an embedding span) and
`IMultimodalPromptExpander` (expand its own placeholders). Audio adds
`IAudioCapableModel` / `IAudioEncoderLoader`; per-axis rotary positions add
`IMRoPEPositionSink`. `ModelMultimodalInjector` owns everything generic --
per-request buckets, span bookkeeping, prefix clamping, trimming, slicing -- and
never names a model type. The CLI, the interactive REPL and the server all drive
that one injector, so a modality wired once works everywhere.

**3. The chat format is a `ChatProtocol`.** Prompt framing, whether to bypass the
GGUF's Jinja template, media placeholder tokens, the output parser, whether that
parser is mandatory, where a structured-output grammar may arm, the KV-cache
generation suffix, and video-frame capping are ONE entry in
`ChatProtocolRegistry`. These used to be roughly two dozen separate
architecture-name comparisons across `ChatTemplate`, `OutputParser`,
`KVCachePromptRenderer` and the server; forgetting one of them failed quietly
(an unparsed reply streams its own reasoning tags to the client; a missing media
placeholder discards the image; a missing generation suffix drops multi-turn
prefix reuse to zero while still answering correctly).

Runtime routing is unchanged and still capability-driven:
`ExecutionCapabilities.FromModel` reads `IBatchedPagedModel`,
`ISpeculativeTarget` and friends once per step. None of the three tables above is
consulted on a per-token path.

### Performance Optimizations

The list below is the cross-architecture summary; each per-model card under
[`docs/models/`](docs/models/README.md) walks through the same kernels in
context, with the exact GGML graph dispatched and the conditions under which
the fused path engages.

- **Fused GPU decode** (Gemma 4): all transformer layers are executed in a single GGML compute graph dispatch on Metal, reducing CPU-GPU round-trips from hundreds per token to one. This achieves ~2.6x speedup over per-operation dispatch.
- **Fused GPU prefill** (Gemma 4): for dense (non-MoE, non-shared, non-PLE/multimodal) layers, `Gemma4LayerPrefill` runs the entire transformer block (RMSNorm + QKV + QK-norm + RoPE + attention + output projection + post-attn norm + GeGLU FFN + post-FFN norm + residual + layer scalar) as a single GGML graph dispatch per layer during prefill, extending the fused approach from decode to multi-token prefill.
- **Chunked prefill** (Gemma 4): long prompts are split into bounded chunks (2x sliding window, max 2048 tokens) to avoid O(n^2) attention score tensors for SWA layers. Chunking is applied automatically when text-only (no multimodal embeddings) and keeps each chunk within the SWA window budget.
- **Fused Qwen 3.5/3.6-family attention layer decode**: a single GGML graph performs RMSNorm + fused QKV + Q/gate deinterleave + per-head QK norm + RoPE + KV cache append + flash attention + sigmoid-gated mix + output projection + residual add for each FullAttention layer. Replaces ~2 standalone GGML calls and ~6 small CPU/GPU sync points per attention layer. Engages once the cached sequence length exceeds 4096 tokens (override with `FUSED_ATTN_LAYER_MIN_SEQ_LEN=N`).
- **Fused prefill attention** (Qwen 3.5/3.6-family): `FusedPrefillAttention` combines Q*K^T, causal mask, softmax, and *V into a single GGML graph dispatch during multi-token prefill, eliminating ~5 separate C#-to-GGML round-trips per attention layer. Handles both initial prefill and continuation with existing KV cache entries.
- **Whole-model Metal prefill and decode** (Qwen 3.5/3.6-family): supported dense single-device models execute all attention and GatedDeltaNet layers, final RMSNorm, and the LM head in one GGML graph. Prefill uses the fused multi-token verify graph; decode retains a per-sequence graph, reads quantized token embeddings directly, moves Metal KV-copy views within a 64-token attention bucket, and overlaps graph submission with logits readback.
- **In-place Metal GatedDeltaNet state** (Qwen 3.5/3.6-family): single-token decode aliases each recurrent layer's fused GDN result with its state input, removing 48 state-copy dispatches and about 302 MB of state read/write traffic per token on the 64-layer Qwen 3.6-27B layout. Set `TS_QWEN35_METAL_GDN_INPLACE_STATE=0` to retain the separate-copy path for diagnostics.
- **Fused output-projection + FFN** (Qwen 3.5/3.6-family): for both FullAttention and GatedDeltaNet layers with dense FFN, `FusedOutProjFFN` merges the output projection, residual add, post-attention RMSNorm, and the full SwiGLU FFN (gate_up matmul + SiLU + down matmul + residual) into a single GGML graph dispatch, reducing two GPU round-trips to one per layer.
- **Fused output-projection + norm + router** (Qwen 3.5/3.6-family MoE): `FusedOutProjNormRouter` merges the GatedDeltaNet output projection, residual add, post-attention RMSNorm, and MoE router projection into one dispatch. The pre-computed router logits are then consumed directly by the batched MoE kernel, eliminating a separate router dispatch per MoE layer.
- **Fused vision encoder** (Qwen 3.5/3.6-family): `FusedVisionAttention` merges LayerNorm + QKV + bias + 2D RoPE + scaled dot-product attention + output projection + bias + residual into one GGML graph dispatch (~8 ops → 1). `FusedVisionMLP` merges LayerNorm + up + bias + GELU + down + bias + residual into one dispatch (7 ops → 1). Combined, these cut the per-block GPU round-trips from ~15 to 2.
- **Fused weight projections**: same-type Q/K/V projections are fused into a single QKV matmul; mixed-type importance-matrix/UD quant projections remain separate to avoid multi-gigabyte FP32 expansion. Gate and up projections are fused into a single gate_up matmul.
- **Native quantized compute**: quantized weights (Q4_K_M, Q6_K, Q8_0, IQ2_XXS, MXFP4, NVFP4, etc.) are used directly in matmul without expanding to FP32, saving memory and bandwidth. A batched `AddmmQuantBatch` kernel handles multiple sub-weight matmuls against a single quantized blob in one dispatch.
- **Direct CUDA kernels**: the `cuda` backend accelerates fill/copy, unary ops, activation fusions, RMSNorm, softmax, index select, causal masking, RoPE/RoPEEx, cuBLAS GEMM, and supported quantized matmul/get-rows while safely falling back for incomplete op coverage.
- **Batched GPU MoE**: `MoEExpertsSwiGLUResidual` (Qwen 3.5/3.6-family) and `MoEExpertsForward` (Nemotron-H) collapse all selected experts -- and, for Qwen 3.5/3.6-family, the optional shared expert and the residual add -- into a single GGML graph dispatch per MoE layer.
- **Whole-model fused decode graph** (Gemma 4 dense + MoE, Qwen 3.5/3.6, GPT OSS): an entire decode token — every layer, the MoE router and experts, the final norm, and the LM head — is submitted as ONE GGML graph rather than one dispatch per layer. On CUDA/Vulkan the graph is built once with stable tensor addresses and replayed (`ggml_set_rows` KV write with the row as an I64 input; a stride-padded attention window with an F16 mask input), which is what lets ggml-cuda capture it as a CUDA graph. GPT OSS decode goes from 24 → 154 tok/s on an A40 and stays flat in context length (133 tok/s at 16K) where the per-layer path collapsed to 2.3. Padded attention windows must be zeroed, not left uninitialized — stale VRAM read as F16 produces NaNs that survive a `-inf` mask. Per-model opt-outs: `TS_GPTOSS_MODEL_DECODE=0`, `TS_GEMMA4_FD_PERSIST=0`, `TS_QWEN35_FD_PERSIST=0`.
- **GLM 5.x whole-model executor**: `glm-dsa` follows the same shape. The native ggml executor (`ggml_ops_glm_dsa.cpp`) loads the split GGUF itself (six shards for GLM-5.2, seven for GLM-5.3's UD-Q2_K_XL) and owns the MLA cache (one 576-wide row per token per layer, the per-head K/V decompression folded into the query and the output) plus the DSA lightning-indexer cache, which only 21 of the 78 layers refresh. It either layer-splits across the visible GPUs or, under `--tp N`, runs every layer on every rank: attention heads column/row-parallel, and the routed experts split row-wise *inside* each expert, because `ggml_mul_mat_id` needs a token's selected expert ids to stay distinct. Concurrency is native sequence slots rather than paged KV, with a default-on batched fused decode over them (`TS_BATCHED_FUSED_DECODE=0` disables it). `TensorSharp.Models/Models/GlmDsa/` holds the managed per-op reference the native path is checked against. The trailing NextN block (`blk.78`) drives [MTP speculative decoding](docs/models/glm.md#nextn--mtp-speculative-decoding) under `--spec`: the trunk graph gains an `h_nextn` output, a second graph runs the draft block, and the block is only loaded when speculation was requested because it is a whole extra decoder layer competing with the KV cache for VRAM.
- **DeepSeek V4 whole-model executors**: `deepseek4` bypasses the generic per-op forward entirely. The native ggml executor (`ggml_ops_deepseek4.cpp`) loads the split GGUF itself, layer-splits the weights across every visible GPU, owns all DSV4 KV state on-device (raw SWA ring, CSA/HCA compressed-K caches, lightning-indexer cache, compressor state rings), and runs each prefill/decode ubatch as a single `ggml_backend_sched` graph with a shape-signature graph cache, so steady-state decode replays a captured CUDA graph. Decode attention gathers a compact `[ring | top-512]` K through a fused index-gather op instead of scanning the full context. The direct-CUDA engine (`TensorSharp.Backends.Cuda/Dsv4/`) implements the same model without ggml, streaming quantized weights from the shards straight into per-device arenas. Both are built on the shared `Tensor` / `IAllocator` / `Ops` stack; only genuinely DSV4-specific compute lives in the DeepSeek V4 files.
- **DSpark block speculative decoding** (DeepSeek V4): a separate drafter GGUF (`--draft-model`) proposes a whole block of tokens per step and the trunk verifies the block in one batched forward. On ggml the drafter is three extra graph layers whose key ring the trunk graph commits itself, so speculation costs no host round-trips. Measured 1.3–1.4× decode on 4×A40 (up to 2.0× on multi-turn chat), greedy output byte-identical to the non-speculative baseline.
- **GEMM-based vision patch embedding** (Qwen 3.5/3.6-family): the patch embedding step is reformulated as parallel im2col + matrix multiplication, replacing a single-threaded scalar quintuple-nested loop with a GPU-accelerated matmul.
- **Parallelized Q/gate deinterleave** (Qwen 3.5/3.6-family): the Q + sigmoid-gate deinterleave in FullAttention prefill is parallelized across tokens, scaling linearly with CPU core count for long prompts.
- **Optimized pure C# CPU path**: managed GEMM fast paths and contiguous float32 kernels accelerate decode, softmax, RMSNorm, RoPE, fused activations, and other hot paths while keeping quantized GGUF weights compressed during CPU loading.
- **Circular KV cache**: sliding-window attention layers use a fixed-size circular buffer, bounding memory usage regardless of sequence length.
- **KV-cache prefix reuse**: multi-turn conversations reuse the longest matching token prefix across turns. Truncation is automatically backed off by the sliding-window size for SWA models so the suffix can rebuild the SWA context.
- **Paged KV cache & block-hash prefix sharing**: the continuous-batching engine partitions KV into fixed-size blocks, content-hashes each full block, and shares them across concurrent and sequential requests. Models that have not implemented `IBatchedPagedModel` still use the engine's isolated per-sequence KV-swap fallback.
- **Native paged-attention kernel**: `TSGgml_PagedAttentionForward` (and the `WithSinks` variant for GPT OSS) does a C++ gather of K/V from the paged buffer, builds a small GGML graph per sequence, and dispatches `ggml_flash_attn_ext` — the same fused GPU flash-attention kernel (Metal/CUDA/Vulkan) the legacy single-sequence path uses. On Ministral-3-14B long-context (4×~800 tokens) it is **~21 % faster than the legacy per-sequence GGML path**.
- **Batched / paged forward passes**: Mistral 3, Gemma 4, GPT OSS, Qwen 3.5/3.6 (incl. GatedDeltaNet recurrent state pool), and Nemotron-H (incl. Mamba2 recurrent state pool + native batched Mamba2 kernel) pack N sequences into a single `ForwardBatch` call with one batched linear-projection matmul per layer, paged K/V scatter via `slotMapping`, and per-sequence attention via the native kernel. Gemma 4 batched path reaches **1.5×** legacy throughput at batch=8 short prompts and **1.6×** at 4×800-token prompts; Nemotron-H Mamba2 batched reaches **3.95×** at batch=3 on Apple M4 Pro. See [docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md).
- **MTP / NextN speculative decoding**: solo sequences can run a multi-token-prediction draft head (Qwen 3.6, GLM 5.2 and GLM-5.3 embedded NextN block - on GLM-5.3 only without `--tp`, since its draft block borrows the column-parallel trunk LM head; Gemma 4 separate `gemma4-assistant` draft GGUF). The draft proposes up to `--spec-draft` tokens and the trunk verifies them in one batched forward, with the request's own sampler driving both, so decode is accelerated without changing the output. On ggml backends, fused single-graph multi-token-verify and draft-step kernels (`NativeGemma4ModelVerify` / `TryFusedMoEModelVerify` / `NativeGemma4DraftStep`, plus the Qwen 3.6 NextN graph) amortize the verify; the Gemma 4 path also adds gallocr verify scratch and a dense fast-rollback that avoids re-running the kept prefix on partial acceptance. The pure-C# `cuda` backend runs a fully GPU-resident per-op verify/draft (donor-cache attention, GQA decode kernel, GPU RoPE) so the verify layer loop issues zero host-sync stalls. Off by default; `--spec` (or `--draft-model` for the Gemma 4 assistant GGUF, which enables speculation by itself).
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

`InferenceWeb.Tests` exercises in-process behavior that doesn't require a running server: managed quantized ops, direct CUDA backend kernels when a CUDA device is available, MLX backend kernels when MLX is available, paged KV cache scheduling (`ContinuousBatchSchedulerTests`, `PagedKvCacheTests`, `PagedKvCacheCodecTests`), batched executor correctness (`BatchedExecutorTests`), per-model batched-forward correctness against the legacy path (`Qwen35BatchedCorrectnessTests`, `Mistral3BatchedForwardTests`, `Gemma4BatchedForwardTests`, `GptOssBatchedCorrectnessTests`, `NemotronBatchedCorrectnessTests`), MTP / NextN speculative-decoding correctness and opt-in end-to-end probes (`SpeculativeExecutionTests`, `Qwen36SpeculativeTests`, `Gemma4SpeculativeTests`), DiffusionGemma denoising / prompt-KV / batched-generation probes (`DiffusionGemmaTests`), per-model batched perf microbenchmarks (`*BatchedPerfBench.cs`), `TurboQuantKvCodec` codec round-trips, prefill chunking, KV cache policies, KV-cache prompt rendering / multi-turn integration, chat-session and session-manager isolation, model service history plumbing, request-logging middleware and file-logger provider, image preprocessing, media helpers, structured-output validation, text-upload helpers, model-service upload logging, web UI chat policy, model context length parsing, backend catalog resolution, the server CLI options builder (`ServerOptionsBuilderTests`), and Agent Skills — `SKILL.md` frontmatter parsing with its warning cases (`SkillManifestParserTests`) and registry discovery, precedence, ZIP install guards and path containment (`SkillRegistryTests`).

```bash
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj
```

#### Test lanes

Tests are tagged on two axes (declared in `InferenceWeb.Tests/TestAssemblyConfig.cs`): `Category=Bench` marks the timing/throughput benchmark classes with a plain `[Trait]`, and `Requires=Cuda|Mlx|Models` marks what a test needs from the environment (`Models` = real GGUF weights via the test model directory). Environment-gated tests are written with the gated attributes from `InferenceWeb.Tests/GatedFacts.cs` — `[CudaFact]`/`[CudaTheory]`, `[MlxFact]`/`[MlxTheory]`, `[ModelFact("ENV_VAR", "gguf-substring")]`/`[ModelTheory(...)]` — which apply the matching `Requires` trait automatically and skip visibly when the prerequisite is missing. Calling `CudaBackend.IsAvailable()`/`MlxBackend.IsAvailable()` directly from a test is a build error (`BannedSymbols.txt`): use the attribute instead. Untagged `[Fact]`/`[Theory]` tests are self-contained correctness tests that run anywhere. An unfiltered `dotnet test` runs everything; `--filter` selects a lane:

```bash
# Inner loop while editing: environment-independent correctness tests, runs anywhere in seconds.
# This is also the lane PR CI runs (.github/workflows/pr-unit-tests.yml).
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj --filter "Category!=Bench&Requires!=Cuda&Requires!=Mlx&Requires!=Models"

# Full correctness (pre-push on a box with a GPU and model files): everything except benchmarks.
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj --filter "Category!=Bench"

# Benchmarks only, run deliberately on quiet hardware — their assertions are timing-sensitive.
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj --filter "Category=Bench"
```

Gated tests report as **Skipped** when their prerequisite is missing (a skipped `[Theory]` counts once, not per data row), so a green run on a bare box reads "N passed, M skipped" rather than silently passing tests that never executed. A few classes with compound prerequisites (multiple env vars, per-method model choice) still gate inside the test body and keep explicit `[Trait("Requires", ...)]` lines.

### Server integration tests

Integration tests for TensorSharp.Server are in `TensorSharp.Server/testdata/`. They cover all three API styles (Web UI SSE, Ollama, OpenAI), multi-turn conversations, thinking mode, tool calling, structured outputs, queue-status compatibility, concurrent requests, and abort support. Architecture-specific features (thinking, tool calling) are auto-detected and skipped when the active model does not support them.

```bash
# Start TensorSharp.Server, then run:
python3 TensorSharp.Server/testdata/test_multiturn.py
# or
bash TensorSharp.Server/testdata/test_multiturn.sh
```

See [TensorSharp.Server.Host/testdata/README.md](TensorSharp.Server.Host/testdata/README.md) for the full test matrix.

### Inference matrix runner

`TensorSharp.TestMatrix` is the broader CLI-driven harness for long-running model/backend coverage. It discovers GGUF files, filters unavailable backends and unsupported prompt types, runs baseline plus env-var sweep cells, writes one JSON result per cell, emits an aggregate Markdown report, and compares results with per-host baselines when requested.

```bash
dotnet build TensorSharp.TestMatrix/TensorSharp.TestMatrix.csproj -c Release
dotnet run --project TensorSharp.TestMatrix -c Release -- --dry-run
```

See [TensorSharp.TestMatrix/README.md](TensorSharp.TestMatrix/README.md) and [docs/env_var_feature_matrix.md](docs/env_var_feature_matrix.md) for the current runner contract.
