# 开发
[English](DEVELOPMENT.md) | [中文](DEVELOPMENT_zh-cn.md)

> [TensorSharp](README_zh-cn.md) 文档的一部分：如何构建 TensorSharp、仓库结构、包分层、内部架构与测试工具。

## 前置要求

### 安装 .NET 10 SDK

TensorSharp 的所有项目都面向 `net10.0`。从源码构建必须安装完整的 **.NET 10 SDK**；仅安装 Runtime 不够。安装 SDK 时也会安装 CLI 与服务器运行所需的 .NET 和 ASP.NET Core Runtime。

| 平台 | 推荐安装方式 |
|---|---|
| **Windows** | 打开 PowerShell 并运行 `winget install Microsoft.DotNet.SDK.10`。也可以按 Microsoft 的 [Windows 安装指南](https://learn.microsoft.com/zh-cn/dotnet/core/install/windows)，选择适合当前架构的 .NET 10 **SDK** 安装程序。 |
| **macOS** | 下载并运行 [.NET 10 SDK 安装程序](https://dotnet.microsoft.com/zh-cn/download/dotnet/10.0)。Apple 芯片选择 **Arm64**，Intel Mac 选择 **x64**；另见 Microsoft 的 [macOS 安装指南](https://learn.microsoft.com/zh-cn/dotnet/core/install/macos)。 |
| **Linux** | 使用 Microsoft 的 [Linux 安装指南](https://learn.microsoft.com/zh-cn/dotnet/core/install/linux)选择发行版和版本、配置相应软件源，再安装 .NET 10 SDK 包（通常名为 `dotnet-sdk-10.0`）。各发行版的软件源与架构支持不同，请以链接中的发行版专用步骤为准。 |

安装后打开新终端，确认列表中包含 `10.0.x` SDK：

```bash
dotnet --list-sdks
```

安装程序、包管理器、手动安装和非管理员安装方式见 Microsoft 的 [.NET 跨平台安装概览](https://learn.microsoft.com/zh-cn/dotnet/core/install/)。

### 其他构建前置

- **`git` 与网络访问：** GGML/CUDA 原生构建会在首次构建时从 [github.com/ggml-org/ggml](https://github.com/ggml-org/ggml) 克隆 ggml 源码到 `ExternalProjects/ggml/`（参见 `eng/fetch-ggml.sh` / `eng/fetch-ggml.ps1`）。克隆默认跟踪 ggml 的默认分支（`master`）；可用 `TENSORSHARP_GGML_GIT_REF` 指定其他引用，或在克隆完成后设置 `TENSORSHARP_GGML_NO_UPDATE=1` 跳过网络更新（用于离线重建）
- **macOS（Metal 后端）：** 用于构建原生 GGML 库的 CMake 3.20+ 与 Xcode 命令行工具；若需使用 MLX 后端，还需通过 `bash TensorSharp.Backends.MLX/build-native-macos.sh` 从 `TensorSharp.Backends.MLX/Native/` 构建 `libmlxc`
- **Windows（GGML CPU / CUDA 后端）：** CMake 3.20+ 与 Visual Studio 2022 C++ 构建工具；若使用 `ggml_cuda` 或 `cuda`，还需要 NVIDIA 驱动和带 cuBLAS 的 CUDA Toolkit 12.x 或其他兼容版本
- **Linux（GGML CPU / CUDA 后端）：** CMake 3.20+；若使用 `ggml_cuda` 或 `cuda`，还需要 NVIDIA 驱动和带 cuBLAS 的 CUDA Toolkit 12.x 或其他兼容版本
- **Windows（GGML Vulkan 后端）：** 机器有 Vulkan 运行时（每个较新的 GPU 驱动都带的 `System32\vulkan-1.dll`）时自动启用。已安装 [LunarG Vulkan SDK](https://vulkan.lunarg.com/) 时直接使用；未安装时构建会通过 `eng/fetch-vulkan-toolchain.ps1` 自动把便携工具链（Vulkan-Headers、由系统 loader 生成的 vulkan-1 导入库、glslc、SPIRV-Headers）准备到 `ExternalProjects/vulkan-toolchain/`。用 `build-windows.ps1 --no-vulkan` 或 `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF` 退出。运行时需要支持 Vulkan 1.3 的 GPU 驱动
- **Linux（GGML Vulkan 后端）：** 已安装 Vulkan loader（`libvulkan.so.1`）时自动启用。存在发行版开发包时直接使用（`apt install libvulkan-dev glslc spirv-headers`）；否则构建会通过 `eng/fetch-vulkan-toolchain.sh` 把缺失的部分（Vulkan-Headers、shaderc CI 预编译的 glslc、SPIRV-Headers）自动下载到 `ExternalProjects/vulkan-toolchain/`。用 `build-linux.sh --no-vulkan` 或 `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF` 退出
- GGUF 模型文件（例如来自 [Hugging Face](https://huggingface.co)）

## 构建

### 构建整个解决方案

```bash
dotnet build TensorSharp.slnx
```

### 构建单独应用

```bash
# 控制台应用
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj

# Web 应用
dotnet build TensorSharp.Server/TensorSharp.Server.csproj
```

### 构建原生 GGML 库

如果原生库不存在，首次执行 `dotnet build` 时会自动构建。也可以手动构建：

```bash
cd TensorSharp.GGML.Native
```

macOS：

```bash
bash build-macos.sh
```

Linux（仅 CPU）：

```bash
bash build-linux.sh
```

Linux（启用 GGML_CUDA）：

```bash
bash build-linux.sh --cuda
```

Windows（仅 CPU）：

```powershell
.\build-windows.ps1 --no-cuda
```

Windows（启用 GGML_CUDA）：

```powershell
.\build-windows.ps1 --cuda
```

在 Windows 和 Linux 上，原生构建脚本会自动检测可见 NVIDIA GPU 的 compute capability，并把一个精简的 `CMAKE_CUDA_ARCHITECTURES` 列表传给 ggml-cuda（例如在 RTX 3080 上为 `86-real`），从而显著降低 CUDA 构建时间。原生构建默认还会以受控的并行任务数运行，避免 `nvcc` 拖慢普通开发机器。

如需覆盖自动检测到的架构列表或默认的并行度，可使用以下任一方式：

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

也可以在 `dotnet build` 时通过环境变量请求 CUDA 版本的原生库：

```bash
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
```

```powershell
$env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA='ON'; dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
```

在 macOS 上会生成带 Metal GPU 支持的 `libGgmlOps.dylib`。在 Windows 和 Linux 上，原生脚本会保留已有的 CUDA 构建，并在检测到 CUDA 工具链时自动启用 GGML_CUDA；也可以通过 `build-windows.ps1 --cuda`、`build-linux.sh --cuda` 或 `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` 显式启用。GGML Vulkan 后端在机器有 Vulkan 运行时时同样自动启用，并在首次使用时下载其构建工具链；`--vulkan` / `--no-vulkan` 或 `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON/OFF` 可显式指定，显式选择会在后续重建中保持（构建自动准备的 Vulkan 工具链见[前置要求](#前置要求)）。构建产物会自动复制到应用输出目录。

Direct `cuda` 后端由托管 C# 代码和 PTX 内核组成。执行 `dotnet build` 时，`TensorSharp.Backends.Cuda` 会在检测到 `nvcc` 后把 `native/kernels/*.cu` 编译到 `native/ptx/*.ptx`；如果缺少 `nvcc`，构建会继续，PTX 覆盖的算子会使用 CPU 回退。cuBLAS GEMM 仍要求运行时能够找到 CUDA 运行库。

### 构建原生 MLX 库（仅 macOS）

MLX 后端依赖 `libmlxc`（[MLX](https://github.com/ml-explore/mlx) 的 C 绑定）。仓库在 `TensorSharp.Backends.MLX/Native/MLX_C_VERSION` 中固定了已知可用的 `mlx-c` tag，并提供一个辅助脚本来获取和构建：

```bash
bash TensorSharp.Backends.MLX/build-native-macos.sh
```

脚本会把生成的库（`libmlxc.dylib`、`libmlx.dylib` 以及任何后端依赖）写入 `TensorSharp.Backends.MLX/Native/dist/`。运行时后端会优先在应用目录下查找；也可以使用 `TENSORSHARP_MLX_LIBRARY=<libmlxc.dylib 路径>` 或 `TENSORSHARP_MLX_LIBRARY_DIR=<包含 libmlxc 的目录>` 指定自定义安装位置。如果找不到对应库，后端会报告不可用，启动时 `--backend mlx` 会被拒绝。


## 项目结构

```text
TensorSharp/
├── TensorSharp.Core/            # 核心张量库（Tensor、Ops、内存、设备抽象，含 CPU SIMD/托管量化内核）
├── TensorSharp.Runtime/         # GGUF、分词器、模板、采样、协议解析
│   ├── Paged/                   # 分页 KV 缓存原语（BlockPool、BlockTable、KvBlock、BlockHashIndex、PagedKvStorage、PagedKvBatchOps、ManagedPagedAttention）
│   ├── Scheduling/              # 连续批处理引擎（InferenceEngine、BatchExecutor、ContinuousBatchScheduler、SequenceState、SchedulerConfig/Output、InferenceRequestHandle）+ MTP 投机解码核心（MtpSpeculativeExecution）
│   ├── PagedKvCacheManager.cs   # 单会话分页 KV 管理（块分配、前缀复用）
│   ├── PagedKvBlockStore.cs     # 带可选 SSD 溢出的 RAM/磁盘分级分页块存储
│   ├── SsdKvBlockTier.cs        # 分页块的 SSD 冷层
│   ├── TurboQuantKvCodec.cs     # 实现 IKvBlockCodec 的量化 KV 块编解码器（2-bit / Q4 / Q8）
│   ├── PrefillChunking.cs       # SWA / 超长 prompt 使用的分块 prefill 辅助
│   ├── KvBlockHash.cs           # 内容寻址的块哈希，用于跨请求前缀复用
│   └── Logging/                 # JSON-line 文件日志器 + 每轮遥测
├── TensorSharp.Models/          # 模型架构实现与多模态编码/注入
│   ├── Models/<Family>/         # 每个架构一个目录（DiffusionGemma、Gemma3、Gemma4、GptOss、Mistral3、Nemotron、Qwen3、Qwen35、QwenImage）
│   │   ├── <Family>Model.cs                # 旧的单序列 ModelBase 实现
│   │   └── <Family>Model.BatchedForward.cs # IBatchedPagedModel.ForwardBatch —— 批处理/分页路径（Mistral3、Gemma4、GptOss、Qwen35、Nemotron、Qwen3）
│   ├── Paged/                   # 张量侧的分页注意力辅助（TensorPagedAttention）
│   ├── KvBlockTransfer.cs       # 跨序列的 KV 块 extract/inject 辅助
│   ├── MtpSpeculativeDecoder.cs # Qwen 3.6 与 Gemma 4 共用的 MTP/NextN 起草-验证-回滚驱动
│   └── ModelMultimodalInjector.cs # 视觉 / 音频 / 视频嵌入注入
├── TensorSharp.Backends.GGML/   # GGML 后端绑定（通过原生库支持 Metal/CUDA/Vulkan/CPU）
├── TensorSharp.Backends.Cuda/   # Direct CUDA 后端（CUDA Driver API、cuBLAS、PTX 内核）
├── TensorSharp.Backends.MLX/    # Apple Silicon MLX 后端（mlx-c / Metal），原生桥接由 `build-native-macos.sh` 编译
├── TensorSharp.GGML.Native/     # 到 ggml 的原生 C++ 桥接（构建 libGgmlOps，拆分为多个专注源文件）
│   ├── ggml_ops_core.cpp                  # 元素级、归约、基础 shape 操作
│   ├── ggml_ops_elementwise.cpp           # 元素级 / 激活融合
│   ├── ggml_ops_matmul.cpp                # GEMM / 量化 matmul
│   ├── ggml_ops_fused.cpp                 # 跨域融合的每层内核
│   ├── ggml_ops_norm_attn.cpp             # Norm + 注意力融合
│   ├── ggml_ops_transformer.cpp           # 通用融合 Transformer 层/整模型 decode 与 flash-attn decode
│   ├── ggml_ops_transformer_common.h      # 共享的 Transformer 辅助函数与 C# 层描述符结构体
│   ├── ggml_ops_transformer_prefill.cpp   # 融合层 prefill（Gemma 4、GPT-OSS、Qwen 3.5）
│   ├── ggml_ops_qwen35_decode.cpp         # Qwen 3.5/3.6 融合 decode（单层、整模型、批量）
│   ├── ggml_ops_qwen35_verify.cpp         # Qwen 3.5/3.6 融合多 token verify
│   ├── ggml_ops_gemma4_decode.cpp         # Gemma 4 稠密整模型 decode（CUDA graph 持久化）
│   ├── ggml_ops_gemma4_batched.cpp        # Gemma 4 稠密 + MoE 按 token 批量 decode
│   ├── ggml_ops_gemma4_verify.cpp         # Gemma 4 稠密 verify + MTP 草稿步
│   ├── ggml_ops_gemma4_moe.cpp            # Gemma 4 MoE 层/整模型 decode 与 verify
│   ├── ggml_ops_moe.cpp                   # 专家混合前向 / 融合路由
│   ├── ggml_ops_gated_delta_net.cpp       # Qwen 3.5/3.6 GatedDeltaNet 内核（按序列 + 批处理）
│   ├── ggml_ops_mamba2.cpp                # Nemotron Mamba2 内核（按序列 + 批处理 SIMD）
│   ├── ggml_ops_paged_attention.cpp       # 分页注意力原生内核（驱动 ggml_flash_attn_ext + sinks 变体）
│   ├── ggml_ops_diffusion.cpp             # DiffusionGemma 融合 decode-layer / 整模型 / lm-head 内核
│   ├── ggml_ops_qwen_image.cpp            # Qwen-Image-Edit MMDiT 整模型前向（CUDA 图捕获）+ CFG-batched 内核
│   ├── ggml_ops_training.cpp              # 仅训练用内核（运行时不使用）
│   └── tests/                              # 原生单元 + 烟雾测试
├── TensorSharp.Server/          # Web 聊天 + API 服务（ASP.NET Core）
│   ├── Program.cs               # 精简启动：DI 注册、中间件、端点映射、paged-KV + 连续批处理 CLI 翻译
│   ├── ModelService.cs          # 保持服务端推理公共 API 稳定的门面，持有 InferenceEngineHost
│   ├── ModelLifecycleService.cs # 模型加载/释放与后端选择（CPU / CUDA / MLX / GGML CPU/Metal/CUDA/Vulkan）
│   ├── InferenceEngineHost.cs   # DI 注册的单模型 InferenceEngine 单例（连续批处理入口）
│   ├── ChatGenerationPipeline.cs # Prompt 渲染，将请求提交到 InferenceEngine，流式返回 token，处理 stop
│   ├── InferenceTelemetry.cs    # Prompt/eval 计时、TTFT、tokens/sec、有界输入摘要与输出日志
│   ├── ChatHistoryPreparer.cs   # 历史归一化、raw token 拼接、多模态顺序辅助
│   ├── ChatSession.cs           # 单会话历史跟踪与 assistant raw token
│   ├── SessionManager.cs        # 线程安全的会话注册（默认会话 + 每个 UI Tab 的会话）
│   ├── InferenceQueue.cs        # 向后兼容的队列状态接口（并发由引擎本身处理）
│   ├── BackendCatalog.cs        # 可用计算后端的发现（CPU / CUDA / MLX / GGML*）
│   ├── TextUploadHelper.cs      # 无损文本上传归一化辅助
│   ├── WebUiChatPolicy.cs       # Web UI 聊天请求合法性校验
│   ├── OpenAIResponseFormatParser.cs  # OpenAI response_format（json_object / json_schema）解析
│   ├── Hosting/                 # 启动期相关：选项装配（ServerOptionsBuilder）、后端选择、日志、wwwroot 解析、paged-KV / 连续批处理 CLI 翻译
│   ├── RequestParsers/          # JSON 请求解析（采样配置、聊天消息、工具函数）
│   ├── ResponseSerializers/     # 各协议响应形状构造（Ollama / OpenAI / Web UI）
│   ├── StreamingWriters/        # SSE 与 NDJSON 线协议辅助
│   ├── ProtocolAdapters/        # 各协议的请求处理器（WebUiAdapter、OllamaAdapter、OpenAIChatAdapter）
│   ├── Endpoints/               # ASP.NET Core 路由映射（每协议一个扩展方法）
│   ├── Logging/                 # 请求日志中间件 + 低噪声路径支持
│   ├── wwwroot/index.html       # 聊天界面
│   ├── testdata/                # 集成测试套件（bash + Python）
│   └── API_EXAMPLES.md          # 详细 API 文档
├── TensorSharp.Cli/             # CLI 应用（单次生成、交互式 REPL、JSONL 批处理、基准）
├── TensorSharp.TestMatrix/      # 测试 / 基准矩阵运行器、默认提示、环境变量扫描与主机基线
├── InferenceWeb.Tests/          # xUnit 单元测试，覆盖算子、KV 缓存、分页调度器、批处理模型正确性以及 Web/服务辅助逻辑
├── AdvUtils/                    # 工具库（日志）
├── docs/                        # 开发者参考文档
│   ├── models/                  # 按模型架构卡片（每个模型一份 .md，中英双语）
│   ├── PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md  # 分页 KV 缓存、前缀共享、调度器、按模型批处理状态
│   └── env_var_feature_matrix.md  # TestMatrix 使用的运行时开关 × 模型/后端/功能覆盖矩阵
├── benchmarks/                  # 可重现的基准脚本
└── ExternalProjects/            # ggml/ 在构建时从 github.com/ggml-org/ggml 克隆（不纳入版本控制）
```

## 项目 / NuGet 包分层

仓库按包边界拆成独立层，使用者可以只引用真正需要的部分。这些是可构建的包项目与包 ID，但当前 Runtime/Models/Backends/CLI/Server 包**尚未发布到 NuGet.org**。目前请从源码 checkout 使用项目引用；在 [NuGet.org](https://www.nuget.org/profiles/TensorSharp) 出现匹配版本之前，不要照抄 `dotnet add package TensorSharp.Models` 一类命令。

| 项目 | NuGet 包 | 对外 namespace | 职责 |
|---|---|---|---|
| `TensorSharp.Core` | `TensorSharp.Tensors` | `TensorSharp` | Tensor 原语、Ops、分配器、存储与设备抽象 |
| `TensorSharp.Runtime` | `TensorSharp.Runtime` | `TensorSharp.Runtime` | GGUF 解析、分词器、Prompt 渲染、采样、输出协议解析、分页 KV 缓存、连续批处理调度器 |
| `TensorSharp.Models` | `TensorSharp.Models` | `TensorSharp.Models` | `ModelBase`、各模型架构、多模态编码器、批处理 / 分页前向、模型侧执行辅助 |
| `TensorSharp.Backends.GGML` | `TensorSharp.Backends.GGML` | `TensorSharp.GGML` | GGML 执行后端与原生互操作 |
| `TensorSharp.Backends.Cuda` | `TensorSharp.Backends.Cuda` | `TensorSharp.Cuda` | Direct CUDA 分配器、存储、cuBLAS GEMM、PTX 内核和量化 CUDA 算子 |
| `TensorSharp.Backends.MLX` | `TensorSharp.Backends.MLX` | `TensorSharp.MLX` | Apple Silicon MLX 后端（mlx-c / Metal），含量化 / 融合 / 编译内核与 MoE 专家 offload |
| `TensorSharp.Server` | `TensorSharp.Server` | `TensorSharp.Server` | ASP.NET Core 服务、OpenAI/Ollama 适配层、推理引擎宿主与 Web UI |
| `TensorSharp.Cli` | `TensorSharp.Cli` | `TensorSharp.Cli` | 控制台宿主、调试工具与 JSONL 批处理 |

这样的拆分让引擎使用者不必带上 Web 依赖，也能把 API 层改动和核心运行时隔离开，并让后续 benchmark / eval harness 更容易独立发布。

> **注意：** 核心层发布的包名是 **`TensorSharp.Tensors`**，不是 `TensorSharp.Core`。NuGet.org 上的 `TensorSharp.Core` 包 ID 属于另一个已废弃的无关项目（所有版本均已 unlist，源码仓库已删除），推送到该 ID 会返回 403。差异仅限于 NuGet 包名——项目名、程序集名与 `TensorSharp` namespace 均保持不变，因此 `using` 语句不受影响，只有 `dotnet add package` 命令行不同。

发布前可验证包元数据与 README 依赖边界：

```powershell
pwsh ./eng/verify-packages.ps1
```

该验证会对上表 7 个公开包运行 `dotnet pack`，并在 `AdvUtils` 等内部依赖泄漏到 `.nuspec`，或 TensorSharp 包依赖了上表之外的分层时失败。

### 平台二进制发行状态

[`Release Binaries`](.github/workflows/release-binaries.yml) 工作流的目标是为 **TensorSharp.Server** 与 **TensorSharp.Cli** 构建包含 .NET 10 运行时及原生库的自包含归档。但是，当前最新的 [v3.0.5.0](https://github.com/zhongkaifu/TensorSharp/releases/tag/v3.0.5.0) **没有上传应用归档**（只有 GitHub 自动生成的源码下载），因此用户目前必须从源码构建。除非先在 [Releases 页面](https://github.com/zhongkaifu/TensorSharp/releases)确认文件确实存在，否则不要根据下方名称自行拼接下载 URL。

发行工作流成功完成后，计划生成的归档矩阵如下：

| 归档后缀 | 内置的原生后端 | 格式 |
|---|---|---|
| `win-x64-cpu` | GGML CPU | `.zip` |
| `win-x64-cuda` | GGML CUDA + 纯 C# CUDA（PTX）+ CUDA 12.x 运行时 | `.zip` |
| `linux-x64-cpu` | GGML CPU | `.tar.gz` |
| `linux-x64-cuda` | GGML CUDA + 纯 C# CUDA（PTX）+ CUDA 12.x 运行时 | `.tar.gz` |
| `osx-arm64` | GGML Metal + MLX | `.tar.gz` |

- 推送 `v*` 标签会触发归档与 NuGet 工作流；只有所需 job 全部成功后才会发布产物。
- `-cuda` 归档已内置 CUDA 运行时库（`cudart` / `cublas` / `cublasLt`），但运行时仍需 NVIDIA GPU 与兼容驱动；`-cpu` 归档可在任意机器运行。macOS 归档需 Apple Silicon。
- 如需预演，可手动触发该工作流（`workflow_dispatch`）并填写 `version` 输入——它会构建全部平台并创建**草稿** Release。可用 `cuda_arch` 输入覆盖 CUDA 构建的目标 GPU 架构。


## 架构说明

TensorSharp 采用分层系统结构：

1. **TensorSharp.Core** 提供核心 `Tensor` 类型、存储抽象和可扩展的操作注册表（`Ops`）。CPU 实现使用 `System.Numerics.Vectors` 进行 SIMD 加速。

2. **TensorSharp.Runtime** 负责运行时契约与通用服务：GGUF 解析、分词（SentencePiece / BPE）、聊天模板渲染、可配置 token 采样、输出解析、分页 KV 缓存（`Runtime/Paged/*`）、连续批处理调度器 / 引擎（`Runtime/Scheduling/*`）、`IKvBlockCodec` 接口及其 `TurboQuantKvCodec` 2-bit / Q4 / Q8 实现，以及 `IModelArchitecture`、`IBatchedPagedModel`、`IPromptRenderer`、`IOutputProtocolParser`、`IMultimodalInjector`、`IKVCachePolicy`、`IBackendExecutionPlan` 等抽象。

3. **TensorSharp.Models** 实现 `ModelBase` 以及各具体模型架构和多模态辅助组件（Gemma 3/4、DiffusionGemma、Qwen 3/3.5、GPT OSS、Nemotron-H、Mistral 3、Qwen-Image-Edit）。自回归架构提供旧的单序列前向，多数架构还提供面向连续批处理的 `IBatchedPagedModel.ForwardBatch` 实现（`<Family>Model.BatchedForward.cs`）。DiffusionGemma 刻意不同：它不支持 `Forward()`，生成必须通过 `DiffusionGemmaSampler` 在固定长度 canvas 上迭代去噪。Qwen-Image-Edit（`QwenImageModel`）同样非自回归：`Forward()` 抛异常，图像编辑通过 `EditImage()` 进行，由它编排 MMDiT 扩散 Transformer、Qwen-Image VAE 与 Qwen2.5-VL 文本编码器。模型通过 `ModelBase.Create()` 加载，并依据 GGUF 元数据自动识别架构。

4. **TensorSharp.Backends.GGML** 通过原生 C++ 桥接库（`libGgmlOps` / `GgmlOps.dll`）注册同名操作的加速实现，并链接 [ggml](https://github.com/ggml-org/ggml)。在 macOS 上可提供 Metal GPU 计算，在 Windows/Linux 上可启用面向 NVIDIA GPU 的 GGML CUDA。除原生量化 matmul（Q4_K_M、Q8_0 等，无需反量化到 FP32）外，还提供分页注意力（`TSGgml_PagedAttentionForward`，含 / 不含注意力 sinks 两种版本）以及架构特定的批处理内核（Mamba2、GatedDeltaNet）。

5. **TensorSharp.Backends.Cuda** 是 Direct CUDA 路径。它使用 CUDA Driver API 管理设备、上下文与存储，用 cuBLAS 执行 Float32 GEMM，用 PTX 内核覆盖热点标量与 Transformer 辅助算子，并对尚未实现的原生内核使用 CPU 回退。

6. **TensorSharp.Backends.MLX** 是 Apple Silicon 上的 MLX 路径。它封装 [mlx-c](https://github.com/ml-explore/mlx-c)（`libmlxc`），提供分配器、存储、异步 worker 派发、量化 / 融合 / 编译内核、MoE 专家 offload，以及对未实现算子的 CPU 回退层。

7. **TensorSharp.Server** 是 HTTP / 应用层，提供兼容 Ollama 与 OpenAI 的 REST API、浏览器聊天 UI、上传处理；其中 `InferenceEngineHost` 持有自回归模型的连续批处理引擎，`DiffusionBatchScheduler` 处理 DiffusionGemma 的 Web UI 轮次，旧的队列状态接口保留作为向后兼容。

8. **TensorSharp.Cli** 是控制台 / 应用层，用于本地 prompt 运行、多模态实验、prompt 检查、JSONL 批处理、交互式 REPL 与内置的 prefill / decode 基准。

### 性能优化

下表是跨架构汇总；[`docs/models/`](docs/models/README_zh-cn.md) 里每个模型卡片会在上下文中走一遍同样的内核，包含具体派发的 GGML 图与触发融合路径的条件。

- **融合 GPU decode**（Gemma 4）：在 Metal 上将所有 Transformer 层合并为单次 GGML 计算图调度，将每个 token 的 CPU-GPU 往返从数百次降低到一次。相较逐算子调度约提升 2.6 倍。
- **融合 GPU prefill**（Gemma 4）：对于密集（非 MoE、非 KV 共享、无 PLE/多模态）层，`Gemma4LayerPrefill` 将整个 Transformer 块（RMSNorm + QKV + QK-norm + RoPE + 注意力 + 输出投影 + post-attn norm + GeGLU FFN + post-FFN norm + 残差 + 层缩放因子）合并为 prefill 期间每层一次的 GGML 计算图调度，将融合方法从单 token decode 扩展到多 token prefill。
- **分块 prefill**（Gemma 4）：长提示被拆分为有界的分块（2 倍滑动窗口，最大 2048 tokens），以避免 SWA 层上 O(n²) 的注意力分数张量。分块在纯文本（无多模态嵌入）时自动应用，确保每个分块在 SWA 窗口预算内。
- **整模型原生 decode**（Qwen 3）：所有 Transformer 层在一次原生调用（`TransformerModelDecode`）中完成，每层权重指针在加载阶段预解析并缓存，从 decode 热点路径中移除托管循环开销。
- **融合 Qwen 3.5/3.6-family attention 层 decode**：单次 GGML 计算图为每个 FullAttention 层完成 RMSNorm + 融合 QKV + Q/gate 反交错 + 每头 QK norm + RoPE + KV 缓存追加 + flash attention + sigmoid 门控混合 + 输出投影 + 残差加法。替换了原本每层 ~2 次独立 GGML 调用与 ~6 个小型 CPU/GPU 同步点。当缓存序列长度超过 4096 token 时启用（可通过 `FUSED_ATTN_LAYER_MIN_SEQ_LEN=N` 覆盖）。
- **融合 prefill 注意力**（Qwen 3.5/3.6-family）：`FusedPrefillAttention` 将 Q*K^T、因果掩码、softmax 和 *V 合并为 prefill 期间每个注意力层一次的 GGML 计算图调度，消除了每个注意力层约 5 次独立的 C# 到 GGML 往返。同时支持初始 prefill 和带有已有 KV 缓存条目的续接。
- **融合输出投影 + FFN**（Qwen 3.5/3.6-family）：对于 FullAttention 和 GatedDeltaNet 中的 dense FFN 层，`FusedOutProjFFN` 将输出投影、残差加法、post-attention RMSNorm 以及完整的 SwiGLU FFN（gate_up matmul + SiLU + down matmul + 残差加法）合并为单次 GGML 计算图调度，将每层 2 次 GPU 往返减少为 1 次。
- **融合输出投影 + 归一化 + 路由器**（Qwen 3.5/3.6-family MoE）：`FusedOutProjNormRouter` 将 GatedDeltaNet 输出投影、残差加法、post-attention RMSNorm 和 MoE 路由器投影合并为一次调度。预计算的路由器 logits 随后由批量 MoE 内核直接消费，消除了每个 MoE 层的独立路由器调度。
- **融合视觉编码器**（Qwen 3.5/3.6-family）：`FusedVisionAttention` 将 LayerNorm + QKV + 偏置 + 2D RoPE + 缩放点积注意力 + 输出投影 + 偏置 + 残差合并为一次 GGML 计算图调度（~8 个算子 → 1）。`FusedVisionMLP` 将 LayerNorm + up + 偏置 + GELU + down + 偏置 + 残差合并为一次调度（7 个算子 → 1）。两者结合将每个编码器块的 GPU 往返从约 15 次减少到 2 次。
- **融合权重投影**：Q/K/V 投影融合为单次 QKV matmul；gate 与 up 投影融合为单次 gate_up matmul。
- **原生量化计算**：量化权重（Q4_K_M、Q6_K、Q8_0、IQ2_XXS、MXFP4 等）直接参与 matmul，无需展开为 FP32，节省内存与带宽。批量 `AddmmQuantBatch` 内核可在一次调度内完成对同一量化权重块的多个子矩阵 matmul。
- **Direct CUDA 内核**：`cuda` 后端加速 fill/copy、unary ops、融合激活、RMSNorm、softmax、index select、因果掩码、RoPE/RoPEEx、cuBLAS GEMM，以及受支持的量化 matmul/get-rows；未覆盖算子会安全回退。
- **批量 GPU MoE**：`MoEExpertsSwiGLUResidual`（Qwen 3.5/3.6-family）和 `MoEExpertsForward`（Nemotron-H）将每个 MoE 层中所有被选中的专家——以及 Qwen 3.5/3.6-family 中可选的 shared expert 与残差加法——合并为一次 GGML 计算图调度。
- **基于 GEMM 的视觉 patch embedding**（Qwen 3.5/3.6-family）：将 patch embedding 重构为并行 im2col + 矩阵乘法，把单线程标量五重嵌套循环替换为可在 GPU 上加速的 matmul。
- **并行化 Q/gate 反交错**（Qwen 3.5/3.6-family）：FullAttention prefill 中的 Q + sigmoid-gate 反交错按 token 并行化，长 prompt 时可随 CPU 核心数线性扩展。
- **优化后的纯 C# CPU 路径**：托管 GEMM 快速路径和连续 Float32 内核加速了 decode、softmax、RMSNorm、RoPE、融合激活等热点路径，同时在 CPU 加载时保持量化 GGUF 权重压缩状态。
- **环形 KV 缓存**：滑动窗口注意力层使用固定大小环形缓冲区，使内存占用不随序列长度增长。
- **KV 缓存前缀复用**：多轮对话会复用各轮之间最长的匹配 token 前缀。对 SWA 模型，截断会自动按滑动窗口大小回退，使后缀部分可以重建 SWA 上下文。
- **分页 KV 缓存 & 块哈希前缀共享**：连续批处理引擎把 KV 切分成固定大小的块，对每个写满的块做内容哈希，并在并发 / 历史请求间共享。尚未实现 `IBatchedPagedModel` 的模型仍会走同一引擎内隔离的按序列 KV-swap 回退路径。
- **原生分页注意力内核**：`TSGgml_PagedAttentionForward`（及面向 GPT OSS 的 `WithSinks` 变体）在 C++ 中按序列从分页缓冲区聚合 K/V，按序列构建小型 GGML 图，并派发 `ggml_flash_attn_ext`——也就是旧的单序列路径所使用的同一融合 GPU flash 注意力内核（Metal/CUDA/Vulkan）。在 Ministral-3-14B 长上下文（4×~800 tokens）上比旧的按序列 GGML 路径**快 ~21%**。
- **批处理 / 分页前向**：Mistral 3、Gemma 4、GPT OSS、Qwen 3.5/3.6（含 GatedDeltaNet 递归状态池）、Nemotron-H（含 Mamba2 递归状态池 + 原生批处理 Mamba2 内核）把 N 个序列打包到一次 `ForwardBatch` 调用中，每层执行一次批处理线性投影 matmul，通过 `slotMapping` 写入分页 K/V，并通过原生内核做按序列注意力。Gemma 4 批处理路径在 batch=8 短 prompt 下达到 **1.5×** 旧吞吐，在 4×800-token prompt 下达到 **1.6×**；Nemotron-H Mamba2 批处理在 Apple M4 Pro 上 batch=3 时达到 **3.95×**。详见 [docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)。
- **MTP / NextN 投机解码**：单序列可运行多 token 预测草稿头（Qwen 3.6 内嵌 NextN 块；Gemma 4 独立 `gemma4-assistant` 草稿 GGUF）。草稿头最多提议 `--mtp-draft` 个 token，主干用一次批量前向验证，二者均由该请求自己的采样器驱动，因此在不改变输出的前提下加速 decode。在 ggml 后端上，融合的单图多 token 验证与草稿步内核（`NativeGemma4ModelVerify` / `TryFusedMoEModelVerify` / `NativeGemma4DraftStep`，以及 Qwen 3.6 的 NextN 图）摊销了验证开销；Gemma 4 路径还增加了 gallocr 验证 scratch 以及部分接受时避免重跑已保留前缀的稠密快速回滚。纯 C# `cuda` 后端运行完全驻留 GPU 的逐算子验证 / 草稿（donor 缓存注意力、GQA decode 内核、GPU RoPE），使验证层循环零宿主端同步停顿。默认关闭；`--mtp-spec`。
- **DiffusionGemma prompt-KV 缓存与融合去噪**：GPU 后端会在每个 block 中只对 `[prompt | canvas]` 的 prompt 部分预填充一次 K/V，并在去噪多步中复用；GGML 后端默认使用融合整模型 diffusion decode 与融合 lm-head tail。Web UI 通过 `DiffusionBatchScheduler` 在 block 边界批处理并发 diffusion 请求。
- **内核预热**：CLI 和 Server 在启动时运行一次微型前向传播，以预编译 GPU 内核（Metal pipeline state、CUDA JIT）并预热内存池，避免首次推理请求的冷启动延迟。
- **Prefill 缓存**（Gemma 4、Qwen 3.5/3.6-family）：逐 forward 传播的 SWA 掩码缓存（Gemma 4）、跨全局层的 NeoX RoPE cos/sin 查找表缓存（Gemma 4）、以及跨层的 RoPE 位置张量缓存（Gemma 4、Qwen 3.5/3.6-family），消除了 prefill 期间的冗余重复计算。
- **原地 QK RMSNorm**（Qwen 3.5/3.6-family）：逐头 QK 归一化通过 `View` 原地执行，避免了每层每个 Q/K 的一次张量分配与拷贝。

### 内存优化

- **零拷贝文件映射量化权重**（Direct CUDA、GGML CUDA、GGML Metal、GGML CPU）：GGUF 模型文件以内存映射方式打开，量化张量通过 host 指针缓冲区直接绑定到原生算子。这样省去了之前每张张量从磁盘复制到新分配原生堆缓冲区的过程——这一过程在 Apple Silicon 上会让大型量化模型的常驻内存几乎翻倍。例如，`Qwen3.5-35B-A3B-IQ2_XXS`（约 10 GB GGUF）在 Metal 后端的实际工作内存峰值从约 17 GB 降至约 7 GB。映射文件由操作系统的页缓存管理，必要时可换出，且在 Apple Silicon（统一内存）上不会带来推理性能损失。
- **最佳匹配内存池**：GGML 主机分配器使用 best-fit 而非 first-fit 在已池化块中检索可重用空间，避免把大块草稿内存交给小型中间张量请求，从而把工作集严格控制在合理范围内。
- **有界池保留量**：集成 GPU / CPU 内存池现在将单个保留块上限设为 64 MB，整池上限设为 32 块。结合 mmap 后的权重，可在快速复用短生命中间张量的同时限制峰值常驻内存。
- **高内存效率模型加载**：大张量直接流式加载到原生内存，避免中间托管分配。F32 权重与 norm 仍按需加载；量化权重在受支持的后端上通过 mmap 方式绑定。
- **可选 SSD 溢出的分页 KV 块池**：`PagedKvBlockStore` 保留了 RAM / SSD 分层块存储能力（`TS_KV_CACHE_MAX_RAM_MB`、`TS_KV_CACHE_SSD_DIR`、`TS_KV_CACHE_MAX_SSD_MB`），主要服务独立分页 KV 组件与后续扩展；服务端请求路径的活跃块由每个引擎的 `BlockPool` 统一管理。
- **KV 块编解码器**：`TurboQuantKvCodec`（2-bit 仿射、Q4 或 Q8）可压缩分页块，以精度换取更小的每块带宽与内存占用——大致减半（Q8）、减为四分之一（Q4）或约十分之一（2-bit，fp32 块）。2-bit 档位使用每组仿射 min+scale（即 llama.cpp Q2_K 背后的 block-min 思路），让四个码值覆盖该组的实际取值范围；它面向超长上下文的远端前缀复用，此时注意力权重远大于量化噪声。带递归状态的模型会自动回退到 passthrough。


## 测试

### 单元测试（xUnit）

`InferenceWeb.Tests` 覆盖无需启动服务的进程内行为：托管量化算子、可用 CUDA 设备上的 Direct CUDA 后端内核、可用 MLX 时的 MLX 后端内核、分页 KV 缓存调度（`ContinuousBatchSchedulerTests`、`PagedKvCacheTests`、`PagedKvCacheCodecTests`）、批处理执行器正确性（`BatchedExecutorTests`）、按模型批处理前向与旧路径的一致性（`Qwen35BatchedCorrectnessTests`、`Mistral3BatchedForwardTests`、`Gemma4BatchedForwardTests`、`GptOssBatchedCorrectnessTests`、`NemotronBatchedCorrectnessTests`）、MTP / NextN 投机解码正确性与可选端到端探针（`MtpSpeculativeExecutionTests`、`Qwen36MtpTests`、`Gemma4MtpTests`）、DiffusionGemma 去噪 / prompt-KV / 批处理生成探针（`DiffusionGemmaTests`）、按模型批处理性能微基准（`*BatchedPerfBench.cs`）、`TurboQuantKvCodec` 编解码往返、prefill 分块、KV 缓存策略、KV 缓存 Prompt 渲染与多轮集成、聊天会话与 SessionManager 隔离、ModelService 历史跟踪、请求日志中间件与文件日志 Provider、图像预处理、媒体辅助逻辑、结构化输出校验、文本上传辅助、ModelService 上传日志、Web UI 聊天策略、模型上下文长度解析、可用后端发现，以及服务器 CLI 选项构造（`ServerOptionsBuilderTests`）。

```bash
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj
```

### 服务端集成测试

TensorSharp.Server 的集成测试位于 `TensorSharp.Server/testdata/`。测试覆盖所有三种 API 风格（Web UI SSE、Ollama、OpenAI）、多轮对话、思维链模式、工具调用、结构化输出、队列状态兼容、并发请求和中断支持。架构特定能力（思维链、工具调用）会自动检测，当前模型不支持时会自动跳过。

```bash
# 先启动 TensorSharp.Server，然后运行：
python3 TensorSharp.Server/testdata/test_multiturn.py
# 或
bash TensorSharp.Server/testdata/test_multiturn.sh
```

完整测试矩阵见 [TensorSharp.Server/testdata/README.md](TensorSharp.Server/testdata/README.md)。

### 推理矩阵运行器

`TensorSharp.TestMatrix` 是更大的 CLI 驱动覆盖工具，用于长时间模型 / 后端验证。它会发现 GGUF 文件，过滤不可用后端与不受支持的提示类型，运行 baseline 与环境变量 sweep，用每个 cell 一个 JSON 的形式保存结果，生成汇总 Markdown 报告，并可按需与每类主机的基线做回归对比。

```bash
dotnet build TensorSharp.TestMatrix/TensorSharp.TestMatrix.csproj -c Release
dotnet run --project TensorSharp.TestMatrix -c Release -- --dry-run
```

当前运行器契约见 [TensorSharp.TestMatrix/README_zh-cn.md](TensorSharp.TestMatrix/README_zh-cn.md) 与 [docs/env_var_feature_matrix_zh-cn.md](docs/env_var_feature_matrix_zh-cn.md)。

