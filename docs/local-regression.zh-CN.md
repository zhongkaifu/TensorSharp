# 本地自动化回归：部署、模型准备与诊断

[English](local-regression.md) · [覆盖清单](local-regression-coverage.md) · [开发与构建](../DEVELOPMENT_zh-cn.md)

统一入口是 `python eng/run-local-regression.py`。运行器根据本机资源、已有文件、配置和场景条件制定计划，按顺序准备模型并执行场景，最后生成本地报告。一次场景可能同时拥有服务器和客户端进程；配对模型的 profile 必须声明该场景内的全部模型和资源。

必须区分三件事：源码支持某项能力、运行器已有对应执行器、某个模型与后端组合已经实测通过。配置数量包含独立模型、LoRA 插件和叠加配置，不能直接当成已验证模型数量。以本次 `report.json` 中的模型身份、构建身份和断言结果为准。

## 1. 先选一种工作方式

| 工作方式 | 所需内容 | 会做什么 |
|---|---|---|
| 配置审计 `--mode audit` | 仓库、Python 3.10+ | 检查配置和覆盖声明；不下载模型载荷、不运行模型 |
| 离线计划 `--mode plan`，也是默认模式 | 同上；可提供已部署程序和本地模型目录 | 检查硬件、文件大小、运行时元数据、预算和场景前提；不执行推理 |
| 使用已构建程序 `--mode run` | Python、完整 CLI/server/probe 发布目录及其运行依赖 | 校验模型身份并执行已选择场景 |
| 从源码部署 | .NET 10 SDK、所选后端的原生工具链、独立构建目录 | 先构建原生库和宿主，再运行回归；构建和模型下载是不同阶段 |

框架依赖型 `.dll` 通过本地 `dotnet` 启动；自包含宿主可以直接传可执行文件。需要构建时安装完整 SDK，不能用只有 runtime 的安装代替。不要只复制一个 DLL：宿主依赖、原生库、CUDA 内核和运行时配置必须一起部署。

在仓库根目录执行：

```powershell
python --version
dotnet --list-sdks
python eng/run-local-regression.py --help
python eng/run-local-regression.py --mode audit
python eng/run-local-regression.py --mode plan --model-root D:\TensorSharp-models --existing-root C:\work\model
```

`--help` 是当前 checkout 的参数契约。切换分支或更新运行器后，应重新审计；不要把旧报告的通过状态迁移到新二进制。

## 2. 后端和机器条件

| 后端 ID | 典型环境 | 需要确认的条件 |
|---|---|---|
| `cpu` | Windows、Linux、macOS | 托管 CPU 路径；具体模型是否接入该路径由目录约束决定 |
| `ggml_cpu` | Windows、Linux、macOS | 对应系统的 GGML 原生库及其依赖 |
| `ggml_cuda` | Windows 或 Linux/WSL 的 NVIDIA GPU | CUDA 驱动、CUDA 版 GGML 库及构建能力证据 |
| `cuda` | Windows 或 Linux/WSL 的 NVIDIA GPU | 独立 C# CUDA 后端、cuBLAS 和匹配的 PTX/fatbin；不是 `ggml_cuda` 的别名 |
| `ggml_metal` | Apple 平台 | Metal 版 GGML 库；本指南给出 macOS 源码构建配方 |
| `mlx` | Apple Silicon macOS | MLX 原生依赖及 Metal 内核；不能由 `ggml_metal` 通过推导出来 |
| `ggml_vulkan` | Windows/Linux 的可用 Vulkan GPU | 编译进库的 Vulkan 后端、相应驱动及模型适用性 |

`--backend auto` 根据能力证据和预算选择一个后端，并记录选择过程。`--backend ggml_cuda` 等明确请求不能悄悄回落到 CPU。`--backend all` 遍历预设声明的后端；没有设备或构建条件的组合会保留跳过、未支持或未测试状态。

本仓库 CUDA 构建要求 CUDA 13 系列；当前开发说明使用 CUDA 13.4，并要求 Turing/SM 7.5 及以上 GPU 和 R580 及以上驱动。实际约束见 [DEVELOPMENT.md](../DEVELOPMENT.md) 和各构建脚本。驱动、工具包、GPU 名称或一个 `GgmlOps.dll` 的存在都不能独自证明正确后端已经执行。

本轮本地证据来自 Windows。Linux/WSL GPU、macOS/Metal/MLX、Vulkan、多 GPU 和分布式执行必须分别验证；这里的命令是源码支持的部署配方，不是这些平台已通过的声明。本任务没有访问 Mac。

## 3. 构建与运行隔离

在独立 checkout 中完成构建，避免不同任务改写同一个 `bin`、`obj` 或原生构建树。不要对正在运行推理或链接原生 DLL 的目录重新配置、清理或复制库文件。

首次构建的 ggml 获取默认跟随上游分支。要求可复现时，通过 `TENSORSHARP_GGML_GIT_REF` 明确选择依赖 revision，并记录实际 checkout；已有依赖离线重建可用 `TENSORSHARP_GGML_NO_UPDATE=1`。不要修改上游 ggml 源码来绕过测试失败。

现有默认输出位置如下；宿主并非都输出到 `bin/Release/net10.0`：

| 程序/资产 | 默认路径 |
|---|---|
| CLI | `TensorSharp.Cli/bin/TensorSharp.Cli.dll` |
| Server | `TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll` |
| 固定历史 logit probe | `eng/ForcedLogitProbe/bin/Release/net10.0/ForcedLogitProbe.dll` |
| HTTP SDK 客户端示例 | `examples/dotnet-local-chat/bin/Release/net10.0/TensorSharp.LocalChat.dll` |
| 原生 IChatClient 示例 | `examples/tensorsharp-native-chat/bin/Release/net10.0/TensorSharp.NativeChat.dll` |
| n-gram 对照基准宿主 | `benchmarks/AgentTurnBench/bin/Release/net10.0/AgentTurnBench.dll` |
| Windows GGML 构建树 | `TensorSharp.GGML.Native/build-windows` |
| Linux/macOS GGML 构建树 | `TensorSharp.GGML.Native/build` |

`--runtime-dir` 只收集部署库、CUDA 内核和 CMake 元数据，**不会部署库，也不会修改 DLL/so/dylib 搜索路径**。为每个实际启动的宿主部署正确库；服务器目录中有库，不代表 probe 目录已经可运行。

### Windows + CUDA

准备 .NET 10 SDK、CMake 3.20+、Visual Studio 2022/2026 的 x64 C++ 构建工具、CUDA 13 和兼容驱动。脚本会定位 VS/CMake/Ninja；多工具包机器可通过 `CUDACXX` 指定真实的 `nvcc` 路径。构建可能联网获取上游 ggml、可选 cuDNN 或工具链，这些是构建依赖，不受运行器的模型下载额度控制。

下面在**独立 checkout 根目录**执行，先完成原生库，再构建托管宿主：

```powershell
$env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA = 'ON'
$env:TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN = 'OFF'
$env:TENSORSHARP_GGML_NATIVE_BUILD_TESTS = 'ON'
& .\TensorSharp.GGML.Native\build-windows.ps1 --cuda --no-vulkan

dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
dotnet build eng/ForcedLogitProbe/ForcedLogitProbe.csproj -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:ForcedLogitProbeRequireGgmlNative=true
```

第一次构建不要提前加 `TensorSharpSkipCudaPtx=true`：CUDA 输出目录还没有合法预构建内核时，这个选项会明确失败。宿主首次构建仍会准备 CUDA 内核。仅 CPU 原生库可用 `build-windows.ps1 --no-cuda --no-vulkan`，并把 CUDA 环境开关改为 `OFF`。

### Linux / WSL + CUDA

在 Linux 环境中安装 Linux 版本的 .NET SDK、CMake、C/C++ 工具链和 CUDA 工具包。WSL 使用其可见的 NVIDIA 驱动接口；先确认该环境内 `nvidia-smi` 和 `nvcc --version` 能正常工作。Windows DLL 不能代替 Linux `.so`。建议用 Linux 文件系统中的独立 checkout 和输出目录。

```bash
export TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON
export TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF
export TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON
bash TensorSharp.GGML.Native/build-linux.sh --cuda --no-vulkan

dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
dotnet build eng/ForcedLogitProbe/ForcedLogitProbe.csproj -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:ForcedLogitProbeRequireGgmlNative=true
```

只构建 GGML CPU 时使用 `build-linux.sh --no-cuda --no-vulkan`，并将 CUDA 开关设为 `OFF`。上述流程未在本轮完成 Linux/WSL GPU 验证。

### macOS + Metal / 可选 MLX

GGML Metal 构建需要 .NET 10 SDK、CMake 3.20+ 和 Xcode Command Line Tools。MLX 还需要完整 Xcode 与 Metal toolchain；详见 [MLX 构建说明](../DEVELOPMENT.md#build-the-native-mlx-library-macos-only)。以下只是供用户在获授权的本机执行的配方：

```bash
bash TensorSharp.GGML.Native/build-macos.sh
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
dotnet build eng/ForcedLogitProbe/ForcedLogitProbe.csproj -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true -p:ForcedLogitProbeRequireGgmlNative=true
```

需要 MLX 时，先单独运行 `bash TensorSharp.Backends.MLX/build-native-macos.sh`，然后重新构建宿主以部署依赖。搬运发布目录时保留 `libmlxc.dylib`、`libmlx.dylib` 及同目录的 `mlx.metallib` 等项目实际复制的文件。GGML Metal 与 MLX 需要分别运行，不能共享通过结论。

### 额外客户端、基准宿主和组件测试

按选择的场景准备额外程序；运行器不会因某个 DLL 缺失而隐式构建宿主：

```powershell
dotnet restore examples/dotnet-local-chat/TensorSharp.LocalChat.csproj --locked-mode --configfile examples/dotnet-local-chat/NuGet.Config
dotnet build examples/dotnet-local-chat/TensorSharp.LocalChat.csproj -c Release --no-restore
dotnet build examples/tensorsharp-native-chat/TensorSharp.NativeChat.csproj -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
dotnet build benchmarks/AgentTurnBench/AgentTurnBench.csproj -c Release -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
```

`--include-suites` 加入 InferenceWeb、Knowledge、Studio 和客户端契约测试。默认要求已构建程序集，并使用 `--no-build --no-restore`。显式 `--allow-builds` 只允许这些套件的托管构建/恢复，可能下载 NuGet 包；它不构建宿主、不编译原生后端。此模式设置 `TensorSharpSkipGgmlNative`、`TensorSharpSkipMlxNative`、`TensorSharpSkipCudaPtx`，要求已有 CUDA 内核资产。已有 GGML/MLX 库仍可能由普通项目复制目标部署。

通用组件通道固定 GGML CPU 并排除 GPU 执行 trait 和真实模型要求；测试发现阶段仍可能查询 GPU 驱动。它保留 GGML CPU 测试，但不能作为 CUDA/Metal 模型通过证据。预构建旧程序集中的 Wan 真实文件夹具也被显式排除。

## 4. 模型、临时文件和报告可放在不同盘

推荐区分四类目录：源码/构建、模型缓存、只读已有模型、报告。系统盘空间不足时不必移动或删除已有权重。

```powershell
$validationRoot = 'D:\TensorSharp-regression'
$checkout = Join-Path $validationRoot 'source'
$modelCache = Join-Path $validationRoot 'models'
$temporaryRoot = Join-Path $validationRoot 'temp'
$artifactRoot = 'E:\TensorSharp-regression-results'
New-Item -ItemType Directory -Force -Path $modelCache,$temporaryRoot,$artifactRoot | Out-Null
$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot
$planOutput = Join-Path $artifactRoot ('plan-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
python "$checkout\eng\run-local-regression.py" --repo $checkout --mode plan `
  --model-root $modelCache --existing-root C:\work\model `
  --output $planOutput --disk-headroom-gib 10
```

`--existing-root` 可重复传入，只按唯一文件名发现候选；重复同名文件不自动猜测，GGUF 分片必须同目录。执行前校验期望大小和 SHA-256。`--model-root` 是新下载及断点文件的位置，不会迁移源码、已有模型或 `bin/obj`。Linux 使用挂载路径，临时目录可用进程级 `TMPDIR`；NuGet 缓存可单独指定 `NUGET_PACKAGES`。

`--output` 可指向尚不存在的目录或空目录；非空目录会在执行前被拒绝，防止复用旧通过证据。模型盘和报告盘分别检查空间：模型盘有余量不能弥补报告盘已满。

## 5. 下载、缓存和预算

- 默认不下载模型。`--allow-download` 允许准备缺失文件；`--max-download-gib` 默认 20 GiB，是整次运行的传输预算。构建依赖下载另算。
- 审计/计划不下载载荷。`--probe-network` 允许对已固定来源查询元数据；`--mode plan --allow-download` 可以检查“允许下载后”的可执行计划，但仍不下载权重。
- 自动模型来源要求目录中已记录的 HTTPS Hugging Face URL、不可变 40 位 revision 和 SHA-256；缺少身份、分片或授权必须记录阻塞，不能补猜地址。需要已授权凭据时在进程环境设置 `HF_TOKEN`，不要放进命令、配置或报告；运行器不会接受 gated-model 协议。
- 下载采用带身份的 `.part` 和恢复元数据，校验范围、最终长度与哈希后原子安装。错误的已有文件不会被覆盖；已有用户模型不会被删除。Ctrl+C 保留可恢复的部分文件，并清理所拥有的进程。
- 普通文本/嵌入规划默认按完整权重估计：主机 RAM 约为 `1.25 × 权重 + 2 GiB`，保留至少 4 GiB 或物理内存的 20%；GPU 还独立计算显存需求和 2 GiB 余量。不会把多张卡显存相加，也不会默认认为 SSD 专家换页就能运行。
- `--ram-budget-gib` / `--vram-budget-gib` 限制可用额度；`--ram-required-gib` / `--vram-required-gib` 是明确的工作负载需求声明。未知大小或可用内存不会成为通过。覆盖值不能证明架构支持某种 offload。
- `--max-models` 限制准备的预设数量；`--timeout-seconds` 限制场景进程；`--context-tokens` 显式设置 1–2048 的上下文。夹具需要更多上下文时会跳过，不能静默缩短夹具以制造通过。
- `--assume-backend-available` 仅覆盖缺失的预检元数据；实际执行仍必须证明请求后端，不能覆盖模型支持矩阵。

规划是保守筛选，不是峰值内存保证。大型生成、配对模型、LoRA、多个 rank 或自定义 profile 必须申报其全部常驻模型及工作区；不要只填激活参数量。

### CPU-MoE 显式配置

`--cpu-moe` 当前只接入 `qwen35/moe` 和 `qwen4exp/moe` 的 `ggml_cuda` / `ggml_cpu` 路径。必须同时给出两个正的需求值；即使在 `ggml_cpu` 上也要提供 `--vram-required-gib`，但不会向 CPU 场景收取显存预算。

```text
--models qwen3.8-flash-next-iq1_m --backend ggml_cuda --cpu-moe --ram-required-gib <已建立的RAM峰值GiB> --vram-required-gib <已建立的VRAM峰值GiB>
```

这是参数形状示意，尖括号必须替换为本机该量化、上下文和场景的已建立需求。它将路由专家放在 CPU；注意力、router 和共享专家仍按请求后端执行。模型全部分片仍需下载/校验并占磁盘。报告同时记录目录架构和实际模型架构、offload 配置、后端断言及需求来源；旧宿主忽略参数不能通过。

## 6. 从一个小场景开始，再扩大矩阵

已部署 Windows CUDA 宿主且已有 Qwen3.5-0.8B 时，一条命令运行三个小场景：

```powershell
python eng/run-local-regression.py --mode run --models qwen3.5-0.8b `
  --scenarios cli_text,cli_prefill,cli_decode --backend ggml_cuda `
  --model-root D:\TensorSharp-models --existing-root C:\work\model `
  --cli .\TensorSharp.Cli\bin\TensorSharp.Cli.dll `
  --runtime-dir .\TensorSharp.GGML.Native\build-windows `
  --max-models 1 --context-tokens 256 --output .\artifacts\qwen08-smoke
```

没有该模型时可显式加 `--allow-download --max-download-gib 2`。改变 `--backend` 后仍要准备相应运行时。图像、工具、会话或 logit 场景不能照抄小上下文限制；默认 2048 适用于普通预置夹具。

CPU 嵌入小场景示例：

```powershell
python eng/run-local-regression.py --mode run --models embedding-minilm `
  --scenarios server_embeddings --backend cpu `
  --server .\TensorSharp.Server.Host\bin\TensorSharp.Server.Host.dll `
  --model-root D:\TensorSharp-models --allow-download --max-download-gib 1 `
  --output .\artifacts\minilm-cpu
```

在完整宿主、客户端、probe、测试程序集和已批准夹具都准备好之后，可以一条命令选择整个目录。下面仍受预算约束；缺失前提会保留状态，不会证明“所有组合已通过”：

```powershell
python eng/run-local-regression.py --mode run --models '*' --scenarios '*' --backend all `
  --model-root D:\TensorSharp-models --existing-root C:\work\model `
  --scenario-profiles .\artifacts\profiles\approved-scenarios.json `
  --allow-download --max-download-gib 40 --disk-headroom-gib 10 `
  --include-suites --strict --output E:\TensorSharp-regression-results\full-001
```

上例的 `approved-scenarios.json` 必须先按下一节生成并补齐已批准的场景输入；文件不存在时不能直接运行。只希望普通预设各选一个适合本机的后端时改成 `--backend auto`；profile 始终采用自身声明的明确后端。`--models`、`--scenarios` 接受逗号分隔的 ID 通配符；全局 `--include-suites`、`--native-test-build-dir` 不受普通每模型选择器限制。

### 固定历史和媒体输入

`forced_logits` / `forced_logits_sequence` 需要 `--logit-probe`；当前 probe 只接入 `cpu`、`cuda`、`ggml_cpu`、`ggml_cuda`、`ggml_metal`，不能把 MLX/Vulkan 的其他测试通过当作 logit 覆盖。后者使用 32/512-token 的固定前缀，比较 32 行 logits，即一次 prefill 行和 31 次 teacher-forced decode；覆盖短短、长长、短长短、长短顺序，并比较复用后的 `ResetKVCache` 与新进程控制。非有限值、历史或模型身份不匹配、后端回落和首个数值分歧都会保留证据。相同后端重复一致并不等于独立 Transformers oracle 正确性。

CLI 媒体夹具从 `--media-dir` 获取 `image.png`、`sample.wav`、`sample.mp4`，对应生成脚本为 [eng/make-test-media.sh](../eng/make-test-media.sh)。图像/视频理解需要匹配 projector。运行器不会扫描私人图片或任意文件来凑测试输入。

`cli_tools` / `server_tools` 校验模型生成的函数名及 JSON 参数，不执行这些调用；`server_agents` 则明确执行受限的协调工作流，并禁用 skills、代码执行和可变 worker 工具。它不能证明所有 agent 工作流或文件系统隔离。

## 7. 自定义场景 profiles：与模型配置分开维护

模型预设描述权重及模型选项；场景 profile 描述本次要验证的具体组合、夹具、参数、期望和资源。不要把可执行 shell 命令、任意下载地址或密钥写进 profile。

统一入口已接入 `--scenario-profiles PATH`，14 个扩展场景均注册了具体执行器。通用格式见 [scenario-profiles.schema.json](../config/regression/scenario-profiles.schema.json)；加载器还验证每个执行器的严格字段与资源契约。注册和合成测试通过不等于模型场景已实测，缺少硬件、批准参考或实现分支支持时会保留明确状态。文件可使用 JSONC 注释和尾逗号，建议生成普通 JSON。各执行器的 `parameters` / `expected` 不能以空对象规避断言。

| 字段 | 类型和含义 |
|---|---|
| `schema_version` | 当前约定为整数 `1` |
| `profiles` | profile 数组 |
| `id` | 本文件内唯一的场景实例 ID |
| `scenario` | 已注册场景 ID，例如 `image_editing`，不能是任意命令名 |
| `preset` | 目录中已存在的主预设 ID |
| `backend` | 明确后端，执行器必须校验适用范围 |
| `models` | `角色 -> 目录预设 ID`；主模型由 `preset` 隐式确定，其他角色声明 drafter、encoder 等实际依赖，角色名由执行器严格限定；LoRA adapter 属于有哈希的 fixture |
| `fixtures` | `角色 -> {path, sha256}`；相对路径按 profile JSON 所在目录解析，每个文件至多 512 MiB；启动前验证真实 SHA-256，缺失和不匹配必须阻止执行 |
| `parameters` | 该执行器允许的输入、采样、输出形状或运行参数 |
| `expected` | 场景特有断言、标签、范围或数值容差；必须能区分错误输出 |
| `resources` | `ram_required_gib` 必须为正值；GPU profile 还必须提供正的 `vram_required_gib`。CPU 可省略显存项，不填写零值。可选 `per_gpu_required_gib` 是 `设备索引 -> GiB` 映射，`scratch_gib` 是正的临时空间需求；需求覆盖完整组合 |

可用 PowerShell `Get-FileHash -Algorithm SHA256 -LiteralPath <夹具文件>` 或 Linux/macOS 的 SHA-256 工具计算夹具身份。不要把示例零哈希、占位预算或另一个模型的峰值直接用于运行。

profile 文档最大 1 MiB，最多 128 个实例。夹具只接受本地路径，不接受 URL 或 UNC 共享；每个执行器可以施加比全局 512 MiB 更小的限制。

### 配对检索 profile 示例

下面生成一份只有合成资料的 TXT 和真实哈希，再生成 typed profile。它使用 Qwen3.5-0.8B 回答、MiniLM CPU encoder 检索。答案只出现在资料中，问题要求附带原文链接。`8 GiB` 是调用者对这两个模型及工作区的保守需求声明示例，不是实测峰值或适用所有机器的保证；应先查看本机计划。

```powershell
$profileDir = Join-Path (Get-Location) 'artifacts\profiles\retrieval-demo'
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
$fixturePath = Join-Path $profileDir 'document.txt'
$profileFile = Join-Path $profileDir 'retrieval.json'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$document = 'The designated maintenance code for LANTERN-STATION is COBALT-7319.'
[System.IO.File]::WriteAllText($fixturePath, $document, $utf8)
$sha256 = [System.Security.Cryptography.SHA256]::Create()
try {
  $documentSha = ([BitConverter]::ToString($sha256.ComputeHash(
    [System.IO.File]::ReadAllBytes($fixturePath)))).Replace('-', '').ToLowerInvariant()
} finally {
  $sha256.Dispose()
}
$profile = @{
  schema_version = 1
  profiles = @(@{
    id = 'qwen08-document-retrieval'
    scenario = 'retrieval'
    preset = 'qwen3.5-0.8b'
    backend = 'cpu'
    models = @{ embedding = 'embedding-minilm' }
    fixtures = @{ document = @{ path = 'document.txt'; sha256 = $documentSha } }
    parameters = @{
      embedding_backend = 'cpu'
      question = 'What is the maintenance code for LANTERN-STATION? Cite the source URL.'
      search_query = 'LANTERN-STATION maintenance code'
      max_tokens = 384
    }
    expected = @{
      answer_contains = 'COBALT-7319'
      source_excerpt = $document
    }
    resources = @{ ram_required_gib = 8 }
  })
}
[System.IO.File]::WriteAllText($profileFile, ($profile | ConvertTo-Json -Depth 12), $utf8)
```

先计划，再运行；不需要修改原始模型预设：

```powershell
python eng/run-local-regression.py --mode plan --models qwen3.5-0.8b --scenarios retrieval `
  --scenario-profiles $profileFile --model-root D:\TensorSharp-models `
  --server .\TensorSharp.Server.Host\bin\TensorSharp.Server.Host.dll

python eng/run-local-regression.py --mode run --models qwen3.5-0.8b --scenarios retrieval `
  --scenario-profiles $profileFile --model-root D:\TensorSharp-models `
  --server .\TensorSharp.Server.Host\bin\TensorSharp.Server.Host.dll `
  --allow-download --max-download-gib 2 --strict --output .\artifacts\retrieval-profile-run
```

该示例验证的是指定合成资料、检索结果与引用路径；不是所有文档格式、OCR、重排算法或权限隔离的通用质量证明。调用者应保留默认 2048 上下文；不要沿用小 CLI smoke 的 256-token 设置。

14 个扩展场景各自使用具体执行器和断言；支持边界以当前源码/schema/审计报告为准，不能因为写了 profile 就认定已覆盖：

| 场景 ID | 必须明确的组合和证据 |
|---|---|
| `embedding_media` | encoder、matching projector、图像/音频/视频夹具；有限向量、维度、重复性与批准的参考向量数值对照，不宣称通用语义质量 |
| `retrieval` | 检索/生成所需模型、文档和查询；排名、集合隔离、引用或原文身份。实际 Qwen3.5-0.8B `ggml_cuda` + MiniLM CPU 运行完成文档/来源检查后，在固定 384-token 输出预算下未正常结束，整体失败 |
| `image_generation` | 完整生成依赖、固定输入；解码后的尺寸、像素有效性及批准的质量指标 |
| `image_editing` | 原图、蒙版、编辑提示；输出解码、预期变化与保留区域约束 |
| `video_generation` | 全部专家/VAE/encoder、短视频参数；帧、尺寸、时长及有效画面 |
| `video_audio_generation` | 另含 audio VAE；音视频解码、声道、采样率、时长和有效音频 |
| `video_reference` | 已实现的 i2v/fl2v 等条件路径及匹配参考像素；不支持的参考模式明确阻塞 |
| `speculation` | 匹配 target/draft，固定历史；真正激活的 verify/accept 统计及输出对照，不由 n-gram 通过替代 |
| `lora` | 匹配 base/adapter；加载、修改输出、卸载恢复及错误组合拒绝 |
| `multi_gpu` | 明确的本机多设备、放置策略、逐 GPU 预算及单设备参考；单卡不能冒充多卡 |
| `distributed` | 明确授权的 worker/rank 拓扑和参考；不自动发现或联系远程机器 |
| `client_live` | 原生 IChatClient 两种模式与 HTTP SDK 四种模式的完整两轮回答对照；原生正常完成/usage/模型后端身份；不涵盖真实取消与 token 级一致性 |
| `native_unit` | 无模型的两个 direct CUDA 合成测试、精确 TRX 结果、实际图捕获/回放与零 CPU fallback；其他原生后端用 CTest，不能扩大通过范围 |
| `jev_media` | 有标签的视觉/视频输入，以及显式配置的 ASR 依赖；文本附件通过不替代媒体判断 |

### 媒体 profile 的具体输入约束

- `image_generation`：`parameters` 包括 `prompt,width,height,steps,cfg,seed`，可选 `negative_prompt`；尺寸为 32–512 且是 32 的倍数。`expected.pixel_sha256` 是经批准参考输出的**解码后 RGBA8 像素**哈希，不是 PNG 文件哈希。没有独立参考就不能据本次输出反填期望并宣布通过。
- `image_editing`：使用同类生成参数，并提供 `fixtures.input`、`fixtures.mask`；`expected` 要有 `pixel_sha256` 和正的 `min_changed_mean_abs_diff`。执行器使用固定 fidelity 工具，需要 NumPy/Pillow，检查未蒙版区域保留、选区变化、黑蒙版无操作和错误蒙版拒绝。
- `embedding_media`：`parameters` 为 `modality`（image/audio/video）、`text`、`dimensions`；`fixtures.media` 与 `fixtures.oracle` 都要固定哈希。oracle JSON 必须记录 `schema_version,model,projector,modality,media_sha256,text,dimensions,vector,provenance`。数值阈值不得宽于相对 L2 0.001 / cosine 0.999999；媒体限 2 MiB，PNG 不超过 512px，WAV 不超过 10 秒/48kHz/双声道，视频不超过 10 秒/32 帧/512px，视频路径需要 ffprobe/ffmpeg。
- 三个 video 场景使用固定 CLI、ffprobe、ffmpeg。`parameters` 必须含 `prompt,width,height,frames,fps,steps,cfg,seed,sampler`；可选 `mode`（t2v/i2v/fl2v/ref）、`audio` 和 `max_output_bytes`。帧数至多 121、时长至多 10 秒、尺寸至多 1024，同时必须满足模型自己的时间/空间网格。输出总量也有上限。`seed` 通过视频专用 `--diffusion-seed` 传入，并核对 CLI 完成记录中实际返回的 seed。
- 视频 oracle 是 `fixtures.reference_rgb`，表示完整 clip 的 RGB24 原始字节；`expected.max_rgb_rmse` 明确数值容差。参考输入可使用 `first_image`、`end_image`、`reference_image`，每张必须是 PNG/JPEG、至多 32 MiB，且头部声明的宽高分别不超过 1024；模型启动前检查尺寸，实际解码仍由生成路径验证。带音频时还必须有 interleaved little-endian float32 stereo 的 `reference_pcm`、`audio_samples_per_channel` 和 `max_audio_rmse`。这验证指定参考下的生成路径，不等于开放式语义质量。LTX profile 采用真实 `cpu/cuda/mlx` 执行器，不能用 GGML 别名冒充 GGML 内核覆盖。
- `jev_media` 使用 `fixtures.media` 和 `parameters.media_kind/state/questions`，`expected` 记录带媒体的 `decisions`、至少一个答案不同的 `control_decisions` 以及 `image_count`。媒体限 2 MiB；图片为可完整解码的 PNG 且各边不超过 512px；视频为 MP4/WebM、至多 10 秒/32 帧/512px，需要本地 ffprobe。image/video 需要匹配 projector 与积极的视觉就绪证据；未拥有本地 ASR 端点时，audio 分支明确未测试，不会转发到任意外部服务。

推荐先对同一文件做 `plan`，再为有明确资源和夹具的子集做 `run`，最后加入全目录运行。缺少设备/输入和没有有效 oracle 的组合必须显示为未测试或阻塞，不能把“脚本退出 0”“文件存在”当作场景通过。

### 推测、多 GPU、LoRA 与分布式

`speculation` 的参数为 `head_kind`（`dflash`、`pertoken` 或 `block`）和 `draft_tokens`（1–16），可声明 `models.draft`；期望必须要求 token 对照和实际接受统计。提供预构建 `--benchmark` 宿主。它不把 n-gram 通过算作学习型 draft 通过，也不从 head 名称推断所有 MTP/NextN 架构已覆盖。

文本 `lora` 提供 `fixtures.adapter`（safetensors，至多 512 MiB）和 `fixtures.prompt`（UTF-8，至多 16 KiB）；参数为 `adapter_name`、非零 `scale`（-8 至 8）及 `max_tokens`（1–128）。期望必须声明不同的 `base_text` 与 `adapted_text`，执行加载、应用、逐请求禁用和卸载恢复。当前只接入源码允许的 `mistral3`、`hunyuan-dense`、`nemotron_h` 系列，后端为 `cpu/ggml_cpu/ggml_cuda/cuda`，且限单设备；其他组合在准备模型前跳过。不覆盖图像/视频 LoRA。

`multi_gpu` 当前使用 `ggml_cuda` 和固定 `DeepSeek41DistributedProbe`，`parameters` 包括 `mode`（`tp` 或 `layer`）、明确的 `devices`、`chunk` 和 `decode_count`，夹具为有哈希的 `probe` 与 token JSON。逐卡预算使用 `per_gpu_required_gib` 对象，例如 `{"0":8,"1":8}`；实际需求必须由对应模型建立。必须证明完成多卡放置，并与新进程单卡控制比较所有约定的 prefill/decode/refill/reset 向量。机器还必须有运行单卡控制的资源。

`distributed` 是显式授权的 Linux SSH 配方，不自动发现或准备机器。profile 必须声明节点、用户、端口、`authorized:true`、每个模型分片/运行时/源码的远端绝对路径与 SHA-256、已有 probe/dotnet、peer endpoints，以及本地 `known_hosts` 固定夹具。执行前检查匹配身份、内存与逐卡预算；使用固定 supervisor、严格主机密钥和进程清理，不接受任意命令，不上传权重。未确认清理时停止后续任务。本轮没有远端/GPU 实测通过证明；不得据此联系未获授权的 Mac 或其他机器。

初始分布式实现要求节点间直接路由：每个 `peer_endpoints[i]` 的主机名/IP 必须与对应 `nodes[i].host` 相同，不做 DNS 别名映射，也不支持隧道或 ProxyJump。`runtime_files` 必须固定真实 `dotnet` 可执行文件、与本地控制一致的 probe DLL、probe 同目录全部 `TensorSharp*.dll` 及 `libGgmlOps.so`。SSH 使用所拥有的空配置、固定 `known_hosts` 和 agent 认证，不继承代理或 ControlMaster；不能照搬“公网 SSH 地址加不同内网 peer 地址”的配置。

每个 `nodes[]` 还必须提供唯一的 `expected_machine_id_sha256`，为 `/etc/machine-id` **原始文件字节（包含末尾换行）**的 64 位小写十六进制 SHA-256。用户应在每台明确授权的节点上手动只读执行 `sha256sum /etc/machine-id`，将结果写入 profile；运行器不自动发现节点、不剥除换行，也不使用 dbus 路径替代。远端 supervisor 在创建输出目录、资源预算检查、模型哈希或启动任何子进程之前验证该值；缺失或不匹配立即失败，不加载模型，控制端也会检查回执。不同地址指向同一操作系统实例不能被算作多个节点。machine-id 标识操作系统实例，不能证明容器/虚拟机之间有独立硬件资源；调用者仍须保证每个节点拥有单独分配且足够的资源。

### 原生客户端与设备单测 profile

`client_live` 使用 `fixtures.native_client`（`TensorSharp.NativeChat.dll`）和 `fixtures.http_client`（`TensorSharp.LocalChat.dll`），两者都必须保留完整构建输出目录并固定 SHA-256。`parameters.source_commit` 是声明的完整 40 位构建 revision；`expected` 固定为 `{"answers":["4","12"],"require_normal_finish":true}`，`models` 为 `{}`。支持原生示例实际实现的 `ggml_cpu/ggml_cuda/ggml_metal/ggml_vulkan`，需要预构建 `--server`、2048 上下文和完整主模型预算。

执行器顺序运行原生 stream/response 两个进程，每个做两轮；再启动所拥有的 loopback server，验证两个 HTTP SDK 的 stream/response 共四种模式。原生示例必须包含 `--regression-json`：回执由程序读取完整 `response.Text` 和实际加载身份生成，启动日志不能冒充回答。新参数只在显式请求时输出 JSON，不改变普通示例调用。这里校验语义答案和正常完成；原生 seed1、HTTP seed42，不能将结果称为精确 token/logit 一致性，也不证明取消恢复或缓存命中。

本地 Windows `ggml_cuda` + Qwen3.5-0.8B 已在干净运行器提交 `d93f4aaa5fede1a5cb084b14c01e7e49703fd921` 下重跑通过：两个原生模式和四个 HTTP SDK 模式，共 12 个完整回答、22 项 profile 断言，整次退出码为 0。证据位于本机 `D:/TensorSharp-regression-validation/artifacts/client-live-profile-d93f4aa/report.json`。六个客户端均正常退出，客户端与所拥有的 server 均验证无残留进程；server 在断言完成后被主动停止，其终止退出码不宣称为自然退出。

原生示例构建源码仍为 `f6e6e506206d62755efa08755c6d83ed19e249f8`，HTTP 示例仍为 `b764dec`；对应 SHA-256 为 `220757da421387082df93fdd732369975165c253ff094ecd4bee1a06e5959d98` 与 `5b2b8e673b91a78dd7b655c07e536720d907329a1ea4c2c74d245e78b9e45c6c`。报告声明的当前引擎 checkout 为 `1783d3b`，不能据此把所有保留的程序集都归为该提交的新构建。此结果只覆盖该模型、后端和固定两轮夹具，不代表所有客户端、取消行为或平台已通过。

`native_unit` 不准备、下载或加载主预设权重；`preset` 仅限定目录/后端范围。`fixtures.test_assembly` 固定为 `InferenceWeb.Tests.dll`，参数同样只有 `source_commit`。`expected.required_tests` 必须精确包含这两个 FQN：

```text
InferenceWeb.Tests.CudaBackendTests.CudaPrefillGraphCache_CaptureReplayAndAbort_BehaveCorrectly
InferenceWeb.Tests.CudaBackendTests.CudaQuantizedWeight_CanRunAfterHostCopyReleased
```

同时声明整数 `max_cpu_fallback_calls:0`。此 profile 仅接入 direct `cuda`；固定 `dotnet vstest` 入口不构建、不 restore，过滤掉 Bench/Models。两个测试都必须在 TRX 中 Passed，CUDA 图捕获/回放日志和完整的零 fallback 计数也必须存在。跳过或诊断缺失是未测试，失败计数和实际 CPU 回落是失败。测试 DLL 的 `.deps.json`、`.runtimeconfig.json` 与依赖库必须相邻。报告区分“调用者声明的构建源码 revision”和“已计算哈希的部署文件”；不能宣称编译输入或实际映射模块已经独立证明。

本地 Windows direct CUDA 已在同一干净运行器提交 `d93f4aaa5fede1a5cb084b14c01e7e49703fd921` 下通过上述两个设备合成测试和全部 7 项 profile 断言，退出码为 0；记录了实际图捕获/回放、两个完整的零 CPU fallback 计数块以及无残留进程的清理结果。测试程序集声明源码仍为 `b764dec2d149aabd725c067a47c6f4c5dfedd010`，SHA-256 为 `0b7a9e3bbdd4f8e39e0a1d39c7ce544335991f043f4ac501866ab73907816053`。部署候选包含 A0 原生库（`a0f0674576b5193222846db339456d86d089fffdcb9f105f64b6a85426d98128`），但候选哈希不证明它被实际映射或执行 GGML。完整身份与结果见本机 `D:/TensorSharp-regression-validation/artifacts/native-cuda-profile-d93f4aa/report.json`。

这两份干净运行器报告取代此前 `*-profile-f6e6e50-dirty` 报告作为相应执行器的最新验证；旧报告仍保留，既有程序集的真实构建来源不改变。两项 CUDA 测试通过不能代表全部原生算子、模型或所有引擎程序集已按运行器提交重建。

profile 结果使用 `profile::<id>` 单独编号。已经绑定模型预设的场景，按每个被选择的“预设/后端/场景”组合检查是否有精确匹配的 profile；缺失者逐项记录 `untested`。提供一个模型或设备的 profile 不会隐藏其他被选择组合的缺口。只有未绑定任何模型预设的场景类，在缺少 profile 时使用 `catalog::<scenario>` 范围的 `untested` 记录。普通聊天通过不能补足这些缺口。

## 8. 原生测试与完整组件通道

CTest 接口要求显式提供已构建目录，并确认其 CMake cache 指向本 checkout：

```powershell
python eng/run-local-regression.py --mode run --scenarios native_ctest `
  --native-test-build-dir .\TensorSharp.GGML.Native\build-windows `
  --native-test-backend cuda --output .\artifacts\native-cuda
```

构建时需要 `TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON`。运行器串行执行选定测试并保留逐条 JUnit 结果；名字/命令不能确定后端的测试保持未测试。`--native-test-filter` 只缩小名称范围，不绕过设备验证。选中范围之外的 fixture 不会自动拉起其他设备。每次执行使用独立临时和 fixture 目录；已有用户模型目录不参与原生合成 fixture 的自动解析。

原生数值/生命周期测试、托管组件测试、真实模型场景是不同层面的证据。CTest 的 CPU 通过不能证明 CUDA，多卡 skip 也不能算通过。

## 9. 报告、失败诊断和复现

每次运行在新目录中写入：

- `report.json`：机器可读结果、来源、资源、断言、状态、进程与复现参数。
- `report.md`：可读摘要，保留失败、跳过、未支持和未测试。
- `junit.xml`：供测试系统收集的结果。
- 各场景目录：有效配置、合成输入、经过清理的 stdout/stderr、模型/构建身份、必要诊断。
- `diagnostics.zip`：限定大小和文件类型的文本诊断包；不包含权重、进程内存或完整环境。大的 float logit 轨迹留在本地场景目录，不进入 ZIP。

使用其他 checkout 构建的宿主时加 `--engine-repo`，分别记录运行器源码和声明的引擎源码；同时保留二进制哈希。源码提交名、磁盘上的候选 DLL 和实际映射的模块不是同一个证据，报告不会自动把旧产物当成当前源码的新构建。

失败时先查看场景 `reason`、`stage`、请求/实际后端、期望与实际值、第一分歧位置、退出码、stdout/stderr 和清理结果。OOM、超时或未知进程清理状态不能算成功；清理未确认时会停止后续工作，避免带着未知 GPU/子进程继续跑下一模型。

从 `report.json` 的 `reproduce` **参数数组**取出复现命令，不要当作任意拼接的 shell 文本执行。复现会保留模型根目录、后端、预算和 profile 等选择，并使用新的输出目录；重复同一复现命令后，需要另选空目录。历史失败和后续定点通过都保留，后一次单场景通过不会改写原始整轮报告。

## 10. 状态与退出码

| 状态 | 含义 |
|---|---|
| `pass` | 本次实际执行满足该场景的全部断言 |
| `fail` | 模型准备或实际执行/断言失败 |
| `skip` | 选择了该组合，但硬件、预算、依赖或夹具不足 |
| `unsupported` | 明确的源码、配置或执行器支持边界 |
| `untested` | 仅计划、未运行、证据不足或尚无完整执行器/独立 oracle |

| 退出码 | 含义 |
|---|---|
| `0` | 审计/计划完成，或执行在其记录范围内通过；不能单看此值推导全覆盖 |
| `1` | 准备或断言失败 |
| `2` | 参数、配置或运行器错误 |
| `3` | 没有任何可执行测试通过，或 `--strict` 下仍有选择的 skip/untested |
| `130` | 用户取消，保留清理结果和已完成证据 |

`--strict` 不会仅因为显式 `unsupported` 就返回 3，所以必须同时审阅状态矩阵。全目录运行遇到配方、未验证硬件或缺少 profile，严格模式失败是诚实披露，而不是应该把这些状态改成通过。

## 11. 维护和合并前检查

### 当前验证快照与未完成项

`1783d3b` 已集成 14 个 typed profile 执行器，其中两项限定 profile 已用干净运行器 `d93f4aa` 重跑通过，但没有证明完整矩阵全绿或已经满足合并条件。下面的已完成运行保留其原始源码和二进制身份；后续定点通过不会改写原始全轮计数。报告名均指本机 `D:/TensorSharp-regression-validation/artifacts/` 下的忽略产物，不是已发布附件。

| 运行 | 已记录结果 | 尚未解决或明确的范围 |
|---|---|---|
| `native-cuda-f6e6e50` | 选中 61 项 CTest：47 通过、1 失败、13 跳过 | `minimax-h3-lora-cuda` 的 f32 base/f32 factors、guard off 参考断言失败，最大差值 `0.000628856`；跳过项仍未验证 |
| `native-cpu-5fa7ab0` | 原始 41 项 CTest：37 通过、2 失败、2 跳过 | `cpu-quantized-ffn-strip` 的 Q2_K 严格相等断言仍失败，最大绝对差值 `2.38418579e-07`；数值小不能抹掉断言失败 |
| `minimax-cpu-5b78c99` | 定点 MiniMax executor 重跑 1 项通过 | 仅修正测试对不同平台 allocation-error 文案的判断，不能据此把原始 CPU 全轮或另一个数值失败改成通过 |
| `native-cuda-profile-d93f4aa` | 干净运行器重跑：两项固定 CUDA 测试、7 项 profile 断言通过，退出 0 | 图捕获/回放、零 CPU fallback、清理确认；测试程序集仍为 `b764dec` 构建 |
| `client-live-profile-d93f4aa` | 干净运行器重跑：六个模式、12 个回答、22 项 profile 断言通过，退出 0 | Windows `ggml_cuda` + Qwen3.5-0.8B Q8_0；原生 `f6e6e50` / HTTP `b764dec` 构建身份保留 |
| `grounded-retrieval-profile-d93f4aa` | 干净运行器；profile 失败，退出 1 | Qwen3.5-0.8B `ggml_cuda` + MiniLM CPU 完成文档/来源检查，随后未满足固定 384-token 输出预算下的正常完成契约；Windows Job 清理确认无残留 |
| `minilm-resumed-run-d93f4aa` | CPU 嵌入场景跳过，退出 3，没有额外传输 | 2 GiB RAM 预算低于 `2178743728` 字节的需求估计；这是资源门禁，不是推理失败 |
| `minilm-resumed-run-d93f4aa-ram4` | CPU 嵌入场景及 19 项断言通过，退出 0 | 4 GiB RAM 预算允许运行；从真实断点恢复下载并校验完整文件。代码为 `d93f4aa`，报告明确披露仅文档工作区修改，不能称为干净 checkout |

检索失败按原始预算与 oracle 保留，不通过增大输出预算掩盖。回执证明生成没有正常完成，尚不足以进一步归因为引擎缺陷或模型/夹具行为。导入、索引和原文身份检查通过不代表完整 profile 通过，也不宣称后续作用域对照已经执行。

`minilm-resume-d93f4aa/controlled-cancellation.json` 记录真实下载在 `2097152` 字节处取消、保留 partial，且最终文件尚未安装。后续被允许的运行恢复剩余 `22910912` 字节；`0.022 GiB` 网络上限低于完整文件的 `25008064` 字节。最终 SHA-256 为 `263215c3cadd6e16740741a7624ab4cbb6c8e777688bd5331ecfbf5681c2f8ed`。下载管理器没有修改；两个 MiniLM 报告均记录运行器 `d93f4aa` 加文档修改。最终 CPU 模型身份、嵌入断言、类型化拒绝响应和所拥有 server 的清理均通过。这只验证该 checkpoint 的恢复与 CPU 路径，不代表所有模型下载都已经实测。

原始 `components-b764dec/report.json` 中 Knowledge 110 项及两个无模型客户端契约通道通过。Server 通道失败并超时：其中 36 项失败可归因于把模型宿主环境设置施加到了无模型套件。`1783d3b` 已修正环境策略，但此快照中的重建/重跑结果仍待完成；独立的 FileLogger `Flush` 失败和整轮超时尚未证明修复。Studio 两项缺少 ffmpeg 的测试已调整为先检查跳过前提再操作目录，其重建后实测也待完成。源码修正不能直接登记为测试通过。

剩余数值失败、检索生成失败、组件重跑以及仍在进行的完整 Python 测试和最终 CI 结果必须完成评估，才能判断合并就绪；不得把未完成项算成通过。本轮仍没有 Linux/WSL GPU、macOS/Metal/MLX、Vulkan、多 GPU 或授权远端集群的实测通过证明；其他 typed profile 场景还需要批准的夹具、预算和实际模型报告。

### 运行器自身与配置维护

不运行模型即可检查运行器自身：

```powershell
python -m unittest discover -s eng/regression/tests -p 'test_*.py' -v
python eng/run-local-regression.py --mode audit
python eng/run-local-regression.py --mode plan --backend all
```

新增架构、模型、量化或场景时，同时更新不可变文件身份、所有必要分片、后端限制、typed profile 契约、资源需求和真正能发现错误的断言。测试参数必须匹配实际 CLI/API；包含 recipe/profile 状态不等于已经进行了实测。

合并前应分别记录：运行器单测/审计、原生/托管组件、真实模型、不同平台、剩余失败和未测试范围。当前提交未执行过的场景不能用旧版本通过替代。忽略的 `artifacts/` 或 `docs/validation/` 保存机器证据；可复用工具和批准夹具保留在源码目录。运行器不会发布诊断包；只有显式授权的分布式 profile 才会联系其中列出的机器。共享前由用户审阅本地报告。
