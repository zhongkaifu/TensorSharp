# 快速上手指南

[English](getting_started.md) | [中文](getting_started_zh-cn.md)

> [TensorSharp](../README_zh-cn.md) 文档的一部分。README 的[快速开始](../README_zh-cn.md#快速开始)是本页的精简版。

## 安装与首次运行

**使用 TensorAgent 桌面应用：** 打开[最新发布](https://github.com/zhongkaifu/TensorSharp/releases/latest)，展开 **Assets**，选择 `tensoragent-desktop-<version>-osx-arm64.dmg`（Apple Silicon，macOS 14+），或 `tensoragent-desktop-<version>-win-x64-cpu.msi` / `win-x64-cuda.msi`（Windows x64）。也提供 PKG 与 ZIP。桌面包包含 .NET 运行库；Windows 缺少 WebView2 时需另外安装。权重通过 **☰ → 模型 → 下载 → 使用** 下载，内置模型最低需要 12 GB 系统内存。安装、校验、首次聊天、工具与故障排查详见[桌面版用户指南](tensoragent_desktop_zh-cn.md)。历史发布可能没有桌面资源，需等待使用更新后 Release Binaries 工作流的发布。

**使用 TensorSharp CLI / 服务端：** [Releases 页面](https://github.com/zhongkaifu/TensorSharp/releases)提供自包含的 Windows x64（CPU/CUDA）、Linux x64（CPU/CUDA）与 macOS arm64 归档。下文的 SDK 与构建步骤用于源码构建。

**NVIDIA DGX Spark / GB10：** 请使用独立的实验性 **CUDA 13、Linux ARM64**
[Docker 构建与归档说明](../DEVELOPMENT_zh-cn.md#gb10--dgx-spark-构建容器实验性)。
归档后缀为 `linux-arm64-cuda13-GB10`，仅面向单个 GB10，而非通用 ARM64 GPU。
打标签的正式发布也会附带这些归档，它们在托管的无 GPU ARM64 runner 上通过 Docker 构建。
CLI 与服务端文本推理曾在真实硬件上做过一次历史性检查，但早于上游重新集成，不能证明当前代码；
托管 CI 只重新检查 CPU 与归档路径。现有 x64 CUDA 归档不适用于 Spark。

源码构建面向 .NET 10。全新开发机器需要安装完整的 **.NET 10 SDK**；只安装 .NET Runtime 无法构建 TensorSharp：

| 平台 | 安装 SDK |
|---|---|
| **Windows** | 在 PowerShell 中运行 `winget install Microsoft.DotNet.SDK.10`，或参阅 Microsoft 的 [Windows 安装说明](https://learn.microsoft.com/zh-cn/dotnet/core/install/windows)。 |
| **macOS** | 使用 [.NET 10 SDK 安装程序](https://dotnet.microsoft.com/zh-cn/download/dotnet/10.0)：Apple 芯片选择 **Arm64**，Intel Mac 选择 **x64**。另见 Microsoft 的 [macOS 安装说明](https://learn.microsoft.com/zh-cn/dotnet/core/install/macos)。 |
| **Linux** | 按照 Microsoft 的 [Linux 发行版指南](https://learn.microsoft.com/zh-cn/dotnet/core/install/linux)为当前发行版配置正确的软件源，并安装其 .NET 10 SDK 包（通常名为 `dotnet-sdk-10.0`）。 |

安装后打开新终端，确认列表中包含 `10.0.x` SDK：

```bash
dotnet --list-sdks
```

更多细节见 [.NET 跨平台安装概览](https://learn.microsoft.com/zh-cn/dotnet/core/install/)或[开发 → 前置要求](../DEVELOPMENT_zh-cn.md#前置要求)。

然后即可在已验证的原生 GGML 快速路径（Gemma 4 E4B）上约 30 秒跑起来。其他前置包括 `git`、`curl`、[CMake](https://cmake.org/download/) 3.20+（原生 GGML 库由它来配置和构建；Windows 上 Visual Studio 的“C++ CMake tools for Windows”组件自带一份，构建脚本会自动找到），以及所选 GPU 后端的工具链（见 [开发 → 前置要求](../DEVELOPMENT_zh-cn.md#前置要求)）。推荐的公开文件是 [`gemma-4-E4B-it-Q8_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q8_0.gguf)（7.48 GiB）；纯文本推理无需投影器。

**Windows + NVIDIA（PowerShell）**

```powershell
git clone https://github.com/zhongkaifu/TensorSharp.git; Set-Location TensorSharp
New-Item -ItemType Directory -Force models | Out-Null
curl.exe -L --fail "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" -o models\gemma-4-E4B-it-Q8_0.gguf
'用一句话回答：TensorSharp 是什么？' | Set-Content prompt.txt
$env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA = 'ON'
dotnet run --project TensorSharp.Cli -c Release -p:TensorSharpSkipMlxNative=true -- --model models\gemma-4-E4B-it-Q8_0.gguf --input prompt.txt --max-tokens 128 --backend ggml_cuda
```

**macOS（Apple Silicon）** —— 去掉 CUDA 环境变量，使用 `--backend ggml_metal`。

**Linux + NVIDIA** —— 在 `dotnet run` 前加 `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON`，使用 `--backend ggml_cuda`。

**AMD / Intel / NVIDIA Vulkan** —— 设置 `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON`，使用 `--backend ggml_vulkan`。

**Linux（Ubuntu）+ 多张 NVIDIA GPU —— 张量并行**

张量并行把一个模型切分到 N 张 GPU 上，可运行在 Direct `cuda` 后端以及 GGML CUDA /
Vulkan 后端（`--backend ggml_cuda`、`ggml_vulkan`）。Qwen 3.8 Flash Next 与
DeepSeek V4 / V4.1 的按层切分改用独立的 `--layer-split N` 参数：每张 GPU 拿一段连续的完整层。
`--tp N` 仅表示张量并行，两种模式互斥；不支持的请求会在启动时失败。
按层切分仅支持单节点，不能与 `--tp-node-id` / `--tp-peers` 组合。
现有按层切分命令需将 `--tp N` 改为 `--layer-split N`，或设置
`TENSORSHARP_LAYER_SPLIT_DEGREE=N`。未配置两种模式时默认使用单设备。GLM 5.x 的 `--layer-split N` 选择整层放置，
GGML GPU 后端上的 `--tp N` 则选择原生本地张量并行路径。
对 Qwen-Image-2.1，`--tp N` 只切分扩散 Transformer（DiT），文本 / 视觉编码器与 VAE 留在第一张 GPU 上。请先安装 CUDA 工具包，然后：

```bash
# 在 RunPod 的 Ubuntu 24.04 镜像上，需要先让动态链接器找到 CUDA 兼容库：
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/compat:$LD_LIBRARY_PATH
# 较旧的 Ubuntu 版本需要从 backports PPA 安装 .NET 10 SDK：
add-apt-repository ppa:dotnet/backports

apt update && apt install dotnet-sdk-10.0
git clone https://github.com/zhongkaifu/TensorSharp.git
cd TensorSharp
mkdir models
wget "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" -O models/gemma-4-E4B-it-Q8_0.gguf
bash TensorSharp.GGML.Native/build-linux.sh
dotnet build -c Release

# 单进程内使用 2 张 GPU
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf \
    --backend cuda --interactive --max-tokens 20000 --tp 2

# 同样的用法也适用于 GGML CUDA 后端（可加 TENSORSHARP_TP_DEVICES=0,2 指定 GPU）
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf \
    --backend ggml_cuda --interactive --max-tokens 20000 --tp 2
```

只需再加上节点 ID 与共享的 peer 列表，同一个模型就能跨机器扩展 —— 2 节点 × 2 GPU 即全局 TP 度为 4：

```bash
# 节点 0
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf --backend cuda --tp 2 \
    --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
# 节点 1（peer 列表相同，节点 ID 不同）
TensorSharp.Cli/bin/TensorSharp.Cli --model models/gemma-4-E4B-it-Q8_0.gguf --backend cuda --tp 2 \
    --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
```

`TensorSharp.Server.Host` 支持同样的 `--tp`、`--tp-node-id`、`--tp-peers` 参数（也可用
`TENSORSHARP_TP_*` 环境变量）；在多节点集群中，服务端必须是节点 `0`（对外提供 HTTP
的 driver），其余节点各运行一个 `TensorSharp.Cli` worker。完整参考：**[张量并行与分布式推理](../USAGE_zh-cn.md#张量并行与分布式推理)**。

将同一模型作为服务托管（浏览器 UI 在 <http://localhost:5000>，另有 Ollama/OpenAI API）：

```bash
dotnet run --project TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --max-tokens 512
```

> 服务端默认绑定 `0.0.0.0:5000`（可用 `--port` / `--host` 或 `PORT` / `HOST` 环境变量修改；macOS 上 5000 端口已被 AirPlay 接收器占用），无内置鉴权或 TLS——请置于防火墙之后，或使用带鉴权的 HTTPS 反向代理。图像/视频/音频需追加伴随文件 [`mmproj-gemma-4-E4B-it-Q8_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/mmproj-gemma-4-E4B-it-Q8_0.gguf)，用 `--mmproj` 指定。

TensorSharp.Server.Host、TensorSharp.Cli 与 TensorAgent 默认对[支持的模型](supported_models_zh-cn.md)各表中的所有自回归家族使用共享引擎的 Radix
KV 前缀缓存（DiffusionGemma 与图像 / 视频模型除外）。它复用公共提示前缀和每个会话的私有状态，并遵守模型与媒体边界；
投机解码（`--spec`）不会关闭它。设置 `TS_SCHED_PREFIX_CACHE=0` 可关闭运行时前缀复用。服务端与 CLI 的 `--no-prefix-cache`
同样会关闭前缀复用和启动预热；在服务端还会关闭磁盘上的前缀检查点持久化。

两个可执行程序在不带参数或使用 `--help` 启动时，都会打印完整的参数参考——逐项列出说明、默认值、取值范围与示例：

```bash
dotnet run --project TensorSharp.Cli -c Release -- --help
dotnet run --project TensorSharp.Server.Host -c Release -- --help
```

完整命令参考：**[CLI](../USAGE_zh-cn.md#控制台应用)** · **[Server](../USAGE_zh-cn.md#web-应用)** · 更多可下载模型：**[模型下载](../MODEL_DOWNLOADS_zh-cn.md)** · 想用配置文件？**[config/](../config/README.md)**。

## 从同一解决方案构建 TensorAgent

`dotnet build TensorSharp.slnx -c Release` 构建引擎、CLI/服务端以及 TensorAgent.Core、TensorAgent.Sharing 和 TensorAgent.Tests。在 macOS 上还会尝试构建 Mac 应用，并在 Apple Silicon 上构建 iOS 模拟器应用；在 Windows 上会尝试构建 `win-x64` 应用。应用构建需要对应的 MAUI 工作负载与原生依赖；缺失时会警告并跳过对应应用头，其他构建错误会使解决方案构建失败。Linux 仅构建共享项目，不构建 MAUI 应用头。可用 `-p:TensorSharpSkipTensorAgentApp=true` 显式排除应用。

构建测试项目不会执行测试。Windows 应用已有有限的 Debug/Release 聊天、合成图像与工具验证；媒体生成和更广的设备覆盖尚未验证。当前覆盖范围与应用专用脚本见 [TensorAgent 源码 / 验证指南](../TensorAgent/README.md)；安装 Mac / Windows 发布包请看[桌面版用户指南](tensoragent_desktop_zh-cn.md)。iPhone / iPad 仍需源码构建流程。

## 文本与代码嵌入

当前源码支持 **Snowflake Arctic Embed L v2.0** 与 **all-MiniLM-L6-v2** 的 GGUF 编码器，通过 OpenAI `/v1/embeddings` 和 Ollama `/api/embed` 提供归一化向量。以下命令在完成上面的源码构建后启动小型 MiniLM 服务：

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

使用 `cpu` 运行 100% 纯 C# 推理，无需原生推理库；或选择原生 GGML 的 `ggml_cpu`、`ggml_metal`、`ggml_cuda`。聊天与嵌入服务分别运行。完整的 Snowflake 下载、批处理、维数缩减、C# API、分词与性能验证见[嵌入指南](embeddings_zh-cn.md)。

## 新视频、嵌入与技能工作流

- [LTX 2.5 Distilled](models/ltx-2.5_zh-cn.md)：五文件预设共 44.76 GB，需要获批发布方访问权限。实验性 48 GB 档位默认使用 1024×1024×17 帧与 8+3 步两阶段配方，联合生成 48 kHz 立体声音频。
- [MiniMax-H3](models/minimax-h3_zh-cn.md)：首帧动画选择 FL2VA，身份参考选择 Ref2VA。二者均生成 32 kHz 立体声音频；当前精度与基准说明取代旧速度结论。
- [EmbeddingGemma 2](embeddings_zh-cn.md)：使用 `--embeddings`，媒体输入另加 `--mmproj`。已实现六个后端，任务前缀、维数与测试限制随模型而异。
- [Agent Skills](agent_skills.md)（英文）：桌面 CLI / 服务端随附 17 个技能。`find-skills` 可达且开启 `--skills-allow-network` 时，原生工具可搜索、检查、按固定版本安装并在同一请求中使用；运行脚本仍需执行权限与依赖。

## 将文件复用为知识库

`v2026.10.11` 源码已包含个人知识集合与回答调整；LTX 2.5、EmbeddingGemma 2 与原生技能发现需要更新的源码，选择发布前请查看[项目状态](PROJECT_STATUS_zh-cn.md)。在 TensorAgent 打开 **☰ → 知识库**，创建集合后上传文件或文件夹，或通过 **添加聊天** 导入已保存对话、可用附件和生成文件的快照。检查每个来源的抽取状态，再选择 **在此聊天中使用** 后提问；上传本身不会选择集合。回答中的引用可打开保留的原文件及相关片段。

知识库支持文本、代码、数据集、部分 Office 内容和本地 PDF OCR。图像与视频采样帧需要已加载的兼容视觉模型；音频可使用已加载的音频塔或已配置的 `TS_JEV_TRANSCRIPTION_URL` 转录伴随服务。能力变化后可对原文件 **重新索引**。默认使用 BM25 检索，可选 GGUF 嵌入模型提供语义检索。覆盖范围、限制与引用详见[知识库指南](knowledge-system_zh-cn.md)。

独立服务通过 `TS_KNOWLEDGE_DIRECTORY` 启用同一存储：

```bash
export TS_KNOWLEDGE_DIRECTORY="$HOME/TensorSharpKnowledge"
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_metal \
  --host 127.0.0.1 --port 5001
```

请使用主机上可用的后端。通过 `/api/knowledge` 管理来源，并在支持的聊天请求中以 `knowledgeCollections` 传入集合 ID；见 [API 示例](../TensorSharp.Server.Host/API_EXAMPLES_zh-cn.md#个人知识库)。独立服务的浏览器界面支持回答调整，TensorAgent 则提供集合管理界面。CLI、TensorAgent 或服务端浏览器界面正在推理或回答时，可发送新指令中断并继续；**停止** 或 CLI 的 Ctrl+C 可单独中止。重定向的 CLI 输入仍按顺序完成各轮。

## 选择后端

后端支持取决于模型架构。嵌入模型支持纯 C# `cpu` 与原生 `ggml_cpu`、`ggml_metal`、`ggml_cuda`；EmbeddingGemma 2 还支持直接 `cuda` 与 `mlx`；其他模型的限制见[状态矩阵](PROJECT_STATUS_zh-cn.md#状态矩阵)。

| 你的硬件 | 推荐后端 | 标志 | 说明 |
|---|---|---|---|
| **Apple Silicon（Mac）** | GGML Metal | `--backend ggml_metal` | 服务端在 macOS 上的默认后端；CLI 在所有系统上默认 `ggml_cpu`，使用 CLI 时需显式传入该参数。`--backend mlx` 是另一条 Apple Silicon GPU 路径。 |
| **Windows / Linux + NVIDIA GPU** | GGML CUDA | `--backend ggml_cuda` | 测试最充分的 NVIDIA 路径。`--backend cuda` 是用于实验的 Direct PTX/cuBLAS 后端。 |
| **Windows / Linux + AMD / Intel / NVIDIA GPU** | GGML Vulkan | `--backend ggml_vulkan` | 与厂商无关的 GPU 路径（ggml-vulkan）。机器有 Vulkan 运行时即自动构建；用 `--no-vulkan` 退出。 |
| **无 GPU / 可移植 / 调试** | 纯 C# CPU | `--backend cpu` | 无原生依赖；matmul 跑在多核工作线程池上。连 DeepSeek V4.1 Flash 在这里也有一整套整模型执行器——它跑在纯 C# 执行器 `DeepSeek4CpuExecutor` 上，不用 ggml、不用 GPU，并在五层 F32 fixture 上以 `atol=rtol=2e-5` 对齐 PyTorch 参照实现 `eng/dsv41-reference.py`（这是与参照实现的架构级一致，而非真实 Q2_K 权重上的对齐）；它是正确性与可移植性路径，而非服务路径。需要更快的 CPU 推理可用 `--backend ggml_cpu`（原生算子）。 |

每个后端的完整说明见 [使用方法 → 计算后端](../USAGE_zh-cn.md#计算后端)。

## 让它跑得更快

按这个顺序选择：

1. **先选对 checkpoint。** Wan 视频优先使用 Turbo/Lightning/4-step 蒸馏 GGUF。Qwen-Image-2.1 可加上 `config/lora/` 中的步数蒸馏 [LoRA 插件](models/qwenimage21_zh-cn.md#lora-插件)（4–8 步，取代默认的 40 步）。
2. **使用匹配的后端。** NVIDIA：`ggml_cuda`；Apple Silicon 和 iOS：`ggml_metal`；CPU：`ggml_cpu`（需要可移植性时使用纯托管 `cpu`）。
3. **先减少工作量，再调参数。** H3 使用 `--cfg 1.0`，想少跑几步时在 FL2VA 上加 [Turbo 插件](models/minimax-h3_zh-cn.md#turbo-插件)（8 或 4 步），而不是调低基础步数；媒体任务优先降低分辨率、帧数或步数。
4. **最后再扩展或投机。** 根据模型和负载尝试 `--draft-model` / `--spec`、`--n-cpu-moe` 或 `--tp N`。

详见[性能指南与快速路径](PROJECT_STATUS_zh-cn.md#让它跑得更快)、[模型卡片](models/README_zh-cn.md)和[环境变量功能矩阵](env_var_feature_matrix_zh-cn.md)。
