# TensorSharp

<p align="center">
  <img src="imgs/banner_1.png" alt="TensorSharp logo" width="320">
</p>

[English](README.md) | [中文](README_zh-cn.md)

**面向 GGUF 模型的原生 .NET LLM 推理引擎** —— 覆盖自回归 LLM *与* DiffusionGemma 风格的文本扩散模型，以及 Qwen-Image-Edit 图像编辑、MiniMax-H3 视频 + 原生 32 kHz 立体声音频联合生成（Wan 2.1/2.2 则只生成视频）。提供控制台应用、浏览器聊天界面，以及兼容 Ollama/OpenAI 的 HTTP API。一个纯 .NET 引擎，在相同 GGUF 文件与相同 GPU 上与手工优化的 C++ `llama.cpp` 互有胜负。可选的 `TensorSharp.AgentHost` 层还提供 Agent Skills，以及用于沙箱化文件和 shell 操作的、有界进程内“模型→工具”循环。

## 《From Tensors to Tokens》—— TensorSharp 实战书籍

<p align="center">
  <a href="https://www.amazon.com/dp/B0H9P44QZZ">
    <img src="website/assets/from-tensors-to-tokens-cover.jpg" alt="From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B" width="220">
  </a>
</p>

Zhongkai Fu 所著的 **[From Tensors to Tokens: Building a Multimodal LLM Inference Engine from Scratch with TensorSharp and Gemma 4 E4B](https://www.amazon.com/dp/B0H9P44QZZ)** 将本仓库串成一条端到端的学习路径。全书以 Gemma 4 E4B 为示例，连接张量基础、模型执行、多模态输入，以及一个可运行 LLM 推理引擎的应用接口。

**[查看书籍介绍与仓库伴读路线](docs/BOOK_zh-cn.md)** · **[在 Amazon 购买平装本](https://www.amazon.com/dp/B0H9P44QZZ)**

## 亮点功能

- **本地原生 .NET 推理。** 可通过 CLI、浏览器 Web UI，以及兼容 Ollama/OpenAI 的 API 运行 GGUF 文本与多模态模型。
- **模型与媒体覆盖广。** 当前源码支持现代文本模型、视觉/音频输入、PDF、图像编辑和视频生成；详见[模型卡片](docs/models/README_zh-cn.md)。
- **关键路径速度有竞争力。** 在相同模型与硬件上，TensorSharp 与 `llama.cpp` 互有胜负，并提供原生 GGML、CUDA、Vulkan、Metal、MLX 与纯 C# CPU 路径；详见[性能报告](docs/engine_comparison_report.md)。
- **智能体能力覆盖 iOS。** `TensorSharp.AgentHost` 提供有界的 Agent Skills 与代码工具；[TensorAgent](TensorAgent/README.md) 使用 iOS 的 `ggml_metal` 后端，把同一套本地聊天与智能体体验带到 iPhone 和 iPad。
- **可扩展的工程能力。** 连续批处理、分页/前缀共享 KV 缓存、投机解码、张量并行和可配置的安全边界，按需启用。详见[功能说明](FEATURES_zh-cn.md)、[使用指南](USAGE_zh-cn.md)与[当前状态](docs/PROJECT_STATUS_zh-cn.md)。

详细实现说明和历史性能数据已移到链接文档，让本页保持清晰、适合作为入口。

## 快速开始

更愿意使用预构建应用？[Releases 页面](https://github.com/zhongkaifu/TensorSharp/releases)提供自包含的 Windows x64（CPU/CUDA）、Linux x64（CPU/CUDA）与 macOS arm64 CLI / Server 归档。

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

更多细节见 [.NET 跨平台安装概览](https://learn.microsoft.com/zh-cn/dotnet/core/install/)或[开发 → 前置要求](DEVELOPMENT_zh-cn.md#前置要求)。

然后即可在已验证的原生 GGML 快速路径（Gemma 4 E4B）上约 30 秒跑起来。其他前置包括 `git`、`curl`、[CMake](https://cmake.org/download/) 3.20+（原生 GGML 库由它来配置和构建；Windows 上 Visual Studio 的“C++ CMake tools for Windows”组件自带一份，构建脚本会自动找到），以及所选 GPU 后端的工具链（见 [开发 → 前置要求](DEVELOPMENT_zh-cn.md#前置要求)）。推荐的公开文件是 [`gemma-4-E4B-it-Q8_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/gemma-4-E4B-it-Q8_0.gguf)（7.48 GiB）；纯文本推理无需投影器。

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
DeepSeek V4 会把同一参数用于按层切分：每张 GPU 拿一段连续的完整层。GLM 5.x
不传参数时也默认按层切分，而 GGML GPU 后端上的 `--tp N` 会选择其原生本地张量并行路径。请先安装 CUDA 工具包，然后：

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

`TensorSharp.Server` 支持同样的 `--tp`、`--tp-node-id`、`--tp-peers` 参数（也可用
`TENSORSHARP_TP_*` 环境变量）；在多节点集群中，服务端必须是节点 `0`（对外提供 HTTP
的 driver），其余节点各运行一个 `TensorSharp.Cli` worker。完整参考：**[张量并行与分布式推理](USAGE_zh-cn.md#张量并行与分布式推理)**。

将同一模型作为服务托管（浏览器 UI 在 <http://localhost:5000>，另有 Ollama/OpenAI API）：

```bash
dotnet run --project TensorSharp.Server.Host -c Release -p:TensorSharpSkipMlxNative=true -- --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --max-tokens 512
```

> 服务端默认绑定 `0.0.0.0:5000`（可用 `--port` / `--host` 或 `PORT` / `HOST` 环境变量修改；macOS 上 5000 端口已被 AirPlay 接收器占用），无内置鉴权或 TLS——请置于防火墙之后，或使用带鉴权的 HTTPS 反向代理。图像/视频/音频需追加伴随文件 [`mmproj-gemma-4-E4B-it-Q8_0.gguf`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/blob/main/mmproj-gemma-4-E4B-it-Q8_0.gguf)，用 `--mmproj` 指定。

两个可执行程序在不带参数或使用 `--help` 启动时，都会打印完整的参数参考——逐项列出说明、默认值、取值范围与示例：

```bash
dotnet run --project TensorSharp.Cli -c Release -- --help
dotnet run --project TensorSharp.Server.Host -c Release -- --help
```

完整命令参考：**[CLI](USAGE_zh-cn.md#控制台应用)** · **[Server](USAGE_zh-cn.md#web-应用)** · 更多可下载模型：**[模型下载](MODEL_DOWNLOADS_zh-cn.md)** · 想用配置文件？**[config/](config/README.md)**。

## 选择后端

每个后端对尚未实现的算子都会回退到 CPU，因此所有后端的输出都正确。

| 你的硬件 | 推荐后端 | 标志 | 说明 |
|---|---|---|---|
| **Apple Silicon（Mac）** | GGML Metal | `--backend ggml_metal` | macOS 默认。`--backend mlx` 是另一条 Apple Silicon GPU 路径。 |
| **Windows / Linux + NVIDIA GPU** | GGML CUDA | `--backend ggml_cuda` | 测试最充分的 NVIDIA 路径。`--backend cuda` 是用于实验的 Direct PTX/cuBLAS 后端。 |
| **Windows / Linux + AMD / Intel / NVIDIA GPU** | GGML Vulkan | `--backend ggml_vulkan` | 与厂商无关的 GPU 路径（ggml-vulkan）。机器有 Vulkan 运行时即自动构建；用 `--no-vulkan` 退出。 |
| **无 GPU / 可移植 / 调试** | 纯 C# CPU | `--backend cpu` | 无原生依赖；matmul 跑在多核工作线程池上。连 DeepSeek V4.1 Flash 在这里也有一整套整模型执行器——它跑在纯 C# 执行器 `DeepSeek4CpuExecutor` 上，不用 ggml、不用 GPU，并在五层 F32 fixture 上以 `atol=rtol=2e-5` 对齐 PyTorch 参照实现 `eng/dsv41-reference.py`（这是与参照实现的架构级一致，而非真实 Q2_K 权重上的对齐）；它是正确性与可移植性路径，而非服务路径。需要更快的 CPU 推理可用 `--backend ggml_cpu`（原生算子）。 |

每个后端的完整说明见 [使用方法 → 计算后端](USAGE_zh-cn.md#计算后端)。

## 已验证模型

以下架构均已实现，并由测试 / 基准矩阵覆盖。请选择适配你硬件的量化（低内存用 Q4_K_M、更高质量用 Q8_0）。更多尺寸与投影器文件见 [模型下载](MODEL_DOWNLOADS_zh-cn.md)。

| 家族 | 示例模型（GGUF） | 图像 / 视频 / 音频 | 思维链 | 工具 | 卡片 |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | [DeepSeek-V4.1-Flash](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/8e0c4de3cb6519bfc11ed69dc87184b457a57bb5)（Q2_K 或 Q4_K_M 分片 + 预处理 Engram sidecar，服务路径为 `ggml_cuda`；`ggml_cpu` 是仍能加载视觉伴随文件的正确性与可移植性路径，`cuda` 与纯 C# `cpu` 执行器则是仅文本的） | ✅（视觉伴随文件） / ✅（视觉伴随文件） / — | ✅ | ✅ | [deepseek41](docs/models/deepseek41_zh-cn.md) |
| DeepSeek V4 Flash | [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)（284B MoE，分片 GGUF） | — / — / — | ✅ | ✅ | [deepseek4](docs/models/deepseek4_zh-cn.md) |
| GLM 5.x | [GLM-5.2](https://huggingface.co/unsloth/GLM-5.2-GGUF)（744B-A40B MoE，分片 GGUF）、[GLM-5.3](https://huggingface.co/unsloth/GLM-5.3-GGUF)（256 个路由专家，仅文本；每个量化档一个子目录，UD-Q2_K_XL 为 7 个分片、236.4 GiB——`--model` 指向 `-00001-of-00007` 那一片）、[GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)（320B MoE，分片 GGUF，+ mmproj） | ✅（仅 5.3-Flash；5.2 与 5.3 均仅文本） / — / — | ✅ | ✅ | [glm](docs/models/glm_zh-cn.md) |
| Qwen 3.8 Flash Next | [Qwen3.8-Flash-Next](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)（GDN + 注意力混合 MoE，512 专家，分片 GGUF，+ mmproj） | ✅ / — / — | ✅ | 否（无解析器） | [qwen38-flash-next](docs/models/qwen38-flash-next_zh-cn.md) |
| Gemma 4 | [gemma-4-E4B-it](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)（另有 31B、26B-A4B MoE） | ✅ / ✅ / ✅ | ✅ | ✅ | [gemma4](docs/models/gemma4_zh-cn.md) |
| Qwen 3.5 / 3.6 | [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)（另有 35B-A3B MoE） | ✅ / — / — | ✅ | ✅ | [qwen35](docs/models/qwen35_zh-cn.md) |
| Bonsai Q1_0 | 本地哈希钉住的 `Bonsai-8B-Q1_0.gguf`（稠密 Qwen 3）与 `Bonsai-27B-Q1_0.gguf`（稠密 Qwen 3.5 混合）；所给 GGUF 未声明发布方 URL 或 license | — / — / — | 8B：不支持（固定空块）；27B：✅ | ✅ | [bonsai](docs/models/bonsai_zh-cn.md) |
| GPT OSS | [gpt-oss-20b](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF)（MoE） | — / — / — | ✅ | ✅ | [gptoss](docs/models/gptoss_zh-cn.md) |
| Nemotron-H | [Nemotron-H-8B](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF)（另有 47B、Omni） | ✅（Omni） / — / — | ✅ | ✅ | [nemotron](docs/models/nemotron_zh-cn.md) |
| Mistral 3 | [Mistral-Small-3.1-24B](https://huggingface.co/bartowski/mistralai_Mistral-Small-3.1-24B-Instruct-2503-GGUF) | ✅ / — / — | — | — | [mistral3](docs/models/mistral3_zh-cn.md) |
| Hunyuan Dense | 腾讯稠密 Hunyuan GGUF（`hunyuan-dense`），例如 Hy-MT2 系列 | — / — / — | — | — | [hunyuan-dense](docs/models/hunyuan-dense_zh-cn.md) |
| DiffusionGemma | [diffusiongemma-26B-A4B-it](https://huggingface.co/unsloth/diffusiongemma-26B-A4B-it-GGUF) | — / — / — | — | — | [diffusiongemma](docs/models/diffusiongemma_zh-cn.md) |
| Muse-Glimmer | [Muse-Glimmer-30B](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF)（+ mmproj） | ✅ / — / — | ✅ | ✅ | [muse-glimmer](docs/models/muse-glimmer_zh-cn.md) |
| Qwen-Image-Edit | [Qwen-Image-Edit-2511](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF)（MMDiT + VAE + Qwen2.5-VL）· 快速路径：[Lightning 4 步 LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) | 🖼️ 图像→图像 | — | — | [qwenimage](docs/models/qwenimage_zh-cn.md) |
| MiniMax-H3 音视频 | [unsloth/MiniMax-H3-GGUF](https://huggingface.co/unsloth/MiniMax-H3-GGUF)（去噪器 + Qwen3-VL-32B 文本编码器）+ [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)（视频 VAE + 音频 VAE） | 🎬🔊 文本→视频、图像→视频、首尾帧、参考（图像/片段/音轨）→视频，**带立体声音频** | — | — | [minimax-h3](docs/models/minimax-h3_zh-cn.md) |
| Wan 2.1 / 2.2 视频 | [Wan2.2-TI2V-5B](https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF)（另有 [T2V-A14B](https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF)、[I2V-A14B](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF)、[Wan2.1-T2V-14B](https://huggingface.co/city96/Wan2.1-T2V-14B-gguf)）+ UMT5-XXL + 视频 VAE · 快速路径：[TI2V-5B-Turbo](https://huggingface.co/hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF)（4 步，DiT 前向次数减少 25×） | 🎬 文本→视频、图像→视频 | — | — | [wan](docs/models/wan_zh-cn.md) |

## 让它跑得更快

按这个顺序选择：

1. **先选对 checkpoint。** Wan 视频优先使用 Turbo/Lightning/4-step 蒸馏 GGUF；Qwen-Image-Edit 使用 Lightning LoRA。
2. **使用匹配的后端。** NVIDIA：`ggml_cuda`；Apple Silicon 和 iOS：`ggml_metal`；CPU：`ggml_cpu`（需要可移植性时使用纯托管 `cpu`）。
3. **先减少工作量，再调参数。** H3 使用 `--cfg 1.0` 和 4–8 步；媒体任务优先降低分辨率、帧数或步数。
4. **最后再扩展或投机。** 根据模型和负载尝试 `--draft-model` / `--spec`、`--n-cpu-moe` 或 `--tp N`。

详见[性能指南与快速路径](docs/PROJECT_STATUS_zh-cn.md#让它跑得更快)、[模型卡片](docs/models/README_zh-cn.md)和[环境变量功能矩阵](docs/env_var_feature_matrix_zh-cn.md)。

## 支持的模型架构

| 架构 | GGUF 架构标识 | 示例模型 | 多模态 | 思维链 | 工具调用 | MTP 投机 | 卡片 |
|---|---|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | `deepseek41` | DeepSeek-V4.1-Flash（40 层，384 个路由专家 top-6 加一个共享专家，四条残差流与延迟 hyper-connection 混合，Engram n-gram 特征，声明 1M 上下文） | 文本；配合准备好的视觉伴随文件（`--mmproj`）支持图像与视频，音频请求被拒绝 | 支持 | 支持（带空格的 DSML，受语法约束） | 不支持（V4 的草稿模型会被拒绝） | [deepseek41](docs/models/deepseek41_zh-cn.md) |
| DeepSeek V4 Flash | `deepseek4` | DeepSeek-V4-Flash（284B MoE，256 专家，压缩稀疏注意力，1M 上下文） | 仅文本 | 支持 | 支持（DSML） | 支持（DSpark 块级草稿，独立 GGUF） | [deepseek4](docs/models/deepseek4_zh-cn.md) |
| GLM 5.x | `glm-dsa`、`glm5next` | GLM-5.2（744B-A40B MoE，256 专家，MLA + DeepSeek 稀疏注意力，1M 上下文）、[GLM-5.3](docs/models/glm_zh-cn.md#glm-53glm-dsa)（与 5.2 完全相同的 79 层 `glm-dsa` 形态——78 层主干加 1 个 NextN，256 个路由专家 top-8 外加 1 个共享专家，带 lightning indexer 的 MLA，rope base 8e6——因此直接走 GLM-5.2 的加载路径，既不需要新代码也不需要新开关；仅文本）、GLM-5.3-Flash（320B MoE，288 专家，KDA 线性注意力 + NoPE MLA 与池化索引器） | 仅文本（5.2 与 5.3）、图像（5.3-Flash） | 支持 | 支持（XML 工具调用） | GLM-5.2 与 GLM-5.3 支持（内嵌 NextN 块；5.3 上投机在默认的按层切分下生效，即不传 `--tp` 时） | [glm](docs/models/glm_zh-cn.md) |
| Qwen 3.8 Flash Next | `qwen4exp` | Qwen3.8-Flash-Next（混合 MoE，512 专家 / 激活 10 个，48 层中 36 层为 GatedDeltaNet 并与 QSA 索引的全注意力层交错，PLE n-gram 块，×4 超连接） | 图像 | 支持 | 否（无结构化工具输出解析器） | — | [qwen38-flash-next](docs/models/qwen38-flash-next_zh-cn.md) |
| Gemma 4 | `gemma4` | gemma-4-E4B、gemma-4-31B、gemma-4-26B-A4B（MoE） | 图像、视频、音频 | 支持 | 支持 | 支持（独立草稿 GGUF） | [gemma4](docs/models/gemma4_zh-cn.md) |
| Qwen 3.5 / 3.6 family | `qwen35`, `qwen35moe`, `qwen3next` | Qwen3.5-9B（混合 Attn+递归）、Qwen3.5/3.6-35B-A3B（MoE） | 图像 | 支持 | 支持 | Qwen 3.6 支持（内嵌 NextN） | [qwen35](docs/models/qwen35_zh-cn.md) |
| Bonsai（Qwen 家族） | `qwen3`（8B）、`qwen35`（27B） | Bonsai-8B（36 层稠密 GQA）、Bonsai-27B（48 层 GatedDeltaNet + 16 层全注意力），均为 Q1_0 | 仅文本 | 27B 支持；8B 模板输出固定的空 think 块 | 支持 | — | [bonsai](docs/models/bonsai_zh-cn.md) |
| GPT OSS | `gptoss`, `gpt-oss` | gpt-oss-20b（MoE） | 仅文本 | 支持（始终） | 支持 | — | [gptoss](docs/models/gptoss_zh-cn.md) |
| Nemotron-H | `nemotron_h`, `nemotron_h_moe` | Nemotron-H-8B/47B（混合 SSM-Transformer，MoE）、Nemotron 3 Nano Omni、Nemotron 3.5 Lightning 30B-A3B（23 Mamba-2 + 23 MoE + 6 注意力） | 图像（Omni） | 支持 | 支持 | Nemotron 3.5 Lightning：DSpark 块级起草（独立草稿 GGUF） | [nemotron](docs/models/nemotron_zh-cn.md) |
| Mistral 3 | `mistral3` | Mistral-Small-3.1-24B-Instruct | 图像 | 不支持 | 不支持 | — | [mistral3](docs/models/mistral3_zh-cn.md) |
| Hunyuan Dense | `hunyuan-dense` | 腾讯稠密 Hunyuan 解码器，例如 Hy-MT2（GQA，per-head QK-norm 在 NeoX RoPE **之后**，SwiGLU） | 仅文本 | 不支持 | 不支持 | — | [hunyuan-dense](docs/models/hunyuan-dense_zh-cn.md) |
| Muse-Glimmer | `muse-glimmer`、`muse_glimmer` | Muse-Glimmer-30B（交错滑动窗口 + NoPE 全注意力层，注意力输出门控） | 图像 | 支持 | 支持（ATEM） | 支持（DFlash 块级草稿，独立 GGUF） | [muse-glimmer](docs/models/muse-glimmer_zh-cn.md) |
| DiffusionGemma | `diffusion-gemma`、`diffusion_gemma` | diffusion-gemma 文本扩散 GGUF | 仅文本 | 不支持 | 不支持 | — | [diffusiongemma](docs/models/diffusiongemma_zh-cn.md) |
| Qwen-Image-Edit | `qwen_image`、`qwen-image` | qwen-image-edit MMDiT GGUF（+ VAE 与 Qwen2.5-VL） | 图像编辑（图像+文本 → 图像） | 不支持 | 不支持 | — | [qwenimage](docs/models/qwenimage_zh-cn.md) |
| MiniMax-H3 | `minimax-h3`、`minimax_h3`（官方发布的 GGUF 完全没有元数据，因此靠张量表识别） | MiniMax-H3 FL2VA / Ref2VA（193 亿参数的打包音视频 DiT + Qwen3-VL-32B 文本编码器、视频 VAE、音频 VAE） | 视频输出 **+ 32 kHz 立体声音频**（文本→视频、图像→视频、首尾帧、参考→视频） | 不支持 | 不支持 | — | [minimax-h3](docs/models/minimax-h3_zh-cn.md) |
| Wan 视频 | `wan`、`wan2.1`、`wan2.2` | Wan 2.1 T2V 1.3B/14B、Wan 2.2 TI2V-5B、Wan 2.2 A14B T2V/I2V（双专家） | 视频输出（文本→视频、图像→视频） | 不支持 | 不支持 | — | [wan](docs/models/wan_zh-cn.md) |

各架构的端到端文档（前向图、组件、参数、prefill/decode 优化）见[按模型架构卡片](docs/models/README_zh-cn.md)。

## 性能数据

### 对比 llama.cpp 的同台评测（引擎对比）

纯 .NET 引擎与手工优化的 C++ `llama.cpp` 正面较量：**相同的 GGUF 文件、相同的 NVIDIA RTX 3080 Laptop GPU（16 GB）、统一的 OpenAI `/v1/chat/completions` 接口**，**两个引擎均分别在 GGML CUDA 与 Vulkan 构建上测量**。下表为 **在相同后端上，TensorSharp 相对 llama.cpp 的几何平均加速比**（单流、贪心采样、关闭 MTP）；**> 1.0× 表示 TensorSharp 更快 / 延迟更低**。完整表格见 [`docs/engine_comparison_report.md`](docs/engine_comparison_report.md)。

| 模型 | 后端 | decode | prefill | TTFT |
|---|---|---:|---:|---:|
| Gemma 4 E4B it（Q8_0，dense 多模态） | CUDA | 1.02× | **1.28×** | **1.27×** |
| Gemma 4 E4B it（Q8_0，dense 多模态） | Vulkan | 1.00× | 1.05× | 1.03× |
| Gemma 4 12B it（QAT UD-Q4_K_XL，dense） | CUDA | 1.04× | **1.17×** | **1.16×** |
| Gemma 4 12B it（QAT UD-Q4_K_XL，dense） | Vulkan | **1.21×** | 1.04× | 1.03× |
| Qwen 3.6 35B-A3B（UD-IQ2_XXS，MoE） | CUDA | 0.98× | **1.28×** | **1.27×** |
| Qwen 3.6 35B-A3B（UD-IQ2_XXS，MoE） | Vulkan | 0.87× | 1.04× | 1.03× |
| Qwen 3.6 27B（UD-IQ2_XXS，dense） | CUDA | **1.07×** | 0.96× | 0.95× |
| Qwen 3.6 27B（UD-IQ2_XXS，dense） | Vulkan | 1.02× | 0.85× | 0.84× |

TensorSharp 在 CUDA 的 prefill / 首 token 延迟上明显领先（多轮 prefill **每个模型**都获胜，最高 **1.49×**），CUDA decode 保持持平或更快，Vulkan 上 dense 12B 的 decode 明显胜出（长上下文最高 **1.32×**）——即便在 2-bit IQ2_XXS 量化下亦然。剩余低于 1.0× 的项仍是正在优化的目标。该框架还提供工具调用、结构化输出、图像编辑（对比 `stable-diffusion.cpp`）、MTP 开/关与并发场景，可通过 [`benchmarks/engine_comparison`](benchmarks/engine_comparison) 在你自己的硬件上运行。完整报告见 [此处](docs/engine_comparison_report.md)。

放不进这台 16 GB 机器的模型，会在各自的卡片里给出同样方式测得的正面对比（两个引擎、同一份 GGUF、同一台机器、背靠背）：[GLM-5.2 744B-A40B，3× RTX PRO 6000](docs/models/glm_zh-cn.md#性能) —— 从约 1k prompt token 起 TensorSharp 的 prefill 领先（pp2048 **1.20×**、pp4096 **1.21×**），decode 领先 1.04×，短 prefill 上则是 llama.cpp 快几个百分点。非 Flash 的 [GLM-5.3](docs/models/glm_zh-cn.md#glm-53glm-dsa) 另有一份自己的对比，测于 8 张 A40 46 GB（无 NVLink，UD-Q2_K_XL，10,531 token 提示，300 个 decode token，3 次取中位数，整层放置）：decode 打平，**20.48** tok/s 对 llama.cpp 的 20.28；TensorSharp 的 prefill 为 251.6 tok/s，加载这份 236.4 GiB 的 checkpoint **快 2.9×**（264 秒对 753 秒）；真正的差距在首 token 延迟——41.9 秒对 29.0 秒，约**慢 1.4×**。该组数据没有记录 llama.cpp 的 prefill tok/s。完整方法与逐次数据见 [`docs/validation/cross-engine-2026-09/README.md`](docs/validation/cross-engine-2026-09/README.md)。llama.cpp 可以作为 `glm-dsa` 的参照引擎，但不能作为 `glm5next`（GLM-5.3-Flash）的参照引擎。

## 文档

初次使用？上面几节足以让你跑起来。其余均为详细参考：

| 文档 | 内容 |
|---|---|
| [书籍指南：《From Tensors to Tokens》](docs/BOOK_zh-cn.md) | 从张量基础走向 Gemma 4 E4B 多模态推理引擎的连贯路线，含出版信息与配套仓库阅读指引 |
| [模型下载](MODEL_DOWNLOADS_zh-cn.md) | 各模型 `huggingface-cli` 下载 + 运行速查（量化档位、投影器、伴随文件） |
| [使用方法](USAGE_zh-cn.md) | 完整 CLI 参考（选项、交互式 REPL、JSONL 批处理）、服务端托管、日志、HTTP API 示例、后端与环境变量矩阵 |
| [功能特性](FEATURES_zh-cn.md) | 连续批处理、投机解码、工具调用、思维链、多模态、MoE、KV 编解码等深入说明 |
| [配置文件](config/README.md) | 把参数写进可复用的 JSON 文件，支持 `${变量}` 与模型自动下载 |
| [开发](DEVELOPMENT_zh-cn.md) | 前置要求、构建原生 GGML/MLX 库、仓库结构、包分层、内部架构与测试工具 |
| [按模型架构卡片](docs/models/README_zh-cn.md) | 各架构端到端文档（前向图、组件、参数、prefill/decode 优化） |
| [分页注意力 & 连续批处理](docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md) | vLLM 风格的分页 KV 缓存、前缀共享与迭代级调度器 |
| [Agent Skills 与智能体工作](docs/agent_skills.md)（英文） | `SKILL.md` 格式、渐进式披露与其预算、进程内工具循环、沙箱化代码执行、工作区与产物、路径 / ZIP / 执行安全模型，以及 HTTP 与 C# 两套接口 |
| [投机解码](docs/speculative_decoding.md)（英文） | 三层设计（模型适配层 / 算法 / 草稿权重）、已内置的 `auto` / `draft-head` / `block` / `ngram` 四种算法，以及新增一种算法需要写什么 |
| [环境变量功能矩阵](docs/env_var_feature_matrix_zh-cn.md) | 哪些高影响运行时开关影响哪些模型、后端与提示类型 |
| [引擎对比报告](docs/engine_comparison_report.md) | TensorSharp 对比 llama.cpp / stable-diffusion.cpp 的完整逐场景表格 |
| [ggml_metal 对比 llama.cpp](docs/perf/metal-vs-llama-cpp.md) | Apple Silicon 上 prefill / decode 的正面对比，找到的四处计算图构建差距，以及每一处的实际收益 |
| [测试 / 基准矩阵运行器](TensorSharp.TestMatrix/README_zh-cn.md) | 扫描 model × backend × feature × env-var 组合并生成回归报告 |
| [服务端 API 示例](TensorSharp.Server.Host/API_EXAMPLES_zh-cn.md) | 完整的 curl 与 Python 示例 |

## 当前状态

| 范围 | 状态 |
|---|---|
| 模型家族 | DeepSeek V4 Flash（`deepseek4`）、DeepSeek V4.1 Flash（`deepseek41`）、GLM 5.x（`glm-dsa`、`glm5next`）、Gemma 4、DiffusionGemma、Qwen 3.5/3.6-family（`qwen35`、`qwen35moe`、`qwen3next`）、Qwen 3.8 Flash Next（`qwen4exp`）、GPT OSS、Nemotron-H（含 Nemotron 3 Nano Omni 与 Nemotron 3.5 Lightning，`nemotron_h_moe`）、Mistral 3、Hunyuan Dense（`hunyuan-dense`）、Muse-Glimmer（`muse-glimmer`、`muse_glimmer`）。图像编辑通过 Qwen-Image-Edit（`qwen_image`、`qwen-image` MMDiT）；音视频联合生成通过 MiniMax-H3（`minimax-h3`、`minimax_h3`），纯视频生成通过 Wan 2.1 / 2.2（`wan`、`wan2.1`、`wan2.2`）。 |
| 推理宿主 | CLI、交互式 REPL、ASP.NET Core Web UI、Ollama 风格 API、OpenAI Chat Completions 风格 API 与 OpenAI Responses 风格 API。 |
| iOS 应用 | TensorAgent 支持 iOS/iPadOS，将 GGML 作为 iOS `.xcframework` 链接，并在真机上使用 `ggml_metal`。它共享与宿主无关的聊天流水线（`TensorSharp.Chat`），但通过进程内 loopback 宿主提供自己的手机版页面——iOS 既没有 ASP.NET Core 运行时包，也不能启动子进程。生成过程在应用离开屏幕后仍然继续；共享提示词前缀的 checkpoint 会按模型持久化，使每次启动的第一条消息只需一次恢复而不必完整预填充（在 iPhone 17 Pro Max 上以 Qwen3.5 9B 实测：原本 54 秒的冷启动首条消息，变成 1.2 秒预热加约 0.6 秒的首条消息）；引擎的内存策略也按 iOS jetsam 实际计费的口径来设定。详见 [TensorAgent](TensorAgent/README.md)。 |
| 后端 | 纯 C# CPU、Direct CUDA/cuBLAS（`cuda`）、MLX Metal（`mlx`）、GGML CPU、GGML Metal、GGML CUDA、GGML Vulkan。DeepSeek V4 另有三套专属的整模型执行器——Direct CUDA、原生 ggml 与纯 C# CPU——都会把权重按层切分到所有可见 GPU（`--tp N` / `TS_DSV4_NGPU` 限定卡数）。DeepSeek V4.1 的服务路径是 `ggml_cuda`；`ggml_cpu` 用同一套原生计算图跑标量回退实现，`cpu` 则是纯 C# 的 V4.1 执行器，两者都是正确性与可移植性通道，而非服务通道。`cuda` 用 Direct CUDA 引擎自己的内核运行 V4.1（不经过 ggml），目前还没有数值门禁。`ggml_vulkan` / `ggml_metal` 需要 `TS_DSV41_ALLOW_NON_CUDA_GPU=1`；`mlx` 会直接拒绝该检查点，而不会把 V4.1 的权重塞进并未实现它的计算图。视频家族中，Wan 是对后端有限制的那一个：它可运行于各 GGML 后端以及 Direct `cuda` / 纯 C# `cpu` 后端，但不支持 MLX。 |
| 多模态 | Gemma 4 图像/视频/音频；Qwen 3.5-family、Qwen 3.8 Flash Next、GLM-5.3-Flash、Mistral 3、Nemotron-H Omni、Muse-Glimmer 图像输入；PDF（CLI `--pdf` + Web UI）。媒体*输出*：Qwen-Image-Edit（图像）、MiniMax-H3（H.264 MP4 **外加一份 32 kHz 立体声 `.wav` 旁挂文件**，两者在同一份打包潜变量里一起生成），以及 Wan 2.1 / 2.2（仅 H.264 MP4 视频，文本→视频与图像→视频）。 |
| 连续批处理 | vLLM 风格分页 KV 缓存、基于内容哈希的前缀共享、共享前缀 checkpoint（所有会话共享的那段提示词末尾的状态会被克隆进每个新会话，因此新会话只需重新 prefill 自己的那条消息；适用于 GGML 上的 Gemma 4 与 Qwen 3.5/3.6，宿主还可通过 `IPrefixCheckpointStore` 让它跨进程重启存活）、迭代级调度器（默认启用，`--no-continuous-batching` 关闭）。分页池常驻主机内存，因此它买到的是内存效率与前缀复用，而不是随并发增长的吞吐。DeepSeek V4 与 GLM 5.x 在同一引擎上通过各自原生的 per-sequence slot 提供服务——压缩后的 MLA 每 token 只有一行缓存，没有可分页的布局——GLM 的批处理融合解码默认启用（设置 `TS_BATCHED_FUSED_DECODE=0` 可切回串行融合 decode；4 路并发下总吞吐 1.81 倍）。Qwen 3.8 Flash Next 出于同样的原因使用逐序列状态持有者——它的 GatedDeltaNet、PLE 与索引器状态同样没有可分页的布局。 |
| 投机解码 | Qwen 3.6、GLM 5.2 与 GLM-5.3（三者均内嵌于 checkpoint——GLM-5.3 的 `blk.78` NextN 块本身是完整的，但它没有自己的 LM head，因此投机只在默认的按层切分下生效，即不传 `--tp` 时）以及 Gemma 4（独立草稿 GGUF，通过 `--draft-model` 加载）的 MTP / NextN 草稿头；DeepSeek V4 的 DSpark 块级起草（仅 `cuda` / `ggml_cuda`）、Muse-Glimmer 与 Qwen 3.8 的 DFlash / DFlash2 块级起草，以及 Nemotron 3.5 Lightning 的 DSpark Markov 头块级起草——它是第一个作为块级起草目标的递归（Mamba-2）主干——这些都通过 `--draft-model` 加载独立的草稿 GGUF；此外还有一个不需要任何草稿权重的 n-gram（prompt-lookup）投机器，用 `--spec-type ngram` 选择，因而在任何检查点上都能用。每个输出 token 都取自主干的一行 logits，并由本次运行自身配置的采样器抽出，因此输出流与普通 decode 产生的完全相同。默认关闭；内嵌草稿头在 CLI 与服务端两端均以 `--spec` 启用，而对以独立 GGUF 发布的草稿器，传入 `--draft-model` 本身即可启用投机。 |
| 张量并行 | Direct `cuda` 后端与 GGML CUDA / Vulkan 后端上的 Megatron-LM 列/行并行 TP（`--tp N` / `TENSORSHARP_TP_DEGREE`，CLI 与服务端均支持）；通过点对点 TCP 的多节点分布式 TP（`--tp-node-id` / `--tp-peers`），采用分层 AllReduce，CUDA P2P 不可用时自动回退到主机中转。覆盖全部自回归架构；GGML 上 Gemma 4 与 Qwen 3.5/3.6 使用 MoE 专家并行与融合的按 rank decode/prefill 计算图。GLM 5.x 默认按层切分，但 GGML GPU 后端上的 `--tp N` 会为 GLM 5.2、GLM-5.3 与 GLM-5.3-Flash 选择仅支持本地单进程的原生 TP（GLM-5.3-Flash 会切 KDA / MLA head 与路由专家隐藏行）；整个 GLM 家族都会在构建模型之前硬拒 `--tp-node-id` / `--tp-peers`，而在 GLM-5.3 上 `--tp N` 只是一个被接受的模式，而非已验证的配置——它会按 rank 复制 MLA 与索引器缓存，占用随之成倍增长、能装下的上下文随之缩短，并且该 checkpoint 上从未跑过 `--tp 1` 以上的配置（`--tp 8` 折算下来每个 rank 需要 41.7 GiB，放不进 46 GB 的卡）。本身不切分权重的架构把同一个 `--tp N` 当作按层切分——每张 GPU 拿一段连续的整层，DeepSeek V4 与 V4.1 即是如此（V4.1 上还可用 `TS_DSV41_TP=N` 打开实验性的 routed-MoE 张量并行，目前实测比按层切分更慢）；Qwen 3.8 Flash Next（`qwen4exp`）上可用 `TS_Q4E_LAYER_SPLIT=20,28` 覆盖自动均衡，遇到无法满足的切分会直接报错而不是静默忽略。启动时会打印实际采用的模式与每张 GPU 的层数/字节分配；两种模式都不支持的架构会在 stderr 上明说，并改用单卡运行。可选 Redis 支撑的 KV 缓存与 Responses API 存储。 |
| Agent Skills | 技能目录来自 `--skills-dir`（或二进制旁的 `skills` 目录），也可通过 `POST /api/skills` 在运行期安装。在 `/v1/chat/completions`、`/v1/responses`、`/api/chat/ollama`（Ollama）与 `/api/chat`（Web UI）上用 `"skills": [...]` 按请求选中，CLI 上用 `--skill`。支持完整工具闭环的家族只接收元数据，并通过进程内应答的 `skills_list` / `skills_read` 激活说明；调用方自己的工具仍照常回传。脚本执行（`skills_run`）默认关闭。Mistral 3 以及没有可解析工具协议的家族（包括 `qwen4exp`）改为内联选中技能正文，且不提供技能 / 代码工具。 |
| 智能体代码工作 | 可选的 `--code-exec` 在同一个有界“模型→工具”循环里提供 `read_file`、`edit_file`、`write_file`、`shell` 与原子 `apply_patch`。Web UI / CLI 聊天保留会话工作区；每个 OpenAI / Ollama 请求只在内部修复轮次间保留一个私有工作区，响应后删除。生成文件可作为产物下载。这是单模型的进程内循环；TensorSharp 目前不提供多智能体委派或逐命令审批工作流。 |
| 沙箱与权限 | 代码执行和技能脚本默认关闭。macOS 使用 Seatbelt，Linux 需要 `bwrap` 0.12.0+；`required` 模式在无法隔离时拒绝运行。Windows 代码执行必须显式传入 `--code-exec-unconfined`，Windows 技能脚本则需以 `--skills-sandbox preferred` 明确接受仅 job-object 的限制。技能脚本联网、代码联网与宿主代办装包是三个独立开关。 |
| 发行状态（核验于 2026-09-08） | GitHub v3.3.0.0 提供十个自包含 CLI / Server 归档，覆盖 Windows x64 CPU/CUDA、Linux x64 CPU/CUDA 与 macOS arm64。源码树的发布集合现为 **13** 个 NuGet 包（`eng/verify-packages.ps1` 是权威清单，并作为发布流水线门禁）。NuGet.org 目前只有其中 8 个——Tensors、Runtime、Models、GGML/CUDA/MLX 后端、Server 与 CLI，版本 **3.1.2**（2026 年 7 月）——它们落后于源码与二进制发行版；已发布的 `TensorSharp.Server` 早于日志层与聊天层拆分。Runtime.Logging、AgentHost、Chat、Server.Host 与 Distributed 已可打包并通过校验，但尚未推送，使用这些层仍需从源码 checkout 添加项目引用。 |
| 服务端模型范围 | 通过 `--model` 显式托管单个 GGUF；可通过 `--mmproj` 显式指定投影器；不扫描目录。 |
| 可观测性 | 结构化每轮日志、队列状态，以及 Web UI / Ollama / OpenAI 中的 KV 缓存复用指标。 |

## 作者

Zhongkai Fu

## 许可证

详见 [LICENSE](LICENSE)。
