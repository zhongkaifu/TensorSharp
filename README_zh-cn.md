# TensorSharp

<p align="center">
  <img src="imgs/banner_1.png" alt="TensorSharp logo" width="320">
</p>

[English](README.md) | [中文](README_zh-cn.md)

一个用于在本地运行大型语言模型（LLM）的 C# 推理引擎，使用 GGUF 模型文件。TensorSharp 提供控制台应用、基于 Web 的聊天界面，以及兼容 Ollama/OpenAI 的 HTTP API 以便程序化调用。

## 功能特性

- **多架构支持** —— Gemma 4、Gemma 3、Qwen 3、Qwen 3.5、GPT OSS、Nemotron-H、Mistral 3
- **多模态推理** —— 图像、视频和音频输入（Gemma 4）；图像输入（Gemma 3 / Qwen 3.5 / Mistral 3）
- **思维链 / 推理模式** —— 通过 `<think>` / `<|channel>thought` / `<|channel>analysis` 标签输出结构化的思维链推理（Qwen 3、Qwen 3.5、Gemma 4、GPT OSS、Nemotron-H）
- **工具调用 / 函数调用** —— 模型可调用用户定义的工具；所有三种 API 风格均支持多轮工具调用对话
- **量化模型支持** —— 加载 Q4_K_M、Q8_0、F16、MXFP4 等量化格式的 GGUF 文件；执行原生量化矩阵乘法（matmul），无需反量化到 FP32，并且纯 C# CPU 后端在加载大型 GGUF 时也会保持量化权重压缩状态
- **GPU 加速** —— 通过 GGML 支持 Apple Metal（macOS）和 GGML CUDA（Linux/NVIDIA）；Gemma 4 在 Metal 上支持整模型融合 GPU decode（相对逐算子调度约提升 2.6 倍）
- **优化后的纯 C# CPU 后端** —— 为 GEMM、RMSNorm、RoPE、softmax、融合激活等推理热点路径提供托管快速路径和 SIMD 内核
- **兼容 Ollama 与 OpenAI API** —— 可作为现有工具链的即插即用替代端点
- **可配置采样** —— temperature、top-k、top-p、min-p、重复/存在/频率惩罚、seed、停止序列
- **聊天模板** —— 从 GGUF 元数据自动加载（Jinja2），并为不同架构提供硬编码回退模板
- **请求队列** —— FIFO 推理队列确保单请求执行以保障 KV 缓存稳定性，并为客户端提供实时排队位置反馈
- **批处理** —— 控制台应用支持 JSONL 输入，并内置用于测量 prefill / decode 吞吐的推理基准
- **流式输出** —— 按 token 输出（Web 通过 SSE，控制台通过 stdout），并支持中断/停止正在生成的请求
- **混合 SSM-Transformer** —— Nemotron-H 在单个模型中混合 Mamba2 SSM 层、纯注意力层和 MoE FFN 层
- **混合注意力-递归网络** —— Qwen 3.5 在同一模型中混合全注意力层与 GatedDeltaNet 递归层
- **专家混合（MoE）** —— 支持 Gemma 4 MoE 变体（例如 gemma-4-26B-A4B）、GPT OSS MoE（例如 gpt-oss-20b）、Qwen 3.5 MoE（`qwen35moe` / `qwen3next` 变体，例如 Qwen3.5-35B-A3B）以及 Nemotron-H MoE FFN 层
- **批量 GPU MoE** —— Qwen 3.5 与 Nemotron-H 在 decode 时通过单次融合的 GGML 计算图调度处理所有被选中的专家（Qwen 3.5 还包括可选的 shared expert 与残差加法），消除每个专家的 CPU-GPU 往返
- **消息编辑** —— 在 Web 聊天界面中编辑或删除历史消息，并从该位置重新生成回复
- **文本/图像/音频/视频上传** —— Web 界面支持最大 500 MB 的文件上传，对超大文本会按 token 预算自动截断

## 支持的模型架构

| 架构 | GGUF 架构标识 | 示例模型 | 多模态 | 思维链 | 工具调用 |
|---|---|---|---|---|---|
| Gemma 4 | `gemma4` | gemma-4-E4B、gemma-4-31B、gemma-4-26B-A4B（MoE） | 图像、视频、音频 | 支持 | 支持 |
| Gemma 3 | `gemma3` | gemma-3-4b | 图像 | 不支持 | 不支持 |
| Qwen 3 | `qwen3` | Qwen3-4B | 仅文本 | 支持 | 支持 |
| Qwen 3.5 | `qwen35`, `qwen35moe`, `qwen3next` | Qwen3.5-9B（混合 Attn+递归）、Qwen3.5-35B-A3B（MoE） | 图像 | 支持 | 支持 |
| GPT OSS | `gptoss`, `gpt-oss` | gpt-oss-20b（MoE） | 仅文本 | 支持 | 不支持 |
| Nemotron-H | `nemotron_h`, `nemotron_h_moe` | Nemotron-H-8B、Nemotron-H-47B（混合 SSM-Transformer，MoE） | 仅文本 | 支持 | 支持 |
| Mistral 3 | `mistral3` | Mistral-Small-3.1-24B-Instruct | 图像 | 不支持 | 不支持 |

各架构的详细文档见[模型架构卡片](docs/model_cards_cn.md)。

## 模型下载（GGUF）

TensorSharp 使用 GGUF 格式模型文件。以下是各架构对应的 Hugging Face 下载链接。请根据硬件条件选择合适的量化版本（Q4_K_M 适合低内存，Q8_0 适合更高质量等）。

| 架构 | 模型 | GGUF 下载 |
|---|---|---|
| Gemma 4 | gemma-4-E4B-it | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) |
| Gemma 4 | gemma-4-31B-it | [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) |
| Gemma 4 | gemma-4-26B-A4B-it（MoE） | [ggml-org/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) |
| Gemma 4 | gemma-4-mmproj（多模态投影器） | 包含在上述 GGUF 仓库中 |
| Gemma 3 | gemma-3-4b-it | [google/gemma-3-4b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf) |
| Qwen 3 | Qwen3-4B | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) |
| Qwen 3.5 | Qwen3.5-9B | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) |
| Qwen 3.5 | Qwen3.5-35B-A3B | [ggml-org/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF) |
| GPT OSS | gpt-oss-20b（MoE） | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) |
| Nemotron-H | Nemotron-H-8B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron-H-47B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) |
| Mistral 3 | Mistral-Small-3.1-24B-Instruct | [bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF) |
| Mistral 3 | mistral3-mmproj（Pixtral 视觉投影器） | [bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF) |

## 计算后端

| 后端 | 参数 | 说明 |
|---|---|---|
| GGML Metal | `--backend ggml_metal` | 通过 Apple Metal（macOS）进行 GPU 加速。推荐用于 Apple Silicon。量化权重通过 host 指针缓冲区从 GGUF 文件零拷贝映射到 Metal command buffer，常驻内存接近模型在磁盘上的大小。 |
| GGML CUDA | `--backend ggml_cuda` | 通过 GGML CUDA 在 Linux + NVIDIA GPU 上进行加速。量化权重在加载时一次性上传到设备显存，之后释放主机端拷贝。 |
| GGML CPU | `--backend ggml_cpu` | 使用原生 GGML 与优化内核进行 CPU 推理。量化权重以零拷贝方式从 GGUF 文件映射。 |
| 纯 C# CPU | `--backend cpu` | 无原生依赖的可移植 CPU 推理。 |

## 项目结构

```text
TensorSharp/
├── TensorSharp.Core/            # 核心张量库（Tensor、Ops、内存、设备抽象）
├── TensorSharp.Runtime/         # GGUF、分词器、模板、采样、协议解析
├── TensorSharp.Models/          # 模型架构实现与多模态编码/注入
├── TensorSharp.Backends.GGML/   # GGML 后端绑定（通过原生库支持 Metal/CUDA/CPU）
├── TensorSharp.GGML.Native/     # 到 ggml 的原生 C++ 桥接（构建 libGgmlOps）
├── TensorSharp.Server/          # Web 聊天 + API 服务（ASP.NET Core）
│   ├── ModelService.cs          # 模型生命周期管理
│   ├── InferenceQueue.cs        # 带排队位置跟踪的 FIFO 请求队列
│   ├── BackendCatalog.cs        # 可用计算后端的发现
│   ├── TextUploadHelper.cs      # 按 token 预算截断的文本上传辅助
│   ├── wwwroot/index.html       # 聊天界面
│   ├── testdata/                # 集成测试套件（bash + Python）
│   └── API_EXAMPLES.md          # 详细 API 文档
├── TensorSharp.Cli/             # CLI 应用
├── InferenceWeb.Tests/          # xUnit 单元测试，覆盖算子、KV 缓存与 Web/服务辅助逻辑
├── AdvUtils/                    # 工具库
├── docs/                        # 开发者参考文档（模型卡片，中英）
└── ExternalProjects/            # 第三方依赖（ggml）
```

## NuGet 包分层

现在仓库按包边界拆成独立层，使用者可以只引用真正需要的部分。

| 项目 | NuGet 包 | 对外 namespace | 职责 |
|---|---|---|---|
| `TensorSharp.Core` | `TensorSharp.Core` | `TensorSharp` | Tensor 原语、Ops、分配器、存储与设备抽象 |
| `TensorSharp.Runtime` | `TensorSharp.Runtime` | `TensorSharp.Runtime` | GGUF 解析、分词器、Prompt 渲染、采样与输出协议解析 |
| `TensorSharp.Models` | `TensorSharp.Models` | `TensorSharp.Models` | `ModelBase`、各模型架构、多模态编码器与模型侧执行辅助 |
| `TensorSharp.Backends.GGML` | `TensorSharp.Backends.GGML` | `TensorSharp.GGML` | GGML 执行后端与原生互操作 |
| `TensorSharp.Server` | `TensorSharp.Server` | `TensorSharp.Server` | ASP.NET Core 服务、OpenAI/Ollama 适配层、队列与 Web UI |
| `TensorSharp.Cli` | `TensorSharp.Cli` | `TensorSharp.Cli` | 控制台宿主、调试工具与 JSONL 批处理 |

这样的拆分让引擎使用者不必带上 Web 依赖，也能把 API 层改动和核心运行时隔离开，并让后续 benchmark / eval harness 更容易独立发布。

## 前置要求

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- **macOS（Metal 后端）：** 用于构建原生 GGML 库的 CMake 3.20+ 与 Xcode 命令行工具
- **Linux（GGML CPU / CUDA 后端）：** CMake 3.20+；若使用 `ggml_cuda`，还需要 NVIDIA 驱动和 CUDA Toolkit 12.x 或其他兼容版本
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

在 Linux 上，`build-linux.sh` 现在会自动检测可见 NVIDIA GPU 的 compute capability，并把一个精简的 `CMAKE_CUDA_ARCHITECTURES` 列表传给 ggml-cuda（例如在 RTX 3080 上为 `86-real`），从而显著降低 CUDA 构建时间。原生构建默认还会以受控的并行任务数运行，避免 `nvcc` 拖慢普通开发机器。

如需覆盖自动检测到的架构列表或默认的并行度，可使用以下任一方式：

```bash
TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES='86-real;89-real' bash build-linux.sh --cuda
bash build-linux.sh --cuda --cuda-arch='86-real;89-real'
TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL=2 bash build-linux.sh --cuda
```

也可以在 `dotnet build` 时通过环境变量请求 CUDA 版本的原生库：

```bash
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
```

在 macOS 上会生成带 Metal GPU 支持的 `libGgmlOps.dylib`。在 Linux 上，`build-linux.sh` 会保留已有的 CUDA 构建，并在检测到 CUDA 工具链时自动启用 GGML_CUDA；也可以通过 `build-linux.sh --cuda` 或 `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` 显式启用。构建产物会自动复制到应用输出目录。

## 使用方法

### 控制台应用

```bash
cd TensorSharp.Cli/bin

# 文本推理
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_metal

# Linux + NVIDIA GPU 文本推理
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_cuda

# 图像推理（Gemma 3/4，Qwen 3.5）
./TensorSharp.Cli --model <model.gguf> --image photo.png --backend ggml_metal

# 视频推理（Gemma 4）
./TensorSharp.Cli --model <model.gguf> --video clip.mp4 --backend ggml_metal

# 音频推理（Gemma 4）
./TensorSharp.Cli --model <model.gguf> --audio speech.wav --backend ggml_metal

# 思维链 / 推理模式
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --backend ggml_metal --think

# 工具调用
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --tools tools.json

# 使用采样参数
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.2 --seed 42

# 批处理（JSONL）
./TensorSharp.Cli --model <model.gguf> --input-jsonl requests.jsonl \
    --output results.txt --backend ggml_metal

# 多轮对话模拟（含 KV 缓存复用，模拟 Web UI 行为）
./TensorSharp.Cli --model <model.gguf> --multi-turn-jsonl chat.jsonl \
    --backend ggml_metal --max-tokens 200

# 吞吐基准测试：N 次最优运行的 prefill 和 decode 计时
./TensorSharp.Cli --model <model.gguf> --backend ggml_metal \
    --benchmark --bench-prefill 256 --bench-decode 128 --bench-runs 3

# 仅查看渲染后的 prompt 和分词结果（不运行推理）
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --dump-prompt

# 对目录下每个 *.gguf 文件，对比硬编码回退模板与 GGUF 内置 Jinja2 模板
# （在适配新架构时尤其有用）
./TensorSharp.Cli --test-templates ~/models
```

**命令行参数：**

| 参数 | 说明 |
|---|---|
| `--model <path>` | GGUF 模型文件路径（必填） |
| `--input <path>` | 包含用户提示词的文本文件 |
| `--input-jsonl <path>` | JSONL 批量请求文件（每行一个 JSON） |
| `--multi-turn-jsonl <path>` | 用于多轮对话模拟（含 KV 缓存复用）的 JSONL 文件 |
| `--output <path>` | 将生成文本写入该文件 |
| `--image <path>` | 用于视觉推理的图像文件 |
| `--video <path>` | 用于视频推理的视频文件 |
| `--audio <path>` | 音频文件（WAV、MP3、OGG）用于音频推理 |
| `--mmproj <path>` | 多模态投影器 GGUF 文件路径 |
| `--max-tokens <N>` | 最大生成 token 数（默认：100） |
| `--backend <type>` | 计算后端：`cpu`、`ggml_cpu`、`ggml_metal` 或 `ggml_cuda` |
| `--think` | 启用思维链/推理模式 |
| `--tools <path>` | 包含工具/函数定义的 JSON 文件 |
| `--temperature <f>` | 采样温度（0 = 贪心） |
| `--top-k <N>` | Top-K 过滤（0 = 关闭） |
| `--top-p <f>` | Nucleus 采样阈值（1.0 = 关闭） |
| `--min-p <f>` | 最小概率过滤（0 = 关闭） |
| `--repeat-penalty <f>` | 重复惩罚（1.0 = 无） |
| `--presence-penalty <f>` | 存在惩罚（0 = 关闭） |
| `--frequency-penalty <f>` | 频率惩罚（0 = 关闭） |
| `--seed <N>` | 随机种子（-1 = 非确定性） |
| `--stop <string>` | 停止序列（可重复指定） |
| `--dump-prompt` | 仅渲染 prompt 与分词后退出（不进行推理） |
| `--benchmark` | 运行合成的 prefill / decode 吞吐基准 |
| `--bench-prefill <N>` | 合成 prefill 的 token 长度（默认：32） |
| `--bench-decode <N>` | 合成 decode 的 token 长度（默认：64） |
| `--bench-runs <N>` | 基准运行次数；输出最佳与平均结果（默认：1） |
| `--test` | 运行内置的分词器、Qwen3 聊天模板与 ollama 对比测试 |
| `--test-templates <dir>` | 对 `<dir>` 下的每个 *.gguf 校验硬编码模板与 GGUF Jinja2 模板的一致性 |

如果把多模态投影器文件放在模型文件同目录并使用可识别命名（例如 `gemma-4-mmproj-F16.gguf`），系统会自动检测。

**JSONL 输入格式：**

每行是一个 JSON 对象，包含 `messages`、可选 `prompt` 和可选采样参数：

```json
{"id": "q1", "messages": [{"role": "user", "content": "What is 2+3?"}], "max_tokens": 50}
{"id": "q2", "messages": [{"role": "user", "content": "Write a haiku."}], "max_tokens": 100, "temperature": 0.8}
```

### Web 应用

```bash
cd TensorSharp.Server/bin

# 通过 --model 指定要托管的模型
./TensorSharp.Server --model ./models/model.gguf --backend ggml_metal

# Linux + NVIDIA GPU
./TensorSharp.Server --model ./models/model.gguf --backend ggml_cuda

# 多模态模型：同时显式指定投影器
./TensorSharp.Server --model ./models/model.gguf --mmproj ./models/mmproj.gguf --backend ggml_cuda
```

在浏览器中打开 `http://localhost:5000`。Web 界面支持：

- 多轮聊天
- 通过 `--model` 显式托管单个 GGUF 模型
- 在需要时通过 `--mmproj` 显式托管多模态投影器
- 上传图像、视频和音频进行多模态推理（最大 500 MB）
- 思维链/推理模式切换
- 带函数定义的工具调用
- 通过 Server-Sent Events 进行流式 token 生成
- 带实时排队位置反馈的请求队列
- 消息编辑和删除，支持从对话中任意位置重新生成

使用 `--model` 选择要托管的 GGUF 文件，使用 `--mmproj` 选择要托管的投影器文件。`TensorSharp.Server` 不再扫描 `MODEL_DIR`。

**服务命令行参数：**

| 参数 | 说明 |
|---|---|
| `--model <path>` | 需要托管的 GGUF 文件（推理时必填；如未指定，服务仍可启动，但 `/api/models/load` 会报告未加载模型） |
| `--mmproj <path>` | 多模态投影器 GGUF（仅给文件名时按模型目录解析；传 `none` 可显式禁用）。需要先指定 `--model`。 |
| `--backend <type>` | 默认计算后端：`cpu`、`ggml_cpu`、`ggml_metal` 或 `ggml_cuda` |
| `--max-tokens <N>` | 当请求未携带 max-tokens 时使用的默认上限（默认：`20000`） |

**运行时环境变量：**

| 变量 | 说明 |
|---|---|
| `BACKEND` | 未传 `--backend` 时使用的默认计算后端（默认：macOS 为 `ggml_metal`，其他平台为 `ggml_cpu`） |
| `MAX_TOKENS` | 当 `--max-tokens` 与请求级上限均未指定时使用的默认生成长度（默认：`20000`） |
| `MAX_TEXT_FILE_CHARS` | 在没有可用分词器时，对纯文本上传按字符数截断的上限（默认：`8000`） |
| `VIDEO_MAX_FRAMES` | 视频提示词中均匀抽取的视频帧上限（默认：`4`） |
| `PORT` / `ASPNETCORE_URLS` | 标准 ASP.NET Core 监听配置（默认端口：`5000`） |

### HTTP API

TensorSharp.Server 暴露三种 API 风格。完整文档及 curl/Python 示例见 [API_EXAMPLES.md](TensorSharp.Server/API_EXAMPLES.md)。

**兼容 Ollama 的 API：**

```bash
# 列出模型
curl http://localhost:5000/api/tags

# 文本生成
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B-Q8_0.gguf", "prompt": "Hello!", "stream": false}'

# 聊天
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "stream": false}'

# 启用思维链模式的聊天
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B-Q8_0.gguf", "messages": [{"role": "user", "content": "计算 17*23"}], "think": true, "stream": false}'

# 带工具调用的聊天
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B-Q8_0.gguf", "messages": [{"role": "user", "content": "天气怎么样？"}], "tools": [{"function": {"name": "get_weather", "description": "获取当前天气", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"]}}}], "stream": false}'
```

**兼容 OpenAI 的 API：**

```bash
# Chat completions
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'

# 结构化输出（OpenAI response_format）
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B-Q8_0.gguf",
    "messages": [{"role": "user", "content": "从“Paris, France”中提取城市与国家。"}],
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

**OpenAI Python SDK：**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="Qwen3-4B-Q8_0.gguf",
    messages=[{"role": "user", "content": "What is 2+3?"}],
    max_tokens=50
)
print(response.choices[0].message.content)
```

**队列状态：**

```bash
curl http://localhost:5000/api/queue/status
# {"busy":false,"pending_requests":0,"total_processed":42}
```

## 思维链 / 推理模式

支持思维链模式的模型（Qwen 3、Qwen 3.5、Gemma 4、GPT OSS、Nemotron-H）可以在生成最终答案之前产出结构化的思维链推理内容。思维内容与主要回复分开，客户端可选择显示或隐藏。

- **Qwen 3 / Qwen 3.5 / Nemotron-H：** 使用 `<think>...</think>` 标签
- **Gemma 4：** 使用 `<|channel>thought\n...<channel|>` 标签
- **GPT OSS：** 使用 Harmony 格式，以 `<|channel|>analysis` 标记思维过程，以 `<|channel|>final` 标记最终回复

通过 `--think`（控制台）、`"think": true`（Ollama API）或 Web 界面中的思维链开关启用。

## 工具调用 / 函数调用

模型可以调用用户定义的工具并参与多轮工具调用对话。将工具定义为 JSON 格式，通过 `--tools`（控制台）或 API 中的 `tools` 参数传入。

各架构使用各自的工具调用格式：

- **Qwen 3 / Qwen 3.5 / Nemotron-H：** `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **Gemma 4：** `<|tool_call>call:function_name{args}<tool_call|>`

输出解析器（`OutputParser.cs`）会自动从模型原始输出中提取工具调用，与架构无关。

## 多模态支持

### Gemma 4

Gemma 4 模型支持图像、视频和音频输入。将多模态投影器（`gemma-4-mmproj-F16.gguf`）放在与模型文件相同目录即可自动加载。

- **图像：** PNG、JPEG
- **视频：** MP4（使用 OpenCV 以 1 fps 抽取最多 8 帧）
- **音频：** WAV（16kHz 单声道）、MP3、OGG Vorbis

### Gemma 3

Gemma 3 支持 PNG/JPEG 图像输入。将其多模态投影器（`mmproj-gemma3-4b-f16.gguf`）放在模型文件相同目录即可自动加载。

### Qwen 3.5

所有 Qwen 3.5 变体（`qwen35`、`qwen35moe` 与 `qwen3next`）共用同一个 `Qwen35Model` 实现。图像输入通过支持动态分辨率的 `Qwen35VisionEncoder` 处理；将投影器（`Qwen3.5-mmproj-F16.gguf`）放到模型 GGUF 同一目录下即可自动加载。MoE 变体（例如 Qwen3.5-35B-A3B）在 decode 时还会启用融合的 `MoEExpertsSwiGLUResidual` GGML 内核，将所有被选中的专家、可选的 shared expert 与残差加法合并到一次 GPU 计算图调度中执行。

### Mistral 3

Mistral 3 通过 Pixtral 视觉编码器支持图像输入。将多模态投影器（`mistral3-mmproj.gguf`）放在与模型文件相同目录即可自动加载。

- **图像：** PNG、JPEG

## 架构说明

TensorSharp 采用分层系统结构：

1. **TensorSharp.Core** 提供核心 `Tensor` 类型、存储抽象和可扩展的操作注册表（`Ops`）。CPU 实现使用 `System.Numerics.Vectors` 进行 SIMD 加速。

2. **TensorSharp.Runtime** 负责运行时契约与通用服务：GGUF 解析、分词（SentencePiece / BPE）、聊天模板渲染、可配置 token 采样、输出解析，以及 `IModelArchitecture`、`IPromptRenderer`、`IOutputProtocolParser`、`IMultimodalInjector`、`IKVCachePolicy`、`IBackendExecutionPlan` 等抽象。

3. **TensorSharp.Models** 实现 `ModelBase` 以及各具体模型架构和多模态辅助组件（Gemma 3/4、Qwen 3/3.5、GPT OSS、Nemotron-H、Mistral 3）。模型通过 `ModelBase.Create()` 加载，并依据 GGUF 元数据自动识别架构。

4. **TensorSharp.Backends.GGML** 通过原生 C++ 桥接库（`libGgmlOps`）注册同名操作的加速实现，并链接 [ggml](https://github.com/ggml-org/ggml)。在 macOS 上可提供 Metal GPU 计算，在 Linux 上可启用面向 NVIDIA GPU 的 GGML CUDA。操作包括原生量化 matmul（Q4_K_M、Q8_0 等），无需反量化到 FP32。

5. **TensorSharp.Server** 是 HTTP / 应用层，提供兼容 Ollama 与 OpenAI 的 REST API、浏览器聊天 UI、上传处理和 FIFO 推理队列。

6. **TensorSharp.Cli** 是控制台 / 应用层，用于本地 prompt 运行、多模态实验、prompt 检查和 JSONL 批处理。

### 性能优化

- **融合 GPU decode**（Gemma 4）：在 Metal 上将所有 Transformer 层合并为单次 GGML 计算图调度，将每个 token 的 CPU-GPU 往返从数百次降低到一次。相较逐算子调度约提升 2.6 倍。
- **整模型原生 decode**（Qwen 3）：所有 Transformer 层在一次原生调用（`TransformerModelDecode`）中完成，每层权重指针在加载阶段预解析并缓存，从 decode 热点路径中移除托管循环开销。
- **融合 Qwen 3.5 attention 层 decode**：单次 GGML 计算图为每个 FullAttention 层完成 RMSNorm + 融合 QKV + Q/gate 反交错 + 每头 QK norm + RoPE + KV 缓存追加 + flash attention + sigmoid 门控混合 + 输出投影 + 残差加法。替换了原本每层 ~2 次独立 GGML 调用与 ~6 个小型 CPU/GPU 同步点。当缓存序列长度超过 4096 token 时启用（可通过 `FUSED_ATTN_LAYER_MIN_SEQ_LEN=N` 覆盖）。
- **融合权重投影**：Q/K/V 投影融合为单次 QKV matmul；gate 与 up 投影融合为单次 gate_up matmul。
- **原生量化计算**：量化权重（Q4_K_M、Q6_K、Q8_0、IQ2_XXS、MXFP4 等）直接参与 matmul，无需展开为 FP32，节省内存与带宽。批量 `AddmmQuantBatch` 内核可在一次调度内完成对同一量化权重块的多个子矩阵 matmul。
- **批量 GPU MoE**：`MoEExpertsSwiGLUResidual`（Qwen 3.5）和 `MoEExpertsForward`（Nemotron-H）将每个 MoE 层中所有被选中的专家——以及 Qwen 3.5 中可选的 shared expert 与残差加法——合并为一次 GGML 计算图调度。
- **基于 GEMM 的视觉 patch embedding**（Qwen 3.5）：将 patch embedding 重构为并行 im2col + 矩阵乘法，把单线程标量五重嵌套循环替换为可在 GPU 上加速的 matmul。
- **优化后的纯 C# CPU 路径**：托管 GEMM 快速路径和连续 Float32 内核加速了 decode、softmax、RMSNorm、RoPE、融合激活等热点路径，同时在 CPU 加载时保持量化 GGUF 权重压缩状态。
- **环形 KV 缓存**：滑动窗口注意力层使用固定大小环形缓冲区，使内存占用不随序列长度增长。
- **KV 缓存前缀复用**：多轮对话会复用各轮之间最长的匹配 token 前缀。对 SWA 模型，截断会自动按滑动窗口大小回退，使后缀部分可以重建 SWA 上下文。

### 内存优化

- **零拷贝文件映射量化权重**（CUDA、Metal、GGML CPU）：GGUF 模型文件以内存映射方式打开，量化张量通过 host 指针缓冲区直接绑定到原生算子。这样省去了之前每张张量从磁盘复制到新分配原生堆缓冲区的过程——这一过程在 Apple Silicon 上会让大型量化模型的常驻内存几乎翻倍。例如，`Qwen3.5-35B-A3B-IQ2_XXS`（约 10 GB GGUF）在 Metal 后端的实际工作内存峰值从约 17 GB 降至约 7 GB。映射文件由操作系统的页缓存管理，必要时可换出，且在 Apple Silicon（统一内存）上不会带来推理性能损失。
- **最佳匹配内存池**：GGML 主机分配器使用 best-fit 而非 first-fit 在已池化块中检索可重用空间，避免把大块草稿内存交给小型中间张量请求，从而把工作集严格控制在合理范围内。
- **有界池保留量**：集成 GPU / CPU 内存池现在将单个保留块上限设为 64 MB，整池上限设为 32 块。结合 mmap 后的权重，可在快速复用短生命中间张量的同时限制峰值常驻内存。
- **高内存效率模型加载**：大张量直接流式加载到原生内存，避免中间托管分配。F32 权重与 norm 仍按需加载；量化权重在受支持的后端上通过 mmap 方式绑定。

## 性能数据

在 `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`（磁盘约 10 GB；256 个路由专家，每 token 激活 8 个；12 个全注意力层 + 30 个 GatedDeltaNet 循环层）上，Apple M4 Pro（24 GB 统一内存）的参考结果：

| 指标 | 优化前（`v1` 基线） | 优化后（当前分支） | 变化 |
|---|---|---|---|
| 进程峰值内存占用 | ~17 GB | **~8 GB** | **-52%** |
| TensorSharp.Server 加载后常驻内存 | ~20 GB | **~8 GB** | **-60%** |
| Decode 吞吐（warm，prefill 256 / decode 64，M4 Pro） | ~3.8 tok/s | **~10.8 tok/s** | **+2.85x** |
| Decode 延迟（warm，prefill 256 / decode 64，M4 Pro） | ~264 ms/token | **~92 ms/token** | **-65%** |

复现命令：

```bash
./TensorSharp.Cli --model Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --backend ggml_metal \
    --benchmark --bench-prefill 256 --bench-decode 64 --bench-runs 3
```

内存节省主要来自不再把 GGUF 模型文件复制到独立的原生堆缓冲区（现在文件是以 mmap 方式零拷贝绑定到 Metal command buffer 中）。Decode 吞吐量提升在很大程度上也是消除约 10 GB 重复工作集的副效应——这部分重复占用此前会在仅有 24 GB 或更少物理内存的机器上触发系统级内存压力。

## 测试

### 单元测试（xUnit）

`InferenceWeb.Tests` 覆盖无需启动服务的进程内行为：托管量化算子、KV 缓存策略、图像预处理、媒体辅助逻辑、结构化输出校验、文本上传辅助、Web UI 聊天策略、模型服务历史、模型上下文长度解析、可用后端发现等。

```bash
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj
```

### 服务端集成测试

TensorSharp.Server 的集成测试位于 `TensorSharp.Server/testdata/`。测试覆盖所有三种 API 风格（Web UI SSE、Ollama、OpenAI）、多轮对话、思维链模式、工具调用、结构化输出、队列行为、并发请求和中断支持。架构特定能力（思维链、工具调用）会自动检测，当前模型不支持时会自动跳过。

```bash
# 先启动 TensorSharp.Server，然后运行：
python3 TensorSharp.Server/testdata/test_multiturn.py
# 或
bash TensorSharp.Server/testdata/test_multiturn.sh
```

完整测试矩阵见 [TensorSharp.Server/testdata/README.md](TensorSharp.Server/testdata/README.md)。

## 作者

Zhongkai Fu

## 许可证

详见 [LICENSE](LICENSE)。

