# TensorSharp

[English](README.md) | [中文](readme_cn.md)

一个用于在本地运行大型语言模型（LLM）的 C# 推理引擎，使用 GGUF 模型文件。TensorSharp 提供控制台应用、基于 Web 的聊天界面，以及兼容 Ollama/OpenAI 的 HTTP API 以便程序化调用。

## 功能特性

- **多架构支持** —— Gemma 4、Gemma 3、Qwen 3、Qwen 3.5
- **多模态推理** —— 图像、视频和音频输入（Gemma 4）；图像输入（Gemma 3 / Qwen 3.5）
- **量化模型支持** —— 加载 Q4_K_M、Q8_0、F16 等量化格式的 GGUF 文件；执行原生量化矩阵乘法（matmul），无需反量化到 FP32
- **GPU 加速** —— 通过 GGML 使用 Apple Metal 后端，并针对 Gemma 4 decode 使用整模型融合 GPU 调度（相对逐算子调度约提升 2.6 倍）
- **兼容 Ollama 与 OpenAI API** —— 可作为现有工具链的即插即用替代端点
- **可配置采样** —— temperature、top-k、top-p、min-p、重复/存在/频率惩罚、seed、停止序列
- **聊天模板** —— 从 GGUF 元数据自动加载（Jinja2），并为不同架构提供硬编码回退模板
- **批处理** —— 控制台应用支持 JSONL 输入
- **流式输出** —— 按 token 输出（Web 通过 SSE，控制台通过 stdout）
- **专家混合（MoE）** —— 支持 Gemma 4 MoE 变体（例如 gemma-4-26B-A4B）

## 支持的模型架构

| 架构 | 示例模型 | 多模态 |
|---|---|---|
| Gemma 4 | gemma-4-E4B、gemma-4-31B、gemma-4-26B-A4B（MoE） | 图像、视频、音频 |
| Gemma 3 | gemma-3-4b | 图像 |
| Qwen 3 | Qwen3-4B | 仅文本 |
| Qwen 3.5 | Qwen3.5-9B | 图像 |

## 计算后端

| 后端 | 参数 | 说明 |
|---|---|---|
| GGML Metal | `--backend ggml_metal` | 通过 Apple Metal（macOS）进行 GPU 加速。推荐用于 Apple Silicon。 |
| GGML CPU | `--backend ggml_cpu` | 使用原生 GGML 与优化内核进行 CPU 推理。 |
| 纯 C# CPU | `--backend cpu` | 无原生依赖的可移植 CPU 推理。 |

## 项目结构

```text
TensorSharp/
├── TensorSharp/                 # 核心张量库（CPU 运算、SIMD）
├── TensorSharp.GGML/            # GGML 后端绑定（通过原生库支持 Metal/CPU）
├── TensorSharp.GGML.Native/     # 到 ggml 的原生 C++ 桥接（构建 libGgmlOps）
├── AdvUtils/                    # 工具库
├── InferenceEngine/             # 模型加载、分词和推理逻辑
│   ├── Models/
│   │   ├── Gemma3/
│   │   ├── Gemma4/              # 视觉编码器、音频编码器、MoE、融合 GPU decode
│   │   ├── Qwen3/
│   │   └── Qwen35/
│   ├── GgufReader.cs            # GGUF 文件解析器
│   ├── ModelBase.cs             # 各模型架构基类
│   ├── ChatTemplate.cs          # 聊天模板渲染（硬编码 + 来自 GGUF 的 Jinja2）
│   ├── Jinja2Template.cs        # Jinja2 模板渲染器
│   ├── SamplingConfig.cs        # 采样参数配置
│   ├── TokenSampler.cs          # Token 采样（greedy、top-k、top-p、min-p、惩罚项）
│   └── MediaHelper.cs           # 视频抽帧、音频解码
├── InferenceConsole/            # CLI 应用
├── InferenceWeb/                # Web 聊天 + API 服务（ASP.NET Core）
│   ├── wwwroot/index.html       # 聊天界面
│   └── API_EXAMPLES.md          # 详细 API 文档
└── ExternalProjects/            # 第三方依赖（ggml）
```

## 前置要求

- [.NET 8.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) 或更高版本（InferenceWeb 需要 .NET 10）
- **macOS（Metal 后端）：** 用于构建原生 GGML 库的 CMake 3.20+ 与 Xcode 命令行工具
- GGUF 模型文件（例如来自 [Hugging Face](https://huggingface.co)）

## 构建

### 构建整个解决方案

```bash
dotnet build TensorSharp.slnx
```

### 构建单独应用

```bash
# 控制台应用
dotnet build InferenceConsole/InferenceConsole.csproj

# Web 应用
dotnet build InferenceWeb/InferenceWeb.csproj
```

### 构建原生 GGML 库（macOS）

如果原生库不存在，首次执行 `dotnet build` 时会自动构建。也可以手动构建：

```bash
cd TensorSharp.GGML.Native
bash build-macos.sh
```

该过程会编译带 Metal GPU 支持的 `libGgmlOps.dylib`。构建产物会自动复制到应用输出目录。

## 使用方法

### 控制台应用

```bash
cd InferenceConsole/bin

# 文本推理
./InferenceConsole --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_metal

# 图像推理（Gemma 3/4，Qwen 3.5）
./InferenceConsole --model <model.gguf> --image photo.png --backend ggml_metal

# 视频推理（Gemma 4）
./InferenceConsole --model <model.gguf> --video clip.mp4 --backend ggml_metal

# 音频推理（Gemma 4）
./InferenceConsole --model <model.gguf> --audio speech.wav --backend ggml_metal

# 使用采样参数
./InferenceConsole --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.2 --seed 42

# 批处理（JSONL）
./InferenceConsole --model <model.gguf> --input-jsonl requests.jsonl \
    --output results.txt --backend ggml_metal
```

**命令行参数：**

| 参数 | 说明 |
|---|---|
| `--model <path>` | GGUF 模型文件路径（必填） |
| `--input <path>` | 包含用户提示词的文本文件 |
| `--input-jsonl <path>` | JSONL 批量请求文件（每行一个 JSON） |
| `--output <path>` | 将生成文本写入该文件 |
| `--image <path>` | 用于视觉推理的图像文件 |
| `--video <path>` | 用于视频推理的视频文件 |
| `--audio <path>` | 音频文件（WAV、MP3、OGG）用于音频推理 |
| `--mmproj <path>` | 多模态投影器 GGUF 文件路径 |
| `--max-tokens <N>` | 最大生成 token 数（默认：100） |
| `--backend <type>` | 计算后端：`cpu`、`ggml_cpu` 或 `ggml_metal` |
| `--temperature <f>` | 采样温度（0 = 贪心） |
| `--top-k <N>` | Top-K 过滤（0 = 关闭） |
| `--top-p <f>` | Nucleus 采样阈值（1.0 = 关闭） |
| `--min-p <f>` | 最小概率过滤（0 = 关闭） |
| `--repeat-penalty <f>` | 重复惩罚（1.0 = 无） |
| `--presence-penalty <f>` | 存在惩罚（0 = 关闭） |
| `--frequency-penalty <f>` | 频率惩罚（0 = 关闭） |
| `--seed <N>` | 随机种子（-1 = 非确定性） |
| `--stop <string>` | 停止序列（可重复指定） |
| `--test` | 运行内置测试套件 |

如果把多模态投影器文件放在模型文件同目录并使用可识别命名（例如 `gemma-4-mmproj-F16.gguf`），系统会自动检测。

**JSONL 输入格式：**

每行是一个 JSON 对象，包含 `messages`、可选 `prompt` 和可选采样参数：

```json
{"id": "q1", "messages": [{"role": "user", "content": "What is 2+3?"}], "max_tokens": 50}
{"id": "q2", "messages": [{"role": "user", "content": "Write a haiku."}], "max_tokens": 100, "temperature": 0.8}
```

### Web 应用

```bash
cd InferenceWeb/bin

# 设置环境变量并运行
MODEL_DIR=./models BACKEND=ggml_metal ./InferenceWeb
```

在浏览器中打开 `http://localhost:5000`。Web 界面支持：

- 多轮聊天
- 从 `MODEL_DIR` 中可用 GGUF 文件列表选择模型
- 上传图像、视频和音频进行多模态推理
- 通过 Server-Sent Events 进行流式 token 生成

**环境变量：**

| 变量 | 说明 |
|---|---|
| `MODEL_DIR` | GGUF 模型文件所在目录 |
| `BACKEND` | 计算后端（默认：`ggml_metal`） |
| `PORT` | HTTP 端口（默认：`5000`） |

### HTTP API

InferenceWeb 暴露三种 API 风格。完整文档及 curl/Python 示例见 [API_EXAMPLES.md](InferenceWeb/API_EXAMPLES.md)。

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
```

**兼容 OpenAI 的 API：**

```bash
# Chat completions
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
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

## 多模态支持

### Gemma 4

Gemma 4 模型支持图像、视频和音频输入。将多模态投影器（`gemma-4-mmproj-F16.gguf`）放在与模型文件相同目录即可自动加载。

- **图像：** PNG、JPEG
- **视频：** MP4（使用 OpenCV 以 1 fps 抽取最多 8 帧）
- **音频：** WAV（16kHz 单声道）、MP3、OGG Vorbis

### Gemma 3 / Qwen 3.5

这两类模型支持图像输入，并需要对应的多模态投影器文件。

## 架构说明

TensorSharp 采用分层系统结构：

1. **TensorSharp** 提供核心 `Tensor` 类型、存储抽象和可扩展的操作注册表（`Ops`）。CPU 实现使用 `System.Numerics.Vectors` 进行 SIMD 加速。

2. **TensorSharp.GGML** 通过原生 C++ 桥接库（`libGgmlOps`）注册同名操作的 GPU 加速实现，并链接 [ggml](https://github.com/ggml-org/ggml)。在 macOS 上可提供 Metal GPU 计算。操作包括原生量化 matmul（Q4_K_M、Q8_0 等），无需反量化到 FP32。

3. **InferenceEngine** 实现模型相关逻辑：GGUF 解析、分词（SentencePiece BPE）、聊天模板渲染（来自 GGUF 元数据的 Jinja2 + 硬编码回退）、可配置 token 采样，以及各架构前向计算。模型通过 `ModelBase.Create()` 加载，并依据 GGUF 元数据自动识别架构。

4. **InferenceConsole** 与 **InferenceWeb** 是应用层，负责 I/O 和用户交互。InferenceWeb 同时提供兼容 Ollama 与 OpenAI 的 REST API，以及浏览器聊天 UI。

### 性能优化

- **融合 GPU decode**（Gemma 4）：在 Metal 上将所有 Transformer 层合并为单次 GGML 计算图调度，将每个 token 的 CPU-GPU 往返从数百次降低到一次。相较逐算子调度约提升 2.6 倍。
- **融合权重投影**：Q/K/V 投影融合为单次 QKV matmul；gate 与 up 投影融合为单次 gate_up matmul。
- **原生量化计算**：量化权重（Q4_K_M、Q6_K、Q8_0 等）直接参与 matmul，无需展开为 FP32，节省内存与带宽。
- **环形 KV 缓存**：滑动窗口注意力层使用固定大小环形缓冲区，使内存占用不随序列长度增长。
- **高内存效率模型加载**：大张量直接流式加载到原生内存，避免中间托管内存分配。

## 作者

Zhongkai Fu

## 许可证

详见 [LICENSE](LICENSE)。
