# TensorSharp.Server 集成测试

[English](README.md) | [中文](README_zh-cn.md)

这些测试套件用于覆盖 TensorSharp.Server 当前对外的兼容接口：

- Web UI SSE：`/api/chat`
- Ollama 聊天兼容接口：`/api/chat/ollama`
- OpenAI Chat Completions 兼容接口：`/v1/chat/completions`

测试脚本会自动检测当前加载模型的架构，并在该模型不支持思维链或工具调用时自动跳过相关用例。它们主要覆盖自回归兼容行为；DiffusionGemma Web UI 的整条消息 `replace` 去噪预览帧，需等专门的 diffusion 套件加入后才会覆盖。

## 当前套件状态

| 接口 | 覆盖内容 |
|---|---|
| Web UI SSE | 会话级流式输出、队列状态兼容事件、done 事件指标、中断处理 |
| Ollama 兼容 | 聊天流式/非流式、多轮历史、思维链、工具调用请求链路 |
| OpenAI 兼容 | Chat Completions 流式/非流式、工具调用、结构化输出、校验错误 |
| 运维行为 | 连续批处理并发、队列状态兼容、混合 API 切换、按架构自动跳过 |
| DiffusionGemma | 当前兼容脚本只覆盖通用端点形状；实时去噪预览需要专门的 Web UI SSE 测试 |

## 快速开始

1. 在仓库根目录下载推荐的公开 Gemma 4 E4B 模型，并通过已验证的原生 GGML
   快速路径启动 TensorSharp.Server，使用
   [ggml-org 仓库](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)
   中的 E4B Q8_0 家族。此方式需要完整的 [.NET 10 SDK](../../DEVELOPMENT_zh-cn.md#安装-net-10-sdk)
   而不只是 Runtime，此外还需要 Git 与 Python（用于 `hf` 下载命令行工具）。复制并运行这些命令约需 30 秒；下载 7.48 GiB 模型以及
   首次还原与原生构建耗时更长，取决于网络速度与机器性能。下面的可复制命令
   面向 Linux + NVIDIA；其他后端选择见代码块之后：

```bash
python -m pip install -U huggingface_hub
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf --local-dir models
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON dotnet build TensorSharp.slnx -c Release -p:TensorSharpSkipMlxNative=true
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll \
  --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --max-tokens 128
```

Windows/Linux + NVIDIA 使用 `ggml_cuda`；Apple Silicon 使用 `ggml_metal`；
Windows/Linux 上带 Vulkan 驱动的 AMD、Intel 或 NVIDIA GPU 使用 `ggml_vulkan`。
已验证的说法限定为 E4B Q8_0 家族与执行路径，不声称基准输入对应某个公开文件的精确 SHA。

纯文本检查不需要投影器。图像、视频或音频检查需要同一仓库中的
`mmproj-gemma-4-E4B-it-Q8_0.gguf`，并在服务启动时传入
`--mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf`。

Windows/Linux + NVIDIA 可使用 `--backend cuda` 或 `--backend ggml_cuda`，Windows/Linux + AMD/Intel/NVIDIA GPU 可使用 `--backend ggml_vulkan`（需启用 Vulkan 的原生构建），macOS 可使用 `--backend ggml_metal` 或 `--backend mlx`，CPU 测试可使用 `--backend ggml_cpu` 或 `--backend cpu`。

服务始终必须在启动时传入 `--model`；无模型服务无法通过
`/api/models/load` 选择 GGUF。多模态测试还必须显式传入 `--mmproj`。

2. 运行任一套件：

```bash
cd TensorSharp.Server/testdata

# Bash 套件（依赖 curl + jq）
bash test_multiturn.sh

# Python 套件（仅使用标准库）
python3 test_multiturn.py
```

## 测试套件覆盖范围

### 通用覆盖项

- Web UI 多轮 SSE 流式响应及结束事件
- Ollama 聊天的多轮行为（流式与非流式）
- OpenAI Chat Completions 的流式与非流式响应
- OpenAI 结构化输出：同时支持 `response_format: {"type":"json_object"}` 与 `response_format.json_schema`
- 队列状态接口的字段结构
- 必填字段缺失时的错误处理
- 结构化输出校验错误及文档化的请求冲突情况

### 按能力门控的覆盖项

- 思维链测试仅在当前 TensorSharp 中支持思维链的架构上运行：
  Gemma 4、Qwen 3、Qwen 3.5、GPT OSS、Nemotron-H
- 工具调用测试仅在当前 TensorSharp 中支持工具调用的架构上运行：
  Gemma 4、Qwen 3、Qwen 3.5、Nemotron-H
- GPT OSS 思维链会被测试，但当前脚本会跳过 GPT OSS 工具调用检查：脚本的能力门控（`test_multiturn.py` 中的 `_detect_capabilities`）对 `gptoss` 仍报告 `tools=False`，这相对于服务端已经过时 —— Harmony 输出解析器实际支持工具调用（commentary channel 的 `to=functions.NAME` 调用），因此这里的跳过并不代表不支持。

不支持的架构会被标记为 `SKIP`，而不是 `FAIL`。

### 仅 Bash 的运维侧检查

- Web UI 流程中的 system prompt 持久化
- 通过连续批处理引擎处理并发请求
- 队列状态兼容字段
- 长对话压力测试
- Ollama / OpenAI 接口混用
- 生成中途中断与请求清理
- Ollama 工具调用请求路径

### 仅 Python 的兼容性检查

- 按架构感知地校验 OpenAI 工具调用
- 独立的通过/失败/跳过统计，并按用例输出 payload

## 注意事项

- 本目录中的 OpenAI 覆盖范围针对的是 Chat Completions 兼容接口。OpenAI 较新的 Responses API 不在 TensorSharp.Server 当前模拟的兼容范围内。
- 结构化输出遵循 Chat Completions 的 `response_format` 协议。`json_schema` 与 `tools` 或 `think` 同时使用时预期返回 HTTP `400`。
- Ollama 与 OpenAI 兼容方案仍在持续演进。这些脚本与服务端当前的契约以及在思维链、工具调用、结构化输出方面的文档化行为保持一致。
- DiffusionGemma 可以通过 append-oriented 兼容端点返回最终文本，但只有 Web UI `/api/chat` 会暴露实时去噪 `replace` 帧。
- 浏览器 UI 位于 `http://localhost:5000/index.html`；`GET /` 是存活检查接口。

## 使用方法

### Bash

```bash
bash test_multiturn.sh [model_name] [base_url]
```

示例：

```bash
bash test_multiturn.sh
bash test_multiturn.sh gemma-4-E4B-it-Q8_0.gguf
bash test_multiturn.sh gemma-4-E4B-it-Q8_0.gguf http://host:5000
```

### Python

```bash
python3 test_multiturn.py [--model MODEL] [--url URL] [--max-tokens N]
```

示例：

```bash
python3 test_multiturn.py
python3 test_multiturn.py --model gemma-4-E4B-it-Q8_0.gguf
python3 test_multiturn.py --max-tokens 120
```
