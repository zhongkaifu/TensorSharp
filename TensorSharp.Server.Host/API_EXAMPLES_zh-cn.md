# TensorSharp.Server.Host API 示例

[English](API_EXAMPLES.md) | [中文](API_EXAMPLES_zh-cn.md)

TensorSharp.Server.Host 提供三种 API 风格以及若干工具型接口：

- **兼容 Ollama**（`/api/generate`、`/api/chat/ollama`、`/api/tags`、`/api/show`）
- **兼容 OpenAI**（`/v1/chat/completions`、`/v1/responses`、`/v1/models`）
- **Web UI**（`/api/chat`、`/api/sessions`、`/api/models`、`/api/models/load`、`/api/upload`、`/api/skills`、`/api/image-edit`、`/api/image-edit/stream`）
- **工具型接口**（`/api/version`、`/api/queue/status`）

启动服务时通过 `--model` 指定承载的模型文件，必要时通过 `--mmproj` **显式**指定多模态投影器；`TensorSharp.Server.Host` 不会自动探测投影器。Web UI 与兼容接口仅暴露启动时指定的模型 / 投影器组合；`/api/models/load` 可以用受支持的后端重新加载同一组合，但无模型启动时不能用它选择模型，也不能在运行时切换到其他文件。

## 当前契约

| 范围 | 契约 |
|---|---|
| 承载模型 | 单个 GGUF 文件，通过 `--model` 选择；请求中的 `model` 必须是该文件名或 basename |
| 投影器 | 可选单个投影器，通过 `--mmproj` 显式选择；供多模态模型使用 |
| 后端 | `mlx`、`cuda`、`ggml_metal`、`ggml_cuda`、`ggml_vulkan`、`ggml_cpu`、`cpu`；`/api/models` 会返回当前主机可用项 |
| 并发 | 自回归聊天使用连续批处理引擎。旧队列 API 只保留状态 / 兼容字段；DiffusionGemma Web UI 请求使用独立的 block 边界 diffusion scheduler。 |
| 生成模式 | 自回归模型流式追加 token chunk。DiffusionGemma 在 append-only 兼容端点返回最终文本，在 Web UI `/api/chat` 上提供整条消息替换式实时去噪预览。 |
| 会话 | Web UI 使用每个浏览器 tab 独立聊天会话。Ollama/OpenAI 兼容端点保留现有的默认推理会话行为，但代码执行工作区绝不会跨 HTTP 请求延续。 |
| 上传 | `/api/upload` 接受图像 / 视频 / 音频 / 文本 / **PDF** 文件；原生数字 PDF 返回抽取出的文本，扫描版 PDF 在加载了具备视觉能力的模型时返回逐页图像（`TS_PDF_MAX_PAGES` 限制读取页数） |
| 图像编辑 | Qwen-Image-Edit（`qwen_image`）模型通过 `/api/image-edit` 与 `/api/image-edit/stream` 提供服务，而不是聊天端点 |
| 视频生成 | 任何视频生成模型 —— MiniMax-H3（`minimax-h3`）、Wan 2.1 / 2.2（`wan`）—— 都通过 `/api/video-generate`、`/api/video-generate/stream` 与 `/v1/videos/generations` 提供服务；MiniMax-H3 在 MP4 之外还会返回一个 32 kHz 立体声 `.wav` 旁挂文件，`/api/models` 会告知当前加载的检查点接受哪些条件输入 |
| Agent Skills | 技能目录来自 `--skills-dir`（或二进制文件旁的 `skills` 目录），在 `/v1/skills` 与 `/api/skills` 列出，也可通过 `POST /api/skills` 以 `.zip` 安装。所有聊天端点都可用 `"skills": [...]` 按请求选中。对同时支持工具声明与输出解析的模型族，模型自己的技能调用在服务端内部应答，因此客户端拿到完整回复；`qwen4exp` 等无工具模型族则以内联方式获得选中技能说明。`skills_run` 只有在服务启动时传入 `--skills-allow-exec` 才可用。 |
| Agent 式代码执行 | `--code-exec` 会为支持工具调用的模型族加入进程内执行的 `shell`、`read_file`、`edit_file`、`write_file` 与 `apply_patch`。Web UI 每个聊天会话保留一个工作区；每个 OpenAI/Ollama HTTP 请求在内部轮次间使用私有工作区，响应结束后由服务删除。联网与安装软件包是相互独立且默认关闭的权限。 |
| 结构化输出 | OpenAI `response_format` 支持 `text`、`json_object`、`json_schema`；`response_format`（`json_object` / `json_schema`）不能与 `think` 或 `tools` 同时使用 |

> **网络安全：**服务监听 `0.0.0.0:5000`，没有 API Key 身份验证或内置 TLS。
> 只应在可信网络中使用，或在前方部署带身份验证与 TLS 的反向代理。

## 启动服务

### 约 30 秒快速开始

已验证的快速路径是在原生 GGML 后端上运行 Gemma 4 E4B Q8_0。下面的命令复制并运行大约只需 30 秒；7.48 GiB 的模型下载与首次 restore/构建耗时更长，取决于网络速度与机器性能。除 [.NET 10 SDK](../DEVELOPMENT_zh-cn.md#安装-net-10-sdk)、Git 与 `curl` 外，这条路径还需要所选后端对应的常规原生 GGML 构建依赖。全新机器请先按链接中的 Windows、macOS 或 Linux 说明安装 SDK；仅安装 Runtime 无法构建 TensorSharp。模型是推荐的公开制品，来自 [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)；同一仓库还提供更省内存的 `gemma-4-E4B-it-Q4_K_M.gguf`。下面的可复制命令面向 Linux + NVIDIA；其他平台的后端选择见代码块之后：

```bash
git clone https://github.com/zhongkaifu/TensorSharp.git
cd TensorSharp
mkdir -p models
curl -L --fail "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true" \
  -o models/gemma-4-E4B-it-Q8_0.gguf
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON dotnet run --project TensorSharp.Server.Host -c Release \
  -p:TensorSharpSkipMlxNative=true -- \
  --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --max-tokens 128
```

Windows/Linux + NVIDIA 使用 `ggml_cuda`；Apple Silicon 使用 `ggml_metal`；
Windows/Linux 上带 Vulkan 驱动的 AMD、Intel 或 NVIDIA GPU 使用 `ggml_vulkan`
（改为设置 `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON`）；没有 GPU 时使用 `ggml_cpu`。
这里验证的是 E4B Q8_0 家族与执行路径，不声称基准输入对应某个公开文件的特定校验和。

纯文本 API 请求不需要投影器。图像、视频或音频输入还需从同一仓库下载
`mmproj-gemma-4-E4B-it-Q8_0.gguf`，并在重启时传入
`--mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf`。

在第二个终端中运行：

```bash
curl -s http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-E4B-it-Q8_0.gguf","messages":[{"role":"user","content":"Reply with one short hello."}],"max_tokens":32}'
```

内置 UI 的地址是 **<http://localhost:5000>** —— `GET /` 直接返回 `index.html`（显式的 `/index.html` 地址同样可用）。`GET /health` 是存活检查接口，返回 `"TensorSharp.Server.Host is running"`；只有在没有 `wwwroot` 内容的无界面部署中，`GET /` 才会返回同样的响应。

### 已构建或已解压的应用目录

构建完成后可从仓库根目录运行下面的命令，也可把 DLL 路径改为解压后的发行归档；应用目录同时包含原生库与 `wwwroot/`。**状态核验于 2026-09-01：**[v3.3.0.0](https://github.com/zhongkaifu/TensorSharp/releases/tag/v3.3.0.0) 提供十个预构建归档——Windows x64（CPU/CUDA）、Linux x64（CPU/CUDA）与 macOS arm64 各有 CLI 和 Server。更新版本请查看 [Releases 页面](https://github.com/zhongkaifu/TensorSharp/releases)。

```bash
# 仅文本模型
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ~/work/model/Qwen3.5-9B-Q8_0.gguf --backend ggml_metal

# Windows/Linux + NVIDIA，Direct CUDA/cuBLAS 后端
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ~/work/model/Qwen3.5-9B-Q8_0.gguf --backend cuda

# Windows/Linux + NVIDIA，GGML CUDA 后端
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ~/work/model/Qwen3.5-9B-Q8_0.gguf --backend ggml_cuda

# Windows/Linux + AMD/Intel/NVIDIA GPU，GGML Vulkan 后端（多 GPU 主机用 --gpu-device 选择设备；见 --list-gpus）
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ~/work/model/Qwen3.5-9B-Q8_0.gguf --backend ggml_vulkan --gpu-device 0

# Apple Silicon，MLX 后端
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ~/work/model/Qwen3.5-9B-Q8_0.gguf --backend mlx

# 多模态模型（显式指定投影器）
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ~/work/model/gemma-4-E4B-it-Q8_0.gguf \
    --mmproj ~/work/model/mmproj-gemma-4-E4B-it-Q8_0.gguf --backend ggml_metal

# DiffusionGemma 文本扩散模型
DIFFUSION_STEPS=48 DIFFUSION_MAX_BATCH=2 \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ~/work/model/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggml_metal

# 覆盖默认 token 预算（默认 20000）。它对每个端点都生效 —— Web UI、Ollama
# 与 OpenAI —— 只要请求省略了 max_tokens / num_predict 就采用该值，并且会把
# 要得更多的请求钳制到该值。
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ~/work/model/Qwen3.5-9B-Q8_0.gguf --backend ggml_metal --max-tokens 4096
```

API 默认监听 `http://localhost:5000`；Web UI 就在同一个根地址上提供。
可以用 `--port` 修改监听端口（用 `--host`
限制绑定的网卡），也可以使用 `PORT` / `HOST` 环境变量——Docker Space 镜像即设置
了 `PORT=7860`：

```bash
# macOS 注意：5000 端口已被 AirPlay 接收器占用，请换一个端口。
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model <model.gguf> --backend ggml_metal --port 8080

# 仅绑定环回地址，使服务无法被其他机器访问。
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model <model.gguf> --host 127.0.0.1 --port 8080
```

推理必须在启动时提供 `--model`。只传 `--backend` 可以启动一个无模型的状态服务，
但 `/api/models/load` 无法选择启动时未提供的文件。多模态推理始终需要显式传入
`--mmproj`；只写投影器文件名时，会相对于模型所在目录解析。

### 服务端 Agent 式代码执行

代码执行必须显式开启。下面是一条较保守的本地启动命令：启用五个内置工具，
同时只监听环回地址，并让模型生成的命令保持离线。Linux 需要 `bwrap` 0.12.0
或更高版本，macOS 则使用系统自带的 Seatbelt 沙箱：

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model <model.gguf> --backend ggml_cpu --host 127.0.0.1 --code-exec
```

`--code-exec-allow-install` 会另行允许由宿主校验并代办的 pip/npm 安装；
`--code-exec-allow-network` 则给予模型生成命令不受限的宿主 IP 网络访问。
两者默认都关闭。Windows 的 Job Object 无法限制文件系统或网络，所以在 Windows
上还必须显式传入逃生开关 `--code-exec-unconfined`；可由不受信任用户访问的服务端
不应使用这组配置。

内置代码与技能工具都在 TensorSharp 内部执行，绝不会作为待执行调用返回给 API
客户端；调用者自己定义的工具仍归调用者所有并照常返回。Web UI 对话在整个聊天会话
内保留工作区。每个 `/v1/chat/completions`、`/v1/responses`、
或 `/api/chat/ollama` HTTP 请求则获得一个只在内部 Agent 轮次间
存活的私有工作区，响应结束后删除。文件与宿主安装的软件包会在这段寿命内可用；
不要依赖 virtualenv 激活状态、PATH 修改或常驻 shell 在调用之间延续。

捕获到的输出文件会复制到服务端制品存储，并通过
`/api/code/artifacts/{runId}/...` 下仅供下载的 URL 暴露。Web UI SSE 的完成元数据
包含 `files: [{name, bytes, url}]`，因此即使模型没有在正文中复述链接，用户也能下载。
完整参数与沙箱说明见 [USAGE](../USAGE_zh-cn.md#代码执行shell-工具)。

后端速查：

| 值 | 含义 |
|---|---|
| `cpu` | 纯 C# CPU 后端 |
| `cuda` | Direct CUDA 后端，使用 CUDA Driver API、cuBLAS、PTX 内核与 CPU 回退 |
| `mlx` | Apple Silicon 上的 MLX Metal 后端 |
| `ggml_cpu` | 原生 GGML CPU 后端 |
| `ggml_metal` | macOS 的 GGML Metal 后端 |
| `ggml_cuda` | NVIDIA GPU 的 GGML CUDA 后端 |
| `ggml_vulkan` | AMD / Intel / NVIDIA GPU 的 GGML Vulkan 后端（与厂商无关；需要在原生构建时启用 Vulkan） |

---

## 1. 兼容 Ollama 的 API

### 列出模型

```bash
curl http://localhost:5000/api/tags
```

响应：
```json
{
  "models": [
    {"name": "Qwen3.5-9B-Q8_0", "model": "Qwen3.5-9B-Q8_0.gguf", "size": 9550000000, "modified_at": "2025-03-15T10:00:00Z"}
  ]
}
```

### 查看模型信息

```bash
curl -X POST http://localhost:5000/api/show \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3.5-9B-Q8_0.gguf"}'
```

### 生成（非流式）

```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q8_0.gguf",
    "prompt": "What is 1+1?",
    "stream": false,
    "options": {
      "num_predict": 50,
      "temperature": 0.7,
      "top_p": 0.9
    }
  }'
```

响应：
```json
{
  "model": "Qwen3.5-9B-Q8_0.gguf",
  "created_at": "2025-03-15T10:00:00Z",
  "response": "1+1 equals 2.",
  "done": true,
  "done_reason": "stop",
  "total_duration": 1500000000,
  "prompt_eval_count": 15,
  "prompt_eval_duration": 300000000,
  "eval_count": 10,
  "eval_duration": 1200000000,
  "prompt_cache_hit_tokens": 0,
  "prompt_cache_hit_ratio": 0.0
}
```

`prompt_cache_hit_tokens` 表示在 `prompt_eval_count` 个 token 中，有多少 token
是直接从上一轮的 KV 缓存中读取的。`/api/generate` 在每次 prefill 之前都会重置
会话，因此该字段始终为 `0`；在 `/api/chat/ollama` 上，当本次请求的 prompt 前
缀与上一轮匹配时，该字段会变为非 0。

### 生成（流式）

```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q8_0.gguf",
    "prompt": "Tell me a joke.",
    "stream": true,
    "options": {"num_predict": 100}
  }'
```

每一行都是一条 JSON（newline-delimited JSON）：
```
{"model":"Qwen3.5-9B-Q8_0.gguf","created_at":"...","response":"Why","done":false}
{"model":"Qwen3.5-9B-Q8_0.gguf","created_at":"...","response":" did","done":false}
...
{"model":"Qwen3.5-9B-Q8_0.gguf","created_at":"...","response":"","done":true,"done_reason":"stop","total_duration":...,"eval_count":...,"prompt_cache_hit_tokens":0,"prompt_cache_hit_ratio":0.0}
```

末尾的 `done` chunk 与非流式响应一样，也会携带 `prompt_cache_hit_tokens` /
`prompt_cache_hit_ratio` 字段。

### 带图片的生成（多模态）

图片以 base64 字节序列传入 `images` 数组：

```bash
IMG_B64=$(base64 < photo.png | tr -d '\n')
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"gemma-4-E4B-it-Q8_0.gguf\",
    \"prompt\": \"What is in this image?\",
    \"images\": [\"$IMG_B64\"],
    \"stream\": false,
    \"options\": {\"num_predict\": 200}
  }"
```

### 聊天（非流式）

```bash
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q8_0.gguf",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "stream": false,
    "options": {"num_predict": 100}
  }'
```

响应：
```json
{
  "model": "Qwen3.5-9B-Q8_0.gguf",
  "created_at": "2025-03-15T10:00:00Z",
  "message": {"role": "assistant", "content": "The capital of France is Paris."},
  "done": true,
  "done_reason": "stop",
  "total_duration": 2000000000,
  "prompt_eval_count": 20,
  "prompt_eval_duration": 500000000,
  "eval_count": 15,
  "eval_duration": 1500000000,
  "prompt_cache_hit_tokens": 0,
  "prompt_cache_hit_ratio": 0.0
}
```

`prompt_cache_hit_tokens` 与 `prompt_cache_hit_ratio` 表示有多少 prompt token
是直接复用了上一轮的 KV 缓存。新会话的第一轮两个值都是 0；在复用上一轮
prefix 的后续轮次中，它们会接近 `prompt_eval_count` / `1.0`。流式模式下末尾
chunk 同样携带这些字段。

### 聊天（流式）

```bash
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q8_0.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true,
    "options": {"num_predict": 50}
  }'
```

### 多轮聊天

```bash
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q8_0.gguf",
    "messages": [
      {"role": "user", "content": "My name is Alice."},
      {"role": "assistant", "content": "Nice to meet you, Alice!"},
      {"role": "user", "content": "What is my name?"}
    ],
    "stream": false,
    "options": {"num_predict": 50}
  }'
```

### 带图片的聊天（多模态）

```bash
IMG_B64=$(base64 < photo.png | tr -d '\n')
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"gemma-4-E4B-it-Q8_0.gguf\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": \"Describe this image.\",
      \"images\": [\"$IMG_B64\"]
    }],
    \"stream\": false,
    \"options\": {\"num_predict\": 200}
  }"
```

### 聊天 + 思维链 / 推理模式

支持思维链的架构（Qwen 3.5/3.6/3.8-family、Gemma 4、GPT OSS、Nemotron-H）可接受 `"think": true`，并将思考过程与可见回答分开返回：

```bash
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q8_0.gguf",
    "messages": [{"role": "user", "content": "Solve 17 * 23 step by step."}],
    "think": true,
    "stream": false,
    "options": {"num_predict": 200}
  }'
```

响应中思维过程位于 `message.thinking`：

```json
{
  "message": {
    "role": "assistant",
    "content": "17 * 23 = 391.",
    "thinking": "17 * 20 = 340. 17 * 3 = 51. 340 + 51 = 391."
  },
  "done": true,
  "done_reason": "stop"
}
```

### 聊天 + 工具调用

工具按 Ollama tool API 的形式定义。服务端会根据当前架构识别工具调用的线协议（如 Qwen / Nemotron-H 使用 `<tool_call>...</tool_call>`，Gemma 4 使用 `<|tool_call>...<tool_call|>`，GPT OSS 使用 Harmony commentary channel `<|channel|>commentary to=functions.NAME ...<|call|>`），并解析为结构化的 `tool_calls`：

```bash
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q8_0.gguf",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取某城市的当前天气。",
        "parameters": {
          "type": "object",
          "properties": {
            "city":  {"type": "string", "description": "目标城市"},
            "units": {"type": "string", "enum": ["c", "f"]}
          },
          "required": ["city"]
        }
      }
    }],
    "stream": false,
    "options": {"num_predict": 200}
  }'
```

模型决定调用工具时的响应：

```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [{
      "function": {
        "name": "get_weather",
        "arguments": {"city": "Paris", "units": "c"}
      }
    }]
  },
  "done": true,
  "done_reason": "tool_calls"
}
```

继续会话时，把 assistant 的 tool call 与一条 `role: "tool"` 的消息（包含函数返回结果）追加到 messages，再次请求 `/api/chat/ollama` 即可。

### 聊天 + Agent Skills

加上一个 `skills` 数组，即可指定本次回答应当遵循哪些 Agent Skills。前期只有每个技能
的一行描述会占用上下文；`SKILL.md` 正文与所需的参考文件由模型通过内置的
`skills_list` / `skills_read` 工具自取，而这些工具**由服务端自己执行**，因此你拿到的
是一条普通回复，而不是一个需要客户端去执行的工具调用。这套渐进披露循环同时要求
工具声明与输出解析支持。Qwen 3.8 Flash Next（`qwen4exp`）目前没有结构化工具解析器，
因此会以内联方式获得选中技能说明，也不会拿到技能或代码执行工具。

`skills_discovery` 可选，默认为 `true` —— 模型还会看到本次请求*没有*选中的那些技能的
名称与描述，以便自己挑出你没想到要点名的那个。设为 `false` 则把本次请求严格限制在它
列出的技能上。

```bash
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-Q8_0.gguf",
    "messages": [{"role": "user", "content": "把 statement.pdf 里的合计表格提取出来，并给出环比变化。"}],
    "skills": ["pdf"],
    "skills_discovery": false,
    "stream": false,
    "options": {"num_predict": 800}
  }'
```

点名一个服务端没有的技能会返回 `400`。`GET /api/skills` 可列出已注册的技能。同样这两个
字段在 `/api/chat`（Web UI）以及 `/v1/chat/completions` / `/v1/responses` 上都可用。

技能与你自己的 `tools` 可以共存：内置的技能工具会并入你发来的工具列表，TensorSharp 只
应答属于它自己的那几个，而对*你的*工具的调用照常回传给你——此时模型从技能里读到的内容
已经留在对话中。

服务端使用 `--code-exec` 启动后，同一个进程内循环还可使用 `shell`、`read_file`、
`edit_file`、`write_file` 与 `apply_patch`。这些内置调用始终留在 TensorSharp 内部；
工作区、沙箱、网络、安装与制品行为见[服务端 Agent 式代码执行](#服务端-agent-式代码执行)。

---

## 2. 兼容 OpenAI 的 API

### 列出模型

```bash
curl http://localhost:5000/v1/models
```

响应：
```json
{
  "object": "list",
  "data": [
    {"id": "Qwen3.5-9B-Q8_0", "object": "model", "owned_by": "local"}
  ]
}
```

### Chat Completions（非流式）

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q8_0.gguf",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is 2+3?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

响应：
```json
{
  "id": "chatcmpl-abc123...",
  "object": "chat.completion",
  "created": 1710500000,
  "model": "Qwen3.5-9B-Q8_0.gguf",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "2 + 3 = 5."},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 8,
    "total_tokens": 28,
    "prompt_tokens_details": {
      "cached_tokens": 0
    }
  }
}
```

`usage.prompt_tokens_details.cached_tokens` 与 OpenAI 官方的 KV 缓存命中扩展字
段一致：当后续轮次复用了上一轮的 prompt 前缀时，该值会接近 `prompt_tokens`，
客户端可由此判断本轮 TTFT 节省的程度，无需打开服务端的 Debug 日志。

### Chat Completions（流式）

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q8_0.gguf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50,
    "stream": true
  }'
```

每个 chunk 以 SSE 形式发送：
```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9,"prompt_tokens_details":{"cached_tokens":0}}}

data: [DONE]
```

末尾 chunk 的 `usage` 块同样会携带 `prompt_tokens_details.cached_tokens`，与
非流式响应保持一致。

### Chat Completions + JSON 模式

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q8_0.gguf",
    "messages": [
      {"role": "user", "content": "Return a JSON object with keys answer and confidence for 2+3."}
    ],
    "response_format": {"type": "json_object"},
    "max_tokens": 80
  }'
```

响应：
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "{\"answer\":5,\"confidence\":\"high\"}"
    },
    "finish_reason": "stop"
  }]
}
```

### Chat Completions + 结构化输出（`json_schema`）

TensorSharp.Server.Host 接收 OpenAI Chat Completions 的 `response_format` 形式，会向 prompt 中注入严格 JSON 指令，并在返回前对最终输出进行校验。

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q8_0.gguf",
    "messages": [
      {
        "role": "system",
        "content": "You are a concise extraction assistant."
      },
      {
        "role": "user",
        "content": "Extract the city and country from: Paris, France."
      }
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "location_extraction",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "city": { "type": "string" },
            "country": { "type": "string" },
            "confidence": { "type": ["string", "null"] }
          },
          "required": ["city", "country", "confidence"],
          "additionalProperties": false
        }
      }
    },
    "max_tokens": 120
  }'
```

响应：
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "{\"city\":\"Paris\",\"country\":\"France\",\"confidence\":null}"
    },
    "finish_reason": "stop"
  }]
}
```

### Chat Completions + 图片（多模态，OpenAI 格式）

```bash
IMG_B64=$(base64 < photo.png | tr -d '\n')
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"gemma-4-E4B-it-Q8_0.gguf\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"What is in this image?\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$IMG_B64\"}}
      ]
    }],
    \"max_tokens\": 200
  }"
```

### Chat Completions + 工具调用

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-9B-Q8_0.gguf",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取某城市的当前天气。",
        "parameters": {
          "type": "object",
          "properties": {
            "city":  {"type": "string"},
            "units": {"type": "string", "enum": ["c", "f"]}
          },
          "required": ["city"]
        }
      }
    }],
    "max_tokens": 200
  }'
```

模型发出工具调用时，响应使用 OpenAI 风格字段：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\":\"Paris\",\"units\":\"c\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

将 assistant 的 `tool_calls` 与一条 `{"role": "tool", "tool_call_id": "...", "content": "..."}` 消息追加到 messages，即可继续工具循环。

### Chat Completions + Agent Skills

同样的 `skills` / `skills_discovery` 字段在 OpenAI 接口以及 `/v1/responses` 上同样可用：

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-Q8_0.gguf",
    "messages": [{"role": "user", "content": "把这些数字做成一张带图表的电子表格。"}],
    "skills": ["xlsx"],
    "max_tokens": 800
  }'
```

返回的是一条普通的 `chat.completion`。对支持工具调用的模型族，所有内置技能或代码
调用都发生在服务端内部；OpenAI SDK 无需改动，也不会看到它无法执行的内置调用。
调用者定义的工具仍会照常返回。包括 `qwen4exp` 在内、没有结构化工具解析器的模型族
使用选中技能内联回退，并且不会获得代码工具。

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="gemma-4-E4B-it-Q8_0.gguf",
    messages=[{"role": "user", "content": "帮我填好这个 AcroForm 表单，并说明你填了什么。"}],
    max_tokens=800,
    extra_body={"skills": ["pdf"], "skills_discovery": False},
)
print(response.choices[0].message.content)
```

提示词预算、披露循环与安全模型等设计说明见
[Agent Skills in TensorSharp](../docs/agent_skills.md)（英文）。

### 工具型接口

```bash
# 兼容旧字段的推理负载快照：并发由连续批处理引擎管理，
# pending_requests 通常为 0
curl http://localhost:5000/api/queue/status

# 旧 Ollama 协议版本（硬编码为 0.1.0，并非 TensorSharp Release 版本）
curl http://localhost:5000/api/version

# 承载模型 + 可用后端 + 默认设置
curl http://localhost:5000/api/models
```

`/api/models` 返回唯一承载的 GGUF（如有投影器一并返回），加载后的后端名、可用后端列表、解析出的架构以及配置好的默认 `max_tokens`。当承载模型会生成视频时，它还会返回一个 `video` 对象 —— `family`（`"minimax-h3"`、`"wan"`）、`supportsAudio`、`supportsImageConditioning`、`supportsEndImageConditioning`、`supportsReferenceConditioning`、`maxReferenceImages` —— 其他模型下该字段为 `null`。客户端正是靠这一块判断该不该提供首帧、尾帧或最多 N 个参考，而不必去匹配架构字符串：同样三张图片，在 MiniMax-H3 的 Ref2VA 检查点上是三个参考，在 FL2VA 上则是一个非法请求。`/api/tags`、`/v1/models`、`/api/show` 中的模型条目始终汇报通过 `--model` 实际启动的文件。如果某个 CUDA 后端没有出现在 `supportedBackends` 中，说明服务启动时未检测到可用的 NVIDIA 驱动/设备或 GGML CUDA 初始化路径；Direct `cuda` 后端在实际推理时仍需要能找到 cuBLAS。如果 `ggml_vulkan` 缺失，说明原生 GGML 桥接库未启用 Vulkan 构建，或未找到支持 Vulkan 1.3 的设备/驱动。如果 `mlx` 缺失，说明主机未检测到可用的 Apple Silicon MLX 运行时。

---

## 3. Web UI SSE（`/api/chat`）

这是内置聊天界面使用的协议，单独列在这里方便外部 Web UI 接入同一接口。每个事
件都是一个 JSON 对象，通过单条 `data: ...` SSE 帧下发。

当承载模型是 DiffusionGemma 时，该端点会使用整条消息替换帧展示实时去噪预览。
Ollama/OpenAI 兼容端点保持 append-oriented 响应形状，只接收最终文本。

### 聊天会话

Web UI 流程是按会话隔离的：每个浏览器 Tab 在加载时会创建自己的会话，并在每次
`/api/chat` 请求中携带该 `sessionId`，因此每个 Tab 都拥有独立的跟踪对话历史。
请求 KV 块与前缀复用由推理引擎管理。Ollama 与 OpenAI 兼容接口共享服务内置的兼
容历史。

```bash
# 创建一个新的会话（返回 id；只有 Web UI 流程需要该步骤）
curl -X POST http://localhost:5000/api/sessions
# {"sessionId":"a3b1c2..."}

# 销毁会话并清除其跟踪历史。引擎请求 KV 块会独立释放。
# 默认会话（__default__）不可删除；当 id 不存在时返回 404。
curl -X DELETE http://localhost:5000/api/sessions/a3b1c2...
```

在多次 `/api/chat` 请求中复用同一个 `sessionId` 会保留跟踪历史，并让引擎在下一
轮复用匹配的 prompt 前缀块（终态 SSE 帧的 `kvReusedTokens` /
`kvReusePercent` 字段会指出复用了多少）。省略 `sessionId` 或传入 `null` 可使用共
享的 `__default__` Web UI 会话；传入 `newChat: true` 会在下一轮前清除跟踪历史，
无需销毁会话。

### 流式聊天

```bash
curl -N -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hi"}],
    "maxTokens": 50,
    "sessionId": null,
    "newChat": false,
    "think": false,
    "tools": []
  }'
```

事件字段：

| 事件字段 | 触发时机 | 含义 |
|---|---|---|
| `queue_position`、`queue_pending` | 请求等待旧队列 shim 时的兼容事件 | 为旧客户端保留的队列位置字段 |
| `token` | 每个生成的 token（启用 `think` / `tools` 时为解析后的内容片段） | 流式正文 |
| `replace`、`diffusionStep`、`diffusionTotal`、`preview` | 每个 DiffusionGemma 去噪预览与最终替换 | 替换整条 assistant 消息，而不是追加 token |
| `thinking` | 解析到的思维链片段（仅当模型输出含思维链时） | 流式思维链 |
| `tool_calls` | 模型输出调用者定义的工具调用 | `{name, arguments}` 数组；内置技能/代码调用改由服务端进程内执行 |
| `tool_progress`、`tool`、`text`、`seconds`、`detail` | 进程内技能/代码调用正在写出或运行时 | 短暂的实时活动：阶段为 `writing`、`running` 或 `finished`；内置 Web UI 只保留有界的当前尾部，并在 `finished` 时清除 |
| `skill_step`、`skill`、`detail`、`ok`、`round`、`files` | 每次进程内技能或代码工具调用完成后 | 工具、目标与结果的完成元数据；生成的制品以可选的 `{name, bytes, url}` 条目出现在 `files` 中 |
| `done`、`tokenCount`、`elapsed`、`tokPerSec`、`aborted`、`error`、`sessionId`、`promptTokens`、`kvReusedTokens`、`kvReusePercent` | 末尾帧 | 终态汇总 |

末尾帧示例：

```
data: {"done":true,"tokenCount":187,"elapsed":2.143,"tokPerSec":87.23,"aborted":false,"error":null,"sessionId":"a3b...","promptTokens":512,"kvReusedTokens":420,"kvReusePercent":82.0}
```

技能步骤帧示例：

```
data: {"skill_step":"skills_read","skill":"pdf","detail":"references/forms.md","ok":true}
```

代码进度与制品帧示例：

```
data: {"tool_progress":"writing","tool":"shell","text":"{\"command\":\"python","seconds":0,"detail":null}
data: {"tool_progress":"running","tool":"shell","text":"writing report.xlsx\n","seconds":2.1,"detail":"python · 1.8 KB code"}
data: {"tool_progress":"finished","tool":"shell","text":"","seconds":2.4,"detail":null}
data: {"skill_step":"shell","skill":null,"detail":null,"ok":true,"round":2,"files":[{"name":"report.xlsx","bytes":18432,"url":"/api/code/artifacts/7f2.../report.xlsx"}]}
```

DiffusionGemma 预览帧示例：

```
data: {"replace":"A refined draft of the whole answer","diffusionStep":12,"diffusionTotal":48,"preview":true}
```

`kvReusedTokens` / `kvReusePercent` 与 Ollama 的 `prompt_cache_hit_*` 以及
OpenAI 的 `usage.prompt_tokens_details.cached_tokens` 含义一致 —— 都表示有多
少 prompt token 直接复用了对应会话上一轮的 KV 缓存。

### 文件上传（`/api/upload`）—— 图像、视频、音频、文本、PDF

```bash
# 上传文件（multipart 表单；使用表单中的第一个文件）
curl -X POST http://localhost:5000/api/upload -F "file=@report.pdf"
```

每个响应都携带 `ok, file, url, mediaType, fileName`；媒体类型按文件扩展名分类
（image / video / audio / pdf / text）。客户端随后在下一次 `/api/chat` 请求中
引用服务端返回的 `file` 文件名 —— 图像通过 `imagePaths`，抽取出的视频帧通过
`isVideo: true` + `imagePaths`，音频通过 `audioPaths`，文本内容则把返回的
`textContent` 内联进消息。

**PDF 文档**采用两段式处理：

- **原生数字 PDF**（含可选中的文本层）：文本被抽取后放入 `textContent` 返回，
  并携带 `renderedAsImages: false`、`pageCount`、`extractedPageCount`。提取的
  文本会完整返回；最终渲染的提示词会根据已加载模型的实际上下文窗口进行检查。
  提取文本的 `truncated` 始终为 `false`。旧版截断/计数字段仍以可空兼容字段
  保留；上传阶段不再仅为填充这些字段而分词。按内置 UI 的方式把它内联到聊天消息中：

```bash
curl -N -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "[File: report.pdf]\n<textContent from the upload response>\n[End of file]\nPlease analyze the attached PDF document and summarize its content.",
      "textFilePaths": ["<上传响应中的 file>"]
    }],
    "maxTokens": 500
  }'
```

- **扫描版 / 纯图像 PDF**：如果加载了具备视觉能力的模型（存在 `--mmproj` 或模
  型内置视觉编码器），页面会被渲染为图像并按视频帧的形式返回
  （`renderedAsImages: true`、`frames[]`、`frameUrls[]`）；在下
  一次 `/api/chat` 请求中把 `frames` 文件名作为 `imagePaths` 传入。没有视觉模型
  时，响应会携带 `needsVision: true` 和一条 `warning`，提示需用具备视觉能力的
  模型重启服务。

设置 `TS_PDF_MAX_PAGES` 环境变量可限制读取的 PDF 页数（默认 `0` = 全部页面）。

### 技能（`/api/skills`）

管理 Agent Skills 注册表。同一份数据有两种形态：偏 OpenAI 风格的 `/v1/skills`
（列表 + 单个技能，只读），以及 Web UI 风格的 `/api/skills`（额外提供加载错误、
上传与删除）。

```bash
# 服务端已注册的全部技能，外加那些“看起来像技能却加载失败”的目录，
# 以及当前是否接受上传。
curl http://localhost:5000/api/skills

# 单个技能，`instructions` 字段里是 SKILL.md 正文。
curl http://localhost:5000/api/skills/pdf

# OpenAI 风格的等价接口。
curl http://localhost:5000/v1/skills
curl http://localhost:5000/v1/skills/pdf
```

`/api/skills`：

```json
{
  "enabled": true,
  "installable": true,
  "skills": [
    {
      "id": "pdf",
      "object": "skill",
      "name": "pdf",
      "description": "Extract text and tables from PDF files, fill in PDF forms, and merge or split documents...",
      "license": "Apache-2.0",
      "compatibility": "Requires python3 with pypdf installed.",
      "files": [
        {"path": "scripts/extract_tables.py", "bytes": 4021, "kind": "script", "text": true},
        {"path": "references/forms.md", "bytes": 18233, "kind": "reference", "text": true}
      ],
      "bytes": 41288,
      "origin": "installed",
      "warnings": [],
      "modified": "2026-08-29T12:00:00Z"
    }
  ],
  "errors": [
    {"path": "/srv/skills/broken", "message": "SKILL.md is missing the required 'description' field"}
  ]
}
```

`/v1/skills` 把同样的对象包成 `{"object": "list", "data": [...]}`。`kind` 取值为
`script` / `reference` / `asset` / `manifest` / `other`；`origin` 为 `discovered`
表示它是扫描配置目录时发现的，为 `installed` 表示它是从这里上传安装的——只有后者
可以被删除。`warnings` 里是那些“不完全合规但仍然加载成功”的问题（`name` 与目录名
不一致、description 超过 1024 字符上限等）。

安装与删除：

```bash
# 上传技能文件夹的 .zip（pdf/SKILL.md），或其内容的 .zip（SKILL.md 位于压缩包根目录）。
# overwrite 表示替换同名的已安装技能；不传时，重名会返回 409。
curl -X POST http://localhost:5000/api/skills \
  -F "file=@pdf.zip" \
  -F "overwrite=true"

# 返回：安装后的 SkillObject，与列表接口返回的结构完全一致。

# 删除一个已安装的技能。
curl -X DELETE http://localhost:5000/api/skills/pdf
# {"removed":true}
```

上传在任何文件落盘之前就会被校验：每个 ZIP 条目都要经过与模型文件读取相同的那道
路径关卡（因此 `../../authorized_keys` 会被拒绝而不是写出去），大小按解压后的字节流
统计而不是采信条目自称的长度；若压缩包内文件超过 4096 个、单个文件解压后超过 64 MB、
整包超过 256 MB，或整体膨胀超过 200 倍，都会被拒绝。包含多个技能目录的压缩包同样会被
拒绝，而不是从中悄悄挑一个装上。

错误遵循服务端统一约定：`/api/*` 返回 `{"error": "..."}`，`/v1/*` 返回
`{"error": {"message": "...", "type": "invalid_request_error"}}`。

`GET /api/models` 会报告以上功能是否可用：

```json
"skills": { "enabled": true, "installable": true, "count": 7 }
```

服务端关闭了技能功能时该字段为 `null`，Web UI 据此决定是否显示技能控件。

### 图像编辑（`/api/image-edit`，Qwen-Image-Edit）

当通过 `--model` 承载的是 Qwen-Image-Edit DiT GGUF（架构 `qwen_image`）时，
图像 + 提示词的轮次走图像编辑端点，而不是 `/api/chat`：

```bash
# 一次性编辑（multipart）。steps=0 / cfg=0 表示自动
# （30 步 / cfg 2.5，或 Lightning LoRA 的步数 / cfg 1.0）。
curl -X POST http://localhost:5000/api/image-edit \
  -F "image=@photo.png" \
  -F "prompt=Replace the background with a sunny beach" \
  -F "steps=0" -F "cfg=0" -F "seed=42"
```

响应：

```json
{"ok": true, "url": "/uploads/edit-<guid>.png", "width": 1184, "height": 544, "elapsedSeconds": 40.4}
```

也接受 JSON body `{ "imagePath": "<file from /api/upload>", "prompt": "...",
"steps": 0, "cfg": 0, "seed": 42 }`（`imagePath` 为先前上传文件的服务端文件名；为兼容旧客户端也接受上传目录内的
文件）。流式变体通过 SSE 发送进度事件与实时去噪预览：

```bash
curl -N -X POST http://localhost:5000/api/image-edit/stream \
  -H "Content-Type: application/json" \
  -d '{"imagePath": "<file from /api/upload>", "prompt": "Replace the background with a sunny beach", "seed": 42}'
```

每步事件形如
`{"imageEdit": true, "step": 2, "total": 4, "image": "data:image/png;base64,...", "width": 1184, "height": 544}`
（`image` 预览快照只在节流后的步骤上出现，每次编辑最多 8 张），最后是一条
`{"done": true, "url": "/uploads/edit-<guid>.png", "width": 1184, "height": 544, "elapsedSeconds": 40.4}`。
对非 Qwen-Image-Edit 模型发起的请求返回 400；并发编辑由进程级锁串行执行。

### 视频生成（`/api/video-generate`、`/v1/videos/generations`）

三个端点共用同一套参数解析和同一道门禁：`POST /api/video-generate`、
`POST /api/video-generate/stream`（SSE，Web UI 聊天使用的就是它）以及 OpenAI 形态
的 `POST /v1/videos/generations`。任何实现了视频生成接口的模型都同时服务这三个端
点 —— MiniMax-H3 或 Wan 2.1 / 2.2 —— 其他模型一律返回 400，并携带
`The loaded model is not a video-generation model.`。`GET /api/models` 上的
`video` 块会说明当前承载的检查点到底接受哪些条件输入。

#### MiniMax-H3 —— 视频**与**原生 32 kHz 立体声音频

MiniMax-H3 把视频和 32 kHz 立体声音轨当作同一个打包潜变量一起去噪，因此一次请求
返回一个 MP4 **以及**写在它旁边的 `.wav`（四个协同网络的细节参见
[docs/models/minimax-h3_zh-cn.md](../docs/models/minimax-h3_zh-cn.md)）。先在一个
终端中启动服务器；文本编码器和两个 VAE 会通过扫描去噪器所在目录及其父目录自动解
析，只有当它们放在别处时才需要用 `--video-text-encoder` / `--video-vae` /
`--audio-vae` 指定 —— 另外文本编码器的 GGUF 不带分词器，需要把 `vocab.json` 与
`merges.txt` 放在它旁边：

```bash
TensorSharp.Server.Host --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_cuda \
  --video-width 640 --video-height 384 --video-steps 20 --video-frames 22
```

`--video-width`/`--video-height` 在这里比在任何其他模型上都更要紧：Web UI 自己不
发送尺寸，因此缺了它们每个片段都会以模型默认尺寸产出。帧数会对齐到 H3 的 `17k+5`
网格（5、22、39、56、73、90……），宽高向上取整到 32 的倍数，fps 不管请求写什么都
固定为 24。H3 是 CFG 蒸馏过的 —— 大于 `1.0` 的引导系数会被直接拒绝 —— 所以
`steps` 才是质量杠杆：20 是模型默认值，4-8 步是快速工作点。

再在另一个终端中：

```bash
curl -X POST http://localhost:5000/api/video-generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "一只红狐在飘落的雪中小跑，电影感",
       "width": 640, "height": 384, "frames": 22, "steps": 8, "cfg": 1.0, "seed": 42}'
```

```json
{"ok": true, "url": "/uploads/video-<guid>.mp4",
 "audioUrl": "/uploads/video-<guid>.wav",
 "width": 640, "height": 384, "frames": 22, "fps": 24, "seed": 42,
 "codec": "h264", "elapsedSeconds": 63.1}
```

（该 640×384 / 22 帧 / 8 步配置在 M5 Pro 上以 Metal 实测生成耗时 63.1 秒 —— 比
stable-diffusion.cpp 在其最佳配置下快 1.7×；256×256 下是 20.9 秒对 49.3 秒，
2.4×。）

`audioUrl`（OpenAI 形态路由上叫 `audio_url`）在模型没有产出音轨时为 `null` ——
仅视频模型、没有解析到音频 VAE 的 H3 运行，或者请求里写了
`"generateAudio": false`。音轨从不混流进 MP4：混流需要一个无法假定存在的编码器，
而 WAV 总是能写出来，客户端可以自己混流。

**一张图像意味着什么，取决于检查点。** `videoMode` 取 `t2v`、`i2v`（图像就是首
帧，并被赋予动作）、`fl2v`（首帧和尾帧）或 `ref`（为新场景提供身份与外观参考）；
省略它则从请求提供的内容推断。`i2v`/`fl2v` 需要 FL2VA 检查点，`ref` 需要 Ref2VA
—— 它们是两个不同的文件，而不是一个开关。让某个检查点去做它没有训练过的模式，会
得到 400，并携带模型自己的解释，告诉你该改加载哪个文件。

```bash
# 首尾帧，FL2VA 检查点。两个名字都是先前 /api/upload 返回的 "file"；
# 上传目录之外的任何路径都会被拒绝。
curl -X POST http://localhost:5000/api/video-generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "缓慢的电影感推镜", "videoMode": "fl2v",
       "imagePath": "<file from /api/upload>", "endImage": "<file from /api/upload>",
       "width": 640, "height": 384, "frames": 22, "steps": 20, "cfg": 1.0}'
```

参考条件生成 —— 主体的身份被延续下来，而镜头、背景和构图来自提示词 —— 需要
Ref2VA 检查点，最多接受**九个**参考，可以是静态图、片段和音轨的任意组合：

```bash
curl -X POST http://localhost:5000/api/video-generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "同一位女性坐在洒满阳光的咖啡馆窗边桌旁，全景镜头",
       "width": 640, "height": 384, "frames": 22, "steps": 20, "cfg": 1.0,
       "referenceImages": ["person.jpg", "bottle.png"], "videoMode": "ref"}'
```

在 Ref2VA 上，一个不带 `videoMode`、不带 `endImage`、也没有具名参考的普通
`imagePath` 会被当作单个参考，因此只会附带一张图片的客户端 —— 内置 Web UI 也在其
中 —— 不需要任何额外字段。

#### Wan 2.1 / 2.2 —— 仅视频

当通过 `--model` 承载的是 Wan DiT GGUF（架构 `wan` —— Wan 2.1 T2V、
Wan 2.2 TI2V-5B 或 Wan 2.2 A14B）时，输入提示词即可生成 H.264 MP4，没有音轨
（配套模型请参阅 [docs/models/wan_zh-cn.md](../docs/models/wan_zh-cn.md)）。先在一个终端中启动
服务器；下列参数为 Web UI 以及省略 `frames`/`fps` 的请求设置默认值。以模型原生的
24 fps 生成 121 帧，播放时长约为五秒：

```bash
TensorSharp.Server.Host --model Wan2.2-TI2V-5B-Q8_0.gguf --backend ggml_cuda \
  --video-frames 121 --fps 24
```

再在另一个终端中发起请求。此请求省略 `frames`/`fps`，因此采用上述启动默认值：

```bash
curl -X POST http://localhost:5000/v1/videos/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "一只可爱的猫", "size": "832x480", "seed": 7}'
```

这些启动参数是默认值，而不是上限。请求中显式提供的 `frames` 或 `fps` 会分别覆盖
对应的启动值。如果启动参数和请求字段均省略，则采用模型配方：TI2V-5B 为 49 帧 /
24 fps，其余模型为 33 帧 / 16 fps。帧数会对齐到 `4k+1`；要改变时长，请保持
模型原生 FPS 并调整 `frames`，因为仅改变 FPS 会改变播放速度。

**图生视频**（Wan 2.2 模型）：在 `"image"` 中加入 base64 编码的首帧（接受
`data:image/...;base64,` 前缀）—— 视频从该图像开始，提示词控制动作、镜头和
场景变化：

```bash
curl -X POST http://localhost:5000/v1/videos/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "猫向镜头跑来，电影感追踪镜头",
       "image": "data:image/png;base64,'"$(base64 -w0 first_frame.png)"'",
       "frames": 81, "seed": 7}'
```

响应（添加 `"response_format": "b64_json"` 可在响应中内联 MP4 字节）：

```json
{"created": 1780000000, "data": [{"url": "/uploads/video-<guid>.mp4"}],
 "width": 832, "height": 480, "frames": 81, "fps": 24, "seed": 7,
 "codec": "h264", "elapsed_seconds": 270.0}
```

可选字段：`cfg` 与 `cfg2`（各模型的官方默认值 —— TI2V-5B 为 5.0；A14B I2V
为 3.5/3.5，T2V 为 4.0/3.0，`cfg2` 是低噪专家的引导系数；Wan 2.1 为 6.0）、
`steps`（TI2V 为 50 / A14B 为 40 / Wan 2.1 为 30）、`fps`（当请求和服务启动
均未指定时，TI2V 为 24，其他模型为 16）、`sampler`（默认 `"unipc"` /
也可选 `"euler"`）、`flowShift`（官方配方）、
`negative_prompt`（默认为官方 Wan 负向提示词），以及会对齐到模型时间网格的
`frames`（Wan 为 `4k+1`，MiniMax-H3 为 `17k+5`；`1` = 静态图像）。未显式指定
`size` 而提供了 `image` 时，输出会遵循图像的纵横比。

#### 所有视频端点通用的字段

条件输入不止首帧、或者与视频一起生成音轨的模型，还接受下列字段 —— 每个都同时接受
camelCase 和 snake_case 两种拼写（两者同时出现时以 camelCase 为准），并且每个都指
向先前通过 `/api/upload` 上传的文件（上传目录之外的路径一律拒绝）：
`endImage`/`end_image`（尾帧条件），
`referenceImages`/`reference_images`、`referenceVideos`/`reference_videos`、
`referenceAudios`/`reference_audios`（数组，在提示词中用
`<Picture N>` / `<Video N>` / `<Audio N>` 指代），
`referenceVideoAudios`/`reference_video_audios`（音轨，**按下标**与
`referenceVideos` 配对），
`generateAudio`/`generate_audio`（默认 `true`；设为 `false` 可跳过音频解码），
以及 `videoMode`/`video_mode`（`t2v`、`i2v`、`fl2v` 或 `ref`；省略即表示“从这次
请求提供的内容推断”）。仅视频模型会忽略它们全部。

其余字段只接受 camelCase —— `width`、`height`、`frames`、`steps`、`cfg`、
`cfg2`、`seed`、`fps`、`flowShift`、`negativePrompt`、`sampler`、
`cfgCacheStride`、`imagePath`（先前上传的文件）以及 `image`（内联 base64 的替代
写法）。`size`、`negative_prompt` 和 `response_format` 只存在于
`/v1/videos/generations` 上。

`POST /api/video-generate` 接受相同的请求体，但以 `width`/`height` 代替 `size`，
返回 `{ ok, url, audioUrl, width, height, frames, fps, seed, codec, elapsedSeconds }`，
其中模型没有产出音轨时 `audioUrl` 为 null。流式变体
`POST /api/video-generate/stream`（Web UI 聊天使用）会发送形如
`{"videoGen": true, "step": 12, "total": 20, "phase": "denoise", "detail": ..., "elapsedSeconds": ..., "etaSeconds": ...}`
的进度事件 —— `phase` 的取值为 `text-encode`、`image-encode`（有图像条件时）、
`denoise`、`vae-decode`、`audio-decode`、`done` —— 最后发送
`{"done": true, "url": ..., "audioUrl": ..., "width": ..., "height": ..., "frames": 22, "fps": 24, "seed": ..., "codec": "h264", "elapsedSeconds": ...}`，
运行失败时则是 `{"done": true, "error": "..."}`。对非视频生成模型发起的请求返回
400，并携带 `The loaded model is not a video-generation model.`；并发生成由进程级
锁串行执行。

---

## 4. 采样选项

### Ollama 风格选项（位于 `options` 对象中）

| 参数               | 类型    | 默认值  | 描述                                   |
| ------------------ | ------- | ------- | -------------------------------------- |
| `num_predict`      | int     | 200     | 生成的最大 token 数                    |
| `temperature`      | float   | 0.8     | 采样温度（0 = 贪心）                   |
| `top_k`            | int     | 40      | Top-K 过滤（0 = 关闭）                 |
| `top_p`            | float   | 0.9     | 核采样阈值                             |
| `min_p`            | float   | 0       | 最小概率过滤                           |
| `repeat_penalty`   | float   | 1.1     | 重复惩罚（1.0 = 不惩罚）               |
| `presence_penalty` | float   | 0       | 出现惩罚                               |
| `frequency_penalty`| float   | 0       | 频率惩罚                               |
| `seed`             | int     | -1      | 随机种子（-1 = 不指定）                |
| `stop`             | array   | null    | 停止序列                               |

这些默认值是服务端配置的采样默认值（与 Ollama 兼容）。可在启动时通过对应的服
务器标志（`--temperature`、`--top-k`、`--top-p`、`--min-p`、`--repeat-penalty`、
`--presence-penalty`、`--frequency-penalty`、`--seed`）或 `TENSORSHARP_*` 环境
变量修改。默认情况下，运维方以这种方式配置过的参数优先于请求体；若希望请求中
的值优先，请以 `--sampling-precedence request` 启动服务。运维方未配置过的参数
始终取请求中的值。

### OpenAI 风格选项（位于顶层）

| 参数                | 类型        | 默认值  | 描述                                |
| ------------------- | ----------- | ------- | ----------------------------------- |
| `max_tokens`        | int         | `--max-tokens`（20000） | 生成的最大 token 数；同时接受 `max_completion_tokens` |
| `temperature`       | float       | 0.8     | 采样温度                            |
| `top_p`             | float       | 0.9     | 核采样阈值                          |
| `presence_penalty`  | float       | 0       | 出现惩罚                            |
| `frequency_penalty` | float       | 0       | 频率惩罚                            |
| `seed`              | int         | -1      | 随机种子                            |
| `stop`              | string/array| null    | 停止序列                            |
| `response_format`   | object      | null    | `text`、`json_object` 或 `json_schema` |
| `think`             | bool        | false   | 非标准扩展：启用思维链 / 推理解析（以 `reasoning_content` 返回 / 流式输出） |

`top_k`、`min_p` 与 `repetition_penalty` 在 OpenAI 接口上**不会被解析** ——
这些参数使用服务端配置的默认值。如果请求需要按调用设置它们，请改用 Ollama 或
Web UI 端点。

---

## 5. Python 客户端示例

### 使用 `requests`（Ollama 风格）

```python
import requests
import json

url = "http://localhost:5000/api/generate"
payload = {
    "model": "Qwen3.5-9B-Q8_0.gguf",
    "prompt": "What is machine learning?",
    "stream": False,
    "options": {"num_predict": 100, "temperature": 0.7}
}

resp = requests.post(url, json=payload)
print(resp.json()["response"])
```

### 使用 `requests` 流式（Ollama 风格）

```python
import requests
import json

url = "http://localhost:5000/api/generate"
payload = {
    "model": "Qwen3.5-9B-Q8_0.gguf",
    "prompt": "Tell me a story.",
    "stream": True,
    "options": {"num_predict": 200}
}

with requests.post(url, json=payload, stream=True) as resp:
    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            if not data["done"]:
                print(data["response"], end="", flush=True)
            else:
                print(f"\n[Done: {data['eval_count']} tokens]")
```

### 使用 `openai` Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen3.5-9B-Q8_0.gguf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+3?"}
    ],
    max_tokens=50,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### 使用 `openai` Python SDK + 结构化输出

```python
from openai import OpenAI
import json

client = OpenAI(base_url="http://localhost:5000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen3.5-9B-Q8_0.gguf",
    messages=[
        {"role": "user", "content": "Extract the city and country from: Tokyo, Japan."}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "location_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "country": {"type": "string"},
                    "confidence": {"type": ["string", "null"]}
                },
                "required": ["city", "country", "confidence"],
                "additionalProperties": False
            }
        }
    }
)

payload = json.loads(response.choices[0].message.content)
print(payload["city"], payload["country"], payload["confidence"])
```

### 使用 `openai` Python SDK 流式

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="Qwen3.5-9B-Q8_0.gguf",
    messages=[{"role": "user", "content": "Tell me about Python."}],
    max_tokens=200,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

注意事项：

- `response_format`（`json_object` 或 `json_schema`）当前不能与 `tools` 或 `think` 同时使用（HTTP `400`）。
- `json_object` / `json_schema` 请求会把**首个采样 token** 约束为以 `{` 开头的候选（效果等同于 llama.cpp 的 JSON grammar），使爱闲聊的模型无法在 JSON 对象前输出散文，流式首 token 时延（TTFT）因此反映 prefill 延迟而不是被过滤掉的前导文本。后续 token 正常采样。设置 `TS_JSON_FORCE_OPEN=0` 可关闭。
- 流式 `json_object` 请求会逐 token 流式返回 JSON 对象（自动剥离 Markdown 代码围栏和多余标签），因此首 token 时延（TTFT）反映的是 prefill 延迟。流式 `json_schema`（strict）请求仍会先在服务端缓存并按 schema 归一化，再以单个 chunk 发出。设置 `TS_STRUCTURED_STREAM_BUFFER=1` 可对两者强制使用旧的“全部缓存”行为。非流式请求始终归一化。
- 非法 schema 返回 HTTP `400`；非流式 / `json_schema` 输出未能通过校验则返回 HTTP `422`（已经开始的 `json_object` 流无法再更改状态码）。

---

## 6. 运行示例请求

`test_requests.jsonl` 文件包含针对所有接口的示例请求。可通过下面的脚本批量运行：

```bash
while IFS= read -r line; do
  ENDPOINT=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['endpoint'])")
  METHOD=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin)['method'])")
  BODY=$(echo "$line" | python3 -c "import sys,json; b=json.load(sys.stdin).get('body'); print(json.dumps(b) if b else '')")

  echo "=== $METHOD $ENDPOINT ==="
  if [ "$METHOD" = "GET" ]; then
    curl -s "http://localhost:5000$ENDPOINT" | python3 -m json.tool
  else
    curl -s -X POST "http://localhost:5000$ENDPOINT" \
      -H "Content-Type: application/json" \
      -d "$BODY" | head -c 500
  fi
  echo -e "\n"
done < test_requests.jsonl
```
