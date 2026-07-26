# TensorSharp.Server API 示例

[English](API_EXAMPLES.md) | [中文](API_EXAMPLES_zh-cn.md)

TensorSharp.Server 提供三种 API 风格以及若干工具型接口：

- **兼容 Ollama**（`/api/generate`、`/api/chat/ollama`、`/api/tags`、`/api/show`）
- **兼容 OpenAI**（`/v1/chat/completions`、`/v1/models`）
- **Web UI**（`/api/chat`、`/api/sessions`、`/api/models`、`/api/models/load`、`/api/upload`、`/api/image-edit`、`/api/image-edit/stream`）
- **工具型接口**（`/api/version`、`/api/queue/status`）

启动服务时通过 `--model` 指定承载的模型文件，必要时通过 `--mmproj` **显式**指定多模态投影器；`TensorSharp.Server` 不会自动探测投影器。Web UI 与兼容接口仅暴露启动时指定的模型 / 投影器组合；`/api/models/load` 可以用受支持的后端重新加载同一组合，但无模型启动时不能用它选择模型，也不能在运行时切换到其他文件。

## 当前契约

| 范围 | 契约 |
|---|---|
| 承载模型 | 单个 GGUF 文件，通过 `--model` 选择；请求中的 `model` 必须是该文件名或 basename |
| 投影器 | 可选单个投影器，通过 `--mmproj` 显式选择；供多模态模型使用 |
| 后端 | `mlx`、`cuda`、`ggml_metal`、`ggml_cuda`、`ggml_vulkan`、`ggml_cpu`、`cpu`；`/api/models` 会返回当前主机可用项 |
| 并发 | 自回归聊天使用连续批处理引擎。旧队列 API 只保留状态 / 兼容字段；DiffusionGemma Web UI 请求使用独立的 block 边界 diffusion scheduler。 |
| 生成模式 | 自回归模型流式追加 token chunk。DiffusionGemma 在 append-only 兼容端点返回最终文本，在 Web UI `/api/chat` 上提供整条消息替换式实时去噪预览。 |
| 会话 | Web UI 使用每个浏览器 tab 独立会话；Ollama/OpenAI 兼容端点共享默认会话 |
| 上传 | `/api/upload` 接受图像 / 视频 / 音频 / 文本 / **PDF** 文件；原生数字 PDF 返回抽取出的文本，扫描版 PDF 在加载了具备视觉能力的模型时返回逐页图像（`TS_PDF_MAX_PAGES` 限制读取页数） |
| 图像编辑 | Qwen-Image-Edit（`qwen_image`）模型通过 `/api/image-edit` 与 `/api/image-edit/stream` 提供服务，而不是聊天端点 |
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
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON dotnet run --project TensorSharp.Server -c Release \
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

内置 UI 的地址是 **<http://localhost:5000/index.html>**。`GET /` 是存活检查接口，返回 `"TensorSharp.Server is running"`。

### 已构建或已解压的应用目录

构建完成后，从仓库根目录运行下面的命令；它们会调用 `TensorSharp.Server/bin/TensorSharp.Server.dll`，同一输出目录也包含复制好的原生库与 `wwwroot/`。目前 `v3.0.5.0` GitHub Release 没有附带二进制资产，因此在真正发布压缩包之前，不应把“下载 Release 压缩包”写成可用路径。

```bash
# 仅文本模型
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend ggml_metal

# Windows/Linux + NVIDIA，Direct CUDA/cuBLAS 后端
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend cuda

# Windows/Linux + NVIDIA，GGML CUDA 后端
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend ggml_cuda

# Windows/Linux + AMD/Intel/NVIDIA GPU，GGML Vulkan 后端（多 GPU 主机用 --gpu-device 选择设备；见 --list-gpus）
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend ggml_vulkan --gpu-device 0

# Apple Silicon，MLX 后端
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend mlx

# 多模态模型（显式指定投影器）
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/gemma-4-E4B-it-Q8_0.gguf \
    --mmproj ~/work/model/mmproj-gemma-4-E4B-it-Q8_0.gguf --backend ggml_metal

# DiffusionGemma 文本扩散模型
DIFFUSION_STEPS=48 DIFFUSION_MAX_BATCH=2 \
  dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggml_metal

# 覆盖 Web UI 的默认 token 上限（默认 20000）。Ollama/OpenAI 兼容端点在请求
# 未提供 max_tokens / num_predict 时默认使用 200 —— 在这些端点上请按请求
# 自行设置该值。
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend ggml_metal --max-tokens 4096
```

API 默认监听 `http://localhost:5000`；Web UI 地址为
`http://localhost:5000/index.html`。当前二进制会把固定的
`http://0.0.0.0:5000` 监听地址传给 ASP.NET Core；Docker Space 文件会在镜像构建时
把这个常量改写为 `7860`。

推理必须在启动时提供 `--model`。只传 `--backend` 可以启动一个无模型的状态服务，
但 `/api/models/load` 无法选择启动时未提供的文件。多模态推理始终需要显式传入
`--mmproj`；只写投影器文件名时，会相对于模型所在目录解析。

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
    {"name": "Qwen3-4B-Q8_0", "model": "Qwen3-4B-Q8_0.gguf", "size": 4530000000, "modified_at": "2025-03-15T10:00:00Z"}
  ]
}
```

### 查看模型信息

```bash
curl -X POST http://localhost:5000/api/show \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B-Q8_0.gguf"}'
```

### 生成（非流式）

```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B-Q8_0.gguf",
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
  "model": "Qwen3-4B-Q8_0.gguf",
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
    "model": "Qwen3-4B-Q8_0.gguf",
    "prompt": "Tell me a joke.",
    "stream": true,
    "options": {"num_predict": 100}
  }'
```

每一行都是一条 JSON（newline-delimited JSON）：
```
{"model":"Qwen3-4B-Q8_0.gguf","created_at":"...","response":"Why","done":false}
{"model":"Qwen3-4B-Q8_0.gguf","created_at":"...","response":" did","done":false}
...
{"model":"Qwen3-4B-Q8_0.gguf","created_at":"...","response":"","done":true,"done_reason":"stop","total_duration":...,"eval_count":...,"prompt_cache_hit_tokens":0,"prompt_cache_hit_ratio":0.0}
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
    "model": "Qwen3-4B-Q8_0.gguf",
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
  "model": "Qwen3-4B-Q8_0.gguf",
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
    "model": "Qwen3-4B-Q8_0.gguf",
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
    "model": "Qwen3-4B-Q8_0.gguf",
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

支持思维链的架构（Qwen 3、Qwen 3.5/3.6-family、Gemma 4、GPT OSS、Nemotron-H）可接受 `"think": true`，并将思考过程与可见回答分开返回：

```bash
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B-Q8_0.gguf",
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
    "model": "Qwen3-4B-Q8_0.gguf",
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
    {"id": "Qwen3-4B-Q8_0", "object": "model", "owned_by": "local"}
  ]
}
```

### Chat Completions（非流式）

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B-Q8_0.gguf",
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
  "model": "Qwen3-4B-Q8_0.gguf",
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
    "model": "Qwen3-4B-Q8_0.gguf",
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
    "model": "Qwen3-4B-Q8_0.gguf",
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

TensorSharp.Server 接收 OpenAI Chat Completions 的 `response_format` 形式，会向 prompt 中注入严格 JSON 指令，并在返回前对最终输出进行校验。

```bash
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B-Q8_0.gguf",
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
    "model": "Qwen3-4B-Q8_0.gguf",
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

`/api/models` 返回唯一承载的 GGUF（如有投影器一并返回），加载后的后端名、可用后端列表、解析出的架构以及配置好的默认 `max_tokens`。`/api/tags`、`/v1/models`、`/api/show` 中的模型条目始终汇报通过 `--model` 实际启动的文件。如果某个 CUDA 后端没有出现在 `supportedBackends` 中，说明服务启动时未检测到可用的 NVIDIA 驱动/设备或 GGML CUDA 初始化路径；Direct `cuda` 后端在实际推理时仍需要能找到 cuBLAS。如果 `ggml_vulkan` 缺失，说明原生 GGML 桥接库未启用 Vulkan 构建，或未找到支持 Vulkan 1.3 的设备/驱动。如果 `mlx` 缺失，说明主机未检测到可用的 Apple Silicon MLX 运行时。

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
| `tool_calls` | 模型输出工具调用 | `{name, arguments}` 数组 |
| `done`、`tokenCount`、`elapsed`、`tokPerSec`、`aborted`、`error`、`sessionId`、`promptTokens`、`kvReusedTokens`、`kvReusePercent` | 末尾帧 | 终态汇总 |

末尾帧示例：

```
data: {"done":true,"tokenCount":187,"elapsed":2.143,"tokPerSec":87.23,"aborted":false,"error":null,"sessionId":"a3b...","promptTokens":512,"kvReusedTokens":420,"kvReusePercent":82.0}
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

每个响应都携带 `ok, path, url, mediaType, fileName`；媒体类型按文件扩展名分类
（image / video / audio / pdf / text）。客户端随后在下一次 `/api/chat` 请求中
引用服务端存储的 `path` —— 图像通过 `imagePaths`，抽取出的视频帧通过
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
      "textFilePaths": ["<上传响应中的 path>"]
    }],
    "maxTokens": 500
  }'
```

- **扫描版 / 纯图像 PDF**：如果加载了具备视觉能力的模型（存在 `--mmproj` 或模
  型内置视觉编码器），页面会被渲染为图像并按视频帧的形式返回
  （`renderedAsImages: true`、`frames[]`、`frameUrls[]`、`framePaths[]`）；在下
  一次 `/api/chat` 请求中把 `framePaths` 作为 `imagePaths` 传入。没有视觉模型
  时，响应会携带 `needsVision: true` 和一条 `warning`，提示需用具备视觉能力的
  模型重启服务。

设置 `TS_PDF_MAX_PAGES` 环境变量可限制读取的 PDF 页数（默认 `0` = 全部页面）。

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

也接受 JSON body `{ "imagePath": "<server path from /api/upload>", "prompt": "...",
"steps": 0, "cfg": 0, "seed": 42 }`（`imagePath` 必须引用上传目录内先前上传过的
文件）。流式变体通过 SSE 发送进度事件与实时去噪预览：

```bash
curl -N -X POST http://localhost:5000/api/image-edit/stream \
  -H "Content-Type: application/json" \
  -d '{"imagePath": "<path from /api/upload>", "prompt": "Replace the background with a sunny beach", "seed": 42}'
```

每步事件形如
`{"imageEdit": true, "step": 2, "total": 4, "image": "data:image/png;base64,...", "width": 1184, "height": 544}`
（`image` 预览快照只在节流后的步骤上出现，每次编辑最多 8 张），最后是一条
`{"done": true, "url": "/uploads/edit-<guid>.png", "width": 1184, "height": 544, "elapsedSeconds": 40.4}`。
对非 Qwen-Image-Edit 模型发起的请求返回 400；并发编辑由进程级锁串行执行。

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
变量修改；请求中显式给出的值始终优先。

### OpenAI 风格选项（位于顶层）

| 参数                | 类型        | 默认值  | 描述                                |
| ------------------- | ----------- | ------- | ----------------------------------- |
| `max_tokens`        | int         | 200     | 生成的最大 token 数                 |
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
    "model": "Qwen3-4B-Q8_0.gguf",
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
    "model": "Qwen3-4B-Q8_0.gguf",
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
    model="Qwen3-4B-Q8_0.gguf",
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
    model="Qwen3-4B-Q8_0.gguf",
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
    model="Qwen3-4B-Q8_0.gguf",
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
