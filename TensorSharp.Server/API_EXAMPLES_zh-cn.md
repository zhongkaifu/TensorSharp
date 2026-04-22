# TensorSharp.Server API 示例

[English](API_EXAMPLES.md) | [中文](API_EXAMPLES_zh-cn.md)

TensorSharp.Server 提供三种 API 风格以及若干工具型接口：
- **兼容 Ollama**（`/api/generate`、`/api/chat/ollama`、`/api/tags`、`/api/show`）
- **兼容 OpenAI**（`/v1/chat/completions`、`/v1/models`）
- **Web UI**（`/api/chat`、`/api/sessions`、`/api/models`、`/api/models/load`、`/api/upload`）
- **工具型接口**（`/api/version`、`/api/queue/status`）

启动服务时通过 `--model` 指定承载的模型文件，必要时通过 `--mmproj` 指定多模态投影器。Web UI 与兼容接口仅暴露这一个承载模型；`/api/models/load` 可以重新加载它，但不会在运行时切换到任意其他文件。

## 启动服务

```bash
# 仅文本模型
./TensorSharp.Server --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend ggml_metal

# 多模态模型（显式指定投影器）
./TensorSharp.Server --model ~/work/model/gemma-4-E4B-it-Q8_0.gguf \
    --mmproj ~/work/model/gemma-4-mmproj-F16.gguf --backend ggml_metal

# 覆盖默认请求 token 上限（请求未提供 max_tokens / num_predict 时使用）
./TensorSharp.Server --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend ggml_metal --max-tokens 4096
```

服务默认监听 `http://localhost:5000`（可通过 ASP.NET Core 标准的 `PORT` / `ASPNETCORE_URLS` 环境变量覆盖）。

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
IMG_B64=$(base64 < photo.png)
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
IMG_B64=$(base64 < photo.png)
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

支持思维链的架构（Qwen 3、Qwen 3.5、Gemma 4、GPT OSS、Nemotron-H）可接受 `"think": true`，并将思考过程与可见回答分开返回：

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

工具按 Ollama tool API 的形式定义。服务端会根据当前架构识别工具调用的线协议（如 Qwen / Nemotron-H 使用 `<tool_call>...</tool_call>`，Gemma 4 使用 `<|tool_call>...<tool_call|>`），并解析为结构化的 `tool_calls`：

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
IMG_B64=$(base64 < photo.png)
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
# 推理队列快照（busy 标志、待处理请求数、累计处理数）
curl http://localhost:5000/api/queue/status

# 服务版本
curl http://localhost:5000/api/version

# 承载模型 + 可用后端 + 默认设置
curl http://localhost:5000/api/models
```

`/api/models` 返回唯一承载的 GGUF（如有投影器一并返回），加载后的后端名、可用后端列表、解析出的架构以及配置好的默认 `max_tokens`。`/api/tags`、`/v1/models`、`/api/show` 中的模型条目始终汇报通过 `--model` 实际启动的文件。

---

## 3. Web UI SSE（`/api/chat`）

这是内置聊天界面使用的协议，单独列在这里方便外部 Web UI 接入同一接口。每个事
件都是一个 JSON 对象，通过单条 `data: ...` SSE 帧下发。

### 聊天会话

Web UI 流程是按会话隔离的：每个浏览器 Tab 在加载时会创建自己的会话，并在每次
`/api/chat` 请求中携带该 `sessionId`，因此每个 Tab 都拥有独立的 KV 缓存。
Ollama 与 OpenAI 兼容接口共享一个内置的 `__default__` 会话，该会话在服务器生命
周期内常驻。

```bash
# 创建一个新的会话（返回 id；只有 Web UI 流程需要该步骤）
curl -X POST http://localhost:5000/api/sessions
# {"sessionId":"a3b1c2..."}

# 销毁会话并释放对应的 KV 缓存。默认会话（__default__）不可删除；
# 当 id 不存在时返回 404。
curl -X DELETE http://localhost:5000/api/sessions/a3b1c2...
```

只要在 `/api/chat` 之间复用同一个 `sessionId`，服务器就能在新一轮请求时把上一轮
助手的原始 token 直接拼接到 KV 缓存前缀里（终态 SSE 帧的 `kvReusedTokens` /
`kvReusePercent` 字段会指出复用了多少）。传入 `sessionId: null` 可使用共享的
`__default__` 会话；传入 `newChat: true` 可在不销毁会话的前提下，在下一轮强制
重置该会话的服务端 KV 缓存。

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
| `queue_position`、`queue_pending` | 排队期间每秒一次 | 当前请求排在队列中的位置 |
| `token` | 每个生成的 token（启用 `think` / `tools` 时为解析后的内容片段） | 流式正文 |
| `thinking` | 解析到的思维链片段（仅当模型输出含思维链时） | 流式思维链 |
| `tool_calls` | 模型输出工具调用 | `{name, arguments}` 数组 |
| `done`、`tokenCount`、`elapsed`、`tokPerSec`、`aborted`、`error`、`sessionId`、`promptTokens`、`kvReusedTokens`、`kvReusePercent` | 末尾帧 | 终态汇总 |

末尾帧示例：

```
data: {"done":true,"tokenCount":187,"elapsed":2.143,"tokPerSec":87.23,"aborted":false,"error":null,"sessionId":"a3b...","promptTokens":512,"kvReusedTokens":420,"kvReusePercent":82.0}
```

`kvReusedTokens` / `kvReusePercent` 与 Ollama 的 `prompt_cache_hit_*` 以及
OpenAI 的 `usage.prompt_tokens_details.cached_tokens` 含义一致 —— 都表示有多
少 prompt token 直接复用了对应会话上一轮的 KV 缓存。

---

## 4. 采样选项

### Ollama 风格选项（位于 `options` 对象中）

| 参数               | 类型    | 默认值  | 描述                                   |
| ------------------ | ------- | ------- | -------------------------------------- |
| `num_predict`      | int     | 200     | 生成的最大 token 数                    |
| `temperature`      | float   | 0       | 采样温度（0 = 贪心）                   |
| `top_k`            | int     | 0       | Top-K 过滤（0 = 关闭）                 |
| `top_p`            | float   | 1.0     | 核采样阈值                             |
| `min_p`            | float   | 0       | 最小概率过滤                           |
| `repeat_penalty`   | float   | 1.0     | 重复惩罚                               |
| `presence_penalty` | float   | 0       | 出现惩罚                               |
| `frequency_penalty`| float   | 0       | 频率惩罚                               |
| `seed`             | int     | -1      | 随机种子（-1 = 不指定）                |
| `stop`             | array   | null    | 停止序列                               |

### OpenAI 风格选项（位于顶层）

| 参数                | 类型        | 默认值  | 描述                                |
| ------------------- | ----------- | ------- | ----------------------------------- |
| `max_tokens`        | int         | 200     | 生成的最大 token 数                 |
| `temperature`       | float       | 0       | 采样温度                            |
| `top_p`             | float       | 1.0     | 核采样阈值                          |
| `presence_penalty`  | float       | 0       | 出现惩罚                            |
| `frequency_penalty` | float       | 0       | 频率惩罚                            |
| `seed`              | int         | -1      | 随机种子                            |
| `stop`              | string/array| null    | 停止序列                            |
| `response_format`   | object      | null    | `text`、`json_object` 或 `json_schema` |

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

- `response_format.type = "json_schema"` 当前不能与 `tools` 或 `think` 同时使用。
- 流式结构化输出请求会先在服务端缓存并校验，再以 chunk 形式发出。
- 非法 schema 返回 HTTP `400`；模型输出未能通过校验则返回 HTTP `422`。

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
