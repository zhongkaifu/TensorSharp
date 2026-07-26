# TensorSharp.Server API Examples

[English](API_EXAMPLES.md) | [中文](API_EXAMPLES_zh-cn.md)

TensorSharp.Server provides three API styles plus a few utility endpoints:

- **Ollama-compatible** (`/api/generate`, `/api/chat/ollama`, `/api/tags`, `/api/show`)
- **OpenAI-compatible** (`/v1/chat/completions`, `/v1/models`)
- **Web UI** (`/api/chat`, `/api/sessions`, `/api/models`, `/api/models/load`, `/api/upload`, `/api/image-edit`, `/api/image-edit/stream`)
- **Utilities** (`/api/version`, `/api/queue/status`)

Start the server with the exact hosted model via `--model` and, when needed, the exact projector via `--mmproj`. The projector is **not auto-detected** by `TensorSharp.Server`. The Web UI and compatibility endpoints expose only that startup model/projector pair; `/api/models/load` can reload the same pair on a supported backend, but it cannot choose a model on a model-less server or switch to another file at runtime.

## Current Contract

| Area | Contract |
|---|---|
| Hosted models | One GGUF file, selected with `--model`; requests must name that hosted file or its basename |
| Projectors | Optional single projector, selected explicitly with `--mmproj`; used for multimodal-capable models |
| Backends | `mlx`, `cuda`, `ggml_metal`, `ggml_cuda`, `ggml_vulkan`, `ggml_cpu`, `cpu`; `/api/models` reports which are available on the host |
| Concurrency | Autoregressive chat uses the continuous-batching engine. The legacy queue API remains for status/compatibility fields; DiffusionGemma Web UI requests use a separate block-boundary diffusion scheduler. |
| Generation modes | Autoregressive models stream appended token chunks. DiffusionGemma returns final text on append-only compatibility endpoints and exposes live whole-message denoising previews on Web UI `/api/chat`. |
| Sessions | Web UI uses per-tab sessions; Ollama/OpenAI compatibility endpoints share the default session |
| Uploads | `/api/upload` accepts image / video / audio / text / **PDF** files; born-digital PDFs return extracted text, scanned PDFs return page images for vision-capable models (`TS_PDF_MAX_PAGES` caps pages read) |
| Image editing | Qwen-Image-Edit (`qwen_image`) models are served through `/api/image-edit` and `/api/image-edit/stream`, not the chat endpoints |
| Structured outputs | OpenAI `response_format` supports `text`, `json_object`, and `json_schema`; `response_format` (`json_object` / `json_schema`) cannot be combined with `think` or `tools` |

> **Network safety:** the server listens on `0.0.0.0:5000` and has no API-key
> authentication or built-in TLS. Keep it on a trusted network or place an
> authenticating TLS reverse proxy in front of it.

## Starting the Server

### Quick start in ~30 seconds

The verified fast path hosts Gemma 4 E4B Q8_0 on a native GGML backend. The commands below take about 30 seconds to copy and run; the 7.48 GiB model download and the first restore/build take longer and depend on the network connection and machine. Besides the [.NET 10 SDK](../DEVELOPMENT.md#install-the-net-10-sdk), Git, and `curl`, this path needs the normal native GGML build prerequisites for the chosen backend. On a new machine, use the linked Windows, macOS, or Linux SDK instructions first; the runtime alone cannot build TensorSharp. The model is the recommended public artifact from [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF); a lower-memory `gemma-4-E4B-it-Q4_K_M.gguf` is in the same repository. The copy/paste block below is for Linux + NVIDIA; platform-specific backend choices follow it:

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

Use `ggml_cuda` on Windows/Linux with NVIDIA, `ggml_metal` on Apple Silicon,
`ggml_vulkan` (set `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON` instead) on
Windows/Linux with a Vulkan-capable AMD, Intel, or NVIDIA GPU, or `ggml_cpu`
when no GPU is available. The verification claim covers the E4B Q8_0
family/path; it does not claim that a specific public-file checksum was the
benchmark input.

For text-only API calls, no projector is needed. For image, video, or audio,
also download `mmproj-gemma-4-E4B-it-Q8_0.gguf` from the same repository and
restart with `--mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf`.

In a second terminal:

```bash
curl -s http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma-4-E4B-it-Q8_0.gguf","messages":[{"role":"user","content":"Reply with one short hello."}],"max_tokens":32}'
```

Open the bundled UI at **<http://localhost:5000/index.html>**. `GET /` is the liveness endpoint and returns `"TensorSharp.Server is running"`.

### Already-built or extracted application folder

Run the commands below from the repository root after building; they invoke `TensorSharp.Server/bin/TensorSharp.Server.dll`, whose folder also contains the copied native libraries and `wwwroot/`. There are currently no binary assets attached to the `v3.0.5.0` GitHub release, so do not present a release-archive command as available until an archive is actually published.

```bash
# Text-only model
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend ggml_metal

# Windows/Linux + NVIDIA, direct CUDA/cuBLAS backend
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend cuda

# Windows/Linux + NVIDIA, GGML CUDA backend
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend ggml_cuda

# Windows/Linux + AMD/Intel/NVIDIA GPU, GGML Vulkan backend (pick the device on multi-GPU hosts with --gpu-device; see --list-gpus)
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend ggml_vulkan --gpu-device 0

# Apple Silicon, MLX backend
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend mlx

# Multimodal model (explicit projector)
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/gemma-4-E4B-it-Q8_0.gguf \
    --mmproj ~/work/model/mmproj-gemma-4-E4B-it-Q8_0.gguf --backend ggml_metal

# DiffusionGemma text-diffusion model
DIFFUSION_STEPS=48 DIFFUSION_MAX_BATCH=2 \
  dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggml_metal

# Override the Web UI default token budget (default 20000). The Ollama/OpenAI
# compatibility endpoints instead default to 200 when a request omits
# max_tokens / num_predict — set the value per request there.
dotnet TensorSharp.Server/bin/TensorSharp.Server.dll --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend ggml_metal --max-tokens 4096
```

The API starts on `http://localhost:5000`; the Web UI is
`http://localhost:5000/index.html`. The current binary passes a fixed
`http://0.0.0.0:5000` listen address to ASP.NET Core; the Docker Space files
patch that constant to `7860` during image build.

`--model` is required for inference. Starting with only `--backend` produces a
model-less status server, but `/api/models/load` cannot select a file that was
not supplied at startup. For multimodal inference, always pass the projector
explicitly with `--mmproj`; a bare projector filename is resolved next to the
model.

Backend quick reference:

| Value | Meaning |
|---|---|
| `cpu` | Pure C# CPU backend |
| `cuda` | Direct CUDA backend using CUDA Driver API, cuBLAS, PTX kernels, and CPU fallbacks |
| `mlx` | MLX Metal backend for Apple Silicon |
| `ggml_cpu` | Native GGML CPU backend |
| `ggml_metal` | GGML Metal backend for macOS |
| `ggml_cuda` | GGML CUDA backend for NVIDIA GPUs |
| `ggml_vulkan` | GGML Vulkan backend for AMD / Intel / NVIDIA GPUs (vendor-neutral; requires a native build with Vulkan enabled) |

---

## 1. Ollama-compatible API

### List Models

```bash
curl http://localhost:5000/api/tags
```

Response:
```json
{
  "models": [
    {"name": "Qwen3-4B-Q8_0", "model": "Qwen3-4B-Q8_0.gguf", "size": 4530000000, "modified_at": "2025-03-15T10:00:00Z"}
  ]
}
```

### Show Model Info

```bash
curl -X POST http://localhost:5000/api/show \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B-Q8_0.gguf"}'
```

### Generate (non-streaming)

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

Response:
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

`prompt_cache_hit_tokens` reports how many of the `prompt_eval_count` tokens
were served straight from the prior turn's KV cache. `/api/generate` always
resets the session before prefilling, so this value is always `0`; it is
non-zero on `/api/chat/ollama` when the prompt prefix matches a previous turn.

### Generate (streaming)

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

Each line is a JSON object (newline-delimited JSON):
```
{"model":"Qwen3-4B-Q8_0.gguf","created_at":"...","response":"Why","done":false}
{"model":"Qwen3-4B-Q8_0.gguf","created_at":"...","response":" did","done":false}
...
{"model":"Qwen3-4B-Q8_0.gguf","created_at":"...","response":"","done":true,"done_reason":"stop","total_duration":...,"eval_count":...,"prompt_cache_hit_tokens":0,"prompt_cache_hit_ratio":0.0}
```

The final `done` chunk also carries the same `prompt_cache_hit_tokens` /
`prompt_cache_hit_ratio` fields as the non-streaming response.

### Generate with Image (multimodal)

Images are sent as base64-encoded bytes in the `images` array:

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

### Chat (non-streaming)

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

Response:
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

`prompt_cache_hit_tokens` and `prompt_cache_hit_ratio` describe how much of the
prompt was served from the previous turn's KV cache. On the first turn of a
fresh conversation both values are zero; on a follow-up turn that reuses the
prior conversation prefix they grow to (often) close to `prompt_eval_count` /
`1.0`. The same fields appear on the final NDJSON chunk in streaming mode.

### Chat (streaming)

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

### Chat with Multi-turn History

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

### Chat with Image (multimodal)

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

### Chat with Thinking / Reasoning Mode

Thinking-capable architectures (Qwen 3, Qwen 3.5/3.6-family, Gemma 4, GPT OSS, Nemotron-H) accept `"think": true` and split chain-of-thought from the visible response:

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

The response carries the chain-of-thought separately in `message.thinking`:

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

### Chat with Tool Calling

Define tools in the same shape as Ollama's tool API. The server detects the architecture's wire format (e.g. `<tool_call>...</tool_call>` for Qwen / Nemotron-H, `<|tool_call>...<tool_call|>` for Gemma 4, and the Harmony `commentary` channel `<|channel|>commentary to=functions.NAME ...<|call|>` for GPT OSS) and parses them into structured `tool_calls`:

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
        "description": "Get current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city":  {"type": "string", "description": "Target city"},
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

The response shape (when the model decides to call the tool):

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

Continue the conversation by appending the assistant tool call and a `role: "tool"` message containing the function result, then call `/api/chat/ollama` again.

---

## 2. OpenAI-compatible API

### List Models

```bash
curl http://localhost:5000/v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {"id": "Qwen3-4B-Q8_0", "object": "model", "owned_by": "local"}
  ]
}
```

### Chat Completions (non-streaming)

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

Response:
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

`usage.prompt_tokens_details.cached_tokens` follows OpenAI's standard
KV-cache-hit extension. On a follow-up turn that shares the prefix of an
earlier turn this value approaches `prompt_tokens`, which lets clients reason
about TTFT savings without enabling Debug logging on the server.

### Chat Completions (streaming)

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

Each chunk is sent as SSE:
```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9,"prompt_tokens_details":{"cached_tokens":0}}}

data: [DONE]
```

The final chunk's `usage` block carries `prompt_tokens_details.cached_tokens`
just like the non-streaming response.

### Chat Completions with JSON mode

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

Response:
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

### Chat Completions with Structured Outputs (`json_schema`)

TensorSharp.Server accepts the OpenAI Chat Completions `response_format` shape, injects strict JSON instructions into the prompt, and validates the final output before returning it.

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

Response:
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

### Chat Completions with Image (multimodal, OpenAI format)

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

### Chat Completions with Tool Calling

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
        "description": "Get current weather for a city.",
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

When the model emits a tool call the response uses OpenAI-style fields:

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

Append the assistant `tool_calls` plus a follow-up `{"role": "tool", "tool_call_id": "...", "content": "..."}` message to continue the loop.

### Utilities

```bash
# Legacy-compatible inference load snapshot: pending_requests is normally 0
# because the continuous-batching engine, not InferenceQueue, owns concurrency
curl http://localhost:5000/api/queue/status

# Legacy Ollama protocol version (hard-coded to 0.1.0; not the TensorSharp release version)
curl http://localhost:5000/api/version

# Hosted model + supported backends + default settings
curl http://localhost:5000/api/models
```

`/api/models` returns the single hosted GGUF (and projector if any), the loaded backend name, the list of available backends, the resolved architecture, and the configured default `max_tokens`. The model entry in `/api/tags`, `/v1/models`, and `/api/show` always reports the file actually launched with `--model`. If a CUDA backend is missing from `supportedBackends`, the host did not detect a usable NVIDIA driver/device or GGML CUDA initialization path at startup; the direct `cuda` backend still needs cuBLAS discoverable when inference runs. If `ggml_vulkan` is missing, the native GGML bridge was not built with Vulkan enabled or no Vulkan 1.3 device/driver was found. If `mlx` is missing, the host did not detect a usable Apple Silicon MLX runtime.

---

## 3. Web UI SSE (`/api/chat`)

This is the protocol the bundled chat UI uses; documented here so external Web
UIs can plug into the same endpoint. Every event is a JSON object delivered as a
single `data: ...` SSE frame.

When the hosted model is DiffusionGemma, this endpoint uses whole-message
replacement frames for live denoising previews. Ollama/OpenAI compatibility
endpoints keep their append-oriented response shapes and receive only final text.

### Chat Sessions

The Web UI flow is session-scoped: every browser tab creates its own session at
load time and attaches the `sessionId` to every `/api/chat` request, so each
tab gets isolated tracked conversation history. Request KV blocks and prefix
reuse are owned by the inference engine. The Ollama and OpenAI-compatible
endpoints share the service's intrinsic compatibility history.

```bash
# Create a fresh session (returns its id; only the Web UI flow needs this)
curl -X POST http://localhost:5000/api/sessions
# {"sessionId":"a3b1c2..."}

# Dispose a session and clear its tracked history. Engine request KV blocks are
# released independently. The default session (__default__) cannot be removed;
# the call returns 404 if the id is unknown.
curl -X DELETE http://localhost:5000/api/sessions/a3b1c2...
```

Reusing the same `sessionId` across `/api/chat` requests preserves tracked
history and lets the engine reuse matching prompt-prefix blocks on the next
turn (the `kvReusedTokens` / `kvReusePercent` fields on the terminal SSE frame
report how much was reused). Omit `sessionId` or pass `null` to use the shared
`__default__` Web UI session. Pass `newChat: true` to clear tracked history
before the next request without disposing the session.

### Streaming Chat

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

Event shapes:

| Event field(s) | When | Meaning |
|---|---|---|
| `queue_position`, `queue_pending` | compatibility event if a request waits on the legacy queue shim | queue-position fields retained for older clients |
| `token` | each generated token (or parsed content chunk when `think`/`tools` are active) | streaming content |
| `replace`, `diffusionStep`, `diffusionTotal`, `preview` | each DiffusionGemma denoising preview and final replacement | replace the whole assistant message body instead of appending a token |
| `thinking` | each parsed reasoning chunk (only when the model emits one) | streaming chain-of-thought |
| `tool_calls` | when the model emits a tool call | array of `{name, arguments}` |
| `done`, `tokenCount`, `elapsed`, `tokPerSec`, `aborted`, `error`, `sessionId`, `promptTokens`, `kvReusedTokens`, `kvReusePercent` | last frame | terminal summary |

Sample terminal frame:

```
data: {"done":true,"tokenCount":187,"elapsed":2.143,"tokPerSec":87.23,"aborted":false,"error":null,"sessionId":"a3b...","promptTokens":512,"kvReusedTokens":420,"kvReusePercent":82.0}
```

Sample DiffusionGemma preview frame:

```
data: {"replace":"A refined draft of the whole answer","diffusionStep":12,"diffusionTotal":48,"preview":true}
```

Use `kvReusedTokens` / `kvReusePercent` in the same way as the Ollama
`prompt_cache_hit_*` and OpenAI `usage.prompt_tokens_details.cached_tokens`
fields - they all measure the same thing (prompt tokens served straight from
the prior turn's KV cache) for the corresponding session.

### File Uploads (`/api/upload`) — images, video, audio, text, PDF

```bash
# Upload a file (multipart form; the first file in the form is used)
curl -X POST http://localhost:5000/api/upload -F "file=@report.pdf"
```

Every response carries `ok, path, url, mediaType, fileName`; the media type is
classified by file extension (image / video / audio / pdf / text). The client
then references the stored server `path` in the next `/api/chat` request —
images via `imagePaths`, extracted video frames via `isVideo: true` +
`imagePaths`, audio via `audioPaths`, and text content by inlining the returned
`textContent` into the message.

**PDF documents** get a two-stage treatment:

- **Born-digital PDF** (has a selectable text layer): the text is extracted and
  returned in `textContent`, with `renderedAsImages: false`, `pageCount`,
  and `extractedPageCount`. The extracted text is returned in full; the final
  rendered prompt is checked against the loaded model's actual context window.
  `truncated` is always `false` for extracted text. Legacy truncation/count
  fields remain present as nullable compatibility fields; uploads are no longer
  tokenized just to populate them.
  Inline it into the chat message the way the bundled UI does:

```bash
curl -N -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "role": "user",
      "content": "[File: report.pdf]\n<textContent from the upload response>\n[End of file]\nPlease analyze the attached PDF document and summarize its content.",
      "textFilePaths": ["<path from the upload response>"]
    }],
    "maxTokens": 500
  }'
```

- **Scanned / image-only PDF**: if a vision-capable model is loaded (`--mmproj`
  present or the model has a built-in vision encoder), pages are rendered to
  images and returned like video frames (`renderedAsImages: true`, `frames[]`,
  `frameUrls[]`, `framePaths[]`); pass the `framePaths` as `imagePaths` on the
  next `/api/chat` request. Without a vision model the response instead carries
  `needsVision: true` plus a `warning` asking for a restart with a
  vision-capable model.

Set the `TS_PDF_MAX_PAGES` environment variable to cap the number of PDF pages
read (default `0` = all pages).

### Image Editing (`/api/image-edit`, Qwen-Image-Edit)

When the hosted `--model` is a Qwen-Image-Edit DiT GGUF (architecture
`qwen_image`), image+prompt turns go to the image-edit endpoints instead of
`/api/chat`:

```bash
# One-shot edit (multipart). steps=0 / cfg=0 mean auto
# (30 steps / cfg 2.5, or the Lightning LoRA's step count / cfg 1.0).
curl -X POST http://localhost:5000/api/image-edit \
  -F "image=@photo.png" \
  -F "prompt=Replace the background with a sunny beach" \
  -F "steps=0" -F "cfg=0" -F "seed=42"
```

Response:

```json
{"ok": true, "url": "/uploads/edit-<guid>.png", "width": 1184, "height": 544, "elapsedSeconds": 40.4}
```

A JSON body `{ "imagePath": "<server path from /api/upload>", "prompt": "...",
"steps": 0, "cfg": 0, "seed": 42 }` is also accepted (`imagePath` must
reference a previously uploaded file inside the upload directory). The
streaming variant emits SSE progress with live denoising previews:

```bash
curl -N -X POST http://localhost:5000/api/image-edit/stream \
  -H "Content-Type: application/json" \
  -d '{"imagePath": "<path from /api/upload>", "prompt": "Replace the background with a sunny beach", "seed": 42}'
```

Per-step events look like
`{"imageEdit": true, "step": 2, "total": 4, "image": "data:image/png;base64,...", "width": 1184, "height": 544}`
(the `image` preview snapshot appears on throttled steps, up to 8 per edit),
followed by a final
`{"done": true, "url": "/uploads/edit-<guid>.png", "width": 1184, "height": 544, "elapsedSeconds": 40.4}`.
Requests against a model that is not Qwen-Image-Edit return 400; concurrent
edits are serialized by a process-wide lock.

---

## 4. Sampling Options

### Ollama-style options (inside `options` object)

| Parameter          | Type    | Default | Description                            |
| ------------------ | ------- | ------- | -------------------------------------- |
| `num_predict`      | int     | 200     | Maximum tokens to generate             |
| `temperature`      | float   | 0.8     | Sampling temperature (0 = greedy)      |
| `top_k`            | int     | 40      | Top-K filtering (0 = disabled)         |
| `top_p`            | float   | 0.9     | Nucleus sampling threshold             |
| `min_p`            | float   | 0       | Minimum probability filtering          |
| `repeat_penalty`   | float   | 1.1     | Repetition penalty (1.0 = none)        |
| `presence_penalty` | float   | 0       | Presence penalty                       |
| `frequency_penalty`| float   | 0       | Frequency penalty                      |
| `seed`             | int     | -1      | Random seed (-1 = random)              |
| `stop`             | array   | null    | Stop sequences                         |

The defaults are the server's configured sampling defaults (Ollama-compatible).
They can be changed at startup with the matching server flags (`--temperature`,
`--top-k`, `--top-p`, `--min-p`, `--repeat-penalty`, `--presence-penalty`,
`--frequency-penalty`, `--seed`) or `TENSORSHARP_*` environment variables;
per-request values always win.

### OpenAI-style options (top-level)

| Parameter           | Type        | Default | Description                        |
| ------------------- | ----------- | ------- | ---------------------------------- |
| `max_tokens`        | int         | 200     | Maximum tokens to generate         |
| `temperature`       | float       | 0.8     | Sampling temperature               |
| `top_p`             | float       | 0.9     | Nucleus sampling threshold         |
| `presence_penalty`  | float       | 0       | Presence penalty                   |
| `frequency_penalty` | float       | 0       | Frequency penalty                  |
| `seed`              | int         | -1      | Random seed                        |
| `stop`              | string/array| null    | Stop sequences                     |
| `response_format`   | object      | null    | `text`, `json_object`, or `json_schema` |
| `think`             | bool        | false   | Non-standard extension: enables thinking/reasoning parsing (returned/streamed as `reasoning_content`) |

`top_k`, `min_p`, and `repetition_penalty` are **not parsed** on the OpenAI
surface — the server's configured defaults apply for those. Use the Ollama or
Web UI endpoints if a request needs to set them per call.

---

## 5. Python Client Examples

### Using `requests` (Ollama-style)

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

### Streaming with `requests` (Ollama-style)

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

### Using `openai` Python SDK

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

### Using `openai` Python SDK with structured outputs

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

### Streaming with `openai` Python SDK

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

Notes:

- `response_format` (`json_object` or `json_schema`) currently cannot be combined with `tools` or `think` (HTTP `400`).
- `json_object` / `json_schema` requests constrain the **first sampled token** to a `{`-opening candidate (the same effect llama.cpp gets from its JSON grammar), so chatty models cannot emit prose before the object and streamed time-to-first-token reflects prefill latency instead of suppressed preamble. Subsequent tokens sample normally. Set `TS_JSON_FORCE_OPEN=0` to disable.
- Streaming `json_object` requests stream the JSON object token-by-token (code fences and stray tags are stripped on the fly), so time-to-first-token reflects prefill latency. Streaming `json_schema` (strict) requests are still buffered and schema-normalized before the single chunk is emitted. Set `TS_STRUCTURED_STREAM_BUFFER=1` to force the legacy buffer-everything behavior for both. Non-streaming requests are always normalized.
- Invalid schemas return HTTP `400`; non-streaming / `json_schema` responses that still fail validation return HTTP `422` (a `json_object` stream that has already started cannot change its status code).

---

## 6. Running Test Requests

The `test_requests.jsonl` file contains sample requests for all endpoints. Run them with:

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
