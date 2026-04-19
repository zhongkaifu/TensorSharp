# TensorSharp.Server API Examples

TensorSharp.Server provides three API styles plus a few utility endpoints:
- **Ollama-compatible** (`/api/generate`, `/api/chat/ollama`, `/api/tags`, `/api/show`)
- **OpenAI-compatible** (`/v1/chat/completions`, `/v1/models`)
- **Web UI** (`/api/chat`, `/api/models`, `/api/models/load`, `/api/upload`)
- **Utilities** (`/api/version`, `/api/queue/status`)

Start the server with the exact hosted model via `--model` and, when needed, the exact projector via `--mmproj`. The Web UI and compatibility endpoints expose only that hosted model; `/api/models/load` can reload it, but it does not switch to arbitrary files at runtime.

## Starting the Server

```bash
# Text-only model
./TensorSharp.Server --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend ggml_metal

# Multimodal model (explicit projector)
./TensorSharp.Server --model ~/work/model/gemma-4-E4B-it-Q8_0.gguf \
    --mmproj ~/work/model/gemma-4-mmproj-F16.gguf --backend ggml_metal

# Override default request budget (used when a request omits max_tokens / num_predict)
./TensorSharp.Server --model ~/work/model/Qwen3-4B-Q8_0.gguf --backend ggml_metal --max-tokens 4096
```

The server starts on `http://localhost:5000` (override with the standard ASP.NET Core `PORT` / `ASPNETCORE_URLS` environment variables).

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
  "eval_duration": 1200000000
}
```

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
{"model":"Qwen3-4B-Q8_0.gguf","created_at":"...","response":"","done":true,"done_reason":"stop","total_duration":...,"eval_count":...}
```

### Generate with Image (multimodal)

Images are sent as base64-encoded bytes in the `images` array:

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
  "eval_duration": 1500000000
}
```

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

### Chat with Thinking / Reasoning Mode

Thinking-capable architectures (Qwen 3, Qwen 3.5, Gemma 4, GPT OSS, Nemotron-H) accept `"think": true` and split chain-of-thought from the visible response:

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

Define tools in the same shape as Ollama's tool API. The server detects the architecture's wire format (e.g. `<tool_call>...</tool_call>` for Qwen / Nemotron-H, `<|tool_call>...<tool_call|>` for Gemma 4) and parses them into structured `tool_calls`:

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
    "total_tokens": 28
  }
}
```

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

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","created":...,"model":"...","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{...}}

data: [DONE]
```

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
# Inference queue snapshot (busy flag, pending requests, total processed)
curl http://localhost:5000/api/queue/status

# Server version
curl http://localhost:5000/api/version

# Hosted model + supported backends + default settings
curl http://localhost:5000/api/models
```

`/api/models` returns the single hosted GGUF (and projector if any), the loaded backend name, the list of available backends, the resolved architecture, and the configured default `max_tokens`. The model entry in `/api/tags`, `/v1/models`, and `/api/show` always reports the file actually launched with `--model`.

---

## 3. Sampling Options

### Ollama-style options (inside `options` object)

| Parameter          | Type    | Default | Description                            |
| ------------------ | ------- | ------- | -------------------------------------- |
| `num_predict`      | int     | 200     | Maximum tokens to generate             |
| `temperature`      | float   | 0       | Sampling temperature (0 = greedy)      |
| `top_k`            | int     | 0       | Top-K filtering (0 = disabled)         |
| `top_p`            | float   | 1.0     | Nucleus sampling threshold             |
| `min_p`            | float   | 0       | Minimum probability filtering          |
| `repeat_penalty`   | float   | 1.0     | Repetition penalty                     |
| `presence_penalty` | float   | 0       | Presence penalty                       |
| `frequency_penalty`| float   | 0       | Frequency penalty                      |
| `seed`             | int     | -1      | Random seed (-1 = random)              |
| `stop`             | array   | null    | Stop sequences                         |

### OpenAI-style options (top-level)

| Parameter           | Type        | Default | Description                        |
| ------------------- | ----------- | ------- | ---------------------------------- |
| `max_tokens`        | int         | 200     | Maximum tokens to generate         |
| `temperature`       | float       | 0       | Sampling temperature               |
| `top_p`             | float       | 1.0     | Nucleus sampling threshold         |
| `presence_penalty`  | float       | 0       | Presence penalty                   |
| `frequency_penalty` | float       | 0       | Frequency penalty                  |
| `seed`              | int         | -1      | Random seed                        |
| `stop`              | string/array| null    | Stop sequences                     |
| `response_format`   | object      | null    | `text`, `json_object`, or `json_schema` |

---

## 4. Python Client Examples

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

- `response_format.type = "json_schema"` currently cannot be combined with `tools` or `think`.
- Streaming structured-output requests are buffered and validated before chunks are emitted.
- Invalid schemas return HTTP `400`; model responses that still fail validation return HTTP `422`.

---

## 5. Running Test Requests

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

