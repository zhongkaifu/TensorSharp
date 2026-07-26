# Thinking Mode and Tool Call Examples

[English](example_api_thinking_tools.md) | [中文](example_api_thinking_tools_zh-cn.md)

These examples cover the current thinking/tool-call surface for the CLI and the two server compatibility APIs. Run source commands from the repository root with the full [.NET 10 SDK](../../DEVELOPMENT.md#install-the-net-10-sdk), not only the runtime. A normal native build also needs Git, network access, CMake, and a C++ toolchain; replace `ggml_metal` with `mlx`, `cuda`, `ggml_cuda`, `ggml_vulkan`, `ggml_cpu`, or `cpu` to match your machine.

| Architecture | Thinking | Tool calls | Notes |
|---|---|---|---|
| Gemma 4 | Yes | Yes | Uses Gemma channel/tool-call tags |
| Qwen 3 | Yes | Yes | Uses `<think>` and JSON-style `<tool_call>` tags |
| Qwen 3.5 / 3.6 family | Yes | Yes | Covers `qwen35`, `qwen35moe`, and `qwen3next` GGUFs |
| GPT OSS | Yes | Yes | Harmony parser separates analysis/final channels; tool calls go on the commentary channel |
| Nemotron-H | Yes | Yes | Uses Qwen-style thinking/tool-call tags |
| DiffusionGemma | No | No | Uses separate text-diffusion generation flags such as `--diffusion-steps`; not a thinking/tool-call template |
| Gemma 3 / Mistral 3 | No | No | Multimodal-capable, but not thinking/tool-call capable in TensorSharp |

## Gemma 4 E4B Setup

The verified model is [gemma-4-E4B-it-Q8_0.gguf](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true) (7.48 GiB) from [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF); a lower-memory `gemma-4-E4B-it-Q4_K_M.gguf` is in the same repository. Text-only use does not need the mmproj file. The commands take seconds to copy and run, but the model download and the first restore/native build take longer and depend on your connection and machine. For the broader setup, see the root [fast-start section](../../README.md#quick-start) and the detailed [Gemma 4 card](../../docs/models/gemma4.md#verified-gemma-4-e4b-native-ggml-fast-path).

macOS / Linux:

```bash
mkdir -p models
curl --fail --location \
  --output models/gemma-4-E4B-it-Q8_0.gguf \
  "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true"

export TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON  # Linux NVIDIA; on macOS omit this and use --backend ggml_metal
dotnet run --project TensorSharp.Cli/TensorSharp.Cli.csproj \
  -p:TensorSharpSkipMlxNative=true -- \
  --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda \
  --max-tokens 32
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force models | Out-Null
curl.exe --fail --location `
  --output models\gemma-4-E4B-it-Q8_0.gguf `
  "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true"

$env:TENSORSHARP_GGML_NATIVE_ENABLE_CUDA = 'ON'
dotnet run --project TensorSharp.Cli\TensorSharp.Cli.csproj `
  -p:TensorSharpSkipMlxNative=true -- `
  --model models\gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda `
  --max-tokens 32
```

On AMD/Intel GPUs, set `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON` and use `--backend ggml_vulkan`; without a GPU, use `--backend ggml_cpu` with no environment variable.

With no `--input`, the CLI uses `What is 1+1?`. A custom one-shot text prompt must be saved to a file and passed with `--input <file>`; `--prompt` is only for Qwen-Image-Edit. The defaults are `--max-tokens 100` and `--backend ggml_cpu`. There is no complete `--help` screen, and unknown arguments are currently ignored, so copy flag names exactly.

## Console Application

The examples below use the same downloaded model and native GGML path, so each command can be copied from the repository root after the setup above. They show the Linux/Windows NVIDIA form; swap `--backend ggml_cuda` (and its environment variable) for `ggml_metal`, `ggml_vulkan`, or `ggml_cpu` to match your machine.

### Thinking Mode

Enable thinking mode with `--think`. The model will show its reasoning process before giving the final answer.

```bash
export TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON

# Basic thinking mode
dotnet run --project TensorSharp.Cli/TensorSharp.Cli.csproj \
  -p:TensorSharpSkipMlxNative=true -- \
  --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda \
  --input TensorSharp.Cli/testdata/input_thinking.txt \
  --think --max-tokens 500

# Thinking mode with sampling
dotnet run --project TensorSharp.Cli/TensorSharp.Cli.csproj \
  -p:TensorSharpSkipMlxNative=true -- \
  --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda \
  --input TensorSharp.Cli/testdata/input_thinking.txt \
  --think --max-tokens 500 \
  --temperature 0.6 --top-p 0.95
```

### Tool Call Mode

Provide tool definitions via `--tools <file.json>`. The model will output structured tool calls.

```bash
export TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON

# Weather tool call
dotnet run --project TensorSharp.Cli/TensorSharp.Cli.csproj \
  -p:TensorSharpSkipMlxNative=true -- \
  --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda \
  --input TensorSharp.Cli/testdata/input_tool_call.txt \
  --tools TensorSharp.Cli/testdata/tools_weather.json --max-tokens 300

# Calculator tool call
dotnet run --project TensorSharp.Cli/TensorSharp.Cli.csproj \
  -p:TensorSharpSkipMlxNative=true -- \
  --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda \
  --input TensorSharp.Cli/testdata/input_tool_calc.txt \
  --tools TensorSharp.Cli/testdata/tools_calculator.json --max-tokens 300

# Combined: thinking + tools
dotnet run --project TensorSharp.Cli/TensorSharp.Cli.csproj \
  -p:TensorSharpSkipMlxNative=true -- \
  --model models/gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda \
  --input TensorSharp.Cli/testdata/input_tool_call.txt \
  --tools TensorSharp.Cli/testdata/tools_weather.json --think --max-tokens 500
```

### Tool Definition Format

Tools are defined in a JSON file as an array of `ToolFunction` objects:

```json
[
  {
    "Name": "function_name",
    "Description": "What this function does",
    "Parameters": {
      "param1": {
        "Type": "string",
        "Description": "Parameter description"
      },
      "param2": {
        "Type": "string",
        "Description": "Another parameter",
        "Enum": ["option1", "option2"]
      }
    },
    "Required": ["param1"]
  }
]
```

## Web API (Ollama-compatible)

### Thinking Mode via Ollama API

```bash
curl -s http://localhost:5000/api/chat/ollama -d '{
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "messages": [{"role": "user", "content": "How many r'\''s in strawberry?"}],
  "think": true,
  "stream": false,
  "options": {"num_predict": 500}
}'
```

Response includes `thinking` field in the message:

```json
{
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "message": {
    "role": "assistant",
    "content": "There are 3 r's in strawberry.",
    "thinking": "Let me spell it out: s-t-r-a-w-b-e-r-r-y..."
  },
  "done": true,
  "done_reason": "stop"
}
```

### Tool Call via Ollama API

```bash
curl -s http://localhost:5000/api/chat/ollama -d '{
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [
    {
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "stream": false,
  "options": {"num_predict": 300}
}'
```

Response includes `tool_calls` when the model decides to use a tool:

```json
{
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "get_current_weather",
          "arguments": {"location": "Paris", "unit": "celsius"}
        }
      }
    ]
  },
  "done": true,
  "done_reason": "tool_calls"
}
```

### Thinking + Tools via Ollama API

```bash
curl -s http://localhost:5000/api/chat/ollama -d '{
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "think": true,
  "tools": [
    {
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "stream": false,
  "options": {"num_predict": 500}
}'
```

Response includes both `thinking` and `tool_calls`:

```json
{
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "message": {
    "role": "assistant",
    "content": "",
    "thinking": "The user wants to know the weather in Paris. I should use the get_current_weather function.",
    "tool_calls": [
      {
        "function": {
          "name": "get_current_weather",
          "arguments": {"location": "Paris"}
        }
      }
    ]
  },
  "done": true,
  "done_reason": "tool_calls"
}
```

## Web API (OpenAI-compatible)

### Thinking Mode via OpenAI API

```bash
curl -s http://localhost:5000/v1/chat/completions -d '{
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "messages": [{"role": "user", "content": "How many r'\''s in strawberry?"}],
  "think": true,
  "max_tokens": 500
}'
```

### Tool Call via OpenAI API

```bash
curl -s http://localhost:5000/v1/chat/completions -d '{
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["location"]
        }
      }
    }
  ],
  "max_tokens": 300
}'
```

Response (OpenAI format):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          {
            "id": "call_...",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\":\"Tokyo\",\"unit\":\"celsius\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ]
}
```

## Python Examples

### Thinking Mode with Python

```python
import requests

response = requests.post("http://localhost:5000/api/chat/ollama", json={
    "model": "gemma-4-E4B-it-Q8_0.gguf",
    "messages": [{"role": "user", "content": "How many r's in strawberry?"}],
    "think": True,
    "stream": False,
    "options": {"num_predict": 500}
})

data = response.json()
if data["message"].get("thinking"):
    print("=== Thinking ===")
    print(data["message"]["thinking"])
    print("=== Answer ===")
print(data["message"]["content"])
```

### Tool Call with Python (OpenAI SDK compatible)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="unused")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
}]

response = client.chat.completions.create(
    model="gemma-4-E4B-it-Q8_0.gguf",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=tools,
    max_tokens=300
)

choice = response.choices[0]
if choice.finish_reason == "tool_calls":
    for tc in choice.message.tool_calls:
        print(f"Tool: {tc.function.name}")
        print(f"Args: {tc.function.arguments}")
else:
    print(choice.message.content)
```

## Supported Model Architectures

| Architecture | Thinking Tags | Tool Call Tags |
|---|---|---|
| Gemma 4 | `<\|channel>thought\n...<channel\|>` | `<\|tool_call>call:NAME{args}<tool_call\|>` |
| Qwen 3 | `<think>...</think>` | `<tool_call>{"name":"...","arguments":{...}}</tool_call>` |
| Qwen 3.5 / 3.6 family | `<think>...</think>` | `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>` |
| GPT OSS | `<\|channel\|>analysis ... <\|channel\|>final` (Harmony) | `<\|channel\|>commentary to=functions.NAME <\|constrain\|>json<\|message\|>{args}<\|call\|>` |
| Nemotron-H | `<think>...</think>` | `<tool_call>{"name":"...","arguments":{...}}</tool_call>` |
| DiffusionGemma | n/a | n/a |

## How It Works

### Thinking Mode

When `think: true` is passed:

1. **Gemma4**: The template injects `<|think|>` into the system turn. The model then outputs thinking inside `<|channel>thought\n...<channel|>` tags before the actual response.
2. **Qwen3**: The template appends `<think>\n` to the generation prompt. The model outputs thinking directly, terminated by `</think>`, followed by the answer.
3. **Qwen3.5 / 3.6-family GGUFs**: Same as Qwen3. When thinking is disabled, an empty `<think>\n\n</think>\n\n` block is prepended.
4. **GPT OSS**: The Harmony-format template always emits structured channel framing: `<|channel|>analysis ... <|channel|>final`. The output parser is always on for this architecture, so thinking content is split out whether or not `think: true` is passed.
5. **Nemotron-H**: Uses the Qwen3-style `<think>...</think>` framing.
6. **DiffusionGemma**: Uses `DiffusionGemmaSampler` rather than the autoregressive chat-template path; use the diffusion CLI options instead of `--think`.

### Tool Calls

When `tools` are provided:

1. **Gemma4**: Tool declarations use `<|tool>declaration:NAME{...}<tool|>` format in the system turn. The model outputs calls as `<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>`.
2. **Qwen3**: Tool definitions are injected as JSON in the system message. The model outputs calls as `<tool_call>{"name":"...","arguments":{...}}</tool_call>`.
3. **Qwen3.5 / 3.6-family GGUFs**: Tool definitions use `<tools>...</tools>` format. The model outputs calls as `<tool_call><function=NAME><parameter=key>\nvalue\n</parameter></function></tool_call>`.
4. **Nemotron-H**: Uses the same `<tool_call>{"name":"...","arguments":{...}}</tool_call>` wire format as Qwen3.
5. **GPT OSS**: Tools are declared in the developer message as a TypeScript namespace (`namespace functions { type NAME = (_: { ... }) => any; }`). The model emits calls on the commentary channel as `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>`, which stops generation on the `<|call|>` token. Tool results are returned as `<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>{result}<|end|>`.
6. **DiffusionGemma**: Does not support tool-call framing in TensorSharp; use it for text-diffusion generation only.
