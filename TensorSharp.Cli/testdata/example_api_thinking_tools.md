# Thinking Mode and Tool Call Examples

[English](example_api_thinking_tools.md) | [中文](example_api_thinking_tools_zh-cn.md)

## Console Application

### Thinking Mode

Enable thinking mode with `--think`. The model will show its reasoning process before giving the final answer.

```bash
# Basic thinking mode
./TensorSharp.Cli --model model.gguf --backend ggml_metal \
  --input testdata/input_thinking.txt --think --max-tokens 500

# Thinking mode with sampling
./TensorSharp.Cli --model model.gguf --backend ggml_metal \
  --input testdata/input_thinking.txt --think --max-tokens 500 \
  --temperature 0.6 --top-p 0.95
```

### Tool Call Mode

Provide tool definitions via `--tools <file.json>`. The model will output structured tool calls.

```bash
# Weather tool call
./TensorSharp.Cli --model model.gguf --backend ggml_metal \
  --input testdata/input_tool_call.txt \
  --tools testdata/tools_weather.json --max-tokens 300

# Calculator tool call
./TensorSharp.Cli --model model.gguf --backend ggml_metal \
  --input testdata/input_tool_calc.txt \
  --tools testdata/tools_calculator.json --max-tokens 300

# Combined: thinking + tools
./TensorSharp.Cli --model model.gguf --backend ggml_metal \
  --input testdata/input_tool_call.txt \
  --tools testdata/tools_weather.json --think --max-tokens 500
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
| Qwen 3.5 | `<think>...</think>` | `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>` |
| GPT OSS | `<\|channel\|>analysis ... <\|channel\|>final` (Harmony) | not supported |
| Nemotron-H | `<think>...</think>` | `<tool_call>{"name":"...","arguments":{...}}</tool_call>` |

## How It Works

### Thinking Mode

When `think: true` is passed:

1. **Gemma4**: The template injects `<|think|>` into the system turn. The model then outputs thinking inside `<|channel>thought\n...<channel|>` tags before the actual response.
2. **Qwen3**: The template appends `<think>\n` to the generation prompt. The model outputs thinking directly, terminated by `</think>`, followed by the answer.
3. **Qwen3.5**: Same as Qwen3. When thinking is disabled, an empty `<think>\n\n</think>\n\n` block is prepended.
4. **GPT OSS**: The Harmony-format template always emits structured channel framing: `<|channel|>analysis ... <|channel|>final`. The output parser is always on for this architecture, so thinking content is split out whether or not `think: true` is passed.
5. **Nemotron-H**: Uses the Qwen3-style `<think>...</think>` framing.

### Tool Calls

When `tools` are provided:

1. **Gemma4**: Tool declarations use `<|tool>declaration:NAME{...}<tool|>` format in the system turn. The model outputs calls as `<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>`.
2. **Qwen3**: Tool definitions are injected as JSON in the system message. The model outputs calls as `<tool_call>{"name":"...","arguments":{...}}</tool_call>`.
3. **Qwen3.5**: Tool definitions use `<tools>...</tools>` format. The model outputs calls as `<tool_call><function=NAME><parameter=key>\nvalue\n</parameter></function></tool_call>`.
4. **Nemotron-H**: Uses the same `<tool_call>{"name":"...","arguments":{...}}</tool_call>` wire format as Qwen3.
5. **GPT OSS**: tool-calling is not currently supported.

