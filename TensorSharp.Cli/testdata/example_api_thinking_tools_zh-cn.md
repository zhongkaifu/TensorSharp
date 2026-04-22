# 思维链与工具调用示例

[English](example_api_thinking_tools.md) | [中文](example_api_thinking_tools_zh-cn.md)

## 控制台应用

### 思维链模式

通过 `--think` 启用思维链。模型会先输出推理过程，再给出最终答案。

```bash
# 基础思维链模式
./TensorSharp.Cli --model model.gguf --backend ggml_metal \
  --input testdata/input_thinking.txt --think --max-tokens 500

# 带采样参数的思维链模式
./TensorSharp.Cli --model model.gguf --backend ggml_metal \
  --input testdata/input_thinking.txt --think --max-tokens 500 \
  --temperature 0.6 --top-p 0.95
```

### 工具调用模式

通过 `--tools <file.json>` 提供工具定义。模型会输出结构化的工具调用。

```bash
# 天气工具调用
./TensorSharp.Cli --model model.gguf --backend ggml_metal \
  --input testdata/input_tool_call.txt \
  --tools testdata/tools_weather.json --max-tokens 300

# 计算器工具调用
./TensorSharp.Cli --model model.gguf --backend ggml_metal \
  --input testdata/input_tool_calc.txt \
  --tools testdata/tools_calculator.json --max-tokens 300

# 思维链 + 工具组合
./TensorSharp.Cli --model model.gguf --backend ggml_metal \
  --input testdata/input_tool_call.txt \
  --tools testdata/tools_weather.json --think --max-tokens 500
```

### 工具定义格式

工具以 JSON 数组形式提供，每个元素是一个 `ToolFunction` 对象：

```json
[
  {
    "Name": "function_name",
    "Description": "函数功能描述",
    "Parameters": {
      "param1": {
        "Type": "string",
        "Description": "参数描述"
      },
      "param2": {
        "Type": "string",
        "Description": "另一个参数",
        "Enum": ["option1", "option2"]
      }
    },
    "Required": ["param1"]
  }
]
```

## Web API（兼容 Ollama）

### 通过 Ollama API 使用思维链

```bash
curl -s http://localhost:5000/api/chat/ollama -d '{
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "messages": [{"role": "user", "content": "How many r'\''s in strawberry?"}],
  "think": true,
  "stream": false,
  "options": {"num_predict": 500}
}'
```

返回值的 message 中包含独立的 `thinking` 字段：

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

### 通过 Ollama API 使用工具调用

```bash
curl -s http://localhost:5000/api/chat/ollama -d '{
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "tools": [
    {
      "function": {
        "name": "get_current_weather",
        "description": "获取指定位置的当前天气",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "城市名"},
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

当模型决定调用工具时，响应中会包含 `tool_calls`：

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

### 通过 Ollama API 同时使用思维链 + 工具

```bash
curl -s http://localhost:5000/api/chat/ollama -d '{
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
  "think": true,
  "tools": [
    {
      "function": {
        "name": "get_current_weather",
        "description": "获取指定位置的当前天气",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "城市名"}
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

返回中同时包含 `thinking` 和 `tool_calls`：

```json
{
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "message": {
    "role": "assistant",
    "content": "",
    "thinking": "用户想知道巴黎的天气。我应该调用 get_current_weather。",
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

## Web API（兼容 OpenAI）

### 通过 OpenAI API 使用思维链

```bash
curl -s http://localhost:5000/v1/chat/completions -d '{
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "messages": [{"role": "user", "content": "How many r'\''s in strawberry?"}],
  "think": true,
  "max_tokens": 500
}'
```

### 通过 OpenAI API 使用工具调用

```bash
curl -s http://localhost:5000/v1/chat/completions -d '{
  "model": "gemma-4-E4B-it-Q8_0.gguf",
  "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取指定位置的天气",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "城市名"},
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

响应（OpenAI 格式）：

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

## Python 示例

### 用 Python 使用思维链

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
    print("=== 思维过程 ===")
    print(data["message"]["thinking"])
    print("=== 最终回答 ===")
print(data["message"]["content"])
```

### 用 Python（OpenAI SDK 兼容）调用工具

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="unused")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取当前天气",
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
        print(f"工具：{tc.function.name}")
        print(f"参数：{tc.function.arguments}")
else:
    print(choice.message.content)
```

## 支持的模型架构

| 架构 | 思维标签 | 工具调用标签 |
|---|---|---|
| Gemma 4 | `<\|channel>thought\n...<channel\|>` | `<\|tool_call>call:NAME{args}<tool_call\|>` |
| Qwen 3 | `<think>...</think>` | `<tool_call>{"name":"...","arguments":{...}}</tool_call>` |
| Qwen 3.5 | `<think>...</think>` | `<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>` |
| GPT OSS | `<\|channel\|>analysis ... <\|channel\|>final`（Harmony 格式） | 暂不支持 |
| Nemotron-H | `<think>...</think>` | `<tool_call>{"name":"...","arguments":{...}}</tool_call>` |

## 工作原理

### 思维链模式

当传入 `think: true` 时：

1. **Gemma4**：模板会在 system 段注入 `<|think|>`。模型会先在 `<|channel>thought\n...<channel|>` 标签内输出思维过程，再给出实际回答。
2. **Qwen3**：模板会在生成 prompt 末尾追加 `<think>\n`。模型直接输出思维内容，并以 `</think>` 结尾，之后给出答案。
3. **Qwen3.5**：与 Qwen3 相同。当思维链被禁用时，会在前面插入一个空的 `<think>\n\n</think>\n\n` 块。
4. **GPT OSS**：Harmony 模板始终输出结构化的 channel 框架：`<|channel|>analysis ... <|channel|>final`。该架构的输出解析器始终启用，无论是否传入 `think: true`，都会拆出思维内容。
5. **Nemotron-H**：使用与 Qwen3 一致的 `<think>...</think>` 框架。

### 工具调用

当提供 `tools` 时：

1. **Gemma4**：工具声明在 system 段使用 `<|tool>declaration:NAME{...}<tool|>` 格式。模型输出调用为 `<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>`。
2. **Qwen3**：工具定义以 JSON 形式注入到 system message。模型输出调用为 `<tool_call>{"name":"...","arguments":{...}}</tool_call>`。
3. **Qwen3.5**：工具定义使用 `<tools>...</tools>` 格式。模型输出调用为 `<tool_call><function=NAME><parameter=key>\nvalue\n</parameter></function></tool_call>`。
4. **Nemotron-H**：与 Qwen3 共用相同的 `<tool_call>{"name":"...","arguments":{...}}</tool_call>` 线协议。
5. **GPT OSS**：当前未支持工具调用。
