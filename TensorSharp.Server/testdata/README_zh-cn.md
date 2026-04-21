# TensorSharp.Server 集成测试

[English](README.md) | [中文](README_zh-cn.md)

两套测试套件用于覆盖 TensorSharp.Server 当前对外的兼容接口：

- Web UI SSE：`/api/chat`
- Ollama 聊天兼容接口：`/api/chat/ollama`
- OpenAI Chat Completions 兼容接口：`/v1/chat/completions`

测试脚本会自动检测当前加载模型的架构，并在该模型不支持思维链或工具调用时自动跳过相关用例。

## 快速开始

1. 启动 TensorSharp.Server：

```bash
./TensorSharp.Server --model ~/models/model.gguf --backend ggml_metal
```

2. 运行任一套件：

```bash
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

不支持的架构会被标记为 `SKIP`，而不是 `FAIL`。

### 仅 Bash 的运维侧检查

- Web UI 流程中的 system prompt 持久化
- 并发请求与 FIFO 队列行为
- 长对话压力测试
- Ollama / OpenAI 接口混用
- 生成中途中断与队列释放
- Ollama 工具调用请求路径

### 仅 Python 的兼容性检查

- 按架构感知地校验 OpenAI 工具调用
- 独立的通过/失败/跳过统计，并按用例输出 payload

## 注意事项

- 本目录中的 OpenAI 覆盖范围针对的是 Chat Completions 兼容接口。OpenAI 较新的 Responses API 不在 TensorSharp.Server 当前模拟的兼容范围内。
- 结构化输出遵循 Chat Completions 的 `response_format` 协议。`json_schema` 与 `tools` 或 `think` 同时使用时预期返回 HTTP `400`。
- Ollama 与 OpenAI 兼容方案仍在持续演进。这些脚本与服务端当前的契约以及在思维链、工具调用、结构化输出方面的文档化行为保持一致。

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
