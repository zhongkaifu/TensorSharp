# TensorSharp.Server Integration Tests

Two test suites exercise TensorSharp.Server's current public compatibility surface:

- Web UI SSE: `/api/chat`
- Ollama chat compatibility: `/api/chat/ollama`
- OpenAI Chat Completions compatibility: `/v1/chat/completions`

The scripts auto-detect the loaded model architecture and skip thinking or tool-calling checks when the active model does not support those capabilities.

## Quick Start

1. Start TensorSharp.Server:

```bash
MODEL_DIR=~/models BACKEND=ggml_metal ./TensorSharp.Server
```

2. Run either suite:

```bash
# Bash suite (requires curl + jq)
bash test_multiturn.sh

# Python suite (standard library only)
python3 test_multiturn.py
```

## What The Suites Cover

### Common coverage

- Web UI multi-turn SSE streaming and done events
- Ollama chat multi-turn behavior in streaming and non-streaming modes
- OpenAI Chat Completions streaming and non-streaming behavior
- OpenAI structured outputs with both `response_format: {"type":"json_object"}` and `response_format.json_schema`
- Queue status endpoint shape
- Error handling for missing required fields
- Structured-output validation errors and documented request conflicts

### Capability-gated coverage

- Thinking-mode tests run only on architectures that currently support thinking in TensorSharp:
  Gemma 4, Qwen 3, Qwen 3.5, GPT OSS, and Nemotron-H
- Tool-calling tests run only on architectures that currently support tool calling in TensorSharp:
  Gemma 4, Qwen 3, Qwen 3.5, and Nemotron-H

Unsupported architectures are reported as `SKIP`, not `FAIL`.

### Bash-only operational checks

- System-prompt persistence in the Web UI flow
- Concurrent requests and FIFO queue behavior
- Long-conversation stress test
- Mixed Ollama/OpenAI handoff
- Abort mid-generation and queue release
- Ollama tool-call request plumbing

### Python-specific compatibility checks

- Architecture-aware OpenAI tool-call validation
- Separate pass/fail/skip accounting with per-test payload dumps

## Notes

- The OpenAI coverage in this folder targets Chat Completions compatibility. OpenAI's newer Responses API is not the compatibility surface TensorSharp.Server currently emulates here.
- Structured outputs follow the Chat Completions `response_format` contract. `json_schema` requests combined with `tools` or `think` are expected to return HTTP `400`.
- The Ollama and OpenAI compatibility projects continue to evolve. These scripts are aligned with the server's current contract plus the current documented behavior around thinking, tool calling, and structured outputs.

## Usage

### Bash

```bash
bash test_multiturn.sh [model_name] [base_url]
```

Examples:

```bash
bash test_multiturn.sh
bash test_multiturn.sh gemma-4-E4B-it-Q8_0.gguf
bash test_multiturn.sh gemma-4-E4B-it-Q8_0.gguf http://host:5000
```

### Python

```bash
python3 test_multiturn.py [--model MODEL] [--url URL] [--max-tokens N]
```

Examples:

```bash
python3 test_multiturn.py
python3 test_multiturn.py --model gemma-4-E4B-it-Q8_0.gguf
python3 test_multiturn.py --max-tokens 120
```
