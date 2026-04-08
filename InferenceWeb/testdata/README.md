# InferenceWeb Multi-Turn Chat Integration Tests

Two test suites that simulate real users having long multi-turn conversations with InferenceWeb across all API surfaces.

## Quick Start

1. Start InferenceWeb:
```bash
MODEL_DIR=~/models BACKEND=ggml_metal ./InferenceWeb
```

2. Run tests (pick one):
```bash
# Bash test suite (requires curl + jq)
bash test_multiturn.sh

# Python test suite (no extra dependencies)
python3 test_multiturn.py
```

## Test Coverage

| # | Test | API | Turns | What it verifies |
|---|------|-----|-------|------------------|
| 1 | Basic multi-turn | Web UI `/api/chat` | 5 | SSE streaming, token accumulation, done event |
| 2 | Context retention | Ollama `/api/chat/ollama` | 4-7 | Model remembers facts (names, numbers, cities) across turns |
| 3 | Non-streaming | Ollama `/api/chat/ollama` | 4 | Non-streaming JSON responses, math follow-ups |
| 4 | Streaming metrics | Ollama `/api/chat/ollama` | 3 | NDJSON streaming, `done_reason`, `eval_count`, `total_duration` |
| 5 | OpenAI streaming | `/v1/chat/completions` | 4-5 | SSE chunks, `delta.content`, `finish_reason`, `[DONE]` |
| 6 | Structured outputs | `/v1/chat/completions` | 1 | `response_format.json_schema`, JSON validation, required keys |
| 7 | OpenAI non-streaming | `/v1/chat/completions` | 4 | Full JSON response, `choices[0].message.content` |
| 8 | System message + long | Web UI `/api/chat` | 8 | System prompt persistence over many turns |
| 9 | Queue status | `/api/queue/status` | - | `busy`, `pending_requests`, `total_processed` fields |
| 10 | Concurrent requests | Web UI `/api/chat` | - | FIFO queue handles 3 simultaneous requests |
| 11 | Thinking mode | Ollama `/api/chat/ollama` | 3 | `think: true` parameter, multi-turn with reasoning |
| 12 | Long conversation | Ollama `/api/chat/ollama` | 12 | Stress test - 25 messages total |
| 13 | Mixed API | Ollama + OpenAI | 3 | Same conversation across different API formats |
| 14 | Error handling | All | - | Missing fields return 400, invalid structured schemas rejected |
| 15 | Tool calls | Ollama `/api/chat/ollama` | 3 | Multi-turn with tool definitions |
| 16 | Abort | Web UI `/api/chat` | 1 | Mid-generation abort, queue release after abort |

## Options

### Bash script
```bash
bash test_multiturn.sh [model_name] [base_url]

# Examples:
bash test_multiturn.sh                                          # auto-detect
bash test_multiturn.sh gemma-4-E4B-it-Q8_0.gguf                # specific model
bash test_multiturn.sh gemma-4-E4B-it-Q8_0.gguf http://host:5000  # remote server
```

### Python script
```bash
python3 test_multiturn.py [--model MODEL] [--url URL] [--max-tokens N]

# Examples:
python3 test_multiturn.py                                       # auto-detect
python3 test_multiturn.py --model gemma-4-E4B-it-Q8_0.gguf     # specific model
python3 test_multiturn.py --max-tokens 120                      # longer responses
```

## What the tests validate

- **Response structure**: Each API returns data in the correct format (SSE, NDJSON, JSON)
- **Context retention**: Model remembers facts, names, and numbers from earlier turns
- **Multi-turn coherence**: Follow-up questions get contextually appropriate answers
- **Done/finish signals**: Every response properly terminates with done event
- **Streaming correctness**: Tokens accumulate correctly across chunks
- **Queue behavior**: Concurrent requests are serialized, queue status is accurate
- **Error handling**: Invalid requests return proper HTTP error codes
- **Structured outputs**: OpenAI-style `response_format` schemas are validated and enforced
- **Abort support**: Mid-generation cancellation works and releases the queue
- **Metrics**: Timing and token count metrics are present in done events
