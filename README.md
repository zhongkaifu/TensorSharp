# TensorSharp

<p align="center">
  <img src="imgs/banner_1.png" alt="TensorSharp logo" width="320">
</p>

[English](README.md) | [中文](README_zh-cn.md)

A C# inference engine for running large language models (LLMs) locally using GGUF model files. TensorSharp provides a console application, a web-based chatbot interface, and Ollama/OpenAI-compatible HTTP APIs for programmatic access.

## Features

- **Multi-architecture support** -- Gemma 4, Gemma 3, Qwen 3, Qwen 3.5, GPT OSS, Nemotron-H, Mistral 3
- **Multimodal inference** -- image, video, and audio inputs (Gemma 4); images for Gemma 3 / Qwen 3.5 / Mistral 3
- **Thinking / reasoning mode** -- structured chain-of-thought output with `<think>` / `<|channel>thought` / `<|channel>analysis` tags (Qwen 3, Qwen 3.5, Gemma 4, GPT OSS, Nemotron-H)
- **Tool calling / function calling** -- models can invoke user-defined tools; multi-turn tool-call conversations supported across all three API styles
- **Quantized model support** -- loads GGUF files with Q4_K_M, Q8_0, F16, MXFP4, and other quantization formats; performs native quantized matmul without dequantizing to FP32, including memory-efficient pure C# CPU loading for large GGUFs
- **GPU-accelerated** -- GGML Metal on macOS and GGML CUDA on Linux/NVIDIA, with fused whole-model GPU dispatch for Gemma 4 decode and prefill on Metal (~2.6x speedup over per-op dispatch)
- **Optimized pure C# CPU backend** -- managed GEMM fast paths plus fused SIMD kernels for RMSNorm, RoPE, softmax, fused activations, and other inference hot paths
- **Ollama & OpenAI API compatibility** -- drop-in replacement endpoints for existing tooling
- **Configurable sampling** -- temperature, top-k, top-p, min-p, repetition/presence/frequency penalties, seed, stop sequences
- **Chat templates** -- auto-loaded from GGUF metadata (Jinja2), with hardcoded fallbacks per architecture
- **Request queue** -- FIFO inference queue ensures single-request execution for KV cache stability, with real-time position tracking for clients
- **Batch processing** -- JSONL input support in the console application, plus a built-in inference benchmark for prefill/decode throughput
- **Streaming** -- token-by-token output via SSE (web) or stdout (console), with abort/stop support for in-flight generations
- **Hybrid SSM-Transformer** -- Nemotron-H mixes Mamba2 SSM layers, attention-only layers, and MoE FFN layers in a single model
- **Hybrid Attention-Recurrent** -- Qwen 3.5 mixes full-attention layers with GatedDeltaNet recurrent layers
- **Mixture of Experts** -- Gemma 4 MoE variants (e.g. gemma-4-26B-A4B), GPT OSS MoE (e.g. gpt-oss-20b), Qwen 3.5 MoE (`qwen35moe` / `qwen3next` variants such as Qwen3.5-35B-A3B), and Nemotron-H MoE FFN layers
- **Batched GPU MoE** -- a single fused GGML graph dispatch handles all selected experts (plus the optional shared expert and residual add) for Qwen 3.5 and Nemotron-H decode, eliminating per-expert round-trips
- **Message editing** -- edit or delete previous messages in the web chat UI and regenerate from that point
- **Text/Image/Audio/Video uploads** -- the web UI accepts file uploads up to 500 MB, with automatic token-budget-aware truncation for large text files
- **Per-turn observability** -- structured logs capture the full user input and the full raw assistant output (both `<think>` reasoning and the final result) plus the KV cache hit ratio. The same cache-hit stats are surfaced through every API: `prompt_cache_hit_tokens` / `prompt_cache_hit_ratio` (Ollama), `usage.prompt_tokens_details.cached_tokens` (OpenAI), and `promptTokens` / `kvReusedTokens` / `kvReusePercent` in the Web UI SSE `done` event

## Supported Model Architectures

| Architecture | GGUF arch keys | Example Models | Multimodal | Thinking | Tool Calling |
|---|---|---|---|---|---|
| Gemma 4 | `gemma4` | gemma-4-E4B, gemma-4-31B, gemma-4-26B-A4B (MoE) | Image, Video, Audio | Yes | Yes |
| Gemma 3 | `gemma3` | gemma-3-4b | Image | No | No |
| Qwen 3 | `qwen3` | Qwen3-4B | Text only | Yes | Yes |
| Qwen 3.5 | `qwen35`, `qwen35moe`, `qwen3next` | Qwen3.5-9B (hybrid Attn+Recurrent), Qwen3.5-35B-A3B (MoE) | Image | Yes | Yes |
| GPT OSS | `gptoss`, `gpt-oss` | gpt-oss-20b (MoE) | Text only | Yes | No |
| Nemotron-H | `nemotron_h`, `nemotron_h_moe` | Nemotron-H-8B, Nemotron-H-47B (Hybrid SSM-Transformer, MoE) | Text only | Yes | Yes |
| Mistral 3 | `mistral3` | Mistral-Small-3.1-24B-Instruct | Image | No | No |

See [Model Architecture Cards](docs/model_cards.md) for detailed documentation of each architecture.

## Model Downloads (GGUF)

TensorSharp loads models in GGUF format. Below are Hugging Face links where you can download GGUF files for each supported architecture. Pick a quantization that fits your hardware (Q4_K_M for low memory, Q8_0 for higher quality, etc.).

| Architecture | Model | GGUF Download |
|---|---|---|
| Gemma 4 | gemma-4-E4B-it | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) |
| Gemma 4 | gemma-4-31B-it | [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) |
| Gemma 4 | gemma-4-26B-A4B-it (MoE) | [ggml-org/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) |
| Gemma 4 | gemma-4-mmproj (multimodal projector) | Included in the GGUF repos above |
| Gemma 3 | gemma-3-4b-it | [google/gemma-3-4b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf) |
| Qwen 3 | Qwen3-4B | [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF) |
| Qwen 3.5 | Qwen3.5-9B | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) |
| Qwen 3.5 | Qwen3.5-35B-A3B | [ggml-org/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF) |
| GPT OSS | gpt-oss-20b (MoE) | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) |
| Nemotron-H | Nemotron-H-8B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-8B-Reasoning-128K-GGUF) |
| Nemotron-H | Nemotron-H-47B-Reasoning-128K | [bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-H-47B-Reasoning-128K-GGUF) |
| Mistral 3 | Mistral-Small-3.1-24B-Instruct | [bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF) |
| Mistral 3 | mistral3-mmproj (Pixtral vision projector) | [bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF](https://huggingface.co/bartowski/Mistral-Small-3.1-24B-Instruct-2503-GGUF) |

## Compute Backends

| Backend | Flag | Description |
|---|---|---|
| GGML Metal | `--backend ggml_metal` | GPU-accelerated via Apple Metal (macOS). Recommended for Apple Silicon. Quantized weights are mapped zero-copy from the GGUF file into Metal command buffers via host-pointer buffers, so the resident set stays close to the on-disk model size. |
| GGML CUDA | `--backend ggml_cuda` | GPU-accelerated via GGML CUDA on Linux with an NVIDIA GPU. Quantized weights are uploaded to device memory once at load time and the host copy is released afterwards. |
| GGML CPU | `--backend ggml_cpu` | CPU inference using native GGML with optimized kernels. Quantized weights are mapped zero-copy from the GGUF file. |
| Pure C# CPU | `--backend cpu` | Portable CPU inference with no native dependencies. |

## Project Structure

```
TensorSharp/
├── TensorSharp.Core/            # Core tensor library (Tensor, Ops, memory, device abstraction)
├── TensorSharp.Runtime/         # GGUF, tokenizers, templates, sampling, protocol parsing
├── TensorSharp.Models/          # Model architectures and multimodal encoders/injectors
├── TensorSharp.Backends.GGML/   # GGML backend bindings (Metal/CUDA/CPU via native library)
├── TensorSharp.GGML.Native/     # Native C++ bridge to ggml (builds libGgmlOps, split into focused source files)
├── TensorSharp.Server/          # Web chatbot + API server (ASP.NET Core)
│   ├── Program.cs               # Slim bootstrap: DI wiring, middleware, endpoint mapping
│   ├── ModelService.cs          # Model lifecycle, KV cache reuse, chat/generate streams + per-turn metrics
│   ├── ChatSession.cs           # Per-conversation KV cache + tracked history
│   ├── SessionManager.cs        # Thread-safe session registry (default + per-tab sessions)
│   ├── InferenceQueue.cs        # FIFO request queue with position tracking
│   ├── BackendCatalog.cs        # Discovery of available compute backends
│   ├── TextUploadHelper.cs      # Token-budget-aware text-file truncation
│   ├── WebUiChatPolicy.cs       # Web UI chat request validation
│   ├── OpenAIResponseFormatParser.cs  # OpenAI response_format (json_object / json_schema) parsing
│   ├── Hosting/                 # Startup-time concerns: options, backend resolution, logging, web root
│   ├── RequestParsers/          # JSON request parsing (sampling, chat messages, tool functions)
│   ├── ResponseSerializers/     # Per-protocol response shape factories (Ollama, OpenAI, Web UI)
│   ├── StreamingWriters/        # SSE + NDJSON wire-format helpers
│   ├── ProtocolAdapters/        # Per-protocol request handlers (WebUiAdapter, OllamaAdapter, OpenAIChatAdapter)
│   ├── Endpoints/               # ASP.NET Core endpoint mapping (one extension method per protocol)
│   ├── Logging/                 # Request logging middleware + low-noise path support
│   ├── wwwroot/index.html       # Chat UI
│   ├── testdata/                # Integration test suites (bash + Python)
│   └── API_EXAMPLES.md          # Detailed API documentation
├── TensorSharp.Cli/             # CLI application
├── InferenceWeb.Tests/          # xUnit unit tests covering ops, KV cache, web/server helpers
├── AdvUtils/                    # Utility library
├── docs/                        # Developer reference (model cards, EN + 中文)
└── ExternalProjects/            # Third-party dependencies (ggml)
```

## NuGet Packages

The repository is now split along package boundaries so consumers can depend on only the layers they actually need.

| Project | NuGet package | Public namespace | Responsibility |
|---|---|---|---|
| `TensorSharp.Core` | `TensorSharp.Core` | `TensorSharp` | Tensor primitives, ops, allocators, storage, and device abstraction |
| `TensorSharp.Runtime` | `TensorSharp.Runtime` | `TensorSharp.Runtime` | GGUF parsing, tokenizers, prompt rendering, sampling, and output protocol parsing |
| `TensorSharp.Models` | `TensorSharp.Models` | `TensorSharp.Models` | `ModelBase`, architecture implementations, multimodal encoders, and model-side execution helpers |
| `TensorSharp.Backends.GGML` | `TensorSharp.Backends.GGML` | `TensorSharp.GGML` | GGML-backed execution and native interop |
| `TensorSharp.Server` | `TensorSharp.Server` | `TensorSharp.Server` | ASP.NET Core server, OpenAI/Ollama adapters, queueing, and web UI |
| `TensorSharp.Cli` | `TensorSharp.Cli` | `TensorSharp.Cli` | Console host and debugging / batch tooling |

This split keeps engine users off the web stack, keeps API-layer changes from leaking into core/runtime packages, and makes future benchmark or eval-harness projects easier to publish independently.

## Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0)
- **macOS (Metal backend):** CMake 3.20+ and Xcode command-line tools for building the native GGML library
- **Linux (GGML CPU / CUDA backends):** CMake 3.20+; for `ggml_cuda`, install an NVIDIA driver plus CUDA Toolkit 12.x or another compatible CUDA toolkit
- GGUF model files (e.g., from [Hugging Face](https://huggingface.co))

## Building

### Build the entire solution

```bash
dotnet build TensorSharp.slnx
```

### Build individual applications

```bash
# Console application
dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj

# Web application
dotnet build TensorSharp.Server/TensorSharp.Server.csproj
```

### Build the native GGML library

The native library is built automatically during the first `dotnet build` if it doesn't exist. To build it manually:

```bash
cd TensorSharp.GGML.Native
```

macOS:

```bash
bash build-macos.sh
```

Linux (CPU-only):

```bash
bash build-linux.sh
```

Linux (GGML_CUDA enabled):

```bash
bash build-linux.sh --cuda
```

On Linux, `build-linux.sh` now auto-detects the visible NVIDIA GPU compute capability and passes a narrow `CMAKE_CUDA_ARCHITECTURES` value to ggml-cuda (for example `86-real` on an RTX 3080), which cuts CUDA build time substantially. The native build also runs in parallel by default with a conservative job cap so `nvcc` does not overwhelm typical developer machines.

If you want to override the auto-detected architecture list or the default build parallelism, use either environment variables or explicit build flags:

```bash
TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES='86-real;89-real' bash build-linux.sh --cuda
bash build-linux.sh --cuda --cuda-arch='86-real;89-real'
TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL=2 bash build-linux.sh --cuda
```

You can also request a CUDA-enabled native build from `dotnet build`:

```bash
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON dotnet build TensorSharp.Cli/TensorSharp.Cli.csproj -c Release
```

On macOS this compiles `libGgmlOps.dylib` with Metal GPU support. On Linux, `build-linux.sh` preserves an existing CUDA-enabled build and auto-enables GGML_CUDA when a CUDA toolchain is detected; `build-linux.sh --cuda` and `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` force CUDA explicitly. The build output is automatically copied to the application's output directory.

## Usage

### Console Application

```bash
cd TensorSharp.Cli/bin

# Text inference
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_metal

# Text inference on Linux + NVIDIA GPU
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_cuda

# Interactive turn-by-turn chat (REPL) with KV cache reuse and slash commands
./TensorSharp.Cli --model <model.gguf> --backend ggml_metal --interactive
./TensorSharp.Cli --model <model.gguf> --backend ggml_metal -i \
    --system "You are a terse assistant." --temperature 0.7 --top-p 0.9 --think

# Image inference (Gemma 3/4, Qwen 3.5)
./TensorSharp.Cli --model <model.gguf> --image photo.png --backend ggml_metal

# Video inference (Gemma 4)
./TensorSharp.Cli --model <model.gguf> --video clip.mp4 --backend ggml_metal

# Audio inference (Gemma 4)
./TensorSharp.Cli --model <model.gguf> --audio speech.wav --backend ggml_metal

# Thinking / reasoning mode
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --backend ggml_metal --think

# Tool calling
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --tools tools.json

# With sampling parameters
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.2 --seed 42

# Batch processing (JSONL)
./TensorSharp.Cli --model <model.gguf> --input-jsonl requests.jsonl \
    --output results.txt --backend ggml_metal

# Multi-turn chat simulation with KV-cache reuse (mirrors the web UI behavior)
./TensorSharp.Cli --model <model.gguf> --multi-turn-jsonl chat.jsonl \
    --backend ggml_metal --max-tokens 200

# Throughput benchmark: best-of-N prefill and decode timing
./TensorSharp.Cli --model <model.gguf> --backend ggml_metal \
    --benchmark --bench-prefill 256 --bench-decode 128 --bench-runs 3

# KV-cache reuse benchmark: measure prefill speedup across multiple chat turns
# (compares with-cache vs forced-reset prefill latency for an 8-turn conversation)
./TensorSharp.Cli --model <model.gguf> --backend ggml_metal \
    --bench-kvcache --bench-kv-turns 4 --max-tokens 64

# Inspect the rendered prompt and tokenization without running inference
./TensorSharp.Cli --model <model.gguf> --input prompt.txt --dump-prompt

# Compare hardcoded fallback templates against GGUF Jinja2 templates for every
# *.gguf file in a directory (useful when adding new architectures)
./TensorSharp.Cli --test-templates ~/models
```

**Command-line options:**

| Option | Description |
|---|---|
| `--model <path>` | Path to a GGUF model file (required) |
| `--input <path>` | Text file containing the user prompt |
| `--input-jsonl <path>` | JSONL file with batch requests (one JSON per line) |
| `--multi-turn-jsonl <path>` | JSONL file for multi-turn chat simulation with KV cache reuse |
| `--output <path>` | Write generated text to this file |
| `--image <path>` | Image file for vision inference |
| `--video <path>` | Video file for video inference |
| `--audio <path>` | Audio file (WAV, MP3, OGG) for audio inference |
| `--mmproj <path>` | Path to the multimodal projector GGUF file |
| `--max-tokens <N>` | Maximum tokens to generate (default: 100) |
| `--backend <type>` | Compute backend: `cpu`, `ggml_cpu`, `ggml_metal`, or `ggml_cuda` |
| `--interactive` / `-i` | Start an interactive REPL chat session (turn-by-turn input/output) with KV cache reuse, slash commands, hot-swappable model/backend/projector, file attachments (image, audio, video, text) and live sampling tuning. See the **Interactive REPL commands** section below for the full list. |
| `--system <text>` | System prompt to seed the interactive session (overridden inside the REPL by `/system`) |
| `--system-file <path>` | Read the initial system prompt from a UTF-8 text file (alternative to `--system`) |
| `--think` | Enable thinking/reasoning mode (chain-of-thought) |
| `--tools <path>` | JSON file with tool/function definitions |
| `--temperature <f>` | Sampling temperature (0 = greedy) |
| `--top-k <N>` | Top-K filtering (0 = disabled) |
| `--top-p <f>` | Nucleus sampling threshold (1.0 = disabled) |
| `--min-p <f>` | Minimum probability filtering (0 = disabled) |
| `--repeat-penalty <f>` | Repetition penalty (1.0 = none) |
| `--presence-penalty <f>` | Presence penalty (0 = disabled) |
| `--frequency-penalty <f>` | Frequency penalty (0 = disabled) |
| `--seed <N>` | Random seed (-1 = non-deterministic) |
| `--stop <string>` | Stop sequence (can be repeated) |
| `--dump-prompt` | Render the prompt + tokenization and exit (no generation) |
| `--benchmark` | Run a synthetic prefill/decode throughput benchmark |
| `--bench-prefill <N>` | Synthetic prefill length in tokens (default: 32) |
| `--bench-decode <N>` | Synthetic decode length in tokens (default: 64) |
| `--bench-runs <N>` | Number of benchmark runs; reports best and average (default: 1) |
| `--bench-kvcache` | Run a multi-turn KV-cache reuse benchmark (with-cache vs forced-reset prefill) |
| `--bench-kv-turns <N>` | Number of conversation turns for `--bench-kvcache` (default: 4, max: 8) |
| `--test` | Run built-in tokenizer + Qwen3 chat-template + ollama-comparison tests |
| `--test-templates <dir>` | Validate hardcoded chat templates against GGUF Jinja2 templates for every *.gguf in `<dir>` |
| `--log-level <lvl>` | Console + file logger level: `trace`, `debug`, `info`, `warning`, `error`, `critical`, `off` |
| `--log-dir <path>` | Directory for the JSON-line file logger (default: `<binDir>/logs`) |
| `--log-file <0\|1>` | Disable (`0`) or enable (`1`) the file logger (default: enabled) |
| `--log-console <0\|1>` | Disable (`0`) or enable (`1`) the console logger (default: enabled) |

The multimodal projector file is auto-detected if placed alongside the model file with a recognized name (e.g., `gemma-4-mmproj-F16.gguf`).

**JSONL input format:**

Each line is a JSON object with `messages`, optional `prompt`, and optional sampling parameters:

```json
{"id": "q1", "messages": [{"role": "user", "content": "What is 2+3?"}], "max_tokens": 50}
{"id": "q2", "messages": [{"role": "user", "content": "Write a haiku."}], "max_tokens": 100, "temperature": 0.8}
```

**Interactive REPL commands:**

Once the CLI is launched with `--interactive` / `-i`, you can drive the running session with slash commands. Type `/help` (or `/?`) inside the REPL for the same list. Anything that does not start with `/` is treated as a user turn.

The prompt header summarizes the current state on every turn — model, backend, architecture, context length, projector, conversation depth, and any attachments queued for the next turn (e.g. `[turn 3 (2 attachments pending)]> `). Press Ctrl+C while generating to interrupt the current reply; press Ctrl+C at the prompt to exit.

Conversation:

| Command | Description |
|---|---|
| `/help`, `/?` | Show all interactive commands |
| `/exit`, `/quit` | Leave the session |
| `/reset`, `/new` | Clear conversation history and KV cache |
| `/history` | Print the conversation history |
| `/save <file>` | Append the current transcript to a UTF-8 file |
| `/system <text>` | Set the system prompt (empty argument clears it). Resets KV cache. |
| `/think on\|off` | Toggle thinking/reasoning mode for supported models |
| `/multiline on\|off` | Toggle multi-line input (terminate the message with a single `.` on its own line) |

Model and runtime:

| Command | Description |
|---|---|
| `/info`, `/status` | Show the loaded model, backend, architecture, context/vocab size, projector, conversation depth, and pending attachments |
| `/model <path>` | Load a different `.gguf` model on the current backend (resets the session) |
| `/backend <name>` | Reload the current model on a different backend: `cpu`, `ggml_cpu`, `ggml_metal`, or `ggml_cuda` |
| `/mmproj <path>` | Load (or replace) the multimodal projector for the current model. Aliases: `/projector` |

Sampling (live, persists across turns):

| Command | Description |
|---|---|
| `/sampling`, `/show` | Print the current sampling configuration |
| `/max <N>` | Maximum reply length in tokens |
| `/temp <float>` | Sampling temperature (0 = greedy) |
| `/topk <int>` | Top-K filtering (0 = disabled) |
| `/topp <float>` | Top-P / nucleus threshold (1.0 = disabled) |
| `/minp <float>` | Min-P filtering (0 = disabled) |
| `/repeat <float>` | Repetition penalty (1.0 = none) |
| `/presence <float>` | Presence penalty |
| `/frequency <float>` | Frequency penalty |
| `/seed <int>` | Random seed (-1 = non-deterministic) |
| `/stop <text>` | Add a stop sequence |
| `/clearstop` | Remove all stop sequences |

Uploads (queued for the next user turn, then auto-cleared after the turn):

| Command | Description |
|---|---|
| `/image <path>`, `/img <path>` | Attach an image (vision-capable models only) |
| `/audio <path>` | Attach an audio file (Gemma 4) |
| `/video <path>`, `/vid <path>` | Attach a video; frames are extracted automatically (Gemma 4) |
| `/text <path>`, `/file <path>`, `/txt <path>` | Inline a UTF-8 text/markdown/csv/code file into the next prompt (large files are token-budget truncated) |
| `/clearattach` | Drop any pending image/audio/video/text attachments without sending a turn |

Quoted paths (single or double quotes) are accepted, so drag-and-drop from a file manager works on macOS. Multimodal commands require a multimodal projector to be loaded — pass `--mmproj` at startup or use `/mmproj <path>` from the REPL.

### Web Application

```bash
cd TensorSharp.Server/bin

# Start the server with the exact hosted model
./TensorSharp.Server --model ./models/model.gguf --backend ggml_metal

# Linux + NVIDIA GPU
./TensorSharp.Server --model ./models/model.gguf --backend ggml_cuda

# Multimodal models: host an explicit projector too
./TensorSharp.Server --model ./models/model.gguf --mmproj ./models/mmproj.gguf --backend ggml_cuda

# Configure server-wide default sampling parameters
# (used whenever a request does not override the value itself)
./TensorSharp.Server --model ./models/model.gguf --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 \
    --presence-penalty 0.0 --frequency-penalty 0.0 --seed 42 \
    --stop "</s>" --stop "<|endoftext|>"
```

Open `http://localhost:5000` in your browser. The web interface supports:

- Multi-turn chat conversations
- Per-tab chat sessions: each browser tab owns its own KV cache; clicking "New Chat" disposes the current session server-side so its cache is released
- A single hosted GGUF selected explicitly with `--model`
- An explicit hosted multimodal projector via `--mmproj` when needed
- Image, video, and audio uploads for multimodal inference (up to 500 MB)
- Thinking/reasoning mode toggle
- Tool calling with function definitions
- Streaming token generation via Server-Sent Events
- Request queue with real-time position feedback
- Message editing and deletion with regeneration from any point in the conversation
- Free scrolling: scroll up to read earlier replies while new tokens stream in; the chat auto-scrolls again as soon as the user scrolls back to the bottom

Use `--model` to choose the hosted GGUF file and `--mmproj` to choose the hosted projector. `TensorSharp.Server` no longer scans a `MODEL_DIR`.

**Server command-line options:**

| Option | Description |
|---|---|
| `--model <path>` | GGUF file to host (required for inference; if omitted, the server starts but `/api/models/load` will report no hosted model) |
| `--mmproj <path>` | Multimodal projector GGUF (resolved relative to the model directory when only a filename is given; pass `none` to disable). Requires `--model`. |
| `--backend <type>` | Default compute backend: `cpu`, `ggml_cpu`, `ggml_metal`, or `ggml_cuda` |
| `--max-tokens <N>` | Default maximum tokens to generate when a request omits the limit (default: `20000`) |
| `--temperature <f>` | Default sampling temperature when a request does not provide one (`0` = greedy) |
| `--top-k <N>` | Default top-K filtering when a request does not provide one (`0` = disabled) |
| `--top-p <f>` | Default nucleus sampling threshold when a request does not provide one (`1.0` = disabled) |
| `--min-p <f>` | Default min-p filtering when a request does not provide one (`0` = disabled) |
| `--repeat-penalty <f>` | Default repetition penalty when a request does not provide one (`1.0` = none) |
| `--presence-penalty <f>` | Default presence penalty when a request does not provide one (`0` = disabled) |
| `--frequency-penalty <f>` | Default frequency penalty when a request does not provide one (`0` = disabled) |
| `--seed <N>` | Default random seed when a request does not provide one (`-1` = non-deterministic) |
| `--stop <string>` | Default stop sequence (can be repeated). Per-request `stop`/`stop_sequences` fully replace the default list rather than merge with it. |

Per-request fields in the chat / generate JSON payloads (e.g. `temperature`,
`top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`,
`frequency_penalty`, `seed`, `stop`/`stop_sequences`) always win over these
server-wide defaults; the defaults only fill in fields the client omits.

**Runtime environment variables:**

| Variable | Description |
|---|---|
| `BACKEND` | Default compute backend, used when `--backend` is not passed (default: `ggml_metal` on macOS, `ggml_cpu` elsewhere) |
| `MAX_TOKENS` | Default maximum generation length when neither `--max-tokens` nor a request-level limit is set (default: `20000`) |
| `MAX_TEXT_FILE_CHARS` | Character cap used to truncate plain-text uploads when no tokenizer is available (default: `8000`) |
| `VIDEO_MAX_FRAMES` | Maximum evenly spaced video frames extracted for video prompts (default: `4`) |
| `PORT` / `ASPNETCORE_URLS` | Standard ASP.NET Core listener configuration (default port: `5000`) |
| `TENSORSHARP_TEMPERATURE` | Default sampling temperature when neither `--temperature` nor the request body sets one |
| `TENSORSHARP_TOP_K` | Default top-K when neither `--top-k` nor the request body sets one |
| `TENSORSHARP_TOP_P` | Default top-P when neither `--top-p` nor the request body sets one |
| `TENSORSHARP_MIN_P` | Default min-P when neither `--min-p` nor the request body sets one |
| `TENSORSHARP_REPEAT_PENALTY` | Default repetition penalty when neither `--repeat-penalty` nor the request body sets one |
| `TENSORSHARP_PRESENCE_PENALTY` | Default presence penalty when neither `--presence-penalty` nor the request body sets one |
| `TENSORSHARP_FREQUENCY_PENALTY` | Default frequency penalty when neither `--frequency-penalty` nor the request body sets one |
| `TENSORSHARP_SEED` | Default random seed when neither `--seed` nor the request body sets one |
| `TENSORSHARP_LOG_LEVEL` | Minimum log level for both console and file loggers: `Trace`, `Debug`, `Information`, `Warning`, `Error`, `Critical` (default: `Information`). Also honored by `TensorSharp.Cli`. |
| `TENSORSHARP_LOG_DIR` | Directory the JSON-line file logger writes to (default: `<binDir>/logs`). Also honored by `TensorSharp.Cli`. |
| `TENSORSHARP_LOG_FILE` | Set to `0` to disable the file logger and keep only the console output (default: enabled). Also honored by `TensorSharp.Cli`. |

Sampling parameter precedence (highest wins):

1. Per-request JSON fields in the API call (e.g. `temperature`, `top_p`, `stop`).
2. Server-wide CLI flags (e.g. `--temperature`, `--top-p`, `--stop`).
3. `TENSORSHARP_*` environment variables listed above.
4. Built-in `SamplingConfig` defaults (`temperature=1.0`, `top_k=0`, `top_p=1.0`, `min_p=0`, `repeat_penalty=1.0`, presence/frequency penalties `0`, `seed=-1`, no stop sequences).

### Server Logging

The server emits one structured Information-level entry at the start and end of
every chat / generate turn, so a single grep over the log file reproduces the
full request-response audit trail without replaying any traffic.

| Event id | Emitted on | Carries |
|---|---|---|
| `ChatStarted` (1500) | `chat.start`, `generate.start`, plus per-protocol request banners | sampling config, message + attachment counts, `userInput=` (full latest user message), `fullInput=` (JSON-encoded array of EVERY message in the request: system prompts + all prior user/assistant turns + the new user message, with attachment counts), or the full prompt for `/api/generate` |
| `ChatCompleted` (1502) | `chat.complete`, `generate.complete` | token counts, KV cache reuse (`kvReused`, `kvReusePercent`), TTFT, elapsed, throughput, finish reason, full raw assistant output (reasoning + result) |
| `ChatAborted` (1503) | client disconnected mid-stream | partial output, KV reuse fraction at the time of abort |
| `KvCacheReusePlan` (1510) | per-prefix-reuse decision | `Debug`-level fine-grained breakdown (exact match / partial / full reset) |
| `HttpRequestStarted/Completed` (1100/1101) | every HTTP request | method, path, remote IP, status, duration; `/api/queue/status` is demoted to `Debug` so high-frequency UI polling does not drown out the per-turn entries |

The raw assistant output captures `<think>...</think>`, `<|channel|>analysis`,
and any other inline framing the model emits, so the log line for a single turn
contains both reasoning and the user-visible result. Combined with the
`fullInput=` field on `chat.start`, every turn is fully reproducible from the
log file alone (request inputs + raw model output). Long uploads or long
reasoning traces can produce multi-kilobyte log lines; raise the log level
(`TENSORSHARP_LOG_LEVEL=Warning`) to suppress them while still keeping the start
banner and error logs.

Sample `fullInput` payload (formatted for readability; it is emitted as a
single line in the actual log):

```json
[
  {"role":"system","content":"You are a helpful assistant."},
  {"role":"user","content":"What is the tallest mountain?"},
  {"role":"assistant","content":"Mount Everest."},
  {"role":"user","content":"How tall is it?","images":1}
]
```

The same per-turn KV cache reuse stats are surfaced through every API:

- **Web UI SSE** (`POST /api/chat`) - the `done` event carries `promptTokens`, `kvReusedTokens`, and `kvReusePercent`.
- **Ollama NDJSON** (`POST /api/generate`, `POST /api/chat/ollama`) - the final chunk and the non-streaming response carry `prompt_cache_hit_tokens` (int) and `prompt_cache_hit_ratio` (0..1).
- **OpenAI** (`POST /v1/chat/completions`) - the `usage` block carries `prompt_tokens_details.cached_tokens`, matching the OpenAI extension that existing SDKs already understand.

The Web UI footer line under each assistant message also surfaces the cache hit
inline (e.g. `187 tokens · 2.1s · 87.2 tok/s · KV 420/512 (82%)`).

### HTTP APIs

TensorSharp.Server exposes three API styles. See [API_EXAMPLES.md](TensorSharp.Server/API_EXAMPLES.md) for full documentation with curl and Python examples.

**Ollama-compatible API:**

```bash
# List models
curl http://localhost:5000/api/tags

# Generate text
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B-Q8_0.gguf", "prompt": "Hello!", "stream": false}'

# Chat
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "stream": false}'

# Chat with thinking mode
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B-Q8_0.gguf", "messages": [{"role": "user", "content": "Solve 17*23"}], "think": true, "stream": false}'

# Chat with tool calling
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B-Q8_0.gguf", "messages": [{"role": "user", "content": "What is the weather?"}], "tools": [{"function": {"name": "get_weather", "description": "Get current weather", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"]}}}], "stream": false}'
```

**OpenAI-compatible API:**

```bash
# Chat completions
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'

# Structured outputs (OpenAI response_format)
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-4B-Q8_0.gguf",
    "messages": [{"role": "user", "content": "Extract the city and country from: Paris, France."}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "location_extraction",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
            "confidence": {"type": ["string", "null"]}
          },
          "required": ["city", "country", "confidence"],
          "additionalProperties": false
        }
      }
    }
  }'
```

**OpenAI Python SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="Qwen3-4B-Q8_0.gguf",
    messages=[{"role": "user", "content": "What is 2+3?"}],
    max_tokens=50
)
print(response.choices[0].message.content)
```

**Queue status:**

```bash
curl http://localhost:5000/api/queue/status
# {"busy":false,"pending_requests":0,"total_processed":42}
```

## Thinking / Reasoning Mode

Models that support thinking mode (Qwen 3, Qwen 3.5, Gemma 4, GPT OSS, Nemotron-H) can produce structured chain-of-thought reasoning before generating the final answer. The thinking content is separated from the main response and can be displayed or hidden by the client.

- **Qwen 3 / Qwen 3.5 / Nemotron-H:** uses `<think>...</think>` tags
- **Gemma 4:** uses `<|channel>thought\n...<channel|>` tags
- **GPT OSS:** uses Harmony format with `<|channel|>analysis` for thinking and `<|channel|>final` for the response

Enable via `--think` (console), `"think": true` (Ollama API), or the thinking toggle in the web UI.

## Tool Calling / Function Calling

Models can invoke user-defined tools and participate in multi-turn tool-call conversations. Define tools as JSON and pass them via `--tools` (console) or the `tools` parameter in the API.

Each architecture uses its own wire format for tool calls:

- **Qwen 3 / Qwen 3.5 / Nemotron-H:** `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **Gemma 4:** `<|tool_call>call:function_name{args}<tool_call|>`

The output parser (`OutputParser.cs`) automatically extracts tool calls from the model's raw output regardless of architecture.

## Multimodal Support

### Gemma 4

Gemma 4 models support image, video, and audio inputs. Place the multimodal projector (`gemma-4-mmproj-F16.gguf`) in the same directory as the model file for automatic loading.

- **Images:** PNG, JPEG, HEIC/HEIF
- **Video:** MP4 (extracts up to 8 frames at 1 fps using OpenCV)
- **Audio:** WAV (16kHz mono), MP3, OGG Vorbis

### Gemma 3

Gemma 3 supports PNG, JPEG, and HEIC/HEIF image inputs. Place its multimodal projector (`mmproj-gemma3-4b-f16.gguf`) next to the model file for automatic loading.

### Qwen 3.5

All Qwen 3.5 variants (`qwen35`, `qwen35moe`, and `qwen3next`) load through the same `Qwen35Model` implementation. Image inputs are supported via the dynamic-resolution `Qwen35VisionEncoder`; place the projector (`Qwen3.5-mmproj-F16.gguf`) next to the model GGUF for automatic loading. The MoE variants (e.g. Qwen3.5-35B-A3B) additionally enable a fused `MoEExpertsSwiGLUResidual` GGML kernel during decode that runs all selected experts, the optional shared expert, and the residual add in a single GPU graph dispatch.

### Mistral 3

Mistral 3 supports image inputs via the Pixtral vision encoder. Place the multimodal projector (`mistral3-mmproj.gguf`) in the same directory as the model file for automatic loading.

- **Images:** PNG, JPEG, HEIC/HEIF

## Architecture

TensorSharp is structured as a layered system:

1. **TensorSharp.Core** provides the core `Tensor` type, storage abstraction, and the extensible operation registry (`Ops`). CPU implementations use `System.Numerics.Vectors` for SIMD acceleration.

2. **TensorSharp.Runtime** owns runtime-facing contracts and services: GGUF parsing, tokenization (SentencePiece / BPE), chat template rendering, configurable token sampling, output parsing, and reusable contracts such as `IModelArchitecture`, `IPromptRenderer`, `IOutputProtocolParser`, `IMultimodalInjector`, `IKVCachePolicy`, and `IBackendExecutionPlan`.

3. **TensorSharp.Models** implements `ModelBase` plus the concrete architectures and multimodal helpers (Gemma 3/4, Qwen 3/3.5, GPT OSS, Nemotron-H, Mistral 3). Models are loaded via `ModelBase.Create()` which auto-detects the architecture from GGUF metadata.

4. **TensorSharp.Backends.GGML** registers accelerated implementations of the same operations via a native C++ bridge (`libGgmlOps`) that links against [ggml](https://github.com/ggml-org/ggml). On macOS this provides Metal GPU compute, and on Linux it can expose GGML CUDA for NVIDIA GPUs. Operations include native quantized matmul (Q4_K_M, Q8_0, etc.) without dequantizing to FP32.

5. **TensorSharp.Server** is the HTTP/application layer. It provides Ollama-compatible and OpenAI-compatible REST APIs, the browser-based chat UI, upload handling, and the FIFO inference queue.

6. **TensorSharp.Cli** is the console/application layer for local prompts, multimodal experiments, prompt inspection, and JSONL batch workflows.

### Performance Optimizations

- **Fused GPU decode** (Gemma 4): all transformer layers are executed in a single GGML compute graph dispatch on Metal, reducing CPU-GPU round-trips from hundreds per token to one. This achieves ~2.6x speedup over per-operation dispatch.
- **Fused GPU prefill** (Gemma 4): for dense (non-MoE, non-shared, non-PLE/multimodal) layers, `Gemma4LayerPrefill` runs the entire transformer block (RMSNorm + QKV + QK-norm + RoPE + attention + output projection + post-attn norm + GeGLU FFN + post-FFN norm + residual + layer scalar) as a single GGML graph dispatch per layer during prefill, extending the fused approach from decode to multi-token prefill.
- **Chunked prefill** (Gemma 4): long prompts are split into bounded chunks (2x sliding window, max 2048 tokens) to avoid O(n^2) attention score tensors for SWA layers. Chunking is applied automatically when text-only (no multimodal embeddings) and keeps each chunk within the SWA window budget.
- **Native whole-model decode** (Qwen 3): all transformer layers run in one native call (`TransformerModelDecode`) with pre-resolved per-layer weight pointers cached at load time, removing managed-loop overhead from the decode hot path.
- **Fused Qwen 3.5 attention layer decode**: a single GGML graph performs RMSNorm + fused QKV + Q/gate deinterleave + per-head QK norm + RoPE + KV cache append + flash attention + sigmoid-gated mix + output projection + residual add for each FullAttention layer. Replaces ~2 standalone GGML calls and ~6 small CPU/GPU sync points per attention layer. Engages once the cached sequence length exceeds 4096 tokens (override with `FUSED_ATTN_LAYER_MIN_SEQ_LEN=N`).
- **Fused prefill attention** (Qwen 3.5): `FusedPrefillAttention` combines Q*K^T, causal mask, softmax, and *V into a single GGML graph dispatch during multi-token prefill, eliminating ~5 separate C#-to-GGML round-trips per attention layer. Handles both initial prefill and continuation with existing KV cache entries.
- **Fused output-projection + FFN** (Qwen 3.5): for both FullAttention and GatedDeltaNet layers with dense FFN, `FusedOutProjFFN` merges the output projection, residual add, post-attention RMSNorm, and the full SwiGLU FFN (gate_up matmul + SiLU + down matmul + residual) into a single GGML graph dispatch, reducing two GPU round-trips to one per layer.
- **Fused output-projection + norm + router** (Qwen 3.5 MoE): `FusedOutProjNormRouter` merges the GatedDeltaNet output projection, residual add, post-attention RMSNorm, and MoE router projection into one dispatch. The pre-computed router logits are then consumed directly by the batched MoE kernel, eliminating a separate router dispatch per MoE layer.
- **Fused vision encoder** (Qwen 3.5): `FusedVisionAttention` merges LayerNorm + QKV + bias + 2D RoPE + scaled dot-product attention + output projection + bias + residual into one GGML graph dispatch (~8 ops → 1). `FusedVisionMLP` merges LayerNorm + up + bias + GELU + down + bias + residual into one dispatch (7 ops → 1). Combined, these cut the per-block GPU round-trips from ~15 to 2.
- **Fused weight projections**: Q/K/V projections are fused into a single QKV matmul; gate and up projections are fused into a single gate_up matmul.
- **Native quantized compute**: quantized weights (Q4_K_M, Q6_K, Q8_0, IQ2_XXS, MXFP4, etc.) are used directly in matmul without expanding to FP32, saving memory and bandwidth. A batched `AddmmQuantBatch` kernel handles multiple sub-weight matmuls against a single quantized blob in one dispatch.
- **Batched GPU MoE**: `MoEExpertsSwiGLUResidual` (Qwen 3.5) and `MoEExpertsForward` (Nemotron-H) collapse all selected experts -- and, for Qwen 3.5, the optional shared expert and the residual add -- into a single GGML graph dispatch per MoE layer.
- **GEMM-based vision patch embedding** (Qwen 3.5): the patch embedding step is reformulated as parallel im2col + matrix multiplication, replacing a single-threaded scalar quintuple-nested loop with a GPU-accelerated matmul.
- **Parallelized Q/gate deinterleave** (Qwen 3.5): the Q + sigmoid-gate deinterleave in FullAttention prefill is parallelized across tokens, scaling linearly with CPU core count for long prompts.
- **Optimized pure C# CPU path**: managed GEMM fast paths and contiguous float32 kernels accelerate decode, softmax, RMSNorm, RoPE, fused activations, and other hot paths while keeping quantized GGUF weights compressed during CPU loading.
- **Circular KV cache**: sliding-window attention layers use a fixed-size circular buffer, bounding memory usage regardless of sequence length.
- **KV-cache prefix reuse**: multi-turn conversations reuse the longest matching token prefix across turns. Truncation is automatically backed off by the sliding-window size for SWA models so the suffix can rebuild the SWA context.
- **Kernel warmup**: both CLI and Server run a tiny forward pass at startup to pre-compile GPU kernels (Metal pipeline states, CUDA JIT) and warm the memory pool, avoiding cold-start latency on the first real inference request.
- **Prefill caching** (Gemma 4, Qwen 3.5): per-forward-pass SWA mask cache (Gemma 4), NeoX RoPE cos/sin lookup table cache across global layers (Gemma 4), and RoPE position tensor cache across layers (Gemma 4, Qwen 3.5) eliminate redundant recomputation during prefill.
- **In-place QK RMSNorm** (Qwen 3.5): per-head QK normalization is performed in-place using a `View`, avoiding one tensor allocation and copy per Q/K per layer.

### Memory Optimizations

- **Zero-copy file-mapped quantized weights** (CUDA, Metal, GGML CPU): the GGUF model file is memory-mapped and quantized tensors are bound directly into native ops via host-pointer buffers. This removes the per-tensor copy from disk into a freshly-allocated native heap buffer that previously roughly doubled the resident set on Apple Silicon for large quantized models. For example, `Qwen3.5-35B-A3B-IQ2_XXS` (~10 GB GGUF) now runs with ~7 GB peak working memory under Metal instead of ~17 GB. The OS keeps the mapped file in its page cache and pages it out under memory pressure without any inference penalty on Apple Silicon (unified memory).
- **Best-fit memory pool**: the GGML host allocator uses a best-fit search across pooled blocks instead of first-fit, which avoids handing out a large scratch block to satisfy a tiny intermediate-tensor request and keeps the working-set tightly bounded across long-running inference.
- **Bounded pool retention**: the integrated-GPU / CPU memory pool now caps individual retained blocks at 64 MB and the total pool at 32 blocks. Combined with mmap-backed weights, this keeps short-lived intermediate tensors recycled fast while bounding the peak resident set.
- **Memory-efficient model loading**: large tensors are streamed directly to native memory without intermediate managed allocations. F32 weights and norms still load on demand; quantized weights are mmap-backed when supported by the backend.

## Benchmarks

Reference numbers measured on `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` (~10 GB on disk, 256 routed experts of which 8 are active per token, with 12 full attention + 30 GatedDeltaNet recurrent layers) on an Apple M4 Pro with 24 GB unified memory:

| Metric | Before (`v1` baseline) | After (this branch) | Change |
|---|---|---|---|
| Process peak memory footprint | ~17 GB | **~8 GB** | **-52%** |
| TensorSharp.Server resident set after load | ~20 GB | **~8 GB** | **-60%** |
| Decode throughput (warm, 256 prefill / 64 decode, M4 Pro) | ~3.8 tok/s | **~10.8 tok/s** | **+2.85x** |
| Decode latency (warm, 256 prefill / 64 decode, M4 Pro) | ~264 ms/token | **~92 ms/token** | **-65%** |

Reproduce with:

```bash
./TensorSharp.Cli --model Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf --backend ggml_metal \
    --benchmark --bench-prefill 256 --bench-decode 64 --bench-runs 3
```

The memory reduction comes primarily from no longer copying the GGUF file into a separate native heap buffer (the file is now mmap-bound zero-copy into Metal command buffers). The decode throughput increase is largely a side effect of removing that ~10 GB duplicate working set, which was previously triggering OS-level memory pressure on machines with 24 GB or less of physical RAM.

## Testing

### Unit tests (xUnit)

`InferenceWeb.Tests` exercises in-process behavior that doesn't require a running server: managed quantized ops, KV cache policies, KV-cache prompt rendering / multi-turn integration, chat-session and session-manager isolation, model service history and KV cache plumbing, request-logging middleware and file-logger provider, image preprocessing, media helpers, structured-output validation, text-upload helpers, model-service upload logging, web UI chat policy, model context length parsing, and backend catalog resolution.

```bash
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj
```

### Server integration tests

Integration tests for TensorSharp.Server are in `TensorSharp.Server/testdata/`. They cover all three API styles (Web UI SSE, Ollama, OpenAI), multi-turn conversations, thinking mode, tool calling, structured outputs, queue behavior, concurrent requests, and abort support. Architecture-specific features (thinking, tool calling) are auto-detected and skipped when the active model does not support them.

```bash
# Start TensorSharp.Server, then run:
python3 TensorSharp.Server/testdata/test_multiturn.py
# or
bash TensorSharp.Server/testdata/test_multiturn.sh
```

See [TensorSharp.Server/testdata/README.md](TensorSharp.Server/testdata/README.md) for the full test matrix.

## Author

Zhongkai Fu

## License

See [LICENSE](LICENSE) for details.
