# TensorSharp

[English](README.md) | [中文](readme_cn.md)

A C# inference engine for running large language models (LLMs) locally using GGUF model files. TensorSharp provides a console application, a web-based chatbot interface, and Ollama/OpenAI-compatible HTTP APIs for programmatic access.

## Features

- **Multi-architecture support** -- Gemma 4, Gemma 3, Qwen 3, Qwen 3.5
- **Multimodal inference** -- image, video, and audio inputs (Gemma 4); images for Gemma 3 / Qwen 3.5
- **Quantized model support** -- loads GGUF files with Q4_K_M, Q8_0, F16, and other quantization formats; performs native quantized matmul without dequantizing to FP32
- **GPU-accelerated** -- Apple Metal backend via GGML with fused whole-model GPU dispatch for Gemma 4 decode (~2.6x speedup over per-op dispatch)
- **Ollama & OpenAI API compatibility** -- drop-in replacement endpoints for existing tooling
- **Configurable sampling** -- temperature, top-k, top-p, min-p, repetition/presence/frequency penalties, seed, stop sequences
- **Chat templates** -- auto-loaded from GGUF metadata (Jinja2), with hardcoded fallbacks per architecture
- **Batch processing** -- JSONL input support in the console application
- **Streaming** -- token-by-token output via SSE (web) or stdout (console)
- **Mixture of Experts** -- Gemma 4 MoE variants (e.g. gemma-4-26B-A4B)

## Supported Model Architectures

| Architecture | Example Models | Multimodal |
|---|---|---|
| Gemma 4 | gemma-4-E4B, gemma-4-31B, gemma-4-26B-A4B (MoE) | Image, Video, Audio |
| Gemma 3 | gemma-3-4b | Image |
| Qwen 3 | Qwen3-4B | Text only |
| Qwen 3.5 | Qwen3.5-9B | Image |

## Compute Backends

| Backend | Flag | Description |
|---|---|---|
| GGML Metal | `--backend ggml_metal` | GPU-accelerated via Apple Metal (macOS). Recommended for Apple Silicon. |
| GGML CPU | `--backend ggml_cpu` | CPU inference using native GGML with optimized kernels. |
| Pure C# CPU | `--backend cpu` | Portable CPU inference with no native dependencies. |

## Project Structure

```
TensorSharp/
├── TensorSharp/                 # Core tensor library (CPU operations, SIMD)
├── TensorSharp.GGML/            # GGML backend bindings (Metal/CPU via native library)
├── TensorSharp.GGML.Native/     # Native C++ bridge to ggml (builds libGgmlOps)
├── AdvUtils/                    # Utility library
├── InferenceEngine/             # Model loading, tokenization, and inference logic
│   ├── Models/
│   │   ├── Gemma3/
│   │   ├── Gemma4/              # Vision encoder, audio encoder, MoE, fused GPU decode
│   │   ├── Qwen3/
│   │   └── Qwen35/
│   ├── GgufReader.cs            # GGUF file parser
│   ├── ModelBase.cs             # Base class for all model architectures
│   ├── ChatTemplate.cs          # Chat template rendering (hardcoded + Jinja2 from GGUF)
│   ├── Jinja2Template.cs        # Jinja2 template renderer
│   ├── SamplingConfig.cs        # Sampling parameter configuration
│   ├── TokenSampler.cs          # Token sampling (greedy, top-k, top-p, min-p, penalties)
│   └── MediaHelper.cs           # Video frame extraction, audio decoding
├── InferenceConsole/            # CLI application
├── InferenceWeb/                # Web chatbot + API server (ASP.NET Core)
│   ├── wwwroot/index.html       # Chat UI
│   └── API_EXAMPLES.md          # Detailed API documentation
└── ExternalProjects/            # Third-party dependencies (ggml)
```

## Prerequisites

- [.NET 8.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) or later (.NET 10 for InferenceWeb)
- **macOS (Metal backend):** CMake 3.20+ and Xcode command-line tools for building the native GGML library
- GGUF model files (e.g., from [Hugging Face](https://huggingface.co))

## Building

### Build the entire solution

```bash
dotnet build TensorSharp.slnx
```

### Build individual applications

```bash
# Console application
dotnet build InferenceConsole/InferenceConsole.csproj

# Web application
dotnet build InferenceWeb/InferenceWeb.csproj
```

### Build the native GGML library (macOS)

The native library is built automatically during the first `dotnet build` if it doesn't exist. To build it manually:

```bash
cd TensorSharp.GGML.Native
bash build-macos.sh
```

This compiles `libGgmlOps.dylib` with Metal GPU support. The build output is automatically copied to the application's output directory.

## Usage

### Console Application

```bash
cd InferenceConsole/bin

# Text inference
./InferenceConsole --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_metal

# Image inference (Gemma 3/4, Qwen 3.5)
./InferenceConsole --model <model.gguf> --image photo.png --backend ggml_metal

# Video inference (Gemma 4)
./InferenceConsole --model <model.gguf> --video clip.mp4 --backend ggml_metal

# Audio inference (Gemma 4)
./InferenceConsole --model <model.gguf> --audio speech.wav --backend ggml_metal

# With sampling parameters
./InferenceConsole --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.2 --seed 42

# Batch processing (JSONL)
./InferenceConsole --model <model.gguf> --input-jsonl requests.jsonl \
    --output results.txt --backend ggml_metal
```

**Command-line options:**

| Option | Description |
|---|---|
| `--model <path>` | Path to a GGUF model file (required) |
| `--input <path>` | Text file containing the user prompt |
| `--input-jsonl <path>` | JSONL file with batch requests (one JSON per line) |
| `--output <path>` | Write generated text to this file |
| `--image <path>` | Image file for vision inference |
| `--video <path>` | Video file for video inference |
| `--audio <path>` | Audio file (WAV, MP3, OGG) for audio inference |
| `--mmproj <path>` | Path to the multimodal projector GGUF file |
| `--max-tokens <N>` | Maximum tokens to generate (default: 100) |
| `--backend <type>` | Compute backend: `cpu`, `ggml_cpu`, or `ggml_metal` |
| `--temperature <f>` | Sampling temperature (0 = greedy) |
| `--top-k <N>` | Top-K filtering (0 = disabled) |
| `--top-p <f>` | Nucleus sampling threshold (1.0 = disabled) |
| `--min-p <f>` | Minimum probability filtering (0 = disabled) |
| `--repeat-penalty <f>` | Repetition penalty (1.0 = none) |
| `--presence-penalty <f>` | Presence penalty (0 = disabled) |
| `--frequency-penalty <f>` | Frequency penalty (0 = disabled) |
| `--seed <N>` | Random seed (-1 = non-deterministic) |
| `--stop <string>` | Stop sequence (can be repeated) |
| `--test` | Run built-in test suite |

The multimodal projector file is auto-detected if placed alongside the model file with a recognized name (e.g., `gemma-4-mmproj-F16.gguf`).

**JSONL input format:**

Each line is a JSON object with `messages`, optional `prompt`, and optional sampling parameters:

```json
{"id": "q1", "messages": [{"role": "user", "content": "What is 2+3?"}], "max_tokens": 50}
{"id": "q2", "messages": [{"role": "user", "content": "Write a haiku."}], "max_tokens": 100, "temperature": 0.8}
```

### Web Application

```bash
cd InferenceWeb/bin

# Set environment variables and run
MODEL_DIR=./models BACKEND=ggml_metal ./InferenceWeb
```

Open `http://localhost:5000` in your browser. The web interface supports:

- Multi-turn chat conversations
- Model selection from available GGUF files in `MODEL_DIR`
- Image, video, and audio uploads for multimodal inference
- Streaming token generation via Server-Sent Events

**Environment variables:**

| Variable | Description |
|---|---|
| `MODEL_DIR` | Directory containing GGUF model files |
| `BACKEND` | Compute backend (default: `ggml_metal`) |
| `PORT` | HTTP port (default: `5000`) |

### HTTP APIs

InferenceWeb exposes three API styles. See [API_EXAMPLES.md](InferenceWeb/API_EXAMPLES.md) for full documentation with curl and Python examples.

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
```

**OpenAI-compatible API:**

```bash
# Chat completions
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-4B-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
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

## Multimodal Support

### Gemma 4

Gemma 4 models support image, video, and audio inputs. Place the multimodal projector (`gemma-4-mmproj-F16.gguf`) in the same directory as the model file for automatic loading.

- **Images:** PNG, JPEG
- **Video:** MP4 (extracts up to 8 frames at 1 fps using OpenCV)
- **Audio:** WAV (16kHz mono), MP3, OGG Vorbis

### Gemma 3 / Qwen 3.5

These models support image inputs with their respective multimodal projector files.

## Architecture

TensorSharp is structured as a layered system:

1. **TensorSharp** provides the core `Tensor` type, storage abstraction, and an extensible operation registry (`Ops`). CPU implementations use `System.Numerics.Vectors` for SIMD acceleration.

2. **TensorSharp.GGML** registers GPU-accelerated implementations of the same operations via a native C++ bridge (`libGgmlOps`) that links against [ggml](https://github.com/ggml-org/ggml). On macOS, this provides Metal GPU compute. Operations include native quantized matmul (Q4_K_M, Q8_0, etc.) without dequantizing to FP32.

3. **InferenceEngine** implements model-specific logic: GGUF parsing, tokenization (SentencePiece BPE), chat template rendering (Jinja2 from GGUF metadata with hardcoded fallbacks), configurable token sampling, and the forward pass for each architecture. Models are loaded via `ModelBase.Create()` which auto-detects the architecture from GGUF metadata.

4. **InferenceConsole** and **InferenceWeb** are application layers that handle I/O and user interaction. InferenceWeb provides Ollama-compatible and OpenAI-compatible REST APIs alongside a browser-based chat UI.

### Performance Optimizations

- **Fused GPU decode** (Gemma 4): all transformer layers are executed in a single GGML compute graph dispatch on Metal, reducing CPU-GPU round-trips from hundreds per token to one. This achieves ~2.6x speedup over per-operation dispatch.
- **Fused weight projections**: Q/K/V projections are fused into a single QKV matmul; gate and up projections are fused into a single gate_up matmul.
- **Native quantized compute**: quantized weights (Q4_K_M, Q6_K, Q8_0, etc.) are used directly in matmul without expanding to FP32, saving memory and bandwidth.
- **Circular KV cache**: sliding-window attention layers use a fixed-size circular buffer, bounding memory usage regardless of sequence length.
- **Memory-efficient model loading**: large tensors are streamed directly to native memory without intermediate managed allocations.

## Author

Zhongkai Fu

## License

See [LICENSE](LICENSE) for details.
