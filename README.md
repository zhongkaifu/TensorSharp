# TensorSharp

A C# inference engine for running large language models (LLMs) locally using GGUF model files. TensorSharp provides both a console application and a web-based chatbot interface for multi-turn conversations with multimodal models.

## Supported Model Architectures

| Architecture | Models | Multimodal |
|---|---|---|
| Gemma 4 | gemma-4-E4B, gemma-4-31B | Image, Video, Audio |
| Gemma 3 | gemma-3-4b, etc. | Image |
| Qwen 3 | Qwen3-4B, etc. | Text only |
| Qwen 3.5 | Qwen3.5-9B, etc. | Image |

## Compute Backends

| Backend | Flag | Description |
|---|---|---|
| GGML Metal | `--backend ggml_metal` | GPU-accelerated via Apple Metal (macOS). Recommended for Apple Silicon. |
| GGML CPU | `--backend ggml_cpu` | CPU inference using the native GGML library with optimized kernels. |
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
│   │   ├── Gemma4/              # Includes vision encoder, audio encoder
│   │   ├── Qwen3/
│   │   └── Qwen35/
│   ├── GgufReader.cs            # GGUF file parser
│   ├── ModelBase.cs             # Base class for all model architectures
│   ├── ChatTemplate.cs          # Chat template rendering
│   └── MediaHelper.cs           # Video frame extraction (OpenCV)
├── InferenceConsole/            # CLI application
├── InferenceWeb/                # Web chatbot application (ASP.NET Core)
└── ExternalProjects/            # Third-party dependencies (ggml)
```

## Prerequisites

- [.NET 8.0 SDK](https://dotnet.microsoft.com/download/dotnet/8.0) or later
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
./InferenceConsole --model <model.gguf> --input input.txt --output output.txt \
    --max-tokens 200 --backend ggml_metal

# Image inference (Gemma 3/4, Qwen 3.5)
./InferenceConsole --model <model.gguf> --image photo.png --backend ggml_metal

# Video inference (Gemma 4)
./InferenceConsole --model <model.gguf> --video clip.mp4 --backend ggml_metal

# Audio inference (Gemma 4)
./InferenceConsole --model <model.gguf> --audio speech.wav --backend ggml_metal
```

**Command-line options:**

| Option | Description |
|---|---|
| `--model <path>` | Path to a GGUF model file (required) |
| `--input <path>` | Text file containing the user prompt |
| `--output <path>` | Write generated text to this file |
| `--image <path>` | Image file for vision inference |
| `--video <path>` | Video file for video inference |
| `--audio <path>` | Audio file (WAV, MP3, OGG) for audio inference |
| `--mmproj <path>` | Path to the multimodal projector GGUF file |
| `--max-tokens <N>` | Maximum number of tokens to generate (default: 100) |
| `--backend <type>` | Compute backend: `cpu`, `ggml_cpu`, or `ggml_metal` |
| `--test` | Run built-in test suite |

The multimodal projector file is auto-detected if placed alongside the model file with a recognized name (e.g., `gemma-4-mmproj-F16.gguf`).

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

2. **TensorSharp.GGML** registers GPU-accelerated implementations of the same operations via a native C++ bridge (`libGgmlOps`) that links against [ggml](https://github.com/ggml-org/ggml). On macOS, this provides Metal GPU compute.

3. **InferenceEngine** implements model-specific logic: GGUF parsing, tokenization (SentencePiece BPE), chat template rendering, and the forward pass for each architecture. Models are loaded via `ModelBase.Create()` which auto-detects the architecture from GGUF metadata.

4. **InferenceConsole** and **InferenceWeb** are thin application layers that handle I/O and user interaction.

## Author

Zhongkai Fu

## License

See [LICENSE](LICENSE) for details.
