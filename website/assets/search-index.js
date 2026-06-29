/* Search index for the TensorSharp wiki. Each entry:
   t = title, p = page label, u = url(+anchor), s = snippet, k = extra keywords. */
window.SEARCH_INDEX = [
  { t: "What is TensorSharp", p: "Home", u: "index.html", s: "Native .NET LLM inference engine for GGUF models — CLI, web server, and Ollama/OpenAI-compatible APIs.", k: "intro introduction about overview llm gguf csharp dotnet" },
  { t: "Quick start in 30 seconds", p: "Home", u: "index.html#quickstart", s: "Clone, build, download a model, and stream your first reply.", k: "begin start tutorial first run hello" },
  { t: "Who is TensorSharp for", p: "Home", u: "index.html#audience", s: "Value for developers, managers, CTOs, students, and businesses adopting local LLM inference.", k: "audience business value executive sales marketing students" },

  { t: "Overview & Architecture", p: "Overview", u: "overview.html#architecture", s: "Layered system: Core tensors, Runtime, Models, Backends, Server, CLI.", k: "design layers stack how it works" },
  { t: "Project structure", p: "Overview", u: "overview.html#structure", s: "Repository layout and the role of each project/package.", k: "folders directories modules" },
  { t: "Current status & capabilities", p: "Overview", u: "overview.html#status", s: "Model families, inference hosts, backends, multimodal, batching, and observability.", k: "support matrix maturity" },
  { t: "Business value", p: "Overview", u: "overview.html#business", s: "Why organizations run TensorSharp: privacy, cost, control, on-prem, no per-token fees.", k: "cost roi privacy on-prem manager cto ceo" },

  { t: "Feature catalog", p: "Features", u: "features.html#catalog", s: "Multi-architecture, multimodal, thinking, tool calling, quantization, continuous batching, speculative decoding.", k: "capabilities list everything" },

  { t: "Getting Started", p: "Get started", u: "getting-started.html", s: "Prerequisites, build, download a model, and run the CLI or server.", k: "install setup begin" },
  { t: "Prerequisites", p: "Get started", u: "getting-started.html#prerequisites", s: ".NET 10 SDK, git, CMake, and optional CUDA / Metal toolchains.", k: "requirements dotnet 10 cmake cuda xcode" },
  { t: "Build the solution", p: "Get started", u: "getting-started.html#build", s: "dotnet build TensorSharp.slnx -c Release. Native GGML library compiles on first build.", k: "compile make native ggml" },
  { t: "Download a GGUF model", p: "Get started", u: "getting-started.html#download", s: "Get a model from Hugging Face, e.g. gemma-4-E4B-it Q8_0.", k: "huggingface weights gguf quantization" },
  { t: "First run", p: "Get started", u: "getting-started.html#first-run", s: "One-shot generation, interactive chat, or start the server.", k: "hello world example run" },

  { t: "Pick a backend", p: "Backends", u: "backends.html#table", s: "Choose ggml_metal (Mac), ggml_cuda (NVIDIA), or cpu/ggml_cpu (portable).", k: "choose hardware gpu which backend" },
  { t: "GGML CUDA backend", p: "Backends", u: "backends.html#ggml", s: "--backend ggml_cuda — most-tested NVIDIA path on Windows/Linux.", k: "nvidia gpu cuda windows linux" },
  { t: "GGML Metal backend", p: "Backends", u: "backends.html#ggml", s: "--backend ggml_metal — default on macOS / Apple Silicon.", k: "apple silicon mac metal" },
  { t: "MLX backend", p: "Backends", u: "backends.html#mlx", s: "--backend mlx — Apple Silicon GPU path built on mlx-c.", k: "apple metal mlx-c" },
  { t: "Direct CUDA backend", p: "Backends", u: "backends.html#cuda", s: "--backend cuda — direct CUDA Driver API + cuBLAS + PTX kernels (pure C#).", k: "nvidia ptx cublas experimental" },
  { t: "Pure C# CPU backend", p: "Backends", u: "backends.html#cpu", s: "--backend cpu — portable, no native dependencies; ggml_cpu for native CPU kernels.", k: "portable debugging no gpu" },
  { t: "Build the native GGML / MLX libraries", p: "Backends", u: "backends.html#native-build", s: "build-windows.ps1 / build-linux.sh / build-macos.sh and CUDA arch detection.", k: "compile native cmake cuda arch" },

  { t: "Supported models", p: "Models", u: "models.html#table", s: "Gemma 3/4, Qwen 3 / 3.5 / 3.6, GPT OSS, Nemotron-H, Mistral 3, DiffusionGemma.", k: "architectures families gemma qwen gptoss nemotron mistral" },
  { t: "Model downloads (GGUF)", p: "Models", u: "models.html#downloads", s: "Hugging Face links for every supported architecture.", k: "huggingface weights download" },
  { t: "Multimodal: image, video, audio", p: "Models", u: "models.html#multimodal", s: "Image/video/audio for Gemma 4; image for Gemma 3, Qwen 3.5-family, Mistral 3, Nemotron-H Omni.", k: "vision image audio video mmproj projector" },
  { t: "Thinking / reasoning mode", p: "Models", u: "models.html#thinking", s: "Structured chain-of-thought with think tags for Qwen, Gemma 4, GPT OSS, Nemotron-H.", k: "reasoning chain of thought think cot" },
  { t: "Tool calling / function calling", p: "Models", u: "models.html#toolcalling", s: "Models invoke user-defined tools; multi-turn across all three API styles.", k: "functions tools agent" },

  { t: "CLI examples", p: "CLI", u: "cli.html#examples", s: "Text, image, video, audio, thinking, tools, batch JSONL, benchmarks.", k: "command line console examples" },
  { t: "CLI options reference", p: "CLI", u: "cli.html#options", s: "--model, --input, --backend, --max-tokens, sampling flags, and more.", k: "flags arguments parameters" },
  { t: "Interactive REPL commands", p: "CLI", u: "cli.html#repl", s: "Slash commands: /reset, /system, /think, /model, /backend, /image, /temp.", k: "chat repl interactive slash commands" },
  { t: "Batch processing (JSONL)", p: "CLI", u: "cli.html#jsonl", s: "Run many prompts from a JSONL file with --input-jsonl.", k: "batch jsonl multi-turn" },
  { t: "DiffusionGemma generation", p: "CLI", u: "cli.html#diffusion", s: "--diffusion-steps, --diffusion-seed, --diffusion-blocks text diffusion.", k: "diffusion denoising entropybound" },

  { t: "Start the server", p: "Server", u: "server.html#start", s: "./TensorSharp.Server --model model.gguf --backend ggml_metal — serves http://localhost:5000.", k: "web server host run port" },
  { t: "Server options", p: "Server", u: "server.html#options", s: "--model, --mmproj, --backend, --max-tokens, default sampling, batching flags.", k: "flags arguments parameters" },
  { t: "Web UI features", p: "Server", u: "server.html#webui", s: "Browser chat: multi-turn, uploads, thinking toggle, tool calling, streaming, message editing.", k: "browser chatbot ui frontend" },
  { t: "Server environment variables", p: "Server", u: "server.html#env", s: "BACKEND, MAX_TOKENS, PORT, scheduler and MLX tunables.", k: "env vars configuration TS_SCHED" },
  { t: "Continuous-batching tunables", p: "Server", u: "server.html#tunables", s: "TS_SCHED_* knobs for paged KV blocks, running sequences, prefill chunk.", k: "scheduler paged kv tuning" },

  { t: "Ollama-compatible API", p: "HTTP API", u: "http-api.html#ollama", s: "/api/generate, /api/chat/ollama, /api/tags, /api/show with curl examples.", k: "ollama rest endpoint curl generate chat" },
  { t: "OpenAI-compatible API", p: "HTTP API", u: "http-api.html#openai", s: "/v1/chat/completions and /v1/models — drop-in for OpenAI clients.", k: "openai chat completions v1 compatible" },
  { t: "Web UI SSE protocol", p: "HTTP API", u: "http-api.html#webui-sse", s: "/api/chat server-sent events, sessions, KV reuse fields.", k: "sse streaming events sessions" },
  { t: "Structured outputs (JSON schema)", p: "HTTP API", u: "http-api.html#structured", s: "response_format text / json_object / json_schema with strict validation.", k: "json mode structured schema response_format" },
  { t: "Tool calling over HTTP", p: "HTTP API", u: "http-api.html#tools", s: "Send a tools array; parse structured tool_calls from the response.", k: "functions tools api" },
  { t: "Sampling parameters", p: "HTTP API", u: "http-api.html#sampling", s: "temperature, top_k, top_p, min_p, penalties, seed, stop, num_predict/max_tokens.", k: "options temperature top_p seed stop" },
  { t: "Python client examples", p: "HTTP API", u: "http-api.html#python", s: "requests and the openai SDK, streaming and structured outputs.", k: "python requests openai sdk client" },

  { t: "Use TensorSharp from C#", p: "C# Library", u: "code-api.html#quickstart", s: "ModelBase.Create, Tokenizer.Encode, Forward, Sample — a decode loop in code.", k: "library code embed nuget dotnet api" },
  { t: "NuGet packages", p: "C# Library", u: "code-api.html#packages", s: "TensorSharp.Core, .Runtime, .Models, .Backends.*, .Server, .Cli.", k: "nuget package dependency namespace" },
  { t: "SamplingConfig", p: "C# Library", u: "code-api.html#sampling", s: "Temperature, TopK, TopP, MinP, penalties, Seed, StopSequences, MaxTokens.", k: "sampling config class properties" },
  { t: "Public namespaces", p: "C# Library", u: "code-api.html#namespaces", s: "TensorSharp, TensorSharp.Runtime, TensorSharp.Models and backend namespaces.", k: "namespace types api" },

  { t: "Continuous batching & paged KV cache", p: "Advanced", u: "advanced.html#continuous-batching", s: "vLLM-style paged KV pool, block-hash prefix sharing, iteration-level scheduler.", k: "vllm paged attention scheduler batching" },
  { t: "Paged attention", p: "Advanced", u: "advanced.html#paged-kv", s: "Native TSGgml_PagedAttentionForward drives ggml_flash_attn_ext on Metal/CUDA.", k: "flash attention kv blocks" },
  { t: "MTP / NextN speculative decoding", p: "Advanced", u: "advanced.html#mtp", s: "Draft heads propose tokens; trunk verifies in one batched forward. --mtp-spec.", k: "speculative decode draft eagle nextn qwen gemma" },
  { t: "Performance optimizations", p: "Advanced", u: "advanced.html#perf", s: "Fused GPU decode/prefill, native quantized compute, batched MoE, KV prefix reuse.", k: "fused kernels speed optimization gpu" },
  { t: "Memory optimizations", p: "Advanced", u: "advanced.html#memory", s: "Zero-copy mmap weights, best-fit pool, SSD KV spillover, KV codecs.", k: "memory mmap footprint ram" },
  { t: "DiffusionGemma text diffusion", p: "Advanced", u: "advanced.html#diffusion", s: "Block-wise EntropyBound denoising over a Gemma-4 MoE backbone.", k: "diffusion denoising text generation" },

  { t: "Head-to-head vs llama.cpp", p: "Benchmarks", u: "benchmarks.html#head-to-head", s: "Pure-.NET TensorSharp out-prefills llama.cpp on every model (geomean 1.18×–1.88×) with decode at parity on the same GGUF + GPU: 26B-A4B MoE leads at 1.88× prefill / 1.69× TTFT; structured-output prefill 5.89×; multi-turn up to 1.93×.", k: "llama.cpp comparison faster speedup geomean moe prefill ttft multi-turn json structured output decode parity vs versus" },
  { t: "Benchmarks", p: "Benchmarks", u: "benchmarks.html#head-to-head", s: "Head-to-head vs llama.cpp on the same GGUF files and hardware.", k: "performance numbers throughput tokens per second" },
  { t: "Testing", p: "Benchmarks", u: "benchmarks.html#testing", s: "xUnit unit tests and server integration tests.", k: "tests xunit integration ci" },

  { t: "CLI flags reference", p: "API Reference", u: "api-reference.html#cli-flags", s: "Complete table of TensorSharp.Cli command-line options.", k: "reference flags cli options table" },
  { t: "Server flags reference", p: "API Reference", u: "api-reference.html#server-flags", s: "Complete table of TensorSharp.Server command-line options.", k: "reference flags server options table" },
  { t: "Environment variables", p: "API Reference", u: "api-reference.html#env-vars", s: "Runtime, scheduler, MTP, MLX, and diffusion environment variables.", k: "env vars TS_ reference" },
  { t: "HTTP endpoints", p: "API Reference", u: "api-reference.html#endpoints", s: "Every Ollama, OpenAI, Web UI, and utility endpoint in one table.", k: "rest endpoints routes api reference" },
  { t: "Sampling parameters reference", p: "API Reference", u: "api-reference.html#sampling-params", s: "Ollama-style and OpenAI-style sampling fields and defaults.", k: "sampling reference temperature defaults" },
  { t: "C# public API", p: "API Reference", u: "api-reference.html#csharp", s: "ModelBase, SamplingConfig, ITokenizer, BackendType and key interfaces.", k: "csharp classes interfaces api reference" },
  { t: "REPL command reference", p: "API Reference", u: "api-reference.html#repl-commands", s: "All interactive slash commands grouped by category.", k: "repl slash commands reference" },

  { t: "Glossary", p: "Glossary", u: "glossary.html#terms", s: "GGUF, quantization, tokens, KV cache, MoE, prefill, decode, TTFT explained.", k: "definitions terms beginner concepts" },
  { t: "FAQ", p: "Glossary", u: "glossary.html#faq", s: "Common questions: which model, which backend, GPU needed, privacy, licensing.", k: "questions help troubleshooting" },
  { t: "What is GGUF?", p: "Glossary", u: "glossary.html#terms", s: "The single-file model format TensorSharp loads, with quantized weights and metadata.", k: "gguf format quantization" },
  { t: "What is quantization?", p: "Glossary", u: "glossary.html#terms", s: "Compressing weights to fewer bits (Q4_K_M, Q8_0) to fit memory and run faster.", k: "quantization q4 q8 bits" },
  { t: "What is a KV cache?", p: "Glossary", u: "glossary.html#terms", s: "Stored attention keys/values that make multi-turn and long contexts efficient.", k: "kv cache attention reuse" }
];
