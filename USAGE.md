# Usage
[English](USAGE.md) | [中文](USAGE_zh-cn.md)

> Part of the [TensorSharp](README.md) documentation. Quick-start commands are in the [README](README.md#quick-start); configuration files are in [config/README.md](config/README.md).

## Compute Backends

| Backend | Flag | Best fit | Description |
|---|---|---|---|
| Direct CUDA/cuBLAS | `--backend cuda` | NVIDIA inference and experimentation | Uses the CUDA Driver API, cuBLAS GEMM, PTX kernels for common float32 ops (fill, unary, binary, ternary, activations, RMSNorm, softmax, RoPE/RoPEEx, SDPA, GQA prefill/decode, causal mask, gather/concat), and native quantized matmul/get-rows for supported GGUF quant types. Unsupported ops route through CPU fallbacks while preserving tensor semantics. |
| MLX Metal | `--backend mlx` | Apple Silicon (alternative to GGML Metal) | GPU-accelerated path built on [mlx-c](https://github.com/ml-explore/mlx-c). Implements quantized ops (Q4_K_M, Q8_0, Q5_K, Q6_K, IQ2_XXS, IQ4_XS, IQ4_NL, MXFP4, etc.) without dequantizing to FP32, fused decode/prefill Metal kernels (fused QKV preprocess, fused gate+up+SiLUMul MoE, fused multi-dim KV write), compiled-graph kernels, async worker dispatch with periodic `async_eval` to overlap GPU/CPU work, batched MoE decode with stacked expert weight slabs, MoE expert offload, GGUF mmap pinned in physical RAM via `mlock(2)`, host-derived allocator caps (`TS_MLX_MEMORY_LIMIT_MB` / `TS_MLX_CACHE_LIMIT_MB` / `TS_MLX_WIRED_LIMIT_MB`), and a CPU fallback for ops that aren't yet wired up. Requires `libmlxc` (built locally by `TensorSharp.Backends.MLX/build-native-macos.sh` or located via `TENSORSHARP_MLX_LIBRARY` / `TENSORSHARP_MLX_LIBRARY_DIR`). |
| GGML Metal | `--backend ggml_metal` | Apple Silicon (default on macOS) | GPU-accelerated via Apple Metal. Quantized weights are mapped zero-copy from the GGUF file into Metal command buffers via host-pointer buffers, so the resident set stays close to the on-disk model size. |
| GGML CUDA | `--backend ggml_cuda` | NVIDIA inference through ggml | GPU-accelerated via GGML CUDA on Windows or Linux. Quantized weights are uploaded to device memory once at load time and the host copy is released afterwards. |
| GGML Vulkan | `--backend ggml_vulkan` | Vendor-neutral GPU inference through ggml | GPU-accelerated via GGML Vulkan on Windows or Linux — runs on AMD, Intel, and NVIDIA GPUs with a Vulkan 1.3 driver, using cooperative-matrix shaders (KHR coopmat / NV coopmat2) where the driver supports them. Weights are device-resident like GGML CUDA and the same fused whole-model decode/prefill graphs are used. Enabled automatically at native build time when the machine has a Vulkan runtime (loader installed); the build downloads a portable Vulkan toolchain (headers, glslc, SPIRV-Headers, and on Windows a loader import lib) via `eng/fetch-vulkan-toolchain.ps1` / `eng/fetch-vulkan-toolchain.sh` when no Vulkan SDK or distro dev packages are installed. Opt out with `--no-vulkan` (or `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF`). |
| GGML CPU | `--backend ggml_cpu` | Native CPU kernels | CPU inference using native GGML with optimized kernels. Quantized weights are mapped zero-copy from the GGUF file. |
| Pure C# CPU | `--backend cpu` | Portability and debugging | Portable CPU inference with no native dependencies. The managed matmuls run on a persistent spin-then-park worker pool sized at half the usable cores by default (`TS_CPU_THREADS` and the other `TS_CPU_*` knobs below), and on the direct video networks (Wan, MiniMax-H3) a quantized weight is multiplied straight out of its GGUF storage type instead of being expanded to F32 at load (`TS_DIRECT_QUANT_WEIGHTS=0` restores the expansion). DeepSeek V4.1 Flash, like DeepSeek V4 Flash below, runs a whole-model executor of its own here — the 100% pure-C# `DeepSeek4CpuExecutor`, a correctness and portability path rather than a serving one, whose compute width comes from `TS_DSV4_THREADS` (every processor by default on this backend) instead of `TS_CPU_THREADS`. |

**DeepSeek V4 Flash is the exception to the table above.** Its 284B compressed-sparse-attention MoE stack runs through one of three dedicated whole-model executors rather than the generic per-op path: a direct-CUDA engine (`--backend cuda`), the native ggml executor (`--backend ggml_cuda` / `ggml_vulkan`), and a 100% pure-C# CPU executor (`--backend cpu`) that serves quantized weights straight from the memory-mapped GGUF shards. All three layer-split the weights across every visible GPU (or, on CPU, stream them from the mapped shards), so a model far larger than one card still runs. See the [DeepSeek V4 card](docs/models/deepseek4.md).

**DeepSeek V4.1 Flash (`deepseek41`) has a managed whole-model executor of its own.** `--backend ggml_cuda` is its serving path, but `--backend cpu` runs the whole V4.1 graph on the 100% pure-C# `DeepSeek4CpuExecutor` — no ggml, no native library and no GPU, so it runs anywhere .NET runs: the ratio-1 and ratio-2 block compressors, the shared compressed and indexer caches with the lightning indexer's top-k, candidate block pruning, the Engram tables, the delayed hyper-connection gates, the shared expert, and the checkpoint's trained cache quantization (FP8 E4M3 raw rows, MXFP4 indexer, NVFP4 compressed). It is held to the independent PyTorch oracle `eng/dsv41-reference.py` at `atol=rtol=2e-5` on a five-layer F32 fixture, plus greedy-argmax agreement across one-shot prefill, chunk sizes 1/3/5/8 and reset, and it is a **correctness and portability path, not a serving one** — no throughput, load time or resident footprint has been measured for a V4.1 full checkpoint on it. The `deepseek41.engram.bin` sidecar is still mandatory; image and video do not work there, because the vision companion is a native ggml component and `--mmproj` throws; and a draft model / `TS_DSV4_DSPARK`, distributed TP groups, `TS_DSV41_TP` other than `0` and `TS_DSV41_ENGRAM_DEVICE` other than `0` are all refused before a weight is read. `--backend ggml_cpu` is a different path — the native graph on scalar ggml CPU kernels. See the [DeepSeek V4.1 card](docs/models/deepseek41.md).

**GLM 5.x (`glm-dsa`) is another exception.** Its 744B-A40B MLA + sparse-attention MoE runs through a native whole-model ggml executor on `--backend ggml_cuda` / `ggml_vulkan` / `ggml_cpu` / `ggml_metal`, and through a managed per-op path on `--backend cpu` (100% managed, no native dependencies) and `--backend cuda`; `TS_GLM_NATIVE=0` selects the managed path on a GGML backend for an A/B. **GLM-5.3 (not Flash) is the same `glm-dsa` block shape as GLM-5.2** — 79 blocks (78 trunk + one NextN), 256 routed experts at top-8 plus one shared expert, MLA with the lightning indexer, rope base 8e6 — so it loads on the GLM-5.2 path, on the same backends, with no new code and no new flag; it is text only twice over, because the repo publishes no mmproj at any quant and `--mmproj` on `glm-dsa` is warned about and ignored rather than failing the run. A split GGUF is handled by `GgufFile` itself — point `--model` at the `-00001-of-` shard of the quant directory (six shards for GLM-5.2, seven and 236.4 GiB for GLM-5.3's UD-Q2_K_XL). MLX does not run it. See the [GLM card](docs/models/glm.md), and [GLM-5.3](docs/models/glm.md#glm-53-glm-dsa) for what differs.

## Configuration file (CLI + Server)

Both `TensorSharp.Cli` and `TensorSharp.Server` can read their options from a JSON
file passed with `--config`, instead of (or in addition to) a long command line:

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/server-basic.json
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll       --config config/cli-basic.json
```

**Command-line options always win.** File values are applied first, then anything
you also pass on the command line overrides them — so one file can be reused across
machines while you override just what differs (`--config config/server-basic.json --backend ggml_cpu`).
Repeat `--config` to layer files; later files win over earlier ones.

The keys are the same long option names listed below (with or without the leading
`--`). Comments (`//`, `/* */`) and trailing commas are allowed.

| JSON value | Becomes | Example |
|---|---|---|
| string / number | `--key value` | `"max-tokens": 4096` → `--max-tokens 4096` |
| `true` | the bare switch `--key` | `"continuous-batching": true` → `--continuous-batching` |
| `false` / `null` | nothing (use the negation key, e.g. `"no-continuous-batching": true`) | |
| array | a repeated flag | `"stop": ["</s>", "<\|eot\|>"]` → `--stop </s> --stop <\|eot\|>` |
| object | a downloadable file (see below) | `{ "path": "...", "urls": ["..."] }` |

**Variables.** Define shared values once under `"variables"` and reference them
with `${name}` in any string value. A `${name}` not defined there falls back to an
environment variable of the same name, and variables may reference other variables.
Declare as many roots as you need — models in different folders each get their own.

```json
{
  "variables": { "modelRoot": "C:/models" },
  "backend": "ggml_cuda",
  "model": "${modelRoot}/Qwen3.5-9B-Q8_0.gguf",
  "mmproj": "${modelRoot}/Qwen3.5-mmproj-F16.gguf"
}
```

**Auto-download.** Any file option can be an object with a local `path` and one or
more `urls` instead of a plain string. If `path` is missing it is downloaded from
the first working URL (mirrors are tried in order), saved there, and reused on every
later run; download progress is printed to stderr. An optional `sha256` verifies a
freshly downloaded file.

```json
{
  "backend": "ggml_cuda",
  "model": {
    "path": "C:/models/Qwen3.5-9B-Q8_0.gguf",
    "urls": [ "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf" ]
  }
}
```

Ready-to-use examples live in [`config/`](config/) (`cli-basic.json`,
`server-basic.json`, `variables.json`, `auto-download.json`, `qwen-image-edit.json`)
— each uses real, public, ungated URLs, so it works on a fresh machine. See
[`config/README.md`](config/README.md) for the full reference.

## Console Application

Running `TensorSharp.Cli` with no arguments prints the full parameter reference —
every option with its description, default, range, and an example — and exits
before any logging or model machinery starts; `--help` (also `-h`, `-?`, `/?`)
does the same.

```bash
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --help
```

```bash
# Text inference
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_metal

# Text inference on Windows/Linux + NVIDIA GPU
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --output result.txt \
    --max-tokens 200 --backend ggml_cuda

# Interactive turn-by-turn chat (REPL) with KV cache reuse and slash commands
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal --interactive
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal -i \
    --system "You are a terse assistant." --temperature 0.7 --top-p 0.9 --think

# Image inference (Gemma 4, Qwen 3.5-family)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --image photo.png --backend ggml_metal

# Video inference (Gemma 4)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --video clip.mp4 --backend ggml_metal

# Audio inference (Gemma 4)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --audio speech.wav --backend ggml_metal

# PDF document input: born-digital PDFs are text-extracted and inlined into the
# prompt; scanned PDFs become page images and need a vision model (--mmproj or a
# built-in vision encoder). --input provides the instruction over the document.
echo "Summarize the key findings of this paper." > question.txt
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --pdf paper.pdf --input question.txt \
    --max-tokens 300 --backend ggml_metal

# DiffusionGemma text-diffusion generation
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <diffusion-gemma.gguf> --input prompt.txt --backend ggml_metal \
    --max-tokens 256 --diffusion-steps 48 --diffusion-seed 0

# Qwen-Image-Edit image editing (prompt + input image -> edited image)
# The VAE + Qwen2.5-VL text-encoder companions are resolved next to the DiT GGUF
# (or set --qwen-image-vae / --qwen-image-vl / --qwen-image-mmproj).
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <qwen-image-edit-DiT.gguf> --image input.png \
    --prompt "Make the sky a dramatic sunset." --output edited.png \
    --backend ggml_cuda --diffusion-steps 30 --cfg 2.5 --diffusion-seed 0

# MiniMax-H3 video generation with sound (prompt -> H.264 MP4 plus a 32 kHz
# stereo .wav sidecar). One diffusion transformer denoises a packed video+audio
# latent, so the soundtrack is model output rather than something dubbed on
# afterwards. The Qwen3-VL-32B text encoder, the video VAE and the audio VAE are
# resolved next to the DiT GGUF (or set --video-text-encoder / --video-vae /
# --audio-vae). H3 is CFG-distilled: --cfg 1.0 is required, and 4-8 steps is the
# fast lane against a 20-step default. See "Video generation with audio
# (MiniMax-H3)" below and docs/models/minimax-h3.md.
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <minimax_h3_fl2va_pruned-Q4_K.gguf> \
    --prompt "a red fox trotting through falling snow, cinematic" --output fox.mp4 \
    --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
    --backend ggml_cuda

# MiniMax-H3 conditioning: on the fl2va checkpoint --image IS the first frame and
# --end-image pins the last one. On the ref2va checkpoint the same picture is an
# identity reference for a brand-new scene instead (--ref-image, repeatable up to
# nine, plus --ref-video / --ref-video-audio / --ref-audio). --video-mode
# t2v|i2v|fl2v|ref states which reading you meant when it is not obvious.
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <minimax_h3_fl2va_pruned-Q4_K.gguf> \
    --image start.png --end-image end.png --video-mode fl2v \
    --prompt "a slow cinematic push-in" --output morph.mp4 \
    --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
    --backend ggml_cuda

# The same run from a shipped config, which auto-downloads all four networks on
# first use (config/minimax-h3-ref2va.json is the reference checkpoint).
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --config config/minimax-h3-fl2va.json \
    --prompt "a red fox trotting through falling snow, cinematic" --output fox.mp4

# Wan video generation, video only (prompt -> H.264 MP4). The UMT5-XXL
# text-encoder GGUF and video-VAE companions are resolved next to the DiT GGUF
# (or set --video-text-encoder / --video-vae). Wan 2.1 T2V, Wan 2.2 TI2V-5B, and
# Wan 2.2 A14B (both experts) are auto-detected, and so are step-distilled
# (Turbo / Lightning / FastWan) checkpoints -- 4 DiT passes instead of 100 for
# the same video. See the "Video generation (Wan)" section below and
# docs/models/wan.md.
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <Wan2.2-TI2V-5B.gguf> \
    --prompt "a lovely cat walking through a garden" --output cat.mp4 \
    --width 832 --height 480 --video-frames 49 --backend ggml_cuda \
    --diffusion-seed 7

# Wan 2.2 image-to-video: --image supplies the first frame; the prompt controls
# motion, camera and scene changes (TI2V-5B or I2V-A14B checkpoints).
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <Wan2.2-TI2V-5B.gguf> \
    --prompt "the cat runs toward the camera, cinematic tracking shot" \
    --image first_frame.png --output cat_run.mp4 --backend ggml_cuda

# Thinking / reasoning mode
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal --think

# Tool calling
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --tools tools.json

# Agent Skills: register a directory of skills and select one for this run. Only
# each skill's one-line description costs context up front -- the model pulls the
# SKILL.md body and any reference files it needs through built-in skills_list /
# skills_read tools that TensorSharp answers in process. --list-skills prints the
# registry (names, descriptions, files, warnings) and exits.
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --skills-dir ~/skills --skill pdf

# With sampling parameters
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.2 --seed 42

# DSpark block speculative decoding (DeepSeek V4): point --model at the first shard
# and --draft-model at the DSpark drafter GGUF. Every emitted token is still drawn
# from a trunk row, so a greedy run stays byte-exact and a sampled one stays in
# distribution -- --temperature 0 below just makes the comparison exact.
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <DeepSeek-V4-Flash-...-00001-of-00005.gguf> \
    --backend ggml_cuda --draft-model <DSpark-drafter.gguf> \
    --input prompt.txt --max-tokens 200 --temperature 0

# Batch processing (JSONL)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input-jsonl requests.jsonl \
    --output results.txt --backend ggml_metal

# Multi-turn chat simulation with KV-cache reuse (mirrors the web UI behavior)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --multi-turn-jsonl chat.jsonl \
    --backend ggml_metal --max-tokens 200

# Throughput benchmark: best-of-N prefill and decode timing
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal \
    --benchmark --bench-prefill 256 --bench-decode 128 --bench-runs 3

# KV-cache reuse benchmark: measure prefill speedup across multiple chat turns
# (compares with-cache vs forced-reset prefill latency for an 8-turn conversation)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_metal \
    --bench-kvcache --bench-kv-turns 4 --max-tokens 64

# Inspect the rendered prompt and tokenization without running inference
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --dump-prompt

# Compare hardcoded fallback templates against GGUF Jinja2 templates for every
# *.gguf file in a directory (useful when adding new architectures)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --test-templates ~/models

# Tensor parallelism: split a model across 2 GPUs in one process
# (--backend cuda, ggml_cuda, or ggml_vulkan)
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --input prompt.txt --backend cuda --tp 2

# Distributed tensor parallelism: 2 nodes × 2 GPUs each (4 GPUs total)
# Node 0:
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
# Node 1:
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"
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
| `--pdf <path>` | PDF document input (one-shot mode). Born-digital PDFs have their complete text layer extracted and inlined into the prompt (page cap via `TS_PDF_MAX_PAGES`); scanned PDFs are rasterized to page images and require a vision model (`--mmproj` or a built-in vision encoder). `--input` text becomes the instruction over the document. |
| `--mmproj <path>` | Path to the multimodal projector GGUF file |
| `--max-tokens <N>` | Maximum tokens to generate (default: 100) |
| `--backend <type>` | Compute backend: `cpu`, `cuda`, `mlx`, `ggml_cpu`, `ggml_metal`, `ggml_cuda`, or `ggml_vulkan` |
| `--gpu-device <N>` | Vulkan device index for the `ggml_vulkan` backend on multi-GPU hosts (e.g. an integrated Intel GPU next to a discrete NVIDIA one). Defaults to device 0; use `--list-gpus` to see the indices. Also settable via the `TS_GGML_VULKAN_DEVICE` env var. |
| `--list-gpus` | List the Vulkan devices ggml-vulkan can see (index + adapter name) and exit |
| `--n-cpu-moe <N>` / `-ncmoe <N>` | Keep the routed Mixture-of-Experts weights of the first N layers in system RAM and multiply them on the CPU; attention, norms, the router and the always-active shared expert stay on the accelerator (llama.cpp's `--n-cpu-moe` equivalent). This is what lets a 35B-A3B MoE fit beside a long-context KV cache on a 12-16 GB card. Pass `all` for every layer. Default: 0 on every architecture, DeepSeek V4 and GLM 5.x included — a model that does not fit is refused at load with the number of layers that would make it fit, rather than silently offloaded (env `TS_N_CPU_MOE`). |
| `--cpu-moe` / `-cmoe` | Shorthand for `--n-cpu-moe all`. Default: off (env `TS_CPU_MOE`). |
| `--cpu-moe-threads <N>` | Worker threads for the host-side expert matmul. Default: **half** the CPU parallelism this process can actually use (`hardware_concurrency` clamped by the scheduler affinity mask and the cgroup CPU quota) on hosts with more than 8, all but one below that. The other half is not waste — the accelerator submission threads, and in `TensorSharp.Server` Kestrel and the scheduler, have to be schedulable too, and .NET sizes its own pool from the machine's CPU count rather than the cgroup quota. Sizing this near the quota is a cliff, not a slope: on a 95-CPU quota the hosted 26B MoE measured 20.7 tok/s at 64 threads and 8.2 at 71. Raise it on a dedicated box (env `TS_CPU_MOE_THREADS`). |

**Backend support.** MoE CPU offload is implemented on the GGML backends
(`ggml_cuda`, `ggml_vulkan`, `ggml_metal`, `ggml_cpu`) for every MoE
architecture, and on the pure-C# `cuda` backend for DeepSeek V4 only — that
engine serves experts from its own stacked-expert device buffer everywhere
else. Asking for offload on a combination that does not implement it prints a
`[moe-offload] WARNING` and proceeds without saving VRAM, rather than failing
quietly. Measured on gemma-4-26B-A4B (`--cpu-moe`, peak VRAM): `ggml_cuda`
15244 → 5756 MiB; `cuda` 14261 → 14253 MiB (no-op, warns).
| `--kv-cache-dtype <type>` | KV cache precision: `f32`, `f16`, `q8_0`, or `q4_0` (default: auto — the backend/model pick; env `KV_CACHE_DTYPE`). Half-precision / quantized KV caches reduce memory at the cost of small numerical drift; `q4_0` (~0.56 bytes/elem, ~1/7 of f32) is the most aggressive tier for very long (128K–256K) contexts where the KV cache dominates memory. Block-quantized caches (`q8_0`/`q4_0`) require the native GGML flash path. |
| `--interactive` / `-i` / `--chat` | Start an interactive REPL chat session (turn-by-turn input/output) with KV cache reuse, slash commands, hot-swappable model/backend/projector, file attachments (image, audio, video, text) and live sampling tuning. See the **Interactive REPL commands** section below for the full list. |
| `--no-prefix-cache` | Do not forward the shared part of the prompt before the first message. By default an interactive chat forwards its system block, tool declarations and skill catalog at startup so the first message continues from them instead of prefilling them (measured 18.5s → 0.3s on an agent configuration); the price is that the session takes that long to become ready. Unrelated to `--warmup-runs` |
| `--system <text>` | System prompt to seed the interactive session (overridden inside the REPL by `/system`) |
| `--system-file <path>` | Read the initial system prompt from a UTF-8 text file (alternative to `--system`) |
| `--think` | Enable thinking/reasoning mode (chain-of-thought). Opt-in on every family, GLM 5.x included: without it the GLM template closes the reasoning block immediately (`<think></think>`) so the model answers directly, and with it the prompt carries `Reasoning Effort: Max` and leaves the block open for the model to close. `/think on\|off` toggles it inside the REPL. |
| `--tools <path>` | JSON file with tool/function definitions. Wire formats differ by family and the parser is picked from the architecture — GLM 5.x emits XML (`<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`, one element per argument, values `tojson`-encoded when they are not plain strings) rather than a JSON body, and the server parses that back into the usual OpenAI tool-call fields so clients see the standard shape. |
| `--skills-dir <path>` | Directory to scan for Agent Skills (a folder holding `SKILL.md` files, or a single skill directory). Repeatable; scanned in the order given, up to three levels deep. Without it, a `skills` directory beside the binary is used and created if missing. A path that does not exist is a startup error naming the flag. Env: `TS_SKILLS_DIR` (a path-separator-separated list). |
| `--skill <name>` | Select a skill for this run, by the name in its `SKILL.md` (which is also its directory name). Repeatable. On tool-capable model families, selection advertises the skill's metadata and scopes the built-in skill tools to it; it does **not** inline the `SKILL.md` body. The body is inlined only as the fallback for a family/request that cannot use tool declarations (including structured-output requests), when it fits the prompt budget. |
| `--list-skills` | Print the skill registry — name, description, origin, bundled files, size, and any load warnings or errors — and exit. |
| `--no-skills` | Turn Agent Skills off entirely: no scanning, no prompt block, no tools. Env: `TS_NO_SKILLS` (anything but `0` counts as on). |
| `--skills-no-discovery` | Do not advertise unselected skills to the model. Without it, every registered skill's name and description are listed so the model can load one you did not think to name; with it, a run sees exactly the skills `--skill` selected. |
| `--skills-allow-exec` | Allow the model to run a selected skill's bundled scripts through `skills_run`. **Off by default; treat every enabled script as untrusted code.** TensorSharp launches only known interpreters (`.py`, `.js`/`.mjs`, `.sh`, `.bash`) without a shell, scrubs the environment, bounds runtime and output, mounts the selected skill read-only, and runs in the active Web/CLI session or OpenAI/Ollama request workspace (or per-call scratch when code execution is off). The default `--skills-sandbox required` adds OS confinement and refuses execution when it is unavailable; network remains denied unless `--skills-allow-network` is also set. Env: `TS_SKILLS_ALLOW_EXEC` (anything but `0` counts as on). |
| `--skills-max-rounds <n>` | How many times the model may fetch skill content - or run, read and fix code - before it must answer, 1-64 (default: `8`, or `24` when `--code-exec` is on, because writing a file, running it and fixing the traceback takes more steps than reading files; a value set here is used as given either way). Each round is a full generation, so this bounds the cost of a model that keeps mis-naming a file. When the budget runs out the model is told so in the conversation, answers from what it read, and says what it did finish. Env: `TS_SKILLS_MAX_ROUNDS`. |
| `--skills-sandbox <off\|preferred\|required>` | How hard to insist on OS isolation for a skill's scripts. `required` (the default) refuses to run them on a host with no safe sandbox rather than running them unconfined; `preferred` explicitly accepts weaker isolation; `off` keeps only the runner's interpreter, environment, time and output limits. macOS uses `sandbox-exec`; Linux requires `bwrap` 0.12.0 or newer (older releases have a known setup-time symlink escape). Windows Job Objects bound only the process tree, not filesystem or network access, so `required` refuses there and `preferred` is the explicit weaker-isolation opt-in. Env: `TS_SKILLS_SANDBOX`. |
| `--skills-allow-network` | Let a sandboxed skill script reach the network. Denied by default. Env: `TS_SKILLS_ALLOW_NETWORK`. |
| `--code-exec` | Offer the model a `shell` tool: it types a real command line, the host runs it in an OS sandbox, and the model reads back the exit code and everything the command printed, stdout and stderr merged. This is how the model RUNS things and looks around — run a program, `rg` for a pattern, move and delete files, install what it needs, check its own output — which is the shape OpenAI Codex and Claude Code use. Reading and changing files is the file tools' job, below. Arguments: `command` (required), `workdir` (that one call only), `timeout_ms`, and `run_in_background` for something meant to keep running, whose result names a log file to read later with an ordinary command. Four more tools come with it, and they are where file work actually happens. `read_file` (`path`, `offset`, `limit`) shows a file's current bytes with line numbers rendered `   42 \| text`, and re-reading an unchanged file says so instead of repeating it. `edit_file` (`path`, `old_string`, `new_string`, `replace_all`) replaces one exact string in one file — Claude Code's `Edit`, parameter for parameter, and the shape Anthropic publishes as `str_replace_based_edit_tool`. It must match exactly and exactly once; if it matches twice the call is refused and the matches are located for you, and if it matches nothing you are shown the file's real numbered lines around the closest thing to what you asked for. `write_file` (`path`, `content`) creates a file or deliberately replaces one whole — and when it replaces one, the host counts how many of the lines came back byte-identical and says so. `apply_patch`, Codex's `*** Begin Patch` envelope, creates, updates, deletes and renames several files in one all-or-nothing call, and is reachable two ways — as a tool call, and by typing it into the shell as a heredoc, which the host intercepts and never executes. The split follows the two references rather than inventing a format: string replacement for the common one-file change, an atomic envelope for the multi-file one. All of them exist beside a shell that could rewrite any file with a heredoc because a heredoc re-emits the *whole* file: a three-line fix costs every already-correct line and re-rolls each of them. And because the **host** places the bytes, from text it either finds or refuses to guess at, rather than the model retyping a file it half-remembers. `apply_patch`'s matcher is a line-for-line port of the reference V4A applier: exact match, else trailing-whitespace-insensitive, else leading-and-trailing-insensitive, else fail — no similarity scoring, no nearest match. `edit_file`'s ladder is the union of what both references tolerate — exact, then typographic quotes and dashes folded to their ASCII forms, then a literal `\uXXXX` decoded, then line-number prefixes stripped — every rung above the first reported on the result, and the replacement written back in the file's own punctuation so tolerance never silently changes bytes nobody asked about. All five tools share a Web/CLI session workspace or an isolated OpenAI/Ollama request workspace across its internal rounds. Only direct callers that provide no workspace receive the shell alone. Every tool is answered in process; none is ever handed back to the client. Off by default. Env: `TS_CODE_EXEC` (anything but `0` counts as on). |
| `--code-exec-allow-install` | Let the model install the packages it needs (pip / npm) into the active Web/CLI-session or HTTP-request workspace, so later commands — and skill scripts — can import them within that workspace's lifetime; requires `--code-exec`. This permission does **not** give a model-authored command network access. The host reads each recognised install for its tool and package names, validates them, and performs the install itself with an argument vector it built, wheels only (`--only-binary=:all:` / `--ignore-scripts`). It substitutes that command with `true` or `false`, preserving `&&`, `\|\|`, pipelines and loops; in `pip install x && python y.py`, `y.py` follows the command network policy (offline by default, unrestricted only with `--code-exec-allow-network`). Source-changing options such as `--index-url`, `-i`, `--find-links`, `--registry`, URL requirements, and installers the host cannot perform (`uv`, `poetry`, `gem`, `cargo`, `go`) are refused; `-r requirements.txt` is read and validated line by line. Env: `TS_CODE_EXEC_ALLOW_INSTALL`. |
| `--code-exec-allow-network` | Give every model-authored command unrestricted host IP-network access: generated code can resolve DNS, fetch URLs, follow redirects, call remote APIs, reach LAN/loopback services and open IP listening sockets. **Off by default** and requires `--code-exec`. Write and home-read confinement remain active on macOS and Linux. Linux additionally bounds descendants with a PID namespace. On macOS, children inherit Seatbelt and ordinary process groups are stopped, but a deliberately detached child can outlive the request; every result reports that gap. macOS denies common `/private/tmp/com.apple.launchd*` pathname sockets while permitting runtime-required Mach lookup and the exact mDNSResponder pathname socket required for DNS, and Linux hides common `/run` endpoints, but this is not a complete local Unix-IPC boundary: macOS retains shared-temporary-directory Unix IPC for compatibility, and Linux's host network namespace may expose abstract sockets and pathname sockets outside `/run`. Other host-readable files and IP services may therefore be reached and exfiltrated; remote prompt injection and untrusted downloads are additional risks. Credential-free host HTTP/SOCKS proxy settings are passed only in this mode. Configured custom-CA bundles up to 16 MiB are read once; only validated public certificates are copied into a read-only per-session snapshot, so the host source path and adjacent data are not exposed. Authenticated proxies need a credential-free host-side forwarder. Package/install-domain allow-lists constrain only the recognised host installer; unrestricted generated code can fetch or execute artifacts directly. This is independent of package installs and of `--skills-allow-network`, which controls `skills_run`. On Windows, `--code-exec-unconfined` is still required. Env: `TS_CODE_EXEC_ALLOW_NETWORK` (anything but `0` counts as on). |
| `--code-exec-packages <list>` | Restrict installs to these package names, comma-separated; anything else is refused and the model is told which names are allowed (default: empty, meaning any package — and at most 16 packages in one install either way). Matching is on the bare name, so a version the model pins (`numpy==2.1.0`) still matches the entry `numpy`. It was retired when the tool surface became a shell, because a model typing its own `pip install` could spell the request in ways a name list could not see; it is back because the **host** performs every install again — it reads the names out of the model's command rather than running that command, and builds the installer's argument vector itself, so the list applies to recognised requests however they were spelled (`pip`, `pip3`, `python -m pip`, a requirements file). Only meaningful with `--code-exec-allow-install`. With `--code-exec-allow-network`, this list is not a security boundary: generated code can fetch or execute artifacts without using the host installer. |
| `--code-exec-install-index <url>` | Point host-performed installs at an operator-selected package index instead of the installer's default. A model-authored `--index-url` is still refused. The index host is added to the install egress allow-list automatically; add any separate download hosts with `--code-exec-install-domains`. Requires `--code-exec-allow-install`. Default: unset. Env: `TS_CODE_EXEC_INSTALL_INDEX`. |
| `--code-exec-install-domains <list>` | Hosts the **host-performed package install** may reach, comma-separated — exact names or `*.suffix` wildcards (default: `pypi.org,files.pythonhosted.org,registry.npmjs.org`; empty disables pinning). Where macOS Seatbelt can pin one loopback port, the installer must use a one-install CONNECT proxy and reaches only these hosts. Bubblewrap cannot pin a port, so Linux instead relies on the host-built installer argument vector, default index and wheels-only policy. This list does not restrict ordinary generated commands after `--code-exec-allow-network` grants them unrestricted host-network access. Env: `TS_CODE_EXEC_INSTALL_DOMAINS`. |
| `--code-exec-timeout <seconds>` | How long one command may run before it is killed (default `120`, up from the 30 s the program runner used, because a shell command is also how packages are installed and how a build or a test suite is run). A call may ask for less or more through `timeout_ms`, up to 10 minutes; a command that runs over is stopped and the model still gets everything it printed first, because a timeout that discards the output only makes it run the command again blind. |
| `--code-exec-temperature <number>` | Optional sampling temperature for turns that can run code, from `0` to `2`. It changes only sampling values still at TensorSharp's built-in defaults, so an explicit client/operator temperature still wins; a negative value clears the override. Coding turns remove the built-in chat repetition penalty even when no temperature is set. Default: unset. |
| `--code-exec-shell <path\|name>` | Which shell to run commands through, when the host's own choice is wrong or absent. Default: `bash`, then `sh`, on macOS and Linux; PowerShell 7 (`pwsh`), then Windows PowerShell, on Windows — where a bare `bash` on PATH is refused on purpose, because it is the WSL launcher and running through it would put the command inside a Linux VM while the job object holds only the launcher on the Windows side. Point this at a real bash (Git Bash, MSYS2) to use one there. The model is told which dialect it is typing into, so this also changes the examples in its tool description. |
| `--code-exec-max-output <bytes>` | How much of one command's output is kept and shown to the model (default `32768`). What does not fit is dropped from the **middle**, keeping the head and the tail: the end of a build or a test run is where the failure is, and head-only truncation discards exactly the part that was wanted. |
| `--code-exec-unconfined` | Run model-authored commands even where the OS cannot confine them. The CLI and server both accept this explicit escape hatch. Windows requires it because a job object bounds the process tree but cannot restrict filesystem or network access; commands there consequently run with this process account's access to both, regardless of the narrower macOS/Linux network switch. Do not enable it on a server reachable by users you do not trust. |

**How the shell is used, and what it may reach.** The model does all its work
through that one command line: it writes a file with a heredoc, runs it, greps
it, reads its own traceback and fixes it — which is why `--skills-max-rounds`
rises from 8 to 24 when this is on. Web/CLI keeps one filesystem workspace for
the whole chat session; each OpenAI Chat, OpenAI Responses, or Ollama HTTP
request gets a private workspace across its internal tool rounds and deletes it
after the response. Skill scripts share that workspace when code execution is
enabled, so files and host-installed packages remain available for the applicable
session or request. Every shell call is nevertheless a fresh confined process;
do not rely on virtualenv activation, PATH changes, or a resident shell carrying
from one call to the next. The network rule is the
host's, not the model's: commands are offline by default, and only the operator's
`--code-exec-allow-network` (or `TS_CODE_EXEC_ALLOW_NETWORK`) opt-in gives every
model-authored command unrestricted host IP-network access, including LAN/loopback
services and IP listening sockets. Write and home-read confinement remains active
on macOS and Linux. Linux additionally bounds descendants with a PID namespace.
On macOS, children inherit Seatbelt and ordinary process groups are stopped, but a
deliberately detached child can outlive the request; every result reports that gap.
macOS denies common `/private/tmp/com.apple.launchd*` pathname sockets while permitting runtime-required Mach lookup and the exact mDNSResponder pathname socket required for DNS, and Linux
hides common `/run` endpoints, but this is not a complete local Unix-IPC boundary:
macOS retains shared-temporary-directory Unix IPC for compatibility, and Linux's
host network namespace may expose abstract sockets and pathname sockets outside
`/run`. It is
independent of `--skills-allow-network`, which applies only to `skills_run`.
`--code-exec-allow-install` also does not imply command networking: the host
performs recognised installs itself — the
line is split into its simple commands (quoting and heredoc bodies respected),
each recognised install is read for its tool and its package names, the host
runs the installer with an argument vector it built, and that command is then
substituted out of the line — `true` when it worked, `false` when it did not —
so the `&&`, the pipeline and the loop body around it still mean exactly what
the model wrote: a failed install does not run what followed a `&&`, and does
run the fallback after a `||`. The failure is reported either way. Giving the install command its own socket was the earlier design, and it
had two holes that could not both be closed: the command line is the model's,
so `--index-url` chose the index, and a socket belongs to the whole line, so
anything sharing it with the install shared its reach — which on a host whose
sandbox cannot pin egress to a proxy was the whole internet. Every result also
says what was *not* confined on this host. The retired `--code-exec-languages`
is refused by name at startup, with no replacement to point at because it has
none: a shell reaches every interpreter on PATH and PATH must contain `/bin`
and `/usr/bin` for the shell to work at all, so `--code-exec` now reports which
interpreters this host has instead of pretending to gate them — and an old
script gets that error instead of watching a setting be ignored.

| Option | Description |
|---|---|
| `--spec` / `--no-spec` | Enable speculative decoding (default off). `--spec` is the explicit opt-in for drafters embedded in the trunk checkpoint — GLM 5.2's NextN block, GLM-5.3's and Qwen 3.6's, with nothing extra to download — because loading them pages extra weights into VRAM; a drafter that ships as its own GGUF is enabled by `--draft-model` alone, and an explicit `--no-spec` vetoes either. The head drafts up to `--spec-draft` tokens and the trunk verifies them in one batched forward; every emitted token still comes from a trunk row, so the stream is the one plain decoding would have produced (argmax under a greedy config, in distribution under a sampler) and this is a speed path only. Engages on every single-sequence path (`--input`, `--input-jsonl`, `--multi-turn-jsonl`, `--interactive`). **Must be on the command line before the model loads**: for glm-dsa it is what tells the native loader to page the ~3 GiB NextN layer into VRAM, and that layer competes with the KV cache for the memory the context is sized against. Refused under `--tp N>1` on a checkpoint whose draft block borrows the trunk's LM head, which includes GLM 5.2 and GLM-5.3 — on those, speculation engages on the default layer split (no `--tp`). Env: `TS_SPEC` (legacy `TS_MTP_SPEC`, still read by the glm-dsa native loader; glm-dsa also honours `TS_GLM_MTP=1`/`0`, which overrides both, for A/B runs). |
| `--spec-type <name>` | Speculation **algorithm**: `auto` (default, use the checkpoint's own drafter), `draft-head`, `block`, or `ngram`. `ngram` needs no trained weights and works on every model — it drafts by finding where the last few tokens occurred earlier in the context and proposing what followed, so it is strong wherever the answer quotes its input (summarizing, editing, translating, repetitive structured output, agentic loops) and falls back to plain decode elsewhere. Measured 45.2 tok/s against 31.4 plain (1.44x) on Qwen3.5-9B (Q8_0, `ggml_metal`, M5 Pro) — a checkpoint that ships no draft head at all — with byte-identical output. Env: `TS_SPEC_TYPE`. See [Speculative Decoding in TensorSharp](docs/speculative_decoding.md). |
| `--spec-draft <N>` | Maximum tokens drafted per speculative step (range 1-64, default `8`). Also sizes the native graph cache at load, so pass it alongside `--spec` rather than relying on the default. A block drafter additionally clamps it to its trained block size, which is also its default there — 5 for DSpark, 15 for Muse-Glimmer's DFlash, 7 for Qwen 3.8's DFlash2. On a **recurrent** trunk (Qwen 3.5/3.8's GatedDeltaNet layers) a narrow window is worth far more than a wide one: it bounds both the verify width and the rollback re-forward, and `--spec-draft 3` was 1.6x faster than the default on Qwen3.8-27B. On Qwen 3.5 under the GGML backends a typed value of 8 or more is also a known **correctness** bug — nine verify rows diverge from plain greedy — so stay at 7 or lower there; see [speculative decoding](docs/speculative_decoding.md). Env: `TS_SPEC_DRAFT` (or `TS_MTP_DRAFT`). |
| `--spec-pmin <f>` | Draft-confidence gate in `[0, 1]`; drafting stops at the first token below it, and `0` means never gate. What the number MEANS is the algorithm's business, so each brings its own default: `0.15` for a per-token head (top-1 probability over its top-10 logits), `0.35` for a block drafter (the CUMULATIVE prefix probability — the product of the confidence head's per-position estimates, so the same number is far stricter; lower drafts further and rolls back more, higher falls back to plain decode more often), `0` for n-gram (where it scales the required match length instead). Env: `TS_SPEC_PMIN` (or `TS_MTP_PMIN`). |
| `--draft-model <path>` | Speculative-decoding drafter GGUF, for every drafter that ships as its own file — DeepSeek V4's DSpark support module (see [DeepSeek V4](docs/models/deepseek4.md#dspark-speculative-decoding)), Muse-Glimmer's DFlash and Qwen 3.8's DFlash2 block drafters (see [Muse-Glimmer](docs/models/muse-glimmer.md#3-dflash-speculative-decoding); env `TS_MUSE_GLIMMER_DFLASH`), and Gemma 4's `gemma4-assistant` per-token head. The file's own `general.architecture` decides how it loads — you never pick a mechanism. Naming a file here enables speculation by itself; no `--spec` is needed beside it, and an explicit `--no-spec` vetoes it. The draft's hidden size must match the target (pair the 12B target with its 12B draft, not the 26B-A4B one); a mismatched, missing, or incomplete draft GGUF fails fast at startup. Qwen 3.6, GLM 5.2 and GLM-5.3 embed their NextN block in the trunk GGUF and need no such flag — they take `--spec` instead. A block drafter drafts a whole block per step and the trunk verifies it in one batched forward. Every emitted token is still drawn from a trunk row — with argmax under a greedy config, with the run's own sampler otherwise — so the output stream is unchanged either way. Block drafting engages on every single-sequence path (`--input`, `--multi-turn-jsonl`, `--interactive`) with `--backend cuda` or `--backend ggml_cuda`. Env: `TS_SPEC_DRAFT_MODEL`, `TS_DSV4_DSPARK`. |
| `--temperature <f>` | Sampling temperature (0 = greedy) |
| `--top-k <N>` | Top-K filtering (0 = disabled) |
| `--top-p <f>` | Nucleus sampling threshold (1.0 = disabled) |
| `--min-p <f>` | Minimum probability filtering (0 = disabled) |
| `--repeat-penalty <f>` | Repetition penalty (1.0 = none) |
| `--penalty-last-n <N>` | How many of the most recent tokens the repeat / presence / frequency penalties consider. `0` disables history penalties, `-1` uses the whole history (default: 64) |
| `--presence-penalty <f>` | Presence penalty (0 = disabled) |
| `--frequency-penalty <f>` | Frequency penalty (0 = disabled) |
| `--seed <N>` | Random seed for **text** sampling (-1 = non-deterministic). Image and video generation take their noise from `--diffusion-seed` instead. |
| `--stop <string>` | Stop sequence (can be repeated) |
| `--dump-prompt` | Render the prompt + tokenization and exit (no generation) |
| `--benchmark` | Run a synthetic prefill/decode throughput benchmark |
| `--bench-prefill <N>` | Synthetic prefill length in tokens (default: 32) |
| `--bench-decode <N>` | Synthetic decode length in tokens (default: 64) |
| `--bench-runs <N>` | Number of benchmark runs; reports best and average (default: 1) |
| `--bench-kvcache` | Run a multi-turn KV-cache reuse benchmark (with-cache vs forced-reset prefill) |
| `--bench-kv-turns <N>` | Number of conversation turns for `--bench-kvcache` (default: 4, max: 8) |
| `--bench-chunked` | Run a chunked-prefill micro-benchmark (Gemma 4) |
| `--bench-fixed-tokens` | Feed a predetermined decode-token stream and time inference without host greedy sampling, for llama-bench-style comparison. An untimed greedy correctness chain is still reported |
| `--paged-bench` | Benchmark the cross-session paged KV cache: pay the full prefill once, then measure how much a second identical-prefix request recovers |
| `--paged-bench-prompt <N>` | Prompt length in tokens for `--paged-bench` (default: 2048) |
| `--paged-bench-trials <N>` | Trials for `--paged-bench` (default: 3) |
| `--warmup-runs <N>` | Number of throw-away forward passes before timing real text / multimodal prompts (default: 0) |
| `--test-chunked-prefill` | Run the chunked-prefill correctness check (compares chunked vs non-chunked logits) |
| `--correct-prefill <N>` | Prompt length used by `--test-chunked-prefill` |
| `--correct-decode <N>` | Decode length used by `--test-chunked-prefill` |
| `--diffusion-steps <N>` | DiffusionGemma denoising steps per block (default: 48). For Qwen-Image-Edit, the FlowMatch-Euler step count — omit for auto (30, or the step count of a loaded Lightning LoRA). |
| `--diffusion-seed <N>` | Noise seed for the diffusion paths: DiffusionGemma's deterministic sampler (default: 0), Qwen-Image-Edit, and video generation (Wan, MiniMax-H3), where leaving it out draws a fresh random seed each run. This is the seed that decides what a clip looks like — `--seed` is the text sampling seed and does not affect it. |
| `--diffusion-blocks <N>` | DiffusionGemma block-autoregressive canvas count. `0` derives the count from `--max-tokens` and the model canvas length. |
| `--image <path>` | Input image for Qwen-Image-Edit (also the image input for multimodal chat). Required to trigger image-edit mode on a `qwen_image` DiT GGUF. |
| `--prompt <text>` | Qwen-Image-Edit edit instruction (falls back to `--input` file contents if omitted). |
| `--output <path>` | Qwen-Image-Edit output PNG path (default: `edited.png`). |
| `--cfg <F>` | Qwen-Image-Edit true-CFG guidance scale (`<= 1` disables the negative pass). Omit for auto: 2.5 (the Qwen-Image-Edit-2511 recommendation; 4.0 over-guides and distorts faces), or 1.0 when a Lightning LoRA is loaded. Shares `--diffusion-steps` / `--diffusion-seed` for step count and seed. On MiniMax-H3 the only accepted value is `1.0` (its default): the checkpoint ships CFG-distilled and anything higher is refused up front rather than run and degraded. `TensorSharp.Server` has no `--cfg` at all — a request body can still carry `cfg`. |
| `--qwen-image-vae <path>` | Override the resolved Qwen-Image VAE companion (`.gguf` or `.safetensors`). |
| `--qwen-image-vl <path>` | Override the resolved Qwen2.5-VL-7B text-encoder GGUF. |
| `--qwen-image-mmproj <path>` | Override the resolved Qwen2.5-VL mmproj (vision grounding) GGUF. |
| `--qwen-image-lora <path>` | Qwen-Image-Edit Lightning distillation LoRA (`.safetensors`). Applied as a runtime F32 side-path next to each targeted projection (`y = W_quant·x + b + (alpha/rank)·up·(down·x)`) with the quantized base weights left untouched — **not** merged into them. Auto-derives the step count from the file name (e.g. 4 or 8), switches CFG to 1.0 and pins the timestep shift to 3, so the default 30 steps × 2 CFG passes (60 DiT forwards) become 4–8. Needs the whole-model or fused per-block CUDA forward — on a path without the side-path it throws rather than emitting noise. Env: `TS_QWEN_IMAGE_LORA`. |
| `--offload-cpu` | Stream the DiT weights from RAM instead of holding them resident in VRAM: slower per step, but native ~1 MP edits fit on small cards. Default: auto — it engages by itself only when the target resolution does not fit beside the resident weights |
| `--width <px>` / `--height <px>` | Output size for Qwen-Image-Edit and video generation. Default: `0` — auto (Qwen-Image-Edit: the source size, VRAM-clamped; MiniMax-H3: 640×384, or that area at the conditioning image's aspect ratio, rounded up to a multiple of 32; Wan: the model's native area at the input image's aspect ratio, 1280×704 for TI2V-5B and 832×480 otherwise). |
| `--video-frames <N>` | Video frame count, snapped to the model's temporal grid (`4k+1` for Wan; `17k+5` for MiniMax-H3 — 5, 22, 39, 56, 73, 90 …). Default: 33; 49 for Wan2.2-TI2V, 22 for MiniMax-H3. `1` generates a still image where the model supports it (use `--output out.png`). |
| `--fps <N>` | Playback frame rate of the saved MP4 (default: 16; 24 for Wan2.2-TI2V). Models trained at a fixed rate (MiniMax-H3, 24 fps) override any other value. |
| `--flow-shift <F>` | FlowMatch timestep shift (default: the model's official recipe — 5.0 for Wan 2.2, 12.0 for A14B T2V, 8.0/3.0/5.0 for Wan 2.1, 12.0 for MiniMax-H3). On models with a joint audio stream this shifts the video stream only. |
| `--sampler <name>` | Sampler: `unipc` (the official Wan sampler, default for Wan) or `euler`. A Wan-family knob — MiniMax-H3 runs its own flow-match schedule and does not read it. |
| `--negative-prompt <text>` | Negative prompt for classifier-free guidance (default: the model's official negative prompt). Unused at `--cfg 1.0`, where no negative pass runs — so it does nothing at all on MiniMax-H3, which only accepts `--cfg 1.0`. |
| _(step-distilled checkpoints)_ | Auto-detected from the DiT file name (`Turbo`, `distill`, `Lightning`, `lightx2v`, `FastWan`, `-dmd`, or an explicit `…-4steps-…` / `…8step…` for 1–16): the pipeline switches to that step count with guidance off, turning the official 50-step × CFG recipe's 100 DiT passes into 4. This is the single biggest speed lever anywhere in TensorSharp — see **[Video generation (Wan)](#video-generation-wan)** below. `--diffusion-steps` / `--cfg` override it. |
| `--cfg-cache-stride <N>` | Wan guidance cache: run the unconditional CFG pass on one step in `N` and reuse the cached guidance direction in between (default off — every step runs both passes). `2` ≈ 1.30x faster, `3` ≈ 1.43x; approximate, so leave it off when matching a reference sample matters. No effect at `--cfg 1.0`, and therefore none on MiniMax-H3. |
| `--video-vae <path>` | Override the resolved video VAE (`wan_2.1_vae.safetensors` / `Wan2.2_VAE.safetensors`; `minimax_h3_video_vae_fp16.safetensors` for MiniMax-H3). Env: `TS_VIDEO_VAE` (`TS_WAN_VAE` also honoured). |
| `--video-text-encoder <path>` | Override the resolved text-encoder GGUF (UMT5-XXL for Wan, Qwen3-VL-32B for MiniMax-H3). Also spelled `--video-te`. Env: `TS_VIDEO_TEXT_ENCODER` (`TS_WAN_TE` also honoured). |
| `--video-dit2 <path>` | Second diffusion expert on dual-expert models (Wan 2.2 A14B's high/low-noise partner). Auto-resolved by name when the pair is co-located. Env: `TS_VIDEO_DIT2` (`TS_WAN_DIT2` also honoured). |
| `--audio-vae <path>` | Audio VAE for models that generate an audio track jointly with the video (`minimax_h3_audio_vae_fp32.safetensors`). Without it such a model still produces video, just no audio. Env: `TS_VIDEO_AUDIO_VAE`. |
| `--video-mode <mode>` | How to read the images you pass, on models with more than one conditioning mode. Default: inferred from what you supply. MiniMax-H3 accepts `t2v` (text only), `i2v` (the image **is** the first frame and gets animated), `fl2v` (first and last frame) and `ref` (the images are identity/appearance references for a **new** scene). `i2v`/`fl2v` need the FL2VA checkpoint and `ref` needs Ref2VA — separate files, not settings, and asking for the wrong one fails with the name of the other. |
| `--end-image <file>` | Last-frame conditioning image, on models that accept one (MiniMax-H3 first/last-frame mode). Combined with `--image` the clip is steered to start and end on the two frames. |
| `--ref-image <file>` | Reference image for reference-conditioned models (MiniMax-H3 Ref2VA): the subject carries over while camera, background and composition come from the prompt. Repeatable up to 9; referred to in the prompt as `<Picture 1>`, `<Picture 2>`, … References are only ever scaled **down** and keep their own aspect ratio, so the output size still comes from `--width`/`--height`. |
| `--ref-video <path>` | Reference video clip — a video **file** or a directory of frames, either way resampled onto the model's own 24 fps and canvas. Repeatable; referred to as `<Video 1>`, `<Video 2>`, … |
| `--ref-video-audio <file>` | Soundtrack for the `--ref-video` at the **same position**: the first pairs with the first, and so on. Separate from `--ref-video` because a container's audio track is not readable through the frame decoder; omit it for a silent reference clip. WAV, MP3 or Ogg. |
| `--ref-audio <file>` | Reference audio clip. Repeatable; referred to as `<Audio 1>`, `<Audio 2>`, … Resampled to the audio VAE's 32 kHz stereo and truncated to the generated clip's duration. |
| `--no-audio` | Skip audio decoding on models that generate an audio track jointly with the video (MiniMax-H3), saving the audio VAE's time and memory. Ignored by video-only models. |
| _(renamed flags)_ | `--wan-vae`, `--wan-te` and `--wan-dit2` became `--video-vae`, `--video-text-encoder` and `--video-dit2` when video generation stopped being Wan-only. The old spellings are still accepted everywhere — on the CLI, on the server, and as config-file keys — so existing configs keep working unchanged. |
| `--tp <N>` | Multi-GPU degree — how many GPUs to spread the model over in a single process (default: `1`). Which of the two multi-GPU modes you get is the architecture's business, not yours: **tensor parallelism** (the weights split *inside* every layer) where it is implemented, and a **layer split** (whole layers per GPU — capacity, not speed) on Qwen 3.8 Flash Next (`qwen4exp`) and DeepSeek V4. GLM 5.x layer-splits when this flag is omitted; on GGML GPU backends the flag selects its native local/single-process TP path for GLM-5.2, GLM-5.3 and GLM-5.3-Flash alike (on GLM-5.3 that is an accepted mode rather than a validated configuration — the caches replicate per rank — and `--spec` is refused there whenever `--tp N>1`, so speculation engages on the default layer split). An architecture that supports neither says so on stderr and runs on one GPU. Requires `--backend cuda`, `ggml_cuda`, or `ggml_vulkan`. See [Tensor Parallelism & Distributed Inference](#tensor-parallelism--distributed-inference). |
| `--tp-node-id <N>` | This node's 0-based ID for multi-node (distributed) tensor parallelism. Requires `--tp-peers`. |
| `--tp-peers <list>` | Comma-separated `host:port` list of all nodes in the distributed TP cluster (e.g. `192.168.1.10:9500,192.168.1.11:9500`). Requires `--tp-node-id`. |
| `--test` | Run built-in tokenizer, ChatML-template, and Ollama-comparison tests |
| `--test-templates <dir>` | Validate hardcoded chat templates against GGUF Jinja2 templates for every *.gguf in `<dir>` |
| `--config <path>` | Read options from a JSON config file (command-line options override it). Supports `${variables}` and auto-downloading models via `{ "path": ..., "urls": [...] }`. Repeatable. See [Configuration file](#configuration-file-cli--server). |
| `--log-level <lvl>` | Console + file logger level: `trace`, `debug`, `info`, `warning`, `error`, `critical`, `off` |
| `--log-dir <path>` | Directory for the JSON-line file logger (default: `<binDir>/logs`) |
| `--log-file <0\|1>` | Disable (`0`) or enable (`1`) the file logger (default: enabled) |
| `--log-console <0\|1>` | Disable (`0`) or enable (`1`) the console logger (default: enabled) |

The CLI recognizes a small set of legacy projector filenames beside the model, but current repositories often use different names. Pass the downloaded file explicitly with `--mmproj` for reliable multimodal runs. `TensorSharp.Server` never auto-detects the projector.

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
| `/skills` | List the registered Agent Skills and which of them are active for this session |
| `/skill <name>` | Toggle one skill on or off for the session. Resets the conversation, as `/system` does — the skills block sits at the front of the prompt. |
| `/think on\|off` | Toggle thinking/reasoning mode for supported models |
| `/multiline on\|off` | Toggle multi-line input (terminate the message with a single `.` on its own line) |

Model and runtime:

| Command | Description |
|---|---|
| `/info`, `/status` | Show the loaded model, backend, architecture, context/vocab size, projector, conversation depth, and pending attachments |
| `/model <path>` | Load a different `.gguf` model on the current backend (resets the session) |
| `/backend <name>` | Reload the current model on a different backend: `cpu`, `cuda`, `mlx`, `ggml_cpu`, `ggml_metal`, `ggml_cuda`, or `ggml_vulkan` |
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
| `/text <path>`, `/file <path>`, `/txt <path>` | Inline up to the first 256 KiB of a UTF-8 text/markdown/csv/code file into the next prompt |
| `/clearattach` | Drop any pending image/audio/video/text attachments without sending a turn |

Quoted paths (single or double quotes) are accepted, so drag-and-drop from a file manager works on macOS. Multimodal commands require a multimodal projector to be loaded — pass `--mmproj` at startup or use `/mmproj <path>` from the REPL.

## Web Application

Run these commands from the repository root after building:

```bash
# Start the server with the exact hosted model
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ./models/model.gguf --backend ggml_metal

# Linux + NVIDIA GPU
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ./models/model.gguf --backend ggml_cuda

# Web research through model-authored code. Network access is deliberately a
# separate opt-in; --skills-allow-network would control bundled skill scripts instead.
./TensorSharp.Server --model ~/work/models/Qwen/Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf \
    --backend ggml_metal --port 5001 --skills-allow-exec --code-exec \
    --code-exec-allow-install --code-exec-allow-network --max-tokens 256000

# Multimodal models: host an explicit projector too
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ./models/model.gguf --mmproj ./models/mmproj.gguf --backend ggml_cuda

# MiniMax-H3: video and its 32 kHz stereo soundtrack generated together. Size,
# steps and frame count are startup flags because the Web UI sends no numbers of
# its own; 640x384 is the documented starting point. Each run writes an .mp4 plus
# a sidecar .wav. See "Video generation with audio (MiniMax-H3)" below.
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model ./models/minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_cuda \
    --video-width 640 --video-height 384 --video-steps 20 --video-frames 22

# The same host from a shipped config (config/minimax-h3-ref2va.json swaps in the
# reference checkpoint; only the denoiser differs, so nothing re-downloads).
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/minimax-h3-fl2va.json

# Wan video generation, video only: use 121 frames at 24 fps (about five seconds)
# whenever the Web UI or an API request does not supply its own frames / fps value.
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ./models/Wan2.2-TI2V-5B.gguf --backend ggml_cuda \
    --video-frames 121 --fps 24

# 121 frames at the TI2V-5B native area is 27k DiT tokens, and self-attention is
# quadratic in that, so a full 50-step run on the BASE checkpoint is hours on a
# laptop-class GPU (measured: ~3 h 30 m on an M5 Pro). Point --model at a
# step-distilled checkpoint instead and the identical request takes 17 m 30 s --
# nothing else changes. See "Video generation (Wan)" below.
# The server logs a per-pass timing plus a running ETA and heartbeats every 30 s,
# and the Web UI shows both; docs/models/wan.md has the full cost table.

# Configure server-wide default sampling parameters
# (used whenever a request does not override the value itself)
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model ./models/model.gguf --backend ggml_metal \
    --temperature 0.7 --top-p 0.9 --top-k 40 --repeat-penalty 1.1 \
    --presence-penalty 0.0 --frequency-penalty 0.0 --seed 42 \
    --stop "</s>" --stop "<|endoftext|>"

# Read all of the above from a reusable JSON file (auto-downloads the model on
# first run). See the Configuration file section and config/ for examples.
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/server-basic.json
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --config config/server-basic.json --backend ggml_cpu
```

Open `http://localhost:5000` in your browser — the root URL serves the chat UI (`GET /health` is the liveness endpoint). The web interface supports:

- Multi-turn chat conversations
- Per-tab chat sessions: each browser tab owns its own tracked conversation history; KV blocks are owned by the inference engine
- A single hosted GGUF selected explicitly with `--model`
- An explicit hosted multimodal projector via `--mmproj` when needed
- Full text and PDF document uploads, plus image, video, and audio uploads for multimodal inference (up to 500 MB)
- Thinking/reasoning mode toggle
- Tool calling with function definitions
- Agent Skills: a picker for the skills the server has registered, and a live trace of every skill file the model reads while answering (hidden when the server reports no skills)
- Streaming token generation via Server-Sent Events
- DiffusionGemma denoising previews when a `diffusion-gemma` GGUF is hosted (the UI replaces the whole assistant message on each denoising step, then emits the final answer)
- Backward-compatible queue-status events (the engine itself handles concurrency)
- Message editing and deletion with regeneration from any point in the conversation
- Free scrolling: scroll up to read earlier replies while new tokens stream in; the chat auto-scrolls again as soon as the user scrolls back to the bottom

Use `--model` to choose the hosted GGUF file and `--mmproj` to choose the hosted projector. `TensorSharp.Server` no longer scans a `MODEL_DIR`.

**Server command-line options:**

Running `TensorSharp.Server` with no arguments prints the full parameter reference (description, default, and an example per option) and exits; `--help` does the same. Pass `--model` at startup for inference. Other options can start a model-less status process, but `/api/models/load` cannot select a GGUF that was not supplied at startup.

| Option | Description |
|---|---|
| `--model <path>` | GGUF file to host (required for inference; when other options are passed without it, the server starts but `/api/models/load` will report no hosted model) |
| `--mmproj <path>` | Multimodal projector GGUF (resolved relative to the model directory when only a filename is given; pass `none` to disable). Requires `--model`. |
| `--backend <type>` | Default compute backend: `cpu`, `cuda`, `mlx`, `ggml_cpu`, `ggml_metal`, `ggml_cuda`, or `ggml_vulkan` |
| `--tp <N>` | Multi-GPU degree — how many local GPUs to spread the hosted model over (default: `1`). Tensor parallelism where the architecture implements it; a layer split (whole layers per GPU — capacity, not speed) on Qwen 3.8 Flash Next (`qwen4exp`) and DeepSeek V4. GLM 5.x layer-splits when this flag is omitted; on GGML GPU backends the flag selects its native local/single-process TP path for GLM-5.2, GLM-5.3 and GLM-5.3-Flash alike (on GLM-5.3 an accepted mode rather than a validated configuration — the caches replicate per rank, so `--tp` multiplies the KV footprint). Requires `--backend cuda`, `ggml_cuda`, or `ggml_vulkan`. Env: `TENSORSHARP_TP_DEGREE`. See [Tensor Parallelism & Distributed Inference](#tensor-parallelism--distributed-inference). |
| `--tp-node-id <N>` | This node's 0-based ID for multi-node (distributed) tensor parallelism. The server can only be node `0` (the driver that serves HTTP); start worker nodes with `TensorSharp.Cli`. Requires `--tp-peers`. Env: `TENSORSHARP_TP_NODE_ID`. |
| `--tp-peers <list>` | Comma-separated `host:port` list of all nodes in the distributed TP cluster, ordered by node ID (e.g. `192.168.1.10:9500,192.168.1.11:9500`). Requires `--tp-node-id`. Env: `TENSORSHARP_TP_PEERS`. |
| `--gpu-device <N>` | Vulkan device index for the `ggml_vulkan` backend on multi-GPU hosts (e.g. an integrated Intel GPU next to a discrete NVIDIA one). Defaults to device 0; use `--list-gpus` to see the indices. Also settable via the `TS_GGML_VULKAN_DEVICE` env var. |
| `--list-gpus` | List the Vulkan devices ggml-vulkan can see (index + adapter name) and exit |
| `--help` | Print the parameter reference (also shown when the server is started with no arguments) and exit |
| `--max-tokens <N>` | Maximum tokens to generate: fills in when a request omits its own limit, and caps a request that asks for more. Applies to every endpoint (Web UI, `/api/chat`, `/api/generate`, `/v1/chat/completions`, `/v1/responses`). Default: `20000`, which is a plain default and does not cap. Env: `MAX_TOKENS`. |
| `--skills-dir <path>` | Directory to scan for Agent Skills. Repeatable; scanned in the order given, up to three levels deep. Without it, a `skills` directory beside the binary is used and created if missing, which is also where `POST /api/skills` uploads land. A path that does not exist is a startup error. Env: `TS_SKILLS_DIR` (a path-separator-separated list). |
| `--skill <name>` | Make a skill active for every request that does not name its own. Repeatable. A request's `skills` array overrides it. |
| `--list-skills` | Print the skill registry — names, descriptions, files, warnings and load errors — and exit. |
| `--no-skills` | Turn Agent Skills off: `/v1/skills` and `/api/skills` report the feature as disabled and a request's `skills` field is inert. Env: `TS_NO_SKILLS` (anything but `0` counts as on). |
| `--skills-no-discovery` | Do not advertise unselected skills to the model, so each request sees exactly the skills it named. Per request, `"skills_discovery": false` does the same. |
| `--skills-allow-exec` | Allow `skills_run`. **Off by default. On a shared server this is a remote code execution primitive** — a skill is content somebody uploaded, and the decision to run one of its scripts is made by a model reading that same person's Markdown. Scripts use only known interpreters without a shell, run from the current Web session or private OpenAI/Ollama request workspace (or per-call scratch without code execution), and see the skill itself read-only. The default required sandbox refuses execution when safe OS confinement is unavailable. Env: `TS_SKILLS_ALLOW_EXEC` (anything but `0` counts as on). |
| `--skills-max-rounds <n>` | How many times the model may fetch skill content - or run, read and fix code - before it must answer, 1-64 (default: `8`, or `24` when `--code-exec` is on; a value set here is used as given). Each round is a full generation. Env: `TS_SKILLS_MAX_ROUNDS`. |
| `--skills-sandbox <off\|preferred\|required>` | How hard to insist on OS isolation for a skill's scripts. `required` (the default) refuses to run them on a host with no safe sandbox rather than running them unconfined; `preferred` explicitly accepts weaker isolation; `off` keeps only the runner's interpreter, environment, time and output limits. macOS uses `sandbox-exec`; Linux requires `bwrap` 0.12.0 or newer. Windows Job Objects bound only the process tree, so `required` refuses there and `preferred` is the explicit weaker-isolation opt-in. Env: `TS_SKILLS_SANDBOX`. |
| `--skills-allow-network` | Let a sandboxed skill script reach the network. Denied by default. Env: `TS_SKILLS_ALLOW_NETWORK`. |
| `--code-exec` | Offer the model the `shell`, `read_file`, `edit_file`, `write_file` and `apply_patch` tools. `shell` runs a real command line and returns its exit code plus merged stdout/stderr; the file tools make exact host-side edits. Web UI keeps one workspace per chat session. Every OpenAI Chat, OpenAI Responses, and Ollama HTTP request gets a private workspace across its internal tool rounds and that workspace is deleted after the response. Commands run in a required OS sandbox on macOS/Linux, and the server refuses to run them if confinement is unavailable unless the operator explicitly sets `--code-exec-unconfined`; Windows always needs that escape hatch. Network is denied independently by default and enabled only by `--code-exec-allow-network`. All built-in tools are answered in process rather than returned to the API client. Off by default. Env: `TS_CODE_EXEC` (anything but `0` counts as on). |
| `--code-exec-allow-install` | Let the model install pip/npm packages into the active Web-session or HTTP-request workspace; requires `--code-exec`. This permission does **not** give model-authored commands a socket. TensorSharp reads recognised install commands, validates package names, performs wheels-only / scripts-disabled installs with a host-built argument vector, and substitutes `true` or `false` back into the command line. Later generated code remains offline unless `--code-exec-allow-network` is separately set. Env: `TS_CODE_EXEC_ALLOW_INSTALL`. |
| `--code-exec-allow-network` | Give every model-authored command unrestricted host IP-network access: generated code can resolve DNS, fetch URLs, follow redirects, call remote APIs, reach LAN/loopback services and open IP listening sockets. **Off by default** and requires `--code-exec`. Write and home-read confinement remain active on macOS and Linux. Linux additionally bounds descendants with a PID namespace. On macOS, children inherit Seatbelt and ordinary process groups are stopped, but a deliberately detached child can outlive the request; every result reports that gap. macOS denies common `/private/tmp/com.apple.launchd*` pathname sockets while permitting runtime-required Mach lookup and the exact mDNSResponder pathname socket required for DNS, and Linux hides common `/run` endpoints, but this is not a complete local Unix-IPC boundary: macOS retains shared-temporary-directory Unix IPC for compatibility, and Linux's host network namespace may expose abstract sockets and pathname sockets outside `/run`. Other host-readable files and IP services may therefore be reached and exfiltrated; remote prompt injection and untrusted downloads are additional risks. Credential-free host HTTP/SOCKS proxy settings are passed only in this mode. Configured custom-CA bundles up to 16 MiB are read once; only validated public certificates are copied into a read-only per-session snapshot, so the host source path and adjacent data are not exposed. Authenticated proxies need a credential-free host-side forwarder. Package/install-domain allow-lists constrain only the recognised host installer; unrestricted generated code can fetch or execute artifacts directly. This is independent of `--code-exec-allow-install` and `--skills-allow-network`. On Windows, `--code-exec-unconfined` is still required. Env: `TS_CODE_EXEC_ALLOW_NETWORK` (anything but `0` counts as on). |
| `--code-exec-packages <list>` | Restrict installs to these package names, comma-separated; anything else is refused and the model is told which names are allowed (default: empty, meaning any package — and at most 16 packages in one install either way). Matching is on the bare name, so a version the model pins (`numpy==2.1.0`) still matches the entry `numpy`. It was retired when the tool surface became a shell, because a model typing its own `pip install` could spell the request in ways a name list could not see; it is back because the **host** performs every install again — it reads the names out of the model's command rather than running that command, and builds the installer's argument vector itself, so the list applies to recognised requests however they were spelled (`pip`, `pip3`, `python -m pip`, a requirements file). Only meaningful with `--code-exec-allow-install`. With `--code-exec-allow-network`, this list is not a security boundary: generated code can fetch or execute artifacts without using the host installer. |
| `--code-exec-install-index <url>` | Point host-performed installs at an operator-selected package index. Model-authored source-changing options remain refused. The index host is admitted through the install egress allow-list automatically; list any separate download host with `--code-exec-install-domains`. Requires `--code-exec-allow-install`. Default: unset. Env: `TS_CODE_EXEC_INSTALL_INDEX`. |
| `--code-exec-install-domains <list>` | Hosts the **host-performed package install** may reach, comma-separated — exact names or `*.suffix` wildcards (default: `pypi.org,files.pythonhosted.org,registry.npmjs.org`; empty disables pinning). This does not restrict ordinary generated commands after `--code-exec-allow-network` grants them unrestricted host-network access. Env: `TS_CODE_EXEC_INSTALL_DOMAINS`. |
| `--code-exec-timeout <seconds>` | How long one command may run before it is killed (default `120`, up from the 30 s the program runner used, because a shell command is also how packages are installed and how a build or a test suite is run). A call may ask for less or more through `timeout_ms`, up to 10 minutes; a command that runs over is stopped and the model still gets everything it printed first, because a timeout that discards the output only makes it run the command again blind. |
| `--code-exec-temperature <number>` | Optional sampling temperature for coding turns, from `0` to `2`; explicit request/operator sampling still wins and a negative value clears the override. Coding turns remove the built-in chat repetition penalty either way. Default: unset. |
| `--code-exec-shell <path\|name>` | Which shell to run commands through, when the host's own choice is wrong or absent. Default: `bash`, then `sh`, on macOS and Linux; PowerShell 7 (`pwsh`), then Windows PowerShell, on Windows — where a bare `bash` on PATH is refused on purpose, because it is the WSL launcher and running through it would put the command inside a Linux VM while the job object holds only the launcher on the Windows side. Point this at a real bash (Git Bash, MSYS2) to use one there. The model is told which dialect it is typing into, so this also changes the examples in its tool description. |
| `--code-exec-max-output <bytes>` | How much of one command's output is kept and shown to the model (default `32768`). What does not fit is dropped from the **middle**, keeping the head and the tail: the end of a build or a test run is where the failure is, and head-only truncation discards exactly the part that was wanted. |
| `--code-exec-unconfined` | Run model-authored commands even where the OS cannot confine them. The server and CLI both accept this explicit escape hatch. Windows requires it and consequently leaves filesystem and network access unconfined; do not enable it on a server reachable by users you do not trust. |

**How the shell is used, and what it may reach.** Everything the model does
with files and code goes through that one command line — write, run, grep, read
the traceback, fix — which is why `--skills-max-rounds` rises from 8 to 24 when
this is on. Web UI keeps one filesystem workspace per chat session. Each OpenAI
Chat, OpenAI Responses, or Ollama HTTP request instead gets a private workspace
across its internal tool rounds, and the server deletes it after the response.
Files and host-installed packages remain available within that lifetime, but
each shell call is a fresh confined process; do not rely on virtualenv activation,
PATH changes, or a resident shell surviving between calls. The network rule
belongs to the host, not to the model: commands are offline by default, while
`--code-exec-allow-network` / `TS_CODE_EXEC_ALLOW_NETWORK` gives every generated
command unrestricted host IP-network access, including LAN/loopback services and
IP listening sockets. Write and home-read confinement remains active on macOS
and Linux. Linux additionally bounds descendants with a PID namespace. On macOS,
children inherit Seatbelt and ordinary process groups are stopped, but a deliberately
detached child can outlive the request; every result reports that gap. macOS denies
common `/private/tmp/com.apple.launchd*` pathname sockets while permitting runtime-required Mach lookup and the exact mDNSResponder pathname socket required for DNS, and Linux hides common
`/run` endpoints, but this is not a complete local Unix-IPC boundary: macOS
retains shared-temporary-directory Unix IPC for compatibility, and Linux's host
network namespace may expose abstract sockets and pathname sockets outside
`/run`. Generated code can send other host-readable data away.
`--skills-allow-network` is separate and applies only to `skills_run`.
Package-install permission also does not grant shell network access;
the host performs recognised installs itself — it reads the tool and package
names out of each recognised install, runs the installer with an argument vector
it built, and substitutes
that command out of the line — `true` when it worked, `false` when it did not —
so the operators the model wrote around it keep exactly their meaning: nothing
after a failed install's `&&` runs, and its `||` fallback does. The failure is
reported either way. Giving the install command its own socket was the earlier
design, and it had two holes
that could not both be closed: the command line is the model's, so
`--index-url` chose the index, and a socket belongs to the whole line, so
anything sharing it with the install shared its reach — which on a host whose
sandbox cannot pin egress to a proxy was the whole internet. The retired
`--code-exec-languages` is refused by name at startup and has no replacement to
point at, because a shell reaches every interpreter on PATH, so a deployment
script carried over from the old surface fails with that in the message instead
of quietly losing a setting.

| Option | Description |
|---|---|
| `--video-width <px>` | Default output width for video generation when a Web UI or API request omits `width`; alias `--width`. **The main quality lever on the server**, because the Web UI sends no size of its own — without it every clip comes out at the model default. `640` (with `--video-height 384`) is the documented starting point for MiniMax-H3. Rounded up to the model's grid. |
| `--video-height <px>` | Default output height when a request omits `height`; alias `--height`. If only one of the two is given, MiniMax-H3 takes the other from the conditioning image's aspect ratio, which is what stops a 4:3 photo being stretched into a 16:9 frame. |
| `--video-steps <N>` | Default denoising steps when a request omits `steps` — the quality/time trade-off after resolution. MiniMax-H3's own default is `20`; `4`–`8` is the fast operating point, `16`–`24` is visibly cleaner, past ~30 gains little. |
| `--video-mode <mode>` | Default conditioning mode when a request omits `videoMode`: `t2v` (text only), `i2v` (the image is the first frame and is animated), `fl2v` (first **and** last frame pinned) or `ref` (the images are identity/appearance references for a new scene). Omit it and each request's mode is inferred from what it supplies, which is usually what you want; pin it for a deployment that only offers one. |
| `--video-frames <N>` | Default output frame count for video generation when a Web UI or API request omits `frames`. Snapped to the model's temporal grid — `4k+1` for Wan, where `121` frames at 24 fps is about five seconds; `17k+5` for MiniMax-H3. Without this flag, the model default is `33` (`49` for Wan 2.2 TI2V, `22` for MiniMax-H3). An explicit request `frames` value overrides this default. |
| `--fps <N>` | Default playback frame rate for the generated MP4 when a Web UI or API request omits `fps`. Without this flag, the model default is `16` fps (`24` for Wan 2.2 TI2V). An explicit request `fps` value overrides this default; changing only FPS changes playback speed rather than the generated frame count. Models trained at a fixed rate (MiniMax-H3, 24 fps) override any other value. |
| `--video-vae <path>` | Override the resolved video VAE (`wan_2.1_vae.safetensors` / `Wan2.2_VAE.safetensors`; `minimax_h3_video_vae_fp16.safetensors` for MiniMax-H3). Default: a same-directory scan next to the DiT, `VAE/` subfolders included. Env: `TS_VIDEO_VAE`; `--wan-vae` still accepted. |
| `--video-text-encoder <path>` | Override the resolved text-encoder GGUF (UMT5-XXL for Wan, Qwen3-VL-32B for MiniMax-H3). Also spelled `--video-te`. Env: `TS_VIDEO_TEXT_ENCODER`; `--wan-te` still accepted. |
| `--video-dit2 <path>` | Second diffusion expert on dual-expert models (Wan 2.2 A14B's high/low-noise partner of `--model`). Auto-resolved by name when the pair is co-located. Env: `TS_VIDEO_DIT2`; `--wan-dit2` still accepted. |
| `--audio-vae <path>` | Audio VAE for models that generate an audio track jointly with the video (`minimax_h3_audio_vae_fp32.safetensors`). Without it such a model still runs and produces video, just no audio. Env: `TS_VIDEO_AUDIO_VAE`. |
| `--temperature <f>` | Sampling temperature (`0` = greedy) |
| `--top-k <N>` | Top-K filtering (`0` = disabled) |
| `--top-p <f>` | Nucleus sampling threshold (`1.0` = disabled) |
| `--min-p <f>` | Min-p filtering (`0` = disabled) |
| `--repeat-penalty <f>` | Repetition penalty (`1.0` = none) |
| `--penalty-last-n <N>` | Tokens of history the repeat / presence / frequency penalties look back over. `0` disables them, `-1` uses the whole history (default: 64) |
| `--presence-penalty <f>` | Presence penalty (`0` = disabled) |
| `--frequency-penalty <f>` | Frequency penalty (`0` = disabled) |
| `--seed <N>` | Random seed (`-1` = non-deterministic) |
| `--stop <string>` | Stop sequence (can be repeated). Under `--sampling-precedence config` a per-request `stop`/`stop_sequences` list is merged with these; under `request` it replaces them. |
| `--sampling-precedence <config\|request>` | Who wins when a request also carries a sampling parameter you set above. `config` (default) keeps your value — clients such as VS Code Copilot Chat hardcode `temperature`/`top_p` into every request and would otherwise silently override your configuration; parameters you did **not** set still come from the request. `request` restores client-always-wins. Env: `TENSORSHARP_SAMPLING_PRECEDENCE`. |
| `--n-cpu-moe <N>` / `-ncmoe <N>` | Keep the routed MoE weights of the first N layers in system RAM and run their FFN on the CPU (see **Mixture-of-Experts CPU offload** above). `all` offloads every layer. Default: 0 on every architecture, DeepSeek V4 and GLM 5.x included; a model that does not fit is refused at load with the number of layers that would make it fit. Env: `TS_N_CPU_MOE`. |
| `--cpu-moe` / `-cmoe` | Shorthand for `--n-cpu-moe all`. Default: off. Env: `TS_CPU_MOE`. |
| `--cpu-moe-threads <N>` | Worker threads for the host-side expert matmul. Default: half the usable CPU parallelism (`hardware_concurrency` clamped by the affinity mask and the cgroup CPU quota) on hosts with more than 8 cores. The server needs the other half for Kestrel, the scheduler and the accelerator submission threads; sizing this near the quota collapses throughput rather than degrading (20.7 tok/s at 64 threads vs 8.2 at 71 on a 95-CPU quota). Env: `TS_CPU_MOE_THREADS`. |
| `--kv-cache-dtype <type>` | KV cache precision for the hosted model: `f32`, `f16`, `q8_0`, or `q4_0` (quantized caches trade small numerical drift for memory; see the CLI table above for the tier trade-offs). Default: auto — the backend/model pick. Env: `KV_CACHE_DTYPE`. |
| `--continuous-batching` / `--no-continuous-batching` | Enable (default) or disable iteration-level paged-batching. When enabled the server admits / preempts sequences mid-batch and packs them into one forward pass on models that implement `IBatchedPagedModel`. `--no-continuous-batching` falls back to per-sequence KV-swap for every model. Alias: `--paged-batching` / `--no-paged-batching`. |
| `--no-webui` | Do not serve the bundled web UI; `GET /` answers the plain liveness text instead. Every HTTP API endpoint, `/uploads` included, stays up. Env: `TS_NO_WEBUI` |
| `--no-prefix-cache` | Do not prepare the prompt every conversation shares before serving, and do not keep it between launches. By default the server forwards that prompt once at startup and saves the result, so the first message of a process costs the same as any other (measured 21.8s → 0.7s on an agent configuration) |
| `--prefill-chunk-size <N>` | Maximum prefill tokens per request in a mixed prefill+decode step, so active streams get frequent GPU turns (default: `256`). Prefill-only batches still divide and consume the full device token budget. Env: `TS_SCHED_PREFILL_CHUNK`. |
| `--spec` / `--no-spec` | Enable speculative decoding (default off). `--spec` is the explicit opt-in for drafters embedded in the trunk checkpoint (Qwen 3.6's, GLM 5.2's and GLM-5.3's NextN blocks), because loading them pages extra weights into VRAM; a drafter that ships as its own GGUF is enabled by `--draft-model` alone, and an explicit `--no-spec` vetoes either. Engages for solo (non-concurrent) sequences: the draft head proposes up to `--spec-draft` tokens per step and the trunk verifies them in one batched forward, with the request's own sampler (penalties included) driving both drafting and verification, so output matches standard decode. Engaged automatically only where profitable: Qwen 3.6 reports its embedded NextN block profitable on every backend, while Gemma 4's separate draft head engages on the ggml backends and on the direct `cuda` backend only. CPU / GGML CPU / MLX serve standard decode. Env: `TS_SPEC` (legacy `TS_MTP_SPEC`). |
| `--spec-type <name>` | Speculation algorithm: `auto` (default) / `draft-head` / `block` / `ngram`. `ngram` needs no trained weights and works on every model — it drafts by finding where the last few tokens occurred earlier in the context and proposing what followed, so it is strong wherever the answer quotes its input. Env: `TS_SPEC_TYPE`. |
| `--spec-draft <N>` | Maximum tokens drafted per speculative step (default `8`; a block drafter clamps it to its trained block size, which is also its default there). On Qwen 3.5 under the GGML backends, keep a typed value at 7 or lower — nine verify rows are a known correctness bug. Env: `TS_SPEC_DRAFT` (or `TS_MTP_DRAFT`). |
| `--spec-pmin <f>` | Draft-confidence gate in `[0, 1]`; drafting stops at the first token below it, and `0` means never gate. Default per algorithm — `0.15` for a per-token draft head (top-1 probability over its top-10 logits), `0.35` for a block drafter (the CUMULATIVE prefix probability, so far stricter), `0` for n-gram. Env: `TS_SPEC_PMIN` (or `TS_MTP_PMIN`). |
| `--draft-model <path>` | Speculative-decoding draft model, for every drafter that ships as its own file: DeepSeek V4's DSpark support GGUF (see [DeepSeek V4](docs/models/deepseek4.md#dspark-speculative-decoding)), Muse-Glimmer's DFlash and Qwen 3.8's DFlash2 block drafters (see [Muse-Glimmer](docs/models/muse-glimmer.md#3-dflash-speculative-decoding), env `TS_MUSE_GLIMMER_DFLASH`), and Gemma 4's `gemma4-assistant` per-token head. The file's own `general.architecture` decides how it loads — the operator never picks a mechanism — and naming a file here enables speculation by itself, with an explicit `--no-spec` as the veto. The draft's hidden size must match the target (e.g. pair the 12B target with its 12B draft, not the 26B-A4B draft); a mismatched or incomplete draft fails fast at startup with a remediation hint. Qwen 3.6, GLM 5.2 and GLM-5.3 embed their NextN block in the trunk GGUF and need no such flag — they take `--spec` instead. A block drafter drafts a whole block per step and the trunk verifies it in one batched forward. Every emitted token is still drawn from a trunk row — with argmax under a greedy config, with the run's own sampler otherwise — so the output stream is unchanged either way. Block drafting engages with `--backend cuda` or `--backend ggml_cuda` (on the CLI, on every single-sequence path — `--input`, `--multi-turn-jsonl` and `--interactive`). One caveat under a penalized sampler: a block drafter proposes its whole block in one pass, so the repetition/presence/frequency penalties that verification applies are not applied to the proposal, and acceptance falls as the penalized history grows. A per-token head does not have that problem — its drafts are penalized with the same history. Env: `TS_SPEC_DRAFT_MODEL` (legacy `TS_MTP_DRAFT_MODEL`), `TS_DSV4_DSPARK`. |
| `--paged-kv` / `--no-paged-kv` | Legacy compatibility flags for the removed per-session paged-KV manager. Current server KV state is engine-owned; use continuous-batching / `TS_SCHED_*` knobs for the engine. Aliases: `--paged-kv-cache` / `--no-paged-kv-cache`. |
| `--paged-kv-block-size <N>` | Legacy standalone paged-KV block size. The current server engine uses `TS_SCHED_BLOCK_SIZE`. |
| `--paged-kv-ram-mb <N>` | Legacy standalone paged-KV RAM-tier cap. |
| `--paged-kv-ssd-dir <dir>` | Legacy standalone paged-KV SSD cold-tier directory. |
| `--paged-kv-ssd-mb <N>` | Legacy standalone paged-KV SSD cap. |
| `--paged-kv-quant-bits <0\|4\|8>` | Legacy standalone paged-KV block quantization accepted by the server (`4`/`8` = symmetric). The runtime env var also accepts `2` for affine min+scale, and the CLI accepts `0\|2\|4\|8`. |
| `--redis-url <url>` | Redis connection string enabling both the shared KV cache tier and the Responses API store (e.g. `localhost:6379`). Sets both `TS_KV_CACHE_REDIS_URL` and `TS_RESPONSES_STORE_REDIS_URL`. |
| `--paged-kv-redis-url <url>` | Redis connection string for the shared KV cache tier only (e.g. `localhost:6379`). Env: `TS_KV_CACHE_REDIS_URL`. |
| `--paged-kv-redis-ttl <min>` | TTL in minutes for Redis KV cache entries; `0` = no TTL (default: `1440`, i.e. 24 hours). Env: `TS_KV_CACHE_REDIS_TTL_MINUTES`. |

Per-request fields in the chat / generate JSON payloads (e.g. `temperature`,
`top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`,
`frequency_penalty`, `seed`, `stop`/`stop_sequences`) fill in every parameter
you did **not** configure above. For a parameter you *did* configure, the
default `--sampling-precedence config` keeps your value and ignores the
request's — many chat clients hardcode `temperature`/`top_p` into every call
with no way for the end user to change them, so an unconfigured server-side
value is the only one a request can move. Pass `--sampling-precedence request`
for the inverse (clients always win), which is what versions before this flag
did unconditionally.

**Runtime environment variables:**

| Variable | Description |
|---|---|
| `BACKEND` | Default compute backend (`cpu`, `cuda`, `mlx`, `ggml_cpu`, `ggml_metal`, `ggml_cuda`, or `ggml_vulkan`), used when `--backend` is not passed (default: `ggml_metal` on macOS, `ggml_cpu` elsewhere) |
| `MAX_TOKENS` | Maximum generation length when `--max-tokens` is not passed: fills in when a request omits its own limit and caps a request that asks for more (default: `20000`, which is a plain default and does not cap) |
| `MAX_CONTEXT` | Context window to allocate, overriding the length the GGUF advertises. Set, it is a **hard limit**: honoured when the caches plus one full `n_ubatch` graph fit, refused with the numbers when they do not. Left unset, the advertised length is a **ceiling** — after the weights load, the runtime asks the devices how much VRAM is actually free, sizes the context to fit, and logs what it picked. GLM-5.2 advertises 1,048,576 tokens (~93 GiB of KV); on 3x RTX PRO 6000 the pick is 342,272 on the layer split, 91,136 with `--tp 3`, and 646,400 with `--n-cpu-moe 30` |
| `VIDEO_SAMPLE_FPS` | Frames sampled per second from an **input video prompt** for multimodal understanding; time-based extraction (default: `1`). This is unrelated to the video-generation output setting `--fps`. |
| `VIDEO_MAX_FRAMES` | Optional upper bound on frames extracted from an **input video prompt** (evenly down-sampled); unset/`0` means no cap (default: no cap). This is unrelated to the video-generation output setting `--video-frames`. |
| `PORT` / `HOST` | Listen port / bind interface when `--port` / `--host` are not passed (defaults: `5000`, `0.0.0.0`) |
| `ASPNETCORE_URLS` | Full listen URL(s) when none of `--port`, `--host`, `--urls`, `PORT`, or `HOST` is set |
| `TENSORSHARP_TEMPERATURE` | Sampling temperature when `--temperature` is not passed. Counts as operator-configured, so it also outranks the request body under the default `--sampling-precedence config` |
| `TENSORSHARP_TOP_K` | Top-K when `--top-k` is not passed (same precedence rule as `TENSORSHARP_TEMPERATURE`) |
| `TENSORSHARP_TOP_P` | Top-P when `--top-p` is not passed (same precedence rule) |
| `TENSORSHARP_MIN_P` | Min-P when `--min-p` is not passed (same precedence rule) |
| `TENSORSHARP_REPEAT_PENALTY` | Repetition penalty when `--repeat-penalty` is not passed (same precedence rule) |
| `TENSORSHARP_PRESENCE_PENALTY` | Presence penalty when `--presence-penalty` is not passed (same precedence rule) |
| `TENSORSHARP_FREQUENCY_PENALTY` | Frequency penalty when `--frequency-penalty` is not passed (same precedence rule) |
| `TENSORSHARP_SEED` | Random seed when `--seed` is not passed (same precedence rule) |
| `TENSORSHARP_SAMPLING_PRECEDENCE` | `config` (default) or `request`: whether server-configured sampling parameters outrank the ones a client sends. `--sampling-precedence` overrides it |
| `TENSORSHARP_LOG_LEVEL` | Minimum log level for both console and file loggers: `Trace`, `Debug`, `Information`, `Warning`, `Error`, `Critical` (default: `Information`). Also honored by `TensorSharp.Cli`. |
| `TENSORSHARP_LOG_DIR` | Directory the JSON-line file logger writes to (default: `<binDir>/logs`). Also honored by `TensorSharp.Cli`. |
| `TENSORSHARP_LOG_FILE` | Set to `0` to disable the file logger and keep only the console output (default: enabled). Also honored by `TensorSharp.Cli`. |
| `TENSORSHARP_TP_DEGREE` | Multi-GPU degree — number of local GPUs to spread the model over (default: `1`). Fallback in `ModelBase.Create` when no `--tp` flag is passed; both `TensorSharp.Cli` and `TensorSharp.Server` expose it as `--tp <N>`. Requires `--backend cuda`, `ggml_cuda`, or `ggml_vulkan`. On the architectures that run a layer split instead of tensor parallelism (`qwen4exp`, DeepSeek V4) it is a device count, not a shard count. |
| `TENSORSHARP_TP_DEVICES` | GPU ordinals the TP ranks map to, comma-separated (e.g. `0,2`; default `0..tp-1`). Used by TP on the GGML backends. |
| `TS_Q4E_LAYER_SPLIT` | Explicit per-GPU layer counts for the Qwen 3.8 Flash Next (`qwen4exp`) multi-GPU layer split, comma-separated (e.g. `20,28`), replacing the automatic VRAM balance. Throws rather than silently ignoring a value it cannot honour. |
| `TENSORSHARP_TP_NODE_ID` | This node's 0-based ID for multi-node distributed tensor parallelism. Must be set together with `TENSORSHARP_TP_PEERS`. |
| `TENSORSHARP_TP_PEERS` | Comma-separated `host:port` list of all nodes in the distributed TP cluster (e.g. `192.168.1.10:9500,192.168.1.11:9500`). Must be set together with `TENSORSHARP_TP_NODE_ID`. |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS` | How long each node keeps retrying outbound connections to its peers before giving up (default: `120`). Raise it when nodes are started far apart by hand or by a slow orchestrator. |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS` | Per-receive timeout for a blocking read from a peer (default: `300`). A stalled peer fails the collective instead of hanging on the OS TCP keepalive (often 2+ hours). |
| `TENSORSHARP_TP_DISABLE_P2P` | Set to `1` to force every cross-GPU transfer through host staging instead of CUDA peer-to-peer DMA. Slower, but reproduces the code path taken by hardware without peer access (A16 vGPU profiles, some consumer cards) and isolates P2P-specific defects. |
| `TENSORSHARP_TP_HOST_ALLREDUCE` | Set to `1` to run the local AllReduce through host memory (device→host, sum, host→device) instead of the device-to-device P2P path. Diagnostic fallback that mirrors the known-good multi-node reduce. |
| `TS_KV_CACHE_REDIS_URL` | Redis connection string for the shared KV cache tier (e.g. `localhost:6379`). When set, KV cache blocks are persisted to Redis for cross-session reuse. CLI: `--redis-url` or `--paged-kv-redis-url`. |
| `TS_KV_CACHE_REDIS_TTL_MINUTES` | TTL in minutes for Redis KV cache entries; `0` = no TTL (default: `1440`). CLI: `--paged-kv-redis-ttl`. |
| `TS_RESPONSES_STORE_REDIS_URL` | Redis connection string for the OpenAI Responses API store. When set, `RedisResponsesStore` replaces the in-memory store. CLI: `--redis-url`. |
| `DIFFUSION_STEPS` | Server-side DiffusionGemma denoising steps per block (default: `48`; CLI equivalent is `--diffusion-steps`) |
| `DIFFUSION_MAX_BATCH` | Maximum concurrent DiffusionGemma requests batched by the Web UI diffusion scheduler (default: `2`) |

**Paged KV cache & continuous-batching tunables (read at process / model start)**

These can be set with either the `--paged-kv*` / `--continuous-batching` CLI flags (which translate to the env vars below) or directly via the environment:

| Variable | Description |
|---|---|
| `TS_KV_PAGED_CACHE` | Legacy compatibility switch for the standalone `PagedKvCacheManager`; current `TensorSharp.Server` request KV state is engine-owned. The CLI shortcuts are `--paged-kv` / `--no-paged-kv`. |
| `TS_KV_BLOCK_SIZE` | Legacy standalone paged-KV block size. The engine uses `TS_SCHED_BLOCK_SIZE`. |
| `TS_KV_CACHE_MAX_RAM_MB` | Legacy standalone paged-KV RAM-tier cap. |
| `TS_KV_CACHE_SSD_DIR` | Legacy standalone paged-KV SSD cold-tier directory. |
| `TS_KV_CACHE_MAX_SSD_MB` | Legacy standalone paged-KV SSD cap. |
| `TS_KV_PAGED_QUANT_BITS` | Legacy standalone paged-KV block quantization bits (`0` = passthrough, `2` = affine, `4`, or `8`). |
| `TS_SCHED_DISABLE_BATCHED` | `1` forces the per-sequence KV-swap fallback even when a model implements `IBatchedPagedModel`. The CLI shortcut is `--no-continuous-batching`. |
| `TS_SCHED_MAX_BATCHED_TOKENS` | Scheduler per-step token budget (default: `4096`). |
| `TS_SCHED_MAX_RUNNING_SEQS` | Maximum in-flight sequences (default: `16`). |
| `TS_SCHED_PREFILL_CHUNK` | Per-request prefill cap in a mixed prefill+decode step (default: `256`); prefill-only steps fairly consume the full batched-token budget. |
| `TS_SCHED_SOLO_PREFILL_CHUNK` | Prefill chunk size for the fresh (start_pos = 0) part of a SOLO prompt — one uncontended request gets big fused-prefill chunks (default: `8192`). |
| `TS_SCHED_NUM_BLOCKS` | Physical blocks in the engine block pool (default: `256`). |
| `TS_SCHED_BLOCK_SIZE` | Tokens per block on the engine side (default: `256`). |
| `TS_SCHED_PREFIX_CACHE` | `0` disables block-hash prefix sharing across requests. |
| `TS_SCHED_STOP_REPETITION` | `0` lets a generation that has locked into a loop run to its token limit instead of being stopped. |
| `TS_SCHED_DECODE_QUANTUM` | Tokens before a sequence-switch is allowed (default: block size). |
| `TS_RETAINED_FUSED_CACHE` | `1` (default) retains a finished request's fused holder so an exact-prefix continuation skips re-prefilling it, on models that advertise support (Gemma 4 K/V; Qwen 3.5/3.6 attention K/V plus GatedDeltaNet recurrent state). `0` disables it (VRAM cap / A-B). |
| `TS_RETAINED_FUSED_CACHE_MAX` | LRU budget of retained fused holders (default: `4`); each pins a complete per-request continuation state. |
| `TS_PREFIX_CHECKPOINTS` | `1` (default) checkpoints the model state at the end of the prompt every conversation shares — system prompt, tools, skills — and starts each **new** chat from a clone of it, so a new chat re-prefills only its own message. Gemma 4 and Qwen 3.5/3.6 on the GGML backends. `0` disables. |
| `TS_PREFIX_CHECKPOINTS_MAX` | How many distinct shared prefixes stay checkpointed at once, LRU (default: `2`). Each holds one copy of that prefix's K/V and, on Qwen, its recurrent state. |
| `TS_KV_INITIAL_TOKENS` | Tokens of K/V a cache is given when it is created — the primary cache at load and every per-request holder — before any request declares a budget. `0` (default) keeps the engine policy: the whole window when `MAX_CONTEXT` is explicit, otherwise a backend default. The cache still grows on demand, so a memory-constrained device sets this small because every kept holder is paid at this size, host copy and device mirror both. |
| `TS_KV_GENERATION_RESERVE_MAX` | Caps the generation share of the K/V a request reserves up front (prompt + `max_new_tokens`). Without it, a reply limit at or above the context window reserves the whole window per request. Past the cap the cache grows on demand. `0` (default) = uncapped. |
| `TS_KV_HOLDER_POOL_MAX` | How many released per-request holders a model may park for reuse instead of freeing (default: `64`). Each parked holder costs its whole K/V allocation while parked. |
| `TS_QWEN35_BATCHED` | Set to `0` to force the Qwen 3.5/3.6 family onto the legacy per-sequence KV-swap path (default: batched/paged). Also implicitly disabled by `--no-continuous-batching`. |
| `TS_QWEN35_BATCHED_GDN_NATIVE` | Use the native batched GatedDeltaNet kernel inside Qwen 3.5/3.6 batched path. |
| `TS_GEMMA4_BATCHED` | Set to `0` to force Gemma 4 onto the legacy per-sequence KV-swap path (default: batched/paged). |
| `TS_GPTOSS_BATCHED` | Set to `0` to force GPT OSS onto the legacy per-sequence KV-swap path (default: batched/paged). |
| `TS_GPTOSS_PAGED_ATTN_MANAGED` | Use the managed (C#) paged-attention-with-sinks kernel inside GPT OSS batched path. |
| `TS_NEMOTRON_BATCHED` | Set to `0` to force Nemotron-H onto the legacy per-sequence KV-swap path (default: batched/paged). |
| `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE` | Use the native Mamba2 batched step kernel inside Nemotron-H batched path. |
| `TS_PAGED_ATTN_KERNEL` | Paged-attention dispatch kernel for `Mistral3Model.BatchedForward`: `native` (default), `tensor` (C# Tensor-based), or `managed` (pure C# scalar). |
| `TS_MLX_PIPELINED_DECODE` | `1` (default) enables pipelined greedy decode on the MLX backend when the request is greedy, has no stop sequences, and the model supports device-side argmax / next-embedding lookup. Set to `0` to disable. CLI only. |
| `TS_MLX_MLOCK_GGUF` | `1` (default) pins the GGUF mmap region in physical RAM via `mlock(2)` so model weights stay resident between forward passes. Set to `0` to skip (use if the process `memlock` rlimit is too low or you want the OS to manage paging). MLX backend only. |
| `TS_MLX_FUSED_KV_WRITE` | `1` (default) uses a single multi-dim `slice_update` to write the per-token KV block. Set to `0` to revert to the per-head loop (A/B testing / regression isolation). |
| `TS_MLX_BATCHED_MOE_DECODE` | `1` (default) collapses K per-expert decode dispatches to one batched dispatch per (gate/up/down) kind for Qwen 3.5/3.6 MoE. Set to `0` on memory-constrained machines (saves ~weight-doubling overhead from the stacked weight slabs). |
| `TS_MLX_MOE_FUSED_GATE_UP_SILU` | `1` (default) fuses gate matmul + up matmul + SiLUMul into one Metal kernel for batched MoE decode. Set to `0` to A/B against the legacy 3-dispatch path. |
| `TS_MLX_DEVICE_ROUTER` | `1` (default) keeps MoE router top-K + softmax on device to skip ~60 host syncs/token on Qwen 3.6-35B-A3B. Set to `0` to disable; the code also falls back automatically when prerequisites are missing. |
| `TS_MLX_MEMORY_LIMIT_MB` / `TS_MLX_CACHE_LIMIT_MB` / `TS_MLX_WIRED_LIMIT_MB` | Override the MLX allocator hard cap / unused-buffer cache cap / wired-buffer residency cap (megabytes). Defaults are derived from the host's unified-memory capacity. |
| `TS_MLX_EVAL_EVERY_N_LAYERS` / `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS` | Periodic `mlx_async_eval` cadence during decode to overlap GPU work with host queueing. Gemma 4 defaults to every 4 layers via `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS`; Qwen 3.5 and Nemotron-H default to every 16 layers via `TS_MLX_EVAL_EVERY_N_LAYERS`. Set to `0` to disable where supported. |
| `TENSORSHARP_MLX_LIBRARY` / `TENSORSHARP_MLX_LIBRARY_DIR` | Override the search path for `libmlxc` when using `--backend mlx`. |

**MTP / speculative-decoding tunables**

These gate the optional speculative decode path (see [Speculative Decoding](FEATURES.md#speculative-decoding) and the [design doc](docs/speculative_decoding.md)). `TS_SPEC_*` are the shared knobs (also set by the `--spec*` CLI flags); the legacy `TS_MTP_*` spellings are accepted too and are written alongside them, because the glm-dsa **native** loader reads `TS_MTP_SPEC` / `TS_MTP_DRAFT` from C++ at load time. `TS_GMTP_*` are Gemma 4 draft-path A/B switches.

| Variable | Description |
|---|---|
| `TS_SPEC` *(legacy `TS_MTP_SPEC`)* | `1` enables speculative decoding for solo sequences (default `0`). CLI: `--spec` / `--no-spec`. |
| `TS_SPEC_TYPE` | Speculation algorithm: `auto` (default) / `draft-head` / `block` / `ngram`. CLI: `--spec-type`. |
| `TS_SPEC_DRAFT` *(legacy `TS_MTP_DRAFT`)* | Maximum tokens drafted per speculative step (default `8`). CLI: `--spec-draft`. |
| `TS_SPEC_PMIN` *(legacy `TS_MTP_PMIN`)* | Draft-confidence gate in `[0, 1]`, `0` = never gate (default per algorithm: `0.15` per-token head, `0.35` block, `0` n-gram). CLI: `--spec-pmin`. |
| `TS_SPEC_DRAFT_MODEL` *(legacy `TS_MTP_DRAFT_MODEL`)* | Path to the separate Gemma 4 `gemma4-assistant` draft GGUF. CLI: `--draft-model`. Ignored by Qwen 3.6 (embedded NextN). |
| `TS_GMTP_NO_FUSED` | `1` disables the Gemma 4 fused multi-token-verify / draft-step GGML kernels and falls back to the per-op path (A/B testing on ggml backends). |
| `TS_GMTP_NO_FAST_ROLLBACK` | `1` restores the kept-prefix rollback path instead of the dense exact-match fast rollback used on partial draft acceptance. |
| `TS_GMTP_BATCHED_TRUNK` | `1` opts the Gemma 4 verify trunk back into the batched paged path; the default runs the faster linear trunk for solo speculation. |

**DiffusionGemma-specific tunables**

| Variable | Description |
|---|---|
| `DIFFUSION_NO_SC` | Set to `1` to disable self-conditioning. Enabled by default. |
| `DIFFUSION_SC_TOPK` | Experimental self-conditioning top-K cutoff (default: `32`). |
| `DIFFUSION_NO_PKV` | Set to `1` to disable prompt-KV caching on device-glue backends. Enabled by default where supported. |
| `DIFFUSION_NO_FUSED_DECODE` | Set to `1` to disable the GGML fused model decode path and fall back to per-op / per-layer diffusion decode. |
| `DIFFUSION_NO_FUSED_LMHEAD_TAIL` | Set to `1` to disable the fused output-norm + lm-head + softcap tail. |
| `DIFFUSION_BATCHED_FORWARD` | Set to `1` to use true batched `DecodeCanvasBatched` for active diffusion canvases; default time-slices the faster fused single-canvas path. |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | Memory cap for batched diffusion lm-head logits before falling back to per-sequence lm-head (default: `300`). |

Sampling parameter precedence (highest wins), with the default
`--sampling-precedence config`:

1. Server-wide CLI flags / config-file keys (e.g. `--temperature`, `--top-p`, `--stop`).
2. `TENSORSHARP_*` environment variables listed above.
3. Per-request JSON fields in the API call (e.g. `temperature`, `top_p`, `stop`)
   — for every parameter that steps 1 and 2 did not set.
4. Built-in `SamplingConfig` defaults (`temperature=0.8`, `top_k=40`, `top_p=0.9`, `min_p=0`, `repeat_penalty=1.1`, `repeat_last_n=64`, presence/frequency penalties `0`, `seed=-1`, no stop sequences).

With `--sampling-precedence request`, steps 1–3 swap: the request wins over the
server-wide flags and env vars for parameters it sends, and they still fill in
the rest. Either way `--stop` sequences pinned on the server stay in force under
`config` (merged with the request's) and are replaced by the request under
`request`.

## Video generation with audio (MiniMax-H3)

MiniMax-H3 generates video **and a native 32 kHz stereo soundtrack together** — one
diffusion transformer denoises a packed video+audio latent in a single token sequence,
so the audio is part of the model output rather than something dubbed on afterwards.
Up to 15 s at 24 fps. It is CFG-distilled, so `--cfg 1.0` is required; the pipeline
defaults to 20 denoising steps and 4–8 is the fast operating point. Measured on an
M5 Pro with Metal at 22 frames, 8 steps and an identical seed, it runs 2.4× faster
than stable-diffusion.cpp at 256×256 (49.3 s → **20.9 s**) and 1.7× at 640×384
(108.5 s → **63.1 s**).

That comparison is hardware-dependent, and on a memory-starved card it inverts. On a
16 GB RTX 3080 Laptop with CUDA — same 22 frames, 8 steps, best of three —
stable-diffusion.cpp finishes first end to end: 1.15× at 256×256 (37.8 s against
43.6 s) and 1.07× at 640×384 (59.8 s against 63.7 s). Per **denoise step** TensorSharp
is still ahead on that card (3.325 s against 3.338 s by the 8-vs-16-step slope); what
it loses is fixed setup cost, and roughly 3 s of the residual 3.9 s is not inference at
all — H.264 encoding, where stable-diffusion.cpp writes MJPEG+PCM into an AVI, plus
.NET process startup against a native binary. That machine holds 16 GB of VRAM and
31.7 GB of RAM against a 33.5 GB model set, so neither the weights nor the page cache
fit and setup dominates the wall clock; peak VRAM was 15 780 MiB against
stable-diffusion.cpp's 12 035 MiB on a 16 384 MiB card. stable-diffusion.cpp ran with
`--auto-fit --stream-layers --diffusion-fa --rng cpu`, because its default
`--offload-to-cpu` path cannot run this model there at all — it tries to pin 17.7 GB
into 12.3 GB of free RAM.

Four networks are needed — one of the two denoisers plus three shared companions.
Each is loaded and released in turn, so peak VRAM is the largest of them rather than
their sum:

| Role | File | Size |
| --- | --- | --- |
| Denoiser — keyframes (`t2v` / `i2v` / `fl2v`) | `minimax_h3_fl2va_pruned-Q4_K.gguf` | 10.64 GiB |
| Denoiser — references (`t2v` / `ref`) | `minimax_h3_ref2va_pruned-Q4_K.gguf` | 10.60 GiB |
| Text encoder (Qwen3-VL-32B, 50 layers) | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 16.97 GiB |
| Video VAE (16× spatial / 4× temporal) | `minimax_h3_video_vae_fp16.safetensors` | 5.21 GB |
| Audio VAE (32 kHz stereo) | `minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB |

Which denoiser is loaded is read off the **file name**: one containing `ref2va` selects
the reference checkpoint, anything else the first/last-frame one — so a rename or a
requantization has to keep `ref2va` in the name.

Companions resolve automatically when they sit next to the denoiser; otherwise name
them with `--video-text-encoder`, `--video-vae` and `--audio-vae`. Leave the audio VAE
out and you still get video, just silent — as does `--no-audio`, which skips the audio
decode to save its time and memory; on Ref2VA that same VAE's encoder is what a
`--ref-audio` goes through, so dropping it costs reference soundtracks as well as the
generated one. The text encoder ships **no tokenizer**, so
`vocab.json` and `merges.txt` from
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/processor)
must be beside it (or point `TS_VIDEO_TOKENIZER` at them).

**What a 16 GB card does with them.** Two behaviours matter once the set stops fitting.
The denoiser's device residency is handed back **before** the video VAE loads, whenever
the two would not fit together — otherwise a finished 10.6 GB denoiser sits beside a
5.2 GB VAE on a 16 GB card, and WDDM does not fail that allocation, it backs the
overflow with shared host memory and the whole decode runs at PCIe speed (peak VRAM
during decode 16 041 MiB → ~5 600 MiB, worth 22 s at 640×384). And because weights are
bound as pointers into the mmapped GGUF, the denoiser file is read through once before
its first upload — started as soon as the text trunk produces its hidden states, and
pipelined with the upload rather than joined before it, since an H2D copy that faults
its pages in as it goes measures 0.91 GB/s on this card against 5.97 GB/s once the
pages are resident. Together, 640×384 went from 89.0 s to 63.7 s on a 16 GB RTX 3080
Laptop (256×256: 67.2 s → 43.6 s), output byte-identical either way. The release is
gated on measured free VRAM and the prefault on free host RAM. When the prefault stands
down it simply carries on; it says so with
`denoiser prefault skipped (not enough free RAM)`, but only under `TS_H3_PHASE=1` — on
default settings a skip is silent. A machine with room for both keeps the previous
behaviour exactly.
`TS_H3_PHASE=1` prints the per-stage breakdown — encoder open / trunk / teardown, the
prefault, every denoise step, VAE open / decode — which is where those seconds show up
on your own card.

The two shipped configs name all four networks, which is what lets them auto-download
into `${modelRoot}` on the first run (~33.5 GB, or ~33.4 GB for the Ref2VA set) and be
reused after. Only the denoiser differs between them, so running the second one
downloads nothing but its own DiT:

```bash
# Keyframes: text-to-video, image-to-video, first-and-last-frame
tensorsharp --config config/minimax-h3-fl2va.json \
  --prompt "a red fox trotting through falling snow, cinematic" --output fox.mp4

# References: same subject, brand-new scene
tensorsharp --config config/minimax-h3-ref2va.json \
  --ref-image person.png --ref-image jacket.png \
  --prompt "she walks through a night market, neon reflections" --output market.mp4

# Either file also hosts the server
./TensorSharp.Server --config config/minimax-h3-fl2va.json
```

Both configs pin `"backend": "ggml_cuda"` and `640×384 × 22` frames at 24 fps;
`--backend ggml_metal` on the command line overrides the backend. Neither sets steps
or guidance, because the two hosts spell steps differently (`--diffusion-steps` on the
CLI, `--video-steps` on the server) and the server takes no `--cfg` at all — so the
model's own defaults apply.

**Text to video.** Writes `fox.mp4` plus `fox.wav` with the generated soundtrack:

```bash
tensorsharp --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --prompt "a red fox trotting through falling snow, cinematic" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output fox.mp4
```

**Image to video — animate a photo.** The image becomes the FIRST FRAME and the
prompt drives what happens next:

```bash
tensorsharp --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --image portrait.jpg \
  --prompt "the person turns toward the camera and smiles, subtle handheld motion" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output animated.mp4
```

**First and last frame.** Both ends are pinned and the model fills in the motion:

```bash
tensorsharp --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --image start.png --end-image end.png --prompt "a slow cinematic push-in" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 8 --cfg 1.0 \
  --output morph.mp4
```

**Picking the mode.** An image means two different things to H3, and they use
different checkpoints. The mode is inferred from what you pass; `--video-mode` states
it explicitly.

| What you want | `--video-mode` | Checkpoint |
| --- | --- | --- |
| "animate this photo" | `i2v` | `minimax_h3_fl2va_pruned-*` |
| "go from photo A to photo B" | `fl2v` | `minimax_h3_fl2va_pruned-*` |
| "use this person, brand-new scene" | `ref` | `minimax_h3_ref2va_pruned-*` |
| "reference this product, new angle and background" | `ref` | `minimax_h3_ref2va_pruned-*` |
| text only | `t2v` | either |

Combinations that cannot mean one thing are refused up front, by name, rather than
half-honoured: `ref` on the FL2VA checkpoint, keyframes on Ref2VA, `i2v` without an
image, `fl2v` without `--image` and/or `--end-image`, `ref` with nothing to reference,
`t2v` with images supplied, and keyframes together with named references — a clip that
comes back looking like the request was honoured is the worse failure. Each message
names the checkpoint to load or the flag to drop.

Reference conditioning keeps the subject and changes everything else — camera,
background and composition come from the prompt, and the first frame need not
resemble the reference at all:

```bash
tensorsharp --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-image person.jpg --ref-image bottle.png \
  --prompt "she holds the bottle up to the light on a rooftop at golden hour, slow orbit" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output rooftop.mp4
```

Up to nine `--ref-image`s. They are only ever scaled down and keep their own aspect
ratio, so the output canvas still comes from `--width`/`--height`. A plain `--image`
on the Ref2VA checkpoint is treated as a reference too, so clients that only attach
"an image" work unchanged.

A reference can also be a **clip** (`--ref-video`, a video file or a directory of
frames) or a **soundtrack** (`--ref-audio`). A clip's own audio goes in separately
with `--ref-video-audio`, paired by position, because a container's audio track is
not readable through the frame decoder:

```bash
tensorsharp --model minimax_h3_ref2va_pruned-Q4_K.gguf --backend ggml_metal \
  --ref-video walk.mp4 --ref-video-audio walk.wav \
  --prompt "the same woman walks along a beach at sunset, wide shot" \
  --width 640 --height 384 --video-frames 22 --diffusion-steps 20 --cfg 1.0 \
  --output beach.mp4
```

A reference clip is the most expensive input H3 takes — a 22-frame 448x320 one adds
980 conditioning tokens on top of the 1680 the output needs, plus about 14 s to
encode. It is resampled to 24 fps, pulled onto the 17k+5 grid, and shown to the
language model at 2 fps.

**References are not free, and the second cost is the one that bites.** Inside the
denoiser they are linear and cheap: a 640×384 reference is 240 tokens, and on an RTX
3080 Laptop a 22-frame 640×384 step goes from 4.37 s with no references to 9.38 s with
eight — ~626 ms per reference per step, flat from one to eight. The Qwen3-VL pass is
where it turns: each reference adds ~250 vision placeholder tokens to the prompt and
all 50 layers prefill them, so two references make a 548-token prompt and eight make a
2086-token one, and text conditioning grows from ~65 s to ~447 s while the whole 8-step
denoise still costs ~75 s. Past roughly four references the encoder, not the denoiser,
is the run — reach for fewer, better references before reaching for fewer steps. The
nine-reference cap is TensorSharp's own: the packed sequence is attended over unmasked,
so its length is a numeric budget as well as a time one. Say who is in frame, too — the
reference supplies the identity, but the prompt still has to describe the shot, or the
scene comes back well rendered with the subject missing.

**Quality.** Two settings dominate:

| Lever | Default | What it does |
| --- | --- | --- |
| `--width` / `--height` | 640×384, or the image's aspect at that area | **Biggest factor.** Faces need pixels — at 256×256 they come back blurry and malformed whatever else you set. |
| `--diffusion-steps` | 20 | Removes coloured fringing around moving subjects. 8 is the fast lane, ~20 clean, past ~30 gains little. |
| `--diffusion-seed` | random | Some seeds compose better; cheapest retry. |
| model quant | — | `-Q8_0` denoiser over `-Q4_K` if memory allows. |

`--cfg` is not a quality lever (H3 only accepts 1.0) and `--negative-prompt` does
nothing, because there is no unconditional pass. Note that the seed here is
`--diffusion-seed`, not `--seed`: `--seed` is the text sampling seed and leaves the
clip alone, and without `--diffusion-seed` every run draws a fresh random one — so
two runs meant to be compared need it set explicitly.

**On the server, size is a startup flag.** The browser sends the prompt plus whatever
conditioning the hosted checkpoint advertises — a first frame, a last frame, or up to
`maxReferenceImages` references — and nothing numeric at all, so every clip inherits
the server defaults:

```bash
./TensorSharp.Server --model minimax_h3_fl2va_pruned-Q4_K.gguf --backend ggml_metal \
  --video-width 640 --video-height 384 --video-steps 20 --video-frames 22 --port 5001
```

`--video-mode` pins the conditioning mode for a deployment that only offers one;
omit it and each request's mode is inferred from what it supplies.

Omit `--video-width`/`--video-height` and each request takes its aspect ratio from
the uploaded image, which avoids stretching a 4:3 photo into a 16:9 frame.

**The HTTP surface.** Three endpoints generate video, all registered with the request
timeout disabled: `POST /api/video-generate`, `POST /api/video-generate/stream` (the
same body over SSE, ticking `{ videoGen, step, total, phase, detail, elapsedSeconds,
etaSeconds }` and ending on `{ done: true, … }`) and the OpenAI-shaped
`POST /v1/videos/generations`. All three share one parser, so one body works
everywhere: `prompt` (required), `width`, `height`, `frames`, `steps`, `cfg`, `cfg2`,
`seed`, `fps`, `flowShift`, `negativePrompt`, `sampler`, `cfgCacheStride`, `videoMode`,
`generateAudio`, `imagePath` (or inline base64 `image`), `endImage`, `referenceImages`,
`referenceVideos`, `referenceAudios` and `referenceVideoAudios` — the last paired **by
index** with `referenceVideos`. The fields added for joint audio-video and reference
conditioning also accept a snake_case spelling (`video_mode`, `generate_audio`,
`end_image`, `reference_images`, …), camelCase winning if both are sent; the older
fields are camelCase only. Every path field must name a file previously uploaded
through `/api/upload` and is confined to the upload directory.

`/api/video-generate` answers
`{ ok, url, audioUrl, width, height, frames, fps, seed, codec, elapsedSeconds }`, with
`audioUrl` null when the model generated no track; `/v1/videos/generations` also takes
`size` as `"832x480"`, `negative_prompt` and `response_format` (`url` or `b64_json`)
and answers `{ created, data: [{ url, b64_json }], audio_url, width, height, frames,
fps, seed, codec, elapsed_seconds }`. A request the model can explain — the wrong
checkpoint for the mode, a mode without its inputs, keyframes and references together —
comes back as a 400 carrying the model's own message instead of a generic 500, and a
host whose model does not generate video answers `The loaded model is not a
video-generation model.`

`GET /api/models` carries a `video` object whenever the hosted model generates video
and `null` otherwise, which is how a client learns what to offer without pattern-matching
an architecture string: `family` (`minimax-h3`), `supportsAudio` (true once an audio VAE
resolved), `supportsImageConditioning`, `supportsEndImageConditioning` (FL2VA only),
`supportsReferenceConditioning` and `maxReferenceImages` (9, on Ref2VA). The Web UI
reads exactly those fields to decide whether to attach a last frame or references.

**Sizes.** Width and height round up to a multiple of 32; the frame count rounds up
onto the `17k+5` grid (5, 22, 39, 56, 73, 90 …) and fps is pinned to 24, giving up to
15 s of clip. Any grid length decodes correctly — the video VAE runs 5 latent frames
at a time with a 2-frame look-ahead and cross-fades the seams. Decoding a long clip in
one call instead washes detail out progressively (measured against the conditioning
photo, frame 0 falls from 0.97 correlation at 22 frames to 0.86 at 90), so the chunking
is a correctness requirement rather than an optimization.

**When a long clip diverges it fails instead of saving.** H3 attends bidirectionally
over the whole clip, so the key count *is* the clip — 2364 packed tokens at 22 frames,
8646 at 107 — and a diverged flow-matching velocity would decode to a file that looks
right in every property except that every pixel is black and every audio sample is
clamped. A non-finite velocity therefore fails the request, naming the step it appeared
at, rather than writing that file. Set `TS_H3_TRACE=1` to print the latent and velocity
magnitudes for every step; the absmax leaves its normal range several steps before it
reaches infinity.

**Audio** is written as a sidecar `.wav` rather than muxed, because muxing needs an
encoder that may not be installed. Combine them with:

```bash
ffmpeg -i fox.mp4 -i fox.wav -c:v copy -c:a aac fox_with_audio.mp4
```

**`--backend cpu` runs the whole thing with no ggml.** Prompt encoding, denoising,
video VAE decode and the audio vocoder all have pure-C# implementations, and so do the
vision tower, the causal 3-D video VAE encoder and the audio VAE encoder — so `t2v`,
`i2v`, `fl2v` and reference images, clips and soundtracks all work there. Everything
runs at F32 on the host, so it is for correctness, portability and machines with no
accelerator rather than for speed; a 256x160x5f single-step `t2v` measured 69 s against
14 s on `ggml_cpu`, while the same `i2v` measured 70 s against 176 s (a single
sample -- do not read too much into that direction). Agreement with
the GGML path and the one stage whose residual is larger than GGML's own internal
spread (the vision tower) are written up in the model card.

Full detail, including the architecture and the verification numbers:
[docs/models/minimax-h3.md](docs/models/minimax-h3.md).

## Video generation (Wan)

A `wan` GGUF turns a prompt — plus an optional first-frame image on the Wan 2.2
models — into an H.264 MP4, from `TensorSharp.Cli`, the server's three video
endpoints, and the Web UI chat. Wan is the **video-only** family: it reports no
audio, no end-frame and no reference conditioning, so `--end-image`, `--ref-image`,
`--ref-video`, `--ref-video-audio`, `--ref-audio`, `--audio-vae` and `--no-audio` are
MiniMax-H3 flags in practice — for a clip that comes back with a soundtrack see
[Video generation with audio (MiniMax-H3)](#video-generation-with-audio-minimax-h3)
above. Full architecture detail is in the
[Wan card](docs/models/wan.md); this section is the operator's view: which
checkpoint to download, and which knobs actually change the wall clock.

### Which checkpoint

| Family | Latent | Modes | Notes |
|---|---|---|---|
| Wan 2.2 TI2V-5B | 48 ch, 16×16×4 (`Wan2.2_VAE.safetensors`) | T2V + I2V | dense 5B, 24 fps, natively 720p; ~2.7× fewer DiT tokens than Wan 2.1 at the same resolution, so it is both the fastest and the highest-quality option on consumer GPUs |
| Wan 2.2 A14B (T2V / I2V) | 16 ch (36 ch I2V input), `wan_2.1_vae.safetensors` | T2V + I2V | two 14B experts switched at a timestep boundary; **both** expert GGUFs must be present (same folder, or `HighNoise/` + `LowNoise/`) |
| Wan 2.1 T2V (1.3B / 14B) | 16 ch, `wan_2.1_vae.safetensors` | T2V | single DiT |

Every family also needs the UMT5-XXL text encoder
(`umt5-xxl-encoder-Q8_0.gguf`) and the matching video VAE. All three companions
are resolved from the DiT's own directory, subfolders such as `VAE/`,
`HighNoise/` and `LowNoise/` included, so one `--local-dir` is enough;
`--video-vae` / `--video-text-encoder` (and `TS_WAN_DIT2` for the second A14B expert)
override the search.

Wan is the one family that rejects a backend outright: it runs on `ggml_cuda`,
`ggml_vulkan`, `ggml_metal`, `ggml_cpu`, `cuda` and `cpu`, and **not** on
`--backend mlx`. `ggml_cuda` is the fastest (RTX 2000 Ada, Wan2.1-1.3B F16,
832×480×33f, 30 steps: `ggml_cuda` 12.0 s/step vs `ggml_vulkan` 17.2 and direct
`cuda` 19.3); `cpu` / `ggml_cpu` are for functional use only.

### The fast lane: step-distilled checkpoints

**This is the single biggest speed lever anywhere in TensorSharp, and it costs
nothing but a different download.** The official Wan2.2-TI2V-5B recipe is 50 steps × 2
classifier-free-guidance passes = **100 DiT passes**. A step-distilled
checkpoint (Turbo / Lightning / FastWan / DMD) is trained to run guidance-free
in a handful of steps, so the same video costs **4 DiT passes** — 1/25th of the
denoising work.

TensorSharp detects one from the DiT **file name**: any of `turbo`, `distill`,
`lightning`, `lightx2v`, `fastwan`, `-dmd` (case-insensitive), or an explicit
`<N>steps` / `<N>_steps` token for 1 ≤ N ≤ 16, which wins over the markers. A
marker with no step count means 4. On load the console prints

```
step-distilled checkpoint detected -> 4 steps, guidance off (--diffusion-steps / --cfg override)
```

so you can confirm from the log that it fired. No flag is involved — a distilled
GGUF is passed as an ordinary `--model`.

Measured on an M5 Pro (20-core GPU, 48 GB unified), `ggml_metal`,
Wan2.2-TI2V-5B Q8_0, 1088×832×121f = 27 404 tokens, image-to-video — i.e. the
full five-second 720p-class request:

| | Base checkpoint | **Turbo checkpoint** |
|---|---|---|
| DiT passes | 100 (50 steps × CFG) | **4** (4 steps, guidance-free) |
| per pass | 120.2 s | 120.2 s |
| denoise total | 12 020 s | **481 s** |
| VAE decode, 121 frames | 563 s | 563 s |
| **end to end** | **≈ 3 h 30 m** | **17 m 30 s** |

Only the `--model` path changes between those two columns. Once distilled, the
VAE decode is the bottleneck (~55% of the run), not the DiT.

Getting one (TI2V-5B — note the file names spell the version with an
**underscore**, `Wan2_2`, unlike the base repo):

```bash
pip install -U huggingface_hub
hf download hum-ma/Wan2.2-TI2V-5B-Turbo-GGUF Wan2_2-TI2V-5B-Turbo-Q8_0.gguf --local-dir models
# the Turbo repo ships no VAE and no text encoder — take them from the base repos
hf download QuantStack/Wan2.2-TI2V-5B-GGUF VAE/Wan2.2_VAE.safetensors --local-dir models
hf download city96/umt5-xxl-encoder-gguf umt5-xxl-encoder-Q8_0.gguf --local-dir models

dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf \
    --backend ggml_metal --image first_frame.png --output out.mp4 \
    --prompt "the cat runs toward the camera, cinematic tracking shot" \
    --video-frames 121 --fps 24

dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/Wan2_2-TI2V-5B-Turbo-Q8_0.gguf \
    --backend ggml_metal --video-frames 121 --fps 24
```

For Wan 2.2 I2V-A14B the drop-in distilled GGUFs are
[jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF](https://huggingface.co/jayn7/WAN2.2-I2V_A14B-DISTILL-LIGHTX2V-4STEP-GGUF),
which ships the distillation already merged into both experts under `high_noise/`
and `low_noise/`; download both and point `--model` at either one. It ships no
VAE and no text encoder, so take `VAE/Wan2.1_VAE.safetensors` from
[QuantStack/Wan2.2-I2V-A14B-GGUF](https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF)
and the UMT5-XXL encoder as above.

> [lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning)
> publishes LoRA `.safetensors` only, and TensorSharp has no Wan LoRA option —
> use a repo that ships the distillation already baked into the GGUF.

### `--cfg-cache-stride` (base checkpoints only)

A guided step is `v = v_cond + (cfg-1)·d` with `d = v_cond - v_uncond`. The
guidance direction `d` changes far more slowly across the schedule than `v`
does, so `--cfg-cache-stride N` runs the unconditional pass on one step in `N`
and reuses the cached `d` in between. At 50 steps, `2` runs 77 of the 100 passes
(**1.30×**) and `3` runs 70 (**1.43×**). The first three steps and the last
always recompute `d`. Server JSON field: `"cfgCacheStride": 2`.

It is an approximation — leave it off when matching a reference sample matters —
and it is pointless on a step-distilled checkpoint, which already runs
guidance-free (the cache is disabled whenever cfg ≤ 1.0).

### Making a large request cheaper, in order of effect

1. **Use a step-distilled checkpoint.** 100 DiT passes become 4; this dwarfs
   everything else.
2. **Fewer frames.** 121 → 61 roughly quarters the attention work (token count
   is `latent_frames × (h/2) × (w/2)` and self-attention is `O(tokens²)`) and
   halves the VAE decode.
3. **Smaller frame area** — but not below Wan's training resolutions. Under
   ~0.3 MP the model is out of distribution and the video gets *worse*, not just
   cheaper; the pipeline warns below that. Wan is trained at 480p (832×480) and
   720p (1280×704), so generate at a supported size and downscale afterwards.
4. **Fewer steps**, base checkpoints only — 30 instead of the official 50 is
   visibly close and 1.7× cheaper.
5. **`--cfg-cache-stride 2` or `3`** — 1.30× / 1.43×, base checkpoints only.

Resolution against wall clock on the same M5 Pro, same Turbo checkpoint and
image:

| Output | Tokens | Denoise | VAE decode | **Total** |
|---|---|---|---|---|
| 736×544 × 81f (3.4 s, 480p class) | 8 211 | 84 s | 159 s | **4 m 09 s** |
| 736×544 × 121f (5 s, 480p class) | 12 121 | 137 s | 237 s | **6 m 19 s** |
| 1088×832 × 121f (5 s, 720p class) | 27 404 | 481 s | 563 s | **17 m 30 s** |

480p (≈0.4 MP) is a resolution Wan is *trained* at, so the first two rows are
in-distribution rather than a degraded mode — that is the setting to reach for
when a few minutes matters.

### Server defaults

`--video-frames N` and `--fps N` set server-wide **defaults**, not caps, for the
Web UI and all three video endpoints; a request that supplies `frames` or `fps`
overrides each independently. With both omitted the model recipe applies: 49
frames at 24 fps for Wan2.2-TI2V, 33 at 16 fps otherwise. Frame counts are
snapped to the VAE's `4k+1` temporal grid. Keep the model's native FPS and change
the frame count to change duration — changing only FPS changes playback speed.

### Other Wan knobs

These exist for A/B and debugging; all of them make things slower except where
noted. `TS_WAN_DIT_KV_F16=0` restores F32 attention keys/values (F16 is the
default and is 2.02× faster on a single 27 k-token self-attention, with no
measurable accuracy cost);
`TS_WAN_VAE_MPS_CONV=0` restores ggml's im2col+GEMM conv lowering on Metal
(MPSGraph is the default and took a 736×544×81f VAE decode from 159 s to 80 s);
`TS_WAN_VAE_GEMM_MAX_MB` sets the im2col budget and `TS_WAN_VAE_TILE=0` disables
tiling; `TS_WAN_DIT_CAPTURE=0` disables the persistent CUDA-graph-captured DiT
graph; `TS_WAN_DIT_FLASH=0` forces materialized attention;
`TS_WAN_HEARTBEAT_S` sets the progress tick interval (default 30 s, `0`
silences it); `TS_FFMPEG` points at the `ffmpeg` used for near-lossless CRF 17
H.264 export.

## Mixture-of-Experts CPU offload (`--n-cpu-moe`)

Large MoE models spend almost all of their bytes on routed experts while
activating only a small fraction of them per token — Qwen3.6-35B-A3B is ~11 GB
of weights for ~3B active parameters. `--n-cpu-moe N` keeps the routed experts
of the first N layers in system RAM, leaving attention, the norms, the router
and the always-active shared expert on the accelerator. `--n-cpu-moe` therefore
means "these experts do not live in VRAM" — not "these experts are multiplied on
the CPU". Where they are multiplied depends on the batch:

* **Decode (one token, or any batch under `TS_HOST_MOE_DEVICE_MIN_BATCH`, default
  128).** The host multiplies them, reading the quantized blocks zero-copy out of
  the GGUF mmap. One token only touches `n_expert_used` experts, so this is a few
  MB of RAM traffic per layer against the hundreds of MB an upload would cost.
* **Prefill (a real batch).** The experts are *streamed* to the accelerator for
  that one graph and multiplied there, then the staging memory is reused by the
  next layer — they are never made resident. A 512-token batch amortizes the
  upload over every token in it, and the arithmetic is what the GPU is for.
  llama.cpp's scheduler makes the same call (`ggml_backend_sched` sends an op
  whose weights live in a host buffer back to the GPU above
  `op_offload_min_batch_size`).

Three things make the streamed side fast, and they are why TensorSharp's
offloaded prefill runs several times faster than llama.cpp's on the same files:

1. **The host pages are locked.** A copy out of a pageable mmap cannot DMA — the
   driver stages it through a small pinned buffer. `cudaHostRegister` on the
   expert range costs ~65 ms/GiB once and takes the transfer from 9.3 GB/s to
   55.6 GB/s on PCIe 5.0 x16. Disable with `TS_HOST_MOE_PIN=0`; bound it with
   `TS_HOST_MOE_PIN_MAX_MB` (default: 60% of the cgroup/host memory limit).
2. **Only the experts this batch routes to are sent**, grouped into consecutive
   runs — the same trick llama.cpp's scheduler plays with its used-expert bitset.
   At 512 tokens a large expert pool is only partly covered, and at the small
   batches speculative verification and light serving produce it is a large
   saving. `TS_HOST_MOE_EXPERT_FILTER=0` restores whole-stack uploads.
3. **The copies are issued asynchronously** on the backend's own stream and
   synchronize once, with the graph, instead of three times per layer.

The offloaded layers stay inside the fused whole-model graph: the accelerator
pauses after each offloaded layer's router, the experts are multiplied (on the
host, or on the accelerator from streamed weights), and the result is handed back
before the next segment runs. Everything else in the token — attention, the
shared or dense FFN, the LM head — stays in one graph submission. This holds for
Qwen3.5/3.6, Gemma 4 MoE, GPT-OSS and DiffusionGemma; Gemma 4 MoE segments its
prefill graph the same way, and DiffusionGemma's block decode hands the host all
of the canvas positions at once so its offloaded side is a GEMM rather than a
matvec. Qwen3.5/3.6 segments its prefill graph the same way, and both it and
Gemma 4 MoE keep the fused graph under tensor parallelism too (see the `--tp`
note below). GPT-OSS now has a fused whole-model prefill graph as well, so its
offloaded prefill is segmented rather than dispatched per layer. Architectures
that still lack one (Nemotron-H) reach the streamed path through their per-layer
MoE op, on a single device; under `--tp N` they stay host-side so that N ranks do
not each stream their own copy of the same unsharded experts.

Measured on 2 x Xeon 6952P + RTX PRO 6000 Blackwell (PCIe 5.0 x16), **pp8192 /
tg128**, peak per-process VRAM. The full comparison against llama.cpp — two prompt
lengths, five offload depths per model, including DeepSeek V4 Flash across two
GPUs — is in
[docs/moe_cpu_offload_benchmark.md](docs/moe_cpu_offload_benchmark.md):

| Model | Setting | Peak VRAM | Prefill | Decode |
| --- | --- | --- | --- | --- |
| gemma-4-26B-A4B (UD-IQ4_XS) | default | 16.4 GiB | 11,274 tok/s | 161 tok/s |
| | `--n-cpu-moe 8` | 15.4 GiB | 6,500 tok/s | 80 tok/s |
| | `--n-cpu-moe 16` | 13.8 GiB | 4,888 tok/s | 55 tok/s |
| | `--cpu-moe` (30 layers) | 10.8 GiB | 3,072 tok/s | 40 tok/s |
| Qwen3.5-35B-A3B (UD-IQ4_XS) | default | 19.4 GiB | 9,405 tok/s | 160 tok/s |
| | `--n-cpu-moe 24` | 15.1 GiB | 5,259 tok/s | 52 tok/s |
| | `--cpu-moe` (48 layers) | 11.3 GiB | 3,709 tok/s | 39 tok/s |
| gpt-oss-20b (Q8_0/MXFP4) | default | 12.9 GiB | 12,925 tok/s | 213 tok/s |
| | `--n-cpu-moe 12` | 9.2 GiB | 6,394 tok/s | 52 tok/s |
| | `--cpu-moe` (24 layers) | 4.7 GiB | 3,798 tok/s | 28 tok/s |
| DeepSeek-V4-Flash (UD-Q8_K_XL, 2 GPUs) | default | 165.2 GiB | 4,387 tok/s | 51 tok/s |
| | `--n-cpu-moe 12` | 128.7 GiB | 428 tok/s | 10 tok/s |
| | `--n-cpu-moe 24` | 77.9 GiB | 236 tok/s | 5.3 tok/s |

At every offload depth that is **4.5-10.9x** llama.cpp's prefill and **2.5-4.5x**
its decode on the three seam models. Peak VRAM is higher than llama.cpp's (1.1x
resident, up to 3.5x fully offloaded) — see the report's VRAM note.

Greedy output is **token-identical** at every offload depth on gemma-4-26B-A4B.

A small-machine picture from the same feature on a laptop (RTX 3080 Laptop
16 GB / i7-11800H, 96-token generation) — note how far under the card's 16 GB
the offloaded configurations sit, and that both the GPT-OSS and DiffusionGemma
baselines were over the WDDM spill threshold, which is why offload makes them
*faster* as well as smaller there:

| Model | Setting | Peak VRAM | Prefill | Decode |
| --- | --- | --- | --- | --- |
| gemma-4-26B-A4B (Q4_K_XL) | default | 16.1 GB | 190 tok/s | 39.7 tok/s |
| | `--n-cpu-moe 8` | 13.1 GB | 52 tok/s | 38.6 tok/s |
| | `--n-cpu-moe 16` | 10.2 GB | 24 tok/s | 21.6 tok/s |
| | `--cpu-moe` (30 layers) | 4.8 GB | 15 tok/s | 17.7 tok/s |
| gpt-oss-20b (Q8_0) | default | 16.2 GB | 2.7 tok/s | 0.3 tok/s |
| | `--n-cpu-moe 12` | 14.0 GB | 58 tok/s | 25.4 tok/s |
| | `--cpu-moe` (24 layers) | 2.9 GB | 29 tok/s | 12.1 tok/s |
| diffusiongemma-26B-A4B (Q4_K_M) | default | 16.0 GB | — | 1709 ms/step |
| | `--n-cpu-moe 16` | 9.8 GB | — | 2287 ms/step |
| | `--cpu-moe` (30 layers) | 3.0 GB | — | 4697 ms/step |

(Those laptop prefill numbers predate the streamed-expert path; the same
configurations are several times faster now.)

Notes:

* **Pick the smallest N that fits.** Each offloaded layer costs decode
  throughput, so offload only as many layers as you need to free the VRAM the
  KV cache wants.
* **Decode pays the most, proportionally.** Prompt processing streams the
  experts to the accelerator and stays within a small factor of the resident
  run; decode is bounded by how fast the host can read `n_expert_used` experts
  per layer out of RAM, which is where the throughput goes.
* **The worker count is capped at 64.** The decode-side matmul is one token
  wide — memory-bound work with a barrier after every op — so past a few dozen
  threads each extra worker only adds a barrier participant, and on a 2-socket
  host it adds cross-socket traffic too. Measured on 2x Xeon 6952P
  (192 cores / 384 threads), gemma-4-26B `--cpu-moe`, ms of host matmul per 30
  offloaded layers: 8 threads 45, 16 threads 35, 32 threads 17.6, 64 threads
  17.3, 96 threads 26, 192 threads 120. `--cpu-moe-threads N` overrides.
* **Routing stays on the accelerator**, so expert selection and weights are the
  same ones the fully resident path would pick.
* **The fused whole-model graph is kept.** Qwen3.5/3.6, Gemma 4 MoE, GPT-OSS and
  DiffusionGemma each run a whole token (for DiffusionGemma, a whole canvas
  block) as ONE accelerator graph, and offload does not give that up. Each
  offloaded layer's expert matmuls are cut out of the graph; everything else —
  attention, norms, the router, the shared/dense FFN and the LM head — keeps
  running fused, with the accelerator pausing only to hand the host that layer's
  routed-expert work. Gemma 4 MoE segments its *prefill* graph the same way, so a
  prompt chunk is still one dispatch with the host doing a real GEMM over every
  token in it.
* Nemotron-H has no whole-model fused graph to preserve — its hybrid
  Mamba2/attention/MoE decode is dispatched per layer regardless — so offload
  there simply routes each layer's MoE through the host path.
* **Combines with tensor parallelism.** Under `--tp N` the offloaded layers'
  seams are merged into the same segment schedule the ranks already use for
  their AllReduce points, so the fused multi-rank graph is kept — Qwen3.5/3.6
  and Gemma 4 MoE run both decode and prefill fused. Each offloaded layer is
  evaluated ONCE, on the host, over the unsharded expert stack: the host expert
  backend is a single thread pool, so splitting that matmul per rank would only
  serialize on it while paying an extra download each. The single result is then
  placed so the graph's own reduction lands on the single-GPU value (rank 0 only
  where the layer output feeds a later AllReduce, every rank where it does not).
  Measured on 2x L4 24 GB (short prompt, prefill 512 / decode 64):

  | Model | Setting | Resident weights (GPU0 + GPU1) | Prefill | Decode |
  | --- | --- | --- | --- | --- |
  | Qwen3.5-35B-A3B (UD-IQ4_XS) | `--tp 2` | 9397 + 8032 MB | 1942 tok/s | 76.5 tok/s |
  | | `--tp 2 --cpu-moe` | 2277 + 912 MB | 66 tok/s | 17.8 tok/s |
  | gemma-4-26B-A4B (UD-IQ4_XS) | `--tp 2` | 6835 + 6087 MB | 3062 tok/s | 59.7 tok/s |
  | | `--tp 2 --n-cpu-moe 8` | 5459 + 4711 MB | 217 tok/s | 36.5 tok/s |
  | | `--tp 2 --cpu-moe` | 1588 + 840 MB | 60 tok/s | 13.2 tok/s |

  Greedy output on Gemma 4 is byte-identical to the same `--tp N` run without
  offload on most prompts and, on the rest, agrees for hundreds of characters
  before taking an equivalent alternative ending; Qwen3.5 likewise tracks the
  resident run to a paraphrase point. Both are deterministic — the gap is the
  host expert matmul's activation quantization, the same ~1-5% relative
  difference `TS_HOST_MOE_VERIFY=1` reports on a single GPU. The pure-C# `cuda`
  backend has no host-MoE seam, so `--tp N --cpu-moe` there prints the
  `[moe-offload] WARNING` and keeps the experts resident.
* **GLM 5.x serves its offloaded experts straight from the mapping.** The native
  glm-dsa executor keeps host-resident routed experts in the GGUF mmap and
  multiplies them in place instead of copying them into a private buffer
  (`TS_GLM_MOE_MMAP=0` copies instead), which is what makes offloading 30 of a
  744B model's layers a load-time no-op rather than a 100 GB memcpy. It composes
  with `--tp`: offloaded layers keep their experts whole and rank 0 evaluates
  them. Measured on 3x RTX PRO 6000, GLM-5.2 UD-IQ2_XXS, prefill 2048 /
  decode 64: `--n-cpu-moe 30` is 94.7 / 16.4 tok/s against 915.9 / 43.9 fully
  resident. Offload buys the fit here, not the speed — it frees enough VRAM to
  raise the sized context from 342,272 to 646,400 tokens.
* `TS_HOST_MOE_VERIFY=1` builds the on-GPU expert chain alongside the host one
  and reports their per-layer divergence — a diagnostic for validating the seam
  on a model that also fits in VRAM. `TS_HOST_MOE_DEBUG=1` prints the segment
  plan (the node cuts) and each seam's activation norms. `TS_HOST_MOE_TIMING=1`
  reports the offloaded side's wall clock split into per-call setup and host
  matmul, which is what tells you whether a slow run is the expert GEMM or the
  scaffolding around it.

## Tensor Parallelism & Distributed Inference

TensorSharp supports **tensor parallelism (TP)** — splitting a single model across
multiple GPUs using the Megatron-LM column/row-parallel pattern — and
**distributed (multi-node) tensor parallelism**, where TP spans multiple
machines connected over a TCP peer-to-peer network.

### TP vs. the layer split — what happens on a multi-GPU box

There are two different ways a model can occupy more than one GPU, and only one
of them is `--tp`.

**Tensor parallelism (`--tp N`)** puts *every* layer on *every* rank and splits
the weights *within* each layer, so a decode step reads `1/N` of the bytes per
device and the ranks all-reduce at each layer boundary. Every architecture in the
table below marked ✅ supports local TP, and it is **opt-in** — nothing splits a
tensor unless you ask. A ✅ does not by itself claim distributed/cross-node support;
local-only paths are called out in the notes.

**The layer split** puts *whole layers* on different devices and runs them in
sequence — device 0 evaluates layers 0..k, hands the hidden state to device 1,
and so on. The cut points are not a naive `n_layer/N`: the loader measures each
device's free VRAM and bin-packs the layers to balance the largest
fraction-of-budget used, so an uneven set of cards still fills up evenly. There
are no collectives and no per-layer split, so it costs nothing on a slow
interconnect, but only one GPU is busy at a time. It applies to the architectures
that run through their own whole-model executors: **DeepSeek V4 Flash
(`deepseek4`)**, **GLM 5.x (`glm-dsa` / `glm5next`, GLM-5.2, GLM-5.3 and GLM-5.3-Flash alike)** and
**Qwen 3.8 Flash Next (`qwen4exp`)**. On the first two it is what they do **by
default, with no flag at all**: they spread across every visible GPU because
neither fits on one card, and `TS_DSV4_NGPU` / `TS_GLM_NGPU` cap how many devices
they use. On `qwen4exp` it is opt-in — one GPU unless you pass `--tp N`.

On **every other architecture**, running without `--tp` uses a **single GPU**.
There is no automatic layer split on the generic per-op or fused-graph paths — a
model that does not fit one card fails at load rather than being spread silently.
(The refusal that names the exact `--n-cpu-moe N` you would need comes from the
DeepSeek V4 and GLM 5.x whole-model loaders.) Pass `--tp N` to an architecture
that supports neither tensor parallelism nor a layer split and it now says so on
stderr and runs on one GPU, instead of silently leaving the other cards idle.

So on a 3-GPU box, `--backend ggml_cuda` alone gives you all three GPUs on GLM 5.x
and DeepSeek V4 (layer split) and one GPU on Gemma 4 and Qwen 3.8 Flash Next;
adding `--tp 3` switches GLM 5.x to its local, single-process native tensor
parallelism (for GLM-5.2, GLM-5.3 and GLM-5.3-Flash alike), caps DeepSeek V4's layer split at three devices
(there `--tp N` is only a device count — the same thing `TS_DSV4_NGPU` sets), and
gives Gemma 4 all three GPUs and Qwen 3.8 Flash Next a three-way layer split.
On the measured GLM-5.2 run that switch is a downgrade in speed and buys only
capacity — see the
**What to expect** measurements below.

**Qwen 3.8 Flash Next (`qwen4exp`) is the new one.** `--tp N` there runs a layer
split: each GPU holds a contiguous run of whole layers. It is not tensor
parallelism — `qwen4exp` shards no weights — and it is the same (and only)
multi-GPU mode llama.cpp offers this architecture, whose `-sm row` refuses to load
it. Treat it as capacity. Measured on 2× A100-80GB with
Qwen3.8-Flash-Next-UD-Q2_K_XL (73.4 GiB), the 1-GPU and 2-GPU runs produce
**byte-identical greedy output** (same SHA-256); VRAM goes from one card holding
everything to 24.2 GB + 26.2 GB, roughly half the model each; and throughput is
unchanged either way — prefill ~1520-1550 t/s, decode ~56 t/s. For reference,
llama.cpp on the same box measures pp1536 1094 / tg128 61.2 on one GPU and
1200 / 61.5 on two with `-sm layer`, so it too gains ~10% prefill and nothing on
decode. Startup prints which mode ran and the per-GPU layer/byte split.

`TS_Q4E_LAYER_SPLIT=20,28` overrides the automatic balance with explicit layer
counts per GPU (llama.cpp's `--tensor-split` in spirit). It throws rather than
silently ignoring a value it cannot honour. It is worth reaching for because the
automatic balance prices weights and cannot see the vision tower, which loads
later and lands on GPU 0.

### Local tensor parallelism (single process, multiple GPUs)

Split the model across N GPUs within one process (direct CUDA, or the GGML CUDA / Vulkan backends). Each GPU holds `1/N` of
the sharded weights (column-parallel QKV/gate/up, row-parallel output/down) plus
a full copy of replicated weights (norms, embeddings, LM head). Per-GPU KV
caches are independent; AllReduce (via CUDA P2P copies + elementwise-add kernel)
reconverges the hidden state after each row-parallel projection.

```bash
# CLI: 2-GPU tensor parallelism
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2

# Server: same flag (TENSORSHARP_TP_DEGREE=2 env var also works)
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
    --model <model.gguf> --backend cuda --tp 2

# Config JSON
{ "tp": 2, "backend": "cuda", "model": "<model.gguf>" }
```

### Distributed tensor parallelism (multi-node, peer-to-peer TCP)

Distribute TP across multiple machines. Each node runs its own process with its
own local GPUs; nodes communicate over a TCP mesh using a length-prefixed
framing protocol. AllReduce is hierarchical: local P2P within each node, then
TCP across node representatives, then broadcast back — minimizing network
traffic to `1/tp_local` of the data per collective.

```bash
# 2 nodes × 2 GPUs each (4 GPUs total)
# Node 0:
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# Node 1:
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda --tp 2 \
    --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# Server as the cluster front-end: the server must be node 0 (the driver that
# owns sampling and serves HTTP); every other node runs a TensorSharp.Cli worker
# with the same model, backend, and peer list. The TENSORSHARP_TP_* env vars
# work as well.
# Node 0 (server / driver):
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model <model.gguf> --backend cuda \
    --tp 2 --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# Node 1 (CLI worker):
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend cuda \
    --tp 2 --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# Config JSON (per node)
{ "tp": 2, "tp-node-id": 0, "tp-peers": "192.168.1.10:9500,192.168.1.11:9500", "backend": "cuda" }
```

Every node must use the same `--tp-peers` list (or `TENSORSHARP_TP_PEERS` env
var) and a unique `--tp-node-id` (or `TENSORSHARP_TP_NODE_ID`). The port
(`9500` in the examples) is not a default — it must be specified explicitly and
must be reachable between all nodes.

### Supported architectures

| Architecture | TP status | Notes |
|---|---|---|
| Mistral 3 | ✅ | Fused/separate QKV, YaRN RoPE |
| Gemma 4 | ✅ | Dense TP + MoE. On GGML the fused whole-model MoE trunk splits *inside* each expert (gate/up column-parallel, down row-parallel) so global expert ids keep working; `TS_GEMMA4_TP_FUSED_MOE=0` falls back to the whole-expert per-op path. Per-expert slicing on direct CUDA |
| Qwen 3.5 / 3.6 family | ✅ | GatedDeltaNet SSM with per-rank V-head ownership; expert-parallel MoE on GGML (whole experts per rank, Megatron-split shared expert), expert slicing on direct CUDA. Runs on both `cuda` and `ggml_cuda` / `ggml_vulkan` — the GGML path uses the packed per-rank GDN kernel (`TSGgml_Qwen35GdnLayerTP`) with device-resident recurrent state |
| Qwen 3.8 Flash Next | layer split | Not tensor parallelism: `--tp N` gives each GPU a contiguous run of whole layers, which is also the only multi-GPU mode llama.cpp offers `qwen4exp` (`-sm row` refuses to load it). Capacity, not speed — 2× A100-80GB on Qwen3.8-Flash-Next-UD-Q2_K_XL (73.4 GiB): greedy output byte-identical to the 1-GPU run (same SHA-256), VRAM 24.2 + 26.2 GB instead of one card holding everything, prefill ~1520-1550 t/s and decode ~56 t/s either way. `TS_Q4E_LAYER_SPLIT=20,28` sets the per-GPU layer counts by hand |
| GPT OSS | ✅ | Attention sinks, YaRN. Runs on `cuda` and the GGML backends; the GGML path is expert-parallel (whole experts per rank, one batched `ggml_mul_mat_id` dispatch per projection per layer) and falls back to per-expert slicing only when the expert count does not divide the TP degree |
| Nemotron-H | ✅ | Mamba2 replicated on rank 0, MoE expert slicing. Still walks experts per token per rank on GGML (no expert parallelism yet) |
| GLM 5.x | ✅ (local only) | GGML GPU backends only; the native TP path is local/single-process for GLM-5.2, GLM-5.3 and GLM-5.3-Flash alike (`--tp-node-id` / `--tp-peers` are hard-refused for the whole family). GLM-5.2 shards MLA heads and hidden rows inside every routed expert. GLM-5.3-Flash also head-shards KDA with per-rank recurrent state; its MLA heads and routed-expert hidden rows are sharded likewise. Attention partials reduce before each nonlinear Sinkhorn hyper-connection. On GLM-5.3-Flash's eligible segmented fast path, routed-MoE partials reduce first, then each rank computes/adds the replicated shared expert locally; hyper-connections, pooled indexer, router, norms, dense layers and embedding also remain unsharded and execute per rank, while output norm / LM head stay on rank 0. CPU MoE, tracing, partial `TS_GLM_TP_SHARD`, oversubscription, or missing native hyper-connection kernels use the combined scheduler fallback, where the shared expert runs once on rank 0; `TS_GLM_TP_FUSED=0` forces that fallback for diagnostics. Plain GLM-5.3 accepts `--tp N` on the same local terms and replicates the MLA and indexer caches on every rank, so `--tp` multiplies the KV footprint; nothing has ever been run on it at `--tp N>1` — the only recorded arithmetic is a non-fit, `--tp 8` wanting 41.7 GiB per rank against 46 GB cards — so treat it as an accepted mode rather than a validated configuration, and note that `--spec` is refused there whenever `--tp N>1`. Omitting `--tp` keeps the default layer split |
| DeepSeek V4 Flash | layer split | Not tensor parallelism: the whole-model executors bin-pack contiguous runs of whole layers against each device's free VRAM, and `--tp N` (or `TS_DSV4_NGPU`) only caps how many GPUs that split uses |
| DeepSeek V4.1 Flash | layer split (+ experimental routed-MoE TP) | Layer placement is the default and the measured path. `TS_DSV41_TP=N` (2-8, and equal to the GPU count selected by `--tp` / `TS_DSV4_NGPU`) shards routed-expert gate/up along the FFN intermediate dimension and down along its input, reducing partials through host-staged F32 buffers; attention, shared experts and caches keep their layer placement. Block-aligned unequal partitions (the 2304-wide intermediate is nine 256-element K-quant blocks: 1280+1024 on two ranks, 768+512+512+512 on four). The first full Q2_K run was slower than the layer split, so treat it as experimental. Attention TP and distributed groups are not implemented |
| Hunyuan Dense | — | Single device: no TP and no layer split. Startup says so on stderr rather than leaving extra GPUs idle |
| DiffusionGemma | — | Not applicable (diffusion model) |
| Qwen-Image-Edit | — | Not applicable (image generation) |

### Backend support

TP runs on the **direct CUDA** backend (`--backend cuda`) and on the **GGML
CUDA / Vulkan** backends (`--backend ggml_cuda`, `ggml_vulkan`). MLX is
single-device.

On the GGML backends each rank gets its own ggml backend on its own GPU, with
its own device-resident weight shards and KV cache. Cross-GPU AllReduce uses
ggml-cuda's collective (NCCL when the build finds it, its P2P pipeline
otherwise); small payloads are reduced in host memory instead, which is cheaper
because GGML activations already live there.

```bash
# 2 GPUs on the GGML CUDA backend
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model <model.gguf> --backend ggml_cuda --tp 2

# Choose which physical GPUs the ranks map to
TENSORSHARP_TP_DEVICES=0,2 dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll \
    --model <model.gguf> --backend ggml_cuda --tp 2
```

**What to expect.** GGML TP started as a *capacity* feature — it lets a model
that does not fit in one GPU's VRAM run entirely on GPUs — and fused per-rank
execution (Stage 1c) made it a latency win as well. Each TP block now runs as a
per-rank fused native graph (attention, dense FFN, MoE trunk, GatedDeltaNet)
instead of op-at-a-time, so a rank issues a handful of graph launches per layer
rather than hundreds of native calls.

Measured on 2× RTX 2000 Ada (16 GB each, PCIe, no NVLink), prefill 512 /
decode 64, tok/s:

| Model | 1 GPU | `--tp 2` |
|---|---|---|
| Gemma 4 E4B Q8_0 | 2760 / 37.3 | 2488 / **51.7** |
| Gemma 4 26B-A4B IQ4_XS | 1845 / 48.5 | 2537 / **51.2** |
| Qwen 3.5-9B Q8_0 | 1461 / 23.1 | 399 / **24.4** |
| Qwen 3.5-35B-A3B IQ4_XS | does not fit | **184 / 18.1** |

Decode — the memory-bound half TP should help — is 1.39× a single GPU on
Gemma 4 E4B and 1.06× on Qwen 3.5-9B; both Gemma 4 models produce output
byte-identical to their single-GPU runs. Prefill is compute-bound and pays the
collectives, so it lands at or below the single-GPU figure on models that fit on
one card. Qwen 3.5-35B does not fit a 16 GB card at all, so TP is the only way
to run it. See `TENSOR_PARALLELISM_PLAN.md` (Stages 1b and 1c) for the full
measurements and what is left to fuse.

How much TP buys depends on the interconnect and on how much of the layer it can
actually split. On GLM-5.2 (3x RTX PRO 6000, PCIe, no NVLink) it buys nothing:
`--tp 3` measures pp2048 505.6 / tg64 17.6 against 915.9 / 43.9 on the plain
layer split, because each of the 78 layers needs two all-reduces of a
`[6144, n_tokens]` hidden state and that costs more bus time than the split saves
in arithmetic. It also holds a full-length cache on *every* rank, which drops the
fitted context from 342,272 to 91,136 tokens, and it changes the reduction order,
so against the recorded llama.cpp goldens a 2-bit MoE reproduces 3 of 6 prompts
under `--tp 3` where the layer split reproduces 5 of 6.
Reach for it there on an NVLink host, or when a model fits no other way.

TP composes with MoE CPU offload: `--tp N --n-cpu-moe M` keeps the fused
multi-rank graph and drops the offloaded layers' expert bytes from every rank's
VRAM. See [Mixture-of-Experts CPU offload](#mixture-of-experts-cpu-offload---n-cpu-moe)
for the combined numbers.

| Variable | Effect |
|---|---|
| `TENSORSHARP_TP_DEVICES` | GPU ordinals per rank, e.g. `0,2` (default `0..tp-1`) |
| `TS_GGML_TP_PARALLEL=0` | Drive ranks sequentially instead of concurrently (diagnostic) |
| `TS_GGML_TP_FUSED_MATMUL=1` | Submit both ranks' linears from one thread (off by default; it allocates a device buffer per rank per call, measured 2.3× slower on Qwen 3.5 35B) |
| `TS_GGML_TP_DEVICE_AR_THRESHOLD` | Element count above which AllReduce uses the device collective (default 262144) |
| `TS_Q4E_LAYER_SPLIT=20,28` | Qwen 3.8 Flash Next only: explicit layer counts per GPU for its layer split, instead of the automatic VRAM balance. Throws rather than silently ignoring a value it cannot honour — useful because the automatic balance prices weights and cannot see the vision tower, which loads later and lands on GPU 0 |
| `TS_GGML_F32_RESIDENT=0` | Bind F32 linear weights per call instead of keeping them device-resident (diagnostic) |
| `TS_GEMMA4_TP_FUSED_MOE=0` | Gemma 4 only: fall back from the fused whole-model MoE trunk (Megatron split inside each expert) to the whole-expert per-op path. The fused path materializes ~10.5 GB of expert slices at load on the 26B (~36 s) in exchange for a ~10× decode. Layers offloaded by `--n-cpu-moe` are skipped by that materialization — they never run on the accelerator — so `--cpu-moe` also removes the load-time cost |
| `GGML_CUDA_AR_BF16_THRESHOLD` | Payload size above which ggml-cuda's collective converts F32 to BF16 before reducing. TensorSharp raises ggml's default (1 byte — i.e. always) to 1 MB so decode-sized collectives reduce exactly; `0` disables the conversion entirely |
| `TS_QWEN35_LAYER_TRACE=1` | Print a per-layer residual-stream summary for the first forward, from both the single-GPU and TP loops (diagnostic) |
| `GGML_CUDA_ALLREDUCE` | `nccl` / `internal` / `none`, passed through to ggml |

### Constraints

- `numHeads`, `numKVHeads`, and `intermediateSize` must be divisible by the TP degree.
- Quantized row-parallel splits require `ne0` divisible by `tp × blockSize`.
- Batched/continuous-batching forward under TP is implemented for Mistral 3; MoE models (Gemma 4, Qwen 3.5/3.6, GPT OSS, Nemotron-H) fall back to per-sequence forward under TP.
- **Muse-Glimmer** caps at `--tp 2`: it has 2 KV heads, and no model here replicates KV heads when `numKVHeads < tp`. Its DFlash drafter and pooled KV-block snapshots stay single-GPU under TP (multi-turn reuse comes from live-cache continuation instead), and it requires the GGML CUDA/Vulkan backends — the fused per-rank plan needs a device collective that ggml-metal does not provide.

### Cluster tuning & diagnostics

Local AllReduce prefers CUDA peer-to-peer DMA. At startup the group enables peer
access for every device pair that reports it, then runs a round-trip self-test:
some topologies (L4 behind certain PCIe switches, IOMMU-enabled hosts, small
BAR1 windows) advertise peer access but silently transfer corrupt data, and any
pair that fails the test is demoted to host staging permanently. Hardware
without peer access at all (A16 vGPU profiles, most consumer cards) simply
stages through host memory from the start.

| Variable | Default | What it does |
|---|---|---|
| `TENSORSHARP_TP_DISABLE_P2P=1` | off | Force every cross-GPU transfer through host memory, exactly as no-peer hardware does. Use it to check whether a multi-GPU defect lives in the P2P DMA path. |
| `TENSORSHARP_TP_HOST_ALLREDUCE=1` | off | Run the local AllReduce as device→host, sum on the CPU, host→device. Slower, but matches the multi-node reduce exactly — useful for isolating P2P correctness issues. |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS=N` | `120` | How long a node retries outbound connections to its peers. Nodes are usually launched by hand seconds or minutes apart, so a peer's listener may not be up yet; raise this for slow orchestrators. |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS=N` | `300` | Per-receive timeout on a peer socket. Without it a stalled peer would block on the OS TCP keepalive (often 2+ hours) instead of failing the collective. |

Startup logs make the topology explicit: the local group prints
`Tensor parallelism: N GPUs (<device names>)`, P2P demotions print a
`TP: P2P disabled…` / self-test warning, and every distributed node prints
`[TcpCommunicator] Rank r/N connected to all peers.` once the mesh is complete.

### Redis-backed shared state

The server can optionally persist KV cache blocks and the OpenAI Responses API
store to Redis, enabling cross-session KV reuse and durable response storage:

```bash
# Enable Redis for both KV cache and Responses API
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model <model.gguf> --backend cuda \
    --redis-url localhost:6379

# KV cache tier only, with a 12-hour TTL
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model <model.gguf> --backend cuda \
    --paged-kv-redis-url localhost:6379 --paged-kv-redis-ttl 720
```

## Feature × environment variable matrix

Quick reference for which environment variables (and matching CLI flags) gate each major feature. Variables in **bold** are required to turn the feature on; everything else is a tunable for a feature that's already enabled by default.

#### Continuous batching & paged KV cache

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Continuous-batching engine (`InferenceEngine` + scheduler) | ON in `TensorSharp.Server` | `TS_SCHED_DISABLE_BATCHED=1` to force per-seq fallback | `--no-continuous-batching` / `--continuous-batching` |
| Legacy per-session paged-KV manager | removed from Server request path | `TS_KV_PAGED_CACHE` (`0` / `1`), `TS_KV_BLOCK_SIZE` retained for compatibility / standalone tests | `--paged-kv` / `--no-paged-kv`, `--paged-kv-block-size N` |
| Legacy paged-KV SSD spillover (standalone manager) | OFF | `TS_KV_CACHE_MAX_RAM_MB`, `TS_KV_CACHE_SSD_DIR`, `TS_KV_CACHE_MAX_SSD_MB` | `--paged-kv-ram-mb`, `--paged-kv-ssd-dir`, `--paged-kv-ssd-mb` |
| Legacy paged-KV block quantization (standalone manager) | OFF (`0` = passthrough) | `TS_KV_PAGED_QUANT_BITS` (`0` / `2` / `4` / `8`) | `--paged-kv-quant-bits` |
| Block-hash prefix sharing across requests | ON | `TS_SCHED_PREFIX_CACHE=0` to disable | — |
| Scheduler tunables (per-step token budget, max in-flight seqs, prefill chunks, block pool size, decode quantum) | engine defaults | `TS_SCHED_MAX_BATCHED_TOKENS`, `TS_SCHED_MAX_RUNNING_SEQS`, `TS_SCHED_PREFILL_CHUNK`, `TS_SCHED_SOLO_PREFILL_CHUNK`, `TS_SCHED_NUM_BLOCKS`, `TS_SCHED_BLOCK_SIZE`, `TS_SCHED_DECODE_QUANTUM` | — |

#### Per-model batched / paged forward (`IBatchedPagedModel.ForwardBatch`)

| Model | Default state | Env var to flip default | Native-kernel sub-toggle |
|---|---|---|---|
| Mistral 3 | ON | — | `TS_PAGED_ATTN_KERNEL` = `native` (default) / `tensor` / `managed` |
| Gemma 4 | ON | `TS_GEMMA4_BATCHED=0` to force legacy per-seq | — |
| Qwen 3.5 / 3.6 family | ON | `TS_QWEN35_BATCHED=0` to force legacy per-seq (or `--no-continuous-batching`) | `TS_QWEN35_BATCHED_GDN_NATIVE=1` enables native batched GDN kernel; `FUSED_ATTN_LAYER_MIN_SEQ_LEN=N` overrides fused-attention engage threshold (default 4096) |
| GPT OSS | ON | `TS_GPTOSS_BATCHED=0` to force legacy per-seq | `TS_GPTOSS_PAGED_ATTN_MANAGED=1` forces the managed (C#) sinks softmax instead of the native paged-attention-with-sinks kernel |
| Nemotron-H | ON | `TS_NEMOTRON_BATCHED=0` to force legacy per-seq | `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` enables the native batched Mamba2 step (NEON SIMD + GCD parallelism) |
| GLM 5.x | not implemented — concurrency runs on native per-sequence **slots** instead (each request owns its MLA and indexer caches and its own `n_past`; binding a request switches the active slot without moving KV bytes). MLA keeps one 576-wide row per token and the DSA indexer scores that same contiguous history, so there is no paged-KV layout to batch over | — | Batched fused decode over those slots is ON by default (one graph, one token per sequence, weights read once): 1.81x aggregate decode at 4 concurrent requests. Set `TS_BATCHED_FUSED_DECODE=0` to use serial fused decode; `TS_GLM_BATCHED_DECODE=0` also makes the GLM native side decline batching. Batching changes GEMM shapes, and a 2-bit MoE can amplify that into different expert picks. |
| DiffusionGemma | Separate diffusion scheduler in the Web UI path; not an `IBatchedPagedModel` autoregressive path | `DIFFUSION_MAX_BATCH`, `DIFFUSION_STEPS` | `DIFFUSION_BATCHED_FORWARD=1` enables true batched canvas decode; fused GGML decode is on by default unless disabled with `DIFFUSION_NO_FUSED_DECODE=1` |

#### Speculative decoding

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Speculative decode engine (solo sequences) | OFF | **`TS_SPEC=1`** (legacy `TS_MTP_SPEC`) | `--spec` / `--no-spec` |
| Speculation algorithm | `auto` | `TS_SPEC_TYPE` | `--spec-type auto\|draft-head\|block\|ngram` |
| Max tokens drafted per step | `8` | `TS_SPEC_DRAFT` (legacy `TS_MTP_DRAFT`) | `--spec-draft N` |
| Draft-confidence gate | per algorithm (`0.15` / `0.35` / `0`) | `TS_SPEC_PMIN` (legacy `TS_MTP_PMIN`) | `--spec-pmin X` |
| Gemma 4 separate draft GGUF (`gemma4-assistant`) | none | `TS_SPEC_DRAFT_MODEL` (legacy `TS_MTP_DRAFT_MODEL`) | `--draft-model <path>` |
| Muse-Glimmer DFlash / DFlash2 drafter GGUF | none | `TS_MUSE_GLIMMER_DFLASH` | `--draft-model <path>` |
| Qwen 3.5 / 3.8 DFlash2 drafter GGUF | none | `TS_QWEN35_DFLASH` | `--draft-model <path>` |
| Fused DFlash graphs (ggml) | ON | `TS_DFLASH_FUSED=0` falls back to the per-op drafter | — |
| DFlash speculative prefill chunk | `1024` (capped by the drafter ring and the trunk's window) | `TS_DFLASH_PREFILL_CHUNK` | — |
| DFlash2 candidate selector | ON when the checkpoint has one | `TS_DFLASH_SELECTOR=0` drafts by per-position argmax instead (diagnostic only) | — |
| DFlash2 grouped convolution | ON when the checkpoint has one | `TS_DFLASH_CONV=0` drops it (diagnostic only) | — |
| Qwen 3.5/3.8 recurrent-state snapshots | ON | `TS_Q35_VERIFY_SNAPSHOTS=0` reverts to restoring a pre-verify state copy and re-forwarding the accepted prefix (slower; see [qwen35.md §12.5](docs/models/qwen35.md)) | — |
| Gemma 4 fused verify / draft kernels (ggml) | ON | `TS_GMTP_NO_FUSED=1` falls back to per-op | — |
| Gemma 4 dense fast rollback on partial accept | ON | `TS_GMTP_NO_FAST_ROLLBACK=1` restores kept-prefix rollback | — |
| Gemma 4 verify trunk path | linear (solo) | `TS_GMTP_BATCHED_TRUNK=1` runs the batched paged trunk | — |

#### GLM 5.x (`glm-dsa`)

The full list, including the debug and A/B knobs, is in the
[GLM card](docs/models/glm.md#environment-knobs).

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Executor | native whole-model ggml graph | `TS_GLM_NATIVE=0` runs the managed per-op path on a GGML backend | `--backend cpu` / `cuda` are managed regardless |
| Prefill micro-batch | `1024` | `TS_GLM_UBATCH=N` — on GLM-5.2, `2048` measures pp2048 1145.8 vs 918.9 when VRAM allows | — |
| GPUs the layer split spreads over | all visible | `TS_GLM_NGPU=N` | — |
| Per-device headroom left for compute buffers | `3072` MB | `TS_GLM_VRAM_RESERVE_MB=N` | — |
| Context window | advertised length as a **ceiling**, refitted to free VRAM after load | `MAX_CONTEXT=N` makes it a hard limit instead | — (env only) |
| Tensor-parallel split halves | `3` (heads + routed experts) | `TS_GLM_TP_SHARD` (1 heads, 2 experts, 3 both), `TS_GLM_TP_OVERSUBSCRIBE=1` packs ranks onto one GPU for testing | `--tp N` |
| GLM-5.3-Flash TP execution | concurrent segmented rank-local graphs when the full local-GPU configuration is eligible | `TS_GLM_TP_FUSED=0` forces the combined scheduler diagnostic fallback; CPU MoE, tracing, partial sharding, oversubscription, or missing native hyper-connection kernels also select it automatically | — |
| Host-resident experts read from the GGUF mmap | ON | `TS_GLM_MOE_MMAP=0` copies into a private buffer instead | `--n-cpu-moe N` selects the layers |
| Batched fused decode across sequences | ON | **`TS_BATCHED_FUSED_DECODE=0`** disables it; `TS_GLM_BATCHED_DECODE=0` makes the native side decline it | — |
| Flash attention / fused lightning indexer | ON | `TS_GLM_FA=0`, `TS_GLM_FUSED_LID=0` fall back to primitives | — |
| Cached built+allocated graphs | `8` | `TS_GLM_GRAPH_CACHE=N` | — |
| Weight-load parallelism | `16` threads / `64` MB chunks | `TS_GLM_LOAD_THREADS`, `TS_GLM_LOAD_CHUNK_MB` | — |

#### DeepSeek V4.1 Flash (`deepseek41`)

The full picture, including the native loader's own knobs, is in the
[DeepSeek V4.1 card](docs/models/deepseek41.md).

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Executor | native whole-model ggml graph; `--backend ggml_cuda` is the serving path | — | `--backend cpu` runs the whole graph on the 100% pure-C# `DeepSeek4CpuExecutor` instead — no ggml, no native library, no GPU — as a correctness and portability path, not a serving one; `--backend ggml_cpu` is a different path again, the native graph on scalar ggml CPU kernels |
| Compute width of the pure-C# executor | `Environment.ProcessorCount` on `--backend cpu`, against min(cores, 32) on the other backends | `TS_DSV4_THREADS=N` | — |
| Numerical gate on the pure-C# executor | `InferenceWeb.Tests.Dsv41CpuExecutorTests` against the independent PyTorch oracle `eng/dsv41-reference.py` at atol=rtol=2e-5 plus exact greedy argmax, over one-shot prefill, chunked prefill at 1/3/5/8 checked at every position, and reset | **`TS_DSV41_FIXTURE_DIR`** — the tests return silently unless it is set. The fixture is a five-layer, 256-hidden, 16-token F32 synthetic model, so what it establishes is architectural agreement with the oracle, not CPU/CUDA parity on the real 246 GiB Q2_K weights | — |
| Per-tensor trace for diffing against the oracle | OFF | `TS_DSV4_CPU_TRACE_DIR=<dir>` writes the same per-tensor files `eng/dsv41-reference.py --output` writes, so the two directories diff tensor by tensor | — |
| Engram sidecar | `deepseek41.engram.bin` is mandatory on every backend, `--backend cpu` included | `TS_DSV41_ENGRAM_DEVICE` must be `0` on `--backend cpu` (any other value is refused before a weight is read); `TS_DSV41_ENGRAM_WARM` / `_THREADS` / `_RANDOM` / `_SIDECAR` are native-loader-only and inert there | — |
| Routed-MoE tensor parallelism | OFF | `TS_DSV41_TP=N` on the native path; any non-zero value is refused on `--backend cpu`, as are distributed TP groups | `--tp N` |
| Native-loader-only attention / gather knobs | — | `TS_DSV41_SPARSE_FA`, `TS_DSV41_COMPACT_RAW_GATHER` — inert on `--backend cpu` | — |
| Speculative drafting | none exists for V4.1 on any backend | `TS_DSV4_DSPARK` is a hard refusal on V4.1, not the warn-and-continue DeepSeek V4 does | `--draft-model` (refused too) |
| Vision companion | a native ggml component | — | not available on `--backend cpu`, where `--mmproj` throws; `--backend ggml_cpu` is the CPU backend the companion follows the text model onto |
| Context window on `--backend cpu` | `MAX_CONTEXT` caps it to `65,536`, and every compressed-cache row lives in host memory, so the advertised 1M is not reachable there | `MAX_CONTEXT=N` | — (env only) |
| Multi-turn KV prefix reuse and per-sequence slots on `--backend cpu` | neither: every diverging turn re-prefills and concurrent requests serialize | — | — |

#### MiniMax-H3 video + audio

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Denoiser residency released before the video VAE loads | ON when the two would not fit in free VRAM (decode peak 16 041 → ~5 600 MiB on a 16 GB card, worth 22 s at 640×384) | — (gated on measured free VRAM) | — |
| Prefault the denoiser GGUF before its first upload | `3` — read pipelined with the upload | `TS_H3_PREFAULT`: `0` off, `1` serial, `2` overlapped with text conditioning (worse — the encoder streams its own 17 GB through the same page cache and evicts what was just placed), `3` default | — |
| Prefault read streams | `1` | `TS_H3_PREFAULT_THREADS=N` — more is slower here, because the read runs concurrently with the teardown and upload it is warming (640×384 best of three: 1 stream 63.9 s, 4 streams 64.9 s, 16 streams 66.6 s on a 16 GB RTX 3080 Laptop) | — |
| Text-encoder trunk run in layer groups, each group's device copy released | OFF | `TS_H3_TE_GROUP=N` layers per group — removes the encoder's own spill (peak 16 041 → 12 981 MiB) and is bit-identical, but measured ~3 s slower on a 16 GB RTX 3080 Laptop: a one-shot prefill reads each weight once, so grouping moves all 17 GB anyway and adds allocate/invalidate churn | — |
| Per-stage timing breakdown (encoder open / trunk / teardown, prefault, each denoise step, VAE open / decode) | OFF | `TS_H3_PHASE=1` | — |
| Per-step latent / velocity magnitudes | OFF | `TS_H3_TRACE=1` | — |
| Execution path | seven native whole-network ggml graphs | — | `--backend cpu` runs the same pipeline in pure C# instead (t2v, i2v, fl2v and reference conditioning) |
| Managed-vs-GGML parity diagnostics | OFF | `TS_H3_DUMP_TE`, `TS_H3_DUMP_VEL_V`, `TS_H3_DUMP_VEL_A`, `TS_H3_DUMP_VIS` write those tensors to disk so ONE forward can be compared across the two paths; `TS_H3_DIT_LAYERS=N` truncates the trunk on BOTH paths, turning the comparison into an error-vs-depth curve; `TS_H3_NO_FLASH=1` runs GGML's explicit-softmax attention instead of its flash kernel | — |
| Companion network + tokenizer overrides | resolved next to the denoiser | `TS_VIDEO_TEXT_ENCODER`, `TS_VIDEO_VAE`, `TS_VIDEO_AUDIO_VAE`, `TS_VIDEO_TOKENIZER` | `--video-te`, `--video-vae`, `--audio-vae` |

#### Tensor parallelism & distributed inference

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Local tensor parallelism (multi-GPU, single process) | OFF (`1` GPU) | **`TENSORSHARP_TP_DEGREE=N`** | `--tp N` (CLI and server) |
| GPU ordinals used by the TP ranks (GGML backends) | `0..tp-1` | `TENSORSHARP_TP_DEVICES=0,2` | — |
| Explicit per-GPU layer counts for the `qwen4exp` layer split (`--tp N`) | automatic VRAM balance | `TS_Q4E_LAYER_SPLIT=20,28` | — |
| Distributed TP node ID (multi-node) | unset (disabled) | **`TENSORSHARP_TP_NODE_ID=N`** | `--tp-node-id N` (CLI and server; the server must be node `0`) |
| Distributed TP peer endpoints | unset (disabled) | **`TENSORSHARP_TP_PEERS=host1:port1,host2:port2`** | `--tp-peers host1:port1,host2:port2` (CLI and server) |
| Peer connect retry window (multi-node) | `120` s | `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS=N` | — |
| Per-receive timeout (multi-node) | `300` s | `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS=N` | — |
| Force host-staged cross-GPU copies | OFF (P2P when available) | `TENSORSHARP_TP_DISABLE_P2P=1` | — |
| Force host-staged local AllReduce | OFF (device-to-device) | `TENSORSHARP_TP_HOST_ALLREDUCE=1` | — |

#### Redis shared state (server)

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Redis KV cache tier | OFF | **`TS_KV_CACHE_REDIS_URL`** | `--redis-url` or `--paged-kv-redis-url` |
| Redis KV cache entry TTL | `1440` min (24 h) | `TS_KV_CACHE_REDIS_TTL_MINUTES` (`0` = no TTL) | `--paged-kv-redis-ttl` |
| Redis Responses API store | OFF (in-memory) | **`TS_RESPONSES_STORE_REDIS_URL`** | `--redis-url` |

#### Backends

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Default compute backend | `ggml_metal` (macOS), `ggml_cpu` (Windows/Linux) | `BACKEND` | `--backend` |
| First-forward logit dump (backend A/B) | OFF | `TS_DUMP_LOGITS=<path>` writes the first REAL forward's logits there once, as raw float32, and deliberately skips the warm-up forwards (`WarmUpKernels` runs its own throwaway decode and prefill first, so dumping those compares two executors on a meaningless token). Lets two backends be compared by logit vector instead of by generated text, where greedy decoding turns a near-tie into a visibly different sentence | — |
| MLX backend library lookup | probe app dir | `TENSORSHARP_MLX_LIBRARY` (full path to `libmlxc`), `TENSORSHARP_MLX_LIBRARY_DIR` (directory) | — |
| MLX pipelined greedy decode (CLI only) | ON when eligible | `TS_MLX_PIPELINED_DECODE=0` disables | — |
| MLX `mlock(2)` of GGUF mmap so weights stay resident | ON | `TS_MLX_MLOCK_GGUF=0` to disable | — |
| MLX fused multi-dim KV write (single `slice_update` per cache block) | ON | `TS_MLX_FUSED_KV_WRITE=0` to revert to per-head loop | — |
| MLX batched MoE decode (Qwen 3.5/3.6 MoE) | ON | `TS_MLX_BATCHED_MOE_DECODE=0` for legacy per-expert path | — |
| MLX fused MoE gate+up+SiLUMul Metal kernel | ON | `TS_MLX_MOE_FUSED_GATE_UP_SILU=0` for legacy 3-dispatch | — |
| MLX on-device MoE router top-K + softmax | ON when prerequisites are met | `TS_MLX_DEVICE_ROUTER=0` disables | — |
| MLX layer-boundary `async_eval` cadence | Gemma 4: every 4 layers; Qwen / Nemotron: every 16 layers | `TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS=N` or `TS_MLX_EVAL_EVERY_N_LAYERS=N` (`0` = disabled where supported) | — |
| MLX allocator caps (memory / cache / wired buffer) | host-derived | `TS_MLX_MEMORY_LIMIT_MB`, `TS_MLX_CACHE_LIMIT_MB`, `TS_MLX_WIRED_LIMIT_MB` | — |

#### Pure C# CPU backend (`--backend cpu`)

The managed matmuls run on a persistent spin-then-park worker pool instead of a
`Parallel.For` per matmul. It deliberately does **not** take every core: the rest of
the CPU path still uses the ThreadPool, and pool workers spin between jobs, so
spinning on every core starves that other work. Measured on gemma-4-E4B-it-Q8_0 with
a 122-CPU allocation (prefill / decode tok/s, two interleaved runs per
cell): pool off 21.7,21.0 / 2.0,2.4; 32 threads 24.9,24.1 / 4.9,5.0; 48 threads
25.6,28.5 / 5.4,6.0; 61 threads 24.2,24.9 / 6.3,5.9; 122 threads 13.5 / 4.8 — so
roughly +15% prefill and 2.8x decode at the default width. At 122 only prefill
regresses; decode still beats the pool-off baseline.

Those tok/s are the generic managed per-op path on gemma-4-E4B-it-Q8_0 and
nothing else. **DeepSeek V4.1 Flash does not run through that path at all — like
DeepSeek V4 Flash, it has a whole-model executor of its own on this backend** —
the 100% pure-C# `DeepSeek4CpuExecutor`, with no ggml, no native library and no
GPU, one sequence at a time, no KV prefix reuse and no per-sequence slots — held
to the PyTorch oracle `eng/dsv41-reference.py` at atol=rtol=2e-5 on a five-layer
F32 fixture. It is a correctness and portability path, not a serving one, and no
throughput, load time or resident footprint has ever been measured for a V4.1
full checkpoint here; its compute width comes from `TS_DSV4_THREADS`, not
`TS_CPU_THREADS`. See
[DeepSeek V4.1 Flash (`deepseek41`)](#deepseek-v41-flash-deepseek41) above.

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Worker-pool width | every core up to 8 CPUs; half above that, never below 8 | `TS_CPU_THREADS=N` | — |
| Worker pool at all | ON | `TS_CPU_POOL=0` reverts to the ThreadPool `Parallel.For` behaviour, so the two can be A/B-ed in one binary | — |
| Spin iterations before a worker parks | `4096` | `TS_CPU_SPIN=N` — parking is the expensive part at this width, so the default spins long enough that the steady state never parks | — |
| Work-item sizing for a managed matmul | `131072` weight bytes per item, at most `4` items per worker | `TS_CPU_TASK_BYTES`, `TS_CPU_TASKS_PER_WORKER` — sized from the work rather than the thread count | — |
| DeepSeek V4.1 Flash's whole-model executor | pure C# (`DeepSeek4CpuExecutor`) — a whole-model executor here rather than the generic per-op path | `TS_DSV4_THREADS=N` sets its compute width (every processor by default), not `TS_CPU_THREADS` | — |
| Quantized weights on the direct video networks (Wan, MiniMax-H3) | kept in their GGUF storage type and multiplied there | `TS_DIRECT_QUANT_WEIGHTS=0` expands every quantized weight to F32 once at load and runs a plain GEMM instead (the previous behaviour; 4x the weight memory). On Wan at 256x160x5f, one step, the in-place path measured 80.9 s against 121.4 s | — |

#### Agent Skills

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Agent Skills | ON (a `skills` directory beside the binary, created if missing) | `TS_NO_SKILLS=1` disables the feature | `--no-skills` |
| Skill directories to scan | `<binDir>/skills` | `TS_SKILLS_DIR` (path-separator-separated list) | `--skills-dir <path>` (repeatable) |
| Skills active up front | none | — | `--skill <name>` (repeatable); per request, `"skills": [...]` |
| Advertising unselected skills to the model | ON | — | `--skills-no-discovery`; per request, `"skills_discovery": false` |
| `skills_run` (executing a skill's scripts) | **OFF** | **`TS_SKILLS_ALLOW_EXEC=1`** | `--skills-allow-exec` |
| Skill-script OS sandbox | `required` (refuse when safe confinement is unavailable) | `TS_SKILLS_SANDBOX` | `--skills-sandbox off\|preferred\|required` |
| Skill-script network access | **OFF** | `TS_SKILLS_ALLOW_NETWORK=1` | `--skills-allow-network` |
| Skill lookups (and code rounds) per turn | `8`, or `24` with `--code-exec` (range 1-64) | `TS_SKILLS_MAX_ROUNDS` | `--skills-max-rounds N` |
| Print the registry and exit | — | — | `--list-skills` |

`TS_NO_SKILLS` and `TS_SKILLS_ALLOW_EXEC` treat any value other than `0` as on.
Both hosts accept the same flag spellings, and a config-file key *is* a CLI flag
(`"skills-dir": ["/srv/skills"]`), so one config file drives either. Full
reference: [Agent Skills in TensorSharp](docs/agent_skills.md).

#### Code execution (the `shell` tool)

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| The `shell`, `read_file`, `edit_file`, `write_file` and `apply_patch` tools | **OFF** | **`TS_CODE_EXEC=1`** | `--code-exec` |
| Installing packages | **OFF** | **`TS_CODE_EXEC_ALLOW_INSTALL=1`** | `--code-exec-allow-install` |
| Unrestricted host IP-network access for model-authored commands | **OFF** | **`TS_CODE_EXEC_ALLOW_NETWORK=1`** | `--code-exec-allow-network` |
| Packages an install may name | any (at most 16 per install) | — | `--code-exec-packages <list>` |
| Package index for host-performed installs | installer default | `TS_CODE_EXEC_INSTALL_INDEX` | `--code-exec-install-index <url>` |
| Hosts an install may reach | `pypi.org`, `files.pythonhosted.org`, `registry.npmjs.org` (empty = no pinning) | `TS_CODE_EXEC_INSTALL_DOMAINS` | `--code-exec-install-domains <list>` |
| Deadline for one command | `120` s (a call may ask for up to 10 min) | — | `--code-exec-timeout N` |
| Sampling temperature for coding turns | unset; coding turns still remove the built-in repetition penalty | — | `--code-exec-temperature 0..2` (negative clears it) |
| Shell | `bash`, then `sh`; PowerShell on Windows | — | `--code-exec-shell <path\|name>` |
| Output kept per command | `32768` bytes, truncated from the middle | — | `--code-exec-max-output N` |
| OS sandbox | `required` (refuse where filesystem/IP-network confinement is unavailable) | — | `--code-exec-unconfined` (CLI and server; required on Windows) |

`TS_CODE_EXEC`, `TS_CODE_EXEC_ALLOW_INSTALL` and `TS_CODE_EXEC_ALLOW_NETWORK`
treat any value other than `0` as on. Model-authored commands get no Internet/IP
socket by default, but local Unix IPC is not a complete isolation boundary. The
network opt-in grants unrestricted host IP-network access without removing macOS/Linux write or home-read confinement. Linux additionally bounds descendants with a PID namespace. On macOS, children inherit Seatbelt and ordinary process groups are stopped, but a deliberately detached child can outlive the request; every result reports that gap. macOS denies common
`/private/tmp/com.apple.launchd*` pathname sockets while permitting runtime-required
Mach lookup and the exact mDNSResponder pathname socket required for DNS, and retains
shared-temporary-directory Unix IPC for compatibility;
Linux hides common `/run` endpoints, but its host network namespace may expose
abstract sockets and pathname sockets outside `/run`. The install opt-in remains
a narrow, host-performed operation. Neither code-exec switch controls skill scripts — use
`TS_SKILLS_ALLOW_NETWORK` / `--skills-allow-network` for those. Windows cannot
provide the filesystem sandbox and therefore still needs
`--code-exec-unconfined` on either host.
`--code-exec-languages` is the one flag that went with the old program-shaped
tools: it is refused by name at startup and has no replacement to point at,
because a shell reaches every interpreter on PATH, so an operator with an old
script gets that error instead of watching a setting be ignored.

#### Sampling defaults (server-only)

These fill in fields the request body omits. CLI flags win over env vars, and
anything set through either outranks the request body unless the server runs
with `--sampling-precedence request` (see [Web Application](#web-application)).

| Sampling field | Env var | CLI equivalent |
|---|---|---|
| `temperature` | `TENSORSHARP_TEMPERATURE` | `--temperature` |
| `top_k` | `TENSORSHARP_TOP_K` | `--top-k` |
| `top_p` | `TENSORSHARP_TOP_P` | `--top-p` |
| `min_p` | `TENSORSHARP_MIN_P` | `--min-p` |
| `repeat_penalty` | `TENSORSHARP_REPEAT_PENALTY` | `--repeat-penalty` |
| `presence_penalty` | `TENSORSHARP_PRESENCE_PENALTY` | `--presence-penalty` |
| `frequency_penalty` | `TENSORSHARP_FREQUENCY_PENALTY` | `--frequency-penalty` |
| `seed` | `TENSORSHARP_SEED` | `--seed` |
| max tokens | `MAX_TOKENS` | `--max-tokens` |
| stop sequences | — (CLI / per-request only) | `--stop` (repeatable) |
| sampling precedence | `TENSORSHARP_SAMPLING_PRECEDENCE` | `--sampling-precedence` |

#### Hosting & uploads (server-only)

| Feature | Default | Env vars |
|---|---|---|
| ASP.NET Core listener | `http://0.0.0.0:5000` | `--port` / `--host` / `--urls`, then `PORT` / `HOST`, then `ASPNETCORE_URLS` |
| Text and born-digital PDF uploads | Full extracted content; the final rendered prompt must fit the loaded model context | — |
| Video-frame extraction | 1 fps (time-based, no cap) | `VIDEO_SAMPLE_FPS`, `VIDEO_MAX_FRAMES` |
| DiffusionGemma Web UI denoising | 48 steps, max batch 2 | `DIFFUSION_STEPS`, `DIFFUSION_MAX_BATCH` |

#### Logging (server + CLI)

| Feature | Default | Env vars | CLI equivalent |
|---|---|---|---|
| Console + file log minimum level | `Information` | `TENSORSHARP_LOG_LEVEL` | `--log-level` |
| File logger output directory | `<binDir>/logs` | `TENSORSHARP_LOG_DIR` | `--log-dir` |
| File logger enabled | ON | `TENSORSHARP_LOG_FILE=0` to disable | `--log-file 0\|1` |
| Console logger enabled | ON | — | `--log-console 0\|1` (CLI only) |

#### Native build (compile-time only)

These are read by `build-linux.sh` / `build-windows.ps1` / the auto-build during `dotnet build` for `TensorSharp.GGML.Native`, not at run time.

| Feature | Default | Env vars | Build-script flag |
|---|---|---|---|
| Enable GGML CUDA in the native build | auto-detected from toolchain | `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` | `--cuda` / `--no-cuda` |
| Enable GGML Vulkan in the native build | auto-detected from the installed Vulkan runtime; a portable toolchain (headers, glslc, SPIRV-Headers) is downloaded when no Vulkan SDK / dev packages are installed | `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON/OFF` | `--vulkan` / `--no-vulkan` |
| Narrow `CMAKE_CUDA_ARCHITECTURES` list | auto-detected from visible GPU | `TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES` | `--cuda-arch='86-real;89-real'` |
| Native build parallelism cap | all CPUs, bounded by RAM (~3 GB per `nvcc` job) | `TENSORSHARP_GGML_NATIVE_BUILD_PARALLEL_LEVEL` | — |
| Native build CMake generator (Windows) | Ninja when available, else `Visual Studio NN` | `CMAKE_GENERATOR` | `-G <generator>` |
| Visual Studio installation used by the native build (Windows) | auto-detected, including installs flagged incomplete | `TENSORSHARP_VS_INSTALL_DIR` | — |

## Server Logging

The server emits one structured Information-level entry at the start and end of
every chat / generate turn, so a single grep over the log file provides a compact
request-response audit trail without replaying any traffic.

| Event id | Emitted on | Carries |
|---|---|---|
| `ChatStarted` (1500) | `chat.start`, `generate.start`, plus per-protocol request banners | sampling config, message + attachment counts, `userInput=` (bounded latest-user preview), and `fullInput=` (JSON array of every message with attachment paths, original character counts, and bodies capped at 512 characters). Inlined uploaded documents are replaced by an omission marker while retaining the user's trailing instruction. `/api/generate` likewise logs a bounded prompt preview. |
| `ChatCompleted` (1502) | `chat.complete`, `generate.complete` | token counts, KV cache reuse (`kvReused`, `kvReusePercent`), TTFT, elapsed, throughput, finish reason, full raw assistant output (reasoning + result) |
| `ChatAborted` (1503) | client disconnected mid-stream | partial output, KV reuse fraction at the time of abort |
| `KvCacheReusePlan` (1510) | per-prefix-reuse decision | `Debug`-level fine-grained breakdown (exact match / partial / full reset) |
| `HttpRequestStarted/Completed` (1100/1101) | every HTTP request | method, path, remote IP, status, duration; `/api/queue/status` is demoted to `Debug` so high-frequency UI polling does not drown out the per-turn entries |

The raw assistant output captures `<think>...</think>`, `<|channel|>analysis`,
and any other inline framing the model emits, so the completion entry contains
both reasoning and the user-visible result. Input bodies are deliberately bounded:
losslessly uploaded documents are not duplicated into logs, avoiding large memory/I/O
spikes and accidental document disclosure. The upload manifest and `contentChars`
retain useful audit metadata; capture requests separately when byte-for-byte replay is
required. Set `TENSORSHARP_LOG_LEVEL=Warning` to suppress per-turn Information logs.

Sample `fullInput` payload (formatted for readability; it is emitted as a
single line in the actual log):

```json
[
  {"role":"system","content":"You are a helpful assistant.","contentChars":28},
  {"role":"user","content":"What is the tallest mountain?","contentChars":29},
  {"role":"assistant","content":"Mount Everest.","contentChars":14},
  {"role":"user","content":"How tall is it?","contentChars":15,"images":["/uploads/mountain.jpg"]}
]
```

The same per-turn KV cache reuse stats are surfaced through every API:

- **Web UI SSE** (`POST /api/chat`) - the `done` event carries `promptTokens`, `kvReusedTokens`, and `kvReusePercent`.
- **Ollama NDJSON** (`POST /api/generate`, `POST /api/chat/ollama`) - the final chunk and the non-streaming response carry `prompt_cache_hit_tokens` (int) and `prompt_cache_hit_ratio` (0..1).
- **OpenAI** (`POST /v1/chat/completions`) - the `usage` block carries `prompt_tokens_details.cached_tokens`, matching the OpenAI extension that existing SDKs already understand.

The Web UI footer line under each assistant message also surfaces the cache hit
inline (e.g. `187 tokens · 2.1s · 87.2 tok/s · KV 420/512 (82%)`).

## HTTP APIs

TensorSharp.Server exposes three API styles. See [API_EXAMPLES.md](TensorSharp.Server.Host/API_EXAMPLES.md) for full documentation with curl and Python examples.

**Ollama-compatible API:**

```bash
# List models
curl http://localhost:5000/api/tags

# Generate text
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "prompt": "Hello!", "stream": false}'

# Chat
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "stream": false}'

# Chat with thinking mode
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Solve 17*23"}], "think": true, "stream": false}'

# Chat with tool calling
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "What is the weather?"}], "tools": [{"function": {"name": "get_weather", "description": "Get current weather", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"]}}}], "stream": false}'

# Chat with Agent Skills. "skills" is accepted on every chat surface
# (/v1/chat/completions, /v1/responses, /api/chat Ollama and Web UI); optional
# "skills_discovery": false restricts the request to exactly the skills it names.
# The model's skills_read calls are answered inside the server, so the response is
# an ordinary completion rather than a tool call the client has to service.
curl -X POST http://localhost:5000/api/chat/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Pull the totals table out of this statement."}], "skills": ["pdf"], "skills_discovery": false, "stream": false}'

# List the registered skills; GET /v1/skills/{name} adds the SKILL.md body as
# "instructions". /api/skills additionally reports load errors and whether
# uploads are accepted; POST /api/skills installs a .zip and DELETE removes one.
curl http://localhost:5000/v1/skills
```

**OpenAI-compatible API:**

```bash
# Chat completions
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it-Q8_0.gguf", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'

# Structured outputs (OpenAI response_format)
#
# Enforced by grammar-constrained decoding: the schema is compiled to a grammar
# and any token that would break it is removed from the distribution before
# sampling, so the response is structurally valid by construction rather than
# repaired afterwards. Supported keywords: type, enum, const, properties,
# required, additionalProperties, items, prefixItems, min/maxItems, anyOf, oneOf,
# allOf, $ref/$defs (including recursive), min/maxLength, pattern, the
# date/time/date-time/uuid formats, and integer minimum/maximum.
# Refused up front (a CFG cannot express them): not, if/then/else,
# dependentSchemas, dependentRequired, multipleOf, patternProperties.
# Set TS_JSON_GRAMMAR=0 to fall back to the older prompt-and-repair behaviour.
#
# NOTE: the grammar guarantees what it emits is well-formed, but it cannot
# guarantee the document FITS in max_tokens. Because end-of-sequence stays
# masked until the JSON is complete, too small a budget truncates mid-object.
# Give structured requests enough headroom.
curl -X POST http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-E4B-it-Q8_0.gguf",
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
    model="gemma-4-E4B-it-Q8_0.gguf",
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
