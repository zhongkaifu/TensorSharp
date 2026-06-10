# Deploying TensorSharp.Server to a Hugging Face Space (Docker, CPU)

[`Dockerfile_CPU.txt`](Dockerfile_CPU.txt) builds and runs `TensorSharp.Server`
on a Hugging Face **Docker** Space using the native **GGML CPU** backend. It is
modeled after the Seq2SeqSharp `Dockerfile_CPU.txt` pattern (clone from GitHub,
build .NET from source, download a model from the Hub at build time).

Target Space: <https://huggingface.co/spaces/zhongkaifu/tensorsharp>

## What the Dockerfile does

1. **Build stage** (`mcr.microsoft.com/dotnet/sdk:10.0`): installs `git`,
   `cmake`, and `build-essential`, clones TensorSharp, and runs
   `dotnet build TensorSharp.Server -c Release`. That transitively compiles the
   runtime/models/backends and the native GGML bridge (`libGgmlOps.so`). No
   CUDA toolchain is present, so ggml-cuda is auto-disabled and only the CPU
   backend is built.
2. **Runtime stage** (`mcr.microsoft.com/dotnet/aspnet:10.0`): installs
   `libgomp1` (OpenMP runtime for the GGML CPU kernels), reuses the image's
   UID‑1000 user (the UID Spaces run as) and makes the app tree writable by it,
   copies the build output, downloads a GGUF model (default:
   `gemma-4-12B-it-abliterated-uncensored` Q4_K_M, ~7.4 GB), and launches the
   server with `--backend ggml_cpu`.

### Port

`TensorSharp.Server` hard-codes its listen address to `http://0.0.0.0:5000` and
passes it to `app.Run`, which overrides `PORT`/`ASPNETCORE_URLS`. The build
stage rewrites that constant to **7860** (the Docker Space default port) with
`sed`, so the app binds the port the Space routes to — **no `app_port` setting
is required**. To use a different port, pass `--build-arg APP_PORT=<port>` and
set a matching `app_port` in the Space README.

## Configure the Space — copy `Dockerfile_CPU.txt` to `Dockerfile`

A Space builds the file named exactly `Dockerfile` at the repo root.

## Steps

```bash
# Clone your (empty) Space repo
git clone https://huggingface.co/spaces/zhongkaifu/tensorsharp
cd tensorsharp

# Add the Dockerfile (rename on copy) and the README with the YAML header below
cp /path/to/TensorSharp/TensorSharp.Server/Dockers/Dockerfile_CPU.txt Dockerfile
#   ...create README.md using the template below...

git add Dockerfile README.md
git commit -m "Deploy TensorSharp.Server (CPU)"
git push
```

## Space `README.md` template

Put this at the top of the Space's `README.md`. The app serves the chat UI at
`/`, so both the direct Space URL (`https://<user>-<space>.hf.space`) and the
embedded iframe open straight into the UI — no `base_path` override is needed.

```yaml
---
title: TensorSharp
emoji: 🔥
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
suggested_hardware: cpu-upgrade
short_description: C# LLM inference server (GGML CPU) with a chat UI + OpenAI/Ollama APIs
---

# TensorSharp

C# inference engine for GGUF LLMs. This Space hosts gemma-4-E2B-it
(abliterated, QAT Q4_0) with its multimodal projector on the native GGML CPU
backend and exposes the web chat UI plus OpenAI- and Ollama-compatible HTTP APIs.
See https://github.com/zhongkaifu/TensorSharp.
```

> The ~3.4 GB Q4_0 weights plus the ~1 GB projector fit comfortably in the
> `cpu-upgrade` tier (8 vCPU, 32 GB) and load on `cpu-basic` (2 vCPU, 16 GB)
> too. For a snappier text-only `cpu-basic` demo, switch to a smaller model via
> the build args below.

## Choosing a different model

The model is a build arg, so you can host any GGUF that TensorSharp supports
without editing the Dockerfile. On the Space, set **Settings → Variables and
secrets → Build args**, or build locally:

```bash
docker build -f Dockerfile \
  --build-arg MODEL_URL=https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf \
  --build-arg MODEL_FILE=Qwen3-1.7B-Q4_K_M.gguf \
  -t tensorsharp-server .
```

Model sizing guidance:

| Model | Size (Q4_K_M) | Notes |
|---|---|---|
| Qwen3-1.7B | ~1.1 GB | Fastest; best for a snappy `cpu-basic` demo |
| Qwen3-4B | ~2.5 GB | Good quality/speed balance on CPU |
| **gemma-4-E2B-it (abliterated, QAT)** (default) | ~3.4 GB | Multimodal; decodes acceptably on CPU |
| gemma-4-12B-it (abliterated) | ~7.4 GB | Higher quality; slow to decode on CPU — prefer `cpu-upgrade` |

The default gemma-4-E2B image already downloads the `mmproj` projector and passes
`--mmproj` in the `CMD`, so vision input works out of the box. To host a
text-only server, build with `--build-arg MMPROJ_FILE=` (empty) so the projector
download and `--mmproj` flag are dropped. For real multimodal throughput, use a
paid GPU hardware tier — base the image on a CUDA runtime, build the native
bridge with `bash build-linux.sh --cuda`, and run with `--backend ggml_cuda`.

## Troubleshooting: Space shows "TensorSharp.Server is running"

The app serves the chat UI at `/` (Program.cs calls `app.UseDefaultFiles()`, so
`/` is rewritten to `/index.html` before the plain-text health route runs). If
the Space root still shows the health string, it is running an **older image
built before this change** — re-copy the current `Dockerfile_CPU.txt` to the
Space's `Dockerfile` and trigger a rebuild (Settings → **Factory rebuild**). The
fix takes effect once the new image is built; no `base_path` override is needed.

Also note `GET /api/chat` returns 404 in a browser because it is a **POST**-only
endpoint — the web UI calls it with POST; it is not a deployment error.

## Verifying locally

```bash
docker build -f TensorSharp.Server/Dockers/Dockerfile_CPU.txt -t tensorsharp-server .
docker run --rm -p 7860:7860 tensorsharp-server
# open http://localhost:7860/index.html
# OpenAI-compatible:  POST http://localhost:7860/v1/chat/completions
# Ollama-compatible:  POST http://localhost:7860/api/chat/ollama
```

First boot loads the model and warms up kernels before serving; allow a minute
or two on CPU. The Space's `startup_duration_timeout` (default 30 min) is ample.
