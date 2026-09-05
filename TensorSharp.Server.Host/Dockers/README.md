# Deploying TensorSharp.Server.Host to a Hugging Face Space (Docker, CPU)

[`Dockerfile_CPU.txt`](Dockerfile_CPU.txt) builds and runs `TensorSharp.Server.Host`
on a Hugging Face **Docker** Space using the native **GGML CPU** backend. It is
modeled after the Seq2SeqSharp `Dockerfile_CPU.txt` pattern (clone from GitHub,
build .NET from source, download a model from the Hub at build time).

Target Space: <https://huggingface.co/spaces/zhongkaifu/tensorsharp>

## What the Dockerfile does

1. **Build stage** (`mcr.microsoft.com/dotnet/sdk:10.0`): installs `git`,
   `cmake`, and `build-essential`, clones TensorSharp, and runs
   `dotnet build TensorSharp.Server.Host -c Release`. That transitively compiles the
   runtime/models/backends and the native GGML bridge (`libGgmlOps.so`). No
   CUDA toolchain is present, so ggml-cuda is auto-disabled and only the CPU
   backend is built.
2. **Runtime stage** (`mcr.microsoft.com/dotnet/aspnet:10.0`): installs
   `libgomp1` (OpenMP runtime for the GGML CPU kernels), reuses the image's
   UID‑1000 user (the UID Spaces run as) and makes the app tree writable by it,
   copies the build output, downloads the default Huihui Gemma 4 E2B QAT Q4_K
   GGUF plus its `mmproj` projector, and launches the server with
   `--backend ggml_cpu`.

### Port

`TensorSharp.Server.Host` listens on `http://0.0.0.0:5000` by default and honours the
`PORT` environment variable (as well as `--port` / `--host` / `--urls` on the
command line). The runtime stage sets `PORT=7860` — the Docker Space default
port — so the app binds the port the Space routes to and **no `app_port` setting
is required**.

To use a different port, either rebuild with `--build-arg APP_PORT=<port>` and
set a matching `app_port` in the Space README, or override it at run time on any
other host:

```bash
docker run --rm -e PORT=8080 -p 8080:8080 tensorsharp-server
```

## Configure the Space — copy `Dockerfile_CPU.txt` to `Dockerfile`

A Space builds the file named exactly `Dockerfile` at the repo root.

## Steps

```bash
# Clone your (empty) Space repo
git clone https://huggingface.co/spaces/zhongkaifu/tensorsharp
cd tensorsharp

# Add the Dockerfile (rename on copy) and the README with the YAML header below
cp /path/to/TensorSharp/TensorSharp.Server.Host/Dockers/Dockerfile_CPU.txt Dockerfile
#   ...create README.md using the template below...

git add Dockerfile README.md
git commit -m "Deploy TensorSharp.Server.Host (CPU)"
git push
```

## Space `README.md` template

Put this at the top of the Space's `README.md`. In the current server contract,
`GET /` serves the chat UI, so the bare Space root
(`https://<user>-<space>.hf.space/`) is the UI link; `/index.html` still works
as an explicit alias. The plain liveness response
(`"TensorSharp.Server.Host is running"`) lives at `/health`.

```yaml
---
title: TensorSharp
emoji: 🔥
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
suggested_hardware: cpu-upgrade
short_description: C# GGUF inference server (GGML CPU) with a chat UI + OpenAI/Ollama APIs
---

# TensorSharp

C# inference engine for GGUF language models. This Space hosts gemma-4-E2B-it
(abliterated, QAT Q4_K) with its multimodal projector on the native GGML CPU
backend and exposes the web chat UI plus OpenAI- and Ollama-compatible HTTP APIs.
See https://github.com/zhongkaifu/TensorSharp.
```

> The ~3.4 GB Q4_K weights plus the ~1 GB projector fit comfortably in the
> `cpu-upgrade` tier (8 vCPU, 32 GB) and load on `cpu-basic` (2 vCPU, 16 GB)
> too. For a snappier text-only `cpu-basic` demo, switch to a smaller model via
> the build args below.

## Choosing a different model

The model is a build arg, so you can host any GGUF that TensorSharp supports
without editing the Dockerfile. On the Space, set **Settings → Variables and
secrets → Build args**, or build locally:

```bash
docker build -f Dockerfile \
  --build-arg 'MODEL_URL=https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf?download=true' \
  --build-arg MODEL_FILE=gemma-4-E4B-it-Q4_K_M.gguf \
  --build-arg MMPROJ_FILE= \
  -t tensorsharp-server .
```

This hosts the verified gemma-4-E4B-it at the size-friendly Q4_K_M quantization
([model repo](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)); for higher
quality, the same repo's `gemma-4-E4B-it-Q8_0.gguf` (7.48 GiB) works too.
The empty `MMPROJ_FILE` makes this a text-only build; the server never
auto-detects a general multimodal projector. For vision input, point
`MMPROJ_URL`/`MMPROJ_FILE` at the repo's `mmproj-gemma-4-E4B-it-Q8_0.gguf`.

Model sizing guidance:

| Model | Download size | Notes |
|---|---|---|
| **gemma-4-E2B-it (abliterated, QAT)** (default) | ~3.4 GB | Multimodal; decodes acceptably on CPU |
| gemma-4-12B-it (abliterated) | ~7.4 GB | Higher quality; slow to decode on CPU — prefer `cpu-upgrade` |
| Qwen3.5-9B Q8_0 | ~8.9 GB | Text-only; highest-quality local quant, but slow on CPU — prefer GPU acceleration |

The default gemma-4-E2B image already downloads the `mmproj` projector and passes
`--mmproj` in the `CMD`, so vision input works out of the box. To host a
text-only server, build with `--build-arg MMPROJ_FILE=` (empty) so the projector
download and `--mmproj` flag are dropped. For real multimodal throughput, use a
paid GPU hardware tier — base the image on a CUDA runtime, build the native
bridge with `bash build-linux.sh --cuda`, and run with `--backend ggml_cuda`.
For a DiffusionGemma GGUF, also leave `MMPROJ_FILE` empty; the Web UI exposes
live denoising previews while the compatibility APIs return final text.

## Space root shows "TensorSharp.Server.Host is running"

That means the image is serving without the Web UI assets: `GET /` sends
`wwwroot/index.html` when it is present and falls back to the liveness response
when it is not. Check that `wwwroot/` was published into the application
directory the container launches from, as both Dockerfiles do — static
image/CSS paths depend on the same thing. The liveness response is always
available at `/health`.

Also note `GET /api/chat` returns 404 in a browser because it is a **POST**-only
endpoint — the web UI calls it with POST; it is not a deployment error.

## Verifying locally

```bash
docker build -f TensorSharp.Server.Host/Dockers/Dockerfile_CPU.txt -t tensorsharp-server .
docker run --rm -p 7860:7860 tensorsharp-server
# open http://localhost:7860

# Use the exact MODEL_FILE configured at image build time.
curl -s http://localhost:7860/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Huihui-gemma-4-E2B-it-qat-q4_0-unquantized-abliterated-Q4_K","messages":[{"role":"user","content":"Reply with one short hello."}],"max_tokens":32}'
```

First boot loads the model and warms up kernels before serving; allow a minute
or two on CPU. The Space's `startup_duration_timeout` (default 30 min) is ample.

---

# Deploying to a GPU Space ([`Dockerfile_GPU.txt`](Dockerfile_GPU.txt))

[`Dockerfile_GPU.txt`](Dockerfile_GPU.txt) is the GPU counterpart of the CPU
file. It builds and runs `TensorSharp.Server.Host` on a **paid GPU** Docker Space
using the native **GGML CUDA** backend (`--backend ggml_cuda`, the most-tested
NVIDIA path).

## What differs from the CPU Dockerfile

1. **Build stage** is based on `nvidia/cuda:<ver>-devel-ubuntu22.04` (provides
   `nvcc` + cuBLAS headers) with the .NET 10 SDK installed via the official
   script. `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` makes `dotnet build` compile
   the native GGML bridge with the **ggml-cuda** backend (it runs
   `build-linux.sh --cuda`), and the direct-CUDA PTX kernels are compiled too.
2. **Runtime stage** is based on `nvidia/cuda:<ver>-runtime-ubuntu22.04`
   (provides the CUDA runtime + cuBLAS shared libraries the bridge loads at
   runtime), with the ASP.NET Core 10 runtime installed via the script. It
   creates the UID‑1000 user Spaces run as, downloads the model, and launches
   with `--backend ggml_cuda`.

The `PORT=7860` default and the chat UI at `/` work exactly as in the CPU file;
`/health` remains the liveness endpoint.

## CUDA architectures (build host has no GPU)

The Space build host has no GPU, so the CUDA targets must be set explicitly via
the `CUDA_ARCHS` build arg (default `75-virtual`). The default emits **compute_75
PTX only** — no SASS — which the driver JIT-compiles at first model load for
whatever GPU runs the Space, so the one image runs on **every** current HF GPU
tier. PTX-only is also the lightest build: it skips `ptxas` (the memory-hungry
SASS pass), which is what makes the full ggml-cuda compile OOM-kill the build
container (exit 137).

For native SASS (marginally faster steady-state, heavier build) set it to your
GPU's `-real` arch instead:

| HF GPU tier | GPU | Arch | `CUDA_ARCHS` for native speed |
|---|---|---|---|
| `t4-small` / `t4-medium` | Tesla T4 | 75 | `75-real` |
| `a10g-small` / `a10g-large` | A10G | 86 | `86-real` |
| `a100-large` | A100 | 80 | `80-real` |
| `l4x1` / `l40sx1` | L4 / L40S | 89 | `89-real` |
| (Hopper) | H100 | 90 | `90-real` |

```bash
docker build -f Dockerfile \
  --build-arg CUDA_ARCHS=86-real \
  -t tensorsharp-server-gpu .
```

Pin the CUDA toolkit with `--build-arg CUDA_VERSION=12.4.1` (the `-devel` and
`-runtime` tags share it). 12.2.2 (default) JIT-compiles on any driver ≥ 535.

## Build OOM-killed (exit code 137 / `OOMKilled`)

The native ggml-cuda compile is memory-hungry, and `build-linux.sh` sizes its
parallel `nvcc` job count from `/proc/meminfo` — which reports the build host's
total RAM, not the build container's cgroup limit — so on a large host it
over-subscribes memory and the container is OOM-killed. The Dockerfile defaults
guard against this by (a) building PTX-only (`CUDA_ARCHS=75-virtual`, no `ptxas`)
and (b) capping parallelism with `GGML_BUILD_JOBS=2`. If a build still OOMs:

```bash
# Serialize the native compile (slowest but lowest peak memory)
docker build -f Dockerfile --build-arg GGML_BUILD_JOBS=1 -t tensorsharp-server-gpu .
```

and keep `CUDA_ARCHS=75-virtual` (don't add `-real` arches, which re-enable the
heavy `ptxas` pass). On a Space, set these under
Settings → **Variables and secrets → Build args**.

## Configure the Space

Copy `Dockerfile_GPU.txt` to `Dockerfile`, pick a **GPU** tier under
Settings → Hardware (a CPU tier builds but fails to start — no CUDA device), and
push. The CUDA compile makes the build noticeably longer than the CPU image.

### Space `README.md` template (GPU)

```yaml
---
title: TensorSharp
emoji: 🔥
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
suggested_hardware: t4-small
short_description: C# GGUF inference server (GGML CUDA) with a chat UI + OpenAI/Ollama APIs
---

# TensorSharp

C# inference engine for GGUF language models, GPU-accelerated via the native GGML CUDA
backend. This Space hosts gemma-4-E2B-it (abliterated, QAT Q4_K) with its
multimodal projector and exposes the web chat UI plus OpenAI- and
Ollama-compatible HTTP APIs. See https://github.com/zhongkaifu/TensorSharp.
```

> Set `suggested_hardware` to the tier you actually selected (`t4-small`,
> `a10g-small`, `a100-large`, …). The default gemma-4-E2B (~3.4 GB + ~1 GB
> projector) fits the 16 GB T4; on a larger GPU host a bigger model via the
> `MODEL_URL`/`MODEL_FILE` build args (see the CPU section above).

## Verifying locally (needs an NVIDIA GPU + nvidia-container-toolkit)

```bash
docker build -f TensorSharp.Server.Host/Dockers/Dockerfile_GPU.txt \
  --build-arg CUDA_ARCHS=75-real -t tensorsharp-server-gpu .
docker run --rm --gpus all -p 7860:7860 tensorsharp-server-gpu
# open http://localhost:7860
```
