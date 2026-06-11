# TensorSharp inference benchmark matrix

[English](inference_benchmark_matrix.md) | [中文](inference_benchmark_matrix_zh-cn.md)

Comparison of **TensorSharp**, **llama.cpp** and **Ollama** running the **same** GGUF files on the **same** machine, across text, prefill, decode and multimodal (image / audio / video) workloads.

TensorSharp is represented by four variants: three on the **GGML Metal** backend with three KV-cache dtypes (F32, F16, Q8_0) to measure the impact of KV-cache precision, and one on the **MLX** backend (mlx-c / Apple Metal, F32 KV) to compare against the GGML Metal path.

All engines were pointed at the same on-disk `.gguf` file. For Ollama this required registering a custom Modelfile (`ts-gemma4-e4b-q8`) so that no quantisation differences would skew the comparison.

This is a captured benchmark snapshot, not an automatically regenerated report
for the current working tree. Re-run the scripts in
`benchmarks/inference_matrix/scripts/` to refresh the numbers for a new commit,
backend version, or machine.

This snapshot predates DiffusionGemma coverage and should not be read as the
current support matrix. See the root README and
[`docs/models/diffusiongemma.md`](models/diffusiongemma.md) for the current
text-diffusion implementation status.

## TL;DR

Headline numbers on Gemma 4 E4B Q8_0, decode throughput on a real text prompt (`long_text`, ~1043-token prompt -> 64 tokens generated), in tokens/second. TensorSharp is represented by four backend / KV-dtype variants (three on the GGML Metal backend with F32 / F16 / Q8 KV, and one on the MLX backend with F32 KV).

| TensorSharp (ggml_metal, F32) | TensorSharp (ggml_metal, F16) | TensorSharp (ggml_metal, Q8) | TensorSharp (mlx, F32) | llama.cpp | Ollama |
|----:|----:|----:|----:|----:|----:|
| 30.3 | 27.7 | 28.6 | 11.1 | 40.2 | 40.6 |

And prefill throughput on the synthetic 2048-token prompt (`pp2048`):

| TensorSharp (ggml_metal, F32) | TensorSharp (ggml_metal, F16) | TensorSharp (ggml_metal, Q8) | TensorSharp (mlx, F32) | llama.cpp | Ollama |
|----:|----:|----:|----:|----:|----:|
| 413.4 | 420.5 | 428.6 | 405.1 | 767.4 | 770.0 |

## 1. Software versions

| Engine | Version / build |
|--------|-----------------|
| TensorSharp | captured at git `90c14f4`, .NET 10.0.103, ggml_metal and mlx backends |
| llama.cpp   | brew package, version: 8990 (660b1b4bd) (Metal + BLAS) |
| Ollama      | 0.23.4 |


## 2. Model under test

| Short id | GGUF file | Family | Notes |
|---|---|---|---|
| `gemma4` | `gemma-4-E4B-it-Q8_0.gguf` | Gemma 4 (8 B dense) | Q8_0; vision + audio + video projector available (`gemma-4-mmproj-F16.gguf`) |

## 3. Tasks

| Task | Description |
|------|-------------|
| `pp512` | Synthetic prefill (512 tok) |
| `tg128` | Synthetic decode (128 tok) |
| `pp2048` | Synthetic prefill (2048 tok) |
| `short_text` | Short text prompt (~32 tok in / 64 tok out) |
| `long_text` | Long text prompt (~1043 tok in / 64 tok out) |
| `image` | Image: apple.png (~282 tok in / 64 tok out) |
| `audio` | Audio: 45 s of speech (~1148 tok in / 64 tok out) |
| `video` | Video: concert clip (frames -> tokens / 64 tok out) |

## 4. Methodology

For each `(engine, model, task)` cell we recorded both **prefill throughput** (tokens/second of the prompt-processing phase) and **decode throughput** (tokens/second of the autoregressive generation phase). Numbers are taken from the engines' own internal timers, not from wall-clock around the process (so model-load and warmup are excluded).

**Warm-up is excluded for every engine.** On Apple Metal the first prefill of a given batch shape pays a non-trivial pipeline-JIT cost that is not representative of steady-state speed.

Per-engine details:

- **TensorSharp** uses its `--benchmark` mode for the synthetic `pp*`/`tg*` tasks (3 runs each, best-run is reported). For real text and multimodal tasks it uses `--warmup-runs 1`. The three `ggml_metal` rows differ only in `--kv-cache-dtype` (`f32`, `f16`, `q8_0`) so each row measures KV-cache precision in isolation; the `mlx` row uses `--backend mlx` with F32 KV.
- **llama.cpp** uses `llama-bench -p N -n M -r 3` for synthetic, `llama-cli` for real text (`-st --no-warmup --no-display-prompt --jinja`), and `llama-mtmd-cli` for image/audio (with `--no-warmup`).
- **Ollama** is driven through its HTTP `/api/generate` endpoint. Before each timed request a 1-token `keep_alive` ping ensures the model is already resident.

Sampling is greedy (`temperature=0`) everywhere. The driver source is `benchmarks/inference_matrix/scripts/run_bench.py` and the raw per-cell JSON outputs are in `benchmarks/inference_matrix/results/`.

## 5. Full results — Gemma 4 E4B Q8_0

All numbers are tokens / second (higher is better). `—` means the metric does not apply to that task, `n/a` means the engine does not support that combination, and `fail` means the engine errored out at runtime.

| Task | TensorSharp (ggml_metal, F32 KV) prefill | TensorSharp (ggml_metal, F32 KV) decode | TensorSharp (ggml_metal, F16 KV) prefill | TensorSharp (ggml_metal, F16 KV) decode | TensorSharp (ggml_metal, Q8 KV) prefill | TensorSharp (ggml_metal, Q8 KV) decode | TensorSharp (mlx, F32 KV) prefill | TensorSharp (mlx, F32 KV) decode | llama.cpp prefill | llama.cpp decode | Ollama prefill | Ollama decode |
|------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| Prefill 512 | 391.0 | — | 429.0 | — | 418.3 | — | 543.5 | — | 782.6 | — | 451.6 | — |
| Decode 128 | 215.4 | 33.3 | 235.5 | 33.8 | 247.6 | 32.0 | 353.1 | 22.5 | 470.0 | 40.3 | 102.2 | 40.9 |
| Prefill 2048 | 413.4 | — | 420.5 | — | 428.6 | — | 405.1 | — | 767.4 | — | 770.0 | — |
| Short text | 216.3 | 32.5 | 235.6 | 32.8 | 253.8 | 31.5 | 64.3 | 22.3 | 69.8 | 41.8 | 240.0 | 41.3 |
| Long text | 411.8 | 30.3 | 412.2 | 27.7 | 422.9 | 28.6 | 499.7 | 11.1 | 643.3 | 40.2 | 753.6 | 40.6 |
| Image | 151.6 | 32.5 | 151.4 | 32.8 | fail | fail | 199.6 | 12.7 | 354.1 | 40.4 | 207.5 | 41.3 |
| Audio | 127.0 | 29.3 | 121.0 | 27.5 | fail | fail | 479.1 | 11.2 | 610.1 | 38.6 | n/a | n/a |
| Video | 135.7 | 30.5 | 127.8 | 27.7 | fail | fail | 485.0 | 11.2 | n/a | n/a | n/a | n/a |

## 6. TensorSharp KV cache dtype comparison

How does the KV cache data type affect TensorSharp throughput? All numbers are tokens / second for Gemma 4 E4B Q8_0.

| Task | F32 prefill | F32 decode | F16 prefill | F16 decode | Q8_0 prefill | Q8_0 decode |
|------|----:|----:|----:|----:|----:|----:|
| Prefill 512 | 391.0 | — | 429.0 | — | 418.3 | — |
| Decode 128 | 215.4 | 33.3 | 235.5 | 33.8 | 247.6 | 32.0 |
| Prefill 2048 | 413.4 | — | 420.5 | — | 428.6 | — |
| Short text | 216.3 | 32.5 | 235.6 | 32.8 | 253.8 | 31.5 |
| Long text | 411.8 | 30.3 | 412.2 | 27.7 | 422.9 | 28.6 |
| Image | 151.6 | 32.5 | 151.4 | 32.8 | fail | fail |
| Audio | 127.0 | 29.3 | 121.0 | 27.5 | fail | fail |
| Video | 135.7 | 30.5 | 127.8 | 27.7 | fail | fail |

## 7. TensorSharp vs. llama.cpp (relative throughput)

Each cell is `TensorSharp t/s / llama.cpp t/s`. 100% means parity; >100% means TensorSharp is faster.

**Prefill**

| KV dtype | Prefill 512 | Prefill 2048 | Long text | Image | Audio |
|---------|----:|----:|----:|----:|----:|
| F32 | 50% | 54% | 64% | 43% | 21% |
| F16 | 55% | 55% | 64% | 43% | 20% |
| Q8_0 | 53% | 56% | 66% | — | — |

**Decode**

| KV dtype | Decode 128 | Short text | Long text | Image | Audio |
|---------|----:|----:|----:|----:|----:|
| F32 | 83% | 78% | 75% | 80% | 76% |
| F16 | 84% | 78% | 69% | 81% | 71% |
| Q8_0 | 79% | 75% | 71% | — | — |

## 8. TensorSharp vs. Ollama (relative throughput)

**Prefill**

| KV dtype | Prefill 512 | Prefill 2048 | Long text | Image |
|---------|----:|----:|----:|----:|
| F32 | 87% | 54% | 55% | 73% |
| F16 | 95% | 55% | 55% | 73% |
| Q8_0 | 93% | 56% | 56% | — |

**Decode**

| KV dtype | Decode 128 | Short text | Long text | Image |
|---------|----:|----:|----:|----:|
| F32 | 81% | 79% | 75% | 79% |
| F16 | 83% | 79% | 68% | 79% |
| Q8_0 | 78% | 76% | 70% | — |

## 9. TensorSharp MLX backend

TensorSharp also exposes an MLX backend ([mlx-c](https://github.com/ml-explore/mlx-c) / Apple Metal). It uses a separate kernel set from the GGML Metal path and a different allocator (`MlxAllocator`), so it's tracked as its own column rather than rolled into the GGML Metal results. Activate it with `--backend mlx` in the CLI / server.

MLX vs. GGML Metal (F32 KV) on the same tasks (tokens / second, higher is better):

| Task | TensorSharp ggml_metal (F32) prefill | TensorSharp ggml_metal (F32) decode | TensorSharp mlx (F32) prefill | TensorSharp mlx (F32) decode |
|------|----:|----:|----:|----:|
| Prefill 512 | 391.0 | — | 543.5 | — |
| Decode 128 | 215.4 | 33.3 | 353.1 | 22.5 |
| Prefill 2048 | 413.4 | — | 405.1 | — |
| Short text | 216.3 | 32.5 | 64.3 | 22.3 |
| Long text | 411.8 | 30.3 | 499.7 | 11.1 |
| Image | 151.6 | 32.5 | 199.6 | 12.7 |
| Audio | 127.0 | 29.3 | 479.1 | 11.2 |
| Video | 135.7 | 30.5 | 485.0 | 11.2 |

## 10. Reproducing this report

```bash
# 1. register the Gemma4 GGUF inside Ollama (one-time)
cd benchmarks/inference_matrix
ollama create ts-gemma4-e4b-q8 -f modelfiles/Modelfile.gemma4-e4b-q8

# 2. drop required media files in <repo>/data/ (apple.png, obama_first_45_secs.mp3, concert.mp4)
mkdir -p ../../data && ln -sf <your-path>/apple.png ../../data/apple.png  # etc.

# 3. (re)build the TensorSharp CLI
dotnet build ../../TensorSharp.Cli/TensorSharp.Cli.csproj -c Release

# 4. run the matrix (one engine at a time keeps the GPU happy)
python3 scripts/run_bench.py --engines ollama
python3 scripts/run_bench.py --engines llamacpp
python3 scripts/run_bench.py --engines tensorsharp_f32
python3 scripts/run_bench.py --engines tensorsharp_f16
python3 scripts/run_bench.py --engines tensorsharp_q80
python3 scripts/run_bench.py --engines tensorsharp_mlx

# 5. regenerate this report from the JSON results
python3 scripts/build_report.py
```

Driver: `benchmarks/inference_matrix/scripts/run_bench.py`  
Report generator: `benchmarks/inference_matrix/scripts/build_report.py`  
Per-cell raw JSON is generated at run time under `benchmarks/inference_matrix/results/<engine>__<model>__<task>.json`.
