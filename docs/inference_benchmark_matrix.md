# TensorSharp inference benchmark matrix

Comparison of **TensorSharp**, **llama.cpp** and **Ollama** running the **same** GGUF files on the **same** machine, across text, prefill, decode and multimodal (image / audio / video) workloads.

TensorSharp is tested with three KV cache data types (F32, F16, Q8_0) to measure the performance impact of KV cache precision.

All three engines were pointed at the same on-disk `.gguf` files. For Ollama this required registering a custom Modelfile (`ts-gemma4-e4b-q8`) so that no quantisation differences would skew the comparison.

## TL;DR

Headline numbers on Gemma 4 E4B Q8_0, decode throughput on a real text prompt (`long_text`, ~1043-token prompt -> 64 tokens generated), in tokens/second:

| TensorSharp (F32 KV) | TensorSharp (F16 KV) | TensorSharp (Q8 KV) | llama.cpp | Ollama |
|----:|----:|----:|----:|----:|
| 28.9 | 30.0 | 29.8 | 27.2 | 36.3 |

And prefill throughput on the synthetic 2048-token prompt (`pp2048`):

| TensorSharp (F32 KV) | TensorSharp (F16 KV) | TensorSharp (Q8 KV) | llama.cpp | Ollama |
|----:|----:|----:|----:|----:|
| 331.3 | 374.9 | 376.8 | 701.1 | 740.1 |

## 1. Software versions

| Engine | Version / build |
|--------|-----------------|
| TensorSharp | git `89593aa` (this repo), .NET 10.0.103, ggml_metal backend |
| llama.cpp   | brew package, version: 8990 (660b1b4bd) (Metal + BLAS) |
| Ollama      | 0.21.2 |


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

- **TensorSharp** uses its `--benchmark` mode for the synthetic `pp*`/`tg*` tasks (3 runs each, best-run is reported). For real text and multimodal tasks it uses `--warmup-runs 1`. Each TensorSharp variant is run with a specific `--kv-cache-dtype` (f32, f16, or q8_0) to measure the impact of KV cache precision on throughput.
- **llama.cpp** uses `llama-bench -p N -n M -r 3` for synthetic, `llama-cli` for real text (`-st --no-warmup --no-display-prompt --jinja`), and `llama-mtmd-cli` for image/audio (with `--no-warmup`).
- **Ollama** is driven through its HTTP `/api/generate` endpoint. Before each timed request a 1-token `keep_alive` ping ensures the model is already resident.

Sampling is greedy (`temperature=0`) everywhere. The driver source is `benchmarks/inference_matrix/scripts/run_bench.py` and the raw per-cell JSON outputs are in `benchmarks/inference_matrix/results/`.

## 5. Full results — Gemma 4 E4B Q8_0

All numbers are tokens / second (higher is better). `—` means the metric does not apply to that task, `n/a` means the engine does not support that combination, and `fail` means the engine errored out at runtime.

| Task | TensorSharp (F32 KV) prefill | TensorSharp (F32 KV) decode | TensorSharp (F16 KV) prefill | TensorSharp (F16 KV) decode | TensorSharp (Q8 KV) prefill | TensorSharp (Q8 KV) decode | llama.cpp prefill | llama.cpp decode | Ollama prefill | Ollama decode |
|------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| Prefill 512 | 268.1 | — | 351.6 | — | 391.2 | — | 762.3 | — | 671.2 | — |
| Decode 128 | 30.2 | 33.0 | 119.5 | 32.7 | 152.5 | 31.0 | 463.1 | 28.7 | 454.7 | 37.0 |
| Prefill 2048 | 331.3 | — | 374.9 | — | 376.8 | — | 701.1 | — | 740.1 | — |
| Short text | 23.6 | 30.8 | 104.6 | 30.8 | 139.8 | 29.7 | 209.1 | 25.5 | 251.4 | 36.2 |
| Long text | 285.9 | 28.9 | 316.5 | 30.0 | 380.3 | 29.8 | 688.3 | 27.2 | 736.0 | 36.3 |
| Image | 143.3 | 20.2 | 146.2 | 25.2 | fail | fail | 416.0 | 29.9 | 197.8 | 37.0 |
| Audio | 130.2 | 19.0 | 130.9 | 23.2 | fail | fail | 612.5 | 36.6 | n/a | n/a |
| Video | 126.1 | 19.0 | 128.0 | 24.4 | fail | fail | n/a | n/a | n/a | n/a |

## 6. TensorSharp KV cache dtype comparison

How does the KV cache data type affect TensorSharp throughput? All numbers are tokens / second for Gemma 4 E4B Q8_0.

| Task | F32 prefill | F32 decode | F16 prefill | F16 decode | Q8_0 prefill | Q8_0 decode |
|------|----:|----:|----:|----:|----:|----:|
| Prefill 512 | 268.1 | — | 351.6 | — | 391.2 | — |
| Decode 128 | 30.2 | 33.0 | 119.5 | 32.7 | 152.5 | 31.0 |
| Prefill 2048 | 331.3 | — | 374.9 | — | 376.8 | — |
| Short text | 23.6 | 30.8 | 104.6 | 30.8 | 139.8 | 29.7 |
| Long text | 285.9 | 28.9 | 316.5 | 30.0 | 380.3 | 29.8 |
| Image | 143.3 | 20.2 | 146.2 | 25.2 | fail | fail |
| Audio | 130.2 | 19.0 | 130.9 | 23.2 | fail | fail |
| Video | 126.1 | 19.0 | 128.0 | 24.4 | fail | fail |

## 7. TensorSharp vs. llama.cpp (relative throughput)

Each cell is `TensorSharp t/s / llama.cpp t/s`. 100% means parity; >100% means TensorSharp is faster.

**Prefill**

| KV dtype | Prefill 512 | Prefill 2048 | Long text | Image | Audio |
|---------|----:|----:|----:|----:|----:|
| F32 | 35% | 47% | 42% | 34% | 21% |
| F16 | 46% | 53% | 46% | 35% | 21% |
| Q8_0 | 51% | 54% | 55% | — | — |

**Decode**

| KV dtype | Decode 128 | Short text | Long text | Image | Audio |
|---------|----:|----:|----:|----:|----:|
| F32 | 115% | 121% | 106% | 68% | 52% |
| F16 | 114% | 121% | 110% | 84% | 63% |
| Q8_0 | 108% | 116% | 110% | — | — |

## 8. TensorSharp vs. Ollama (relative throughput)

**Prefill**

| KV dtype | Prefill 512 | Prefill 2048 | Long text | Image |
|---------|----:|----:|----:|----:|
| F32 | 40% | 45% | 39% | 72% |
| F16 | 52% | 51% | 43% | 74% |
| Q8_0 | 58% | 51% | 52% | — |

**Decode**

| KV dtype | Decode 128 | Short text | Long text | Image |
|---------|----:|----:|----:|----:|
| F32 | 89% | 85% | 80% | 55% |
| F16 | 88% | 85% | 83% | 68% |
| Q8_0 | 84% | 82% | 82% | — |

## 9. Reproducing this report

```bash
# 1. register the Gemma4 GGUF inside Ollama
cd benchmarks/inference_matrix
ollama create ts-gemma4-e4b-q8 -f modelfiles/Modelfile.gemma4-e4b-q8

# 2. (re)build the TensorSharp CLI
dotnet build ../../TensorSharp.Cli/TensorSharp.Cli.csproj -c Release

# 3. run the matrix (one engine at a time keeps the GPU happy)
python3 scripts/run_bench.py --engines ollama
python3 scripts/run_bench.py --engines llamacpp
python3 scripts/run_bench.py --engines tensorsharp_f32
python3 scripts/run_bench.py --engines tensorsharp_f16
python3 scripts/run_bench.py --engines tensorsharp_q80

# 4. regenerate this report from the JSON results
python3 scripts/build_report.py
```

Driver: `benchmarks/inference_matrix/scripts/run_bench.py`  
Report generator: `benchmarks/inference_matrix/scripts/build_report.py`  
Per-cell raw JSON: `benchmarks/inference_matrix/results/<engine>__<model>__<task>.json`
