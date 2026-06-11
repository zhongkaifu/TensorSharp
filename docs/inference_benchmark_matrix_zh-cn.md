# TensorSharp 推理基准矩阵

[English](inference_benchmark_matrix.md) | [中文](inference_benchmark_matrix_zh-cn.md)

本文对比 **TensorSharp**、**llama.cpp** 与 **Ollama** 在同一台机器、同一份
GGUF 文件上的文本、prefill、decode 以及多模态（图像 / 音频 / 视频）工作负载。

TensorSharp 用四个变体表示：三个 **GGML Metal** 后端变体分别使用 F32、F16、
Q8_0 KV cache，用来衡量 KV 精度影响；另一个使用 **MLX** 后端（mlx-c /
Apple Metal，F32 KV），用于和 GGML Metal 路径对比。

所有引擎都指向同一份磁盘 `.gguf` 文件。Ollama 需要注册自定义 Modelfile
（`ts-gemma4-e4b-q8`），以避免量化差异影响对比。

这是一次捕获的基准快照，并不是当前工作树自动重跑生成的报告。若要刷新到新的
commit、后端版本或机器，请重新运行 `benchmarks/inference_matrix/scripts/`
中的脚本。

该快照早于 DiffusionGemma 覆盖，不应当视为当前支持矩阵。当前文本扩散实现状态见
项目根 README 与 [`docs/models/diffusiongemma_zh-cn.md`](models/diffusiongemma_zh-cn.md)。

## TL;DR

Gemma 4 E4B Q8_0 在真实长文本 prompt（`long_text`，约 1043-token prompt ->
生成 64 tokens）上的 decode 吞吐，单位 tokens/second：

| TensorSharp (ggml_metal, F32) | TensorSharp (ggml_metal, F16) | TensorSharp (ggml_metal, Q8) | TensorSharp (mlx, F32) | llama.cpp | Ollama |
|----:|----:|----:|----:|----:|----:|
| 30.3 | 27.7 | 28.6 | 11.1 | 40.2 | 40.6 |

合成 2048-token prompt（`pp2048`）上的 prefill 吞吐：

| TensorSharp (ggml_metal, F32) | TensorSharp (ggml_metal, F16) | TensorSharp (ggml_metal, Q8) | TensorSharp (mlx, F32) | llama.cpp | Ollama |
|----:|----:|----:|----:|----:|----:|
| 413.4 | 420.5 | 428.6 | 405.1 | 767.4 | 770.0 |

## 1. 软件版本

| 引擎 | 版本 / 构建 |
|---|---|
| TensorSharp | 捕获于 git `90c14f4`，.NET 10.0.103，ggml_metal 与 mlx 后端 |
| llama.cpp | brew package，version: 8990 (660b1b4bd) (Metal + BLAS) |
| Ollama | 0.23.4 |

## 2. 测试模型

| 简称 | GGUF 文件 | 家族 | 说明 |
|---|---|---|---|
| `gemma4` | `gemma-4-E4B-it-Q8_0.gguf` | Gemma 4 (8 B dense) | Q8_0；有视觉 / 音频 / 视频 projector（`gemma-4-mmproj-F16.gguf`） |

## 3. 任务

| 任务 | 说明 |
|---|---|
| `pp512` | 合成 prefill（512 tok） |
| `tg128` | 合成 decode（128 tok） |
| `pp2048` | 合成 prefill（2048 tok） |
| `short_text` | 短文本 prompt（约 32 tok in / 64 tok out） |
| `long_text` | 长文本 prompt（约 1043 tok in / 64 tok out） |
| `image` | 图像：apple.png（约 282 tok in / 64 tok out） |
| `audio` | 音频：45 秒语音（约 1148 tok in / 64 tok out） |
| `video` | 视频：演唱会片段（frames -> tokens / 64 tok out） |

## 4. 方法

每个 `(engine, model, task)` cell 记录 **prefill throughput**（prompt 处理阶段
tokens/second）和 **decode throughput**（自回归生成阶段 tokens/second）。
数字来自各引擎内部计时，不包含模型加载与 warmup。

**所有引擎都排除了 warmup。** 在 Apple Metal 上，某个 batch shape 的第一次
prefill 会支付不可忽略的 pipeline JIT 成本，不代表稳态速度。

- **TensorSharp**：合成 `pp*` / `tg*` 任务使用 `--benchmark`（每项 3 次，取最佳）。
  文本与多模态任务使用 `--warmup-runs 1`。三个 `ggml_metal` 行只改变
  `--kv-cache-dtype`；`mlx` 行使用 `--backend mlx` 与 F32 KV。
- **llama.cpp**：合成任务使用 `llama-bench -p N -n M -r 3`，真实文本使用
  `llama-cli`（`-st --no-warmup --no-display-prompt --jinja`），图像 / 音频使用
  `llama-mtmd-cli`（`--no-warmup`）。
- **Ollama**：通过 HTTP `/api/generate` 驱动。每个计时请求前先做一次 1-token
  `keep_alive` ping，确保模型已驻留。

所有场景均使用贪心采样（`temperature=0`）。驱动脚本是
`benchmarks/inference_matrix/scripts/run_bench.py`；每个 cell 的 JSON 输出会在运行时
写入 `benchmarks/inference_matrix/results/`。

## 5. 完整结果 - Gemma 4 E4B Q8_0

所有数字为 tokens / second（越高越好）。`—` 表示该任务不适用该指标，`n/a`
表示引擎不支持该组合，`fail` 表示运行时失败。

| Task | TensorSharp (ggml_metal, F32 KV) prefill | TensorSharp (ggml_metal, F32 KV) decode | TensorSharp (ggml_metal, F16 KV) prefill | TensorSharp (ggml_metal, F16 KV) decode | TensorSharp (ggml_metal, Q8 KV) prefill | TensorSharp (ggml_metal, Q8 KV) decode | TensorSharp (mlx, F32 KV) prefill | TensorSharp (mlx, F32 KV) decode | llama.cpp prefill | llama.cpp decode | Ollama prefill | Ollama decode |
|---|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| Prefill 512 | 391.0 | — | 429.0 | — | 418.3 | — | 543.5 | — | 782.6 | — | 451.6 | — |
| Decode 128 | 215.4 | 33.3 | 235.5 | 33.8 | 247.6 | 32.0 | 353.1 | 22.5 | 470.0 | 40.3 | 102.2 | 40.9 |
| Prefill 2048 | 413.4 | — | 420.5 | — | 428.6 | — | 405.1 | — | 767.4 | — | 770.0 | — |
| Short text | 216.3 | 32.5 | 235.6 | 32.8 | 253.8 | 31.5 | 64.3 | 22.3 | 69.8 | 41.8 | 240.0 | 41.3 |
| Long text | 411.8 | 30.3 | 412.2 | 27.7 | 422.9 | 28.6 | 499.7 | 11.1 | 643.3 | 40.2 | 753.6 | 40.6 |
| Image | 151.6 | 32.5 | 151.4 | 32.8 | fail | fail | 199.6 | 12.7 | 354.1 | 40.4 | 207.5 | 41.3 |
| Audio | 127.0 | 29.3 | 121.0 | 27.5 | fail | fail | 479.1 | 11.2 | 610.1 | 38.6 | n/a | n/a |
| Video | 135.7 | 30.5 | 127.8 | 27.7 | fail | fail | 485.0 | 11.2 | n/a | n/a | n/a | n/a |

## 6. TensorSharp KV cache dtype 对比

Gemma 4 E4B Q8_0 上不同 KV cache dtype 对 TensorSharp 吞吐的影响：

| Task | F32 prefill | F32 decode | F16 prefill | F16 decode | Q8_0 prefill | Q8_0 decode |
|---|----:|----:|----:|----:|----:|----:|
| Prefill 512 | 391.0 | — | 429.0 | — | 418.3 | — |
| Decode 128 | 215.4 | 33.3 | 235.5 | 33.8 | 247.6 | 32.0 |
| Prefill 2048 | 413.4 | — | 420.5 | — | 428.6 | — |
| Short text | 216.3 | 32.5 | 235.6 | 32.8 | 253.8 | 31.5 |
| Long text | 411.8 | 30.3 | 412.2 | 27.7 | 422.9 | 28.6 |
| Image | 151.6 | 32.5 | 151.4 | 32.8 | fail | fail |
| Audio | 127.0 | 29.3 | 121.0 | 27.5 | fail | fail |
| Video | 135.7 | 30.5 | 127.8 | 27.7 | fail | fail |

## 7. TensorSharp vs. llama.cpp（相对吞吐）

每个 cell 是 `TensorSharp t/s / llama.cpp t/s`。100% 表示持平；超过 100%
表示 TensorSharp 更快。

**Prefill**

| KV dtype | Prefill 512 | Prefill 2048 | Long text | Image | Audio |
|---|----:|----:|----:|----:|----:|
| F32 | 50% | 54% | 64% | 43% | 21% |
| F16 | 55% | 55% | 64% | 43% | 20% |
| Q8_0 | 53% | 56% | 66% | — | — |

**Decode**

| KV dtype | Decode 128 | Short text | Long text | Image | Audio |
|---|----:|----:|----:|----:|----:|
| F32 | 83% | 78% | 75% | 80% | 76% |
| F16 | 84% | 78% | 69% | 81% | 71% |
| Q8_0 | 79% | 75% | 71% | — | — |

## 8. TensorSharp vs. Ollama（相对吞吐）

**Prefill**

| KV dtype | Prefill 512 | Prefill 2048 | Long text | Image |
|---|----:|----:|----:|----:|
| F32 | 87% | 54% | 55% | 73% |
| F16 | 95% | 55% | 55% | 73% |
| Q8_0 | 93% | 56% | 56% | — |

**Decode**

| KV dtype | Decode 128 | Short text | Long text | Image |
|---|----:|----:|----:|----:|
| F32 | 81% | 79% | 75% | 79% |
| F16 | 83% | 79% | 68% | 79% |
| Q8_0 | 78% | 76% | 70% | — |

## 9. TensorSharp MLX 后端

TensorSharp 也提供 MLX 后端（[mlx-c](https://github.com/ml-explore/mlx-c) /
Apple Metal）。它使用独立于 GGML Metal 的 kernel 集与分配器（`MlxAllocator`），
因此作为单独列跟踪。

MLX 与 GGML Metal（F32 KV）在相同任务上的对比：

| Task | TensorSharp ggml_metal (F32) prefill | TensorSharp ggml_metal (F32) decode | TensorSharp mlx (F32) prefill | TensorSharp mlx (F32) decode |
|---|----:|----:|----:|----:|
| Prefill 512 | 391.0 | — | 543.5 | — |
| Decode 128 | 215.4 | 33.3 | 353.1 | 22.5 |
| Prefill 2048 | 413.4 | — | 405.1 | — |
| Short text | 216.3 | 32.5 | 64.3 | 22.3 |
| Long text | 411.8 | 30.3 | 499.7 | 11.1 |
| Image | 151.6 | 32.5 | 199.6 | 12.7 |
| Audio | 127.0 | 29.3 | 479.1 | 11.2 |
| Video | 135.7 | 30.5 | 485.0 | 11.2 |

## 10. 复现报告

```bash
# 1. 在 Ollama 中注册 Gemma4 GGUF（一次性）
cd benchmarks/inference_matrix
ollama create ts-gemma4-e4b-q8 -f modelfiles/Modelfile.gemma4-e4b-q8

# 2. 将所需媒体放到 <repo>/data/
#    apple.png, obama_first_45_secs.mp3, concert.mp4
mkdir -p ../../data && ln -sf <your-path>/apple.png ../../data/apple.png

# 3. 构建 TensorSharp CLI
dotnet build ../../TensorSharp.Cli/TensorSharp.Cli.csproj -c Release

# 4. 运行矩阵（一次一个 engine，避免 GPU 互相干扰）
python3 scripts/run_bench.py --engines ollama
python3 scripts/run_bench.py --engines llamacpp
python3 scripts/run_bench.py --engines tensorsharp_f32
python3 scripts/run_bench.py --engines tensorsharp_f16
python3 scripts/run_bench.py --engines tensorsharp_q80
python3 scripts/run_bench.py --engines tensorsharp_mlx

# 5. 从 JSON 结果重新生成报告
python3 scripts/build_report.py
```

驱动脚本：`benchmarks/inference_matrix/scripts/run_bench.py`  
报告生成器：`benchmarks/inference_matrix/scripts/build_report.py`  
每个 cell 的原始 JSON 会在运行时生成到
`benchmarks/inference_matrix/results/<engine>__<model>__<task>.json`。
