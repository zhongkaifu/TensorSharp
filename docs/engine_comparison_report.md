# Engine comparison benchmark — TensorSharp vs llama.cpp vs vLLM

Same GGUF files, same host, one uniform OpenAI `/v1/chat/completions` surface, across text / image / audio / video / single-turn / multi-turn / function-call / structured-output scenarios on the selected compute backends (ggml_cuda / ggml_vulkan / ggml_metal / ggml_cpu / cpu / ...).

Numbers are tokens/second (higher is better). `—` = not applicable / skipped, `fail` = errored at runtime, `n/a` = combination never attempted.

## Software / hardware

| Component | Version / detail |
|---|---|
| TensorSharp | git `a8d8f40`, .NET 10.0.204 (backends: ggml_cuda / ggml_vulkan / ggml_metal / cuda / mlx / ggml_cpu / cpu) |
| llama.cpp | `C:\Works\llama.cpp\build-cuda\bin\Release\llama-server.exe` |
| vLLM | endpoint `http://127.0.0.1:8000` (connect-only) |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU, 16384 MiB |


## Methodology

- Each `(engine, backend, model)` group launches its server once; all of that group's scenarios run against it, so per-scenario timings exclude model-load cost.
- Metrics come from the **streamed** response: `ttft` is time-to-first-token (prefill latency proxy), `prefill_tps = prompt_tokens / ttft`, and `decode_tps = (completion_tokens - 1) / (t_last - t_first)`.
- DiffusionGemma denoises whole blocks (no token stream), so it is run non-streaming and its `decode_tps` is wall-clock tokens/second.
- Greedy sampling (`temperature=0`); one warmup request per server is discarded.
- The headline per-engine tables are the **single-stream, MTP-off** baseline. MTP on/off and parallel-request scaling are reported in their own sections below.

## Performance ratio — TensorSharp vs reference engines

Geomean of TensorSharp's per-scenario speedup over each reference engine on the **same backend**, across every scenario both engines ran (single-stream, MTP-off). A value **> 1.0× means TensorSharp is faster** (for decode / prefill throughput) or lower-latency (for TTFT); `—` = no overlapping cells. Per-scenario ratios are in each model's section below.

_No overlapping TensorSharp / reference cells to compare._

## Gemma 4 26B-A4B it (QAT UD-Q4_K_XL, MoE)  (`gemma4-26b-a4b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · CUDA | TensorSharp · Direct CUDA |
|---|---:|---:|
| text_short | 78.7 | 35.3 |
| text_long | 78.4 | 38.5 |
| multi_turn | 79.1 | 38.5 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · CUDA | TensorSharp · Direct CUDA |
|---|---:|---:|
| text_short | 1832.1 | 128.2 |
| text_long | 1847.3 | 48.0 |
| multi_turn | 1940.7 | 114.2 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · CUDA | TensorSharp · Direct CUDA |
|---|---:|---:|
| text_short | 1078.0 | 15407.0 |
| text_long | 1703.0 | 65532.0 |
| multi_turn | 1079.0 | 18329.0 |

## Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE)  (`qwen36-35b-a3b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · CUDA | TensorSharp · Direct CUDA |
|---|---:|---:|
| text_short | 73.9 | 24.4 |
| text_long | 76.6 | 23.5 |
| multi_turn | 76.7 | 25.0 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · CUDA | TensorSharp · Direct CUDA |
|---|---:|---:|
| text_short | 1472.8 | 270.4 |
| text_long | 1690.1 | 267.4 |
| multi_turn | 1702.4 | 263.4 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · CUDA | TensorSharp · Direct CUDA |
|---|---:|---:|
| text_short | 1360.0 | 7407.0 |
| text_long | 1891.0 | 11954.0 |
| multi_turn | 1250.0 | 8078.0 |

## Output quality — TensorSharp vs llama.cpp

Both engines decode the **same GGUF greedily** (temperature=0) on the same backend, so their outputs should agree closely. `similarity` is a whitespace-normalized SequenceMatcher ratio between the two outputs (1.00 = identical); low similarity, an invalid JSON object in `json_mode`, or a missing tool call in `function_call` flags an output-quality problem on one side. Prefill scenarios (8-token outputs) are excluded. Side-by-side excerpts follow the table, lowest agreement first.

_No overlapping ok cells with captured output to compare._

## Image editing (stable-diffusion)

Same input image, prompt, resolution, step count, cfg and seed for every engine. Timings are each engine's **own pipeline timers** (TensorSharp's `[pipe-timing]` phases + server `elapsedSeconds`; sd.cpp's phase logs + `generate_image` total), so weight-file loading and HTTP/process overhead are excluded on both sides. `total (warm)` is the steady-state request on an already-running server; `first request (cold)` additionally pays TensorSharp's per-request DiT rebuild + graph capture on a fresh server (a CLI engine has no such distinction). Lower is better.

_No image-edit cells were run (see the `image_edit` scenario)._

## MTP / NextN speculative decoding (on vs off)

Single-stream decode tok/s with MTP/NextN speculative decoding off vs on (TensorSharp only). Speedup `< 1.0×` means speculation cost more than it saved for that cell — expected when the fused full-model decode path is already the fast path.

_No MTP on/off pairs were run (use `--mtp off,on`)._

## Parallel-request scaling (concurrency)

`decode/req` is the mean per-request decode tok/s; `aggregate` is the system-wide decode throughput (total generated tokens / the wall window during which any sequence was decoding) when N identical requests are fired at one server at once.

_No parallel-request cells were run (use `--concurrency 1,4,8`)._

## Function-calling correctness

_No function-call cells were run._
