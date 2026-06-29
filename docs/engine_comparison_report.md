# Engine comparison benchmark — TensorSharp vs llama.cpp vs vLLM

Same GGUF files, same host, one uniform OpenAI `/v1/chat/completions` surface, across text / image / audio / video / single-turn / multi-turn / function-call / structured-output scenarios on GPU and CPU backends.

Numbers are tokens/second (higher is better). `—` = not applicable / skipped, `fail` = errored at runtime, `n/a` = combination never attempted.

## Software / hardware

| Component | Version / detail |
|---|---|
| TensorSharp | git `162eb55`, .NET 10.0.204 (backends: ggml_cuda / ggml_cpu) |
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

## Gemma 4 E4B it (Q8_0, dense multimodal)  (`gemma4-e4b`)

**Decode throughput (tok/s)**

| Scenario | llama.cpp · GPU |
|---|---:|
| video | — |

**Prefill throughput (tok/s)**

| Scenario | llama.cpp · GPU |
|---|---:|
| video | — |

**Time to first token (ms, lower is better)**

| Scenario | llama.cpp · GPU |
|---|---:|
| video | — |

## Gemma 4 12B it (QAT UD-Q4_K_XL, dense)  (`gemma4-12b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 39.8 | n/a |
| text_long | 38.9 | n/a |
| function_call | 57.1 | n/a |
| image | 39.5 | n/a |
| video | n/a | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 159.2 | n/a |
| text_long | 980.3 | n/a |
| function_call | 172.5 | n/a |
| image | 134.0 | n/a |
| video | n/a | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 157.0 | n/a |
| text_long | 1219.0 | n/a |
| function_call | 719.0 | n/a |
| image | 2141.0 | n/a |
| video | n/a | — |

## Gemma 4 26B-A4B it (QAT UD-Q4_K_XL, MoE)  (`gemma4-26b-a4b`)

**Decode throughput (tok/s)**

| Scenario | llama.cpp · GPU |
|---|---:|
| video | — |

**Prefill throughput (tok/s)**

| Scenario | llama.cpp · GPU |
|---|---:|
| video | — |

**Time to first token (ms, lower is better)**

| Scenario | llama.cpp · GPU |
|---|---:|
| video | — |

## Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE)  (`qwen36-35b-a3b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| audio | — | — |
| video | — | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| audio | — | — |
| video | — | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| audio | — | — |
| video | — | — |

## MTP / NextN speculative decoding (on vs off)

Single-stream decode tok/s with MTP/NextN speculative decoding off vs on (TensorSharp only). Speedup `< 1.0×` means speculation cost more than it saved for that cell — expected when the fused full-model decode path is already the fast path.

_No MTP on/off pairs were run (use `--mtp off,on`)._

## Parallel-request scaling (concurrency)

`decode/req` is the mean per-request decode tok/s; `aggregate` is the system-wide decode throughput (total generated tokens / the wall window during which any sequence was decoding) when N identical requests are fired at one server at once.

| Engine · Backend · Model · Scenario | metric | c=1 | c=2 |
|---|---|---:|---:|
| tensorsharp · gpu · gemma4-12b · text_short | decode/req t/s | 39.8 | 19.8 |
| tensorsharp · gpu · gemma4-12b · text_short | aggregate t/s | 39.8 | 39.0 |

## Function-calling correctness

| Engine · Backend · Model | tool_call emitted |
|---|:---:|
| tensorsharp · gpu · gemma4-12b | yes |
