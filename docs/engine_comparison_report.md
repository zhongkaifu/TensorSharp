# Engine comparison benchmark — TensorSharp vs llama.cpp vs vLLM

Same GGUF files, same host, one uniform OpenAI `/v1/chat/completions` surface, across text / image / audio / video / single-turn / multi-turn / function-call / structured-output scenarios on GPU and CPU backends.

Numbers are tokens/second (higher is better). `—` = not applicable / skipped, `fail` = errored at runtime, `n/a` = combination never attempted.

## Software / hardware

| Component | Version / detail |
|---|---|
| TensorSharp | git `6b15f31`, .NET 10.0.204 (backends: ggml_cuda / ggml_cpu) |
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

| Model | Comparison | decode | prefill | TTFT |
|---|---|---:|---:|---:|
| Gemma 4 E4B it (Q8_0, dense multimodal) | vs llama.cpp · GPU | 0.95× | 1.21× | 1.07× |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | vs llama.cpp · GPU | 0.93× | 1.23× | 1.10× |
| Gemma 4 26B-A4B it (QAT UD-Q4_K_XL, MoE) | vs llama.cpp · GPU | 0.92× | 1.88× | 1.69× |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | vs llama.cpp · GPU | 0.94× | 1.18× | 0.95× |

## Gemma 4 E4B it (Q8_0, dense multimodal)  (`gemma4-e4b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 51.8 | 57.6 |
| text_long | 51.5 | 56.0 |
| multi_turn | 51.1 | 56.8 |
| json_mode | 62.4 | 56.8 |
| video | n/a | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 200.0 | 123.3 |
| text_long | 1499.4 | 2011.8 |
| multi_turn | 837.2 | 724.1 |
| json_mode | 408.9 | 269.0 |
| video | n/a | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 125.0 | 219.0 |
| text_long | 797.0 | 593.0 |
| multi_turn | 172.0 | 203.0 |
| json_mode | 203.0 | 171.0 |
| video | n/a | — |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.90× |
| text_long | 0.92× |
| multi_turn | 0.90× |
| json_mode | 1.10× |
| video | — |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.62× |
| text_long | 0.75× |
| multi_turn | 1.16× |
| json_mode | 1.52× |
| video | — |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.75× |
| text_long | 0.74× |
| multi_turn | 1.18× |
| json_mode | 0.84× |
| video | — |

## Gemma 4 12B it (QAT UD-Q4_K_XL, dense)  (`gemma4-12b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 40.7 | 43.9 |
| text_long | 40.0 | 41.9 |
| multi_turn | 39.7 | 43.9 |
| json_mode | 40.9 | 43.9 |
| video | n/a | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 123.2 | 119.7 |
| text_long | 840.4 | 1124.3 |
| multi_turn | 460.1 | 296.0 |
| json_mode | 354.7 | 188.0 |
| video | n/a | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 203.0 | 234.0 |
| text_long | 1422.0 | 1062.0 |
| multi_turn | 313.0 | 500.0 |
| json_mode | 234.0 | 250.0 |
| video | n/a | — |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.93× |
| text_long | 0.95× |
| multi_turn | 0.90× |
| json_mode | 0.93× |
| video | — |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.03× |
| text_long | 0.75× |
| multi_turn | 1.55× |
| json_mode | 1.89× |
| video | — |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.15× |
| text_long | 0.75× |
| multi_turn | 1.60× |
| json_mode | 1.07× |
| video | — |

## Gemma 4 26B-A4B it (QAT UD-Q4_K_XL, MoE)  (`gemma4-26b-a4b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 83.7 | 89.3 |
| text_long | 78.7 | 81.3 |
| multi_turn | 78.2 | 86.5 |
| json_mode | 78.6 | 88.3 |
| video | n/a | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 145.3 | 179.5 |
| text_long | 1778.3 | 1272.9 |
| multi_turn | 657.5 | 350.7 |
| json_mode | 354.7 | 60.2 |
| video | n/a | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 172.0 | 156.0 |
| text_long | 672.0 | 938.0 |
| multi_turn | 219.0 | 422.0 |
| json_mode | 234.0 | 781.0 |
| video | n/a | — |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.94× |
| text_long | 0.97× |
| multi_turn | 0.90× |
| json_mode | 0.89× |
| video | — |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.81× |
| text_long | 1.40× |
| multi_turn | 1.87× |
| json_mode | 5.89× |
| video | — |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.91× |
| text_long | 1.40× |
| multi_turn | 1.93× |
| json_mode | 3.34× |
| video | — |

## Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE)  (`qwen36-35b-a3b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 78.7 | 86.5 |
| text_long | 73.7 | 84.7 |
| multi_turn | 79.5 | 85.6 |
| json_mode | 90.9 | 84.7 |
| audio | — | — |
| video | — | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 123.2 | 77.4 |
| text_long | 1173.8 | 1215.0 |
| multi_turn | 479.2 | 404.1 |
| json_mode | 247.0 | 232.6 |
| audio | — | — |
| video | — | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 203.0 | 297.0 |
| text_long | 1047.0 | 1000.0 |
| multi_turn | 313.0 | 344.0 |
| json_mode | 328.0 | 172.0 |
| audio | — | — |
| video | — | — |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.91× |
| text_long | 0.87× |
| multi_turn | 0.93× |
| json_mode | 1.07× |
| audio | — |
| video | — |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.59× |
| text_long | 0.97× |
| multi_turn | 1.19× |
| json_mode | 1.06× |
| audio | — |
| video | — |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.46× |
| text_long | 0.96× |
| multi_turn | 1.10× |
| json_mode | 0.52× |
| audio | — |
| video | — |

## MTP / NextN speculative decoding (on vs off)

Single-stream decode tok/s with MTP/NextN speculative decoding off vs on (TensorSharp only). Speedup `< 1.0×` means speculation cost more than it saved for that cell — expected when the fused full-model decode path is already the fast path.

_No MTP on/off pairs were run (use `--mtp off,on`)._

## Parallel-request scaling (concurrency)

`decode/req` is the mean per-request decode tok/s; `aggregate` is the system-wide decode throughput (total generated tokens / the wall window during which any sequence was decoding) when N identical requests are fired at one server at once.

_No parallel-request cells were run (use `--concurrency 1,4,8`)._

## Function-calling correctness

_No function-call cells were run._
