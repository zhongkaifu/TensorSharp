# Engine comparison benchmark — TensorSharp vs llama.cpp vs vLLM

Same GGUF files, same host, one uniform OpenAI `/v1/chat/completions` surface, across text / image / audio / video / single-turn / multi-turn / function-call / structured-output scenarios on GPU and CPU backends.

Numbers are tokens/second (higher is better). `—` = not applicable / skipped, `fail` = errored at runtime, `n/a` = combination never attempted.

## Software / hardware

| Component | Version / detail |
|---|---|
| TensorSharp | git `ca2f808`, .NET 10.0.204 (backends: ggml_cuda / ggml_cpu) |
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
| Gemma 4 E4B it (Q8_0, dense multimodal) | vs llama.cpp · GPU | 1.46× | 0.83× | 0.82× |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | vs llama.cpp · GPU | 1.17× | 1.01× | 0.99× |
| Gemma 4 26B-A4B it (QAT UD-Q4_K_XL, MoE) | vs llama.cpp · GPU | 0.96× | 1.32× | 1.30× |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | vs llama.cpp · GPU | 0.92× | 0.99× | 0.97× |

## Gemma 4 E4B it (Q8_0, dense multimodal)  (`gemma4-e4b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 51.2 | 53.4 |
| text_long | 51.3 | 52.9 |
| multi_turn | 51.2 | 52.9 |
| function_call | 50.9 | 53.0 |
| json_mode | 405.1 | 52.4 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 2578.3 | 2112.8 |
| text_long | 2757.2 | 2555.4 |
| multi_turn | 2733.7 | 2363.4 |
| function_call | 2758.9 | 2453.6 |
| json_mode | 504.3 | 2248.0 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 766.0 | 922.0 |
| text_long | 1141.0 | 1219.0 |
| multi_turn | 766.0 | 875.0 |
| function_call | 734.0 | 829.0 |
| json_mode | 4031.0 | 875.0 |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.96× |
| text_long | 0.97× |
| multi_turn | 0.97× |
| function_call | 0.96× |
| json_mode | 7.73× |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.22× |
| text_long | 1.08× |
| multi_turn | 1.16× |
| function_call | 1.12× |
| json_mode | 0.22× |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.20× |
| text_long | 1.07× |
| multi_turn | 1.14× |
| function_call | 1.13× |
| json_mode | 0.22× |

## Gemma 4 12B it (QAT UD-Q4_K_XL, dense)  (`gemma4-12b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 39.6 | 39.5 |
| text_long | 39.3 | 37.6 |
| multi_turn | 39.5 | 38.5 |
| function_call | 81.0 | 39.5 |
| json_mode | 39.4 | 39.0 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 1139.0 | 1199.4 |
| text_long | 1170.4 | 1120.5 |
| multi_turn | 1196.6 | 1076.5 |
| function_call | 897.1 | 1094.1 |
| json_mode | 1161.7 | 999.5 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 1734.0 | 1625.0 |
| text_long | 2688.0 | 2781.0 |
| multi_turn | 1750.0 | 1922.0 |
| function_call | 2312.0 | 1860.0 |
| json_mode | 1750.0 | 1969.0 |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.00× |
| text_long | 1.05× |
| multi_turn | 1.03× |
| function_call | 2.05× |
| json_mode | 1.01× |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.95× |
| text_long | 1.04× |
| multi_turn | 1.11× |
| function_call | 0.82× |
| json_mode | 1.16× |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.94× |
| text_long | 1.03× |
| multi_turn | 1.10× |
| function_call | 0.80× |
| json_mode | 1.13× |

## Gemma 4 26B-A4B it (QAT UD-Q4_K_XL, MoE)  (`gemma4-26b-a4b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 56.0 | 74.7 |
| text_long | 50.9 | 70.7 |
| multi_turn | 48.7 | 72.4 |
| function_call | 174.3 | 73.4 |
| json_mode | 68.4 | 73.3 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 1599.2 | 1006.2 |
| text_long | 1548.2 | 1444.6 |
| multi_turn | 1739.2 | 1337.4 |
| function_call | 1411.8 | 1315.4 |
| json_mode | 1626.4 | 954.4 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 1235.0 | 1937.0 |
| text_long | 2032.0 | 2157.0 |
| multi_turn | 1204.0 | 1547.0 |
| function_call | 1469.0 | 1547.0 |
| json_mode | 1250.0 | 2062.0 |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.75× |
| text_long | 0.72× |
| multi_turn | 0.67× |
| function_call | 2.37× |
| json_mode | 0.93× |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.59× |
| text_long | 1.07× |
| multi_turn | 1.30× |
| function_call | 1.07× |
| json_mode | 1.70× |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.57× |
| text_long | 1.06× |
| multi_turn | 1.28× |
| function_call | 1.05× |
| json_mode | 1.65× |

## Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE)  (`qwen36-35b-a3b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 71.3 | 81.4 |
| text_long | 75.5 | 81.4 |
| multi_turn | 75.9 | 81.2 |
| function_call | 78.6 | 82.2 |
| json_mode | 75.0 | 81.1 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 1259.0 | 1186.1 |
| text_long | 1188.7 | 1246.2 |
| multi_turn | 1198.1 | 1189.2 |
| function_call | 1196.5 | 1196.7 |
| json_mode | 1140.0 | 1207.5 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 1610.0 | 1687.0 |
| text_long | 2719.0 | 2563.0 |
| multi_turn | 1797.0 | 1781.0 |
| function_call | 2000.0 | 1922.0 |
| json_mode | 1828.0 | 1672.0 |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.88× |
| text_long | 0.93× |
| multi_turn | 0.93× |
| function_call | 0.96× |
| json_mode | 0.92× |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.06× |
| text_long | 0.95× |
| multi_turn | 1.01× |
| function_call | 1.00× |
| json_mode | 0.94× |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.05× |
| text_long | 0.94× |
| multi_turn | 0.99× |
| function_call | 0.96× |
| json_mode | 0.91× |

## MTP / NextN speculative decoding (on vs off)

Single-stream decode tok/s with MTP/NextN speculative decoding off vs on (TensorSharp only). Speedup `< 1.0×` means speculation cost more than it saved for that cell — expected when the fused full-model decode path is already the fast path.

_No MTP on/off pairs were run (use `--mtp off,on`)._

## Parallel-request scaling (concurrency)

`decode/req` is the mean per-request decode tok/s; `aggregate` is the system-wide decode throughput (total generated tokens / the wall window during which any sequence was decoding) when N identical requests are fired at one server at once.

_No parallel-request cells were run (use `--concurrency 1,4,8`)._

## Function-calling correctness

| Engine · Backend · Model | tool_call emitted |
|---|:---:|
| llamacpp · gpu · gemma4-12b | yes |
| llamacpp · gpu · gemma4-26b-a4b | yes |
| llamacpp · gpu · gemma4-e4b | yes |
| llamacpp · gpu · qwen36-35b-a3b | yes |
| tensorsharp · gpu · gemma4-12b | yes |
| tensorsharp · gpu · gemma4-26b-a4b | yes |
| tensorsharp · gpu · gemma4-e4b | yes |
| tensorsharp · gpu · qwen36-35b-a3b | no |
