# Engine comparison benchmark — TensorSharp vs llama.cpp vs vLLM

Same GGUF files, same host, one uniform OpenAI `/v1/chat/completions` surface, across text / image / audio / video / single-turn / multi-turn / function-call / structured-output scenarios on GPU and CPU backends.

Numbers are tokens/second (higher is better). `—` = not applicable / skipped, `fail` = errored at runtime, `n/a` = combination never attempted.

## Software / hardware

| Component | Version / detail |
|---|---|
| TensorSharp | git `5c1d7d7`, .NET 10.0.204 (backends: ggml_cuda / ggml_cpu) |
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
| Gemma 4 E4B it (Q8_0, dense multimodal) | vs llama.cpp · GPU | 0.88× | 0.51× | 0.89× |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | vs llama.cpp · GPU | 1.02× | 0.49× | 0.81× |
| Gemma 4 26B-A4B it (QAT UD-Q4_K_XL, MoE) | vs llama.cpp · GPU | 1.11× | 1.32× | 1.22× |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | vs llama.cpp · GPU | 0.62× | 0.62× | 0.60× |

## Gemma 4 E4B it (Q8_0, dense multimodal)  (`gemma4-e4b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 49.3 | 56.8 |
| text_long | 49.3 | 56.1 |
| multi_turn | 49.6 | 57.2 |
| function_call | 50.5 | 56.8 |
| json_mode | — | 56.4 |
| image | 49.9 | 56.8 |
| audio | 49.3 | 56.0 |
| video | 49.3 | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 160.3 | 144.4 |
| text_long | 1275.3 | 2008.4 |
| multi_turn | 766.0 | 671.2 |
| function_call | 480.8 | 657.0 |
| json_mode | 87.1 | 294.9 |
| image | 291.7 | 616.2 |
| audio | 142.9 | 1387.7 |
| video | 481.0 | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 156.0 | 187.0 |
| text_long | 937.0 | 594.0 |
| multi_turn | 188.0 | 219.0 |
| function_call | 156.0 | 172.0 |
| json_mode | 953.0 | 156.0 |
| image | 984.0 | 469.0 |
| audio | 140.0 | 828.0 |
| video | 2266.0 | — |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.87× |
| text_long | 0.88× |
| multi_turn | 0.87× |
| function_call | 0.89× |
| json_mode | — |
| image | 0.88× |
| audio | 0.88× |
| video | — |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.11× |
| text_long | 0.63× |
| multi_turn | 1.14× |
| function_call | 0.73× |
| json_mode | 0.30× |
| image | 0.47× |
| audio | 0.10× |
| video | — |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.20× |
| text_long | 0.63× |
| multi_turn | 1.16× |
| function_call | 1.10× |
| json_mode | 0.16× |
| image | 0.48× |
| audio | 5.91× |
| video | — |

## Gemma 4 12B it (QAT UD-Q4_K_XL, dense)  (`gemma4-12b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 39.5 | 44.2 |
| text_long | 39.0 | 41.7 |
| multi_turn | 38.7 | 44.2 |
| function_call | 77.2 | 45.6 |
| json_mode | — | 43.9 |
| image | 40.1 | 43.5 |
| audio | 40.8 | 41.5 |
| video | 39.3 | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 114.2 | 128.4 |
| text_long | 849.3 | 1140.4 |
| multi_turn | 461.5 | 278.7 |
| function_call | 158.8 | 405.7 |
| json_mode | 75.9 | 188.0 |
| image | 131.2 | 378.6 |
| audio | 98.5 | 908.4 |
| video | 365.3 | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 219.0 | 218.0 |
| text_long | 1407.0 | 1047.0 |
| multi_turn | 312.0 | 531.0 |
| function_call | 781.0 | 281.0 |
| json_mode | 1093.0 | 250.0 |
| image | 2188.0 | 766.0 |
| audio | 203.0 | 1266.0 |
| video | 2984.0 | — |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.89× |
| text_long | 0.94× |
| multi_turn | 0.88× |
| function_call | 1.69× |
| json_mode | — |
| image | 0.92× |
| audio | 0.98× |
| video | — |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.89× |
| text_long | 0.74× |
| multi_turn | 1.66× |
| function_call | 0.39× |
| json_mode | 0.40× |
| image | 0.35× |
| audio | 0.11× |
| video | — |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.00× |
| text_long | 0.74× |
| multi_turn | 1.70× |
| function_call | 0.36× |
| json_mode | 0.23× |
| image | 0.35× |
| audio | 6.24× |
| video | — |

## Gemma 4 26B-A4B it (QAT UD-Q4_K_XL, MoE)  (`gemma4-26b-a4b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 82.4 | 83.8 |
| text_long | 80.0 | 76.0 |
| multi_turn | 78.2 | 81.3 |
| function_call | 170.9 | 82.7 |
| json_mode | — | 79.7 |
| image | 77.2 | 92.4 |
| audio | 78.7 | fail |
| video | 74.6 | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 106.8 | 162.8 |
| text_long | 1625.9 | 1213.4 |
| multi_turn | 657.5 | 315.6 |
| function_call | 248.0 | 383.8 |
| json_mode | 132.8 | 94.0 |
| image | 75.0 | 23.3 |
| audio | 142.9 | fail |
| video | 81.4 | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 234.0 | 172.0 |
| text_long | 735.0 | 984.0 |
| multi_turn | 219.0 | 469.0 |
| function_call | 500.0 | 297.0 |
| json_mode | 625.0 | 500.0 |
| image | 3828.0 | 12438.0 |
| audio | 140.0 | fail |
| video | 13391.0 | — |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.98× |
| text_long | 1.05× |
| multi_turn | 0.96× |
| function_call | 2.07× |
| json_mode | — |
| image | 0.84× |
| audio | — |
| video | — |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.66× |
| text_long | 1.34× |
| multi_turn | 2.08× |
| function_call | 0.65× |
| json_mode | 1.41× |
| image | 3.22× |
| audio | — |
| video | — |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.74× |
| text_long | 1.34× |
| multi_turn | 2.14× |
| function_call | 0.59× |
| json_mode | 0.80× |
| image | 3.25× |
| audio | — |
| video | — |

## Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE)  (`qwen36-35b-a3b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 75.0 | 85.5 |
| text_long | 69.7 | 83.8 |
| multi_turn | 75.0 | 84.7 |
| function_call | — | 86.3 |
| json_mode | — | 84.7 |
| image | 18.7 | 82.1 |
| audio | — | — |
| video | — | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 106.4 | 73.7 |
| text_long | 1139.0 | 1215.0 |
| multi_turn | 505.1 | 387.2 |
| function_call | — | 735.2 |
| json_mode | 99.6 | 213.9 |
| image | 40.3 | 369.8 |
| audio | — | — |
| video | — | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 235.0 | 312.0 |
| text_long | 1079.0 | 1000.0 |
| multi_turn | 297.0 | 359.0 |
| function_call | — | 438.0 |
| json_mode | 813.0 | 187.0 |
| image | 49656.0 | 10984.0 |
| audio | — | — |
| video | — | — |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.88× |
| text_long | 0.83× |
| multi_turn | 0.89× |
| function_call | — |
| json_mode | — |
| image | 0.23× |
| audio | — |
| video | — |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.44× |
| text_long | 0.94× |
| multi_turn | 1.30× |
| function_call | — |
| json_mode | 0.47× |
| image | 0.11× |
| audio | — |
| video | — |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.33× |
| text_long | 0.93× |
| multi_turn | 1.21× |
| function_call | — |
| json_mode | 0.23× |
| image | 0.22× |
| audio | — |
| video | — |

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
| llamacpp · gpu · gemma4-e4b | no |
| llamacpp · gpu · qwen36-35b-a3b | yes |
| tensorsharp · gpu · gemma4-12b | yes |
| tensorsharp · gpu · gemma4-26b-a4b | yes |
| tensorsharp · gpu · gemma4-e4b | no |
| tensorsharp · gpu · qwen36-35b-a3b | no |
