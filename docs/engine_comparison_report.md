# Engine comparison benchmark — TensorSharp vs llama.cpp vs vLLM

Same GGUF files, same host, one uniform OpenAI `/v1/chat/completions` surface, across text / image / audio / video / single-turn / multi-turn / function-call / structured-output scenarios on GPU and CPU backends.

Numbers are tokens/second (higher is better). `—` = not applicable / skipped, `fail` = errored at runtime, `n/a` = combination never attempted.

## Software / hardware

| Component | Version / detail |
|---|---|
| TensorSharp | git `4631a25`, .NET 10.0.204 (backends: ggml_cuda / ggml_cpu) |
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
| Gemma 4 E4B it (Q8_0, dense multimodal) | vs llama.cpp · GPU | 1.29× | 0.58× | 1.03× |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | vs llama.cpp · GPU | 1.12× | 0.67× | 1.19× |
| Gemma 4 26B-A4B it (QAT UD-Q4_K_XL, MoE) | vs llama.cpp · GPU | 1.20× | 1.46× | 1.44× |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | vs llama.cpp · GPU | 0.41× | 0.31× | 0.34× |

## Gemma 4 E4B it (Q8_0, dense multimodal)  (`gemma4-e4b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 51.6 | 53.1 |
| text_long | 51.1 | 52.4 |
| multi_turn | 51.0 | 52.5 |
| function_call | 50.7 | 52.9 |
| json_mode | 392.6 | 52.1 |
| image | 50.7 | 53.4 |
| audio | 50.3 | 52.0 |
| video | 49.6 | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 2578.3 | 2150.1 |
| text_long | 2796.4 | 2492.0 |
| multi_turn | 2627.4 | 2280.0 |
| function_call | 2643.6 | 2410.0 |
| json_mode | 506.2 | 2333.3 |
| image | 301.2 | 528.3 |
| audio | 141.8 | 1313.1 |
| video | 505.6 | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 766.0 | 906.0 |
| text_long | 1125.0 | 1250.0 |
| multi_turn | 797.0 | 907.0 |
| function_call | 766.0 | 844.0 |
| json_mode | 4016.0 | 843.0 |
| image | 953.0 | 547.0 |
| audio | 141.0 | 875.0 |
| video | 2156.0 | — |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.97× |
| text_long | 0.98× |
| multi_turn | 0.97× |
| function_call | 0.96× |
| json_mode | 7.54× |
| image | 0.95× |
| audio | 0.97× |
| video | — |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.20× |
| text_long | 1.12× |
| multi_turn | 1.15× |
| function_call | 1.10× |
| json_mode | 0.22× |
| image | 0.57× |
| audio | 0.11× |
| video | — |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.18× |
| text_long | 1.11× |
| multi_turn | 1.14× |
| function_call | 1.10× |
| json_mode | 0.21× |
| image | 0.57× |
| audio | 6.21× |
| video | — |

## Gemma 4 12B it (QAT UD-Q4_K_XL, dense)  (`gemma4-12b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 39.3 | 39.4 |
| text_long | 39.7 | 37.4 |
| multi_turn | 39.4 | 38.2 |
| function_call | 83.9 | 39.2 |
| json_mode | 38.6 | 38.8 |
| image | 39.5 | 41.3 |
| audio | 39.7 | 39.9 |
| video | 38.5 | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 1128.6 | 1188.4 |
| text_long | 1163.9 | 1120.1 |
| multi_turn | 1185.7 | 1059.4 |
| function_call | 902.9 | 1067.7 |
| json_mode | 1171.8 | 991.4 |
| image | 129.3 | 247.4 |
| audio | 91.3 | 826.7 |
| video | 361.4 | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 1750.0 | 1640.0 |
| text_long | 2703.0 | 2782.0 |
| multi_turn | 1766.0 | 1953.0 |
| function_call | 2297.0 | 1906.0 |
| json_mode | 1735.0 | 1985.0 |
| image | 2219.0 | 1172.0 |
| audio | 219.0 | 1391.0 |
| video | 3016.0 | — |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.00× |
| text_long | 1.06× |
| multi_turn | 1.03× |
| function_call | 2.14× |
| json_mode | 0.99× |
| image | 0.96× |
| audio | 0.99× |
| video | — |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.95× |
| text_long | 1.04× |
| multi_turn | 1.12× |
| function_call | 0.85× |
| json_mode | 1.18× |
| image | 0.52× |
| audio | 0.11× |
| video | — |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.94× |
| text_long | 1.03× |
| multi_turn | 1.11× |
| function_call | 0.83× |
| json_mode | 1.14× |
| image | 0.53× |
| audio | 6.35× |
| video | — |

## Gemma 4 26B-A4B it (QAT UD-Q4_K_XL, MoE)  (`gemma4-26b-a4b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 73.5 | 73.9 |
| text_long | 77.6 | 69.5 |
| multi_turn | 77.6 | 71.3 |
| function_call | 173.5 | 72.7 |
| json_mode | 75.8 | 71.4 |
| image | 80.5 | 83.9 |
| audio | 80.0 | fail |
| video | 78.9 | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 826.0 | 685.3 |
| text_long | 1610.9 | 1337.9 |
| multi_turn | 1786.7 | 1285.1 |
| function_call | 1427.4 | 1264.8 |
| json_mode | 1734.6 | 947.1 |
| image | 73.2 | 31.7 |
| audio | 128.2 | fail |
| video | 70.3 | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 2391.0 | 2844.0 |
| text_long | 1953.0 | 2329.0 |
| multi_turn | 1172.0 | 1610.0 |
| function_call | 1453.0 | 1609.0 |
| json_mode | 1172.0 | 2078.0 |
| image | 3922.0 | 9140.0 |
| audio | 156.0 | fail |
| video | 15500.0 | — |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.99× |
| text_long | 1.12× |
| multi_turn | 1.09× |
| function_call | 2.39× |
| json_mode | 1.06× |
| image | 0.96× |
| audio | — |
| video | — |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.21× |
| text_long | 1.20× |
| multi_turn | 1.39× |
| function_call | 1.13× |
| json_mode | 1.83× |
| image | 2.31× |
| audio | — |
| video | — |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.19× |
| text_long | 1.19× |
| multi_turn | 1.37× |
| function_call | 1.11× |
| json_mode | 1.77× |
| image | 2.33× |
| audio | — |
| video | — |

## Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE)  (`qwen36-35b-a3b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 70.6 | 81.5 |
| text_long | 22.3 | 79.6 |
| multi_turn | 20.5 | 77.2 |
| function_call | 25.1 | 77.4 |
| json_mode | 21.2 | 76.8 |
| image | 60.0 | 78.0 |
| audio | — | — |
| video | — | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 1259.0 | 1164.7 |
| text_long | 373.4 | 1223.8 |
| multi_turn | 413.8 | 1120.0 |
| function_call | 420.8 | 1123.6 |
| json_mode | 416.8 | 1133.6 |
| image | 17.3 | 316.6 |
| audio | — | — |
| video | — | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · GPU | llama.cpp · GPU |
|---|---:|---:|
| text_short | 1610.0 | 1718.0 |
| text_long | 8656.0 | 2610.0 |
| multi_turn | 5203.0 | 1891.0 |
| function_call | 5687.0 | 2047.0 |
| json_mode | 5000.0 | 1781.0 |
| image | 115984.0 | 12829.0 |
| audio | — | — |
| video | — | — |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 0.87× |
| text_long | 0.28× |
| multi_turn | 0.27× |
| function_call | 0.32× |
| json_mode | 0.28× |
| image | 0.77× |
| audio | — |
| video | — |

_Prefill throughput_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.08× |
| text_long | 0.31× |
| multi_turn | 0.37× |
| function_call | 0.37× |
| json_mode | 0.37× |
| image | 0.05× |
| audio | — |
| video | — |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · GPU |
|---|---:|
| text_short | 1.07× |
| text_long | 0.30× |
| multi_turn | 0.36× |
| function_call | 0.36× |
| json_mode | 0.36× |
| image | 0.11× |
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
| llamacpp · gpu · gemma4-e4b | yes |
| llamacpp · gpu · qwen36-35b-a3b | yes |
| tensorsharp · gpu · gemma4-12b | yes |
| tensorsharp · gpu · gemma4-26b-a4b | yes |
| tensorsharp · gpu · gemma4-e4b | yes |
| tensorsharp · gpu · qwen36-35b-a3b | no |
