# Engine comparison benchmark — TensorSharp vs llama.cpp vs vLLM

Same GGUF files, same host, one uniform OpenAI `/v1/chat/completions` surface, across text / image / audio / video / single-turn / multi-turn / function-call / structured-output scenarios on the compute backends selected from the harness's `backends` registry (ggml_cuda / ggml_vulkan / ggml_metal / ggml_cpu / cpu / ...).

Numbers are tokens/second (higher is better). `—` = not applicable / skipped, `fail` = errored at runtime, `n/a` = combination never attempted.

This document aggregates two runs on the same host: the **GGML CUDA baseline** (all four model families, both engines on their CUDA builds) and a **GGML Vulkan run** (Gemma 4 E4B + 12B, both engines on their Vulkan builds).

## Software / hardware

| Component | Version / detail |
|---|---|
| TensorSharp | CUDA run: git `ca2f808` · Vulkan run: git `bfaa4d3`; .NET 10.0.204 (backends: ggml_cuda / ggml_vulkan / ggml_metal / cuda / mlx / ggml_cpu / cpu) |
| llama.cpp | CUDA build `C:\Works\llama.cpp\build-cuda\bin\Release\llama-server.exe` · Vulkan build `C:\Works\llama.cpp\build-vulkan\bin\Release\llama-server.exe` |
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
| Gemma 4 E4B it (Q8_0, dense multimodal) | vs llama.cpp · CUDA | 1.46× | 0.83× | 0.82× |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | vs llama.cpp · CUDA | 1.17× | 1.01× | 0.99× |
| Gemma 4 26B-A4B it (QAT UD-Q4_K_XL, MoE) | vs llama.cpp · CUDA | 0.96× | 1.32× | 1.30× |
| Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE) | vs llama.cpp · CUDA | 0.92× | 0.99× | 0.97× |
| Gemma 4 E4B it (Q8_0, dense multimodal) | vs llama.cpp · Vulkan | 0.92× | 0.68× | 1.21× |
| Gemma 4 12B it (QAT UD-Q4_K_XL, dense) | vs llama.cpp · Vulkan | 1.08× | 0.70× | 1.23× |

## GGML CUDA backend (baseline)

### Gemma 4 E4B it (Q8_0, dense multimodal)  (`gemma4-e4b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · CUDA | llama.cpp · CUDA |
|---|---:|---:|
| text_short | 51.2 | 53.4 |
| text_long | 51.3 | 52.9 |
| multi_turn | 51.2 | 52.9 |
| function_call | 50.9 | 53.0 |
| json_mode | 405.1 | 52.4 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · CUDA | llama.cpp · CUDA |
|---|---:|---:|
| text_short | 2578.3 | 2112.8 |
| text_long | 2757.2 | 2555.4 |
| multi_turn | 2733.7 | 2363.4 |
| function_call | 2758.9 | 2453.6 |
| json_mode | 504.3 | 2248.0 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · CUDA | llama.cpp · CUDA |
|---|---:|---:|
| text_short | 766.0 | 922.0 |
| text_long | 1141.0 | 1219.0 |
| multi_turn | 766.0 | 875.0 |
| function_call | 734.0 | 829.0 |
| json_mode | 4031.0 | 875.0 |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · CUDA |
|---|---:|
| text_short | 0.96× |
| text_long | 0.97× |
| multi_turn | 0.97× |
| function_call | 0.96× |
| json_mode | 7.73× |

_Prefill throughput_

| Scenario | vs llama.cpp · CUDA |
|---|---:|
| text_short | 1.22× |
| text_long | 1.08× |
| multi_turn | 1.16× |
| function_call | 1.12× |
| json_mode | 0.22× |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · CUDA |
|---|---:|
| text_short | 1.20× |
| text_long | 1.07× |
| multi_turn | 1.14× |
| function_call | 1.13× |
| json_mode | 0.22× |

### Gemma 4 12B it (QAT UD-Q4_K_XL, dense)  (`gemma4-12b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · CUDA | llama.cpp · CUDA |
|---|---:|---:|
| text_short | 39.6 | 39.5 |
| text_long | 39.3 | 37.6 |
| multi_turn | 39.5 | 38.5 |
| function_call | 81.0 | 39.5 |
| json_mode | 39.4 | 39.0 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · CUDA | llama.cpp · CUDA |
|---|---:|---:|
| text_short | 1139.0 | 1199.4 |
| text_long | 1170.4 | 1120.5 |
| multi_turn | 1196.6 | 1076.5 |
| function_call | 897.1 | 1094.1 |
| json_mode | 1161.7 | 999.5 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · CUDA | llama.cpp · CUDA |
|---|---:|---:|
| text_short | 1734.0 | 1625.0 |
| text_long | 2688.0 | 2781.0 |
| multi_turn | 1750.0 | 1922.0 |
| function_call | 2312.0 | 1860.0 |
| json_mode | 1750.0 | 1969.0 |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · CUDA |
|---|---:|
| text_short | 1.00× |
| text_long | 1.05× |
| multi_turn | 1.03× |
| function_call | 2.05× |
| json_mode | 1.01× |

_Prefill throughput_

| Scenario | vs llama.cpp · CUDA |
|---|---:|
| text_short | 0.95× |
| text_long | 1.04× |
| multi_turn | 1.11× |
| function_call | 0.82× |
| json_mode | 1.16× |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · CUDA |
|---|---:|
| text_short | 0.94× |
| text_long | 1.03× |
| multi_turn | 1.10× |
| function_call | 0.80× |
| json_mode | 1.13× |

### Gemma 4 26B-A4B it (QAT UD-Q4_K_XL, MoE)  (`gemma4-26b-a4b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · CUDA | llama.cpp · CUDA |
|---|---:|---:|
| text_short | 56.0 | 74.7 |
| text_long | 50.9 | 70.7 |
| multi_turn | 48.7 | 72.4 |
| function_call | 174.3 | 73.4 |
| json_mode | 68.4 | 73.3 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · CUDA | llama.cpp · CUDA |
|---|---:|---:|
| text_short | 1599.2 | 1006.2 |
| text_long | 1548.2 | 1444.6 |
| multi_turn | 1739.2 | 1337.4 |
| function_call | 1411.8 | 1315.4 |
| json_mode | 1626.4 | 954.4 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · CUDA | llama.cpp · CUDA |
|---|---:|---:|
| text_short | 1235.0 | 1937.0 |
| text_long | 2032.0 | 2157.0 |
| multi_turn | 1204.0 | 1547.0 |
| function_call | 1469.0 | 1547.0 |
| json_mode | 1250.0 | 2062.0 |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · CUDA |
|---|---:|
| text_short | 0.75× |
| text_long | 0.72× |
| multi_turn | 0.67× |
| function_call | 2.37× |
| json_mode | 0.93× |

_Prefill throughput_

| Scenario | vs llama.cpp · CUDA |
|---|---:|
| text_short | 1.59× |
| text_long | 1.07× |
| multi_turn | 1.30× |
| function_call | 1.07× |
| json_mode | 1.70× |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · CUDA |
|---|---:|
| text_short | 1.57× |
| text_long | 1.06× |
| multi_turn | 1.28× |
| function_call | 1.05× |
| json_mode | 1.65× |

### Qwen 3.6 35B-A3B (UD-IQ2_XXS, MoE)  (`qwen36-35b-a3b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · CUDA | llama.cpp · CUDA |
|---|---:|---:|
| text_short | 71.3 | 81.4 |
| text_long | 75.5 | 81.4 |
| multi_turn | 75.9 | 81.2 |
| function_call | 78.6 | 82.2 |
| json_mode | 75.0 | 81.1 |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · CUDA | llama.cpp · CUDA |
|---|---:|---:|
| text_short | 1259.0 | 1186.1 |
| text_long | 1188.7 | 1246.2 |
| multi_turn | 1198.1 | 1189.2 |
| function_call | 1196.5 | 1196.7 |
| json_mode | 1140.0 | 1207.5 |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · CUDA | llama.cpp · CUDA |
|---|---:|---:|
| text_short | 1610.0 | 1687.0 |
| text_long | 2719.0 | 2563.0 |
| multi_turn | 1797.0 | 1781.0 |
| function_call | 2000.0 | 1922.0 |
| json_mode | 1828.0 | 1672.0 |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · CUDA |
|---|---:|
| text_short | 0.88× |
| text_long | 0.93× |
| multi_turn | 0.93× |
| function_call | 0.96× |
| json_mode | 0.92× |

_Prefill throughput_

| Scenario | vs llama.cpp · CUDA |
|---|---:|
| text_short | 1.06× |
| text_long | 0.95× |
| multi_turn | 1.01× |
| function_call | 1.00× |
| json_mode | 0.94× |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · CUDA |
|---|---:|
| text_short | 1.05× |
| text_long | 0.94× |
| multi_turn | 0.99× |
| function_call | 0.96× |
| json_mode | 0.91× |

## GGML Vulkan backend

Both engines on their Vulkan builds (TensorSharp `--backend ggml_vulkan`; llama.cpp built with `GGML_VULKAN=ON`), same GPU as the CUDA run. Text-scenario decode and prefill sit near parity with llama.cpp; the sub-1.0× prefill geomeans are dominated by the image / audio prefill path, which is not yet optimized on Vulkan.

### Gemma 4 E4B it (Q8_0, dense multimodal)  (`gemma4-e4b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · Vulkan | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 45.3 | 47.9 |
| text_long | 44.2 | 47.3 |
| multi_turn | 43.3 | 47.4 |
| function_call | 43.6 | 47.6 |
| json_mode | 43.5 | 47.2 |
| image | 43.8 | 48.2 |
| audio | 44.7 | 47.8 |
| video | 44.6 | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · Vulkan | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 1857.9 | 1782.3 |
| text_long | 1315.8 | 1813.2 |
| multi_turn | 1861.3 | 1557.2 |
| function_call | 1727.8 | 1690.8 |
| json_mode | 1689.9 | 1701.6 |
| image | 229.6 | 401.9 |
| audio | 160.0 | 1185.8 |
| video | 352.3 | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · Vulkan | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 1063.0 | 1093.0 |
| text_long | 2391.0 | 1718.0 |
| multi_turn | 1125.0 | 1328.0 |
| function_call | 1172.0 | 1203.0 |
| json_mode | 1203.0 | 1156.0 |
| image | 1250.0 | 719.0 |
| audio | 125.0 | 969.0 |
| video | 3094.0 | — |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · Vulkan |
|---|---:|
| text_short | 0.95× |
| text_long | 0.93× |
| multi_turn | 0.91× |
| function_call | 0.92× |
| json_mode | 0.92× |
| image | 0.91× |
| audio | 0.94× |
| video | — |

_Prefill throughput_

| Scenario | vs llama.cpp · Vulkan |
|---|---:|
| text_short | 1.04× |
| text_long | 0.73× |
| multi_turn | 1.20× |
| function_call | 1.02× |
| json_mode | 0.99× |
| image | 0.57× |
| audio | 0.13× |
| video | — |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · Vulkan |
|---|---:|
| text_short | 1.03× |
| text_long | 0.72× |
| multi_turn | 1.18× |
| function_call | 1.03× |
| json_mode | 0.96× |
| image | 0.58× |
| audio | 7.75× |
| video | — |

### Gemma 4 12B it (QAT UD-Q4_K_XL, dense)  (`gemma4-12b`)

**Decode throughput (tok/s)**

| Scenario | TensorSharp · Vulkan | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 32.5 | 34.6 |
| text_long | 32.7 | 33.2 |
| multi_turn | 33.2 | 33.7 |
| function_call | 67.6 | 34.5 |
| json_mode | 32.5 | 34.2 |
| image | 34.5 | 35.4 |
| audio | 35.0 | 34.4 |
| video | 33.4 | — |

**Prefill throughput (tok/s)**

| Scenario | TensorSharp · Vulkan | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 826.0 | 804.7 |
| text_long | 651.6 | 782.1 |
| multi_turn | 673.3 | 675.5 |
| function_call | 653.8 | 719.3 |
| json_mode | 828.8 | 754.0 |
| image | 122.4 | 268.8 |
| audio | 128.2 | 608.5 |
| video | 280.2 | — |

**Time to first token (ms, lower is better)**

| Scenario | TensorSharp · Vulkan | llama.cpp · Vulkan |
|---|---:|---:|
| text_short | 2391.0 | 2422.0 |
| text_long | 4828.0 | 3984.0 |
| multi_turn | 3110.0 | 3063.0 |
| function_call | 3172.0 | 2829.0 |
| json_mode | 2453.0 | 2610.0 |
| image | 2344.0 | 1079.0 |
| audio | 156.0 | 1890.0 |
| video | 3890.0 | — |

**Performance ratio — TensorSharp vs reference (> 1.0× = TensorSharp faster)**

_Decode throughput_

| Scenario | vs llama.cpp · Vulkan |
|---|---:|
| text_short | 0.94× |
| text_long | 0.98× |
| multi_turn | 0.99× |
| function_call | 1.96× |
| json_mode | 0.95× |
| image | 0.97× |
| audio | 1.02× |
| video | — |

_Prefill throughput_

| Scenario | vs llama.cpp · Vulkan |
|---|---:|
| text_short | 1.03× |
| text_long | 0.83× |
| multi_turn | 1.00× |
| function_call | 0.91× |
| json_mode | 1.10× |
| image | 0.46× |
| audio | 0.21× |
| video | — |

_Time to first token (latency; > 1.0× = TensorSharp lower)_

| Scenario | vs llama.cpp · Vulkan |
|---|---:|
| text_short | 1.01× |
| text_long | 0.83× |
| multi_turn | 0.98× |
| function_call | 0.89× |
| json_mode | 1.06× |
| image | 0.46× |
| audio | 12.12× |
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
| llamacpp · ggml_cuda · gemma4-12b | yes |
| llamacpp · ggml_cuda · gemma4-26b-a4b | yes |
| llamacpp · ggml_cuda · gemma4-e4b | yes |
| llamacpp · ggml_cuda · qwen36-35b-a3b | yes |
| llamacpp · ggml_vulkan · gemma4-12b | yes |
| llamacpp · ggml_vulkan · gemma4-e4b | yes |
| tensorsharp · ggml_cuda · gemma4-12b | yes |
| tensorsharp · ggml_cuda · gemma4-26b-a4b | yes |
| tensorsharp · ggml_cuda · gemma4-e4b | yes |
| tensorsharp · ggml_cuda · qwen36-35b-a3b | no |
| tensorsharp · ggml_vulkan · gemma4-12b | yes |
| tensorsharp · ggml_vulkan · gemma4-e4b | yes |
