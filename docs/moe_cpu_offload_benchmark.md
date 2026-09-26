# MoE CPU-offload benchmark — TensorSharp vs llama.cpp

What `--n-cpu-moe N` (llama.cpp `-ncmoe N`) costs and saves on both engines: the
same GGUF files, the same host, the same benchmark shape. Three mixture-of-experts
models x five offload depths x two engines, plus a 150.7 GiB DeepSeek V4 Flash
across two GPUs at three depths.

Prompts are **4,096 and 8,192 tokens** — the sizes a document, a code file or a
long chat history actually arrives at. **Peak VRAM** is the per-process figure in
MiB (lower is better); **pp4096** / **pp8192** are prefill and **tg128** decode
throughput in tokens/second (higher is better).

## Headline

Offload trades VRAM for throughput, and on long prompts TensorSharp trades it far
better. A Gemma 4 26B-A4B that needs 16.4 GiB runs in 10.8 GiB with every expert
in system RAM, and prefills **5.6-6.2x faster** than llama.cpp does in the same
configuration:

| Model | `--n-cpu-moe` | TS pp8192 | llama pp8192 | TS tg128 | llama tg128 |
|---|---|---:|---:|---:|---:|
| Gemma 4 26B-A4B | 16 of 30 | **4,888** | 854 | **54.5** | 21.9 |
| Gemma 4 26B-A4B | all 30 | **3,072** | 495 | **39.7** | 12.9 |
| Qwen 3.5 35B-A3B | 24 of 48 | **5,259** | 484 | **52.3** | 15.8 |
| Qwen 3.5 35B-A3B | all 48 | **3,709** | 457 | **38.6** | 10.2 |
| GPT-OSS 20B | 12 of 24 | **6,394** | 1,188 | **51.7** | 18.3 |
| GPT-OSS 20B | all 24 | **3,798** | 548 | **27.7** | 9.4 |
| DeepSeek V4 (2 GPUs) | 24 of 43 | **236** | 63 | 5.3 | **7.2** |

Across every offloaded depth on the three seam models, prefill is **4.5-10.9x**
faster than llama.cpp and decode **2.5-4.5x**, at 1.0-3.5x the VRAM. DeepSeek V4
offloads through a different mechanism and is the one model where llama.cpp's
offloaded decode is ahead — see its section.

On the **resident** path (`--n-cpu-moe 0`) at these prompt lengths TensorSharp is
ahead of llama.cpp's default configuration on Gemma 4 (1.03-1.06x) and Qwen 3.5
(1.16-1.17x), and ahead on DeepSeek V4 (1.44-1.97x). That lead is a batching
difference rather than a kernel one and it reverses when llama.cpp's ubatch is
matched to our chunk — the numbers are in [Is the win just chunk
size?](#is-the-win-just-chunk-size).

**Moving the benchmark from 512- to 8,192-token prompts found a real defect**, in
code added earlier in this same pass: GPT-OSS's new whole-model prefill kernel
built one attention mask per layer instead of one per distinct geometry, which is
invisible at 512 tokens and cost it more than half its throughput at 8,192.
Fixing it took GPT-OSS pp8192 from 5,825 to **12,925 tok/s**.

Also fixed here, all on the resident path, all recorded as pre-existing and
deferred by the previous revision:

| | before | after | llama.cpp |
|---|---:|---:|---:|
| GPT-OSS pp512, `--n-cpu-moe 0` | 1,030 | **12,291** | 17,349 |
| GPT-OSS peak VRAM | 24,816 MiB | **12,482 MiB** | 12,020 MiB |
| GPT-OSS JSON mode | `{"city":"...","population":0}` | **`{"city":"Paris","population":2140526}`** | — |
| Gemma 4 image on `--backend cuda` | hallucinated | **reads it correctly** | — |
| `TensorSharp.Cli --tools` on a JSON-Schema tool file | crashed (exit 134) | **works** | — |

Correctness across all of it: **182 checks, all passing** — 110 server (19
configurations x 6 capabilities across three GPU backends, ±offload, TP2) and 72
CLI (5 models x 3 backends x 5 features) — plus DeepSeek V4 speculative decoding
at 1.39x. The GPT-OSS server cells were re-run after the mask change above and
pass 20/20 on `ggml_cuda` (±offload), `ggml_vulkan` and `cuda`.


## Host and software

| Component | Detail |
|---|---|
| GPU | 2 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB each, driver 580.126.20, PCIe 5.0 x16 |
| CPU | 2 x Intel Xeon 6952P (384 threads, 6 NUMA nodes), cgroup quota 81.6 CPUs |
| RAM | 1,511 GiB |
| Storage | Models on a MooseFS network mount (page-cache warm for every measured run) |
| OS | Ubuntu 24.04.3 LTS, CUDA 12.8 |
| TensorSharp | branch `feature/support_moe_offload_to_cpu`, .NET 10.0.110, backend `ggml_cuda` |
| llama.cpp | `llama-bench` build 4308a4f, CUDA backend, default `-t 192` |

## Methodology

- **Identical shape on both engines**: 4,096 and 8,192 synthetic prompt tokens,
  128 decode steps. `llama-bench -p 4096,8192 -n 128 -r 2`; TensorSharp
  `--benchmark --bench-prefill {4096,8192} --bench-decode 128 --bench-runs 3
  --bench-fixed-tokens`, which keeps sampling out of the timed loop the way
  `llama-bench` does. Best-of is reported on both sides.
- **Matched context.** `llama-bench` fixes `n_ctx = n_prompt + n_gen`; TensorSharp
  is pinned to the same figure rounded up to a multiple of 256 (`MAX_CONTEXT=4352`
  and `8448`), because ggml-cuda's flash-attention kernel selection aborts on a
  Gemma 4 KV size that is not a multiple of 256.
- **Peak VRAM** is `nvidia-smi --query-compute-apps=pid,used_memory` sampled every
  250 ms over the process's lifetime and summed across the run's PIDs — a real
  per-process figure, not a whole-device delta. The reported number is the larger
  of the 4,096- and 8,192-token runs.
- **One GPU per cell** (`CUDA_VISIBLE_DEVICES=0`) except DeepSeek V4, which needs
  both.
- Every cell is a cold process, run one at a time, on an otherwise idle host.
- **The decode column is not measured at the same context on both engines.**
  `llama-bench` runs `tg128` as its own test at `n_ctx = 128`; TensorSharp's
  `--bench-decode` runs after the prefill in the same process, i.e. decoding at
  ~8K context. Re-measuring TensorSharp's decode at a short context to match:
  gemma-4-26B **165.7** (vs 161.4 after the 8K prefill), Qwen 3.5-35B **160.6**
  (vs 160.0), GPT-OSS **297.8** (vs 212.8). It barely moves the two models whose
  decode is dominated by the whole-model graph, and it moves GPT-OSS a lot — so
  the GPT-OSS resident decode ratio below reads 0.62x where the matched-context
  figure is 0.87x. The offloaded decode ratios are, if anything, understated.
- **Each engine uses its own default batching.** TensorSharp prefills in
  2,048-token chunks; llama.cpp's default `n_ubatch` is 512. A control run with
  matched `-ub 2048` is reported alongside — see
  [Is the win just chunk size?](#is-the-win-just-chunk-size).

## Results by model

Each row is one offload depth, with TensorSharp, llama.cpp and the ratio between
them side by side for every metric. Ratios are TensorSharp / llama.cpp: **>1.0x
means TensorSharp is faster**, and for VRAM **>1.0x means TensorSharp is
heavier**.

### Gemma 4 26B-A4B it (UD-IQ4_XS, 30 MoE layers)

| `--n-cpu-moe` | TS VRAM (MiB) | llama VRAM (MiB) | ratio | TS pp4096 | llama pp4096 | ratio | TS pp8192 | llama pp8192 | ratio | TS tg128 | llama tg128 | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 _(baseline)_ | 16,822 | 14,602 | 1.15x | 11,173 | 10,843 | **1.03x** | 11,274 | 10,628 | **1.06x** | 161.4 | 206.7 | 0.78x |
| 8 | 15,724 | 11,874 | 1.32x | 7,063 | 1,459 | **4.84x** | 6,500 | 1,459 | **4.46x** | 80.2 | 32.7 | **2.45x** |
| 16 | 14,128 | 9,122 | 1.55x | 4,183 | 833 | **5.02x** | 4,888 | 854 | **5.72x** | 54.5 | 21.9 | **2.49x** |
| 24 | 12,346 | 6,368 | 1.94x | 3,500 | 667 | **5.25x** | 3,958 | 689 | **5.74x** | 49.1 | 16.7 | **2.93x** |
| 30 _(`--cpu-moe`)_ | 11,038 | 4,134 | 2.67x | 3,035 | 543 | **5.59x** | 3,072 | 495 | **6.21x** | 39.7 | 12.9 | **3.07x** |

### Qwen 3.5 35B-A3B (UD-IQ4_XS, 48 MoE layers)

| `--n-cpu-moe` | TS VRAM (MiB) | llama VRAM (MiB) | ratio | TS pp4096 | llama pp4096 | ratio | TS pp8192 | llama pp8192 | ratio | TS tg128 | llama tg128 | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 _(baseline)_ | 19,862 | 17,522 | 1.13x | 9,538 | 8,149 | **1.17x** | 9,405 | 8,073 | **1.16x** | 160.0 | 228.4 | 0.70x |
| 12 | 18,148 | 13,282 | 1.37x | 6,755 | 988 | **6.84x** | 6,648 | 954 | **6.97x** | 75.4 | 27.5 | **2.74x** |
| 24 | 15,414 | 9,010 | 1.71x | 4,412 | 498 | **8.85x** | 5,259 | 484 | **10.86x** | 52.3 | 15.8 | **3.31x** |
| 36 | 12,684 | 4,738 | 2.68x | 3,772 | 523 | **7.21x** | 4,223 | 517 | **8.17x** | 50.7 | 11.3 | **4.50x** |
| 48 _(`--cpu-moe`)_ | 11,606 | 3,314 | 3.50x | 3,917 | 477 | **8.21x** | 3,709 | 457 | **8.11x** | 38.6 | 10.2 | **3.77x** |

### GPT-OSS 20B (Q8_0 / MXFP4, 24 MoE layers)

| `--n-cpu-moe` | TS VRAM (MiB) | llama VRAM (MiB) | ratio | TS pp4096 | llama pp4096 | ratio | TS pp8192 | llama pp8192 | ratio | TS tg128 | llama tg128 | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 _(baseline)_ | 13,186 | 12,204 | 1.08x | 13,964 | 17,856 | 0.78x | 12,925 | 17,642 | 0.73x | 212.8 | 344.2 | 0.62x |
| 6 | 11,560 | 9,812 | 1.18x | 8,975 | 1,747 | **5.14x** | 7,617 | 1,666 | **4.57x** | 85.8 | 32.2 | **2.67x** |
| 12 | 9,378 | 7,386 | 1.27x | 6,470 | 1,176 | **5.50x** | 6,394 | 1,188 | **5.38x** | 51.7 | 18.3 | **2.83x** |
| 18 | 7,192 | 4,962 | 1.45x | 4,315 | 807 | **5.35x** | 4,393 | 751 | **5.85x** | 30.7 | 12.1 | **2.54x** |
| 24 _(`--cpu-moe`)_ | 4,762 | 2,536 | 1.88x | 4,277 | 568 | **7.53x** | 3,798 | 548 | **6.93x** | 27.7 | 9.4 | **2.95x** |

> GPT-OSS is the one model still behind on the resident path at these prompt
> lengths (0.73–0.78x). Moving to 4,096/8,192-token prompts is what exposed it:
> the whole-model prefill kernel built **one attention mask per layer** where only
> two distinct geometries exist (the sliding-window one on even layers, the full
> causal one on odd layers). At an 8,192-token chunk that was ~200M fp16 host
> writes and ~380 MB of uploads per forward for masks that are bit-identical in
> pairs. Sharing them by geometry took pp8192 from **5,825 to 12,925 tok/s
> (2.2x)** and pp4096 from 7,055 to 13,964 — and it also flipped the chunk-size
> choice back: with the mask cost gone, 2,048 beats 512 again.


### DeepSeek V4 Flash (UD-Q8_K_XL, 5 shards / 150.7 GiB, 43 layers, both GPUs)

DeepSeek V4 does not use the host-MoE seam: it hands its offloaded layers to
`ggml_backend_sched`, which streams the weights back to the GPU for a
prefill-sized batch exactly the way llama.cpp does.

| `--n-cpu-moe` | TS VRAM (MiB) | llama VRAM (MiB) | ratio | TS pp4096 | llama pp4096 | ratio | TS pp8192 | llama pp8192 | ratio | TS tg128 | llama tg128 | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 _(baseline, both GPUs)_ | 169,132 | 155,608 | 1.09x | 3,448 | 2,398 | **1.44x** | 4,387 | 2,232 | **1.97x** | 51.1 | 49.6 | **1.03x** |
| 12 | 131,818 | 117,150 | 1.13x | 392 | 126 | **3.11x** | 428 | 124 | **3.46x** | 10.3 | 13.7 | 0.75x |
| 24 | 79,742 | 78,954 | 1.01x | 218 | 64 | **3.42x** | 236 | 63 | **3.72x** | 5.3 | 7.2 | 0.74x |

DeepSeek V4 remains the one model where llama.cpp's offloaded *decode* is ahead
(0.74–0.75x): both engines run that matmul through the same ggml CPU backend, and
DSV4's per-token host read is ~260 MB (6 of 256 experts, `n_ff` 2048, `n_embd`
7168) against ~40 MB for the seam architectures, which puts it in a different part
of the thread-scaling curve. That is why DSV4 keeps
`available_cpu_parallelism()` workers rather than the seam's capped default.

### A note on VRAM

TensorSharp is heavier, and **the gap widens with both context and offload
depth** — 1.13–1.15x at `--n-cpu-moe 0` but 2.7–3.5x with every expert offloaded.
Once the experts are on the host, what is left in VRAM is the KV cache and the
compute buffers, and that is where the difference lives: llama.cpp sizes a
sliding-window layer's KV to the window rather than to the full context, and its
512-token ubatch needs proportionally smaller activation buffers than our
2,048-token chunk. Both are real and neither is measured apart here; it is the
clearest remaining item on the offload path, and it did not show up in the
previous 512-token revision of this document (1.20–1.64x) because at that context
the KV is noise next to the weights.

### Is the win just chunk size?

Partly, and it matters enough to state plainly. TensorSharp prefills a long
prompt in **2,048-token chunks**; llama.cpp's default `n_ubatch` is **512**. Those
are the two engines' own defaults, and the tables above compare them as shipped.
Re-running llama.cpp with a matched `-ub 2048 -b 8192`:

| model | llama `-ub 512` (default) | llama `-ub 2048` | TensorSharp |
|---|---:|---:|---:|
| gemma-4-26B pp4096 / pp8192 | 10,815 / 10,608 | **14,485 / 14,643** | 11,173 / 11,274 |
| Qwen 3.5-35B pp4096 / pp8192 | 8,117 / 8,006 | **11,655 / 11,855** | 9,538 / 9,405 |
| gpt-oss-20B pp4096 / pp8192 | 17,712 / 17,504 | **20,728 / 21,668** | 13,964 / 12,925 |

So: **at each engine's defaults TensorSharp leads on resident prefill for Gemma 4
and Qwen 3.5 (1.03–1.17x); with llama.cpp's batch matched to ours it does not
(0.77–0.82x).** The lead is a batching-configuration difference, not a
kernel-efficiency one, and it would be dishonest to report the first number
without the second.

The reverse control says our chunk is not the knob: TensorSharp is flat across
chunk sizes at pp8192 (`TS_PREFILL_CHUNK`), so we are already on the plateau and
llama.cpp is simply getting more out of the same 2,048-token batch.

| chunk | gemma-4-26B pp8192 | Qwen 3.5-35B pp8192 |
|---|---:|---:|
| 2,048 _(default)_ | 11,119 | 9,386 |
| 4,096 | 11,445 | 9,204 |
| 8,192 | 11,194 | 9,215 |

None of this touches the offload results, which are the point of this document:
there the margin is **4.5–10.9x on prefill** and **2.5–4.5x on decode**, far
outside anything batch size accounts for.

> **A note on run-to-run variance.** The offloaded cells move around between runs
> more than the resident ones do — both engines' offloaded paths end in a host
> matmul whose throughput depends on how the scheduler places threads across two
> sockets. Treat a single offloaded ratio as ±30% and the shape of the curve as
> the result.



## What changed in this pass

The previous revision of this document recorded four defects on the *resident*
(`--n-cpu-moe 0`) path as pre-existing and deferred them. All four are fixed here,
plus a fifth the widened test sweep turned up and a sixth that only appeared once
the benchmark moved to 8,192-token prompts.

### 1. GPT-OSS had no whole-model prefill graph

GPT-OSS was the one MoE architecture still prefilling **layer by layer**, and per
layer that cost:

```
fused attention graph → post-attn RMSNorm → router mul_mat → DOWNLOAD the router
scores → host top-k → fill(0) → fused MoE graph (rebuilt each call, with the
stacked expert biases re-uploaded) → DOWNLOAD the MoE output → residual add
```

Six graph submissions and **two host round trips per layer**, 24 layers deep,
twice over because anything past 256 tokens was chunked. 512 tokens took ~510 ms
against llama.cpp's ~30 ms.

`TSGgml_GptOssModelPrefill` (`ggml_ops_gptoss_prefill.cpp`) is the sibling of the
existing whole-model *decode* kernel with the token axis restored: hidden is
`[H, N]`, RoPE positions are an `I32[N]`, the KV write is one `ggml_cpy` per layer
into a `[start_pos, start_pos+N)` row view, and the mask is F16 `[window, N]`
filled causally with the sliding-window floor on GPT-OSS's even layers. The router
top-k runs on the accelerator (`ggml_top_k` over the raw logits, the SOFTMAX_WEIGHT
gate llama.cpp's `build_moe_ffn` uses for this architecture), so **nothing crosses
the bus** between the embeddings going in and the logits coming out. Offloaded
layers keep their host-MoE seam, so `--n-cpu-moe` still works — the graph just
pauses at each offloaded router instead of at every layer.

| gpt-oss-20b pp512 | before | after | gain |
|---|---:|---:|---:|
| `--n-cpu-moe 0` | 1,030.5 | **12,290.7** | 11.9x |
| `--n-cpu-moe 6` | 840.3 | 5,109.4 | 6.1x |
| `--n-cpu-moe 12` | 691.3 | 2,814.8 | 4.1x |

The sliding-window layers also stopped attending over the whole cache: a query at
position `p` reads keys `[p-127, p]`, so the graph views only
`[start_pos-127, totalSeqLen)` for those layers.

With the O(seqLen²) score tensor gone, the prefill chunk is no longer capped at
256 tokens — it is sized for GEMM efficiency instead (2048, `TS_GPTOSS_PREFILL_CHUNK`).

### 2. GPT-OSS held every expert byte in VRAM twice

24,816 MiB resident against llama.cpp's 12,018 for the same 11.5 GiB of weights.
`TS_GGML_LOG_VRAM=1` puts the second copy on the record: `preload-weight` totals
11,501 MB and then `devcopy` climbs to **9,717 MB** on top of it.

The routed experts live in the GGUF as one 3-D block per tensor, which the loader
both splits into per-expert 2-D views *and* keeps as a stacked block. Every graph
that matters — whole-model decode, the new whole-model prefill, the fused MoE
kernel — reads the **stacked** block, which gets its own device copy on first use.
Preloading the per-expert views as well is a second full copy of every expert.
Gemma 4 and Qwen 3.5 already excluded them; GPT-OSS's exclusion was gated on
`UsesExpertParallelMoE`, so it only applied under tensor parallelism (where the
same bug had been caught from the other direction — rank 0 holding 16,019 MB
against rank 1's 5,163). Single-GPU took the double copy and nothing excluded it.

| gpt-oss-20b, `--n-cpu-moe 0` | before | after | llama.cpp |
|---|---:|---:|---:|
| peak VRAM (MiB) | 24,816 | **12,482** | 12,020 |
| device-resident preload | 11,501 MB / 1,586 tensors | 1,819 MB / 50 tensors | — |

### 3. Gemma 4 vision was broken on the pure-C# `cuda` backend

Asked to describe a picture of a red circle, a blue square and the text
"APPLE 42", the `cuda` backend answered *"a red circle on the left and a blue
square on the right, both set against a light gray background. Below these shapes
is a large, realistic image of a red apple. There is no text written in the
image."* — some vision signal was arriving, and the rest was invention. The same
model on `ggml_cuda` and `ggml_vulkan` described it correctly.

`TS_VISION_TRACE=1` (new) prints per-stage activation statistics through the
encoder. All 27 encoder blocks matched the working backend to five decimal
places; the divergence appeared in a single stage:

| stage | `ggml_cuda` | `cuda` (before) |
|---|---|---|
| `pool.scale` | mean −32.31, rms 2413.3 | mean −32.32, rms 2413.5 |
| `pool.std` | mean **0.008**, rms **1.13** | mean −31.82, rms 2397.9 |

`PoolAndProject` standardises the pooled patches with
`Ops.Sub(pooled, pooled, stdBias)` then `Ops.Mul(pooled, pooled, stdScale)` —
`x[130, 1152] OP bias[1152]`, a row broadcast. `Add` had a dedicated broadcast
path; **`Sub`, `Mul` and `Div` did not**, and the generic `Apply3` fallback they
land on advances all three iterators together and stops at the shortest operand's
last block. Only the first of 130 patches was ever standardised, and the other 129
went into the projection with values ~2000× too large.

Fixed at the root, in `TensorApplyCPU`, so it holds for the CPU backend and for
every backend whose kernel falls back to it; the CUDA backend also gets a proper
row-broadcast kernel (`ts_binary_row_bcast_f32`) so the shape no longer leaves the
GPU at all. `ElementwiseRowBroadcastTests` covers all four operators.

### 4. GPT-OSS JSON mode answered the shape instead of the question

`{"city":"...","population":0}` — schema-valid, semantically empty, on every
backend. GPT-OSS speaks the harmony format: its first emitted token is
`<|channel|>`, it reasons in the `analysis` channel, and only then answers in
`final`. A grammar armed from token 0 **forbids the model's own first token**, so
it was pushed straight into a JSON object having done no reasoning at all.

`GrammarConstraint.ActivateAfter(trigger)` holds the constraint dormant — masking
nothing, feeding the grammar nothing — until the trigger text has been generated.
For harmony that is `final<|message|>`, so the analysis channel runs free and the
schema constrains only the answer. Mirrors llama.cpp's lazy grammar triggers.

| gpt-oss-20b, JSON schema `{city, population}` | result |
|---|---|
| before | `{"city":"...","population":1}` |
| after | `{"city":"Paris","population":2140526}` |

### 5. `TensorSharp.Cli --tools` crashed on the standard tool format

Found by running the CLI sweep rather than only the server one. The CLI
deserialized the `--tools` file straight into `List<ToolFunction>`, whose
`Parameters` is a flat `Dictionary<string, ToolParameter>`. A tool definition in
the **JSON Schema** shape — the one anyone copying a tool out of an API request
has, and the one the server already accepts —

```json
"parameters": { "type": "object", "properties": { "city": {...} }, "required": ["city"] }
```

made the deserializer try to read the schema's own `"type": "object"` as a
`ToolParameter` and the process died with an unhandled `JsonException` (exit
134) before the model was even loaded.

`ToolFunction.ParseList` now accepts the flat shape, the JSON Schema shape and
either of them inside the OpenAI `{"type":"function","function":{…}}` wrapper,
and the CLI reports a bad file as one error line instead of a stack trace.
`ToolFunctionParseTests` covers all three shapes plus the malformed cases.

### 6. GPT-OSS built one attention mask per layer

Found by moving the benchmark from 512- to 8,192-token prompts — a regression in
the whole-model prefill kernel added earlier in this same pass, invisible at the
old size.

The kernel gave every layer its own F16 `[window, N]` causal mask. GPT-OSS has
exactly **two** distinct geometries: the sliding-window one on even layers and the
full causal one on odd layers, so 24 masks were being built where 2 would do. At
an 8,192-token chunk that is ~200M fp16 host writes and ~380 MB of uploads per
forward, all of it duplicated. Masks are now shared by
`(wstart, wlen, is_swa, sliding_window)`.

| gpt-oss-20b, `--n-cpu-moe 0` | before | after | gain |
|---|---:|---:|---:|
| pp4096 | 7,055 | **13,964** | 1.98x |
| pp8192 | 5,825 | **12,925** | 2.22x |

It also flipped the prefill chunk choice back. With the mask cost in place a
512-token chunk beat 2,048 (7,419 vs 6,717 at pp8192) because it made each mask
smaller; with it gone, 2,048 wins again (13,898 vs 11,533), which is the default.

Greedy output is unchanged, and the GPT-OSS server suite passes 20/20 on
`ggml_cuda` (with and without offload), `ggml_vulkan` and `cuda`.

### Also

- **Redundant per-bind tensor init removed.** A whole-model graph binds ~600
  weights per call and each bind re-ran the backend's `init_tensor`, which for a
  quantized weight on ggml-cuda is a `cudaMemset` over the block padding. The
  cache entry now records the alloc size it was initialised at and repeat binds
  attach directly (`try_bind_cached_tensor`).
- **Two new diagnostics**, both env-gated and both used to produce the numbers in
  this document: `TS_GGML_PHASE_TIMING` (build / bind / alloc / upload / compute /
  download per whole-model kernel) and `TS_GGML_NODE_PROFILE` (per-op-type cost
  inside a whole-model graph).

## Correctness

Offload is only worth measuring if the answers stay right, and a perf fix is only
worth shipping if it does not move them. Both serving surfaces were swept across
all three GPU backends: **182 checks, all passing** — 110 through the server and
72 through the CLI.

### Server

Nineteen configurations (`TensorSharp.Server.Host`, OpenAI-compatible endpoint) × six
capabilities, with and without offload and with tensor parallelism:

| Configuration | short text | long text | multi-turn | tools | JSON mode | image |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| gemma4-26B `ggml_cuda` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-26B `ggml_cuda` `-ncmoe 16` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-26B `ggml_cuda` `--cpu-moe` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-26B `ggml_vulkan` `-ncmoe 16` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-26B `cuda` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-26B `ggml_cuda` `--tp 2 -ncmoe 8` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-E4B `ggml_cuda` (dense) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-E4B `cuda` (dense) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-E4B `ggml_vulkan` (dense) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen35-35B `ggml_cuda` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen35-35B `ggml_cuda` `-ncmoe 20` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen35-35B `ggml_vulkan` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen35-9B `ggml_cuda` (dense) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen35-9B `cuda` (dense) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen35-9B `ggml_vulkan` (dense) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| gptoss-20B `ggml_cuda` | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| gptoss-20B `ggml_cuda` `-ncmoe 12` | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| gptoss-20B `ggml_vulkan` | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| gptoss-20B `cuda` | ✅ | ✅ | ✅ | ✅ | ✅ | — |

**110 of 110 checks pass**, against the previous revision's 77 of 82 with five
image failures and two JSON-mode failures left open. The four GPT-OSS rows were
re-run after the JSON-mode fix landed — they were the only red cells in the first
pass of this sweep, all at `json_mode`; every other row is from the single
19-configuration run.

### The image check now measures vision, not OCR

The previous revision's image test showed a 512×256 PNG of the words "APPLE 42"
(with the "2" clipped by the frame) and asked *"What text is written in this
image?"*. That is an OCR test on rendered text, and it is a poor proxy for
whether the engine's vision path works: llama.cpp's own `llama-mtmd-cli` returns
the same wrong answer as TensorSharp on the same file for Qwen 3.5 9B
(`areplay`), token for token, so those five red cells were measuring the
checkpoint rather than the engine.

The test image is now a red circle, a blue square and the text "APPLE 42" with
margins, and the prompt asks the model to describe it. The check is that both
shapes arrive with the right colours — a property that fails loudly when the
embeddings are wrong and passes on every checkpoint whose vision tower works.
That is what caught the real bug: Gemma 4 on `--backend cuda` got the two shapes
right and then invented an apple and a grey background (see
[What changed](#3-gemma-4-vision-was-broken-on-the-pure-c-cuda-backend)).

### The same models through the CLI

`TensorSharp.TestMatrix` drives `TensorSharp.Cli` as a subprocess: 5 models × 3
backends × 5 features. (There is no `json_mode` here — structured output is a
server-request feature, and the server sweep above covers it.)

| Model | Backend | short text | long text | multi-turn | tools | image |
|---|---|:--:|:--:|:--:|:--:|:--:|
| gemma4-26B-A4B | `ggml_cuda` | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-26B-A4B | `ggml_vulkan` | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-26B-A4B | `cuda` | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-E4B | `ggml_cuda` | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-E4B | `ggml_vulkan` | ✅ | ✅ | ✅ | ✅ | ✅ |
| gemma4-E4B | `cuda` | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen35-35B-A3B | `ggml_cuda` | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen35-35B-A3B | `ggml_vulkan` | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen35-35B-A3B | `cuda` | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen35-9B | `ggml_cuda` | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen35-9B | `ggml_vulkan` | ✅ | ✅ | ✅ | ✅ | ✅ |
| qwen35-9B | `cuda` | ✅ | ✅ | ✅ | ✅ | ✅ |
| gptoss-20B | `ggml_cuda` | ✅ | ✅ | ✅ | ✅ | — |
| gptoss-20B | `ggml_vulkan` | ✅ | ✅ | ✅ | ✅ | — |
| gptoss-20B | `cuda` | ✅ | ✅ | ✅ | ✅ | — |

**72 of 72 cells pass.** The first pass of this sweep did not: `tools` aborted
on six cells before the model was even loaded (defect 5 above), and GPT-OSS's
`multi_turn` came back empty on a 32-token-per-turn budget (see below).

### Speculative decoding

The only drafter pair on this host is DeepSeek V4 Flash with its DSpark drafter
(3 stages, block size 5, target layers 40–42, drafter on device 1). Same prompt,
same 128 generated tokens, both GPUs:

| | tok/s | wall |
|---|---:|---:|
| no drafter | 52.7 | 2,429.6 ms |
| DSpark drafter | **73.5** | 1,740.5 ms |

**1.39x**, at acceptance 0.658 over a 5-token window — 117 drafted, 77 accepted,
51 verify steps and 26 rollbacks for 128 tokens. Unchanged by this work; checked
because the whole-model prefill graph shares the verify batch shapes.

### One harness correction

The matrix's multi-turn recall file gave each turn a 32-token budget. That is
fine for a model that answers directly and not for one that reasons first:
GPT-OSS spends its first ~30 tokens in the `analysis` channel, hits the cap
before reaching `final`, and returns empty content — which reads as a recall
failure but is a budget failure. Raised to 96, at which GPT-OSS answers "Alex"
and "teal" on turns 2 and 3 with room to spare. The engine was never wrong here;
the test was measuring the wrong thing, the same way the old image check was.

### Unit coverage added

- `ElementwiseRowBroadcastTests` — Add/Sub/Mul/Div with a `[cols]` right operand
  must reach every row, plus an in-place case and a same-shape control.
- `GrammarConstrainedDecodingTests` — a lazily-armed constraint masks nothing
  before its trigger, enforces from the token after it, and does not feed the
  prelude to the grammar; an un-triggered constraint still enforces from token 0.

## The prefill curve: why the ratio moves with prompt length

The tables above are at 4,096 and 8,192 tokens, where TensorSharp is ahead of
llama.cpp's default configuration. It was not always: the ratio climbs steadily
with prompt length, because the difference is cost that does **not** scale with
the token count — dominant at 128 tokens, invisible at 8,192. This section is the
measurement of that cost, and what came off it.

Two pieces of it were removed in a follow-up pass (see
[Cutting the fixed cost](#cutting-the-fixed-cost)); prefill throughput before and
after, best of 3 cold runs at `MAX_CONTEXT=1024`, `ggml_cuda`:

| model | | pp128 | pp512 |
|---|---|---:|---:|
| gemma-4-26B-A4B | before | 2,293 | 6,663 |
| | **after** | **3,021** | **8,214** |
| | llama.cpp | 4,409 | 10,571 |
| | ratio before → after | 0.52x → **0.69x** | 0.62x → **0.78x** |
| Qwen 3.5-35B-A3B | before | 1,583 | 5,060 |
| | **after** | **2,150** | **6,108** |
| | llama.cpp | 3,450 | 8,048 |
| | ratio before → after | 0.45x → **0.62x** | 0.62x → **0.76x** |

**+21% to +36%**, and the ordering is unchanged: TensorSharp still trails at
pp128/pp512 and leads from pp2048 up. Greedy output is token-identical before and
after on both models, on `ggml_cuda`, `ggml_vulkan` and with `--n-cpu-moe`.

The earlier measurement of the same curve, for the fixed/marginal split:

| model | pp128 | pp512 | pp2048 |
|---|---:|---:|---:|
| gemma-4-26B-A4B TensorSharp | 2,293 | 7,683 | **11,872** |
| gemma-4-26B-A4B llama.cpp | 4,434 | 10,610 | 10,869 |
| ratio | 0.52x | 0.72x | **1.09x** |
| Qwen 3.5-35B-A3B TensorSharp | 1,583 | 5,060 | **8,768** |
| Qwen 3.5-35B-A3B llama.cpp | 3,479 | 8,157 | 8,212 |
| ratio | 0.45x | 0.62x | **1.07x** |

Fitting `time = fixed + marginal × tokens` over the 512→2048 range:

| | fixed cost per forward | marginal cost per token |
|---|---:|---:|
| gemma-4-26B TensorSharp | **31.4 ms** | 0.069 ms |
| gemma-4-26B llama.cpp | 1.5 ms | 0.091 ms |
| Qwen 3.5-35B TensorSharp | **57.1 ms** | 0.086 ms |
| Qwen 3.5-35B llama.cpp | 0.6 ms | 0.122 ms |

TensorSharp's *per-token* cost is already lower than llama.cpp's on both models —
the kernels are not the problem. What it pays instead is 31–57 ms of per-call
overhead, which is half the wall clock at 512 tokens and noise at 2048. (The
fixed cost tracks layer count: 30 layers → 31 ms, 48 layers → 57 ms, i.e. about
1.2 ms per layer.)

The dense controls say the same thing from another angle — a dense model has no
MoE routing, expert gather or expert sum, so its graph is far smaller:

| model (`--n-cpu-moe 0`) | TS pp512 | llama pp512 | ratio | TS tg128 | llama tg128 | ratio |
|---|---:|---:|---:|---:|---:|---:|
| gemma-4-E4B Q8_0 (dense) | 13,084 | 12,818 | **1.02x** | 165.6 | 177.2 | 0.93x |
| Qwen 3.5-9B Q8_0 (dense + GDN) | 7,170 | 10,067 | 0.71x | 130.8 | 139.7 | 0.94x |

`TS_GGML_PHASE_TIMING=1` splits the call (gemma-4-26B, pp512, steady state):

```
[phase] gemma4 MoE model verify total=63.88ms build=0.58 expand=0.19 \
        bind=7.86 alloc=0.21 compute=54.28 download=0.75
```

- **bind = 7.9 ms.** The graph is rebuilt per call, so every weight in it (about
  20 per layer over 30 layers, plus the LM head) is re-bound to its cached device
  buffer on every prefill. llama.cpp binds once at load into a model buffer.
- **compute = 54.3 ms**, of which ~22 ms does not scale with N. `TS_GGML_NODE_PROFILE`
  puts the verify graph at **2,791 nodes for 30 layers** — 93 per layer, of which
  ~55 launch a kernel: 12 `MUL`, 11 `RMS_NORM`, 10 `ADD`, 5 `MUL_MAT`, 2
  `MUL_MAT_ID` and the rest routing/attention plumbing. The elementwise count is
  where the launches are: the MoE expert weighting and its `n_used`-way sum, the
  routing gather and the per-head norms are each separate ops.

### Cutting the fixed cost

Two of those were attacked directly; both come from comparing what ggml-cuda does
for llama.cpp's graph and not for ours.

**1. The MoE gating chain was invisible to ggml-cuda's fused top-k kernel.**
ggml-cuda collapses a router's entire gating chain into one kernel
(`ggml_cuda_op_topk_moe`), but `ggml_cuda_topk_moe_fusion` recognises it by
literal node sequence — `SOFT_MAX → RESHAPE → ARGSORT → VIEW → GET_ROWS →
RESHAPE → SUM_ROWS → CLAMP → DIV → RESHAPE`, which is what llama.cpp's
`build_moe_ffn` emits. TensorSharp used `ggml_top_k`, which is its own op
(`GGML_OP_TOP_K`), so the pattern never matched. Three things followed from that:
the chain ran as ~10 separate kernels per layer instead of 1; `ggml_top_k` fully
sorts all `n_expert` scores per token where the fused kernel does a partial
selection; and its compact `[k, tokens]` output could not have been used by the
fused kernel anyway, which recovers `n_expert` from the ids' row stride
(`ids->nb[1] / ids->nb[0]`) and so *needs* the argsort view.

`build_topk_moe_routing` now emits llama.cpp's shape, and the callers expand the
chain into the graph as a unit — left to the expert matmuls to pull in, the ids
half and the weights half are emitted at different points and nothing fuses.

**2. Every weight was re-bound to its device buffer on every prefill.**
llama.cpp binds its weights once at load into a model buffer and never touches
them again; TensorSharp rebuilds the graph per call and re-bound ~450 weights
each time, at ~17 µs apiece. Not slow for any single reason — two mutexes and a
backend alloc-size query in the lookup, then `ggml_backend_tensor_alloc` running
the backend's `init_tensor`, which on ggml-cuda is a `cudaMemset` over the quant
padding — just slow 450 times, for 8 ms a call. The cache entry now records the
address the first bind resolved to, so a repeat bind is two assignments and no
backend calls at all.

| gemma-4-26B pp512, phase | before | after |
|---|---:|---:|
| weight bind | 8.2 ms | **6.1 ms** |
| compute | 52.8 ms | **49.9 ms** |
| whole kernel call | 63.8 ms | **57.1 ms** |

**What is left.** The remaining fixed cost is ~6 ms of weight binding plus ~1.2 ms
of graph build/alloc/download inside the kernel, and ~5.7 ms on the C# side of the
call (host embedding gather, logits copy). Removing all of it would put the call
at ~50 ms against llama.cpp's 47.3 at 512 tokens — i.e. **graph reuse alone does
not overtake llama.cpp there**; the compute would have to come down too, and at
49.9 ms it is already within ~8% of llama.cpp's entire forward.

That ~8% is the same thing the matched-ubatch control above exposes at long
prompts: with `-ub 2048` llama.cpp gets 14,485 tok/s out of a 2,048-token batch
on gemma-4-26B where we get 11,119, and our own chunk sweep shows we are already
on the plateau. So the outstanding item is per-batch compute efficiency, not
batching and not host overhead. The two concrete leads are a shape-keyed reuse of
the built-and-bound graph (worth the ~7 ms of host cost per call) and closing the
~150-node gap against llama.cpp's 2,644-node graph (ours is 2,791) — mostly the
5 `CONT` copies per layer in the attention path.

Decode is a narrower story: the dense models are at 0.93–0.94x and the MoE models
at 0.73–0.82x, against the same per-token whole-model graph submission.

### Things ruled out along the way

- **Prefill is not chunked at 512.** Gemma 4's chunk is 2048; GPT-OSS's is now
  2048 too.
- **The kernels are the same.** Both engines link the same vendored ggml, and the
  MMQ-vs-cuBLAS decision is made inside it from the same shapes.
- **The GPT-OSS whole-model prefill graph does not cost decode.** A/B on the same
  cells with `TS_GPTOSS_MODEL_PREFILL=0`:

  | gpt-oss `--n-cpu-moe` | prefill graph | pp512 | tg128 |
  |---|---|---:|---:|
  | 6 | on | **4,464** | 99.2 |
  | 6 | off (per-layer) | 995 | 95.8 |
  | 12 | on | **3,102** | 69.3 |
  | 12 | off (per-layer) | 834 | 68.7 |

  Decode is unchanged (and the offloaded decode figures in the results tables sit
  below these because that sweep ran while the host was busier — see the note
  about offloaded variance).

## Reproducing

```bash
# TensorSharp
MAX_CONTEXT=1024 TensorSharp.Cli --model <model.gguf> --backend ggml_cuda \
    --n-cpu-moe <N> --benchmark --bench-prefill 512 --bench-decode 128 \
    --bench-runs 4 --bench-fixed-tokens

# llama.cpp
llama-bench -m <model.gguf> -p 512 -n 128 -r 3 -ncmoe <N>

# CLI capability sweep (models x backends x features)
dotnet TensorSharp.TestMatrix/bin/TensorSharp.TestMatrix.dll --config <matrix.json> \
    --features short_text,long_text,multi_turn,tools,image --env-vars none

# speculative decoding
TensorSharp.Cli --model <model.gguf> --backend ggml_cuda \
    --draft-model <drafter.gguf> --input <prompt.txt> --max-tokens 128
```

The server sweep is a per-host script rather than a checked-in harness: it starts
`TensorSharp.Server.Host` per configuration and drives `/v1/chat/completions` for the
six capabilities. `--env-vars none` on the matrix runner turns off the offload
cross-product, which the server sweep already covers.

Diagnostics worth knowing:

| Env var | What it does |
|---|---|
| `TS_GGML_PHASE_TIMING=1` | Splits a whole-model prefill kernel into graph build / weight bind / buffer alloc / upload / compute / download. |
| `TS_GGML_NODE_PROFILE=1` | Runs a whole-model graph node by node and dumps the cost per op type. `TS_GGML_NODE_PROFILE_EVERY` sets the dump interval (default 64 graphs). |
| `TS_VISION_TRACE=1` | Per-stage mean/rms/min/max of the Gemma 4 vision encoder — how the backend divergence below was located. |
| `TS_HOST_MOE_TIMING=1` | Splits an offloaded layer into setup vs host matmul; prints streamed bytes, used experts, effective bandwidth. `=2` adds a per-weight transfer rate. |
| `TS_HOST_MOE_DEBUG=1` | Prints the segment plan and each seam's activation norms. |
| `TS_HOST_MOE_VERIFY=1` | Runs the on-GPU expert chain alongside the host one and reports the divergence. |
| `TS_GPTOSS_MODEL_PREFILL=0` | Falls back to GPT-OSS's per-layer prefill (for A/B). |
