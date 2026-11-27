# GPT OSS

[← back to model index](README.md) | [中文](gptoss_zh-cn.md)

| Property | Value |
|---|---|
| Provider | OpenAI |
| GGUF architecture keys | `gptoss`, `gpt-oss` |
| Source class | [`GptOssRuntimeModel`](../../TensorSharp.Models/Models/GptOss/GptOssRuntimeModel.cs) on the shared decoder runtime (`StandardDecoderModel`; its rotary and routed experts in [`GptOssRuntimeModel.Graph.cs`](../../TensorSharp.Models/Models/GptOss/GptOssRuntimeModel.Graph.cs), whole-forward graphs on every executor). The legacy `GptOssModel` was removed (§7, §11). |
| Example models | gpt-oss-20b |
| Modalities | Text only |
| Thinking mode | Yes (Harmony format: `<\|channel>analysis ... <\|channel>final`) |
| Tool calling | Yes (Harmony `commentary` channel — `to=functions.NAME`); eligible for skills, the code tools and server-side [sub-agent delegation](../multi_agent.md) |
| Speculative decoding | n-gram drafts (`--spec`, `--spec-type ngram`) and the published gpt-oss-20b DFlash block drafter (`--draft-model`), verified in windows that roll back exactly. The legacy model had no speculative trunk. See §8. |
| Batched / paged forward | **Default ON** (the shared runtime's batched slots: each request's own device-resident K/V, with a ring for each sliding-window layer; prompt chunks and decode rows of several requests run in one step) — `--no-continuous-batching` forces the per-sequence path. Batching is kept under `--tp` and `--layer-split`. See §8. |
| Expert offload | `--n-cpu-moe` / `--cpu-moe` on one device or over `--layer-split` stages, every backend; refused with `--tp`. See §10. |
| Output parser | `HarmonyOutputParser` (always required) |

## Downloads

Verified GGUF pointer:

| Model | HF repo | Recommended file | Notes |
|---|---|---|---|
| gpt-oss-20b (MoE) | [ggml-org/gpt-oss-20b-GGUF](https://huggingface.co/ggml-org/gpt-oss-20b-GGUF) | `gpt-oss-20b-MXFP4.gguf` (12.110 GB) | Native MXFP4 expert quantization; text only. Thinking is always on (Harmony analysis channel) and tool calling is supported. Official upstream: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) (Apache-2.0). |

Command-line download (one line per file; requires `pip install -U huggingface_hub`):

```bash
python -m pip install -U huggingface_hub
hf download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf --local-dir models
```

CLI one-shot (the text prompt comes from a file via `--input`; CLI sampling
defaults to greedy and `--max-tokens` defaults to 100):

```bash
dotnet run --project TensorSharp.Cli -c Release -- --model models/gpt-oss-20b-MXFP4.gguf \
  --input prompt.txt --max-tokens 512 --backend ggml_cuda
```

Server (Web UI + OpenAI/Ollama-compatible APIs on `http://localhost:5000`):

```bash
dotnet run --project TensorSharp.Server.Host -c Release -- --model models/gpt-oss-20b-MXFP4.gguf \
  --backend ggml_cuda --max-tokens 4096
```

## 1. Origin and intent

GPT OSS is OpenAI's open-weights MoE family. Several design choices set it
apart from the other architectures in TensorSharp:

- **MoE in every block**, with TopK routing followed by a softmax over the
  selected experts only (in contrast to Gemma 4, which softmax-then-TopKs).
- **Attention sinks** — a learned per-head bias that participates in the
  softmax as a "virtual token" with zero K and zero V, providing an
  always-available attention target.
- **Bias on every linear projection** (Q, K, V, attn output, gate, up,
  down, router). This is unique among the architectures TensorSharp supports.
- **Clamped GLU activation** — the FFN uses a SiLU-like activation
  `x · sigmoid(α · x) · (y + 1)` with the gate clamped to `[-∞, 7]` and the
  `up` value clamped to `[-7, 7]`.
- **Harmony output format** — every output is wrapped in
  `<|channel>analysis ... <|channel>final ...` tags. The output parser is
  marked `AlwaysRequired = true` because there is no "no thinking" mode.
- **MXFP4 expert weights** (GGML quantization type 39) — a 4-bit
  microscaling format used by the gpt-oss-20b release.

## 2. Model architecture

```
                tokens (int[])
                      │
              token_embd.weight
                      │
        ┌──── × NumLayers ───────────────────────────────┐
        │  RMSNorm(attn_norm)                             │
        │  Q+bias, K+bias, V+bias (no QK-norm)             │
        │  RoPE_NeoX with YaRN scaling                     │
        │  attention with sinks                            │
        │  attn_output + bias ─► residual                  │
        │                                                  │
        │  RMSNorm(post_attention_norm)                    │
        │  router_linear+bias                              │
        │  topK then softmax(selected experts)             │
        │  per-expert: swiglu_oai(gate+bias, up+bias)      │
        │              ─► down+bias                        │
        │  weighted sum of expert outputs                  │
        │  ─► residual                                     │
        └──────────────────────────────────────────────────┘
                      │
              RMSNorm(output_norm)
                      │
              LM head (output.weight)
                      │
                      ▼
                   logits
```

## 3. Forward graph

For each layer L:

```
hidden ─► RMSNorm(attn_norm.weight, eps)
       ─► Q+bias, K+bias, V+bias (each linear is a matmul + per-row bias add)
       ─► RoPE_NeoX with Yarn scaling
            • origCtxLen = Config.OriginalContextLength (4096)
            • freqScale  = 1 / RopeScale
            • beta_fast  = 32, beta_slow = 1
       ─► append (K, V) to KV cache
       ─► attention with sinks:
            scores = Q × K^T
            scores ← causal mask (and the 128-token window on even layers)
            sinks  = attn_sinks.weight                   # [numHeads]
            sums   = exp(scores - max) + exp(sinks - max)
            scores = exp(scores - max) / sums            # softmax including sink
            out    = scores × V
       ─► attn_output.weight matmul + bias → o
       ─► residual = hidden + o

       moeInput = (last layer of a prompt step) ? its sequences' last rows : residual
       moeInput ─► RMSNorm(post_attention_norm)
                ─► router_linear + bias
                ─► topK(4) of the logits
                ─► softmax over the 4
                ─► For each selected expert e (one mul_mat_id + add_id per projection):
                       up   = ffn_up_exps[e]   × moeInput + up_bias[e]
                       gate = ffn_gate_exps[e] × moeInput + gate_bias[e]
                       h    = swiglu_oai(gate, up)          # min(gate, 7), clamp(up, ±7), alpha 1.702
                       out_e = ffn_down_exps[e] × h + down_bias[e]
                ─► weighted Σ out_e using router weights
                ─► residual += weighted_sum
```

After all layers:

```
hidden ─► the sequences' last rows (a prompt step; the last layer's MoE already ran on them only)
       ─► RMSNorm(output_norm.weight, eps)
       ─► output.weight matmul → logits
```

## 4. Components in detail

### 4.1 Attention

- **GQA**: 64 query heads over 8 KV heads, head dimension 64, on gpt-oss-20b.
- **Biases on every projection.** Q, K and V fuse into one `attn_qkv` product at load, layer by layer, where the three
  share a type, and their biases into one `[q | k | v]` bias added after it. A layer whose three types differ keeps
  three products, each with its own bias. unsloth's requants mix them, for example IQ4_NL q/k with a Q5_0 v.
  The output projection adds `attn_output.bias`.
- **No QK-norm.**
- **Attention sinks** on every layer: `attn_sinks.weight` (`[numHeads]`) is one more logit in each head's softmax, with
  no value. It is the attention op's fifth operand. The GGML builder passes it to `ggml_flash_attn_ext_add_sinks`. The
  tensor executors (`cuda`, `mlx`, `cpu`) run a sink layer on their generic products, the sink as one more column of each
  row's softmax (`TensorGraphExecutor.FlashAttentionWithSinks`).
- **RoPE**: NeoX with YaRN (`freq_base` 150000, factor 32 over an original context of 4096, `beta_fast` 32,
  `beta_slow` 1), as one `GraphRotary` spec.
- **Attention pattern**: even layers attend a 128-token sliding window, odd layers the whole context. A sliding layer's
  K/V live in a ring of 768 cells a slot (§9).

### 4.2 FFN — clamped SwiGLU (`swiglu_oai`)

```
g = min(gate + gate_bias, 7)
u = clamp(up + up_bias, -7, 7)
h = g · sigmoid(1.702 · g) · (u + 1)
out = down · h + down_bias
```

One graph op, `GraphOp.SwigluOai` (alpha 1.702, limit 7): `ggml_swiglu_oai` on GGML, a managed composition on the tensor
executors. The per-expert biases are `GraphOp.AddId` (`ggml_add_id`): row `e` of a `[width, experts]` bias added to each
route of expert `e`.

### 4.3 MoE routing

- 32 experts, 4 a token (gpt-oss-20b).
- `ffn_gate_inp · x + ffn_gate_inp.bias`, the top 4 of those logits, then a softmax over the 4 (llama.cpp's
  `SOFTMAX_WEIGHT`). This is **top-k-then-softmax**, the inverse of Gemma 4's softmax-then-top-k.
- The experts are read in place from the GGUF's 3-D stacks (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, MXFP4)
  with their `[width, experts]` F32 biases.
- The graph follows llama.cpp's `openai-moe-iswa.cpp` op for op: up `mul_mat_id` + `add_id`, gate `mul_mat_id` +
  `add_id`, `swiglu_oai`, down `mul_mat_id` + `add_id`, the routes' weights, their sum. ggml-cuda's fusions match that
  chain: the top-k MoE selection, `mul_mat_id` + `add_id` + GLU, and the experts' weighted sum.
- On direct CUDA the router and experts are one region (`GraphFusion.OaiRoutedExperts`), served by the MoE engine
  (`CudaMoe`): the top-k softmax on the device, then the grouped expert kernels with two GPT-OSS kernels for the biased,
  clamped SwiGLU and the down bias (`ts_moe_oai_glu_bias_f32`, `ts_moe_expert_bias_add_f32`). Decode rows and verify
  windows run its per-slot kernels.
- A prompt step's last layer runs its experts on the rows the head reads only (`PrunesExpertLastLayer`), as llama.cpp's
  graph and the legacy model did.

### 4.4 Tokenizer

GPT-4o BPE pre-tokenizer with `\p{N}{1,3}` number grouping. Handled by the
runtime tokenizer when `tokenizer.ggml.pre == "gpt-4o"`.

## 5. Parameters and settings

| Key | Type | Meaning |
|---|---|---|
| `gptoss.expert_count` | uint32 | Number of experts (32) |
| `gptoss.expert_used_count` | uint32 | TopK experts per token (4) |
| `gptoss.attention.sliding_window` | uint32 | SWA window size (128) |
| `gptoss.expert_feed_forward_length` | uint32 | Expert FFN dim |
| `gptoss.rope.scaling.original_context_length` | uint32 | Yarn original context (4096) |
| `tokenizer.ggml.pre` | string | Pre-tokenizer type (`gpt-4o`) |

## 6. Weight naming convention

```
token_embd.weight
output_norm.weight
output.weight

blk.{L}.attn_norm.weight
blk.{L}.attn_q.weight    / attn_q.bias
blk.{L}.attn_k.weight    / attn_k.bias
blk.{L}.attn_v.weight    / attn_v.bias
blk.{L}.attn_qkv.weight  / attn_qkv.bias       # after load, where q/k/v share a type
blk.{L}.attn_output.weight / attn_output.bias
blk.{L}.attn_sinks.weight                      # per-head sink logits [numHeads]
blk.{L}.post_attention_norm.weight             # the FFN norm

blk.{L}.ffn_gate_inp.weight / ffn_gate_inp.bias   # router [hidden, experts] + [experts]
blk.{L}.ffn_gate_exps.weight / ffn_gate_exps.bias # [hidden, ff, experts] + [ff, experts]
blk.{L}.ffn_up_exps.weight   / ffn_up_exps.bias   # [hidden, ff, experts] + [ff, experts]
blk.{L}.ffn_down_exps.weight / ffn_down_exps.bias # [ff, hidden, experts] + [hidden, experts]
```

## 7. TensorSharp implementation: the shared decoder runtime

GPT-OSS runs on the shared decoder runtime (`StandardDecoderModel`), as Gemma 4, Qwen 3.5, Nemotron-H and Mistral 3 do.
[`GptOssRuntimeModel`](../../TensorSharp.Models/Models/GptOss/GptOssRuntimeModel.cs) declares the family:
- even layers' sliding windows;
- the sinks and the biased projections, which the runtime's attention serves;
- `post_attention_norm` as the FFN norm;
- the YaRN rotary;
- the routed experts of §4.3 (`EmitFeedForward`).

The runtime provides batching, the prefix cache, checkpoints, placements, speculation and expert offload. The family runs
whole-forward graphs only (`RequiresGraphs`): every step is one graph on the backend's executor, whether a prompt chunk,
a decode step, batched slots or a verify window. That executor is the GGML graph builder on `ggml_*`,
`CudaGraphExecutor` on `cuda`, `MlxGraphExecutor` on `mlx` and the managed executor on `cpu`.

What the migration added, all in TensorSharp-owned code (ggml upstream unchanged):
- **Graph IR**: `GraphOp.SwigluOai`, `GraphOp.AddId` on the tensor executors, and a sink operand on the attention op.
- **ggml-cuda fusions in prompt steps**: `GraphProgram.BackendFusions` runs ggml-cuda's graph optimizer on a prompt step,
  after declaring the allocation dependencies it needs. Without it, the experts' weighted sum fused in 1 of 24 layers of a
  prompt step and ran as a broadcast multiply and three adds in the rest.
- **Direct CUDA**: the `OaiRoutedExperts` region on the MoE engine (§4.3).
- **Host experts with biases** (§10).
- **DFlash drafters with biased projections** (§8).

The legacy `GptOssModel` was removed after the validation in §11, with everything that served only it:
- its whole-model decode, token-batched decode and prefill graphs (`ggml_ops_gptoss_decode.cpp`,
  `ggml_ops_gptoss_batched.cpp`, `ggml_ops_gptoss_prefill.cpp`, `ggml_ops_gptoss_kv.h`);
- the fused attention-layer prefill and its device K/V windows (`ggml_ops_transformer_prefill.cpp`);
- the paged attention with sinks (`TSGgml_PagedAttentionForwardWithSinks`, `ManagedPagedAttention.ForwardWithSinks`);
- the MLX `gather_qmm` stacked-expert path and the MLX sink-decode, `swiglu_oai` bias-gather and biased weighted-sum
  kernels;
- the K/V host reader of its linear-to-paged migration.

Its switches went too: `TS_GPTOSS_FUSED_DECODE_MAX_CTX`, `TS_GPTOSS_FUSED_ATTN_MAX_SEQ`, `TS_GPTOSS_PREFILL_F16` and
`TS_GPTOSS_PREFILL_CHUNK` (prompt steps follow `TS_PREFILL_CHUNK`, and the rings cap a step at 512 rows). Setting one
now stops startup and names the change (`RemovedCliFlags`), as the earlier `TS_GPTOSS_*` switches already did.

## 8. Serving mechanisms

| Mechanism | Backends | Notes |
|---|---|---|
| Continuous batching | all | Batched slots: each request's own K/V (a 768-cell ring for each sliding layer). Concurrent requests' prompt chunks and decode rows run in one step. |
| Prefix cache, checkpoints | all | The runtime's holder adapter: captures, clones, donated holders, export/import. A rewind past a whole Harmony answer is exact while the rings still hold the positions the next rows' windows read (§9). |
| `--tp N` | cuda, ggml_cuda, ggml_vulkan | Per-rank graphs. Attention is sharded by heads (the sinks with them), and every expert by its width (gate/up columns, down rows; the down bias added once). Batching is kept. gpt-oss-20b's MXFP4 down rows are 90 blocks, so `--tp 2` splits them and `--tp 4` / `--tp 8` are refused at load with the reason. Covered on simulated devices only (§11). |
| `--layer-split N` | cuda, ggml_cuda, ggml_vulkan | Whole-layer stages, batching kept. New with the runtime. |
| Hybrid (`--tp` x `--layer-split`) | none (gap) | The graph-only family has no graphs for tensor-parallel stages yet. |
| `--n-cpu-moe` / `--cpu-moe` | all | On one device or over stages; refused with `--tp`. See §10. |
| Speculation | all | n-gram drafts and the DFlash block drafter (below), verified in windows whose rings roll back exactly. |

**DFlash drafter.** [z-lab/gpt-oss-20b-DFlash](https://huggingface.co/z-lab/gpt-oss-20b-DFlash) (0.8B, Apache-2.0)
attaches with `--draft-model`. A Q5_K_M GGUF is
[EntityDeletr/gpt-oss-20b-DFlash-GGUF](https://huggingface.co/EntityDeletr/gpt-oss-20b-DFlash-GGUF) `gpt-oss-20b-DFlash.gguf`
(0.599 GB, revision `93855ebf`):

```bash
hf download EntityDeletr/gpt-oss-20b-DFlash-GGUF gpt-oss-20b-DFlash.gguf --local-dir models
dotnet run --project TensorSharp.Cli -c Release -- --model models/gpt-oss-20b-MXFP4.gguf \
  --draft-model models/gpt-oss-20b-DFlash.gguf --backend ggml_cuda --input prompt.txt
```

It has 8 layers, an 8-row block and q/k norms. Its Q, K, V and output projections carry biases, which the runtime's
drafter now reads (`DfAttnQBias` ... `DfOutputBias`). It uses the target's YaRN rotary and reads trunk layers 2, 7, 12, 17
and 22.

Measured on one A40 (`ggml_cuda`, greedy, 5 prompts of 79-96 tokens, 128 generated tokens each):
- **DFlash, 7-token windows:** 60-93 % of the drafts accepted, decoding at 0.84-1.26x the plain rate. It is fastest on
  code and arithmetic, slowest on free prose.
- **n-gram, 4-token windows:** 0.95-1.11x.
- **Parity:** a speculative stream follows plain greedy decoding until a verify window's kernels flip a close pick
  (`SpeculativeDecodingMatchesPlainGreedy` covers the synthetic files).

The speculation cost governor (`SpeculationCostGovernor`) measures drafting against plain steps and parks it where it
loses; the runs above went through it. TensorAgent, whose speculation
setting defaults to n-gram drafts, now speculates on GPT-OSS too (its catalog lists gpt-oss-20b as experimental).

## 9. Memory and KV cache

- **Full-attention layers** (the odd ones): each slot's K/V holds the whole context, F16 by default
  (`--kv-cache-dtype`).
- **Sliding-window layers** (the even ones): a ring per slot of R = 768 cells (the 128-token window plus a 512-row step,
  aligned to 256). Position p lives in cell p mod R. A request shorter than R uses the ring as a linear cache. A step
  never writes more than 512 rows of one sequence, so a longer prompt runs in 512-row steps. gpt-oss-20b's 12 rings cost
  9.4 MB a slot in F16, and a token costs the 12 full layers' 24 KiB. The legacy model's whole-context caches cost
  48 KiB a token.
- **Quantized K/V**: a `q8_0` request is kept on the GGML backends. On gpt-oss-20b it moved the mean NLL by -0.0004 on
  `ggml_cuda` (384 tokens: 2.3060 -> 2.3056) and by +0.008 on `ggml_cpu` (160 tokens), and `ggml_cuda` decoded at
  158 tok/s. The tensor executors' graphs do not run the rings over a block-quantized cache (direct CUDA's q8_0 row writes
  do not wrap), so there a `q8_0` request becomes f16 with a notice. A `q4_0` request becomes f16 everywhere: on
  `ggml_cuda` it cost 0.48 nat a token (2.3060 -> 2.7811). The legacy model made every block type f16.
- **Rewinds**: a rewind keeps exactly the positions the next rows' windows read, and is refused once a ring has
  overwritten them (`CanTruncateKVCache`). Harmony drops the analysis channel when it re-renders a past answer, so the
  next turn rewinds the whole previous answer. That rewind is exact while the conversation holds fewer positions than a
  ring's cells, and the prefix cache's checkpoints serve the rest.
  `ARewindPastALongAnswerContinuesAsACacheThatNeverHeldIt` pins it.
- **K/V block snapshots** (`TryExtractKVBlock`) are not offered: a ring is not position-addressed. The prefix cache uses
  the runtime's slot clones and checkpoints instead. The real-model conformance script passes on gpt-oss-20b
  (`GptOss20B_PassesTheConformanceScript`).
- **Weights** stay in their GGUF formats on every executor (MXFP4 experts, Q8_0 attention and head on the ggml-org file).

## 10. Expert offload (`--n-cpu-moe` / `--cpu-moe`)

An offloaded layer's routed experts stay in host memory, and the step's graph pauses for them (`GraphOp.MoeHostExperts`
with GPT-OSS's activation and the three biases). The router and the selection stay on the device.
- **GGML backends:** the host MoE segment computes them from the GGUF mapping with ggml's CPU kernels: the biased gate
  and up products, the clamped SwiGLU, then the biased down product. The one-token decode team takes decode rows; the
  batched CPU MoE graph takes prompt chunks.
- **Tensor executors:** the runtime's expert host computes them with the same kernels (`TSGgml_HostExpertRowsOai`,
  through the same decode team).
- **Direct CUDA** keeps a step on the device:
  - Decode rows and short chunks ring the host through the expert doorbell (`CudaExpertDoorbell`), which computes the
    host's routes while the device runs its own. The device then adds every route's down bias, the host's included.
  - A layer's call stages at most one host expert over the link (`OaiGpuMisses`): a gpt-oss-20b expert is 13.2 MB of
    MXFP4. Measured decode rates: 14.0 tok/s staging all four routes, 28.7 at two, 31.6 at one, 29.7 at none.
  - Prompt chunks of 32 rows or more prefetch every host expert of the next layer by DMA while the current one runs
    (`OaiPrefetchMinRows`): a chunk that long selects about 98 % of the experts anyway. The grouped MoE kernels then run
    them on the device.

With `--tp`, offload is refused at load: the per-rank graphs keep their experts on the devices.

Measured on one A40 (ggml `bb877c0c`), 12 of 24 MoE layers offloaded (`TS_N_CPU_MOE=12`), in a container with 7.6 CPUs
(5 host-expert threads):

| Backend | Prefill 512 tok/s | Decode tok/s | NLL resident -> offloaded |
|---|---|---|---|
| ggml_cuda, legacy -> runtime | 1814-2415 -> 2116-2460 | 33.3-39.5 -> 34.0-41.4 | — |
| cuda, runtime | 551 (547 at 2048) | 29.2-31.6 | 2.7706 -> 2.7700 |

The ranges span two sessions. The legacy model ignored `TS_N_CPU_MOE` on `cuda`: it kept every expert on the GPU (prefill 723 tok/s,
decode 51 tok/s), so direct-CUDA offload is new. Greedy text on `cuda` with offload follows the resident run until near ties
(top-2 margins 0.03-0.07 at the first difference).

## 11. Validation of the migration

Dependency: ggml `bb877c0c51f38c52ee2d5e41d58e3d0c20f3fed4`, upstream and unchanged. `ggml_swiglu_oai`, `ggml_add_id`
and `ggml_flash_attn_ext_add_sinks` are its own ops. The CUDA PTX baseline was rebuilt with CUDA 13.0.88 for the two new
kernels.

**Synthetic fixtures** (`GptOssSyntheticModelBuilder`, `GptOssRuntimeTests`, 41 tests). The four cases are MXFP4 experts,
Q8_0 experts, a long one (window 16 over 83 tokens) and one sized for `--tp`. Each is held to the legacy model's logits,
frozen on the managed backend before it was deleted (`Fixtures/GptOss/LegacySynthetic/managed.json`). `ggml_cpu`,
`ggml_cuda` and direct `cuda` are checked against the managed backend. Also covered:
- chunked prefill, every placement over simulated devices, batched slots, concurrent requests;
- checkpoints, rewinds past a long answer, speculation;
- host experts on every executor (on direct CUDA: decode rows through the doorbell and prompt chunks staged on the
  device, with no step falling back to the executor's host path);
- mixed Q/K/V types fusing layer by layer;
- the refusal of offload under `--tp`.

The direct-CUDA prefill suite (`DirectCudaPrefillModelTests`) gained GPT-OSS's verify rows, which are bit-identical to
its decode rows.

**Real checkpoint** (gpt-oss-20b MXFP4, one A40). Teacher-forced argmax agreement with the legacy model is 5 prompts'
greedy steps, the legacy tokens forced through both. NLL is 448 tokens of the README, reported as perplexity. Speeds are
the CLI's benchmark, ABBA-ordered:

| Backend | Argmax vs legacy | Perplexity legacy -> runtime | Decode tok/s (after 512) | Prefill 512 tok/s | Decode tok/s after 4K | Prefill 4K tok/s |
|---|---|---|---|---|---|---|
| ggml_cuda | 551/561 (near ties, top-2 margin < 0.15) | 10.579 -> 10.546 | 141-150 -> 155-162 | 6095-6348 -> 6191-6531 | 131 -> 157.5 | 5923-6447 -> ~6440 |
| cuda | 555/561 | 10.620 -> 10.576 | 51 -> 118-120 | ~730 -> ~1750 | — | — |

- **Logits** against an F32 managed-CPU reference: the legacy model and the runtime are equally close (KL ≈ 0).
- **Server** (`TensorSharp.Server`, aggregate decode tok/s):

  | Requests | ggml_cuda legacy -> runtime | cuda legacy -> runtime |
  |---|---|---|
  | 1 | 107.8 -> 140.6 | 39.4 -> 89.3 |
  | 4 concurrent | 125.0 -> 212.7 | 4.7 -> 127.7 |
  | 8 concurrent | 151.0 -> 230.4 | 4.7 -> 132.3 |

  Concurrent outputs part from solo ones at near-tie word choices, as the legacy model's did: its concurrent outputs
  matched solo ones on 1 of 4 and 4 of 8 prompts.
- **Coverage limits.** These were not run:
  - `mlx`, `ggml_metal` and `ggml_vulkan`;
  - multi-GPU `--tp` and `--layer-split`, which were covered on simulated devices only;
  - gpt-oss-120b.
- **Known costs.**
  - `--tp 4` / `--tp 8` are refused for gpt-oss-20b, where the legacy model ran them as expert parallelism.
    `--layer-split N` serves any N.
  - Direct-CUDA decode steps past the sliding window run as plans rather than CUDA-graph replays, because a recording
    would replay its first window's attention floor. Output is exact, but each step pays its launches on the host
    (118-120 tok/s against ggml_cuda's 155-162).
  - Direct-CUDA prompt attention on the sink layers runs the generic products (1750 tok/s against ggml_cuda's 6400).
  - Direct-CUDA offloaded prompts reach 551 tok/s against ggml_cuda's 2460 (§10).
  - On the synthetic file, chunked prefill on direct CUDA sits up to 3.0e-2 from the single pass: chunks of decode-size
    rows run the decode kernels. The legacy model kept it within 7.5e-3. Against ggml_cpu the runtime stays inside the
    bounds the legacy model was held to.

## 12. Output parser and chat template

- **Harmony format** is non-optional. The parser (`HarmonyOutputParser`) is
  marked `AlwaysRequired = true` because the model always wraps its output
  in channel tags:
  - `<|channel>analysis ...` for chain-of-thought reasoning.
  - `<|channel>final ...` for the user-visible answer.
- The output parser strips the `<|channel>analysis ...` block (or surfaces
  it as `<think>` content for the API) and exposes the `<|channel>final`
  payload as the assistant message.
- **Reasoning cannot be switched off, only shortened.** The model always
  opens the analysis channel before it answers; the one lever over how long
  it reasons is the `Reasoning: low|medium|high` line of the Harmony system
  message. The OpenAI `reasoning_effort` request field (`low`, `medium`,
  `high`; anything else is HTTP 400) sets that line, default `medium`; the
  Responses API spelling is `reasoning: {"effort": ...}`. A request that
  explicitly sends `"think": false` and names no effort is rendered at `low`
  (an absent `think` keeps `medium`) — previously the line was hard-coded to
  `medium`, so a 256-token request with thinking off spent its whole budget
  in analysis and ended with `finish_reason=length` and no content. The
  level is part of the shared-prefix checkpoint key, and the Web UI applies
  the same `think:false → low` mapping so the startup warm-up prepares both
  prefixes.
- **`response_format` works with `think: true`.** The final channel opens
  with the same `final<|message|>` header in both modes, so the JSON grammar
  arms there either way (`ThinkingGrammarActivationTrigger`) and the model is
  free to reason before the constrained answer.
- **Tool calling is supported** via the Harmony `commentary` channel. When
  the request includes `tools`:
  - The system message gains the line *"Calls to these tools must go to the
    commentary channel: 'functions'."* and the developer message gains a
    `# Tools` block declaring each tool as a TypeScript namespace
    (`namespace functions { type NAME = (_: { ... }) => any; }`).
  - The model emits a call as
    `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>`.
    `HarmonyOutputParser` parses the channel + `to=functions.NAME` recipient
    and decodes the JSON arguments into a `ToolCall` (max one call per turn,
    matching the reference implementations).
  - Tool results are fed back as
    `<|start|>functions.NAME to=assistant<|channel|>commentary<|message|>{result}<|end|>`.
  - **Stop tokens:** a tool call terminates with `<|call|>` (id 200012),
    which the gpt-oss GGUF does *not* list as an eos (only `<|return|>`,
    id 200002, is). `ModelBase` adds `<|call|>` to the stop set for the
    `gptoss`/`gpt-oss` architectures so generation halts after the call and
    it can be parsed.
  - Round-trip validated against gpt-oss-20b in
    [`HarmonyToolCallIntegrationTests`](../../InferenceWeb.Tests/HarmonyToolCallIntegrationTests.cs);
    format/parse coverage in
    [`HarmonyToolCallTests`](../../InferenceWeb.Tests/HarmonyToolCallTests.cs).
- Chat template uses the GPT-4o pre-tokenizer. Rendering is handled by the
  hardcoded `ChatTemplate.RenderHarmony` (the GGUF's embedded Harmony Jinja
  relies on recursive macros / `namespace()` / `strftime_now` that the
  lightweight Jinja2 engine does not fully support, especially on the tool
  path), so gpt-oss is routed to the hardcoded renderer like `mistral3` and
  `nemotron_h`.

## 13. Optimization opportunities

- **Direct-CUDA prompt attention with sinks.** A sink layer's prompt attention runs on the generic products. A flash
  kernel with the sink in its softmax would close most of the gap to ggml_cuda's prefill. The CUDA kernels
  `TryGqaPrefillAttentionWithSinks` / `TryGqaDecodeAttentionWithSinks` exist but are not wired to the executor's rings.
- **Recorded decode past the window on direct CUDA.** Recording the ring's floor as an input would let those steps replay.
- **Direct-CUDA offloaded prompts.** gpt-oss-20b's 2880-wide experts take the grouped MoE engine's warp-per-row route
  (2880 is neither a multiple of 128 nor of 256). A tensor-core route for that width would serve the offloaded and
  resident prompt experts alike.
- **`--tp 4` / `--tp 8`** need the down projection's rows split inside an MXFP4 row's 90 blocks, or an expert-parallel
  placement.
