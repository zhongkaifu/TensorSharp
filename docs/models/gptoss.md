# GPT OSS

[← back to model index](README.md)

| Property | Value |
|---|---|
| Provider | OpenAI |
| GGUF architecture keys | `gptoss`, `gpt-oss` |
| Source class | [`GptOssModel`](../../TensorSharp.Models/Models/GptOss/GptOssModel.cs) (legacy per-seq) + [`GptOssModel.BatchedForward.cs`](../../TensorSharp.Models/Models/GptOss/GptOssModel.BatchedForward.cs) (`IBatchedPagedModel`) |
| Example models | gpt-oss-20b |
| Modalities | Text only |
| Thinking mode | Yes (Harmony format: `<\|channel>analysis ... <\|channel>final`) |
| Tool calling | No |
| Batched / paged forward | **Default ON** — set `TS_GPTOSS_BATCHED=0` to force the legacy per-sequence KV-swap path for A/B comparison. Per-layer paged K/V plus attention sinks via native `TSGgml_PagedAttentionForwardWithSinks` (or managed C# fallback via `TS_GPTOSS_PAGED_ATTN_MANAGED=1`). See §11. |
| Output parser | `HarmonyOutputParser` (always required) |

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
        │  per-expert: SiLUAlphaLimit(gate+bias, up+bias)  │
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
            scores ← apply causal mask (and SWA on even layers when desired)
            sinks  = attn_sinks.weight                   # [numHeads]
            sums   = exp(scores - max) + exp(sinks - max)
            scores = exp(scores - max) / sums            # softmax including sink
            out    = scores × V
       ─► attn_output.weight matmul + bias → o
       ─► residual = hidden + o

       moeInput = (last layer in prefill) ? narrow(seq_len-1) : residual
       moeInput ─► RMSNorm(post_attention_norm)
                ─► router_linear + bias
                ─► topK(_numExpertsUsed)
                ─► softmax(selected weights)
                ─► For each selected expert e:
                       gate = ffn_gate_up_exps[e][:, :nFf] × moeInput + bias
                       up   = ffn_gate_up_exps[e][:, nFf:] × moeInput + bias
                       gate = clamp(gate, -∞, 7)
                       up   = clamp(up,   -7, 7)
                       out_e = gate · sigmoid(α · gate) · (up + 1)
                       out_e = ffn_down_exps[e] × out_e + bias
                ─► weighted Σ out_e using router weights
                ─► residual += weighted_sum
```

After all layers:

```
hidden ─► narrow(seq_len-1) if prefill          # GPT OSS narrows BEFORE MoE
                                                # in the last layer, see § 8
       ─► RMSNorm(output_norm.weight, eps)
       ─► output.weight matmul → logits
       ─► copy to float[VocabSize]
```

## 4. Components in detail

### 4.1 Attention

- **GQA** with `Config.NumKVHeads < Config.NumHeads`.
- **Optional fused QKV** (`_isQkvFused`) — if the GGUF ships a single
  `attn_qkv.weight`, `FuseQKVWeights()` builds a fused tensor; otherwise the
  three linear ops are dispatched separately.
- **Bias on every projection**: implemented in `LinearForwardWithBias()`,
  which performs the matmul and then adds the bias row-by-row. The bias is
  looked up from `_weights[biasName]`.
- **No QK-norm**: unlike Gemma / Qwen, GPT OSS skips the per-head Q and K
  normalization.
- **Attention sinks**: per-head bias `attn_sinks.weight` (shape
  `[numHeads]`). Treated as a virtual token in the softmax: its value
  participates in both max-finding and exp-sum, effectively adding an
  always-available attention target. Implemented in
  `ApplySoftmaxWithSinks()` and `AttentionDecodeWithSinks()`.
- **RoPE**: NeoX-style with YaRN scaling. `Ops.RoPEEx` is called with
  `origCtxLen = Config.OriginalContextLength` (4096), `freqScale = 1 /
  RopeScale`, `beta_fast = 32`, `beta_slow = 1`.
- **Attention pattern**: even layers ⇒ SWA (window from `_slidingWindow`,
  default 128); odd layers ⇒ full causal. *(Implementation note: in the
  current code path all layers attend to `totalSeqLen` — the SWA bound is
  read from GGUF metadata but not yet used to bound the actual softmax
  width. This is on the optimization-opportunities list.)*

### 4.2 FFN — clamped GLU (`SiLUAlphaLimit`)

```
gate = clamp(gate, -∞, SiluLimit)        # SiluLimit = 7.0
up   = clamp(up,   -SiluLimit, SiluLimit)
out  = gate · sigmoid(SiluAlpha · gate) · (up + 1)   # SiluAlpha = 1.702
```

The constants `SiluAlpha` and `SiluLimit` are hardcoded. The activation is
implemented with SIMD vectorization in `SiLUAlphaLimitInPlace` and is also
exposed to the fused MoE prefill kernel as `swiglu_oai`.

### 4.3 MoE routing

- Every layer has `_numExperts` experts (32 for gpt-oss-20b) with TopK
  routing (`_numExpertsUsed = 4`).
- Routing is `linear(hidden) + bias → TopK → softmax(selected only)`.
  This is **TopK-then-softmax**, the inverse of Gemma 4's softmax-then-TopK
  flow.
- Expert weights are stored as fused `gate ‖ up` rows in
  `ffn_gate_up_exps.{E}.weight`, with a matching fused
  `ffn_gate_up_exps.{E}.bias`. `FuseExpertGateUpWeights()` performs this
  fusion at load time.
- Expert biases are stored in the GGUF as packed `[numExperts, biasDim]`
  tensors (`ffn_gate_exps.bias`). `SplitExpertBiases()` unpacks these into
  per-expert `[biasDim]` tensors before the fusion step.

### 4.4 Stacked MoE prefill kernel (`TryMoEPrefillFused`)

For prefill on a GGML backend, GPT OSS dispatches one
`ggml_mul_mat_id` + `ggml_add_id` + `swiglu_oai` graph per layer instead of
looping over active experts per token. The kernel reads:

- The original 3D `ffn_gate_exps.weight`, `ffn_up_exps.weight`,
  `ffn_down_exps.weight` blocks (zero-cost views into the mmap'd model).
- A contiguous `[2 * nFf, numExperts]` f32 stacked bias for gate / up
  (built once by `InitMoeStackedWeights` from the per-expert biases captured
  before fusion).
- A contiguous `[hidden, numExperts]` f32 stacked bias for down (also built
  once at init time).

Per-token routing is computed in C# (TopK + softmax) and the resulting
`(token, expert)` map is fed to the kernel.

### 4.5 Tokenizer

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
blk.{L}.attn_qkv.weight  / attn_qkv.bias       # post-fusion (when QKV is fused)
blk.{L}.attn_output.weight / attn_output.bias
blk.{L}.attn_sinks.weight                      # per-head bias [numHeads]
blk.{L}.post_attention_norm.weight

blk.{L}.ffn_gate_inp.weight / ffn_gate_inp.bias   # router [numExperts, hidden]
blk.{L}.ffn_gate_up_exps.{E}.weight                # fused expert gate+up
blk.{L}.ffn_gate_up_exps.{E}.bias                  # fused expert gate+up bias
blk.{L}.ffn_down_exps.{E}.weight                   # expert down
blk.{L}.ffn_down_exps.{E}.bias                     # expert down bias
```

## 7. TensorSharp implementation walkthrough

Constructor (`GptOssModel(string ggufPath, BackendType backend)`):

1. `ParseBaseConfig()`.
2. Reads MoE counts, the SWA window, expert FFN length, and the YaRN
   original context length.
3. `ParseTokenizer()`.
4. `LoadWeights()`.
5. `SplitExpertBiases()` — unpacks the GGUF's packed expert bias tensors.
6. **Snapshot** the gate / up biases per expert *before* fusing. The fused
   MoE prefill kernel needs the biases in their original split shape to
   build a contiguous stacked bias table.
7. `FuseExpertGateUpWeights()` — concatenates each expert's gate and up
   into `ffn_gate_up_exps.{E}.weight` and the matching fused bias.
8. `FuseQKVWeights()`.
9. `PrepareCudaQuantizedWeightsForInference()`.
10. `InitKVCache(maxSeqLen)`.
11. `PrecomputeConstants()` — pre-allocates per-layer weight name arrays and
    per-expert name arrays so the hot loops do no string interpolation.
12. `InitMoeStackedWeights(preFuseGateBias, preFuseUpBias)` — builds the
    stacked biases used by the fused MoE prefill kernel. Also resolves the
    per-layer `StackedExpertWeights` views into the original 3D
    `_exps.weight` blocks.

`Forward(int[] tokens)` then runs the per-op managed loop:

- Embedding lookup.
- For each layer: `LinearForwardWithBias` for Q / K / V, RoPE,
  `AttentionDecodeWithSinks` (or `ApplySoftmaxWithSinks` for prefill), output
  projection + bias, MoE block (or fused MoE prefill kernel).
- For the **last** transformer layer in prefill, the residual is narrowed to
  the last token *before* MoE (an early-exit optimization that skips the MoE
  computation on positions that do not feed the LM head).
- Final RMSNorm, LM head, copy to `_logitsBuffer`.

## 8. Prefill optimization

- **Last-layer narrow before MoE**. Inside the last `TransformerBlock`,
  `moeInput` is narrowed to `[1, hidden]` so the MoE block only computes the
  one row that matters for the LM head. All earlier layers compute the MoE
  branch over the full sequence.
- **Stacked MoE prefill kernel** (`TryMoEPrefillFused`). One
  `mul_mat_id + add_id + swiglu_oai` graph dispatch per layer for the
  routed-expert FFN, replacing per-token per-expert dispatches. See § 4.4.
- **Pre-built per-layer name arrays** (`_layerNames[L][]`) and per-expert
  name arrays (`_expertNames[L][E][]`) eliminate string interpolation in the
  hot loops.
- **Cached attention sinks** in `_layerSinks[L][numHeads]` so the softmax
  doesn't need to re-fetch the sinks every step.

## 9. Decode optimization

- **Per-step routing buffers** (`_moeExpertCounts`, `_moeExpertOffsets`,
  `_moeTokenMap`, `_moeWeightMap`) are reused across tokens.
- **SIMD-vectorized bias addition and SiLUAlphaLimit activation** in
  `LinearForwardWithBias` and `SiLUAlphaLimitInPlace`.
- **Attention sinks softmax** runs on CPU (scalar with optional SIMD on the
  exp-sum). The fused GPU kernel for sinks is on the optimization
  opportunities list — currently the softmax with sinks is the slow path on
  Metal / CUDA.
- **MXFP4 expert weights** stay quantized in `_quantWeights`; matmul is
  dispatched through the backend's quantized matmul.

There is **no whole-model native decode path** for GPT OSS (unlike Qwen 3),
so decode dispatches are still per-op. This is the largest open optimization
target.

## 10. Memory and KV cache strategy

- Per-layer K and V tensors of shape `[NumKVHeads, maxSeqLen, headDim]`. KV
  dtype is configurable as `f32`, `f16`, or `q8_0`.
- `ResetKVCache()` zeroes everything.
- The expert FFN weights live in the original 3D
  `ffn_gate_exps.weight` / `ffn_up_exps.weight` / `ffn_down_exps.weight`
  blocks loaded by `ModelBase`. `FuseExpertGateUpWeights()` only disposes the
  per-expert *views*, not the underlying bulk buffer. This means the fused
  MoE prefill kernel can still address the original blocks directly via
  `_layerStackedGate` / `_layerStackedUp` / `_layerStackedDown`.

## 11. Batched / paged forward (continuous batching)

GPT OSS implements `IBatchedPagedModel.ForwardBatch`
([`GptOssModel.BatchedForward.cs`](../../TensorSharp.Models/Models/GptOss/GptOssModel.BatchedForward.cs))
and runs it by default — concurrent requests can only be served truly
in parallel through the batched path (the per-sequence fallback forwards
at most one sequence per step). Set `TS_GPTOSS_BATCHED=0` to force the
legacy per-sequence KV-swap fallback for A/B comparison. The batched port has to
preserve GPT OSS's three architecture-distinguishing features —
**attention sinks**, **bias on every projection**, and **per-layer
alternating SWA** — inside the paged scheduling stack:

- **Per-layer paged K/V** of layout
  `[numBlocks * blockSize * numKvHeads * headDim]`, lazily grown
  copy-on-write.
- **Batched QKV with bias** per token; NeoX + YaRN RoPE is dispatched
  with an explicit `positions[]` array.
- **Per-layer SWA window** — the same alternating local / global window
  pattern as the legacy path is passed per-call into the paged kernel,
  so the batched scheduler still sees one uniform model while each
  layer respects its own attention horizon.
- **Native paged attention with per-head sinks**:
  `TSGgml_PagedAttentionForwardWithSinks`
  ([`ggml_ops_paged_attention.cpp`](../../TensorSharp.GGML.Native/ggml_ops_paged_attention.cpp))
  combines `ggml_flash_attn_ext` with the `add_sinks` variant so the
  sink logits participate in softmax normalization without contributing
  to V — the same numerical behaviour as the legacy CPU sinks softmax.
  Exposed through `GgmlBasicOps.PagedAttentionForwardWithSinks`.
  - The managed C# fallback,
    [`ManagedPagedAttention.ForwardWithSinks`](../../TensorSharp.Runtime/Paged/ManagedPagedAttention.cs),
    is selected when running on non-GGML backends or when
    `TS_GPTOSS_PAGED_ATTN_MANAGED=1` forces the C# path. Both produce
    bit-identical greedy output.
- **MoE FFN** runs through the existing `MoEForward(numTokens)`
  token-parallel path; no GPT-OSS-specific batched MoE kernel.

### Verified correctness and throughput

- **100% greedy match** vs legacy (12/12 tokens) in
  [`GptOssBatchedCorrectnessTests`](../../InferenceWeb.Tests/GptOssBatchedCorrectnessTests.cs)
  — preserved across both the managed sinks fallback and the native
  sinks kernel.
- **Throughput remains flat** (~0.93–1.05× across n=1, 3, 5 in
  [`GptOssBatchedPerfBench`](../../InferenceWeb.Tests/GptOssBatchedPerfBench.cs)).
  The native sinks kernel didn't move the needle because the legacy
  per-seq path already runs one fused per-layer kernel
  (`TryFusedAttnLayerPrefill`: RMSNorm + fused QKV + RoPE + KV-append
  + sinks-aware softmax + attn + output proj + residual in one
  cgraph), whereas the batched path issues ~5 separate graphs per
  layer (norm, QKV, RoPE, paged-attn, output, residual). Closing this
  gap needs a fused-per-layer batched kernel for GPT OSS — substantial
  follow-up work.

## 12. Output parser and chat template

- **Harmony format** is non-optional. The parser (`HarmonyOutputParser`) is
  marked `AlwaysRequired = true` because the model always wraps its output
  in channel tags:
  - `<|channel>analysis ...` for chain-of-thought reasoning.
  - `<|channel>final ...` for the user-visible answer.
- The output parser strips the `<|channel>analysis ...` block (or surfaces
  it as `<think>` content for the API) and exposes the `<|channel>final`
  payload as the assistant message.
- Tool calling is **not supported** — the Harmony format reserves a tool
  channel but the open-weights gpt-oss-20b release does not invoke it
  reliably; the API accepts `tools` but does not surface tool calls.
- Chat template uses the GPT-4o pre-tokenizer and a Harmony-aware Jinja2
  template loaded from the GGUF.

## 13. Optimization opportunities

- **Fused per-layer batched kernel** — the legacy path's
  `TryFusedAttnLayerPrefill` does one cgraph per layer (norm + fused
  QKV + RoPE + KV-append + sinks-aware softmax + attn + output proj +
  residual). The batched path currently issues ~5 graphs per layer.
  Folding them into one batched cgraph is the largest remaining win
  for batched GPT OSS perf.
- **Fused QKV in every release** — when the GGUF ships split Q / K / V the
  per-projection bias add still happens after each separate matmul. A
  `FusedQKVWithBias` graph would cut 3 dispatches into 1.
- **Native whole-model decode** (legacy) — like Qwen 3's
  `TransformerModelDecode`, with attention sinks and bias support, would
  remove most managed overhead on the per-seq path.
- **GPU-fused sinks softmax (legacy)** — the legacy CPU sinks softmax is
  the per-seq counterpart to the batched native `*_WithSinks` kernel. A
  custom Metal / CUDA fused kernel for the legacy path would close that
  gap and let the per-seq path keep up at single-sequence workloads.
- **Per-expert decode batching** — even with the stacked MoE prefill
  kernel, decode still runs experts sequentially per token. A batched
  decode path (analogous to Qwen 3.5's `MoEExpertsSwiGLUResidual`) would
  collapse `numExpertsUsed` dispatches into one.
- **SWA bound on even layers** — `_slidingWindow` is read from GGUF
  metadata but the current attention path does not yet bound the softmax
  width to it. Wiring it through `AttentionDecodeWithSinks` would lower
  per-token compute on long contexts.
