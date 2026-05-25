# Qwen 3

[← back to model index](README.md)

| Property | Value |
|---|---|
| Provider | Alibaba |
| GGUF architecture key | `qwen3` |
| Source class | [`Qwen3Model`](../../TensorSharp.Models/Models/Qwen3/Qwen3Model.cs) (legacy per-seq) + [`Qwen3Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Qwen3/Qwen3Model.BatchedForward.cs) (`IBatchedPagedModel`) |
| Example models | Qwen3-4B, Qwen3-8B, Qwen3-14B, Qwen3-32B |
| Modalities | Text only |
| Thinking mode | Yes (`<think> ... </think>`) |
| Tool calling | Yes (`<tool_call>{...}</tool_call>`) |
| Batched / paged forward | Reference port — first model in TensorSharp to implement `IBatchedPagedModel.ForwardBatch`. Template for Mistral 3 / Gemma 4 / Qwen 3.5 / Nemotron-H batched ports. See §11. |
| Output parser | `Qwen3OutputParser` |

## 1. Origin and intent

Qwen 3 is the dense-only baseline transformer in the Alibaba Qwen line and the
simplest fully featured architecture in TensorSharp. It serves two purposes:

1. **Reference implementation** — a clean GQA-with-RoPE-and-RMSNorm
   transformer that exercises every facet of `ModelBase` without needing any
   exotic per-layer machinery.
2. **Throughput baseline** — Qwen 3 hosts the fastest "whole-model" decode
   path in the codebase: a single native call that runs every transformer
   layer in C++ with pre-resolved weight pointers cached at load time.

Qwen 3 supports thinking mode (`<think>...</think>`) and tool calling via
`<tool_call>` JSON blocks, both parsed by `Qwen3OutputParser`.

## 2. Model architecture

```
                    tokens (int[])
                          │
                    token_embd.weight
                          │
        ┌──── × NumLayers ──────────────────────────────┐
        │  RMSNorm(attn_norm)                            │
        │  attn_qkv (fused) ──► Q, K, V (GQA)            │
        │  per-head RMSNorm(Q) using attn_q_norm.weight  │
        │  per-head RMSNorm(K) using attn_k_norm.weight  │
        │  RoPE_NeoX(Q, K)                                │
        │  Q ← Q * (1/sqrt(headDim))                      │
        │  attention(Q, KCache, VCache)                   │
        │  attn_output ──► residual                       │
        │  RMSNorm(ffn_norm)                              │
        │  ffn_gate_up (fused) ──► gate ‖ up              │
        │  SwiGLU: SiLU(gate) * up                        │
        │  ffn_down ──► residual                          │
        └─────────────────────────────────────────────────┘
                          │
                    RMSNorm(output_norm)
                          │
                    LM head (output.weight)
                          │
                          ▼
                       logits
```

Two RMSNorms per block (`attn_norm`, `ffn_norm`) — no extra post-norms — keeps
the per-layer compute very lean.

## 3. Forward graph

For each layer L:

```
hidden ─► RMSNorm(attn_norm.weight, eps)
       ─► attn_qkv.weight matmul → split into Q [seq, qDim],
                                          K [seq, kvDim],
                                          V [seq, kvDim]
       ─► per-head RMSNorm(Q, attn_q_norm.weight)
       ─► per-head RMSNorm(K, attn_k_norm.weight)
       ─► RoPE_NeoX(Q, K, ropeFreqs[]) at positions startPos..startPos+seqLen-1
       ─► Q ← Q * (1/sqrt(headDim))
       ─► append (K, V) to KV cache
       ─► attention(Q, KCache, VCache)        // full GQA, no window
       ─► attn_output.weight matmul → o
       ─► hidden = hidden + o
       ─► h2 = RMSNorm(ffn_norm.weight, eps) on hidden
       ─► ffn_gate_up.weight matmul → [gate ‖ up]
       ─► g = SiLU(gate)
       ─► h3 = ffn_down.weight × (g * up)
       ─► hidden = hidden + h3
```

After all layers:

```
hidden ─► narrow(seq_len-1) if prefill
       ─► RMSNorm(output_norm.weight, eps)
       ─► matmul against output.weight       # not tied for Qwen 3
       ─► copy to float[VocabSize]
```

## 4. Components in detail

### 4.1 Attention

- **GQA** with `Config.NumKVHeads < Config.NumHeads`. The fused
  `attn_qkv.weight` has row layout `[qDim ‖ kvDim ‖ kvDim]`, where
  `qDim = NumHeads * headDim` and `kvDim = NumKVHeads * headDim`.
- **QK-norm**: per-head RMSNorm with weights `attn_q_norm.weight` (size
  `headDim`) and `attn_k_norm.weight` (size `headDim`). RMSNorm is applied
  to each Q/K head independently.
- **RoPE**: NeoX-style. Frequencies are precomputed once in
  `_ropeFreqs[halfDim]` using `Config.RopeBase` and `Config.RopeScale`.
  - **Decode** uses an inlined C# loop with stackalloc cos/sin tables.
  - **Prefill** uses `Ops.RoPEEx` which dispatches to the registered backend
    (managed SIMD on CPU, native kernels on GGML / CUDA).
- **No attention sinks, no SWA, no Yarn / partial RoPE.** Standard causal
  attention only.

### 4.2 FFN — SwiGLU

`ffn_gate_up.weight` packs `[gate ‖ up]` along the row dimension. After the
matmul, `Ops.SiLUMul` computes `SiLU(gate) * up` in one call. `ffn_down`
matmul follows. No biases anywhere.

### 4.3 Normalization

Two RMSNorms per block (`attn_norm`, `ffn_norm`) plus the per-head QK norms
and the final `output_norm`. Epsilon comes from `Config.Eps` (typically
`1e-6`).

### 4.4 Embedding and LM head

`token_embd.weight` is the input embedding; `output.weight` is the LM head and
is **not** tied (Qwen 3 ships both).

## 5. Parameters and settings

Qwen 3 uses only the standard GGUF metadata — no architecture-specific keys
beyond the namespaced ones:

| Key | Type | Meaning |
|---|---|---|
| `qwen3.attention.head_count_kv` | uint32 | KV head count for GQA |
| `qwen3.rope.freq_base` | float32 | RoPE base (default ~ 1e6 for Qwen 3 long-context variants) |
| `qwen3.rope.scale` | float32 | RoPE frequency scale divisor |

Plus the standard `general.*` keys handled by `ParseBaseConfig()`.

## 6. Weight naming convention

```
token_embd.weight                          # [vocab, hidden]
blk.{L}.attn_norm.weight                   # pre-attention RMSNorm
blk.{L}.attn_qkv.weight                    # fused QKV [qDim+kvDim+kvDim, hidden]
blk.{L}.attn_q_norm.weight                 # per-head Q RMSNorm
blk.{L}.attn_k_norm.weight                 # per-head K RMSNorm
blk.{L}.attn_output.weight                 # output projection
blk.{L}.ffn_norm.weight                    # pre-FFN RMSNorm
blk.{L}.ffn_gate_up.weight                 # fused gate+up [2*intermediate, hidden]
blk.{L}.ffn_down.weight                    # FFN down projection
output_norm.weight                         # final RMSNorm
output.weight                              # LM head
```

`FuseQKVWeights()` and `FuseGateUpWeights()` are called at load time, so even
GGUFs that ship Q / K / V or gate / up separately collapse into the fused
forms shown above.

## 7. TensorSharp implementation walkthrough

Constructor (`Qwen3Model(string ggufPath, BackendType backend)`):

1. `ParseBaseConfig()` reads `general.*` and `qwen3.*` core fields.
2. `Config.NumKVHeads = (int)_gguf.GetUint32("qwen3.attention.head_count_kv")`.
3. `ParseTokenizer()` builds the BPE tokenizer.
4. `LoadWeights()`.
5. `FuseQKVWeights()`, `FuseGateUpWeights()`.
6. `PrepareCudaQuantizedWeightsForInference()`.
7. `InitKVCache(maxSeqLen)` allocates per-layer K and V tensors of shape
   `[NumKVHeads, maxSeqLen, headDim]`.
8. `PrecomputeConstants()` populates `_ropeFreqs[]` and pre-allocates per-
   head decode position arrays (`_decodeQPositions`, `_decodeKPositions`).
9. `BuildModelDecodeArrays()` packs per-layer pointers, types, and dimensions
   for the **whole-model** native decode call.
10. `DetermineNativeLayerDecodeAvailability()` chooses between the per-layer
    and whole-model native paths based on backend support.

`Forward(int[] tokens)` then chooses one of three execution paths:

- **`useNativeModelDecode`** (default for `seqLen == 1` on a GGML backend
  with the model decode arrays available): one native call processes every
  layer.
- **`useNativeDecode`** (per-layer native fallback): each layer runs through
  `NativeTransformerLayerDecode`.
- **Managed C# loop**: full per-op managed path; used for prefill (`seqLen >
  1`), the pure CPU backend, and the direct CUDA backend.

After the decode loop, the LM head runs (using a `Narrow(seq_len-1)` for
prefill so the matmul only produces a `[1, vocab]` row), and the logits are
copied into `_logitsBuffer`.

## 8. Prefill optimization

Qwen 3's prefill path uses the standard managed loop with backend-dispatched
ops:

- Fused QKV and fused gate / up cut the per-layer GEMM count from 5 to 3
  (`qkv`, `out`, `gate_up`, `down`).
- `Ops.RoPEEx` runs as a single native kernel on GGML / CUDA backends.
- `Ops.SiLUMul` fuses the gate activation and elementwise multiply.
- Last-row narrow is applied before `output_norm` so the LM head matmul only
  produces one row of logits.

There is no chunked prefill or fused per-layer prefill kernel — the model is
small enough that the per-op managed path on a GGML / CUDA backend already
hits good prefill throughput. Adding a fused per-layer prefill kernel would
be the obvious next step (see § 12).

## 9. Decode optimization

### Whole-model native decode (`TransformerModelDecode`)

This is the fastest decode path in TensorSharp today. At load time,
`BuildModelDecodeArrays()` packs into `_modelDecodeArrays`:

- A flat `IntPtr[]` of all per-layer weight pointers (norms, QKV, Q/K norms,
  output, FFN norm, gate+up, down).
- Each layer's quantized weight metadata (`type`, `ne0`, `ne1`, raw byte
  count).
- KV cache pointers that the native side writes into directly.
- The RoPE frequency table.

`NativeTransformerModelDecode()` is then a single P/Invoke that runs the
entire forward pass for one token in C++:

- No managed loop overhead between layers.
- No dictionary lookups for weight names.
- No tensor allocations between layers; the native side reuses scratch
  buffers held inside `_modelDecodeArrays`.

The host KV cache copy is marked dirty (`_kvCacheHostDirty = true`); the next
prefill or KV-cache truncation calls `EnsureKvCacheHostSynchronized()` to
copy the device data back so the managed path sees the right state.

### Per-layer native decode (`TransformerLayerDecode`)

Used as a fallback when the whole-model arrays are not available (rare; can
happen if a quant type is not yet supported by the native loop). Each layer
runs through one P/Invoke instead of one P/Invoke for the whole stack — still
much cheaper than a per-op call sequence.

### Hand-tuned C# decode

When neither native path is available (pure CPU backend, direct CUDA backend
without the matching native build), the managed decode loop:

- Caches per-layer weight name strings once in `_layerWeightNames[]`.
- Uses `stackalloc` cos/sin tables for the inline RoPE.
- Uses pre-allocated `_decodeQPositions[]` / `_decodeKPositions[]` arrays.

## 10. Memory and KV cache strategy

- Per-layer K and V tensors of shape `[NumKVHeads, maxSeqLen, headDim]`. KV
  dtype configurable as `f32`, `f16`, or `q8_0` via `--kv-cache-dtype`.
- `ResetKVCache()` zeroes everything and clears the dirty flag.
- `TruncateKVCache(int tokenCount)` is supported (calls
  `EnsureKvCacheHostSynchronized()` first), which is what the server uses for
  multi-turn KV-cache reuse.
- Quantized weights stay quantized in `_quantWeights`; matmul calls run
  through the backend's native quantized matmul.

## 11. Batched / paged forward (continuous batching)

`Qwen3Model.BatchedForward.cs` is the **reference implementation** of
`IBatchedPagedModel.ForwardBatch` in TensorSharp — the simplest hybrid-
free dense transformer port and the template Mistral 3, Gemma 4,
Qwen 3.5/3.6, and Nemotron-H later built on top of. It runs through the
shared `InferenceEngine` continuous-batching stack documented in
[`docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md).

Implementation:

- **Lazy per-layer paged K/V buffers** of layout
  `[numBlocks * blockSize * numKvHeads * headDim]`, sized by the
  engine's `BlockPool`. `AllocateLayerBuffer(numBlocks, blockSize,
  numKvHeads, headDim)` is the helper used by every later batched
  port.
- **Per-token RoPE** built from `ctx.Positions` and dispatched through
  `Ops.RoPEEx` (which already accepts an arbitrary positions tensor —
  this is the kernel change Qwen 3's batched port introduced and the
  other batched models reuse).
- **K/V scatter via `PagedKvBatchOps.ScatterKv(slotMapping)`** writes
  the layer's K and V at `blockId * blockSize + offset` in the layer's
  paged buffer.
- **Per-sequence attention** via the native paged kernel
  `TSGgml_PagedAttentionForward` (default) or the C# fallback
  `ManagedPagedAttention.Forward`. K/V are gathered per sequence from
  `blockTables` + `seqLens`.
- **Last-token gather** via
  `PagedKvBatchOps.GatherLastTokenPerSeq` so only the LM head runs on
  `numSeqs` rows instead of `numTokens`.
- **Reuses existing helpers** — `Embedding`, `RMSNormOp`,
  `LinearForward`, and `FFN` on `ModelBase` already operate over the
  full token axis, so embeddings / projections / norms / FFN batch
  automatically.

### Verified correctness

End-to-end logit-level correctness against a real base-Qwen3 GGUF is not
yet verified in this tree (only Qwen 3.6 / GatedDeltaNet GGUFs are
available locally). The opt-in
[`Qwen3BatchedForwardTests`](../../InferenceWeb.Tests/Qwen3BatchedForwardTests.cs)
self-validate as soon as a base Qwen3 GGUF is dropped into
`TS_TEST_MODEL_DIR`. `EngineParallelInferenceTests` exercise the engine
side of the path against the available Qwen GGUFs.

## 12. Output parser and chat template

- `Qwen3OutputParser` extracts:
  - Thinking content from `<think> ... </think>` blocks.
  - Tool calls from `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` blocks.
- Chat template uses the standard Qwen `<|im_start|>` / `<|im_end|>` rolling
  message format. Falls back to the hardcoded template when the GGUF lacks a
  Jinja2 template; otherwise the embedded Jinja2 is rendered directly.

## 13. Optimization opportunities

- **GPU-fused per-token decode** — the whole-model native path already
  eliminates managed overhead, but layers still run as separate GGML graphs
  internally. Fusing them into a single graph (à la `Gemma4ModelDecode`) would
  cut the remaining round-trip cost on Metal / CUDA.
- **Native prefill** — moving prefill into a native single-call path would
  significantly improve TTFT, especially on long prompts where the managed
  per-op overhead is not amortized away.
- **Validate batched correctness on a real base-Qwen3 GGUF** — the
  reference test exists but needs a model file checked into the test
  fixture (or a CI helper that downloads one) to enforce the cosine
  bound automatically.
