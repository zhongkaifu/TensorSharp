# Mistral 3

[← back to model index](README.md)

| Property | Value |
|---|---|
| Provider | Mistral AI |
| GGUF architecture key | `mistral3` |
| Source class | [`Mistral3Model`](../../TensorSharp.Models/Models/Mistral3/Mistral3Model.cs) (legacy per-seq) + [`Mistral3Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Mistral3/Mistral3Model.BatchedForward.cs) (`IBatchedPagedModel`) |
| Vision encoder | [`Mistral3VisionEncoder`](../../TensorSharp.Models/Models/Mistral3/Mistral3VisionEncoder.cs) (Pixtral) |
| Image processor | [`Mistral3ImageProcessor`](../../TensorSharp.Models/Models/Mistral3/Mistral3ImageProcessor.cs) |
| Example models | Mistral-Small-3.1-24B-Instruct, Ministral-3-14B-Instruct |
| Modalities | Text, image |
| Thinking mode | No |
| Tool calling | No |
| Batched / paged forward | **Default** — reference `IBatchedPagedModel.ForwardBatch`. Verified end-to-end on Ministral-3-14B; native paged-attention kernel ~21% faster than legacy on long context. See §11. |
| Output parser | `PassthroughOutputParser` |

## 1. Origin and intent

Mistral 3 is Mistral's third-generation LLaMA-style dense transformer with two
distinguishing extensions:

- **YaRN-corrected RoPE** — partial, per-frequency interpolation between the
  original RoPE schedule and an extrapolated one. This lets the model attend
  cleanly across context lengths well past the original training context.
- **Position-dependent Q scaling** — once the position exceeds the original
  context length, Q is multiplied by `1 + β · log(1 + ⌊pos / origCtx⌋)`,
  which keeps attention magnitudes well-conditioned at long ranges.
- **Pixtral vision encoder** for image input.

Otherwise the architecture is the standard LLaMA / Mistral recipe: GQA, no
QK-norm, SwiGLU FFN, RMSNorm, GPT-J style RoPE pairings, optional tied LM
head.

## 2. Model architecture

```
                    tokens (int[])
                          │
                    token_embd.weight
                          │
        [optional]  InjectVisionEmbeddings (image tokens)
                          │
        ┌──── × NumLayers ──────────────────────────────┐
        │  RMSNorm(attn_norm)                            │
        │  Q, K, V projections (or fused QKV)            │
        │  RoPE_GPT-J(Q, K) with YaRN-corrected freqs    │
        │  Q ← Q * (1 + β · log(1 + ⌊pos / origCtx⌋))    │
        │  attention(Q, KCache, VCache) (full causal)    │
        │  attn_output ─► residual                        │
        │  RMSNorm(ffn_norm)                              │
        │  ffn_gate_up ─► [gate ‖ up]                    │
        │  SwiGLU: SiLU(gate) * up                        │
        │  ffn_down ─► residual                           │
        └────────────────────────────────────────────────┘
                          │
                    RMSNorm(output_norm)
                          │
                    LM head (output.weight or tied)
                          │
                          ▼
                       logits
```

## 3. Forward graph

For each layer L:

```
hidden ─► RMSNorm(attn_norm.weight, eps)
       ─► QKV (fused via attn_qkv.weight when fused at load time, otherwise
                    three separate attn_q/k/v matmuls)
       ─► RoPE_GPT-J(Q, K, ropeFreqs[]) at positions startPos..startPos+seqLen-1
            • GPT-J style: pairs adjacent elements (x[2i], x[2i+1])
            • Frequencies are YaRN-corrected when ropeType == "yarn"
       ─► [if _ropeOrigCtx > 0]:
            scale = 1 + _ropeScalingBeta * log(1 + floor(pos / _ropeOrigCtx))
            Q ← Q * scale
       ─► Q ← Q * (1/sqrt(headDim))
       ─► append (K, V) to KV cache
       ─► attention(Q, KCache, VCache)        # standard causal GQA
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
       ─► matmul against output.weight (or token_embd.weight when tied)
       ─► copy to float[VocabSize]
```

## 4. Components in detail

### 4.1 Attention

- **GQA** with `Config.NumKVHeads < Config.NumHeads`. Separate
  `key_length` and `value_length` are supported (`_attnKeyLen`,
  `_attnValLen`); they default to `Config.HeadDim` when not set.
- **Optional fused QKV**. `FuseQKVWeights()` builds `attn_qkv.weight` when
  the GGUF stores Q / K / V separately. The decoder uses
  `_layerQkvFused[l]` to pick the fused matmul vs. the three separate
  matmuls per layer.
- **No QK-norm** — unlike Qwen 3, Gemma 3, and Gemma 4, Mistral 3 skips the
  per-head RMSNorm of Q and K.

### 4.2 RoPE — GPT-J style with YaRN

- **Pairing**: adjacent elements `(x[2i], x[2i+1])` are rotated together
  (GPT-J convention), as opposed to the `(x[i], x[i + halfDim])` pairing of
  NeoX.
- **YaRN frequency correction**: when `mistral3.rope.scaling.type == "yarn"`
  and `mistral3.rope.scaling.original_context_length > 0`,
  `ApplyYarnFreqCorrection()` interpolates between extrapolated and
  interpolated frequencies based on whether each frequency band is in the
  "slow" or "fast" rotation range:

  ```
  lowFreqWavelen  = origCtx / betaSlow
  highFreqWavelen = origCtx / betaFast
  for each freq f:
      wavelen = 2π / f
      if wavelen < highFreqWavelen:        # high-freq band: extrapolate
          f stays
      elif wavelen > lowFreqWavelen:       # low-freq band: interpolate
          f *= 1 / scale
      else:                                # medium band: smooth interp
          ramp = ...                       # linear ramp between the two
          f = mix(f, f / scale, ramp)
  ```

- **Position-dependent Q scaling**: when `_ropeOrigCtx > 0`,
  ```
  q *= 1 + _ropeScalingBeta * log(1 + floor(pos / _ropeOrigCtx))
  ```
  This keeps attention magnitudes well-conditioned past the original
  context length.

### 4.3 FFN — SwiGLU

`ffn_gate_up.weight` packs `[gate ‖ up]`. `Ops.SiLUMul` then computes
`SiLU(gate) * up`, followed by `ffn_down`. No biases.

### 4.4 Embedding and LM head

`token_embd.weight` is the input embedding. The LM head uses
`output.weight` when present; otherwise it's tied to `token_embd.weight`
(`_hasTiedOutput`).

### 4.5 Vision pipeline (Pixtral)

`Mistral3VisionEncoder` is the Pixtral architecture:

- **Conv2D patch embedding**: `v.patch_conv.weight` (shape
  `[hidden, channels, patchSize, patchSize]`) with optional bias.
- **RMSNorm** at the encoder input.
- **2D RoPE** positional embedding for spatial position.
- **SiLU-gated MLP transformer blocks**: LayerNorm + multi-head attention +
  residual + LayerNorm + SiLU-gated MLP + residual.
- **Spatial patch merging**: groups neighboring patches into one merged
  token at the projector boundary.
- **Multimodal projector**: RMSNorm → PatchMerger → Linear → GELU →
  Linear, mapping to the LM hidden dim.

`Mistral3ImageProcessor` resizes images preserving aspect ratio:

1. Composite RGBA over a white background.
2. Resize so the longest edge equals `longest_edge` while keeping aspect
   ratio.
3. Pad to a multiple of `patch_size`.
4. Normalize with the CLIP default mean / std.

The multimodal injector (`ProcessMistral3History` in
`ModelMultimodalInjector.cs`) walks the chat history, runs the encoder once
per unique image, and queues a `PreparedEmbeddingSpan` so the embeddings
land at the right positions in the prompt before `Forward()`.

## 5. Parameters and settings

| Key | Type | Meaning |
|---|---|---|
| `mistral3.rope.dimension_count` | uint32 | RoPE dimension count |
| `mistral3.rope.scaling.type` | string | RoPE scaling type (e.g. `yarn`) |
| `mistral3.attention.temperature_scale` | float32 | Position-dependent Q scaling β (default 0.1) |
| `mistral3.rope.scaling.original_context_length` | uint32 | YaRN original context length |
| `mistral3.rope.scaling.extrapolation_factor` | float32 | YaRN extrapolation factor (default 1.0) |
| `mistral3.rope.scaling.yarn_beta_fast` | float32 | YaRN fast rotation threshold (default 32.0) |
| `mistral3.rope.scaling.yarn_beta_slow` | float32 | YaRN slow rotation threshold (default 1.0) |
| `mistral3.rope.scaling.mscale` | float32 | YaRN mscale (default 0) |
| `mistral3.rope.scaling.mscale_all_dim` | float32 | YaRN mscale_all_dim (default 0) |

For the Pixtral vision projector (`mmproj` GGUF), the standard
`clip.vision.*` keys cover the encoder dims, plus `mm.*` keys for the
projector layers.

## 6. Weight naming convention

```
token_embd.weight                          # [vocab, hidden]
blk.{L}.attn_norm.weight                   # pre-attention RMSNorm
blk.{L}.attn_q.weight                      # Q projection (before fusion)
blk.{L}.attn_k.weight                      # K projection (before fusion)
blk.{L}.attn_v.weight                      # V projection (before fusion)
blk.{L}.attn_qkv.weight                    # fused Q+K+V (after fusion)
blk.{L}.attn_output.weight                 # output projection
blk.{L}.ffn_norm.weight                    # pre-FFN RMSNorm
blk.{L}.ffn_gate.weight  }                 # before fusion
blk.{L}.ffn_up.weight    }
blk.{L}.ffn_gate_up.weight                 # after fusion: [2*intermediate, hidden]
blk.{L}.ffn_down.weight                    # FFN down projection
output_norm.weight                         # final RMSNorm
output.weight                              # LM head (optional if tied)
```

### Vision encoder weights (Pixtral)

```
v.patch_conv.weight                        # Conv2D patch embedding [hidden, C, P, P]
v.patch_conv.bias                          # Conv2D bias (optional)
v.encoder_norm.weight                      # encoder input RMSNorm
v.blk.{L}.attn_norm.weight                 # pre-attention RMSNorm
v.blk.{L}.attn_q.weight                    # Q projection
v.blk.{L}.attn_k.weight                    # K projection
v.blk.{L}.attn_v.weight                    # V projection
v.blk.{L}.attn_output.weight               # output projection
v.blk.{L}.ffn_norm.weight                  # pre-FFN RMSNorm
v.blk.{L}.ffn_gate.weight                  # SiLU gate
v.blk.{L}.ffn_up.weight                    # up projection
v.blk.{L}.ffn_down.weight                  # down projection
mm.norm.weight                             # projector RMSNorm
mm.patch_merger.merging_layer.weight       # spatial patch merger
mm.linear_1.weight                         # projector linear 1
mm.linear_2.weight                         # projector linear 2
```

## 7. TensorSharp implementation walkthrough

Constructor (`Mistral3Model(string ggufPath, BackendType backend)`):

1. `ParseBaseConfig()`.
2. Reads `_attnKeyLen` / `_attnValLen` from `KeyLength` / `ValueLength` (or
   defaults to `HeadDim`). Reads `_ropeDim`.
3. Reads YaRN parameters.
4. `ParseTokenizer()`, `LoadWeights()`.
5. `FuseQKVWeights()`, `FuseGateUpWeights()`.
6. `PrepareCudaQuantizedWeightsForInference()`.
7. `InitKVCache(maxSeqLen)` allocates per-layer K and V tensors of shape
   `[NumKVHeads, maxSeqLen, headDim]` (separate `_attnKeyLen` /
   `_attnValLen` are honored).
8. `PrecomputeConstants()` builds:
   - The per-layer weight name arrays — different shapes depending on
     whether QKV / gate-up are fused for that layer.
   - The RoPE frequency table `_ropeFreqs[halfDim]`.
   - YaRN-corrected frequencies via `ApplyYarnFreqCorrection()` when
     `_ropeType == "yarn"` and `_ropeOrigCtx > 0`.

`Forward(int[] tokens)` runs the per-op managed loop:

- Embedding lookup.
- Optional vision injection at `<image_pad>`-marked positions.
- For each layer: RMSNorm, QKV (fused or split), RoPE with YaRN-corrected
  frequencies, optional position-dependent Q scaling, attention, output
  projection + residual, FFN, residual.
- For prefill, the residual is narrowed to the last token before
  `output_norm` so the LM head matmul produces a single row.
- Final RMSNorm, LM head, copy to `_logitsBuffer`.

## 8. Prefill optimization

- **Fused QKV and gate / up** cut the per-layer GEMM count.
- **Last-row narrow** before `output_norm` so the LM head matmul only
  produces `[1, vocab]`.
- **YaRN frequency table built once** at load time. The decode path uses
  the same `_ropeFreqs[]` array.
- **Backend ops** dispatch through `Ops.RoPEEx` and `Ops.SiLUMul`,
  which run as native kernels on GGML / CUDA backends.

There is no chunked prefill (full attention only — no SWA), no fused
per-layer prefill kernel, and no native whole-model decode for Mistral 3.
Adding either would shrink decode latency and TTFT.

## 9. Decode optimization

- Pre-resolved per-layer weight name arrays (`_layerWeightNames[L][]`)
  allocated in `PrecomputeConstants()` keep the hot loop allocation-free.
- The `_layerQkvFused[]` flag picks the fused vs. split QKV path per layer
  without allocations.
- Pre-allocated decode position arrays and YaRN-corrected `_ropeFreqs[]`
  avoid recomputation per token.
- Quantized weights stay quantized in `_quantWeights`; matmul calls run
  through the backend's native quantized matmul.

## 10. Memory and KV cache strategy

- Per-layer K and V tensors of shape `[NumKVHeads, maxSeqLen, headDim]`
  using `_attnKeyLen` / `_attnValLen` for the per-head dim.
- `ResetKVCache()` zeroes everything and calls
  `InvalidateTensorDeviceCache()` to sync GPU state.
- `TruncateKVCache(int tokenCount)` is supported (used for multi-turn KV
  reuse on the server).
- The Pixtral vision encoder dequantizes its weights to F32 at load time;
  the LM weights stay quantized when supported by the backend.

## 11. Batched / paged forward (continuous batching)

Mistral 3 is the **reference implementation** of `IBatchedPagedModel.ForwardBatch`
in TensorSharp ([`Mistral3Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Mistral3/Mistral3Model.BatchedForward.cs)).
It runs through the shared continuous-batching engine (`InferenceEngine` +
`ContinuousBatchScheduler` + `BatchExecutor`) described in
[`docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md).

Key properties:

- **Default-on, no opt-in env var.** Continuous batching for Mistral 3 is
  always available; the server's `--no-continuous-batching` flag forces the
  legacy per-seq KV-swap path for every model, including Mistral 3.
- **Per-layer paged K/V buffers** of layout
  `[numBlocks * blockSize * numKvHeads * headDim]`, lazily grown by
  `EnsurePagedBuffersAllocated`. The "grow" path copies existing K/V into
  the new buffer so already-scheduled sequences don't lose their state.
- **Fused vs. unfused QKV per layer** — `ForwardBatch` branches on
  `_layerQkvFused[layer]` to pick between the fused `attn_qkv.weight`
  matmul and the three independent `attn_q/k/v.weight` matmuls. Both paths
  batch across the entire token axis in a single GEMM per layer.
- **Per-token YaRN RoPE.** The `_ropeFreqs[]` array built at load time is
  applied through `Ops.RoPEEx` with an explicit `positions[]` array, and
  per-token YaRN position-dependent Q scaling
  (`q *= 1 + beta * log(1 + floor(pos / orig_ctx))`) is performed inside
  the batched loop so each token in the batch picks its own scale factor.
- **K/V scatter via `slotMapping`** writes fresh K and V into the layer's
  paged buffer at `blockId * blockSize + offset`. No KV-state extract /
  inject between sequences in the batched path.
- **Three paged-attention kernels selectable via `TS_PAGED_ATTN_KERNEL`**:
  - `native` (default): `TSGgml_PagedAttentionForward` —
    C++ memcpy gather of K/V from the paged buffer per sequence, then
    dispatch `ggml_flash_attn_ext` (the same fused Metal/CUDA flash
    attention kernel the legacy per-seq path uses).
  - `tensor`: `TensorPagedAttention.Forward` — C# Tensor-based gather plus
    `Ops.AddmmBatch` + `GgmlBasicOps.AttentionSoftmaxWithSinks` per
    sequence. Slower than `native` because of repeated GPU dispatches.
  - `managed`: `ManagedPagedAttention.Forward` — pure-C# online-softmax
    loop, parallelised over `(seq, head)`. Correctness fallback on any
    backend.
- **Vision-embedding injection** runs upstream of the per-layer loop
  (same path as legacy forward). The multimodal injector serialises
  prompt preparation behind a lock for multimodal turns; text-only turns
  prepare in parallel.

**Verified correctness on the real GGUF**
([`Mistral3BatchedForwardTests`](../../InferenceWeb.Tests/Mistral3BatchedForwardTests.cs)):
- `Mistral3_BatchSize1_ForwardBatchMatchesLegacyTop1` — legacy top-1 =
  batched top-1 = 1091 on a 6-token probe prompt.
- `Mistral3_BatchSize2_DistinctSequencesProduceDistinctLogits` — two
  diverging prompts in one batch produce distinct top-1 tokens, proving
  the K/V is correctly partitioned.

**Measured throughput on Ministral-3-14B-Instruct-2512-Q4_K_M (Apple
M4 Pro, GgmlMetal, 4 parallel chat-style prompts)**:

| Path | Long context (~800 tok/seq) wall | Tokens/sec | vs. legacy |
|---|---|---|---|
| Batched + Native kernel (default) | **42.37 s** | **0.6** | **1.21×** |
| Per-sequence KV-swap (legacy GGML) | 53.95 s | 0.4 | 1.0× |
| Batched + Tensor kernel | 71.42 s | 0.3 | 0.76× |
| Batched + Managed kernel | 111.02 s | 0.2 | 0.49× |

The native kernel beats the legacy GGML per-sequence path by ~21% on
long context — the headline result the whole continuous-batching effort
has been building toward.

**Prefix-cache validation**: in the same long-context run the engine
shared six full prompt blocks across the four sequences
(`reused=1536`, `hashedCached=3`), exercising the block-hash prefix
cache end-to-end on a real GGUF.

## 12. Output parser and chat template

- `PassthroughOutputParser` — Mistral 3 has no thinking / tool-call wire
  format.
- Chat template uses Mistral's standard chat format
  (`[INST]...[/INST]<s>...</s>`). Falls back to the hardcoded template when
  the GGUF lacks a Jinja2 template.
- The image placeholder is `<image_pad>` and `ChatTemplate.ExpandImageTokens`
  expands one `<image_pad>` into the right number of placeholder tokens for
  the corresponding image's encoded length.

## 13. Optimization opportunities

- **Native whole-model decode** — the entire legacy forward pass is
  managed C# with backend-dispatched matmul. A native single-call decode
  path (analogous to Qwen 3's `TransformerModelDecode`) would remove most
  of the managed overhead from the per-seq path.
- **Fused single-graph decode** — a `Mistral3ModelDecode` GGML kernel
  would significantly improve Metal / CUDA throughput by collapsing the
  per-layer dispatches in the legacy path.
- **Quantized vision encoder** — Pixtral weights are currently
  dequantized to F32 at load time, increasing memory usage for image-heavy
  workloads. Supporting quantized vision weights directly would shrink the
  resident set and cut load time.
- **Fused prefill attention** — adopting Qwen 3.5's `FusedPrefillAttention`
  approach would reduce per-layer prefill round-trips on GGML / CUDA.
- **One batched ggml graph for the whole batch** — `ForwardBatch`
  currently builds a separate paged-attention graph per sequence per
  layer. Folding them into a single per-layer graph would amortise the
  compile / launch overhead for batches with many short sequences.
