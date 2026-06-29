# Qwen 3.5 / 3.6 family

[← back to model index](README.md)

| Property | Value |
|---|---|
| Provider | Alibaba |
| GGUF architecture keys | `qwen35`, `qwen35moe`, `qwen3next` |
| Source class | [`Qwen35Model`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.cs) (legacy per-seq) + partial in [`Qwen35Model.GatedDeltaNet.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.GatedDeltaNet.cs) + [`Qwen35Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.BatchedForward.cs) (`IBatchedPagedModel`) |
| Vision encoder | [`Qwen35VisionEncoder`](../../TensorSharp.Models/Models/Qwen35/Qwen35VisionEncoder.cs) |
| Image processor | [`Qwen35ImageProcessor`](../../TensorSharp.Models/Models/Qwen35/ImageProcessor.cs) |
| Example models | Qwen3.5-9B (dense hybrid), Qwen3.5-32B, Qwen3.5-35B-A3B / Qwen3.6-35B-A3B (MoE-family) |
| Modalities | Text, image |
| Thinking mode | Yes (`<think> ... </think>`) |
| Tool calling | Yes (`<tool_call>{...}</tool_call>`) |
| Batched / paged forward | **Default ON** — set `TS_QWEN35_BATCHED=0` (or `--no-continuous-batching`) to force the legacy per-sequence KV-swap path for A/B comparison. Includes a per-slot GatedDeltaNet recurrent-state pool and optional native batched GDN kernel (`TS_QWEN35_BATCHED_GDN_NATIVE=1`). See §11. |
| MTP speculative decoding | Qwen 3.6 — NextN draft block embedded in the trunk GGUF (no separate file); engage with `--mtp-spec`. GDN recurrent-state snapshot/rollback on partial accept. Profitable on ggml backends and the pure-C# `cuda` backend. See §12. |
| Output parser | `Qwen35OutputParser` (inherits `Qwen3OutputParser`) |

## 1. Origin and intent

The Qwen 3.5 / 3.6 family is Alibaba's first "hybrid" Qwen line and is the
most heavily optimized model in TensorSharp:

- **Hybrid attention + recurrent stack.** Most layers are
  GatedDeltaNet (a linear-attention / SSM-style recurrent layer); a smaller
  subset are FullAttention layers placed every `full_attention_interval`-th
  position. The recurrent layers carry a `[numVHeads, headVDim, headKDim]`
  hidden state across tokens; the attention layers carry a standard KV cache.
- **MoE FFN** in `qwen35moe` / `qwen3next` variants (e.g. Qwen3.5-35B-A3B,
  Qwen3.6-35B-A3B). 256 routed experts with 8 active per token, plus an
  optional always-on shared expert.
- **Sigmoid-gated FullAttention.** The Q projection emits twice as many rows
  (`2 × numHeads × headDim`); the second half is deinterleaved as a
  sigmoid gate that multiplies the attention output before the output
  projection.
- **Dynamic-resolution vision.** Image inputs go through
  `Qwen35VisionEncoder`, a 2D-RoPE ViT that supports arbitrary image
  resolutions via patch embedding plus a spatial merger.

All three GGUF arch keys (`qwen35`, `qwen35moe`, `qwen3next`) load through the
same `Qwen35Model`. The MoE variants additionally light up the fused MoE
kernels described in § 8 / § 9.

## 2. Model architecture

```
                tokens (int[])
                      │
              token_embd.weight
                      │
        [optional] InjectVisionEmbeddings (image)
                      │
        ┌──── × NumLayers ──────────────────────────────┐
        │ if (l + 1) % fullAttentionInterval != 0:      │
        │     GatedDeltaNet(l)         (recurrent)      │
        │ else:                                         │
        │     FullAttention(l)         (gated GQA)      │
        │                                               │
        │ residual = block_input + branch_output        │
        │ post_attention_norm                            │
        │ if MoE layer:                                 │
        │     route ─► weighted Σ SwiGLU experts        │
        │     + optional shared expert                  │
        │     residual += moe_out                       │
        │ else:                                         │
        │     ffn_gate_up ─► SwiGLU ─► ffn_down         │
        │     residual += ffn_out                       │
        └───────────────────────────────────────────────┘
                      │
              RMSNorm(output_norm)
                      │
              LM head (output.weight)
                      │
                      ▼
                   logits
```

For a 48-layer model with `full_attention_interval == 4`: 36 GatedDeltaNet
layers + 12 FullAttention layers; each FullAttention layer is the 4th, 8th,
12th, ... layer.

## 3. Forward graph

### 3.1 FullAttention layer

```
hidden ─► RMSNorm(attn_norm)
       ─► attn_qkv.weight matmul
            output rows are laid out as
            [Q (2 * qDim) ‖ K (kvDim) ‖ V (kvDim)]
       ─► deinterleave Q rows: even half = Q, odd half = sigmoid gate
       ─► per-head RMSNorm(Q, attn_q_norm.weight)
       ─► per-head RMSNorm(K, attn_k_norm.weight)
       ─► RoPE_NeoX(Q, K) at positions startPos..startPos+seqLen-1
       ─► append (K, V) to KV cache
       ─► attention(Q, KCache, VCache)        // standard causal GQA
       ─► attn_out *= sigmoid(gate)
       ─► attn_output.weight matmul → o
       ─► residual = hidden + o
       ─► RMSNorm(post_attention_norm)
       ─► (FFN block, see § 3.3)
```

### 3.2 GatedDeltaNet layer (recurrent)

```
hidden ─► RMSNorm(attn_norm)
       ─► attn_qkv (fused Q+K+V) ─► QKV (concatenated along time)
       ─► attn_gate ─► z (sigmoid gate)
       ─► ssm_beta ─► β (per-head)
       ─► ssm_alpha ─► α (per-head)
       ─► conv1d_step(QKV) using ssm_conv1d.weight (kernel size convK)
       ─► SiLU
       ─► split into Q, K, V along the head/feature dim
       ─► if (numKHeads != numVHeads): repeat Q and K per head
       ─► L2 normalize Q and K per head
       ─► per-token recurrent update (per V-head):
              g = exp(α)              // forget gate ∈ (0, 1]
              s = sigmoid(β)          // input gate
              kv_mem = state @ k      // [headVDim, 1]
              state = g * state + s * (v - kv_mem) ⊗ kᵀ
              y = state @ q           // [headVDim, 1]
       ─► out = SiLU(z) * RMSNorm(y, ssm_norm.weight)
       ─► out matmul ssm_out.weight → o
       ─► residual = hidden + o
       ─► RMSNorm(post_attention_norm)
       ─► (FFN block, see § 3.3)
```

The recurrent state has shape `[numVHeads, headVDim, headKDim]` per layer and
is held in `_deltaStateTensor[layer]`. The conv state
(`(convKernel - 1) × qkvDim` floats per layer) lives in `_convState[layer]`.

### 3.3 FFN block (dense or MoE)

**Dense FFN** (when `_isMoeLayer[l] == false`):

```
hidden ─► RMSNorm(post_attention_norm)
       ─► ffn_gate_up ─► [gate ‖ up]
       ─► g = SiLU(gate)
       ─► ffn_down × (g * up) → o
       ─► residual += o
```

**MoE FFN** (`qwen35moe` / `qwen3next`):

```
hidden ─► RMSNorm(post_attention_norm)
       ─► logits = ffn_gate_inp.weight × hidden
       ─► weights, idx = topK(softmax(logits), _numExpertsUsed)
       ─► if _normTopKProb: weights /= sum(weights)
       ─► moe_out = Σ_e weights[e] * (
              ffn_down_exps[idx[e]] × SiLUMul(
                  ffn_gate_up_exps[idx[e]] × hidden) )
       ─► if shared expert: moe_out += SiLUMul(
              ffn_gate_up_shexp × hidden) × ffn_down_shexp
       ─► residual += moe_out
```

On a GGML backend the entire MoE block (routing, all selected experts, the
shared expert, and the residual add) collapses into one
`MoEExpertsSwiGLUResidual` kernel dispatch (see § 9).

## 4. Components in detail

### 4.1 GatedDeltaNet (linear-attention recurrent layer)

GatedDeltaNet is a "delta-net" linear attention layer with a per-head
recurrent state that updates by:

```
state' = exp(α_t) · state + sigmoid(β_t) · (v_t − state · k_t) ⊗ k_tᵀ
y_t    = state' · q_t
```

Where:
- `state ∈ ℝ^{headVDim × headKDim}` per V-head.
- `q_t, k_t ∈ ℝ^{headKDim}`, `v_t ∈ ℝ^{headVDim}`.
- `α_t, β_t ∈ ℝ` are per-head, per-token gates produced by `ssm_alpha` and
  `ssm_beta` linear projections.

This is computed via `Ops.AddmmBatch` for the rank-1 outer-product update
(batched over heads), and the per-step output `y_t = state · q_t` is then
RMSNorm-ed and gated by `SiLU(z_t)`. `Ops.SiLUMul` and `Ops.RMSNorm` carry
those steps.

`_convKernel` is small (typically 4) and the conv1d sliding window is
maintained in `_convState[layer][]` as a flat float buffer. `Conv1dStep`
applies `ssm_conv1d.weight` over the window followed by SiLU before the QKV
split.

### 4.2 FullAttention with sigmoid gate

`attn_qkv.weight` rows are laid out so that one matmul produces:

- 2 × `qDim` rows, interleaved Q and gate (every other column).
- `kvDim` rows of K.
- `kvDim` rows of V.

The deinterleave step (`SplitQGateInterleaved`) writes Q into one buffer and
the gate into another. Per-head RMSNorm runs on Q and K, RoPE rotates Q and
K, and the standard causal attention proceeds. Before the output projection,
`Ops.SigmoidMul` does `attn_out *= sigmoid(gate)` in place.

### 4.3 Mixture-of-Experts FFN

| Setting | Source |
|---|---|
| Routed expert count `_numExperts` | `qwen35.expert_count` (256) |
| Active experts per token `_numExpertsUsed` | `qwen35.expert_used_count` (8) |
| Routed expert FFN width `_expertFfnLength` | `qwen35.expert_feed_forward_length` |
| Shared expert FFN width `_sharedExpertFfnLength` | `qwen35.expert_shared_feed_forward_length` |
| Renormalize TopK weights | `qwen35.expert_weights_norm` (forced true for Qwen 3.5+) |

Routing computes `linear(hidden) → softmax → topK(_numExpertsUsed)`. When
`_normTopKProb` is true the selected weights are renormalized to sum to 1. A
shared expert (always-on, untouched by the router) optionally adds in
parallel.

### 4.4 RoPE

NeoX-style rotary embeddings. Frequencies are cached in `_ropeFreqs[halfDim]`
once. Position tensors are cached across attention layers per forward pass:
`_cachedRoPEPosQ` / `_cachedRoPEPosK` are reused when `(seqLen, startPos)`
match across all 12 attention layers, eliminating duplicate allocation +
population.

`qwen3next` GGUFs additionally ship `qwen35.rope.dimension_sections` which
defines MRoPE section boundaries. When present, the RoPE step uses
multi-modal RoPE; when absent, plain NeoX RoPE is used.

### 4.5 Vision encoder (`Qwen35VisionEncoder`)

A SigLIP-style ViT with:

- Conv-style patch embedding implemented as parallel im2col + matmul (replaces
  the original quintuple-nested scalar loop).
- 2D RoPE position embeddings.
- LayerNorm + multi-head attention + residual + LayerNorm + GELU MLP +
  residual blocks.
- Spatial patch merging (`SpatialMergeSize × SpatialMergeSize` neighbours into
  one merged token).
- A multimodal projector (RMSNorm + Linear + GELU + Linear) to the LM hidden
  dim.

On a GGML backend each block runs through two fused kernels
(`FusedVisionAttention` and `FusedVisionMLP`); see § 8.

## 5. Parameters and settings

| Key | Type | Meaning |
|---|---|---|
| `qwen35.ssm.inner_size` | uint32 | `headVDim * numVHeads` |
| `qwen35.ssm.state_size` | uint32 | `headKDim` |
| `qwen35.ssm.group_count` | uint32 | `numKHeads` |
| `qwen35.ssm.time_step_rank` | uint32 | `numVHeads` |
| `qwen35.ssm.conv_kernel` | uint32 | Conv1d kernel size |
| `qwen35.full_attention_interval` | uint32 | Every Nth layer is full attention (default 4) |
| `qwen35.rope.dimension_sections` | int32[] | MRoPE section boundaries (qwen3next) |
| `qwen35.rope.dimension_count` | uint32 | RoPE dim count |
| `qwen35.expert_count` | uint32 | Routed expert count (0 ⇒ dense FFN) |
| `qwen35.expert_used_count` | uint32 | TopK routed experts per token |
| `qwen35.expert_feed_forward_length` | uint32 | Routed expert FFN width |
| `qwen35.expert_shared_feed_forward_length` | uint32 | Shared expert FFN width (0 ⇒ no shared expert) |
| `qwen35.expert_weights_norm` | bool | Renormalize TopK weights to sum=1 |

## 6. Weight naming convention

```
# FullAttention layers (every Nth):
blk.{L}.attn_norm.weight
blk.{L}.attn_q.weight                      # [numHeads*headDim*2, hidden]  (Q+gate interleaved)
blk.{L}.attn_k.weight                      # [numKVHeads*headDim, hidden]
blk.{L}.attn_v.weight                      # [numKVHeads*headDim, hidden]
blk.{L}.attn_q_norm.weight
blk.{L}.attn_k_norm.weight
blk.{L}.attn_output.weight
blk.{L}.post_attention_norm.weight

# GatedDeltaNet layers:
blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight                    # [qkDim*2 + vDim, hidden]
blk.{L}.attn_gate.weight                   # z gate [ssmDInner, hidden]
blk.{L}.ssm_beta.weight                    # [numVHeads, hidden]
blk.{L}.ssm_alpha.weight                   # [numVHeads, hidden]
blk.{L}.ssm_conv1d.weight                  # [qkvDim, convKernel]
blk.{L}.ssm_dt.bias                        # dt bias [numVHeads]
blk.{L}.ssm_a                              # a parameter [numVHeads]
blk.{L}.ssm_norm.weight                    # output RMSNorm
blk.{L}.ssm_out.weight                     # output projection
blk.{L}.post_attention_norm.weight

# Dense FFN (when ffn_gate_inp.weight is absent):
blk.{L}.ffn_gate_up.weight
blk.{L}.ffn_down.weight

# MoE FFN (qwen35moe / qwen3next):
blk.{L}.ffn_gate_inp.weight                # router [numExperts, hidden]
blk.{L}.ffn_gate_up_exps.{E}.weight        # fused expert gate+up
blk.{L}.ffn_down_exps.{E}.weight           # expert down
blk.{L}.ffn_gate_up_shexp.weight           # shared expert gate+up (optional)
blk.{L}.ffn_down_shexp.weight              # shared expert down (optional)
blk.{L}.ffn_gate_inp_shexp.weight          # shared expert gate (optional)

# Standard final pieces:
output_norm.weight
output.weight
```

`FuseAttentionProjectionWeights()` fuses the per-layer Q+K+V into
`attn_qkv.weight` for FullAttention layers when the GGUF stores them
separately. `FuseRecurrentInputWeights()` packs five recurrent projections
(`attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha`, plus an empty slot) into
one fused `ssm_in_proj.weight` for GatedDeltaNet layers. `FuseGateUpWeights()`
fuses dense FFN gate and up.

## 7. TensorSharp implementation walkthrough

Constructor:

1. `ParseBaseConfig()`.
2. `ParseGdnConfig(arch)` reads the SSM dims (`_ssmDInner`, `_ssmDState`,
   `_ssmNGroup`, `_ssmDtRank`, `_convKernel`).
3. `_fullAttentionInterval` and the per-layer `_isRecurrent[]` array.
4. `_mropeSections`, `_ropeDimCount`.
5. MoE config (`_numExperts`, `_numExpertsUsed`, `_expertFfnLength`,
   `_sharedExpertFfnLength`).
6. `ParseTokenizer()`, `LoadWeights()`.
7. `FuseAttentionProjectionWeights()`, `FuseRecurrentInputWeights()`,
   `FuseGateUpWeights()`.
8. `DetectMoeLayers()` walks every block and sets `_isMoeLayer[l]` and the
   `_hasSharedExperts[l]` / `_hasSharedExpertGate[l]` flags.
9. `BuildLayerKeys()` precomputes every weight name string used by the hot
   loops (attn_norm, attn_qkv, attn_q_norm, attn_k_norm, attn_output,
   ffn_gate_up, ffn_down, ffn_gate_inp, ffn_gate_up_exps[e],
   ffn_down_exps[e], ...).
10. `InitMoeBuffers()` pre-allocates router / expert pointer arrays, MoE
    work tensors, and the `IntPtr[]` arrays used by the batched MoE kernel.
11. `PrepareCudaQuantizedWeightsForInference()`.
12. `InitCaches(initialCacheLength, maxContextLength)` allocates KV caches
    for FullAttention layers and zeroes the recurrent states.
13. `PrecomputeRoPE()`, `InitGDNBuffers()`, `CacheRecurrentWeights()`.

`Forward(int[] tokens)` distinguishes prefill (`seqLen > 1`) from decode
(`seqLen == 1`) and routes each layer through one of:

- The fused per-layer attention decode kernel
  (`TryFusedAttnLayerDecode`) when the cached sequence length is past the
  `FUSED_ATTN_LAYER_MIN_SEQ_LEN` threshold (default 4096).
- The fused prefill attention kernel (`FusedPrefillAttention`) for
  multi-token prefill on a GGML backend.
- The fused output-projection + FFN kernel (`FusedOutProjFFN`) for both
  attention and recurrent layers with dense FFN.
- The fused output-projection + post-norm + router kernel
  (`FusedOutProjNormRouter`) for recurrent layers feeding into MoE.
- The batched MoE kernel (`MoEExpertsSwiGLUResidual`) for MoE FFN decode.
- A standard managed C# path otherwise.

## 8. Prefill optimization

### Fused prefill attention (`FusedPrefillAttention`)

A single GGML graph that performs:

1. `Q × K^T` (scaled) with GQA head broadcasting.
2. Causal mask (lower-triangular, with `maskStartPos` offset for continuation
   prefill).
3. Softmax.
4. Scores × V → attention output.

Replaces ~5 separate C#-to-GGML round-trips per attention layer. Handles
both initial prefill (empty KV cache) and continuation prefill (existing KV
cache entries from prior turns). The `inputFormat` parameter supports both
`[seqLen, numHeads, headDim]` and `[numHeads, seqLen, headDim]` layouts.

### Fused output-projection + FFN (`FusedOutProjFFN`)

For both FullAttention and GatedDeltaNet layers with dense FFN, one GGML
graph performs the output projection + residual add + post-attention RMSNorm
+ ffn_gate_up matmul + SiLU + ffn_down matmul + residual. Two GPU
round-trips collapse into one. Disable for A/B benchmarking with
`QWEN35_DISABLE_FUSED_FFN=1`.

### Parallelized Q / gate deinterleave

The Q + sigmoid-gate deinterleave in FullAttention prefill runs through
`Parallel.For` across tokens. Each token's deinterleave is independent, so
this scales linearly with CPU core count for long prompts.

### In-place QK RMSNorm

`ApplyPerHeadRMSNorm()` reshapes the input via `View(seqLen * numHeads,
headDim)` and runs RMSNorm directly into the same memory. Row-independent
RMSNorm is invariant to row order, so the reshaped view is valid for both Q
and K. This avoids one tensor allocation and one `Ops.CopyTo` per Q/K per
layer.

### RoPE position tensor caching

`_cachedRoPEPosQ` / `_cachedRoPEPosK` tensors are regenerated only when
`(seqLen, startPos)` changes, then reused across all attention layers.

### GEMM-based vision patch embedding

`Qwen35VisionEncoder` reformulates patch embedding as parallel im2col +
matmul, replacing a single-threaded scalar quintuple-nested loop with a
GPU-accelerated matmul. Combined with the fused vision encoder blocks this
significantly accelerates the multi-tile image path.

### Fused vision encoder blocks

- **`FusedVisionAttention`**: LayerNorm + QKV projection + bias + 2D RoPE +
  scaled dot-product attention + output projection + bias + residual into
  one GGML graph dispatch (~8 ops → 1).
- **`FusedVisionMLP`**: LayerNorm + up projection + bias + GELU + down
  projection + bias + residual into one GGML graph dispatch (7 ops → 1).

Combined, each vision encoder block goes from ~15 GPU round-trips to 2.

### Cross-layer caches and parallelism

- `Parallel.For` across block rows in `ReorderToBlockOrder` for spatial
  merge.
- GPU-aware QKV split: on GPU backends `Narrow` + `NewContiguous` keeps data
  on-device; on CPU a fused parallel `Buffer.MemoryCopy` splits Q/K/V in one
  sweep.

### Chunked parallel GatedDeltaNet recurrent prefill

For `seqLen ≥ 64` on a GGML backend the per-token recurrent loop is replaced
by a fused chunked SSM scan (`GatedDeltaNetChunkedPrefill` →
`GgmlBasicOps.GatedDeltaNetChunked`, native side
`TSGgml_GatedDeltaNetChunkedF32` in `ggml_ops_gated_delta_net.cpp`). The
implementation follows the chunked-scan pattern used by Mamba's parallel-scan
kernels:

1. **CPU prep (parallel across tokens):** the per-channel 1D convolution +
   SiLU + Q/K/V/Z/α/β packing is fused into one `Parallel.ForEach` pass with
   thread-local SiLU scratch (`ApplySiLUInPlaceScratch`). The recurrent ring
   state is updated from the last `convKernel - 1` tokens in a single sweep
   so subsequent layers see the correct conv state.
2. **CPU pre-compute of the gate / β-sigmoid:** the trivially small
   `[seqLen, H]` `softplus(α + dt_bias)·a_log` and `sigmoid(β)` ops run on the
   CPU via `TensorPrimitives.Sigmoid`. This replaces four GPU ops on a
   tensor that fits in L2 anyway and removes two per-layer constant uploads.
3. **Single fused GGML graph dispatch (`GatedDeltaNetChunked`):** the entire
   delta-net block (Q/K L2-norm, scale, β-multiply, chunked triangular-solve
   attention with `(I − decay·k·kᵀ)⁻¹·decay·k·kᵀ`, cross-chunk recurrent state
   propagation, RMSNorm, `silu(z)` gating) runs as one Metal/CUDA graph. The
   graph is built once per `(T, H, D, chunkSize, eps)` shape and reused via
   `g_gdn_chunked_cache`; subsequent calls only re-bind input data via
   `ggml_backend_tensor_set`. `ggml_solve_tri` does the per-chunk
   `(I − attn_lower)·X = attn_init` solve in parallel across heads + chunks.
4. **Cross-chunk dependency** is the one remaining sequential edge: chunk
   `c+1`'s recurrent state depends on chunk `c`'s. The loop has length
   `nC = ceil(T/64)`, so the sequential depth is `O(T/64)` instead of `O(T)`,
   which is the same asymptotic span as Mamba's parallel-scan kernels.

Tunable env vars:

- `GDN_CHUNK_PREFILL_MIN_SEQ_LEN=N` overrides the threshold (default: 2 on
  ggml_cuda, 6 elsewhere). On CUDA the chunked kernel wins for every
  multi-token batch — the per-token loop's per-layer host/device sync dwarfs
  the 64-padding waste; measured on Qwen3.6-27B IQ2_XXS it cut MTP
  speculative-verify decode from 217 to 174 ms/token. Single-token decode
  steps still use the per-token loop. Set to `64` for the old
  long-prefill-only behavior; set very high to disable.
- `GDN_DISABLE_CHUNKED_PREFILL=1` forces the per-token CPU loop on every
  prefill call; useful for A/B comparison.
- `GDN_VERIFY_CHUNKED=1` runs the chunked path on a snapshot of the
  recurrent state, restores the snapshot, runs the per-token loop on the
  same starting state, and reports the maximum absolute / relative drift
  between the two outputs and recurrent states. The per-token result is
  used downstream so a divergent chunked path cannot poison subsequent
  layers. Verified call count and max diffs are reported in
  `PrintGdnTimingStats`.

Measured speedups on Qwen3.6-35B-A3B (UD-IQ2_XXS, 30 GDN layers, MoE FFN)
running on Apple Metal (M-series), KV cache f16, 3 runs each:

| prefill tokens | chunked ms / call | per-token ms / call | GDN-block speedup | total prefill speedup | tokens match |
|---:|---:|---:|---:|---:|:---:|
|  128 |   3.77 |   7.32 | **1.94×** | 1.03× | ✓ |
|  512 |  12.91 |  29.98 | **2.32×** | 1.09× | ✓ |
| 2048 |  49.66 | 118.88 | **2.39×** | 1.11× | ✓ |

`GDN_VERIFY_CHUNKED=1` reports max output `|Δ|` ≈ 3 × 10⁻³ and max state `|Δ|`
≈ 1 × 10⁻² across all three prefill lengths, with zero warnings (the warn
threshold is 5 × 10⁻³ absolute / 5 × 10⁻² relative). This is the expected
FP32 noise floor for accumulating 256–2048 token-step multiplications across
30 layers under different summation orders (Metal MM accumulation vs. scalar
CPU). Top sampled tokens match exactly between the two paths at all three
lengths.

## 9. Decode optimization

### Fused per-layer attention decode (`Qwen35AttentionLayerDecode`)

A single GGML graph that performs the entire FullAttention block:

1. RMSNorm(input) * `attn_norm.weight`.
2. Fused QKV matmul → `[Q+gate (2*qDim), K (kvDim), V (kvDim)]`.
3. Strided view + contiguous copy split for Q, gate, K, V.
4. Per-head RMSNorm(Q) * `attn_q_norm.weight`, per-head RMSNorm(K) *
   `attn_k_norm.weight`.
5. NeoX-style RoPE on Q and K at the current `position`.
6. Append the new K, V into the persistent KV cache via `ggml_cpy` views.
7. `ggml_flash_attn_ext` against the populated KV cache window (handles GQA
   broadcasting).
8. `attn_out *= sigmoid(gate)`.
9. Output projection + residual add → updated hidden state written through
   the host pointer.

Replaces 1 standalone `FusedRmsNormMatMulQuant` + ~6 small CPU-side ops + 1
standalone `FusedMatMulQuantAdd` with one fused dispatch. The kernel only
engages once `position + 1 >= FusedAttnLayerDecodeMinSeqLen` (default 4096;
override via `FUSED_ATTN_LAYER_MIN_SEQ_LEN=N`) because the GPU flash-attn
path has a fixed setup cost that only amortizes for long contexts. Below
the threshold the existing `FusedRmsNormMatMulQuant` + CPU-SIMD attention +
`FusedMatMulQuantAdd` path is retained.

### Fused output-projection + norm + router (MoE recurrent decode)

For GatedDeltaNet layers followed by MoE FFN during decode,
`FusedOutProjNormRouter` merges the GDN output projection, residual add,
post-attention RMSNorm, and the MoE router projection into one GGML graph
dispatch (3 ops → 1). The pre-computed router logits feed directly into
`TryMoEResidualDecodeWithRouter()`, eliminating a separate router dispatch
per MoE layer.

### Batched GPU MoE (`MoEExpertsSwiGLUResidual`)

For `qwen35moe` / `qwen3next` on a GGML backend, MoE expert computation
during decode collapses into a single `MoEExpertsSwiGLUResidual()` call per
layer:

1. All routed experts: `SwiGLU(hidden × gate_up_exps[e]) × down_exps[e]`,
   weighted by the router probabilities.
2. The optional shared expert: `SwiGLU(hidden × gate_up_shexp) × down_shexp`.
3. Residual addition into the running hidden state.

Pre-cached `QuantizedWeight` references and `IntPtr[]` arrays
(`_routedExpertGateUpQW`, `_routedExpertDownQW`, `_sharedExpertGateUpQW`,
`_sharedExpertDownQW`, plus parallel `*Ptrs` arrays) eliminate dictionary
lookups and allocations in the hot decode loop.

### Pre-allocated GDN decode buffers

Allocated once in `InitGDNBuffers()`:

| Buffer | Shape | Purpose |
|---|---|---|
| `_gdnConvOutT` | `[1, qkvDim]` | Conv1d output + SiLU |
| `_gdnKBuf` / `_gdnQBuf` | `[numVHeads, headKDim]` | Q/K after expansion + L2 norm |
| `_gdnVBuf` | `[numVHeads, headVDim]` | V split |
| `_gdnKvMemBuf` | `[numVHeads, headVDim, 1]` | `state @ k` intermediate |
| `_gdnCoreOutBuf` | `[numVHeads, headVDim, 1]` | `state @ q` output |
| `_gdnGatedOutT` | `[1, ssmDInner]` | Final gated output |

### Pre-allocated FullAttention decode buffers

| Buffer | Shape | Purpose |
|---|---|---|
| `_attnDecodeQBuf` | `[1, numHeads * headDim]` | Deinterleaved Q |
| `_attnDecodeGBuf` | `[1, numHeads * headDim]` | Deinterleaved gate |
| `_attnDecodeOutBuf` | `[1, numHeads * headDim]` | Attention output |
| `_attnDecodeQkvBuf` | `[1, qFullDim + 2*kvDim]` | Reused fused QKV output |
| `_ffnDecodeGateUpBuf` | `[1, 2 * intermediateSize]` | Reused fused gate+up output |

## 10. Memory and KV cache strategy

- **FullAttention layers**: standard KV cache `[numKVHeads, maxSeqLen,
  headDim]` per layer. KV dtype configurable via `--kv-cache-dtype` (`f32`,
  `f16`, `q8_0`).
- **GatedDeltaNet layers**: `_convState[layer]` float array of size
  `(convKernel - 1) * qkvDim` for the conv1d sliding window, and
  `_deltaStateTensor[layer]` of shape `[numVHeads, headVDim, headKDim]` for
  the SSM hidden state.
- `ResetKVCache()` zeroes both kinds of cache.
- Initial CUDA cache allocation can be smaller than `maxContextLength` and
  grows on demand (printed at startup).

### File-mapped quantized weights

When the backend supports it (direct CUDA, GGML CUDA, Apple Silicon Metal,
GGML CPU, integrated GPUs), the loader avoids copying quantized tensors into
a fresh native heap buffer and instead binds the GGUF file directly via
`MemoryMappedFile` + `QuantizedWeight.CreateExternalView`. Combined with
GGML's host-pointer buffer mapping this lets Metal command buffers read the
weights straight from the OS page cache. The peak resident set for
`Qwen3.5-35B-A3B-IQ2_XXS` (~10 GB GGUF) drops from ~17 GB to ~7 GB on
M-series Macs without any per-token copy.

## 11. Batched / paged forward (continuous batching)

Qwen 3.5 / 3.6 implements `IBatchedPagedModel.ForwardBatch`
([`Qwen35Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.BatchedForward.cs))
and runs it by default — the batched paged path supports every Qwen3.5
layer type (attention, GDN recurrent, MoE) and is what continuous-batching
multi-request workloads need to serve concurrent sequences in parallel.
Set `TS_QWEN35_BATCHED=0` (or pass `--no-continuous-batching` to the
server) to force the legacy per-sequence KV-swap fallback for A/B
comparison or regression isolation.

Qwen 3.5/3.6 is hybrid (FullAttention + GatedDeltaNet recurrent layers),
so the batched port has to manage **two orthogonal kinds of cache** —
paged K/V for attention layers, and a separate **per-slot recurrent
state pool** for GatedDeltaNet layers:

### FullAttention layers — paged K/V

- Per-layer paged buffer of layout
  `[numBlocks * blockSize * numKvHeads * headDim]`, lazily grown.
- Per-token NeoX RoPE via `Ops.RoPEEx` with an explicit positions array.
- Per-head QK RMSNorm performed in-place via `View` (same as legacy).
- Q + sigmoid-gate are deinterleaved out of the fused
  `[2 × qDim ‖ kDim ‖ vDim]` Q projection — same layout as the legacy
  forward, walked per-token in the batched loop.
- K/V scatter via `slotMapping` into the layer's paged buffer.
- Attention via `GgmlBasicOps.PagedAttentionForward` (the native paged
  kernel that drives `ggml_flash_attn_ext`), per sequence.

### GatedDeltaNet layers — per-slot recurrent state pool

The recurrent state is per-sequence and evolves across the entire input.
Phase 5c introduced a per-slot state pool keyed on each sequence's
**primary block id** (matching vLLM's `state_indices_tensor`):

- `_q35GdnSlotConvBuf[layer][slot]` — per-slot conv1d ring buffer
  (`float[(convKernel - 1) * qkvDim]`).
- `_q35GdnSlotConvWriteIdx[layer][slot]` — per-slot ring write head.
- `_q35GdnSlotSsmTensor[layer][slot]` — per-slot SSM state Tensor
  shape `[numVHeads, headVDim, headKDim]`.
- `_q35GdnSlotInit[layer][slot]` — initialization flag.

Slots are allocated lazily on first touch and freed when the engine
retires the sequence. Compared to the Phase-2 approach that copied state
in / out of the per-model scratch buffer twice per layer, this avoids
roughly **2 MB of SSM-tensor memcpy plus tens-of-KB conv memcpy per GDN
layer per sequence** on every decode step.

A **native batched GatedDeltaNet kernel** —
`TSGgml_GatedDeltaNetBatchedStepF32`
([`ggml_ops_gated_delta_net.cpp`](../../TensorSharp.GGML.Native/ggml_ops_gated_delta_net.cpp))
— is gated behind `TS_QWEN35_BATCHED_GDN_NATIVE=1`. When enabled,
`GgmlBasicOps.GatedDeltaNetBatchedStep` replaces the managed per-token
GDN step with one native dispatch that updates all in-flight sequences'
conv + SSM state in parallel; when disabled (default), the batched
forward calls the managed reference path through
[`Qwen35Model.GatedDeltaNet.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.GatedDeltaNet.cs).

### Multimodal in the batched path

`SupportsBatchedMultimodal` returns true while the batched path is
active (i.e. unless `TS_QWEN35_BATCHED=0` is set). `ForwardBatch` builds
a global MRoPE position table per batch:
each sequence's MRoPE positions are fetched from the multimodal
injector, globally offset into the batched hidden tensor, and threaded
into the per-layer RoPE call. Vision embeddings are injected row-wise
before the per-layer loop, exactly as in the legacy forward.

### Verified correctness and throughput

- 100% greedy-match against legacy on short prompts
  ([`Qwen35BatchedCorrectnessTests`](../../InferenceWeb.Tests/Qwen35BatchedCorrectnessTests.cs)).
- **~1.83× tps at n=3** on Qwen 3.6-27B (Apple M4 Pro, GgmlMetal,
  legacy-vs-batched in-process toggle).

## 12. MTP / NextN speculative decoding (Qwen 3.6)

Qwen 3.6 GGUFs ship a **NextN / multi-token-prediction (MTP) draft block** that
`TensorSharp.Server` uses for lossless speculative decoding on solo (non-concurrent)
sequences. Source:
[`Qwen35Model.Mtp.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.Mtp.cs),
driven through the shared
[`MtpSpeculativeExecution`](../../TensorSharp.Runtime/Scheduling/MtpSpeculativeExecution.cs)
draft / verify / rollback core. Unlike Gemma 4, no separate draft GGUF is needed —
the block is embedded in the trunk file, so `--mtp-draft-model` is ignored.

### 12.1 Embedded NextN block

A Qwen 3.6 GGUF carries one extra decoder block past the main stack (`blk.N`,
where `N` == trunk layer count), flagged by `{arch}.nextn_predict_layers`. It is a
standard full-attention Qwen 3.5 decoder block (dense FFN on 27B, MoE FFN on
35B-A3B) plus four NextN-specific tensors:

```
nextn.eh_proj          [2*hidden, hidden]  input projection over concat(embedding, hidden)
nextn.enorm            [hidden]            RMS norm over the token embedding
nextn.hnorm            [hidden]            RMS norm over the trunk hidden state
nextn.shared_head_norm [hidden]            final norm before the LM head
```

(`nextn.embed_tokens` / `nextn.shared_head_head` are optional and absent in the
stock GGUFs; the code falls back to the trunk token embedding / LM head.) The MTP
step consumes (token `x_p`, trunk hidden `h_{p-1}`) at position `p` and produces
logits predicting `x_{p+1}` plus its own hidden state to chain further draft steps
— mirroring llama.cpp's `graph_mtp` and vLLM's `Qwen3_5MultiTokenPredictor`.

### 12.2 Recurrent-state rollback

Because the Qwen 3.6 trunk mixes GatedDeltaNet recurrent layers with full-attention
layers, a partially-rejected verify batch cannot simply truncate the KV cache: the
GDN recurrent state can't be rewound in place. The MTP path snapshots the GDN state
(`_mtpGdnSnapshot`) before each verify batch and restores it on partial acceptance.
On the CUDA backend the snapshot is taken device-side to avoid host round-trips.

### 12.3 Profitability and tuning

Speculation is off by default; enable with `--mtp-spec` (env `TS_MTP_SPEC=1`). It
engages where it pays off — ggml backends and the pure-C# `cuda` backend — and
otherwise the engine serves standard decode. `--mtp-draft` (default `8`) bounds the
draft window and `--mtp-pmin` (default `0.75`) is the minimum draft confidence to
keep a token. On `ggml_cuda`, the GDN chunked-prefill kernel also speeds the
speculative verify: measured on Qwen3.6-27B IQ2_XXS it cut MTP speculative-verify
decode from 217 to 174 ms/token (see §8, `GDN_CHUNK_PREFILL_MIN_SEQ_LEN`). See the
[README MTP section](../../README.md#mtp--nextn-speculative-decoding) for the full
flag list.

## 13. Output parser and chat template

- `Qwen35OutputParser` inherits `Qwen3OutputParser`, so the wire format is
  identical: `<think> ... </think>` for chain-of-thought reasoning, and
  `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` for tool calls.
- Chat template uses the standard Qwen `<|im_start|>` / `<|im_end|>` rolling
  message format with the additional vision token `<|image_pad|>`.
- `ChatTemplate.ExpandImageTokens(inputTokens, imagePadId, tokenCounts)`
  expands each `<|image_pad|>` placeholder into the right number of placeholder
  tokens for the corresponding image, and the multimodal injector then writes
  the encoded embeddings into those positions before `Forward()`.

## 14. Optimization opportunities

- **Native GDN decode (legacy path)** — the legacy per-seq GDN decode
  still runs in managed C# (with pre-allocated buffers and
  `Ops.AddmmBatch`). Moving the per-token recurrent update into native
  C / CUDA on that path would remove the remaining managed overhead.
- **Vectorized conv1d** — `Conv1dStep` is a scalar loop. A SIMD or native
  vectorized version would shave a few percent off the decode hot path.
- **MoE prefill batching** — MoE prefill currently iterates per token. A
  batched expert prefill kernel (analogous to the decode path) would
  speed up long prompts on MoE variants.
- **Promote native batched GDN out of opt-in.** The
  `TS_QWEN35_BATCHED_GDN_NATIVE=1` kernel exists today but is gated on
  perf verification. Once the n=1 regression is closed it becomes the
  default GDN dispatch in the batched path.
