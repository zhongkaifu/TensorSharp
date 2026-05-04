# Gemma 4

[← back to model index](README.md)

| Property | Value |
|---|---|
| Provider | Google |
| GGUF architecture key | `gemma4` |
| Source class | [`Gemma4Model`](../../TensorSharp.Models/Models/Gemma4/Gemma4Model.cs) |
| Vision encoder | [`Gemma4VisionEncoder`](../../TensorSharp.Models/Models/Gemma4/Gemma4VisionEncoder.cs) (SigLIP-style ViT) |
| Audio encoder | [`Gemma4AudioEncoder`](../../TensorSharp.Models/Models/Gemma4/Gemma4AudioEncoder.cs) (USM-style chunked transformer) |
| Audio frontend | [`Gemma4AudioPreprocessor`](../../TensorSharp.Models/Models/Gemma4/Gemma4AudioPreprocessor.cs) (16 kHz mono → 128-bin log-mel) |
| Image processor | [`Gemma4ImageProcessor`](../../TensorSharp.Models/Models/Gemma4/Gemma4ImageProcessor.cs) |
| Example models | gemma-4-E4B (8B effective), gemma-4-31B, gemma-4-26B-A4B (MoE) |
| Modalities | Text, image, video (frame stack), audio |
| Thinking mode | Yes (`<\|channel>thought ... <channel\|>`) |
| Tool calling | Yes (`<\|tool_call>call:name{...}<tool_call\|>`) |
| Output parser | `Gemma4OutputParser` |

## 1. Origin and intent

Gemma 4 is the most feature-rich architecture currently supported by
TensorSharp. It pushes Google's Gemma line into:

- **Per-layer heterogeneity**: each layer can independently choose its
  attention pattern (SWA / global), its head dimension, its KV head count, and
  whether it shares KVs with an earlier "donor" layer.
- **Per-Layer Embedding (PLE)**: a per-layer side embedding pulled from a
  small `per_layer_token_embd.weight` table is mixed into the residual stream
  inside every block.
- **Mixture-of-Experts variants**: e.g. `gemma-4-26B-A4B`, where every block
  runs a dense MLP and a sparse MoE branch in parallel and adds them after
  separate post-norms.
- **True multimodality**: vision, video frame stacks, and audio all flow into
  the same residual stream.

Compared to Gemma 3, the SWA mask and RoPE caches are precomputed across all
layers in a forward pass, the SWA cache is a circular buffer (so memory is
bounded regardless of context length), and the entire transformer can be
executed as a single fused GGML graph during decode.

## 2. Model architecture

```
                              tokens (int[])
                                   │
                              token_embd.weight × sqrt(hidden)
                                   │
                       [optional]  InjectVisionEmbeddings (image / video frames)
                       [optional]  InjectVisionEmbeddings (audio embeddings; same op)
                                   │
                       [if PLE]    ComputePLE(tokens, hidden) ──► perLayerInputs
                                   │
              ┌────── × NumLayers ─────────────────────────────────────┐
              │ HeadDim, KVHeads, attn pattern (SWA/global) per layer  │
              │  RMSNorm(attn_norm)                                     │
              │  QKV (fused or Q-only on KV-shared layers)              │
              │  per-head RMSNorm(Q,K) + unweighted RMSNorm(V)          │
              │  RoPE (NeoX local OR global+proportional/partial)       │
              │  Attention (SWA window OR full causal)                  │
              │  attn_output ─► RMSNorm(post_attn_norm) + residual      │
              │                                                         │
              │  if MoE layer:                                          │
              │     RMSNorm(ffn_norm)  ─► GeGLU(dense MLP) ─► PostNorm1 │
              │     RMSNorm(pre_ffw_norm_2) ─► MoE(route+experts)       │
              │                              ─► PostNorm2               │
              │     residual += PostNorm(PostNorm1 + PostNorm2)         │
              │  else:                                                  │
              │     RMSNorm(ffn_norm) ─► GeGLU ─► RMSNorm(post_ffw)     │
              │     residual += branch                                  │
              │                                                         │
              │  if PLE:                                                │
              │     residual += proj(GELU(inp_gate(hidden)) * pleInput) │
              │                                                         │
              │  hidden *= layer_output_scale[layer]                    │
              └─────────────────────────────────────────────────────────┘
                                   │
                              RMSNorm(output_norm)
                                   │
                              LM head (output.weight or tied)
                                   │
                              [optional] tanh-softcap
                                   │
                                   ▼
                                logits
```

## 3. Forward graph

### 3.1 Per-layer (dense)

```
hidden ─► RMSNorm(attn_norm)
       ─► QKV matmul (fused weight) ─► split into Q [seq, qDim],
                                              K [seq, kvDim],
                                              V [seq, kvDim]
       ─► per-head RMSNorm(Q) * attn_q_norm.weight
       ─► per-head RMSNorm(K) * attn_k_norm.weight
       ─► unweighted RMSNorm(V)                       // V-norm: weight ≡ 1
       ─► RoPE(Q, K, freqs[layer])
            • local layers: standard NeoX RoPE on full headDim
            • global layers: NeoX RoPE on the first _partialRotaryDims dims
              with proportional frequency factors from rope_freqs.weight
       ─► Q ← Q * (1/sqrt(headDim))
       ─► append (K, V) to per-layer cache (circular for SWA)
       ─► attention(Q, K_cache, V_cache,
                    window = slidingWindow if SWA else totalSeq)
       ─► attn_output matmul → o
       ─► RMSNorm(post_attn_norm)
       ─► residual = hidden + o
       ─► h2 = RMSNorm(ffn_norm) on residual
       ─► ffn_gate_up matmul → [gate ‖ up]
       ─► g = GELU(gate)
       ─► h3 = ffn_down × (g * up)
       ─► RMSNorm(post_ffw_norm)
       ─► residual += h3
       ─► (PLE branch — see § 4.4)
       ─► hidden = residual * layer_output_scale[layer]
```

### 3.2 Per-layer (MoE)

```
... attention block as above ...

# Dense MLP branch
b1 = RMSNorm(ffn_norm) on residual
b1 = GeGLU(b1) using ffn_gate_up.weight + ffn_down.weight
b1 = RMSNorm(post_ffw_norm_1) on b1

# MoE branch
b2 = RMSNorm(pre_ffw_norm_2) on residual
logits = ffn_gate_inp.weight × b2
logits = unweighted_RMSNorm(logits) * ffn_gate_inp.scale   # learned scale
weights, idx = topK(softmax(logits), _numExpertsUsed)
b2 = weighted_sum_experts(SwiGLU on b2 using ffn_gate_up_exps[idx] +
                          ffn_down_exps[idx])
b2 = RMSNorm(post_ffw_norm_2) on b2

# combine
combined = RMSNorm(post_ffw_norm)(b1 + b2)
residual += combined
```

### 3.3 Decode vs prefill

- **Decode** (`seqLen == 1`) on a GGML backend with all dense layers and all
  weights quantized: a single native call (`Gemma4ModelDecode`) processes the
  whole stack, including PLE, per-layer head dims, circular SWA caches, and
  per-layer scalars in one GPU graph dispatch.
- **Prefill** (`seqLen > 1`): each eligible dense layer is fused into a single
  per-layer GGML graph (`Gemma4LayerPrefill`). MoE / KV-shared / PLE layers
  fall back to the per-op managed path. Long prompts are chunked into
  `min(2 × slidingWindow, 2048)` to bound the score tensor for SWA layers.

## 4. Components in detail

### 4.1 Per-layer heterogeneity

- `_slidingWindowPattern[layer]` from
  `gemma4.attention.sliding_window_pattern`. `IsLocalLayer(layer)` returns
  this entry. `true` ⇒ SWA layer; `false` ⇒ full causal layer.
- `_localHeadDim` from `gemma4.attention.key_length_swa` (default 256),
  `_globalHeadDim` from `gemma4.attention.key_length` (default 512).
  `HeadDimForLayer(layer)` selects which one to use.
- `_numGlobalKVHeads` from `gemma4.attention.global_head_count_kv`. Local
  layers use `Config.NumKVHeads`. `KVHeadsForLayer(layer)` resolves the
  count.
- `DetectHeadDimsFromWeights()` rebinds the head dims when the GGUF metadata
  contradicts the actual attention weight shapes.

### 4.2 KV sharing

The last `_sharedKVLayers` layers (from
`gemma4.attention.shared_kv_layers`) reuse another layer's KV cache. The
mapping is built by `BuildKVDonorMap()`; shared layers project only Q and skip
K/V matmuls and cache writes. The shared cache implementation:

- `_kvCacheK[shared] = _kvCacheK[donor]` (alias).
- During chunked prefill, the donor's freshly-computed K and V are stashed in
  `_prefillSWAKV` so KV-shared SWA layers can attend to the *full* chunk's K/V
  rather than the partial rolling cache.

### 4.3 Per-Layer Embedding (PLE)

When `gemma4.embedding_length_per_layer_input > 0`:

```
perLayerEmbeddings = lookup(per_layer_token_embd.weight, tokens)
perLayerEmbeddings = RMSNorm(per_layer_proj_norm.weight) on perLayerEmbeddings
perLayerEmbeddings = perLayerEmbeddings × per_layer_model_proj.weight^T

# inside each layer:
ple = GELU(inp_gate.weight × hidden) * extract_layer_slice(perLayerInputs, l)
ple = proj.weight × ple
ple = RMSNorm(post_norm.weight) on ple
residual += ple
```

`ComputePLE()` runs the full PLE pipeline once per forward pass and returns
`[seqLen, NumLayers * pleDim]` so each layer can `Narrow` out its slice.

### 4.4 RoPE flavors

- **Local layers**: standard NeoX RoPE on the full headDim, base
  `_ropeLocalBase` (from `gemma4.rope.freq_base_swa` or
  `gemma4.rope.local.freq_base`, default 10000). Implemented as
  `ApplyNeoXRoPEDecode` / `ApplyNeoXRoPEPrefill` with vectorized cos/sin
  tables.
- **Global layers**: partial NeoX RoPE applied to the first
  `_partialRotaryDims` dims of the headDim (the rest passes through). The
  frequency vector is the standard NeoX schedule scaled by per-frequency
  factors loaded from `rope_freqs.weight`. The full cos/sin lookup table is
  cached across global layers in `_neoXRopeCos` / `_neoXRopeSin`, eliminating
  ~35M `MathF.Cos`/`MathF.Sin` calls per chunk.

### 4.5 V-norm

After the V projection, `ApplyUnweightedRMSNorm()` runs RMSNorm with an
all-ones weight tensor (`_onesForVNorm`) on each value vector. This is unique
to Gemma 4 in TensorSharp's matrix.

### 4.6 MoE

`HasMoE(layer)` is true when `blk.{L}.ffn_gate_inp.weight` exists. The MoE
branch:

1. RMSNorm(pre_ffw_norm_2) on residual.
2. Router matmul → unweighted RMSNorm → multiply by learned scale
   `ffn_gate_inp.scale`.
3. softmax → TopK(`_numExpertsUsed`).
4. For each selected expert: SwiGLU(`ffn_gate_up_exps.{e}.weight`,
   `ffn_down_exps.{e}.weight`) followed by weighted accumulation.
5. RMSNorm(post_ffw_norm_2).

The dense and MoE branches are then summed and passed through
`post_ffw_norm`.

### 4.7 Per-layer output scaling

`_layerScalars[layer]` is loaded from `blk.{L}.layer_output_scale.weight`
(scalar `[1]`). Each layer's hidden output is multiplied by this scalar before
returning to the next layer.

### 4.8 Logit softcap

Identical to Gemma 3: `tanh(logits / cap) * cap` when
`gemma4.final_logit_softcapping > 0`.

### 4.9 Vision pipeline (images and video frames)

`Gemma4VisionEncoder` is a SigLIP-style ViT with 2D position embeddings,
GELU-Tanh MLP, and a final linear projector to the LM hidden dim. Video frames
are extracted from MP4 inputs (default up to 8 frames at 1 fps via OpenCV /
SkiaSharp) and each frame is encoded independently; the resulting embeddings
are concatenated and injected at successive `<|image>` markers.

### 4.10 Audio pipeline

`Gemma4AudioPreprocessor` decodes WAV / MP3 / OGG, resamples to 16 kHz mono,
emits a 128-bin log-mel spectrogram in 10 ms hops, and pads to the encoder's
chunk size (12 frames × 12 chunks).

`Gemma4AudioEncoder` is a chunked-attention USM-style transformer with:

- Conv subsampling in the time dimension.
- Per-chunk causal attention with a 12-frame past context window.
- A `logit_cap = 50` clip on attention logits (the same idea as the LM logit
  softcap).
- A residual scaling factor of `0.5`.
- A final linear projector to the LM hidden dim.

The resulting audio embeddings flow into the same `InjectVisionEmbeddings`
path as image embeddings; from the language model's point of view, audio and
image tokens are interchangeable carriers of pre-computed embeddings.

## 5. Parameters and settings (GGUF metadata)

| Key | Type | Meaning |
|---|---|---|
| `gemma4.attention.sliding_window_pattern` | bool[] | Per-layer SWA pattern |
| `gemma4.attention.sliding_window` | uint32 | SWA window size (default 512) |
| `gemma4.attention.key_length` | uint32 | Global head dim (default 512) |
| `gemma4.attention.key_length_swa` | uint32 | Local head dim (default 256) |
| `gemma4.attention.global_head_count_kv` | uint32 | KV heads for global layers |
| `gemma4.attention.head_count_kv` | int32[] | Per-layer KV head counts |
| `gemma4.attention.shared_kv_layers` | uint32 | Number of KV-sharing tail layers |
| `gemma4.rope.dimension_count` | uint32 | Partial rotary dims |
| `gemma4.rope.partial_rotary_factor` | float32 | Fraction of head dim to rotate |
| `gemma4.rope.freq_base_swa` | float32 | Local RoPE base |
| `gemma4.embedding_length_per_layer_input` | uint32 | PLE dim (0 disables PLE) |
| `gemma4.expert_count` | uint32 | MoE expert count (0 ⇒ dense) |
| `gemma4.expert_used_count` | uint32 | TopK routed experts per token |
| `gemma4.final_logit_softcapping` | float32 | LM head softcap |

## 6. Weight naming convention

```
token_embd.weight
output_norm.weight
output.weight                              # (optional if tied to token_embd)

blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight                    # fused QKV (non-shared layers)
blk.{L}.attn_q.weight                      # Q-only (KV-shared layers)
blk.{L}.attn_q_norm.weight
blk.{L}.attn_k_norm.weight
blk.{L}.attn_output.weight
blk.{L}.post_attention_norm.weight         # or attn_post_norm.weight
blk.{L}.ffn_norm.weight
blk.{L}.ffn_gate_up.weight                 # fused gate+up
blk.{L}.ffn_down.weight
blk.{L}.post_ffw_norm.weight               # or ffn_post_norm.weight
blk.{L}.layer_output_scale.weight          # scalar [1]

# MoE only:
blk.{L}.ffn_gate_inp.weight                # router
blk.{L}.ffn_gate_inp.scale                 # learned router scale
blk.{L}.ffn_gate_up_exps.{E}.weight        # fused expert gate+up
blk.{L}.ffn_down_exps.{E}.weight           # expert down
blk.{L}.pre_ffw_norm_2.weight              # MoE input norm
blk.{L}.post_ffw_norm_1.weight             # dense MLP post-norm
blk.{L}.post_ffw_norm_2.weight             # MoE post-norm

# PLE only:
per_layer_token_embd.weight
per_layer_model_proj.weight
per_layer_proj_norm.weight
blk.{L}.inp_gate.weight                    # PLE gate
blk.{L}.proj.weight                        # PLE projection
blk.{L}.post_norm.weight                   # PLE post-norm

# Global RoPE:
rope_freqs.weight                          # proportional frequency factors
```

## 7. TensorSharp implementation walkthrough

Constructor (`Gemma4Model(string ggufPath, BackendType backend)`):

1. `ParseBaseConfig()` (general fields).
2. Reads SWA pattern and window, both head dims, both KV head counts, both
   RoPE bases, partial rotary dims, PLE dim, shared KV layer count, and the
   MoE expert counts. Sets `Config.UsesCircularKvCache` from the SWA pattern.
3. `BuildKVDonorMap()` — produces `_kvDonorMap[layer] → donorLayer`.
4. `ParseTokenizer()`, `LoadWeights()`.
5. `_hasTiedOutput` detection.
6. `DetectHeadDimsFromWeights()` — repairs head dims if metadata lies.
7. `LoadLayerScalars()` — loads `_layerScalars[NumLayers]`.
8. `FuseQKVWeights()`, `FuseGateUpWeights()`, `FuseExpertGateUpWeights()`.
9. `PrepareCudaQuantizedWeightsForInference()` — uploads / reorders quant
   blobs for the direct CUDA backend.
10. `PrecomputeRoPE()` — fills both local and global frequency tables.
11. `InitKVCache(maxSeqLen)` — per-layer cache with capacity
    `slidingWindow` for SWA layers, `maxSeqLen` for global layers; aliases
    shared layers to their donors.
12. `BuildGemma4DecodeArrays()` — packs per-layer pointers, types, dims, and
    the optional MoE / PLE flags into the `_decodeArrays` struct used by the
    fused single-graph decode kernel.

`Forward(int[] tokens)` orchestrates:

- Embedding lookup + scaling.
- Optional vision / audio injection (drains the pending lists, marking the
  injected positions in `exceptPositions` so the layer code skips RoPE / cache
  writes for those positions).
- Optional PLE computation.
- Either:
  - **Fused decode** (one native call, see § 9), or
  - **Per-layer C# loop** with optional `TryFusedLayerPrefill` per dense
    layer.
- Final RMSNorm, LM head, optional softcap, copy to `_logitsBuffer`.

`ForwardRefill(int[] tokens)` is the prefill-then-decode entry point. It
splits the prefix into `min(2 × slidingWindow, 2048)`-token chunks
(overridable via `TS_PREFILL_CHUNK`) and ends with `Forward([lastToken])`.
Multimodal prompts skip chunking because injection positions are absolute.

## 8. Prefill optimization

### Fused per-layer prefill (`Gemma4LayerPrefill`)

For each eligible layer (dense, non-shared KV, no PLE injection in the
current chunk, all weights quantized), `TryFusedLayerPrefill()` invokes a
single GGML graph that performs the entire transformer block in one dispatch:

1. RMSNorm(attn_norm)
2. Fused QKV matmul → split Q, K, V
3. Per-head QK RMSNorm, V unweighted RMSNorm
4. RoPE (local NeoX or global proportional)
5. Causal attention (SWA or full causal) with KV cache append
6. Output projection + RMSNorm(post_attn_norm) + residual
7. RMSNorm(ffn_norm) → ffn_gate_up matmul → GELU(gate)*up → ffn_down matmul
8. RMSNorm(post_ffw_norm) + residual + layer_output_scale

Layers with MoE, KV sharing, or active PLE injection fall back to the
standard per-op C# path. Setting `TS_FUSED_LAYER_PREFILL=0` disables the
fused path entirely (useful for debugging or A/B benchmarking).

### Chunked prefill

`prefillChunkSize = min(2048, max(2 × slidingWindow, 1024))`. Each chunk
processes through all layers before advancing, keeping the SWA score tensor
small. Override with `TS_PREFILL_CHUNK=N` for tuning.

### Cross-layer caches

Three caches re-used across layers within one forward pass:

- `_cachedSWAMaskWidths` — per-row sliding-window mask widths for the current
  `(queryLen, startPos)`; recomputed only when `seqLen` or `startPos` changes.
- `_neoXRopeCos` / `_neoXRopeSin` — global-layer NeoX RoPE cos/sin tables
  built once per forward and reused across all global layers.
- `_cachedRoPEPosQ` / `_cachedRoPEPosK` — local-layer RoPE position tensors
  cached across layers when `seqLen` and `startPos` match.

### SWA prev-window gather

Long prompts wrap the SWA cache mid-chunk, so a chunk-2 query near the start
needs the (W − 1) positions that the chunk just overwrote. Before any layer in
the new chunk runs, `PrepareSwaPrevWindowsForChunk(startPos, seqLen)` snapshots
the live SWA window; SWA layers then concatenate the snapshot in front of the
freshly-computed K/V before running attention.

### Donor SWA-K/V publishing

When KV-shared SWA layers exist, the donor layer publishes its freshly-
computed K/V into `_prefillSWAKV` so shared layers attend to the full chunk's
K/V instead of the partial rolling cache.

## 9. Decode optimization

### Fused single-graph decode (`Gemma4ModelDecode`)

When `_canUseFusedDecode` (all dense, all quantized) and `seqLen == 1`, the
decode path collapses to a single native call. `BuildGemma4DecodeArrays()`
packs the following per layer once at load time:

- Quantized weight metadata (`type`, `ne0`, `ne1`, raw byte pointers).
- Per-layer head dim, KV head count, RoPE base / kind.
- KV cache pointers (writable scratch on GPU).
- Layer scalar value.
- PLE flags / projection pointers.

`NativeGemma4ModelDecode()` then runs the entire stack — embedding lookup is
in C# but every layer (RMSNorm + QKV + QK-norm + V-norm + RoPE + circular
cache append + attention + output projection + post-attn norm + GeGLU FFN +
post-FFN norm + residual + layer scalar) executes in one GGML graph dispatch
on Metal / CUDA. This eliminates hundreds of CPU↔GPU round-trips per token.

`EnsureKvCacheHostSynchronized()` bridges between the fused and unfused
paths: when the next forward is unfused (e.g. mid-conversation prefill), the
host KV cache copy is updated from the device.

### Circular KV cache for SWA layers

`CopyToCacheCircular()` writes new K/V slots at `pos % cacheSize` for SWA
layers. `AttentionDecodeCircular()` traverses the circular buffer for read.
SWA layers therefore allocate `slidingWindow` slots regardless of context
length — the resident set is bounded.

## 10. Memory and KV cache strategy

- **SWA layers**: capacity `_slidingWindow` slots, circular write/read via
  `CopyToCacheCircular()` and `AttentionDecodeCircular()`.
- **Global layers**: capacity `maxSeqLen` slots, linear append/read via
  `CopyToCacheLinear()` and `AttentionDecodeWithWindow()`.
- **Shared layers**: alias the donor's `_kvCacheK[donor]` /
  `_kvCacheV[donor]` — no separate allocation.
- **Quantized weight binding**: zero-copy mmap on GGML CPU / Metal / CUDA (the
  GGUF file is `MemoryMappedFile` + `QuantizedWeight.CreateExternalView`).
  Direct CUDA uploads quantized blobs to device memory once and frees the host
  copy.

## 11. Output parser and chat template

`Gemma4OutputParser` understands two structural framings:

- **Thinking** — `<|channel>thought ... <channel|>` chain-of-thought, then
  the final answer.
- **Tool calling** — `<|tool_call>call:function_name{...args...}<tool_call|>`
  blocks, which `OutputParser` extracts into structured tool calls regardless
  of the surrounding content.

Chat template falls back to the hardcoded Gemma 4 template when the GGUF does
not ship a Jinja2 one.

## 12. Optimization opportunities

- **Fused MoE GPU kernel** for Gemma 4 — MoE layers currently disable both
  `Gemma4ModelDecode` and `Gemma4LayerPrefill`. A batched expert kernel
  similar to Qwen 3.5's `MoEExpertsSwiGLUResidual` would recover the speedup
  for the fused decode / prefill graphs. (The expert-batched FFN kernel
  described below already removes the sequential per-expert dispatches from
  the unfused path.)
- **Audio prefill on GPU** — the audio encoder currently dispatches its
  conv subsampling on CPU. Moving the conv stack onto Metal / CUDA would
  improve TTFT for long audio prompts.

### Completed

- **Expert-batched FFN (GEGLU)** — `MoEForward` previously ran a
  batched-by-expert loop that degenerated into `num_experts_used` single-row
  matmuls on the decode path. It now dispatches the entire layer through
  `GgmlBasicOps.MoEFFNPrefill(..., MoEActivation.GEGLUSplit)`, which issues
  two or three `ggml_mul_mat_id` ops (gate[+up] / down) plus a fused
  `ggml_geglu_split` activation and expert aggregation — a constant-bounded
  graph submission per MoE layer regardless of `seq_len` or
  `num_experts_used`. The kernel consumes the original 3D
  `ffn_gate_up_exps.weight` / `ffn_down_exps.weight` blocks as zero-cost
  stacked views (mmap-backed on Apple Silicon, single shared buffer on
  Windows / Linux), and the pre-existing `ffn_down_exps.scale` per-expert
  factor is folded into the routing weights in C# so the native kernel
  stays activation-agnostic. Falls back to the previous batched-by-expert
  C# path when a layer's stacked views are unavailable (e.g. F32-only
  tensors).
