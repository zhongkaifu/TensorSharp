# Model Architecture Cards — Developer Reference

[English](model_cards.md) | [中文](model_cards_cn.md)

TensorSharp supports six model architectures. This document is a developer reference for engineers who need to modify, optimize, or extend the model implementations.

All model classes live under `InferenceEngine/Models/<Name>/` and inherit from `ModelBase` (in `InferenceEngine/ModelBase.cs`). `ModelBase` provides shared primitives: GGUF loading, weight storage (`_weights` for F32, `_quantWeights` for quantized), KV cache helpers, embedding lookup, RMSNorm, linear forward, RoPE utilities, timing instrumentation, and the `Forward(int[] tokens) → float[]` interface.

---

## Gemma 3

| Property | Value |
|---|---|
| Source file | `InferenceEngine/Models/Gemma3/Gemma3Model.cs` |
| Provider | Google |
| GGUF architecture key | `gemma3` |
| Example models | gemma-3-4b, gemma-3-12b, gemma-3-27b |
| Modalities | Text, Image |
| Thinking mode | No |
| Tool calling | No |
| Output parser | `PassthroughOutputParser` |

### Architecture Overview

Gemma 3 is a dense transformer with alternating local/global attention.

- **Attention pattern**: `IsGlobalLayer(layer)` returns true when `(layer + 1) % 6 == 0`. Global layers use full causal attention; all other layers use sliding-window attention (SWA).
- **Head dimensions**: configurable `_attnKeyLen` (default 256) and `_attnValLen` (default 256), read from GGUF `gemma3.attention.key_length` / `value_length`.
- **RoPE**: NeoX-style with separate frequency bases. Local layers use `_ropeLocalBase` (from `gemma3.rope.local.freq_base`, default 10000); global layers use `_ropeGlobalBase` (from `gemma3.rope.freq_base`) scaled by `1 / _ropeScale`. For the 27B variant (`NumLayers == 34`), `_ropeScale` is hardcoded to 8.0. Frequencies are precomputed once in `PrecomputeRoPE()`.
- **Normalization**: four RMSNorm per block — `attn_norm`, `post_attention_norm`, `ffn_norm`, `post_ffw_norm`. Q and K also get per-head RMSNorm via `ApplyBatchRMSNorm()` or `RMSNormInPlace()`.
- **FFN**: GeGLU — `GELU(gate) * up` via `Ops.GELUMul`, then `down` projection. Uses fused `ffn_gate_up.weight`.
- **Embedding**: token embeddings scaled by `sqrt(hidden_size)` in `ScaleEmbedding()`.
- **Output**: tied to `token_embd.weight` when `output.weight` is absent.
- **Logit softcap**: `tanh(logits / cap) * cap`, applied when `_finalLogitSoftcap > 0`.
- **Vision**: `Gemma3VisionEncoder` loaded separately via `LoadVisionEncoder(mmProjPath)`. Vision embeddings are injected into the text hidden state at `<start_of_image>` positions via `InjectVisionEmbeddings()`.

### GGUF Metadata Keys

| Key | Type | Description |
|---|---|---|
| `gemma3.attention.sliding_window` | uint32 | SWA window size (default 1024) |
| `gemma3.attention.key_length` | uint32 | Key head dim (default 256) |
| `gemma3.attention.value_length` | uint32 | Value head dim (default 256) |
| `gemma3.rope.local.freq_base` | float32 | Local RoPE base (default 10000) |
| `gemma3.rope.freq_base` | float32 | Global RoPE base |
| `gemma3.final_logit_softcapping` | float32 | Logit softcap (0 = disabled) |

### Weight Naming Convention

```
token_embd.weight                         # Token embedding [vocab, hidden]
blk.{L}.attn_norm.weight                  # Pre-attention RMSNorm
blk.{L}.attn_q.weight                     # Q projection
blk.{L}.attn_k.weight                     # K projection
blk.{L}.attn_v.weight                     # V projection
blk.{L}.attn_q_norm.weight                # Per-head Q RMSNorm
blk.{L}.attn_k_norm.weight                # Per-head K RMSNorm
blk.{L}.attn_output.weight                # Output projection
blk.{L}.post_attention_norm.weight         # Post-attention RMSNorm
blk.{L}.ffn_norm.weight                   # Pre-FFN RMSNorm
blk.{L}.ffn_gate.weight  }                # Before fusion: separate gate/up
blk.{L}.ffn_up.weight    }
blk.{L}.ffn_gate_up.weight                # After fusion: concatenated [2*intermed, hidden]
blk.{L}.ffn_down.weight                   # FFN down projection
blk.{L}.post_ffw_norm.weight              # Post-FFN RMSNorm
output_norm.weight                        # Final RMSNorm
output.weight                             # LM head (optional if tied)
```

### Forward Pass (per token)

```
tokens → Embedding → ScaleEmbedding(sqrt(hidden)) → [InjectVision]
For each layer L:
  hidden → RMSNorm(attn_norm)
         → Q,K,V projections → QK-norm → RoPE(local or global) → scale Q
         → Attention(SWA or full causal) → O projection
         → RMSNorm(post_attention_norm) → residual add
         → RMSNorm(ffn_norm)
         → GateUp → GELU(gate)*up → Down
         → RMSNorm(post_ffw_norm) → residual add
hidden → RMSNorm(output_norm) → LM head → [softcap] → logits
```

### KV Cache

- Shape: `[numKVHeads, maxSeqLen, headDim]` per layer.
- Same capacity for all layers (both SWA and global). SWA bounding is done via `AttentionDecodeWithWindow()` which limits `attendStart..totalSeqLen`.
- `ResetKVCache()` fills all caches with 0 and calls `InvalidateTensorDeviceCache()` to sync GPU state.

### Optimization Opportunities

- No fused QKV yet — Q, K, V are separate projections. Fusing would halve attention dispatch.
- No fused GPU decode path — each operation dispatches independently. A single-graph approach (like Gemma 4) would significantly improve Metal/CUDA throughput.
- SWA layers could use circular KV cache (like Gemma 4) to bound memory.

---

## Gemma 4

| Property | Value |
|---|---|
| Source file | `InferenceEngine/Models/Gemma4/Gemma4Model.cs` |
| Provider | Google |
| GGUF architecture key | `gemma4` |
| Example models | gemma-4-E4B, gemma-4-31B, gemma-4-26B-A4B (MoE) |
| Modalities | Text, Image, Video, Audio |
| Thinking mode | Yes (`<\|channel>thought`) |
| Tool calling | Yes (`<\|tool_call>`) |
| Output parser | `Gemma4OutputParser` |

### Architecture Overview

Gemma 4 is the most feature-rich architecture in TensorSharp with per-layer heterogeneity.

- **Attention pattern**: `_slidingWindowPattern[]` boolean array from GGUF. `IsLocalLayer(layer)` checks this array. Local → SWA, non-local → full causal.
- **Per-layer head dimensions**: `_localHeadDim` (from `key_length_swa`, default 256) and `_globalHeadDim` (from `key_length`, default 512). `HeadDimForLayer(layer)` returns the appropriate one. `DetectHeadDimsFromWeights()` auto-corrects if GGUF metadata doesn't match actual weight shapes.
- **Per-layer KV head count**: `_numGlobalKVHeads` may differ from `Config.NumKVHeads` (local). `KVHeadsForLayer(layer)` returns the correct count.
- **KV sharing**: the last `_sharedKVLayers` layers reuse KV caches from earlier "donor" layers. `_kvDonorMap[layer]` → donor layer. Shared layers skip K/V projection and cache writes; they only project Q.
- **Per-Layer Embedding (PLE)**: when `_pleDim > 0`, a per-layer input vector is computed from `per_layer_token_embd.weight` + `per_layer_model_proj.weight`, normalized, and injected into each block via `GELU(inp_gate(hidden)) * pleInput → proj → post_norm → residual add`.
- **Per-layer output scaling**: `_layerScalars[layer]` from `blk.{L}.layer_output_scale.weight`. Applied after the residual add.
- **RoPE**: Local layers use standard NeoX RoPE with `_ropeLocalBase`. Global layers use partial rotary dims (`_partialRotaryDims`) and proportional frequency factors from `rope_freqs.weight`.
- **V norm**: unweighted RMSNorm applied to V projections via `ApplyUnweightedRMSNorm()` using a ones tensor.
- **MoE**: some variants have MoE layers. `HasMoE(layer)` checks for `ffn_gate_inp.weight`. MoE layers run dense MLP and sparse MoE in parallel, then combine: `PostNorm1(MLP) + PostNorm2(MoE) → PostNorm → residual`. Routing uses unweighted RMSNorm + learned scale + linear → softmax → TopK.
- **Logit softcap**: same as Gemma 3.
- **Vision/Audio**: `Gemma4VisionEncoder` for images/video frames; `Gemma4AudioEncoder` for audio via mel-spectrogram.

### GGUF Metadata Keys

| Key | Type | Description |
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
| `gemma4.embedding_length_per_layer_input` | uint32 | PLE dimension (0 = disabled) |
| `gemma4.expert_count` | uint32 | MoE expert count (0 = dense) |
| `gemma4.expert_used_count` | uint32 | TopK experts per token |

### Weight Naming Convention

```
blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight                   # Fused QKV (non-shared layers)
blk.{L}.attn_q.weight                     # Q-only (shared KV layers)
blk.{L}.attn_q_norm.weight / attn_k_norm.weight
blk.{L}.attn_output.weight
blk.{L}.post_attention_norm.weight         # (or attn_post_norm.weight)
blk.{L}.ffn_norm.weight
blk.{L}.ffn_gate_up.weight                # Fused gate+up
blk.{L}.ffn_down.weight
blk.{L}.post_ffw_norm.weight               # (or ffn_post_norm.weight)
blk.{L}.layer_output_scale.weight          # Scalar [1]
blk.{L}.inp_gate.weight                   # PLE gate
blk.{L}.proj.weight                       # PLE projection
blk.{L}.post_norm.weight                  # PLE post-norm
# MoE layers add:
blk.{L}.ffn_gate_inp.weight               # Router
blk.{L}.ffn_gate_inp.scale                # Router learned scale
blk.{L}.ffn_gate_up_exps.{E}.weight       # Fused expert gate+up
blk.{L}.ffn_down_exps.{E}.weight          # Expert down
blk.{L}.pre_ffw_norm_2.weight             # MoE input norm
blk.{L}.post_ffw_norm_1.weight            # Dense MLP post-norm
blk.{L}.post_ffw_norm_2.weight            # MoE post-norm
rope_freqs.weight                         # Proportional RoPE frequency factors
per_layer_token_embd.weight               # PLE token embedding
per_layer_model_proj.weight               # PLE hidden→PLE projection
per_layer_proj_norm.weight                # PLE projection norm
```

### Forward Pass (per token)

```
tokens → Embedding → ScaleEmbedding → [InjectVision/Audio]
ComputePLE(tokens, hidden) → perLayerInputs
For each layer L:
  hidden → RMSNorm(attn_norm)
         → QKV (or Q-only if shared) → QK-norm → V-norm(unweighted)
         → RoPE(local or global+proportional) → Attention → O projection
         → RMSNorm(post_attn_norm) → residual add
  If MoE:
         → RMSNorm(ffn_norm) → GeGLU(dense) → PostNorm1
         → RMSNorm(pre_ffw_norm_2) → MoE(route+experts) → PostNorm2
         → PostNorm1 + PostNorm2 → PostNorm → residual add
  Else:
         → RMSNorm(ffn_norm) → GeGLU → RMSNorm(post_ffw) → residual add
  If PLE:
         → GELU(inp_gate(hidden)) * perLayerInput → proj → RMSNorm → residual add
  hidden *= layerScalar
hidden → RMSNorm(output_norm) → LM head → [softcap] → logits
```

### KV Cache

- **Per-layer capacity**: SWA layers get `_slidingWindow` slots; global layers get `maxSeqLen`.
- **Circular cache for SWA**: `CopyToCacheCircular()` and `AttentionDecodeCircular()` use `pos % cacheSize`.
- **Shared layers**: `_kvCacheK[sharedLayer]` points to `_kvCacheK[donorLayer]` — no separate allocation.

### Fused GPU Decode

Enabled automatically when all layers are dense (no MoE) and all weights are quantized. `BuildGemma4DecodeArrays()` packs per-layer pointers, types, and dimensions into arrays. `NativeGemma4ModelDecode()` calls `GgmlBasicOps.Gemma4ModelDecode()` — a single native call that processes all layers, PLE, heterogeneous head dims, circular KV cache, and layer scalars in one GPU dispatch.

### Optimization Opportunities

- MoE layers disable the fused decode path. A fused MoE GPU kernel would recover the speedup.
- Expert FFN runs sequentially per-token per-expert. Batching across experts would improve throughput.

---

## Qwen 3

| Property | Value |
|---|---|
| Source file | `InferenceEngine/Models/Qwen3/Qwen3Model.cs` |
| Provider | Alibaba |
| GGUF architecture key | `qwen3` |
| Example models | Qwen3-4B, Qwen3-8B, Qwen3-14B, Qwen3-32B |
| Modalities | Text only |
| Thinking mode | Yes (`<think>`) |
| Tool calling | Yes (`<tool_call>`) |
| Output parser | `Qwen3OutputParser` |

### Architecture Overview

Qwen 3 is the simplest dense transformer in TensorSharp, serving as a reference implementation.

- **Attention**: GQA with `Config.NumKVHeads < Config.NumHeads`. Fused QKV projection into a single `attn_qkv.weight`.
- **QK-norm**: per-head RMSNorm on Q and K.
- **RoPE**: NeoX-style. Frequencies precomputed in `_ropeFreqs[]`. Decode path uses a hand-optimized C# loop with stackalloc cos/sin tables. Prefill path uses `Ops.RoPEEx`.
- **FFN**: SwiGLU — `SiLU(gate) * up` via `Ops.SiLUMul` (in the `FFN` base method), then `down`. Uses fused `ffn_gate_up.weight`.
- **Normalization**: two RMSNorm per block — `attn_norm` and `ffn_norm`.

### GGUF Metadata Keys

Standard keys only: `qwen3.attention.head_count_kv`, `qwen3.rope.freq_base`, etc.

### Weight Naming Convention

```
blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight                   # Fused Q+K+V [qDim+kDim+kDim, hidden]
blk.{L}.attn_q_norm.weight / attn_k_norm.weight
blk.{L}.attn_output.weight
blk.{L}.ffn_norm.weight
blk.{L}.ffn_gate_up.weight                # Fused gate+up [2*intermed, hidden]
blk.{L}.ffn_down.weight
output_norm.weight
output.weight
```

### Forward Pass (per token)

```
tokens → Embedding
If decode (seqLen==1) and GGML backend:
  → NativeTransformerModelDecode (all layers in one native call)
Else:
  For each layer L:
    hidden → RMSNorm(attn_norm)
           → QKV(fused) → split Q,K,V → QK-norm → RoPE → Attention → O projection
           → residual add
           → RMSNorm(ffn_norm)
           → GateUp → SiLU(gate)*up → Down
           → residual add
hidden → RMSNorm(output_norm) → narrow to last token → LM head → logits
```

### Native Decode Paths

**Per-layer** (`NativeTransformerLayerDecode`): calls `GgmlBasicOps.TransformerLayerDecode()` with raw pointers to norm weights, QKV, Q/K norms, output, FFN norm, gate+up, down, KV caches. Processes one layer in native code.

**Whole-model** (`NativeTransformerModelDecode`): calls `GgmlBasicOps.TransformerModelDecode()` with arrays of per-layer pointers. Processes ALL layers in a single native call. Built by `BuildModelDecodeArrays()` which pre-resolves all weight pointers at model load time. The `_modelDecodeArrays` struct caches quantized weight metadata (type, ne0, ne1, rawBytes) so no dictionary lookups occur at decode time.

### Optimization Opportunities

- The native whole-model path is the fastest available. Further optimization would involve a GPU-fused path similar to Gemma 4.
- Prefill still uses the C# managed path. A native prefill path would improve prompt processing.

---

## Qwen 3.5

| Property | Value |
|---|---|
| Source file | `InferenceEngine/Models/Qwen35/Qwen35Model.cs` |
| Provider | Alibaba |
| GGUF architecture key | `qwen35`, `qwen35moe`, `qwen3next` |
| Example models | Qwen3.5-9B, Qwen3.5-32B |
| Modalities | Text, Image |
| Thinking mode | Yes (`<think>`) |
| Tool calling | Yes (`<tool_call>`) |
| Output parser | `Qwen35OutputParser` (inherits `Qwen3OutputParser`) |

### Architecture Overview

Qwen 3.5 is a hybrid model mixing full-attention and GatedDeltaNet recurrent layers.

- **Layer type**: `_isRecurrent[layer]` is true when `(layer + 1) % _fullAttentionInterval != 0` (default interval 4). For 48 layers with interval 4: 36 recurrent + 12 attention.
- **Full attention layers**: same as Qwen 3 but with **gated Q** — the Q projection outputs `2 * numHeads * headDim`. The output is deinterleaved: first half is Q, second half is a sigmoid gate. Attention output is multiplied by `sigmoid(gate)` via `Ops.SigmoidMul`.
- **GatedDeltaNet recurrent layers**: a linear SSM with:
  - **Projections**: `attn_qkv.weight` (Q+K+V), `attn_gate.weight` (z), `ssm_beta.weight`, `ssm_alpha.weight`.
  - **Conv1d**: kernel size from GGUF, applied over a sliding window of QKV states. State stored in `_convState[layer][]`.
  - **SiLU activation** on conv output.
  - **Head expansion**: if `_numKHeads != _numVHeads`, Q and K are repeated per-head.
  - **L2 normalization** on Q and K per head.
  - **Recurrent state update**: `state = exp(gate) * state + sigmoid(beta) * (v - state@k) ⊗ k^T`. Implemented via `Ops.AddmmBatch` for batched rank-1 outer products.
  - **Output**: `SiLU(z) * RMSNorm(state @ q)` via `Ops.SiLUMul` and `Ops.RMSNorm`.
  - **State shape**: `[numVHeads, headVDim, headKDim]` per layer.
- **Prefill**: recurrent layers process tokens **sequentially** (loop over `seqLen`), while attention layers use batched prefill.
- **Vision**: `Qwen35VisionEncoder` with dynamic resolution. Vision embeddings injected at `<|image_pad|>` positions.

### GGUF Metadata Keys

| Key | Type | Description |
|---|---|---|
| `qwen35.ssm.inner_size` | uint32 | `headVDim * numVHeads` |
| `qwen35.ssm.state_size` | uint32 | `headKDim` |
| `qwen35.ssm.group_count` | uint32 | `numKHeads` |
| `qwen35.ssm.time_step_rank` | uint32 | `numVHeads` |
| `qwen35.ssm.conv_kernel` | uint32 | Conv1d kernel size |
| `qwen35.full_attention_interval` | uint32 | Every Nth layer is full attention (default 4) |
| `qwen35.rope.dimension_sections` | int32[] | MRoPE section boundaries |
| `qwen35.rope.dimension_count` | uint32 | RoPE dim count |

### Weight Naming Convention

```
# Attention layers (every Nth):
blk.{L}.attn_norm.weight
blk.{L}.attn_q.weight                     # [numHeads*headDim*2, hidden] (Q+gate interleaved)
blk.{L}.attn_k.weight / attn_v.weight
blk.{L}.attn_q_norm.weight / attn_k_norm.weight
blk.{L}.attn_output.weight
blk.{L}.post_attention_norm.weight
blk.{L}.ffn_gate_up.weight / ffn_down.weight

# Recurrent layers:
blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight                   # [qkDim*2 + vDim, hidden]
blk.{L}.attn_gate.weight                  # z gate [ssmDInner, hidden]
blk.{L}.ssm_beta.weight                   # [numVHeads, hidden]
blk.{L}.ssm_alpha.weight                  # [numVHeads, hidden]
blk.{L}.ssm_conv1d.weight                 # [qkvDim, convKernel]
blk.{L}.ssm_dt.bias                       # dt bias [numVHeads]
blk.{L}.ssm_a                             # a parameter [numVHeads]
blk.{L}.ssm_norm.weight                   # Output RMSNorm
blk.{L}.ssm_out.weight                    # Output projection
blk.{L}.post_attention_norm.weight
blk.{L}.ffn_gate_up.weight / ffn_down.weight
```

### Cache Architecture

- **Attention layers**: standard KV cache `[numKVHeads, maxSeqLen, headDim]`.
- **Recurrent layers**: `_convState[layer]` float array of size `(convKernel-1) * qkvDim` for the conv1d sliding window, and `_deltaStateTensor[layer]` tensor of shape `[numVHeads, headVDim, headKDim]` for the SSM state.
- `ResetKVCache()` zeroes both KV caches and recurrent states.

### Pre-allocated Buffers

To avoid per-step allocation in the hot GDN decode path, the following buffers are allocated once in `InitGDNBuffers()`:

| Buffer | Shape | Purpose |
|---|---|---|
| `_gdnConvOutT` | [1, qkvDim] | Conv1d output + SiLU |
| `_gdnKBuf` / `_gdnQBuf` | [numVHeads, headKDim] | Q/K after expansion + L2 norm |
| `_gdnVBuf` | [numVHeads, headVDim] | V split |
| `_gdnKvMemBuf` | [numVHeads, headVDim, 1] | state@k intermediate |
| `_gdnCoreOutBuf` | [numVHeads, headVDim, 1] | state@q output |
| `_gdnGatedOutT` | [1, ssmDInner] | Final gated output |

### Optimization Opportunities

- Recurrent prefill is sequential (one token at a time). Chunked parallel prefill (processing multiple tokens through the SSM simultaneously) would dramatically improve prompt throughput.
- No native decode path yet. Moving the GDN decode to native C/CUDA would avoid managed overhead.
- Conv1d is implemented as a scalar loop. A vectorized SIMD implementation would help.

---

## GPT OSS

| Property | Value |
|---|---|
| Source file | `InferenceEngine/Models/GptOss/GptOssModel.cs` |
| Provider | OpenAI |
| GGUF architecture key | `gptoss`, `gpt-oss` |
| Example models | gpt-oss-20b |
| Modalities | Text only |
| Thinking mode | Yes (Harmony: `<\|channel>analysis` / `<\|channel>final`) |
| Tool calling | No |
| Output parser | `HarmonyOutputParser` (`AlwaysRequired = true`) |

### Architecture Overview

GPT OSS is a Mixture-of-Experts transformer with several unique design choices.

- **MoE routing**: every layer has `_numExperts` experts (32) with TopK routing (`_numExpertsUsed` = 4). Routing is `linear(hidden) + bias → TopK → softmax(selected only)`. Note: unlike Gemma 4 which does softmax-then-TopK, GPT OSS does TopK-then-softmax (matches Ollama's implementation).
- **Attention sinks**: per-head learned bias `attn_sinks.weight` (shape `[numHeads]`). Incorporated into softmax as a virtual token: its value participates in both max-finding and exp-sum, effectively adding an always-available attention target. Implemented in `ApplySoftmaxWithSinks()` and `AttentionDecodeWithSinks()`.
- **Attention pattern**: even layers → SWA (window from `_slidingWindow`, default 128), odd layers → full causal. Note: unlike Gemma 3/4 which use a configurable pattern, GPT OSS uses a simple even/odd rule (not explicitly coded — all layers attend to `totalSeqLen` in the current implementation).
- **Activation**: SiLUAlphaLimit — `x * sigmoid(alpha * x) * (y + 1)`, where alpha=1.702, limit=7.0. Gate (x) clamped to `[-inf, limit]`, up (y) clamped to `[-limit, limit]`. Constants `SiluAlpha` and `SiluLimit` are hardcoded.
- **Bias everywhere**: all linear projections use `LinearForwardWithBias()` which adds bias after matmul. The bias is looked up from `_weights[biasName]` and added row-by-row.
- **RoPE**: NeoX-style with Yarn scaling. `Ops.RoPEEx` is called with `origCtxLen=Config.OriginalContextLength` (4096), `freqScale=1/RopeScale`, `beta_fast=32`, `beta_slow=1`.
- **Normalization**: two RMSNorm per block — `attn_norm` and `post_attention_norm`.
- **Prefill optimization**: only the **last** transformer layer narrows to the last token before MoE. All other layers process the full sequence. Controlled by `isLastLayer` parameter in `TransformerBlock()`.
- **Quantization**: expert weights may be MXFP4 (GGML type 39).
- **Tokenizer**: GPT-4o BPE pre-tokenizer with `\p{N}{1,3}` number grouping.
- **Chat template**: Harmony format. The parser (`HarmonyOutputParser`) is `AlwaysRequired` — it always runs even without explicit thinking/tool flags because the model always wraps output in channel tags.

### GGUF Metadata Keys

| Key | Type | Description |
|---|---|---|
| `gptoss.expert_count` | uint32 | Number of experts (32) |
| `gptoss.expert_used_count` | uint32 | TopK experts per token (4) |
| `gptoss.attention.sliding_window` | uint32 | SWA window size (128) |
| `gptoss.expert_feed_forward_length` | uint32 | Expert FFN dim |
| `gptoss.rope.scaling.original_context_length` | uint32 | Yarn original context (4096) |
| `tokenizer.ggml.pre` | string | Pre-tokenizer type (`gpt-4o`) |

### Weight Naming Convention

```
blk.{L}.attn_norm.weight
blk.{L}.attn_q.weight / attn_q.bias
blk.{L}.attn_k.weight / attn_k.bias
blk.{L}.attn_v.weight / attn_v.bias
blk.{L}.attn_output.weight / attn_output.bias
blk.{L}.attn_sinks.weight                  # [numHeads] attention sink biases
blk.{L}.post_attention_norm.weight
blk.{L}.ffn_gate_inp.weight / ffn_gate_inp.bias   # Router [numExperts, hidden]
blk.{L}.ffn_gate_up_exps.{E}.weight        # Fused expert gate+up
blk.{L}.ffn_gate_up_exps.{E}.bias          # Fused expert gate+up bias
blk.{L}.ffn_down_exps.{E}.weight           # Expert down
blk.{L}.ffn_down_exps.{E}.bias             # Expert down bias
output_norm.weight
output.weight
```

Note: the GGUF file stores biases in packed `[numExperts, biasDim]` tensors (e.g. `ffn_gate_exps.bias`). `SplitExpertBiases()` unpacks these into per-expert tensors at load time. `FuseExpertGateUpWeights()` then concatenates gate+up weights and their biases.

### Forward Pass (per token)

```
tokens → Embedding
For each layer L:
  hidden → RMSNorm(attn_norm)
         → Q+bias, K+bias, V+bias → RoPE(Yarn) → Attention+Sinks → O+bias
         → residual add
  moeInput = hidden (or narrow to last token if last layer and prefill)
  moeInput → RMSNorm(post_attention_norm)
           → Router(linear+bias) → TopK → softmax(selected)
           → For each selected expert: SiLUAlphaLimit(gate+bias, up+bias) → down+bias
           → weighted sum of expert outputs
           → residual add
hidden → RMSNorm(output_norm) → narrow to last token → LM head → logits
```

### Optimization Opportunities

- No fused QKV projection — Q, K, V are separate (each with bias). Fusing would reduce dispatches.
- No native decode path. The entire forward pass is managed C#.
- Expert FFN runs sequentially per token. Batching experts or using a sparse kernel would help.
- Attention sinks prevent the use of standard softmax implementations — the custom softmax with sinks is scalar C#. A SIMD or GPU-fused version would improve attention performance.

---

## Nemotron-H

| Property | Value |
|---|---|
| Source file | `InferenceEngine/Models/Nemotron/NemotronModel.cs` |
| Provider | NVIDIA |
| GGUF architecture key | `nemotron_h`, `nemotron_h_moe` |
| Example models | Nemotron-H-8B-Reasoning-128K, Nemotron-H-47B-Reasoning-128K |
| Modalities | Text only |
| Thinking mode | Yes (`<think>`) |
| Tool calling | Yes (`<tool_call>`) |
| Output parser | `Qwen3OutputParser` |

### Architecture Overview

Nemotron-H is a hybrid model that mixes three distinct layer types in a single stack: Mamba2 SSM layers, attention-only layers, and FFN-only layers (optionally with MoE). The per-layer type is determined by GGUF metadata arrays (`head_count_kv` and `feed_forward_length`).

- **Layer type classification**: for each layer, `head_count_kv[l]` and `feed_forward_length[l]` determine the type:
  - Mamba2: `head_count_kv == 0 AND feed_forward_length == 0`
  - Attention-only: `head_count_kv > 0 AND feed_forward_length == 0`
  - FFN-only: `feed_forward_length > 0`
- **Mamba2 SSM layers**: selective state space model with grouped heads. Input is projected via `ssm_in.weight` to produce z (gate), xBC (state input), and dt (time step). xBC passes through a conv1d sliding window, SiLU activation, then the SSM scan step computes `state = state * exp(-softplus(dt) * A) + B * x * dt`, output `y = state @ C`. Finally `SiLU(z) * GroupRMSNorm(y)` is projected via `ssm_out.weight`. The conv state and SSM state are maintained per-layer across tokens.
- **Attention layers**: standard GQA with no RoPE. Per-layer head counts and KV head counts are read from GGUF arrays. Fused QKV projection (`attn_qkv.weight`). Attention scale is read from GGUF (`attention.scale`) or defaults to `1/sqrt(headDim)`.
- **FFN layers**: use ReLU-squared activation (`max(0, x)^2`), implemented with SIMD vectorization. When MoE is enabled, FFN layers use sigmoid-based routing with optional expert bias (`exp_probs_b`), top-K selection, and optional weight normalization/scaling. Supports latent bottleneck projections (`ffn_latent_in` / `ffn_latent_out`) and shared experts (`ffn_up_shexp` / `ffn_down_shexp`).
- **MoE routing**: `sigmoid(logits) + bias → TopK → normalize`. Unlike softmax-based routers, Nemotron-H uses per-expert sigmoid probabilities with optional additive bias for expert selection. `expert_weights_norm` normalizes selected weights to sum to 1; `expert_weights_scale` applies a global scale factor.
- **Normalization**: single RMSNorm per block (`attn_norm`), two RMSNorm total per layer (pre-block + output).
- **Chat template**: uses the Qwen 3 chat template format (`<|im_start|>` / `<|im_end|>`).
- **Decode optimization**: for decode (seqLen=1), small operations (RMSNorm, residual add, small matmuls like expert and router) can be executed on CPU to avoid Metal GPU dispatch overhead (~1ms+ per dispatch). Large matmuls (SSM in/out, attention QKV/output, LM head) remain on GPU.

### GGUF Metadata Keys

| Key | Type | Description |
|---|---|---|
| `nemotron_h.ssm.conv_kernel` | uint32 | Mamba2 conv1d kernel size |
| `nemotron_h.ssm.inner_size` | uint32 | SSM inner dimension (nHead * headDim) |
| `nemotron_h.ssm.state_size` | uint32 | SSM state dimension per head |
| `nemotron_h.ssm.time_step_rank` | uint32 | Number of SSM heads |
| `nemotron_h.ssm.group_count` | uint32 | Number of SSM groups |
| `nemotron_h.attention.head_count_kv` | uint32[] | Per-layer KV head count (0 → Mamba2) |
| `nemotron_h.attention.head_count` | uint32[] | Per-layer Q head count |
| `nemotron_h.feed_forward_length` | uint32[] | Per-layer FFN size (0 → no FFN) |
| `nemotron_h.attention.scale` | float32 | Attention scale factor (0 = auto) |
| `nemotron_h.expert_count` | uint32 | MoE expert count (0 = dense) |
| `nemotron_h.expert_used_count` | uint32 | TopK experts per token |
| `nemotron_h.expert_weights_norm` | bool | Normalize expert weights to sum=1 |
| `nemotron_h.expert_weights_scale` | float32 | Global scale for expert weights |

### Weight Naming Convention

```
token_embd.weight
output_norm.weight
output.weight

# Mamba2 layers:
blk.{L}.attn_norm.weight                  # Pre-block RMSNorm
blk.{L}.ssm_in.weight                     # Input projection [hidden → 2*dInner + 2*nGroup*dState + nHead]
blk.{L}.ssm_conv1d.weight                 # Conv1d kernel [xBCSize, convKernel]
blk.{L}.ssm_conv1d.bias                   # Conv1d bias (optional)
blk.{L}.ssm_dt.bias                       # Time step bias [nHead]
blk.{L}.ssm_a                             # A parameter [nHead]
blk.{L}.ssm_d                             # D parameter [nHead] (optional)
blk.{L}.ssm_norm.weight                   # Group RMSNorm [dInner]
blk.{L}.ssm_out.weight                    # Output projection [dInner → hidden]

# Attention layers:
blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight                   # Fused Q+K+V (or separate attn_q/k/v.weight)
blk.{L}.attn_output.weight

# FFN layers (dense):
blk.{L}.attn_norm.weight
blk.{L}.ffn_up.weight                     # Up projection
blk.{L}.ffn_down.weight                   # Down projection

# FFN layers (MoE):
blk.{L}.attn_norm.weight
blk.{L}.ffn_gate_inp.weight               # Router [numExperts, hidden]
blk.{L}.exp_probs_b.bias                  # Router bias (optional) [numExperts]
blk.{L}.ffn_latent_in.weight              # Latent bottleneck in (optional)
blk.{L}.ffn_latent_out.weight             # Latent bottleneck out (optional)
blk.{L}.ffn_up_exps.{E}.weight            # Expert up projection
blk.{L}.ffn_down_exps.{E}.weight          # Expert down projection
blk.{L}.ffn_up_shexp.weight               # Shared expert up (optional)
blk.{L}.ffn_down_shexp.weight             # Shared expert down (optional)
```

### Forward Pass (per token)

```
tokens → Embedding
For each layer L:
  If Mamba2:
    hidden → RMSNorm(attn_norm)
           → ssm_in projection → split z, xBC, dt
           → Conv1d(xBC) → SiLU → SSM scan(dt, A, B, C, state) → y
           → SiLU(z) * GroupRMSNorm(y) → ssm_out projection
           → residual add
  If Attention:
    hidden → RMSNorm(attn_norm)
           → QKV(fused) → split Q,K,V → Attention(no RoPE) → O projection
           → residual add
  If FFN (dense):
    hidden → RMSNorm(attn_norm)
           → Up → ReLU² → Down
           → residual add
  If FFN (MoE):
    hidden → RMSNorm(attn_norm)
           → [latent_in] → Router(sigmoid+bias) → TopK → normalize
           → For each expert: Up → ReLU² → Down → weighted sum
           → [latent_out] → [+ shared_expert] → residual add
hidden → RMSNorm(output_norm) → narrow to last token → LM head → logits
```

### Cache Architecture

- **Attention layers**: standard KV cache `[numKVHeads, maxSeqLen, headDim]`.
- **Mamba2 layers**: `_convState[layer]` float array of size `(convKernel-1) * (dInner + 2*nGroup*dState)` for the conv1d sliding window, and `_ssmState[layer]` float array of size `dState * headDim * nHead` for the SSM state.
- `ResetKVCache()` zeroes both KV caches and SSM/conv states.
- `SupportsKVCacheTruncation` returns `false` because SSM states are sequential and cannot be partially reused.

### Pre-allocated Buffers

To avoid per-step allocation in hot paths, the following buffers are allocated once:

| Buffer | Size | Purpose |
|---|---|---|
| `_mamba2ConvOutBuf` | dInner + 2*nGroup*dState | Conv1d output + SiLU |
| `_mamba2YBuf` | dInner | SSM scan output |
| `_moeProbs` / `_moeSelectionProbs` | numExperts | Router probabilities |
| `_moeTopExperts` / `_moeRouteW` | numExpertsUsed | Selected experts and weights |
| `_moeLatentAccum` | max(hiddenSize, latentDim) | Latent-space accumulator |
| `_expertUpResult` / `_expertDownResult` | max expert dims | Reusable expert matmul outputs |
| `_latentAccumTensor` / `_latentOutResult` | latentDim / hiddenSize | Latent bottleneck reuse tensors |

### Batched GPU MoE

When running on a GGML backend (Metal/CUDA), MoE expert computation during decode is batched into a single `GgmlBasicOps.MoEExpertsForward()` call that processes all selected experts in one GPU graph dispatch, avoiding per-expert dispatch overhead. Pre-cached `QuantizedWeight` references (`_expertUpQW`, `_expertDownQW`) and pre-allocated pointer arrays (`_moeUpPtrs`, `_moeDownPtrs`) eliminate dictionary lookups and allocation in the hot loop.

### Optimization Opportunities

- No RoPE in attention — positions are implicitly tracked by the SSM state, which simplifies the attention implementation but prevents KV cache reuse across sessions.
- Mamba2 SSM processes tokens sequentially during prefill. Chunked parallel SSM scanning would improve prompt throughput.
- Conv1d is implemented as a scalar loop. A SIMD or native vectorized version would improve SSM layer performance.
- No native whole-model decode path. Moving the forward pass to a single native call (like Qwen 3) would reduce managed overhead.
- ReLU-squared activation is SIMD-vectorized but expert FFN still runs sequentially per token. Batching across experts would help.

---

## Architecture Comparison

| Feature | Gemma 3 | Gemma 4 | Qwen 3 | Qwen 3.5 | GPT OSS | Nemotron-H |
|---|---|---|---|---|---|---|
| Layer type | Dense | Dense / MoE | Dense | Hybrid (Attn + Recurrent) | MoE | Hybrid (Mamba2 + Attn + MoE FFN) |
| Attention | SWA + Global | SWA + Global | Full GQA | Full GQA + Gated | Full + Sinks | Full GQA (no RoPE) |
| FFN activation | GeGLU | GeGLU | SwiGLU | SwiGLU | SiLUAlphaLimit | ReLU² |
| RoPE variant | NeoX (dual base) | NeoX + proportional | NeoX | NeoX / MRoPE | NeoX + Yarn | None |
| QK norm | Yes | Yes | Yes | Yes | No | No |
| V norm | No | Yes (unweighted) | No | No | No | No |
| Bias in projections | No | No | No | No | Yes (all) | No |
| Per-layer scaling | No | Yes | No | No | No | No |
| PLE | No | Yes | No | No | No | No |
| KV sharing | No | Yes | No | No | No | No |
| Attention sinks | No | No | No | No | Yes | No |
| Circular KV cache | No | Yes (SWA layers) | No | No | No | No |
| SSM (Mamba2) | No | No | No | No | No | Yes |
| Shared experts | No | No | No | No | No | Yes (optional) |
| Latent bottleneck | No | No | No | No | No | Yes (optional) |
| Vision | Yes | Yes | No | Yes | No | No |
| Audio | No | Yes | No | No | No | No |
| Video | No | Yes | No | No | No | No |
| Thinking | No | Yes | Yes | Yes | Yes (always) | Yes |
| Tool calling | No | Yes | Yes | Yes | No | Yes |
| Fused QKV | No | Yes | Yes | No | No | Yes |
| Fused GPU decode | No | Yes (Metal) | No | No | No | No |
| Native model decode | No | No | Yes | No | No | No |
| Batched GPU MoE | No | No | No | No | No | Yes |
| Output parser | Passthrough | Gemma4 | Qwen3 | Qwen35 | Harmony (always on) | Qwen3 |

---

## Adding a New Model Architecture

1. Create `InferenceEngine/Models/<Name>/<Name>Model.cs` inheriting `ModelBase`.
2. In the constructor: read GGUF metadata via `_gguf.GetXxx()`, call `ParseBaseConfig()` and `ParseTokenizer()`, call `LoadWeights()`, then fuse weights and init caches.
3. Implement `Forward(int[] tokens) → float[]`: embedding → transformer blocks → norm → LM head → logits copy.
4. Implement `ResetKVCache()` and `Dispose()`.
5. Register in `ModelBase.Create()` switch expression (in `ModelBase.cs`).
6. Add an `IOutputParser` implementation in `OutputParser.cs` if the model uses a non-standard output format. Register in `OutputParserFactory.Create()`. Set `AlwaysRequired = true` if the model always wraps output in structural tags.
7. Add chat template support in `ChatTemplate.cs` / `Jinja2Template.cs` if the model uses a novel template format.
