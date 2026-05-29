# Nemotron-H

[← back to model index](README.md)

| Property | Value |
|---|---|
| Provider | NVIDIA |
| GGUF architecture keys | `nemotron_h`, `nemotron_h_moe` |
| Source class | [`NemotronModel`](../../TensorSharp.Models/Models/Nemotron/NemotronModel.cs) (legacy per-seq) + [`NemotronModel.BatchedForward.cs`](../../TensorSharp.Models/Models/Nemotron/NemotronModel.BatchedForward.cs) (`IBatchedPagedModel`) |
| Vision encoder | [`NemotronVisionEncoder`](../../TensorSharp.Models/Models/Nemotron/NemotronVisionEncoder.cs) (RADIO / v2_vl ViT) |
| Image processor | [`NemotronImageProcessor`](../../TensorSharp.Models/Models/Nemotron/NemotronImageProcessor.cs) |
| Audio frontend | [`NemotronAudioPreprocessor`](../../TensorSharp.Models/Models/Nemotron/NemotronAudioPreprocessor.cs) (Parakeet-style log-mel) |
| Example models | Nemotron-H-8B-Reasoning-128K, Nemotron-H-47B-Reasoning-128K, Nemotron 3 Nano Omni |
| Modalities | Text, image (Omni-class with `mmproj` loaded). Audio is preprocessed for Omni distributions but inference requires a Parakeet `mmproj` that is not shipped with these GGUFs. |
| Thinking mode | Yes (`<think> ... </think>`) |
| Tool calling | Yes (`<tool_call>{...}</tool_call>`) |
| Batched / paged forward | **Default ON** — set `TS_NEMOTRON_BATCHED=0` to force the legacy per-sequence KV-swap path for A/B comparison. Per-slot Mamba2 conv + SSM state pool, paged K/V for attention layers. Optional native batched Mamba2 step kernel (`TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1`). See §11. |
| Output parser | `Qwen3OutputParser` |

## 1. Origin and intent

Nemotron-H is NVIDIA's hybrid **Mamba2 + Transformer** family. The same
backbone covers the dense `nemotron_h` line (e.g. Nemotron-H-8B / 47B) and the
MoE `nemotron_h_moe` line. The Omni distribution (Nemotron 3 Nano Omni)
additionally ships a RADIO / v2_vl vision encoder and a Parakeet audio
frontend.

The defining traits are:

- **Three layer types in one stack** — Mamba2 SSM layers, attention-only
  layers, and FFN-only layers (optionally MoE). The per-layer type is
  determined by GGUF metadata arrays: `head_count_kv[l]` and
  `feed_forward_length[l]` together select the layer type.
- **Mamba2 SSM** — selective state-space model with grouped heads. The conv
  state and the SSM state are maintained per layer and per token.
- **No RoPE** — attention layers carry no positional encoding. Position
  information is implicit in the SSM state.
- **ReLU² FFN** activation (`max(0, x)²`).
- **Sigmoid-routed MoE** — uses per-expert sigmoid probabilities with
  optional additive bias for expert selection, top-K, and renormalization /
  scaling.
- **Optional latent bottleneck and shared experts** — MoE FFN can have an
  in / out latent bottleneck (`ffn_latent_in` / `ffn_latent_out`) and a
  shared SwiGLU branch (`ffn_up_shexp` / `ffn_down_shexp`) added in
  parallel.
- **Decode CPU offload threshold** — small decode ops (RMSNorm, residual
  add, small matmuls) can be run on CPU to dodge per-dispatch GPU overhead;
  large matmuls (SSM in/out, attention QKV/output, LM head) stay on GPU.

## 2. Model architecture

```
                tokens (int[])
                      │
              token_embd.weight
                      │
        [optional] InjectVisionEmbeddings (image)
                      │
        ┌──── × NumLayers ───────────────────────────────┐
        │ select layer type from (head_count_kv[l],      │
        │   feed_forward_length[l]):                     │
        │     • Mamba2:    kv == 0  AND  ff == 0         │
        │     • Attention: kv  > 0  AND  ff == 0         │
        │     • FFN:                       ff  > 0       │
        │                                                │
        │ Mamba2:                                        │
        │   RMSNorm(attn_norm) ─► ssm_in ─► z, xBC, dt  │
        │   conv1d(xBC) ─► SiLU                          │
        │   SSM scan(dt, A, B, C, state) ─► y            │
        │   SiLU(z) * GroupRMSNorm(y) ─► ssm_out         │
        │   residual += ssm_out                          │
        │                                                │
        │ Attention:                                     │
        │   RMSNorm(attn_norm) ─► QKV (fused)           │
        │   attention (no RoPE)                          │
        │   attn_output ─► residual                      │
        │                                                │
        │ FFN (dense):                                   │
        │   RMSNorm(attn_norm) ─► up ─► ReLU² ─► down   │
        │   residual += ffn_out                          │
        │                                                │
        │ FFN (MoE):                                     │
        │   RMSNorm(attn_norm)                           │
        │   [optional latent_in]                         │
        │   route(sigmoid(logits) + bias) ─► topK        │
        │   For each expert:                             │
        │       up ─► ReLU² ─► down ─► weighted Σ        │
        │   [optional latent_out]                        │
        │   [optional + shared_expert]                   │
        │   residual += moe_out                          │
        └────────────────────────────────────────────────┘
                      │
              RMSNorm(output_norm)
                      │
              LM head (output.weight)
                      │
                      ▼
                   logits
```

## 3. Forward graph

For each layer L, dispatch by `_layerTypes[l]`:

```
if Mamba2:
  hidden ─► RMSNorm(attn_norm)
         ─► ssm_in.weight matmul ─► split into z, xBC, dt
         ─► Conv1dStep(xBC) using ssm_conv1d.weight (state in _convState[l])
         ─► SiLU
         ─► SSM scan step:
                state ← state * exp(-softplus(dt) * A) + B * x * dt
                y     ← state @ C  (+ x * D if D present)
         ─► out = SiLU(z) * GroupRMSNorm(y, ssm_norm.weight)
         ─► out matmul ssm_out.weight → o
         ─► residual = hidden + o

if Attention:
  hidden ─► RMSNorm(attn_norm)
         ─► attn_qkv.weight matmul ─► split Q, K, V (per-layer head counts)
         ─► append (K, V) to KV cache
         ─► attention(Q, KCache, VCache, scale = _attentionScale or 1/sqrt(headDim))
         ─► attn_output.weight matmul → o
         ─► residual = hidden + o

if FFN (dense):
  hidden ─► RMSNorm(attn_norm)
         ─► ffn_up.weight matmul → x
         ─► ReluSquaredInPlace(x)            # x ← max(0, x)²
         ─► ffn_down.weight × x → o
         ─► residual = hidden + o

if FFN (MoE):
  hidden ─► RMSNorm(attn_norm)
         ─► [if HasLatentIn] ffn_latent_in × hidden → latent
         ─► logits = ffn_gate_inp × (hidden or latent)
         ─► probs  = sigmoid(logits) + exp_probs_b      # element-wise
         ─► weights, idx = topK(probs)
         ─► if expert_weights_norm: weights /= sum(weights)
         ─► weights *= expert_weights_scale
         ─► For each selected expert e:
                 up   = ffn_up_exps[e]   × (hidden or latent) → x
                 ReluSquaredInPlace(x)
                 down = ffn_down_exps[e] × x
                 accum += weights[e] * down
         ─► [if HasLatentOut] accum = ffn_latent_out × accum
         ─► [if HasSharedExperts] accum += SiLU(...)·... using ffn_up_shexp / ffn_down_shexp
         ─► residual = hidden + accum
```

After all layers:

```
hidden ─► narrow(seq_len-1) if prefill
       ─► RMSNorm(output_norm)
       ─► output.weight matmul → logits
       ─► copy to float[VocabSize]
```

## 4. Components in detail

### 4.1 Mamba2 SSM layer

The Mamba2 block is the most computationally specialized piece. The
`ssm_in.weight` projection produces a concatenation of three streams:

- **z** — gate, dim `_ssmDInner`.
- **xBC** — the joint `(x, B, C)` stream, dim `_ssmDInner + 2 * _ssmNGroup * _ssmDState`.
- **dt** — selective timestep, dim `_ssmNHead`.

`xBC` is then run through a conv1d sliding window of kernel size `_ssmDConv`
(state held in `_convState[l]`). After SiLU, the SSM scan step computes:

```
ΔA   = exp( -softplus(dt + ssm_dt.bias) * A )           # A ∈ ℝ^{nHead}
ΔB   = (B * x * dt)                                      # B ∈ ℝ^{nGroup × dState}
state = ΔA · state + ΔB                                  # state ∈ ℝ^{dState × headDim × nHead}
y    = state · C                                         # C ∈ ℝ^{nGroup × dState}
y   += x * ssm_d                                         # if `ssm_d` is present
```

`Mamba2SSMStepSIMD` carries this in pure SIMD-vectorized C# during decode
(one token at a time).  `GroupRMSNorm(y, ssm_norm.weight)` then normalizes
per-group before the gate `SiLU(z) * y`. Finally `ssm_out` projects back to
the hidden dim.

### 4.2 Attention layer

- Standard GQA. Per-layer head counts `_layerNumHeads[l]` and KV head counts
  `_layerNumKVHeads[l]` are read from GGUF arrays.
- Fused QKV when `attn_qkv.weight` exists; otherwise the constructor's
  `FuseQKVWeights()` builds it from separate `attn_q/k/v.weight` tensors.
- Attention scale is `nemotron_h.attention.scale` when set, else
  `1/sqrt(headDim)`.
- **No RoPE** — the SSM state implicitly carries position information, so
  attention does not need positional encoding.

### 4.3 FFN layer (dense)

`ReluSquaredInPlace(x)` applies `x ← max(0, x)²` SIMD-vectorized in place.
The dense FFN uses a single `ffn_up` matmul followed by ReLU² and
`ffn_down`.

### 4.4 FFN layer (MoE)

- **Routing**: `sigmoid(logits) + exp_probs_b` (per-expert additive bias) →
  TopK → optional renormalization → optional global scale.
- **Latent bottleneck** (optional): `ffn_latent_in` projects hidden to a
  smaller `latentDim`, the experts run in latent space, then
  `ffn_latent_out` projects back to hidden.
- **Shared experts** (optional): a SwiGLU branch using `ffn_up_shexp` and
  `ffn_down_shexp` added in parallel.
- **Batched MoE GPU dispatch**: on a GGML backend (Metal / CUDA), one
  `MoEExpertsForward` call processes all selected experts in a single GGML
  graph dispatch. Pre-cached `QuantizedWeight` references (`_expertUpQW`,
  `_expertDownQW`) and pre-allocated `IntPtr[]` arrays (`_moeUpPtrs`,
  `_moeDownPtrs`) avoid dictionary lookups and per-token allocations.

### 4.5 Vision encoder (`NemotronVisionEncoder`)

Mirrors NVIDIA's RADIO / CLIP-style ViT used by the v2_vl projector:

- **Linear patch embedding** with optional bias.
- **Position embedding** stored as a `[hidden, posTokens]` grid, bilinearly
  resized (align-corners = false) when the source patch grid differs.
  Resized embeddings are cached per `(gridW, gridH)` in
  `_resizedPositionEmbeddings`.
- **Class embedding** (optional) prepended to the sequence.
- **32 encoder blocks** (default for Nemotron Omni): LayerNorm → fused-QKV
  self-attention → residual → LayerNorm → up linear → GELU → down linear →
  residual.
- After the encoder, the class tokens are stripped and a pixel-shuffle by
  `scaleFactor` reduces the spatial resolution.
- A projector (RMSNorm → Linear → ReLU² → Linear) emits embeddings in the
  LM hidden dim.

`NemotronImageProcessor` (mirrors `nemotronh.ImageProcessor`):

1. Composite RGBA over white background.
2. Choose a tile grid that best matches the source aspect ratio (max
   `maxTiles`), or run dynamic-resolution mode if min/max patches metadata
   is present. Runtime inference caps tiled images to 1 tile by default to
   keep first-token latency practical for server image chats; set
   `TS_NEMOTRON_IMAGE_MAX_TILES=12` (or the model's advertised max) to restore
   full-resolution tiling.
3. Bicubic resize to `gridW * imageSize × gridH * imageSize` (or dynamic
   patch grid).
4. Crop into `imageSize × imageSize` tiles, channel-first `[C, H, W]`.
5. Optional thumbnail tile when more than one tile was produced.
6. Normalize each tile with the CLIP mean/std loaded from the mmproj.

The multimodal injector (`ProcessNemotronHistory` in
`ModelMultimodalInjector.cs`) tokenizes each image into one
`<image>` placeholder, expands it into `<img>` + N tokens + `</img>`, runs
the vision encoder on every tile, concatenates the per-tile embeddings, and
queues a `PreparedEmbeddingSpan` so the model splices the embeddings into the
right positions before `Forward()`.

### 4.6 Audio frontend (`NemotronAudioPreprocessor`)

Parakeet-style log-mel spectrogram extraction (mirrors ollama's
`process_audio.go`):

- Mono resample to 16 kHz.
- 0.97 pre-emphasis.
- STFT: `n_fft = 512`, `hop = 160`, `win = 400`, center-padded constant.
- Slaney-style mel filter bank with 128 bins, 0..8 kHz.
- `log(power + 2⁻²⁴)` followed by per-mel mean/var normalization across
  valid (non-padded) frames.

The chat template emits a `<so_embedding>` token for each uploaded audio
file so the model "sees" the modality, but real audio inference is gated on
a Parakeet audio mmproj that is not shipped with the public Nemotron-H or
Nemotron Omni GGUFs in this distribution. The CLI still preprocesses the
audio file for verification.

## 5. Parameters and settings

| Key | Type | Meaning |
|---|---|---|
| `nemotron_h.ssm.conv_kernel` | uint32 | Mamba2 conv1d kernel size |
| `nemotron_h.ssm.inner_size` | uint32 | SSM inner dim (`nHead * headDim`) |
| `nemotron_h.ssm.state_size` | uint32 | SSM state dim per head |
| `nemotron_h.ssm.time_step_rank` | uint32 | Number of SSM heads |
| `nemotron_h.ssm.group_count` | uint32 | Number of SSM groups |
| `nemotron_h.attention.head_count_kv` | uint32[] | Per-layer KV head count (0 ⇒ Mamba2) |
| `nemotron_h.attention.head_count` | uint32[] | Per-layer Q head count |
| `nemotron_h.feed_forward_length` | uint32[] | Per-layer FFN size (0 ⇒ no FFN) |
| `nemotron_h.attention.scale` | float32 | Attention scale factor (0 ⇒ auto) |
| `nemotron_h.expert_count` | uint32 | MoE expert count (0 ⇒ dense) |
| `nemotron_h.expert_used_count` | uint32 | TopK experts per token |
| `nemotron_h.expert_weights_norm` | bool | Renormalize selected expert weights to sum=1 |
| `nemotron_h.expert_weights_scale` | float32 | Global scale factor applied to expert weights |

For the Omni vision projector (`mmproj` GGUF), the standard `clip.vision.*`
keys (`embedding_length`, `feed_forward_length`, `attention.head_count`,
`block_count`, `image_size`, `patch_size`, `num_channels`, `projection_dim`,
`projector.scale_factor`, `attention.layer_norm_epsilon`, `use_gelu`) are
read by `NemotronVisionEncoder`.

## 6. Weight naming convention

```
token_embd.weight
output_norm.weight
output.weight

# Mamba2 layers:
blk.{L}.attn_norm.weight
blk.{L}.ssm_in.weight       # [hidden → 2*dInner + 2*nGroup*dState + nHead]
blk.{L}.ssm_conv1d.weight   # [xBCSize, convKernel]
blk.{L}.ssm_conv1d.bias     # (optional)
blk.{L}.ssm_dt.bias         # [nHead]
blk.{L}.ssm_a               # [nHead]
blk.{L}.ssm_d               # [nHead] (optional)
blk.{L}.ssm_norm.weight     # group RMSNorm [dInner]
blk.{L}.ssm_out.weight      # [dInner → hidden]

# Attention layers:
blk.{L}.attn_norm.weight
blk.{L}.attn_qkv.weight     # fused Q+K+V (or separate attn_q/k/v.weight)
blk.{L}.attn_output.weight

# FFN layers (dense):
blk.{L}.attn_norm.weight
blk.{L}.ffn_up.weight
blk.{L}.ffn_down.weight

# FFN layers (MoE):
blk.{L}.attn_norm.weight
blk.{L}.ffn_gate_inp.weight       # router [numExperts, hidden]
blk.{L}.exp_probs_b.bias          # router bias (optional)
blk.{L}.ffn_latent_in.weight      # latent bottleneck in (optional)
blk.{L}.ffn_latent_out.weight     # latent bottleneck out (optional)
blk.{L}.ffn_up_exps.{E}.weight
blk.{L}.ffn_down_exps.{E}.weight
blk.{L}.ffn_up_shexp.weight       # shared expert (optional)
blk.{L}.ffn_down_shexp.weight
```

## 7. TensorSharp implementation walkthrough

Constructor:

1. `ParseBaseConfig()`.
2. Reads SSM dims (`_ssmDConv`, `_ssmDInner`, `_ssmDState`, `_ssmNHead`,
   `_ssmNGroup`, derives `_ssmHeadDim`).
3. Reads MoE counts and bias / scale flags.
4. Reads per-layer arrays (`head_count_kv`, `head_count`,
   `feed_forward_length`) and classifies each layer into Mamba2 / Attention
   / FFN.
5. `ParseTokenizer()`, `LoadWeights()`.
6. `FuseFFNWeights()` (placeholder for future dense FFN fusion),
   `FuseQKVWeights()` (only on attention layers).
7. `PrepareCudaQuantizedWeightsForInference()`.
8. `InitCaches(maxSeqLen)` allocates KV caches for attention layers, conv
   states for Mamba2 layers, and SSM states for Mamba2 layers.
9. `InitMamba2Buffers()` pre-allocates `_mamba2ConvOutBuf` /
   `_mamba2YBuf`.
10. `InitLayerInfo()` precomputes layer-prefix strings and per-MoE-layer
    flags (`HasLatentIn`, `HasSharedExperts`, `LatentDim`).
11. `InitMoEBuffers()` pre-allocates the routing and expert pointer arrays.

`Forward(int[] tokens)` runs the per-op managed loop. For each layer it
dispatches to `Mamba2Block` / `AttentionBlock` / `FFNBlock` based on
`_layerTypes[l]`. Optional CPU offload (`CPU_MATMUL_THRESHOLD`, off by
default on Apple Silicon) routes small matmuls to the managed CPU path even
when a GPU backend is selected.

## 8. Prefill optimization

- **Per-layer dispatch tables** (`_layerPrefixes`, `_layerWeightNames`)
  avoid string interpolation in hot loops.
- **MoE prefill** still iterates per token. Each token uses the batched
  MoE GPU kernel (`MoEExpertsForward`) so all selected experts run in a
  single dispatch per token, but the token loop is still managed C# — see
  the optimization opportunities below.
- **Attention prefill** uses the standard managed loop. There is no fused
  prefill attention kernel for Nemotron-H yet because the score tensor is
  small (no SWA window machinery is needed since attention layers have no
  RoPE).
- **Mamba2 prefill** processes tokens **sequentially** (loop over `seqLen`)
  through the SSM scan; chunked parallel scanning is on the optimization
  list.
- **Multimodal prefill** supports chunked server prefill by slicing prepared
  image/audio embedding spans per prompt chunk, so long image prompts no longer
  have to run as one monolithic forward pass.
- **Multimodal warmup** runs a tiny vision encode plus an image-token prefill
  during server startup when a Nemotron `mmproj` is loaded. This shifts Metal
  pipeline setup away from the first real image request; set
  `TS_NEMOTRON_MULTIMODAL_WARMUP=0` to disable it.

## 9. Decode optimization

### Batched GPU MoE (`MoEExpertsForward`)

For MoE FFN layers on a GGML backend (Metal / CUDA), all selected experts
are processed in one GGML graph dispatch:

- Pre-cached `QuantizedWeight[]` arrays (`_expertUpQW[layer][e]`,
  `_expertDownQW[layer][e]`) — populated once at init time.
- Pre-allocated `IntPtr[] _moeUpPtrs` / `_moeDownPtrs` arrays of length
  `_numExpertsUsed`.
- Pre-allocated reusable result tensors (`_expertUpResult`,
  `_expertDownResult`, `_latentAccumTensor`, `_latentOutResult`).

### CPU offload for tiny ops

Small ops on the decode hot path (RMSNorm, residual add, expert / router
matmuls) can be executed on CPU via the CPU SIMD kernels even when the
backend is GPU. This avoids the per-dispatch GPU overhead (~1 ms+ per call
on Metal). Large matmuls (SSM in/out, attention QKV/output, LM head)
remain on GPU. The threshold is `CPU_MATMUL_THRESHOLD` (0 by default on
Apple Silicon unified memory; raise it for discrete GPU systems).

### SIMD ReLU² and bias add

`ReluSquaredInPlace` and `LinearForwardInto` are SIMD-vectorized using
`System.Numerics.Vector<float>` so the dense FFN compute on CPU runs at
near-peak vector throughput.

### Pre-allocated decode buffers

| Buffer | Size | Purpose |
|---|---|---|
| `_mamba2ConvOutBuf` | `dInner + 2 * nGroup * dState` | Conv1d output + SiLU |
| `_mamba2YBuf` | `dInner` | SSM scan output |
| `_moeProbs` / `_moeSelectionProbs` | `numExperts` | Router probabilities |
| `_moeTopExperts` / `_moeRouteW` | `numExpertsUsed` | Selected experts and weights |
| `_moeLatentAccum` | `max(hiddenSize, latentDim)` | Latent-space accumulator |
| `_expertUpResult` / `_expertDownResult` | max expert dims | Reusable expert matmul outputs |
| `_latentAccumTensor` / `_latentOutResult` | `latentDim` / `hiddenSize` | Latent bottleneck reuse tensors |

## 10. Memory and KV cache strategy

- **Attention layers**: standard KV cache `[numKVHeads, maxSeqLen, headDim]`.
- **Mamba2 layers**: `_convState[layer]` (size `(convKernel - 1) * (dInner +
  2 * nGroup * dState)` floats) and `_ssmState[layer]` (size `dState *
  headDim * nHead` floats).
- `ResetKVCache()` zeroes all three (KV caches, conv states, SSM states).
- `SupportsKVCacheTruncation` returns **false** because SSM states are
  sequential and cannot be partially reused. Multi-turn KV-cache reuse
  therefore is not enabled for Nemotron-H — the server falls back to a full
  reset between turns.

## 11. Batched / paged forward (continuous batching)

Nemotron-H implements `IBatchedPagedModel.ForwardBatch`
([`NemotronModel.BatchedForward.cs`](../../TensorSharp.Models/Models/Nemotron/NemotronModel.BatchedForward.cs))
that runs through the shared `InferenceEngine` continuous-batching stack
([`docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md))
by default — concurrent requests can only be served truly in parallel
through the batched path. Set `TS_NEMOTRON_BATCHED=0` to force the
legacy per-sequence KV-swap fallback for A/B comparison.

Nemotron-H is the most demanding of the batched ports because it
combines **three different layer types** — Mamba2 SSM, attention-only,
and FFN (dense or MoE) — and Mamba2 is recurrent (per-sequence state).
The batched path therefore needs two orthogonal caches:

### Attention layers — paged K/V

- Mirrors Mistral 3 layout
  (`[numBlocks * blockSize * numKvHeads * headDim]` per layer).
- No RoPE — Nemotron-H attention layers carry no positional encoding.
- Per-sequence attention dispatch via `ManagedPagedAttention.Forward`
  (the pure-C# online-softmax kernel) as the correctness reference. The
  native paged kernel path is also wired through `GgmlBasicOps`.

### Mamba2 layers — per-slot conv + SSM state pool

Each sequence's recurrent state lives in a slot keyed on its **primary
block id** (matching vLLM's `state_indices_tensor`):

- `_nemoSlotMamba2NativeDecodeProjected[layer][slot]` — per-slot conv
  ring buffer.
- `_nemoSlotMamba2NativeDecodeHidden[layer][slot]` — per-slot SSM state.
- `_nemoSlotMamba2NativeDecodeStateInitialized[layer][slot]` —
  initialization flag.

Slots are allocated lazily on first touch and freed when the engine
retires the sequence.

A **native batched Mamba2 step kernel** —
`TSGgml_NemotronMamba2BatchedStepF32`
([`ggml_ops_mamba2.cpp`](../../TensorSharp.GGML.Native/ggml_ops_mamba2.cpp))
— is gated behind `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1`. It uses NEON
SIMD + GCD per-head parallelism (mirrors the Qwen 3.5 batched GDN
kernel structure) and replaces N C# `Mamba2Block` calls with one native
dispatch plus batched `ssm_in` / `ssm_out` projections. Exposed through
`GgmlBasicOps.NemotronMamba`.

### FFN — dense and MoE

- Dense FFN runs token-parallel ReLU² (`up → ReLU² → down`) across the
  entire batched token axis in a single matmul per projection.
- MoE FFN runs through the existing `MoEForward` token-parallel router
  + per-token expert dispatch (no new batched-MoE kernel for
  Nemotron-H yet).

### Multimodal in the batched path

Vision and audio embeddings inject directly into the batched
`[numTokens, hidden]` tensor via the same row-wise
`InjectMultimodalEmbeddings` path the legacy forward uses.
`SupportsBatchedMultimodal` returns true while the batched path is
active (i.e. unless `TS_NEMOTRON_BATCHED=0` is set).

### Verified correctness and throughput

- **100% greedy match** vs legacy on text-only prompts
  ([`NemotronBatchedCorrectnessTests`](../../InferenceWeb.Tests/NemotronBatchedCorrectnessTests.cs)).
- Multimodal-prompt correctness is structurally validated (text-only
  stays 100% after removing the multimodal pre-flight rejection) but
  lacks a local audio/image fixture for end-to-end verification.

**Throughput on NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-IQ2_XXS
(Apple M4 Pro, GgmlMetal, legacy-vs-batched in-process toggle in
[`NemotronBatchedPerfBench`](../../InferenceWeb.Tests/NemotronBatchedPerfBench.cs))**:

| Path | n=1 | n=3 | n=5 |
|---|---|---|---|
| Batched + managed Mamba2 step | 1.94× | 3.43× | 1.67× |
| Batched + native Mamba2 step (`*_NATIVE=1`) | 0.75× | **3.95×** | 2.93× |

`n=1` is the only regression: batched scaffolding outweighs the benefit
for single-sequence decode. From `n=2` onward the batched path wins
across the board. `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` extends the
win to multi-batch decode by replacing the C# Mamba2 inner loop with the
native NEON kernel.

A latent bug was also fixed during the port: `s_nemoBatchedOptIn` used to
be `static readonly`, which captured the env var at class-load time —
tests setting `TS_NEMOTRON_BATCHED=1` at runtime never actually toggled
the path. Now exposed as a method getter (same pattern as Qwen 3.5).

## 12. Output parser and chat template

- `Qwen3OutputParser` is reused: `<think> ... </think>` for chain-of-thought
  reasoning and `<tool_call>{...}</tool_call>` for tool calls.
- Chat template uses the Qwen 3 chat template format (`<|im_start|>` /
  `<|im_end|>`). Multimodal placeholders include `<image>` (later expanded
  into `<img>` + N + `</img>`) and `<so_embedding>` (audio).

## 13. Optimization opportunities

- **Native whole-model decode** — the entire legacy forward pass runs
  in managed C# today. A native `NemotronModelDecode` (analogous to
  Qwen 3) would eliminate the managed loop overhead on the per-seq path.
- **Native Mamba2 decode for the legacy path** — the SIMD-vectorized
  scan in `Mamba2SSMStepSIMD` is already fast on CPU, but a native CUDA
  / Metal kernel would unblock GPU-side execution of the full Mamba2
  path on the per-seq path. The batched path already has the native
  kernel under `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1`.
- **Chunked parallel SSM scanning** — Mamba2 prefill processes tokens
  sequentially. A chunked parallel scan (à la Mamba's reference CUDA
  implementation) would dramatically improve TTFT.
- **Vectorized conv1d** — `Mamba2Conv1dStep` is a scalar loop. SIMD or
  native vectorization would speed up Mamba2 layers further.
- **Per-token MoE batching** — even with `MoEExpertsForward`, the per-token
  managed loop is still the outer driver. A batched kernel that handles
  multiple tokens in one dispatch would help long prompts.
- **Audio mmproj support** — the audio frontend is fully wired, but
  inference requires a Parakeet audio projector that is not shipped in
  current GGUFs. Once available, plugging it into the same
  `_pendingAudioEmbeddings` injection path that Gemma 4 uses is a small
  amount of code.
