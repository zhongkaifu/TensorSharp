# Gemma 4

[← back to model index](README.md) | [中文](gemma4_zh-cn.md)

> **Looking for the guided, from-scratch journey?** Zhongkai Fu's [From Tensors to Tokens](../BOOK.md) uses Gemma 4 E4B to connect tensor foundations, multimodal model execution, and a complete TensorSharp inference application. [View the paperback on Amazon](https://www.amazon.com/dp/B0H9P44QZZ).

| Property | Value |
|---|---|
| Provider | Google |
| GGUF architecture key | `gemma4` |
| Source class | [`Gemma4RuntimeModel`](../../TensorSharp.Models/Models/Gemma4/Gemma4RuntimeModel.cs) on the shared decoder runtime (`StandardDecoderModel`; graph programs on every executor) |
| Vision encoder | [`Gemma4VisionEncoder`](../../TensorSharp.Models/Models/Gemma4/Gemma4VisionEncoder.cs) (SigLIP-style ViT) |
| Audio encoder | [`Gemma4AudioEncoder`](../../TensorSharp.Models/Models/Gemma4/Gemma4AudioEncoder.cs) (USM-style chunked transformer) |
| Audio frontend | [`Gemma4AudioPreprocessor`](../../TensorSharp.Models/Models/Gemma4/Gemma4AudioPreprocessor.cs) (16 kHz mono → 128-bin log-mel) |
| Image processor | [`Gemma4ImageProcessor`](../../TensorSharp.Models/Models/Gemma4/Gemma4ImageProcessor.cs) |
| Example models | gemma-4-E4B (8B effective), gemma-4-12B, gemma-4-31B, gemma-4-26B-A4B (MoE) |
| Modalities | Text, image, video (frame stack), audio |
| Thinking mode | Yes (`<\|channel>thought ... <channel\|>`) |
| Tool calling | Yes (`<\|tool_call>call:name{...}<tool_call\|>`) |
| Batched / paged forward | The shared runtime's continuous batching: requests run in batched steps over their own slots (prompt chunks and decode rows together), with the prefix cache and checkpoints. See §10. |
| MTP speculative decoding | Optional — loads a separate `gemma4-assistant` EAGLE-style draft GGUF via `--draft-model` (`TS_SPEC_DRAFT_MODEL`); naming the file enables speculation by itself (an explicit `--no-spec` vetoes it). The flag works on **either host**: `TensorSharp.Cli` and `TensorSharp.Server` share [`SpeculativeCliFlags`](../../TensorSharp.Runtime/Speculative/SpeculativeCliFlags.cs). Runs on every graph executor. See §12. |
| Output parser | `Gemma4OutputParser` |

## Downloads

Verified GGUF pointers:

| Model | HF repo | Recommended file | mmproj (image / video / audio) | MTP draft |
|---|---|---|---|---|
| gemma-4-E4B-it | [ggml-org/gemma-4-E4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF) | `gemma-4-E4B-it-Q8_0.gguf` (8.031 GB; also `gemma-4-E4B-it-Q4_0.gguf`, 4.588 GB) | `mmproj-gemma-4-E4B-it-Q8_0.gguf` (0.560 GB; same repo; alt: `mmproj-F16.gguf`, 0.990 GB, in [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF)) | `gemma-4-E4B-it-assistant.Q8_0.gguf` (0.100 GB) from [AtomicChat/gemma-4-E4B-it-assistant-GGUF](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF) |
| gemma-4-12B-it (QAT) | [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) | `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` (6.716 GB) | `mmproj-BF16.gguf` (0.175 GB; same repo) | `mtp-gemma-4-12B-it.gguf` (0.254 GB; same repo root; quantized variants under `MTP/`) |
| gemma-4-26B-A4B-it (MoE, QAT) | [unsloth/gemma-4-26B-A4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-qat-GGUF) | `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` (14.249 GB) | `mmproj-BF16.gguf` (1.195 GB; same repo) | `mtp-gemma-4-26B-A4B-it.gguf` (0.252 GB; same repo), or `gemma-4-26B-A4B-it-assistant.Q8_0.gguf` (0.462 GB) from [AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF](https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF) |
| gemma-4-26B-A4B-it (MoE) | [ggml-org/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) | `gemma-4-26B-A4B-it-Q4_K_M.gguf` (16.796 GB) or `gemma-4-26B-A4B-it-Q8_0.gguf` (26.860 GB) | `mmproj-gemma-4-26B-A4B-it-Q8_0.gguf` (0.806 GB) / `mmproj-gemma-4-26B-A4B-it-bf16.gguf` (1.195 GB) | same drafts as the QAT row |
| gemma-4-31B-it | [ggml-org/gemma-4-31B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) | `gemma-4-31B-it-Q4_K_M.gguf` (18.687 GB) or `gemma-4-31B-it-Q8_0.gguf` (32.636 GB) | `mmproj-gemma-4-31B-it-Q8_0.gguf` (0.810 GB) / `mmproj-gemma-4-31B-it-bf16.gguf` (1.201 GB) | — |

Hugging Face identifies every row above as derived from the matching Google
Gemma 4 base model. Some conversion cards do not declare a license; anonymous
download does not replace the upstream model terms, so review the linked base
model and conversion card before redistribution.

The MTP draft head ships as a separate `gemma4-assistant` GGUF whose backbone
dimension must equal the target's hidden size — always pair draft and target
from the same size (E4B draft ↔ E4B target, 12B ↔ 12B, 26B-A4B ↔ 26B-A4B).
A mismatched draft fails fast at server startup (§12.2).

Command-line download (one line per file; requires `pip install -U huggingface_hub`):

```bash
python -m pip install -U huggingface_hub
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download ggml-org/gemma-4-E4B-it-GGUF mmproj-gemma-4-E4B-it-Q8_0.gguf --local-dir models
hf download unsloth/gemma-4-12B-it-qat-GGUF gemma-4-12B-it-qat-UD-Q4_K_XL.gguf --local-dir models
hf download unsloth/gemma-4-12B-it-qat-GGUF mtp-gemma-4-12B-it.gguf --local-dir models
```

### Verified Gemma 4 E4B native GGML fast path

This is TensorSharp's verified quick-start lane. The E4B Q8_0 family/path
uses the public
[`ggml-org/gemma-4-E4B-it-GGUF`](https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF)
artifact, a normal native build, and a native GGML GPU backend. Verification is
at the E4B Q8_0 family/path level; it does not assert that a particular public
file checksum was the benchmark input. This copy/paste block is for
Linux + NVIDIA; platform-specific backend choices follow it. On a new machine,
install and verify the [.NET 10 SDK](../../DEVELOPMENT.md#install-the-net-10-sdk)
before running the build command; the runtime alone is not sufficient.

```bash
python -m pip install -U huggingface_hub
hf download ggml-org/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q8_0.gguf --local-dir models
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON dotnet build TensorSharp.slnx -c Release -p:TensorSharpSkipMlxNative=true
printf '%s\n' 'Answer in one short sentence: what is TensorSharp?' > prompt.txt
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model models/gemma-4-E4B-it-Q8_0.gguf \
  --input prompt.txt --max-tokens 64 --backend ggml_cuda
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model models/gemma-4-E4B-it-Q8_0.gguf \
  --backend ggml_cuda --max-tokens 128
```

Use `ggml_cuda` on Windows/Linux with NVIDIA, `ggml_metal` on Apple Silicon,
or `ggml_vulkan` on Windows/Linux with a Vulkan-capable AMD, Intel, or NVIDIA
GPU. The matching `mmproj-gemma-4-E4B-it-Q8_0.gguf` is optional for text-only
requests; download it from the same repository and pass `--mmproj` for image,
video, or audio input.

Every step runs as a whole-forward graph program on the backend's executor (§7, §8); decode steps of one width replay
as the previous step advanced.


Multimodal CLI one-shot (the text prompt comes from a file via `--input`; with
`--image` and no `--input` a default describe-the-image prompt is used; CLI
sampling defaults to greedy and `--max-tokens` defaults to 100):

```bash
dotnet run --project TensorSharp.Cli -c Release -- --model models/gemma-4-E4B-it-Q8_0.gguf \
  --mmproj models/mmproj-gemma-4-E4B-it-Q8_0.gguf \
  --image photo.png --max-tokens 512 --backend ggml_cuda
```

With MTP speculative decoding. `--draft-model` is parsed by the
same [`SpeculativeCliFlags`](../../TensorSharp.Runtime/Speculative/SpeculativeCliFlags.cs)
on **both hosts**, so the spelling is identical on `TensorSharp.Cli` and
`TensorSharp.Server`. Naming the draft GGUF enables speculation by itself — no
`--spec` is needed beside it (an explicit `--no-spec` vetoes it). It must be on
the command line that *loads* the model — the draft GGUF is
attached to the target at startup, and a `--draft-model` that cannot be
activated is a fail-fast startup error rather than a silent fallback (§12.2):

```bash
# server
dotnet run --project TensorSharp.Server.Host -c Release -- --model models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  --backend ggml_cuda --draft-model models/mtp-gemma-4-12B-it.gguf

# CLI — speculation engages on --input, --input-jsonl, --multi-turn-jsonl and --interactive
dotnet run --project TensorSharp.Cli -c Release -- --model models/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf \
  --backend ggml_cuda --draft-model models/mtp-gemma-4-12B-it.gguf --input prompt.txt --max-tokens 512
```

Then open `http://localhost:5000` for the chat UI.

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

The SWA mask and RoPE caches are precomputed across all layers in a forward
pass, the SWA cache is a circular buffer (so memory is
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

Both are the same graph program at different widths (§8): a prompt chunk attends the cached prefix and its own rows, a
decode row the padded cache window under a causal (or band) mask, and media soft tokens attend each other in both
directions within their forward.

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

Uses `tanh(logits / cap) * cap` when
`gemma4.final_logit_softcapping > 0`.

### 4.9 Vision pipeline (images and video frames)

`Gemma4VisionEncoder` is a SigLIP-style ViT with 2D position embeddings, a
gated GELU-tanh MLP (`gelu_pytorch_tanh`, as the vision config declares; the
fused native block and the managed fallback both use it, where earlier builds
used QuickGELU), and a pooled projection to the LM hidden dim: an unweighted
RMSNorm over the vision hidden size (`embedding_pre_projection_norm`) followed
by the linear projector. Earlier builds normalized after the projection
instead; the order now matches upstream `Gemma4MultimodalEmbedder`.

The tower loads from an mmproj GGUF or, through the same `--mmproj` option,
from a HuggingFace `.safetensors` shard that holds it under
`model.encoder.vision_tower.*` with the projector
`model.encoder.embed_vision.embedding_projection.weight`
([`Gemma4VisionEncoder.Safetensors.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4VisionEncoder.Safetensors.cs)).
Geometry comes from the tensor shapes; the pooling factor and RMSNorm epsilon
come from a `config.json` beside the shard when its `vision_config` matches the
tower's width, otherwise from the Gemma 4 defaults with a one-time warning.
This path exists for checkpoints published without an mmproj
([DiffusionGemma](diffusiongemma.md) is the case it was written for). A shard
carries no audio tower, so audio input still needs an mmproj GGUF. Video frames
are extracted from MP4 inputs using time-based sampling (one frame per second by
default via OpenCV, configurable with `VIDEO_SAMPLE_FPS`; `VIDEO_MAX_FRAMES` adds
an optional upper bound) and each frame is encoded independently; the resulting
embeddings are concatenated and injected at successive `<|image>` markers.

Still images are sized by projector type. A `gemma4v` (SigLIP) tower, such as
E4B's or DiffusionGemma's, gets the reference `Gemma4ImageProcessor` sizing: the
image is scaled up or down to the largest canvas within the 280-soft-token budget
whose sides are multiples of 48 px (patch 16 x pooling 3), then resized with an
antialiased bicubic filter and stretched to fill, with no letterbox. Earlier
builds used a letterboxing port of llama.cpp's smart resize that padded the
picture with black bars, so together with the activation and norm fixes above,
image answers on these towers change after upgrading. The resized pixels are
checked against a numpy transcription of the reference (`Gemma4VisionOracleTests`,
which needs a local fixture directory and is skipped without one). The 12B's
`gemma4uv` unified embedder, a block-less path (one large-patch conv, LayerNorms,
learned 2D positions, an RMSNorm and a linear projection) rather than the SigLIP
stack above, keeps the older letterboxed sizing it was validated on. Video frames
use a fixed 70-soft-token budget with that older sizing.

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

## 7. Implementation on the shared decoder runtime

Gemma 4 runs on the shared decoder runtime: [`Gemma4RuntimeModel`](../../TensorSharp.Models/Models/Gemma4/Gemma4RuntimeModel.cs)
derives from [`Gemma4DecoderModel`](../../TensorSharp.Models/Models/Gemma4/Gemma4DecoderModel.cs), the recipe it shares with
[DiffusionGemma](diffusiongemma.md), a [`StandardDecoderModel`](../../TensorSharp.Models/SharedDecoder/StandardDecoderModel.cs) family, so slots, the unified
step, continuous batching, the prefix cache, checkpoints, placements and speculation are the runtime's, and every step is a
whole-forward graph program run by the backend's executor (GGML, the managed CPU, direct CUDA, MLX). The family supplies
the recipe through the runtime's hooks:

| Part | Where | What |
|---|---|---|
| Per-layer attention geometry | `Gemma4DecoderModel.cs` (`LayerKeyDim`, `LayerKvHeads`, `SlidingWindow`, `AttentionScale`) | 256-wide sliding-window heads and 512-wide global heads, per-layer KV head counts, scale 1 |
| KV sharing | `KvSourceLayer`; the runtime's `EmitSharedKvAttention` | the last `shared_kv_layers` layers attend their source's operands of the same step and project queries only |
| Q/K/V | `Gemma4DecoderModel.Graph.cs` (`EmitAttentionProjections`, `EmitQueryKey`) | weighted Q/K RMSNorm, unweighted V norm, V = K on global layers without `attn_v` |
| Rotary | `RotarySpec` with `rope_freqs` factors (`GraphOp.RopeExt` with a factor input) | NeoX local rotary at base 10 000; global rotary over 512 dims with the proportional factors |
| Feed-forward | `EmitFeedForward`, `EmitRoutedExperts` | GeGLU; MoE layers: dense branch beside routed GeGLU experts over the fused `ffn_gate_up_exps` stack, with the router scale and per-expert output scale |
| Per-layer embeddings | `EmitStreamInput` / `EmitPerLayerInputs` / `EmitBlockOutput` | token rows of `per_layer_token_embd` plus the normed model projection, gated into the stream after each layer |
| Media | [`Gemma4RuntimeModel.Multimodal.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4RuntimeModel.Multimodal.cs) | the towers' soft-token rows replace their placeholders before the `sqrt(n_embd)` scale (which skips them) and attend each other in both directions |
| MTP head | [`Gemma4RuntimeModel.Assistant.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4RuntimeModel.Assistant.cs) | the gemma4-assistant drafter (§12) |

The runtime pieces Gemma 4 added for everyone: per-layer key/value widths and KV head counts, KV-shared layers, rotary
frequency factors on every executor (`Ops.RoPEExFactors`, a CUDA kernel, MLX `fast.rope` with frequencies), bidirectional
media spans, a stage carry for values the stream input makes that later stages read, read-only attention over a trunk
layer's cache (`EmitCacheReadAttention`) and per-token family drafters (`FamilyTokenDrafter`).

## 8. Prefill and decode

A prompt chunk and a decode row are the same graph program at different widths. Decode steps of one width replay as the
previous step advanced (positions, masks, write rows and the per-layer-embedding token ids are refilled in place), and on
direct CUDA they replay as recorded kernel sequences. Sliding-window layers keep a ring of the window plus a prompt step's
rows. On direct CUDA the routed experts of a decode row run as one fused region (`GraphFusion.GegluRoutedExperts`: the
fused-stack MoE decode kernels with dp4a), which took 26B-A4B decode from 34 to 80 tok/s on an A40.

## 9. Memory and KV cache

KV-shared layers own no cache. Every cache type the runtime serves works, the block-quantized ones included (`q8_0`,
`q4_0`; the legacy model declined them and fell back to `f16`): on E4B `ggml_cuda` a `q8_0` cache matched the `f16`
reference at cos 0.9999 over 96 teacher-forced steps. Routed experts may stay in host memory (`--n-cpu-moe` / `--cpu-moe`)
on every backend: GGML's host MoE segment and the tensor executors' expert host run GeGLU over the fused stack in the
native host-expert kernel (`TSGgml_HostExpertRowsAct`); 26B-A4B with 12 host layers on `ggml_cuda` agreed with the
legacy model's all-device run on 95 of 96 teacher-forced steps (the other a 0.06-nat near-tie).

## 10. Continuous batching, prefix cache and checkpoints

Requests run in batched steps over their own slots (prompt chunks and decode rows in one step); the prefix cache,
retained conversations, checkpoints and their export/import are the runtime's (see
[Paged attention and continuous batching](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md)).

## 11. Validation against the legacy model

The runtime model replaced the legacy `Gemma4Model`, which was deleted with its native whole-model graphs
(`TSGgml_Gemma4ModelDecode`, `...Verify`, `...DecodeBatched`, the MoE variants and `TSGgml_Gemma4DraftStep`). Measured
on one A40 (ggml at d9fe7bff), with the legacy and runtime binaries built from the same tree:

| Check | Result |
|---|---|
| Teacher-forced argmax agreement (E4B Q8_0, 12B UD-Q4_K_XL, 26B-A4B UD-Q4_K_XL; ggml_cuda, cuda, ggml_cpu, cpu) | every step agrees except near-ties (the largest legacy margin at a disagreement 0.29 nat) |
| Verify windows against plain steps (E4B, windows of 4, with rollbacks) | ggml_cuda relL2 1.43e-2 (legacy 1.42e-2); direct CUDA bit-identical (legacy 1.3e-2) |
| Images (E4B, 12B, 26B; ggml_cuda and cuda) and speech (E4B) through the CLI | the same descriptions and the same transcription as the legacy model |
| Server, 5 prompts, 3 concurrent (ggml_cuda) | E4B 102 tok/s against legacy 90; 26B-A4B 74 against 56 |
| Prefill (ggml_cuda, warm) | 1.5-2.4x the legacy model's |

Greedy streams of the two models part where a token's top two logits are within the backends' rounding; the frozen legacy
logits of the synthetic fixtures (`InferenceWeb.Tests/Fixtures/Gemma4/LegacySynthetic`) hold the runtime to the legacy
math in [`Gemma4RuntimeTests`](../../InferenceWeb.Tests/Gemma4RuntimeTests.cs). Not run here: Metal, Vulkan, MLX and
more than one GPU (tensor parallelism and stages were tested over simulated devices on the managed executor).

## 12. MTP speculative decoding (gemma4-assistant draft head)

Gemma 4 supports **multi-token-prediction (MTP) speculative decoding**
on both hosts. Unlike Qwen 3.6,
whose NextN block is embedded in the trunk GGUF, the Gemma 4 draft head ships as
a **separate small `gemma4-assistant` GGUF** loaded with `--draft-model`
(env `TS_SPEC_DRAFT_MODEL`)
and attached to the target at startup by
[`SpeculativeDraftHeadLoader`](../../TensorSharp.Models/SpeculativeDraftHeadLoader.cs).
Source:
[`Gemma4RuntimeModel.Assistant.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4RuntimeModel.Assistant.cs),
driven through the shared
[`SpeculativeExecution`](../../TensorSharp.Runtime/Speculative/SpeculativeExecution.cs)
draft / verify / rollback core — Gemma 4 is the `draft-head` algorithm of the
pluggable `--spec-type` layer, which `auto` already picks from the checkpoint.

### 12.1 EAGLE-style recurrent drafter

The draft head is an EAGLE-style recurrent drafter (mirrors llama.cpp's
`gemma4-assistant.cpp` + `speculative.cpp` draft-mtp). For each drafted token:

```
x      = target.tok_embd[token] * sqrt(n_embd_backbone)   // backbone embedding (e.g. 3840)
xh     = concat(x, h_prev)                                 // [2*backbone]
cur    = nextn.pre_projection @ xh                         // -> draft hidden (e.g. 1024)
for il in 0..n_nextn-1:                                    // a few Gemma-style decoder blocks
    Q   = rope(q_norm(wq @ attn_norm(cur)))               // draft computes ONLY Q
    a   = attn(Q, target_KV)                               // attends the TARGET's last local
                                                          //   (layer N-2) / global (layer N-1) K/V
    cur = block_residuals(a, ffn, out_scale)
cur    = output_norm(cur)
logits = draft.tok_embd @ cur                             // draft's own LM head
h_next = nextn.post_projection @ cur                      // -> backbone, chains the next draft step
```

The draft holds **no K/V of its own** (no `wk`/`wv` tensors): every draft layer
queries into the target model's existing per-layer KV cache, reusing the same
position for every drafted token (the recurrence flows purely through `h`). That
makes the drafter stateless given `(token, h)`, so the only rollback on rejection
is an attention-KV position rewind — no recurrent-state snapshot is needed (in
contrast to Qwen 3.6's GatedDeltaNet trunk).

### 12.2 Draft / target pairing (fail-fast)

The draft's output backbone dimension (`{arch}.embedding_length_out`) **must equal
the target's hidden size** — pair the 12B target with its 12B draft, not the
26B-A4B draft. When `--draft-model` is given but the draft can't be activated
(file missing, hidden-size mismatch, or required draft tensors absent), the server
**fails fast at startup** with a remediation hint
([`SpeculationStartupValidation`](../../TensorSharp.Chat/Hosting/SpeculationStartupValidation.cs))
rather than silently running without speculation.

### 12.3 The drafter on the runtime

The assistant is a per-token family drafter ([`Gemma4RuntimeModel.Assistant.cs`](../../TensorSharp.Models/Models/Gemma4/Gemma4RuntimeModel.Assistant.cs)):
each step is one graph program over the active slot whose blocks read the trunk's donor caches (through the KV-sharing
map) with `EmitCacheReadAttention` at the trunk's committed length, so there is nothing to catch up and a draft resumes at
any position. It runs on every graph executor. Under `--tp` the head is refused (it reads whole caches, which tensor
parallelism shards by KV head); n-gram speculation still runs. Measured on an A40 (greedy, 128 tokens, 3 drafts):

| Model, backend | Plain tok/s | With the head | Acceptance | Legacy with its head |
|---|---|---|---|---|
| E4B Q8_0, ggml_cuda | 86-89 | 1.0-2.2x | 22-83 % | 0.9-1.8x |
| E4B Q8_0, cuda | 69-74 | 1.0-2.1x | 21-83 % | 0.9-1.8x |
| 12B UD-Q4_K_XL, ggml_cuda | 65-72 | 1.5-2.3x | 52-86 % | 0.7-1.4x |
| 26B-A4B UD-Q4_K_XL, ggml_cuda | 114-141 | 0.9-1.4x | 44-89 % | 0.6-0.9x |

Opt-in is the `--draft-model` file itself; opt-out (`--no-spec`) and tuning (`--spec-draft` / `--spec-pmin`) use the
shared flags, which `TensorSharp.Cli` and `TensorSharp.Server` parse from the same
[`SpeculativeCliFlags`](../../TensorSharp.Runtime/Speculative/SpeculativeCliFlags.cs); see
[Speculative decoding](../../FEATURES.md#speculative-decoding) for the other algorithms (`--spec-type ngram` needs no
draft GGUF at all).

### 12.4 Greedy parity

Every emitted token is drawn from a trunk row, so speculation cannot change
which prefix a token is conditioned on, only the kernel that computed its row.
Two consequences, both measured with `AgentTurnBench --spec-diagnostic
--spec-diagnostic-teacher-force`:

* **Fixed:** on the dense checkpoints without per-layer embeddings (12B measured; 31B takes the same path)
  under the ggml and CPU backends, a fully accepted verify past the sliding
  window used to put the evicted positions back over its committed rows, so the
  next verify was 30-47 logits off and 12B streams diverged from plain greedy
  within ten tokens. `SpecOnVerifyAccepted` now restores slots only for a window
  the executor re-forwards; 12B spec and checkpoint-clone streams on an A40 are
  identical to plain greedy again, and n-gram acceptance on the spec prompt went
  from 57% to 89%.
* **Tolerance:** a K+1-row verify and a one-row decode are different kernels. On
  `ggml_cuda` their rows differ by 0.5 logits median on E4B (whose BF16
  per-layer-embedding projection is batch-shape dependent in ggml-cuda) and 1.7-2.2
  on 12B/26B-QAT; on Metal 0.003 (E4B) and 0.15 (12B); on `ggml_cpu`, 0. A greedy
  token whose top-two margin is inside that error can differ: over 512 prose
  tokens on E4B that happened 4-5 times, every one at a margin under 0.25.

Details and the full table: [What greedy parity delivers](../speculative_decoding.md#what-greedy-parity-delivers).

## 13. Output parser and chat template

`Gemma4OutputParser` understands two structural framings:

- **Thinking** — `<|channel>thought ... <channel|>` chain-of-thought, then
  the final answer.
- **Tool calling** — `<|tool_call>call:function_name{...args...}<tool_call|>`
  blocks, which `OutputParser` extracts into structured tool calls regardless
  of the surrounding content. The arguments use Gemma's own syntax (bare keys,
  strings wrapped in `<|"|>`), and the model regularly writes a string value
  bare when it looks like an identifier — `call:read_invoice{invoice_id:INV-472}`,
  `{path:src/main.py}`, `{ids:[INV-1, INV-2]}`. The converter quotes every bare
  value that is not a JSON number / `true` / `false` / `null` (inside arrays
  too; numbers stay numbers). A call whose arguments still do not parse is
  surfaced verbatim as content rather than as an empty message, and each call
  in a multi-call turn carries its own `index` so streaming clients can pair
  the argument deltas.

Chat template falls back to the hardcoded Gemma 4 template when the GGUF does
not ship a Jinja2 one.

### Thought channel with thinking off

No Gemma 4 template primes anything after a tool result when thinking is off:
the canonical E2B/E4B template and the 12B/26B/DiffusionGemma templates all
end the prompt at `<tool_response|>` and let the model continue its own turn
(the larger templates prime a closed `<|channel>thought\n<channel|>` only for a
fresh `<|turn>model`). The renderer keeps that prompt. Priming the closed block
after a tool result anyway was measured on E4B (Metal, Q8_0, skills preamble)
and made it worse: the model wrote its reasoning unmarked, closed the block
again and only then answered, so the reasoning reached `content`.

What E4B does at that boundary (campaign 2026-09-16, B10) is handled while
sampling instead, from the protocol's `ThinkingBudgetOpenToken` /
`ThinkingBudgetEndToken` / `SuppressUnopenedThinkingEndAfter`:

- **A channel the model opens with thinking off is capped.** The budget counts
  from `<|channel>` and closes the channel with `<channel|>` at the first line
  break past a quarter of `max_tokens` (at most 64 tokens, hard stop at twice
  that). The parser hides the thought, the answer follows. Before this the
  `agentic` final turn spent all 256 tokens in the channel and returned empty
  content. Closing at exactly 16 or 64 tokens (mid-sentence) was also measured:
  the model then leaked reasoning or turned the cut-off sentence into a tool
  call. At the first line break past 16, 32, 48 and 64 tokens it answered
  correctly every time.
- **After a tool result, a `<channel|>` with no open channel is masked.** E4B
  wrote the answer, closed a channel it never opened, and wrote the answer
  again. A stream had already sent the first copy, so the client got both. With
  the close masked the model ends the turn after the first copy. Masking needs
  host logits, so these requests do not use the device-argmax path. The mask is
  limited to that boundary: elsewhere a stray close still splits reasoning from
  the answer in the parser.
- With thinking on, the same budget counts from the model's `<|channel>` (or from
  the first token when the prompt primed `<|channel>thought\n` after a tool
  result) and closes it at exactly `TS_THINKING_BUDGET`. Before, Gemma 4 had no
  trained end token declared, and the generic hard stop looked for `</think>`,
  which Gemma never writes. `TS_THINKING_BUDGET=0` disables both caps.

`response_format` combines with `"think": true`: the JSON grammar stays dormant
through the thought channel and arms after `<channel|>`
(`ThinkingGrammarActivationTrigger`). With thinking off it enforces from the
first token, which also excludes `<|channel>`.

## 13a. Tensor parallelism and whole-layer stages

`--tp N` runs per-rank graphs: Q/K/V and gate/up column-parallel, the attention output and down row-parallel, a MoE
layer's fused `[gate | up]` experts split by halves (each rank's slice is `[gate_r | up_r]`, on GGML as its own expert
stack); the `[dense | routed]` partials reduce in one all-reduce before their norms. KV heads must divide by N on every
layer. `--layer-split N` runs one graph per stage; the per-layer embeddings travel from stage to stage beside the stream,
and the stage planner never starts a stage between a KV-shared layer and its source. `--tp T --layer-split P` (the hybrid)
runs every stage's layers as per-rank graphs over its own T devices, the per-layer embeddings carried from stage to stage as
without tensor parallelism (Gemma4RuntimeTests.PlacementMatchesOneDevice, simulated devices).

## 14. Open items

- Direct CUDA E4B decode at 512/128 runs 72-78 tok/s against the legacy model's 77.5-79 on an A40 (12B: 39-40.5 against
  34; 26B-A4B: 80-85 against 66-67). The decode-once kernels take four warps an output row on Ampere (69-72 tok/s at the
  RTX 5090's widths); the rest is the d256 decode attention's split flash kernel and its per-layer mask cells (~5 us a
  local layer over the legacy partition kernel). The other checkpoints and backends are above the legacy model.
- The first long prefill of a fresh 12B / 26B-A4B process on `ggml_cuda` pays the graph compile for its step shape (and the
  lazy expert upload); warm prefill is 1.5-2.4x the legacy model's.
- Metal, Vulkan, MLX and multi-GPU placements were not run on hardware for this migration.
