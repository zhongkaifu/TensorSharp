# Qwen 3.5 / 3.6 family

[← back to model index](README.md) | [中文](qwen35_zh-cn.md)

| Property | Value |
|---|---|
| Provider | Alibaba |
| GGUF architecture keys | `qwen35`, `qwen35moe`, `qwen3next` |
| Source class | [`Qwen35RuntimeModel`](../../TensorSharp.Models/Models/Qwen35/Runtime/Qwen35RuntimeModel.cs) on the shared decoder runtime; [`Qwen35CudaModel`](../../TensorSharp.Models/Models/Qwen35/Qwen35CudaModel.cs) for dense NVFP4 on `--backend cuda` |
| Vision encoder | [`Qwen35VisionEncoder`](../../TensorSharp.Models/Models/Qwen35/Qwen35VisionEncoder.cs) |
| Image processor | [`Qwen35ImageProcessor`](../../TensorSharp.Models/Models/Qwen35/ImageProcessor.cs) |
| Example models | Qwen3.5-9B (dense hybrid), Qwen3.5-35B-A3B / Qwen3.6-35B-A3B (MoE-family), Qwen3.6-27B / Qwen3.8-27B (dense); also [Bonsai2 27B](bonsai2.md) (PRISM PQ2_0 / PTQ1_0) |
| Modalities | Text, image |
| Thinking mode | Yes (`<think> ... </think>`) |
| Tool calling | Yes (`<tool_call>{...}</tool_call>`) |
| Batched / paged forward | **Default ON** — `--no-continuous-batching` forces the per-sequence KV-swap path. Includes a per-slot GatedDeltaNet recurrent-state pool. See §11. |
| MTP speculative decoding | Qwen 3.6 — NextN draft block embedded in the trunk GGUF (no separate file; MTP-retaining GGUFs only, see [Downloads](#downloads)); engage with `--spec` on **either host** — `TensorSharp.Cli` and `TensorSharp.Server` share [`SpeculativeCliFlags`](../../TensorSharp.Runtime/Speculative/SpeculativeCliFlags.cs). GDN recurrent-state snapshot/rollback on partial accept. Engages for solo (non-concurrent) sequences whenever the GGUF retains the NextN block. Qwen 3.8 additionally accepts a **DFlash2** block drafter as a separate `--draft-model` GGUF (§12.4). See §12. |
| Output parser | `Qwen35OutputParser` (inherits `ChatMlOutputParser`) |

## Downloads

Verified GGUF sources (one folder per model below — the 9B and 3.6 repos both
name their projector `mmproj-F16.gguf`, so they must not share a directory):

| Model | HF repo | Recommended file | Vision projector |
|---|---|---|---|
| Qwen3.5-9B (dense hybrid) | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | `Qwen3.5-9B-UD-Q4_K_XL.gguf` (5.966 GB) or `Qwen3.5-9B-Q8_0.gguf` (9.528 GB) | `mmproj-F16.gguf` (0.918 GB) |
| Qwen3.5-35B-A3B (MoE) | [ggml-org/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/ggml-org/Qwen3.5-35B-A3B-GGUF) | `Qwen3.5-35B-A3B-Q8_0.gguf` (36.903 GB) | `mmproj-Qwen3.5-35B-A3B-Q8_0.gguf` (0.614 GB) |
| Qwen3.6-35B-A3B with NextN / MTP (MoE) | [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.663 GB) or `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf` (11.819 GB) | `mmproj-F16.gguf` (0.899 GB) |
| Qwen3.8-27B in NVFP4 (dense hybrid, 4-bit FP) | [akopytko/Qwen3.8-27B-NVFP4-GGUF](https://huggingface.co/akopytko/Qwen3.8-27B-NVFP4-GGUF) | `Qwen3.8-27B-NVFP4-MTP-N4_0.gguf` (15.716 GB, self-contained NVFP4) | `mmproj-BF16.gguf` (0.931 GB) |

> **Qwen 3.6 MTP:** the NextN draft block (GGUF metadata key
> `nextn_predict_layers`) is retained **only** in the GGUFs from
> `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`. The base-repo
> [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
> files (same file names) strip it — on those, `--spec` falls back to standard
> decode, with a one-time warning that the checkpoint carries no draft head
> (`--spec-type ngram` still works there, since it needs no trained drafter
> weights).

> **NVFP4 (GGML type 40):** 4-bit E2M1 values with UE4M3 per-16 sub-block
> scales (36 bytes / 64 weights, 4.5 bpw), NVIDIA's Blackwell-native FP4
> format. Two GGUF flavors exist: self-contained files (all scale factors
> folded into the blocks, e.g. the repo above) and checkpoint-converted files
> that carry extra 1-element `blk.N.<proj>.scale` tensors (HF
> `weight_scale_2`), which TensorSharp multiplies into the projection outputs
> exactly as llama.cpp does (`.input_scale` sidecars are calibration metadata
> and are ignored at inference). On sm_120 GPUs prefill uses ggml-cuda's
> native block-scaled FP4 tensor-core path.

The conversion cards identify the corresponding official Qwen repositories as
their base models; those upstream cards declare Apache-2.0.

```bash
python -m pip install -U huggingface_hub
hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-UD-Q4_K_XL.gguf --local-dir models/qwen3.5-9b
hf download unsloth/Qwen3.5-9B-GGUF mmproj-F16.gguf --local-dir models/qwen3.5-9b
hf download ggml-org/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-Q8_0.gguf --local-dir models/qwen3.5-35b-a3b
hf download ggml-org/Qwen3.5-35B-A3B-GGUF mmproj-Qwen3.5-35B-A3B-Q8_0.gguf --local-dir models/qwen3.5-35b-a3b
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir models/qwen3.6-35b-a3b-mtp
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF mmproj-F16.gguf --local-dir models/qwen3.6-35b-a3b-mtp
```

CLI one-shot (the text prompt comes from a file via `--input`; default sampling
is greedy and the default `--max-tokens` is 100 — raise it for real answers):

```bash
printf '%s\n' 'Describe this image.' > prompt.txt
dotnet run --project TensorSharp.Cli -c Release -- --model models/qwen3.5-9b/Qwen3.5-9B-UD-Q4_K_XL.gguf --mmproj models/qwen3.5-9b/mmproj-F16.gguf --input prompt.txt --image photo.png --max-tokens 512
```

MTP demo. `--spec` and the rest of the `--spec-*` family are parsed by the same
[`SpeculativeCliFlags`](../../TensorSharp.Runtime/Speculative/SpeculativeCliFlags.cs)
on **both hosts**, so `TensorSharp.Cli` and `TensorSharp.Server` cannot drift on
names, validation or defaults. They have to be on the command line that *loads* the model, because
`--spec-draft` also sizes the native graph cache at load time:

```bash
# server
dotnet run --project TensorSharp.Server.Host -c Release -- --model models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda --spec

# CLI — speculation engages on --input, --input-jsonl, --multi-turn-jsonl and --interactive
dotnet run --project TensorSharp.Cli -c Release -- --model models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --backend ggml_cuda --spec --spec-draft 4 --input prompt.txt --max-tokens 512
```

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
same `Qwen35RuntimeModel` (§ 7); the MoE variants add routed experts to its FFN.

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

#### Positions after an image: the M-RoPE delta

An image whose merged grid is H x W takes H*W rows of the KV cache but only
max(H, W) rotary positions. The injector lays the prompt out the way HF and
SGLang `get_rope_index` do (`ModelMultimodalInjector.LayoutQwenVLPrompt`):
the tokens of the span sit at `(base, base + h, base + w)`, and the text after
it resumes at `base + max(H, W)`. Every token past that position table sits at

```
rope position = KV index + delta,   delta = max(last table row) + 1 - prompt length
```

which is SGLang's `mrope_position_delta` (decode position `delta - 1 + seq_len`).
The delta is per sequence ([`Qwen35RopePositions`](../../TensorSharp.Models/Models/Qwen35/Qwen35RopePositions.cs)):

- **Settled before any kernel runs.** `BeginRopePositions` takes it from the staged
  table of a prefill chunk; a forward from KV index 0 without a table starts a new
  history at delta 0; any other forward continues the active sequence's delta. A
  chunk whose rows are all text at `KV index + delta` drops the table and runs the
  ordinary scalar-RoPE graphs, which rotate identically.
- **Used by every path.** The whole-model decode (`rope_pos_delta`), the fused
  prefill/verify graph (`rope_pos_delta` for scalar rows), the arena batched decode
  (`rope_positions` per holder), the paged batched forward (the request's delta
  from the injector), the per-op attention, the direct-CUDA prefill and decode
  graphs, tensor parallelism and the MTP draft head. The per-layer native attention
  kernels rotate at their KV index, so a sequence past an image does not take them.
  (A DFlash drafter keeps its own
  positions; they only affect how many drafts are accepted, never the output.)
- **Stored with the state it describes.** Each per-request holder carries it through
  swaps, retention, re-keying and pooling, a shared-prefix checkpoint and its clone
  copy it, the per-block KV snapshot of the KV-swap concurrency path ends with it (like
  the recurrent state, as of the block's end; an injected block whose size does not fit,
  or whose delta would put the next token at a negative position, is refused before
  anything is written, so a refusal leaves the model holding exactly the blocks before
  it), and the checkpoint file format (`Q5KC`) is **version 2**, which writes it
  after the row count. A version-1 file names no delta and is refused on import; the
  engine logs that the saved checkpoint does not fit, prefills the prefix and saves it
  again.

Before this, decode, speculative verify and every scalar path rotated at the KV index
itself. Two things followed:

- **Single requests.** The reply to an image was generated at positions the model was
  not trained to see after that image: past the prompt, each token sat further along
  than the reference position by the image's token count minus max(H, W) (a 1253x836
  test picture: about a thousand positions). **Outputs after an image change** with this
  fix: they are now what a correct Qwen-VL implementation produces. Measured on the Mac
  (Metal, Qwen3.5-9B-Q8_0, greedy, 96 tokens), the image turn's reply before (584f8f71,
  the same decode as 6db6dbf6) and after:
  - OpenAI, image on turn 1: `...sitting gracefully against a futuristic, glowing
    background filled with floating cubes and digital particles.` became `...sitting
    gracefully amidst a futuristic, glowing digital landscape filled with floating cubes
    and light trails.`
  - Web UI, image on turn 3: `...amidst a futuristic digital environment.` became
    `...amidst a futuristic digital landscape.`
  - OpenAI, image on turn 3: `This digital artwork features an anime-style woman...`
    became `This image features an anime-style illustration of a young woman...`
  - Web UI, image on turn 1: unchanged over 96 tokens.
- **Reuse.** Preserving the per-sequence rotary delta lets a follow-up continue an
  exact cached endpoint past an identical image or video span. Radix keys check
  media identity and span boundaries as well as tokens; changed media or another
  conversation scope cannot claim that conversation's state. GDN state still
  cannot rewind, so a shorter partial match must prefill.

**Validation.**

- `Qwen35MRopeReferencePositionTests` compares the prompt layout, the delta, the
  model's own chunked-prefill and decode positions, and a follow-up turn laid out from
  scratch, against a fixture generated from SGLang's `get_rope_index(model_type="qwen3_5")`
  and its decode rule (`eng/validation/qwen35_mrope_reference`, SGLang 2733afe5) for one
  image, two images, tall and wide grids, a two-pair video and plain text. With decode
  at the KV index (the old rule) 10 of its 26 cases fail.
- `Qwen35ImageFollowUpExactnessTests` (model-gated, `TS_TEST_MODEL_DIR`) runs a
  conversation text -> image -> text -> text on Qwen3.5-9B-Q8_0. In the engine, turns
  3 and 4 must reuse the previous turn past the image, and every turn's greedy tokens
  are compared with a cold engine's, also with a text conversation decoding beside it
  (per-request holders and the arena batched decode). On the model directly, a turn
  built by decoding the previous reply and prefilling the new suffix is compared with a
  cold prefill of the whole prompt, logits step by step for 24 steps, and the same
  comparison runs on a text-only conversation as a control. A checkpoint taken after
  the image survives export, import and clone with bit-identical decode logits, and a
  version-1 file is refused. With the delta disabled the test fails: on Metal the
  image conversation's worst logit difference rises from 0.024 to 3.24 while the text
  control stays at 0.010, and on CUDA from 2.39 to 8.11 (8.8x its 0.92 control) with a
  turn-4 token change that is not a tie (on an earlier revision of the test it failed on turn 3's
  tokens: `...there is no roof visible. The scene depicts...` reused vs `...features...`
  cold).
  The measurements below used a real photo (`TS_TEST_QWEN35_IMAGE`). Without one the
  test draws a synthetic picture: 448x336 for the direct comparisons and 896x672 for the
  concurrent case (the radix tree retains a finished request from 32 tokens,
  `MinRetainTokens`).

**Logit tolerance.** Reuse and cold are not bit-identical: the reused turn's reply rows
were written by the decode graph and the cold turn's by the prefill graph (different
attention and matmul kernels, quantized-activation matmuls on CUDA, NeoX vs interleaved
M-RoPE on equal axes). The test therefore checks three things:

- the worst max |dlogit| over the vocabulary stays under the backend's tolerance
  (`TS_TEST_QWEN35_LOGIT_TOLERANCE`; default 0.1 on Metal, 3.0 elsewhere);
- it is at most 4x the text-only control's (`TS_TEST_QWEN35_CONTROL_RATIO`): the image
  may amplify the kernel difference through its longer context, but must add no error
  of its own;
- greedy tokens are identical, except at a step where the cold run's top-2 margin is
  smaller than the logit difference measured there - a tie the two kernels may break
  either way.

Measured (24 steps, turns 2-4):

| Backend | Image conversation worst \|dlogit\| | Text control | Ratio | Token differences | Checkpoint round trip |
|---|---|---|---|---|---|
| Metal (M-series, Qwen3.5-9B-Q8_0) | 0.024 | 0.010 | 2.3x | none | 0.0 |
| CUDA (A40, Qwen3.5-9B-Q8_0, mmproj F16) | 2.39 | 0.92 | 2.6x | turn 3 step 0 (margin 0.034 < 0.47) and turn 4 step 1 (margin 0.13 < 0.29): ties | 0.0 |

On CUDA the decode and prefill kernels differ by up to about one logit even for a
text-only conversation, so over 24 greedy tokens a low-margin token can flip between a
reused and a cold turn with or without images.

The concurrent test (holders and the arena) applies the same tie rule. At the first step
where a request's tokens differ from its cold engine run it captures the cold logits (the
same prompt prefilled cold straight on the model, which must have decoded the same tokens
up to that step) and accepts the difference only when their top-2 margin is below the
backend's logit tolerance; the rest of that request is then not compared. Its text-only
control opens with a system prompt longer than one block (a 346-token first turn): with
the one-line prompt every text turn was shorter than a block, so under the legacy path
no turn was retained, and
the control reused nothing. It must now reuse everything the previous turn left.
Measured 2026-09-17:

| Backend | Picture | Text turns 2-4 reused | Token differences |
|---|---|---|---|
| Metal (Qwen3.5-9B-Q8_0, mmproj BF16) | built-in 896x672 | 367 / 411 / 446 | none |
| Metal | real photo | 367 / 411 / 446 | none |
| CUDA (A40, Qwen3.5-9B-Q8_0, mmproj F16) | built-in 896x672 | 366 / 410 / 445 | image turn 3 step 0 (margin 0.23), image turn 4 step 5 (0.017), text turn 3 step 0 (0.041): ties |
| CUDA | real photo | 366 / 410 / 445 | text turn 3 step 0 (margin 0.041): tie |

Before the tie rule the CUDA built-in-picture run failed on its image turn 3 step 0.

**Through the server** (the Phase 0 IMG probe: Web UI and OpenAI conversations with the
image on turn 1 or turn 3, plus text controls; Metal, Qwen3.5-9B-Q8_0, greedy, 96 tokens
per turn), every turn after an image now continues the cache:

| Scenario | Turns after the image | Reused before (clamp) | Reused now | TTFT (Web UI) / wall (OpenAI) before | now |
|---|---|---|---|---|---|
| Web UI, image on turn 1 | 2, 3, 4 | 0, 0, 0 | 1,099 / 1,187 / 1,286 (98.0-98.2%) | 1.00 / 1.10 / 1.18 s | 0.13 / 0.13 / 0.13 s |
| Web UI, image on turn 3 | 4 | 0 | 1,340 (98.2%) | 1.20 s | 0.13 s |
| OpenAI, image on turn 1 | 2, 3, 4 | 0, 0, 0 | 1,099 / 1,188 / 1,286 (97.9-98.2%) | 3.67 / 3.64 / 3.31 s | 2.20 / 2.39 / 1.41 s |
| OpenAI, image on turn 3 | 4 | 0 | 1,301 (98.1%) | 2.18 s | 0.94 s |

All 24 replies of that run equal a run of the same build with prompt reuse off
(`TS_SCHED_PREFIX_CACHE=0`), the image was encoded once, and the text controls are
unchanged.

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

`FuseAttentionProjectionWeights()` fuses per-layer Q+K+V into
`attn_qkv.weight` when all three GGUF tensors share a quantization type.
Mixed-type importance-matrix/UD quant projections remain separate, avoiding
multi-gigabyte FP32 expansion. `FuseRecurrentInputWeights()` packs five recurrent projections
(`attn_qkv`, `attn_gate`, `ssm_beta`, `ssm_alpha`, plus an empty slot) into
one fused `ssm_in_proj.weight` for GatedDeltaNet layers. `FuseGateUpWeights()`
fuses dense FFN gate and up.

## 7. TensorSharp implementation

Every load goes through [`Qwen35Architecture`](../../TensorSharp.Models/Models/Qwen35/Qwen35Architecture.cs):

| Load | Served by |
|---|---|
| `--backend cuda`, one GPU, a dense NVFP4 checkpoint, no `--tp` / `--layer-split` / host experts | [`Qwen35CudaModel`](../../TensorSharp.Models/Models/Qwen35/Qwen35CudaModel.cs), the direct-CUDA engine (`TS_Q35_NATIVE=0` declines it) |
| everything else: every backend (`cpu`, `ggml_cpu`, `ggml_metal`, `ggml_cuda`, `ggml_vulkan`, `mlx`, `cuda`), one device, `--tp`, `--layer-split`, their hybrid, a multi-node pipeline, host experts | [`Qwen35RuntimeModel`](../../TensorSharp.Models/Models/Qwen35/Runtime/Qwen35RuntimeModel.cs) on the [shared decoder runtime](../design/unified-execution.md) |

The family's own model class (`Qwen35Model`, its fused whole-model GGML graphs, the slot-stable
arena, the persistent verify graphs and their native kernels) was removed in 2026-11; the runtime
serves every placement it served.

`Qwen35RuntimeModel` is the family's recipe on the runtime:

- **Gated attention.** `attn_q` emits `[q_h | gate_h]` per head; per-head Q/K RMSNorm runs before the
  partial NeoX rotary (M-RoPE sections after an image), and the attention output is multiplied by
  `sigmoid(gate)` before `attn_output`.
- **Gated DeltaNet as a recurrent mixer.** The in-projection (`attn_qkv`, `attn_gate`, `ssm_beta`,
  `ssm_alpha`) is packed into one product where the four share a quantized type; then `ssm_conv1d`,
  q/k L2 norms, the delta rule (`GatedDeltaNet`, ggml's `ggml_gated_delta_net` on the GGML backends,
  the runtime's own kernels on direct CUDA, MLX and the managed CPU), the gated RMSNorm with `silu(z)`
  and `ssm_out`. The conv window and the delta state live per sequence slot (and per rank under
  `--tp`).
- **FFN.** Dense SwiGLU with gate/up fused where their types match, or routed experts with a shared
  expert and its sigmoid gate.
- **Images.** [`Qwen35VisionEncoder`](../../TensorSharp.Models/Models/Qwen35/Qwen35VisionEncoder.cs)
  embeds the patches, the rows are injected at the image span, and the text positions after it
  follow the compressed M-RoPE delta ([`Qwen35RopePositions`](../../TensorSharp.Models/Models/Qwen35/Qwen35RopePositions.cs)).
- **Checkpoint variants.** Bonsai2's PRISM rotations run on the GGML backends (the graph builder
  rotates each rotated product's input; other backends and `--tp` refuse the load). NVFP4 sidecar
  scales multiply their projections, except on routed experts, which the runtime refuses.

Each step is lowered to one graph: a whole-forward GGML graph on the GGML backends (decode steps
replay the previous step's graph with new inputs), a program for the tensor executors on direct
CUDA, MLX and the managed CPU. The golden-logit fixtures in
[`Fixtures/Qwen35Legacy`](../../InferenceWeb.Tests/Fixtures/Qwen35Legacy/README.md) hold the
removed implementation's rows, and `Qwen35RuntimeTests` keeps the runtime within each backend's
rounding of them.

## 8. Prefill

A prompt chunk is one graph over every layer. Full-attention layers write their K/V rows by index
and attend causally (flash attention where the backend has a kernel for the head size); the
recurrent layers run the chunked delta rule over the chunk. Prompt chunks are 2,048 rows by default
(`TS_PREFILL_CHUNK`). A prompt with an image runs the vision encoder first and injects its rows.

## 9. Decode and backend notes

Decode is one graph a step for every running sequence together: each sequence's row reads its
own slot's K/V window and recurrent state. On the GGML backends the next step replays that graph
with the positions, write rows and masks rewritten in place. On ggml_metal a decode row's
recurrence updates its state in place.

### Direct-CUDA NVFP4 route: LM head, verify GEMMs and prefill GEMMs

`--backend cuda` on the NVFP4 GGUF (`Q35CudaEngine`) reads every weight once per step, so a
step costs what its kernels achieve of the RTX 5090's memory bandwidth. Three kernels fell
well short and were replaced in 2026-11. Each computes the same values bit for bit (checked
on every shape against the previous kernel); the answers did not change.

| Kernel | Before | Now (RTX 5090) |
|---|---|---|
| LM head, Q6_K 248,320 x 5,120 (the target head and the DFlash2 draft head, each step) | 210-byte superblocks read 2 bytes at a time, one superblock per `__syncthreads`: 0.83 TB/s, 1.27 ms at 32 rows | repacked at load into 64-row tiles (`ts_q35_repack_q6k_tiled`, the same bytes regrouped) and streamed through a cp.async pipeline: 32 rows 0.74 ms (1.41 TB/s), 8 rows 0.89 -> 0.69 ms, 1 row 1.06 -> 0.87 ms |
| Split-K of the W4A4 verify GEMMs (9-64 rows) | split until the CTAs cover every SM: the 129-tile GDN input projection ran 2 splits in two waves | the most splits that still fit one wave: GDN input projection 38.1 -> 30.6 us, the 5,120-row output projections 18.1 -> 14.3 us at 32 rows |
| Prefill W4A4 GEMMs (65+ rows) | cp.async 128-row tiles, about 830 TFLOPS | the TMA-fed, warp-specialized kernel of the NInfer route, plus an F32 SwiGLU epilogue for gate/up: 1,230-1,390 TFLOPS at 2,048-4,096 rows |

End to end (GGUF + DFlash2 K7, 400-token answers to short prompts): 4 concurrent chats
464 -> 521 tok/s, 2 concurrent chats 284 -> 308 tok/s. Prefill TTFT: an 11K-token prompt
1.51 -> 1.20 s, a 36K-token prompt 6.57 -> 5.38 s, four 3.6K-token prompts at once 1.9-2.1 ->
1.6-1.7 s.

Still open on this route: the 1-8-row W4A8 GEMV reaches 1.17-1.37 TB/s at 8 rows. An int8
tensor-core version that keeps the GEMV's summation order (one accumulator per GEMV lane)
needed 242 registers and ran at 0.3-0.9 TB/s, so it was not kept.

**Two concurrent chats and session turns on a long document.** With two chats a DFlash2 verify
has 16 rows, so the projections leave the 8-row GEMV for the W4A4 GEMM, whose 32-row tile is then
half empty. Profiling showed that this costs nothing: at 16 rows the GEMMs are memory-bound and as
fast as or faster than the 8-row GEMV (gate/up 65 vs 73 us, GDN input projection 31.5 vs 40.5 us),
and a bit-identical 16-row tile gained at most 3%. Three other costs were real and are fixed:

| Cost | Before | Now (RTX 5090) |
|---|---|---|
| Pre-norm + NVFP4 quantization of 9+ rows (two per layer) | one dependent global-memory round trip per element group: 5.5 us | `ts_q35_add_rmsnorm_nvfp4_5120_f32` keeps the row in registers and issues every load first: 3.5 us, bit-identical |
| A short prompt chunk (a session turn, a new chat's suffix) over a long cache | the 8-bit prompt attention, with no key split: a 47-token chunk over 127K cached tokens ran 8 CTAs, 5.3 ms a layer, after 0.65 ms converting the band | the 16-bit kernels, which split the keys across the device, until the 8-bit CTAs cover about 40% of the SMs (about 350 rows; both routes): that turn's first token 140 -> 50 ms |
| Chats admitted in one step on a cached prefix | a request restored from the radix cache has no holder yet, so the batched prefill skipped it and each suffix ran on its own | the executor binds the holders and injects the prefixes first (`PrebindCachedPrefixPrefills`), so the suffixes share one forward: three such chats on a 120K-token document reach their first tokens within 3 ms of each other |

### MLX: UD quants and speculative decoding

Measured on an M5 Pro (48 GB), Qwen3.8-27B GGUF with the DFlash2 Q4_K_M drafter (`--spec-draft 7`),
a 2.3K-token prompt and 512-token answers; speculative numbers are the second request of one
process (kernels compiled).

- **Low-bit types on the device.** Unsloth's UD-IQ3_XXS keeps 1.2 GB of its 10.9 GB in Q2_K, Q3_K,
  IQ2_XS, IQ1_S and IQ1_M, which had no MLX kernels and ran on the CPU managed-dequant path. They
  now have device kernels (`MlxLowBitQuantKernels.cs`, the dequant arithmetic of
  `ManagedQuantizedOps`): decode 1.8 -> 15.7 tok/s.
- **Prompt chunks.** The raw i-quants' 32+-row products dequantize the weight to F16 slices and run
  MLX's GEMM (`TS_MLX_IQ_DEQUANT_GEMM_MIN_ROWS`), as Q6_K and IQ4_XS do: the 2.3K-token prefill
  361 s -> 6.05 s (ggml_metal: 7.0 s).
- **Verify windows.** 2-31 rows take the few-row kernel (`TS_MLX_IQ_FEWROW_MAX_ROWS`), now laid
  out as ggml-metal's few-row MMA kernels: the simdgroups of a threadgroup split K, each lane
  decodes its 16 weights of a chunk once straight into simdgroup-matrix registers (F32), and the
  partial tiles are summed once at the end. The kernel it replaces staged 64 decoded columns in
  threadgroup memory with a barrier pair per 64 weights; even with a near-free decoder it took
  106 of an 8-row forward's 128 ms. An 8-row forward 128 -> 79 ms (ggml_metal 87), 2 rows
  159 -> 75, 16 rows 212 -> 151 (ggml_metal 153), 31 rows 383 -> 279; the verify window 155 ->
  92 ms (ggml_metal 88). IQ4_XS and Q6_K decode their 8 values with one scale read instead of
  value by value.
- **No re-forward after a rejection.** The verify keeps the recurrent state after every row and the
  accepted row's state is committed on the device, as the ggml verify's snapshots do; the
  pre-verify snapshot is a reference to the immutable MLX arrays instead of a host copy (6 ms ->
  0.1 ms a step). A rejection used to restore and forward the accepted prefix again: 135 ms each.
- **Drafter.** The DFlash2 block convolution runs on the device and the sliding-window mask is
  added, instead of reading coefficients, rows and scores back four to five times per layer: a
  draft 51 -> 32 ms.
- **Decode matvec.** The one-row kernel of the codebook i-quants and the low-bit types gave every
  weight row its own 256-thread threadgroup, which re-read the whole activation row (20 KB at
  K = 5,120) and ended in a barrier. It now has ggml-metal's mul_mv layout: two simdgroups of four
  weight rows each, a lane loading 32 activations once for all four rows. Plain decode
  15.7 -> 20.4 tok/s (49 ms a token), the rate of ggml_metal (20.6) and llama.cpp (20.5) on the file.
- **Short prompts and wider windows on the matrix units.** Past 8 rows that kernel is bound by
  its F32 simdgroup products (an 8-row forward is ~430 GFLOP), and from 32 rows the dequant GEMM
  wrote the whole weight as F16 on every call (~590 ms whatever the row count). On M5 the same K
  split now multiplies on the tensor API: each simdgroup decodes its chunk of 32 weight rows into
  a tile of its own and multiplies it against 16, 32 or 64 input rows (`TS_MLX_FEWROW_TENSOR`,
  9-1024 rows). A forward of 16 rows 151 -> 82 ms (ggml_metal 153), 32 rows 599 -> 104 (291),
  128 rows 723 -> 325 (363), 512 rows 1377 -> 1205 (1289); it ties the dequant GEMM at 1024 rows,
  which keeps the longer chunks.
- **Affine weights: the drafter.** Q4_K and Q8_0 load as MLX affine weights, which ran through MLX's
  `quantized_matmul`: at the drafter's 7-8 rows it read them at ~120 GB/s. They now take the same
  few-row kernels with an affine decoder (`TS_MLX_AFFINE_FEWROW_MAX_ROWS`, up to 128 rows with the
  tensor API): a DFlash2 draft 17.6 -> 13.4 ms (layers 11.0 -> 9.0, the 248K-row head 5.9 -> 3.7),
  and the verify window 92-96 -> 89 ms from the target's own Q4_K tensors.
- **Unrolled register arrays.** The first port of that layout took 225 ms for the 8-row forward:
  the loops over its accumulator and pointer arrays were not unrolled, so the arrays lived in
  thread memory. With every such loop unrolled (`clang loop unroll(full)`, ggml's `FOR_UNROLL`)
  it takes 79 ms. Tiling, by contrast, hardly matters on the M5 Pro (79-81 ms for every shape
  tried; eight weight tiles 85 ms).
- **Retained conversations on a 48 GB Mac.** Each retained conversation is sized for the whole
  context (about 2.5 GB at 32K), and sixteen of them took the server past 34 GB and into swap,
  where every scenario ran 3x slower. Retained state is now bounded by what macOS can give
  (see [the batching doc](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md)): the full benchmark
  below stayed at 18-23 GB.

Server, greedy, `--spec-draft 7 --spec-pmin 0`, MAX_CONTEXT 32768, two runs each (TTFT / decode):

| UD-IQ3_XXS + DFlash2 | ggml_metal | MLX before | MLX now |
|---|---:|---:|---:|
| Long generation (74-token prompt, 1,220 tokens) | 0.57-0.60 s / 30.1-31.3 tok/s | 1.27-1.43 s / 17.6-17.7 | 0.34-0.37 s / 31.0-31.6 |
| 4K-token prompt | 12.9 s / 35-52 | 10.6-10.7 s / 15-18 | 9.8-10.0 s / 39-41 |
| 16K-token prompt | 53.5 s / 33 | 43.1-43.3 s / 14-15 | 41.1 s / 29 |
| Multi-turn: first turn, then cached turns | 9.2-9.3 s, 0.32-0.59 s | 8.4-8.8 s, 0.56-0.67 s | 6.4-8.5 s, 0.12 s |
| Short chat | 0.57 s | 0.66-1.05 s | 0.21 s |

Plain decode is 46.8 ms a token on MLX against 48.4 on ggml_metal. Decode after a 16K-token prompt
(29 against 33 tok/s, over 56 tokens) is the one figure still behind.

## 10. Memory and KV cache strategy

- **Full-attention layers**: a K/V cache per slot, `[kv_heads, capacity, head_dim]`, in the
  `--kv-cache-dtype` type (`f32`, `f16`, `q8_0`, `q4_0`). A slot starts small and grows on demand
  up to the context length; idle retained slots give their device copies back before a step would
  run out of device memory.
- **Recurrent layers**: the conv window `(conv_kernel - 1) x qkv_dim` and the delta state
  `[v_heads, head_v_dim, head_k_dim]` per slot. A recurrent state cannot rewind, so a slot truncates
  only to its length or to zero.
- **File-mapped weights**: quantized tensors bind the GGUF mapping instead of a copy where the
  backend can read it (GGML CUDA, Metal, CPU, integrated GPUs).

## 11. Continuous batching and prefix reuse

Concurrent requests run on their own slots; a step batches every running request's rows (prompt
chunks of different lengths and decode rows together). The radix prefix cache reuses a finished
turn's retained slot for the next turn of the same conversation, and shared-prefix checkpoints (a
copy of the K/V rows and the recurrent state at a boundary) let a new request start from a common
prefix. Because the state cannot rewind, the automatic checkpoint of a prompt sits one token before
its end, so the same prompt can be served again. `Qwen35RuntimeTests` checks batched slots against
each request alone, concurrent requests through the engine in every placement, and checkpoint
clones and files continuing exactly.

## 12. Speculative decoding

The runtime verifies a window of drafted tokens in one forward and keeps the recurrent state after
every row of it, so a rejection commits the accepted row's state without re-forwarding (the GGML
graphs keep per-row snapshots; the per-op path keeps the window's inputs and replays the accepted
prefix). Drafters:

- **n-gram** (`--spec --spec-type ngram`), on every Qwen 3.5-family checkpoint.
- **NextN / MTP** (Qwen 3.6, Qwen 3.8 27B, `--spec`): the embedded NextN block loads as a draft layer
  on the last stage; an MTP head that ships as its own file loads with `--draft-model`.
- **DFlash / DFlash2** block drafters (Qwen 3.8): `--draft-model <drafter.gguf>`; the drafter reads
  the trunk's captured hidden rows.

Speculation runs on one device, under `--tp` (per-rank graphs) and across a multi-node pipeline (the
driver runs the last stage and drafts there). `--spec-draft` caps the drafts a step takes, and the
engine's governor parks speculation while it does not pay.

## 13. Output parser and chat template

- `Qwen35OutputParser` inherits `ChatMlOutputParser`, so the wire format is
  identical: `<think> ... </think>` for chain-of-thought reasoning, and
  `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` for tool calls.
- Chat template uses the standard Qwen `<|im_start|>` / `<|im_end|>` rolling
  message format with the additional vision token `<|image_pad|>`.
- `ChatTemplate.ExpandImageTokens(inputTokens, imagePadId, tokenCounts)`
  expands each `<|image_pad|>` placeholder into the right number of placeholder
  tokens for the corresponding image, and the multimodal injector then writes
  the encoded embeddings into those positions before `Forward()`.
- **Thinking budget.** With thinking on, a reasoning block that reaches
  `TS_THINKING_BUDGET` (default 75% of `max_tokens` from 512 up) is closed and
  the answer follows inside `max_tokens`, in the server and in the interactive
  CLI. `</think>` is one trained token (248069), and ahead of it the host writes
  Qwen's published hand-over sentence ("Considering the limited time by the
  user, I have to give the solution based on the thinking directly now."), which
  shows at the end of the reasoning. Before 2026-09-29 the family had no closing
  token: such a turn was stopped with an EMPTY answer (`finish_reason`
  `thinking_budget`), and a bare `</think>` left Qwen3.5-9B reasoning on inside
  its answer. Qwen3.5-9B IQ4_XS on Metal, `max_tokens` 600, two concurrent
  three-turn conversations: 0/2 passed before (empty answers), 2/2 after.
- **Reasoning that ends without an answer** (issue #338). Qwen3.8-27B NVFP4
  under Aider ended 3 of 53 turns on a stop token in the middle of its
  reasoning: each had settled into a counting loop (`Need maybe tests expect
  to_string for 1:1533 -> 02:33. Good.`, then `1:1534`, for thousands of
  tokens), and an OpenAI client received an empty `stop`. Three changes, for
  every family with a trained closing token, whatever `TS_THINKING_BUDGET` is:
  - a stop token drawn while the reasoning is open closes it instead - the
    hand-over sentence, then `</think>` - and the model answers
    (`ThinkingTokenBudget.CloseOnStop`, applied where every token is drawn:
    host sampling, the device-argmax stash and each speculative verify row);
  - reasoning that repeats a templated line 32 times in a row (digit runs
    folded, up to a fifth of the line changed, so `twenty-nine` to `thirty`
    still repeats: `ReasoningLoopDetector`) is closed the same way. Templated
    lines in the answer are never touched;
  - a turn that still ends with reasoning and no answer (a family without a
    closing token) reports `thinking_budget`, which clients see as `length`.

  On Qwen3.8-27B UD-IQ3_XXS (Metal), the issue's loop primed into the
  reasoning was carried on by the model and closed after 32 generated lines,
  and the answer was right (`02:33`); a stand-in stop mid-reasoning was closed
  and answered `391` for 17 x 23.

## 13a. Multi-GPU placements

| Request | What runs |
|---|---|
| `--tp N` | per-rank graphs where the executor runs ranks (`ggml_cuda`, direct `cuda`): each rank its heads, its shard of every projection and its recurrent state; a rank owns a block of key heads and the value heads that read them, so only `ssm_out`'s and the FFN's partial sums are reduced. Elsewhere each rank's Gated DeltaNet block runs as one native graph (`TSGgml_Qwen35GdnLayerTP`) and the rest per op |
| `--layer-split P` | whole-layer stages, one graph per stage on its device |
| `--tp T --layer-split P` | stages of T ranks each |
| multi-node `--layer-split` | each node loads its stages (which may be `--tp` groups of their own size); speculation drafts on the driver |
| `--n-cpu-moe N` / `--cpu-moe` | the routed experts of those layers on the host |

Concurrent requests stay batched in every placement. `SharedRuntimeConformanceTests` and
`Qwen35RuntimeTests` check tensor parallelism, layer split and their hybrid against one device on
simulated devices; the GPU lanes run the same checks on two real devices.

## 14. Status against the removed implementation

Measured 2026-11-12, latest `main` (`032fdf24`, `Qwen35Model`) against the runtime, the same CLI
script on both (warm-up, then each prompt twice: "short" is 55 tokens, "4K" 3.6K tokens, 128
generated tokens, greedy), alternating builds. Prefill ms / decode tok/s:

| Placement | Model | `main` | runtime |
|---|---|---|---|
| ggml_vulkan, RTX A5000 | Qwen3.5-0.8B Q8_0, short | 39-41 / 271-278 | 45-47 / 323-327 |
| | 4K | 299-313 / 126-146 | 354-359 / 276-278 |
| | Qwen3.5-9B Q8_0, short | 152-222 / 61 | 175-198 / 62 |
| | 4K | 1,777-1,911 / 53-54 | 1,808-1,825 / 63 |
| `--tp 2` ggml_cuda, 2x A40 (host-staged transfers) | Qwen3.5-9B Q8_0, short | 105-108 / 94-95 | 112-114 / 85-86 |
| | 4K | 903-934 / 91 | 977-988 / 84 |
| | Qwen3.6-35B-A3B UD-Q4_K_M, short | 127-130 / 107-112 | 164-176 / 106-108 |
| | 4K | 950-979 / 105-106 | 900-917 / 104 |
| `--tp 2` direct cuda, 2x A40 | Qwen3.5-9B Q8_0, short | 351-376 / 23 | 239-310 / 43-45 |
| | 4K | 14,923-15,171 / 17-18 | 6,354-7,413 / 43-45 |
| direct cuda, RTX A5000 (not NVFP4) | Qwen3.5-0.8B Q8_0, short | 13 / 323-324 | 38-39 / 154-163 |
| | 4K | 363 / 298 | 686-687 / 158-187 |
| | Qwen3.5-9B Q8_0, short | 51 / 66 | 111-140 / 63 |
| | 4K | 1,431-1,436 / 65 | 2,007-2,019 / 63 |
| managed cpu, M5 Pro, 64 tokens | Qwen3.5-0.8B Q8_0, short | 311-367 / 105-108 | 290-300 / 73-75 |

The runtime's ggml_cuda, ggml_metal, ggml_cpu and MLX single-device numbers were measured when those
routes moved ([unified execution](../design/unified-execution.md)). Open gaps:

- **Direct CUDA on one GPU** (checkpoints the direct engine does not serve): the runtime's CUDA tensor
  executor launches each kernel from the host; small models are bound by that (Qwen3.5-0.8B decode
  at about half of `main`), and its prompt GEMMs trail the removed fused path (9B 4K prefill +40%).
- **Managed CPU**: decode about 30% behind on Qwen3.5-0.8B: the matmuls take the same time; the
  remaining cost is the executor's per-node work.
- **`--tp 2` decode on Qwen3.5-9B**: 8-10% behind (the 35B MoE is level).
- **ggml_vulkan prompts on Qwen3.5-0.8B**: about 15% slower (decode is faster).

