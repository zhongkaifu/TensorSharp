# Qwen 3.5 / 3.6 family

[← back to model index](README.md) | [中文](qwen35_zh-cn.md)

| Property | Value |
|---|---|
| Provider | Alibaba |
| GGUF architecture keys | `qwen35`, `qwen35moe`, `qwen3next` |
| Source class | [`Qwen35Model`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.cs) (single-sequence) + partial in [`Qwen35Model.GatedDeltaNet.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.GatedDeltaNet.cs) + [`Qwen35Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.BatchedForward.cs) (`IBatchedPagedModel`) |
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
The delta is per sequence (`Qwen35Model.RopePositions.cs`):

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

- A fused whole-model GGML graph for supported dense single-device GPU
  prefill and decode. On Metal, this is the default path for the 27B dense
  Qwen 3.6 model.
- The fused per-layer attention decode kernel
  (`TryFusedAttnLayerDecode`) at every cached sequence length.
- The fused prefill attention kernel (`FusedPrefillAttention`) for
  multi-token prefill on a GGML backend.
- The fused output-projection + FFN kernel (`FusedOutProjFFN`) for both
  attention and recurrent layers with dense FFN.
- The fused output-projection + post-norm + router kernel
  (`FusedOutProjNormRouter`) for recurrent layers feeding into MoE.
- The batched MoE kernel (`MoEExpertsSwiGLUResidual`) for MoE FFN decode.
- A standard managed C# path otherwise.

## 8. Prefill optimization

### Whole-model fused prefill

On supported dense GGML GPU models, prefill is dispatched through
`TSGgml_Qwen35ModelVerifyOwned`: one graph evaluates every FullAttention and
GatedDeltaNet layer, final RMSNorm, and the LM head. Intermediate activations
remain device-resident, mixed-quant Q/K/V projections keep their native
quantized representations, and logits are copied directly into the managed
output buffer. Unsupported layouts continue through the per-layer path below.

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
round-trips collapse into one.

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

### Fused QK-Norm + RoPE (direct CUDA)

On the direct `cuda` backend, the text-only prefill path fuses the per-head QK
RMSNorm and NeoX RoPE into one CUDA kernel (`ts_qk_norm_rope_neox_f32`). Default
on; set `TS_FUSED_QKNORM_ROPE=0` to fall back to separate Norm + RoPE ops
(multimodal MRoPE and non-CUDA backends always use the separate path).

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
The whole-encoder fast path fuses all blocks into one graph. Its default CUDA
attention writes each query tile into one output allocation, avoiding repeated
copies of preceding tiles while preserving the existing attention arithmetic.

Validation on 2026-10-03 used the Flash Next BF16 projector, the supplied photo
(7,920 patches / 1,980 tokens), and a single RTX 3080 Laptop GPU (16 GB, WDDM,
CUDA 12.6), against unchanged upstream ggml
`353b63b439f27ab2cc19dac97ab1681ba6d2d084`. All 5,068,800 projected float32
values were bit-identical to the previous encoder. The attention microbenchmark
reduced scratch from 394.8 MB to 343.6 MB (13%). With one warmup and three
standalone encoder samples, median time was 2,947.4 ms versus 2,982.0 ms;
this small, noisy difference does not establish an end-to-end latency gain.

Experimental `TS_QWEN_VISION_F32=1` enables TensorSharp-owned streaming F32
CUDA attention for 72-wide heads. The same standalone encoder measured
2,635.7 ms (11.6% faster than the previous encoder), with 36.5 MB of attention
scratch. Its full-embedding comparison failed the conservative minimum-row
cosine gate: 0.999856 versus a required 0.9999; aggregate relative L2 was
0.002069. It remains opt-in and has no end-to-end model-quality qualification.
The native suites passed 168 numerical cases (78 existing attention cases and
six vision cases on each of CPU and CUDA) plus ten broader CPU regression
targets, without failures or skips. These measurements cover one projector,
photo and GPU; other projectors, videos and devices were not benchmarked.
Reusable checks are `GgmlOpsCudaAttentionPrecisionTest --benchmark-vision
7920 7920 16 3` and
[`compare-vision-embeddings.py`](../../eng/validation/compare-vision-embeddings.py).
Generated evidence stays in ignored `artifacts/qwen-ttft/`.

### Cross-layer caches and parallelism

- `Parallel.For` across block rows in `ReorderToBlockOrder` for spatial
  merge.
- GPU-aware QKV split: on GPU backends `Narrow` + `NewContiguous` keeps data
  on-device; on CPU a fused parallel `Buffer.MemoryCopy` splits Q/K/V in one
  sweep.

### Chunked parallel GatedDeltaNet recurrent prefill

For prefills of at least `GDN_CHUNK_PREFILL_MIN_SEQ_LEN` tokens (default 2 on
ggml_cuda, 6 elsewhere) on a GGML backend the per-token recurrent loop is replaced
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
  steps still use the per-token loop.
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

### Persistent whole-model Metal decode

`TSGgml_Qwen35ModelDecode` retains a complete per-sequence decode graph across
tokens. It includes all 64 transformer layers, final RMSNorm, and the LM head;
the Metal path can also perform `GET_ROWS` directly from the quantized token
embedding table. A 64-token padded attention bucket keeps graph topology
stable, movable `CPY` destinations append K/V without a scatter kernel, and
asynchronous submission combines graph execution and logits readback behind
one synchronization.

GatedDeltaNet K=1 output is laid out as
`[attention output | recurrent state]`. On single-device Metal, the managed
state view and native result share that backing buffer, so the 48 recurrent
layers update state in place instead of issuing a 3 MiB copy apiece. The
native side validates backend, pointer offset, alignment, and exact tensor
geometry before omitting any copy.

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
standalone `FusedMatMulQuantAdd` with one fused dispatch, at every context
length: once the rest of the model runs on the device, the GPU flash-attn
setup cost is amortized even for short contexts (omlx and vLLM likewise run
`mx.fast.scaled_dot_product_attention` unconditionally). The
`FusedRmsNormMatMulQuant` + CPU-SIMD attention + `FusedMatMulQuantAdd` path
remains for the cases the fused kernel declines (e.g. a sequence past an
image, whose M-RoPE delta the kernel does not apply).

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

- **FullAttention layers**: standard KV cache `[numKVHeads, maxSeqLen,
  headDim]` per layer. KV dtype configurable via `--kv-cache-dtype` (`f32`,
  `f16`, `q8_0`, `q4_0`).
- **GatedDeltaNet layers**: `_convState[layer]` float array of size
  `(convKernel - 1) * qkvDim` for the conv1d sliding window, and
  `_deltaStateTensor[layer]` of shape `[numVHeads, headVDim, headKDim]` for
  the SSM hidden state.
- `ResetKVCache()` zeroes both kinds of cache.
- Initial CUDA cache allocation can be smaller than `maxContextLength` and
  grows on demand (printed at startup).
- **The whole-model decode's conv scratch** (`_fdConvScratch`) belongs to each cache
  (the primary and every per-request holder) and is swapped with it. It is allocated
  whenever the active cache has none, reseeded from the GDN host ring, not only when
  the decode descriptors are first built. A cache that had never decoded by then - the
  model's original primary cache, saved aside when a holder was bound first - used to
  keep a null scratch, and its first decode after being restored threw a
  `NullReferenceException` in `TryFullModelDecodeCore`. On the engine that is a freshly
  loaded model whose first scheduled step is a concurrent one, followed later by a solo
  decode on the primary cache. The arena batched decode treats a holder without a
  scratch the same way. `Qwen35ConvScratchTests` (model-gated) covers it.

### Radix prefix reuse

The `Qwen35Model` family, including Qwen3.8 dense models, declares
`DeferPrimaryConversion`: an eligible finished primary stays live for an exact
next turn. The next turn claims that state without allocating another cache.
Only a request that displaces the primary attempts to move it into a retained
holder and allocate an empty replacement. Concurrent per-request holders and
shared-prefix checkpoint copies keep their existing retention paths. Every path
preserves attention KV, native GDN state and the per-sequence rotary delta;
reuse remains exact because the recurrent state cannot rewind.

Retained holders and checkpoint copies are extra payloads governed by the radix
tree's count and byte caps. Default device/state caps are resolved from half the
spare memory when the engine is created, then bounded by current spare memory
minus running-request reserves. The live-primary marker adds no extra retained
bytes, so an exact continuation does not depend on enough headroom to allocate a
second primary. Disabling extra holder retention still permits that live cache
to continue while prefix caching is enabled. If displacement cannot retain the
old state, the new request prefills normally. Before allocating, the tree measures
the live primary's eventual holder footprint and declines conversion when an
absolute option or family cap cannot admit it.

Adoption allocates the replacement before publishing the moved primary.
`PrefixCheckpointOwnershipTests.QwenPrimaryAdoptionFailure_…` injects failures at
each of six allocations, checks unchanged live state and allocation ownership,
and verifies a later checkpoint remains independent. The Qwen4Exp native
continuation regression exercises the same ordering. Primary-measurement tests
require estimate/adopted-footprint parity, unchanged state and no tensor allocation,
with invalid lengths and checked-out holders refused. Runtime
`DeferredPrimaryCacheTests` verifies exact continuation, displacement, a zero
extra-retention budget and recovery from a conversion failure with cold-output
parity. These are synthetic state/ownership checks, separate from trained-model
quality and performance validation.

On 2026-10-03, HTTP validation used the supplied
`Qwen3.8-27B-UD-IQ4_XS.gguf`, GGML CUDA, an i7-11800H, 32 GB RAM and one
RTX 3080 Laptop GPU with 16 GB VRAM. Five two-turn conversations used the same
short Chinese prompt, greedy sampling, repetition penalty 1 and a 256-token
limit. All ten replies completed at EOS and matched the original build exactly.

| Web UI request | Original median TTFT | Updated median TTFT | Updated reuse |
|---|---:|---:|---:|
| Initial question | 581.6 ms | 524.8 ms | 0/28 tokens |
| `请继续` | 664.8 ms | 435.5 ms | 55/70 tokens |

The OpenAI-compatible endpoint also completed both turns at EOS with 55/70
tokens reused. A separate three-turn, 128-token dialogue reused 171/187 and
315/331 tokens; an earlier-history branch and a changed independent question
reused zero, and the independent arithmetic answer remained `579`. Longer
cached replies differed from cold-prefill replies, so later differing histories
are not qualified timing or greedy-parity comparisons. The initial long-prompt
sample took 792 ms versus 573 ms in the original run; the repeated short-prompt
results do not establish a latency guarantee for every independent request.

Loading and startup warmup are excluded; GPU clocks, WDDM paging and the OS file
cache were uncontrolled. Upstream ggml was unchanged at
`353b63b439f27ab2cc19dac97ab1681ba6d2d084`. Commands, strict reuse checks and
comparison rules are in [the HTTP benchmark guide](../../eng/validation/README-qwen-chat-cache-benchmark.md).
Generated answers, timings and test results remain in ignored
`docs/validation/qwen-ttft/`; missing external models and the two-GPU case are
excluded from passing coverage.

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
`--no-continuous-batching` forces the per-sequence KV-swap path instead.

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
retires the sequence. Swapping references instead of copying each slot's
state in and out of the per-model scratch buffer avoids roughly **2 MB of
SSM-tensor memcpy plus tens-of-KB conv memcpy per GDN layer per sequence**
on every decode step. Each sequence's slice then runs through the model's
GatedDeltaNet step
([`Qwen35Model.GatedDeltaNet.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.GatedDeltaNet.cs))
against its own slot.

### Multimodal in the batched path

`SupportsBatchedMultimodal` is true: `ForwardBatch` builds
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

### Concurrent decode with K/IQ-quantized embeddings on CUDA

The arena fused decode path supports GGUFs whose token embedding table cannot
use CUDA `GET_ROWS`, including the Q2_K table in
`Qwen3.8-27B-UD-IQ3_XXS.gguf`. TensorSharp gathers only the current token's
embedding row for each request on the CPU, then uploads those F32 rows into
the persistent batched graph. QKV, recurrent input/output, FFN and vocabulary
projections remain batched on the GPU. For three requests with hidden size 5120,
the embedding payload is 60 KiB before arena padding.

This needs both updated managed assemblies and a rebuilt `GgmlOps` library
(`TSGgml_Qwen35ArenaDecodeBatchedHidden`). An older library receives an explicit
decline reason. Embedding types supported by CUDA retain the original native
entry point. Upstream ggml sources are unchanged.

The CUDA graph also preserves the normalized Q/K tensors before RoPE. Solo
decode already separates those operations with a reshape; without the same
boundary, the arena can select CUDA's combined RMSNorm/multiply/RoPE kernel
and diverge from solo decoding. Keeping these two intermediate tensors blocks
that fusion while retaining ordinary RMSNorm/multiply fusion, batched weight
projections and CUDA graph capture. Other models and Metal are unaffected by
this CUDA-only boundary. Numerical agreement still needs validation for each
model/backend/cache configuration; it is not a universal bitwise guarantee.

CUDA attention uses one single-request attention node per arena slot within
that same graph, with each holder's effective attention-window length matching
solo decode. Masking a larger shared window can change CUDA's reduction
partitioning and accumulate numerical drift, especially with quantized caches.
The graph rebuilds when a window changes, flushing and restoring KV and recurrent
state; ordinary steps retain the captured graph. The weight projections still
process the whole batch together. Metal retains its existing batched attention
layout.

The scheduler's decline warning describes one step. A cold model may need one
serial decode step to initialize weight descriptors, and a growing cache may
temporarily be served separately. Persistent failures include their actual
managed/native reason in the server log; successful batching is also logged
once. Disabling batched decode or using unsupported configurations still
selects the existing fallback.

Run the [Qwen35 decode probe](../../eng/validation/Qwen35BatchedDecodeProbe/README.md)
for teacher-forced distribution comparisons, explicit native engagement, reordered requests,
different cache capacities, serial continuation and paired throughput measurements.
It reports raw logit errors, softmax KL, and argmax changes with their reference
margins, and counts actual requests rather than padded GPU lanes. Use the
[two-reviewer HTTP probe](../../eng/validation/probe_qwen35_reviewers.py)
separately to validate a [sub-agent](../multi_agent.md) turn (delegation is on by
default on the server chat paths): it checks the reported prompt's delegation
order and final answer; decode-only throughput does not measure a complete agent
turn.

## 12. MTP / NextN speculative decoding (Qwen 3.6)

Qwen 3.6 GGUFs can ship a **NextN / multi-token-prediction (MTP) draft block** that
both hosts use for lossless speculative decoding on solo (non-concurrent)
sequences. Of the public conversions, only
[unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
retains the block (see [Downloads](#downloads)); GGUFs without it serve standard
decode after a one-time warning. Source:
[`Qwen35Model.Speculative.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.Speculative.cs),
driven through the shared
[`SpeculativeExecution`](../../TensorSharp.Runtime/Speculative/SpeculativeExecution.cs)
draft / verify / rollback core — the `draft-head` algorithm of the pluggable
`--spec-type` layer, not a Qwen-specific loop. Unlike Gemma 4, no separate draft
GGUF is needed — the block is embedded in the trunk file, so the NextN head
needs no `--draft-model` (that flag is used only to attach a Qwen 3.8 DFlash2
drafter, §12.4; explicitly passing one that cannot be
activated on the startup model is a fail-fast startup error).

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

Speculation is off by default; enable with `--spec` (env `TS_SPEC=1`) on **either** `TensorSharp.Cli` or
`TensorSharp.Server` — [`SpeculativeCliFlags`](../../TensorSharp.Runtime/Speculative/SpeculativeCliFlags.cs)
is shared by both hosts. It engages on solo
(non-concurrent) sequences whenever the loaded GGUF retains the NextN block; on
GGUFs without it the engine logs once that speculation is unavailable and
serves standard decode.
`--spec-draft` (range 1-64) bounds the draft window and also sizes the native
graph cache at load, so it belongs on the same command line as `--spec`. The
shared default is `8`, but this hybrid trunk narrows a window the operator did
not set to 3 (§12.5); an explicit `--spec-draft` always wins. `--spec-pmin`
(default `0.15` for a per-token head — top-1 probability over its top-10
logits, where a completely flat draft scores 0.10) is the minimum draft
confidence to keep a token (`0` never gates). `--spec-type draft-head` pins this path explicitly;
`auto` (the default) already selects it from the checkpoint. On `ggml_cuda`, the
GDN chunked-prefill kernel also speeds the speculative verify: measured on
Qwen3.6-27B IQ2_XXS it cut MTP speculative-verify decode from 217 to 174 ms/token
(see §8, `GDN_CHUNK_PREFILL_MIN_SEQ_LEN`). See
[Speculative decoding](../../FEATURES.md#speculative-decoding) for the full flag
list and the other three algorithms — including `--spec-type ngram`, which needs
no trained weights at all and therefore also runs on the Qwen 3.5 checkpoints that
ship no draft block.

**After a reused prefix.** `--spec` does not turn prompt reuse off: the default
radix prefix cache stays on (only `--no-prefix-cache` or `TS_SCHED_PREFIX_CACHE=0`
disables it). A sequence
that starts from a reused prefix (a retained chat turn, a startup-warmed or
disk-restored shared prefix) has no draft hidden rows for those positions, so
the NextN head restarts only its own private attention cache at the end of that
gap (`DraftHeadResumesAfterGap`), keeping absolute rotary positions; the trunk's
KV and recurrent state are untouched. The first decode step captures a fresh
trunk hidden row before drafting resumes, so early acceptance can be lower than
with a fully replayed head. This covers the default fused-verify route for a
solo request; the paged speculative route selected by disabling fused verify
still needs a position-zero prefill. A DFlash drafter (§12.4), when attached,
replaces the NextN head and survives a gap through its own KV ring instead.
Details:
[Arming after a reused KV prefix](../speculative_decoding.md#arming-after-a-reused-kv-prefix).

### 12.4 DFlash2 block drafting (Qwen 3.8)

Qwen 3.8 has a second, unrelated drafter available: **DFlash2**, a 5-layer
block-diffusion model that ships as its own GGUF
([z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF),
`general.architecture = dflash`). Where the NextN block drafts one token per
forward, DFlash2 proposes a whole block of 7 in one, reading the trunk's own
residuals rather than only its last hidden state. Attach it with `--draft-model`;
naming the file IS the request, no `--spec` needed. The drafter and the NextN
block are alternatives, not layers - when a DFlash file is attached it wins, and
the loader says so.

The algorithm, the drafter's KV ring, its fused graphs and the candidate-selector
lattice are all shared code
([`ModelBase.DFlash.cs`](../../TensorSharp.Models/Speculative/ModelBase.DFlash.cs),
[`speculative_decoding.md`](../speculative_decoding.md#dflash-and-dflash2)). What
belongs to this model is only the residual tap: `SpecForward` additionally writes
the residual ENTERING each layer named in `dflash.target_layers` (`[6, 20, 34, 48,
62]` for the shipped drafter), packed into one 25600-wide row per token. It is
done inside the fused whole-model verify kernel - one `ggml_cpy` per tapped layer
into a `[hidden, N]` output block - so speculation does not force the op-by-op
layer loop
([`Qwen35Model.DFlash.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.DFlash.cs)).

**What it is worth depends almost entirely on the workload.** Measured on one
RTX 3080 Laptop with Qwen3.8-27B-UD-IQ3_XXS, greedy, best of two runs: on
free-form prose 18.3 tok/s plain against 20.9 with DFlash2 (1.14x) and 19.1 with
the trunk's own NextN block; on a highly predictable prompt ("list the first 20
primes") 19.5 plain against 31.7 (1.63x), or 34.8 at `--spec-draft 7`. The
default window is 3 (§12.5) and the wider window is only worth taking when
acceptance is high - on prose it costs 40%.

Quality is unaffected: on the factual prompt the DFlash2 continuation was
byte-identical to the plain one, and to the pre-snapshot rollback path.

### 12.5 Recurrent-state snapshots (why speculation stopped losing)

Both drafters used to be a NET LOSS on this trunk - 15.5 tok/s for DFlash2 and
15.7 for MTP against 18.3 plain - and the reason was not the drafters. It was
what a REJECTION cost. The GDN recurrent state of §12.2 cannot be truncated like
a KV cache, so a partially-rejected verify restored a pre-verify copy of it and
re-forwarded the accepted prefix through all 64 layers: a second whole-model
forward per rejection. The state also crossed PCIe twice per step - 151 MB up
into the verify graph, 151 MB back down after it - whether anything was rejected
or not.

Three changes in the fused verify kernel and its caller removed all of it:

1. `ggml_gated_delta_net` already accepts a snapshot count K and emits the last
   K per-token states; the conv state after row *m* is a window of the
   `conv_input` tensor the graph already builds. The verify now keeps one
   snapshot per row, so the state a rollback wants is never recomputed - it is
   slot `N-1-accepted`.
2. `TSGgml_Qwen35CommitStateSnapshotOwned` writes that slot into the LIVE state
   entirely on the device. Every cached verify graph binds `*_state_in` from one
   shared device buffer (`g_q35v_state_buf`), so the write is visible to the next
   verify whatever shape it runs at - and that verify then skips its state
   upload, as this step skipped its download.
3. A verify only READS the live slices (it writes `*_state_out` and the snapshot
   slots), so those slices ARE the pre-verify state until a commit overwrites
   them - and a commit only happens after the rollback decision.
   `SpecSnapshotRecurrentState` therefore copies nothing.

4. The single-row steps a speculative session falls back to (when the drafter
   declines) defer their download too. Such a step's post-window state is just
   the `*_state_out` slices and nothing decides anything about it later, so the
   caller commits slot -1 at once. Before this, each one broke the device-state
   chain - 151 MB down, and the next verify uploaded it again - which on an MTP
   run was 46 steps out of 125.

Effect on 256 prose tokens with DFlash2: `rollbackMs` 3604 -> 0, `snapshotMs`
919 -> 69. Deferring the one-row steps is worth a further 5-20% on top,
paired-run (DFlash2 factual 22.0 -> 27.1, MTP prose 19.1 -> 21.4), and it is also
the more exact path: committing on the device is a raw copy of the tensor the
graph produced, where the host round trip went through an unpack-and-repack, so
on the factual prompt it reproduces plain decoding byte for byte while the host
path drifted in the last few tokens.

The cost is VRAM - the GDN op's output grows by one ~150 MB state per slot across
the 48 recurrent layers - which is why the default window is 3 and not 8.
`TS_Q35_VERIFY_SNAPSHOTS=0` restores the restore-and-re-forward path (the
workaround for the open wide-verify divergence on `ggml_cuda`, see
[speculative_decoding.md](../speculative_decoding.md)); it is also the automatic
fallback for any shape the kernel will not persist.

### 12.6 Folding the MTP catch-up into the first draft step

llama.cpp's `draft-mtp` runs the MTP block once over `n_accepted + 1` rows: one
pass that both replays the verified tokens through the draft head and takes the
first draft step. TensorSharp ran those as two calls, and because a draft call's
cost is mostly fixed - its own graph, launch and readback, ~6 ms of which only
about 1 ms is arithmetic - that extra call was the largest single difference
between the two engines on this model.

`Qwen35Model.DraftCatchUpAndStep` does both in one `TryFusedMtpBlock` call over
all rows. The kernel already folds the LM head over the LAST `n_logits` rows, so
asking for one logit row returns exactly the row the draft needs; the normed
hidden comes back for every row and the last one chains the next step. The fold
is an identity rather than an approximation: the block is causal over its own
KV, so its last row sees exactly the replayed rows either way, and the output is
byte-identical with acceptance unchanged.

`DraftHeadSpeculator` stashes the commit and folds it into the next `Propose`,
flushing it as an ordinary catch-up whenever the stashed rows do not run right up
to the next step's position. Measured on Qwen3.8-27B-UD-IQ3_XXS: `catchUpMs`
191 -> 0, +4.0% at 256 tokens and +5.3% on prose.

The remaining per-step difference is `MtpProjectInput`, the C# front end that
builds the block's input (embedding, `enorm`, `hnorm`, concat, `eh_proj`). Over a
256-token run it costs 462 ms against the fused kernel's 804 ms across 208
calls - 2.2 ms of every 6.1 ms draft call - spent on about six separate device op
launches, each of which synchronises because the lazy-sync path is Metal-only.
Folding it into the fused MTP graph is the next step.

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

## 13a. Tensor parallelism

Qwen 3.5/3.6 runs under `--tp N` on the direct `cuda` backend and on the GGML
CUDA / Vulkan backends. Four pieces make the GGML path work:

- **Per-rank packed GatedDeltaNet kernel** (`TSGgml_Qwen35GdnLayerTP`) — one
  ggml graph per rank covering input RMSNorm, the packed column-parallel
  in-projection, `ggml_ssm_conv`, q/k L2-norm and head tiling,
  `ggml_gated_delta_net`, the gated RMSNorm, and the SiLU(z) gate. Each rank
  owns a block-cyclic slice of the V heads, so the recurrent path needs no
  cross-rank communication.
- **Device-resident recurrent state** — the conv window and delta state upload
  once and are updated in place; downloading them would cost ~1 MB per layer
  per token per rank. `ResetKVCache` drops the device copies.
- **Expert-parallel MoE** (`Qwen35Model.TensorParallelGgmlMoE.cs`) — whole
  experts partition across ranks (128 of 256 each), so a rank still issues one
  batched `ggml_mul_mat_id` dispatch per projection instead of a per-(token,
  expert) loop. The shared expert stays Megatron-split, since every token uses
  it.
- **Column-parallel LM head** — the vocabulary is the output dimension, so each
  rank owns a contiguous row range and the gather is two copies into disjoint
  halves of the logits buffer, with no collective at all.

Measured on 2× RTX 2000 Ada (`--backend ggml_cuda --tp 2`): Qwen3.5-35B-A3B
UD-IQ4_XS — which does not fit a 16 GB card — decodes at 20.4 tok/s short-prompt
(12.1 at 3 K context) with prefill 26.3 / 80.2 / 75.7 tok/s at 23 / 512 / 2966
tokens, using 9.4 + 8.0 GB. Decode time splits: attention block 32%, MoE experts
31%, GDN 15%, ssm_out + AllReduce 6%, router 6%, LM head 2.5%. On Qwen3.5-9B
Q8_0 (which fits one card) `--tp 2` is byte-identical to the single-GPU run over
60 greedy tokens with the device AllReduce. Full detail:
[`TENSOR_PARALLELISM_PLAN.md`](../../TENSOR_PARALLELISM_PLAN.md).

## 14. Optimization opportunities

- **Native GDN decode (fallback path)** — the fused whole-model path is
  native, but the per-operation fallback still runs GDN decode in
  managed C# (with pre-allocated buffers and `Ops.AddmmBatch`). Moving that
  fallback update into native C / CUDA would remove its remaining managed
  overhead.
- **Vectorized conv1d** — `Conv1dStep` is a scalar loop. A SIMD or native
  vectorized version would shave a few percent off the decode hot path.
- **MoE prefill batching** — MoE prefill currently iterates per token. A
  batched expert prefill kernel (analogous to the decode path) would
  speed up long prompts on MoE variants.
