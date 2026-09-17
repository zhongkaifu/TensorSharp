# Qwen 3.5 / 3.6 family

[← back to model index](README.md)

| Property | Value |
|---|---|
| Provider | Alibaba |
| GGUF architecture keys | `qwen35`, `qwen35moe`, `qwen3next` |
| Source class | [`Qwen35Model`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.cs) (legacy per-seq) + partial in [`Qwen35Model.GatedDeltaNet.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.GatedDeltaNet.cs) + [`Qwen35Model.BatchedForward.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.BatchedForward.cs) (`IBatchedPagedModel`) |
| Vision encoder | [`Qwen35VisionEncoder`](../../TensorSharp.Models/Models/Qwen35/Qwen35VisionEncoder.cs) |
| Image processor | [`Qwen35ImageProcessor`](../../TensorSharp.Models/Models/Qwen35/ImageProcessor.cs) |
| Example models | Qwen3.5-9B (dense hybrid), Qwen3.5-35B-A3B / Qwen3.6-35B-A3B (MoE-family), Qwen3.6-27B (dense) |
| Modalities | Text, image |
| Thinking mode | Yes (`<think> ... </think>`) |
| Tool calling | Yes (`<tool_call>{...}</tool_call>`) |
| Batched / paged forward | **Default ON** — set `TS_QWEN35_BATCHED=0` (or `--no-continuous-batching`) to force the legacy per-sequence KV-swap path for A/B comparison. Includes a per-slot GatedDeltaNet recurrent-state pool and optional native batched GDN kernel (`TS_QWEN35_BATCHED_GDN_NATIVE=1`). See §11. |
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
> files (same file names) strip it — on those, `--spec` silently falls back
> to standard decode (`--spec-type ngram` still works there, since it needs no
> trained drafter weights).

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
  `TSGgml_Qwen35RopePositionAbi` guards against a native library built before this
  contract: such a library disables the fused graphs, loudly, instead of being
  called with the wrong number of arguments. (A DFlash drafter keeps its own
  positions; they only affect how many drafts are accepted, never the output.)
- **Stored with the state it describes.** Each per-request holder carries it through
  swaps, retention, re-keying and pooling, a shared-prefix checkpoint and its clone
  copy it, the per-block KV snapshot of the KV-swap concurrency path ends with it (like
  the recurrent state, as of the block's end), and the checkpoint file format (`Q5KC`) is **version 2**, which writes it
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
- **Reuse.** A cache that went through an image turn was not the state a re-prefill of
  the same history builds, so Phase 0 of the prefix cache stopped every reuse path at
  the first image (`SupportsReuseAcrossMediaSpan = false`). Qwen 3.5/3.6 now declare
  `true`: follow-up turns continue the cache past the image.

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
reused and a cold turn with or without images; the concurrent test (holders and the
arena) produced identical tokens on both backends.

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

### Whole-model fused prefill

On supported dense GGML GPU models, prefill is dispatched through
`TSGgml_Qwen35ModelVerify`: one graph evaluates every FullAttention and
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
geometry before omitting any copy. Set
`TS_QWEN35_METAL_GDN_INPLACE_STATE=0` for an A/B run with separate state
buffers, or `TS_QWEN35_FD_PERSIST=0` to rebuild the decode graph per token.

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

Qwen 3.6 GGUFs can ship a **NextN / multi-token-prediction (MTP) draft block** that
both hosts use for lossless speculative decoding on solo (non-concurrent)
sequences. Of the public conversions, only
[unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)
retains the block (see [Downloads](#downloads)); GGUFs without it silently serve
standard decode. Source:
[`Qwen35Model.Speculative.cs`](../../TensorSharp.Models/Models/Qwen35/Qwen35Model.Speculative.cs),
driven through the shared
[`SpeculativeExecution`](../../TensorSharp.Runtime/Speculative/SpeculativeExecution.cs)
draft / verify / rollback core — the `draft-head` algorithm of the pluggable
`--spec-type` layer, not a Qwen-specific loop. Unlike Gemma 4, no separate draft
GGUF is needed — the block is embedded in the trunk file, so `--draft-model`
is not used (explicitly passing one that cannot be
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

Speculation is off by default; enable with `--spec` (env `TS_SPEC`, or the legacy
`TS_MTP_SPEC=1` the native loader reads) on **either** `TensorSharp.Cli` or
`TensorSharp.Server` — [`SpeculativeCliFlags`](../../TensorSharp.Runtime/Speculative/SpeculativeCliFlags.cs)
is shared by both hosts. It engages on solo
(non-concurrent) sequences whenever the loaded GGUF retains the NextN block; on
GGUFs without it the engine silently serves standard decode.
`--spec-draft` (range 1-64, default `8`) bounds the draft
window and also sizes the native graph cache at load, so it belongs on the same
command line as `--spec`; `--spec-pmin` (default `0.75` for a
per-token head — top-1 probability over its top-10 logits) is the minimum draft
confidence to keep a token (`0` never gates). `--spec-type draft-head` pins this path explicitly;
`auto` (the default) already selects it from the checkpoint. On `ggml_cuda`, the
GDN chunked-prefill kernel also speeds the speculative verify: measured on
Qwen3.6-27B IQ2_XXS it cut MTP speculative-verify decode from 217 to 174 ms/token
(see §8, `GDN_CHUNK_PREFILL_MIN_SEQ_LEN`). See
[Speculative decoding](../../FEATURES.md#speculative-decoding) for the full flag
list and the other three algorithms — including `--spec-type ngram`, which needs
no trained weights at all and therefore also runs on the Qwen 3.5 checkpoints that
ship no draft block.

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
2. `TSGgml_Qwen35CommitStateSnapshot` writes that slot into the LIVE state
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
`TS_Q35_VERIFY_SNAPSHOTS=0` restores the old restore-and-re-forward path and
`TS_Q35_VERIFY_DEFER_STATE=0` keeps the snapshots but restores the download, so
the two halves can be measured apart; either is also the automatic fallback for
any shape the kernel will not persist.

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
191 -> 0, +4.0% at 256 tokens and +5.3% on prose. `TS_MTP_FOLD_CATCHUP=0`
restores the two-call shape.

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
  native, but the legacy per-operation fallback still runs GDN decode in
  managed C# (with pre-allocated buffers and `Ops.AddmmBatch`). Moving that
  fallback update into native C / CUDA would remove its remaining managed
  overhead.
- **Vectorized conv1d** — `Conv1dStep` is a scalar loop. A SIMD or native
  vectorized version would shave a few percent off the decode hot path.
- **MoE prefill batching** — MoE prefill currently iterates per token. A
  batched expert prefill kernel (analogous to the decode path) would
  speed up long prompts on MoE variants.
- **Promote native batched GDN out of opt-in.** The
  `TS_QWEN35_BATCHED_GDN_NATIVE=1` kernel exists today but is gated on
  perf verification. Once the n=1 regression is closed it becomes the
  default GDN dispatch in the batched path.
