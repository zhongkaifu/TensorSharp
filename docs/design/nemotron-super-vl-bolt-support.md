# Nemotron 3.5 Super VL / Bolt support - Design & gap analysis

**Status:** All gaps addressed. G1/G2/G6 implemented; G3/G4 verified already covered; G5 already supported by existing vision code.
**Date:** 2026-10-19 / 2026-10-20
**Target models:**
- `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` (ggml-org GGUF, ~79GB)
- `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-GGUF` (unsloth)

## 1. What the model actually is (from the downloaded GGUF header)

Parsed the first 8 MB of `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4.gguf` (the header carries all
metadata + the full tensor-info array; tensor data is later in the file).

| Key | Value |
|---|---|
| `general.architecture` | **`nemotron_h_moe`** -- already a registered TensorSharp architecture alias |
| `block_count` | 88 |
| `embedding_length` | 4096 |
| `vocab_size` | 131072 |
| `context_length` | 1048576 (1M) |
| `ssm.inner_size` / `state_size` / `conv_kernel` / `time_step_rank` | 8192 / 128 / 4 / 128 |
| `expert_count` | **512** |
| `expert_used_count` | **22** |
| `expert_group_count` / `expert_group_used_count` | 1 / 1 |
| `expert_feed_forward_length` | 2688 |
| `expert_shared_count` | 1 |
| `expert_shared_feed_forward_length` | 5376 |
| `moe_latent_size` | **1024** |
| `expert_weights_norm` / `expert_weights_scale` | true / 5.0 |

Layer pattern is interleaved Mamba2 / Attention / MoE, encoded the same way Nemotron-H already does:
per-layer `feed_forward_length` (`0` = non-FFN, `2688` = MoE) and `attention.head_count_kv` (`0` = Mamba2,
`2` = attention).

## 2. Confirmed tensor layout (from the header)

**MoE layer** (e.g. `blk.1`):
```
blk.1.ffn_latent_down.weight    [4096, 1024]    BF16(30)   hidden -> latent
blk.1.ffn_latent_up.weight      [1024, 4096]    BF16(30)   latent -> hidden
blk.1.ffn_gate_inp.weight       [4096, 512]     F32        router: hidden -> 512 experts
blk.1.exp_probs_b.bias          [512]           F32        routing logit bias
blk.1.ffn_up_exps.weight        [1024, 2688, 512]  NVFP4(40)  [latent, n_ff, experts]
blk.1.ffn_down_exps.weight      [2688, 1024, 512]  NVFP4(40)  [n_ff, latent, experts]
blk.1.ffn_up_exps.scale         [512]           F32        per-expert NVFP4 weight scale
blk.1.ffn_up_exps.input_scale   [512]           F32        per-expert NVFP4 input scale
blk.1.ffn_up_shexp.weight       [4096, 5376]    NVFP4(40)  shared expert (gated, 2x n_ff)
blk.1.ffn_down_shexp.weight     [5376, 4096]    NVFP4(40)  shared expert down
blk.1.attn_norm.weight          [4096]          F32
```

**Mamba2 layer** (e.g. `blk.0`):
```
blk.0.ssm_in.weight    [4096, 18560]  BF16(30)
blk.0.ssm_out.weight   [8192, 4096]   BF16(30)
blk.0.ssm_conv1d.weight [4, 10240]    F32
blk.0.ssm_conv1d.bias   [10240]       F32
blk.0.ssm_a / ssm_d     [1, 128]      F32
blk.0.ssm_dt.bias       [128]         F32
blk.0.ssm_norm.weight   [1024, 8]     F32
```

**Attention layer** (e.g. `blk.7`): separate `attn_q` / `attn_k` / `attn_v` / `attn_output` (BF16), not a
fused `attn_qkv`.

## 3. What TensorSharp already supports

`NemotronModel` (architecture id `nemotron_h`, aliases `nemotron_h`/`nemotron_h_moe`/`nemotron_h_omni`)
already implements the hard parts of this family:

- Hybrid `Mamba2` / `Attention` / `FFN(MoE)` layer dispatch from the two per-layer arrays.
- Latent-space MoE (applies a latent projection, routes in latent space, projects back out).
- Shared experts (`ffn_up_shexp` / `ffn_down_shexp`, gated `ReLU^2`), see `moeInfo.HasSharedExperts`.
- Routing logit bias (`exp_probs_b` / `exp_probs_b.bias`).
- NVFP4 weights + per-tensor `scale` (scale2) sidecars (`qw.Scale`).
- MTP / speculative (DFlash) and RADIO vision (`NemotronVisionEncoder`).

## 4. Gaps

### G1 -- Latent tensor naming (the load-blocker) -- **FIXED**

TensorSharp's `NemotronModel` looks up the latent projections as
`ffn_latent_in.weight` / `ffn_latent_out.weight`, but the Super-VL/Bolt GGUF uses
**`ffn_latent_down.weight` / `ffn_latent_up.weight`** (llama.cpp `MOE_LATENT_DOWN`/`MOE_LATENT_UP`, the
"nemotron 3 super" naming).

Semantic mapping (confirmed by dims):
- `ffn_latent_down` [4096,1024] = `ffn_latent_in` (hidden -> latent)
- `ffn_latent_up` [1024,4096] = `ffn_latent_out` (latent -> hidden)

**Fix (in tree):** `NemotronModel.NormalizeLatentWeightNames()` rewrites the dictionary keys
(`down->in`, `up->out`) once, right after `LoadWeights()`, in both `_quantWeights` and `_weights`. Weight
objects are untouched (their `CacheKey`/device buffers are independent of the dictionary key), so the
legacy `ffn_latent_in/out` path and the new `ffn_latent_down/up` checkpoint both serve the same downstream
code.

### G2 -- Per-expert NVFP4 scale2 (`<base>.scale`) -- **IMPLEMENTED**

Routed/non-shared experts carry a **per-expert** sidecar in addition to the NVFP4 blocks:
- `ffn_*_exps.weight` [.., 512] NVFP4(40) -- self-contained per-block scale (36-byte/64-elem block
  format embeds the block scale).
- `ffn_*_exps.scale` [512] F32 -- the **`weight_scale_2`** (per llama.cpp `_generate_nvfp4_tensors`, written
  separately and applied inside the up/down matmuls at inference).
- `ffn_*_exps.input_scale` [512] F32 -- `input_global_scale` calibration metadata.

`AttachSidecarWeightScales` only attaches a **1-element** `.scale` (covering the shared experts' scalar
`.scale[1]`); the routed experts' vector `.scale[512]` stays in `_weights` with `Scale = 1.0f` on the views.

**Fix (in tree):** `NemotronModel.InitExpertScales()` reads `ffn_*_exps.scale` and folds one per-expert
multiplier `downScale * upScale^2` into the routing weight, in both `MoEForward` and
`TryMoEPrefillBatchedByExpert`. The square is required because llama.cpp applies `up_exps_s` inside the up
matmul -- *before* the ReLU^2 non-linearity (`ReLU^2(scale*x) = scale^2 * ReLU^2(x)` for the positive scale2
values) -- while `down_exps_s` lands after. Experts without sidecars (IQ-quantized, e.g. Bolt) are a no-op.
Still needs DGX-Spark (Blackwell) logit validation, since NVFP4 only runs there.

### G6 -- Per-block MoE config ("Puzzle"/Bolt) -- **IMPLEMENTED**

Bolt-74B-A9B (the "Puzzle" / `block_configs` variant) gives each MoE layer its own expert count,
FF width and top-k, unlike the uniform Super-VL:

```
blk.1  ffn_up_exps [1024, 2048, 240]  # 240 experts, n_ff 2048
blk.3  ffn_up_exps [1024, 1536, 240]  # n_ff 1536
blk.21 ffn_up_exps [1024, 2560, 432]  # 432 experts, n_ff 2560
```

`expert_used_count` and `expert_feed_forward_length` are per-layer arrays `[88]`, not scalars.

**Fix (in tree):** `NemotronModel.InitPerLayerMoEConfig()` derives `_numExpertsPerLayer[]` (from the
stacked expert tensor's last dim) and `_numExpertsUsedPerLayer[]` (from the `expert_used_count` array,
scalar fallback). `_numExperts`/`_numExpertsUsed` are grown to the observed maximum so every pre-allocated
routing buffer still fits; `MoEForward` and `TryMoEPrefillBatchedByExpert` route against the per-layer
counts. Uniform checkpoints are binary-identical to before (per-layer == global). Per-block under **tensor
parallelism is rejected with a clear error** for now (the TP MoE paths are not per-layer aware yet).

Note: Bolt's experts are `IQ2_XXS` (type 16), so G2 (NVFP4 per-expert scale2) does **not** apply to Bolt --
it only affects the 120B NVFP4 file.

### G3 -- 512 experts / top-22 -- **verified already covered**

`expert_count=512`, `expert_used_count=22`. The stacked-expert storage, per-expert view creation,
`SelectTopKInPlace` (delegates to `TensorComputePrimitives`, no pool-size limit), and the per-block
config infrastructure already support 512 experts. Routing buffers are correctly sized
(`_moeProbs`[512], `_moeTopExperts`[22]). The outstanding verification is CPU-offload memory for
512 NVFP4 experts per MoE layer (~2.8 GB/layer) -- `--n-cpu-moe` should be validated on the DGX Spark.

### G4 -- BF16 non-expert weights -- **verified already covered**

`ssm_in/ssm_out/attn_*/ffn_latent_*` are BF16 (type 30). BF16 is explicitly listed in
`CanStoreDirectCudaCompressedWeight`, `ManagedQuantizedOps.SupportsCpuQuantizedStorage`, and the
`ManagedQuantGemm.Float` cheap-convert path. The Nemotron DSpark drafter already exercises BF16
attention weights in production. No gap.

### G5 -- Vision -- **already supported**

Super-VL / Bolt are `image-text-to-text` with a RADIO-family vision tower in the mmproj.
`NemotronVisionEncoder` handles the exact `clip` mmproj + `nemotron_omni` projector format:
it reads `clip.vision.*` config keys, loads `v.blk.*` ViT blocks (with bias and GELU) and
`mm.model.mlp.*` projector tensors. `LoadVisionEncoder` is already wired. The
RADIO ViT (32 blocks, 1280 embed, 16 heads, GELU, bias, scale_factor=2, MLP 1280->20480->4096)
matches the existing code exactly. No additional work needed.

## 5. Summary

All gaps are addressed. The text path has G1 (latent naming), G2 (NVFP4 scale2), and G6 (per-block MoE)
implemented. BF16 and 512-expert routing are already covered by existing infrastructure. Vision is
already supported by the existing `NemotronVisionEncoder`. The remaining verification items (NVFP4
scale2 correctness, CPU-offload under 512 experts, and end-to-end vision) must be validated on the
DGX Spark since NVFP4 requires Blackwell and the models are too large for the local 5090.

## 6. Verification strategy

No local GPU can hold these (79GB NVFP4, needs Blackwell; Bolt needs >46GB). The user will test on a
DGX Spark. Text-only load should be validated first by a **no-run metadata/weight-inspection pass**
(load model -> dump layer types, expert counts, per-layer tensor names -> assert against the header),
then a short-context token-generate check on the DGX Spark.