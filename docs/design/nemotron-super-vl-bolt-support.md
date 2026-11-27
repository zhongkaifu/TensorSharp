# Nemotron 3.5 Super VL / Bolt support - Design & gap analysis

**Status:** G1, G2, G3 and G6 implemented. G2 and G3 work only after fixes made when the DGX Spark run found them broken (section 5). G4 was covered as claimed. G5 (vision) was **not** covered for Bolt, whose mmproj splits q/k/v; that was fixed after the PR #282 review. Super VL's `v.post_ln` is applied since 2026-11-27, when Super VL ran end to end on one 48 GB A40 with `--cpu-moe` (see the [model card](../models/nemotron.md#nemotron-35-super-vl-and-bolt)). Tensor parallelism now applies the NVFP4 scales and Nemotron opts in, but only synthetic tests cover it, on CPU and in a gated `ggml_cuda` `--tp 2` test; no real Super VL `--tp 2` generation has completed on GPUs (section 6).
**Date:** 2026-10-19 / 2026-10-20; corrected 2026-10-28 after the PR #282 review
**Note (2026-11):** this note describes the legacy per-op `NemotronModel`, which has since been removed; Nemotron-H now runs on the shared decoder runtime (`NemotronRuntimeModel`), where `--tp`, `--layer-split`, expert offload on one device and speculation work as the [model card](../models/nemotron.md#7-tensorsharp-implementation-the-shared-decoder-runtime) describes.
**Target models:**
- `NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` (ggml-org GGUF, about 71.6 GiB / 76.919 GB)
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

**Fix (in the legacy model, since removed):** `NemotronModel.NormalizeLatentWeightNames()` rewrites the dictionary keys
(`down->in`, `up->out`) once, right after `LoadWeights()`, in both `_quantWeights` and `_weights`. Weight
objects are untouched (their `CacheKey`/device buffers are independent of the dictionary key), so the
legacy `ffn_latent_in/out` path and the new `ffn_latent_down/up` checkpoint both serve the same downstream
code.

### G2 -- Per-expert NVFP4 scale2 (`<base>.scale`) -- **IMPLEMENTED, then fixed twice**

Routed/non-shared experts carry a **per-expert** sidecar in addition to the NVFP4 blocks:
- `ffn_*_exps.weight` [.., 512] NVFP4(40) -- self-contained per-block scale (36-byte/64-elem block
  format embeds the block scale).
- `ffn_*_exps.scale` [512] F32 -- the **`weight_scale_2`** (per llama.cpp `_generate_nvfp4_tensors`, written
  separately and applied inside the up/down matmuls at inference).
- `ffn_*_exps.input_scale` [512] F32 -- `input_global_scale` calibration metadata.

`AttachSidecarWeightScales` only attaches a **1-element** `.scale` (covering the shared experts' scalar
`.scale[1]`); the routed experts' vector `.scale[512]` stays in `_weights` with `Scale = 1.0f` on the views.

**Fix (in the legacy model, since removed):** `NemotronModel.InitExpertScales()` reads `ffn_*_exps.scale` and applies the per-expert
multipliers in both `MoEForward` and `TryMoEPrefillBatchedByExpert`. Experts without sidecars
(IQ-quantized, e.g. Bolt) are a no-op.

**Validated on the DGX Spark, and the first implementation was wrong.** It originally folded a single
multiplier `downScale * upScale^2` into the *routing weight*. That is algebraically correct --
`ReLU^2(us*x) = us^2 * ReLU^2(x)` for positive scale2 -- but numerically fatal: it defers the up
correction until *after* the squaring, so the kernel squares an `up_out` still `1/us` too large. At
NVFP4 `us ~ 1e-5` that overflows to `+Inf`, and the Inf met a `~1e-15` route weight to produce NaN.
Super VL aborted warmup with "non-finite router logits" at its first MoE layer.

Now (`6be5296e`, `a0941ceb`) the scales are kept **separate**: `us` is applied to the up projection
immediately before `ReluSquaredInPlace`, and only `ds` is folded into the routing weight (a single
multiply on correctly-magnituded values, which is safe). The batched GGML kernel gained an optional
trailing `up_scales` array so it can apply `us` between the up matmul and the ReLU.

**Separately**, the fused residual-add path dropped the *per-tensor* 1-element scale entirely
(`b3aba946`): `TryLinearAddInto` -> `FusedMatMulQuantAdd` takes no scale argument, so
`ssm_out.weight` on all 40 Mamba2 layers was added to the residual unscaled. `LinearForward` and
`LinearForwardInto` both apply `qw.Scale`, so only the fused decode path was wrong -- which is why the
native and CPU Mamba2 paths produced *identical* garbage and kernel-level debugging was a dead end.
Bolt's symptom was fluent text for 3-4 tokens, then collapse. Gate every fused quant fast path on
`qw.Scale == 1.0f` (the Qwen3.5 pattern); note two MoE call sites `throw` on fallback rather than
degrading, so `CanMoEDecodeResidualAdd` needs the same gate.

### G6 -- Per-block MoE config ("Puzzle"/Bolt) -- **IMPLEMENTED**

Bolt-74B-A9B (the "Puzzle" / `block_configs` variant) gives each MoE layer its own expert count,
FF width and top-k, unlike the uniform Super-VL:

```
blk.1  ffn_up_exps [1024, 2048, 240]  # 240 experts, n_ff 2048
blk.3  ffn_up_exps [1024, 1536, 240]  # n_ff 1536
blk.21 ffn_up_exps [1024, 2560, 432]  # 432 experts, n_ff 2560
```

`expert_used_count` and `expert_feed_forward_length` are per-layer arrays `[88]`, not scalars.

**Fix (in the legacy model, since removed):** `NemotronModel.InitPerLayerMoEConfig()` derives `_numExpertsPerLayer[]` (from the
stacked expert tensor's last dim) and `_numExpertsUsedPerLayer[]` (from the `expert_used_count` array,
scalar fallback). `_numExperts`/`_numExpertsUsed` are grown to the observed maximum so every pre-allocated
routing buffer still fits; `MoEForward` and `TryMoEPrefillBatchedByExpert` route against the per-layer
counts. Uniform checkpoints are binary-identical to before (per-layer == global). Per-block under **tensor
parallelism was at first rejected with a clear error** (the TP MoE paths were not per-layer aware); G7
removed that refusal.

Note: Bolt's experts are `IQ2_XXS` (type 16), so G2 (NVFP4 per-expert scale2) does **not** apply to Bolt --
it only affects the 120B NVFP4 file.

### G3 -- 512 experts / top-22 -- **was NOT covered; fixed**

`expert_count=512`, `expert_used_count=22`. The stacked-expert storage, per-expert view creation,
`SelectTopKInPlace` (delegates to `TensorComputePrimitives`, no pool-size limit), and the per-block
config infrastructure do support 512 experts, and routing buffers are correctly sized
(`_moeProbs`[512], `_moeTopExperts`[22]).

**But this analysis missed the GGML kernel limit.** `GgmlNative.MoEExpertsForward` (native
`TSGgml_MoEExpertsForwardF32`) rejects `numExperts` outside **[1, 16]**, so top-22 aborted on the first
MoE forward with `MoE: num_experts must be 1..16`. The sibling SwiGLU variant allows 1..32. Fixed in
`57a20f89` by splitting wider routing into chunks of <=16 and accumulating with `VecScaleAdd`; models
at or below 16 take a single chunk and are unchanged. The tensor-parallel equivalent got the same
treatment in `69201abb`.

Lesson for the next shape review: "the managed routing data structures support N" is not the same as
"every native kernel on the path supports N". Check the kernel's own argument validation.

On one device, CPU offload does not cover these experts. 512 NVFP4 experts are about 1.6 GB per MoE
layer (512 x 2 x 1024 x 2688 values at 36 bytes per 64), and Nemotron's single-device MoE path does
not implement `--n-cpu-moe` (section 6). Under `--tp`, only expert parallelism honours the flag. Since
G7 that path also serves latent MoE, but `--n-cpu-moe` with `--tp` has not been run.

### G4 -- BF16 non-expert weights -- **verified already covered (confirmed on Spark)**

`ssm_in/ssm_out/attn_*/ffn_latent_*` are BF16 (type 30). BF16 is explicitly listed in
`CanStoreDirectCudaCompressedWeight`, `ManagedQuantizedOps.SupportsCpuQuantizedStorage`, and the
`ManagedQuantGemm.Float` cheap-convert path. The Nemotron DSpark drafter already exercises BF16
attention weights in production. No gap.

**Confirmed empirically, and worth stating because the opposite looks plausible.** Super VL's Mamba2
`ssm_in/ssm_out` are BF16 and its NaN appeared right after a Mamba2 layer, which makes BF16 an
attractive suspect. It is not the cause. `addmm_quant_f32_impl` passes the type straight to
`ggml_new_tensor_2d` with **no** block-quant guard, and BF16 `mul_mat` works; the in-tree comment in
`ModelBase.DFlash.cs` claiming the AddmmQuant family "silently returns a zeroed output" for F16/BF16
is drafter-specific and stale, not a general constraint.

Do **not** "fix" this by dequantizing BF16 to F32 at load: doing so OOM-killed Super VL during prefill
warmup on a Spark. That Spark's unified pool is 124,546 MiB (121.6 GiB), the total in the
`model load refused` message quoted in section 6; the original report gave it as 121 GB. The NVFP4 file
holds about 71.6 GiB (76.919 GB) of weights, and F32 doubles every BF16 tensor in section 2:
`ssm_in`/`ssm_out` on the 40 Mamba2 layers, `attn_q/k/v` and `attn_output` on the 8 attention layers
(4096 x 4096 for q and output, 4096 x 256 for k and v) and `ffn_latent_down/up` on the 40 MoE layers.
Estimated from those shapes, that adds about 9.3 GiB, for about 81-83 GiB of weights; the range is
whether the token embedding and LM head (131072 x 4096 each, another 1 GiB apiece if stored as BF16)
were expanded too. This is an estimate, not a measurement: the original Spark report gave ~73 before
and ~85 after without recording whether the unit was GB or GiB, and neither figure matches the file
size in either unit. Run models at the quantization they ship in -- these GGUFs are sized deliberately
for the platform. If a dtype genuinely is unsupported, fix it in TensorSharp-owned native code rather
than inflating the footprint.

### G5 -- Vision -- **was NOT covered for Bolt; fixed**

Super-VL / Bolt are `image-text-to-text` with a RADIO-family vision tower in the mmproj. This section
first said the encoder "handles the exact `clip` mmproj + `nemotron_omni` projector format" and
"matches the existing code exactly", and the PR #282 description called this "verified against the
actual mmproj header". **That was wrong for Bolt.**

The published mmprojs use three layouts. These were read from their GGUF headers, with shapes in
GGUF ne order:

| mmproj | q/k/v | Extra tensors | Projector out |
|---|---|---|---|
| Bolt: `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF` `mmproj-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf`; unsloth `mmproj-BF16.gguf` (519 tensors) | **split** `attn_q` / `attn_k` / `attn_v` weights [1280, 1280] (BF16) and biases [1280] | `v.patch_embd.weight.1` [16, 16, 6, 1280] (video only); no `v.post_ln` | 4096 (`nemotron_omni`) |
| Super VL: ggml-org `mmproj-NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-{BF16,Q8_0}.gguf` (393 tensors) | fused `attn_qkv` [1280, 3840], bias [3840] | `v.patch_embd.weight.1` and `v.post_ln.{weight,bias}` [1280] | 4096 (`nemotron_omni`) |
| Nano Omni: unsloth `mmproj-BF16.gguf` (390 tensors) | fused `attn_qkv` | none | 2688 (`nemotron_v2_vl`) |

`SelfAttention` read only `v.blk.N.attn_qkv`, and `LinearForward` returned `null` for a weight that was not
loaded. So on Bolt every image request threw `NullReferenceException` (HTTP 500), and startup printed
`Nemotron multimodal warmup skipped: Object reference not set to an instance of an object`. Text
requests were unaffected, which is how the model "loaded and worked".

The projector was also misdescribed. It is RMSNorm, then 5120 -> 20480, ReLU^2, then 20480 -> 4096
(2688 for Nano Omni). Its input is the 1280-wide tower output after the 2x2 pixel shuffle (1280 x 4 =
5120), not 1280.

**Fix (in tree):**
- `NemotronVisionEncoder.LoadWeights` concatenates each block's split `attn_q` / `attn_k` / `attn_v`
  weights and biases into `attn_qkv`. Output rows go in Q, K, V order, after the dequantize to F32 that
  every vision weight already went through. A split file and a fused file therefore run identical
  weights. A block stored in both layouts is refused, and so is a split triple with a part missing,
  checked from the tensor names before any data is read.
- `ValidateWeights` checks every tensor the forward reads against the `clip.vision.*` metadata. The
  patch embedding, the norms, `attn_qkv`, `attn_out`, the FFN and `mm.model.mlp.{0,1,3}` are required.
  Biases and the class and position embeddings are optional, but checked when present. Any problem
  fails the load with `InvalidDataException` naming the file, the missing or mis-shaped tensors and the
  expected layout, rather than failing the first image. `LinearForward` now throws with the weight's
  name instead of returning `null`.
- `LoadVisionEncoder` (now `NemotronRuntimeModel.LoadVisionEncoder`) refuses a projector whose output width is not the language model's
  hidden size, e.g. a Nano Omni mmproj (2688) with Bolt (4096).
- `v.patch_embd.weight.1` is skipped with one log line. Nemotron takes no video input.
- A warmup failure now prints `Nemotron multimodal warmup FAILED (image requests will fail): ...`.

How a refused mmproj surfaces:
- **Server startup with `--mmproj`:** the server logs `Model load refused`, prints
  `error: model load refused: ...` and exits with code 2.
- **Web UI model switch:** HTTP 500 carrying the reason.
- **CLI:** the same error line and exit code 2, whether the projector was named with `--mmproj` or found
  beside the model for an `--image` request. The second path had no handler and ended in an unhandled
  exception.
- **Interactive `/mmproj`:** prints the reason and keeps the loaded model.

The Qwen3.5, Gemma 4 and Mistral 3 mmproj loaders are untouched.

Evidence:
- `NemotronVisionEncoderTests`, on synthetic mmprojs whose split tensors are row slices of the fused ones:
  - encoder output from a split file is bit-identical to the fused file's, for F32, BF16 and Q8_0 q/k/v
    on the managed path, F32 and Q8_0 on ggml_cpu, and F32 and BF16 on ggml_cuda;
  - language-model logits after an image prefill are identical;
  - the refusals above.
- `HostLoadRefusalProcessTests.BothHosts_NemotronMmprojMissingVisionTensors_IsRefusedAtLoad` covers the
  real CLI and server processes.
- No real Bolt image was run for this fix.

**Follow-ups (not done):**
- **`v.post_ln`** (done 2026-11-27). Super VL's mmproj ships a post-encoder LayerNorm, which llama.cpp's
  `build_vit` (`tools/mtmd/clip.cpp`) applies when the tensor is present. The encoder now applies it after
  the blocks. Super VL ran end to end with its experts on the host (`--cpu-moe`) and described the test image
  correctly. Bolt's and Nano Omni's mmprojs have no `post_ln`.
- **Dynamic resolution.** The `nemotron_omni` mmprojs declare `clip.vision.image_min_pixels` and
  `clip.vision.image_max_pixels`, which llama.cpp uses for dynamic resolution. This encoder tiles
  instead, with one tile by default.
- **GELU.** llama.cpp uses the exact erf GELU for `nemotron_omni`, and this encoder uses the tanh
  approximation.

Lesson: "loads and prints the right config" is not evidence. Diff the file's tensor list against what the
forward reads, for every published file of the family.

### G7 -- Tensor parallelism for Super VL and Bolt -- **IMPLEMENTED, NOT YET RUN**

The per-tensor `.scale` handling, the down-scale fold, the batched TP decode kernel's up scales and the
per-layer router are covered under G2/G6 and "Still unverified". On top of those, `--tp` still refused
Bolt's per-block MoE config, the fast expert-parallel MoE path refused latent MoE (both models) and NVFP4
up-scaled experts (Super VL), and the server could not host a TP Nemotron at all.

What changed:

| Piece | Change |
|---|---|
| Expert-parallel (EP) MoE | Now serves latent MoE (each rank's kernel reads the latent projection) and per-block expert counts (each layer is partitioned on its own). New native entry `TSGgml_MoEFFNPrefillRouteScaledF32` (TensorSharp-owned `ggml_ops_moe.cpp`; ggml is unchanged; removed in 2026-11 with the legacy model, which was its only caller) multiplies the ReLU^2 projection by one scale per route before the activation, so Super VL's NVFP4 experts run EP too. A route the kernel neutralises (weight 0) takes the scale of the expert it is pointed at, so it cannot overflow into `Inf * 0 = NaN`. |
| Per-block MoE (Bolt) | The refusal in `InitPerLayerMoEConfig` is gone; sharding and the expert loops use each layer's own counts. The per-expert (Megatron) split refuses, naming the tensor, an expert width the TP degree cannot split between quant blocks instead of silently dropping the remainder. |
| Across nodes | Image/audio embeddings exist only on the driver. A forward whose tokens hold `<image>`/`<so_embedding>` syncs the driver's embedding rows to every node with one AllReduce (the driver contributes them, every other rank zeros); the decision is taken from the tokens, which every node sees. The multimodal warm-up is skipped across nodes, since only nodes given the mmproj would join its collectives. |
| Server (KV snapshots) | `NemotronModel.TensorParallelSnapshot.cs`: each rank's K/V heads are a contiguous run of the single-device block layout, so the block format is unchanged. Across nodes the driver broadcasts each extract/inject (`ModelBase.BroadcastModelControl`) and the bytes move as raw bytes over the TCP mesh (`ICrossNodeByteGroup`, protocol 4): an inject sends the driver's block to every node, an extract gathers each node's head shards on the driver. |

Tests: `NemotronExpertParallelTests` runs the new kernel on the GGML CPU backend against a C# reference
and checks the per-rank route table; `TcpCommunicatorTests` covers the byte broadcast and gather. There is
no CPU harness for the Nemotron TP forward itself; the snapshot path was run only as two processes sharing
one RTX 5090 (Lightning), before this branch was rebased onto the maintainer's changes.

Limits that remain:
- The server serves a TP Nemotron on the per-sequence route (one sequence's forward at a time). Run the
  server as node 0 and `TensorSharp.Cli` as every other node. Each owner swap re-injects the incoming
  sequence's pages, its ~50 MB Mamba2 state included (Lightning), so the interconnect bounds concurrent
  serving; with the engine's 16-token pages for this family a swap is one mirrored inject per page.
- MTP/DFlash speculation is not mirrored to worker nodes.
- `--n-cpu-moe` with TP has not been run.

#### DGX Spark test plan (two nodes, one GB10 each)

Same model, backend and flags on both nodes; node 0 is the driver and prints the output. For the server,
start `TensorSharp.Server.Host` with the same `--tp 1 --tp-node-id 0 --tp-peers ...` on node 0 instead of the
CLI driver, and the CLI worker on node 1.

```bash
# node 0 (driver)
TensorSharp.Cli --model <model.gguf> [--mmproj <mmproj.gguf>] --backend ggml_cuda --tp 1 \
  --tp-node-id 0 --tp-peers <spark0>:9500,<spark1>:9500 --temperature 0 --max-tokens 64 --prompt "<prompt>"
# node 1 (worker)
TensorSharp.Cli --model <model.gguf> --backend ggml_cuda --tp 1 \
  --tp-node-id 1 --tp-peers <spark0>:9500,<spark1>:9500
```

1. **Load banner.** Expect `Nemotron MoE: expert-parallel across 2 GPU(s), N experts per GPU per layer` (Super VL:
   256; Bolt: a range) and, for Bolt, `MoE per block: ...`. `per-expert tensor parallelism (<reason>)` means EP was
   refused and gives the reason.
2. **Logits parity, the real check.** Run the same prompt once single-device and once across the two nodes with
   `TS_DUMP_LOGITS=<file>` on node 0, then compare the two float32 dumps (cosine ~1, same argmax). Text alone can
   differ at near-ties.
3. **Greedy text** on a few prompts, both models, `--temperature 0`.
4. **Images** (Super VL, Bolt): one `--image` prompt across nodes; the worker does not need `--mmproj`.
5. **Debugging aids:** `TS_TP_SCALE_DEBUG=1` prints each sharded weight's scale as it is applied;
   `CUDA_LAUNCH_BLOCKING=1` for an exact failing kernel.

## 5. Summary

**Validated on a DGX Spark (GB10) on 2026-10-24/25. Both models now produce correct output.** The
original gap analysis was directionally right but declared three things covered that were not, and all
three were real bugs. "Loads and reports the right shapes" turned out to be weak evidence: both models
loaded cleanly, printed correct layer/expert counts, and then produced garbage or NaN.

What was actually broken:

| Gap | Claimed | Actual | Fix |
|---|---|---|---|
| G2 per-tensor scale2 | implemented | fused residual-add dropped it entirely | `b3aba946` |
| G2 per-expert scale2 | implemented | folded past ReLU^2, overflowed to Inf/NaN | `6be5296e`, `a0941ceb` |
| G3 top-22 | "verified already covered" | GGML kernel rejects >16 experts | `57a20f89`, `69201abb` |
| G4 BF16 | "no gap" | **correct as written** | none needed |
| G5 vision | "already supported" | Bolt's split q/k/v: NullReferenceException on every image (found by the PR #282 review, not on the Spark); Super VL's `v.post_ln` ignored | split q/k/v fused at load + load-time validation; `post_ln` still open |

G4 is the one claim that held up, and it is worth trusting: BF16 works natively through AddmmQuant.

## 6. Verification results (DGX Spark, GB10, aarch64, CUDA 13.0)

Both models, `--temperature 0`, native quantization, no weight expansion:

- **Bolt 74B** on `ggml_cuda` and on `cuda`: correct answers, `finishReason=eos`. Byte-identical
  output across the pre- and post-merge trees.
- **Super VL 120B** on `ggml_cuda`: `"The capital of France is Paris."` and a correct multi-step
  reasoning answer, both `finishReason=eos`. About 71.6 GiB (76.919 GB) of weights loads and runs in
  the 121.6 GiB pool.

Symptoms before the fixes, for anyone who hits something similar: Bolt was fluent for 3-4 tokens then
collapsed into `>>(>>(` and file-extension soup; Super VL aborted warmup with "non-finite router
logits" at MoE layer 3. A per-layer probe located it -- `after L0 (Mamba2): finite absmax=0.85`,
`after L1 (MoE): NON-FINITE` -- which is what separated the MoE scale bug from the Mamba2 one.

**Benchmarking on this host:** end-to-end model timing is useless for kernel-level work. Two runs of
the *identical binary* gave Attention = 63553 ms and 81341 ms, a 28% swing wider than the spread
across four different code paths. Use `benchmarks/FlashAttnBench` (median of N, isolated kernel, plus
a CPU-reference `maxErr`). Note the profiler's "Attention" bucket also accumulates Mamba2 time
(`_attnTicks` is incremented inside `Mamba2Forward` and `TryMamba2NativeDecode`), so on this hybrid
model that label is mostly Mamba2 -- which is why an attention-kernel win is invisible end-to-end.

**Backend choice:** prefer `ggml_cuda` for Nemotron. Bolt decoded at ~8.8 tok/s there versus 0.7 tok/s
on the direct `cuda` backend, whose attention is the bottleneck. The direct backend's matmuls are ~11x
faster, so it is worth optimizing, but Mamba2 -- not attention -- is where its time goes.

**A `model load refused ... CUDA device has 5486 MiB free of 124546 MiB` is memory pressure, not a
platform limit.** Leftover `dotnet` runs and Ollama hold the unified pool. Clear them and retry; check
holders with `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`.

### Building on a Spark

The supported path is `eng/Dockerfile.gb10` (see `DEVELOPMENT.md`), which pins CUDA 13.0.2, compiles
`compute_121` PTX for the direct CUDA backend, and **disables Vulkan**. Committing `.cu` source is
therefore how a kernel reaches Spark users: `SelectCudaPtxContent` swaps locally compiled PTX in
whenever nvcc is present, so the committed baseline is not on their path.

Two traps if you build on the host instead of in the container:

- The GGML native build fails in the **Vulkan** target, because
  `ExternalProjects/vulkan-toolchain/shaderc-linux/bin/glslc` is an **x86-64 ELF** on an aarch64 host.
  The kernel returns ENOEXEC and `sh` parses the binary as a script, giving the misleading
  `glslc: 1: Syntax error: "(" unexpected`. Build with
  `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=false bash TensorSharp.GGML.Native/build-linux.sh`. The
  `curl: (22) ... 404` alongside it is that toolchain fetch failing, not upstream flakiness.
- Do not then run `dotnet build` without a skip flag: the csproj re-invokes `build-linux.sh`, where the
  same 404 is fatal (`exited with code -1` -> MSB3073). Use `-p:TensorSharpSkipGgmlNative=true`;
  `CopyGgmlNativeBinary` has no skip condition, so it still copies the fresh `.so`.

The committed PTX baseline targets `compute_75` and is refreshed with nvcc 13.0 on Linux
(`-p:TensorSharpUpdateCommittedPtx=true -p:CudaNvcc=<13.0>/bin/nvcc`); any host with a 13.0 nvcc can
refresh it, a Spark included once one is installed side by side:
`CommittedPtx_HasAnEntryForEveryKernelInTheSource` fails until every kernel in the source is in it.
Any newly added kernel must still be resolved with `GetOptionalFunction`, not `GetFunction`: the latter
calls `ThrowOnError` and handles are resolved eagerly in the `CudaKernels` constructor, so a missing
symbol in a stale or custom PTX would take the whole direct-CUDA backend down on no-nvcc builds. Gate
the feature on the handle being non-zero and fall back.

### Still unverified

- **Tensor parallelism.** With the fix for PR #282 review item 4, every Nemotron TP path applies the
  NVFP4 scales:
  - The router folds the per-expert down scale into the route weights.
  - The batched decode kernel and the per-expert loop apply the up scale before ReLU^2.
  - The TP linears apply the per-tensor `.scale`.
  - Nemotron opts in to `SupportsTensorParallelWeightScales`.

  What this means for each model:
  - **Super VL NVFP4:** `--tp 2` is no longer refused, but the file does not fit two A40s with ECC
    enabled (below). `--tp 2` is the only degree, since the attention layers have 2 KV heads.
  - **Super VL Q4_0:** has no sidecars and was never refused.
  - **Bolt:** no longer refused under `--tp`; each block's own expert count and top-k are used (G7).

  Expert parallelism now serves these models (G7): every rank's kernel reads the latent projection,
  and `TSGgml_MoEFFNPrefillRouteScaledF32` (since removed with the legacy model) applied each route's up scale before ReLU^2. A model whose
  stacked experts cannot be used for EP falls back to the per-expert row split, which refuses an expert
  width that does not split on quant-block boundaries.

  This is covered by synthetic tests only (`NemotronExpertScaleTests`, TP against one device and
  against folded twins). Besides the CPU tests, a gated `ggml_cuda` `--tp 2` test
  (`GgmlCudaTensorParallel_ScaledSidecarsMatchFoldedTwin`) runs both TP MoE paths and checks which one
  served each case: EP for plain and latent experts, with and without up scales
  (`TSGgml_MoEFFNPrefillRouteScaledF32`), and, at a top-k larger than a rank's share of the experts,
  the per-expert split, whose batched expert decode applies the up scales before ReLU^2 through
  `TSGgml_MoEExpertsForwardF32`. It needs `TS_TEST_GGML_BACKEND=cuda` and either two GPUs or
  `GGML_CUDA_DEVICES=2`, which emulates two devices on one GPU. It has passed both ways: emulated on
  one RTX 3080 and on two A40s. No real Super VL `--tp 2` generation has completed on GPUs. On two
  A40s with ECC enabled (48 GB, about 45.0 GiB usable each) the NVFP4 GGUF does not fit: rank 0, which besides its half of
  the routed experts holds all of the Mamba2, latent, shared-expert, router and LM-head weights, runs
  out of memory while preloading its weights. Estimated from the tensor shapes, rank 0's weights alone
  come to about 41.8 GiB and rank 1's to about 29.8 GiB. With ECC off, 48 GB cards (an A40, RTX A6000
  or RTX 6000 Ada) report about 49,140 MiB (about 48.0 GiB usable), which would leave rank 0 about
  6 GiB; none has been measured. The Q4_0 GGUF's `--tp 2` fit has not been measured.
- **Vision (G5):** split q/k/v and validation are covered by the synthetic tests above. No real Bolt
  image has been run since the fix. Super VL's image was run on 2026-11-27 with `v.post_ln` applied
  (answered correctly on `ggml_cuda` and `cuda`).
- **CPU offload (`--n-cpu-moe` / `--cpu-moe`):** implemented on the shared runtime since 2026-11 (the legacy
  single-device MoE path had none); Super VL ran with it on one 48 GB A40.
  The hosts still log "MoE CPU offload active", but no VRAM is saved:
  - on `ggml_cuda`, the selected layers' experts skip the load-time upload and are uploaded when the MoE
    kernel first uses them;
  - on `cuda`, every quantized weight is uploaded at load.

  Only expert-parallel MoE under `--tp` on `ggml_cuda` keeps experts on the host. Since G7 that path
  can serve Super VL and Bolt, but `--n-cpu-moe` with `--tp` has not been run on either, so whether it
  lowers rank 0's footprint enough for two A40s is unmeasured.
- **Group-16/head_dim-128 flash prefill kernel:** restricted to `windowSize == 0`. It implements sliding
  windows, but only full attention has been run. Builds that compile the kernels locally with nvcc have
  it, and so does the committed compute_75 baseline PTX used without nvcc: it was refreshed with CUDA
  13.0, and `CommittedPtx_HasAnEntryForEveryKernelInTheSource` keeps it current. Its body has no
  `__CUDA_ARCH__ >= 800` guard, so unlike the vision MMA and Q8_0 MMQ kernels it is not disabled when
  the loaded code targets a lower arch.

  Only a module without it (a stale or custom PTX) leaves Nemotron's direct-`cuda` prefill on the
  generic GQA kernel. Either way the prefill no longer takes the materialized path, so greedy text can
  change at near-ties (`docs/models/nemotron.md` section 8).
