# Qwen3.8-Flash-Next (`qwen4exp`) on the shared decoder runtime

Status: steps 1 to 3 of §3 and QSA phase 1 are in (`Qwen4ExpRuntimeModel`, not routed yet). `Qwen4ExpModel` and its engines
stay until the runtime matches them.

**Done (2026-11-11).** `Qwen4ExpRuntimeModel` derives from `Qwen35RuntimeModel` and adds only the family's own
parts: the hyper-connection hooks (§2.1), the sigmoid output gate of the recurrence (§2.2), the PLE layer (§2.4)
and the QSA budget (§2.3 phase 1). On the synthetic checkpoint (`Qwen4ExpRuntimeTests`) it matches
`Qwen4ExpModel`:

| backend | one prompt | chunked prompt |
|---|---|---|
| managed CPU | same bits within 1.5e-7 | ~5e-3 (each model moves as much against its own unchunked run) |
| ggml_cpu, ggml_metal | 4-6e-3 (`Qwen4ExpModel` itself differs from its managed run by 6e-3) | same |
| MLX | under 2e-2 against `Qwen4ExpModel`'s managed run | same |

What the port needed in the runtime: `EmitLayerPrologue` (a layer's addition to the stream before its blocks),
`StepTokens` (the rows' tokens, for the n-gram hashes), `RequiresGraphs` (no per-op fallback for a family the
per-op layers do not implement), `FamilySpeculationRefusal`, a state factory (`CreateDeviceGdnState`; the PLE
layer's state adds the conv history and the n-gram history, the latter as floats in a tensor so copies and
prefix checkpoints carry it), `attention.recurrent_layers`, and `Sqr`/`Sqrt`/`Clamp` in the tensor executors.
One trap: a replayable step is emitted twice when its signature misses the cache (the signature pass, then the
recording), so hashing a step must use the history as of its start (the state keeps one level of undo).

**Host experts (§2.5, first half).** `GraphOp.MoeHostExperts` runs a layer's routed experts from host memory at a
pause in the program (GGML: the native `HostMoeSegment` seam, with a registered accelerator tier running the
routes it holds). The runtime plans the offload itself on one GGML device (`StandardDecoderModel.ExpertPlacement`,
for every MoE family): explicit `--n-cpu-moe` / `--cpu-moe` win; otherwise a model whose experts do not fit
keeps whole trailing layers on the accelerator, or - with a routing profile from earlier loads, or on unified
memory with window slots only while the profile is collected - runs every layer at a seam with its most-routed
experts in fixed slots. The budgets are `ExpertPlacementPlanner`'s (moved out of `Qwen4ExpModel`, which uses
them too). With it the real UD-Q2_K_XL checkpoint (79 GB) loads and generates on the 48 GB M5 Pro (ggml_metal).
The native builder sizes its cgraph from the context's tensor capacity: a tiered pause lowers to about a dozen
ggml tensors per layer, which overflowed a graph sized from the program's node count.

**Speculation (§2.6, MTP).** The shared-target MTP head (`--draft-model mtp-*.gguf`) loads as the runtime's
draft layer (`OpenSeparateDraftHead`, before the caches exist). The draft layer is the shared one, split into
hooks: `EmitDraftInput` (here: each stream sees [rms(embedding) * enorm, grouped rms(stream) * hnorm] through
eh_proj), the block hooks (hyper-connections), `EmitDraftHead` (here: the `nextn.hc_head_*` mixer), and
`DraftChainsStream` (the head reads the trunk's whole stream and chains its own). A verify window keeps the PLE
layer's conv input in a history buffer and leaves the live histories alone; the accept count commits the kept
prefix's conv window (`EmitCommitExtras`) and n-gram history (`CommitHostVerify`). Greedy speculation (n-gram
drafts and the MTP head) emits the plain greedy tokens on the managed CPU, ggml_cpu, ggml_metal and MLX, and
the head's drafts match `Qwen4ExpModel`'s within 3e-2 (ggml_cpu, ggml_metal).

**The real checkpoint on the Mac.** UD-Q2_K_XL on the 48 GB M5 Pro (ggml_metal), the runtime against `Qwen4ExpModel`,
both tiered by the same routing profile with the host expert cache pinned (`TS_HOST_MOE_IO_CACHE_GB=6`; without it
the window follows what macOS can give at load, which moves with the run order):

| | runtime | `Qwen4ExpModel` |
|---|---|---|
| decode, 200 greedy tokens | 28.9-29.0 tok/s | 29.1-29.2 tok/s |
| 54-token prompt | 0.77-0.79 s | 0.84-0.86 s |
| perplexity, 2 x 1024 tokens (route-tail skip off) | 13.714 | 13.778 |
| the same run's prompt halves (2 x 513 tokens) | 4.37 s | 4.35 s |

`TS_HOST_MOE_TIMING=3` splits each pass the same way for both (accelerator, host experts, window reads). What
closed the gap from -9%, in order: Strata's read-ahead (`MoeHostExperts` takes the next offloaded layer's router and
gate stack, and the GGML builder computes `pred_next` from them); the PLE rows gathered as `Qwen4ExpModel` does
(random-access advice on the table, a decode's 16 faults concurrently, a prompt's rows by pread), 1.7 -> 0.35 ms a
token; and ggml-metal's own fusions, about 280 kernels a token fewer. Those need the node shapes ggml-metal
matches: `TopK` lowers to `ggml_argsort_top_k` (descending, as the IR defines it; `ggml_top_k`'s order is
unspecified, so the host seams' route-tail skip took a route's position for a rank it was not), the router is
llama.cpp's chain with its clamp and closing scale, a pause pins the route weights first and holds one-row ids with
views no node consumes (an output flag on the argsort refuses the fusion), a VIEW of the router weight keeps the
logits allocated past the chain (the alloc dependency `ggml_backend_sched` would add; the builder allocates with
gallocr), and the stream broadcasts are one-term products (`[1, n] x [1, hc]` mul_mat) instead of hc copies.
Route-tail skip makes decode depend on what the window holds, so text and perplexity move between runs with it on.

**Speculation (DFlash2).** A DFlash drafter (`--draft-model` with `general.architecture = dflash`) attaches through
the shared `AttachDraftModel`. The runtime already captured the stream entering each `dflash.target_layers` id,
which is the completed output `Qwen4ExpModel` taps at id - 1; the one family hook is `EmitCaptureRow`, the mean of
the hc streams. On the synthetic checkpoint with a synthetic drafter the feature rows and logits of a prompt and
a verify match `Qwen4ExpModel`'s native taps within 2e-2 (ggml_cpu, ggml_metal), and greedy block speculation
emits the plain greedy tokens on the managed CPU, ggml_cpu, ggml_metal and MLX.

Speculation on the same checkpoint and machine (200 greedy tokens, each side's defaults): MTP 31.8-34.1 tok/s
(runtime, gate 0.5 as `Qwen4ExpModel`'s) against 32.0, with identical text; DFlash2 37.0 against 23.1. With
route-tail skip off, plain, MTP and DFlash2 text agree until a near-tie (three different spellings of one heading),
where a verify window's batched Metal kernels round differently from one row's, as `Qwen4ExpModel`'s do.

**Next:** QSA phase 2 above the budget (8192 tokens), and the real model on ggml_cuda.

Flash Next has 125 B parameters in total and 6 B active. Its routed experts take 46 GB at UD-Q2_K_XL.

Today the family runs on its own code:

- `Qwen4ExpModel`: about 10.5 k lines over 21 files. It includes GGML token spans (fused native graphs), per-sequence holders, arenas, MTP, DFlash2, expert tiers and tensor parallelism.
- `Qwen4ExpCudaModel`: the direct-CUDA engine with the Q4e kernels.
- The Metal window engine.

This document has three parts:

1. What the runtime is missing.
2. How each missing piece lands as a family hook or an IR op that every executor plugin runs.
3. The order in which to migrate, and what each step must prove.

## 1. Inventory

The runtime column describes `refactor/unified-execution` at `e317b781`: graph IR in Core, GGML executor plugin.

| Mechanism | `Qwen4ExpModel` today | Shared runtime | Gap |
|---|---|---|---|
| Hyper-connections: the residual stream is `hc` = 4 times wide. A block reads it through a grouped RMS norm, a low-rank sigmoid gate (hc·n → 320 → hc·n) and the stream mean, and writes back through a per-stream scatter of `2·sigmoid(inject/hc)`. The final mixer is the output norm. | `HcMix` / `HcCombine` on the host, plus fused into the token-span graphs (`ggml_ops_qwen4exp*`). | The residual stream is one `[n, rows]` node: `h = h + mix`, `h = h + ffn`. | A residual-stream hook (§2.1). |
| Gated DeltaNet. The input arrives already normed by the mixer, and the output gate is a sigmoid (Qwen 3.5 uses SiLU). | GGML `GatedDeltaNetChunked`; host fallback. | `Qwen35RuntimeModel.EmitRecurrentMixer`: the conv, L2 norms, `GatedDeltaNet` op, gated RMS norm and SiLU(z). | A parameter on the shared mixer: no input norm, a sigmoid gate (§2.2). |
| Full attention. A joint interleaved Q\|gate projection, Q/K RMS norm, partial IMRoPE, and a sigmoid gate on the output. | Fused attention block. | `GatedAttentionOutput`, `EmitQueryKey`, multi-axis rotary (`RopeMulti`). | The interleaved Q\|gate split: a family `EmitQueryKey` plus views. |
| QSA on the full-attention layers. An indexer scores blocks of `compress_ratio` cells, and the layer attends densely to the `indexer_top_k` best blocks. Below the budget it is exactly dense. | `Qwen4ExpModel.Qsa.cs` and the native kernels. | Dense attention only. | Dense is exact below the budget. Above it, the runtime needs an indexer cache and a block-select step (§2.3). |
| PLE n-gram embeddings. One layer adds rows of a ~320 M-row host table, addressed by a hash of the token's bigram/trigram context. The table is 51 B parameters, read one row per head per token. | `Qwen4ExpModel.Ple.cs`: host gather with prefetch, and the history committed per request. | Media embeddings only (`QueueMedia` / `InjectMedia`). | A per-layer host input: the family computes the rows before the step and the program adds them (§2.4). The per-request n-gram history becomes slot state. |
| MoE: 512 experts, top 10, softmax then renormalized, plus a shared expert behind a scalar sigmoid gate. | Fused MoE graphs; host seams (route-tail skip). | `EmitMoe`: `SoftMaxExt` / `TopK` / `MulMatId`, shared expert, `NormalizeExpertWeights`. Host-offloaded experts decline the graph and run per-op. | The shared-expert gate form (scalar sigmoid), and host experts inside a program (§2.5). |
| Expert tiers. All layers route at a host seam, and the hottest experts run from a compact device copy inside the same graph (`ggml_ops_moe_hot_tier.cpp`). Route-tail skip drops missed low-rank routes at decode. | `Qwen4ExpModel.ExpertTiers.cs`, `ExpertPlacement.cs`. | `MoeCpuOffloadConfig`: whole layers on the host, per-op. | A tiered MoE op in the IR whose GGML lowering is the existing hot-tier kernel (§2.5). |
| MTP: a shared-weight draft head over the target's streams. DFlash2 rings per request. | `Mtp.cs`, `DFlash.cs`, `BatchedSpeculative`. | Draft layers (NextN) as programs, DFlash/DFlash2/DSpine through `AttachDraftModel`, per-op speculation. | The MTP head reads the HC streams, not one hidden row. It is a draft layer whose input is the mixer output (§2.6). |
| Prefix cache, per-sequence holders, retained cache, arenas. | Family holders. | Slots, radix prefix cache, checkpoints. | Covered by the runtime once the state (KV, GDN, PLE history, QSA indexer cache) lives in slots. |
| Tensor parallelism and layer split. | GGML per-rank graphs (`--tp`); the direct-CUDA engine places whole layers. | Stages as programs; meta-device TP (opt-in); per-op TP. | The HC streams are replicated per rank, like the hidden row. QSA splits by head. |
| Vision (Qwen-VL tower). | `Vision.cs`, `Multimodal.cs`. | `IVisionCapableModel` through the shared encoder; real-tower validation is still owed. | The same as Qwen 3.5. |
| direct `cuda` backend (`Qwen4ExpCudaModel`, Q4e kernels). | Its own engine. | The CUDA tensor executor is in progress on the PR branch. | The Q4e kernels become the CUDA plugin's kernels for the IR ops (quantized products, routed experts, the recurrence). |

## 2. Design

### 2.1 Residual-stream hook (hyper-connections)

The layer loop in `EmitStep` becomes three overridable steps. The default reproduces today's program node for node, so every existing family is unchanged.

```csharp
// The stream entering the stack from the embedding: default h; HC: hc copies (Concat or Repeat of views).
protected virtual int EmitStreamInput(GraphStep step, int embedded);
// What a block reads: default rms_norm(h) * w; HC: grouped norm -> low-rank gate -> stream mean (+ inject).
protected virtual BlockInput EmitBlockInput(GraphStep step, int layer, int stream, BlockSite site); // site = mixer | ffn
// How a block's output joins the stream: default h + out; HC: per-stream scatter of 2*sigmoid(inject/hc).
protected virtual int EmitBlockOutput(GraphStep step, int layer, int stream, BlockInput input, int output);
// The head's input: default rms_norm(h) * output_norm; HC: the final mixer.
protected virtual int EmitStreamOutput(GraphStep step, int stream);
```

Every HC operation is already an IR op:

- **Grouped norm:** `Reshape` to `[n, hc, rows]`, then `RmsNorm`, then `Mul` by the `[n·hc]` gammas.
- **Gate:** two `MulMat`, `Scale`, `Silu`, `Sigmoid`.
- **Stream mean:** `View` per stream, `Add`, `Scale`.
- **Scatter:** a broadcast `Mul` of `[n, 1, rows]` by `[1, hc, rows]`, then `Add`.

Programs therefore run on every executor plugin without new kernels. A plugin may fuse the pattern later, as MLX fuses residual add plus RMSNorm today.

The slot, the captures (`CaptureLayers`), the stage hand-off (`StepOutputs.Hidden`) and the draft layers carry the stream width `hc·n`. Today they assume `n`. The hidden width becomes `StreamWidth`, which defaults to `HiddenSize`.

### 2.2 Recurrent mixer variants

The Qwen 3.5 mixer gains two family switches:

- `MixerInputNormed`: the HC mixer already normed the input, so the mixer skips `attn_norm`.
- `MixerOutputGate`: `Silu` or `Sigmoid` on z.

The packed in-projection, the snapshots for verify windows and the commit programs stay as they are. `Qwen4ExpRuntimeModel` derives from the same recipe, so one GDN emission serves Qwen 3.5, 3.8 and Flash Next.

### 2.3 QSA

Phase 1 runs the attention dense. That is exact while a request's KV length is at most `indexer_top_k · compress_ratio`. Above that budget the runtime refuses with a reason. It never approximates silently.

Phase 2 adds the following:

- **Indexer cache:** a per-slot, per-layer state of compressed indexer keys. It is written when a block of `compress_ratio` cells completes.
- **Selection:** `MulMat` for the indexer scores, then `TopK` over blocks.
- **Gather:** a `GetRows` of the selected cells' K/V into a window, then `FlashAttnExt` under the matching mask.

The GGML lowering can keep the existing QSA kernels as a composite op, `GraphOp.SparseAttention`, with its primitive decomposition as the reference. The tensor executors run the decomposition. Ties at the cut-off are resolved by index, as ggml's `top_k` does, so programs agree across backends.

### 2.4 PLE

The family computes the rows on the host before the step. These are the table gathers by n-gram hash, with the existing prefetch. It hands them to the step as an input of the layer that owns PLE:

```csharp
protected virtual void PrepareStepInputs(GraphStep step, StepRows rows);   // family: gathers PLE rows per sequence
protected virtual int EmitLayerExtras(GraphStep step, int layer, int x);    // PLE: x + conv1d(ple rows) ...
```

The n-gram history (the last `ngram − 1` tokens per request) is slot state. It is snapshot, restored, checkpointed and prefix-cached like the recurrent states, so prefix reuse and speculation rewind it.

### 2.5 MoE with host seams and tiers in a program

Today a host-offloaded layer declines the whole graph. The design adds a routed-expert op, `GraphOp.MulMatIdTiered`, with these inputs:

- the stacked device copy of the tier (a compact table of the hot experts plus the id → slot map);
- the host experts;
- the routes.

Its semantics are those of `MulMatId` over all experts.

- The GGML lowering is the hot-tier seam that already exists (`ggml_ops_moe_hot_tier.cpp`).
- A tensor executor runs the device part with its expert products and the host part with the managed kernels.

Route-tail skip becomes a parameter of the op: drop routes below rank R whose expert misses the tier, at decode widths only. It stays opt-out with the user's existing default.

### 2.6 MTP

The Flash Next MTP head is a draft layer whose input is the final mixer's collapsed row. That row is a capture the program already outputs. With §2.1 in place the existing draft-layer programs (`RunDraftLayer`) serve it. The shared-weight head reads the same `output.weight`.

## 3. Migration order and what each step proves

1. **§2.1 and §2.2 with dense attention and dense MoE.** These run on a synthetic `qwen4exp` fixture (`Qwen4ExpSyntheticModelBuilder`; HC 4, two GDN layers and one attention layer, PLE off, 8 experts). They must match `Qwen4ExpModel` on the managed and `ggml_cpu` backends (same tolerances as `Qwen35RuntimeTests`), then match under slots, layer split and speculation.
2. **§2.4 PLE** with prefix reuse and rewind. The fixture gets PLE on. Concurrent requests' PLE histories must stay separate.
3. **§2.5 tiers and host experts in a program** on ggml_cuda. The fixture must match with every expert held, then with tiers and host experts. Route-tail skip changes decode rows only within the existing tolerance and never the prompt (the `RouteTailSkip_*` tests).
4. **§2.3 QSA phase 1 (dense below the budget)**, then phase 2. Phase 2 must give the same cells as the native kernels on tie-free inputs.
5. **The real model** (UD-Q2_K_XL, 79 GB, on 2× RTX PRO 6000) on ggml_cuda, against `Qwen4ExpModel`: greedy text, NLL, prefill/decode tok/s, MTP and DFlash2 acceptance and speed, concurrency and prefix reuse. The route changes only where the runtime is at parity.
6. **The direct `cuda` engine:** its Q4e kernels become the CUDA executor plugin's kernels. Metal and MLX follow through their plugins.

`Qwen4ExpModel`, `Qwen4ExpCudaModel` and the window engine are deleted only after step 5 (GGML) and step 6 (cuda) are at parity.
