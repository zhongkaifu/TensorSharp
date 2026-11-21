# Muse-Glimmer on the unified runtime and tensor executors

Status: design, plus the legacy baseline measured on the Mac. Stage S0's Mac part (the llama.cpp goldens and the frozen
legacy outputs), stage S1 (the runtime hooks G1-G3), stage S2's Mac part (the text decoder on the runtime, behind its
test seam; [§5.1](#51-s2-the-text-decoder-status)), stage S3's Mac part (serving: slots, rings, prefix cache,
checkpoints, checkpoint files, concurrency, the probe on the runtime; [§5.2](#52-s3-serving-status)), stage S4's Mac
part (the vision tower as graph programs, shared with Muse Spark, and images on the runtime;
[§5.3](#53-s4-vision-status)) and stage S5's Mac part (DFlash and n-gram through the shared speculation layer, the Metal
defaults, G11; [§5.4](#54-s5-speculation-status)) are implemented, and so is stage S6 (multi-GPU: `--tp 2` on ggml_cuda
and ggml_vulkan, `--layer-split 2`, DFlash declined under `--tp`; [§5.5](#55-s6-multi-gpu-status)), and stage S7's Mac
part (performance: every Mac backend within the gates against legacy, the measured settings kept but one;
[§5.6](#56-s7-performance-status)); stage S8's Mac part switched the descriptor to the runtime family
([§5.7](#57-s8-the-switch-status)), and stage S9's Mac part deleted the legacy model ([§5.8](#58-s9-the-deletion-status)). S0's CUDA part is done (its frozen lanes are fixtures); S2's, S3's, S4's, S5's and S7's CUDA parts wait for the GPU VM. Written
2026-11-20 on `feature/muse-glimmer-unified`, cut from `origin/main` 2d759459. ggml is upstream
d9fe7bff, unchanged (AGENTS.md), built with `TENSORSHARP_GGML_NO_UPDATE=1`. Every statement about
TensorSharp comes from the code at 2d759459. The numbers in [§4.1](#41-legacy-baseline-measured)
come from real weights on this Mac, and each one names its run.

This document covers one piece of work, in stages:

1. Move Muse-Glimmer (GGUF `muse-glimmer` / `muse_glimmer`) onto the shared decoder runtime
   ([unified-execution.md](unified-execution.md): `StandardDecoderModel`, the graph IR in
   `TensorSharp.Core/Graph`, the executor plugins). The move covers the text decoder, the vision
   tower, DFlash speculation, tensor parallelism, the prefix cache and every serving surface.
2. Show that it has no quality or performance regression against today's `MuseGlimmerModel`, on
   every backend that model serves.
3. Delete the legacy model, its native graphs, its switches and its settings. Retired names go
   through the removed-names registry.

Sections 1-3 are the inventory, the mapping and the gap analysis.
[Section 4](#4-validation-plan-and-no-regression-gates) gives the baseline and the gates.
[Section 5](#5-staged-implementation-plan) is the staged plan,
[section 6](#6-decisions-for-the-user) lists the decisions only the user can make, and
[section 7](#7-hardware-and-disk) lists the hardware each stage needs.

---

## 1. Today's Muse-Glimmer path

### 1.1 One request, end to end

```
TensorSharp.Cli / TensorSharp.Server.Host (Web UI, OpenAI and Ollama endpoints) / TensorAgent
  -> ModelBase.Create -> MuseGlimmerArchitecture.Descriptor.Factory               (MuseGlimmerArchitecture.cs)
     new MuseGlimmerModel(gguf, backend, tp, tpGroup, draftModelPath)             (MuseGlimmerModel.cs:108)
       ParseBaseConfig, ResolveSwaPattern, LoadWeights, FuseGateUpWeights
       TP: ValidateMuseGlimmerTpConstraints + ShardMuseGlimmerWeightsForTP         (.TensorParallel.cs)
       InitKVCache (or InitTpKVCache): ring rows for the SWA layers, a growing cache for the rest
       LoadDFlashDraftWeights when --draft-model names a dflash GGUF (single GPU only)
  -> InferenceEngine / BatchExecutor: ServingImplementation.Bespoke, ContinuousBatching = None,
     so concurrent requests take turns through the per-sequence KV swap (one primary cache)
  -> ForwardRefillCore (prompt, chunks of TS_PREFILL_CHUNK = 2048, capped at ring - window)
     ForwardCore (each chunk / decode row)
       TP           -> ForwardTP (per-rank fused graphs, TSGgml_TensorParallelExecutePlans)
       GGML backend -> TryFusedForward -> TSGgml_MuseGlimmerModelForward          (ggml_ops_muse_glimmer.cpp)
                       one ggml graph: in-graph embedding (Metal/CPU, tied CUDA), 52 layers, final norm,
                       LM head on the last row, logit scale, tanh softcap; decode graphs persist and replay
       otherwise    -> per-op path (Embedding, TransformerBlock x 52, LinearForward, ApplyOutputTransform)
       MLX --benchmark decode -> SubmitGreedyDecodeStep (pipelined greedy, .MlxDecode.cs)
  -> speculation (--draft-model): BlockDraftSpeculator -> ISpeculativeModel (.DFlash.cs)
       SpecForward captures the 5 target-layer inputs; ModelBase.DFlash(.Fused).cs drafts
       (TSGgml_DFlashInject / TSGgml_DFlashDraftBlock, ggml_ops_dflash.cpp); greedy verify; cost governor
  -> images: ChatProtocol emits <|patch|>; ModelMultimodalInjector.ProcessMuseGlimmerHistory expands it to
     <|image_start|> + N rows + <|image_end|>; MuseGlimmerVisionEncoder.Encode rows go to SetVisionEmbeddings
     and replace the filler rows before the weightless input RMSNorm
  -> MuseGlimmerOutputParser: `to=self` reasoning, ATEM tool calls, `to=user` answer
```

### 1.2 Text architecture

From `docs/models/muse-glimmer.md` §1 and the code (`MuseGlimmerModel.cs`, `ggml_ops_muse_glimmer.cpp`).

| Item | Muse-Glimmer-30B | Where it lives today |
|---|---|---|
| Layers | 52, dense; no MoE, no recurrent mixer | `Config.NumLayers` |
| Width | `n_embd` 6656, `n_ff` 19968, vocab 202048 | `ParseBaseConfig` |
| Attention | GQA, 32 query heads, 2 KV heads, `head_dim` 128, `kq_scale` = 1/sqrt(128) | `.Fused.cs:406` |
| Layer kinds | `attention.sliding_window_pattern`: a scalar period P (4) or a per-layer bool array; layer l is SWA when `l % P < P - 1`, so 39 SWA layers and 13 full layers (l = 3, 7, ..., 51) | `ResolveSwaPattern` |
| Window rule | llama.cpp STANDARD SWA: key p0 visible to query p1 iff `p1 - p0 < n_swa` (2048) | `fill_mg_mask` |
| Positions | RoPE on the SWA layers only, ggml NORM flavour (GPT-J pairs, mode 0); the full layers are **NoPE** | native step 4 |
| QK norm | per-head RMSNorm on Q and K before RoPE (eps 1e-5); the Q weight carries `qk_scale_factor`, the K weight is ones | native step 3 |
| Output gate | `attn * sigmoid(attn_gate · attn_norm(x))`, **elementwise** over `q_dim` (not one gate per head), before `attn_output` | native step 7 |
| Norms | four per layer: `attn_norm`, `ffn_norm` at `f_norm_rms_eps` (1e-5); `post_attention_norm`, `post_ffw_norm` at a hardcoded **1e-8**; the post norms take the raw block output and come before the residual add | `PostNormEps` |
| Input | the embedded rows get an **unweighted** RMSNorm (no learned scale), after any image rows replace their placeholders | `ApplyInputEmbeddingNorm` |
| FFN | SwiGLU over the fused `[gate \| up]` (`ggml_glu`, `swapped=false`) | `FuseGateUpWeights` |
| Head | `lm_head` (or tied `token_embd`), then `logits *= logit_scale` (0.19612), then tanh softcap at 20.0 | `ApplyOutputTransform` |
| Sink tokens | none | - |

### 1.3 Files and what happens to them

| File | Lines | Content | Fate |
|---|---:|---|---|
| `TensorSharp.Models/Models/MuseGlimmer/MuseGlimmerModel.cs` | 1,650 | config, SWA pattern, KV caches (ring + grow), truncation/ring slack, snapshots, per-op forward, chunked refill, vision span plumbing | delete (S9) |
| `.Fused.cs` | 478 | fused-graph pointer tables, layer trace, host sync of fused KV | delete |
| `.TensorParallel.cs` | 885 | TP validation, sharding, per-rank KV, per-op TP forward | delete |
| `.TensorParallelFused.cs` | 476 | per-rank fused graphs | delete |
| `.MlxDecode.cs` | 747 | MLX pipelined greedy decode (CLI `--benchmark` only) | delete (D7) |
| `.DFlash.cs` | 347 | `ISpeculativeModel` wiring, `SpecForward` capture | delete; the runtime's shared drafter replaces it |
| `.PrefixCache.cs` | 37 | page-family capabilities (class S, ring slack) | delete |
| `.Multimodal.cs` | 24 | `IVisionCapableModel`, `IMultimodalPromptExpander` | replaced by the runtime family's |
| `MuseGlimmerArchitecture.cs` | 36 | descriptor, Bespoke contract (TP <= 2 on ggml_cuda/vulkan) | rewritten (S8): SharedRuntime contract |
| `MuseGlimmerVisionEncoder.cs` | 1,308 | per-op ViT, CUDA-only fused block; was also used by Muse Spark | ported to a graph program (S4, D3: `MuseGlimmerVisionTower.cs`, which Muse Spark now loads); kept as the oracle until S9 |
| `MuseGlimmerImageProcessor.cs` | 221 | grid choice + Pillow Lanczos-3 stretch | kept (host code, shared with Muse Spark) |
| `TensorSharp.GGML.Native/ggml_ops_muse_glimmer.cpp` | 1,441 | `TSGgml_MuseGlimmerModelForward`, `_ResetDecodeCache`, `_ReleaseTpGraphs`, mask fills | delete (S9) |
| `TensorSharp.GGML.Native/ggml_ops_muse_glimmer_vision.cpp` | 469 | `TSGgml_MuseGlimmerVisionBlockQuantF32` | delete after S4 |
| `GgmlNative.cs` / `GgmlBasicOps.cs` wrappers, `GgmlMuseGlimmerVisionBlockArgs` | - | P/Invoke | delete with the exports |
| `ggml_ops_core.cpp` (3 calls to `TSGgml_MuseGlimmerResetDecodeCache`), `ggml_ops_mask.cu` (`tsg_cuda_fill_ring_mask_f16`, Glimmer's ring mask), `CMakeLists.txt`, `TensorAgent.Maui/GgmlExportedSymbols.targets` (4 symbols) | - | hooks of the native graphs | edit at deletion; check other callers of the ring-mask kernel first |
| `MlxFusedOps.TryPrefillAttentionBanded`, `RunDecodeGraphBatched` | - | used only by Glimmer's per-op MLX prefill and its pipelined decode | delete with them if nothing else calls them |
| `ModelBase.DFlash.cs`, `ModelBase.DFlash.Fused.cs`, `ggml_ops_dflash.cpp`, `DFlashConfig.cs` | - | the shared drafter (also the runtime's fallback below its graph programs) | kept |
| `ModelMultimodalInjector.ProcessMuseGlimmerHistory` | - | `<\|patch\|>` expansion, embedding cache by image path | kept, re-typed to `IMuseGlimmerVisionTower` (S4): both towers, Glimmer and Muse Spark |
| `TensorSharp.Runtime`: `ChatProtocolRegistry` (`muse-glimmer` entry), `MuseGlimmerOutputParser`, `ChatTemplate`/`KVCachePromptRenderer` header anchors, `Jinja2Template` | - | chat protocol, keyed by architecture | unchanged |

### 1.4 Execution by backend today

| Backend | Forward | SWA ring | Decode graph | Prefill | TP | DFlash |
|---|---|---|---|---|---|---|
| `ggml_cuda` | fused whole-model graph | 4352 rows, read whole | persistent, CUDA-graph captured; padded window copied only where ggml-cuda would pick the FA VEC kernel (`kv_window_needs_cuda_flash_attn_copy`) | 2048-row chunks, masks filled on the device | `--tp 2`, per-rank fused graphs, **no ring** (uniform caches) | fused inject + block, captured |
| `ggml_vulkan` | fused | 4352, read whole | persistent | 2048 chunks | `--tp 2` (first measured in S6, [§5.5](#55-s6-multi-gpu-status)) | fused |
| `ggml_metal` | fused, in-graph embedding | 4352; reads the moving span until it wraps, the whole ring after | persistent replay, `graph_optimize` on persist builds only | 2048 chunks, host masks (parallel span fills) | no (one device) | fused, queue drained before reading ids |
| `ggml_cpu` | fused, in-graph embedding | **none**: uniform caches (ggml-cpu FA evaluates every masked column) | persistent | 2048 chunks | no | fused |
| `mlx` | per-op, lazy flush every 4 layers (`TS_MLX_EVAL_EVERY_N_LAYERS`), KV materialized every 16 writes | none | `--benchmark`: pipelined greedy (one MLX graph a token, device argmax and embedding); real generation: per-op | per-op, banded device mask (`TryPrefillAttentionBanded`) | no | per-op drafter |
| `cuda` (direct) | per-op only (`CanUseFusedForward` excludes it) | none | per-op | per-op | no | per-op |
| `cpu` (managed) | per-op, correctness reference (docs: 2.7 / 0.2 tok/s) | none | per-op | per-op | no | per-op |

### 1.5 Caches

* **One primary cache** per model. Full layers grow by doubling from 2048 rows on Metal (8192 on the
  other GPU backends) up to the configured context. SWA layers are a fixed ring of
  `pad256(n_swa + chunk + 1)` = 4352 rows, armed only where a fused kernel can read it. The `+1` is
  load-bearing (a 4651-token prompt diverged from llama.cpp without it).
* **Concurrency** is the per-sequence ownership swap, not batching: one request runs at a time.
* **Prefix cache**: a radix page family (`IPageOnlyPrefixCacheModel`, class S,
  `TruncationKind.WithinRingSlack` with slack `ring - window - 1` = 2303). Pooled snapshots are
  capped at the ring (`MaxReusablePrefixTokens` = 4352). Live continuation of the primary cache goes
  past the cap. A rewind is capped at 16 tokens and stays inside the ring slack. A prompt with an
  image reuses nothing past the start of its first image.
* **The 2026-09-29 bug class** (memory note `muse-glimmer-kv-grow-loop`, docs §5): (1) the grow
  copied a stale host mirror of KV the fused graph had written on the device. `SyncHostBuffer`
  silently skipped an oversized request, recycled pool addresses attached stale device copies, and
  cached graphs kept the old tensors. (2) The ring was addressed by the full layers' capacity, so
  every device-to-host sync of the SWA layers was a no-op and pages captured zeros. (3) The
  scheduler swapped out an owner past the pooled cap and freed its blocks mid-step. All three are
  fixed and covered by `MuseGlimmerKvGrowTests`, `MuseGlimmerKvSnapshotTests`,
  `PerSequenceSwapPastReuseCapTests`, `PrefixCache/PerSequenceWindowSwapReproTests` and the probe
  `eng/validation/MuseGlimmerKvGrowProbe`. The CUDA real-weight run and the TP grow/snapshot run
  were never done.

### 1.6 Speculation

DFlash (docs §3) is a separate 5-layer GGUF (`general.architecture = dflash`, `dflash-kquant.gguf`,
1.63 GB, pinned in the TensorAgent catalog at unsloth `faa5b025`). It borrows the target's
`token_embd` and LM head and keeps its own SWA KV ring (`RingRows` = `pad32(n_swa + block + 1)`).
Encode: the target's per-layer **input** residuals at `dflash.target_layers` [2, 14, 26, 38, 50]
are concatenated (33280 wide), projected by `fc` and RMS-normed. Inject: K/V of every draft layer,
K gets a head norm and **NeoX** RoPE. Draft: `[anchor, MASK x (b-1)]` runs non-causally over
`[ring | block]`, and the target head makes the logits **without** the target's logit scale or
softcap. Verification is greedy against the target, with the cost governor
(`SpeculationCostGovernor`), the confidence gate (`--spec-pmin`, 0.35 for a block drafter), and
`--spec-draft` (15 in the docs). Under `--tp` the drafter is declined (the CLI warns; the server
exits 2). There is no n-gram path: `SpeculationProfitable => HasDFlash`.

### 1.7 Multimodal

* **Tower** (`MuseGlimmerVisionEncoder`, docs §2): a 50-layer ViT, width 1536, `n_ff` 8960, 16 heads
  of 96, patch 14, LayerNorm with bias, erf-GELU MLP, a learned 32x32 position embedding. Window
  attention over 32x32 windows of permuted patches; layers with `(il + 1) % 4 == 0` and the last
  layer attend globally. 2D NORM RoPE (theta 10000): the first half of `head_dim` rotates with the
  1-indexed column, the second half with the row. 2x2 pixel shuffle with channel-outer packing,
  then the adapter 6144 -> 4096 -> 4096 -> 6656 with erf-GELU. Weights stay quantized
  (`AddmmQuant`, MLX quantized products). One fused native block runs on ggml_cuda only
  (`TSGgml_MuseGlimmerVisionBlockQuantF32`); everything else runs per op.
* **Preprocessing** (`MuseGlimmerImageProcessor`): llama.cpp's `muse_glimmer_grid_size`
  (aspect-closest of the floor/ceil candidates, ties toward more tokens, capped at 4096 merged
  tokens), a Pillow-compatible Lanczos-3 stretch to `grid * 28`, and mean/std 0.5.
* **Prompt**: one `<|patch|>` per image becomes `<|image_start|>` + N filler rows +
  `<|image_end|>`. The rows replace the filler embeddings before the input RMSNorm. Chunked
  prompts re-slice the pending spans per chunk.
* **Muse Spark** loads the same tower (`MuseSparkModel.Vision.cs`) and uses the same injector path.
* **Video**: the chat protocol emits `<|video|>` for frames, but only `<|patch|>` is expanded. Muse-Glimmer
  has no video path today, and the migration does not add one.

### 1.8 Chat surface (unchanged by the migration)

These parts are keyed by architecture in `TensorSharp.Runtime` and do not depend on the model class:

* `ChatProtocolRegistry` `muse-glimmer`. It sets `TemplateAssistantHeaderAnchor` (KV-reuse rendering),
  `DirectAnswerGenerationSuffix` (" to=user<|message|>"), the thinking-budget transitions
  (`<|eom|><|start|>assistant to=user<|message|>`), reasoning effort, the grammar trigger for JSON
  with thinking, and the media placeholders.
* `MuseGlimmerOutputParser`. It maps the `to=self` channel to thinking and parses ATEM XML tool
  calls with typed arguments. Headerless JSON counts as content.
* The eos handling in `ModelBase` (stop tokens other than `eos_token_id`).
* `ModelService.SamplingForBoundedKnowledgePlanning` (medium effort for bounded knowledge planning).

The migration must keep them byte-identical. Their tests are `MuseGlimmerOutputParserTests`,
`MuseGlimmerDirectAnswerTests`, `MuseGlimmerKvReuseRenderTests`, `MuseThinkingTransitionTests`,
`MuseAtemSchemaTests`, `ThinkingChannelBudgetTests` and `ToolCallProgressStreamingTests`. Those tests
must pass unchanged, and the E2E gates in §4.8 check the same behaviour through the real surfaces.

### 1.9 Options, environment variables, switches and settings

| Name | Where read | Kind | Fate |
|---|---|---|---|
| `TS_MUSE_GLIMMER_PREFILL_CHUNK` (2048; 0 = no chunking) | `MuseGlimmerModel.cs:773` | legacy chunk width (sizes the ring too) | retire to the removed-names registry: the runtime's steps follow `TS_SCHED_MAX_BATCHED_TOKENS` and the ring step rows (D1) |
| `TS_MUSE_GLIMMER_SWA_RING` (=0 uniform caches) | `MuseGlimmerModel.cs:324` | documented A/B knob | retire; the tests use `RingGeometryForTests` (D8) |
| `TS_MUSE_GLIMMER_SWA_ROWS` | `MuseGlimmerModel.cs:378` | diagnostic ring size | retire (same seam) |
| `TS_MUSE_GLIMMER_LAYER_TRACE`, `_POS`, `_N`, `_DIR` | `.Fused.cs:106-137`, `ggml_ops_muse_glimmer.cpp:561-1296` | observer (keep-list class) whose implementation is deleted | retire with a pointer to the runtime's tools (D8) |
| `TS_MUSE_GLIMMER_VENC_TRACE` | `MuseGlimmerVisionEncoder.cs:134` | observer | retire with the per-op tower (D8) |
| `TS_MUSE_GLIMMER_MLX_STAGE_PROFILE` | `.MlxDecode.cs:123` | profiler | retire with `.MlxDecode.cs` |
| `TS_MLX_KV_MATERIALIZE_INTERVAL` (16) | `MuseGlimmerModel.cs:91`, read by Glimmer only | MLX per-op tuning | retire |
| `TS_MLX_EVAL_EVERY_N_LAYERS` | Glimmer, Gemma 4, GPT-OSS, Nemotron, the runtime | generic MLX | kept |
| `TS_PREFILL_CHUNK` | Glimmer's `ForwardRefillCore`, other families, the runtime | generic | kept (Glimmer stops reading it) |
| `TS_KV_FATTN_COPY` (`0` / `force`) | `ggml_ops_transformer_common.h` (`kv_window_needs_cuda_flash_attn_copy`), read by every caller of `view_kv_cache_window`: the legacy Glimmer graph, Gemma 4's graphs and the runtime's `GraphOp.KvWindow` | CUDA FA-VEC guard oracle | kept: the runtime's builder reads it too (G10) |
| `TS_DFLASH_PREFILL_CHUNK`, `TS_DFLASH_TIMING`, `TS_DFLASH_SELECTOR(_DEBUG)`, `TS_DFLASH_CONV`, `TS_DFLASH_DRAFT_VOCAB`, `TS_DFLASH_BATCHED_INJECT`, `TS_DFLASH_METAL_FUSE`, `TS_DFLASH_RING_BLOCK` | `ModelBase.DFlash*.cs`, `ggml_ops_dflash.cpp` | the shared drafter | kept |
| `TS_MUSE_GLIMMER_BACKEND`, `TS_MUSE_GLIMMER_REF`, `TS_MUSE_GLIMMER_REF_LONG`, `TS_TEST_MODEL_DIR`, `TS_TEST_GGML_BACKEND`, `TS_TEST_MUSE_GLIMMER_MODEL` | `MuseGlimmerParityTests`, `MuseGlimmerKvSnapshotTests`, the probe | test inputs (read, never set) | kept or renamed with the tests; `TS_MUSE_GLIMMER_BACKEND` falls back silently to GgmlCuda on an unparsable value (docs §4): fix the fallback when the tests move |
| Already retired: `TS_MUSE_GLIMMER_DFLASH`, `_FUSED`, `_FUSED_CPU`, `_TP_FUSED`, `_VENC_FUSED`, `_INGRAPH_EMBED`, `_VENC_F32`, `_GELU_TANH`, `_PERSIST`, `TS_MLX_MUSE_GLIMMER_EVAL_EVERY_N_LAYERS`, `TS_MLX_MUSE_GLIMMER_KV_MATERIALIZE` | `RemovedCliFlags.cs`, `SpeculationEnvVars.RemovedNames` | registry entries | kept |
| `--model`, `--mmproj` (hint `*mmproj*Muse*Glimmer*.gguf`), `--image`, `--draft-model`, `--spec`/`--no-spec`, `--spec-type`, `--spec-draft`, `--spec-pmin`, `--tp` (<= 2), `--backend`, `--kv-cache-dtype`, `--think`, `--no-prefix-cache`, `--no-continuous-batching` / `TS_SCHED_DISABLE_BATCHED` | `CliUsage.cs`, `ServerUsage.cs` | shared flags | kept. The usage pages' Glimmer sentences change: n-gram works without a drafter, and continuous batching, `--layer-split` and hybrid placement become available |
| `config/muse-glimmer-30b.json` | the config loader | ggml_cuda, UD-Q4_K_XL + mmproj Q8_0 pinned at `faa5b025`, greedy, 512 tokens | kept; add the drafter and its sha256 (S8) |
| `config/regression/source-pins.json`, `config/regression/catalog.json`, `eng/validation/release-model-catalog.json`, `eng/regression/tests/test_catalog_inventory.py` | regression | pins and profiles | kept; add UD-IQ2_XXS (`3e9bd0e6...`, 10,746,373,152 B) and `dflash-kquant.gguf` (`27d9a805...`, 1,631,205,312 B), both at `faa5b025` |
| TensorAgent `ModelCatalog` `muse-glimmer-30b-q4kxl` | `TensorAgent.Core/Catalog/ModelCatalog.cs:316` | `KvCacheDtype = q8_0`, `LeanCaches`, `MinDeviceMemoryGB = 32`, context 32768, sampling (1.0, 64, 0.95), optional mmproj and drafter | kept; re-measure footprint and speed on the runtime (S8) |
| `TensorSharp.TestMatrix` (`ModelDiscovery`, `FeatureCatalog`), `docs/env_var_feature_matrix*.md`, `TENSOR_PARALLELISM_PLAN.md`, `docs/models/muse-glimmer*.md`, `docs/supported_models*`, `docs/models/capability-reference*`, `FEATURES*`, `USAGE*`, `website/*` and its search indexes | docs and matrices | descriptions | updated (S8, S9) |

### 1.10 Tests and tools

| Test or tool | Covers | Fate |
|---|---|---|
| `MuseGlimmerSyntheticModelBuilder` | a tiny Glimmer GGUF (SWA pattern, gate, norms, softcap) | kept; extended (S0) |
| `MuseGlimmerKvGrowTests` (synthetic, any GPU lane) | grow mid-sequence == presized; restore and truncate below the ring | legacy's, kept until S9. The runtime's counterparts are in `MuseGlimmerRuntimeServingTests` (S3): the primary and a request's slot grown mid-sequence == presized, bit for bit, on every executor; rewinds within the rings or refused |
| `MuseGlimmerKvSnapshotTests` (real weights, `TS_TEST_MODEL_DIR`) | captured prefix == live; grow == presized | legacy's, kept until S9; since S3 it reads its backend as the other real-weight tests do (it defaulted to ggml_cuda and skipped elsewhere). The runtime's counterparts are `MuseGlimmerRuntimeKvTests` (S3): decode across the primary's and a slot's growth == presized, and a checkpoint of wrapped rings (cloned, and read back from a file in a new instance) == live |
| `MuseGlimmerParityTests` (real weights + `.parity` goldens) | greedy and long context vs llama.cpp, tokenizer, SWA pattern, DFlash lossless, image geometry | **the goldens are committed (S0)**: `InferenceWeb.Tests/Fixtures/MuseGlimmer/LlamaCpp/`, made by `eng/validation/muse-glimmer-llamacpp-goldens.py` with llama.cpp d3c3e242c (D9). The tests teacher-force llama.cpp's continuations and skip only without the IQ2_XXS weights; the legacy model agrees at 768/768 positions on ggml_metal and MLX |
| `MuseGlimmerMlxDecodeTests` | pipelined greedy contract, `RunDecodeGraphBatched` | delete with `.MlxDecode.cs` |
| `GgmlMuseGlimmerVisionBlockTests` | the fused CUDA vision block vs a host reference | replaced by tower-program tests (S4: `MuseGlimmerVisionTowerTests`, `MuseGlimmerVisionParityTests`); delete with the block |
| `MuseGlimmerRegressionTests`, `MuseGlimmerOutputParserTests`, `MuseGlimmerDirectAnswerTests`, `MuseGlimmerKvReuseRenderTests`, `MuseThinkingTransitionTests`, `MuseAtemSchemaTests` | geometry, tokenizer, placeholders, parser, protocol | kept unchanged |
| `KvBlockTransferRingTests` (Glimmer ring slack), `PerSequenceSwapPastReuseCapTests`, `PrefixCache/PerSequenceWindowSwapReproTests` | the 2026-09-29 fixes | the Glimmer-specific cases move to runtime ring tests; the scheduler cases stay (other families still swap) |
| `ServingContractConformanceTests`, `TensorParallelSupportGateTests`, `PrefixCacheFamilyCoverageTests`, `SpeculatorRegistryTests`, `SpeculativeDraftHeadLoaderTests`, `ModelMultimodalInjectorCoverageTests`, `MediaProviderContractTests`, `KvStateFingerprintNonEmptyTests`, `DirectCudaPrefillModelTests` | Glimmer rows in shared tests | updated for the SharedRuntime contract (S8) |
| `eng/validation/MuseGlimmerKvGrowProbe` (+ `compare.py`, `webui_chat_e2e.py`) | teacher-forced logits vs a presized cache; Web UI loop detection; since this branch `--nll-text` (perplexity over a fixed text) | kept, model-agnostic (it builds through `ModelBase.Create`) |
| `.parity/bench_mg_*` harness (README only in the repo) | CUDA ladder vs llama.cpp | the scripts are not in the repo; rebuild them in `eng/` if the CUDA ladder is needed (S7) |
| `benchmarks/VisionEncoderBench`, `TensorSharp.Server.Host/testdata/test_multiturn.{py,sh}`, the release-profile tools (`eng/validation/make-native-release-plans.py`, `make-text-http-release-profiles.py`, `tests/test_native_release_contract.py`) | Glimmer rows in shared harnesses | re-pointed at the tower program (VisionEncoderBench) or unchanged (they drive the hosts) |

---

## 2. The unified runtime, and what Muse-Glimmer needs from it

### 2.1 What exists

Muse-Glimmer is close to LFM3, which was built natively on the runtime (`Lfm3RuntimeModel`). LFM3
already uses most of what Glimmer needs:

| Glimmer feature | Runtime support at 2d759459 | Precedent |
|---|---|---|
| Per-layer SWA, STANDARD window rule | `SlidingWindow(layer)`; ring caches for every family (`StandardDecoderModel.SlidingRings.cs`): R = align256(W + 512) cells, linear until positions reach R, whole ring under a position mask after; prompt steps at most 512 rows a sequence; truncation refused once the ring overwrote what the next row reads; copies move a ring whole | LFM3 |
| RoPE on some layers, NoPE on others, NORM flavour | `EmitQueryKey` / `RotateQueryKey` per layer; `RotarySpec(dims, neoX: false, ...)` | LFM3 (per-layer base), Qwen 3.5 |
| Per-head QK RMSNorm before RoPE | `step.RmsNorm(q, attn_q_norm)` in `EmitQueryKey`; `ReplicatedLayerWeights` under TP | LFM3 |
| Output gate from the attention input | `EmitAttentionOutput` / `GateAttentionOutput` (per op), `HeadParallelLayerWeights` (shards the gate rows in the query's head blocks) | LFM3 (head-wise; Glimmer's is elementwise, a simpler body) |
| Sandwich post norms | `MixerPostNormWeight`, `FfnPostNormWeight`; graph `EmitBlockOutput` (normed output first, for the Metal norm+add fusion) | LFM3 |
| Fused or split SwiGLU FFN | `_gateUpFused`, `KeepsGateUpApart` by measurement | LFM3 |
| Whole-step graphs, replay, decode advance | `StandardDecoderModel.Graph.cs` + `ggml_ops_graph_builder.cpp`; `DecodeStepsAdvance`, `GraphConcurrencyReorder`, `PrunesLastLayerRows`, `OverlapsPromptSteps`, `PadsPromptWindowsOnVulkan`, `PaddedCacheAttention` | LFM3, Qwen 3.5 |
| Continuous batching, slots, prefix cache, checkpoints, checkpoint files | `ExecutionPathKind.Unified`, `StandardDecoderModel.Slots/Checkpoints/PrefixCache.cs` (class P, truncation "Any" bounded by the rings, donated slots, media-span reuse) | all runtime families |
| DFlash (plain) block drafter | `StandardDecoderModel.BlockDraft.cs`: inject + block as graph programs on one device (plain DFlash: top-1 of the target head's raw softmax), the fused/per-op `ModelBase.DFlash*` below them, batched verify over slots, `SpeculationCostGovernor` | Qwen 3.8 DFlash / DFlash2 |
| n-gram speculation | shared (no drafter) | Qwen 3.5 |
| Tensor parallelism | per-rank graphs (`EmitRankedStep`, `GraphOp.AllReduce`) on ggml_cuda and the tensor executors; KV-group-aware sharding; fused gate/up sharded per segment | Qwen 3.5, LFM3 |
| Layer split, hybrid TP x PP, multi-node | `DeviceMesh`, staged graphs, `DistributedLayerSplitGroup` | Qwen 3.5 |
| Media rows | `QueueMedia` + `InjectMedia` into the embedded rows before `EmitStreamInput` (so before Glimmer's input norm, as legacy) | LFM3, Qwen 3.5 |
| Vision towers as graph programs | `NetworkGraph` (`TensorSharp.Models/Graph`): `GraphOp.Norm`, `GeluErf`, `RopeExt`, `FlashAttnExt`, `Permute`/`Cont`, run on every executor | `Lfm3VisionEncoder` |
| Executors | GGML (cuda, vulkan, metal, cpu), `MlxGraphExecutor`, `CudaGraphExecutor`, `ManagedGraphExecutor` | - |
| KV dtype default | `ApplyModelAlignedKvCacheDefault` (the same call legacy makes) | - |

### 2.2 Gaps

Each gap is runtime code (shared, so other families benefit) unless it says otherwise. None needs a
ggml change.

| # | Gap | Where it lands | Also needed by | Stage |
|---|---|---|---|---|
| G1 | **Logit transform** (scale, then tanh softcap): the runtime head has none. Graph `EmitHead` (all-row, last-row, multi-sequence rows), the ranked vocabulary-split head, the per-op head, the greedy-id head (argmax is invariant: no transform there), and the verify rows. Logprobs, sampling and the perplexity probe read these logits. | `StandardDecoderModel(.Graph).cs`: the family declares `LogitScale` and `FinalLogitSoftcap`; the runtime emits them on every logits path (`EmitLogitTransform`) and per op (`ApplyLogitTransform`), so the two paths cannot disagree (done in S1) | Gemma 4 (`final_logit_softcapping`) | S1 |
| G2 | **Post-norm epsilon** per block site: the graph `EmitBlockOutput` and the per-op path use `Config.Eps`. Glimmer's post norms take 1e-8. | `protected virtual float PostNormEps(int layer, BlockSite site)` used by the graph, per-op and per-op tensor-parallel layers (done in S1) | - | S1 |
| G3 | **Per-op stream input**: the graph has `EmitStreamInput`; the per-op path (multi-node stages, declined executors) has no twin for the weightless input RMSNorm. | per-op `ApplyStreamInput` hook, in place on the embedded rows after media injection, before the first layer (`RunLayers`, a multi-node stage's first layer) (done in S1, D6) | Gemma 4 (`sqrt(n_embd)` scale) | S1 |
| G4 | **(Measured in S7, no change, [§5.6](#56-s7-performance-status): 512-row steps on every Mac backend; rings on ggml_cpu; MLX's wider-step gain recorded.)** **Ring geometry per family / backend.** The runtime's ring is W + 512 = 2560 cells and caps prompt steps at 512 rows a sequence. Legacy prefills 2048-row chunks over a 4352-row ring. A 30B dense model's prompt GEMMs may run slower at 512 rows, so measure it. The evidence so far says no: llama.cpp prefills in 512-row ubatches by default and measured pp4096 381.4 against legacy's 382.6 on this Mac (§4.1). ggml_cpu: legacy keeps uniform caches because ggml-cpu's FA scores every masked column; the runtime rings there. | `RingStepRows` as a family/backend choice set by measurement; a CPU policy (linear caches where whole-ring reads cost more) | every SWA family (LFM3, Gemma 4) | S7 (D1) |
| G5 | **(Measured in S7, not implemented, [§5.6](#56-s7-performance-status): 0.02-0.5% of a prefill, 0.06% of a decoded token.)** **In-graph token embedding.** The runtime embeds on the host and uploads F32 rows (54 MB per 2048-row chunk). Legacy gathers the quantized table in-graph on Metal and CPU. Measure first; implement as `GetRows` over the quantized table leaf where the executor binds it zero-copy. | `StandardDecoderModel.Graph.cs`, GGML builder | every family | S7 |
| G6 | **(Done in S6, D4, [§5.5](#55-s6-multi-gpu-status): `RunsRanks` takes ggml_vulkan, whose partials the native driver sums through host staging.)** **ggml_vulkan tensor parallelism.** `GgmlGraphExecutor` declares `TensorParallel` only for CUDA, so `--tp` on Vulkan falls to the runtime's per-op TP. Legacy Glimmer runs fused per-rank graphs on Vulkan through the same native driver (`TSGgml_TensorParallelExecutePlans`). | `GgmlGraphExecutor` capability + Vulkan collective | every family on Vulkan | S6 (D4) |
| G7 | **(Done in S4, [§5.3](#53-s4-vision-status).)** **The vision tower as a program.** The pieces exist (`Norm` + bias, `GeluErf`, `FlashAttnExt`, `Permute`), but 2D NORM RoPE over the two half-heads does not. llama.cpp (`clip.cpp` `build_rope_2d`, called by `tools/mtmd/models/muse-glimmer.cpp`) rotates the first 48 dims at the column position, then the next 48 at the row position through `ggml_rope_set_offset`, which the pinned ggml has. The IR's `RopeExt` gets a dims-offset parameter (lowered to that call on GGML; the tensor executors honour it), or the tower rotates two half-head views and concatenates them. Window attention can run as `FlashAttnExt` per window over contiguous row ranges (the permutation makes them contiguous) or with one block-diagonal mask. Verify both on every executor. Also needed: a tower interface the injector takes, shared by Glimmer and Muse Spark. **Built:** both rotaries ran on managed, ggml_cpu, ggml_metal and MLX: the same rows on the first three, the offset 1.6% faster on ggml_metal and 6.7e-3 off on MLX (its tensor-executor emulation), and it needed a new IR parameter, so the half-head views are kept (no shared change); windows run one attention each (the mask was not needed). | `MuseGlimmerVisionTower` (new, on `NetworkGraph`), `ModelMultimodalInjector` | Muse Spark | S4 |
| G8 | **(Measured in S7, not ported, [§5.6](#56-s7-performance-status): the runtime's MLX real-generation decode is 3-7% above legacy's; `.MlxDecode.cs` goes in S9.)** **MLX pipelined greedy decode.** Legacy's `--benchmark` decode on MLX keeps argmax and the next embedding on the device. The runtime's MLX decode reads logits back. Port it only if the runtime's MLX real generation trails legacy: `GreedyIdOutputs` on `MlxGraphExecutor` plus a device-side embedding of the chosen id. | `MlxGraphExecutor`, the runtime's decode loop | every family on MLX | S7 (D7) |
| G9 | **The probe on the runtime.** `MuseGlimmerKvGrowProbe` reads legacy fields (`_kvCacheCapacity`, `_kvSwaRows`) for its report. Read the runtime's ring cells and slot capacity through internal accessors, and add a scenario for slot growth and checkpoint restore. **Done in S3** ([§5.2](#52-s3-serving-status)): the runtime's active-cache and slot capacities and ring cells through internal accessors; `checkpoint`, `slot-growth`, and `snapshot` as the runtime restores a prefix (a checkpoint file); every restore and rewind also mid-decode | `eng/validation/MuseGlimmerKvGrowProbe` | - | S3 |
| G10 | **The CUDA FA-VEC hazard.** Legacy materializes the padded K/V window on CUDA where ggml-cuda would pick its VEC kernel, which misread a truncated-prefix view (all query heads of a KV head got the same wrong vector; Ada+ one-row decode under 8192 rows). **Closed in S2, no change.** (1) The premise was wrong: the runtime's builder lowers `GraphOp.KvWindow` through `view_kv_cache_window`, which applies `kv_window_needs_cuda_flash_attn_copy` itself; the runtime passes the query rows and the GQA ratio (`ip[6]`, `ip[7]`), so it has always carried legacy's guard. (2) At d9fe7bff the hazard is gone on Blackwell (RTX PRO 4500, sm_120): the real-weight probe with the guard off (`TS_KV_FATTN_COPY=0`, 52 window copies fewer by a gdb count) gave byte-identical fp16 logits over 500 decode steps at 4352-4864-row windows, and the synthetic legacy freeze (head 64, GQA 2) was byte-identical too. Ada (sm_89) was not run. The guard stays (TensorSharp-owned, about 0.8% of decode there, within noise); dropping it is an S7 measurement | `ggml_ops_transformer_common.h` (already) | every family on ggml_cuda | S2 (closed) |
| G11 | **DFlash prompt catch-up width.** Legacy catches the drafter up in 1024-row trunk forwards (`TS_DFLASH_PREFILL_CHUNK`). The runtime follows the prompt in ring steps (512). Measure the DFlash prefill cost. **Closed in S5, no change** ([§5.4](#54-s5-speculation-status)): the drafter adds 4.9-5.1% to the runtime's TTFT after 0.5K-2.6K prompts and 0.5% after 16K, legacy 6.2-7.6% and 3.8%. The measurement found that a chunk past one ring step reached the drafter as its first step only (fixed, 754b5813) | `StandardDecoderModel.BlockDraft.cs` | Qwen 3.8 | S5 (closed) |
| G12 | **(Closed in S6 with no change, [§5.5](#55-s6-multi-gpu-status): the rings stay on under `--tp 2`; a 16K prompt that laps them six times gives the one-GPU text.)** **Rings under TP with one KV head per rank.** Legacy turned the ring off under `--tp` because ggml-cuda answered "17 plus 25" with fluent nonsense on a whole-ring read when a rank held one KV head (ne[2] == 1). The runtime's per-rank graphs ring by default. Gate it. If it fails, either find the kernel cause and avoid that shape in TensorSharp code, or use linear caches under TP for single-KV-head ranks (costs ~2.5 GB of KV at 64K). | `StandardDecoderModel.Placement/SlidingRings.cs` | any family with 1 KV head per rank | S6 |

### 2.3 Shared speculation

The user rule (memory `speculative-decoding-in-shared-layer`): each algorithm exists once, in the
runtime's generic layer. Glimmer adds no drafter code.

* **DFlash**: `AttachDraftModel(draftModelPath)` loads `dflash-kquant.gguf` through
  `LoadDFlashDraftWeights`. `DFlashConfig` already parses its NeoX RoPE, its window and its target
  layers. The trunk taps are `GraphRequest.CaptureLayers` = the target layers (the stream entering
  each, as `SpecForward` captured). The head stays untransformed for the drafter (`EmitDFlashTop1`
  already applies no scale or softcap) and transformed for verification (G1). Verification is the
  runtime's batched verify over slots: a request's verify window rolls back exactly because
  R >= W + rows - 1. Glimmer has no recurrent state.
* **Defaults**: `--spec-draft` 15 (the docs) against the runtime's per-family preferences
  (`SpecPreferredDraftWindow`, `SpecPreferredBlockGate`). Pick the defaults by measurement on CUDA
  and Metal (S5), as Qwen 3.8's catalog tuning did. **Done for ggml_metal in S5** (7 drafts, no gate;
  [§5.4](#54-s5-speculation-status)); CUDA keeps the speculator's defaults until the VM measures them.
* **New for Glimmer**: n-gram speculation without a drafter, DFlash with continuous batching
  (legacy has none), and DFlash across a layer split (the driver stage drafts).
* **Under `--tp`**: the runtime's graph drafter runs on one device (`GraphBlockDraftRefusal`).
  Legacy declines DFlash under TP. Keeping that decline is the parity choice (D5), declared by the family in S5 and
  checked through the real server and CLI in S6: the server refuses `--draft-model` as legacy's does.

### 2.4 Shared multimodal

* The family implements `IVisionCapableModel` / `IMultimodalPromptExpander` and calls `QueueMedia`,
  as LFM3 and Qwen 3.5 do. `InjectMedia` writes the rows into the embedded stream before
  `EmitStreamInput`. Glimmer's weightless input norm therefore runs over the merged stream, which
  is legacy's order (llama.cpp's `build_inp_embd`).
* The tower becomes a `NetworkGraph` program (G7). The same object serves Muse Spark.
  `MuseGlimmerImageProcessor` and the grid rule stay host code.
* **Prefix reuse across an image span**: the runtime declares `reuseAcrossMediaSpan: true`, while
  legacy reuses nothing past the first image. This is a gain, and it needs its exactness gate: a
  follow-up turn after an image must match a re-prefill (as
  `Qwen35ImageFollowUpExactnessTests` does for Qwen).
* The encode yields the GPU lock between blocks (`RunOutsideGpuComputeLock` as in LFM3), so
  concurrent requests keep stepping. Legacy yields between blocks through `SetHostModel`.
  **As built (S4):** the injector decodes and stretches an image off the lock, and the tower runs a
  large image as programs over layer ranges, every 1024 patches, with the lock yielded between them
  (a 448 x 448 image is one program of about 0.3 s on ggml_metal, a 1036 x 1036 one five).

### 2.5 Tensor parallelism, layer split, hybrid

* `ShardingProblems`: `NumKVHeads % tp` (2 KV heads, so `--tp` <= 2), with the legacy message. The
  refusal exits 2 with one stderr line (memory `tensorsharp-load-refusal-exit-codes`).
* `HeadParallelLayerWeights(l)` = `attn_gate.weight` (its rows follow the query heads);
  `ReplicatedLayerWeights(l)` = the Q/K norms. The post norms, `output_norm`, `token_embd` and
  `output` are replicated as usual (the vocabulary-split head is the runtime's).
* Both all-reduces land on the raw `attn_output` / `ffn_down` products, before the 1e-8 post norms:
  the runtime's `EmitBlockOutput` applies the post norm after the reduce. That is the legacy rule,
  and it is required because RMSNorm is not linear.
* `--layer-split`, hybrid and multi-node pipelines are new capabilities for Glimmer (legacy refuses
  them). They need the per-op hooks (G3) or graph-only stages (D6).
* The vision tower runs on rank 0 / the first stage, as legacy does.

### 2.6 Per executor

| Executor | What it must run for Glimmer | Risk |
|---|---|---|
| GGML ggml_metal | whole-step graphs, padded 64-row windows, ring reads as legacy (span until the wrap, whole ring after), concurrency reorder (legacy uses `graph_optimize` on persist builds only: measure on and off) | prefill step width (G4), in-graph embedding (G5) |
| GGML ggml_cuda | the same; ggml-cuda captures its own CUDA graphs; 256-padded windows for GQA grouping; device greedy ids | the FA-VEC hazard (G10) |
| GGML ggml_vulkan | the same, `PadsPromptWindowsOnVulkan` | TP (G6) |
| GGML ggml_cpu | the same; threads as today | rings vs uniform caches (G4) |
| `MlxGraphExecutor` | quantized products (IQ2_XXS / IQ2_S / IQ3_XXS / Q6_K mixes in the UD files), fused SDPA on cache views with the ring mask, elementwise gate | legacy MLX now runs at ggml_metal's speed (§4.1), so MLX is a full performance gate; pipelined decode (G8); wired-memory growth (§8) |
| `CudaGraphExecutor` (direct CUDA) | the same program; GQA decode attention; flash prefill. `--prompt-attention` (issue #383) stays f16 for Glimmer: int8 needs 256-wide heads, and Glimmer's are 128. The device-memory budget guard (`TENSORSHARP_CUDA_DEVICE_BUDGET_MB`) applies as for every runtime family | new capability (legacy is per-op); recordings for decode |
| `ManagedGraphExecutor` (cpu) | the reference | IQ2_XXS has no direct integer-dot plan (slow, as legacy) |

---

## 3. The family on the runtime

One class, `MuseGlimmerRuntimeModel : StandardDecoderModel`, in
`TensorSharp.Models/Models/MuseGlimmer/`. Its files: `MuseGlimmerRuntimeModel.cs` (the recipe and
the per-op hooks), `.Graph.cs` (the graph hooks), `.Multimodal.cs`, and `MuseGlimmerVisionTower.cs`.
The recipe, written against the hooks above:

```
ctor:   ParseBaseConfig; ResolveSwaPattern (scalar period or bool array, as legacy); window, logit scale,
        softcap, post-norm eps 1e-8; InitializeDecoder(128, 128, fingerprint "muse-glimmer|...|pattern bits");
        AttachDraftModel(draftModelPath)
SlidingWindow(l)         = isSwa[l] ? 2048 : 0
EmitQueryKey(l)          q, k = RmsNorm(q, attn_q_norm), RmsNorm(k, attn_k_norm); if isSwa[l]: Rope(NORM, base, scale)
EmitAttentionOutput(l)   attn * Sigmoid(MatMul(attn_gate, x))            (elementwise over q_dim)
EmitStreamInput          RmsNorm(embedded) without a weight, eps 1e-5
EmitBlockOutput          stream + RmsNorm(output, post_attention_norm | post_ffw_norm, 1e-8)   (G2)
EmitLogitTransform       Scale(logit_scale) -> Scale(1/cap) -> Tanh -> Scale(cap)               (G1)
MixerPostNormWeight / FfnPostNormWeight, HeadParallelLayerWeights (attn_gate), ReplicatedLayerWeights (q/k norms),
ShardingProblems (KV heads), DecodeStepsAdvance, GraphConcurrencyReorder (measured), PrunesLastLayerRows (measured),
KeepsGateUpApart (measured per backend and type), OverlapsPromptSteps, PadsPromptWindowsOnVulkan
per-op twins: RotateQueryKey, GateAttentionOutput, the stream-input and logit hooks (G3)
```

The descriptor keeps the id and aliases, the projector hints, and the factory to the new class,
and declares `ServingContract.SharedRuntime(hasRoutedExperts: false)`. The runtime validates the TP
degree. Until the switch (S8), the new class is reachable through a test seam, and the descriptor
still builds `MuseGlimmerModel`. After the switch the legacy class is reachable only from the
oracle tests, until S9 deletes it.

---

## 4. Validation plan and no-regression gates

All logs, reports and snapshots go to the ignored `artifacts/` or `docs/validation/`. Reusable tools
go in `eng/`, and fixtures in `InferenceWeb.Tests/Fixtures/MuseGlimmer/`. Every report records the
ggml revision (d9fe7bff), the device, the driver and the exact files with their sha256. A skipped or
unavailable scenario is reported as **not run**, never as passed. Tests never set process
environment variables; they use model seams.

### 4.1 Legacy baseline (measured)

Measured 2026-11-20 on the Mac: Apple M5 Pro, 48 GB unified memory, macOS 27.2, `ggml_metal` and
`mlx`. The build is TensorSharp 2d759459 (Release CLI) with ggml d9fe7bff. The weights are
`Muse-Glimmer-30B-UD-IQ2_XXS.gguf` (sha256 `3e9bd0e6...`), `mmproj-Muse-Glimmer-30B-Q8_0.gguf`
(`01ff73c9...`) and `dflash-kquant.gguf` (`27d9a805...`), all from unsloth/Muse-Glimmer-30B-GGUF
at `faa5b025`. Each run is one process under the Mac lock (exclusive) and memguard; the load average
was below 1 at each start. Every run used greedy sampling and the default KV dtype. Logs, memguard
CSVs and generated texts are in the ignored `artifacts/glimmer-baseline/out/`, and the scripts are
`artifacts/glimmer-baseline/run.sh` and `run-probe.sh`.

**`ggml_metal`**

| Measure | Command | Result |
|---|---|---|
| prefill 512 / decode at 512 | `--benchmark --bench-prefill 512 --bench-decode 128 --bench-runs 3` | pp **403.8** tok/s (402.8 mean); tg **21.9** tok/s (model only 22.0) |
| prefill 4K / decode at 4K | `--bench-prefill 4096 --bench-decode 128 --bench-runs 2` | pp **382.6** (381.1 mean); tg **21.2** |
| prefill 16K / decode at 16K | `--bench-prefill 16384 --bench-decode 64` | pp **358.9**; tg **19.9** |
| real generation, short prompt (93 tokens, `--no-skills`) | `--input p_short.txt --max-tokens 512` | TTFT (prefill complete) **417 ms**; decode **21.8** tok/s over 512 tokens |
| real generation, long prompt (4,142 tokens) | `--input p_4k.txt --max-tokens 256` | TTFT **11.20 s** (370 tok/s); decode **21.0** tok/s |
| DFlash, short prompt | as above + `--draft-model dflash-kquant.gguf --spec-draft 15` | decode **20.6** tok/s (plain 21.8); drafted 126, accepted 90 (**71%**), 77 verify steps, governor parked 320 steps (plain 46.7 vs spec 48.9 ms/token); text **identical** to plain (2,631 characters) |
| peak wired (memguard) | - | 14.2 GB (512), 15.5 GB (16K), 17.7 GB (4K real generation), 18.0 GB (DFlash) |
| DFlash, long prompt (4,142 tokens) | `--input p_4k.txt --max-tokens 256` + drafter | decode **20.9** tok/s (plain 21.0); drafted 170, accepted 117 (**69%**), parked 32; TTFT 11.58 s (plain 11.20: the drafter's catch-up adds 3.4%); text identical to plain; peak wired 24.5 GB |
| one image question each (mmproj Q8_0), 4 images | `--mmproj ... --input-jsonl vision.jsonl` (`eng/minimax-m3-bench-media`) | **4/4 correct**: "7319", "Red", "OCEAN", "There are 3 green squares in this image." Encode 1.19 s (448x224, 128 tokens), 2.22 s (448x448, 256), 1.24 s (560x196, 140), 2.17 s (448x448); prefill with the image 654-967 ms; peak wired 16.5 GB |
| 3-turn chat (`--multi-turn-jsonl turns3.jsonl`, 160/160/64 tokens) | radix reuse and prefill per turn | turn 1: 0/89 reused, 415 ms; turn 2: **80/273**, 673 ms; turn 3: **256/454**, 671 ms; turn 3 recalls name and colour correctly. Reuse stops at a 16-token page boundary before the previous reply's header, because the template drops the reasoning from history |
| perplexity, `prefill_corpus.txt` | `MuseGlimmerKvGrowProbe --nll-text` (64-token prefix, 2,246 teacher-forced one-row steps, crosses the 2048-row grow) | ppl **4.1689** (mean NLL 1.42766), top-1 agreement 62.96%; 21.4 tok/s |
| llama.cpp anchor (`llama-bench`, build 42916d83f, `-fa 1`, 6 threads) | `-p 512,4096 -n 128 -r 2` | pp512 411.0, pp4096 381.4, tg128 22.15. Legacy TensorSharp is 0.98x, 1.00x and 0.99x of it |

**`mlx`**

| Measure | Command | Result |
|---|---|---|
| prefill 512 / decode at 512 | `--benchmark --bench-prefill 512 --bench-decode 128 --bench-runs 2` | pp **424.2** (396.3 mean, the first run 368.5); tg **21.6** (pipelined greedy, `.MlxDecode.cs`) |
| prefill 4K / decode at 4K | `--bench-prefill 4096 --bench-decode 128 --bench-runs 2` | pp **396.9** (390.9 mean); tg **20.4** |
| prefill 16K / decode at 16K | `--bench-prefill 16384 --bench-decode 64` | pp **369.6**; tg **18.3** |
| real generation, short prompt (the engine's per-op path, not the pipelined one) | `--no-skills --input p_short.txt --max-tokens 512` | TTFT **538 ms**; decode **20.7** tok/s; text **identical** to ggml_metal's (2,631 characters) |
| perplexity, same text and probe | `--nll-text` | ppl **4.1692** (mean NLL 1.42772), top-1 62.96% (ggml_metal: 4.1689; the two backends are 0.007% apart) |
| peak wired | - | 14.4-17.8 GB |

The docs' MLX figures (2026-08-14: pp512 29.0, decode 14.2) are stale. MLX's IQ-quant kernels have
since reached ggml_metal's speed, so MLX is a full performance gate, not a correctness-only lane.

Not run on the Mac, and so not part of this baseline: ggml_cpu (real weights decode at about 8
tok/s), DFlash on MLX, server concurrency (it needs a second build; S3 measures both arms in one
session), and every CUDA, Vulkan and TP measure (§7).

Earlier records for context, from `docs/models/muse-glimmer.md` §5 and §6, not re-measured here:
* CUDA (RTX PRO 6000, Q8_0, 2026-08-13): pp 459-1317 and tg 26.6-35.0 from 60 to 124K tokens; DFlash
  43.5-164.6 tok/s.
* TP 2 (2x RTX PRO 4000, IQ2_XXS): pp512 1569 tok/s, tg 63.2 (one GPU: 1171 / 40.2).

Every CUDA, Vulkan and TP number is re-measured on the validation box in S0, before any runtime
arm runs.

### 4.2 Quality gates

* **Synthetic parity (every executor, every stage).** `MuseGlimmerSyntheticModelBuilder` fixtures,
  extended to 2 full + 6 SWA layers, a window shorter than the prompt, the gate, the softcap, a
  real `logit_scale` (the builder writes 1.0 today, which hides G1) and a tied and an untied head. Runtime logits vs legacy's on the same backend: ggml_cpu and ggml_metal
  (fused), managed (`cpu`, per-op), MLX, and ggml_cuda / direct CUDA on the VM. Bound: no worse
  than the legacy fused-vs-per-op floor measured on the same fixture. Memory says synthetic
  fixtures carry about 40x the rounding noise, so set the bound from the measured floor, not from
  a constant. **Bit-exact** where the emission matches node for node: the goal on ggml_cpu,
  checked by digest.
* **Frozen legacy goldens.** Before S9 deletes legacy, freeze its outputs on the fixtures (logit
  digests, a few hundred KB) into `Fixtures/MuseGlimmer/`, so coverage survives the deletion (as
  `Fixtures/Qwen35Legacy` did).
* **Real weights, greedy.** 5 prompts x 128 tokens plus the 4.6K long prompt, on ggml_metal and MLX
  (the Mac, IQ2_XXS) and ggml_cuda (VM, UD-Q4_K_XL and IQ2_XXS): token-identical to legacy, or the
  first divergence is a near-tie (legacy's own top-2 margin below 0.05 nat at that step), and the
  continuation stays on topic.
* **Teacher-forced perplexity** (`MuseGlimmerKvGrowProbe --nll-text`,
  `benchmarks/engine_comparison/assets/prefill_corpus.txt`, about 2.4K tokens, so it crosses
  Metal's 2048-row grow and the runtime's 2560-cell ring wrap). The legacy baseline is 4.1689 on
  ggml_metal and 4.1692 on MLX (§4.1). Gate: `|ppl_new / ppl_legacy - 1| <= 0.3%` and top-1
  agreement within 0.5 points, per backend. Legacy's own spread between two backends is 0.007%, so
  a runtime arm that moves more than 0.05% is investigated before it is accepted. For a wider margin, run a paired
  per-chunk comparison (`eng/validation/paired-ppl-chunks.py` style) on a 16K-token text: the
  difference must not be significant (|t| < 2).
* **Against llama.cpp** (independent): regenerate `.parity` goldens with `~/work/llama.cpp`
  (committed in S0: llama.cpp d3c3e242c; the legacy model's greedy continuations are llama.cpp's on ggml_metal and MLX)
  and run `MuseGlimmerParityTests` on the runtime. Expect the documented near-tie
  flips on IQ2_XXS (docs §4), not more.

### 4.3 Performance gates

Same session, the same build where possible (the legacy arm is the same binary through the test
seam until S8; afterwards a `~/work/TensorSharp-baseline`-style worktree at the pre-switch commit).
Interleave the arms ABBA, three rounds, and record the load average. Evict the page cache before a
load arm on the VMs (memory `gpu-benchmark-method`).

* **Decode tok/s** at depth 512, 4K and 16K (`--benchmark --bench-prefill D --bench-decode N`, and
  real generations with 512-token answers, since short answers charge the prompt-end checkpoint to
  decode): new >= legacy x 0.98 per backend.
* **Prefill** pp512 / pp4K / pp16K (`--benchmark`, and the server's chunked path): new >= legacy x 0.98.
* **TTFT**: 60-token and 4.5K-token prompts, cold and warm (CLI real generation, server
  `/v1/chat/completions` stream): new <= legacy x 1.02 + 5 ms.
* **Concurrency** (server, `eng/validation/parallel-multiturn-webui.py` lockstep c4 and
  `--sequential`): legacy serializes requests, so the runtime must beat legacy's aggregate. Each
  request's text must equal its sequential run (4/4).
* **Memory**: peak wired/VRAM <= legacy + 5% at 512, 16K and 32K contexts (the runtime sizes slots
  per request, so expect less).
* MLX and direct CUDA get the same gates against their legacy numbers. Direct CUDA's legacy is the
  per-op path, so its gate is "not slower than legacy, correct".

### 4.4 Speculation gates

* **Lossless**: DFlash text == plain greedy (up to near-ties) on the 5 prompts, on ggml_metal and
  ggml_cuda. The served output is reproducible run to run at temperature 0 (the #371 lesson:
  perplexity scores only the plain path).
* **Acceptance**: accepted / drafted within 2 points of legacy on the same prompts and settings.
  Decode tok/s >= legacy x 0.98 at 512-token answers after 60 / 500 / 2K / 16K prompts on CUDA
  (where DFlash pays: legacy 1.3-5x in the docs). On Metal it does not pay. Legacy measured 71% and
  69% acceptance at 0.94x and 1.00x of plain (§4.1), with the governor parking the drafter, so the
  runtime's drafted path must stay within 5% of plain there. A Metal speed-up is a bonus, not a gate.
* n-gram (new): quote-back prompts on Metal and CUDA, lossless; record the speed, no legacy gate.

### 4.5 Vision gates

* **Embeddings**: the tower program vs legacy `MuseGlimmerVisionEncoder` on the same backend
  (Q8_0 mmproj): cosine >= 0.9999 and relative L2 within the legacy per-op vs fused-CUDA floor,
  on 336x336 (144 tokens), 1024x1024 (1369 tokens) and a non-square portrait, on ggml_metal, MLX
  and ggml_cuda.
* **Answers**: the four `eng/minimax-m3-bench-media` questions (digits, colour, word, count) must
  pass their regexes, as legacy does (§4.1). An image description must match legacy near-verbatim.
* **Encode time**: <= legacy x 1.02 per image size (ggml_metal: 1.19 s for 128 tokens, 2.17-2.22 s
  for 256; §4.1), and recorded on MLX and CUDA.
* **Geometry**: `MuseGlimmerRegressionTests` and the parity geometry cases are unchanged.
* **Muse Spark**: its vision tests (`MuseSparkVisionProjectorTests`) pass on the ported tower.
* **Follow-up exactness** (new): an image turn followed by a text turn reuses the image prefix
  and matches a re-prefill.

### 4.6 Long context, KV growth and the ring (the legacy bug class)

* Probe on the runtime (G9): teacher-forced logits at the default sizing vs a presized cache
  (`MAX_CONTEXT=8192`), 500+ steps over a 4.6K prompt. Argmax 100% and cosine >= 0.9999 before and
  after every mark: 2048 (Metal's old grow), 2560 (the runtime ring wraps), 4096, 4352. Scenarios
  decode, split, snapshot and truncate, plus two new ones: slot growth mid-generation, and
  checkpoint restore. On ggml_metal, MLX, ggml_cpu (synthetic), ggml_cuda (VM, real weights; never
  run for legacy), and TP 2 on the 2-GPU box.
* Teacher-forced NLL at 16K and 32K: runtime vs legacy within the §4.2 bound.
* Server E2E: `webui_chat_e2e.py` with loop detection, the prompt that looped in September
  (请详细介绍最终幻想7), think on and off, 3 runs each, past 4K tokens of output. Finish reasons
  are only eos, max_tokens or thinking_budget; tail diversity >= 0.98.
* Rewind across a wrapped ring: refused when it would read overwritten cells (the runtime's
  `RingsKeepTruncation`), and the turn re-prefills. Synthetic tests carry the legacy ring-slack
  cases (`KvBlockTransferRingTests`' Glimmer rows) over to the runtime.

### 4.7 Multi-turn prefix reuse

* CLI `--multi-turn-jsonl` 3 turns: reused tokens per turn >= legacy (§4.1: 80/273 and 256/454),
  the turn-3 recall answer is correct, and TTFT for turns 2-3 is <= legacy (673 and 671 ms on
  ggml_metal). The runtime reuses exact token prefixes, not 16-token pages, so it should reach the
  last token before the previous reply's header (89 and about 270). Record the number.
* Server: 6-turn sessions (Web UI API) with and without thinking, and a tool loop
  (`client-tool-loops-get-no-server-tools` rules). Reuse matches the render (the
  `TemplateAssistantHeaderAnchor` case of `MuseGlimmerKvReuseRenderTests`), with no
  `LiveContinuationDeclined` beyond legacy's. Concurrent sessions keep their own reuse
  (the parallel KV-reuse caps in memory `parallel-kv-reuse-all-families`).
* Checkpoint files (`Persistable`): a restarted server resumes a conversation with full reuse
  (new; record it).

### 4.8 Real surfaces, end to end

* **Web UI**: `webui_chat_e2e.py` (text, image upload, thinking toggle, stop).
* **OpenAI and Ollama APIs**: `/v1/chat/completions` streamed and not (and the Ollama chat endpoint once), with tools (ATEM calls parsed into
  `tool_calls`), `response_format` `json_object` and `json_schema` with thinking on and off.
  Repeat `validate_inference.py --thinking` (docs: 38/45 on legacy, every failure a `max_tokens`
  stop) and require at least that.
* **CLI**: `--input`, `--input-jsonl` with images, `--multi-turn-jsonl`, `--draft-model`, `--think`.
* **TensorAgent** (Mac app): `TensorAgent/scripts/chat-e2e.py`, the seven scenarios, on the catalog
  entry (q8_0 KV, LeanCaches). Check the footprint against the catalog's measured 10.9 GB and the
  speed against 17.0 tok/s.
* **Sub-agents, skills, code tools**: one multi-agent run (the September loop came from a parent
  beside its children).

### 4.9 What this Mac can and cannot validate

The Mac (M5 Pro, 48 GB, 14 GiB free disk after the downloads) runs the synthetic fixtures on cpu,
ggml_cpu, ggml_metal and MLX, and real IQ2_XXS weights on ggml_metal and MLX (ggml_cpu real
weights decode at about 8 tok/s). It cannot run CUDA, Vulkan, TP, or UD-Q4_K_XL (no disk). Those
gates are **not run** until a GPU box exists (§7).

---

## 5. Staged implementation plan

Each stage is reviewable and pushable on its own. It leaves main-quality code (tests green, docs
updated for what it changes) and records its validation in the ignored `docs/validation/`.
Estimates are agent-days, including GPU runs.

| Stage | Deliverable | Main files | Tests and gates | Hardware | Effort |
|---|---|---|---|---|---:|
| **S0** Oracles | the legacy baseline (§4.1; the Mac part is in this commit, the CUDA part runs on the first VM); the probe's `--nll-text` mode (this commit) and runtime accessors (G9); llama.cpp goldens regenerated and committed as fixtures (D9); the synthetic builder extended; legacy logit digests frozen | `eng/validation/MuseGlimmerKvGrowProbe`, `InferenceWeb.Tests/MuseGlimmerSyntheticModelBuilder.cs`, `Fixtures/MuseGlimmer/` | `MuseGlimmerParityTests` run (they skip today); the legacy probe on ggml_cuda, real weights (never run before) | Mac; 1 GPU | 2 |
| **S1** Runtime hooks | G1 logit transform (graph head, ranked head, per-op head, greedy-id heads untouched), G2 post-norm epsilon, G3 per-op stream input | `StandardDecoderModel.cs`, `.Graph.cs` | hook tests on synthetic models on managed, ggml_cpu, ggml_metal and MLX; the existing runtime families' digests unchanged (Qwen 3.5, Qwen 3.8, LFM3, Hunyuan, Mistral 3) | Mac | 2 |
| **S2** Text decoder (Mac part done, [§5.1](#51-s2-the-text-decoder-status)) | `MuseGlimmerRuntimeModel` (+ `.Graph.cs`), one device, every executor, behind a test seam (the descriptor still builds legacy); G10 checked | family folder, `ggml_ops_graph_builder.cpp` only if G10 holds | §4.2 synthetic parity on every executor; real weights greedy + perplexity on ggml_metal and MLX (Mac), ggml_cuda, ggml_vulkan and direct CUDA (VM, UD-Q4_K_XL and IQ2_XXS) | Mac; 1 GPU (Ada or Blackwell) | 4 |
| **S3** Serving (Mac part done, [§5.2](#52-s3-serving-status)) | the unified step: slots, rings, prefix cache, checkpoints, checkpoint files, concurrency; the probe on the runtime | family folder; runtime fixes as found | §4.6 (every mark, every scenario, ggml_metal + MLX + ggml_cuda); §4.7; c4 concurrency 4/4 identical to sequential | Mac; 1 GPU | 4 |
| **S4** Vision (Mac part done, [§5.3](#53-s4-vision-status)) | `MuseGlimmerVisionTower` on `NetworkGraph` (G7), a tower interface used by Glimmer and Muse Spark, the injector re-typed, `QueueMedia`, reuse across image spans | `MuseGlimmerVisionTower.cs`, `.Multimodal.cs`, `ModelMultimodalInjector.cs`, `MuseSparkModel.Vision.cs` | §4.5 on ggml_metal, MLX, ggml_cuda; Muse Spark synthetic vision tests | Mac; 1 GPU | 5 |
| **S5** Speculation (Mac part done, [§5.4](#54-s5-speculation-status)) | DFlash through the shared block drafter (graph programs; fused/per-op below), n-gram, defaults by measurement (`--spec-draft`, gate), G11 | family folder (declarations only), `StandardDecoderModel.BlockDraft.cs` if G11 needs it | §4.4 on ggml_metal and ggml_cuda; MLX lossless | Mac; 1 GPU | 3 |
| **S6** Multi-GPU (done, [§5.5](#55-s6-multi-gpu-status)) | `--tp 2` on per-rank graphs (ggml_cuda; Vulkan per D4), G12, `--layer-split 2` and hybrid (new), DFlash under TP per D5 | family folder, `GgmlGraphExecutor.cs` (D4), `StandardDecoderModel.Placement.cs` (G12) | TP greedy vs one GPU (legacy tracked 468/500 characters before a paraphrase); TP decode and prefill vs legacy TP (docs: 63.2 tok/s, pp512 1569 on 2x PRO 4000, IQ2_XXS); grow and snapshot under TP (never run for legacy); layer split == one GPU | 2 GPUs in one box | 4 |
| **S7** Performance (Mac part done, [§5.6](#56-s7-performance-status)) | G4 (prompt step width, CPU ring policy), G5 (in-graph embedding), concurrency reorder, `PrunesLastLayerRows`, `KeepsGateUpApart` by measurement, G8 only if MLX trails | runtime + family folder | §4.3 on every backend (Mac: ggml_metal, MLX, ggml_cpu; VM: ggml_cuda, ggml_vulkan, direct CUDA) | Mac; 1 GPU | 5 |
| **S8** Switch (Mac part done, [§5.7](#57-s8-the-switch-status)) | the descriptor builds `MuseGlimmerRuntimeModel` with `ServingContract.SharedRuntime(false)`; usage pages (n-gram, batching, layer split), the docs, `config/muse-glimmer-30b.json` (+ drafter pin), regression pins (IQ2_XXS, drafter), TensorAgent catalog re-measured; legacy only through the oracle seam | `MuseGlimmerArchitecture.cs`, `CliUsage.cs`, `ServerUsage.cs`, docs, configs, `TensorAgent.Core/Catalog` | §4.8 on the Mac (Web UI, OpenAI, CLI, TensorAgent app) and the VM; `ServingContractConformanceTests`; both directions of the usage drift tests | Mac; 1 GPU | 3 |
| **S9** Deletion (Mac part done, [§5.8](#58-s9-the-deletion-status)) | delete §1.3's legacy files, native exports, `GgmlExportedSymbols.targets` entries, the `ggml_ops_core.cpp` hooks, MLX helpers with no other caller, legacy-only tests (their outputs frozen first); retire §1.9's names through the removed-names registry (hard errors naming the replacement); docs, website, search indexes, zh-cn | §1.3, `RemovedCliFlags.cs`, docs | whole suites on both Mac lanes and the CPU lane, the CUDA VM suite; a final real-weights smoke (text, image, DFlash, multi-turn) on Metal and CUDA | Mac; 1 GPU | 2 |

Order: S0 -> S1 -> S2 -> S3, then S4 and S5 (independent), then S6 and S7, then S8 and S9. S7's Metal
work can start right after S3. The total is about 34 agent-days.

### 5.1 S2, the text decoder: status

**The Mac part is done** (2026-11-20, M5 Pro, ggml d9fe7bff unchanged). The CUDA part (ggml_cuda, ggml_vulkan and direct
CUDA, UD-Q4_K_XL and IQ2_XXS) waits for the VM.

* **The family.** `MuseGlimmerRuntimeModel` (+ `.Graph.cs`), §3's recipe. It is `internal` and is built only by the test
  seam `MuseGlimmerArchitecture.CreateRuntimeModel(gguf, backend, ...)`, which the tests, the probe (`--family runtime`)
  and its `bench` scenario call; no option, switch or environment variable reaches it. The descriptor still builds
  `MuseGlimmerModel` (`TheDescriptorStillBuildsTheLegacyModel`).
* **Declarations.**
  * SWA pattern as legacy reads it; window 2048 on the SWA layers (39 of 52), so 2560-cell rings and 512-row prompt steps
    (D1's default).
  * QK RMSNorm, then the NORM rotary on the SWA layers only.
  * The elementwise gate from the attention input (graph `EmitAttentionOutput`, per-op `GateAttentionOutput`).
  * The weightless input RMSNorm (`EmitStreamInput`, `ApplyStreamInput`).
  * Post norms at 1e-8 (`PostNormEps`); `LogitScale` and `FinalLogitSoftcap` from the GGUF.
  * TP: `HeadParallelLayerWeights` = `attn_gate`, `ReplicatedLayerWeights` = the Q/K norms. The KV-head limit is the
    runtime's own check ("2 KV heads must each divide by N"; a `ShardingProblems` entry would repeat it).
    `ShardingProblems` adds legacy's other check: the row-parallel weights' quantization blocks a rank.
* **Measured choices: LFM3's for now, S7 measures them.**
  * `GraphConcurrencyReorder` on (decode graphs only, as legacy's persist builds).
  * `PrunesLastLayerRows` on.
  * `KeepsGateUpApart`: LFM3's rule. ggml_cuda keeps same-type Q8_0 and Q4_K pairs apart, so UD-Q4_K_XL's Q4_K layers
    differ from legacy's fused weight; ggml_cpu keeps the types it repacks apart. The IQ2_S pairs on Metal and MLX stay
    fused, as legacy.
  * `OverlapsPromptSteps`, `PadsPromptWindowsOnVulkan` and `DecodeStepsAdvance` on.
* **G10 is closed with no change** (§2.2).

| Gate (Mac) | Result |
|---|---|
| Synthetic parity vs the frozen legacy lanes (`MuseGlimmerRuntimeTests`, 7 cases: decode, prompt steps, ring wrap, KV growth 2048 -> 4096, rewind, tied head) | **passed** on managed, ggml_cpu, ggml_metal and MLX, as graphs and per op (56 runs). Bound 2e-3 (relative L2 and largest difference over the largest logit) against legacy on the same lane: legacy's own worst spread between its lanes on these fixtures is 1.365e-3 (ggml_metal vs MLX), and the bound is held to 1-1.5x it by a test. Worst per path: managed per-op 7e-6, managed graphs 1.0e-4, ggml_cpu per-op 9e-5, ggml_cpu graphs 4.6e-4, ggml_metal graphs 6.7e-4, ggml_metal per-op 1.1e-3, MLX per-op 6.9e-4, MLX graphs 1.6e-3. Every argmax is legacy's |
| Placements on simulated devices (tp 2, layer split 2, hybrid 2x2) | **passed**, within 4.5e-5 of one device |
| llama.cpp parity, real IQ2_XXS (`MuseGlimmerParityTests.Runtime*`) | ggml_metal **768/768**, every continuation llama.cpp's, worst first-position top-10 delta 0.0498 nat (legacy 0.0496). MLX **767/768**: haiku position 96 is a near-tie (0.0021 nat ours, 0.0096 llama.cpp's), worst first-position delta 0.0865 (legacy 0.0560) |
| Greedy vs frozen legacy (`MuseGlimmerRuntimeRealTests`) | ggml_metal 6/6 identical. MLX 5/6: haiku leaves legacy at position 96, where legacy's own top-2 margin is 0.0005 nat |
| Perplexity, `prefill_corpus.txt` | ggml_metal 4.168974 (legacy 4.168933, +0.0010%); MLX 4.169168 (legacy 4.169164, +0.0001%); top-1 62.96% on both, as legacy |
| Speed, probe `bench` (the CLI's procedure), ABBA in one binary | ggml_metal pp512 408.3 / 407.7 vs legacy 404.0 / 404.0, tg128 21.87 / 21.89 vs 21.83 / 21.81; pp4096 392.0 / 391.0 vs 383.3 / 385.6, tg128 at 4K 21.21 / 21.21 vs 21.26 / 21.26 (-0.2%). MLX pp512 438.8 / 438.5 vs 425.1 / 422.6, tg128 21.28 / 21.22 vs 20.59 / 20.61. Peak wired 13.0-13.2 GB vs 14.9 GB at 4K on ggml_metal |
| No regression elsewhere | **passed**. The AR regression set (S1's filter) on both Mac lanes has every pre-existing test's outcome unchanged; the only additions are the 58 new tests (cpu 1613: 1290 passed, 323 skipped; metal 1625: 1321 passed, 304 skipped; 0 failed). `RuntimeFamilyDigestTests` (42 lines a lane) is byte-identical to S1's on both lanes. No shared code changed |

On ggml_cpu the runtime's graphs sit about 3e-4 from legacy, and per op 4e-5. The cause is shared and not Glimmer's: there
(`FreshRowViews`, no `PaddedCacheAttention`) a prompt chunk attends its own fresh K/V rows in F32, where legacy and
llama.cpp attend them through the F16 cache. Fed row by row, the same prompt is 2.6e-5 from legacy.

### 5.2 S3, serving: status

**The Mac part is done** (2026-11-20, M5 Pro, ggml d9fe7bff unchanged). The family needed no change: the runtime serves it as
it serves every family, with per-request slots on the unified step, 2560-cell rings on the 39 sliding-window layers,
class P prefix pages with truncation bounded by the rings, end states copied or donated, and persistable checkpoints
(format 6: the rings whole). The CUDA part (ggml_cuda with UD-Q4_K_XL and IQ2_XXS, including `TS_KV_INITIAL_TOKENS=2048`,
plus the direct-CUDA and ggml_vulkan synthetic lanes) waits for the VM.

* **The probe on the runtime (G9).** `MuseGlimmerKvGrowProbe` reads the runtime's active-cache and slot capacities and ring
  cells through internal accessors. New scenarios:
  * `checkpoint`: a checkpoint cloned into a new request's slot;
  * `slot-growth`: one request's slot, reserved for its prompt and first token, stepped through the unified batch entry and
    grown by the decode;
  * `snapshot`: on the runtime, a checkpoint file written, everything released, read back and cloned.

  `truncate`, `checkpoint` and `snapshot` also act mid-decode (`--split` past the prompt). A rewind the rings refuse
  re-prefills, as the server's turn does. `--logits-ref`/`--logits-ref2` record per-step cosines against two reference runs.
* **Serving before the switch.** `eng/validation/MuseGlimmerRuntimeHost` runs the real `TensorSharp.Server.Host` (or
  `TensorSharp.Cli`) with the runtime family: an internal seam on the descriptor's factory
  (`MuseGlimmerArchitecture.DescriptorBuildsTheRuntimeFamily`), set only by that tool. No option, switch or environment
  variable reaches it. S8 deletes it.
* **Tests.**
  * `MuseGlimmerRuntimeServingTests` (synthetic; managed, the pinned GGML backend, MLX, and direct CUDA on the VM; 16-cell
    rings, so every script laps them) cover:
    * slot batches against each request alone, in every local placement;
    * four concurrent requests against sequential ones;
    * checkpoints of wrapped rings (clones, device copies, pooled slots, checkpoint files);
    * follow-up turns: full reuse, and a cold run's answers;
    * edited follow-ups, which rewind within the rings' slack or fall back;
    * the primary and a slot grown mid-sequence, bit for bit against presized ones;
    * rewinds that keep the window or are refused.
  * `MuseGlimmerRuntimeKvTests` re-points the legacy real-weight KV tests at the runtime, through model seams.

**§4.6, the probe gate** (UD-IQ2_XXS). Every runtime scenario ran at the default sizing and was teacher-forced against two
references: legacy presized and the runtime presized (`MAX_CONTEXT=8192`). Two configurations:
* **A** is a fixed text: a 1,600-token prefix, then 3,000 teacher-forced steps, so positions 1600-4599 put decode steps
  on both sides of every mark (2048, 2560, 4096 and 4352).
* **B** is the 4.1K chat prompt (`p_4k.txt`), then 500 steps teacher-forced by legacy's greedy tokens.

| | ggml_metal | MLX |
|---|---|---|
| runtime scenarios vs the runtime presized: decode (the primary grows at 2049 and 4097), slot growth (1792 -> 3584 -> 7168 mid-decode), checkpoint at 2600 (wrapped rings, then the clone's slot grows at 2817), checkpoint file at 4200 and 4400, rewind of 16 at 4400, prompt split 800 / 2000, rewind inside the prompt to 4000 (accepted) and to 3000 (refused, re-prefilled), checkpoint / file inside the prompt at 3000 | **passed**: 16/16 arms bit-identical (top-8 logits) at every step | **passed**: 16/16 arms argmax-identical at every step. The 10 arms that keep the prompt's step shapes are bit-identical. The 6 whose prompt cut falls off a 512-row step (splits at 800 and 2000, the checkpoint and file at 3000, the rewinds inside the prompt) carry MLX's step-shape rounding: cosine >= 0.999999, except A's split at 800 (0.999628 at one step, position 4456, where the top-two gap is 12.4 nat). Cuts on the steps (1024, 2048, 3072) are bit-identical |
| the same arms vs legacy presized | **passed**: worst cosine 0.999952 (A, past 4352) and 0.999995 (B). One teacher-forced argmax difference in A's 3,000 steps (position 2801), a near-tie: a top-two gap of 0.0015 nat in both families. It is in the presized pair too, so it is ring-geometry rounding (2560- vs 4352-cell rings), not growth | **passed**: worst cosine 0.999951 (A) and 0.999954 (B); the split at 800 is 0.999763. One argmax difference in A (position 2311), a near-tie: legacy's top-two gap is 0.0016 nat, the runtime's 0.0115. It is in the presized pair too |
| legacy default vs legacy presized (control) | 3000/3000, cosine 1.000000 | 3000/3000, cosine 1.000000 |

**§4.7, multi-turn prefix reuse** through the real server (`--no-multi-agent --no-skills`, greedy request sampling;
`eng/validation/multiturn-reuse.py`). Each follow-up turn reused exactly what the previous turn held (the previous prompt and
reply), on every turn:
* OpenAI, thinking off, 5 turns: 797, 977, 1349, 1494 tokens;
* Web UI, thinking off, 5 turns;
* Web UI, thinking on, 4 turns: 996, 1427, 2034 tokens.

That is the same as legacy on the same conversations, turn for turn (legacy reuses its live primary cache below its
4352-token cap). Follow-up TTFT on ggml_metal is 173-674 ms, against legacy's 205-693. Peak wired is 14.5 GB, against
legacy's 20.3.

**Re-prefill.** Each thinking-off conversation's last turn, sent to a fresh `--no-prefix-cache` server, answered
identically (2/2). The thinking-on conversation is not comparable that way: a fresh server renders the client's history
(1,009 tokens; the template drops past reasoning), where the session continued its raw transcript of 2,058 tokens. The
synthetic follow-up tests cover that path's exactness against cold runs. MLX gives the same: reuse equal to what the previous turn held on all 13 follow-ups (follow-up TTFT 92-618 ms), and both thinking-off replays identical.

**Concurrency** (`parallel-multiturn-webui.py --greedy --plain`, 4 conversations x 3 turns, each arm on a fresh server).
On ggml_metal the c4 run's answers are the sequential control's, **12/12 turns**, with no preemption and no capacity wait.
Its reuse equals the sequential run's on every turn. Wall time is 27.2 s against 53.6 s sequential. Legacy, which
serializes requests, took 55.8 s against its own sequential 53.8 s. It also gave one answer different from its control
("FFALCON-318"), and its follow-ups reused less under concurrency (224 against 237). MLX: 12/12, 28.0 s against 52.8 s, no preemption.

**Checkpoint files.** A 4.1K-token system prompt (past the rings' wrap) and one turn:
* the first server saved a 157 MB checkpoint of the 4,115-token shared prefix;
* a second process read it back: 4,115 tokens reused, TTFT 12.7 s -> 1.79 s;
* that reply is identical to the first run's and to a cold run's. MLX gives the same: TTFT 11.2 s -> 1.47 s, three identical replies.

**The same-chunking rule** (the CUDA S0 finding: legacy's cache operations are exact on ggml_cuda, and its near-tie flips
come from the prefill chunks' shapes).
* A cut on a step boundary must be bit-identical to the unsplit run, and it is.
* A cut off the boundary gets a near-tie analysis instead.
* On ggml_metal every runtime arm is bit-identical, aligned or not. Legacy's 2048-row chunks are bit-identical there both
  off the boundary (split 3000) and on it (2048), so ggml-metal's kernels are row-shape invariant.
* On MLX only the cut matters: aligned cuts are bit-identical, and off-boundary cuts keep every argmax.

**Also on the Mac.**
* **The CLI** (`--multi-turn-jsonl`, 3 turns, through the runtime host) reused 0/89, **86/273** and **270/454** tokens,
  against legacy's 80/273 and 256/454. It reaches the token before the previous reply's header, as §4.7 expected. The
  recall answer is legacy's.
  * Turns 2-3 TTFT is 682 and 683 ms, against legacy's 673 and 671. That is within §4.3's legacy x 1.02 + 5 ms, but not
    below legacy: the §4.7 wording.
  * The turns prefill fewer tokens (187 and 184 against 193 and 198) but no faster. S7 owns prompt-step costs.
* **The real-weight KV tests.**
  * `MuseGlimmerRuntimeKvTests` passed on ggml_metal and MLX: decode across the primary's and a slot's growth equals
    presized over 360 steps; a checkpoint of wrapped rings, cloned and read back from a file, has max |delta| 0.
  * Legacy's `MuseGlimmerKvSnapshotTests`, which used to skip here, passed 4/4 on ggml_metal.
* **No regression elsewhere.** The S1/S2 AR regression filter plus the new tests, on both Mac lanes:
  * cpu: 1646 tests, 1314 passed, 332 skipped;
  * metal: 1658 tests, 1345 passed, 313 skipped;
  * 0 failed, every pre-existing outcome unchanged, and the 33 additions are the new tests.
  * `RuntimeFamilyDigestTests` is byte-identical to S2's on both lanes.
  * The only shared-code change is two internal test accessors.

* **A shared finding, not fixed here.** On a GGML backend, `ModelBase.Dispose` runs process-wide cleanups:
  `GgmlBasicOps.ClearOffloadableState`, `ReleaseReuseComputeBuffers` and `ClearHostBufferCache`. So disposing one model frees
  the host-buffer wrappers and reuse buffers that a second live model's cached graphs bind. That model's next forward then
  faults in ggml-metal (`ggml_metal_op_norm` -> `ggml_metal_buffer_get_id`, a tensor with no buffer).
  * It reproduces with any two models on one backend: load A, load B, dispose B, forward A.
  * The serving tests keep one model alive at a time.
  * A process that serves one model at a time (the server, the CLI) never meets it. A host that unloads one GGML model
    while another stays loaded does.
  * The fix needs per-model ownership of the host-buffer registrations: clearing only the disposed model's wrappers, and
    freeing the shared reuse buffers with the last model. Leaving the wrappers would reattach stale device copies to
    recycled addresses (the 2026-09-29 bug class). That is a lifetime change for every family, so it is left for its own
    change, with the repro.

### 5.3 S4, vision: status

**The Mac part is done** (2026-11-20, M5 Pro, ggml d9fe7bff unchanged). The CUDA part (ggml_cuda, direct CUDA) waits for
the VM.

* **The position grid (a legacy defect, fixed first).** The model reads its learned 32 x 32 position table resampled
  to the *transposed* grid (the reference passes `meshgrid(pos_h, pos_w, indexing="xy")` to `grid_sample`): HF
  transformers 7394e56 (#50847) and llama.cpp d3c3e242c (#32433, "confirmed from meta team") read it so. Legacy read
  the table's rows over the grid's rows, wrong on every non-square image. The reference also zero-pads taps off the
  table (`grid_sample`'s default; vLLM and SGLang port it), where ggml's bilinear clamps: that matters only on a side of
  more than 32 patches, and llama.cpp d3c3e242c still clamps there. Both fixes are in legacy (the oracle) as their own
  commits, in `MuseGlimmerVisionGrid` (shared), and equal the reference's own arithmetic on every grid
  (`MuseGlimmerVisionPositionGridTests`, a transformers-made fixture). Real weights, ggml_metal, llama.cpp's greedy
  continuations teacher-forced: legacy with the old order 200/216 positions (first-position delta up to 1.11 nat, its
  448 x 224 description no longer names "7319"); with the order fixed 214/216 (0.09 nat or less); the short answers are
  correct either way.
* **The tower.** `MuseGlimmerVisionTower`: `NetworkGraph` programs on the allocator's executor, behind
  `IMuseGlimmerVisionTower`, which the injector takes (the legacy per-op tower implements it too). Muse Spark loads it.
  The 2-D NORM rotary runs on two half-head views (packed first on the tensor executors, whose MLX rope is host-only for a
  strided view); a dims-offset `RopeExt` was measured too and dropped (§2.2 G7). Window attention is one attention per
  window over its contiguous rows. GGML attends F16 keys and values (llama.cpp's and legacy's fused CUDA choice).
  Large images run as programs over layer ranges every 1024 patches, with the GPU lock yielded between them.
* **Images on the runtime.** `MuseGlimmerRuntimeModel.Multimodal.cs`: `IVisionCapableModel` / `IMultimodalPromptExpander`,
  `QueueMedia`; the rows are written before `EmitStreamInput`. Reuse across image spans (the runtime's
  `reuseAcrossMediaSpan`) has its exactness gate.

| Gate (Mac) | Result |
|---|---|
| Synthetic tower parity (`MuseGlimmerVisionTowerTests`: six grids, square and non-square, one to six windows) | **passed** on managed, ggml_cpu, ggml_metal and MLX against the legacy tower on the same executor: F32 products 6.1e-7 / 1.3e-4 (3.9e-7 with F32 K/V) / 5.7e-4 / 7.2e-4, bound 2e-3 (legacy's own spread between executors 4.0e-3, held under it by a test); Q8_0 products within 5e-2 (2.2e-2 is the activation quantization of ggml-cpu's and the managed executor's Q8_0 products, legacy's own spread). Both rotary lowerings: the same rows on managed, ggml_cpu and ggml_metal |
| Muse Spark on the shared tower (`MuseSparkVisionTowerTests`, `MuseSparkVisionProjectorTests`, `MuseSparkModelTests`) | **passed** |
| Real weights, rows (`MuseGlimmerVisionParityTests.TheGraphTower...`): the gate images, 336 x 336, 1036 x 1036, 588 x 868 | ggml_metal cosine 0.99996-0.99999 (relL2 3.7e-3 to 8.6e-3); MLX 0.99996-0.99999 |
| Real weights, answers (llama.cpp d3c3e242c image goldens, teacher-forced; nine prompts) | ggml_metal: runtime 215/216, legacy 214/216, misses near-ties, every answer llama.cpp's (7319, Red, OCEAN, 3, Left, bottom; descriptions token for token); MLX 215/216 each |
| Encode time (VisionEncoderBench, ABBA, one process an arm, medians of 5) | ggml_metal: 0.11-0.14x legacy (448 x 448: 306 vs 2213 ms); MLX: 0.41-0.55x (337 vs 612 ms) |
| Follow-up exactness (`MuseGlimmerImageFollowUpExactnessTests`, synthetic) | **passed** on managed, ggml_cpu, ggml_metal, MLX: the engine reuses all 996 tokens turn 1 left past the rings, stops at another image's span; logits 1.2-1.8x a text control (bound 4x) |
| No regression elsewhere | **passed**. The S2 regression set plus the Muse Spark, multimodal-injector, media, NetworkGraph, Gemma 4 vision and reflection-list classes, on both Mac lanes: every pre-existing test's outcome unchanged (cpu 2305 -> 2357, metal 2314 -> 2366; 0 failed), the additions the new tests. `RuntimeFamilyDigestTests` byte-identical on both lanes. No shared runtime, executor or native code changed |

### 5.4 S5, speculation: status

**The Mac part is done** (2026-11-20, M5 Pro, ggml d9fe7bff unchanged; on the vision branch merged at 01896b87). The CUDA
part (ggml_cuda and direct CUDA, UD-Q4_K_XL and IQ2_XXS) waits for the VM: `glimmer/cuda-vm/s5-cuda.sh` (`build`, `synth`,
`regr`, `lossless`, `speed`, `lite`, `long`, `ngram`), which also picks the CUDA defaults.

* **The family adds no drafter code.** DFlash runs on the runtime's block drafter (its inject and block graph programs on
  every executor that runs them; the fused and per-op paths below), n-gram on the runtime's weight-free speculator, the
  verify on the batched verify over slots, under the runtime's cost governor. Its declarations:
  * D5: under `--tp` the drafter is declined, as legacy declines it (the model loads without it; the host's attach refuses
    it as the model's own decline, so the server exits 2 and the CLI warns). n-gram stays available there. A whole-layer
    split drafts (staged programs; per op the runtime speculates on one device only and says so).
  * Defaults by measurement (`SpecPreferredDraftWindow`, `SpecPreferredBlockGate`; no new flag, an explicit `--spec-draft` /
    `--spec-pmin` wins): ggml_metal 7 drafts for the drafter and n-gram drafts alike (its 9-row verify costs 135-147 ms
    against 80 at 8), MLX the drafter's whole block (15 drafts: its verify costs the same up to 16 rows) and n-gram the
    speculator's 8, no confidence gate for the drafter on either. Elsewhere the speculator's defaults (8 drafts, the block
    gate 0.35) until the VM measures them (`s5-cuda.sh speed` sweeps 3/5/7/15 drafts x gates 0/0.35).
* **Five shared changes, all found by the speed gate** (DFlash lost to legacy after long prompts and on its draft, to plain
  decoding on prose, and on MLX to everything; MLX could not ask for more than 8 drafts):
  * *A prompt chunk past a ring step reaches its request's drafter whole* (754b5813). A ring family forwards a chunk
    longer than 512 rows as rounds of a step; the follow registration the first round took left the later rounds' rows
    out of the DFlash ring, and no carry was held at the prompt's end. Glimmer's prompts past 512 tokens drafted from their
    first 512 rows: 302 drafts kept after a 520-token prompt (24.3 tok/s) where legacy kept 409 (46.7). Families without
    rings take no rounds.
  * *DEFAULT CHANGE: the cost governor judges a lone request's unified step* (12b11994). The runtime families speculate on
    the unified step's batched path for one request as for several, and it never consulted the governor (only the solo
    `DecodeStep`, which they do not take, did). A step serving one sequence is now governed as the solo step is: a
    declined step verifies its anchor alone, so the drafter's ring stays in step. Steps of several sequences stay
    ungoverned. Glimmer's essay: 21.6 tok/s governed, 20.9 ungoverned, 21.9 plain. Qwen3.8-27B with DFlash2 (drafting wins
    on all five prompts): 35.7 against 36.1 (-1.1%, at most -2.2%: the calibration steps), plain 21.2.
  * *DEFAULT CHANGE: a one-pick top-k over a row wider than 4096 lowers to `ggml_argmax`* (20c5407c, the GGML graph builder).
    The plain DFlash block's top-1 over the 202,048-token vocabulary was a full argsort of every row: a block 29.3 ms, legacy's
    fused drafter 25.4; with the argmax 24.8. Five prompts 34.1 -> 35.0 tok/s, 16K 24.4 -> 25.2. Routers and greedy heads keep
    their lowering; the CUDA executor already took this pick.
  * *The tensor executors' host top-k selects a row's few picks in one pass* (f3fd6d9d, `TensorGraphExecutor.TopK`: the
    managed CPU's and MLX's). It sorted every row whole with a comparer: on MLX a block took 274-412 ms (15 rows of 202,048)
    and the governor parked the drafter (18.0 tok/s against 21.1 plain); with the selection 23-34 ms, the same picks in the
    same order.
  * *A trunk's preferred draft window replaces the speculator's default, wider as well as narrower* (a0197dc6,
    `SpeculatorRegistry.ResolveDraftWindow`). Every other family prefers 8 or less, so none changes.
* **G11 (the drafter's prompt catch-up width): no change.** The runtime follows the prompt inside the trunk's own 512-row
  steps (the encoder and the draft layers' K/V over the rows the ring keeps at the prompt's end), where legacy catches up
  in 1024-row trunk forwards whose features cross the host. The drafter adds 4.9-5.1% to the runtime's TTFT after 0.5K and
  2.6K prompts and 0.5% after 16K (the ring keeps a prompt's last 2,080 rows only), against legacy's 7.6%, 6.2% and 3.8%:
  the runtime's DFlash TTFT is 3-4% below legacy's.
* **Found on the way (not S5's).** `MuseGlimmerRuntimeTests.TheParityBoundIsLegacysOwnSpread` failed at 01896b87: with the
  CUDA box's frozen lanes among the fixtures, legacy's spread between lanes is 3.29e-3 (cuda vs ggml_vulkan), past the 2e-3
  bound. The coordinator's 29fa6649 (the bound with the GPU box's lanes, 4e-3) applied onto this branch during S5 fixes it.

All runs: UD-IQ2_XXS + `dflash-kquant.gguf`, greedy, one process an arm under the Mac lock and memguard (43 GB), the same
build for both families. "mix5": an essay (93 tokens), a code module, a list, a 520-token and a 2.6K-token rewrite, 512
tokens each (`artifacts/glimmer-s5/prompts/mix5.jsonl`); "16K": 16.2K tokens of the corpus and a rewrite, 512 tokens.

| Gate (Mac) | Result |
|---|---|
| Lossless, mix5 and 16K (texts, CLI) | **passed**. At the family's defaults every text equals plain greedy's except the essay, which leaves plain at one token (character 2230, "mechanically tabulate" for "automate") wherever an 8-row verify covers it: legacy at the same settings and n-gram drafts flip it too, and governed runs that park past it do not (a 0.0084-nat tie: the real-weight test on MLX below). After 16K the runtime's DFlash text is plain's; legacy's DFlash leaves its own plain at character 99 ("\n" for "\n\n") |
| Lossless, real-weight test (`MuseGlimmerRuntimeSpeculationRealTests`, the engine as served: the five golden prompts and the 4.7K one x 128 tokens at the defaults and ungated at 15 drafts, one at a time and all six at once; the three chat prompts x 512) | ggml_metal: **passed**, 3/3 tests. Every stream equals plain greedy's token for token: the six golden prompts (5 x 128, the 4.7K one x 128) at the defaults (drafted 2,369, kept 1,060) and ungated at 15 drafts (3,167, 1,049), one at a time and all six at once; the three chat prompts x 512 at 7 drafts (4,580, 1,823), one at a time and together. No divergence, so no near-tie to resume past. MLX: **passed**, 3/3: haiku leaves plain at token 96 ("lit" against " lit", -0.0077 nat: the plain model over the common prefix prefers DFlash's), the essay at 437 (" automate" against " mechanically", -0.0084 nat: the CLI's near-tie), and the essay served beside the others at 373 (0.0115 nat); past each the plain model picks every later DFlash token but one 0.0115-nat tie; the other streams identical, alone and together |
| Speed against plain and legacy, mix5 (decode tok/s: essay / code / list / 0.5K / 2.6K, all) | plain: runtime 21.9 / 21.9 / 21.9 / 21.7 / 21.3, **21.7**; legacy 21.8 / 21.8 / 21.8 / 21.6 / 21.2, 21.7. DFlash, runtime at its defaults: 21.5 / 48.0 / 33.8 / 61.9 / 33.8, **35.0** (1.61x plain); legacy at its default (8 drafts, gate 0.35) 20.7 / 37.5 / 29.2 / 46.7 / 31.3, 30.7; legacy at the runtime's settings (7, no gate) 20.4 / 43.2 / 31.5 / 59.4 / 32.5, 32.9; legacy `--spec-draft 15` 20.7 / 39.1 / 29.0 / 59.5 / 32.0, 31.8. **passed**: the runtime's DFlash is at least legacy's on every prompt, at both families' defaults and at equal settings; the essay (drafts lose) is 0.98x plain governed (parked 64 steps), 0.95x ungoverned |
| Speed after 16K (512 tokens) | plain 20.2 both; DFlash runtime **25.2** (1.25x), legacy 24.9. **passed** |
| Acceptance (accepted drafts per verify step, the gate's ratio confounded by drafts per step) | equal settings (7, no gate): runtime 1.39 / 4.19 / 2.62 / 5.76 / 2.85 against legacy 1.15 / 3.77 / 2.45 / 5.76 / 2.77; accepted / drafted 20-84% against 20-87%: within 2 points on the essay and the code, 3.7-5.8 points lower on the list and the rewrites, where the runtime drafts more a step (6.9 against 5.7 on the essay) and keeps as many or more. **passed** on drafts kept a step |
| G11 TTFT (ms; plain -> DFlash) | runtime: 0.5K 1581 -> 1662 (+5.1%), 2.6K 5509 -> 5780 (+4.9%), 16K 44389 -> 44619 (+0.5%); legacy: 1601 -> 1722 (+7.6%), 5659 -> 6013 (+6.2%), 44777 -> 46470 (+3.8%). **passed**, no change: the runtime's DFlash TTFT is 3-4% below legacy's |
| Governor (the drafted path never slower than plain by default) | DFlash: the essay 21.5-21.6 tok/s governed against 21.9 plain (0.98-0.99x), every other prompt 1.25-2.85x. n-gram: see below. **passed** within the §4.4 5% |
| n-gram (new; quote-back, a code edit, the essay, the code module) | ggml_metal at the family's 7 drafts: **93.0 / 54.2 / 21.5 / 22.9** tok/s against plain 21.7 / 21.7 / 21.8 / 21.8 (4.3x and 2.5x where the answer quotes its prompt; 0.99x and 1.05x on free text); at the speculator's 8 (a 9-row verify, 135-147 ms) 59.5 / 37.0 / 20.7 / 20.9, ungoverned 59.9 / 36.7 / 19.1 / 18.6. MLX at 8 drafts (its 9-row verify 80 ms): 105.0 / 58.5 / 21.1 / 22.6 against 21.0 / 21.2 / 21.4 / 21.4. Every text plain greedy's but the essay's near-tie token on MLX and at 8 drafts on ggml_metal. **passed** (lossless; no legacy gate: legacy has no n-gram) |
| DFlash with continuous batching | synthetic (managed, ggml_cpu, ggml_metal, MLX): one request and three at once through the engine equal plain's streams, the batched verify ran. Real weights (ggml_metal): the six golden prompts at once and the three chat prompts at once, each stream identical to its sequential DFlash run and to plain greedy. **passed** |
| DFlash across a layer split | synthetic, two stages on simulated devices (staged programs): plain greedy's stream; per op the runtime refuses speculation off one device ("runs on one device"). **passed** |
| D5, the drafter under `--tp` | synthetic: declined at load (tp 2 on simulated devices), n-gram available; the host's attach refuses it as the model's own (the shared loader's test). **passed**; the real server's exit 2 needs two GPUs (S6), **not run** |
| MLX | DFlash (the runtime's graph drafter; legacy's per-op one), mix5: plain runtime 21.4 / 21.3 / 21.3 / 21.0 / 20.7, **21.1**, legacy 20.6 / 20.6 / 20.7 / 20.2 / 19.5, 20.3. Runtime at its defaults (15 drafts, no gate) 22.6 / 65.0 / 35.7 / 97.3 / 36.6, **39.7** (1.88x plain; 8 drafts ungated 37.0, 7 ungated 35.0, 3 ungated 26.5, the speculator's 8 at the gate 33.4); legacy at its default 19.1 / 43.1 / 30.8 / 57.5 / 31.2, 31.7. **passed**: above legacy on every prompt, and the essay above plain. Before the top-k fix the runtime's MLX blocks took 274-412 ms and it decoded 18.0-18.1 tok/s (0.85x plain). Lossless: every text plain's but the essay's 0.008-nat tie, which MLX's plain greedy and legacy's plain greedy decide apart too. Memory: peak wired 20.3 GB with DFlash, 16.9 plain (legacy 19.7 / 18.0) |
| Memory (peak wired, mix5) | runtime DFlash 15.9-16.1 GB, legacy DFlash 21.4-25.1; plain 14.0 / 17.9 |
| No regression elsewhere | **passed**. The S3 regression filter plus the speculation classes (SpeculativeExecution, SpeculatorRegistry, SpeculativeGreedyVerify, BatchedVerifyHeldLogits, BatchedDecodeRecording, DistributedSpeculation, BlockDraftCausalWidth, SpeculativeDraftHeadLoader, ExecutionPlanner, DraftContextContinuation; the S3 set already holds Qwen 3.5, Qwen 3.8 with DFlash/DSpine, LFM3, GraphBlockDraft) on both Mac lanes, 01896b87's shared code against the final tree: cpu 1943 -> 1950, metal 1955 -> 1962, 0 failed. Every pre-existing outcome unchanged but three that now pass: the two ring-step follow tests (754b5813) and `TheParityBoundIsLegacysOwnSpread` (the coordinator's 29fa6649); the additions are the new tests and the coordinator's fixture rows. Two `DraftContextContinuationTests` failed under the governor (a third request inside a park) and take `GovernorDisabledForTests` now. `RuntimeFamilyDigestTests` byte-identical to S3's (42 lines a lane). After the window change (a0197dc6, 6851774b) again: cpu 1953, metal 1965, 0 failed, the additions the three registry rows |

### 5.5 S6, multi-GPU: status

**Done** (2026-11-20, ggml d9fe7bff unchanged before and after every build). The box: 2x RTX PRO 4500 Blackwell Server
Edition (sm_120, 32 GB each, PCIe, topology SYS, peer copies checked numerically both ways), driver 595.112.07, CUDA 13.2,
NCCL 2.32.6, Vulkan loader 1.3.275. Weights UD-Q4_K_XL and UD-IQ2_XXS (unsloth @ faa5b025, sha256 checked); for other
families Qwen3.5-0.8B Q8_0, LFM3-1.7B Q4_K_M and Qwen3.5-9B Q4_K_M. One model process on the GPUs at a time, the model's
pages evicted before every load arm. Logs are in the VM's ignored `artifacts/glimmer-tp/`.

* **Legacy's `--tp 2` baseline** (S0 never ran it), one GPU -> `--tp 2`:

  | | Q4_K_XL ggml_cuda | IQ2_XXS ggml_cuda | IQ2_XXS ggml_vulkan |
  |---|---|---|---|
  | prefill 512 / 4K / 16K (tok/s) | 1697/1817/1743 -> 2479/2839/2704 | 1546/1626/1569 -> 2309/2590/2474 | 1296/1272/1169 -> 1977/2242/1892 |
  | decode at 512 / 4K / 16K | 42.3/41.4/38.3 -> 68.6/66.6/44.2 | 50.0/48.6/44.7 -> 80.6/77.3/44.3 | 35.7/35.4/31.7 -> 52.5/51.9/32.0 |
  | perplexity | 3.6063 -> 3.6050 | 4.1724 -> 4.1695 | 4.1695 -> 4.1700 |
  | p_4k, 500 steps teacher-forced to one GPU | cos 0.999853, 3 flips, KL 1.71e-3 | cos 0.99973, 6 flips (one 0.84-nat swing), KL 1.47e-3 | cos 0.999938, 0 flips, KL 1.9e-4 |

  Legacy keeps uniform caches under `--tp`, so its 16K decode falls back to the one-GPU rate. Grow and snapshot under TP
  (never run before) give its own TP bytes when the cut is on its 2048-row chunks; the unaligned snapshot at 2560 flips
  near-ties (S0's chunk-shape effect).
* **The family needed no tensor-parallel change.** §2.5 holds as built: per-rank graphs, the rings on (2560 cells, one KV
  head a rank), both all-reduces on the raw `attn_output` / `ffn_down` products before the 1e-8 post norms
  (`EmitRankedStep` reduces the block's partial, then `EmitBlockOutput` norms it); images under `--tp 2` get the one-GPU
  answers.
* **G12 is closed with no change.** A 16,195-token prompt laps the rings about six times and reads them whole with one KV
  head a rank; its 512-token greedy answer is the one-GPU text byte for byte on both quants and both GPU backends, and
  "17 plus 25" is 42. The rings stay on under `--tp`, so the runtime's 16K decode under `--tp 2` is 1.35-1.39x legacy's.
* **D5 is in.** The family attaches no drafter on a tensor-parallel trunk (before, it drafted under `--tp 2`). The server
  exits 2 with legacy's stderr line for line, one `error: model load refused` line last; the CLI warns, decodes without
  the drafter and answers 42.
* **D4 is in, as a DEFAULT CHANGE.** `GgmlGraphExecutor.RunsRanks` takes ggml_vulkan as well as ggml_cuda: the native
  driver (`tsg::tp_execute_plans`) already runs the ranks' plans together there, and ggml-vulkan's collective
  (`ggml-vulkan-comm`: each partial through an imported host buffer, its peers adding it from there) sums them, legacy's
  Vulkan transport.
  Before, `--tp` on Vulkan fell to the per-op tensor-parallel path: Glimmer decoded at 2.9 tok/s (legacy Vulkan TP 52.5).
  Every runtime family with ranked lowering moves with it: Qwen3.5-0.8B's Vulkan `--tp 2` decode went from 9.5 to 194.9
  tok/s and LFM3-1.7B's from 6.5 to 243.6, each with the per-op path's greedy text byte for byte.
* **The whole LM head under per-rank TP (a DEFAULT CHANGE, every family).** Rank 0 kept the whole `output.weight` resident
  beside the vocabulary-split slices, +430 MiB a GPU at short context against legacy. With the split on and no draft
  layer, the whole head now stays on the host and its readers (the all-row logits of speculative verify steps, a draft
  model's head, the per-op path) bind it on rank 0 when they first run. Rank 0 holds 1052 MB less for Glimmer IQ2_XXS (882
  for Q4_K_XL, 796 for Qwen3.5-9B); every output is byte-identical (greedy and n-gram texts, teacher-forced logits); with
  n-gram speculation the head comes back on the first verify step, so the peak is the old one there.
* **`--layer-split 2`** (new for Glimmer): byte-identical to one GPU (the 500-step teacher-forced and free-running fp16
  logits, and the perplexity, on both quants).
* **Hybrid TP x PP was not run**: two stages of two ranks need four devices. The hosts refuse `--tp 2 --layer-split 2` at
  load (exit 2, one line) because the descriptor's serving contract is still legacy's until S8 (D11). The simulated-device
  tests (`PlacementsOnSimulatedDevicesGiveTheOneDeviceLogits`, tp 2 x 2 stages) stand for it.
* **Prompt steps where the ranks' sums cross the host (a DEFAULT CHANGE, every family).** Each prompt step pays a host
  round trip for each of its two all-reduces a layer where they go through host memory, so 512-row steps cross four times
  as often as legacy's 2048-row chunks: Vulkan `--tp 2` prefill was 0.83x legacy at 4K and 0.89x at 16K. The placement now
  chooses the step: `IRankedGraphExecutor.RanksSumOnHost` (ggml-vulkan's collective, or no collective that passed its
  probe), recorded by the mesh, and `StandardDecoderModel.RingStepRowsForPlacement`, the step width's seam for D1 / G4:
  2048 rows there (rings of W + 2048 cells), 512 everywhere else (one device, layer split, ggml-cuda with NCCL or peers,
  the tensor executors). Glimmer's Vulkan `--tp 2` prefill is now 0.991x / 0.995x / 1.045x legacy at 512 / 4K / 16K, decode
  unchanged, 341-419 MiB a GPU more for the rings (still under legacy); greedy texts unchanged, and every other path's
  logits byte-identical. LFM3-1.7B's Vulkan `--tp 2` answers are byte-identical too.
* **For S7:**
  * The step-width seam above (`RingStepRowsForPlacement`) is where a measured per-backend width (Metal, MLX, ggml_cpu)
    goes; it changes no path off `--tp` today.
  * Serving on ggml_cuda is not shape-invariant (not a TP fault: one GPU shows it, and legacy where it batches nothing).
    A c4 step batches the four sequences' rows, and ggml-cuda rounds a 4-row step differently from a 1-row step, so three
    of four first answers leave their solo answer at a near-tie paraphrase (characters 153-468), under `--tp 2` and on one
    GPU alike; ggml_metal and MLX give 12/12 (§5.2), and legacy 12/12 only by serializing the requests (2.5x the wall time
    under TP). A reused turn against a re-prefill differs the same way, for legacy too. §4.3's "4/4 identical" (and S3's
    re-prefill replays) hold on Metal and MLX only; on ggml_cuda they need a near-tie-aware form, as the speculation gates
    have.

| Gate | Result |
|---|---|
| ggml_cuda `--tp 2` vs one GPU, p_4k teacher-forced (cos min / flips / KL mean) | **accepted** as legacy's noise class: Q4_K_XL 0.999848 / 5 / 2.23e-3 (legacy 0.999853 / 3 / 1.71e-3), IQ2_XXS 0.999677 / 7 / 8.0e-4 (legacy 0.99973 / 6 / 1.47e-3); every flip at a one-GPU top-two gap of 0.29 nat or less. Summing every partial exactly in F32 (`GGML_CUDA_ALLREDUCE=internal`, `GGML_CUDA_AR_BF16_THRESHOLD=0`; NCCL carries sums of 32,768+ floats in BF16), the runtime is no worse than legacy: Q4 3 flips, KL 1.64e-3 (legacy 3, 1.80e-3); IQ2 5 flips at gaps of 0.161 nat or less, KL 7.4e-4 (legacy 3 at 0.226 or less, 8.8e-4). The rest is the row-parallel products' split sums, in both families |
| ggml_cuda perplexity, one GPU -> `--tp 2` | **passed**: 3.6043 -> 3.6060 (+0.05%), 4.1718 -> 4.1695 (-0.06%) |
| G12: 16K-prompt greedy and "17 plus 25" under `--tp 2` | **passed** on ggml_cuda (both quants) and ggml_vulkan: identical to one GPU; 42 |
| ggml_cuda speed, ABBA in one binary, runtime / legacy `--tp 2` | **passed**: prefill 1.25 / 1.07 / 1.04 (Q4_K_XL, 512 / 4K / 16K), 1.18 / 1.00 / 1.01 (IQ2_XXS); decode 1.03 / 1.02 / 1.37 and 1.05 / 1.04 / 1.39 |
| Peak VRAM per GPU vs legacy `--tp 2` | **passed** after the head change: at 512, IQ2_XXS 5658 + 5661 MiB against legacy's 6286 + 5227, Q4_K_XL 8250 + 8237 against 8722 + 7835 (the larger GPU 5-10% lower, totals 0.4-1.7% lower; the slices balance the head over both GPUs). Before it, +4.7% and +7.5%. At 16K 5-7% lower (the rings) |
| ggml_vulkan `--tp 2` vs one GPU (strict: cos >= 0.9999, no flip above 0.15 nat) | **passed**: IQ2_XXS cos 0.999930, one flip at a 0.008-nat tie, KL 1.94e-4 (legacy's own pair 0.999938, none, 1.90e-4); perplexity 4.1695 -> 4.1700, as legacy's. With 2048-row steps under `--tp`: cos 0.999933, one flip at 0.056 nat, KL 2.49e-4 against one GPU's 512-row steps (1.80e-4 against one GPU at 2048 rows) |
| ggml_vulkan speed vs legacy Vulkan `--tp 2` (within 3%) | **passed** with the host-staged step rule: prefill 0.991 / 0.995 / 1.045 at 512 / 4K / 16K, decode 1.08 / 1.04 / 1.43 (ABBA). With 512-row steps it had failed at 4K and 16K (0.83, 0.89) |
| `--layer-split 2` vs one GPU | **passed**: byte-identical logits and perplexity, both quants |
| Serving under `--tp 2` (IQ2_XXS, the real server): multi-turn reuse | **passed** for the reuse: every later turn reuses its previous transcript whole on OpenAI and the Web UI (waste 0 or -1), as legacy's. A reused last turn's reply is not a fresh re-prefill's on ggml_cuda (`--replay`: runtime `--tp 2` 0 of 2 identical; the controls, runtime on one GPU and legacy `--tp 2`, 1 of 2 each), each a near-tie paraphrase from ggml-cuda's shape-dependent rounding, the c4 class below |
| Serving under `--tp 2`: c4 vs sequential | **failed** (3 of 4 conversations leave their sequential answer in turn 1), **as on one ggml_cuda GPU** (the same 3 of 4): a ggml_cuda property of batched serving, not of `--tp` (above, S7) |
| Images under `--tp 2` | **passed**: the four gate images answered as llama.cpp and legacy answer them (7319, Red, OCEAN, 3), on ggml_cuda and ggml_vulkan |
| DFlash under `--tp` (D5) | **passed**: declined as legacy declines it |
| Synthetic two-GPU tests, cuda and vulkan pins | **passed.** New: `PlacementsOnTwoGgml{Cuda,Vulkan}GpusGiveTheOneGpuLogits` (decode, ring-wrap, tied ring-wrap): tp 2 within 5.2e-3 (cuda) and 2.2e-3 (vulkan) of one GPU against legacy's own `--tp 2` spread of 4.6e-3 and 3.0e-3 on the same cases (the bound is 1.5x legacy's); two stages give the one-GPU bytes. The Qwen 3.5 and SharedRuntime Vulkan rank-graph rows pass; the GGML rank-graph rows also run speculation with the head on the host |
| No regression elsewhere | **passed.** The runtime families' test classes (Qwen 3.5, Qwen 3.8, LFM3, SharedRuntime, Glimmer, block drafts, TP gates) on both pins, test by test against the base commit: 0 changed outcomes (the 5 failures on the cuda pin and 3 on the vulkan pin fail at the base too). On the Mac, S4's regression filter on the cpu and metal lanes keeps every outcome, and `RuntimeFamilyDigestTests` is byte-identical |
| Hybrid TP x PP | **not run** (needs four devices) |

### 5.6 S7, performance: status

**The Mac part is done** (2026-11-20, M5 Pro, ggml d9fe7bff unchanged): ggml_metal (UD-IQ2_XXS and UD-Q4_K_XL), MLX and
ggml_cpu (UD-IQ2_XXS). The CUDA part (ggml_cuda, ggml_vulkan and direct CUDA, both quants) waits for the VM:
`glimmer/cuda-vm/s7-cuda.sh` (`build`, `bench`, `settings`, `cli`, `serve`, `sum`). One setting changed, for this family on
ggml_cpu only (its IQ2_S gate/up pairs apart, a DEFAULT CHANGE); every other measured switch kept its value, the prompt
step stays 512 rows on every Mac backend, and no shared runtime, executor or native code changed.

* **The tools.** The probe's bench takes the runtime's measured settings as arms in one binary (internal seams only, no
  option of the shipped hosts): `--ring-step-rows N|linear` (`RingGeometryForTests`), `--gate-up apart|fused`
  (`MuseGlimmerRuntimeModel.GateUpApartForTests`), `--no-prune` (`LastLayerPruningDisabledForTests`) and `--bench-refill`;
  the concurrency reorder is ggml-metal's own `TS_METAL_GRAPH_OPTIMIZE=0`. Its report records the host embedding time of
  each run. Real generation ran through the CLI and the real server (`MuseGlimmerRuntimeHost` for the runtime).
* **Method.** Every arm is its own process under the Mac lock and memguard, ABBA (legacy, runtime, runtime, legacy; or A, B,
  B, A for a setting), the same build on both sides. The bench is the CLI's `--benchmark` procedure; real generation is the
  CLI (`--input`, 93 tokens x 512 and 4,142 x 256) and the server (`/v1/chat/completions`, streamed, greedy, 113 x 512 and
  4,162 x 256, cold). Logs in the ignored `artifacts/glimmer-s7/`.

| Gate (§4.3) | ggml_metal IQ2_XXS | ggml_metal Q4_K_XL | MLX IQ2_XXS | ggml_cpu IQ2_XXS |
|---|---|---|---|---|
| prefill, runtime / legacy (512 / 4K / 16K) | 1.005 / 1.017 / 1.041 (32K 1.027) | 1.026 / 1.031 / 1.029 | 1.034 / 1.049 / 1.112 | 1.004-1.013 / 1.021 / not run |
| decode, runtime / legacy (at 512 / 4K / 16K) | 1.006 / 0.999 / 1.006 (32K 1.007) | 1.006 / 1.001 / 1.008 | 1.031 / 1.048 / 1.118 | 0.984-0.987 / 0.98-1.05 (noisy) / not run |
| CLI TTFT, short / 4K | 0.982 / 0.982 | 0.916 / 0.969 | 0.821 / 0.910 | 0.980 / 0.979 |
| CLI decode, short / 4K | 1.002 / 1.000 | 1.007 / 1.005 | 1.034 / 1.069 | 1.006 / 0.993 (one pair) |
| server TTFT / decode, short; 4K | 0.980 / 1.005; 0.990 / 1.002 | 0.921 / 1.008; 0.974 / 1.005 | 0.869 / 1.039; 0.926 / 1.069 | 0.978 / 1.004 (113 x 256); 4K not run |
| peak memory, legacy -> runtime (GB) | wired 15.9-17.0 -> 13.8-15.0 (32K 16.9 -> 14.5) | wired 20.1-21.3 -> 19.2-19.6 | wired 16.0-19.9 -> 15.9-18.3 | resident 21.6-23.8 -> 13.4 (17.1-18.1 before 624b2860) |
| result | **passed** | **passed** | **passed** | **passed** at 512 and 4K (the bench decode 1.3-1.6% under, in the gate; real generation equal); 16K **not run** |

* **The measured switches** (A/B on the runtime; ratios are the other setting against the current one):
  * `GraphConcurrencyReorder` on: off decodes 1.4-1.5% slower on IQ2_XXS (512 and 4K), 0.2-0.3% on Q4_K_XL; prompts
    unchanged (the runtime reorders decode graphs only). It is ggml-metal's only.
  * `PrunesLastLayerRows` on: off prefills 1.4-1.5% slower at 512 on ggml_metal (both quants) and MLX; decode unchanged.
    On ggml_cpu off is 1.2% slower (one clean B arm).
  * `KeepsGateUpApart`: fused stays. Apart (the IQ2_S pairs, or Q4_K_XL's Q4_K pairs) measured 0.997-1.000 on ggml_metal
    and 0.975-0.991 on MLX. On ggml_cpu the IQ2_S pairs now stay apart (624b2860, a DEFAULT CHANGE for this family only): the same speed and byte-identical logits, 4.4 GB less resident (17.8 -> 13.4 GB), because ggml-cpu then reads both products from the mapped GGUF instead of an anonymous fused copy. The CUDA rule (Q4_K and Q8_0 apart on ggml_cuda) is the VM's to measure.
* **G4 / D1, the step width: 512 everywhere on the Mac.** The runtime's prefill at 512-row steps is legacy's or faster on
  every backend (above), so D1 keeps it. 2048-row steps: ggml_metal 1.001-1.002 (IQ2_XXS) and 0.987-0.989 (Q4_K_XL), at
  0.4-0.7 GB more wired. MLX is the exception, recorded and not changed: 2048 rows prefill 4.4-5.4% faster at 4K (1.8% at
  16K) but decode 0.9-1.3% slower and wire 3 GB more; 1024 rows prefill 2.3% faster at 4K with decode unchanged. The cause
  is per-step host time: `MlxGraphExecutor` has no `OverlapSteps`, so a step's host work does not overlap the previous
  step's GPU work (about 60-70 ms a 512-row step). The fix, if wanted, is that overlap (every MLX family), or 1024 rows on
  MLX through S6's `RingStepRowsForPlacement`.
* **G4, the CPU ring policy: rings.** Legacy's reason for uniform caches (its 4352-row rings read whole once wrapped) does not carry over to 2560-cell rings. At 4K, both arms in 512-row steps, linear caches prefill 1.001x and decode 0.985x the rings, at 2.9 GB more resident memory, so no CPU policy is added. ggml_cpu at 16K was not run: an arm prefills for 32-45 min, past the Mac lock's budget, and 4K already reads the wrapped rings whole.
* **G5: not implemented.** The host lookup and upload of a step's rows cost 0.06-0.17% of a prefill on ggml_metal, 0.02-0.48%
  on MLX and under 0.01% on ggml_cpu, and 0.02-0.06% of a decoded token; Metal's prompt steps overlap it.
* **G8 / D7: not ported; `MuseGlimmerModel.MlxDecode.cs` is deleted with legacy in S9.** The runtime's MLX real-generation
  decode is 3.4-6.9% above legacy's (CLI and server). Against legacy's benchmark-only pipelined greedy path it is 1.4% below
  at depth 512 and equal at 4K (CLI `--benchmark`, 21.2 against 21.5 tok/s; 20.4 both).
* **Found on the way.** An identical stateless `/v1/chat/completions` resend reuses no prefix on either family (the radix has
  no end state for it); follow-up turns do (§5.2). The CPU and MLX benchmarks ran beside another agent's builds and jobs at
  times; `artifacts/glimmer-s7/out/cpuwatch.log` records the other processes' load from 12:10 on, and the noisy rounds were
  repeated.

### 5.7 S8, the switch: status

**The Mac part is done** (2026-11-20, M5 Pro, ggml d9fe7bff unchanged). The VM part (the same E2E on ggml_cuda and direct CUDA)
waits for the CUDA box: `glimmer/cuda-vm/s8-cuda.sh` (`build`, `cli B Q`, `server B Q`, `tests`).

* **The switch (e6df6ff5, a DEFAULT CHANGE).** The descriptor builds `MuseGlimmerRuntimeModel` with
  `ServingContract.SharedRuntime(hasRoutedExperts: false)`. The legacy `MuseGlimmerModel` is reachable only through the oracle
  seam `MuseGlimmerArchitecture.CreateLegacyModel` (the frozen and parity tests, `MuseGlimmerKvGrowTests`,
  `MuseGlimmerKvSnapshotTests`, the probe's `--family legacy`); no option, switch or environment variable selects it. The
  serving seam `DescriptorBuildsTheRuntimeFamily` and `eng/validation/MuseGlimmerRuntimeHost` are deleted.
* **Shared tests.** `ServingContractConformanceTests` takes the family off the bespoke list and its gap rows;
  `SharedRuntimeConformanceTests.DedicatedSuites` proves its claims through `MuseGlimmerRuntimeConformanceTests` (placements,
  batched slots, continuous batching, checkpoints, DFlash and n-gram through the engine, graphs against per op, on fixtures
  whose rings lap); `MuseGlimmerRuntimeTests.TheDescriptorBuildsTheRuntimeFamily` replaces the S2 assertion. The other
  inventory classes needed no change (they already list the runtime class, or test the legacy class while it exists).
* **Usage and configuration.** `--spec-type`'s help (CLI and server): n-gram runs on Muse-Glimmer without a drafter.
  `config/muse-glimmer-30b.json` pins the drafter (`dflash-kquant.gguf`, sha256 `27d9a805...`); the new
  `config/muse-glimmer-30b-iq2xxs.json` is the Mac's UD-IQ2_XXS preset; `config/regression/source-pins.json` and
  `catalog.json` carry both (the inventory tests pass, 610 regression tests).
* **Docs (9387abb9).** The model card (EN, zh) gains the runtime section with S7's table; usage, the batching and
  speculation pages, the matrices, the supported-models and capability pages and the website (with its search indexes) drop
  the legacy-only claims (n-gram only with a drafter, page reuse, the per-sequence swap, the `TS_MUSE_GLIMMER_*` switches,
  which now do nothing and are retired in S9).
* **The pre-switch baseline** for any later A/B is `~/work/TensorSharp-glimmer-pre2`, detached at c012af49 (the last commit
  whose descriptor builds legacy); rebuild it before use.

| Gate (§4.8, the Mac) | Result |
|---|---|
| Synthetic suite (`Category!=Bench`), cpu and metal pins | **passed**: 15,007 / 15,012 tests, 13,795 / 13,743 passed, one failure each (`EverySharedRuntimeFamilyHasAConformanceFixture`), closed by the dedicated suite; the conformance classes then pass on both pins (78 passed, 5 skipped) |
| CLI, ggml_metal and MLX (IQ2_XXS) | **passed**: text (TTFT 403 ms, 22.0 tok/s; MLX 373 ms, 21.5); the four image questions 4/4 (7319, Red, OCEAN, 3); `--multi-turn-jsonl` reused 86/273 and 270/454; DFlash and n-gram engaged (n-gram 99% and 61% acceptance on the quote-back prompts) |
| The documented commands, as written (the card's `dotnet run --project ...` quick start and the website's `dotnet TensorSharp.Cli/bin/...` lines; `ggml_cuda` read as `ggml_metal`) | **passed**: all six ran; the image ones answered 7319 |
| Server, OpenAI API (image streamed and not, `json_schema` with thinking off and on, `json_object` with thinking on, tools streamed and not) | **passed**. `json_object` with thinking off returns 422 (the reply repeated `"population": 2` until the repetition guard stopped it) on the legacy model too, the same request on the pre-switch build: pre-existing |
| Server, multi-turn reuse (OpenAI; Web UI with thinking off and on) | **passed**: every follow-up reuses what the previous turn held (4,284 / 4,515 / 4,922 tokens; thinking on 4,486 / 4,917) |
| Web UI loop check (`webui_chat_e2e.py`, thinking off and on, 1,200 tokens) | **passed** |
| DFlash and n-gram servers | **passed**: reuse as plain; speculation armed |
| Parallel chats (c4 against sequential, 4 x 3 turns) | **passed**: 12/12 turns identical to sequential, 27.2 s against 53.8 s |
| TensorAgent Mac app, `muse-glimmer-30b-q4kxl`, chat-e2e's seven scenarios | **passed** 7/7 with the user's settings (KV q4_0, speculation on) and with the entry's own (KV q8_0, speculation off). Against the legacy model built from c012af49, same session and settings: the story turn 17.0 tok/s (legacy 17.1); a new chat reuses 99.7% (legacy 62.3%, its 4,352-token ring cap), first token 0.83 s (8.20 s); the image turn's first token 6.8 s (35.1 s); app footprint 13.4 GB at its highest (legacy 12.7), system wired 22.5 GB (26.6). The catalog keeps its 32 GB tier: by its rule (weights + footprint + 5 GB) both families now come to 33.6-34.3 GB, a question for the catalog, not a regression of the switch |
| ggml_cuda and direct CUDA E2E | **not run** (the VM) |

**After S8: the TensorAgent footprint (question a).** Measured again with the same two app builds, `footprint -p` and `vmmap
-summary` after the load and after chat-e2e's fact, newchat and image turns. The runtime's extra 0.7 GB is ggml-metal buffers
(vmmap's "shared memory": 9.8 GB virtual, 8.4 GB dirty, 413 regions after the turns against legacy's 8.9 GB, 7.8 GB, 322);
Malloc Large (about 2.0 GB, dirty plus compressed) and Untagged (2.1-2.2 GB) are equal within 0.1 GB. They are the graph tower's
programs' compute, kept for replay (0.26-0.38 GB for the 1260 x 840 test image by the tower's own sizing), and the runtime's
step-graph arenas; not the caches (2 KV heads: 54 MB of rings a slot) or the checkpoint files (on disk). The +0.9 GB the app
logs right after loading is transient (vmmap 5 s later: 11.3 against 11.4 GiB). Legacy wired 4.9 GB more during its image
turn than after it (27.0 against 22.1 GB; the runtime 23.7 against 21.4), outside its footprint (peak 12.7 GB): in all the app
causes less memory on the runtime (+0.7 GB footprint, -3.3 to -4.1 GB wired), so it stays. The catalog's 32 GB tier is
unchanged; its rule gives 34.3 GB on the runtime and 33.6 on legacy, an excess that predates the switch (for the user).

### 5.8 S9, the deletion: status

**The Mac part is done** (2026-11-20, M5 Pro, ggml d9fe7bff unchanged before and after every build; `wip/glimmer-s9` on
757e0517, each deletion its own commit). The VM part (the cuda and vulkan pins' suites, a real-weight smoke on ggml_cuda and
direct CUDA, the S8 sessions) waits for the CUDA box: `glimmer/cuda-vm/s9w/s9-cuda.sh` (`build`, `suite P`, `family P`,
`real B`, `cli B Q`, `server B Q`), on the S4-fix agent's corrected S8 script.

* **Frozen first (d492bfe9).** The legacy tower's rows on the synthetic mmprojs (six grids, F32 and Q8_0 products, Muse
  Spark's width) and the legacy model's DFlash drafter features, frozen on the managed backend while legacy existed
  (`Fixtures/MuseGlimmer/LegacySynthetic/managed.json`), joined the frozen logits (S0), real-weight continuations and
  perplexity (S0) and image figures (S4).
* **Tests on frozen figures (24cb451d).** The tower tests against the frozen managed rows, within the parity bound on the
  managed executor and within legacy's own spread between executors (6e-3) elsewhere (measured: managed 6.1e-7, ggml_cpu
  1.3e-4, ggml_metal 1.6e-3, MLX 4.2e-3); Muse Spark's tower and the drafter features against the same file. The real
  mmproj's tower rows against the F32 tower, by the rule ggml_cuda and ggml_vulkan already had, on every lane but direct CUDA:
  1 - cosine summed over the nine pictures at most 1.25x legacy's frozen sum, each picture's ratio logged. The figures for
  ggml_metal, MLX and ggml_cpu were frozen from d492bfe9 (the last tree with the legacy tower; the M5 Pro, ggml d9fe7bff,
  2026-11-20; `Fixtures/MuseGlimmer/LlamaCpp/legacy_vision.<lane>.json`), where the graph tower's sum was 0.71x, 0.81x and 0.92x
  legacy's. Direct CUDA keeps cosine 0.99999 per picture. (An absolute bound per lane, c57920ce, stood for one commit.) Deleted:
  the tests of legacy against itself, the writers, the legacy-only KV, MLX-decode and truncation tests.
* **Deleted.** `MuseGlimmerModel` and its partials, the oracle seam, the probe's `--family` (refused: exit 2, one stderr line);
  `MuseGlimmerVisionEncoder` and the native vision block; `ggml_ops_muse_glimmer.cpp`, its exports and `ggml_ops_core.cpp`'s
  hooks; the CUDA ring-mask kernel only that graph called (`ggml_ops_mask.cu`; no committed PTX is involved, so nothing is
  regenerated); the MLX helpers only legacy used; `KvBlockTransfer`'s ring addressing, which only legacy passed (no other
  caller: Gemma 4, GPT-OSS, Muse Spark and the shared runtime address every layer linearly).
* **Names retired** (`RemovedCliFlags.cs`, one entry a name): the D8 list, each failing startup with its replacement.
* **Docs.** The model card, the env-var feature matrix, PROJECT_STATUS, the development guides and the capability reference
  (EN and zh); the website pages and both search indexes name no legacy code or switch.

| Gate (the Mac) | Result |
|---|---|
| Synthetic suite (`Category!=Bench`), cpu and metal pins | **passed**: cpu 13,753 passed, 1,199 skipped, 0 failed (14,952); metal 13,702, 1,255, 0 (14,957). 55 fewer than S8's: the deleted legacy tests |
| Real weights, ggml_metal (IQ2_XXS) | **passed** 11/11: llama.cpp 640/640 and 128/128 teacher-forced, first position within 0.050; legacy's frozen continuations 6/6 identical; perplexity 4.168974 against legacy's frozen 4.168933 (+0.0010%), top-1 62.96% both; three growth paths 360 steps identical to presized; a checkpoint across the rings' wrap continues as the live cache; DFlash identical to plain on all six prompts, alone and together |
| Real weights, MLX | **passed** 11/11: 639/640 (one near-tie) and 128/128; 5/6 continuations legacy's, the sixth leaving at a near-tie; perplexity +0.0001% |
| Image prompts | **passed**: ggml_metal 215/216 (first position within 0.18), MLX 215/216 (0.35) |
| Tower rows against the F32 tower (real mmproj, nine pictures) | **passed** by the frozen-sum rule: the graph tower's 1 - cosine summed over the pictures 0.71x legacy's frozen sum on ggml_metal (7.01e-4 against 9.90e-4), 0.81x on MLX (6.06e-4 against 7.44e-4), 0.92x on ggml_cpu (2.14e-2 against 2.34e-2); bound 1.25x. The S9 graph tower gives the same sums as at the freeze |
| Checkpoint files and checkpoints (the probe, ggml_metal, the 4.1K prompt teacher-forced by legacy's frozen tokens, 500 steps) | **passed**: a checkpoint file written at 3000 and at 4400 positions (160.8 MB) and restored into a new request's slot, a checkpoint cloned at both, and a decode-time rewind: every step's logits at cosine 1.000000 to the presized run, argmax legacy's token 500/500 |
| CLI, ggml_metal and MLX | **passed**: TTFT 405 / 372 ms, 21.7 / 21.4 tok/s; images 4/4; multi-turn reuse 86/273 and 270/454; DFlash and n-gram (99%) engaged. `TS_MUSE_GLIMMER_SWA_RING=0` stops the CLI before loading ("... was removed: ... Unset it.", exit 1); the probe refuses `--family` |
| Server (OpenAI, Web UI, DFlash and n-gram servers, c4) | **passed**, S8's figures: OpenAI 7/9 (the same two pre-existing ones), reuse 4,515 / 4,922 (thinking 4,486 / 4,917), the Web UI loop check, c4 12/12 identical to sequential, 27.2 s against 53.8 s |
| No legacy symbol | **passed**: the class, export and helper names occur in no code, project, build or script file; only in the frozen fixtures' provenance, the history sections of the model card and this document |
| ggml_cuda, ggml_vulkan, direct CUDA | **not run** (the VM) |


---

## 6. Decisions for the user

| # | Decision | Recommendation |
|---|---|---|
| D1 | Prompt step width with rings: the runtime's 512 rows (2560-cell rings, llama.cpp's ubatch) or legacy's 2048 rows (4352 cells) | **Measure in S7, then choose per backend.** Keep 512 unless Glimmer's prefill is more than 2% slower than legacy on a backend. In that case the family declares 2048-row steps through a family hook (no new flag), in a DEFAULT CHANGE commit, since the step width changes the attention windows' arithmetic. On Metal, llama.cpp's 512-row ubatches already match legacy (pp4096 381.4 vs 382.6), and 512 halves the SWA KV (2560 vs 4352 cells) |
| D2 | When legacy goes: in the switching change, or a stage after it | **S9, after the S8 switch.** Between them legacy is reachable only through the oracle test seam (no user switch), because it is the reference while S4-S7 change shared code (the H3-Image precedent) |
| D3 | The vision tower: port it to a graph program (shared with Muse Spark), or wrap the per-op tower | **Port it (done in S4).** That deletes `ggml_ops_muse_glimmer_vision.cpp`, gives the tower every executor (MLX, direct CUDA, managed), and moves Muse Spark in the same stage under its tests |
| D4 | `--tp` on ggml_vulkan: the runtime's ranked graphs are CUDA-only | **Done in S6 as a DEFAULT CHANGE** ([§5.5](#55-s6-multi-gpu-status)): ranked graphs on Vulkan, measured on 2x RTX PRO 4500. The recommendation was: **Enable ranked graphs on Vulkan (G6)** if a 2-GPU box can run Vulkan (A40s or A5000s can). Otherwise refuse `--tp` on ggml_vulkan with a message naming ggml_cuda. The refusal removes a legacy capability that was never measured, so the user decides |
| D5 | DFlash under `--tp`: legacy declines it | **Keep the decline** in this series (parity; done in S6, [§5.5](#55-s6-multi-gpu-status)). Drafting on rank 0 over per-rank verify graphs is a follow-up |
| D6 | The per-op hooks (G3), or `RequiresGraphs => true` | **The hooks** (about half a day). Every SharedRuntime mechanism then holds, multi-node stages included, and Gemma 4 reuses them |
| D7 | MLX pipelined greedy decode (`.MlxDecode.cs`, CLI `--benchmark` only) | **Delete it with legacy** if the runtime's MLX real-generation decode is at least legacy's. CLI `--benchmark` on MLX then measures the path serving uses. Port device argmax into `MlxGraphExecutor` (G8) only if the gate fails |
| D8 | Retired names: `TS_MUSE_GLIMMER_PREFILL_CHUNK`, `_SWA_RING`, `_SWA_ROWS`, `_LAYER_TRACE` (+ `_POS`, `_N`, `_DIR`), `_VENC_TRACE`, `_MLX_STAGE_PROFILE`, `TS_MLX_KV_MATERIALIZE_INTERVAL` | **Retire all of them through the removed-names registry** (a hard error naming the replacement). The two traces are observers, a kept class, but nothing would implement them. Point them at the probe and `TS_GGML_NODE_PROFILE` rather than port a Glimmer-only trace |
| D9 | Commit llama.cpp goldens as test fixtures | **Yes.** They are small JSON files (`MuseGlimmerParityTests` skips without them today). Done in S0 with llama.cpp d3c3e242c (`~/work/llama.cpp-lfm3`, a clean Release Metal build): the 1a24df163 checkout's build directory mixes four builds and its llama-server aborts at startup; d3c3e242c's text graph is 1a24df163's after a refactor |
| D10 | Validation weights | **UD-Q4_K_XL** (the config's file) on the VM, plus **UD-IQ2_XXS** on the VM and the Mac for a cross-device anchor. Q8_0 only if a stage needs a near-exact reference |
| D11 | Capabilities legacy lacks: continuous batching, layer split, hybrid, multi-node, n-gram, reuse across image spans, direct-CUDA graphs | **Turn them on at the switch.** Each has a gate in §4, and a SharedRuntime contract cannot declare less |
| D12 | Hardware (§7) | **One Ada or Blackwell GPU (32 GB+) for S0-S8, and a 2-GPU box for S6** |

---

## 7. Hardware and disk

| Set | Files (unsloth/Muse-Glimmer-30B-GGUF @ faa5b025) | Disk |
|---|---|---:|
| Mac set (downloaded, sha256 verified) | UD-IQ2_XXS 10.75 GB + mmproj Q8_0 2.05 GB + `dflash-kquant.gguf` 1.63 GB | 14.4 GB |
| VM set | UD-Q4_K_XL 15.88 GB + UD-IQ2_XXS 10.75 + mmproj 2.05 + drafter 1.63 | 30.3 GB |
| optional near-exact reference | Q8_0 29.61 GB | +29.6 GB |

* **This Mac** (M5 Pro, 48 GB, ggml_metal, MLX, ggml_cpu): S0, S1, the synthetic gates of every
  stage, and real IQ2_XXS runs on Metal and MLX (§4.1). It cannot hold UD-Q4_K_XL: 14 GiB of disk
  are free after the downloads, and the brief keeps 12 GiB free. It cannot run CUDA, Vulkan or TP.
  Real-weight runs go under the Mac lock (excl) and `artifacts/ab/memguard.sh` at 43 GB wired. The
  legacy IQ2_XXS runs peaked at 14.2-24.5 GB wired (§4.1), so the runtime arm has room beside them
  but never beside a second model process.
* **One CUDA GPU, Ada (sm_89) or Blackwell (sm_120), 32 GB or more** (an RTX PRO 4500 32 GB is the
  right size), for S0 (the CUDA baseline), S2, S3, S5, S7 and S8. Ada or newer is needed because
  the FA-VEC hazard (G10) appeared there. UD-Q4_K_XL is 15.9 GB; legacy's Q8_0 peaked at 29.7-37.8
  GB at 0.5K-124K context on one PRO 6000 (docs §5), so expect about 18-25 GB for Q4_K_XL to 64K.
  The same box runs ggml_vulkan and direct CUDA. Disk: about 100 GB (the VM set, two builds,
  llama.cpp for the goldens). One agent-week of use, intermittently.
* **Two GPUs in one box** for S6 (`--tp 2`, `--layer-split 2`, hybrid with one stage per GPU,
  Vulkan TP for D4): 2x A40 or 2x RTX PRO 4500 with working peer-to-peer. One 2x A40 VM had broken
  PXB peer reads (memory); the host-staged fallback exists but measures differently. Both quants
  fit 2 x 24 GB. About 2 days.
* An A40 (sm_86) is optional Ampere coverage (the VEC hazard does not apply there per legacy's
  analysis).
* Muse Spark has no weights here (816B). S4's Muse Spark coverage is synthetic, plus one real run
  on whatever VM next holds Muse Spark, if the user wants that gate.

---

## 8. Risks and open questions

* **Bit-exactness is not expected on real weights.** The runtime's lowering differs from the legacy
  graph in its attention windows (512 vs 2048 prompt steps, 64- vs 256-row decode padding on Metal),
  in mask shapes and possibly in the gate/up split. IQ2_XXS leaves near-ties of 0.009 nat at long
  context (docs §4), so greedy text comparisons are noisy. The teacher-forced perplexity and
  near-tie checks are the primary quality gates; synthetic digests are the exactness gates.
* **G12** (closed in S6): the rings stay on under `--tp 2`, read whole with one KV head a rank, and give the one-GPU
  text; no linear caches were needed.
* **G10** (closed in S2): the runtime's builder applies legacy's guard (`view_kv_cache_window`), and at d9fe7bff the
  hazard is gone on Blackwell. Ada was not run, and a ggml bump could bring the fault back; the guard covers both.
* **MLX memory**: the 48 GB Mac panicked at 49.1 GB wired under MLX (memory
  `mac-wired-memory-watchdog-panics`). Every MLX run goes under memguard at 43 GB; IQ2_XXS is below
  the 12 GB ceiling that note set for MLX models.
* **Muse Spark** depends on the tower that S4 ports; its real-weight vision is validated only where
  its weights live.
* **DFlash numbers move with the text.** Any arithmetic change re-rolls a single prompt's greedy
  continuation and the governor's verdicts (the #371 lesson), so compare several prompts and served
  turns, and judge speed with 512-token answers.
* **The legacy CUDA side was never measured after the September fix**: the probe and E2E on CUDA,
  and the TP grow/snapshot run. S0 runs them on legacy first, so a CUDA regression is not blamed on
  the runtime.
* **The `.parity` goldens are absent**, so nothing currently checks main's Glimmer against
  llama.cpp. If the goldens show a legacy defect, fix it in legacy first (it is the oracle) or
  record it as a known difference.
* **Disk on this Mac** (14 GiB free, shared): a second worktree build or UD-Q4_K_XL does not fit.
  The legacy arm on the Mac is the same binary through the test seam until S8.
