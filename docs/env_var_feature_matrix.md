# Environment Variable x Feature Matrix

[English](env_var_feature_matrix.md) | [中文](env_var_feature_matrix_zh-cn.md)

This document is the curated runtime-flag reference used by
[`TensorSharp.TestMatrix`](../TensorSharp.TestMatrix/README.md). It focuses on
environment variables that materially change correctness, throughput, memory
use, or model routing for real inference workloads.

The code source of truth is
[`TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs`](../TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs).
The default sweep list is configured in
[`TensorSharp.TestMatrix/Defaults/matrix-config.json`](../TensorSharp.TestMatrix/Defaults/matrix-config.json).

## How TestMatrix Uses This

- Every applicable `(model, backend, feature)` cell first runs a **baseline**
  case with no forced sweep variable.
- For each selected env var, the runner creates one case per listed value and
  passes only that variable to the `TensorSharp.Cli` subprocess.
- Before each subprocess starts, inherited `TS_*`, `GDN_*`, `QWEN35_*`,
  `FUSED_*`, `KV_CACHE_DTYPE`, `MAX_CONTEXT`, `MAX_TOKENS`,
  `VIDEO_MAX_FRAMES`, and `VIDEO_SAMPLE_FPS` variables are scrubbed so the
  matrix value is authoritative.
- `--env-vars none` disables sweep cases. If a config file has an empty
  `default_env_vars` list and the CLI does not override it, the runner uses all
  registered `EnvVarMatrix.All` entries.

The "Runtime baseline" column below describes the behavior when the variable is
unset. The "Swept by default" column describes the current default config, not
the full set of registered variables.

DiffusionGemma is currently outside the registered TestMatrix feature catalog:
there is no diffusion prompt type, no diffusion-specific env sweep, and inherited
`DIFFUSION_*` variables are not scrubbed by the runner. Use explicit model
configs plus a dedicated feature/env registration before treating diffusion
results as part of the standard matrix.

## Continuous Batching / Batched Forward

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_GPTOSS_BATCHED` | GPT OSS | Batched paged forward vs per-sequence fallback | ON | `0`, `1` | yes |
| `TS_QWEN35_BATCHED` | Qwen 3.5 / 3.6 family, `qwen3next` | Batched paged forward vs per-sequence fallback | ON | `0`, `1` | yes |
| `TS_QWEN35_BATCHED_GDN_NATIVE` | Qwen 3.5 / 3.6 family, `qwen3next` | Native batched GatedDeltaNet kernel | OFF | `0`, `1` | no |
| `TS_NEMOTRON_BATCHED` | Nemotron-H | Batched paged forward vs per-sequence fallback | ON | `0`, `1` | yes |
| `TS_GEMMA4_BATCHED` | Gemma 4 | Batched paged forward vs per-sequence fallback | ON | `0`, `1` | yes |
| `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE` | Nemotron-H | Native batched Mamba2 step | OFF | `0`, `1` | no |
| `TS_BATCHED_N1_FAST_PATH` | all | Fused N=1 fast-path decode for solo sequences; `0` forces those steps onto the fully-batched path | ON | `0`, `1` | yes |
| `TS_PER_SEQ_FUSED` | fused-capable models (Gemma 4, Qwen 3.5/3.6, DeepSeek V4, GLM 5.x) | Per-request fused Forward for concurrent (N>=2) sequences; `0` forces the op-by-op batched paged path | ON | `0`, `1` | no |
| `TS_BATCHED_FUSED_DECODE` | fused-capable models | True token-batched fused decode inside the per-seq fused path (one graph for all N). On GLM 5.x this is 1.81x aggregate decode at 4 concurrent requests; on DeepSeek V4.1 Flash at Q4_K_M it is 2.0x (24.3 → 48.9 tok/s), bounded by routing because each token picks its own 6 of 384 experts. Batching changes GEMM shapes and a 2-bit MoE can turn that into different expert picks; set `0` for a serial-path A/B. | ON | `0`, `1` | no |
| `TS_RETAINED_FUSED_CACHE` | models with retainable request-owned fused holders (Gemma 4; Qwen 3.5/3.6) | Retain a finished holder for exact-prefix continuation. Qwen's holder includes attention K/V and matching GatedDeltaNet recurrent state | ON | `0`, `1` | no |
| `TS_RETAINED_FUSED_CACHE_MAX` | models with retainable request-owned fused holders | LRU budget of retained holders (VRAM cap; includes recurrent state where applicable) | `4` | n/a | no |
| `TS_PREFIX_CHECKPOINTS` | Gemma 4; Qwen 3.5/3.6 (GGML backends) | Checkpoint the model's complete state where the prompt every conversation shares ends (system prompt, tools, skills) and start each NEW chat from a clone of it, so a new chat re-prefills only its own message | ON | `0`, `1` | no |
| `TS_PREFIX_CHECKPOINTS_MAX` | same | How many distinct shared prefixes stay checkpointed (each holds one copy of the prefix's K/V and, for Qwen, recurrent state) | `2` | n/a | no |
| `TS_KV_INITIAL_TOKENS` | families that size their cache through `ModelBase.ResolveInitialCacheAllocationLength` (Qwen 3.5/3.6, Gemma 4, GPT-OSS and the other ModelBase families; not DeepSeek V4 / GLM 5.x, which size their own) | Tokens of K/V a cache is given when created (the primary cache at load, every per-request holder) before any request declares a budget; `0` keeps the engine policy (whole window when `MAX_CONTEXT` is explicit, else a backend default). The cache still grows on demand. Memory-constrained devices set this small because every kept holder is paid at this size, host and device mirror both | `0` | n/a | no |
| `TS_KV_GENERATION_RESERVE_MAX` | all | Cap on the generation share of the K/V a request reserves up front (prompt + max_new_tokens); a reply limit at or above the window otherwise reserves the whole window per request. Past the cap the cache grows on demand. `0` = no cap | `0` | n/a | no |
| `TS_KV_HOLDER_POOL_MAX` | models with per-request fused holders (Qwen 3.5/3.6, GPT-OSS) | How many released holders may be parked for reuse instead of freed; each parked holder costs its whole K/V allocation | `64` | n/a | no |
| `TS_SCHED_DISABLE_BATCHED` | all | Global per-sequence KV-swap fallback | OFF | `0`, `1` | yes |

All executor-level switches in this section are read through
`ExecutionOptions.FromEnvironment()` and consumed by `ExecutionPlanner`
(see `docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`, "Execution Planning");
per-model `TS_*_BATCHED` opt-outs surface as the model's declared
`BatchedForwardAvailable` capability.

## KV Cache / Context

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `KV_CACHE_DTYPE` | all | KV cache element type | auto (model-aligned: `f16` when the model's weights are below F32, else `f32`) | `f32`, `f16`, `q8_0` (runtime also accepts `q4_0`, not swept) | yes |
| `TS_KV_PAGED_QUANT_BITS` | all paged-KV models (not `glm-dsa`: MLA keeps one compressed 576-wide row per token and the DSA indexer scores that same contiguous history, so there is no paged block layout to quantize) | TurboQuant paged-KV block codec (2-bit uses the affine min+scale layout) | off (`0`) | `0`, `4`, `8` (the runtime also accepts `2`; not swept) | yes |
| `TS_N_CPU_MOE` | MoE models | Routed experts of the first N layers stay in system RAM: multiplied on the host at decode, streamed to the accelerator for one graph at prefill | off (`0`) | `0`, `16`, `all` | yes (GGML backends, MoE families) |
| `TS_CPU_MOE` | MoE models | Offload every layer's routed experts (equivalent to `TS_N_CPU_MOE=all`) | off | `0`, `1` | no |
| `TS_CPU_MOE_THREADS` | MoE models | Worker threads for the host-side expert matmul. Default is half the usable CPU parallelism (hardware threads clamped by the affinity mask and the cgroup CPU quota), capped at 64: the decode-side matmul is one token wide, so past a few dozen workers each extra thread only adds a barrier participant (measured 7x slower at 192 threads than at 32 on a 2-socket Xeon) | min(usable/2, 64) | - | no |
| `TS_HOST_MOE_DEVICE_MIN_BATCH` | MoE models with offload | Batch size at or above which an offloaded layer is computed on the accelerator with its experts streamed in, rather than on the host. `0` restores host-only offload | `128` | `0`, `32`, `128` | no |
| `TS_HOST_MOE_PIN` | MoE models with offload | Page-lock (`cudaHostRegister`) the offloaded expert ranges so the streamed prefill DMAs instead of staging through the driver (9.3 -> 55.6 GB/s on PCIe 5.0) | ON | `0`, `1` | no |
| `TS_HOST_MOE_PIN_MAX_MB` | MoE models with offload | Budget for the pinned expert ranges | 60% of the cgroup/host memory limit | - | no |
| `TS_HOST_MOE_EXPERT_FILTER` | MoE models with offload | Stream only the experts the batch actually routes to, grouped into consecutive runs | ON | `0`, `1` | no |
| `MAX_CONTEXT` | long text / uploaded text | Hard context cap. Set, it is a requirement: honoured if the caches fit and refused with the numbers if not. Unset, the GGUF's advertised length is a ceiling the loader may cap to what the devices hold — GLM-5.2 advertises 1M tokens, which is ~93 GiB of KV | model default (a ceiling, not a promise) | `4096`, `8192`, `16384` | yes |

## Prefill / Decode Tuning

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_PREFILL_CHUNK` | swept on GPT OSS, Qwen 3.5 / 3.6 family long-context features; honored at runtime by Gemma 4, Nemotron-H, and Mistral 3 as well | Chunked prefill block size | architecture default | `256`, `512`, `1024` | yes |
| `GDN_DISABLE_CHUNKED_PREFILL` | `qwen3next` | Disable GDN chunked prefill | OFF | `0`, `1` | no |
| `TS_GGML_ASYNC_COMPUTE` | GGML backends | Async compute submission | ON on `ggml_metal` (`0` disables), OFF on other GGML backends | `0`, `1` | yes |
| `TS_QWEN35_FD_PERSIST` | Qwen 3.5 / 3.6 family on GGML GPU backends | Retain and replay the whole-model single-token decode graph | ON | `0`, `1` | no |
| `TS_GPTOSS_MODEL_DECODE` | GPT OSS on GGML backends | Run the whole transformer (all layers + MoE + final norm + LM head) as ONE graph per decode token; `0` falls back to the per-layer fused kernels | ON | `0`, `1` | no |
| `TS_GPTOSS_FD_PERSIST` | GPT OSS on `ggml_cuda` / `ggml_vulkan` | Retain and replay that whole-model decode graph (padded KV window + `set_rows`), which is what lets ggml-cuda capture it | ON | `0`, `1` | no |
| `TS_NEMOTRON_FLASH_DECODE` | Nemotron-H on GGML backends | Device-side single-token attention against the resident KV cache; `0` restores the host C# decode attention | ON | `0`, `1` | no |
| `TS_QWEN35_METAL_GDN_INPLACE_STATE` | Qwen 3.5 / 3.6 family on single-device `ggml_metal` | Alias K=1 GatedDeltaNet output and recurrent state, eliminating the per-layer state copy | ON | `0`, `1` | no |
| `TS_QWEN35_METAL_TOKEN_INPUT` | Qwen 3.5 / 3.6 family on `ggml_metal` | Read token embeddings directly from the quantized table inside the decode graph | ON | `0`, `1` | no |
| `TS_QWEN35_METAL_KV_CPY` | Qwen 3.5 / 3.6 family on `ggml_metal` | Append K/V through movable `CPY` views instead of indexed scatter | ON | `0`, `1` | no |
| `TS_QWEN35_METAL_ASYNC_SUBMIT` | Qwen 3.5 / 3.6 family on `ggml_metal` | Submit decode and logits readback before a single synchronization | ON | `0`, `1` | no |

## Multimodal

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `VIDEO_SAMPLE_FPS` | video features | Time-based frame sampling rate | `1` | `1`, `2` | yes |
| `VIDEO_MAX_FRAMES` | video features | Upper bound on sampled video frames | no cap | `8`, `16` | yes |
| `TS_NEMOTRON_IMAGE_MAX_TILES` | Nemotron-H image features | Maximum image tiles | architecture default | `4`, `8`, `12` | yes |

## MLX-Specific

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_MLX_BATCHED_MOE_DECODE` | Qwen 3.5 / 3.6 MoE on MLX | One batched dispatch per gate/up/down instead of per-expert dispatches | ON | `0`, `1` | yes |
| `TS_MLX_DEVICE_ROUTER` | Qwen 3.5 / 3.6 MoE on MLX | Device-side top-K + softmax router when prerequisites are met | ON with automatic fallback | `0`, `1` | yes |
| `TS_MLX_PIPELINED_DECODE` | MLX decode features | Pipelined greedy decode with device-side argmax where supported | ON when eligible | `0`, `1` | yes |
| `TS_MLX_DEVICE_KV_COPY` | MLX | On-device KV scatter | ON | `0`, `1` | no |
| `TS_MLX_QWEN35_GDN_PACKED_KERNELS` | Qwen 3.5 / 3.6 family on MLX | Packed GDN kernels | OFF | `0`, `1` | yes |

## Out-of-Matrix Pure-C# CPU Backend Knobs

These tune the persistent worker pool and the quantized-weight handling
behind `--backend cpu`. They are real runtime knobs but are not registered
in `EnvVarMatrix.All` and are not swept by the default TestMatrix config.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_CPU_THREADS` | `cpu` backend (100% pure C#) | Width of the persistent worker pool that runs the managed matmuls. Default is HALF the usable CPUs, deliberately not all of them: the rest of the CPU path still uses the ThreadPool, and pool workers spin between jobs, so taking every core starves that other work. Measured on a 122-CPU quota, two interleaved runs per cell (prefill / decode tok/s): pool off 21.7,21.0 / 2.0,2.4; 32 threads 24.9,24.1 / 4.9,5.0; 48 threads 25.6,28.5 / 5.4,6.0; 61 threads 24.2,24.9 / 6.3,5.9; 122 threads 13.5 / 4.8. At 122 only prefill regresses - decode still beats the pool-off baseline | every core at <=8 CPUs, else max(8, usable/2) | not registered | no |
| `TS_CPU_POOL` | `cpu` backend | `0` reverts to the pre-pool behaviour - ThreadPool `Parallel.For` with thread-count-scaled chunks - so the two can be A/B-ed in one binary | ON | not registered | no |
| `TS_CPU_SPIN` | `cpu` backend | Spin iterations a pool worker takes before parking. Parking is the expensive part at this width (waking N workers costs more than the ~60 us of work being handed out), so the default spins long enough that the steady state never parks: at 256 the same model measured 0.1 tok/s against 7.0 at 4096 | `4096` | not registered | no |
| `TS_CPU_TASK_BYTES` / `TS_CPU_TASKS_PER_WORKER` | `cpu` backend | Chunking of a managed matmul: weight bytes per work item, and the cap on work items per worker. Sized from the WORK rather than the thread count - the old thread-count-scaled rule built 1024 tiny tasks per matmul at 122 threads and stopped scaling past 8 | `131072` / `4` | not registered | no |

## Out-of-Matrix DiffusionGemma Knobs

These variables are real runtime knobs, but they are not registered in
`EnvVarMatrix.All` today and are not swept by the default TestMatrix config.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `DIFFUSION_STEPS` | DiffusionGemma Web UI | Denoising steps per block in the server path | `48` | not registered | no |
| `DIFFUSION_MAX_BATCH` | DiffusionGemma Web UI | Max active requests in `DiffusionBatchScheduler` | `2` | not registered | no |
| `DIFFUSION_BATCHED_FORWARD` | DiffusionGemma | True batched canvas decode vs time-sliced fused single-canvas decode | OFF | not registered | no |
| `DIFFUSION_NO_PKV` | DiffusionGemma | Disable prompt-KV caching on device-glue backends | OFF | not registered | no |
| `DIFFUSION_NO_SC` / `DIFFUSION_SC_TOPK` | DiffusionGemma | Self-conditioning enablement and experimental top-K cutoff | ON / `32` | not registered | no |
| `DIFFUSION_NO_FUSED_DECODE` / `DIFFUSION_NO_FUSED_LMHEAD_TAIL` | DiffusionGemma on GGML backends | Disable fused whole-model diffusion decode or fused lm-head tail | OFF | not registered | no |
| `DIFFUSION_LMHEAD_BATCH_CAP_MB` | DiffusionGemma | Transient lm-head logits memory cap before per-sequence fallback | `300` | not registered | no |
| `DIFFUSION_VRAM_HEADROOM_MB` | DiffusionGemma on ggml_cuda | VRAM kept free of preloaded weights (compute buffers, device copies) | `2048` | not registered | no |
| `DIFFUSION_DEVICE_COPY_BUDGET_MB` | DiffusionGemma on ggml_cuda | Device-copy cache cap when the model does not fit VRAM (prompt K/V, masks, activations) | `768` | not registered | no |
| `DIFFUSION_SEGMENTED_DECODE` | DiffusionGemma on ggml_cuda | Force per-layer fused decode on (`1`) / off (`0`); auto-selected when the model does not fit VRAM | auto | not registered | no |
| `DIFFUSION_PIN_STREAMED` | DiffusionGemma on ggml_cuda | Re-home streamed (non-resident) weights into page-locked copies for DMA-speed uploads (costs RAM) | OFF | not registered | no |

## Out-of-Matrix Speculative-Decoding Knobs

These gate the optional speculative decode path in `TensorSharp.Cli` and
`TensorSharp.Server` (Qwen 3.6 / GLM 5.2 embedded NextN block; Gemma 4's separate
`gemma4-assistant` draft GGUF; DeepSeek V4 DSpark and Muse-Glimmer DFlash block
drafters; the weight-free n-gram speculator). Speculation engages only for solo
(non-concurrent) sequences and only where it is profitable (ggml backends and the
pure-C# `cuda` backend). They are not registered in `EnvVarMatrix.All` and are not
swept by the default TestMatrix config — the matrix feature catalog has no
speculative-decode feature today, so use explicit runs to exercise these.

Each knob has a current `TS_SPEC_*` spelling and a legacy `TS_MTP_*` one. Hosts
publish **both** when a flag is applied, and readers accept either: the glm-dsa
**native** loader reads `TS_MTP_SPEC` and `TS_MTP_DRAFT` from C++ while the model
is loading (it decides whether to page a whole extra 256-expert decoder layer into
VRAM, and sizes its graph cache), so those names are a cross-language contract
that cannot simply be renamed. All are also settable via the `--spec*` flags (or
their `--mtp-*` aliases) on both hosts.

| Env var | Legacy spelling | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|---|
| `TS_SPEC` | `TS_MTP_SPEC` | Qwen 3.5/3.6, GLM 5.2, Gemma 4, DeepSeek V4, Muse-Glimmer (CLI + server) | Enable speculative decode for solo sequences | OFF (`0`) | not registered | no |
| `TS_SPEC_TYPE` | — | all of the above | Speculation algorithm: `auto` \| `draft-head` \| `block` \| `ngram` | `auto` | not registered | no |
| `TS_SPEC_DRAFT` | `TS_MTP_DRAFT` | all of the above | Max tokens drafted per speculative step (1-64) | `8` | not registered | no |
| `TS_SPEC_PMIN` | `TS_MTP_PMIN` | all of the above | Draft-confidence gate; meaning is per algorithm | per algorithm (`0.15` / `0.35` / `0`) | not registered | no |
| `TS_SPEC_DRAFT_MODEL` | `TS_MTP_DRAFT_MODEL` | Gemma 4 (CLI + server) | Path to the separate `gemma4-assistant` draft GGUF | none | not registered | no |
| `TS_GLM_MTP` | — | GLM 5.2 | Force the NextN block on (`1`) or off (`0`), overriding `TS_SPEC`/`TS_MTP_SPEC` in both directions | unset | not registered | no |
| `TS_GMTP_NO_FUSED` | — | Gemma 4 on ggml backends | Disable fused multi-token-verify / draft-step kernels (per-op fallback) | OFF | not registered | no |
| `TS_GMTP_NO_FAST_ROLLBACK` | — | Gemma 4 | Restore kept-prefix rollback instead of dense fast rollback on partial accept | OFF | not registered | no |
| `TS_GMTP_BATCHED_TRUNK` | — | Gemma 4 | Run the verify trunk through the batched paged path instead of the linear trunk | OFF | not registered | no |

The design behind these — the three-layer split of model architecture,
speculation algorithm and speculator weights — is documented in
[Speculative Decoding in TensorSharp](speculative_decoding.md).

## Out-of-Matrix Muse-Glimmer & DFlash Knobs

Muse-Glimmer's fused whole-model kernel and its DFlash block drafter each have an
A/B switch, plus long-context sizing knobs. None are registered in
`EnvVarMatrix.All`. The full list, including the layer-trace knobs, is in the
[Muse-Glimmer card](models/muse-glimmer.md#7-environment-variables).

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_MUSE_GLIMMER_FUSED` | Muse-Glimmer on GGML CUDA / Vulkan | Fused whole-model graph vs the per-op path | ON | not registered | no |
| `TS_MUSE_GLIMMER_PERSIST` | Muse-Glimmer (fused) | Persistent, CUDA-graph-capturable graph vs rebuild per call | ON | not registered | no |
| `TS_MUSE_GLIMMER_INGRAPH_EMBED` | Muse-Glimmer (fused) | Do the embedding gather + weightless input norm inside the graph (a loss unless the LM head is tied) | auto (on only when tied) | not registered | no |
| `TS_MUSE_GLIMMER_PREFILL_CHUNK` | Muse-Glimmer | Tokens per prefill forward; `0` disables chunking | `2048` | not registered | no |
| `TS_MUSE_GLIMMER_SWA_RING` | Muse-Glimmer (fused) | Ring the 39 sliding-window layers at `pad(n_swa + chunk + 1, 256)` rows instead of sizing every layer for the full context | ON | not registered | no |
| `TS_MUSE_GLIMMER_SWA_ROWS` | Muse-Glimmer (fused) | Override the SWA ring size in rows (diagnostics) | auto | not registered | no |
| `TS_MUSE_GLIMMER_VENC_F32` | Muse-Glimmer vision tower | Dequantize the tower to F32 (~7.4 GB) instead of feeding the GGUF quantization to `AddmmQuant` | OFF | not registered | no |
| `TS_MUSE_GLIMMER_VENC_FUSED` | Muse-Glimmer vision tower on CUDA | Fused vision-block / flash-attention path | ON | not registered | no |
| `TS_MUSE_GLIMMER_DFLASH` | Muse-Glimmer | DFlash drafter GGUF path (same as the CLI's `--draft-model`) | none | not registered | no |
| `TS_QWEN35_DFLASH` | Qwen 3.5 / 3.8 | DFlash / DFlash2 drafter GGUF path (same as the CLI's `--draft-model`) | none | not registered | no |
| `TS_DFLASH_FUSED` | any DFlash drafter | Fused `TSGgml_DFlashInject` / `TSGgml_DFlashDraftBlock` graphs vs the per-op drafter | ON | not registered | no |
| `TS_DFLASH_PERSIST` | any DFlash drafter | Replay the persistent draft graphs instead of rebuilding every step | ON | not registered | no |
| `TS_DFLASH_PREFILL_CHUNK` | any DFlash drafter | Tokens per speculative prefill forward (drives the TRUNK, not only the drafter) | `1024`, capped by the drafter ring and the trunk's own window | not registered | no |
| `TS_DFLASH_SELECTOR` | DFlash2 drafter | `0` drafts by per-position argmax instead of the candidate lattice (attribution only - the weights were trained with it) | ON | not registered | no |
| `TS_DFLASH_CONV` | DFlash2 drafter | `0` drops the grouped dynamic convolution (attribution only, as above) | ON | not registered | no |
| `TS_DFLASH_SELECTOR_DEBUG` | DFlash2 drafter (per-op path) | `1` prints the first blocks' lattice attribution: unary spread, transition spread, and whether the walk left the unary argmax | OFF | not registered | no |
| `TS_Q35_VERIFY_SNAPSHOTS` | Qwen 3.5 / 3.8 speculative verify | `0` reverts to restoring a pre-verify recurrent-state copy and re-forwarding the accepted prefix instead of keeping one snapshot per row | ON | not registered | no |
| `TS_Q35_VERIFY_DEFER_STATE` | Qwen 3.5 / 3.8 speculative verify | `0` downloads the post-window recurrent state after every persisted call instead of leaving it on the device for a slot commit; separable from the snapshots because it also covers the single-row steps a speculative session interleaves with verifies | ON | not registered | no |
| `TS_Q35_VERIFY_STRIDED_VIEWS` | Qwen 3.5 / 3.8 speculative verify | `0` disables the contiguous strided KV views on CUDA and Metal, falling back to per-head `set_rows` writes | ON | not registered | no |
| `TS_QWEN35_SPEC_DEVICE_STATE` | Qwen 3.5 / 3.8 speculation, Metal and CUDA | `0` drains the gated-delta-net recurrent state to host mirrors and re-uploads it around every speculative step, instead of keeping it on the device. Worth 1.67x on the `ggml_cuda` n-gram scenario, so this is a kill switch rather than a tuning knob | ON | not registered | no |
| `TS_QWEN35_VERIFY_RESIDENT` | Qwen 3.5 / 3.8 speculative verify | `1` keeps the conv and delta state device-resident across a verify instead of moving ~60 MB per call. **Currently produces a wrong stream** — a resident call updates the state in place, so a rejected draft has nothing to roll back to. Opt-in for measurement only; see [speculative decoding](speculative_decoding.md) | OFF | not registered | no |
| `TS_Q35_MTP_DRAFT_PERSIST` | Qwen 3.5 / 3.8 MTP draft graph | `1` lets the single-layer MTP draft graph use the persist/replay cache. Default off: the graph used to deadlock on CUDA-graph capture replay, and the knob exists to re-test that on a current ggml. Worth ~1% | OFF | not registered | no |
| `TS_MTP_FOLD_CATCHUP` | Qwen 3.5 / 3.6 NextN/MTP speculation | `0` runs the draft-head catch-up and the first draft step as two calls instead of folding them into one pass over `n_accepted + 1` rows (llama.cpp's draft-mtp shape). Worth ~4-5% | ON | not registered | no |
| `TS_SPEC_ADAPTIVE` | Speculative decoding (all drafters) | `0` disables the cost governor, so drafting is never measured against a plain baseline and never parked. For A/B measurement: a governor round's baseline steps are plain decodes and they are not free | ON | not registered | no |
| `TS_GGML_LOG_DEBUG` | GGML backends | `1` passes ggml's DEBUG log channel through instead of dropping it. Carries the CUDA backend's "CUDA graph warmup complete"/"reset" lines, which are the only way to see whether a graph is actually being CUDA-graph-captured | OFF | not registered | no |

## Out-of-Matrix DeepSeek V4 / V4.1 Knobs

These configure the DeepSeek whole-model executors. DeepSeek V4 has three of
them (direct CUDA, native ggml, pure C#); DeepSeek V4.1 serves on one native
`ggml_cuda` graph, with `ggml_cpu` as a scalar correctness path and its own
pure-C# and direct-CUDA executors as portability paths. None are
registered in `EnvVarMatrix.All`, so the default TestMatrix sweep does not touch
them. The full context is in the [V4 card](models/deepseek4.md) and the
[V4.1 card](models/deepseek41.md).

| Variable | Applies to | Effect | Baseline | In matrix |
|---|---|---|---|---|
| `TS_DSV4_NGPU` | V4 and V4.1 | How many GPUs the layer split spreads whole layers over — the same thing `--tp N` sets for these architectures. `0` selects every visible device | `0` (all visible) | no |
| `TS_DSV4_UBATCH` | V4 and V4.1 | Prefill micro-batch width | `256` in the conservative profile; the measured eight-A40 V4.1 profile uses `1024` | no |
| `TS_DSV4_THREADS` | V4 and V4.1 | Native thread pool for GPU-only loads. CPU expert offload uses the detected available parallelism instead, and `--cpu-moe-threads N` / `TS_CPU_MOE_THREADS` sets that. On the pure-C# `--backend cpu` executor it sizes that executor's own worker pool and defaults to `ProcessorCount` rather than min(cores, 32) | min(cores, 32) | no |
| `TS_DSV4_PERF` | V4 and V4.1 | `1` prints per-stage timing | off | no |
| `TS_DSV4_VRAM_RESERVE_MB` / `TS_DSV4_GRAPH_CACHE` / `TS_DSV4_LOAD_THREADS` / `TS_DSV4_LOAD_CHUNK_MB` / `TS_DSV4_MOE_MMAP` | V4 and V4.1 | Placement headroom, graph-cache depth, weight-load parallelism, and whether host-resident experts are multiplied in place out of the GGUF mapping | see the cards | no |
| `TS_DSV4_DSPARK` | V4 only | DSpark drafter GGUF, equivalent to `--draft-model`. V4.1 rejects V4 drafters — it has no DSpark path | unset | no |
| `TS_DSV41_TP` | V4.1 | `0` disables it; `2`-`8` enables **experimental routed-MoE tensor parallelism** over exactly that many GPUs, which must equal the count `--tp` / `TS_DSV4_NGPU` selected. Gate/up split along the FFN intermediate, down along its input, partials reduced through host-staged F32 buffers. Measured slower than the layer split on the first full Q2_K run | `0` | no |
| `TS_DSV41_ENGRAM_DEVICE` | V4.1 | `1` requires GPU-resident Engram tables and fails if they do not fit; `0` forces host mappings, which is also what a bit-exact CPU-oracle comparison needs. Unset is automatic and conservative: GPU-resident unless that would force routed-expert CPU offload | auto (GPU-resident when it fits) | no |
| `TS_DSV41_ENGRAM_WARM` | V4.1, host mappings only | Reads the Engram table pages so a lookup is a RAM read instead of a storage round trip. Unset warms in the BACKGROUND after the model is serving (the default); `1` warms synchronously during load as before; `0` never warms. Worth 200-252 → 452-492 prefill tok/s and 23-26 → 31-33 decode tok/s on eight A40s at Q4_K_M, and irrelevant on the GPU-resident default. Costs the table bytes in host page cache (60 GiB at Q2_K, 103 GiB at Q4_K_M) | background warm | no |
| `TS_DSV4_GRAPH_CACHE_HEADROOM_MB` | V4 and V4.1 | Device memory the graph cache must leave free for the next graph, on top of the largest entry it holds. Least-recently-used entries are freed until it fits. `0` restores the pure entry-count cap, which four concurrent 10.8k-token prefills could run out of memory | `1024` | no |
| `TS_DSV41_ENGRAM_THREADS` | V4.1, host mappings only | Persistent lookup workers, `1`-`32`. One token selects 24 independent rows per table, so serial reads mean serialized page faults | min(16, hardware threads) | no |
| `TS_DSV41_ENGRAM_RANDOM` | V4.1, host mappings on Linux | Random-access advice for the mapped Engram ranges. `0` disables, `1` forces | auto | no |
| `TS_DSV41_ENGRAM_SIDECAR` | V4.1 | Explicit path to the prepared `deepseek41.engram.bin` sidecar instead of the one beside the shards | beside the GGUF | no |
| `TS_DSV4_LOAD_CONTIGUOUS` | V4 and V4.1 | `0` reverts the weight loader to handing its chunk jobs out from a shared cursor, which makes every reader stride `threads x chunk` through the file instead of reading one contiguous run. Kept only so the contiguous default can be A/B'd: on a MooseFS mount it measured 2.5x slower (363-382 s against 144-155 s to load the Q4_K_M release on eight A40s) | on | no |
| `TS_DSV4_LOAD_DROP_CACHE` | V4 and V4.1 | `1` releases each weight chunk's page cache once the chunk is on the device. Does not change load time; ends the load with ~39 GiB of page cache instead of ~330 GiB, which leaves room for the host-resident experts. Off by default because the call costs real time on FUSE | off | no |
| `TS_DSV41_REWIND_CHECKPOINT` | V4.1, native executor | `0` drops the per-slot rewind checkpoint (a shadow copy of the raw sliding-window and compressor-state rings, taken at every prompt boundary). Without it a partial KV reuse can only rewind as far as the live ring reaches — 385 positions on the released checkpoint — so a multi-turn thinking chat re-prefills instead. Costs ~21 MiB of VRAM per sequence slot | on | no |
| `TS_DSV41_SPARSE_FA` | V4.1 | `1` selects sparse flash attention. Opt-in, with documented floating-point differences from the dense path | off | no |
| `TS_DSV41_COMPACT_RAW_GATHER` | V4.1 | `1` selects compact gathering for the raw sliding window. Opt-in, with the same floating-point caveat | off | no |
| `TS_DSV41_ALLOW_NON_CUDA_GPU` | V4.1 | `1` permits `ggml_vulkan` / `ggml_metal`, where the ordinary graph runs on the GPU and only the architecture-specific ops fall back to the CPU backend, at a host round trip each. Opt-in because the refusal it lifts was closing a *silent* fallback | off | no |
| `TS_DSV41_VISION_FA` / `TS_DSV41_VISION_BF16_GEMM` | V4.1 vision companion | `TS_DSV41_VISION_FA=1` selects flash attention with F16 intermediates in the image encoder (larger real-image feature differences); `TS_DSV41_VISION_BF16_GEMM=0` selects the diagnostic F32-promoted matrix path | dense F32 attention, BF16 GEMM with F32 accumulation | no |
| `TS_DSV41_TRACE_DIR` / `TS_DSV41_VISION_TRACE_DIR` | V4.1 (diagnostic) | Directories for text-graph and vision-encoder tensor dumps. Both retain intermediates and add device transfers — leave unset for benchmarks | unset | no |
| `TS_DSV4_CPU_TRACE_DIR` / `TS_DSV4_CUDA_TRACE_DIR` | V4.1 (diagnostic) | The same per-tensor dumps from the pure-C# and direct-CUDA V4.1 executors, in the same file naming as `TS_DSV41_TRACE_DIR`, so two backends can be diffed tensor by tensor to find the first divergence | unset | no |

## Out-of-Matrix GLM 5.x (`glm-dsa`) Knobs

These configure the GLM 5.x (`glm-dsa`) executor — the native whole-model ggml
path used by `ggml_cuda` / `ggml_vulkan` / `ggml_cpu` / `ggml_metal`, and the
managed per-op path used by `cpu` and `cuda`. None are registered in
`EnvVarMatrix.All`, so the default TestMatrix sweep does not touch them; the
full list with context is in the [GLM card](models/glm.md#environment-knobs).
The tensor-parallel knobs (`TS_GLM_TP_SHARD`, `TS_GLM_TP_OVERSUBSCRIBE`,
`TS_GLM_TP_FUSED`) live in the TP table below.

| Variable | Applies to | Effect | Baseline | Values swept | In matrix |
|---|---|---|---|---|---|
| `TS_GLM_NATIVE` | GLM 5.x | `0` runs the managed per-op path on a GGML backend instead of the native whole-model graph — the A/B that proves the two agree | `1` (native) | `0`, `1` | no |
| `TS_GLM_NGPU` | GLM 5.x on GGML | How many GPUs the layer split spreads the trunk layers over | `0` (all visible) | `1`, `2`, `3` | no |
| `TS_GLM_UBATCH` | GLM 5.x | Prefill micro-batch. `2048` is faster on long prompts when VRAM allows: pp2048 1145.8 vs 918.9 t/s on 3x RTX PRO 6000 | `1024` | `512`, `1024`, `2048` | no |
| `TS_GLM_THREADS` | GLM 5.x on `ggml_cpu` | CPU-backend thread count (the routed-expert matmul takes its own count from `--cpu-moe-threads`) | min(cores, 32) | — | no |
| `TS_GLM_FA` | GLM 5.x | `0` disables flash attention and falls back to an explicit `soft_max` chain | `1` (flash) | `0`, `1` | no |
| `TS_GLM_FUSED_LID` | GLM 5.x | `0` builds the DSA lightning indexer out of primitives instead of the fused `ggml_lightning_indexer` op | `1` (fused) | `0`, `1` | no |
| `TS_GLM_TOPK` | GLM 5.x | `0` attends densely past the indexer top-k — an A/B for the sparse selection itself, not a production setting | `1` (sparse) | `0`, `1` | no |
| `TS_GLM_OP_OFFLOAD` | GLM 5.x on GGML | Scheduler op-offload; turned off automatically once any layer's experts are host-resident | auto | `0`, `1` | no |
| `TS_GLM_HC_NATIVE` | GLM 5.3-Flash | `0` decomposes the Sinkhorn hyper-connection pre/post ops into batched mul_mats instead of the fused `ggml_dsv4_hc_*` kernels (A/B; auto-decomposed where the backend has no kernel) | probed | `0`, `1` | no |
| `TS_GLM_VENC_FUSED` | GLM 5.3-Flash vision | `0` runs the GLM-OCR ViT block-by-block through managed ops instead of the one-graph native encoder (`TSGgml_GlmVisionEncoderF32`) | `1` (fused) | `0`, `1` | no |
| `TS_GLM_VRAM_RESERVE_MB` | GLM 5.x on GGML | Per-device headroom the layer split leaves for compute buffers before it starts placing layers | `3072` | — | no |
| `TS_GLM_GRAPH_CACHE` | GLM 5.x on GGML | How many built+allocated graphs are kept, so a repeated shape replays instead of rebuilding | `8` | — | no |
| `TS_GLM_NODES_PER_LAYER` | GLM 5.x on GGML | Graph node budget per layer per rank | `256` | — | no |
| `TS_GLM_MOE_MMAP` | GLM 5.x with `--n-cpu-moe` | `0` copies host-resident experts into a private buffer instead of multiplying them in place out of the GGUF mapping | `1` (mapped) | `0`, `1` | no |
| `TS_GLM_BATCHED_DECODE` | GLM 5.x | `0` makes the native side decline every batched decode, forcing the per-sequence slot path even while the global batched fused decode is enabled | `1` (accepted) | `0`, `1` | no |
| `TS_GLM_LOAD_THREADS` / `TS_GLM_LOAD_CHUNK_MB` | GLM 5.x | Weight-load parallelism and chunk size — 16 reader threads across the six shards move 218 GiB in ~37 s (5.9 GiB/s) from a warm page cache | `16` / `64` | — | no |
| `TS_GLM_TRACE` | GLM 5.x (diagnostic) | Layer list (or `all`) to dump per-layer activation sums in `llama-eval-callback`'s layout, for diffing against llama.cpp | unset | — | no |
| `TS_GLM_BD_DEBUG` | GLM 5.x (diagnostic) | `1` narrates each batched decode step: which slots took part, whether the graph was reused or rebuilt, and how far it got | `0` | `0`, `1` | no |
| `TS_GLM_DEBUG` / `TS_GLM_DEBUG_LAYERS` | GLM 5.x on the managed per-op path (`cpu` / `cuda` / `TS_GLM_NATIVE=0`, diagnostic) | Per-layer activation trace: shape, sum and leading values of every named intermediate, tagged to match `llama-eval-callback` so the two can be diffed tag by tag. `TS_GLM_DEBUG=1` traces layer 0 only; `TS_GLM_DEBUG_LAYERS` takes a layer list. For the native executor use `TS_GLM_TRACE` instead | unset | — | no |

## Out-of-Matrix Tensor Parallelism & Distributed Inference Knobs

These variables configure tensor parallelism (splitting a model across multiple
GPUs) and distributed multi-node TP over a peer-to-peer TCP mesh. They are
not registered in `EnvVarMatrix.All` and are not swept by the default TestMatrix
config — TP requires multiple GPUs, which the standard single-GPU test harness
does not exercise. TP runs on the direct `cuda` backend and on the GGML CUDA /
Vulkan backends (`ggml_cuda`, `ggml_vulkan`). `TENSORSHARP_TP_DEGREE`,
`TENSORSHARP_TP_NODE_ID`, and `TENSORSHARP_TP_PEERS` are also settable via the
`--tp`, `--tp-node-id`, and `--tp-peers` flags on both `TensorSharp.Cli` and
`TensorSharp.Server`.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TENSORSHARP_TP_DEGREE` | all autoregressive models; `cuda`, `ggml_cuda`, `ggml_vulkan` backends | Number of local GPUs to split the model across (Megatron-LM column/row-parallel) | `1` (single GPU) | not registered | no |
| `TENSORSHARP_TP_DEVICES` | local TP on the GGML backends | Comma-separated GPU ordinals the ranks map to (e.g. `0,2`) | `0..tp-1` | not registered | no |
| `TENSORSHARP_TP_NODE_ID` | all autoregressive models; `cuda`, `ggml_cuda`, `ggml_vulkan` backends | This node's 0-based ID for multi-node distributed TP; must be set with `TENSORSHARP_TP_PEERS` | unset (disabled) | not registered | no |
| `TENSORSHARP_TP_PEERS` | all autoregressive models; `cuda`, `ggml_cuda`, `ggml_vulkan` backends | Comma-separated `host:port` list of all nodes in the distributed TP cluster; must be set with `TENSORSHARP_TP_NODE_ID` | unset (disabled) | not registered | no |
| `TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS` | distributed TP only | How long each node retries outbound connections to its peers before failing | `120` seconds | not registered | no |
| `TENSORSHARP_TP_RECV_TIMEOUT_SECONDS` | distributed TP only | Per-receive timeout on a peer socket; a stalled peer fails the collective instead of hanging | `300` seconds | not registered | no |
| `TENSORSHARP_TP_DISABLE_P2P` | local TP, `cuda` backend | `1` forces every cross-GPU transfer through host staging instead of CUDA peer-to-peer DMA (matches no-peer hardware such as A16 vGPU profiles) | off (P2P used when the pair passes the DMA self-test) | not registered | no |
| `TENSORSHARP_TP_HOST_ALLREDUCE` | local TP, `cuda` backend | `1` runs the local AllReduce through host memory (device→host, sum, host→device) instead of the device-to-device path — diagnostic fallback | off (device-to-device) | not registered | no |
| `TS_GGML_TP_PARALLEL` | local TP, GGML backends | `0` drives the ranks sequentially instead of concurrently (diagnostic) | on (concurrent rank workers) | not registered | no |
| `TS_GGML_TP_FUSED_MATMUL` | local TP, GGML backends | `1` submits both ranks' linears from one thread; allocates a device buffer per rank per call and measured 2.3× slower on Qwen 3.5 35B | off (generic per-rank path) | not registered | no |
| `TS_GGML_TP_DEVICE_AR_THRESHOLD` | local TP, GGML backends | Element count above which AllReduce uses the device collective instead of the host reduction | `262144` | not registered | no |
| `TS_GGML_F32_RESIDENT` | GGML backends | `0` binds F32 linear weights per call instead of keeping them device-resident (diagnostic) | on (device-resident) | not registered | no |
| `TS_GEMMA4_TP_FUSED_MOE` | Gemma 4 MoE under TP on GGML | `0` falls back from the fused whole-model MoE trunk (Megatron split inside each expert) to the whole-expert per-op path | on (fused trunk) | not registered | no |
| `TS_GLM_TP_SHARD` | GLM 5.x under TP on GGML | Which halves of the split are applied: `1` heads, `2` routed experts, `3` both. The experts are split row-wise inside every expert rather than by expert id, because `ggml_mul_mat_id` needs a token's selected expert ids to stay distinct | `3` (both) | `1`, `2`, `3` | no |
| `TS_GLM_TP_OVERSUBSCRIBE` | GLM 5.x under TP on GGML | `1` packs several ranks onto one GPU so the split can be checked for correctness on a single-GPU machine | `0` (one rank per GPU) | `0`, `1` | no |
| `TS_GLM_TP_FUSED` | GLM-5.3-Flash local TP on GGML | `0` forces the combined scheduler diagnostic fallback instead of concurrent segmented rank-local graphs. The fallback is also selected automatically by CPU MoE, tensor tracing, partial `TS_GLM_TP_SHARD`, oversubscribed ranks, or a backend without native hyper-connection kernels | auto (segmented when eligible) | `0`, `1` | no |
| `TS_Q4E_LAYER_SPLIT` | Qwen 3.8 Flash Next (`qwen4exp`) multi-GPU layer split under `--tp N` | Explicit layer counts per GPU, comma-separated (e.g. `20,28`), instead of the automatic VRAM balance; throws rather than silently ignoring a value it cannot honour. `--tp N` on this architecture is a layer split, not tensor parallelism — `qwen4exp` shards no weights | automatic (layers bin-packed to each device's free VRAM) | not registered | no |
| `GGML_CUDA_ALLREDUCE` | local TP, `ggml_cuda` | `nccl` / `internal` / `none` — passed through to ggml's collective selection; setting it explicitly also skips the pre-flight probe | auto (NCCL when the build finds it and it passes the probe) | not registered | no |
| `TS_GGML_TP_CUDA_GRAPHS` | local TP, `ggml_cuda` | `0` turns CUDA graph capture off for multi-GPU runs. Capture is ON by default under TP because a tensor-parallel token is dozens of small per-rank submissions that replay far more cheaply than they re-issue (4×A40: Qwen3.5-9B tp4 88 → 128.5 tok/s, Qwen3.5-35B-A3B tp2 71.3 → 104.1). It was historically disabled over a capture-poisoning hazard that no longer applies — ggml captures with `cudaStreamCaptureModeRelaxed`. The opt-out is translated into a native `GGML_CUDA_DISABLE_GRAPHS` before the first backend call, because ggml latches that value on first use | capture enabled | not registered | no |
| `TS_GGML_TP_AR_PROBE` | local TP, `ggml_cuda` | `0` skips both pre-flight probes; `force` re-probes, ignoring the cached verdicts (`~/.cache/tensorsharp/tp-collective-probe`). Before model load the group checks that peer copies between advertised device pairs actually deliver bytes, and that one small NCCL AllReduce completes end to end — some cloud hosts advertise P2P that never arrives, and NCCL's first collective then spins every GPU forever. A failed peer check keeps NCCL but takes peer transport away from it (`NCCL_P2P_DISABLE=1`), which is what preserves a device collective past 2 GPUs | probes on, verdicts cached per driver/NCCL/GPU set | not registered | no |
| `TS_GGML_TP_AR_PROBE_MS` | local TP, `ggml_cuda` | Deadline for each probe (peer copy, then AllReduce) before that transport is declared broken; the collective then falls back to the pinned-host `internal` pipeline at 2 GPUs, or to the host reduction beyond it. `0` disables the probes | `10000` ms | not registered | no |
| `GGML_CUDA_AR_BF16_THRESHOLD` | local TP, `ggml_cuda` | Payload size above which ggml converts F32 collectives to BF16; TensorSharp raises ggml's default to 1 MB so decode-sized reductions stay exact | `1 MB` (set by `TSGgml_TensorParallelInit`) | not registered | no |
| `TS_QWEN35_LAYER_TRACE` | Qwen 3.5/3.6 | `1` prints a per-layer residual-stream summary for the first forward, from both the single-GPU and TP loops (diagnostic) | off | not registered | no |

## Out-of-Matrix Redis Shared-State Knobs

These variables configure optional Redis-backed shared state in
`TensorSharp.Server`: a shared KV cache tier for cross-session reuse and a
Redis-backed OpenAI Responses API store. They are not registered in
`EnvVarMatrix.All`. `TS_KV_CACHE_REDIS_URL` is also settable via `--redis-url`
or `--paged-kv-redis-url`; `TS_KV_CACHE_REDIS_TTL_MINUTES` via
`--paged-kv-redis-ttl`; `TS_RESPONSES_STORE_REDIS_URL` via `--redis-url`.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_KV_CACHE_REDIS_URL` | `TensorSharp.Server` | Redis connection string for the shared KV cache tier; when set, KV blocks are persisted to Redis for cross-session reuse | unset (disabled) | not registered | no |
| `TS_KV_CACHE_REDIS_TTL_MINUTES` | `TensorSharp.Server` | TTL in minutes for Redis KV cache entries; `0` = no TTL | `1440` (24 h) | not registered | no |
| `TS_RESPONSES_STORE_REDIS_URL` | `TensorSharp.Server` | Redis connection string for the OpenAI Responses API store; when set, `RedisResponsesStore` replaces the in-memory store | unset (disabled, in-memory) | not registered | no |

## Out-of-Matrix General Runtime Knobs

These variables are real runtime knobs, but they are not registered in
`EnvVarMatrix.All` today and are not swept by the default TestMatrix config.

| Env var | Applies to | Feature impact | Runtime baseline | Sweep values | Swept by default |
|---|---|---|---|---|---|
| `TS_PDF_MAX_PAGES` | PDF document input (CLI `--pdf`, server `/api/upload`) | Cap on the number of PDF pages read for text extraction and page-image rendering | `0` (all pages) | not registered | no |
| `TS_GGUF_PREFAULT` / `TS_GGUF_PREFAULT_THREADS` | Model load, every architecture that goes through `GgufReader` | `0` skips the parallel page-cache warm-up that runs before the loaders touch the file; the threads variable sets its stream count. The load path itself reads through one or two streams, which caps a cold load at single-stream throughput — on a MooseFS volume that is ~440 MB/s against ~1.8 GB/s at 8-16 streams. The pass skips itself on iOS and on files larger than half of available RAM, and is a no-op on an already-cached file | on, `min(16, cores)` | not registered | no |
| `TS_DIRECT_QUANT_WEIGHTS` | `cpu` backend, direct video networks (Wan, MiniMax-H3) | `0` expands every quantized weight to F32 once at load and runs a plain GEMM, instead of keeping it in its GGUF storage type and multiplying it straight out of there. The expansion costs 4x the weight memory and reads 4x the bytes on every forward; it is kept so the two can be A/B-ed for numeric drift in one binary | ON (weights stay quantized) | not registered | no |
| `TS_DUMP_LOGITS` | all models, all backends | Path the FIRST real forward's logits are written to, once, as raw float32. It deliberately SKIPS the warm-up forwards: `WarmUpKernels` drives its own throwaway decode and prefill before the real prompt, so dumping those would compare two executors on a meaningless token rather than on the model. Lets two backends be compared by logit vector instead of by generated text, where greedy decoding turns a near-tie into a visibly different sentence | unset (no dump) | not registered | no |
| `TS_FUSED_QKNORM_ROPE` | Qwen 3.5 / 3.6 text-only prefill on the direct `cuda` backend | Fused QK-Norm + NeoX-RoPE CUDA kernel; `0` falls back to separate norm + RoPE ops (multimodal MRoPE and other backends always use the separate path) | ON | not registered | no |
| `TS_CUDA_QMM_F16GEMM` | direct `cuda` backend, quantized matmuls with ≥ `TS_CUDA_QMM_F16GEMM_MIN_ROWS` activation rows | Dequantize the weight once to F16 and run a tensor-core cuBLAS GEMM (ggml-style prefill route) instead of the block-tile quant kernels; `0` reverts to the quant kernels | ON | not registered | no |
| `TS_CUDA_QMM_F16GEMM_MIN_ROWS` | direct `cuda` backend | Activation-row threshold for the F16 GEMM route | `32` | not registered | no |
| `TS_CUDA_QMM_F16GEMM_MAX_MB` | direct `cuda` backend | F16 weight-scratch cap in MB; weights above it (e.g. the LM head) keep the quant kernels | `768` | not registered | no |
| `TS_CUDA_Q80_VEC` | direct `cuda` backend, Q8_0 single-row (decode) matmuls | Warp-per-column dp4a matvec over a q8_1-quantized activation row (like ggml `mul_mat_vec_q`); `0` reverts to the exact FP32 dequant kernel | ON | not registered | no |
| `TS_CUDA_Q80_VEC_MIN_OUT` | direct `cuda` backend | Minimum output width for the Q8_0 dp4a matvec (diagnostic gate) | `0` | not registered | no |
| `TS_CUDA_Q80_MMQ` | direct `cuda` backend, Q8_0 matmuls with 32..`TS_CUDA_Q80_MMQ_MAX_ROWS` activation rows | Direct int8 tensor-core GEMM over raw Q8_0 blocks (mma.m16n8k32, ggml MMQ-style) instead of the dequant+cuBLAS F16 route; `0` reverts to F16 GEMM | ON | not registered | no |
| `TS_CUDA_Q80_MMQ_MAX_ROWS` | direct `cuda` backend | Row-count crossover above which the F16 GEMM route wins (MMQ weight sweeps grow as ceil(rows/128)) | `512` | not registered | no |
| `TS_CUDA_Q80_MMQ2` | direct `cuda` backend | cp.async staging variant of the MMQ GEMM (split q8_1 activation scratch + raw weight windows async-copied to shared; taken when inDim % 256 == 0, bit-identical results, ~18% faster prefill); `0` pins the register-prefetch MMQ kernel | ON | not registered | no |
| `TS_CUDA_GDN_PREFILL_SPLIT` | Qwen 3.5 / 3.6 GDN prefill on the direct `cuda` backend | 3-phase sync-free GDN prefill (parallel conv/norm → register-resident row scan → parallel RMS+gate); `0` pins the legacy single-kernel sequential walk | ON (seqLen ≥ 8, headKDim = 128) | not registered | no |
| `TS_CUDA_PREFILL_GRAPH` | Qwen 3.5 / 3.6 text-only multi-token prefill on the direct `cuda` backend | Capture the per-op prefill layer loop as a CUDA graph on the second run of a (seqLen, startPos, cache-identity) shape and replay it in one `cuGraphLaunch` afterwards (bit-identical results; falls back plainly on any capture failure); `0` disables ALL cuda graph capture, including decode graphs | ON | not registered | no |
| `TS_CUDA_DECODE_GRAPH` | Qwen 3.5 / 3.6 text-only decode on the direct `cuda` backend | Capture the per-op decode step (seqLen = 1) as a CUDA graph and replay it every token; position-dependent values (attention length, KV write slot, GDN conv ring index, RoPE position) are re-read from a pinned-host-backed device parameter block, so one graph serves all positions until the KV cache grows (bit-identical results; falls back plainly on any capture failure); `0` disables | ON | not registered | no |
| `TS_CUDA_PREFILL_GRAPH_MAX` | direct `cuda` backend | Cached prefill + decode graphs kept (LRU-evicted; each pins its captured working-set pool blocks) | `4` | not registered | no |
| `TS_CUDA_PREFILL_GRAPH_LOG` | direct `cuda` backend | Log graph capture/replay/abort events (`1`) | OFF | not registered | no |
| `TENSORSHARP_CUDA_POOL_LARGE_MB` | direct `cuda` backend | Budget for the global large-block (≥ 2 MB) device-memory cache; keeps prefill-sized activations pooled instead of re-issuing cuMemAlloc/cuMemFree per layer | `1024` | not registered | no |
| `TS_CUDA_PROFILE` | direct `cuda` backend | Print CPU-fallback op and host↔device sync counters at exit (`1`), with call-site attribution (`2`) | OFF | not registered | no |

## Feature Coverage

The matrix feature catalog lives in
[`TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs`](../TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs).
The current feature set is:

| Feature | Driver | Capability gate |
|---|---|---|
| `pp512` | `--benchmark --bench-prefill 512 --bench-decode 0` | all models |
| `pp2048` | `--benchmark --bench-prefill 2048 --bench-decode 0` | all models |
| `tg128` | `--benchmark --bench-prefill 32 --bench-decode 128` | all models |
| `short_text` | `--input prompts/short_text.txt --max-tokens 64` | all models |
| `long_text` | `--input prompts/long_text.txt --max-tokens 64` | all models |
| `uploaded_text` | `--input prompts/upload_text.txt --max-tokens 64` | all models |
| `multi_turn` | `--multi-turn-jsonl multi_turn/three_turn.jsonl` | all models |
| `tools` | `--tools tools/weather_tools.json` | models whose matrix capability says tool calling is supported |
| `thinking` | `--think` | models whose matrix capability says thinking is supported |
| `image` | `--image media/apple.png --mmproj ...` | image-capable models with an mmproj |
| `audio` | `--audio media/sample.mp3 --mmproj ...` | audio-capable models with an mmproj |
| `video` | `--video media/sample.mp4 --mmproj ...` | video-capable models with an mmproj |

Default semantic checks are intentionally weak and catch catastrophic failures:
`blue`, `paged`, `08:01:12`, `alex` + `teal`,
`get_current_weather` + `tokyo`, `10:38`, and `apple` for the relevant text,
multi-turn, tools, thinking, and image features. Audio and video have no default
expected substring because the sample media is runner-provided.

## Filters

The runner filters the combinatorial product before execution:

1. Backend availability: CUDA and Vulkan backends are skipped on macOS (Metal
   is the GPU backend there); MLX requires Apple Silicon; GGML Metal requires
   macOS.
2. Model capability: image/audio/video/tool/thinking features are skipped when
   the discovered or configured model does not advertise that capability.
3. Projector availability: multimodal features require an mmproj path.
4. Env-var applicability: each `EnvVarSpec.AppliesTo` predicate decides whether
   a variable is meaningful for the `(model, backend, feature)` cell.

## Updating The Matrix

To add a new high-impact env var:

1. Register an `EnvVarSpec` in
   [`TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs`](../TensorSharp.TestMatrix/Matrix/EnvVarMatrix.cs).
2. Add it to `default_env_vars` in
   [`Defaults/matrix-config.json`](../TensorSharp.TestMatrix/Defaults/matrix-config.json)
   if it should run in the default sweep.
3. Add or update the row in this document and its Chinese counterpart.
4. If the variable changes feature applicability, update
   [`FeatureCatalog.cs`](../TensorSharp.TestMatrix/Matrix/FeatureCatalog.cs)
   or model discovery capability heuristics as needed.
