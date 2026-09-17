# Paged Attention & Continuous Batching in TensorSharp

[English](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md) | [中文](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)

This document is the current implementation reference for TensorSharp's
vLLM-style paged KV cache, block-hash prefix sharing, and iteration-level
continuous batching. The server now routes inference through this engine by
default; the old single-request FIFO queue object remains only as a no-op
compatibility shim for queue-status/event shapes.

Read it as an implementation reference, not a universal performance claim. The
generic paged K/V pool is host-resident and remains a bottleneck, while supported
model/backend pairs can instead use device-resident, token-batched fused decode.
Throughput therefore depends on the selected and accepted execution path. See
[Measured Concurrency Behavior](#measured-concurrency-behavior).

## Current Status

| Area | Status |
|---|---|
| Server engine | `TensorSharp.Server` owns one `InferenceEngineHost` per loaded model. `ChatGenerationPipeline` submits rendered prompts to the engine and streams tokens from `InferenceRequestHandle`. |
| Scheduler | `ContinuousBatchScheduler` admits waiting requests, preempts running work when block pressure requires it, applies a per-step token budget, and shares full prefix blocks by hash. |
| KV storage | `BlockPool`, `BlockTable`, `PagedKvStorage`, and `BlockHashIndex` hold fixed-size physical blocks with ref counts, LRU free ordering, and content-addressed lookup. The block bytes live in **managed host memory**. |
| Batched execution | Models that implement `IBatchedPagedModel.ForwardBatch` pack all scheduled sequences into one model call with explicit `positions`, `slotMapping`, `queryStartLoc`, and per-sequence block tables. |
| Fallback execution | Path selection is centralized in `ExecutionPlanner`: model+backend capabilities (`ExecutionCapabilities`), operator overrides (`ExecutionOptions`), and per-step request features produce an `ExecutionPlan` (selected path, fallback chain, rejection reasons). A model may still decline a specific batch with `NotSupportedException`; the step then falls to the plan's next candidate, ending in the per-sequence KV-swap path. |
| Native attention | `TSGgml_PagedAttentionForward` gathers paged K/V in C++ and dispatches `ggml_flash_attn_ext`; GPT OSS uses `TSGgml_PagedAttentionForwardWithSinks`. |
| Speculative decoding | Optional MTP / NextN draft heads accelerate solo (non-concurrent) sequences. `BatchExecutor` drives the shared `SpeculativeExecution` draft / verify / rollback core for models that implement `IBatchedSpeculativeTarget` (Qwen 3.6 embedded NextN; Gemma 4 separate `gemma4-assistant` draft GGUF). Off by default; server `--spec`. See [Speculative decoding (MTP / NextN)](#speculative-decoding-mtp--nextn). |
| Throughput under concurrency | The generic host-resident `BatchedPaged` route does not itself scale aggregate decode; one Gemma 4 measurement saturated near **69 tok/s**. On a model that implements `TryForwardBatchedFusedDecode`, the per-sequence-fused route first attempts one device graph for the active decode subset and falls back per request only when that batch is ineligible or declined. See [Measured Concurrency Behavior](#measured-concurrency-behavior). |
| Device-resident paged pool | **Built, not wired.** `TSGgml_PagedKvPool*` (`TensorSharp.GGML.Native/ggml_ops_paged_kv_pool.cpp`) and its managed wrapper `DevicePagedKvCache` (`TensorSharp.Models/Paged/DevicePagedKvCache.cs`) exist, but no model or executor calls them. It is not a shipped feature. |
| Queue API | `InferenceQueue` is a no-op shim. `/api/queue/status` and queue-position event shapes are retained for clients that expect the fields, not because requests are serialized there. |
| Diffusion models | DiffusionGemma does not enter this autoregressive `ForwardBatch` contract. CLI generation uses `DiffusionGemmaSampler`; the Web UI uses `DiffusionBatchScheduler` to batch denoising work at block boundaries. |

## Measured Concurrency Behavior

The results below are path- and model-specific. They explain why the generic
host-resident paged route remains slow; they do not describe a model/backend pair
that successfully takes token-batched fused decode.

- **The measured host-paged path did not turn concurrency into aggregate
  throughput.** On gemma-4-E4B / 1x Blackwell through the server's chat endpoint, the
  `BatchedPaged` route saturates at roughly **69 tok/s** no matter how many
  sequences are in flight.
- **The cause is where the K/V lives.** `PagedKvStorage` is managed host memory,
  so the batched path gathers a sequence's history out of host memory and pushes
  it across the bus for every layer of every step. Even the native kernels
  zero-copy only Q and OUT — `ggml_ops_paged_attention.cpp` says as much: "K and
  V are still passed as host scratch arrays (the caller gathers …)". That is
  about a **7.7x per-token penalty** against per-sequence fused decode, and it
  grows with total history.
- **The default concurrent path is capability-driven.** On models that declare
  `SupportsPerSequenceFusedForward`, `ExecutionPlanner` selects `PerSequenceFused`
  at N >= 2. The executor first offers an eligible decode subset to
  `TryForwardBatchedFusedDecode`, which can amortize the weight read in one graph
  (including Qwen 3.5/3.6's slot-stable arena on GGML CUDA/Metal). If the model
  declines, the same step falls back to N isolated fused forwards.
- **What does work.** Iteration-level scheduling, block-hash prefix sharing,
  preemption, per-sequence native slots, per-request fused holders, and
  capability-gated token-batched fused decode. A decline preserves correctness
  and fairness but may return throughput to the round-robin ceiling.
- **Built but not wired.** A device-resident paged K/V pool exists
  (`TensorSharp.GGML.Native/ggml_ops_paged_kv_pool.cpp`, wrapped by
  `TensorSharp.Models/Paged/DevicePagedKvCache.cs`): the pool is backend tensors,
  a step's K/V is written with `ggml_set_rows`, and a sequence's history is
  gathered on device with `ggml_get_rows` inside the attention graph. **No model
  or executor calls it.** It is not a shipped feature, and it does not change any
  number above.

## Layered Architecture

```text
Adapters (Web UI / Ollama / OpenAI)
        |
        v
ChatGenerationPipeline
  - render prompt
  - prepare multimodal embeddings
  - submit SequenceState
  - stream InferenceRequestHandle tokens
        |
        v
InferenceEngine
  - worker thread
  - submit / abort API
  - completion futures
        |
        +--> ContinuousBatchScheduler
        |      - waiting / running sets
        |      - token and sequence budgets
        |      - block allocation / preemption
        |      - prefix block adoption
        |
        +--> BatchExecutor
               - runs the path ExecutionPlanner selected
               - per-sequence fused, batched ForwardBatch,
                 or the per-sequence KV-swap fallback
               - samples decode tokens
               - captures newly full blocks
        |
        v
BlockPool + PagedKvStorage + BlockHashIndex   (managed host memory)
```

### Core Components

| Component | File | Role |
|---|---|---|
| `KvBlock` | `TensorSharp.Runtime/Paged/KvBlock.cs` | Physical block metadata, ref counts, hash metadata. |
| `BlockPool` | `TensorSharp.Runtime/Paged/BlockPool.cs` | Allocates, frees, ref-counts, and evicts blocks. |
| `BlockTable` | `TensorSharp.Runtime/Paged/BlockTable.cs` | Maps each sequence's logical block ids to physical block ids. |
| `PagedKvStorage` | `TensorSharp.Runtime/Paged/PagedKvStorage.cs` | Byte slabs keyed by physical block id. |
| `BlockHashIndex` | `TensorSharp.Runtime/Paged/BlockHashIndex.cs` | Content hash to block lookup for prefix reuse. |
| `PagedKvBatchOps` | `TensorSharp.Runtime/Paged/PagedKvBatchOps.cs` | Batched K/V scatter and last-token gather helpers. |
| `ManagedPagedAttention` | `TensorSharp.Runtime/Paged/ManagedPagedAttention.cs` | Pure C# correctness fallback for paged attention. |
| `TensorPagedAttention` | `TensorSharp.Models/Paged/TensorPagedAttention.cs` | Tensor-op paged attention fallback. |
| `DevicePagedKvCache` | `TensorSharp.Models/Paged/DevicePagedKvCache.cs` | Device-resident paged K/V pool over `TSGgml_PagedKvPool*`. **Built, not wired into any model** — nothing constructs it today. |
| `SequenceState` | `TensorSharp.Runtime/Scheduling/SequenceState.cs` | Mutable per-request status, tokens, blocks, logits, and sampling state. |
| `ContinuousBatchScheduler` | `TensorSharp.Runtime/Scheduling/ContinuousBatchScheduler.cs` | Iteration-level scheduler with prefix caching and preemption. |
| `BatchExecutor` | `TensorSharp.Runtime/Scheduling/BatchExecutor.cs` | Executes the planned step, samples, and captures KV blocks. |
| `ExecutionPlanner` | `TensorSharp.Runtime/Scheduling/ExecutionPlanner.cs` | Pure-function path selection: capabilities + options + step features → `ExecutionPlan`. |
| `ExecutionCapabilities` | `TensorSharp.Runtime/Scheduling/ExecutionCapabilities.cs` | Declared capability snapshot of the loaded model × backend combination. |
| `ExecutionOptions` | `TensorSharp.Runtime/Scheduling/ExecutionOptions.cs` | Structured snapshot of the executor-level `TS_*` overrides (single place that reads them). |
| `InferenceEngine` | `TensorSharp.Runtime/Scheduling/InferenceEngine.cs` | Worker loop and public submit/abort surface. |
| `InferenceEngineHost` | `TensorSharp.Server/InferenceEngineHost.cs` | Server-side per-model engine singleton. |

## Request Flow

1. A protocol adapter builds a normalized chat request.
2. `ChatGenerationPipeline` renders the prompt, resolves sampling options, and prepares any image/audio/video embeddings.
3. The pipeline creates a `SequenceState` and calls `InferenceEngine.SubmitRequest`.
4. The engine worker asks `ContinuousBatchScheduler` for the next step.
5. The scheduler admits waiting sequences while token and sequence budgets allow. Before allocating new blocks, it looks up full prompt blocks in `BlockHashIndex` and adopts shared blocks on a hit.
   A waiting request is admitted only when the free pool can hold its whole prompt on top of the prompt blocks the running requests still have to allocate; otherwise it stays queued until one finishes (a lone request is always admitted).
6. If the pool is under pressure (decode growth is not reserved), the scheduler can preempt a running sequence that ranks below the one needing blocks (lower priority, or submitted later), commit its full blocks, free the remainder, and requeue it. A sequence never preempts an older one: it waits a step instead, so a full pool drains oldest-first rather than livelocking long prefills against each other.
7. `BatchExecutor` executes the scheduled step. It asks `ExecutionPlanner` for the step's `ExecutionPlan` and runs the first candidate path that accepts the step (see [Execution Planning](#execution-planning-capability-model)).
8. The engine emits sampled tokens to the request handle, checks EOS / max-tokens / abort state, and releases blocks for completed sequences.

Prefix adoption is capped so at least one prompt token still runs through the
model. That keeps logits fresh for sampling even when the entire visible prefix
is already present in the block-hash cache.

## Batched Forward Contract

`IBatchedPagedModel.ForwardBatch(BatchedForwardContext ctx)` receives a compact
batch description:

| Field | Meaning |
|---|---|
| `Sequences` | Scheduled sequence states in output order. |
| `InputTokens` | Concatenated prefill or decode tokens across all scheduled sequences. |
| `Positions` | Absolute position per token. |
| `QueryStartLoc` | Prefix-sum offsets into `InputTokens`, length `numSeqs + 1`. |
| `SlotMapping` | Flat paged write slot per token: `blockId * blockSize + offset`. |
| `BlockTables` | Per-sequence physical block table used by paged attention. |

The model batches embedding, projections, norms, FFN/MoE, and final logits over
the concatenated token axis. It scatters fresh K/V into paged buffers using
`SlotMapping`, then reads the per-sequence block tables during attention. It
returns one logits array per sequence, in the same order as `ctx.Sequences`.

## Execution Planning (Capability Model)

The number of path combinations (batched / fallback, fused / op-by-op,
multimodal / text, speculative / standard, per-model opt-outs, `TS_*`
overrides) grew past what ad-hoc `if` chains could keep reviewable, so path
selection is a single pure function:

```text
ExecutionCapabilities (model × backend, declared)
        +
ExecutionOptions      (operator TS_* overrides, read in one place)
        +
SchedulerConfig       (engine config, e.g. --spec)
        +
ExecutionStepFeatures (this step's requests: N, multimodal pending,
        |              KV residency, fused-cache residency, swap needs)
        v
ExecutionPlanner.PlanStep(...)
        |
        v
ExecutionPlan
  - Selected path + ordered fallback chain
  - Rejections: every plausible path that was not taken, with the reason
```

Key points:

- **Declared capabilities, not exception probing.** Models declare what they
  can run via `IBatchedPagedModel` getters (`BatchedForwardAvailable`,
  `SupportsBatchedMultimodal`, `SupportsPerSequenceFusedForward`,
  `SupportsLinearKVMigration`, …) and the MTP interfaces
  (`HasMtp`, `SpeculationProfitable`, `SupportsBatchedSpecTrunk`).
  `ExecutionCapabilities.FromModel` snapshots them per step. A per-model
  opt-out such as `TS_QWEN35_BATCHED=0` now surfaces through
  `BatchedForwardAvailable=false` so the planner routes around the batched
  path up front; `ForwardBatch` throwing `NotSupportedException` remains only
  as a per-batch decline, not the routing mechanism.
- **Plan candidates are ordered and safe.** Declinable candidates
  (`SpecBatchedTrunk` arming/continuity, `BatchedPaged` migration/refusal)
  fall through to the next entry; every plan ends in a path that cannot
  decline. `ExecutionPlannerTests` sweeps the capability/feature space to
  assert this invariant.
- **Observability.** `InferenceEngine` logs a one-time capability report at
  startup (which paths are statically available and why the others are not).
  `BatchExecutor` logs the plan — selected path, fallback chain, rejection
  reasons — whenever the decision changes (e.g. a concurrency transition),
  so "why did this request not take the fast path?" is a logged fact.
- **Path kinds** (`ExecutionPathKind`): `SpecBatchedTrunk`, `SpecPerSequence`,
  `PerSequenceFused`, `MixedMultimodalSplit`, `SingleSequenceFused` (the N=1
  fast path), `BatchedPaged`, `PerSequence`.

## Execution Paths

### Batched Path

The batched path packs every scheduled sequence into one forward: it avoids K/V
ownership swaps and amortizes linear projections across all scheduled tokens. It
is the only path that amortizes the weight read across a batch, but its K/V pool
is host-resident, so that amortization does not currently show up as throughput
(see [Measured Concurrency Behavior](#measured-concurrency-behavior)). On models
that declare a per-sequence fused forward the planner takes `PerSequenceFused` at
N >= 2 instead, so `BatchedPaged` serves models without one — or an explicit
`TS_PER_SEQ_FUSED=0` A/B. Most current batched ports use native paged attention
for GGML backends:

| Kernel | Scope | Notes |
|---|---|---|
| `TSGgml_PagedAttentionForward` | Standard causal / sliding-window attention | C++ K/V gather plus `ggml_flash_attn_ext`. Default for Mistral 3 and most paged attention layers on GGML backends. |
| `TSGgml_PagedAttentionForwardWithSinks` | GPT OSS attention sinks | Adds the learned per-head sink logits to the softmax denominator. |
| `TensorPagedAttention.Forward` | Tensor-op fallback | Uses tensor gathers plus batched matmul/softmax ops. Useful for A/B testing. |
| `ManagedPagedAttention.Forward` | Pure C# fallback | Online-softmax implementation used for correctness and unsupported backend fallback. |

`TS_PAGED_ATTN_KERNEL=native|tensor|managed` selects the Mistral 3 dispatch path.
GPT OSS can force the managed sinks path with `TS_GPTOSS_PAGED_ATTN_MANAGED=1`.

### Per-Sequence Fallback

The fallback path still runs inside `InferenceEngine`; it is no longer the
server's outer concurrency primitive. It temporarily installs one sequence's
K/V state into the legacy model cache, calls `model.Forward(tokens)`, captures
full blocks, and moves to the next scheduled sequence. This keeps older or
feature-limited paths correct while they are being ported to true batched
compute.

### Speculative decoding (MTP / NextN)

When the server's `--spec` flag (env `TS_SPEC=1`, legacy `TS_MTP_SPEC=1`) is set, `BatchExecutor` runs an optional
multi-token-prediction speculative path for **solo (non-concurrent)** sequences
on models that implement `IBatchedSpeculativeTarget`. The flow per step:

1. **Draft.** The model's draft head proposes up to `TS_MTP_DRAFT` (default `8`)
   future tokens, stopping at the first token whose draft confidence falls below
   `TS_MTP_PMIN` (default `0.15`). The request's own sampler — temperature,
   top-k/p, and repetition/presence/frequency penalties — drives the drafting so
   the speculation stays aligned with what standard decode would have produced.
2. **Verify.** The trunk verifies all drafted tokens in a single batched forward
   and the same sampler accepts the longest matching prefix. Because verification
   re-derives every committed token, the output is **identical** to standard
   decode; speculation only changes how many forward passes it takes.
3. **Rollback.** On partial acceptance, KV (and any recurrent state) past the
   accepted prefix is rolled back before the next step.

Two draft-head shapes share the `SpeculativeExecution` core:

| Model | Draft head | State on rejection |
|---|---|---|
| Qwen 3.6 | Embedded NextN block in the trunk GGUF (`{arch}.nextn_predict_layers`); no extra file. `--draft-model` is ignored. | GatedDeltaNet recurrent-state snapshot/restore (device-side on CUDA). |
| Gemma 4 | Separate EAGLE-style `gemma4-assistant` GGUF via `--draft-model`; draft layers attend the **target's** last local / global KV (no draft K/V of its own). | Attention-KV position rewind only — the drafter is stateless given `(token, h)`. |

Speculation engages only where it is profitable (`SpeculationProfitable`):
ggml backends (fused multi-token-verify + draft-step kernels) and the pure-C#
`cuda` backend (GPU-resident per-op verify/draft). On CPU / GGML CPU / MLX the
verify can't keep up, so the engine serves standard decode. Concurrent batches
never speculate — when more than one sequence is running, every sequence uses the
normal batched/fallback step. A mismatched or incomplete Gemma 4 draft GGUF
fails fast at server startup (`SpeculationStartupValidation`).

## Model Status

| Model family | Batched / paged status | Opt-out / sub-toggle |
|---|---|---|
| Mistral 3 | Default `ForwardBatch` path. Uses paged K/V, YaRN-aware positions, native paged attention, and vision embedding injection after prompt preparation. Validated on Ministral-3-14B; long-context native paged attention is about 21% faster than the legacy per-sequence GGML path. | `TS_PAGED_ATTN_KERNEL` selects `native`, `tensor`, or `managed`. |
| Gemma 4 | Default batched path for dense text workloads, including per-layer SWA/global attention, variable head dims, PLE, and KV-donor layer aliasing. Current fallback cases include pending multimodal embeddings, MoE layers, and block-quantized KV cache. Finished request-owned fused K/V holders can be retained for an exact-prefix continuation. Optional MTP speculative decode via a separate `gemma4-assistant` draft GGUF. | `TS_GEMMA4_BATCHED=0` forces per-sequence fallback. `TS_RETAINED_FUSED_CACHE=0` disables retained-holder continuation. Server `--draft-model` enables speculation by itself (`--no-spec` vetoes); `TS_GMTP_*` are draft-path A/B switches. |
| Qwen 3.5 / 3.6 family | Default batched path. Handles full-attention layers, GatedDeltaNet recurrent layers via per-slot state pools, MoE variants, vision injection, and multimodal RoPE tables. Its request-owned fused holder keeps attention K/V and the matching GDN recurrent state together; a cleanly finished holder can be retained and re-keyed for an exact-prefix continuation. Qwen 3.6 additionally supports MTP speculative decode via its embedded NextN block (GDN recurrent-state snapshot/rollback). | `TS_QWEN35_BATCHED=0`; `TS_QWEN35_BATCHED_GDN_NATIVE=1` enables the native batched GDN kernel; `TS_RETAINED_FUSED_CACHE=0` disables retained-holder continuation; server `--spec` enables speculation on Qwen 3.6. |
| GPT OSS | Default batched path. Handles Q/K/V/O bias, YaRN RoPE, sliding-window layers, attention sinks, MXFP4 MoE experts, and native sinks attention. Greedy correctness has been validated against the legacy path; performance remains limited by per-layer graph construction. | `TS_GPTOSS_BATCHED=0`; `TS_GPTOSS_PAGED_ATTN_MANAGED=1`. |
| Nemotron-H | Default batched path. Attention layers use paged K/V; Mamba2 layers use per-slot conv/SSM state pools; MoE layers use batched expert kernels; prepared image/audio embeddings can be injected into the batched hidden state. | `TS_NEMOTRON_BATCHED=0`; `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` enables the native batched Mamba2 step. |
| GLM 5.x | No `ForwardBatch`: MLA keeps one compressed row per token and the DSA indexer scores against that same contiguous history, so there is no paged-KV layout to batch over. Concurrency runs on native **sequence slots** instead (`TSGgml_GlmSlotAlloc` / `SetActiveSlot` / `SlotFree`) — binding a request switches the active slot without moving KV bytes, and each slot's graphs are cached and captured independently. A default-on batched fused decode (one graph, one token per sequence, weights read once for the batch) is implemented on top: 1.81x aggregate decode at 4 concurrent requests. Batching changes GEMM shapes, and a 2-bit MoE can amplify that into different expert picks. | `TS_BATCHED_FUSED_DECODE=0` disables the batched decode; `TS_GLM_BATCHED_DECODE=0` makes the native side decline it. |
| DiffusionGemma | Separate text-diffusion path. `Forward(int[] tokens)` is intentionally unsupported; generation iteratively denoises fixed-length canvas blocks. Web UI requests share `DiffusionBatchScheduler`, which admits concurrent requests between blocks and can optionally batch active canvases. | `DIFFUSION_STEPS`, `DIFFUSION_MAX_BATCH`, `DIFFUSION_BATCHED_FORWARD`; `DIFFUSION_NO_FUSED_DECODE=1` disables the GGML whole-model diffusion decode. |

### Retained fused-holder continuation

This is distinct from the shared paged-prefix cache. A model may be unable to
reconstruct its complete continuation state from byte-level paged snapshots but
still own a self-contained per-request fused holder. When the model advertises
`SupportsRetainedFusedCache`, the executor may keep a cleanly finished holder in
a small LRU and re-key it when a later request exactly extends the recorded token
prefix. Gemma 4 retains its circular attention K/V; Qwen 3.5/3.6 retains the
attention K/V and matching GatedDeltaNet recurrent state as one hybrid holder.
Models that do not advertise this capability ignore the retained-cache setting.

## Test Coverage

| Area | Tests |
|---|---|
| Scheduler / block pool | `ContinuousBatchSchedulerTests`, `PagedKvCacheTests`, `PagedKvCacheCodecTests` |
| Batched executor primitives | `BatchedExecutorTests`, including managed paged-attention correctness and multi-sequence logits routing; `RetainedFusedCacheTests` for capability-gated holder retention/re-keying and LRU cleanup |
| Per-model correctness | `Qwen35BatchedCorrectnessTests`, `Mistral3BatchedForwardTests`, `Gemma4BatchedForwardTests`, `GptOssBatchedCorrectnessTests`, `NemotronBatchedCorrectnessTests` |
| MTP speculative decoding | `SpeculativeExecutionTests` (draft/verify/rollback core), opt-in end-to-end `Qwen36SpeculativeTests` (`TS_MTP_E2E=1`) and `Gemma4SpeculativeTests` (`TS_GMTP_E2E=1`) with real GGUFs |
| Per-model performance probes | `Gemma4BatchedPerfBench`, `Qwen35BatchedPerfBench`, `GptOssBatchedPerfBench`, `NemotronBatchedPerfBench` |
| DiffusionGemma path | `DiffusionGemmaTests` for denoising, prompt-KV caching, and batched generation probes |
| End-to-end engine behavior | `EngineParallelInferenceTests` with opt-in real GGUFs via `TS_TEST_MODEL_DIR` |
| Server option translation | `ServerOptionsBuilderTests` for `--continuous-batching`, `--no-continuous-batching`, and paged-KV compatibility flags |

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `TS_SCHED_DISABLE_BATCHED` | `0` | `1` forces the per-sequence KV-swap fallback even when a model implements `IBatchedPagedModel`. |
| `TS_SCHED_MAX_BATCHED_TOKENS` | `4096` | Per-step token budget. |
| `TS_SCHED_MAX_RUNNING_SEQS` | `16` | Maximum in-flight sequences. |
| `TS_SCHED_PREFILL_CHUNK` | `256` | Per-request prefill cap while a decode is active. Prefill-only steps divide the full token budget fairly. Server flag: `--prefill-chunk-size N`. |
| `TS_SCHED_SOLO_PREFILL_CHUNK` | `8192` | Per-step prefill cap for a solo (uncontended) request — feeds the prompt through the fused whole-graph prefill path in big chunks. Bounded by `TS_SCHED_MAX_BATCHED_TOKENS`. |
| `TS_SCHED_NUM_BLOCKS` | `256` | Physical blocks in the engine pool. |
| `TS_SCHED_BLOCK_SIZE` | `256` | Tokens per block. |
| `TS_SCHED_PREFIX_CACHE` | `1` | Set `0` to disable block-hash prefix reuse. |
| `TS_SCHED_STOP_REPETITION` | `1` | Set `0` to let a looping generation run to its token limit rather than ending it with finish reason `repetition`. |
| `TS_SCHED_DECODE_QUANTUM` | `256` | Number of decode tokens before a sequence switch is allowed in fallback-heavy execution. |
| `TS_BATCHED_N1_FAST_PATH` | `1` | Solo single-sequence steps use the fused N=1 fast-path decode; set `0` to force those steps onto the fully-batched path (A/B testing). |
| `TS_PER_SEQ_FUSED` | `1` | Concurrent (N≥2) sequences on fused-capable models run per-request fused Forward; set `0` to force the op-by-op batched paged path (A/B testing). |
| `TS_BATCHED_FUSED_DECODE` | `1` | `0` disables true token-batched fused decode inside the per-sequence fused path (one graph decodes all N sequences). |
| `TS_RETAINED_FUSED_CACHE` | `1` | Retain finished request-owned fused holders for exact-prefix continuation on models that advertise support; `0` disables (VRAM cap / A/B). Supported holders include Gemma 4 K/V and Qwen 3.5/3.6 attention K/V plus GDN recurrent state. |
| `TS_RETAINED_FUSED_CACHE_MAX` | `4` | LRU budget of retained fused holders (each pins the model's complete per-request continuation state). |
| `TS_PREFIX_CHECKPOINTS` | `1` | Checkpoint the model's complete state at the end of the shared prompt prefix (the boundary the chat layer marks on the request) and start each new chat from a clone of it, on models that can copy their state (Gemma 4, Qwen 3.5/3.6). `0` disables. |
| `TS_PREFIX_CHECKPOINTS_MAX` | `2` | How many distinct shared prefixes stay checkpointed at once (LRU). |
| `TS_KV_INITIAL_TOKENS` | `0` | Tokens of K/V a cache is given when created, before any request declares a budget; `0` keeps the engine policy (the whole window when `MAX_CONTEXT` is explicit). The cache still grows on demand. |
| `TS_KV_GENERATION_RESERVE_MAX` | `0` | Cap on the generation share of a request's up-front K/V reservation (prompt + max_new_tokens); `0` = uncapped. Past the cap the cache grows on demand. |
| `TS_KV_HOLDER_POOL_MAX` | `64` | How many released per-request holders a model may park for reuse; each costs its whole K/V allocation while parked. |
| `TS_KV_PAGED_QUANT_BITS` | `0` | Optional TurboQuant codec bits for paged KV blocks (`2`, `4`, or `8`); recurrent-state models may fall back to passthrough. |
| `TS_MTP_SPEC` | `0` | `1` enables MTP / NextN speculative decoding for solo sequences (server `--spec`). |
| `TS_MTP_DRAFT` | `8` | Max tokens drafted per speculative step (server `--spec-draft`). |
| `TS_MTP_PMIN` | `0.15` | Min draft-head confidence to keep a drafted token (server `--spec-pmin`; `0` never gates). |
| `TS_MTP_DRAFT_MODEL` | none | Path to the separate Gemma 4 `gemma4-assistant` draft GGUF (server `--draft-model`); ignored by Qwen 3.6. |
| `TS_GMTP_NO_FUSED` / `TS_GMTP_NO_FAST_ROLLBACK` / `TS_GMTP_BATCHED_TRUNK` | off | Gemma 4 draft-path A/B switches (disable fused verify/draft kernels; restore kept-prefix rollback; run the batched trunk instead of the linear trunk). |
| `DIFFUSION_STEPS` | `48` | Web UI DiffusionGemma denoising steps per block. This is separate from autoregressive scheduler step budgets. |
| `DIFFUSION_MAX_BATCH` | `2` | Maximum active DiffusionGemma Web UI requests in the diffusion scheduler. |
| `DIFFUSION_BATCHED_FORWARD` | `0` | Enables true batched canvas decode for active DiffusionGemma canvases; the default favors the fused single-canvas path. |

A host can make a shared-prefix checkpoint outlive the process by attaching an
`IPrefixCheckpointStore` to the engine (`InferenceEngine.PrefixCheckpointStore`, or
`InferenceEngineHost.PrefixCheckpointStore` in TensorSharp.Chat). The executor then
reads a saved checkpoint at admission for a prompt whose shared prefix no in-memory
checkpoint covers, and writes one the moment it takes a checkpoint; the model family
owns the byte format (`IBatchedPagedModel.TryExportRetainedCache` /
`TryImportRetainedCache`, implemented by Qwen 3.5/3.6 and Gemma 4) and validates it
on the way in. TensorAgent's `PrefixCheckpointFileStore` is the file-backed store;
it is what makes the first message after a launch as fast as the second.


Server CLI aliases:

```bash
--continuous-batching      # default, sets TS_SCHED_DISABLE_BATCHED=0
--no-continuous-batching   # sets TS_SCHED_DISABLE_BATCHED=1
--paged-batching           # alias for --continuous-batching
--no-paged-batching        # alias for --no-continuous-batching
--prefill-chunk-size N     # sets TS_SCHED_PREFILL_CHUNK
```

The older `--paged-kv*` flags are retained for compatibility with the removed
standalone per-session paged-KV manager. Current server request KV state is
owned by `InferenceEngine`.

Separately from the paged TurboQuant codec (`TS_KV_PAGED_QUANT_BITS`), the KV
cache itself can be stored at reduced precision with
`--kv-cache-dtype <f32|f16|q8_0|q4_0>` (or the `KV_CACHE_DTYPE` env var) on
both the CLI and the server. The default is chosen automatically per model
(`f16` when the model's weights are below F32, `f32` otherwise); the block-quantized tiers
(`q8_0`, `q4_0`) require the native GGML flash-attention path, and `q4_0`
(~1/7 of f32) targets very long 128K–256K contexts where the KV cache
dominates memory.

## Remaining Work

- Build one native GGML graph for an entire attention batch instead of one
  small graph per sequence. This should reduce launch/compile overhead on many
  short sequences.
- Wire the device-resident paged K/V pool into a model. It is already built
  (`ggml_ops_paged_kv_pool.cpp` plus `DevicePagedKvCache`) and replaces the host
  gather with `ggml_set_rows` writes and on-device `ggml_get_rows` reads, but
  nothing allocates its paged K/V there, so the batched path keeps its
  host-resident cache and its measured ceiling. This is the prerequisite for
  concurrency buying any throughput at all.
- Complete Gemma 4 batched coverage for MoE variants, multimodal pending
  embeddings, and block-quantized KV cache.
- Decide whether DiffusionGemma scheduler metrics should be surfaced through
  `/api/queue/status` or a dedicated diffusion endpoint once operational usage
  needs per-batch visibility.
- Move prepared multimodal embedding lists out of model-level mutable state and
  into `SequenceState`, so multimodal prompt preparation can run fully in
  parallel instead of being serialized before submission.
- Remove queue-position compatibility chunks once clients no longer depend on
  the old fields.
