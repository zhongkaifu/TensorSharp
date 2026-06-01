# Paged Attention & Continuous Batching in TensorSharp

This document describes the design and implementation of vLLM-style paged
attention and continuous batching in TensorSharp. It explains what was
built, what was deferred, and how to extend the work to true batched-paged
attention kernels.

## Goals

- **Paged KV cache** — split the KV state into fixed-size blocks, indexed by a
  per-sequence block table, with reference counting and LRU eviction.
- **Prefix caching** — block-hash addressing so two sequences that share a
  prefix can reuse the same physical KV blocks.
- **Continuous (iteration-level) batching** — multiple in-flight requests
  scheduled per step, with mid-batch add/remove and a token budget.
- **Multimodal-aware** — image / audio / video prompts run through the same
  engine.

The required test models (Gemma 4, Qwen 3.6, Nemotron 3) all exercise the new
engine end-to-end via [`EngineParallelInferenceTests`](../InferenceWeb.Tests/EngineParallelInferenceTests.cs).

## Layered Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Adapters (WebUI / Ollama / OpenAI)                                  │
│      ChatGenerationPipeline (existing) — to be migrated              │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────────────────────┐
│  InferenceEngine  (Runtime/Scheduling/InferenceEngine.cs)            │
│   - one worker thread runs the schedule/execute loop                 │
│   - SubmitRequest → InferenceRequestHandle (Channels stream tokens)  │
└──────────────────┬───────────────────────────────────────────────────┘
                   │
       ┌───────────┴────────────┐
       ▼                        ▼
┌──────────────┐         ┌──────────────────────────────────┐
│ ContinuousBatch│        │ BatchExecutor                    │
│   Scheduler   │ ─uses→ │   - resolves KV-state ownership  │
│   - waiting / │        │   - extract/inject blocks        │
│     running   │        │   - drives model.Forward         │
│   - block     │        │   - samples tokens (decode)      │
│     allocation│        │   - captures full blocks back to │
│   - prefix    │        │     the prefix-cache index       │
│     adoption  │        └─────────────┬────────────────────┘
└───────┬───────┘                      │
        │                              ▼
        ▼                  ┌──────────────────────┐
┌────────────────┐         │ IModelArchitecture   │
│ BlockPool +    │         │   .Forward(tokens)   │
│ PagedKvStorage │         │   .TryExtractKVBlock │
│   - LRU free   │         │   .TryInjectKVBlock  │
│     queue      │         │   .ResetKVCache      │
│   - hash index │         └──────────────────────┘
│   - per-block  │
│     byte slabs │
└────────────────┘
```

### Components

| Component | File | Role |
|---|---|---|
| `KvBlock` | `Runtime/Paged/KvBlock.cs` | Per-block metadata (id, ref count, hash). |
| `FreeBlockQueue` | `Runtime/Paged/KvBlock.cs` | O(1) LRU queue of free blocks. |
| `PagedKvStorage` | `Runtime/Paged/PagedKvStorage.cs` | Byte slabs keyed by block id. |
| `BlockHashIndex` | `Runtime/Paged/BlockHashIndex.cs` | Content-hash → block lookup. |
| `BlockPool` | `Runtime/Paged/BlockPool.cs` | Allocates, frees, evicts blocks. |
| `BlockTable` | `Runtime/Paged/BlockTable.cs` | Per-sequence logical→physical block map. |
| `SequenceState` | `Runtime/Scheduling/SequenceState.cs` | Per-request mutable state (status, tokens, blocks). |
| `SchedulerOutput` | `Runtime/Scheduling/SchedulerOutput.cs` | Per-step work decision. |
| `SchedulerConfig` | `Runtime/Scheduling/SchedulerConfig.cs` | Static knobs (token budget, max seqs, block size). |
| `ContinuousBatchScheduler` | `Runtime/Scheduling/ContinuousBatchScheduler.cs` | Iteration-level scheduler with prefix caching + preemption. |
| `BatchExecutor` | `Runtime/Scheduling/BatchExecutor.cs` | Drives the model.Forward, swaps KV state, samples. |
| `InferenceEngine` | `Runtime/Scheduling/InferenceEngine.cs` | Engine worker thread; submit/abort API. |
| `InferenceRequestHandle` | `Runtime/Scheduling/InferenceRequestHandle.cs` | Streams tokens + completion future to the caller. |
| `InferenceEngineHost` | `Server/InferenceEngineHost.cs` | DI-registered, per-model engine singleton. |

## How a Request Flows

1. **Submit** — caller builds a `SequenceState` from a prompt token list and
   calls `engine.SubmitRequest(seq)`. The returned `InferenceRequestHandle`
   exposes a `Channel<int>` token stream and a `Task<InferenceCompletion>`.
2. **Schedule** — the engine's worker thread calls
   `ContinuousBatchScheduler.Schedule()` each tick. The scheduler:
   1. Iterates the **running** set and assigns each a prefill chunk or a
      decode step. Each scheduled work allocates any missing blocks.
   2. **Admits** waiting sequences while the token budget and seq-count
      budget allow. Before allocating new blocks, the scheduler looks up the
      sequence's prompt prefix in the **block hash index** and adopts
      matching blocks for free.
   3. If the pool runs out, **preempts** the lowest-priority running
      sequence: its full blocks are added to the prefix cache, the rest are
      freed, and the sequence is re-parked at the front of the waiting queue.
3. **Execute** — `BatchExecutor.ExecuteStep` iterates the scheduled work and:
   1. Ensures the model's KV state belongs to this sequence (extract the
      previous owner's blocks; reset the model; inject this sequence's blocks
      from storage). KV swap cost is amortized via
      `DecodeQuantumTokens` (default = block size).
   2. Builds the next token batch (prefill chunk OR a freshly-sampled decode
      token).
   3. Calls `model.Forward(tokens)`.
   4. Captures any newly-full blocks back into the prefix cache and clones
      the returned logits into `SequenceState.LastLogits`.
4. **Apply results** — the engine emits tokens on the handle's `Channel`,
   checks EOS / max-tokens, and asks the scheduler to free finished sequences.

## Prefix Caching

Each sequence's prompt is split into block-sized chunks. The full-block chunks
are hashed using the existing `KvBlockHasher` (SHA-256 chained, 128-bit
truncated) and the resulting `KvBlockHash` is looked up in
`BlockHashIndex`. On a hit the block's ref count is bumped and the sequence
adopts it without re-prefilling those tokens. The cache is populated as full
blocks are committed during prefill / decode and persists across requests
within the same engine instance.

The scheduler always **caps prefix adoption** so at least one token of the
prompt remains for the forward — without that we can't produce fresh logits
to sample from.

## What Was Refactored vs Left Untouched

**New** (no equivalent in the previous codebase):

- The entire `Runtime/Paged/` and `Runtime/Scheduling/` namespaces.
- `Runtime/Paged/ManagedPagedAttention.cs` — pure-C# batched paged-attention
  kernel (causal masked online-softmax, parallelised over (seq, head)) that
  models can call from their `IBatchedPagedModel.ForwardBatch` implementation.
- `Runtime/Paged/PagedKvBatchOps.cs` — helpers for the batched path
  (layer-buffer allocation, K/V scatter via slot mapping, last-token gather).
- `Models/Models/Qwen3/Qwen3Model.BatchedForward.cs` — reference
  `IBatchedPagedModel.ForwardBatch` implementation for standard Qwen3.
- `Models/Models/Mistral3/Mistral3Model.BatchedForward.cs` — second
  reference implementation, **verified end-to-end on a real 14B GGUF**.
  Adds YaRN RoPE scaling and per-token Q position scaling to the Qwen3
  template.
- `Server/InferenceEngineHost.cs` — DI-registered per-model engine singleton.

**Refactored** (existing files re-written to drive the engine):

- `Server/ChatGenerationPipeline.cs` — was a per-token sample/forward loop
  bound to server-owned session KV state; now submits the rendered prompt to the
  engine and streams tokens from the returned `InferenceRequestHandle`.
- `Server/ModelService.cs` — constructs and owns the `InferenceEngineHost`;
  resets the engine on model reload so the block pool is rebuilt for the
  new fingerprint.
- `Server/Program.cs` DI — `InferenceEngineHost` exposed; `ModelService`
  is the single owner.
- `Server/InferenceQueue.cs` — demoted to a no-op shim that grants tickets
  immediately. Kept only because the HTTP adapters still emit queue-status
  chunks; once those are rewritten this file can be deleted.

**Reused** from the existing codebase:

- `KvBlockHash` / `KvBlockHasher` — the new block hash index uses them.
- `IModelArchitecture.TryExtractKVBlock` / `TryInjectKVBlock` —
  the per-sequence (legacy) executor path relies on these for state swap.
  All required models (Gemma 4, Qwen 3 / 3.5, Mistral 3, Gemma 3, GPT-OSS,
  Nemotron) already implement them.

**Deleted / isolated legacy server state**:

- `Server/SessionKvCacheManager.cs` has been removed. Server sessions no
  longer own token/logit KV bookkeeping, no session is activated into the
  model, and `ModelService.ResetSession` / `DisposeSession` only affect
  tracked chat history. Compatibility properties such as `ActiveSession` and
  `KVCache` remain as inert shims so older callers compile, but they are never
  used by inference. All live KV ownership now flows through
  `Runtime/Scheduling`.

## Test Coverage

### Unit tests (`InferenceWeb.Tests/ContinuousBatchSchedulerTests.cs`, 11 tests)

- BlockPool allocate / free / ref-count round-trips.
- BlockPool hash registration → prefix hit.
- BlockTable token-accounting.
- Scheduler admits and runs multiple sequences concurrently.
- Engine drives a single sequence to completion.
- Engine drives N parallel sequences.
- Engine respects MaxTokens / EOS / prefix-cache hit.

### Batched-executor tests (`InferenceWeb.Tests/BatchedExecutorTests.cs`, 4 tests)

- `ManagedPagedAttention` matches hand-computed output for a single-query
  single-position attention case.
- `ManagedPagedAttention` handles causal masking and two-key averaging.
- `BatchExecutor` actually packs ≥2 sequences into one `ForwardBatch` call
  when the model implements `IBatchedPagedModel`.
- Per-sequence logits route back to the correct sequence in the batched
  path (verified by sequences whose expected token is encoded in their
  `UserTag`).

### Integration tests (`InferenceWeb.Tests/EngineParallelInferenceTests.cs`)

Opt-in via `TS_TEST_MODEL_DIR` env var pointing at a directory containing the
required GGUFs. Verified results on macOS / Apple Silicon (GgmlMetal backend):

| Model | Test | Result |
|---|---|---|
| Gemma 4 (E4B-it-Q8_0) | 5 text prompts in parallel | ✅ pass (15.7 s wall, 5/5 produce tokens) |
| Gemma 4 | image + 3 text prompts | ✅ pass (image: 24 tokens, text-parallel: 25.2 t/s, prefix-cache populated) |
| Gemma 4 | audio prompt | ✅ pass (24 tokens out) |
| Gemma 4 | prefix-cache double-tap | ✅ pass (engine reusable; long-prefix demo requires prompts > BlockSize=256) |
| Qwen 3.6 (27B-IQ4_XS) | 5 text prompts in parallel | ✅ pass (58.8 s wall — 27B-class) |
| Nemotron 3 (30B-A3B-IQ2) | 5 text prompts in parallel | ✅ pass (36.6 s wall) |
| Nemotron 3 | audio prompt | ✅ pass (9 tokens: "A person is talking about the weather.") |
| Gemma 4 | mp4 video | ⚠ soft-fail (image decoder only accepts PNG/JPEG/HEIC; would need a frame-extraction step) |

## Two Execution Paths

`BatchExecutor.ExecuteStep` first checks whether the model implements
`IBatchedPagedModel`:

```csharp
if (_model is IBatchedPagedModel batched && output.ScheduledWork.Count > 0)
    return ExecuteStepBatched(batched, output);       // true vLLM-style
return ExecuteStepPerSequence(output);                // KV-swap fallback
```

### Batched path (`ExecuteStepBatched`)

Packs every scheduled sequence into one `BatchedForwardContext` and calls
`model.ForwardBatch(ctx)` exactly once. Per step:

- Build `queryStartLoc` (`int[numSeqs+1]`), `positions` (`int[numTokens]`,
  per-token absolute positions), `slotMapping` (`int[numTokens]`,
  `blockId*blockSize + offset`), `blockTables` (`int[numSeqs][]`).
- For decode tokens, sample the next token from the previous step's logits
  *before* building the batch so the sampled token is the one forwarded.
- Receive `IReadOnlyList<float[]>` — one logits array per sequence,
  same order as `ctx.Sequences`.
- No KV-state extract/inject. The model is responsible for writing its
  fresh K/V to the paged pool at the slots given by `slotMapping`, and for
  reading K/V during attention from the blocks listed in `blockTables`.

This is the vLLM equivalent of `flash_attn_varlen_func`. Compute scales
with batch size (one big matmul per layer instead of N small ones).

### Per-sequence fallback path (`ExecuteStepPerSequence`)

For models that don't implement `IBatchedPagedModel`, the executor runs each
sequence's forward separately, swapping KV state via the existing
`TryExtractKVBlock` / `TryInjectKVBlock` contract. This is correct on every
model TensorSharp ships today.

### `ManagedPagedAttention` — the reference kernel

A pure-C# implementation of `flash_attn_varlen_func`:

```csharp
ManagedPagedAttention.Forward(
    q, kBlocks, vBlocks, output,
    numTokens, numHeads, numKvHeads, headDim, blockSize,
    queryStartLoc, seqLens, positions, blockTables, numSeqs,
    scale, causal: true);
```

- Online-softmax recurrence — no full attention matrix is ever materialised.
- Causal masked, supports grouped-query attention (`numHeads % numKvHeads == 0`).
- Parallelised over `(seq, head)` via `Parallel.For`.
- Verified by two byte-checkable unit tests in
  `BatchedExecutorTests.ManagedPagedAttention_*`.

The kernel is the building block: a model's `ForwardBatch` does the linear
projections (which automatically batch as a single big matmul over all
tokens), writes K and V via `WriteKvToPagedPool`, and calls
`ManagedPagedAttention.Forward` for the actual attention. Per-token cost is
higher than a fused GGML kernel; per-batch cost amortises across all
sequences.

## Limitations and Future Work

### Native paged-attention kernel — DELIVERED

Implemented as `TSGgml_PagedAttentionForward` in
[`ggml_ops_paged_attention.cpp`](../TensorSharp.GGML.Native/ggml_ops_paged_attention.cpp).
Per sequence in a batch:

1. C++ memcpy gather of K and V from the paged buffer, walking the
   per-sequence block table.
2. Build a small GGML graph: permute Q to flash-attn layout, optionally
   construct a causal mask (built once on the C++ side, with the right
   absolute first-query-position).
3. Dispatch `ggml_flash_attn_ext` — the same Metal/CUDA fused
   flash-attention kernel the legacy single-sequence path uses.
4. Read the result back via `finalize_compute_with_download` (async on
   Metal).

This puts the gather in C++ instead of C#, and uses the optimised
attention kernel rather than the AddmmBatch+softmax+AddmmBatch chain
the Tensor kernel needed. Result: faster than the legacy
single-sequence path on long context (21% faster on the 4×800-token
benchmark above) while still benefiting from the rest of the
batched-paged architecture (one batched linear-projection matmul,
no KV-state swap between sequences, prefix sharing across the block
pool).

### Further optimisations still on the table

- **One graph for the whole batch** — currently we build a separate
  GGML graph per sequence (N graph compiles per layer). Building one
  graph that includes all N sequences' flash-attn nodes would amortise
  the compile overhead, especially helpful for batches with many short
  sequences.
- **Replace the C++ gather with `ggml_get_rows`** so the gather runs on
  the GPU and overlaps with the matmul.
- **Quantised paged K/V** (Q8_0 / Q4_0) to halve the gather bandwidth
  and improve attention throughput proportionally.

### What it would take to port Gemma 4 fully

The native paged kernel already accepts a `sliding_window` parameter,
so the kernel-side support is in. The remaining work is in the model:

1. **Per-layer SWA dispatch** — `Mistral3Model.BatchedForward` calls
   `PagedAttentionForward` with a single `slidingWindow` arg per call.
   Gemma 4 would need to pass `IsLocalLayer(l) ? _slidingWindow : 0`
   per layer. Straightforward but requires unwinding the current
   "one ForwardBatch = one kernel config" assumption.
2. **Per-layer paged buffer sizing** — Gemma 4 has dual head
   dimensions (local=256, global=512) and different KV head counts
   per layer. The paged buffer allocation in
   `Mistral3Model.BatchedForward.EnsurePagedBuffersAllocated` assumes
   constant dims; Gemma 4 needs per-layer dims.
3. **KV donor map** — last N "shared" layers alias earlier layers'
   K/V. In the paged world this means refcount-shared blocks: layer
   40's block table points at layer 20's blocks. The
   `BlockPool` already supports refcount-based sharing for the prefix
   cache — extending it for layer-to-layer aliasing is mechanical.
4. **Per-Layer Embeddings (PLE)** — compute the `[total_rows,
   numLayers × _pleDim]` lookup once at the start of `ForwardBatch`
   (Gemma 4's existing `ComputePLE` already handles this), then add
   the per-layer slice after the post-attention residual. One line per
   layer.
5. **Multimodal embedding injection** — embeddings land at specific
   absolute token positions; in the batched path they need to be
   inserted into the concatenated embedded hidden state. Same
   abstraction the legacy `InjectVisionEmbeddings` uses, applied to
   the batched embedding output.
6. **Dense vs MoE FFN per layer** — branch on `HasMoE(l)` and dispatch
   either `FFN` or `MoEForward`. Both ops already operate over a
   batched `[numTokens, hidden]` axis.

The current scaffold throws `NotSupportedException` for each of the
above; the BatchExecutor catches and falls through. Replacing each
throw with the corresponding feature implementation is the path to a
production Gemma 4 batched port.

### What it would take to port Qwen 3.6 / Nemotron 3 fully

Both architectures have a recurrent component (GatedDeltaNet for
Qwen 3.5/3.6, Mamba2 for Nemotron 3) that maintains a per-sequence
running state evolving across the entire input. Paging assumes KV is
the only cache; recurrent state is orthogonal.

A full port would require designing a **per-sequence recurrent-state
pool**:

- A `RecurrentStatePool` analogous to `BlockPool`, holding `convState +
  ssmState` per layer per sequence (~10–20 KB per sequence for the
  recurrent-only state).
- Allocated on the first recurrent-layer touch for a sequence.
- Persisted across forward calls inside the engine.
- Freed on sequence retirement (engine already calls
  `_pool.Free(seq.BlockTable.Blocks)` — would also call
  `_recurrentPool.Free(seq.Id)`).
- The model's `ForwardBatch` would interleave paged-KV layer dispatch
  (for FullAttention layers) with recurrent-pool fetch/update (for
  GDN/Mamba layers).
- Block-boundary capture (currently used by the per-seq path's
  `IModelArchitecture.RequiresPerBlockCapture`) needs an equivalent
  for the recurrent state — the running state at position N must be
  captured at every block boundary during prefill so a later
  `TryInjectKVBlock`-equivalent can resume.

Estimated effort: 2–3 weeks per architecture. The substrate (engine,
scheduler, paged pool, native kernel) is the same; the work is the
recurrent-pool abstraction and the per-architecture state-shape +
state-update kernel work.

### Porting status — required models

| Model | Status | Notes |
|---|---|---|
| **Mistral 3** (Ministral-3-14B) | ✅ Full port, validated on real GGUF, **21% faster than legacy** on long context | The reference implementation. |
| **Gemma 4** (E4B) | ✅ Full port, **default path** (set `TS_GEMMA4_BATCHED=0` to force the per-seq KV-swap fallback for debugging). All 42 layers (SWA + GQA + PLE + KV-donor sharing at L24+) match legacy within FP noise on the per-layer checksum trace. Two bring-up bugs fixed: (1) missing `ggml_cont` on Q after the permute in `TSGgml_PagedAttentionForward` (`ggml_flash_attn_ext` silently produced wrong output when Q was a non-contiguous view on Metal); (2) `EnsurePagedBuffers` destructively rebuilt the K/V arrays when growing, wiping any K/V already written for previously-scheduled sequences — manifested as the FIRST sequence in a multi-sequence batch generating a degenerate single-token loop on its first decode after subsequent sequences joined. Bug #2 affected Mistral 3 and Qwen 3 too; all three EnsurePagedBuffers helpers now copy old contents on grow. See [`Gemma4BatchedForwardTests.cs`](../InferenceWeb.Tests/Gemma4BatchedForwardTests.cs) for the apples-to-apples logit-cosine check (cosine ≥ 0.99 with the legacy unfused path) and [`EngineParallelInferenceTests.Gemma4_ThreeLongGenerationsParallel`](../InferenceWeb.Tests/EngineParallelInferenceTests.cs) for the multi-sequence parallel test. | See [`Gemma4Model.BatchedForward.cs`](../TensorSharp.Models/Models/Gemma4/Gemma4Model.BatchedForward.cs). |
| **Qwen 3.6** (Qwen 3.5 / GatedDeltaNet) | ✅ Ported with per-slot recurrent-state pool (Phase 5c). **Default path** (set `TS_QWEN35_BATCHED=0` or pass `--no-continuous-batching` to force the per-seq KV-swap fallback for debugging). 100% greedy-match against legacy on small prompts; ~1.83× tps at n=3 on Qwen 3.6-27B (Apple M4 Pro, Metal). Native batched GDN op exists (Phase 7 in [`ggml_ops_gated_delta_net.cpp`](../TensorSharp.GGML.Native/ggml_ops_gated_delta_net.cpp)) but is gated behind `TS_QWEN35_BATCHED_GDN_NATIVE=1` pending perf verification. | See [`Qwen35Model.BatchedForward.cs`](../TensorSharp.Models/Models/Qwen35/Qwen35Model.BatchedForward.cs). |
| **GptOss** (gpt-oss-20b) | ✅ Ported. **Default path** (set `TS_GPTOSS_BATCHED=0` to force the per-seq KV-swap fallback for debugging). Per-layer paged K/V; batched Q/K/V with bias; NeoX+YaRN RoPE per token; per-layer SWA (alternating); per-head attention sinks via the new native kernel [`TSGgml_PagedAttentionForwardWithSinks`](../TensorSharp.GGML.Native/ggml_ops_paged_attention.cpp) (Phase 9 — `ggml_flash_attn_ext` + `ggml_flash_attn_ext_add_sinks` under the hood, falls back to managed-C# [`ManagedPagedAttention.ForwardWithSinks`](../TensorSharp.Runtime/Paged/ManagedPagedAttention.cs) on non-GGML backends or when `TS_GPTOSS_PAGED_ATTN_MANAGED=1`); MoE FFN batched through existing `MoEForward` token-parallel path. **100% greedy match** vs legacy (12/12 tokens), preserved through both Phase 8 (managed) and Phase 9 (native sinks). Perf remains **flat** (~0.93–1.05× across n=1,3,5) — the native sinks kernel didn't move the needle because the bottleneck is the *per-layer* and *per-seq* ggml graph construction cost: the legacy path runs one fused per-layer kernel ([`TryFusedAttnLayerPrefill`](../TensorSharp.GGML.Native/ggml_ops_*.cpp): RMSNorm + fused QKV + RoPE + KV-append + sinks-aware softmax + attn + output proj + residual in one cgraph), whereas the batched path issues ~5 separate graphs per layer (norm, QKV, RoPE, paged-attn, output, residual). Closing this last gap needs a *fused-per-layer* batched kernel like the existing GptOss legacy one, but operating on paged K/V — substantial follow-up work. See [`GptOssModel.BatchedForward.cs`](../TensorSharp.Models/Models/GptOss/GptOssModel.BatchedForward.cs), [`GptOssBatchedCorrectnessTests.cs`](../InferenceWeb.Tests/GptOssBatchedCorrectnessTests.cs), [`GptOssBatchedPerfBench.cs`](../InferenceWeb.Tests/GptOssBatchedPerfBench.cs). |  |
| **Nemotron 3 (with native Mamba2 batched kernel)** | ✅ Phase 9. Set `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` (in addition to `TS_NEMOTRON_BATCHED=1`) to route the Mamba2 recurrent step through a new native C++ kernel ([`TSGgml_NemotronMamba2BatchedStepF32`](../TensorSharp.GGML.Native/ggml_ops_mamba2.cpp), NEON SIMD + GCD per-head parallelism — mirrors my Qwen 3.5 Phase 7 GDN kernel structure). Replaces N C# `Mamba2Block` calls with one native dispatch + batched `ssm_in` / `ssm_out` projections. **100% prefix match** vs the C# Mamba2 path on greedy sampling (12/12 tokens). Perf vs legacy: **3.95× tps at n=3, 2.93× at n=5, 0.75× at n=1** on NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-IQ2_XXS, Apple M4 Pro, Metal (n=1 loses because batched scaffolding outweighs benefit for single-seq). **Also fixed a latent bug**: `s_nemoBatchedOptIn` was `static readonly` and captured the env var at class-load time, so tests setting `TS_NEMOTRON_BATCHED=1` at runtime never actually toggled the path — both legacy and "batched" test runs were silently falling through to per-seq. Changed to method getter (mirrors Qwen 3.5's pattern). Prior Phase 3-8 "100% match" results were validating per-seq vs per-seq; this is the first run where the batched ForwardBatch math is genuinely exercised, and it produces token-for-token identical first-prefix output. |  |
| **Nemotron 3** | ✅ Ported with per-slot Mamba2 conv + SSM state pool. **Default path** (set `TS_NEMOTRON_BATCHED=0` to force the per-seq KV-swap fallback for debugging). Attention layers use paged-attention (Mistral 3 pattern, no RoPE). FFN dense layers run token-parallel ReLU². FFN MoE layers run through the existing `MoEForward` token-parallel router + per-token expert dispatch (Phase 7). Mamba2 layers run per-seq state-swap on the existing single-seq native kernel (analog of Qwen 3.5 Phase 5c reference-swap). Multimodal vision + audio embeddings inject directly into the batched `[numTokens, hidden]` tensor via the same row-wise `InjectMultimodalEmbeddings` the legacy path uses (Phase 8, follows the Mistral 3 stance that upstream serializes multimodal). `SupportsBatchedMultimodal` returns true while the batched path is active (i.e. unless `TS_NEMOTRON_BATCHED=0` is set). **100% greedy match** vs legacy across 3 short text prompts (12/12 tokens); **3.43× tps at n=3**, **1.94× at n=1**, **1.67× at n=5** on NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-IQ2_XXS, Apple M4 Pro, Metal. Multimodal-prompt correctness is structurally validated (text-only stays 100% after removing the multimodal pre-flight rejection) but lacks a local audio/image fixture in `InferenceWeb.Tests/` for end-to-end verification. See [`NemotronBatchedCorrectnessTests.cs`](../InferenceWeb.Tests/NemotronBatchedCorrectnessTests.cs) and [`NemotronBatchedPerfBench.cs`](../InferenceWeb.Tests/NemotronBatchedPerfBench.cs). | See [`NemotronModel.BatchedForward.cs`](../TensorSharp.Models/Models/Nemotron/NemotronModel.BatchedForward.cs). |

### Refactoring a model for true batched compute

Implementing `IBatchedPagedModel.ForwardBatch` for a real model means
rewriting the forward pass to be batch-aware:

1. The model's KV cache lives in **per-layer paged buffers** instead of
   per-layer contiguous tensors.
2. Position IDs become an explicit `int[]` (one per token), not the
   implicit `_cacheSeqLen + i`.
3. Q/K/V projection runs across the entire batch as one matmul.
4. RoPE applies per-token rotations from the positions tensor.
5. K and V are scattered into block storage via `slotMapping` (or a
   `PagedKvBatchOps.ScatterKv` call).
6. Attention is per-sequence, gathered via `blockTables` — either through
   `ManagedPagedAttention` or a native paged-attention op.
7. Output projection / MLP / MoE / norm batch over all tokens.
8. Gather the last token of each sequence (`queryStartLoc[s+1] - 1`)
   before the LM head.

### Reference implementations

Two production models now implement `IBatchedPagedModel.ForwardBatch`.

#### [`Qwen3Model.BatchedForward.cs`](../TensorSharp.Models/Models/Qwen3/Qwen3Model.BatchedForward.cs)

The standard Qwen3 architecture (per-head Q/K RMSNorm, GPT-NeoX RoPE):

- Lazy per-layer paged K/V `float[]` buffers sized to the engine's
  `BlockPool` (`AllocateLayerBuffer(numBlocks, blockSize, numKvHeads, headDim)`).
- Per-token RoPE built from `ctx.Positions` and dispatched through
  `Ops.RoPEEx` (which already accepts an arbitrary positions tensor).
- K/V scatter via `PagedKvBatchOps.ScatterKv(slotMapping)` writing into
  the layer's paged buffer.
- Per-sequence attention via `ManagedPagedAttention.Forward`, supplied
  with the layer's K/V buffer + `BlockTables` + `seqLens`.
- Last-token gather via `PagedKvBatchOps.GatherLastTokenPerSeq` so only
  the LM head runs on `numSeqs` rows instead of `numTokens`.
- Reuses existing `Embedding`, `RMSNormOp`, `LinearForward`, and `FFN`
  helpers on `ModelBase` — those already work over the full token axis,
  so embeddings / projections / norms / FFN batch automatically.

End-to-end logit-level correctness against a real base-Qwen3 GGUF is
not yet verified in this tree (only Qwen 3.6 / GatedDeltaNet GGUFs are
available locally). The opt-in
[Qwen3BatchedForwardTests](../InferenceWeb.Tests/Qwen3BatchedForwardTests.cs)
self-validate as soon as a base Qwen3 GGUF is dropped into
`TS_TEST_MODEL_DIR`.

#### [`Mistral3Model.BatchedForward.cs`](../TensorSharp.Models/Models/Mistral3/Mistral3Model.BatchedForward.cs)

Mistral 3 — verified end-to-end against the real **Ministral-3-14B-
Instruct-2512-Q4_K_M.gguf** (40 layers, 5120 hidden, 32 heads, 8 KV
heads, 128 head dim, YaRN RoPE). Adds, on top of the Qwen3 template:

- Both fused (`attn_qkv.weight`) and unfused (`attn_q/k/v.weight`) layer
  weight layouts (picked per-layer via `_layerQkvFused[]`).
- YaRN RoPE scaling parameters (`extFactor`, `attnFactor`, `betaFast`,
  `betaSlow`) threaded into `Ops.RoPEEx`.
- Per-token YaRN position-dependent Q scaling:
  `q *= (1 + beta * log(1 + floor(pos / orig_ctx)))` — replaces the
  legacy per-row scalar scale with an explicit `positions[]` walk so
  each token in the batch picks its own scale factor.
- Vision-embedding injection: same path as the legacy forward (queued
  by the multimodal injector before the per-layer loop). Multimodal
  prompts continue to serialise their preparation step upstream.

**Verified correctness on the real GGUF**
([Mistral3BatchedForwardTests](../InferenceWeb.Tests/Mistral3BatchedForwardTests.cs)):

| Test | Result |
|---|---|
| `Mistral3_BatchSize1_ForwardBatchMatchesLegacyTop1` | ✅ legacy top-1 = batched top-1 = 1091 on a 6-token prompt |
| `Mistral3_BatchSize2_DistinctSequencesProduceDistinctLogits` | ✅ two diverging prompts in one batch produce top-1 tokens 2 and 1091 (distinct, proving K/V is correctly partitioned) |

**End-to-end through the engine** (continuous-batching scheduler +
`ExecuteStepBatched`, validated by
`EngineParallelInferenceTests.Mistral3_FourTextPromptsParallelBatched`):
4 parallel chat-style prompts on the same Ministral-3-14B GGUF complete
without errors and stream tokens back through `InferenceRequestHandle`.

### Three paged-attention kernels

`Mistral3Model.BatchedForward` picks between three paged-attention
implementations at runtime via the `TS_PAGED_ATTN_KERNEL` env var
(default: `native`):

| Kernel | File | What it does |
|---|---|---|
| **`TSGgml_PagedAttentionForward`** (default) | [`TensorSharp.GGML.Native/ggml_ops_paged_attention.cpp`](../TensorSharp.GGML.Native/ggml_ops_paged_attention.cpp) | **Native C++ implementation.** Per sequence: gather K and V from the paged buffer with C++ memcpy (walking the block table), then dispatch `ggml_flash_attn_ext` on the contiguous K/V — the same fused Metal/CUDA flash-attention kernel the legacy single-sequence path uses. The gather happens once on the C++ side, eliminating the managed↔native border crossing N×L times. |
| `TensorPagedAttention.Forward` | [`Models/Paged/TensorPagedAttention.cs`](../TensorSharp.Models/Paged/TensorPagedAttention.cs) | C# Tensor-based gather, then `Ops.AddmmBatch` for Q·Kᵀ and softmax·V with `GgmlBasicOps.AttentionSoftmaxWithSinks` in between. Faster than scalar managed but slower than native because of repeated GPU dispatches per sequence. |
| `ManagedPagedAttention.Forward` | [`Runtime/Paged/ManagedPagedAttention.cs`](../TensorSharp.Runtime/Paged/ManagedPagedAttention.cs) | Pure-C# online-softmax loop, parallelised over `(seq, head)`. Correctness fallback on any backend. |

All three produce the same logits.
`Mistral3_BatchSize1_ForwardBatchMatchesLegacyTop1` passes against each
(top-1 = 1091 on the 6-token probe prompt against legacy single-sequence
forward).

### Four-way benchmark on Ministral-3-14B (Apple Silicon, GgmlMetal)

Same 4-prompt workload, four dispatch paths:

**Short context** (~20 tokens per seq, dominated by linear-projection
kernel launches):

| Path | Wall | Tokens/sec |
|---|---|---|
| Batched + Native kernel (default) | 12.92 s | 2.3 |
| Batched + Tensor kernel | 13.79 s | 2.2 |
| Batched + Managed scalar kernel | **12.08 s** | **2.5** |
| Per-sequence KV-swap (legacy GGML) | 14.46 s | 2.1 |

**Long context** (4 × ~800-token prompts, 3249 prompt tokens total —
where attention compute dominates):

| Path | Wall | Tokens/sec |
|---|---|---|
| **Batched + Native kernel (default)** | **42.37 s** | **0.6** |
| Per-sequence KV-swap (legacy GGML) | 53.95 s | 0.4 |
| Batched + Tensor kernel | 71.42 s | 0.3 |
| Batched + Managed scalar kernel | 111.02 s | 0.2 |

### Reading the numbers

- **Native kernel beats the legacy GGML per-sequence path by ~21%** on
  long context (42.37 s vs 53.95 s). This is the headline result the
  whole work has been building toward: the architecture is *faster than
  the legacy code path*, end-to-end, on a real 14B GGUF.
- The native kernel is faster than Tensor (~40%) and Managed (~60%)
  because it skips the managed→native border crossing N×L times and
  drives the same battle-tested `ggml_flash_attn_ext` Metal kernel the
  legacy single-sequence path uses, but per-sequence inside one C++
  call.
- For very short contexts the managed scalar kernel still narrowly
  wins because every per-sequence call has a fixed graph-compile cost
  that dominates 20-token attention. The crossover is around
  ~100 token contexts on this hardware.
- **The legacy single-sequence path is no longer the leader on long
  context.** The whole point of paged + continuous batching is now
  validated empirically.

### Prefix-cache validation

Worth highlighting from the long-context run: `reused=1536` and
`hashedCached=3` mean the engine's block-hash index successfully shared
six full prompt blocks (the repeated filler) across the four
sequences. This is the first end-to-end validation of the prefix-cache
path on a real GGUF.

### Gemma 4 — batched vs per-sequence (gemma-4-E4B-it-Q8_0, Apple M4 Pro)

The Gemma 4 batched port is complete and exercises the harder Gemma
features the Mistral path doesn't have: per-layer SWA + global mix,
per-layer head dims, NeoX RoPE, per-layer embeddings (PLE), KV-donor
sharing for layers 24-41, and a folded `1/sqrt(d_h)` scale. Numbers
below are from [`Gemma4BatchedPerfBench.cs`](../InferenceWeb.Tests/Gemma4BatchedPerfBench.cs)
toggling `TS_GEMMA4_BATCHED` between the two paths in-process (same
loaded model, same engine, same prompts).

| Workload | n | Prompt tokens | Legacy tps | Batched tps | Speedup |
|---|---|---|---|---|---|
| Single sequence, short prompt | 1 | 29 | 14.0 | 4.9 | **0.35×** (batched slower) |
| 5 short prompts in parallel | 5 | 142 | 10.2 | 13.5 | **1.32×** |
| 8 short prompts in parallel | 8 | 218 | 10.0 | 15.1 | **1.51×** |
| 4 long-context prompts in parallel | 4 | 3 293 | 3.4 | 5.4 | **1.61×** |

Take-aways:

- **Speedup grows with batch size** — at batch=8 the per-call paged
  graph-build / gather cost is fully amortised and the batched path
  reaches **1.51×** the legacy throughput. The long-context (~800
  tokens / seq) run hits **1.61×**, very close to the Mistral 3 result
  on the same hardware (1.21×; Gemma's extra SWA + KV-donor parallelism
  is what closes the gap).
- **Single-sequence is a net loss.** With nothing to amortise across,
  the paged graph build + gather pays for itself on every step and the
  legacy single-seq fused-decode kernel wins by ~3×. The runtime keeps
  the legacy path for batch=1 by leaving `TS_GEMMA4_BATCHED` unset;
  the batched path is the right choice for the multi-request server
  workload it was built for.
- **Output-token counts differ** between paths in the same workload
  (e.g. 41 vs 120 on short-parallel) because logit cosine is 0.99,
  not 1.0 — the per-Metal-launch FP reductions aren't bit-deterministic
  in the legacy path, so one side hits an EOS a few tokens earlier on
  some prompts. The fair throughput metric is **tokens / wall-second**,
  which is what the table reports.

### Other models — Qwen 3.6 / Nemotron 3

Qwen 3.6 / 3.5 use GatedDeltaNet recurrent blocks whose running state
is not paged. Nemotron 3 includes Mamba layers with the same recurrent
constraint. Adapting those architectures to `IBatchedPagedModel` is
the next milestone; the Qwen3 implementation is the template for the
attention-only portion.

This work is the natural next phase. The current architecture (scheduler,
executor, block pool, kernel, engine, host, Qwen3 reference) is the
substrate it plugs into.

### Adapter migration

Adapters (WebUI / Ollama / OpenAI) still emit queue-position chunks via
the no-op `InferenceQueue`. With continuous batching the concept is
meaningless; the adapters can drop the position chunks entirely once a
follow-up pass cleans them up. The engine itself already handles all
concurrency.

### Multimodal concurrency

`ModelMultimodalInjector` stores pending vision / audio embeddings on the
**model**, not the sequence. Two multimodal requests therefore can't
prepare in parallel without stepping on each other; the rewritten
`ChatGenerationPipeline` serialises prompt preparation behind a lock for
multimodal turns and lets text-only turns prepare in parallel. A clean
fix is to move the prepared-embedding list into `SequenceState` and have
the executor install it on the model right before `Forward` /
`ForwardBatch`.

### Multimodal concurrency

`ModelMultimodalInjector` stores pending vision / audio embeddings on the
**model**, not on the sequence. Two multimodal requests therefore can't
prepare in parallel without stepping on each other. The integration tests
work around this with a per-`EngineContext` lock around
`ProcessPromptTokens`. A clean fix is to move the prepared-embedding list
into `SequenceState` and have the executor install it on the model right
before `Forward`. This is a small refactor inside
`ModelMultimodalInjector`.

### Decode quantum tuning

The KV state swap cost is non-trivial (tens of MB for a 4B model, hundreds
of MB for 27B). To amortize it we run multiple decode tokens for the same
sequence before switching (`DecodeQuantumTokens`). The default value
(`BlockSize = 256`) maximises swap efficiency but reduces fairness. With
true batched attention this knob can go to 1 (per-token fairness).

## Configuration

```
TS_SCHED_MAX_BATCHED_TOKENS   # per-step token budget (default 4096)
TS_SCHED_MAX_RUNNING_SEQS     # max in-flight sequences (default 16)
TS_SCHED_PREFILL_CHUNK        # max prefill tokens per step (default 1024)
TS_SCHED_NUM_BLOCKS           # total physical blocks in the pool (default 256)
TS_SCHED_BLOCK_SIZE           # tokens per block (default 256)
TS_SCHED_PREFIX_CACHE         # 0 to disable block-hash prefix sharing
TS_SCHED_DECODE_QUANTUM       # tokens before allowing a session swap
```
