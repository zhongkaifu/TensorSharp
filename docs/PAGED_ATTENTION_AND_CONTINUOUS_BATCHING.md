# Paged Attention & Continuous Batching in TensorSharp

[English](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md) | [中文](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)

This document is the current implementation reference for TensorSharp's
vLLM-style paged KV cache, its Radix prefix cache, and iteration-level continuous
batching. The server routes inference through this engine.

Read it as an implementation reference, not a universal performance claim. The
generic paged K/V pool is host-resident and remains a bottleneck, while supported
model/backend pairs can instead use device-resident, token-batched fused decode.
Throughput therefore depends on the selected and accepted execution path. See
[Measured Concurrency Behavior](#measured-concurrency-behavior).

## Current Status

| Area | Status |
|---|---|
| Server engine | `ModelService` (TensorSharp.Chat, shared by the server and TensorAgent) owns one `InferenceEngineHost` per loaded model. `ChatGenerationPipeline` submits rendered prompts to the engine and streams tokens from `InferenceRequestHandle`. |
| Scheduler | `ContinuousBatchScheduler` admits waiting requests, preempts running work when block pressure requires it, applies a per-step token budget, and adopts each admitted prompt's reusable prefix from the prefix cache. |
| KV storage | `BlockPool`, `BlockTable` and `PagedKvStorage` hold fixed-size physical blocks with ref counts and LRU free ordering. The block bytes live in **managed host memory**. |
| Prefix cache | The Radix tree (`PrefixTree`, driven by `PrefixCacheCoordinator`) is the default in the server, CLI generation and TensorAgent: one index for reusable pages, retained conversation states and shared-prefix checkpoints. `TS_SCHED_PREFIX_CACHE=0` or `--no-prefix-cache` turns reuse off. See [Radix prefix cache](#radix-prefix-cache). |
| Batched execution | Models that implement `IBatchedPagedModel.ForwardBatch` pack all scheduled sequences into one model call with explicit `positions`, `slotMapping`, `queryStartLoc`, and per-sequence block tables. |
| Fallback execution | Path selection is centralized in `ExecutionPlanner`: model+backend capabilities (`ExecutionCapabilities`), operator overrides (`ExecutionOptions`), and per-step request features produce an `ExecutionPlan` (selected path, fallback chain, rejection reasons). A model may still decline a specific batch with `NotSupportedException`; the step then falls to the plan's next candidate, ending in the per-sequence KV-swap path. |
| Native attention | `TSGgml_PagedAttentionForward` gathers paged K/V in C++ and dispatches `ggml_flash_attn_ext`. |
| Speculative decoding | Optional speculation for solo (non-concurrent) sequences, and for concurrent ones on the shared decoder runtime (Qwen 3.5 family, Qwen 3.8 Flash Next and the other families it serves): learned draft heads (embedded NextN / MTP, Gemma 4's separate `gemma4-assistant` GGUF, Qwen 3.8 Flash Next's shared MTP head), block drafters (DSpark, DFlash) and the weight-free n-gram speculator. `BatchExecutor` drives the shared `SpeculativeExecution` draft / verify / rollback core for models that implement `ISpeculativeTarget`. Off by default; `--spec` (CLI and server) or `--draft-model`. See [Speculative decoding (MTP / NextN)](#speculative-decoding-mtp--nextn). |
| Throughput under concurrency | The generic host-resident `BatchedPaged` route does not itself scale aggregate decode; one Gemma 4 measurement saturated near **69 tok/s**. On a model that implements `TryForwardBatchedFusedDecode`, the per-sequence-fused route first attempts one device graph for the active decode subset and falls back per request only when that batch is ineligible or declined. See [Measured Concurrency Behavior](#measured-concurrency-behavior). |
| Queue API | `/api/queue/status` reports the engine's live load: requests processing, requests waiting for a batch slot, and requests completed. Nothing serializes requests outside the engine. |
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
  (including Gemma 4's kernel with per-sequence cache capacities). The shared
  decoder runtime's families (the Qwen 3.5 / 3.6 / 3.8 family and Qwen 3.8 Flash
  Next among them) take the unified step instead: every running request's rows in
  one forward over its own slot. If the model declines, the same step falls back
  to N isolated fused forwards; the server logs the first successful batch and the
  reason for the first decline.
- **What does work.** Iteration-level scheduling, prefix sharing (the Radix tree
  by default), preemption, per-sequence native slots, per-request fused holders, and
  capability-gated token-batched fused decode. A decline preserves correctness
  and fairness but may return throughput to the round-robin ceiling.

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
        |      - prefix adoption (Radix tree)
        |
        +--> BatchExecutor
               - runs the path ExecutionPlanner selected
               - per-sequence fused, batched ForwardBatch,
                 or the per-sequence KV-swap fallback
               - samples decode tokens
               - captures newly full blocks
        |
        v
BlockPool + PagedKvStorage   (managed host memory)
```

### Core Components

| Component | File | Role |
|---|---|---|
| `KvBlock` | `TensorSharp.Runtime/Paged/KvBlock.cs` | Physical block metadata and ref counts. |
| `BlockPool` | `TensorSharp.Runtime/Paged/BlockPool.cs` | Allocates, frees, ref-counts, and evicts blocks. |
| `BlockTable` | `TensorSharp.Runtime/Paged/BlockTable.cs` | Maps each sequence's logical block ids to physical block ids. |
| `PagedKvStorage` | `TensorSharp.Runtime/Paged/PagedKvStorage.cs` | Byte slabs keyed by physical block id. |
| `PrefixTree` / `PrefixCacheCoordinator` | `TensorSharp.Runtime/Scheduling/PrefixCache/` | The Radix prefix cache: radix keys over prompt tokens and media identities, scopes, budgets, and the payloads each node can resume from. |
| `IPrefixCacheModel` | `TensorSharp.Runtime/Scheduling/PrefixCache/IPrefixCacheModel.cs` | Per-family contract: the model declares its reuse capabilities and implements only the state operations (materialize, capture, release). |
| `PrefixCheckpointFileStore` | `TensorSharp.Runtime/Scheduling/PrefixCheckpointFileStore.cs` | File-backed `IPrefixCheckpointStore` that keeps shared-prefix checkpoints across launches (server and TensorAgent). |
| `PagedKvBatchOps` | `TensorSharp.Runtime/Paged/PagedKvBatchOps.cs` | Batched K/V scatter and last-token gather helpers. |
| `ManagedPagedAttention` | `TensorSharp.Runtime/Paged/ManagedPagedAttention.cs` | Pure C# correctness fallback for paged attention. |
| `SequenceState` | `TensorSharp.Runtime/Scheduling/SequenceState.cs` | Mutable per-request status, tokens, blocks, logits, and sampling state. |
| `ContinuousBatchScheduler` | `TensorSharp.Runtime/Scheduling/ContinuousBatchScheduler.cs` | Iteration-level scheduler with prefix caching and preemption. |
| `ISoloPrefillBudgetModel` | `TensorSharp.Runtime/Scheduling/ISoloPrefillBudgetModel.cs` | Optional model contract: sizes a lone request's prefill steps (Muse Spark with pipelined prefill), unless `TS_SCHED_SOLO_PREFILL_CHUNK` is set. |
| `BatchExecutor` | `TensorSharp.Runtime/Scheduling/BatchExecutor.cs` | Executes the planned step, samples, and captures KV blocks. |
| `ExecutionPlanner` | `TensorSharp.Runtime/Scheduling/ExecutionPlanner.cs` | Pure-function path selection: capabilities + options + step features → `ExecutionPlan`. |
| `ExecutionCapabilities` | `TensorSharp.Runtime/Scheduling/ExecutionCapabilities.cs` | Declared capability snapshot of the loaded model × backend combination. |
| `ExecutionOptions` | `TensorSharp.Runtime/Scheduling/ExecutionOptions.cs` | Structured snapshot of the executor-level `TS_*` overrides (single place that reads them). |
| `InferenceEngine` | `TensorSharp.Runtime/Scheduling/InferenceEngine.cs` | Worker loop and public submit/abort surface. |
| `InferenceEngineHost` | `TensorSharp.Chat/InferenceEngineHost.cs` | Per-model engine host owned by `ModelService`; used by the server and TensorAgent. |

## Request Flow

1. A protocol adapter builds a normalized chat request.
2. `ChatGenerationPipeline` renders the prompt, resolves sampling options, and prepares any image/audio/video embeddings.
3. The pipeline creates a `SequenceState` and calls `InferenceEngine.SubmitRequest`.
4. The engine worker asks `ContinuousBatchScheduler` for the next step.
5. The scheduler admits waiting sequences while token and sequence budgets allow. Before allocating new blocks, it asks the Radix prefix cache for the prompt's longest reusable prefix (pages, a retained conversation state or a shared-prefix checkpoint) and adopts it.
   A waiting request is admitted only when the free pool can hold its whole prompt on top of the prompt blocks the running requests still have to allocate; otherwise it stays queued until one finishes (a lone request is always admitted).
6. If the pool is under pressure (decode growth is not reserved), the scheduler can preempt a running sequence that ranks below the one needing blocks (lower priority, or submitted later), commit its full blocks, free the remainder, and requeue it. A sequence never preempts an older one: it waits a step instead, so a full pool drains oldest-first rather than livelocking long prefills against each other.
7. `BatchExecutor` executes the scheduled step. It asks `ExecutionPlanner` for the step's `ExecutionPlan` and runs the first candidate path that accepts the step (see [Execution Planning](#execution-planning-capability-model)).
8. The engine emits sampled tokens to the request handle, checks EOS / max-tokens / abort state and the repetition guard, and releases blocks for completed sequences.

The repetition guard (`RepetitionGuard`) ends a generation that has locked into an exact loop, with finish
reason `repetition`. It is asked after every emitted token, on the output tokens only, and it has two rules:

- **A short unit:** up to 64 tokens, repeated at least 8 times over at least 128 tokens. Examples are
  `","+` 43 times, or 128 copies of one token.
- **A long block:** 65 to 4,096 tokens, with the last `max(4096, 4 × period)` tokens an exact copy of the
  block over and over. That is 18 copies of a 222-token block, and at least four of any block. This
  rule runs only once the output is 4,096 tokens long, and then every 64 tokens. Each check reads a fixed
  window at the end of the output, so its cost does not grow with the reply.

Both rules need copies that are identical token for token. Output that repeats its structure but changes
something in each copy is never stopped: CSV rows, a table, getters, a list, or sections under the same
headings. One looping round of a tool-call body (an `apply_patch` file, say) is stopped the same way, and
the skills loop tells the model what repeated and lets it try once more. `TS_SCHED_STOP_REPETITION=0`
turns the guard off for the host. A request that sets `ignore_eos` runs to its token limit with the guard
off (`SamplingConfig.StopRepetition = false`).

Prefix adoption is capped so at least one prompt token still runs through the
model. That keeps logits fresh for sampling even when the entire visible prefix
is already present in the prefix cache.

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
  `SupportsLinearKVMigration`, …) and the speculative-decoding interfaces
  (`HasDraftHead`, `SpeculationProfitable`, `SupportsBatchedSpecTrunk`).
  `ExecutionCapabilities.FromModel` snapshots them per step. A model's static
  limitation (Gemma 4 with MoE layers or a block-quantized KV cache, tensor
  parallelism) surfaces through `BatchedForwardAvailable=false` so the planner
  routes around the batched path up front; `ForwardBatch` throwing
  `NotSupportedException` remains only as a per-batch decline, not the routing
  mechanism. `--no-continuous-batching` (`TS_SCHED_DISABLE_BATCHED=1`) switches
  every batched route off.
- **Plan candidates are ordered and safe.** Declinable candidates
  (`BatchedPaged` migration/refusal)
  fall through to the next entry; every plan ends in a path that cannot
  decline. `ExecutionPlannerTests` sweeps the capability/feature space to
  assert this invariant.
- **Observability.** `InferenceEngine` logs a one-time capability report at
  startup (which paths are statically available and why the others are not).
  `BatchExecutor` logs the plan — selected path, fallback chain, rejection
  reasons — whenever the decision changes (e.g. a concurrency transition),
  so "why did this request not take the fast path?" is a logged fact.
- **Path kinds** (`ExecutionPathKind`): `SpeculativePerSequence`,
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
N >= 2 instead, so `BatchedPaged` serves models without one — or all of them under
`TS_PER_SEQ_FUSED=0`, which keeps their K/V in the shared host block pool instead of
per-request device holders. Most current batched ports use native paged attention
for GGML backends:

| Kernel | Scope | Notes |
|---|---|---|
| `TSGgml_PagedAttentionForward` | Standard causal / sliding-window attention | C++ K/V gather plus `ggml_flash_attn_ext`. Default for Mistral 3 and most paged attention layers on GGML backends. A head size the backend has no flash kernel for (ggml-cuda: anything outside 40/64/72/80/96/112/128/256 and the grouped-query 192/320/512/576) runs as explicit attention with a one-time warning instead of aborting in `fattn.cu`. |
| `ManagedPagedAttention.Forward` | Pure C# fallback | Online-softmax implementation used for correctness and unsupported backend fallback. |

Mistral 3 and Hunyuan Dense run the native kernel on GGML backends and the managed one elsewhere.
GPT OSS does not use these kernels: on the shared decoder runtime its sinks are an operand of each
step graph's attention (see [the GPT OSS card](models/gptoss.md#41-attention)).

### Output identity under concurrency

The same batches must give the same logits, bit for bit. Different batches may not.
Which requests share a step, and how many tokens each forwards, decides which
kernels run: ggml-cuda serves a quantized MXFP4 `mul_mat_id` with MMVQ for up to 7
tokens on Turing and Ampere (8 on Volta, Ada and Blackwell) and with MMQ above that,
and a sequence that arrives
alone takes the solo fused path first. Those kernels agree only up to floating-point
near-ties, and greedy decoding turns a near-tie into a different continuation. A
concurrent round whose arrival order is not fixed can therefore produce different
tokens from run to run without any defect. Two runs that schedule the same batches
cannot.

To tell the two apart:

- `TS_CB_DEBUG=1` prints one `[cb] step#N <path>` line per engine step with every
  scheduled request (`id:P|D fwd=<tokens> computed=<tokens after the step>`) and a
  fingerprint of the logits it left: the top two tokens, their margin and a hash of
  the whole row. Diff two runs step by step. A different composition before the
  first differing hash is the near-tie class; the same composition with a different
  hash is a bug.
- `AgentTurnBench --conc-gate` holds the engine's compute gate until a whole
  concurrent round is queued, so every run admits the round in the same batches.
  Its rows record `ArrivalOrderFixed: true`, and `compare.py` requires identical
  tokens for them. Concurrent rows without it report token differences as
  informational unless `--require-concurrent-identity` is passed.

**Measured (2026-09-17, gpt-oss-20b MXFP4, 1x A40, `ggml_cuda`, `TS_PER_SEQ_FUSED=0`,
`AgentTurnBench --conc 1,4,8`).** Four concurrent greedy requests diverged at output
tokens 3-26 between runs, including between the warm-up pass and the measured pass of
one process; in those runs eight did not. The step trace showed the same batches in both runs and,
at the first decode step, logits several units apart (41.28 against 38.84 for the
same token) with every margin still wide. That was a defect, not a near-tie. The
standalone MoE kernel (`TSGgml_MoEFFNPrefillSwiGLUQuantF32`, which the batched paged
path of the since-removed legacy GPT OSS model used for its experts) uploaded its
per-expert biases as graph leafs in the reusable compute buffer. The allocator freed the gate bias after its `add_id` and
placed the SwiGLU activation on top of it. ggml-cuda fuses `{mul_mat_id, add_id,
mul_mat_id, add_id, swiglu_oai}` into one MMVQ kernel for 1-7 tokens. That kernel
reads the biases while it writes the activation, and its overlap check skips leafs
because llama.cpp's biases are weights. So the kernel overwrote the bias it was still
reading. On this Ampere card eight tokens take MMQ, which does not fuse, which is why
the 8-request round stayed stable; on Ada or Blackwell eight tokens still take MMVQ and
would have been hit too. The builder now pins every small uploaded parameter (ids, routing
weights, biases, post-norm weight) with the allocator's output flag. The CTest
`moe-fused-bias-alias-cuda` (`GgmlOpsMoeFusedBiasAliasTest`) compares the kernel
against an exact host evaluation for 1, 4 and 7 tokens. Before the fix the 4- and
7-token cases were off by up to 644 and 1118 against tolerances of 21 and 25 and were
not repeatable (the 1-token case passed); after it they match the CPU backend.
`moe-fused-bias-alias-metal` runs the same check on Metal, which does not fuse this
chain and passed before the fix too.

After the fix, with `--conc-gate`, three passes gave bit-identical logits at every
step of the 1-, 4- and 8-request rounds. Without the gate, one pass in three still
changed the 8-request round. Its first request had been scheduled alone on the solo
fused path before the other seven arrived. The first logits already differed at
that step (42.94 against 42.86), and the argmax flips came later, at margins of
0.011-0.11. That is the near-tie class, and `compare.py` reports it without failing.

### Per-Sequence Fallback

The fallback path still runs inside `InferenceEngine`; it is no longer the
server's outer concurrency primitive. It temporarily installs one sequence's
K/V state into the legacy model cache, calls `model.Forward(tokens)`, captures
full blocks, and moves to the next scheduled sequence. This keeps older or
feature-limited paths correct while they are being ported to true batched
compute.

### Speculative decoding (MTP / NextN)

When speculation is on — `--spec` on the CLI or the server (env `TS_SPEC=1`), or
`--draft-model`, which enables it by itself unless `--no-spec`
is given — `BatchExecutor` runs the shared `SpeculativeExecution` draft / verify /
rollback core for **solo (non-concurrent)** sequences on models that implement
`ISpeculativeTarget`, over the model's linear cache. The flow per step:

1. **Draft.** The speculator `--spec-type` selects (`auto`, the default, uses
   whatever drafter the checkpoint carries and declines when there is none;
   `ngram` needs no drafter) proposes up to `--spec-draft` tokens
   (`TS_SPEC_DRAFT`; default `8`). Unless the window is set
   explicitly, a model's preferred window replaces it: 3 on the Qwen 3.5-family
   recurrent trunks, GLM-5.3-Flash and Qwen 3.8 Flash Next, 7 on Gemma 4 on the
   ggml backends (shorter for IQ4_XS weights on Metal). N-gram has its own
   preference: on `ggml_metal` a dense (non-MoE) Qwen 3.5-family recurrent trunk
   whose matrices are mostly IQ4_XS, not under TP, widens it to 12 (a 13-row
   verify). `--spec-pmin`
   (`TS_SPEC_PMIN`) is a confidence gate whose default is
   per algorithm: `0.15` for a per-token head, `0.35` for a block drafter, `0` for
   n-gram.
2. **Verify.** The trunk verifies all drafted tokens in a single batched forward.
   The request's own sampler — temperature, top-k/p, and repetition/presence/frequency
   penalties — draws every row, and drafts are accepted while the drawn token
   matches. Every emitted token is drawn from a trunk row, so speculation changes
   how many forward passes a stream takes, not which tokens it may contain. A
   multi-row verify and a one-row decode are different kernels, though, so greedy
   output equals standard decode except at near-ties in the logits (see
   [What greedy parity delivers](speculative_decoding.md#what-greedy-parity-delivers)).
3. **Rollback.** On partial acceptance, KV (and any recurrent state) past the
   accepted prefix is rolled back before the next step.

Three algorithms share the `SpeculativeExecution` core; the design is in
[Speculative Decoding in TensorSharp](speculative_decoding.md):

| Algorithm | Drafter |
|---|---|
| `draft-head` | A per-token head. Embedded NextN / MTP in the trunk GGUF (`{arch}.nextn_predict_layers`: Qwen 3.6, Qwen 3.8 27B, GLM 5.2, GLM-5.3); Gemma 4's separate EAGLE-style `gemma4-assistant` GGUF on `--draft-model`, whose draft layers attend the **target's** last local / global KV (no draft K/V of its own); Qwen 3.8 Flash Next's shared MTP head GGUF on `--draft-model`, which keeps its own K/V and so speculates only for a request that prefilled from position 0. |
| `block` | A block drafter on `--draft-model`: DeepSeek V4 DSpark, and DFlash / DFlash2 for Muse-Glimmer and the Qwen 3.5 family. DeepSeek V4.1 DSpark is experimental: a `deepseek41-dspark` drafter on `ggml_cuda` / `ggml_cpu`; initial text/image HTTP probes with trained weights passed using two-GPU layer split on `ggml_cuda`; broad quality and throughput remain unqualified. |
| `ngram` | No weights: suffix matching over the sequence's own tokens. |

On rejection a Qwen 3.5-family or Qwen 3.8 Flash Next trunk (the shared decoder
runtime) commits the GatedDeltaNet state after the accepted rows without a second
forward (the GGML graphs keep a state per verify row; the per-op path keeps the
window's inputs and replays the accepted prefix); Gemma 4
rewinds its attention-KV position and puts back the sliding-window rows the verify
overwrote (its drafter is stateless given `(token, h)`); GLM-5.3-Flash restores a
snapshot of its KDA state (a device copy on the native executor) and re-forwards
the accepted prefix.

Speculation engages only where the model declares it profitable
(`SpeculationProfitable`), and each model decides: Qwen 3.5/3.6/3.8 and GLM 5.2 /
GLM-5.3 on every backend; Gemma 4 on the ggml backends (`ggml_cpu` included) and
the pure-C# `cuda` backend; Qwen 3.8 Flash Next on every backend it loads on;
GLM-5.3-Flash where its KDA rollback is available; Muse Spark on the GGML
backends (n-gram only), except under its native `--tp`; Muse-Glimmer on every backend; DeepSeek V4 / V4.1 only with their drafter loaded; Nemotron-H on every backend (n-gram drafts or the Nemotron 3.5 MTP head); GPT OSS on every backend (n-gram drafts or the gpt-oss-20b DFlash drafter). Mistral 3 and Hunyuan Dense implement no speculative trunk and
serve standard decode for every algorithm, n-gram included. Outside the shared
decoder runtime, concurrent batches never speculate — when more than one sequence
is running, every sequence uses the normal batched/fallback step; the runtime's
unified step verifies every speculating request's window in one forward. A `--draft-model` that cannot be attached (a
mismatched or incomplete drafter, or a model that refuses speculation) fails fast
at server startup (`SpeculationStartupValidation`). `--spec` leaves the Radix
prefix cache on; a Qwen 3.5-family NextN head restarts its private cache after a
reused prefix (see
[Arming after a reused KV prefix](speculative_decoding.md#arming-after-a-reused-kv-prefix)).

## Model Status

`--no-continuous-batching` (`TS_SCHED_DISABLE_BATCHED=1`) forces every model onto
the per-sequence path.

| Model family | Batched / paged status | Sub-toggles |
|---|---|---|
| Mistral 3 | Default `ForwardBatch` path. Uses paged K/V, YaRN-aware positions, native paged attention, and vision embedding injection after prompt preparation. Validated on Ministral-3-14B; long-context native paged attention is about 21% faster than the per-sequence GGML path. | `TS_PAGED_ATTN_KERNEL` selects `native`, `tensor`, or `managed`. |
| Gemma 4 | The shared decoder runtime's batched steps (`Gemma4RuntimeModel`): every checkpoint (E2B/E4B with per-layer embeddings and KV-shared layers, 12B/31B, the 26B-A4B MoE), media prompts, block-quantized caches and speculation (the `gemma4-assistant` head via `--draft-model`, n-gram, DFlash). | `--no-continuous-batching` and the runtime's own switches. |
| Qwen 3.5 / 3.6 / 3.8 family | The shared decoder runtime (`Qwen35RuntimeModel`). Every request has its own slot: attention K/V plus the GatedDeltaNet recurrent states. The running requests' decode and verify steps run as one graph (the paged batched forward on GGML, per-slot programs elsewhere). Finished conversations and shared-prefix checkpoints are retained by the radix prefix cache, exact prefix only, because the recurrence cannot rewind. Speculation uses the embedded NextN/MTP block, DFlash / DFlash2 drafters or n-gram. Every backend, dense NVFP4 checkpoints on `--backend cuda` included, runs this runtime. | `TS_RETAINED_FUSED_CACHE_MAX=0` turns retention off; `--spec` enables speculation. |
| GPT OSS | The shared decoder runtime (`GptOssRuntimeModel`). Every request has its own slot of device-resident K/V; each even layer's 128-token sliding window lives in a 768-cell ring. The running requests' prompt chunks and decode rows run as one graph on every backend, also under `--tp` and `--layer-split`. Finished conversations and checkpoints are retained through the runtime's holder adapter (captures, clones, donated holders, export/import); K/V block snapshots are not offered, because a ring is not position-addressed. Speculation uses n-gram drafts or the gpt-oss-20b DFlash drafter (`--draft-model`). On one A40 the server's aggregate decode for 8 concurrent requests on `ggml_cuda` is 230.4 tok/s, against 151.0 tok/s on the removed legacy model ([card](models/gptoss.md#11-validation-of-the-migration)). | `--no-continuous-batching` / `TS_SCHED_DISABLE_BATCHED` as for every runtime family. |
| Nemotron-H | The shared decoder runtime (`NemotronRuntimeModel`). Every request has its own slot: its Mamba-2 conv window and SSM state, plus paged K/V for the attention layers. The running requests' steps run as one graph on every backend, also under `--tp` and `--layer-split`. Finished conversations and checkpoints are retained through the runtime's holder adapter; a cached prefix carries the Mamba-2 states at its checkpoints, so a continuation restores K/V rows and recurrent state together. Image and audio embeddings enter as queued media rows. Speculation uses n-gram drafts or the Nemotron 3.5 MTP head. On one A40 the server finished 8 concurrent Nemotron 3.5 Lightning prompts in 2.4 s, against 33.3 s on the removed legacy model ([card](models/nemotron.md#11-validation-of-the-migration)). | `--no-continuous-batching` / `TS_SCHED_DISABLE_BATCHED` as for every runtime family. |
| GLM 5.x | No `ForwardBatch`: MLA keeps one compressed row per token and the DSA indexer scores against that same contiguous history, so there is no paged-KV layout to batch over. Concurrency runs on native **sequence slots** instead (`TSGgml_GlmSlotAlloc` / `SetActiveSlot` / `SlotFree`) — binding a request switches the active slot without moving KV bytes, and each slot's graphs are cached and captured independently. A default-on batched fused decode (one graph, one token per sequence, weights read once for the batch) is implemented on top: 1.81x aggregate decode at 4 concurrent requests. Batching changes GEMM shapes, and a 2-bit MoE can amplify that into different expert picks. | `TS_BATCHED_FUSED_DECODE=0` disables the batched decode. |
| Hunyuan Dense | Default `ForwardBatch` path on F32 paged buffers. A block-quantized (`q8_0` / `q4_0`) KV cache keeps the KV-snapshot swap path, which serves those dtypes exactly. | — |
| Muse-Glimmer | On the shared decoder runtime: the unified step batches concurrent requests in per-request slots (the 39 sliding-window layers keep 2560-cell rings), also under `--tp` and `--layer-split`. A Radix holder family (end states, checkpoints, checkpoint files). DFlash with `--draft-model`, n-gram without a drafter. | `--no-continuous-batching` / `TS_SCHED_DISABLE_BATCHED` as for every runtime family. |
| Muse Spark | On `ggml_cuda` / `ggml_vulkan` the native whole-model executor serves concurrency on its own **sequence slots** (per-slot F16 K/V on each layer's GPU, or split by KV group across the GPUs under `--tp N` on `ggml_cuda`; a ring for the 46 sliding-window layers): decode steps of every running request share one graph with per-sequence masks, text prompts that arrive together are prefilled in one graph, and image embeddings replace their placeholder rows in the graph. A finished slot is retained and donated to the request that continues the conversation (it may be rewound first), and the template preamble and shared system prompt are checkpointed into a slot cloned into new conversations. When its layer split pipelines prefill across CUDA GPUs, a lone prompt prefills in steps the model chooses (8,192 tokens; see `TS_SCHED_SOLO_PREFILL_CHUNK` below). The per-op path (`TS_MUSE_SPARK_NATIVE=0`, and `--tp` on `ggml_vulkan`) is a Radix page family; its packed paged `ForwardBatch` needs F32 KV and text-only requests. | `TS_MUSE_SPARK_NATIVE=0` selects the per-op path; `TS_MUSE_SPARK_PLAN_SLOTS` reserves slots in the placement plan; `TS_MUSE_SPARK_PIPELINE=0` turns pipelined prefill off (see [the Muse Spark card](models/muse-spark.md#options)). |
| DeepSeek V4 / V4.1 | No `ForwardBatch`: the compressed attention caches have no paged layout. Concurrency runs on the native executor's sequence slots (the pure-C# `cpu` and direct-CUDA `cuda` executors stay serial). A default-on token-batched fused decode reads the weights once per step for up to 16 sequences, in windows beyond that; it is declined while a DSpark drafter is loaded. V4.1 retains a finished conversation's slot for its next turn (not while a DSpark drafter is loaded). | `TS_BATCHED_FUSED_DECODE=0`; `TS_DSV41_RETAINED_CACHE_MB` (default 2048) budgets V4.1's retained slots. |
| Qwen 3.8 Flash Next (`qwen4exp`) | The shared decoder runtime (`Qwen4ExpRuntimeModel`), with the same slots, batched steps and retention as Qwen 3.5, plus the family's hyper-connection streams, PLE n-gram rows and QSA. `--layer-split N` places whole layers, and `--tp N` runs per-rank graphs on GGML and on `cuda`. The shared MTP head or a DFlash2 drafter (`--draft-model`) speculates. `--backend cuda` runs this runtime too, its graphs on the CUDA graph executor. | `TS_RETAINED_FUSED_CACHE_MAX=0` turns retention off; `TS_MOE_HOT_TIER_GB` sizes the accelerator tier of host-served experts. |
| DiffusionGemma | Separate text-diffusion path on the shared decoder runtime. `Forward(int[] tokens)` is intentionally unsupported; generation iteratively denoises fixed-length canvas blocks, each step one graph over the prompt's cache. Web UI requests share `DiffusionBatchScheduler`, which admits concurrent requests between blocks. | `DIFFUSION_STEPS`, `DIFFUSION_MAX_BATCH`. |

### Radix prefix cache

Cross-request prompt reuse is owned by one index, the Radix tree (`PrefixTree` in
`TensorSharp.Runtime/Scheduling/PrefixCache/`, driven by `PrefixCacheCoordinator` on
the engine thread). It serves the server, CLI generation and TensorAgent. A request's radix key is its prompt tokens, with
each media span keyed by its content identity, and each node records what state a
request can resume from there:

- **Pages**: host-slab KV snapshots or the model's own paged blocks, for the
  families without continuation holders (Mistral 3,
  Hunyuan Dense, Muse-Glimmer).
- **End states**: a model-owned continuation state. Gemma 4, the Qwen 3.5 / 3.6 / 3.8
  family (`qwen35`, `qwen35moe`, `qwen3next`), Qwen 3.8 Flash Next, Nemotron-H
  (whose states include the Mamba-2 conv and SSM states) and GPT OSS can copy theirs, so a finished conversation is retained for its next turn and the state at
  the end of the shared public prefix is checkpointed and cloned into each new chat.
  They do so only where they run per-request state: Gemma 4 on the GGML backends
  with its fused holders, and the Qwen 3.5 family, Flash Next and Nemotron-H on the shared
  decoder runtime's slots wherever their recurrent state can be copied (not a
  multi-node pipeline); none of them with `TS_PER_SEQ_FUSED=0`. GPT OSS does so on the
  runtime's slots on every backend. Elsewhere only the
  primary cache is continued.
  GLM 5.x (native executor) and DeepSeek V4.1 cannot copy a native slot: they can
  only hand one over whole to the turn that continues it (unless
  `TS_RETAINED_FUSED_CACHE_MAX=0`; V4.1 not while a DSpark drafter is loaded),
  and take no shared-prefix checkpoint.

The budgets reuse the existing knobs: `TS_PREFIX_CHECKPOINTS_MAX` bounds the public
checkpoints and `TS_RETAINED_FUSED_CACHE_MAX` the retained per-conversation end
states (unset, the scheduler's running-request limit and at least 4, so every
request that runs in parallel keeps its conversation's state; memory, not this
count, bounds the bytes), and the family byte cap (`TS_DSV41_RETAINED_CACHE_MB`)
still applies. Reuse needs identical rendered tokens
(tool schemas, chat template and thinking setting included), stops at each model's
resumable boundaries and at explicit cache breakpoints, and always leaves at least
one prompt token to compute. The conversation-scope rules below apply unchanged, and
so does media identity (spans keyed by content; a reuse length never cuts a span).
Whether reuse continues past a media span comes from each family's prefix-cache
capabilities: it does on Gemma 4, the Qwen 3.5 / 3.6 / 3.8 family, Qwen 3.8 Flash
Next and GPT OSS; on the other tree families (Mistral 3, Nemotron-H, Muse-Glimmer,
Hunyuan Dense, GLM 5.x, DeepSeek V4 / V4.1) reuse stops at the
first image, video frame or audio clip. DiffusionGemma and the image/video models
do not use it.

On a Mac the bytes are bounded by what macOS can give. The GPU and everything else
share one memory there, and a retained state is wired like the weights, so a spare
figure measured after it was allocated never drops below what is cached: a
Qwen3.8-27B MLX server on a 48 GB Mac kept sixteen conversations sized for the whole
context, passed 34 GB and decoded 3x slower once it swapped. The model reports how
far what macOS can give without compressing (free and purgeable pages and the clean
file cache) has fallen below the larger of 7 GB and 15% of RAM, and the prefix cache
evicts that many bytes of retained state, oldest first and short of the public
prefixes, before it admits a request and after it keeps one
(`IPrefixCacheModel.QueryHostMemoryDeficitBytes`). The host spare that clones and
holder pools are sized against is bounded the same way. Elsewhere nothing changes.

That figure cannot see a full GPU working set when the weights are mapped: their clean
file pages count as available. Qwen 3.8 Flash Next on `ggml_metal` therefore also
reports what the working set lacks for one more request's state beside its headroom,
after the idle graph arenas are released, and the deficit is the larger of the two.
Under Strata's concurrency rounds a request was refused a holder for 42 s while three
finished conversations stayed retained, then ran on the primary cache, which has no
admission check, and the GPU ran out of memory; now the retained holders go first.

`TS_SCHED_PREFIX_CACHE=0` disables reuse; `--no-prefix-cache` (CLI and server)
sets it and also skips the startup warm-up of the shared system/tool prefix and, on
the server, the checkpoint files. `--spec` leaves the prefix cache on. (The pre-Radix
block-hash sharing and its separate reuse paths were removed; `TS_PREFIX_CACHE_MODE`
is rejected.)

### Retained end states

A model may be unable to reconstruct its complete continuation state from
byte-level paged snapshots but still own a self-contained per-request fused holder.
When the model advertises `SupportsRetainedFusedCache`, a cleanly finished holder
becomes an end state of its conversation in the Radix tree, re-bound when a later
request of that conversation extends the recorded token prefix. Gemma 4 retains its
circular attention K/V; Qwen 3.5/3.6/3.8 and Qwen 3.8 Flash Next retain the
request's slot (attention K/V and the recurrent states); DeepSeek V4.1 retains
a native slot (not while a DSpark drafter is loaded). `TS_RETAINED_FUSED_CACHE_MAX=0`
turns retention off, including GLM 5.x's donated native slot (native executor, one
slot). A request of a scoped conversation that finished on the primary (N=1) cache
is kept the same way when a fused step takes the model over, so a conversation does
not lose its state because another chat arrived between two of its turns.

On `--backend cuda` with one GPU, the shared decoder runtime (every family on it: Qwen
3.5/3.6/3.8, Qwen 3.8 Flash Next, Mistral 3, Hunyuan Dense) accounts its device
memory itself, counting the large blocks the CUDA allocator caches as free (it gives
them back on demand). A finished request's slot is retained as it is (donated, no
copy): the conversation's next turn runs on it and replays the decode steps and
verify windows its earlier turn recorded. Only a short slot (up to 512 MiB of K/V) is
copied, into a pooled slot that already holds it, so the next request of another
conversation runs on the finished request's slot and its recordings. Before it
allocates a request's slot, copies one (a prompt-end checkpoint, a clone at admission)
or runs a lone prompt's step, it makes room in a fixed order, the cheapest loss first:
the pool's spare slots, then parked decode recordings, then the small verify-window
buffers retained slots keep (where they alone cover the shortfall; that conversation's
next turn records its windows again), then the finished conversations the prefix cache
keeps, oldest first, and for a prompt step the cached graphs last. A conversation released
this way prefills its next turn again, and one stderr line says how many went and
for what. A new request whose slot does not fit waits while other requests run;
alone, it is refused with the numbers before any of its prompt runs, instead of
failing part way with CUDA error 2. The prefix cache also evicts its oldest payloads
whenever free memory falls below the device's headroom (the larger of 1 GB and 1/32
of the device), and the CUDA allocator asks the model to release finished
conversations before its last resort (a full GC and a drain of every cached block).

There a request's slot is sized to the device's room, not held at its reservation
(prompt + max_new_tokens + verify window). It is made at the reservation's capacity
class with an eighth more (classes are whole 1,024s up to 4,096 cells, then 1.25, 1.5,
1.75 and 2 times a power of two) where that fits without releasing a finished
conversation, else at the largest whole 1,024 that fits; finished conversations are
released only for the cells the step needs (the prompt and its first token). A slot
that needs more grows one cache tensor at a time, so a growth needs the added cells
and one tensor beside the slot, not a second copy of it, and a conversation's next
turn usually fits the class its slot already has. A spare slot is no larger than the
largest short slot a request ran in (one whose copy is retained, up to 512 MiB of K/V),
so long conversations leave no multi-gigabyte spares, and a finished request's slot is
pooled only while the device keeps as much again free beside its headroom and a prompt
step. A request's tokens do not depend on the capacity its slot gets.

### Prompt reuse across requests: conversation scopes and media identity

Every cross-request reuse - the live primary cache, retained end states, the
shared-prefix checkpoint and pages - honours two rules.

**Conversation scope.** Each `SequenceState` carries a `CacheScope` (an opaque
hash) and its public boundary `SharedPrefixTokens` (the leading system/developer
messages plus tool declarations). State produced by another scope is reused only up
to that public prefix: through the shared-prefix checkpoint, which is cloned, or by
rewinding the live cache to exactly that prefix (the only public reuse a model without
checkpoints has, e.g. DeepSeek V4.1; the new request's prefill would overwrite that cache
anyway, and a checkpoint is preferred where one exists). Another conversation's retained
holder is never adopted, rewound into or moved away from its owner, its live cache is
never continued past the public prefix, and pages past the public prefix belong to
their scope (the Radix tree keeps the scope on its nodes). A scoped request never clones a
checkpoint longer than its own public prefix. The scope comes from the chat layer:

| Request | Scope |
|---|---|
| Web UI / TensorAgent with a `sessionId` | the session and its new-chat epoch (`newChat:true` starts a new one); a host that binds sessions to its saved conversations (`WebUiChatService.BindSessionConversation`, which TensorAgent calls for every session it opens) uses the conversation instead, so reopening a chat continues its own cached state |
| OpenAI Chat / Responses, Ollama chat, Web UI without a `sessionId` | the conversation the request's history proves it continues: its last assistant message is a turn this server generated and sent (see below), and sent to that conversation only; otherwise (including when two conversations were sent the same turn after the same history, such as a greedy reply to a common opening) a fresh scope |
| Skills / code tool-loop rounds | the scope of the client turn that started the loop |
| CLI | one scope per conversation; `/new` and `/reset` start a new one, and each JSONL conversation has its own |
| Engine callers that set no scope (benchmarks) | a fresh scope per request, so only the declared public prefix is shared |

The chat layer's raw-token splice follows the same identity. Each generated turn is
recorded under the content-hash chain of the client-visible history that preceded
it (roles, content, tool calls, media and attached files by content), with the raw output tokens AND
what the client was sent for them (the parsed content and tool calls, or the raw
text). A later assistant message is rendered from the recorded tokens only when it
equals that emitted form, ignoring whitespace; an assistant message a client wrote or
edited renders from its own text. Before this, the stateless APIs shared one tracked
history and spliced another client's generated turn over a client's own message.
Concurrent conversations no longer overwrite each other's records.

For a stateless request this is proof by content, and it has a residual. A request
that reproduces ANY earlier generated turn of a conversation - not only its latest -
continues that conversation's scope, including state its later turns left behind, and
a deterministic (greedy) reply can be reproduced outside this server. Such a request
reuses only tokens it sent itself, but `cached_tokens` then reflects how far its
prompt matches that conversation's later turns: whole pages on a page family, the
last few tokens of a holder on Gemma 4, further on models with exact native
rewinds (DeepSeek V4.1). Clients that need strict isolation use a Web UI / TensorAgent
`sessionId`; the stateless APIs have no per-request cache key yet.

**Media identity.** Each image, video frame (pair) and audio clip is identified by the
SHA-256 of its bytes. Base64 attachments (OpenAI `image_url`, Responses
`input_image`, Ollama `images`, audio) are stored as `<sha256>.<ext>` and written
once, so a client resending the same picture every turn keeps one file. The vision
and audio embedding cache is keyed by that content id, bounded by
`TS_MM_EMBEDDING_CACHE_MB` and evicted least-recently-used, never while a prepared
prompt still references an entry. A request carries its media as positional spans
(`SequenceState.MediaSpans`); a cached prefix is reusable when every span inside it is
the same content at the same place, and a reuse length is clamped to the start of any
span it would cut. Text before the first image is therefore always reusable: the radix
key carries each span's content id at its position, so two prompts that differ only in
a picture share everything before it.

A model whose cache cannot be continued past media exactly declares
`SupportsReuseAcrossMediaSpan = false`, and reuse then stops at the first media
span. The Radix tree takes the rule from each family's prefix-cache capabilities,
which continue past a media span only on Gemma 4, the Qwen 3.5/3.6/3.8 family, Qwen 3.8 Flash Next and GPT OSS
(see [Radix prefix cache](#radix-prefix-cache)). Gemma 4 uses absolute positions. Qwen 3.5/3.6's
M-RoPE prompt positions compress after an image, and every token past the position table
- decode, speculative verify, a text continuation - rotates at its KV index plus the
sequence's M-RoPE delta, which every holder, checkpoint and checkpoint file (format
version 2) stores; follow-up turns therefore continue the cache past the image and match
a re-prefill up to the backend's decode-versus-prefill kernel differences (see [the Qwen 3.5 card](models/qwen35.md#positions-after-an-image-the-m-rope-delta):
on Metal the Web UI turns after an image reuse 98% of the prompt and reach the first token
in 0.13 s instead of about 1.1 s). Until that fix Qwen 3.5/3.6 declared `false`, because
decode ran at the absolute index.

Prefilling an image *after* a reused prefix is a separate question. A model that cannot
do it exactly returns false from `IModelArchitecture.CanPrefillMediaAfterReusedPrefix`, and
such a turn then reuses nothing past its public prefix. The public prefix itself is still
cloned from the shared-prefix checkpoint: while checkpoints are in use every prefill is cut
at that boundary, so the media runs after it with or without reuse; a turn with no public
prefix prefills from zero. No shipped model returns false. Gemma 4 did past its sliding
window until its fused prefill applied the image's bidirectional mask at any start position
and its per-op path mapped the mask onto a wrapped window; an image turn now reuses the
conversation's text and prefills the image on the fused graph (E4B/Metal: a 457-token image
turn reusing 179 tokens reaches its first token in 0.57 s, against 1.25 s on the per-op
path and 0.64 s cold; an 889-token one past the window reuses 611 tokens, 0.62 s against
0.85 to 0.90 s without reuse). See the [Gemma 4 card](models/gemma4.md#image-and-audio-turns-after-a-reused-prefix).

On Gemma 4 the live cache is continued for turns of `MaxReusablePrefixTokens` (the
sliding window) tokens or fewer too. Rewinds on a wrapped ring are still refused.

The admission log names what served each request - `radix pages`, `radix end state`,
`radix batched end state` or `radix primary cache` - with token counts, and when a
request reused nothing, the first specific reason a source declined.

## Test Coverage

| Area | Tests |
|---|---|
| Scheduler / block pool | `ContinuousBatchSchedulerTests`, `PagedKvCacheTests`, `PagedKvCacheCodecTests` |
| Radix prefix cache | `InferenceWeb.Tests/PrefixCache/`: `PrefixTree*Tests`, `RadixKeyTests`, `ResumabilityRulesTests`, `PrefixCacheContractConformanceTests` and `PrefixCacheFamilyCoverageTests` (per-family contracts), `RadixPagedEngineTests` / `RadixHolderEngineTests` (engine integration), `TreeTraceHarnessTests` (`Category=PrefixCacheProperty`); `PrefixCacheStartupTests`; model-gated `Qwen35MtpPrefixCacheTests` (NextN speculation after a reused prefix) |
| Batched executor primitives | `BatchedExecutorTests`, including managed paged-attention correctness and multi-sequence logits routing; `InjectShortfallAccountingTests` (a page inject that stops early is a miss, never a partial state) |
| Cross-request isolation and media identity | `ModelServiceRawTokenHistoryTests` and `ToolTranscriptSpliceTests` (content-verified raw-token splice), `ContentAddressedMediaTests`; `Gemma4MediaAfterReusedPrefixExactnessTests` (model-gated: an image or audio turn after a reused prefix against a cold prefill) and `Gemma4SoftTokenMaskTests` |
| Reuse past media (Qwen 3.5 M-RoPE) | `Qwen35MRopeReferencePositionTests` (positions against an SGLang `get_rope_index` fixture), opt-in `Qwen35ImageFollowUpExactnessTests` (reuse vs cold after an image with real weights, solo and concurrent, checkpoint file round trip) |
| Per-model correctness | `Qwen35BatchedCorrectnessTests`, `Mistral3BatchedForwardTests`, `Gemma4BatchedForwardTests`, `NemotronBatchedCorrectnessTests`, `NemotronRuntimeTests`, `GptOssRuntimeTests` |
| Batched MoE kernel under backend fusion | Native CTest `moe-fused-bias-alias-cpu` / `moe-fused-bias-alias-cuda` / `moe-fused-bias-alias-metal` (`GgmlOpsMoeFusedBiasAliasTest`): the standalone MoE FFN kernel with per-expert biases against an exact host evaluation, 1, 4 and 7 tokens, repeated |
| Speculative decoding | `SpeculativeExecutionTests` (draft/verify/rollback core), `SpeculatorRegistryTests`, `NGramSpeculatorTests`, `SpeculationCostGovernorTests`, `NemotronRuntimeTests` and `GptOssRuntimeTests` (`SpeculativeDecodingMatchesPlainGreedy`), opt-in end-to-end `Qwen36SpeculativeTests` (`TS_TEST_QWEN36_SPEC=1`) and `Gemma4SpeculativeTests` (`TS_GMTP_E2E=1`) with real GGUFs |
| Per-model performance probes | `Gemma4BatchedPerfBench`, `Qwen35BatchedPerfBench`, `NemotronBatchedPerfBench` |
| DiffusionGemma path | `DiffusionGemmaTests` for denoising, prompt-KV caching, and batched generation probes |
| End-to-end engine behavior | `EngineParallelInferenceTests` with opt-in real GGUFs via `TS_TEST_MODEL_DIR` |
| Server option translation | `ServerOptionsBuilderTests` for `--continuous-batching`, `--no-continuous-batching` and `--no-prefix-cache` (and `--spec` keeping Radix on) |

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `TS_SCHED_DISABLE_BATCHED` | `0` | `1` forces the per-sequence KV-swap fallback even when a model implements `IBatchedPagedModel`. |
| `TS_SCHED_MAX_BATCHED_TOKENS` | `4096` | Per-step token budget. |
| `TS_SCHED_MAX_RUNNING_SEQS` | `16` | Maximum in-flight sequences. |
| `TS_SCHED_PREFILL_CHUNK` | `256` | Per-request prefill cap while a decode is active. Prefill-only steps divide the full token budget fairly. Server flag: `--prefill-chunk-size N`. |
| `TS_SCHED_SOLO_PREFILL_CHUNK` | `8192` | Per-step prefill cap for a solo (uncontended) request — feeds the prompt through the fused whole-graph prefill path in big chunks. Bounded by `TS_SCHED_MAX_BATCHED_TOKENS`. A model with its own lone-prompt steps (`ISoloPrefillBudgetModel`) — Muse Spark with pipelined prefill: 8192 tokens, a remainder of at most 2048 joined to the step before it — sizes them unless this variable is set, and its steps may exceed `TS_SCHED_MAX_BATCHED_TOKENS`; a set value applies as described here and turns the model's steps off. `TS_SCHED_MAX_BATCHED_TOKENS` alone does not. The engine log names which rule is in force. |
| `TS_SCHED_NUM_BLOCKS` | `256` | Physical blocks in the engine pool. |
| `TS_SCHED_BLOCK_SIZE` | `16` for a page-only family (Hunyuan Dense, Mistral 3), `256` otherwise | Tokens per block. A family whose only cross-request reuse is whole pages reuses at this granularity: at 256, a turn shorter than a block reused nothing. |
| `TS_SCHED_PREFIX_CACHE` | `1` | Set `0` to disable all admission-time prompt reuse: Radix pages, end states and checkpoints. `--no-prefix-cache` (CLI and server) sets it. |
| `TS_SCHED_STOP_REPETITION` | `1` | Set `0` to let a looping generation run to its token limit rather than ending it with finish reason `repetition`; reasoning that repeats a templated line is then not closed either. |
| `TS_SCHED_DECODE_QUANTUM` | `256` | Number of decode tokens before a sequence switch is allowed in fallback-heavy execution. |
| `TS_PER_SEQ_FUSED` | `1` | Concurrent (N≥2) sequences on fused-capable models run per-request fused Forward, each on its own device K/V holder; `0` serves them on the op-by-op batched paged path instead, whose K/V lives in the shared host block pool (less device memory, a lower decode rate, and no retained holders). |
| `TS_BATCHED_FUSED_DECODE` | `1` | `0` disables true token-batched fused decode inside the per-sequence fused path (one graph decodes all N sequences). |
| `TS_RETAINED_FUSED_CACHE_MAX` | the running-request limit, at least `4` | How many finished conversations' end states stay retained for their next turn, on models that advertise retained holders: Gemma 4 K/V, Qwen 3.5/3.6/3.8 attention K/V plus GDN recurrent state, Qwen 3.8 Flash Next's per-sequence holder, DeepSeek V4.1's native slot, and GLM 5.x's donated native slot. Each pins the model's complete per-request continuation state. Unset, it follows `TS_SCHED_MAX_RUNNING_SEQS`, so every request that runs in parallel keeps its own; `0` turns retention off. |
| `TS_PREFIX_CHECKPOINTS_MAX` | `4` | How many checkpoints of the model's complete state at the end of the shared prompt prefix (the boundaries the chat layer marks on the request) the Radix tree keeps; each new chat starts from a clone of one. Applies on models that can copy their state (Gemma 4 on the GGML backends; Qwen 3.5/3.6/3.8 and Qwen 3.8 Flash Next on the shared decoder runtime wherever their slots' recurrent state can be copied, not a multi-node pipeline). A prompt publishes one per declared boundary (the end of the system instructions and the end of the shared prefix), so 4 covers a host that warms both thinking modes; `0` turns checkpoints off. |
| `TS_MM_EMBEDDING_CACHE_MB` | 1/64 of host memory, 512-4096 MB | Byte budget of the vision/audio embedding cache, which is keyed by media content (SHA-256); least-recently-used entries no prepared prompt references are evicted past it. |
| `TS_KV_INITIAL_TOKENS` | `0` | Tokens of K/V a cache is given when created, before any request declares a budget; `0` keeps the engine policy (the whole window when `MAX_CONTEXT` is explicit). On `--backend cuda` the shared decoder runtime serves every request in a slot of its own, so with `0` its primary cache starts at a fresh slot (1,024 cells) whatever `MAX_CONTEXT` says, and `MAX_CONTEXT` only bounds the window. The cache still grows on demand. |
| `TS_KV_GENERATION_RESERVE_MAX` | `0` | Cap on the generation share of a request's up-front K/V reservation (prompt + max_new_tokens); `0` = uncapped. Past the cap the cache grows on demand. |
| `TS_KV_HOLDER_POOL_MAX` | `64` | How many released per-request holders a model may park for reuse; each costs its whole K/V allocation while parked. |
| `TS_SPEC` | `0` | `1` enables speculative decoding for solo sequences (`--spec`). |
| `TS_SPEC_TYPE` | `auto` | Speculation algorithm: `auto`, `draft-head`, `block` or `ngram` (`--spec-type`). |
| `TS_SPEC_DRAFT` | `8` | Max tokens drafted per speculative step, 1-64 (`--spec-draft`); a model's preferred window replaces the default (usually narrower; 12 for n-gram on a dense, mostly IQ4_XS Qwen 3.5-family trunk on `ggml_metal`). |
| `TS_SPEC_PMIN` | per algorithm | Draft-confidence gate (`--spec-pmin`): `0.15` for a per-token head, `0.35` for a block drafter, `0` for n-gram; `0` never gates. |
| `TS_SPEC_DRAFT_MODEL` | none | Path to a drafter that ships as its own GGUF (`--draft-model`): Gemma 4's `gemma4-assistant`, Qwen 3.8 Flash Next's shared MTP head, a DSpark or DFlash drafter. The architecture is read from the file. |
| `DIFFUSION_STEPS` | `48` | Web UI DiffusionGemma denoising steps per block. This is separate from autoregressive scheduler step budgets. |
| `DIFFUSION_MAX_BATCH` | `2` | Maximum active DiffusionGemma Web UI requests in the diffusion scheduler. |

A host can make a shared-prefix checkpoint outlive the process by attaching an
`IPrefixCheckpointStore` to the engine (`InferenceEngine.PrefixCheckpointStore`, or
`InferenceEngineHost.PrefixCheckpointStore` in TensorSharp.Chat). The executor then
reads a saved checkpoint at admission for a prompt whose shared prefix no in-memory
checkpoint covers, and writes one the moment it takes a checkpoint; the model family
owns the byte format (`IBatchedPagedModel.TryExportRetainedCache` /
`TryImportRetainedCache`, implemented by Qwen 3.5/3.6 and Gemma 4) and validates it
on the way in. `PrefixCheckpointFileStore` (`TensorSharp.Runtime.Scheduling`) is the
file-backed store, keeping at most two files per model; it is what makes the first
message after a launch as fast as the second. The server attaches it for its startup
model under `prefix-cache/<model-file-stem>-<hash>/` beside the binary (moved with
`TENSORSHARP_PREFIX_CACHE_DIR`, turned off with `--no-prefix-cache`) and warms the
Web UI's shared prompt before it accepts requests. `--no-webui` skips this prompt
prefill while retaining model kernel warmup and normal API prefix reuse.
TensorAgent attaches one per model in the
app's cache directory. The CLI keeps checkpoints in memory only.


Server CLI aliases:

```bash
--continuous-batching      # default, sets TS_SCHED_DISABLE_BATCHED=0
--no-continuous-batching   # sets TS_SCHED_DISABLE_BATCHED=1
--prefill-chunk-size N     # sets TS_SCHED_PREFILL_CHUNK
```

Request KV state is owned by `InferenceEngine`. The KV cache can be stored at reduced precision with
`--kv-cache-dtype <f32|f16|q8_0|q4_0>` (or the `KV_CACHE_DTYPE` env var) on
both the CLI and the server. The default is chosen automatically per model
(`f16` when the model's weights are below F32, `f32` otherwise); the block-quantized tiers
(`q8_0`, `q4_0`) run on the GGML backends' flash-attention path, and `q8_0` also
on `--backend cuda`, whose executor's flash kernels read it directly for every
family on the shared decoder runtime (Qwen 3.5/3.6/3.8, Mistral 3, Hunyuan Dense;
about half the bytes of `f16` a step; its slots, copies and growth halve too), while `q4_0`
keeps a model off whole-forward graphs there. `q4_0` (~1/7 of f32) targets very
long 128K–256K contexts where the KV cache dominates memory.

`f16` stays the default on `--backend cuda`. `q8_0` holds about twice the context in the same
device memory, but on an RTX PRO 4500 Blackwell (sm_120) its kernels are bound by their arithmetic,
not by memory traffic: with Qwen3.8-27B NVFP4 a verify step took the same time as with `f16` (6.6K-69K
tokens), and a fresh prefill took 1-5% longer, more for longer prompts (6.6K-130K tokens; issue #376's reproducer). Its
perplexity stayed within 0.16% of `f16` on Qwen3.5-0.8B and Ministral 3 3B (8K-96K tokens) and within 0.3% on
Qwen3.8-27B with Q6_K weights (8K-16K). Qwen3.8-27B NVFP4 runs its prompt rows on 4-bit activations, which amplify any
change of attention arithmetic: there `q8_0` moved a few long code windows' perplexity a lot, in both directions (as the
8-bit prompt attention does), and lost the document order in 1 of 18 retrieval prompts (8K-128K) that `f16` answered.
Choose `q8_0` on `--backend cuda` when a long context does not fit at `f16`.

## Remaining Work

- Build one native GGML graph for an entire attention batch instead of one
  small graph per sequence. This should reduce launch/compile overhead on many
  short sequences.
- Move the paged K/V pool onto the device: write a step's K/V with
  `ggml_set_rows` and gather each sequence's history with `ggml_get_rows` inside
  the attention graph, instead of the host gather. Until then the batched path
  keeps its host-resident cache and its measured ceiling. This is the
  prerequisite for the generic `BatchedPaged` route (models without a
  token-batched fused decode) to gain aggregate throughput under concurrency.
- Complete Gemma 4 batched coverage for MoE variants, multimodal pending
  embeddings, and block-quantized KV cache.
- Decide whether DiffusionGemma scheduler metrics should be surfaced through
  `/api/queue/status` or a dedicated diffusion endpoint once operational usage
  needs per-batch visibility.
- Move prepared multimodal embedding lists out of model-level mutable state and
  into `SequenceState`, so multimodal prompt preparation can run fully in
  parallel instead of being serialized before submission.
