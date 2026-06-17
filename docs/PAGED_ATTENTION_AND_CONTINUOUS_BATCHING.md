# Paged Attention & Continuous Batching in TensorSharp

[English](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md) | [中文](PAGED_ATTENTION_AND_CONTINUOUS_BATCHING_zh-cn.md)

This document is the current implementation reference for TensorSharp's
vLLM-style paged KV cache, block-hash prefix sharing, and iteration-level
continuous batching. The server now routes inference through this engine by
default; the old single-request FIFO queue object remains only as a no-op
compatibility shim for queue-status/event shapes.

## Current Status

| Area | Status |
|---|---|
| Server engine | `TensorSharp.Server` owns one `InferenceEngineHost` per loaded model. `ChatGenerationPipeline` submits rendered prompts to the engine and streams tokens from `InferenceRequestHandle`. |
| Scheduler | `ContinuousBatchScheduler` admits waiting requests, preempts running work when block pressure requires it, applies a per-step token budget, and shares full prefix blocks by hash. |
| KV storage | `BlockPool`, `BlockTable`, `PagedKvStorage`, and `BlockHashIndex` hold fixed-size physical blocks with ref counts, LRU free ordering, and content-addressed lookup. |
| Batched execution | Models that implement `IBatchedPagedModel.ForwardBatch` pack all scheduled sequences into one model call with explicit `positions`, `slotMapping`, `queryStartLoc`, and per-sequence block tables. |
| Fallback execution | Models or feature combinations that cannot run batched throw `NotSupportedException`; `BatchExecutor` falls back to the isolated per-sequence KV-swap path inside the same engine. |
| Native attention | `TSGgml_PagedAttentionForward` gathers paged K/V in C++ and dispatches `ggml_flash_attn_ext`; GPT OSS uses `TSGgml_PagedAttentionForwardWithSinks`. |
| Speculative decoding | Optional MTP / NextN draft heads accelerate solo (non-concurrent) sequences. `BatchExecutor` drives the shared `MtpSpeculativeExecution` draft / verify / rollback core for models that implement `IMtpBatchedSpeculativeModel` (Qwen 3.6 embedded NextN; Gemma 4 separate `gemma4-assistant` draft GGUF). Off by default; `--mtp-spec`. See [Speculative decoding (MTP / NextN)](#speculative-decoding-mtp--nextn). |
| Queue API | `InferenceQueue` is a no-op shim. `/api/queue/status` and queue-position event shapes are retained for clients that expect the fields, not because requests are serialized there. |
| Diffusion models | DiffusionGemma does not enter this autoregressive `ForwardBatch` contract. CLI generation uses `DiffusionGemmaSampler`; the Web UI uses `DiffusionBatchScheduler` to batch denoising work at block boundaries. |

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
               - calls ForwardBatch when available
               - otherwise swaps per-sequence KV blocks
               - samples decode tokens
               - captures newly full blocks
        |
        v
BlockPool + PagedKvStorage + BlockHashIndex
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
| `SequenceState` | `TensorSharp.Runtime/Scheduling/SequenceState.cs` | Mutable per-request status, tokens, blocks, logits, and sampling state. |
| `ContinuousBatchScheduler` | `TensorSharp.Runtime/Scheduling/ContinuousBatchScheduler.cs` | Iteration-level scheduler with prefix caching and preemption. |
| `BatchExecutor` | `TensorSharp.Runtime/Scheduling/BatchExecutor.cs` | Executes scheduled work, samples, and captures KV blocks. |
| `InferenceEngine` | `TensorSharp.Runtime/Scheduling/InferenceEngine.cs` | Worker loop and public submit/abort surface. |
| `InferenceEngineHost` | `TensorSharp.Server/InferenceEngineHost.cs` | Server-side per-model engine singleton. |

## Request Flow

1. A protocol adapter builds a normalized chat request.
2. `ChatGenerationPipeline` renders the prompt, resolves sampling options, and prepares any image/audio/video embeddings.
3. The pipeline creates a `SequenceState` and calls `InferenceEngine.SubmitRequest`.
4. The engine worker asks `ContinuousBatchScheduler` for the next step.
5. The scheduler admits waiting sequences while token and sequence budgets allow. Before allocating new blocks, it looks up full prompt blocks in `BlockHashIndex` and adopts shared blocks on a hit.
6. If the pool is under pressure, the scheduler can preempt lower-priority running sequences, commit their full blocks, free the remainder, and requeue them.
7. `BatchExecutor` executes the scheduled step. It uses `ForwardBatch` when the model and feature combination support it, otherwise it uses the per-sequence KV-swap fallback.
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

## Execution Paths

### Batched Path

The batched path is the fast multi-request path. It avoids K/V ownership swaps
and amortizes linear projections across all scheduled tokens. Most current
batched ports use native paged attention for GGML backends:

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

When `--mtp-spec` (`TS_MTP_SPEC=1`) is set, `BatchExecutor` runs an optional
multi-token-prediction speculative path for **solo (non-concurrent)** sequences
on models that implement `IMtpBatchedSpeculativeModel`. The flow per step:

1. **Draft.** The model's draft head proposes up to `TS_MTP_DRAFT` (default `8`)
   future tokens, stopping at the first token whose draft confidence falls below
   `TS_MTP_PMIN` (default `0.75`). The request's own sampler — temperature,
   top-k/p, and repetition/presence/frequency penalties — drives the drafting so
   the speculation stays aligned with what standard decode would have produced.
2. **Verify.** The trunk verifies all drafted tokens in a single batched forward
   and the same sampler accepts the longest matching prefix. Because verification
   re-derives every committed token, the output is **identical** to standard
   decode; speculation only changes how many forward passes it takes.
3. **Rollback.** On partial acceptance, KV (and any recurrent state) past the
   accepted prefix is rolled back before the next step.

Two draft-head shapes share the `MtpSpeculativeExecution` core:

| Model | Draft head | State on rejection |
|---|---|---|
| Qwen 3.6 | Embedded NextN block in the trunk GGUF (`{arch}.nextn_predict_layers`); no extra file. `--mtp-draft-model` is ignored. | GatedDeltaNet recurrent-state snapshot/restore (device-side on CUDA). |
| Gemma 4 | Separate EAGLE-style `gemma4-assistant` GGUF via `--mtp-draft-model`; draft layers attend the **target's** last local / global KV (no draft K/V of its own). | Attention-KV position rewind only — the drafter is stateless given `(token, h)`. |

Speculation engages only where it is profitable (`MtpSpeculationProfitable`):
ggml backends (fused multi-token-verify + draft-step kernels) and the pure-C#
`cuda` backend (GPU-resident per-op verify/draft). On CPU / GGML CPU / MLX the
verify can't keep up, so the engine serves standard decode. Concurrent batches
never speculate — when more than one sequence is running, every sequence uses the
normal batched/fallback step. A mismatched or incomplete Gemma 4 draft GGUF
fails fast at server startup (`MtpStartupValidation`).

## Model Status

| Model family | Batched / paged status | Opt-out / sub-toggle |
|---|---|---|
| Mistral 3 | Default `ForwardBatch` path. Uses paged K/V, YaRN-aware positions, native paged attention, and vision embedding injection after prompt preparation. Validated on Ministral-3-14B; long-context native paged attention is about 21% faster than the legacy per-sequence GGML path. | `TS_PAGED_ATTN_KERNEL` selects `native`, `tensor`, or `managed`. |
| Gemma 4 | Default batched path for dense text workloads, including per-layer SWA/global attention, variable head dims, PLE, and KV-donor layer aliasing. Current fallback cases include pending multimodal embeddings, MoE layers, and block-quantized KV cache. Optional MTP speculative decode via a separate `gemma4-assistant` draft GGUF. | `TS_GEMMA4_BATCHED=0` forces per-sequence fallback. `--mtp-spec` + `--mtp-draft-model` enables speculation; `TS_GMTP_*` are draft-path A/B switches. |
| Qwen 3 | Reference attention-only batched port with paged K/V, per-token RoPE positions, and last-token gather. Optional tests self-validate when a base Qwen 3 GGUF is available. | No model-specific opt-out; global `TS_SCHED_DISABLE_BATCHED=1` forces fallback. |
| Qwen 3.5 / 3.6 family | Default batched path. Handles full-attention layers, GatedDeltaNet recurrent layers via per-slot state pools, MoE variants, vision injection, and multimodal RoPE tables. Qwen 3.6 additionally supports MTP speculative decode via its embedded NextN block (GDN recurrent-state snapshot/rollback). | `TS_QWEN35_BATCHED=0`; `TS_QWEN35_BATCHED_GDN_NATIVE=1` enables the native batched GDN kernel; `--mtp-spec` enables speculation on Qwen 3.6. |
| GPT OSS | Default batched path. Handles Q/K/V/O bias, YaRN RoPE, sliding-window layers, attention sinks, MXFP4 MoE experts, and native sinks attention. Greedy correctness has been validated against the legacy path; performance remains limited by per-layer graph construction. | `TS_GPTOSS_BATCHED=0`; `TS_GPTOSS_PAGED_ATTN_MANAGED=1`. |
| Nemotron-H | Default batched path. Attention layers use paged K/V; Mamba2 layers use per-slot conv/SSM state pools; MoE layers use batched expert kernels; prepared image/audio embeddings can be injected into the batched hidden state. | `TS_NEMOTRON_BATCHED=0`; `TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1` enables the native batched Mamba2 step. |
| Gemma 3 | No true `ForwardBatch` port yet; runs through the engine's per-sequence fallback. | Global fallback only. |
| DiffusionGemma | Separate text-diffusion path. `Forward(int[] tokens)` is intentionally unsupported; generation iteratively denoises fixed-length canvas blocks. Web UI requests share `DiffusionBatchScheduler`, which admits concurrent requests between blocks and can optionally batch active canvases. | `DIFFUSION_STEPS`, `DIFFUSION_MAX_BATCH`, `DIFFUSION_BATCHED_FORWARD`; `DIFFUSION_NO_FUSED_DECODE=1` disables the GGML whole-model diffusion decode. |

## Test Coverage

| Area | Tests |
|---|---|
| Scheduler / block pool | `ContinuousBatchSchedulerTests`, `PagedKvCacheTests`, `PagedKvCacheCodecTests` |
| Batched executor primitives | `BatchedExecutorTests`, including managed paged-attention correctness and multi-sequence logits routing |
| Per-model correctness | `Qwen35BatchedCorrectnessTests`, `Mistral3BatchedForwardTests`, `Gemma4BatchedForwardTests`, `GptOssBatchedCorrectnessTests`, `NemotronBatchedCorrectnessTests`, optional `Qwen3BatchedForwardTests` |
| MTP speculative decoding | `MtpSpeculativeExecutionTests` (draft/verify/rollback core), opt-in end-to-end `Qwen36MtpTests` (`TS_MTP_E2E=1`) and `Gemma4MtpTests` (`TS_GMTP_E2E=1`) with real GGUFs |
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
| `TS_SCHED_PREFILL_CHUNK` | `1024` | Maximum prefill tokens scheduled per step. |
| `TS_SCHED_NUM_BLOCKS` | `256` | Physical blocks in the engine pool. |
| `TS_SCHED_BLOCK_SIZE` | `256` | Tokens per block. |
| `TS_SCHED_PREFIX_CACHE` | `1` | Set `0` to disable block-hash prefix reuse. |
| `TS_SCHED_DECODE_QUANTUM` | block size | Number of decode tokens before a sequence switch is allowed in fallback-heavy execution. |
| `TS_BATCHED_N1_FAST_PATH` | `0` | Routes eligible single-sequence steps through the batched path for A/B testing. |
| `TS_KV_PAGED_QUANT_BITS` | `0` | Optional TurboQuant codec bits for paged KV blocks (`4` or `8`); recurrent-state models may fall back to passthrough. |
| `TS_MTP_SPEC` | `0` | `1` enables MTP / NextN speculative decoding for solo sequences (CLI `--mtp-spec`). |
| `TS_MTP_DRAFT` | `8` | Max tokens drafted per speculative step (CLI `--mtp-draft`). |
| `TS_MTP_PMIN` | `0.75` | Min draft-head confidence to keep a drafted token (CLI `--mtp-pmin`). |
| `TS_MTP_DRAFT_MODEL` | none | Path to the separate Gemma 4 `gemma4-assistant` draft GGUF (CLI `--mtp-draft-model`); ignored by Qwen 3.6. |
| `TS_GMTP_NO_FUSED` / `TS_GMTP_NO_FAST_ROLLBACK` / `TS_GMTP_BATCHED_TRUNK` | off | Gemma 4 draft-path A/B switches (disable fused verify/draft kernels; restore kept-prefix rollback; run the batched trunk instead of the linear trunk). |
| `DIFFUSION_STEPS` | `48` | Web UI DiffusionGemma denoising steps per block. This is separate from autoregressive scheduler step budgets. |
| `DIFFUSION_MAX_BATCH` | `2` | Maximum active DiffusionGemma Web UI requests in the diffusion scheduler. |
| `DIFFUSION_BATCHED_FORWARD` | `0` | Enables true batched canvas decode for active DiffusionGemma canvases; the default favors the fused single-canvas path. |

Server CLI aliases:

```bash
--continuous-batching      # default, sets TS_SCHED_DISABLE_BATCHED=0
--no-continuous-batching   # sets TS_SCHED_DISABLE_BATCHED=1
--paged-batching           # alias for --continuous-batching
--no-paged-batching        # alias for --no-continuous-batching
```

The older `--paged-kv*` flags are retained for compatibility with the removed
standalone per-session paged-KV manager. Current server request KV state is
owned by `InferenceEngine`.

## Remaining Work

- Build one native GGML graph for an entire attention batch instead of one
  small graph per sequence. This should reduce launch/compile overhead on many
  short sequences.
- Move the K/V gather from CPU memcpy to GPU-side `ggml_get_rows` or equivalent
  indexed gathers where backend support makes it worthwhile.
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
