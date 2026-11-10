# Tensor Parallelism Plan

> **Superseded for new work** by [docs/design/unified-execution.md](docs/design/unified-execution.md): placement
> (TP, layer split, their hybrid) and expert offload are now runtime mechanisms declared per family in a
> mandatory serving contract. This document records how the per-family TP paths were built.

## Overview

This document describes the plan for adding tensor parallelism (TP) to TensorSharp,
progressing from local multi-GPU parallelism through network-distributed inference
with shared state, to RDMA-class memory access if the performance profile warrants it.

---

## Stage 1 — Local Tensor Parallelism (Implemented)

**Goal:** Split a single model across multiple CUDA GPUs within one process, using
the Megatron-LM column/row-parallel pattern.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Process                                                │
│                                                         │
│  ┌──────────┐  ┌──────────┐       ┌──────────┐        │
│  │ GPU 0    │  │ GPU 1    │  ...  │ GPU N-1  │        │
│  │ Allocator│  │ Allocator│       │ Allocator│        │
│  │ Stream   │  │ Stream   │       │ Stream   │        │
│  │ Weights  │  │ Weights  │       │ Weights  │        │
│  │ (shard)  │  │ (shard)  │       │ (shard)  │        │
│  │ KV Cache │  │ KV Cache │       │ KV Cache │        │
│  └────┬─────┘  └────┬─────┘       └────┬─────┘        │
│       │              │                  │              │
│       └──────────────┼──────────────────┘              │
│                      │                                 │
│              P2P AllReduce                             │
│         (cuMemcpyPeerAsync + add kernel)               │
└─────────────────────────────────────────────────────────┘
```

### Transformer Block TP Pattern

```
Replicated hidden state (all GPUs hold identical copy)
        │
        ▼
  ┌─ RMSNorm (replicated, each GPU independently) ─┐
  │                                                 │
  ▼                                                 │
  Column-Parallel QKV (split output heads)          │
  │  GPU 0: heads [0..H/tp)                        │
  │  GPU 1: heads [H/tp..2H/tp)                    │
  │  ...                                           │
  ▼                                                 │
  Per-GPU Attention (independent head subsets)      │
  │  Each GPU: QK norm, RoPE, KV cache, SDPA       │
  ▼                                                 │
  Row-Parallel Output Proj + AllReduce ─────────────┘
  │
  ▼
  Replicated hidden state (restored by AllReduce)
        │
        ▼
  ┌─ RMSNorm (replicated) ─────────────────────────┐
  │                                                 │
  ▼                                                 │
  Column-Parallel Gate/Up (split intermediate dim)  │
  │                                                 │
  ▼                                                 │
  Per-GPU SiLU·mul (independent)                    │
  │                                                 │
  ▼                                                 │
  Row-Parallel Down + AllReduce ────────────────────┘
  │
  ▼
  Replicated hidden state
```

### Files Created / Modified

| File | Change |
|------|--------|
| `TensorSharp.Backends.Cuda/Interop/CudaDriverApi.cs` | Added P2P bindings: `cuDeviceCanAccessPeer`, `cuCtxEnablePeerAccess`, `cuMemcpyPeerAsync`, `cuEventSynchronize` |
| `TensorSharp.Backends.Cuda/CudaEvent.cs` | **New.** CUDA event wrapper for cross-stream/device synchronization |
| `TensorSharp.Backends.Cuda/CudaP2PCommunicator.cs` | **New.** AllReduce via P2P copies + elementwise-add kernel. Reduce-to-zero + broadcast algorithm. Host-memory fallback when P2P is unavailable |
| `TensorSharp.Backends.Cuda/TensorParallelGroup.cs` | **New.** Multi-GPU coordinator: owns N `CudaAllocator`s + P2P communicator. Public API: `AllReduce(Tensor[])`, `GetAllocator(rank)`, `Synchronize()` |
| `TensorSharp.Models/ModelBase.cs` | Added TP fields, `tpDegree` constructor param, `ShardWeightsForTensorParallelism()`, `TpColumnParallelLinear()`, `TpRowParallelLinear()`, `TpRMSNorm()`, `TpResidualAdd()`, `BroadcastTensorToAllRanks()`, `PrepareCudaQuantizedWeightsForInferenceTP()`, TP cleanup in `Dispose()`, `Create()` accepts `tpDegree` |
| `TensorSharp.Models/Models/Mistral3/Mistral3Model.cs` | Since 2026-11: built on the shared decoder runtime (`TensorSharp.Models/SharedDecoder/`), which provides TP, layer split, their hybrid and batched serving for every family on it; see [docs/design/unified-execution.md](docs/design/unified-execution.md) |
| `TensorSharp.Models/Models/Gemma4/Gemma4Model.cs` | Constructor accepts `tpDegree`, dispatches to `ForwardTP()` when TP active |
| `TensorSharp.Models/Models/Gemma4/Gemma4Model.TensorParallel.cs` | **New.** TP forward pass with dense GeGLU + MoE dual-path FFN, per-layer head dims, shared KV layers |
| `TensorSharp.Models/Models/Qwen35/Qwen35Model.cs` | Constructor accepts `tpDegree`, dispatches to `ForwardTP()` when TP active |
| `TensorSharp.Models/Models/Qwen35/Qwen35Model.TensorParallel.cs` | **New.** TP forward pass with GatedDeltaNet SSM + full attention + MoE, block-cyclic V-head assignment, CUDA-native GDN kernels |
| `TensorSharp.Models/Models/GptOss/GptOssModel.cs` | Constructor accepts `tpDegree`, dispatches to `ForwardTP()` when TP active |
| `TensorSharp.Models/Models/GptOss/GptOssModel.TensorParallel.cs` | **New.** TP forward pass with biased QKV, attention sinks, YaRN, clamped SiLU GLU MoE |
| `TensorSharp.Models/Models/Nemotron/NemotronModel.cs` | Constructor accepts `tpDegree`, dispatches to `ForwardTP()` when TP active |
| `TensorSharp.Models/Models/Nemotron/NemotronModel.TensorParallel.cs` | **New.** TP forward pass with Mamba2 (replicated), attention (no RoPE), dense + MoE FFN |
| `TensorSharp.Cli/Program.cs` | Added `--tp <N>` CLI argument |
| `TensorSharp.Models/Models/Gemma4/Gemma4Model.BatchedForward.cs` | `BatchedForwardAvailable` returns false under TP (per-seq fallback) |
| `TensorSharp.Models/Models/Qwen35/Qwen35Model.BatchedForward.cs` | `BatchedForwardAvailable` returns false under TP (per-seq fallback) |
| `TensorSharp.Models/Models/GptOss/GptOssModel.BatchedForward.cs` | `BatchedForwardAvailable` returns false under TP (per-seq fallback) |
| `TensorSharp.Models/Models/Nemotron/NemotronModel.BatchedForward.cs` | `BatchedForwardAvailable` returns false under TP (per-seq fallback) |

### Weight Sharding

| Weight Pattern | Parallel Type | Split Dimension | Notes |
|---------------|---------------|-----------------|-------|
| `attn_qkv.weight` | Column | ne1 (output) | Consecutive rows → zero-copy view |
| `ffn_gate_up.weight` | Column | ne1 (output) | Consecutive rows → zero-copy view |
| `attn_output.weight` | Row | ne0 (input) | Block-aligned column extraction |
| `ffn_down.weight` | Row | ne0 (input) | Block-aligned column extraction |
| `*_norm.weight` | Replicated | — | Full copy on every GPU |
| `token_embd.weight` | Replicated | — | Embedding lookup on GPU 0 |
| `output.weight` | Replicated | — | LM head on GPU 0 (post-AllReduce) |

Quantized weights (Q4_0, Q8_0, etc.) are split at block boundaries (32 elements).
Column-parallel splits are zero-copy views into the original mmap'd data.
Row-parallel splits copy the relevant blocks per row into new aligned buffers.

### Usage

```bash
# CLI: 2-GPU tensor parallelism
TensorSharp.Cli --model qwen3.5-9b.gguf --backend cuda --tp 2

# Server: via environment variable
TENSORSHARP_TP_DEGREE=2 dotnet run --project TensorSharp.Server.Host

# Config JSON (auto-expanded to CLI args)
{ "tp": 2, "backend": "cuda", "model": "qwen3.5-9b.gguf" }
```

### Constraints

- Direct CUDA and the GGML CUDA/Vulkan backends (see Stage 1b); MLX is single-device
- `numHeads` and `numKVHeads` must be divisible by TP degree
- `intermediateSize` must be divisible by TP degree
- Quantized row-parallel splits require `ne0` divisible by `tp × blockSize`
- Single-process: one thread issues commands to all GPUs sequentially; CUDA
  streams provide the actual parallelism; AllReduce is the synchronization point

### MoE and SSM TP Strategies

MoE and SSM architectures required non-standard TP approaches beyond the
column/row-parallel pattern:

| Architecture | Strategy | Details |
|-------------|----------|---------|
| **MoE (Gemma4, Qwen3.5, GptOss, Nemotron)** | Expert slicing | Each GPU holds 1/tp slice of every expert's weights (column-parallel gate/up, row-parallel down). Router is replicated. AllReduce after weighted expert sum. |
| **GatedDeltaNet SSM (Qwen3.5)** | Per-rank V-head ownership | Block-cyclic V-head assignment. Each rank runs its own GDN kernel on its V-head subset with independent delta/conv state — no cross-rank communication needed for the recurrent path. Requires a device-resident packed-GDN kernel: `ts_qwen35_gdn_*` on direct CUDA, `TSGgml_Qwen35GdnLayerTP` on GGML. |
| **Mamba2 SSM (Nemotron)** | Replicated on rank 0 | Mamba2 layers run on rank 0 only, result broadcast to all ranks. SSM state lives in host arrays with a managed per-token loop; sharding would require a device-resident per-rank kernel. Attention + FFN/MoE layers hold the bulk of weights, so TP still delivers most memory savings. |

### Known Limitations / Future Work (Stage 1)

- [x] Extend TP to other model architectures (Gemma4, Qwen3.5, Mistral3, etc.)
- [x] TP-aware batched forward (Mistral3 implemented; MoE models fall back to per-seq)
- [ ] TP-aware batched forward for MoE models (Gemma4, Qwen3.5, GptOss, Nemotron)
- [ ] TP-aware CUDA graph capture (current graphs are single-device)
- [ ] Overlap AllReduce with next layer's norm (pipeline communication)
- [ ] NCCL backend for Linux (higher throughput than P2P for N > 2)
- [ ] Column-parallel LM head with AllGather (currently replicated on GPU 0)
- [ ] GptOss: expert down-projection bias is skipped in TP mode (small correction, documented as follow-up)
- [ ] Nemotron: Mamba2 layers are replicated on rank 0 rather than sharded (requires device-resident SSM kernel)

---

## Stage 1b — Tensor Parallelism on the GGML Backends (Implemented)

**Goal:** Let the GGML CUDA / Vulkan backends span several GPUs (and, via
Stage 2's TCP layer, several nodes) so a model larger than one GPU's VRAM can
run without falling back to CPU offload.

### Why this needed its own stage

The GGML backend has a different execution model from direct CUDA. Tensors live
in **host** memory (`GgmlStorage` allocates from an mmap/VirtualAlloc pool); ops
upload to the device, run a ggml graph, and copy the result back. There was also
exactly one ggml backend per process (`g_backend`), so "which GPU" was not a
concept the bridge had.

Three things had to change:

1. **One ggml backend per GPU.** `tsg::g_device_states[]` holds a backend plus
   its own buffer caches per rank; the active rank is a `thread_local`, so a
   worker pool can drive several GPUs at once. `g_backend` and the cache globals
   became macros onto the active slot, which kept ~500 existing use sites
   working unchanged.
2. **Per-rank device residency.** The host-pointer-keyed buffer caches are now
   per rank. This matters for *replicated* weights: one host pointer, but a
   distinct device copy per GPU. Without it rank 1 would be handed a buffer
   living on rank 0's card.
3. **A collective.** `ggml_backend_reg_get_proc_address("ggml_backend_comm_*")`
   exposes ggml-cuda's AllReduce — NCCL when available, its P2P pipeline
   otherwise, butterfly as a last resort. This is the same collective
   llama.cpp's `--split-mode tensor` uses. A multi-threaded host reduction is
   the fallback for backends without one, and is the *faster* choice for small
   payloads: TensorSharp's activations are already in host RAM, so summing them
   there costs nothing extra, while the device path pays two PCIe crossings.
   `TS_GGML_TP_DEVICE_AR_THRESHOLD` (default 256 Ki elements) picks between them.

### Files Created / Modified

| File | Change |
|------|--------|
| `TensorSharp.GGML.Native/ggml_ops_internal.h` | `DeviceState` table, `TSG_MAX_DEVICES`, `thread_local g_active_rank`, `ScopedRank`, compatibility macros for the old globals |
| `TensorSharp.GGML.Native/ggml_ops_core.cpp` | Per-rank backend creation (`create_backend_instance_on_device`, `gpu_device_count`), per-rank reuse buffers/gallocr, rank-wide teardown and budget config |
| `TensorSharp.GGML.Native/ggml_ops_tensor_parallel.cpp` | **New.** TP init, collective resolution, device + host AllReduce, and the fused multi-rank matmul |
| `TensorSharp.GGML.Native/ggml_ops_qwen35_gdn_tp.cpp` | **New.** Per-rank packed GatedDeltaNet block (norm + in-projection + ssm_conv + delta-rule scan + gated norm) as one cached graph, with device-resident recurrent state |
| `TensorSharp.Models/Models/Qwen35/Qwen35Model.TensorParallelGgmlMoE.cs` | **New.** Expert-parallel MoE for Qwen 3.5/3.6 on GGML: whole-expert slices, per-rank route tables, Megatron-split shared expert |
| `TensorSharp.GGML.Native/CMakeLists.txt` | Auto-detect NCCL instead of forcing `GGML_CUDA_NCCL=OFF` |
| `TensorSharp.Backends.GGML/GgmlContext.cs` | Accepts N device ids; exposes `Degree`, `DeviceIds`, `HasDeviceAllReduce` |
| `TensorSharp.Backends.GGML/GgmlTensorParallel.cs` | **New.** P/Invoke surface: device enumeration, rank selection, AllReduce, multi-rank matmul |
| `TensorSharp.Backends.GGML/GgmlTensorParallelGroup.cs` | **New.** `ITensorParallelGroup` over N ggml backends: per-rank allocators, rank worker pool, op dispatch hook |
| `TensorSharp.Core/ITensorParallelGroup.cs` | Moved out of the CUDA project; `GetAllocator` returns `IAllocator`; added `RunPerRank`; added `INestedTensorParallelGroup` |
| `TensorSharp.Core/OpRegistry.cs` | `PreInvokeHook` so a multi-device backend can route an op to its result's device |
| `TensorSharp.Distributed/DistributedTensorParallelGroup.cs` | Takes any local group, so multi-node works over GGML too |
| `TensorSharp.Models/ModelBase.cs` | GGML TP context/group construction, `TENSORSHARP_TP_DEVICES`, per-rank weight preload, GGML branch in `AddmmQuantManaged`, fused TP linear path |
| `TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp` | **New.** GLM 5.x runs TP inside its own whole-model executor rather than through the shared group: the graph builder emits every rank's slice into one `ggml_backend_sched` graph and reduces the partials with plain `ggml_add`, so the scheduler owns the cross-device copies. Heads split column/row-parallel; the routed experts split **row-wise inside each expert** (`slice_mid` / `slice_lo`), because `ggml_mul_mat_id` needs a token's selected expert ids to stay distinct and an id-space split cannot provide that. See [glm.md](docs/models/glm.md#tensor-parallelism) for the measured cost of the two all-reduces per layer on a PCIe-only host. |

### How an op finds its GPU

Two mechanisms, because rank is a property of a *block of work*, not of a
single op:

* `ITensorParallelGroup.RunPerRank(body)` pins each rank's worker thread to its
  GPU for the duration of `body`. This is the reliable base, and it is what
  makes the ranks actually run concurrently — GGML ops submit *and* synchronize
  in one call, so a sequential rank loop runs the GPUs strictly one after
  another and TP ends up slower than a single device.
* `OpRegistry.PreInvokeHook` selects the device from an op's result tensor.
  TensorSharp ops take the destination first, so the result's allocator names
  the rank. This catches ops issued outside a `RunPerRank` scope without every
  model file having to say which GPU it means.

### Two hazards the concurrent-rank design had to solve

**ggml-cuda's memory pool is not shareable between threads.**
`ggml_cuda_pool_vmm` is a lock-free bump allocator on the backend context that
asserts every free is the exact reverse of its alloc
(`GGML_ASSERT(ptr == pool_addr + pool_used)`). Rank fan-out is safe only because
each thread owns a *different* backend — so the per-op dispatch hook must never
move a worker onto another rank's backend. `RunPerRank` therefore **pins** the
thread's rank for the duration of the body, and the hook yields to an active pin
(`SetActiveRankIfUnpinned`). Without the pin, an op whose result happened to live
on another rank dragged the worker onto a backend its owner was still using and
tripped the assert mid-prefill.

**Preloaded weights are addressed by an opaque cache key.** Once C# pins a
quantized weight it hands the bridge a GCHandle, not a host pointer, so the
device copy survives the host buffer being released. Every lookup is keyed on it
— but the *miss* path uploaded from that same pointer, dereferencing a GCHandle
as if it were weight bytes and segfaulting inside `cudaMemcpyAsync`. The bridge
now records `cache key -> host data` at preload (`register_cache_key`) and
resolves it in `upload_binding`, so a miss is merely a re-upload. Set
`TS_GGML_LOG_VRAM=1` to see when that fires — each occurrence is a whole weight
re-uploaded, so a steady stream of them is a performance signal.

### CUDA graph capture

ggml-cuda records graphs with `cudaStreamBeginCapture`. Capture is process-wide:
while one rank's thread is capturing, an unsafe CUDA call from another rank's
thread poisons it (`operation failed due to a previous error during capture`).
Concurrent ranks are the point of TP here, so `TSGgml_TensorParallelInit` sets
`GGML_CUDA_DISABLE_GRAPHS=1` for multi-GPU runs with concurrent dispatch. It is
set natively rather than from C# because .NET's `Environment.SetEnvironmentVariable`
does not reach the native `getenv`.

### Usage

```bash
# 2 GPUs on the GGML CUDA backend
TensorSharp.Cli --model qwen3.5-9b.gguf --backend ggml_cuda --tp 2

# Pick specific GPUs
TENSORSHARP_TP_DEVICES=0,2 TensorSharp.Cli --model m.gguf --backend ggml_cuda --tp 2

# Multi-node (2 nodes x 2 local GPUs)
TensorSharp.Cli --model m.gguf --backend ggml_cuda --tp 2 \
  --tp-node-id 0 --tp-peers "10.0.0.1:9500,10.0.0.2:9500"
```

| Variable | Effect |
|---|---|
| `TENSORSHARP_TP_DEVICES` | Explicit GPU ordinals per rank (default `0..tp-1`) |
| `TS_GGML_TP_PARALLEL=0` | Sequential rank dispatch (diagnostic) |
| `TS_GGML_TP_DEVICE_AR_THRESHOLD` | Element count above which AllReduce uses the device collective |
| `TS_GGML_F32_RESIDENT=0` | Bind F32 linear weights per call instead of device-resident (diagnostic) |
| `TS_QWEN35_LAYER_TRACE=1` | Per-layer residual-stream summary for the first forward, from both the single-GPU and TP loops |
| `GGML_CUDA_ALLREDUCE` | `nccl` / `internal` / `none` — passed through to ggml |

### Measured results

2× RTX 2000 Ada (16 GB each, PCIe, no NVLink), CUDA 12.8, NCCL 2.25.

**Memory — the capacity win, verified:**

| Model | 1 GPU | TP=2 rank 0 | TP=2 rank 1 |
|---|---|---|---|
| Gemma-4-26B-A4B IQ4_XS | 12908 MB | 6835 MB | 6087 MB |

---

## Stage 1c — Fused TP execution (Implemented)

**Goal:** stop TP being slower than one GPU.

### The problem

Stage 1b measured TP as a capacity win and a latency loss, and blamed dispatch
count. That was right, and the magnitude was worse than the initial measurements
suggested. On Gemma-4-E4B Q8_0, `--tp 2` ran prefill at 37.5 tok/s against
1000+ on one GPU, and decode at 5.5 against 36.8. Every GGML op submits a graph,
synchronizes, and copies its result back to host memory; a 42-layer decode is
~2000 of those per token, doubled across ranks. AllReduce was not the cost.

### The mechanism

`tsg::TpRankPlan` + `tsg::tp_execute_plans` (`ggml_ops_tensor_parallel.cpp`).

A fused kernel already builds its whole-model (or whole-layer) graph in one
piece. Given `tp_degree > 1` it now builds that same graph over **this rank's
shards**, records the nodes where a row-parallel projection leaves a *partial*
sum, and returns a plan instead of executing. The caller collects one plan per
rank and hands them to the driver, which runs exactly the schedule
`ggml-backend-meta.cpp` runs for llama.cpp's `--split-mode tensor`:

```
for each segment k:
    for each rank r:  ggml_graph_view(graph[r], start[r], end[r])
                      ggml_backend_graph_compute_async(backend[r], view)
    if k is not last: ggml_backend_comm_allreduce_tensor(partial[0..N])   # NCCL / P2P, in VRAM
synchronize every rank; download the result
```

Activations never leave VRAM, the collective reduces device buffers in place,
and the host issues `2·L` graph launches per token instead of ~2000 op round
trips. Because every rank is submitted from one thread, ggml-cuda's graph
capture also stays valid. Capture is **on by default** under TP: it measured
within noise on the 2-GPU box this section was written against, but on wider
groups it is worth ~45% of decode throughput (4×A40: Qwen 3.5-9B `--tp 4`
88 → 128.5 tok/s, Qwen 3.5-35B-A3B `--tp 2` 71.3 → 104.1). Opt out with
`TS_GGML_TP_CUDA_GRAPHS=0`.

Everything outside the row-parallel projections is replicated work — each rank
runs the norms, RoPE, PLE injection and layer scalars on identical values and
gets identical answers. Only rank 0 folds the final norm and the LM head: it is
the largest tensor left after sharding, and replicating it would give back a
large part of what TP just saved.

### Kernels wired onto the executor

| Kernel | Covers | Split points per layer |
|---|---|---|
| `TSGgml_Gemma4ModelDecode` | Gemma 4 dense decode (whole model) | attn output proj, FFN down proj |
| `TSGgml_Gemma4ModelVerify` | Gemma 4 dense prefill (whole model) | attn output proj, FFN down proj |
| `TSGgml_Qwen35AttentionLayerPrefill` | Qwen 3.5/3.6 full-attention block | attn output proj |
| `TSGgml_FusedFFNSwiGLUQuantF32` | any dense SwiGLU FFN block | FFN down proj |
| `TSGgml_FusedMatMulQuantAddF32` | any row-parallel linear + residual | the matmul |

The last two are architecture-agnostic: any TP forward that ends a block with
`row-parallel linear → AllReduce → residual add` can replace those three host
round trips with two graph launches. Qwen 3.5 uses it for the GatedDeltaNet
output projection.

### Measured (2× RTX 2000 Ada, PCIe, no NVLink, `--tp 2`, prefill 512 / decode 64)

| Model | 1 GPU | TP=2 before | TP=2 after |
|---|---|---|---|
| Gemma-4-E4B Q8_0 | 2760 / 37.3 | 37.5 / 5.5 | **2488 / 51.7** |
| Qwen3.5-9B Q8_0 | 1461 / 23.1 | 54.8 / 15.6 | **399 / 24.4** |
| Qwen3.5-35B-A3B IQ4_XS | does not fit | 98.7 / 14.2 | **184 / 18.1** |
| Gemma-4-26B-A4B IQ4_XS | 1845 / 48.5 | 43.0 / 4.9 | 43.1 / 5.1 (unchanged) |

**Muse-Glimmer-30B (2× RTX PRO 4000 Blackwell 24 GB, PCIe, prefill 512 / decode 64):**

| Model | 1 GPU | TP=2 |
|---|---|---|
| 30B-UD-IQ2_XXS (10.2 GB) | 1171 / 40.2 | **1569 / 63.2** |
| 30B-Q8_0 (28.2 GB) | does not fit on 24 GB | **1691 / 34.3** |

1.34× prefill and 1.57× decode — TP beats one GPU on BOTH phases here, which the
earlier models did not manage on prefill. Resident: IQ2_XXS 9178 MB on one GPU
versus 5115 + 4063 MB at tp=2; Q8_0 15474 + 12748 MB, i.e. the only way to run it
on these cards at all. `--tp 2` output is byte-identical across repeat runs and
tracks the single-GPU greedy continuation for 468 of 500 characters.

(prefill tok/s / decode tok/s.) Decode is where TP should win and now does —
1.39× single-GPU on E4B, 1.06× on Qwen3.5-9B — while prefill, which is
compute-bound and pays the collectives, lands at 0.90× / 0.27× instead of 0.01×
/ 0.04×. Qwen3.5-35B does not fit on one 16 GB card at all, so TP is the only
way to run it, and it is now 1.9× / 1.3× faster than it was.

**Correctness.** Gemma-4-E4B and Gemma-4-26B produce output **byte-identical** to
their single-GPU runs over 48–64 greedy tokens. Qwen3.5-9B tracks the single-GPU
run and diverges at a paraphrase point around token 25 — the TP path composes
per-block fused graphs where the single-GPU path uses one whole-model graph, so
the quantized matmuls round differently; both continuations are correct.

### AllReduce precision

ggml-cuda's internal collective converts F32 payloads to BF16 before reducing,
with a default threshold of 1 byte — i.e. always. For the ~10 KB collectives a
decode step issues, halving the transfer saves nothing and costs 16 mantissa
bits on the residual stream. `TSGgml_TensorParallelInit` raises the threshold to
1 MB, so decode reduces exactly and only megabyte-scale prefill payloads take
the trade — where it is worth 2.4× (Gemma-4-E4B 512-token prefill: 1038 → 2539
tok/s, output still byte-identical). `GGML_CUDA_AR_BF16_THRESHOLD` overrides it;
`0` disables the round-trip entirely.

### Traps, in the order they cost time

1. **Fused kernels upload weights with a raw `ggml_backend_tensor_set`,
   bypassing `resolve_upload_source`.** Harmless on rank 0, where every
   quantized weight is preloaded; on rank 1 the *replicated* weights (Gemma 4's
   PLE gate/proj) miss the cache and the fallback "upload" dereferences a
   GCHandle — segfault inside `cudaMemcpyAsync`. Every `upload_list` site in the
   native tree now resolves the key first.
2. **Do not invalidate the KV cache after a fused TP block.**
   `InvalidateTensorDeviceCache` drops the device copy on every rank, so the
   next layer re-uploads the whole cache (16 MB per layer per rank at an 8K
   context) — and re-uploads the *stale* host copy, because the fused kernel
   wrote only the device one. Decode 15.6 → 11.4 tok/s. Keep KV device-resident
   and sync back only when falling back to the per-op path.
3. **A fresh `ggml_backend_alloc_ctx_tensors` per call is a cudaMalloc/cudaFree
   pair.** Invisible in a 512-token prefill, dominant in decode, where the
   per-layer kernel runs once per rank per token. Small batches take the
   persistent per-rank reuse buffer, large ones `gallocr`'s lifetime packing:
   11.4 → 18.3 tok/s.
4. **The replicated activation is one buffer PER RANK, not one buffer.** The
   per-op blocks accumulate into each rank's copy independently, so a fused
   block that writes back only rank 0's leaves the others a layer behind and the
   model drifts from the first token. Each rank's graph reads and writes its own.
   (For the same reason `BroadcastTensorToAllRanks` must keep copying on GGML
   even though every rank can read the same host memory — aliasing the ranks
   would make `TpResidualAdd` apply the same residual `tp` times.)
5. **Parked graphs freed by static destructors after the CUDA driver shuts
   down** abort the process with "CUDA error: driver shutting down".
   `TSGgml_Shutdown` releases them while the backends are still alive.

`PooledContextHandle` is now movable, so a kernel can hand its context to
something that outlives the call — required for any TP graph that is not already
persistent.

### MoE trunk (Gemma-4-26B-A4B) — Implemented

The whole-model MoE kernels (`TSGgml_Gemma4MoEModelDecode` / `...Verify`)
compute the router *inside* the graph, and `ggml_top_k` returns **global**
expert ids. The whole-expert sharding the per-op TP path used cannot feed
those kernels: rank r would have to rewrite ids it does not own, and ggml-cuda's
`mul_mat_id` binds each (expert, token) pair with a `break` after the first
match and then asserts the binding count — a token that repeats an id (which any
filler scheme can produce) is a crash, not a wrong answer.

So under TP the MoE layers are sharded the Megatron way instead — *inside* each
expert. Every rank keeps all 128 experts and holds 1/tp of each expert's FFN
width: gate/up column-parallel on the intermediate dim (keeping the
`[gate_r | up_r]` fused layout), down row-parallel on it (whole quant blocks).
`sel` stays global, the graph is the single-GPU graph with narrower expert
matrices, and the expert-sum output becomes the third row-parallel partial per
layer — reduced right after the routing-weighted sum, which is linear, so
summing partials then weighting equals weighting then summing. Three AllReduce
points per layer (attention output, dense FFN down, MoE expert sum), 90 per
token on the 26B.

The slices must be materialized at load (the whole-expert view was zero-copy):
~10.5 GB, parallelized over (layer, rank, projection), ~36 s — first-touch
bound, not copy-bound. One-time cost for a 10× decode; `TS_GEMMA4_TP_FUSED_MOE=0`
falls back to the whole-expert per-op path.

Measured (2× RTX 2000 Ada, `--tp 2`, prefill 512 / decode 64, tok/s):

| | 1 GPU | TP=2 per-op | TP=2 fused |
|---|---|---|---|
| Gemma-4-26B-A4B IQ4_XS | 1845 / 48.5 | 43.0 / 4.9 | **2537 / 51.2** |

Output is **byte-identical** to the single-GPU run on every prompt tested
(three prompts × 48–96 greedy tokens, plus a 3-turn conversation).

### Known Limitations / Future Work (Stage 1c)

- [x] **Gemma 4 MoE (26B)** — done, see above.
- [ ] **Qwen 3.5's GatedDeltaNet block** still runs its norm + packed projection
      + conv + scan through `TSGgml_Qwen35GdnLayerTP` and then the rest per op.
      Folding the ssm output projection into that kernel (it is already the
      `FusedMatMulQuantAdd` shape) would remove the last per-layer round trip.
- [ ] **Qwen 3.5 MoE block** under TP still walks the router and experts per op.
- [ ] The whole-model TP graph is per-rank and single-process; multi-node keeps
      the per-op forward (`GlobalTpDegree == TpDegree` gates every fused path).

---

### Known Limitations / Future Work (Stage 1b)

- [x] **Fused TP decode.** Done — see Stage 1c.
- [x] **Qwen3.5 on GGML.** Done — see "Qwen 3.5 / 3.6 on GGML" below.
- [ ] **MoE TP throughput on other architectures.** GptOss/Nemotron still walk
      experts per token per rank. Gemma 4 and Qwen 3.5 now use expert
      parallelism; the same treatment applies to them.
- [x] **Fused multi-rank linear.** Removed in 2026-11 (it was OFF by default and
      `TS_GGML_TP_FUSED_MATMUL=1` is now a hard error). Submitting both ranks
      from one thread avoids a worker hand-off, but its graphs run
      asynchronously, so it cannot share the per-rank compute buffer and must
      allocate a backend buffer per rank per call — a cudaMalloc/cudaFree pair
      per linear. On a nearly-full card that dominates: Qwen3.5-35B decode
      measured 8.7 tok/s with it on against 20.4 tok/s with it off, and the LM
      head alone went from 44 ms/token to 1.6 ms/token. The generic path is not
      serial either — `RunPerRank` fans the ranks across worker threads — so the
      only thing given up is the in-call device AllReduce, which
      `ITensorParallelGroup.AllReduce` performs anyway. Its separate multi-node
      correctness discrepancy (wrong results when each node contributes one rank)
      is still unexplained and still worth root-causing.
- [ ] **Fused per-rank attention block.** The Qwen 3.5 TP attention layers still
      run op-at-a-time (norm, QKV, host deinterleave, QK norm, RoPE, host SDPA,
      gate, output projection). The single-GPU path has fused kernels for exactly
      this block (`TSGgml_Qwen35AttentionLayer{Decode,Prefill}`), but they fold
      the residual add in, which a row-parallel output projection cannot do
      before the AllReduce; giving them a "write the block output" mode would
      make them usable per rank. This is now the largest remaining item: 32% of
      decode and ~75% of prefill time.
### Expert parallelism for MoE (Gemma 4, GGML)

Slicing *inside* each expert (column-parallel gate/up, row-parallel down) leaves
every rank holding a piece of every expert, so the layer cannot be expressed as
one `ggml_mul_mat_id` dispatch and degenerates into a per-token, per-expert loop
— hundreds of thousands of tiny matmuls per prefill.

Whole experts partition cleanly instead. The stacked expert tensor has the
expert index as its outer dimension, so rank r's share is a contiguous byte
range (a zero-copy `StackedExpertWeights` view), and each rank runs the same
batched kernel the single-GPU path uses. Each rank sums only the experts it
owns, so the existing AllReduce over the layer output is exactly the right
recombination.

Two things this needed:

* **Distinct filler expert ids.** The kernel takes a dense `[seqLen][nUsed]`
  route table, so routes belonging to another rank are neutralised rather than
  removed. Pointing them all at expert 0 (or at `e % perRank`) produces
  *duplicate* ids within a token — which real top-k routing never does, and
  which the batched gather/scatter relies on: two destination slots end up
  claiming one source row, and the layer faults with
  `an illegal memory access was encountered` inside `launch_mul_mat_q`. Each
  filler now takes a distinct unused local id (`perRank >= nUsed` guarantees
  one exists).
* **Preloading the per-rank slices.** They are not `QuantizedWeight` shards, so
  `PrepareGgmlQuantizedWeightsForInferenceTP` gained a
  `PreloadGgmlTpAuxiliaryWeights` hook. Without it the first forward paid the
  whole upload — 51 s on Gemma-4-26B.

Measured, Gemma-4-26B-A4B IQ4_XS, `--tp 2` (per-expert loop → expert parallel):

| | before | after |
|---|---|---|
| model load | 149 s (cold) / 18.6 s (warm) | 8.3 s |
| decode warmup | 2.1 s | 0.34 s |
| time to first interactive turn | ~200 s | ~11 s |
| interactive decode | 2.2 tok/s | 4.2–5.3 tok/s |
| prefill 512 | (did not finish in 25 min) | 42.7 tok/s |

Output is **byte-identical to a single-GPU run** over a 60-token greedy
generation.

### Qwen 3.5 / 3.6 on GGML (SSM + MoE)

Qwen 3.5 was the one architecture GGML TP rejected outright: its per-rank
GatedDeltaNet ran as a single fused *CUDA* kernel and the GGML bridge exposed
only the unpacked chunked and batched-step forms, both of which take Q/K/V/Z/
beta/alpha already split apart with the conv1d and the gate arithmetic done on
the host. Under TP that shape is unusable — each rank owns a block-cyclic slice
of the V heads, so every one of those host steps would run per rank per layer
with a device round-trip on either side.

Four pieces made it work:

* **A packed per-rank GDN kernel** (`TSGgml_Qwen35GdnLayerTP`,
  `ggml_ops_qwen35_gdn_tp.cpp`). One ggml graph per rank covering input RMSNorm,
  the packed column-parallel in-projection, `ggml_ssm_conv`, q/k L2-norm and head
  tiling, `ggml_gated_delta_net`, the gated RMSNorm and the SiLU(z) gate. Folding
  the projection in removes a separate multi-rank matmul dispatch and an
  activation round-trip per recurrent layer, and there are 30 of them per token.
* **Device-resident recurrent state.** The conv window and the delta state are
  bound through the per-rank cacheable-buffer cache, keyed on the caller's host
  pointer, so they upload once and are then updated in place. Downloading the
  delta state would cost ~1 MB per layer per token per rank — ~60 MB of PCIe
  traffic per token. `ResetKVCache` drops the device copies so the host reset is
  picked up.
* **Expert-parallel MoE.** Same treatment as Gemma 4: whole experts partition
  cleanly (128 of 256 per rank), so each rank runs one batched
  `ggml_mul_mat_id` dispatch per projection instead of a per-(token, expert)
  loop. The shared expert stays Megatron-split, since every token uses it.
* **Column-parallel LM head.** The head is the largest tensor left after the
  layers are sharded (398 MB Q6_K here) and is read in full per token. The
  vocabulary is the output dimension, so each rank owns a contiguous row range
  and the "gather" is two copies into disjoint halves of the logits buffer — no
  collective at all.

**The graph-cache hazard that this design has to handle.** The per-rank GDN graph
is cached per (rank, shape) and reused by all 30 recurrent layers, with the
weights and state re-pointed per call. A ggml *view* resolves its data pointer
once, when the graph is allocated — so re-pointing only the base state tensors
left the 4D state view feeding `ggml_gated_delta_net` and both `ggml_cpy`
destinations still addressing layer 0's buffers. Every layer then read and wrote
layer 0's recurrent state. The symptom was subtle: layer 0 correct, layers 1+
diverging, no NaNs, coherent-looking gibberish. `TS_QWEN35_LAYER_TRACE=1` prints
a per-layer residual summary from both the single-GPU and TP loops, which is what
localized it; the fix re-points the views alongside their bases each call.

**Two general GGML fixes fell out of profiling this**, and they help every model
on the backend, not just this one:

* `LinearForward` bound F32 weights through the generic `Ops.Addmm` path, which
  has no weight cache and re-uploaded the whole weight per call. For a matmul
  that runs once per layer per token — an MoE router — that was the dominant
  cost of the layer: 12.8 s of a 20.2 s decode, ~4 ms each to push 2 MB for a
  [1,2048]×[2048,256] product. It now routes through the quantized entry point,
  which is device-resident.
* `GgmlBasicOps.AddmmQuant` is a direct native call, so it never passed through
  `OpRegistry.PreInvokeHook` and inherited whatever rank the calling thread was
  left on. Under TP that silently ran the LM head on the wrong GPU, missing its
  preloaded weight. It now selects the rank from its result tensor.

**Measured, Qwen3.5-35B-A3B-UD-IQ4_XS, 2× RTX 2000 Ada, `--backend ggml_cuda --tp 2`:**

| | first working version | shipped |
|---|---|---|
| decode (short prompt) | 3.9 tok/s | **20.4 tok/s** |
| prefill (23 tok) | 23.0 tok/s | **26.3 tok/s** |
| prefill (512 tok) | — | **80.2 tok/s** |
| prefill (2966 tok) | — | **75.7 tok/s** |
| decode at 3 K context | — | **12.1 tok/s** |
| VRAM | 9516 + 7833 MB | 9397 + 8032 MB |

The model is 16.6 GB and does not fit a 16 GB card, so this is the capacity case
TP exists for. Where decode time goes now (59 ms/token): attention block 32%,
MoE experts 31%, GDN 15%, ssm_out+AllReduce 6%, router 6%, LM head 2.5%.

**Correctness.** On Qwen3.5-9B Q8_0 (which fits on one card) `--tp 2` produced
text byte-identical to the single-GPU run over 60 greedy tokens with the device
AllReduce, and diverges only at a natural branch point (~18 tokens) once the
partials are summed by the host reduction instead — the expected consequence of a
different summation order. Per-layer residual traces match the single-GPU loop to
floating-point noise from layer 0 onward.

### Multi-node

Verified on the GGML CUDA backend with two processes (one GPU each) over TCP
loopback: the driver/worker lockstep, hierarchical AllReduce, and shutdown all
work, and the generated text matches both the single-GPU and the direct-CUDA
multi-node runs.

```bash
# node 0                                         # node 1
TENSORSHARP_TP_DEVICES=0 ... --tp 1 \            TENSORSHARP_TP_DEVICES=1 ... --tp 1 \
  --tp-node-id 0 --tp-peers "h0:9500,h1:9500"      --tp-node-id 1 --tp-peers "h0:9500,h1:9500"
```

---

## Stage 2 — Network Parallelism with Shared State

**Goal:** Distribute TP across multiple machines connected by a network, with
reconvergent operations and shared response/KV-cache state via a coordination
layer (Redis initially).

### Architecture

```
┌──────────────────┐         ┌──────────────────┐
│  Node 0          │         │  Node 1          │
│  ┌─────┐┌─────┐ │  TCP/   │ ┌─────┐┌─────┐  │
│  │GPU 0││GPU 1│ │  RDMA   │ │GPU 2││GPU 3│  │
│  └──┬──┘└──┬──┘ │  network│ └──┬──┘└──┬──┘  │
│     └──┬───┘    │         │    └──┬───┘     │
│  Local AllReduce│         │  Local AllReduce │
│     └──┬───┘    │         │    └──┬───┘     │
│        │        │         │       │         │
│   Rank 0-1      │         │   Rank 2-3      │
└────────┼────────┘         └───────┼─────────┘
         │                          │
         └──────────┬───────────────┘
                    │
         ┌──────────▼──────────┐
         │  Redis / Valkey     │
         │  ┌───────────────┐  │
         │  │ KV Cache Pool │  │
         │  │ Response Queue│  │
         │  │ Rank Barrier  │  │
         │  │ Weight Registry│ │
         │  └───────────────┘  │
         └─────────────────────┘
```

### Components

#### 2.1 Network Communicator (`INetworkCommunicator`)

```csharp
public interface INetworkCommunicator
{
    int Rank { get; }
    int WorldSize { get; }

    // Collective operations (blocking, all ranks must call).
    void AllReduce(Span<float> buffer);
    void AllGather(Span<float> send, Span<float> recv);
    void Barrier();

    // Point-to-point.
    void Send(int destRank, ReadOnlySpan<byte> data);
    void Recv(int srcRank, Span<byte> buffer);
}
```

Initial implementation: TCP sockets with a simple framing protocol.
Each node runs a `NetworkCommunicatorServer` thread that handles
incoming connections from peer ranks.

#### 2.2 Hierarchical AllReduce

For multi-node TP, AllReduce is decomposed into two phases:

1. **Intra-node**: Local P2P AllReduce (Stage 1 code) reduces within each node
2. **Inter-node**: Network AllReduce across node representatives (rank 0 of each node)
3. **Intra-node broadcast**: Result propagated to local peers

This minimizes network traffic: only `1/tp_local` of the data crosses the network.

#### 2.3 Shared KV Cache (Redis)

The KV cache is stored in Redis as binary blobs keyed by
`kv:{session_id}:{layer}:{rank}`. This enables:

- **Session migration**: a request can be served by any node that
  fetches the KV cache from Redis
- **Prefix caching**: shared prefixes are stored once and referenced
  by multiple sessions
- **Crash recovery**: KV state survives node restarts

```
Redis key layout:
  kv:{session}:meta          → JSON { layers, heads, headDim, seqLen, dtype }
  kv:{session}:L{l}:R{r}:K  → binary blob (numKVHeads/tp × seqLen × headDim × dtypeSize)
  kv:{session}:L{l}:R{r}:V  → binary blob
```

Optimization: use Redis `MEMORY` commands and pipelining to batch
layer transfers. For large caches, use Redis Streams or a dedicated
binary protocol (Stage 3).

#### 2.4 Shared Response Queue

Generated tokens are published to a Redis Stream
`response:{session_id}` so that:

- Any API server node can stream tokens to the client regardless of
  which compute node produced them
- Multiple consumers (logging, metrics) can subscribe independently

#### 2.5 Rank Coordination

A Redis-based barrier and rank registry:

```
tp:group:{group_id}:ranks  → SET of "node:rank" members
tp:group:{group_id}:barrier → INCR/DECR counter for sync
tp:group:{group_id}:config → JSON { worldSize, localTp, nodeCount }
```

### Changes Required (Stage 2 — Implemented)

| Component | Change |
|-----------|--------|
| `TensorSharp.Backends.Cuda/ITensorParallelGroup.cs` | **New.** Interface abstracting local and distributed TP groups |
| `TensorSharp.Backends.Cuda/TensorParallelGroup.cs` | Implements `ITensorParallelGroup`; adds `GlobalDegree`, `GlobalRankOffset`, `NodeCount` |
| New project: `TensorSharp.Distributed` | TCP communicator, distributed TP group, config parsing |
| `TensorSharp.Distributed/TcpCommunicator.cs` | **New.** TCP mesh with length-prefixed framing; AllReduce, Broadcast, Barrier |
| `TensorSharp.Distributed/DistributedTensorParallelGroup.cs` | **New.** Hierarchical AllReduce: local P2P → TCP → local broadcast |
| `TensorSharp.Distributed/DistributedTpConfig.cs` | **New.** Peer endpoint parsing, env-var configuration |
| `TensorSharp.Models/ModelBase.cs` | `ITensorParallelGroup` field, `GlobalTpDegree`/`TpRankOffset` properties, multi-node weight sharding |
| `TensorSharp.Cli/Program.cs` | `--tp-node-id`, `--tp-peers` arguments |
| `TensorSharp.Server/ModelLifecycleService.cs` | `TENSORSHARP_TP_NODE_ID`, `TENSORSHARP_TP_PEERS` env-var support |

### Configuration

```bash
# CLI: 2-node tensor parallelism (each node has 2 GPUs)
# Node 0:
TensorSharp.Cli --model qwen3.5-9b.gguf --backend cuda --tp 2 \
  --tp-node-id 0 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# Node 1:
TensorSharp.Cli --model qwen3.5-9b.gguf --backend cuda --tp 2 \
  --tp-node-id 1 --tp-peers "192.168.1.10:9500,192.168.1.11:9500"

# Server: via environment variables
# Node 0:
TENSORSHARP_TP_DEGREE=2 TENSORSHARP_TP_NODE_ID=0 \
TENSORSHARP_TP_PEERS=192.168.1.10:9500,192.168.1.11:9500 \
  dotnet run --project TensorSharp.Server.Host

# Config JSON (auto-expanded to CLI args)
{ "tp": 2, "tp-node-id": 0, "tp-peers": "192.168.1.10:9500,192.168.1.11:9500", "backend": "cuda" }
```

---

## Stage 3 — RDMA Memory Access

**Goal:** Replace TCP-based inter-node communication with RDMA
(Remote Direct Memory Access) for microsecond-latency collective
operations, if profiling shows network latency is the bottleneck.

### When RDMA Helps

RDMA is beneficial when:
- Inter-node AllReduce latency dominates compute time (small batch decode)
- Network bandwidth is the bottleneck for KV cache transfers
- Tail latency matters (real-time serving with strict SLAs)

RDMA is **not** beneficial when:
- Compute dominates (large-batch prefill)
- Network is already fast enough (100Gbps+ TCP with kernel bypass)
- Hardware doesn't support it (consumer GPUs, no InfiniBand/RoCE NICs)

### Architecture

```
┌──────────────────┐         ┌──────────────────┐
│  Node 0          │         │  Node 1          │
│  GPU 0 ←─ GPUDirect RDMA ─→ GPU 2            │
│  GPU 1 ←─ GPUDirect RDMA ─→ GPU 3            │
│                  │         │                  │
│  (NIC registers GPU VRAM   │                  │
│   for zero-copy transfers) │                  │
└──────────────────┘         └──────────────────┘
```

### Components

#### 3.1 RDMA Transport

Two options depending on hardware:

| Transport | Hardware | API | Latency |
|-----------|----------|-----|---------|
| InfiniBand | Mellanox ConnectX + IB switch | libibverbs | ~1-2 µs |
| RoCE v2 | Any RDMA-capable NIC + Ethernet | libibverbs over UDP | ~2-5 µs |
| iWARP | Intel/Chelsio NICs + Ethernet | librdmacm | ~5-10 µs |

On Windows, use the WinRDMA API (`ndis.sys` NDK) or fall back to
`NetworkDirect` (Intel/Chelsio). On Linux, use `libibverbs` directly.

#### 3.2 GPUDirect RDMA

NVIDIA GPUDirect RDMA allows the NIC to read/write GPU VRAM directly,
bypassing the CPU and system memory:

```
GPU VRAM → PCIe → NIC → Network → NIC → PCIe → GPU VRAM
```

Requires:
- NVIDIA GPU with PCIe BAR1 mapping (all datacenter GPUs, some consumer)
- NIC on the same PCIe switch/root complex as the GPU
- `nvidia-peermem` kernel module (Linux) or NDK (Windows)

#### 3.3 NCCL Integration (Linux)

On Linux, the simplest path is NCCL, which handles RDMA transparently:

```csharp
// P/Invoke to libnccl.so
[DllImport("nccl")]
static extern int ncclAllReduce(IntPtr sendbuff, IntPtr recvbuff,
    long count, ncclDataType_t datatype, ncclRedOp_t op,
    IntPtr comm, IntPtr stream);
```

NCCL auto-detects the best transport (NVLink > PCIe P2P > IB/RoCE > TCP)
and handles GPUDirect RDMA setup.

#### 3.4 Custom RDMA (Windows / Fine-grained Control)

For Windows or when NCCL isn't suitable:

1. **Memory registration**: Register GPU buffers with the NIC via
   `ndkRegisterBuffer` (Windows NDK) or `ibvRegMr` (Linux verbs)
2. **Queue pairs**: Create RDMA queue pairs between ranks
3. **RDMA Write/Read**: One-sided operations for AllReduce:
   - Each rank writes its partial to a remote rank's registered buffer
   - Remote rank sums in-place after a completion notification
4. **Completion queue**: Poll for operation completion

### Changes Required

| Component | Change |
|-----------|--------|
| `TensorSharp.Distributed` | Add `RdmaCommunicator` implementing `INetworkCommunicator` |
| `TensorSharp.Backends.Cuda` | Add NCCL P/Invoke bindings (Linux), GPUDirect buffer registration |
| New: `TensorSharp.Rdma` | Low-level RDMA bindings (libibverbs / WinNDK) |
| `TensorSharp.Models/ModelBase.cs` | Transport selection: auto-detect best available |

### Decision Criteria

Before implementing Stage 3, profile Stage 2 with TCP:

| Metric | TCP Sufficient | RDMA Needed |
|--------|---------------|-------------|
| AllReduce latency (per layer) | < 100 µs | > 500 µs |
| KV cache transfer (1K tokens) | < 1 ms | > 5 ms |
| Decode throughput degradation | < 10% vs local TP | > 30% |
| Tail latency (P99) | < 2× median | > 5× median |

If TCP meets the latency targets, Stage 3 adds complexity without
meaningful user-facing improvement.

---

## Implementation Priority

```
Stage 1 (Local TP)          ████████████████████ NEARLY COMPLETE
  ├── Mistral3               ████████████████████ DONE
  ├── Gemma4 (dense+MoE)     ████████████████████ DONE
  ├── Qwen3.5 (SSM+MoE)     ████████████████████ DONE
  ├── GptOss (MoE)           ██████████████████░░ DONE (down-bias gap)
  ├── Nemotron (SSM+MoE)     ████████████████████ DONE (Mamba2 replicated)
  ├── DiffusionGemma         ──────────────────── N/A (diffusion model)
  ├── QwenImage              ──────────────────── N/A (image generation)
  ├── Batched forward TP     ████████████████████ DONE (Mistral3; MoE models fall back to per-seq)
  └── NCCL (Linux)           ░░░░░░░░░░░░░░░░░░░░ TODO

Stage 2 (Network TP)        ██████████░░░░░░░░░░ IN PROGRESS
  ├── TCP communicator       ████████████████████ DONE
  ├── Hierarchical AllReduce ████████████████████ DONE
  ├── ITensorParallelGroup   ████████████████████ DONE
  ├── ModelBase multi-node   ████████████████████ DONE
  ├── CLI --tp-node-id/peers ████████████████████ DONE
  ├── Server env-var config  ████████████████████ DONE
  ├── Model-specific sharding████████████████░░░░ IN PROGRESS
  ├── Redis KV cache         ░░░░░░░░░░░░░░░░░░░░ DEFERRED (direct TCP instead)
  ├── Redis response queue   ░░░░░░░░░░░░░░░░░░░░ DEFERRED (direct TCP instead)
  └── Multi-node server      ████████████████████ DONE

Stage 3 (RDMA)              ░░░░░░░░░░░░░░░░░░░░ CONDITIONAL
  ├── Profile Stage 2 first  ░░░░░░░░░░░░░░░░░░░░
  ├── NCCL integration       ░░░░░░░░░░░░░░░░░░░░
  └── Custom RDMA transport  ░░░░░░░░░░░░░░░░░░░░
```

### Model TP Feasibility

| Model | Architecture | TP Status | Notes |
|-------|-------------|-----------|-------|
| Mistral3 | Dense transformer | ✅ Done | Fused/separate QKV, YaRN RoPE |
| Gemma4 | Dense + MoE | ✅ Done | Expert slicing, dual dense+MoE FFN, per-layer head dims, shared KV layers |
| Qwen3.5 | SSM + MoE | ✅ Done | GatedDeltaNet SSM with per-rank V-head ownership, packed GDN kernels on direct CUDA and GGML; expert-parallel MoE + column-parallel LM head on GGML |
| GptOss | MoE | ✅ Done | Expert slicing with biased projections, attention sinks, YaRN; expert down-bias skipped in TP |
| Nemotron | SSM + MoE | ✅ Done | Mamba2 replicated on rank 0, attention (no RoPE), MoE expert slicing |
| Muse-Glimmer | Dense + vision | ✅ Done | Interleaved SWA/NoPE layers, per-head QK norm (replicated), attention output gate (column-parallel by head), segmented fused gate_up; **tp=2 max — 2 KV heads**; vision tower replicated on rank 0; DFlash and pooled KV snapshots stay single-GPU |
| Muse Spark | MoE + vision | ✅ Done (native + per-op path) | **Native** (`--tp N` on `ggml_cuda`): the whole-model executor runs on upstream ggml's meta backend (the mechanism behind llama.cpp's `-sm tensor`) with a TensorSharp split plan (`ggml_ops_muse_spark_tp.cpp`): attention by KV group (16 groups over N GPUs, rotating per layer), routed experts along n_ff in 256-element units, output head by vocabulary rows; norms, router and latent projections on every GPU; two all-reduces per layer; N at most 16; local only. On ten A40s it decoded 22.63 tok/s, 1.40x llama.cpp's `-sm tensor` ([report](docs/perf/muse-spark-vs-llama-cpp.md)). **Per-op** (`TS_MUSE_SPARK_NATIVE=0`, and `--tp N` on `ggml_vulkan`): Q/K/V/gate split by whole GQA head groups, attention output by input columns, routed experts partitioned whole with per-expert norm before the cross-GPU sum; uneven partitions allowed (9 GPUs for 128 query / 16 KV heads); router, latent projections and output head on rank 0; local only |
| DiffusionGemma | Diffusion | ❌ N/A | Not autoregressive text generation |
| QwenImage | Image gen | ❌ N/A | Not autoregressive text generation |
