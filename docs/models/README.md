# Model Architecture Cards

[English](README.md) | [中文](README_zh-cn.md)

This folder is the canonical, per-model reference for every architecture that
TensorSharp can run. Each card is a self-contained brief: it walks an engineer
or researcher from "I have never heard of this model" all the way to "I can
explain the forward graph and reproduce the inference path in TensorSharp." If
you only need a top-level pointer, use the table below; otherwise jump into the
individual cards.

## What every card contains

Each card follows the same shape so you can diff architectures cleanly:

1. **Origin and intent** — who designed the model, what the GGUF arch keys are,
   and which capabilities (modalities, thinking, tools) it exposes.
2. **Model architecture** — the high-level block diagram, layer counts, and any
   per-layer heterogeneity.
3. **Forward graph** — the exact ordered list of ops a single token (decode) and
   a multi-token sequence (prefill) flow through, including residuals and
   normalizations.
4. **Components** — every sub-block (attention, FFN/SSM, routing, normalization,
   RoPE flavor, vision/audio encoder) explained in detail with the math that
   governs it.
5. **Parameters and settings** — the GGUF metadata keys, weight tensor naming
   convention, and dtype expectations.
6. **TensorSharp implementation** — pointers to the C# source files, the
   instantiation order, the cache layout, and the way the model plugs into
   `ModelBase` / `Ops` / native GGML kernels.
7. **Prefill optimization** — chunking, fused per-layer kernels, parallelization,
   cross-layer caches.
8. **Decode optimization** — fused single-call kernels, pre-resolved weight
   pointers, batched MoE, in-place kernels, cache reuse.
9. **Memory and KV cache strategy** — circular vs. linear caches, mmap-backed
   weights, pre-allocated decode buffers.
10. **Multimodal pipeline** — how images / audio / video are processed,
    encoded, and injected into the language model.
11. **Output / chat template** — protocol parser, stop tokens, thinking / tool
    formats.
12. **Optimization opportunities** — work that has not been done yet but that
    we know would unlock more performance or capability.

## Implementation matrix

| Architecture | Card | Source class | GGUF keys | Modalities | Reasoning | Tools | Batched / paged forward | Notable acceleration |
|---|---|---|---|---|---|---|---|---|
| Gemma 3 | [gemma3.md](gemma3.md) | `Gemma3Model` | `gemma3` | Text, image | No | No | No (legacy per-seq) | Alternating SWA / global attention, GeGLU FFN, QK-norm, V-norm |
| Gemma 4 | [gemma4.md](gemma4.md) | `Gemma4Model` | `gemma4` | Text, image, video, audio | Yes | Yes | **Default** (toggle off with `TS_GEMMA4_BATCHED=0`) | Single-graph fused decode (all layers in one GGML dispatch), fused per-layer prefill, chunked prefill, circular SWA cache, PLE, KV sharing, MoE variants. Batched path matches legacy logits within FP noise (`Gemma4BatchedForwardTests`); reaches ~1.5× legacy at batch=8 and ~1.6× at 4×800-token prompts. |
| Qwen 3 | [qwen3.md](qwen3.md) | `Qwen3Model` | `qwen3` | Text | Yes | Yes | Reference port (`Qwen3Model.BatchedForward.cs`) — exercised by `Qwen3BatchedForwardTests` when a base-Qwen3 GGUF is provided | Native whole-model decode with pre-resolved weight pointers |
| Qwen 3.5 / 3.6 family | [qwen35.md](qwen35.md) | `Qwen35Model` | `qwen35`, `qwen35moe`, `qwen3next` | Text, image | Yes | Yes | **Default** (toggle off with `TS_QWEN35_BATCHED=0` or `--no-continuous-batching`). Per-slot recurrent-state pool + optional native GatedDeltaNet kernel (`TS_QWEN35_BATCHED_GDN_NATIVE=1`) | Hybrid FullAttention + GatedDeltaNet recurrent, fused attention layer decode, fused prefill attention, fused output-projection + FFN, fused output-projection + norm + router, batched MoE (routed + shared + residual in a single kernel), fused vision encoder blocks |
| GPT OSS | [gptoss.md](gptoss.md) | `GptOssModel` | `gptoss`, `gpt-oss` | Text | Yes (always) | Yes | **Default** (toggle off with `TS_GPTOSS_BATCHED=0`). Per-head attention sinks via `TSGgml_PagedAttentionForwardWithSinks` (or `TS_GPTOSS_PAGED_ATTN_MANAGED=1` for the C# fallback). 100% greedy match vs legacy in `GptOssBatchedCorrectnessTests`. | Stacked MoE prefill kernel (mul_mat_id + add_id + swiglu_oai), attention sinks, MXFP4 expert weights |
| Nemotron-H | [nemotron.md](nemotron.md) | `NemotronModel` | `nemotron_h`, `nemotron_h_moe` | Text, image (Omni-class) | Yes | Yes | **Default** (toggle off with `TS_NEMOTRON_BATCHED=0`). Per-slot Mamba2 conv + SSM state pool; optional native batched Mamba2 step (`TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1`). 100% greedy match vs legacy; up to 3.95× tps at batch=3 on Apple M4 Pro. | Mamba2 + attention + MoE FFN hybrid stack, batched GPU MoE, optional Parakeet audio frontend, RADIO/v2_vl image encoder |
| Mistral 3 | [mistral3.md](mistral3.md) | `Mistral3Model` | `mistral3` | Text, image | No | No | **Default** — reference IBatchedPagedModel implementation. End-to-end validated on Ministral-3-14B; native paged-attention kernel is ~21% faster than the legacy per-seq path on long context. | YaRN-corrected RoPE with position-dependent Q scaling, fused QKV / gate_up, Pixtral vision encoder |

## Backend notes

Model code is intentionally backend-agnostic. `ModelBase` selects tensor
storage through `BackendType` and the registered execution plan, then delegates
the actual ops to the backend that owns those allocators:

| Backend type | Package | Notes |
|---|---|---|
| `Cpu` | `TensorSharp.Core` | Pure managed tensors with SIMD/managed quantized fast paths (RMSNorm, RoPE, softmax, fused activations, GEMM, dequant). |
| `Cuda` | `TensorSharp.Backends.Cuda` | Direct CUDA Driver-API allocator and storage, cuBLAS GEMM, PTX kernels for hot ops (RMSNorm, softmax, RoPE/RoPEEx, SDPA, GQA prefill/decode, causal mask, gather/concat, activation fusions), native quantized matmul / get_rows for supported quant types, CPU fallback for ops that are not yet implemented. |
| `Mlx` | `TensorSharp.Backends.MLX` | Apple Silicon `mlx-c` bridge with quantized / fused / compiled kernels, async worker dispatch, MoE expert offload, and a CPU fallback layer. Requires `libmlxc`. |
| `GgmlCpu` / `GgmlMetal` / `GgmlCuda` | `TensorSharp.Backends.GGML` + `TensorSharp.GGML.Native` | Native ggml bridge with quantized graph dispatch and platform backends. mmap-backed quantized weights are bound zero-copy through host-pointer buffers. Includes the paged-attention kernel (`TSGgml_PagedAttentionForward`, plus the GPT OSS sinks variant) that powers the batched / paged execution path. |

When a card mentions a fused GGML kernel (for example `Qwen35AttentionLayerDecode`,
`Gemma4LayerPrefill`, or `MoEExpertsSwiGLUResidual`), the kernel is compiled from
`TensorSharp.GGML.Native/ggml_ops_*.cpp` and exposed through
`TensorSharp.Backends.GGML/GgmlBasicOps.cs`. The native bridge is the place to
look when a fused path engages on GGML CPU / Metal / CUDA but not on the pure
managed CPU or direct CUDA backends.

## Continuous batching & paged KV cache

All architectures listed above also run through the shared
`InferenceEngine` + `ContinuousBatchScheduler` + `BatchExecutor` stack documented
in [`docs/PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md`](../PAGED_ATTENTION_AND_CONTINUOUS_BATCHING.md).
Models that implement `IBatchedPagedModel.ForwardBatch` execute one batched
forward per scheduler step (with `slotMapping`-based K/V scatter into a
shared paged buffer and per-sequence attention via the native paged kernel);
the others run through the per-sequence KV-swap fallback inside the same engine.
The opt-in env vars are summarised in the matrix above and in the project root
README.

## Architecture comparison

| Feature | Gemma 3 | Gemma 4 | Qwen 3 | Qwen 3.5 / 3.6 family | GPT OSS | Nemotron-H | Mistral 3 |
|---|---|---|---|---|---|---|---|
| Layer type | Dense | Dense / MoE | Dense | Hybrid (Attn + Recurrent) ± MoE | MoE | Hybrid (Mamba2 + Attn + MoE FFN) | Dense |
| Attention | SWA + Global | SWA + Global | Full GQA | Full GQA + Sigmoid Gate | Full + Sinks | Full GQA (no RoPE) | Full GQA |
| FFN activation | GeGLU | GeGLU | SwiGLU | SwiGLU | SiLUAlphaLimit (clamped GLU) | ReLU² | SwiGLU |
| RoPE variant | NeoX (dual base) | NeoX + proportional / partial | NeoX | NeoX / MRoPE | NeoX + YaRN | None | GPT-J + YaRN |
| QK-norm | Yes | Yes | Yes | Yes | No | No | No |
| V-norm | No | Yes (unweighted) | No | No | No | No | No |
| Bias in projections | No | No | No | No | Yes (all linear) | No | No |
| Per-layer scaling | No | Yes | No | No | No | No | No |
| Per-Layer Embedding (PLE) | No | Yes | No | No | No | No | No |
| KV sharing | No | Yes (tail layers) | No | No | No | No | No |
| Attention sinks | No | No | No | No | Yes | No | No |
| Circular KV cache | No | Yes (SWA layers) | No | No | No | No | No |
| SSM / recurrent layers | No | No | No | Yes (GatedDeltaNet) | No | Yes (Mamba2) | No |
| Shared experts | No | No | No | Yes (qwen35moe / qwen3next) | No | Yes (optional) | No |
| Latent bottleneck FFN | No | No | No | No | No | Yes (optional) | No |
| Position-dependent Q scaling | No | No | No | No | No | No | Yes (with YaRN) |
| Vision | Yes | Yes | No | Yes | No | Yes (Omni) | Yes (Pixtral) |
| Audio | No | Yes | No | No | No | Yes (Parakeet, when mmproj present) | No |
| Video | No | Yes | No | No | No | No | No |
| Thinking | No | Yes | Yes | Yes | Yes (always) | Yes | No |
| Tool calling | No | Yes | Yes | Yes | Yes | Yes | No |
| Fused QKV | No | Yes | Yes | Mixed (full attention layers split, recurrent layers fuse a 5-way pack) | Yes | Yes | Yes |
| Fused single-graph decode | No | Yes (Gemma4ModelDecode) | Yes (TransformerModelDecode, native loop) | Per-layer fused (Qwen35AttentionLayerDecode, FusedOutProjFFN, FusedOutProjNormRouter) | Per-layer | Per-layer / batched MoE | No |
| Fused single-graph prefill | No | Yes (Gemma4LayerPrefill, dense layers) | No | Yes (FusedPrefillAttention, FusedOutProjFFN, MoE prefill) | Yes (MoE prefill via mul_mat_id) | No | No |
| Batched GPU MoE | n/a | Pending | n/a | Yes (routed + shared + residual fused) | Yes (stacked weight slabs) | Yes | n/a |
| Fused vision encoder | n/a | Standard | n/a | Yes (FusedVisionAttention + FusedVisionMLP) | n/a | Standard (RADIO ViT) | Standard (Pixtral) |
| Output parser | `PassthroughOutputParser` | `Gemma4OutputParser` | `Qwen3OutputParser` | `Qwen35OutputParser` | `HarmonyOutputParser` (always required) | `Qwen3OutputParser` | `PassthroughOutputParser` |

## Adding a new architecture

When you add a new model:

1. Create `TensorSharp.Models/Models/<Name>/<Name>Model.cs` inheriting
   `ModelBase`.
2. In the constructor: read GGUF metadata via `_gguf.GetXxx()`, call
   `ParseBaseConfig()` and `ParseTokenizer()`, call `LoadWeights()`, fuse
   weights, then initialize caches.
3. Implement `Forward(int[] tokens) → float[]`: embedding → optional
   multimodal injection → transformer blocks → final norm → LM head → logit
   copy.
4. Implement `ResetKVCache()` and `Dispose()`. Implement `TruncateKVCache()`
   when KV-cache reuse is supported.
5. Register in `ModelBase.Create()` switch expression in
   `TensorSharp.Models/ModelBase.cs`.
6. Add an `IOutputParser` implementation in
   `TensorSharp.Runtime/OutputParser.cs` if the model uses a non-standard
   output format and register it in `OutputParserFactory.Create()`.
7. Add chat template support in `TensorSharp.Runtime/ChatTemplate.cs` /
   `Jinja2Template.cs` if the model uses a novel template format.
8. Add a card under `docs/models/<name>.md` (and `<name>_zh-cn.md` if you want
   bilingual coverage), update this README's matrix, and link the card from
   the project root README.
9. Update `TensorSharp.Server/testdata/` capability gates if the model exposes
   new modalities, thinking, or tool capabilities.
