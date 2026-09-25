// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include <cstddef>
#include <cstdint>

// ABI mirrored by QwenImage21Native.cs. Weights retain their original GGUF type.
struct TSGQi21Weight {
    void* data;
    std::int32_t type, reserved;
    std::int64_t ne0, ne1, bytes;
};
struct TSGQi21Block {
    TSGQi21Weight q, k, v, out, gate, up, down;
    void* norm_q;
    void* norm_k;
};
struct TSGQi21Segment {
    std::int32_t start, end, source_start, is_image;
};

// A LoRA update of one projection, applied unmerged: y = row_scale * (W x) + up (down x).
// The quantized base weight is never modified. Merging a distillation LoRA's tiny delta
// into Q8_0 rounds most of it away, so the update keeps its own F16/F32 factors.
//   down: ggml [ne0 = in,   ne1 = rank] (PyTorch lora_A / lora_down [rank, in])
//   up:   ggml [ne0 = rank, ne1 = out]  (PyTorch lora_B / lora_up [out, rank]) with every
//         scale (strength, alpha / rank, DoRA magnitude) already folded in.
//   row_scale: optional F32 [out] multiplying the base projection (DoRA), else null.
// Several LoRAs on one projection are concatenated along the rank by the caller. Updates
// of projections that read the same input may share one contiguous `down` allocation
// (q, k, v and gate, up): the graph then runs one stacked shrink for all of them.
struct TSGQi21Lora {
    void* down;
    void* up;
    float* row_scale;
    std::int32_t type;   // GGML_TYPE_F16 or GGML_TYPE_F32, for both factors
    std::int32_t rank;   // 0 = no low-rank term
    std::int64_t in, out;
};
struct TSGQi21BlockLora {
    TSGQi21Lora q, k, v, out, gate, up, down;
};

// Everything a LoRA plug-in changes in the transformer (TSGQi21Desc::adapter).
// A fused gate_up weight is described by the caller as its two halves (block
// weights `gate` and `up` as row views) whenever gate or up carries an update.
struct TSGQi21Adapter {
    std::int32_t struct_bytes, num_layers;
    TSGQi21Lora image_in, text_in, text_out, time_in, time_out, modulation, norm_out, proj_out;
    const TSGQi21BlockLora* blocks;   // num_layers entries, or null
    // Optional replacement of proj_out's weight for this call ([in = dim, out = channels],
    // F16 or F32), e.g. a per-step head of a parallel-decoding distillation. It is uploaded
    // as an input every call, so switching heads between steps keeps the retained graph
    // and the prefix cache.
    const void* output_head;
    std::int32_t output_head_type, reserved;
};

// Storage of a prefix KV cache. AUTO stores what the attention kernel consumes
// (F16 for Metal and CUDA flash attention, F32 otherwise), so reading the cache
// reproduces the uncached computation. Q8_0 stores K and V in 8 bits and Q8_0_V
// only V, the counterparts of vLLM-Omni's "fp8" and "fp8_v" prefix caches; both
// are dequantized to the attention type each step, so only the prefix rounds.
enum TSGQi21PrefixCacheType : std::int32_t {
    TSG_QI21_PREFIX_AUTO = 0,
    TSG_QI21_PREFIX_F32 = 1,
    TSG_QI21_PREFIX_F16 = 2,
    TSG_QI21_PREFIX_Q8_0 = 3,
    TSG_QI21_PREFIX_Q8_0_V = 4,
};

// Nonzero TSGgml_QwenImage21Forward results: which graph produced the output.
enum TSGQi21ForwardPath : std::int32_t {
    TSG_QI21_PATH_FULL = 1,      // whole sequence, no cache requested
    TSG_QI21_PATH_EXTRACT = 2,   // whole sequence; prefix K/V stored in the cache
    TSG_QI21_PATH_CACHED = 3,    // target tokens only, attending to the cached prefix
    TSG_QI21_PATH_DECLINED = 4,  // whole sequence; the cache did not fit the device
};

struct TSGQi21Desc {
    const float* images;
    const float* text;
    const float* time_embedding;
    const float* cos;
    const float* sin;
    float* output;
    TSGQi21Weight image_in, text_in, text_out, time_in, time_out, modulation, norm_out, proj_out;
    void* text_norm;
    const TSGQi21Block* blocks;
    const TSGQi21Segment* segments;
    std::int32_t struct_bytes, dim, heads, head_dim, channels, text_dim;
    std::int32_t image_seq, text_seq, total_seq, prefix_seq, num_layers, num_segments;
    float eps;
    // Text and reference-image tokens are modulated at t=0, so their per-layer
    // K/V do not depend on the denoising step. A nonzero key names one request's
    // prefix: the first forward with a key stores the prefix K/V, later ones run
    // only the target tokens. The caller must not reuse a key for other text,
    // references or layout; release it with TSGgml_QwenImage21ReleasePrefixCache.
    std::uint64_t prefix_cache_key;
    std::int32_t prefix_cache_type;
    // Tensor parallelism (TSGgml_QwenImage21ForwardTp): the number of ranks the
    // block weights are sharded over, 0 or 1 when unsharded. A rank holds whole
    // attention heads (heads * head_dim * tp_ranks == dim) and a slice of the
    // MLP; to_out and img_mlp.out are row-parallel partial sums that the ranks
    // all-reduce, twice per block. Everything outside the blocks is replicated.
    std::int32_t tp_ranks;
    // Optional LoRA plug-in (null = the checkpoint as stored). Its buffers are keyed like
    // weights, so a different adapter retires retained graphs and stored prefix K/V.
    const TSGQi21Adapter* adapter;
};

// TSGgml_QwenImage21GetPrefixCacheInfo. state: 0 = no cache for the key,
// 1 = stored, 2 = declined (did not fit). Types are ggml_type values.
struct TSGQi21PrefixCacheInfo {
    std::int32_t state, key_type, value_type, tokens;
    std::int64_t bytes;
};

// Pinned by QwenImageNativeAbiTests on the managed side.
static_assert(sizeof(void*) != 8 || sizeof(TSGQi21Desc) == 472, "QwenImage21ForwardArgs layout");
static_assert(sizeof(void*) != 8 || offsetof(TSGQi21Desc, prefix_cache_key) == 448, "QwenImage21ForwardArgs layout");
static_assert(sizeof(void*) != 8 || offsetof(TSGQi21Desc, adapter) == 464, "QwenImage21ForwardArgs layout");
static_assert(sizeof(void*) != 8 || sizeof(TSGQi21Lora) == 48, "QwenImage21Lora layout");
static_assert(sizeof(void*) != 8 || sizeof(TSGQi21BlockLora) == 336, "QwenImage21BlockLora layout");
static_assert(sizeof(void*) != 8 || sizeof(TSGQi21Adapter) == 416, "QwenImage21Adapter layout");
static_assert(sizeof(TSGQi21PrefixCacheInfo) == 24, "QwenImage21PrefixCacheInfo layout");
