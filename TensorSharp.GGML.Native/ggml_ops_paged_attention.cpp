// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Native paged-attention kernel. For each sequence in a batch, gathers
// K and V from the per-layer paged buffer into a contiguous scratch
// buffer (C++ memcpy walking the sequence's block table), then drives
// `ggml_flash_attn_ext` on the contiguous K/V. The Metal/CUDA backend
// then runs its highly-optimised fused flash-attention kernel.
//
// Goes through the same end-to-end path as the legacy single-sequence
// `flash_attn_decode_impl`, but extended to accept variable
// `(num_query_tokens, total_seq_len, first_query_position)` per sequence
// so the same call covers both decode and prefill-chunk steps. The
// gather happens once per sequence per layer in C++, NOT in the
// managed-C# `TensorPagedAttention` path, eliminating the
// `float[] -> Tensor` materialisation cost and the AddmmBatch/softmax
// kernel-launch overhead of the Tensor-based fallback.
//
// Algorithm per sequence:
//   1. Gather K and V at the slots given by the block table -> [head_dim,
//      num_kv_heads, seq_len] contiguous host buffer.
//   2. Build a small ggml graph: permute Q to [head_dim, num_q,
//      num_heads], permute K/V to [head_dim, seq_len, num_kv_heads],
//      optionally build a causal mask, call `ggml_flash_attn_ext`,
//      permute back and copy into the output slot.
//   3. Execute on the active backend and download the result.
//
// Future optimisations (not in this first cut):
//   - One graph for the whole batch (today: per-seq graph compile).
//   - Replace the C++ gather with `ggml_get_rows` so the gather runs on
//     the GPU and overlaps with the matmul.
//   - Quantised paged K/V (Q8_0 / Q4_0) to halve bandwidth.

#include "ggml_ops_internal.h"

using namespace tsg;

namespace
{
    constexpr int k_default_alignment = 256; // matches ggml flash-attn pad multiple

    // Same predicate as ggml_ops_transformer.cpp's flash_attn_requires_masked_padding.
    // Duplicated here to keep this translation unit self-contained.
    inline bool paged_flash_attn_requires_masked_padding(int head_dim)
    {
        return head_dim == 512 || head_dim == 576;
    }

    inline int paged_padded_kv_length(int valid_len, int head_dim)
    {
        if (!paged_flash_attn_requires_masked_padding(head_dim))
            return valid_len;
        return ((valid_len + k_default_alignment - 1) / k_default_alignment) * k_default_alignment;
    }

    inline void fill_causal_mask_fp16(
        std::vector<ggml_fp16_t>& mask,
        int padded_kv_len,
        int num_q,
        int seq_len,
        int first_q_pos,
        int sliding_window)
    {
        mask.assign(static_cast<std::size_t>(padded_kv_len) * num_q,
                    ggml_fp32_to_fp16(0.0f));
        const ggml_fp16_t neg_inf_h = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());

        for (int q = 0; q < num_q; ++q)
        {
            int q_pos = first_q_pos + q;
            ggml_fp16_t* row = mask.data() + static_cast<std::size_t>(q) * padded_kv_len;

            // Causal cutoff: queries can only attend to keys at <= q_pos.
            int allowed_end = std::min(q_pos + 1, seq_len);
            for (int k = allowed_end; k < padded_kv_len; ++k)
                row[k] = neg_inf_h;

            // Sliding-window cutoff: queries can only attend to keys whose
            // position is within the last `sliding_window` tokens (inclusive
            // of the query itself). I.e. block keys at k <= q_pos - W.
            if (sliding_window > 0)
            {
                int window_start = q_pos - sliding_window + 1;
                if (window_start > 0)
                {
                    for (int k = 0; k < window_start && k < padded_kv_len; ++k)
                        row[k] = neg_inf_h;
                }
            }
        }
    }

    int paged_attention_single_seq_impl(
        const float* q_data,        // [num_q * num_heads * head_dim] (row-major)
        const float* k_data,        // [seq_len * num_kv_heads * head_dim] (gathered)
        const float* v_data,        // [seq_len * num_kv_heads * head_dim] (gathered)
        float* out_data,            // [num_q * num_heads * head_dim] (write)
        int num_q,
        int seq_len,
        int first_q_pos,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        int sliding_window,         // 0 = no SWA, else block keys older than W tokens
        float scale,
        const float* sinks_data)    // [num_heads] F32, or null for no sinks
    {
        if (!ensure_backend()) return 0;

        const int q_bytes = num_q * num_heads * head_dim * static_cast<int>(sizeof(float));
        const int kv_bytes = seq_len * num_kv_heads * head_dim * static_cast<int>(sizeof(float));

        const int padded_kv_len = paged_padded_kv_length(seq_len, head_dim);

        // Reuse a per-call ggml context big enough for the graph nodes.
        PooledContextHandle context;
        if (!context.init(1024 * 1024))
        {
            set_last_error("Failed to acquire ggml context for paged attention.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // Q tensor: layout matches C# row-major [num_q, num_heads, head_dim]
        // -> GGML ne[0]=head_dim (innermost), ne[1]=num_heads, ne[2]=num_q.
        ggml_tensor* q_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, num_heads, num_q);
        // K, V tensors: laid out [seq_len, num_kv_heads, head_dim] row-major
        // -> ne[0]=head_dim, ne[1]=num_kv_heads, ne[2]=padded_kv_len (with
        // padding slots zeroed and masked off).
        ggml_tensor* k_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, num_kv_heads, padded_kv_len);
        ggml_tensor* v_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, num_kv_heads, padded_kv_len);
        // Output: same shape as Q.
        ggml_tensor* attn_result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, num_heads, num_q);

        // Build mask. We always provide one for prefill chunks (num_q > 1),
        // for non-zero first_q_pos (mid-sequence resume), and when the head
        // dimension forces padding. Decode steps at position 0 with no
        // padding need no mask.
        ggml_tensor* attn_mask = nullptr;
        std::vector<ggml_fp16_t> mask_data;
        const bool needs_mask =
            (num_q > 1) ||
            (first_q_pos > 0) ||
            (padded_kv_len != seq_len) ||
            (sliding_window > 0 && sliding_window < seq_len);
        if (needs_mask)
        {
            // GGML's flash-attn mask shape is [padded_kv_len, num_q, 1, 1] F16.
            attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, padded_kv_len, num_q, 1, 1);
            fill_causal_mask_fp16(mask_data, padded_kv_len, num_q, seq_len, first_q_pos, sliding_window);
        }

        // Permute Q to [head_dim, num_q, num_heads] expected by flash_attn_ext,
        // then ggml_cont it. The cont() is load-bearing: ggml_flash_attn_ext
        // reads Q through its strides but several backends (Metal, in
        // particular) silently produce wrong attention output when Q is a
        // non-contiguous view, which manifests as a divergence of ~10% in
        // logits for GQA prefill with num_q > 1. K and V already get cont
        // below for the same reason.
        ggml_tensor* q_perm = ggml_permute(ctx, q_in, 0, 2, 1, 3);
        ggml_tensor* q_attn = ggml_cont(ctx, q_perm);
        // Permute K, V to [head_dim, padded_kv_len, num_kv_heads].
        ggml_tensor* k_attn_perm = ggml_permute(ctx, k_in, 0, 2, 1, 3);
        ggml_tensor* v_attn_perm = ggml_permute(ctx, v_in, 0, 2, 1, 3);
        ggml_tensor* k_attn = ggml_cont(ctx, k_attn_perm);
        ggml_tensor* v_attn = ggml_cont(ctx, v_attn_perm);

        ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q_attn, k_attn, v_attn,
                                                   attn_mask, scale, 0.0f, 0.0f);

        // Optional per-head attention sinks (gpt-oss): a learned scalar per
        // head that participates in the softmax denominator as a virtual
        // position with zero V contribution. Sinks tensor is [num_heads] F32
        // (matches Q's head dimension; ggml asserts this internally).
        ggml_tensor* sinks_tensor = nullptr;
        if (sinks_data != nullptr)
        {
            sinks_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_heads);
            ggml_flash_attn_ext_add_sinks(attn_out, sinks_tensor);
        }
        // flash_attn_ext output layout: [n_embd_head_v, n_head, n_tokens, n_batch]
        // (note: heads and tokens are SWAPPED relative to the Q layout).
        // For us that's exactly [head_dim, num_heads, num_q, 1] - same as the
        // row-major [num_q, num_heads, head_dim] output we want to write back.
        // Directly cpy into attn_result; no permute needed.
        ggml_tensor* result = ggml_cpy(ctx, attn_out, attn_result);
        ggml_set_output(result);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate backend buffer for paged attention.");
            return 0;
        }

        host_read_barrier();

        ggml_backend_tensor_set(q_in, q_data, 0, q_bytes);
        // For padded K/V, we set the valid portion. The padding slots stay
        // zero (already zero-initialised by ggml_backend_alloc_ctx_tensors).
        ggml_backend_tensor_set(k_in, k_data, 0, kv_bytes);
        ggml_backend_tensor_set(v_in, v_data, 0, kv_bytes);
        if (padded_kv_len > seq_len)
        {
            // Zero-fill the padding slots so flash-attn doesn't read garbage.
            std::vector<float> zeros((padded_kv_len - seq_len) * num_kv_heads * head_dim, 0.0f);
            const std::size_t pad_offset = static_cast<std::size_t>(kv_bytes);
            ggml_backend_tensor_set(k_in, zeros.data(), pad_offset, zeros.size() * sizeof(float));
            ggml_backend_tensor_set(v_in, zeros.data(), pad_offset, zeros.size() * sizeof(float));
        }
        if (attn_mask != nullptr && !mask_data.empty())
        {
            ggml_backend_tensor_set(attn_mask, mask_data.data(), 0,
                                     mask_data.size() * sizeof(ggml_fp16_t));
        }
        if (sinks_tensor != nullptr)
        {
            ggml_backend_tensor_set(sinks_tensor, sinks_data, 0,
                                     static_cast<std::size_t>(num_heads) * sizeof(float));
        }

        ggml_status st = ggml_backend_graph_compute(g_backend, graph);
        if (st != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml graph compute failed for paged attention.");
            return 0;
        }

        finalize_compute_with_download(attn_result, out_data,
                                       static_cast<std::size_t>(num_q) * num_heads * head_dim * sizeof(float));
        return 1;
    }
} // anonymous namespace

// ============================================================================
// TSGgml_PagedAttentionForward
//
// Batched paged-attention forward. One call per (layer, step) drives every
// scheduled sequence's attention through `ggml_flash_attn_ext`. K and V come
// in as a single flat float[] block-pool (laid out [num_blocks, block_size,
// num_kv_heads, head_dim] row-major); per-sequence `block_tables` index into
// the block dimension.
// ============================================================================
TSG_EXPORT int TSGgml_PagedAttentionForward(
    const float* q_data,            // [num_tokens, num_heads * head_dim]
    const float* paged_k_data,      // [num_blocks * block_size * num_kv_heads * head_dim]
    const float* paged_v_data,      // same
    float* out_data,                // [num_tokens, num_heads * head_dim] (output)
    const int* query_start_loc,     // [num_seqs + 1]
    const int* seq_lens,            // [num_seqs]
    const int* positions,           // [num_tokens] (absolute position of each query token)
    const int* block_table_flat,    // concatenated per-seq block tables
    const int* block_table_offsets, // [num_seqs] - offset of each seq's table in flat
    int num_seqs,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int sliding_window,             // 0 = full attention, >0 = SWA with that window
    float scale)
{
    try
    {
        if (q_data == nullptr || paged_k_data == nullptr || paged_v_data == nullptr ||
            out_data == nullptr || query_start_loc == nullptr || seq_lens == nullptr ||
            block_table_flat == nullptr || block_table_offsets == nullptr)
        {
            set_last_error("Null pointer passed to paged attention forward.");
            return 0;
        }
        if (num_seqs <= 0 || num_tokens <= 0 || num_heads <= 0 || num_kv_heads <= 0 ||
            head_dim <= 0 || block_size <= 0)
        {
            set_last_error("Invalid dimensions for paged attention forward.");
            return 0;
        }
        if (num_heads % num_kv_heads != 0)
        {
            set_last_error("num_heads must be divisible by num_kv_heads.");
            return 0;
        }

        const int per_token_kv_stride = num_kv_heads * head_dim;
        const int per_token_q_stride = num_heads * head_dim;

        // Reusable scratch buffer for the gather. Sized for the longest
        // sequence in this batch.
        int max_seq_len = 0;
        for (int s = 0; s < num_seqs; ++s)
            max_seq_len = std::max(max_seq_len, seq_lens[s]);
        if (max_seq_len <= 0)
        {
            set_last_error("All seq_lens are zero in paged attention forward.");
            return 0;
        }

        std::vector<float> k_scratch(static_cast<std::size_t>(max_seq_len) * per_token_kv_stride);
        std::vector<float> v_scratch(static_cast<std::size_t>(max_seq_len) * per_token_kv_stride);

        for (int s = 0; s < num_seqs; ++s)
        {
            const int q_start = query_start_loc[s];
            const int q_end = query_start_loc[s + 1];
            const int num_q = q_end - q_start;
            const int seq_len = seq_lens[s];
            if (num_q <= 0) continue;
            if (seq_len <= 0)
            {
                set_last_error("Encountered zero seq_len in paged attention forward.");
                return 0;
            }

            // Absolute position of the first query token. This is the
            // critical input to the causal mask.
            const int first_q_pos = positions != nullptr ? positions[q_start]
                                                          : (seq_len - num_q);

            // Gather K, V for this sequence from the paged buffer.
            const int* table = block_table_flat + block_table_offsets[s];
            const int n_blocks = (seq_len + block_size - 1) / block_size;
            const std::size_t per_token_kv_bytes =
                static_cast<std::size_t>(per_token_kv_stride) * sizeof(float);
            for (int blk = 0; blk < n_blocks; ++blk)
            {
                const int block_id = table[blk];
                const int tokens_in_block = std::min(block_size, seq_len - blk * block_size);
                const std::size_t src_offset =
                    static_cast<std::size_t>(block_id) * block_size * per_token_kv_stride;
                const std::size_t dst_offset =
                    static_cast<std::size_t>(blk) * block_size * per_token_kv_stride;
                const std::size_t bytes =
                    static_cast<std::size_t>(tokens_in_block) * per_token_kv_bytes;
                std::memcpy(k_scratch.data() + dst_offset, paged_k_data + src_offset, bytes);
                std::memcpy(v_scratch.data() + dst_offset, paged_v_data + src_offset, bytes);
            }

            int ok = paged_attention_single_seq_impl(
                q_data + static_cast<std::size_t>(q_start) * per_token_q_stride,
                k_scratch.data(),
                v_scratch.data(),
                out_data + static_cast<std::size_t>(q_start) * per_token_q_stride,
                num_q, seq_len, first_q_pos,
                num_heads, num_kv_heads, head_dim, sliding_window, scale,
                /* sinks_data */ nullptr);
            if (!ok)
                return 0;
        }

        // Drain any pending work from the LAST sequence's impl before
        // returning to C#, so the caller sees the full out_data without
        // having to remember to issue a barrier itself. The impl uses
        // async finalize on Metal, so without this the final seq's
        // download could still be in flight when the caller reads out_data.
        host_read_barrier();
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in TSGgml_PagedAttentionForward.");
        return 0;
    }
}

// ============================================================================
// TSGgml_PagedAttentionForwardWithSinks
//
// Same as TSGgml_PagedAttentionForward but with per-head attention sinks
// (gpt-oss style). Sinks is a [num_heads] F32 array. Routed through
// ggml_flash_attn_ext_add_sinks so the native Metal/CUDA flash-attn kernel
// includes the sink as a virtual position in the softmax denominator —
// matching the legacy GptOss per-seq AttentionDecodeWithSinks /
// AttentionPrefillWithSinks math, just under the paged-K/V layout.
// ============================================================================
TSG_EXPORT int TSGgml_PagedAttentionForwardWithSinks(
    const float* q_data,
    const float* paged_k_data,
    const float* paged_v_data,
    float* out_data,
    const int* query_start_loc,
    const int* seq_lens,
    const int* positions,
    const int* block_table_flat,
    const int* block_table_offsets,
    int num_seqs,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int sliding_window,
    float scale,
    const float* sinks_data)        // [num_heads] F32 (nullable — null == identical to PagedAttentionForward)
{
    try
    {
        if (q_data == nullptr || paged_k_data == nullptr || paged_v_data == nullptr ||
            out_data == nullptr || query_start_loc == nullptr || seq_lens == nullptr ||
            block_table_flat == nullptr || block_table_offsets == nullptr)
        {
            set_last_error("Null pointer passed to paged attention forward (with sinks).");
            return 0;
        }
        if (num_seqs <= 0 || num_tokens <= 0 || num_heads <= 0 || num_kv_heads <= 0 ||
            head_dim <= 0 || block_size <= 0)
        {
            set_last_error("Invalid dimensions for paged attention forward (with sinks).");
            return 0;
        }
        if (num_heads % num_kv_heads != 0)
        {
            set_last_error("num_heads must be divisible by num_kv_heads.");
            return 0;
        }

        const int per_token_kv_stride = num_kv_heads * head_dim;
        const int per_token_q_stride = num_heads * head_dim;

        int max_seq_len = 0;
        for (int s = 0; s < num_seqs; ++s)
            max_seq_len = std::max(max_seq_len, seq_lens[s]);
        if (max_seq_len <= 0)
        {
            set_last_error("All seq_lens are zero in paged attention forward (with sinks).");
            return 0;
        }

        std::vector<float> k_scratch(static_cast<std::size_t>(max_seq_len) * per_token_kv_stride);
        std::vector<float> v_scratch(static_cast<std::size_t>(max_seq_len) * per_token_kv_stride);

        for (int s = 0; s < num_seqs; ++s)
        {
            const int q_start = query_start_loc[s];
            const int q_end = query_start_loc[s + 1];
            const int num_q = q_end - q_start;
            const int seq_len = seq_lens[s];
            if (num_q <= 0) continue;
            if (seq_len <= 0)
            {
                set_last_error("Encountered zero seq_len in paged attention forward (with sinks).");
                return 0;
            }

            const int first_q_pos = positions != nullptr ? positions[q_start]
                                                          : (seq_len - num_q);

            const int* table = block_table_flat + block_table_offsets[s];
            const int n_blocks = (seq_len + block_size - 1) / block_size;
            const std::size_t per_token_kv_bytes =
                static_cast<std::size_t>(per_token_kv_stride) * sizeof(float);
            for (int blk = 0; blk < n_blocks; ++blk)
            {
                const int block_id = table[blk];
                const int tokens_in_block = std::min(block_size, seq_len - blk * block_size);
                const std::size_t src_offset =
                    static_cast<std::size_t>(block_id) * block_size * per_token_kv_stride;
                const std::size_t dst_offset =
                    static_cast<std::size_t>(blk) * block_size * per_token_kv_stride;
                const std::size_t bytes =
                    static_cast<std::size_t>(tokens_in_block) * per_token_kv_bytes;
                std::memcpy(k_scratch.data() + dst_offset, paged_k_data + src_offset, bytes);
                std::memcpy(v_scratch.data() + dst_offset, paged_v_data + src_offset, bytes);
            }

            int ok = paged_attention_single_seq_impl(
                q_data + static_cast<std::size_t>(q_start) * per_token_q_stride,
                k_scratch.data(),
                v_scratch.data(),
                out_data + static_cast<std::size_t>(q_start) * per_token_q_stride,
                num_q, seq_len, first_q_pos,
                num_heads, num_kv_heads, head_dim, sliding_window, scale,
                sinks_data);
            if (!ok)
                return 0;
        }

        host_read_barrier();
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in TSGgml_PagedAttentionForwardWithSinks.");
        return 0;
    }
}
