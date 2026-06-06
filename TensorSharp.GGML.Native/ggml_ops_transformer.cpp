// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
#include "ggml_ops_internal.h"

using namespace tsg;

// ============================================================================
// Batched transformer layer decode: full layer in a single GGML graph.
// Handles: attn_norm → QKV matmul → QK norm → RoPE → flash attention →
//          O projection → residual → FFN norm → GateUp matmul → SiLU*Mul →
//          Down matmul → residual.
// Updates hidden state in-place and writes new K/V to the KV cache.
// ============================================================================
namespace
{
    // KV-cache element size in bytes for the given GGML tensor type.
    // F32 = 4, F16 = 2. For block-quantized types (Q8_0) the *element size* is
    // fractional (1.0625 bytes for Q8_0) so callers that need a per-element byte
    // count should NOT use this helper - they should use ggml's per-row stride
    // (`tensor->nb[1]`) directly, which already accounts for block padding.
    // We still expose this helper for the linear types because some byte-offset
    // arithmetic happens before the cache tensor is materialised.
    inline std::size_t kv_cache_elem_size(int kv_cache_type)
    {
        switch (static_cast<ggml_type>(kv_cache_type))
        {
            case GGML_TYPE_F32:  return 4;
            case GGML_TYPE_F16:  return 2;
            // Block-quantized: callers should use nb[1] / row_size instead.
            // Returning 0 here makes any accidental `head_dim * elem_size` use
            // visibly wrong rather than silently miscounting.
            case GGML_TYPE_Q8_0: return 0;
            default:             return 4;
        }
    }

    inline bool kv_cache_is_block_quantized(int kv_cache_type)
    {
        return static_cast<ggml_type>(kv_cache_type) == GGML_TYPE_Q8_0;
    }

    // Bytes occupied by a [kv_heads, cache_size, head_dim] cache tensor of the
    // given GGML type. Uses ggml_row_size so block-quantized layouts (Q8_0) are
    // accounted for correctly: a Q8_0 row of 256 elements occupies 8 blocks * 34
    // bytes = 272 bytes (vs. 256 raw bytes if we used a fractional 1.0625 value).
    std::size_t kv_cache_bytes(int kv_heads, int cache_size, int head_dim, int kv_cache_type = GGML_TYPE_F32)
    {
        const std::size_t row_bytes = ggml_row_size(static_cast<ggml_type>(kv_cache_type), head_dim);
        return static_cast<std::size_t>(kv_heads) *
               static_cast<std::size_t>(cache_size) *
               row_bytes;
    }

    constexpr int kFlashAttnKvStride = 256;

    bool flash_attn_requires_masked_padding(int head_dim)
    {
        // The custom CUDA kernels added for 512/576-dim attention only support
        // the grouped-query path, which expects a non-null mask and a KV length
        // aligned to FATTN_KQ_STRIDE.
        return head_dim == 512 || head_dim == 576;
    }

    int flash_attn_kv_length(int valid_len, int cache_size, int head_dim)
    {
        if (!flash_attn_requires_masked_padding(head_dim))
            return valid_len;

        const int padded = ((valid_len + kFlashAttnKvStride - 1) / kFlashAttnKvStride) * kFlashAttnKvStride;
        return std::min(cache_size, std::max(valid_len, padded));
    }

    void fill_flash_attn_mask(std::vector<ggml_fp16_t>& mask, int padded_len, int valid_len)
    {
        mask.assign(static_cast<std::size_t>(padded_len), ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        const int unclamped_valid = std::max(valid_len, 0);
        const int clamped_valid = std::min(unclamped_valid, padded_len);
        std::fill_n(mask.begin(), clamped_valid, static_cast<ggml_fp16_t>(0));
    }

    ggml_tensor* view_kv_cache_window(
        ggml_context* ctx,
        ggml_tensor* cache,
        int head_dim,
        int cache_size,
        int kv_heads,
        int start_idx,
        int length,
        int kv_cache_type = GGML_TYPE_F32)
    {
        if (ctx == nullptr || cache == nullptr || head_dim <= 0 || cache_size <= 0 || kv_heads <= 0 || length <= 0)
            return nullptr;

        start_idx %= cache_size;
        if (start_idx < 0)
            start_idx += cache_size;

        // ggml_row_size handles block-quantized types (Q8_0) correctly: a row of
        // 256 Q8_0 elements is 8 blocks * 34 bytes = 272 bytes, not 256/1.0625.
        // For linear types it reduces to head_dim * sizeof(elem).
        const std::size_t nb1 = ggml_row_size(static_cast<ggml_type>(kv_cache_type), head_dim);
        const std::size_t nb2 = static_cast<std::size_t>(cache_size) * nb1;

        if (start_idx + length <= cache_size)
        {
            return ggml_view_3d(
                ctx,
                cache,
                head_dim,
                length,
                kv_heads,
                nb1,
                nb2,
                static_cast<std::size_t>(start_idx) * nb1);
        }

        const int tail_length = cache_size - start_idx;
        const int head_length = length - tail_length;
        ggml_tensor* tail = ggml_view_3d(
            ctx,
            cache,
            head_dim,
            tail_length,
            kv_heads,
            nb1,
            nb2,
            static_cast<std::size_t>(start_idx) * nb1);
        ggml_tensor* head = ggml_view_3d(ctx, cache, head_dim, head_length, kv_heads, nb1, nb2, 0);
        if (tail == nullptr || head == nullptr)
            return nullptr;

        // GPU concat kernels only implement F32 inputs. Wrapped circular windows may
        // come from F16/Q8_0 KV caches, so materialize both slices as F32 first.
        if (static_cast<ggml_type>(kv_cache_type) != GGML_TYPE_F32)
        {
            ggml_tensor* tail_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, tail_length, kv_heads);
            ggml_tensor* head_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, head_length, kv_heads);
            if (tail_f32 == nullptr || head_f32 == nullptr)
                return nullptr;

            tail = ggml_cpy(ctx, tail, tail_f32);
            head = ggml_cpy(ctx, head, head_f32);
        }

        return ggml_concat(ctx, tail, head, 1);
    }

    // ============================================================================
    // Stand-alone flash attention decode kernel.
    //
    // Performs (for a single query position):
    //   1. Append the new K/V vectors to the persistent KV cache at `position`.
    //   2. Run ggml_flash_attn_ext on the device, which reads Q, the populated
    //      cache (length = position + 1), and writes the attention result.
    //
    // Inputs and the KV cache live in C# memory and are mapped zero-copy where
    // the backend permits it. Q/K/V here are *already* normalized and RoPE'd by
    // the C# host: this kernel exists purely to fold the cache append + softmax-
    // attention + value mix into one GPU graph (instead of the previous CPU-side
    // SIMD path).
    //
    // Used by Qwen3.5 (and other architectures with a custom attention pre-
    // processing stage that can't be expressed inside ggml_flash_attn_ext).
    // ============================================================================
    int flash_attn_decode_impl(
        const float* q_data,        // [num_heads * head_dim]      Q (post-norm, post-RoPE)
        const float* k_data,        // [num_kv_heads * head_dim]   K (post-norm, post-RoPE)
        const float* v_data,        // [num_kv_heads * head_dim]   V
        void* k_cache_data,         // [num_kv_heads, max_seq_len, head_dim]  (F32 or F16)
        void* v_cache_data,         // [num_kv_heads, max_seq_len, head_dim]  (F32 or F16)
        float* out_data,            // [num_heads * head_dim]      (writeable)
        int num_heads, int num_kv_heads, int head_dim,
        int max_seq_len, int position,
        float scale,
        int kv_cache_type = GGML_TYPE_F32)
    {
        if (!ensure_backend())
            return 0;

        if (q_data == nullptr || k_data == nullptr || v_data == nullptr ||
            k_cache_data == nullptr || v_cache_data == nullptr || out_data == nullptr)
        {
            set_last_error("Null pointer passed to flash attention decode kernel.");
            return 0;
        }

        if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || max_seq_len <= 0 || position < 0)
        {
            set_last_error("Invalid dimensions passed to flash attention decode kernel.");
            return 0;
        }

        const int q_dim = num_heads * head_dim;
        const int kv_dim = num_kv_heads * head_dim;
        const int totalSeqLen = position + 1;
        const int attnKvLen = flash_attn_kv_length(totalSeqLen, max_seq_len, head_dim);
        std::vector<ggml_fp16_t> attn_mask_data;

        PooledContextHandle context;
        if (!context.init(512 * 1024))
        {
            set_last_error("Failed to create ggml context for flash attention decode.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // Inputs (host-side staging; copy in via backend tensor set).
        ggml_tensor* q_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, q_dim);
        ggml_tensor* k_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kv_dim);
        ggml_tensor* v_in = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kv_dim);

        // KV cache (zero-copy bound to C# memory). Type can be F32 or F16
        // depending on the model's KV cache configuration. ggml_cpy handles
        // F32 -> F16 conversion automatically when the destination is F16.
        ggml_tensor* k_cache_base = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, max_seq_len, num_kv_heads);
        ggml_tensor* v_cache_base = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, max_seq_len, num_kv_heads);

        // Output download target.
        ggml_tensor* attn_result = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, q_dim);

        // Optional flash-attn mask (only required for some head dims).
        ggml_tensor* attn_mask = nullptr;
        if (flash_attn_requires_masked_padding(head_dim))
        {
            attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
            fill_flash_attn_mask(attn_mask_data, attnKvLen, totalSeqLen);
        }

        // === Build computation graph ===

        // 1. Reshape Q to [head_dim, 1, num_heads] for flash_attn_ext.
        //    (Input layout is contiguous head-major, i.e. h0_d0..h0_dn h1_d0..)
        ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_in, head_dim, num_heads, 1);
        ggml_tensor* q_attn = ggml_permute(ctx, q_3d, 0, 2, 1, 3);

        // 2. Reshape K/V and append into the cache at `position`.
        ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_in, head_dim, num_kv_heads, 1);
        ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_in, head_dim, num_kv_heads, 1);

        ggml_tensor* k_perm = ggml_permute(ctx, k_3d, 0, 2, 1, 3);
        ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
        ggml_tensor* k_write = ggml_cont(ctx, k_perm);
        ggml_tensor* v_write = ggml_cont(ctx, v_perm);

        // Use the K cache tensor's row stride to compute the per-position byte
        // offset; this naturally adjusts for F32 vs F16 cache layouts.
        const std::size_t kv_byte_offset =
            static_cast<std::size_t>(position) * k_cache_base->nb[1];
        ggml_tensor* k_dst = ggml_view_3d(ctx, k_cache_base,
            head_dim, 1, num_kv_heads,
            k_cache_base->nb[1], k_cache_base->nb[2], kv_byte_offset);
        ggml_tensor* v_dst = ggml_view_3d(ctx, v_cache_base,
            head_dim, 1, num_kv_heads,
            v_cache_base->nb[1], v_cache_base->nb[2], kv_byte_offset);
        ggml_tensor* k_cache_cpy = ggml_cpy(ctx, k_write, k_dst);
        ggml_tensor* v_cache_cpy = ggml_cpy(ctx, v_write, v_dst);

        // 3. Build a view over the populated portion of the cache.
        ggml_tensor* k_full = view_kv_cache_window(ctx, k_cache_base, head_dim, max_seq_len, num_kv_heads, 0, attnKvLen, kv_cache_type);
        ggml_tensor* v_full = view_kv_cache_window(ctx, v_cache_base, head_dim, max_seq_len, num_kv_heads, 0, attnKvLen, kv_cache_type);
        if (k_full == nullptr || v_full == nullptr)
        {
            set_last_error("Failed to create KV cache views for flash attention decode.");
            return 0;
        }

        // 4. Flash attention (handles GQA broadcasting automatically).
        //    q: [head_dim, 1, num_heads], k/v: [head_dim, attnKvLen, num_kv_heads]
        ggml_tensor* attn_out = ggml_flash_attn_ext(ctx,
            q_attn, k_full, v_full, attn_mask, scale, 0.0f, 0.0f);

        // 5. Reshape back to [num_heads * head_dim] for download.
        ggml_tensor* attn_flat = ggml_reshape_1d(ctx, attn_out, q_dim);
        ggml_tensor* result = ggml_cpy(ctx, attn_flat, attn_result);
        ggml_set_output(result);

        // Build graph: cache writes must execute before flash attention reads.
        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, k_cache_cpy);
        ggml_build_forward_expand(graph, v_cache_cpy);
        ggml_build_forward_expand(graph, result);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);

        struct HostBinding { ggml_tensor* tensor; const void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr)
                return;

            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs_upload, usage))
                {
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, addr);
                    if (st == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload)
                            upload_list.push_back({t, data, bytes});
                        return;
                    }

                    invalidate_cached_buffer(data);
                }
            }

            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable)
                        ephemeral_bufs.emplace_back(buf);
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, data);
                    if (st == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        // Cache buffers are persistent across calls and benefit from the cacheable mapping.
        bind_or_mark(k_cache_base, k_cache_data, kv_cache_bytes(num_kv_heads, max_seq_len, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        bind_or_mark(v_cache_base, v_cache_data, kv_cache_bytes(num_kv_heads, max_seq_len, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        if (attn_mask != nullptr && !attn_mask_data.empty())
            bind_or_mark(attn_mask, attn_mask_data.data(), attn_mask_data.size() * sizeof(ggml_fp16_t), false);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate backend buffer for flash attention decode.");
            return 0;
        }

        // Drain pending async work before CPU memcpys from C# tensor buffers.
        host_read_barrier();

        // Upload non-host-ptr tensors.
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);

        ggml_backend_tensor_set(q_in, q_data, 0, static_cast<std::size_t>(q_dim) * sizeof(float));
        ggml_backend_tensor_set(k_in, k_data, 0, static_cast<std::size_t>(kv_dim) * sizeof(float));
        ggml_backend_tensor_set(v_in, v_data, 0, static_cast<std::size_t>(kv_dim) * sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for flash attention decode.");
            return 0;
        }

        finalize_compute_with_download(attn_result, out_data, static_cast<std::size_t>(q_dim) * sizeof(float));

        clear_last_error();
        return 1;
    }

    int transformer_layer_decode_impl(
        float* hidden_data, int hidden_size,
        float* attn_norm_data,
        void* qkv_data, int qkv_type, std::int64_t qkv_ne0, std::int64_t qkv_ne1, std::int64_t qkv_bytes,
        float* q_norm_data, float* k_norm_data, int head_dim,
        void* o_data, int o_type, std::int64_t o_ne0, std::int64_t o_ne1, std::int64_t o_bytes,
        float* ffn_norm_data,
        void* gu_data, int gu_type, std::int64_t gu_ne0, std::int64_t gu_ne1, std::int64_t gu_bytes,
        void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_bytes,
        void* k_cache_data, void* v_cache_data,
        int num_heads, int num_kv_heads,
        int max_seq_len, int position,
        float eps, float rope_base, float rope_freq_scale,
        int intermediate_size, int rope_mode,
        int kv_cache_type = GGML_TYPE_F32)
    {
        if (!ensure_backend())
            return 0;

        const int qDim = num_heads * head_dim;
        const int kDim = num_kv_heads * head_dim;
        const int totalSeqLen = position + 1;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int attnKvLen = flash_attn_kv_length(totalSeqLen, max_seq_len, head_dim);
        std::vector<ggml_fp16_t> attn_mask_data;

        PooledContextHandle context;
        if (!context.init(2 * 1024 * 1024))
        {
            set_last_error("Failed to create ggml context for transformer layer decode.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // === Input / weight tensors ===
        ggml_tensor* input        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* attn_norm_w  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* q_norm_w     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
        ggml_tensor* k_norm_w     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
        ggml_tensor* ffn_norm_w   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        ggml_tensor* qkv_w  = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkv_type), qkv_ne0, qkv_ne1);
        ggml_tensor* o_w     = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type), o_ne0, o_ne1);
        ggml_tensor* gu_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gu_type), gu_ne0, gu_ne1);
        ggml_tensor* down_w  = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type), down_ne0, down_ne1);

        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* k_cache_base = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, max_seq_len, num_kv_heads);
        ggml_tensor* v_cache_base = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, max_seq_len, num_kv_heads);
        ggml_tensor* attn_mask = nullptr;
        if (flash_attn_requires_masked_padding(head_dim))
        {
            attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
            fill_flash_attn_mask(attn_mask_data, attnKvLen, totalSeqLen);
        }

        // Output download target
        ggml_tensor* hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        // === Build computation graph ===

        // 1. Attention norm: RMSNorm + element-wise scale
        ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, input, eps), attn_norm_w);

        // 2. Fused QKV projection (quantized matmul)
        ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, hidden_size, 1);
        ggml_tensor* qkv_flat  = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, qkv_w, normed_2d), qDim + 2 * kDim);

        // 3. Split Q, K, V
        ggml_tensor* q_raw = ggml_view_1d(ctx, qkv_flat, qDim, 0);
        ggml_tensor* k_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim)  * sizeof(float));
        ggml_tensor* v_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim + kDim) * sizeof(float));

        // 4. Per-head QK norm
        ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_raw, head_dim, num_heads);
        ggml_tensor* k_2d = ggml_reshape_2d(ctx, k_raw, head_dim, num_kv_heads);

        ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), q_norm_w);
        ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d, eps), k_norm_w);

        // 5. RoPE (NeoX mode)
        // ggml_rope_ext expects: ne[0]=head_dim, ne[1]=n_heads, ne[2]=seqLen
        // positions tensor ne[0] must equal ne[2]
        ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, head_dim, num_heads, 1);
        ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, head_dim, num_kv_heads, 1);

        ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, nullptr,
            head_dim, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
        ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, nullptr,
            head_dim, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);

        // 6. Build full KV for attention: concat cached + new
        // After RoPE: q_rope=[head_dim, num_heads, 1], k_rope=[head_dim, num_kv_heads, 1]
        // flash_attn_ext expects: q=[head_dim, n_batch, n_head], k/v=[head_dim, n_kv, n_head_kv]
        // Need to permute dims 1,2: [head_dim, n_heads, 1] → [head_dim, 1, n_heads]
        ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);

        ggml_tensor* k_rope_perm = ggml_permute(ctx, k_rope, 0, 2, 1, 3);
        ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_raw, head_dim, num_kv_heads, 1);
        ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
        ggml_tensor* k_write = ggml_cont(ctx, k_rope_perm);
        ggml_tensor* v_write = ggml_cont(ctx, v_perm);
        const std::size_t kv_byte_offset =
            static_cast<std::size_t>(position) * k_cache_base->nb[1];
        ggml_tensor* k_dst = ggml_view_3d(ctx, k_cache_base,
            head_dim, 1, num_kv_heads,
            k_cache_base->nb[1], k_cache_base->nb[2], kv_byte_offset);
        ggml_tensor* v_dst = ggml_view_3d(ctx, v_cache_base,
            head_dim, 1, num_kv_heads,
            v_cache_base->nb[1], v_cache_base->nb[2], kv_byte_offset);
        ggml_tensor* k_cache_cpy = ggml_cpy(ctx, k_write, k_dst);
        ggml_tensor* v_cache_cpy = ggml_cpy(ctx, v_write, v_dst);
        ggml_tensor* k_full = view_kv_cache_window(ctx, k_cache_base, head_dim, max_seq_len, num_kv_heads, 0, attnKvLen, kv_cache_type);
        ggml_tensor* v_full = view_kv_cache_window(ctx, v_cache_base, head_dim, max_seq_len, num_kv_heads, 0, attnKvLen, kv_cache_type);
        if (k_full == nullptr || v_full == nullptr)
        {
            set_last_error("Failed to create KV cache views for transformer layer decode.");
            return 0;
        }

        // 7. Flash attention (handles GQA broadcasting automatically)
        // q: [head_dim, 1, num_heads], k/v: [head_dim, attnKvLen, num_kv_heads]
        ggml_tensor* attn_out = ggml_flash_attn_ext(ctx,
            q_attn, k_full, v_full, attn_mask, scale, 0.0f, 0.0f);

        // 8. O projection
        ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, qDim, 1);
        ggml_tensor* o_flat    = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, o_w, attn_flat), hidden_size);

        // 9. First residual
        ggml_tensor* residual1 = ggml_add(ctx, input, o_flat);

        // 10. FFN norm
        ggml_tensor* normed2 = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), ffn_norm_w);

        // 11. Fused GateUp projection
        ggml_tensor* normed2_2d = ggml_reshape_2d(ctx, normed2, hidden_size, 1);
        ggml_tensor* gu_flat    = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, gu_w, normed2_2d), 2 * intermediate_size);

        // 12. Split gate / up, SiLU(gate) * up
        ggml_tensor* gate = ggml_view_1d(ctx, gu_flat, intermediate_size, 0);
        ggml_tensor* up   = ggml_view_1d(ctx, gu_flat, intermediate_size,
                                          static_cast<std::size_t>(intermediate_size) * sizeof(float));
        ggml_tensor* ffn_hidden = ggml_mul(ctx, ggml_silu(ctx, gate), up);

        // 13. Down projection
        ggml_tensor* ffn_2d   = ggml_reshape_2d(ctx, ffn_hidden, intermediate_size, 1);
        ggml_tensor* down_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, down_w, ffn_2d), hidden_size);

        // 14. Second residual
        ggml_tensor* result = ggml_add(ctx, residual1, down_flat);

        // Mark graph output: updated hidden state
        ggml_tensor* out_hidden = ggml_cpy(ctx, result, hidden_out);
        ggml_set_output(out_hidden);

        // Build graph: add KV cache writes first to ensure they execute before reads
        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, k_cache_cpy);
        ggml_build_forward_expand(graph, v_cache_cpy);
        ggml_build_forward_expand(graph, out_hidden);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);

        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr)
                return;

            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs_upload, usage))
                {
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, addr);
                    if (st == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload)
                            upload_list.push_back({t, data, bytes});
                        return;
                    }

                    invalidate_cached_buffer(data);
                }
            }

            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable)
                        ephemeral_bufs.emplace_back(buf);
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, data);
                    if (st == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        bind_or_mark(qkv_w,  qkv_data,  static_cast<std::size_t>(qkv_bytes), true);
        bind_or_mark(o_w,    o_data,    static_cast<std::size_t>(o_bytes), true);
        bind_or_mark(gu_w,   gu_data,   static_cast<std::size_t>(gu_bytes), true);
        bind_or_mark(down_w, down_data, static_cast<std::size_t>(down_bytes), true);

        bind_or_mark(attn_norm_w, attn_norm_data, static_cast<std::size_t>(hidden_size) * sizeof(float), true);
        bind_or_mark(ffn_norm_w,  ffn_norm_data,  static_cast<std::size_t>(hidden_size) * sizeof(float), true);
        bind_or_mark(q_norm_w,    q_norm_data,    static_cast<std::size_t>(head_dim) * sizeof(float), true);
        bind_or_mark(k_norm_w,    k_norm_data,    static_cast<std::size_t>(head_dim) * sizeof(float), true);
        bind_or_mark(k_cache_base, k_cache_data, kv_cache_bytes(num_kv_heads, max_seq_len, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        bind_or_mark(v_cache_base, v_cache_data, kv_cache_bytes(num_kv_heads, max_seq_len, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        if (attn_mask != nullptr && !attn_mask_data.empty())
            bind_or_mark(attn_mask, attn_mask_data.data(), attn_mask_data.size() * sizeof(ggml_fp16_t), false);

        // Allocate backend buffer for remaining tensors (intermediates + non-host-ptr tensors)
        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate backend buffer for transformer layer decode.");
            return 0;
        }

        // Drain pending async work before CPU memcpys from C# tensor buffers.
        host_read_barrier();

        // Upload non-host-ptr tensors
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, u.bytes > 0 ? 0 : 0, u.bytes);

        ggml_backend_tensor_set(input, hidden_data, 0, static_cast<std::size_t>(hidden_size) * sizeof(float));

        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));

        // Execute
        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for transformer layer decode.");
            return 0;
        }

        // Download updated hidden state (queued async on Metal in async mode)
        finalize_compute_with_download(hidden_out, hidden_data, static_cast<std::size_t>(hidden_size) * sizeof(float));

        clear_last_error();
        return 1;
    }
}

TSG_EXPORT int TSGgml_TransformerLayerDecode(
    float* hidden_data, int hidden_size,
    float* attn_norm_data,
    void* qkv_data, int qkv_type, std::int64_t qkv_ne0, std::int64_t qkv_ne1, std::int64_t qkv_bytes,
    float* q_norm_data, float* k_norm_data, int head_dim,
    void* o_data, int o_type, std::int64_t o_ne0, std::int64_t o_ne1, std::int64_t o_bytes,
    float* ffn_norm_data,
    void* gu_data, int gu_type, std::int64_t gu_ne0, std::int64_t gu_ne1, std::int64_t gu_bytes,
    void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_bytes,
    void* k_cache_data, void* v_cache_data,
    int num_heads, int num_kv_heads,
    int max_seq_len, int position,
    float eps, float rope_base, float rope_freq_scale,
    int intermediate_size, int rope_mode,
    int kv_cache_type)
{
    try
    {
        return transformer_layer_decode_impl(
            hidden_data, hidden_size,
            attn_norm_data,
            qkv_data, qkv_type, qkv_ne0, qkv_ne1, qkv_bytes,
            q_norm_data, k_norm_data, head_dim,
            o_data, o_type, o_ne0, o_ne1, o_bytes,
            ffn_norm_data,
            gu_data, gu_type, gu_ne0, gu_ne1, gu_bytes,
            down_data, down_type, down_ne0, down_ne1, down_bytes,
            k_cache_data, v_cache_data,
            num_heads, num_kv_heads,
            max_seq_len, position,
            eps, rope_base, rope_freq_scale,
            intermediate_size, rope_mode,
            kv_cache_type);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in transformer layer decode.");
        return 0;
    }
}

// ============================================================================
// Qwen3.5 attention layer decode kernel (single token, single layer).
//
// Performs the full Qwen3.5 FullAttention block in a single GGML graph:
//   1. RMSNorm(hidden) * attn_norm_w
//   2. fused QKV matmul -> [Q_with_gate_interleaved (2*qDim), K (kvDim), V (kvDim)]
//   3. deinterleave Q and gate (each [num_heads, head_dim])
//   4. RMSNorm(Q) * q_norm_w  per head
//      RMSNorm(K) * k_norm_w  per head
//   5. RoPE on Q and K at `position`
//   6. append K, V into the persistent KV cache at `position`
//   7. flash attention against the populated KV cache window -> attn_out
//   8. attn_out *= sigmoid(gate)
//   9. residual += matmul(attn_out_flat, output_w)
//
// Replaces:
//   - 1 FusedRmsNormMatMulQuant call (norm + qkv)
//   - ~6 small CPU ops between (QK norm, RoPE, sigmoid gate, KV cache write)
//   - 1 FusedMatMulQuantAdd call (output + residual)
// with a single graph dispatch. Eliminates ~2 Metal command buffer dispatches
// + several CPU/GPU sync points per attention layer per decode token.
//
// All weights and the KV cache are bound zero-copy via host-pointer buffers
// when supported (Apple Silicon Metal, GGML CPU backend, integrated GPUs).
// ============================================================================
namespace
{
    int qwen35_attn_layer_decode_impl(
        float* residual_data, int hidden_size,
        float* attn_norm_data,
        void* qkv_data, int qkv_type,
        std::int64_t qkv_ne0, std::int64_t qkv_ne1, std::int64_t qkv_bytes,
        float* q_norm_data, float* k_norm_data, int head_dim,
        void* o_data, int o_type,
        std::int64_t o_ne0, std::int64_t o_ne1, std::int64_t o_bytes,
        void* k_cache_data, void* v_cache_data,
        int num_heads, int num_kv_heads,
        int max_seq_len, int position,
        float eps, float rope_base, float rope_freq_scale,
        int rope_mode,
        int kv_cache_type = GGML_TYPE_F32)
    {
        if (!ensure_backend())
            return 0;

        if (residual_data == nullptr || attn_norm_data == nullptr ||
            qkv_data == nullptr || q_norm_data == nullptr || k_norm_data == nullptr ||
            o_data == nullptr || k_cache_data == nullptr || v_cache_data == nullptr)
        {
            set_last_error("Null pointer passed to Qwen3.5 attention layer decode kernel.");
            return 0;
        }
        if (num_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 || max_seq_len <= 0 || position < 0)
        {
            set_last_error("Invalid dimensions passed to Qwen3.5 attention layer decode kernel.");
            return 0;
        }

        const int qDim = num_heads * head_dim;          // post-deinterleave Q dim
        const int qFullDim = qDim * 2;                  // pre-deinterleave Q+gate dim
        const int kDim = num_kv_heads * head_dim;
        const int totalSeqLen = position + 1;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int attnKvLen = flash_attn_kv_length(totalSeqLen, max_seq_len, head_dim);
        std::vector<ggml_fp16_t> attn_mask_data;

        const std::size_t ctx_size = 2 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for Qwen3.5 attention layer decode.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // Inputs / outputs
        ggml_tensor* residual_in   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* attn_norm_w   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* q_norm_w      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
        ggml_tensor* k_norm_w      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
        ggml_tensor* qkv_w         = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkv_type), qkv_ne0, qkv_ne1);
        ggml_tensor* o_w           = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type), o_ne0, o_ne1);
        ggml_tensor* pos_tensor    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* k_cache_base  = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, max_seq_len, num_kv_heads);
        ggml_tensor* v_cache_base  = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, max_seq_len, num_kv_heads);
        ggml_tensor* residual_out  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* attn_mask = nullptr;
        if (flash_attn_requires_masked_padding(head_dim))
        {
            attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
            fill_flash_attn_mask(attn_mask_data, attnKvLen, totalSeqLen);
        }

        if (residual_in == nullptr || attn_norm_w == nullptr || q_norm_w == nullptr ||
            k_norm_w == nullptr || qkv_w == nullptr || o_w == nullptr || pos_tensor == nullptr ||
            k_cache_base == nullptr || v_cache_base == nullptr || residual_out == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors for Qwen3.5 attention layer decode.");
            return 0;
        }

        // === Build computation graph ===

        // 1. Attention norm: RMSNorm + element-wise scale
        ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual_in, eps), attn_norm_w);

        // 2. Fused QKV projection: [hidden] -> [qFullDim + 2*kvDim]
        ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, hidden_size, 1);
        ggml_tensor* qkv_flat  = ggml_reshape_1d(
            ctx,
            ggml_mul_mat(ctx, qkv_w, normed_2d),
            qFullDim + 2 * kDim);

        // 3. Slice fused QKV into Q+gate, K, V
        //    The Q part has layout [head0_Q, head0_gate, head1_Q, head1_gate, ...] in memory:
        //    interpreted as a 3D tensor [head_dim, 2, num_heads] with row-major (C) layout
        //    where the innermost stride is sizeof(float).
        ggml_tensor* qg_part = ggml_view_1d(ctx, qkv_flat, qFullDim, 0);
        ggml_tensor* k_raw   = ggml_view_1d(ctx, qkv_flat, kDim,
            static_cast<std::size_t>(qFullDim) * sizeof(float));
        ggml_tensor* v_raw   = ggml_view_1d(ctx, qkv_flat, kDim,
            static_cast<std::size_t>(qFullDim + kDim) * sizeof(float));

        ggml_tensor* qg_3d = ggml_reshape_3d(ctx, qg_part, head_dim, 2, num_heads);

        // Q view: [head_dim, num_heads] strided (skip the gate half)
        ggml_tensor* q_view = ggml_view_2d(
            ctx, qg_3d, head_dim, num_heads,
            qg_3d->nb[2], 0);
        ggml_tensor* gate_view = ggml_view_2d(
            ctx, qg_3d, head_dim, num_heads,
            qg_3d->nb[2], qg_3d->nb[1]);

        // We need contiguous Q for the per-head RMSNorm + RoPE that follow.
        ggml_tensor* q_2d_raw = ggml_cont(ctx, q_view);
        ggml_tensor* k_2d_raw = ggml_reshape_2d(ctx, k_raw, head_dim, num_kv_heads);

        // 4. Per-head QK norm
        ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d_raw, eps), q_norm_w);
        ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d_raw, eps), k_norm_w);

        // 5. RoPE (NeoX style for Qwen3.5)
        ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, head_dim, num_heads, 1);
        ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, head_dim, num_kv_heads, 1);

        ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, nullptr,
            head_dim, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
        ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, nullptr,
            head_dim, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);

        // 6. Append K, V into the persistent cache at `position`
        // q_rope: [head_dim, num_heads, 1] -> q_attn: [head_dim, 1, num_heads]
        ggml_tensor* q_attn       = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
        ggml_tensor* k_rope_perm  = ggml_permute(ctx, k_rope, 0, 2, 1, 3);
        ggml_tensor* v_3d         = ggml_reshape_3d(ctx, v_raw, head_dim, num_kv_heads, 1);
        ggml_tensor* v_perm       = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
        ggml_tensor* k_write      = ggml_cont(ctx, k_rope_perm);
        ggml_tensor* v_write      = ggml_cont(ctx, v_perm);
        const std::size_t kv_byte_offset =
            static_cast<std::size_t>(position) * k_cache_base->nb[1];
        ggml_tensor* k_dst = ggml_view_3d(ctx, k_cache_base,
            head_dim, 1, num_kv_heads,
            k_cache_base->nb[1], k_cache_base->nb[2], kv_byte_offset);
        ggml_tensor* v_dst = ggml_view_3d(ctx, v_cache_base,
            head_dim, 1, num_kv_heads,
            v_cache_base->nb[1], v_cache_base->nb[2], kv_byte_offset);
        ggml_tensor* k_cache_cpy = ggml_cpy(ctx, k_write, k_dst);
        ggml_tensor* v_cache_cpy = ggml_cpy(ctx, v_write, v_dst);

        ggml_tensor* k_full = view_kv_cache_window(ctx, k_cache_base, head_dim, max_seq_len, num_kv_heads, 0, attnKvLen, kv_cache_type);
        ggml_tensor* v_full = view_kv_cache_window(ctx, v_cache_base, head_dim, max_seq_len, num_kv_heads, 0, attnKvLen, kv_cache_type);
        if (k_full == nullptr || v_full == nullptr)
        {
            set_last_error("Failed to create KV cache views for Qwen3.5 attention layer decode.");
            return 0;
        }

        // 7. Flash attention (handles GQA broadcasting)
        ggml_tensor* attn_out_4d = ggml_flash_attn_ext(ctx,
            q_attn, k_full, v_full, attn_mask, scale, 0.0f, 0.0f);

        // attn_out_4d: [head_dim, num_heads, 1] -> reshape to [head_dim, num_heads]
        ggml_tensor* attn_out_2d = ggml_reshape_2d(ctx, attn_out_4d, head_dim, num_heads);

        // 8. Sigmoid-gated mix: attn_out *= sigmoid(gate)
        // gate_view is the strided view into the QKV output; need it contiguous for elementwise mul.
        ggml_tensor* gate_2d = ggml_cont(ctx, gate_view);
        ggml_tensor* gate_sig = ggml_sigmoid(ctx, gate_2d);
        ggml_tensor* attn_gated = ggml_mul(ctx, attn_out_2d, gate_sig);

        // 9. Output projection + residual: residual += matmul(attn_gated_flat, o_w)
        ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_gated, qDim, 1);
        ggml_tensor* o_flat    = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, o_w, attn_flat), hidden_size);
        ggml_tensor* result    = ggml_add(ctx, residual_in, o_flat);

        ggml_tensor* out_residual = ggml_cpy(ctx, result, residual_out);
        ggml_set_output(out_residual);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, k_cache_cpy);
        ggml_build_forward_expand(graph, v_cache_cpy);
        ggml_build_forward_expand(graph, out_residual);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);

        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr)
                return;

            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs_upload, usage))
                {
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, addr);
                    if (st == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload)
                            upload_list.push_back({t, data, bytes});
                        return;
                    }
                    invalidate_cached_buffer(data);
                }
            }

            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable)
                        ephemeral_bufs.emplace_back(buf);
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, data);
                    if (st == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        bind_or_mark(qkv_w,        qkv_data,        static_cast<std::size_t>(qkv_bytes), true);
        bind_or_mark(o_w,          o_data,          static_cast<std::size_t>(o_bytes),   true);
        bind_or_mark(attn_norm_w,  attn_norm_data,  static_cast<std::size_t>(hidden_size) * sizeof(float), true);
        bind_or_mark(q_norm_w,     q_norm_data,     static_cast<std::size_t>(head_dim)    * sizeof(float), true);
        bind_or_mark(k_norm_w,     k_norm_data,     static_cast<std::size_t>(head_dim)    * sizeof(float), true);
        bind_or_mark(k_cache_base, k_cache_data,    kv_cache_bytes(num_kv_heads, max_seq_len, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        bind_or_mark(v_cache_base, v_cache_data,    kv_cache_bytes(num_kv_heads, max_seq_len, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        if (attn_mask != nullptr && !attn_mask_data.empty())
            bind_or_mark(attn_mask, attn_mask_data.data(), attn_mask_data.size() * sizeof(ggml_fp16_t), false);

        // Bind the input residual buffer directly so that the output write goes
        // back into the caller's memory without an explicit download. Falls back
        // to upload+download when the host pointer is not cacheable.
        ggml_backend_buffer_t res_in_buf = nullptr;
        bool residual_zero_copy = try_get_host_ptr_buffer(g_backend, dev, residual_data,
            static_cast<std::size_t>(hidden_size) * sizeof(float), false, res_in_buf);
        if (residual_zero_copy)
        {
            ephemeral_bufs.emplace_back(res_in_buf);
            ggml_status st = ggml_backend_tensor_alloc(res_in_buf, residual_in, residual_data);
            if (st != GGML_STATUS_SUCCESS)
                residual_zero_copy = false;
        }

        ggml_backend_buffer_t res_out_buf = nullptr;
        bool residual_out_zero_copy = try_get_host_ptr_buffer(g_backend, dev, residual_data,
            static_cast<std::size_t>(hidden_size) * sizeof(float), false, res_out_buf);
        if (residual_out_zero_copy)
        {
            ephemeral_bufs.emplace_back(res_out_buf);
            ggml_status st = ggml_backend_tensor_alloc(res_out_buf, residual_out, residual_data);
            if (st != GGML_STATUS_SUCCESS)
                residual_out_zero_copy = false;
        }

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate backend buffer for Qwen3.5 attention layer decode.");
            return 0;
        }

        // Drain pending async work before CPU memcpys from C# tensor buffers.
        host_read_barrier();

        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);

        if (!residual_zero_copy)
            ggml_backend_tensor_set(residual_in, residual_data,
                0, static_cast<std::size_t>(hidden_size) * sizeof(float));

        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for Qwen3.5 attention layer decode.");
            return 0;
        }

        finalize_compute(residual_out_zero_copy, residual_out, residual_data,
            static_cast<std::size_t>(hidden_size) * sizeof(float));

        clear_last_error();
        return 1;
    }
}

TSG_EXPORT int TSGgml_Qwen35AttentionLayerDecode(
    float* residual_data, int hidden_size,
    float* attn_norm_data,
    void* qkv_data, int qkv_type, std::int64_t qkv_ne0, std::int64_t qkv_ne1, std::int64_t qkv_bytes,
    float* q_norm_data, float* k_norm_data, int head_dim,
    void* o_data, int o_type, std::int64_t o_ne0, std::int64_t o_ne1, std::int64_t o_bytes,
    void* k_cache_data, void* v_cache_data,
    int num_heads, int num_kv_heads,
    int max_seq_len, int position,
    float eps, float rope_base, float rope_freq_scale,
    int rope_mode,
    int kv_cache_type)
{
    try
    {
        return qwen35_attn_layer_decode_impl(
            residual_data, hidden_size,
            attn_norm_data,
            qkv_data, qkv_type, qkv_ne0, qkv_ne1, qkv_bytes,
            q_norm_data, k_norm_data, head_dim,
            o_data, o_type, o_ne0, o_ne1, o_bytes,
            k_cache_data, v_cache_data,
            num_heads, num_kv_heads,
            max_seq_len, position,
            eps, rope_base, rope_freq_scale, rope_mode,
            kv_cache_type);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in Qwen3.5 attention layer decode.");
        return 0;
    }
}

// ============================================================================
// Flash attention decode (single-token, single-layer).
//
// Use this when the surrounding architecture pre-processes Q/K/V (e.g. fused
// gated projections, sigmoid-gated Q outputs, custom QK normalization) in a
// way that prevents folding the entire layer into the model-decode kernel.
// ============================================================================
TSG_EXPORT int TSGgml_FlashAttnDecodeF32(
    const float* q_data,
    const float* k_data,
    const float* v_data,
    void* k_cache_data,
    void* v_cache_data,
    float* out_data,
    int num_heads, int num_kv_heads, int head_dim,
    int max_seq_len, int position,
    float scale,
    int kv_cache_type)
{
    try
    {
        return flash_attn_decode_impl(
            q_data, k_data, v_data,
            k_cache_data, v_cache_data,
            out_data,
            num_heads, num_kv_heads, head_dim,
            max_seq_len, position, scale,
            kv_cache_type);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in flash attention decode.");
        return 0;
    }
}

// ============================================================================
// Full-model decode: ALL transformer layers in a single GGML graph.
// Eliminates per-layer Metal synchronization overhead.
// ============================================================================

TSG_EXPORT int TSGgml_TransformerModelDecode(
    float* hidden_data, int hidden_size, int num_layers,
    void** attn_norm_arr, void** qkv_arr, void** q_norm_arr, void** k_norm_arr,
    void** o_arr, void** ffn_norm_arr, void** gu_arr, void** down_arr,
    void** k_cache_arr, void** v_cache_arr,
    int qkv_type, std::int64_t qkv_ne0, std::int64_t qkv_ne1, std::int64_t qkv_bytes,
    int o_type, std::int64_t o_ne0, std::int64_t o_ne1, std::int64_t o_bytes,
    int gu_type, std::int64_t gu_ne0, std::int64_t gu_ne1, std::int64_t gu_bytes,
    int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_bytes,
    int head_dim, int num_heads, int num_kv_heads,
    int max_seq_len, int position,
    float eps, float rope_base, float rope_freq_scale,
    int intermediate_size, int rope_mode,
    int kv_cache_type)
{
    try
    {
        if (!ensure_backend())
            return 0;

        const int qDim = num_heads * head_dim;
        const int kDim = num_kv_heads * head_dim;
        const int totalSeqLen = position + 1;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int attnKvLen = flash_attn_kv_length(totalSeqLen, max_seq_len, head_dim);
        std::vector<ggml_fp16_t> attn_mask_data;

        // Large context for all layers
        const std::size_t ctx_size = 16 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for model decode.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // Input tensor (shared across graph)
        ggml_tensor* current = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* attn_mask = nullptr;
        if (flash_attn_requires_masked_padding(head_dim))
        {
            attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
            fill_flash_attn_mask(attn_mask_data, attnKvLen, totalSeqLen);
        }

        // Per-layer weight tensors and KV cache tensors
        struct LayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* q_norm_w;
            ggml_tensor* k_norm_w;
            ggml_tensor* o_w;
            ggml_tensor* ffn_norm_w;
            ggml_tensor* gu_w;
            ggml_tensor* down_w;
            ggml_tensor* k_cache_base;
            ggml_tensor* v_cache_base;
            ggml_tensor* k_cache_cpy;
            ggml_tensor* v_cache_cpy;
        };
        std::vector<LayerTensors> layers(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            lt.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.qkv_w  = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkv_type), qkv_ne0, qkv_ne1);
            lt.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            lt.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            lt.o_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type), o_ne0, o_ne1);
            lt.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.gu_w   = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gu_type), gu_ne0, gu_ne1);
            lt.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type), down_ne0, down_ne1);
            lt.k_cache_base = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, max_seq_len, num_kv_heads);
            lt.v_cache_base = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, max_seq_len, num_kv_heads);
        }

        // Build computation graph: chain all layers
        ggml_tensor* hidden = current;

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];

            // Attention norm
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), lt.attn_norm_w);

            // Fused QKV projection
            ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, hidden_size, 1);
            ggml_tensor* qkv_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.qkv_w, normed_2d), qDim + 2 * kDim);

            // Split Q, K, V
            ggml_tensor* q_raw = ggml_view_1d(ctx, qkv_flat, qDim, 0);
            ggml_tensor* k_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim) * sizeof(float));
            ggml_tensor* v_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim + kDim) * sizeof(float));

            // Per-head QK norm
            ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_raw, head_dim, num_heads);
            ggml_tensor* k_2d = ggml_reshape_2d(ctx, k_raw, head_dim, num_kv_heads);

            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), lt.q_norm_w);
            ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d, eps), lt.k_norm_w);

            // RoPE
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, head_dim, num_heads, 1);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, head_dim, num_kv_heads, 1);

            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, nullptr,
                head_dim, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
            ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, nullptr,
                head_dim, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);

            // Build full KV sequence
            ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
            ggml_tensor* k_rope_perm = ggml_permute(ctx, k_rope, 0, 2, 1, 3);
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_raw, head_dim, num_kv_heads, 1);
            ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
            ggml_tensor* k_write = ggml_cont(ctx, k_rope_perm);
            ggml_tensor* v_write = ggml_cont(ctx, v_perm);
            const std::size_t kv_byte_offset =
                static_cast<std::size_t>(position) * lt.k_cache_base->nb[1];
            ggml_tensor* k_dst = ggml_view_3d(ctx, lt.k_cache_base,
                head_dim, 1, num_kv_heads,
                lt.k_cache_base->nb[1], lt.k_cache_base->nb[2], kv_byte_offset);
            ggml_tensor* v_dst = ggml_view_3d(ctx, lt.v_cache_base,
                head_dim, 1, num_kv_heads,
                lt.v_cache_base->nb[1], lt.v_cache_base->nb[2], kv_byte_offset);
            lt.k_cache_cpy = ggml_cpy(ctx, k_write, k_dst);
            lt.v_cache_cpy = ggml_cpy(ctx, v_write, v_dst);
            ggml_tensor* k_full = view_kv_cache_window(ctx, lt.k_cache_base, head_dim, max_seq_len, num_kv_heads, 0, attnKvLen, kv_cache_type);
            ggml_tensor* v_full = view_kv_cache_window(ctx, lt.v_cache_base, head_dim, max_seq_len, num_kv_heads, 0, attnKvLen, kv_cache_type);
            if (k_full == nullptr || v_full == nullptr)
            {
                set_last_error("Failed to create KV cache views for transformer model decode.");
                return 0;
            }

            // Flash attention
            ggml_tensor* attn_out = ggml_flash_attn_ext(ctx,
                q_attn, k_full, v_full, attn_mask, scale, 0.0f, 0.0f);

            // O projection + residual
            ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, qDim, 1);
            ggml_tensor* o_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.o_w, attn_flat), hidden_size);
            ggml_tensor* residual1 = ggml_add(ctx, hidden, o_flat);

            // FFN
            ggml_tensor* normed2 = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), lt.ffn_norm_w);
            ggml_tensor* normed2_2d = ggml_reshape_2d(ctx, normed2, hidden_size, 1);
            ggml_tensor* gu_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.gu_w, normed2_2d), 2 * intermediate_size);

            ggml_tensor* gate = ggml_view_1d(ctx, gu_flat, intermediate_size, 0);
            ggml_tensor* up = ggml_view_1d(ctx, gu_flat, intermediate_size,
                                           static_cast<std::size_t>(intermediate_size) * sizeof(float));
            ggml_tensor* ffn_hidden = ggml_mul(ctx, ggml_silu(ctx, gate), up);

            ggml_tensor* ffn_2d = ggml_reshape_2d(ctx, ffn_hidden, intermediate_size, 1);
            ggml_tensor* down_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.down_w, ffn_2d), hidden_size);

            // Second residual - this becomes 'hidden' for the next layer
            hidden = ggml_add(ctx, residual1, down_flat);

        }

        // Output: copy hidden state so we can download it
        ggml_tensor* hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* out_hidden = ggml_cpy(ctx, hidden, hidden_out);
        ggml_set_output(out_hidden);

        // Build graph: add KV cache writes first to ensure they execute before reads
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * 64 + 256;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
        {
            ggml_build_forward_expand(graph, layers[l].k_cache_cpy);
            ggml_build_forward_expand(graph, layers[l].v_cache_cpy);
        }
        ggml_build_forward_expand(graph, out_hidden);

        // Bind weights via cached host_ptr
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);

        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr)
                return;

            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs_upload, usage))
                {
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, addr);
                    if (st == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload)
                            upload_list.push_back({t, data, bytes});
                        return;
                    }

                    invalidate_cached_buffer(data);
                }
            }

            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable)
                        ephemeral_bufs.emplace_back(buf);
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, data);
                    if (st == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            bind_or_mark(lt.qkv_w,  qkv_arr[l],  static_cast<std::size_t>(qkv_bytes), true);
            bind_or_mark(lt.o_w,    o_arr[l],     static_cast<std::size_t>(o_bytes), true);
            bind_or_mark(lt.gu_w,   gu_arr[l],    static_cast<std::size_t>(gu_bytes), true);
            bind_or_mark(lt.down_w, down_arr[l],  static_cast<std::size_t>(down_bytes), true);

            bind_or_mark(lt.attn_norm_w, attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.ffn_norm_w,  ffn_norm_arr[l],  static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.q_norm_w,    q_norm_arr[l],    static_cast<std::size_t>(head_dim) * sizeof(float), true);
            bind_or_mark(lt.k_norm_w,    k_norm_arr[l],    static_cast<std::size_t>(head_dim) * sizeof(float), true);
            bind_or_mark(lt.k_cache_base, k_cache_arr[l], kv_cache_bytes(num_kv_heads, max_seq_len, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            bind_or_mark(lt.v_cache_base, v_cache_arr[l], kv_cache_bytes(num_kv_heads, max_seq_len, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        if (attn_mask != nullptr && !attn_mask_data.empty())
            bind_or_mark(attn_mask, attn_mask_data.data(), attn_mask_data.size() * sizeof(ggml_fp16_t), false);

        // Allocate backend buffer for intermediates
        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate backend buffer for model decode.");
            return 0;
        }

        // Drain pending async work before CPU memcpys from C# tensor buffers.
        host_read_barrier();

        // Upload non-bound tensors
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);

        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(hidden_size) * sizeof(float));

        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));

        // Execute single graph
        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for model decode.");
            return 0;
        }

        // Download hidden state back to caller (async blit on Metal in async mode)
        finalize_compute_with_download(hidden_out, hidden_data, static_cast<std::size_t>(hidden_size) * sizeof(float));

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
        set_last_error("Unknown error in transformer model decode.");
        return 0;
    }
}

// ============================================================================
// Gemma4 full-model decode: ALL dense transformer layers in a single GGML graph.
// Handles Gemma4-specific features: GELU activation, V norm, post-attn/FFN norms,
// layer scalars, different head dims per layer type, sliding window, softcap.
// ============================================================================

TSG_EXPORT int TSGgml_Gemma4ModelDecode(
    float* hidden_data, int hidden_size, int num_layers,
    // Per-layer weight pointers (arrays of size num_layers)
    void** attn_norm_arr,
    void** qkv_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr,
    void** post_attn_norm_arr,
    void** ffn_norm_arr,
    void** gu_arr, void** down_arr,
    void** post_ffn_norm_arr,
    // Per-layer KV caches
    void** k_cache_arr, void** v_cache_arr,
    // Per-layer metadata (arrays of size num_layers)
    int* head_dim_arr,
    int* kv_heads_arr,
    int* cache_size_arr,
    int* is_local_arr,
    int* kv_source_arr,
    float* rope_base_arr,
    float* layer_scalar_arr,
    // Per-layer weight shapes
    int* qkv_type_arr, std::int64_t* qkv_ne0_arr, std::int64_t* qkv_ne1_arr, std::int64_t* qkv_bytes_arr,
    int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    int* gu_type_arr, std::int64_t* gu_ne0_arr, std::int64_t* gu_ne1_arr, std::int64_t* gu_bytes_arr,
    int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    // Global params
    int num_heads, int position,
    float eps, int sliding_window,
    // RoPE freq_factors (nullable, for global layers with proportional RoPE)
    float* rope_freq_factors, int rope_freq_factors_len,
    int* rope_n_dims_arr,
    // PLE data (nullable)
    float* ple_data, int ple_dim,
    void** ple_gate_arr, int* ple_gate_type_arr, std::int64_t* ple_gate_ne0_arr, std::int64_t* ple_gate_ne1_arr, std::int64_t* ple_gate_bytes_arr,
    void** ple_proj_arr, int* ple_proj_type_arr, std::int64_t* ple_proj_ne0_arr, std::int64_t* ple_proj_ne1_arr, std::int64_t* ple_proj_bytes_arr,
    void** ple_post_norm_arr,
    int kv_cache_type)
{
    try
    {
        if (!ensure_backend())
            return 0;

        const int totalSeqLen = position + 1;

        // Compute max head dim for context sizing
        int maxHd = 0;
        for (int l = 0; l < num_layers; l++)
            if (head_dim_arr[l] > maxHd) maxHd = head_dim_arr[l];

        // Prepare per-layer KV cache metadata
        struct LayerInfo {
            int hd;
            int kvHeads;
            int qDim;
            int kDim;
            int cacheSize;
            bool isLocal;
            bool isShared;
            int kvSource;
            int attendLen;
        };
        std::vector<LayerInfo> li(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            auto& info = li[l];
            info.hd = head_dim_arr[l];
            info.kvHeads = kv_heads_arr[l];
            info.qDim = num_heads * info.hd;
            info.kDim = info.kvHeads * info.hd;
            info.kvSource = kv_source_arr[l];
            info.isShared = (info.kvSource != l);

            // For shared layers, use the donor's cache size/local flag
            int kvSrc = info.kvSource;
            info.cacheSize = cache_size_arr[kvSrc];
            info.isLocal = is_local_arr[kvSrc] != 0;
            info.attendLen = info.isLocal ? std::min(totalSeqLen, sliding_window) : totalSeqLen;
        }

        // Create GGML context
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for Gemma4 model decode.");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_tensor* current = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

        ggml_tensor* freq_factors_t = nullptr;
        if (rope_freq_factors != nullptr && rope_freq_factors_len > 0)
            freq_factors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rope_freq_factors_len);

        // PLE input
        ggml_tensor* ple_input = nullptr;
        if (ple_data != nullptr && ple_dim > 0)
            ple_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_layers * ple_dim);

        struct LayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* q_norm_w;
            ggml_tensor* k_norm_w;
            ggml_tensor* o_w;
            ggml_tensor* post_attn_norm_w;
            ggml_tensor* ffn_norm_w;
            ggml_tensor* gu_w;
            ggml_tensor* down_w;
            ggml_tensor* post_ffn_norm_w;
            ggml_tensor* k_cached_t;
            ggml_tensor* v_cached_t;
            ggml_tensor* k_cpy;
            ggml_tensor* v_cpy;
            // PLE
            ggml_tensor* ple_gate_w;
            ggml_tensor* ple_proj_w;
            ggml_tensor* ple_post_norm_w;
        };
        std::vector<LayerTensors> layers(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];

            lt.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkv_type_arr[l]), qkv_ne0_arr[l], qkv_ne1_arr[l]);
            lt.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type_arr[l]), o_ne0_arr[l], o_ne1_arr[l]);
            lt.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gu_type_arr[l]), gu_ne0_arr[l], gu_ne1_arr[l]);
            lt.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type_arr[l]), down_ne0_arr[l], down_ne1_arr[l]);
            lt.post_ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            if (!info.isShared)
            {
                lt.k_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, info.cacheSize, info.kvHeads);
                lt.v_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, info.cacheSize, info.kvHeads);
            }
            else
            {
                lt.k_cached_t = nullptr;
                lt.v_cached_t = nullptr;
            }

            lt.k_cpy = nullptr;
            lt.v_cpy = nullptr;

            lt.ple_gate_w = nullptr;
            lt.ple_proj_w = nullptr;
            lt.ple_post_norm_w = nullptr;
            if (ple_data != nullptr && ple_gate_arr != nullptr && ple_gate_arr[l] != nullptr)
            {
                lt.ple_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_gate_type_arr[l]),
                    ple_gate_ne0_arr[l], ple_gate_ne1_arr[l]);
                lt.ple_proj_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_proj_type_arr[l]),
                    ple_proj_ne0_arr[l], ple_proj_ne1_arr[l]);
                lt.ple_post_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            }
        }

        // Link shared layers to donor KV tensors
        for (int l = 0; l < num_layers; l++)
        {
            auto& info = li[l];
            if (info.isShared)
            {
                layers[l].k_cached_t = layers[info.kvSource].k_cached_t;
                layers[l].v_cached_t = layers[info.kvSource].v_cached_t;
            }
        }

        // Build compute graph
        ggml_tensor* hidden = current;

        // Track the active KV tensors produced by each donor layer.
        std::vector<ggml_tensor*> layer_k_full(num_layers, nullptr);
        std::vector<ggml_tensor*> layer_v_full(num_layers, nullptr);
        std::vector<ggml_tensor*> layer_attn_mask(num_layers, nullptr);
        std::vector<std::vector<ggml_fp16_t>> layer_attn_mask_data(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];
            float rope_base = rope_base_arr[l];

            // 1. Attn norm
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), lt.attn_norm_w);

            ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, hidden_size, 1);
            ggml_tensor* q_rope;
            ggml_tensor* k_full;
            ggml_tensor* v_full;

            if (!info.isShared)
            {
                // 2. Fused QKV projection
                ggml_tensor* qkv_flat = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, lt.qkv_w, normed_2d), info.qDim + 2 * info.kDim);
                ggml_tensor* q_raw = ggml_view_1d(ctx, qkv_flat, info.qDim, 0);
                ggml_tensor* k_raw = ggml_view_1d(ctx, qkv_flat, info.kDim,
                    static_cast<std::size_t>(info.qDim) * sizeof(float));
                ggml_tensor* v_raw = ggml_view_1d(ctx, qkv_flat, info.kDim,
                    static_cast<std::size_t>(info.qDim + info.kDim) * sizeof(float));

                // Per-head Q/K norm
                ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_raw, info.hd, num_heads);
                ggml_tensor* k_2d = ggml_reshape_2d(ctx, k_raw, info.hd, info.kvHeads);
                ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), lt.q_norm_w);
                ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d, eps), lt.k_norm_w);

                // V norm (unweighted RMSNorm)
                ggml_tensor* v_2d = ggml_reshape_2d(ctx, v_raw, info.hd, info.kvHeads);
                ggml_tensor* v_normed = ggml_rms_norm(ctx, v_2d, eps);

                // RoPE (use per-layer n_dims and optional freq_factors)
                int rope_dims = rope_n_dims_arr[l];
                ggml_tensor* rope_ff = info.isLocal ? nullptr : freq_factors_t;
                ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, info.hd, num_heads, 1);
                ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, info.hd, info.kvHeads, 1);
                q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff,
                    rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);
                ggml_tensor* k_rope_t = ggml_rope_ext(ctx, k_3d, pos_tensor, rope_ff,
                    rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);

                ggml_tensor* k_rope_perm = ggml_permute(ctx, k_rope_t, 0, 2, 1, 3);
                ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_normed, info.hd, info.kvHeads, 1);
                ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
                ggml_tensor* k_write = ggml_cont(ctx, k_rope_perm);
                ggml_tensor* v_write = ggml_cont(ctx, v_perm);
                const int cachePos = info.isLocal ? (position % info.cacheSize) : position;
                const int activeStart = info.isLocal ? ((totalSeqLen - info.attendLen) % info.cacheSize) : 0;
                const int attnKvLen = flash_attn_kv_length(info.attendLen, info.cacheSize, info.hd);
                const std::size_t kv_byte_offset =
                    static_cast<std::size_t>(cachePos) * lt.k_cached_t->nb[1];
                ggml_tensor* k_dst = ggml_view_3d(ctx, lt.k_cached_t,
                    info.hd, 1, info.kvHeads,
                    lt.k_cached_t->nb[1], lt.k_cached_t->nb[2], kv_byte_offset);
                ggml_tensor* v_dst = ggml_view_3d(ctx, lt.v_cached_t,
                    info.hd, 1, info.kvHeads,
                    lt.v_cached_t->nb[1], lt.v_cached_t->nb[2], kv_byte_offset);
                lt.k_cpy = ggml_cpy(ctx, k_write, k_dst);
                lt.v_cpy = ggml_cpy(ctx, v_write, v_dst);
                if (flash_attn_requires_masked_padding(info.hd))
                {
                    layer_attn_mask[l] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
                    fill_flash_attn_mask(layer_attn_mask_data[l], attnKvLen, info.attendLen);
                }
                k_full = view_kv_cache_window(ctx, lt.k_cached_t, info.hd, info.cacheSize, info.kvHeads, activeStart, attnKvLen, kv_cache_type);
                v_full = view_kv_cache_window(ctx, lt.v_cached_t, info.hd, info.cacheSize, info.kvHeads, activeStart, attnKvLen, kv_cache_type);
                if (k_full == nullptr || v_full == nullptr)
                {
                    set_last_error("Failed to create Gemma4 KV cache views.");
                    return 0;
                }
                layer_k_full[l] = k_full;
                layer_v_full[l] = v_full;
            }
            else
            {
                // Shared layer: Q-only projection (qkv_w is just Q weight)
                ggml_tensor* q_flat = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, lt.qkv_w, normed_2d), info.qDim);
                ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_flat, info.hd, num_heads);
                ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), lt.q_norm_w);
                int rope_dims = rope_n_dims_arr[l];
                ggml_tensor* rope_ff = info.isLocal ? nullptr : freq_factors_t;
                ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, info.hd, num_heads, 1);
                q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff,
                    rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);

                // Use the donor layer's K/V (already computed earlier in the graph)
                int donor = info.kvSource;
                k_full = layer_k_full[donor];
                v_full = layer_v_full[donor];
                layer_attn_mask[l] = layer_attn_mask[donor];
                if (k_full == nullptr || v_full == nullptr)
                {
                    set_last_error("Shared layer has no KV data available.");
                    return 0;
                }
            }

            layer_k_full[l] = k_full;
            layer_v_full[l] = v_full;

            // Flash attention (scale=1.0 due to QK-Norm, no attention softcap)
            ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
            ggml_tensor* attn_out = ggml_flash_attn_ext(ctx,
                q_attn, k_full, v_full, layer_attn_mask[l], 1.0f, 0.0f, 0.0f);

            // 8. O projection
            ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, info.qDim, 1);
            ggml_tensor* o_flat = ggml_reshape_1d(ctx,
                ggml_mul_mat(ctx, lt.o_w, attn_flat), hidden_size);

            // 9. Post-attn norm + residual
            ggml_tensor* post_attn_normed = ggml_mul(ctx,
                ggml_rms_norm(ctx, o_flat, eps), lt.post_attn_norm_w);
            ggml_tensor* residual1 = ggml_add(ctx, hidden, post_attn_normed);

            // 10. FFN: norm → gate_up → GELU*up → down → post_ffn_norm
            ggml_tensor* ffn_normed = ggml_mul(ctx,
                ggml_rms_norm(ctx, residual1, eps), lt.ffn_norm_w);
            ggml_tensor* ffn_normed_2d = ggml_reshape_2d(ctx, ffn_normed, hidden_size, 1);

            std::int64_t intermediate_size = gu_ne1_arr[l] / 2;
            ggml_tensor* gu_flat = ggml_reshape_1d(ctx,
                ggml_mul_mat(ctx, lt.gu_w, ffn_normed_2d), 2 * intermediate_size);
            ggml_tensor* gate = ggml_view_1d(ctx, gu_flat, intermediate_size, 0);
            ggml_tensor* up = ggml_view_1d(ctx, gu_flat, intermediate_size,
                static_cast<std::size_t>(intermediate_size) * sizeof(float));
            ggml_tensor* ffn_hidden = ggml_mul(ctx, ggml_gelu(ctx, gate), up);

            ggml_tensor* ffn_2d = ggml_reshape_2d(ctx, ffn_hidden, intermediate_size, 1);
            ggml_tensor* down_flat = ggml_reshape_1d(ctx,
                ggml_mul_mat(ctx, lt.down_w, ffn_2d), hidden_size);

            // 11. Post-FFN norm + residual
            ggml_tensor* post_ffn_normed = ggml_mul(ctx,
                ggml_rms_norm(ctx, down_flat, eps), lt.post_ffn_norm_w);
            ggml_tensor* residual2 = ggml_add(ctx, residual1, post_ffn_normed);

            // 12. PLE injection (if present)
            if (lt.ple_gate_w != nullptr && ple_input != nullptr)
            {
                ggml_tensor* ple_slice = ggml_view_1d(ctx, ple_input, ple_dim,
                    static_cast<std::size_t>(l) * ple_dim * sizeof(float));
                ggml_tensor* ple_slice_2d = ggml_reshape_2d(ctx, residual2, hidden_size, 1);
                ggml_tensor* ple_gate_proj = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, lt.ple_gate_w, ple_slice_2d), ple_dim);
                ggml_tensor* ple_gated = ggml_mul(ctx, ggml_gelu(ctx, ple_gate_proj), ple_slice);
                ggml_tensor* ple_gated_2d = ggml_reshape_2d(ctx, ple_gated, ple_dim, 1);
                ggml_tensor* ple_proj = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, lt.ple_proj_w, ple_gated_2d), hidden_size);
                ggml_tensor* ple_normed = ggml_mul(ctx,
                    ggml_rms_norm(ctx, ple_proj, eps), lt.ple_post_norm_w);
                residual2 = ggml_add(ctx, residual2, ple_normed);
            }

            // 13. Layer scalar
            float scalar = layer_scalar_arr[l];
            if (std::fabs(scalar - 1.0f) > 1e-6f)
                residual2 = ggml_scale(ctx, residual2, scalar);

            hidden = residual2;
        }

        // Output: copy hidden state
        ggml_tensor* hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* out_hidden = ggml_cpy(ctx, hidden, hidden_out);
        ggml_set_output(out_hidden);

        // Build graph: add KV cache writes first to ensure they execute before reads
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * 128 + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].k_cpy != nullptr)
            {
                ggml_build_forward_expand(graph, layers[l].k_cpy);
                ggml_build_forward_expand(graph, layers[l].v_cpy);
            }
        }
        ggml_build_forward_expand(graph, out_hidden);

        // Bind weight data
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);

        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;

            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs_upload, usage))
                {
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, addr);
                    if (st == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload)
                            upload_list.push_back({t, data, bytes});
                        return;
                    }

                    invalidate_cached_buffer(data);
                }
            }

            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable)
                        ephemeral_bufs.emplace_back(buf);
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, data);
                    if (st == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];

            bind_or_mark(lt.qkv_w, qkv_arr[l], static_cast<std::size_t>(qkv_bytes_arr[l]), true);
            bind_or_mark(lt.o_w, o_arr[l], static_cast<std::size_t>(o_bytes_arr[l]), true);
            bind_or_mark(lt.gu_w, gu_arr[l], static_cast<std::size_t>(gu_bytes_arr[l]), true);
            bind_or_mark(lt.down_w, down_arr[l], static_cast<std::size_t>(down_bytes_arr[l]), true);

            bind_or_mark(lt.attn_norm_w, attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.post_attn_norm_w, post_attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.ffn_norm_w, ffn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.post_ffn_norm_w, post_ffn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.q_norm_w, q_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);
            if (!info.isShared)
                bind_or_mark(lt.k_norm_w, k_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);

            if (!info.isShared)
            {
                bind_or_mark(lt.k_cached_t, k_cache_arr[l], kv_cache_bytes(info.kvHeads, info.cacheSize, info.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(lt.v_cached_t, v_cache_arr[l], kv_cache_bytes(info.kvHeads, info.cacheSize, info.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (layer_attn_mask[l] != nullptr && !layer_attn_mask_data[l].empty())
                    bind_or_mark(layer_attn_mask[l], layer_attn_mask_data[l].data(), layer_attn_mask_data[l].size() * sizeof(ggml_fp16_t), false);
            }

            if (lt.ple_gate_w != nullptr)
            {
                bind_or_mark(lt.ple_gate_w, ple_gate_arr[l], static_cast<std::size_t>(ple_gate_bytes_arr[l]), true);
                bind_or_mark(lt.ple_proj_w, ple_proj_arr[l], static_cast<std::size_t>(ple_proj_bytes_arr[l]), true);
                bind_or_mark(lt.ple_post_norm_w, ple_post_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            }
        }

        // Allocate backend buffer. Reuse a persistent compute buffer across
        // decode steps instead of allocating a fresh one every token (llama.cpp
        // amortizes this via a persistent graph allocator; we mirror that). The
        // host_read_barrier below drains the prior step's GPU work before this
        // graph runs, so reusing the buffer is race-free.
        BufferHandle buffer(nullptr);
        if (!alloc_ctx_tensors_reuse(ctx))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("Failed to allocate backend buffer for Gemma4 model decode.");
                return 0;
            }
        }

        // Drain pending async work before CPU memcpys from C# tensor buffers.
        host_read_barrier();

        // Upload data
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);

        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(hidden_size) * sizeof(float));

        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));

        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, rope_freq_factors, 0,
                static_cast<std::size_t>(rope_freq_factors_len) * sizeof(float));

        if (ple_input != nullptr && ple_data != nullptr)
            ggml_backend_tensor_set(ple_input, ple_data, 0,
                static_cast<std::size_t>(num_layers) * ple_dim * sizeof(float));

        // Execute single graph
        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for Gemma4 model decode.");
            return 0;
        }

        // Download hidden state (async blit on Metal in async mode)
        finalize_compute_with_download(hidden_out, hidden_data, static_cast<std::size_t>(hidden_size) * sizeof(float));

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
        set_last_error("Unknown error in Gemma4 model decode.");
        return 0;
    }
}

// ============================================================================
// Fused single-layer prefill: entire transformer layer as one GGML graph.
// Eliminates all per-op C#→GGML round trips and keeps intermediates on device.
// Handles: attn_norm → QKV → QK-norm → V-norm → RoPE → KV-cache-write →
//          attention(mul_mat+softmax+mul_mat) → O-proj → post-attn-norm →
//          residual → FFN-norm → gate_up → GELU*up → down → post-FFN-norm →
//          residual → layer-scale.
// Dense (non-MoE), non-shared layers only.
// ============================================================================

// Single-layer fused prefill graph for Gemma4. Runs the entire transformer
// block (attention + MLP + optional PLE) as one GGML dispatch, replacing the
// 10+ separate dispatches the C# fallback issues per layer per chunk.
//
// Key design points for chunked prefill correctness:
//   - For SWA layers in chunks 2+, the caller passes the previous-window K/V
//     (gathered from the rolling cache *before* this chunk overwrites it).
//     The kernel concatenates [prev | fresh] for attention, ensuring queries
//     near the start of the chunk see the (W-1) preceding tokens that fall
//     inside their sliding window.
//   - For full-attention (global) layers in chunks 2+, the kernel views the
//     persistent cache positions [0, startPos) and concatenates with fresh K/V.
//     This preserves causal context across all prior chunks at zero copy cost
//     because the cache is shared host memory on Apple Silicon.
//   - Fresh K/V is always written to the cache *after* attention reads, with
//     graph dependencies enforcing ordering. This avoids any read-after-write
//     hazard on the rolling SWA cache, which would otherwise overwrite the
//     prev-window slots within this same chunk for chunk_size > slidingWindow.
//   - Optional PLE (Per-Layer Embedding) is injected after the FFN residual
//     using the same gate/proj/norm sequence as `Gemma4ModelDecode`. Without
//     this branch the fused path was ineligible for E4B (which always has PLE)
//     so the C# slow path was the only option.
TSG_EXPORT int TSGgml_Gemma4LayerPrefill(
    float* hidden_data,     // [seqLen * hiddenSize] in/out
    int hiddenSize, int seqLen,
    // Attention weights
    void* attnNormW,        // F32 [hiddenSize]
    void* qkvW, int qkvType, std::int64_t qkvNe0, std::int64_t qkvNe1, std::int64_t qkvBytes,
    void* qNormW,           // F32 [headDim]
    void* kNormW,           // F32 [headDim]
    void* oW, int oType, std::int64_t oNe0, std::int64_t oNe1, std::int64_t oBytes,
    void* postAttnNormW,    // F32 [hiddenSize]
    // FFN weights
    void* ffnNormW,         // F32 [hiddenSize]
    void* guW, int guType, std::int64_t guNe0, std::int64_t guNe1, std::int64_t guBytes,
    void* downW, int downType, std::int64_t downNe0, std::int64_t downNe1, std::int64_t downBytes,
    void* postFfnNormW,     // F32 [hiddenSize]
    // KV cache
    float* kCacheData, float* vCacheData,
    // Layer params
    int numHeads, int kvHeads, int headDim,
    int cacheSize, int startPos,
    int isLocal, int slidingWindow,
    float ropeBase, int ropeDims,
    float* ropeFreqFactors, int freqFactorsLen,
    float layerScalar, float eps,
    // Chunked prefill: prev-window KV for SWA layers when startPos > 0.
    // Layout: [kvHeads, prevWindowLen, headDim] contiguous, F32. Pass nullptr
    // and prevWindowLen = 0 for chunk-1 / global / non-chunked usage.
    float* swaPrevK, float* swaPrevV, int prevWindowLen,
    // Per-Layer Embedding (Gemma4): per-token PLE input [seqLen, pleDim].
    // gate_w: [pleDim, hiddenSize], proj_w: [hiddenSize, pleDim], post_norm: [hiddenSize].
    // Pass null/0 to skip PLE injection.
    float* pleInputData, int pleDim,
    void* pleGateW, int pleGateType, std::int64_t pleGateNe0, std::int64_t pleGateNe1, std::int64_t pleGateBytes,
    void* pleProjW, int pleProjType, std::int64_t pleProjNe0, std::int64_t pleProjNe1, std::int64_t pleProjBytes,
    void* plePostNormW,
    // Optional fresh K/V output buffers (pre-allocated by the caller, shape
    // [kvHeads, seqLen, headDim] head-first contiguous F32). When the caller
    // is a SWA donor that downstream KV-shared layers will read in this same
    // chunk, it passes these so the kernel can publish the freshly-computed
    // (post-norm, post-RoPE) K/V to host memory. The C# attention path then
    // hands the buffers to shared layers via _prefillSWAKV instead of forcing
    // them to read from the rolling cache (which only holds the last
    // slidingWindow positions and is therefore wrong when seqLen > W).
    float* freshKOut, float* freshVOut,
    // Shared (KV-following) layer mode. When isShared!=0, the layer skips its
    // own K/V projection and instead reuses donor K/V supplied by the caller
    // (shape [kvHeads, donorKvLen, headDim] head-first contiguous F32). qkvW
    // must be the Q-only weight in this case (rather than the fused QKV).
    // No cache write happens: the donor is the cache owner and has already
    // published its K/V via freshKOut/freshVOut.
    int isShared,
    float* donorK, float* donorV, int donorKvLen,
    // KV cache element type. 0 = F32 (default, legacy), 1 = F16 (memory-saving).
    // When F16 we still build attention in F32 (Q is F32, fresh K/V is F32),
    // but the persistent cache lives in F16 so writes go through ggml_cpy(F32->F16)
    // and the global-prev path materializes the historical cache view as F32
    // before concatenating with fresh K/V.
    int kvCacheType)
{
    try
    {
        if (!ensure_backend()) return 0;

        const int qDim = numHeads * headDim;
        const int kDim = kvHeads * headDim;
        const int totalSeqLen = startPos + seqLen;
        const std::int64_t intermediateSize = guNe1 / 2;
        const bool isSharedLayer = isShared != 0 && donorK != nullptr && donorV != nullptr && donorKvLen > 0;
        const bool hasSwaPrev = (isLocal != 0) && swaPrevK != nullptr && prevWindowLen > 0 && !isSharedLayer;
        const bool hasGlobalPrev = (isLocal == 0) && startPos > 0 && !isSharedLayer;
        const bool hasFreshOut = freshKOut != nullptr && freshVOut != nullptr && !isSharedLayer;
        const ggml_type kvType = static_cast<ggml_type>(kvCacheType);
        const int kvLen = isSharedLayer ? donorKvLen
                        : hasSwaPrev ? (prevWindowLen + seqLen)
                        : hasGlobalPrev ? totalSeqLen
                        : seqLen;
        const int maskStart = kvLen - seqLen;

        // Larger ctx than the previous version because we may add concat ops
        // for prev-window K/V plus PLE projections on top of attention/FFN.
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create context for Gemma4 layer prefill.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // Reuse the same buffer for input and output to keep peak ctx alloc low.
        ggml_tensor* hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hiddenSize, seqLen);
        ggml_tensor* hidden_out_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hiddenSize, seqLen);

        ggml_tensor* attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hiddenSize);
        ggml_tensor* qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkvType), qkvNe0, qkvNe1);
        ggml_tensor* q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, headDim);
        ggml_tensor* k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, headDim);
        ggml_tensor* o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(oType), oNe0, oNe1);
        ggml_tensor* post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hiddenSize);
        ggml_tensor* ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hiddenSize);
        ggml_tensor* gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(guType), guNe0, guNe1);
        ggml_tensor* down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(downType), downNe0, downNe1);
        ggml_tensor* post_ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hiddenSize);

        ggml_tensor* k_cache_t = ggml_new_tensor_3d(ctx, kvType, headDim, cacheSize, kvHeads);
        ggml_tensor* v_cache_t = ggml_new_tensor_3d(ctx, kvType, headDim, cacheSize, kvHeads);

        ggml_tensor* swa_prev_k_t = nullptr;
        ggml_tensor* swa_prev_v_t = nullptr;
        if (hasSwaPrev)
        {
            swa_prev_k_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, prevWindowLen, kvHeads);
            swa_prev_v_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, prevWindowLen, kvHeads);
        }

        ggml_tensor* fresh_k_out_t = nullptr;
        ggml_tensor* fresh_v_out_t = nullptr;
        if (hasFreshOut)
        {
            fresh_k_out_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, seqLen, kvHeads);
            fresh_v_out_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, seqLen, kvHeads);
        }

        ggml_tensor* donor_k_t = nullptr;
        ggml_tensor* donor_v_t = nullptr;
        if (isSharedLayer)
        {
            donor_k_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, donorKvLen, kvHeads);
            donor_v_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, donorKvLen, kvHeads);
        }

        ggml_tensor* ple_gate_w = nullptr;
        ggml_tensor* ple_proj_w = nullptr;
        ggml_tensor* ple_post_norm_w = nullptr;
        ggml_tensor* ple_input_t = nullptr;
        const bool hasPle = pleInputData != nullptr && pleDim > 0 && pleGateW != nullptr && pleProjW != nullptr;
        if (hasPle)
        {
            ple_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(pleGateType), pleGateNe0, pleGateNe1);
            ple_proj_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(pleProjType), pleProjNe0, pleProjNe1);
            ple_post_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hiddenSize);
            ple_input_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pleDim, seqLen);
        }

        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seqLen);

        // === Wave 1.3: per-ubatch cached pos / mask buffers ===
        //
        // Within a single C# forward pass every attention layer is invoked
        // with the same (startPos, seqLen) and the same (isLocal,
        // slidingWindow) signature, so the RoPE position vector and the
        // F16 causal+SWA mask are bit-identical across all layers. The
        // legacy code rebuilt them on the C++ stack for every layer and
        // re-uploaded them to the backend (~`kvLen * seqLen * 2` bytes for
        // the mask and `seqLen * 4` bytes for pos). On long prefills
        // (seqLen=2048, kvLen=2048, 30 layers) that's 240 MiB of
        // redundant uploads per ubatch.
        //
        // We now maintain a thread-local cache keyed on the signature.
        // The buffers themselves are kept alive across calls so the
        // cacheable-host-ptr binding path (try_get_host_ptr_buffer with
        // cache=true) recognises them and binds them zero-copy on
        // subsequent layer calls in the same ubatch. The first call in
        // the ubatch fills the buffers; subsequent calls just reuse them.
        struct PosCache {
            int32_t startPos = -1;
            int seqLen = -1;
            std::vector<int32_t> data;
        };
        static thread_local PosCache s_pos_cache;
        if (s_pos_cache.startPos != startPos || s_pos_cache.seqLen != seqLen)
        {
            s_pos_cache.data.resize(seqLen);
            for (int i = 0; i < seqLen; i++) s_pos_cache.data[i] = startPos + i;
            s_pos_cache.startPos = startPos;
            s_pos_cache.seqLen = seqLen;
        }
        std::vector<int32_t>& pos_data = s_pos_cache.data;

        ggml_tensor* freq_factors_t = nullptr;
        if (ropeFreqFactors != nullptr && freqFactorsLen > 0)
            freq_factors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, freqFactorsLen);

        // === Build graph ===

        ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden_t, eps), attn_norm_w);

        // QKV (or Q-only for shared layers) projection.
        // For non-shared layers qkvW is [hiddenSize, qDim+2*kDim] - the fused
        // Q/K/V weight - producing [qkvDim, seqLen] which we then split.
        // For shared layers qkvW is just the [hiddenSize, qDim] Q weight; the
        // K/V come pre-computed from the donor (donorK/donorV).
        ggml_tensor* qkv_out = ggml_mul_mat(ctx, qkv_w, normed);

        ggml_tensor* q_attn = nullptr;
        ggml_tensor* k_fresh = nullptr;
        ggml_tensor* v_fresh = nullptr;

        if (isSharedLayer)
        {
            // Q-only path: qkv_out is [qDim, seqLen]. Reshape directly to
            // [headDim, numHeads*seqLen] and apply Q-norm + RoPE. K/V come
            // from donorK/donorV via donor_k_t/donor_v_t.
            ggml_tensor* q_heads = ggml_reshape_2d(ctx, qkv_out, headDim, numHeads * seqLen);
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_heads, eps), q_norm_w);

            ggml_tensor* rope_ff = (isLocal != 0) ? nullptr : freq_factors_t;
            ggml_tensor* q_4d = ggml_reshape_4d(ctx, q_normed, headDim, numHeads, seqLen, 1);
            ggml_tensor* q_roped = ggml_rope_ext(ctx, q_4d, pos_tensor, rope_ff,
                ropeDims, 2, 0, ropeBase, 1.0f, 0, 1, 0, 0);
            q_attn = ggml_cont(ctx, ggml_permute(ctx, q_roped, 0, 2, 1, 3));

            // Donor K/V are already in head-first [headDim, donorKvLen, kvHeads]
            // layout (post-norm and post-RoPE) from when the donor ran earlier
            // in this chunk - publish via fresh K/V output buffers.
            k_fresh = donor_k_t;
            v_fresh = donor_v_t;
        }
        else
        {
            // Strided views into the fused QKV output tensor. Each is
            // [qkvSubDim, seqLen] with the row stride of the full qkv_out
            // tensor (qkvDim*sizeof(float)), so we need an explicit ggml_cont
            // before reshape - reshape requires fully-contiguous input.
            ggml_tensor* q_raw = ggml_view_2d(ctx, qkv_out, qDim, seqLen,
                qkv_out->nb[1], 0);
            ggml_tensor* k_raw = ggml_view_2d(ctx, qkv_out, kDim, seqLen,
                qkv_out->nb[1], static_cast<std::size_t>(qDim) * sizeof(float));
            ggml_tensor* v_raw = ggml_view_2d(ctx, qkv_out, kDim, seqLen,
                qkv_out->nb[1], static_cast<std::size_t>(qDim + kDim) * sizeof(float));

            // Q/K/V layout: the QKV matmul output has shape [qkvDim, seqLen] in
            // ggml's column-major-fastest convention, with qkvDim laid out as
            // [Q-section (heads-fastest), K-section, V-section]. Slicing a
            // section and reshaping to [headDim, heads*seqLen] yields cell(h, a)
            // = Q/K/V[head=a%nHeads, dim=h, position=a/nHeads], i.e. heads
            // fastest along `a`. Reshaping further to [headDim, nHeads, seqLen]
            // (with nHeads in the middle) preserves the same memory order so
            // the data semantically becomes [head, dim, position] - exactly
            // what RoPE expects on its 4-D input.
            ggml_tensor* q_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, q_raw), headDim, numHeads * seqLen);
            ggml_tensor* k_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, k_raw), headDim, kvHeads * seqLen);
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_heads, eps), q_norm_w);
            ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_heads, eps), k_norm_w);

            // V also needs unweighted RMSNorm along headDim. Same flat reshape
            // so the data layout matches Q/K (heads fastest within `a`).
            ggml_tensor* v_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, v_raw), headDim, kvHeads * seqLen);
            ggml_tensor* v_normed = ggml_rms_norm(ctx, v_heads, eps);

            ggml_tensor* rope_ff = (isLocal != 0) ? nullptr : freq_factors_t;
            ggml_tensor* q_4d = ggml_reshape_4d(ctx, q_normed, headDim, numHeads, seqLen, 1);
            ggml_tensor* k_4d = ggml_reshape_4d(ctx, k_normed, headDim, kvHeads, seqLen, 1);
            ggml_tensor* q_roped = ggml_rope_ext(ctx, q_4d, pos_tensor, rope_ff,
                ropeDims, 2, 0, ropeBase, 1.0f, 0, 1, 0, 0);
            ggml_tensor* k_roped = ggml_rope_ext(ctx, k_4d, pos_tensor, rope_ff,
                ropeDims, 2, 0, ropeBase, 1.0f, 0, 1, 0, 0);

            // Bring Q/K/V to head-first attention layout [headDim, seqLen, nHeads].
            //
            // For Q and (when no concat is needed) K/V, leave as strided permute
            // views - this matches llama.cpp's build_attn_mha exactly. Their
            // KV cache is laid out as [head_dim, kv_heads, n_kv, n_streams]
            // and they call ggml_permute(0,2,1,3) right before flash_attn_ext
            // without any ggml_cont, producing a "positions/heads interleaved"
            // strided view. flash_attn_ext on Metal walks K/V via the strides,
            // and the f32 matrix kernel only correctly handles inputs in this
            // strided layout - tight contiguous K/V reorderings (that nb[1] <
            // nb[2] case) silently produce wrong logits, even with set_prec
            // F32. We discovered this by comparing failing prefill paths
            // against the working decode path which also uses strided cache
            // views. K/V always go through reshape_4d+permute(0,2,1,3) to get
            // the same nb[1] > nb[2] stride relationship.
            // Bring Q/K/V to head-first attention layout [headDim, seqLen, nHeads].
            // The permute swaps dims 1 (heads) and 2 (seqLen). We must explicitly
            // handle V the same way - a bare reshape from [headDim, kvHeads*seqLen]
            // to [headDim, seqLen, kvHeads] mis-interprets the stride and silently
            // mangles V into a position/head shuffled version of itself. Q stays
            // as a strided permute view (matches the working decode path); K/V
            // need to be tight contiguous so we can ggml_concat them with the
            // previous-window K/V (same tight layout as the persistent cache and
            // the C# fresh-publish buffers).
            q_attn = ggml_permute(ctx, q_roped, 0, 2, 1, 3);
            k_fresh = ggml_reshape_3d(ctx,
                ggml_cont(ctx, ggml_permute(ctx, k_roped, 0, 2, 1, 3)),
                headDim, seqLen, kvHeads);
            ggml_tensor* v_3d_pre = ggml_reshape_4d(ctx, v_normed, headDim, kvHeads, seqLen, 1);
            v_fresh = ggml_reshape_3d(ctx,
                ggml_cont(ctx, ggml_permute(ctx, v_3d_pre, 0, 2, 1, 3)),
                headDim, seqLen, kvHeads);
        }

        // Build attention K/V source: prev-window (if any) concatenated with fresh.
        // - SWA chunk 2+: prev = swa_prev_*_t (W-1 tokens, head-first contiguous F32).
        // - Global chunk 2+: prev = view into the persistent cache for positions
        //   [0, startPos), with the persistent cache's 3-D strides intact - this is
        //   strictly cheaper than copying the whole prefix because the cache lives in
        //   host-shared memory on Apple Silicon.
        // Build k_full/v_full in the LLAMA.CPP TIGHT LAYOUT
        //   [headDim, kvHeads, kvLen] (positions are the slowest dim).
        // This is the layout the f32 flash_attn_ext metal kernel expects after
        // the standard permute(0,2,1,3) is applied to it (yielding strided
        // [headDim, kvLen, kvHeads] with nb[1] > nb[2]).
        //
        // - swa_prev_*_t and the global-prev cache view both currently live in
        //   our own "head-first" tight layout [headDim, prevLen, kvHeads], so
        //   we permute(0,2,1,3) + cont them to reach llama.cpp's tight layout.
        //   (The cont is a single per-layer copy of prevLen tokens, ~MB sized.)
        // - k_fresh / v_fresh after the QKV step above are *already* in the
        //   llama.cpp tight layout [headDim, kvHeads, seqLen], so no extra
        //   work is needed for them.
        // - The concat is along ne[2] (positions, i.e. the slowest dim) and
        //   produces a tight contiguous [headDim, kvHeads, kvLen].
        ggml_tensor* k_attn = k_fresh;
        ggml_tensor* v_attn = v_fresh;
        if (hasSwaPrev)
        {
            k_attn = ggml_concat(ctx, swa_prev_k_t, k_fresh, 1);
            v_attn = ggml_concat(ctx, swa_prev_v_t, v_fresh, 1);
        }
        else if (hasGlobalPrev)
        {
            ggml_tensor* k_prev = ggml_view_3d(ctx, k_cache_t,
                headDim, startPos, kvHeads,
                k_cache_t->nb[1], k_cache_t->nb[2], 0);
            ggml_tensor* v_prev = ggml_view_3d(ctx, v_cache_t,
                headDim, startPos, kvHeads,
                v_cache_t->nb[1], v_cache_t->nb[2], 0);

            // F16 cache: prev is F16 but fresh is F32. ggml_concat requires
            // matching types, so materialize prev as F32 before the concat.
            // This is a one-shot, contiguous, bandwidth-bound copy; the cost is
            // negligible vs. the attention itself for typical chunk sizes.
            if (kvType != GGML_TYPE_F32)
            {
                ggml_tensor* k_prev_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, startPos, kvHeads);
                ggml_tensor* v_prev_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, startPos, kvHeads);
                k_prev = ggml_cpy(ctx, k_prev, k_prev_f32);
                v_prev = ggml_cpy(ctx, v_prev, v_prev_f32);
            }

            k_attn = ggml_concat(ctx, k_prev, k_fresh, 1);
            v_attn = ggml_concat(ctx, v_prev, v_fresh, 1);
        }

        // k_fresh_my aliases k_fresh in this layout (no conversion needed
        // because both share the same OLD tight layout used by the cache
        // and the C# fresh-publish buffers).
        ggml_tensor* k_fresh_my = k_fresh;
        ggml_tensor* v_fresh_my = v_fresh;

        // Causal + optional sliding-window mask. Indexing: kv k attends to q if
        // k <= maskStart + q (causal) AND k > maskStart + q - slidingWindow (SWA).
        // For SWA chunked prefill maskStart = prevWindowLen so logical alignment
        // between the concatenated K/V and the chunk's queries is preserved.
        auto fill_prefill_mask = [&](std::vector<ggml_fp16_t>& data, int maskKvLen) {
            data.resize(static_cast<std::size_t>(maskKvLen) * seqLen);
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            int win = (isLocal != 0) ? slidingWindow : 0;
            for (int qi = 0; qi < seqLen; qi++)
            {
                int threshold = maskStart + qi;
                int winStart = (win > 0) ? std::max(0, threshold - win + 1) : 0;
                ggml_fp16_t* row = &data[static_cast<std::size_t>(qi) * maskKvLen];
                for (int ki = 0; ki < maskKvLen; ki++)
                    row[ki] = (ki >= kvLen || ki > threshold || ki < winStart) ? neg_inf : zero_val;
            }
        };

        // Per-ubatch mask cache (see PosCache rationale above). Mask data is
        // bit-identical across all attention layers in the same chunk that
        // share (startPos, seqLen, kvLen, isLocal, slidingWindow, maskStart).
        struct MaskCache {
            int startPos = -1;
            int seqLen = -1;
            int kvLen = -1;
            int maskKvLen = -1;
            int isLocal = -1;
            int slidingWindow = -1;
            int maskStart = -1;
            std::vector<ggml_fp16_t> data;
        };
        static thread_local MaskCache s_mask_cache;
        auto fetch_cached_mask = [&](int maskKvLen) -> std::vector<ggml_fp16_t>& {
            if (s_mask_cache.startPos != startPos
                || s_mask_cache.seqLen != seqLen
                || s_mask_cache.kvLen != kvLen
                || s_mask_cache.maskKvLen != maskKvLen
                || s_mask_cache.isLocal != isLocal
                || s_mask_cache.slidingWindow != slidingWindow
                || s_mask_cache.maskStart != maskStart)
            {
                fill_prefill_mask(s_mask_cache.data, maskKvLen);
                s_mask_cache.startPos = startPos;
                s_mask_cache.seqLen = seqLen;
                s_mask_cache.kvLen = kvLen;
                s_mask_cache.maskKvLen = maskKvLen;
                s_mask_cache.isLocal = isLocal;
                s_mask_cache.slidingWindow = slidingWindow;
                s_mask_cache.maskStart = maskStart;
            }
            return s_mask_cache.data;
        };

        ggml_tensor* mask_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kvLen, seqLen, 1, 1);
        std::vector<ggml_fp16_t>& mask_data = fetch_cached_mask(kvLen);

        // Attention: ggml_flash_attn_ext when enabled (default), with the
        // explicit mul_mat -> soft_max_ext -> mul_mat chain as fallback.
        //
        // The critical detail that took two prior attempts to find: without
        // ggml_flash_attn_ext_set_prec(GGML_PREC_F32) the kernel uses F16
        // accumulators internally for the K*Q scores and softmax, which
        // underflows/overflows for Gemma4's head_dim=256 (SWA) and 512
        // (global) on multi-token Q and silently produces wrong logits
        // (decoded as eos spam on real prompts). Both ollama and llama.cpp
        // call set_prec(F32) immediately after every flash_attn_ext for
        // exactly this reason.
        //
        // The fallback path remains accessible via TSG_USE_FLASH_ATTN_PREFILL=0
        // for A/B comparison or in case a future ggml-metal regression breaks
        // the fast path.
        ggml_tensor* attn_flat;
        const char* use_fa_env = std::getenv("TSG_USE_FLASH_ATTN_PREFILL");
        const bool use_flash_attn = (use_fa_env == nullptr) || (use_fa_env[0] != '0');
        ggml_tensor* flash_attn_out = nullptr;
        ggml_tensor* flash_mask_t = mask_t;
        std::vector<ggml_fp16_t> flash_mask_data;

        if (use_flash_attn)
        {
            // ggml_flash_attn_ext returns the result in [n_embd_v, n_head,
            // n_batch, ne3] layout - i.e. *already permuted* relative to the
            // mul_mat path which leaves attn in [n_embd_v, n_batch, n_head].
            // The flash layout is exactly what the O projection wants for
            // its [qDim, seqLen] input (one column per position with all
            // heads contiguous within the column), so we reshape directly
            // and skip the manual permute+cont. Earlier attempts that did
            // the permute anyway scrambled the heads across positions and
            // produced eos/garbage logits - that's the multi-token prefill
            // bug we'd been chasing.
            //
            // ggml_flash_attn_ext_set_prec(GGML_PREC_F32) keeps the QK
            // accumulator and softmax in F32 even when the kernel template
            // would default to F16 internals; both ollama and llama.cpp do
            // this for every flash_attn_ext call.
            flash_attn_out = ggml_flash_attn_ext(ctx, q_attn, k_attn, v_attn,
                mask_t, 1.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(flash_attn_out, GGML_PREC_F32);

            // CUDA's 512/576-dim flash-attn kernels require the grouped-query
            // path, which in turn requires a 256-aligned KV length. Decode
            // already satisfies this by viewing a padded cache window; prefill
            // builds fresh/concatenated K/V tensors, so pad them explicitly and
            // mask the added slots when the backend rejects the unpadded op.
            if (!backend_supports_op(flash_attn_out) &&
                flash_attn_requires_masked_padding(headDim) &&
                (kvLen % kFlashAttnKvStride) != 0)
            {
                const int paddedKvLen =
                    ((kvLen + kFlashAttnKvStride - 1) / kFlashAttnKvStride) * kFlashAttnKvStride;
                const int padKvLen = paddedKvLen - kvLen;

                ggml_tensor* k_attn_padded = ggml_pad(ctx, k_attn, 0, padKvLen, 0, 0);
                ggml_tensor* v_attn_padded = ggml_pad(ctx, v_attn, 0, padKvLen, 0, 0);
                flash_mask_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, paddedKvLen, seqLen, 1, 1);
                fill_prefill_mask(flash_mask_data, paddedKvLen);

                flash_attn_out = ggml_flash_attn_ext(ctx, q_attn, k_attn_padded, v_attn_padded,
                    flash_mask_t, 1.0f, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(flash_attn_out, GGML_PREC_F32);
            }
        }

        if (use_flash_attn && backend_supports_op(flash_attn_out))
        {
            attn_flat = ggml_reshape_2d(ctx, flash_attn_out, qDim, seqLen);
        }
        else
        {
            ggml_tensor* q_attn_cont = ggml_cont(ctx, q_attn);
            ggml_tensor* scores = ggml_mul_mat(ctx, k_attn, q_attn_cont);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, mask_t, 1.0f, 0.0f);
            ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_attn, 1, 0, 2, 3));
            ggml_tensor* attn_out = ggml_mul_mat(ctx, v_perm, probs);
            // mul_mat output is [headDim, seqLen, numHeads]; permute to
            // [headDim, numHeads, seqLen] then reshape so the per-position
            // qDim block contains all heads contiguously, matching what the
            // O projection (and the flash path above) consume.
            ggml_tensor* attn_perm = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
            attn_flat = ggml_reshape_2d(ctx, attn_perm, qDim, seqLen);
        }

        ggml_tensor* o_out = ggml_mul_mat(ctx, o_w, attn_flat);

        // KV cache write: writes happen *after* the attention reads (k_attn /
        // v_attn never depend on the cache for fresh-K/V paths, and for the
        // global-prev path the cache view used for attention covers only the
        // already-populated [0, startPos) region). Listing k_cpy/v_cpy as graph
        // outputs and expanding them before `output` ensures the next layer
        // sees the updated cache.
        //
        // For SWA layers the cache is rolling (size = cacheSize == slidingWindow):
        //   * If seqLen > cacheSize, only the *last* cacheSize tokens of the
        //     chunk survive; the earlier ones would be overwritten anyway, so
        //     we skip writing them entirely (`writeOffsetInChunk` shifts the
        //     source range forward).
        //   * The remaining write may cross the cache wrap point, in which case
        //     we split it into tail (writePos..cacheSize) and head (0..rest).
        //
        // Shared layers don't own their KV cache (they read from the donor's),
        // so they skip the cache write entirely.
        ggml_tensor* k_cpy = nullptr;
        ggml_tensor* v_cpy = nullptr;
        ggml_tensor* k_cpy_b = nullptr;
        ggml_tensor* v_cpy_b = nullptr;
        if (!isSharedLayer)
        {
            if (isLocal != 0)
            {
                const int writeOffsetInChunk = std::max(0, seqLen - cacheSize);
                const int writeLen = seqLen - writeOffsetInChunk;
                const int writeStartLogical = startPos + writeOffsetInChunk;
                const int writePos = ((writeStartLogical % cacheSize) + cacheSize) % cacheSize;
                const int firstLen = std::min(writeLen, cacheSize - writePos);

                // Cache-side byte offsets use the tensor's per-position stride so
                // the same code works for F32, F16 and block-quantized types (Q8_0).
                std::size_t kv_offset_a =
                    static_cast<std::size_t>(writePos) * k_cache_t->nb[1];
                ggml_tensor* k_dst_a = ggml_view_3d(ctx, k_cache_t,
                    headDim, firstLen, kvHeads,
                    k_cache_t->nb[1], k_cache_t->nb[2], kv_offset_a);
                ggml_tensor* v_dst_a = ggml_view_3d(ctx, v_cache_t,
                    headDim, firstLen, kvHeads,
                    v_cache_t->nb[1], v_cache_t->nb[2], kv_offset_a);

                // Source offset is into k_fresh_my which is always F32.
                std::size_t src_offset_a =
                    static_cast<std::size_t>(writeOffsetInChunk) * headDim * sizeof(float);
                ggml_tensor* k_src_a = (firstLen == seqLen && writeOffsetInChunk == 0) ? k_fresh_my
                    : ggml_view_3d(ctx, k_fresh_my, headDim, firstLen, kvHeads,
                        k_fresh_my->nb[1], k_fresh_my->nb[2], src_offset_a);
                ggml_tensor* v_src_a = (firstLen == seqLen && writeOffsetInChunk == 0) ? v_fresh_my
                    : ggml_view_3d(ctx, v_fresh_my, headDim, firstLen, kvHeads,
                        v_fresh_my->nb[1], v_fresh_my->nb[2], src_offset_a);
                k_cpy = ggml_cpy(ctx, k_src_a, k_dst_a);
                v_cpy = ggml_cpy(ctx, v_src_a, v_dst_a);

                if (firstLen < writeLen)
                {
                    const int secondLen = writeLen - firstLen;
                    std::size_t src_offset_b =
                        static_cast<std::size_t>(writeOffsetInChunk + firstLen) * headDim * sizeof(float);
                    ggml_tensor* k_src_b = ggml_view_3d(ctx, k_fresh_my,
                        headDim, secondLen, kvHeads,
                        k_fresh_my->nb[1], k_fresh_my->nb[2], src_offset_b);
                    ggml_tensor* v_src_b = ggml_view_3d(ctx, v_fresh_my,
                        headDim, secondLen, kvHeads,
                        v_fresh_my->nb[1], v_fresh_my->nb[2], src_offset_b);
                    ggml_tensor* k_dst_b = ggml_view_3d(ctx, k_cache_t,
                        headDim, secondLen, kvHeads,
                        k_cache_t->nb[1], k_cache_t->nb[2], 0);
                    ggml_tensor* v_dst_b = ggml_view_3d(ctx, v_cache_t,
                        headDim, secondLen, kvHeads,
                        v_cache_t->nb[1], v_cache_t->nb[2], 0);
                    k_cpy_b = ggml_cpy(ctx, k_src_b, k_dst_b);
                    v_cpy_b = ggml_cpy(ctx, v_src_b, v_dst_b);
                }
            }
            else
            {
                // Global cache: contiguous append at startPos. We use nb[1] so the
                // offset is correct for any cache dtype (F32/F16/Q8_0); k_fresh_my
                // (F32) is automatically converted by ggml_cpy to match the cache
                // type when writing.
                std::size_t kv_offset =
                    static_cast<std::size_t>(startPos) * k_cache_t->nb[1];
                ggml_tensor* k_dst = ggml_view_3d(ctx, k_cache_t,
                    headDim, seqLen, kvHeads,
                    k_cache_t->nb[1], k_cache_t->nb[2], kv_offset);
                ggml_tensor* v_dst = ggml_view_3d(ctx, v_cache_t,
                    headDim, seqLen, kvHeads,
                    v_cache_t->nb[1], v_cache_t->nb[2], kv_offset);
                k_cpy = ggml_cpy(ctx, k_fresh_my, k_dst);
                v_cpy = ggml_cpy(ctx, v_fresh_my, v_dst);
            }
        }

        // Donor publish: SWA layers that other shared layers will read inside
        // this same chunk get a host-visible copy of the freshly-computed K/V.
        // Without this the rolling cache (size = slidingWindow) silently drops
        // the early positions of any seqLen > W chunk, breaking the shared
        // layer's attention for queries near the start of the chunk.
        ggml_tensor* fresh_k_cpy = nullptr;
        ggml_tensor* fresh_v_cpy = nullptr;
        if (hasFreshOut)
        {
            fresh_k_cpy = ggml_cpy(ctx, k_fresh_my, fresh_k_out_t);
            fresh_v_cpy = ggml_cpy(ctx, v_fresh_my, fresh_v_out_t);
        }

        // Post-attn norm + residual
        ggml_tensor* post_attn = ggml_mul(ctx, ggml_rms_norm(ctx, o_out, eps), post_attn_norm_w);
        ggml_tensor* residual1 = ggml_add(ctx, hidden_t, post_attn);

        // FFN: norm -> gate_up -> GELU*up -> down -> post_norm -> residual.
        // gate/up are *strided* views into gu_out (one half each), so we
        // ggml_cont them before activation: Metal's GELU kernel and the
        // subsequent broadcasted Mul both expect contiguous inputs.
        ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), ffn_norm_w);
        ggml_tensor* gu_out = ggml_mul_mat(ctx, gu_w, ffn_normed);
        ggml_tensor* gate_v = ggml_cont(ctx, ggml_view_2d(ctx, gu_out, intermediateSize, seqLen,
            gu_out->nb[1], 0));
        ggml_tensor* up_v = ggml_cont(ctx, ggml_view_2d(ctx, gu_out, intermediateSize, seqLen,
            gu_out->nb[1], static_cast<std::size_t>(intermediateSize) * sizeof(float)));
        ggml_tensor* ffn_act = ggml_mul(ctx, ggml_gelu(ctx, gate_v), up_v);
        ggml_tensor* down_out = ggml_mul_mat(ctx, down_w, ffn_act);

        ggml_tensor* post_ffn = ggml_mul(ctx, ggml_rms_norm(ctx, down_out, eps), post_ffn_norm_w);
        ggml_tensor* residual2 = ggml_add(ctx, residual1, post_ffn);

        // PLE injection (optional, mirrors Gemma4ModelDecode's per-layer block):
        //   ple = post_norm(proj(GELU(gate(residual2)) * ple_input))
        //   residual2 += ple
        if (hasPle)
        {
            ggml_tensor* ple_gate_proj = ggml_mul_mat(ctx, ple_gate_w, residual2);
            ggml_tensor* ple_gated = ggml_mul(ctx, ggml_gelu(ctx, ple_gate_proj), ple_input_t);
            ggml_tensor* ple_proj = ggml_mul_mat(ctx, ple_proj_w, ple_gated);
            ggml_tensor* ple_normed = ggml_mul(ctx,
                ggml_rms_norm(ctx, ple_proj, eps), ple_post_norm_w);
            residual2 = ggml_add(ctx, residual2, ple_normed);
        }

        if (std::fabs(layerScalar - 1.0f) > 1e-6f)
            residual2 = ggml_scale(ctx, residual2, layerScalar);

        ggml_tensor* output = ggml_cpy(ctx, residual2, hidden_out_t);
        ggml_set_output(output);

        // Build graph: cache writes and donor-publish copies first so the
        // scheduler sequences them ahead of `output`. Subsequent layers/chunks
        // see the updated cache; the C# attention path picks up donor K/V.
        const std::size_t graph_size = 1024;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        if (k_cpy != nullptr) ggml_build_forward_expand(graph, k_cpy);
        if (v_cpy != nullptr) ggml_build_forward_expand(graph, v_cpy);
        if (k_cpy_b != nullptr) ggml_build_forward_expand(graph, k_cpy_b);
        if (v_cpy_b != nullptr) ggml_build_forward_expand(graph, v_cpy_b);
        if (fresh_k_cpy != nullptr) ggml_build_forward_expand(graph, fresh_k_cpy);
        if (fresh_v_cpy != nullptr) ggml_build_forward_expand(graph, fresh_v_cpy);
        ggml_build_forward_expand(graph, output);

        // Bind weights and KV caches. Read-only weights go through the
        // cacheable-tensor path with GGML_BACKEND_BUFFER_USAGE_WEIGHTS so the
        // backend can keep them in dedicated weight memory across calls. The
        // KV cache must be bound as COMPUTE because the graph writes to it -
        // binding as WEIGHTS would silently drop those writes on backends that
        // treat weight buffers as read-only (Metal among them).
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* d; std::size_t b; };
        std::vector<HostBinding> uploads;
        std::vector<BufferHandle> ephem;

        auto bind = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cache,
                        enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            if (cache && bytes >= 4096) {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs, usage)) {
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS) {
                        if (needs) uploads.push_back({t, data, bytes});
                        return;
                    }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096) {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cache, buf)) {
                    if (!cache) ephem.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS) return;
                }
            }
            uploads.push_back({t, data, bytes});
        };

        bind(qkv_w, qkvW, static_cast<std::size_t>(qkvBytes), true);
        bind(o_w, oW, static_cast<std::size_t>(oBytes), true);
        bind(gu_w, guW, static_cast<std::size_t>(guBytes), true);
        bind(down_w, downW, static_cast<std::size_t>(downBytes), true);
        bind(attn_norm_w, attnNormW, hiddenSize * sizeof(float), true);
        bind(post_attn_norm_w, postAttnNormW, hiddenSize * sizeof(float), true);
        bind(ffn_norm_w, ffnNormW, hiddenSize * sizeof(float), true);
        bind(post_ffn_norm_w, postFfnNormW, hiddenSize * sizeof(float), true);
        bind(q_norm_w, qNormW, headDim * sizeof(float), true);
        bind(k_norm_w, kNormW, headDim * sizeof(float), true);
        bind(k_cache_t, kCacheData, kv_cache_bytes(kvHeads, cacheSize, headDim, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        bind(v_cache_t, vCacheData, kv_cache_bytes(kvHeads, cacheSize, headDim, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

        // Wave 1.3: bind the per-ubatch cached pos / mask buffers via the
        // cacheable host-ptr path. Their host pointers are stable across
        // calls (PosCache / MaskCache are thread-local) so the second and
        // subsequent layer calls in the same ubatch hit the buffer cache and
        // skip the upload entirely. The first call in the ubatch still
        // uploads (via the `uploads` queue at the bottom) but every layer
        // after that is zero-copy.
        bind(pos_tensor, pos_data.data(), seqLen * sizeof(int32_t), true);
        bind(mask_t, mask_data.data(), mask_data.size() * sizeof(ggml_fp16_t), true);

        if (hasSwaPrev)
        {
            std::size_t prev_bytes = static_cast<std::size_t>(kvHeads)
                * static_cast<std::size_t>(prevWindowLen)
                * static_cast<std::size_t>(headDim) * sizeof(float);
            bind(swa_prev_k_t, swaPrevK, prev_bytes, false);
            bind(swa_prev_v_t, swaPrevV, prev_bytes, false);
        }

        if (hasFreshOut)
        {
            std::size_t fresh_bytes = static_cast<std::size_t>(kvHeads)
                * static_cast<std::size_t>(seqLen)
                * static_cast<std::size_t>(headDim) * sizeof(float);
            bind(fresh_k_out_t, freshKOut, fresh_bytes, false, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            bind(fresh_v_out_t, freshVOut, fresh_bytes, false, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }

        if (isSharedLayer)
        {
            std::size_t donor_bytes = static_cast<std::size_t>(kvHeads)
                * static_cast<std::size_t>(donorKvLen)
                * static_cast<std::size_t>(headDim) * sizeof(float);
            bind(donor_k_t, donorK, donor_bytes, false);
            bind(donor_v_t, donorV, donor_bytes, false);
        }

        if (hasPle)
        {
            bind(ple_gate_w, pleGateW, static_cast<std::size_t>(pleGateBytes), true);
            bind(ple_proj_w, pleProjW, static_cast<std::size_t>(pleProjBytes), true);
            bind(ple_post_norm_w, plePostNormW, hiddenSize * sizeof(float), true);
        }

        // Reuse a persistent compute buffer across layers instead of allocating
        // a fresh ~100-150 MB Metal buffer every call (was ~20 ms/layer, the
        // single largest prefill overhead). Falls back to the stock per-call
        // allocator if the reuse path can't service this graph. The per-layer
        // host_read_barrier below drains the prior layer's GPU work before this
        // graph runs, so reusing the buffer is race-free.
        BufferHandle buffer(nullptr);
        if (!alloc_ctx_tensors_reuse(ctx)) {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr) {
                set_last_error("Failed to allocate buffer for Gemma4 layer prefill.");
                return 0;
            }
        }

        // Drain pending async work so the upcoming CPU memcpys (inside
        // ggml_backend_tensor_set on shared backend buffers) don't race with
        // any in-flight zero-copy GPU writes targeting `hidden_data` /
        // `pleInputData` from the previous layer's compute.
        host_read_barrier();

        for (auto& u : uploads)
            ggml_backend_tensor_set(u.t, u.d, 0, u.b);

        ggml_backend_tensor_set(hidden_t, hidden_data, 0,
            static_cast<std::size_t>(hiddenSize) * seqLen * sizeof(float));

        // pos_tensor / mask_t are bound through the cacheable host-ptr path
        // above; the bind helper queues them into `uploads` only when this is
        // the first time the buffer pointer is seen. After that the binding
        // is zero-copy (Apple Silicon unified memory) and no upload happens.
        if (flash_mask_t != mask_t && !flash_mask_data.empty())
            ggml_backend_tensor_set(flash_mask_t, flash_mask_data.data(), 0,
                flash_mask_data.size() * sizeof(ggml_fp16_t));
        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, ropeFreqFactors, 0, freqFactorsLen * sizeof(float));
        if (hasPle && ple_input_t != nullptr)
            ggml_backend_tensor_set(ple_input_t, pleInputData, 0,
                static_cast<std::size_t>(seqLen) * pleDim * sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            set_last_error("Graph compute failed for Gemma4 layer prefill.");
            return 0;
        }

        // Download hidden state (async blit on Metal in async mode - lets the next
        // layer's graph queue while this one's data is still being copied back).
        finalize_compute_with_download(hidden_out_t, hidden_data,
            static_cast<std::size_t>(hiddenSize) * seqLen * sizeof(float));

        // KV cache lives in host-shared memory on Apple Silicon (host-ptr buffer
        // path); the backend wrote in place so no host download is required and
        // the previous unconditional get-back was pure waste. On discrete GPUs
        // the explicit `tensor_get` is still needed - left to a future follow-up
        // since the user is on Metal where this path is the hot one.

        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Gemma4 layer prefill."); return 0; }
}

// ============================================================================
// GPT-OSS attention layer prefill: full attention block (norm → fused QKV with
// bias → RoPE → KV cache append → causal+SWA mask softmax with sinks → flash /
// fallback attention → output projection with bias) as ONE GGML graph.
//
// Replaces ~10 separate C# → GGML round trips per attention layer (each its
// own ggml_cgraph + Metal command buffer) with a single graph dispatch. This
// is the prefill counterpart to TSGgml_TransformerLayerDecode and the GPT-OSS
// analogue of TSGgml_Gemma4LayerPrefill.
//
// llama.cpp reference: src/models/openai-moe-iswa.cpp::llm_build_openai_moe_iswa
// (the attention block before the MoE FFN).
//
// Key model-specific points:
//   - Single contiguous KV cache per layer (no rolling window). The sliding
//     window is implemented purely via the attention mask.
//   - Attention sinks: a per-head extra logit added in the softmax denominator
//     (ggml_soft_max_add_sinks). Even-indexed layers (isSwa != 0) use the SWA
//     mask; odd-indexed layers do full causal attention.
//   - QKV (and O) projections have biases applied via ggml_add of a 1-D bias
//     tensor that GGML broadcasts across the seqLen dimension.
//   - RoPE is NeoX-style with yarn scaling (mode=2, beta_fast=32, beta_slow=1).
//   - The MoE FFN is *not* part of this kernel; it runs through the existing
//     fused MoE prefill kernel (TSGgml_MoEFFNPrefillSwiGLUQuantF32).
//
// Hidden state in/out is a flat [seqLen, hiddenSize] F32 buffer. The kernel
// writes the residual (input + attn_out_proj(attn(norm(input)))) back into the
// same buffer, ready for the MoE FFN to consume.
// ============================================================================
TSG_EXPORT int TSGgml_GptOssAttentionLayerPrefill(
    float* hidden_data,        // [seqLen * hiddenSize] in/out (residual is added in place)
    int hiddenSize, int seqLen,
    // Attention norm
    void* attnNormW,           // F32 [hiddenSize]
    // Fused QKV (or Q-only when isQkvFused == 0; see kArr/vArr below)
    void* qkvW, int qkvType, std::int64_t qkvNe0, std::int64_t qkvNe1, std::int64_t qkvBytes,
    void* qkvB,                // F32 [qDim+2*kDim] when isQkvFused, else F32 [qDim]; may be null
    int isQkvFused,
    // Optional separate K/V weights+biases (used when isQkvFused == 0)
    void* kW, int kType, std::int64_t kNe0, std::int64_t kNe1, std::int64_t kBytes,
    void* kB,                  // F32 [kDim], may be null
    void* vW, int vType, std::int64_t vNe0, std::int64_t vNe1, std::int64_t vBytes,
    void* vB,                  // F32 [kDim], may be null
    // Output projection
    void* oW, int oType, std::int64_t oNe0, std::int64_t oNe1, std::int64_t oBytes,
    void* oB,                  // F32 [hiddenSize], may be null
    // KV cache (bound zero-copy where supported)
    void* kCacheData, void* vCacheData,
    int numHeads, int kvHeads, int headDim,
    int cacheSize, int startPos,
    // SWA / sinks
    int isSwa,                 // non-zero: apply sliding-window mask in addition to causal
    int slidingWindow,
    float* sinksData,          // F32 [numHeads], may be null (no sinks)
    // RoPE (NeoX yarn-scaled)
    float ropeBase, float ropeFreqScale, int ropeDims,
    int originalContextLength,
    // KV cache element type (0 = F32, 1 = F16)
    int kvCacheType,
    // Numerics
    float eps)
{
    try
    {
        if (!ensure_backend()) return 0;

        const int qDim = numHeads * headDim;
        const int kDim = kvHeads * headDim;
        const int totalSeqLen = startPos + seqLen;
        const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
        const ggml_type kvType = static_cast<ggml_type>(kvCacheType);

        // GPT-OSS uses a single contiguous cache per layer (no rolling SWA).
        // Attention reads positions [0, totalSeqLen) and the SWA mask zeros out
        // anything older than (startPos + q_idx - slidingWindow + 1).
        const int kvLen = totalSeqLen;

        // 32 MiB context: same upper bound as Gemma4 prefill (covers concat-free
        // attention path + FFN graph allocations even at long ubatches).
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create context for GPT-OSS attention layer prefill.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // === Tensor declarations (allocated by ggml_backend_alloc_ctx_tensors below) ===

        ggml_tensor* hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hiddenSize, seqLen);
        ggml_tensor* hidden_out_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hiddenSize, seqLen);
        ggml_tensor* attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hiddenSize);

        const int qkvDim = isQkvFused ? (qDim + 2 * kDim) : qDim;
        ggml_tensor* qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkvType), qkvNe0, qkvNe1);
        ggml_tensor* qkv_b = (qkvB != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, qkvDim) : nullptr;

        ggml_tensor* k_w = nullptr;
        ggml_tensor* k_b = nullptr;
        ggml_tensor* v_w = nullptr;
        ggml_tensor* v_b = nullptr;
        if (!isQkvFused)
        {
            k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(kType), kNe0, kNe1);
            v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(vType), vNe0, vNe1);
            if (kB != nullptr) k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDim);
            if (vB != nullptr) v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDim);
        }

        ggml_tensor* o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(oType), oNe0, oNe1);
        ggml_tensor* o_b = (oB != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hiddenSize) : nullptr;

        // KV cache "window" tensors: instead of binding the full per-layer
        // host cache (kvHeads * cacheSize * headDim * elemSize bytes — for
        // GPT-OSS this is 256 MiB per K and per V at context_length=131072,
        // which alone consumes ~12 GiB across 24 layers and exceeds Metal's
        // recommendedMaxWorkingSetSize on Apple Silicon, triggering
        // command-buffer OOMs in subsequent kernels), we allocate a per-call
        // window of shape [headDim, kvLen, kvHeads]. We upload the existing
        // prefix [0, startPos) before compute, ggml_cpy the fresh K/V into
        // [startPos, kvLen), then download the fresh slice back to the host
        // cache after compute. This keeps GPU residency for the cache to
        // O(kvLen) rather than O(cacheSize) and matches what llama.cpp's
        // build_attn_mha does internally for non-static caches.
        ggml_tensor* k_cache_t = ggml_new_tensor_3d(ctx, kvType, headDim, kvLen, kvHeads);
        ggml_tensor* v_cache_t = ggml_new_tensor_3d(ctx, kvType, headDim, kvLen, kvHeads);

        ggml_tensor* sinks_t = (sinksData != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, numHeads) : nullptr;

        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seqLen);
        std::vector<int32_t> pos_data(seqLen);
        for (int i = 0; i < seqLen; i++) pos_data[i] = startPos + i;

        // === Build graph ===

        // 1. attention norm (RMSNorm + scale)
        ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden_t, eps), attn_norm_w);

        // 2. QKV projection (+ bias). For fused QKV the output is [qDim+2kDim, seqLen]
        // and we slice into Q/K/V views; for the separate path we run three matmuls.
        ggml_tensor* q_heads = nullptr;
        ggml_tensor* k_heads = nullptr;
        ggml_tensor* v_heads = nullptr;

        if (isQkvFused)
        {
            ggml_tensor* qkv_out = ggml_mul_mat(ctx, qkv_w, normed);
            if (qkv_b != nullptr)
                qkv_out = ggml_add(ctx, qkv_out, qkv_b);

            // Strided 2-D views into the fused QKV output, then ggml_cont +
            // reshape to [headDim, n_heads * seqLen] so RoPE sees the standard
            // 4-D [headDim, n_heads, seqLen, 1] layout (heads-fastest within the
            // flattened second dim, matching Gemma4LayerPrefill's QKV split).
            ggml_tensor* q_raw = ggml_view_2d(ctx, qkv_out, qDim, seqLen,
                qkv_out->nb[1], 0);
            ggml_tensor* k_raw = ggml_view_2d(ctx, qkv_out, kDim, seqLen,
                qkv_out->nb[1], static_cast<std::size_t>(qDim) * sizeof(float));
            ggml_tensor* v_raw = ggml_view_2d(ctx, qkv_out, kDim, seqLen,
                qkv_out->nb[1], static_cast<std::size_t>(qDim + kDim) * sizeof(float));

            q_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, q_raw), headDim, numHeads * seqLen);
            k_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, k_raw), headDim, kvHeads * seqLen);
            v_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, v_raw), headDim, kvHeads * seqLen);
        }
        else
        {
            ggml_tensor* q_proj = ggml_mul_mat(ctx, qkv_w, normed);
            if (qkv_b != nullptr) q_proj = ggml_add(ctx, q_proj, qkv_b);
            ggml_tensor* k_proj = ggml_mul_mat(ctx, k_w, normed);
            if (k_b != nullptr) k_proj = ggml_add(ctx, k_proj, k_b);
            ggml_tensor* v_proj = ggml_mul_mat(ctx, v_w, normed);
            if (v_b != nullptr) v_proj = ggml_add(ctx, v_proj, v_b);

            q_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, q_proj), headDim, numHeads * seqLen);
            k_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, k_proj), headDim, kvHeads * seqLen);
            v_heads = ggml_reshape_2d(ctx, ggml_cont(ctx, v_proj), headDim, kvHeads * seqLen);
        }

        // 3. RoPE (NeoX yarn-scaled). Q and K share the same position tensor.
        ggml_tensor* q_4d = ggml_reshape_4d(ctx, q_heads, headDim, numHeads, seqLen, 1);
        ggml_tensor* k_4d = ggml_reshape_4d(ctx, k_heads, headDim, kvHeads, seqLen, 1);
        ggml_tensor* q_roped = ggml_rope_ext(ctx, q_4d, pos_tensor, nullptr,
            ropeDims, /*mode=*/2, originalContextLength, ropeBase, ropeFreqScale,
            /*ext_factor=*/1.0f, /*attn_factor=*/1.0f,
            /*beta_fast=*/32.0f, /*beta_slow=*/1.0f);
        ggml_tensor* k_roped = ggml_rope_ext(ctx, k_4d, pos_tensor, nullptr,
            ropeDims, 2, originalContextLength, ropeBase, ropeFreqScale,
            1.0f, 1.0f, 32.0f, 1.0f);

        // 4. Reshape to attention layout. Q stays as a strided permute view (matches
        // the Gemma4 prefill kernel and llama.cpp's build_attn_mha). Fresh K/V are
        // brought to the tight [headDim, kvHeads, seqLen] layout so they can be
        // ggml_cpy'd into the cache and (for the fallback path) directly attended.
        ggml_tensor* q_attn = ggml_permute(ctx, q_roped, 0, 2, 1, 3);
        ggml_tensor* k_fresh = ggml_reshape_3d(ctx,
            ggml_cont(ctx, ggml_permute(ctx, k_roped, 0, 2, 1, 3)),
            headDim, seqLen, kvHeads);
        ggml_tensor* v_3d_pre = ggml_reshape_4d(ctx, v_heads, headDim, kvHeads, seqLen, 1);
        ggml_tensor* v_fresh = ggml_reshape_3d(ctx,
            ggml_cont(ctx, ggml_permute(ctx, v_3d_pre, 0, 2, 1, 3)),
            headDim, seqLen, kvHeads);

        // 5. KV cache write: contiguous append at startPos. Uses nb[1] so the offset
        // is correct for any cache dtype (F32/F16); ggml_cpy converts F32 fresh K/V
        // to the cache type as needed.
        std::size_t kv_offset = static_cast<std::size_t>(startPos) * k_cache_t->nb[1];
        ggml_tensor* k_dst = ggml_view_3d(ctx, k_cache_t,
            headDim, seqLen, kvHeads,
            k_cache_t->nb[1], k_cache_t->nb[2], kv_offset);
        ggml_tensor* v_dst = ggml_view_3d(ctx, v_cache_t,
            headDim, seqLen, kvHeads,
            v_cache_t->nb[1], v_cache_t->nb[2], kv_offset);
        ggml_tensor* k_cpy = ggml_cpy(ctx, k_fresh, k_dst);
        ggml_tensor* v_cpy = ggml_cpy(ctx, v_fresh, v_dst);

        // 6. Build the attention K/V source in the llama.cpp / Gemma4 tight
        // layout [headDim, kvLen, kvHeads] (heads slowest, positions in the
        // middle). For chunk 1 we use the fresh K/V directly (already in this
        // layout); for chunk 2+ we view the cache prefix [headDim, startPos,
        // kvHeads] (materialising as F32 when the cache is F16) and ggml_concat
        // it with fresh along ne[1] (positions).
        ggml_tensor* k_attn = k_fresh;
        ggml_tensor* v_attn = v_fresh;
        if (startPos > 0)
        {
            ggml_tensor* k_prev = ggml_view_3d(ctx, k_cache_t,
                headDim, startPos, kvHeads,
                k_cache_t->nb[1], k_cache_t->nb[2], 0);
            ggml_tensor* v_prev = ggml_view_3d(ctx, v_cache_t,
                headDim, startPos, kvHeads,
                v_cache_t->nb[1], v_cache_t->nb[2], 0);

            if (kvType != GGML_TYPE_F32)
            {
                ggml_tensor* k_prev_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, startPos, kvHeads);
                ggml_tensor* v_prev_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, startPos, kvHeads);
                k_prev = ggml_cpy(ctx, k_prev, k_prev_f32);
                v_prev = ggml_cpy(ctx, v_prev, v_prev_f32);
            }
            // Concat along ne[1] (positions); both inputs share [headDim, *, kvHeads].
            k_attn = ggml_concat(ctx, k_prev, k_fresh, 1);
            v_attn = ggml_concat(ctx, v_prev, v_fresh, 1);
        }

        // 7. Causal + optional SWA mask. The GPT-OSS C# attention path uses
        // exactly this: cell (q_idx, kv_idx) is unmasked iff kv_idx <= startPos +
        // q_idx (causal) AND, for SWA layers, kv_idx > startPos + q_idx -
        // slidingWindow.
        ggml_tensor* mask_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kvLen, seqLen, 1, 1);
        std::vector<ggml_fp16_t> mask_data(static_cast<std::size_t>(kvLen) * seqLen);
        {
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            const int win = (isSwa != 0) ? slidingWindow : 0;
            for (int qi = 0; qi < seqLen; qi++)
            {
                int threshold = startPos + qi;
                int winStart = (win > 0) ? std::max(0, threshold - win + 1) : 0;
                ggml_fp16_t* row = &mask_data[static_cast<std::size_t>(qi) * kvLen];
                for (int ki = 0; ki < kvLen; ki++)
                    row[ki] = (ki > threshold || ki < winStart) ? neg_inf : zero_val;
            }
        }

        // 8. Attention. Use ggml_flash_attn_ext (with optional sinks) when the
        // backend supports the op for the current K/V dtype + head_dim — this
        // avoids materialising the [kvLen, seqLen, numHeads] scores tensor
        // (which is multi-tens-of-MiB at long contexts and triggers GPU OOM
        // when several layers' worth are in-flight on Metal). Fall back to the
        // explicit mul_mat → soft_max → mul_mat chain only when flash_attn_ext
        // isn't supported.
        ggml_tensor* attn_flat = nullptr;
        ggml_tensor* fa_test = ggml_flash_attn_ext(ctx, q_attn, k_attn, v_attn, mask_t,
            scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa_test, GGML_PREC_F32);
        if (sinks_t != nullptr)
            ggml_flash_attn_ext_add_sinks(fa_test, sinks_t);
        const bool fa_supported = backend_supports_op(fa_test);
        if (fa_supported)
        {
            attn_flat = ggml_reshape_2d(ctx, fa_test, qDim, seqLen);
        }
        else
        {
            ggml_tensor* q_attn_cont = ggml_cont(ctx, q_attn);
            ggml_tensor* scores = ggml_mul_mat(ctx, k_attn, q_attn_cont);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, mask_t, scale, 0.0f);
            if (sinks_t != nullptr)
                ggml_soft_max_add_sinks(probs, sinks_t);
            ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_attn, 1, 0, 2, 3));
            ggml_tensor* attn_out = ggml_mul_mat(ctx, v_perm, probs);
            ggml_tensor* attn_perm = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
            attn_flat = ggml_reshape_2d(ctx, attn_perm, qDim, seqLen);
        }

        // 9. Output projection (+ bias) and residual add.
        ggml_tensor* o_out = ggml_mul_mat(ctx, o_w, attn_flat);
        if (o_b != nullptr)
            o_out = ggml_add(ctx, o_out, o_b);
        ggml_tensor* residual = ggml_add(ctx, hidden_t, o_out);

        ggml_tensor* output = ggml_cpy(ctx, residual, hidden_out_t);
        ggml_set_output(output);

        // === Build & bind the graph ===
        const std::size_t graph_size = 1024;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(graph, k_cpy);
        ggml_build_forward_expand(graph, v_cpy);
        ggml_build_forward_expand(graph, output);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* d; std::size_t b; };
        std::vector<HostBinding> uploads;
        std::vector<BufferHandle> ephem;

        auto bind = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cache,
                        enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            if (cache && bytes >= 4096) {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs, usage)) {
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS) {
                        if (needs) uploads.push_back({t, data, bytes});
                        return;
                    }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096) {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cache, buf)) {
                    if (!cache) ephem.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS) return;
                }
            }
            uploads.push_back({t, data, bytes});
        };

        bind(qkv_w, qkvW, static_cast<std::size_t>(qkvBytes), true);
        bind(o_w, oW, static_cast<std::size_t>(oBytes), true);
        bind(attn_norm_w, attnNormW, hiddenSize * sizeof(float), true);
        if (qkv_b != nullptr) bind(qkv_b, qkvB, qkvDim * sizeof(float), true);
        if (k_w != nullptr) bind(k_w, kW, static_cast<std::size_t>(kBytes), true);
        if (v_w != nullptr) bind(v_w, vW, static_cast<std::size_t>(vBytes), true);
        if (k_b != nullptr) bind(k_b, kB, kDim * sizeof(float), true);
        if (v_b != nullptr) bind(v_b, vB, kDim * sizeof(float), true);
        if (o_b != nullptr) bind(o_b, oB, hiddenSize * sizeof(float), true);
        if (sinks_t != nullptr) bind(sinks_t, sinksData, numHeads * sizeof(float), true);
        // NOTE: k_cache_t / v_cache_t are now small per-call windows
        // [headDim, kvLen, kvHeads] sized to the active prefix only. They are
        // intentionally NOT bound to the host cache (kCacheData / vCacheData)
        // here — they're allocated by ggml_backend_alloc_ctx_tensors below as
        // GPU-only scratch, then we manually upload the prefix (per head, F32
        // path only when startPos > 0) before compute and download the freshly
        // appended slice back to the host cache after compute. This keeps the
        // KV cache GPU residency to O(kvLen * kvHeads * headDim) per layer
        // instead of O(cacheSize * kvHeads * headDim), which on GPT-OSS at
        // context_length=131072 would otherwise exhaust Metal's working set.

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr) {
            set_last_error("Failed to allocate buffer for GPT-OSS attention layer prefill.");
            return 0;
        }

        // Drain any pending async work targeting hidden_data (the previous layer
        // / MoE FFN may have written it via a deferred-sync zero-copy path).
        host_read_barrier();

        for (auto& u : uploads)
            ggml_backend_tensor_set(u.t, u.d, 0, u.b);

        ggml_backend_tensor_set(hidden_t, hidden_data, 0,
            static_cast<std::size_t>(hiddenSize) * seqLen * sizeof(float));
        ggml_backend_tensor_set(pos_tensor, pos_data.data(), 0, seqLen * sizeof(int32_t));
        ggml_backend_tensor_set(mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

        // Upload the existing K/V cache prefix [0, startPos) into the per-call
        // window. Host cache layout is [headDim, cacheSize, kvHeads] (heads
        // slowest, contiguous within head), and the window is
        // [headDim, kvLen, kvHeads] — same layout but with kvLen instead of
        // cacheSize for the position dim. We therefore upload per-head: for
        // each head h, copy `startPos * headDim * elemSize` bytes from the
        // host cache (offset h * cacheSize * headDim * elemSize) into the
        // window (offset h * kvLen * headDim * elemSize). For chunk 1
        // (startPos == 0) no upload is needed.
        const std::size_t elemSize = ggml_type_size(kvType);
        if (startPos > 0)
        {
            const std::size_t hostStrideBytes   = static_cast<std::size_t>(cacheSize) * headDim * elemSize;
            const std::size_t windowStrideBytes = static_cast<std::size_t>(kvLen)     * headDim * elemSize;
            const std::size_t prefixBytes       = static_cast<std::size_t>(startPos)  * headDim * elemSize;
            char* kHost = static_cast<char*>(kCacheData);
            char* vHost = static_cast<char*>(vCacheData);
            for (int h = 0; h < kvHeads; h++)
            {
                ggml_backend_tensor_set(k_cache_t, kHost + h * hostStrideBytes,
                    h * windowStrideBytes, prefixBytes);
                ggml_backend_tensor_set(v_cache_t, vHost + h * hostStrideBytes,
                    h * windowStrideBytes, prefixBytes);
            }
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            set_last_error("Graph compute failed for GPT-OSS attention layer prefill.");
            return 0;
        }

        // Synchronously download hidden_out + the freshly written K/V slice
        // and wait for the GPU to retire all command buffers before returning.
        // We cannot use the async download here because BufferHandle's
        // destructor frees the per-call compute buffer immediately, while
        // pipelined MoE work can exhaust the GPU working set otherwise.
        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(hidden_out_t, hidden_data, 0,
            static_cast<std::size_t>(hiddenSize) * seqLen * sizeof(float));

        // Download the freshly appended K/V slice [startPos, kvLen) per head
        // back to the host cache (mirror of the upload step).
        {
            const std::size_t hostStrideBytes   = static_cast<std::size_t>(cacheSize) * headDim * elemSize;
            const std::size_t windowStrideBytes = static_cast<std::size_t>(kvLen)     * headDim * elemSize;
            const std::size_t freshOffsetBytes  = static_cast<std::size_t>(startPos)  * headDim * elemSize;
            const std::size_t freshBytes        = static_cast<std::size_t>(seqLen)    * headDim * elemSize;
            char* kHost = static_cast<char*>(kCacheData);
            char* vHost = static_cast<char*>(vCacheData);
            for (int h = 0; h < kvHeads; h++)
            {
                ggml_backend_tensor_get(k_cache_t,
                    kHost + h * hostStrideBytes + freshOffsetBytes,
                    h * windowStrideBytes + freshOffsetBytes, freshBytes);
                ggml_backend_tensor_get(v_cache_t,
                    vHost + h * hostStrideBytes + freshOffsetBytes,
                    h * windowStrideBytes + freshOffsetBytes, freshBytes);
            }
        }

        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in GPT-OSS attention layer prefill."); return 0; }
}

// ============================================================================
// Qwen3.5 attention layer prefill: full attention block (norm → fused QKV with
// interleaved Q+gate → per-head Q/K norm → RoPE → KV cache append → causal-
// masked softmax → attention → sigmoid-gated mix → output projection +
// residual) as ONE GGML graph dispatch per layer.
//
// llama.cpp reference: src/models/qwen35moe.cpp::llm_build_qwen35moe::build_qkvz
// + the surrounding attention block in qwen35moe.cpp.
//
// Key model-specific points:
//   - Fused QKV produces interleaved Q+gate per head: layout
//     [head_dim, 2, num_heads, seqLen] in memory, where the inner ne[1]==2
//     selects between the Q half (index 0) and the gate half (index 1).
//   - Per-head Q-norm and K-norm (RMSNorm+scale) before RoPE.
//   - Sigmoid-gated mix: attn_out *= sigmoid(gate).
//   - Output projection has NO bias.
//   - Single contiguous KV cache per layer (no rolling window). Mask is plain
//     causal (no SWA).
// ============================================================================
TSG_EXPORT int TSGgml_Qwen35AttentionLayerPrefill(
    float* hidden_data,        // [seqLen * hiddenSize] in/out
    int hiddenSize, int seqLen,
    void* attnNormW,
    void* qkvW, int qkvType, std::int64_t qkvNe0, std::int64_t qkvNe1, std::int64_t qkvBytes,
    void* qNormW, void* kNormW,
    void* oW, int oType, std::int64_t oNe0, std::int64_t oNe1, std::int64_t oBytes,
    void* kCacheData, void* vCacheData,
    int numHeads, int kvHeads, int headDim,
    int cacheSize, int startPos,
    float ropeBase, float ropeFreqScale, int ropeDims,
    int ropeMode,
    int kvCacheType,
    float eps)
{
    try
    {
        if (!ensure_backend()) return 0;

        const int qDim = numHeads * headDim;          // post-deinterleave Q dim
        const int qFullDim = qDim * 2;                // pre-deinterleave Q+gate dim
        const int kDim = kvHeads * headDim;
        const int totalSeqLen = startPos + seqLen;
        const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
        const ggml_type kvType = static_cast<ggml_type>(kvCacheType);
        const int kvLen = totalSeqLen;

        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create context for Qwen3.5 attention layer prefill.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // === Tensor declarations ===
        ggml_tensor* hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hiddenSize, seqLen);
        ggml_tensor* hidden_out_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hiddenSize, seqLen);
        ggml_tensor* attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hiddenSize);
        ggml_tensor* q_norm_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, headDim);
        ggml_tensor* k_norm_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, headDim);
        ggml_tensor* qkv_w       = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkvType), qkvNe0, qkvNe1);
        ggml_tensor* o_w         = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(oType), oNe0, oNe1);
        ggml_tensor* k_cache_t   = ggml_new_tensor_3d(ctx, kvType, headDim, cacheSize, kvHeads);
        ggml_tensor* v_cache_t   = ggml_new_tensor_3d(ctx, kvType, headDim, cacheSize, kvHeads);
        ggml_tensor* pos_tensor  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seqLen);
        std::vector<int32_t> pos_data(seqLen);
        for (int i = 0; i < seqLen; i++) pos_data[i] = startPos + i;

        // === Build graph ===

        // 1. Attention norm
        ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden_t, eps), attn_norm_w);

        // 2. Fused QKV projection -> [qFullDim + 2*kDim, seqLen]
        ggml_tensor* qkv_out = ggml_mul_mat(ctx, qkv_w, normed);

        // 3. Slice Q+gate / K / V from the fused output. Q+gate occupies the first
        // qFullDim rows; per-token they're laid out [head0_Q (headDim), head0_gate
        // (headDim), head1_Q, head1_gate, ...]. We expose them as a 4-D view
        // [head_dim, 2, num_heads, seqLen] so a strided 3-D view picks Q (idx 0
        // along ne[1]) or gate (idx 1) per token contiguously per head.
        ggml_tensor* qg_part = ggml_view_2d(ctx, qkv_out, qFullDim, seqLen,
            qkv_out->nb[1], 0);
        ggml_tensor* k_raw = ggml_view_2d(ctx, qkv_out, kDim, seqLen,
            qkv_out->nb[1], static_cast<std::size_t>(qFullDim) * sizeof(float));
        ggml_tensor* v_raw = ggml_view_2d(ctx, qkv_out, kDim, seqLen,
            qkv_out->nb[1], static_cast<std::size_t>(qFullDim + kDim) * sizeof(float));

        // Reshape qg to [head_dim, 2, num_heads, seqLen]. The view over the fused
        // QKV output is contiguous along the row direction so the reshape is free.
        ggml_tensor* qg_4d = ggml_reshape_4d(ctx, ggml_cont(ctx, qg_part),
            headDim, 2, numHeads, seqLen);

        // Q view: pick ne[1] == 0 per head per token. Shape [head_dim, num_heads, seqLen].
        ggml_tensor* q_view = ggml_view_3d(ctx, qg_4d,
            headDim, numHeads, seqLen,
            qg_4d->nb[2], qg_4d->nb[3], 0);
        ggml_tensor* gate_view = ggml_view_3d(ctx, qg_4d,
            headDim, numHeads, seqLen,
            qg_4d->nb[2], qg_4d->nb[3], qg_4d->nb[1]);

        ggml_tensor* q_cont = ggml_cont(ctx, q_view);     // [headDim, numHeads, seqLen]
        ggml_tensor* gate_cont = ggml_cont(ctx, gate_view); // [headDim, numHeads, seqLen]
        ggml_tensor* k_3d_raw = ggml_reshape_3d(ctx, ggml_cont(ctx, k_raw), headDim, kvHeads, seqLen);
        ggml_tensor* v_3d_raw = ggml_reshape_3d(ctx, ggml_cont(ctx, v_raw), headDim, kvHeads, seqLen);

        // 4. Per-head Q/K norm. RMSNorm normalizes along ne[0] (head_dim); we
        // reshape to 2D [head_dim, numHeads*seqLen] so each "row" is one head's
        // worth of values. Then multiply by the per-dim scale weights.
        ggml_tensor* q_norm_in = ggml_reshape_2d(ctx, q_cont, headDim, numHeads * seqLen);
        ggml_tensor* k_norm_in = ggml_reshape_2d(ctx, k_3d_raw, headDim, kvHeads * seqLen);
        ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_norm_in, eps), q_norm_w);
        ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_norm_in, eps), k_norm_w);

        // 5. RoPE (NeoX) on Q and K. Reshape back to 4D [head_dim, n_heads, seqLen, 1].
        ggml_tensor* q_4d = ggml_reshape_4d(ctx, q_normed, headDim, numHeads, seqLen, 1);
        ggml_tensor* k_4d = ggml_reshape_4d(ctx, k_normed, headDim, kvHeads, seqLen, 1);

        ggml_tensor* q_roped = ggml_rope_ext(ctx, q_4d, pos_tensor, nullptr,
            ropeDims, ropeMode, 0, ropeBase, ropeFreqScale,
            0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor* k_roped = ggml_rope_ext(ctx, k_4d, pos_tensor, nullptr,
            ropeDims, ropeMode, 0, ropeBase, ropeFreqScale,
            0.0f, 1.0f, 0.0f, 0.0f);

        // 6. Build attention layout. q_attn: [headDim, seqLen, numHeads] (heads
        // become the slowest dim, matching what build_attn_mha consumes).
        ggml_tensor* q_attn = ggml_permute(ctx, q_roped, 0, 2, 1, 3);

        // Bring fresh K/V to [headDim, seqLen, kvHeads] so cache_cpy + the
        // chunk-1 attention path see the same tight layout the cache uses.
        ggml_tensor* k_fresh = ggml_reshape_3d(ctx,
            ggml_cont(ctx, ggml_permute(ctx, k_roped, 0, 2, 1, 3)),
            headDim, seqLen, kvHeads);
        // v_3d_raw is [headDim, kvHeads, seqLen]; permute to [headDim, seqLen, kvHeads].
        ggml_tensor* v_3d_pre = ggml_reshape_4d(ctx, v_3d_raw, headDim, kvHeads, seqLen, 1);
        ggml_tensor* v_fresh = ggml_reshape_3d(ctx,
            ggml_cont(ctx, ggml_permute(ctx, v_3d_pre, 0, 2, 1, 3)),
            headDim, seqLen, kvHeads);

        // 7. KV cache append at startPos.
        std::size_t kv_offset = static_cast<std::size_t>(startPos) * k_cache_t->nb[1];
        ggml_tensor* k_dst = ggml_view_3d(ctx, k_cache_t,
            headDim, seqLen, kvHeads,
            k_cache_t->nb[1], k_cache_t->nb[2], kv_offset);
        ggml_tensor* v_dst = ggml_view_3d(ctx, v_cache_t,
            headDim, seqLen, kvHeads,
            v_cache_t->nb[1], v_cache_t->nb[2], kv_offset);
        ggml_tensor* k_cpy = ggml_cpy(ctx, k_fresh, k_dst);
        ggml_tensor* v_cpy = ggml_cpy(ctx, v_fresh, v_dst);

        // 8. K/V attention source in the llama.cpp tight layout
        // [headDim, kvLen, kvHeads]. Chunk 1 uses fresh directly; continuations
        // view the cache prefix [headDim, startPos, kvHeads] (materialising as
        // F32 when stored F16) and concat with fresh along ne[1] (positions).
        ggml_tensor* k_attn = k_fresh;
        ggml_tensor* v_attn = v_fresh;
        if (startPos > 0)
        {
            ggml_tensor* k_prev = ggml_view_3d(ctx, k_cache_t,
                headDim, startPos, kvHeads,
                k_cache_t->nb[1], k_cache_t->nb[2], 0);
            ggml_tensor* v_prev = ggml_view_3d(ctx, v_cache_t,
                headDim, startPos, kvHeads,
                v_cache_t->nb[1], v_cache_t->nb[2], 0);

            if (kvType != GGML_TYPE_F32)
            {
                ggml_tensor* k_prev_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, startPos, kvHeads);
                ggml_tensor* v_prev_f32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, startPos, kvHeads);
                k_prev = ggml_cpy(ctx, k_prev, k_prev_f32);
                v_prev = ggml_cpy(ctx, v_prev, v_prev_f32);
            }
            k_attn = ggml_concat(ctx, k_prev, k_fresh, 1);
            v_attn = ggml_concat(ctx, v_prev, v_fresh, 1);
        }

        // 9. Plain causal mask (no SWA on Qwen3.5 dense / MoE layers).
        ggml_tensor* mask_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kvLen, seqLen, 1, 1);
        std::vector<ggml_fp16_t> mask_data(static_cast<std::size_t>(kvLen) * seqLen);
        {
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            for (int qi = 0; qi < seqLen; qi++)
            {
                int threshold = startPos + qi;
                ggml_fp16_t* row = &mask_data[static_cast<std::size_t>(qi) * kvLen];
                for (int ki = 0; ki < kvLen; ki++)
                    row[ki] = (ki > threshold) ? neg_inf : zero_val;
            }
        }

        // 10. Attention. Use ggml_flash_attn_ext when supported (no sinks needed
        // for Qwen3.5 dense); fall back to mul_mat → soft_max → mul_mat otherwise.
        ggml_tensor* attn_flat = nullptr;
        ggml_tensor* fa_test = ggml_flash_attn_ext(ctx, q_attn, k_attn, v_attn, mask_t,
            scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(fa_test, GGML_PREC_F32);
        if (backend_supports_op(fa_test))
        {
            attn_flat = ggml_reshape_2d(ctx, fa_test, qDim, seqLen);
        }
        else
        {
            ggml_tensor* q_attn_cont = ggml_cont(ctx, q_attn);
            ggml_tensor* scores = ggml_mul_mat(ctx, k_attn, q_attn_cont);
            ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
            ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, mask_t, scale, 0.0f);
            ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_attn, 1, 0, 2, 3));
            ggml_tensor* attn_out = ggml_mul_mat(ctx, v_perm, probs);
            ggml_tensor* attn_perm = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
            attn_flat = ggml_reshape_2d(ctx, attn_perm, qDim, seqLen);
        }

        // 11. Sigmoid-gated mix: attn_flat *= sigmoid(gate). gate_cont is
        // [headDim, numHeads, seqLen] with the same per-head per-token order
        // attn_flat ([qDim, seqLen]) flattens to, so a reshape is enough.
        ggml_tensor* gate_flat = ggml_reshape_2d(ctx, gate_cont, qDim, seqLen);
        ggml_tensor* gate_sig = ggml_sigmoid(ctx, gate_flat);
        ggml_tensor* attn_gated = ggml_mul(ctx, attn_flat, gate_sig);

        // 12. Output projection (no bias) and residual add.
        ggml_tensor* o_out = ggml_mul_mat(ctx, o_w, attn_gated);
        ggml_tensor* residual = ggml_add(ctx, hidden_t, o_out);

        ggml_tensor* output = ggml_cpy(ctx, residual, hidden_out_t);
        ggml_set_output(output);

        // === Build & bind ===
        const std::size_t graph_size = 1024;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(graph, k_cpy);
        ggml_build_forward_expand(graph, v_cpy);
        ggml_build_forward_expand(graph, output);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* d; std::size_t b; };
        std::vector<HostBinding> uploads;
        std::vector<BufferHandle> ephem;

        auto bind = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cache,
                        enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            if (cache && bytes >= 4096) {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs, usage)) {
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS) {
                        if (needs) uploads.push_back({t, data, bytes});
                        return;
                    }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096) {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cache, buf)) {
                    if (!cache) ephem.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS) return;
                }
            }
            uploads.push_back({t, data, bytes});
        };

        bind(qkv_w, qkvW, static_cast<std::size_t>(qkvBytes), true);
        bind(o_w, oW, static_cast<std::size_t>(oBytes), true);
        bind(attn_norm_w, attnNormW, hiddenSize * sizeof(float), true);
        bind(q_norm_w, qNormW, headDim * sizeof(float), true);
        bind(k_norm_w, kNormW, headDim * sizeof(float), true);
        const std::size_t kvCacheBytes = kv_cache_bytes(kvHeads, cacheSize, headDim, kvType);
        bind(k_cache_t, kCacheData, kvCacheBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        bind(v_cache_t, vCacheData, kvCacheBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr) {
            set_last_error("Failed to allocate buffer for Qwen3.5 attention layer prefill.");
            return 0;
        }

        host_read_barrier();

        for (auto& u : uploads)
            ggml_backend_tensor_set(u.t, u.d, 0, u.b);

        ggml_backend_tensor_set(hidden_t, hidden_data, 0,
            static_cast<std::size_t>(hiddenSize) * seqLen * sizeof(float));
        ggml_backend_tensor_set(pos_tensor, pos_data.data(), 0, seqLen * sizeof(int32_t));
        ggml_backend_tensor_set(mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS) {
            set_last_error("Graph compute failed for Qwen3.5 attention layer prefill.");
            return 0;
        }

        finalize_compute_with_download(hidden_out_t, hidden_data,
            static_cast<std::size_t>(hiddenSize) * seqLen * sizeof(float));

        // The K/V cache writes (in-graph ggml_cpy(k_fresh -> k_dst)) land in
        // the cacheable backend buffer for kCacheData / vCacheData. On
        // unified-memory backends (Apple Silicon Metal HostPtr buffers) this
        // is a no-op since the device buffer IS the host pointer. On the
        // DeviceCopy path (which Apple Silicon Metal currently takes because
        // GGML's metal device props don't initialise `integrated`), the
        // GPU-side writes need to be explicitly downloaded back to host so
        // the legacy CPU SIMD decode path (AttentionDecodePureCS, which reads
        // kCache via GetFloatPtr) sees the freshly-written K/V. Without this
        // sync, decode reads stale host memory and produces degenerate or
        // repeating output. This is cheap when the path is HostPtr (single
        // atomic check) and sized to bytes when the path is DeviceCopy.
        sync_cached_buffer_to_host(kCacheData, kvCacheBytes);
        sync_cached_buffer_to_host(vCacheData, kvCacheBytes);

        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Qwen3.5 attention layer prefill."); return 0; }
}
