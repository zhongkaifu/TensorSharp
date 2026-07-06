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
#include <chrono>
#include <cstdio>

using namespace tsg;

#ifdef TSG_GGML_USE_CUDA
// On-device causal-mask fill (ggml_ops_mask.cu): generate the verify kernel's
// [kvLen, N] F16 causal(+windowed) masks straight into their device buffers,
// eliminating the host fill + H2D upload. Bit-identical to the host path.
extern "C" bool tsg_cuda_fill_causal_mask_f16(
    void* mask_dev, int kvLen, int N, int nPast, int window, int validLen);
extern "C" bool tsg_cuda_sync_stream0(void);
#endif

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
            case GGML_TYPE_Q4_0: return 0;
            default:             return 4;
        }
    }

    inline bool kv_cache_is_block_quantized(int kv_cache_type)
    {
        const ggml_type t = static_cast<ggml_type>(kv_cache_type);
        return t == GGML_TYPE_Q8_0 || t == GGML_TYPE_Q4_0;
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

    // Upload one captured-decode INPUT tensor without a per-copy stream sync.
    // On CUDA the copy is queued on the backend stream (ordered ahead of the graph
    // replay that reads it; the post-compute download syncs the stream), so we drop
    // the redundant cudaStreamSynchronize that the synchronous setter issues per
    // call — meaningful when a decode token refreshes ~2*num_layers small inputs.
    // CUDA pageable host->device async is host-synchronous w.r.t. the source, so a
    // caller's transient buffer (e.g. a per-layer mask vector) is safe to free right
    // after this returns. Non-CUDA backends fall back to the synchronous setter.
    inline void decode_input_set_async(ggml_tensor* tensor, const void* data, std::size_t bytes)
    {
        if (tensor == nullptr || data == nullptr || bytes == 0)
            return;
        if (g_backend_type == BACKEND_TYPE_CUDA)
            ggml_backend_tensor_set_async(g_backend, tensor, data, 0, bytes);
        else
            ggml_backend_tensor_set(tensor, data, 0, bytes);
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
// Qwen3.5/3.6 FULL-MODEL decode: the whole hybrid transformer (full-attention +
// GatedDeltaNet recurrent layers, with a per-layer dense or MoE FFN) executed as
// ONE GGML graph per decode token. This collapses the ~120-400 per-op kernel
// dispatches/token (each a WDDM submit + host sync, the dominant decode cost on
// this architecture) down to a single graph_compute, mirroring llama.cpp's
// single-graph decode (src/models/qwen35moe.cpp / delta-net-base.cpp).
//
// Attention layers: built exactly like TSGgml_Qwen35AttentionLayerDecode (fused
// QKV with interleaved Q+gate, per-head q/k RMSNorm, NeoX RoPE, device-resident
// circular KV cache append, flash attention, sigmoid-gated output, o-proj).
// Recurrent (GDN) layers: built like llama.cpp's build_layer_attn_linear +
// build_delta_net_fused — qkv/z/beta/alpha projections, ssm_conv over a host
// conv-state ring, SiLU, q/k L2-norm + head tiling, the fused ggml_gated_delta_net
// op (K=1), gated RMSNorm with z, and the ssm output projection.
//
// GDN recurrent state (conv ring + delta state) is passed per-token via host
// in/out pointers (no device residency / latch), which keeps this path trivially
// correct alongside the per-op fallback and MTP (the host buffers are always the
// single source of truth). Attention KV cache stays device-resident (cacheable
// bind, in-place append) exactly like the per-op attention kernel.
//
// Output is the final pre-output-norm hidden state [hidden_size]; the C# caller
// owns output_norm + the LM head. Returns 0 on anything it cannot handle so the
// caller falls back to the per-op decode.
// ============================================================================
struct TSGgmlQwen35LayerDesc
{
    // --- pointers (host memory); 8-byte each, FIRST per interop convention ---
    void* attn_norm_w;       // [hidden] F32 (input norm, both layer kinds)
    void* post_attn_norm_w;  // [hidden] F32 (FFN input norm)
    // attention layer
    void* qkv_w;             // attn_qkv [hidden, 2*qDim + 2*kDim]
    void* q_norm_w;          // [head_dim] F32
    void* k_norm_w;          // [head_dim] F32
    void* o_w;               // attn_output [qDim, hidden]
    void* k_cache;           // device-resident [kv_heads, cache, head_dim]
    void* v_cache;
    // gated-delta-net layer
    void* gdn_qkv_w;         // attn_qkv (recurrent) [hidden, conv_dim]
    void* gdn_gate_w;        // attn_gate / z [hidden, value_dim]
    void* ssm_beta_w;        // [hidden, num_v_heads]
    void* ssm_alpha_w;       // [hidden, num_v_heads]
    void* conv1d_w;          // [conv_kernel, conv_dim] F32
    void* ssm_dt_w;          // [num_v_heads] F32 (dt bias)
    void* ssm_a_w;           // [num_v_heads] F32 (-exp(A_log))
    void* ssm_norm_w;        // [head_v_dim] F32
    void* ssm_out_w;         // [value_dim, hidden]
    void* conv_state_in;     // host [conv_kernel-1, conv_dim] ggml layout (ne0=time)
    void* delta_state_in;    // host [head_k_dim, head_v_dim, num_v_heads]
    void* conv_state_out;    // host, same layout as conv_state_in
    void* delta_state_out;   // host, same layout as delta_state_in
    // dense FFN
    void* gu_w;              // ffn_gate_up [hidden, 2*ff_dense]
    void* down_w;            // ffn_down [ff_dense, hidden]
    // separate attention K/V (when separate_qkv: qkv_w holds Q+gate [hidden, 2*qDim])
    void* k_w;
    void* v_w;
    // MoE FFN (used when is_moe != 0)
    void* gate_inp_w;        // router [hidden, num_experts]
    void* gate_exps;         // stacked [hidden, expert_ff, num_experts]
    void* up_exps;           // stacked [hidden, expert_ff, num_experts]
    void* down_exps;         // stacked [expert_ff, hidden, num_experts]
    void* shexp_gate_w;      // [hidden, shared_ff]
    void* shexp_up_w;        // [hidden, shared_ff]
    void* shexp_down_w;      // [shared_ff, hidden]
    void* shexp_gate_inp_w;  // [hidden] F32 (shared-expert sigmoid gate)

    // --- int64 weight shapes/bytes ---
    std::int64_t qkv_ne0, qkv_ne1, qkv_bytes;
    std::int64_t o_ne0, o_ne1, o_bytes;
    std::int64_t k_ne0, k_ne1, k_bytes;
    std::int64_t v_ne0, v_ne1, v_bytes;
    std::int64_t gdn_qkv_ne0, gdn_qkv_ne1, gdn_qkv_bytes;
    std::int64_t gdn_gate_ne0, gdn_gate_ne1, gdn_gate_bytes;
    std::int64_t ssm_beta_ne0, ssm_beta_ne1, ssm_beta_bytes;
    std::int64_t ssm_alpha_ne0, ssm_alpha_ne1, ssm_alpha_bytes;
    std::int64_t ssm_out_ne0, ssm_out_ne1, ssm_out_bytes;
    std::int64_t gu_ne0, gu_ne1, gu_bytes;
    std::int64_t down_ne0, down_ne1, down_bytes;
    std::int64_t gate_inp_ne0, gate_inp_ne1, gate_inp_bytes;
    std::int64_t gate_exps_bytes, up_exps_bytes, down_exps_bytes;
    std::int64_t shexp_gate_ne0, shexp_gate_ne1, shexp_gate_bytes;
    std::int64_t shexp_up_ne0, shexp_up_ne1, shexp_up_bytes;
    std::int64_t shexp_down_ne0, shexp_down_ne1, shexp_down_bytes;

    // --- int32 scalars ---
    std::int32_t struct_bytes;
    std::int32_t is_recurrent;
    std::int32_t is_moe;
    std::int32_t qkv_type, o_type;
    std::int32_t gdn_qkv_type, gdn_gate_type, ssm_beta_type, ssm_alpha_type, ssm_out_type;
    std::int32_t gu_type, down_type;
    std::int32_t ff_dense;
    std::int32_t separate_qkv, k_type, v_type;
    std::int32_t gate_inp_type, gate_exps_type, up_exps_type, down_exps_type;
    std::int32_t shexp_gate_type, shexp_up_type, shexp_down_type;
};

namespace
{
    // Persistent decode-graph cache (single entry). When TS_QWEN35_FD_PERSIST is
    // set, the whole-model decode graph is built ONCE with stable tensor addresses
    // (raw ggml ctx + ggml_backend_alloc_ctx_tensors) and reused across tokens, so
    // ggml-cuda's CUDA-graph capture engages (key = cgraph->nodes[0]; replays one
    // captured graph instead of re-launching ~2000 kernels/token, which on WDDM the
    // GPU is starved waiting for — measured ~35% util / 21 W without it). The KV
    // write uses ggml_set_rows (position is an I64 input, not a baked view offset)
    // and attention uses a padded window + F16 mask input, so the graph topology is
    // identical token-to-token within a 256-token stride. Dropped + rebuilt when the
    // window grows a stride or the GDN state re-seeds (TSGgml_Qwen35ResetDecodeCache).
    struct Q35DecodeCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_t = nullptr;
        ggml_tensor* hidden_out = nullptr;
        ggml_tensor* pos_tensor = nullptr;
        ggml_tensor* kv_index = nullptr;   // I64 [1] = write position (shared, all attn layers)
        ggml_tensor* attn_mask = nullptr;  // F16 [window] causal padding mask (shared)
        const void* sig_disc = nullptr;    // model-instance discriminator
        const void* sig_kcache0 = nullptr; // first attention layer's KV ptr (per-holder identity)
        int num_layers = 0;
        int hidden_size = 0;
        int window = 0;
        bool folded = false;               // hidden_out holds logits (final norm + lm_head folded in)
        int out_count = 0;                 // element count of hidden_out (vocab when folded, else hidden)

        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_t = hidden_out = pos_tensor = kv_index = attn_mask = nullptr;
            sig_disc = sig_kcache0 = nullptr; num_layers = hidden_size = window = 0;
            folded = false; out_count = 0;
        }
    };

    // Concurrent (N>=2) Qwen3.5 requests each decode through their OWN per-request
    // KV + GDN state holder (Qwen35Model.BindSequenceCache swaps _kvCacheK /
    // _deltaStateTensor / _fdConvScratch), so a single shared decode-graph entry —
    // whose captured graph bakes those device addresses — would be busted on EVERY
    // request switch and rebuilt from scratch, collapsing aggregate throughput AND
    // (worse) replaying the previous request's baked addresses if the cheap reuse
    // key didn't include the holder identity (wrong output). Keep a small pool keyed
    // by (sig_disc, sig_kcache0 = first attention layer's KV ptr) so each in-flight
    // request retains its own persistent, CUDA-graph-captured decode graph. Exact
    // analogue of the dense Gemma4 g_g4dc_pool (see its comment). ResetDecodeCache
    // drops them all.
    constexpr int kQ35MaxDecodeCaches = 8;
    struct Q35DecodeCachePool
    {
        Q35DecodeCache entries[kQ35MaxDecodeCaches];
        std::uint64_t used[kQ35MaxDecodeCaches] = {};   // LRU clock per slot
        std::uint64_t clock = 0;

        Q35DecodeCache* find(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kQ35MaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                { used[i] = ++clock; return &entries[i]; }
            return nullptr;
        }

        Q35DecodeCache& claim(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kQ35MaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            for (int i = 0; i < kQ35MaxDecodeCaches; i++)
                if (!entries[i].valid) { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            int lru = 0;
            for (int i = 1; i < kQ35MaxDecodeCaches; i++) if (used[i] < used[lru]) lru = i;
            entries[lru].reset(); used[lru] = ++clock; return entries[lru];
        }

        void drop(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kQ35MaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                    entries[i].reset();
        }

        void reset_all() { for (auto& e : entries) e.reset(); }
    };
    Q35DecodeCachePool g_q35dc_pool;

    int qwen35_model_decode_impl(
        const TSGgmlQwen35LayerDesc* layers, int num_layers,
        void* hidden_data, int hidden_size, int position,
        int num_heads, int num_kv_heads, int head_dim, int cache_size,
        int rope_n_dims, int rope_mode, int kv_cache_type,
        int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
        float eps, float rope_base, float rope_freq_scale,
        int num_experts, int num_experts_used, int expert_ff, int shared_ff,
        int norm_topk, float expert_weights_scale,
        void* logits_data, int vocab_size,
        const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
        const void* final_norm_data)
    {
        if (!ensure_backend())
            return 0;
        if (layers == nullptr || num_layers <= 0 || hidden_data == nullptr)
        {
            set_last_error("Qwen3.5 model decode: invalid arguments.");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwen35LayerDesc)))
        {
            set_last_error("Qwen3.5 model decode: descriptor size mismatch (C# " +
                std::to_string(layers[0].struct_bytes) + " vs native " +
                std::to_string(sizeof(TSGgmlQwen35LayerDesc)) + ").");
            return 0;
        }

        const int H = hidden_size;
        const int totalSeqLen = position + 1;
        const int qDim = num_heads * head_dim;
        const int qFullDim = qDim * 2;            // Q + gate interleaved per head
        const int kDim = num_kv_heads * head_dim;
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int convDim = conv_kernel - 1;
        const int key_dim = head_k_dim * num_k_heads;
        const int value_dim = head_v_dim * num_v_heads;
        const int conv_dim = 2 * key_dim + value_dim;
        const int head_tile = (num_k_heads > 0) ? (num_v_heads / num_k_heads) : 1;

        static const bool fd_timing = std::getenv("TS_QWEN35_FD_TIMING") != nullptr;
        // Persistent capturable decode graph: default ON; TS_QWEN35_FD_PERSIST=0 disables.
        // Persist mode uses ggml_set_rows (KV write) + a CUDA-graph-captured replay,
        // both of which are CUDA-only — set_rows SEGFAULTs in ggml_metal_op_set_rows
        // and there is no Metal graph capture. So gate persist to CUDA; Metal runs
        // the whole-model decode through the NON-persist path (cpy KV write + reused
        // gallocr, every op — incl. ggml_gated_delta_net / ssm_conv / top_k — has a
        // Metal kernel in ggml v0.15.3). Exact analogue of the dense Gemma4 decode's
        // `can_persist = g4_persist && g_backend_type == BACKEND_TYPE_CUDA`.
        static const bool persist_cfg = []{ const char* e = std::getenv("TS_QWEN35_FD_PERSIST"); return e == nullptr || e[0] != '0'; }();
        const bool persist = persist_cfg && g_backend_type == BACKEND_TYPE_CUDA;
        // Persist mode pads the attention window to a fixed stride so the graph is
        // identical token-to-token (CUDA-graph capture); the F16 mask zeroes valid
        // positions and -inf's the padding. Non-persist keeps the exact window.
        constexpr int kPersistKvStride = 256;
        const int attnKvLen = persist
            ? std::min(cache_size, ((totalSeqLen + kPersistKvStride - 1) / kPersistKvStride) * kPersistKvStride)
            : flash_attn_kv_length(totalSeqLen, cache_size, head_dim);
        const bool use_persist_mask = persist;
        const void* sig_disc = layers[0].attn_norm_w;
        // Per-holder identity: first attention layer's KV cache device ptr. With
        // per-request fused-decode holders (Qwen35Model.BindSequenceCache) this is
        // distinct per concurrent request, so each retains its own captured graph
        // in g_q35dc_pool instead of busting/rebuilding (or, worse, replaying the
        // other request's baked addresses) on every switch.
        const void* sig_kcache0 = nullptr;
        for (int l = 0; l < num_layers; l++)
            if (!layers[l].is_recurrent && layers[l].k_cache != nullptr) { sig_kcache0 = layers[l].k_cache; break; }
        // Fold final-norm + lm_head into the graph so the whole token (incl. the
        // 248K-vocab logits) is one captured replay -> no separate lm_head submit.
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;

        auto t_start = std::chrono::high_resolution_clock::now();

        // ===== Persist reuse fast-path: replay THIS request's captured graph =====
        // find() already matched (sig_disc, sig_kcache0); check the finer shape here.
        Q35DecodeCache* dc = persist ? g_q35dc_pool.find(sig_disc, sig_kcache0) : nullptr;
        if (dc != nullptr && dc->graph != nullptr &&
            dc->num_layers == num_layers && dc->hidden_size == H &&
            dc->window == attnKvLen)
        {
            host_read_barrier();
            ggml_backend_tensor_set(dc->hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * sizeof(float));
            std::int32_t pos_val = position;
            ggml_backend_tensor_set(dc->pos_tensor, &pos_val, 0, sizeof(std::int32_t));
            std::int64_t kv_idx = position;
            ggml_backend_tensor_set(dc->kv_index, &kv_idx, 0, sizeof(std::int64_t));
            std::vector<ggml_fp16_t> mask_data;
            fill_flash_attn_mask(mask_data, attnKvLen, totalSeqLen);
            ggml_backend_tensor_set(dc->attn_mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
            auto t_setup = std::chrono::high_resolution_clock::now();
            ggml_status st = ggml_backend_graph_compute(g_backend, dc->graph);
            if (st != GGML_STATUS_SUCCESS)
            {
                set_last_error("Qwen3.5 model decode: cached graph execution failed.");
                dc->reset();
                return 0;
            }
            auto t_compute = std::chrono::high_resolution_clock::now();
            void* out_data = dc->folded ? logits_data : hidden_data;
            finalize_compute_with_download(dc->hidden_out, out_data, static_cast<std::size_t>(dc->out_count) * sizeof(float));
            host_read_barrier();
            if (fd_timing)
            {
                auto t_end = std::chrono::high_resolution_clock::now();
                auto ms = [](auto a, auto b){ return std::chrono::duration<double, std::milli>(b - a).count(); };
                fprintf(stderr, "[fd-timing] REUSE setup=%.2f compute=%.2f download=%.2f total=%.2f ms\n",
                    ms(t_start, t_setup), ms(t_setup, t_compute), ms(t_compute, t_end), ms(t_start, t_end));
                fflush(stderr);
            }
            return 1;
        }
        // Miss -> (re)build into this request's slot (reset in place / evict LRU).
        Q35DecodeCache* dcb = persist ? &g_q35dc_pool.claim(sig_disc, sig_kcache0) : nullptr;

        // no_alloc ctx: tensor metadata only. Non-persist uses the pooled 32 MB
        // block; persist uses a raw ctx kept alive in g_q35dc for graph reuse.
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr)
            {
                set_last_error("Qwen3.5 model decode: failed to init persist ggml context.");
                return 0;
            }
        }
        else
        {
            if (!context.init(ctx_size))
            {
                set_last_error("Qwen3.5 model decode: failed to acquire ggml context.");
                return 0;
            }
            ctx = context.value;
        }

        ggml_tensor* hidden_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* lm_head_t = fold ? ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1) : nullptr;
        ggml_tensor* final_norm_t = fold ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
        // Shared per-token inputs for the static (capturable) graph.
        ggml_tensor* shared_kv_index = use_persist_mask ? ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1) : nullptr;
        ggml_tensor* shared_attn_mask = use_persist_mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1) : nullptr;
        if (use_persist_mask)
        {
            ggml_set_input(hidden_t);
            ggml_set_input(pos_tensor);
            ggml_set_input(shared_kv_index);
            ggml_set_input(shared_attn_mask);
        }

        struct LayerTensors {
            // attention
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* k_w;
            ggml_tensor* v_w;
            ggml_tensor* q_norm_w;
            ggml_tensor* k_norm_w;
            ggml_tensor* o_w;
            ggml_tensor* k_cache_base;
            ggml_tensor* v_cache_base;
            ggml_tensor* attn_mask;
            ggml_tensor* k_cpy;
            ggml_tensor* v_cpy;
            std::vector<ggml_fp16_t> attn_mask_data;
            // gdn
            ggml_tensor* gdn_qkv_w;
            ggml_tensor* gdn_gate_w;
            ggml_tensor* ssm_beta_w;
            ggml_tensor* ssm_alpha_w;
            ggml_tensor* conv1d_w;
            ggml_tensor* ssm_dt_w;
            ggml_tensor* ssm_a_w;
            ggml_tensor* ssm_norm_w;
            ggml_tensor* ssm_out_w;
            ggml_tensor* conv_state_in;
            ggml_tensor* delta_state_in;
            ggml_tensor* conv_state_out;
            ggml_tensor* delta_state_out;
            // ffn (dense)
            ggml_tensor* post_attn_norm_w;
            ggml_tensor* gu_w;
            ggml_tensor* down_w;
            // ffn (MoE)
            ggml_tensor* gate_inp_w;
            ggml_tensor* gate_exps;
            ggml_tensor* up_exps;
            ggml_tensor* down_exps;
            ggml_tensor* shexp_gate_w;
            ggml_tensor* shexp_up_w;
            ggml_tensor* shexp_down_w;
            ggml_tensor* shexp_gate_inp_w;
        };
        std::vector<LayerTensors> lt(num_layers);

        // --- create per-layer tensors ---
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            if (d.is_recurrent == 0)
            {
                t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
                if (d.separate_qkv != 0)
                {
                    t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
                    t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
                }
                t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
                t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
                t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
                t.k_cache_base = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, cache_size, num_kv_heads);
                t.v_cache_base = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, cache_size, num_kv_heads);
            }
            else
            {
                t.gdn_qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_qkv_type), d.gdn_qkv_ne0, d.gdn_qkv_ne1);
                t.gdn_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_gate_type), d.gdn_gate_ne0, d.gdn_gate_ne1);
                t.ssm_beta_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_beta_type), d.ssm_beta_ne0, d.ssm_beta_ne1);
                t.ssm_alpha_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_alpha_type), d.ssm_alpha_ne0, d.ssm_alpha_ne1);
                t.conv1d_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_kernel, conv_dim);
                t.ssm_dt_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_a_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_v_dim);
                t.ssm_out_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_out_type), d.ssm_out_ne0, d.ssm_out_ne1);
                t.conv_state_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, convDim, conv_dim);
                t.delta_state_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads);
            }
            // FFN
            if (d.is_moe == 0)
            {
                t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
                t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            }
            else
            {
                t.gate_inp_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gate_inp_type), d.gate_inp_ne0, d.gate_inp_ne1);
                t.gate_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gate_exps_type), hidden_size, expert_ff, num_experts);
                t.up_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.up_exps_type), hidden_size, expert_ff, num_experts);
                t.down_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.down_exps_type), expert_ff, hidden_size, num_experts);
                t.shexp_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_gate_type), d.shexp_gate_ne0, d.shexp_gate_ne1);
                t.shexp_up_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_up_type), d.shexp_up_ne0, d.shexp_up_ne1);
                t.shexp_down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_down_type), d.shexp_down_ne0, d.shexp_down_ne1);
                t.shexp_gate_inp_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            }
        }

        // --- build the chained graph ---
        ggml_tensor* hidden = hidden_t;
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];

            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w);
            ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, H, 1);
            ggml_tensor* block_out; // the attention / gdn output added to residual

            if (d.is_recurrent == 0)
            {
                // ===== Full attention =====
                ggml_tensor* qg_part;
                ggml_tensor* k_raw;
                ggml_tensor* v_raw;
                if (d.separate_qkv != 0)
                {
                    qg_part = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.qkv_w, normed_2d), qFullDim);
                    k_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.k_w, normed_2d), kDim);
                    v_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.v_w, normed_2d), kDim);
                }
                else
                {
                    ggml_tensor* qkv_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.qkv_w, normed_2d), qFullDim + 2 * kDim);
                    qg_part = ggml_view_1d(ctx, qkv_flat, qFullDim, 0);
                    k_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qFullDim) * sizeof(float));
                    v_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qFullDim + kDim) * sizeof(float));
                }

                ggml_tensor* qg_3d = ggml_reshape_3d(ctx, qg_part, head_dim, 2, num_heads);
                ggml_tensor* q_view = ggml_view_2d(ctx, qg_3d, head_dim, num_heads, qg_3d->nb[2], 0);
                ggml_tensor* gate_view = ggml_view_2d(ctx, qg_3d, head_dim, num_heads, qg_3d->nb[2], qg_3d->nb[1]);

                ggml_tensor* q_2d_raw = ggml_cont(ctx, q_view);
                ggml_tensor* k_2d_raw = ggml_reshape_2d(ctx, k_raw, head_dim, num_kv_heads);
                ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d_raw, eps), t.q_norm_w);
                ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d_raw, eps), t.k_norm_w);

                ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, head_dim, num_heads, 1);
                ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, head_dim, num_kv_heads, 1);
                ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, nullptr, rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
                ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, nullptr, rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);

                ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
                ggml_tensor* k_rope_perm = ggml_permute(ctx, k_rope, 0, 2, 1, 3);
                ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_raw, head_dim, num_kv_heads, 1);
                ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
                ggml_tensor* k_write = ggml_cont(ctx, k_rope_perm);
                ggml_tensor* v_write = ggml_cont(ctx, v_perm);
                ggml_tensor* mask_for_attn;
                if (persist)
                {
                    // KV write via set_rows: the write position is an I64 INPUT
                    // (shared_kv_index), not a baked view offset, so the graph
                    // topology is identical token-to-token (CUDA-graph capturable).
                    t.k_cpy = ggml_set_rows(ctx, t.k_cache_base, k_write, shared_kv_index);
                    t.v_cpy = ggml_set_rows(ctx, t.v_cache_base, v_write, shared_kv_index);
                    mask_for_attn = shared_attn_mask;
                }
                else
                {
                    const std::size_t kv_byte_offset = static_cast<std::size_t>(position) * t.k_cache_base->nb[1];
                    ggml_tensor* k_dst = ggml_view_3d(ctx, t.k_cache_base, head_dim, 1, num_kv_heads, t.k_cache_base->nb[1], t.k_cache_base->nb[2], kv_byte_offset);
                    ggml_tensor* v_dst = ggml_view_3d(ctx, t.v_cache_base, head_dim, 1, num_kv_heads, t.v_cache_base->nb[1], t.v_cache_base->nb[2], kv_byte_offset);
                    t.k_cpy = ggml_cpy(ctx, k_write, k_dst);
                    t.v_cpy = ggml_cpy(ctx, v_write, v_dst);
                    if (flash_attn_requires_masked_padding(head_dim))
                    {
                        t.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
                        fill_flash_attn_mask(t.attn_mask_data, attnKvLen, totalSeqLen);
                    }
                    mask_for_attn = t.attn_mask;
                }
                ggml_tensor* k_full = view_kv_cache_window(ctx, t.k_cache_base, head_dim, cache_size, num_kv_heads, 0, attnKvLen, kv_cache_type);
                ggml_tensor* v_full = view_kv_cache_window(ctx, t.v_cache_base, head_dim, cache_size, num_kv_heads, 0, attnKvLen, kv_cache_type);
                if (k_full == nullptr || v_full == nullptr)
                {
                    set_last_error("Qwen3.5 model decode: failed to build KV cache views.");
                    if (persist) ggml_free(ctx);
                    return 0;
                }
                ggml_tensor* attn_out_4d = ggml_flash_attn_ext(ctx, q_attn, k_full, v_full, mask_for_attn, attn_scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(attn_out_4d, GGML_PREC_F32);
                ggml_tensor* attn_out_2d = ggml_reshape_2d(ctx, attn_out_4d, head_dim, num_heads);
                ggml_tensor* gate_2d = ggml_cont(ctx, gate_view);
                ggml_tensor* attn_gated = ggml_mul(ctx, attn_out_2d, ggml_sigmoid(ctx, gate_2d));
                ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_gated, qDim, 1);
                block_out = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.o_w, attn_flat), H);
            }
            else
            {
                // ===== Gated Delta Net (linear attention) =====
                ggml_tensor* qkv_mixed = ggml_mul_mat(ctx, t.gdn_qkv_w, normed_2d);          // [conv_dim, 1]
                ggml_tensor* z = ggml_mul_mat(ctx, t.gdn_gate_w, normed_2d);                 // [value_dim, 1]
                ggml_tensor* beta_raw = ggml_mul_mat(ctx, t.ssm_beta_w, normed_2d);          // [num_v_heads, 1]
                ggml_tensor* alpha_raw = ggml_mul_mat(ctx, t.ssm_alpha_w, normed_2d);        // [num_v_heads, 1]

                ggml_tensor* beta = ggml_sigmoid(ctx, beta_raw);
                beta = ggml_reshape_4d(ctx, beta, 1, num_v_heads, 1, 1);

                ggml_tensor* alpha_1d = ggml_reshape_1d(ctx, alpha_raw, num_v_heads);
                ggml_tensor* g = ggml_softplus(ctx, ggml_add(ctx, alpha_1d, t.ssm_dt_w));    // softplus(alpha + dt)
                g = ggml_mul(ctx, g, t.ssm_a_w);                                             // * (-exp(A_log))
                g = ggml_reshape_4d(ctx, g, 1, num_v_heads, 1, 1);

                // conv over the host ring state + the new mixed input
                ggml_tensor* qkv_T = ggml_cont(ctx, ggml_transpose(ctx, qkv_mixed));          // [1, conv_dim]
                ggml_tensor* conv_input = ggml_concat(ctx, t.conv_state_in, qkv_T, 0);        // [convDim+1, conv_dim]
                ggml_tensor* conv_out = ggml_ssm_conv(ctx, conv_input, t.conv1d_w);           // [conv_dim, 1]
                conv_out = ggml_silu(ctx, conv_out);
                ggml_tensor* conv_out_1d = ggml_reshape_1d(ctx, conv_out, conv_dim);

                // new conv state = the most recent convDim time-steps of conv_input,
                // written in-place back to the device-resident conv-state buffer.
                ggml_tensor* new_conv = ggml_cont(ctx, ggml_view_2d(ctx, conv_input, convDim, conv_dim,
                    conv_input->nb[1], static_cast<std::size_t>(1) * conv_input->nb[0]));
                t.conv_state_out = ggml_cpy(ctx, new_conv, t.conv_state_in);

                // split q/k/v
                ggml_tensor* q_c = ggml_cont(ctx, ggml_view_2d(ctx, conv_out_1d, head_k_dim, num_k_heads,
                    static_cast<std::size_t>(head_k_dim) * sizeof(float), 0));
                ggml_tensor* k_c = ggml_cont(ctx, ggml_view_2d(ctx, conv_out_1d, head_k_dim, num_k_heads,
                    static_cast<std::size_t>(head_k_dim) * sizeof(float), static_cast<std::size_t>(key_dim) * sizeof(float)));
                ggml_tensor* v_c = ggml_cont(ctx, ggml_view_2d(ctx, conv_out_1d, head_v_dim, num_v_heads,
                    static_cast<std::size_t>(head_v_dim) * sizeof(float), static_cast<std::size_t>(2 * key_dim) * sizeof(float)));

                q_c = ggml_l2_norm(ctx, q_c, eps);
                k_c = ggml_l2_norm(ctx, k_c, eps);

                // tile q/k from num_k_heads to num_v_heads (h -> h % num_k_heads)
                ggml_tensor* q_t = q_c;
                ggml_tensor* k_t = k_c;
                for (int r = 1; r < head_tile; r++)
                {
                    q_t = ggml_concat(ctx, q_t, q_c, 1);
                    k_t = ggml_concat(ctx, k_t, k_c, 1);
                }

                ggml_tensor* q4 = ggml_reshape_4d(ctx, ggml_cont(ctx, q_t), head_k_dim, num_v_heads, 1, 1);
                ggml_tensor* k4 = ggml_reshape_4d(ctx, ggml_cont(ctx, k_t), head_k_dim, num_v_heads, 1, 1);
                ggml_tensor* v4 = ggml_reshape_4d(ctx, v_c, head_v_dim, num_v_heads, 1, 1);
                ggml_tensor* state4 = ggml_reshape_4d(ctx, t.delta_state_in, head_k_dim, head_v_dim, num_v_heads, 1);

                ggml_tensor* gdn = ggml_gated_delta_net(ctx, q4, k4, v4, g, beta, state4, 1);
                ggml_tensor* gdn_out = ggml_view_4d(ctx, gdn, head_v_dim, num_v_heads, 1, 1,
                    ggml_row_size(gdn->type, head_v_dim),
                    ggml_row_size(gdn->type, head_v_dim * num_v_heads),
                    ggml_row_size(gdn->type, head_v_dim * num_v_heads), 0);
                ggml_tensor* new_state = ggml_view_4d(ctx, gdn, head_k_dim, head_v_dim, num_v_heads, 1,
                    ggml_row_size(gdn->type, head_k_dim),
                    ggml_row_size(gdn->type, head_k_dim * head_v_dim),
                    ggml_row_size(gdn->type, head_k_dim * head_v_dim * num_v_heads),
                    ggml_row_size(gdn->type, head_v_dim * num_v_heads));
                // write the new recurrent state in-place back to the device-resident
                // delta-state buffer (state4 aliases delta_state_in).
                t.delta_state_out = ggml_cpy(ctx, new_state, state4);

                // gated RMSNorm: rms_norm(core, ssm_norm) * silu(z)
                ggml_tensor* out_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, gdn_out), head_v_dim, num_v_heads);
                ggml_tensor* out_n = ggml_mul(ctx, ggml_rms_norm(ctx, out_2d, eps), t.ssm_norm_w);
                ggml_tensor* z_2d = ggml_reshape_2d(ctx, z, head_v_dim, num_v_heads);
                ggml_tensor* gated = ggml_mul(ctx, out_n, ggml_silu(ctx, z_2d));
                ggml_tensor* gated_flat = ggml_reshape_2d(ctx, gated, value_dim, 1);
                block_out = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.ssm_out_w, gated_flat), H);
            }

            ggml_tensor* residual1 = ggml_add(ctx, hidden, block_out);

            // ===== FFN =====
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.post_attn_norm_w);
            ggml_tensor* ffn_normed_2d = ggml_reshape_2d(ctx, ffn_normed, H, 1);
            ggml_tensor* ffn_down;
            if (d.is_moe == 0)
            {
                // dense SwiGLU
                const std::int64_t ffDense = d.ff_dense;
                ggml_tensor* gu_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.gu_w, ffn_normed_2d), 2 * ffDense);
                ggml_tensor* g_part = ggml_view_1d(ctx, gu_flat, ffDense, 0);
                ggml_tensor* u_part = ggml_view_1d(ctx, gu_flat, ffDense, static_cast<std::size_t>(ffDense) * sizeof(float));
                ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, ggml_cont(ctx, g_part)), u_part);
                ggml_tensor* act_2d = ggml_reshape_2d(ctx, act, ffDense, 1);
                ffn_down = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.down_w, act_2d), H);
            }
            else
            {
                // ----- MoE: router -> top-k -> renorm -> stacked experts + gated shared expert -----
                ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, ffn_normed_2d); // [num_experts, 1]
                ggml_tensor* probs = ggml_soft_max(ctx, router_logits);
                ggml_tensor* sel = ggml_top_k(ctx, probs, num_experts_used);                 // [num_used, 1]
                ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, num_experts, 1);
                ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);                            // [1, num_used, 1]
                ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, num_experts_used, 1);
                if (norm_topk != 0)
                {
                    ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);
                    w_2d = ggml_div(ctx, w_2d, w_sum);
                }
                if (expert_weights_scale != 1.0f)
                    w_2d = ggml_scale(ctx, w_2d, expert_weights_scale);
                ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, num_experts_used, 1);

                ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, ffn_normed, H, 1, 1);
                ggml_tensor* g_exp = ggml_mul_mat_id(ctx, t.gate_exps, moe_in_3d, sel);       // [expert_ff, num_used, 1]
                ggml_tensor* u_exp = ggml_mul_mat_id(ctx, t.up_exps, moe_in_3d, sel);
                ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, g_exp), u_exp);               // [expert_ff, num_used, 1]
                ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps, act, sel);          // [hidden, num_used, 1]
                ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);

                ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, 1, weighted->nb[2], 0);
                for (int u = 1; u < num_experts_used; ++u)
                {
                    ggml_tensor* vu = ggml_view_2d(ctx, weighted, H, 1, weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                    moe_out = ggml_add(ctx, moe_out, vu);
                }
                ggml_tensor* moe_out_1d = ggml_reshape_1d(ctx, moe_out, H);

                // gated shared expert
                ggml_tensor* sh_g = ggml_mul_mat(ctx, t.shexp_gate_w, ffn_normed_2d);         // [shared_ff, 1]
                ggml_tensor* sh_u = ggml_mul_mat(ctx, t.shexp_up_w, ffn_normed_2d);
                ggml_tensor* sh_act = ggml_mul(ctx, ggml_silu(ctx, sh_g), sh_u);
                ggml_tensor* sh_down = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.shexp_down_w, sh_act), H);
                ggml_tensor* sh_gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, t.shexp_gate_inp_w, H, 1), ffn_normed_2d)); // [1,1]
                ggml_tensor* sh_out = ggml_mul(ctx, sh_down, sh_gate);

                ffn_down = ggml_add(ctx, moe_out_1d, sh_out);
            }

            hidden = ggml_add(ctx, residual1, ffn_down);
        }

        ggml_tensor* hidden_out;
        ggml_tensor* out_cpy;
        if (fold)
        {
            // Final RMSNorm * weight, then lm_head -> logits, folded into the graph
            // so the lm_head matmul + the 248K-vocab output are part of the captured
            // replay (no separate per-token lm_head graph_compute / submit).
            ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);
            ggml_tensor* fn_2d = ggml_reshape_2d(ctx, fn, H, 1);
            ggml_tensor* logits = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lm_head_t, fn_2d), vocab_size);
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab_size);
            out_cpy = ggml_cpy(ctx, logits, hidden_out);
        }
        else
        {
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            out_cpy = ggml_cpy(ctx, hidden, hidden_out);
        }
        ggml_set_output(out_cpy);

        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * 160 + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].is_recurrent == 0)
            {
                ggml_build_forward_expand(graph, lt[l].k_cpy);
                ggml_build_forward_expand(graph, lt[l].v_cpy);
            }
            else
            {
                ggml_set_output(lt[l].conv_state_out);
                ggml_set_output(lt[l].delta_state_out);
                ggml_build_forward_expand(graph, lt[l].conv_state_out);
                ggml_build_forward_expand(graph, lt[l].delta_state_out);
            }
        }
        ggml_build_forward_expand(graph, out_cpy);

        // --- bind tensors ---
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* tgt, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (tgt == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, tgt, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, tgt, addr) == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload) upload_list.push_back({tgt, data, bytes});
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
                    if (!cacheable) ephemeral_bufs.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        const std::size_t convStateBytes = static_cast<std::size_t>(convDim) * conv_dim * sizeof(float);
        const std::size_t deltaStateBytes = static_cast<std::size_t>(head_k_dim) * head_v_dim * num_v_heads * sizeof(float);
        int state_uploads = 0;
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            if (d.is_moe == 0)
            {
                bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes), true);
                bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes), true);
            }
            else
            {
                bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(d.gate_inp_bytes), true);
                bind_or_mark(t.gate_exps, d.gate_exps, static_cast<std::size_t>(d.gate_exps_bytes), true);
                bind_or_mark(t.up_exps, d.up_exps, static_cast<std::size_t>(d.up_exps_bytes), true);
                bind_or_mark(t.down_exps, d.down_exps, static_cast<std::size_t>(d.down_exps_bytes), true);
                bind_or_mark(t.shexp_gate_w, d.shexp_gate_w, static_cast<std::size_t>(d.shexp_gate_bytes), true);
                bind_or_mark(t.shexp_up_w, d.shexp_up_w, static_cast<std::size_t>(d.shexp_up_bytes), true);
                bind_or_mark(t.shexp_down_w, d.shexp_down_w, static_cast<std::size_t>(d.shexp_down_bytes), true);
                bind_or_mark(t.shexp_gate_inp_w, d.shexp_gate_inp_w, static_cast<std::size_t>(H) * sizeof(float), true);
            }
            if (d.is_recurrent == 0)
            {
                bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
                if (d.separate_qkv != 0)
                {
                    bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true);
                    bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true);
                }
                bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
                bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(head_dim) * sizeof(float), true);
                bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(head_dim) * sizeof(float), true);
                bind_or_mark(t.k_cache_base, d.k_cache, kv_cache_bytes(num_kv_heads, cache_size, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.v_cache_base, d.v_cache, kv_cache_bytes(num_kv_heads, cache_size, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (t.attn_mask != nullptr && !t.attn_mask_data.empty())
                    bind_or_mark(t.attn_mask, t.attn_mask_data.data(), t.attn_mask_data.size() * sizeof(ggml_fp16_t), false);
            }
            else
            {
                bind_or_mark(t.gdn_qkv_w, d.gdn_qkv_w, static_cast<std::size_t>(d.gdn_qkv_bytes), true);
                bind_or_mark(t.gdn_gate_w, d.gdn_gate_w, static_cast<std::size_t>(d.gdn_gate_bytes), true);
                bind_or_mark(t.ssm_beta_w, d.ssm_beta_w, static_cast<std::size_t>(d.ssm_beta_bytes), true);
                bind_or_mark(t.ssm_alpha_w, d.ssm_alpha_w, static_cast<std::size_t>(d.ssm_alpha_bytes), true);
                bind_or_mark(t.conv1d_w, d.conv1d_w, static_cast<std::size_t>(conv_kernel) * conv_dim * sizeof(float), true);
                bind_or_mark(t.ssm_dt_w, d.ssm_dt_w, static_cast<std::size_t>(num_v_heads) * sizeof(float), true);
                bind_or_mark(t.ssm_a_w, d.ssm_a_w, static_cast<std::size_t>(num_v_heads) * sizeof(float), true);
                bind_or_mark(t.ssm_norm_w, d.ssm_norm_w, static_cast<std::size_t>(head_v_dim) * sizeof(float), true);
                bind_or_mark(t.ssm_out_w, d.ssm_out_w, static_cast<std::size_t>(d.ssm_out_bytes), true);
                // GDN recurrent state is device-resident across decode tokens
                // (cacheable COMPUTE buffer, updated in-place each token); the C#
                // seed path invalidates these on (re)seed so the host state is
                // re-uploaded. Mirrors the KV-cache binding.
                size_t ul_before = upload_list.size();
                bind_or_mark(t.conv_state_in, d.conv_state_in, convStateBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.delta_state_in, d.delta_state_in, deltaStateBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                state_uploads += (int)(upload_list.size() - ul_before);
            }
        }
        if (fold)
        {
            bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
            bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(H) * sizeof(float), true);
        }

        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (persist)
        {
            // STABLE addresses for CUDA-graph capture: give every tensor its own
            // slot (no gallocr lifetime packing, whose plan can move addresses).
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("Qwen3.5 model decode: failed to allocate persist backend buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else
        {
            // Non-persist (Metal): the whole-model GDN decode graph must NOT use the
            // shared reuse gallocr. Its lifetime-packing mis-aliases this graph's
            // intermediates across the gated_delta_net + in-place recurrent-state
            // ggml_cpy chain, so the packed scratch retains the PREVIOUS token's
            // activations and they leak into the current token — producing coherent
            // but interleaved-garbage output (two tokens' continuations merged word
            // by word). A fresh per-call backend buffer fixes it; the decode graph's
            // scratch is small so the alloc costs only ~4 ms/token (still ~5x faster
            // than the op-by-op path). The dense Gemma4 decode is unaffected (it has
            // no in-place recurrent state) and keeps the reuse gallocr.
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("Qwen3.5 model decode: failed to allocate backend buffer.");
                return 0;
            }
        }

        host_read_barrier();

        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);

        ggml_backend_tensor_set(hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * sizeof(float));
        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));
        if (persist)
        {
            std::int64_t kv_idx = position;
            ggml_backend_tensor_set(shared_kv_index, &kv_idx, 0, sizeof(std::int64_t));
            std::vector<ggml_fp16_t> mask_data;
            fill_flash_attn_mask(mask_data, attnKvLen, totalSeqLen);
            ggml_backend_tensor_set(shared_attn_mask, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
        }

        auto t_setup = std::chrono::high_resolution_clock::now();

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Qwen3.5 model decode: graph execution failed.");
            if (persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
            return 0;
        }
        auto t_compute = std::chrono::high_resolution_clock::now();

        // GDN state is device-resident (updated in-place). Download either the
        // folded logits or the bare hidden state.
        const int out_count = fold ? vocab_size : H;
        void* out_data = fold ? logits_data : hidden_data;
        finalize_compute_with_download(hidden_out, out_data, static_cast<std::size_t>(out_count) * sizeof(float));
        if (persist || buffer.value != nullptr) host_read_barrier();

        if (persist && dcb != nullptr)
        {
            // Keep the ctx/graph/buffer alive for capture+replay on later tokens.
            dcb->ctx = ctx;
            dcb->buffer = persist_buf;
            dcb->graph = graph;
            dcb->hidden_t = hidden_t;
            dcb->hidden_out = hidden_out;
            dcb->pos_tensor = pos_tensor;
            dcb->kv_index = shared_kv_index;
            dcb->attn_mask = shared_attn_mask;
            dcb->sig_disc = sig_disc;
            dcb->sig_kcache0 = sig_kcache0;
            dcb->num_layers = num_layers;
            dcb->hidden_size = H;
            dcb->window = attnKvLen;
            dcb->folded = fold;
            dcb->out_count = out_count;
            dcb->valid = true;
        }
        if (fd_timing)
        {
            auto t_end = std::chrono::high_resolution_clock::now();
            auto ms = [](auto a, auto b){ return std::chrono::duration<double, std::milli>(b - a).count(); };
            fprintf(stderr, "[fd-timing] %s setup=%.1f compute=%.1f download=%.1f total=%.1f ms (upload_list=%zu state_uploads=%d)\n",
                persist ? "BUILD" : "", ms(t_start, t_setup), ms(t_setup, t_compute), ms(t_compute, t_end), ms(t_start, t_end), upload_list.size(), state_uploads);
            fflush(stderr);
        }
        clear_last_error();
        return 1;
    }
}

TSG_EXPORT int TSGgml_Qwen35ModelDecode(
    const TSGgmlQwen35LayerDesc* layers, int num_layers,
    void* hidden_data, int hidden_size, int position,
    int num_heads, int num_kv_heads, int head_dim, int cache_size,
    int rope_n_dims, int rope_mode, int kv_cache_type,
    int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
    float eps, float rope_base, float rope_freq_scale,
    int num_experts, int num_experts_used, int expert_ff, int shared_ff,
    int norm_topk, float expert_weights_scale,
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data)
{
    try
    {
        int r = qwen35_model_decode_impl(
            layers, num_layers, hidden_data, hidden_size, position,
            num_heads, num_kv_heads, head_dim, cache_size,
            rope_n_dims, rope_mode, kv_cache_type,
            conv_kernel, head_k_dim, head_v_dim, num_k_heads, num_v_heads,
            eps, rope_base, rope_freq_scale,
            num_experts, num_experts_used, expert_ff, shared_ff,
            norm_topk, expert_weights_scale,
            logits_data, vocab_size,
            lm_head_data, lm_head_type, lm_head_ne0, lm_head_ne1, lm_head_bytes,
            final_norm_data);
        if (r == 0 && std::getenv("TS_QWEN35_FD_TIMING") != nullptr)
            fprintf(stderr, "[fd-err] %s\n", g_last_error.c_str());
        return r;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in Qwen3.5 model decode.");
        return 0;
    }
}

// Drop the persistent decode-graph cache. Called from C# whenever the GDN
// recurrent state is re-seeded / the per-op path runs (the cached graph pins the
// conv/delta device-buffer addresses, which move on re-seed), so the next fused
// decode rebuilds against the fresh state.
TSG_EXPORT void TSGgml_Qwen35ResetDecodeCache()
{
    g_q35dc_pool.reset_all();
}

// ============================================================================
// TSGgml_Qwen35ModelVerify  --  the N-token sibling of TSGgml_Qwen35ModelDecode.
//
// MTP speculative decoding verifies a window of (1 + draft) tokens in one trunk
// pass. The per-op SpecForward fallback runs that pass op-by-op (~1 s/step on
// WDDM ggml_cuda); this kernel runs the WHOLE hybrid transformer over N tokens as
// ONE ggml graph: prefill-style causal flash attention (read cache prefix [0,
// start_pos) + the N fresh K/V, causal F16 mask, append the N rows to the cache),
// GDN recurrence over the N tokens via ggml_gated_delta_net(K=N) (the op also
// emits a state snapshot per prefix length so partial-acceptance rollback can pick
// the committed state without a re-forward), batched MoE/dense FFN, and a folded
// final-norm that outputs BOTH the per-row logits [vocab, N] AND the post-norm
// hidden [hidden, N] the MTP draft head consumes.
//
// GDN state is passed per-layer via host in/out pointers (conv_state_in/out in
// ggml [time, channel] layout, delta_state_in/out as [S, S, H]); the kernel reads
// the pre-window state and writes the post-window state. Non-persistent (rebuilt
// per call) -- correctness first; CUDA-graph capture is a follow-up that needs a
// fixed-N persistent cache (cf. g_q35dc). Returns 0 on anything it cannot handle
// so the C# caller falls back to the per-op SpecForward.
// ============================================================================
namespace
{
    // Persistent verify-graph cache (multi-entry, keyed by N + window stride). With
    // TS_Q35_VERIFY_PERSIST (default ON) the per-(N,window) graph is built ONCE in a
    // raw ggml ctx with ggml_backend_alloc_ctx_tensors (each tensor its own slot =
    // stable addresses), so subsequent verify steps of the same shape only upload the
    // per-call inputs (hidden / pos / kv_index / mask / GDN state) + recompute -> the
    // ~150-580 ms C++ build is amortized AND ggml-cuda can CUDA-graph-capture the
    // replay. Multi-entry because spec alternates N=draft+1 (speculative) and
    // N=accepted+1 (rollback re-forward); a single entry would evict + rebuild every
    // call. Small GDN weights (dt_bias/ssm_a/...) get their own stable alloc_ctx slot
    // uploaded once, so they don't go stale on reuse (the gallocr-recycle bug).
    struct Q35VerifyCache
    {
        bool valid = false;
        int n = 0, window = 0, num_layers = 0, out_vocab = 0, n_logits = 0;
        bool has_normed = false;
        const void* sig = nullptr;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_t = nullptr;
        ggml_tensor* pos_t = nullptr;
        ggml_tensor* kv_index = nullptr;
        ggml_tensor* mask_t = nullptr;
        ggml_tensor* logits_out = nullptr;
        ggml_tensor* normed_out = nullptr;
        std::vector<ggml_tensor*> conv_in, delta_in, conv_out, delta_out;
        std::uint64_t lru = 0;
        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_t = pos_t = kv_index = mask_t = logits_out = normed_out = nullptr;
            conv_in.clear(); delta_in.clear(); conv_out.clear(); delta_out.clear();
            n = window = num_layers = out_vocab = n_logits = 0; has_normed = false; sig = nullptr;
        }
    };
    Q35VerifyCache g_q35vc[16];
    std::uint64_t g_q35vc_clock = 0;

    void fill_verify_causal_mask(std::vector<ggml_fp16_t>& mask, int window, int n, int start_pos, int total_len)
    {
        mask.resize(static_cast<std::size_t>(window) * n);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
        for (int qi = 0; qi < n; qi++)
        {
            int threshold = start_pos + qi;
            ggml_fp16_t* row = &mask[static_cast<std::size_t>(qi) * window];
            for (int ki = 0; ki < window; ki++)
                row[ki] = (ki <= threshold && ki < total_len) ? zero_val : neg_inf;
        }
    }

    int qwen35_model_verify_impl(
        const TSGgmlQwen35LayerDesc* layers, int num_layers,
        void* hidden_data, int hidden_size, int start_pos, int num_tokens,
        int num_heads, int num_kv_heads, int head_dim, int cache_size,
        int rope_n_dims, int rope_mode, int kv_cache_type,
        int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
        float eps, float rope_base, float rope_freq_scale,
        int num_experts, int num_experts_used, int expert_ff, int shared_ff,
        int norm_topk, float expert_weights_scale,
        void* logits_data, int vocab_size,
        const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
        const void* final_norm_data, void* normed_out, int n_logit_rows,
        const std::int32_t* mrope_pos, const std::int32_t* mrope_sections)
    {
        if (!ensure_backend())
            return 0;
        if (layers == nullptr || num_layers <= 0 || hidden_data == nullptr || num_tokens < 1)
        {
            set_last_error("Qwen3.5 model verify: invalid arguments.");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwen35LayerDesc)))
        {
            set_last_error("Qwen3.5 model verify: descriptor size mismatch.");
            return 0;
        }
        // gated_delta_net requires S_k == S_v (state is [S_v, S_v, H]).
        if (head_k_dim != head_v_dim)
        {
            set_last_error("Qwen3.5 model verify: head_k_dim != head_v_dim unsupported.");
            return 0;
        }

        const int N = num_tokens;
        const int H = hidden_size;
        const int totalSeqLen = start_pos + N;
        const int qDim = num_heads * head_dim;
        const int qFullDim = qDim * 2;            // Q + gate interleaved per head
        const int kDim = num_kv_heads * head_dim;
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int convDim = conv_kernel - 1;
        const int key_dim = head_k_dim * num_k_heads;
        const int value_dim = head_v_dim * num_v_heads;
        const int conv_dim = 2 * key_dim + value_dim;
        const int head_tile = (num_k_heads > 0) ? (num_v_heads / num_k_heads) : 1;
        const ggml_type kvType = static_cast<ggml_type>(kv_cache_type);
        if (convDim <= 0 || totalSeqLen > cache_size)
        {
            set_last_error("Qwen3.5 model verify: bad conv dim or sequence exceeds cache.");
            return 0;
        }
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;
        if (!fold)
        {
            set_last_error("Qwen3.5 model verify: folded lm_head required.");
            return 0;
        }

        // Prefill only needs the LAST token's logits (to sample the first decode
        // token); MTP verify needs all N rows. n_logit_rows in [1, N) computes the
        // lm_head only over the last n_logit_rows columns of the post-norm hidden,
        // so a 2048-token prefill writes vocab*1 floats (not vocab*2048 ~ 2 GB) and
        // skips the lm_head matmul over the first N-1 tokens. <=0 or >=N => all N.
        const int n_logits = (n_logit_rows > 0 && n_logit_rows < N) ? n_logit_rows : N;

        static const bool fv_timing = std::getenv("TS_Q35_VERIFY_TIMING") != nullptr;
        // Persistent per-(N,window) graph cache (build amortization + CUDA-graph
        // capture). DEFAULT ON: the earlier reuse access-violation (0xC0000005) was the
        // 3D N-row set_rows (heads in ne2) faulting on cgraph reuse; replacing it with a
        // 2D set_rows PER HEAD (llama.cpp's proven KV-write shape) made reuse stable
        // (validated 252 reuses, no crash). Reuse: setup ~8 ms + compute ~12-20 ms vs
        // ~61 ms non-persist build. TS_Q35_VERIFY_PERSIST=0 forces the rebuild path.
        static const bool fv_persist_cfg = []{ const char* e = std::getenv("TS_Q35_VERIFY_PERSIST"); return e == nullptr || e[0] != '0'; }();
        // Multimodal MRoPE: per-axis positions (T/H/W/E axis-concatenated, [4N] I32)
        // route the attention RoPE through ggml_rope_multi (interleaved MRoPE, the
        // Qwen3-VL LLM rope). Text prompts keep the plain sequential NeoX rope.
        const bool use_mrope = mrope_pos != nullptr && mrope_sections != nullptr;

        // PREFILL (n_logits < N) processes a long prompt one-shot, so it never reuses
        // the cached graph; force the NON-PERSIST path (pooled ctx + gallocr lifetime-
        // packing) which reuses activation buffers across the graph. The persist path
        // gives every intermediate its own slot (no reuse) — for a 40-layer × N-token
        // graph on the VRAM-tight 35B that thrashes WDDM paging (N=512) or OOMs (N>=1024).
        // MTP verify (n_logits == N) keeps the persist+capture fast-replay reuse.
        // MRoPE calls are prefill-only; keep them off the persist cache too.
        const bool fv_persist = fv_persist_cfg && (n_logits >= N) && !use_mrope;
        auto t_start = std::chrono::high_resolution_clock::now();

        const std::size_t convStateBytes = static_cast<std::size_t>(convDim) * conv_dim * sizeof(float);
        const std::size_t deltaStateBytes = static_cast<std::size_t>(head_k_dim) * head_v_dim * num_v_heads * sizeof(float);
        constexpr int kVerifyKvStride = 256;
        // Persist mode pads the attention window to a fixed stride so one cached graph
        // serves every start_pos in that stride (the mask masks the unused tail).
        const int window = fv_persist
            ? std::min(cache_size, ((totalSeqLen + kVerifyKvStride - 1) / kVerifyKvStride) * kVerifyKvStride)
            : std::min(cache_size, totalSeqLen);
        const void* sig = layers[0].attn_norm_w;

        // Device-resident GDN state: when the C# caller points each recurrent layer's
        // conv_state_in and conv_state_out at the SAME buffer (the decode's device-
        // resident _fdConvScratch slot + _deltaStateTensor), the verify reads/writes the
        // GDN state IN-PLACE on the device (cacheable COMPUTE binding) instead of
        // uploading + downloading it every call (~60 MB delta + 3 MB conv). The state
        // persists across verify/plain steps exactly like the captured decode's; the C#
        // snapshots it (drain) only before a draft-verify for rollback.
        bool resident_state = false;
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].is_recurrent != 0)
            {
                resident_state = (layers[l].conv_state_in == layers[l].conv_state_out) && layers[l].conv_state_in != nullptr;
                break;
            }
        }

        // ===== Persist reuse fast-path: upload the per-call inputs + replay =====
        if (fv_persist)
        {
            for (auto& c : g_q35vc)
            {
                if (!c.valid || c.n != N || c.window != window || c.sig != sig ||
                    c.num_layers != num_layers || c.out_vocab != vocab_size ||
                    c.n_logits != n_logits ||
                    c.has_normed != (normed_out != nullptr))
                    continue;
                // llama.cpp pattern (llama-context.cpp): before re-setting the inputs of
                // a REUSED graph we must fully synchronize, else we overwrite input
                // tensors the previous (async) graph_compute is still reading -> the
                // pipeline accumulates across reuses and faults. host_read_barrier()
                // only syncs conditionally, so force a full backend sync here.
                ggml_backend_synchronize(g_backend);
                ggml_backend_tensor_set(c.hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * N * sizeof(float));
                std::vector<std::int32_t> pv(N);
                std::vector<std::int64_t> kv(N);
                for (int i = 0; i < N; i++) { pv[i] = start_pos + i; kv[i] = start_pos + i; }
                ggml_backend_tensor_set(c.pos_t, pv.data(), 0, static_cast<std::size_t>(N) * sizeof(std::int32_t));
                ggml_backend_tensor_set(c.kv_index, kv.data(), 0, static_cast<std::size_t>(N) * sizeof(std::int64_t));
                std::vector<ggml_fp16_t> mk;
                fill_verify_causal_mask(mk, window, N, start_pos, totalSeqLen);
                ggml_backend_tensor_set(c.mask_t, mk.data(), 0, mk.size() * sizeof(ggml_fp16_t));
                // Host mode uploads the per-call GDN state; resident keeps it device-
                // resident (cacheable, in-place), so no upload/download here.
                if (!resident_state)
                {
                    int gi = 0;
                    for (int l = 0; l < num_layers; l++)
                    {
                        if (layers[l].is_recurrent == 0) continue;
                        ggml_backend_tensor_set(c.conv_in[gi], layers[l].conv_state_in, 0, convStateBytes);
                        ggml_backend_tensor_set(c.delta_in[gi], layers[l].delta_state_in, 0, deltaStateBytes);
                        gi++;
                    }
                }
                auto t_su = std::chrono::high_resolution_clock::now();
                if (ggml_backend_graph_compute(g_backend, c.graph) != GGML_STATUS_SUCCESS) { c.reset(); break; }
                auto t_cu = std::chrono::high_resolution_clock::now();
                if (!resident_state)
                {
                    int gi = 0;
                    for (int l = 0; l < num_layers; l++)
                    {
                        if (layers[l].is_recurrent == 0) continue;
                        finalize_compute_with_download(c.conv_out[gi], layers[l].conv_state_out, convStateBytes);
                        finalize_compute_with_download(c.delta_out[gi], layers[l].delta_state_out, deltaStateBytes);
                        gi++;
                    }
                }
                if (normed_out != nullptr && c.normed_out != nullptr)
                    finalize_compute_with_download(c.normed_out, normed_out, static_cast<std::size_t>(H) * N * sizeof(float));
                finalize_compute_with_download(c.logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_logits * sizeof(float));
                host_read_barrier();
                c.lru = ++g_q35vc_clock;
                if (fv_timing)
                {
                    auto t_end = std::chrono::high_resolution_clock::now();
                    auto ms = [](auto a, auto b){ return std::chrono::duration<double, std::milli>(b - a).count(); };
                    fprintf(stderr, "[fv-timing] REUSE N=%d setup=%.2f compute=%.2f total=%.2f ms\n",
                        N, ms(t_start, t_su), ms(t_su, t_cu), ms(t_start, t_end));
                    fflush(stderr);
                }
                clear_last_error();
                return 1;
            }
        }

        // ===== Build a fresh graph. Persist: raw ctx + alloc_ctx_tensors (each tensor
        // its OWN slot = stable addresses, required for reuse/capture). Non-persist:
        // pooled ctx + gallocr lifetime-packing. =====
        ggml_context* ctx = nullptr;
        PooledContextHandle context;
        if (fv_persist)
        {
            ggml_init_params ip = { 32 * 1024 * 1024, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("Qwen3.5 model verify: failed to init persist ctx."); return 0; }
        }
        else
        {
            if (!context.init(32 * 1024 * 1024)) { set_last_error("Qwen3.5 model verify: failed to acquire ggml context."); return 0; }
            ctx = context.value;
        }

        ggml_tensor* hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
        // MRoPE positions are axis-concatenated [T0..Tn-1, H.., W.., E..] = 4N ints
        // (ggml_rope_multi asserts pos->ne[0] == 4 * a->ne[2]).
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, use_mrope ? 4 * N : N);
        ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
        ggml_tensor* final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

        ggml_tensor* kv_index = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, N);
        ggml_tensor* attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, window, N, 1, 1);
        ggml_set_input(hidden_t);
        ggml_set_input(pos_tensor);
        ggml_set_input(kv_index);
        ggml_set_input(attn_mask);
        std::vector<std::int64_t> kv_index_data(N);
        for (int i = 0; i < N; i++) kv_index_data[i] = start_pos + i;
        std::vector<ggml_fp16_t> attn_mask_data;
        fill_verify_causal_mask(attn_mask_data, window, N, start_pos, totalSeqLen);

        struct LayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w; ggml_tensor* k_w; ggml_tensor* v_w;
            ggml_tensor* q_norm_w; ggml_tensor* k_norm_w; ggml_tensor* o_w;
            ggml_tensor* k_cache_base; ggml_tensor* v_cache_base;
            ggml_tensor* k_cpy; ggml_tensor* v_cpy;
            std::vector<ggml_fp16_t> mask_data;
            ggml_tensor* mask_t;
            // gdn
            ggml_tensor* gdn_qkv_w; ggml_tensor* gdn_gate_w;
            ggml_tensor* ssm_beta_w; ggml_tensor* ssm_alpha_w;
            ggml_tensor* conv1d_w; ggml_tensor* ssm_dt_w; ggml_tensor* ssm_a_w;
            ggml_tensor* ssm_norm_w; ggml_tensor* ssm_out_w;
            ggml_tensor* conv_state_in; ggml_tensor* delta_state_in;
            ggml_tensor* conv_state_out; ggml_tensor* delta_state_out;
            // ffn
            ggml_tensor* post_attn_norm_w; ggml_tensor* gu_w; ggml_tensor* down_w;
            ggml_tensor* gate_inp_w; ggml_tensor* gate_exps; ggml_tensor* up_exps; ggml_tensor* down_exps;
            ggml_tensor* shexp_gate_w; ggml_tensor* shexp_up_w; ggml_tensor* shexp_down_w; ggml_tensor* shexp_gate_inp_w;
        };
        std::vector<LayerTensors> lt(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            if (d.is_recurrent == 0)
            {
                t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
                if (d.separate_qkv != 0)
                {
                    t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
                    t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
                }
                t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
                t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
                t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
                t.k_cache_base = ggml_new_tensor_3d(ctx, kvType, head_dim, cache_size, num_kv_heads);
                t.v_cache_base = ggml_new_tensor_3d(ctx, kvType, head_dim, cache_size, num_kv_heads);
            }
            else
            {
                t.gdn_qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_qkv_type), d.gdn_qkv_ne0, d.gdn_qkv_ne1);
                t.gdn_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_gate_type), d.gdn_gate_ne0, d.gdn_gate_ne1);
                t.ssm_beta_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_beta_type), d.ssm_beta_ne0, d.ssm_beta_ne1);
                t.ssm_alpha_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_alpha_type), d.ssm_alpha_ne0, d.ssm_alpha_ne1);
                t.conv1d_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_kernel, conv_dim);
                t.ssm_dt_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_a_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_v_dim);
                t.ssm_out_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_out_type), d.ssm_out_ne0, d.ssm_out_ne1);
                t.conv_state_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, convDim, conv_dim);
                t.delta_state_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads);
                if (resident_state)
                {
                    // Resident: conv/delta state lives in a device-resident cacheable
                    // COMPUTE buffer (bound below); the post-window state is written
                    // IN-PLACE to conv_state_in / delta_state_in (no separate out tensor,
                    // no per-call upload/download). Saves the ~60 MB delta out alloc too.
                    t.conv_state_out = nullptr;
                    t.delta_state_out = nullptr;
                }
                else
                {
                    t.conv_state_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, convDim, conv_dim);
                    t.delta_state_out = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads);
                    // Per-call GDN state inputs: preserved (set_input) + uploaded each call.
                    ggml_set_input(t.conv_state_in);
                    ggml_set_input(t.delta_state_in);
                }
            }
            if (d.is_moe == 0)
            {
                t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
                t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            }
            else
            {
                t.gate_inp_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gate_inp_type), d.gate_inp_ne0, d.gate_inp_ne1);
                t.gate_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gate_exps_type), hidden_size, expert_ff, num_experts);
                t.up_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.up_exps_type), hidden_size, expert_ff, num_experts);
                t.down_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.down_exps_type), expert_ff, hidden_size, num_experts);
                t.shexp_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_gate_type), d.shexp_gate_ne0, d.shexp_gate_ne1);
                t.shexp_up_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_up_type), d.shexp_up_ne0, d.shexp_up_ne1);
                t.shexp_down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_down_type), d.shexp_down_ne0, d.shexp_down_ne1);
                t.shexp_gate_inp_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            }
        }

        // --- build the chained graph over N tokens ---
        std::vector<ggml_tensor*> state_writes;
        std::vector<ggml_tensor*> kv_writes;   // per-head 2D set_rows results (all attn layers)
        ggml_tensor* hidden = hidden_t;
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];

            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w); // [H, N]
            ggml_tensor* block_out;

            if (d.is_recurrent == 0)
            {
                // ===== Full attention (prefill-style causal flash attention) =====
                ggml_tensor* qg_part; ggml_tensor* k_raw; ggml_tensor* v_raw;
                if (d.separate_qkv != 0)
                {
                    qg_part = ggml_mul_mat(ctx, t.qkv_w, normed);  // [qFullDim, N]
                    k_raw = ggml_mul_mat(ctx, t.k_w, normed);      // [kDim, N]
                    v_raw = ggml_mul_mat(ctx, t.v_w, normed);
                }
                else
                {
                    ggml_tensor* qkv_out = ggml_mul_mat(ctx, t.qkv_w, normed); // [qFullDim+2kDim, N]
                    qg_part = ggml_view_2d(ctx, qkv_out, qFullDim, N, qkv_out->nb[1], 0);
                    k_raw = ggml_view_2d(ctx, qkv_out, kDim, N, qkv_out->nb[1], static_cast<std::size_t>(qFullDim) * sizeof(float));
                    v_raw = ggml_view_2d(ctx, qkv_out, kDim, N, qkv_out->nb[1], static_cast<std::size_t>(qFullDim + kDim) * sizeof(float));
                }

                ggml_tensor* qg_4d = ggml_reshape_4d(ctx, ggml_cont(ctx, qg_part), head_dim, 2, num_heads, N);
                ggml_tensor* q_view = ggml_view_3d(ctx, qg_4d, head_dim, num_heads, N, qg_4d->nb[2], qg_4d->nb[3], 0);
                ggml_tensor* gate_view = ggml_view_3d(ctx, qg_4d, head_dim, num_heads, N, qg_4d->nb[2], qg_4d->nb[3], qg_4d->nb[1]);
                ggml_tensor* q_cont = ggml_cont(ctx, q_view);       // [head_dim, num_heads, N]
                ggml_tensor* gate_cont = ggml_cont(ctx, gate_view); // [head_dim, num_heads, N]
                ggml_tensor* k_3d_raw = ggml_reshape_3d(ctx, ggml_cont(ctx, k_raw), head_dim, num_kv_heads, N);
                ggml_tensor* v_3d_raw = ggml_reshape_3d(ctx, ggml_cont(ctx, v_raw), head_dim, num_kv_heads, N);

                ggml_tensor* q_norm_in = ggml_reshape_2d(ctx, q_cont, head_dim, num_heads * N);
                ggml_tensor* k_norm_in = ggml_reshape_2d(ctx, k_3d_raw, head_dim, num_kv_heads * N);
                ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_norm_in, eps), t.q_norm_w);
                ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_norm_in, eps), t.k_norm_w);

                ggml_tensor* q_4d = ggml_reshape_4d(ctx, q_normed, head_dim, num_heads, N, 1);
                ggml_tensor* k_4d = ggml_reshape_4d(ctx, k_normed, head_dim, num_kv_heads, N, 1);
                ggml_tensor* q_roped;
                ggml_tensor* k_roped;
                if (use_mrope)
                {
                    // Interleaved MRoPE (Qwen3-VL LLM): per-pair modality assignment
                    // comes from the GGUF's mrope sections; positions per token are
                    // the (T,H,W,E) axes uploaded into pos_tensor [4N].
                    int sections_local[4] = { mrope_sections[0], mrope_sections[1], mrope_sections[2], mrope_sections[3] };
                    q_roped = ggml_rope_multi(ctx, q_4d, pos_tensor, nullptr, rope_n_dims, sections_local, GGML_ROPE_TYPE_IMROPE, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
                    k_roped = ggml_rope_multi(ctx, k_4d, pos_tensor, nullptr, rope_n_dims, sections_local, GGML_ROPE_TYPE_IMROPE, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
                }
                else
                {
                    q_roped = ggml_rope_ext(ctx, q_4d, pos_tensor, nullptr, rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
                    k_roped = ggml_rope_ext(ctx, k_4d, pos_tensor, nullptr, rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
                }

                ggml_tensor* q_attn = ggml_permute(ctx, q_roped, 0, 2, 1, 3); // [head_dim, N, num_heads]
                ggml_tensor* k_fresh = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, k_roped, 0, 2, 1, 3)), head_dim, N, num_kv_heads);
                ggml_tensor* v_3d_pre = ggml_reshape_4d(ctx, v_3d_raw, head_dim, num_kv_heads, N, 1);
                ggml_tensor* v_fresh = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, v_3d_pre, 0, 2, 1, 3)), head_dim, N, num_kv_heads);

                // KV write: a 2D ggml_set_rows PER HEAD — dst [head_dim, cache_size]
                // (a contiguous ne2 slice of the head-major cache), src [head_dim, N],
                // idx [N]. This is EXACTLY llama.cpp's proven 2D set_rows shape
                // (n_embd_gqa == head_dim for one head). The single 3D N-row set_rows
                // (heads broadcast via ne2) is the untested combo that faults on
                // cgraph reuse (the decode is fine only because it writes N=1). The
                // write position is an I64 INPUT (kv_index), keeping the graph constant
                // within a window stride for reuse/capture.
                for (int h = 0; h < num_kv_heads; h++)
                {
                    ggml_tensor* k_dst_h = ggml_view_2d(ctx, t.k_cache_base, head_dim, cache_size,
                        t.k_cache_base->nb[1], static_cast<std::size_t>(h) * t.k_cache_base->nb[2]);
                    ggml_tensor* v_dst_h = ggml_view_2d(ctx, t.v_cache_base, head_dim, cache_size,
                        t.v_cache_base->nb[1], static_cast<std::size_t>(h) * t.v_cache_base->nb[2]);
                    ggml_tensor* k_src_h = ggml_view_2d(ctx, k_fresh, head_dim, N,
                        k_fresh->nb[1], static_cast<std::size_t>(h) * k_fresh->nb[2]);
                    ggml_tensor* v_src_h = ggml_view_2d(ctx, v_fresh, head_dim, N,
                        v_fresh->nb[1], static_cast<std::size_t>(h) * v_fresh->nb[2]);
                    kv_writes.push_back(ggml_set_rows(ctx, k_dst_h, k_src_h, kv_index));
                    kv_writes.push_back(ggml_set_rows(ctx, v_dst_h, v_src_h, kv_index));
                }

                // Attend over the fixed window [0, window) (now holds the N fresh rows);
                // the shared causal mask zeroes valid keys and -inf's the rest.
                ggml_tensor* k_full = view_kv_cache_window(ctx, t.k_cache_base, head_dim, cache_size, num_kv_heads, 0, window, kv_cache_type);
                ggml_tensor* v_full = view_kv_cache_window(ctx, t.v_cache_base, head_dim, cache_size, num_kv_heads, 0, window, kv_cache_type);
                if (k_full == nullptr || v_full == nullptr)
                {
                    set_last_error("Qwen3.5 model verify: failed to build KV cache views.");
                    return 0;
                }

                ggml_tensor* attn_flat;
                ggml_tensor* fa = ggml_flash_attn_ext(ctx, q_attn, k_full, v_full, attn_mask, attn_scale, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
                if (backend_supports_op(fa))
                {
                    attn_flat = ggml_reshape_2d(ctx, fa, qDim, N);
                }
                else
                {
                    ggml_tensor* q_attn_cont = ggml_cont(ctx, q_attn);
                    ggml_tensor* scores = ggml_mul_mat(ctx, k_full, q_attn_cont);
                    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
                    ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, attn_mask, attn_scale, 0.0f);
                    ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_full, 1, 0, 2, 3));
                    ggml_tensor* attn_out = ggml_mul_mat(ctx, v_perm, probs);
                    ggml_tensor* attn_perm = ggml_cont(ctx, ggml_permute(ctx, attn_out, 0, 2, 1, 3));
                    attn_flat = ggml_reshape_2d(ctx, attn_perm, qDim, N);
                }

                ggml_tensor* gate_flat = ggml_reshape_2d(ctx, gate_cont, qDim, N);
                ggml_tensor* attn_gated = ggml_mul(ctx, attn_flat, ggml_sigmoid(ctx, gate_flat));
                block_out = ggml_mul_mat(ctx, t.o_w, attn_gated); // [H, N]
            }
            else
            {
                // ===== Gated Delta Net over N tokens (one ggml_gated_delta_net, K=N) =====
                ggml_tensor* qkv_mixed = ggml_mul_mat(ctx, t.gdn_qkv_w, normed);  // [conv_dim, N]
                ggml_tensor* z_all = ggml_mul_mat(ctx, t.gdn_gate_w, normed);     // [value_dim, N]
                ggml_tensor* beta_all = ggml_sigmoid(ctx, ggml_mul_mat(ctx, t.ssm_beta_w, normed)); // [num_v_heads, N]
                ggml_tensor* alpha_all = ggml_mul_mat(ctx, t.ssm_alpha_w, normed); // [num_v_heads, N]
                ggml_tensor* g_all = ggml_softplus(ctx, ggml_add(ctx, alpha_all, t.ssm_dt_w));
                g_all = ggml_mul(ctx, g_all, t.ssm_a_w); // [num_v_heads, N]

                // conv over the N new timesteps prepended with the conv ring state.
                ggml_tensor* qkv_T = ggml_cont(ctx, ggml_transpose(ctx, qkv_mixed));  // [N, conv_dim]
                ggml_tensor* conv_input = ggml_concat(ctx, t.conv_state_in, qkv_T, 0); // [convDim+N, conv_dim]
                ggml_tensor* conv_out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, t.conv1d_w)); // [conv_dim, N]
                // new conv state = the last convDim timesteps (rows [N, N+convDim)).
                ggml_tensor* new_conv = ggml_cont(ctx, ggml_view_2d(ctx, conv_input, convDim, conv_dim, conv_input->nb[1], static_cast<std::size_t>(N) * conv_input->nb[0]));
                // Resident: write the post-window conv state IN-PLACE to conv_state_in
                // (the device-resident buffer); host mode: to the separate out tensor.
                t.conv_state_out = ggml_cpy(ctx, new_conv, resident_state ? t.conv_state_in : t.conv_state_out);

                ggml_tensor* q_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out, key_dim, N, conv_out->nb[1], 0));
                ggml_tensor* k_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out, key_dim, N, conv_out->nb[1], static_cast<std::size_t>(key_dim) * sizeof(float)));
                ggml_tensor* v_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_out, value_dim, N, conv_out->nb[1], static_cast<std::size_t>(2 * key_dim) * sizeof(float)));

                // l2-norm over head_k_dim, then tile q/k from num_k_heads -> num_v_heads.
                ggml_tensor* q_hn = ggml_l2_norm(ctx, ggml_reshape_2d(ctx, q_part, head_k_dim, num_k_heads * N), eps);
                ggml_tensor* k_hn = ggml_l2_norm(ctx, ggml_reshape_2d(ctx, k_part, head_k_dim, num_k_heads * N), eps);
                ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_hn, head_k_dim, num_k_heads, N);
                ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_hn, head_k_dim, num_k_heads, N);
                ggml_tensor* q_tl = q_3d; ggml_tensor* k_tl = k_3d;
                for (int r = 1; r < head_tile; r++) { q_tl = ggml_concat(ctx, q_tl, q_3d, 1); k_tl = ggml_concat(ctx, k_tl, k_3d, 1); }
                ggml_tensor* q4 = ggml_reshape_4d(ctx, ggml_cont(ctx, q_tl), head_k_dim, num_v_heads, N, 1);
                ggml_tensor* k4 = ggml_reshape_4d(ctx, ggml_cont(ctx, k_tl), head_k_dim, num_v_heads, N, 1);
                ggml_tensor* v4 = ggml_reshape_4d(ctx, v_part, head_v_dim, num_v_heads, N, 1);
                ggml_tensor* g4 = ggml_reshape_4d(ctx, ggml_cont(ctx, g_all), 1, num_v_heads, N, 1);
                ggml_tensor* beta4 = ggml_reshape_4d(ctx, ggml_cont(ctx, beta_all), 1, num_v_heads, N, 1);
                ggml_tensor* state4 = ggml_reshape_4d(ctx, t.delta_state_in, head_k_dim, head_v_dim, num_v_heads, 1);

                // K=1: the op recurs over all N tokens internally and emits the per-
                // token outputs (rows [0,N)) + ONLY the FINAL state snapshot (we roll
                // back via host snapshot/re-forward, not the per-prefix snapshots, so
                // requesting K=N would waste ~19 MB/layer of VRAM on unused states).
                ggml_tensor* gdn = ggml_gated_delta_net(ctx, q4, k4, v4, g4, beta4, state4, 1);
                // Per-token outputs occupy the first N rows ([S_v*H] each).
                ggml_tensor* gdn_out = ggml_view_2d(ctx, gdn, value_dim, N, ggml_row_size(gdn->type, value_dim), 0);
                // Final state snapshot (slot 0, most-recent) at offset N * (S_v*H).
                ggml_tensor* new_state = ggml_view_4d(ctx, gdn, head_k_dim, head_v_dim, num_v_heads, 1,
                    ggml_row_size(gdn->type, head_k_dim),
                    ggml_row_size(gdn->type, head_k_dim * head_v_dim),
                    ggml_row_size(gdn->type, head_k_dim * head_v_dim * num_v_heads),
                    ggml_row_size(gdn->type, value_dim) * static_cast<std::size_t>(N));
                // Resident: write the post-window delta state IN-PLACE to delta_state_in
                // (state4 aliases it); host mode: to the separate out tensor.
                t.delta_state_out = ggml_cpy(ctx, new_state, resident_state ? state4 : t.delta_state_out);

                // gated RMSNorm with z, per token: rms_norm(out) * ssm_norm * silu(z).
                ggml_tensor* out_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, gdn_out), head_v_dim, num_v_heads * N);
                ggml_tensor* out_n = ggml_mul(ctx, ggml_rms_norm(ctx, out_2d, eps), t.ssm_norm_w);
                ggml_tensor* out_n_3d = ggml_reshape_3d(ctx, out_n, head_v_dim, num_v_heads, N);
                ggml_tensor* z_3d = ggml_reshape_3d(ctx, z_all, head_v_dim, num_v_heads, N);
                ggml_tensor* gated = ggml_mul(ctx, out_n_3d, ggml_silu(ctx, z_3d));
                ggml_tensor* gated_flat = ggml_reshape_2d(ctx, gated, value_dim, N);
                block_out = ggml_mul_mat(ctx, t.ssm_out_w, gated_flat); // [H, N]

                state_writes.push_back(t.conv_state_out);
                state_writes.push_back(t.delta_state_out);
            }

            ggml_tensor* residual1 = ggml_add(ctx, hidden, block_out); // [H, N]

            // ===== FFN =====
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.post_attn_norm_w); // [H, N]
            ggml_tensor* ffn_out;
            if (d.is_moe == 0)
            {
                const std::int64_t ffDense = d.ff_dense;
                ggml_tensor* gu = ggml_mul_mat(ctx, t.gu_w, ffn_normed); // [2*ffDense, N]
                ggml_tensor* g_part = ggml_cont(ctx, ggml_view_2d(ctx, gu, ffDense, N, gu->nb[1], 0));
                ggml_tensor* u_part = ggml_cont(ctx, ggml_view_2d(ctx, gu, ffDense, N, gu->nb[1], static_cast<std::size_t>(ffDense) * sizeof(float)));
                ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, g_part), u_part);
                ffn_out = ggml_mul_mat(ctx, t.down_w, act); // [H, N]
            }
            else
            {
                ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, ffn_normed); // [num_experts, N]
                ggml_tensor* probs = ggml_soft_max(ctx, router_logits);
                ggml_tensor* sel = ggml_top_k(ctx, probs, num_experts_used);              // [num_used, N]
                ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, num_experts, N);
                ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);                         // [1, num_used, N]
                ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, num_experts_used, N);
                if (norm_topk != 0)
                {
                    ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);
                    w_2d = ggml_div(ctx, w_2d, w_sum);
                }
                if (expert_weights_scale != 1.0f)
                    w_2d = ggml_scale(ctx, w_2d, expert_weights_scale);
                ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, num_experts_used, N);

                ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, ffn_normed, H, 1, N);
                ggml_tensor* g_exp = ggml_mul_mat_id(ctx, t.gate_exps, moe_in_3d, sel);     // [expert_ff, num_used, N]
                ggml_tensor* u_exp = ggml_mul_mat_id(ctx, t.up_exps, moe_in_3d, sel);
                ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, g_exp), u_exp);
                ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps, act, sel);        // [H, num_used, N]
                ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);
                ggml_tensor* moe_out = ggml_cont(ctx, ggml_view_3d(ctx, weighted, H, 1, N, weighted->nb[1], weighted->nb[2], 0));
                for (int u = 1; u < num_experts_used; ++u)
                {
                    ggml_tensor* vu = ggml_view_3d(ctx, weighted, H, 1, N, weighted->nb[1], weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                    moe_out = ggml_add(ctx, moe_out, vu);
                }
                ggml_tensor* moe_out_2d = ggml_reshape_2d(ctx, moe_out, H, N);

                ggml_tensor* sh_g = ggml_mul_mat(ctx, t.shexp_gate_w, ffn_normed); // [shared_ff, N]
                ggml_tensor* sh_u = ggml_mul_mat(ctx, t.shexp_up_w, ffn_normed);
                ggml_tensor* sh_act = ggml_mul(ctx, ggml_silu(ctx, sh_g), sh_u);
                ggml_tensor* sh_down = ggml_mul_mat(ctx, t.shexp_down_w, sh_act); // [H, N]
                ggml_tensor* sh_gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, t.shexp_gate_inp_w, H, 1), ffn_normed)); // [1, N]
                ggml_tensor* sh_out = ggml_mul(ctx, sh_down, sh_gate);
                ffn_out = ggml_add(ctx, moe_out_2d, sh_out);
            }

            hidden = ggml_add(ctx, residual1, ffn_out); // [H, N]
        }

        // Final norm over all N rows -> the MTP head's input AND the LM head.
        ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t); // [H, N]
        ggml_tensor* normed_out_t = nullptr;
        ggml_tensor* normed_cpy = nullptr;
        if (normed_out != nullptr)
        {
            normed_out_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
            normed_cpy = ggml_cpy(ctx, fn, normed_out_t);
            ggml_set_output(normed_cpy);
        }
        // Prefill (n_logits < N) folds the lm_head over only the LAST n_logits
        // columns of the post-norm hidden — the trailing token(s) we sample from.
        ggml_tensor* fn_head_in = fn;                                  // [H, N]
        if (n_logits < N)
        {
            ggml_tensor* fn_last = ggml_view_2d(ctx, fn, H, n_logits, fn->nb[1],
                static_cast<std::size_t>(N - n_logits) * fn->nb[1]);
            fn_head_in = ggml_cont(ctx, fn_last);                      // [H, n_logits]
        }
        ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn_head_in); // [vocab, n_logits]
        ggml_tensor* logits_out_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_logits);
        ggml_tensor* logits_cpy = ggml_cpy(ctx, logits, logits_out_t);
        ggml_set_output(logits_cpy);

        // Per-head set_rows adds ~2*num_kv_heads nodes per attention layer, so size
        // the graph generously to avoid GGML_ASSERT(n_nodes < size).
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (260 + 2 * num_kv_heads) + 1024;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (ggml_tensor* w : kv_writes)
            ggml_build_forward_expand(graph, w);
        for (ggml_tensor* w : state_writes)
        {
            ggml_set_output(w);
            ggml_build_forward_expand(graph, w);
        }
        if (normed_cpy != nullptr)
            ggml_build_forward_expand(graph, normed_cpy);
        ggml_build_forward_expand(graph, logits_cpy);

        // --- bind tensors ---
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* tgt, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (tgt == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, tgt, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, tgt, addr) == GGML_STATUS_SUCCESS)
                    { if (needs_upload) upload_list.push_back({tgt, data, bytes}); return; }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                { if (!cacheable) ephemeral_bufs.emplace_back(buf);
                  if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS) return; }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            if (d.is_moe == 0)
            {
                bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes), true);
                bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes), true);
            }
            else
            {
                bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(d.gate_inp_bytes), true);
                bind_or_mark(t.gate_exps, d.gate_exps, static_cast<std::size_t>(d.gate_exps_bytes), true);
                bind_or_mark(t.up_exps, d.up_exps, static_cast<std::size_t>(d.up_exps_bytes), true);
                bind_or_mark(t.down_exps, d.down_exps, static_cast<std::size_t>(d.down_exps_bytes), true);
                bind_or_mark(t.shexp_gate_w, d.shexp_gate_w, static_cast<std::size_t>(d.shexp_gate_bytes), true);
                bind_or_mark(t.shexp_up_w, d.shexp_up_w, static_cast<std::size_t>(d.shexp_up_bytes), true);
                bind_or_mark(t.shexp_down_w, d.shexp_down_w, static_cast<std::size_t>(d.shexp_down_bytes), true);
                bind_or_mark(t.shexp_gate_inp_w, d.shexp_gate_inp_w, static_cast<std::size_t>(H) * sizeof(float), true);
            }
            if (d.is_recurrent == 0)
            {
                bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
                if (d.separate_qkv != 0)
                {
                    bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true);
                    bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true);
                }
                bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
                bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(head_dim) * sizeof(float), true);
                bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(head_dim) * sizeof(float), true);
                bind_or_mark(t.k_cache_base, d.k_cache, kv_cache_bytes(num_kv_heads, cache_size, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.v_cache_base, d.v_cache, kv_cache_bytes(num_kv_heads, cache_size, head_dim, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (!t.mask_data.empty())
                    bind_or_mark(t.mask_t, t.mask_data.data(), t.mask_data.size() * sizeof(ggml_fp16_t), false);
            }
            else
            {
                bind_or_mark(t.gdn_qkv_w, d.gdn_qkv_w, static_cast<std::size_t>(d.gdn_qkv_bytes), true);
                bind_or_mark(t.gdn_gate_w, d.gdn_gate_w, static_cast<std::size_t>(d.gdn_gate_bytes), true);
                bind_or_mark(t.ssm_beta_w, d.ssm_beta_w, static_cast<std::size_t>(d.ssm_beta_bytes), true);
                bind_or_mark(t.ssm_alpha_w, d.ssm_alpha_w, static_cast<std::size_t>(d.ssm_alpha_bytes), true);
                bind_or_mark(t.conv1d_w, d.conv1d_w, static_cast<std::size_t>(conv_kernel) * conv_dim * sizeof(float), true);
                bind_or_mark(t.ssm_dt_w, d.ssm_dt_w, static_cast<std::size_t>(num_v_heads) * sizeof(float), true);
                bind_or_mark(t.ssm_a_w, d.ssm_a_w, static_cast<std::size_t>(num_v_heads) * sizeof(float), true);
                bind_or_mark(t.ssm_norm_w, d.ssm_norm_w, static_cast<std::size_t>(head_v_dim) * sizeof(float), true);
                bind_or_mark(t.ssm_out_w, d.ssm_out_w, static_cast<std::size_t>(d.ssm_out_bytes), true);
                if (resident_state)
                {
                    // Device-resident GDN state: bind cacheable COMPUTE (keyed by the
                    // decode's _fdConvScratch / _deltaStateTensor host ptrs) so the
                    // buffer persists across calls and is updated in-place. The cacheable
                    // path uploads only when the host key is invalidated (the C# seed);
                    // subsequent calls cache-hit (no upload).
                    bind_or_mark(t.conv_state_in, d.conv_state_in, convStateBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                    bind_or_mark(t.delta_state_in, d.delta_state_in, deltaStateBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                }
                // else: conv_state_in / delta_state_in are per-call set_input inputs,
                // uploaded each call below.
            }
        }
        bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
        bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(H) * sizeof(float), true);

        // Persist: give every still-unbound tensor (intermediates, inputs, outputs,
        // small weights) its OWN stable slot via alloc_ctx_tensors -> reuse/capture.
        ggml_backend_buffer_t persist_buf = nullptr;
        if (fv_persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("Qwen3.5 model verify: failed to allocate persist buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else if (!alloc_graph_reuse_gallocr(graph))
        {
            BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
            if (buffer.value == nullptr)
            {
                set_last_error("Qwen3.5 model verify: failed to allocate backend buffer.");
                return 0;
            }
        }

        host_read_barrier();
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);
        ggml_backend_tensor_set(hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * N * sizeof(float));
        if (use_mrope)
        {
            ggml_backend_tensor_set(pos_tensor, mrope_pos, 0, static_cast<std::size_t>(4) * N * sizeof(std::int32_t));
        }
        else
        {
            std::vector<std::int32_t> pos_vals(N);
            for (int i = 0; i < N; i++) pos_vals[i] = start_pos + i;
            ggml_backend_tensor_set(pos_tensor, pos_vals.data(), 0, static_cast<std::size_t>(N) * sizeof(std::int32_t));
        }
        ggml_backend_tensor_set(kv_index, kv_index_data.data(), 0, static_cast<std::size_t>(N) * sizeof(std::int64_t));
        ggml_backend_tensor_set(attn_mask, attn_mask_data.data(), 0, attn_mask_data.size() * sizeof(ggml_fp16_t));
        if (!resident_state)
        {
            // Host mode: upload the per-call GDN state. Resident mode skips this — the
            // state is device-resident (cacheable), seeded only on invalidation.
            for (int l = 0; l < num_layers; l++)
            {
                if (layers[l].is_recurrent == 0) continue;
                ggml_backend_tensor_set(lt[l].conv_state_in, layers[l].conv_state_in, 0, convStateBytes);
                ggml_backend_tensor_set(lt[l].delta_state_in, layers[l].delta_state_in, 0, deltaStateBytes);
            }
        }

        auto t_setup = std::chrono::high_resolution_clock::now();
        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Qwen3.5 model verify: graph execution failed.");
            return 0;
        }
        auto t_compute = std::chrono::high_resolution_clock::now();

        // Download the post-window GDN state (per recurrent layer) + outputs. Resident
        // mode skips the state download (it stays device-resident, updated in-place).
        if (!resident_state)
        {
            for (int l = 0; l < num_layers; l++)
            {
                const TSGgmlQwen35LayerDesc& d = layers[l];
                if (d.is_recurrent == 0) continue;
                if (d.conv_state_out != nullptr)
                    finalize_compute_with_download(lt[l].conv_state_out, d.conv_state_out, convStateBytes);
                if (d.delta_state_out != nullptr)
                    finalize_compute_with_download(lt[l].delta_state_out, d.delta_state_out, deltaStateBytes);
            }
        }
        if (normed_out != nullptr && normed_out_t != nullptr)
            finalize_compute_with_download(normed_out_t, normed_out, static_cast<std::size_t>(H) * N * sizeof(float));
        finalize_compute_with_download(logits_out_t, logits_data, static_cast<std::size_t>(vocab_size) * n_logits * sizeof(float));
        host_read_barrier();

        if (fv_timing)
        {
            auto t_end = std::chrono::high_resolution_clock::now();
            auto ms = [](auto a, auto b){ return std::chrono::duration<double, std::milli>(b - a).count(); };
            fprintf(stderr, "[fv-timing] BUILD N=%d setup=%.1f compute=%.1f download=%.1f total=%.1f ms\n",
                N, ms(t_start, t_setup), ms(t_setup, t_compute), ms(t_compute, t_end), ms(t_start, t_end));
            fflush(stderr);
        }

        // Persist: keep ctx/graph/buffer alive + record tensor handles so later steps
        // of the same (N, window) shape just upload inputs + replay (capturable).
        if (fv_persist)
        {
            Q35VerifyCache* slot = nullptr;
            for (auto& c : g_q35vc) { if (!c.valid) { slot = &c; break; } }
            if (slot == nullptr)
            {
                slot = &g_q35vc[0];
                for (auto& c : g_q35vc) if (c.lru < slot->lru) slot = &c;
                slot->reset();
            }
            slot->valid = true;
            slot->n = N; slot->window = window; slot->sig = sig;
            slot->num_layers = num_layers; slot->out_vocab = vocab_size;
            slot->n_logits = n_logits;
            slot->has_normed = (normed_out != nullptr);
            slot->ctx = ctx; slot->buffer = persist_buf; slot->graph = graph;
            slot->hidden_t = hidden_t; slot->pos_t = pos_tensor;
            slot->kv_index = kv_index; slot->mask_t = attn_mask;
            slot->logits_out = logits_out_t; slot->normed_out = normed_out_t;
            slot->conv_in.clear(); slot->delta_in.clear(); slot->conv_out.clear(); slot->delta_out.clear();
            for (int l = 0; l < num_layers; l++)
            {
                if (layers[l].is_recurrent == 0) continue;
                slot->conv_in.push_back(lt[l].conv_state_in);
                slot->delta_in.push_back(lt[l].delta_state_in);
                slot->conv_out.push_back(lt[l].conv_state_out);
                slot->delta_out.push_back(lt[l].delta_state_out);
            }
            slot->lru = ++g_q35vc_clock;
        }
        clear_last_error();
        return 1;
    }

    void reset_qwen35_verify_cache()
    {
        for (auto& c : g_q35vc) c.reset();
    }
}

TSG_EXPORT int TSGgml_Qwen35ModelVerify(
    const TSGgmlQwen35LayerDesc* layers, int num_layers,
    void* hidden_data, int hidden_size, int start_pos, int num_tokens,
    int num_heads, int num_kv_heads, int head_dim, int cache_size,
    int rope_n_dims, int rope_mode, int kv_cache_type,
    int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
    float eps, float rope_base, float rope_freq_scale,
    int num_experts, int num_experts_used, int expert_ff, int shared_ff,
    int norm_topk, float expert_weights_scale,
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, void* normed_out, int n_logit_rows,
    const std::int32_t* mrope_pos, const std::int32_t* mrope_sections)
{
    try
    {
        int r = qwen35_model_verify_impl(
            layers, num_layers, hidden_data, hidden_size, start_pos, num_tokens,
            num_heads, num_kv_heads, head_dim, cache_size,
            rope_n_dims, rope_mode, kv_cache_type,
            conv_kernel, head_k_dim, head_v_dim, num_k_heads, num_v_heads,
            eps, rope_base, rope_freq_scale,
            num_experts, num_experts_used, expert_ff, shared_ff,
            norm_topk, expert_weights_scale,
            logits_data, vocab_size,
            lm_head_data, lm_head_type, lm_head_ne0, lm_head_ne1, lm_head_bytes,
            final_norm_data, normed_out, n_logit_rows, mrope_pos, mrope_sections);
        if (r == 0 && std::getenv("TS_Q35_VERIFY_TIMING") != nullptr)
            fprintf(stderr, "[fv-err] %s\n", g_last_error.c_str());
        return r;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Qwen3.5 model verify."); return 0; }
}

// Drop the persistent verify-graph cache. Called from C# whenever the attention KV
// device buffer or GDN-state buffers may have moved (KV cache grow / reset), since
// the cached graphs pin those addresses.
TSG_EXPORT void TSGgml_Qwen35ResetVerifyCache()
{
    reset_qwen35_verify_cache();
}

// ============================================================================
// TSGgml_Qwen35ModelDecodeBatched
//
// TRUE token-batched fused decode (vLLM-style continuous batching): processes
// N sequences' decode tokens (one token per sequence) through the WHOLE hybrid
// transformer in ONE ggml graph. The heavy per-layer matmuls (QKV/GDN input
// projections, dense/MoE FFN, lm-head) run BATCHED over all N tokens, so the
// quantized weights are read from VRAM ONCE per step and amortized across the
// batch — the core throughput win over running N separate single-sequence
// decodes. The cheap per-token recurrent/attention ops (ssm_conv +
// gated_delta_net for GDN layers, flash-attn for attention layers) are emitted
// per-sequence inside the same graph (N small), reusing the exact validated
// single-sequence shapes.
//
// KV cache is PAGED: each attention layer owns a device-resident pool
// [head_dim*num_kv_heads, total_slots]; token t writes its K/V to slot_mapping[t]
// via ggml_set_rows, and each sequence gathers its own history from the pool via
// ggml_get_rows over its block-table-derived slot list (gather_idx/gather_off).
//
// GDN recurrent state is host-passed per batch ([.., n_seqs] conv ring + delta
// state), updated and downloaded each step (the C# layer scatters it back to the
// per-sequence slots), keeping this path trivially correct alongside the per-op
// fallback (host buffers are the single source of truth).
//
// Returns the final pre-output-norm hidden state [hidden, n_tokens]; the C#
// caller owns output_norm + the per-sequence LM head. Returns 0 on anything it
// cannot handle so the caller falls back to the op-by-op batched path.
// ============================================================================
namespace
{
    // Persistent batched-decode graph cache (single entry) for CUDA-graph capture.
    // Built ONCE with stable tensor addresses (raw ctx + alloc_ctx_tensors) and
    // reused across decode steps so ggml-cuda's CUDA-graph capture engages
    // (key = cgraph->nodes[0]); replays one captured graph instead of relaunching
    // the whole ~Nlayers*Nseqs-node graph per step (the WDDM per-node launch tax).
    // Per-step inputs (hidden, positions, slot_mapping, padded gather idx, per-seq
    // mask, GDN conv/delta state) are uploaded to stable addresses each step; the
    // KV pools + weights stay in cached device buffers. Dropped + rebuilt when the
    // shape signature (n_seqs / pad_kv / model) changes or the pools move
    // (TSGgml_Qwen35ResetBatchedDecodeCache from C#).
    struct Q35BatchedDecodeCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_t = nullptr;
        ggml_tensor* hidden_out = nullptr;
        ggml_tensor* pos_t = nullptr;
        ggml_tensor* slot_t = nullptr;
        std::vector<ggml_tensor*> gidx;        // [n_seqs] padded gather idx
        std::vector<ggml_tensor*> mask;        // [n_seqs] F16 attn mask
        std::vector<ggml_tensor*> conv_state;  // per recurrent layer
        std::vector<ggml_tensor*> delta_state; // per recurrent layer
        std::vector<int> gdn_layer;            // layer index for each gdn state entry
        const void* sig = nullptr;
        int num_layers = 0, hidden_size = 0, n_seqs = 0, pad_kv = 0;
        std::size_t conv_bytes = 0, delta_bytes = 0;
        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_t = hidden_out = pos_t = slot_t = nullptr;
            gidx.clear(); mask.clear(); conv_state.clear(); delta_state.clear(); gdn_layer.clear();
            sig = nullptr; num_layers = hidden_size = n_seqs = pad_kv = 0; conv_bytes = delta_bytes = 0;
        }
    };
    Q35BatchedDecodeCache g_q35bdc;

    inline void bfd_upload_mask(ggml_tensor* mask_t, int pad_kv, int seq_len)
    {
        std::vector<ggml_fp16_t> m;
        fill_flash_attn_mask(m, pad_kv, seq_len);
        ggml_backend_tensor_set(mask_t, m.data(), 0, m.size() * sizeof(ggml_fp16_t));
    }

    int qwen35_model_decode_batched_impl(
        const TSGgmlQwen35LayerDesc* layers, int num_layers,
        void* hidden_data, int hidden_size, int n_tokens, int n_seqs,
        const int* positions,            // [n_tokens] I32 absolute positions
        const std::int64_t* slot_mapping,// [n_tokens] I64 global KV slot per token
        const int* gather_idx,           // [n_seqs * pad_kv] I32 padded per-seq slot lists
        const int* seq_lens,             // [n_seqs] I32 valid length per seq (drives attn mask)
        int pad_kv, int total_slots,
        int num_heads, int num_kv_heads, int head_dim,
        int rope_n_dims, int rope_mode, int kv_cache_type,
        int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
        float eps, float rope_base, float rope_freq_scale,
        int num_experts, int num_experts_used, int expert_ff, int shared_ff,
        int norm_topk, float expert_weights_scale)
    {
        if (!ensure_backend()) return 0;
        if (layers == nullptr || num_layers <= 0 || hidden_data == nullptr || n_tokens <= 0 || n_seqs <= 0)
        {
            set_last_error("Qwen3.5 batched decode: invalid arguments.");
            return 0;
        }
        if (n_tokens != n_seqs)
        {
            set_last_error("Qwen3.5 batched decode: V1 requires one token per sequence (n_tokens==n_seqs).");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwen35LayerDesc)))
        {
            set_last_error("Qwen3.5 batched decode: descriptor size mismatch.");
            return 0;
        }

        const int H = hidden_size;
        const int T = n_tokens;
        const int qDim = num_heads * head_dim;
        const int qFullDim = qDim * 2;            // Q + gate interleaved per head
        const int kDim = num_kv_heads * head_dim;
        const int kvFlat = num_kv_heads * head_dim; // pool row size
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int convDim = conv_kernel - 1;
        const int key_dim = head_k_dim * num_k_heads;
        const int value_dim = head_v_dim * num_v_heads;
        const int conv_dim = 2 * key_dim + value_dim;
        const int head_tile = (num_k_heads > 0) ? (num_v_heads / num_k_heads) : 1;

        const void* sig = layers[0].attn_norm_w;   // model-instance discriminator
        // Persistent capturable graph: opt-in (TS_QWEN35_BFD_PERSIST=1). On WDDM
        // (Windows) the captured replay regresses for this batched graph (the
        // dynamic paged gather + per-step GDN-state up/downloads force per-step
        // re-instantiation); on Linux/WSL ggml_cuda (no WDDM per-node tax) the
        // non-captured graph already runs near the kernel floor, so capture is
        // left as a knob for experimentation rather than the default.
        static const bool persist = []{ const char* e = std::getenv("TS_QWEN35_BFD_PERSIST"); return e != nullptr && e[0] == '1'; }();
        const std::size_t convStateBytes = static_cast<std::size_t>(convDim) * conv_dim * n_seqs * sizeof(float);
        const std::size_t deltaStateBytes = static_cast<std::size_t>(head_k_dim) * head_v_dim * num_v_heads * n_seqs * sizeof(float);

        // ===== Persist reuse fast-path: replay the captured graph =====
        if (persist && g_q35bdc.valid && g_q35bdc.graph != nullptr &&
            g_q35bdc.sig == sig && g_q35bdc.num_layers == num_layers &&
            g_q35bdc.hidden_size == H && g_q35bdc.n_seqs == n_seqs && g_q35bdc.pad_kv == pad_kv)
        {
            host_read_barrier();
            ggml_backend_tensor_set(g_q35bdc.hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * T * sizeof(float));
            ggml_backend_tensor_set(g_q35bdc.pos_t, positions, 0, static_cast<std::size_t>(T) * sizeof(std::int32_t));
            ggml_backend_tensor_set(g_q35bdc.slot_t, slot_mapping, 0, static_cast<std::size_t>(T) * sizeof(std::int64_t));
            for (int s = 0; s < n_seqs; s++)
            {
                ggml_backend_tensor_set(g_q35bdc.gidx[s], gather_idx + static_cast<std::size_t>(s) * pad_kv, 0, static_cast<std::size_t>(pad_kv) * sizeof(std::int32_t));
                bfd_upload_mask(g_q35bdc.mask[s], pad_kv, seq_lens[s]);
            }
            for (std::size_t gi = 0; gi < g_q35bdc.gdn_layer.size(); ++gi)
            {
                int l = g_q35bdc.gdn_layer[gi];
                ggml_backend_tensor_set(g_q35bdc.conv_state[gi], layers[l].conv_state_in, 0, convStateBytes);
                ggml_backend_tensor_set(g_q35bdc.delta_state[gi], layers[l].delta_state_in, 0, deltaStateBytes);
            }
            ggml_status st = ggml_backend_graph_compute(g_backend, g_q35bdc.graph);
            if (st != GGML_STATUS_SUCCESS)
            {
                set_last_error("Qwen3.5 batched decode: cached graph execution failed.");
                g_q35bdc.reset();
                return 0;
            }
            finalize_compute_with_download(g_q35bdc.hidden_out, hidden_data, static_cast<std::size_t>(H) * T * sizeof(float));
            for (std::size_t gi = 0; gi < g_q35bdc.gdn_layer.size(); ++gi)
            {
                int l = g_q35bdc.gdn_layer[gi];
                finalize_compute_with_download(g_q35bdc.conv_state[gi], layers[l].conv_state_out, convStateBytes);
                finalize_compute_with_download(g_q35bdc.delta_state[gi], layers[l].delta_state_out, deltaStateBytes);
            }
            host_read_barrier();
            return 1;
        }
        if (persist) g_q35bdc.reset();

        // 32 MB matches the single-seq decode + the pool's max slot size; the
        // metadata-only (no_alloc) ctx needs only ~1-2 MB even for a 40-layer
        // N-seq graph, so this is ample. Persist uses a raw ctx kept alive in the
        // cache for capture/replay; non-persist uses the pooled block.
        const std::size_t ctx_size = static_cast<std::size_t>(32) * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("Qwen3.5 batched decode: failed to init persist ctx."); return 0; }
        }
        else
        {
            if (!context.init(ctx_size))
            {
                set_last_error("Qwen3.5 batched decode: failed to acquire ggml context.");
                return 0;
            }
            ctx = context.value;
        }

        // --- per-token inputs ---
        ggml_tensor* hidden_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
        ggml_tensor* pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
        ggml_tensor* slot_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, T);
        // Per-seq PADDED gather index + attention mask (fixed size pad_kv so the
        // graph topology is identical token-to-token = CUDA-graph capturable).
        std::vector<ggml_tensor*> gidx(n_seqs, nullptr);
        std::vector<ggml_tensor*> mask(n_seqs, nullptr);
        for (int s = 0; s < n_seqs; s++)
        {
            gidx[s] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, pad_kv);
            mask[s] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, pad_kv, 1, 1, 1);
            if (persist) { ggml_set_input(gidx[s]); ggml_set_input(mask[s]); }
        }
        if (persist)
        {
            ggml_set_input(hidden_t);
            ggml_set_input(pos_t);
            ggml_set_input(slot_t);
        }

        struct LayerTensors {
            ggml_tensor* attn_norm_w; ggml_tensor* post_attn_norm_w;
            ggml_tensor* qkv_w; ggml_tensor* k_w; ggml_tensor* v_w;
            ggml_tensor* q_norm_w; ggml_tensor* k_norm_w; ggml_tensor* o_w;
            ggml_tensor* k_pool; ggml_tensor* v_pool;
            ggml_tensor* k_cpy; ggml_tensor* v_cpy;
            ggml_tensor* gdn_qkv_w; ggml_tensor* gdn_gate_w; ggml_tensor* ssm_beta_w; ggml_tensor* ssm_alpha_w;
            ggml_tensor* conv1d_w; ggml_tensor* ssm_dt_w; ggml_tensor* ssm_a_w; ggml_tensor* ssm_norm_w; ggml_tensor* ssm_out_w;
            ggml_tensor* conv_state_in; ggml_tensor* delta_state_in;
            ggml_tensor* conv_state_out; ggml_tensor* delta_state_out;
            ggml_tensor* gu_w; ggml_tensor* down_w;
            ggml_tensor* gate_inp_w; ggml_tensor* gate_exps; ggml_tensor* up_exps; ggml_tensor* down_exps;
            ggml_tensor* shexp_gate_w; ggml_tensor* shexp_up_w; ggml_tensor* shexp_down_w; ggml_tensor* shexp_gate_inp_w;
        };
        std::vector<LayerTensors> lt(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            t = {};
            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            if (d.is_recurrent == 0)
            {
                t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
                if (d.separate_qkv != 0)
                {
                    t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
                    t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
                }
                t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
                t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
                t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
                t.k_pool = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(kv_cache_type), kvFlat, total_slots);
                t.v_pool = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(kv_cache_type), kvFlat, total_slots);
            }
            else
            {
                t.gdn_qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_qkv_type), d.gdn_qkv_ne0, d.gdn_qkv_ne1);
                t.gdn_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_gate_type), d.gdn_gate_ne0, d.gdn_gate_ne1);
                t.ssm_beta_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_beta_type), d.ssm_beta_ne0, d.ssm_beta_ne1);
                t.ssm_alpha_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_alpha_type), d.ssm_alpha_ne0, d.ssm_alpha_ne1);
                t.conv1d_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_kernel, conv_dim);
                t.ssm_dt_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_a_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_v_dim);
                t.ssm_out_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_out_type), d.ssm_out_ne0, d.ssm_out_ne1);
                t.conv_state_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, convDim, conv_dim, n_seqs);
                t.delta_state_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads, n_seqs);
                if (persist) { ggml_set_input(t.conv_state_in); ggml_set_input(t.delta_state_in); }
            }
            if (d.is_moe == 0)
            {
                t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
                t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            }
            else
            {
                t.gate_inp_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gate_inp_type), d.gate_inp_ne0, d.gate_inp_ne1);
                t.gate_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gate_exps_type), hidden_size, expert_ff, num_experts);
                t.up_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.up_exps_type), hidden_size, expert_ff, num_experts);
                t.down_exps = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.down_exps_type), expert_ff, hidden_size, num_experts);
                t.shexp_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_gate_type), d.shexp_gate_ne0, d.shexp_gate_ne1);
                t.shexp_up_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_up_type), d.shexp_up_ne0, d.shexp_up_ne1);
                t.shexp_down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.shexp_down_type), d.shexp_down_ne0, d.shexp_down_ne1);
                t.shexp_gate_inp_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            }
        }

        // --- build the chained graph ---
        std::vector<ggml_tensor*> gdn_state_writes; // in-place conv/delta state writes (graph outputs)
        ggml_tensor* hidden = hidden_t;   // [H, T]
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];

            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w); // [H, T]
            ggml_tensor* block_out; // [H, T]

            if (d.is_recurrent == 0)
            {
                // ===== Full attention (batched proj + per-seq flash-attn) =====
                ggml_tensor* qg_part; ggml_tensor* k_raw; ggml_tensor* v_raw;
                if (d.separate_qkv != 0)
                {
                    qg_part = ggml_mul_mat(ctx, t.qkv_w, normed);   // [qFullDim, T]
                    k_raw = ggml_mul_mat(ctx, t.k_w, normed);       // [kDim, T]
                    v_raw = ggml_mul_mat(ctx, t.v_w, normed);       // [kDim, T]
                }
                else
                {
                    ggml_tensor* qkv = ggml_mul_mat(ctx, t.qkv_w, normed); // [qFullDim+2kDim, T]
                    qg_part = ggml_cont(ctx, ggml_view_2d(ctx, qkv, qFullDim, T, qkv->nb[1], 0));
                    k_raw = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kDim, T, qkv->nb[1], static_cast<std::size_t>(qFullDim) * sizeof(float)));
                    v_raw = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kDim, T, qkv->nb[1], static_cast<std::size_t>(qFullDim + kDim) * sizeof(float)));
                }

                // Deinterleave Q from gate: qg_part is [head_dim*2*num_heads, T].
                ggml_tensor* qg_4d = ggml_reshape_4d(ctx, ggml_cont(ctx, qg_part), head_dim, 2, num_heads, T);
                ggml_tensor* q_view = ggml_view_4d(ctx, qg_4d, head_dim, 1, num_heads, T, qg_4d->nb[1], qg_4d->nb[2], qg_4d->nb[3], 0);
                ggml_tensor* gate_view = ggml_view_4d(ctx, qg_4d, head_dim, 1, num_heads, T, qg_4d->nb[1], qg_4d->nb[2], qg_4d->nb[3], qg_4d->nb[1]);
                ggml_tensor* q_hd = ggml_cont(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, q_view), head_dim, num_heads, T));   // [head_dim, num_heads, T]
                ggml_tensor* gate_hd = ggml_cont(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, gate_view), head_dim, num_heads, T));
                ggml_tensor* k_hd = ggml_reshape_3d(ctx, k_raw, head_dim, num_kv_heads, T);
                ggml_tensor* v_hd = ggml_reshape_3d(ctx, v_raw, head_dim, num_kv_heads, T);

                // per-head q/k RMSNorm (over head_dim = ne0).
                ggml_tensor* q_n = ggml_mul(ctx, ggml_rms_norm(ctx, q_hd, eps), t.q_norm_w);
                ggml_tensor* k_n = ggml_mul(ctx, ggml_rms_norm(ctx, k_hd, eps), t.k_norm_w);

                // RoPE per token (pos_t [T]).
                ggml_tensor* q_rope = ggml_rope_ext(ctx, q_n, pos_t, nullptr, rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0); // [head_dim, num_heads, T]
                ggml_tensor* k_rope = ggml_rope_ext(ctx, k_n, pos_t, nullptr, rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0); // [head_dim, num_kv_heads, T]

                // Write K/V to the paged pool: flatten heads, set_rows by slot.
                ggml_tensor* k_flat = ggml_reshape_2d(ctx, ggml_cont(ctx, k_rope), kvFlat, T);
                ggml_tensor* v_flat = ggml_reshape_2d(ctx, ggml_cont(ctx, v_hd), kvFlat, T);
                t.k_cpy = ggml_set_rows(ctx, t.k_pool, k_flat, slot_t);
                t.v_cpy = ggml_set_rows(ctx, t.v_pool, v_flat, slot_t);

                // Per-seq attention: gather this seq's KV history, flash-attn the
                // seq's single query token. Depends on the set_rows writes above
                // (so the seq's own freshly-written token is visible).
                std::vector<ggml_tensor*> attn_per_seq(n_seqs);
                for (int s = 0; s < n_seqs; s++)
                {
                    ggml_tensor* kf = ggml_get_rows(ctx, t.k_cpy, gidx[s]); // [kvFlat, pad_kv]
                    ggml_tensor* vf = ggml_get_rows(ctx, t.v_cpy, gidx[s]);
                    ggml_tensor* k3 = ggml_reshape_3d(ctx, kf, head_dim, num_kv_heads, pad_kv);
                    ggml_tensor* v3 = ggml_reshape_3d(ctx, vf, head_dim, num_kv_heads, pad_kv);
                    ggml_tensor* kperm = ggml_cont(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3)); // [head_dim, pad_kv, num_kv_heads]
                    ggml_tensor* vperm = ggml_cont(ctx, ggml_permute(ctx, v3, 0, 2, 1, 3));
                    // q for seq s: [head_dim, num_heads, 1] -> permute [head_dim, 1, num_heads]
                    ggml_tensor* qs = ggml_view_3d(ctx, q_rope, head_dim, num_heads, 1, q_rope->nb[1], q_rope->nb[2], static_cast<std::size_t>(s) * q_rope->nb[2]);
                    ggml_tensor* qperm = ggml_cont(ctx, ggml_permute(ctx, qs, 0, 2, 1, 3)); // [head_dim, 1, num_heads]
                    // Padded gather: positions [seq_len, pad_kv) point at slot 0 and
                    // are masked out by mask[s] (0 valid, -inf padding).
                    ggml_tensor* o4 = ggml_flash_attn_ext(ctx, qperm, kperm, vperm, mask[s], attn_scale, 0.0f, 0.0f);
                    ggml_flash_attn_ext_set_prec(o4, GGML_PREC_F32);
                    // o4: [head_dim, num_heads, 1, 1] -> [head_dim*num_heads, 1]
                    attn_per_seq[s] = ggml_reshape_2d(ctx, o4, qDim, 1);
                }
                ggml_tensor* attn_cat = attn_per_seq[0];
                for (int s = 1; s < n_seqs; s++)
                    attn_cat = ggml_concat(ctx, attn_cat, attn_per_seq[s], 1); // [qDim, T]

                // Sigmoid-gated output: attn * sigmoid(gate).
                ggml_tensor* gate_flat = ggml_reshape_2d(ctx, gate_hd, qDim, T);
                ggml_tensor* attn_gated = ggml_mul(ctx, attn_cat, ggml_sigmoid(ctx, gate_flat));
                block_out = ggml_mul_mat(ctx, t.o_w, attn_gated); // [H, T]
            }
            else
            {
                // ===== Gated Delta Net (batched proj + per-seq recurrence) =====
                ggml_tensor* qkv_mixed = ggml_mul_mat(ctx, t.gdn_qkv_w, normed);   // [conv_dim, T]
                ggml_tensor* z_all = ggml_mul_mat(ctx, t.gdn_gate_w, normed);      // [value_dim, T]
                ggml_tensor* beta_all = ggml_sigmoid(ctx, ggml_mul_mat(ctx, t.ssm_beta_w, normed));   // [num_v_heads, T]
                ggml_tensor* alpha_all = ggml_mul_mat(ctx, t.ssm_alpha_w, normed); // [num_v_heads, T]
                // g = softplus(alpha + dt) * a  (per head, broadcast over T)
                ggml_tensor* g_all = ggml_softplus(ctx, ggml_add(ctx, alpha_all, t.ssm_dt_w));
                g_all = ggml_mul(ctx, g_all, t.ssm_a_w);                            // [num_v_heads, T]

                std::vector<ggml_tensor*> gdn_per_seq(n_seqs);
                for (int s = 0; s < n_seqs; s++)
                {
                    ggml_tensor* qkv_s = ggml_cont(ctx, ggml_view_2d(ctx, qkv_mixed, conv_dim, 1, qkv_mixed->nb[1], static_cast<std::size_t>(s) * qkv_mixed->nb[1])); // [conv_dim, 1]
                    ggml_tensor* conv_state_s = ggml_view_2d(ctx, t.conv_state_in, convDim, conv_dim, t.conv_state_in->nb[1], static_cast<std::size_t>(s) * t.conv_state_in->nb[2]); // [convDim, conv_dim]
                    ggml_tensor* qkv_T = ggml_cont(ctx, ggml_transpose(ctx, qkv_s));   // [1, conv_dim]
                    ggml_tensor* conv_input = ggml_concat(ctx, conv_state_s, qkv_T, 0); // [convDim+1, conv_dim]
                    ggml_tensor* conv_out = ggml_silu(ctx, ggml_ssm_conv(ctx, conv_input, t.conv1d_w)); // [conv_dim, 1]
                    ggml_tensor* conv_out_1d = ggml_reshape_1d(ctx, conv_out, conv_dim);
                    ggml_tensor* new_conv = ggml_cont(ctx, ggml_view_2d(ctx, conv_input, convDim, conv_dim, conv_input->nb[1], static_cast<std::size_t>(1) * conv_input->nb[0]));
                    ggml_tensor* conv_save = ggml_cpy(ctx, new_conv, conv_state_s);

                    ggml_tensor* q_c = ggml_cont(ctx, ggml_view_2d(ctx, conv_out_1d, head_k_dim, num_k_heads, static_cast<std::size_t>(head_k_dim) * sizeof(float), 0));
                    ggml_tensor* k_c = ggml_cont(ctx, ggml_view_2d(ctx, conv_out_1d, head_k_dim, num_k_heads, static_cast<std::size_t>(head_k_dim) * sizeof(float), static_cast<std::size_t>(key_dim) * sizeof(float)));
                    ggml_tensor* v_c = ggml_cont(ctx, ggml_view_2d(ctx, conv_out_1d, head_v_dim, num_v_heads, static_cast<std::size_t>(head_v_dim) * sizeof(float), static_cast<std::size_t>(2 * key_dim) * sizeof(float)));
                    q_c = ggml_l2_norm(ctx, q_c, eps);
                    k_c = ggml_l2_norm(ctx, k_c, eps);
                    ggml_tensor* q_tl = q_c; ggml_tensor* k_tl = k_c;
                    for (int r = 1; r < head_tile; r++) { q_tl = ggml_concat(ctx, q_tl, q_c, 1); k_tl = ggml_concat(ctx, k_tl, k_c, 1); }
                    ggml_tensor* q4 = ggml_reshape_4d(ctx, ggml_cont(ctx, q_tl), head_k_dim, num_v_heads, 1, 1);
                    ggml_tensor* k4 = ggml_reshape_4d(ctx, ggml_cont(ctx, k_tl), head_k_dim, num_v_heads, 1, 1);
                    ggml_tensor* v4 = ggml_reshape_4d(ctx, v_c, head_v_dim, num_v_heads, 1, 1);
                    ggml_tensor* beta_s = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, beta_all, num_v_heads, 1, beta_all->nb[1], static_cast<std::size_t>(s) * beta_all->nb[1])), 1, num_v_heads, 1, 1);
                    ggml_tensor* g_s = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, g_all, num_v_heads, 1, g_all->nb[1], static_cast<std::size_t>(s) * g_all->nb[1])), 1, num_v_heads, 1, 1);
                    ggml_tensor* state_s = ggml_view_4d(ctx, t.delta_state_in, head_k_dim, head_v_dim, num_v_heads, 1, t.delta_state_in->nb[1], t.delta_state_in->nb[2], t.delta_state_in->nb[3], static_cast<std::size_t>(s) * t.delta_state_in->nb[3]);
                    ggml_tensor* state4 = ggml_cont(ctx, state_s);

                    ggml_tensor* gdn = ggml_gated_delta_net(ctx, q4, k4, v4, g_s, beta_s, state4, 1);
                    ggml_tensor* gdn_out = ggml_view_4d(ctx, gdn, head_v_dim, num_v_heads, 1, 1,
                        ggml_row_size(gdn->type, head_v_dim), ggml_row_size(gdn->type, head_v_dim * num_v_heads), ggml_row_size(gdn->type, head_v_dim * num_v_heads), 0);
                    ggml_tensor* new_state = ggml_view_4d(ctx, gdn, head_k_dim, head_v_dim, num_v_heads, 1,
                        ggml_row_size(gdn->type, head_k_dim), ggml_row_size(gdn->type, head_k_dim * head_v_dim), ggml_row_size(gdn->type, head_k_dim * head_v_dim * num_v_heads),
                        ggml_row_size(gdn->type, head_v_dim * num_v_heads));
                    ggml_tensor* state_save = ggml_cpy(ctx, new_state, state_s);

                    // gated RMSNorm with z, then collect [value_dim, 1].
                    ggml_tensor* out_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, gdn_out), head_v_dim, num_v_heads);
                    ggml_tensor* out_n = ggml_mul(ctx, ggml_rms_norm(ctx, out_2d, eps), t.ssm_norm_w);
                    ggml_tensor* z_s = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, z_all, value_dim, 1, z_all->nb[1], static_cast<std::size_t>(s) * z_all->nb[1])), head_v_dim, num_v_heads);
                    ggml_tensor* gated = ggml_mul(ctx, out_n, ggml_silu(ctx, z_s));
                    gdn_per_seq[s] = ggml_reshape_2d(ctx, gated, value_dim, 1);
                    // The in-place conv/delta state writes are graph outputs (the
                    // updated state is downloaded after compute); collect them so
                    // ggml_build_forward_expand includes them.
                    gdn_state_writes.push_back(conv_save);
                    gdn_state_writes.push_back(state_save);
                }
                ggml_tensor* gdn_cat = gdn_per_seq[0];
                for (int s = 1; s < n_seqs; s++)
                    gdn_cat = ggml_concat(ctx, gdn_cat, gdn_per_seq[s], 1); // [value_dim, T]
                block_out = ggml_mul_mat(ctx, t.ssm_out_w, gdn_cat); // [H, T]
            }

            ggml_tensor* residual1 = ggml_add(ctx, hidden, block_out); // [H, T]

            // ===== FFN =====
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.post_attn_norm_w); // [H, T]
            ggml_tensor* ffn_out;
            if (d.is_moe == 0)
            {
                const std::int64_t ffDense = d.ff_dense;
                ggml_tensor* gu = ggml_mul_mat(ctx, t.gu_w, ffn_normed); // [2*ffDense, T]
                ggml_tensor* g_part = ggml_cont(ctx, ggml_view_2d(ctx, gu, ffDense, T, gu->nb[1], 0));
                ggml_tensor* u_part = ggml_cont(ctx, ggml_view_2d(ctx, gu, ffDense, T, gu->nb[1], static_cast<std::size_t>(ffDense) * sizeof(float)));
                ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, g_part), u_part); // [ffDense, T]
                ffn_out = ggml_mul_mat(ctx, t.down_w, act); // [H, T]
            }
            else
            {
                // MoE: router -> top-k -> renorm -> stacked experts (mul_mat_id over T tokens) + gated shared expert.
                ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, ffn_normed); // [num_experts, T]
                ggml_tensor* probs = ggml_soft_max(ctx, router_logits);                   // [num_experts, T]
                ggml_tensor* sel = ggml_top_k(ctx, probs, num_experts_used);              // [num_used, T] I32
                ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, num_experts, T);
                ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);                         // [1, num_used, T]
                ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, num_experts_used, T);
                if (norm_topk != 0)
                {
                    ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);                          // [1, T]
                    w_2d = ggml_div(ctx, w_2d, w_sum);
                }
                if (expert_weights_scale != 1.0f)
                    w_2d = ggml_scale(ctx, w_2d, expert_weights_scale);
                ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, num_experts_used, T);

                ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, ffn_normed, H, 1, T);
                ggml_tensor* g_exp = ggml_mul_mat_id(ctx, t.gate_exps, moe_in_3d, sel);     // [expert_ff, num_used, T]
                ggml_tensor* u_exp = ggml_mul_mat_id(ctx, t.up_exps, moe_in_3d, sel);
                ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, g_exp), u_exp);
                ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps, act, sel);        // [H, num_used, T]
                ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);                   // [H, num_used, T]
                // sum over num_used (ne1)
                ggml_tensor* moe_out = ggml_cont(ctx, ggml_view_3d(ctx, weighted, H, 1, T, weighted->nb[1], weighted->nb[2], 0));
                for (int u = 1; u < num_experts_used; ++u)
                {
                    ggml_tensor* vu = ggml_view_3d(ctx, weighted, H, 1, T, weighted->nb[1], weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                    moe_out = ggml_add(ctx, moe_out, vu);
                }
                ggml_tensor* moe_out_2d = ggml_reshape_2d(ctx, moe_out, H, T);

                // gated shared expert
                ggml_tensor* sh_g = ggml_mul_mat(ctx, t.shexp_gate_w, ffn_normed); // [shared_ff, T]
                ggml_tensor* sh_u = ggml_mul_mat(ctx, t.shexp_up_w, ffn_normed);
                ggml_tensor* sh_act = ggml_mul(ctx, ggml_silu(ctx, sh_g), sh_u);
                ggml_tensor* sh_down = ggml_mul_mat(ctx, t.shexp_down_w, sh_act); // [H, T]
                ggml_tensor* sh_gate = ggml_sigmoid(ctx, ggml_mul_mat(ctx, ggml_reshape_2d(ctx, t.shexp_gate_inp_w, H, 1), ffn_normed)); // [1, T]
                ggml_tensor* sh_out = ggml_mul(ctx, sh_down, sh_gate);
                ffn_out = ggml_add(ctx, moe_out_2d, sh_out);
            }

            hidden = ggml_add(ctx, residual1, ffn_out); // [H, T]
        }

        ggml_tensor* hidden_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, T);
        ggml_tensor* out_cpy = ggml_cpy(ctx, hidden, hidden_out);
        ggml_set_output(out_cpy);

        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (160 + 32 * n_seqs) + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].is_recurrent == 0)
            {
                ggml_build_forward_expand(graph, lt[l].k_cpy);
                ggml_build_forward_expand(graph, lt[l].v_cpy);
            }
        }
        for (ggml_tensor* w : gdn_state_writes)
        {
            ggml_set_output(w);
            ggml_build_forward_expand(graph, w);
        }
        ggml_build_forward_expand(graph, out_cpy);

        // --- bind tensors ---
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* tgt, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (tgt == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, tgt, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, tgt, addr) == GGML_STATUS_SUCCESS)
                    { if (needs_upload) upload_list.push_back({tgt, data, bytes}); return; }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                { if (!cacheable) ephemeral_bufs.emplace_back(buf);
                  if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS) return; }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        const std::size_t poolBytes = kv_cache_bytes(num_kv_heads, total_slots, head_dim, kv_cache_type);
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            LayerTensors& t = lt[l];
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            if (d.is_moe == 0)
            {
                bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes), true);
                bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes), true);
            }
            else
            {
                bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(d.gate_inp_bytes), true);
                bind_or_mark(t.gate_exps, d.gate_exps, static_cast<std::size_t>(d.gate_exps_bytes), true);
                bind_or_mark(t.up_exps, d.up_exps, static_cast<std::size_t>(d.up_exps_bytes), true);
                bind_or_mark(t.down_exps, d.down_exps, static_cast<std::size_t>(d.down_exps_bytes), true);
                bind_or_mark(t.shexp_gate_w, d.shexp_gate_w, static_cast<std::size_t>(d.shexp_gate_bytes), true);
                bind_or_mark(t.shexp_up_w, d.shexp_up_w, static_cast<std::size_t>(d.shexp_up_bytes), true);
                bind_or_mark(t.shexp_down_w, d.shexp_down_w, static_cast<std::size_t>(d.shexp_down_bytes), true);
                bind_or_mark(t.shexp_gate_inp_w, d.shexp_gate_inp_w, static_cast<std::size_t>(H) * sizeof(float), true);
            }
            if (d.is_recurrent == 0)
            {
                bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
                if (d.separate_qkv != 0)
                {
                    bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true);
                    bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true);
                }
                bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
                bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(head_dim) * sizeof(float), true);
                bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(head_dim) * sizeof(float), true);
                bind_or_mark(t.k_pool, d.k_cache, poolBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.v_pool, d.v_cache, poolBytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
            else
            {
                bind_or_mark(t.gdn_qkv_w, d.gdn_qkv_w, static_cast<std::size_t>(d.gdn_qkv_bytes), true);
                bind_or_mark(t.gdn_gate_w, d.gdn_gate_w, static_cast<std::size_t>(d.gdn_gate_bytes), true);
                bind_or_mark(t.ssm_beta_w, d.ssm_beta_w, static_cast<std::size_t>(d.ssm_beta_bytes), true);
                bind_or_mark(t.ssm_alpha_w, d.ssm_alpha_w, static_cast<std::size_t>(d.ssm_alpha_bytes), true);
                bind_or_mark(t.conv1d_w, d.conv1d_w, static_cast<std::size_t>(conv_kernel) * conv_dim * sizeof(float), true);
                bind_or_mark(t.ssm_dt_w, d.ssm_dt_w, static_cast<std::size_t>(num_v_heads) * sizeof(float), true);
                bind_or_mark(t.ssm_a_w, d.ssm_a_w, static_cast<std::size_t>(num_v_heads) * sizeof(float), true);
                bind_or_mark(t.ssm_norm_w, d.ssm_norm_w, static_cast<std::size_t>(head_v_dim) * sizeof(float), true);
                bind_or_mark(t.ssm_out_w, d.ssm_out_w, static_cast<std::size_t>(d.ssm_out_bytes), true);
                bind_or_mark(t.conv_state_in, d.conv_state_in, convStateBytes, false, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.delta_state_in, d.delta_state_in, deltaStateBytes, false, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
        }

        // Allocate the graph tensors. Persist uses alloc_ctx_tensors (each tensor
        // its own slot = STABLE addresses, required for CUDA-graph capture); non-
        // persist tries gallocr lifetime-packing first.
        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("Qwen3.5 batched decode: failed to allocate persist backend buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("Qwen3.5 batched decode: failed to allocate backend buffer.");
                return 0;
            }
        }

        host_read_barrier();
        for (auto& u : upload_list) ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);
        ggml_backend_tensor_set(hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * T * sizeof(float));
        ggml_backend_tensor_set(pos_t, positions, 0, static_cast<std::size_t>(T) * sizeof(std::int32_t));
        ggml_backend_tensor_set(slot_t, slot_mapping, 0, static_cast<std::size_t>(T) * sizeof(std::int64_t));
        for (int s = 0; s < n_seqs; s++)
        {
            ggml_backend_tensor_set(gidx[s], gather_idx + static_cast<std::size_t>(s) * pad_kv, 0, static_cast<std::size_t>(pad_kv) * sizeof(std::int32_t));
            bfd_upload_mask(mask[s], pad_kv, seq_lens[s]);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Qwen3.5 batched decode: graph execution failed.");
            if (persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
            return 0;
        }

        finalize_compute_with_download(hidden_out, hidden_data, static_cast<std::size_t>(H) * T * sizeof(float));
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].is_recurrent != 0)
            {
                finalize_compute_with_download(lt[l].conv_state_in, layers[l].conv_state_out, convStateBytes);
                finalize_compute_with_download(lt[l].delta_state_in, layers[l].delta_state_out, deltaStateBytes);
            }
        }
        host_read_barrier();

        if (persist)
        {
            g_q35bdc.ctx = ctx;
            g_q35bdc.buffer = persist_buf;
            g_q35bdc.graph = graph;
            g_q35bdc.hidden_t = hidden_t;
            g_q35bdc.hidden_out = hidden_out;
            g_q35bdc.pos_t = pos_t;
            g_q35bdc.slot_t = slot_t;
            g_q35bdc.gidx = gidx;
            g_q35bdc.mask = mask;
            g_q35bdc.conv_state.clear(); g_q35bdc.delta_state.clear(); g_q35bdc.gdn_layer.clear();
            for (int l = 0; l < num_layers; l++)
            {
                if (layers[l].is_recurrent != 0)
                {
                    g_q35bdc.conv_state.push_back(lt[l].conv_state_in);
                    g_q35bdc.delta_state.push_back(lt[l].delta_state_in);
                    g_q35bdc.gdn_layer.push_back(l);
                }
            }
            g_q35bdc.sig = sig;
            g_q35bdc.num_layers = num_layers;
            g_q35bdc.hidden_size = H;
            g_q35bdc.n_seqs = n_seqs;
            g_q35bdc.pad_kv = pad_kv;
            g_q35bdc.conv_bytes = convStateBytes;
            g_q35bdc.delta_bytes = deltaStateBytes;
            g_q35bdc.valid = true;
        }
        clear_last_error();
        return 1;
    }
}

TSG_EXPORT int TSGgml_Qwen35ModelDecodeBatched(
    const TSGgmlQwen35LayerDesc* layers, int num_layers,
    void* hidden_data, int hidden_size, int n_tokens, int n_seqs,
    const int* positions, const std::int64_t* slot_mapping,
    const int* gather_idx, const int* seq_lens, int pad_kv, int total_slots,
    int num_heads, int num_kv_heads, int head_dim,
    int rope_n_dims, int rope_mode, int kv_cache_type,
    int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
    float eps, float rope_base, float rope_freq_scale,
    int num_experts, int num_experts_used, int expert_ff, int shared_ff,
    int norm_topk, float expert_weights_scale)
{
    try
    {
        int r = qwen35_model_decode_batched_impl(
            layers, num_layers, hidden_data, hidden_size, n_tokens, n_seqs,
            positions, slot_mapping, gather_idx, seq_lens, pad_kv, total_slots,
            num_heads, num_kv_heads, head_dim, rope_n_dims, rope_mode, kv_cache_type,
            conv_kernel, head_k_dim, head_v_dim, num_k_heads, num_v_heads,
            eps, rope_base, rope_freq_scale,
            num_experts, num_experts_used, expert_ff, shared_ff,
            norm_topk, expert_weights_scale);
        if (r == 0 && std::getenv("TS_QWEN35_FD_TIMING") != nullptr)
            fprintf(stderr, "[fd-batched-err] %s\n", g_last_error.c_str());
        return r;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Qwen3.5 batched decode."); return 0; }
}

// Drop the persistent batched-decode graph cache (C# calls this when the device
// KV pools are reallocated or the model state is reset, since the cached graph
// pins those device addresses).
TSG_EXPORT void TSGgml_Qwen35ResetBatchedDecodeCache()
{
    g_q35bdc.reset();
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

namespace
{
    // Persistent decode-graph cache for the dense Gemma 4 trunk — the exact
    // analogue of g_q35dc (see its comment). The whole-model decode graph is
    // built ONCE with stable tensor addresses (raw ggml ctx + alloc_ctx_tensors)
    // and reused across tokens, so ggml-cuda's CUDA-graph capture engages (one
    // captured replay instead of re-launching ~1300 kernels/token, which on WDDM
    // starves the GPU — measured ~25 tok/s vs ~38 for llama.cpp before this).
    //
    // To keep the graph topology byte-identical token-to-token (the requirement
    // for replay, see ggml_cuda_graph_update_required): the KV write uses
    // ggml_set_rows (write row = an I64 INPUT, not a baked view offset) and the
    // attention reads a FIXED padded window [0, window) with an F16 mask INPUT.
    // The window is padded up to a 256-token stride so it only changes (forcing a
    // rebuild) every 256 tokens. Global layers grow their window; local (SWA)
    // layers saturate at the sliding-window size and then read the whole circular
    // cache flat (attention is permutation-invariant over the KV axis, so the
    // circular order is irrelevant). PLE / per-layer-input embeddings are an
    // extra per-token input. Dropped + rebuilt when any layer window grows a
    // stride, the model instance changes, or the KV-cache buffer is reallocated.
    constexpr int kG4PersistKvStride = 256;

    struct G4DecodeCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_in = nullptr;
        ggml_tensor* hidden_out = nullptr;
        ggml_tensor* pos_tensor = nullptr;
        ggml_tensor* ple_input = nullptr;            // nullable (uploaded-PLE mode)
        ggml_tensor* ple_ids = nullptr;              // nullable (in-kernel PLE gather mode: per-token I32 id)
        std::vector<ggml_tensor*> kv_index;          // per-layer I64 write row (null for shared)
        std::vector<ggml_tensor*> attn_mask;         // per-layer F16 padding mask (shared -> donor's)
        std::vector<int> layer_window;               // per-layer padded window length
        const void* sig_disc = nullptr;              // model-instance discriminator (attn_norm[0])
        const void* sig_kcache0 = nullptr;           // first KV buffer ptr (detects realloc/grow)
        int num_layers = 0;
        int hidden_size = 0;
        int ple_dim = 0;
        bool ple_gather = false;                     // in-kernel PLE gather graph vs uploaded ple_data
        bool folded = false;                         // hidden_out holds logits (final norm + lm_head folded in)
        int out_count = 0;                           // floats to download (vocab when folded, else hidden)

        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_in = hidden_out = pos_tensor = ple_input = ple_ids = nullptr;
            kv_index.clear(); attn_mask.clear(); layer_window.clear();
            sig_disc = sig_kcache0 = nullptr;
            num_layers = hidden_size = ple_dim = 0;
            ple_gather = false;
            folded = false; out_count = 0;
        }
    };

    // Concurrent (N>=2) requests each decode through their OWN KV-cache holder
    // (Gemma4Model.BindSequenceCache swaps _kvCacheK), so a single shared cache
    // entry — whose identity key includes sig_kcache0, the holder's layer-0 KV
    // pointer — is busted on EVERY request switch and rebuilt from scratch,
    // collapsing aggregate throughput BELOW the single-stream rate. Keep a small
    // pool keyed by (sig_disc, sig_kcache0) so each in-flight request retains its
    // own persistent, CUDA-graph-captured decode graph. ggml-cuda already caches a
    // separate captured cuda graph per distinct cgraph->nodes[0]
    // (ggml_cuda_graph_get_key), so switching requests is a cheap replay instead
    // of a full graph rebuild + recapture.
    //
    // Address-keying is self-consistent: a captured graph bakes in the KV device
    // addresses, and the per-call inputs (hidden / pos / kv_index / mask) are
    // uploaded fresh each replay, so an entry is only ever matched when a LIVE
    // holder presents its kc0 — a freed-then-reallocated address simply binds the
    // new live holder's buffers. Entries' own ctx/buffer stay alive until reset or
    // LRU eviction; ResetKVCache drops them all.
    constexpr int kG4MaxDecodeCaches = 8;
    struct G4DecodeCachePool
    {
        G4DecodeCache entries[kG4MaxDecodeCaches];
        std::uint64_t used[kG4MaxDecodeCaches] = {};   // LRU clock per slot
        std::uint64_t clock = 0;

        // Live entry matching this identity (model instance + KV holder), or null.
        G4DecodeCache* find(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kG4MaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                { used[i] = ++clock; return &entries[i]; }
            return nullptr;
        }

        // A reset slot to (re)build this identity's graph into: reuse its existing
        // slot if present (shape change -> rebuild in place), else an empty slot,
        // else evict the least-recently-used.
        G4DecodeCache& claim(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kG4MaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            for (int i = 0; i < kG4MaxDecodeCaches; i++)
                if (!entries[i].valid) { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            int lru = 0;
            for (int i = 1; i < kG4MaxDecodeCaches; i++) if (used[i] < used[lru]) lru = i;
            entries[lru].reset(); used[lru] = ++clock; return entries[lru];
        }

        // Drop the entry for one identity (a call that cannot persist for it).
        void drop(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kG4MaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                    entries[i].reset();
        }

        void reset_all() { for (auto& e : entries) e.reset(); }
    };
    G4DecodeCachePool g_g4dc_pool;
}

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
    int kv_cache_type,
    // Separate K/V projection weights for mixed-quantization models (e.g.
    // UD-IQ2_M) where attn_q/attn_k/attn_v carry DIFFERENT ggml types and so
    // cannot be fused into a single attn_qkv tensor. When k_arr[l] != nullptr
    // the layer runs three separate Q/K/V matmuls and qkv_arr[l] then holds
    // the Q weight (with qkv_*_arr[l] describing Q). When k_arr == nullptr or
    // k_arr[l] == nullptr the layer uses the fused attn_qkv weight as before,
    // so existing fully-fused callers are unaffected.
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    // Folded final-norm + lm_head (nullable). When logits_data / lm_head_data /
    // final_norm_data are non-null and vocab_size > 0, the graph appends the
    // output RMSNorm, the lm_head matmul and (when logit_softcap > 0) the
    // tanh logit softcap, writing logits[vocab] to logits_data so the whole
    // token — including the 262K-vocab projection — is one captured replay and
    // the C# caller skips its separate final-norm/lm_head/softcap dispatches.
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, float logit_softcap,
    // In-kernel PLE gather (nullable, mirrors TSGgml_Gemma4ModelVerify): when the
    // quantized per_layer_token_embd table (+ optionally the quantized
    // per_layer_model_proj and its F32 norm) is supplied along with the token id,
    // the graph reproduces C#'s ComputePLE on-device — get_rows + hidden
    // projection + RMSNorm + combine — and `ple_data` is ignored. This removes
    // ~5 per-op device dispatches per decode token (measured ~4.8 ms/token on
    // ggml-vulkan, ~1 ms on ggml-cuda).
    const void* ple_token_embd_data, int ple_token_embd_type,
    std::int64_t ple_token_embd_ne0, std::int64_t ple_token_embd_ne1, std::int64_t ple_token_embd_bytes,
    int ple_token_id,
    const void* ple_model_proj_data, int ple_model_proj_type,
    std::int64_t ple_model_proj_ne0, std::int64_t ple_model_proj_ne1, std::int64_t ple_model_proj_bytes,
    const float* ple_model_proj_norm_data)
{
    try
    {
        if (!ensure_backend())
            return 0;

        const int totalSeqLen = position + 1;

        // In-kernel PLE gather mode (see the parameter block above).
        const bool ple_gather = ple_dim > 0 && ple_token_embd_data != nullptr && ple_token_id >= 0;

        // Fold final-norm + lm_head into the graph when the caller supplies the
        // lm_head weight, output-norm weight and a logits output buffer.
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;

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

        // ============================ persist decode ============================
        // Per-token capturable-graph setup (mirrors Qwen3.5 g_q35dc). Compute each
        // layer's PADDED attention window, its valid (unmasked) length, and the KV
        // write row, then either replay the cached graph or fall through to build
        // a fresh persistent one.
        static const bool g4_fd_timing = std::getenv("TS_GEMMA4_FD_TIMING") != nullptr;
        static const bool g4_persist = []{ const char* e = std::getenv("TS_GEMMA4_FD_PERSIST"); return e == nullptr || e[0] != '0'; }();

        std::vector<int> pwindow(num_layers, 0);          // padded window length per layer
        std::vector<int> pvalid(num_layers, 0);           // unmasked length per layer
        std::vector<std::int64_t> pwrite(num_layers, 0);  // set_rows write row per layer
        // Persist (capturable-graph) decode writes the KV cache with ggml_set_rows
        // (write row = an I64 INPUT). On the Metal backend ggml_set_rows over a
        // gallocr-allocated graph hands a null-context buffer to
        // ggml_metal_op_set_rows -> ggml_metal_buffer_get_id (EXC_BAD_ACCESS at
        // 0x14, SIGSEGV), so Metal falls through to the proven ggml_cpy path
        // below, which STILL folds the LM head and keeps the KV cache
        // device-resident. On CUDA persist enables CUDA-graph capture; on Vulkan
        // there is no capture but reusing the built graph still removes the
        // per-token rebuild + bind + gallocr work (~1 ms/token measured) and lets
        // ggml-vulkan's rope+view+set_rows subgraph fusion apply. Vulkan's
        // set_rows covers every KV dtype used here (F32/F16/BF16/Q8_0/Q4_0).
        bool can_persist = g4_persist &&
            (g_backend_type == BACKEND_TYPE_CUDA || g_backend_type == BACKEND_TYPE_VULKAN);
        {
            auto roundup_stride = [](int v){ return ((v + kG4PersistKvStride - 1) / kG4PersistKvStride) * kG4PersistKvStride; };
            for (int l = 0; l < num_layers; l++)
            {
                const int csz = li[l].cacheSize;
                if (csz <= 0) { can_persist = false; break; }
                if (li[l].isLocal)
                {
                    if (totalSeqLen <= csz) { pwindow[l] = std::min(csz, roundup_stride(totalSeqLen)); pvalid[l] = totalSeqLen; }
                    else { pwindow[l] = csz; pvalid[l] = csz; }   // saturated: read whole circular cache flat
                }
                else
                {
                    pwindow[l] = std::min(csz, roundup_stride(totalSeqLen)); pvalid[l] = totalSeqLen;
                }
                pwrite[l] = li[l].isLocal ? (position % csz) : position;
                // A global cache that already overflowed can't be expressed as a
                // single padded window -> let the legacy per-op path handle it.
                if (!li[l].isShared && pvalid[l] > pwindow[l]) { can_persist = false; break; }
            }
        }
        const void* g4_sig = attn_norm_arr[0];
        const void* g4_kc0 = k_cache_arr != nullptr ? k_cache_arr[0] : nullptr;

        // ---- reuse fast-path: replay THIS request's captured graph ----
        // Look the graph up by (model instance, KV holder). A per-request hit lets
        // concurrent requests each replay their own captured graph instead of one
        // shared entry that rebuilds on every switch. find() already matched
        // sig_disc + sig_kcache0, so only the finer shape fields are checked here.
        G4DecodeCache* dc = (can_persist && g4_persist) ? g_g4dc_pool.find(g4_sig, g4_kc0) : nullptr;
        if (dc != nullptr && dc->graph != nullptr &&
            dc->num_layers == num_layers && dc->hidden_size == hidden_size &&
            dc->ple_dim == ple_dim && dc->ple_gather == ple_gather &&
            dc->layer_window == pwindow &&
            dc->folded == fold && dc->out_count == (fold ? vocab_size : hidden_size))
        {
            auto t_start = std::chrono::high_resolution_clock::now();
            host_read_barrier();
            // Per-token input refresh. These ~N*2 small host->device copies feed the
            // captured graph's INPUT tensors; they are not graph nodes. Issue them
            // ASYNC on the backend stream so they queue (stream-ordered) ahead of the
            // graph replay below instead of each blocking on its own stream sync —
            // the replay and the trailing finalize_compute_with_download() sync make
            // the data visible at the right points. (CUDA pageable H2D async is
            // host-synchronous w.r.t. the source, so the per-iteration mask buffer is
            // safe to free immediately.) Removes ~N redundant per-copy syncs/token.
            decode_input_set_async(dc->hidden_in, hidden_data, static_cast<std::size_t>(hidden_size) * sizeof(float));
            std::int32_t pos_val = position;
            decode_input_set_async(dc->pos_tensor, &pos_val, sizeof(std::int32_t));
            std::int32_t ple_id_val = ple_token_id;
            if (dc->ple_ids != nullptr)
                decode_input_set_async(dc->ple_ids, &ple_id_val, sizeof(std::int32_t));
            else if (dc->ple_input != nullptr && ple_data != nullptr)
                decode_input_set_async(dc->ple_input, ple_data, static_cast<std::size_t>(num_layers) * ple_dim * sizeof(float));
            for (int l = 0; l < num_layers; l++)
            {
                if (dc->kv_index[l] != nullptr)
                    decode_input_set_async(dc->kv_index[l], &pwrite[l], sizeof(std::int64_t));
                if (dc->attn_mask[l] != nullptr && !li[l].isShared)
                {
                    std::vector<ggml_fp16_t> md;
                    fill_flash_attn_mask(md, pwindow[l], pvalid[l]);
                    decode_input_set_async(dc->attn_mask[l], md.data(), md.size() * sizeof(ggml_fp16_t));
                }
            }
            auto t_setup = std::chrono::high_resolution_clock::now();
            ggml_status st = ggml_backend_graph_compute(g_backend, dc->graph);
            if (st != GGML_STATUS_SUCCESS)
            {
                set_last_error("Gemma4 model decode: cached graph execution failed.");
                dc->reset();
                return 0;
            }
            auto t_compute = std::chrono::high_resolution_clock::now();
            void* out_data = dc->folded ? logits_data : hidden_data;
            finalize_compute_with_download(dc->hidden_out, out_data, static_cast<std::size_t>(dc->out_count) * sizeof(float));
            host_read_barrier();
            if (g4_fd_timing)
            {
                auto t_end = std::chrono::high_resolution_clock::now();
                auto ms = [](auto a, auto b){ return std::chrono::duration<double, std::milli>(b - a).count(); };
                fprintf(stderr, "[g4-fd] REUSE setup=%.2f compute=%.2f download=%.2f total=%.2f ms\n",
                    ms(t_start, t_setup), ms(t_setup, t_compute), ms(t_compute, t_end), ms(t_start, t_end));
                fflush(stderr);
            }
            clear_last_error();
            return 1;
        }
        // Miss -> (re)build. Claim this request's slot (reset in place / evict LRU)
        // for a persistable call; otherwise drop any stale entry for this identity.
        G4DecodeCache* g4dc = nullptr;
        if (g4_persist)
        {
            if (can_persist) g4dc = &g_g4dc_pool.claim(g4_sig, g4_kc0);
            else             g_g4dc_pool.drop(g4_sig, g4_kc0);
        }

        auto t_build_start = std::chrono::high_resolution_clock::now();

        // Create GGML context. Persist mode uses a raw no_alloc ctx kept alive in
        // g_g4dc (stable tensor addresses for capture); legacy uses the pool.
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("Gemma4 model decode: failed to init persist ggml context."); return 0; }
        }
        else
        {
            if (!context.init(ctx_size))
            {
                set_last_error("Failed to create ggml context for Gemma4 model decode.");
                return 0;
            }
            ctx = context.value;
        }

        // Per-layer persist inputs (created in the build loop below; null in legacy).
        std::vector<ggml_tensor*> layer_kv_index(num_layers, nullptr);

        ggml_tensor* current = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

        ggml_tensor* freq_factors_t = nullptr;
        if (rope_freq_factors != nullptr && rope_freq_factors_len > 0)
            freq_factors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rope_freq_factors_len);

        // PLE input: either gathered in-kernel from the resident quantized table
        // (ple_gather; per-token input = the token id) or uploaded as an F32
        // buffer computed by C#'s ComputePLE (legacy path).
        const int total_ple_dim = num_layers * ple_dim;
        ggml_tensor* ple_input = nullptr;
        ggml_tensor* ple_table_t = nullptr;
        ggml_tensor* ple_ids_t = nullptr;
        ggml_tensor* ple_model_proj_t = nullptr;
        ggml_tensor* ple_model_proj_norm_t = nullptr;
        if (ple_gather)
        {
            static std::atomic<int> s_ple_dbg_once{0};
            if (g4_fd_timing && s_ple_dbg_once.fetch_add(1) == 0)
            {
                fprintf(stderr, "[g4-fd] ple_gather dims: table ne0=%lld ne1=%lld type=%d, proj ne0=%lld ne1=%lld type=%d, num_layers=%d ple_dim=%d total=%d hidden=%d token=%d\n",
                    (long long)ple_token_embd_ne0, (long long)ple_token_embd_ne1, ple_token_embd_type,
                    (long long)ple_model_proj_ne0, (long long)ple_model_proj_ne1, ple_model_proj_type,
                    num_layers, ple_dim, total_ple_dim, hidden_size, ple_token_id);
                fflush(stderr);
            }
            ple_table_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_token_embd_type),
                ple_token_embd_ne0, ple_token_embd_ne1);
            ple_ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            // Token-embedding component: sqrt(ple_dim) * get_rows(table, id).
            ggml_tensor* ple_tok = ggml_get_rows(ctx, ple_table_t, ple_ids_t);
            ple_tok = ggml_scale(ctx, ple_tok, sqrtf(static_cast<float>(ple_dim)));
            ple_tok = ggml_reshape_1d(ctx, ple_tok, total_ple_dim);

            if (ple_model_proj_data != nullptr && ple_model_proj_norm_data != nullptr)
            {
                // Hidden-projection component: rmsnorm((hidden @ proj)/sqrt(hidden), norm).
                ple_model_proj_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_model_proj_type),
                    ple_model_proj_ne0, ple_model_proj_ne1);
                ple_model_proj_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ple_dim);
                ggml_tensor* proj = ggml_mul_mat(ctx, ple_model_proj_t, current);   // [total_ple_dim]
                proj = ggml_scale(ctx, proj, 1.0f / sqrtf(static_cast<float>(hidden_size)));
                // Per-layer RMSNorm over ple_dim: view [total_ple_dim] as
                // [ple_dim, num_layers], norm rows, scale by the norm weight.
                ggml_tensor* proj_r = ggml_reshape_2d(ctx, ggml_cont(ctx, proj), ple_dim, num_layers);
                proj_r = ggml_mul(ctx, ggml_rms_norm(ctx, proj_r, eps), ple_model_proj_norm_t);
                proj = ggml_reshape_1d(ctx, proj_r, total_ple_dim);
                // combined = (proj + tok) / sqrt(2)
                ple_input = ggml_scale(ctx, ggml_add(ctx, proj, ple_tok), 1.0f / sqrtf(2.0f));
            }
            else
            {
                ple_input = ple_tok;
            }
        }
        else if (ple_data != nullptr && ple_dim > 0)
        {
            ple_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_layers * ple_dim);
        }

        if (can_persist)
        {
            ggml_set_input(current);
            ggml_set_input(pos_tensor);
            if (ple_gather && ple_ids_t != nullptr) ggml_set_input(ple_ids_t);
            else if (ple_input != nullptr) ggml_set_input(ple_input);
        }

        struct LayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* k_w;   // separate K weight (mixed-quant); null when fused
            ggml_tensor* v_w;   // separate V weight (mixed-quant); null when fused
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
            // Mixed-quant layers carry separate K/V weights (qkv_w then holds Q
            // only). Shared layers never run their own K/V projection.
            const bool separate_qkv = (!info.isShared && k_arr != nullptr && k_arr[l] != nullptr);
            if (separate_qkv)
            {
                lt.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(k_type_arr[l]), k_ne0_arr[l], k_ne1_arr[l]);
                lt.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(v_type_arr[l]), v_ne0_arr[l], v_ne1_arr[l]);
            }
            else
            {
                lt.k_w = nullptr;
                lt.v_w = nullptr;
            }
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
            // ple_input exists in BOTH PLE modes: uploaded (ple_data) or gathered
            // in-kernel (ple_gather); the per-layer injection weights are needed
            // whenever the graph carries per-layer embeddings at all.
            if ((ple_data != nullptr || ple_gather) && ple_gate_arr != nullptr && ple_gate_arr[l] != nullptr)
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
                // 2. QKV projection. Mixed-quant layers (lt.k_w != nullptr) run
                // three separate matmuls because Q/K/V carry different ggml
                // types and cannot share one fused weight; otherwise a single
                // fused attn_qkv matmul is sliced into Q/K/V.
                ggml_tensor* q_raw;
                ggml_tensor* k_raw;
                ggml_tensor* v_raw;
                if (lt.k_w != nullptr)
                {
                    q_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.qkv_w, normed_2d), info.qDim);
                    k_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.k_w, normed_2d), info.kDim);
                    v_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.v_w, normed_2d), info.kDim);
                }
                else
                {
                    ggml_tensor* qkv_flat = ggml_reshape_1d(ctx,
                        ggml_mul_mat(ctx, lt.qkv_w, normed_2d), info.qDim + 2 * info.kDim);
                    q_raw = ggml_view_1d(ctx, qkv_flat, info.qDim, 0);
                    k_raw = ggml_view_1d(ctx, qkv_flat, info.kDim,
                        static_cast<std::size_t>(info.qDim) * sizeof(float));
                    v_raw = ggml_view_1d(ctx, qkv_flat, info.kDim,
                        static_cast<std::size_t>(info.qDim + info.kDim) * sizeof(float));
                }

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
                if (can_persist)
                {
                    // KV write via set_rows: the write row is an I64 INPUT, so the
                    // graph topology is identical token-to-token (capturable). The
                    // attention reads a FIXED padded window [0, pwindow[l]) with an
                    // F16 mask INPUT zeroing valid positions and -inf'ing padding.
                    const int win = pwindow[l];
                    ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
                    ggml_set_input(kv_idx);
                    layer_kv_index[l] = kv_idx;
                    lt.k_cpy = ggml_set_rows(ctx, lt.k_cached_t, k_write, kv_idx);
                    lt.v_cpy = ggml_set_rows(ctx, lt.v_cached_t, v_write, kv_idx);
                    ggml_tensor* mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, win, 1, 1, 1);
                    ggml_set_input(mask);
                    layer_attn_mask[l] = mask;
                    k_full = view_kv_cache_window(ctx, lt.k_cached_t, info.hd, info.cacheSize, info.kvHeads, 0, win, kv_cache_type);
                    v_full = view_kv_cache_window(ctx, lt.v_cached_t, info.hd, info.cacheSize, info.kvHeads, 0, win, kv_cache_type);
                }
                else
                {
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
                }
                if (k_full == nullptr || v_full == nullptr)
                {
                    set_last_error("Failed to create Gemma4 KV cache views.");
                    if (can_persist) ggml_free(ctx);
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

        // Output: either the bare hidden state, or — when folding — the final
        // RMSNorm * output_norm, the lm_head projection and the tanh logit
        // softcap, so the 262K-vocab logits are part of the captured replay.
        ggml_tensor* lm_head_t = nullptr;
        ggml_tensor* final_norm_t = nullptr;
        ggml_tensor* hidden_out;
        ggml_tensor* out_hidden;
        if (fold)
        {
            lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
            final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);
            ggml_tensor* fn_2d = ggml_reshape_2d(ctx, fn, hidden_size, 1);
            ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn_2d);   // [vocab, 1]
            if (logit_softcap > 0.0f)
            {
                logits = ggml_scale(ctx, logits, 1.0f / logit_softcap);
                logits = ggml_tanh(ctx, logits);
                logits = ggml_scale(ctx, logits, logit_softcap);
            }
            logits = ggml_reshape_1d(ctx, logits, vocab_size);
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab_size);
            out_hidden = ggml_cpy(ctx, logits, hidden_out);
        }
        else
        {
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            out_hidden = ggml_cpy(ctx, hidden, hidden_out);
        }
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

        auto t_graph_done = std::chrono::high_resolution_clock::now();

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
            if (lt.k_w != nullptr)
            {
                bind_or_mark(lt.k_w, k_arr[l], static_cast<std::size_t>(k_bytes_arr[l]), true);
                bind_or_mark(lt.v_w, v_arr[l], static_cast<std::size_t>(v_bytes_arr[l]), true);
            }
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

        if (fold)
        {
            bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
            bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(hidden_size) * sizeof(float), true);
        }

        if (ple_gather)
        {
            bind_or_mark(ple_table_t, const_cast<void*>(ple_token_embd_data), static_cast<std::size_t>(ple_token_embd_bytes), true);
            if (ple_model_proj_t != nullptr)
                bind_or_mark(ple_model_proj_t, const_cast<void*>(ple_model_proj_data), static_cast<std::size_t>(ple_model_proj_bytes), true);
            if (ple_model_proj_norm_t != nullptr)
                bind_or_mark(ple_model_proj_norm_t, const_cast<void*>(static_cast<const void*>(ple_model_proj_norm_data)), static_cast<std::size_t>(ple_dim) * sizeof(float), true);
        }

        auto t_bind_done = std::chrono::high_resolution_clock::now();

        // Allocate backend buffer. Reuse a persistent compute buffer across
        // decode steps instead of allocating a fresh one every token (llama.cpp
        // amortizes this via a persistent graph allocator; we mirror that). The
        // host_read_barrier below drains the prior step's GPU work before this
        // graph runs, so reusing the buffer is race-free.
        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            // STABLE addresses for CUDA-graph capture: every tensor gets its own
            // slot (no gallocr lifetime packing, whose plan can move addresses).
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("Gemma4 model decode: failed to allocate persist backend buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else if (!alloc_ctx_tensors_reuse(ctx))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("Failed to allocate backend buffer for Gemma4 model decode.");
                return 0;
            }
        }

        auto t_alloc_done = std::chrono::high_resolution_clock::now();

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

        if (ple_gather && ple_ids_t != nullptr)
        {
            std::int32_t ple_id_val = ple_token_id;
            ggml_backend_tensor_set(ple_ids_t, &ple_id_val, 0, sizeof(std::int32_t));
        }
        else if (ple_input != nullptr && ple_data != nullptr)
            ggml_backend_tensor_set(ple_input, ple_data, 0,
                static_cast<std::size_t>(num_layers) * ple_dim * sizeof(float));

        if (can_persist)
        {
            // Per-layer set_rows write rows + padding masks (the per-token inputs
            // that the reuse fast-path updates before each replay).
            for (int l = 0; l < num_layers; l++)
            {
                if (layer_kv_index[l] != nullptr)
                    ggml_backend_tensor_set(layer_kv_index[l], &pwrite[l], 0, sizeof(std::int64_t));
                if (!li[l].isShared && layer_attn_mask[l] != nullptr)
                {
                    std::vector<ggml_fp16_t> md;
                    fill_flash_attn_mask(md, pwindow[l], pvalid[l]);
                    ggml_backend_tensor_set(layer_attn_mask[l], md.data(), 0, md.size() * sizeof(ggml_fp16_t));
                }
            }
        }

        auto t_upload_done = std::chrono::high_resolution_clock::now();

        // Execute single graph
        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for Gemma4 model decode.");
            if (can_persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
            return 0;
        }

        auto t_compute_done = std::chrono::high_resolution_clock::now();

        static const bool g4_ple_debug = std::getenv("TS_G4_PLE_DEBUG") != nullptr;
        if (g4_ple_debug && ple_input != nullptr)
        {
            std::vector<float> dbg(static_cast<std::size_t>(total_ple_dim));
            ggml_backend_synchronize(g_backend);
            ggml_backend_tensor_get(ple_input, dbg.data(), 0, dbg.size() * sizeof(float));
            double sum = 0; for (float v : dbg) sum += v;
            fprintf(stderr, "[g4-ple] mode=%s pos=%d tok=%d nodes=%d first=[%.6f %.6f %.6f %.6f %.6f %.6f] sum=%.4f l1=[%.6f %.6f] l41=[%.6f %.6f]\n",
                ple_gather ? "gather" : "upload", position, ple_token_id, ggml_graph_n_nodes(graph),
                dbg[0], dbg[1], dbg[2], dbg[3], dbg[4], dbg[5], sum,
                dbg[256], dbg[257], dbg[41*256], dbg[41*256+1]);
            fflush(stderr);
        }

        // Download hidden state (async blit on Metal in async mode), or the
        // folded logits[vocab] when the lm_head was folded into the graph.
        const int g4_out_count = fold ? vocab_size : hidden_size;
        void* g4_out_data = fold ? logits_data : hidden_data;
        finalize_compute_with_download(hidden_out, g4_out_data, static_cast<std::size_t>(g4_out_count) * sizeof(float));
        if (can_persist) host_read_barrier();

        if (can_persist && g4dc != nullptr)
        {
            // Keep the ctx/graph/buffer + input handles alive for capture+replay,
            // in this request's own pool slot.
            g4dc->ctx = ctx;
            g4dc->buffer = persist_buf;
            g4dc->graph = graph;
            g4dc->hidden_in = current;
            g4dc->hidden_out = hidden_out;
            g4dc->pos_tensor = pos_tensor;
            // In gather mode ple_input is a computed node — only the token id is a
            // per-token input, so that's what the REUSE fast-path refreshes.
            g4dc->ple_input = ple_gather ? nullptr : ple_input;
            g4dc->ple_ids = ple_gather ? ple_ids_t : nullptr;
            g4dc->kv_index = layer_kv_index;
            g4dc->attn_mask = layer_attn_mask;
            g4dc->layer_window = pwindow;
            g4dc->sig_disc = g4_sig;
            g4dc->sig_kcache0 = g4_kc0;
            g4dc->num_layers = num_layers;
            g4dc->hidden_size = hidden_size;
            g4dc->ple_dim = ple_dim;
            g4dc->ple_gather = ple_gather;
            g4dc->folded = fold;
            g4dc->out_count = g4_out_count;
            g4dc->valid = true;
        }
        if (g4_fd_timing)
        {
            auto t_end = std::chrono::high_resolution_clock::now();
            auto ms = [](auto a, auto b){ return std::chrono::duration<double, std::milli>(b - a).count(); };
            fprintf(stderr, "[g4-fd] BUILD total=%.2f ms (persist=%d upload_list=%zu graph=%.2f bind=%.2f alloc=%.2f upload=%.2f compute=%.2f download=%.2f)\n",
                ms(t_build_start, t_end), can_persist ? 1 : 0, upload_list.size(),
                ms(t_build_start, t_graph_done), ms(t_graph_done, t_bind_done),
                ms(t_bind_done, t_alloc_done),
                ms(t_alloc_done, t_upload_done), ms(t_upload_done, t_compute_done),
                ms(t_compute_done, t_end));
            fflush(stderr);
        }

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

// Drop the persistent (CUDA-graph-captured) Gemma4 decode graph. The captured
// graph pins ggml-cuda's compute-pool scratch addresses and the KV-cache device
// buffers; a prefill (which grows the pool) or a KV reset/grow can move those,
// so the C# caller drops the cache before any prefill and on ResetKVCache. The
// next decode rebuilds + re-captures against the current pool state. No-op when
// persist mode is off (the cache is never populated).
TSG_EXPORT void TSGgml_Gemma4ResetDecodeCache()
{
    g_g4dc_pool.reset_all();
}

// PERSIST / CUDA-graph capture pool for the token-batched dense decode
// (TSGgml_Gemma4ModelDecodeBatched). Analogue of the single-stream g_g4dc_pool,
// but each entry's identity is the SET of N per-request KV caches (sig_kc, in the
// C#-canonicalised request-id order) + N + per-layer padded window. A recurring
// concurrent request-set replays its captured graph (set_rows KV write + fixed
// padded window + F16 mask inputs => identical topology + stable addresses) so
// ggml-cuda's CUDA-graph capture engages. Dropped on prefill / KV reset
// (TSGgml_Gemma4ResetBatchedDecodeCache) since those move device addresses.
namespace
{
    constexpr int kG4BatchPersistKvStride = 256;

    struct G4BatchedDecodeCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_gallocr_t galloc = nullptr;      // dedicated packed allocator (MoE batched: VRAM-frugal capture)
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_in = nullptr;
        ggml_tensor* pos_tensor = nullptr;
        ggml_tensor* logits_out = nullptr;
        std::vector<ggml_tensor*> kv_index;   // [num_layers * n_seqs] I64 set_rows write rows
        std::vector<ggml_tensor*> attn_mask;  // [num_layers] F16 [win,1,1,n_seqs]
        std::vector<int> layer_window;        // [num_layers]
        const void* sig_disc = nullptr;
        std::vector<const void*> sig_kc;      // [n_seqs] layer-0 K cache ptrs (canonical order)
        int num_layers = 0, hidden_size = 0, n_seqs = 0, vocab = 0;

        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (galloc != nullptr) { ggml_gallocr_free(galloc); galloc = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_in = pos_tensor = logits_out = nullptr;
            kv_index.clear(); attn_mask.clear(); layer_window.clear();
            sig_disc = nullptr; sig_kc.clear();
            num_layers = hidden_size = n_seqs = vocab = 0;
        }
    };

    constexpr int kG4BatchedMaxCaches = 4;
    struct G4BatchedDecodeCachePool
    {
        G4BatchedDecodeCache entries[kG4BatchedMaxCaches];
        std::uint64_t used[kG4BatchedMaxCaches] = {};
        std::uint64_t clock = 0;

        G4BatchedDecodeCache* find(const void* sig, const std::vector<const void*>& kc, int n)
        {
            for (int i = 0; i < kG4BatchedMaxCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].n_seqs == n && entries[i].sig_kc == kc)
                { used[i] = ++clock; return &entries[i]; }
            return nullptr;
        }

        G4BatchedDecodeCache& claim(const void* sig, const std::vector<const void*>& kc, int n)
        {
            for (int i = 0; i < kG4BatchedMaxCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].n_seqs == n && entries[i].sig_kc == kc)
                { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            for (int i = 0; i < kG4BatchedMaxCaches; i++)
                if (!entries[i].valid) { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            int lru = 0;
            for (int i = 1; i < kG4BatchedMaxCaches; i++) if (used[i] < used[lru]) lru = i;
            entries[lru].reset(); used[lru] = ++clock; return entries[lru];
        }

        void reset_all() { for (auto& e : entries) e.reset(); }
    };
    G4BatchedDecodeCachePool g_g4batched_pool;
}

// ============================================================================
// TRUE TOKEN-BATCHED dense decode (N concurrent sequences, one token each, in
// ONE ggml graph + ONE compute buffer). This is the llama-parity concurrency
// path: where TSGgml_Gemma4ModelDecode (above) decodes ONE token and the engine
// round-robins N serial calls for N concurrent requests (N weight loads ->
// aggregate ~= single-stream), this kernel processes all N decode tokens
// together so every weight is loaded ONCE and applied to N tokens. Decode is
// memory-bandwidth bound, so that amortisation is the win (and one compute
// buffer instead of N fixes the per-request-buffer VRAM blowup).
//
// Each sequence owns its OWN KV cache (the per-request holders the C# engine
// manages); k_cache_arr / v_cache_arr are sized num_layers * n_seqs, indexed
// [layer * n_seqs + seq]. positions[seq] is each sequence's current length.
// Hidden in/out and logits are packed column-major [.., n_seqs].
//
// v1 scope (correctness-first): DENSE only (no MoE), no PLE (ple_dim==0), no
// KV-donor/shared layers, requires the folded lm_head, and the NO-WRAP regime
// (every sequence's total length <= every layer's cache size, so each query
// attends [0, total) with a simple per-sequence padding mask). The C# caller
// (Gemma4Model.TryForwardBatchedFusedDecode) enforces these and otherwise falls
// back to the round-robin per-sequence path. Non-persist (no CUDA-graph capture)
// in v1; capture is a follow-up once correctness is proven.
//
// Attention is one ggml_flash_attn_ext with batch dim ne3 = n_seqs: Q is
// reshaped to [head_dim, 1, num_heads, n_seqs]; each sequence's KV window is
// cont()'d and concat()'d along ne3 into [head_dim, win, kv_heads, n_seqs]; the
// mask is [win, 1, 1, n_seqs] with column s zeroing [0, total_s) and -inf'ing
// the padding. Everything else (norms, projections, FFN, lm_head) operates on
// [hidden, n_seqs] for free.
// ============================================================================
TSG_EXPORT int TSGgml_Gemma4ModelDecodeBatched(
    float* hidden_data, int hidden_size, int num_layers, int n_seqs,
    void** attn_norm_arr,
    void** qkv_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr,
    void** post_attn_norm_arr,
    void** ffn_norm_arr,
    void** gu_arr, void** down_arr,
    void** post_ffn_norm_arr,
    // Per-(layer,seq) KV caches: k_cache_arr[layer * n_seqs + seq].
    void** k_cache_arr, void** v_cache_arr,
    int* head_dim_arr,
    int* kv_heads_arr,
    int* cache_size_arr,
    int* is_local_arr,
    float* rope_base_arr,
    float* layer_scalar_arr,
    int* qkv_type_arr, std::int64_t* qkv_ne0_arr, std::int64_t* qkv_ne1_arr, std::int64_t* qkv_bytes_arr,
    int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    int* gu_type_arr, std::int64_t* gu_ne0_arr, std::int64_t* gu_ne1_arr, std::int64_t* gu_bytes_arr,
    int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    int num_heads,
    const int* positions,           // [n_seqs]
    float eps, int sliding_window,
    float* rope_freq_factors, int rope_freq_factors_len,
    int* rope_n_dims_arr,
    int kv_cache_type,
    // Separate K/V projection weights for mixed-quant layers (null => fused qkv).
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    // Folded final-norm + lm_head (required for this kernel).
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, float logit_softcap)
{
    try
    {
        if (!ensure_backend())
            return 0;
        if (hidden_data == nullptr || n_seqs <= 0 || num_layers <= 0)
        {
            set_last_error("Gemma4 batched decode: invalid arguments.");
            return 0;
        }
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;
        if (!fold)
        {
            set_last_error("Gemma4 batched decode: folded lm_head required.");
            return 0;
        }

        struct LayerInfo { int hd; int kvHeads; int qDim; int kDim; int cacheSize; bool isLocal; int win; };
        std::vector<LayerInfo> li(num_layers);

        int maxTotal = 0;
        for (int s = 0; s < n_seqs; s++)
            maxTotal = std::max(maxTotal, positions[s] + 1);

        // Persist (CUDA-graph capture) gate: identical to the single-stream
        // g_g4dc. Capturable when on CUDA and TS_GEMMA4_FD_PERSIST != 0.
        static const bool g4b_timing = std::getenv("TS_GEMMA4_FD_TIMING") != nullptr;
        static const bool g4b_persist = []{ const char* e = std::getenv("TS_GEMMA4_FD_PERSIST"); return e == nullptr || e[0] != '0'; }();
        bool can_persist = g4b_persist && g_backend_type == BACKEND_TYPE_CUDA;

        auto roundup_stride = [](int v){ return ((v + kG4BatchPersistKvStride - 1) / kG4BatchPersistKvStride) * kG4BatchPersistKvStride; };
        for (int l = 0; l < num_layers; l++)
        {
            auto& info = li[l];
            info.hd = head_dim_arr[l];
            info.kvHeads = kv_heads_arr[l];
            info.qDim = num_heads * info.hd;
            info.kDim = info.kvHeads * info.hd;
            info.cacheSize = cache_size_arr[l];
            info.isLocal = is_local_arr[l] != 0;
            // v1 NO-WRAP gate: every sequence must fit the cache window so each
            // query attends a contiguous [0, total) prefix.
            if (info.cacheSize <= 0 || maxTotal > info.cacheSize)
            {
                set_last_error("Gemma4 batched decode: sequence exceeds cache window (wrap not supported in v1).");
                return 0;
            }
            // Persist: pad the window to a 256-stride (bounded by the cache) so the
            // graph topology is identical token-to-token (only changes every 256
            // tokens), which is what lets ggml-cuda capture engage. Non-persist
            // uses the tight flash_attn length.
            info.win = can_persist
                ? std::min(info.cacheSize, std::max(roundup_stride(maxTotal), flash_attn_kv_length(maxTotal, info.cacheSize, info.hd)))
                : flash_attn_kv_length(maxTotal, info.cacheSize, info.hd);
        }

        // Per-seq attention-mask fill: column s zeroes [0, pos_s+1) (clamped to
        // win), -inf elsewhere. Shared by the reuse fast-path and the build path.
        auto fill_batched_mask = [&](std::vector<ggml_fp16_t>& md, int win)
        {
            md.assign(static_cast<std::size_t>(win) * n_seqs, ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
            for (int s = 0; s < n_seqs; s++)
            {
                const int valid = std::min(positions[s] + 1, win);
                for (int k = 0; k < valid; k++)
                    md[static_cast<std::size_t>(s) * win + k] = static_cast<ggml_fp16_t>(0);
            }
        };

        // Pool identity: model instance + the canonical-order set of N layer-0 K
        // caches + N + per-layer window. (C# canonicalises the request order.)
        const void* sig_disc = attn_norm_arr[0];
        std::vector<const void*> sig_kc(n_seqs);
        for (int s = 0; s < n_seqs; s++) sig_kc[s] = k_cache_arr[s];   // layer 0, seq s
        std::vector<int> winvec(num_layers);
        for (int l = 0; l < num_layers; l++) winvec[l] = li[l].win;

        // ---- reuse fast-path: replay this request-set's captured graph ----
        G4BatchedDecodeCache* dc = can_persist ? g_g4batched_pool.find(sig_disc, sig_kc, n_seqs) : nullptr;
        if (dc != nullptr && dc->graph != nullptr &&
            dc->num_layers == num_layers && dc->hidden_size == hidden_size &&
            dc->vocab == vocab_size && dc->layer_window == winvec)
        {
            host_read_barrier();
            ggml_backend_tensor_set(dc->hidden_in, hidden_data, 0, static_cast<std::size_t>(hidden_size) * n_seqs * sizeof(float));
            ggml_backend_tensor_set(dc->pos_tensor, positions, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));
            for (int l = 0; l < num_layers; l++)
            {
                for (int s = 0; s < n_seqs; s++)
                {
                    std::int64_t row = positions[s];
                    ggml_backend_tensor_set(dc->kv_index[l * n_seqs + s], &row, 0, sizeof(std::int64_t));
                }
                std::vector<ggml_fp16_t> md;
                fill_batched_mask(md, li[l].win);
                ggml_backend_tensor_set(dc->attn_mask[l], md.data(), 0, md.size() * sizeof(ggml_fp16_t));
            }
            if (ggml_backend_graph_compute(g_backend, dc->graph) != GGML_STATUS_SUCCESS)
            {
                set_last_error("Gemma4 batched decode: replay graph compute failed.");
                dc->reset();
                return 0;
            }
            finalize_compute_with_download(dc->logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_seqs * sizeof(float));
            host_read_barrier();
            if (g4b_timing) { fprintf(stderr, "[g4-batched] replay N=%d\n", n_seqs); fflush(stderr); }
            clear_last_error();
            return 1;
        }

        // ---- build context: persist = raw no_alloc ctx kept alive in the pool
        // (stable addresses for capture); non-persist = pooled (<=32MB). ----
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("Gemma4 batched decode: failed to init persist ctx."); return 0; }
        }
        else
        {
            if (!context.init(ctx_size))
            {
                set_last_error("Gemma4 batched decode: failed to create ggml context.");
                return 0;
            }
            ctx = context.value;
        }

        ggml_tensor* current = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_seqs);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
        if (can_persist) { ggml_set_input(current); ggml_set_input(pos_tensor); }

        ggml_tensor* freq_factors_t = nullptr;
        if (rope_freq_factors != nullptr && rope_freq_factors_len > 0)
            freq_factors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rope_freq_factors_len);

        // Per-(layer,seq) I64 set_rows write-row inputs (persist only); kept for
        // the pool entry so the reuse path can refresh them each replay.
        std::vector<ggml_tensor*> kv_index_all(static_cast<std::size_t>(num_layers) * n_seqs, nullptr);

        struct LayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* k_w; ggml_tensor* v_w;
            ggml_tensor* q_norm_w; ggml_tensor* k_norm_w;
            ggml_tensor* o_w;
            ggml_tensor* post_attn_norm_w;
            ggml_tensor* ffn_norm_w;
            ggml_tensor* gu_w; ggml_tensor* down_w;
            ggml_tensor* post_ffn_norm_w;
            std::vector<ggml_tensor*> k_cached;   // per seq
            std::vector<ggml_tensor*> v_cached;   // per seq
            std::vector<ggml_tensor*> k_cpy;      // per seq (KV write op)
            std::vector<ggml_tensor*> v_cpy;
            ggml_tensor* attn_mask;
            std::vector<ggml_fp16_t> attn_mask_data;
        };
        std::vector<LayerTensors> layers(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];

            lt.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkv_type_arr[l]), qkv_ne0_arr[l], qkv_ne1_arr[l]);
            const bool separate_qkv = (k_arr != nullptr && k_arr[l] != nullptr);
            if (separate_qkv)
            {
                lt.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(k_type_arr[l]), k_ne0_arr[l], k_ne1_arr[l]);
                lt.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(v_type_arr[l]), v_ne0_arr[l], v_ne1_arr[l]);
            }
            else { lt.k_w = nullptr; lt.v_w = nullptr; }
            lt.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type_arr[l]), o_ne0_arr[l], o_ne1_arr[l]);
            lt.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gu_type_arr[l]), gu_ne0_arr[l], gu_ne1_arr[l]);
            lt.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type_arr[l]), down_ne0_arr[l], down_ne1_arr[l]);
            lt.post_ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            lt.k_cached.resize(n_seqs);
            lt.v_cached.resize(n_seqs);
            lt.k_cpy.resize(n_seqs, nullptr);
            lt.v_cpy.resize(n_seqs, nullptr);
            for (int s = 0; s < n_seqs; s++)
            {
                lt.k_cached[s] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, info.cacheSize, info.kvHeads);
                lt.v_cached[s] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, info.cacheSize, info.kvHeads);
            }
            lt.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, info.win, 1, 1, n_seqs);
            if (can_persist) ggml_set_input(lt.attn_mask);
        }

        // lm_head + final norm
        ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
        ggml_tensor* final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        // ---- build graph ----
        ggml_tensor* hidden = current;   // [hidden_size, n_seqs]
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];
            float rope_base = rope_base_arr[l];
            int rope_dims = rope_n_dims_arr[l];
            ggml_tensor* rope_ff = info.isLocal ? nullptr : freq_factors_t;

            // 1. attn norm
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), lt.attn_norm_w);   // [H, N]

            // 2. QKV projection -> [qDim, N] / [kDim, N]
            ggml_tensor* q_raw; ggml_tensor* k_raw; ggml_tensor* v_raw;
            if (lt.k_w != nullptr)
            {
                q_raw = ggml_mul_mat(ctx, lt.qkv_w, normed);   // [qDim, N]
                k_raw = ggml_mul_mat(ctx, lt.k_w, normed);     // [kDim, N]
                v_raw = ggml_mul_mat(ctx, lt.v_w, normed);     // [kDim, N]
            }
            else
            {
                ggml_tensor* qkv = ggml_mul_mat(ctx, lt.qkv_w, normed);   // [qDim+2kDim, N]
                q_raw = ggml_view_2d(ctx, qkv, info.qDim, n_seqs, qkv->nb[1], 0);
                k_raw = ggml_view_2d(ctx, qkv, info.kDim, n_seqs, qkv->nb[1],
                    static_cast<std::size_t>(info.qDim) * sizeof(float));
                v_raw = ggml_view_2d(ctx, qkv, info.kDim, n_seqs, qkv->nb[1],
                    static_cast<std::size_t>(info.qDim + info.kDim) * sizeof(float));
            }

            // 3. per-head Q/K norm + V norm -> [hd, heads, N]
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, q_raw), info.hd, num_heads, n_seqs);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, k_raw), info.hd, info.kvHeads, n_seqs);
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, v_raw), info.hd, info.kvHeads, n_seqs);
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_3d, eps), lt.q_norm_w);
            ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_3d, eps), lt.k_norm_w);
            ggml_tensor* v_normed = ggml_rms_norm(ctx, v_3d, eps);

            // 4. RoPE (pos[seq] applied per ne2 slice)
            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_normed, pos_tensor, rope_ff,
                rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);   // [hd, num_heads, N]
            ggml_tensor* k_rope = ggml_rope_ext(ctx, k_normed, pos_tensor, rope_ff,
                rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);   // [hd, kvHeads, N]

            // 5. per-seq KV write + windowed read; concat windows along ne3
            ggml_tensor* k_cat = nullptr;
            ggml_tensor* v_cat = nullptr;
            for (int s = 0; s < n_seqs; s++)
            {
                const int cachePos = positions[s];   // no-wrap: index == position
                // slice this seq's new K/V: [hd, kvHeads, 1] -> permute -> [hd, 1, kvHeads]
                ggml_tensor* k_s = ggml_view_3d(ctx, k_rope, info.hd, info.kvHeads, 1,
                    k_rope->nb[1], k_rope->nb[2], static_cast<std::size_t>(s) * k_rope->nb[2]);
                ggml_tensor* v_s = ggml_view_3d(ctx, v_normed, info.hd, info.kvHeads, 1,
                    v_normed->nb[1], v_normed->nb[2], static_cast<std::size_t>(s) * v_normed->nb[2]);
                ggml_tensor* k_write = ggml_cont(ctx, ggml_permute(ctx, k_s, 0, 2, 1, 3));   // [hd, 1, kvHeads]
                ggml_tensor* v_write = ggml_cont(ctx, ggml_permute(ctx, v_s, 0, 2, 1, 3));   // [hd, 1, kvHeads]
                // k_src/v_src is the tensor the window read sees. For persist we
                // read from the set_rows RESULT (full-cache tensor), which gives a
                // real graph edge write->read so the topological sort can never
                // place the window read before this seq's KV write (without the
                // edge, ggml relied on insertion order, which reordered the last
                // sequence's read ahead of its write at N>=4 -> stale K/V).
                ggml_tensor* k_src; ggml_tensor* v_src;
                if (can_persist)
                {
                    // set_rows write (row = an I64 INPUT) keeps the graph topology
                    // identical token-to-token so CUDA-graph capture engages.
                    ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
                    ggml_set_input(kv_idx);
                    kv_index_all[static_cast<std::size_t>(l) * n_seqs + s] = kv_idx;
                    lt.k_cpy[s] = ggml_set_rows(ctx, lt.k_cached[s], k_write, kv_idx);
                    lt.v_cpy[s] = ggml_set_rows(ctx, lt.v_cached[s], v_write, kv_idx);
                    k_src = lt.k_cpy[s];   // full-cache result of the write
                    v_src = lt.v_cpy[s];
                }
                else
                {
                    ggml_tensor* k_dst = ggml_view_3d(ctx, lt.k_cached[s], info.hd, 1, info.kvHeads,
                        lt.k_cached[s]->nb[1], lt.k_cached[s]->nb[2],
                        static_cast<std::size_t>(cachePos) * lt.k_cached[s]->nb[1]);
                    ggml_tensor* v_dst = ggml_view_3d(ctx, lt.v_cached[s], info.hd, 1, info.kvHeads,
                        lt.v_cached[s]->nb[1], lt.v_cached[s]->nb[2],
                        static_cast<std::size_t>(cachePos) * lt.v_cached[s]->nb[1]);
                    lt.k_cpy[s] = ggml_cpy(ctx, k_write, k_dst);
                    lt.v_cpy[s] = ggml_cpy(ctx, v_write, v_dst);
                    k_src = lt.k_cached[s];
                    v_src = lt.v_cached[s];
                }

                // windowed read [0, win). Pass the STRIDED view straight to concat
                // (its non-contiguous kernel reads via strides) — the previous
                // ggml_cont here copied the whole padded window per seq/layer EVERY
                // step (only 1 KV row actually changed), a big redundant memcpy.
                // ggml_concat already produces a contiguous output for flash.
                ggml_tensor* k_win = view_kv_cache_window(ctx, k_src, info.hd, info.cacheSize, info.kvHeads, 0, info.win, kv_cache_type);
                ggml_tensor* v_win = view_kv_cache_window(ctx, v_src, info.hd, info.cacheSize, info.kvHeads, 0, info.win, kv_cache_type);
                k_cat = (k_cat == nullptr) ? k_win : ggml_concat(ctx, k_cat, k_win, 3);
                v_cat = (v_cat == nullptr) ? v_win : ggml_concat(ctx, v_cat, v_win, 3);
            }

            // 6. flash attention, batched over ne3 = n_seqs
            ggml_tensor* q4d = ggml_reshape_4d(ctx, ggml_cont(ctx, q_rope), info.hd, 1, num_heads, n_seqs);
            ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q4d, k_cat, v_cat, lt.attn_mask, 1.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);

            // 7. O projection -> [H, N]
            ggml_tensor* attn_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, attn_out), info.qDim, n_seqs);
            ggml_tensor* o_out = ggml_mul_mat(ctx, lt.o_w, attn_2d);   // [H, N]

            // 8. post-attn norm + residual
            ggml_tensor* post_attn = ggml_mul(ctx, ggml_rms_norm(ctx, o_out, eps), lt.post_attn_norm_w);
            ggml_tensor* residual1 = ggml_add(ctx, hidden, post_attn);

            // 9. FFN
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), lt.ffn_norm_w);
            std::int64_t inter = gu_ne1_arr[l] / 2;
            ggml_tensor* gu = ggml_mul_mat(ctx, lt.gu_w, ffn_normed);   // [2*inter, N]
            ggml_tensor* gate = ggml_view_2d(ctx, gu, inter, n_seqs, gu->nb[1], 0);
            ggml_tensor* up = ggml_view_2d(ctx, gu, inter, n_seqs, gu->nb[1],
                static_cast<std::size_t>(inter) * sizeof(float));
            ggml_tensor* ffn_hidden = ggml_mul(ctx, ggml_gelu(ctx, ggml_cont(ctx, gate)), ggml_cont(ctx, up));   // [inter, N]
            ggml_tensor* down_out = ggml_mul_mat(ctx, lt.down_w, ffn_hidden);   // [H, N]

            // 10. post-FFN norm + residual
            ggml_tensor* post_ffn = ggml_mul(ctx, ggml_rms_norm(ctx, down_out, eps), lt.post_ffn_norm_w);
            ggml_tensor* residual2 = ggml_add(ctx, residual1, post_ffn);

            float scalar = layer_scalar_arr[l];
            if (std::fabs(scalar - 1.0f) > 1e-6f)
                residual2 = ggml_scale(ctx, residual2, scalar);

            hidden = residual2;
        }

        // final norm + lm_head -> logits [vocab, N]
        ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);
        ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn);   // [vocab, N]
        if (logit_softcap > 0.0f)
        {
            logits = ggml_scale(ctx, logits, 1.0f / logit_softcap);
            logits = ggml_tanh(ctx, logits);
            logits = ggml_scale(ctx, logits, logit_softcap);
        }
        ggml_tensor* logits_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_seqs);
        ggml_tensor* out_op = ggml_cpy(ctx, logits, logits_out);
        ggml_set_output(out_op);

        // build graph: KV writes first, then output
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (256 + 16 * n_seqs) + 1024;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
            for (int s = 0; s < n_seqs; s++)
            {
                if (layers[l].k_cpy[s] != nullptr) ggml_build_forward_expand(graph, layers[l].k_cpy[s]);
                if (layers[l].v_cpy[s] != nullptr) ggml_build_forward_expand(graph, layers[l].v_cpy[s]);
            }
        ggml_build_forward_expand(graph, out_op);

        // ---- bind weights ----
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload) upload_list.push_back({t, data, bytes});
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
                    if (!cacheable) ephemeral_bufs.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS) return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];
            bind_or_mark(lt.qkv_w, qkv_arr[l], static_cast<std::size_t>(qkv_bytes_arr[l]), true);
            if (lt.k_w != nullptr)
            {
                bind_or_mark(lt.k_w, k_arr[l], static_cast<std::size_t>(k_bytes_arr[l]), true);
                bind_or_mark(lt.v_w, v_arr[l], static_cast<std::size_t>(v_bytes_arr[l]), true);
            }
            bind_or_mark(lt.o_w, o_arr[l], static_cast<std::size_t>(o_bytes_arr[l]), true);
            bind_or_mark(lt.gu_w, gu_arr[l], static_cast<std::size_t>(gu_bytes_arr[l]), true);
            bind_or_mark(lt.down_w, down_arr[l], static_cast<std::size_t>(down_bytes_arr[l]), true);
            bind_or_mark(lt.attn_norm_w, attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.post_attn_norm_w, post_attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.ffn_norm_w, ffn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.post_ffn_norm_w, post_ffn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.q_norm_w, q_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);
            bind_or_mark(lt.k_norm_w, k_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);
            for (int s = 0; s < n_seqs; s++)
            {
                bind_or_mark(lt.k_cached[s], k_cache_arr[l * n_seqs + s], kv_cache_bytes(info.kvHeads, info.cacheSize, info.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(lt.v_cached[s], v_cache_arr[l * n_seqs + s], kv_cache_bytes(info.kvHeads, info.cacheSize, info.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
            // per-seq attention mask (host scratch, not cacheable)
            lt.attn_mask_data.assign(static_cast<std::size_t>(info.win) * n_seqs,
                ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
            for (int s = 0; s < n_seqs; s++)
            {
                const int valid = std::min(positions[s] + 1, info.win);
                for (int k = 0; k < valid; k++)
                    lt.attn_mask_data[static_cast<std::size_t>(s) * info.win + k] = static_cast<ggml_fp16_t>(0);
            }
            bind_or_mark(lt.attn_mask, lt.attn_mask_data.data(), lt.attn_mask_data.size() * sizeof(ggml_fp16_t), false);
        }
        bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
        bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(hidden_size) * sizeof(float), true);

        // Persist: every tensor gets its own slot (stable addresses for capture),
        // kept alive in the pool. Non-persist: reuse the pooled compute buffer.
        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("Gemma4 batched decode: failed to allocate persist buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else if (!alloc_ctx_tensors_reuse(ctx))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("Gemma4 batched decode: failed to allocate backend buffer.");
                return 0;
            }
        }

        host_read_barrier();
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);

        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(hidden_size) * n_seqs * sizeof(float));
        ggml_backend_tensor_set(pos_tensor, positions, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));
        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, rope_freq_factors, 0, static_cast<std::size_t>(rope_freq_factors_len) * sizeof(float));
        if (can_persist)
        {
            for (int l = 0; l < num_layers; l++)
                for (int s = 0; s < n_seqs; s++)
                {
                    std::int64_t row = positions[s];
                    ggml_backend_tensor_set(kv_index_all[static_cast<std::size_t>(l) * n_seqs + s], &row, 0, sizeof(std::int64_t));
                }
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Gemma4 batched decode: graph compute failed.");
            if (can_persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
            return 0;
        }

        finalize_compute_with_download(logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_seqs * sizeof(float));

        if (can_persist)
        {
            host_read_barrier();
            G4BatchedDecodeCache& e = g_g4batched_pool.claim(sig_disc, sig_kc, n_seqs);
            e.ctx = ctx; e.buffer = persist_buf; e.graph = graph;
            e.hidden_in = current; e.pos_tensor = pos_tensor; e.logits_out = logits_out;
            e.kv_index = kv_index_all;
            e.attn_mask.resize(num_layers);
            for (int l = 0; l < num_layers; l++) e.attn_mask[l] = layers[l].attn_mask;
            e.layer_window = winvec;
            e.sig_disc = sig_disc; e.sig_kc = sig_kc;
            e.num_layers = num_layers; e.hidden_size = hidden_size; e.n_seqs = n_seqs; e.vocab = vocab_size;
            e.valid = true;
            if (g4b_timing) { fprintf(stderr, "[g4-batched] BUILT persist N=%d win0=%d\n", n_seqs, winvec.empty()?0:winvec[0]); fflush(stderr); }
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Gemma4 batched decode."); return 0; }
}

// Drop all captured token-batched decode graphs. The captured graphs pin
// ggml-cuda's compute-pool scratch + the per-request KV buffers; a prefill (grows
// the pool) or a KV reset/grow can move those, so the C# caller drops the cache
// before any prefill and on ResetKVCache. No-op when persist is off.
TSG_EXPORT void TSGgml_Gemma4ResetBatchedDecodeCache()
{
    g_g4batched_pool.reset_all();
}

// (Relocated above its first use — the batched MoE decode kernel.) MoE layer descriptor:
struct TSGgmlGemma4MoELayerDesc
{
    // --- pointers (host memory) ---
    void* hidden;            // [hidden_size] F32, in/out (residual stream)
    void* attn_norm_w;       // [hidden_size] F32
    void* qkv_w;             // fused QKV, or Q-only when separate_qkv
    void* k_w;               // separate K weight (null unless separate_qkv)
    void* v_w;               // separate V weight (null unless separate_qkv)
    void* q_norm_w;          // [head_dim] F32
    void* k_norm_w;          // [head_dim] F32 (null for shared layers)
    void* o_w;               // attn_output weight
    void* post_attn_norm_w;  // [hidden_size] F32
    void* k_cache;           // [kv_heads, cache_size, head_dim] (donor's for shared)
    void* v_cache;
    void* freq_factors;      // [freq_factors_len] F32 (null for local/no-scaling)
    void* ffn_norm_w;        // [hidden_size] F32
    void* gu_w;              // dense fused gate_up weight [hidden, 2*ff_dense]
    void* down_w;            // dense down weight [ff_dense, hidden]
    void* post_ffw_norm_1_w; // [hidden_size] F32
    void* gate_inp_w;        // router [hidden, num_experts] F32
    void* gate_inp_scale;    // [hidden] F32 (null if absent)
    void* pre_ffw_norm_2_w;  // [hidden_size] F32 (expert input norm)
    void* gate_up_exps;      // stacked experts [hidden, 2*ff_moe, num_experts]
    void* down_exps;         // stacked experts [ff_moe, hidden, num_experts]
    void* down_exps_scale;   // [num_experts] F32 (null if absent)
    void* post_ffw_norm_2_w; // [hidden_size] F32
    void* post_ffw_norm_w;   // [hidden_size] F32

    // --- int64 weight shapes ---
    std::int64_t qkv_ne0, qkv_ne1, qkv_bytes;
    std::int64_t k_ne0, k_ne1, k_bytes;
    std::int64_t v_ne0, v_ne1, v_bytes;
    std::int64_t o_ne0, o_ne1, o_bytes;
    std::int64_t gu_ne0, gu_ne1, gu_bytes;
    std::int64_t down_ne0, down_ne1, down_bytes;
    std::int64_t gue_ne0, gue_ne1, gue_bytes; // per-expert ne0/ne1 + TOTAL bytes
    std::int64_t de_ne0, de_ne1, de_bytes;

    // --- int32 scalars / shapes ---
    std::int32_t struct_bytes;       // sizeof sanity check
    std::int32_t hidden_size;
    std::int32_t num_heads;
    std::int32_t num_kv_heads;
    std::int32_t head_dim;
    std::int32_t cache_size;
    std::int32_t is_local;
    std::int32_t is_shared;
    std::int32_t sliding_window;
    std::int32_t position;
    std::int32_t rope_n_dims;
    std::int32_t kv_cache_type;
    std::int32_t num_experts;
    std::int32_t num_experts_used;
    std::int32_t freq_factors_len;
    std::int32_t qkv_type;
    std::int32_t k_type;
    std::int32_t v_type;
    std::int32_t o_type;
    std::int32_t gu_type;
    std::int32_t down_type;
    std::int32_t gue_type;
    std::int32_t de_type;
    std::int32_t separate_qkv;

    // --- float scalars ---
    float eps;
    float rope_base;
    float inv_sqrt_hidden;     // 1/sqrt(hidden_size) for the router
    float layer_output_scale;
};

namespace { G4BatchedDecodeCachePool g_g4moebatched_pool; }

// ============================================================================
// TRUE TOKEN-BATCHED MoE decode (N concurrent sequences, one token each, one
// ggml graph + one compute buffer). The MoE sibling of
// TSGgml_Gemma4ModelDecodeBatched: identical batched attention (per-request KV,
// flash_attn_ext over ne3=N, set_rows write + fixed padded window for capture)
// but the FFN is the Gemma-4 MoE block (dense shared FFN + in-graph router +
// stacked experts via ggml_mul_mat_id over N tokens — token-parallel, copied
// from the verify kernel). Weights come from the existing per-layer
// TSGgmlGemma4MoELayerDesc array (its k_cache/v_cache/position fields are
// IGNORED — overridden by k_cache_arr[layer*n_seqs+seq] / positions[seq]).
// v1 scope mirrors the dense path: all-MoE, no PLE, no KV-donor, folded lm_head,
// no-wrap regime; CUDA-graph capture via g_g4moebatched_pool.
// ============================================================================
TSG_EXPORT int TSGgml_Gemma4MoEModelDecodeBatched(
    const TSGgmlGemma4MoELayerDesc* layers, int num_layers, int n_seqs,
    void* hidden_data,
    void** k_cache_arr, void** v_cache_arr,   // [layer * n_seqs + seq]
    const int* positions,                     // [n_seqs]
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, float logit_softcap)
{
    try
    {
        if (!ensure_backend()) return 0;
        if (layers == nullptr || hidden_data == nullptr || n_seqs <= 0 || num_layers <= 0)
        { set_last_error("Gemma4 MoE batched decode: invalid arguments."); return 0; }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlGemma4MoELayerDesc)))
        { set_last_error("Gemma4 MoE batched decode: descriptor size mismatch."); return 0; }
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;
        if (!fold) { set_last_error("Gemma4 MoE batched decode: folded lm_head required."); return 0; }

        const int H = layers[0].hidden_size;
        const int num_heads = layers[0].num_heads;
        const float eps = layers[0].eps;
        const int kvType = layers[0].kv_cache_type;

        int maxTotal = 0;
        for (int s = 0; s < n_seqs; s++) maxTotal = std::max(maxTotal, positions[s] + 1);

        static const bool g4mb_timing = std::getenv("TS_GEMMA4_FD_TIMING") != nullptr;
        static const bool g4mb_persist = []{ const char* e = std::getenv("TS_GEMMA4_FD_PERSIST"); return e == nullptr || e[0] != '0'; }();
        bool can_persist = g4mb_persist && g_backend_type == BACKEND_TYPE_CUDA;
        auto roundup_stride = [](int v){ return ((v + kG4BatchPersistKvStride - 1) / kG4BatchPersistKvStride) * kG4BatchPersistKvStride; };

        struct LInfo { int hd, kvH, qDim, cacheSize, win; bool isLocal; };
        std::vector<LInfo> li(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            const auto& d = layers[l];
            if (d.is_shared != 0) { set_last_error("Gemma4 MoE batched decode: KV-donor layers unsupported."); return 0; }
            auto& info = li[l];
            info.hd = d.head_dim; info.kvH = d.num_kv_heads; info.qDim = num_heads * d.head_dim;
            info.cacheSize = d.cache_size; info.isLocal = d.is_local != 0;
            if (info.cacheSize <= 0 || maxTotal > info.cacheSize)
            { set_last_error("Gemma4 MoE batched decode: sequence exceeds cache window (wrap unsupported)."); return 0; }
            info.win = can_persist
                ? std::min(info.cacheSize, std::max(roundup_stride(maxTotal), flash_attn_kv_length(maxTotal, info.cacheSize, info.hd)))
                : flash_attn_kv_length(maxTotal, info.cacheSize, info.hd);
        }

        auto fill_batched_mask = [&](std::vector<ggml_fp16_t>& md, int win)
        {
            md.assign(static_cast<std::size_t>(win) * n_seqs, ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
            for (int s = 0; s < n_seqs; s++)
            {
                const int valid = std::min(positions[s] + 1, win);
                for (int k = 0; k < valid; k++) md[static_cast<std::size_t>(s) * win + k] = static_cast<ggml_fp16_t>(0);
            }
        };

        const void* sig_disc = layers[0].attn_norm_w;
        std::vector<const void*> sig_kc(n_seqs);
        for (int s = 0; s < n_seqs; s++) sig_kc[s] = k_cache_arr[s];
        std::vector<int> winvec(num_layers);
        for (int l = 0; l < num_layers; l++) winvec[l] = li[l].win;

        // ---- reuse fast-path ----
        G4BatchedDecodeCache* dc = can_persist ? g_g4moebatched_pool.find(sig_disc, sig_kc, n_seqs) : nullptr;
        if (dc != nullptr && dc->graph != nullptr &&
            dc->num_layers == num_layers && dc->hidden_size == H &&
            dc->vocab == vocab_size && dc->layer_window == winvec)
        {
            host_read_barrier();
            ggml_backend_tensor_set(dc->hidden_in, hidden_data, 0, static_cast<std::size_t>(H) * n_seqs * sizeof(float));
            ggml_backend_tensor_set(dc->pos_tensor, positions, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));
            for (int l = 0; l < num_layers; l++)
            {
                for (int s = 0; s < n_seqs; s++)
                {
                    std::int64_t row = positions[s];
                    ggml_backend_tensor_set(dc->kv_index[l * n_seqs + s], &row, 0, sizeof(std::int64_t));
                }
                std::vector<ggml_fp16_t> md; fill_batched_mask(md, li[l].win);
                ggml_backend_tensor_set(dc->attn_mask[l], md.data(), 0, md.size() * sizeof(ggml_fp16_t));
            }
            if (ggml_backend_graph_compute(g_backend, dc->graph) != GGML_STATUS_SUCCESS)
            { set_last_error("Gemma4 MoE batched decode: replay failed."); dc->reset(); return 0; }
            finalize_compute_with_download(dc->logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_seqs * sizeof(float));
            host_read_barrier();
            if (g4mb_timing) { fprintf(stderr, "[g4moe-batched] replay N=%d\n", n_seqs); fflush(stderr); }
            clear_last_error();
            return 1;
        }

        // ---- build ----
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("Gemma4 MoE batched decode: ctx init failed."); return 0; }
        }
        else { if (!context.init(ctx_size)) { set_last_error("Gemma4 MoE batched decode: ctx pool failed."); return 0; } ctx = context.value; }

        ggml_tensor* current = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n_seqs);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
        if (can_persist) { ggml_set_input(current); ggml_set_input(pos_tensor); }
        std::vector<ggml_tensor*> kv_index_all(static_cast<std::size_t>(num_layers) * n_seqs, nullptr);

        struct LT {
            ggml_tensor *attn_norm_w, *qkv_w, *k_w, *v_w, *q_norm_w, *k_norm_w, *o_w, *post_attn_norm_w;
            ggml_tensor *ffn_norm_w, *gu_w, *down_w, *post_ffw_norm_1_w;
            ggml_tensor *gate_inp_w, *gate_inp_scale_t, *pre_ffw_norm_2_w, *gate_up_exps_t, *down_exps_t, *down_exps_scale_t, *post_ffw_norm_2_w, *post_ffw_norm_w;
            ggml_tensor *freq_factors_t;
            std::vector<ggml_tensor*> k_cached, v_cached, k_cpy, v_cpy;
            ggml_tensor* attn_mask; std::vector<ggml_fp16_t> attn_mask_data;
        };
        std::vector<LT> lt(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            const auto& d = layers[l]; auto& t = lt[l]; auto& info = li[l];
            const int nExp = d.num_experts;
            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
            if (d.separate_qkv != 0) { t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1); t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1); }
            else { t.k_w = nullptr; t.v_w = nullptr; }
            t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
            t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            t.post_ffw_norm_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nExp);
            t.gate_inp_scale_t = (d.gate_inp_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
            t.pre_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_up_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gue_type), d.gue_ne0, d.gue_ne1, nExp);
            t.down_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.de_type), d.de_ne0, d.de_ne1, nExp);
            t.down_exps_scale_t = (d.down_exps_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nExp) : nullptr;
            t.post_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.post_ffw_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.freq_factors_t = (!info.isLocal && d.freq_factors != nullptr && d.freq_factors_len > 0)
                ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.freq_factors_len) : nullptr;
            t.k_cached.resize(n_seqs); t.v_cached.resize(n_seqs); t.k_cpy.resize(n_seqs, nullptr); t.v_cpy.resize(n_seqs, nullptr);
            for (int s = 0; s < n_seqs; s++)
            {
                t.k_cached[s] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), info.hd, info.cacheSize, info.kvH);
                t.v_cached[s] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), info.hd, info.cacheSize, info.kvH);
            }
            t.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, info.win, 1, 1, n_seqs);
            if (can_persist) ggml_set_input(t.attn_mask);
        }

        ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
        ggml_tensor* final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);


        ggml_tensor* hidden = current;
        for (int l = 0; l < num_layers; l++)
        {
            const auto& d = layers[l]; auto& t = lt[l]; auto& info = li[l];
            const int nExp = d.num_experts, nUsed = d.num_experts_used;
            const std::int64_t ffDense = d.gu_ne1 / 2, ffMoe = d.gue_ne1 / 2;
            ggml_tensor* rope_ff = t.freq_factors_t;

            // ---- attention (batched, per-seq KV) ----
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w);
            ggml_tensor* q_raw; ggml_tensor* k_raw; ggml_tensor* v_raw;
            if (t.k_w != nullptr) { q_raw = ggml_mul_mat(ctx, t.qkv_w, normed); k_raw = ggml_mul_mat(ctx, t.k_w, normed); v_raw = ggml_mul_mat(ctx, t.v_w, normed); }
            else
            {
                ggml_tensor* qkv = ggml_mul_mat(ctx, t.qkv_w, normed);
                q_raw = ggml_view_2d(ctx, qkv, info.qDim, n_seqs, qkv->nb[1], 0);
                k_raw = ggml_view_2d(ctx, qkv, info.kvH * info.hd, n_seqs, qkv->nb[1], static_cast<std::size_t>(info.qDim) * sizeof(float));
                v_raw = ggml_view_2d(ctx, qkv, info.kvH * info.hd, n_seqs, qkv->nb[1], static_cast<std::size_t>(info.qDim + info.kvH * info.hd) * sizeof(float));
            }
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, q_raw), info.hd, num_heads, n_seqs);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, k_raw), info.hd, info.kvH, n_seqs);
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, v_raw), info.hd, info.kvH, n_seqs);
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_3d, eps), t.q_norm_w);
            ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_3d, eps), t.k_norm_w);
            ggml_tensor* v_normed = ggml_rms_norm(ctx, v_3d, eps);
            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_normed, pos_tensor, rope_ff, d.rope_n_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);
            ggml_tensor* k_rope = ggml_rope_ext(ctx, k_normed, pos_tensor, rope_ff, d.rope_n_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);

            ggml_tensor* k_cat = nullptr; ggml_tensor* v_cat = nullptr;
            for (int s = 0; s < n_seqs; s++)
            {
                const int cachePos = positions[s];
                ggml_tensor* k_s = ggml_view_3d(ctx, k_rope, info.hd, info.kvH, 1, k_rope->nb[1], k_rope->nb[2], static_cast<std::size_t>(s) * k_rope->nb[2]);
                ggml_tensor* v_s = ggml_view_3d(ctx, v_normed, info.hd, info.kvH, 1, v_normed->nb[1], v_normed->nb[2], static_cast<std::size_t>(s) * v_normed->nb[2]);
                ggml_tensor* k_write = ggml_cont(ctx, ggml_permute(ctx, k_s, 0, 2, 1, 3));
                ggml_tensor* v_write = ggml_cont(ctx, ggml_permute(ctx, v_s, 0, 2, 1, 3));
                ggml_tensor* k_src; ggml_tensor* v_src;
                if (can_persist)
                {
                    ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
                    ggml_set_input(kv_idx);
                    kv_index_all[static_cast<std::size_t>(l) * n_seqs + s] = kv_idx;
                    t.k_cpy[s] = ggml_set_rows(ctx, t.k_cached[s], k_write, kv_idx);
                    t.v_cpy[s] = ggml_set_rows(ctx, t.v_cached[s], v_write, kv_idx);
                    k_src = t.k_cpy[s]; v_src = t.v_cpy[s];
                }
                else
                {
                    ggml_tensor* k_dst = ggml_view_3d(ctx, t.k_cached[s], info.hd, 1, info.kvH, t.k_cached[s]->nb[1], t.k_cached[s]->nb[2], static_cast<std::size_t>(cachePos) * t.k_cached[s]->nb[1]);
                    ggml_tensor* v_dst = ggml_view_3d(ctx, t.v_cached[s], info.hd, 1, info.kvH, t.v_cached[s]->nb[1], t.v_cached[s]->nb[2], static_cast<std::size_t>(cachePos) * t.v_cached[s]->nb[1]);
                    t.k_cpy[s] = ggml_cpy(ctx, k_write, k_dst);
                    t.v_cpy[s] = ggml_cpy(ctx, v_write, v_dst);
                    k_src = t.k_cached[s]; v_src = t.v_cached[s];
                }
                ggml_tensor* k_win = view_kv_cache_window(ctx, k_src, info.hd, info.cacheSize, info.kvH, 0, info.win, kvType);
                ggml_tensor* v_win = view_kv_cache_window(ctx, v_src, info.hd, info.cacheSize, info.kvH, 0, info.win, kvType);
                k_cat = (k_cat == nullptr) ? k_win : ggml_concat(ctx, k_cat, k_win, 3);
                v_cat = (v_cat == nullptr) ? v_win : ggml_concat(ctx, v_cat, v_win, 3);
            }
            ggml_tensor* q4d = ggml_reshape_4d(ctx, ggml_cont(ctx, q_rope), info.hd, 1, num_heads, n_seqs);
            ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q4d, k_cat, v_cat, t.attn_mask, 1.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
            ggml_tensor* attn_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, attn_out), info.qDim, n_seqs);
            ggml_tensor* o_out = ggml_mul_mat(ctx, t.o_w, attn_2d);
            ggml_tensor* post_attn = ggml_mul(ctx, ggml_rms_norm(ctx, o_out, eps), t.post_attn_norm_w);
            ggml_tensor* residual1 = ggml_add(ctx, hidden, post_attn);   // [H, N]

            // ---- dense shared FFN (N tokens) ----
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.ffn_norm_w);
            ggml_tensor* gu = ggml_mul_mat(ctx, t.gu_w, ffn_normed);
            ggml_tensor* dense_gate = ggml_cont(ctx, ggml_view_2d(ctx, gu, ffDense, n_seqs, gu->nb[1], 0));
            ggml_tensor* dense_up = ggml_cont(ctx, ggml_view_2d(ctx, gu, ffDense, n_seqs, gu->nb[1], static_cast<std::size_t>(ffDense) * sizeof(float)));
            ggml_tensor* dense_h = ggml_mul(ctx, ggml_gelu(ctx, dense_gate), dense_up);
            ggml_tensor* dense_down = ggml_mul_mat(ctx, t.down_w, dense_h);
            ggml_tensor* mlp = ggml_mul(ctx, ggml_rms_norm(ctx, dense_down, eps), t.post_ffw_norm_1_w);

            // ---- MoE router (N tokens) ----
            ggml_tensor* route_n = ggml_rms_norm(ctx, residual1, eps);
            route_n = ggml_scale(ctx, route_n, d.inv_sqrt_hidden);
            if (t.gate_inp_scale_t != nullptr) route_n = ggml_mul(ctx, route_n, t.gate_inp_scale_t);
            ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, route_n);
            ggml_tensor* probs = ggml_soft_max(ctx, router_logits);
            ggml_tensor* sel = ggml_top_k(ctx, probs, nUsed);
            ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, nExp, n_seqs);
            ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);
            ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, nUsed, n_seqs);
            ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);
            w_2d = ggml_div(ctx, w_2d, w_sum);
            if (t.down_exps_scale_t != nullptr)
            {
                ggml_tensor* scale_b = ggml_repeat(ctx, ggml_reshape_3d(ctx, t.down_exps_scale_t, 1, nExp, 1), probs_r);
                ggml_tensor* sel_scale = ggml_get_rows(ctx, scale_b, sel);
                w_2d = ggml_mul(ctx, w_2d, ggml_reshape_2d(ctx, sel_scale, nUsed, n_seqs));
            }
            ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, nUsed, n_seqs);

            // ---- MoE experts (N tokens) ----
            ggml_tensor* moe_in = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.pre_ffw_norm_2_w);
            ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, moe_in, H, 1, n_seqs);
            ggml_tensor* gate_up = ggml_mul_mat_id(ctx, t.gate_up_exps_t, moe_in_3d, sel);
            ggml_tensor* moe_gate = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
            ggml_tensor* moe_up = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], static_cast<std::size_t>(ffMoe) * gate_up->nb[0]);
            ggml_tensor* moe_act = ggml_geglu_split(ctx, moe_gate, moe_up);
            ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps_t, moe_act, sel);
            ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);
            ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, n_seqs, weighted->nb[2], 0);
            for (int u = 1; u < nUsed; ++u)
            {
                ggml_tensor* view_u = ggml_view_2d(ctx, weighted, H, n_seqs, weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                moe_out = ggml_add(ctx, moe_out, view_u);
            }
            ggml_tensor* moe_normed = ggml_mul(ctx, ggml_rms_norm(ctx, moe_out, eps), t.post_ffw_norm_2_w);
            mlp = ggml_add(ctx, mlp, moe_normed);

            ggml_tensor* mlp_normed = ggml_mul(ctx, ggml_rms_norm(ctx, mlp, eps), t.post_ffw_norm_w);
            ggml_tensor* result = ggml_add(ctx, residual1, mlp_normed);
            if (std::fabs(d.layer_output_scale - 1.0f) > 1e-9f) result = ggml_scale(ctx, result, d.layer_output_scale);
            hidden = result;
        }

        // fold final-norm + lm_head -> logits [vocab, N]
        ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);
        ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn);
        if (logit_softcap > 0.0f)
        {
            logits = ggml_scale(ctx, logits, 1.0f / logit_softcap);
            logits = ggml_tanh(ctx, logits);
            logits = ggml_scale(ctx, logits, logit_softcap);
        }
        ggml_tensor* logits_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_seqs);
        ggml_tensor* out_op = ggml_cpy(ctx, logits, logits_out);
        ggml_set_output(out_op);

        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (384 + 24 * n_seqs) + 1024;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
            for (int s = 0; s < n_seqs; s++)
            {
                if (lt[l].k_cpy[s] != nullptr) ggml_build_forward_expand(graph, lt[l].k_cpy[s]);
                if (lt[l].v_cpy[s] != nullptr) ggml_build_forward_expand(graph, lt[l].v_cpy[s]);
            }
        ggml_build_forward_expand(graph, out_op);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS) { if (needs_upload) upload_list.push_back({t, data, bytes}); return; }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                { if (!cacheable) ephemeral_bufs.emplace_back(buf); if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS) return; }
            }
            upload_list.push_back({t, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            const auto& d = layers[l]; auto& t = lt[l]; auto& info = li[l];
            bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
            if (t.k_w != nullptr) { bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true); bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true); }
            bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
            bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes), true);
            bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes), true);
            bind_or_mark(t.gate_up_exps_t, d.gate_up_exps, static_cast<std::size_t>(d.gue_bytes), true);
            bind_or_mark(t.down_exps_t, d.down_exps, static_cast<std::size_t>(d.de_bytes), true);
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.ffn_norm_w, d.ffn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_1_w, d.post_ffw_norm_1_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.pre_ffw_norm_2_w, d.pre_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_2_w, d.post_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_w, d.post_ffw_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(info.hd) * sizeof(float), true);
            bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(info.hd) * sizeof(float), true);
            bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(H) * d.num_experts * sizeof(float), true);
            if (t.gate_inp_scale_t != nullptr) bind_or_mark(t.gate_inp_scale_t, d.gate_inp_scale, static_cast<std::size_t>(H) * sizeof(float), true);
            if (t.down_exps_scale_t != nullptr) bind_or_mark(t.down_exps_scale_t, d.down_exps_scale, static_cast<std::size_t>(d.num_experts) * sizeof(float), true);
            if (t.freq_factors_t != nullptr) bind_or_mark(t.freq_factors_t, d.freq_factors, static_cast<std::size_t>(d.freq_factors_len) * sizeof(float), true);
            for (int s = 0; s < n_seqs; s++)
            {
                bind_or_mark(t.k_cached[s], k_cache_arr[l * n_seqs + s], kv_cache_bytes(info.kvH, info.cacheSize, info.hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(t.v_cached[s], v_cache_arr[l * n_seqs + s], kv_cache_bytes(info.kvH, info.cacheSize, info.hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
            fill_batched_mask(t.attn_mask_data, info.win);
            bind_or_mark(t.attn_mask, t.attn_mask_data.data(), t.attn_mask_data.size() * sizeof(ggml_fp16_t), false);
        }
        bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
        bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(H) * sizeof(float), true);

        // NOTE: a dedicated gallocr (packed, VRAM-frugal) was tried here to fit the
        // 26B on 16GB — it ran FAST (N=2 ~99 t/s) but produced garbage under this
        // fork's CUDA-graph capture (gallocr slot-REUSE is incompatible with the
        // captured graph; own-slot works precisely because it never reuses). So we
        // keep the capture-safe own-slot alloc; the 26B batched is VRAM-bound on 16GB
        // and stays gated off (see TS_BATCHED_FUSED_MOE).
        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr) { set_last_error("Gemma4 MoE batched decode: persist alloc failed."); ggml_free(ctx); return 0; }
        }
        else if (!alloc_ctx_tensors_reuse(ctx))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr) { set_last_error("Gemma4 MoE batched decode: buffer alloc failed."); return 0; }
        }

        host_read_barrier();
        for (auto& u : upload_list) ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);
        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(H) * n_seqs * sizeof(float));
        ggml_backend_tensor_set(pos_tensor, positions, 0, static_cast<std::size_t>(n_seqs) * sizeof(std::int32_t));
        if (can_persist)
            for (int l = 0; l < num_layers; l++)
                for (int s = 0; s < n_seqs; s++)
                { std::int64_t row = positions[s]; ggml_backend_tensor_set(kv_index_all[static_cast<std::size_t>(l) * n_seqs + s], &row, 0, sizeof(std::int64_t)); }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        { set_last_error("Gemma4 MoE batched decode: graph compute failed."); if (can_persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); } return 0; }

        finalize_compute_with_download(logits_out, logits_data, static_cast<std::size_t>(vocab_size) * n_seqs * sizeof(float));

        if (can_persist)
        {
            host_read_barrier();
            G4BatchedDecodeCache& e = g_g4moebatched_pool.claim(sig_disc, sig_kc, n_seqs);
            e.ctx = ctx; e.buffer = persist_buf; e.graph = graph;
            e.hidden_in = current; e.pos_tensor = pos_tensor; e.logits_out = logits_out;
            e.kv_index = kv_index_all;
            e.attn_mask.resize(num_layers);
            for (int l = 0; l < num_layers; l++) e.attn_mask[l] = lt[l].attn_mask;
            e.layer_window = winvec; e.sig_disc = sig_disc; e.sig_kc = sig_kc;
            e.num_layers = num_layers; e.hidden_size = H; e.n_seqs = n_seqs; e.vocab = vocab_size;
            e.valid = true;
            if (g4mb_timing) { fprintf(stderr, "[g4moe-batched] BUILT persist N=%d win0=%d\n", n_seqs, winvec.empty()?0:winvec[0]); fflush(stderr); }
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in Gemma4 MoE batched decode."); return 0; }
}

TSG_EXPORT void TSGgml_Gemma4ResetMoEBatchedDecodeCache()
{
    g_g4moebatched_pool.reset_all();
}

// ============================================================================
// Fused MULTI-TOKEN verify (seqLen == num_tokens > 1): runs the whole dense
// Gemma 4 transformer over a small batch of tokens [start_pos, start_pos+N) as
// ONE GGML graph — the speculative-decoding verify. The single-token decode
// kernel above (TSGgml_Gemma4ModelDecode) is the only thing fast enough to beat
// the per-op verify, and it is seqLen==1 only; this is its multi-token sibling.
//
// Supports the dense (non-MoE) trunk including per-layer embeddings (PLE) and
// shared-KV (KV-donor) layers — both ported from the decode kernel — so the
// Gemma 4 E-series (e.g. E4B) verifies on this fused path. Still enforced by the
// C# caller's gate: dense only (no MoE), and for GLOBAL layers
// total_seq_len = start_pos + N <= the cache size
// (the SWA window). The last condition means the SWA cache has NOT wrapped, so
// every query's window covers [0, total_seq_len) and attention is PURE CAUSAL —
// no circular-window gymnastics, one ggml_diag_mask_inf(start_pos) mask. The
// caller still owns the post-final-norm + LM head; this returns the per-row
// hidden state [hidden_size, N] (the layer-stack output, pre output_norm).
// ============================================================================
TSG_EXPORT int TSGgml_Gemma4ModelVerify(
    float* hidden_data, int hidden_size, int num_layers, int num_tokens,
    void** attn_norm_arr, void** qkv_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr, void** post_attn_norm_arr,
    void** ffn_norm_arr, void** gu_arr, void** down_arr, void** post_ffn_norm_arr,
    void** k_cache_arr, void** v_cache_arr,
    int* head_dim_arr, int* kv_heads_arr, int* cache_size_arr, int* is_local_arr,
    float* rope_base_arr, float* layer_scalar_arr,
    int* qkv_type_arr, std::int64_t* qkv_ne0_arr, std::int64_t* qkv_ne1_arr, std::int64_t* qkv_bytes_arr,
    int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    int* gu_type_arr, std::int64_t* gu_ne0_arr, std::int64_t* gu_ne1_arr, std::int64_t* gu_bytes_arr,
    int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    int num_heads, int start_pos,
    float eps,
    float* rope_freq_factors, int rope_freq_factors_len,
    int* rope_n_dims_arr,
    int kv_cache_type,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    // Shared-KV (KV-donor) map: kv_source_arr[l] == l for a normal layer; a
    // different layer index means layer l reads that donor's K/V (Gemma 4
    // E-series shared_kv_layers). Nullable (treated as the identity map).
    int* kv_source_arr,
    // PLE (per-layer-embedding) data, per token per layer — nullable. Layout is
    // [num_tokens, num_layers * ple_dim] row-major (the C# perLayerInputs tensor).
    float* ple_data, int ple_dim,
    void** ple_gate_arr, int* ple_gate_type_arr, std::int64_t* ple_gate_ne0_arr, std::int64_t* ple_gate_ne1_arr, std::int64_t* ple_gate_bytes_arr,
    void** ple_proj_arr, int* ple_proj_type_arr, std::int64_t* ple_proj_ne0_arr, std::int64_t* ple_proj_ne1_arr, std::int64_t* ple_proj_bytes_arr,
    void** ple_post_norm_arr,
    // Multimodal bidirectional-span mask, nullable, length N (one byte per token,
    // 1 = "soft" image/audio token). When set (multimodal prefill, start_pos==0)
    // the attention mask is causal PLUS bidirectional within the soft-token spans:
    // a soft-token query may attend forward to a soft-token key. Mirrors the C#
    // per-op ApplyCausalMask exceptPositions path. Null for text / MTP verify.
    const unsigned char* is_except_arr,
    // In-kernel PLE gather. When ple_token_embd_data != nullptr, the per-layer
    // embeddings are gathered INSIDE this graph via ggml_get_rows on the resident
    // quantized per_layer_token_embd table (ple_token_ids = the chunk's N token
    // ids), instead of being computed in C# and uploaded as ple_data. This avoids
    // the device->host->device round-trip of the ~88 MB gathered PLE per chunk
    // (the PLE table is already resident; only the tiny id list is uploaded).
    // When set, ple_data is ignored. ple_token_embd is [ne0 = num_layers*ple_dim,
    // ne1 = vocab]; get_rows(table, ids) -> [num_layers*ple_dim, N], byte-identical
    // layout to the uploaded ple_data.
    const void* ple_token_embd_data, int ple_token_embd_type,
    std::int64_t ple_token_embd_ne0, std::int64_t ple_token_embd_ne1, std::int64_t ple_token_embd_bytes,
    const std::int32_t* ple_token_ids,
    // Hidden-projection PLE component (nullable): ple = (sqrt(ple_dim)*get_rows +
    // rmsnorm((hidden @ per_layer_model_proj)/sqrt(hidden), per_layer_proj_norm)) / sqrt(2).
    // per_layer_model_proj is [ne0=hidden, ne1=num_layers*ple_dim]; per_layer_proj_norm
    // is F32 [ple_dim]. When ple_proj_w_data is null the token-embedding component is
    // used alone (matches ComputePLE's pleProj==null branch).
    const void* ple_proj_w_data, int ple_proj_w_type,
    std::int64_t ple_proj_w_ne0, std::int64_t ple_proj_w_ne1, std::int64_t ple_proj_w_bytes,
    const void* ple_proj_norm_data)
{
    try
    {
        if (!ensure_backend())
            return 0;

        // TS_G4_VERIFY_TIMING=1: print a per-call phase breakdown (build / bind /
        // alloc / upload / compute / download) so prefill overhead outside the
        // GPU kernels is visible. Mirrors TS_GEMMA4_FD_TIMING on the decode path.
        static const bool g4v_timing = std::getenv("TS_G4_VERIFY_TIMING") != nullptr;
        using vt_clock = std::chrono::steady_clock;
        const auto vt0 = vt_clock::now();
        auto vt_ms = [](vt_clock::time_point a, vt_clock::time_point b)
            { return std::chrono::duration<double, std::milli>(b - a).count(); };
        vt_clock::time_point vt_built{}, vt_bound{}, vt_alloced{}, vt_uploaded{}, vt_computed{};

        const int N = num_tokens;
        const int totalSeqLen = start_pos + N;
        if (N <= 1)
            return 0;

        // Batch (query-count) padding, Vulkan only. ggml-vulkan's quantized
        // GEMMs and flash attention both run measurably faster when the batch
        // dimension is tile-aligned: every weight GEMM here has ne11 == N, and
        // whole-prefill was ~8% slower at N=1985 than at N=2016/2048 on
        // gemma4-12B (the GEMM kernels drop from ~28 to ~24 TFLOPS, flash from
        // its aligned to its unaligned shader). llama.cpp never hits this
        // because its ubatches are 512-token aligned. Pad the query batch to a
        // multiple of 64 with dummy rows: their input embeddings are uploaded
        // as ZEROS (finite through every column-wise op — matmul / norms / GLU
        // touch each column independently), causality already excludes key
        // positions >= N from every real query row (dummy keys sit after the
        // real ones), the KV-cache writes below copy only the first N rows,
        // and only the first N rows of the output hidden are downloaded. The
        // dummy columns cost <= 63 columns of extra GEMM work (< 3% at 2k
        // tokens) and their outputs are discarded. Gated to N > 64 so the MTP
        // speculative verify (N <= 16, latency-critical) keeps its exact
        // shapes. TS_G4_VERIFY_NPAD=0 disables.
        static const bool g4v_npad_enabled = []{ const char* e = std::getenv("TS_G4_VERIFY_NPAD"); return e == nullptr || e[0] != '0'; }();
        const bool pad_batch = g4v_npad_enabled && g_backend_type == BACKEND_TYPE_VULKAN && N > 64;
        const int NQ = pad_batch ? ((N + 63) & ~63) : N;

        // The bidirectional-span mask maps view-index == logical position only at
        // start_pos==0 (nPast==0); the C# gate guarantees that for multimodal.
        const unsigned char* is_except = (start_pos == 0) ? is_except_arr : nullptr;

        struct LayerInfo { int hd; int kvHeads; int qDim; int kDim; int cacheSize; bool isLocal; bool isShared; int kvSource; };
        std::vector<LayerInfo> li(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            auto& info = li[l];
            info.hd = head_dim_arr[l];
            info.kvHeads = kv_heads_arr[l];
            info.qDim = num_heads * info.hd;
            info.kDim = info.kvHeads * info.hd;
            info.kvSource = (kv_source_arr != nullptr) ? kv_source_arr[l] : l;
            info.isShared = (info.kvSource != l);
            // Shared layers borrow the donor's cache size / locality (the donor
            // physically owns the K/V buffer).
            info.cacheSize = cache_size_arr[info.kvSource];
            info.isLocal = is_local_arr[info.kvSource] != 0;
            // Global (full-attention) layers use a linear cache that must cover the
            // whole sequence (the C# caller grows it via EnsureCacheCapacity). SWA
            // (local) layers use a circular window cache and are handled at any
            // length below (windowed read + wrap-aware write).
            if (!info.isLocal && totalSeqLen > info.cacheSize)
                return 0;
        }

        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for Gemma4 model verify.");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_tensor* current = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, NQ);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NQ);

        ggml_tensor* freq_factors_t = nullptr;
        if (rope_freq_factors != nullptr && rope_freq_factors_len > 0)
            freq_factors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rope_freq_factors_len);

        // PLE input: all N tokens' per-layer embeddings, laid out
        // [num_tokens, num_layers * ple_dim] row-major (matches perLayerInputs).
        // In-kernel mode gathers it here via get_rows on the resident table; the
        // uploaded-ple_data mode allocates a plain F32 tensor filled by H2D below.
        const bool ple_in_kernel = ple_token_embd_data != nullptr && ple_token_ids != nullptr && ple_dim > 0;
        const int total_ple_dim = num_layers * ple_dim;
        ggml_tensor* ple_input = nullptr;
        ggml_tensor* ple_table = nullptr;
        ggml_tensor* ple_ids = nullptr;
        ggml_tensor* ple_proj_w = nullptr;
        ggml_tensor* ple_proj_norm_w = nullptr;
        if (ple_in_kernel)
        {
            ple_table = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_token_embd_type),
                ple_token_embd_ne0, ple_token_embd_ne1);
            ple_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NQ);
            // Token-embedding component: sqrt(ple_dim) * get_rows(table, ids).
            // get_rows(table[ne0=total_ple_dim, ne1=vocab], ids[N]) -> [total_ple_dim, N]
            // F32, same memory layout as the uploaded ple_data.
            ggml_tensor* ple_tok = ggml_get_rows(ctx, ple_table, ple_ids);
            ple_tok = ggml_scale(ctx, ple_tok, sqrtf(static_cast<float>(ple_dim)));

            if (ple_proj_w_data != nullptr)
            {
                // Hidden-projection component: rmsnorm((hidden @ proj)/sqrt(hidden), norm).
                ple_proj_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_proj_w_type),
                    ple_proj_w_ne0, ple_proj_w_ne1);
                ple_proj_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ple_dim);
                ggml_tensor* proj = ggml_mul_mat(ctx, ple_proj_w, current);   // [total_ple_dim, N]
                proj = ggml_scale(ctx, proj, 1.0f / sqrtf(static_cast<float>(hidden_size)));
                // Per-(token,layer) RMSNorm over ple_dim: view [total_ple_dim, N] as
                // [ple_dim, num_layers*N], norm rows, scale by norm weight, reshape back.
                ggml_tensor* proj_r = ggml_reshape_2d(ctx, ggml_cont(ctx, proj), ple_dim, static_cast<std::int64_t>(num_layers) * NQ);
                proj_r = ggml_mul(ctx, ggml_rms_norm(ctx, proj_r, eps), ple_proj_norm_w);
                proj = ggml_reshape_2d(ctx, proj_r, total_ple_dim, NQ);
                // combined = (proj + tok) / sqrt(2)
                ple_input = ggml_scale(ctx, ggml_add(ctx, proj, ple_tok), 1.0f / sqrtf(2.0f));
            }
            else
            {
                ple_input = ple_tok;
            }
        }
        else if (ple_data != nullptr && ple_dim > 0)
        {
            ple_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                static_cast<std::int64_t>(NQ) * num_layers * ple_dim);
        }
        const bool has_ple = ple_input != nullptr;

        struct LayerTensors {
            ggml_tensor* attn_norm_w; ggml_tensor* qkv_w; ggml_tensor* k_w; ggml_tensor* v_w;
            ggml_tensor* q_norm_w; ggml_tensor* k_norm_w; ggml_tensor* o_w; ggml_tensor* post_attn_norm_w;
            ggml_tensor* ffn_norm_w; ggml_tensor* gu_w; ggml_tensor* down_w; ggml_tensor* post_ffn_norm_w;
            ggml_tensor* k_cached_t; ggml_tensor* v_cached_t;
            ggml_tensor* k_cpy; ggml_tensor* v_cpy;     // primary cache write
            ggml_tensor* k_cpy2; ggml_tensor* v_cpy2;   // wrapped tail (circular SWA write past the buffer end)
            ggml_tensor* k_prev; ggml_tensor* v_prev;   // swaPrev: F32 copy of the prev window (read before the cache write)
            ggml_tensor* ple_gate_w; ggml_tensor* ple_proj_w; ggml_tensor* ple_post_norm_w;
        };
        std::vector<LayerTensors> layers(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];
            lt.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkv_type_arr[l]), qkv_ne0_arr[l], qkv_ne1_arr[l]);
            // Mixed-quant layers carry separate K/V weights (qkv_w then holds Q
            // only). Shared layers never run their own K/V projection.
            const bool separate_qkv = (!info.isShared && k_arr != nullptr && k_arr[l] != nullptr);
            if (separate_qkv)
            {
                lt.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(k_type_arr[l]), k_ne0_arr[l], k_ne1_arr[l]);
                lt.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(v_type_arr[l]), v_ne0_arr[l], v_ne1_arr[l]);
            }
            else { lt.k_w = nullptr; lt.v_w = nullptr; }
            lt.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type_arr[l]), o_ne0_arr[l], o_ne1_arr[l]);
            lt.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gu_type_arr[l]), gu_ne0_arr[l], gu_ne1_arr[l]);
            lt.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type_arr[l]), down_ne0_arr[l], down_ne1_arr[l]);
            lt.post_ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            // Shared layers borrow the donor's cache tensors (linked below); they
            // own no K/V buffer of their own.
            if (!info.isShared)
            {
                lt.k_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, info.cacheSize, info.kvHeads);
                lt.v_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, info.cacheSize, info.kvHeads);
            }
            else { lt.k_cached_t = nullptr; lt.v_cached_t = nullptr; }
            lt.k_cpy = nullptr; lt.v_cpy = nullptr; lt.k_cpy2 = nullptr; lt.v_cpy2 = nullptr;

            lt.ple_gate_w = nullptr; lt.ple_proj_w = nullptr; lt.ple_post_norm_w = nullptr;
            if (has_ple && ple_gate_arr != nullptr && ple_gate_arr[l] != nullptr)
            {
                lt.ple_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_gate_type_arr[l]),
                    ple_gate_ne0_arr[l], ple_gate_ne1_arr[l]);
                lt.ple_proj_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_proj_type_arr[l]),
                    ple_proj_ne0_arr[l], ple_proj_ne1_arr[l]);
                lt.ple_post_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            }
        }

        // Link shared layers to their donor's KV cache tensors (the donor, an
        // earlier layer, writes them in this same graph).
        for (int l = 0; l < num_layers; l++)
        {
            if (li[l].isShared)
            {
                layers[l].k_cached_t = layers[li[l].kvSource].k_cached_t;
                layers[l].v_cached_t = layers[li[l].kvSource].v_cached_t;
            }
        }

        // Additive causal masks for ggml_flash_attn_ext. Attention runs through
        // flash_attn_ext (not a manual mul_mat + soft_max chain): flash is O(seq)
        // memory — it never materializes the [attendLen, N, heads] score tensor that
        // reached multiple GB at multi-thousand-token prefill (forcing a one-time
        // multi-GB gallocr growth on the first long prompt, the dominant cold-start
        // cost) — and is the proven path on Metal AND CUDA (mirrors the MoE verify
        // and the decode path; the manual chain is numerically fragile at head_dim
        // 256/512). One mask per distinct (kvLen, validLen, window), shared across
        // layers. mask[qi][ki] = 0 iff ki < validLen (real, not flash padding) AND
        // ki <= (validLen-N)+qi (causal) AND (window==0 || ki > threshold-window)
        // (sliding-window low bound). The windowed variant (window>0) serves SWA
        // prefill that attends the FRESH K/V of all N tokens (start_pos==0, N >
        // sliding window); window==0 covers the circular-cache read and global.
        // On-device causal-mask generation (CUDA, text prefill only): fill each
        // [kvLen, N] mask straight into its device buffer after allocation instead
        // of the O(N*kvLen) host fill + H2D upload. The bidirectional multimodal
        // case (is_except != nullptr) keeps the host path. dataIdx == -1 in the
        // cache marks a GPU-filled mask (no host data / no upload).
#ifdef TSG_GGML_USE_CUDA
        static const bool gpu_mask_enabled = []{ const char* e = std::getenv("TS_G4_GPU_MASK"); return e == nullptr || e[0] != '0'; }();
        const bool gpu_mask = gpu_mask_enabled && g_backend_type == BACKEND_TYPE_CUDA && is_except == nullptr;
#else
        const bool gpu_mask = false;
#endif
        struct GpuMaskFill { ggml_tensor* tensor; int kvLen; int mN; int nPast; int window; int validLen; };
        std::vector<GpuMaskFill> gpu_mask_fills;

        struct VerifyMask { int kvLen; int validLen; int window; ggml_tensor* tensor; int dataIdx; };
        std::vector<VerifyMask> mask_cache;
        std::vector<std::vector<ggml_fp16_t>> mask_data_store;
        auto get_causal_mask = [&](int kvLen, int validLen, int window) -> ggml_tensor* {
            for (auto& m : mask_cache)
                if (m.kvLen == kvLen && m.validLen == validLen && m.window == window) return m.tensor;
            // Mask rows cover the padded query batch (NQ >= N); the extra dummy
            // rows keep a non-empty in-range band (their causal threshold is past
            // validLen, so the band clamps to real keys) — never fully masked, so
            // flash attention's row softmax stays finite. nPast derives from the
            // REAL token count: validLen - N == start_pos for real rows.
            ggml_tensor* mt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kvLen, NQ);
            if (gpu_mask)
            {
                // Defer to a device-side fill after gallocr allocates mt->data.
                gpu_mask_fills.push_back({mt, kvLen, NQ, validLen - N, window, validLen});
                mask_cache.push_back({kvLen, validLen, window, mt, -1});
                return mt;
            }
            std::vector<ggml_fp16_t> data(static_cast<std::size_t>(kvLen) * NQ);
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            const int nPast = validLen - N;
            for (int qi = 0; qi < NQ; qi++)
            {
                const int threshold = nPast + qi;
                const int low = (window > 0) ? (threshold - window + 1) : 0;
                // Bidirectional within soft-token (image/audio) spans: a soft query
                // also keeps soft keys ahead of it. Window low-bound is not applied
                // to the bidi branch (mirrors the C# per-op exceptPositions path).
                const bool q_except = is_except != nullptr && qi < N && is_except[qi] != 0;
                ggml_fp16_t* row = &data[static_cast<std::size_t>(qi) * kvLen];
                if (!q_except)
                {
                    // Fast path (no multimodal bidi): the unmasked (zero) keys form a
                    // single contiguous band [lo, hi]: lo = sliding-window low bound,
                    // hi = min(causal threshold, last real key). Fill it analytically
                    // instead of a per-element branch + fp16 convert over [0, kvLen)
                    // — this host loop is O(N*kvLen) and at multi-thousand-token
                    // prefill it blocks the GPU (the [N,N] mask reaches 134M entries).
                    const int lo = (low > 0) ? low : 0;
                    const int hi = std::min(threshold, validLen - 1);
                    std::fill(row, row + kvLen, neg_inf);
                    if (hi >= lo && lo < kvLen)
                        std::fill(row + lo, row + std::min(hi + 1, kvLen), zero_val);
                    continue;
                }
                for (int ki = 0; ki < kvLen; ki++)
                {
                    bool causal = (ki < validLen) && (ki <= threshold) && !(window > 0 && ki < low);
                    bool bidi = q_except && ki < N && is_except[ki] != 0;
                    row[ki] = (causal || bidi) ? zero_val : neg_inf;
                }
            }
            const int idx = static_cast<int>(mask_data_store.size());
            mask_data_store.push_back(std::move(data));
            mask_cache.push_back({kvLen, validLen, window, mt, idx});
            return mt;
        };

        // Tiled sliding-window attention (mirrors llama.cpp's iSWA windowed KV):
        // a SWA prefill that overflows the window (start_pos==0, N > window) attends
        // the FRESH chunk's full N keys with a sliding-window mask, but ggml flash
        // only skips the causal-FUTURE region (flash_attn_mask_to_KV_max), not the
        // sliding-window PAST, so a single full-N flash iterates [0, qpos] per query
        // and wastes ~93% of its work at long prefill (local flash was ~10% of
        // prefill GPU time). Instead split the N queries into tiles of TS_G4_SWA_TILE
        // (default 1024, a multiple of the 512 window so each tile's key slice stays
        // FATTN_KQ_STRIDE-aligned -> keeps the fast GQA flash kernel). Each query
        // tile attends only its window slice [tileStart-W, tileEnd) of the fresh K/V.
        // Bidi multimodal spans (is_except) need forward attention past a query tile
        // -> the caller falls back to the full-N flash there.
        static const bool swa_tiled = []{ const char* e = std::getenv("TS_G4_SWA_TILED"); return e == nullptr || e[0] != '0'; }();
        static const int swa_tile = []{ const char* e = std::getenv("TS_G4_SWA_TILE"); int v = e ? std::atoi(e) : 0; return (v >= 256) ? v : 1024; }();

        // Flash-attention KV alignment (Vulkan only). ggml-vulkan's FA dispatch
        // only picks the fast "aligned" shader variant when the KV length is a
        // multiple of the pipeline's block_cols (32/64 for these shapes — see
        // ggml_vk_flash_attn's `aligned`); llama.cpp always satisfies it because
        // its unified KV cache pads n_kv. Our SWA fresh-chunk/extended-buffer
        // reads used the raw length, so any chunk with N % 32 != 0 silently
        // dropped all 40 SWA layers' flash onto the slow unaligned shader
        // (~10% whole-prefill regression measured on gemma4-12B at N=1985 vs
        // N=2016/2048). Pad the flash KV length to a multiple of 64: ggml_pad
        // zero-fills the tail rows (finite) and the causal mask already marks
        // ki >= validLen as -inf, so results are unchanged. CUDA handles
        // unaligned KV without a slow path — keep other backends' graphs
        // unchanged. TS_G4_FLASH_KV_PAD=0 disables.
        static const bool flash_kv_pad_enabled = []{ const char* e = std::getenv("TS_G4_FLASH_KV_PAD"); return e == nullptr || e[0] != '0'; }();
        const bool pad_flash_kv = flash_kv_pad_enabled && g_backend_type == BACKEND_TYPE_VULKAN;
        // 64 covers every FA pipeline's block_cols on the shapes used here.
        auto flash_pad_len = [pad_flash_kv](int len) {
            return pad_flash_kv ? ((len + 63) & ~63) : len;
        };
        // Zero-pad a fresh [hd, len, kvHeads] F32 K/V tensor to padLen rows.
        auto pad_kv_rows = [ctx](ggml_tensor* t, int len, int padLen) {
            return (padLen > len) ? ggml_pad(ctx, t, 0, padLen - len, 0, 0) : t;
        };
        struct TileMask { int kLen; int qLen; int qStart; int kStart; int window; ggml_tensor* tensor; int dataIdx; };
        std::vector<TileMask> tile_mask_cache;
        auto get_window_tile_mask = [&](int kLen, int qLen, int qStart, int kStart, int window) -> ggml_tensor* {
            for (auto& m : tile_mask_cache)
                if (m.kLen == kLen && m.qLen == qLen && m.qStart == qStart && m.kStart == kStart && m.window == window) return m.tensor;
            ggml_tensor* mt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kLen, qLen);
            std::vector<ggml_fp16_t> data(static_cast<std::size_t>(kLen) * qLen);
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            for (int qi = 0; qi < qLen; qi++)
            {
                const int gQ = qStart + qi;
                // keep global key gK in (gQ - window, gQ]  ->  ki in [gQ-window+1-kStart, gQ-kStart]
                int lo = gQ - window + 1 - kStart; if (lo < 0) lo = 0;
                int hi = gQ - kStart; if (hi > kLen - 1) hi = kLen - 1;
                ggml_fp16_t* row = &data[static_cast<std::size_t>(qi) * kLen];
                std::fill(row, row + kLen, neg_inf);
                if (hi >= lo) std::fill(row + lo, row + hi + 1, zero_val);
            }
            const int idx = static_cast<int>(mask_data_store.size());
            mask_data_store.push_back(std::move(data));
            tile_mask_cache.push_back({kLen, qLen, qStart, kStart, window, mt, idx});
            return mt;
        };

        ggml_tensor* hidden = current;

        // Retain each non-shared layer's FRESH full-chunk K/V (all N positions,
        // post norm/RoPE) so a shared (KV-donor) SWA layer whose sequence exceeds
        // the circular window at start_pos==0 can attend over the donor's whole
        // chunk directly — the W-sized cache only kept the last W positions, which
        // drops keys for the early query rows. Null for shared layers / unused.
        std::vector<ggml_tensor*> layer_k_full(num_layers, nullptr);
        std::vector<ggml_tensor*> layer_v_full(num_layers, nullptr);

        // swaPrev (start_pos>0, SWA window wrapped): retain each non-shared SWA
        // layer's gathered previous-window F32 copy so a shared (KV-donor) SWA layer
        // downstream can prepend the SAME prev window the donor used. Mirrors
        // layer_k_full but for the [start_pos-prevCount, start_pos) rolling history.
        std::vector<ggml_tensor*> layer_k_prev(num_layers, nullptr);
        std::vector<ggml_tensor*> layer_v_prev(num_layers, nullptr);

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];
            float rope_base = rope_base_arr[l];

            // attn norm
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), lt.attn_norm_w);  // [hidden, N]

            // Q projection (+ K/V for non-shared layers). Shared (KV-donor) layers
            // project ONLY Q (qkv_w is the Q-only weight) and read the donor's K/V.
            int rope_dims = rope_n_dims_arr[l];
            ggml_tensor* rope_ff = info.isLocal ? nullptr : freq_factors_t;
            ggml_tensor* q_lin;
            ggml_tensor* k_lin = nullptr;
            ggml_tensor* v_lin = nullptr;
            if (info.isShared)
            {
                q_lin = ggml_mul_mat(ctx, lt.qkv_w, normed);   // Q-only weight -> [qDim, N]
            }
            else if (lt.k_w != nullptr)
            {
                q_lin = ggml_mul_mat(ctx, lt.qkv_w, normed);
                k_lin = ggml_mul_mat(ctx, lt.k_w, normed);
                v_lin = ggml_mul_mat(ctx, lt.v_w, normed);
            }
            else
            {
                ggml_tensor* qkv = ggml_mul_mat(ctx, lt.qkv_w, normed);  // [qDim+2kDim, NQ]
                q_lin = ggml_cont(ctx, ggml_view_2d(ctx, qkv, info.qDim, NQ, qkv->nb[1], 0));
                k_lin = ggml_cont(ctx, ggml_view_2d(ctx, qkv, info.kDim, NQ, qkv->nb[1],
                    static_cast<std::size_t>(info.qDim) * sizeof(float)));
                v_lin = ggml_cont(ctx, ggml_view_2d(ctx, qkv, info.kDim, NQ, qkv->nb[1],
                    static_cast<std::size_t>(info.qDim + info.kDim) * sizeof(float)));
            }

            // per-head Q norm + RoPE (always; Q is this layer's own)
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_lin, info.hd, num_heads, NQ);
            q_3d = ggml_mul(ctx, ggml_rms_norm(ctx, q_3d, eps), lt.q_norm_w);
            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff,
                rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);  // [hd, num_heads, N]

            lt.k_cpy = nullptr; lt.v_cpy = nullptr; lt.k_cpy2 = nullptr; lt.v_cpy2 = nullptr;
            lt.k_prev = nullptr; lt.v_prev = nullptr;
            // Fresh post-norm/RoPE K/V for this chunk, retained so SWA layers whose
            // sequence exceeds the (circular) window at start_pos==0 can attend over
            // the whole chunk directly instead of the W-sized cache view.
            ggml_tensor* k_write = nullptr;
            ggml_tensor* v_write = nullptr;

            // swaPrev: a SWA (local) layer at start_pos>0 whose window has wrapped
            // (totalSeqLen > W) must attend up to W-1 keys from the PREVIOUS chunk(s).
            // Those live in the rolling cache right now but this chunk's own K/V write
            // (below) will overwrite them, so we gather them into an F32 copy first and
            // prepend to the fresh chunk (mirrors the swaFresh start_pos==0 case and the
            // MoE verify swaPrev). prevCount/swaBase are shared by a KV-donor and the
            // SWA layers that follow it (same cacheSize / start_pos).
            const bool swaPrev = info.isLocal && start_pos != 0 && totalSeqLen > info.cacheSize;
            const int prevCount = swaPrev ? std::min(info.cacheSize, start_pos) : 0;
            const int swaBase = start_pos - prevCount;   // logical position of the prev-window start

            if (!info.isShared)
            {
                // per-head K norm + V norm (unweighted), then RoPE on K.
                ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_lin, info.hd, info.kvHeads, NQ);
                ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_lin, info.hd, info.kvHeads, NQ);
                k_3d = ggml_mul(ctx, ggml_rms_norm(ctx, k_3d, eps), lt.k_norm_w);
                v_3d = ggml_rms_norm(ctx, v_3d, eps);
                ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, rope_ff,
                    rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);  // [hd, kvHeads, NQ]

                // Write the K/V into the persistent cache. Global: linear append of
                // all N at start_pos. SWA (circular, size = window): only the LAST
                // min(N, cacheSize) positions survive — when N exceeds the window the
                // earlier positions would be immediately overwritten, so skip them
                // (writeOffsetInChunk) rather than wrapping the buffer multiple times
                // (which would overflow the cache view). The remaining run may still
                // cross the wrap point once, so it splits into up to two cpy ops.
                k_write = ggml_cont(ctx, ggml_permute(ctx, k_rope, 0, 2, 1, 3));  // [hd, N, kvHeads]
                v_write = ggml_cont(ctx, ggml_permute(ctx, v_3d, 0, 2, 1, 3));     // [hd, N, kvHeads]

                // swaPrev gather: materialise [swaBase, start_pos) from the rolling
                // cache into F32 BEFORE the write below overwrites it (the cpy is built
                // into the graph ahead of the write cpy, so on the single execution
                // stream it reads the OLD cache contents). Retained in layer_k_prev so a
                // following shared SWA layer reuses the donor's prev window.
                if (swaPrev && prevCount > 0)
                {
                    ggml_tensor* kpv = view_kv_cache_window(ctx, lt.k_cached_t, info.hd, info.cacheSize, info.kvHeads, swaBase, prevCount, kv_cache_type);
                    ggml_tensor* vpv = view_kv_cache_window(ctx, lt.v_cached_t, info.hd, info.cacheSize, info.kvHeads, swaBase, prevCount, kv_cache_type);
                    if (kpv == nullptr || vpv == nullptr)
                    {
                        set_last_error("Gemma4 model verify: failed to view prev SWA window.");
                        return 0;
                    }
                    ggml_tensor* kpf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, info.hd, prevCount, info.kvHeads);
                    ggml_tensor* vpf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, info.hd, prevCount, info.kvHeads);
                    lt.k_prev = ggml_cpy(ctx, kpv, kpf);
                    lt.v_prev = ggml_cpy(ctx, vpv, vpf);
                    layer_k_prev[l] = lt.k_prev;
                    layer_v_prev[l] = lt.v_prev;
                }

                const int writeOffsetInChunk = (info.isLocal && N > info.cacheSize) ? (N - info.cacheSize) : 0;
                const int writeLen = N - writeOffsetInChunk;                       // <= cacheSize for SWA
                const int writeStartLogical = start_pos + writeOffsetInChunk;
                const int cacheBase = info.isLocal ? (writeStartLogical % info.cacheSize) : writeStartLogical;
                const int n1 = (info.isLocal && cacheBase + writeLen > info.cacheSize) ? (info.cacheSize - cacheBase) : writeLen;
                auto writePart = [&](ggml_tensor* cache, ggml_tensor* src, int srcOff, int dstSlot, int cnt) -> ggml_tensor* {
                    ggml_tensor* s = ggml_view_3d(ctx, src, info.hd, cnt, info.kvHeads,
                        src->nb[1], src->nb[2], static_cast<std::size_t>(srcOff) * src->nb[1]);
                    ggml_tensor* d = ggml_view_3d(ctx, cache, info.hd, cnt, info.kvHeads,
                        cache->nb[1], cache->nb[2], static_cast<std::size_t>(dstSlot) * cache->nb[1]);
                    return ggml_cpy(ctx, s, d);
                };
                lt.k_cpy = writePart(lt.k_cached_t, k_write, writeOffsetInChunk, cacheBase, n1);
                lt.v_cpy = writePart(lt.v_cached_t, v_write, writeOffsetInChunk, cacheBase, n1);
                if (n1 < writeLen)
                {
                    lt.k_cpy2 = writePart(lt.k_cached_t, k_write, writeOffsetInChunk + n1, 0, writeLen - n1);
                    lt.v_cpy2 = writePart(lt.v_cached_t, v_write, writeOffsetInChunk + n1, 0, writeLen - n1);
                }
                // Publish the fresh full-chunk K/V so shared SWA layers can attend it.
                layer_k_full[l] = k_write;
                layer_v_full[l] = v_write;
            }

            // SWA prefill that overflows the circular window at start_pos==0: the
            // W-sized cache can't hold all N keys, so attend over the FRESH chunk
            // K/V (k_write/v_write, all N positions) with a sliding-window causal
            // mask. Only valid when start_pos==0 (the fresh chunk IS the whole
            // history) and the layer computed its own K/V (non-shared). The cache
            // still received the last W positions above for subsequent decode.
            const bool swaFresh = info.isLocal && !info.isShared && start_pos == 0
                && totalSeqLen > info.cacheSize && k_write != nullptr;

            // Same overflow case for a SHARED (KV-donor) SWA layer: its donor (a
            // non-shared SWA layer, processed earlier in this graph) retained its
            // FRESH full chunk in layer_k_full/v_full. Attend that directly with the
            // sliding-window mask instead of the donor's W-sized circular cache view
            // (which lost the early positions). This is what lets E-series models
            // (KV-donor layers) run prompts > window through the fused kernel.
            const bool swaFreshShared = info.isLocal && info.isShared && start_pos == 0
                && totalSeqLen > info.cacheSize && layer_k_full[info.kvSource] != nullptr;

            // Read the attention window. SWA overflow (swaFresh/swaFreshShared):
            // attend the FRESH full chunk [0, N) with a sliding-window mask. Else:
            // SWA reads the last min(total, W) circular-cache positions; global reads
            // [0, total) (flash-padded for head_dim 512). Attention runs through
            // ggml_flash_attn_ext below (O(seq) memory, no materialized score tensor).
            int attendLen;
            int attnKvLen;
            int maskWindow;
            ggml_tensor* k_full;
            ggml_tensor* v_full;
            if (swaFresh || swaFreshShared)
            {
                ggml_tensor* k_fresh = swaFresh ? k_write : layer_k_full[info.kvSource];  // [hd, NQ, kvHeads]
                ggml_tensor* v_fresh = swaFresh ? v_write : layer_v_full[info.kvSource];  // [hd, NQ, kvHeads]
                attendLen = N;                       // real keys: the fresh chunk's first N rows
                // Batch padding already leaves the fresh K/V 64-row aligned (the
                // dummy tail rows are finite and causally masked); without it,
                // zero-pad for the FA aligned-shader requirement.
                attnKvLen = (NQ > N) ? NQ : flash_pad_len(N);
                maskWindow = info.cacheSize;         // sliding window W
                k_fresh = pad_kv_rows(k_fresh, NQ, attnKvLen);
                v_fresh = pad_kv_rows(v_fresh, NQ, attnKvLen);
                // flash_attn_ext needs K/V in the cache dtype; convert the fresh F32
                // chunk when the cache is F16 (no-op when F32).
                if (kv_cache_type == GGML_TYPE_F32)
                {
                    k_full = k_fresh;
                    v_full = v_fresh;
                }
                else
                {
                    ggml_tensor* kf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, attnKvLen, info.kvHeads);
                    ggml_tensor* vf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, attnKvLen, info.kvHeads);
                    k_full = ggml_cpy(ctx, k_fresh, kf);
                    v_full = ggml_cpy(ctx, v_fresh, vf);
                }
            }
            else if (swaPrev && ((!info.isShared && lt.k_prev != nullptr)
                                 || (info.isShared && layer_k_prev[info.kvSource] != nullptr
                                     && layer_k_full[info.kvSource] != nullptr)))
            {
                // SWA window overflow at start_pos>0: attend the extended K/V
                // [prev window (prevCount) ++ fresh chunk (N)], covering logical
                // [swaBase, start_pos+N) in chronological order. Non-shared layers use
                // their own gathered prev window + fresh K/V; shared (KV-donor) layers
                // reuse the donor's retained prev window + fresh chunk. The buffer is in
                // chronological order, so get_causal_mask(bufLen, bufLen, W) below maps
                // query qi (causal cutoff prevCount+qi, W-wide window) correctly — no
                // explicit keyBase needed (nPast = bufLen - N = prevCount).
                ggml_tensor* k_p = info.isShared ? layer_k_prev[info.kvSource] : lt.k_prev;
                ggml_tensor* v_p = info.isShared ? layer_v_prev[info.kvSource] : lt.v_prev;
                ggml_tensor* k_f = info.isShared ? layer_k_full[info.kvSource] : k_write;
                ggml_tensor* v_f = info.isShared ? layer_v_full[info.kvSource] : v_write;
                // The fresh chunk contributes NQ rows (batch padding): keys are
                // [prevCount real ++ N real ++ NQ-N dummy]; validLen = prevCount+N
                // excludes the finite dummy tail via the causal mask.
                const int bufLen = prevCount + NQ;
                attendLen = prevCount + N;
                attnKvLen = flash_pad_len(bufLen);   // FA-alignment padding (Vulkan); mask covers the tail
                maskWindow = info.cacheSize;     // sliding window W
                ggml_tensor* kext = ggml_concat(ctx, k_p, k_f, 1);   // [hd, prevCount+NQ, kvHeads] F32
                ggml_tensor* vext = ggml_concat(ctx, v_p, v_f, 1);
                kext = pad_kv_rows(kext, bufLen, attnKvLen);
                vext = pad_kv_rows(vext, bufLen, attnKvLen);
                if (kv_cache_type == GGML_TYPE_F32)
                {
                    k_full = kext;
                    v_full = vext;
                }
                else
                {
                    ggml_tensor* kf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, attnKvLen, info.kvHeads);
                    ggml_tensor* vf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), info.hd, attnKvLen, info.kvHeads);
                    k_full = ggml_cpy(ctx, kext, kf);
                    v_full = ggml_cpy(ctx, vext, vf);
                }
            }
            else
            {
                attendLen = info.isLocal ? std::min(totalSeqLen, info.cacheSize) : totalSeqLen;
                const int activeStart = info.isLocal ? ((totalSeqLen - attendLen) % info.cacheSize) : 0;
                // Flash attention reads a (possibly padded) KV window; padding slots
                // beyond attendLen are masked out. head_dim 512/576 requires it
                // (custom CUDA kernels); on Vulkan pad every layer so the FA
                // dispatch stays on the aligned fast shader (never-written cache
                // rows are finite — same contract the 512-dim path relies on).
                // This branch's SWA reads only occur unwrapped (activeStart==0:
                // overflow goes through swaFresh/swaPrev above), so a padded view
                // never crosses the circular boundary.
                attnKvLen = flash_attn_kv_length(attendLen, info.cacheSize, info.hd);
                if (pad_flash_kv)
                    attnKvLen = std::min(info.cacheSize, std::max(attnKvLen, flash_pad_len(attendLen)));
                maskWindow = 0;                 // window already enforced by the cache view
                k_full = view_kv_cache_window(ctx, lt.k_cached_t, info.hd, info.cacheSize, info.kvHeads, activeStart, attnKvLen, kv_cache_type);
                v_full = view_kv_cache_window(ctx, lt.v_cached_t, info.hd, info.cacheSize, info.kvHeads, activeStart, attnKvLen, kv_cache_type);
            }
            if (k_full == nullptr || v_full == nullptr)
            {
                set_last_error("Failed to create Gemma4 verify KV cache views.");
                return 0;
            }

            // Flash attention (Gemma scale = 1.0). q_t [hd, N, num_heads]; the mask
            // (kvLen=attnKvLen, validLen=attendLen, window=maskWindow) encodes causal
            // + sliding-window + flash-padding + multimodal-bidi semantics.
            ggml_tensor* q_t = ggml_permute(ctx, q_rope, 0, 2, 1, 3);                   // [hd, N, num_heads]
            ggml_tensor* attn_out;
            // SWA window overflow (swaFresh/swaFreshShared) attends the fresh chunk's
            // N keys with a window mask (maskWindow>0). Tile the queries so each tile
            // only reads its window slice instead of the full N (see comment above).
            // Not applied to global / circular-cache reads (maskWindow==0), the rare
            // multimodal-bidi case (handled by the full-N mask), or tiny N.
            // Only tile when there are >= 3 tiles (N > 2*tile): below that the full-N
            // flash's wasted (masked) work is small and the per-tile launch + concat
            // overhead would erase the win.
            // maskWindow > 0 implies the swaFresh/swaFreshShared branch above:
            // k_full/v_full hold the fresh chunk's N rows (+ FA-alignment padding).
            const bool use_tiled = swa_tiled && maskWindow > 0 && is_except == nullptr
                && N > 2 * swa_tile && attendLen == N;
            if (use_tiled)
            {
                ggml_tensor* acc = nullptr;
                for (int qs = 0; qs < NQ; qs += swa_tile)
                {
                    const int qe = (qs + swa_tile < NQ) ? (qs + swa_tile) : NQ;
                    const int qLen = qe - qs;
                    const int ks = (qs > maskWindow) ? (qs - maskWindow) : 0;
                    // Pad each tile's key slice for FA alignment (Vulkan). ks is
                    // 64-aligned (qs is a multiple of swa_tile, maskWindow of the
                    // 512/1024 window), so the padded slice stays within the padded
                    // parent [0, attnKvLen); rows beyond the real qe - ks keys are
                    // causal-future (finite) or zero padding, and the tile mask
                    // marks them -inf either way.
                    const int kLenReal = qe - ks;
                    const int kLen = std::min(flash_pad_len(kLenReal), attnKvLen - ks);
                    ggml_tensor* q_tile = ggml_view_3d(ctx, q_t, info.hd, qLen, num_heads,
                        q_t->nb[1], q_t->nb[2], static_cast<std::size_t>(qs) * q_t->nb[1]);
                    ggml_tensor* k_tile = ggml_view_3d(ctx, k_full, info.hd, kLen, info.kvHeads,
                        k_full->nb[1], k_full->nb[2], static_cast<std::size_t>(ks) * k_full->nb[1]);
                    ggml_tensor* v_tile = ggml_view_3d(ctx, v_full, info.hd, kLen, info.kvHeads,
                        v_full->nb[1], v_full->nb[2], static_cast<std::size_t>(ks) * v_full->nb[1]);
                    ggml_tensor* m_tile = get_window_tile_mask(kLen, qLen, qs, ks, maskWindow);
                    ggml_tensor* fa = ggml_flash_attn_ext(ctx, q_tile, k_tile, v_tile, m_tile, 1.0f, 0.0f, 0.0f);
                    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
                    if (qs == 0 && !backend_supports_op(fa))
                    {
                        set_last_error("Gemma4 model verify: tiled flash attention unsupported for this shape; use per-op path.");
                        return 0;
                    }
                    acc = (acc == nullptr) ? fa : ggml_concat(ctx, acc, fa, 2);          // along the query (N) dim
                }
                attn_out = acc;                                                          // [hd, num_heads, N]
            }
            else
            {
                ggml_tensor* fa_mask = get_causal_mask(attnKvLen, attendLen, maskWindow);
                attn_out = ggml_flash_attn_ext(ctx, q_t, k_full, v_full, fa_mask, 1.0f, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
                if (!backend_supports_op(attn_out))
                {
                    // No supported flash kernel for this shape (e.g. an exotic head_dim):
                    // fall back to the per-op verify rather than the numerically-fragile
                    // manual chain.
                    set_last_error("Gemma4 model verify: flash attention unsupported for this shape; use per-op path.");
                    return 0;
                }
            }
            // flash_attn_ext returns [hd, num_heads, NQ, 1] — each column holds all
            // heads contiguously, exactly what the O projection wants.
            ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, info.qDim, NQ);

            // O projection -> post-attn norm -> residual
            ggml_tensor* o_out = ggml_mul_mat(ctx, lt.o_w, attn_flat);                  // [hidden, N]
            ggml_tensor* post_attn = ggml_mul(ctx, ggml_rms_norm(ctx, o_out, eps), lt.post_attn_norm_w);
            ggml_tensor* residual1 = ggml_add(ctx, hidden, post_attn);

            // FFN: norm -> gate_up -> gelu*up -> down -> post_ffn norm -> residual
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), lt.ffn_norm_w);
            ggml_tensor* gu = ggml_mul_mat(ctx, lt.gu_w, ffn_normed);                   // [2*ff, N]
            // Fused GeGLU: gelu(gate) * up computed directly on the contiguous
            // [2*ff, N] tensor (gate = first half, up = second half -> non-swapped
            // ggml_geglu). Avoids two full [ff, N] ggml_cont materializations plus
            // the separate gelu/mul ops (cpy_scalar F32->F32 was ~14% of prefill
            // GPU time). Bit-identical: ggml_vec_geglu / op_gelu use the same tanh
            // gelu as the old ggml_gelu. Mirrors llama.cpp build_ffn (LLM_FFN_GELU).
            ggml_tensor* ffn_hidden = ggml_geglu(ctx, gu);                              // [ff, N]
            ggml_tensor* down = ggml_mul_mat(ctx, lt.down_w, ffn_hidden);               // [hidden, N]
            ggml_tensor* post_ffn = ggml_mul(ctx, ggml_rms_norm(ctx, down, eps), lt.post_ffn_norm_w);
            ggml_tensor* residual2 = ggml_add(ctx, residual1, post_ffn);

            // PLE injection (mirrors Gemma4ModelDecode, batched over the N rows).
            // ple_slice is a strided view of ple_input: column i (row i) at layer l.
            if (lt.ple_gate_w != nullptr && ple_input != nullptr)
            {
                ggml_tensor* ple_slice = ggml_cont(ctx, ggml_view_2d(ctx, ple_input, ple_dim, NQ,
                    static_cast<std::size_t>(num_layers) * ple_dim * sizeof(float),
                    static_cast<std::size_t>(l) * ple_dim * sizeof(float)));               // [ple_dim, NQ]
                ggml_tensor* ple_gate_proj = ggml_mul_mat(ctx, lt.ple_gate_w, residual2);  // [ple_dim, N]
                ggml_tensor* ple_gated = ggml_mul(ctx, ggml_gelu(ctx, ple_gate_proj), ple_slice);  // [ple_dim, N]
                ggml_tensor* ple_proj = ggml_mul_mat(ctx, lt.ple_proj_w, ple_gated);       // [hidden, N]
                ggml_tensor* ple_normed = ggml_mul(ctx, ggml_rms_norm(ctx, ple_proj, eps), lt.ple_post_norm_w);
                residual2 = ggml_add(ctx, residual2, ple_normed);
            }

            float scalar = layer_scalar_arr[l];
            if (std::fabs(scalar - 1.0f) > 1e-6f)
                residual2 = ggml_scale(ctx, residual2, scalar);

            hidden = residual2;
        }

        // hidden holds NQ columns; only the first N (real) columns are downloaded
        // below — they are the contiguous prefix of the buffer.
        ggml_tensor* hidden_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, NQ);
        ggml_tensor* out_hidden = ggml_cpy(ctx, hidden, hidden_out);
        ggml_set_output(out_hidden);

        // Tiled SWA attention adds ~8 nodes (flash + concat + views + mask) per query
        // tile per local layer; budget for it so the graph never overflows.
        const int swa_tiles = (swa_tiled && NQ > swa_tile) ? ((NQ + swa_tile - 1) / swa_tile) : 1;
        // +16/layer headroom for swaPrev (gather view+cpy, concat, optional dtype cpy).
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (208 + static_cast<std::size_t>(swa_tiles) * 8) + 512;
        if (g4v_timing) vt_built = vt_clock::now();
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        // swaPrev gathers MUST be expanded (and thus scheduled) before the cache
        // writes: the gather reads the rolling window the write then overwrites, so
        // it has to run first on the single execution stream (mirrors the MoE verify).
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].k_prev != nullptr) ggml_build_forward_expand(graph, layers[l].k_prev);
            if (layers[l].v_prev != nullptr) ggml_build_forward_expand(graph, layers[l].v_prev);
        }
        for (int l = 0; l < num_layers; l++)
        {
            // Shared (KV-donor) layers write no K/V of their own.
            if (layers[l].k_cpy != nullptr) ggml_build_forward_expand(graph, layers[l].k_cpy);
            if (layers[l].v_cpy != nullptr) ggml_build_forward_expand(graph, layers[l].v_cpy);
            if (layers[l].k_cpy2 != nullptr) ggml_build_forward_expand(graph, layers[l].k_cpy2);
            if (layers[l].v_cpy2 != nullptr) ggml_build_forward_expand(graph, layers[l].v_cpy2);
        }
        ggml_build_forward_expand(graph, out_hidden);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        // g4v-timing attribution: where bind time goes (cache lookups vs
        // tensor_alloc vs fallthrough), and how many tensors were re-uploaded.
        double vtb_lookup_ms = 0.0, vtb_talloc_ms = 0.0;
        int vtb_hits = 0, vtb_misses = 0;
        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            // 512-byte floor (was 4096): the per-head q/k norm weights are 1-2 KB,
            // and leaving them below the cacheable floor re-uploaded ~96 tiny
            // tensors per prefill call — each ggml_backend_tensor_set on Vulkan is
            // a full submit+wait, so those uploads dominated the setup phase.
            if (cacheable && bytes >= 512)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                const auto tl0 = g4v_timing ? vt_clock::now() : vt_clock::time_point{};
                bool got = try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs_upload, usage);
                const auto tl1 = g4v_timing ? vt_clock::now() : vt_clock::time_point{};
                if (g4v_timing) vtb_lookup_ms += vt_ms(tl0, tl1);
                if (got)
                {
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, addr);
                    if (g4v_timing) { vtb_talloc_ms += vt_ms(tl1, vt_clock::now()); vtb_hits++; }
                    if (st == GGML_STATUS_SUCCESS) { if (needs_upload) upload_list.push_back({t, data, bytes}); return; }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable) ephemeral_bufs.emplace_back(buf);
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, data);
                    if (st == GGML_STATUS_SUCCESS) return;
                }
            }
            if (g4v_timing) vtb_misses++;
            upload_list.push_back({t, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];
            bind_or_mark(lt.qkv_w, qkv_arr[l], static_cast<std::size_t>(qkv_bytes_arr[l]), true);
            if (lt.k_w != nullptr)
            {
                bind_or_mark(lt.k_w, k_arr[l], static_cast<std::size_t>(k_bytes_arr[l]), true);
                bind_or_mark(lt.v_w, v_arr[l], static_cast<std::size_t>(v_bytes_arr[l]), true);
            }
            bind_or_mark(lt.o_w, o_arr[l], static_cast<std::size_t>(o_bytes_arr[l]), true);
            bind_or_mark(lt.gu_w, gu_arr[l], static_cast<std::size_t>(gu_bytes_arr[l]), true);
            bind_or_mark(lt.down_w, down_arr[l], static_cast<std::size_t>(down_bytes_arr[l]), true);
            bind_or_mark(lt.attn_norm_w, attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.post_attn_norm_w, post_attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.ffn_norm_w, ffn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.post_ffn_norm_w, post_ffn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.q_norm_w, q_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);
            if (!info.isShared)
            {
                bind_or_mark(lt.k_norm_w, k_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);
                // Shared layers reuse the donor's cache tensor (bound when the
                // donor layer is processed); binding it again would double-alloc.
                bind_or_mark(lt.k_cached_t, k_cache_arr[l], kv_cache_bytes(info.kvHeads, info.cacheSize, info.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                bind_or_mark(lt.v_cached_t, v_cache_arr[l], kv_cache_bytes(info.kvHeads, info.cacheSize, info.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
            if (lt.ple_gate_w != nullptr)
            {
                bind_or_mark(lt.ple_gate_w, ple_gate_arr[l], static_cast<std::size_t>(ple_gate_bytes_arr[l]), true);
                bind_or_mark(lt.ple_proj_w, ple_proj_arr[l], static_cast<std::size_t>(ple_proj_bytes_arr[l]), true);
                bind_or_mark(lt.ple_post_norm_w, ple_post_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            }
        }

        // In-kernel PLE: bind the per_layer_token_embd table (and the projection +
        // norm weights) resident so the gather / projection read them on-device.
        if (ple_in_kernel && ple_table != nullptr)
        {
            bind_or_mark(ple_table, const_cast<void*>(ple_token_embd_data),
                static_cast<std::size_t>(ple_token_embd_bytes), true);
            if (ple_proj_w != nullptr)
                bind_or_mark(ple_proj_w, const_cast<void*>(ple_proj_w_data),
                    static_cast<std::size_t>(ple_proj_w_bytes), true);
            if (ple_proj_norm_w != nullptr)
                bind_or_mark(ple_proj_norm_w, const_cast<void*>(ple_proj_norm_data),
                    static_cast<std::size_t>(ple_dim) * sizeof(float), true);
        }

        if (g4v_timing) vt_bound = vt_clock::now();

        // Allocation strategy. Small N (MTP speculative verify, N<=16) keeps the
        // bump allocator (alloc_ctx_tensors_reuse: each tensor its own slot, stable
        // addresses, lowest per-call overhead). Large N (prefill routed through this
        // kernel) MUST use the gallocr lifetime-packing allocator: the bump
        // allocator's footprint is the SUM of every layer's N-token intermediates
        // (~31 GB at N=776 over 48 layers → OOM), whereas gallocr packs by tensor
        // lifetime so the peak is one layer's working set (~10-20x smaller). The
        // pre-bound weights / KV caches above already own buffers and are skipped
        // by both allocators.
        const bool useGallocr = (N > 16);
        BufferHandle buffer(nullptr);
        if (useGallocr)
        {
            if (!alloc_graph_reuse_gallocr(graph))
            {
                set_last_error("Failed to allocate backend buffer for Gemma4 model verify (gallocr).");
                return 0;
            }
        }
        else if (!alloc_ctx_tensors_reuse(ctx))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("Failed to allocate backend buffer for Gemma4 model verify.");
                return 0;
            }
        }

        if (g4v_timing) vt_alloced = vt_clock::now();

        host_read_barrier();

        if (g4v_timing)
        {
            static std::atomic<int> s_uplist_dump{0};
            if (s_uplist_dump.fetch_add(1) == 1)   // second call = steady state
            {
                std::size_t tot = 0;
                for (auto& u : upload_list) tot += u.bytes;
                fprintf(stderr, "[g4v-uplist] %zu tensors, %zu bytes total:\n", upload_list.size(), tot);
                for (auto& u : upload_list)
                    fprintf(stderr, "  %s type=%s ne=[%lld,%lld,%lld] bytes=%zu\n",
                        u.tensor->name[0] ? u.tensor->name : "(unnamed)",
                        ggml_type_name(u.tensor->type),
                        (long long)u.tensor->ne[0], (long long)u.tensor->ne[1], (long long)u.tensor->ne[2], u.bytes);
                fflush(stderr);
            }
        }

        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);

        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(hidden_size) * N * sizeof(float));
        if (NQ > N)
        {
            // Zero the dummy (batch-padding) embedding rows so every value that
            // flows out of them is finite: rms_norm(0)*w == 0, matmul of a zero
            // column is zero, and the causal mask already keeps them out of
            // every real query's attention.
            const std::vector<float> zero_rows(static_cast<std::size_t>(hidden_size) * (NQ - N), 0.0f);
            ggml_backend_tensor_set(current, zero_rows.data(),
                static_cast<std::size_t>(hidden_size) * N * sizeof(float),
                zero_rows.size() * sizeof(float));
        }

        std::vector<std::int32_t> pos_vals(NQ);
        for (int i = 0; i < NQ; i++) pos_vals[i] = start_pos + i;
        ggml_backend_tensor_set(pos_tensor, pos_vals.data(), 0, static_cast<std::size_t>(NQ) * sizeof(std::int32_t));

        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, rope_freq_factors, 0,
                static_cast<std::size_t>(rope_freq_factors_len) * sizeof(float));

        for (auto& m : mask_cache)
            if (m.dataIdx >= 0)   // dataIdx == -1: GPU-filled below, no host upload
                ggml_backend_tensor_set(m.tensor, mask_data_store[m.dataIdx].data(), 0,
                    mask_data_store[m.dataIdx].size() * sizeof(ggml_fp16_t));
        for (auto& m : tile_mask_cache)
            ggml_backend_tensor_set(m.tensor, mask_data_store[m.dataIdx].data(), 0,
                mask_data_store[m.dataIdx].size() * sizeof(ggml_fp16_t));

#ifdef TSG_GGML_USE_CUDA
        // Generate the deferred causal masks directly in their (now-allocated)
        // device buffers, then block until they complete so the backend-stream
        // graph compute below sees them (bit-identical to the host fill).
        if (!gpu_mask_fills.empty())
        {
            for (auto& g : gpu_mask_fills)
                tsg_cuda_fill_causal_mask_f16(g.tensor->data, g.kvLen, g.mN, g.nPast, g.window, g.validLen);
            tsg_cuda_sync_stream0();
        }
#endif

        if (ple_in_kernel)
        {
            // Upload only the tiny token-id list; get_rows gathers the PLE on-device.
            if (ple_ids != nullptr)
            {
                ggml_backend_tensor_set(ple_ids, ple_token_ids, 0,
                    static_cast<std::size_t>(N) * sizeof(std::int32_t));
                if (NQ > N)
                {
                    // Dummy rows gather a valid (finite) PLE row; their columns
                    // are discarded.
                    const std::vector<std::int32_t> tail_ids(static_cast<std::size_t>(NQ - N), ple_token_ids[N - 1]);
                    ggml_backend_tensor_set(ple_ids, tail_ids.data(),
                        static_cast<std::size_t>(N) * sizeof(std::int32_t),
                        tail_ids.size() * sizeof(std::int32_t));
                }
            }
        }
        else if (ple_input != nullptr)
        {
            ggml_backend_tensor_set(ple_input, ple_data, 0,
                static_cast<std::size_t>(N) * num_layers * ple_dim * sizeof(float));
            if (NQ > N)
            {
                const std::vector<float> zero_ple(static_cast<std::size_t>(NQ - N) * num_layers * ple_dim, 0.0f);
                ggml_backend_tensor_set(ple_input, zero_ple.data(),
                    static_cast<std::size_t>(N) * num_layers * ple_dim * sizeof(float),
                    zero_ple.size() * sizeof(float));
            }
        }

        if (g4v_timing) vt_uploaded = vt_clock::now();

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for Gemma4 model verify.");
            return 0;
        }

        if (g4v_timing) vt_computed = vt_clock::now();

        finalize_compute_with_download(hidden_out, hidden_data, static_cast<std::size_t>(hidden_size) * N * sizeof(float));

        if (g4v_timing)
        {
            const auto vt_done = vt_clock::now();
            fprintf(stderr,
                "[g4v-timing] N=%d start=%d nodes=%d build=%.1f bind=%.1f (lookup=%.1f talloc=%.1f hits=%d miss=%d uplist=%zu) alloc=%.1f upload=%.1f compute=%.1f download=%.1f total=%.1f ms\n",
                N, start_pos, ggml_graph_n_nodes(graph),
                vt_ms(vt0, vt_built), vt_ms(vt_built, vt_bound),
                vtb_lookup_ms, vtb_talloc_ms, vtb_hits, vtb_misses, upload_list.size(),
                vt_ms(vt_bound, vt_alloced),
                vt_ms(vt_alloced, vt_uploaded), vt_ms(vt_uploaded, vt_computed),
                vt_ms(vt_computed, vt_done), vt_ms(vt0, vt_done));
            fflush(stderr);
        }

        // If this call allocated a per-call backend buffer (the reuse-buffer
        // fallback), drain the queued async download before BufferHandle frees
        // it at scope exit — otherwise the in-flight blit reads a freed buffer
        // and Metal keeps it resident. No-op (and zero cost) on the common path
        // where the persistent reuse buffer is used (buffer.value == nullptr).
        if (buffer.value != nullptr) host_read_barrier();
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
        set_last_error("Unknown error in Gemma4 model verify.");
        return 0;
    }
}

// ============================================================================
// Fused Gemma 4 MTP draft step (the "gemma4-assistant" recurrent draft head):
// runs the whole draft head — backbone-embed + concat(h_prev) + pre-projection,
// num_dlayers Gemma blocks whose attention reads the TARGET's donor KV cache
// (no K/V of its own), output norm, draft LM head, post-projection — as ONE GGML
// graph. Without this, the C# draft alternates device matmuls with host-side
// RoPE/attention, and the per-step device↔host ping-pong makes a 4-layer head
// cost as much as a full 48-layer decode. Single query per call, so attention
// is unmasked (every cached key is in the past). Gated by the C# caller to
// fixed_pos <= donor cache size (the SWA window has not wrapped).
// Outputs: logits[vocab] (draft LM head, no softcap) and h_out[backbone]
// (post-projection — the recurrent input chaining the next draft step).
// ============================================================================
TSG_EXPORT int TSGgml_Gemma4DraftStep(
    int token, const float* h_prev, int fixed_pos,
    int backbone, int draft_hidden, int num_dlayers, int num_heads, int vocab,
    float eps, int kv_cache_type,
    float* rope_freq_factors, int rope_freq_factors_len,
    // singleton weights
    void* tgt_tok_embd, int tte_type, std::int64_t tte_ne0, std::int64_t tte_ne1, std::int64_t tte_bytes,
    void* nextn_pre, int npre_type, std::int64_t npre_ne0, std::int64_t npre_ne1, std::int64_t npre_bytes,
    void* nextn_post, int npost_type, std::int64_t npost_ne0, std::int64_t npost_ne1, std::int64_t npost_bytes,
    void* draft_tok_embd, int dte_type, std::int64_t dte_ne0, std::int64_t dte_ne1, std::int64_t dte_bytes,
    void* output_norm_w,
    // per-layer (size num_dlayers)
    void** attn_norm_arr, void** wq_arr, int* wq_type, std::int64_t* wq_ne0, std::int64_t* wq_ne1, std::int64_t* wq_bytes,
    void** q_norm_arr, void** wo_arr, int* wo_type, std::int64_t* wo_ne0, std::int64_t* wo_ne1, std::int64_t* wo_bytes,
    void** post_attn_norm_arr, void** ffn_norm_arr,
    void** gate_arr, int* gate_type, std::int64_t* gate_ne0, std::int64_t* gate_ne1, std::int64_t* gate_bytes,
    void** up_arr, int* up_type, std::int64_t* up_ne0, std::int64_t* up_ne1, std::int64_t* up_bytes,
    void** down_arr, int* down_type, std::int64_t* down_ne0, std::int64_t* down_ne1, std::int64_t* down_bytes,
    void** post_ffw_norm_arr, float* out_scale_arr,
    int* hd_arr, int* kv_heads_arr, int* is_local_arr, float* rope_base_arr, int* rope_dims_arr,
    void** donor_k_arr, void** donor_v_arr, int* donor_cache_size_arr,
    // outputs
    float* logits_out, float* h_out)
{
    try
    {
        if (!ensure_backend())
            return 0;
        if (fixed_pos <= 0)
            return 0;
        // SWA donor caches are circular windows (handled below); a global donor's
        // linear cache must cover fixed_pos (the target's forward grows it).
        for (int l = 0; l < num_dlayers; l++)
            if (is_local_arr[l] == 0 && fixed_pos > donor_cache_size_arr[l])
                return 0;

        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for Gemma4 draft step.");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_tensor* tok_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* h_prev_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, backbone);
        ggml_tensor* pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* freq_t = nullptr;
        if (rope_freq_factors != nullptr && rope_freq_factors_len > 0)
            freq_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rope_freq_factors_len);

        ggml_tensor* tte_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(tte_type), tte_ne0, tte_ne1);
        ggml_tensor* npre_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(npre_type), npre_ne0, npre_ne1);
        ggml_tensor* npost_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(npost_type), npost_ne0, npost_ne1);
        ggml_tensor* dte_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(dte_type), dte_ne0, dte_ne1);
        ggml_tensor* onorm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, draft_hidden);

        struct DL {
            ggml_tensor* attn_norm; ggml_tensor* wq; ggml_tensor* q_norm; ggml_tensor* wo;
            ggml_tensor* post_attn_norm; ggml_tensor* ffn_norm; ggml_tensor* gate; ggml_tensor* up;
            ggml_tensor* down; ggml_tensor* post_ffw_norm; ggml_tensor* k_cache; ggml_tensor* v_cache;
            int hd; int kvHeads; int csize;
        };
        std::vector<DL> dl(num_dlayers);
        for (int l = 0; l < num_dlayers; l++)
        {
            auto& d = dl[l];
            d.hd = hd_arr[l]; d.kvHeads = kv_heads_arr[l]; d.csize = donor_cache_size_arr[l];
            d.attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, draft_hidden);
            d.wq = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(wq_type[l]), wq_ne0[l], wq_ne1[l]);
            d.q_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d.hd);
            d.wo = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(wo_type[l]), wo_ne0[l], wo_ne1[l]);
            d.post_attn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, draft_hidden);
            d.ffn_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, draft_hidden);
            d.gate = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gate_type[l]), gate_ne0[l], gate_ne1[l]);
            d.up = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(up_type[l]), up_ne0[l], up_ne1[l]);
            d.down = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type[l]), down_ne0[l], down_ne1[l]);
            d.post_ffw_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, draft_hidden);
            d.k_cache = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), d.hd, d.csize, d.kvHeads);
            d.v_cache = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), d.hd, d.csize, d.kvHeads);
        }

        // x = target.tok_embd[token] * sqrt(backbone) ; xh = concat(x, h_prev)
        ggml_tensor* x = ggml_get_rows(ctx, tte_w, tok_idx);            // [backbone]
        x = ggml_scale(ctx, x, sqrtf((float) backbone));
        ggml_tensor* x1 = ggml_reshape_1d(ctx, x, backbone);
        ggml_tensor* xh = ggml_concat(ctx, x1, h_prev_t, 0);           // [2*backbone]
        ggml_tensor* cur = ggml_mul_mat(ctx, npre_w, ggml_reshape_2d(ctx, xh, 2 * backbone, 1)); // [draft_hidden,1]
        cur = ggml_reshape_1d(ctx, cur, draft_hidden);

        for (int l = 0; l < num_dlayers; l++)
        {
            auto& d = dl[l];
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), d.attn_norm);
            ggml_tensor* q = ggml_mul_mat(ctx, d.wq, ggml_reshape_2d(ctx, normed, draft_hidden, 1)); // [num_heads*hd,1]
            ggml_tensor* q2 = ggml_reshape_2d(ctx, q, d.hd, num_heads);
            q2 = ggml_mul(ctx, ggml_rms_norm(ctx, q2, eps), d.q_norm);
            ggml_tensor* q3 = ggml_reshape_3d(ctx, q2, d.hd, num_heads, 1);
            ggml_tensor* rff = (is_local_arr[l] != 0) ? nullptr : freq_t;
            ggml_tensor* q_rope = ggml_rope_ext(ctx, q3, pos_t, rff, rope_dims_arr[l], 2, 0, rope_base_arr[l], 1.0f, 0, 1, 0, 0);

            // SWA donor: read the last min(fixed_pos, window) positions (circular,
            // unwrapped by view_kv_cache_window). Global donor: read [0, fixed_pos).
            const int dAttendLen = (is_local_arr[l] != 0) ? std::min(fixed_pos, d.csize) : fixed_pos;
            const int dActiveStart = (is_local_arr[l] != 0) ? ((fixed_pos - dAttendLen) % d.csize) : 0;
            ggml_tensor* k_full = view_kv_cache_window(ctx, d.k_cache, d.hd, d.csize, d.kvHeads, dActiveStart, dAttendLen, kv_cache_type);
            ggml_tensor* v_full = view_kv_cache_window(ctx, d.v_cache, d.hd, d.csize, d.kvHeads, dActiveStart, dAttendLen, kv_cache_type);
            if (k_full == nullptr || v_full == nullptr) { set_last_error("draft donor cache view failed"); return 0; }

            ggml_tensor* q_t = ggml_cont(ctx, ggml_permute(ctx, q_rope, 0, 2, 1, 3));  // [hd,1,num_heads]
            ggml_tensor* kq = ggml_mul_mat(ctx, k_full, q_t);                          // [dAttendLen,1,num_heads]
            kq = ggml_soft_max(ctx, kq);                                               // single query: all window keys valid
            ggml_tensor* v_t = ggml_cont(ctx, ggml_permute(ctx, v_full, 1, 0, 2, 3));   // [fixed_pos,hd,kvHeads]
            ggml_tensor* kqv = ggml_mul_mat(ctx, v_t, kq);                             // [hd,1,num_heads]
            ggml_tensor* attn = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));     // [hd,num_heads,1]
            ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn, d.hd * num_heads, 1);

            ggml_tensor* o = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, d.wo, attn_flat), draft_hidden);
            o = ggml_mul(ctx, ggml_rms_norm(ctx, o, eps), d.post_attn_norm);
            ggml_tensor* attn_out = ggml_add(ctx, cur, o);

            ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, attn_out, eps), d.ffn_norm);
            ggml_tensor* fn2 = ggml_reshape_2d(ctx, fn, draft_hidden, 1);
            ggml_tensor* gate = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, d.gate, fn2), gate_ne1[l]);
            ggml_tensor* up = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, d.up, fn2), up_ne1[l]);
            ggml_tensor* fh = ggml_mul(ctx, ggml_gelu(ctx, gate), up);
            ggml_tensor* down = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, d.down, ggml_reshape_2d(ctx, fh, gate_ne1[l], 1)), draft_hidden);
            down = ggml_mul(ctx, ggml_rms_norm(ctx, down, eps), d.post_ffw_norm);
            ggml_tensor* res = ggml_add(ctx, attn_out, down);

            float sc = out_scale_arr[l];
            if (std::fabs(sc - 1.0f) > 1e-6f)
                res = ggml_scale(ctx, res, sc);
            cur = res;
        }

        cur = ggml_mul(ctx, ggml_rms_norm(ctx, cur, eps), onorm_w);
        ggml_tensor* cur2 = ggml_reshape_2d(ctx, cur, draft_hidden, 1);
        ggml_tensor* logits = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, dte_w, cur2), vocab);   // [vocab]
        ggml_tensor* hnext = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, npost_w, cur2), backbone); // [backbone]

        ggml_tensor* logits_dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab);
        ggml_tensor* hnext_dst = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, backbone);
        ggml_tensor* logits_cpy = ggml_cpy(ctx, logits, logits_dst);
        ggml_tensor* hnext_cpy = ggml_cpy(ctx, hnext, hnext_dst);
        ggml_set_output(logits_cpy);
        ggml_set_output(hnext_cpy);

        const std::size_t graph_size = static_cast<std::size_t>(num_dlayers) * 160 + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(graph, logits_cpy);
        ggml_build_forward_expand(graph, hnext_cpy);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs_upload, usage))
                {
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, addr);
                    if (st == GGML_STATUS_SUCCESS) { if (needs_upload) upload_list.push_back({t, data, bytes}); return; }
                    invalidate_cached_buffer(data);
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable) ephemeral_bufs.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS) return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        bind_or_mark(tte_w, tgt_tok_embd, static_cast<std::size_t>(tte_bytes), true);
        bind_or_mark(npre_w, nextn_pre, static_cast<std::size_t>(npre_bytes), true);
        bind_or_mark(npost_w, nextn_post, static_cast<std::size_t>(npost_bytes), true);
        bind_or_mark(dte_w, draft_tok_embd, static_cast<std::size_t>(dte_bytes), true);
        bind_or_mark(onorm_w, output_norm_w, static_cast<std::size_t>(draft_hidden) * sizeof(float), true);
        for (int l = 0; l < num_dlayers; l++)
        {
            auto& d = dl[l];
            bind_or_mark(d.attn_norm, attn_norm_arr[l], static_cast<std::size_t>(draft_hidden) * sizeof(float), true);
            bind_or_mark(d.wq, wq_arr[l], static_cast<std::size_t>(wq_bytes[l]), true);
            bind_or_mark(d.q_norm, q_norm_arr[l], static_cast<std::size_t>(d.hd) * sizeof(float), true);
            bind_or_mark(d.wo, wo_arr[l], static_cast<std::size_t>(wo_bytes[l]), true);
            bind_or_mark(d.post_attn_norm, post_attn_norm_arr[l], static_cast<std::size_t>(draft_hidden) * sizeof(float), true);
            bind_or_mark(d.ffn_norm, ffn_norm_arr[l], static_cast<std::size_t>(draft_hidden) * sizeof(float), true);
            bind_or_mark(d.gate, gate_arr[l], static_cast<std::size_t>(gate_bytes[l]), true);
            bind_or_mark(d.up, up_arr[l], static_cast<std::size_t>(up_bytes[l]), true);
            bind_or_mark(d.down, down_arr[l], static_cast<std::size_t>(down_bytes[l]), true);
            bind_or_mark(d.post_ffw_norm, post_ffw_norm_arr[l], static_cast<std::size_t>(draft_hidden) * sizeof(float), true);
            bind_or_mark(d.k_cache, donor_k_arr[l], kv_cache_bytes(d.kvHeads, d.csize, d.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            bind_or_mark(d.v_cache, donor_v_arr[l], kv_cache_bytes(d.kvHeads, d.csize, d.hd, kv_cache_type), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }

        BufferHandle buffer(nullptr);
        if (!alloc_ctx_tensors_reuse(ctx))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr) { set_last_error("Failed to allocate backend buffer for Gemma4 draft step."); return 0; }
        }

        host_read_barrier();
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);

        std::int32_t tok = token;
        ggml_backend_tensor_set(tok_idx, &tok, 0, sizeof(std::int32_t));
        ggml_backend_tensor_set(h_prev_t, h_prev, 0, static_cast<std::size_t>(backbone) * sizeof(float));
        std::int32_t posv = fixed_pos;
        ggml_backend_tensor_set(pos_t, &posv, 0, sizeof(std::int32_t));
        if (freq_t != nullptr)
            ggml_backend_tensor_set(freq_t, rope_freq_factors, 0, static_cast<std::size_t>(rope_freq_factors_len) * sizeof(float));

        if (ggml_backend_graph_compute(g_backend, graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for Gemma4 draft step.");
            return 0;
        }

        host_read_barrier();
        ggml_backend_tensor_get(logits_dst, logits_out, 0, static_cast<std::size_t>(vocab) * sizeof(float));
        ggml_backend_tensor_get(hnext_dst, h_out, 0, static_cast<std::size_t>(backbone) * sizeof(float));

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
        set_last_error("Unknown error in Gemma4 draft step.");
        return 0;
    }
}

// ============================================================================
// Fused single-layer MoE decode (seqLen == 1): runs an ENTIRE Gemma 4 MoE
// transformer block as one GGML graph on the device, eliminating the ~18-20
// per-op C#→GGML dispatches the legacy TransformerBlock issues per MoE layer
// (each of which allocates+frees a Metal buffer and synchronises). Handles:
//   attn_norm → QKV (fused or separate/mixed-quant) → QK/V-norm → RoPE →
//   KV-cache write (circular for SWA) → flash_attn → O-proj →
//   post_attn_norm → residual1
//   ┌ dense shared FFN: ffn_norm → gate_up → gelu*up → down → post_ffw_norm_1
//   └ MoE: in-graph router (rms_norm·1/√H·gate_inp_scale → mul_mat → softmax →
//          top_k → gather+renorm → ×down_exps_scale) → mul_mat_id experts
//          (geglu) → weighted sum → post_ffw_norm_2 → add into dense output
//   result = residual1 + post_ffw_norm(mlp); ×layer_output_scale
// The in-graph router mirrors Gemma4Model.MoERoute + TryMoEForwardResidual
// exactly so the device path is numerically equivalent to the per-op path.
// ============================================================================

// Descriptor passed by pointer from C#. Layout MUST match
// Gemma4MoELayerDecodeDesc in GgmlNative.cs. 8-byte fields (pointers + int64)
// are grouped first, then 4-byte (int32 + float), so natural alignment is
// identical on both sides with no implicit padding surprises.

TSG_EXPORT int TSGgml_Gemma4MoELayerDecode(const TSGgmlGemma4MoELayerDesc* d)
{
    try
    {
        if (!ensure_backend())
            return 0;
        if (d == nullptr)
        {
            set_last_error("Gemma4 MoE layer decode: null descriptor.");
            return 0;
        }
        if (d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlGemma4MoELayerDesc)))
        {
            set_last_error("Gemma4 MoE layer decode: descriptor size mismatch (C#/native struct layout drift).");
            return 0;
        }

        const int H = d->hidden_size;
        const int position = d->position;
        const int totalSeqLen = position + 1;
        const int hd = d->head_dim;
        const int nH = d->num_heads;
        const int kvH = d->num_kv_heads;
        const int qDim = nH * hd;
        const int kDim = kvH * hd;
        const int cacheSize = d->cache_size;
        const bool isLocal = d->is_local != 0;
        const bool isShared = d->is_shared != 0;
        const bool separate_qkv = d->separate_qkv != 0;
        const int kvType = d->kv_cache_type;
        const float eps = d->eps;
        const int nExp = d->num_experts;
        const int nUsed = d->num_experts_used;
        const int attendLen = isLocal ? std::min(totalSeqLen, d->sliding_window) : totalSeqLen;
        const std::int64_t ffDense = d->gu_ne1 / 2;
        const std::int64_t ffMoe = d->gue_ne1 / 2;

        const std::size_t ctx_size = 16 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Gemma4 MoE layer decode: failed to acquire ggml context.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // --- input / weight tensors ---
        ggml_tensor* hidden_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* freq_factors_t = (d->freq_factors != nullptr && d->freq_factors_len > 0)
            ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d->freq_factors_len) : nullptr;

        ggml_tensor* attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->qkv_type), d->qkv_ne0, d->qkv_ne1);
        ggml_tensor* k_w = separate_qkv ? ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->k_type), d->k_ne0, d->k_ne1) : nullptr;
        ggml_tensor* v_w = separate_qkv ? ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->v_type), d->v_ne0, d->v_ne1) : nullptr;
        ggml_tensor* q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        ggml_tensor* k_norm_w = isShared ? nullptr : ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        ggml_tensor* o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->o_type), d->o_ne0, d->o_ne1);
        ggml_tensor* post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* k_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, cacheSize, kvH);
        ggml_tensor* v_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, cacheSize, kvH);

        ggml_tensor* ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->gu_type), d->gu_ne0, d->gu_ne1);
        ggml_tensor* down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->down_type), d->down_ne0, d->down_ne1);
        ggml_tensor* post_ffw_norm_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

        ggml_tensor* gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nExp);
        ggml_tensor* gate_inp_scale_t = (d->gate_inp_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
        ggml_tensor* pre_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* gate_up_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d->gue_type), d->gue_ne0, d->gue_ne1, nExp);
        ggml_tensor* down_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d->de_type), d->de_ne0, d->de_ne1, nExp);
        ggml_tensor* down_exps_scale_t = (d->down_exps_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nExp) : nullptr;
        ggml_tensor* post_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* post_ffw_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

        // ===================== Attention =====================
        ggml_tensor* hidden = hidden_t;
        ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), attn_norm_w);
        ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, H, 1);

        ggml_tensor* q_rope;
        ggml_tensor* k_full = nullptr;
        ggml_tensor* v_full = nullptr;
        ggml_tensor* k_cpy = nullptr;
        ggml_tensor* v_cpy = nullptr;
        ggml_tensor* attn_mask = nullptr;
        std::vector<ggml_fp16_t> attn_mask_data;

        const int rope_dims = d->rope_n_dims;
        ggml_tensor* rope_ff = isLocal ? nullptr : freq_factors_t;

        if (!isShared)
        {
            ggml_tensor* q_raw;
            ggml_tensor* k_raw;
            ggml_tensor* v_raw;
            if (separate_qkv)
            {
                q_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, qkv_w, normed_2d), qDim);
                k_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, k_w, normed_2d), kDim);
                v_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, v_w, normed_2d), kDim);
            }
            else
            {
                ggml_tensor* qkv_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, qkv_w, normed_2d), qDim + 2 * kDim);
                q_raw = ggml_view_1d(ctx, qkv_flat, qDim, 0);
                k_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim) * sizeof(float));
                v_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim + kDim) * sizeof(float));
            }

            ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_raw, hd, nH);
            ggml_tensor* k_2d = ggml_reshape_2d(ctx, k_raw, hd, kvH);
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), q_norm_w);
            ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d, eps), k_norm_w);
            ggml_tensor* v_2d = ggml_reshape_2d(ctx, v_raw, hd, kvH);
            ggml_tensor* v_normed = ggml_rms_norm(ctx, v_2d, eps);

            ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, hd, nH, 1);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, hd, kvH, 1);
            q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff, rope_dims, 2, 0, d->rope_base, 1.0f, 0, 1, 0, 0);
            ggml_tensor* k_rope_t = ggml_rope_ext(ctx, k_3d, pos_tensor, rope_ff, rope_dims, 2, 0, d->rope_base, 1.0f, 0, 1, 0, 0);

            ggml_tensor* k_rope_perm = ggml_permute(ctx, k_rope_t, 0, 2, 1, 3);
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_normed, hd, kvH, 1);
            ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
            ggml_tensor* k_write = ggml_cont(ctx, k_rope_perm);
            ggml_tensor* v_write = ggml_cont(ctx, v_perm);

            const int cachePos = isLocal ? (position % cacheSize) : position;
            const int activeStart = isLocal ? ((totalSeqLen - attendLen) % cacheSize) : 0;
            const int attnKvLen = flash_attn_kv_length(attendLen, cacheSize, hd);
            const std::size_t kv_byte_offset = static_cast<std::size_t>(cachePos) * k_cached_t->nb[1];
            ggml_tensor* k_dst = ggml_view_3d(ctx, k_cached_t, hd, 1, kvH, k_cached_t->nb[1], k_cached_t->nb[2], kv_byte_offset);
            ggml_tensor* v_dst = ggml_view_3d(ctx, v_cached_t, hd, 1, kvH, v_cached_t->nb[1], v_cached_t->nb[2], kv_byte_offset);
            k_cpy = ggml_cpy(ctx, k_write, k_dst);
            v_cpy = ggml_cpy(ctx, v_write, v_dst);
            if (flash_attn_requires_masked_padding(hd))
            {
                attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
                fill_flash_attn_mask(attn_mask_data, attnKvLen, attendLen);
            }
            k_full = view_kv_cache_window(ctx, k_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType);
            v_full = view_kv_cache_window(ctx, v_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType);
        }
        else
        {
            ggml_tensor* q_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, qkv_w, normed_2d), qDim);
            ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_flat, hd, nH);
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), q_norm_w);
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, hd, nH, 1);
            q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff, rope_dims, 2, 0, d->rope_base, 1.0f, 0, 1, 0, 0);

            const int activeStart = isLocal ? ((totalSeqLen - attendLen) % cacheSize) : 0;
            const int attnKvLen = flash_attn_kv_length(attendLen, cacheSize, hd);
            if (flash_attn_requires_masked_padding(hd))
            {
                attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
                fill_flash_attn_mask(attn_mask_data, attnKvLen, attendLen);
            }
            k_full = view_kv_cache_window(ctx, k_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType);
            v_full = view_kv_cache_window(ctx, v_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType);
        }

        if (k_full == nullptr || v_full == nullptr)
        {
            set_last_error("Gemma4 MoE layer decode: failed to build KV cache views.");
            return 0;
        }

        ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
        ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q_attn, k_full, v_full, attn_mask, 1.0f, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
        ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, qDim, 1);
        ggml_tensor* o_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, o_w, attn_flat), H);
        ggml_tensor* post_attn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, o_flat, eps), post_attn_norm_w);
        ggml_tensor* residual1 = ggml_add(ctx, hidden, post_attn_normed);

        // ===================== Dense shared FFN =====================
        ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), ffn_norm_w);
        ggml_tensor* ffn_normed_2d = ggml_reshape_2d(ctx, ffn_normed, H, 1);
        ggml_tensor* gu_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, gu_w, ffn_normed_2d), 2 * ffDense);
        ggml_tensor* dense_gate = ggml_view_1d(ctx, gu_flat, ffDense, 0);
        ggml_tensor* dense_up = ggml_view_1d(ctx, gu_flat, ffDense, static_cast<std::size_t>(ffDense) * sizeof(float));
        ggml_tensor* dense_h = ggml_mul(ctx, ggml_gelu(ctx, dense_gate), dense_up);
        ggml_tensor* dense_h_2d = ggml_reshape_2d(ctx, dense_h, ffDense, 1);
        ggml_tensor* dense_down = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, down_w, dense_h_2d), H);
        ggml_tensor* mlp = ggml_mul(ctx, ggml_rms_norm(ctx, dense_down, eps), post_ffw_norm_1_w);

        // ===================== MoE router (in-graph) =====================
        ggml_tensor* route_n = ggml_rms_norm(ctx, ggml_reshape_2d(ctx, residual1, H, 1), eps);
        route_n = ggml_scale(ctx, route_n, d->inv_sqrt_hidden);
        if (gate_inp_scale_t != nullptr)
            route_n = ggml_mul(ctx, route_n, gate_inp_scale_t);
        ggml_tensor* logits = ggml_mul_mat(ctx, gate_inp_w, route_n);          // [nExp, 1]
        ggml_tensor* probs = ggml_soft_max(ctx, logits);                       // softmax over nExp
        ggml_tensor* sel = ggml_top_k(ctx, probs, nUsed);                      // [nUsed, 1] i32
        ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, nExp, 1);
        ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);                     // [1, nUsed, 1]
        ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, nUsed, 1);
        ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);                         // [1, 1]
        w_2d = ggml_div(ctx, w_2d, w_sum);                                     // renormalised over selected
        if (down_exps_scale_t != nullptr)
        {
            ggml_tensor* scale_r = ggml_reshape_3d(ctx, down_exps_scale_t, 1, nExp, 1);
            ggml_tensor* sel_scale = ggml_get_rows(ctx, scale_r, sel);         // [1, nUsed, 1]
            w_2d = ggml_mul(ctx, w_2d, ggml_reshape_2d(ctx, sel_scale, nUsed, 1));
        }
        ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, nUsed, 1);

        // ===================== MoE experts =====================
        ggml_tensor* moe_in = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_2d(ctx, residual1, H, 1), eps), pre_ffw_norm_2_w);
        ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, moe_in, H, 1, 1);
        ggml_tensor* gate_up = ggml_mul_mat_id(ctx, gate_up_exps_t, moe_in_3d, sel);   // [2*ffMoe, nUsed, 1]
        ggml_tensor* moe_gate = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        ggml_tensor* moe_up = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], static_cast<std::size_t>(ffMoe) * gate_up->nb[0]);
        ggml_tensor* moe_act = ggml_geglu_split(ctx, moe_gate, moe_up);               // [ffMoe, nUsed, 1]
        ggml_tensor* moe_down = ggml_mul_mat_id(ctx, down_exps_t, moe_act, sel);       // [H, nUsed, 1]
        ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);                     // broadcast [H, nUsed, 1]

        // aggregate over the nUsed dim → [H, 1]
        ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, 1, weighted->nb[2], 0);
        for (int u = 1; u < nUsed; ++u)
        {
            ggml_tensor* view_u = ggml_view_2d(ctx, weighted, H, 1, weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
            moe_out = ggml_add(ctx, moe_out, view_u);
        }
        ggml_tensor* moe_out_1d = ggml_reshape_1d(ctx, moe_out, H);
        ggml_tensor* moe_normed = ggml_mul(ctx, ggml_rms_norm(ctx, moe_out_1d, eps), post_ffw_norm_2_w);
        mlp = ggml_add(ctx, mlp, moe_normed);

        // ===================== Final residual + layer scale =====================
        ggml_tensor* mlp_normed = ggml_mul(ctx, ggml_rms_norm(ctx, mlp, eps), post_ffw_norm_w);
        ggml_tensor* result = ggml_add(ctx, residual1, mlp_normed);
        if (std::fabs(d->layer_output_scale - 1.0f) > 1e-9f)
            result = ggml_scale(ctx, result, d->layer_output_scale);

        ggml_tensor* hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* out_cpy = ggml_cpy(ctx, result, hidden_out);
        ggml_set_output(out_cpy);

        // --- build graph (KV writes first to order them before reads) ---
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 2048, false);
        if (k_cpy != nullptr) ggml_build_forward_expand(graph, k_cpy);
        if (v_cpy != nullptr) ggml_build_forward_expand(graph, v_cpy);
        ggml_build_forward_expand(graph, out_cpy);

        // --- bind tensors ---
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
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload) upload_list.push_back({t, data, bytes});
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
                    if (!cacheable) ephemeral_bufs.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        bind_or_mark(qkv_w, d->qkv_w, static_cast<std::size_t>(d->qkv_bytes), true);
        if (separate_qkv)
        {
            bind_or_mark(k_w, d->k_w, static_cast<std::size_t>(d->k_bytes), true);
            bind_or_mark(v_w, d->v_w, static_cast<std::size_t>(d->v_bytes), true);
        }
        bind_or_mark(o_w, d->o_w, static_cast<std::size_t>(d->o_bytes), true);
        bind_or_mark(gu_w, d->gu_w, static_cast<std::size_t>(d->gu_bytes), true);
        bind_or_mark(down_w, d->down_w, static_cast<std::size_t>(d->down_bytes), true);
        bind_or_mark(gate_up_exps_t, d->gate_up_exps, static_cast<std::size_t>(d->gue_bytes), true);
        bind_or_mark(down_exps_t, d->down_exps, static_cast<std::size_t>(d->de_bytes), true);
        bind_or_mark(gate_inp_w, d->gate_inp_w, static_cast<std::size_t>(H) * nExp * sizeof(float), true);

        bind_or_mark(attn_norm_w, d->attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(post_attn_norm_w, d->post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(ffn_norm_w, d->ffn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(post_ffw_norm_1_w, d->post_ffw_norm_1_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(pre_ffw_norm_2_w, d->pre_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(post_ffw_norm_2_w, d->post_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(post_ffw_norm_w, d->post_ffw_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
        bind_or_mark(q_norm_w, d->q_norm_w, static_cast<std::size_t>(hd) * sizeof(float), true);
        if (!isShared)
            bind_or_mark(k_norm_w, d->k_norm_w, static_cast<std::size_t>(hd) * sizeof(float), true);
        if (gate_inp_scale_t != nullptr)
            bind_or_mark(gate_inp_scale_t, d->gate_inp_scale, static_cast<std::size_t>(H) * sizeof(float), true);
        if (down_exps_scale_t != nullptr)
            bind_or_mark(down_exps_scale_t, d->down_exps_scale, static_cast<std::size_t>(nExp) * sizeof(float), true);

        bind_or_mark(k_cached_t, d->k_cache, kv_cache_bytes(kvH, cacheSize, hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        bind_or_mark(v_cached_t, d->v_cache, kv_cache_bytes(kvH, cacheSize, hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        if (attn_mask != nullptr && !attn_mask_data.empty())
            bind_or_mark(attn_mask, attn_mask_data.data(), attn_mask_data.size() * sizeof(ggml_fp16_t), false);

        // Allocate intermediates (reuse persistent compute buffer across tokens).
        BufferHandle buffer(nullptr);
        if (!alloc_ctx_tensors_reuse(ctx))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("Gemma4 MoE layer decode: failed to allocate backend buffer.");
                return 0;
            }
        }

        host_read_barrier();

        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);

        ggml_backend_tensor_set(hidden_t, d->hidden, 0, static_cast<std::size_t>(H) * sizeof(float));
        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));
        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, d->freq_factors, 0, static_cast<std::size_t>(d->freq_factors_len) * sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Gemma4 MoE layer decode: graph execution failed.");
            return 0;
        }

        finalize_compute_with_download(hidden_out, d->hidden, static_cast<std::size_t>(H) * sizeof(float));
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
        set_last_error("Unknown error in Gemma4 MoE layer decode.");
        return 0;
    }
}

// ============================================================================
// Gemma4 MoE MODEL-WIDE decode: the whole transformer (all layers) as ONE
// GGML graph, dispatched/synchronised once per token instead of once per
// layer. This is the throughput fix for MoE Gemma 4 (e.g. gemma-4-26B-A4B):
// the per-layer TSGgml_Gemma4MoELayerDecode rebuilds a graph + encodes a Metal
// command buffer + synchronises ~30x per token, leaving the GPU idle in the
// inter-layer CPU gaps (~60% utilisation). Folding all layers into one graph
// amortises the build/encode/sync across the model, keeping the GPU saturated.
//
// Each layer's graph is byte-for-byte the same construction as the proven
// single-layer kernel above (attention + dense shared FFN + in-graph MoE
// router/experts), just chained: layer L's output residual feeds layer L+1.
// Adding every layer's KV-cache write to the graph before the final output (the
// same ordering the dense TSGgml_Gemma4ModelDecode relies on) guarantees each
// layer's cache write executes before that layer's attention reads it.
//
// Scope: non-shared (no KV-donor) layers, no PLE. The C# caller only routes the
// all-MoE / no-PLE / no-donor shape here and falls back to the per-layer path
// otherwise, so this kernel rejects (returns 0) anything it doesn't handle.
// The per-layer descriptor array reuses TSGgmlGemma4MoELayerDesc unchanged;
// `hidden`/`position` are taken from the shared params, the per-desc copies are
// ignored.
//
// PERSIST / CUDA-graph capture (g_g4moe): the exact analogue of the dense
// g_g4dc. Without it this kernel rebuilds the whole-model graph and re-binds
// every weight per token, and the position-dependent KV-cache view offset +
// attention window make the graph topology change token-to-token, so
// ggml-cuda's CUDA-graph capture never engages — on WDDM the GPU is starved in
// the per-node scheduling gaps (~16 tok/s, 92% host "Other"). With persist the
// graph is built ONCE with stable tensor addresses (raw ggml ctx +
// alloc_ctx_tensors); the KV write uses ggml_set_rows (write row = an I64
// INPUT) and attention reads a FIXED padded window [0, window) with an F16 mask
// INPUT, so the topology is identical token-to-token and capture engages. The
// window is padded up to a 256-token stride so it only changes (forcing a
// rebuild) every 256 tokens. Dropped + rebuilt when any layer window grows a
// stride, the model instance changes, or the KV-cache buffer is reallocated
// (TSGgml_Gemma4MoEResetDecodeCache, called by C# before any prefill / on KV
// reset). MoE Gemma 4 has no PLE and no KV-donor (shared) layers, so the
// persist bookkeeping is simpler than the dense path's.
// ============================================================================
namespace
{
    constexpr int kG4MoePersistKvStride = 256;

    struct G4MoEDecodeCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_in = nullptr;
        ggml_tensor* hidden_out = nullptr;
        ggml_tensor* pos_tensor = nullptr;
        std::vector<ggml_tensor*> kv_index;          // per-layer I64 set_rows write row
        std::vector<ggml_tensor*> attn_mask;         // per-layer F16 padding mask
        std::vector<int> layer_window;               // per-layer padded window length
        const void* sig_disc = nullptr;              // model-instance discriminator (attn_norm[0])
        const void* sig_kcache0 = nullptr;           // first KV buffer ptr (detects realloc/grow)
        int num_layers = 0;
        int hidden_size = 0;
        bool folded = false;                         // hidden_out holds logits (final norm + lm_head folded in)
        int out_count = 0;                           // floats to download (vocab when folded, else hidden)

        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_in = hidden_out = pos_tensor = nullptr;
            kv_index.clear(); attn_mask.clear(); layer_window.clear();
            sig_disc = sig_kcache0 = nullptr;
            num_layers = hidden_size = 0;
            folded = false; out_count = 0;
        }
    };

    // Per-request pool, identical in purpose to g_g4dc_pool (see its comment):
    // concurrent requests each decode through their own KV holder, so a single
    // shared entry would rebuild on every switch and collapse N>=2 throughput.
    // Keyed by (sig_disc, sig_kcache0) so each in-flight request keeps its own
    // persistent captured graph; ggml-cuda caches a cuda graph per nodes[0].
    constexpr int kG4MoeMaxDecodeCaches = 8;
    struct G4MoEDecodeCachePool
    {
        G4MoEDecodeCache entries[kG4MoeMaxDecodeCaches];
        std::uint64_t used[kG4MoeMaxDecodeCaches] = {};
        std::uint64_t clock = 0;

        G4MoEDecodeCache* find(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kG4MoeMaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                { used[i] = ++clock; return &entries[i]; }
            return nullptr;
        }

        G4MoEDecodeCache& claim(const void* sig, const void* kc0)
        {
            for (int i = 0; i < kG4MoeMaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0)
                { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            for (int i = 0; i < kG4MoeMaxDecodeCaches; i++)
                if (!entries[i].valid) { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            int lru = 0;
            for (int i = 1; i < kG4MoeMaxDecodeCaches; i++) if (used[i] < used[lru]) lru = i;
            entries[lru].reset(); used[lru] = ++clock; return entries[lru];
        }

        void reset_all() { for (auto& e : entries) e.reset(); }
    };
    G4MoEDecodeCachePool g_g4moe_pool;
}

TSG_EXPORT int TSGgml_Gemma4MoEModelDecode(
    const TSGgmlGemma4MoELayerDesc* layers, int num_layers,
    void* hidden_data, int hidden_size, int position,
    // Folded final-norm + lm_head (nullable). When logits_data / lm_head_data /
    // final_norm_data are non-null and vocab_size > 0, the graph appends the
    // output RMSNorm, the lm_head matmul and (when logit_softcap > 0) the tanh
    // logit softcap, writing logits[vocab] to logits_data so the whole token —
    // including the 256K-vocab projection — is one captured replay (no separate
    // per-token lm_head graph_compute disturbing the captured graph). The C#
    // caller then skips its own final-norm/lm_head/softcap dispatches.
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, float logit_softcap)
{
    try
    {
        if (!ensure_backend())
            return 0;
        if (layers == nullptr || num_layers <= 0 || hidden_data == nullptr)
        {
            set_last_error("Gemma4 MoE model decode: invalid arguments.");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlGemma4MoELayerDesc)))
        {
            set_last_error("Gemma4 MoE model decode: descriptor size mismatch (C#/native struct layout drift).");
            return 0;
        }
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].is_shared != 0)
            {
                set_last_error("Gemma4 MoE model decode: KV-donor (shared) layers unsupported; use per-layer path.");
                return 0;
            }
        }

        const int H = hidden_size;
        const int totalSeqLen = position + 1;
        const int num_heads = layers[0].num_heads;
        const float eps = layers[0].eps;
        const int kvType = layers[0].kv_cache_type;

        // Fold final-norm + lm_head into the graph when the caller supplies them.
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;
        const int out_count = fold ? vocab_size : H;

        // ===== Persist / CUDA-graph capture setup (mirrors dense g_g4dc) =====
        static const bool g4moe_timing = std::getenv("TS_GEMMA4_FD_TIMING") != nullptr;
        static const bool g4moe_persist = []{ const char* e = std::getenv("TS_GEMMA4_FD_PERSIST"); return e == nullptr || e[0] != '0'; }();

        std::vector<int> pwindow(num_layers, 0);          // padded window length per layer
        std::vector<int> pvalid(num_layers, 0);           // unmasked (valid) length per layer
        std::vector<std::int64_t> pwrite(num_layers, 0);  // set_rows write row per layer
        // CUDA-gate persist: ggml_set_rows segfaults on Metal (null-context buffer
        // in ggml_metal_op_set_rows). Metal falls through to the ggml_cpy path,
        // which still folds the LM head and keeps the KV cache device-resident.
        // See the dense TSGgml_Gemma4ModelDecode for the full rationale.
        bool can_persist = g4moe_persist && g_backend_type == BACKEND_TYPE_CUDA;
        {
            auto roundup_stride = [](int v){ return ((v + kG4MoePersistKvStride - 1) / kG4MoePersistKvStride) * kG4MoePersistKvStride; };
            for (int l = 0; l < num_layers; l++)
            {
                const int csz = layers[l].cache_size;
                if (csz <= 0) { can_persist = false; break; }
                const bool isLocal = layers[l].is_local != 0;
                if (isLocal)
                {
                    if (totalSeqLen <= csz) { pwindow[l] = std::min(csz, roundup_stride(totalSeqLen)); pvalid[l] = totalSeqLen; }
                    else { pwindow[l] = csz; pvalid[l] = csz; }   // saturated: read whole circular cache flat
                    pwrite[l] = position % csz;
                }
                else
                {
                    pwindow[l] = std::min(csz, roundup_stride(totalSeqLen)); pvalid[l] = totalSeqLen;
                    pwrite[l] = position;
                    // A global cache that already overflowed can't be expressed as a
                    // single padded window -> let the legacy per-token path handle it.
                    if (pvalid[l] > pwindow[l]) { can_persist = false; break; }
                }
            }
        }
        const void* g4moe_sig = layers[0].attn_norm_w;
        const void* g4moe_kc0 = layers[0].k_cache;

        // ---- reuse fast-path: replay THIS request's captured graph ----
        // Per-request lookup (model instance + KV holder); find() already matched
        // sig_disc + sig_kcache0, so only the finer shape fields are checked here.
        G4MoEDecodeCache* dc = can_persist ? g_g4moe_pool.find(g4moe_sig, g4moe_kc0) : nullptr;
        if (dc != nullptr && dc->graph != nullptr &&
            dc->num_layers == num_layers && dc->hidden_size == H &&
            dc->layer_window == pwindow &&
            dc->folded == fold && dc->out_count == out_count)
        {
            auto t_start = std::chrono::high_resolution_clock::now();
            host_read_barrier();
            // Async per-token input refresh (stream-ordered ahead of the captured
            // replay below); see the dense TSGgml_Gemma4ModelDecode reuse path for
            // the full rationale. Removes ~N redundant per-copy stream syncs/token.
            decode_input_set_async(dc->hidden_in, hidden_data, static_cast<std::size_t>(H) * sizeof(float));
            std::int32_t pos_val = position;
            decode_input_set_async(dc->pos_tensor, &pos_val, sizeof(std::int32_t));
            for (int l = 0; l < num_layers; l++)
            {
                if (dc->kv_index[l] != nullptr)
                    decode_input_set_async(dc->kv_index[l], &pwrite[l], sizeof(std::int64_t));
                if (dc->attn_mask[l] != nullptr)
                {
                    std::vector<ggml_fp16_t> md;
                    fill_flash_attn_mask(md, pwindow[l], pvalid[l]);
                    decode_input_set_async(dc->attn_mask[l], md.data(), md.size() * sizeof(ggml_fp16_t));
                }
            }
            auto t_setup = std::chrono::high_resolution_clock::now();
            ggml_status st = ggml_backend_graph_compute(g_backend, dc->graph);
            if (st != GGML_STATUS_SUCCESS)
            {
                set_last_error("Gemma4 MoE model decode: cached graph execution failed.");
                dc->reset();
                return 0;
            }
            auto t_compute = std::chrono::high_resolution_clock::now();
            void* reuse_out = dc->folded ? logits_data : hidden_data;
            finalize_compute_with_download(dc->hidden_out, reuse_out, static_cast<std::size_t>(dc->out_count) * sizeof(float));
            host_read_barrier();
            if (g4moe_timing)
            {
                auto t_end = std::chrono::high_resolution_clock::now();
                auto ms = [](auto a, auto b){ return std::chrono::duration<double, std::milli>(b - a).count(); };
                fprintf(stderr, "[g4moe-fd] REUSE setup=%.2f compute=%.2f download=%.2f total=%.2f ms\n",
                    ms(t_start, t_setup), ms(t_setup, t_compute), ms(t_compute, t_end), ms(t_start, t_end));
                fflush(stderr);
            }
            return 1;
        }
        // Miss -> claim this request's slot to (re)build into (when persistable).
        G4MoEDecodeCache* g4moe = can_persist ? &g_g4moe_pool.claim(g4moe_sig, g4moe_kc0) : nullptr;

        // ctx holds only tensor metadata (no_alloc: data is bound externally). Non-
        // persist uses a pooled 32 MB block; persist uses a raw ctx kept alive in
        // g_g4moe for graph reuse + CUDA-graph capture (stable tensor addresses).
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr)
            {
                set_last_error("Gemma4 MoE model decode: failed to init persist ggml context.");
                return 0;
            }
        }
        else
        {
            if (!context.init(ctx_size))
            {
                set_last_error("Gemma4 MoE model decode: failed to acquire ggml context.");
                return 0;
            }
            ctx = context.value;
        }

        // Per-layer persist inputs (created in the build loop; null in legacy mode).
        std::vector<ggml_tensor*> layer_kv_index(num_layers, nullptr);

        ggml_tensor* hidden_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        if (can_persist)
        {
            ggml_set_input(hidden_t);
            ggml_set_input(pos_tensor);
        }

        // Single shared rope_freqs tensor (same weight across all global layers).
        ggml_tensor* freq_factors_t = nullptr;
        void* freq_data = nullptr;
        int freq_len = 0;
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].freq_factors != nullptr && layers[l].freq_factors_len > 0)
            {
                freq_data = layers[l].freq_factors;
                freq_len = layers[l].freq_factors_len;
                break;
            }
        }
        if (freq_data != nullptr)
            freq_factors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, freq_len);

        struct MoeLayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* k_w;
            ggml_tensor* v_w;
            ggml_tensor* q_norm_w;
            ggml_tensor* k_norm_w;
            ggml_tensor* o_w;
            ggml_tensor* post_attn_norm_w;
            ggml_tensor* k_cached_t;
            ggml_tensor* v_cached_t;
            ggml_tensor* ffn_norm_w;
            ggml_tensor* gu_w;
            ggml_tensor* down_w;
            ggml_tensor* post_ffw_norm_1_w;
            ggml_tensor* gate_inp_w;
            ggml_tensor* gate_inp_scale_t;
            ggml_tensor* pre_ffw_norm_2_w;
            ggml_tensor* gate_up_exps_t;
            ggml_tensor* down_exps_t;
            ggml_tensor* down_exps_scale_t;
            ggml_tensor* post_ffw_norm_2_w;
            ggml_tensor* post_ffw_norm_w;
            ggml_tensor* k_cpy;
            ggml_tensor* v_cpy;
            ggml_tensor* attn_mask;
            std::vector<ggml_fp16_t> attn_mask_data;
        };
        std::vector<MoeLayerTensors> lt(num_layers);

        // --- create per-layer weight / cache tensors ---
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGemma4MoELayerDesc& d = layers[l];
            MoeLayerTensors& t = lt[l];
            const int hd = d.head_dim;
            const int kvH = d.num_kv_heads;
            const int cacheSize = d.cache_size;
            const int nExp = d.num_experts;
            const bool separate_qkv = d.separate_qkv != 0;

            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
            if (separate_qkv)
            {
                t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
                t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
            }
            else { t.k_w = nullptr; t.v_w = nullptr; }
            t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.k_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, cacheSize, kvH);
            t.v_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, cacheSize, kvH);
            t.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
            t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            t.post_ffw_norm_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nExp);
            t.gate_inp_scale_t = (d.gate_inp_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
            t.pre_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_up_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gue_type), d.gue_ne0, d.gue_ne1, nExp);
            t.down_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.de_type), d.de_ne0, d.de_ne1, nExp);
            t.down_exps_scale_t = (d.down_exps_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nExp) : nullptr;
            t.post_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.post_ffw_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.k_cpy = nullptr;
            t.v_cpy = nullptr;
            t.attn_mask = nullptr;
        }

        // --- build the chained graph ---
        ggml_tensor* hidden = hidden_t;
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGemma4MoELayerDesc& d = layers[l];
            MoeLayerTensors& t = lt[l];

            const int hd = d.head_dim;
            const int nH = num_heads;
            const int kvH = d.num_kv_heads;
            const int qDim = nH * hd;
            const int kDim = kvH * hd;
            const int cacheSize = d.cache_size;
            const bool isLocal = d.is_local != 0;
            const bool separate_qkv = d.separate_qkv != 0;
            const int nExp = d.num_experts;
            const int nUsed = d.num_experts_used;
            const int attendLen = isLocal ? std::min(totalSeqLen, d.sliding_window) : totalSeqLen;
            const std::int64_t ffDense = d.gu_ne1 / 2;
            const std::int64_t ffMoe = d.gue_ne1 / 2;
            const int rope_dims = d.rope_n_dims;
            ggml_tensor* rope_ff = isLocal ? nullptr : freq_factors_t;

            // ===== Attention =====
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w);
            ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, H, 1);

            ggml_tensor* q_raw;
            ggml_tensor* k_raw;
            ggml_tensor* v_raw;
            if (separate_qkv)
            {
                q_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.qkv_w, normed_2d), qDim);
                k_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.k_w, normed_2d), kDim);
                v_raw = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.v_w, normed_2d), kDim);
            }
            else
            {
                ggml_tensor* qkv_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.qkv_w, normed_2d), qDim + 2 * kDim);
                q_raw = ggml_view_1d(ctx, qkv_flat, qDim, 0);
                k_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim) * sizeof(float));
                v_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim + kDim) * sizeof(float));
            }

            ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_raw, hd, nH);
            ggml_tensor* k_2d = ggml_reshape_2d(ctx, k_raw, hd, kvH);
            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), t.q_norm_w);
            ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d, eps), t.k_norm_w);
            ggml_tensor* v_2d = ggml_reshape_2d(ctx, v_raw, hd, kvH);
            ggml_tensor* v_normed = ggml_rms_norm(ctx, v_2d, eps);

            ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, hd, nH, 1);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, hd, kvH, 1);
            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff, rope_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);
            ggml_tensor* k_rope_t = ggml_rope_ext(ctx, k_3d, pos_tensor, rope_ff, rope_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);

            ggml_tensor* k_rope_perm = ggml_permute(ctx, k_rope_t, 0, 2, 1, 3);
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_normed, hd, kvH, 1);
            ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
            ggml_tensor* k_write = ggml_cont(ctx, k_rope_perm);
            ggml_tensor* v_write = ggml_cont(ctx, v_perm);

            ggml_tensor* k_full;
            ggml_tensor* v_full;
            if (can_persist)
            {
                // KV write via set_rows: the write row is an I64 INPUT, so the graph
                // topology is identical token-to-token (capturable). Attention reads
                // a FIXED padded window [0, win) with an F16 mask INPUT zeroing valid
                // positions and -inf'ing the padding.
                const int win = pwindow[l];
                ggml_tensor* kv_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, 1);
                ggml_set_input(kv_idx);
                layer_kv_index[l] = kv_idx;
                t.k_cpy = ggml_set_rows(ctx, t.k_cached_t, k_write, kv_idx);
                t.v_cpy = ggml_set_rows(ctx, t.v_cached_t, v_write, kv_idx);
                t.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, win, 1, 1, 1);
                ggml_set_input(t.attn_mask);
                k_full = view_kv_cache_window(ctx, t.k_cached_t, hd, cacheSize, kvH, 0, win, kvType);
                v_full = view_kv_cache_window(ctx, t.v_cached_t, hd, cacheSize, kvH, 0, win, kvType);
            }
            else
            {
                const int cachePos = isLocal ? (position % cacheSize) : position;
                const int activeStart = isLocal ? ((totalSeqLen - attendLen) % cacheSize) : 0;
                const int attnKvLen = flash_attn_kv_length(attendLen, cacheSize, hd);
                const std::size_t kv_byte_offset = static_cast<std::size_t>(cachePos) * t.k_cached_t->nb[1];
                ggml_tensor* k_dst = ggml_view_3d(ctx, t.k_cached_t, hd, 1, kvH, t.k_cached_t->nb[1], t.k_cached_t->nb[2], kv_byte_offset);
                ggml_tensor* v_dst = ggml_view_3d(ctx, t.v_cached_t, hd, 1, kvH, t.v_cached_t->nb[1], t.v_cached_t->nb[2], kv_byte_offset);
                t.k_cpy = ggml_cpy(ctx, k_write, k_dst);
                t.v_cpy = ggml_cpy(ctx, v_write, v_dst);
                if (flash_attn_requires_masked_padding(hd))
                {
                    t.attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, attnKvLen, 1, 1, 1);
                    fill_flash_attn_mask(t.attn_mask_data, attnKvLen, attendLen);
                }
                k_full = view_kv_cache_window(ctx, t.k_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType);
                v_full = view_kv_cache_window(ctx, t.v_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType);
            }
            if (k_full == nullptr || v_full == nullptr)
            {
                set_last_error("Gemma4 MoE model decode: failed to build KV cache views.");
                if (can_persist) ggml_free(ctx);
                return 0;
            }

            ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
            ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q_attn, k_full, v_full, t.attn_mask, 1.0f, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
            ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, qDim, 1);
            ggml_tensor* o_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.o_w, attn_flat), H);
            ggml_tensor* post_attn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, o_flat, eps), t.post_attn_norm_w);
            ggml_tensor* residual1 = ggml_add(ctx, hidden, post_attn_normed);

            // ===== Dense shared FFN =====
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.ffn_norm_w);
            ggml_tensor* ffn_normed_2d = ggml_reshape_2d(ctx, ffn_normed, H, 1);
            ggml_tensor* gu_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.gu_w, ffn_normed_2d), 2 * ffDense);
            ggml_tensor* dense_gate = ggml_view_1d(ctx, gu_flat, ffDense, 0);
            ggml_tensor* dense_up = ggml_view_1d(ctx, gu_flat, ffDense, static_cast<std::size_t>(ffDense) * sizeof(float));
            ggml_tensor* dense_h = ggml_mul(ctx, ggml_gelu(ctx, dense_gate), dense_up);
            ggml_tensor* dense_h_2d = ggml_reshape_2d(ctx, dense_h, ffDense, 1);
            ggml_tensor* dense_down = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, t.down_w, dense_h_2d), H);
            ggml_tensor* mlp = ggml_mul(ctx, ggml_rms_norm(ctx, dense_down, eps), t.post_ffw_norm_1_w);

            // ===== MoE router (in-graph) =====
            ggml_tensor* route_n = ggml_rms_norm(ctx, ggml_reshape_2d(ctx, residual1, H, 1), eps);
            route_n = ggml_scale(ctx, route_n, d.inv_sqrt_hidden);
            if (t.gate_inp_scale_t != nullptr)
                route_n = ggml_mul(ctx, route_n, t.gate_inp_scale_t);
            ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, route_n);
            ggml_tensor* probs = ggml_soft_max(ctx, router_logits);
            ggml_tensor* sel = ggml_top_k(ctx, probs, nUsed);
            ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, nExp, 1);
            ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);
            ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, nUsed, 1);
            ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);
            w_2d = ggml_div(ctx, w_2d, w_sum);
            if (t.down_exps_scale_t != nullptr)
            {
                ggml_tensor* scale_r = ggml_reshape_3d(ctx, t.down_exps_scale_t, 1, nExp, 1);
                ggml_tensor* sel_scale = ggml_get_rows(ctx, scale_r, sel);
                w_2d = ggml_mul(ctx, w_2d, ggml_reshape_2d(ctx, sel_scale, nUsed, 1));
            }
            ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, nUsed, 1);

            // ===== MoE experts =====
            ggml_tensor* moe_in = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_2d(ctx, residual1, H, 1), eps), t.pre_ffw_norm_2_w);
            ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, moe_in, H, 1, 1);
            ggml_tensor* gate_up = ggml_mul_mat_id(ctx, t.gate_up_exps_t, moe_in_3d, sel);
            ggml_tensor* moe_gate = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
            ggml_tensor* moe_up = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], static_cast<std::size_t>(ffMoe) * gate_up->nb[0]);
            ggml_tensor* moe_act = ggml_geglu_split(ctx, moe_gate, moe_up);
            ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps_t, moe_act, sel);
            ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);

            ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, 1, weighted->nb[2], 0);
            for (int u = 1; u < nUsed; ++u)
            {
                ggml_tensor* view_u = ggml_view_2d(ctx, weighted, H, 1, weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                moe_out = ggml_add(ctx, moe_out, view_u);
            }
            ggml_tensor* moe_out_1d = ggml_reshape_1d(ctx, moe_out, H);
            ggml_tensor* moe_normed = ggml_mul(ctx, ggml_rms_norm(ctx, moe_out_1d, eps), t.post_ffw_norm_2_w);
            mlp = ggml_add(ctx, mlp, moe_normed);

            // ===== Final residual + layer scale =====
            ggml_tensor* mlp_normed = ggml_mul(ctx, ggml_rms_norm(ctx, mlp, eps), t.post_ffw_norm_w);
            ggml_tensor* result = ggml_add(ctx, residual1, mlp_normed);
            if (std::fabs(d.layer_output_scale - 1.0f) > 1e-9f)
                result = ggml_scale(ctx, result, d.layer_output_scale);

            hidden = result;
        }

        // Output: either the bare hidden state, or — when folding — the final
        // RMSNorm * output_norm, the lm_head projection and the tanh logit softcap,
        // so the 256K-vocab logits are part of the captured replay.
        ggml_tensor* lm_head_t = nullptr;
        ggml_tensor* final_norm_t = nullptr;
        ggml_tensor* hidden_out;
        ggml_tensor* out_cpy;
        if (fold)
        {
            lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
            final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);
            ggml_tensor* fn_2d = ggml_reshape_2d(ctx, fn, H, 1);
            ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, fn_2d);   // [vocab, 1]
            if (logit_softcap > 0.0f)
            {
                logits = ggml_scale(ctx, logits, 1.0f / logit_softcap);
                logits = ggml_tanh(ctx, logits);
                logits = ggml_scale(ctx, logits, logit_softcap);
            }
            logits = ggml_reshape_1d(ctx, logits, vocab_size);
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, vocab_size);
            out_cpy = ggml_cpy(ctx, logits, hidden_out);
        }
        else
        {
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            out_cpy = ggml_cpy(ctx, hidden, hidden_out);
        }
        ggml_set_output(out_cpy);

        // KV writes first so they are ordered before the reads (mirrors the dense
        // model-wide decode).
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * 160 + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
        {
            if (lt[l].k_cpy != nullptr)
            {
                ggml_build_forward_expand(graph, lt[l].k_cpy);
                ggml_build_forward_expand(graph, lt[l].v_cpy);
            }
        }
        ggml_build_forward_expand(graph, out_cpy);

        // --- bind tensors ---
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* tgt, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (tgt == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, tgt, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, tgt, addr) == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload) upload_list.push_back({tgt, data, bytes});
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
                    if (!cacheable) ephemeral_bufs.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGemma4MoELayerDesc& d = layers[l];
            MoeLayerTensors& t = lt[l];
            const int hd = d.head_dim;
            const int kvH = d.num_kv_heads;
            const int nExp = d.num_experts;
            const int cacheSize = d.cache_size;

            bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
            if (t.k_w != nullptr)
            {
                bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true);
                bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true);
            }
            bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
            bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes), true);
            bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes), true);
            bind_or_mark(t.gate_up_exps_t, d.gate_up_exps, static_cast<std::size_t>(d.gue_bytes), true);
            bind_or_mark(t.down_exps_t, d.down_exps, static_cast<std::size_t>(d.de_bytes), true);
            bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(H) * nExp * sizeof(float), true);
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.ffn_norm_w, d.ffn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_1_w, d.post_ffw_norm_1_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.pre_ffw_norm_2_w, d.pre_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_2_w, d.post_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_w, d.post_ffw_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(hd) * sizeof(float), true);
            bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(hd) * sizeof(float), true);
            if (t.gate_inp_scale_t != nullptr)
                bind_or_mark(t.gate_inp_scale_t, d.gate_inp_scale, static_cast<std::size_t>(H) * sizeof(float), true);
            if (t.down_exps_scale_t != nullptr)
                bind_or_mark(t.down_exps_scale_t, d.down_exps_scale, static_cast<std::size_t>(nExp) * sizeof(float), true);
            bind_or_mark(t.k_cached_t, d.k_cache, kv_cache_bytes(kvH, cacheSize, hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            bind_or_mark(t.v_cached_t, d.v_cache, kv_cache_bytes(kvH, cacheSize, hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            if (t.attn_mask != nullptr && !t.attn_mask_data.empty())
                bind_or_mark(t.attn_mask, t.attn_mask_data.data(), t.attn_mask_data.size() * sizeof(ggml_fp16_t), false);
        }
        if (freq_factors_t != nullptr)
            bind_or_mark(freq_factors_t, freq_data, static_cast<std::size_t>(freq_len) * sizeof(float), true);
        if (fold)
        {
            bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
            bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), static_cast<std::size_t>(H) * sizeof(float), true);
        }

        // Non-persist (legacy / TS_GEMMA4_FD_PERSIST=0) keeps the peak-packed gallocr
        // (shared with the MoE verify): the bump allocator's footprint is the SUM of
        // every intermediate (~870 MB on the 26B-A4B), which on top of the ~16 GB
        // resident weights/KV would OOM; gallocr packs by tensor LIFETIME (peak).
        // Persist: stable tensor addresses (every intermediate its own slot) so the
        // built graph + KV buffers keep fixed addresses for CUDA-graph capture; the
        // ctx/graph/buffer are kept alive in g_g4moe. The N=1 padded-window decode's
        // intermediate footprint is small (a few MB), unlike the verify's. Non-
        // persist keeps the peak-packed gallocr (the original behaviour).
        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("Gemma4 MoE model decode: failed to allocate persist backend buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            {
                set_last_error("Gemma4 MoE model decode: failed to allocate backend buffer.");
                return 0;
            }
        }

        host_read_barrier();

        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);

        ggml_backend_tensor_set(hidden_t, hidden_data, 0, static_cast<std::size_t>(H) * sizeof(float));
        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));
        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, freq_data, 0, static_cast<std::size_t>(freq_len) * sizeof(float));
        if (can_persist)
        {
            // Per-token inputs for the first build (the reuse path updates these).
            for (int l = 0; l < num_layers; l++)
            {
                if (layer_kv_index[l] != nullptr)
                    ggml_backend_tensor_set(layer_kv_index[l], &pwrite[l], 0, sizeof(std::int64_t));
                if (lt[l].attn_mask != nullptr)
                {
                    std::vector<ggml_fp16_t> md;
                    fill_flash_attn_mask(md, pwindow[l], pvalid[l]);
                    ggml_backend_tensor_set(lt[l].attn_mask, md.data(), 0, md.size() * sizeof(ggml_fp16_t));
                }
            }
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Gemma4 MoE model decode: graph execution failed.");
            if (can_persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
            return 0;
        }

        void* out_data = fold ? logits_data : hidden_data;
        finalize_compute_with_download(hidden_out, out_data, static_cast<std::size_t>(out_count) * sizeof(float));
        // If we used the per-call fallback buffer (not the persistent gallocr),
        // drain the queued async download before BufferHandle frees it. No-op on
        // the common gallocr path (buffer.value == nullptr).
        if (buffer.value != nullptr || can_persist) host_read_barrier();

        if (can_persist && g4moe != nullptr)
        {
            // Keep the ctx/graph/buffer + input handles alive for capture+replay,
            // in this request's own pool slot.
            g4moe->ctx = ctx;
            g4moe->buffer = persist_buf;
            g4moe->graph = graph;
            g4moe->hidden_in = hidden_t;
            g4moe->hidden_out = hidden_out;
            g4moe->pos_tensor = pos_tensor;
            g4moe->kv_index = layer_kv_index;
            g4moe->attn_mask.resize(num_layers);
            for (int l = 0; l < num_layers; l++) g4moe->attn_mask[l] = lt[l].attn_mask;
            g4moe->layer_window = pwindow;
            g4moe->sig_disc = g4moe_sig;
            g4moe->sig_kcache0 = g4moe_kc0;
            g4moe->num_layers = num_layers;
            g4moe->hidden_size = H;
            g4moe->folded = fold;
            g4moe->out_count = out_count;
            g4moe->valid = true;
        }
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
        set_last_error("Unknown error in Gemma4 MoE model decode.");
        return 0;
    }
}

// Drop the persistent (CUDA-graph-captured) Gemma4 MoE decode graph. The captured
// graph pins ggml-cuda's compute-pool scratch addresses and the KV-cache device
// buffers; a prefill (which grows the pool) or a KV reset/grow can move those, so
// the C# caller drops the cache before any prefill and on ResetKVCache. The next
// decode rebuilds + re-captures against the current pool state. No-op when persist
// mode is off (the cache is never populated).
TSG_EXPORT void TSGgml_Gemma4MoEResetDecodeCache()
{
    g_g4moe_pool.reset_all();
}

// ============================================================================
// Gemma4 MoE MODEL-WIDE multi-token VERIFY: the whole MoE transformer over N
// tokens as ONE GGML graph. This is the MoE sibling of the dense
// TSGgml_Gemma4ModelVerify and the multi-token sibling of
// TSGgml_Gemma4MoEModelDecode — it is what makes MTP speculative decoding pay
// off on MoE Gemma 4 (e.g. gemma-4-26B-A4B): a K+1 verify batch runs as a single
// dispatch/sync instead of (K+1) fused single-token decodes or, far worse, the
// per-op TransformerBlock fallback (~390 ms/step that made spec net-negative).
//
// Attention is built exactly like the dense verify (manual masked attention:
// mul_mat + ggml_diag_mask_inf(attendLen-N) + soft_max, robust at head_dim 512
// and for SWA windows; wrap-aware circular KV write + windowed read), and the
// FFN is built exactly like the MoE decode (dense shared FFN + in-graph router +
// stacked-expert ggml_mul_mat_id), generalised from 1 to N tokens. Output is the
// per-row layer-stack hidden state [hidden_size, N] (pre output_norm); the C#
// caller owns output_norm + the LM head. Reuses TSGgmlGemma4MoELayerDesc unchanged
// (hidden/position per-desc are ignored; start_pos + num_tokens are shared params).
//
// Scope (enforced by the C# gate in Gemma4Model.NativeGemma4MoEModelVerify):
// all-MoE, non-shared (no KV donor), no PLE, F32/F16 KV cache, and (for global
// layers) start_pos + N <= cache_size. Returns 0 on anything it cannot handle so
// the caller falls back to the per-op verify.
// ============================================================================
TSG_EXPORT int TSGgml_Gemma4MoEModelVerify(
    const TSGgmlGemma4MoELayerDesc* layers, int num_layers,
    void* hidden_data, int hidden_size, int start_pos, int num_tokens)
{
    try
    {
        if (!ensure_backend())
            return 0;
        if (layers == nullptr || num_layers <= 0 || hidden_data == nullptr)
        {
            set_last_error("Gemma4 MoE model verify: invalid arguments.");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlGemma4MoELayerDesc)))
        {
            set_last_error("Gemma4 MoE model verify: descriptor size mismatch (C#/native struct layout drift).");
            return 0;
        }

        const int N = num_tokens;
        if (N <= 1)
            return 0;
        const int H = hidden_size;
        const int totalSeqLen = start_pos + N;
        const int num_heads = layers[0].num_heads;
        const float eps = layers[0].eps;
        const int kvType = layers[0].kv_cache_type;

        // The whole layer (attention + dense FFN + experts) is processed in query
        // tiles of moe_attn_tile (see the per-layer loop / get_tile_mask), so the
        // [*, N] FFN/router/expert intermediates are tile-bounded rather than scaling
        // with N — this is what mirrors llama.cpp's n_ubatch and keeps the gallocr
        // peak (~the residual stream + K/V) under the ~1 GB headroom the 26B-A4B
        // leaves on a 16 GB card. (The old per-FFN TS_G4_MOE_FFN_TILE loop is
        // subsumed by the whole-layer tiling; the knob no longer applies.)

        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].is_shared != 0)
            {
                set_last_error("Gemma4 MoE model verify: KV-donor (shared) layers unsupported; use per-op path.");
                return 0;
            }
            // Global (full-attention) layers use a linear cache that must cover the
            // whole sequence (totalSeqLen <= cache_size), so global overflow always
            // bails. SWA (local) layers handle overflow (totalSeqLen > window) at any
            // start_pos: at start_pos==0 the fresh chunk is the whole history
            // (swaFresh); at start_pos>0 the kernel gathers the previous window from
            // the rolling cache and prepends it to the fresh chunk (swaPrev) — this is
            // what lets a long prompt be processed as bounded ubatches (the C# wrapper
            // sub-chunks; each native call keeps the gallocr peak small). (totalSeqLen
            // <= window is always exact: causal == windowed.)
            const bool isLocal = layers[l].is_local != 0;
            if (totalSeqLen > layers[l].cache_size && !isLocal)
            {
                set_last_error("Gemma4 MoE model verify: global cache too small for sequence; use per-op path.");
                return 0;
            }
        }

        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Gemma4 MoE model verify: failed to acquire ggml context.");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_tensor* current = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);

        ggml_tensor* freq_factors_t = nullptr;
        void* freq_data = nullptr;
        int freq_len = 0;
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].freq_factors != nullptr && layers[l].freq_factors_len > 0)
            {
                freq_data = layers[l].freq_factors;
                freq_len = layers[l].freq_factors_len;
                break;
            }
        }
        if (freq_data != nullptr)
            freq_factors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, freq_len);

        struct MoeLayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* k_w;
            ggml_tensor* v_w;
            ggml_tensor* q_norm_w;
            ggml_tensor* k_norm_w;
            ggml_tensor* o_w;
            ggml_tensor* post_attn_norm_w;
            ggml_tensor* k_cached_t;
            ggml_tensor* v_cached_t;
            ggml_tensor* ffn_norm_w;
            ggml_tensor* gu_w;
            ggml_tensor* down_w;
            ggml_tensor* post_ffw_norm_1_w;
            ggml_tensor* gate_inp_w;
            ggml_tensor* gate_inp_scale_t;
            ggml_tensor* pre_ffw_norm_2_w;
            ggml_tensor* gate_up_exps_t;
            ggml_tensor* down_exps_t;
            ggml_tensor* down_exps_scale_t;
            ggml_tensor* post_ffw_norm_2_w;
            ggml_tensor* post_ffw_norm_w;
            ggml_tensor* k_cpy; ggml_tensor* v_cpy;     // primary cache write
            ggml_tensor* k_cpy2; ggml_tensor* v_cpy2;   // wrapped tail (circular SWA)
            ggml_tensor* k_prev; ggml_tensor* v_prev;   // swaPrev: F32 copy of the prev window (read before the cache write)
        };
        std::vector<MoeLayerTensors> lt(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGemma4MoELayerDesc& d = layers[l];
            MoeLayerTensors& t = lt[l];
            const int hd = d.head_dim;
            const int kvH = d.num_kv_heads;
            const int cacheSize = d.cache_size;
            const int nExp = d.num_experts;
            const bool separate_qkv = d.separate_qkv != 0;

            t.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.qkv_type), d.qkv_ne0, d.qkv_ne1);
            if (separate_qkv)
            {
                t.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.k_type), d.k_ne0, d.k_ne1);
                t.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.v_type), d.v_ne0, d.v_ne1);
            }
            else { t.k_w = nullptr; t.v_w = nullptr; }
            t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
            t.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.k_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, cacheSize, kvH);
            t.v_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, cacheSize, kvH);
            t.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
            t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
            t.post_ffw_norm_1_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_inp_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, nExp);
            t.gate_inp_scale_t = (d.gate_inp_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H) : nullptr;
            t.pre_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.gate_up_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.gue_type), d.gue_ne0, d.gue_ne1, nExp);
            t.down_exps_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(d.de_type), d.de_ne0, d.de_ne1, nExp);
            t.down_exps_scale_t = (d.down_exps_scale != nullptr) ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nExp) : nullptr;
            t.post_ffw_norm_2_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.post_ffw_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
            t.k_cpy = nullptr; t.v_cpy = nullptr; t.k_cpy2 = nullptr; t.v_cpy2 = nullptr;
            t.k_prev = nullptr; t.v_prev = nullptr;
        }

        // Causal masks for ggml_flash_attn_ext. Attention here mirrors the MoE
        // decode and the prefill paths (flash_attn_ext, not a manual mul_mat +
        // soft_max chain): the manual chain is numerically unreliable for
        // multi-token Q at head_dim 256/512 on Metal (silent F16 accumulator
        // overflow plus barely-exercised soft_max/mul_mat kernels), while
        // flash_attn_ext is the proven path on Metal AND CUDA and is faster.
        // One mask per distinct (kvLen, validLen, window) — at most three: a
        // circular-windowed SWA mask, a swaFresh sliding-window mask (window>0,
        // attends the fresh chunk), and a (flash-padded) global mask — shared
        // across layers. mask[qi][ki] = 0 iff ki < validLen (real, not padding)
        // AND ki <= (validLen-N)+qi (causal) AND (window==0 || ki > threshold-window)
        // (sliding-window low bound). For the circular-window read (window==0) the
        // SWA low-end is already below the cache view so no extra windowing is
        // needed; the swaFresh path (window=W) attends the full fresh chunk so it
        // applies the window low bound explicitly (same as the dense verify).
        struct VerifyMask { int kvLen; int validLen; int window; ggml_tensor* tensor; int dataIdx; };
        std::vector<VerifyMask> mask_cache;
        std::vector<std::vector<ggml_fp16_t>> mask_data_store;
        auto get_causal_mask = [&](int kvLen, int validLen, int window) -> ggml_tensor* {
            for (auto& m : mask_cache)
                if (m.kvLen == kvLen && m.validLen == validLen && m.window == window) return m.tensor;
            ggml_tensor* mt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kvLen, N);
            std::vector<ggml_fp16_t> data(static_cast<std::size_t>(kvLen) * N);
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            const int nPast = validLen - N;
            for (int qi = 0; qi < N; qi++)
            {
                const int threshold = nPast + qi;
                const int low = (window > 0) ? (threshold - window + 1) : 0;
                ggml_fp16_t* row = &data[static_cast<std::size_t>(qi) * kvLen];
                // Unmasked keys form a single contiguous band [lo, hi]; fill it
                // analytically rather than a per-element branch over [0, kvLen)
                // (the host loop blocks the GPU at long prefill — see dense verify).
                const int lo = (low > 0) ? low : 0;
                const int hi = std::min(threshold, validLen - 1);
                std::fill(row, row + kvLen, neg_inf);
                if (hi >= lo && lo < kvLen)
                    std::fill(row + lo, row + std::min(hi + 1, kvLen), zero_val);
            }
            const int idx = static_cast<int>(mask_data_store.size());
            mask_data_store.push_back(std::move(data));
            mask_cache.push_back({kvLen, validLen, window, mt, idx});
            return mt;
        };

        // Tiled attention (mirrors llama.cpp's n_ubatch and the dense verify's
        // TS_G4_SWA_TILE). A single full-N flash builds an O(N^2) F16 mask plus
        // O(N) q_rope / flash-output / score transients that, on the near-full
        // 26B-A4B GPU (~1 GB free with 14 GB experts resident), spill into WDDM
        // shared memory at long prefill (the 4k/8k cliff). Splitting the N queries
        // into tiles bounds the mask to [kLen, qLen<=tile]; for LOCAL (SWA) layers
        // each tile reads only its sliding-window key slice (less compute too),
        // while GLOBAL (full-attention) layers read [0, qe) — bounded mask, same
        // O(N^2) compute as llama. Both default-on; TS_G4_MOE_ATTN_TILED=0 reverts.
        static const bool moe_attn_tiled = []{ const char* e = std::getenv("TS_G4_MOE_ATTN_TILED"); return e == nullptr || e[0] != '0'; }();
        static const int moe_attn_tile = []{ const char* e = std::getenv("TS_G4_MOE_ATTN_TILE"); int v = e ? std::atoi(e) : 0; return (v >= 256) ? v : 1024; }();
        // Unified causal/sliding-window tile mask. gQ is the ABSOLUTE query
        // position (start_pos + tile-local index); ki indexes the key slice that
        // starts at logical position kStart. window<=0 means no sliding-window low
        // bound (global / full causal). Keys form a contiguous unmasked band
        // [lo, hi] per query row — filled analytically (a per-element host loop
        // blocks the GPU at long prefill).
        struct MoeTileMask { int kLen; int qLen; int qStartAbs; int kStart; int window; ggml_tensor* tensor; int dataIdx; };
        std::vector<MoeTileMask> tile_mask_cache;
        auto get_tile_mask = [&](int kLen, int qLen, int qStartAbs, int kStart, int window) -> ggml_tensor* {
            for (auto& m : tile_mask_cache)
                if (m.kLen == kLen && m.qLen == qLen && m.qStartAbs == qStartAbs && m.kStart == kStart && m.window == window) return m.tensor;
            ggml_tensor* mt = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kLen, qLen);
            std::vector<ggml_fp16_t> data(static_cast<std::size_t>(kLen) * qLen);
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
            for (int qi = 0; qi < qLen; qi++)
            {
                const int gQ = qStartAbs + qi;
                const int lo = (window > 0) ? std::max(0, gQ - window + 1 - kStart) : 0;
                int hi = gQ - kStart; if (hi > kLen - 1) hi = kLen - 1;
                ggml_fp16_t* row = &data[static_cast<std::size_t>(qi) * kLen];
                std::fill(row, row + kLen, neg_inf);
                if (hi >= lo && lo < kLen) std::fill(row + lo, row + hi + 1, zero_val);
            }
            const int idx = static_cast<int>(mask_data_store.size());
            mask_data_store.push_back(std::move(data));
            tile_mask_cache.push_back({kLen, qLen, qStartAbs, kStart, window, mt, idx});
            return mt;
        };

        ggml_tensor* hidden = current;
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGemma4MoELayerDesc& d = layers[l];
            MoeLayerTensors& t = lt[l];

            const int hd = d.head_dim;
            const int nH = num_heads;
            const int kvH = d.num_kv_heads;
            const int qDim = nH * hd;
            const int kDim = kvH * hd;
            const int cacheSize = d.cache_size;
            const bool isLocal = d.is_local != 0;
            const bool separate_qkv = d.separate_qkv != 0;
            const int nExp = d.num_experts;
            const int nUsed = d.num_experts_used;
            const std::int64_t ffDense = d.gu_ne1 / 2;
            const std::int64_t ffMoe = d.gue_ne1 / 2;
            const int rope_dims = d.rope_n_dims;
            ggml_tensor* rope_ff = isLocal ? nullptr : freq_factors_t;

            // ===== Attention (multi-token, flash) =====
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w);  // [H, N]

            // Tiling decision (see get_tile_mask). swaWindow = local SWA prefill that
            // overflows the circular window: at start_pos==0 attend the FRESH chunk
            // (swaFresh); at start_pos>0 gather the previous window from the rolling
            // cache and prepend it to the fresh chunk (swaPrev). GLOBAL (full-attention)
            // layers always tile when N is large; their Q/attn tensors are [qDim=8192,N]
            // (16 heads x 512) and dominate the spill. When separate_qkv, Q is projected
            // from `normed` per tile so the full-N q_rope is never materialised (tileQ);
            // a fused QKV weight keeps the full q_rope.
            const bool swaFresh = isLocal && start_pos == 0 && totalSeqLen > cacheSize;
            const bool swaPrev  = isLocal && start_pos != 0 && totalSeqLen > cacheSize;
            // Tile whenever it bounds the working set: swaPrev ALWAYS (its extended K/V
            // + the M-token layer tail otherwise peak higher than the start-of-prompt
            // chunk), otherwise once there are >= 3 query tiles.
            const bool moe_use_tiled = moe_attn_tiled
                && (swaPrev || (N > 2 * moe_attn_tile && (swaFresh || !isLocal)));
            const bool tileQ = moe_use_tiled && separate_qkv;

            ggml_tensor* q_lin = nullptr; ggml_tensor* k_lin; ggml_tensor* v_lin;
            if (separate_qkv)
            {
                if (!tileQ) q_lin = ggml_mul_mat(ctx, t.qkv_w, normed);  // Q projected per-tile when tileQ
                k_lin = ggml_mul_mat(ctx, t.k_w, normed);
                v_lin = ggml_mul_mat(ctx, t.v_w, normed);
            }
            else
            {
                ggml_tensor* qkv = ggml_mul_mat(ctx, t.qkv_w, normed);  // [qDim+2kDim, N]
                q_lin = ggml_cont(ctx, ggml_view_2d(ctx, qkv, qDim, N, qkv->nb[1], 0));
                k_lin = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kDim, N, qkv->nb[1], static_cast<std::size_t>(qDim) * sizeof(float)));
                v_lin = ggml_cont(ctx, ggml_view_2d(ctx, qkv, kDim, N, qkv->nb[1], static_cast<std::size_t>(qDim + kDim) * sizeof(float)));
            }

            // K/V projection + norm + RoPE (always full N — needed for the cache write
            // and as the attention key/value source for every query tile).
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_lin, hd, kvH, N);
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_lin, hd, kvH, N);
            k_3d = ggml_mul(ctx, ggml_rms_norm(ctx, k_3d, eps), t.k_norm_w);
            v_3d = ggml_rms_norm(ctx, v_3d, eps);
            ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, rope_ff, rope_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);  // [hd, kvH, N]

            // Full-N Q only when NOT projecting Q per tile (small N, or fused QKV).
            ggml_tensor* q_rope = nullptr;
            if (!tileQ)
            {
                ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_lin, hd, nH, N);
                q_3d = ggml_mul(ctx, ggml_rms_norm(ctx, q_3d, eps), t.q_norm_w);
                q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff, rope_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);  // [hd, nH, N]
            }

            // Write N new K/V (wrap-aware circular write for SWA; linear for global).
            ggml_tensor* k_write = ggml_cont(ctx, ggml_permute(ctx, k_rope, 0, 2, 1, 3));  // [hd, N, kvH]
            ggml_tensor* v_write = ggml_cont(ctx, ggml_permute(ctx, v_3d, 0, 2, 1, 3));     // [hd, N, kvH]

            // swaPrev: gather the previous window [start_pos-prevCount, start_pos) from
            // the rolling circular cache into a MATERIALISED F32 copy. The copy is built
            // into the graph (and ggml_build_forward_expand'd) AHEAD of the cache write
            // below, so on the single execution stream it reads the OLD cache contents
            // before this chunk overwrites them. prevCount = min(W, start_pos).
            const int prevCount = swaPrev ? std::min(cacheSize, start_pos) : 0;
            const int swaBase = start_pos - prevCount;   // logical position of the prev-window start
            if (swaPrev)
            {
                ggml_tensor* kpv = view_kv_cache_window(ctx, t.k_cached_t, hd, cacheSize, kvH, swaBase, prevCount, kvType);
                ggml_tensor* vpv = view_kv_cache_window(ctx, t.v_cached_t, hd, cacheSize, kvH, swaBase, prevCount, kvType);
                if (kpv == nullptr || vpv == nullptr)
                {
                    set_last_error("Gemma4 MoE model verify: failed to view prev SWA window.");
                    return 0;
                }
                ggml_tensor* kpf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, prevCount, kvH);
                ggml_tensor* vpf = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hd, prevCount, kvH);
                t.k_prev = ggml_cpy(ctx, kpv, kpf);   // F16/F32 view -> F32 copy (materialised before the write)
                t.v_prev = ggml_cpy(ctx, vpv, vpf);
            }

            // SWA prefill overflowing the circular window: only the LAST W positions are
            // written to the circular cache (for subsequent decode) — writing all N
            // would wrap the W buffer multiple times and corrupt it. Mirrors the dense
            // verify (TSGgml_Gemma4ModelVerify). Applies at any start_pos (swaFresh/swaPrev).
            const int writeOff = (isLocal && N > cacheSize) ? (N - cacheSize) : 0;
            const int writeLen = N - writeOff;
            const int writeStart = start_pos + writeOff;
            const int cacheBase = isLocal ? (writeStart % cacheSize) : writeStart;
            const int n1 = (isLocal && cacheBase + writeLen > cacheSize) ? (cacheSize - cacheBase) : writeLen;
            auto writePart = [&](ggml_tensor* cache, ggml_tensor* src, int srcOff, int dstSlot, int cnt) -> ggml_tensor* {
                ggml_tensor* s = ggml_view_3d(ctx, src, hd, cnt, kvH, src->nb[1], src->nb[2], static_cast<std::size_t>(srcOff) * src->nb[1]);
                ggml_tensor* dd = ggml_view_3d(ctx, cache, hd, cnt, kvH, cache->nb[1], cache->nb[2], static_cast<std::size_t>(dstSlot) * cache->nb[1]);
                return ggml_cpy(ctx, s, dd);
            };
            t.k_cpy = writePart(t.k_cached_t, k_write, writeOff, cacheBase, n1);
            t.v_cpy = writePart(t.v_cached_t, v_write, writeOff, cacheBase, n1);
            if (n1 < writeLen)
            {
                t.k_cpy2 = writePart(t.k_cached_t, k_write, writeOff + n1, 0, writeLen - n1);
                t.v_cpy2 = writePart(t.v_cached_t, v_write, writeOff + n1, 0, writeLen - n1);
            }

            int attendLen;
            int attnKvLen;
            int maskWindow;
            int keyBase;                             // logical position of k_full index 0
            ggml_tensor* k_full;
            ggml_tensor* v_full;
            if (swaFresh)
            {
                // Attend the FRESH full chunk (all N positions, post norm/RoPE) with a
                // sliding-window mask. ki == logical position (k_write is in chrono
                // order [0, N)). Local layers (head_dim 256) need no flash KV padding.
                attendLen = N;
                attnKvLen = N;
                maskWindow = cacheSize;              // sliding window W
                keyBase = 0;
                // flash_attn_ext needs K/V in the cache dtype; convert the fresh F32
                // chunk when the cache is F16 (no-op when F32).
                if (kvType == GGML_TYPE_F32)
                {
                    k_full = k_write;
                    v_full = v_write;
                }
                else
                {
                    ggml_tensor* kf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, N, kvH);
                    ggml_tensor* vf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, N, kvH);
                    k_full = ggml_cpy(ctx, k_write, kf);
                    v_full = ggml_cpy(ctx, v_write, vf);
                }
            }
            else if (swaPrev)
            {
                // Extended K/V = [prev window (prevCount) ++ fresh chunk (N)], covering
                // logical [swaBase, start_pos+N). Attended with a sliding-window mask;
                // keyBase = swaBase so the mask maps a key's index to its logical pos.
                const int bufLen = prevCount + N;
                attendLen = bufLen;
                attnKvLen = bufLen;
                maskWindow = cacheSize;
                keyBase = swaBase;
                ggml_tensor* kext = ggml_concat(ctx, t.k_prev, k_write, 1);   // [hd, prevCount+N, kvH] F32
                ggml_tensor* vext = ggml_concat(ctx, t.v_prev, v_write, 1);
                if (kvType == GGML_TYPE_F32)
                {
                    k_full = kext;
                    v_full = vext;
                }
                else
                {
                    ggml_tensor* kf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, bufLen, kvH);
                    ggml_tensor* vf = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kvType), hd, bufLen, kvH);
                    k_full = ggml_cpy(ctx, kext, kf);
                    v_full = ggml_cpy(ctx, vext, vf);
                }
            }
            else
            {
                attendLen = isLocal ? std::min(totalSeqLen, cacheSize) : totalSeqLen;
                const int activeStart = isLocal ? ((totalSeqLen - attendLen) % cacheSize) : 0;
                // Flash attention reads a (possibly padded) KV window; the padding slots
                // beyond attendLen are masked out. Padding only ever applies to global
                // (linear, activeStart==0) layers — local SWA layers (head_dim 256) need
                // no padding, so the wrap-around window is never combined with padding.
                attnKvLen = flash_attn_kv_length(attendLen, cacheSize, hd);
                maskWindow = 0;                       // window already enforced by the cache view
                keyBase = 0;
                k_full = view_kv_cache_window(ctx, t.k_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType);
                v_full = view_kv_cache_window(ctx, t.v_cached_t, hd, cacheSize, kvH, activeStart, attnKvLen, kvType);
            }
            if (k_full == nullptr || v_full == nullptr)
            {
                set_last_error("Gemma4 MoE model verify: failed to build KV cache views.");
                return 0;
            }

            // Post-attention layer tail (dense shared FFN + in-graph-routed experts +
            // final residual/scale) over a residual stream of M tokens. Factored out
            // so the query-tiled path runs it PER TILE (M = qLen) — bounding every
            // [*, M] FFN/router/MoE intermediate — while the small-N path runs it once
            // (M = N). The ops are token-independent, so per-tile output is byte-
            // identical to the full-N output. This is what bounds the gallocr peak to
            // ~the residual stream + K/V (the whole-layer [*, N] intermediates summed
            // to ~2.4 GB at N=8192 and spilled into WDDM shared memory on the 26B).
            auto layer_tail = [&](ggml_tensor* residual1, int M) -> ggml_tensor* {
                // dense shared FFN (fused GeGLU = gelu(gate)*up on the [2*ffDense, M] tensor)
                ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.ffn_norm_w);
                ggml_tensor* gu = ggml_mul_mat(ctx, t.gu_w, ffn_normed);                 // [2*ffDense, M]
                ggml_tensor* dense_h = ggml_geglu(ctx, gu);                             // [ffDense, M]
                ggml_tensor* dense_down = ggml_mul_mat(ctx, t.down_w, dense_h);         // [H, M]
                ggml_tensor* mlp = ggml_mul(ctx, ggml_rms_norm(ctx, dense_down, eps), t.post_ffw_norm_1_w);
                // MoE router (in-graph)
                ggml_tensor* route_n = ggml_rms_norm(ctx, residual1, eps);              // [H, M]
                route_n = ggml_scale(ctx, route_n, d.inv_sqrt_hidden);
                if (t.gate_inp_scale_t != nullptr)
                    route_n = ggml_mul(ctx, route_n, t.gate_inp_scale_t);
                ggml_tensor* router_logits = ggml_mul_mat(ctx, t.gate_inp_w, route_n);  // [nExp, M]
                ggml_tensor* probs = ggml_soft_max(ctx, router_logits);                 // [nExp, M]
                ggml_tensor* sel = ggml_top_k(ctx, probs, nUsed);                       // [nUsed, M] i32
                ggml_tensor* probs_r = ggml_reshape_3d(ctx, probs, 1, nExp, M);
                ggml_tensor* w = ggml_get_rows(ctx, probs_r, sel);                      // [1, nUsed, M]
                ggml_tensor* w_2d = ggml_reshape_2d(ctx, w, nUsed, M);                  // [nUsed, M]
                ggml_tensor* w_sum = ggml_sum_rows(ctx, w_2d);                          // [1, M]
                w_2d = ggml_div(ctx, w_2d, w_sum);                                      // renormalise over selected
                if (t.down_exps_scale_t != nullptr)
                {
                    ggml_tensor* scale_b = ggml_repeat(ctx, ggml_reshape_3d(ctx, t.down_exps_scale_t, 1, nExp, 1), probs_r);  // [1, nExp, M]
                    ggml_tensor* sel_scale = ggml_get_rows(ctx, scale_b, sel);          // [1, nUsed, M]
                    w_2d = ggml_mul(ctx, w_2d, ggml_reshape_2d(ctx, sel_scale, nUsed, M));
                }
                ggml_tensor* w_final = ggml_reshape_3d(ctx, w_2d, 1, nUsed, M);
                // MoE experts (M is already tile-bounded — no inner token tiling needed)
                ggml_tensor* moe_in = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.pre_ffw_norm_2_w);  // [H, M]
                ggml_tensor* moe_in_3d = ggml_reshape_3d(ctx, moe_in, H, 1, M);
                ggml_tensor* gate_up = ggml_mul_mat_id(ctx, t.gate_up_exps_t, moe_in_3d, sel);   // [2*ffMoe, nUsed, M]
                ggml_tensor* moe_gate = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
                ggml_tensor* moe_up = ggml_view_3d(ctx, gate_up, ffMoe, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], static_cast<std::size_t>(ffMoe) * gate_up->nb[0]);
                ggml_tensor* moe_act = ggml_geglu_split(ctx, moe_gate, moe_up);                  // [ffMoe, nUsed, M]
                ggml_tensor* moe_down = ggml_mul_mat_id(ctx, t.down_exps_t, moe_act, sel);       // [H, nUsed, M]
                ggml_tensor* weighted = ggml_mul(ctx, moe_down, w_final);                        // [H, nUsed, M]
                ggml_tensor* moe_out = ggml_view_2d(ctx, weighted, H, M, weighted->nb[2], 0);
                for (int u = 1; u < nUsed; ++u)
                {
                    ggml_tensor* view_u = ggml_view_2d(ctx, weighted, H, M, weighted->nb[2], static_cast<std::size_t>(u) * weighted->nb[1]);
                    moe_out = ggml_add(ctx, moe_out, view_u);
                }
                ggml_tensor* moe_normed = ggml_mul(ctx, ggml_rms_norm(ctx, moe_out, eps), t.post_ffw_norm_2_w);
                mlp = ggml_add(ctx, mlp, moe_normed);
                // final residual + layer scale
                ggml_tensor* mlp_normed = ggml_mul(ctx, ggml_rms_norm(ctx, mlp, eps), t.post_ffw_norm_w);
                ggml_tensor* result = ggml_add(ctx, residual1, mlp_normed);
                if (std::fabs(d.layer_output_scale - 1.0f) > 1e-9f)
                    result = ggml_scale(ctx, result, d.layer_output_scale);
                return result;
            };

            // Attention (Gemma scale = 1.0) fused with the layer tail. The tiled path
            // processes the WHOLE layer (attention + FFN + experts) in query tiles so
            // only the residual stream + K/V are full-N; everything else is tile-
            // bounded. The full-N path (small N / in-window-circular reads) keeps the
            // single flash + one layer_tail call.
            ggml_tensor* result;
            if (moe_use_tiled)
            {
                ggml_tensor* result_acc = nullptr;
                // Fused QKV (no separate Q weight): Q can't be re-projected from
                // `normed` per tile, so keep a full-N q_rope and view it per tile.
                ggml_tensor* q_t_full = (!tileQ) ? ggml_permute(ctx, q_rope, 0, 2, 1, 3) : nullptr;  // [hd, N, nH]
                for (int qs = 0; qs < N; qs += moe_attn_tile)
                {
                    const int qe = (qs + moe_attn_tile < N) ? (qs + moe_attn_tile) : N;
                    const int qLen = qe - qs;

                    // Per-tile Q (project from the normed-column slice) or a view of
                    // the full q_t when QKV is fused.
                    ggml_tensor* q_tile;
                    if (tileQ)
                    {
                        ggml_tensor* normed_tile = ggml_view_2d(ctx, normed, H, qLen,
                            normed->nb[1], static_cast<std::size_t>(qs) * normed->nb[1]);
                        ggml_tensor* q_lin_t = ggml_mul_mat(ctx, t.qkv_w, normed_tile);     // [qDim, qLen]
                        ggml_tensor* q_3d_t = ggml_reshape_3d(ctx, q_lin_t, hd, nH, qLen);
                        q_3d_t = ggml_mul(ctx, ggml_rms_norm(ctx, q_3d_t, eps), t.q_norm_w);
                        ggml_tensor* pos_t = ggml_view_1d(ctx, pos_tensor, qLen,
                            static_cast<std::size_t>(qs) * sizeof(std::int32_t));
                        ggml_tensor* q_rope_t = ggml_rope_ext(ctx, q_3d_t, pos_t, rope_ff,
                            rope_dims, 2, 0, d.rope_base, 1.0f, 0, 1, 0, 0);               // [hd, nH, qLen]
                        q_tile = ggml_permute(ctx, q_rope_t, 0, 2, 1, 3);                  // [hd, qLen, nH]
                    }
                    else
                    {
                        q_tile = ggml_view_3d(ctx, q_t_full, hd, qLen, nH,
                            q_t_full->nb[1], q_t_full->nb[2], static_cast<std::size_t>(qs) * q_t_full->nb[1]);
                    }

                    int ks, kLen, window, kStartLogical;
                    if (maskWindow > 0)
                    {
                        // Windowed (swaFresh keyBase==0, or swaPrev keyBase==swaBase).
                        // k_full covers logical [keyBase, ...]; bufQ = index of this
                        // tile's first query's own key. Attend its window slice; the low
                        // bound is enforced by the mask.
                        const int bufQ = (start_pos + qs) - keyBase;
                        ks = (bufQ > maskWindow) ? (bufQ - maskWindow) : 0;
                        kLen = (bufQ + qLen) - ks;
                        if (kLen > attnKvLen) kLen = attnKvLen;
                        window = maskWindow;            // sliding window W
                        kStartLogical = keyBase + ks;   // logical position of the slice start
                    }
                    else
                    {
                        // Global linear cache: attend [0, start_pos+qe). Pad kLen for the
                        // head_dim-512 flash kernel (kvLen % 256 == 0).
                        ks = 0;
                        kLen = flash_attn_kv_length(start_pos + qe, cacheSize, hd);
                        if (kLen > attnKvLen) kLen = attnKvLen;
                        window = 0;                     // pure causal
                        kStartLogical = 0;
                    }
                    ggml_tensor* k_tile = ggml_view_3d(ctx, k_full, hd, kLen, kvH,
                        k_full->nb[1], k_full->nb[2], static_cast<std::size_t>(ks) * k_full->nb[1]);
                    ggml_tensor* v_tile = ggml_view_3d(ctx, v_full, hd, kLen, kvH,
                        v_full->nb[1], v_full->nb[2], static_cast<std::size_t>(ks) * v_full->nb[1]);
                    ggml_tensor* m_tile = get_tile_mask(kLen, qLen, start_pos + qs, kStartLogical, window);
                    ggml_tensor* fa = ggml_flash_attn_ext(ctx, q_tile, k_tile, v_tile, m_tile, 1.0f, 0.0f, 0.0f);
                    ggml_flash_attn_ext_set_prec(fa, GGML_PREC_F32);
                    if (qs == 0 && !backend_supports_op(fa))
                    {
                        set_last_error("Gemma4 MoE model verify: tiled flash attention unsupported for this shape; use per-op path.");
                        return 0;
                    }
                    // O projection (per tile) → post-attn norm → residual → layer tail.
                    ggml_tensor* fa_flat = ggml_reshape_2d(ctx, fa, qDim, qLen);
                    ggml_tensor* o_tile = ggml_mul_mat(ctx, t.o_w, fa_flat);               // [H, qLen]
                    ggml_tensor* post_attn = ggml_mul(ctx, ggml_rms_norm(ctx, o_tile, eps), t.post_attn_norm_w);
                    ggml_tensor* hidden_tile = ggml_view_2d(ctx, hidden, H, qLen,
                        hidden->nb[1], static_cast<std::size_t>(qs) * hidden->nb[1]);
                    ggml_tensor* residual1 = ggml_add(ctx, hidden_tile, post_attn);        // [H, qLen]
                    ggml_tensor* result_tile = layer_tail(residual1, qLen);                // [H, qLen]
                    result_acc = (result_acc == nullptr) ? result_tile : ggml_concat(ctx, result_acc, result_tile, 1);  // [H, qe]
                }
                result = result_acc;                                                      // [H, N]
            }
            else
            {
                ggml_tensor* q_t = ggml_permute(ctx, q_rope, 0, 2, 1, 3);                  // [hd, N, nH]
                // Windowed reads (swaFresh/swaPrev) use the keyBase-aware tile mask over
                // the whole chunk (handles keyBase != 0 for swaPrev); global / in-window
                // circular reads use the position-relative causal mask.
                ggml_tensor* fa_mask = (maskWindow > 0)
                    ? get_tile_mask(attnKvLen, N, start_pos, keyBase, maskWindow)
                    : get_causal_mask(attnKvLen, attendLen, maskWindow);
                ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q_t, k_full, v_full, fa_mask, 1.0f, 0.0f, 0.0f);
                ggml_flash_attn_ext_set_prec(attn_out, GGML_PREC_F32);
                if (!backend_supports_op(attn_out))
                {
                    // No supported flash kernel for this shape (e.g. an exotic head_dim):
                    // fall back to the per-op verify rather than the numerically-fragile
                    // manual chain.
                    set_last_error("Gemma4 MoE model verify: flash attention unsupported for this shape; use per-op path.");
                    return 0;
                }
                // flash_attn_ext returns [hd, nH, N, 1] — each column holds all heads
                // contiguously, exactly what the O projection wants.
                ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, qDim, N);
                ggml_tensor* o_out = ggml_mul_mat(ctx, t.o_w, attn_flat);                  // [H, N]
                ggml_tensor* post_attn = ggml_mul(ctx, ggml_rms_norm(ctx, o_out, eps), t.post_attn_norm_w);
                ggml_tensor* residual1 = ggml_add(ctx, hidden, post_attn);                // [H, N]
                result = layer_tail(residual1, N);                                         // [H, N]
            }

            hidden = result;
        }

        ggml_tensor* hidden_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, N);
        ggml_tensor* out_cpy = ggml_cpy(ctx, hidden, hidden_out);
        ggml_set_output(out_cpy);

        // Each query tile runs the whole layer (per-tile Q proj/norm/rope + flash +
        // O proj + post-attn + residual + dense FFN + router + experts + concat) ≈ 56
        // nodes; budget for that plus the per-layer full-N K/V projection + the
        // optional swaPrev gather/concat/convert (~128 fixed headroom).
        const int attnTilesPerLayer = moe_attn_tiled ? ((N + moe_attn_tile - 1) / moe_attn_tile) : 1;
        const std::size_t graph_size = static_cast<std::size_t>(num_layers)
            * (128 + static_cast<std::size_t>(attnTilesPerLayer) * 56) + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        // swaPrev: add the prev-window F32 copies FIRST so they are ordered ahead of
        // the cache writes below — on the single execution stream they then read the
        // old (previous-chunk) cache contents before this chunk overwrites them.
        for (int l = 0; l < num_layers; l++)
        {
            if (lt[l].k_prev != nullptr) ggml_build_forward_expand(graph, lt[l].k_prev);
            if (lt[l].v_prev != nullptr) ggml_build_forward_expand(graph, lt[l].v_prev);
        }
        for (int l = 0; l < num_layers; l++)
        {
            ggml_build_forward_expand(graph, lt[l].k_cpy);
            ggml_build_forward_expand(graph, lt[l].v_cpy);
            if (lt[l].k_cpy2 != nullptr) ggml_build_forward_expand(graph, lt[l].k_cpy2);
            if (lt[l].v_cpy2 != nullptr) ggml_build_forward_expand(graph, lt[l].v_cpy2);
        }
        ggml_build_forward_expand(graph, out_cpy);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;
        auto bind_or_mark = [&](ggml_tensor* tgt, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (tgt == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, tgt, data, bytes, buf, addr, needs_upload, usage))
                {
                    if (ggml_backend_tensor_alloc(buf, tgt, addr) == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload) upload_list.push_back({tgt, data, bytes});
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
                    if (!cacheable) ephemeral_bufs.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, tgt, data) == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({tgt, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlGemma4MoELayerDesc& d = layers[l];
            MoeLayerTensors& t = lt[l];
            const int hd = d.head_dim;
            const int kvH = d.num_kv_heads;
            const int nExp = d.num_experts;
            const int cacheSize = d.cache_size;

            bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes), true);
            if (t.k_w != nullptr)
            {
                bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes), true);
                bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes), true);
            }
            bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes), true);
            bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes), true);
            bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes), true);
            bind_or_mark(t.gate_up_exps_t, d.gate_up_exps, static_cast<std::size_t>(d.gue_bytes), true);
            bind_or_mark(t.down_exps_t, d.down_exps, static_cast<std::size_t>(d.de_bytes), true);
            bind_or_mark(t.gate_inp_w, d.gate_inp_w, static_cast<std::size_t>(H) * nExp * sizeof(float), true);
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.ffn_norm_w, d.ffn_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_1_w, d.post_ffw_norm_1_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.pre_ffw_norm_2_w, d.pre_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_2_w, d.post_ffw_norm_2_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.post_ffw_norm_w, d.post_ffw_norm_w, static_cast<std::size_t>(H) * sizeof(float), true);
            bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(hd) * sizeof(float), true);
            bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(hd) * sizeof(float), true);
            if (t.gate_inp_scale_t != nullptr)
                bind_or_mark(t.gate_inp_scale_t, d.gate_inp_scale, static_cast<std::size_t>(H) * sizeof(float), true);
            if (t.down_exps_scale_t != nullptr)
                bind_or_mark(t.down_exps_scale_t, d.down_exps_scale, static_cast<std::size_t>(nExp) * sizeof(float), true);
            bind_or_mark(t.k_cached_t, d.k_cache, kv_cache_bytes(kvH, cacheSize, hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            bind_or_mark(t.v_cached_t, d.v_cache, kv_cache_bytes(kvH, cacheSize, hd, kvType), true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        if (freq_factors_t != nullptr)
            bind_or_mark(freq_factors_t, freq_data, static_cast<std::size_t>(freq_len) * sizeof(float), true);

        // Graph-aware allocation: gallocr packs the N-token intermediates by tensor
        // LIFETIME (peak, not sum), unlike the linear alloc_ctx_tensors_reuse bump
        // allocator the single-token decode uses. For a K+1 verify over 30 layers
        // the linear sum is hundreds of MB — on the 26B-A4B (model already ~13 GB
        // resident) that exhausts VRAM and starves the draft/weight caches (the
        // draft step then thrashes). gallocr's peak is ~10-20x smaller. The
        // pre-bound weights/KV caches above already own buffers and are skipped.
        // Persistent gallocr (reused across verify calls, grown on demand) instead
        // of a fresh ggml_gallocr_new()/_free() each step. The MoE verify graph's
        // intermediate buffer is ~400 MB; allocating and freeing that on every
        // verify churned Metal's shared (vm_allocate) device memory and fragmented
        // it over hundreds of steps until a contiguous allocation failed
        // (kIOGPUCommandBufferCallbackErrorOutOfMemory). Reusing one gallocr keeps
        // the buffer resident and stable, which removes the fragmentation source
        // (and the per-call ~20 ms allocation).
        if (!alloc_graph_reuse_gallocr(graph))
        {
            set_last_error("Gemma4 MoE model verify: graph allocation failed.");
            return 0;
        }

        host_read_barrier();

        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);

        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(H) * N * sizeof(float));

        std::vector<std::int32_t> pos_vals(N);
        for (int i = 0; i < N; i++) pos_vals[i] = start_pos + i;
        ggml_backend_tensor_set(pos_tensor, pos_vals.data(), 0, static_cast<std::size_t>(N) * sizeof(std::int32_t));

        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, freq_data, 0, static_cast<std::size_t>(freq_len) * sizeof(float));

        for (auto& m : mask_cache)
            ggml_backend_tensor_set(m.tensor, mask_data_store[m.dataIdx].data(), 0,
                mask_data_store[m.dataIdx].size() * sizeof(ggml_fp16_t));
        for (auto& m : tile_mask_cache)
            ggml_backend_tensor_set(m.tensor, mask_data_store[m.dataIdx].data(), 0,
                mask_data_store[m.dataIdx].size() * sizeof(ggml_fp16_t));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("Gemma4 MoE model verify: graph execution failed.");
            return 0;
        }

        // The persistent gallocr buffer is NOT freed here — it is reused by the
        // next verify. The finalize below queues an async device->host blit of
        // hidden_out (which lives in that persistent buffer); it is drained either
        // by the C# caller's first logits read (host_read_barrier in
        // GetFloatPointer) or by the next verify's leading host_read_barrier
        // before it reuses the buffer, so there is no read-after-write hazard.
        finalize_compute_with_download(hidden_out, hidden_data, static_cast<std::size_t>(H) * N * sizeof(float));
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
        set_last_error("Unknown error in Gemma4 MoE model verify.");
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
