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
#include "ggml_ops_transformer_common.h"
#include <chrono>
#include <cstdio>

using namespace tsg;

namespace
{
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
        ggml_tensor* attn_out = flash_attn_ext_guarded(ctx, "flash attention decode",
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
        tsg::PendingDeviceCopies pending;  // cached device copies bound below and not yet written: dropped unless committed after the upload
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
                        if (needs_upload) { upload_list.push_back({t, data, bytes}); pending.add(data); }
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
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);
        pending.commit();

        ggml_backend_tensor_set(q_in, q_data, 0, static_cast<std::size_t>(q_dim) * sizeof(float));
        ggml_backend_tensor_set(k_in, k_data, 0, static_cast<std::size_t>(kv_dim) * sizeof(float));
        ggml_backend_tensor_set(v_in, v_data, 0, static_cast<std::size_t>(kv_dim) * sizeof(float));

        ggml_status status = tsg::compute_graph(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for flash attention decode.");
            return 0;
        }

        finalize_compute_with_download(attn_result, out_data, static_cast<std::size_t>(q_dim) * sizeof(float));

        clear_last_error();
        return 1;
    }
}

// ============================================================================
// Flash attention decode (single-token, single-layer).
//
// Use this when the surrounding architecture pre-processes Q/K/V (e.g. fused
// gated projections, sigmoid-gated Q outputs, custom QK normalization) in a
// way that prevents folding the entire layer into one fused graph.
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
