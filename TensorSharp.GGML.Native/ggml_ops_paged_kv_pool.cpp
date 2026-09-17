// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// DEVICE-RESIDENT paged K/V pool.
//
// This is the follow-up TSGgml_PagedAttentionForwardDevice's comment asks for:
// "K/V scratch is still host-side because the paged K/V pool lives in C#
// managed arrays for now; making that zero-copy is a follow-up."
//
// Why it matters. With the pool in managed float[], every layer of every step
// gathers the WHOLE sequence history out of host memory and uploads it, so the
// batched path costs O(total_history x layers) of PCIe traffic per step. That
// is what caps continuous batching: measured on gemma-4-E4B / 1x Blackwell, the
// batched path ran 7.7x slower per token than per-sequence fused decode at N=1
// and saturated at ~69 tok/s no matter how many sequences were in flight, so
// concurrency bought no throughput at all.
//
// Here the pool is a pair of ggml tensors per layer, allocated once on the
// backend and never leaving it:
//
//     k[layer], v[layer] : [kv_dim, num_slots]   (kv_dim = num_kv_heads*head_dim)
//
// One ROW per token slot, which is what makes both halves ordinary ggml ops:
//
//   * WRITE  ggml_set_rows(pool, new_kv, slot_ids)   - only the step's new
//            tokens cross the bus, O(num_tokens), not O(history).
//   * READ   ggml_get_rows(pool, row_ids)            - the sequence's history
//            is gathered ON DEVICE straight into the flash-attention graph;
//            nothing is downloaded.
//
// The same ggml_set_rows / ggml_get_rows pairing DeepSeek V4's executor already
// uses for its device-resident KV rings (ggml_ops_deepseek4.cpp), so this is the
// established in-repo idiom rather than a new mechanism.
//
// Only the tiny index vectors (one int per token, per sequence) are host data.

#include "ggml_ops_internal.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

using namespace tsg;

namespace
{
    struct paged_kv_pool
    {
        int num_layers   = 0;
        int num_blocks   = 0;
        int block_size   = 0;
        int num_kv_heads = 0;
        int head_dim     = 0;
        int64_t kv_dim   = 0;
        int64_t num_slots = 0;

        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        std::vector<ggml_tensor*> k;
        std::vector<ggml_tensor*> v;

        ~paged_kv_pool()
        {
            if (buf) ggml_backend_buffer_free(buf);
            if (ctx) ggml_free(ctx);
        }
    };

    // ggml_set_rows / ggml_get_rows take their indices as a tensor. Both want
    // an I64 index vector for set_rows and I32 for get_rows, so keep one small
    // scratch of each rather than reallocating per call.
    std::mutex g_pool_mutex;

    bool build_and_run(ggml_context* ctx, ggml_cgraph* graph, const char* tag)
    {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
        if (buf == nullptr)
        {
            set_last_error("paged-kv-pool: failed to allocate graph tensors.");
            return false;
        }
        // The graph's leaf uploads happen before compute; the caller has already
        // filled them via ggml_backend_tensor_set.
        ggml_status st = graph_compute_profiled(g_backend, graph, tag);
        ggml_backend_buffer_free(buf);
        if (st != GGML_STATUS_SUCCESS)
        {
            set_last_error("paged-kv-pool: graph compute failed.");
            return false;
        }
        return true;
    }
}

// ============================================================================
// TSGgml_PagedKvPoolCreate
//
// Allocates the per-layer K/V pool on the active backend and returns an opaque
// handle. num_blocks * block_size slots per layer; a slot holds one token's
// K (and V) for every KV head.
// ============================================================================
TSG_EXPORT void* TSGgml_PagedKvPoolCreate(
    int num_layers, int num_blocks, int block_size, int num_kv_heads, int head_dim)
{
    try
    {
        if (!ensure_backend()) return nullptr;
        if (num_layers <= 0 || num_blocks <= 0 || block_size <= 0 ||
            num_kv_heads <= 0 || head_dim <= 0)
        {
            set_last_error("paged-kv-pool: invalid geometry.");
            return nullptr;
        }

        auto pool = std::make_unique<paged_kv_pool>();
        pool->num_layers   = num_layers;
        pool->num_blocks   = num_blocks;
        pool->block_size   = block_size;
        pool->num_kv_heads = num_kv_heads;
        pool->head_dim     = head_dim;
        pool->kv_dim       = static_cast<int64_t>(num_kv_heads) * head_dim;
        pool->num_slots    = static_cast<int64_t>(num_blocks) * block_size;

        ggml_init_params cp = {
            static_cast<size_t>(num_layers * 2 + 16) * ggml_tensor_overhead(),
            nullptr,
            /*no_alloc*/ true
        };
        pool->ctx = ggml_init(cp);
        if (!pool->ctx)
        {
            set_last_error("paged-kv-pool: ggml_init failed.");
            return nullptr;
        }

        pool->k.resize(num_layers);
        pool->v.resize(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            pool->k[l] = ggml_new_tensor_2d(pool->ctx, GGML_TYPE_F32, pool->kv_dim, pool->num_slots);
            pool->v[l] = ggml_new_tensor_2d(pool->ctx, GGML_TYPE_F32, pool->kv_dim, pool->num_slots);
            if (!pool->k[l] || !pool->v[l])
            {
                set_last_error("paged-kv-pool: tensor creation failed.");
                return nullptr;
            }
            ggml_format_name(pool->k[l], "paged_k.%d", l);
            ggml_format_name(pool->v[l], "paged_v.%d", l);
        }

        pool->buf = ggml_backend_alloc_ctx_tensors(pool->ctx, g_backend);
        if (!pool->buf)
        {
            set_last_error("paged-kv-pool: backend allocation failed (out of VRAM?).");
            return nullptr;
        }

        return pool.release();
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return nullptr; }
    catch (...) { set_last_error("Unknown error in TSGgml_PagedKvPoolCreate."); return nullptr; }
}

TSG_EXPORT void TSGgml_PagedKvPoolFree(void* handle)
{
    delete static_cast<paged_kv_pool*>(handle);
}

// Bytes one layer's K (or V) occupies; the caller uses this to report VRAM.
TSG_EXPORT int64_t TSGgml_PagedKvPoolBytes(void* handle)
{
    auto* pool = static_cast<paged_kv_pool*>(handle);
    if (!pool) return 0;
    return static_cast<int64_t>(pool->num_layers) * 2 * pool->kv_dim * pool->num_slots *
           static_cast<int64_t>(sizeof(float));
}

// ============================================================================
// TSGgml_PagedKvPoolGrow
//
// The engine hands out block ids lazily, so the pool may need to grow mid-run.
// A host float[] could just be reallocated and memcpy'd; a device pool cannot,
// so allocate a second set of tensors, copy the live prefix ON DEVICE, and swap.
// Returns 1 on success, 0 on failure (caller keeps the old pool and can decline
// the batched path rather than lose K/V).
// ============================================================================
TSG_EXPORT int TSGgml_PagedKvPoolGrow(void* handle, int new_num_blocks)
{
    try
    {
        auto* pool = static_cast<paged_kv_pool*>(handle);
        if (!pool) { set_last_error("paged-kv-pool: bad handle (grow)."); return 0; }
        if (new_num_blocks <= pool->num_blocks) return 1;   // already big enough
        if (!ensure_backend()) return 0;

        std::lock_guard<std::mutex> guard(g_pool_mutex);

        const int64_t new_slots = static_cast<int64_t>(new_num_blocks) * pool->block_size;

        ggml_init_params cp = {
            static_cast<size_t>(pool->num_layers * 2 + 16) * ggml_tensor_overhead(),
            nullptr, /*no_alloc*/ true
        };
        ggml_context* nctx = ggml_init(cp);
        if (!nctx) { set_last_error("paged-kv-pool: ggml_init failed (grow)."); return 0; }

        std::vector<ggml_tensor*> nk(pool->num_layers), nv(pool->num_layers);
        for (int l = 0; l < pool->num_layers; l++)
        {
            nk[l] = ggml_new_tensor_2d(nctx, GGML_TYPE_F32, pool->kv_dim, new_slots);
            nv[l] = ggml_new_tensor_2d(nctx, GGML_TYPE_F32, pool->kv_dim, new_slots);
            ggml_format_name(nk[l], "paged_k.%d", l);
            ggml_format_name(nv[l], "paged_v.%d", l);
        }
        ggml_backend_buffer_t nbuf = ggml_backend_alloc_ctx_tensors(nctx, g_backend);
        if (!nbuf)
        {
            ggml_free(nctx);
            set_last_error("paged-kv-pool: backend allocation failed (grow; out of VRAM?).");
            return 0;
        }

        // Copy the live prefix, one graph for all layers.
        {
            PooledContextHandle context;
            if (!context.init(static_cast<size_t>(pool->num_layers) * 8 * ggml_tensor_overhead() + (1 << 20)))
            {
                ggml_backend_buffer_free(nbuf); ggml_free(nctx);
                set_last_error("paged-kv-pool: failed to acquire ggml context (grow).");
                return 0;
            }
            ggml_context* ctx = context.value;
            ggml_cgraph* graph = ggml_new_graph_custom(ctx, static_cast<size_t>(pool->num_layers) * 8 + 32, false);
            for (int l = 0; l < pool->num_layers; l++)
            {
                ggml_tensor* dstk = ggml_view_2d(ctx, nk[l], pool->kv_dim, pool->num_slots, nk[l]->nb[1], 0);
                ggml_tensor* dstv = ggml_view_2d(ctx, nv[l], pool->kv_dim, pool->num_slots, nv[l]->nb[1], 0);
                ggml_tensor* ck = ggml_cpy(ctx, pool->k[l], dstk);
                ggml_tensor* cv = ggml_cpy(ctx, pool->v[l], dstv);
                ggml_build_forward_expand(graph, ck);
                ggml_build_forward_expand(graph, cv);
            }
            ggml_status st = graph_compute_profiled(g_backend, graph, "paged_kv_grow");
            if (st != GGML_STATUS_SUCCESS)
            {
                ggml_backend_buffer_free(nbuf); ggml_free(nctx);
                set_last_error("paged-kv-pool: grow copy failed.");
                return 0;
            }
        }

        if (pool->buf) ggml_backend_buffer_free(pool->buf);
        if (pool->ctx) ggml_free(pool->ctx);
        pool->ctx = nctx;
        pool->buf = nbuf;
        pool->k = std::move(nk);
        pool->v = std::move(nv);
        pool->num_blocks = new_num_blocks;
        pool->num_slots = new_slots;
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in TSGgml_PagedKvPoolGrow."); return 0; }
}

// ============================================================================
// TSGgml_PagedKvPoolScatter
//
// Writes this step's new K and V into the pool at the slots given by
// slot_mapping. Only num_tokens rows cross the bus - the history stays put.
// ============================================================================
TSG_EXPORT int TSGgml_PagedKvPoolScatter(
    void* handle,
    int layer,
    const float* k_data,          // [num_tokens, kv_dim] host
    const float* v_data,          // [num_tokens, kv_dim] host
    const int* slot_mapping,      // [num_tokens]
    int num_tokens)
{
    try
    {
        auto* pool = static_cast<paged_kv_pool*>(handle);
        if (!pool || layer < 0 || layer >= pool->num_layers)
        {
            set_last_error("paged-kv-pool: bad handle or layer.");
            return 0;
        }
        if (num_tokens <= 0) return 1;
        if (!ensure_backend()) return 0;

        for (int t = 0; t < num_tokens; t++)
        {
            if (slot_mapping[t] < 0 || slot_mapping[t] >= pool->num_slots)
            {
                set_last_error("paged-kv-pool: slot_mapping out of range.");
                return 0;
            }
        }

        std::lock_guard<std::mutex> guard(g_pool_mutex);

        PooledContextHandle context;
        if (!context.init(4 * 1024 * 1024))
        {
            set_last_error("paged-kv-pool: failed to acquire ggml context (scatter).");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_tensor* k_in  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pool->kv_dim, num_tokens);
        ggml_tensor* v_in  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pool->kv_dim, num_tokens);
        ggml_tensor* idx   = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, num_tokens);

        ggml_tensor* wk = ggml_set_rows(ctx, pool->k[layer], k_in, idx);
        ggml_tensor* wv = ggml_set_rows(ctx, pool->v[layer], v_in, idx);
        ggml_set_output(wk);
        ggml_set_output(wv);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, wk);
        ggml_build_forward_expand(graph, wv);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
        if (!buf)
        {
            set_last_error("paged-kv-pool: scatter allocation failed.");
            return 0;
        }

        std::vector<int64_t> idx64(num_tokens);
        for (int t = 0; t < num_tokens; t++) idx64[t] = slot_mapping[t];

        const size_t row_bytes = static_cast<size_t>(pool->kv_dim) * sizeof(float);
        ggml_backend_tensor_set(k_in, k_data, 0, static_cast<size_t>(num_tokens) * row_bytes);
        ggml_backend_tensor_set(v_in, v_data, 0, static_cast<size_t>(num_tokens) * row_bytes);
        ggml_backend_tensor_set(idx, idx64.data(), 0, idx64.size() * sizeof(int64_t));

        ggml_status st = graph_compute_profiled(g_backend, graph, "paged_kv_scatter");
        ggml_backend_buffer_free(buf);
        if (st != GGML_STATUS_SUCCESS)
        {
            set_last_error("paged-kv-pool: scatter compute failed.");
            return 0;
        }
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in TSGgml_PagedKvPoolScatter."); return 0; }
}

// ============================================================================
// TSGgml_PagedKvPoolAttention
//
// Flash attention for a whole batch against the device-resident pool. Per
// sequence the history is gathered with ggml_get_rows INSIDE the graph, so the
// K/V never leaves the device; only the row-index vector is uploaded.
//
// q_data / out_data are [num_tokens, num_heads*head_dim] host buffers. Keeping
// them host-side here (rather than the zero-copy device bind the older entry
// point does) is deliberate for this first cut: they are O(num_tokens), the
// small term, and it keeps the contract identical to the managed path being
// replaced so correctness can be compared directly.
// ============================================================================
TSG_EXPORT int TSGgml_PagedKvPoolAttention(
    void* handle,
    int layer,
    const float* q_data,
    float* out_data,
    const int* query_start_loc,   // [num_seqs+1]
    const int* seq_lens,          // [num_seqs]
    const int* positions,         // [num_tokens]
    const int* block_table_flat,
    const int* block_table_offsets, // [num_seqs+1]
    int num_seqs,
    int num_tokens,
    int num_heads,
    int sliding_window,
    float scale)
{
    try
    {
        auto* pool = static_cast<paged_kv_pool*>(handle);
        if (!pool || layer < 0 || layer >= pool->num_layers)
        {
            set_last_error("paged-kv-pool: bad handle or layer.");
            return 0;
        }
        if (num_seqs <= 0 || num_tokens <= 0) return 1;
        if (!ensure_backend()) return 0;
        if (num_heads % pool->num_kv_heads != 0)
        {
            set_last_error("paged-kv-pool: num_heads must be divisible by num_kv_heads.");
            return 0;
        }

        const int head_dim = pool->head_dim;
        const int num_kv_heads = pool->num_kv_heads;
        const int q_dim = num_heads * head_dim;

        std::lock_guard<std::mutex> guard(g_pool_mutex);

        for (int s = 0; s < num_seqs; s++)
        {
            const int q_start = query_start_loc[s];
            const int q_end   = query_start_loc[s + 1];
            const int num_q   = q_end - q_start;
            if (num_q <= 0) continue;

            const int seq_len = seq_lens[s];
            if (seq_len <= 0) continue;
            const int* table = block_table_flat + block_table_offsets[s];

            // Row ids: slot of every position this sequence attends over.
            std::vector<int32_t> rows(seq_len);
            for (int p = 0; p < seq_len; p++)
            {
                const int slot = table[p / pool->block_size] * pool->block_size + (p % pool->block_size);
                if (slot < 0 || slot >= pool->num_slots)
                {
                    set_last_error("paged-kv-pool: block table produced an out-of-range slot.");
                    return 0;
                }
                rows[p] = slot;
            }

            PooledContextHandle context;
            if (!context.init(8 * 1024 * 1024))
            {
                set_last_error("paged-kv-pool: failed to acquire ggml context (attention).");
                return 0;
            }
            ggml_context* ctx = context.value;

            ggml_tensor* q_in    = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, num_heads, num_q);
            ggml_tensor* row_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len);
            ggml_tensor* mask    = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, seq_len, num_q, 1, 1);
            ggml_tensor* result  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, num_heads, num_q);

            // THE POINT: the history is gathered on device, from the pool.
            ggml_tensor* k_rows = ggml_get_rows(ctx, pool->k[layer], row_idx); // [kv_dim, seq_len]
            ggml_tensor* v_rows = ggml_get_rows(ctx, pool->v[layer], row_idx);

            ggml_tensor* k3 = ggml_reshape_3d(ctx, k_rows, head_dim, num_kv_heads, seq_len);
            ggml_tensor* v3 = ggml_reshape_3d(ctx, v_rows, head_dim, num_kv_heads, seq_len);

            ggml_tensor* q_attn = ggml_cont(ctx, ggml_permute(ctx, q_in, 0, 2, 1, 3));
            ggml_tensor* k_attn = ggml_cont(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3));
            ggml_tensor* v_attn = ggml_cont(ctx, ggml_permute(ctx, v3, 0, 2, 1, 3));

            ggml_tensor* attn = flash_attn_ext_guarded(ctx, "paged KV pool attention",
                q_attn, k_attn, v_attn, mask, scale, 0.0f, 0.0f);
            ggml_tensor* copy = ggml_cpy(ctx, attn, result);
            ggml_set_output(copy);

            ggml_cgraph* graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, copy);

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (!buf)
            {
                set_last_error("paged-kv-pool: attention allocation failed.");
                return 0;
            }

            // Causal (and optionally sliding-window) mask over this sequence's
            // own positions - each query sees keys up to its absolute position.
            std::vector<ggml_fp16_t> mask_host(static_cast<size_t>(seq_len) * num_q);
            const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t ninf = ggml_fp32_to_fp16(-INFINITY);
            for (int qi = 0; qi < num_q; qi++)
            {
                const int abs_pos = positions[q_start + qi];
                for (int kp = 0; kp < seq_len; kp++)
                {
                    bool visible = kp <= abs_pos;
                    if (visible && sliding_window > 0 && (abs_pos - kp) >= sliding_window)
                        visible = false;
                    mask_host[static_cast<size_t>(qi) * seq_len + kp] = visible ? zero : ninf;
                }
            }

            ggml_backend_tensor_set(q_in, q_data + static_cast<size_t>(q_start) * q_dim, 0,
                                    static_cast<size_t>(num_q) * q_dim * sizeof(float));
            ggml_backend_tensor_set(row_idx, rows.data(), 0, rows.size() * sizeof(int32_t));
            ggml_backend_tensor_set(mask, mask_host.data(), 0, mask_host.size() * sizeof(ggml_fp16_t));

            ggml_status st = graph_compute_profiled(g_backend, graph, "paged_kv_attention");
            if (st == GGML_STATUS_SUCCESS)
            {
                ggml_backend_tensor_get(result, out_data + static_cast<size_t>(q_start) * q_dim, 0,
                                        static_cast<size_t>(num_q) * q_dim * sizeof(float));
            }
            ggml_backend_buffer_free(buf);
            if (st != GGML_STATUS_SUCCESS)
            {
                set_last_error("paged-kv-pool: attention compute failed.");
                return 0;
            }
        }
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in TSGgml_PagedKvPoolAttention."); return 0; }
}
