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

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>

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

    // Round up `valid_len` to a bucket so that small changes in seq_len
    // (decode step grows it by 1 each token) hit the same cached session
    // instead of forcing a graph + backend-buffer rebuild every step.
    // Powers of two keep the bucket count tiny: for a 4K-token conversation
    // we see at most ~7 distinct buckets across the whole run, and SWA
    // models cap at the window size after the cache fills.
    inline int bucket_padded_kv_length(int padded_kv_len)
    {
        if (padded_kv_len <= 64) return 64;
        // Next power of two >= padded_kv_len.
        int v = padded_kv_len - 1;
        v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
        return v + 1;
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

    // ---------------------------------------------------------------------
    // Session cache for the paged-attention inner kernel.
    //
    // For a long decode run the per-call cost of ggml_backend_alloc_ctx_tensors
    // (a fresh Metal/CUDA buffer allocation) plus rebuilding the graph nodes
    // dominates for N=1, num_q=1 work — flash_attn_ext itself is fast on
    // those shapes. We amortise that overhead by keeping the ggml context,
    // tensors, graph, and the backend buffer alive across calls and reusing
    // them when the shape signature matches.
    //
    // Bucketing padded_kv_len by next-power-of-2 keeps the live cache size
    // small: a SWA model with window 512 settles into a single bucket once
    // the window fills; a 4K-token full-attention conversation visits ~7
    // buckets total. Scale is part of the key because flash_attn_ext bakes
    // it into the op's params at construction time.
    // ---------------------------------------------------------------------
    struct PagedAttnSession
    {
        bool valid = false;
        int num_q = 0;
        int padded_kv_len_bucket = 0;
        int num_heads = 0;
        int num_kv_heads = 0;
        int head_dim = 0;
        // Scale is encoded as a bit pattern so equality is exact, not float
        // tolerance — the cached graph baked one specific scale value.
        std::uint32_t scale_bits = 0;
        bool has_sinks = false;

        std::unique_ptr<unsigned char[]> ctx_mem;
        std::size_t ctx_mem_size = 0;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;

        ggml_tensor* q_in = nullptr;
        ggml_tensor* k_in = nullptr;
        ggml_tensor* v_in = nullptr;
        ggml_tensor* attn_mask = nullptr;     // always allocated; mask data set per call
        ggml_tensor* sinks_tensor = nullptr;  // only when has_sinks
        ggml_tensor* attn_result = nullptr;

        ggml_cgraph* graph = nullptr;

        std::uint64_t lru = 0;

        // Track whether we already zero-initialised the K/V padding region
        // for the highest seq_len we've seen in this session. We pad up to
        // padded_kv_len_bucket; once a call zeroes [seq_len .. bucket), any
        // later call with seq_len' >= seq_len doesn't need to re-zero
        // [seq_len' .. bucket) (those positions are already zero).
        int kv_zero_covered_from = 0;

        void destroy()
        {
            if (buffer != nullptr)
            {
                ggml_backend_buffer_free(buffer);
                buffer = nullptr;
            }
            if (ctx != nullptr)
            {
                ggml_free(ctx);
                ctx = nullptr;
            }
            ctx_mem.reset();
            ctx_mem_size = 0;
            q_in = k_in = v_in = attn_mask = sinks_tensor = attn_result = nullptr;
            graph = nullptr;
            valid = false;
            kv_zero_covered_from = 0;
        }

        ~PagedAttnSession() { destroy(); }

        PagedAttnSession() = default;
        PagedAttnSession(const PagedAttnSession&) = delete;
        PagedAttnSession& operator=(const PagedAttnSession&) = delete;
    };

    constexpr std::size_t kPagedAttnCacheSize = 16;
    thread_local std::array<PagedAttnSession, kPagedAttnCacheSize> g_paged_attn_cache;
    thread_local std::uint64_t g_paged_attn_lru_counter = 0;

    // Thread-local mask scratch so we don't allocate a fresh vector per call.
    thread_local std::vector<ggml_fp16_t> g_paged_attn_mask_scratch;
    // Thread-local zero buffer for padding K/V slots. Grown to max needed.
    thread_local std::vector<float> g_paged_attn_zero_scratch;

    inline std::uint32_t float_bits(float v)
    {
        std::uint32_t b;
        std::memcpy(&b, &v, sizeof(b));
        return b;
    }

    PagedAttnSession* find_paged_attn_session(
        int num_q, int padded_kv_len_bucket, int num_heads, int num_kv_heads,
        int head_dim, std::uint32_t scale_bits, bool has_sinks)
    {
        for (auto& sess : g_paged_attn_cache)
        {
            if (!sess.valid) continue;
            if (sess.num_q == num_q &&
                sess.padded_kv_len_bucket == padded_kv_len_bucket &&
                sess.num_heads == num_heads &&
                sess.num_kv_heads == num_kv_heads &&
                sess.head_dim == head_dim &&
                sess.scale_bits == scale_bits &&
                sess.has_sinks == has_sinks)
            {
                sess.lru = ++g_paged_attn_lru_counter;
                return &sess;
            }
        }
        return nullptr;
    }

    PagedAttnSession& acquire_paged_attn_session_slot()
    {
        // Prefer an invalid slot.
        for (auto& sess : g_paged_attn_cache)
            if (!sess.valid) { sess.lru = ++g_paged_attn_lru_counter; return sess; }
        // Otherwise evict the LRU entry.
        PagedAttnSession* victim = &g_paged_attn_cache[0];
        for (auto& sess : g_paged_attn_cache)
            if (sess.lru < victim->lru) victim = &sess;
        victim->destroy();
        victim->lru = ++g_paged_attn_lru_counter;
        return *victim;
    }

    bool build_paged_attn_session(
        PagedAttnSession& sess,
        int num_q, int padded_kv_len_bucket, int num_heads, int num_kv_heads,
        int head_dim, float scale, bool has_sinks)
    {
        // 1 MiB matches what PooledContextHandle uses; ggml graph metadata
        // for ~10 nodes fits comfortably.
        constexpr std::size_t kCtxMemSize = 1024 * 1024;
        sess.ctx_mem_size = kCtxMemSize;
        sess.ctx_mem.reset(new (std::nothrow) unsigned char[kCtxMemSize]);
        if (!sess.ctx_mem)
        {
            set_last_error("Failed to allocate paged-attention session memory.");
            return false;
        }

        ggml_init_params params = {};
        params.mem_size = kCtxMemSize;
        params.mem_buffer = sess.ctx_mem.get();
        params.no_alloc = true;
        sess.ctx = ggml_init(params);
        if (sess.ctx == nullptr)
        {
            set_last_error("Failed to ggml_init paged-attention session context.");
            sess.ctx_mem.reset();
            return false;
        }

        sess.q_in = ggml_new_tensor_3d(sess.ctx, GGML_TYPE_F32, head_dim, num_heads, num_q);
        sess.k_in = ggml_new_tensor_3d(sess.ctx, GGML_TYPE_F32, head_dim, num_kv_heads, padded_kv_len_bucket);
        sess.v_in = ggml_new_tensor_3d(sess.ctx, GGML_TYPE_F32, head_dim, num_kv_heads, padded_kv_len_bucket);
        sess.attn_result = ggml_new_tensor_3d(sess.ctx, GGML_TYPE_F32, head_dim, num_heads, num_q);
        // Always allocate a mask tensor. For shapes where the old code
        // skipped the mask, our cached graph will get a mask of all zeros,
        // which is the identity for flash_attn_ext softmax. Slightly more
        // work on a tiny number of corner cases is a fair price for keeping
        // a single cached graph per shape.
        sess.attn_mask = ggml_new_tensor_4d(sess.ctx, GGML_TYPE_F16, padded_kv_len_bucket, num_q, 1, 1);

        // Permute + cont for flash_attn_ext layout (see legacy comments).
        ggml_tensor* q_attn = ggml_cont(sess.ctx, ggml_permute(sess.ctx, sess.q_in, 0, 2, 1, 3));
        ggml_tensor* k_attn = ggml_cont(sess.ctx, ggml_permute(sess.ctx, sess.k_in, 0, 2, 1, 3));
        ggml_tensor* v_attn = ggml_cont(sess.ctx, ggml_permute(sess.ctx, sess.v_in, 0, 2, 1, 3));

        ggml_tensor* attn_out = ggml_flash_attn_ext(
            sess.ctx, q_attn, k_attn, v_attn, sess.attn_mask, scale, 0.0f, 0.0f);

        if (has_sinks)
        {
            sess.sinks_tensor = ggml_new_tensor_1d(sess.ctx, GGML_TYPE_F32, num_heads);
            ggml_flash_attn_ext_add_sinks(attn_out, sess.sinks_tensor);
        }

        ggml_tensor* result = ggml_cpy(sess.ctx, attn_out, sess.attn_result);
        ggml_set_output(result);

        sess.graph = ggml_new_graph(sess.ctx);
        ggml_build_forward_expand(sess.graph, result);

        sess.buffer = ggml_backend_alloc_ctx_tensors(sess.ctx, g_backend);
        if (sess.buffer == nullptr)
        {
            set_last_error("Failed to allocate backend buffer for paged-attention session.");
            ggml_free(sess.ctx);
            sess.ctx = nullptr;
            sess.ctx_mem.reset();
            return false;
        }

        sess.num_q = num_q;
        sess.padded_kv_len_bucket = padded_kv_len_bucket;
        sess.num_heads = num_heads;
        sess.num_kv_heads = num_kv_heads;
        sess.head_dim = head_dim;
        sess.scale_bits = float_bits(scale);
        sess.has_sinks = has_sinks;
        // K/V buffer is zero-initialised by ggml_backend_alloc_ctx_tensors,
        // so the entire padded range is already clean.
        sess.kv_zero_covered_from = 0;
        sess.valid = true;
        return true;
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

        const int padded_kv_len = paged_padded_kv_length(seq_len, head_dim);
        const int padded_kv_len_bucket = bucket_padded_kv_length(padded_kv_len);
        const bool has_sinks = (sinks_data != nullptr);
        const std::uint32_t scale_bits = float_bits(scale);

        PagedAttnSession* sess = find_paged_attn_session(
            num_q, padded_kv_len_bucket, num_heads, num_kv_heads, head_dim,
            scale_bits, has_sinks);
        if (sess == nullptr)
        {
            PagedAttnSession& slot = acquire_paged_attn_session_slot();
            if (!build_paged_attn_session(slot, num_q, padded_kv_len_bucket,
                                          num_heads, num_kv_heads, head_dim,
                                          scale, has_sinks))
                return 0;
            sess = &slot;
        }

        const std::size_t q_bytes = static_cast<std::size_t>(num_q) * num_heads * head_dim * sizeof(float);
        const std::size_t kv_bytes = static_cast<std::size_t>(seq_len) * num_kv_heads * head_dim * sizeof(float);

        // Drain any prior async GPU work so the upcoming tensor_set lands
        // after the previous step's reads of the same cached tensors. (For
        // multi-seq outer calls reusing the same cached session, the prior
        // seq's graph_compute may still be reading from sess->q_in / k_in /
        // v_in.) Cheap when no work is pending.
        host_read_barrier();

        ggml_backend_tensor_set(sess->q_in, q_data, 0, q_bytes);
        ggml_backend_tensor_set(sess->k_in, k_data, 0, kv_bytes);
        ggml_backend_tensor_set(sess->v_in, v_data, 0, kv_bytes);

        // Zero-fill the K/V padding region only for the positions we
        // haven't already covered in this session. The buffer started at
        // zero; each call writes the leading [0, seq_len) range, so
        // positions in [seq_len, kv_zero_covered_from) might be stale
        // garbage from a previous larger call. Zero them once and track.
        if (seq_len < sess->kv_zero_covered_from)
        {
            const int zero_from = seq_len;
            const int zero_to = sess->kv_zero_covered_from;
            const std::size_t zero_elems =
                static_cast<std::size_t>(zero_to - zero_from) * num_kv_heads * head_dim;
            if (g_paged_attn_zero_scratch.size() < zero_elems)
                g_paged_attn_zero_scratch.assign(zero_elems, 0.0f);
            const std::size_t pad_offset =
                static_cast<std::size_t>(zero_from) * num_kv_heads * head_dim * sizeof(float);
            ggml_backend_tensor_set(sess->k_in, g_paged_attn_zero_scratch.data(),
                                     pad_offset, zero_elems * sizeof(float));
            ggml_backend_tensor_set(sess->v_in, g_paged_attn_zero_scratch.data(),
                                     pad_offset, zero_elems * sizeof(float));
        }
        if (seq_len > sess->kv_zero_covered_from)
            sess->kv_zero_covered_from = seq_len;

        // Build and upload the mask. The cached graph always references the
        // mask tensor; we just refresh its content per call.
        fill_causal_mask_fp16(g_paged_attn_mask_scratch, padded_kv_len_bucket,
                              num_q, seq_len, first_q_pos, sliding_window);
        ggml_backend_tensor_set(sess->attn_mask, g_paged_attn_mask_scratch.data(),
                                 0, g_paged_attn_mask_scratch.size() * sizeof(ggml_fp16_t));

        if (has_sinks)
        {
            ggml_backend_tensor_set(sess->sinks_tensor, sinks_data, 0,
                                     static_cast<std::size_t>(num_heads) * sizeof(float));
        }

        ggml_status st = ggml_backend_graph_compute(g_backend, sess->graph);
        if (st != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml graph compute failed for paged attention.");
            return 0;
        }

        finalize_compute_with_download(
            sess->attn_result, out_data,
            static_cast<std::size_t>(num_q) * num_heads * head_dim * sizeof(float));
        return 1;
    }

    // ---------------------------------------------------------------------
    // Device-pointer variant.
    //
    // The host-array variant above is called from C# right after
    // `q.GetElementsAsFloat()`, which forces a `ggml_backend_synchronize`
    // (draining every queued projection/RoPE/scatter op) just to copy Q
    // into a managed array. That sync is the dominant cost for N=1 decode.
    //
    // The device variant accepts Q and OUT as raw IntPtr-like void* that
    // already live in a backend buffer (i.e., a tensor's storage on Metal /
    // CUDA). We:
    //   - zero-copy bind q_in to Q's existing backend buffer (no upload),
    //   - zero-copy bind attn_result to OUT's backend buffer (no download),
    //   - skip the redundant top-of-call host_read_barrier (queue ordering
    //     between the Q-projection compute and our flash-attn compute is
    //     guaranteed because both run on the same g_backend command queue).
    //
    // K and V are still passed as host scratch arrays (the caller gathers
    // from the paged block pool into a contiguous scratch); making those
    // zero-copy would require a separate native paged-storage refactor and
    // is a follow-up.
    //
    // We don't reuse the session cache here because that cache binds q_in /
    // attn_result to its own internal buffer at session-build time, while
    // this path needs them to alias the caller's buffer each call. Building
    // a fresh graph per call is cheap (CPU work + one ctx_alloc that only
    // covers k_in/v_in/mask).
    // ---------------------------------------------------------------------
    // CPU fast path for decode (num_q==1). Apple Silicon Metal kernel
    // dispatch latency (~100μs - 1ms) dominates over the actual compute
    // for one query token, and we have 42+ such dispatches per token on
    // Gemma 4. A tight scalar loop on the host pointer (which is in
    // unified memory and CPU-cache-resident anyway) avoids the dispatch
    // entirely. Mirrors the legacy per-seq path's AttentionDecodeCircular.
    int paged_attention_decode_cpu_n1(
        const float* q,             // [num_heads * head_dim] (one query token)
        const float* k_data,        // [seq_len, num_kv_heads, head_dim]
        const float* v_data,        // [seq_len, num_kv_heads, head_dim]
        float* out,                 // [num_heads * head_dim] (output)
        int seq_len,
        int first_q_pos,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        int sliding_window,
        float scale,
        const float* sinks_data)
    {
        const int group_size = num_heads / num_kv_heads;
        const int kv_stride = num_kv_heads * head_dim;

        // SWA window cutoff: the query attends to positions
        // [q_pos - sliding_window + 1, q_pos], capped at [0, seq_len-1].
        // Outside this range the K/V are masked out.
        int window_start = 0;
        if (sliding_window > 0)
        {
            const int q_pos = first_q_pos;
            window_start = q_pos - sliding_window + 1;
            if (window_start < 0) window_start = 0;
        }
        const int window_end = std::min(first_q_pos + 1, seq_len);

        thread_local std::vector<float> scores_scratch;
        if (static_cast<int>(scores_scratch.size()) < seq_len)
            scores_scratch.resize(seq_len);
        float* scores = scores_scratch.data();

        for (int h = 0; h < num_heads; ++h)
        {
            const float* q_head = q + h * head_dim;
            const int kv_head = h / group_size;
            const float* k_head = k_data + kv_head * head_dim;
            const float* v_head = v_data + kv_head * head_dim;

            float max_score = -std::numeric_limits<float>::infinity();
            for (int t = window_start; t < window_end; ++t)
            {
                const float* k_pos = k_head + t * kv_stride;
                float s = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                    s += q_head[d] * k_pos[d];
                s *= scale;
                scores[t] = s;
                if (s > max_score) max_score = s;
            }

            float sum_exp = 0.0f;
            if (sinks_data != nullptr)
            {
                // Sinks act as a virtual position at the very front of the
                // softmax with zero V contribution. Include it in the
                // normaliser only.
                const float sink_score = sinks_data[h];
                if (sink_score > max_score) max_score = sink_score;
                sum_exp = std::exp(sink_score - max_score);
            }
            for (int t = window_start; t < window_end; ++t)
            {
                const float e = std::exp(scores[t] - max_score);
                scores[t] = e;
                sum_exp += e;
            }
            const float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;

            float* out_head = out + h * head_dim;
            for (int d = 0; d < head_dim; ++d) out_head[d] = 0.0f;
            for (int t = window_start; t < window_end; ++t)
            {
                const float w = scores[t] * inv_sum;
                const float* v_pos = v_head + t * kv_stride;
                for (int d = 0; d < head_dim; ++d)
                    out_head[d] += w * v_pos[d];
            }
        }
        return 1;
    }

    int paged_attention_single_seq_impl_device(
        void* q_data_device,        // q's tensor storage (Metal-mapped) host ptr
        const float* k_data,        // host scratch (gathered)
        const float* v_data,        // host scratch (gathered)
        void* out_data_device,      // out tensor storage (Metal-mapped) host ptr
        int num_q,
        int seq_len,
        int first_q_pos,
        int num_heads,
        int num_kv_heads,
        int head_dim,
        int sliding_window,
        float scale,
        const float* sinks_data)
    {
        if (!ensure_backend()) return 0;

        // CPU fast path for num_q==1. Avoids per-layer Metal kernel
        // dispatch (~ms latency) that otherwise dominates over the small
        // actual compute for one query token. Mirrors the per-seq path's
        // AttentionDecodeCircular approach.
        if (num_q == 1)
        {
            // Drain pending GPU work so Q's host buffer is stable for
            // CPU read. Single sync per layer, matching the per-seq
            // path's per-layer GetFloatPtr/EnsureHostReadable cost.
            host_read_barrier();
            return paged_attention_decode_cpu_n1(
                static_cast<const float*>(q_data_device),
                k_data, v_data,
                static_cast<float*>(out_data_device),
                seq_len, first_q_pos,
                num_heads, num_kv_heads, head_dim,
                sliding_window, scale, sinks_data);
        }

        const int padded_kv_len = paged_padded_kv_length(seq_len, head_dim);
        const std::size_t q_bytes = static_cast<std::size_t>(num_q) * num_heads * head_dim * sizeof(float);
        const std::size_t out_bytes = q_bytes;
        const std::size_t kv_bytes = static_cast<std::size_t>(seq_len) * num_kv_heads * head_dim * sizeof(float);
        const bool has_sinks = (sinks_data != nullptr);

        PooledContextHandle context;
        if (!context.init(1024 * 1024))
        {
            set_last_error("Failed to acquire ggml context for paged attention (device).");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_tensor* q_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, num_heads, num_q);
        ggml_tensor* k_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, num_kv_heads, padded_kv_len);
        ggml_tensor* v_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, num_kv_heads, padded_kv_len);
        ggml_tensor* attn_result = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, num_heads, num_q);

        // Always allocate a mask tensor; mask-of-zeros is a no-op for
        // flash_attn_ext softmax, so this is correct for the cases where
        // the host-array path would have skipped the mask.
        ggml_tensor* attn_mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, padded_kv_len, num_q, 1, 1);

        ggml_tensor* q_attn = ggml_cont(ctx, ggml_permute(ctx, q_in, 0, 2, 1, 3));
        ggml_tensor* k_attn = ggml_cont(ctx, ggml_permute(ctx, k_in, 0, 2, 1, 3));
        ggml_tensor* v_attn = ggml_cont(ctx, ggml_permute(ctx, v_in, 0, 2, 1, 3));

        ggml_tensor* attn_out = ggml_flash_attn_ext(ctx, q_attn, k_attn, v_attn,
                                                   attn_mask, scale, 0.0f, 0.0f);

        ggml_tensor* sinks_tensor = nullptr;
        if (has_sinks)
        {
            sinks_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_heads);
            ggml_flash_attn_ext_add_sinks(attn_out, sinks_tensor);
        }

        ggml_tensor* result = ggml_cpy(ctx, attn_out, attn_result);
        ggml_set_output(result);

        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);

        // Try zero-copy bind for Q. The host pointer comes from a tensor
        // storage allocated by GgmlAllocator on Metal; that memory is
        // already a host-mapped MTLBuffer, so ggml_backend_dev_buffer_from_host_ptr
        // returns a thin wrapper. Falls through to upload if the device
        // refuses host-ptr buffers (e.g., on CUDA without UVA).
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);

        // Zero-copy bind for Q and OUT. We can't use try_get_host_ptr_buffer
        // here: it gates on prefers_device_local_cache, which (incorrectly)
        // says Metal isn't integrated because ggml's Metal backend reports
        // type=GPU and leaves props.integrated=0. Fixing that globally
        // regresses ops that rely on DeviceCopy semantics, so we skip the
        // gate locally — for paged attention we know host-ptr zero-copy
        // is the right thing on Metal (unified memory, no DMA needed).
        //
        // To avoid re-creating MTLBuffer wrappers on every call we keep a
        // small thread-local cache keyed on (host_ptr, bytes). Each entry
        // holds a ggml_backend_buffer_t we created via
        // ggml_backend_dev_buffer_from_host_ptr; subsequent calls with the
        // same pointer reuse it.
        struct PagedAttnHostPtrCacheEntry {
            void* host_ptr = nullptr;
            std::size_t bytes = 0;
            ggml_backend_buffer_t buffer = nullptr;
            std::uint64_t lru = 0;
        };
        constexpr std::size_t kHostPtrCacheSize = 16;
        thread_local std::array<PagedAttnHostPtrCacheEntry, kHostPtrCacheSize> g_host_ptr_cache;
        thread_local std::uint64_t g_host_ptr_lru = 0;

        auto get_or_make_host_ptr_buffer = [&](void* host_ptr, std::size_t bytes) -> ggml_backend_buffer_t {
            if (host_ptr == nullptr || bytes == 0 || dev == nullptr) return nullptr;
            const std::size_t alignment = get_host_ptr_alignment(g_backend, dev);
            if (!is_pointer_aligned(host_ptr, alignment)) return nullptr;
            if (!get_device_static_props(dev).buffer_from_host_ptr) return nullptr;

            // Lookup.
            for (auto& e : g_host_ptr_cache) {
                if (e.buffer != nullptr && e.host_ptr == host_ptr && e.bytes == bytes) {
                    e.lru = ++g_host_ptr_lru;
                    return e.buffer;
                }
            }
            // Pick victim (invalid slot, else LRU).
            PagedAttnHostPtrCacheEntry* victim = nullptr;
            for (auto& e : g_host_ptr_cache) {
                if (e.buffer == nullptr) { victim = &e; break; }
            }
            if (!victim) {
                victim = &g_host_ptr_cache[0];
                for (auto& e : g_host_ptr_cache)
                    if (e.lru < victim->lru) victim = &e;
                if (victim->buffer != nullptr) {
                    ggml_backend_buffer_free(victim->buffer);
                    victim->buffer = nullptr;
                }
            }
            ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(
                dev, host_ptr, bytes, bytes);
            if (buf == nullptr) return nullptr;
            victim->host_ptr = host_ptr;
            victim->bytes = bytes;
            victim->buffer = buf;
            victim->lru = ++g_host_ptr_lru;
            return buf;
        };

        auto try_bind_host_ptr = [&](void* host_ptr, std::size_t bytes,
                                     ggml_tensor* t) -> bool {
            ggml_backend_buffer_t buf = get_or_make_host_ptr_buffer(host_ptr, bytes);
            if (buf == nullptr) return false;
            return ggml_backend_tensor_alloc(buf, t, host_ptr) == GGML_STATUS_SUCCESS;
        };

        bool q_zero_copy = try_bind_host_ptr(q_data_device, q_bytes, q_in);
        bool out_zero_copy = try_bind_host_ptr(out_data_device, out_bytes, attn_result);

        // Allocate the remaining (non-zero-copy) tensors. ggml only
        // allocates the ones still unbound, so this just covers k_in, v_in,
        // attn_mask, sinks, and the intermediate graph nodes.
        BufferHandle ctx_buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (ctx_buffer.value == nullptr)
        {
            set_last_error("Failed to allocate backend buffer for paged attention (device).");
            return 0;
        }

        // Fall-back upload for Q if zero-copy didn't take. (Requires a sync
        // because we'd be CPU-memcpying from a possibly-still-being-written
        // GPU buffer; without zero-copy we're back to the slow path.)
        if (!q_zero_copy)
        {
            host_read_barrier();
            ggml_backend_tensor_set(q_in, q_data_device, 0, q_bytes);
        }

        // K, V are host scratch — already stable, no GPU race.
        ggml_backend_tensor_set(k_in, k_data, 0, kv_bytes);
        ggml_backend_tensor_set(v_in, v_data, 0, kv_bytes);

        if (padded_kv_len > seq_len)
        {
            thread_local std::vector<float> zeros_scratch;
            const std::size_t pad_elems =
                static_cast<std::size_t>(padded_kv_len - seq_len) * num_kv_heads * head_dim;
            if (zeros_scratch.size() < pad_elems)
                zeros_scratch.assign(pad_elems, 0.0f);
            const std::size_t pad_offset =
                static_cast<std::size_t>(seq_len) * num_kv_heads * head_dim * sizeof(float);
            ggml_backend_tensor_set(k_in, zeros_scratch.data(), pad_offset, pad_elems * sizeof(float));
            ggml_backend_tensor_set(v_in, zeros_scratch.data(), pad_offset, pad_elems * sizeof(float));
        }

        thread_local std::vector<ggml_fp16_t> mask_scratch;
        fill_causal_mask_fp16(mask_scratch, padded_kv_len, num_q, seq_len, first_q_pos, sliding_window);
        ggml_backend_tensor_set(attn_mask, mask_scratch.data(), 0,
                                 mask_scratch.size() * sizeof(ggml_fp16_t));

        if (has_sinks)
        {
            ggml_backend_tensor_set(sinks_tensor, sinks_data, 0,
                                     static_cast<std::size_t>(num_heads) * sizeof(float));
        }

        ggml_status st = ggml_backend_graph_compute(g_backend, graph);
        if (st != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml graph compute failed for paged attention (device).");
            return 0;
        }

        if (out_zero_copy)
        {
            // The compute wrote attn_result directly into the caller's
            // tensor buffer. The next GPU op the caller queues against
            // that tensor will serialise behind our compute on the backend
            // queue, so no CPU sync is required here. Just mark the queue
            // dirty so HostReadBarrier knows to drain if host code reads.
            mark_pending_gpu_work();
        }
        else
        {
            finalize_compute_with_download(attn_result, out_data_device, out_bytes);
        }

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

        // Thread-local reusable gather scratch. Sized for the largest
        // sequence we've ever seen on this thread; subsequent calls reuse
        // the same allocation. We don't zero-initialise on grow because
        // the gather memcpy below writes the entire used range before any
        // read.
        thread_local std::unique_ptr<float[]> k_scratch_buf;
        thread_local std::unique_ptr<float[]> v_scratch_buf;
        thread_local std::size_t scratch_capacity_elems = 0;
        const std::size_t needed_elems =
            static_cast<std::size_t>(max_seq_len) * per_token_kv_stride;
        if (scratch_capacity_elems < needed_elems)
        {
            k_scratch_buf.reset(new float[needed_elems]);
            v_scratch_buf.reset(new float[needed_elems]);
            scratch_capacity_elems = needed_elems;
        }
        float* k_scratch = k_scratch_buf.get();
        float* v_scratch = v_scratch_buf.get();

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
                std::memcpy(k_scratch + dst_offset, paged_k_data + src_offset, bytes);
                std::memcpy(v_scratch + dst_offset, paged_v_data + src_offset, bytes);
            }

            int ok = paged_attention_single_seq_impl(
                q_data + static_cast<std::size_t>(q_start) * per_token_q_stride,
                k_scratch,
                v_scratch,
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

        // Thread-local reusable gather scratch. Sized for the largest
        // sequence we've ever seen on this thread; subsequent calls reuse
        // the same allocation. We don't zero-initialise on grow because
        // the gather memcpy below writes the entire used range before any
        // read.
        thread_local std::unique_ptr<float[]> k_scratch_buf;
        thread_local std::unique_ptr<float[]> v_scratch_buf;
        thread_local std::size_t scratch_capacity_elems = 0;
        const std::size_t needed_elems =
            static_cast<std::size_t>(max_seq_len) * per_token_kv_stride;
        if (scratch_capacity_elems < needed_elems)
        {
            k_scratch_buf.reset(new float[needed_elems]);
            v_scratch_buf.reset(new float[needed_elems]);
            scratch_capacity_elems = needed_elems;
        }
        float* k_scratch = k_scratch_buf.get();
        float* v_scratch = v_scratch_buf.get();

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
                std::memcpy(k_scratch + dst_offset, paged_k_data + src_offset, bytes);
                std::memcpy(v_scratch + dst_offset, paged_v_data + src_offset, bytes);
            }

            int ok = paged_attention_single_seq_impl(
                q_data + static_cast<std::size_t>(q_start) * per_token_q_stride,
                k_scratch,
                v_scratch,
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

// ============================================================================
// TSGgml_PagedAttentionForwardDeviceWithSinks
//
// GPU-resident variant of TSGgml_PagedAttentionForward. q_data and out_data
// are pointers into existing backend-allocated buffers (e.g. C# Tensor
// storage on Metal — host-mapped MTLBuffers). This avoids the per-layer
// `q.GetElementsAsFloat()` -> ggml_backend_synchronize round-trip the
// caller would otherwise do, plus the matching `CreateFloatTensor(out)`
// upload on the way back. K/V scratch is still host-side because the paged
// K/V pool lives in C# managed arrays for now; making that zero-copy is a
// follow-up.
//
// Pass sinks_data = nullptr to degenerate to no-sinks (same shape as
// TSGgml_PagedAttentionForwardDevice).
// ============================================================================
TSG_EXPORT int TSGgml_PagedAttentionForwardDeviceWithSinks(
    void* q_data,                   // device pointer (backend buffer)
    const float* paged_k_data,      // host paged K/V (gathered per-seq below)
    const float* paged_v_data,
    void* out_data,                 // device pointer (backend buffer)
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
    const float* sinks_data)
{
    try
    {
        if (q_data == nullptr || paged_k_data == nullptr || paged_v_data == nullptr ||
            out_data == nullptr || query_start_loc == nullptr || seq_lens == nullptr ||
            block_table_flat == nullptr || block_table_offsets == nullptr)
        {
            set_last_error("Null pointer passed to paged attention forward (device).");
            return 0;
        }
        if (num_seqs <= 0 || num_tokens <= 0 || num_heads <= 0 || num_kv_heads <= 0 ||
            head_dim <= 0 || block_size <= 0)
        {
            set_last_error("Invalid dimensions for paged attention forward (device).");
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
            set_last_error("All seq_lens are zero in paged attention forward (device).");
            return 0;
        }

        thread_local std::unique_ptr<float[]> k_scratch_buf;
        thread_local std::unique_ptr<float[]> v_scratch_buf;
        thread_local std::size_t scratch_capacity_elems = 0;
        const std::size_t needed_elems =
            static_cast<std::size_t>(max_seq_len) * per_token_kv_stride;
        if (scratch_capacity_elems < needed_elems)
        {
            k_scratch_buf.reset(new float[needed_elems]);
            v_scratch_buf.reset(new float[needed_elems]);
            scratch_capacity_elems = needed_elems;
        }
        float* k_scratch = k_scratch_buf.get();
        float* v_scratch = v_scratch_buf.get();

        for (int s = 0; s < num_seqs; ++s)
        {
            const int q_start = query_start_loc[s];
            const int q_end = query_start_loc[s + 1];
            const int num_q = q_end - q_start;
            const int seq_len = seq_lens[s];
            if (num_q <= 0) continue;
            if (seq_len <= 0)
            {
                set_last_error("Encountered zero seq_len in paged attention forward (device).");
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
                std::memcpy(k_scratch + dst_offset, paged_k_data + src_offset, bytes);
                std::memcpy(v_scratch + dst_offset, paged_v_data + src_offset, bytes);
            }

            // Slice the device pointers to this seq's window. q_data and
            // out_data are laid out [num_tokens, num_heads * head_dim] row-
            // major, so seq s's slice starts at q_start tokens in.
            void* q_data_seq = static_cast<unsigned char*>(q_data) +
                static_cast<std::size_t>(q_start) * per_token_q_stride * sizeof(float);
            void* out_data_seq = static_cast<unsigned char*>(out_data) +
                static_cast<std::size_t>(q_start) * per_token_q_stride * sizeof(float);

            int ok = paged_attention_single_seq_impl_device(
                q_data_seq,
                k_scratch,
                v_scratch,
                out_data_seq,
                num_q, seq_len, first_q_pos,
                num_heads, num_kv_heads, head_dim, sliding_window, scale,
                sinks_data);
            if (!ok)
                return 0;
        }

        // NOTE: NOT calling host_read_barrier() at the end. The whole point
        // of the device path is to keep the work queued; the caller will
        // hit EnsureHostReadable() (or queue another GPU op that consumes
        // out_data, which serialises behind us automatically).
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
        set_last_error("Unknown error in TSGgml_PagedAttentionForwardDeviceWithSinks.");
        return 0;
    }
}

// Convenience wrapper without sinks. Forwards to the WithSinks variant
// with sinks_data=nullptr so the kernel skips the ggml_flash_attn_ext_add_sinks
// op and runs the standard softmax.
TSG_EXPORT int TSGgml_PagedAttentionForwardDevice(
    void* q_data,
    const float* paged_k_data,
    const float* paged_v_data,
    void* out_data,
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
    float scale)
{
    return TSGgml_PagedAttentionForwardDeviceWithSinks(
        q_data, paged_k_data, paged_v_data, out_data,
        query_start_loc, seq_lens, positions,
        block_table_flat, block_table_offsets,
        num_seqs, num_tokens, num_heads, num_kv_heads, head_dim,
        block_size, sliding_window, scale,
        /* sinks_data */ nullptr);
}
