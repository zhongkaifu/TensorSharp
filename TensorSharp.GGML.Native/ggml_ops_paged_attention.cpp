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
// Goes through the same end-to-end path as the single-sequence
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

    // Same predicate as ggml_ops_transformer_common.h's flash_attn_requires_masked_padding.
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
        ggml_backend_t backend = nullptr;
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
                test_note_cache_release();
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
            backend = nullptr;
            valid = false;
            kv_zero_covered_from = 0;
        }

        ~PagedAttnSession() { destroy(); }

        PagedAttnSession() = default;
        PagedAttnSession(const PagedAttnSession&) = delete;
        PagedAttnSession& operator=(const PagedAttnSession&) = delete;
    };

    constexpr std::size_t kPagedAttnCacheSize = 16;

    // One thread's sessions, owned by a process-wide registry rather than by a
    // thread_local (see ThreadCacheRegistry in ggml_ops_internal.h). Continuous
    // batching calls this from server and ThreadPool threads; one that exited
    // holding a session freed its CUDA buffer inside the DLL's thread detach,
    // and ggml aborted the process.
    struct PagedAttnThreadCache
    {
        std::array<PagedAttnSession, kPagedAttnCacheSize> sessions;
        std::uint64_t lru = 0;
    };
    using PagedAttnCaches = ThreadCacheRegistry<PagedAttnThreadCache>;

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
        PagedAttnThreadCache& cache,
        int num_q, int padded_kv_len_bucket, int num_heads, int num_kv_heads,
        int head_dim, std::uint32_t scale_bits, bool has_sinks)
    {
        for (auto& sess : cache.sessions)
        {
            // Equal packed shapes on different layer-placement ranks must
            // never reuse device buffers allocated by another backend.
            if (!sess.valid || sess.backend != g_backend) continue;
            if (sess.num_q == num_q &&
                sess.padded_kv_len_bucket == padded_kv_len_bucket &&
                sess.num_heads == num_heads &&
                sess.num_kv_heads == num_kv_heads &&
                sess.head_dim == head_dim &&
                sess.scale_bits == scale_bits &&
                sess.has_sinks == has_sinks)
            {
                sess.lru = ++cache.lru;
                return &sess;
            }
        }
        return nullptr;
    }

    PagedAttnSession& acquire_paged_attn_session_slot(PagedAttnThreadCache& cache)
    {
        // Prefer an invalid slot.
        for (auto& sess : cache.sessions)
            if (!sess.valid) { sess.lru = ++cache.lru; return sess; }
        // Otherwise evict the LRU entry. On Metal async the victim's last call can
        // still have its result download queued (finalize_compute_with_download),
        // and that blit reads the buffer destroy() frees: drain it first.
        PagedAttnSession* victim = &cache.sessions[0];
        for (auto& sess : cache.sessions)
            if (sess.lru < victim->lru) victim = &sess;
        host_read_barrier();
        victim->destroy();
        victim->lru = ++cache.lru;
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

        // Permute + cont for flash_attn_ext layout (see flash_attn_decode_impl).
        ggml_tensor* q_attn = ggml_cont(sess.ctx, ggml_permute(sess.ctx, sess.q_in, 0, 2, 1, 3));
        ggml_tensor* k_attn = ggml_cont(sess.ctx, ggml_permute(sess.ctx, sess.k_in, 0, 2, 1, 3));
        ggml_tensor* v_attn = ggml_cont(sess.ctx, ggml_permute(sess.ctx, sess.v_in, 0, 2, 1, 3));

        if (has_sinks)
            sess.sinks_tensor = ggml_new_tensor_1d(sess.ctx, GGML_TYPE_F32, num_heads);
        ggml_tensor* attn_out = flash_attn_ext_guarded(sess.ctx, "paged attention",
            q_attn, k_attn, v_attn, sess.attn_mask, scale, 0.0f, 0.0f, sess.sinks_tensor);

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

        // The kv_zero_covered_from tracking below relies on the K/V padding
        // region [seq_len, bucket) starting out zero, but backend buffers are
        // NOT zero-initialised (CPU is posix_memalign, CUDA is cudaMalloc, and
        // both hand back memory a freed buffer just used). The mask only puts
        // -inf on the padded keys; CUDA flash attention still forms q.k for
        // them, and NaN + -inf = NaN poisons the whole softmax row. Nemotron
        // 3.5 hit this after a long prompt's buffers were freed: the next
        // batched prefill's first attention layer returned NaN for every row
        // and the MoE router then had no finite logits. Clear once per build.
        ggml_backend_buffer_clear(sess.buffer, 0);

        sess.num_q = num_q;
        sess.padded_kv_len_bucket = padded_kv_len_bucket;
        sess.num_heads = num_heads;
        sess.num_kv_heads = num_kv_heads;
        sess.head_dim = head_dim;
        sess.scale_bits = float_bits(scale);
        sess.has_sinks = has_sinks;
        sess.backend = g_backend;
        // The buffer was explicitly cleared above, so the entire padded
        // range is already clean.
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

        PagedAttnCaches& caches = PagedAttnCaches::instance();
        caches.free_retired();
        PagedAttnCaches::Slot& owned = caches.this_thread();
        std::lock_guard<std::mutex> cache_lock(owned.mutex);
        PagedAttnThreadCache& cache = owned.cache;
        PagedAttnSession* sess = find_paged_attn_session(
            cache, num_q, padded_kv_len_bucket, num_heads, num_kv_heads, head_dim,
            scale_bits, has_sinks);
        if (sess == nullptr)
        {
            PagedAttnSession& slot = acquire_paged_attn_session_slot(cache);
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

        ggml_status st = tsg::compute_graph(g_backend, sess->graph);
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

        // Only the union of the queries' visible windows is needed. Retain
        // earlier KV rows in the page pool for prefix reuse, but do not gather
        // or upload them on sliding-attention layers.
        int max_seq_len = 0;
        for (int s = 0; s < num_seqs; ++s)
        {
            const int q_start = query_start_loc[s];
            const int num_q = query_start_loc[s + 1] - q_start;
            if (num_q <= 0) continue;
            const int first_q_pos = positions != nullptr ? positions[q_start] : seq_lens[s] - num_q;
            const int first_key = sliding_window > 0 ? std::max(0, first_q_pos - sliding_window + 1) : 0;
            max_seq_len = std::max(max_seq_len, seq_lens[s] - first_key);
        }
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
            const int first_key = sliding_window > 0 ? std::max(0, first_q_pos - sliding_window + 1) : 0;
            const int live_len = seq_len - first_key;
            if (live_len <= 0)
            {
                set_last_error("No visible keys for paged attention queries.");
                return 0;
            }

            // The first visible key may lie inside a page, and physical page
            // IDs need not be consecutive. Copy each intersecting span into a
            // zero-based live buffer, then shift the mask positions to match.
            const int* table = block_table_flat + block_table_offsets[s];
            const std::size_t per_token_kv_bytes =
                static_cast<std::size_t>(per_token_kv_stride) * sizeof(float);
            for (int key = first_key; key < seq_len;)
            {
                const int block_id = table[key / block_size];
                const int offset_in_block = key % block_size;
                const int tokens_in_block = std::min(block_size - offset_in_block, seq_len - key);
                const std::size_t src_offset =
                    (static_cast<std::size_t>(block_id) * block_size + offset_in_block) * per_token_kv_stride;
                const std::size_t dst_offset =
                    static_cast<std::size_t>(key - first_key) * per_token_kv_stride;
                const std::size_t bytes =
                    static_cast<std::size_t>(tokens_in_block) * per_token_kv_bytes;
                std::memcpy(k_scratch + dst_offset, paged_k_data + src_offset, bytes);
                std::memcpy(v_scratch + dst_offset, paged_v_data + src_offset, bytes);
                key += tokens_in_block;
            }

            int ok = paged_attention_single_seq_impl(
                q_data + static_cast<std::size_t>(q_start) * per_token_q_stride,
                k_scratch,
                v_scratch,
                out_data + static_cast<std::size_t>(q_start) * per_token_q_stride,
                num_q, live_len, first_q_pos - first_key,
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

#if defined(TSG_GGML_TEST_DIAGNOSTICS)
// Diagnostic only: counts the retained sessions of every thread, live or exited,
// with no backend lookup or mutation. Used to prove that backend teardown
// released them.
TSG_EXPORT int TSGgml_PagedAttentionSessionCount()
{
    int count = 0;
    PagedAttnCaches::instance().for_each([&](const PagedAttnThreadCache& cache) {
        for (const auto& session : cache.sessions)
            if (session.valid && session.buffer != nullptr) ++count;
    });
    return count;
}
#endif

namespace tsg
{
    // Frees every thread's cached paged-attention sessions, live and retired.
    // Called from TSGgml_ClearHostBufferCache and TSGgml_Shutdown so their backend
    // buffers are released while the backend and the driver are alive. A thread
    // inside a call finishes it first; its next call builds the session again.
    void free_paged_attn_sessions()
    {
        PagedAttnCaches::instance().release_all([](PagedAttnThreadCache& cache) {
            for (auto& session : cache.sessions) session.destroy();
        });
    }
}
