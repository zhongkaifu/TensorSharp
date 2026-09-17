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
#include <cmath>
#include <cstdio>
#include <thread>

using namespace tsg;

// ============================================================================
// Muse-Glimmer whole-model forward: every transformer layer in ONE ggml graph.
//
// One entry point serves both shapes:
//   n_tokens == 1  (decode)  -> a PERSISTENT graph with stable tensor addresses,
//                               so ggml-cuda's CUDA-graph capture engages and a
//                               token is one replay instead of ~600 dispatches.
//   n_tokens >  1  (prefill) -> a transient graph, still one submission instead
//                               of ~600 * n_tokens per-op round trips.
//
// Muse-Glimmer differences from the Gemma 4 kernel this is modelled on:
//   * an attention OUTPUT GATE: attn * sigmoid(W_gate @ attn_norm(x)) before o_proj;
//   * RoPE is ggml NORM (interleaved pairs, mode 0), not NeoX, and runs ONLY on
//     the sliding-window layers - the full-attention layers are NoPE;
//   * post-attention / post-FFN norms use their own epsilon (1e-8), distinct
//     from the pre-norms' f_norm_rms_eps;
//   * SwiGLU (SiLU) rather than GeGLU;
//   * the folded output applies logit_scale BEFORE the tanh softcap;
//   * the KV cache is flat (non-circular), so a sliding-window layer just reads
//     [0, totalSeqLen) and masks everything outside its window.
//
// Optional: `capture_*` copies the residual entering selected layers into a host
// buffer, which is what the DFlash drafter's encoder consumes. Without it DFlash
// would be stuck on the slow per-op path purely to observe those residuals.
// ============================================================================

namespace
{
    // Pad the attention window to this stride so the decode graph's topology (and
    // therefore its CUDA-graph capture) only changes once every N tokens.
    constexpr int kMgPersistKvStride = 256;

    // Persist (and on CUDA, capture) graphs up to this many rows. Decode is 1
    // row; a DFlash verify batch is block_size rows. Without this the verify
    // graph is rebuilt every speculative step, which costs more than the target
    // forwards speculation saves - measured 16.4 tok/s with rebuilds against
    // 19.7 for plain decode, i.e. speculation was a net loss purely from graph
    // construction. The cache is keyed by n_tokens so each shape gets its own.
    constexpr int kMgMaxPersistTokens = 64;

    struct MgDecodeCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* hidden_in = nullptr;
        ggml_tensor* hidden_out = nullptr;
        ggml_tensor* pos_tensor = nullptr;
        ggml_tensor* kv_index = nullptr;                 // shared I64 write row
        ggml_tensor* kv_index_swa = nullptr;             // ring write row, SWA layers
        int swa_rows = 0;                                // ring size baked into the graph
        std::vector<ggml_tensor*> attn_mask;             // per-layer alias into class_mask
        ggml_tensor* class_mask[2] = { nullptr, nullptr }; // [0] full, [1] sliding-window
        // Host staging for the two masks, kept HERE rather than on the stack of
        // every replay: at 128K the full-class mask is 250 KB, so a per-call
        // std::vector was a 250 KB malloc/fill/free on the critical path of every
        // decoded token. `mask_pos` is the start_pos the buffer currently
        // describes (-1 = not built yet), which is what lets a 1-row decode
        // extend the causal mask by the single element that changed instead of
        // regenerating O(window) entries.
        std::vector<ggml_fp16_t> mask_host[2];
        int mask_pos[2] = { -1, -1 };
        std::vector<ggml_tensor*> capture_out;           // per captured layer
        int window_swa = 0;                              // padded span length, SWA layers
        int swa_start = 0;                               // padded span start, SWA layers
        int window_full = 0;                             // padded span length, full layers
        const void* sig_disc = nullptr;                  // model-instance discriminator
        const void* sig_kcache0 = nullptr;               // KV holder discriminator
        int num_layers = 0;
        int hidden_size = 0;
        bool folded = false;
        int out_count = 0;
        int capture_count = 0;
        int n_tokens = 0;
        bool embed_in_graph = false;
        bool fold_all = false;
        // Tensor-parallel execution plan for this rank: the SAME graph, cut at the
        // two row-parallel projections per layer (o_proj and the FFN down) whose
        // outputs are this rank's partial sums. `tp_planned` is part of the entry's
        // identity, not just a validity flag: a graph built to be executed here and
        // one built to be handed to the TP driver differ in what the caller does
        // with them, so a mode switch must rebuild rather than replay.
        tsg::TpRankPlan tp_plan;
        bool tp_planned = false;

        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            hidden_in = hidden_out = pos_tensor = kv_index = kv_index_swa = nullptr;
            attn_mask.clear(); capture_out.clear();
            class_mask[0] = class_mask[1] = nullptr;
            mask_host[0].clear(); mask_host[1].clear();
            mask_pos[0] = mask_pos[1] = -1;
            window_swa = window_full = swa_start = swa_rows = 0;
            sig_disc = sig_kcache0 = nullptr;
            num_layers = hidden_size = n_tokens = 0;
            folded = false; out_count = 0; capture_count = 0; embed_in_graph = false; fold_all = false;
            tp_plan.clear(); tp_planned = false;
        }
    };

    // A small pool keyed by (model instance, KV holder) so concurrent requests
    // each keep their own captured graph instead of busting a single shared entry
    // on every switch. Same rationale as g_g4dc_pool in ggml_ops_gemma4_decode.cpp.
    constexpr int kMgMaxDecodeCaches = 8;
    struct MgDecodeCachePool
    {
        MgDecodeCache entries[kMgMaxDecodeCaches];
        std::uint64_t used[kMgMaxDecodeCaches] = {};
        std::uint64_t clock = 0;

        // n_tokens is part of the IDENTITY, not just a validity check: a
        // speculative run alternates between 1-row decode and k-row verify batches
        // whose k varies with how much the drafter proposed. Keyed on (sig, kc0)
        // alone those shapes evict each other out of one slot and every step pays a
        // full graph rebuild - which is what made speculation a net loss.
        MgDecodeCache* find(const void* sig, const void* kc0, int n_tokens)
        {
            for (int i = 0; i < kMgMaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0
                    && entries[i].n_tokens == n_tokens)
                { used[i] = ++clock; return &entries[i]; }
            return nullptr;
        }

        MgDecodeCache& claim(const void* sig, const void* kc0, int n_tokens)
        {
            for (int i = 0; i < kMgMaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0
                    && entries[i].n_tokens == n_tokens)
                { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            for (int i = 0; i < kMgMaxDecodeCaches; i++)
                if (!entries[i].valid) { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            int lru = 0;
            for (int i = 1; i < kMgMaxDecodeCaches; i++) if (used[i] < used[lru]) lru = i;
            entries[lru].reset(); used[lru] = ++clock; return entries[lru];
        }

        void drop(const void* sig, const void* kc0, int n_tokens)
        {
            for (int i = 0; i < kMgMaxDecodeCaches; i++)
                if (entries[i].valid && entries[i].sig_disc == sig && entries[i].sig_kcache0 == kc0
                    && entries[i].n_tokens == n_tokens)
                    entries[i].reset();
        }

        void reset_all() { for (auto& e : entries) e.reset(); }
    };

    MgDecodeCachePool g_mg_pools[tsg::TSG_MAX_DEVICES];
    inline MgDecodeCachePool& mg_pool() { return g_mg_pools[tsg::g_active_rank]; }

    // Resources a TRANSIENT (prefill) tensor-parallel graph borrows between
    // "build" and "execute". The persistent decode graph already outlives its
    // call - it is the whole point of the cache above - but the prefill path
    // allocates from the context pool and the shared gallocr, so it has nothing
    // that naturally survives the return. Under TP the driver runs the graph only
    // after EVERY rank has built one, so each rank parks its context (and any
    // per-call buffer) here and recycles the slot when it next builds. Mirrors
    // G4VerifyTpPending in ggml_ops_gemma4_verify.cpp.
    struct MgTpPending
    {
        PooledContextHandle context;
        BufferHandle buffer{ nullptr };
        std::vector<BufferHandle> ephemeral;
        tsg::TpRankPlan plan;

        void reset()
        {
            plan.clear();
            ephemeral.clear();
            buffer = BufferHandle(nullptr);
            context = PooledContextHandle();
        }
    };

    // Deliberately leaked: freeing a parked context or backend buffer from a
    // static destructor runs after the CUDA driver has torn down and aborts the
    // process ("CUDA error: driver shutting down"). The supported release path is
    // TSGgml_MuseGlimmerReleaseTpGraphs, which the C# side calls on dispose and
    // on KV reset while the backends are still alive.
    MgTpPending* const g_mg_tp = new MgTpPending[tsg::TSG_MAX_DEVICES];

    // Point one rank's plan at the caller's host buffers. Only RANK 0 downloads:
    // every rank ends the graph with the same replicated hidden state (the last
    // AllReduce made it so) and they all share one host buffer, and the LM head is
    // folded on rank 0 alone - which is also the only rank the caller hands a
    // logits buffer. Called on every build AND on every replay, because the host
    // pointers are per-call while the graph is not.
    void mg_tp_bind_outputs(tsg::TpRankPlan& plan,
                            ggml_tensor* out_tensor, void* out_host, std::size_t out_bytes,
                            const std::vector<ggml_tensor*>& capture_out, float* capture_data,
                            int hidden_size, int n_tokens)
    {
        plan.out_tensor = nullptr;
        plan.out_host = nullptr;
        plan.out_bytes = 0;
        plan.extra_out.clear();
        if (tsg::g_active_rank != 0)
            return;
        if (out_tensor != nullptr && out_host != nullptr && out_bytes != 0)
        {
            plan.out_tensor = out_tensor;
            plan.out_host = out_host;
            plan.out_bytes = out_bytes;
        }
        // DFlash residual capture: extra downloads the driver performs alongside
        // the result, after every rank has synchronized.
        if (capture_data == nullptr)
            return;
        const std::size_t block = static_cast<std::size_t>(hidden_size) * n_tokens;
        for (std::size_t c = 0; c < capture_out.size(); c++)
            if (capture_out[c] != nullptr)
                plan.extra_out.push_back({ capture_out[c], capture_data + c * block, block * sizeof(float) });
    }

    struct MgLayerTensors
    {
        ggml_tensor* attn_norm_w = nullptr;
        ggml_tensor* q_w = nullptr;
        ggml_tensor* k_w = nullptr;
        ggml_tensor* v_w = nullptr;
        ggml_tensor* gate_w = nullptr;
        ggml_tensor* q_norm_w = nullptr;
        ggml_tensor* k_norm_w = nullptr;
        ggml_tensor* o_w = nullptr;
        ggml_tensor* post_attn_norm_w = nullptr;
        ggml_tensor* ffn_norm_w = nullptr;
        ggml_tensor* gu_w = nullptr;
        ggml_tensor* down_w = nullptr;
        ggml_tensor* post_ffn_norm_w = nullptr;
        ggml_tensor* k_cached_t = nullptr;
        ggml_tensor* v_cached_t = nullptr;
        ggml_tensor* k_cpy = nullptr;
        ggml_tensor* v_cpy = nullptr;
    };

    // Causal (+ sliding-window) mask for a chunk of `n_tokens` queries whose first
    // query sits at absolute position `start_pos`, over KV rows [0, mask_kv_len).
    // Matches llama.cpp's STANDARD SWA predicate: key p0 is visible to query p1 iff
    // p1 - p0 < window (and p0 <= p1). window <= 0 means full causal.
    // `span_start` is the absolute position of column 0 of the mask, i.e. the flat
    // KV-cache row the attention view begins at. Columns are span_start + ki, so
    // everything outside a query's causal + sliding window -- including the stride
    // padding at either end of the span -- stays -inf.
    // Runs fn(q0, q1) over [0, n_tokens), fanned out across threads when the
    // mask is large. A 2048-row prefill chunk at 64K context is a 256 MB fill;
    // single-threaded that is tens of milliseconds per chunk on the backends
    // without a device-side fill kernel (Metal, CPU). Decode shapes (1-64 rows,
    // small rings) stay on the calling thread.
    template <typename F>
    void mg_parallel_rows(int n_tokens, std::size_t row_len, F&& fn)
    {
        const std::size_t total = row_len * static_cast<std::size_t>(n_tokens);
        const unsigned hw = std::thread::hardware_concurrency();
        int nthreads = (total >= (std::size_t(1) << 21) && hw > 1)
            ? static_cast<int>(std::min(hw, 8u)) : 1;
        nthreads = std::min(nthreads, n_tokens);
        if (nthreads <= 1)
        {
            fn(0, n_tokens);
            return;
        }
        const int per = (n_tokens + nthreads - 1) / nthreads;
        std::vector<std::thread> workers;
        workers.reserve(static_cast<std::size_t>(nthreads) - 1);
        for (int t = 1; t < nthreads; t++)
        {
            const int q0 = t * per;
            const int q1 = std::min(n_tokens, q0 + per);
            if (q0 >= q1) break;
            workers.emplace_back([&fn, q0, q1] { fn(q0, q1); });
        }
        fn(0, std::min(per, n_tokens));
        for (auto& w : workers) w.join();
    }

    void fill_mg_mask(std::vector<ggml_fp16_t>& data, int mask_kv_len, int n_tokens,
                      int start_pos, int valid_kv_len, int window, int span_start)
    {
        data.resize(static_cast<std::size_t>(mask_kv_len) * n_tokens);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const ggml_fp16_t zero_val = static_cast<ggml_fp16_t>(0);
        ggml_fp16_t* base = data.data();
        mg_parallel_rows(n_tokens, static_cast<std::size_t>(mask_kv_len),
            [=](int q_first, int q_last)
        {
            for (int qi = q_first; qi < q_last; qi++)
            {
                const int q_pos = start_pos + qi;
                const int hi = std::min(q_pos, valid_kv_len - 1);                   // causal
                const int lo = (window > 0) ? std::max(0, q_pos - window + 1) : 0;  // sliding window
                ggml_fp16_t* row = base + static_cast<std::size_t>(qi) * mask_kv_len;
                // Visible columns are the single contiguous span
                // [max(lo, span_start) - span_start, hi - span_start], clamped to
                // the mask; everything else is -inf. Three block fills replace the
                // per-element loop of the original.
                int k_lo = std::max(lo - span_start, 0);
                int k_hi = std::min(hi - span_start, mask_kv_len - 1);
                if (k_hi < k_lo)
                {
                    std::fill(row, row + mask_kv_len, neg_inf);
                    continue;
                }
                std::fill(row, row + k_lo, neg_inf);
                std::fill(row + k_lo, row + k_hi + 1, zero_val);
                std::fill(row + k_hi + 1, row + mask_kv_len, neg_inf);
            }
        });
    }

    // Mask for a RING cache: slot (pos % ring_rows) holds position pos. The caller
    // guarantees ring_rows >= window + n_tokens, so the positions one query can see
    // never alias onto the same slot, and every slot this query must NOT see (stale
    // rows from an earlier chunk, or positions outside the window) simply stays at
    // -inf. That is what lets the SWA layers keep a small cache while the graph shape
    // stays FIXED at ring_rows - constant shape is also what keeps the CUDA capture.
    void fill_mg_ring_mask(std::vector<ggml_fp16_t>& data, int ring_rows, int n_tokens,
                           int start_pos, int window)
    {
        data.resize(static_cast<std::size_t>(ring_rows) * n_tokens);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const ggml_fp16_t zero_val = static_cast<ggml_fp16_t>(0);
        ggml_fp16_t* base = data.data();
        mg_parallel_rows(n_tokens, static_cast<std::size_t>(ring_rows),
            [=](int q_first, int q_last)
        {
            for (int qi = q_first; qi < q_last; qi++)
            {
                const int q_pos = start_pos + qi;
                const int lo = (window > 0) ? std::max(0, q_pos - window + 1) : 0;
                ggml_fp16_t* row = base + static_cast<std::size_t>(qi) * ring_rows;
                std::fill(row, row + ring_rows, neg_inf);
                // The visible positions [lo, q_pos] map to at most two contiguous
                // slot spans (one wrap): [lo % rows, ...) then [0, remainder).
                // The caller guarantees rows >= window + n_tokens so live
                // positions never alias a slot; clamp defensively anyway.
                const int len = std::min(q_pos - lo + 1, ring_rows);
                const int s = lo % ring_rows;
                const int first = std::min(len, ring_rows - s);
                std::fill(row + s, row + s + first, zero_val);
                std::fill(row, row + (len - first), zero_val);
            }
        });
    }
}

TSG_EXPORT int TSGgml_MuseGlimmerModelForward(
    // [hidden_size, n_tokens] host buffer, already embedded and input-normed.
    float* hidden_data, int hidden_size, int n_tokens, int num_layers,
    // Per-layer weights (arrays of size num_layers)
    void** attn_norm_arr,
    void** q_arr, void** k_arr, void** v_arr, void** gate_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr,
    void** post_attn_norm_arr,
    void** ffn_norm_arr,
    void** gu_arr, void** down_arr,
    void** post_ffn_norm_arr,
    // Per-layer KV caches, each [kv_heads, cache_size, head_dim]
    void** k_cache_arr, void** v_cache_arr,
    // Per-layer flags: non-zero = sliding-window layer (also the layers that RoPE)
    int* is_swa_arr,
    // Per-layer weight shapes
    int* q_type_arr, std::int64_t* q_ne0_arr, std::int64_t* q_ne1_arr, std::int64_t* q_bytes_arr,
    int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    int* gate_type_arr, std::int64_t* gate_ne0_arr, std::int64_t* gate_ne1_arr, std::int64_t* gate_bytes_arr,
    int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    int* gu_type_arr, std::int64_t* gu_ne0_arr, std::int64_t* gu_ne1_arr, std::int64_t* gu_bytes_arr,
    int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    // Global params
    int num_heads, int num_kv_heads, int head_dim, int cache_size,
    // Rows in each sliding-window layer's cache when they use a small ring
    // (indexed by position % swa_cache_size); 0 = every layer is cache_size.
    int swa_cache_size,
    int start_pos, int sliding_window,
    float eps, float post_norm_eps, float rope_base, float rope_freq_scale,
    float kq_scale, int kv_cache_type,
    // Folded output head (nullable): final RMSNorm -> lm_head -> logit_scale ->
    // tanh softcap, evaluated on the LAST row only.
    void* logits_data, int vocab_size,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data, float logit_scale, float logit_softcap,
    // Optional per-layer residual capture for the DFlash drafter's encoder:
    // capture_data receives capture_count blocks of [hidden_size, n_tokens], one
    // per entry of capture_layers, holding the residual ENTERING that layer.
    float* capture_data, int* capture_layers, int capture_count,
    // Optional in-graph input stage. When the quantized token-embedding table and
    // the token ids are supplied, the graph starts with get_rows + the weightless
    // input RMSNorm and `hidden_data` is an OUTPUT-only buffer. That removes the
    // two host-visible dispatches (embedding gather, input norm) the caller would
    // otherwise run per token, and makes the token id the only per-token input -
    // which is what keeps the captured decode graph replayable.
    // Left null for multimodal steps, where the caller has already spliced
    // projected image embeddings into `hidden_data`.
    const void* tok_embd_data, int tok_embd_type,
    std::int64_t tok_embd_ne0, std::int64_t tok_embd_ne1, std::int64_t tok_embd_bytes,
    const int* token_ids,
    // Non-zero: run the folded output head on EVERY row, not just the last,
    // writing [vocab, n_tokens] to logits_data. This is what a speculative
    // verify batch needs - it scores every drafted position in one pass.
    int all_logits_rows,
    // Tensor parallelism - same contract as TSGgml_Gemma4ModelVerify / ...Decode.
    // With tp_degree > 1 the caller drives one rank at a time (tsg::g_active_rank,
    // set through TSGgml_SetActiveDevice): every weight/KV pointer above is THIS
    // rank's shard, num_heads / num_kv_heads are already this rank's head counts,
    // and the call BUILDS AND BINDS the graph instead of running it - handing back
    // a plan (tp_plan_out[0]) cut at the two row-parallel projections per layer.
    // The caller collects one plan per rank and runs them together through
    // TSGgml_TensorParallelExecutePlans, so the GPUs overlap and the partial sums
    // are reduced in VRAM. tp_degree <= 1 (or a null tp_plan_out) is the ordinary
    // single-device path and ignores both parameters.
    int tp_degree, void** tp_plan_out)
{
    try
    {
        if (!ensure_backend())
            return 0;

        // Plan mode needs BOTH a place to put the plan and a group that actually
        // spans ranks. The degree alone is not enough: one local rank per node is
        // a legitimate distributed shape. Nor is the pointer alone - this entry is
        // marshalled as `out IntPtr`, so C# hands it a non-null slot on EVERY call,
        // including a plain single-GPU decode. Gating on the pointer by itself put
        // that decode into plan mode: the graph was built, never executed, and the
        // caller read an untouched logits buffer - fluent-looking <pad> spam at
        // thousands of tokens/sec because no compute had happened at all.
        const bool tp_mode = tp_plan_out != nullptr &&
            (tp_degree > 1 || tsg::tp_global_degree(tp_degree) > 1);
        if (tp_mode)
        {
            tp_plan_out[0] = nullptr;
            // Recycle this rank's previously parked transient graph: the driver
            // has executed it by the time the caller asks for another.
            g_mg_tp[tsg::g_active_rank].reset();
        }

        if (n_tokens <= 0 || num_layers <= 0 || hidden_size <= 0)
        {
            set_last_error("Muse-Glimmer forward: invalid arguments.");
            return 0;
        }
        // hidden_data may be null only when the graph produces the input itself
        // (token ids + table) AND nothing is written back to it - either because
        // the head is folded here, or because this is a tensor-parallel rank that
        // does not fold and therefore downloads nothing (only rank 0 does).
        {
            const bool in_graph_input = tok_embd_data != nullptr && token_ids != nullptr;
            const bool no_hidden_download = logits_data != nullptr || (tp_mode && tsg::g_active_rank != 0);
            if (hidden_data == nullptr && !(in_graph_input && no_hidden_download))
            {
                set_last_error("Muse-Glimmer forward: hidden_data is required unless the input and output stages are both in-graph.");
                return 0;
            }
        }

        const int total_seq_len = start_pos + n_tokens;
        if (total_seq_len > cache_size)
        {
            set_last_error("Muse-Glimmer forward: sequence exceeds the KV cache capacity.");
            return 0;
        }

        const int q_dim = num_heads * head_dim;
        const int kv_dim = num_kv_heads * head_dim;
        const bool fold = logits_data != nullptr && lm_head_data != nullptr &&
                          final_norm_data != nullptr && vocab_size > 0;
        const int cap_count = (capture_data != nullptr && capture_layers != nullptr && capture_count > 0)
            ? capture_count : 0;
        const bool embed_in_graph = tok_embd_data != nullptr && token_ids != nullptr && tok_embd_bytes > 0;

        // Attention spans. The KV cache is FLAT (row i holds absolute position i).
        //
        // A FULL layer reads [0, total). A SLIDING-WINDOW layer only ever needs keys
        // in (p - n_swa, p] for its queries, so it reads a MOVING span - which matters
        // enormously at long context: at 64K a 2048-window layer reading [0, total)
        // would do 32x the necessary attention work, and 39 of the 52 layers are
        // sliding-window.
        //
        // Both the span start and its length snap to kMgPersistKvStride so the graph
        // shape (and therefore its CUDA capture) changes once every 256 tokens rather
        // than every token. The mask handles the exact per-query cutoff inside it.
        auto roundup = [](int v) { return ((v + kMgPersistKvStride - 1) / kMgPersistKvStride) * kMgPersistKvStride; };
        auto rounddown = [](int v) { return (v / kMgPersistKvStride) * kMgPersistKvStride; };

        const bool swa_ring = swa_cache_size > 0 && swa_cache_size < cache_size;
        const int window_full = std::min(cache_size, roundup(total_seq_len));
        int swa_start = 0;
        int window_swa = window_full;
        if (swa_ring)
        {
            // A ring is read whole: its slots are not in position order, so there is
            // no contiguous sub-span to narrow to. The mask carries the liveness.
            swa_start = 0;
            window_swa = swa_cache_size;
        }
        else if (sliding_window > 0 && sliding_window < total_seq_len)
        {
            const int needed_start = std::max(0, total_seq_len - n_tokens - (sliding_window - 1));
            swa_start = rounddown(needed_start);
            window_swa = std::min(cache_size - swa_start, roundup(total_seq_len - swa_start));
        }

        static const bool mg_persist = [] {
            const char* e = std::getenv("TS_MUSE_GLIMMER_PERSIST");
            return e == nullptr || e[0] != '0';
        }();

        // Diagnostics only (TS_MUSE_GLIMMER_LAYER_TRACE=1): copy the residual
        // ENTERING every layer, plus the final residual, out of the graph and print a
        // checksum for the FIRST forward. Diffing that against the per-op loop's
        // matching trace is what localizes a fused-vs-per-op divergence to a layer.
        // TS_MUSE_GLIMMER_LAYER_TRACE_POS selects the first forward at or past that
        // absolute position (which is what skips the startup warmup, whose own
        // prefill and decode run at unrelated positions), and ..._N how many
        // consecutive forwards to trace from there.
        static const bool mg_trace = [] {
            const char* e = std::getenv("TS_MUSE_GLIMMER_LAYER_TRACE");
            return e != nullptr && e[0] != '0' && e[0] != '\0';
        }();
        static const int mg_trace_pos = [] {
            const char* e = std::getenv("TS_MUSE_GLIMMER_LAYER_TRACE_POS");
            return e == nullptr ? 0 : std::atoi(e);
        }();
        static const int mg_trace_n = [] {
            const char* e = std::getenv("TS_MUSE_GLIMMER_LAYER_TRACE_N");
            const int v = e == nullptr ? 1 : std::atoi(e);
            return v > 0 ? v : 1;
        }();
        static int mg_trace_calls = 0;
        const bool trace_this_call = mg_trace && mg_trace_calls < mg_trace_n &&
            start_pos >= mg_trace_pos;

        // Only single-row decode gets the persistent capturable graph; a prefill
        // chunk is a different shape every time and is cheap to rebuild relative
        // to the work it does.
        //
        // Metal has no CUDA-graph analog (every submit re-encodes the nodes), but
        // the persist/replay path still removes the per-token graph metadata
        // rebuild, ~790 tensor re-binds, the gallocr lifetime re-plan, the 104
        // small norm re-uploads and the O(context) full-class mask regeneration
        // (the replay path extends that mask by 2 bytes per token instead).
        // Measured on an M5 Pro this is what closes the decode gap to llama.cpp,
        // whose graph-reuse path (llm_graph_result::can_reuse) does the same.
        const bool can_persist = mg_persist && n_tokens <= kMgMaxPersistTokens &&
            (g_backend_type == BACKEND_TYPE_CUDA || g_backend_type == BACKEND_TYPE_VULKAN ||
             g_backend_type == BACKEND_TYPE_METAL || g_backend_type == BACKEND_TYPE_CPU);

        const void* mg_sig = attn_norm_arr[0];
        const void* mg_kc0 = k_cache_arr != nullptr ? k_cache_arr[0] : nullptr;

        const bool fold_all = fold && all_logits_rows != 0;
        const int out_count = fold ? (fold_all ? vocab_size * n_tokens : vocab_size)
                                   : hidden_size * n_tokens;

        // ---------------- replay fast path ----------------
        MgDecodeCache* dc = can_persist ? mg_pool().find(mg_sig, mg_kc0, n_tokens) : nullptr;
        if (dc != nullptr && dc->graph != nullptr &&
            dc->num_layers == num_layers && dc->hidden_size == hidden_size &&
            dc->window_swa == window_swa && dc->window_full == window_full &&
            dc->swa_start == swa_start && dc->swa_rows == swa_cache_size &&
            dc->folded == fold && dc->out_count == out_count &&
            dc->capture_count == cap_count && dc->embed_in_graph == embed_in_graph &&
            dc->fold_all == fold_all && dc->n_tokens == n_tokens &&
            dc->tp_planned == tp_mode)
        {
            host_read_barrier();
            if (embed_in_graph)
                decode_input_set_async(dc->hidden_in, token_ids,
                    static_cast<std::size_t>(n_tokens) * sizeof(std::int32_t));
            else
                decode_input_set_async(dc->hidden_in, hidden_data,
                    static_cast<std::size_t>(hidden_size) * n_tokens * sizeof(float));

            std::vector<std::int32_t> pos_vals(n_tokens);
            std::vector<std::int64_t> row_vals(n_tokens);
            for (int i = 0; i < n_tokens; i++) { pos_vals[i] = start_pos + i; row_vals[i] = start_pos + i; }
            decode_input_set_async(dc->pos_tensor, pos_vals.data(), pos_vals.size() * sizeof(std::int32_t));
            decode_input_set_async(dc->kv_index, row_vals.data(), row_vals.size() * sizeof(std::int64_t));
            if (dc->kv_index_swa != nullptr)
            {
                std::vector<std::int64_t> ring_vals(n_tokens);
                for (int i = 0; i < n_tokens; i++) ring_vals[i] = (start_pos + i) % swa_cache_size;
                decode_input_set_async(dc->kv_index_swa, ring_vals.data(), ring_vals.size() * sizeof(std::int64_t));
            }

            // The SWA class is small (one ring, 4352 rows) and its liveness is not
            // monotone in start_pos, so it is regenerated whole. The FULL class is
            // O(window_full) - 125K entries at 128K - but for a 1-row decode its
            // content is a pure prefix of zeros that only ever GROWS by one entry
            // per token, so an in-place extension plus a 2-byte upload replaces a
            // 250 KB regenerate-and-resend. Any other shape (verify batches, a KV
            // rollback that moves start_pos backwards, a rebuilt window) falls back
            // to the full path.
            std::vector<ggml_fp16_t>& md_swa = dc->mask_host[1];
            std::vector<ggml_fp16_t>& md_full = dc->mask_host[0];
            if (swa_ring)
                fill_mg_ring_mask(md_swa, window_swa, n_tokens, start_pos, sliding_window);
            else
                fill_mg_mask(md_swa, window_swa, n_tokens, start_pos, total_seq_len, sliding_window, swa_start);
            dc->mask_pos[1] = start_pos;

            const std::size_t full_len = static_cast<std::size_t>(window_full) * n_tokens;
            const int prev_full_pos = dc->mask_pos[0];
            const bool full_extend = n_tokens == 1 && prev_full_pos >= 0 &&
                start_pos > prev_full_pos && start_pos < window_full &&
                md_full.size() == full_len;
            std::size_t full_off = 0, full_bytes = full_len * sizeof(ggml_fp16_t);
            if (full_extend)
            {
                const ggml_fp16_t zero_val = ggml_fp32_to_fp16(0.0f);
                for (int k = prev_full_pos + 1; k <= start_pos; k++)
                    md_full[static_cast<std::size_t>(k)] = zero_val;
                full_off = static_cast<std::size_t>(prev_full_pos + 1) * sizeof(ggml_fp16_t);
                full_bytes = static_cast<std::size_t>(start_pos - prev_full_pos) * sizeof(ggml_fp16_t);
            }
            else
            {
                fill_mg_mask(md_full, window_full, n_tokens, start_pos, total_seq_len, 0, 0);
            }
            dc->mask_pos[0] = start_pos;

            // Two uploads, not num_layers: every layer of a class points at the same
            // shared tensor, so uploading per layer would re-send the same bytes 39x/13x.
            if (dc->class_mask[1] != nullptr && !md_swa.empty())
                decode_input_set_async(dc->class_mask[1], md_swa.data(), md_swa.size() * sizeof(ggml_fp16_t));
            if (dc->class_mask[0] != nullptr && !md_full.empty())
                decode_input_set_range_async(dc->class_mask[0],
                    reinterpret_cast<const char*>(md_full.data()) + full_off, full_off, full_bytes);

            if (tp_mode)
            {
                // Every per-step input is staged on this rank's stream; the driver
                // runs the segments across all ranks and reduces at the boundaries.
                if (!dc->tp_plan.valid())
                {
                    set_last_error("Muse-Glimmer forward: cached graph has no tensor-parallel plan.");
                    dc->reset();
                    return 0;
                }
                mg_tp_bind_outputs(dc->tp_plan, dc->hidden_out,
                    dc->folded ? logits_data : hidden_data,
                    static_cast<std::size_t>(dc->out_count) * sizeof(float),
                    dc->capture_out, capture_data, hidden_size, n_tokens);
                tp_plan_out[0] = &dc->tp_plan;
                clear_last_error();
                return 1;
            }

            ggml_status st = tsg::graph_compute_profiled(g_backend, dc->graph, "muse-glimmer forward");
            if (st != GGML_STATUS_SUCCESS)
            {
                set_last_error("Muse-Glimmer forward: cached graph execution failed.");
                dc->reset();
                return 0;
            }
            // See the transient path: Metal must drain before a shared-buffer
            // tensor_get, or the capture rows are read mid-flight.
            if (dc->capture_count > 0 && g_backend_type == BACKEND_TYPE_METAL)
                tsg::sync_backend(g_backend);
            for (int c = 0; c < dc->capture_count; c++)
            {
                ggml_backend_tensor_get(dc->capture_out[c],
                    capture_data + static_cast<std::size_t>(c) * hidden_size * n_tokens,
                    0, static_cast<std::size_t>(hidden_size) * n_tokens * sizeof(float));
            }
            void* out_data = dc->folded ? logits_data : hidden_data;
            finalize_compute_with_download(dc->hidden_out, out_data,
                static_cast<std::size_t>(dc->out_count) * sizeof(float));
            host_read_barrier();
            clear_last_error();
            return 1;
        }

        MgDecodeCache* mgdc = nullptr;
        if (mg_persist)
        {
            if (can_persist) mgdc = &mg_pool().claim(mg_sig, mg_kc0, n_tokens);
            else             mg_pool().drop(mg_sig, mg_kc0, n_tokens);
        }

        // ---------------- build ----------------
        // Must not exceed ggml_pool's k_pool_buffer_size (32 MB) or acquire()
        // refuses and the transient (prefill) path cannot get a context at all.
        // The graph metadata for 52 layers is ~2 MB, so this is ample.
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("Muse-Glimmer forward: failed to init persist ggml context."); return 0; }
        }
        else
        {
            if (!context.init(ctx_size))
            {
                set_last_error("Muse-Glimmer forward: failed to create ggml context.");
                return 0;
            }
            ctx = context.value;
        }

        std::vector<MgLayerTensors> layers(num_layers);
        // ONE mask per attention class, not one per layer. The mask depends only on
        // (window, n_tokens, start_pos) and every sliding-window layer shares one
        // window while every full layer shares the other, so 52 tensors held 52 copies
        // of two distinct buffers. At 64K with a 2048-row chunk that was 13 x
        // [65536,2048] + 39 x [4352,2048] F16 = 3.90 GB of device masks (and the same
        // again on the host in layer_mask_data), on a card with 16 GB total holding
        // 9.2 GB of weights. Sharing brings it to 273 MB. llama.cpp has always done
        // this - one inp_kq_mask and one inp_kq_mask_swa per graph.
        // Index 0 = full-attention layers, index 1 = sliding-window layers.
        ggml_tensor* class_mask[2] = { nullptr, nullptr };
        std::vector<ggml_fp16_t> class_mask_data[2];
        // Generate the mask straight into its device buffer instead of building it
        // on the host and uploading. Only on the prefill/first-build path: the
        // persistent decode graph keeps decode_input_set_async, which is the
        // CUDA-capture-safe way in and is already cheap at one row.
        // Class 0 (full layers) always has span_start == 0, so the existing causal
        // kernel applies verbatim. Class 1 needs the ring kernel, and only when the
        // ring is actually on - the moving-span layout has a non-zero span_start
        // that neither device kernel models, so that case stays on the host.
        bool device_mask_fill[2] = { false, false };
#ifdef TSG_GGML_USE_CUDA
        if (g_backend_type == BACKEND_TYPE_CUDA && !can_persist)
        {
            device_mask_fill[0] = true;
            device_mask_fill[1] = swa_ring;
        }
#endif
        std::vector<ggml_tensor*> layer_attn_mask(num_layers, nullptr);
        std::vector<ggml_tensor*> capture_out(cap_count, nullptr);
        std::vector<ggml_tensor*> trace_out;   // TS_MUSE_GLIMMER_LAYER_TRACE only

        // Tensor-parallel cut points. `tp_partial` is the row-parallel matmul
        // output the collective reduces; `tp_boundary` is the last graph node that
        // may run before that reduction. Muse-Glimmer has no reshape or view
        // between either matmul and the norm that consumes it (unlike Gemma 4's
        // decode kernel), so the two are the same tensor - but they are recorded
        // separately because tp_plan_segments resolves the SEGMENT END from the
        // boundary and reduces the partial.
        //
        // The cut has to land on the RAW matmul output, before the post-attention
        // / post-FFN norms: those norms sit between the row-parallel projection and
        // the residual add, and RMSNorm is not linear, so norming per-rank partials
        // and summing afterwards is not the same function as summing then norming.
        // It produces plausible-looking output, which is exactly what makes it
        // expensive to find later.
        std::vector<ggml_tensor*> tp_partial;
        std::vector<ggml_tensor*> tp_boundary;
        if (tp_mode)
        {
            tp_partial.reserve(static_cast<std::size_t>(num_layers) * 2);
            tp_boundary.reserve(static_cast<std::size_t>(num_layers) * 2);
        }

        // Input stage: either the caller's hidden rows, or an in-graph
        // get_rows(token_embd, ids) + weightless input RMSNorm.
        ggml_tensor* hidden = nullptr;
        ggml_tensor* hidden_in = nullptr;   // per-call INPUT the replay refreshes
        ggml_tensor* tok_embd_t = nullptr;
        ggml_tensor* token_ids_t = nullptr;
        if (embed_in_graph)
        {
            tok_embd_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(tok_embd_type), tok_embd_ne0, tok_embd_ne1);
            token_ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
            ggml_set_input(token_ids_t);
            hidden = ggml_get_rows(ctx, tok_embd_t, token_ids_t);          // [hidden, n_tokens]
            hidden = ggml_rms_norm(ctx, hidden, eps);                      // weightless input norm
            hidden_in = token_ids_t;
        }
        else
        {
            hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_tokens);
            ggml_set_input(hidden);
            hidden_in = hidden;
        }
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
        ggml_tensor* kv_index = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
        ggml_set_input(pos_tensor);
        ggml_set_input(kv_index);
        // SWA layers write to (position % swa_cache_size), which is a different row
        // than the full layers' (position).
        ggml_tensor* kv_index_swa = nullptr;
        if (swa_ring)
        {
            kv_index_swa = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
            ggml_set_input(kv_index_swa);
        }

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            lt.attn_norm_w      = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.ffn_norm_w       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.post_ffn_norm_w  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.q_norm_w         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            lt.k_norm_w         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            lt.q_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(q_type_arr[l]),    q_ne0_arr[l],    q_ne1_arr[l]);
            lt.k_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(k_type_arr[l]),    k_ne0_arr[l],    k_ne1_arr[l]);
            lt.v_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(v_type_arr[l]),    v_ne0_arr[l],    v_ne1_arr[l]);
            lt.gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gate_type_arr[l]), gate_ne0_arr[l], gate_ne1_arr[l]);
            lt.o_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type_arr[l]),    o_ne0_arr[l],    o_ne1_arr[l]);
            lt.gu_w   = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gu_type_arr[l]),   gu_ne0_arr[l],   gu_ne1_arr[l]);
            lt.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type_arr[l]), down_ne0_arr[l], down_ne1_arr[l]);
            const int rows = (swa_ring && is_swa_arr[l] != 0) ? swa_cache_size : cache_size;
            lt.k_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, rows, num_kv_heads);
            lt.v_cached_t = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(kv_cache_type), head_dim, rows, num_kv_heads);
        }

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            const bool swa = is_swa_arr[l] != 0;
            const int window = swa ? window_swa : window_full;
            const int span_start = swa ? swa_start : 0;

            for (int c = 0; c < cap_count; c++)
            {
                if (capture_layers[c] != l) continue;
                ggml_tensor* dst = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_tokens);
                capture_out[c] = ggml_cpy(ctx, hidden, dst);
                ggml_set_output(capture_out[c]);
            }

            if (trace_this_call && !tp_mode)
            {
                ggml_tensor* td = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_tokens);
                trace_out.push_back(ggml_cpy(ctx, hidden, td));
                ggml_set_output(trace_out.back());
            }

            // 1. pre-attention norm; the same tensor feeds Q/K/V and the output gate
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), lt.attn_norm_w);

            // 2. projections (Q/K/V carry different quant types, so three matmuls)
            ggml_tensor* q_raw = ggml_mul_mat(ctx, lt.q_w, normed);        // [q_dim, n_tokens]
            ggml_tensor* k_raw = ggml_mul_mat(ctx, lt.k_w, normed);        // [kv_dim, n_tokens]
            ggml_tensor* v_raw = ggml_mul_mat(ctx, lt.v_w, normed);        // [kv_dim, n_tokens]
            ggml_tensor* gate  = ggml_mul_mat(ctx, lt.gate_w, normed);     // [q_dim, n_tokens]

            // 3. per-head QK RMSNorm (the Q weight carries qk_scale_factor)
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_raw, head_dim, num_heads, n_tokens);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_raw, head_dim, num_kv_heads, n_tokens);
            q_3d = ggml_mul(ctx, ggml_rms_norm(ctx, q_3d, eps), lt.q_norm_w);
            k_3d = ggml_mul(ctx, ggml_rms_norm(ctx, k_3d, eps), lt.k_norm_w);

            // 4. RoPE - NORM flavour (mode 0), sliding-window layers only.
            //    The full-attention layers are NoPE.
            if (swa)
            {
                q_3d = ggml_rope_ext(ctx, q_3d, pos_tensor, nullptr,
                    head_dim, /*mode=*/0, 0, rope_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
                k_3d = ggml_rope_ext(ctx, k_3d, pos_tensor, nullptr,
                    head_dim, /*mode=*/0, 0, rope_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
            }

            // 5. write K/V into the flat cache. set_rows keeps the write row an
            //    INPUT, so the decode graph stays byte-identical token-to-token.
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_raw, head_dim, num_kv_heads, n_tokens);
            // ggml_set_rows asserts ggml_is_contiguous_rows(b), not full contiguity,
            // and a 0,2,1,3 permute leaves nb[0] == 4 - so the cont was pure copy.
            // Two fewer kernels per layer, 104 fewer per decoded token.
            ggml_tensor* k_write = ggml_permute(ctx, k_3d, 0, 2, 1, 3);  // [hd, n_tokens, kv_heads]
            ggml_tensor* v_write = ggml_permute(ctx, v_3d, 0, 2, 1, 3);
            ggml_tensor* write_index = (swa_ring && swa) ? kv_index_swa : kv_index;
            lt.k_cpy = ggml_set_rows(ctx, lt.k_cached_t, k_write, write_index);
            lt.v_cpy = ggml_set_rows(ctx, lt.v_cached_t, v_write, write_index);

            // 6. attention over [0, window) with the causal + SWA mask. Built once
            //    per class and shared by every layer of that class; flash_attn_ext and
            //    soft_max_ext both read the mask without modifying it.
            const int mask_cls = swa ? 1 : 0;
            if (class_mask[mask_cls] == nullptr)
            {
                class_mask[mask_cls] = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, window, n_tokens, 1, 1);
                ggml_set_input(class_mask[mask_cls]);
                if (!device_mask_fill[mask_cls])
                {
                    if (swa_ring && swa)
                        fill_mg_ring_mask(class_mask_data[mask_cls], window, n_tokens, start_pos, sliding_window);
                    else
                        fill_mg_mask(class_mask_data[mask_cls], window, n_tokens, start_pos, total_seq_len,
                                     swa ? sliding_window : 0, span_start);
                }
            }
            ggml_tensor* mask = class_mask[mask_cls];
            layer_attn_mask[l] = mask;
            const int layer_rows = (swa_ring && swa) ? swa_cache_size : cache_size;

            // num_heads/num_kv_heads is already this rank's ratio under --tp, and
            // that is the ratio ggml-cuda sees (16 single-GPU, 16 per rank at
            // --tp 2 where each rank holds one KV head). It selects the flash-
            // attention kernel, which is what decides whether the window below
            // has to be materialised - see kv_window_needs_cuda_flash_attn_copy.
            const int fattn_gqa_ratio = num_kv_heads > 0 ? num_heads / num_kv_heads : 0;
            ggml_tensor* k_full = view_kv_cache_window(ctx, lt.k_cached_t, head_dim, layer_rows, num_kv_heads, span_start, window, kv_cache_type, n_tokens, fattn_gqa_ratio);
            ggml_tensor* v_full = view_kv_cache_window(ctx, lt.v_cached_t, head_dim, layer_rows, num_kv_heads, span_start, window, kv_cache_type, n_tokens, fattn_gqa_ratio);
            if (k_full == nullptr || v_full == nullptr)
            {
                set_last_error("Muse-Glimmer forward: failed to create KV cache views.");
                if (can_persist) ggml_free(ctx);
                return 0;
            }
            // NOTE: the windows above come back MATERIALISED on CUDA only when
            // ggml-cuda would actually dispatch this shape to its flash-attention
            // VEC kernel, which is the one that misreads a truncated-prefix view
            // (see kv_window_needs_cuda_flash_attn_copy in
            // ggml_ops_transformer_common.h for the measurement and for why the
            // predicate must not be spelled ggml_is_contiguous). For this model
            // that leaves exactly one case: a single-row decode on an Ada-or-newer
            // card with fewer than 8192 KV rows in the window. Turing/Ampere and
            // every window >= 8192 rows go to the MMA kernel, which honours the
            // strides (measured bit-identical, and the prefill path has always
            // relied on it). Prefill (n_tokens > 1) is on the MMA kernels too, so
            // its windows stay raw views; the sliding-window layers read the WHOLE
            // ring (window == rows) and are untouched either way.

            ggml_tensor* q_attn = ggml_permute(ctx, q_3d, 0, 2, 1, 3);     // [hd, n_tokens, n_heads]
            ggml_tensor* attn_flat;
            ggml_tensor* fa = flash_attn_ext_guarded(ctx, "Muse-Glimmer model", q_attn, k_full, v_full, mask,
                kq_scale, 0.0f, 0.0f, nullptr, GGML_PREC_F32);
            attn_flat = ggml_reshape_2d(ctx, fa, q_dim, n_tokens);

            // 7. attention output gate, then o_proj
            attn_flat = ggml_mul(ctx, attn_flat, ggml_sigmoid(ctx, gate));
            ggml_tensor* o_out = ggml_mul_mat(ctx, lt.o_w, attn_flat);
            // TP cut 1/2: o_proj is row-parallel, so o_out is this rank's partial
            // sum. Reduce it BEFORE the post-attention norm below.
            if (tp_mode) { tp_partial.push_back(o_out); tp_boundary.push_back(o_out); }

            // 8. post-attention norm (its own epsilon) + residual
            ggml_tensor* post_attn = ggml_mul(ctx, ggml_rms_norm(ctx, o_out, post_norm_eps), lt.post_attn_norm_w);
            ggml_tensor* residual1 = ggml_add(ctx, post_attn, hidden);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel

            // 9. FFN: pre-norm -> SwiGLU -> down
            ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), lt.ffn_norm_w);
            const std::int64_t interm = gu_ne1_arr[l] / 2;
            ggml_tensor* gu = ggml_mul_mat(ctx, lt.gu_w, ffn_normed);      // [2*interm, n_tokens]
            // One GLU node instead of view + 2x cont + silu + mul. With src1 == null
            // and swapped == false the kernel computes silu(first half) * second half
            // (ggml-cuda/unary.cu ggml_cuda_op_unary_gated), which is exactly the chain
            // this replaces - but register-resident instead of writing three [interm,
            // n_tokens] temporaries. Saves ~479 KB of traffic and 3 launches per layer
            // per row; at 52 layers that is ~156 fewer dispatches per decoded token.
            // NOTE: ggml.h's comment on ggml_glu claims the gate is the SECOND half,
            // which contradicts the kernel. The kernel is authoritative here, and
            // MuseGlimmer_GreedyContinuationMatchesLlamaCpp fails loudly if the halves
            // are swapped.
            (void) interm;
            ggml_tensor* act = ggml_glu(ctx, gu, GGML_GLU_OP_SWIGLU, /*swapped=*/false);
            ggml_tensor* down = ggml_mul_mat(ctx, lt.down_w, act);
            // TP cut 2/2: the FFN down projection is row-parallel too. Same rule -
            // the reduction lands on `down`, ahead of the post-FFN norm.
            if (tp_mode) { tp_partial.push_back(down); tp_boundary.push_back(down); }

            // 10. post-FFN norm (its own epsilon) + residual
            ggml_tensor* post_ffn = ggml_mul(ctx, ggml_rms_norm(ctx, down, post_norm_eps), lt.post_ffn_norm_w);
            hidden = ggml_add(ctx, post_ffn, residual1);   // normed first: lets ggml-metal fuse rms_norm+mul+add into one kernel
        }

        if (trace_this_call && !tp_mode)
        {
            ggml_tensor* td = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_tokens);
            trace_out.push_back(ggml_cpy(ctx, hidden, td));
            ggml_set_output(trace_out.back());
        }

        // ---------------- output ----------------
        ggml_tensor* lm_head_t = nullptr;
        ggml_tensor* final_norm_t = nullptr;
        ggml_tensor* hidden_out = nullptr;
        ggml_tensor* out_node = nullptr;
        if (fold)
        {
            lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
            final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            // Logits are only needed for the last row of the chunk.
            const int out_rows = fold_all ? n_tokens : 1;
            ggml_tensor* rows = (fold_all || n_tokens == 1)
                ? hidden
                : ggml_cont(ctx, ggml_view_2d(ctx, hidden, hidden_size, 1, hidden->nb[1],
                                              static_cast<std::size_t>(n_tokens - 1) * hidden->nb[1]));
            ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, rows, eps), final_norm_t);
            ggml_tensor* logits = ggml_mul_mat(ctx, lm_head_t, ggml_reshape_2d(ctx, fn, hidden_size, out_rows));
            if (logit_scale != 1.0f && logit_scale > 0.0f)
                logits = ggml_scale(ctx, logits, logit_scale);
            if (logit_softcap > 0.0f)
            {
                logits = ggml_scale(ctx, logits, 1.0f / logit_softcap);
                logits = ggml_tanh(ctx, logits);
                logits = ggml_scale(ctx, logits, logit_softcap);
            }
            logits = ggml_reshape_1d(ctx, logits, static_cast<std::int64_t>(vocab_size) * out_rows);
            hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, static_cast<std::int64_t>(vocab_size) * out_rows);
            out_node = ggml_cpy(ctx, logits, hidden_out);
        }
        else
        {
            hidden_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_tokens);
            out_node = ggml_cpy(ctx, hidden, hidden_out);
        }
        ggml_set_output(out_node);

        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * 128 + 512
            + trace_out.size() * 2;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        for (int l = 0; l < num_layers; l++)
        {
            ggml_build_forward_expand(graph, layers[l].k_cpy);
            ggml_build_forward_expand(graph, layers[l].v_cpy);
        }
        for (int c = 0; c < cap_count; c++)
            if (capture_out[c] != nullptr)
                ggml_build_forward_expand(graph, capture_out[c]);
        for (std::size_t t = 0; t < trace_out.size(); t++)
            ggml_build_forward_expand(graph, trace_out[t]);
        ggml_build_forward_expand(graph, out_node);

        // ---------------- bind ----------------
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                                enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            if (t == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, t, data, bytes, needs_upload, usage))
                {
                    if (needs_upload) upload_list.push_back({ t, data, bytes });
                    return;
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
            upload_list.push_back({ t, data, bytes });
        };

        const std::size_t norm_bytes = static_cast<std::size_t>(hidden_size) * sizeof(float);
        const std::size_t head_norm_bytes = static_cast<std::size_t>(head_dim) * sizeof(float);
        const std::size_t kv_bytes = kv_cache_bytes(num_kv_heads, cache_size, head_dim, kv_cache_type);
        const std::size_t kv_bytes_swa = swa_ring
            ? kv_cache_bytes(num_kv_heads, swa_cache_size, head_dim, kv_cache_type)
            : kv_bytes;
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            bind_or_mark(lt.q_w,    q_arr[l],    static_cast<std::size_t>(q_bytes_arr[l]),    true);
            bind_or_mark(lt.k_w,    k_arr[l],    static_cast<std::size_t>(k_bytes_arr[l]),    true);
            bind_or_mark(lt.v_w,    v_arr[l],    static_cast<std::size_t>(v_bytes_arr[l]),    true);
            bind_or_mark(lt.gate_w, gate_arr[l], static_cast<std::size_t>(gate_bytes_arr[l]), true);
            bind_or_mark(lt.o_w,    o_arr[l],    static_cast<std::size_t>(o_bytes_arr[l]),    true);
            bind_or_mark(lt.gu_w,   gu_arr[l],   static_cast<std::size_t>(gu_bytes_arr[l]),   true);
            bind_or_mark(lt.down_w, down_arr[l], static_cast<std::size_t>(down_bytes_arr[l]), true);

            bind_or_mark(lt.attn_norm_w,      attn_norm_arr[l],      norm_bytes, true);
            bind_or_mark(lt.post_attn_norm_w, post_attn_norm_arr[l], norm_bytes, true);
            bind_or_mark(lt.ffn_norm_w,       ffn_norm_arr[l],       norm_bytes, true);
            bind_or_mark(lt.post_ffn_norm_w,  post_ffn_norm_arr[l],  norm_bytes, true);
            bind_or_mark(lt.q_norm_w,         q_norm_arr[l],         head_norm_bytes, true);
            bind_or_mark(lt.k_norm_w,         k_norm_arr[l],         head_norm_bytes, true);

            const std::size_t lt_kv_bytes = (swa_ring && is_swa_arr[l] != 0) ? kv_bytes_swa : kv_bytes;
            bind_or_mark(lt.k_cached_t, k_cache_arr[l], lt_kv_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            bind_or_mark(lt.v_cached_t, v_cache_arr[l], lt_kv_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }
        if (fold)
        {
            bind_or_mark(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);
            bind_or_mark(final_norm_t, const_cast<void*>(final_norm_data), norm_bytes, true);
        }
        if (embed_in_graph)
            bind_or_mark(tok_embd_t, const_cast<void*>(tok_embd_data), static_cast<std::size_t>(tok_embd_bytes), true);

        // Metal only: run the backend's graph_optimize hook (alias-aware reorder
        // that widens the encoder's concurrent sets and groups fusable
        // norm+mul/add chains). ggml_backend_sched does this for llama.cpp; a
        // direct ggml_backend_graph_compute call does not. Persist builds only:
        // there the ~1 ms reorder runs once and every replay benefits, whereas
        // on the transient path it ran per prefill chunk / per non-persist token
        // and measured as a net loss (decode 20.6 -> 19.5 tok/s with persist
        // disabled, 16K prefill -2.7% on the M5 Pro). Must run BEFORE
        // allocation - reordering after lifetime planning can invalidate the
        // allocator's alias plan. Skipped under tensor parallelism, whose plan
        // segments are resolved from node positions.
        if (!tp_mode && can_persist)
            optimize_graph_for_metal(graph);

        // ---------------- allocate + upload ----------------
        // Decode (1 row) gets its own slot per tensor so the addresses are STABLE
        // and ggml-cuda can capture the graph. A prefill chunk must not do that:
        // its intermediates are per-row (the [2*n_ff, n_tokens] gate/up alone is
        // 163 MB per layer at 1024 rows), so allocating every node its own slot
        // asks for ~47 GB. The reusable gallocr packs them by lifetime, which is
        // what makes a multi-row fused graph fit at all.
        BufferHandle buffer(nullptr);
        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("Muse-Glimmer forward: failed to allocate persist backend buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else
        {
            // Inputs and any still-unbound weights need real storage before the
            // graph allocator plans the intermediates.
            for (auto& u : upload_list)
                ggml_set_input(u.tensor);
            if (!embed_in_graph)
                ggml_set_input(hidden_in);
            ggml_set_input(pos_tensor);
            ggml_set_input(kv_index);
            // The two class masks were marked as inputs where they were created.

            if (!alloc_graph_reuse_gallocr(graph))
            {
                set_last_error("Muse-Glimmer forward: failed to allocate the prefill graph.");
                return 0;
            }
        }

        host_read_barrier();

        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

        if (embed_in_graph)
            ggml_backend_tensor_set(token_ids_t, token_ids, 0,
                static_cast<std::size_t>(n_tokens) * sizeof(std::int32_t));
        else
            ggml_backend_tensor_set(hidden_in, hidden_data, 0,
                static_cast<std::size_t>(hidden_size) * n_tokens * sizeof(float));

        std::vector<std::int32_t> pos_vals(n_tokens);
        std::vector<std::int64_t> row_vals(n_tokens);
        for (int i = 0; i < n_tokens; i++)
        {
            pos_vals[i] = start_pos + i;
            row_vals[i] = start_pos + i;
        }
        ggml_backend_tensor_set(pos_tensor, pos_vals.data(), 0, pos_vals.size() * sizeof(std::int32_t));
        ggml_backend_tensor_set(kv_index, row_vals.data(), 0, row_vals.size() * sizeof(std::int64_t));
        if (kv_index_swa != nullptr)
        {
            std::vector<std::int64_t> ring_vals(n_tokens);
            for (int i = 0; i < n_tokens; i++) ring_vals[i] = (start_pos + i) % swa_cache_size;
            ggml_backend_tensor_set(kv_index_swa, ring_vals.data(), 0, ring_vals.size() * sizeof(std::int64_t));
        }

        // Two masks, not num_layers - every layer of a class shares one tensor.
        {
            bool filled_on_device = false;
            for (int cls = 0; cls < 2; cls++)
            {
                if (class_mask[cls] == nullptr) continue;
#ifdef TSG_GGML_USE_CUDA
                if (device_mask_fill[cls])
                {
                    const bool ok = (cls == 1)
                        ? tsg_cuda_fill_ring_mask_f16(class_mask[cls]->data, window_swa, n_tokens,
                                                      start_pos, sliding_window)
                        : tsg_cuda_fill_causal_mask_f16(class_mask[cls]->data, window_full, n_tokens,
                                                        start_pos, 0, total_seq_len);
                    if (ok) { filled_on_device = true; continue; }
                    // Kernel launch failed - rebuild on the host and upload instead.
                    if (cls == 1)
                        fill_mg_ring_mask(class_mask_data[cls], window_swa, n_tokens, start_pos, sliding_window);
                    else
                        fill_mg_mask(class_mask_data[cls], window_full, n_tokens, start_pos, total_seq_len, 0, 0);
                }
#endif
                if (class_mask_data[cls].empty()) continue;
                ggml_backend_tensor_set(class_mask[cls], class_mask_data[cls].data(), 0,
                    class_mask_data[cls].size() * sizeof(ggml_fp16_t));
            }
#ifdef TSG_GGML_USE_CUDA
            // Block until the stream-0 fills land, so the backend-stream graph
            // compute below sees finished masks.
            if (filled_on_device) tsg_cuda_sync_stream0();
#else
            (void) filled_on_device;
#endif
        }

        // ---------------- run ----------------
        // ...except under tensor parallelism, where the graph is only BUILT here:
        // the driver runs every rank's segments together so the AllReduces at the
        // boundaries have all the partials to reduce.
        void* out_data = fold ? logits_data : hidden_data;
        if (!tp_mode)
        {
            ggml_status status = tsg::compute_graph(g_backend, graph);
            if (status != GGML_STATUS_SUCCESS)
            {
                set_last_error("Muse-Glimmer forward: graph execution failed.");
                if (can_persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
                return 0;
            }

            // Metal's graph_compute returns with the GPU still running and a
            // shared-buffer tensor_get is a raw memcpy (no implicit sync), so
            // reading the capture/trace outputs below without draining first
            // returns stale bytes. CUDA/Vulkan get_tensor already synchronize.
            if ((cap_count > 0 || !trace_out.empty()) && g_backend_type == BACKEND_TYPE_METAL)
                tsg::sync_backend(g_backend);

            for (int c = 0; c < cap_count; c++)
            {
                if (capture_out[c] == nullptr) continue;
                ggml_backend_tensor_get(capture_out[c],
                    capture_data + static_cast<std::size_t>(c) * hidden_size * n_tokens,
                    0, static_cast<std::size_t>(hidden_size) * n_tokens * sizeof(float));
            }

            if (!trace_out.empty())
            {
                const int trace_seq = mg_trace_calls++;
                std::vector<float> tb(static_cast<std::size_t>(hidden_size) * n_tokens);
                for (std::size_t t = 0; t < trace_out.size(); t++)
                {
                    ggml_backend_tensor_get(trace_out[t], tb.data(), 0, tb.size() * sizeof(float));
                    double sum = 0.0, abs_sum = 0.0;
                    float mx = -std::numeric_limits<float>::infinity();
                    int nonfinite = 0;
                    for (std::size_t j = 0; j < tb.size(); j++)
                    {
                        sum += tb[j];
                        abs_sum += std::fabs(static_cast<double>(tb[j]));
                        if (tb[j] > mx) mx = tb[j];
                        if (!std::isfinite(tb[j])) nonfinite++;
                    }
                    std::fprintf(stderr,
                        "[MGTRACE-fused] pos=%d rows=%d layer=%d n=%zu sum=%.4f abs=%.4f max=%.6f nonfinite=%d\n",
                        start_pos, n_tokens, static_cast<int>(t), tb.size(), sum, abs_sum,
                        static_cast<double>(mx), nonfinite);
                    const char* dir = std::getenv("TS_MUSE_GLIMMER_LAYER_TRACE_DIR");
                    if (dir != nullptr && dir[0] != '\0')
                    {
                        char path[1024];
                        std::snprintf(path, sizeof(path), "%s/fused_S%02d_L%02d.bin", dir,
                                      trace_seq, static_cast<int>(t));
                        if (FILE* fp = std::fopen(path, "wb"))
                        {
                            std::fwrite(tb.data(), sizeof(float), tb.size(), fp);
                            std::fclose(fp);
                        }
                    }
                }
                std::fflush(stderr);
            }

            finalize_compute_with_download(hidden_out, out_data, static_cast<std::size_t>(out_count) * sizeof(float));
            // Unconditional: out_data is the caller's host buffer and on Metal async
            // mode the download above is only QUEUED.
            host_read_barrier();
        }

        if (can_persist && mgdc != nullptr)
        {
            mgdc->ctx = ctx;
            mgdc->buffer = persist_buf;
            mgdc->graph = graph;
            mgdc->hidden_in = hidden_in;
            mgdc->hidden_out = hidden_out;
            mgdc->pos_tensor = pos_tensor;
            mgdc->kv_index = kv_index;
            mgdc->kv_index_swa = kv_index_swa;
            mgdc->swa_rows = swa_cache_size;
            mgdc->attn_mask = layer_attn_mask;
            mgdc->class_mask[0] = class_mask[0];
            mgdc->class_mask[1] = class_mask[1];
            // Hand the host staging buffers to the cache entry so replays extend
            // them in place. An empty one (the device-fill path) leaves mask_pos
            // at -1, which makes the first replay regenerate from scratch.
            for (int cls = 0; cls < 2; cls++)
            {
                mgdc->mask_host[cls] = std::move(class_mask_data[cls]);
                mgdc->mask_pos[cls] = mgdc->mask_host[cls].empty() ? -1 : start_pos;
            }
            mgdc->capture_out = capture_out;
            mgdc->window_swa = window_swa;
            mgdc->swa_start = swa_start;
            mgdc->window_full = window_full;
            mgdc->sig_disc = mg_sig;
            mgdc->sig_kcache0 = mg_kc0;
            mgdc->num_layers = num_layers;
            mgdc->hidden_size = hidden_size;
            mgdc->folded = fold;
            mgdc->out_count = out_count;
            mgdc->capture_count = cap_count;
            mgdc->embed_in_graph = embed_in_graph;
            mgdc->fold_all = fold_all;
            mgdc->n_tokens = n_tokens;
            mgdc->tp_planned = tp_mode;
            mgdc->valid = true;
        }

        if (tp_mode)
        {
            // Whoever owns the context has to outlive this call: the persist cache
            // slot already does (and keeps the plan alongside the graph it
            // describes, so a replay hands back the same plan without rebuilding),
            // while a transient prefill graph parks its pooled context and any
            // per-call buffer in this rank's pending slot until its next build.
            tsg::TpRankPlan* plan = nullptr;
            if (can_persist)
            {
                if (mgdc == nullptr)
                {
                    set_last_error("Muse-Glimmer forward: tensor-parallel build produced no cache slot.");
                    ggml_backend_buffer_free(persist_buf);
                    ggml_free(ctx);
                    return 0;
                }
                plan = &mgdc->tp_plan;
            }
            else
            {
                auto& pending = g_mg_tp[tsg::g_active_rank];
                pending.context = std::move(context);
                pending.buffer = std::move(buffer);
                pending.ephemeral = std::move(ephemeral_bufs);
                plan = &pending.plan;
            }

            plan->clear();
            plan->graph = graph;
            plan->ar_tensor = tp_partial;
            if (!tp_plan_segments(*plan, tp_boundary))
            {
                if (can_persist) mgdc->reset();
                else             g_mg_tp[tsg::g_active_rank].reset();
                return 0;
            }
            mg_tp_bind_outputs(*plan, hidden_out, out_data,
                static_cast<std::size_t>(out_count) * sizeof(float),
                capture_out, capture_data, hidden_size, n_tokens);
            tp_plan_out[0] = plan;
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
        set_last_error("Unknown error in Muse-Glimmer forward.");
        return 0;
    }
}

// Drop the persistent (CUDA-graph-captured) Muse-Glimmer decode graphs. The
// captured graph pins ggml-cuda's compute-pool scratch addresses and the KV-cache
// device buffers, so a prefill (which grows the pool) or a KV reset/grow must
// invalidate it; the next decode rebuilds and re-captures.
TSG_EXPORT void TSGgml_MuseGlimmerResetDecodeCache()
{
    for (auto& pool : g_mg_pools)
        pool.reset_all();
}

// Release every rank's parked tensor-parallel Muse-Glimmer graph (the transient
// prefill path's context + per-call buffers). The C# side calls this on model
// teardown and on KV reset, while the backends are still alive - freeing these
// from a static destructor would run after the CUDA driver has shut down. The
// PERSISTENT decode graphs carry their plan inside the decode cache and are
// dropped by TSGgml_MuseGlimmerResetDecodeCache.
TSG_EXPORT void TSGgml_MuseGlimmerReleaseTpGraphs()
{
    for (int r = 0; r < tsg::TSG_MAX_DEVICES; ++r)
    {
        if (g_mg_tp[r].plan.graph == nullptr && g_mg_tp[r].context.value == nullptr)
            continue;
        tsg::ScopedRank rank(r);
        g_mg_tp[r].reset();
    }
}
