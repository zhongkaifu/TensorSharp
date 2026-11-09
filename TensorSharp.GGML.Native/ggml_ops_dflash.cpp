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
#if defined(TSG_GGML_USE_METAL)
#include "tsm/tsm_moe.h"
#endif
#include <cmath>
#include <cstdio>

using namespace tsg;

// ============================================================================
// DFlash drafter, fused.
//
// The trunk already runs as one graph (ggml_ops_muse_glimmer.cpp); the drafter
// did not, and that is what kept speculation a net loss. Per speculative step the
// managed drafter issued ~150 separate GPU dispatches, while its actual work is
// only ~2.5 GB of weight reads (5 blocks of 467M params plus the borrowed
// 202K-vocab LM head) - about a quarter of one target forward. Dispatch overhead,
// not arithmetic, was costing ~100 ms/step.
//
// Two entry points, one graph each:
//
//   TSGgml_DFlashInject - PASS A+B. fc(feat) -> RMSNorm -> per draft layer
//       {k_proj, v_proj, per-head k RMSNorm, NeoX RoPE, set_rows into the ring}.
//       No Q, no attention, no FFN.
//
//   TSGgml_DFlashDraftBlock - PASS C. [anchor, MASK x (b-1)] through the draft
//       blocks, then the TARGET's LM head and a softmax, returning [vocab, b]
//       probabilities. Attention reads the WHOLE ring plus this block's own keys.
//
// Reading the whole ring (rather than the live window) is deliberate: attention is
// permutation-invariant over the KV axis, so the ring's circular order does not
// matter, and a FIXED key length keeps the graph shape constant across steps -
// which is what lets ggml-cuda capture and replay it. The block's own keys are
// written into the ring at their slots (df_ring_block; the ring has a block of
// headroom past the window), so the key length is ring_rows; TS_DFLASH_RING_BLOCK=0
// concatenates the ring with the block instead ([ring_rows + b], a copy of the whole
// ring per layer). The host supplies a per-slot position map so the mask can express
// exactly which slots are live and visible.
//
// Mask semantics mirror the (llama.cpp-verified) managed implementation: a ring
// slot holding position p is masked from a block query at position qp when
// qp - p >= n_swa or p > qp; the block's own b columns are never masked, because
// DFlash drafts with llama_set_causal_attn(ctx_dft, false).
//
// ----------------------------------------------------------------------------
// DFlash2 (conv_taps > 0 / sel_rank > 0) adds two things to the draft graph:
//
//   GROUPED DYNAMIC CONVOLUTION around every attention and every FFN sublayer.
//   One projection of the sublayer INPUT yields both filters; tap t of channel c
//   at block position r is (base[side][t][c] + delta[r][t][c / group_size]) and
//   multiplies x[r-t][c], zeroed for r < t. The shift is a get_rows with a
//   constant index vector and the boundary mask a constant [1,1,b] multiply -
//   both baked once, because the block layout never changes between steps.
//
//   CANDIDATE SELECTOR instead of the per-row argmax. The LM head runs over the
//   gamma = b-1 PROPOSAL rows only (not the anchor's), a top-k keeps the
//   candidates, and the pairwise transition scores come back to the host as
//   k + k*k*(gamma-1) floats - ~7 KB, against 12.9 MB for a [vocab, b] readback.
//   The walk itself is gamma steps over k candidates and stays on the host: it is
//   inherently sequential and the data is already small.
// ============================================================================

namespace
{
    // Packed ABI table shared with ModelBase.DFlash.Fused: fc, embedding,
    // LM head, selector hidden/predecessor/successor; then Q,K,V,O,gate,up,down,
    // attention-conv and FFN-conv for each layer. These are quantization scales,
    // independent of model-level embedding/logit transforms.
    std::uint64_t df_scale_signature(const float* scales, int count)
    {
        std::uint64_t hash = 1469598103934665603ULL;
        for (int i = 0; i < count; ++i) {
            std::uint32_t bits;
            std::memcpy(&bits, scales + i, sizeof(bits));
            hash = (hash ^ bits) * 1099511628211ULL;
        }
        return scales == nullptr ? 0 : hash;
    }

    ggml_tensor* df_scaled(ggml_context* ctx, ggml_tensor* x, const float* scales, int index)
    {
        return scales == nullptr || scales[index] == 1.f ? x : ggml_scale(ctx, x, scales[index]);
    }

    // Persistent draft graphs are planned by a private gallocr so activations share
    // storage: allocating every intermediate separately kept a full F32 copy of the
    // ring (ring + block keys, K and V) for every layer - ~100 MB per single-lane
    // graph and ~340 MB per four-lane graph, enough to exhaust the device once each
    // request drafted over its own ring. Inputs, uploaded leaves and the outputs the
    // caller reads after compute keep their own storage.
    bool df_alloc_persistent(ggml_context* ctx, ggml_cgraph* graph, ggml_gallocr_t& galloc)
    {
        for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t))
        {
            if (t->op == GGML_OP_NONE && t->view_src == nullptr && t->data == nullptr)
            {
                ggml_set_input(t);
                ggml_set_output(t);
            }
            if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) != 0)
            {
                ggml_tensor* root = t;
                while (root->view_src != nullptr) root = root->view_src;
                ggml_set_output(root);
            }
        }
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (galloc == nullptr || !ggml_gallocr_alloc_graph(galloc, graph))
        {
            if (galloc != nullptr) { ggml_gallocr_free(galloc); galloc = nullptr; }
            return false;
        }
        return true;
    }

    struct DfLayer
    {
        ggml_tensor* attn_norm_w = nullptr;
        ggml_tensor* q_w = nullptr;
        ggml_tensor* k_w = nullptr;
        ggml_tensor* v_w = nullptr;
        ggml_tensor* q_norm_w = nullptr;
        ggml_tensor* k_norm_w = nullptr;
        ggml_tensor* o_w = nullptr;
        ggml_tensor* ffn_norm_w = nullptr;
        ggml_tensor* gate_w = nullptr;
        ggml_tensor* up_w = nullptr;
        ggml_tensor* down_w = nullptr;
        ggml_tensor* ring_k = nullptr;
        ggml_tensor* ring_v = nullptr;
        ggml_tensor* k_cpy = nullptr;
        ggml_tensor* v_cpy = nullptr;
        // DFlash2 only.
        ggml_tensor* attn_conv_base = nullptr;
        ggml_tensor* attn_conv_proj = nullptr;
        ggml_tensor* ffn_conv_base = nullptr;
        ggml_tensor* ffn_conv_proj = nullptr;
    };

    // Persistent graph cache. Both entry points key on
    // (model instance, ring buffer, row count) - see the pool in
    // ggml_ops_muse_glimmer.cpp for why n_tokens has to be part of the identity.
    struct DfCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* in_main = nullptr;      // feature rows (inject) or token ids (draft)
        ggml_tensor* pos = nullptr;
        ggml_tensor* kv_index = nullptr;     // inject only
        ggml_tensor* mask = nullptr;         // draft only (shared across layers)
        ggml_tensor* out = nullptr;
        ggml_tensor* out_conf = nullptr;   // draft only
        ggml_tensor* out_s0 = nullptr;     // DFlash2 selector: [k] anchor row
        ggml_tensor* out_scores = nullptr; // DFlash2 selector: [k, k, gamma-1]
        ggml_tensor* out_cand = nullptr;   // DFlash2 selector: [k, gamma] I32
        ggml_tensor* tail_h = nullptr;     // DFlash2 on tsm::df_tail_run: the final normed rows
        ggml_tensor* tail_ph = nullptr;    // and the selector's hidden projection
        ggml_tensor* tail_head = nullptr, *tail_pred = nullptr, *tail_succ = nullptr;   // the tail's weights (bound)
        const void* sig = nullptr;
        const void* sig_ring = nullptr;
        int n_rows = 0;
        int out_count = 0;
        std::uint64_t scale_signature = 0;
        ggml_gallocr_t galloc = nullptr;     // draft graphs: storage planned by a private gallocr

        void reset()
        {
            if (galloc != nullptr) { ggml_gallocr_free(galloc); galloc = nullptr; }
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            in_main = pos = kv_index = mask = out = out_conf = nullptr;
            out_s0 = out_scores = out_cand = nullptr;
            tail_h = tail_ph = tail_head = tail_pred = tail_succ = nullptr;
            sig = sig_ring = nullptr;
            n_rows = out_count = 0;
            scale_signature = 0;
        }
    };

    // Each request speculates over its own pooled ring (ModelBase.DFlash.Rings.cs),
    // and every ring has one draft graph plus one inject graph per row count the
    // executor commits (1 .. window). Eight entries thrashed with two rings.
    constexpr int kDfMaxCaches = 64;
    struct DfCachePool
    {
        DfCache entries[kDfMaxCaches];
        std::uint64_t used[kDfMaxCaches] = {};
        std::uint64_t clock = 0;

        DfCache* find(const void* sig, const void* ring, int n_rows, std::uint64_t scale_sig)
        {
            for (int i = 0; i < kDfMaxCaches; i++)
                if (entries[i].valid && entries[i].sig == sig && entries[i].sig_ring == ring
                    && entries[i].n_rows == n_rows && entries[i].scale_signature == scale_sig)
                { used[i] = ++clock; return &entries[i]; }
            return nullptr;
        }

        DfCache& claim(const void* sig, const void* ring, int n_rows, std::uint64_t scale_sig)
        {
            for (int i = 0; i < kDfMaxCaches; i++)
                if (entries[i].valid && entries[i].sig == sig && entries[i].sig_ring == ring
                    && entries[i].n_rows == n_rows && entries[i].scale_signature == scale_sig)
                { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            for (int i = 0; i < kDfMaxCaches; i++)
                if (!entries[i].valid) { entries[i].reset(); used[i] = ++clock; return entries[i]; }
            int lru = 0;
            for (int i = 1; i < kDfMaxCaches; i++) if (used[i] < used[lru]) lru = i;
            entries[lru].reset(); used[lru] = ++clock; return entries[lru];
        }

        void reset_all() { for (auto& e : entries) e.reset(); }
    };

    DfCachePool g_df_inject_pools[tsg::TSG_MAX_DEVICES];
    DfCachePool g_df_draft_pools[tsg::TSG_MAX_DEVICES];
    inline DfCachePool& df_inject_pool() { return g_df_inject_pools[tsg::g_active_rank]; }
    inline DfCachePool& df_draft_pool()  { return g_df_draft_pools[tsg::g_active_rank]; }

    struct DfBinder
    {
        struct Upload { ggml_tensor* t; void* data; std::size_t bytes; };
        std::vector<Upload> uploads;
        std::vector<BufferHandle> ephemeral;
        ggml_backend_dev_t dev = nullptr;

        void bind(ggml_tensor* t, void* data, std::size_t bytes, bool cacheable,
                  enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
        {
            if (t == nullptr || data == nullptr) return;
            if (cacheable && bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, t, data, bytes, needs_upload, usage))
                {
                    if (needs_upload) uploads.push_back({ t, data, bytes });
                    return;
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, data, bytes, cacheable, buf))
                {
                    if (!cacheable) ephemeral.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, t, data) == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            uploads.push_back({ t, data, bytes });
        }

        void flush()
        {
            for (auto& u : uploads)
                ggml_backend_tensor_set(u.t, resolve_upload_source(u.data), 0, u.bytes);
        }
    };

    // DFlash2 grouped dynamic depthwise convolution over one block.
    //
    //   out[r][c] = sum_t (base[side][t][c] + delta[r][t][c / group_size]) * x[r-t][c]
    //
    // with tap t masked off for the first t rows of the block. `x` is [hidden, n]
    // and `coef` the [2 * taps * groups, n] projection of the sublayer input, laid
    // out side-major then tap then group (the order the checkpoint exports).
    // shift_idx[t] / tap_mask[t] are the constant row-shift and boundary mask for
    // tap t (both null at t == 0, which is never shifted and never masked).
    // The all-taps form of df_grouped_conv on Metal (TS_DFLASH_METAL_FUSE=0 keeps one chain per tap). A tap costs the
    // per-tap form five to seven dependent kernels, and a draft runs four convolutions; this form runs every tap in
    // each kernel (~7 a convolution with 3 taps instead of 18) with the same products and sums in the same order.
    bool df_metal_fused_conv()
    {
        const char* e = std::getenv("TS_DFLASH_METAL_FUSE");
        return g_backend_type == BACKEND_TYPE_METAL && !(e != nullptr && e[0] == '0');
    }

    // The draft block's keys and values written into the ring at their own slots (position % ring_rows) and the
    // attention run over the ring alone, instead of over a concatenation of the whole ring with the block in every
    // layer. TS_DFLASH_RING_BLOCK=0 keeps the concatenation. The ring has a block and an anchor of headroom past the
    // sliding window (DFlashConfig.RingRows), so a block slot only ever held a position no query can see; the rows
    // stay until the committed positions' inject overwrites them, and the slot map never labels them live.
    bool df_ring_block()
    {
        const char* e = std::getenv("TS_DFLASH_RING_BLOCK");
        return !(e != nullptr && e[0] == '0');
    }

    // shift_all [n * taps] I32 / mask_all [1, n, taps] F32, tap-major: row r of tap t reads row shift_all[t*n + r]
    // (r - t, or 0), kept where mask_all is 1. Tap 0 is the identity, unmasked.
    ggml_tensor* df_grouped_conv_all_taps(
        ggml_context* ctx, ggml_tensor* x, ggml_tensor* coef, ggml_tensor* base_w,
        int side, int taps, int groups, int group_size, int hidden, int n,
        ggml_tensor* shift_all, ggml_tensor* mask_all)
    {
        // Every tap's shifted, masked rows: [hidden, n, taps]. (x * 1 and x * 0 are exact, so masking before the
        // product gives the per-tap form's values.)
        ggml_tensor* xs = ggml_reshape_3d(ctx, ggml_get_rows(ctx, x, shift_all), hidden, n, taps);
        xs = ggml_mul(ctx, xs, mask_all);
        // base[side][t] for every tap, [hidden, 1, taps], over the rows.
        ggml_tensor* b3 = ggml_view_3d(ctx, base_w, hidden, 1, taps,
                                       static_cast<std::size_t>(hidden) * sizeof(float),
                                       static_cast<std::size_t>(hidden) * sizeof(float),
                                       static_cast<std::size_t>(side) * taps * hidden * sizeof(float));
        ggml_tensor* brep = ggml_repeat_4d(ctx, b3, hidden, n, taps, 1);
        // delta[r][side][t][g]: coef's side block viewed [G, taps, n], laid out [G, n, taps].
        ggml_tensor* dv = ggml_view_3d(ctx, coef, groups, taps, n,
                                       static_cast<std::size_t>(groups) * sizeof(float), coef->nb[1],
                                       static_cast<std::size_t>(side) * taps * groups * sizeof(float));
        ggml_tensor* d = ggml_cont(ctx, ggml_permute(ctx, dv, 0, 2, 1, 3));
        ggml_tensor* kern = ggml_add(ctx, ggml_reshape_4d(ctx, brep, group_size, groups, n, taps),
                                     ggml_reshape_4d(ctx, d, 1, groups, n, taps));            // [S, G, n, taps]
        ggml_tensor* term = ggml_mul(ctx, kern, ggml_reshape_4d(ctx, xs, group_size, groups, n, taps));
        term = ggml_reshape_3d(ctx, term, hidden, n, taps);
        // Taps summed in order (an ADD chain ggml-metal runs as one kernel).
        ggml_tensor* out = ggml_view_2d(ctx, term, hidden, n, term->nb[1], 0);
        for (int tap = 1; tap < taps; tap++)
            out = ggml_add(ctx, out, ggml_view_2d(ctx, term, hidden, n, term->nb[1],
                                                  static_cast<std::size_t>(tap) * term->nb[2]));
        return out;
    }

    ggml_tensor* df_grouped_conv(
        ggml_context* ctx, ggml_tensor* x, ggml_tensor* coef, ggml_tensor* base_w,
        int side, int taps, int groups, int group_size, int hidden, int n,
        ggml_tensor** shift_idx, ggml_tensor** tap_mask,
        ggml_tensor* shift_all = nullptr, ggml_tensor* mask_all = nullptr)
    {
        if (shift_all != nullptr && mask_all != nullptr && taps >= 2)
            return df_grouped_conv_all_taps(ctx, x, coef, base_w, side, taps, groups, group_size, hidden, n,
                                            shift_all, mask_all);
        ggml_tensor* x3 = ggml_reshape_3d(ctx, x, group_size, groups, n);   // [S, G, n]
        ggml_tensor* out = nullptr;
        for (int tap = 0; tap < taps; tap++)
        {
            // Static, per-channel half of the kernel: base[side][tap] as [S, G, 1].
            const std::size_t base_off =
                (static_cast<std::size_t>(side) * taps + tap) * static_cast<std::size_t>(hidden) * sizeof(float);
            ggml_tensor* b3 = ggml_view_3d(ctx, base_w, group_size, groups, 1,
                                           static_cast<std::size_t>(group_size) * sizeof(float),
                                           static_cast<std::size_t>(hidden) * sizeof(float), base_off);

            // Dynamic, per-token per-group half: the (side, tap) slice of coef.
            // Materialized rather than viewed because the slice is strided and the
            // add below broadcasts it over the group's channels.
            const std::size_t coef_off =
                (static_cast<std::size_t>(side) * taps + tap) * static_cast<std::size_t>(groups) * sizeof(float);
            ggml_tensor* d2 = ggml_cont(ctx, ggml_view_2d(ctx, coef, groups, n, coef->nb[1], coef_off));
            ggml_tensor* d3 = ggml_reshape_3d(ctx, d2, 1, groups, n);

            ggml_tensor* kern = ggml_add(ctx, ggml_repeat(ctx, b3, x3), d3);  // [S, G, n]

            ggml_tensor* xs = x3;
            if (tap > 0)
                xs = ggml_reshape_3d(ctx, ggml_get_rows(ctx, x, shift_idx[tap]), group_size, groups, n);

            ggml_tensor* term = ggml_mul(ctx, kern, xs);
            if (tap > 0)
                term = ggml_mul(ctx, term, tap_mask[tap]);                    // [1, 1, n]
            out = (out == nullptr) ? term : ggml_add(ctx, out, term);
        }
        return ggml_reshape_2d(ctx, out, hidden, n);
    }
}

// ---------------------------------------------------------------------------
// PASS A + B: encode the target's per-layer residual features and inject the
// resulting K/V into the drafter's ring.
// ---------------------------------------------------------------------------
static int df_inject(
    const float* feat_rows, int feature_size, int n_rows,
    const std::int64_t* ring_rows_idx,          // ring slot per row (position % ring_rows)
    const int* positions,                        // absolute position per row (for RoPE)
    int num_layers, int hidden_size, int head_dim, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale,
    const void* fc_data, int fc_type, std::int64_t fc_ne0, std::int64_t fc_ne1, std::int64_t fc_bytes,
    const void* enc_norm_data,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** k_norm_arr,
    void** ring_k_arr, void** ring_v_arr,
    int ring_dtype,
    const float* scales, int scale_count)
{
    try
    {
        if (!ensure_backend()) return 0;
        if ((scales == nullptr && scale_count != 0) ||
            (scales != nullptr && (num_layers <= 0 || scale_count != 6 + 9 * num_layers)))
        { set_last_error("DFlash: invalid sidecar scale table."); return 0; }
        const std::uint64_t scale_sig = df_scale_signature(scales, scale_count);
        if (n_rows <= 0 || num_layers <= 0) { set_last_error("DFlash inject: invalid arguments."); return 0; }

        const void* sig = k_arr[0];
        const void* sig_ring = ring_k_arr[0];
        // Metal replays the persistent graph without a capture (every submit
        // re-encodes) but still skips the per-step metadata rebuild, re-binds
        // and gallocr re-plan - same rationale as the trunk kernel.
        const bool can_persist =
            g_backend_type == BACKEND_TYPE_CUDA || g_backend_type == BACKEND_TYPE_VULKAN ||
            g_backend_type == BACKEND_TYPE_METAL;

        DfCache* dc = can_persist ? df_inject_pool().find(sig, sig_ring, n_rows, scale_sig) : nullptr;
        if (dc != nullptr && dc->graph != nullptr)
        {
            host_read_barrier();
            decode_input_set_async(dc->in_main, feat_rows,
                static_cast<std::size_t>(feature_size) * n_rows * sizeof(float));
            decode_input_set_async(dc->pos, positions, static_cast<std::size_t>(n_rows) * sizeof(std::int32_t));
            decode_input_set_async(dc->kv_index, ring_rows_idx, static_cast<std::size_t>(n_rows) * sizeof(std::int64_t));
            if (tsg::graph_compute_profiled(g_backend, dc->graph, "dflash inject") != GGML_STATUS_SUCCESS)
            {
                set_last_error("DFlash inject: cached graph execution failed.");
                dc->reset();
                return 0;
            }
            host_read_barrier();
            clear_last_error();
            return 1;
        }

        DfCache* slot = can_persist ? &df_inject_pool().claim(sig, sig_ring, n_rows, scale_sig) : nullptr;

        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle pooled;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("DFlash inject: failed to init context."); return 0; }
        }
        else
        {
            if (!pooled.init(ctx_size)) { set_last_error("DFlash inject: failed to create context."); return 0; }
            ctx = pooled.value;
        }

        ggml_tensor* feat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feature_size, n_rows);
        ggml_tensor* pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_rows);
        ggml_tensor* idx_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_rows);
        ggml_set_input(feat); ggml_set_input(pos_t); ggml_set_input(idx_t);

        ggml_tensor* fc_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(fc_type), fc_ne0, fc_ne1);
        ggml_tensor* enc_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        // g = rmsnorm(fc @ feat) * enc_output_norm
        ggml_tensor* g = ggml_mul(ctx, ggml_rms_norm(ctx, df_scaled(ctx, ggml_mul_mat(ctx, fc_t, feat), scales, 0), eps), enc_norm_t);

        std::vector<DfLayer> layers(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            lt.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(k_type_arr[l]), k_ne0_arr[l], k_ne1_arr[l]);
            lt.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(v_type_arr[l]), v_ne0_arr[l], v_ne1_arr[l]);
            lt.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            lt.ring_k = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(ring_dtype), head_dim, ring_rows, num_kv_heads);
            lt.ring_v = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(ring_dtype), head_dim, ring_rows, num_kv_heads);

            ggml_tensor* k = df_scaled(ctx, ggml_mul_mat(ctx, lt.k_w, g), scales, 6 + 9 * l + 1);                 // [kv_dim, n]
            ggml_tensor* v = df_scaled(ctx, ggml_mul_mat(ctx, lt.v_w, g), scales, 6 + 9 * l + 2);
            ggml_tensor* k3 = ggml_reshape_3d(ctx, k, head_dim, num_kv_heads, n_rows);
            ggml_tensor* v3 = ggml_reshape_3d(ctx, v, head_dim, num_kv_heads, n_rows);
            k3 = ggml_mul(ctx, ggml_rms_norm(ctx, k3, eps), lt.k_norm_w);
            // NeoX rope: llama.cpp maps LLM_ARCH_DFLASH to LLAMA_ROPE_TYPE_NEOX,
            // which differs from the Muse-Glimmer trunk's NORM flavour. V is neither
            // normed nor roped.
            k3 = ggml_rope_ext(ctx, k3, pos_t, nullptr, head_dim, /*mode=*/2, 0,
                               rope_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);

            ggml_tensor* kw = ggml_cont(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3));   // [hd, n, kv_heads]
            ggml_tensor* vw = ggml_cont(ctx, ggml_permute(ctx, v3, 0, 2, 1, 3));
            lt.k_cpy = ggml_set_rows(ctx, lt.ring_k, kw, idx_t);
            lt.v_cpy = ggml_set_rows(ctx, lt.ring_v, vw, idx_t);
        }

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, static_cast<std::size_t>(num_layers) * 64 + 256, false);
        for (int l = 0; l < num_layers; l++)
        {
            ggml_build_forward_expand(graph, layers[l].k_cpy);
            ggml_build_forward_expand(graph, layers[l].v_cpy);
        }

        DfBinder binder;
        binder.dev = ggml_backend_get_device(g_backend);
        binder.bind(fc_t, const_cast<void*>(fc_data), static_cast<std::size_t>(fc_bytes), true);
        binder.bind(enc_norm_t, const_cast<void*>(enc_norm_data),
                    static_cast<std::size_t>(hidden_size) * sizeof(float), true);
        const std::size_t ring_bytes = kv_cache_bytes(num_kv_heads, ring_rows, head_dim, ring_dtype);
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            binder.bind(lt.k_w, k_arr[l], static_cast<std::size_t>(k_bytes_arr[l]), true);
            binder.bind(lt.v_w, v_arr[l], static_cast<std::size_t>(v_bytes_arr[l]), true);
            binder.bind(lt.k_norm_w, k_norm_arr[l], static_cast<std::size_t>(head_dim) * sizeof(float), true);
            binder.bind(lt.ring_k, ring_k_arr[l], ring_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            binder.bind(lt.ring_v, ring_v_arr[l], ring_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }

        // Metal: reorder for encoder concurrency before allocation (no-op on
        // other backends) - same as the trunk kernel.
        optimize_graph_for_metal(ctx, graph);

        ggml_backend_buffer_t persist_buf = nullptr;
        if (can_persist)
        {
            persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (persist_buf == nullptr)
            {
                set_last_error("DFlash inject: failed to allocate persist buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else
        {
            for (auto& u : binder.uploads) ggml_set_input(u.t);
            if (!alloc_graph_reuse_gallocr(graph))
            {
                set_last_error("DFlash inject: failed to allocate graph.");
                return 0;
            }
        }

        host_read_barrier();
        binder.flush();
        ggml_backend_tensor_set(feat, feat_rows, 0, static_cast<std::size_t>(feature_size) * n_rows * sizeof(float));
        ggml_backend_tensor_set(pos_t, positions, 0, static_cast<std::size_t>(n_rows) * sizeof(std::int32_t));
        ggml_backend_tensor_set(idx_t, ring_rows_idx, 0, static_cast<std::size_t>(n_rows) * sizeof(std::int64_t));

        if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("DFlash inject: graph execution failed.");
            if (can_persist) { ggml_backend_buffer_free(persist_buf); ggml_free(ctx); }
            return 0;
        }
        host_read_barrier();

        if (can_persist && slot != nullptr)
        {
            slot->ctx = ctx; slot->buffer = persist_buf; slot->graph = graph;
            slot->in_main = feat; slot->pos = pos_t; slot->kv_index = idx_t;
            slot->sig = sig; slot->sig_ring = sig_ring; slot->n_rows = n_rows;
            slot->valid = true;
            slot->scale_signature = scale_sig;
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in DFlash inject."); return 0; }
}

// ---------------------------------------------------------------------------
// PASS A + B for several rings at once. A batched speculative step commits every
// lane's accepted rows and then carries each lane's last row into its draft; one
// inject per lane cost a graph launch and a stream sync each, eight per step at
// four lanes. Here every lane's rows go through one encode and each lane's K/V
// rows land in its own ring.
//
// Lanes are padded to `max_rows` rows. `gather` maps every padded row to one of
// the `total_rows` compact feature rows the caller uploads; a lane's padding
// repeats its last row (feature, slot and position alike), so the extra set_rows
// writes store the bytes that slot receives anyway. Rows are independent through
// the encode (per-column matmuls, per-row norms, per-row RoPE), so each lane's
// K/V match a per-lane inject up to the GEMM kernel the batch size selects.
// CUDA only; the graph is cached per (weights, rings, max_rows, lanes).
// ---------------------------------------------------------------------------
namespace
{
    struct DfInjectBatchCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* feat = nullptr;
        ggml_tensor* gather = nullptr;
        ggml_tensor* pos = nullptr;
        ggml_tensor* kv_index = nullptr;
        const void* sig = nullptr;
        std::vector<const void*> rings;
        int max_rows = 0, lanes = 0;
        std::uint64_t scale_signature = 0;
        std::uint64_t lru = 0;

        void reset()
        {
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            feat = gather = pos = kv_index = nullptr;
            sig = nullptr; rings.clear(); max_rows = lanes = 0; scale_signature = 0;
        }
    };
    // One graph per padded row count (1 for the carried rows, 1..window at commit)
    // for the current lane set, plus room for the lane set to change.
    constexpr int kDfInjectBatchCaches = 24;
    struct DfInjectBatchPool
    {
        DfInjectBatchCache entries[kDfInjectBatchCaches];
        std::uint64_t clock = 0;
    };
    DfInjectBatchPool g_df_inject_batch_pools[tsg::TSG_MAX_DEVICES];
    inline DfInjectBatchPool& df_inject_batch_pool() { return g_df_inject_batch_pools[tsg::g_active_rank]; }
}

static int df_inject_batched(
    int n_lanes, int max_rows, int total_rows,
    const float* feat_rows, int feature_size,     // [total_rows, feature_size], compact
    const std::int32_t* gather,                   // [n_lanes * max_rows] -> compact row
    const std::int64_t* ring_rows_idx,            // [n_lanes * max_rows] ring slot per padded row
    const int* positions,                         // [n_lanes * max_rows] absolute position per padded row
    int num_layers, int hidden_size, int head_dim, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale,
    const void* fc_data, int fc_type, std::int64_t fc_ne0, std::int64_t fc_ne1, std::int64_t fc_bytes,
    const void* enc_norm_data,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** k_norm_arr,
    void** ring_k_arr, void** ring_v_arr,         // [n_lanes * num_layers], lane-major
    int ring_dtype,
    const float* scales, int scale_count)
{
    try
    {
        if (!ensure_backend()) return 0;
        if (g_backend_type != BACKEND_TYPE_CUDA)
        { set_last_error("DFlash batched inject: requires CUDA."); return 0; }
        if ((scales == nullptr && scale_count != 0) ||
            (scales != nullptr && (num_layers <= 0 || scale_count != 6 + 9 * num_layers)))
        { set_last_error("DFlash: invalid sidecar scale table."); return 0; }
        const int L = n_lanes;
        const int R = max_rows;
        const int nr = L * R;
        if (L < 1 || R < 1 || num_layers <= 0 || total_rows < 1 || total_rows > nr
            || feat_rows == nullptr || gather == nullptr || ring_rows_idx == nullptr || positions == nullptr)
        { set_last_error("DFlash batched inject: invalid arguments."); return 0; }
        for (int i = 0; i < nr; i++)
            if (gather[i] < 0 || gather[i] >= total_rows || ring_rows_idx[i] < 0 || ring_rows_idx[i] >= ring_rows)
            { set_last_error("DFlash batched inject: row index out of range."); return 0; }
        const std::uint64_t scale_sig = df_scale_signature(scales, scale_count);

        const void* sig = k_arr[0];
        std::vector<const void*> rings(ring_k_arr, ring_k_arr + static_cast<std::size_t>(L) * num_layers);
        DfInjectBatchPool& pool = df_inject_batch_pool();
        DfInjectBatchCache* dc = nullptr;
        for (auto& e : pool.entries)
            if (e.valid && e.sig == sig && e.max_rows == R && e.lanes == L && e.scale_signature == scale_sig && e.rings == rings)
            { dc = &e; break; }

        auto run = [&](DfInjectBatchCache& c) {
            decode_input_set_async(c.feat, feat_rows, static_cast<std::size_t>(feature_size) * total_rows * sizeof(float));
            decode_input_set_async(c.gather, gather, static_cast<std::size_t>(nr) * sizeof(std::int32_t));
            decode_input_set_async(c.pos, positions, static_cast<std::size_t>(nr) * sizeof(std::int32_t));
            decode_input_set_async(c.kv_index, ring_rows_idx, static_cast<std::size_t>(nr) * sizeof(std::int64_t));
            return tsg::graph_compute_profiled(g_backend, c.graph, "dflash batched inject") == GGML_STATUS_SUCCESS;
        };

        host_read_barrier();
        if (dc != nullptr)
        {
            dc->lru = ++pool.clock;
            if (!run(*dc))
            {
                set_last_error("DFlash batched inject: cached graph execution failed.");
                dc->reset();
                return 0;
            }
            clear_last_error();
            return 1;
        }

        DfInjectBatchCache* slot = nullptr;
        for (auto& e : pool.entries)
            if (!e.valid) { slot = &e; break; }
        if (slot == nullptr)
        {
            slot = &pool.entries[0];
            for (auto& e : pool.entries)
                if (e.lru < slot->lru) slot = &e;
        }
        slot->reset();

        ggml_init_params ip = { 32 * 1024 * 1024, nullptr, /*no_alloc=*/true };
        ggml_context* ctx = ggml_init(ip);
        if (ctx == nullptr) { set_last_error("DFlash batched inject: failed to init context."); return 0; }

        ggml_tensor* feat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, feature_size, nr);
        ggml_tensor* gather_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nr);
        ggml_tensor* pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nr);
        ggml_tensor* idx_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nr);
        ggml_set_input(feat); ggml_set_input(gather_t); ggml_set_input(pos_t); ggml_set_input(idx_t);

        ggml_tensor* fc_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(fc_type), fc_ne0, fc_ne1);
        ggml_tensor* enc_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        // Only the first total_rows rows of `feat` are uploaded; the gather reads nothing else.
        ggml_tensor* x = ggml_get_rows(ctx, feat, gather_t);                                   // [F, nr]
        ggml_tensor* g = ggml_mul(ctx, ggml_rms_norm(ctx, df_scaled(ctx, ggml_mul_mat(ctx, fc_t, x), scales, 0), eps), enc_norm_t);

        std::vector<DfLayer> layers(num_layers);
        std::vector<ggml_tensor*> ring_k(static_cast<std::size_t>(L) * num_layers);
        std::vector<ggml_tensor*> ring_v(static_cast<std::size_t>(L) * num_layers);
        std::vector<ggml_tensor*> writes;
        writes.reserve(static_cast<std::size_t>(L) * num_layers * 2);
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            lt.k_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(k_type_arr[l]), k_ne0_arr[l], k_ne1_arr[l]);
            lt.v_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(v_type_arr[l]), v_ne0_arr[l], v_ne1_arr[l]);
            lt.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);

            ggml_tensor* k = df_scaled(ctx, ggml_mul_mat(ctx, lt.k_w, g), scales, 6 + 9 * l + 1);   // [kv_dim, nr]
            ggml_tensor* v = df_scaled(ctx, ggml_mul_mat(ctx, lt.v_w, g), scales, 6 + 9 * l + 2);
            ggml_tensor* k3 = ggml_reshape_3d(ctx, k, head_dim, num_kv_heads, nr);
            ggml_tensor* v3 = ggml_reshape_3d(ctx, v, head_dim, num_kv_heads, nr);
            k3 = ggml_mul(ctx, ggml_rms_norm(ctx, k3, eps), lt.k_norm_w);
            k3 = ggml_rope_ext(ctx, k3, pos_t, nullptr, head_dim, /*mode=*/2, 0,
                               rope_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
            // [hd, kv_heads, R, L] -> [hd, R, kv_heads, L]: each lane's rows contiguous.
            ggml_tensor* kw = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, k3, head_dim, num_kv_heads, R, L), 0, 2, 1, 3));
            ggml_tensor* vw = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v3, head_dim, num_kv_heads, R, L), 0, 2, 1, 3));
            for (int lane = 0; lane < L; lane++)
            {
                const std::size_t ix = static_cast<std::size_t>(lane) * num_layers + l;
                ring_k[ix] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(ring_dtype), head_dim, ring_rows, num_kv_heads);
                ring_v[ix] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(ring_dtype), head_dim, ring_rows, num_kv_heads);
                ggml_tensor* kl = ggml_view_3d(ctx, kw, head_dim, R, num_kv_heads, kw->nb[1], kw->nb[2],
                                               static_cast<std::size_t>(lane) * kw->nb[3]);
                ggml_tensor* vl = ggml_view_3d(ctx, vw, head_dim, R, num_kv_heads, vw->nb[1], vw->nb[2],
                                               static_cast<std::size_t>(lane) * vw->nb[3]);
                ggml_tensor* il = ggml_view_1d(ctx, idx_t, R, static_cast<std::size_t>(lane) * R * sizeof(std::int64_t));
                writes.push_back(ggml_set_rows(ctx, ring_k[ix], kl, il));
                writes.push_back(ggml_set_rows(ctx, ring_v[ix], vl, il));
            }
        }

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, static_cast<std::size_t>(num_layers) * (64 + 8 * L) + 256, false);
        for (ggml_tensor* w : writes)
            ggml_build_forward_expand(graph, w);

        DfBinder binder;
        binder.dev = ggml_backend_get_device(g_backend);
        binder.bind(fc_t, const_cast<void*>(fc_data), static_cast<std::size_t>(fc_bytes), true);
        binder.bind(enc_norm_t, const_cast<void*>(enc_norm_data),
                    static_cast<std::size_t>(hidden_size) * sizeof(float), true);
        const std::size_t ring_bytes = kv_cache_bytes(num_kv_heads, ring_rows, head_dim, ring_dtype);
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            binder.bind(lt.k_w, k_arr[l], static_cast<std::size_t>(k_bytes_arr[l]), true);
            binder.bind(lt.v_w, v_arr[l], static_cast<std::size_t>(v_bytes_arr[l]), true);
            binder.bind(lt.k_norm_w, k_norm_arr[l], static_cast<std::size_t>(head_dim) * sizeof(float), true);
            for (int lane = 0; lane < L; lane++)
            {
                const std::size_t ix = static_cast<std::size_t>(lane) * num_layers + l;
                binder.bind(ring_k[ix], ring_k_arr[ix], ring_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                binder.bind(ring_v[ix], ring_v_arr[ix], ring_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
        }

        ggml_backend_buffer_t persist_buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
        if (persist_buf == nullptr)
        {
            set_last_error("DFlash batched inject: failed to allocate persist buffer.");
            ggml_free(ctx);
            return 0;
        }
        binder.flush();

        slot->ctx = ctx; slot->buffer = persist_buf; slot->graph = graph;
        slot->feat = feat; slot->gather = gather_t; slot->pos = pos_t; slot->kv_index = idx_t;
        slot->sig = sig; slot->rings = std::move(rings);
        slot->max_rows = R; slot->lanes = L;
        slot->scale_signature = scale_sig;
        slot->lru = ++pool.clock;
        slot->valid = true;
        if (!run(*slot))
        {
            set_last_error("DFlash batched inject: graph execution failed.");
            slot->reset();
            return 0;
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in DFlash batched inject."); return 0; }
}

// ---------------------------------------------------------------------------
// PASS C: the block draft.
// ---------------------------------------------------------------------------
static int df_draft_block(
    const int* block_ids, int block_len, const int* positions,
    int num_layers, int hidden_size, int head_dim, int num_heads, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale, float kq_scale,
    // ring slot -> absolute position, or -1 when the slot is not live
    const int* ring_slot_pos, int sliding_window,
    void** attn_norm_arr,
    void** q_arr, int* q_type_arr, std::int64_t* q_ne0_arr, std::int64_t* q_ne1_arr, std::int64_t* q_bytes_arr,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr, int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    void** ffn_norm_arr,
    void** gate_arr, int* gate_type_arr, std::int64_t* gate_ne0_arr, std::int64_t* gate_ne1_arr, std::int64_t* gate_bytes_arr,
    void** up_arr, int* up_type_arr, std::int64_t* up_ne0_arr, std::int64_t* up_ne1_arr, std::int64_t* up_bytes_arr,
    void** down_arr, int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    void** ring_k_arr, void** ring_v_arr, int ring_dtype,
    const void* out_norm_data,
    const void* tok_embd_data, int tok_embd_type, std::int64_t tok_embd_ne0, std::int64_t tok_embd_ne1, std::int64_t tok_embd_bytes,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    int vocab_size, int* ids_out, float* conf_out,
    // ---- DFlash2 grouped dynamic convolution (conv_taps == 0 disables) ----
    int conv_taps, int conv_group_size, int conv_num_groups,
    void** attn_conv_base_arr,
    void** attn_conv_proj_arr, int* attn_conv_proj_type_arr,
    std::int64_t* attn_conv_proj_ne0_arr, std::int64_t* attn_conv_proj_ne1_arr, std::int64_t* attn_conv_proj_bytes_arr,
    void** ffn_conv_base_arr,
    void** ffn_conv_proj_arr, int* ffn_conv_proj_type_arr,
    std::int64_t* ffn_conv_proj_ne0_arr, std::int64_t* ffn_conv_proj_ne1_arr, std::int64_t* ffn_conv_proj_bytes_arr,
    // ---- DFlash2 candidate selector (sel_rank == 0 disables) ----
    int sel_rank, int sel_top_k, float sel_logit_scale, float sel_logit_softcap,
    const void* sel_hidden_data, int sel_hidden_type, std::int64_t sel_hidden_ne0, std::int64_t sel_hidden_ne1, std::int64_t sel_hidden_bytes,
    const void* sel_pred_data, int sel_pred_type, std::int64_t sel_pred_ne0, std::int64_t sel_pred_ne1, std::int64_t sel_pred_bytes,
    const void* sel_succ_data, int sel_succ_type, std::int64_t sel_succ_ne0, std::int64_t sel_succ_ne1, std::int64_t sel_succ_bytes,
    float* sel_scores_out, int* sel_cand_out,
    const float* scales, int scale_count)
{
    try
    {
        if (!ensure_backend()) return 0;
        if ((scales == nullptr && scale_count != 0) ||
            (scales != nullptr && (num_layers <= 0 || scale_count != 6 + 9 * num_layers)))
        { set_last_error("DFlash: invalid sidecar scale table."); return 0; }
        const std::uint64_t scale_sig = df_scale_signature(scales, scale_count);
        if (block_len <= 0 || num_layers <= 0 || ids_out == nullptr || conf_out == nullptr)
        { set_last_error("DFlash draft: invalid arguments."); return 0; }

        const int b = block_len;
        // The block at its own ring slots (df_ring_block) needs the ring's headroom past the window for a whole block.
        // A block slot last held position p - ring_rows for a block position p; the nearest block query sees it only
        // when ring_rows - (b - 1) < the window.
        const bool ring_block = df_ring_block() && sliding_window > 0 && b < ring_rows
            && ring_rows - (b - 1) >= sliding_window;
        const int kv_len = ring_block ? ring_rows : ring_rows + b;      // FIXED across steps -> capturable
        const int q_dim = num_heads * head_dim;
        const int out_count = b;

        const bool use_conv = conv_taps > 0 && conv_group_size > 0 && conv_num_groups > 0
            && attn_conv_base_arr != nullptr && attn_conv_proj_arr != nullptr
            && ffn_conv_base_arr != nullptr && ffn_conv_proj_arr != nullptr;
        const bool use_selector = sel_rank > 0 && sel_top_k > 0
            && sel_hidden_data != nullptr && sel_pred_data != nullptr && sel_succ_data != nullptr
            && sel_scores_out != nullptr && sel_cand_out != nullptr;
        const int gamma = b - 1;               // proposal rows (row 0 is the anchor)
        if (use_selector && (gamma < 1 || sel_top_k > vocab_size))
        {
            set_last_error("DFlash draft: selector needs at least one proposal row and top_k <= vocab.");
            return 0;
        }
        // The head, top-k and selector lattice on TensorSharp's Metal kernels after the graph (tsm::df_tail_run).
#if defined(TSG_GGML_USE_METAL)
        const bool use_tail = use_selector && g_backend_type == BACKEND_TYPE_METAL && lm_head_ne0 == hidden_size
            && lm_head_ne1 == vocab_size && sel_pred_ne0 == sel_rank && sel_succ_ne0 == sel_rank
            && sel_pred_ne1 == vocab_size && sel_succ_ne1 == vocab_size
            && tsm::df_tail_supported(g_backend, lm_head_type, hidden_size, vocab_size, sel_rank, sel_top_k,
                                      sel_pred_type, sel_succ_type, gamma);
#else
        const bool use_tail = false;
#endif
        if (use_conv && conv_taps > b)
        {
            set_last_error("DFlash draft: conv_kernel_size exceeds the block width.");
            return 0;
        }
        const std::size_t sel_scores_floats = use_selector
            ? static_cast<std::size_t>(sel_top_k)
              + static_cast<std::size_t>(sel_top_k) * sel_top_k * (gamma - 1)
            : 0;

        const void* sig = attn_norm_arr[0];
        const void* sig_ring = ring_k_arr[0];
        // Metal is included for the same reason as the inject graph above.
        const bool can_persist =
            g_backend_type == BACKEND_TYPE_CUDA || g_backend_type == BACKEND_TYPE_VULKAN ||
            g_backend_type == BACKEND_TYPE_METAL;

        // Mask [kv_len, b]: ring slot s holds ring_slot_pos[s] (or -1 = dead);
        // it is visible to block query i (position positions[i]) when the slot is
        // live, not in the future, and inside the sliding window. The block's own
        // b columns are always visible - DFlash drafts non-causally within a block.
        std::vector<ggml_fp16_t> mask_data(static_cast<std::size_t>(kv_len) * b,
                                           ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        const ggml_fp16_t vis = static_cast<ggml_fp16_t>(0);
        const int anchor_pos = positions[0];
        for (int i = 0; i < b; i++)
        {
            const int qp = positions[i];
            ggml_fp16_t* row = &mask_data[static_cast<std::size_t>(i) * kv_len];
            for (int s = 0; s < ring_rows; s++)
            {
                const int p = ring_slot_pos[s];
                // Cutoff is the ANCHOR's position, not this query's. After a
                // partially-rejected draft the ring still holds keys the drafter
                // wrote for positions past the anchor; those rows are stale and no
                // query in the block may see them, even the later ones whose own
                // position is higher. The per-op path gets this for free by only
                // ever gathering [winStart, anchor).
                if (p < 0 || p >= anchor_pos) continue;
                if (sliding_window > 0 && qp - p >= sliding_window) continue;
                row[s] = vis;
            }
            for (int j = 0; j < b; j++)
                row[ring_block ? positions[j] % ring_rows : ring_rows + j] = vis;
        }
        // The block's ring slots (df_ring_block), written by every layer before its attention.
        std::vector<std::int64_t> block_slots(ring_block ? b : 0);
        for (int j = 0; j < static_cast<int>(block_slots.size()); j++)
            block_slots[j] = positions[j] % ring_rows;

        // The tail after the graph (use_tail): it runs behind the graph on the same queue and waits for itself.
        auto run_tail = [&](ggml_tensor* tail_h, ggml_tensor* tail_ph, ggml_tensor* head_t, ggml_tensor* pred_t,
                            ggml_tensor* succ_t) -> bool {
#if defined(TSG_GGML_USE_METAL)
            tsm::DfTailIO io;
            io.h = tail_h; io.ph = tail_ph; io.head = head_t; io.pred = pred_t; io.succ = succ_t;
            io.gamma = gamma; io.anchor = block_ids[0];
            io.head_scale = scales != nullptr ? scales[2] : 1.0f;
            io.pred_scale = scales != nullptr ? scales[4] : 1.0f;
            io.succ_scale = scales != nullptr ? scales[5] : 1.0f;
            io.logit_scale = sel_logit_scale; io.softcap = sel_logit_softcap;
            io.cand_out = sel_cand_out; io.scores_out = sel_scores_out;
            return tsm::df_tail_run(g_backend, io);
#else
            (void)tail_h; (void)tail_ph; (void)head_t; (void)pred_t; (void)succ_t;
            return false;
#endif
        };

        DfCache* dc = can_persist ? df_draft_pool().find(sig, sig_ring, b, scale_sig) : nullptr;
        if (dc != nullptr && ((dc->kv_index != nullptr) != ring_block || (dc->tail_h != nullptr) != use_tail))
        {
            dc->reset();   // built under the other TS_DFLASH_RING_BLOCK / TS_METAL_DF_TAIL setting
            dc = nullptr;
        }
        if (dc != nullptr && dc->graph != nullptr && dc->out_count == out_count)
        {
            host_read_barrier();
            decode_input_set_async(dc->in_main, block_ids, static_cast<std::size_t>(b) * sizeof(std::int32_t));
            decode_input_set_async(dc->pos, positions, static_cast<std::size_t>(b) * sizeof(std::int32_t));
            decode_input_set_async(dc->mask, mask_data.data(), mask_data.size() * sizeof(ggml_fp16_t));
            if (dc->kv_index != nullptr)
                decode_input_set_async(dc->kv_index, block_slots.data(), block_slots.size() * sizeof(std::int64_t));
            if (tsg::graph_compute_profiled(g_backend, dc->graph, "dflash draft") != GGML_STATUS_SUCCESS)
            {
                set_last_error("DFlash draft: cached graph execution failed.");
                dc->reset();
                return 0;
            }
            if (dc->tail_h != nullptr)
            {
                if (!run_tail(dc->tail_h, dc->tail_ph, dc->tail_head, dc->tail_pred, dc->tail_succ))
                {
                    dc->reset();
                    return 0;
                }
                clear_last_error();
                return 1;
            }
            // Metal's graph_compute is async and a shared-buffer tensor_get is a
            // raw memcpy, so the draft ids must not be read until the queue
            // drains - otherwise the executor verifies stale drafts (correct
            // output, silently collapsed acceptance).
            if (g_backend_type == BACKEND_TYPE_METAL)
                tsg::sync_backend(g_backend);
            if (use_selector)
            {
                ggml_backend_tensor_get(dc->out_cand, sel_cand_out, 0,
                    static_cast<std::size_t>(sel_top_k) * gamma * sizeof(std::int32_t));
                ggml_backend_tensor_get(dc->out_s0, sel_scores_out, 0,
                    static_cast<std::size_t>(sel_top_k) * sizeof(float));
                if (dc->out_scores != nullptr)
                {
                    finalize_compute_with_download(dc->out_scores, sel_scores_out + sel_top_k,
                        (sel_scores_floats - sel_top_k) * sizeof(float));
                }
            }
            else
            {
                ggml_backend_tensor_get(dc->out, ids_out, 0, static_cast<std::size_t>(b) * sizeof(std::int32_t));
                finalize_compute_with_download(dc->out_conf, conf_out, static_cast<std::size_t>(b) * sizeof(float));
            }
            host_read_barrier();
            clear_last_error();
            return 1;
        }

        DfCache* slot = can_persist ? &df_draft_pool().claim(sig, sig_ring, b, scale_sig) : nullptr;

        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle pooled;
        ggml_context* ctx = nullptr;
        if (can_persist)
        {
            ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
            ctx = ggml_init(ip);
            if (ctx == nullptr) { set_last_error("DFlash draft: failed to init context."); return 0; }
        }
        else
        {
            if (!pooled.init(ctx_size)) { set_last_error("DFlash draft: failed to create context."); return 0; }
            ctx = pooled.value;
        }

        ggml_tensor* ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, b);
        ggml_tensor* pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, b);
        ggml_tensor* mask_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv_len, b, 1, 1);
        ggml_set_input(ids_t); ggml_set_input(pos_t); ggml_set_input(mask_t);
        ggml_tensor* block_slot_t = nullptr;
        if (ring_block)
        {
            block_slot_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, b);
            ggml_set_input(block_slot_t);
        }

        // Constant per-tap row shift and block-boundary mask for the DFlash2
        // convolution. They depend only on the block width, which is part of the
        // cache key, so they are written once at build time and survive replay.
        std::vector<ggml_tensor*> conv_shift(use_conv ? conv_taps : 0, nullptr);
        std::vector<ggml_tensor*> conv_mask(use_conv ? conv_taps : 0, nullptr);
        std::vector<std::vector<std::int32_t>> conv_shift_data(use_conv ? conv_taps : 0);
        std::vector<std::vector<float>> conv_mask_data(use_conv ? conv_taps : 0);
        // Metal takes every tap at once (df_grouped_conv_all_taps): the same shifts and masks, tap-major, and no
        // per-tap inputs (an input no node reads is never allocated).
        const bool conv_all_taps = use_conv && conv_taps >= 2 && df_metal_fused_conv();
        ggml_tensor* conv_shift_all = nullptr;
        ggml_tensor* conv_mask_all = nullptr;
        std::vector<std::int32_t> conv_shift_all_data;
        std::vector<float> conv_mask_all_data;
        if (conv_all_taps)
        {
            conv_shift_all = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<std::int64_t>(b) * conv_taps);
            conv_mask_all = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, b, conv_taps);
            ggml_set_input(conv_shift_all);
            ggml_set_input(conv_mask_all);
            conv_shift_all_data.resize(static_cast<std::size_t>(b) * conv_taps);
            conv_mask_all_data.resize(static_cast<std::size_t>(b) * conv_taps);
            for (int tap = 0; tap < conv_taps; tap++)
                for (int r = 0; r < b; r++)
                {
                    conv_shift_all_data[static_cast<std::size_t>(tap) * b + r] = r >= tap ? r - tap : 0;
                    conv_mask_all_data[static_cast<std::size_t>(tap) * b + r] = r >= tap ? 1.0f : 0.0f;
                }
        }
        for (int tap = 1; tap < (use_conv && !conv_all_taps ? conv_taps : 0); tap++)
        {
            conv_shift[tap] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, b);
            conv_mask[tap] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, b);
            ggml_set_input(conv_shift[tap]);
            ggml_set_input(conv_mask[tap]);
            conv_shift_data[tap].resize(b);
            conv_mask_data[tap].resize(b);
            for (int r = 0; r < b; r++)
            {
                conv_shift_data[tap][r] = r >= tap ? r - tap : 0;
                conv_mask_data[tap][r] = r >= tap ? 1.0f : 0.0f;
            }
        }

        ggml_tensor* tok_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(tok_embd_type), tok_embd_ne0, tok_embd_ne1);
        ggml_tensor* out_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
        ggml_tensor* sel_hidden_t = nullptr;
        ggml_tensor* sel_pred_t = nullptr;
        ggml_tensor* sel_succ_t = nullptr;
        if (use_selector)
        {
            sel_hidden_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(sel_hidden_type), sel_hidden_ne0, sel_hidden_ne1);
            sel_pred_t   = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(sel_pred_type),   sel_pred_ne0,   sel_pred_ne1);
            sel_succ_t   = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(sel_succ_type),   sel_succ_ne0,   sel_succ_ne1);
        }

        // llama.cpp's dflash graph feeds build_inp_embd straight in: no embedding
        // scale and (unlike the Muse-Glimmer trunk) no weightless input RMSNorm.
        ggml_tensor* inpL = df_scaled(ctx, ggml_get_rows(ctx, tok_t, ids_t), scales, 1);              // [hidden, b]

        std::vector<DfLayer> layers(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            lt.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.ffn_norm_w  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.q_norm_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            lt.k_norm_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            lt.q_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(q_type_arr[l]),    q_ne0_arr[l],    q_ne1_arr[l]);
            lt.k_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(k_type_arr[l]),    k_ne0_arr[l],    k_ne1_arr[l]);
            lt.v_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(v_type_arr[l]),    v_ne0_arr[l],    v_ne1_arr[l]);
            lt.o_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type_arr[l]),    o_ne0_arr[l],    o_ne1_arr[l]);
            lt.gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gate_type_arr[l]), gate_ne0_arr[l], gate_ne1_arr[l]);
            lt.up_w   = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(up_type_arr[l]),   up_ne0_arr[l],   up_ne1_arr[l]);
            lt.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type_arr[l]), down_ne0_arr[l], down_ne1_arr[l]);
            lt.ring_k = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(ring_dtype), head_dim, ring_rows, num_kv_heads);
            lt.ring_v = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(ring_dtype), head_dim, ring_rows, num_kv_heads);
            if (use_conv)
            {
                lt.attn_conv_base = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, conv_taps, 2);
                lt.ffn_conv_base = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, conv_taps, 2);
                lt.attn_conv_proj = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(attn_conv_proj_type_arr[l]),
                                                       attn_conv_proj_ne0_arr[l], attn_conv_proj_ne1_arr[l]);
                lt.ffn_conv_proj = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ffn_conv_proj_type_arr[l]),
                                                      ffn_conv_proj_ne0_arr[l], ffn_conv_proj_ne1_arr[l]);
            }
        }

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];

            ggml_tensor* h = ggml_mul(ctx, ggml_rms_norm(ctx, inpL, eps), lt.attn_norm_w);

            // DFlash2: one projection of the sublayer input carries both filters,
            // so the output-side coefficients are computed here and held across the
            // attention - they are keyed on the INPUT, not on what attention made.
            ggml_tensor* attn_conv_coef = nullptr;
            if (use_conv)
            {
                attn_conv_coef = df_scaled(ctx, ggml_mul_mat(ctx, lt.attn_conv_proj, h), scales, 6 + 9 * l + 7);   // [2*taps*G, b]
                h = df_grouped_conv(ctx, h, attn_conv_coef, lt.attn_conv_base, /*side=*/0,
                                    conv_taps, conv_num_groups, conv_group_size, hidden_size, b,
                                    conv_shift.data(), conv_mask.data(), conv_shift_all, conv_mask_all);
            }

            ggml_tensor* q = df_scaled(ctx, ggml_mul_mat(ctx, lt.q_w, h), scales, 6 + 9 * l + 0);
            ggml_tensor* k = df_scaled(ctx, ggml_mul_mat(ctx, lt.k_w, h), scales, 6 + 9 * l + 1);
            ggml_tensor* v = df_scaled(ctx, ggml_mul_mat(ctx, lt.v_w, h), scales, 6 + 9 * l + 2);

            ggml_tensor* q3 = ggml_reshape_3d(ctx, q, head_dim, num_heads, b);
            ggml_tensor* k3 = ggml_reshape_3d(ctx, k, head_dim, num_kv_heads, b);
            ggml_tensor* v3 = ggml_reshape_3d(ctx, v, head_dim, num_kv_heads, b);
            q3 = ggml_mul(ctx, ggml_rms_norm(ctx, q3, eps), lt.q_norm_w);
            k3 = ggml_mul(ctx, ggml_rms_norm(ctx, k3, eps), lt.k_norm_w);
            q3 = ggml_rope_ext(ctx, q3, pos_t, nullptr, head_dim, /*NeoX=*/2, 0,
                               rope_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
            k3 = ggml_rope_ext(ctx, k3, pos_t, nullptr, head_dim, /*NeoX=*/2, 0,
                               rope_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);

            // [ring | this block's own keys]; the ring is read whole and the mask
            // decides which slots are live (attention is order-invariant over KV).
            // With df_ring_block the block's keys go into the ring at their own slots instead.
            ggml_tensor* kcat;
            ggml_tensor* vcat;
            if (ring_block)
            {
                kcat = ggml_set_rows(ctx, lt.ring_k, ggml_permute(ctx, k3, 0, 2, 1, 3), block_slot_t);
                vcat = ggml_set_rows(ctx, lt.ring_v, ggml_permute(ctx, v3, 0, 2, 1, 3), block_slot_t);
            }
            else
            {
                ggml_tensor* kb = ggml_cont(ctx, ggml_permute(ctx, k3, 0, 2, 1, 3));   // [hd, b, kv_heads]
                ggml_tensor* vb = ggml_cont(ctx, ggml_permute(ctx, v3, 0, 2, 1, 3));
                kcat = ggml_concat(ctx, lt.ring_k, kb, 1);
                vcat = ggml_concat(ctx, lt.ring_v, vb, 1);
            }

            ggml_tensor* qa = ggml_permute(ctx, q3, 0, 2, 1, 3);                   // [hd, b, heads]
            ggml_tensor* attn_flat;
            ggml_tensor* fa = flash_attn_ext_guarded(ctx, "DFlash draft", qa, kcat, vcat, mask_t,
                kq_scale, 0.0f, 0.0f, nullptr, GGML_PREC_F32);
            attn_flat = ggml_reshape_2d(ctx, fa, q_dim, b);

            ggml_tensor* attn_out = df_scaled(ctx, ggml_mul_mat(ctx, lt.o_w, attn_flat), scales, 6 + 9 * l + 3);
            if (use_conv)
            {
                attn_out = df_grouped_conv(ctx, attn_out, attn_conv_coef, lt.attn_conv_base, /*side=*/1,
                                           conv_taps, conv_num_groups, conv_group_size, hidden_size, b,
                                           conv_shift.data(), conv_mask.data(), conv_shift_all, conv_mask_all);
            }
            ggml_tensor* ffn_inp = ggml_add(ctx, attn_out, inpL);

            ggml_tensor* fh = ggml_mul(ctx, ggml_rms_norm(ctx, ffn_inp, eps), lt.ffn_norm_w);
            ggml_tensor* ffn_conv_coef = nullptr;
            if (use_conv)
            {
                ffn_conv_coef = df_scaled(ctx, ggml_mul_mat(ctx, lt.ffn_conv_proj, fh), scales, 6 + 9 * l + 8);
                fh = df_grouped_conv(ctx, fh, ffn_conv_coef, lt.ffn_conv_base, /*side=*/0,
                                     conv_taps, conv_num_groups, conv_group_size, hidden_size, b,
                                     conv_shift.data(), conv_mask.data(), conv_shift_all, conv_mask_all);
            }
            ggml_tensor* gate = df_scaled(ctx, ggml_mul_mat(ctx, lt.gate_w, fh), scales, 6 + 9 * l + 4);
            ggml_tensor* up   = df_scaled(ctx, ggml_mul_mat(ctx, lt.up_w, fh), scales, 6 + 9 * l + 5);
            ggml_tensor* act  = ggml_mul(ctx, ggml_silu(ctx, gate), up);
            ggml_tensor* down = df_scaled(ctx, ggml_mul_mat(ctx, lt.down_w, act), scales, 6 + 9 * l + 6);
            if (use_conv)
            {
                down = df_grouped_conv(ctx, down, ffn_conv_coef, lt.ffn_conv_base, /*side=*/1,
                                       conv_taps, conv_num_groups, conv_group_size, hidden_size, b,
                                       conv_shift.data(), conv_mask.data(), conv_shift_all, conv_mask_all);
            }
            inpL = ggml_add(ctx, down, ffn_inp);
        }

        ggml_tensor* cur = ggml_mul(ctx, ggml_rms_norm(ctx, inpL, eps), out_norm_t);

        ggml_tensor* sel_s0 = nullptr;
        ggml_tensor* sel_scores = nullptr;
        ggml_tensor* sel_cand = nullptr;
        ggml_tensor* tail_h = nullptr;
        ggml_tensor* tail_ph = nullptr;
        if (use_tail)
        {
            // The graph ends at the final rows and the selector's hidden projection; tsm::df_tail_run reads them.
            ggml_tensor* pred_h = ggml_view_2d(ctx, cur, hidden_size, gamma, cur->nb[1], cur->nb[1]);
            tail_ph = df_scaled(ctx, ggml_mul_mat(ctx, sel_hidden_t, pred_h), scales, 3);   // [r, gamma]
            tail_h = cur;
            ggml_set_output(tail_h);
            ggml_set_output(tail_ph);
        }
        else if (use_selector)
        {
            // Only the gamma PROPOSAL rows reach the head; row 0 is the anchor's
            // own prediction, which the selector never consumes.
            ggml_tensor* pred_h = ggml_view_2d(ctx, cur, hidden_size, gamma, cur->nb[1], cur->nb[1]);
            ggml_tensor* sel_logits = df_scaled(ctx, ggml_mul_mat(ctx, lm_head_t, pred_h), scales, 2);        // [vocab, gamma]
            sel_cand = ggml_top_k(ctx, sel_logits, sel_top_k);                     // [k, gamma] I32
            if (!backend_supports_op(sel_cand))
            {
                set_last_error("DFlash draft: this backend has no top-k over the vocabulary.");
                if (can_persist) ggml_free(ctx);
                return 0;
            }
            ggml_set_output(sel_cand);

            // unary[e][c]: the head logit of candidate c at position e.
            ggml_tensor* logits3 = ggml_reshape_3d(ctx, sel_logits, 1, vocab_size, gamma);
            ggml_tensor* unary = ggml_get_rows(ctx, logits3, sel_cand);            // [1, k, gamma]
            ggml_tensor* unary_kg = ggml_reshape_3d(ctx, unary, sel_top_k, 1, gamma);
            // The target's LM-head transform. Applied here, after the top-k, because
            // both halves are monotonic (the candidate set cannot change) and this
            // touches k*gamma values instead of vocab*gamma. Without it the unary
            // term enters the lattice at the wrong scale and swamps the transition
            // scores it is meant to compete with.
            if (sel_logit_scale != 1.0f)
                unary_kg = ggml_scale(ctx, unary_kg, sel_logit_scale);
            if (sel_logit_softcap > 0.0f)
            {
                unary_kg = ggml_scale(ctx,
                    ggml_tanh(ctx, ggml_scale(ctx, unary_kg, 1.0f / sel_logit_softcap)),
                    sel_logit_softcap);
            }

            ggml_tensor* ph = df_scaled(ctx, ggml_mul_mat(ctx, sel_hidden_t, pred_h), scales, 3);             // [r, gamma]
            ggml_tensor* cand_flat = ggml_reshape_1d(ctx, sel_cand, static_cast<std::int64_t>(sel_top_k) * gamma);
            ggml_tensor* keys = df_scaled(ctx, ggml_get_rows(ctx, sel_succ_t, cand_flat), scales, 5);         // [r, k*gamma]
            ggml_tensor* keys3 = ggml_reshape_3d(ctx, keys, sel_rank, sel_top_k, gamma);

            // Position 0's predecessor is the verified anchor, which is block_ids[0]
            // - the same tensor the embedding lookup already reads.
            ggml_tensor* anchor_id = ggml_view_1d(ctx, ids_t, 1, 0);
            ggml_tensor* a0 = df_scaled(ctx, ggml_get_rows(ctx, sel_pred_t, anchor_id), scales, 4);           // [r, 1]
            ggml_tensor* ph0 = ggml_view_2d(ctx, ph, sel_rank, 1, ph->nb[1], 0);
            ggml_tensor* m0 = ggml_mul(ctx, a0, ph0);                              // [r, 1]
            ggml_tensor* keys0 = ggml_view_2d(ctx, keys3, sel_rank, sel_top_k, keys3->nb[1], 0);
            sel_s0 = ggml_add(ctx, ggml_mul_mat(ctx, keys0, m0),
                              ggml_view_2d(ctx, unary_kg, sel_top_k, 1, unary_kg->nb[1], 0));
            ggml_set_output(sel_s0);                                               // [k, 1]

            if (gamma > 1)
            {
                // Position e's predecessors are position e-1's candidates, so the
                // predecessor ids are the same tensor shifted by one slot.
                ggml_tensor* prev_flat = ggml_view_1d(ctx, sel_cand,
                    static_cast<std::int64_t>(sel_top_k) * (gamma - 1), 0);
                ggml_tensor* preds = df_scaled(ctx, ggml_get_rows(ctx, sel_pred_t, prev_flat), scales, 4);
                ggml_tensor* preds3 = ggml_reshape_3d(ctx, preds, sel_rank, sel_top_k, gamma - 1);
                ggml_tensor* ph_rest = ggml_view_3d(ctx, ph, sel_rank, 1, gamma - 1,
                                                    ph->nb[1], ph->nb[1], ph->nb[1]);
                ggml_tensor* m = ggml_mul(ctx, preds3, ph_rest);                   // [r, k, gamma-1]
                ggml_tensor* keys_rest = ggml_view_3d(ctx, keys3, sel_rank, sel_top_k, gamma - 1,
                                                      keys3->nb[1], keys3->nb[2], keys3->nb[2]);
                // [k(candidate), k(predecessor), gamma-1] -- candidate fastest, which
                // is the row the host walk scans.
                sel_scores = ggml_mul_mat(ctx, keys_rest, m);
                ggml_tensor* u_rest = ggml_view_3d(ctx, unary_kg, sel_top_k, 1, gamma - 1,
                                                   unary_kg->nb[1], unary_kg->nb[2], unary_kg->nb[2]);
                sel_scores = ggml_add(ctx, sel_scores, u_rest);
                ggml_set_output(sel_scores);
            }
        }

        // The TARGET's LM head, with NEITHER logit_scale NOR the tanh softcap:
        // llama.cpp's dflash graph ends at build_lora_mm(output, cur).
        ggml_tensor* logits = use_selector ? nullptr : df_scaled(ctx, ggml_mul_mat(ctx, lm_head_t, cur), scales, 2);        // [vocab, b]
        // Softmax on device: argmax is invariant under it, and the winning
        // probability IS the confidence the executor multiplies cumulatively.
        ggml_tensor* probs = use_selector ? nullptr : ggml_soft_max(ctx, logits);

        // Reduce to (argmax id, winning probability) ON DEVICE. llama.cpp pulls the
        // whole [vocab, b] block back every draft step -- 202048*16*4 = 12.9 MB of
        // PCIe traffic plus a 3.2 M-element host scan -- and that readback is a
        // large share of its per-step cost. Two b-element tensors carry everything
        // the caller needs: argmax is invariant under softmax, and the winning
        // probability IS the confidence the executor multiplies cumulatively.
        ggml_tensor* out = nullptr;
        ggml_tensor* out_conf = nullptr;
        ggml_tensor* out_node = nullptr;
        ggml_tensor* conf_node = nullptr;
        if (!use_selector)
        {
            ggml_tensor* am = ggml_argmax(ctx, probs);                            // [b] I32
            ggml_tensor* am2 = ggml_reshape_2d(ctx, am, 1, b);                    // [1, b]
            ggml_tensor* pr3 = ggml_reshape_3d(ctx, probs, 1, vocab_size, b);     // [1, vocab, b]
            ggml_tensor* mp = ggml_get_rows(ctx, pr3, am2);                       // [1, 1, b] F32
            out = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, b);
            out_conf = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, b);
            out_node = ggml_cpy(ctx, am, out);
            conf_node = ggml_cpy(ctx, ggml_reshape_1d(ctx, mp, b), out_conf);
            ggml_set_output(out_node);
            ggml_set_output(conf_node);
        }

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, static_cast<std::size_t>(num_layers) * 256 + 1024, false);
        if (use_tail)
        {
            ggml_build_forward_expand(graph, tail_h);
            ggml_build_forward_expand(graph, tail_ph);
        }
        else if (use_selector)
        {
            ggml_build_forward_expand(graph, sel_cand);
            ggml_build_forward_expand(graph, sel_s0);
            if (sel_scores != nullptr)
                ggml_build_forward_expand(graph, sel_scores);
        }
        else
        {
            ggml_build_forward_expand(graph, out_node);
            ggml_build_forward_expand(graph, conf_node);
        }

        DfBinder binder;
        binder.dev = ggml_backend_get_device(g_backend);
        const std::size_t norm_bytes = static_cast<std::size_t>(hidden_size) * sizeof(float);
        const std::size_t head_norm_bytes = static_cast<std::size_t>(head_dim) * sizeof(float);
        const std::size_t ring_bytes = kv_cache_bytes(num_kv_heads, ring_rows, head_dim, ring_dtype);
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            binder.bind(lt.q_w,    q_arr[l],    static_cast<std::size_t>(q_bytes_arr[l]), true);
            binder.bind(lt.k_w,    k_arr[l],    static_cast<std::size_t>(k_bytes_arr[l]), true);
            binder.bind(lt.v_w,    v_arr[l],    static_cast<std::size_t>(v_bytes_arr[l]), true);
            binder.bind(lt.o_w,    o_arr[l],    static_cast<std::size_t>(o_bytes_arr[l]), true);
            binder.bind(lt.gate_w, gate_arr[l], static_cast<std::size_t>(gate_bytes_arr[l]), true);
            binder.bind(lt.up_w,   up_arr[l],   static_cast<std::size_t>(up_bytes_arr[l]), true);
            binder.bind(lt.down_w, down_arr[l], static_cast<std::size_t>(down_bytes_arr[l]), true);
            binder.bind(lt.attn_norm_w, attn_norm_arr[l], norm_bytes, true);
            binder.bind(lt.ffn_norm_w,  ffn_norm_arr[l],  norm_bytes, true);
            binder.bind(lt.q_norm_w,    q_norm_arr[l],    head_norm_bytes, true);
            binder.bind(lt.k_norm_w,    k_norm_arr[l],    head_norm_bytes, true);
            binder.bind(lt.ring_k, ring_k_arr[l], ring_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            binder.bind(lt.ring_v, ring_v_arr[l], ring_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            if (use_conv)
            {
                const std::size_t conv_base_bytes =
                    2u * static_cast<std::size_t>(conv_taps) * hidden_size * sizeof(float);
                binder.bind(lt.attn_conv_base, attn_conv_base_arr[l], conv_base_bytes, true);
                binder.bind(lt.ffn_conv_base, ffn_conv_base_arr[l], conv_base_bytes, true);
                binder.bind(lt.attn_conv_proj, attn_conv_proj_arr[l],
                            static_cast<std::size_t>(attn_conv_proj_bytes_arr[l]), true);
                binder.bind(lt.ffn_conv_proj, ffn_conv_proj_arr[l],
                            static_cast<std::size_t>(ffn_conv_proj_bytes_arr[l]), true);
            }
        }
        if (use_selector)
        {
            binder.bind(sel_hidden_t, const_cast<void*>(sel_hidden_data),
                        static_cast<std::size_t>(sel_hidden_bytes), true);
            binder.bind(sel_pred_t, const_cast<void*>(sel_pred_data),
                        static_cast<std::size_t>(sel_pred_bytes), true);
            binder.bind(sel_succ_t, const_cast<void*>(sel_succ_data),
                        static_cast<std::size_t>(sel_succ_bytes), true);
        }
        binder.bind(out_norm_t, const_cast<void*>(out_norm_data), norm_bytes, true);
        binder.bind(tok_t, const_cast<void*>(tok_embd_data), static_cast<std::size_t>(tok_embd_bytes), true);
        binder.bind(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);

        // Metal: reorder for encoder concurrency before allocation (no-op on
        // other backends).
        optimize_graph_for_metal(ctx, graph);

        ggml_gallocr_t persist_galloc = nullptr;
        if (can_persist)
        {
            if (!df_alloc_persistent(ctx, graph, persist_galloc))
            {
                set_last_error("DFlash draft: failed to allocate persist buffer.");
                ggml_free(ctx);
                return 0;
            }
        }
        else
        {
            for (auto& u : binder.uploads) ggml_set_input(u.t);
            if (!alloc_graph_reuse_gallocr(graph))
            {
                set_last_error("DFlash draft: failed to allocate graph.");
                return 0;
            }
        }

        host_read_barrier();
        binder.flush();
        ggml_backend_tensor_set(ids_t, block_ids, 0, static_cast<std::size_t>(b) * sizeof(std::int32_t));
        ggml_backend_tensor_set(pos_t, positions, 0, static_cast<std::size_t>(b) * sizeof(std::int32_t));
        ggml_backend_tensor_set(mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
        if (block_slot_t != nullptr)
            ggml_backend_tensor_set(block_slot_t, block_slots.data(), 0, block_slots.size() * sizeof(std::int64_t));
        for (int tap = 1; tap < (use_conv && !conv_all_taps ? conv_taps : 0); tap++)
        {
            ggml_backend_tensor_set(conv_shift[tap], conv_shift_data[tap].data(), 0,
                static_cast<std::size_t>(b) * sizeof(std::int32_t));
            ggml_backend_tensor_set(conv_mask[tap], conv_mask_data[tap].data(), 0,
                static_cast<std::size_t>(b) * sizeof(float));
        }
        if (conv_all_taps)
        {
            ggml_backend_tensor_set(conv_shift_all, conv_shift_all_data.data(), 0,
                conv_shift_all_data.size() * sizeof(std::int32_t));
            ggml_backend_tensor_set(conv_mask_all, conv_mask_all_data.data(), 0,
                conv_mask_all_data.size() * sizeof(float));
        }

        if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("DFlash draft: graph execution failed.");
            if (can_persist) { ggml_gallocr_free(persist_galloc); ggml_free(ctx); }
            return 0;
        }
        if (use_tail)
        {
            if (!run_tail(tail_h, tail_ph, lm_head_t, sel_pred_t, sel_succ_t))
            {
                if (can_persist) { ggml_gallocr_free(persist_galloc); ggml_free(ctx); }
                return 0;
            }
        }
        else
        {
            // See the replay path above: Metal must drain before reading the ids.
            if (g_backend_type == BACKEND_TYPE_METAL)
                tsg::sync_backend(g_backend);
            if (use_selector)
            {
                ggml_backend_tensor_get(sel_cand, sel_cand_out, 0,
                    static_cast<std::size_t>(sel_top_k) * gamma * sizeof(std::int32_t));
                ggml_backend_tensor_get(sel_s0, sel_scores_out, 0,
                    static_cast<std::size_t>(sel_top_k) * sizeof(float));
                if (sel_scores != nullptr)
                {
                    finalize_compute_with_download(sel_scores, sel_scores_out + sel_top_k,
                        (sel_scores_floats - sel_top_k) * sizeof(float));
                }
            }
            else
            {
                ggml_backend_tensor_get(out, ids_out, 0, static_cast<std::size_t>(b) * sizeof(std::int32_t));
                finalize_compute_with_download(out_conf, conf_out, static_cast<std::size_t>(b) * sizeof(float));
            }
            // Unconditional: the outputs above land in caller host arrays and on Metal
            // async mode the download is only QUEUED.
            host_read_barrier();
        }

        if (can_persist && slot != nullptr)
        {
            slot->ctx = ctx; slot->galloc = persist_galloc; slot->graph = graph;
            slot->in_main = ids_t; slot->pos = pos_t; slot->mask = mask_t; slot->kv_index = block_slot_t;
            slot->out = out; slot->out_conf = out_conf;
            slot->out_s0 = sel_s0; slot->out_scores = sel_scores; slot->out_cand = sel_cand;
            slot->tail_h = tail_h; slot->tail_ph = tail_ph;
            if (use_tail) { slot->tail_head = lm_head_t; slot->tail_pred = sel_pred_t; slot->tail_succ = sel_succ_t; }
            slot->sig = sig; slot->sig_ring = sig_ring; slot->n_rows = b; slot->out_count = out_count;
            slot->valid = true;
            slot->scale_signature = scale_sig;
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in DFlash draft."); return 0; }
}

// ---------------------------------------------------------------------------
// PASS C for SEVERAL sequences at once: one graph drafts a block for each of
// n_lanes sequences, each attending to its own ring. The batched speculative
// verify (ggml_ops_qwen35_batched_verify.cpp) serves every decoding request in
// one trunk forward; drafting those requests one graph at a time then cost
// 2.7 ms per request per step - 10.8 of a 37 ms four-request step - mostly the
// drafter's and LM head's weight sweep repeated per request. Here projections,
// FFNs, the LM head and the selector run once over every lane's block.
//
// Inputs and outputs are lane-major: block_ids/positions [n_lanes * b],
// ring_slot_pos [n_lanes * ring_rows], ring_k/v_arr [n_lanes * num_layers];
// ids/conf [n_lanes * b]; selector scores [n_lanes * (k + k*k*(b-2))] and
// candidates [n_lanes * k * (b-1)], each lane's block laid out exactly as the
// single-lane graph returns it.
// ---------------------------------------------------------------------------
namespace
{
    struct DfBatchCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* ids = nullptr;
        ggml_tensor* pos = nullptr;
        ggml_tensor* mask = nullptr;
        ggml_tensor* anchors = nullptr;
        ggml_tensor* block_slots = nullptr;   // df_ring_block: each lane's block slots
        ggml_tensor* out = nullptr;
        ggml_tensor* out_conf = nullptr;
        ggml_tensor* out_s0 = nullptr;
        ggml_tensor* out_scores = nullptr;
        ggml_tensor* out_cand = nullptr;
        const void* sig = nullptr;
        std::vector<const void*> rings;
        int b = 0, lanes = 0;
        std::uint64_t scale_signature = 0;
        std::uint64_t lru = 0;
        ggml_gallocr_t galloc = nullptr;

        void reset()
        {
            if (galloc != nullptr) { ggml_gallocr_free(galloc); galloc = nullptr; }
            if (buffer != nullptr) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr; valid = false;
            ids = pos = mask = anchors = block_slots = out = out_conf = out_s0 = out_scores = out_cand = nullptr;
            sig = nullptr; rings.clear(); b = lanes = 0; scale_signature = 0;
        }
    };
    constexpr int kDfBatchCaches = 2;
    struct DfBatchPool
    {
        DfBatchCache entries[kDfBatchCaches];
        std::uint64_t clock = 0;
    };
    DfBatchPool g_df_batch_pools[tsg::TSG_MAX_DEVICES];
    inline DfBatchPool& df_batch_pool() { return g_df_batch_pools[tsg::g_active_rank]; }
}

TSG_EXPORT int TSGgml_DFlashDraftBlockBatched(
    int n_lanes, const int* block_ids, int block_len, const int* positions,
    int num_layers, int hidden_size, int head_dim, int num_heads, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale, float kq_scale,
    const int* ring_slot_pos, int sliding_window,
    void** attn_norm_arr,
    void** q_arr, int* q_type_arr, std::int64_t* q_ne0_arr, std::int64_t* q_ne1_arr, std::int64_t* q_bytes_arr,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr, int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    void** ffn_norm_arr,
    void** gate_arr, int* gate_type_arr, std::int64_t* gate_ne0_arr, std::int64_t* gate_ne1_arr, std::int64_t* gate_bytes_arr,
    void** up_arr, int* up_type_arr, std::int64_t* up_ne0_arr, std::int64_t* up_ne1_arr, std::int64_t* up_bytes_arr,
    void** down_arr, int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    void** ring_k_arr, void** ring_v_arr, int ring_dtype,
    const void* out_norm_data,
    const void* tok_embd_data, int tok_embd_type, std::int64_t tok_embd_ne0, std::int64_t tok_embd_ne1, std::int64_t tok_embd_bytes,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    int vocab_size, int* ids_out, float* conf_out,
    int conv_taps, int conv_group_size, int conv_num_groups,
    void** attn_conv_base_arr,
    void** attn_conv_proj_arr, int* attn_conv_proj_type_arr,
    std::int64_t* attn_conv_proj_ne0_arr, std::int64_t* attn_conv_proj_ne1_arr, std::int64_t* attn_conv_proj_bytes_arr,
    void** ffn_conv_base_arr,
    void** ffn_conv_proj_arr, int* ffn_conv_proj_type_arr,
    std::int64_t* ffn_conv_proj_ne0_arr, std::int64_t* ffn_conv_proj_ne1_arr, std::int64_t* ffn_conv_proj_bytes_arr,
    int sel_rank, int sel_top_k, float sel_logit_scale, float sel_logit_softcap,
    const void* sel_hidden_data, int sel_hidden_type, std::int64_t sel_hidden_ne0, std::int64_t sel_hidden_ne1, std::int64_t sel_hidden_bytes,
    const void* sel_pred_data, int sel_pred_type, std::int64_t sel_pred_ne0, std::int64_t sel_pred_ne1, std::int64_t sel_pred_bytes,
    const void* sel_succ_data, int sel_succ_type, std::int64_t sel_succ_ne0, std::int64_t sel_succ_ne1, std::int64_t sel_succ_bytes,
    float* sel_scores_out, int* sel_cand_out,
    const float* scales, int scale_count)
{
    try
    {
        if (!ensure_backend()) return 0;
        if (g_backend_type != BACKEND_TYPE_CUDA)
        { set_last_error("DFlash batched draft: requires CUDA."); return 0; }
        if ((scales == nullptr && scale_count != 0) ||
            (scales != nullptr && (num_layers <= 0 || scale_count != 6 + 9 * num_layers)))
        { set_last_error("DFlash: invalid sidecar scale table."); return 0; }
        const std::uint64_t scale_sig = df_scale_signature(scales, scale_count);
        if (n_lanes < 1 || block_len <= 0 || num_layers <= 0 || ids_out == nullptr || conf_out == nullptr)
        { set_last_error("DFlash batched draft: invalid arguments."); return 0; }

        const int L = n_lanes;
        const int b = block_len;
        const int nb = L * b;
        // As the single draft (df_ring_block): each lane's block goes into its own ring at the block's slots.
        const bool ring_block = df_ring_block() && sliding_window > 0 && b < ring_rows
            && ring_rows - (b - 1) >= sliding_window;
        const int kv_len = ring_block ? ring_rows : ring_rows + b;
        const int q_dim = num_heads * head_dim;
        const bool use_conv = conv_taps > 0 && conv_group_size > 0 && conv_num_groups > 0
            && attn_conv_base_arr != nullptr && attn_conv_proj_arr != nullptr
            && ffn_conv_base_arr != nullptr && ffn_conv_proj_arr != nullptr;
        const bool use_selector = sel_rank > 0 && sel_top_k > 0
            && sel_hidden_data != nullptr && sel_pred_data != nullptr && sel_succ_data != nullptr
            && sel_scores_out != nullptr && sel_cand_out != nullptr;
        const int gamma = b - 1;
        const int k = sel_top_k;
        if (use_selector && (gamma < 1 || k > vocab_size))
        { set_last_error("DFlash batched draft: selector needs a proposal row and top_k <= vocab."); return 0; }
        if (use_conv && conv_taps > b)
        { set_last_error("DFlash batched draft: conv_kernel_size exceeds the block width."); return 0; }
        const std::size_t lane_scores = use_selector
            ? static_cast<std::size_t>(k) + static_cast<std::size_t>(k) * k * (gamma - 1) : 0;

        // Per-lane masks, exactly the single-lane rule (see df_draft_block).
        std::vector<ggml_fp16_t> mask_data(static_cast<std::size_t>(kv_len) * b * L,
                                           ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity()));
        const ggml_fp16_t vis = static_cast<ggml_fp16_t>(0);
        std::vector<std::int32_t> anchors(L);
        for (int lane = 0; lane < L; lane++)
        {
            const int* lpos = positions + static_cast<std::size_t>(lane) * b;
            const int* slots = ring_slot_pos + static_cast<std::size_t>(lane) * ring_rows;
            const int anchor_pos = lpos[0];
            anchors[lane] = block_ids[static_cast<std::size_t>(lane) * b];
            for (int i = 0; i < b; i++)
            {
                const int qp = lpos[i];
                ggml_fp16_t* row = &mask_data[(static_cast<std::size_t>(lane) * b + i) * kv_len];
                for (int s = 0; s < ring_rows; s++)
                {
                    const int p = slots[s];
                    if (p < 0 || p >= anchor_pos) continue;
                    if (sliding_window > 0 && qp - p >= sliding_window) continue;
                    row[s] = vis;
                }
                for (int j = 0; j < b; j++)
                    row[ring_block ? lpos[j] % ring_rows : ring_rows + j] = vis;
            }
        }
        std::vector<std::int64_t> block_slots(ring_block ? nb : 0);
        for (int i = 0; i < static_cast<int>(block_slots.size()); i++)
            block_slots[i] = positions[i] % ring_rows;

        const void* sig = attn_norm_arr[0];
        std::vector<const void*> rings(ring_k_arr, ring_k_arr + static_cast<std::size_t>(L) * num_layers);
        DfBatchPool& pool = df_batch_pool();
        DfBatchCache* dc = nullptr;
        for (auto& e : pool.entries)
            if (e.valid && e.sig == sig && e.b == b && e.lanes == L && e.scale_signature == scale_sig && e.rings == rings
                && (e.block_slots != nullptr) == ring_block)
            { dc = &e; break; }

        auto download = [&](DfBatchCache& c) {
            if (use_selector)
            {
                ggml_backend_tensor_get(c.out_cand, sel_cand_out, 0,
                    static_cast<std::size_t>(k) * gamma * L * sizeof(std::int32_t));
                std::vector<float> s0(static_cast<std::size_t>(k) * L);
                ggml_backend_tensor_get(c.out_s0, s0.data(), 0, s0.size() * sizeof(float));
                std::vector<float> rest(c.out_scores != nullptr ? static_cast<std::size_t>(k) * k * (gamma - 1) * L : 0);
                if (c.out_scores != nullptr)
                    finalize_compute_with_download(c.out_scores, rest.data(), rest.size() * sizeof(float));
                host_read_barrier();
                for (int lane = 0; lane < L; lane++)
                {
                    float* dst = sel_scores_out + static_cast<std::size_t>(lane) * lane_scores;
                    std::memcpy(dst, s0.data() + static_cast<std::size_t>(lane) * k, static_cast<std::size_t>(k) * sizeof(float));
                    if (!rest.empty())
                        std::memcpy(dst + k, rest.data() + static_cast<std::size_t>(lane) * k * k * (gamma - 1),
                            static_cast<std::size_t>(k) * k * (gamma - 1) * sizeof(float));
                }
            }
            else
            {
                ggml_backend_tensor_get(c.out, ids_out, 0, static_cast<std::size_t>(nb) * sizeof(std::int32_t));
                finalize_compute_with_download(c.out_conf, conf_out, static_cast<std::size_t>(nb) * sizeof(float));
                host_read_barrier();
            }
        };

        if (dc != nullptr)
        {
            dc->lru = ++pool.clock;
            host_read_barrier();
            decode_input_set_async(dc->ids, block_ids, static_cast<std::size_t>(nb) * sizeof(std::int32_t));
            decode_input_set_async(dc->pos, positions, static_cast<std::size_t>(nb) * sizeof(std::int32_t));
            decode_input_set_async(dc->mask, mask_data.data(), mask_data.size() * sizeof(ggml_fp16_t));
            if (dc->anchors != nullptr)
                decode_input_set_async(dc->anchors, anchors.data(), anchors.size() * sizeof(std::int32_t));
            if (dc->block_slots != nullptr)
                decode_input_set_async(dc->block_slots, block_slots.data(), block_slots.size() * sizeof(std::int64_t));
            if (tsg::graph_compute_profiled(g_backend, dc->graph, "dflash batched draft") != GGML_STATUS_SUCCESS)
            {
                set_last_error("DFlash batched draft: cached graph execution failed.");
                dc->reset();
                return 0;
            }
            download(*dc);
            clear_last_error();
            return 1;
        }

        DfBatchCache* slot = nullptr;
        for (auto& e : pool.entries)
            if (!e.valid) { slot = &e; break; }
        if (slot == nullptr)
        {
            slot = &pool.entries[0];
            for (auto& e : pool.entries)
                if (e.lru < slot->lru) slot = &e;
        }
        slot->reset();

        const std::size_t ctx_size = 64 * 1024 * 1024;
        ggml_init_params ip = { ctx_size, nullptr, /*no_alloc=*/true };
        ggml_context* ctx = ggml_init(ip);
        if (ctx == nullptr) { set_last_error("DFlash batched draft: failed to init context."); return 0; }
        auto fail = [&](const char* msg) { set_last_error(msg); ggml_free(ctx); return 0; };

        ggml_tensor* ids_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nb);
        ggml_tensor* pos_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nb);
        ggml_tensor* mask_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv_len, b, 1, L);
        ggml_set_input(ids_t); ggml_set_input(pos_t); ggml_set_input(mask_t);
        ggml_tensor* block_slot_t = nullptr;
        if (ring_block)
        {
            block_slot_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, nb);
            ggml_set_input(block_slot_t);
        }
        ggml_tensor* anchors_t = nullptr;
        if (use_selector)
        {
            anchors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);
            ggml_set_input(anchors_t);
        }

        // The convolution's shift/mask never cross a lane: row r of lane l reads
        // row r - tap of the same lane, masked off for its first `tap` rows.
        std::vector<ggml_tensor*> conv_shift(use_conv ? conv_taps : 0, nullptr);
        std::vector<ggml_tensor*> conv_mask(use_conv ? conv_taps : 0, nullptr);
        std::vector<std::vector<std::int32_t>> conv_shift_data(use_conv ? conv_taps : 0);
        std::vector<std::vector<float>> conv_mask_data(use_conv ? conv_taps : 0);
        // As the single draft (df_grouped_conv_all_taps on Metal), each lane's rows shifted within its own block.
        const bool conv_all_taps = use_conv && conv_taps >= 2 && df_metal_fused_conv();
        ggml_tensor* conv_shift_all = nullptr;
        ggml_tensor* conv_mask_all = nullptr;
        std::vector<std::int32_t> conv_shift_all_data;
        std::vector<float> conv_mask_all_data;
        if (conv_all_taps)
        {
            conv_shift_all = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<std::int64_t>(nb) * conv_taps);
            conv_mask_all = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, nb, conv_taps);
            ggml_set_input(conv_shift_all);
            ggml_set_input(conv_mask_all);
            conv_shift_all_data.resize(static_cast<std::size_t>(nb) * conv_taps);
            conv_mask_all_data.resize(static_cast<std::size_t>(nb) * conv_taps);
            for (int tap = 0; tap < conv_taps; tap++)
                for (int lane = 0; lane < L; lane++)
                    for (int r = 0; r < b; r++)
                    {
                        const std::size_t i = static_cast<std::size_t>(tap) * nb + static_cast<std::size_t>(lane) * b + r;
                        conv_shift_all_data[i] = lane * b + (r >= tap ? r - tap : 0);
                        conv_mask_all_data[i] = r >= tap ? 1.0f : 0.0f;
                    }
        }
        for (int tap = 1; tap < (use_conv && !conv_all_taps ? conv_taps : 0); tap++)
        {
            conv_shift[tap] = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nb);
            conv_mask[tap] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, nb);
            ggml_set_input(conv_shift[tap]);
            ggml_set_input(conv_mask[tap]);
            conv_shift_data[tap].resize(nb);
            conv_mask_data[tap].resize(nb);
            for (int lane = 0; lane < L; lane++)
                for (int r = 0; r < b; r++)
                {
                    conv_shift_data[tap][lane * b + r] = lane * b + (r >= tap ? r - tap : 0);
                    conv_mask_data[tap][lane * b + r] = r >= tap ? 1.0f : 0.0f;
                }
        }

        ggml_tensor* tok_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(tok_embd_type), tok_embd_ne0, tok_embd_ne1);
        ggml_tensor* out_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
        ggml_tensor* sel_hidden_t = nullptr;
        ggml_tensor* sel_pred_t = nullptr;
        ggml_tensor* sel_succ_t = nullptr;
        if (use_selector)
        {
            sel_hidden_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(sel_hidden_type), sel_hidden_ne0, sel_hidden_ne1);
            sel_pred_t   = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(sel_pred_type),   sel_pred_ne0,   sel_pred_ne1);
            sel_succ_t   = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(sel_succ_type),   sel_succ_ne0,   sel_succ_ne1);
        }

        ggml_tensor* inpL = df_scaled(ctx, ggml_get_rows(ctx, tok_t, ids_t), scales, 1);   // [hidden, nb]

        std::vector<DfLayer> layers(num_layers);
        // ring_k/v per (lane, layer): [hd, ring_rows, kv_heads]
        std::vector<ggml_tensor*> ring_k(static_cast<std::size_t>(L) * num_layers);
        std::vector<ggml_tensor*> ring_v(static_cast<std::size_t>(L) * num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            lt.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.ffn_norm_w  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.q_norm_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            lt.k_norm_w    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            lt.q_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(q_type_arr[l]),    q_ne0_arr[l],    q_ne1_arr[l]);
            lt.k_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(k_type_arr[l]),    k_ne0_arr[l],    k_ne1_arr[l]);
            lt.v_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(v_type_arr[l]),    v_ne0_arr[l],    v_ne1_arr[l]);
            lt.o_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type_arr[l]),    o_ne0_arr[l],    o_ne1_arr[l]);
            lt.gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gate_type_arr[l]), gate_ne0_arr[l], gate_ne1_arr[l]);
            lt.up_w   = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(up_type_arr[l]),   up_ne0_arr[l],   up_ne1_arr[l]);
            lt.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type_arr[l]), down_ne0_arr[l], down_ne1_arr[l]);
            for (int lane = 0; lane < L; lane++)
            {
                const std::size_t ix = static_cast<std::size_t>(lane) * num_layers + l;
                ring_k[ix] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(ring_dtype), head_dim, ring_rows, num_kv_heads);
                ring_v[ix] = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(ring_dtype), head_dim, ring_rows, num_kv_heads);
            }
            if (use_conv)
            {
                lt.attn_conv_base = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, conv_taps, 2);
                lt.ffn_conv_base = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, conv_taps, 2);
                lt.attn_conv_proj = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(attn_conv_proj_type_arr[l]),
                                                       attn_conv_proj_ne0_arr[l], attn_conv_proj_ne1_arr[l]);
                lt.ffn_conv_proj = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ffn_conv_proj_type_arr[l]),
                                                      ffn_conv_proj_ne0_arr[l], ffn_conv_proj_ne1_arr[l]);
            }
        }

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            ggml_tensor* h = ggml_mul(ctx, ggml_rms_norm(ctx, inpL, eps), lt.attn_norm_w);
            ggml_tensor* attn_conv_coef = nullptr;
            if (use_conv)
            {
                attn_conv_coef = df_scaled(ctx, ggml_mul_mat(ctx, lt.attn_conv_proj, h), scales, 6 + 9 * l + 7);
                h = df_grouped_conv(ctx, h, attn_conv_coef, lt.attn_conv_base, /*side=*/0,
                                    conv_taps, conv_num_groups, conv_group_size, hidden_size, nb,
                                    conv_shift.data(), conv_mask.data(), conv_shift_all, conv_mask_all);
            }
            ggml_tensor* q = df_scaled(ctx, ggml_mul_mat(ctx, lt.q_w, h), scales, 6 + 9 * l + 0);
            ggml_tensor* kk = df_scaled(ctx, ggml_mul_mat(ctx, lt.k_w, h), scales, 6 + 9 * l + 1);
            ggml_tensor* v = df_scaled(ctx, ggml_mul_mat(ctx, lt.v_w, h), scales, 6 + 9 * l + 2);
            ggml_tensor* q3 = ggml_reshape_3d(ctx, q, head_dim, num_heads, nb);
            ggml_tensor* k3 = ggml_reshape_3d(ctx, kk, head_dim, num_kv_heads, nb);
            ggml_tensor* v3 = ggml_reshape_3d(ctx, v, head_dim, num_kv_heads, nb);
            q3 = ggml_mul(ctx, ggml_rms_norm(ctx, q3, eps), lt.q_norm_w);
            k3 = ggml_mul(ctx, ggml_rms_norm(ctx, k3, eps), lt.k_norm_w);
            q3 = ggml_rope_ext(ctx, q3, pos_t, nullptr, head_dim, /*NeoX=*/2, 0,
                               rope_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
            k3 = ggml_rope_ext(ctx, k3, pos_t, nullptr, head_dim, /*NeoX=*/2, 0,
                               rope_base, rope_freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);

            ggml_tensor* attn_all = nullptr;
            for (int lane = 0; lane < L; lane++)
            {
                const std::size_t ix = static_cast<std::size_t>(lane) * num_layers + l;
                ggml_tensor* kl = ggml_view_3d(ctx, k3, head_dim, num_kv_heads, b, k3->nb[1], k3->nb[2],
                    static_cast<std::size_t>(lane) * b * k3->nb[2]);
                ggml_tensor* vl = ggml_view_3d(ctx, v3, head_dim, num_kv_heads, b, v3->nb[1], v3->nb[2],
                    static_cast<std::size_t>(lane) * b * v3->nb[2]);
                ggml_tensor* kcat;
                ggml_tensor* vcat;
                if (ring_block)
                {
                    ggml_tensor* sl = ggml_view_1d(ctx, block_slot_t, b,
                        static_cast<std::size_t>(lane) * b * sizeof(std::int64_t));
                    kcat = ggml_set_rows(ctx, ring_k[ix], ggml_permute(ctx, kl, 0, 2, 1, 3), sl);
                    vcat = ggml_set_rows(ctx, ring_v[ix], ggml_permute(ctx, vl, 0, 2, 1, 3), sl);
                }
                else
                {
                    ggml_tensor* kb = ggml_cont(ctx, ggml_permute(ctx, kl, 0, 2, 1, 3));   // [hd, b, kv_heads]
                    ggml_tensor* vb = ggml_cont(ctx, ggml_permute(ctx, vl, 0, 2, 1, 3));
                    kcat = ggml_concat(ctx, ring_k[ix], kb, 1);
                    vcat = ggml_concat(ctx, ring_v[ix], vb, 1);
                }
                ggml_tensor* ql = ggml_view_3d(ctx, q3, head_dim, num_heads, b, q3->nb[1], q3->nb[2],
                    static_cast<std::size_t>(lane) * b * q3->nb[2]);
                ggml_tensor* qa = ggml_permute(ctx, ql, 0, 2, 1, 3);                    // [hd, b, heads]
                ggml_tensor* ml = ggml_view_2d(ctx, mask_t, kv_len, b, mask_t->nb[1],
                    static_cast<std::size_t>(lane) * mask_t->nb[3]);
                ggml_tensor* fa = flash_attn_ext_guarded(ctx, "DFlash batched draft", qa, kcat, vcat, ml,
                    kq_scale, 0.0f, 0.0f, nullptr, GGML_PREC_F32);
                ggml_tensor* attn_l = ggml_reshape_2d(ctx, fa, q_dim, b);
                attn_all = attn_all == nullptr ? attn_l : ggml_concat(ctx, attn_all, attn_l, 1);
            }

            ggml_tensor* attn_out = df_scaled(ctx, ggml_mul_mat(ctx, lt.o_w, attn_all), scales, 6 + 9 * l + 3);
            if (use_conv)
            {
                attn_out = df_grouped_conv(ctx, attn_out, attn_conv_coef, lt.attn_conv_base, /*side=*/1,
                                           conv_taps, conv_num_groups, conv_group_size, hidden_size, nb,
                                           conv_shift.data(), conv_mask.data(), conv_shift_all, conv_mask_all);
            }
            ggml_tensor* ffn_inp = ggml_add(ctx, attn_out, inpL);
            ggml_tensor* fh = ggml_mul(ctx, ggml_rms_norm(ctx, ffn_inp, eps), lt.ffn_norm_w);
            ggml_tensor* ffn_conv_coef = nullptr;
            if (use_conv)
            {
                ffn_conv_coef = df_scaled(ctx, ggml_mul_mat(ctx, lt.ffn_conv_proj, fh), scales, 6 + 9 * l + 8);
                fh = df_grouped_conv(ctx, fh, ffn_conv_coef, lt.ffn_conv_base, /*side=*/0,
                                     conv_taps, conv_num_groups, conv_group_size, hidden_size, nb,
                                     conv_shift.data(), conv_mask.data(), conv_shift_all, conv_mask_all);
            }
            ggml_tensor* gate = df_scaled(ctx, ggml_mul_mat(ctx, lt.gate_w, fh), scales, 6 + 9 * l + 4);
            ggml_tensor* up   = df_scaled(ctx, ggml_mul_mat(ctx, lt.up_w, fh), scales, 6 + 9 * l + 5);
            ggml_tensor* act  = ggml_mul(ctx, ggml_silu(ctx, gate), up);
            ggml_tensor* down = df_scaled(ctx, ggml_mul_mat(ctx, lt.down_w, act), scales, 6 + 9 * l + 6);
            if (use_conv)
            {
                down = df_grouped_conv(ctx, down, ffn_conv_coef, lt.ffn_conv_base, /*side=*/1,
                                       conv_taps, conv_num_groups, conv_group_size, hidden_size, nb,
                                       conv_shift.data(), conv_mask.data(), conv_shift_all, conv_mask_all);
            }
            inpL = ggml_add(ctx, down, ffn_inp);
        }

        ggml_tensor* cur = ggml_mul(ctx, ggml_rms_norm(ctx, inpL, eps), out_norm_t);   // [hidden, nb]

        ggml_tensor* sel_s0 = nullptr;
        ggml_tensor* sel_scores = nullptr;
        ggml_tensor* sel_cand = nullptr;
        ggml_tensor* out = nullptr;
        ggml_tensor* out_conf = nullptr;
        ggml_tensor* out_node = nullptr;
        ggml_tensor* conf_node = nullptr;
        if (use_selector)
        {
            // Columns are lane-major (lane * gamma + e): each lane's proposal rows 1..b-1.
            ggml_tensor* pred_h = ggml_reshape_2d(ctx, ggml_cont(ctx,
                ggml_view_3d(ctx, cur, hidden_size, gamma, L, cur->nb[1], static_cast<std::size_t>(b) * cur->nb[1], cur->nb[1])),
                hidden_size, static_cast<std::int64_t>(gamma) * L);
            const std::int64_t gl = static_cast<std::int64_t>(gamma) * L;
            ggml_tensor* sel_logits = df_scaled(ctx, ggml_mul_mat(ctx, lm_head_t, pred_h), scales, 2);    // [vocab, gamma*L]
            sel_cand = ggml_top_k(ctx, sel_logits, k);                                                   // [k, gamma*L]
            if (!backend_supports_op(sel_cand))
                return fail("DFlash batched draft: this backend has no top-k over the vocabulary.");
            ggml_set_output(sel_cand);
            ggml_tensor* logits3 = ggml_reshape_3d(ctx, sel_logits, 1, vocab_size, gl);
            ggml_tensor* unary = ggml_get_rows(ctx, logits3, sel_cand);                                  // [1, k, gamma*L]
            ggml_tensor* unary_kg = ggml_reshape_3d(ctx, unary, k, 1, gl);
            if (sel_logit_scale != 1.0f)
                unary_kg = ggml_scale(ctx, unary_kg, sel_logit_scale);
            if (sel_logit_softcap > 0.0f)
                unary_kg = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, unary_kg, 1.0f / sel_logit_softcap)), sel_logit_softcap);

            ggml_tensor* ph = df_scaled(ctx, ggml_mul_mat(ctx, sel_hidden_t, pred_h), scales, 3);         // [r, gamma*L]
            ggml_tensor* cand_flat = ggml_reshape_1d(ctx, sel_cand, static_cast<std::int64_t>(k) * gl);
            ggml_tensor* keys = df_scaled(ctx, ggml_get_rows(ctx, sel_succ_t, cand_flat), scales, 5);     // [r, k*gamma*L]
            ggml_tensor* keys3 = ggml_reshape_3d(ctx, keys, sel_rank, k, gl);

            // Position 0 of each lane: its predecessor is that lane's anchor.
            ggml_tensor* a0 = df_scaled(ctx, ggml_get_rows(ctx, sel_pred_t, anchors_t), scales, 4);       // [r, L]
            ggml_tensor* ph0 = ggml_cont(ctx, ggml_view_2d(ctx, ph, sel_rank, L, static_cast<std::size_t>(gamma) * ph->nb[1], 0));
            ggml_tensor* m0 = ggml_reshape_3d(ctx, ggml_mul(ctx, a0, ph0), sel_rank, 1, L);
            ggml_tensor* keys0 = ggml_cont(ctx, ggml_view_3d(ctx, keys3, sel_rank, k, L,
                keys3->nb[1], static_cast<std::size_t>(gamma) * keys3->nb[2], 0));                     // [r, k, L]
            ggml_tensor* u0 = ggml_cont(ctx, ggml_view_3d(ctx, unary_kg, k, 1, L,
                unary_kg->nb[1], static_cast<std::size_t>(gamma) * unary_kg->nb[2], 0));               // [k, 1, L]
            sel_s0 = ggml_add(ctx, ggml_mul_mat(ctx, keys0, m0), u0);                                    // [k, 1, L]
            ggml_set_output(sel_s0);

            if (gamma > 1)
            {
                const std::int64_t rl = static_cast<std::int64_t>(gamma - 1) * L;
                // Predecessors of positions 1..gamma-1 are positions 0..gamma-2 of the same lane.
                ggml_tensor* prev = ggml_cont(ctx, ggml_view_3d(ctx, sel_cand, k, gamma - 1, L,
                    sel_cand->nb[1], static_cast<std::size_t>(gamma) * sel_cand->nb[1], 0));
                ggml_tensor* preds = df_scaled(ctx, ggml_get_rows(ctx, sel_pred_t,
                    ggml_reshape_1d(ctx, prev, static_cast<std::int64_t>(k) * rl)), scales, 4);
                ggml_tensor* preds3 = ggml_reshape_3d(ctx, preds, sel_rank, k, rl);
                ggml_tensor* ph_rest = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_3d(ctx, ph, sel_rank, gamma - 1, L,
                    ph->nb[1], static_cast<std::size_t>(gamma) * ph->nb[1], ph->nb[1])), sel_rank, 1, rl);
                ggml_tensor* m = ggml_mul(ctx, preds3, ph_rest);                                         // [r, k, rl]
                ggml_tensor* keys_rest = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_4d(ctx, keys3, sel_rank, k, gamma - 1, L,
                    keys3->nb[1], keys3->nb[2], static_cast<std::size_t>(gamma) * keys3->nb[2], keys3->nb[2])), sel_rank, k, rl);
                sel_scores = ggml_mul_mat(ctx, keys_rest, m);                                            // [k, k, rl]
                ggml_tensor* u_rest = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_4d(ctx, unary_kg, k, 1, gamma - 1, L,
                    unary_kg->nb[1], unary_kg->nb[2], static_cast<std::size_t>(gamma) * unary_kg->nb[2], unary_kg->nb[2])), k, 1, rl);
                sel_scores = ggml_add(ctx, sel_scores, u_rest);
                ggml_set_output(sel_scores);
            }
        }
        else
        {
            ggml_tensor* logits = df_scaled(ctx, ggml_mul_mat(ctx, lm_head_t, cur), scales, 2);          // [vocab, nb]
            ggml_tensor* probs = ggml_soft_max(ctx, logits);
            ggml_tensor* am = ggml_argmax(ctx, probs);
            ggml_tensor* am2 = ggml_reshape_2d(ctx, am, 1, nb);
            ggml_tensor* pr3 = ggml_reshape_3d(ctx, probs, 1, vocab_size, nb);
            ggml_tensor* mp = ggml_get_rows(ctx, pr3, am2);
            out = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, nb);
            out_conf = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nb);
            out_node = ggml_cpy(ctx, am, out);
            conf_node = ggml_cpy(ctx, ggml_reshape_1d(ctx, mp, nb), out_conf);
            ggml_set_output(out_node);
            ggml_set_output(conf_node);
        }

        ggml_cgraph* graph = ggml_new_graph_custom(ctx,
            static_cast<std::size_t>(num_layers) * (256 + 32 * static_cast<std::size_t>(L)) + 2048, false);
        if (use_selector)
        {
            ggml_build_forward_expand(graph, sel_cand);
            ggml_build_forward_expand(graph, sel_s0);
            if (sel_scores != nullptr)
                ggml_build_forward_expand(graph, sel_scores);
        }
        else
        {
            ggml_build_forward_expand(graph, out_node);
            ggml_build_forward_expand(graph, conf_node);
        }

        DfBinder binder;
        binder.dev = ggml_backend_get_device(g_backend);
        const std::size_t norm_bytes = static_cast<std::size_t>(hidden_size) * sizeof(float);
        const std::size_t head_norm_bytes = static_cast<std::size_t>(head_dim) * sizeof(float);
        const std::size_t ring_bytes = kv_cache_bytes(num_kv_heads, ring_rows, head_dim, ring_dtype);
        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            binder.bind(lt.q_w,    q_arr[l],    static_cast<std::size_t>(q_bytes_arr[l]), true);
            binder.bind(lt.k_w,    k_arr[l],    static_cast<std::size_t>(k_bytes_arr[l]), true);
            binder.bind(lt.v_w,    v_arr[l],    static_cast<std::size_t>(v_bytes_arr[l]), true);
            binder.bind(lt.o_w,    o_arr[l],    static_cast<std::size_t>(o_bytes_arr[l]), true);
            binder.bind(lt.gate_w, gate_arr[l], static_cast<std::size_t>(gate_bytes_arr[l]), true);
            binder.bind(lt.up_w,   up_arr[l],   static_cast<std::size_t>(up_bytes_arr[l]), true);
            binder.bind(lt.down_w, down_arr[l], static_cast<std::size_t>(down_bytes_arr[l]), true);
            binder.bind(lt.attn_norm_w, attn_norm_arr[l], norm_bytes, true);
            binder.bind(lt.ffn_norm_w,  ffn_norm_arr[l],  norm_bytes, true);
            binder.bind(lt.q_norm_w,    q_norm_arr[l],    head_norm_bytes, true);
            binder.bind(lt.k_norm_w,    k_norm_arr[l],    head_norm_bytes, true);
            for (int lane = 0; lane < L; lane++)
            {
                const std::size_t ix = static_cast<std::size_t>(lane) * num_layers + l;
                binder.bind(ring_k[ix], ring_k_arr[ix], ring_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                binder.bind(ring_v[ix], ring_v_arr[ix], ring_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            }
            if (use_conv)
            {
                const std::size_t conv_base_bytes =
                    2u * static_cast<std::size_t>(conv_taps) * hidden_size * sizeof(float);
                binder.bind(lt.attn_conv_base, attn_conv_base_arr[l], conv_base_bytes, true);
                binder.bind(lt.ffn_conv_base, ffn_conv_base_arr[l], conv_base_bytes, true);
                binder.bind(lt.attn_conv_proj, attn_conv_proj_arr[l],
                            static_cast<std::size_t>(attn_conv_proj_bytes_arr[l]), true);
                binder.bind(lt.ffn_conv_proj, ffn_conv_proj_arr[l],
                            static_cast<std::size_t>(ffn_conv_proj_bytes_arr[l]), true);
            }
        }
        if (use_selector)
        {
            binder.bind(sel_hidden_t, const_cast<void*>(sel_hidden_data), static_cast<std::size_t>(sel_hidden_bytes), true);
            binder.bind(sel_pred_t, const_cast<void*>(sel_pred_data), static_cast<std::size_t>(sel_pred_bytes), true);
            binder.bind(sel_succ_t, const_cast<void*>(sel_succ_data), static_cast<std::size_t>(sel_succ_bytes), true);
        }
        binder.bind(out_norm_t, const_cast<void*>(out_norm_data), norm_bytes, true);
        binder.bind(tok_t, const_cast<void*>(tok_embd_data), static_cast<std::size_t>(tok_embd_bytes), true);
        binder.bind(lm_head_t, const_cast<void*>(lm_head_data), static_cast<std::size_t>(lm_head_bytes), true);

        ggml_gallocr_t persist_galloc = nullptr;
        if (!df_alloc_persistent(ctx, graph, persist_galloc))
            return fail("DFlash batched draft: failed to allocate persist buffer.");

        host_read_barrier();
        binder.flush();
        for (int tap = 1; tap < (use_conv && !conv_all_taps ? conv_taps : 0); tap++)
        {
            ggml_backend_tensor_set(conv_shift[tap], conv_shift_data[tap].data(), 0, static_cast<std::size_t>(nb) * sizeof(std::int32_t));
            ggml_backend_tensor_set(conv_mask[tap], conv_mask_data[tap].data(), 0, static_cast<std::size_t>(nb) * sizeof(float));
        }
        if (conv_all_taps)
        {
            ggml_backend_tensor_set(conv_shift_all, conv_shift_all_data.data(), 0, conv_shift_all_data.size() * sizeof(std::int32_t));
            ggml_backend_tensor_set(conv_mask_all, conv_mask_all_data.data(), 0, conv_mask_all_data.size() * sizeof(float));
        }
        slot->ctx = ctx; slot->galloc = persist_galloc; slot->graph = graph;
        slot->ids = ids_t; slot->pos = pos_t; slot->mask = mask_t; slot->anchors = anchors_t;
        slot->block_slots = block_slot_t;
        slot->out = out; slot->out_conf = out_conf;
        slot->out_s0 = sel_s0; slot->out_scores = sel_scores; slot->out_cand = sel_cand;
        slot->sig = sig; slot->rings = rings; slot->b = b; slot->lanes = L;
        slot->scale_signature = scale_sig;
        slot->lru = ++pool.clock;
        slot->valid = true;

        ggml_backend_tensor_set(ids_t, block_ids, 0, static_cast<std::size_t>(nb) * sizeof(std::int32_t));
        ggml_backend_tensor_set(pos_t, positions, 0, static_cast<std::size_t>(nb) * sizeof(std::int32_t));
        ggml_backend_tensor_set(mask_t, mask_data.data(), 0, mask_data.size() * sizeof(ggml_fp16_t));
        if (anchors_t != nullptr)
            ggml_backend_tensor_set(anchors_t, anchors.data(), 0, anchors.size() * sizeof(std::int32_t));
        if (block_slot_t != nullptr)
            ggml_backend_tensor_set(block_slot_t, block_slots.data(), 0, block_slots.size() * sizeof(std::int64_t));
        if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("DFlash batched draft: graph execution failed.");
            slot->reset();
            return 0;
        }
        download(*slot);
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in DFlash batched draft."); return 0; }
}

TSG_EXPORT void TSGgml_DFlashResetCaches()
{
    for (auto& p : g_df_inject_pools) p.reset_all();
    for (auto& p : g_df_draft_pools)  p.reset_all();
    for (auto& p : g_df_batch_pools)
        for (auto& e : p.entries) e.reset();
    for (auto& p : g_df_inject_batch_pools)
        for (auto& e : p.entries) e.reset();
}

// Preserve the unscaled ABI for older callers. Scaled graphs have distinct
// cache identities even when their weights and ring pointers are shared.
TSG_EXPORT int TSGgml_DFlashInject(
    const float* feat_rows, int feature_size, int n_rows,
    const std::int64_t* ring_rows_idx,          // ring slot per row (position % ring_rows)
    const int* positions,                        // absolute position per row (for RoPE)
    int num_layers, int hidden_size, int head_dim, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale,
    const void* fc_data, int fc_type, std::int64_t fc_ne0, std::int64_t fc_ne1, std::int64_t fc_bytes,
    const void* enc_norm_data,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** k_norm_arr,
    void** ring_k_arr, void** ring_v_arr,
    int ring_dtype)
{
    return df_inject(
        feat_rows, feature_size, n_rows, ring_rows_idx, positions, num_layers, hidden_size, head_dim,
        num_kv_heads, ring_rows, eps, rope_base, rope_freq_scale, fc_data, fc_type, fc_ne0, fc_ne1, fc_bytes,
        enc_norm_data, k_arr, k_type_arr, k_ne0_arr, k_ne1_arr, k_bytes_arr, v_arr, v_type_arr, v_ne0_arr,
        v_ne1_arr, v_bytes_arr, k_norm_arr, ring_k_arr, ring_v_arr, ring_dtype, nullptr, 0);
}

TSG_EXPORT int TSGgml_DFlashInjectScaled(
    const float* feat_rows, int feature_size, int n_rows,
    const std::int64_t* ring_rows_idx,          // ring slot per row (position % ring_rows)
    const int* positions,                        // absolute position per row (for RoPE)
    int num_layers, int hidden_size, int head_dim, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale,
    const void* fc_data, int fc_type, std::int64_t fc_ne0, std::int64_t fc_ne1, std::int64_t fc_bytes,
    const void* enc_norm_data,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** k_norm_arr,
    void** ring_k_arr, void** ring_v_arr,
    int ring_dtype,
    const float* scales, int scale_count)
{
    return df_inject(
        feat_rows, feature_size, n_rows, ring_rows_idx, positions, num_layers, hidden_size, head_dim,
        num_kv_heads, ring_rows, eps, rope_base, rope_freq_scale, fc_data, fc_type, fc_ne0, fc_ne1, fc_bytes,
        enc_norm_data, k_arr, k_type_arr, k_ne0_arr, k_ne1_arr, k_bytes_arr, v_arr, v_type_arr, v_ne0_arr,
        v_ne1_arr, v_bytes_arr, k_norm_arr, ring_k_arr, ring_v_arr, ring_dtype, scales, scale_count);
}

TSG_EXPORT int TSGgml_DFlashInjectBatched(
    int n_lanes, int max_rows, int total_rows,
    const float* feat_rows, int feature_size,
    const std::int32_t* gather, const std::int64_t* ring_rows_idx, const int* positions,
    int num_layers, int hidden_size, int head_dim, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale,
    const void* fc_data, int fc_type, std::int64_t fc_ne0, std::int64_t fc_ne1, std::int64_t fc_bytes,
    const void* enc_norm_data,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** k_norm_arr,
    void** ring_k_arr, void** ring_v_arr,
    int ring_dtype,
    const float* scales, int scale_count)
{
    return df_inject_batched(
        n_lanes, max_rows, total_rows, feat_rows, feature_size, gather, ring_rows_idx, positions,
        num_layers, hidden_size, head_dim, num_kv_heads, ring_rows, eps, rope_base, rope_freq_scale,
        fc_data, fc_type, fc_ne0, fc_ne1, fc_bytes, enc_norm_data,
        k_arr, k_type_arr, k_ne0_arr, k_ne1_arr, k_bytes_arr, v_arr, v_type_arr, v_ne0_arr, v_ne1_arr, v_bytes_arr,
        k_norm_arr, ring_k_arr, ring_v_arr, ring_dtype, scales, scale_count);
}

TSG_EXPORT int TSGgml_DFlashDraftBlock(
    const int* block_ids, int block_len, const int* positions,
    int num_layers, int hidden_size, int head_dim, int num_heads, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale, float kq_scale,
    // ring slot -> absolute position, or -1 when the slot is not live
    const int* ring_slot_pos, int sliding_window,
    void** attn_norm_arr,
    void** q_arr, int* q_type_arr, std::int64_t* q_ne0_arr, std::int64_t* q_ne1_arr, std::int64_t* q_bytes_arr,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr, int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    void** ffn_norm_arr,
    void** gate_arr, int* gate_type_arr, std::int64_t* gate_ne0_arr, std::int64_t* gate_ne1_arr, std::int64_t* gate_bytes_arr,
    void** up_arr, int* up_type_arr, std::int64_t* up_ne0_arr, std::int64_t* up_ne1_arr, std::int64_t* up_bytes_arr,
    void** down_arr, int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    void** ring_k_arr, void** ring_v_arr, int ring_dtype,
    const void* out_norm_data,
    const void* tok_embd_data, int tok_embd_type, std::int64_t tok_embd_ne0, std::int64_t tok_embd_ne1, std::int64_t tok_embd_bytes,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    int vocab_size, int* ids_out, float* conf_out,
    // ---- DFlash2 grouped dynamic convolution (conv_taps == 0 disables) ----
    int conv_taps, int conv_group_size, int conv_num_groups,
    void** attn_conv_base_arr,
    void** attn_conv_proj_arr, int* attn_conv_proj_type_arr,
    std::int64_t* attn_conv_proj_ne0_arr, std::int64_t* attn_conv_proj_ne1_arr, std::int64_t* attn_conv_proj_bytes_arr,
    void** ffn_conv_base_arr,
    void** ffn_conv_proj_arr, int* ffn_conv_proj_type_arr,
    std::int64_t* ffn_conv_proj_ne0_arr, std::int64_t* ffn_conv_proj_ne1_arr, std::int64_t* ffn_conv_proj_bytes_arr,
    // ---- DFlash2 candidate selector (sel_rank == 0 disables) ----
    int sel_rank, int sel_top_k, float sel_logit_scale, float sel_logit_softcap,
    const void* sel_hidden_data, int sel_hidden_type, std::int64_t sel_hidden_ne0, std::int64_t sel_hidden_ne1, std::int64_t sel_hidden_bytes,
    const void* sel_pred_data, int sel_pred_type, std::int64_t sel_pred_ne0, std::int64_t sel_pred_ne1, std::int64_t sel_pred_bytes,
    const void* sel_succ_data, int sel_succ_type, std::int64_t sel_succ_ne0, std::int64_t sel_succ_ne1, std::int64_t sel_succ_bytes,
    float* sel_scores_out, int* sel_cand_out)
{
    return df_draft_block(
        block_ids, block_len, positions, num_layers, hidden_size, head_dim, num_heads, num_kv_heads, ring_rows,
        eps, rope_base, rope_freq_scale, kq_scale, ring_slot_pos, sliding_window, attn_norm_arr, q_arr,
        q_type_arr, q_ne0_arr, q_ne1_arr, q_bytes_arr, k_arr, k_type_arr, k_ne0_arr, k_ne1_arr, k_bytes_arr,
        v_arr, v_type_arr, v_ne0_arr, v_ne1_arr, v_bytes_arr, q_norm_arr, k_norm_arr, o_arr, o_type_arr,
        o_ne0_arr, o_ne1_arr, o_bytes_arr, ffn_norm_arr, gate_arr, gate_type_arr, gate_ne0_arr, gate_ne1_arr,
        gate_bytes_arr, up_arr, up_type_arr, up_ne0_arr, up_ne1_arr, up_bytes_arr, down_arr, down_type_arr,
        down_ne0_arr, down_ne1_arr, down_bytes_arr, ring_k_arr, ring_v_arr, ring_dtype, out_norm_data,
        tok_embd_data, tok_embd_type, tok_embd_ne0, tok_embd_ne1, tok_embd_bytes, lm_head_data, lm_head_type,
        lm_head_ne0, lm_head_ne1, lm_head_bytes, vocab_size, ids_out, conf_out, conv_taps, conv_group_size,
        conv_num_groups, attn_conv_base_arr, attn_conv_proj_arr, attn_conv_proj_type_arr,
        attn_conv_proj_ne0_arr, attn_conv_proj_ne1_arr, attn_conv_proj_bytes_arr, ffn_conv_base_arr,
        ffn_conv_proj_arr, ffn_conv_proj_type_arr, ffn_conv_proj_ne0_arr, ffn_conv_proj_ne1_arr,
        ffn_conv_proj_bytes_arr, sel_rank, sel_top_k, sel_logit_scale, sel_logit_softcap, sel_hidden_data,
        sel_hidden_type, sel_hidden_ne0, sel_hidden_ne1, sel_hidden_bytes, sel_pred_data, sel_pred_type,
        sel_pred_ne0, sel_pred_ne1, sel_pred_bytes, sel_succ_data, sel_succ_type, sel_succ_ne0, sel_succ_ne1,
        sel_succ_bytes, sel_scores_out, sel_cand_out, nullptr, 0);
}

TSG_EXPORT int TSGgml_DFlashDraftBlockScaled(
    const int* block_ids, int block_len, const int* positions,
    int num_layers, int hidden_size, int head_dim, int num_heads, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale, float kq_scale,
    // ring slot -> absolute position, or -1 when the slot is not live
    const int* ring_slot_pos, int sliding_window,
    void** attn_norm_arr,
    void** q_arr, int* q_type_arr, std::int64_t* q_ne0_arr, std::int64_t* q_ne1_arr, std::int64_t* q_bytes_arr,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr, int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    void** ffn_norm_arr,
    void** gate_arr, int* gate_type_arr, std::int64_t* gate_ne0_arr, std::int64_t* gate_ne1_arr, std::int64_t* gate_bytes_arr,
    void** up_arr, int* up_type_arr, std::int64_t* up_ne0_arr, std::int64_t* up_ne1_arr, std::int64_t* up_bytes_arr,
    void** down_arr, int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    void** ring_k_arr, void** ring_v_arr, int ring_dtype,
    const void* out_norm_data,
    const void* tok_embd_data, int tok_embd_type, std::int64_t tok_embd_ne0, std::int64_t tok_embd_ne1, std::int64_t tok_embd_bytes,
    const void* lm_head_data, int lm_head_type, std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    int vocab_size, int* ids_out, float* conf_out,
    // ---- DFlash2 grouped dynamic convolution (conv_taps == 0 disables) ----
    int conv_taps, int conv_group_size, int conv_num_groups,
    void** attn_conv_base_arr,
    void** attn_conv_proj_arr, int* attn_conv_proj_type_arr,
    std::int64_t* attn_conv_proj_ne0_arr, std::int64_t* attn_conv_proj_ne1_arr, std::int64_t* attn_conv_proj_bytes_arr,
    void** ffn_conv_base_arr,
    void** ffn_conv_proj_arr, int* ffn_conv_proj_type_arr,
    std::int64_t* ffn_conv_proj_ne0_arr, std::int64_t* ffn_conv_proj_ne1_arr, std::int64_t* ffn_conv_proj_bytes_arr,
    // ---- DFlash2 candidate selector (sel_rank == 0 disables) ----
    int sel_rank, int sel_top_k, float sel_logit_scale, float sel_logit_softcap,
    const void* sel_hidden_data, int sel_hidden_type, std::int64_t sel_hidden_ne0, std::int64_t sel_hidden_ne1, std::int64_t sel_hidden_bytes,
    const void* sel_pred_data, int sel_pred_type, std::int64_t sel_pred_ne0, std::int64_t sel_pred_ne1, std::int64_t sel_pred_bytes,
    const void* sel_succ_data, int sel_succ_type, std::int64_t sel_succ_ne0, std::int64_t sel_succ_ne1, std::int64_t sel_succ_bytes,
    float* sel_scores_out, int* sel_cand_out,
    const float* scales, int scale_count)
{
    return df_draft_block(
        block_ids, block_len, positions, num_layers, hidden_size, head_dim, num_heads, num_kv_heads, ring_rows,
        eps, rope_base, rope_freq_scale, kq_scale, ring_slot_pos, sliding_window, attn_norm_arr, q_arr,
        q_type_arr, q_ne0_arr, q_ne1_arr, q_bytes_arr, k_arr, k_type_arr, k_ne0_arr, k_ne1_arr, k_bytes_arr,
        v_arr, v_type_arr, v_ne0_arr, v_ne1_arr, v_bytes_arr, q_norm_arr, k_norm_arr, o_arr, o_type_arr,
        o_ne0_arr, o_ne1_arr, o_bytes_arr, ffn_norm_arr, gate_arr, gate_type_arr, gate_ne0_arr, gate_ne1_arr,
        gate_bytes_arr, up_arr, up_type_arr, up_ne0_arr, up_ne1_arr, up_bytes_arr, down_arr, down_type_arr,
        down_ne0_arr, down_ne1_arr, down_bytes_arr, ring_k_arr, ring_v_arr, ring_dtype, out_norm_data,
        tok_embd_data, tok_embd_type, tok_embd_ne0, tok_embd_ne1, tok_embd_bytes, lm_head_data, lm_head_type,
        lm_head_ne0, lm_head_ne1, lm_head_bytes, vocab_size, ids_out, conf_out, conv_taps, conv_group_size,
        conv_num_groups, attn_conv_base_arr, attn_conv_proj_arr, attn_conv_proj_type_arr,
        attn_conv_proj_ne0_arr, attn_conv_proj_ne1_arr, attn_conv_proj_bytes_arr, ffn_conv_base_arr,
        ffn_conv_proj_arr, ffn_conv_proj_type_arr, ffn_conv_proj_ne0_arr, ffn_conv_proj_ne1_arr,
        ffn_conv_proj_bytes_arr, sel_rank, sel_top_k, sel_logit_scale, sel_logit_softcap, sel_hidden_data,
        sel_hidden_type, sel_hidden_ne0, sel_hidden_ne1, sel_hidden_bytes, sel_pred_data, sel_pred_type,
        sel_pred_ne0, sel_pred_ne1, sel_pred_bytes, sel_succ_data, sel_succ_type, sel_succ_ne0, sel_succ_ne1,
        sel_succ_bytes, sel_scores_out, sel_cand_out, scales, scale_count);
}
