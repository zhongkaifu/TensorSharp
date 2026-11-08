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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

using namespace tsg;

// ============================================================================
// Qwen3.5/3.8 BATCHED SPECULATIVE VERIFY: N concurrent sequences, T tokens each
// ([next token, draft_1 .. draft_{T-1}]), in ONE ggml graph.
//
// The holder-bound arena decode (ggml_ops_qwen35_batched_arena.cpp) batches one
// token per sequence; with speculation off for concurrent requests that is one
// weight sweep per generated token per step. A draft window costs little more
// than one token in a memory-bound sweep, so verifying every sequence's window in
// the same graph multiplies the tokens a step produces by the acceptance length.
//
// Layout, per layer (slot s is sequence s; every slot binds its holder's own
// resident K/V, conv and delta copies under the solo decode's keys and sizes,
// exactly like the bound arena):
//   * projections, FFN and LM head run over all N*T columns (slot-major).
//   * ATTENTION: per slot one set_rows per K/V writing rows [pos, pos+T) and one
//     flash attention over the slot's window with a causal [window, T] mask built
//     on the device from per-(slot, row) thresholds.
//   * GDN: the slots' conv states are gathered into one batch so a single
//     ssm_conv serves every slot (per-slot recurrent chains cost ~11 kernels per
//     slot per layer, which is what made a two-sequence arena step 19 ms against
//     13 ms for one). The recurrence itself runs once per slot over that slot's
//     resident delta state, so the 3 MB state is never copied into a batch.
//
// The verify NEVER writes recurrent state. It keeps, per GDN layer, the inputs
// the recurrence consumed (l2-normed q/k, v, gate, beta and the conv window), and
// TSGgml_Qwen35BatchedVerifyCommit replays only the ACCEPTED prefix of each slot
// from its untouched state into the holder's copies. Keeping a state snapshot per
// token instead (what the solo verify does) costs T x 151 MB per sequence on
// Qwen3.8-27B - 2.4 GB for four sequences of four tokens - and writes all of it
// every step; the replay touches each accepted state once. K/V rows past the
// accepted prefix stay in the cache masked and are overwritten by later writes.
// One commit graph serves every slot and acceptance (see qbv_commit_graph), so
// it is captured once and replayed as a single CUDA graph each round.
//
// Greedy only: the graph returns each row's argmax and the DFlash residual taps
// of every row; nothing vocabulary-sized crosses the bus. CUDA only, holder-bound
// only (the managed caller checks TSGgml_Qwen35ArenaUsesHolderBuffers), dense FFN
// only. Declines (0) leave every holder untouched; -1 after execution started.
// ============================================================================
namespace
{
    constexpr int kQbvMaxEntries = 2;
    constexpr const char* kQbvKernel = "Qwen3.5 batched verify";
    constexpr const char* kQbvCommitKernel = "Qwen3.5 batched verify commit";

    // Windows grow in coarser steps than the decode graphs' 256 rows: every
    // window change rebuilds this (larger) graph, and masked keys are cheap.
    std::int64_t qbv_window_round(std::int64_t rows)
    {
        return ((rows + 511) / 512) * 512;
    }

    struct QbvLayerWeights
    {
        ggml_tensor* attn_norm_w = nullptr;
        ggml_tensor* post_attn_norm_w = nullptr;
        ggml_tensor* qkv_w = nullptr;
        ggml_tensor* k_w = nullptr;
        ggml_tensor* v_w = nullptr;
        ggml_tensor* q_norm_w = nullptr;
        ggml_tensor* k_norm_w = nullptr;
        ggml_tensor* o_w = nullptr;
        ggml_tensor* gdn_qkv_w = nullptr;
        ggml_tensor* gdn_gate_w = nullptr;
        ggml_tensor* ssm_beta_w = nullptr;
        ggml_tensor* ssm_alpha_w = nullptr;
        ggml_tensor* conv1d_w = nullptr;
        ggml_tensor* ssm_dt_w = nullptr;
        ggml_tensor* ssm_a_w = nullptr;
        ggml_tensor* ssm_norm_w = nullptr;
        ggml_tensor* ssm_out_w = nullptr;
        ggml_tensor* gu_w = nullptr;
        ggml_tensor* ffn_gate_w = nullptr;
        ggml_tensor* ffn_up_w = nullptr;
        ggml_tensor* down_w = nullptr;
        ggml_tensor* psc[TSQ35_SC_COUNT] = {};
    };

    // The weight tensors of every layer, as leaves the caller binds with
    // qbv_bind_layer_weights.
    std::vector<QbvLayerWeights> qbv_new_layer_weights(ggml_context* ctx, const TSGgmlQwen35LayerDesc* layers,
        int num_layers, int H, int hd, int conv_kernel, int conv_dim, int num_v_heads, int head_v_dim)
    {
        std::vector<QbvLayerWeights> lw(num_layers);
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            QbvLayerWeights& t = lw[l];
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
                t.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
                t.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
                t.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.o_type), d.o_ne0, d.o_ne1);
            }
            else
            {
                t.gdn_qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_qkv_type), d.gdn_qkv_ne0, d.gdn_qkv_ne1);
                if (d.gdn_gate_w != nullptr)
                {
                    t.gdn_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gdn_gate_type), d.gdn_gate_ne0, d.gdn_gate_ne1);
                    t.ssm_beta_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_beta_type), d.ssm_beta_ne0, d.ssm_beta_ne1);
                    t.ssm_alpha_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_alpha_type), d.ssm_alpha_ne0, d.ssm_alpha_ne1);
                }
                t.conv1d_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, conv_kernel, conv_dim);
                t.ssm_dt_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_a_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_v_heads);
                t.ssm_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_v_dim);
                t.ssm_out_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ssm_out_type), d.ssm_out_ne0, d.ssm_out_ne1);
            }
            if (d.gu_w != nullptr)
                t.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.gu_type), d.gu_ne0, d.gu_ne1);
            else
            {
                t.ffn_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ffn_gate_type), d.ffn_gate_ne0, d.ffn_gate_ne1);
                t.ffn_up_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.ffn_up_type), d.ffn_up_ne0, d.ffn_up_ne1);
            }
            t.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d.down_type), d.down_ne0, d.down_ne1);
        }
        return lw;
    }

    template <typename BindFn>
    void qbv_bind_layer_weights(std::vector<QbvLayerWeights>& lw, const TSGgmlQwen35LayerDesc* layers,
        int num_layers, int H, int hd, int conv_kernel, int conv_dim, int num_v_heads, int head_v_dim, BindFn&& bind_or_mark)
    {
        for (int l = 0; l < num_layers; l++)
        {
            const TSGgmlQwen35LayerDesc& d = layers[l];
            QbvLayerWeights& t = lw[l];
            for (int slot = 0; slot < TSQ35_SC_COUNT; ++slot)
                if (t.psc[slot] != nullptr)
                    bind_or_mark(t.psc[slot], static_cast<float*>(d.proj_scales) + slot, sizeof(float));
            bind_or_mark(t.attn_norm_w, d.attn_norm_w, static_cast<std::size_t>(H) * sizeof(float));
            bind_or_mark(t.post_attn_norm_w, d.post_attn_norm_w, static_cast<std::size_t>(H) * sizeof(float));
            if (d.is_recurrent == 0)
            {
                bind_or_mark(t.qkv_w, d.qkv_w, static_cast<std::size_t>(d.qkv_bytes));
                if (d.separate_qkv != 0)
                {
                    bind_or_mark(t.k_w, d.k_w, static_cast<std::size_t>(d.k_bytes));
                    bind_or_mark(t.v_w, d.v_w, static_cast<std::size_t>(d.v_bytes));
                }
                bind_or_mark(t.o_w, d.o_w, static_cast<std::size_t>(d.o_bytes));
                bind_or_mark(t.q_norm_w, d.q_norm_w, static_cast<std::size_t>(hd) * sizeof(float));
                bind_or_mark(t.k_norm_w, d.k_norm_w, static_cast<std::size_t>(hd) * sizeof(float));
            }
            else
            {
                bind_or_mark(t.gdn_qkv_w, d.gdn_qkv_w, static_cast<std::size_t>(d.gdn_qkv_bytes));
                bind_or_mark(t.gdn_gate_w, d.gdn_gate_w, static_cast<std::size_t>(d.gdn_gate_bytes));
                bind_or_mark(t.ssm_beta_w, d.ssm_beta_w, static_cast<std::size_t>(d.ssm_beta_bytes));
                bind_or_mark(t.ssm_alpha_w, d.ssm_alpha_w, static_cast<std::size_t>(d.ssm_alpha_bytes));
                bind_or_mark(t.conv1d_w, d.conv1d_w, static_cast<std::size_t>(conv_kernel) * conv_dim * sizeof(float));
                bind_or_mark(t.ssm_dt_w, d.ssm_dt_w, static_cast<std::size_t>(num_v_heads) * sizeof(float));
                bind_or_mark(t.ssm_a_w, d.ssm_a_w, static_cast<std::size_t>(num_v_heads) * sizeof(float));
                bind_or_mark(t.ssm_norm_w, d.ssm_norm_w, static_cast<std::size_t>(head_v_dim) * sizeof(float));
                bind_or_mark(t.ssm_out_w, d.ssm_out_w, static_cast<std::size_t>(d.ssm_out_bytes));
            }
            if (t.gu_w != nullptr)
                bind_or_mark(t.gu_w, d.gu_w, static_cast<std::size_t>(d.gu_bytes));
            else
            {
                bind_or_mark(t.ffn_gate_w, d.ffn_gate_w, static_cast<std::size_t>(d.ffn_gate_bytes));
                bind_or_mark(t.ffn_up_w, d.ffn_up_w, static_cast<std::size_t>(d.ffn_up_bytes));
            }
            bind_or_mark(t.down_w, d.down_w, static_cast<std::size_t>(d.down_bytes));
        }
    }

    struct QbvCommit
    {
        ggml_context* ctx = nullptr;
        ggml_gallocr_t galloc = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* keep_in = nullptr;      // F32 [1, 1, T, N]: 1 for a kept row, 0 past it
        ggml_tensor* window_in = nullptr;    // I32 [convDim, N]: conv rows ending at the last kept row
        bool attempted = false;

        void release()
        {
            if (galloc != nullptr) { ggml_gallocr_free(galloc); galloc = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr;
            keep_in = window_in = nullptr;
            attempted = false;
        }
    };

    struct QbvEntry
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_gallocr_t galloc = nullptr;
        ggml_cgraph* graph = nullptr;
        std::vector<std::uintptr_t> sig;
        int n_seqs = 0, n_tok = 0, gdn_layers = 0, kvH = 0, value_dim = 0;
        int convDim = 0, conv_dim = 0, head_k = 0, head_v = 0, num_k_heads = 0, num_v_heads = 0;
        int H = 0, n_capture = 0;
        ggml_tensor* token_in = nullptr;     // I32 [N*T]
        ggml_tensor* pos_in = nullptr;       // I32 [N*T] RoPE positions
        ggml_tensor* idx_in = nullptr;       // I64 [N*T*kvH] set_rows targets
        ggml_tensor* thr_in = nullptr;       // F32 [1, T, 1, N] last visible key per row
        ggml_tensor* sampled_out = nullptr;  // I32 [N*T]
        ggml_tensor* capture_out = nullptr;  // F32 [H*n_capture, N*T]
        std::vector<int> cache_rows;
        // Per (GDN layer gl, slot s) holder bindings, index gl*N + s.
        std::vector<ggml_tensor*> conv_bound, delta_bound;
        // Per GDN layer, batched over slots: the recurrence inputs the commit replays.
        std::vector<ggml_tensor*> sv_q, sv_k, sv_v, sv_g, sv_beta, sv_conv;
        struct BoundKey { const void* host; ggml_backend_buffer_t buffer; void* addr; std::size_t bytes; };
        std::vector<BoundKey> bound_keys;
        QbvCommit commit;
        // Graph metadata memory that outlives rebuilds. Every new request (and every
        // change of the lanes decoding together) rebuilds this graph for new holders;
        // built into the same bytes, a graph of the same shape puts every node at the
        // same address, so ggml-cuda finds its captured CUDA graph by the first node
        // and updates it in place (cudaGraphExecUpdate) instead of instantiating a new
        // one - ~28 ms for the ~1.9K-kernel Qwen3.8 graph, per window shape, per request.
        // llama.cpp gets the same effect by rebuilding into one compute buffer.
        std::vector<std::uint8_t> arena;
        std::vector<std::uint8_t> commit_arena;
        int arena_seqs = 0, arena_tok = 0;   // the shape last built into the arena
        std::vector<std::int32_t> tok_stage, pos_stage, sampled_stage, window_stage;
        std::vector<std::int64_t> idx_stage;
        std::vector<float> thr_stage, keep_stage;
        std::uint64_t lru = 0;

        void release()
        {
            commit.release();
            if (galloc != nullptr) { ggml_gallocr_free(galloc); galloc = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr;
            token_in = pos_in = idx_in = thr_in = sampled_out = capture_out = nullptr;
            conv_bound.clear(); delta_bound.clear();
            sv_q.clear(); sv_k.clear(); sv_v.clear(); sv_g.clear(); sv_beta.clear(); sv_conv.clear();
            bound_keys.clear();
            sig.clear();
            cache_rows.clear();
            valid = false;
        }

        // The graph bakes the device addresses of its holders' resident copies; a
        // demotion, reseed or resize of any of them invalidates it.
        bool bindings_current() const
        {
            for (const BoundKey& k : bound_keys)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                if (!try_peek_cached_device_copy(k.host, k.bytes, buf, addr) || buf != k.buffer || addr != k.addr)
                    return false;
            }
            return true;
        }
    };

    struct QbvPool
    {
        QbvEntry entries[kQbvMaxEntries];
        std::uint64_t clock = 0;
        // The entry whose verify ran last and still awaits its commit.
        QbvEntry* pending = nullptr;
    };
    QbvPool g_qbv_pools[tsg::TSG_MAX_DEVICES];
    QbvPool& qbv_pool() { return g_qbv_pools[tsg::g_active_rank]; }

    std::mutex& qbv_mutex()
    {
        static std::mutex m;
        return m;
    }

    // Pin every tensor the graph must keep across replays and until the commit:
    // inputs and small uploaded leaves (written once at build), and the roots of
    // all outputs. Mirrors the packed persist allocation of the solo verify.
    void qbv_pin_persistent(ggml_context* ctx)
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
    }

    // A leaf descriptor over an existing allocation, so a commit graph reads and
    // writes the verify's retained tensors and the holders' copies without
    // re-running their producers.
    ggml_tensor* qbv_leaf(ggml_context* ctx, ggml_tensor* source)
    {
        if (source == nullptr || !ggml_is_contiguous(source) || source->buffer == nullptr || source->data == nullptr)
            return nullptr;
        ggml_tensor* t = ggml_dup_tensor(ctx, source);
        if (ggml_backend_tensor_alloc(source->buffer, t, source->data) != GGML_STATUS_SUCCESS)
            return nullptr;
        return t;
    }

    // The commit graph of an entry: per GDN layer and slot, replay the recurrence
    // from the holder's state and store it, and take the conv window that ends at
    // the last kept row. Every slot runs the whole window with the gate and beta of
    // its rejected rows zeroed (keep_in): below 64 rows ggml steps the recurrence
    // row by row, and a row with g = 0 and beta = 0 computes S = exp(0) * S + k * 0,
    // which leaves the state exactly where the kept prefix put it. The window is
    // gathered by index (window_in). Only those two inputs change between rounds,
    // so the graph keeps its node addresses and ggml-cuda replays one captured CUDA
    // graph; a graph per (slot, acceptance) pair - up to N x T of them - ran kernel
    // by kernel or was captured again in most rounds (~4-5 ms of host time in a
    // four-slot round).
    ggml_cgraph* qbv_commit_graph(QbvEntry& e)
    {
        QbvCommit& c = e.commit;
        if (c.attempted) return c.graph;
        c.attempted = true;

        const int N = e.n_seqs, T = e.n_tok;
        const std::size_t n_nodes = static_cast<std::size_t>(e.gdn_layers) * (16 + 16 * static_cast<std::size_t>(N)) + 64;
        const std::size_t bytes = ggml_graph_overhead_custom(n_nodes, false) + 2 * n_nodes * ggml_tensor_overhead() + 4096;
        if (e.commit_arena.size() < bytes)
            e.commit_arena.assign(bytes, 0);
        ggml_init_params ip = { e.commit_arena.size(), e.commit_arena.data(), /*no_alloc=*/true };
        c.ctx = ggml_init(ip);
        if (c.ctx == nullptr) return nullptr;
        ggml_cgraph* graph = ggml_new_graph_custom(c.ctx, n_nodes, false);
        auto fail = [&]() -> ggml_cgraph* { c.release(); c.attempted = true; return nullptr; };

        c.keep_in = ggml_new_tensor_4d(c.ctx, GGML_TYPE_F32, 1, 1, T, N);
        c.window_in = ggml_new_tensor_2d(c.ctx, GGML_TYPE_I32, e.convDim, N);
        ggml_set_input(c.keep_in);
        ggml_set_input(c.window_in);

        for (int gl = 0; gl < e.gdn_layers; gl++)
        {
            ggml_tensor* q = qbv_leaf(c.ctx, e.sv_q[gl]);
            ggml_tensor* k = qbv_leaf(c.ctx, e.sv_k[gl]);
            ggml_tensor* v = qbv_leaf(c.ctx, e.sv_v[gl]);
            ggml_tensor* g = qbv_leaf(c.ctx, e.sv_g[gl]);
            ggml_tensor* b = qbv_leaf(c.ctx, e.sv_beta[gl]);
            ggml_tensor* conv = qbv_leaf(c.ctx, e.sv_conv[gl]);
            if (q == nullptr || k == nullptr || v == nullptr || g == nullptr || b == nullptr || conv == nullptr)
                return fail();
            ggml_tensor* g_kept = ggml_mul(c.ctx, g, c.keep_in);    // [1, nv, T, N]
            ggml_tensor* b_kept = ggml_mul(c.ctx, b, c.keep_in);
            // conv is [convDim + T, conv_dim, N]; time-major rows gather by index.
            ggml_tensor* by_time = ggml_cont(c.ctx, ggml_permute(c.ctx, conv, 1, 0, 2, 3));
            ggml_tensor* windows = ggml_get_rows(c.ctx, by_time, c.window_in);   // [conv_dim, convDim, N]

            for (int s = 0; s < N; s++)
            {
                const std::size_t ix = static_cast<std::size_t>(gl) * N + s;
                ggml_tensor* state = qbv_leaf(c.ctx, e.delta_bound[ix]);
                ggml_tensor* conv_state = qbv_leaf(c.ctx, e.conv_bound[ix]);
                if (state == nullptr || conv_state == nullptr)
                    return fail();
                auto slot = [&](ggml_tensor* t) {
                    return ggml_view_4d(c.ctx, t, t->ne[0], t->ne[1], T, 1,
                        t->nb[1], t->nb[2], t->nb[3], static_cast<std::size_t>(s) * t->nb[3]);
                };
                ggml_tensor* gdn = ggml_gated_delta_net(c.ctx, slot(q), slot(k), slot(v), slot(g_kept), slot(b_kept), state, 1);
                if (gl == 0 && s == 0 && !backend_supports_op(gdn))
                    return fail();
                ggml_tensor* new_state = ggml_view_4d(c.ctx, gdn,
                    e.head_k, e.head_v, e.num_v_heads, 1,
                    ggml_row_size(gdn->type, e.head_k),
                    ggml_row_size(gdn->type, static_cast<std::int64_t>(e.head_k) * e.head_v),
                    ggml_row_size(gdn->type, static_cast<std::int64_t>(e.head_k) * e.head_v * e.num_v_heads),
                    ggml_row_size(gdn->type, e.value_dim) * static_cast<std::size_t>(T));
                ggml_build_forward_expand(graph, ggml_cpy(c.ctx, new_state, state));

                ggml_tensor* window = ggml_view_2d(c.ctx, windows, e.conv_dim, e.convDim, windows->nb[1],
                    static_cast<std::size_t>(s) * windows->nb[2]);
                ggml_build_forward_expand(graph, ggml_cpy(c.ctx, ggml_transpose(c.ctx, window), conv_state));
            }
        }

        c.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (c.galloc == nullptr || !ggml_gallocr_alloc_graph(c.galloc, graph))
            return fail();
        c.graph = graph;
        return graph;
    }
}

// ============================================================================
// layers: the slot-invariant descriptors the arena decode uses (their cache and
//         state pointers are ignored; the arrays below carry each slot's).
// token_ids:      [n_seqs * n_tok], sequence-major.
// positions:      [n_seqs] KV index of each sequence's first token.
// rope_positions: [n_seqs * n_tok] RoPE positions, or null for KV indices.
// k/v_cache_arr:  [attn_layer * n_seqs + s]; conv/delta_arr: [gdn_layer * n_seqs + s]
//                 (conv state in ggml [time, channel] layout, as the arena binds).
// capture_layers: DFlash taps (residual entering each listed layer); capture_data
//                 receives [n_seqs * n_tok rows][capture_count * hidden] row-major.
// sampled_data:   [n_seqs * n_tok] argmax of each row (first max on ties).
// Returns 1, 0 (declined, nothing written) or -1 (failed after K/V writes began;
// recurrent state is still untouched, but the call must not be retried blindly).
// ============================================================================
TSG_EXPORT int TSGgml_Qwen35BatchedVerify(
    const TSGgmlQwen35LayerDesc* layers, int num_layers, int n_seqs, int n_tok,
    const std::int32_t* token_ids, const std::int32_t* positions, const std::int32_t* rope_positions,
    void** k_cache_arr, void** v_cache_arr, void** conv_state_arr, void** delta_state_arr,
    const std::int32_t* cache_sizes,
    int num_heads, int num_kv_heads, int head_dim, int rope_n_dims, int rope_mode, int kv_cache_type,
    int conv_kernel, int head_k_dim, int head_v_dim, int num_k_heads, int num_v_heads,
    float eps, float rope_base, float rope_freq_scale,
    int vocab_size,
    const void* lm_head_data, int lm_head_type,
    std::int64_t lm_head_ne0, std::int64_t lm_head_ne1, std::int64_t lm_head_bytes,
    const void* final_norm_data,
    const void* token_embd_data, int token_embd_type,
    std::int64_t token_embd_ne0, std::int64_t token_embd_ne1, std::int64_t token_embd_bytes,
    const std::int32_t* capture_layers, int capture_count, float* capture_data,
    std::int32_t* sampled_data)
{
    bool compute_started = false;
    try
    {
        if (!ensure_backend())
            return 0;
        if (g_backend_type != BACKEND_TYPE_CUDA)
        {
            set_last_error("Qwen3.5 batched verify: requires CUDA.");
            return 0;
        }
        if (layers == nullptr || num_layers <= 0 || n_seqs < 1 || n_tok < 1 || n_tok > 16 ||
            token_ids == nullptr || positions == nullptr || cache_sizes == nullptr ||
            k_cache_arr == nullptr || v_cache_arr == nullptr || conv_state_arr == nullptr ||
            delta_state_arr == nullptr || sampled_data == nullptr ||
            lm_head_data == nullptr || final_norm_data == nullptr || token_embd_data == nullptr ||
            (capture_count > 0 && (capture_layers == nullptr || capture_data == nullptr)))
        {
            set_last_error("Qwen3.5 batched verify: invalid arguments.");
            return 0;
        }
        if (layers[0].struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwen35LayerDesc)))
        {
            set_last_error("Qwen3.5 batched verify: descriptor size mismatch.");
            return 0;
        }
        if (kv_cache_type != GGML_TYPE_F32 && kv_cache_type != GGML_TYPE_F16 &&
            kv_cache_type != GGML_TYPE_Q8_0 && kv_cache_type != GGML_TYPE_Q4_0)
        {
            set_last_error("Qwen3.5 batched verify: F32/F16/Q8_0/Q4_0 KV cache only.");
            return 0;
        }
        if (head_k_dim != head_v_dim || conv_kernel <= 1)
        {
            set_last_error("Qwen3.5 batched verify: unsupported GDN geometry.");
            return 0;
        }
        int attn_layers = 0, gdn_layers = 0;
        for (int l = 0; l < num_layers; l++)
        {
            if (layers[l].is_moe != 0)
            {
                set_last_error("Qwen3.5 batched verify: dense FFN only.");
                return 0;
            }
            (layers[l].is_recurrent != 0 ? gdn_layers : attn_layers)++;
        }
        if (attn_layers == 0 || gdn_layers == 0)
        {
            set_last_error("Qwen3.5 batched verify: hybrid attention/GDN stack required.");
            return 0;
        }
        for (int c = 0; c < capture_count; c++)
            if (capture_layers[c] < 0 || capture_layers[c] >= num_layers)
            {
                set_last_error("Qwen3.5 batched verify: a DFlash capture layer is outside the trunk.");
                return 0;
            }
        if (token_embd_type < 0 || token_embd_type >= GGML_TYPE_COUNT || token_embd_ne0 <= 0 ||
            token_embd_ne0 > std::numeric_limits<int>::max() ||
            token_embd_ne0 % ggml_blck_size(static_cast<ggml_type>(token_embd_type)) != 0)
        {
            set_last_error("Qwen3.5 batched verify: bad token embedding.");
            return 0;
        }
        const int N = n_seqs, T = n_tok, NT = n_seqs * n_tok;
        for (int i = 0; i < NT; i++)
            if (token_ids[i] < 0 || token_ids[i] >= token_embd_ne1)
            {
                set_last_error("Qwen3.5 batched verify: token id out of range.");
                return 0;
            }
        for (int s = 0; s < N; s++)
            if (positions[s] < 0 || cache_sizes[s] <= 0 ||
                static_cast<std::int64_t>(positions[s]) + T > cache_sizes[s])
            {
                set_last_error("Qwen3.5 batched verify: a window exceeds its cache rows (no-wrap).");
                return 0;
            }

        const int H = static_cast<int>(token_embd_ne0);
        const ggml_type kvType = static_cast<ggml_type>(kv_cache_type);
        const int hd = head_dim, kvH = num_kv_heads, nH = num_heads;
        const int qDim = nH * hd, qFullDim = qDim * 2, kDim = kvH * hd;
        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(hd));
        const int convDim = conv_kernel - 1;
        const int key_dim = head_k_dim * num_k_heads;
        const int value_dim = head_v_dim * num_v_heads;
        const int conv_dim = 2 * key_dim + value_dim;
        const std::size_t convBytes = static_cast<std::size_t>(convDim) * conv_dim * sizeof(float);
        const std::size_t deltaBytes = static_cast<std::size_t>(head_k_dim) * head_v_dim * num_v_heads * sizeof(float);
        const std::size_t row_bytes = ggml_row_size(kvType, hd);

        std::vector<int> windows(N);
        std::vector<std::uintptr_t> sig;
        sig.reserve(8 + static_cast<std::size_t>(N) * (2 + 2 * num_layers) + capture_count);
        sig.push_back(reinterpret_cast<std::uintptr_t>(layers[0].attn_norm_w));
        sig.push_back(static_cast<std::uintptr_t>(N));
        sig.push_back(static_cast<std::uintptr_t>(T));
        sig.push_back(static_cast<std::uintptr_t>(kvType));
        for (int s = 0; s < N; s++)
        {
            windows[s] = static_cast<int>(std::min<std::int64_t>(cache_sizes[s],
                qbv_window_round(static_cast<std::int64_t>(positions[s]) + T)));
            sig.push_back(static_cast<std::uintptr_t>(cache_sizes[s]));
            sig.push_back(static_cast<std::uintptr_t>(windows[s]));
        }
        for (std::size_t j = 0; j < static_cast<std::size_t>(attn_layers) * N; j++)
        {
            sig.push_back(reinterpret_cast<std::uintptr_t>(k_cache_arr[j]));
            sig.push_back(reinterpret_cast<std::uintptr_t>(v_cache_arr[j]));
        }
        for (std::size_t j = 0; j < static_cast<std::size_t>(gdn_layers) * N; j++)
        {
            sig.push_back(reinterpret_cast<std::uintptr_t>(conv_state_arr[j]));
            sig.push_back(reinterpret_cast<std::uintptr_t>(delta_state_arr[j]));
        }
        sig.push_back(static_cast<std::uintptr_t>(capture_count));
        for (int c = 0; c < capture_count; c++)
            sig.push_back(static_cast<std::uintptr_t>(capture_layers[c]));

        std::lock_guard<std::mutex> lock(qbv_mutex());
        QbvPool& pool = qbv_pool();
        pool.pending = nullptr;   // an uncommitted earlier verify is abandoned (its holders were never advanced)

        QbvEntry* entry = nullptr;
        for (auto& e : pool.entries)
            if (e.valid && e.sig == sig) { entry = &e; break; }
        if (entry != nullptr && !entry->bindings_current())
        {
            entry->release();
            entry = nullptr;
        }

        if (entry == nullptr)
        {
            // Windows only grow while the same holders decode together, so an entry
            // for the same holder keys at older windows will not replay again.
            for (auto& e : pool.entries)
                if (e.valid && e.sig.size() == sig.size() && e.sig[1] == sig[1] && e.sig[2] == sig[2] &&
                    std::equal(sig.begin() + 4 + 2 * N, sig.end(), e.sig.begin() + 4 + 2 * N))
                    e.release();
            // Prefer the slot whose arena last held this shape: the rebuild then lands
            // at the same node addresses and updates its CUDA graph in place.
            for (auto& e : pool.entries)
                if (!e.valid && e.arena_seqs == N && e.arena_tok == T) { entry = &e; break; }
            if (entry == nullptr)
                for (auto& e : pool.entries)
                    if (e.valid && e.arena_seqs == N && e.arena_tok == T) { entry = &e; break; }
            if (entry == nullptr)
                for (auto& e : pool.entries)
                    if (!e.valid) { entry = &e; break; }
            if (entry == nullptr)
            {
                entry = &pool.entries[0];
                for (auto& e : pool.entries)
                    if (e.lru < entry->lru) entry = &e;
            }
            entry->release();
            QbvEntry& e = *entry;
            e.arena_seqs = N;
            e.arena_tok = T;
            e.sig = sig;
            e.n_seqs = N; e.n_tok = T; e.gdn_layers = gdn_layers; e.kvH = kvH; e.value_dim = value_dim;
            e.convDim = convDim; e.conv_dim = conv_dim; e.head_k = head_k_dim; e.head_v = head_v_dim;
            e.num_k_heads = num_k_heads; e.num_v_heads = num_v_heads; e.H = H; e.n_capture = capture_count;
            e.cache_rows.assign(cache_sizes, cache_sizes + N);
            const std::int64_t mask_len = *std::max_element(windows.begin(), windows.end());

            const std::size_t graph_size = static_cast<std::size_t>(num_layers) * (256 + 48 * static_cast<std::size_t>(N)) + 8192;
            const std::size_t ctx_bytes = ggml_graph_overhead_custom(graph_size, false) + graph_size * 3 * ggml_tensor_overhead() + (1 << 20);
            if (e.arena.size() < ctx_bytes)
                e.arena.assign(ctx_bytes, 0);
            ggml_init_params ip = { e.arena.size(), e.arena.data(), /*no_alloc=*/true };
            e.ctx = ggml_init(ip);
            if (e.ctx == nullptr)
            {
                set_last_error("Qwen3.5 batched verify: failed to init ggml context.");
                return 0;
            }
            ggml_context* ctx = e.ctx;
            auto abort_build = [&](const char* msg) -> int {
                set_last_error(std::string("Qwen3.5 batched verify: ") + msg);
                e.release();
                return 0;
            };

            e.token_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NT);
            e.pos_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, NT);
            e.idx_in = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, static_cast<std::int64_t>(NT) * kvH);
            e.thr_in = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, T, 1, N);
            ggml_set_input(e.token_in);
            ggml_set_input(e.pos_in);
            ggml_set_input(e.idx_in);
            ggml_set_input(e.thr_in);

            // Holder bindings (allocated in place over the resident copies below).
            std::vector<ggml_tensor*> k_bound(static_cast<std::size_t>(attn_layers) * N);
            std::vector<ggml_tensor*> v_bound(static_cast<std::size_t>(attn_layers) * N);
            e.conv_bound.assign(static_cast<std::size_t>(gdn_layers) * N, nullptr);
            e.delta_bound.assign(static_cast<std::size_t>(gdn_layers) * N, nullptr);
            for (int al = 0; al < attn_layers; al++)
                for (int s = 0; s < N; s++)
                {
                    const std::int64_t rows = static_cast<std::int64_t>(kvH) * cache_sizes[s];
                    k_bound[static_cast<std::size_t>(al) * N + s] = ggml_new_tensor_2d(ctx, kvType, hd, rows);
                    v_bound[static_cast<std::size_t>(al) * N + s] = ggml_new_tensor_2d(ctx, kvType, hd, rows);
                }
            for (std::size_t j = 0; j < e.conv_bound.size(); j++)
            {
                e.conv_bound[j] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, convDim, conv_dim);
                e.delta_bound[j] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, head_k_dim, head_v_dim, num_v_heads, 1);
            }

            std::vector<QbvLayerWeights> lw = qbv_new_layer_weights(ctx, layers, num_layers, H, hd, conv_kernel,
                conv_dim, num_v_heads, head_v_dim);
            ggml_tensor* token_embd_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(token_embd_type), token_embd_ne0, token_embd_ne1);
            ggml_tensor* lm_head_t = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(lm_head_type), lm_head_ne0, lm_head_ne1);
            ggml_tensor* final_norm_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);

            tsg::BonsaiGraphScope bonsai_scope;
            std::vector<ggml_tensor*> side_effects;

            // ---- causal masks: one [mask_len, T] plane per slot, built on device ----
            // step(key - last_visible) * -1e30 is 0 on visible keys and overflows to
            // -inf in F16 on hidden ones (the solo verify's device mask).
            ggml_tensor* keys = ggml_repeat_4d(ctx,
                ggml_reshape_4d(ctx, ggml_arange(ctx, 0.0f, static_cast<float>(mask_len), 1.0f), mask_len, 1, 1, 1),
                mask_len, T, 1, N);
            ggml_tensor* hidden_keys = ggml_step_inplace(ctx, ggml_sub_inplace(ctx, keys, e.thr_in));
            ggml_tensor* attn_mask = ggml_cast(ctx, ggml_scale_inplace(ctx, hidden_keys, -1e30f), GGML_TYPE_F16); // [mask_len, T, 1, N]
            // flash attention wants a contiguous [window, T] mask: one per slot, shared
            // by every attention layer.
            std::vector<ggml_tensor*> slot_masks(N);
            for (int s = 0; s < N; s++)
                slot_masks[s] = ggml_cont(ctx, ggml_view_2d(ctx, attn_mask, windows[s], T, attn_mask->nb[1],
                    static_cast<std::size_t>(s) * attn_mask->nb[3]));

            ggml_tensor* rows = ggml_get_rows(ctx, token_embd_t, e.token_in);
            if (!backend_supports_op(rows))
                return abort_build("token embedding get_rows unsupported.");
            ggml_tensor* hidden = tsg::bonsai_transform(ctx, rows, token_embd_data, true);   // [H, NT]

            if (capture_count > 0)
            {
                e.capture_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, static_cast<std::int64_t>(H) * capture_count, NT);
                ggml_set_output(e.capture_out);
            }
            e.sv_q.assign(gdn_layers, nullptr); e.sv_k.assign(gdn_layers, nullptr); e.sv_v.assign(gdn_layers, nullptr);
            e.sv_g.assign(gdn_layers, nullptr); e.sv_beta.assign(gdn_layers, nullptr); e.sv_conv.assign(gdn_layers, nullptr);

            for (int l = 0, al = 0, gl = 0; l < num_layers; l++)
            {
                const TSGgmlQwen35LayerDesc& d = layers[l];
                QbvLayerWeights& t = lw[l];
                for (int c = 0; c < capture_count; c++)
                {
                    if (capture_layers[c] != l) continue;
                    ggml_tensor* dst = ggml_view_2d(ctx, e.capture_out, H, NT, e.capture_out->nb[1],
                        static_cast<std::size_t>(c) * H * sizeof(float));
                    side_effects.push_back(ggml_cpy(ctx, hidden, dst));
                }
                ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), t.attn_norm_w);   // [H, NT]
                ggml_tensor* block_out;

                if (d.is_recurrent == 0)
                {
                    ggml_tensor* qg_part;
                    ggml_tensor* k_raw;
                    ggml_tensor* v_raw;
                    if (d.separate_qkv != 0)
                    {
                        qg_part = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.qkv_w, normed, d.qkv_w), q35_psc(ctx, t, d, TSQ35_SC_QKV));
                        k_raw = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.k_w, normed, d.k_w), q35_psc(ctx, t, d, TSQ35_SC_K));
                        v_raw = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.v_w, normed, d.v_w), q35_psc(ctx, t, d, TSQ35_SC_V));
                    }
                    else
                    {
                        ggml_tensor* qkv = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.qkv_w, normed, d.qkv_w), q35_psc(ctx, t, d, TSQ35_SC_QKV));
                        qg_part = ggml_view_2d(ctx, qkv, qFullDim, NT, qkv->nb[1], 0);
                        k_raw = ggml_view_2d(ctx, qkv, kDim, NT, qkv->nb[1], static_cast<std::size_t>(qFullDim) * sizeof(float));
                        v_raw = ggml_view_2d(ctx, qkv, kDim, NT, qkv->nb[1], static_cast<std::size_t>(qFullDim + kDim) * sizeof(float));
                    }
                    ggml_tensor* qg_4d = ggml_reshape_4d(ctx, ggml_cont(ctx, qg_part), hd, 2, nH, NT);
                    ggml_tensor* q_view = ggml_view_3d(ctx, qg_4d, hd, nH, NT, qg_4d->nb[2], qg_4d->nb[3], 0);
                    ggml_tensor* gate_view = ggml_view_3d(ctx, qg_4d, hd, nH, NT, qg_4d->nb[2], qg_4d->nb[3], qg_4d->nb[1]);
                    ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_cont(ctx, q_view), eps), t.q_norm_w);
                    ggml_tensor* k_normed = ggml_mul(ctx,
                        ggml_rms_norm(ctx, ggml_reshape_3d(ctx, ggml_cont(ctx, k_raw), hd, kvH, NT), eps), t.k_norm_w);
                    // Same RMS_NORM+MUL / ROPE boundary as the arena and solo decode.
                    ggml_set_output(q_normed);
                    ggml_set_output(k_normed);
                    ggml_tensor* q_rope = ggml_rope_ext(ctx, q_normed, e.pos_in, nullptr,
                        rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);   // [hd, nH, NT]
                    ggml_tensor* k_rope = ggml_rope_ext(ctx, k_normed, e.pos_in, nullptr,
                        rope_n_dims, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);   // [hd, kvH, NT]
                    ggml_tensor* v_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, v_raw), hd, kvH, NT);

                    ggml_tensor* attn_all = nullptr;
                    for (int s = 0; s < N; s++)
                    {
                        const std::size_t ix = static_cast<std::size_t>(al) * N + s;
                        // Rows of slot s in (token, head) order; idx carries head*rows + pos + t.
                        ggml_tensor* idx_s = ggml_view_1d(ctx, e.idx_in, static_cast<std::int64_t>(T) * kvH,
                            static_cast<std::size_t>(s) * T * kvH * sizeof(std::int64_t));
                        ggml_tensor* k_rows = ggml_view_2d(ctx, k_rope, hd, static_cast<std::int64_t>(kvH) * T,
                            k_rope->nb[1], static_cast<std::size_t>(s) * T * k_rope->nb[2]);
                        ggml_tensor* v_rows = ggml_view_2d(ctx, v_3d, hd, static_cast<std::int64_t>(kvH) * T,
                            v_3d->nb[1], static_cast<std::size_t>(s) * T * v_3d->nb[2]);
                        ggml_tensor* ks = ggml_set_rows(ctx, k_bound[ix], k_rows, idx_s);
                        ggml_tensor* vs = ggml_set_rows(ctx, v_bound[ix], v_rows, idx_s);
                        if (!backend_supports_op(ks))
                            return abort_build("set_rows unsupported for the verify shapes.");
                        side_effects.push_back(ks);
                        side_effects.push_back(vs);

                        const std::int64_t cap = cache_sizes[s];
                        ggml_tensor* k_slot = ggml_view_3d(ctx, ks, hd, cap, kvH, row_bytes, static_cast<std::size_t>(cap) * row_bytes, 0);
                        ggml_tensor* v_slot = ggml_view_3d(ctx, vs, hd, cap, kvH, row_bytes, static_cast<std::size_t>(cap) * row_bytes, 0);
                        ggml_tensor* k_view = view_kv_cache_window(ctx, k_slot, hd, static_cast<int>(cap), kvH, 0, windows[s],
                            kv_cache_type, T, nH / kvH);
                        ggml_tensor* v_view = view_kv_cache_window(ctx, v_slot, hd, static_cast<int>(cap), kvH, 0, windows[s],
                            kv_cache_type, T, nH / kvH);
                        if (k_view == nullptr || v_view == nullptr)
                            return abort_build("failed to create per-slot KV windows.");
                        ggml_tensor* q_slot = ggml_view_3d(ctx, q_rope, hd, T, nH, q_rope->nb[2], q_rope->nb[1],
                            static_cast<std::size_t>(s) * T * q_rope->nb[2]);   // [hd, T, nH]
                        ggml_tensor* fa = flash_attn_ext_guarded(ctx, kQbvKernel, q_slot, k_view, v_view, slot_masks[s],
                            attn_scale, 0.0f, 0.0f, nullptr, GGML_PREC_F32);   // [hd, nH, T]
                        ggml_tensor* attn_s = ggml_reshape_2d(ctx, fa, qDim, T);
                        attn_all = attn_all == nullptr ? attn_s : ggml_concat(ctx, attn_all, attn_s, 1);
                    }
                    ggml_tensor* gate_flat = ggml_reshape_2d(ctx, ggml_cont(ctx, gate_view), qDim, NT);
                    ggml_tensor* attn_gated = ggml_mul(ctx, attn_all, ggml_sigmoid(ctx, gate_flat));
                    block_out = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.o_w, attn_gated, d.o_w), q35_psc(ctx, t, d, TSQ35_SC_O));
                    al++;
                }
                else
                {
                    ggml_tensor* qkv_mixed;
                    ggml_tensor* z;
                    ggml_tensor* beta_raw;
                    ggml_tensor* alpha_raw;
                    if (t.gdn_gate_w == nullptr)
                    {
                        ggml_tensor* packed = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.gdn_qkv_w, normed, d.gdn_qkv_w), q35_psc(ctx, t, d, TSQ35_SC_GDN_QKV));
                        qkv_mixed = ggml_view_2d(ctx, packed, conv_dim, NT, packed->nb[1], 0);
                        z = ggml_view_2d(ctx, packed, value_dim, NT, packed->nb[1], static_cast<std::size_t>(conv_dim) * sizeof(float));
                        beta_raw = ggml_view_2d(ctx, packed, num_v_heads, NT, packed->nb[1],
                            static_cast<std::size_t>(conv_dim + value_dim) * sizeof(float));
                        alpha_raw = ggml_view_2d(ctx, packed, num_v_heads, NT, packed->nb[1],
                            static_cast<std::size_t>(conv_dim + value_dim + num_v_heads) * sizeof(float));
                    }
                    else
                    {
                        qkv_mixed = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.gdn_qkv_w, normed, d.gdn_qkv_w), q35_psc(ctx, t, d, TSQ35_SC_GDN_QKV));
                        z = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.gdn_gate_w, normed, d.gdn_gate_w), q35_psc(ctx, t, d, TSQ35_SC_GDN_GATE));
                        beta_raw = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.ssm_beta_w, normed, d.ssm_beta_w), q35_psc(ctx, t, d, TSQ35_SC_BETA));
                        alpha_raw = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.ssm_alpha_w, normed, d.ssm_alpha_w), q35_psc(ctx, t, d, TSQ35_SC_ALPHA));
                    }
                    ggml_tensor* beta_all = ggml_cont(ctx, ggml_sigmoid(ctx, ggml_cont(ctx, beta_raw)));            // [nv, NT]
                    ggml_tensor* g_all = ggml_mul(ctx,
                        ggml_softplus(ctx, ggml_add(ctx, ggml_cont(ctx, alpha_raw), t.ssm_dt_w)), t.ssm_a_w);    // [nv, NT]

                    // Gather the slots' conv states into a batch with a balanced concat
                    // tree: real source edges order every write before its readers, and
                    // the batch is an ordinary activation the allocator recycles (a
                    // scratch LEAF per layer would be allocated up front and stay live
                    // for the whole graph). The delta states are not gathered: each
                    // slot's recurrence reads its holder's state in place below, since
                    // copying 3 MB per slot and layer cost ~1.5 ms a 4-slot round.
                    std::vector<ggml_tensor*> conv_parts(N);
                    for (int s = 0; s < N; s++)
                        conv_parts[s] = e.conv_bound[static_cast<std::size_t>(gl) * N + s];
                    auto gather = [&](std::vector<ggml_tensor*> parts, int dim) {
                        while (parts.size() > 1)
                        {
                            std::vector<ggml_tensor*> next;
                            for (std::size_t i = 0; i + 1 < parts.size(); i += 2)
                                next.push_back(ggml_concat(ctx, parts[i], parts[i + 1], dim));
                            if (parts.size() % 2 != 0)
                                next.push_back(parts.back());
                            parts.swap(next);
                        }
                        return parts[0];
                    };
                    ggml_tensor* conv_states = gather(conv_parts, 2);     // [convDim, conv_dim, N]

                    // [convDim + T, conv_dim, N]: each slot's window prepended with its state.
                    ggml_tensor* x_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, qkv_mixed), conv_dim, T, N);
                    ggml_tensor* conv_input = ggml_concat(ctx, conv_states, ggml_permute(ctx, x_3d, 1, 0, 2, 3), 0);
                    ggml_set_output(conv_input);
                    ggml_tensor* conv_out = ggml_silu(ctx, tsg::ssm_conv_any(ctx, conv_input, t.conv1d_w));   // [conv_dim, T, N]
                    ggml_tensor* conv_2d = ggml_reshape_2d(ctx, conv_out, conv_dim, NT);

                    ggml_tensor* q_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_2d, key_dim, NT, conv_2d->nb[1], 0));
                    ggml_tensor* k_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_2d, key_dim, NT, conv_2d->nb[1],
                        static_cast<std::size_t>(key_dim) * sizeof(float)));
                    ggml_tensor* v_part = ggml_cont(ctx, ggml_view_2d(ctx, conv_2d, value_dim, NT, conv_2d->nb[1],
                        static_cast<std::size_t>(2 * key_dim) * sizeof(float)));
                    ggml_tensor* q_hn = tsg::gdn_l2_norm_heads(ctx, q_part, head_k_dim, num_k_heads, NT, eps);
                    ggml_tensor* k_hn = tsg::gdn_l2_norm_heads(ctx, k_part, head_k_dim, num_k_heads, NT, eps);
                    ggml_tensor* q4 = ggml_reshape_4d(ctx, q_hn, head_k_dim, num_k_heads, T, N);
                    ggml_tensor* k4 = ggml_reshape_4d(ctx, k_hn, head_k_dim, num_k_heads, T, N);
                    ggml_tensor* v4 = ggml_reshape_4d(ctx, v_part, head_v_dim, num_v_heads, T, N);
                    ggml_tensor* g4 = ggml_reshape_4d(ctx, ggml_cont(ctx, g_all), 1, num_v_heads, T, N);
                    ggml_tensor* b4 = ggml_reshape_4d(ctx, beta_all, 1, num_v_heads, T, N);
                    for (ggml_tensor* kept : { q4, k4, v4, g4, b4 })
                        ggml_set_output(kept);
                    e.sv_q[gl] = q4; e.sv_k[gl] = k4; e.sv_v[gl] = v4; e.sv_g[gl] = g4; e.sv_beta[gl] = b4;
                    e.sv_conv[gl] = conv_input;

                    // One recurrence per slot over its holder's state; only the [value_dim, T]
                    // outputs are joined.
                    std::vector<ggml_tensor*> out_parts(N);
                    for (int s = 0; s < N; s++)
                    {
                        auto slot = [&](ggml_tensor* t) {
                            return ggml_view_4d(ctx, t, t->ne[0], t->ne[1], T, 1, t->nb[1], t->nb[2], t->nb[3],
                                static_cast<std::size_t>(s) * t->nb[3]);
                        };
                        ggml_tensor* gdn = ggml_gated_delta_net(ctx, slot(q4), slot(k4), slot(v4), slot(g4), slot(b4),
                            e.delta_bound[static_cast<std::size_t>(gl) * N + s], 1);
                        if (gl == 0 && s == 0 && !backend_supports_op(gdn))
                            return abort_build("gated_delta_net unsupported for the verify shapes.");
                        out_parts[s] = ggml_view_2d(ctx, gdn, value_dim, T, ggml_row_size(gdn->type, value_dim), 0);
                    }
                    ggml_tensor* gdn_out = N == 1 ? ggml_cont(ctx, out_parts[0]) : gather(out_parts, 1);   // [value_dim, NT]

                    ggml_tensor* out_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, gdn_out), head_v_dim, num_v_heads, NT);
                    ggml_tensor* out_n = ggml_mul(ctx, ggml_rms_norm(ctx, out_3d, eps), t.ssm_norm_w);
                    ggml_tensor* z_3d = ggml_reshape_3d(ctx, ggml_cont(ctx, z), head_v_dim, num_v_heads, NT);
                    ggml_tensor* gated = ggml_reshape_2d(ctx, ggml_mul(ctx, out_n, ggml_silu(ctx, z_3d)), value_dim, NT);
                    block_out = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.ssm_out_w, gated, d.ssm_out_w), q35_psc(ctx, t, d, TSQ35_SC_SSM_OUT));
                    gl++;
                }

                ggml_tensor* residual1 = ggml_add(ctx, hidden, block_out);
                ggml_tensor* ffn_normed = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), t.post_attn_norm_w);
                ggml_tensor* act;
                if (t.gu_w != nullptr)
                    act = ggml_swiglu(ctx, q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.gu_w, ffn_normed, d.gu_w), q35_psc(ctx, t, d, TSQ35_SC_GU)));
                else
                {
                    ggml_tensor* g2 = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.ffn_gate_w, ffn_normed, d.ffn_gate_w), q35_psc(ctx, t, d, TSQ35_SC_FFN_GATE));
                    ggml_tensor* u2 = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.ffn_up_w, ffn_normed, d.ffn_up_w), q35_psc(ctx, t, d, TSQ35_SC_FFN_UP));
                    act = ggml_mul(ctx, ggml_silu(ctx, g2), u2);
                }
                ggml_tensor* ffn_down = q35_scaled(ctx, tsg::bonsai_mul_mat(ctx, t.down_w, act, d.down_w), q35_psc(ctx, t, d, TSQ35_SC_DOWN));
                hidden = ggml_add(ctx, residual1, ffn_down);
            }

            ggml_tensor* fn = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), final_norm_t);
            ggml_tensor* logits = tsg::bonsai_mul_mat(ctx, lm_head_t, fn, lm_head_data);   // [vocab, NT]
            e.sampled_out = ggml_argmax(ctx, logits);
            if (!backend_supports_op(e.sampled_out))
                return abort_build("argmax unsupported.");
            ggml_set_output(e.sampled_out);

            e.graph = ggml_new_graph_custom(ctx, graph_size, false);
            for (ggml_tensor* node : side_effects)
            {
                ggml_set_output(node);
                ggml_build_forward_expand(e.graph, node);
            }
            ggml_build_forward_expand(e.graph, e.sampled_out);

            // ---- bind weights and holder copies ----
            ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
            struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
            std::vector<HostBinding> upload_list;
            auto bind_or_mark = [&](ggml_tensor* tgt, const void* data, std::size_t bytes) {
                if (tgt == nullptr || data == nullptr) return;
                if (bytes >= 4096)
                {
                    bool needs_upload = false;
                    if (try_bind_cached_tensor(g_backend, dev, tgt, const_cast<void*>(data), bytes, needs_upload,
                                               GGML_BACKEND_BUFFER_USAGE_WEIGHTS))
                    {
                        if (needs_upload) upload_list.push_back({tgt, const_cast<void*>(data), bytes});
                        return;
                    }
                }
                upload_list.push_back({tgt, const_cast<void*>(data), bytes});
            };
            qbv_bind_layer_weights(lw, layers, num_layers, H, hd, conv_kernel, conv_dim, num_v_heads, head_v_dim, bind_or_mark);
            bind_or_mark(lm_head_t, lm_head_data, static_cast<std::size_t>(lm_head_bytes));
            bind_or_mark(final_norm_t, final_norm_data, static_cast<std::size_t>(H) * sizeof(float));
            bind_or_mark(token_embd_t, token_embd_data, static_cast<std::size_t>(token_embd_bytes));

            // Holders: the solo decode's usage and sizes, so both resolve to one copy.
            // A copy created here was never written; fill it from the host bytes.
            auto bind_holder = [&](ggml_tensor* tgt, const void* host, std::size_t bytes) -> bool {
                bool needs_upload = false;
                if (!try_bind_cached_tensor(g_backend, dev, tgt, const_cast<void*>(host), bytes, needs_upload,
                                            GGML_BACKEND_BUFFER_USAGE_COMPUTE))
                    return false;
                if (needs_upload)
                    ggml_backend_tensor_set(tgt, host, 0, bytes);
                e.bound_keys.push_back({host, tgt->buffer, tgt->data, bytes});
                return true;
            };
            host_read_barrier();
            for (int al = 0; al < attn_layers; al++)
                for (int s = 0; s < N; s++)
                {
                    const std::size_t ix = static_cast<std::size_t>(al) * N + s;
                    const std::size_t kv_bytes = kv_cache_bytes(kvH, cache_sizes[s], hd, kv_cache_type);
                    if (!bind_holder(k_bound[ix], k_cache_arr[ix], kv_bytes) || !bind_holder(v_bound[ix], v_cache_arr[ix], kv_bytes))
                        return abort_build("failed to bind a holder's resident K/V copy.");
                }
            for (std::size_t j = 0; j < e.conv_bound.size(); j++)
                if (!bind_holder(e.conv_bound[j], conv_state_arr[j], convBytes) ||
                    !bind_holder(e.delta_bound[j], delta_state_arr[j], deltaBytes))
                    return abort_build("failed to bind a holder's resident recurrent state.");

            qbv_pin_persistent(ctx);
            e.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
            if (e.galloc == nullptr || !ggml_gallocr_alloc_graph(e.galloc, e.graph))
                return abort_build("failed to allocate the verify graph.");
            host_read_barrier();
            for (auto& u : upload_list)
                ggml_backend_tensor_set(u.tensor, resolve_upload_source(u.data), 0, u.bytes);

            e.commit.release();
            e.keep_stage.assign(NT, 0.0f);
            e.window_stage.assign(static_cast<std::size_t>(convDim) * N, 0);
            e.tok_stage.assign(NT, 0);
            e.pos_stage.assign(NT, 0);
            e.idx_stage.assign(static_cast<std::size_t>(NT) * kvH, 0);
            e.thr_stage.assign(NT, 0.0f);
            e.sampled_stage.assign(NT, 0);
            e.valid = true;
        }

        QbvEntry& e = *entry;
        e.lru = ++pool.clock;
        for (int s = 0; s < N; s++)
            for (int t = 0; t < T; t++)
            {
                const int j = s * T + t;
                const std::int64_t pos = static_cast<std::int64_t>(positions[s]) + t;
                e.tok_stage[j] = token_ids[j];
                e.pos_stage[j] = rope_positions != nullptr ? rope_positions[j] : static_cast<std::int32_t>(pos);
                e.thr_stage[j] = static_cast<float>(pos);
                for (int h = 0; h < kvH; h++)
                    e.idx_stage[static_cast<std::size_t>(j) * kvH + h] = static_cast<std::int64_t>(h) * e.cache_rows[s] + pos;
            }
        host_read_barrier();
        decode_input_set_async(e.token_in, e.tok_stage.data(), e.tok_stage.size() * sizeof(std::int32_t));
        decode_input_set_async(e.pos_in, e.pos_stage.data(), e.pos_stage.size() * sizeof(std::int32_t));
        decode_input_set_async(e.idx_in, e.idx_stage.data(), e.idx_stage.size() * sizeof(std::int64_t));
        decode_input_set_async(e.thr_in, e.thr_stage.data(), e.thr_stage.size() * sizeof(float));

        compute_started = true;
        if (tsg::graph_compute_profiled(g_backend, e.graph, kQbvKernel) != GGML_STATUS_SUCCESS)
        {
            set_last_error("Qwen3.5 batched verify: graph execution failed.");
            e.release();
            return -1;
        }
        finalize_compute_with_download(e.sampled_out, e.sampled_stage.data(), e.sampled_stage.size() * sizeof(std::int32_t));
        if (capture_count > 0)
            finalize_compute_with_download(e.capture_out, capture_data,
                static_cast<std::size_t>(NT) * H * capture_count * sizeof(float));
        host_read_barrier();
        std::memcpy(sampled_data, e.sampled_stage.data(), static_cast<std::size_t>(NT) * sizeof(std::int32_t));
        pool.pending = &e;
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return compute_started ? -1 : 0; }
    catch (...) { set_last_error("Unknown error in Qwen3.5 batched verify."); return compute_started ? -1 : 0; }
}

// Advance each slot of the last verify by its accepted prefix: accepted[s] drafts
// were kept, so accepted[s] + 1 tokens enter the recurrent state. Must follow the
// verify it commits with nothing touching those holders in between. Returns 1, 0
// when there is no pending verify (nothing written), -1 if a commit graph failed
// after some slots may have been advanced.
TSG_EXPORT int TSGgml_Qwen35BatchedVerifyCommit(int n_seqs, const std::int32_t* accepted)
{
    bool started = false;
    try
    {
        std::lock_guard<std::mutex> lock(qbv_mutex());
        QbvPool& pool = qbv_pool();
        QbvEntry* e = pool.pending;
        pool.pending = nullptr;
        if (e == nullptr || !e->valid || accepted == nullptr || n_seqs != e->n_seqs)
        {
            set_last_error("Qwen3.5 batched verify commit: no matching pending verify.");
            return 0;
        }
        for (int s = 0; s < n_seqs; s++)
            if (accepted[s] < 0 || accepted[s] >= e->n_tok)
            {
                set_last_error("Qwen3.5 batched verify commit: accepted count out of range.");
                return 0;
            }
        ggml_cgraph* graph = qbv_commit_graph(*e);
        if (graph == nullptr)
        {
            set_last_error("Qwen3.5 batched verify commit: could not build the commit graph.");
            return 0;
        }
        for (int s = 0; s < n_seqs; s++)
        {
            const int keep = accepted[s] + 1;
            for (int t = 0; t < e->n_tok; t++)
                e->keep_stage[static_cast<std::size_t>(s) * e->n_tok + t] = t < keep ? 1.0f : 0.0f;
            for (int j = 0; j < e->convDim; j++)
                e->window_stage[static_cast<std::size_t>(s) * e->convDim + j] = keep + j;
        }
        host_read_barrier();
        decode_input_set_async(e->commit.keep_in, e->keep_stage.data(), e->keep_stage.size() * sizeof(float));
        decode_input_set_async(e->commit.window_in, e->window_stage.data(), e->window_stage.size() * sizeof(std::int32_t));
        started = true;
        if (tsg::graph_compute_profiled(g_backend, graph, kQbvCommitKernel) != GGML_STATUS_SUCCESS)
        {
            set_last_error("Qwen3.5 batched verify commit: graph execution failed.");
            return -1;
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return started ? -1 : 0; }
    catch (...) { set_last_error("Unknown error in Qwen3.5 batched verify commit."); return started ? -1 : 0; }
}

// Drop every cached verify graph (model teardown, cache resets).
TSG_EXPORT void TSGgml_Qwen35BatchedVerifyReset()
{
    std::lock_guard<std::mutex> lock(qbv_mutex());
    for (auto& pool : g_qbv_pools)
    {
        for (auto& e : pool.entries)
            e.release();
        pool.pending = nullptr;
    }
}
