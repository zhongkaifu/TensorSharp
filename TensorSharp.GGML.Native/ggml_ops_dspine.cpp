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
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

using namespace tsg;

// ============================================================================
// DSpine drafter (Tencent DSpine, https://github.com/Tencent/DSpine, Apache-2.0), one graph per draft step.
//
// A DSpine draft is a DFlash block draft with two changes (see eng/dspine-reference.py for the math, written from
// the reference's dspine_mlx / sglang_dspine sources):
//
//   * block-causal attention: block row i sees the visible context and block rows 0..i (DFlash: the whole block);
//   * a code bridge after every layer. With raw the layer's output rows 1..b-1 (row 0, the anchor, passes through):
//         rms     = sqrt(max(mean(raw^2), FLT_MIN))
//         codes   = sqrt(C) * p / (|p| + eps),   p = (raw / (rms + eps)) @ read_l^T
//         q, s, v = codes @ [gate; predecessor_gate; prev]^T
//         out_t   = raw_t + rms_t * (sigmoid(q_t + s_{t-1} + bias) * v_{t-1}) @ write_l^T
//     The anchor row's (s, v) are those of sqrt(C) * codebook[anchor], i.e. the table row TSV[anchor], so the graph
//     reads the table instead of the codebook.
//
// The context (fc -> hidden_norm -> K/V into the ring) is DFlash's inject, unchanged: TSGgml_DFlashInject serves
// both drafters. The proposal is the reference's greedy "exact sender" refinement of the last write, returned as
// DFlash2's selector lattice so the host walks it the same way:
//
//     cand    = top-K of head @ final_rows[1..b-1]                                        [K, g]   g = b - 1
//     s0[c]   = base[0][c] / rms_1 + m(anchor) . TG[cand[0][c]]                           [K]
//     sc[e][p][c] = base[e][c] / rms_e + m_e(cand[e-1][p]) . TG[cand[e][c]]                [K(c), K(p), g-1]
//     base[e][c] = (raw_e * final_norm_w) . head[cand[e][c]],  m_e(t) = sigmoid(q_e + TSV[t][:R] + bias) * TSV[t][R:]
//
// using the last layer's pre-write raw / rms / q. The tables TG [V, R] = (head * final_norm_w) @ write_last and
// TSV [V, 2R] = (sqrt(C) * codebook) @ [predecessor_gate; prev]^T are built once on the device
// (TSGgml_DSpineTablesCreate). Dividing by rms keeps every row's argmax and makes the lattice's softmax the
// reference's confidence (1 / sum exp((s - s_best) / rms)).
//
// Activations are F32 throughout; the reference rounds them to BF16 at fixed points (stated in the port's report).
// ============================================================================

extern "C" {
// One weight as the graph binds it: host bytes (also the device-cache key), ggml dims and type.
struct TsgDsWeight
{
    const void* data;
    std::int64_t ne0;
    std::int64_t ne1;
    std::int64_t bytes;
    std::int32_t type;
    std::int32_t reserved;
};

struct TsgDsParams
{
    std::int32_t num_layers, hidden, num_heads, num_kv_heads, head_dim, vocab;
    std::int32_t code_size, rank, top_k, ring_rows, ring_dtype, sliding_window;
    float eps, bridge_eps, rope_base, rope_freq_scale, kq_scale;
    std::int32_t reserved[3];
};

// Test probes: every stage of one draft, row-major, or null. raw/out [L][b][H], codes [L][b-1][C] (rows 1..b-1),
// final_rows [b][H], base [b-1][K] in the order of the returned candidates.
struct TsgDsProbe
{
    float* raw;
    float* codes;
    float* out;
    float* final_rows;
    float* base;
};
}

namespace
{
    // Weight slots of the TsgDsWeight array: globals, then kDsPerLayer per draft layer.
    enum : int
    {
        kDsTokEmbd = 0, kDsLmHead = 1, kDsOutNorm = 2, kDsCodebook = 3, kDsBridge = 4, kDsGateBias = 5,
        kDsGlobals = 8,
        kDsAttnNorm = 0, kDsQ, kDsK, kDsV, kDsQNorm, kDsKNorm, kDsO, kDsFfnNorm, kDsGate, kDsUp, kDsDown,
        kDsRead, kDsWrite,
        kDsPerLayer = 16,
    };

    const TsgDsWeight& ds_layer(const TsgDsWeight* w, int l, int slot) { return w[kDsGlobals + l * kDsPerLayer + slot]; }

    bool ds_fail(const char* message)
    {
        set_last_error(message);
        return false;
    }

    bool ds_shape(const TsgDsWeight& w, std::int64_t ne0, std::int64_t ne1, const char* what)
    {
        if (w.data == nullptr || w.ne0 != ne0 || w.ne1 != ne1 || w.bytes <= 0)
        {
            char buf[256];
            std::snprintf(buf, sizeof(buf), "DSpine: %s is [%lld, %lld], expected [%lld, %lld].", what,
                static_cast<long long>(w.ne0), static_cast<long long>(w.ne1),
                static_cast<long long>(ne0), static_cast<long long>(ne1));
            set_last_error(buf);
            return false;
        }
        return true;
    }

    bool ds_f32(const TsgDsWeight& w, std::int64_t n, const char* what)
    {
        if (w.type != GGML_TYPE_F32 || w.bytes != n * static_cast<std::int64_t>(sizeof(float)))
        {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "DSpine: %s must be %lld F32 values.", what, static_cast<long long>(n));
            set_last_error(buf);
            return false;
        }
        return ds_shape(w, n, 1, what);
    }

    // Every shape the graph assumes, checked once per call (cheap next to the graph).
    bool ds_validate(const TsgDsParams* p, const TsgDsWeight* w, int n_weights)
    {
        if (p == nullptr || w == nullptr) return ds_fail("DSpine: null parameters.");
        const int L = p->num_layers, H = p->hidden, C = p->code_size, R = p->rank, V = p->vocab;
        if (L <= 0 || H <= 0 || C <= 0 || R <= 0 || V <= 0 || p->num_heads <= 0 || p->num_kv_heads <= 0
            || p->num_heads % p->num_kv_heads != 0 || p->head_dim <= 0 || (p->head_dim & 1) != 0)
            return ds_fail("DSpine: invalid geometry.");
        if (p->top_k < 1 || p->top_k > V) return ds_fail("DSpine: top_k must be in [1, vocab].");
        if (n_weights != kDsGlobals + L * kDsPerLayer) return ds_fail("DSpine: weight table has the wrong length.");
        if (!ds_shape(w[kDsTokEmbd], H, V, "token_embd") || !ds_shape(w[kDsLmHead], H, V, "the LM head")
            || !ds_f32(w[kDsOutNorm], H, "output_norm") || !ds_shape(w[kDsBridge], C, 3 * R, "dspine.bridge")
            || !ds_f32(w[kDsGateBias], R, "dspine.gate.bias"))
            return false;
        const int qd = p->num_heads * p->head_dim, kvd = p->num_kv_heads * p->head_dim;
        for (int l = 0; l < L; ++l)
        {
            const std::int64_t ff = ds_layer(w, l, kDsGate).ne1;
            if (!ds_f32(ds_layer(w, l, kDsAttnNorm), H, "attn_norm") || !ds_f32(ds_layer(w, l, kDsFfnNorm), H, "ffn_norm")
                || !ds_f32(ds_layer(w, l, kDsQNorm), p->head_dim, "attn_q_norm")
                || !ds_f32(ds_layer(w, l, kDsKNorm), p->head_dim, "attn_k_norm")
                || !ds_shape(ds_layer(w, l, kDsQ), H, qd, "attn_q") || !ds_shape(ds_layer(w, l, kDsK), H, kvd, "attn_k")
                || !ds_shape(ds_layer(w, l, kDsV), H, kvd, "attn_v") || !ds_shape(ds_layer(w, l, kDsO), qd, H, "attn_output")
                || !ds_shape(ds_layer(w, l, kDsGate), H, ff, "ffn_gate") || !ds_shape(ds_layer(w, l, kDsUp), H, ff, "ffn_up")
                || !ds_shape(ds_layer(w, l, kDsDown), ff, H, "ffn_down")
                || !ds_shape(ds_layer(w, l, kDsRead), H, C, "dspine_read") || !ds_shape(ds_layer(w, l, kDsWrite), R, H, "dspine_write"))
                return false;
        }
        return true;
    }

    // Binds weights to cached device copies (or host-pointer wraps), queueing uploads otherwise.
    struct DsBinder
    {
        struct Upload { ggml_tensor* t; const void* data; std::size_t bytes; };
        std::vector<Upload> uploads;
        std::vector<BufferHandle> ephemeral;
        ggml_backend_dev_t dev = nullptr;

        void bind(ggml_tensor* t, const void* data, std::size_t bytes, bool cacheable,
                  enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS)
        {
            if (t == nullptr || data == nullptr) return;
            void* d = const_cast<void*>(data);
            if (cacheable && bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, t, d, bytes, needs_upload, usage))
                {
                    if (needs_upload) uploads.push_back({ t, data, bytes });
                    return;
                }
            }
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, d, bytes, cacheable, buf))
                {
                    if (!cacheable) ephemeral.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, t, d) == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            uploads.push_back({ t, data, bytes });
        }

        ggml_tensor* weight(ggml_context* ctx, const TsgDsWeight& w)
        {
            ggml_tensor* t = w.ne1 == 1 && w.type == GGML_TYPE_F32
                ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, w.ne0)
                : ggml_new_tensor_2d(ctx, static_cast<ggml_type>(w.type), w.ne0, w.ne1);
            pending.push_back({ t, w.data, static_cast<std::size_t>(w.bytes) });
            return t;
        }

        // Binds every tensor made by weight() (after the graph is built, before allocation).
        void bind_pending(bool cacheable)
        {
            for (auto& u : pending) bind(u.t, u.data, u.bytes, cacheable);
            pending.clear();
        }

        void flush()
        {
            for (auto& u : uploads)
                ggml_backend_tensor_set(u.t, resolve_upload_source(u.data), 0, u.bytes);
        }

        std::vector<Upload> pending;
    };

    // Graph storage planned by a private gallocr (inputs and outputs keep their own), as the DFlash draft does.
    bool ds_alloc(ggml_context* ctx, ggml_cgraph* graph, ggml_gallocr_t& galloc)
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

    // ------------------------------------------------------------------------------------------------------------
    // The refinement tables
    // ------------------------------------------------------------------------------------------------------------
    struct DsTables
    {
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buffer = nullptr;
        ggml_backend_t backend = nullptr;
        ggml_tensor* tg = nullptr;    // [R, V]
        ggml_tensor* tsv = nullptr;   // [2R, V]
        int vocab = 0, rank = 0;

        ~DsTables()
        {
            if (buffer != nullptr) ggml_backend_buffer_free(buffer);
            if (ctx != nullptr) ggml_free(ctx);
        }
    };

    // ------------------------------------------------------------------------------------------------------------
    // Persistent draft graphs, keyed by (weights, ring, block width, tables)
    // ------------------------------------------------------------------------------------------------------------
    struct DsCache
    {
        bool valid = false;
        ggml_context* ctx = nullptr;
        ggml_gallocr_t galloc = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_tensor* ids = nullptr, *pos = nullptr, *mask = nullptr, *slots = nullptr;
        ggml_tensor* cand = nullptr, *s0 = nullptr, *scores = nullptr;
        const void* sig = nullptr;
        const void* ring = nullptr;
        const DsTables* tables = nullptr;
        int b = 0;
        std::uint64_t used = 0;

        void reset()
        {
            if (galloc != nullptr) { ggml_gallocr_free(galloc); galloc = nullptr; }
            if (ctx != nullptr) { ggml_free(ctx); ctx = nullptr; }
            graph = nullptr;
            ids = pos = mask = slots = cand = s0 = scores = nullptr;
            sig = ring = nullptr;
            tables = nullptr;
            b = 0;
            valid = false;
        }
    };

    // One graph per block width (2..16) per ring; rings are per request (ModelBase.DFlash.Rings.cs).
    constexpr int kDsMaxCaches = 48;
    struct DsCachePool
    {
        DsCache entries[kDsMaxCaches];
        std::uint64_t clock = 0;

        DsCache* find(const void* sig, const void* ring, int b, const DsTables* tables)
        {
            for (auto& e : entries)
                if (e.valid && e.sig == sig && e.ring == ring && e.b == b && e.tables == tables)
                { e.used = ++clock; return &e; }
            return nullptr;
        }

        DsCache& claim()
        {
            DsCache* lru = &entries[0];
            for (auto& e : entries)
            {
                if (!e.valid) { lru = &e; break; }
                if (e.used < lru->used) lru = &e;
            }
            lru->reset();
            lru->used = ++clock;
            return *lru;
        }

        void drop_tables(const DsTables* tables)
        {
            for (auto& e : entries)
                if (e.valid && e.tables == tables) e.reset();
        }

        void reset_all() { for (auto& e : entries) e.reset(); }
    };

    DsCachePool g_ds_pools[tsg::TSG_MAX_DEVICES];
    inline DsCachePool& ds_pool() { return g_ds_pools[tsg::g_active_rank]; }

    struct DsGraph
    {
        ggml_tensor* ids = nullptr, *pos = nullptr, *mask = nullptr, *slots = nullptr;
        ggml_tensor* cand = nullptr, *s0 = nullptr, *scores = nullptr;
        std::vector<ggml_tensor*> probe_raw, probe_codes, probe_out;
        ggml_tensor* probe_final = nullptr, *probe_base = nullptr;
        ggml_cgraph* graph = nullptr;
    };

    // TS_DSPINE_BF16_POINTS=1 (diagnostic): round the bridge's activations to BF16 where the reference rounds them
    // (sglang_dspine/kernels/dspine.py: raw, the normalized rows, the read projection, the codes, the packed
    // projection, the message, the write delta, the written row, and the next norm's two steps), to measure what F32
    // activations change. Off: those tensors stay F32 and the graph is unchanged.
    bool ds_bf16_points()
    {
        static const bool on = [] {
            const char* e = std::getenv("TS_DSPINE_BF16_POINTS");
            return e != nullptr && e[0] == '1';
        }();
        return on;
    }

    ggml_tensor* ds_bf16(ggml_context* ctx, ggml_tensor* x)
    {
        return ds_bf16_points() ? ggml_cast(ctx, ggml_cast(ctx, ggml_cont(ctx, x), GGML_TYPE_BF16), GGML_TYPE_F32) : x;
    }

    // The next layer's (or the final) norm of a written row: bf16(bf16(rms_norm(x)) * w) at the BF16 points.
    ggml_tensor* ds_next_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, float eps, bool written)
    {
        if (!written || !ds_bf16_points())
            return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
        return ds_bf16(ctx, ggml_mul(ctx, ds_bf16(ctx, ggml_rms_norm(ctx, x, eps)), w));
    }

    // The draft graph for a b-row block. probe: also mark every stage as an output.
    bool ds_build(ggml_context* ctx, const TsgDsParams* p, const TsgDsWeight* w, const DsTables* tables,
                  void** ring_k, void** ring_v, int b, bool probe, DsBinder& binder, DsGraph& g)
    {
        const int L = p->num_layers, H = p->hidden, C = p->code_size, R = p->rank, K = p->top_k;
        const int NH = p->num_heads, NKV = p->num_kv_heads, HD = p->head_dim;
        const int ring_rows = p->ring_rows;
        const int gm = b - 1;
        const float sqrt_c = std::sqrt(static_cast<float>(C));
        const std::size_t f32 = sizeof(float);

        g.ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, b);
        g.pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, b);
        g.mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, ring_rows, b, 1, 1);
        g.slots = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, b);
        ggml_set_input(g.ids); ggml_set_input(g.pos); ggml_set_input(g.mask); ggml_set_input(g.slots);

        ggml_tensor* tok = binder.weight(ctx, w[kDsTokEmbd]);
        ggml_tensor* head = binder.weight(ctx, w[kDsLmHead]);
        ggml_tensor* out_norm = binder.weight(ctx, w[kDsOutNorm]);
        ggml_tensor* bridge = binder.weight(ctx, w[kDsBridge]);
        ggml_tensor* gate_bias = binder.weight(ctx, w[kDsGateBias]);

        // The anchor's sender halves (s, v): TSV[anchor].
        ggml_tensor* anchor_id = ggml_view_1d(ctx, g.ids, 1, 0);
        ggml_tensor* tsv_anchor = ggml_get_rows(ctx, tables->tsv, anchor_id);                     // [2R, 1]

        ggml_tensor* inpL = ggml_get_rows(ctx, tok, g.ids);                                         // [H, b]
        ggml_tensor* last_r1 = nullptr, *last_rms = nullptr, *last_packed = nullptr;
        for (int l = 0; l < L; ++l)
        {
            ggml_tensor* attn_norm = binder.weight(ctx, ds_layer(w, l, kDsAttnNorm));
            ggml_tensor* q_w = binder.weight(ctx, ds_layer(w, l, kDsQ));
            ggml_tensor* k_w = binder.weight(ctx, ds_layer(w, l, kDsK));
            ggml_tensor* v_w = binder.weight(ctx, ds_layer(w, l, kDsV));
            ggml_tensor* q_norm = binder.weight(ctx, ds_layer(w, l, kDsQNorm));
            ggml_tensor* k_norm = binder.weight(ctx, ds_layer(w, l, kDsKNorm));
            ggml_tensor* o_w = binder.weight(ctx, ds_layer(w, l, kDsO));
            ggml_tensor* ffn_norm = binder.weight(ctx, ds_layer(w, l, kDsFfnNorm));
            ggml_tensor* gate_w = binder.weight(ctx, ds_layer(w, l, kDsGate));
            ggml_tensor* up_w = binder.weight(ctx, ds_layer(w, l, kDsUp));
            ggml_tensor* down_w = binder.weight(ctx, ds_layer(w, l, kDsDown));
            ggml_tensor* read_w = binder.weight(ctx, ds_layer(w, l, kDsRead));
            ggml_tensor* write_w = binder.weight(ctx, ds_layer(w, l, kDsWrite));
            ggml_tensor* rk = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(p->ring_dtype), HD, ring_rows, NKV);
            ggml_tensor* rv = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(p->ring_dtype), HD, ring_rows, NKV);
            const std::size_t ring_bytes = kv_cache_bytes(NKV, ring_rows, HD, p->ring_dtype);
            binder.bind(rk, ring_k[l], ring_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
            binder.bind(rv, ring_v[l], ring_bytes, true, GGML_BACKEND_BUFFER_USAGE_COMPUTE);

            // Layer 0 norms its embedding rows; later layers get the previous bridge's next_norm(out).
            ggml_tensor* h = ds_next_norm(ctx, inpL, attn_norm, p->eps, l > 0);

            ggml_tensor* q3 = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, q_w, h), HD, NH, b);
            ggml_tensor* k3 = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, k_w, h), HD, NKV, b);
            ggml_tensor* v3 = ggml_reshape_3d(ctx, ggml_mul_mat(ctx, v_w, h), HD, NKV, b);
            q3 = ggml_mul(ctx, ggml_rms_norm(ctx, q3, p->eps), q_norm);
            k3 = ggml_mul(ctx, ggml_rms_norm(ctx, k3, p->eps), k_norm);
            q3 = ggml_rope_ext(ctx, q3, g.pos, nullptr, HD, /*NeoX=*/2, 0, p->rope_base, p->rope_freq_scale,
                               0.0f, 1.0f, 0.0f, 0.0f);
            k3 = ggml_rope_ext(ctx, k3, g.pos, nullptr, HD, /*NeoX=*/2, 0, p->rope_base, p->rope_freq_scale,
                               0.0f, 1.0f, 0.0f, 0.0f);

            // The block's K/V at its own ring slots (the ring has a block of headroom past the window).
            ggml_tensor* kcat = ggml_set_rows(ctx, rk, ggml_permute(ctx, k3, 0, 2, 1, 3), g.slots);
            ggml_tensor* vcat = ggml_set_rows(ctx, rv, ggml_permute(ctx, v3, 0, 2, 1, 3), g.slots);
            ggml_tensor* fa = flash_attn_ext_guarded(ctx, "DSpine draft", ggml_permute(ctx, q3, 0, 2, 1, 3), kcat, vcat,
                                                     g.mask, p->kq_scale, 0.0f, 0.0f, nullptr, GGML_PREC_F32);
            ggml_tensor* attn_out = ggml_mul_mat(ctx, o_w, ggml_reshape_2d(ctx, fa, NH * HD, b));
            ggml_tensor* ffn_inp = ggml_add(ctx, attn_out, inpL);
            ggml_tensor* fh = ggml_mul(ctx, ggml_rms_norm(ctx, ffn_inp, p->eps), ffn_norm);
            ggml_tensor* act = ggml_mul(ctx, ggml_silu(ctx, ggml_mul_mat(ctx, gate_w, fh)), ggml_mul_mat(ctx, up_w, fh));
            ggml_tensor* raw = ds_bf16(ctx, ggml_add(ctx, ggml_mul_mat(ctx, down_w, act), ffn_inp));    // [H, b]

            // ---- the code bridge, on rows 1..b-1 ----
            ggml_tensor* r1 = ggml_view_2d(ctx, raw, H, gm, raw->nb[1], raw->nb[1]);                    // [H, g]
            ggml_tensor* rms = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_mean(ctx, ggml_sqr(ctx, r1)),
                                                         std::numeric_limits<float>::min(),
                                                         std::numeric_limits<float>::max()));         // [1, g]
            ggml_tensor* nrm = ds_bf16(ctx, ggml_div(ctx, r1, ggml_scale_bias(ctx, rms, 1.0f, p->bridge_eps)));
            ggml_tensor* proj = ds_bf16(ctx, ggml_mul_mat(ctx, read_w, nrm));                          // [C, g]
            ggml_tensor* pn = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, proj)));                  // [1, g]
            ggml_tensor* codes = ds_bf16(ctx, ggml_div(ctx, ggml_scale(ctx, proj, sqrt_c),
                                                       ggml_scale_bias(ctx, pn, 1.0f, p->bridge_eps)));
            ggml_tensor* packed = ds_bf16(ctx, ggml_mul_mat(ctx, bridge, codes));                      // [3R, g]
            ggml_tensor* sv_prev = tsv_anchor;                                                          // [2R, g]
            if (gm > 1)
                sv_prev = ggml_concat(ctx, tsv_anchor,
                                      ggml_view_2d(ctx, packed, 2 * R, gm - 1, packed->nb[1], R * f32), 1);
            ggml_tensor* qv = ggml_view_2d(ctx, packed, R, gm, packed->nb[1], 0);
            ggml_tensor* s_prev = ggml_view_2d(ctx, sv_prev, R, gm, sv_prev->nb[1], 0);
            ggml_tensor* v_prev = ggml_view_2d(ctx, sv_prev, R, gm, sv_prev->nb[1], R * f32);
            ggml_tensor* msg = ds_bf16(ctx, ggml_mul(ctx, ggml_sigmoid(ctx,
                ggml_add(ctx, ggml_add(ctx, qv, s_prev), gate_bias)), v_prev));
            ggml_tensor* delta = ds_bf16(ctx, ggml_mul_mat(ctx, write_w, msg));                        // [H, g]
            ggml_tensor* upd = ds_bf16(ctx, ggml_add(ctx, r1, ggml_mul(ctx, delta, rms)));
            ggml_tensor* out = ggml_concat(ctx, ggml_view_2d(ctx, raw, H, 1, raw->nb[1], 0), upd, 1);   // [H, b]
            if (probe)
            {
                ggml_set_output(raw); ggml_set_output(codes); ggml_set_output(out);
                g.probe_raw.push_back(raw); g.probe_codes.push_back(codes); g.probe_out.push_back(out);
            }
            last_r1 = r1;
            last_rms = rms;
            last_packed = packed;
            inpL = out;
        }

        // ---- the proposal ----
        ggml_tensor* cur = ds_next_norm(ctx, inpL, out_norm, p->eps, true);                              // [H, b]
        ggml_tensor* pred = ggml_view_2d(ctx, cur, H, gm, cur->nb[1], cur->nb[1]);
        ggml_tensor* logits = ggml_mul_mat(ctx, head, pred);                                            // [V, g]
        g.cand = ggml_top_k(ctx, logits, K);                                                            // [K, g]
        if (!backend_supports_op(g.cand))
            return ds_fail("DSpine draft: this backend has no top-k over the vocabulary.");
        ggml_set_output(g.cand);
        ggml_tensor* cand_flat = ggml_reshape_1d(ctx, g.cand, static_cast<std::int64_t>(K) * gm);

        // base[e][c] / rms_e
        ggml_tensor* hr = ggml_reshape_3d(ctx, ggml_get_rows(ctx, head, cand_flat), H, K, gm);         // [H, K, g]
        ggml_tensor* u = ggml_reshape_3d(ctx, ggml_mul(ctx, last_r1, out_norm), H, 1, gm);              // [H, 1, g]
        ggml_tensor* base = ggml_mul_mat(ctx, hr, u);                                                   // [K, 1, g]
        if (probe)
        {
            ggml_set_output(base);
            g.probe_base = base;
            ggml_set_output(cur);
            g.probe_final = cur;
        }
        ggml_tensor* base_s = ggml_div(ctx, base, ggml_reshape_3d(ctx, last_rms, 1, 1, gm));

        ggml_tensor* tgc = ggml_reshape_3d(ctx, ggml_get_rows(ctx, tables->tg, cand_flat), R, K, gm);   // [R, K, g]

        // Row 0's sender is the anchor for every predecessor lane.
        ggml_tensor* tvg_a = ggml_view_2d(ctx, tsv_anchor, R, 1, tsv_anchor->nb[1], 0);
        ggml_tensor* tvv_a = ggml_view_2d(ctx, tsv_anchor, R, 1, tsv_anchor->nb[1], R * f32);
        ggml_tensor* q0 = ggml_view_2d(ctx, last_packed, R, 1, last_packed->nb[1], 0);
        ggml_tensor* m0 = ggml_mul(ctx, ggml_sigmoid(ctx, ggml_add(ctx, ggml_add(ctx, q0, tvg_a), gate_bias)), tvv_a);
        ggml_tensor* tg0 = ggml_view_2d(ctx, tgc, R, K, tgc->nb[1], 0);
        g.s0 = ggml_add(ctx, ggml_mul_mat(ctx, tg0, m0), ggml_view_2d(ctx, base_s, K, 1, base_s->nb[1], 0));   // [K, 1]
        ggml_set_output(g.s0);

        if (gm > 1)
        {
            ggml_tensor* prev_flat = ggml_view_1d(ctx, cand_flat, static_cast<std::int64_t>(K) * (gm - 1), 0);
            ggml_tensor* tsvp = ggml_reshape_3d(ctx, ggml_get_rows(ctx, tables->tsv, prev_flat), 2 * R, K, gm - 1);
            ggml_tensor* tvg = ggml_view_3d(ctx, tsvp, R, K, gm - 1, tsvp->nb[1], tsvp->nb[2], 0);
            ggml_tensor* tvv = ggml_view_3d(ctx, tsvp, R, K, gm - 1, tsvp->nb[1], tsvp->nb[2], R * f32);
            ggml_tensor* q_rest = ggml_view_3d(ctx, last_packed, R, 1, gm - 1, last_packed->nb[1], last_packed->nb[1],
                                               last_packed->nb[1]);
            ggml_tensor* m = ggml_mul(ctx, ggml_sigmoid(ctx, ggml_add(ctx, ggml_add(ctx, tvg, q_rest), gate_bias)), tvv);
            ggml_tensor* tg_rest = ggml_view_3d(ctx, tgc, R, K, gm - 1, tgc->nb[1], tgc->nb[2], tgc->nb[2]);
            ggml_tensor* sc = ggml_mul_mat(ctx, tg_rest, m);                                            // [K(c), K(p), g-1]
            ggml_tensor* base_rest = ggml_view_3d(ctx, base_s, K, 1, gm - 1, base_s->nb[1], base_s->nb[2], base_s->nb[2]);
            g.scores = ggml_add(ctx, sc, base_rest);
            ggml_set_output(g.scores);
        }

        g.graph = ggml_new_graph_custom(ctx, static_cast<std::size_t>(L) * 256 + 1024, false);
        ggml_build_forward_expand(g.graph, g.cand);
        ggml_build_forward_expand(g.graph, g.s0);
        if (g.scores != nullptr) ggml_build_forward_expand(g.graph, g.scores);
        if (probe)
        {
            for (int l = 0; l < L; ++l)
            {
                ggml_build_forward_expand(g.graph, g.probe_raw[l]);
                ggml_build_forward_expand(g.graph, g.probe_codes[l]);
                ggml_build_forward_expand(g.graph, g.probe_out[l]);
            }
            ggml_build_forward_expand(g.graph, g.probe_final);
            ggml_build_forward_expand(g.graph, g.probe_base);
        }
        binder.bind_pending(true);
        return true;
    }

    // [ring_rows, b] mask: ring slot s holds ring_slot_pos[s] (-1 = dead); it is context for block query i when live,
    // before the anchor and inside the window. Block row j sits at slot positions[j] % ring_rows and is visible to
    // query i only when j <= i (the slots a block takes only ever held positions no query can see).
    void ds_mask(std::vector<ggml_fp16_t>& mask, std::vector<std::int64_t>& slots, const int* positions, int b,
                 const int* ring_slot_pos, int ring_rows, int window)
    {
        const ggml_fp16_t hidden = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const ggml_fp16_t vis = static_cast<ggml_fp16_t>(0);
        mask.assign(static_cast<std::size_t>(ring_rows) * b, hidden);
        const int anchor_pos = positions[0];
        for (int i = 0; i < b; i++)
        {
            const int qp = positions[i];
            ggml_fp16_t* row = &mask[static_cast<std::size_t>(i) * ring_rows];
            for (int s = 0; s < ring_rows; s++)
            {
                const int pp = ring_slot_pos[s];
                if (pp < 0 || pp >= anchor_pos) continue;
                if (window > 0 && qp - pp >= window) continue;
                row[s] = vis;
            }
            for (int j = 0; j < b; j++)
                row[positions[j] % ring_rows] = j <= i ? vis : hidden;
        }
        slots.assign(b, 0);
        for (int j = 0; j < b; j++)
            slots[j] = positions[j] % ring_rows;
    }

    bool ds_download(ggml_tensor* t, void* dst, std::size_t bytes)
    {
        if (t == nullptr || dst == nullptr || bytes == 0) return true;
        if (ggml_nbytes(t) < bytes) return ds_fail("DSpine: probe buffer larger than its stage.");
        ggml_backend_tensor_get(t, dst, 0, bytes);
        return true;
    }

    // [ne0, ne1] tensor (contiguous) -> row-major [ne1][ne0] floats: the same bytes.
    bool ds_probe_out(const TsgDsProbe* probe, const DsGraph& g, const TsgDsParams* p, int b)
    {
        const int L = p->num_layers, H = p->hidden, C = p->code_size, K = p->top_k, gm = b - 1;
        const std::size_t f = sizeof(float);
        for (int l = 0; l < L; ++l)
        {
            if (probe->raw != nullptr && !ds_download(g.probe_raw[l], probe->raw + static_cast<std::size_t>(l) * b * H, f * b * H)) return false;
            if (probe->codes != nullptr && !ds_download(g.probe_codes[l], probe->codes + static_cast<std::size_t>(l) * gm * C, f * gm * C)) return false;
            if (probe->out != nullptr && !ds_download(g.probe_out[l], probe->out + static_cast<std::size_t>(l) * b * H, f * b * H)) return false;
        }
        return ds_download(g.probe_final, probe->final_rows, f * b * H)
            && ds_download(g.probe_base, probe->base, f * gm * K);
    }
}

// ============================================================================
// Exports
// ============================================================================

// Builds TG [V, R] and TSV [V, 2R] on the active backend in table_type (F32, F16 or BF16). Reads the LM head,
// output_norm, the last layer's dspine_write, the packed bridge weight and the codebook from `w`. The codebook is
// bound without a device-cache entry: only the tables need it. Returns null with the last error set on failure.
TSG_EXPORT void* TSGgml_DSpineTablesCreate(const TsgDsParams* p, const TsgDsWeight* w, int n_weights, int table_type)
{
    try
    {
        if (!ensure_backend()) return nullptr;
        if (!ds_validate(p, w, n_weights)) return nullptr;
        if (table_type != GGML_TYPE_F32 && table_type != GGML_TYPE_F16 && table_type != GGML_TYPE_BF16)
        { set_last_error("DSpine tables: table type must be F32, F16 or BF16."); return nullptr; }
        const int C = p->code_size, R = p->rank, V = p->vocab, L = p->num_layers;
        if (!ds_shape(w[kDsCodebook], C, V, "dspine.codebook")) return nullptr;

        auto tables = std::make_unique<DsTables>();
        tables->backend = g_backend;
        tables->vocab = V;
        tables->rank = R;
        ggml_init_params tp = { 4 * ggml_tensor_overhead(), nullptr, /*no_alloc=*/true };
        tables->ctx = ggml_init(tp);
        if (tables->ctx == nullptr) { set_last_error("DSpine tables: context init failed."); return nullptr; }
        tables->tg = ggml_new_tensor_2d(tables->ctx, static_cast<ggml_type>(table_type), R, V);
        tables->tsv = ggml_new_tensor_2d(tables->ctx, static_cast<ggml_type>(table_type), 2 * R, V);
        tables->buffer = ggml_backend_alloc_ctx_tensors(tables->ctx, g_backend);
        if (tables->buffer == nullptr) { set_last_error("DSpine tables: device allocation failed."); return nullptr; }

        constexpr int kChunk = 4096;
        const int chunk = std::min(kChunk, V);
        ggml_init_params ip = { 16 * 1024 * 1024, nullptr, /*no_alloc=*/true };
        ggml_context* ctx = ggml_init(ip);
        if (ctx == nullptr) { set_last_error("DSpine tables: graph context init failed."); return nullptr; }
        struct CtxGuard { ggml_context* c; ~CtxGuard() { ggml_free(c); } } ctx_guard{ ctx };

        DsBinder binder;
        binder.dev = ggml_backend_get_device(g_backend);
        ggml_tensor* ids32 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, chunk);
        ggml_tensor* ids64 = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, chunk);
        ggml_set_input(ids32); ggml_set_input(ids64);
        ggml_tensor* head = binder.weight(ctx, w[kDsLmHead]);
        ggml_tensor* norm = binder.weight(ctx, w[kDsOutNorm]);
        ggml_tensor* write = binder.weight(ctx, ds_layer(w, L - 1, kDsWrite));
        ggml_tensor* bridge = binder.weight(ctx, w[kDsBridge]);
        binder.bind_pending(true);
        ggml_tensor* codebook = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(w[kDsCodebook].type), C, V);
        binder.bind(codebook, w[kDsCodebook].data, static_cast<std::size_t>(w[kDsCodebook].bytes), /*cacheable=*/false);

        // TG chunk = (head rows * norm) @ write: write is [R, H] (ne0 = R), so contract over its transpose.
        ggml_tensor* write_t = ggml_cont(ctx, ggml_transpose(ctx, ggml_cast(ctx, write, GGML_TYPE_F32)));     // [H, R]
        ggml_tensor* u = ggml_mul(ctx, ggml_get_rows(ctx, head, ids32), norm);                               // [H, chunk]
        ggml_tensor* tg_chunk = ggml_mul_mat(ctx, write_t, u);                                               // [R, chunk]
        // TSV chunk = (sqrt(C) * codebook rows) @ [predecessor_gate; prev]^T.
        ggml_tensor* pv = ggml_view_2d(ctx, bridge, C, 2 * R, bridge->nb[1], static_cast<std::size_t>(R) * bridge->nb[1]);
        ggml_tensor* codes = ggml_scale(ctx, ggml_get_rows(ctx, codebook, ids32), std::sqrt(static_cast<float>(C)));
        ggml_tensor* tsv_chunk = ggml_mul_mat(ctx, pv, codes);                                               // [2R, chunk]
        ggml_tensor* w1 = ggml_set_rows(ctx, tables->tg, tg_chunk, ids64);
        ggml_tensor* w2 = ggml_set_rows(ctx, tables->tsv, tsv_chunk, ids64);
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, w1);
        ggml_build_forward_expand(graph, w2);

        ggml_gallocr_t galloc = nullptr;
        if (!ds_alloc(ctx, graph, galloc)) { set_last_error("DSpine tables: graph allocation failed."); return nullptr; }
        struct AllocGuard { ggml_gallocr_t a; ~AllocGuard() { ggml_gallocr_free(a); } } alloc_guard{ galloc };
        host_read_barrier();
        binder.flush();

        std::vector<std::int32_t> i32(chunk);
        std::vector<std::int64_t> i64(chunk);
        for (int start = 0; start < V; start += chunk)
        {
            // The last chunk repeats its last row: the same values written twice into the same table row.
            for (int i = 0; i < chunk; ++i)
            {
                const int row = std::min(start + i, V - 1);
                i32[i] = row;
                i64[i] = row;
            }
            ggml_backend_tensor_set(ids32, i32.data(), 0, i32.size() * sizeof(std::int32_t));
            ggml_backend_tensor_set(ids64, i64.data(), 0, i64.size() * sizeof(std::int64_t));
            if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
            { set_last_error("DSpine tables: graph execution failed."); return nullptr; }
            tsg::sync_backend(g_backend);
        }
        clear_last_error();
        return tables.release();
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return nullptr; }
    catch (...) { set_last_error("Unknown error building the DSpine tables."); return nullptr; }
}

TSG_EXPORT void TSGgml_DSpineTablesFree(void* handle)
{
    if (handle == nullptr) return;
    auto* tables = static_cast<DsTables*>(handle);
    for (auto& pool : g_ds_pools) pool.drop_tables(tables);
    delete tables;
}

// Rows [row0, row0 + n) of TG (which = 0, [n][R]) or TSV (which = 1, [n][2R]) as F32, for tests.
TSG_EXPORT int TSGgml_DSpineTablesRead(void* handle, int which, int row0, int n, float* out)
{
    try
    {
        auto* tables = static_cast<DsTables*>(handle);
        if (tables == nullptr || out == nullptr || row0 < 0 || n <= 0 || row0 + n > tables->vocab || (which != 0 && which != 1))
        { set_last_error("DSpine tables read: invalid arguments."); return 0; }
        ggml_tensor* t = which == 0 ? tables->tg : tables->tsv;
        const std::int64_t cols = t->ne[0];
        std::vector<std::uint8_t> raw(static_cast<std::size_t>(n) * t->nb[1]);
        ggml_backend_tensor_get(t, raw.data(), static_cast<std::size_t>(row0) * t->nb[1], raw.size());
        const auto* traits = ggml_get_type_traits(t->type);
        if (t->type == GGML_TYPE_F32)
            std::memcpy(out, raw.data(), raw.size());
        else
            traits->to_float(raw.data(), out, static_cast<std::int64_t>(n) * cols);
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error reading the DSpine tables."); return 0; }
}

TSG_EXPORT void TSGgml_DSpineResetCaches()
{
    for (auto& pool : g_ds_pools) pool.reset_all();
}

// One DSpine draft of a b-row block ([anchor, MASK x (b-1)] at positions[0..b-1]) over the ring the DFlash inject
// filled. Returns the lattice the host walks (ModelBase.DFlashWalkLattice's layout): scores_out = s0 [K] then
// [g-1][K(pred)][K(cand)], cand_out [g][K]. probe (tests): every stage, from a graph built for the call.
TSG_EXPORT int TSGgml_DSpineDraftBlock(
    const TsgDsParams* p, const TsgDsWeight* w, int n_weights, void* tables_handle,
    void** ring_k, void** ring_v,
    const int* block_ids, int block_len, const int* positions, const int* ring_slot_pos,
    float* scores_out, int* cand_out, const TsgDsProbe* probe)
{
    try
    {
        if (!ensure_backend()) return 0;
        if (!ds_validate(p, w, n_weights)) return 0;
        auto* tables = static_cast<const DsTables*>(tables_handle);
        const int b = block_len;
        if (tables == nullptr || tables->backend != g_backend || tables->vocab != p->vocab || tables->rank != p->rank)
        { set_last_error("DSpine draft: the tables were built for another backend or model."); return 0; }
        if (b < 2 || block_ids == nullptr || positions == nullptr || ring_slot_pos == nullptr || ring_k == nullptr
            || ring_v == nullptr || scores_out == nullptr || cand_out == nullptr || p->ring_rows <= 0)
        { set_last_error("DSpine draft: invalid arguments."); return 0; }
        const int K = p->top_k, gm = b - 1;
        const int ring_rows = p->ring_rows, window = p->sliding_window;
        // The block goes into the ring at its own slots, which needs a block of headroom past the window
        // (DFlashConfig.RingRows: window + block + 1, rounded up).
        if (window <= 0 || b >= ring_rows || ring_rows - (b - 1) < window)
        { set_last_error("DSpine draft: the ring has no block of headroom past the sliding window."); return 0; }

        std::vector<ggml_fp16_t> mask;
        std::vector<std::int64_t> slots;
        ds_mask(mask, slots, positions, b, ring_slot_pos, ring_rows, window);

        auto set_inputs = [&](const DsGraph& g, bool async) {
            auto set = [&](ggml_tensor* t, const void* d, std::size_t n) {
                if (async) decode_input_set_async(t, d, n); else ggml_backend_tensor_set(t, d, 0, n);
            };
            set(g.ids, block_ids, static_cast<std::size_t>(b) * sizeof(std::int32_t));
            set(g.pos, positions, static_cast<std::size_t>(b) * sizeof(std::int32_t));
            set(g.mask, mask.data(), mask.size() * sizeof(ggml_fp16_t));
            set(g.slots, slots.data(), slots.size() * sizeof(std::int64_t));
        };
        auto read_lattice = [&](const DsGraph& g) {
            // Metal computes asynchronously and a shared-buffer get is a raw memcpy: drain first.
            tsg::sync_backend(g_backend);
            ggml_backend_tensor_get(g.cand, cand_out, 0, static_cast<std::size_t>(K) * gm * sizeof(std::int32_t));
            ggml_backend_tensor_get(g.s0, scores_out, 0, static_cast<std::size_t>(K) * sizeof(float));
            if (g.scores != nullptr)
                ggml_backend_tensor_get(g.scores, scores_out + K, 0,
                    static_cast<std::size_t>(K) * K * (gm - 1) * sizeof(float));
        };

        const void* sig = ds_layer(w, 0, kDsAttnNorm).data;
        if (probe == nullptr)
        {
            DsCache* dc = ds_pool().find(sig, ring_k[0], b, tables);
            if (dc != nullptr)
            {
                DsGraph g;
                g.ids = dc->ids; g.pos = dc->pos; g.mask = dc->mask; g.slots = dc->slots;
                g.cand = dc->cand; g.s0 = dc->s0; g.scores = dc->scores;
                host_read_barrier();
                set_inputs(g, true);
                if (tsg::graph_compute_profiled(g_backend, dc->graph, "dspine draft") != GGML_STATUS_SUCCESS)
                {
                    set_last_error("DSpine draft: cached graph execution failed.");
                    dc->reset();
                    return 0;
                }
                read_lattice(g);
                host_read_barrier();
                clear_last_error();
                return 1;
            }
        }

        ggml_init_params ip = { 32 * 1024 * 1024, nullptr, /*no_alloc=*/true };
        ggml_context* ctx = ggml_init(ip);
        if (ctx == nullptr) { set_last_error("DSpine draft: context init failed."); return 0; }
        DsBinder binder;
        binder.dev = ggml_backend_get_device(g_backend);
        DsGraph g;
        if (!ds_build(ctx, p, w, tables, ring_k, ring_v, b, probe != nullptr, binder, g))
        {
            ggml_free(ctx);
            return 0;
        }
        optimize_graph_for_metal(ctx, g.graph);
        ggml_gallocr_t galloc = nullptr;
        if (!ds_alloc(ctx, g.graph, galloc))
        {
            set_last_error("DSpine draft: graph allocation failed.");
            ggml_free(ctx);
            return 0;
        }
        host_read_barrier();
        binder.flush();
        set_inputs(g, false);
        if (tsg::compute_graph(g_backend, g.graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("DSpine draft: graph execution failed.");
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
            return 0;
        }
        read_lattice(g);
        bool ok = true;
        if (probe != nullptr)
            ok = ds_probe_out(probe, g, p, b);
        host_read_barrier();

        if (probe != nullptr || !ok)
        {
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
            if (ok) clear_last_error();
            return ok ? 1 : 0;
        }
        DsCache& slot = ds_pool().claim();
        slot.ctx = ctx; slot.galloc = galloc; slot.graph = g.graph;
        slot.ids = g.ids; slot.pos = g.pos; slot.mask = g.mask; slot.slots = g.slots;
        slot.cand = g.cand; slot.s0 = g.s0; slot.scores = g.scores;
        slot.sig = sig; slot.ring = ring_k[0]; slot.tables = tables; slot.b = b;
        slot.valid = true;
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Unknown error in the DSpine draft."); return 0; }
}
