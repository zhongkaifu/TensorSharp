// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The generic graph: managed code emits a model's whole forward as ggml nodes, one call per
// node, and runs it as ONE graph. This replaces the per-family fused graphs (one hand-written
// C++ file per family and per mode): a family describes its layers once, in managed code,
// against these ops, and every family and placement gets whole-graph execution.
//
// Leaves are bound by role:
//   WEIGHT - read-only host bytes; bound to the cached device copy keyed by the host pointer
//            (uploaded once), wrapped zero-copy where the device can, or uploaded per run.
//   STATE  - read-write host bytes the graph updates in place (KV caches, recurrent states).
//            Bound like a weight but with compute usage; a STATE leaf that had to be uploaded
//            is downloaded after the run, so its writes are never lost.
//   INPUT  - per-run values (token ids, positions, masks, write indices), uploaded each run.
// A graph keeps its own allocator, so a decode graph is replayed with new inputs and no
// rebuild while its structure and its leaves' host pointers stay the same (the caller keys
// its graph cache on both).
//
// Upstream ggml is used unchanged; everything here is TensorSharp-owned.

#include "ggml_ops_internal.h"
#include "ggml_ops_transformer_common.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstring>
#include <vector>

using namespace tsg;

namespace
{
    enum TsgLeafRole : int
    {
        LEAF_WEIGHT = 0,
        LEAF_STATE = 1,
        LEAF_INPUT = 2,
    };

    // Op codes shared with TensorSharp.GGML.GgmlGraphOp (keep in sync).
    enum TsgGraphOp : int
    {
        OP_GET_ROWS = 1,
        OP_ADD = 2,
        OP_SUB = 3,
        OP_MUL = 4,
        OP_DIV = 5,
        OP_SCALE = 6,           // fp: s
        OP_SCALE_BIAS = 7,      // fp: s, b
        OP_RMS_NORM = 8,        // fp: eps
        OP_NORM = 9,            // fp: eps
        OP_L2_NORM = 10,        // fp: eps
        OP_GROUP_NORM = 11,     // ip: groups; fp: eps
        OP_MUL_MAT = 12,        // srcs: weight, x; ip: [acc precision]
        OP_MUL_MAT_ID = 13,     // srcs: stacked weights, x, ids
        OP_SILU = 14,
        OP_GELU = 15,
        OP_GELU_QUICK = 16,
        OP_SIGMOID = 17,
        OP_SOFTPLUS = 18,
        OP_EXP = 19,
        OP_NEG = 20,
        OP_TANH = 21,
        OP_RELU = 22,
        OP_SQR = 23,
        OP_SQRT = 24,
        OP_SWIGLU_SPLIT = 25,   // silu(a) * b
        OP_GEGLU_SPLIT = 26,    // gelu(a) * b
        OP_RESHAPE = 27,        // ip: ne0..ne(n-1)
        OP_VIEW = 28,           // ip: nd, ne0..ne3, nb1..nb3, offset
        OP_PERMUTE = 29,        // ip: axis0..axis3
        OP_TRANSPOSE = 30,
        OP_CONT = 31,
        OP_CAST = 32,           // ip: ggml type
        OP_CPY = 33,            // copy a into b; the result views b
        OP_SET_ROWS = 34,       // srcs: dst, src, row ids; the result views dst
        OP_ROPE_EXT = 35,       // srcs: x, positions, [freq factors]; ip: n_dims, mode, n_ctx_orig; fp: base, freq_scale, ext, attn, beta_fast, beta_slow
        OP_FLASH_ATTN_EXT = 36, // srcs: q, k, v, [mask]; fp: scale, max_bias, softcap; ip: [acc precision]
        OP_SOFT_MAX_EXT = 37,   // srcs: x, [mask]; fp: scale, max_bias
        OP_ARGSORT = 38,        // ip: order (0 asc, 1 desc)
        OP_TOP_K = 39,          // ip: k
        OP_SUM_ROWS = 40,
        OP_SSM_CONV = 41,       // srcs: window, kernel
        OP_GATED_DELTA_NET = 42,// srcs: q, k, v, g, beta, state; ip: snapshots K
        OP_CONCAT = 43,         // ip: dim
        OP_CLAMP = 44,          // fp: min, max
        OP_REPEAT = 45,         // srcs: x, shape-of
        OP_PAD = 46,            // ip: p0..p3
        OP_MEAN = 47,
        OP_ADD_ID = 48,         // srcs: a, b, ids
        OP_ARGMAX = 49,
        OP_KV_WINDOW = 50,      // srcs: cache; ip: head_dim, cache_size, kv_heads, start, length, kv_type, query_rows, gqa_ratio
                                // (a [head_dim, length, kv_heads] window, copied where a backend's attention needs it)
        OP_SWIGLU = 51,         // silu(first half) * second half, along ne0
        OP_GEGLU = 52,
    };

    struct Leaf
    {
        ggml_tensor* tensor;
        void* data;
        std::size_t bytes;
        int role;
        bool uploaded;   // bound to compute memory: uploaded every run (and downloaded after, for STATE)
    };

    struct Output
    {
        ggml_tensor* tensor;
        void* data;
        std::size_t bytes;
    };

    struct TsgGraph
    {
        ggml_context* ctx = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_gallocr_t galloc = nullptr;
        int rank = 0;
        int max_nodes = 0;
        bool built = false;
        bool replayable = false;   // owns its allocator (else it may run once)
        int runs = 0;
        std::vector<ggml_tensor*> nodes;
        std::vector<Leaf> leaves;
        std::vector<Output> outputs;
        std::vector<ggml_tensor*> effects;
        std::vector<BufferHandle> ephemeral;

        ~TsgGraph()
        {
            if (galloc != nullptr)
            {
                host_read_barrier();
                ggml_gallocr_free(galloc);
            }
            ephemeral.clear();
            if (ctx != nullptr) ggml_free(ctx);
        }
    };

    ggml_tensor* node_at(TsgGraph* g, int index)
    {
        if (index < 0) return nullptr;
        if (index >= static_cast<int>(g->nodes.size()))
            throw std::runtime_error("graph builder: source node index out of range");
        return g->nodes[static_cast<std::size_t>(index)];
    }

    int push(TsgGraph* g, ggml_tensor* t)
    {
        if (t == nullptr) throw std::runtime_error("graph builder: ggml returned no node");
        g->nodes.push_back(t);
        return static_cast<int>(g->nodes.size()) - 1;
    }

    ggml_tensor* build_op(TsgGraph* g, int op, const int* s, int ns, const std::int64_t* ip, int nip, const float* fp, int nfp)
    {
        ggml_context* ctx = g->ctx;
        auto src = [&](int i) -> ggml_tensor* { return i < ns ? node_at(g, s[i]) : nullptr; };
        auto need = [&](int count) { if (ns < count) throw std::runtime_error("graph builder: missing source operand"); };
        auto ipar = [&](int i, std::int64_t fallback) { return i < nip ? ip[i] : fallback; };
        auto fpar = [&](int i, float fallback) { return i < nfp ? fp[i] : fallback; };

        switch (op)
        {
        case OP_GET_ROWS: need(2); return ggml_get_rows(ctx, src(0), src(1));
        case OP_ADD: need(2); return ggml_add(ctx, src(0), src(1));
        case OP_SUB: need(2); return ggml_sub(ctx, src(0), src(1));
        case OP_MUL: need(2); return ggml_mul(ctx, src(0), src(1));
        case OP_DIV: need(2); return ggml_div(ctx, src(0), src(1));
        case OP_SCALE: need(1); return ggml_scale(ctx, src(0), fpar(0, 1.0f));
        case OP_SCALE_BIAS: need(1); return ggml_scale_bias(ctx, src(0), fpar(0, 1.0f), fpar(1, 0.0f));
        case OP_RMS_NORM: need(1); return ggml_rms_norm(ctx, src(0), fpar(0, 1e-6f));
        case OP_NORM: need(1); return ggml_norm(ctx, src(0), fpar(0, 1e-6f));
        case OP_L2_NORM: need(1); return ggml_l2_norm(ctx, src(0), fpar(0, 1e-6f));
        case OP_GROUP_NORM: need(1); return ggml_group_norm(ctx, src(0), static_cast<int>(ipar(0, 32)), fpar(0, 1e-6f));
        case OP_MUL_MAT:
        {
            need(2);
            ggml_tensor* r = ggml_mul_mat(ctx, src(0), src(1));
            if (ipar(0, 0) != 0) ggml_prec_set_acc(r, static_cast<ggml_prec>(ipar(0, 0)));
            return r;
        }
        case OP_MUL_MAT_ID: need(3); return ggml_mul_mat_id(ctx, src(0), src(1), src(2));
        case OP_SILU: need(1); return ggml_silu(ctx, src(0));
        case OP_GELU: need(1); return ggml_gelu(ctx, src(0));
        case OP_GELU_QUICK: need(1); return ggml_gelu_quick(ctx, src(0));
        case OP_SIGMOID: need(1); return ggml_sigmoid(ctx, src(0));
        case OP_SOFTPLUS: need(1); return ggml_softplus(ctx, src(0));
        case OP_EXP: need(1); return ggml_exp(ctx, src(0));
        case OP_NEG: need(1); return ggml_neg(ctx, src(0));
        case OP_TANH: need(1); return ggml_tanh(ctx, src(0));
        case OP_RELU: need(1); return ggml_relu(ctx, src(0));
        case OP_SQR: need(1); return ggml_sqr(ctx, src(0));
        case OP_SQRT: need(1); return ggml_sqrt(ctx, src(0));
        case OP_SWIGLU_SPLIT: need(2); return ggml_swiglu_split(ctx, src(0), src(1));
        case OP_GEGLU_SPLIT: need(2); return ggml_geglu_split(ctx, src(0), src(1));
        case OP_RESHAPE:
        {
            need(1);
            switch (nip)
            {
            case 1: return ggml_reshape_1d(ctx, src(0), ip[0]);
            case 2: return ggml_reshape_2d(ctx, src(0), ip[0], ip[1]);
            case 3: return ggml_reshape_3d(ctx, src(0), ip[0], ip[1], ip[2]);
            case 4: return ggml_reshape_4d(ctx, src(0), ip[0], ip[1], ip[2], ip[3]);
            default: throw std::runtime_error("graph builder: reshape needs 1-4 extents");
            }
        }
        case OP_VIEW:
        {
            need(1);
            if (nip < 9) throw std::runtime_error("graph builder: view needs nd, ne0..ne3, nb1..nb3, offset");
            const std::size_t off = static_cast<std::size_t>(ip[8]);
            switch (ip[0])
            {
            case 1: return ggml_view_1d(ctx, src(0), ip[1], off);
            case 2: return ggml_view_2d(ctx, src(0), ip[1], ip[2], static_cast<std::size_t>(ip[5]), off);
            case 3: return ggml_view_3d(ctx, src(0), ip[1], ip[2], ip[3], static_cast<std::size_t>(ip[5]), static_cast<std::size_t>(ip[6]), off);
            case 4: return ggml_view_4d(ctx, src(0), ip[1], ip[2], ip[3], ip[4], static_cast<std::size_t>(ip[5]),
                                        static_cast<std::size_t>(ip[6]), static_cast<std::size_t>(ip[7]), off);
            default: throw std::runtime_error("graph builder: view rank must be 1-4");
            }
        }
        case OP_PERMUTE:
            need(1);
            return ggml_permute(ctx, src(0), static_cast<int>(ipar(0, 0)), static_cast<int>(ipar(1, 1)),
                                static_cast<int>(ipar(2, 2)), static_cast<int>(ipar(3, 3)));
        case OP_TRANSPOSE: need(1); return ggml_transpose(ctx, src(0));
        case OP_CONT: need(1); return ggml_cont(ctx, src(0));
        case OP_CAST: need(1); return ggml_cast(ctx, src(0), static_cast<ggml_type>(ipar(0, GGML_TYPE_F32)));
        case OP_CPY: need(2); return ggml_cpy(ctx, src(0), src(1));
        case OP_SET_ROWS: need(3); return ggml_set_rows(ctx, src(0), src(1), src(2));
        case OP_ROPE_EXT:
            need(2);
            return ggml_rope_ext(ctx, src(0), src(1), src(2),
                                 static_cast<int>(ipar(0, 0)), static_cast<int>(ipar(1, 0)), static_cast<int>(ipar(2, 0)),
                                 fpar(0, 10000.0f), fpar(1, 1.0f), fpar(2, 0.0f), fpar(3, 1.0f), fpar(4, 32.0f), fpar(5, 1.0f));
        case OP_FLASH_ATTN_EXT:
        {
            need(3);
            // srcs[4]: optional attention sinks.
            return flash_attn_ext_guarded(ctx, "graph builder", src(0), src(1), src(2), src(3),
                                          fpar(0, 1.0f), fpar(1, 0.0f), fpar(2, 0.0f), src(4),
                                          static_cast<ggml_prec>(ipar(0, 0)));
        }
        case OP_SOFT_MAX_EXT: need(1); return ggml_soft_max_ext(ctx, src(0), src(1), fpar(0, 1.0f), fpar(1, 0.0f));
        case OP_ARGSORT: need(1); return ggml_argsort(ctx, src(0), ipar(0, 0) == 0 ? GGML_SORT_ORDER_ASC : GGML_SORT_ORDER_DESC);
        case OP_TOP_K: need(1); return ggml_top_k(ctx, src(0), static_cast<int>(ipar(0, 1)));
        case OP_SUM_ROWS: need(1); return ggml_sum_rows(ctx, src(0));
        case OP_SSM_CONV: need(2); return ssm_conv_any(ctx, src(0), src(1));
        case OP_GATED_DELTA_NET: need(6); return ggml_gated_delta_net(ctx, src(0), src(1), src(2), src(3), src(4), src(5), ipar(0, 1));
        case OP_CONCAT: need(2); return ggml_concat(ctx, src(0), src(1), static_cast<int>(ipar(0, 0)));
        case OP_CLAMP: need(1); return ggml_clamp(ctx, src(0), fpar(0, -INFINITY), fpar(1, INFINITY));
        case OP_REPEAT: need(2); return ggml_repeat(ctx, src(0), src(1));
        case OP_PAD:
            need(1);
            return ggml_pad(ctx, src(0), static_cast<int>(ipar(0, 0)), static_cast<int>(ipar(1, 0)),
                            static_cast<int>(ipar(2, 0)), static_cast<int>(ipar(3, 0)));
        case OP_MEAN: need(1); return ggml_mean(ctx, src(0));
        case OP_ADD_ID: need(3); return ggml_add_id(ctx, src(0), src(1), src(2));
        case OP_ARGMAX: need(1); return ggml_argmax(ctx, src(0));
        case OP_SWIGLU: need(1); return ggml_swiglu(ctx, src(0));
        case OP_GEGLU: need(1); return ggml_geglu(ctx, src(0));
        case OP_KV_WINDOW:
        {
            need(1);
            if (nip < 6) throw std::runtime_error("graph builder: kv window needs head_dim, cache_size, kv_heads, start, length, kv_type");
            ggml_tensor* w = view_kv_cache_window(ctx, src(0), static_cast<int>(ip[0]), static_cast<int>(ip[1]),
                                                  static_cast<int>(ip[2]), static_cast<int>(ip[3]), static_cast<int>(ip[4]),
                                                  static_cast<int>(ip[5]), static_cast<int>(ipar(6, 1)), static_cast<int>(ipar(7, 0)));
            if (w == nullptr) throw std::runtime_error("graph builder: kv window view failed");
            return w;
        }
        default:
            throw std::runtime_error("graph builder: unknown op " + std::to_string(op));
        }
    }

    // Bind the leaves, then let the graph's own allocator place every other tensor.
    bool build(TsgGraph* g)
    {
        g->graph = ggml_new_graph_custom(g->ctx, g->max_nodes, false);
        if (g->graph == nullptr) { set_last_error("graph builder: graph allocation failed."); return false; }
        for (ggml_tensor* e : g->effects) ggml_build_forward_expand(g->graph, e);
        for (const Output& o : g->outputs) ggml_build_forward_expand(g->graph, o.tensor);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        for (Leaf& leaf : g->leaves)
        {
            leaf.uploaded = true;
            if (leaf.role == LEAF_INPUT || leaf.data == nullptr) continue;
            const auto usage = leaf.role == LEAF_STATE ? GGML_BACKEND_BUFFER_USAGE_COMPUTE : GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
            if (leaf.bytes >= 4096)
            {
                bool needs_upload = false;
                if (try_bind_cached_tensor(g_backend, dev, leaf.tensor, leaf.data, leaf.bytes, needs_upload, usage))
                {
                    if (needs_upload)
                        ggml_backend_tensor_set(leaf.tensor, resolve_upload_source(leaf.data), 0, leaf.bytes);
                    leaf.uploaded = false;
                    continue;
                }
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(g_backend, dev, leaf.data, leaf.bytes, true, buf)
                    && ggml_backend_tensor_alloc(buf, leaf.tensor, leaf.data) == GGML_STATUS_SUCCESS)
                {
                    leaf.uploaded = false;
                    continue;
                }
            }
            // Compute memory: the allocator must neither reuse nor clobber it before it is read.
            ggml_set_input(leaf.tensor);
        }
        for (const Output& o : g->outputs) ggml_set_output(o.tensor);

        // A one-shot graph borrows the backend's persistent, reused compute allocator (the
        // memory the fused graphs share, grown on demand); a graph that will be replayed keeps
        // its own, so no other graph can reuse its intermediates between runs.
        if (!g->replayable && alloc_graph_reuse_gallocr(g->graph))
        {
            g->built = true;
            return true;
        }
        g->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        if (g->galloc == nullptr || !ggml_gallocr_alloc_graph(g->galloc, g->graph))
        {
            set_last_error("graph builder: failed to allocate the graph's compute memory.");
            return false;
        }
        g->replayable = true;
        g->built = true;
        return true;
    }
}

TSG_EXPORT void* TSGgml_GraphCreate(int max_nodes, int replayable)
{
    try
    {
        if (!ensure_backend()) return nullptr;
        if (max_nodes < 16) max_nodes = 16;
        auto* g = new TsgGraph();
        g->rank = g_active_rank;
        g->max_nodes = max_nodes;
        g->replayable = replayable != 0;
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * static_cast<std::size_t>(max_nodes) * 2
                          + ggml_graph_overhead_custom(static_cast<std::size_t>(max_nodes), false);
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        g->ctx = ggml_init(params);
        if (g->ctx == nullptr)
        {
            delete g;
            set_last_error("graph builder: ggml_init failed.");
            return nullptr;
        }
        return g;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return nullptr;
    }
}

TSG_EXPORT void TSGgml_GraphFree(void* handle)
{
    auto* g = static_cast<TsgGraph*>(handle);
    if (g == nullptr) return;
    ScopedRank scope(g->rank);
    delete g;
}

// A leaf of `type` with `nd` extents; `data` is null only for an INPUT supplied per run.
TSG_EXPORT int TSGgml_GraphLeaf(void* handle, int role, int type, int nd, const std::int64_t* ne,
                                void* data, std::int64_t bytes)
{
    auto* g = static_cast<TsgGraph*>(handle);
    try
    {
        if (g == nullptr || g->built || nd < 1 || nd > 4 || ne == nullptr)
            throw std::runtime_error("graph builder: bad leaf (or graph already built)");
        ggml_tensor* t = ggml_new_tensor(g->ctx, static_cast<ggml_type>(type), nd, ne);
        if (t == nullptr) throw std::runtime_error("graph builder: leaf allocation failed");
        if (static_cast<std::size_t>(bytes) < ggml_nbytes(t) && data != nullptr)
            throw std::runtime_error("graph builder: leaf host buffer is smaller than its tensor");
        g->leaves.push_back({ t, data, data != nullptr ? ggml_nbytes(t) : 0, role, true });
        if (role == LEAF_INPUT) ggml_set_input(t);
        return push(g, t);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return -1;
    }
}

TSG_EXPORT int TSGgml_GraphOp(void* handle, int op, const int* srcs, int nsrc,
                              const std::int64_t* ip, int nip, const float* fp, int nfp)
{
    auto* g = static_cast<TsgGraph*>(handle);
    try
    {
        if (g == nullptr || g->built) throw std::runtime_error("graph builder: no graph, or graph already built");
        return push(g, build_op(g, op, srcs, nsrc, ip, nip, fp, nfp));
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return -1;
    }
}

// Shape of a node (ne0..ne3) and its ggml type, so managed code can derive views and checks.
TSG_EXPORT int TSGgml_GraphShape(void* handle, int node, std::int64_t* ne_out, std::int64_t* nb_out, int* type_out)
{
    auto* g = static_cast<TsgGraph*>(handle);
    try
    {
        ggml_tensor* t = node_at(g, node);
        if (t == nullptr) return 0;
        for (int i = 0; i < 4; i++)
        {
            if (ne_out != nullptr) ne_out[i] = t->ne[i];
            if (nb_out != nullptr) nb_out[i] = static_cast<std::int64_t>(t->nb[i]);
        }
        if (type_out != nullptr) *type_out = static_cast<int>(t->type);
        return 1;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
}

// A node whose value leaves the graph: downloaded to `data` after every run.
TSG_EXPORT int TSGgml_GraphOutput(void* handle, int node, void* data, std::int64_t bytes)
{
    auto* g = static_cast<TsgGraph*>(handle);
    try
    {
        if (g == nullptr || g->built) throw std::runtime_error("graph builder: no graph, or graph already built");
        ggml_tensor* t = node_at(g, node);
        if (data != nullptr && static_cast<std::size_t>(bytes) < ggml_nbytes(t))
            throw std::runtime_error("graph builder: output buffer is smaller than its node");
        g->outputs.push_back({ t, data, data != nullptr ? ggml_nbytes(t) : 0 });
        return 1;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
}

// A node run for its side effect (a cache or state write), in emission order.
TSG_EXPORT int TSGgml_GraphEffect(void* handle, int node)
{
    auto* g = static_cast<TsgGraph*>(handle);
    try
    {
        if (g == nullptr || g->built) throw std::runtime_error("graph builder: no graph, or graph already built");
        g->effects.push_back(node_at(g, node));
        return 1;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
}

// Run the graph: build and bind it on the first run, then upload this run's inputs (and the
// leaves living in compute memory), compute, download the outputs and write back any STATE
// leaf that lives in compute memory. `inputs`/`input_data`/`input_bytes` name per-run values
// for INPUT leaves (or replace a leaf's host pointer for this run when it lives in compute memory).
TSG_EXPORT int TSGgml_GraphRun(void* handle, const int* inputs, void* const* input_data,
                               const std::int64_t* input_bytes, int ninputs)
{
    auto* g = static_cast<TsgGraph*>(handle);
    if (g == nullptr) { set_last_error("graph builder: no graph."); return 0; }
    try
    {
        ScopedRank scope(g->rank);
        if (!ensure_backend()) return 0;
        if (!g->built && !build(g)) return 0;
        if (g->runs > 0 && !g->replayable)
        {
            set_last_error("graph builder: a one-shot graph cannot run twice.");
            return 0;
        }
        g->runs++;

        // Device work queued by earlier calls may still read or write these host buffers.
        host_read_barrier();
        for (const Leaf& leaf : g->leaves)
            if (leaf.uploaded && leaf.data != nullptr)
                ggml_backend_tensor_set(leaf.tensor, resolve_upload_source(leaf.data), 0, leaf.bytes);
        for (int i = 0; i < ninputs; i++)
        {
            ggml_tensor* t = node_at(g, inputs[i]);
            const std::size_t n = std::min(static_cast<std::size_t>(input_bytes[i]), ggml_nbytes(t));
            if (input_data[i] != nullptr && n > 0)
                ggml_backend_tensor_set(t, input_data[i], 0, n);
        }

        if (compute_graph(g_backend, g->graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("graph builder: backend graph execution failed.");
            return 0;
        }
        sync_backend(g_backend);
        for (const Output& o : g->outputs)
            if (o.data != nullptr)
                ggml_backend_tensor_get(o.tensor, o.data, 0, o.bytes);
        for (const Leaf& leaf : g->leaves)
            if (leaf.role == LEAF_STATE && leaf.uploaded && leaf.data != nullptr)
                ggml_backend_tensor_get(leaf.tensor, leaf.data, 0, leaf.bytes);
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
}

// Whether a leaf was bound to device memory that outlives the run (its host bytes are then
// stale after the run on a device with its own memory until synced).
TSG_EXPORT int TSGgml_GraphLeafIsResident(void* handle, int node)
{
    auto* g = static_cast<TsgGraph*>(handle);
    if (g == nullptr || !g->built) return 0;
    ggml_tensor* t = node < static_cast<int>(g->nodes.size()) && node >= 0 ? g->nodes[static_cast<std::size_t>(node)] : nullptr;
    for (const Leaf& leaf : g->leaves)
        if (leaf.tensor == t) return leaf.uploaded ? 0 : 1;
    return 0;
}
