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
#include "ggml_ops_graph_optimize.h"
#include "ggml_ops_transformer_common.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <atomic>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(TSG_GGML_USE_METAL)
#include "tsm/tsm_moe.h"
#endif

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
        OP_ROPE_MULTI = 53,     // srcs: x, positions [4 * rows], [freq factors]; ip: n_dims, mode, n_ctx_orig, sections[4]; fp: as ROPE_EXT
        OP_MOE_HOST = 54,       // srcs: x [n, T], ids [k, T], weights [k, T], gate/up/down expert stacks (host); ip: layer
        OP_LIGHTNING_INDEXER = 55, // srcs: q [d, heads, T], k [d, 1, n_kv], weights [heads, T], mask [n_kv, T] (F16)
        OP_TOP_K_MASK = 56,     // srcs: mask [n_kv, T], top_k [n_top, T], then (ip[0..2] flags) score, blocks, extra
    };

    struct Leaf
    {
        ggml_tensor* tensor;
        void* data;
        std::size_t bytes;
        int role;
        bool uploaded;   // bound to compute memory: uploaded every run (and downloaded after, for STATE)
        bool host_only = false;                        // read from host memory at a host-expert pause, never bound
        bool has_split = false;                        // tensor parallelism: how the leaf spreads over the ranks
        ggml_backend_meta_split_state split{};
    };

    struct Output
    {
        ggml_tensor* tensor;
        void* data;
        std::size_t bytes;
    };

    // ---- tensor parallelism: a graph on ggml's meta device over several ranks (upstream's
    // tensor-split mechanism). Weights and states live in meta buffers, spread by the split state
    // the managed side gives each leaf (heads, channels, experts; mirrored otherwise); compute
    // tensors take the meta backend's inferred states, and its collectives reduce partial sums.
    struct TpGroup
    {
        std::vector<int> ranks;
        ggml_backend_dev_t meta = nullptr;
        ggml_backend_t backend = nullptr;
    };

    std::mutex g_tp_mu;
    std::vector<std::unique_ptr<TpGroup>> g_tp_groups;

    // Split states of meta-resident tensors, by tensor name (the meta buffer asks by tensor).
    std::mutex g_split_mu;
    std::unordered_map<std::string, ggml_backend_meta_split_state> g_splits;

    ggml_backend_meta_split_state tp_split_cb(const ggml_tensor* t, void*)
    {
        ggml_backend_meta_split_state st;
        std::memset(&st, 0, sizeof(st));
        st.axis = GGML_BACKEND_SPLIT_AXIS_MIRRORED;
        st.nr[0] = 1;
        st.n_segments = 1;
        std::lock_guard<std::mutex> lk(g_split_mu);
        auto it = g_splits.find(t->name);
        return it == g_splits.end() ? st : it->second;
    }

    TpGroup* tp_group(const int* ranks, int n)
    {
        std::lock_guard<std::mutex> lk(g_tp_mu);
        for (const auto& grp : g_tp_groups)
            if (static_cast<int>(grp->ranks.size()) == n && std::equal(grp->ranks.begin(), grp->ranks.end(), ranks))
                return grp.get();
        if (n < 2 || n > GGML_BACKEND_META_MAX_DEVICES)
            throw std::runtime_error("graph builder: tensor parallelism spans 2 to 16 ranks");
        auto grp = std::make_unique<TpGroup>();
        ggml_backend_dev_t devs[GGML_BACKEND_META_MAX_DEVICES] = {};
        int device_indices[GGML_BACKEND_META_MAX_DEVICES] = {};
        for (int i = 0; i < n; i++)
        {
            ScopedRank scope(ranks[i]);
            if (!ensure_backend() || g_backend == nullptr)
                throw std::runtime_error("graph builder: a tensor-parallel rank has no backend");
            devs[i] = ggml_backend_get_device(g_backend);
            device_indices[i] = dev().device_index >= 0 ? dev().device_index : ranks[i];
            grp->ranks.push_back(ranks[i]);
        }
#ifdef TSG_GGML_USE_CUDA
        tp_cuda_prepare_collective(device_indices, n);   // before the meta backend creates its communicator
#else
        (void) device_indices;
#endif
        static std::mutex meta_mu;   // ggml's meta device cache is not thread-safe
        std::lock_guard<std::mutex> meta_lk(meta_mu);
        grp->meta = ggml_backend_meta_device(devs, static_cast<size_t>(n), tp_split_cb, nullptr);
        grp->backend = grp->meta != nullptr ? ggml_backend_dev_init(grp->meta, nullptr) : nullptr;
        if (grp->backend == nullptr)
            throw std::runtime_error("graph builder: creating the tensor-parallel backend failed");
        g_tp_groups.push_back(std::move(grp));
        return g_tp_groups.back().get();
    }

    // A weight or state resident in a meta buffer, keyed by its host pointer; a graph's leaf is
    // placed at the same address, so it aliases the resident slices on every rank.
    struct MetaResident
    {
        ggml_context* ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor* t = nullptr;
        std::size_t bytes = 0;
        ~MetaResident()
        {
            if (buf != nullptr) ggml_backend_buffer_free(buf);
            if (ctx != nullptr) ggml_free(ctx);
        }
    };

    std::mutex g_resident_mu;   // also serializes meta set_tensor (not thread-safe)
    std::map<std::pair<TpGroup*, void*>, std::unique_ptr<MetaResident>> g_residents;

    MetaResident* meta_resident(TpGroup* grp, const Leaf& leaf)
    {
        std::lock_guard<std::mutex> lk(g_resident_mu);
        auto key = std::make_pair(grp, leaf.data);
        auto it = g_residents.find(key);
        if (it != g_residents.end())
        {
            if (it->second->bytes == leaf.bytes && ggml_are_same_shape(it->second->t, leaf.tensor) && it->second->t->type == leaf.tensor->type)
                return it->second.get();
            g_residents.erase(it);
        }
        auto r = std::make_unique<MetaResident>();
        char name[GGML_MAX_NAME];
        std::snprintf(name, sizeof(name), "tsgtp:%p:%d", leaf.data, leaf.has_split ? static_cast<int>(leaf.split.axis) : -1);
        if (leaf.has_split)
        {
            std::lock_guard<std::mutex> slk(g_split_mu);
            g_splits[name] = leaf.split;
        }
        ggml_init_params params{ ggml_tensor_overhead() * 2, nullptr, true };
        r->ctx = ggml_init(params);
        if (r->ctx == nullptr) throw std::runtime_error("graph builder: tensor-parallel context allocation failed");
        r->t = ggml_new_tensor(r->ctx, leaf.tensor->type, GGML_MAX_DIMS, leaf.tensor->ne);
        ggml_set_name(r->t, name);
        r->buf = ggml_backend_alloc_ctx_tensors_from_buft(r->ctx, ggml_backend_dev_buffer_type(grp->meta));
        if (r->buf == nullptr) throw std::runtime_error("graph builder: tensor-parallel weight allocation failed (out of device memory?)");
        r->bytes = leaf.bytes;
        ggml_backend_tensor_set(r->t, resolve_upload_source(leaf.data), 0, leaf.bytes);
        MetaResident* raw = r.get();
        g_residents[key] = std::move(r);
        return raw;
    }

    // The cgraph's slots: every tensor the context can hold (lowering emits several ggml tensors for one program
    // node - a tiered host-expert pause builds its hot part, about a dozen per MoE layer), plus leaves from other
    // contexts (tier copies). Sized from the program's node count alone, a 48-layer tiered prefill overflowed it.
    std::size_t graph_slots(int max_nodes) { return static_cast<std::size_t>(max_nodes) * 2 + 256; }

    // Fused regions (GraphFusion, TSGgml_GraphFusedBegin) a backend kernel computes as one device operation between
    // graph segments, in place of the region's ops: the operands it reads and writes live in an IO buffer outside the
    // graph allocator (which frees a tensor after its last graph consumer and never frees one without consumers), one
    // region per role that every such region of the graph reuses (the queue orders the kernels and the segments).
    enum FusedKind
    {
        FUSED_DELTA_NET_PROMPT = 5,   // GraphFusion.DeltaNetPrompt
        FUSED_HC_READ = 6,            // GraphFusion.HcRead
        SELECTED_ATTENTION = 100,     // a prompt chunk's flash attention under a selection's mask (no region: the op)
    };

    struct DeviceIo
    {
        ggml_backend_buffer_t buf = nullptr;
        std::size_t size = 0;
        ~DeviceIo() { if (buf != nullptr) ggml_backend_buffer_free(buf); }
    };

    // The newest IO buffer per backend, reused by every graph while it is big enough (a new Metal buffer costs ~75 ms
    // a GB); a graph keeps the one it bound to, so growing never frees memory a built graph still points into. Like
    // the reused compute allocator one-shot graphs share (alloc_graph_reuse_gallocr), it assumes a backend runs one
    // graph at a time.
    std::mutex g_device_io_mu;
    std::map<ggml_backend_t, std::shared_ptr<DeviceIo>> g_device_io;

    std::shared_ptr<DeviceIo> device_io(ggml_backend_t backend, std::size_t bytes)
    {
        std::lock_guard<std::mutex> lk(g_device_io_mu);
        std::shared_ptr<DeviceIo>& io = g_device_io[backend];
        if (io != nullptr && io->size >= bytes) return io;
        io.reset();
        auto fresh = std::make_shared<DeviceIo>();
        fresh->size = bytes + bytes / 4;   // headroom: a prompt's chunks grow by a few rows at a time
        fresh->buf = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), fresh->size);
        if (fresh->buf == nullptr) throw std::runtime_error("graph builder: the device operations' IO buffer could not be allocated");
        io = fresh;
        return io;
    }

    // A served region: its operands (the program's inputs to it, in the kind's order) and its results.
    struct DeviceRegion
    {
        int kind = 0;
        std::vector<ggml_tensor*> in;
        std::vector<std::int64_t> ip;
        std::vector<float> fp;
        std::vector<ggml_tensor*> results;
    };

    // The IO buffer's roles: each operand a kind's operation keeps there, one region per role (every region of the
    // kind reuses it; different kinds never share one).
    enum IoRole
    {
        IO_GDN_QKV, IO_GDN_Z, IO_GDN_ALPHA, IO_GDN_BETA, IO_GDN_Y, IO_HC_MIXED, IO_HC_INJECT,
        IO_SEL_Q, IO_SEL_MASK, IO_SEL_IDX, IO_SEL_CNT, IO_SEL_OUT, IO_ROLES
    };

    // A served region's operands in the IO buffer, by role.
    std::vector<std::pair<IoRole, ggml_tensor*>> io_operands(const DeviceRegion& r)
    {
        switch (r.kind)
        {
        case FUSED_DELTA_NET_PROMPT:
            return { { IO_GDN_QKV, r.in[0] }, { IO_GDN_Z, r.in[1] }, { IO_GDN_ALPHA, r.in[2] }, { IO_GDN_BETA, r.in[3] },
                     { IO_GDN_Y, r.results[0] } };
        case FUSED_HC_READ:
            return { { IO_HC_MIXED, r.results[0] }, { IO_HC_INJECT, r.results[1] } };
        case SELECTED_ATTENTION:
            return { { IO_SEL_Q, r.in[0] }, { IO_SEL_MASK, r.in[3] }, { IO_SEL_IDX, r.results[0] }, { IO_SEL_CNT, r.results[1] },
                     { IO_SEL_OUT, r.results[2] } };
        default:
            return {};
        }
    }

    struct TsgGraph
    {
        TpGroup* tp = nullptr;     // tensor parallelism: the meta device this graph runs on
        ggml_context* ctx = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_gallocr_t galloc = nullptr;
        int rank = 0;
        int max_nodes = 0;
        bool built = false;
        bool replayable = false;   // owns its allocator (else it may run once)
        bool reorder = false;      // let the backend reorder for concurrency (Metal), opted in per family
        int runs = 0;
        std::vector<ggml_tensor*> nodes;
        std::vector<Leaf> leaves;
        std::vector<Output> outputs;
        std::vector<ggml_tensor*> effects;
        std::vector<BufferHandle> ephemeral;
        struct Alias { ggml_tensor* node; ggml_tensor* base; std::size_t offset; ggml_tensor* fallback; };
        std::vector<Alias> aliases;
        // Host-expert pauses (OP_MOE_HOST, tsg::HostMoeSegment) and the node each segment ends after.
        std::vector<tsg::HostMoeSegment> host_moe;
        // Per pause, nodes expanded after its route weights and before its ids: views that keep a fused router
        // chain's input and ids allocated (see build_host_experts). Null where not needed.
        std::vector<std::pair<ggml_tensor*, ggml_tensor*>> host_moe_holds;
        std::vector<int> seg_end;
        // Effects and pauses in program order (kind 0: effects[i], 1: host_moe[i]): the graph is expanded in this
        // order, so a pause's boundary tensors precede everything that reads its result.
        std::vector<std::pair<int, int>> order;
        // Fused regions served by a device operation (TSGgml_GraphFusedBegin), the open regions (the served one's
        // index, -1 for one whose ops are built), the served region being recorded (-1: none) and the nodes its ops
        // would have produced (no tensor: nothing outside the region reads them).
        std::vector<DeviceRegion> regions;
        std::vector<int> open_regions;
        int serving = -1;
        std::vector<char> skipped;
        std::shared_ptr<DeviceIo> io;

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

    ggml_backend_t graph_backend(const TsgGraph* g) { return g->tp != nullptr ? g->tp->backend : g_backend; }


    ggml_tensor* node_at(TsgGraph* g, int index)
    {
        if (index < 0) return nullptr;
        if (index >= static_cast<int>(g->nodes.size()))
            throw std::runtime_error("graph builder: source node index out of range");
        if (g->skipped[static_cast<std::size_t>(index)])
            throw std::runtime_error("graph builder: node " + std::to_string(index)
                                     + " is inside a region a device operation computes; only the region's results are read after it");
        return g->nodes[static_cast<std::size_t>(index)];
    }

    int push(TsgGraph* g, ggml_tensor* t)
    {
        if (t == nullptr) throw std::runtime_error("graph builder: ggml returned no node");
        g->nodes.push_back(t);
        g->skipped.push_back(0);
        return static_cast<int>(g->nodes.size()) - 1;
    }

    // A node of a served region's ops: numbered like the program's, never built.
    int push_skipped(TsgGraph* g)
    {
        g->nodes.push_back(nullptr);
        g->skipped.push_back(1);
        return static_cast<int>(g->nodes.size()) - 1;
    }

    Leaf* leaf_of(TsgGraph* g, const ggml_tensor* t)
    {
        for (Leaf& leaf : g->leaves)
            if (leaf.tensor == t) return &leaf;
        return nullptr;
    }

    /// Routed experts from host memory (tsg::HostMoeSegment): the graph pauses after the router's outputs, the
    /// host computes the experts into moe_out, and an accelerator tier registered for the gate stack
    /// (ggml_ops_moe_hot_tier.cpp) runs the routes it holds in the next segment.
    ggml_tensor* build_host_experts(TsgGraph* g, const int* s, int ns, const std::int64_t* ip, int nip)
    {
        if (ns < 6) throw std::runtime_error("graph builder: host experts take x, ids, weights and three expert stacks");
        if (g->tp != nullptr) throw std::runtime_error("graph builder: host experts under tensor parallelism are not supported");
        ggml_context* ctx = g->ctx;
        ggml_tensor* x = node_at(g, s[0]);
        ggml_tensor* ids = node_at(g, s[1]);
        ggml_tensor* weights = node_at(g, s[2]);
        Leaf* parts[3] = { leaf_of(g, node_at(g, s[3])), leaf_of(g, node_at(g, s[4])), leaf_of(g, node_at(g, s[5])) };
        for (Leaf* leaf : parts)
            if (leaf == nullptr || leaf->data == nullptr || leaf->role != LEAF_WEIGHT)
                throw std::runtime_error("graph builder: host experts need weight leaves with host data");
        const std::int64_t n = x->ne[0], T = ggml_nelements(x) / n, n_used = ids->ne[0];
        const ggml_tensor* gate = parts[0]->tensor;
        const std::int64_t n_ff = gate->ne[1], experts = gate->ne[2];
        auto boundary = [&](ggml_tensor* t) {
            if (!ggml_is_contiguous(t)) t = ggml_cont(ctx, t);
            for (ggml_tensor* p = t; p != nullptr; p = p->view_src) ggml_set_output(p);
            return t;
        };
        tsg::HostMoeSegment hm;
        hm.layer = nip > 0 ? static_cast<int>(ip[0]) : 0;
        hm.moe_in = boundary(ggml_reshape_2d(ctx, ggml_is_contiguous(x) ? x : ggml_cont(ctx, x), n, T));
        hm.weights = boundary(ggml_reshape_2d(ctx, ggml_is_contiguous(weights) ? weights : ggml_cont(ctx, weights), n_used, T));
        // A router written as llama.cpp's chain (softmax, argsort top-k, get_rows, renormalise, scale) is one kernel on
        // ggml-metal, which declines it when a chain node is a graph output or the allocator reused the logits'
        // storage inside it. So contiguous ids (one row) are not pinned (ggml_set_output would flag the argsort):
        // views no node consumes hold the argsort's storage (ggml-alloc frees a view source only after its last
        // view is consumed). And a VIEW of the router weight - a bound constant the allocator never frees - reading
        // the logits as a second source keeps them allocated until after the chain: what ggml_backend_sched's alloc
        // deps do for a fused kernel's inputs, for this graph that gallocr allocates directly.
        ggml_tensor* keep_logits = nullptr;
        ggml_tensor* hold_ids = nullptr;
        if (ggml_is_contiguous(ids))
        {
            hold_ids = ggml_view_1d(ctx, ids, 1, 0);
            hm.sel_ids = ggml_view_2d(ctx, ids, n_used, T, ggml_row_size(ids->type, n_used), 0);
        }
        else
        {
            hm.sel_ids = boundary(ggml_reshape_2d(ctx, ggml_cont(ctx, ids), n_used, T));
        }
        if (const ggml_tensor* sort = ids->view_src; sort != nullptr && sort->op == GGML_OP_ARGSORT)
            if (ggml_tensor* soft = sort->src[0]; soft != nullptr && soft->op == GGML_OP_SOFT_MAX)
                if (ggml_tensor* logits = soft->src[0]; logits != nullptr && logits->op == GGML_OP_MUL_MAT)
                    if (Leaf* router = leaf_of(g, logits->src[0]); router != nullptr && router->role == LEAF_WEIGHT)
                    {
                        keep_logits = ggml_view_1d(ctx, router->tensor, 1, 0);
                        keep_logits->src[1] = logits;
                    }
        hm.moe_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, T);
        ggml_set_input(hm.moe_out);
        ggml_set_output(hm.moe_out);
        void** data[3] = { &hm.gate_data, &hm.up_data, &hm.down_data };
        int* type[3] = { &hm.gate_type, &hm.up_type, &hm.down_type };
        std::int64_t* bytes[3] = { &hm.gate_bytes, &hm.up_bytes, &hm.down_bytes };
        for (int i = 0; i < 3; i++)
        {
            parts[i]->host_only = true;
            *data[i] = parts[i]->data;
            *type[i] = parts[i]->tensor->type;
            *bytes[i] = static_cast<std::int64_t>(parts[i]->bytes);
        }
        hm.gate_ne0 = n; hm.gate_ne1 = n_ff;
        hm.up_ne0 = n; hm.up_ne1 = n_ff;
        hm.down_ne0 = n_ff; hm.down_ne1 = n;
        hm.activation = 0;   // silu(gate) * up
        hm.num_experts = static_cast<int>(experts);
        hm.n_used = static_cast<int>(n_used);
        hm.n_ff = static_cast<int>(n_ff);
        hm.seq_len = static_cast<int>(T);
        hm.hidden = static_cast<int>(n);
        ggml_tensor* hot = nullptr;
        if (tsg::MoeHotTier* tier = tsg::moe_hot_tier_find(hm.gate_data))
            if (T <= tsg::host_moe_tier_max_rows(*tier) && tier->num_experts == hm.num_experts && tier->count >= hm.n_used
                && tier->type[0] == hm.gate_type && tier->type[1] == hm.up_type && tier->type[2] == hm.down_type)
                hot = tsg::host_moe_build_hot_part(ctx, hm, hm.moe_in, *tier, ggml_mul_mat_id, 0, nullptr, nullptr);
        // Strata's read-ahead (srcs 6 and 7: the next offloaded layer's router and gate stack): that router applied
        // to this layer's FFN input before the pause, so the seam reads the next tier's predicted misses into its
        // window while the accelerator works (moe_hot_tier_prefetch). Only where the seam computes the tier's routes
        // itself (decode and verify widths) and the next tier has a window to fill.
        if (ns >= 8)
        {
            Leaf* next_router = leaf_of(g, node_at(g, s[6]));
            Leaf* next_gate = leaf_of(g, node_at(g, s[7]));
            if (next_router == nullptr || next_gate == nullptr || next_gate->data == nullptr || next_gate->role != LEAF_WEIGHT)
                throw std::runtime_error("graph builder: the read-ahead takes the next layer's router and gate stack leaves");
            next_gate->host_only = true;   // host experts' data, never bound
            if (hm.tsm_hot != nullptr && tsg::moe_tier_prefetch_k() > 0)
                if (tsg::MoeHotTier* next_tier = tsg::moe_hot_tier_find(next_gate->data))
                    if (next_tier->cpu_visible && next_tier->count > next_tier->fixed && next_tier->num_experts == hm.num_experts)
                        hm.pred_next = boundary(ggml_mul_mat(ctx, next_router->tensor, hm.moe_in));   // [experts, T]
        }
        g->host_moe.push_back(hm);
        g->host_moe_holds.emplace_back(keep_logits, hold_ids);
        g->order.emplace_back(1, static_cast<int>(g->host_moe.size()) - 1);
        return hot != nullptr ? ggml_add(ctx, hm.moe_out, hot) : hm.moe_out;
    }

    ggml_tensor* try_selected_attention(TsgGraph* g, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask,
                                        float scale, std::int64_t width);

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
        case OP_MOE_HOST: return build_host_experts(g, s, ns, ip, nip);
        case OP_ROPE_MULTI:
        {
            need(2);
            int sections[GGML_MROPE_SECTIONS] = {};
            for (int i = 0; i < GGML_MROPE_SECTIONS && i < 4; i++) sections[i] = static_cast<int>(ipar(3 + i, 0));
            return ggml_rope_multi(ctx, src(0), src(1), src(2),
                                   static_cast<int>(ipar(0, 0)), sections, static_cast<int>(ipar(1, 0)), static_cast<int>(ipar(2, 0)),
                                   fpar(0, 10000.0f), fpar(1, 1.0f), fpar(2, 0.0f), fpar(3, 1.0f), fpar(4, 32.0f), fpar(5, 1.0f));
        }
        case OP_FLASH_ATTN_EXT:
        {
            need(3);
            if (g->tp != nullptr)
            {
                // Tensor parallelism: the explicit-ops fallback would pair query heads with KV heads
                // split differently, which the meta device cannot run (build() refuses unsupported nodes).
                ggml_tensor* fa = ggml_flash_attn_ext(ctx, src(0), src(1), src(2), src(3), fpar(0, 1.0f), fpar(1, 0.0f), fpar(2, 0.0f));
                ggml_flash_attn_ext_set_prec(fa, static_cast<ggml_prec>(ipar(0, 0)));
                if (src(4) != nullptr) ggml_flash_attn_ext_add_sinks(fa, src(4));
                return fa;
            }
            // srcs[4]: optional attention sinks. ip[1] > 0 bounds every mask row's finite entries (a selection's
            // width): decode and verify rows then attend only those (ggml's sparse path, a one-row kernel); a prompt
            // chunk over a wide window does on TensorSharp's kernels, else densely.
            std::int64_t width = ipar(1, 0);
            if (width > 0 && src(4) == nullptr && src(0)->ne[1] > 16)
            {
                if (ggml_tensor* sel = try_selected_attention(g, src(0), src(1), src(2), src(3), fpar(0, 1.0f), width))
                    return sel;
                width = 0;
            }
            ggml_tensor* fa = flash_attn_ext_guarded(ctx, "graph builder", src(0), src(1), src(2), src(3),
                                                     fpar(0, 1.0f), fpar(1, 0.0f), fpar(2, 0.0f), src(4),
                                                     static_cast<ggml_prec>(ipar(0, 0)));
            if (width > 0 && fa->op == GGML_OP_FLASH_ATTN_EXT)
                ggml_flash_attn_ext_set_n_kv_max(fa, static_cast<std::int32_t>(width));
            return fa;
        }
        case OP_SOFT_MAX_EXT: need(1); return ggml_soft_max_ext(ctx, src(0), src(1), fpar(0, 1.0f), fpar(1, 0.0f));
        case OP_ARGSORT: need(1); return ggml_argsort(ctx, src(0), ipar(0, 0) == 0 ? GGML_SORT_ORDER_ASC : GGML_SORT_ORDER_DESC);
        // Descending, as the IR defines it (ggml_top_k's are in no particular order): a route's position is its rank
        // to the host seams' route-tail skip, and argsort + view is the chain ggml-metal's top-k MoE fusion matches.
        case OP_TOP_K: need(1); return ggml_argsort_top_k(ctx, src(0), static_cast<int>(ipar(0, 1)));
        case OP_SUM_ROWS: need(1); return ggml_sum_rows(ctx, src(0));
        case OP_LIGHTNING_INDEXER: need(4); return ggml_lightning_indexer(ctx, src(0), src(1), src(2), src(3));
        case OP_TOP_K_MASK:
        {
            // The optional operands follow in order, each present when its flag is set.
            need(2);
            int at = 2;
            ggml_tensor* score = ipar(0, 0) != 0 ? src(at++) : nullptr;
            ggml_tensor* blocks = ipar(1, 0) != 0 ? src(at++) : nullptr;
            ggml_tensor* extra = ipar(2, 0) != 0 ? src(at++) : nullptr;
            if (at > ns) throw std::runtime_error("graph builder: top_k_mask is missing an optional operand");
            return ggml_top_k_mask(ctx, src(0), src(1), score, blocks, extra);
        }
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

    // Prompt-width device operations from this many rows (the same switch as Qwen4ExpModel's spans; the kernels'
    // own switches are TS_METAL_GDN_PROMPT and TS_METAL_HC_READ).
    int device_ops_min_rows()
    {
        static const int v = [] {
            const char* e = std::getenv("TS_METAL_PROMPT_OPS_MIN_ROWS");
            const int n = e != nullptr ? std::atoi(e) : 0;
            return n > 0 ? n : 64;
        }();
        return v;
    }

    // A computed result the IO buffer can hold in place of the allocator (not a view, not yet placed).
    bool io_bindable(const ggml_tensor* t, ggml_type type = GGML_TYPE_F32)
    {
        return t != nullptr && t->op != GGML_OP_NONE && t->view_src == nullptr && t->data == nullptr && t->buffer == nullptr
            && t->type == type && ggml_is_contiguous(t);
    }

    // A prompt chunk's flash attention under a selection's mask (every row at most `width` finite cells) over a wide
    // window, as one device operation: the mask compacted into each row's cell list (tsm_qsa_compact), then the
    // attention over those cells only (tsm_qsa_attn, Qwen4ExpModel's prompt kernel). Null where it does not apply.
    ggml_tensor* try_selected_attention(TsgGraph* g, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask,
                                        float scale, std::int64_t width)
    {
#if defined(TSG_GGML_USE_METAL)
        if (g->tp != nullptr || g_backend_type != BACKEND_TYPE_METAL || mask == nullptr || width <= 0) return nullptr;
        const std::int64_t D = q->ne[0], T = q->ne[1], heads = q->ne[2], cells = k->ne[1], kv_heads = k->ne[2];
        if (T < device_ops_min_rows() || q->type != GGML_TYPE_F32 || q->ne[3] != 1 || v->ne[0] != D || v->ne[1] != cells
            || v->ne[2] != kv_heads || mask->ne[0] != cells || mask->ne[1] != T || !io_bindable(mask, GGML_TYPE_F16))
            return nullptr;
        for (const ggml_tensor* c : { k, v })
            if (c->type != GGML_TYPE_F16 || c->nb[0] != 2 || c->nb[1] % 8 != 0 || c->nb[2] % 8 != 0) return nullptr;
        if (!tsm::qsa_attn_supported(graph_backend(g), static_cast<int>(D), static_cast<int>(heads), static_cast<int>(kv_heads),
                                     k->type, static_cast<int>(cells)))
            return nullptr;
        ggml_context* ctx = g->ctx;
        DeviceRegion r;
        r.kind = SELECTED_ATTENTION;
        r.in = { ggml_cont(ctx, q), k, v, mask };   // the queries as the kernel reads them, [D, T, H]
        r.results = { ggml_new_tensor_2d(ctx, GGML_TYPE_I32, width, T), ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T),
                      ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, heads, T) };   // the flash attention's [D, H, T]
        ggml_tensor *qc = r.in[0], *idx = r.results[0], *cnt = r.results[1], *out = r.results[2];
        const int t = static_cast<int>(T), n_cells = static_cast<int>(cells), w = static_cast<int>(width);
        const int n_head = static_cast<int>(heads), n_head_kv = static_cast<int>(kv_heads);
        tsg::HostMoeSegment seg;
        seg.layer = static_cast<int>(g->regions.size());
        seg.device_name = "selected attention";
        seg.seq_len = t;
        // After the queries, the mask and this layer's cache writes (effects, expanded before in program order).
        seg.device_inputs = { qc, mask, k, v };
        seg.device_outputs = { idx, cnt, out };
        seg.device_op = [qc, k, v, mask, idx, cnt, out, t, n_cells, w, n_head, n_head_kv, scale](ggml_backend_t backend) {
            return tsm::qsa_compact_encode(backend, mask, idx, cnt, t, n_cells, w)
                && tsm::qsa_attn_encode(backend, qc, k, v, idx, cnt, mask, out, t, n_head, n_head_kv, n_cells, w, scale);
        };
        g->regions.push_back(std::move(r));
        g->host_moe.push_back(std::move(seg));
        g->host_moe_holds.emplace_back(nullptr, nullptr);
        g->order.emplace_back(1, static_cast<int>(g->host_moe.size()) - 1);
        return out;
#else
        (void) g; (void) q; (void) k; (void) v; (void) mask; (void) scale; (void) width;
        return nullptr;
#endif
    }

    // Whether this graph's backend runs region `kind` over these operands as one device operation; if so, records
    // the region (its ops are then skipped until its end).
    bool try_serve_region(TsgGraph* g, int kind, const int* s, int ns, const std::int64_t* ip, int nip, const float* fp, int nfp)
    {
#if defined(TSG_GGML_USE_METAL)
        if (g->tp != nullptr || g_backend_type != BACKEND_TYPE_METAL) return false;
        std::vector<ggml_tensor*> in(static_cast<std::size_t>(ns));
        for (int i = 0; i < ns; i++) in[static_cast<std::size_t>(i)] = node_at(g, s[i]);
        if (kind == FUSED_DELTA_NET_PROMPT)
        {
            // qkv, z, alpha, beta, conv kernel, dt, a, norm weight, conv window, recurrent state; ip dk, kHeads, dv,
            // vHeads, gate (1 sigmoid), conv width; fp eps.
            if (ns != 10 || nip < 6 || nfp < 1) return false;
            const std::int64_t dk = ip[0], kh = ip[1], dv = ip[2], vh = ip[3], width = ip[5];
            const std::int64_t T = in[0]->ne[1], channels = in[0]->ne[0];
            if (ip[4] != 1 || T < device_ops_min_rows()) return false;   // the kernels gate with sigmoid(z) only
            if (!tsm::gdn_prompt_supported(graph_backend(g), static_cast<int>(dk), static_cast<int>(dv), static_cast<int>(kh),
                                           static_cast<int>(vh), static_cast<int>(width)))
                return false;
            for (int i = 0; i < 4; i++) if (!io_bindable(in[i]) || in[i]->ne[1] != T) return false;
            if (channels != 2 * kh * dk + vh * dv || in[1]->ne[0] != vh * dv || in[2]->ne[0] != vh || in[3]->ne[0] != vh)
                return false;
            for (int i = 4; i < 10; i++) if (in[i]->type != GGML_TYPE_F32 || !ggml_is_contiguous(in[i])) return false;
            if (in[4]->ne[0] != width || in[4]->ne[1] != channels || in[8]->ne[0] != width - 1 || in[8]->ne[1] != channels
                || ggml_nelements(in[9]) != dk * dv * vh || in[7]->ne[0] != dv)
                return false;
        }
        else if (kind == FUSED_HC_READ)
        {
            // the residual streams [hc * n, T], the norm's gammas [hc * n], the down, inject and up weights; ip n, hc,
            // low rank; fp eps.
            if (ns != 5 || nip < 3 || nfp < 1) return false;
            const std::int64_t n = ip[0], hc = ip[1], lr = ip[2];
            const std::int64_t T = in[0]->ne[1];
            if (T < device_ops_min_rows() || in[0]->type != GGML_TYPE_F32 || !ggml_is_contiguous(in[0]) || in[0]->ne[0] != n * hc
                || in[1]->type != GGML_TYPE_F32 || ggml_nelements(in[1]) != n * hc)
                return false;
            if (in[2]->ne[0] != n * hc || in[2]->ne[1] != lr || in[3]->ne[0] != n * hc || in[3]->ne[1] != hc
                || in[4]->ne[0] != lr || in[4]->ne[1] != n * hc)
                return false;
            if (!tsm::hc_read_supported(graph_backend(g), in[2]->type, in[3]->type, in[4]->type, static_cast<int>(n),
                                        static_cast<int>(hc), static_cast<int>(lr)))
                return false;
        }
        else
        {
            return false;
        }
        DeviceRegion r;
        r.kind = kind;
        r.in = std::move(in);
        r.ip.assign(ip, ip + nip);
        r.fp.assign(fp, fp + nfp);
        g->regions.push_back(std::move(r));
        return true;
#else
        (void) g; (void) kind; (void) s; (void) ns; (void) ip; (void) nip; (void) fp; (void) nfp;
        return false;
#endif
    }

    // The served region's end: its results are the device operation's outputs, which the next segment reads.
    void finish_region(TsgGraph* g, const int* results, int nresults)
    {
#if defined(TSG_GGML_USE_METAL)
        DeviceRegion& r = g->regions[static_cast<std::size_t>(g->serving)];
        const std::int64_t T = r.in[0]->ne[1];
        tsg::HostMoeSegment seg;
        seg.layer = static_cast<int>(g->regions.size()) - 1;
        seg.seq_len = static_cast<int>(T);
        if (r.kind == FUSED_DELTA_NET_PROMPT)
        {
            if (nresults != 1) throw std::runtime_error("graph builder: a delta-net prompt region has one result, the gated output");
            const std::int64_t dv = r.ip[2], vh = r.ip[3];
            r.results = { ggml_new_tensor_3d(g->ctx, GGML_TYPE_F32, dv, vh, T) };   // bound into the IO buffer at build
            tsm::GdnPromptIO io;
            io.qkv = r.in[0]; io.z = r.in[1]; io.alpha = r.in[2]; io.beta = r.in[3];
            io.conv_w = r.in[4]; io.dt = r.in[5]; io.a = r.in[6]; io.norm_w = r.in[7];
            io.conv_state = r.in[8]; io.ssm_state = r.in[9]; io.y = r.results[0];
            io.T = static_cast<int>(T); io.h_k = static_cast<int>(r.ip[1]); io.h_v = static_cast<int>(vh);
            io.channels = static_cast<int>(r.in[0]->ne[0]); io.eps = r.fp[0];
            seg.device_name = "gdn prompt";
            // Everything the kernels read or write: the projections end the segment before them; the state view and
            // the leaves join the graph so the allocator places them (leaves place no cut).
            seg.device_inputs = { r.in[0], r.in[1], r.in[2], r.in[3], r.in[9], r.in[8], r.in[4], r.in[5], r.in[6], r.in[7] };
            seg.device_op = [io](ggml_backend_t backend) { return tsm::gdn_prompt_encode(backend, io); };
        }
        else
        {
            if (nresults != 2) throw std::runtime_error("graph builder: a hyper-connection read has two results, the mix and the inject");
            const std::int64_t n = r.ip[0], hc = r.ip[1];
            r.results = { ggml_new_tensor_2d(g->ctx, GGML_TYPE_F32, n, T), ggml_new_tensor_2d(g->ctx, GGML_TYPE_F32, hc, T) };
            tsm::HcReadIO io;
            io.res = r.in[0]; io.gamma = r.in[1]; io.w_down = r.in[2]; io.w_inject = r.in[3]; io.w_up = r.in[4];
            io.mixed = r.results[0]; io.inject = r.results[1];
            io.T = static_cast<int>(T); io.n_embd = static_cast<int>(n); io.hc = static_cast<int>(hc);
            io.lr = static_cast<int>(r.ip[2]); io.eps = r.fp[0];
            seg.device_name = "hc read";
            // The residual is read only: its last graph reader (the block's scatter) comes after the cut, so the
            // allocator keeps it.
            seg.device_inputs = { r.in[0], r.in[1], r.in[2], r.in[3], r.in[4] };
            seg.device_op = [io](ggml_backend_t backend) { return tsm::hc_read_encode(backend, io); };
        }
        for (int i = 0; i < nresults; i++)
        {
            if (results[i] < 0 || results[i] >= static_cast<int>(g->nodes.size()))
                throw std::runtime_error("graph builder: a fused region's result is not a node");
            g->nodes[static_cast<std::size_t>(results[i])] = r.results[static_cast<std::size_t>(i)];
            g->skipped[static_cast<std::size_t>(results[i])] = 0;
        }
        seg.device_outputs = r.results;
        g->host_moe.push_back(std::move(seg));
        g->host_moe_holds.emplace_back(nullptr, nullptr);
        g->order.emplace_back(1, static_cast<int>(g->host_moe.size()) - 1);
#else
        (void) g; (void) results; (void) nresults;
#endif
    }

    // The served regions' operands into the IO buffer: one region per role, which every region of the kind reuses.
    void bind_regions(TsgGraph* g)
    {
        if (g->regions.empty()) return;
        auto aligned = [](std::size_t b) { return (b + 255) & ~static_cast<std::size_t>(255); };
        std::size_t size[IO_ROLES] = {};
        for (const DeviceRegion& r : g->regions)
            for (const auto& [role, t] : io_operands(r))
                size[role] = std::max(size[role], aligned(ggml_nbytes(t)));
        std::size_t off[IO_ROLES] = {}, bytes = 0;
        for (int role = 0; role < IO_ROLES; role++)
        {
            off[role] = bytes;
            bytes += size[role];
        }
        g->io = device_io(graph_backend(g), bytes);
        auto* base = static_cast<char*>(ggml_backend_buffer_get_base(g->io->buf));
        for (const DeviceRegion& r : g->regions)
            for (const auto& [role, t] : io_operands(r))
                if (ggml_backend_tensor_alloc(g->io->buf, t, base + off[role]) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("graph builder: binding a device operation's operand failed");
    }

    // Bind the leaves, then let the graph's own allocator place every other tensor.
    bool build(TsgGraph* g)
    {
        g->graph = ggml_new_graph_custom(g->ctx, graph_slots(g->max_nodes), false);
        if (g->graph == nullptr) { set_last_error("graph builder: graph allocation failed."); return false; }
        if (g->serving >= 0 || !g->open_regions.empty()) { set_last_error("graph builder: a fused region is still open."); return false; }
        bind_regions(g);
        // Effects and host-expert pauses in program order: a pause's boundary tensors come before any node that
        // reads its result (moe_out has no producer, so nothing else orders them).
        for (const auto& [kind, index] : g->order)
        {
            if (kind == 0)
            {
                ggml_build_forward_expand(g->graph, g->effects[static_cast<std::size_t>(index)]);
                continue;
            }
            const tsg::HostMoeSegment& hm = g->host_moe[static_cast<std::size_t>(index)];
            if (hm.device_op)
            {
                for (ggml_tensor* t : hm.device_inputs) ggml_build_forward_expand(g->graph, t);
                continue;
            }
            const auto& [keep_logits, hold_ids] = g->host_moe_holds[static_cast<std::size_t>(index)];
            ggml_build_forward_expand(g->graph, hm.moe_in);
            // The route weights before the ids: their parents then come in llama.cpp's order (softmax, reshape,
            // argsort, view, get_rows, ...), the node sequence ggml-metal's top-k MoE fusion matches.
            ggml_build_forward_expand(g->graph, hm.weights);
            if (keep_logits != nullptr) ggml_build_forward_expand(g->graph, keep_logits);
            ggml_build_forward_expand(g->graph, hm.sel_ids);
            if (hold_ids != nullptr) ggml_build_forward_expand(g->graph, hold_ids);
            if (hm.pred_next != nullptr) ggml_build_forward_expand(g->graph, hm.pred_next);   // read at this pause
        }
        for (const Output& o : g->outputs) ggml_build_forward_expand(g->graph, o.tensor);
        // Before allocation: reordering after lifetime planning would invalidate the alias plan (and a
        // reorder would move nodes across a pause).
        if (g->reorder && g->host_moe.empty()) optimize_graph_for_metal(g->ctx, g->graph);
        // Otherwise ggml-metal still fuses the graph's chains when it encodes them, and runs independent neighbours
        // concurrently: without the fusions' allocation dependencies the allocator hands a fused kernel's input to a
        // concurrent node (Qwen3.6-35B-A3B decoded two different texts in alternate runs on an M5 Pro).
        else if (g_backend_type == BACKEND_TYPE_METAL) tsg_add_metal_fusion_alloc_dependencies(g->ctx, g->graph);

        if (g->tp != nullptr)
        {
            // A node the ranks cannot run would fall back to the CPU and come back mirrored, which
            // aborts the meta device mid-graph: refuse the graph instead (Muse Spark's tp_check_graph).
            for (int i = 0; i < ggml_graph_n_nodes(g->graph); i++)
            {
                ggml_tensor* node = ggml_graph_node(g->graph, i);
                if (node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE
                    || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) continue;
                if (ggml_nelements(node) == 0 || node->op == GGML_OP_ARANGE || !ggml_backend_supports_op(g->tp->backend, node))
                {
                    char why[256];
                    std::snprintf(why, sizeof(why), "graph builder: %s [%lld, %lld, %lld, %lld] cannot run on the tensor-parallel devices",
                                  ggml_op_desc(node), (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2], (long long) node->ne[3]);
                    throw std::runtime_error(why);
                }
            }
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(graph_backend(g));
        for (Leaf& leaf : g->leaves)
        {
            leaf.uploaded = !leaf.host_only;
            if (leaf.role == LEAF_INPUT || leaf.data == nullptr || leaf.host_only) continue;
            if (g->tp != nullptr)
            {
                // Weights and states are resident in meta buffers (uploaded once, spread by their split).
                MetaResident* r = meta_resident(g->tp, leaf);
                ggml_set_name(leaf.tensor, r->t->name);
                if (ggml_backend_tensor_alloc(r->buf, leaf.tensor, r->t->data) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("graph builder: binding a tensor-parallel leaf failed");
                leaf.uploaded = false;
                continue;
            }
            const auto usage = leaf.role == LEAF_STATE ? GGML_BACKEND_BUFFER_USAGE_COMPUTE : GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
            // Weights live as long as the model, so even a small one (a norm, a per-head gate
            // constant) keeps a device copy: in compute memory it would be uploaded every run, a
            // synchronous copy per tensor on CUDA (~180 per Qwen3.8-27B decode step).
            if (leaf.bytes >= 4096 || leaf.role == LEAF_WEIGHT)
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
                if (leaf.bytes >= 4096 && try_get_host_ptr_buffer(g_backend, dev, leaf.data, leaf.bytes, true, buf)
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

        // Place aliased op results inside their (persistent) base buffers, so the op writes the
        // state in place. A base living in compute memory cannot host it: run the fallback copy.
        bool fallbacks = false;
        for (const auto& a : g->aliases)
        {
            const bool resident = a.base->buffer != nullptr && a.base->data != nullptr && a.node->data == nullptr
                && a.offset + ggml_nbytes(a.node) <= ggml_nbytes(a.base)
                && ggml_backend_tensor_alloc(a.base->buffer, a.node, static_cast<char*>(a.base->data) + a.offset) == GGML_STATUS_SUCCESS;
            if (!resident)
            {
                ggml_build_forward_expand(g->graph, a.fallback);
                fallbacks = true;
            }
        }
        (void) fallbacks;
        if (!g->host_moe.empty() && !tsg::host_moe_build_segment_ends(g->graph, g->host_moe, g->seg_end, "graph builder"))
            return false;

        // A one-shot graph borrows the backend's persistent, reused compute allocator (the
        // memory the fused graphs share, grown on demand); a graph that will be replayed keeps
        // its own, so no other graph can reuse its intermediates between runs.
        if (g->tp == nullptr && !g->replayable && alloc_graph_reuse_gallocr(g->graph))
        {
            g->built = true;
            return true;
        }
        g->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(graph_backend(g)));
        if (g->galloc == nullptr || !ggml_gallocr_alloc_graph(g->galloc, g->graph))
        {
            set_last_error("graph builder: failed to allocate the graph's compute memory.");
            return false;
        }
        // A graph that replays keeps every node and binding for its lifetime: a stable non-zero id
        // lets ggml-cuda reuse its captured CUDA graph without re-comparing every node per run (it
        // treats id 0 as unknown). One-shot graphs keep 0.
        static std::atomic<std::uint64_t> next_uid{UINT64_C(0x5453475200000000)};   // "TSGR"
        g->graph->uid = next_uid.fetch_add(1, std::memory_order_relaxed);
        g->replayable = true;
        g->built = true;
        return true;
    }
}

// A graph over `nranks` ranks with tensor parallelism (ggml's meta device): leaves given a split
// (TSGgml_GraphLeafSplit) spread over the ranks, the rest are mirrored.
TSG_EXPORT void* TSGgml_GraphCreateTp(int max_nodes, int replayable, const int* ranks, int nranks)
{
    try
    {
        if (ranks == nullptr || nranks < 2) throw std::runtime_error("graph builder: tensor parallelism needs 2 or more ranks");
        TpGroup* grp = tp_group(ranks, nranks);
        if (max_nodes < 16) max_nodes = 16;
        auto* g = new TsgGraph();
        g->tp = grp;
        g->rank = ranks[0];
        g->max_nodes = max_nodes;
        g->replayable = replayable != 0;
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * static_cast<std::size_t>(max_nodes) * 2
                          + ggml_graph_overhead_custom(graph_slots(max_nodes), false);
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

// How leaf `node` spreads over a tensor-parallel graph's ranks: split along `axis` (0-3) with
// `ne` = per (segment, rank) extents, segment s repeating nr[s] times (ggml_backend_meta_split_state).
TSG_EXPORT int TSGgml_GraphLeafSplit(void* handle, int node, int axis, const std::int64_t* ne, int nne,
                                     const std::uint32_t* nr, int nsegments)
{
    auto* g = static_cast<TsgGraph*>(handle);
    try
    {
        if (g == nullptr || g->tp == nullptr || g->built) throw std::runtime_error("graph builder: a split needs an unbuilt tensor-parallel graph");
        if (axis < 0 || axis >= GGML_MAX_DIMS || nsegments < 1 || nsegments > 16 || nne != nsegments * static_cast<int>(g->tp->ranks.size()))
            throw std::runtime_error("graph builder: bad leaf split");
        ggml_tensor* t = node_at(g, node);
        for (Leaf& leaf : g->leaves)
        {
            if (leaf.tensor != t) continue;
            std::memset(&leaf.split, 0, sizeof(leaf.split));
            leaf.split.axis = static_cast<ggml_backend_meta_split_axis>(axis);
            for (int i = 0; i < nne; i++) leaf.split.ne[i] = ne[i];
            for (int s2 = 0; s2 < nsegments; s2++) leaf.split.nr[s2] = nr != nullptr ? nr[s2] : 1;
            leaf.split.n_segments = static_cast<std::uint32_t>(nsegments);
            leaf.has_split = true;
            return 1;
        }
        throw std::runtime_error("graph builder: a split names a node that is not a leaf");
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
}

TSG_EXPORT void* TSGgml_GraphCreate(int max_nodes, int replayable, int reorder)
{
    try
    {
        if (!ensure_backend()) return nullptr;
        if (max_nodes < 16) max_nodes = 16;
        auto* g = new TsgGraph();
        g->rank = g_active_rank;
        g->max_nodes = max_nodes;
        g->replayable = replayable != 0;
        g->reorder = reorder != 0;
        ggml_init_params params{};
        params.mem_size = ggml_tensor_overhead() * static_cast<std::size_t>(max_nodes) * 2
                          + ggml_graph_overhead_custom(graph_slots(max_nodes), false);
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
        if (g->serving >= 0) return push_skipped(g);   // a device operation computes the region
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
        if (g->serving >= 0) return 1;   // the region's state writes are the device operation's
        g->effects.push_back(node_at(g, node));
        g->order.emplace_back(0, static_cast<int>(g->effects.size()) - 1);
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
        // Leaves living in compute memory are uploaded every run, weights included: the allocator
        // may reuse an input's memory after its last use within the graph. A leaf nothing in the
        // graph reads (the trace may emit more than the outputs and effects need) has no memory.
        for (const Leaf& leaf : g->leaves)
            if (leaf.uploaded && leaf.data != nullptr && leaf.tensor->buffer != nullptr)
                ggml_backend_tensor_set(leaf.tensor, resolve_upload_source(leaf.data), 0, leaf.bytes);
        for (int i = 0; i < ninputs; i++)
        {
            ggml_tensor* t = node_at(g, inputs[i]);
            const std::size_t n = std::min(static_cast<std::size_t>(input_bytes[i]), ggml_nbytes(t));
            if (input_data[i] != nullptr && n > 0 && t->buffer != nullptr)
                ggml_backend_tensor_set(t, input_data[i], 0, n);
        }

        // Enqueue the graph and the downloads back to back, then synchronize once (as llama.cpp's
        // scheduler does): a synchronous compute followed by a download is a second command-buffer
        // round trip per step. The run is complete when this returns.
        ggml_status status = GGML_STATUS_SUCCESS;
        if (!g->host_moe.empty())
        {
            // Accelerator segments and host-expert pauses in turn (the outputs are read below).
            if (!tsg::host_moe_execute_segments(g->graph, g->host_moe, g->seg_end, "graph builder", graph_backend(g)))
            {
                sync_backend(graph_backend(g));
                return 0;
            }
        }
        else if (graph_node_profile_enabled())
        {
            status = graph_compute_profiled(graph_backend(g), g->graph, g->replayable ? "graph builder decode" : "graph builder prefill");
        }
        else
        {
            const std::uint64_t errors_before = g_ggml_error_count.load(std::memory_order_acquire);
            status = ggml_backend_graph_compute_async(graph_backend(g), g->graph);
            if (g_ggml_error_count.load(std::memory_order_acquire) != errors_before)
                g_backend_compute_failed.store(true, std::memory_order_release);
        }
        if (status != GGML_STATUS_SUCCESS)
        {
            sync_backend(graph_backend(g));
            set_last_error("graph builder: backend graph execution failed.");
            return 0;
        }
        for (const Output& o : g->outputs)
            if (o.data != nullptr)
                ggml_backend_tensor_get_async(graph_backend(g), o.tensor, o.data, 0, o.bytes);
        for (const Leaf& leaf : g->leaves)
            if (leaf.role == LEAF_STATE && leaf.uploaded && leaf.data != nullptr && leaf.tensor->buffer != nullptr)
                ggml_backend_tensor_get_async(graph_backend(g), leaf.tensor, leaf.data, 0, leaf.bytes);
        sync_backend(graph_backend(g));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
}

// Drops the kept device-operation IO buffers (graphs still holding one keep it until they are freed): before the
// backends are torn down (TSGgml_Shutdown), whose device asserts at exit that no buffer is left.
extern "C" void TSGgml_GraphReleaseDeviceIo()
{
    std::lock_guard<std::mutex> lk(g_device_io_mu);
    g_device_io.clear();
}

// A fused region of GraphFusion `kind` over `srcs` begins: 1 when a device operation computes it (its ops, effects and
// aliases up to the matching TSGgml_GraphFusedEnd are then numbered but not built), 0 when its ops are built. A region
// nested in a served one is part of it.
TSG_EXPORT int TSGgml_GraphFusedBegin(void* handle, int kind, const int* srcs, int nsrc,
                                      const std::int64_t* ip, int nip, const float* fp, int nfp)
{
    auto* g = static_cast<TsgGraph*>(handle);
    try
    {
        if (g == nullptr || g->built) throw std::runtime_error("graph builder: no graph, or graph already built");
        if (g->serving >= 0)
        {
            g->open_regions.push_back(-1);
            return 0;
        }
        const bool served = try_serve_region(g, kind, srcs, nsrc, ip, nip, fp, nfp);
        g->open_regions.push_back(served ? static_cast<int>(g->regions.size()) - 1 : -1);
        if (served) g->serving = g->open_regions.back();
        return served ? 1 : 0;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return -1;
    }
}

// The innermost open region ends; `results` are its result nodes in the kind's order.
TSG_EXPORT int TSGgml_GraphFusedEnd(void* handle, const int* results, int nresults)
{
    auto* g = static_cast<TsgGraph*>(handle);
    try
    {
        if (g == nullptr || g->built || g->open_regions.empty()) throw std::runtime_error("graph builder: no open fused region");
        const int region = g->open_regions.back();
        g->open_regions.pop_back();
        if (region >= 0 && region == g->serving)
        {
            finish_region(g, results, nresults);
            g->serving = -1;
        }
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

// Make `node`'s result live in `base`'s buffer at `offset` bytes (an in-place state update).
// When `base` is not resident in a persistent buffer, `fallback` (a node that copies the
// state out of `node`) runs instead.
TSG_EXPORT int TSGgml_GraphAlias(void* handle, int node, int base, std::int64_t offset, int fallback)
{
    auto* g = static_cast<TsgGraph*>(handle);
    try
    {
        if (g == nullptr || g->built) throw std::runtime_error("graph builder: no graph, or graph already built");
        if (g->serving >= 0) return 1;
        g->aliases.push_back({ node_at(g, node), node_at(g, base), static_cast<std::size_t>(offset), node_at(g, fallback) });
        return 1;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
}

namespace tsg_graphtp
{
    // The host bytes behind `ptr` changed (or are being freed): drop its meta residents (graphs
    // bound to them are dropped by the managed side first).
    void on_drop(void* ptr)
    {
        std::lock_guard<std::mutex> lk(g_resident_mu);
        for (auto it = g_residents.begin(); it != g_residents.end();)
            it = it->first.second == ptr ? g_residents.erase(it) : std::next(it);
    }

    // Gather a meta-resident state back into its host bytes; false when `ptr` has none.
    bool sync_to_host(void* ptr, std::size_t size)
    {
        std::lock_guard<std::mutex> lk(g_resident_mu);
        bool any = false;
        for (auto& [key, r] : g_residents)
        {
            if (key.second != ptr) continue;
            ggml_backend_synchronize(key.first->backend);
            ggml_backend_tensor_get(r->t, ptr, 0, std::min(size, r->bytes));
            any = true;
        }
        return any;
    }
}
