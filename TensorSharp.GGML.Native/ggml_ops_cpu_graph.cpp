// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// See ggml_ops_cpu_graph.h. Upstream ggml is used unchanged: the repacked layout is ggml-cpu's own CPU_REPACK buffer
// type (found through the CPU registry's "ggml_backend_dev_get_extra_bufts"), whose buffers repack a tensor's bytes when
// they are set and whose kernels ggml-cpu runs for any MUL_MAT whose weight lives in such a buffer.

#include "ggml_ops_cpu_graph.h"
#include "ggml_ops_internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace tsg
{
    struct CpuRepacked
    {
        ggml_backend_buffer_t buf = nullptr;
        ggml_type type = GGML_TYPE_COUNT;
        std::int64_t ne0 = 0, ne1 = 0;
        bool ready = false;   // the bytes are repacked (else the graph that created it repacks them)
        ~CpuRepacked()
        {
            if (buf != nullptr) ggml_backend_buffer_free(buf);
        }
    };

    namespace
    {
        // ggml-cpu's CPU_REPACK buffer type, or null when this build has none (GGML_CPU_REPACK off).
        ggml_backend_buffer_type_t repack_buffer_type()
        {
            static ggml_backend_buffer_type_t buft = [] {
                ggml_backend_reg_t reg = ggml_backend_cpu_reg();
                if (reg == nullptr || ggml_backend_reg_dev_count(reg) == 0) return static_cast<ggml_backend_buffer_type_t>(nullptr);
                auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts"));
                if (get_extra == nullptr) return static_cast<ggml_backend_buffer_type_t>(nullptr);
                for (ggml_backend_buffer_type_t* b = get_extra(ggml_backend_reg_dev_get(reg, 0)); b != nullptr && *b != nullptr; ++b)
                    if (std::strcmp(ggml_backend_buft_name(*b), "CPU_REPACK") == 0) return *b;
                return static_cast<ggml_backend_buffer_type_t>(nullptr);
            }();
            return buft;
        }

        // A zero-size buffer of the repack type: placed on a tensor for a moment, it lets ggml-cpu's own supports_op say
        // whether its repacked kernels take a product (they need a repack layout for the weight's type and rows on this
        // CPU, and F32 activations).
        ggml_backend_buffer_t repack_probe()
        {
            static ggml_backend_buffer_t probe = ggml_backend_buft_alloc_buffer(repack_buffer_type(), 0);
            return probe;
        }

        // The types ggml-cpu repacks (repack.cpp ggml_repack_get_optimal_repack_type), less the ones TensorSharp reads
        // otherwise: Q1_0 (Bonsai's PRISM products, ggml_ops_bonsai.cpp) and Q2_K (AVX-512 only, never measured).
        bool repackable_type(ggml_type t)
        {
            switch (t)
            {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_MXFP4:
                return true;
            default:
                return false;
            }
        }

        bool supported_with_probe(ggml_backend_dev_t dev, ggml_tensor* op, ggml_tensor* weight)
        {
            ggml_backend_buffer_t saved = weight->buffer;
            weight->buffer = repack_probe();
            const bool ok = ggml_backend_dev_supports_op(dev, op);
            weight->buffer = saved;
            return ok;
        }

        // The leaf itself has a repack layout here (its type and rows): a product of it over one F32 row, asked of
        // ggml-cpu with the leaf in the repack buffer type (remembered per type and shape; a prompt chunk's graph is
        // built once per chunk).
        bool leaf_repacks(ggml_backend_dev_t dev, const ggml_tensor* leaf)
        {
            static std::mutex mu;
            static std::map<std::tuple<int, std::int64_t, std::int64_t>, bool> known;
            const auto key = std::make_tuple(static_cast<int>(leaf->type), leaf->ne[0], leaf->ne[1]);
            {
                std::lock_guard<std::mutex> lock(mu);
                if (auto it = known.find(key); it != known.end()) return it->second;
            }
            ggml_init_params params{ ggml_tensor_overhead() * 4, nullptr, true };
            ggml_context* ctx = ggml_init(params);
            if (ctx == nullptr) return false;
            ggml_tensor* w = ggml_new_tensor_2d(ctx, leaf->type, leaf->ne[0], leaf->ne[1]);
            ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, leaf->ne[0], 1);
            ggml_tensor* y = ggml_mul_mat(ctx, w, x);
            const bool ok = supported_with_probe(dev, y, w);
            ggml_free(ctx);
            std::lock_guard<std::mutex> lock(mu);
            known[key] = ok;
            return ok;
        }

        // A product ggml-cpu's repacked kernels compute (repack.cpp forward_mul_mat): the weight (or a row block of it)
        // first, F32 activation rows of at most three dimensions, an F32 result.
        bool product_takes_repacked(ggml_backend_dev_t dev, ggml_tensor* node)
        {
            const ggml_tensor* x = node->src[1];
            if (node->op != GGML_OP_MUL_MAT || x == nullptr || node->type != GGML_TYPE_F32) return false;
            if (x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float) || x->ne[3] != 1 || node->ne[3] != 1) return false;
            return supported_with_probe(dev, node, node->src[0]);
        }

        // A view of whole 8-row groups of the weight (GraphStep.MatMulRows): in every repack layout a group of up to 8
        // rows occupies the bytes of its plain rows, so the view addresses repacked groups.
        bool row_block_view(const ggml_tensor* v, const ggml_tensor* w)
        {
            return v->op == GGML_OP_VIEW && v->view_src == w && v->src[0] == w && v->type == w->type
                && v->ne[0] == w->ne[0] && v->nb[1] == w->nb[1] && v->ne[2] == 1 && v->ne[3] == 1
                && v->ne[1] % 8 == 0 && v->view_offs % (8 * w->nb[1]) == 0;
        }

        std::mutex g_repack_mu;   // the cache, and making copies (one graph at a time)
        std::unordered_map<const void*, CpuRepackedRef> g_repacked;

        bool debug_log()
        {
            static const bool v = [] {
                const char* e = std::getenv("TS_GGML_LOG_DEBUG");
                return e != nullptr && (std::strcmp(e, "1") == 0 || std::strcmp(e, "true") == 0);
            }();
            return v;
        }
    }

    bool cpu_repack_enabled()
    {
        static const bool on = [] {
#if defined(__aarch64__) || defined(_M_ARM64)
            bool v = true;
#else
            bool v = false;
#endif
            if (const char* e = std::getenv("TS_GGML_CPU_REPACK"); e != nullptr && *e != '\0')
                v = std::strcmp(e, "0") != 0 && std::strcmp(e, "false") != 0 && std::strcmp(e, "off") != 0;
            if (v && (repack_buffer_type() == nullptr || repack_probe() == nullptr))
            {
                std::fprintf(stderr,
                    "[ggml-cpu] this ggml-cpu build has no CPU_REPACK buffer type (GGML_CPU_REPACK is off): quantized "
                    "weights run on the plain dot kernels, about half the prompt speed of the repacked ones\n");
                std::fflush(stderr);
                v = false;
            }
            return v;
        }();
        return on;
    }

    std::vector<CpuRepackedRef> cpu_repack_bind(ggml_cgraph* graph, const std::vector<CpuWeightLeaf>& weights,
                                                std::vector<char>& bound)
    {
        bound.assign(weights.size(), 0);
        std::vector<CpuRepackedRef> held;
        if (weights.empty() || !cpu_repack_enabled()) return held;
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(repack_buffer_type());

        // Candidates: 2-D weights of a repackable type with a repack layout here.
        std::vector<char> ok(weights.size(), 0), used(weights.size(), 0);
        std::unordered_map<const ggml_tensor*, std::size_t> index;
        for (std::size_t i = 0; i < weights.size(); i++)
        {
            const ggml_tensor* t = weights[i].tensor;
            if (!repackable_type(t->type) || t->ne[2] != 1 || t->ne[3] != 1 || !ggml_is_contiguous(t)
                || t->buffer != nullptr || t->data != nullptr || !leaf_repacks(dev, t))
                continue;
            ok[i] = 1;
            index[t] = i;
        }
        if (index.empty()) return held;

        // Every reader of a candidate must be a product the repacked kernels compute (directly, or through a row-block
        // view): a GET_ROWS, a copy or any other read would see interleaved bytes.
        std::unordered_map<const ggml_tensor*, std::size_t> view_of;
        const int n = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n; i++)
        {
            ggml_tensor* node = ggml_graph_node(graph, i);
            for (int j = 0; j < GGML_MAX_SRC; j++)
            {
                const ggml_tensor* s = node->src[j];
                if (s == nullptr) continue;
                std::size_t w;
                bool through_view = false;
                if (auto it = index.find(s); it != index.end()) w = it->second;
                else if (auto v = view_of.find(s); v != view_of.end()) { w = v->second; through_view = true; }
                else continue;
                if (!ok[w]) continue;
                if (!through_view && j == 0 && row_block_view(node, s))
                {
                    view_of[node] = w;
                    continue;
                }
                if (j == 0 && product_takes_repacked(dev, node))
                {
                    used[w] = 1;
                    continue;
                }
                ok[w] = 0;
            }
        }

        std::lock_guard<std::mutex> lock(g_repack_mu);
        std::vector<std::size_t> fresh;            // weights whose copy this call makes (one leaf per copy)
        std::unordered_set<const CpuRepacked*> made_here;
        for (std::size_t i = 0; i < weights.size(); i++)
        {
            if (!ok[i] || !used[i]) continue;
            ggml_tensor* t = weights[i].tensor;
            CpuRepackedRef& e = g_repacked[weights[i].key];
            if (e == nullptr || e->type != t->type || e->ne0 != t->ne[0] || e->ne1 != t->ne[1]
                || (!e->ready && made_here.count(e.get()) == 0))
            {
                auto made = std::make_shared<CpuRepacked>();
                made->type = t->type;
                made->ne0 = t->ne[0];
                made->ne1 = t->ne[1];
                made->buf = ggml_backend_buft_alloc_buffer(repack_buffer_type(), ggml_nbytes(t));
                if (made->buf == nullptr)
                {
                    g_repacked.erase(weights[i].key);
                    continue;   // out of memory for the copy: this weight stays on the plain kernels
                }
                ggml_backend_buffer_set_usage(made->buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                e = made;
                made_here.insert(made.get());
                fresh.push_back(i);
            }
            // Binding runs the buffer's init_tensor, which gives the leaf its repack traits (tensor->extra).
            if (ggml_backend_tensor_alloc(e->buf, t, ggml_backend_buffer_get_base(e->buf)) != GGML_STATUS_SUCCESS)
            {
                g_repacked.erase(weights[i].key);
                continue;
            }
            bound[i] = 1;
            held.push_back(e);
        }

        if (!fresh.empty())
        {
            // Repack the new copies on several threads (each repack reads its weight's bytes once): a model's first
            // prompt would otherwise pay the whole model's repack on one core.
            const auto t0 = std::chrono::steady_clock::now();
            std::atomic<std::size_t> next{0};
            std::size_t total = 0;
            for (std::size_t i : fresh) total += ggml_nbytes(weights[i].tensor);
            auto work = [&] {
                for (std::size_t k = next.fetch_add(1); k < fresh.size(); k = next.fetch_add(1))
                {
                    const CpuWeightLeaf& w = weights[fresh[k]];
                    ggml_backend_tensor_set(w.tensor, w.bytes, 0, ggml_nbytes(w.tensor));
                }
            };
            const std::size_t nthreads = std::min<std::size_t>(fresh.size(),
                std::max<std::size_t>(1, static_cast<std::size_t>(available_cpu_parallelism())));
            std::vector<std::thread> pool;
            for (std::size_t k = 1; k < nthreads; k++) pool.emplace_back(work);
            work();
            for (std::thread& th : pool) th.join();
            for (std::size_t i : fresh) g_repacked[weights[i].key]->ready = true;
            if (debug_log())
            {
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr, "[ggml-cpu] repacked %zu weights (%.1f MiB) into CPU_REPACK in %.1f ms on %zu threads\n",
                    fresh.size(), total / (1024.0 * 1024.0), ms, nthreads);
                std::fflush(stderr);
            }
        }
        return held;
    }

    void cpu_repack_forget(const void* key)
    {
        if (key == nullptr) return;
        std::lock_guard<std::mutex> lock(g_repack_mu);
        if (!g_repacked.empty()) g_repacked.erase(key);
    }

    void cpu_repack_clear()
    {
        std::lock_guard<std::mutex> lock(g_repack_mu);
        g_repacked.clear();
    }
}
