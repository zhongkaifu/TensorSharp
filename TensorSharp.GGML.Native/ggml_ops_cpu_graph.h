// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The ggml-cpu backend's side of the generic graph (ggml_ops_graph_builder.cpp), through public ggml APIs only:
//   - weights the graph reads only as the first operand of products are placed in ggml-cpu's CPU_REPACK buffer type
//     (its interleaved GEMM/GEMV layouts, which llama.cpp loads its weights into), and
//   - a step runs on the thread pool its width calls for: decode steps on fewer threads than prompt steps.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace tsg
{
    /// A weight's copy in ggml-cpu's CPU_REPACK buffer, shared by every graph bound to it (a graph keeps the copies it
    /// reads alive after the cache lets them go).
    struct CpuRepacked;
    using CpuRepackedRef = std::shared_ptr<CpuRepacked>;

    struct CpuWeightLeaf
    {
        ggml_tensor* tensor;   // the graph's leaf, not yet bound
        const void* key;       // the host pointer the weight is cached by
        const void* bytes;     // where its bytes are read from (the key, unless a preload redirected it)
    };

    /// Whether weights are repacked on this build and machine: TS_GGML_CPU_REPACK (0 off, 1 on), by default on ARM64,
    /// where ggml-cpu's repacked i8mm/dotprod kernels were measured (aarch64 Apple M5 Pro).
    bool cpu_repack_enabled();

    /// Binds every weight of `weights` that the graph reads only as the first operand of products ggml-cpu's repacked
    /// kernels take (MUL_MAT over F32 rows, directly or through row-block views of whole 8-row groups) to its repacked
    /// copy, made on first use (in parallel) and cached by host pointer. Sets bound[i] for each weight it bound and
    /// returns the copies the graph must hold. Any other weight is left for the caller to bind as before.
    std::vector<CpuRepackedRef> cpu_repack_bind(ggml_cgraph* graph, const std::vector<CpuWeightLeaf>& weights,
                                                std::vector<char>& bound);

    /// The host bytes behind `key` changed (or are going away): the next graph that binds them repacks them again.
    void cpu_repack_forget(const void* key);

    /// Drops every cached copy (graphs still holding one keep it until they are freed).
    void cpu_repack_clear();

    /// w · x for a weight ggml-cpu has no fast kernel for on this CPU, as TensorSharp's product (a GGML_OP_CUSTOM node
    /// over the graph's thread pool, ggml_ops_cpu_products.cpp), or null to keep ggml's MUL_MAT. ARM64: BF16 weights
    /// (ggml-cpu's BF16 dot is scalar there) at any width, through Accelerate's sgemm at prompt width on Apple, over F32
    /// activations unrounded; NVFP4 weights from two activation rows, as a GEMM over ggml's own Q8_0 activation rows that
    /// gives every row the bits ggml's product gives it. w a 2-D weight or a row block of one. The graph builder's weight
    /// products and the per-op ones (tsg::weight_mul_mat) both take it, so the two paths run one product per weight.
    ggml_tensor* cpu_weight_product(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x);

    /// The activation rows `node` reads when it is the product node of a cpu_weight_product (0 for any other node), so a
    /// per-op graph's pool is chosen from it as from a MUL_MAT (cpu_pools_select_for).
    std::int64_t cpu_weight_product_rows(const ggml_tensor* node);

    /// Creates the CPU backend's thread pools (the prompt pool always; a decode pool when its width differs) and sets
    /// the prompt pool; the pools are appended to `owned` (freed at shutdown).
    void cpu_pools_create(ggml_backend_t backend, std::vector<ggml_threadpool_t>& owned);

    /// Runs the next graph on the pool for its width (cpu_pool_for_step): `product_rows` is the most activation rows any
    /// weight product of the graph reads (0 when it has none). The graph runtime (TSGgml_GraphRun) selects from its weight
    /// products; the ggml computes outside it that go through tsg::compute_graph select from their products'
    /// (cpu_pools_select_for).
    void cpu_pools_select(ggml_backend_t backend, std::int64_t product_rows);

    /// A compute outside the graph runtime (a per-op graph through tsg::compute_graph): the pool for the most activation
    /// rows any of its products (MUL_MAT, MUL_MAT_ID, cpu_weight_product's) reads, as cpu_pools_select; a graph without
    /// products keeps the pool it finds, so the small per-op graphs between decode steps never pause the decode pool. A
    /// no-op inside the graph runtime's own computes (CpuGraphRuntimeScope), which selected their pool from their weight
    /// products.
    void cpu_pools_select_for(ggml_backend_t backend, const ggml_cgraph* graph);

    /// Marks this thread's computes as the graph runtime's for its lifetime (see cpu_pools_select_for).
    class CpuGraphRuntimeScope
    {
    public:
        CpuGraphRuntimeScope();
        ~CpuGraphRuntimeScope();
        CpuGraphRuntimeScope(const CpuGraphRuntimeScope&) = delete;
        CpuGraphRuntimeScope& operator=(const CpuGraphRuntimeScope&) = delete;

    private:
        bool was_ = false;
    };

    /// The width of the pool the backend's next compute runs on (0 when it has no TensorSharp pools).
    int cpu_pool_threads(ggml_backend_t backend);

    /// Forgets every backend's pools (before TSGgml_Shutdown frees them).
    void cpu_pools_forget_all();

    /// Decode-width steps: one sequence's token, or a few sequences' (a small batch still reads every weight once).
    constexpr std::int64_t kCpuDecodeRows = 4;

    /// The pool a graph step runs on: 1 the decode pool (weight products of at most kCpuDecodeRows activation rows), 0
    /// the prompt pool (wider products), -1 the pool the previous step used (no weight products: an auxiliary graph
    /// between two decode steps would otherwise pause the decode pool and resume it again around itself).
    constexpr int cpu_pool_for_step(std::int64_t product_rows)
    {
        return product_rows <= 0 ? -1 : product_rows <= kCpuDecodeRows ? 1 : 0;
    }
}
