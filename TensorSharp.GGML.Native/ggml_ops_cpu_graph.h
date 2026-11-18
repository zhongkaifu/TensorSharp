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

    /// Creates the CPU backend's thread pools (the prompt pool always; a decode pool when its width differs) and sets
    /// the prompt pool; the pools are appended to `owned` (freed at shutdown).
    void cpu_pools_create(ggml_backend_t backend, std::vector<ggml_threadpool_t>& owned);

    /// Runs the next graph on the pool for its width: `product_rows` is the most activation rows any weight product of
    /// the graph reads (0 when it has none). Steps of at most kCpuDecodeRows rows run on the decode pool.
    void cpu_pools_select(ggml_backend_t backend, std::int64_t product_rows);

    /// Forgets every backend's pools (before TSGgml_Shutdown frees them).
    void cpu_pools_forget_all();

    /// Decode-width steps: one sequence's token, or a few sequences' (a small batch still reads every weight once).
    constexpr std::int64_t kCpuDecodeRows = 4;
}
