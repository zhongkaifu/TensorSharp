// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Fused DeepSeek V4 (Flash) ops, injected into the ggml graph as
// GGML_OP_CUSTOM nodes and executed by a small TensorSharp-owned ggml-backend
// that launches CUDA kernels on the paired ggml-cuda backend's stream. This
// keeps all custom-kernel work inside the TensorSharp native project — the
// vendored ggml tree is used strictly through its public backend API.
//
// Each fused node's op_params carry a ggml_custom_op_params whose userdata
// points at a tsg_dsv4_fused_desc (owned by the graph build result). The
// descs double as the dispatch table for both the CUDA launcher and the CPU
// reference fallback.
// ---------------------------------------------------------------------------

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>

enum tsg_dsv4_fused_kind : int32_t
{
    TSG_DSV4_FUSED_COMPRESS      = 1,
    TSG_DSV4_FUSED_ATTN_PREP     = 2,
    TSG_DSV4_FUSED_ATTN_FINISH   = 3,
    TSG_DSV4_FUSED_MOE_TOPK      = 4,
    TSG_DSV4_FUSED_MOE_WEIGHTS   = 5,
    TSG_DSV4_FUSED_EXPERT_REDUCE = 6,
    TSG_DSV4_FUSED_SWIGLU_CLAMP  = 7,
    TSG_DSV4_FUSED_HC_GATES      = 8,
    TSG_DSV4_FUSED_TOPK_MASK     = 9,
    TSG_DSV4_FUSED_TOPK_SELECT   = 10,
    TSG_DSV4_FUSED_KGATHER       = 11,
    // F32 contiguous src[0] -> same shape F32, including BF16 rounding.
    // i0: 0=FP8/E8M0 block32, 1=FP4/E8M0 block32, 2=FP4/E4M3 block16.
    TSG_DSV41_FUSED_QUANT        = 12,
    // src[0] F32 [nKV, nt] masked index scores; src[1] I32 positions[nt].
    // dst F32 [ceil(nKV/i0), nt], i0=block size; newest block pinned +inf.
    TSG_DSV41_CANDIDATE_SCORES   = 13,
    // src[0] pooled F32 scores, src[1] I32 top-k block IDs.
    // dst F16 [nKV,nt] additive mask (0/-inf), i0=block size.
    TSG_DSV41_CANDIDATE_MASK     = 14,
    TSG_MATMUL_F32              = 15,
    TSG_MATMUL_ID_F32           = 16,
    TSG_ATTN_F32_PARTIAL        = 17,
    TSG_ATTN_F32_FINISH         = 18,
    TSG_ATTN_F32_SOFTMAX        = 19,
    TSG_ATTN_MASK_COMPACT       = 20,
    TSG_MATMUL_Q8_F32           = 21,
    // Indexed Q2_K/Q4_K gate/up strip; i0=unsplit rows, i1=first row.
    TSG_MATMUL_ID_QUANT_STRIP   = 22,
    TSG_MATMUL_ID_QUANT_PAIR    = 23,
};

#define TSG_DSV4_FUSED_MAGIC 0x5453445356344655ull  // "TSDSV4FU"

struct tsg_dsv4_fused_desc
{
    uint64_t magic = TSG_DSV4_FUSED_MAGIC;
    int32_t  kind = 0;
    int32_t  i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    float    f0 = 0.0f, f1 = 0.0f;
};

// CPU reference implementation (ggml_custom_op_t signature); used when a
// fused node is scheduled onto the CPU backend.
void tsg_dsv4_fused_cpu(struct ggml_tensor * dst, int ith, int nth, void * userdata);

#ifdef TSG_GGML_USE_CUDA
// NVIDIA BF16 GEMM requires Ampere or newer. Check the backend's physical
// device, including virtual-backend mappings, without assuming device zero.
bool tsg_dsv4_cuda_supports_native_bf16(ggml_backend_t cuda_backend);

// Create a fused-op backend bound to `cuda_backend`'s device and stream.
// The returned backend claims GGML_OP_CUSTOM nodes carrying a
// tsg_dsv4_fused_desc AND everything `cuda_backend`'s device supports, so it
// REPLACES that backend in ggml_backend_sched rather than joining it: the
// scheduler then keeps a device's whole subgraph in one split. Ordinary nodes
// are forwarded to `cuda_backend` as graph views and fused nodes launch
// directly, all asynchronously on that backend's stream.
// The cuda_backend must outlive the returned backend.
ggml_backend_t tsg_dsv4_fused_backend_init(ggml_backend_t cuda_backend);

// Diagnostics for TS_DSV4_PERF>=3: how the device subgraph was submitted.
// `views` counts the forwarded runs of ordinary nodes (each one a separate
// ggml-cuda submission, and therefore a separate CUDA-graph launch or capture),
// `fused` the custom kernels launched between them, and `nodes` every node the
// scheduler handed this backend. Reset before a forward, read after it.
struct tsg_dsv4_fused_counters
{
    unsigned long long calls = 0, views = 0, fused = 0, nodes = 0;
    double submit_ms = 0.0;
};
void tsg_dsv4_fused_counters_reset();
tsg_dsv4_fused_counters tsg_dsv4_fused_counters_read();
#endif
