// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// The diffusion-transformer offload: the nodes of a DiT's ggml graph that run
// outside ggml on ggml_cuda, and the graph run that executes them
// (ggml_ops_dit_offload.cpp). TensorSharp-owned; ggml is not changed.
//
//   * cuDNN's fused SDPA for the GGML_OP_FLASH_ATTN_EXT nodes it takes;
//   * wide BF16 and Q4_K weight products on cuBLAS (or int8 tensor cores), and
//     TS_H3_CUDA_F32_PRODUCTS' pedantic F32 products;
//   * the fused stages (tsg_cuda_h3_fused.cu, tsg_cuda_cudnn_sdpa.cu,
//     tsg_cuda_q4k_gemm.cu), each one GGML_OP_CUSTOM node made by
//     h3_fused_node, with int8 products left unscaled for the one fused reader
//     that takes them (h3_deferrable).
//
// MiniMax-H3's native graphs (ggml_ops_minimax_h3.cpp) build these nodes by hand;
// the generic graph builder (ggml_ops_graph_builder.cpp) serves GraphFusion's DiT
// regions as the same nodes. Both run them through dit_offload_compute.
// ============================================================================
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#if defined(TSG_GGML_USE_CUDA)
extern "C" int tsg_h3_fused_max_segments(void);
extern "C" void tsg_cuda_f16_gemm_forget_weights(void);
extern "C" void tsg_cuda_q4k_gemm_forget_weights(void);
#if defined(TSG_HAVE_CUDNN)
extern "C" bool tsg_cudnn_sdpa_prepare(int64_t heads, int64_t seq, int64_t head_dim);
extern "C" bool tsg_cudnn_sdpa_prepare_batched(int64_t heads, int64_t seq, int64_t head_dim, int64_t batch);
#endif
#endif

namespace tsg
{
// ---------------------------------------------------------------------------
// Fused DiT stages (ggml_cuda only; tsg_cuda_h3_fused.cu).
//
// Each stands in the graph as one GGML_OP_CUSTOM node with the stage's real inputs
// as sources, so the allocator sizes and orders it like any other node, and
// graph_compute_offload executes it; ggml itself never runs one (they are only
// built when the offload will). Parameters ride in op_params after ggml's own
// ggml_custom_op_params (24 bytes):
//   H3_FUSED_ATTENTION  q, k, v [D, H, S] views, q_norm, k_norm, cos, signed sin,
//                       qkv bias (nullable) -> the attention output [D*H, S]
//                       (h3_attend's `merged`)
//   H3_FUSED_LORA_GEMM  w (Q4_K), x, then (down, up) per LoRA part -> (W + sum of
//                       up*down on each part's rows) x on cuBLAS BF16; a = which of
//                       up to three equal row slots carry a part, b = slot rows,
//                       f0 = 1 when x is fc1's [gate | value] and the GEMM reads its
//                       SwiGLU
//   H3_FUSED_F16_GEMM   w (F16), x, x bias (nullable) -> W act(x) on cuBLAS, act being
//                       x + bias or, with f0 = 1, SwiGLU(x + bias) (the video VAE); a BF16
//                       w, no bias, is a BF16 DiT's fc2 over fc1's [gate | value]
//                       (tsg_cuda_bf16_gemm, cuBLAS BF16 or int8)
//   H3_FUSED_SCALED_RESIDUAL base, mm, bias, scale -> base + (mm + bias) * scale
//   H3_FUSED_MODULATE   x, norm weight, mods -> RMSNorm(x)*w modulated per segment
//   H3_FUSED_RESIDUAL   base, mm, bias, delta, mods -> base + (delta+(mm+bias))*gate
// TS_H3_FUSED=0 builds the unfused graph.
// ---------------------------------------------------------------------------
enum H3FusedKind : std::int32_t { H3_FUSED_NONE = 0, H3_FUSED_ATTENTION = 1, H3_FUSED_MODULATE = 2, H3_FUSED_RESIDUAL = 3,
                                   H3_FUSED_LORA_GEMM = 4, H3_FUSED_F16_GEMM = 5, H3_FUSED_SCALED_RESIDUAL = 6 };

struct H3FusedParams
{
    std::int32_t kind;
    std::int32_t a, b;          // rot, bias head stride | shift, scale | gate
    float f0, f1, f2;           // eps, V scale, attention scale | eps | SwiGLU input
    const void* segments;       // TSGH3DitSegment[nseg] (the forward's descriptor)
    std::int32_t nseg;
    std::int32_t c;             // attention: distance between the q, k and v biases (0 = D*H)
};
constexpr std::size_t kH3FusedParamsOffset = 24;   // after ggml_custom_op_params
static_assert(kH3FusedParamsOffset + sizeof(H3FusedParams) <= GGML_MAX_OP_PARAMS, "fused params fit op_params");

// The parameters of a fused node (kind H3_FUSED_NONE for any other node).
H3FusedParams h3_fused_params(const ggml_tensor* node);

// A fused node of ne0 x ne1 F32 over `srcs` (null entries allowed); see above.
ggml_tensor* h3_fused_node(ggml_context* ctx, std::int64_t ne0, std::int64_t ne1,
                           const std::vector<ggml_tensor*>& srcs, const H3FusedParams& p);

// The fused stages are built (ggml_cuda, TS_H3_FUSED not 0).
bool h3_fused_enabled();

// cuDNN's fused SDPA serves the attentions it takes (ggml_cuda with cuDNN, TS_H3_CUDNN_ATTENTION not 0).
bool h3_cudnn_attention_enabled();

// TS_H3_CUDA_F32_PRODUCTS=1: the weight products of a wide graph on a pedantic F32 cuBLAS handle (ggml_cuda only).
bool h3_f32_products_enabled();

#if defined(TSG_GGML_USE_CUDA)
// Q4_K and BF16 products on cuBLAS BF16 (ggml_cuda, sm_80+, TS_H3_CUBLAS_MATMUL not 0) from this many rows
// (TS_H3_CUBLAS_MIN_ROWS, 1024).
bool h3_cublas_matmul_enabled();
std::int64_t h3_cublas_min_rows();

// Whether a DiT projection being built will run on cuBLAS BF16 (it then needs no q8_1 overflow guard).
bool h3_q4k_gemm_takes(const ggml_tensor* w, const ggml_tensor* x);

// The products of `graph` an int8 GEMM may leave as int32 sums for their one fused reader.
std::unordered_set<const ggml_tensor*> h3_deferrable(ggml_cgraph* graph);
#endif

// Runs `graph` with the DiT offload: cuDNN attention wherever it takes a node, the fused nodes, and with
// `wide_token_products` the cuBLAS/int8 weight products (and TS_H3_CUDA_F32_PRODUCTS' F32 ones); when nothing is
// offloaded, the graph as it is (profiled under TS_GGML_NODE_PROFILE). Fails a run that left an int8 product
// unscaled and never read. Synchronous.
ggml_status dit_offload_compute(ggml_backend_t backend, ggml_cgraph* graph, const char* tag, bool wide_token_products);
} // namespace tsg
