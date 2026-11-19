// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The diffusion-transformer offload (ggml_ops_dit_offload.h): moved out of
// ggml_ops_minimax_h3.cpp, where MiniMax-H3's native graphs used it alone, so the
// generic graph builder runs the same nodes. The code is that file's, unchanged.
#include "ggml_ops_dit_offload.h"
#include "ggml_ops_internal.h"
#include "ggml_ops_matmul_precision.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(TSG_GGML_USE_CUDA) && defined(TSG_HAVE_CUDNN)
extern "C" bool tsg_cudnn_sdpa_available(void);
#endif

namespace tsg
{
namespace
{
// ---------------------------------------------------------------------------
// Attention on cuDNN's fused SDPA (ggml_cuda only).
//
// The DiT's 50 full attentions over one packed clip are half of a denoising step
// on ggml-cuda: its MMA F16 flash-attention kernel runs a 480x864x158 clip's
// 20,511-token attention (56 x 128) in about 730 ms where cuDNN's SDPA takes
// about 177 ms, Q narrowing and output widening included. The
// graph keeps its GGML_OP_FLASH_ATTN_EXT nodes; graph_compute_offload executes the
// ones cuDNN can take (F32 Q, F16 K/V, no mask, sinks, ALiBi or softcap) into the
// node's own buffer, and any other or declined node runs on ggml as before.
// cuDNN accumulates P*V in F32 where ggml's kernel accumulates in F16 (see the
// note above h3_attend). TS_H3_CUDNN_ATTENTION=0 keeps every attention on ggml.
// ---------------------------------------------------------------------------
#if defined(TSG_GGML_USE_CUDA) && defined(TSG_HAVE_CUDNN)
extern "C" bool tsg_cudnn_sdpa_f32_strided(const float* q, int64_t q_stride_s, int64_t q_stride_h, const void* k,
                                           const void* v, float* out, int64_t heads, int64_t seq_q, int64_t seq_kv,
                                           int64_t head_dim, int64_t k_stride_s, int64_t k_stride_h,
                                           int64_t v_stride_s, int64_t v_stride_h, float scale);

// stridedQ also admits a Q that is a strided view (a permutation of [D, H, Sq]).
bool sdpa_wants(const ggml_tensor* node, bool stridedQ);

bool h3_sdpa_wants(const ggml_tensor* node, void*)
{
    return sdpa_wants(node, false);
}

bool sdpa_wants(const ggml_tensor* node, bool stridedQ)
{
    if (node->op != GGML_OP_FLASH_ATTN_EXT) return false;
    const ggml_tensor* q = node->src[0];
    const ggml_tensor* k = node->src[1];
    const ggml_tensor* v = node->src[2];
    for (int i = 3; i < GGML_MAX_SRC; i++)
        if (node->src[i] != nullptr) return false;          // mask, sinks, gathered rows
    if (q == nullptr || k == nullptr || v == nullptr) return false;
    float params[3];
    std::memcpy(params, node->op_params, sizeof(params));   // scale, max_bias, softcap
    if (params[1] != 0.0f || params[2] != 0.0f) return false;
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
        node->type != GGML_TYPE_F32)
        return false;
    const int64_t D = q->ne[0];
    if (D % 8 != 0 || D > 256 || k->ne[0] != D || v->ne[0] != D) return false;
    if (q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1) return false;
    if (k->ne[2] != q->ne[2] || v->ne[2] != q->ne[2] || v->ne[1] != k->ne[1]) return false;   // no GQA
    if (!ggml_is_contiguous(node)) return false;
    if (stridedQ ? q->nb[0] != sizeof(float) || q->nb[1] % sizeof(float) != 0 || q->nb[2] % sizeof(float) != 0
                 : !ggml_is_contiguous(q))
        return false;
    if (node->ne[0] != D || node->ne[1] != q->ne[2] || node->ne[2] != q->ne[1]) return false;
    for (const ggml_tensor* t : { k, v })
        if (t->nb[0] != sizeof(ggml_fp16_t) || t->nb[1] % 16 != 0 || t->nb[2] % 16 != 0) return false;
    for (const ggml_tensor* t : { q, k, v, static_cast<const ggml_tensor*>(node) })
        if (reinterpret_cast<std::uintptr_t>(t->data) % 16 != 0) return false;
    return true;
}

bool h3_sdpa_run_node(ggml_tensor* node)
{
    const ggml_tensor* q = node->src[0];
    const ggml_tensor* k = node->src[1];
    const ggml_tensor* v = node->src[2];
    float scale = 0.0f;
    std::memcpy(&scale, node->op_params, sizeof(float));
    const int64_t h = sizeof(ggml_fp16_t);
    const int64_t f = sizeof(float);
    return tsg_cudnn_sdpa_f32_strided(static_cast<const float*>(q->data), static_cast<int64_t>(q->nb[1]) / f,
                              static_cast<int64_t>(q->nb[2]) / f, k->data, v->data, static_cast<float*>(node->data),
                              q->ne[2], q->ne[1], k->ne[1], q->ne[0],
                              static_cast<int64_t>(k->nb[1]) / h, static_cast<int64_t>(k->nb[2]) / h,
                              static_cast<int64_t>(v->nb[1]) / h, static_cast<int64_t>(v->nb[2]) / h, scale);
}
#endif

// ---------------------------------------------------------------------------
// The DiT's Q4_K projections on cuBLAS BF16 (ggml_cuda only; tsg_cuda_q4k_gemm.cu).
//
// At 20,511 tokens ggml's MMQ ran them at about 52 TOPS where a BF16 cuBLAS GEMM
// ran at about 89 TFLOP/s; the weights are re-expanded to BF16 per call
// (about 1 ms for the fused qkv) so nothing extra stays resident. Only the
// dit-forward graph's wide matmuls take it: the text encoder's raw hidden states
// carry load-bearing outliers and too few tokens to gain. TS_H3_CUBLAS_MATMUL=0
// keeps them on MMQ; TS_H3_CUBLAS_MIN_ROWS (default 1024) is the token threshold.
// ---------------------------------------------------------------------------
} // namespace
#if defined(TSG_GGML_USE_CUDA)
extern "C" bool tsg_cuda_bf16_gemm(const void* w_bf16, int64_t K, int64_t N, const float* x, int64_t x_row_stride,
                                   int x_swiglu, int64_t T, float* dst, int64_t dst_ld);
extern "C" bool tsg_cuda_q4k_gemm_bf16(const void* w_q4k, int64_t K, int64_t N, const float* x,
                                       int64_t x_row_stride, int64_t T, float* dst, int64_t dst_ld);
extern "C" bool tsg_cuda_q4k_gemm_supported(void);

bool h3_cublas_matmul_enabled()
{
    static const bool on = []{
        const char* e = std::getenv("TS_H3_CUBLAS_MATMUL");
        return !(e != nullptr && e[0] == '0') && g_backend_type == BACKEND_TYPE_CUDA &&
               tsg_cuda_q4k_gemm_supported();
    }();
    return on;
}

int64_t h3_cublas_min_rows()
{
    static const int64_t rows = []{
        const char* e = std::getenv("TS_H3_CUBLAS_MIN_ROWS");
        const long long v = e != nullptr ? std::atoll(e) : 0;
        return static_cast<int64_t>(v > 0 ? v : 1024);
    }();
    return rows;
}

namespace
{
bool h3_q4k_matmul_wants(const ggml_tensor* node)
{
    if (node->op != GGML_OP_MUL_MAT) return false;
    const ggml_tensor* w = node->src[0];
    const ggml_tensor* x = node->src[1];
    if (w == nullptr || x == nullptr || w->type != GGML_TYPE_Q4_K) return false;
    if (x->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
    if (w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1) return false;
    if (w->ne[0] % 256 != 0 || x->ne[0] != w->ne[0] || x->ne[1] < h3_cublas_min_rows()) return false;
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(node)) return false;
    if (x->nb[0] != sizeof(float) || x->nb[1] % sizeof(float) != 0) return false;
    return true;
}

} // namespace
// Whether a DiT projection being built will run on cuBLAS BF16 (h3_q4k_matmul_wants
// at run time). Such a matmul needs no q8_1 overflow guard (h3_mm): BF16 keeps
// F32's exponent range, so its two full-tensor scale passes are left out. Should
// cuBLAS ever decline the node, MMQ then sees the unguarded activation; an
// overflow there surfaces as the pipeline's non-finite-output error, not as
// silently different output.
bool h3_q4k_gemm_takes(const ggml_tensor* w, const ggml_tensor* x)
{
    return h3_cublas_matmul_enabled() && w->type == GGML_TYPE_Q4_K && w->ne[0] % 256 == 0 &&
           x->type == GGML_TYPE_F32 && x->ne[1] >= h3_cublas_min_rows() && x->ne[2] == 1 && x->ne[3] == 1;
}

extern "C" bool tsg_cuda_lora_gemm_supported(void);

bool h3_lora_kcat_takes(const ggml_tensor* w, const ggml_tensor* x)
{
    const bool blocked = (w->type == GGML_TYPE_Q4_K || w->type == GGML_TYPE_Q6_K) && w->ne[0] % 256 == 0;
    return h3_cublas_matmul_enabled() && !h3_f32_products_enabled() && tsg_cuda_lora_gemm_supported() &&
           (blocked || w->type == GGML_TYPE_BF16) && ggml_is_contiguous(w) && w->ne[2] == 1 && w->ne[3] == 1 &&
           x->type == GGML_TYPE_F32 && x->ne[1] >= h3_cublas_min_rows() && x->ne[2] == 1 && x->ne[3] == 1;
}

namespace
{
// A wide product of a BF16 weight (MiniMax-H3-Image's DiT, read from its BF16 release)
// on cuBLAS: the weight as it is, the activations narrowed to BF16, F32 accumulation;
// with TS_H3_INT8_MATMUL=1 on int8 tensor cores, as for Q4_K (tsg_cuda_bf16_gemm). Only
// bound weights: a computed left operand stays ggml's.
bool h3_bf16_matmul_wants(const ggml_tensor* node)
{
    if (node->op != GGML_OP_MUL_MAT) return false;
    const ggml_tensor* w = node->src[0];
    const ggml_tensor* x = node->src[1];
    if (w == nullptr || x == nullptr || w->type != GGML_TYPE_BF16) return false;
    if ((w->view_src != nullptr ? w->view_src : w)->op != GGML_OP_NONE) return false;
    if (x->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32) return false;
    if (w->ne[2] != 1 || w->ne[3] != 1 || x->ne[2] != 1 || x->ne[3] != 1) return false;
    if (x->ne[0] != w->ne[0] || x->ne[1] < h3_cublas_min_rows()) return false;
    if (!ggml_is_contiguous(w) || !ggml_is_contiguous(node)) return false;
    if (x->nb[0] != sizeof(float) || x->nb[1] % sizeof(float) != 0) return false;
    return true;
}

bool h3_bf16_matmul_run(ggml_tensor* node)
{
    const ggml_tensor* w = node->src[0];
    const ggml_tensor* x = node->src[1];
    return tsg_cuda_bf16_gemm(w->data, w->ne[0], w->ne[1], static_cast<const float*>(x->data),
                              static_cast<int64_t>(x->nb[1] / sizeof(float)), 0, x->ne[1],
                              static_cast<float*>(node->data), node->ne[0]);
}

bool h3_q4k_matmul_run(ggml_tensor* node)
{
    const ggml_tensor* w = node->src[0];
    const ggml_tensor* x = node->src[1];
    return tsg_cuda_q4k_gemm_bf16(w->data, w->ne[0], w->ne[1], static_cast<const float*>(x->data),
                                  static_cast<int64_t>(x->nb[1] / sizeof(float)), x->ne[1],
                                  static_cast<float*>(node->data), node->ne[0]);
}
} // namespace
#endif

namespace
{
void h3_fused_unreachable(ggml_tensor*, int, int, void*)
{
    GGML_ABORT("MiniMax-H3: a fused CUDA node reached ggml; it runs only through graph_compute_offload");
}

} // namespace
H3FusedParams h3_fused_params(const ggml_tensor* node)
{
    H3FusedParams p{};
    if (node->op != GGML_OP_CUSTOM) return p;
    ggml_custom_op_t fun = nullptr;
    std::memcpy(&fun, node->op_params, sizeof(fun));
    if (fun != h3_fused_unreachable) return p;
    std::memcpy(&p, reinterpret_cast<const char*>(node->op_params) + kH3FusedParamsOffset, sizeof(p));
    return p;
}

ggml_tensor* h3_fused_node(ggml_context* ctx, std::int64_t ne0, std::int64_t ne1,
                           const std::vector<ggml_tensor*>& srcs, const H3FusedParams& p)
{
    ggml_tensor* args[GGML_MAX_SRC] = {};
    int n = 0;
    for (ggml_tensor* s : srcs) args[n++] = s;
    ggml_tensor* node = ggml_custom_4d(ctx, GGML_TYPE_F32, ne0, ne1, 1, 1, args, n, h3_fused_unreachable, 1, nullptr);
    std::memcpy(reinterpret_cast<char*>(node->op_params) + kH3FusedParamsOffset, &p, sizeof(p));
    // The node profiler splits GGML_OP_CUSTOM by name: the stage and, for a GEMM, its width.
    static const char* const names[] = { "", "attention", "modulate", "residual", "gemm", "f16gemm", "scaled-residual", "lora-gemm" };
    if (p.kind == H3_FUSED_LORA_GEMM || p.kind == H3_FUSED_F16_GEMM || p.kind == H3_FUSED_LORA_KCAT)
        ggml_format_name(node, "%s %lldx%lld%s",
                         p.kind == H3_FUSED_F16_GEMM && srcs[0]->type == GGML_TYPE_BF16 ? "bf16gemm" : names[p.kind],
                         static_cast<long long>(srcs[0]->ne[0]),
                         static_cast<long long>(srcs[0]->ne[1]), p.f0 != 0.0f ? " swiglu" : "");
    else
        ggml_set_name(node, names[p.kind]);
    return node;
}

bool h3_fused_enabled()
{
#if defined(TSG_GGML_USE_CUDA)
    static const bool on = []{
        const char* e = std::getenv("TS_H3_FUSED");
        return !(e != nullptr && e[0] == '0') && g_backend_type == BACKEND_TYPE_CUDA;
    }();
    return on;
#else
    return false;
#endif
}

#if defined(TSG_GGML_USE_CUDA)
namespace
{
extern "C" int tsg_h3_fused_max_segments(void);
extern "C" bool tsg_h3_modulate(const float* x, int64_t x_rs, const float* w, const float* mods, int64_t m_rs,
                                int shiftIdx, int scaleIdx, const int32_t* segs, int nseg, float eps,
                                int hidden, int64_t rows, float* out);
extern "C" bool tsg_h3_gated_residual(const float* base, int64_t b_rs, const float* mm, int64_t mm_rs, const float* bias,
                                      const float* delta, int64_t d_rs, const float* mods, int64_t m_rs, int gateIdx,
                                      const int32_t* segs, int nseg, int hidden, int64_t rows, float* out,
                                      const uint32_t* mmRow, const uint32_t* mmTok);
extern "C" bool tsg_h3_fused_sync(void);
extern "C" bool tsg_h3_scaled_residual(const float* base, int64_t b_rs, const float* mm, int64_t mm_rs, const float* bias,
                                       const float* scale, int hidden, int64_t rows, float* out,
                                       const uint32_t* mmRow, const uint32_t* mmTok);
extern "C" bool tsg_cuda_f16_gemm(const void* w_f16, int64_t K, int64_t N, const float* x, int64_t x_row_stride,
                                  const float* x_bias, int swiglu, int64_t T, float* dst, int64_t dst_ld, int flags);
extern "C" void tsg_cuda_f16_gemm_forget_weights(void);
extern "C" void tsg_cuda_q4k_gemm_forget_weights(void);
extern "C" bool tsg_cuda_q4k_lora_gemm_bf16(const void* w_q4k, int64_t K, int64_t N, const float* x, int64_t x_row_stride,
                                            int64_t T, float* dst, int64_t dst_ld, int nparts, const void* const* downs,
                                            const void* const* ups, const int64_t* ranks, const int64_t* outs,
                                            const int64_t* offsets, int factor_f32, int x_swiglu, int flags);
extern "C" bool tsg_cuda_lora_gemm_bf16(int wtype, const void* w, int64_t K, int64_t N, const float* x, int64_t x_row_stride,
                                        int64_t T, float* dst, int64_t dst_ld, int nparts, const void* const* downs,
                                        const void* const* ups, const int64_t* ranks, const int64_t* outs,
                                        const int64_t* offsets, int factor_f32, int x_swiglu);
// A product an int8 GEMM left as int32 sums for its one reader (tsg_cuda_int8_scale.h).
extern "C" bool tsg_cuda_int8_pending(const void* data, const uint32_t** rowpeak, const uint32_t** tokpeak, int64_t* rows,
                                      int64_t* ld);
extern "C" void tsg_cuda_int8_consumed(const void* data);
extern "C" bool tsg_cuda_int8_materialize(const void* data);
extern "C" int tsg_cuda_int8_forget_pending(void);
#if defined(TSG_HAVE_CUDNN)
extern "C" bool tsg_cudnn_sdpa_prepare(int64_t heads, int64_t seq, int64_t head_dim);
extern "C" bool tsg_cudnn_sdpa_prepare_batched(int64_t heads, int64_t seq, int64_t head_dim, int64_t batch);
extern "C" bool tsg_cudnn_h3_attention(const float* q, int64_t q_sh, int64_t q_ss, const float* k, int64_t k_sh, int64_t k_ss,
                                       const float* v, int64_t v_sh, int64_t v_ss, const float* qb, const float* kb,
                                       const float* vb, int64_t bias_hs, const float* qn, const float* kn,
                                       const float* cosT, const float* ssinT, int rot, float eps, float vScale, float scale,
                                       int heads, int seq, int head_dim, int batch, float* out, const uint32_t* rowpeak,
                                       const uint32_t* tokpeak, int64_t q_off, int64_t k_off, int64_t v_off);
#endif

std::int64_t h3_rows(const ggml_tensor* t) { return static_cast<std::int64_t>(t->nb[1] / sizeof(float)); }

const ggml_tensor* h3_view_base(const ggml_tensor* t) { return t->view_src != nullptr ? t->view_src : t; }

} // namespace
// The products of a graph that an int8 GEMM may leave as int32 sums: those of a fused
// projection whose result is read by exactly one node, and by that node only through
// an input its kernel can take that way (the attention's q, k and v, a projection's
// x, a residual's mm; see h3_fused_run). Views read nothing themselves. Any other
// reader, a second reader or an output flag keeps the product F32. A projection with
// a BF16 weight (tsg_cuda_bf16_gemm) neither leaves int32 sums nor reads them.
std::unordered_set<const ggml_tensor*> h3_deferrable(ggml_cgraph* graph)
{
    const int n = ggml_graph_n_nodes(graph);
    auto int8Projection = [](const ggml_tensor* node, std::int32_t kind) {
        return kind == H3_FUSED_LORA_GEMM ||
               (kind == H3_FUSED_F16_GEMM && node->src[0] != nullptr && node->src[0]->type != GGML_TYPE_BF16);
    };
    std::unordered_map<const ggml_tensor*, int> reader;     // -1 none yet, -2 not deferrable
    for (int i = 0; i < n; i++)
    {
        const ggml_tensor* node = ggml_graph_node(graph, i);
        if (!int8Projection(node, h3_fused_params(node).kind)) continue;
        reader[node] = (node->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ? -2 : -1;
    }
    std::unordered_set<const ggml_tensor*> out;
    if (reader.empty()) return out;
    for (int i = 0; i < n; i++)
    {
        const ggml_tensor* node = ggml_graph_node(graph, i);
        if (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_PERMUTE ||
            node->op == GGML_OP_TRANSPOSE)
            continue;
        const std::int32_t kind = h3_fused_params(node).kind;
        for (int j = 0; j < GGML_MAX_SRC; j++)
        {
            const ggml_tensor* src = node->src[j];
            if (src == nullptr) continue;
            auto it = reader.find(h3_view_base(src));
            if (it == reader.end()) continue;
            const bool whole = src->view_src == nullptr;
            const bool takes = (kind == H3_FUSED_ATTENTION && j <= 2) ||
                               (int8Projection(node, kind) && j == 1 && whole) ||
                               ((kind == H3_FUSED_RESIDUAL || kind == H3_FUSED_SCALED_RESIDUAL) && j == 1 && whole);
            if (!takes || (it->second != -1 && it->second != i)) it->second = -2;
            else it->second = i;
        }
    }
    for (const auto& kv : reader)
        if (kv.second >= 0) out.insert(kv.first);
    return out;
}

namespace
{
// `deferrable` (null = none): the graph's h3_deferrable set.
bool h3_fused_run(ggml_tensor* node, const std::unordered_set<const ggml_tensor*>* deferrable)
{
    const H3FusedParams p = h3_fused_params(node);
    const auto* segs = static_cast<const std::int32_t*>(p.segments);
    auto f = [](const ggml_tensor* t) { return t != nullptr ? static_cast<const float*>(t->data) : nullptr; };
    const int defer = deferrable != nullptr && deferrable->count(node) != 0 ? 1 : 0;
    // A residual's mm as int32 sums: its peaks when the kernel can read it in place
    // (the whole product, token for token), else it is scaled to F32 first.
    struct Sums { const uint32_t* row = nullptr; const uint32_t* tok = nullptr; bool pending = false; bool ok = true; };
    auto sumsOf = [&](const ggml_tensor* mm) {
        Sums s;
        std::int64_t rows = 0, ld = 0;
        if (!tsg_cuda_int8_pending(h3_view_base(mm)->data, &s.row, &s.tok, &rows, &ld)) return s;
        if (mm->view_src == nullptr && h3_rows(mm) == ld && node->ne[0] <= rows) { s.pending = true; return s; }
        s.row = s.tok = nullptr;
        s.ok = tsg_cuda_int8_materialize(h3_view_base(mm)->data);
        return s;
    };
    bool ok = false;
    switch (p.kind)
    {
    case H3_FUSED_ATTENTION:
    {
#if defined(TSG_HAVE_CUDNN)
        const ggml_tensor* q = node->src[0];
        const ggml_tensor* k = node->src[1];
        const ggml_tensor* v = node->src[2];
        const std::int64_t fl = sizeof(float);
        const float* bias = f(node->src[7]);
        // The DiT's qkv bias is stacked [q | k | v]; the video VAE's interleaves the
        // three per head (c = D, b = 3*D).
        const std::int64_t part = p.c > 0 ? p.c : q->ne[0] * q->ne[1];
        // A q longer than the rotation tables is a batch of independent sequences of
        // the tables' length (the image VAE's row of tiles, h3_vit_decode).
        const ggml_tensor* cosT = node->src[5];
        const std::int64_t tokens = q->ne[2];
        const std::int64_t perSeq = cosT != nullptr && cosT->ne[1] > 0 && tokens % cosT->ne[1] == 0 ? cosT->ne[1] : tokens;
        // q, k and v as views of one product still in int32: its peaks and the row each
        // view starts at, when all three are token-for-token views of it from its first
        // token (one node for the whole batch); otherwise whatever is pending behind
        // them becomes F32 first.
        const ggml_tensor* qkv = h3_view_base(q);
        const uint32_t* rowpeak = nullptr;
        const uint32_t* tokpeak = nullptr;
        std::int64_t rows = 0, ld = 0, off[3] = { 0, 0, 0 };
        bool pending = tsg_cuda_int8_pending(qkv->data, &rowpeak, &tokpeak, &rows, &ld);
        if (pending)
        {
            const ggml_tensor* views[3] = { q, k, v };
            bool inPlace = true;
            for (int i = 0; i < 3 && inPlace; i++)
            {
                const ggml_tensor* t = views[i];
                off[i] = static_cast<std::int64_t>(static_cast<const char*>(t->data) - static_cast<const char*>(qkv->data)) / fl;
                const std::int64_t last = off[i] + (t->ne[1] - 1) * static_cast<std::int64_t>(t->nb[1] / fl) + t->ne[0];
                inPlace = h3_view_base(t) == qkv && static_cast<std::int64_t>(t->nb[2] / fl) == ld && off[i] >= 0 &&
                          last <= rows;
            }
            if (!inPlace) { pending = false; rowpeak = tokpeak = nullptr; }
        }
        if (!pending)
            for (const ggml_tensor* t : { q, k, v })
                if (!tsg_cuda_int8_materialize(h3_view_base(t)->data)) return false;
        ok = tsg_cudnn_h3_attention(f(q), q->nb[1] / fl, q->nb[2] / fl, f(k), k->nb[1] / fl, k->nb[2] / fl,
                                    f(v), v->nb[1] / fl, v->nb[2] / fl, bias, bias != nullptr ? bias + part : nullptr,
                                    bias != nullptr ? bias + 2 * part : nullptr, p.b, f(node->src[3]), f(node->src[4]),
                                    f(cosT), f(node->src[6]), p.a, p.f0, p.f1, p.f2,
                                    static_cast<int>(q->ne[1]), static_cast<int>(tokens), static_cast<int>(q->ne[0]),
                                    static_cast<int>(tokens / perSeq), static_cast<float*>(node->data),
                                    rowpeak, tokpeak, off[0], off[1], off[2]);
        if (pending) tsg_cuda_int8_consumed(qkv->data);
#endif
        return ok;   // synchronized inside
    }
    case H3_FUSED_LORA_KCAT:
    case H3_FUSED_LORA_GEMM:
    {
        const ggml_tensor* w = node->src[0];
        const ggml_tensor* x = node->src[1];
        const void* downs[3]; const void* ups[3];
        std::int64_t ranks[3], outs[3], offsets[3];
        int n = 0;
        for (int slot = 0; slot < 3; slot++)
        {
            if ((p.a & (1 << slot)) == 0) continue;
            const ggml_tensor* down = node->src[2 + 2 * n];
            const ggml_tensor* up = node->src[3 + 2 * n];
            downs[n] = down->data; ups[n] = up->data;
            ranks[n] = down->ne[1]; outs[n] = up->ne[1];
            offsets[n] = static_cast<std::int64_t>(slot) * p.b;
            n++;
        }
        const bool f32 = n > 0 && node->src[2]->type == GGML_TYPE_F32;
        if (p.kind == H3_FUSED_LORA_KCAT)
            return tsg_cuda_lora_gemm_bf16(static_cast<int>(w->type), w->data, w->ne[0], w->ne[1], f(x), h3_rows(x), x->ne[1],
                                           static_cast<float*>(node->data), node->ne[0], n, downs, ups, ranks, outs, offsets,
                                           f32 ? 1 : 0, p.f0 != 0.0f ? 1 : 0);   // synchronized inside
        // 2: the weight's int8 rows may be kept; it and its factors stay bound until the
        // pipeline releases the scratch.
        return tsg_cuda_q4k_lora_gemm_bf16(w->data, w->ne[0], w->ne[1], f(x), h3_rows(x), x->ne[1],
                                           static_cast<float*>(node->data), node->ne[0], n, downs, ups, ranks, outs,
                                           offsets, f32 ? 1 : 0, p.f0 != 0.0f ? 1 : 0, defer | 2);   // synchronized inside
    }
    case H3_FUSED_F16_GEMM:
    {
        const ggml_tensor* w = node->src[0];
        const ggml_tensor* x = node->src[1];
        // A BF16 weight is the DiT's fc2 reading fc1's [gate | value] (no x bias): the
        // BF16 cuBLAS route or, with TS_H3_INT8_MATMUL, the int8 one, as h3_bf16_matmul_run.
        if (w->type == GGML_TYPE_BF16)
            return node->src[2] == nullptr &&
                   tsg_cuda_bf16_gemm(w->data, w->ne[0], w->ne[1], f(x), h3_rows(x), p.f0 != 0.0f ? 1 : 0, x->ne[1],
                                      static_cast<float*>(node->data), node->ne[0]);
        return tsg_cuda_f16_gemm(w->data, w->ne[0], w->ne[1], f(x), h3_rows(x), f(node->src[2]),
                                 p.f0 != 0.0f ? 1 : 0, x->ne[1], static_cast<float*>(node->data), node->ne[0], defer);
    }
    case H3_FUSED_SCALED_RESIDUAL:
    {
        const ggml_tensor* base = node->src[0];
        const ggml_tensor* mm = node->src[1];
        const Sums sums = sumsOf(mm);
        if (!sums.ok) return false;
        ok = tsg_h3_scaled_residual(f(base), h3_rows(base), f(mm), h3_rows(mm), f(node->src[2]), f(node->src[3]),
                                    static_cast<int>(node->ne[0]), node->ne[1], static_cast<float*>(node->data),
                                    sums.row, sums.tok)
            && tsg_h3_fused_sync();
        if (sums.pending) tsg_cuda_int8_consumed(mm->data);
        return ok;
    }
    case H3_FUSED_MODULATE:
    {
        const ggml_tensor* x = node->src[0];
        const ggml_tensor* mods = node->src[2];
        ok = tsg_h3_modulate(f(x), h3_rows(x), f(node->src[1]), f(mods), h3_rows(mods), p.a, p.b, segs, p.nseg,
                             p.f0, static_cast<int>(node->ne[0]), node->ne[1], static_cast<float*>(node->data));
        break;
    }
    case H3_FUSED_RESIDUAL:
    {
        const ggml_tensor* base = node->src[0];
        const ggml_tensor* mm = node->src[1];
        const ggml_tensor* delta = node->src[3];
        const ggml_tensor* mods = node->src[4];
        const Sums sums = sumsOf(mm);
        if (!sums.ok) return false;
        ok = tsg_h3_gated_residual(f(base), h3_rows(base), f(mm), h3_rows(mm), f(node->src[2]),
                                   f(delta), delta != nullptr ? h3_rows(delta) : 0, f(mods), h3_rows(mods), p.a,
                                   segs, p.nseg, static_cast<int>(node->ne[0]), node->ne[1],
                                   static_cast<float*>(node->data), sums.row, sums.tok)
            && tsg_h3_fused_sync();
        if (sums.pending) tsg_cuda_int8_consumed(mm->data);
        return ok;
    }
    default:
        return false;
    }
    return ok && tsg_h3_fused_sync();
}
} // namespace
#endif

namespace
{
// What one graph offloads: attention everywhere cuDNN takes it, wide Q4_K matmuls
// and the fused stages only in the DiT forward.
struct H3Offload
{
    bool attention = false;
    bool matmul = false;
    bool fused = false;
    bool f32 = false;                  // TS_H3_CUDA_F32_PRODUCTS: weight products on the pedantic handle
    ggml_backend_t backend = nullptr;
    std::unordered_set<const ggml_tensor*> deferrable;     // h3_deferrable of the graph
};

// ---------------------------------------------------------------------------
// The DiT forward's weight products in F32 (ggml_cuda only, TS_H3_CUDA_F32_PRODUCTS=1).
//
// ggml-cuda runs a float matmul on cuBLAS handles it sets to
// CUBLAS_TF32_TENSOR_OP_MATH, which rounds both operands to TF32's 10-bit
// mantissa. With the switch, each product of the DiT forward whose left operand
// is a bound float tensor (an F32/F16/BF16 weight or a LoRA plug-in's down or up
// factor, or a row view of one) runs instead on TensorSharp's F32 GEMM over a
// private CUBLAS_PEDANTIC_MATH handle (ggml_ops_matmul_precision.cu, the seam the
// Gemma 4 audio graphs use). Quantized weights keep their kernels, and the
// attention's own products keep ggml's. Off by default: ggml_cuda keeps its
// TF32/BF16 speed, and the switch's cost on a real checkpoint is not measured.
//
// On the all-F32 LoRA parity fixture (MiniMaxH3LoraParityTests) the native step
// against the direct CPU path, without a plug-in, went from rel 6.08e-4 to 8.07e-6;
// every float product including the attention's: 1.24e-7; the plug-in's products
// alone left it at 6.08e-4, since TF32 noise in the base path does not cancel
// between the step with and without the update.
// ---------------------------------------------------------------------------
[[maybe_unused]] void h3_f32_warn_once(std::atomic<bool>& said, const std::string& what)
{
    if (!said.exchange(true))
        std::fprintf(stderr, "[minimax-h3] TS_H3_CUDA_F32_PRODUCTS=1: %s\n", what.c_str());
}

} // namespace
[[maybe_unused]] bool h3_f32_products_enabled()
{
    // Read per graph, as TS_H3_NO_FLASH is: a test process switches it between steps.
    const char* e = std::getenv("TS_H3_CUDA_F32_PRODUCTS");
    if (e == nullptr || std::strcmp(e, "1") != 0) return false;
#if defined(TSG_GGML_USE_CUDA)
    if (g_backend_type == BACKEND_TYPE_CUDA) return true;
#endif
    static std::atomic<bool> said{false};
    h3_f32_warn_once(said, "the switch applies to ggml_cuda only; this backend keeps its own matrix products.");
    return false;
}

namespace
{

#if defined(TSG_GGML_USE_CUDA)
bool h3_f32_product_wants(const ggml_tensor* node)
{
    if (node->op != GGML_OP_MUL_MAT || node->src[0] == nullptr || node->src[1] == nullptr) return false;
    const ggml_tensor* a = node->src[0];
    const ggml_tensor* bound = a->view_src != nullptr ? a->view_src : a;
    if (bound->op != GGML_OP_NONE) return false;   // the attention's products and other computed operands
    if (a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16 && a->type != GGML_TYPE_BF16) return false;
    if (node->type == GGML_TYPE_F32 && node->src[1]->type == GGML_TYPE_F32) return true;
    static std::atomic<bool> said{false};
    h3_f32_warn_once(said, std::string("a ") + ggml_type_name(a->type) + " x " + ggml_type_name(node->src[1]->type) +
                     " product (" + node->name + ") has no F32 path; it runs on ggml-cuda's TF32 cuBLAS.");
    return false;
}

bool h3_f32_product_run(ggml_tensor* node, ggml_backend_t backend)
{
    // One private pedantic handle per backend, on that backend's stream.
    static std::mutex lock;
    static std::vector<std::pair<ggml_backend_t, tsg_matmul_cuda_state*>> states;
    tsg_matmul_cuda_state* state = nullptr;
    {
        std::lock_guard<std::mutex> hold(lock);
        for (auto& s : states)
            if (s.first == backend) state = s.second;
        if (state == nullptr && (state = tsg_matmul_cuda_try_init(backend)) != nullptr)
            states.emplace_back(backend, state);
    }
    if (state == nullptr)
    {
        static std::atomic<bool> said{false};
        h3_f32_warn_once(said, "the pedantic cuBLAS handle could not be created (device memory); the weight products run "
                               "on ggml-cuda's TF32 cuBLAS.");
        return false;   // compute_split runs the node on ggml
    }
    tsg_matmul_cuda_compute(state, node);
    tsg::sync_backend(backend);
    return true;
}
#endif

bool h3_offload_wants(const ggml_tensor* node, void* ctx)
{
    const auto* o = static_cast<const H3Offload*>(ctx);
#if defined(TSG_GGML_USE_CUDA) && defined(TSG_HAVE_CUDNN)
    if (o->attention && h3_sdpa_wants(node, nullptr)) return true;
#endif
#if defined(TSG_GGML_USE_CUDA)
    if (o->matmul && h3_q4k_matmul_wants(node)) return true;
    if (o->matmul && !o->f32 && h3_bf16_matmul_wants(node)) return true;
    if (o->fused && h3_fused_params(node).kind != H3_FUSED_NONE) return true;
    if (o->f32 && h3_f32_product_wants(node)) return true;
#endif
    (void) o; (void) node;
    return false;
}

bool h3_offload_run(ggml_tensor* node, void* ctx)
{
#if defined(TSG_GGML_USE_CUDA) && defined(TSG_HAVE_CUDNN)
    if (node->op == GGML_OP_FLASH_ATTN_EXT) return h3_sdpa_run_node(node);
#endif
#if defined(TSG_GGML_USE_CUDA)
    // A wanted float product is the F32 path's; a wanted quantized one is a Q4_K weight on cuBLAS BF16.
    if (node->op == GGML_OP_MUL_MAT && !ggml_is_quantized(node->src[0]->type))
    {
        const auto* o = static_cast<const H3Offload*>(ctx);
        if (o->matmul && !o->f32 && h3_bf16_matmul_wants(node)) return h3_bf16_matmul_run(node);
        return h3_f32_product_run(node, o->backend);
    }
    if (node->op == GGML_OP_MUL_MAT) return h3_q4k_matmul_run(node);
    if (node->op == GGML_OP_CUSTOM) return h3_fused_run(node, &static_cast<const H3Offload*>(ctx)->deferrable);
#endif
    (void) node;
    return false;
}

} // namespace
bool h3_cudnn_attention_enabled()
{
#if defined(TSG_GGML_USE_CUDA) && defined(TSG_HAVE_CUDNN)
    static const bool on = []{
        const char* e = std::getenv("TS_H3_CUDNN_ATTENTION");
        return !(e != nullptr && e[0] == '0') && g_backend_type == BACKEND_TYPE_CUDA && tsg_cudnn_sdpa_available();
    }();
    return on;
#else
    return false;
#endif
}

ggml_status dit_offload_compute(ggml_backend_t backend, ggml_cgraph* graph, const char* tag, bool wide_token_products)
{
    H3Offload which;
    which.attention = h3_cudnn_attention_enabled();
#if defined(TSG_GGML_USE_CUDA)
    which.matmul = h3_cublas_matmul_enabled() && wide_token_products;
    // Fused nodes exist only where a graph built them (the DiT and the video VAE decoder).
    which.fused = h3_fused_enabled();
    which.f32 = wide_token_products && h3_f32_products_enabled();
    which.backend = backend;
    // TS_H3_INT8_DEFER=0 has every int8 product scaled to F32 by its own pass, as before.
    static const bool defer = []{
        const char* e = std::getenv("TS_H3_INT8_DEFER");
        return !(e != nullptr && e[0] == '0');
    }();
    if (which.fused && defer) which.deferrable = h3_deferrable(graph);
#endif
    ggml_status status;
    if (which.attention || which.matmul || which.fused || which.f32)
    {
        NodeOffload offload;
        offload.wants = h3_offload_wants;
        offload.run = h3_offload_run;
        offload.ctx = &which;
        status = graph_compute_offload(backend, graph, tag, offload);
#if defined(TSG_GGML_USE_CUDA)
        // Every product left as int32 has a reader that takes it; one still pending
        // means some node read sums as F32.
        if (which.fused && tsg_cuda_int8_forget_pending() != 0 && status == GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr, "[minimax-h3] %s: an int8 product was left unscaled and never read\n", tag != nullptr ? tag : "");
            status = GGML_STATUS_FAILED;
        }
#endif
    }
    else
        status = graph_node_profile_enabled()
            ? graph_compute_profiled(backend, graph, tag)
            : compute_graph(backend, graph);
    return status;
}

// Declared in ggml_ops_internal.h (C++ linkage) for other models' graphs.
    bool cudnn_sdpa_node_wants(const ggml_tensor* node)
    {
#if defined(TSG_GGML_USE_CUDA) && defined(TSG_HAVE_CUDNN)
        return g_backend_type == BACKEND_TYPE_CUDA && sdpa_wants(node, true) && tsg_cudnn_sdpa_available();
#else
        (void) node;
        return false;
#endif
    }

    bool cudnn_sdpa_node_run(ggml_tensor* node)
    {
#if defined(TSG_GGML_USE_CUDA) && defined(TSG_HAVE_CUDNN)
        return h3_sdpa_run_node(node);
#else
        (void) node;
        return false;
#endif
    }
} // namespace tsg
