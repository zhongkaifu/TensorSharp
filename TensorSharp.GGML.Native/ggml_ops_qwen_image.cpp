// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// Qwen-Image MMDiT native fused kernels (Stage 7 performance path). The managed
// QwenImageDiT is the verified correctness reference; these kernels fold the
// elementwise + matmul work of a block sub-layer into a single ggml graph so the
// whole thing runs on-device in one dispatch (no per-op host round-trips, which
// are what leave the GPU idle in the managed path).
//
// First kernel: the modulated GEGLU MLP sub-layer for one stream:
//   normed     = layernorm(x)                       (no affine, eps 1e-6)
//   modulated  = normed * (1 + scale) + shift       (AdaLN; scale/shift per token)
//   h          = net0_w @ modulated + net0_b
//   h          = gelu(h)                            (tanh approximation)
//   mlp        = net2_w @ h + net2_b
//   out        = x + gate * mlp                     (gated residual)
// scale_plus1 / shift / gate are precomputed on the host (folding modulate_index)
// and passed as [dim, seq] tensors.
// ============================================================================
#include "ggml_ops_internal.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

using namespace tsg;

extern "C" {

// ============================================================================
// Single 2D convolution on the device (ggml_conv_2d). The Qwen-Image VAE encode/
// decode was running its conv stack as pure-C# scalar loops (~459 s to encode a
// 1 MP image on CPU while the GPU sat idle). Routing each conv through this kernel
// moves that work onto the GPU.
//
// Layouts match the C# VaeReferenceMath exactly (no transposes):
//   input  C# Feature [C,H,W] (x contiguous) == ggml [W,H,C,1]
//   weight C# [OC,IC,KH,KW] (kw contiguous)  == ggml [KW,KH,IC,OC]
//   output C# [OC,OH,OW]                      == ggml [OW,OH,OC,1]
// Padding: symmetric (padL==padR && padT==padB) uses ggml_conv_2d's built-in pad;
// the encoder Downsample's asymmetric (0,1,0,1) ZeroPad2d maps to a ggml_pad
// (end-pad of ne0/ne1) followed by an unpadded conv.
// ============================================================================
struct TSGgmlConv2dDesc
{
    void* input;  std::int32_t W, H, C;                 // [W,H,C] F32
    void* weight; std::int32_t wtype, KW, KH, IC, OC;   // [KW,KH,IC,OC]
    std::int64_t weight_bytes;
    void* bias;                                          // [OC] F32 or null
    void* output;                                        // [OW,OH,OC] F32 (caller-allocated)
    std::int32_t strideW, strideH, padL, padR, padT, padB;
    std::int32_t struct_bytes;
};

TSG_EXPORT int TSGgml_Conv2d(const TSGgmlConv2dDesc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlConv2dDesc)))
        { set_last_error("Conv2d: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        const int W = d->W, H = d->H, C = d->C;
        const int KW = d->KW, KH = d->KH, IC = d->IC, OC = d->OC;
        const int sW = d->strideW, sH = d->strideH;
        const bool symmetric = (d->padL == d->padR) && (d->padT == d->padB);

        PooledContextHandle context;
        if (!context.init(32 * 1024 * 1024)) { set_last_error("Conv2d: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        ggml_tensor* inp = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_tensor* ker = ggml_new_tensor_4d(ctx, static_cast<ggml_type>(d->wtype), KW, KH, IC, OC);
        ggml_tensor* bias = d->bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC) : nullptr;

        ggml_tensor* x = inp;
        int p0 = d->padL, p1 = d->padT;
        if (!symmetric)
        {
            // Only the encoder Downsample is asymmetric: ZeroPad2d((0,1,0,1)) = pad
            // right/bottom by 1, then conv with pad 0. ggml_pad end-pads ne0/ne1.
            x = ggml_pad(ctx, inp, d->padR - d->padL, d->padB - d->padT, 0, 0);
            p0 = 0; p1 = 0;
        }
        ggml_tensor* conv = ggml_conv_2d(ctx, ker, x, sW, sH, p0, p1, 1, 1);  // [OW,OH,OC,1]
        if (bias) conv = ggml_add(ctx, conv, ggml_reshape_4d(ctx, bias, 1, 1, OC, 1));

        const int OW = static_cast<int>(conv->ne[0]), OH = static_cast<int>(conv->ne[1]);
        ggml_tensor* out = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, OW, OH, OC, 1);
        ggml_tensor* output = ggml_cpy(ctx, conv, out);
        ggml_set_output(output);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(graph, output);

        // VAE conv weights are managed float[] with unstable addresses (no benefit from
        // the host-pointer cacheable cache, and reuse would be unsafe), so upload fresh.
        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr) { set_last_error("Conv2d: buffer alloc failed (im2col OOM?)."); return 0; }

        host_read_barrier();
        ggml_backend_tensor_set(ker, d->weight, 0, static_cast<std::size_t>(d->weight_bytes));
        ggml_backend_tensor_set(inp, d->input, 0, static_cast<std::size_t>(W) * H * C * sizeof(float));
        if (bias) ggml_backend_tensor_set(bias, d->bias, 0, static_cast<std::size_t>(OC) * sizeof(float));

        if (ggml_backend_graph_compute(g_backend, graph) != GGML_STATUS_SUCCESS)
        { set_last_error("Conv2d: graph compute failed."); return 0; }
        // Synchronous readback. The VAE runs this conv as a long C# chain (each conv's
        // output is the next conv's input), so the result MUST be on the host before this
        // returns. The async finalize_compute_with_download path only QUEUES a non-blocking
        // ggml_backend_tensor_get_async on Metal async mode and marks pending; the C# caller
        // reads d->output immediately (no host_read_barrier between native calls), so every
        // VAE conv layer consumed STALE/uninitialized data — cascading into the garbled/gray
        // decode that was long misdiagnosed as Q2_K quantization. CUDA/CPU were unaffected
        // (finalize takes the synchronous branch off Metal-async). Match the attn/fused-block
        // kernels and drain synchronously here.
        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(out, d->output, 0, static_cast<std::size_t>(OW) * OH * OC * sizeof(float));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Conv2d: unknown error."); return 0; }
}

struct TSGgmlQwenImageModMlpDesc
{
    // in/out residual stream
    void* x;                 // [dim * seq] F32
    // per-token modulation (host-precomputed), [dim * seq] F32 each
    void* scale_plus1;       // (1 + scale)
    void* shift;
    void* gate;
    // weights
    void* net0_w; int net0_type; std::int64_t net0_ne0, net0_ne1, net0_bytes;  // [dim, ff]
    void* net0_b;            // [ff] F32
    void* net2_w; int net2_type; std::int64_t net2_ne0, net2_ne1, net2_bytes;  // [ff, dim]
    void* net2_b;            // [dim] F32
    // shapes
    std::int32_t struct_bytes;
    std::int32_t dim;        // 3072
    std::int32_t ff;         // 12288
    std::int32_t seq;
    float eps;
};

TSG_EXPORT int TSGgml_QwenImageModMlp(const TSGgmlQwenImageModMlpDesc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwenImageModMlpDesc)))
        {
            set_last_error("QwenImageModMlp: bad descriptor.");
            return 0;
        }
        if (!ensure_backend()) return 0;

        const int dim = d->dim, ff = d->ff, seq = d->seq;

        PooledContextHandle context;
        if (!context.init(32 * 1024 * 1024))
        {
            set_last_error("QwenImageModMlp: context alloc failed.");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_tensor* x_t       = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, seq);
        ggml_tensor* out_t     = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, seq);
        ggml_tensor* scale1_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, seq);
        ggml_tensor* shift_t   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, seq);
        ggml_tensor* gate_t    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, seq);
        ggml_tensor* net0_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->net0_type), d->net0_ne0, d->net0_ne1);
        ggml_tensor* net0_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ff);
        ggml_tensor* net2_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(d->net2_type), d->net2_ne0, d->net2_ne1);
        ggml_tensor* net2_b    = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);

        // LayerNorm (no affine) over ne[0] = features.
        ggml_tensor* normed = ggml_norm(ctx, x_t, d->eps);
        // modulate: normed * (1+scale) + shift
        ggml_tensor* modulated = ggml_add(ctx, ggml_mul(ctx, normed, scale1_t), shift_t);
        // overflow-safe activation pre-scaling for the quantized matmuls (q8_1 FP16-sum
        // overflows for large GELU activations; scale-invariant in precision).
        const float K = 1024.0f;
        // net0: [dim,ff] weight, modulated [dim,seq] -> [ff,seq]; + bias (broadcast)
        ggml_tensor* h = ggml_add(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, net0_w, ggml_scale(ctx, modulated, 1.0f / K)), K), net0_b);
        h = ggml_gelu(ctx, h);
        // net2: [ff,dim] weight, h [ff,seq] -> [dim,seq]; + bias
        ggml_tensor* mlp = ggml_add(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, net2_w, ggml_scale(ctx, h, 1.0f / K)), K), net2_b);
        // gated residual: x + gate * mlp
        ggml_tensor* res = ggml_add(ctx, x_t, ggml_mul(ctx, mlp, gate_t));
        ggml_tensor* output = ggml_cpy(ctx, res, out_t);
        ggml_set_output(output);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 1024, false);
        ggml_build_forward_expand(graph, output);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* d; std::size_t b; };
        std::vector<HostBinding> uploads;
        std::vector<BufferHandle> ephem;

        auto bind = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cache) {
            if (t == nullptr || data == nullptr) return;
            if (cache && bytes >= 4096) {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs)) {
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS) {
                        if (needs) uploads.push_back({t, data, bytes});
                        return;
                    }
                    invalidate_cached_buffer(data);
                }
            }
            uploads.push_back({t, data, bytes});
        };

        bind(net0_w, d->net0_w, static_cast<std::size_t>(d->net0_bytes), true);
        bind(net2_w, d->net2_w, static_cast<std::size_t>(d->net2_bytes), true);
        bind(net0_b, d->net0_b, static_cast<std::size_t>(ff) * sizeof(float), true);
        bind(net2_b, d->net2_b, static_cast<std::size_t>(dim) * sizeof(float), true);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr) { set_last_error("QwenImageModMlp: buffer alloc failed."); return 0; }

        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
        const std::size_t actBytes = static_cast<std::size_t>(dim) * seq * sizeof(float);
        ggml_backend_tensor_set(x_t, d->x, 0, actBytes);
        ggml_backend_tensor_set(scale1_t, d->scale_plus1, 0, actBytes);
        ggml_backend_tensor_set(shift_t, d->shift, 0, actBytes);
        ggml_backend_tensor_set(gate_t, d->gate, 0, actBytes);

        if (ggml_backend_graph_compute(g_backend, graph) != GGML_STATUS_SUCCESS)
        {
            set_last_error("QwenImageModMlp: graph compute failed.");
            return 0;
        }
        // Synchronous readback (matches TSGgml_QwenImageJointAttn / TSGgml_QwenImageBlock).
        // The async finalize_compute_with_download path returns stale data on Metal async mode:
        // it queues a non-blocking ggml_backend_tensor_get_async + marks pending, but the C#
        // caller reads the host buffer immediately (no host_read_barrier between), so the 3-call
        // MLP sub-layer saw uninitialized/stale output (cosine ~0.12 vs the managed reference).
        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(out_t, d->x, 0, actBytes);
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("QwenImageModMlp: unknown error."); return 0; }
}

// ============================================================================
// Fused joint (double-stream) attention sub-layer. Per stream:
//   normed     = layernorm(x)
//   modulated  = normed * (1+scale) + shift
//   q/k/v      = proj(modulated) + bias  (img: to_q/k/v ; txt: add_q/k/v)
//   q,k        = qk_rmsnorm(per head, *norm_w) then interleaved RoPE
// joint attention over concat[txt, q ; img, q] (bidirectional, no mask), split back,
//   out        = out_proj(attn) + bias
//   x_out      = x + gate * out
// scale/shift/gate precomputed on host; RoPE cos/sin passed in the interleaved-
// duplicated [head_dim, seq] layout (cos[2i]=cos[2i+1]=cos(angle_i)).
// ============================================================================
struct TSGImgAttnW { void* w; int type; std::int64_t ne0, ne1, bytes; void* b; };

struct TSGgmlQwenImageJointAttnDesc
{
    void* img; void* txt;
    void* img_scale1; void* img_shift; void* img_gate;
    void* txt_scale1; void* txt_shift; void* txt_gate;
    void* img_cos; void* img_sin;
    void* txt_cos; void* txt_sin;
    TSGImgAttnW to_q, to_k, to_v, to_out;       // img stream
    TSGImgAttnW add_q, add_k, add_v, to_add_out; // txt stream
    void* norm_q; void* norm_k; void* norm_aq; void* norm_ak;  // [head_dim]
    std::int32_t struct_bytes, dim, heads, head_dim, img_seq, txt_seq;
    float eps;
};

namespace {

// LayerNorm(x) * (1+scale) + shift   (x, scale1, shift all [dim, seq])
ggml_tensor* qi_mod_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* scale1, ggml_tensor* shift, float eps)
{
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), scale1), shift);
}

// Quantized matmul with overflow-safe activation pre-scaling. ggml quantizes the
// activation to q8_1, whose per-block FP16 sum overflows for large activations (this
// DiT's GELU / attention outputs reach thousands). Scaling the activation by 1/K
// before the matmul (and the result by K) keeps that FP16 sum in range; q8_1 is
// scale-invariant in precision (the per-block scale adapts), so this is exact.
constexpr float QI_MM_SCALE = 1024.0f;
ggml_tensor* qi_mm(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x)
{
    ggml_tensor* xs = ggml_scale(ctx, x, 1.0f / QI_MM_SCALE);
    return ggml_scale(ctx, ggml_mul_mat(ctx, w, xs), QI_MM_SCALE);
}

ggml_tensor* qi_lin_bias(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, ggml_tensor* b)
{
    ggml_tensor* o = qi_mm(ctx, w, x);
    return b ? ggml_add(ctx, o, b) : o;
}

// per-head RMSNorm over head_dim then * weight. in: [head_dim, heads, seq]
ggml_tensor* qi_qk_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, int head_dim, int heads, int seq, float eps)
{
    ggml_tensor* flat = ggml_reshape_2d(ctx, x, head_dim, heads * seq);
    ggml_tensor* n = ggml_mul(ctx, ggml_rms_norm(ctx, flat, eps), w);
    return ggml_reshape_3d(ctx, n, head_dim, heads, seq);
}

// interleaved RoPE: x [head_dim, heads, seq]; cos/sin [head_dim, seq] (duplicated per pair).
ggml_tensor* qi_rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cosf, ggml_tensor* sinf, int head_dim, int heads, int seq)
{
    ggml_tensor* x4 = ggml_reshape_4d(ctx, x, 2, head_dim / 2, heads, seq);
    ggml_tensor* even = ggml_view_4d(ctx, x4, 1, head_dim / 2, heads, seq, x4->nb[1], x4->nb[2], x4->nb[3], 0);
    ggml_tensor* odd  = ggml_view_4d(ctx, x4, 1, head_dim / 2, heads, seq, x4->nb[1], x4->nb[2], x4->nb[3], x4->nb[0]);
    ggml_tensor* rot4 = ggml_concat(ctx, ggml_neg(ctx, ggml_cont(ctx, odd)), ggml_cont(ctx, even), 0);
    ggml_tensor* rot  = ggml_reshape_3d(ctx, rot4, head_dim, heads, seq);
    ggml_tensor* cos3 = ggml_reshape_3d(ctx, cosf, head_dim, 1, seq);
    ggml_tensor* sin3 = ggml_reshape_3d(ctx, sinf, head_dim, 1, seq);
    return ggml_add(ctx, ggml_mul(ctx, x, cos3), ggml_mul(ctx, rot, sin3));
}

} // namespace

TSG_EXPORT int TSGgml_QwenImageJointAttn(const TSGgmlQwenImageJointAttnDesc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwenImageJointAttnDesc)))
        { set_last_error("QwenImageJointAttn: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        const int dim = d->dim, heads = d->heads, hd = d->head_dim;
        const int iseq = d->img_seq, tseq = d->txt_seq, total = iseq + tseq;
        const float eps = d->eps, scale = 1.0f / std::sqrt(static_cast<float>(hd));

        PooledContextHandle context;
        if (!context.init(32 * 1024 * 1024)) { set_last_error("QwenImageJointAttn: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        ggml_tensor* img_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        ggml_tensor* txt_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        ggml_tensor* i_out  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        ggml_tensor* t_out  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        ggml_tensor* i_s1   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        ggml_tensor* i_sh   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        ggml_tensor* i_g    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        ggml_tensor* t_s1   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        ggml_tensor* t_sh   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        ggml_tensor* t_g    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        ggml_tensor* i_cos  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, iseq);
        ggml_tensor* i_sin  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, iseq);
        ggml_tensor* t_cos  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, tseq);
        ggml_tensor* t_sin  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, tseq);

        auto declW = [&](const TSGImgAttnW& s, ggml_tensor*& w, ggml_tensor*& b) {
            w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(s.type), s.ne0, s.ne1);
            b = s.b ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim) : nullptr;
        };
        ggml_tensor *toQw,*toQb,*toKw,*toKb,*toVw,*toVb,*toOw,*toOb;
        ggml_tensor *aQw,*aQb,*aKw,*aKb,*aVw,*aVb,*aOw,*aOb;
        declW(d->to_q,toQw,toQb); declW(d->to_k,toKw,toKb); declW(d->to_v,toVw,toVb); declW(d->to_out,toOw,toOb);
        declW(d->add_q,aQw,aQb); declW(d->add_k,aKw,aKb); declW(d->add_v,aVw,aVb); declW(d->to_add_out,aOw,aOb);
        ggml_tensor* nQ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        ggml_tensor* nK = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        ggml_tensor* nAQ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        ggml_tensor* nAK = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);

        // --- modulate ---
        ggml_tensor* imgMod = qi_mod_norm(ctx, img_t, i_s1, i_sh, eps);
        ggml_tensor* txtMod = qi_mod_norm(ctx, txt_t, t_s1, t_sh, eps);

        // --- projections + reshape to heads ---
        ggml_tensor* iq = ggml_reshape_3d(ctx, qi_lin_bias(ctx, toQw, imgMod, toQb), hd, heads, iseq);
        ggml_tensor* ik = ggml_reshape_3d(ctx, qi_lin_bias(ctx, toKw, imgMod, toKb), hd, heads, iseq);
        ggml_tensor* iv = ggml_reshape_3d(ctx, qi_lin_bias(ctx, toVw, imgMod, toVb), hd, heads, iseq);
        ggml_tensor* tq = ggml_reshape_3d(ctx, qi_lin_bias(ctx, aQw, txtMod, aQb), hd, heads, tseq);
        ggml_tensor* tk = ggml_reshape_3d(ctx, qi_lin_bias(ctx, aKw, txtMod, aKb), hd, heads, tseq);
        ggml_tensor* tv = ggml_reshape_3d(ctx, qi_lin_bias(ctx, aVw, txtMod, aVb), hd, heads, tseq);

        // --- QK norm + RoPE ---
        iq = qi_rope(ctx, qi_qk_norm(ctx, iq, nQ, hd, heads, iseq, eps), i_cos, i_sin, hd, heads, iseq);
        ik = qi_rope(ctx, qi_qk_norm(ctx, ik, nK, hd, heads, iseq, eps), i_cos, i_sin, hd, heads, iseq);
        tq = qi_rope(ctx, qi_qk_norm(ctx, tq, nAQ, hd, heads, tseq, eps), t_cos, t_sin, hd, heads, tseq);
        tk = qi_rope(ctx, qi_qk_norm(ctx, tk, nAK, hd, heads, tseq, eps), t_cos, t_sin, hd, heads, tseq);

        // --- concat [txt, img] along seq (ne2) ---
        ggml_tensor* q = ggml_concat(ctx, tq, iq, 2);   // [hd, heads, total]
        ggml_tensor* k = ggml_concat(ctx, tk, ik, 2);
        ggml_tensor* v = ggml_concat(ctx, tv, iv, 2);

        // --- attention (bidirectional, no mask). layout [hd, total, heads] ---
        ggml_tensor* q_attn = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor* k_attn = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor* v_attn = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        ggml_tensor* scores = ggml_mul_mat(ctx, k_attn, q_attn);  // [total, total, heads]
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
        ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);
        ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_attn, 1, 0, 2, 3)); // [total, hd, heads]
        ggml_tensor* attn = ggml_mul_mat(ctx, v_perm, probs);    // [hd, total, heads]
        ggml_tensor* attn_perm = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3)); // [hd, heads, total]
        ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_perm, dim, total);   // [dim, total]

        // --- split [txt, img] ---
        ggml_tensor* txt_attn = ggml_cont(ctx, ggml_view_2d(ctx, attn_flat, dim, tseq, attn_flat->nb[1], 0));
        ggml_tensor* img_attn = ggml_cont(ctx, ggml_view_2d(ctx, attn_flat, dim, iseq, attn_flat->nb[1],
                                                            static_cast<std::size_t>(tseq) * attn_flat->nb[1]));

        // --- out proj + gated residual ---
        ggml_tensor* img_o = qi_lin_bias(ctx, toOw, img_attn, toOb);
        ggml_tensor* txt_o = qi_lin_bias(ctx, aOw, txt_attn, aOb);
        ggml_tensor* img_res = ggml_add(ctx, img_t, ggml_mul(ctx, img_o, i_g));
        ggml_tensor* txt_res = ggml_add(ctx, txt_t, ggml_mul(ctx, txt_o, t_g));
        ggml_tensor* oi = ggml_cpy(ctx, img_res, i_out); ggml_set_output(oi);
        ggml_tensor* ot = ggml_cpy(ctx, txt_res, t_out); ggml_set_output(ot);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(graph, oi);
        ggml_build_forward_expand(graph, ot);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HB { ggml_tensor* t; void* d; std::size_t b; };
        std::vector<HB> uploads;
        auto bindW = [&](ggml_tensor* w, ggml_tensor* b, const TSGImgAttnW& s) {
            if (w && s.w) {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, w, s.w, static_cast<std::size_t>(s.bytes), buf, addr, needs)
                    && ggml_backend_tensor_alloc(buf, w, addr) == GGML_STATUS_SUCCESS) {
                    if (needs) uploads.push_back({w, s.w, static_cast<std::size_t>(s.bytes)});
                } else uploads.push_back({w, s.w, static_cast<std::size_t>(s.bytes)});
            }
            if (b && s.b) uploads.push_back({b, s.b, static_cast<std::size_t>(dim) * sizeof(float)});
        };
        bindW(toQw,toQb,d->to_q); bindW(toKw,toKb,d->to_k); bindW(toVw,toVb,d->to_v); bindW(toOw,toOb,d->to_out);
        bindW(aQw,aQb,d->add_q); bindW(aKw,aKb,d->add_k); bindW(aVw,aVb,d->add_v); bindW(aOw,aOb,d->to_add_out);
        auto bind1 = [&](ggml_tensor* t, void* dd) { if (t && dd) uploads.push_back({t, dd, static_cast<std::size_t>(hd) * sizeof(float)}); };
        bind1(nQ, d->norm_q); bind1(nK, d->norm_k); bind1(nAQ, d->norm_aq); bind1(nAK, d->norm_ak);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr) { set_last_error("QwenImageJointAttn: buffer alloc failed."); return 0; }
        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
        auto setF = [&](ggml_tensor* t, void* dd, int rows, int seq) { ggml_backend_tensor_set(t, dd, 0, static_cast<std::size_t>(rows) * seq * sizeof(float)); };
        setF(img_t, d->img, dim, iseq); setF(txt_t, d->txt, dim, tseq);
        setF(i_s1, d->img_scale1, dim, iseq); setF(i_sh, d->img_shift, dim, iseq); setF(i_g, d->img_gate, dim, iseq);
        setF(t_s1, d->txt_scale1, dim, tseq); setF(t_sh, d->txt_shift, dim, tseq); setF(t_g, d->txt_gate, dim, tseq);
        setF(i_cos, d->img_cos, hd, iseq); setF(i_sin, d->img_sin, hd, iseq);
        setF(t_cos, d->txt_cos, hd, tseq); setF(t_sin, d->txt_sin, hd, tseq);

        if (ggml_backend_graph_compute(g_backend, graph) != GGML_STATUS_SUCCESS)
        { set_last_error("QwenImageJointAttn: graph compute failed."); return 0; }
        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(i_out, d->img, 0, static_cast<std::size_t>(dim) * iseq * sizeof(float));
        ggml_backend_tensor_get(t_out, d->txt, 0, static_cast<std::size_t>(dim) * tseq * sizeof(float));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("QwenImageJointAttn: unknown error."); return 0; }
}

// ============================================================================
// Whole double-stream block in ONE graph: joint attention sub-layer + both
// (img/txt) modulated GEGLU MLP sub-layers. Folds the 3 per-block native calls
// into a single dispatch — the attention residual feeds straight into the MLP
// on-device with no intermediate img/txt host round-trip.
//
// Reuses the verified qi_* helpers (same math as the two split kernels). The
// graph-building bodies are factored into qi_build_attn / qi_build_mlp so the
// upcoming whole-DiT graph can stack them 60x.
// ============================================================================
struct TSGgmlQwenImageBlockDesc
{
    void* img; void* txt;                                  // in/out [dim*seq]
    void* i_s1a; void* i_sha; void* i_ga;                  // attn modulation (mod index 0)
    void* t_s1a; void* t_sha; void* t_ga;
    void* i_s1m; void* i_shm; void* i_gm;                  // mlp modulation (mod index 1)
    void* t_s1m; void* t_shm; void* t_gm;
    void* i_cos; void* i_sin; void* t_cos; void* t_sin;
    TSGImgAttnW to_q, to_k, to_v, to_out;
    TSGImgAttnW add_q, add_k, add_v, to_add_out;
    void* norm_q; void* norm_k; void* norm_aq; void* norm_ak;
    TSGImgAttnW i_net0, i_net2, t_net0, t_net2;            // mlp weights (+bias in .b)
    std::int32_t struct_bytes, dim, heads, head_dim, ff, img_seq, txt_seq;
    float eps;
};

namespace {

// Tensors that hold all the inputs/weights for one block subgraph (declared once,
// bound to device buffers). Activations (img/txt/mod/rope) are set per invocation;
// weights are bound as cacheable (resident) leaves.
struct QiBlockTensors
{
    ggml_tensor *img, *txt;
    ggml_tensor *i_s1a, *i_sha, *i_ga, *t_s1a, *t_sha, *t_ga;
    ggml_tensor *i_s1m, *i_shm, *i_gm, *t_s1m, *t_shm, *t_gm;
    ggml_tensor *i_cos, *i_sin, *t_cos, *t_sin;
    ggml_tensor *toQw, *toQb, *toKw, *toKb, *toVw, *toVb, *toOw, *toOb;
    ggml_tensor *aQw, *aQb, *aKw, *aKb, *aVw, *aVb, *aOw, *aOb;
    ggml_tensor *nQ, *nK, *nAQ, *nAK;
    ggml_tensor *iN0w, *iN0b, *iN2w, *iN2b, *tN0w, *tN0b, *tN2w, *tN2b;
    ggml_tensor *mask = nullptr;   // [total_pad, total] F16 flash mask (null = materialized attn)
};

// ggml-cuda's flash-attention tiles K over FATTN_KQ_STRIDE-wide blocks; the n_kv
// (K->ne[1]) must be padded to that stride and an F16 mask must -inf the padding,
// or the final partial tile reads past valid K and poisons the softmax -> NaN/black
// (a NULL mask + unpadded K is the bug). Matches the proven decode/verify setup.
constexpr int kQiKvStride = 256;

// Stable host F16 mask for the bidirectional all-valid case: SQUARE [total_pad, total_pad]
// with 0 for kv<total and -inf for kv in [total,total_pad), identical for every query
// column. BOTH dims are padded to the KV stride: the large-batch mask-scan optimization
// (flash_attn_mask_to_KV_max) reads the mask in ncols1-wide query tiles up to
// round_up(Q->ne[1], ncols1), so an unpadded n_q makes it read OOB (illegal access).
// Cached per `total` so the device upload (cacheable, keyed by this pointer) happens
// once per shape and is reused across all 60 blocks.
const ggml_fp16_t* qi_attn_mask_host(int total, int& total_pad)
{
    total_pad = ((total + kQiKvStride - 1) / kQiKvStride) * kQiKvStride;
    static std::unordered_map<int, std::vector<ggml_fp16_t>> cache;
    auto it = cache.find(total);
    if (it != cache.end()) return it->second.data();
    std::vector<ggml_fp16_t>& m = cache[total];
    m.assign(static_cast<std::size_t>(total_pad) * total_pad, ggml_fp32_to_fp16(0.0f));
    const ggml_fp16_t ninf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
    for (int q = 0; q < total_pad; q++)
        for (int kv = total; kv < total_pad; kv++)
            m[static_cast<std::size_t>(q) * total_pad + kv] = ninf;
    return m.data();
}

inline bool qi_flash_enabled()
{
    // Default ON now that the ggml-cuda flash path is fixed for this DiT (correct +
    // ~14% faster denoise + frees the O(n^2) scores -> higher res). A backend_supports_op
    // guard falls back to the materialized path on GPUs that can't run it. TS_QWEN_DIT_FLASH=0 to force off.
    static const bool on = []{ const char* e = std::getenv("TS_QWEN_DIT_FLASH"); return e == nullptr || e[0] != '0'; }();
    return on;
}

// Bidirectional joint attention -> [dim, total]. q/k/v are [hd, total, heads]. When a
// (caller-supplied, F16) `mask` [total_pad, total] is given AND flash is enabled, run
// ggml_flash_attn_ext (no [total,total] scores materialized — fits high-res + faster);
// otherwise the explicit scores+softmax reference path (always correct; CPU + non-flash).
ggml_tensor* qi_attention(ggml_context* ctx, ggml_tensor* q_attn, ggml_tensor* k_attn, ggml_tensor* v_attn,
                          ggml_tensor* mask, int dim, int total, float scale)
{
    if (qi_flash_enabled() && mask != nullptr)
    {
        // Pad n_kv (K/V) to the KV stride; the mask -inf's the padded keys. Pass F32
        // K/V: the MMA/tile/wmma kernels do their OWN F16 conversion into the extra dst
        // workspace that get_alloc_size reserves for an F32 K/V — pre-casting to F16
        // mis-sizes that workspace and the kernel writes out of bounds (illegal access).
        const int total_pad = static_cast<int>(mask->ne[0]);
        ggml_tensor* kpad = (total_pad > total) ? ggml_pad(ctx, k_attn, 0, total_pad - total, 0, 0) : k_attn;
        ggml_tensor* vpad = (total_pad > total) ? ggml_pad(ctx, v_attn, 0, total_pad - total, 0, 0) : v_attn;
        // The MMA kernel converts K/V to F16 internally; at later denoise timesteps the
        // V magnitudes exceed F16's 65504 limit -> inf -> all-NaN attention (the F32
        // materialized path is unaffected). Scale K/V down by S before the kernel (so
        // K/V/S fits F16) and the result back up by S. Scores stay exact: pass scale*S
        // so (q·(K/S))·(scale·S) == q·K·scale; output softmax·(V/S) is rescaled by S.
        constexpr float QI_FA_SCALE = 16.0f;
        ggml_tensor* ks = ggml_scale(ctx, kpad, 1.0f / QI_FA_SCALE);
        ggml_tensor* vs = ggml_scale(ctx, vpad, 1.0f / QI_FA_SCALE);
        ggml_tensor* faop = ggml_flash_attn_ext(ctx, q_attn, ks, vs, mask, scale * QI_FA_SCALE, 0.0f, 0.0f); // [hd, heads, total]
        ggml_flash_attn_ext_set_prec(faop, GGML_PREC_F32);
        // Only take the flash path if the active backend actually supports this op on
        // this GPU — else an unsupported op runs anyway and crashes (illegal access).
        // The decode/verify kernels guard the same way and fall back to materialized.
        if (backend_supports_op(faop))
        {
            static bool logged = false;
            if (!logged) { std::fprintf(stderr, "[qwen-image] flash-attn ENGAGED (supported)\n"); logged = true; }
            ggml_tensor* fa = ggml_scale(ctx, faop, QI_FA_SCALE);
            return ggml_reshape_2d(ctx, fa, dim, total);
        }
        static bool loggedNo = false;
        if (!loggedNo) { std::fprintf(stderr, "[qwen-image] flash-attn NOT supported on this backend -> materialized fallback\n"); loggedNo = true; }
        // fall through to the materialized path below
    }
    ggml_tensor* scores = ggml_mul_mat(ctx, k_attn, q_attn);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    ggml_tensor* probs = ggml_soft_max_ext(ctx, scores, nullptr, scale, 0.0f);
    ggml_tensor* v_perm = ggml_cont(ctx, ggml_permute(ctx, v_attn, 1, 0, 2, 3));
    ggml_tensor* attn = ggml_mul_mat(ctx, v_perm, probs);
    ggml_tensor* attn_perm = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    return ggml_reshape_2d(ctx, attn_perm, dim, total);
}

// Attention sub-layer subgraph. Inputs: img/txt residual streams (+ their attn
// modulation + rope). Returns the gated-residual outputs (img1/txt1).
void qi_build_attn(ggml_context* ctx, const QiBlockTensors& t,
                   int dim, int heads, int hd, int iseq, int tseq, float eps,
                   ggml_tensor*& img1, ggml_tensor*& txt1)
{
    const int total = iseq + tseq;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    ggml_tensor* imgMod = qi_mod_norm(ctx, t.img, t.i_s1a, t.i_sha, eps);
    ggml_tensor* txtMod = qi_mod_norm(ctx, t.txt, t.t_s1a, t.t_sha, eps);

    ggml_tensor* iq = ggml_reshape_3d(ctx, qi_lin_bias(ctx, t.toQw, imgMod, t.toQb), hd, heads, iseq);
    ggml_tensor* ik = ggml_reshape_3d(ctx, qi_lin_bias(ctx, t.toKw, imgMod, t.toKb), hd, heads, iseq);
    ggml_tensor* iv = ggml_reshape_3d(ctx, qi_lin_bias(ctx, t.toVw, imgMod, t.toVb), hd, heads, iseq);
    ggml_tensor* tq = ggml_reshape_3d(ctx, qi_lin_bias(ctx, t.aQw, txtMod, t.aQb), hd, heads, tseq);
    ggml_tensor* tk = ggml_reshape_3d(ctx, qi_lin_bias(ctx, t.aKw, txtMod, t.aKb), hd, heads, tseq);
    ggml_tensor* tv = ggml_reshape_3d(ctx, qi_lin_bias(ctx, t.aVw, txtMod, t.aVb), hd, heads, tseq);

    iq = qi_rope(ctx, qi_qk_norm(ctx, iq, t.nQ, hd, heads, iseq, eps), t.i_cos, t.i_sin, hd, heads, iseq);
    ik = qi_rope(ctx, qi_qk_norm(ctx, ik, t.nK, hd, heads, iseq, eps), t.i_cos, t.i_sin, hd, heads, iseq);
    tq = qi_rope(ctx, qi_qk_norm(ctx, tq, t.nAQ, hd, heads, tseq, eps), t.t_cos, t.t_sin, hd, heads, tseq);
    tk = qi_rope(ctx, qi_qk_norm(ctx, tk, t.nAK, hd, heads, tseq, eps), t.t_cos, t.t_sin, hd, heads, tseq);

    ggml_tensor* q = ggml_concat(ctx, tq, iq, 2);
    ggml_tensor* k = ggml_concat(ctx, tk, ik, 2);
    ggml_tensor* v = ggml_concat(ctx, tv, iv, 2);

    ggml_tensor* q_attn = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor* k_attn = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor* v_attn = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor* attn_flat = qi_attention(ctx, q_attn, k_attn, v_attn, t.mask, dim, total, scale);

    ggml_tensor* txt_attn = ggml_cont(ctx, ggml_view_2d(ctx, attn_flat, dim, tseq, attn_flat->nb[1], 0));
    ggml_tensor* img_attn = ggml_cont(ctx, ggml_view_2d(ctx, attn_flat, dim, iseq, attn_flat->nb[1],
                                                        static_cast<std::size_t>(tseq) * attn_flat->nb[1]));

    ggml_tensor* img_o = qi_lin_bias(ctx, t.toOw, img_attn, t.toOb);
    ggml_tensor* txt_o = qi_lin_bias(ctx, t.aOw, txt_attn, t.aOb);
    img1 = ggml_add(ctx, t.img, ggml_mul(ctx, img_o, t.i_ga));
    txt1 = ggml_add(ctx, t.txt, ggml_mul(ctx, txt_o, t.t_ga));
}

// One modulated GEGLU MLP sub-layer. x: residual stream; returns gated residual.
ggml_tensor* qi_build_mlp(ggml_context* ctx, ggml_tensor* x, ggml_tensor* s1, ggml_tensor* sh, ggml_tensor* g,
                          ggml_tensor* n0w, ggml_tensor* n0b, ggml_tensor* n2w, ggml_tensor* n2b, float eps)
{
    ggml_tensor* mod = qi_mod_norm(ctx, x, s1, sh, eps);
    ggml_tensor* h = qi_mm(ctx, n0w, mod);
    if (n0b) h = ggml_add(ctx, h, n0b);
    h = ggml_gelu(ctx, h);
    ggml_tensor* mlp = qi_mm(ctx, n2w, h);
    if (n2b) mlp = ggml_add(ctx, mlp, n2b);
    return ggml_add(ctx, x, ggml_mul(ctx, mlp, g));
}

// ----------------------------------------------------------------------------
// Persistent + CUDA-graph-captured block path. The DiT loop is launch-bound on
// WDDM (measured ~62% util / ~60 W: many small ops, the GPU starved between
// kernel submits) — call-fusion alone didn't help because it cut allocs/syncs,
// not op-launches. The fix is CUDA-graph capture: keep ONE block graph per
// (iseq,tseq) shape alive with stable addresses (alloc_ctx_tensors) so ggml-cuda
// captures it after warmup and replays the whole block as one graph.
//
// All 60 blocks of a forward share the same shape, so they reuse one cached graph
// (warmup completes at block 2, blocks 3..60 replay). Weights can't stay at their
// distinct resident addresses (capture bakes addresses), so each block's quantized
// weights are uploaded into the graph's fixed weight slots before its replay
// (~73 MB/block; the launch-overhead it removes is far larger).
struct QiBlockPersist
{
    bool valid = false;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph* graph = nullptr;
    QiBlockTensors t{};
    ggml_tensor* i_out = nullptr;
    ggml_tensor* t_out = nullptr;
    int dim = 0, heads = 0, hd = 0, ff = 0, iseq = 0, tseq = 0;
    int wtype[12] = {0};   // per-weight quant types — this is a MIXED-QUANT (unsloth
                           // dynamic) model: different blocks use different quant types
                           // for the same weight, so the type signature is part of the key.

    bool matches(const TSGgmlQwenImageBlockDesc* d) const
    {
        if (!(valid && dim == d->dim && heads == d->heads && hd == d->head_dim &&
              ff == d->ff && iseq == d->img_seq && tseq == d->txt_seq))
            return false;
        const TSGImgAttnW* w[12] = { &d->to_q, &d->to_k, &d->to_v, &d->to_out,
                                     &d->add_q, &d->add_k, &d->add_v, &d->to_add_out,
                                     &d->i_net0, &d->i_net2, &d->t_net0, &d->t_net2 };
        for (int i = 0; i < 12; i++) if (wtype[i] != w[i]->type) return false;
        return true;
    }
    void reset()
    {
        if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        graph = nullptr; valid = false; i_out = t_out = nullptr;
        dim = heads = hd = ff = iseq = tseq = 0;
    }
};

// Up to ~5 quant profiles x 2 true-CFG txt lengths coexist; cap the ring with
// headroom so a resolution change can't leak graphs.
constexpr int kQiBlockCacheMax = 12;
QiBlockPersist g_qiBlocks[kQiBlockCacheMax];
int g_qiBlockRR = 0;

QiBlockPersist* qi_block_find(const TSGgmlQwenImageBlockDesc* d)
{
    for (auto& e : g_qiBlocks) if (e.matches(d)) return &e;
    return nullptr;
}

// Build a fresh persistent block graph for d's shape (stable addresses for capture).
QiBlockPersist* qi_block_build(const TSGgmlQwenImageBlockDesc* d)
{
    const int dim = d->dim, heads = d->heads, hd = d->head_dim, ff = d->ff;
    const int iseq = d->img_seq, tseq = d->txt_seq;
    const float eps = d->eps;

    // graph metadata + tensor headers (data lives in the backend buffer).
    const std::size_t meta = ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(8192, false) + (1 << 20);
    ggml_init_params ip{ meta, nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    if (ctx == nullptr) return nullptr;

    QiBlockTensors t{};
    t.img = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
    t.txt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
    ggml_tensor* i_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
    ggml_tensor* t_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
    t.i_s1a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
    t.i_sha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
    t.i_ga  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
    t.t_s1a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
    t.t_sha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
    t.t_ga  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
    t.i_s1m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
    t.i_shm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
    t.i_gm  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
    t.t_s1m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
    t.t_shm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
    t.t_gm  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
    t.i_cos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, iseq);
    t.i_sin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, iseq);
    t.t_cos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, tseq);
    t.t_sin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, tseq);

    // Flash-attention mask (bidirectional all-valid + KV-stride padding). Wiring flash
    // into the CAPTURED path removes the O(total^2) materialized scores, so capture (and
    // its launch-overhead elimination) extends to high-resolution token counts instead of
    // OOMing. The mask is constant per shape, lives in the entry's own buffer (stable
    // address for capture), and is uploaded ONCE after buffer alloc below.
    const ggml_fp16_t* maskHost = nullptr; int total_pad = 0;
    if (qi_flash_enabled())
    {
        maskHost = qi_attn_mask_host(iseq + tseq, total_pad);
        t.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, total_pad, total_pad);
    }

    auto declW = [&](const TSGImgAttnW& s, ggml_tensor*& w, ggml_tensor*& b, int blen) {
        w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(s.type), s.ne0, s.ne1);
        b = s.b ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, blen) : nullptr;
    };
    declW(d->to_q, t.toQw, t.toQb, dim); declW(d->to_k, t.toKw, t.toKb, dim);
    declW(d->to_v, t.toVw, t.toVb, dim); declW(d->to_out, t.toOw, t.toOb, dim);
    declW(d->add_q, t.aQw, t.aQb, dim); declW(d->add_k, t.aKw, t.aKb, dim);
    declW(d->add_v, t.aVw, t.aVb, dim); declW(d->to_add_out, t.aOw, t.aOb, dim);
    t.nQ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
    t.nK = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
    t.nAQ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
    t.nAK = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
    declW(d->i_net0, t.iN0w, t.iN0b, ff);  declW(d->i_net2, t.iN2w, t.iN2b, dim);
    declW(d->t_net0, t.tN0w, t.tN0b, ff);  declW(d->t_net2, t.tN2w, t.tN2b, dim);

    ggml_tensor *img1, *txt1;
    qi_build_attn(ctx, t, dim, heads, hd, iseq, tseq, eps, img1, txt1);
    ggml_tensor* img2 = qi_build_mlp(ctx, img1, t.i_s1m, t.i_shm, t.i_gm, t.iN0w, t.iN0b, t.iN2w, t.iN2b, eps);
    ggml_tensor* txt2 = qi_build_mlp(ctx, txt1, t.t_s1m, t.t_shm, t.t_gm, t.tN0w, t.tN0b, t.tN2w, t.tN2b, eps);
    ggml_tensor* oi = ggml_cpy(ctx, img2, i_out); ggml_set_output(oi);
    ggml_tensor* ot = ggml_cpy(ctx, txt2, t_out); ggml_set_output(ot);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, oi);
    ggml_build_forward_expand(graph, ot);

    // Only build the captured entry if it fits dedicated VRAM with headroom; otherwise bail
    // to the non-persist path. At high res two single-branch entries (cond/neg txt lengths)
    // would otherwise oversubscribe VRAM and spill to WDDM shared memory (~5x slower). The
    // CFG-batched captured path avoids this by sharing one entry across both branches.
    {
        std::size_t need = 0;
        for (ggml_tensor* tt = ggml_get_first_tensor(ctx); tt != nullptr; tt = ggml_get_next_tensor(ctx, tt))
            need += GGML_PAD(ggml_nbytes(tt), 256);
        ggml_backend_dev_t mdev = ggml_backend_get_device(g_backend);
        std::size_t freeb = 0, totalb = 0;
        if (mdev) ggml_backend_dev_memory(mdev, &freeb, &totalb);
        if (totalb > 0 && freeb < need + static_cast<std::size_t>(768) * 1024 * 1024)
        {
            std::fprintf(stderr, "[qwen-image-block] capture entry needs %zu MiB, free %zu MiB -> non-persist fallback\n",
                         need >> 20, freeb >> 20);
            ggml_free(ctx);
            return nullptr;
        }
    }

    // Stable addresses for capture: each tensor its own slot (no gallocr packing).
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
    if (buffer == nullptr) { ggml_free(ctx); return nullptr; }

    // Upload the (constant per-shape) flash mask once into its stable slot. The mask is
    // never touched again by qi_block_run's per-block weight streaming, so the captured
    // graph replays against valid mask data on every block/step.
    if (t.mask && maskHost)
        ggml_backend_tensor_set(t.mask, maskHost, 0,
                                static_cast<std::size_t>(total_pad) * total_pad * sizeof(ggml_fp16_t));

    // Evict a slot (round-robin) and install.
    QiBlockPersist* e = nullptr;
    for (auto& c : g_qiBlocks) if (!c.valid) { e = &c; break; }
    if (e == nullptr) { e = &g_qiBlocks[g_qiBlockRR]; g_qiBlockRR = (g_qiBlockRR + 1) % kQiBlockCacheMax; e->reset(); }
    e->ctx = ctx; e->buffer = buffer; e->graph = graph; e->t = t;
    e->i_out = i_out; e->t_out = t_out;
    e->dim = dim; e->heads = heads; e->hd = hd; e->ff = ff; e->iseq = iseq; e->tseq = tseq;
    const TSGImgAttnW* wv[12] = { &d->to_q, &d->to_k, &d->to_v, &d->to_out,
                                  &d->add_q, &d->add_k, &d->add_v, &d->to_add_out,
                                  &d->i_net0, &d->i_net2, &d->t_net0, &d->t_net2 };
    for (int i = 0; i < 12; i++) e->wtype[i] = wv[i]->type;
    e->valid = true;
    return e;
}

// Upload this block's weights + activations + modulation + rope into the cached
// graph's fixed slots, run (capture-replay after warmup), read img/txt back.
int qi_block_run(QiBlockPersist* e, const TSGgmlQwenImageBlockDesc* d)
{
    const int dim = e->dim, hd = e->hd, iseq = e->iseq, tseq = e->tseq;
    host_read_barrier();
    bool ok = true;
    auto chk = [&](ggml_tensor* tt, const void* dd, std::size_t bytes, const char* nm) {
        if (!tt || !dd || !ok) return;
        std::size_t cap = ggml_nbytes(tt);
        if (bytes > cap) {
            set_last_error(std::string("QwenImageBlock(persist): oob set ") + nm +
                " bytes=" + std::to_string(bytes) + " cap=" + std::to_string(cap) +
                " ne=[" + std::to_string(tt->ne[0]) + "," + std::to_string(tt->ne[1]) + "]");
            ok = false; return;
        }
        ggml_backend_tensor_set(tt, dd, 0, bytes);
    };
    auto upW = [&](ggml_tensor* w, ggml_tensor* b, const TSGImgAttnW& s, int blen, const char* nm) {
        chk(w, s.w, static_cast<std::size_t>(s.bytes), nm);
        if (b) chk(b, s.b, static_cast<std::size_t>(blen) * sizeof(float), nm);
    };
    QiBlockTensors& t = e->t;
    upW(t.toQw, t.toQb, d->to_q, dim, "to_q"); upW(t.toKw, t.toKb, d->to_k, dim, "to_k");
    upW(t.toVw, t.toVb, d->to_v, dim, "to_v"); upW(t.toOw, t.toOb, d->to_out, dim, "to_out");
    upW(t.aQw, t.aQb, d->add_q, dim, "add_q"); upW(t.aKw, t.aKb, d->add_k, dim, "add_k");
    upW(t.aVw, t.aVb, d->add_v, dim, "add_v"); upW(t.aOw, t.aOb, d->to_add_out, dim, "add_out");
    upW(t.iN0w, t.iN0b, d->i_net0, e->ff, "i_net0"); upW(t.iN2w, t.iN2b, d->i_net2, dim, "i_net2");
    upW(t.tN0w, t.tN0b, d->t_net0, e->ff, "t_net0"); upW(t.tN2w, t.tN2b, d->t_net2, dim, "t_net2");
    auto up1 = [&](ggml_tensor* tt, void* dd, const char* nm) { chk(tt, dd, static_cast<std::size_t>(hd) * sizeof(float), nm); };
    up1(t.nQ, d->norm_q, "nQ"); up1(t.nK, d->norm_k, "nK"); up1(t.nAQ, d->norm_aq, "nAQ"); up1(t.nAK, d->norm_ak, "nAK");
    auto setF = [&](ggml_tensor* tt, void* dd, int rows, int seq, const char* nm) { chk(tt, dd, static_cast<std::size_t>(rows) * seq * sizeof(float), nm); };
    setF(t.img, d->img, dim, iseq, "img"); setF(t.txt, d->txt, dim, tseq, "txt");
    setF(t.i_s1a, d->i_s1a, dim, iseq, "i_s1a"); setF(t.i_sha, d->i_sha, dim, iseq, "i_sha"); setF(t.i_ga, d->i_ga, dim, iseq, "i_ga");
    setF(t.t_s1a, d->t_s1a, dim, tseq, "t_s1a"); setF(t.t_sha, d->t_sha, dim, tseq, "t_sha"); setF(t.t_ga, d->t_ga, dim, tseq, "t_ga");
    setF(t.i_s1m, d->i_s1m, dim, iseq, "i_s1m"); setF(t.i_shm, d->i_shm, dim, iseq, "i_shm"); setF(t.i_gm, d->i_gm, dim, iseq, "i_gm");
    setF(t.t_s1m, d->t_s1m, dim, tseq, "t_s1m"); setF(t.t_shm, d->t_shm, dim, tseq, "t_shm"); setF(t.t_gm, d->t_gm, dim, tseq, "t_gm");
    setF(t.i_cos, d->i_cos, hd, iseq, "i_cos"); setF(t.i_sin, d->i_sin, hd, iseq, "i_sin");
    setF(t.t_cos, d->t_cos, hd, tseq, "t_cos"); setF(t.t_sin, d->t_sin, hd, tseq, "t_sin");
    if (!ok) return 0;

    if (ggml_backend_graph_compute(g_backend, e->graph) != GGML_STATUS_SUCCESS)
    { set_last_error("QwenImageBlock(persist): graph compute failed."); return 0; }
    ggml_backend_synchronize(g_backend);
    ggml_backend_tensor_get(e->i_out, d->img, 0, static_cast<std::size_t>(dim) * iseq * sizeof(float));
    ggml_backend_tensor_get(e->t_out, d->txt, 0, static_cast<std::size_t>(dim) * tseq * sizeof(float));
    return 1;
}

// ----------------------------------------------------------------------------
// Persistent + CUDA-graph-captured CFG-batched block. Capturing the single-branch
// block at high res needs TWO entries per quant profile (the cond/neg txt lengths
// differ), which doubles the capture VRAM and spills to shared memory. Capturing the
// COMBINED block instead keeps ONE entry per profile (both branches in one graph),
// halving the entries so it fits, and folds in the CFG-batch win (one dispatch +
// shared weight upload). This is the high-res launch-bound fix.
struct QiBlockCfgPersist
{
    bool valid = false;
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph* graph = nullptr;
    QiBlockTensors w{};            // shared weight leaves
    QiBlockTensors tc{}, tn{};     // per-branch activations (+ shared weight ptrs)
    ggml_tensor *ic_out = nullptr, *tc_out = nullptr, *in_out = nullptr, *tn_out = nullptr;
    int dim = 0, heads = 0, hd = 0, ff = 0, iseq = 0, tseqc = 0, tseqn = 0;
    int wtype[12] = {0};

    bool matches(const TSGgmlQwenImageBlockDesc* dc, const TSGgmlQwenImageBlockDesc* dn) const
    {
        if (!(valid && dim == dc->dim && heads == dc->heads && hd == dc->head_dim &&
              ff == dc->ff && iseq == dc->img_seq && tseqc == dc->txt_seq && tseqn == dn->txt_seq))
            return false;
        const TSGImgAttnW* wv[12] = { &dc->to_q, &dc->to_k, &dc->to_v, &dc->to_out,
                                      &dc->add_q, &dc->add_k, &dc->add_v, &dc->to_add_out,
                                      &dc->i_net0, &dc->i_net2, &dc->t_net0, &dc->t_net2 };
        for (int i = 0; i < 12; i++) if (wtype[i] != wv[i]->type) return false;
        return true;
    }
    void reset()
    {
        if (buffer) { ggml_backend_buffer_free(buffer); buffer = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        graph = nullptr; valid = false; ic_out = tc_out = in_out = tn_out = nullptr;
        dim = heads = hd = ff = iseq = tseqc = tseqn = 0;
    }
};

constexpr int kQiCfgCacheMax = 8;   // one entry per quant profile (both branches share it)
QiBlockCfgPersist g_qiCfgBlocks[kQiCfgCacheMax];
int g_qiCfgRR = 0;

QiBlockCfgPersist* qi_cfg_find(const TSGgmlQwenImageBlockDesc* dc, const TSGgmlQwenImageBlockDesc* dn)
{
    for (auto& e : g_qiCfgBlocks) if (e.matches(dc, dn)) return &e;
    return nullptr;
}

// Declare the shared weight leaves + the two branches' activations into ctx, build both
// block subgraphs (sharing the weights), and return the graph + tensor handles. Shared by
// the captured build (stable addresses) below. Masks are returned for a one-time upload.
ggml_cgraph* qi_cfg_build_graph(ggml_context* ctx, const TSGgmlQwenImageBlockDesc* dc,
                                const TSGgmlQwenImageBlockDesc* dn,
                                QiBlockTensors& w, QiBlockTensors& tc, QiBlockTensors& tn,
                                ggml_tensor*& ic_out, ggml_tensor*& tc_out,
                                ggml_tensor*& in_out, ggml_tensor*& tn_out,
                                const ggml_fp16_t*& maskC, int& tpadC,
                                const ggml_fp16_t*& maskN, int& tpadN)
{
    const int dim = dc->dim, heads = dc->heads, hd = dc->head_dim, ff = dc->ff;
    const int iseq = dc->img_seq, tseqc = dc->txt_seq, tseqn = dn->txt_seq;
    const float eps = dc->eps;

    auto declW = [&](const TSGImgAttnW& s, ggml_tensor*& wt, ggml_tensor*& bt, int blen) {
        wt = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(s.type), s.ne0, s.ne1);
        bt = s.b ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, blen) : nullptr;
    };
    declW(dc->to_q, w.toQw, w.toQb, dim); declW(dc->to_k, w.toKw, w.toKb, dim);
    declW(dc->to_v, w.toVw, w.toVb, dim); declW(dc->to_out, w.toOw, w.toOb, dim);
    declW(dc->add_q, w.aQw, w.aQb, dim); declW(dc->add_k, w.aKw, w.aKb, dim);
    declW(dc->add_v, w.aVw, w.aVb, dim); declW(dc->to_add_out, w.aOw, w.aOb, dim);
    w.nQ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
    w.nK = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
    w.nAQ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
    w.nAK = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
    declW(dc->i_net0, w.iN0w, w.iN0b, ff);  declW(dc->i_net2, w.iN2w, w.iN2b, dim);
    declW(dc->t_net0, w.tN0w, w.tN0b, ff);  declW(dc->t_net2, w.tN2w, w.tN2b, dim);

    auto declAct = [&](QiBlockTensors& t, int ts, ggml_tensor*& iout, ggml_tensor*& tout,
                       const ggml_fp16_t*& mh, int& tp)
    {
        t.toQw=w.toQw; t.toQb=w.toQb; t.toKw=w.toKw; t.toKb=w.toKb;
        t.toVw=w.toVw; t.toVb=w.toVb; t.toOw=w.toOw; t.toOb=w.toOb;
        t.aQw=w.aQw; t.aQb=w.aQb; t.aKw=w.aKw; t.aKb=w.aKb;
        t.aVw=w.aVw; t.aVb=w.aVb; t.aOw=w.aOw; t.aOb=w.aOb;
        t.nQ=w.nQ; t.nK=w.nK; t.nAQ=w.nAQ; t.nAK=w.nAK;
        t.iN0w=w.iN0w; t.iN0b=w.iN0b; t.iN2w=w.iN2w; t.iN2b=w.iN2b;
        t.tN0w=w.tN0w; t.tN0b=w.tN0b; t.tN2w=w.tN2w; t.tN2b=w.tN2b;
        t.img = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        t.txt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
        iout  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        tout  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
        t.i_s1a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        t.i_sha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        t.i_ga  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        t.t_s1a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
        t.t_sha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
        t.t_ga  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
        t.i_s1m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        t.i_shm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        t.i_gm  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        t.t_s1m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
        t.t_shm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
        t.t_gm  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
        t.i_cos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, iseq);
        t.i_sin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, iseq);
        t.t_cos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, ts);
        t.t_sin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, ts);
        mh = nullptr; tp = 0;
        if (qi_flash_enabled()) { mh = qi_attn_mask_host(iseq + ts, tp); t.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, tp, tp); }
    };
    declAct(tc, tseqc, ic_out, tc_out, maskC, tpadC);
    declAct(tn, tseqn, in_out, tn_out, maskN, tpadN);

    auto buildBranch = [&](QiBlockTensors& t, int ts, ggml_tensor* iout, ggml_tensor* tout,
                           ggml_tensor*& oi, ggml_tensor*& ot)
    {
        ggml_tensor *img1, *txt1;
        qi_build_attn(ctx, t, dim, heads, hd, iseq, ts, eps, img1, txt1);
        ggml_tensor* img2 = qi_build_mlp(ctx, img1, t.i_s1m, t.i_shm, t.i_gm, t.iN0w, t.iN0b, t.iN2w, t.iN2b, eps);
        ggml_tensor* txt2 = qi_build_mlp(ctx, txt1, t.t_s1m, t.t_shm, t.t_gm, t.tN0w, t.tN0b, t.tN2w, t.tN2b, eps);
        oi = ggml_cpy(ctx, img2, iout); ggml_set_output(oi);
        ot = ggml_cpy(ctx, txt2, tout); ggml_set_output(ot);
    };
    ggml_tensor *oic, *otc, *oin, *otn;
    buildBranch(tc, tseqc, ic_out, tc_out, oic, otc);
    buildBranch(tn, tseqn, in_out, tn_out, oin, otn);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 16384, false);
    ggml_build_forward_expand(graph, oic);
    ggml_build_forward_expand(graph, otc);
    ggml_build_forward_expand(graph, oin);
    ggml_build_forward_expand(graph, otn);
    return graph;
}

QiBlockCfgPersist* qi_cfg_build(const TSGgmlQwenImageBlockDesc* dc, const TSGgmlQwenImageBlockDesc* dn)
{
    const std::size_t meta = ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(16384, false) + (1 << 20);
    ggml_init_params ip{ meta, nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    if (ctx == nullptr) return nullptr;

    QiBlockTensors w{}, tc{}, tn{};
    ggml_tensor *ic_out, *tc_out, *in_out, *tn_out;
    const ggml_fp16_t *maskC = nullptr, *maskN = nullptr; int tpadC = 0, tpadN = 0;
    ggml_cgraph* graph = qi_cfg_build_graph(ctx, dc, dn, w, tc, tn,
                                            ic_out, tc_out, in_out, tn_out, maskC, tpadC, maskN, tpadN);

    // The captured entry gives every intermediate its own slot (no buffer reuse). Sum the
    // exact footprint and only build it if it fits dedicated VRAM with headroom — otherwise
    // bail to the non-persist path (per-call alloc/free, so it always fits), avoiding a WDDM
    // shared-memory spill that would make the block ~5x SLOWER than the non-captured path.
    std::size_t need = 0;
    for (ggml_tensor* tt = ggml_get_first_tensor(ctx); tt != nullptr; tt = ggml_get_next_tensor(ctx, tt))
        need += GGML_PAD(ggml_nbytes(tt), 256);
    ggml_backend_dev_t mdev = ggml_backend_get_device(g_backend);
    std::size_t freeb = 0, totalb = 0;
    if (mdev) ggml_backend_dev_memory(mdev, &freeb, &totalb);
    if (totalb > 0 && freeb < need + static_cast<std::size_t>(768) * 1024 * 1024)
    {
        std::fprintf(stderr, "[qwen-image-block-cfg] capture entry needs %zu MiB, free %zu MiB -> non-persist fallback\n",
                     need >> 20, freeb >> 20);
        ggml_free(ctx);
        return nullptr;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
    if (buffer == nullptr) { ggml_free(ctx); return nullptr; }
    if (tc.mask && maskC) ggml_backend_tensor_set(tc.mask, maskC, 0, static_cast<std::size_t>(tpadC) * tpadC * sizeof(ggml_fp16_t));
    if (tn.mask && maskN) ggml_backend_tensor_set(tn.mask, maskN, 0, static_cast<std::size_t>(tpadN) * tpadN * sizeof(ggml_fp16_t));

    QiBlockCfgPersist* e = nullptr;
    for (auto& c : g_qiCfgBlocks) if (!c.valid) { e = &c; break; }
    if (e == nullptr) { e = &g_qiCfgBlocks[g_qiCfgRR]; g_qiCfgRR = (g_qiCfgRR + 1) % kQiCfgCacheMax; e->reset(); }
    e->ctx = ctx; e->buffer = buffer; e->graph = graph; e->w = w; e->tc = tc; e->tn = tn;
    e->ic_out = ic_out; e->tc_out = tc_out; e->in_out = in_out; e->tn_out = tn_out;
    e->dim = dc->dim; e->heads = dc->heads; e->hd = dc->head_dim; e->ff = dc->ff;
    e->iseq = dc->img_seq; e->tseqc = dc->txt_seq; e->tseqn = dn->txt_seq;
    const TSGImgAttnW* wv[12] = { &dc->to_q, &dc->to_k, &dc->to_v, &dc->to_out,
                                  &dc->add_q, &dc->add_k, &dc->add_v, &dc->to_add_out,
                                  &dc->i_net0, &dc->i_net2, &dc->t_net0, &dc->t_net2 };
    for (int i = 0; i < 12; i++) e->wtype[i] = wv[i]->type;
    e->valid = true;
    return e;
}

// Upload the shared weights (from dc) + both branches' activations into the captured
// graph's fixed slots, run (capture-replays after warmup), read all four outputs back.
int qi_cfg_run(QiBlockCfgPersist* e, const TSGgmlQwenImageBlockDesc* dc, const TSGgmlQwenImageBlockDesc* dn)
{
    const int dim = e->dim, hd = e->hd, iseq = e->iseq, tseqc = e->tseqc, tseqn = e->tseqn, ff = e->ff;
    host_read_barrier();
    bool ok = true;
    auto chk = [&](ggml_tensor* tt, const void* dd, std::size_t bytes, const char* nm) {
        if (!tt || !dd || !ok) return;
        if (bytes > ggml_nbytes(tt)) {
            set_last_error(std::string("QwenImageBlockCfg(persist): oob set ") + nm +
                " bytes=" + std::to_string(bytes) + " cap=" + std::to_string(ggml_nbytes(tt)));
            ok = false; return;
        }
        ggml_backend_tensor_set(tt, dd, 0, bytes);
    };
    QiBlockTensors& w = e->w;
    auto upW = [&](ggml_tensor* wt, ggml_tensor* bt, const TSGImgAttnW& s, int blen, const char* nm) {
        chk(wt, s.w, static_cast<std::size_t>(s.bytes), nm);
        if (bt) chk(bt, s.b, static_cast<std::size_t>(blen) * sizeof(float), nm);
    };
    upW(w.toQw, w.toQb, dc->to_q, dim, "to_q"); upW(w.toKw, w.toKb, dc->to_k, dim, "to_k");
    upW(w.toVw, w.toVb, dc->to_v, dim, "to_v"); upW(w.toOw, w.toOb, dc->to_out, dim, "to_out");
    upW(w.aQw, w.aQb, dc->add_q, dim, "add_q"); upW(w.aKw, w.aKb, dc->add_k, dim, "add_k");
    upW(w.aVw, w.aVb, dc->add_v, dim, "add_v"); upW(w.aOw, w.aOb, dc->to_add_out, dim, "add_out");
    upW(w.iN0w, w.iN0b, dc->i_net0, ff, "i_net0"); upW(w.iN2w, w.iN2b, dc->i_net2, dim, "i_net2");
    upW(w.tN0w, w.tN0b, dc->t_net0, ff, "t_net0"); upW(w.tN2w, w.tN2b, dc->t_net2, dim, "t_net2");
    auto up1 = [&](ggml_tensor* tt, void* dd, const char* nm) { chk(tt, dd, static_cast<std::size_t>(hd) * sizeof(float), nm); };
    up1(w.nQ, dc->norm_q, "nQ"); up1(w.nK, dc->norm_k, "nK"); up1(w.nAQ, dc->norm_aq, "nAQ"); up1(w.nAK, dc->norm_ak, "nAK");

    auto setBranch = [&](QiBlockTensors& t, const TSGgmlQwenImageBlockDesc* d, int ts) {
        auto setF = [&](ggml_tensor* tt, void* dd, int rows, int seq, const char* nm) { chk(tt, dd, static_cast<std::size_t>(rows) * seq * sizeof(float), nm); };
        setF(t.img, d->img, dim, iseq, "img"); setF(t.txt, d->txt, dim, ts, "txt");
        setF(t.i_s1a, d->i_s1a, dim, iseq, "i_s1a"); setF(t.i_sha, d->i_sha, dim, iseq, "i_sha"); setF(t.i_ga, d->i_ga, dim, iseq, "i_ga");
        setF(t.t_s1a, d->t_s1a, dim, ts, "t_s1a"); setF(t.t_sha, d->t_sha, dim, ts, "t_sha"); setF(t.t_ga, d->t_ga, dim, ts, "t_ga");
        setF(t.i_s1m, d->i_s1m, dim, iseq, "i_s1m"); setF(t.i_shm, d->i_shm, dim, iseq, "i_shm"); setF(t.i_gm, d->i_gm, dim, iseq, "i_gm");
        setF(t.t_s1m, d->t_s1m, dim, ts, "t_s1m"); setF(t.t_shm, d->t_shm, dim, ts, "t_shm"); setF(t.t_gm, d->t_gm, dim, ts, "t_gm");
        setF(t.i_cos, d->i_cos, hd, iseq, "i_cos"); setF(t.i_sin, d->i_sin, hd, iseq, "i_sin");
        setF(t.t_cos, d->t_cos, hd, ts, "t_cos"); setF(t.t_sin, d->t_sin, hd, ts, "t_sin");
    };
    setBranch(e->tc, dc, tseqc);
    setBranch(e->tn, dn, tseqn);
    if (!ok) return 0;

    if (ggml_backend_graph_compute(g_backend, e->graph) != GGML_STATUS_SUCCESS)
    { set_last_error("QwenImageBlockCfg(persist): graph compute failed."); return 0; }
    ggml_backend_synchronize(g_backend);
    ggml_backend_tensor_get(e->ic_out, dc->img, 0, static_cast<std::size_t>(dim) * iseq * sizeof(float));
    ggml_backend_tensor_get(e->tc_out, dc->txt, 0, static_cast<std::size_t>(dim) * tseqc * sizeof(float));
    ggml_backend_tensor_get(e->in_out, dn->img, 0, static_cast<std::size_t>(dim) * iseq * sizeof(float));
    ggml_backend_tensor_get(e->tn_out, dn->txt, 0, static_cast<std::size_t>(dim) * tseqn * sizeof(float));
    return 1;
}

} // namespace

// Persist-mode dispatch flag (default ON; TS_QWEN_DIT_CAPTURE=0 forces the
// non-persistent build-every-call path).
static bool qi_block_persist_enabled()
{
    static const bool on = []{ const char* e = std::getenv("TS_QWEN_DIT_CAPTURE"); return e == nullptr || e[0] != '0'; }();
    return on;
}

// Max (img+txt) token count eligible for the captured path. With flash-attn (default on)
// the captured entry is O(total) VRAM, so the historical 768 cap (set for the O(total^2)
// materialized path) can be lifted to cover high-res edits. With flash OFF the materialized
// scores return, so keep the conservative 768. Override with TS_QWEN_DIT_CAPTURE_MAXTOK
// (0 = no cap / always capture).
static int qi_capture_max_tokens()
{
    static const int v = []{
        const char* e = std::getenv("TS_QWEN_DIT_CAPTURE_MAXTOK");
        if (e != nullptr) return std::atoi(e);
        return qi_flash_enabled() ? 4096 : 768;
    }();
    return v;
}

TSG_EXPORT int TSGgml_QwenImageBlock(const TSGgmlQwenImageBlockDesc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwenImageBlockDesc)))
        { set_last_error("QwenImageBlock: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        // Fast path: persistent + CUDA-graph-captured block. All same-profile blocks
        // share one cached graph; capture replays it after warmup, eliminating the
        // per-op WDDM launch overhead that starves the launch-bound DiT (~40% GPU idle).
        // The capture cache gives every intermediate its OWN slot (stable addresses, no
        // buffer reuse). Historically this cost O(total^2) VRAM (the materialized
        // [total,total] attention scores) and was capped at 768 tokens. Now that flash-
        // attention is wired into the captured path (qi_block_build sets t.mask), the
        // scores are gone and per-entry VRAM grows ~linearly, so capture extends to the
        // high-resolution regime. Cap is configurable via TS_QWEN_DIT_CAPTURE_MAXTOK
        // (0 = always capture); default raised when flash is on.
        // Single-branch capture keeps the conservative 768 cap: at higher res the cond/neg
        // branches have different txt lengths -> two captured entries per profile, doubling
        // VRAM. The CFG-batched captured path (TSGgml_QwenImageBlockCfg) is the high-res
        // capture route (one entry per profile for both branches). The VRAM guard in
        // qi_block_build still protects this path if a small image is unexpectedly large.
        const int kCaptureMaxTokens = 768;
        if (qi_block_persist_enabled() && (d->img_seq + d->txt_seq) <= kCaptureMaxTokens)
        {
            QiBlockPersist* e = qi_block_find(d);
            if (e == nullptr) e = qi_block_build(d);
            if (e != nullptr)
            {
                int r = qi_block_run(e, d);
                if (r) { clear_last_error(); return 1; }
                std::fprintf(stderr, "[qwen-image-block persist FAIL] %s\n", g_last_error.c_str());
                e->reset();   // fall through to the non-persist path on failure
            }
        }

        const int dim = d->dim, heads = d->heads, hd = d->head_dim, ff = d->ff;
        const int iseq = d->img_seq, tseq = d->txt_seq;
        const float eps = d->eps;

        PooledContextHandle context;
        if (!context.init(32 * 1024 * 1024)) { set_last_error("QwenImageBlock: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        QiBlockTensors t{};
        t.img = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        t.txt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        ggml_tensor* i_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        ggml_tensor* t_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        t.i_s1a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        t.i_sha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        t.i_ga  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        t.t_s1a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        t.t_sha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        t.t_ga  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        t.i_s1m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        t.i_shm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        t.i_gm  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        t.t_s1m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        t.t_shm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        t.t_gm  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        t.i_cos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, iseq);
        t.i_sin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, iseq);
        t.t_cos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, tseq);
        t.t_sin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, tseq);

        auto declW = [&](const TSGImgAttnW& s, ggml_tensor*& w, ggml_tensor*& b, int blen) {
            w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(s.type), s.ne0, s.ne1);
            b = s.b ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, blen) : nullptr;
        };
        declW(d->to_q, t.toQw, t.toQb, dim); declW(d->to_k, t.toKw, t.toKb, dim);
        declW(d->to_v, t.toVw, t.toVb, dim); declW(d->to_out, t.toOw, t.toOb, dim);
        declW(d->add_q, t.aQw, t.aQb, dim); declW(d->add_k, t.aKw, t.aKb, dim);
        declW(d->add_v, t.aVw, t.aVb, dim); declW(d->to_add_out, t.aOw, t.aOb, dim);
        t.nQ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        t.nK = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        t.nAQ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        t.nAK = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        declW(d->i_net0, t.iN0w, t.iN0b, ff);  declW(d->i_net2, t.iN2w, t.iN2b, dim);
        declW(d->t_net0, t.tN0w, t.tN0b, ff);  declW(d->t_net2, t.tN2w, t.tN2b, dim);

        // Flash mask [total_pad, total] F16 (bidirectional all-valid + KV-stride padding),
        // cached per shape and bound resident below so it uploads once and is reused.
        const ggml_fp16_t* maskHost = nullptr; int total_pad = 0;
        if (qi_flash_enabled())
        {
            maskHost = qi_attn_mask_host(iseq + tseq, total_pad);
            t.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, total_pad, total_pad);
        }

        // --- attention sub-layer -> img1/txt1, then per-stream MLP sub-layers ---
        ggml_tensor *img1, *txt1;
        qi_build_attn(ctx, t, dim, heads, hd, iseq, tseq, eps, img1, txt1);
        ggml_tensor* img2 = qi_build_mlp(ctx, img1, t.i_s1m, t.i_shm, t.i_gm, t.iN0w, t.iN0b, t.iN2w, t.iN2b, eps);
        ggml_tensor* txt2 = qi_build_mlp(ctx, txt1, t.t_s1m, t.t_shm, t.t_gm, t.tN0w, t.tN0b, t.tN2w, t.tN2b, eps);
        ggml_tensor* oi = ggml_cpy(ctx, img2, i_out); ggml_set_output(oi);
        ggml_tensor* ot = ggml_cpy(ctx, txt2, t_out); ggml_set_output(ot);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(graph, oi);
        ggml_build_forward_expand(graph, ot);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HB { ggml_tensor* t; void* d; std::size_t b; };
        std::vector<HB> uploads;
        auto bindW = [&](ggml_tensor* w, ggml_tensor* b, const TSGImgAttnW& s, int blen) {
            if (w && s.w) {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, w, s.w, static_cast<std::size_t>(s.bytes), buf, addr, needs)
                    && ggml_backend_tensor_alloc(buf, w, addr) == GGML_STATUS_SUCCESS) {
                    if (needs) uploads.push_back({w, s.w, static_cast<std::size_t>(s.bytes)});
                } else uploads.push_back({w, s.w, static_cast<std::size_t>(s.bytes)});
            }
            if (b && s.b) uploads.push_back({b, s.b, static_cast<std::size_t>(blen) * sizeof(float)});
        };
        bindW(t.toQw, t.toQb, d->to_q, dim); bindW(t.toKw, t.toKb, d->to_k, dim);
        bindW(t.toVw, t.toVb, d->to_v, dim); bindW(t.toOw, t.toOb, d->to_out, dim);
        bindW(t.aQw, t.aQb, d->add_q, dim); bindW(t.aKw, t.aKb, d->add_k, dim);
        bindW(t.aVw, t.aVb, d->add_v, dim); bindW(t.aOw, t.aOb, d->to_add_out, dim);
        bindW(t.iN0w, t.iN0b, d->i_net0, ff); bindW(t.iN2w, t.iN2b, d->i_net2, dim);
        bindW(t.tN0w, t.tN0b, d->t_net0, ff); bindW(t.tN2w, t.tN2b, d->t_net2, dim);
        auto bind1 = [&](ggml_tensor* tt, void* dd) { if (tt && dd) uploads.push_back({tt, dd, static_cast<std::size_t>(hd) * sizeof(float)}); };
        bind1(t.nQ, d->norm_q); bind1(t.nK, d->norm_k); bind1(t.nAQ, d->norm_aq); bind1(t.nAK, d->norm_ak);
        if (t.mask && maskHost)
        {
            // Allocate the mask in THIS call's ctx buffer and upload it each call. The
            // cacheable cache evicts/reuses buffers during the 60-block weight streaming,
            // which corrupted the mask after the first forward (-> all-(-inf) rows ->
            // softmax NaN from step 2 on). A per-call alloc avoids that entirely.
            const std::size_t mbytes = static_cast<std::size_t>(total_pad) * total_pad * sizeof(ggml_fp16_t);
            uploads.push_back({t.mask, const_cast<ggml_fp16_t*>(maskHost), mbytes});
        }

        // Pack the per-block intermediates with a liveness-reusing graph allocator
        // (ggml_gallocr) instead of giving every tensor its own slot
        // (ggml_backend_alloc_ctx_tensors). At 1MP the per-slot scratch reaches ~8.6 GB
        // (no reuse); gallocr packs it to ~1-2 GB, which is what lets the Q4_K_M DiT
        // (13 GB weights) fit a 19 GB Metal working set at high resolution. Every leaf we
        // fill from the host (the img/txt streams, the modulation arrays, the rope tables,
        // and the flash mask) MUST be flagged INPUT so gallocr never reuses its slot for an
        // intermediate; the pre-bound cacheable weights already own buffers and are skipped.
        // Falls back to per-tensor allocation when the reuse path is unavailable.
        auto markIn = [](ggml_tensor* x) { if (x) ggml_set_input(x); };
        markIn(t.img); markIn(t.txt);
        markIn(t.i_s1a); markIn(t.i_sha); markIn(t.i_ga);
        markIn(t.t_s1a); markIn(t.t_sha); markIn(t.t_ga);
        markIn(t.i_s1m); markIn(t.i_shm); markIn(t.i_gm);
        markIn(t.t_s1m); markIn(t.t_shm); markIn(t.t_gm);
        markIn(t.i_cos); markIn(t.i_sin); markIn(t.t_cos); markIn(t.t_sin);
        markIn(t.mask);

        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr) { set_last_error("QwenImageBlock: buffer alloc failed."); return 0; }
        }
        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
        auto setF = [&](ggml_tensor* tt, void* dd, int rows, int seq) { ggml_backend_tensor_set(tt, dd, 0, static_cast<std::size_t>(rows) * seq * sizeof(float)); };
        setF(t.img, d->img, dim, iseq); setF(t.txt, d->txt, dim, tseq);
        setF(t.i_s1a, d->i_s1a, dim, iseq); setF(t.i_sha, d->i_sha, dim, iseq); setF(t.i_ga, d->i_ga, dim, iseq);
        setF(t.t_s1a, d->t_s1a, dim, tseq); setF(t.t_sha, d->t_sha, dim, tseq); setF(t.t_ga, d->t_ga, dim, tseq);
        setF(t.i_s1m, d->i_s1m, dim, iseq); setF(t.i_shm, d->i_shm, dim, iseq); setF(t.i_gm, d->i_gm, dim, iseq);
        setF(t.t_s1m, d->t_s1m, dim, tseq); setF(t.t_shm, d->t_shm, dim, tseq); setF(t.t_gm, d->t_gm, dim, tseq);
        setF(t.i_cos, d->i_cos, hd, iseq); setF(t.i_sin, d->i_sin, hd, iseq);
        setF(t.t_cos, d->t_cos, hd, tseq); setF(t.t_sin, d->t_sin, hd, tseq);

        if (ggml_backend_graph_compute(g_backend, graph) != GGML_STATUS_SUCCESS)
        { set_last_error("QwenImageBlock: graph compute failed."); return 0; }
        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(i_out, d->img, 0, static_cast<std::size_t>(dim) * iseq * sizeof(float));
        ggml_backend_tensor_get(t_out, d->txt, 0, static_cast<std::size_t>(dim) * tseq * sizeof(float));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("QwenImageBlock: unknown error."); return 0; }
}

// ----------------------------------------------------------------------------
// CFG-batched block: run the SAME transformer layer for BOTH true-CFG branches
// (conditional dc + unconditional dn) in ONE graph that SHARES the identical
// quantized weight leaves. The DiT denoise is launch-bound (every per-block GPU
// sync + host round-trip starves the GPU ~40% idle, ~55W/350W); batching the two
// branches into a single dispatch doubles the work between syncs and HALVES the
// per-block weight upload + host round-trips, filling the idle GPU. The branches
// keep independent attention (each over its own [txt,img]) so NO cross-branch
// masking is needed, and their txt lengths may differ. Non-persist path only
// (the high-res >768-token regime that skips capture); the C# caller falls back
// to two single TSGgml_QwenImageBlock calls on failure. Weights are bound from
// dc only (dc/dn reference the same per-layer GGUF tensors).
TSG_EXPORT int TSGgml_QwenImageBlockCfg(const TSGgmlQwenImageBlockDesc* dc,
                                        const TSGgmlQwenImageBlockDesc* dn)
{
    try
    {
        if (dc == nullptr || dn == nullptr ||
            dc->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwenImageBlockDesc)) ||
            dn->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwenImageBlockDesc)))
        { set_last_error("QwenImageBlockCfg: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        // Fast path: persistent + CUDA-graph-captured combined block (one entry per quant
        // profile holds BOTH branches, so it fits VRAM where two separate single-branch
        // captures would spill). Capture replays after warmup, removing the per-op launch
        // overhead. Falls through to the non-persist build-every-call path on failure.
        if (qi_block_persist_enabled())
        {
            const int cap = qi_capture_max_tokens();
            const int totMax = std::max(dc->img_seq + dc->txt_seq, dc->img_seq + dn->txt_seq);
            if (cap <= 0 || totMax <= cap)
            {
                QiBlockCfgPersist* e = qi_cfg_find(dc, dn);
                if (e == nullptr) e = qi_cfg_build(dc, dn);
                if (e != nullptr)
                {
                    int r = qi_cfg_run(e, dc, dn);
                    if (r) { clear_last_error(); return 1; }
                    std::fprintf(stderr, "[qwen-image-block-cfg persist FAIL] %s\n", g_last_error.c_str());
                    e->reset();
                }
            }
        }

        const int dim = dc->dim, heads = dc->heads, hd = dc->head_dim, ff = dc->ff;
        const int iseq = dc->img_seq, tseqc = dc->txt_seq, tseqn = dn->txt_seq;
        const float eps = dc->eps;

        PooledContextHandle context;
        if (!context.init(32 * 1024 * 1024)) { set_last_error("QwenImageBlockCfg: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        // ---- shared weight leaves (declared once, bound from dc) ----
        QiBlockTensors w{};
        auto declW = [&](const TSGImgAttnW& s, ggml_tensor*& wt, ggml_tensor*& bt, int blen) {
            wt = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(s.type), s.ne0, s.ne1);
            bt = s.b ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, blen) : nullptr;
        };
        declW(dc->to_q, w.toQw, w.toQb, dim); declW(dc->to_k, w.toKw, w.toKb, dim);
        declW(dc->to_v, w.toVw, w.toVb, dim); declW(dc->to_out, w.toOw, w.toOb, dim);
        declW(dc->add_q, w.aQw, w.aQb, dim); declW(dc->add_k, w.aKw, w.aKb, dim);
        declW(dc->add_v, w.aVw, w.aVb, dim); declW(dc->to_add_out, w.aOw, w.aOb, dim);
        w.nQ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        w.nK = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        w.nAQ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        w.nAK = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
        declW(dc->i_net0, w.iN0w, w.iN0b, ff);  declW(dc->i_net2, w.iN2w, w.iN2b, dim);
        declW(dc->t_net0, w.tN0w, w.tN0b, ff);  declW(dc->t_net2, w.tN2w, w.tN2b, dim);

        // ---- per-branch activations (img/txt/mod/rope/mask); reuse shared weight ptrs ----
        auto declAct = [&](QiBlockTensors& t, int ts, ggml_tensor*& iout, ggml_tensor*& tout,
                           const ggml_fp16_t*& maskHost, int& tpad)
        {
            // share the weight leaves
            t.toQw=w.toQw; t.toQb=w.toQb; t.toKw=w.toKw; t.toKb=w.toKb;
            t.toVw=w.toVw; t.toVb=w.toVb; t.toOw=w.toOw; t.toOb=w.toOb;
            t.aQw=w.aQw; t.aQb=w.aQb; t.aKw=w.aKw; t.aKb=w.aKb;
            t.aVw=w.aVw; t.aVb=w.aVb; t.aOw=w.aOw; t.aOb=w.aOb;
            t.nQ=w.nQ; t.nK=w.nK; t.nAQ=w.nAQ; t.nAK=w.nAK;
            t.iN0w=w.iN0w; t.iN0b=w.iN0b; t.iN2w=w.iN2w; t.iN2b=w.iN2b;
            t.tN0w=w.tN0w; t.tN0b=w.tN0b; t.tN2w=w.tN2w; t.tN2b=w.tN2b;
            // own activation slots
            t.img = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
            t.txt = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
            iout  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
            tout  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
            t.i_s1a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
            t.i_sha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
            t.i_ga  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
            t.t_s1a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
            t.t_sha = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
            t.t_ga  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
            t.i_s1m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
            t.i_shm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
            t.i_gm  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
            t.t_s1m = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
            t.t_shm = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
            t.t_gm  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, ts);
            t.i_cos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, iseq);
            t.i_sin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, iseq);
            t.t_cos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, ts);
            t.t_sin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, ts);
            maskHost = nullptr; tpad = 0;
            if (qi_flash_enabled())
            {
                maskHost = qi_attn_mask_host(iseq + ts, tpad);
                t.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, tpad, tpad);
            }
        };

        QiBlockTensors tc{}, tn{};
        ggml_tensor *ic_out, *tc_out, *in_out, *tn_out;
        const ggml_fp16_t *maskC = nullptr, *maskN = nullptr; int tpadC = 0, tpadN = 0;
        declAct(tc, tseqc, ic_out, tc_out, maskC, tpadC);
        declAct(tn, tseqn, in_out, tn_out, maskN, tpadN);

        // ---- build both branch subgraphs (shared weight leaves) ----
        auto buildBranch = [&](QiBlockTensors& t, int ts, ggml_tensor* iout, ggml_tensor* tout,
                               ggml_tensor*& oi, ggml_tensor*& ot)
        {
            ggml_tensor *img1, *txt1;
            qi_build_attn(ctx, t, dim, heads, hd, iseq, ts, eps, img1, txt1);
            ggml_tensor* img2 = qi_build_mlp(ctx, img1, t.i_s1m, t.i_shm, t.i_gm, t.iN0w, t.iN0b, t.iN2w, t.iN2b, eps);
            ggml_tensor* txt2 = qi_build_mlp(ctx, txt1, t.t_s1m, t.t_shm, t.t_gm, t.tN0w, t.tN0b, t.tN2w, t.tN2b, eps);
            oi = ggml_cpy(ctx, img2, iout); ggml_set_output(oi);
            ot = ggml_cpy(ctx, txt2, tout); ggml_set_output(ot);
        };
        ggml_tensor *oic, *otc, *oin, *otn;
        buildBranch(tc, tseqc, ic_out, tc_out, oic, otc);
        buildBranch(tn, tseqn, in_out, tn_out, oin, otn);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 16384, false);
        ggml_build_forward_expand(graph, oic);
        ggml_build_forward_expand(graph, otc);
        ggml_build_forward_expand(graph, oin);
        ggml_build_forward_expand(graph, otn);

        // ---- bind shared weights ONCE (cacheable) + collect remaining uploads ----
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HB { ggml_tensor* t; void* d; std::size_t b; };
        std::vector<HB> uploads;
        auto bindW = [&](ggml_tensor* wt, ggml_tensor* bt, const TSGImgAttnW& s, int blen) {
            if (wt && s.w) {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, wt, s.w, static_cast<std::size_t>(s.bytes), buf, addr, needs)
                    && ggml_backend_tensor_alloc(buf, wt, addr) == GGML_STATUS_SUCCESS) {
                    if (needs) uploads.push_back({wt, s.w, static_cast<std::size_t>(s.bytes)});
                } else uploads.push_back({wt, s.w, static_cast<std::size_t>(s.bytes)});
            }
            if (bt && s.b) uploads.push_back({bt, s.b, static_cast<std::size_t>(blen) * sizeof(float)});
        };
        bindW(w.toQw, w.toQb, dc->to_q, dim); bindW(w.toKw, w.toKb, dc->to_k, dim);
        bindW(w.toVw, w.toVb, dc->to_v, dim); bindW(w.toOw, w.toOb, dc->to_out, dim);
        bindW(w.aQw, w.aQb, dc->add_q, dim); bindW(w.aKw, w.aKb, dc->add_k, dim);
        bindW(w.aVw, w.aVb, dc->add_v, dim); bindW(w.aOw, w.aOb, dc->to_add_out, dim);
        bindW(w.iN0w, w.iN0b, dc->i_net0, ff); bindW(w.iN2w, w.iN2b, dc->i_net2, dim);
        bindW(w.tN0w, w.tN0b, dc->t_net0, ff); bindW(w.tN2w, w.tN2b, dc->t_net2, dim);
        auto bind1 = [&](ggml_tensor* tt, void* dd) { if (tt && dd) uploads.push_back({tt, dd, static_cast<std::size_t>(hd) * sizeof(float)}); };
        bind1(w.nQ, dc->norm_q); bind1(w.nK, dc->norm_k); bind1(w.nAQ, dc->norm_aq); bind1(w.nAK, dc->norm_ak);
        if (tc.mask && maskC) uploads.push_back({tc.mask, const_cast<ggml_fp16_t*>(maskC), static_cast<std::size_t>(tpadC) * tpadC * sizeof(ggml_fp16_t)});
        if (tn.mask && maskN) uploads.push_back({tn.mask, const_cast<ggml_fp16_t*>(maskN), static_cast<std::size_t>(tpadN) * tpadN * sizeof(ggml_fp16_t)});

        // Liveness-packing allocator (gallocr): pack BOTH branches' intermediates by
        // liveness instead of one slot each. The two branches feed independent outputs,
        // so branch C's scratch is freed before branch N's peak — gallocr reclaims it,
        // roughly halving the combined-block scratch vs the per-tensor allocator. This is
        // what makes CFG-batching viable at high token counts (the budget that gates it can
        // rise once the combined block no longer doubles VRAM). Every host-filled leaf in
        // BOTH branches is flagged INPUT; the shared, pre-bound cacheable weights are skipped.
        auto markActIn = [](QiBlockTensors& t) {
            auto markIn = [](ggml_tensor* x) { if (x) ggml_set_input(x); };
            markIn(t.img); markIn(t.txt);
            markIn(t.i_s1a); markIn(t.i_sha); markIn(t.i_ga);
            markIn(t.t_s1a); markIn(t.t_sha); markIn(t.t_ga);
            markIn(t.i_s1m); markIn(t.i_shm); markIn(t.i_gm);
            markIn(t.t_s1m); markIn(t.t_shm); markIn(t.t_gm);
            markIn(t.i_cos); markIn(t.i_sin); markIn(t.t_cos); markIn(t.t_sin);
            markIn(t.mask);
        };
        markActIn(tc);
        markActIn(tn);

        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr) { set_last_error("QwenImageBlockCfg: buffer alloc failed."); return 0; }
        }
        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.d, 0, u.b);

        auto setBranch = [&](QiBlockTensors& t, const TSGgmlQwenImageBlockDesc* d, int ts) {
            auto setF = [&](ggml_tensor* tt, void* dd, int rows, int seq) { ggml_backend_tensor_set(tt, dd, 0, static_cast<std::size_t>(rows) * seq * sizeof(float)); };
            setF(t.img, d->img, dim, iseq); setF(t.txt, d->txt, dim, ts);
            setF(t.i_s1a, d->i_s1a, dim, iseq); setF(t.i_sha, d->i_sha, dim, iseq); setF(t.i_ga, d->i_ga, dim, iseq);
            setF(t.t_s1a, d->t_s1a, dim, ts); setF(t.t_sha, d->t_sha, dim, ts); setF(t.t_ga, d->t_ga, dim, ts);
            setF(t.i_s1m, d->i_s1m, dim, iseq); setF(t.i_shm, d->i_shm, dim, iseq); setF(t.i_gm, d->i_gm, dim, iseq);
            setF(t.t_s1m, d->t_s1m, dim, ts); setF(t.t_shm, d->t_shm, dim, ts); setF(t.t_gm, d->t_gm, dim, ts);
            setF(t.i_cos, d->i_cos, hd, iseq); setF(t.i_sin, d->i_sin, hd, iseq);
            setF(t.t_cos, d->t_cos, hd, ts); setF(t.t_sin, d->t_sin, hd, ts);
        };
        setBranch(tc, dc, tseqc);
        setBranch(tn, dn, tseqn);

        if (ggml_backend_graph_compute(g_backend, graph) != GGML_STATUS_SUCCESS)
        { set_last_error("QwenImageBlockCfg: graph compute failed."); return 0; }
        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(ic_out, dc->img, 0, static_cast<std::size_t>(dim) * iseq * sizeof(float));
        ggml_backend_tensor_get(tc_out, dc->txt, 0, static_cast<std::size_t>(dim) * tseqc * sizeof(float));
        ggml_backend_tensor_get(in_out, dn->img, 0, static_cast<std::size_t>(dim) * iseq * sizeof(float));
        ggml_backend_tensor_get(tn_out, dn->txt, 0, static_cast<std::size_t>(dim) * tseqn * sizeof(float));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("QwenImageBlockCfg: unknown error."); return 0; }
}

// ============================================================================
// WHOLE-DiT forward in ONE graph against RESIDENT weights (the high-performance
// path, mirroring stable-diffusion.cpp's QwenImageModel::forward_orig).
//
// The per-block kernels above run 60 separate native dispatches per forward, each
// with a device sync and ~143 MB/block of HOST-PRECOMPUTED per-token modulation
// uploaded (PrecomputeMod expands temb -> [seq,3072] x6 on the CPU, then ships it
// over PCIe). That serialises CPU<->GPU ~180 times/forward and starves the GPU.
//
// This kernel instead:
//   * binds ALL 60 blocks' weights resident (cached by their stable GGUF mmap
//     pointer -> uploaded once, reused every step), and
//   * computes the AdaLN modulation IN-graph from temb (2x3072) + the resident
//     per-block mod weights via a ggml_get_rows over the modulate_index, so img/txt
//     flow block-to-block entirely on-device.
// Per forward we upload only the small inputs (img tokens, txt cond, temb, rope,
// modulate_index, flash mask) and do ONE compute + ONE sync. Intermediates are
// liveness-packed by the reused gallocr (flash-attn keeps them O(n)).
// ============================================================================
struct TSGImgBlockW
{
    TSGImgAttnW img_mod, txt_mod;                      // [dim, 6*dim] (+bias)
    TSGImgAttnW to_q, to_k, to_v, to_out;
    TSGImgAttnW add_q, add_k, add_v, to_add_out;
    void* norm_q; void* norm_k; void* norm_aq; void* norm_ak;   // [head_dim] f32
    TSGImgAttnW i_net0, i_net2, t_net0, t_net2;        // mlp (+bias in .b)
};

struct TSGgmlQwenImageForwardDesc
{
    // in/out residual streams (post img_in / post txt_norm+txt_in — the C# prelude, shared
    // with the per-block path so the 60-block middle is byte-identical to it).
    void* img;            // [dim * img_seq] F32
    void* txt;            // [dim * txt_seq] F32
    void* temb;           // [dim * 2] F32  (row0 = embed(t), row1 = embed(0))
    void* img_cos; void* img_sin;   // [head_dim * img_seq] F32 (interleave-duplicated)
    void* txt_cos; void* txt_sin;   // [head_dim * txt_seq] F32
    void* modulate_index;           // [img_seq] I32 (0 = generated, 1 = reference)
    // blocks
    const TSGImgBlockW* blocks;     // [num_layers]
    std::int32_t struct_bytes, dim, heads, head_dim, ff, img_seq, txt_seq, num_layers;
    float eps;
};

namespace {

// LayerNorm(x)*(1+scale) + shift with RAW scale (not pre-added 1). scale/shift may
// be [dim, seq] (img, from get_rows) or [dim, 1] (txt, broadcast). Equivalent to
// the host PrecomputeMod's normed*(1+scale)+shift.
ggml_tensor* qi_mod_norm_raw(ggml_context* ctx, ggml_tensor* x, ggml_tensor* scale, ggml_tensor* shift, float eps)
{
    ggml_tensor* n = ggml_norm(ctx, x, eps);
    return ggml_add(ctx, ggml_add(ctx, n, ggml_mul(ctx, n, scale)), shift);
}

// One modulated GEGLU MLP sub-layer with raw scale/shift/gate. x: residual stream.
ggml_tensor* qi_build_mlp_raw(ggml_context* ctx, ggml_tensor* x, ggml_tensor* scale, ggml_tensor* shift, ggml_tensor* gate,
                              ggml_tensor* n0w, ggml_tensor* n0b, ggml_tensor* n2w, ggml_tensor* n2b, float eps)
{
    ggml_tensor* mod = qi_mod_norm_raw(ctx, x, scale, shift, eps);
    ggml_tensor* h = qi_mm(ctx, n0w, mod);
    if (n0b) h = ggml_add(ctx, h, n0b);
    h = ggml_gelu(ctx, h);
    ggml_tensor* mlp = qi_mm(ctx, n2w, h);
    if (n2b) mlp = ggml_add(ctx, mlp, n2b);
    return ggml_add(ctx, x, ggml_mul(ctx, mlp, gate));
}

// Device-side weight leaves for one whole-model block.
struct QiFullBlockW
{
    ggml_tensor *imgModW, *imgModB, *txtModW, *txtModB;
    ggml_tensor *toQw, *toQb, *toKw, *toKb, *toVw, *toVb, *toOw, *toOb;
    ggml_tensor *aQw, *aQb, *aKw, *aKb, *aVw, *aVb, *aOw, *aOb;
    ggml_tensor *nQ, *nK, *nAQ, *nAK;
    ggml_tensor *iN0w, *iN0b, *iN2w, *iN2b, *tN0w, *tN0b, *tN2w, *tN2b;
};

// Build one double-stream block: in-graph AdaLN modulation (from silu_temb + this
// block's mod weights, gathered per token by modIdx) then the verified attn + MLP.
// Updates img/txt (the on-device residual streams).
void qi_full_block(ggml_context* ctx, const QiFullBlockW& w, ggml_tensor* silu_temb, ggml_tensor* modIdx,
                   ggml_tensor* iCos, ggml_tensor* iSin, ggml_tensor* tCos, ggml_tensor* tSin, ggml_tensor* mask,
                   int dim, int heads, int hd, int iseq, int tseq, float eps,
                   ggml_tensor*& img, ggml_tensor*& txt)
{
    const int total = iseq + tseq;
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    // mod = silu(temb) @ mod_w + mod_b  -> [6*dim, 2]; img gathers per token by modIdx,
    // txt always uses row 0. Layout (matches host PrecomputeMod): 6 chunks of dim =
    // [shiftA | scaleA | gateA | shiftM | scaleM | gateM].
    ggml_tensor* imgModAll = qi_lin_bias(ctx, w.imgModW, silu_temb, w.imgModB);   // [6*dim, 2]
    ggml_tensor* txtModAll = qi_lin_bias(ctx, w.txtModW, silu_temb, w.txtModB);   // [6*dim, 2]
    ggml_tensor* modImg = ggml_get_rows(ctx, imgModAll, modIdx);                  // [6*dim, iseq]
    auto ichunk = [&](int k) {
        return ggml_cont(ctx, ggml_view_2d(ctx, modImg, dim, iseq, modImg->nb[1],
                                           static_cast<std::size_t>(k) * dim * sizeof(float)));
    };
    auto tchunk = [&](int k) {
        return ggml_cont(ctx, ggml_view_2d(ctx, txtModAll, dim, 1, txtModAll->nb[1],
                                           static_cast<std::size_t>(k) * dim * sizeof(float)));
    };
    ggml_tensor *iShA = ichunk(0), *iScA = ichunk(1), *iGA = ichunk(2);
    ggml_tensor *iShM = ichunk(3), *iScM = ichunk(4), *iGM = ichunk(5);
    ggml_tensor *tShA = tchunk(0), *tScA = tchunk(1), *tGA = tchunk(2);
    ggml_tensor *tShM = tchunk(3), *tScM = tchunk(4), *tGM = tchunk(5);

    // --- attention sub-layer ---
    ggml_tensor* imgMod = qi_mod_norm_raw(ctx, img, iScA, iShA, eps);
    ggml_tensor* txtMod = qi_mod_norm_raw(ctx, txt, tScA, tShA, eps);

    ggml_tensor* iq = ggml_reshape_3d(ctx, qi_lin_bias(ctx, w.toQw, imgMod, w.toQb), hd, heads, iseq);
    ggml_tensor* ik = ggml_reshape_3d(ctx, qi_lin_bias(ctx, w.toKw, imgMod, w.toKb), hd, heads, iseq);
    ggml_tensor* iv = ggml_reshape_3d(ctx, qi_lin_bias(ctx, w.toVw, imgMod, w.toVb), hd, heads, iseq);
    ggml_tensor* tq = ggml_reshape_3d(ctx, qi_lin_bias(ctx, w.aQw, txtMod, w.aQb), hd, heads, tseq);
    ggml_tensor* tk = ggml_reshape_3d(ctx, qi_lin_bias(ctx, w.aKw, txtMod, w.aKb), hd, heads, tseq);
    ggml_tensor* tv = ggml_reshape_3d(ctx, qi_lin_bias(ctx, w.aVw, txtMod, w.aVb), hd, heads, tseq);

    iq = qi_rope(ctx, qi_qk_norm(ctx, iq, w.nQ, hd, heads, iseq, eps), iCos, iSin, hd, heads, iseq);
    ik = qi_rope(ctx, qi_qk_norm(ctx, ik, w.nK, hd, heads, iseq, eps), iCos, iSin, hd, heads, iseq);
    tq = qi_rope(ctx, qi_qk_norm(ctx, tq, w.nAQ, hd, heads, tseq, eps), tCos, tSin, hd, heads, tseq);
    tk = qi_rope(ctx, qi_qk_norm(ctx, tk, w.nAK, hd, heads, tseq, eps), tCos, tSin, hd, heads, tseq);

    ggml_tensor* q = ggml_concat(ctx, tq, iq, 2);
    ggml_tensor* k = ggml_concat(ctx, tk, ik, 2);
    ggml_tensor* v = ggml_concat(ctx, tv, iv, 2);

    ggml_tensor* q_attn = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    ggml_tensor* k_attn = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    ggml_tensor* v_attn = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
    ggml_tensor* attn_flat = qi_attention(ctx, q_attn, k_attn, v_attn, mask, dim, total, scale);

    ggml_tensor* txt_attn = ggml_cont(ctx, ggml_view_2d(ctx, attn_flat, dim, tseq, attn_flat->nb[1], 0));
    ggml_tensor* img_attn = ggml_cont(ctx, ggml_view_2d(ctx, attn_flat, dim, iseq, attn_flat->nb[1],
                                                        static_cast<std::size_t>(tseq) * attn_flat->nb[1]));
    ggml_tensor* img_o = qi_lin_bias(ctx, w.toOw, img_attn, w.toOb);
    ggml_tensor* txt_o = qi_lin_bias(ctx, w.aOw, txt_attn, w.aOb);
    img = ggml_add(ctx, img, ggml_mul(ctx, img_o, iGA));
    txt = ggml_add(ctx, txt, ggml_mul(ctx, txt_o, tGA));

    // --- MLP sub-layer ---
    img = qi_build_mlp_raw(ctx, img, iScM, iShM, iGM, w.iN0w, w.iN0b, w.iN2w, w.iN2b, eps);
    txt = qi_build_mlp_raw(ctx, txt, tScM, tShM, tGM, w.tN0w, w.tN0b, w.tN2w, w.tN2b, eps);
}

} // namespace

TSG_EXPORT int TSGgml_QwenImageForward(const TSGgmlQwenImageForwardDesc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwenImageForwardDesc)))
        { set_last_error("QwenImageForward: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        const int dim = d->dim, heads = d->heads, hd = d->head_dim, ff = d->ff;
        const int iseq = d->img_seq, tseq = d->txt_seq, nl = d->num_layers;
        const int total = iseq + tseq;
        const float eps = d->eps;

        // Context sized for the whole 60-block graph's tensor headers + node list.
        // ~200 ops/block (heavy: interleaved RoPE + per-chunk cont); size generously so
        // ggml_new_graph_custom never asserts n_nodes < size, and give the ctx enough
        // header space for ~one tensor per node plus the leaves.
        const std::size_t nodes = static_cast<std::size_t>(nl) * 384 + 2048;
        const std::size_t meta = ggml_tensor_overhead() * (nodes + 1024)
                                 + ggml_graph_overhead_custom(nodes, false) + (8u << 20);
        ggml_init_params ip{ meta, nullptr, true };
        ContextHandle context(ggml_init(ip));
        if (context.value == nullptr) { set_last_error("QwenImageForward: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        // ---- input leaves (uploaded per forward) ----
        ggml_tensor* imgIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        ggml_tensor* txtIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        ggml_tensor* imgOut = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, iseq);
        ggml_tensor* txtOut = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tseq);
        ggml_tensor* temb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 2);
        ggml_tensor* iCos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, iseq);
        ggml_tensor* iSin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, iseq);
        ggml_tensor* tCos = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, tseq);
        ggml_tensor* tSin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, tseq);
        ggml_tensor* modIdx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, iseq);

        // flash mask (bidirectional all-valid + KV-stride padding), one per forward.
        const ggml_fp16_t* maskHost = nullptr; int total_pad = 0;
        ggml_tensor* mask = nullptr;
        if (qi_flash_enabled()) { maskHost = qi_attn_mask_host(total, total_pad); mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, total_pad, total_pad); }

        auto declW = [&](const TSGImgAttnW& s, ggml_tensor*& wt, ggml_tensor*& bt, int blen) {
            wt = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(s.type), s.ne0, s.ne1);
            bt = s.b ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, blen) : nullptr;
        };

        // ---- per-block weight leaves ----
        std::vector<QiFullBlockW> bw(nl);
        for (int l = 0; l < nl; l++)
        {
            const TSGImgBlockW& s = d->blocks[l];
            QiFullBlockW& b = bw[l];
            declW(s.img_mod, b.imgModW, b.imgModB, 6 * dim);
            declW(s.txt_mod, b.txtModW, b.txtModB, 6 * dim);
            declW(s.to_q, b.toQw, b.toQb, dim); declW(s.to_k, b.toKw, b.toKb, dim);
            declW(s.to_v, b.toVw, b.toVb, dim); declW(s.to_out, b.toOw, b.toOb, dim);
            declW(s.add_q, b.aQw, b.aQb, dim); declW(s.add_k, b.aKw, b.aKb, dim);
            declW(s.add_v, b.aVw, b.aVb, dim); declW(s.to_add_out, b.aOw, b.aOb, dim);
            b.nQ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            b.nK = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            b.nAQ = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            b.nAK = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
            declW(s.i_net0, b.iN0w, b.iN0b, ff); declW(s.i_net2, b.iN2w, b.iN2b, dim);
            declW(s.t_net0, b.tN0w, b.tN0b, ff); declW(s.t_net2, b.tN2w, b.tN2b, dim);
        }

        // ---- build the graph: silu(temb), then the 60 double-stream blocks ----
        ggml_tensor* silu_temb = ggml_silu(ctx, temb);                                 // [dim, 2]
        ggml_tensor* img = imgIn;
        ggml_tensor* txt = txtIn;
        for (int l = 0; l < nl; l++)
            qi_full_block(ctx, bw[l], silu_temb, modIdx, iCos, iSin, tCos, tSin, mask,
                          dim, heads, hd, iseq, tseq, eps, img, txt);
        ggml_tensor* oi = ggml_cpy(ctx, img, imgOut); ggml_set_output(oi);
        ggml_tensor* ot = ggml_cpy(ctx, txt, txtOut); ggml_set_output(ot);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
        ggml_build_forward_expand(graph, oi);
        ggml_build_forward_expand(graph, ot);

        // ---- bind weights resident (cached by stable GGUF pointer) ----
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HB { ggml_tensor* t; void* d; std::size_t b; };
        std::vector<HB> uploads;
        // Bind one tensor: prefer a resident (cached-by-pointer) device buffer; otherwise
        // mark it INPUT so the gallocr allocates a slot for it (a non-input leaf with no
        // buffer would be assumed pre-allocated and assert at tensor_set). Either way the
        // host data is uploaded into the tensor once below.
        auto bind = [&](ggml_tensor* tt, void* dd, std::size_t bytes) {
            if (!tt || !dd) return;
            if (bytes >= 4096) {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, tt, dd, bytes, buf, addr, needs)
                    && ggml_backend_tensor_alloc(buf, tt, addr) == GGML_STATUS_SUCCESS) {
                    if (needs) uploads.push_back({tt, dd, bytes});
                    return;
                }
                invalidate_cached_buffer(dd);
            }
            ggml_set_input(tt);   // gallocr allocates a slot; uploaded each call
            uploads.push_back({tt, dd, bytes});
        };
        auto bindW = [&](ggml_tensor* wt, ggml_tensor* bt, const TSGImgAttnW& s, int blen) {
            if (wt && s.w) bind(wt, s.w, static_cast<std::size_t>(s.bytes));
            if (bt && s.b) bind(bt, s.b, static_cast<std::size_t>(blen) * sizeof(float));
        };
        auto bind1 = [&](ggml_tensor* tt, void* dd, int len) {
            bind(tt, dd, static_cast<std::size_t>(len) * sizeof(float));
        };
        for (int l = 0; l < nl; l++)
        {
            const TSGImgBlockW& s = d->blocks[l];
            QiFullBlockW& b = bw[l];
            bindW(b.imgModW, b.imgModB, s.img_mod, 6 * dim);
            bindW(b.txtModW, b.txtModB, s.txt_mod, 6 * dim);
            bindW(b.toQw, b.toQb, s.to_q, dim); bindW(b.toKw, b.toKb, s.to_k, dim);
            bindW(b.toVw, b.toVb, s.to_v, dim); bindW(b.toOw, b.toOb, s.to_out, dim);
            bindW(b.aQw, b.aQb, s.add_q, dim); bindW(b.aKw, b.aKb, s.add_k, dim);
            bindW(b.aVw, b.aVb, s.add_v, dim); bindW(b.aOw, b.aOb, s.to_add_out, dim);
            bind1(b.nQ, s.norm_q, hd); bind1(b.nK, s.norm_k, hd);
            bind1(b.nAQ, s.norm_aq, hd); bind1(b.nAK, s.norm_ak, hd);
            bindW(b.iN0w, b.iN0b, s.i_net0, ff); bindW(b.iN2w, b.iN2b, s.i_net2, dim);
            bindW(b.tN0w, b.tN0b, s.t_net0, ff); bindW(b.tN2w, b.tN2b, s.t_net2, dim);
        }

        // ---- mark per-call inputs, gallocr-pack the rest ----
        auto markIn = [](ggml_tensor* x) { if (x) ggml_set_input(x); };
        markIn(imgIn); markIn(txtIn); markIn(temb);
        markIn(iCos); markIn(iSin); markIn(tCos); markIn(tSin);
        markIn(modIdx); markIn(mask);

        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr) { set_last_error("QwenImageForward: buffer alloc failed."); return 0; }
        }

        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
        ggml_backend_tensor_set(imgIn, d->img, 0, static_cast<std::size_t>(dim) * iseq * sizeof(float));
        ggml_backend_tensor_set(txtIn, d->txt, 0, static_cast<std::size_t>(dim) * tseq * sizeof(float));
        ggml_backend_tensor_set(temb, d->temb, 0, static_cast<std::size_t>(dim) * 2 * sizeof(float));
        ggml_backend_tensor_set(iCos, d->img_cos, 0, static_cast<std::size_t>(hd) * iseq * sizeof(float));
        ggml_backend_tensor_set(iSin, d->img_sin, 0, static_cast<std::size_t>(hd) * iseq * sizeof(float));
        ggml_backend_tensor_set(tCos, d->txt_cos, 0, static_cast<std::size_t>(hd) * tseq * sizeof(float));
        ggml_backend_tensor_set(tSin, d->txt_sin, 0, static_cast<std::size_t>(hd) * tseq * sizeof(float));
        ggml_backend_tensor_set(modIdx, d->modulate_index, 0, static_cast<std::size_t>(iseq) * sizeof(std::int32_t));
        if (mask && maskHost)
            ggml_backend_tensor_set(mask, maskHost, 0, static_cast<std::size_t>(total_pad) * total_pad * sizeof(ggml_fp16_t));

        if (ggml_backend_graph_compute(g_backend, graph) != GGML_STATUS_SUCCESS)
        { set_last_error("QwenImageForward: graph compute failed."); return 0; }
        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(imgOut, d->img, 0, static_cast<std::size_t>(dim) * iseq * sizeof(float));
        ggml_backend_tensor_get(txtOut, d->txt, 0, static_cast<std::size_t>(dim) * tseq * sizeof(float));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("QwenImageForward: unknown error."); return 0; }
}

} // extern "C"
