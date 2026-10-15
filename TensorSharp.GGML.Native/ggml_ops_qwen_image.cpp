// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// Qwen-Image-2.1 native kernels shared by the managed VAE and conditioning code:
//   * TSGgml_Conv2dF32        - one F32 VAE convolution (per-conv VAE path)
//   * TSGgml_QwenVaeAttention - tiled spatial VAE attention
//   * TSGgml_QwenVaeRun       - the whole VAE encode/decode as one graph
//   * TSGgml_QwenTeTrunk      - the Qwen3-VL-8B text-encoder trunk as one graph
// The Qwen-Image-2.1 transformer itself lives in ggml_ops_qwen_image21.cpp.
// ============================================================================
#include "ggml_ops_internal.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace tsg;

extern "C" {

// ============================================================================
// Single F32 2D convolution on the device. The Qwen-Image-2.1 VAE's per-conv path
// routes each convolution of its managed op chain through this kernel instead of
// pure-C# scalar loops.
//
// Layouts match the C# VaeReferenceMath exactly (no transposes):
//   input  C# Feature [C,H,W] (x contiguous) == ggml [W,H,C,1]
//   weight C# [OC,IC,KH,KW] (kw contiguous)  == ggml [KW,KH,IC,OC]
//   output C# [OC,OH,OW]                      == ggml [OW,OH,OC,1]
// Padding: symmetric (padL==padR && padT==padB) uses the conv's built-in pad;
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
// Managed mirrors: Conv2dArgs (pinned by QwenImageNativeAbiTests) and the native test
// tests/qwen_image21_vae_shortcut_test.cpp.
static_assert(sizeof(void*) != 8 || sizeof(TSGgmlConv2dDesc) == 112, "TSGgmlConv2dDesc must match managed Conv2dArgs");

// Qwen-Image-2.1's trained decoder can exceed the F16 range before its final
// normalization. Upstream ggml_conv_2d uses an F16 im2col for F32 weights, so
// requesting only F32 GEMM accumulation cannot preserve those activations.
// Keep this alternative lowering in TensorSharp and retain F32 throughout.
static ggml_tensor* vae_conv_2d_f32(ggml_context* ctx, ggml_tensor* kernel, ggml_tensor* input,
                                   int sw, int sh, int pw, int ph)
{
    // Metal's F32/F32 matrix-matrix kernel still stages its operands in F16.
    // The direct convolution reads and accumulates F32, including when MPS is
    // disabled or cannot accept this node's stride/padding. A precision flag on
    // MUL_MAT only controls accumulation and cannot prevent that conversion.
    if (g_backend_type == BACKEND_TYPE_METAL)
        return ggml_conv_2d_direct(ctx, kernel, input, sw, sh, pw, ph, 1, 1);
    auto columns = ggml_im2col(ctx, kernel, input, sw, sh, pw, ph, 1, 1, true, GGML_TYPE_F32);
    auto result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, columns, columns->ne[0], columns->ne[1] * columns->ne[2] * columns->ne[3]),
        ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1] * kernel->ne[2], kernel->ne[3]));
    ggml_prec_set_acc(result, GGML_PREC_F32);
    result = ggml_reshape_4d(ctx, result, columns->ne[1], columns->ne[2], columns->ne[3], kernel->ne[3]);
    return ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2));
}

static int vae_run_conv2d(const TSGgmlConv2dDesc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlConv2dDesc)))
        { set_last_error("Conv2d: bad descriptor."); return 0; }
        if (d->wtype != GGML_TYPE_F32)
        { set_last_error("Conv2dF32: requires F32 weights."); return 0; }
        if (!ensure_backend()) return 0;

        const int W = d->W, H = d->H, C = d->C;
        const int KW = d->KW, KH = d->KH, IC = d->IC, OC = d->OC;
        const int sW = d->strideW, sH = d->strideH;
        const bool symmetric = (d->padL == d->padR) && (d->padT == d->padB);
        const bool use_fast_conv = g_backend_type == BACKEND_TYPE_METAL && fast_conv_enabled();

        PooledContextHandle context;
        if (!context.init(32 * 1024 * 1024)) { set_last_error("Conv2d: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        ggml_tensor* inp = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_tensor* ker = ggml_new_tensor_4d(ctx, static_cast<ggml_type>(d->wtype), KW, KH, IC, OC);
        ggml_tensor* bias = d->bias ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, OC) : nullptr;

        // ggml-vulkan multiplies F32 matrices through F16 cooperative-matrix
        // operands (accumulating in F32) wherever the device has them, and the
        // 2.1 VAE feeds convolutions activations far above 65504. A power-of-two
        // scale is exact in both formats and the convolution is linear: bring the
        // input under the F16 range, then undo the scale on the F32 result.
        float input_scale = 1.f;
        if (g_backend_type == BACKEND_TYPE_VULKAN)
        {
            const float* values = static_cast<const float*>(d->input);
            float peak = 0.f;
            for (std::size_t i = 0, n = static_cast<std::size_t>(W) * H * C; i < n; ++i)
                peak = std::max(peak, std::fabs(values[i]));
            while (std::isfinite(peak) && peak * input_scale > 32768.f) input_scale *= 0.5f;
        }
        ggml_tensor* x = input_scale != 1.f ? ggml_scale(ctx, inp, input_scale) : inp;
        int p0 = d->padL, p1 = d->padT;
        if (!symmetric)
        {
            // Only the encoder Downsample is asymmetric: ZeroPad2d((0,1,0,1)) = pad
            // right/bottom by 1, then conv with pad 0. ggml_pad end-pads ne0/ne1.
            x = ggml_pad(ctx, x, d->padR - d->padL, d->padB - d->padT, 0, 0);
            p0 = 0; p1 = 0;
        }
        ggml_tensor* conv = vae_conv_2d_f32(ctx, ker, x, sW, sH, p0, p1);  // [OW,OH,OC,1]
        if (conv->op == GGML_OP_CONV_2D)
            conv->op_params[k_conv_full_precision_param] = 1;
        if (input_scale != 1.f) conv = ggml_scale(ctx, conv, 1.f / input_scale);
        if (bias) conv = ggml_add(ctx, conv, ggml_reshape_4d(ctx, bias, 1, 1, OC, 1));

        const int OW = static_cast<int>(conv->ne[0]), OH = static_cast<int>(conv->ne[1]);
        ggml_tensor* out = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, OW, OH, OC, 1);
        ggml_tensor* output = ggml_cpy(ctx, conv, out);
        ggml_set_output(output);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(graph, output);

        // VAE conv weights are managed float[] with unstable addresses (no benefit from
        // the host-pointer cacheable cache, and reuse would be unsafe), so upload fresh.
        BufferHandle buffer(alloc_ctx_tensors_budgeted(ctx, g_backend));
        if (buffer.value == nullptr) { set_last_error("Conv2d: buffer alloc failed (im2col OOM?)."); return 0; }

        host_read_barrier();
        ggml_backend_tensor_set(ker, d->weight, 0, static_cast<std::size_t>(d->weight_bytes));
        ggml_backend_tensor_set(inp, d->input, 0, static_cast<std::size_t>(W) * H * C * sizeof(float));
        if (bias) ggml_backend_tensor_set(bias, d->bias, 0, static_cast<std::size_t>(OC) * sizeof(float));

        // The Metal path uses MPS F32 convolution when possible;
        // its direct ggml fallback also preserves inputs beyond the F16 range.
        const auto status = use_fast_conv
            ? graph_compute_fast_conv(graph, "qwen-vae-conv-f32")
            : tsg::compute_graph(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        { set_last_error("Conv2d: graph compute failed."); return 0; }
        // Synchronous readback. The VAE runs this conv as a long C# chain (each conv's
        // output is the next conv's input), so the result MUST be on the host before this
        // returns. The async finalize_compute_with_download path only QUEUES a non-blocking
        // ggml_backend_tensor_get_async on Metal async mode and marks pending; the C# caller
        // reads d->output immediately (no host_read_barrier between native calls), so every
        // VAE conv layer consumed STALE/uninitialized data — cascading into the garbled/gray
        // decode that was long misdiagnosed as Q2_K quantization. CUDA/CPU were unaffected
        // (finalize takes the synchronous branch off Metal-async). Match the attention
        // kernel and drain synchronously here.
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(out, d->output, 0, static_cast<std::size_t>(OW) * OH * OC * sizeof(float));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("Conv2d: unknown error."); return 0; }
}

TSG_EXPORT int TSGgml_Conv2dF32(const TSGgmlConv2dDesc* d) { return vae_run_conv2d(d); }

// Spatial VAE attention with query tiling. Q/K/V are channel-planar [3,C,H*W].
// Each query still attends to every key: tiling bounds scratch without changing
// the softmax domain or introducing image seams. This also handles the 2.1 VAE's
// 768/1152-channel heads, which are too wide for many flash-attention backends.
TSG_EXPORT int TSGgml_QwenVaeAttention(const float* qkv, float* output, int channels, int sequence)
{
    try {
        if (!qkv || !output || channels <= 0 || sequence <= 0 ||
            static_cast<int64_t>(channels) * sequence > INT32_MAX / 3)
            throw std::invalid_argument("QwenVaeAttention: invalid shape or pointers");
        if (!ensure_backend()) return 0;
        // Keep each score matrix at most 16 MiB, including for rectangular 2K.
        const int tile = std::min(sequence, std::max(1, std::min(256, (4 * 1024 * 1024) / sequence)));
        const size_t nodes = static_cast<size_t>((sequence + tile - 1) / tile) * 12 + 64;
        ggml_init_params params{ggml_tensor_overhead() * (nodes + 64) + ggml_graph_overhead_custom(nodes, false), nullptr, true};
        ContextHandle context(ggml_init(params));
        if (!context.value) throw std::runtime_error("QwenVaeAttention: context allocation failed");
        auto ctx = context.value;
        auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sequence, 3LL * channels);
        ggml_set_input(input);
        auto part = [&](int channel) {
            return ggml_view_2d(ctx, input, sequence, channels, input->nb[1], channel * input->nb[1]);
        };
        auto q = ggml_cont(ctx, ggml_transpose(ctx, part(0)));
        auto k = ggml_cont(ctx, ggml_transpose(ctx, part(channels)));
        auto v = part(2 * channels);
        auto result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sequence, channels);
        ggml_set_output(result);
        auto graph = ggml_new_graph_custom(ctx, nodes, false);
        for (int start = 0; start < sequence; start += tile) {
            int count = std::min(tile, sequence - start);
            auto queries = ggml_view_2d(ctx, q, channels, count, q->nb[1], start * q->nb[1]);
            auto scores = ggml_mul_mat(ctx, k, queries);
            ggml_prec_set_acc(scores, GGML_PREC_F32);
            auto probs = ggml_soft_max_ext(ctx, scores, nullptr, 1.f / std::sqrt(static_cast<float>(channels)), 0.f);
            auto values = ggml_mul_mat(ctx, probs, v);
            ggml_prec_set_acc(values, GGML_PREC_F32);
            auto destination = ggml_view_2d(ctx, result, count, channels, result->nb[1], start * sizeof(float));
            // Expanding each copy in order lets gallocr recycle a tile's scores
            // before evaluating the next tile, instead of retaining all scores.
            ggml_build_forward_expand(graph, ggml_cpy(ctx, values, destination));
        }
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            if (!backend_supports_op(ggml_graph_node(graph, i)))
                throw std::runtime_error("QwenVaeAttention: operation unsupported by backend");
        if (!alloc_graph_reuse_gallocr(graph)) throw std::runtime_error("QwenVaeAttention: graph allocation failed");
        host_read_barrier();
        ggml_backend_tensor_set(input, qkv, 0, ggml_nbytes(input));
        if (compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("QwenVaeAttention: graph compute failed");
        sync_backend(g_backend);
        ggml_backend_tensor_get(result, output, 0, ggml_nbytes(result));
        clear_last_error();
        return 1;
    } catch (const std::exception& e) { set_last_error(e.what()); return 0; }
    catch (...) { set_last_error("QwenVaeAttention: unknown error"); return 0; }
}

// ============================================================================
// Whole-VAE fused graph. The per-conv TSGgml_Conv2dF32 path above runs the VAE as
// a long C# chain: every conv re-uploads its weights AND the full feature map, then
// downloads the result, while SiLU / channel-RMSNorm / nearest-upsample run as C#
// loops over the (hundreds-of-MB at ~1 MP) host arrays between convs. That is
// GBs of PCIe traffic + CPU elementwise passes per encode/decode.
//
// This kernel instead executes the WHOLE encoder/decoder as ONE ggml graph:
// the C# side emits a flat op list (single source of truth stays the verified
// Qwen-Image-2.1 VAE topology), features stay on-device end-to-end, weights are
// bound resident by their stable unmanaged pointers (uploaded once, reused across
// encode/decode/edits), and every conv keeps F32 intermediates (aux=1 is
// required). Per call: one input upload, one compute, one sync, one output
// download.
// ============================================================================
enum TSGVaeOpKind : std::int32_t
{
    TSG_VAE_CONV = 0,       // conv2d(weight w, bias b) with stride/pad
    TSG_VAE_NORM = 1,       // channel RMS norm * gamma (w = gamma index)
    TSG_VAE_SILU = 2,
    TSG_VAE_UPSAMPLE = 3,   // nearest x2
    TSG_VAE_SAVE = 4,       // slots[dst] = slots[src] (alias, no compute)
    TSG_VAE_ADD = 5,        // slots[dst] = slots[src] + slots[aux]
    TSG_VAE_ATTN = 6,       // spatial single-head attention over slots[src]=[W,H,3C] -> [W,H,C]
    TSG_VAE_AVERAGE_DOWN21 = 7, // oc=output channels, kh=time factor, kw=spatial factor
    TSG_VAE_DUPLICATE_UP21 = 8, // same parameters; retains the final temporal sample
};

struct TSGVaeWeightRef { void* data; std::int64_t bytes; };   // stable F32 host ptr
static_assert(sizeof(void*) != 8 || sizeof(TSGVaeWeightRef) == 16, "TSGVaeWeightRef must match managed QwenVaeWeightRef");

struct TSGVaeOp
{
    std::int32_t kind;
    std::int32_t w, b;                       // weight / bias table indices (-1 = none)
    std::int32_t oc, ic, kh, kw;             // conv shape (attn: oc = C)
    std::int32_t sh, sw, pt, pb, pl, pr;     // conv stride / padding
    std::int32_t src, dst, aux;              // virtual feature slots; conv aux=1 requests F32 intermediates
};
static_assert(sizeof(TSGVaeOp) == 64, "TSGVaeOp must match managed QwenVaeOp");

struct TSGgmlQwenVaeDesc
{
    void* input; std::int32_t in_w, in_h, in_c;   // [W,H,C] F32 (C# Feature [C,H,W])
    void* output; std::int64_t out_len;           // expected element count of the final slot-0
    const TSGVaeOp* ops; std::int32_t num_ops;
    const TSGVaeWeightRef* weights; std::int32_t num_weights;
    std::int32_t struct_bytes;
};

TSG_EXPORT int TSGgml_QwenVaeRun(const TSGgmlQwenVaeDesc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwenVaeDesc)) ||
            d->input == nullptr || d->output == nullptr || d->ops == nullptr || d->num_ops <= 0)
        { set_last_error("QwenVaeRun: bad descriptor."); return 0; }
        // Every convolution must request F32 intermediates: the Qwen-Image-2.1
        // decoder exceeds the F16 range before its final norm.
        for (int i = 0; i < d->num_ops; ++i)
            if (d->ops[i].kind == TSG_VAE_CONV && d->ops[i].aux != 1)
            { set_last_error("QwenVaeRun: convolutions must request F32 intermediates (aux=1)."); return 0; }
        if (!ensure_backend()) return 0;

        // Prefer cuDNN for a single-frame graph that has convolutions; a
        // convolution-free op list keeps the plain graph compute.
        bool prefer_cuda_convolution = false;
        for (int i = 0; i < d->num_ops; ++i)
            if (d->ops[i].kind == TSG_VAE_CONV && d->ops[i].aux == 1)
            { prefer_cuda_convolution = true; break; }
        const bool use_fast_conv = tsg::fast_conv_enabled(prefer_cuda_convolution);

        PooledContextHandle context;
        if (!context.init(32 * 1024 * 1024)) { set_last_error("QwenVaeRun: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* d; std::size_t b; };
        std::vector<HostBinding> uploads;
        host_read_barrier();
        auto bindW = [&](ggml_tensor* t, int idx) -> bool {
            if (idx < 0 || idx >= d->num_weights) return false;
            void* data = d->weights[idx].data;
            std::size_t bytes = static_cast<std::size_t>(d->weights[idx].bytes);
            if (data == nullptr || ggml_nbytes(t) != bytes) return false;
            if (bytes >= 4096) {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs)) {
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS) {
                        // A later graph-allocation/support failure must not leave
                        // an uninitialized weight published in the resident cache.
                        if (needs) ggml_backend_tensor_set(t, data, 0, bytes);
                        return true;
                    }
                    invalidate_cached_buffer(data);
                }
            }
            ggml_set_input(t);
            uploads.push_back({t, data, bytes});
            return true;
        };

        ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d->in_w, d->in_h, d->in_c, 1);
        ggml_set_input(input);

        constexpr int kSlots = 8;
        ggml_tensor* slots[kSlots] = {};
        slots[0] = input;

        for (int i = 0; i < d->num_ops; i++)
        {
            const TSGVaeOp& op = d->ops[i];
            if (op.src < 0 || op.src >= kSlots || op.dst < 0 || op.dst >= kSlots || slots[op.src] == nullptr)
            { set_last_error("QwenVaeRun: bad op slots."); return 0; }
            ggml_tensor* x = slots[op.src];

            switch (op.kind)
            {
                case TSG_VAE_CONV:
                {
                    int p0 = op.pl, p1 = op.pt;
                    if (op.pl != op.pr || op.pt != op.pb)
                    {
                        // only the encoder Downsample's end-pad (0,1,0,1) is asymmetric
                        if (op.pl != 0 || op.pt != 0) { set_last_error("QwenVaeRun: unsupported asymmetric pad."); return 0; }
                        x = ggml_pad(ctx, x, op.pr, op.pb, 0, 0);
                        p0 = 0; p1 = 0;
                    }
                    ggml_tensor* ker = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, op.kw, op.kh, op.ic, op.oc);
                    if (!bindW(ker, op.w)) { set_last_error("QwenVaeRun: bad conv weight."); return 0; }
                    // Prefer the im2col+GEMM conv (tensor cores; ~3x the direct kernel) while
                    // its transient F32 im2col fits a budget — the gallocr reuses that scratch
                    // across the graph, so the peak is ONE conv's im2col, not the sum. Above
                    // the budget fall back to ggml_conv_2d_direct (no materialization).
                    static const long long kIm2colBudget = []() {
                        const char* e = std::getenv("TS_QWEN_VAE_FUSED_IM2COL_BUDGET");
                        long long v = e ? std::atoll(e) : 0;
                        return v > 0 ? v : 2LL * 1024 * 1024 * 1024;
                    }();
                    const long long oh = (x->ne[1] + 2LL * p1 - op.kh) / op.sh + 1;
                    const long long ow = (x->ne[0] + 2LL * p0 - op.kw) / op.sw + 1;
                    const long long im2col = static_cast<long long>(op.ic) * op.kh * op.kw * oh * ow * 4;
                    // With MPSGraph/cuDNN available the convolution is executed
                    // whole, so emit the un-lowered node and skip im2col entirely.
                    ggml_tensor* y = use_fast_conv
                        ? ggml_conv_2d_direct(ctx, ker, x, op.sw, op.sh, p0, p1, 1, 1)
                        : (im2col <= kIm2colBudget
                            ? vae_conv_2d_f32(ctx, ker, x, op.sw, op.sh, p0, p1)
                            : ggml_conv_2d_direct(ctx, ker, x, op.sw, op.sh, p0, p1, 1, 1));
                    if (y->op == GGML_OP_CONV_2D)
                        y->op_params[tsg::k_conv_full_precision_param] = 1;
                    if (op.b >= 0)
                    {
                        ggml_tensor* bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, op.oc);
                        if (!bindW(bt, op.b)) { set_last_error("QwenVaeRun: bad conv bias."); return 0; }
                        y = ggml_add(ctx, y, ggml_reshape_4d(ctx, bt, 1, 1, op.oc, 1));
                    }
                    slots[op.dst] = y;
                    break;
                }
                case TSG_VAE_NORM:
                {
                    // F.normalize over the channel dim * sqrt(C) * gamma == rms_norm over C
                    // (with eps mapped from the reference's sum-space 1e-12 to mean-space).
                    const std::int64_t hw = x->ne[0] * x->ne[1], C = x->ne[2];
                    ggml_tensor* r = ggml_reshape_2d(ctx, x, hw, C);
                    ggml_tensor* tr = ggml_cont(ctx, ggml_transpose(ctx, r));            // [C, hw]
                    ggml_tensor* n = ggml_rms_norm(ctx, tr, 1e-12f / static_cast<float>(C));
                    ggml_tensor* gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, C);
                    if (!bindW(gamma, op.w)) { set_last_error("QwenVaeRun: bad norm gamma."); return 0; }
                    n = ggml_mul(ctx, n, gamma);
                    ggml_tensor* back = ggml_cont(ctx, ggml_transpose(ctx, n));          // [hw, C]
                    slots[op.dst] = ggml_reshape_3d(ctx, back, x->ne[0], x->ne[1], C);
                    break;
                }
                case TSG_VAE_SILU:
                    slots[op.dst] = ggml_silu(ctx, x);
                    break;
                case TSG_VAE_UPSAMPLE:
                    slots[op.dst] = ggml_upscale(ctx, x, 2, GGML_SCALE_MODE_NEAREST);
                    break;
                case TSG_VAE_SAVE:
                    slots[op.dst] = x;
                    break;
                case TSG_VAE_ADD:
                    if (op.aux < 0 || op.aux >= kSlots || slots[op.aux] == nullptr)
                    { set_last_error("QwenVaeRun: bad add slot."); return 0; }
                    slots[op.dst] = ggml_add(ctx, x, slots[op.aux]);
                    break;
                case TSG_VAE_AVERAGE_DOWN21:
                {
                    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2];
                    const int64_t sf = op.kw, tf = op.kh, OC = op.oc;
                    if (sf <= 0 || tf <= 0 || OC <= 0 || W % sf || H % sf ||
                        (C * tf * sf * sf) % OC || x->ne[3] != 1)
                    { set_last_error("QwenVaeRun: invalid average-down shape."); return 0; }
                    const int64_t OW = W / sf, OH = H / sf, hw = OW * OH;
                    const int64_t group = C * tf * sf * sf / OC;
                    auto r = ggml_reshape_4d(ctx, ggml_cont(ctx, x), sf, OW, H, C);
                    r = ggml_cont(ctx, ggml_permute(ctx, r, 1, 0, 2, 3));
                    r = ggml_reshape_4d(ctx, r, OW, sf * sf, OH, C);
                    r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));
                    r = ggml_reshape_4d(ctx, r, hw, sf * sf, 1, C);
                    // Single-frame causal input is front padded in time. Its
                    // zeros still participate in the channel-group average.
                    if (tf > 1)
                    {
                        ggml_tensor* padded = ggml_pad_ext(ctx, r, 0, 0, 0, 0, int(tf - 1), 0, 0, 0);
                        // Upstream ggml-metal pads only at the end of a dimension, and one
                        // unsupported node refuses the whole graph below: every Metal edit
                        // encoded its reference image on the per-convolution path instead.
                        // The same tensor -- tf-1 slices of zeros, then the frame -- as a
                        // concat runs there; fill writes its constant whatever r holds, so
                        // a non-finite frame cannot leak into the zero slices.
                        if (!ggml_backend_supports_op(g_backend, padded))
                        {
                            ggml_tensor* shape = tf == 2 ? r : ggml_repeat_4d(ctx, r, r->ne[0], r->ne[1], tf - 1, r->ne[3]);
                            padded = ggml_concat(ctx, ggml_fill(ctx, shape, 0.0f), r, 2);
                        }
                        r = padded;
                    }
                    r = ggml_reshape_3d(ctx, r, hw, group, OC);
                    r = ggml_cont(ctx, ggml_permute(ctx, r, 1, 0, 2, 3));
                    slots[op.dst] = ggml_reshape_3d(ctx, ggml_mean(ctx, r), OW, OH, OC);
                    break;
                }
                case TSG_VAE_DUPLICATE_UP21:
                {
                    const int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], hw = W * H;
                    const int64_t sf = op.kw, tf = op.kh, OC = op.oc;
                    if (sf <= 0 || tf <= 0 || OC <= 0 || (OC * tf * sf * sf) % C || x->ne[3] != 1)
                    { set_last_error("QwenVaeRun: invalid duplicate-up shape."); return 0; }
                    const int64_t repeats = OC * tf * sf * sf / C;
                    auto r = ggml_reshape_3d(ctx, ggml_cont(ctx, x), hw, 1, C);
                    r = ggml_repeat_4d(ctx, r, hw, repeats, C, 1);
                    r = ggml_reshape_4d(ctx, r, hw, sf * sf, tf, OC);
                    r = ggml_cont(ctx, ggml_view_4d(ctx, r, hw, sf * sf, 1, OC,
                        r->nb[1], r->nb[2], r->nb[3], (tf - 1) * r->nb[2]));
                    r = ggml_reshape_4d(ctx, r, W, H, sf * sf, OC);
                    r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));
                    r = ggml_reshape_4d(ctx, r, W, sf, sf * H, OC);
                    r = ggml_cont(ctx, ggml_permute(ctx, r, 1, 0, 2, 3));
                    slots[op.dst] = ggml_reshape_3d(ctx, r, W * sf, H * sf, OC);
                    break;
                }
                case TSG_VAE_ATTN:
                {
                    // x = qkv [W,H,3C], channel-planar. Single head over hw positions, dim C.
                    const std::int64_t W = x->ne[0], H = x->ne[1], C = op.oc, hw = W * H;
                    if (x->ne[2] != 3 * C) { set_last_error("QwenVaeRun: attn qkv shape."); return 0; }
                    ggml_tensor* q2 = ggml_reshape_2d(ctx, x, hw, 3 * C);
                    auto part = [&](std::int64_t o) {
                        return ggml_view_2d(ctx, q2, hw, C, q2->nb[1], static_cast<std::size_t>(o) * q2->nb[1]);
                    };
                    ggml_tensor* qT = ggml_cont(ctx, ggml_transpose(ctx, part(0)));      // [C, hw]
                    ggml_tensor* kT = ggml_cont(ctx, ggml_transpose(ctx, part(C)));      // [C, hw]
                    ggml_tensor* v = ggml_cont(ctx, part(2 * C));                        // [hw, C]
                    // scores[i,j] = q[:,i]·k[:,j] / sqrt(C); softmax over j; out[c,i] = sum_j P[i,j] v[j,c]
                    ggml_tensor* S = ggml_mul_mat(ctx, kT, qT);                          // [hw(j), hw(i)]
                    S = ggml_scale(ctx, S, 1.0f / std::sqrt(static_cast<float>(C)));
                    ggml_tensor* P = ggml_soft_max(ctx, S);
                    ggml_tensor* o2 = ggml_mul_mat(ctx, P, v);                           // [hw(i), C]
                    slots[op.dst] = ggml_reshape_3d(ctx, o2, W, H, C);
                    break;
                }
                default:
                    set_last_error("QwenVaeRun: unknown op kind.");
                    return 0;
            }
        }

        ggml_tensor* fin = slots[0];
        if (fin == nullptr || ggml_nelements(fin) != d->out_len)
        { set_last_error("QwenVaeRun: output shape mismatch."); return 0; }
        ggml_tensor* out = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, fin->ne[0], fin->ne[1], fin->ne[2], fin->ne[3]);
        ggml_tensor* copied = ggml_cpy(ctx, fin, out);
        ggml_set_output(copied);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);
        ggml_build_forward_expand(graph, copied);

        // Bail out (so the C# per-conv fallback runs) if the backend can't execute any
        // node — e.g. a backend without the direct GGML_OP_CONV_2D kernel.
        for (int i = 0; i < ggml_graph_n_nodes(graph); i++)
        {
            ggml_tensor* node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(g_backend, node))
            { set_last_error("QwenVaeRun: op unsupported by backend."); return 0; }
        }

        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = alloc_ctx_tensors_budgeted(ctx, g_backend);
            if (buffer.value == nullptr) { set_last_error("QwenVaeRun: buffer alloc failed."); return 0; }
        }

        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
        ggml_backend_tensor_set(input, d->input,
            0, static_cast<std::size_t>(d->in_w) * d->in_h * d->in_c * sizeof(float));

        const ggml_status vaeSt = use_fast_conv
            ? tsg::graph_compute_fast_conv(graph, "qwen-image vae")
            : tsg::compute_graph(g_backend, graph);
        if (vaeSt != GGML_STATUS_SUCCESS)
        { set_last_error("QwenVaeRun: graph compute failed."); return 0; }
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(out, d->output, 0, static_cast<std::size_t>(d->out_len) * sizeof(float));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("QwenVaeRun: unknown error."); return 0; }
}

// One projection weight (+ optional F32 bias) of the text-encoder trunk below.
// Mirrored by the managed QImgAttnW; the layer array holding these is not size
// checked by the kernel, so the layout is pinned here and in QwenImageNativeAbiTests.
struct TSGImgAttnW
{
    void* w; int type; std::int64_t ne0, ne1, bytes; void* b;
};
static_assert(sizeof(void*) != 8 || sizeof(TSGImgAttnW) == 48, "TSGImgAttnW must match managed QImgAttnW");

namespace {

// Quantized matmul with overflow-safe activation pre-scaling. ggml quantizes the
// activation to q8_1, whose per-block FP16 sum overflows for large activations.
// Scaling the activation by 1/K before the matmul (and the result by K) keeps that
// FP16 sum in range; q8_1 is scale-invariant in precision (the per-block scale
// adapts), so this is exact. `prescale = false` skips both passes. The trunk below
// uses it only for the TS_QWEN21_TE_PRESCALE=1 diagnostic.
constexpr float QI_MM_SCALE = 1024.0f;
ggml_tensor* qi_mm(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, bool prescale = true)
{
    if (!prescale) return ggml_mul_mat(ctx, w, x);
    ggml_tensor* xs = ggml_scale(ctx, x, 1.0f / QI_MM_SCALE);
    return ggml_scale(ctx, ggml_mul_mat(ctx, w, xs), QI_MM_SCALE);
}

} // namespace

// ============================================================================
// Fused transformer trunk for the Qwen-Image-2.1 text encoder (TSGgml_QwenTeTrunk):
// the Qwen3-VL-8B LLM (36 layers, GQA, causal, per-head Q/K RMS norms) ran per-op
// in C#, paying ~10 device<->host round-trips per layer. This kernel runs the
// WHOLE trunk as ONE graph: weights resident (GGUF mmap ptr for the big matrices),
// M-RoPE cos/sin precomputed on the host, one upload of the input states + one
// download of the last block's hidden states (the conditioning is taken before
// the final norm). The vision tower's DeepStack embeddings are added after the
// first deepstack_count blocks.
//
// Layer: x += o_proj(attn(rms_norm(x)*ln1)); x += down(silu(gate)*up over
// rms_norm(x)*ln2). RoPE is rotate-half (NeoX): out = x*cos + rotate_half(x)*sin
// with per-token duplicated-half cos/sin tables [head_dim, seq]. Attention is
// materialized and causal.
// ============================================================================
struct TSGTeLayerW
{
    void* ln1; void* ln2;                        // [hidden] F32 (stable host ptrs)
    TSGImgAttnW q, k, v, o, gate, up, down;      // .b = optional F32 bias
    void* q_norm; void* k_norm;                  // [head_dim] F32, required
};
static_assert(sizeof(void*) != 8 || sizeof(TSGTeLayerW) == 368, "TSGTeLayerW must match managed QwenTeLayerW");

struct TSGgmlQwenTeTrunkDesc
{
    void* x;                 // [hidden, seq] F32 input states
    void* out;               // [hidden, seq] F32 output (last block, before the final norm)
    void* cosf; void* sinf;  // [head_dim, seq] F32 rotate-half tables
    const TSGTeLayerW* layers; std::int32_t num_layers;
    std::int32_t struct_bytes, hidden, heads, kv_heads, head_dim, seq;
    float eps;
    void* deepstack;
    std::int32_t deepstack_count;
};
static_assert(sizeof(void*) != 8 || sizeof(TSGgmlQwenTeTrunkDesc) == 88, "TSGgmlQwenTeTrunkDesc must match managed QwenTeTrunkArgs");

namespace {

// rotate-half RoPE: x [head_dim, heads, seq]; cos/sin [head_dim, seq].
ggml_tensor* qte_rope_half(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cosf, ggml_tensor* sinf,
                           int hd, int heads, int seq)
{
    const int half = hd / 2;
    ggml_tensor* top = ggml_view_3d(ctx, x, half, heads, seq, x->nb[1], x->nb[2], 0);
    ggml_tensor* bot = ggml_view_3d(ctx, x, half, heads, seq, x->nb[1], x->nb[2],
                                    static_cast<std::size_t>(half) * x->nb[0]);
    ggml_tensor* rot = ggml_concat(ctx, ggml_neg(ctx, ggml_cont(ctx, bot)), ggml_cont(ctx, top), 0);
    ggml_tensor* cos3 = ggml_reshape_3d(ctx, cosf, hd, 1, seq);
    ggml_tensor* sin3 = ggml_reshape_3d(ctx, sinf, hd, 1, seq);
    return ggml_add(ctx, ggml_mul(ctx, x, cos3), ggml_mul(ctx, rot, sin3));
}

} // namespace

TSG_EXPORT int TSGgml_QwenTeTrunk(const TSGgmlQwenTeTrunkDesc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlQwenTeTrunkDesc)) ||
            d->x == nullptr || d->out == nullptr || d->cosf == nullptr || d->sinf == nullptr ||
            d->layers == nullptr || d->num_layers <= 0)
        { set_last_error("QwenTeTrunk: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;
        const int hidden = d->hidden, heads = d->heads, kvh = d->kv_heads, hd = d->head_dim, seq = d->seq;
        const int nl = d->num_layers;
        if (heads <= 0 || kvh <= 0 || heads % kvh != 0 || hidden != heads * hd)
        { set_last_error("QwenTeTrunk: bad head geometry."); return 0; }
        for (int l = 0; l < nl; l++)
            if (d->layers[l].q_norm == nullptr || d->layers[l].k_norm == nullptr)
            { set_last_error("QwenTeTrunk: every layer needs its Q/K head norms."); return 0; }
        const float eps = d->eps, scale = 1.0f / std::sqrt(static_cast<float>(hd));
        const char* trace_dir = std::getenv("TS_QWEN_TE_TRACE_DIR");
        const char* trace_layer_env = std::getenv("TS_QWEN_TE_TRACE_LAYER");
        const int trace_layer = trace_layer_env ? std::atoi(trace_layer_env) : 0;
        struct TraceTensor { ggml_tensor* tensor; std::string name; float scale; };
        std::vector<TraceTensor> traces;
        auto trace = [&](int layer, const char* name, ggml_tensor* tensor, float multiplier = 1.f) {
            if (!trace_dir || !trace_dir[0] || layer != trace_layer) return;
            ggml_set_output(tensor); // preserve it after its last graph consumer
            traces.push_back({tensor, name, multiplier});
        };

        PooledContextHandle context;
        if (!context.init(32 * 1024 * 1024)) { set_last_error("QwenTeTrunk: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* dd; std::size_t b; };
        std::vector<HostBinding> uploads;
        host_read_barrier();
        auto bind = [&](ggml_tensor* t, void* data, std::size_t bytes) {
            if (t == nullptr || data == nullptr) return;
            if (bytes >= 4096) {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs)) {
                    if (ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS) {
                        // The per-op fallback can reuse this cache entry if
                        // graph construction later fails. Publish valid data.
                        if (needs) ggml_backend_tensor_set(t, data, 0, bytes);
                        return;
                    }
                    invalidate_cached_buffer(data);
                }
            }
            ggml_set_input(t);
            uploads.push_back({t, data, bytes});
        };
        auto declW = [&](const TSGImgAttnW& s, ggml_tensor*& wt, ggml_tensor*& bt) {
            wt = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(s.type), s.ne0, s.ne1);
            bind(wt, s.w, static_cast<std::size_t>(s.bytes));
            bt = nullptr;
            if (s.b) {
                bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.ne1);
                bind(bt, s.b, static_cast<std::size_t>(s.ne1) * sizeof(float));
            }
        };
        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, seq);
        ggml_tensor* cosf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, seq);
        ggml_tensor* sinf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, seq);
        ggml_tensor* outT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, seq);
        // ggml-metal implements NEITHER a diag-mask kernel nor a supports_op case
        // for GGML_OP_DIAG_MASK_INF, so ggml_diag_mask_inf + ggml_soft_max would
        // fail the supports_op sweep on Metal. An explicit additive causal mask
        // fed to ggml_soft_max_ext is the same maths (this is what llama.cpp
        // does) and is supported on every backend.
        ggml_tensor* causalMask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq);
        std::vector<float> causalMaskData(static_cast<std::size_t>(seq) * seq, 0.0f);
        {
            const float neg_inf = -std::numeric_limits<float>::infinity();
            // Row j = query position, column i = key position; mask i > j.
            for (int j = 0; j < seq; j++)
                for (int i = j + 1; i < seq; i++)
                    causalMaskData[static_cast<std::size_t>(j) * seq + i] = neg_inf;
        }

        ggml_tensor* h = x;
        const char* prescale_env = std::getenv("TS_QWEN21_TE_PRESCALE");
        const bool restore_qwen21_prescale = prescale_env && prescale_env[0] == '1';
        for (int l = 0; l < nl; l++)
        {
            const TSGTeLayerW& lw = d->layers[l];
            trace(l, "input", h);
            // Match the per-op projection math: dividing small normalized
            // activations by 1024 can underflow CUDA's F16 q8_1 scales.
            // TS_QWEN21_TE_PRESCALE=1 restores the guard for diagnosis.
            const bool prescale = restore_qwen21_prescale;
            auto mm = [&](ggml_tensor* w, ggml_tensor* xx, ggml_tensor* bias) {
                auto o = prescale && ggml_is_quantized(w->type) ? qi_mm(ctx, w, xx) : ggml_mul_mat(ctx, w, xx);
                return bias ? ggml_add(ctx, o, bias) : o;
            };
            ggml_tensor *qw, *qb, *kw, *kb, *vw, *vb, *ow, *ob, *gw, *gb, *uw, *ub, *dw, *db;
            declW(lw.q, qw, qb); declW(lw.k, kw, kb); declW(lw.v, vw, vb); declW(lw.o, ow, ob);
            declW(lw.gate, gw, gb); declW(lw.up, uw, ub); declW(lw.down, dw, db);
            ggml_tensor* ln1 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
            ggml_tensor* ln2 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
            bind(ln1, lw.ln1, static_cast<std::size_t>(hidden) * sizeof(float));
            bind(ln2, lw.ln2, static_cast<std::size_t>(hidden) * sizeof(float));

            // --- attention ---
            ggml_tensor* n1 = ggml_mul(ctx, ggml_rms_norm(ctx, h, eps), ln1);
            trace(l, "norm1", n1);
            ggml_tensor* q = ggml_reshape_3d(ctx, mm(qw, n1, qb), hd, heads, seq);
            ggml_tensor* k = ggml_reshape_3d(ctx, mm(kw, n1, kb), hd, kvh, seq);
            ggml_tensor* v = ggml_reshape_3d(ctx, mm(vw, n1, vb), hd, kvh, seq);
            trace(l, "q", q); trace(l, "k", k); trace(l, "v", v);
            {
                ggml_tensor* qn = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
                ggml_tensor* kn = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hd);
                bind(qn, lw.q_norm, static_cast<std::size_t>(hd) * sizeof(float));
                bind(kn, lw.k_norm, static_cast<std::size_t>(hd) * sizeof(float));
                q = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), qn);
                k = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), kn);
            }
            trace(l, "qnorm", q); trace(l, "knorm", k);
            q = qte_rope_half(ctx, q, cosf, sinf, hd, heads, seq);
            k = qte_rope_half(ctx, k, cosf, sinf, hd, kvh, seq);
            trace(l, "qrope", q); trace(l, "krope", k);

            ggml_tensor* qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));   // [hd, seq, heads]
            ggml_tensor* kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));   // [hd, seq, kvh]
            ggml_tensor* kq = ggml_mul_mat(ctx, kp, qp);                          // [kv, q, heads] (GQA broadcast)
            trace(l, "scores", kq, scale);
            ggml_tensor* probs = ggml_soft_max_ext(ctx, kq, causalMask, scale, 0.0f);
            trace(l, "probs", probs);
            ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));   // [kv, hd, kvh]
            ggml_tensor* kqv = ggml_mul_mat(ctx, vt, probs);                      // [hd, q, heads]
            ggml_tensor* merged = ggml_reshape_2d(ctx,
                ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), hidden, seq); // [hidden, seq]
            trace(l, "merged", merged);
            auto attn_out = mm(ow, merged, ob);
            trace(l, "attn_out", attn_out);
            h = ggml_add(ctx, h, attn_out);
            trace(l, "attn_residual", h);

            // --- SwiGLU MLP ---
            ggml_tensor* n2 = ggml_mul(ctx, ggml_rms_norm(ctx, h, eps), ln2);
            trace(l, "norm2", n2);
            ggml_tensor* g = mm(gw, n2, gb);
            ggml_tensor* u = mm(uw, n2, ub);
            trace(l, "gate", g); trace(l, "up", u);
            ggml_tensor* ff = ggml_mul(ctx, ggml_silu(ctx, g), u);
            trace(l, "activated", ff);
            auto down = mm(dw, ff, db);
            trace(l, "down", down);
            h = ggml_add(ctx, h, down);
            trace(l, "output", h);
            if (d->deepstack && l < d->deepstack_count)
            {
                ggml_tensor* extra = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, seq);
                ggml_set_input(extra);
                uploads.push_back({extra, static_cast<float*>(d->deepstack) + static_cast<std::size_t>(l) * hidden * seq,
                    static_cast<std::size_t>(hidden) * seq * sizeof(float)});
                h = ggml_add(ctx, h, extra);
            }
        }

        ggml_tensor* copied = ggml_cpy(ctx, h, outT);
        ggml_set_output(copied);

        const std::size_t nodes = static_cast<std::size_t>(nl) * 96 + 1024;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
        if (graph == nullptr) { set_last_error("QwenTeTrunk: graph alloc failed."); return 0; }
        ggml_build_forward_expand(graph, copied);

        for (int i = 0; i < ggml_graph_n_nodes(graph); i++)
        {
            if (!ggml_backend_supports_op(g_backend, ggml_graph_node(graph, i)))
            { set_last_error("QwenTeTrunk: op unsupported by backend."); return 0; }
        }

        ggml_set_input(x); ggml_set_input(cosf); ggml_set_input(sinf);
        ggml_set_input(causalMask);

        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = alloc_ctx_tensors_budgeted(ctx, g_backend);
            if (buffer.value == nullptr) { set_last_error("QwenTeTrunk: buffer alloc failed."); return 0; }
        }

        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.dd, 0, u.b);
        ggml_backend_tensor_set(x, d->x, 0, static_cast<std::size_t>(hidden) * seq * sizeof(float));
        ggml_backend_tensor_set(cosf, d->cosf, 0, static_cast<std::size_t>(hd) * seq * sizeof(float));
        ggml_backend_tensor_set(sinf, d->sinf, 0, static_cast<std::size_t>(hd) * seq * sizeof(float));
        ggml_backend_tensor_set(causalMask, causalMaskData.data(), 0, causalMaskData.size() * sizeof(float));

        if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        { set_last_error("QwenTeTrunk: graph compute failed."); return 0; }
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(outT, d->out, 0, static_cast<std::size_t>(hidden) * seq * sizeof(float));
        for (const auto& item : traces) {
            std::vector<float> values(static_cast<size_t>(ggml_nelements(item.tensor)));
            ggml_backend_tensor_get(item.tensor, values.data(), 0, values.size() * sizeof(float));
            if (item.scale != 1.f) for (float& value : values) value *= item.scale;
            char file_name[128];
            std::snprintf(file_name, sizeof(file_name), "/fused.L%02d.%s.f32", trace_layer, item.name.c_str());
            std::string path = std::string(trace_dir) + file_name;
            FILE* file = std::fopen(path.c_str(), "wb");
            if (!file) throw std::runtime_error("QwenTeTrunk: cannot create trace file " + path);
            const size_t written = std::fwrite(values.data(), sizeof(float), values.size(), file);
            std::fclose(file);
            if (written != values.size()) throw std::runtime_error("QwenTeTrunk: incomplete trace file " + path);
        }
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("QwenTeTrunk: unknown error."); return 0; }
}

} // extern "C"
