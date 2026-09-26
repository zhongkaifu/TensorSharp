// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// MiniMax-H3 native whole-network graphs.
//
// H3 generates video and 32 kHz stereo audio jointly. Its four networks are all
// built here as single ggml graphs, one submission per call, with weights bound
// resident straight from the caller's mmap / stable host buffers.
//
// This file currently implements:
//   TSGgml_MiniMaxH3VideoVaeDecode  - the video VAE's ViT decoder
//
// The video VAE decoder is unusual and worth stating plainly: there is NO
// convolutional decoder. `AutoencoderKLLegacy` refuses to build anything but a
// pure transformer, so `ch_mult`, `space_up`, resblocks and group norms are all
// encoder-only config. Every latent voxel becomes one token via Linear(24->2048),
// 36 pre-norm blocks run over it, and a single final Linear(2048->3072) plus a
// depth-to-space reshape performs the entire 4x temporal + 16x spatial upsample
// in one step.
//
// Layout notes that are easy to get wrong:
//   * `to_qkv` is PER-HEAD INTERLEAVED. The reference does
//     `qkv.view(B, N, -1, 3*dim_head).chunk(3, dim=-1)`, so for head h the 192
//     contiguous values are [q(64) k(64) v(64)] -- NOT three 2048-wide blocks.
//   * qk-norm is RMSNorm with NO learnable weight (`qk_norm_affine: false`).
//   * norm1/norm2 are RMSNorm WITH affine; norm_out is LayerNorm with bias.
//   * scale1/scale2 are LayerScale vectors applied to the branch output, not norms.
//   * RoPE covers the first 48 of 64 head dims (rope_dim_ratio 0.75) as rotate-half
//     with pairs (j, j+24); the remaining 16 dims pass through untouched. The 5
//     suffix tokens (4 learned registers + 1 zero) get position 0, i.e. identity.
// ============================================================================
#include "ggml_ops_internal.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace tsg;

extern "C" {

// One linear layer's weights. ne0/ne1 are ggml order, i.e. ne0 = in_features and
// ne1 = out_features. The weight keeps its on-disk dtype (F16 for the video VAE,
// which halves both footprint and bandwidth); the bias is always F32.
struct TSGH3Lin
{
    void* w;
    void* b;                 // nullable
    std::int64_t ne0, ne1;
    std::int64_t bytes;      // weight byte count
    std::int32_t type;       // ggml_type of the weight
    std::int32_t pad_;
};

// One ViT decoder block.
struct TSGH3VitBlockW
{
    void* norm1;             // [dim] F32, RMSNorm affine
    void* norm2;             // [dim] F32
    void* scale1;            // [dim] F32, LayerScale on the attention branch
    void* scale2;            // [dim] F32, LayerScale on the FFN branch
    TSGH3Lin qkv;            // [dim, 3*dim]
    TSGH3Lin out;            // [dim, dim]
    TSGH3Lin w1;             // [dim, 2*inner]  (gate | value)
    TSGH3Lin w2;             // [inner, dim]
};

struct TSGgmlMiniMaxH3VideoVaeDecodeDesc
{
    std::int32_t struct_bytes;
    std::int32_t num_blocks;

    void* latent;            // [latent_c, tokens] F32 -- caller packs (t,h,w) order
    void* out;               // [patch_dim, tokens] F32
    void* cosf;              // [rot_dim, tokens + num_suffix] F32
    void* sinf;              // [rot_dim, tokens + num_suffix] F32
    void* register_tokens;   // [dim, num_register] F32, nullable
    void* norm_out_w;        // [dim] F32
    void* norm_out_b;        // [dim] F32, nullable

    TSGH3Lin post_quant;     // [latent_c, latent_c] -- the 1x1x1 conv
    TSGH3Lin x_embedder;     // [latent_c, dim]
    TSGH3Lin proj_out;       // [dim, patch_dim]

    const TSGH3VitBlockW* blocks;

    std::int32_t tokens;
    std::int32_t latent_c;
    std::int32_t dim;
    std::int32_t heads;
    std::int32_t head_dim;
    std::int32_t inner;      // FFN inner width (w1 emits 2*inner)
    std::int32_t rot_dim;    // rotated head dims
    std::int32_t num_register;
    std::int32_t patch_dim;
    float eps;
};

namespace {

// Rotate-half RoPE over the leading `rot` dims of each head.
// x: [head_dim, heads, seq]; cos/sin: [rot, seq].
ggml_tensor* h3_vit_rope(ggml_context* ctx, ggml_tensor* x,
                         ggml_tensor* cosf, ggml_tensor* sinf,
                         int hd, int heads, int seq, int rot)
{
    if (rot <= 0) return x;
    const std::size_t f = sizeof(float);
    ggml_tensor* xr = ggml_cont(ctx, ggml_view_3d(ctx, x, rot, heads, seq, x->nb[1], x->nb[2], 0));
    const int half = rot / 2;
    ggml_tensor* a = ggml_cont(ctx, ggml_view_3d(ctx, xr, half, heads, seq, xr->nb[1], xr->nb[2], 0));
    ggml_tensor* b = ggml_cont(ctx, ggml_view_3d(ctx, xr, half, heads, seq, xr->nb[1], xr->nb[2],
                                                 static_cast<std::size_t>(half) * f));
    // rotate_half(x) = [-x2, x1]
    ggml_tensor* rot_half = ggml_concat(ctx, ggml_neg(ctx, b), a, 0);
    ggml_tensor* c3 = ggml_reshape_3d(ctx, cosf, rot, 1, seq);
    ggml_tensor* s3 = ggml_reshape_3d(ctx, sinf, rot, 1, seq);
    ggml_tensor* rotated = ggml_add(ctx, ggml_mul(ctx, xr, c3), ggml_mul(ctx, rot_half, s3));
    if (rot == hd) return rotated;
    ggml_tensor* tail = ggml_cont(ctx, ggml_view_3d(ctx, x, hd - rot, heads, seq,
                                                    x->nb[1], x->nb[2],
                                                    static_cast<std::size_t>(rot) * f));
    return ggml_concat(ctx, rotated, tail, 0);
}

// Effective keys one FP16 softmax numerator is allowed to sum over before V is
// pre-scaled. See h3_flash_v_scale for the arithmetic; 256 leaves the
// accumulator 8 * 65504 / 256 = 2047 of room on |V| itself, an order of
// magnitude above the activations h3_mm below documents as "reaching the
// thousands".
constexpr int kH3FlashKeyBudget = 256;

// Smallest power of two that brings `keys` down to the budget, and exactly 1.0
// for any sequence already short enough -- so short sequences (every oracle
// fixture in the test suite among them) stay bit-identical.
float h3_flash_v_scale(int keys)
{
    float s = 1.0f;
    while (keys > kH3FlashKeyBudget * s) s *= 2.0f;
    return s;
}

// Full bidirectional attention over a packed sequence.
//
// q/k/v arrive as [head_dim, heads, seq]. The explicit path materializes a
// [seq, seq, heads] score matrix, which at 640x384 would be ~14 GB for the DiT and
// ~6 GB per VAE layer -- the reference sidesteps that by tiling the VAE. Flash
// attention never materializes it, so one call covers the whole frame with no tile
// seams. K/V are cast to F16 because that is what the kernel takes.
//
// THE ACCUMULATE IS NOT RELIABLY F32, whatever ggml_prec_set_acc below looks
// like it asks for. ggml-cuda's tensor-core kernel keeps the softmax NUMERATOR
// -- sum_j exp(s_j - max) * V_j -- in FP16 registers (T_C_VKQ = tile<16, 8,
// half2>, ggml-cuda/fattn-mma-f16.cuh), and neither ggml-cuda nor ggml-metal
// reads the precision slot of GGML_OP_FLASH_ATTN_EXT (op_params[3]; they read
// only scale, max_bias, logit_softcap and n_kv_max), so the request cannot move
// them. ggml-metal's F16 kernels accumulate in float anyway (FA_TYPES,
// ggml-metal/kernels/fa.metal); ggml-vulkan is the one backend that honours the
// request and switches to an F32 accumulator. (Checked against the vendored ggml
// at 353b63b.) What the CUDA kernel does give its accumulator is three bits of
// headroom: FATTN_KQ_MAX_OFFSET (ggml-cuda/fattn-common.cuh) inflates the
// running maximum by log(8), capping every softmax weight at 1/8. A row of N keys
// therefore reaches N/8 * |V| and overflows to Inf once N * |V| > 8 * 65504.
//
// H3 walks straight into that ceiling and no other model here does. Its attention
// is bidirectional over ONE packed sequence with no mask, so N is the whole clip
// rather than a sliding window (2364 tokens for 22 frames, 8646 for 107), and V is
// the one stream carrying no norm: q_norm and k_norm exist in the checkpoint,
// v_norm does not, which is the same unbounded magnitude that makes h3_mm's guard
// necessary one op later. Measured at 640x384: 73 frames (N=6134, tolerates
// |V| <= 85) renders correctly and 107 frames (N=8646, tolerates only 61) came
// back pure black, because one Inf here poisons the shared trunk and takes the
// video AND the audio stream down with it.
//
// Attention is LINEAR in V, so scaling V down before the kernel and scaling the
// result back up afterwards is exact -- and by a power of two it is exact through
// the FP16 cast as well, being nothing but an exponent shift.
ggml_tensor* h3_attend(ggml_context* ctx, ggml_backend_t backend,
                       ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                       int hd, int heads, int seq, float scale, bool allowFlash)
{
    ggml_tensor* qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));   // [hd, seq, heads]
    ggml_tensor* kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

    // TS_H3_NO_FLASH=1 forces the explicit softmax path, for isolating a
    // flash-attention kernel problem from a modelling one.
    if (const char* off = std::getenv("TS_H3_NO_FLASH"))
        if (off[0] == '1') allowFlash = false;

    if (allowFlash)
    {
        ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        // Keep the kernel's FP16 numerator in range no matter how long the clip
        // is: the accumulator now sums over at most kH3FlashKeyBudget effective
        // keys' worth of V, and the output is scaled back on the way out.
        const float vScale = h3_flash_v_scale(seq);
        ggml_tensor* kf = ggml_cast(ctx, kp, GGML_TYPE_F16);
        ggml_tensor* vf = ggml_cast(ctx,
            vScale == 1.0f ? vp : ggml_scale(ctx, vp, 1.0f / vScale), GGML_TYPE_F16);
        ggml_tensor* out = ggml_flash_attn_ext(ctx, qp, kf, vf, nullptr, scale, 0.0f, 0.0f);
        if (out != nullptr && ggml_backend_supports_op(backend, out))
        {
            // Inert on ggml-cuda and ggml-metal, honoured only by ggml-vulkan (see
            // the note above the function); kept because the request is the right
            // one and costs nothing, but the V pre-scale is what actually keeps the
            // FP16 accumulator finite.
            ggml_prec_set_acc(out, GGML_PREC_F32);
            // [hd, heads, seq] contiguous -> [inner, seq].
            ggml_tensor* merged = ggml_reshape_2d(ctx, ggml_cont(ctx, out), hd * heads, seq);
            return vScale == 1.0f ? merged : ggml_scale(ctx, merged, vScale);
        }
    }

    ggml_tensor* kq = ggml_mul_mat(ctx, kp, qp);
    ggml_tensor* probs = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    ggml_tensor* kqv = ggml_mul_mat(ctx, vt, probs);
    return ggml_reshape_2d(ctx,
        ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), hd * heads, seq);
}

// Quantized matmul with overflow-safe activation pre-scaling. ggml quantizes the
// activation to q8_1, whose per-block FP16 sum overflows past roughly 2000 per
// element. Two inputs in this file are unbounded and need the guard: the attention
// output feeding o_proj, and the SwiGLU hidden state feeding down_proj (which
// reaches the thousands). Everything else consumes an RMSNorm output that stays
// O(10), where the guard would only cost two extra full-tensor passes.
//
// q8_1 is scale-invariant in precision because the per-block scale adapts, so
// scaling by 1/K before and K after is exact rather than approximate.
constexpr float kH3MmScale = 1024.0f;

ggml_tensor* h3_mm(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, bool guard)
{
    if (!guard || !ggml_is_quantized(w->type)) return ggml_mul_mat(ctx, w, x);
    ggml_tensor* xs = ggml_scale(ctx, x, 1.0f / kH3MmScale);
    return ggml_scale(ctx, ggml_mul_mat(ctx, w, xs), kH3MmScale);
}

} // namespace

TSG_EXPORT int TSGgml_MiniMaxH3VideoVaeDecode(const TSGgmlMiniMaxH3VideoVaeDecodeDesc* d)
{
    try
    {
        if (d == nullptr ||
            d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlMiniMaxH3VideoVaeDecodeDesc)) ||
            d->latent == nullptr || d->out == nullptr || d->cosf == nullptr || d->sinf == nullptr ||
            d->blocks == nullptr || d->num_blocks <= 0 || d->tokens <= 0 ||
            d->norm_out_w == nullptr)
        { set_last_error("MiniMaxH3VideoVaeDecode: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        const int nl = d->num_blocks;
        const int dim = d->dim, heads = d->heads, hd = d->head_dim;
        const int inner = d->inner, rot = d->rot_dim;
        const int ntok = d->tokens, nreg = d->num_register;
        const int nsuffix = nreg + 1;               // registers + the zero "cls" token
        const int seq = ntok + nsuffix;
        const int lc = d->latent_c, pdim = d->patch_dim;
        const float eps = d->eps;
        if (heads <= 0 || hd <= 0 || dim != heads * hd)
        { set_last_error("MiniMaxH3VideoVaeDecode: bad head geometry."); return 0; }
        if (rot < 0 || rot > hd || (rot % 2) != 0)
        { set_last_error("MiniMaxH3VideoVaeDecode: bad rotary width."); return 0; }

        PooledContextHandle context;
        if (!context.init(32ull * 1024 * 1024))
        { set_last_error("MiniMaxH3VideoVaeDecode: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* data; std::size_t bytes; };
        std::vector<HostBinding> uploads;
        auto bind = [&](ggml_tensor* t, void* data, std::size_t bytes) {
            if (t == nullptr || data == nullptr) return;
            // Large weights go through the device buffer cache so a second call
            // reuses them instead of re-uploading gigabytes every chunk.
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs) &&
                    ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                {
                    if (needs) uploads.push_back({ t, data, bytes });
                    return;
                }
                invalidate_cached_buffer(data);
            }
            ggml_set_input(t);
            uploads.push_back({ t, data, bytes });
        };
        auto declLin = [&](const TSGH3Lin& s, ggml_tensor*& wt, ggml_tensor*& bt) {
            wt = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(s.type), s.ne0, s.ne1);
            // ggml owns the authoritative byte size for a quantized block layout;
            // trusting a caller-computed size here silently uploads garbage.
            bind(wt, s.w, ggml_nbytes(wt));
            bt = nullptr;
            if (s.b != nullptr)
            {
                bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.ne1);
                bind(bt, s.b, static_cast<std::size_t>(s.ne1) * sizeof(float));
            }
        };
        auto declVec = [&](void* data, int n) -> ggml_tensor* {
            if (data == nullptr) return nullptr;
            ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            bind(t, data, static_cast<std::size_t>(n) * sizeof(float));
            return t;
        };
        auto lin = [&](ggml_tensor* w, ggml_tensor* x, ggml_tensor* b) {
            ggml_tensor* o = ggml_mul_mat(ctx, w, x);
            return b ? ggml_add(ctx, o, b) : o;
        };

        ggml_tensor* latent = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lc, ntok);
        ggml_tensor* cosf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rot, seq);
        ggml_tensor* sinf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rot, seq);
        ggml_tensor* outT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, pdim, ntok);
        ggml_set_input(latent); ggml_set_input(cosf); ggml_set_input(sinf);

        ggml_tensor *pqw, *pqb, *xew, *xeb, *pow_, *pob;
        declLin(d->post_quant, pqw, pqb);
        declLin(d->x_embedder, xew, xeb);
        declLin(d->proj_out, pow_, pob);

        // post_quant_conv is a 1x1x1 Conv3d, i.e. a per-token linear.
        ggml_tensor* h = lin(pqw, latent, pqb);
        h = lin(xew, h, xeb);                                    // [dim, ntok]

        // Append the learned register tokens and the zero token. The reference
        // uses `torch.zeros_like`, not the `mask_token` parameter, which exists
        // in the checkpoint but is unused on this path.
        if (nreg > 0)
        {
            ggml_tensor* reg = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, nreg);
            bind(reg, d->register_tokens, static_cast<std::size_t>(dim) * nreg * sizeof(float));
            h = ggml_concat(ctx, h, reg, 1);
        }
        {
            ggml_tensor* zero = ggml_scale(ctx,
                ggml_cont(ctx, ggml_view_2d(ctx, h, dim, 1, h->nb[1], 0)), 0.0f);
            h = ggml_concat(ctx, h, zero, 1);
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        const std::size_t f = sizeof(float);

        for (int l = 0; l < nl; l++)
        {
            const TSGH3VitBlockW& bw = d->blocks[l];
            ggml_tensor* n1w = declVec(bw.norm1, dim);
            ggml_tensor* n2w = declVec(bw.norm2, dim);
            ggml_tensor* s1 = declVec(bw.scale1, dim);
            ggml_tensor* s2 = declVec(bw.scale2, dim);
            ggml_tensor *qkvw, *qkvb, *ow, *ob, *w1w, *w1b, *w2w, *w2b;
            declLin(bw.qkv, qkvw, qkvb);
            declLin(bw.out, ow, ob);
            declLin(bw.w1, w1w, w1b);
            declLin(bw.w2, w2w, w2b);

            // ---- attention ----
            ggml_tensor* n1 = ggml_mul(ctx, ggml_rms_norm(ctx, h, eps), n1w);
            ggml_tensor* qkv = lin(qkvw, n1, qkvb);              // [3*dim, seq]

            // Per-head interleaved: head h occupies [h*3*hd, (h+1)*3*hd).
            const std::size_t headStride = static_cast<std::size_t>(3) * hd * f;
            ggml_tensor* q = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
                                                         headStride, qkv->nb[1], 0));
            ggml_tensor* k = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
                                                         headStride, qkv->nb[1],
                                                         static_cast<std::size_t>(hd) * f));
            ggml_tensor* v = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
                                                         headStride, qkv->nb[1],
                                                         static_cast<std::size_t>(2) * hd * f));

            // qk-norm: RMSNorm over head_dim with no learnable gain.
            q = ggml_rms_norm(ctx, q, eps);
            k = ggml_rms_norm(ctx, k, eps);
            q = h3_vit_rope(ctx, q, cosf, sinf, hd, heads, seq, rot);
            k = h3_vit_rope(ctx, k, cosf, sinf, hd, heads, seq, rot);

            ggml_tensor* merged = h3_attend(ctx, g_backend, q, k, v, hd, heads, seq,
                                            scale, /*allowFlash*/ true);
            ggml_tensor* attn = lin(ow, merged, ob);
            h = ggml_add(ctx, h, s1 ? ggml_mul(ctx, attn, s1) : attn);

            // ---- gated FFN ----
            ggml_tensor* n2 = ggml_mul(ctx, ggml_rms_norm(ctx, h, eps), n2w);
            ggml_tensor* gv = lin(w1w, n2, w1b);                 // [2*inner, seq]
            ggml_tensor* gate = ggml_cont(ctx, ggml_view_2d(ctx, gv, inner, seq, gv->nb[1], 0));
            ggml_tensor* val = ggml_cont(ctx, ggml_view_2d(ctx, gv, inner, seq, gv->nb[1],
                                                            static_cast<std::size_t>(inner) * f));
            ggml_tensor* ff = lin(w2w, ggml_mul(ctx, ggml_silu(ctx, gate), val), w2b);
            h = ggml_add(ctx, h, s2 ? ggml_mul(ctx, ff, s2) : ff);
        }

        // norm_out is a LayerNorm (mean-centred, with bias), unlike the blocks' RMSNorm.
        ggml_tensor* now = declVec(d->norm_out_w, dim);
        ggml_tensor* nob = declVec(d->norm_out_b, dim);
        h = ggml_mul(ctx, ggml_norm(ctx, h, eps), now);
        if (nob) h = ggml_add(ctx, h, nob);

        h = lin(pow_, h, pob);                                   // [patch_dim, seq]
        // Drop the suffix tokens; only the real patches carry pixels.
        ggml_tensor* patches = ggml_cont(ctx, ggml_view_2d(ctx, h, pdim, ntok, h->nb[1], 0));
        ggml_tensor* copied = ggml_cpy(ctx, patches, outT);
        ggml_set_output(copied);

        const std::size_t nodes = static_cast<std::size_t>(nl) * 104 + 2048;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
        if (graph == nullptr)
        { set_last_error("MiniMaxH3VideoVaeDecode: graph alloc failed."); return 0; }
        ggml_build_forward_expand(graph, copied);

        for (int i = 0; i < ggml_graph_n_nodes(graph); i++)
        {
            if (!ggml_backend_supports_op(g_backend, ggml_graph_node(graph, i)))
            { set_last_error("MiniMaxH3VideoVaeDecode: op unsupported by backend."); return 0; }
        }

        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            { set_last_error("MiniMaxH3VideoVaeDecode: buffer alloc failed."); return 0; }
        }

        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.data, 0, u.bytes);
        ggml_backend_tensor_set(latent, d->latent, 0, static_cast<std::size_t>(lc) * ntok * f);
        ggml_backend_tensor_set(cosf, d->cosf, 0, static_cast<std::size_t>(rot) * seq * f);
        ggml_backend_tensor_set(sinf, d->sinf, 0, static_cast<std::size_t>(rot) * seq * f);

        if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        { set_last_error("MiniMaxH3VideoVaeDecode: graph compute failed."); return 0; }
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(outT, d->out, 0, static_cast<std::size_t>(pdim) * ntok * f);
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("MiniMaxH3VideoVaeDecode: unknown error."); return 0; }
}

// ============================================================================
// Qwen3-VL text encoder trunk.
//
// H3 conditions on Qwen3-VL-32B truncated to 50 language layers WITH THE FINAL
// NORM REMOVED -- the DiT consumes the raw layer-50 hidden state. There is no
// chat template and no lm_head; this is a pure prefill that returns hidden
// states.
//
// Differences from the original Qwen-Image (Qwen2.5-VL) trunk, since removed, that
// made a separate op worthwhile: this encoder adds per-head QK-RMSNorm, and the MLP keeps gate/up/down as
// three separate matrices. RoPE is supplied as host-built cos/sin tables, which
// also means interleaved M-RoPE needs no special casing here: for text tokens all
// three position axes are equal, so M-RoPE collapses to ordinary RoPE and the
// caller just builds the tables with theta 5e6.
// ============================================================================

struct TSGH3TeLayerW
{
    void* input_norm;        // [hidden] F32
    void* post_attn_norm;    // [hidden] F32
    void* q_norm;            // [head_dim] F32, nullable
    void* k_norm;            // [head_dim] F32, nullable
    TSGH3Lin q, k, v, o;
    TSGH3Lin gate, up, down;
};

struct TSGgmlMiniMaxH3TextEncodeDesc
{
    std::int32_t struct_bytes;
    std::int32_t num_layers;

    void* embeddings;        // [hidden, seq] F32 input hidden states
    void* out;               // [hidden, seq] F32 output
    void* cosf;              // [head_dim, seq] F32 rotate-half tables
    void* sinf;
    void* final_norm;        // [hidden] F32, nullable = skip (H3 skips it)
    // Qwen3-VL "DeepStack": residuals added after each of the first num_deepstack
    // layers. Dense [hidden, seq, num_deepstack] F32, zero outside the image spans,
    // and null when the prompt carries no reference images.
    void* deepstack;

    const TSGH3TeLayerW* layers;

    std::int32_t hidden;
    std::int32_t heads;
    std::int32_t kv_heads;
    std::int32_t head_dim;
    std::int32_t seq;
    std::int32_t causal;     // 1 = causal mask
    float eps;
    std::int32_t num_deepstack;
};

TSG_EXPORT int TSGgml_MiniMaxH3TextEncode(const TSGgmlMiniMaxH3TextEncodeDesc* d)
{
    try
    {
        if (d == nullptr ||
            d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlMiniMaxH3TextEncodeDesc)) ||
            d->embeddings == nullptr || d->out == nullptr || d->cosf == nullptr ||
            d->sinf == nullptr || d->layers == nullptr || d->num_layers <= 0 || d->seq <= 0)
        { set_last_error("MiniMaxH3TextEncode: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        const int nl = d->num_layers;
        const int hidden = d->hidden, heads = d->heads, kvh = d->kv_heads;
        const int hd = d->head_dim, seq = d->seq;
        const float eps = d->eps;
        if (heads <= 0 || kvh <= 0 || heads % kvh != 0 || hidden <= 0 || hd <= 0)
        { set_last_error("MiniMaxH3TextEncode: bad head geometry."); return 0; }
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

        PooledContextHandle context;
        if (!context.init(32ull * 1024 * 1024))
        { set_last_error("MiniMaxH3TextEncode: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* data; std::size_t bytes; };
        std::vector<HostBinding> uploads;
        auto bind = [&](ggml_tensor* t, void* data, std::size_t bytes) {
            if (t == nullptr || data == nullptr) return;
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs) &&
                    ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                {
                    if (needs) uploads.push_back({ t, data, bytes });
                    return;
                }
                invalidate_cached_buffer(data);
            }
            ggml_set_input(t);
            uploads.push_back({ t, data, bytes });
        };
        auto declLin = [&](const TSGH3Lin& s, ggml_tensor*& wt, ggml_tensor*& bt) {
            wt = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(s.type), s.ne0, s.ne1);
            // ggml owns the authoritative byte size for a quantized block layout;
            // trusting a caller-computed size here silently uploads garbage.
            bind(wt, s.w, ggml_nbytes(wt));
            bt = nullptr;
            if (s.b != nullptr)
            {
                bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.ne1);
                bind(bt, s.b, static_cast<std::size_t>(s.ne1) * sizeof(float));
            }
        };
        auto declVec = [&](void* data, int n) -> ggml_tensor* {
            if (data == nullptr) return nullptr;
            ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            bind(t, data, static_cast<std::size_t>(n) * sizeof(float));
            return t;
        };
        // The original Qwen-Image trunk prescaled EVERY quantized matmul, not just the two
        // with unbounded inputs; follow it rather than second-guessing which
        // activations stay inside q8_1's FP16 block-sum range.
        auto lin = [&](ggml_tensor* w, ggml_tensor* x, ggml_tensor* b, bool guard = true) {
            ggml_tensor* o = h3_mm(ctx, w, x, guard);
            return b ? ggml_add(ctx, o, b) : o;
        };

        ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, seq);
        ggml_tensor* cosf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, seq);
        ggml_tensor* sinf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, seq);
        ggml_tensor* outT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, seq);
        ggml_set_input(x); ggml_set_input(cosf); ggml_set_input(sinf);

        // An explicit additive mask, not ggml_diag_mask_inf: Metal implements no
        // diag-mask kernel, so the whole-trunk graph would fail its supports_op
        // sweep and fall back to the per-op path.
        ggml_tensor* causalMask = nullptr;
        std::vector<float> causalMaskData;
        if (d->causal != 0)
        {
            causalMask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq);
            ggml_set_input(causalMask);
            causalMaskData.assign(static_cast<std::size_t>(seq) * seq, 0.0f);
            const float neg_inf = -std::numeric_limits<float>::infinity();
            for (int q = 0; q < seq; q++)
                for (int k = q + 1; k < seq; k++)
                    causalMaskData[static_cast<std::size_t>(q) * seq + k] = neg_inf;
        }

        // Qwen3-VL injects the vision tower's DeepStack taps into the first layers.
        const int nds = d->deepstack != nullptr ? d->num_deepstack : 0;
        ggml_tensor* deepstack = nullptr;
        if (nds > 0)
        {
            deepstack = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, seq, nds);
            ggml_set_input(deepstack);
        }

        ggml_tensor* h = x;
        for (int l = 0; l < nl; l++)
        {
            const TSGH3TeLayerW& lw = d->layers[l];
            ggml_tensor* inNorm = declVec(lw.input_norm, hidden);
            ggml_tensor* paNorm = declVec(lw.post_attn_norm, hidden);
            ggml_tensor* qNorm = declVec(lw.q_norm, hd);
            ggml_tensor* kNorm = declVec(lw.k_norm, hd);
            ggml_tensor *qw, *qb, *kw, *kb, *vw, *vb, *ow, *ob, *gw, *gb, *uw, *ub, *dw, *db;
            declLin(lw.q, qw, qb); declLin(lw.k, kw, kb);
            declLin(lw.v, vw, vb); declLin(lw.o, ow, ob);
            declLin(lw.gate, gw, gb); declLin(lw.up, uw, ub); declLin(lw.down, dw, db);

            // ---- attention ----
            ggml_tensor* n1 = ggml_mul(ctx, ggml_rms_norm(ctx, h, eps), inNorm);
            ggml_tensor* q = ggml_reshape_3d(ctx, lin(qw, n1, qb), hd, heads, seq);
            ggml_tensor* k = ggml_reshape_3d(ctx, lin(kw, n1, kb), hd, kvh, seq);
            ggml_tensor* v = ggml_reshape_3d(ctx, lin(vw, n1, vb), hd, kvh, seq);

            // The text encoder normalizes Q and K per head BEFORE RoPE.
            if (qNorm) q = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), qNorm);
            if (kNorm) k = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), kNorm);

            q = h3_vit_rope(ctx, q, cosf, sinf, hd, heads, seq, hd);
            k = h3_vit_rope(ctx, k, cosf, sinf, hd, kvh, seq, hd);

            ggml_tensor* qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
            ggml_tensor* kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
            ggml_tensor* kq = ggml_mul_mat(ctx, kp, qp);              // GQA broadcasts kv heads
            ggml_tensor* probs = ggml_soft_max_ext(ctx, kq, causalMask, scale, 0.0f);
            ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
            ggml_tensor* kqv = ggml_mul_mat(ctx, vt, probs);
            ggml_tensor* merged = ggml_reshape_2d(ctx,
                ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), heads * hd, seq);
            h = ggml_add(ctx, h, lin(ow, merged, ob, /*guard*/ true));

            // ---- SwiGLU ----
            ggml_tensor* n2 = ggml_mul(ctx, ggml_rms_norm(ctx, h, eps), paNorm);
            ggml_tensor* g = lin(gw, n2, gb);
            ggml_tensor* u = lin(uw, n2, ub);
            h = ggml_add(ctx, h,
                         lin(dw, ggml_mul(ctx, ggml_silu(ctx, g), u), db, /*guard*/ true));

            if (l < nds)
                h = ggml_add(ctx, h, ggml_view_2d(ctx, deepstack, hidden, seq, deepstack->nb[1],
                                                  static_cast<std::size_t>(l) * deepstack->nb[2]));
        }

        // H3's checkpoint has no final norm; the DiT wants the raw layer-50 state.
        if (d->final_norm != nullptr)
        {
            ggml_tensor* fn = declVec(d->final_norm, hidden);
            h = ggml_mul(ctx, ggml_rms_norm(ctx, h, eps), fn);
        }

        ggml_tensor* copied = ggml_cpy(ctx, h, outT);
        ggml_set_output(copied);

        const std::size_t nodes = static_cast<std::size_t>(nl) * 104 + 2048;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
        if (graph == nullptr)
        { set_last_error("MiniMaxH3TextEncode: graph alloc failed."); return 0; }
        ggml_build_forward_expand(graph, copied);

        for (int i = 0; i < ggml_graph_n_nodes(graph); i++)
        {
            if (!ggml_backend_supports_op(g_backend, ggml_graph_node(graph, i)))
            { set_last_error("MiniMaxH3TextEncode: op unsupported by backend."); return 0; }
        }

        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            { set_last_error("MiniMaxH3TextEncode: buffer alloc failed."); return 0; }
        }

        const std::size_t f = sizeof(float);
        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.data, 0, u.bytes);
        ggml_backend_tensor_set(x, d->embeddings, 0, static_cast<std::size_t>(hidden) * seq * f);
        if (deepstack != nullptr)
            ggml_backend_tensor_set(deepstack, d->deepstack, 0, ggml_nbytes(deepstack));
        ggml_backend_tensor_set(cosf, d->cosf, 0, static_cast<std::size_t>(hd) * seq * f);
        ggml_backend_tensor_set(sinf, d->sinf, 0, static_cast<std::size_t>(hd) * seq * f);
        if (causalMask)
            ggml_backend_tensor_set(causalMask, causalMaskData.data(), 0, causalMaskData.size() * f);

        if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        { set_last_error("MiniMaxH3TextEncode: graph compute failed."); return 0; }
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(outT, d->out, 0, static_cast<std::size_t>(hidden) * seq * f);
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("MiniMaxH3TextEncode: unknown error."); return 0; }
}

// ============================================================================
// MiniMax-H3 diffusion transformer.
//
// Single-stream: text, conditioning frames, target audio and target video are ONE
// token sequence with full bidirectional self-attention and no cross-attention.
//
// Two details drive the shape of this code:
//
//  * AdaLN is per-SEGMENT, not per-token. Each block projects an 8-wide time
//    embedding to 18 x hidden = 6 modulation vectors for each of 3 modalities, and
//    a run of tokens picks one (timestep, modality) row. Broadcasting one row over
//    a segment keeps activations at [hidden, seq]; gathering per-token would
//    materialize six [hidden, seq] tensors per block instead.
//
//  * The released checkpoints have no timestep MLP. The caller interpolates the
//    learned adaln_t_table on the host and passes the resulting [8, nTimesteps]
//    embedding, which also means no SiLU here (the curve variant skips it).
//
// RoPE covers the first 96 of 128 head dims (3 axes x 16 learned inverse
// frequencies x 2) as rotate-half; the remaining 32 pass through.
// ============================================================================

// A run of tokens sharing one AdaLN row. `col` is the base column into the
// modulation matrix viewed as [hidden, 18 * nTimesteps]; parameter p sits at
// col + p.
// One run of conditioning tokens. `kind` picks the projection: 0 = video patches
// (sharing the target video's patch embedding), 1 = audio latents.
struct TSGH3CondChunk
{
    std::int32_t kind;
    std::int32_t count;
};

struct TSGH3DitSegment
{
    std::int32_t start;
    std::int32_t end;
    std::int32_t col;
    std::int32_t pad_;
};

// The token refiner reuses the block layout but is unmodulated, so it has no adaln.
struct TSGH3RefinerBlockW
{
    void* norm1;
    void* norm2;
    void* q_norm;
    void* k_norm;
    TSGH3Lin qkv;
    TSGH3Lin out;
    TSGH3Lin fc1;
    TSGH3Lin fc2;
};

struct TSGH3DitBlockW
{
    void* norm1;
    void* norm2;
    void* q_norm;
    void* k_norm;
    TSGH3Lin adaln;      // [timeEmbedDim, 18*hidden] + bias
    TSGH3Lin qkv;        // [hidden, 3*inner]
    TSGH3Lin out;        // [inner, hidden]
    TSGH3Lin fc1;        // [hidden, 2*ffn]  (gate | value)
    TSGH3Lin fc2;        // [ffn, hidden]
};

struct TSGgmlMiniMaxH3DitForwardDesc
{
    std::int32_t struct_bytes;
    std::int32_t num_blocks;
    std::int32_t num_refiner_blocks;
    std::int32_t num_segments;

    void* video_tokens;      // [videoPatchDim, videoCount] F32, already patchified
    void* audio_tokens;      // [audioChannels, audioCount] F32
    void* text_hidden;       // [textDim, textCount] F32
    void* time_embed;        // [timeEmbedDim, nTimesteps] F32
    void* cosf;              // [rotDim, nTok] F32
    void* sinf;
    void* video_out;         // [videoPatchDim, videoCount] F32 written
    void* audio_out;         // [audioChannels, audioCount] F32 written

    TSGH3Lin video_patch_proj;
    TSGH3Lin audio_patch_proj;
    TSGH3Lin condition_proj;

    const TSGH3RefinerBlockW* refiner;
    void* refiner_final_norm;
    const TSGH3DitBlockW* blocks;
    const TSGH3DitSegment* segments;

    void* final_norm;
    TSGH3Lin final_adaln;    // [timeEmbedDim, 2*hidden] + bias
    TSGH3Lin final_video_out;
    TSGH3Lin final_audio_out;

    std::int32_t n_tok;
    std::int32_t text_count;
    /// Conditioning video tokens, prepended to video_tokens and sharing its
    /// projection. They occupy the sequence between the text and the target audio.
    std::int32_t condition_count;
    // Ref2VA conditioning may interleave sound with pictures, so the conditioning
    // run is described as an ordered list of chunks rather than assumed to be one
    // block of video patches. Null/zero keeps the FL2VA behaviour: condition_count
    // rows of video patches and nothing else.
    void* condition_audio;              // [audio_channels, condition_audio_count] F32
    const TSGH3CondChunk* cond_chunks;
    std::int32_t num_cond_chunks;
    std::int32_t condition_audio_count;
    std::int32_t audio_start, audio_count;
    std::int32_t video_start, video_count;
    std::int32_t audio_col, video_col;   // final-layer column bases
    std::int32_t hidden, heads, head_dim, inner, ffn;
    std::int32_t rot_dim, time_embed_dim, n_timesteps;
    std::int32_t video_patch_dim, audio_channels, text_dim;
    float eps;
    float video_scale;       // -1
    float audio_scale;       // -d(sigma_audio)/d(sigma_video)
};

TSG_EXPORT int TSGgml_MiniMaxH3DitForward(const TSGgmlMiniMaxH3DitForwardDesc* d)
{
    try
    {
        if (d == nullptr ||
            d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlMiniMaxH3DitForwardDesc)) ||
            d->video_tokens == nullptr || d->audio_tokens == nullptr ||
            d->time_embed == nullptr || d->cosf == nullptr || d->sinf == nullptr ||
            d->video_out == nullptr || d->audio_out == nullptr ||
            d->blocks == nullptr || d->segments == nullptr || d->num_blocks <= 0 ||
            d->num_segments <= 0 || d->n_tok <= 0)
        { set_last_error("MiniMaxH3DitForward: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        const int nl = d->num_blocks, nref = d->num_refiner_blocks;
        const int hidden = d->hidden, heads = d->heads, hd = d->head_dim;
        const int inner = d->inner, ffn = d->ffn, rot = d->rot_dim;
        const int ntok = d->n_tok, nseg = d->num_segments;
        const int ted = d->time_embed_dim, nts = d->n_timesteps;
        const float eps = d->eps;
        if (heads <= 0 || hd <= 0 || inner != heads * hd)
        { set_last_error("MiniMaxH3DitForward: bad head geometry."); return 0; }
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        const std::size_t f = sizeof(float);

        // Validate the packed layout before building anything. Every one of these
        // would otherwise surface as a ggml_abort inside a view, which says nothing
        // about which segment or column was wrong.
        {
            const int modCols = 18 * nts;
            long long covered = 0;
            for (int i = 0; i < nseg; i++)
            {
                const TSGH3DitSegment& sg = d->segments[i];
                if (sg.start < 0 || sg.end < sg.start || sg.end > ntok)
                {
                    set_last_error("MiniMaxH3DitForward: segment " + std::to_string(i) +
                                   " range [" + std::to_string(sg.start) + "," +
                                   std::to_string(sg.end) + ") is outside [0," +
                                   std::to_string(ntok) + ").");
                    return 0;
                }
                if (sg.col < 0 || sg.col + 5 >= modCols)
                {
                    set_last_error("MiniMaxH3DitForward: segment " + std::to_string(i) +
                                   " column base " + std::to_string(sg.col) +
                                   " + 5 exceeds " + std::to_string(modCols) + ".");
                    return 0;
                }
                covered += sg.end - sg.start;
            }
            if (covered != ntok)
            {
                set_last_error("MiniMaxH3DitForward: segments cover " + std::to_string(covered) +
                               " tokens but the sequence has " + std::to_string(ntok) + ".");
                return 0;
            }
            if (d->audio_start < 0 || d->audio_start + d->audio_count > ntok ||
                d->video_start < 0 || d->video_start + d->video_count > ntok)
            { set_last_error("MiniMaxH3DitForward: audio/video span outside the sequence."); return 0; }
            if (d->audio_col + 1 >= 2 * nts || d->video_col + 1 >= 2 * nts ||
                d->audio_col < 0 || d->video_col < 0)
            {
                set_last_error("MiniMaxH3DitForward: final-layer column base out of range (audio " +
                               std::to_string(d->audio_col) + ", video " + std::to_string(d->video_col) +
                               ", columns " + std::to_string(2 * nts) + ").");
                return 0;
            }
            if (d->text_count < 0 || d->text_count > ntok)
            { set_last_error("MiniMaxH3DitForward: text_count outside the sequence."); return 0; }
        }

        PooledContextHandle context;
        if (!context.init(32ull * 1024 * 1024))
        { set_last_error("MiniMaxH3DitForward: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* data; std::size_t bytes; };
        std::vector<HostBinding> uploads;
        auto bind = [&](ggml_tensor* t, void* data, std::size_t bytes) {
            if (t == nullptr || data == nullptr) return;
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs) &&
                    ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                {
                    if (needs) uploads.push_back({ t, data, bytes });
                    return;
                }
                invalidate_cached_buffer(data);
            }
            ggml_set_input(t);
            uploads.push_back({ t, data, bytes });
        };
        auto declLin = [&](const TSGH3Lin& s, ggml_tensor*& wt, ggml_tensor*& bt) {
            wt = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(s.type), s.ne0, s.ne1);
            bind(wt, s.w, ggml_nbytes(wt));
            bt = nullptr;
            if (s.b != nullptr)
            {
                bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.ne1);
                bind(bt, s.b, static_cast<std::size_t>(s.ne1) * f);
            }
        };
        auto declVec = [&](void* data, int n) -> ggml_tensor* {
            if (data == nullptr) return nullptr;
            ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            bind(t, data, static_cast<std::size_t>(n) * f);
            return t;
        };
        auto lin = [&](ggml_tensor* w, ggml_tensor* x, ggml_tensor* b, bool guard = true) {
            ggml_tensor* o = h3_mm(ctx, w, x, guard);
            return b ? ggml_add(ctx, o, b) : o;
        };

        // ---- inputs ----
        const int condCount = d->condition_count;
        ggml_tensor* vin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d->video_patch_dim,
                                              condCount + d->video_count);
        ggml_tensor* ain = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d->audio_channels, d->audio_count);
        ggml_tensor* acin = nullptr;
        if (d->condition_audio != nullptr && d->condition_audio_count > 0)
        {
            acin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d->audio_channels,
                                      d->condition_audio_count);
            ggml_set_input(acin);
        }
        ggml_tensor* temb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ted, nts);
        ggml_tensor* cosf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rot, ntok);
        ggml_tensor* sinf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rot, ntok);
        ggml_set_input(vin); ggml_set_input(ain);
        ggml_set_input(temb); ggml_set_input(cosf); ggml_set_input(sinf);
        ggml_tensor* tin = nullptr;
        if (d->text_count > 0 && d->text_hidden != nullptr)
        {
            tin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d->text_dim, d->text_count);
            ggml_set_input(tin);
        }
        ggml_tensor* voutT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d->video_patch_dim, d->video_count);
        ggml_tensor* aoutT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d->audio_channels, d->audio_count);

        // Shared attention over a packed sequence: no mask anywhere in this model.
        // cos/sin are per-token tables over the FULL packed sequence; the token
        // refiner runs on the text prefix only, so it needs the matching prefix of
        // the tables rather than the whole thing.
        auto attention = [&](ggml_tensor* h, ggml_tensor* qkvw, ggml_tensor* qkvb,
                             ggml_tensor* qn, ggml_tensor* kn, ggml_tensor* ow, ggml_tensor* ob,
                             int seq, ggml_tensor* cosT, ggml_tensor* sinT) {
            ggml_tensor* qkv = lin(qkvw, h, qkvb);                 // [3*inner, seq]
            ggml_tensor* q = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
                static_cast<std::size_t>(hd) * f, qkv->nb[1], 0));
            ggml_tensor* k = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
                static_cast<std::size_t>(hd) * f, qkv->nb[1],
                static_cast<std::size_t>(inner) * f));
            ggml_tensor* v = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
                static_cast<std::size_t>(hd) * f, qkv->nb[1],
                static_cast<std::size_t>(2) * inner * f));
            if (qn) q = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps), qn);
            if (kn) k = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps), kn);
            q = h3_vit_rope(ctx, q, cosT, sinT, hd, heads, seq, rot);
            k = h3_vit_rope(ctx, k, cosT, sinT, hd, heads, seq, rot);
            ggml_tensor* merged = h3_attend(ctx, g_backend, q, k, v, hd, heads, seq,
                                            scale, /*allowFlash*/ true);
            return lin(ow, merged, ob);
        };

        // SwiGLU with the gate first: fc1 emits [gate | value].
        auto mlp = [&](ggml_tensor* h, ggml_tensor* w1, ggml_tensor* b1,
                       ggml_tensor* w2, ggml_tensor* b2) {
            ggml_tensor* gv = lin(w1, h, b1);
            int seq = static_cast<int>(gv->ne[1]);
            ggml_tensor* gate = ggml_cont(ctx, ggml_view_2d(ctx, gv, ffn, seq, gv->nb[1], 0));
            ggml_tensor* val = ggml_cont(ctx, ggml_view_2d(ctx, gv, ffn, seq, gv->nb[1],
                static_cast<std::size_t>(ffn) * f));
            return lin(w2, ggml_mul(ctx, ggml_silu(ctx, gate), val), b2);
        };

        // ---- text: project, then refine ----
        ggml_tensor* ctxTok = nullptr;
        if (tin != nullptr)
        {
            ggml_tensor *cpw, *cpb;
            declLin(d->condition_proj, cpw, cpb);
            ctxTok = lin(cpw, tin, cpb);
            ggml_tensor* tcos = ggml_cont(ctx, ggml_view_2d(ctx, cosf, rot, d->text_count,
                                                            cosf->nb[1], 0));
            ggml_tensor* tsin = ggml_cont(ctx, ggml_view_2d(ctx, sinf, rot, d->text_count,
                                                            sinf->nb[1], 0));
            for (int i = 0; i < nref; i++)
            {
                const TSGH3RefinerBlockW& rw = d->refiner[i];
                ggml_tensor* n1w = declVec(rw.norm1, hidden);
                ggml_tensor* n2w = declVec(rw.norm2, hidden);
                ggml_tensor* qn = declVec(rw.q_norm, hd);
                ggml_tensor* kn = declVec(rw.k_norm, hd);
                ggml_tensor *qkvw, *qkvb, *ow, *ob, *w1, *b1, *w2, *b2;
                declLin(rw.qkv, qkvw, qkvb); declLin(rw.out, ow, ob);
                declLin(rw.fc1, w1, b1); declLin(rw.fc2, w2, b2);
                ggml_tensor* n1 = ggml_mul(ctx, ggml_rms_norm(ctx, ctxTok, eps), n1w);
                ctxTok = ggml_add(ctx, ctxTok,
                    attention(n1, qkvw, qkvb, qn, kn, ow, ob, d->text_count, tcos, tsin));
                ggml_tensor* n2 = ggml_mul(ctx, ggml_rms_norm(ctx, ctxTok, eps), n2w);
                ctxTok = ggml_add(ctx, ctxTok, mlp(n2, w1, b1, w2, b2));
            }
            ggml_tensor* rfn = declVec(d->refiner_final_norm, hidden);
            if (rfn) ctxTok = ggml_mul(ctx, ggml_rms_norm(ctx, ctxTok, eps), rfn);
        }

        // ---- patchify video and audio, then pack ----
        ggml_tensor *vpw, *vpb, *apw, *apb;
        declLin(d->video_patch_proj, vpw, vpb);
        declLin(d->audio_patch_proj, apw, apb);
        // Conditioning frames share the target's patch projection, exactly as the
        // reference concatenates them before projecting once.
        ggml_tensor* allVideo = lin(vpw, vin, vpb);
        ggml_tensor* audioTok = lin(apw, ain, apb);

        ggml_tensor* condAudioTok = acin != nullptr ? lin(apw, acin, apb) : nullptr;

        ggml_tensor* condTok = nullptr;
        ggml_tensor* videoTok = allVideo;
        if (condCount > 0)
        {
            condTok = ggml_cont(ctx, ggml_view_2d(ctx, allVideo, hidden, condCount,
                                                  allVideo->nb[1], 0));
            videoTok = ggml_cont(ctx, ggml_view_2d(ctx, allVideo, hidden, d->video_count,
                                                   allVideo->nb[1],
                                                   static_cast<std::size_t>(condCount) * allVideo->nb[1]));
        }
        // With a chunk list the conditioning run is rebuilt in the caller's order,
        // taking each chunk from whichever projection it named.
        if (d->cond_chunks != nullptr && d->num_cond_chunks > 0)
        {
            ggml_tensor* rebuilt = nullptr;
            int videoRow = 0, audioRow = 0;
            for (int i = 0; i < d->num_cond_chunks; i++)
            {
                const TSGH3CondChunk& ch = d->cond_chunks[i];
                if (ch.count <= 0) continue;
                ggml_tensor* part;
                if (ch.kind == 0)
                {
                    if (condTok == nullptr || videoRow + ch.count > condCount)
                    { set_last_error("MiniMaxH3DitForward: conditioning video chunks overrun."); return 0; }
                    part = ggml_cont(ctx, ggml_view_2d(ctx, condTok, hidden, ch.count,
                        condTok->nb[1], static_cast<std::size_t>(videoRow) * condTok->nb[1]));
                    videoRow += ch.count;
                }
                else
                {
                    if (condAudioTok == nullptr || audioRow + ch.count > d->condition_audio_count)
                    { set_last_error("MiniMaxH3DitForward: conditioning audio chunks overrun."); return 0; }
                    part = ggml_cont(ctx, ggml_view_2d(ctx, condAudioTok, hidden, ch.count,
                        condAudioTok->nb[1], static_cast<std::size_t>(audioRow) * condAudioTok->nb[1]));
                    audioRow += ch.count;
                }
                rebuilt = rebuilt == nullptr ? part : ggml_concat(ctx, rebuilt, part, 1);
            }
            if (videoRow != condCount || audioRow != d->condition_audio_count)
            { set_last_error("MiniMaxH3DitForward: conditioning chunks do not cover every row."); return 0; }
            condTok = rebuilt;
        }

        // Sequence order is fixed: text, conditioning frames, target audio, target video.
        ggml_tensor* x = ctxTok;
        auto append = [&](ggml_tensor* v) { x = x == nullptr ? v : ggml_concat(ctx, x, v, 1); };
        if (condTok != nullptr) append(condTok);
        append(audioTok);
        append(videoTok);

        // Apply one modulation parameter per segment, broadcasting its single row.
        // `mods` is the block's projection viewed as [hidden, 18*nTimesteps].
        auto modColumn = [&](ggml_tensor* mods, int col) {
            return ggml_view_2d(ctx, mods, hidden, 1, mods->nb[1],
                                static_cast<std::size_t>(col) * mods->nb[1]);
        };
        auto modulate = [&](ggml_tensor* src, ggml_tensor* mods, int shiftIdx, int scaleIdx) {
            ggml_tensor* out = nullptr;
            for (int i = 0; i < nseg; i++)
            {
                const TSGH3DitSegment& sg = d->segments[i];
                int n = sg.end - sg.start;
                if (n <= 0) continue;
                ggml_tensor* part = ggml_cont(ctx, ggml_view_2d(ctx, src, hidden, n, src->nb[1],
                    static_cast<std::size_t>(sg.start) * src->nb[1]));
                ggml_tensor* sh = modColumn(mods, sg.col + shiftIdx);
                ggml_tensor* sc = modColumn(mods, sg.col + scaleIdx);
                part = ggml_add(ctx, ggml_add(ctx, part, ggml_mul(ctx, part, sc)), sh);
                out = out == nullptr ? part : ggml_concat(ctx, out, part, 1);
            }
            return out;
        };
        auto gatedResidual = [&](ggml_tensor* base, ggml_tensor* upd,
                                 ggml_tensor* mods, int gateIdx) {
            ggml_tensor* out = nullptr;
            for (int i = 0; i < nseg; i++)
            {
                const TSGH3DitSegment& sg = d->segments[i];
                int n = sg.end - sg.start;
                if (n <= 0) continue;
                ggml_tensor* b = ggml_cont(ctx, ggml_view_2d(ctx, base, hidden, n, base->nb[1],
                    static_cast<std::size_t>(sg.start) * base->nb[1]));
                ggml_tensor* u = ggml_cont(ctx, ggml_view_2d(ctx, upd, hidden, n, upd->nb[1],
                    static_cast<std::size_t>(sg.start) * upd->nb[1]));
                ggml_tensor* g = modColumn(mods, sg.col + gateIdx);
                ggml_tensor* part = ggml_add(ctx, b, ggml_mul(ctx, u, g));
                out = out == nullptr ? part : ggml_concat(ctx, out, part, 1);
            }
            return out;
        };

        for (int l = 0; l < nl; l++)
        {
            const TSGH3DitBlockW& bw = d->blocks[l];
            ggml_tensor* n1w = declVec(bw.norm1, hidden);
            ggml_tensor* n2w = declVec(bw.norm2, hidden);
            ggml_tensor* qn = declVec(bw.q_norm, hd);
            ggml_tensor* kn = declVec(bw.k_norm, hd);
            ggml_tensor *adw, *adb, *qkvw, *qkvb, *ow, *ob, *w1, *b1, *w2, *b2;
            declLin(bw.adaln, adw, adb); declLin(bw.qkv, qkvw, qkvb);
            declLin(bw.out, ow, ob); declLin(bw.fc1, w1, b1); declLin(bw.fc2, w2, b2);

            // No SiLU: the curve-table variant feeds the interpolated embedding straight in.
            ggml_tensor* mods = ggml_reshape_2d(ctx, lin(adw, temb, adb, false), hidden, 18 * nts);

            ggml_tensor* h = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), n1w);
            h = modulate(h, mods, 0, 1);
            ggml_tensor* a = attention(h, qkvw, qkvb, qn, kn, ow, ob, ntok, cosf, sinf);
            x = gatedResidual(x, a, mods, 2);

            h = ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), n2w);
            h = modulate(h, mods, 3, 4);
            ggml_tensor* m = mlp(h, w1, b1, w2, b2);
            x = gatedResidual(x, m, mods, 5);
        }

        // ---- final layer ----
        // Its projection emits only shift+scale, and the spans index it by TIMESTEP
        // row rather than timestep*3+modality, so the column bases differ.
        ggml_tensor *faw, *fab;
        declLin(d->final_adaln, faw, fab);
        ggml_tensor* fmods = ggml_reshape_2d(ctx, lin(faw, temb, fab, false), hidden, 2 * nts);
        ggml_tensor* fnw = declVec(d->final_norm, hidden);

        auto finalSlice = [&](int start, int count, int col) {
            ggml_tensor* part = ggml_cont(ctx, ggml_view_2d(ctx, x, hidden, count, x->nb[1],
                static_cast<std::size_t>(start) * x->nb[1]));
            part = ggml_mul(ctx, ggml_rms_norm(ctx, part, eps), fnw);
            ggml_tensor* sh = ggml_view_2d(ctx, fmods, hidden, 1, fmods->nb[1],
                static_cast<std::size_t>(col) * fmods->nb[1]);
            ggml_tensor* sc = ggml_view_2d(ctx, fmods, hidden, 1, fmods->nb[1],
                static_cast<std::size_t>(col + 1) * fmods->nb[1]);
            return ggml_add(ctx, ggml_add(ctx, part, ggml_mul(ctx, part, sc)), sh);
        };

        ggml_tensor *fvw, *fvb, *faw2, *fab2;
        declLin(d->final_video_out, fvw, fvb);
        declLin(d->final_audio_out, faw2, fab2);

        ggml_tensor* videoOut = lin(fvw, finalSlice(d->video_start, d->video_count, d->video_col), fvb);
        ggml_tensor* audioOut = lin(faw2, finalSlice(d->audio_start, d->audio_count, d->audio_col), fab2);

        // The model emits velocities: video negated, audio scaled by
        // d(sigma_audio)/d(sigma_video) so one Euler step advances both streams.
        videoOut = ggml_scale(ctx, videoOut, d->video_scale);
        audioOut = ggml_scale(ctx, audioOut, d->audio_scale);

        ggml_tensor* vcopy = ggml_cpy(ctx, videoOut, voutT);
        ggml_tensor* acopy = ggml_cpy(ctx, audioOut, aoutT);
        ggml_set_output(vcopy); ggml_set_output(acopy);

        // 104 rather than 96 per layer: h3_attend's V pre-scale and output
        // re-scale are two more nodes per attention, and the budget is an
        // upper bound rather than a count.
        //
        // 32 rather than 24 per SEGMENT because that is what the segment loops
        // actually cost. modulate() emits, per segment, a view + cont of the
        // slice, two modColumn views, a mul, two adds and (past the first) a
        // concat = 8; gatedResidual() emits two view+cont pairs, one modColumn
        // view, a mul, an add and a concat = 8. A block runs two of each, so a
        // layer spends 32 * nseg on modulation alone. At 24 the shortfall was
        // 8 * nseg - 24 per layer, hidden by the flat 4096 for as long as the
        // segment count stayed small: fl2va has about six segments, but Ref2VA
        // adds one or two per reference block plus one per vision span in the
        // prompt, and at fourteen segments a 50-layer graph runs past the end
        // and ggml_build_forward_expand trips GGML_ASSERT(n_nodes < size),
        // which aborts the process rather than returning an error.
        const std::size_t nodes =
            static_cast<std::size_t>(nl) * (104 + static_cast<std::size_t>(nseg) * 32) +
            static_cast<std::size_t>(nref) * 104 + 4096;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
        if (graph == nullptr)
        { set_last_error("MiniMaxH3DitForward: graph alloc failed."); return 0; }
        ggml_build_forward_expand(graph, vcopy);
        ggml_build_forward_expand(graph, acopy);

        for (int i = 0; i < ggml_graph_n_nodes(graph); i++)
        {
            if (!ggml_backend_supports_op(g_backend, ggml_graph_node(graph, i)))
            { set_last_error("MiniMaxH3DitForward: op unsupported by backend."); return 0; }
        }

        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            { set_last_error("MiniMaxH3DitForward: buffer alloc failed."); return 0; }
        }

        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.data, 0, u.bytes);
        ggml_backend_tensor_set(vin, d->video_tokens, 0, ggml_nbytes(vin));
        ggml_backend_tensor_set(ain, d->audio_tokens, 0, ggml_nbytes(ain));
        if (acin != nullptr)
            ggml_backend_tensor_set(acin, d->condition_audio, 0, ggml_nbytes(acin));
        ggml_backend_tensor_set(temb, d->time_embed, 0, ggml_nbytes(temb));
        ggml_backend_tensor_set(cosf, d->cosf, 0, ggml_nbytes(cosf));
        ggml_backend_tensor_set(sinf, d->sinf, 0, ggml_nbytes(sinf));
        if (tin) ggml_backend_tensor_set(tin, d->text_hidden, 0, ggml_nbytes(tin));

        if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        { set_last_error("MiniMaxH3DitForward: graph compute failed."); return 0; }
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(voutT, d->video_out, 0, ggml_nbytes(voutT));
        ggml_backend_tensor_get(aoutT, d->audio_out, 0, ggml_nbytes(aoutT));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("MiniMaxH3DitForward: unknown error."); return 0; }
}

// ============================================================================
// MiniMax-H3 video VAE ENCODER (single frame).
//
// Needed for image conditioning: an init/end frame has to become a latent before
// it can be presented to the DiT as conditioning tokens.
//
// The encoder is a causal 3-D CNN, but for a SINGLE frame the causal padding is
// two leading ZERO frames, so only the last temporal slice of each 3x3x3 kernel
// contributes:  conv3d([0, 0, x]) == conv2d(W[:, :, 2], x).  This op therefore
// runs an exact 2-D reduction of the 3-D network -- the caller slices kt=2 out of
// each kernel. Nothing here is an approximation.
//
// Spatial padding is REFLECT (the config's padding_mode), which ggml's conv has no
// mode for, so it is materialized explicitly. The downsamples pad only the right
// and bottom edge by one before a stride-2 conv, matching Downsample3D.
// ============================================================================

// A 2-D convolution kernel, ggml order [KW, KH, IC, OC].
struct TSGH3Conv
{
    void* w;
    void* b;                 // nullable, F32 [OC]
    std::int64_t kw, kh, ic, oc;
    std::int32_t type;
    std::int32_t pad_;
};

struct TSGH3EncResBlock
{
    void* norm1_w; void* norm1_b;    // [in_ch]  F32
    void* norm2_w; void* norm2_b;    // [out_ch] F32
    TSGH3Conv conv1, conv2;
    TSGH3Conv shortcut;              // .w null when in_ch == out_ch
};

struct TSGH3EncLevel
{
    TSGH3EncResBlock block0, block1;
    TSGH3Conv downsample;            // .w null when the level does not downsample
    std::int32_t space_stride;
    std::int32_t pad_;
};

struct TSGgmlMiniMaxH3VideoVaeEncodeDesc
{
    std::int32_t struct_bytes;
    std::int32_t num_levels;

    void* image;             // [W, H, 3] F32, ImageNet-normalized
    void* out;               // [W/16, H/16, latent_channels] F32

    TSGH3Conv conv_in;
    const TSGH3EncLevel* levels;
    void* norm_out_w; void* norm_out_b;
    TSGH3Conv conv_out;      // -> 2*z (mean | logvar)
    TSGH3Conv quant_conv;    // 1x1

    std::int32_t width, height;
    std::int32_t latent_channels;
    std::int32_t groups;
    float eps;
};

namespace {

// Reflect-pad an [W, H, C] tensor. PyTorch reflect with pad 1 mirrors about the
// edge WITHOUT repeating it, i.e. the new first column is old column 1.
ggml_tensor* h3_reflect_pad(ggml_context* ctx, ggml_tensor* x,
                            int left, int right, int top, int bottom)
{
    const std::size_t f = ggml_element_size(x);
    if (left > 0 || right > 0)
    {
        std::vector<ggml_tensor*> parts;
        for (int i = left; i >= 1; --i)
            parts.push_back(ggml_cont(ctx, ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2],
                                                        x->nb[1], x->nb[2],
                                                        static_cast<std::size_t>(i) * f)));
        parts.push_back(x);
        for (int i = 1; i <= right; ++i)
            parts.push_back(ggml_cont(ctx, ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2],
                                                        x->nb[1], x->nb[2],
                                                        static_cast<std::size_t>(x->ne[0] - 1 - i) * f)));
        ggml_tensor* acc = parts[0];
        for (std::size_t i = 1; i < parts.size(); ++i) acc = ggml_concat(ctx, acc, parts[i], 0);
        x = acc;
    }
    if (top > 0 || bottom > 0)
    {
        std::vector<ggml_tensor*> parts;
        for (int i = top; i >= 1; --i)
            parts.push_back(ggml_cont(ctx, ggml_view_3d(ctx, x, x->ne[0], 1, x->ne[2],
                                                        x->nb[1], x->nb[2],
                                                        static_cast<std::size_t>(i) * x->nb[1])));
        parts.push_back(x);
        for (int i = 1; i <= bottom; ++i)
            parts.push_back(ggml_cont(ctx, ggml_view_3d(ctx, x, x->ne[0], 1, x->ne[2],
                                                        x->nb[1], x->nb[2],
                                                        static_cast<std::size_t>(x->ne[1] - 1 - i) * x->nb[1])));
        ggml_tensor* acc = parts[0];
        for (std::size_t i = 1; i < parts.size(); ++i) acc = ggml_concat(ctx, acc, parts[i], 1);
        x = acc;
    }
    return x;
}

} // namespace

TSG_EXPORT int TSGgml_MiniMaxH3VideoVaeEncode(const TSGgmlMiniMaxH3VideoVaeEncodeDesc* d)
{
    try
    {
        if (d == nullptr ||
            d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlMiniMaxH3VideoVaeEncodeDesc)) ||
            d->image == nullptr || d->out == nullptr || d->levels == nullptr ||
            d->num_levels <= 0 || d->width <= 0 || d->height <= 0)
        { set_last_error("MiniMaxH3VideoVaeEncode: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        PooledContextHandle context;
        if (!context.init(32ull * 1024 * 1024))
        { set_last_error("MiniMaxH3VideoVaeEncode: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* data; std::size_t bytes; };
        std::vector<HostBinding> uploads;
        auto bind = [&](ggml_tensor* t, void* data, std::size_t bytes) {
            if (t == nullptr || data == nullptr) return;
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs) &&
                    ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                {
                    if (needs) uploads.push_back({ t, data, bytes });
                    return;
                }
                invalidate_cached_buffer(data);
            }
            ggml_set_input(t);
            uploads.push_back({ t, data, bytes });
        };
        auto declVec = [&](void* data, int n) -> ggml_tensor* {
            if (data == nullptr) return nullptr;
            ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            bind(t, data, static_cast<std::size_t>(n) * sizeof(float));
            return t;
        };
        struct Conv { ggml_tensor* w; ggml_tensor* b; };
        auto declConv = [&](const TSGH3Conv& c) -> Conv {
            if (c.w == nullptr) return { nullptr, nullptr };
            ggml_tensor* w = ggml_new_tensor_4d(ctx, static_cast<ggml_type>(c.type),
                                                c.kw, c.kh, c.ic, c.oc);
            bind(w, c.w, ggml_nbytes(w));
            ggml_tensor* b = nullptr;
            if (c.b != nullptr)
            {
                b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.oc);
                bind(b, c.b, static_cast<std::size_t>(c.oc) * sizeof(float));
            }
            return { w, b };
        };
        // conv + bias, with the bias broadcast over the spatial plane.
        auto applyConv = [&](const Conv& c, ggml_tensor* x, int stride, int pad) {
            ggml_tensor* o = ggml_conv_2d(ctx, c.w, x, stride, stride, pad, pad, 1, 1);
            if (c.b) o = ggml_add(ctx, o, ggml_reshape_3d(ctx, c.b, 1, 1, c.b->ne[0]));
            return o;
        };
        auto groupNormSilu = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
            ggml_tensor* n = ggml_group_norm(ctx, x, d->groups, d->eps);
            if (w) n = ggml_mul(ctx, n, ggml_reshape_3d(ctx, w, 1, 1, w->ne[0]));
            if (b) n = ggml_add(ctx, n, ggml_reshape_3d(ctx, b, 1, 1, b->ne[0]));
            return ggml_silu(ctx, n);
        };

        ggml_tensor* img = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d->width, d->height, 3);
        ggml_set_input(img);

        Conv convIn = declConv(d->conv_in);
        ggml_tensor* h = applyConv(convIn, h3_reflect_pad(ctx, img, 1, 1, 1, 1), 1, 0);

        for (int l = 0; l < d->num_levels; l++)
        {
            const TSGH3EncLevel& lv = d->levels[l];
            for (int bi = 0; bi < 2; bi++)
            {
                const TSGH3EncResBlock& rb = bi == 0 ? lv.block0 : lv.block1;
                ggml_tensor* n1w = declVec(rb.norm1_w, static_cast<int>(rb.conv1.ic));
                ggml_tensor* n1b = declVec(rb.norm1_b, static_cast<int>(rb.conv1.ic));
                ggml_tensor* n2w = declVec(rb.norm2_w, static_cast<int>(rb.conv2.ic));
                ggml_tensor* n2b = declVec(rb.norm2_b, static_cast<int>(rb.conv2.ic));
                Conv c1 = declConv(rb.conv1), c2 = declConv(rb.conv2);
                Conv sc = declConv(rb.shortcut);

                ggml_tensor* t = groupNormSilu(h, n1w, n1b);
                t = applyConv(c1, h3_reflect_pad(ctx, t, 1, 1, 1, 1), 1, 0);
                t = groupNormSilu(t, n2w, n2b);
                t = applyConv(c2, h3_reflect_pad(ctx, t, 1, 1, 1, 1), 1, 0);
                ggml_tensor* skip = sc.w ? applyConv(sc, h, 1, 0) : h;
                h = ggml_add(ctx, skip, t);
            }
            if (lv.downsample.w != nullptr)
            {
                // Downsample3D pads only the right/bottom edge by one, then strides.
                Conv ds = declConv(lv.downsample);
                ggml_tensor* padded = lv.space_stride == 2
                    ? h3_reflect_pad(ctx, h, 0, 1, 0, 1)
                    : h;
                h = applyConv(ds, padded, lv.space_stride, 0);
            }
        }

        ggml_tensor* noW = declVec(d->norm_out_w, static_cast<int>(d->conv_out.ic));
        ggml_tensor* noB = declVec(d->norm_out_b, static_cast<int>(d->conv_out.ic));
        h = groupNormSilu(h, noW, noB);
        Conv convOut = declConv(d->conv_out);
        h = applyConv(convOut, h3_reflect_pad(ctx, h, 1, 1, 1, 1), 1, 0);

        Conv quant = declConv(d->quant_conv);
        if (quant.w) h = applyConv(quant, h, 1, 0);

        // Keep the posterior MEAN only; the reference never samples at inference.
        const int lc = d->latent_channels;
        ggml_tensor* mean = ggml_cont(ctx, ggml_view_3d(ctx, h, h->ne[0], h->ne[1], lc,
                                                        h->nb[1], h->nb[2], 0));
        ggml_tensor* outT = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, mean->ne[0], mean->ne[1], lc);
        ggml_tensor* copied = ggml_cpy(ctx, mean, outT);
        ggml_set_output(copied);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);
        if (graph == nullptr)
        { set_last_error("MiniMaxH3VideoVaeEncode: graph alloc failed."); return 0; }
        ggml_build_forward_expand(graph, copied);

        for (int i = 0; i < ggml_graph_n_nodes(graph); i++)
        {
            if (!ggml_backend_supports_op(g_backend, ggml_graph_node(graph, i)))
            { set_last_error("MiniMaxH3VideoVaeEncode: op unsupported by backend."); return 0; }
        }

        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            { set_last_error("MiniMaxH3VideoVaeEncode: buffer alloc failed."); return 0; }
        }

        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.data, 0, u.bytes);
        ggml_backend_tensor_set(img, d->image, 0, ggml_nbytes(img));

        if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        { set_last_error("MiniMaxH3VideoVaeEncode: graph compute failed."); return 0; }
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(outT, d->out, 0, ggml_nbytes(outT));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("MiniMaxH3VideoVaeEncode: unknown error."); return 0; }
}

// ============================================================================
// MiniMax-H3 audio VAE (BigVGAN) decoder.
//
// 32 latent channels at 40 Hz -> mono 32 kHz PCM; the caller runs it once per
// stereo plane. Seven upsample stages multiply to 800, which is exactly
// 32000 / 40.
//
// The awkward part is the anti-aliased activation. BigVGAN wraps every SnakeBeta
// in a 2x upsample / activate / 2x downsample sandwich so the nonlinearity cannot
// fold high frequencies back into the band. Both resamples are DEPTHWISE with a
// filter SHARED across channels, which ggml has no grouped op for -- but a shared
// filter means the channel axis can simply be treated as a batch of 1-channel
// signals, so a plain conv with Cin = Cout = 1 does the job exactly.
//
// The kaiser filters ship inside the checkpoint, so nothing here has to
// reconstruct a window.
// ============================================================================

struct TSGH3Conv1d
{
    void* w;                 // [K, IC, OC] ggml order
    void* b;                 // nullable F32
    std::int64_t k, ic, oc;
    // A transposed conv's weight is [Cin, Cout, K] in torch, so reversing the dims
    // puts Cout in `ic` while its bias is still Cout long. Carrying the bias length
    // explicitly avoids having to special-case that at every use.
    std::int64_t bias_len;
    std::int32_t type;
    std::int32_t stride, padding, dilation;
};

// Alias-free SnakeBeta. alpha/beta are LOG-scale, per channel.
struct TSGH3Act1d
{
    void* alpha;             // [C] F32, already exp()'d
    void* beta;              // [C] F32, already exp()'d with the eps guard folded in
    void* up_filter;         // [K, 1, 1] F32, kaiser, PRE-REVERSED (see the op)
    void* down_filter;       // [K, 1, 1] F32
    std::int32_t channels;
    std::int32_t kernel;
};

struct TSGgmlMiniMaxH3AudioVaeDecodeDesc
{
    std::int32_t struct_bytes;
    std::int32_t num_stages;
    std::int32_t num_convs;   // ups + per-amp convs, in the documented order
    std::int32_t num_acts;

    void* latent;             // [T, latent_channels] F32
    void* out;                // [samples] F32

    TSGH3Conv1d dec_in_proj;
    TSGH3Conv1d conv_pre;
    TSGH3Conv1d conv_post;

    // convs: [num_stages ups][per (stage, amp): convs1[3] then convs2[3]]
    const TSGH3Conv1d* convs;
    // acts:  [per (stage, amp): 6][activation_post]
    const TSGH3Act1d* acts;
    const std::int32_t* rates;      // [num_stages]

    std::int32_t latent_len;
    std::int32_t latent_channels;
    std::int32_t amps_per_stage;
    std::int32_t samples;
    float snake_eps;
};

namespace {

// Replicate-pad along the time axis of a [T, C, 1] tensor.
ggml_tensor* h3_replicate_pad_1d(ggml_context* ctx, ggml_tensor* x, int left, int right)
{
    if (left <= 0 && right <= 0) return x;
    const std::size_t f = ggml_element_size(x);
    std::vector<ggml_tensor*> parts;
    auto column = [&](int index) {
        return ggml_cont(ctx, ggml_view_3d(ctx, x, 1, x->ne[1], x->ne[2],
                                           x->nb[1], x->nb[2],
                                           static_cast<std::size_t>(index) * f));
    };
    for (int i = 0; i < left; ++i) parts.push_back(column(0));
    parts.push_back(x);
    for (int i = 0; i < right; ++i) parts.push_back(column(static_cast<int>(x->ne[0]) - 1));
    ggml_tensor* acc = parts[0];
    for (std::size_t i = 1; i < parts.size(); ++i) acc = ggml_concat(ctx, acc, parts[i], 0);
    return acc;
}

} // namespace

TSG_EXPORT int TSGgml_MiniMaxH3AudioVaeDecode(const TSGgmlMiniMaxH3AudioVaeDecodeDesc* d)
{
    try
    {
        if (d == nullptr ||
            d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlMiniMaxH3AudioVaeDecodeDesc)) ||
            d->latent == nullptr || d->out == nullptr || d->convs == nullptr ||
            d->acts == nullptr || d->rates == nullptr || d->num_stages <= 0)
        { set_last_error("MiniMaxH3AudioVaeDecode: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        PooledContextHandle context;
        if (!context.init(32ull * 1024 * 1024))
        { set_last_error("MiniMaxH3AudioVaeDecode: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* data; std::size_t bytes; };
        std::vector<HostBinding> uploads;
        auto bind = [&](ggml_tensor* t, void* data, std::size_t bytes) {
            if (t == nullptr || data == nullptr) return;
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs) &&
                    ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                {
                    if (needs) uploads.push_back({ t, data, bytes });
                    return;
                }
                invalidate_cached_buffer(data);
            }
            ggml_set_input(t);
            uploads.push_back({ t, data, bytes });
        };
        auto declVec = [&](void* data, int n) -> ggml_tensor* {
            if (data == nullptr) return nullptr;
            ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            bind(t, data, static_cast<std::size_t>(n) * sizeof(float));
            return t;
        };
        auto declFilter = [&](void* data, int k) -> ggml_tensor* {
            if (data == nullptr) return nullptr;
            ggml_tensor* t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, 1);
            bind(t, data, static_cast<std::size_t>(k) * sizeof(float));
            return t;
        };
        struct Conv { ggml_tensor* w; ggml_tensor* b; int stride, padding, dilation; };
        auto declConv = [&](const TSGH3Conv1d& c) -> Conv {
            if (c.w == nullptr) return { nullptr, nullptr, 1, 0, 1 };
            ggml_tensor* w = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(c.type), c.k, c.ic, c.oc);
            bind(w, c.w, ggml_nbytes(w));
            ggml_tensor* b = nullptr;
            if (c.b != nullptr)
            {
                b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.bias_len);
                bind(b, c.b, static_cast<std::size_t>(c.bias_len) * sizeof(float));
            }
            return { w, b, c.stride, c.padding, c.dilation };
        };
        // x: [T, C, 1]. Bias is broadcast over time.
        auto applyConv = [&](const Conv& c, ggml_tensor* x) {
            ggml_tensor* o = ggml_conv_1d(ctx, c.w, x, c.stride, c.padding, c.dilation);
            if (c.b) o = ggml_add(ctx, o, ggml_reshape_2d(ctx, c.b, 1, c.b->ne[0]));
            return o;
        };

        // SnakeBeta: x + sin(alpha*x)^2 / beta. The checkpoint stores alpha/beta in
        // LOG scale; the caller exponentiates them and folds the divide-by-zero guard
        // into beta, since both are per-channel vectors and doing it on the host keeps
        // the graph free of scalar constants (which a no_alloc context cannot hold).
        auto snake = [&](ggml_tensor* x, ggml_tensor* alpha, ggml_tensor* beta) {
            ggml_tensor* a = ggml_reshape_2d(ctx, alpha, 1, alpha->ne[0]);
            ggml_tensor* b = ggml_reshape_2d(ctx, beta, 1, beta->ne[0]);
            ggml_tensor* s = ggml_sin(ctx, ggml_mul(ctx, x, a));
            s = ggml_mul(ctx, s, s);
            return ggml_add(ctx, x, ggml_div(ctx, s, b));
        };

        // Alias-free activation: 2x up, snake, 2x down. The resample filters are
        // shared across channels, so the channel axis rides along as a batch of
        // 1-channel signals.
        auto activation1d = [&](ggml_tensor* x, const TSGH3Act1d& a) {
            const int ratio = 2, k = a.kernel;
            ggml_tensor* alpha = declVec(a.alpha, a.channels);
            ggml_tensor* beta = declVec(a.beta, a.channels);
            ggml_tensor* upF = declFilter(a.up_filter, k);
            ggml_tensor* downF = declFilter(a.down_filter, k);

            const int upPad = k / ratio - 1;
            const int padLeft = upPad * ratio + (k - ratio) / 2;
            const int padRight = upPad * ratio + (k - ratio + 1) / 2;

            ggml_tensor* h = h3_replicate_pad_1d(ctx, x, upPad, upPad);
            const int channels = static_cast<int>(h->ne[1]);
            const int len = static_cast<int>(h->ne[0]);
            // [T, C, 1] and [T, 1, C] share a memory layout, so treating the channel
            // axis as a batch of 1-channel signals is a reshape, not a transpose.
            ggml_tensor* flat = ggml_reshape_3d(ctx, ggml_cont(ctx, h), len, 1, channels);

            // ggml_conv_transpose_1d only accepts a 2-D input, so it cannot batch over
            // channels. Zero-stuffing and convolving with the REVERSED filter is the
            // same operation and does batch. (The caller pre-reverses the filter.)
            ggml_tensor* col = ggml_reshape_3d(ctx, flat, 1, len, channels);
            ggml_tensor* stuffed = col;
            for (int i = 1; i < ratio; ++i)
                stuffed = ggml_concat(ctx, stuffed, ggml_scale(ctx, col, 0.0f), 0);
            stuffed = ggml_reshape_3d(ctx, ggml_cont(ctx, stuffed), len * ratio, 1, channels);

            ggml_tensor* up = ggml_conv_1d(ctx, upF, stuffed, 1, k - 1, 1);
            const int outTime = (len - 1) * ratio + k;
            if (up->ne[0] > outTime)
                up = ggml_cont(ctx, ggml_view_3d(ctx, up, outTime, 1, channels,
                                                 up->nb[1], up->nb[2], 0));
            up = ggml_scale(ctx, up, static_cast<float>(ratio));
            up = ggml_cont(ctx, ggml_view_3d(ctx, up, outTime - padLeft - padRight, 1, channels,
                                             up->nb[1], up->nb[2],
                                             static_cast<std::size_t>(padLeft) * ggml_element_size(up)));
            ggml_tensor* back = ggml_reshape_3d(ctx, up, up->ne[0], channels, 1);

            back = snake(back, alpha, beta);

            const int downLeft = k / 2 - (k % 2 == 0 ? 1 : 0);
            const int downRight = k / 2;
            ggml_tensor* padded = h3_replicate_pad_1d(ctx, back, downLeft, downRight);
            ggml_tensor* dflat = ggml_reshape_3d(ctx, ggml_cont(ctx, padded),
                                                 padded->ne[0], 1, channels);
            ggml_tensor* down = ggml_conv_1d(ctx, downF, dflat, ratio, 0, 1);
            return ggml_reshape_3d(ctx, ggml_cont(ctx, down), down->ne[0], channels, 1);
        };

        ggml_tensor* z = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                            d->latent_len, d->latent_channels, 1);
        ggml_set_input(z);

        ggml_tensor* h = applyConv(declConv(d->dec_in_proj), z);
        h = applyConv(declConv(d->conv_pre), h);

        const int amps = d->amps_per_stage;
        int convIdx = d->num_stages;   // the ups occupy the first slots
        int actIdx = 0;
        for (int st = 0; st < d->num_stages; st++)
        {
            // Upsample. ggml's transposed conv has no padding, so run it unpadded and
            // trim (k - rate) / 2 from each end, which is what PyTorch's padding does.
            const Conv up = declConv(d->convs[st]);
            const int rate = d->rates[st];
            const int kernel = static_cast<int>(d->convs[st].k);
            const int trim = (kernel - rate) / 2;
            h = ggml_conv_transpose_1d(ctx, up.w, h, rate, 0, 1);
            if (trim > 0)
                h = ggml_cont(ctx, ggml_view_3d(ctx, h, h->ne[0] - 2 * trim, h->ne[1], h->ne[2],
                                                h->nb[1], h->nb[2],
                                                static_cast<std::size_t>(trim) * ggml_element_size(h)));
            if (up.b) h = ggml_add(ctx, h, ggml_reshape_2d(ctx, up.b, 1, up.b->ne[0]));

            // Multi-receptive-field fusion: the amp blocks are AVERAGED, not summed.
            ggml_tensor* acc = nullptr;
            for (int am = 0; am < amps; am++)
            {
                ggml_tensor* x = h;
                for (int dl = 0; dl < 3; dl++)
                {
                    const Conv c1 = declConv(d->convs[convIdx + dl]);
                    const Conv c2 = declConv(d->convs[convIdx + 3 + dl]);
                    ggml_tensor* t = applyConv(c1, activation1d(x, d->acts[actIdx + dl * 2]));
                    t = applyConv(c2, activation1d(t, d->acts[actIdx + dl * 2 + 1]));
                    x = ggml_add(ctx, x, t);
                }
                convIdx += 6;
                actIdx += 6;
                acc = acc == nullptr ? x : ggml_add(ctx, acc, x);
            }
            h = ggml_scale(ctx, acc, 1.0f / amps);
        }

        h = activation1d(h, d->acts[actIdx]);
        h = applyConv(declConv(d->conv_post), h);
        // use_tanh_at_final is false for this checkpoint, so the output is clamped.
        h = ggml_clamp(ctx, h, -1.0f, 1.0f);

        ggml_tensor* outT = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d->samples);
        ggml_tensor* trimmed = ggml_cont(ctx, ggml_view_1d(ctx, h, d->samples, 0));
        ggml_tensor* copied = ggml_cpy(ctx, trimmed, outT);
        ggml_set_output(copied);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 32768, false);
        if (graph == nullptr)
        { set_last_error("MiniMaxH3AudioVaeDecode: graph alloc failed."); return 0; }
        ggml_build_forward_expand(graph, copied);

        for (int i = 0; i < ggml_graph_n_nodes(graph); i++)
        {
            if (!ggml_backend_supports_op(g_backend, ggml_graph_node(graph, i)))
            { set_last_error("MiniMaxH3AudioVaeDecode: op unsupported by backend."); return 0; }
        }

        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            { set_last_error("MiniMaxH3AudioVaeDecode: buffer alloc failed."); return 0; }
        }

        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.data, 0, u.bytes);
        ggml_backend_tensor_set(z, d->latent, 0, ggml_nbytes(z));

        if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        { set_last_error("MiniMaxH3AudioVaeDecode: graph compute failed."); return 0; }
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(outT, d->out, 0, ggml_nbytes(outT));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("MiniMaxH3AudioVaeDecode: unknown error."); return 0; }
}

// ============================================================================
// Qwen3-VL vision tower, for MiniMax-H3 reference conditioning.
//
// 27 blocks at width 1152, then FOUR outputs at width 5120: the final merger plus
// three "DeepStack" mergers tapped after blocks 8/16/24. Output[0] replaces the
// <|image_pad|> embeddings in the language model's input; outputs[1..3] are added
// residually to its hidden states after decoder layers 0, 1 and 2.
//
// Two things differ from the other transformers in this file and are easy to get
// wrong:
//   * the norms are LayerNorm WITH bias, not RMSNorm; and
//   * the final merger normalizes at width 1152 BEFORE the 2x2 spatial merge, while
//     the DeepStack mergers normalize at width 4608 AFTER it. Swapping those is a
//     silent numerical bug, not a shape error.
//
// Position embeddings (a bilinear resample of the learned 48x48 grid) and the 2-D
// RoPE tables are built on the host, so neither appears here.
// ============================================================================

struct TSGH3VisBlockW
{
    void* norm1_w; void* norm1_b;
    void* norm2_w; void* norm2_b;
    TSGH3Lin qkv;            // [dim, 3*dim] + bias
    TSGH3Lin proj;           // [dim, dim] + bias
    TSGH3Lin fc1;            // [dim, ffn] + bias
    TSGH3Lin fc2;            // [ffn, dim] + bias
};

struct TSGH3VisMerger
{
    void* norm_w; void* norm_b;
    TSGH3Lin fc1;            // [merge_dim, merge_dim] + bias
    TSGH3Lin fc2;            // [merge_dim, out_dim] + bias
    // 1 = normalize at `dim` BEFORE the merge (the final merger);
    // 0 = normalize at `merge_dim` AFTER it (the DeepStack mergers).
    std::int32_t norm_before_merge;
    std::int32_t pad_;
};

struct TSGgmlMiniMaxH3VisionEncodeDesc
{
    std::int32_t struct_bytes;
    std::int32_t num_blocks;
    std::int32_t num_deepstack;
    std::int32_t pad_;

    void* patches;           // [patch_dim, tokens] F32, host-patchified
    void* pos_embed;         // [dim, tokens] F32, host-resampled
    void* cosf;              // [head_dim, tokens] F32
    void* sinf;
    void* out;               // [out_dim, merged * (1 + num_deepstack)] F32

    TSGH3Lin patch_embed;    // [patch_dim, dim] + bias
    const TSGH3VisBlockW* blocks;
    const std::int32_t* deepstack_layers;   // block indices to tap
    const TSGH3VisMerger* mergers;          // [0] final, [1..] deepstack

    std::int32_t tokens;
    std::int32_t dim;
    std::int32_t heads;
    std::int32_t head_dim;
    std::int32_t patch_dim;
    std::int32_t merge_size;
    std::int32_t out_dim;
    float eps;
};

TSG_EXPORT int TSGgml_MiniMaxH3VisionEncode(const TSGgmlMiniMaxH3VisionEncodeDesc* d)
{
    try
    {
        if (d == nullptr ||
            d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlMiniMaxH3VisionEncodeDesc)) ||
            d->patches == nullptr || d->pos_embed == nullptr || d->cosf == nullptr ||
            d->sinf == nullptr || d->out == nullptr || d->blocks == nullptr ||
            d->mergers == nullptr || d->num_blocks <= 0 || d->tokens <= 0)
        { set_last_error("MiniMaxH3VisionEncode: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        const int nl = d->num_blocks, nds = d->num_deepstack;
        const int dim = d->dim, heads = d->heads, hd = d->head_dim;
        const int seq = d->tokens, ms = d->merge_size;
        const int mergeDim = dim * ms * ms;
        const int outDim = d->out_dim;
        const float eps = d->eps;
        if (heads <= 0 || hd <= 0 || dim != heads * hd)
        { set_last_error("MiniMaxH3VisionEncode: bad head geometry."); return 0; }
        if (ms <= 0 || seq % (ms * ms) != 0)
        { set_last_error("MiniMaxH3VisionEncode: token count is not a whole number of merge blocks."); return 0; }
        const int merged = seq / (ms * ms);
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        const std::size_t f = sizeof(float);

        PooledContextHandle context;
        if (!context.init(32ull * 1024 * 1024))
        { set_last_error("MiniMaxH3VisionEncode: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* data; std::size_t bytes; };
        std::vector<HostBinding> uploads;
        auto bind = [&](ggml_tensor* t, void* data, std::size_t bytes) {
            if (t == nullptr || data == nullptr) return;
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs) &&
                    ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                {
                    if (needs) uploads.push_back({ t, data, bytes });
                    return;
                }
                invalidate_cached_buffer(data);
            }
            ggml_set_input(t);
            uploads.push_back({ t, data, bytes });
        };
        auto declLin = [&](const TSGH3Lin& s, ggml_tensor*& wt, ggml_tensor*& bt) {
            wt = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(s.type), s.ne0, s.ne1);
            bind(wt, s.w, ggml_nbytes(wt));
            bt = nullptr;
            if (s.b != nullptr)
            {
                bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.ne1);
                bind(bt, s.b, static_cast<std::size_t>(s.ne1) * f);
            }
        };
        auto declVec = [&](void* data, int n) -> ggml_tensor* {
            if (data == nullptr) return nullptr;
            ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            bind(t, data, static_cast<std::size_t>(n) * f);
            return t;
        };
        auto lin = [&](ggml_tensor* w, ggml_tensor* x, ggml_tensor* b) {
            ggml_tensor* o = h3_mm(ctx, w, x, /*guard*/ true);
            return b ? ggml_add(ctx, o, b) : o;
        };
        // LayerNorm with bias, unlike the RMSNorm used elsewhere in this file.
        auto layerNorm = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
            ggml_tensor* n = ggml_norm(ctx, x, eps);
            if (w) n = ggml_mul(ctx, n, w);
            if (b) n = ggml_add(ctx, n, b);
            return n;
        };

        ggml_tensor* patches = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d->patch_dim, seq);
        ggml_tensor* posEmbed = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, seq);
        ggml_tensor* cosf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, seq);
        ggml_tensor* sinf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, seq);
        ggml_set_input(patches); ggml_set_input(posEmbed);
        ggml_set_input(cosf); ggml_set_input(sinf);

        ggml_tensor *pew, *peb;
        declLin(d->patch_embed, pew, peb);
        ggml_tensor* h = ggml_add(ctx, lin(pew, patches, peb), posEmbed);

        // Parity aid: TS_H3_VIS_STOP=N cuts the graph after stage N (0 = the patch
        // embedding, k = after block k-1) and TS_H3_VIS_DUMP names a file to write it
        // to, so a NaN localizes to one stage instead of "the tower is wrong".
        int visStop = -1;
        if (const char* raw = std::getenv("TS_H3_VIS_STOP")) visStop = std::atoi(raw);
        ggml_tensor* visProbe = visStop == 0 ? h : nullptr;

        // One merger: optionally normalize, fold each 2x2 spatial block into one
        // token of width dim*4, then two linears with an erf-GELU between them.
        auto runMerger = [&](const TSGH3VisMerger& m, ggml_tensor* x) {
            ggml_tensor* nw = declVec(m.norm_w, m.norm_before_merge ? dim : mergeDim);
            ggml_tensor* nb = declVec(m.norm_b, m.norm_before_merge ? dim : mergeDim);
            ggml_tensor *w1, *b1, *w2, *b2;
            declLin(m.fc1, w1, b1);
            declLin(m.fc2, w2, b2);

            ggml_tensor* t = x;
            if (m.norm_before_merge) t = layerNorm(t, nw, nb);
            // Tokens already arrive in merge-block order, so folding is a reshape.
            t = ggml_reshape_2d(ctx, ggml_cont(ctx, t), mergeDim, merged);
            if (!m.norm_before_merge) t = layerNorm(t, nw, nb);
            t = lin(w1, t, b1);
            t = ggml_gelu_erf(ctx, t);
            return lin(w2, t, b2);
        };

        std::vector<ggml_tensor*> outputs;
        outputs.reserve(1 + nds);
        outputs.push_back(nullptr);   // filled with the final merger below

        for (int l = 0; l < nl && visProbe == nullptr; l++)
        {
            const TSGH3VisBlockW& bw = d->blocks[l];
            ggml_tensor* n1w = declVec(bw.norm1_w, dim);
            ggml_tensor* n1b = declVec(bw.norm1_b, dim);
            ggml_tensor* n2w = declVec(bw.norm2_w, dim);
            ggml_tensor* n2b = declVec(bw.norm2_b, dim);
            ggml_tensor *qkvw, *qkvb, *pw, *pb, *w1, *b1, *w2, *b2;
            declLin(bw.qkv, qkvw, qkvb);
            declLin(bw.proj, pw, pb);
            declLin(bw.fc1, w1, b1);
            declLin(bw.fc2, w2, b2);

            ggml_tensor* n1 = layerNorm(h, n1w, n1b);
            ggml_tensor* qkv = lin(qkvw, n1, qkvb);
            ggml_tensor* q = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
                static_cast<std::size_t>(hd) * f, qkv->nb[1], 0));
            ggml_tensor* k = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
                static_cast<std::size_t>(hd) * f, qkv->nb[1], static_cast<std::size_t>(dim) * f));
            ggml_tensor* v = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
                static_cast<std::size_t>(hd) * f, qkv->nb[1], static_cast<std::size_t>(2) * dim * f));
            q = h3_vit_rope(ctx, q, cosf, sinf, hd, heads, seq, hd);
            k = h3_vit_rope(ctx, k, cosf, sinf, hd, heads, seq, hd);
            // Qwen3-VL runs FULL attention on every layer; the windowed variant in the
            // Qwen2.5-VL config is dead for this path.
            ggml_tensor* merged2 = h3_attend(ctx, g_backend, q, k, v, hd, heads, seq,
                                             scale, /*allowFlash*/ true);
            h = ggml_add(ctx, h, lin(pw, merged2, pb));

            ggml_tensor* n2 = layerNorm(h, n2w, n2b);
            // The block MLP uses the tanh-approximated GELU; the mergers use erf.
            h = ggml_add(ctx, h, lin(w2, ggml_gelu(ctx, lin(w1, n2, b1)), b2));

            for (int t = 0; t < nds; t++)
                if (d->deepstack_layers[t] == l)
                    outputs.push_back(runMerger(d->mergers[1 + t], h));
            if (visStop == l + 1) visProbe = h;
        }

        if (visProbe != nullptr)
        {
            ggml_tensor* pOut = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                                   visProbe->ne[0], visProbe->ne[1]);
            ggml_tensor* pCopy = ggml_cpy(ctx, visProbe, pOut);
            ggml_set_output(pCopy);
            ggml_cgraph* pg = ggml_new_graph_custom(ctx, 16384, false);
            ggml_build_forward_expand(pg, pCopy);
            BufferHandle pbuf(nullptr);
            if (!alloc_graph_reuse_gallocr(pg))
            {
                pbuf.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
                if (pbuf.value == nullptr)
                { set_last_error("MiniMaxH3VisionEncode: probe buffer alloc failed."); return 0; }
            }
            host_read_barrier();
            // A truncated graph does not contain every declared tensor, and an
            // unallocated one has no buffer to upload into.
            auto put = [&](ggml_tensor* t, void* src) {
                if (t != nullptr && t->buffer != nullptr && src != nullptr)
                    ggml_backend_tensor_set(t, src, 0, ggml_nbytes(t));
            };
            for (auto& u : uploads) put(u.t, u.data);
            put(patches, d->patches);
            put(posEmbed, d->pos_embed);
            put(cosf, d->cosf);
            put(sinf, d->sinf);
            if (tsg::compute_graph(g_backend, pg) != GGML_STATUS_SUCCESS)
            { set_last_error("MiniMaxH3VisionEncode: probe compute failed."); return 0; }
            tsg::sync_backend(g_backend);
            std::vector<float> tmp(static_cast<std::size_t>(ggml_nelements(pOut)));
            ggml_backend_tensor_get(pOut, tmp.data(), 0, ggml_nbytes(pOut));
            std::size_t nans = 0; double acc = 0;
            for (float v : tmp) { if (std::isnan(v) || std::isinf(v)) nans++; else acc += (double)v * v; }
            char line[256];
            std::snprintf(line, sizeof(line),
                          "[h3-vis-probe] stage %d: %lld x %lld, nan/inf=%zu of %zu rms=%.6f\n",
                          visStop, (long long)pOut->ne[0], (long long)pOut->ne[1], nans,
                          tmp.size(), std::sqrt(acc / std::max<std::size_t>(1, tmp.size() - nans)));
            std::fputs(line, stderr);
            if (const char* path = std::getenv("TS_H3_VIS_DUMP"))
                if (FILE* fp = std::fopen(path, "a")) { std::fputs(line, fp); std::fclose(fp); }
            clear_last_error();
            return 1;
        }

        outputs[0] = runMerger(d->mergers[0], h);
        if (static_cast<int>(outputs.size()) != 1 + nds)
        { set_last_error("MiniMaxH3VisionEncode: deepstack tap layers did not all fire."); return 0; }

        ggml_tensor* outT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, outDim,
                                               static_cast<std::int64_t>(merged) * (1 + nds));
        ggml_tensor* stacked = outputs[0];
        for (std::size_t i = 1; i < outputs.size(); ++i)
            stacked = ggml_concat(ctx, stacked, outputs[i], 1);
        ggml_tensor* copied = ggml_cpy(ctx, stacked, outT);
        ggml_set_output(copied);

        const std::size_t nodes = static_cast<std::size_t>(nl) * 104 + 4096;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
        if (graph == nullptr)
        { set_last_error("MiniMaxH3VisionEncode: graph alloc failed."); return 0; }
        ggml_build_forward_expand(graph, copied);

        for (int i = 0; i < ggml_graph_n_nodes(graph); i++)
        {
            if (!ggml_backend_supports_op(g_backend, ggml_graph_node(graph, i)))
            { set_last_error("MiniMaxH3VisionEncode: op unsupported by backend."); return 0; }
        }

        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            { set_last_error("MiniMaxH3VisionEncode: buffer alloc failed."); return 0; }
        }

        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.data, 0, u.bytes);
        ggml_backend_tensor_set(patches, d->patches, 0, ggml_nbytes(patches));
        ggml_backend_tensor_set(posEmbed, d->pos_embed, 0, ggml_nbytes(posEmbed));
        ggml_backend_tensor_set(cosf, d->cosf, 0, ggml_nbytes(cosf));
        ggml_backend_tensor_set(sinf, d->sinf, 0, ggml_nbytes(sinf));

        if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        { set_last_error("MiniMaxH3VisionEncode: graph compute failed."); return 0; }
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(outT, d->out, 0, ggml_nbytes(outT));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("MiniMaxH3VisionEncode: unknown error."); return 0; }
}

// ============================================================================
// MiniMax-H3 audio VAE ENCODER: mono 32 kHz PCM -> 32 latent channels at 40 Hz.
//
// This is a DAC encoder, and it is NOT the mirror image of the BigVGAN decoder
// that sits beside it. The decoder upsamples in seven stages {5,5,2,2,2,2,2};
// the encoder downsamples in five, {2,4,4,5,5}. Only the product, 800, matches --
// which is what makes 32000 / 800 = the 40 Hz latent rate.
//
// The activations differ too. The decoder's SnakeBeta has separate alpha/beta
// stored in LOG scale; the encoder's Snake1d has alpha only, stored LINEARLY:
//     x + sin(alpha*x)^2 / (alpha + 1e-9)
// Exponentiating the encoder's alpha would be a silent, plausible-looking error,
// so the two act structs are deliberately kept distinct rather than shared.
//
// The tail is unusual for a conv autoencoder: a single causal-attention block
// that projects 2048 -> 32 by averaging over attention heads and then adaptively
// average-pooling 256 -> 32, rather than by a linear layer.
// ============================================================================

// alpha is LINEAR here (unlike the decoder's log-scale alpha/beta). The caller
// also supplies alpha + 1e-9 so the divide guard needs no scalar in the graph.
struct TSGH3Snake1d
{
    void* alpha;        // [channels] F32
    void* alpha_eps;    // [channels] F32, = alpha + 1e-9
    std::int32_t channels;
    std::int32_t pad_;
};

struct TSGH3AudioEncResUnit
{
    TSGH3Snake1d act1;
    TSGH3Conv1d conv1;      // k7, dilated
    TSGH3Snake1d act2;
    TSGH3Conv1d conv2;      // k1
};

struct TSGH3AudioEncBlock
{
    TSGH3AudioEncResUnit units[3];   // dilations 1, 3, 9
    TSGH3Snake1d act;
    TSGH3Conv1d down;                // k = 2*stride, stride, pad = ceil(stride/2)
};

// The 2048 -> 32 causal-attention projection. `qkv` has no bias in the
// checkpoint; the caller builds one by concatenating q_bias, ZEROS and v_bias,
// which is how the reference fakes "no key bias".
struct TSGH3AudioAttnProj
{
    void* norm1_w; void* norm1_b;      // [trunk]
    void* norm3_w; void* norm3_b;      // [trunk]
    void* norm2_w; void* norm2_b;      // [latent]
    void* mlp_norm_w; void* mlp_norm_b;// [latent]
    TSGH3Lin qkv;                      // [trunk, 3*trunk]
    void* qkv_bias;                    // [3*trunk] F32
    TSGH3Lin attn_proj;                // [latent, latent]
    TSGH3Lin proj;                     // [trunk, latent]
    TSGH3Lin w0; TSGH3Lin w1; TSGH3Lin w2;
    std::int32_t heads;
    std::int32_t pad_;
};

struct TSGgmlMiniMaxH3AudioVaeEncodeDesc
{
    std::int32_t struct_bytes;
    std::int32_t num_blocks;

    void* wave;              // [samples] F32, one mono plane
    void* out;               // [frames, latent_channels] F32 written

    TSGH3Conv1d conv_in;     // 1 -> 64, k7
    const TSGH3AudioEncBlock* blocks;
    TSGH3Snake1d final_act;
    TSGH3Conv1d final_conv;  // trunk -> trunk, k3
    TSGH3AudioAttnProj pre;
    TSGH3Conv1d mean_proj;   // latent -> latent, k1

    std::int32_t samples;
    std::int32_t frames;
    std::int32_t latent_channels;
    std::int32_t trunk_channels;
    float eps;
    std::int32_t pad_;
};

TSG_EXPORT int TSGgml_MiniMaxH3AudioVaeEncode(const TSGgmlMiniMaxH3AudioVaeEncodeDesc* d)
{
    try
    {
        if (d == nullptr ||
            d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlMiniMaxH3AudioVaeEncodeDesc)) ||
            d->wave == nullptr || d->out == nullptr || d->blocks == nullptr ||
            d->num_blocks <= 0 || d->samples <= 0 || d->frames <= 0)
        { set_last_error("MiniMaxH3AudioVaeEncode: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        const int nb = d->num_blocks;
        const int lat = d->latent_channels;
        const int trunk = d->trunk_channels;
        const int heads = d->pre.heads;
        const int seq = d->frames;
        const float eps = d->eps;
        const std::size_t f = sizeof(float);
        if (heads <= 0 || trunk % heads != 0)
        { set_last_error("MiniMaxH3AudioVaeEncode: bad head geometry."); return 0; }
        const int hd = trunk / heads;
        if (hd % lat != 0)
        { set_last_error("MiniMaxH3AudioVaeEncode: head dim is not a whole multiple of the latent width."); return 0; }
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

        PooledContextHandle context;
        if (!context.init(32ull * 1024 * 1024))
        { set_last_error("MiniMaxH3AudioVaeEncode: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* data; std::size_t bytes; };
        std::vector<HostBinding> uploads;
        auto bind = [&](ggml_tensor* t, void* data, std::size_t bytes) {
            if (t == nullptr || data == nullptr) return;
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs) &&
                    ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                {
                    if (needs) uploads.push_back({ t, data, bytes });
                    return;
                }
                invalidate_cached_buffer(data);
            }
            ggml_set_input(t);
            uploads.push_back({ t, data, bytes });
        };
        auto declVec = [&](void* data, int n) -> ggml_tensor* {
            if (data == nullptr) return nullptr;
            ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            bind(t, data, static_cast<std::size_t>(n) * f);
            return t;
        };
        struct Conv { ggml_tensor* w; ggml_tensor* b; int stride, padding, dilation; };
        auto declConv = [&](const TSGH3Conv1d& c) -> Conv {
            if (c.w == nullptr) return { nullptr, nullptr, 1, 0, 1 };
            ggml_tensor* w = ggml_new_tensor_3d(ctx, static_cast<ggml_type>(c.type), c.k, c.ic, c.oc);
            bind(w, c.w, ggml_nbytes(w));
            ggml_tensor* b = nullptr;
            if (c.b != nullptr)
            {
                b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.bias_len);
                bind(b, c.b, static_cast<std::size_t>(c.bias_len) * f);
            }
            return { w, b, c.stride, c.padding, c.dilation };
        };
        // x: [T, C, 1]; bias broadcasts over time.
        auto applyConv = [&](const Conv& c, ggml_tensor* x) {
            ggml_tensor* o = ggml_conv_1d(ctx, c.w, x, c.stride, c.padding, c.dilation);
            if (c.b) o = ggml_add(ctx, o, ggml_reshape_2d(ctx, c.b, 1, c.b->ne[0]));
            return o;
        };
        struct Snake { ggml_tensor* alpha; ggml_tensor* alphaEps; };
        auto declSnake = [&](const TSGH3Snake1d& s) -> Snake {
            return { declVec(s.alpha, s.channels), declVec(s.alpha_eps, s.channels) };
        };
        // Snake1d: x + sin(alpha*x)^2 / (alpha + 1e-9). Note the numerator uses
        // alpha and the denominator alpha+eps, which is why both are supplied.
        auto snake = [&](const Snake& s, ggml_tensor* x) {
            ggml_tensor* a = ggml_reshape_2d(ctx, s.alpha, 1, s.alpha->ne[0]);
            ggml_tensor* ae = ggml_reshape_2d(ctx, s.alphaEps, 1, s.alphaEps->ne[0]);
            ggml_tensor* o = ggml_sin(ctx, ggml_mul(ctx, x, a));
            o = ggml_mul(ctx, o, o);
            return ggml_add(ctx, x, ggml_div(ctx, o, ae));
        };
        auto declLin = [&](const TSGH3Lin& s, ggml_tensor*& wt, ggml_tensor*& bt) {
            wt = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(s.type), s.ne0, s.ne1);
            bind(wt, s.w, ggml_nbytes(wt));
            bt = nullptr;
            if (s.b != nullptr)
            {
                bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.ne1);
                bind(bt, s.b, static_cast<std::size_t>(s.ne1) * f);
            }
        };
        auto lin = [&](ggml_tensor* w, ggml_tensor* x, ggml_tensor* b) {
            ggml_tensor* o = h3_mm(ctx, w, x, /*guard*/ true);
            return b ? ggml_add(ctx, o, b) : o;
        };
        auto layerNorm = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
            ggml_tensor* n = ggml_norm(ctx, x, eps);
            if (w) n = ggml_mul(ctx, n, w);
            if (b) n = ggml_add(ctx, n, b);
            return n;
        };

        ggml_tensor* wave = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d->samples, 1, 1);
        ggml_set_input(wave);

        // ---- conv trunk, in [T, C] layout ----
        ggml_tensor* h = applyConv(declConv(d->conv_in), wave);
        for (int i = 0; i < nb; i++)
        {
            const TSGH3AudioEncBlock& b = d->blocks[i];
            for (int u = 0; u < 3; u++)
            {
                const TSGH3AudioEncResUnit& ru = b.units[u];
                ggml_tensor* y = snake(declSnake(ru.act1), h);
                y = applyConv(declConv(ru.conv1), y);
                y = snake(declSnake(ru.act2), y);
                y = applyConv(declConv(ru.conv2), y);
                h = ggml_add(ctx, h, y);
            }
            h = snake(declSnake(b.act), h);
            h = applyConv(declConv(b.down), h);
        }
        h = snake(declSnake(d->final_act), h);
        h = applyConv(declConv(d->final_conv), h);        // [T, trunk]

        if (static_cast<int>(h->ne[0]) != seq)
        { set_last_error("MiniMaxH3AudioVaeEncode: trunk produced an unexpected frame count."); return 0; }

        // ---- causal attention projection, in [C, T] layout ----
        ggml_tensor* x = ggml_cont(ctx, ggml_transpose(ctx, h));   // [trunk, T]

        ggml_tensor* n1w = declVec(d->pre.norm1_w, trunk);
        ggml_tensor* n1b = declVec(d->pre.norm1_b, trunk);
        ggml_tensor* n3w = declVec(d->pre.norm3_w, trunk);
        ggml_tensor* n3b = declVec(d->pre.norm3_b, trunk);
        ggml_tensor* n2w = declVec(d->pre.norm2_w, lat);
        ggml_tensor* n2b = declVec(d->pre.norm2_b, lat);
        ggml_tensor* mnw = declVec(d->pre.mlp_norm_w, lat);
        ggml_tensor* mnb = declVec(d->pre.mlp_norm_b, lat);
        ggml_tensor* qkvBias = declVec(d->pre.qkv_bias, 3 * trunk);
        ggml_tensor *qkvW, *unusedB, *apW, *apB, *pW, *pB, *w0, *b0, *w1, *b1, *w2, *b2;
        declLin(d->pre.qkv, qkvW, unusedB);
        declLin(d->pre.attn_proj, apW, apB);
        declLin(d->pre.proj, pW, pB);
        declLin(d->pre.w0, w0, b0);
        declLin(d->pre.w1, w1, b1);
        declLin(d->pre.w2, w2, b2);

        // An explicit additive mask rather than ggml_diag_mask_inf: Metal has no
        // diag-mask kernel, so the whole-graph supports_op sweep would reject it.
        ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, seq);
        ggml_set_input(mask);
        std::vector<float> maskData(static_cast<std::size_t>(seq) * seq, 0.0f);
        const float negInf = -std::numeric_limits<float>::infinity();
        for (int q = 0; q < seq; q++)
            for (int k = q + 1; k < seq; k++)
                maskData[static_cast<std::size_t>(q) * seq + k] = negInf;

        ggml_tensor* n1 = layerNorm(x, n1w, n1b);
        ggml_tensor* qkv = ggml_add(ctx, h3_mm(ctx, qkvW, n1, /*guard*/ true), qkvBias);
        ggml_tensor* q = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
            static_cast<std::size_t>(hd) * f, qkv->nb[1], 0));
        ggml_tensor* k = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
            static_cast<std::size_t>(hd) * f, qkv->nb[1], static_cast<std::size_t>(trunk) * f));
        ggml_tensor* v = ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
            static_cast<std::size_t>(hd) * f, qkv->nb[1], static_cast<std::size_t>(2) * trunk * f));

        ggml_tensor* qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor* kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor* kq = ggml_mul_mat(ctx, kp, qp);
        ggml_tensor* probs = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);
        ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor* kqv = ggml_mul_mat(ctx, vt, probs);           // [hd, T, heads]

        // The projection to the latent width is a pair of averages, not a linear:
        // first over the attention heads, then adaptively over the head dimension.
        ggml_tensor* byHead = ggml_cont(ctx, ggml_permute(ctx, kqv, 1, 2, 0, 3));  // [heads, hd, T]
        ggml_tensor* pooled = ggml_reshape_2d(ctx, ggml_mean(ctx, byHead), hd, seq);
        pooled = ggml_reshape_3d(ctx, pooled, hd / lat, lat, seq);
        pooled = ggml_reshape_2d(ctx, ggml_mean(ctx, pooled), lat, seq);
        ggml_tensor* attn = lin(apW, pooled, apB);                  // [lat, T]

        ggml_tensor* n3 = layerNorm(x, n3w, n3b);
        x = ggml_add(ctx, lin(pW, n3, pB), attn);                   // [lat, T]

        // GeGLU, and note the second normalization: the block norms with norm2 and
        // the MLP norms again with its own, which is what the reference does.
        ggml_tensor* mh = layerNorm(layerNorm(x, n2w, n2b), mnw, mnb);
        ggml_tensor* gate = ggml_gelu(ctx, lin(w0, mh, b0));
        mh = lin(w2, ggml_mul(ctx, gate, lin(w1, mh, b1)), b2);
        x = ggml_add(ctx, x, mh);                                   // [lat, T]

        // Back to [T, lat] for the 1-wide output convolution.
        ggml_tensor* z = applyConv(declConv(d->mean_proj), ggml_cont(ctx, ggml_transpose(ctx, x)));

        ggml_tensor* outT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq, lat);
        ggml_tensor* copied = ggml_cpy(ctx, z, outT);
        ggml_set_output(copied);

        const std::size_t nodes = static_cast<std::size_t>(nb) * 256 + 4096;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
        if (graph == nullptr)
        { set_last_error("MiniMaxH3AudioVaeEncode: graph alloc failed."); return 0; }
        ggml_build_forward_expand(graph, copied);

        for (int i = 0; i < ggml_graph_n_nodes(graph); i++)
        {
            if (!ggml_backend_supports_op(g_backend, ggml_graph_node(graph, i)))
            { set_last_error("MiniMaxH3AudioVaeEncode: op unsupported by backend."); return 0; }
        }

        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            { set_last_error("MiniMaxH3AudioVaeEncode: buffer alloc failed."); return 0; }
        }

        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.data, 0, u.bytes);
        ggml_backend_tensor_set(wave, d->wave, 0, ggml_nbytes(wave));
        ggml_backend_tensor_set(mask, maskData.data(), 0, ggml_nbytes(mask));

        if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        { set_last_error("MiniMaxH3AudioVaeEncode: graph compute failed."); return 0; }
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(outT, d->out, 0, ggml_nbytes(outT));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("MiniMaxH3AudioVaeEncode: unknown error."); return 0; }
}

// ============================================================================
// MiniMax-H3 video VAE ENCODER, multi-frame (the real causal 3-D network).
//
// The single-frame op above is an exact 2-D reduction of this one and stays the
// fast path for keyframes and image references. A reference CLIP needs the real
// thing: 17 pixel frames collapse to 5 latent frames, and that only happens if
// the temporal convolutions actually run.
//
// Two details carry the whole design:
//
//   * CAUSALITY. A nominal temporal padding of 1 becomes TWO zero frames at the
//     FRONT and none at the back, so every output frame sees only its own past.
//     Zeros, not edge replication -- which is exactly why the single-frame case
//     reduces to the kernel's last temporal slice.
//   * NORMALIZATION IS PER FRAME. The group norm treats time as BATCH, so frame t
//     is normalized by its own statistics alone. Folding time into the group
//     reduction instead would still produce plausible latents, silently coupling
//     frames that the model expects to be independent.
//
// ggml wants the channel axis last for a 3-D convolution and third for a group
// norm, so the graph transposes between the two layouts rather than picking one
// and paying for it with a wrong reduction.
// ============================================================================

// A 3-D convolution kernel, supplied as KD CONTIGUOUS 2-D kernels: slice j is a
// [KW, KH, IC, OC] block at offset j*KW*KH*IC*OC.
//
// ggml has two 3-D convolutions and neither is usable here: ggml_conv_3d lowers
// to IM2COL_3D, which Metal does not implement, and ggml_conv_3d_direct is a
// naive kernel that measured about 5x slower than the equivalent 2-D work. A
// k=3 temporal convolution is just three 2-D convolutions of shifted frames
// summed, and ggml_conv_2d takes the well-optimized im2col + GEMM path, so the
// decomposition is both portable and fast.
struct TSGH3Conv3d
{
    void* w;                 // KD contiguous [KW, KH, IC, OC] slices
    void* b;                 // nullable, F32 [OC]
    std::int64_t kw, kh, kd, ic, oc;
    std::int32_t type;
    std::int32_t pad_;
};

struct TSGH3EncResBlock3D
{
    void* norm1_w; void* norm1_b;
    void* norm2_w; void* norm2_b;
    TSGH3Conv3d conv1, conv2;
    TSGH3Conv3d shortcut;            // .w null when in_ch == out_ch
};

struct TSGH3EncLevel3D
{
    TSGH3EncResBlock3D block0, block1;
    TSGH3Conv3d downsample;          // .w null when the level does not downsample
    std::int32_t space_stride;
    std::int32_t time_stride;
};

struct TSGgmlMiniMaxH3VideoVaeEncode3DDesc
{
    std::int32_t struct_bytes;
    std::int32_t num_levels;

    void* video;             // [W, H, T, 3] F32, ImageNet-normalized
    void* out;               // [W/16, H/16, Tl, latent_channels] F32

    TSGH3Conv3d conv_in;
    const TSGH3EncLevel3D* levels;
    void* norm_out_w; void* norm_out_b;
    TSGH3Conv3d conv_out;
    TSGH3Conv3d quant_conv;

    std::int32_t width, height, frames;
    std::int32_t latent_frames;
    std::int32_t latent_channels;
    std::int32_t groups;
    float eps;
};

namespace {

// Reflect-pad the W and H axes of a [W, H, D, C] tensor, mirroring about the edge
// without repeating it (PyTorch's 'reflect').
ggml_tensor* h3_reflect_pad_4d(ggml_context* ctx, ggml_tensor* x,
                               int left, int right, int top, int bottom)
{
    const std::size_t f = ggml_element_size(x);
    auto column = [&](std::int64_t index) {
        return ggml_cont(ctx, ggml_view_4d(ctx, x, 1, x->ne[1], x->ne[2], x->ne[3],
                                           x->nb[1], x->nb[2], x->nb[3],
                                           static_cast<std::size_t>(index) * f));
    };
    if (left > 0 || right > 0)
    {
        std::vector<ggml_tensor*> parts;
        for (int i = left; i >= 1; --i) parts.push_back(column(i));
        parts.push_back(x);
        for (int i = 1; i <= right; ++i) parts.push_back(column(x->ne[0] - 1 - i));
        ggml_tensor* acc = parts[0];
        for (std::size_t i = 1; i < parts.size(); ++i) acc = ggml_concat(ctx, acc, parts[i], 0);
        x = acc;
    }
    auto row = [&](std::int64_t index) {
        return ggml_cont(ctx, ggml_view_4d(ctx, x, x->ne[0], 1, x->ne[2], x->ne[3],
                                           x->nb[1], x->nb[2], x->nb[3],
                                           static_cast<std::size_t>(index) * x->nb[1]));
    };
    if (top > 0 || bottom > 0)
    {
        std::vector<ggml_tensor*> parts;
        for (int i = top; i >= 1; --i) parts.push_back(row(i));
        parts.push_back(x);
        for (int i = 1; i <= bottom; ++i) parts.push_back(row(x->ne[1] - 1 - i));
        ggml_tensor* acc = parts[0];
        for (std::size_t i = 1; i < parts.size(); ++i) acc = ggml_concat(ctx, acc, parts[i], 1);
        x = acc;
    }
    return x;
}

// Prepend `count` ZERO frames to a [W, H, C, T] tensor, where T is the BATCH
// axis. The zeros are made by scaling an existing frame rather than bound from
// the host, so no buffer has to be sized and uploaded per level.
ggml_tensor* h3_causal_pad_front(ggml_context* ctx, ggml_tensor* x, int count)
{
    if (count <= 0) return x;
    ggml_tensor* one = ggml_cont(ctx, ggml_view_4d(ctx, x, x->ne[0], x->ne[1], x->ne[2], 1,
                                                   x->nb[1], x->nb[2], x->nb[3], 0));
    ggml_tensor* zero = ggml_scale(ctx, one, 0.0f);
    ggml_tensor* pad = zero;
    for (int i = 1; i < count; i++) pad = ggml_concat(ctx, pad, zero, 3);
    return ggml_concat(ctx, pad, x, 3);
}

} // namespace

TSG_EXPORT int TSGgml_MiniMaxH3VideoVaeEncode3D(const TSGgmlMiniMaxH3VideoVaeEncode3DDesc* d)
{
    try
    {
        if (d == nullptr ||
            d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlMiniMaxH3VideoVaeEncode3DDesc)) ||
            d->video == nullptr || d->out == nullptr || d->levels == nullptr ||
            d->num_levels <= 0 || d->width <= 0 || d->height <= 0 || d->frames <= 0)
        { set_last_error("MiniMaxH3VideoVaeEncode3D: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        PooledContextHandle context;
        if (!context.init(32ull * 1024 * 1024))
        { set_last_error("MiniMaxH3VideoVaeEncode3D: ctx alloc failed."); return 0; }
        ggml_context* ctx = context.value;

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        struct HostBinding { ggml_tensor* t; void* data; std::size_t bytes; };
        std::vector<HostBinding> uploads;
        auto bind = [&](ggml_tensor* t, void* data, std::size_t bytes) {
            if (t == nullptr || data == nullptr) return;
            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
                if (try_get_cacheable_tensor_buffer(g_backend, dev, t, data, bytes, buf, addr, needs) &&
                    ggml_backend_tensor_alloc(buf, t, addr) == GGML_STATUS_SUCCESS)
                {
                    if (needs) uploads.push_back({ t, data, bytes });
                    return;
                }
                invalidate_cached_buffer(data);
            }
            ggml_set_input(t);
            uploads.push_back({ t, data, bytes });
        };
        auto declVec = [&](void* data, int n) -> ggml_tensor* {
            if (data == nullptr) return nullptr;
            ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            bind(t, data, static_cast<std::size_t>(n) * sizeof(float));
            return t;
        };
        // One 3-D kernel, held as its KD separate 2-D slices.
        struct Conv { std::vector<ggml_tensor*> slices; ggml_tensor* b; std::int64_t ic, oc, kd; };
        auto declConv = [&](const TSGH3Conv3d& c) -> Conv {
            Conv out{};
            out.b = nullptr; out.ic = c.ic; out.oc = c.oc; out.kd = c.kd;
            if (c.w == nullptr) return out;
            const ggml_type type = static_cast<ggml_type>(c.type);
            const std::size_t slice = static_cast<std::size_t>(c.kw) * c.kh * c.ic * c.oc
                                      * ggml_type_size(type) / ggml_blck_size(type);
            for (std::int64_t j = 0; j < c.kd; j++)
            {
                ggml_tensor* w = ggml_new_tensor_4d(ctx, type, c.kw, c.kh, c.ic, c.oc);
                bind(w, static_cast<char*>(c.w) + static_cast<std::size_t>(j) * slice,
                     ggml_nbytes(w));
                out.slices.push_back(w);
            }
            if (c.b != nullptr)
            {
                out.b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.oc);
                bind(out.b, c.b, static_cast<std::size_t>(c.oc) * sizeof(float));
            }
            return out;
        };

        // A temporal convolution of a [W, H, C, T] tensor, done as KD batched 2-D
        // convolutions of shifted frames. With the causal zeros already in front,
        // output frame o reads input frame o*timeStride + j, so slice j convolves a
        // stride-timeStride window starting at j and the slices are summed.
        auto applyConv = [&](const Conv& c, ggml_tensor* x, int sw, int sh, int timeStride) {
            const std::int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], T = x->ne[3];
            const std::int64_t outT = (T - c.kd) / timeStride + 1;
            ggml_tensor* acc = nullptr;
            for (std::int64_t j = 0; j < c.kd; j++)
            {
                ggml_tensor* window = ggml_view_4d(ctx, x, W, H, C, outT,
                                                   x->nb[1], x->nb[2],
                                                   static_cast<std::size_t>(timeStride) * x->nb[3],
                                                   static_cast<std::size_t>(j) * x->nb[3]);
                if (timeStride != 1) window = ggml_cont(ctx, window);
                ggml_tensor* o = ggml_conv_2d(ctx, c.slices[j], window, sw, sh, 0, 0, 1, 1);
                acc = acc == nullptr ? o : ggml_add(ctx, acc, o);
            }
            if (c.b) acc = ggml_add(ctx, acc, ggml_reshape_4d(ctx, c.b, 1, 1, c.b->ne[0], 1));
            return acc;
        };
        // A 1x1x1 convolution: no padding and no temporal reach, so one slice.
        auto applyPointwise = [&](const Conv& c, ggml_tensor* x) {
            ggml_tensor* o = ggml_conv_2d(ctx, c.slices[0], x, 1, 1, 0, 0, 1, 1);
            if (c.b) o = ggml_add(ctx, o, ggml_reshape_4d(ctx, c.b, 1, 1, c.b->ne[0], 1));
            return o;
        };
        // Per-frame group norm, expressed as ONE group norm over a folded axis.
        //
        // ggml's group norm nominally treats ne3 as a batch, which would give the
        // per-frame reduction directly -- but the Metal kernel reduces over
        // ne0*ne1*ne2 only and dispatches one threadgroup per group, so everything
        // past the first batch entry is left unnormalized. That is invisible for a
        // single frame and silently wrong for a clip.
        //
        // Folding time into the channel axis avoids the batch entirely: with the
        // tensor laid out [W, H, C, T] the flattened channel index is t*C + c, and
        // asking for groups*T groups makes every group exactly one frame's worth of
        // one group's channels. Same arithmetic, one node, no batching.
        auto groupNormSilu = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b) {
            const std::int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], T = x->ne[3];
            ggml_tensor* folded = ggml_reshape_4d(ctx, ggml_cont(ctx, x), W, H, C * T, 1);
            ggml_tensor* n = ggml_group_norm(ctx, folded,
                                             static_cast<int>(d->groups * T), d->eps);
            n = ggml_reshape_4d(ctx, n, W, H, C, T);
            if (w) n = ggml_mul(ctx, n, ggml_reshape_4d(ctx, w, 1, 1, w->ne[0], 1));
            if (b) n = ggml_add(ctx, n, ggml_reshape_4d(ctx, b, 1, 1, b->ne[0], 1));
            return ggml_silu(ctx, n);
        };
        // A CausalConv3d: reflect in space, KD-1 zero frames at the front in time.
        auto causalConv = [&](const Conv& c, ggml_tensor* x, int spaceStride, int timeStride,
                              bool spatialPad) {
            ggml_tensor* t = x;
            if (spatialPad) t = h3_reflect_pad_4d(ctx, t, 1, 1, 1, 1);
            else if (spaceStride == 2) t = h3_reflect_pad_4d(ctx, t, 0, 1, 0, 1);
            t = h3_causal_pad_front(ctx, t, static_cast<int>(c.kd - 1));
            return applyConv(c, t, spaceStride, spaceStride, timeStride);
        };

        ggml_tensor* video = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                d->width, d->height, 3, d->frames);
        ggml_set_input(video);

        ggml_tensor* h = causalConv(declConv(d->conv_in), video, 1, 1, /*spatialPad*/ true);

        // Parity aid: TS_H3_ENC3D_STOP=N cuts the graph off after stage N so a
        // mismatch localizes to one stage instead of "the encoder is wrong".
        // 0 = right after conv_in, k = after level k-1.
        int stopAfter = -1;
        if (const char* raw = getenv("TS_H3_ENC3D_STOP")) stopAfter = std::atoi(raw);
        ggml_tensor* probe = stopAfter == 0 ? h : nullptr;

        for (int l = 0; l < d->num_levels && probe == nullptr; l++)
        {
            const TSGH3EncLevel3D& lv = d->levels[l];
            for (int bi = 0; bi < 2; bi++)
            {
                const TSGH3EncResBlock3D& rb = bi == 0 ? lv.block0 : lv.block1;
                ggml_tensor* n1w = declVec(rb.norm1_w, static_cast<int>(rb.conv1.ic));
                ggml_tensor* n1b = declVec(rb.norm1_b, static_cast<int>(rb.conv1.ic));
                ggml_tensor* n2w = declVec(rb.norm2_w, static_cast<int>(rb.conv2.ic));
                ggml_tensor* n2b = declVec(rb.norm2_b, static_cast<int>(rb.conv2.ic));
                Conv c1 = declConv(rb.conv1), c2 = declConv(rb.conv2);
                Conv sc = declConv(rb.shortcut);

                ggml_tensor* t = groupNormSilu(h, n1w, n1b);
                t = causalConv(c1, t, 1, 1, true);
                t = groupNormSilu(t, n2w, n2b);
                t = causalConv(c2, t, 1, 1, true);
                // The shortcut is a 1x1x1 convolution: no padding of any kind.
                ggml_tensor* skip = sc.slices.empty() ? h : applyPointwise(sc, h);
                h = ggml_add(ctx, skip, t);
            }
            if (lv.downsample.w != nullptr)
                h = causalConv(declConv(lv.downsample), h, lv.space_stride, lv.time_stride,
                               /*spatialPad*/ false);
            if (stopAfter == l + 1) probe = h;
        }

    emit_probe:
        if (probe != nullptr)
        {
            // The probe is dumped raw and the caller is expected to be a test.
            ggml_tensor* pOut = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                   probe->ne[0], probe->ne[1], probe->ne[2], probe->ne[3]);
            ggml_tensor* pCopy = ggml_cpy(ctx, probe, pOut);
            ggml_set_output(pCopy);
            ggml_cgraph* pg = ggml_new_graph_custom(ctx, 8192, false);
            ggml_build_forward_expand(pg, pCopy);
            BufferHandle pbuf(nullptr);
            if (!alloc_graph_reuse_gallocr(pg))
            {
                pbuf.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
                if (pbuf.value == nullptr)
                { set_last_error("MiniMaxH3VideoVaeEncode3D: probe buffer alloc failed."); return 0; }
            }
            host_read_barrier();
            for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.data, 0, u.bytes);
            ggml_backend_tensor_set(video, d->video, 0, ggml_nbytes(video));
            if (tsg::compute_graph(g_backend, pg) != GGML_STATUS_SUCCESS)
            { set_last_error("MiniMaxH3VideoVaeEncode3D: probe compute failed."); return 0; }
            tsg::sync_backend(g_backend);
            if (const char* path = getenv("TS_H3_ENC3D_DUMP"))
            {
                std::vector<float> tmp(static_cast<std::size_t>(ggml_nelements(pOut)));
                ggml_backend_tensor_get(pOut, tmp.data(), 0, ggml_nbytes(pOut));
                if (FILE* fp = fopen(path, "wb"))
                {
                    std::int32_t hdr[4] = { (std::int32_t)pOut->ne[0], (std::int32_t)pOut->ne[1],
                                            (std::int32_t)pOut->ne[2], (std::int32_t)pOut->ne[3] };
                    fwrite(hdr, sizeof(std::int32_t), 4, fp);
                    fwrite(tmp.data(), sizeof(float), tmp.size(), fp);
                    fclose(fp);
                }
            }
            clear_last_error();
            return 1;
        }

        ggml_tensor* noW = declVec(d->norm_out_w, static_cast<int>(d->conv_out.ic));
        ggml_tensor* noB = declVec(d->norm_out_b, static_cast<int>(d->conv_out.ic));
        h = groupNormSilu(h, noW, noB);
        if (stopAfter == 100) { probe = h; goto emit_probe; }
        h = causalConv(declConv(d->conv_out), h, 1, 1, true);
        if (stopAfter == 101) { probe = h; goto emit_probe; }

        Conv quant = declConv(d->quant_conv);
        if (!quant.slices.empty()) h = applyPointwise(quant, h);

        if (static_cast<int>(h->ne[3]) != d->latent_frames)
        { set_last_error("MiniMaxH3VideoVaeEncode3D: temporal arithmetic produced an unexpected latent frame count."); return 0; }

        // Keep the posterior MEAN; the logvar half is computed and discarded, exactly
        // as the reference does at inference.
        const int lc = d->latent_channels;
        ggml_tensor* mean = ggml_cont(ctx, ggml_view_4d(ctx, h, h->ne[0], h->ne[1], lc, h->ne[3],
                                                        h->nb[1], h->nb[2], h->nb[3], 0));
        ggml_tensor* outT = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                               mean->ne[0], mean->ne[1], lc, mean->ne[3]);
        ggml_tensor* copied = ggml_cpy(ctx, mean, outT);
        ggml_set_output(copied);

        const std::size_t nodes = static_cast<std::size_t>(d->num_levels) * 512 + 8192;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
        if (graph == nullptr)
        { set_last_error("MiniMaxH3VideoVaeEncode3D: graph alloc failed."); return 0; }
        ggml_build_forward_expand(graph, copied);

        for (int i = 0; i < ggml_graph_n_nodes(graph); i++)
        {
            if (!ggml_backend_supports_op(g_backend, ggml_graph_node(graph, i)))
            {
                set_last_error(std::string("MiniMaxH3VideoVaeEncode3D: op unsupported by backend: ") +
                               ggml_op_name(ggml_graph_node(graph, i)->op));
                return 0;
            }
        }

        BufferHandle buffer(nullptr);
        if (!alloc_graph_reuse_gallocr(graph))
        {
            buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
            if (buffer.value == nullptr)
            { set_last_error("MiniMaxH3VideoVaeEncode3D: buffer alloc failed."); return 0; }
        }

        host_read_barrier();
        for (auto& u : uploads) ggml_backend_tensor_set(u.t, u.data, 0, u.bytes);
        ggml_backend_tensor_set(video, d->video, 0, ggml_nbytes(video));

        if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
        { set_last_error("MiniMaxH3VideoVaeEncode3D: graph compute failed."); return 0; }
        tsg::sync_backend(g_backend);
        ggml_backend_tensor_get(outT, d->out, 0, ggml_nbytes(outT));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("MiniMaxH3VideoVaeEncode3D: unknown error."); return 0; }
}

} // extern "C"
