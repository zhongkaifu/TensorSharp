// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// Wan 2.1 text-to-video native kernels. Three whole-graph entry points drive
// the WanVideo pipeline (TensorSharp.Models/Models/WanVideo):
//
//   TSGgml_WanT5Encode   – the UMT5-XXL text encoder (24 layers, per-layer
//                          relative-attention bias) as ONE graph: token ids in,
//                          final hidden states out. Weights resident-cached by
//                          their GGUF mmap pointer.
//   TSGgml_WanDitForward – one denoising-step velocity prediction of the Wan
//                          DiT (patch embedding, time/text conditioning, N
//                          single-stream blocks with self-attn + cross-attn,
//                          head) as ONE resident-weight graph. On CUDA the
//                          graph is kept persistent per shape so ggml-cuda's
//                          CUDA-graph capture engages.
//   TSGgml_WanVaeDecode  – the Wan causal 3D video VAE decoder: latent
//                          [w,h,t,zc] -> pixels as one graph that iterates the
//                          temporal chunks internally with the causal feature
//                          cache carried between chunks (port of
//                          stable-diffusion.cpp wan_vae.hpp). version 2 adds
//                          the Wan 2.2 TI2V decoder (48-channel latent, DupUp3D
//                          residual shortcuts, 2x2 pixel unpatchify).
//   TSGgml_WanVaeEncode  – the Wan causal 3D video VAE encoder (both the
//                          Wan 2.1 16-channel and Wan 2.2 48-channel variants):
//                          pixels in, posterior mean out, chunked 1+4k frames
//                          with the causal feature caches inside one graph.
//                          Image-to-video conditioning is built on this.
//
// The graph topology mirrors stable-diffusion.cpp's verified Wan implementation
// (src/model/diffusion/wan.hpp, src/model/vae/wan_vae.hpp, src/model/te/t5.hpp);
// the managed side supplies pre-flattened weights where that removes graph work
// (patch embedding as a matmul, causal conv3d as per-temporal-tap 2D kernels).
// ============================================================================
#include "ggml_ops_internal.h"
#include "ggml-impl.h"   // ggml_graph_view, for the segmented MPS conv runner
#include "ggml-alloc.h"
#include "ggml_ops_metal_flash_pad.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace tsg;

extern "C" {

// One (possibly quantized) weight matrix + optional F32 bias for the Wan kernels.
// MUST match managed WanW (TensorSharp.Backends.GGML/GgmlNative.cs).
struct TSGWanW
{
    void* w;                    // weight data (GGUF mmap or stable dequant buffer)
    std::int32_t type;          // ggml_type
    std::int32_t reserved;
    std::int64_t ne0, ne1;      // ggml dims: ne0 = input dim, ne1 = output dim
    std::int64_t bytes;
    void* b;                    // [ne1] F32 bias or null
};

// ---- UMT5-XXL encoder ------------------------------------------------------

struct TSGWanT5LayerW
{
    TSGWanW q, k, v, o;         // attention projections (no bias)
    TSGWanW gate, up, down;     // wi_0 / wi_1 / wo (gated GELU FFN)
    void* attn_norm;            // [dim] F32 RMS gain
    void* ffn_norm;             // [dim] F32 RMS gain
    void* rel_b;                // [heads, 32] F32 relative-attention bias table
};

struct TSGgmlWanT5Desc
{
    const std::int32_t* tokens;       // [n_tokens]
    const std::int32_t* rel_bucket;   // [n_tokens * n_tokens], bucket[q*n + k]
    const float* attn_mask;           // [n_tokens] additive (0 / -inf) or null
    TSGWanW tok_embd;                 // [dim, vocab]
    const TSGWanT5LayerW* layers;
    void* final_norm;                 // [dim] F32 RMS gain
    float* out;                       // [dim * n_tokens] written
    std::int32_t n_tokens, num_layers, dim, ff, heads, head_dim;
    float eps;
    std::int32_t struct_bytes;
};

// ---- Wan DiT ---------------------------------------------------------------

struct TSGWanDitBlockW
{
    void* modulation;           // [6*dim] F32 (per-block learned AdaLN bias)
    TSGWanW sq, sk, sv, so;     // self-attention (+F32 bias each)
    void* s_norm_q;             // [dim] F32 full-dim RMS gain
    void* s_norm_k;             // [dim] F32
    void* norm3_w;              // [dim] F32 (cross-attn LayerNorm affine)
    void* norm3_b;              // [dim] F32
    TSGWanW xq, xk, xv, xo;     // cross-attention (+bias)
    void* x_norm_q;             // [dim] F32
    void* x_norm_k;             // [dim] F32
    TSGWanW ffn0, ffn2;         // FFN (+bias)
};

struct TSGgmlWanDitDesc
{
    float* x;                   // [in_dim*pt*ph*pw, seq] patchified latent tokens (in)
    float* out;                 // [out_dim*pt*ph*pw, seq] velocity tokens (out)
    float* context;             // [text_dim, ctx_len] UMT5 states (zero-padded)
    float* tsin;                // [freq_dim] sinusoidal timestep embedding
    float* tsin0;               // [freq_dim] sinusoid for tokens [0, seq0) or null
    float* cosf;                // [head_dim, seq] RoPE cos (pair-duplicated)
    float* sinf;                // [head_dim, seq] RoPE sin
    TSGWanW patch;              // [in_dim*pt*ph*pw, dim] F32 (pre-flattened conv) + bias
    TSGWanW text0, text2;       // text_embedding.0 / .2 (+bias)
    TSGWanW time0, time2;       // time_embedding.0 / .2 (+bias)
    TSGWanW tproj;              // time_projection.1 (+bias) -> [6*dim]
    TSGWanW head;               // head.head (+bias) -> [out_dim*pt*ph*pw]
    void* head_mod;             // [2*dim] F32 head modulation
    const TSGWanDitBlockW* blocks;
    // seq0 > 0 splits the AdaLN modulation into two token segments: tokens
    // [0, seq0) are modulated with tsin0's timestep, the rest with tsin's
    // (Wan 2.2 TI2V image-to-video: the first latent frame is the clean image
    // conditioned at timestep 0). 0 = uniform timestep over all tokens.
    std::int32_t num_layers, dim, ff, heads, head_dim, seq, ctx_len, freq_dim, text_dim, seq0;
    float eps;
    std::int32_t struct_bytes;
};

// ---- Wan VAE decoder -------------------------------------------------------

// One causal conv3d, decomposed by the managed side into per-temporal-tap 2D
// kernels: tap[j] is the [k, k, ic, oc] kernel for temporal offset j
// (0 = oldest frame). kd == 1 uses tap[0] only.
struct TSGWanVaeConv
{
    void* tap0; void* tap1; void* tap2;
    void* bias;                 // [oc] F32 or null
    std::int32_t kd, k, ic, oc;
    // 1 = taps are F16 (pre-converted managed-side; same round-to-nearest the
    // graph's F32->F16 cast used to apply), 0 = legacy F32 taps that the graph
    // casts in place. F16 halves the resident weight bytes and removes one
    // cast node per conv tap per chunk from every VAE graph.
    std::int32_t tap_type;
    std::int32_t reserved2;
};

struct TSGWanVaeNorm { void* gamma; std::int32_t c; std::int32_t pad; };

struct TSGWanVaeResBlockW
{
    TSGWanVaeNorm n0, n3;
    TSGWanVaeConv c2, c6;       // 3x3x3 causal convs
    TSGWanVaeConv shortcut;     // 1x1x1 (tap0 == null when in == out: identity)
};

struct TSGWanVaeAttnW
{
    TSGWanVaeNorm norm;
    void* qkv_w;                // [c, 3c] F32 matmul weight (1x1 conv)
    void* qkv_b;                // [3c] F32
    void* proj_w;               // [c, c] F32
    void* proj_b;               // [c] F32
    std::int32_t c, pad;
};

struct TSGWanVaeUpsampleW
{
    TSGWanVaeConv time_conv;    // (3,1,1) dim -> 2*dim; tap0 == null => spatial-only
    TSGWanVaeConv sconv;        // 3x3 2D conv after nearest x2 (kd == 1)
};

struct TSGgmlWanVaeDecodeDesc
{
    float* z;                   // [w, h, t, zc] latent (already de-normalized)
    float* out;                 // [w*8, h*8, 1+(t-1)*4, 3] written
    std::int64_t out_len;       // expected element count of out
    TSGWanVaeConv conv2;        // post-quant 1x1x1 (zc -> zc)
    TSGWanVaeConv conv1;        // 3x3x3 (zc -> 384 / 1024)
    TSGWanVaeResBlockW mid0, mid2;
    TSGWanVaeAttnW mid1;
    TSGWanVaeResBlockW res[12]; // 4 scales x 3 residual blocks
    TSGWanVaeUpsampleW up[3];   // after scales 0,1 (temporal+spatial) and 2 (spatial)
    TSGWanVaeNorm head_norm;
    TSGWanVaeConv head_conv;    // 3x3x3 (96 -> 3, or 256 -> 12 for wan2.2)
    std::int32_t zw, zh, zt, zc;
    std::int32_t version;       // 1 (or 0) = wan2.1; 2 = wan2.2 TI2V (DupUp shortcuts)
    std::int32_t patch;         // pixel unpatchify factor (2 for wan2.2, else 1)
    std::int32_t struct_bytes;
};

// One encoder Resample stage: right/bottom-padded 3x3 stride-2 spatial conv,
// plus (downsample3d only) a stride-2 temporal conv with a 1-frame causal cache.
struct TSGWanVaeDownW
{
    TSGWanVaeConv sconv;        // 3x3 stride-2 2D conv (kd == 1)
    TSGWanVaeConv tconv;        // (3,1,1) stride-2 temporal conv; tap0 == null => downsample2d
};

struct TSGgmlWanVaeEncodeDesc
{
    float* x;                   // [px_w, px_h, px_c, px_t] pixels in [-1,1] (pre-patchified for wan2.2)
    float* out;                 // [lw, lh, z_dim, lt] posterior mean, written
    std::int64_t out_len;       // expected element count of out
    TSGWanVaeConv stem;         // encoder conv1: px_c -> dim, 3x3x3 causal
    TSGWanVaeResBlockW res[8];  // 4 scales x 2 residual blocks
    TSGWanVaeDownW down[3];     // after scales 0 (spatial), 1 and 2 (temporal+spatial)
    TSGWanVaeResBlockW mid0, mid2;
    TSGWanVaeAttnW mid1;
    TSGWanVaeNorm head_norm;
    TSGWanVaeConv head_conv;    // 3x3x3 -> 2*z_dim
    TSGWanVaeConv quant;        // quant conv 1x1x1 (2z -> 2z)
    std::int32_t px_w, px_h, px_c, px_t;
    std::int32_t z_dim;
    std::int32_t version;       // 1 (or 0) = wan2.1; 2 = wan2.2 TI2V (AvgDown shortcuts)
    std::int32_t struct_bytes;
    std::int32_t reserved;
};

} // extern "C" (struct declarations)

namespace {

// ---------------------------------------------------------------------------
// Shared graph helpers
// ---------------------------------------------------------------------------

struct WanUpload { ggml_tensor* t; void* d; std::size_t b; };

// Collects the weight-leaf bindings for one graph build: a resident-cache MISS
// is filled immediately (see bind); everything else becomes a gallocr input slot
// uploaded per call from `uploads`.
struct WanBind
{
    ggml_context* ctx = nullptr;
    ggml_backend_dev_t dev = nullptr;
    std::vector<WanUpload> uploads;
    bool barriered = false;

    void bind(ggml_tensor* tt, void* dd, std::size_t bytes)
    {
        if (tt == nullptr || dd == nullptr) return;
        // Read per call, not cached in a static: wan_dit_resident_cache_test
        // toggles this between its reference and its poisoning phase, and a
        // getenv per weight bind is nothing next to the device allocation it guards.
        const char* noResidentEnv = std::getenv("TS_WAN_NO_RESIDENT");
        const bool noResident = noResidentEnv != nullptr && noResidentEnv[0] == '1';
        if (bytes >= 4096 && !noResident)
        {
            ggml_backend_buffer_t buf = nullptr; void* addr = nullptr; bool needs = false;
            if (try_get_cacheable_tensor_buffer(g_backend, dev, tt, dd, bytes, buf, addr, needs)
                && ggml_backend_tensor_alloc(buf, tt, addr) == GGML_STATUS_SUCCESS)
            {
                // A cache MISS just allocated a device buffer that nothing else will
                // ever fill, and the entry is already published in the process-wide
                // resident cache. Fill it HERE rather than deferring to the caller's
                // upload loop: every path that abandons a built graph before reaching
                // that loop — wan_dit_build_persist's VRAM spill guard, a gallocr
                // failure, ggml_new_graph_custom returning null — would otherwise
                // leave the cache holding uninitialised device memory, and the NEXT
                // build would hit that entry with needs == false and silently compute
                // with whatever the driver handed back. Freshly mapped VRAM reads as
                // zeros, so the symptom is a whole model of zero weights: a finite,
                // NaN-free, token-CONSTANT output that no assert catches. This cost
                // the 1088x832x121f Wan request its entire video.
                if (needs)
                {
                    if (!barriered) { host_read_barrier(); barriered = true; }
                    ggml_backend_tensor_set(tt, dd, 0, bytes);
                }
                return;
            }
            invalidate_cached_buffer(dd);
        }
        ggml_set_input(tt);
        uploads.push_back({tt, dd, bytes});
    }

    // Declare + bind a weight matrix (+ optional F32 bias) from a TSGWanW.
    ggml_tensor* w2d(const TSGWanW& s, ggml_tensor** bias_out = nullptr)
    {
        ggml_tensor* wt = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(s.type), s.ne0, s.ne1);
        bind(wt, s.w, static_cast<std::size_t>(s.bytes));
        if (bias_out != nullptr)
        {
            ggml_tensor* bt = nullptr;
            if (s.b != nullptr)
            {
                bt = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.ne1);
                bind(bt, s.b, static_cast<std::size_t>(s.ne1) * sizeof(float));
            }
            *bias_out = bt;
        }
        return wt;
    }

    // Declare + bind a small F32 vector (norm gains, modulation, biases).
    ggml_tensor* f32v(void* data, std::int64_t n)
    {
        if (data == nullptr) return nullptr;
        ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        bind(t, data, static_cast<std::size_t>(n) * sizeof(float));
        return t;
    }
};

// Overflow-safe matmul for quantized weights (see qi_mm in ggml_ops_qwen_image.cpp:
// ggml quantizes the activation to q8_1 whose per-block FP16 sum overflows for
// large activations; scaling is exact for the scale-invariant q8 formats).
constexpr float WAN_MM_SCALE = 1024.0f;

ggml_tensor* wan_mm(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, bool prescale)
{
    if (!prescale) return ggml_mul_mat(ctx, w, x);
    ggml_tensor* xs = ggml_scale(ctx, x, 1.0f / WAN_MM_SCALE);
    return ggml_scale(ctx, ggml_mul_mat(ctx, w, xs), WAN_MM_SCALE);
}

ggml_tensor* wan_lin(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x, ggml_tensor* b, bool prescale = false)
{
    ggml_tensor* o = wan_mm(ctx, w, x, prescale);
    return b != nullptr ? ggml_add(ctx, o, b) : o;
}

// Concatenate `parts` along `dim` as a BALANCED tree instead of the obvious
// left-leaning chain.
//
// `acc = ggml_concat(acc, next, dim)` re-copies everything accumulated so far on
// every step, so joining B pieces of total size N moves O(N*B) bytes. Pairwise
// merging moves O(N*log2(B)). Concatenation is associative and the pairing keeps
// the pieces in order, so the result is bit-identical — this is pure data movement,
// no arithmetic. It matters here because a full-resolution VAE decode joins ~11
// conv bands per convolution and 31 temporal chunks per band, and the decode is
// bandwidth-bound on exactly this kind of generic strided copy (the profile that
// motivated the band budget above put CONT+CONCAT+PAD at ~72% of decode time).
// 31 chunks: 31x the output copied, down to ~5x.
ggml_tensor* wan_concat_all(ggml_context* ctx, std::vector<ggml_tensor*>& parts, int dim)
{
    if (parts.empty()) return nullptr;
    std::vector<ggml_tensor*> next;
    while (parts.size() > 1)
    {
        next.clear();
        next.reserve((parts.size() + 1) / 2);
        std::size_t i = 0;
        for (; i + 1 < parts.size(); i += 2)
            next.push_back(ggml_concat(ctx, parts[i], parts[i + 1], dim));
        if (i < parts.size()) next.push_back(parts[i]);
        parts.swap(next);
    }
    return parts[0];
}

// Token budget for one FFN chunk (see wan_ffn). TS_WAN_DIT_FFN_CHUNK_MB pins the
// intermediate's size; 0 disables chunking.
inline long long wan_ffn_chunk_bytes()
{
    static const long long mb = []() -> long long {
        const char* e = std::getenv("TS_WAN_DIT_FFN_CHUNK_MB");
        return e != nullptr ? std::strtoll(e, nullptr, 10) : 256;
    }();
    return mb << 20;
}

// Rows of `x` one FFN chunk covers; `seq` (i.e. unchunked) when chunking is off or
// the whole intermediate already fits. wan_ffn and the graph's node budget must
// agree on this, so both go through here.
inline long long wan_ffn_chunk_rows(int ff, int seq)
{
    const long long budget = wan_ffn_chunk_bytes();
    if (budget <= 0) return seq;
    const long long perTok = static_cast<long long>(ff) * static_cast<long long>(sizeof(float));
    const long long rows = budget / std::max<long long>(1, perTok);
    return (rows <= 0 || rows >= seq) ? seq : rows;
}

inline int wan_ffn_chunk_count(int ff, int seq)
{
    const long long rows = wan_ffn_chunk_rows(ff, seq);
    return rows >= seq ? 1 : static_cast<int>((seq + rows - 1) / rows);
}

// Nodes one DiT block costs, including the extra matmul/gelu/join nodes chunking
// adds. A block is ~130 nodes with the TI2V two-segment modulation active; each
// extra FFN chunk adds 5 compute nodes plus one tree-join node. ggml ASSERTS on
// graph overflow (ggml.c:7175), so this is sized generously — the only cost of
// slack is graph metadata, a few hundred KB.
inline std::size_t wan_dit_nodes(int num_layers, int ff, int seq)
{
    const std::size_t perChunk = 8;
    const std::size_t perBlock = 160 + perChunk * static_cast<std::size_t>(wan_ffn_chunk_count(ff, seq));
    return static_cast<std::size_t>(num_layers) * perBlock + 2048;
}

// The DiT feed-forward, evaluated in TOKEN CHUNKS.
//
// The intermediate is [ff, seq] — at Wan 2.2 TI2V's ff = 14336 and the 121-frame
// seq = 27404 that is 1.5 GiB in one tensor, the single largest allocation in the
// graph. It has no cross-token dependencies (both linears and the GELU are
// per-token), so slicing the tokens and joining the results is EXACT: each chunk
// runs the identical arithmetic on a contiguous row range, and a row range of a
// contiguous [dim, seq] tensor is itself contiguous, so the matmuls see exactly
// the operands they saw before.
//
// This is the memory-for-nothing trade diffusers exposes as enable_forward_chunking
// (src/diffusers/models/attention.py _chunked_feed_forward) and FastVideo as
// _chunked_feed_forward. It matters here because the whole 27404-token graph
// measured 11.4 GiB on a 16 GB card, leaving 359 MiB free — under the spill guard,
// so the persistent/CUDA-graph path stayed off and every pass ran against a
// device whose working set did not fit.
ggml_tensor* wan_ffn(ggml_context* ctx, ggml_tensor* w0, ggml_tensor* b0,
                     ggml_tensor* w2, ggml_tensor* b2, ggml_tensor* x,
                     int dim, int ff, int seq)
{
    auto once = [&](ggml_tensor* xs) {
        return wan_lin(ctx, w2, ggml_gelu(ctx, wan_lin(ctx, w0, xs, b0)), b2);
    };

    // Rows per chunk that keep the intermediate under the budget. ggml_gelu can be
    // aliased onto its parent by gallocr, so the intermediate counts once.
    const long long chunk = wan_ffn_chunk_rows(ff, seq);
    if (chunk >= seq || !ggml_is_contiguous(x))
        return once(x);

    // Keep the chunk count modest: each chunk adds two matmul launches per block,
    // and the join costs one copy of the [dim, seq] result per tree level.
    const int nChunks = static_cast<int>((seq + chunk - 1) / chunk);
    std::vector<ggml_tensor*> parts;
    parts.reserve(static_cast<std::size_t>(nChunks));
    for (long long t0 = 0; t0 < seq; t0 += chunk)
    {
        const long long rows = std::min<long long>(chunk, seq - t0);
        ggml_tensor* xs = ggml_view_2d(ctx, x, dim, rows, x->nb[1],
                                       static_cast<std::size_t>(t0) * x->nb[1]);
        parts.push_back(once(xs));
    }
    return wan_concat_all(ctx, parts, 1);
}

// RMS norm over ne0 then * gain.
ggml_tensor* wan_rms(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, float eps)
{
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// Interleaved RoPE with pair-duplicated cos/sin tables [head_dim, seq]; the
// output uses the half-split channel layout, which is dot-product-invariant when
// applied to both q and k. The half-split form needs only a 3-D dim0 concat
// (one kernel launch); the interleaved concat of 4-D pair views launches one
// kernel per ne3 slice on ggml-cuda.
ggml_tensor* wan_rope(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cosf, ggml_tensor* sinf,
                      int head_dim, int heads, int seq)
{
    const int half = head_dim / 2;
    ggml_tensor* x4 = ggml_reshape_4d(ctx, x, 2, half, heads, seq);
    ggml_tensor* even = ggml_view_4d(ctx, x4, 1, half, heads, seq, x4->nb[1], x4->nb[2], x4->nb[3], 0);
    ggml_tensor* odd  = ggml_view_4d(ctx, x4, 1, half, heads, seq, x4->nb[1], x4->nb[2], x4->nb[3], x4->nb[0]);
    ggml_tensor* cosh = ggml_view_4d(ctx, cosf, 1, half, 1, seq, 2 * cosf->nb[0], cosf->nb[1], cosf->nb[1], 0);
    ggml_tensor* sinh = ggml_view_4d(ctx, sinf, 1, half, 1, seq, 2 * sinf->nb[0], sinf->nb[1], sinf->nb[1], 0);
    ggml_tensor* ep = ggml_sub(ctx, ggml_mul(ctx, even, cosh), ggml_mul(ctx, odd, sinh));
    ggml_tensor* op = ggml_add(ctx, ggml_mul(ctx, odd, cosh), ggml_mul(ctx, even, sinh));
    ggml_tensor* ep3 = ggml_reshape_3d(ctx, ep, half, heads, seq);
    ggml_tensor* op3 = ggml_reshape_3d(ctx, op, half, heads, seq);
    return ggml_concat(ctx, ep3, op3, 0);
}

inline bool wan_flash_enabled()
{
    static const bool on = []{ const char* e = std::getenv("TS_WAN_DIT_FLASH"); return e == nullptr || e[0] != '0'; }();
    return on;
}

// Keys/values are handed to attention as F16. Every backend's flash-attention
// kernel is built around an F16 KV cache: ggml-metal instantiates the F32-KV
// kernel with simdgroup_float8x8 accumulators (FA_TYPES_F32) where the F16 one
// uses simdgroup_half8x8, and the F32 tiles also cost twice the bandwidth in a
// kernel that re-streams K and V once per 8-query threadgroup. Measured on an
// M5 Pro at the Wan 2.2 TI2V 720p/121-frame shape (seq 27404, 24 heads, head
// dim 128), one self-attention: 4872 ms F32 KV vs 2439 ms F16 KV — 2.0x, and
// 30 blocks of it is the bulk of a denoising step. Q stays F32 (the Metal
// kernel asserts it), and F16 K/V is what every reference implementation feeds
// its attention (PyTorch/diffusers run the whole DiT in bf16/fp16;
// stable-diffusion.cpp casts K/V to F16 before ggml_flash_attn_ext), so this
// costs no accuracy the reference pipelines do not already spend.
// TS_WAN_DIT_KV_F16=0 restores F32 keys/values.
inline bool wan_kv_f16_enabled()
{
    static const bool on = []{ const char* e = std::getenv("TS_WAN_DIT_KV_F16"); return e == nullptr || e[0] != '0'; }();
    return on;
}

// Attention over q [hd, n_q, heads], k/v [hd, n_kv, heads]. Wan DiT
// self-attention is fully bidirectional, so its flash path must be unmasked.
// ggml-cuda handles a non-aligned KV tail directly; padding it to 256 used to
// require a dense [seq_pad, seq_pad] F16 mask, which is quadratic (about 18 GiB
// for a 245-frame 480p request) and contains no useful model information.
// Falls back to materialized scores+softmax on backends without flash support.
// Returns [hd*heads, n_q].
ggml_tensor* wan_attention(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                           ggml_tensor* mask, int dim, int n_q, float scale)
{
    if (wan_flash_enabled())
    {
        ggml_tensor* fa = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_prec_set_acc(fa, GGML_PREC_F32);
        if (backend_supports_op(fa))
            return ggml_reshape_2d(ctx, fa, dim, n_q);
    }
    // Materialized reference path (backends without flash support; O(n_kv * n_q)
    // scores). The caller's k/v are already padded when a mask is given, and
    // soft_max_ext folds the scale and the (F16) additive mask in one op.
    // k/v arrive F16 here (wan_heads_seq_kv); ggml_mul_mat takes an F16 src0
    // against an F32 src1, so this path needs no change for that.
    ggml_tensor* kq = ggml_mul_mat(ctx, k, q);                       // [n_kv, n_q, heads]
    ggml_tensor* m = mask != nullptr
        ? ggml_view_2d(ctx, mask, k->ne[1], n_q, mask->nb[1], 0)
        : nullptr;
    kq = ggml_soft_max_ext(ctx, kq, m, scale, 0.0f);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));  // [n_kv, hd, heads]
    ggml_tensor* kqv = ggml_mul_mat(ctx, vt, kq);                        // [hd, n_q, heads]
    ggml_tensor* merged = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)); // [hd, heads, n_q]
    return ggml_reshape_2d(ctx, merged, dim, n_q);
}

// [hd, heads, seq] -> [hd, seq, heads] (flash-attn layout)
ggml_tensor* wan_heads_seq(ggml_context* ctx, ggml_tensor* x)
{
    return ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3));
}

// Same reshape for a key/value projection, landing in F16 when the KV cast is
// enabled. ggml_cpy into a pre-typed destination does the permute and the
// narrowing in ONE pass, so the F16 path also writes half the bytes the plain
// ggml_cont did — it is strictly cheaper than the F32 layout change it replaces.
ggml_tensor* wan_heads_seq_kv(ggml_context* ctx, ggml_tensor* x)
{
    if (!wan_kv_f16_enabled()) return wan_heads_seq(ctx, x);
    ggml_tensor* p = ggml_permute(ctx, x, 0, 2, 1, 3);            // [hd, seq, heads]
    ggml_tensor* dst = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, p->ne[0], p->ne[1], p->ne[2]);
    ggml_tensor* cast = ggml_cpy(ctx, p, dst);
    // A strided F32 -> F16 copy is supported on ggml-cpu / -metal / -cuda; keep the
    // F32 layout change for any backend whose dup kernel rejects this combination
    // rather than failing the whole graph.
    return backend_supports_op(cast) ? cast : wan_heads_seq(ctx, x);
}

// ---------------------------------------------------------------------------
// UMT5-XXL encoder graph
// ---------------------------------------------------------------------------

bool wan_t5_build_and_run(const TSGgmlWanT5Desc* d)
{
    const int n = d->n_tokens, dim = d->dim, heads = d->heads, hd = d->head_dim, nl = d->num_layers;

    const std::size_t nodes = static_cast<std::size_t>(nl) * 48 + 512;
    const std::size_t meta = ggml_tensor_overhead() * (nodes + 512)
                             + ggml_graph_overhead_custom(nodes, false) + (4u << 20);
    ggml_init_params ip{ meta, nullptr, true };
    ContextHandle context(ggml_init(ip));
    if (context.value == nullptr) { set_last_error("WanT5Encode: ctx alloc failed."); return false; }
    ggml_context* ctx = context.value;

    WanBind wb; wb.ctx = ctx; wb.dev = ggml_backend_get_device(g_backend);

    ggml_tensor* tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_tensor* bucket = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, static_cast<std::int64_t>(n) * n);
    ggml_tensor* mask = d->attn_mask != nullptr ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n) : nullptr;
    ggml_tensor* outT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, n);

    ggml_tensor* embd = wb.w2d(d->tok_embd);
    ggml_tensor* x = ggml_get_rows(ctx, embd, tokens);               // [dim, n]

    ggml_tensor* mask3 = mask != nullptr ? ggml_reshape_3d(ctx, mask, n, 1, 1) : nullptr;

    for (int l = 0; l < nl; l++)
    {
        const TSGWanT5LayerW& s = d->layers[l];
        ggml_tensor* qw = wb.w2d(s.q); ggml_tensor* kw = wb.w2d(s.k);
        ggml_tensor* vw = wb.w2d(s.v); ggml_tensor* ow = wb.w2d(s.o);
        ggml_tensor* gw = wb.w2d(s.gate); ggml_tensor* uw = wb.w2d(s.up); ggml_tensor* dw = wb.w2d(s.down);
        ggml_tensor* an = wb.f32v(s.attn_norm, dim);
        ggml_tensor* fn = wb.f32v(s.ffn_norm, dim);
        ggml_tensor* rb = wb.f32v(s.rel_b, static_cast<std::int64_t>(heads) * 32);

        // relative position bias: gather per (q,k) bucket -> [n_k, n_q, heads]
        ggml_tensor* rb2 = ggml_reshape_2d(ctx, rb, heads, 32);
        ggml_tensor* pos = ggml_get_rows(ctx, rb2, bucket);          // [heads, n*n]
        pos = ggml_reshape_3d(ctx, pos, heads, n, n);                // [heads, k, q]
        pos = ggml_cont(ctx, ggml_permute(ctx, pos, 2, 0, 1, 3));    // [k, q, heads]
        ggml_tensor* bias = mask3 != nullptr ? ggml_add(ctx, pos, mask3) : pos;

        // self-attention (T5: no dot-product scaling, bias added to logits)
        ggml_tensor* h = wan_rms(ctx, x, an, d->eps);
        ggml_tensor* q = wan_mm(ctx, qw, h, true);
        ggml_tensor* k = wan_mm(ctx, kw, h, true);
        ggml_tensor* v = wan_mm(ctx, vw, h, true);
        ggml_tensor* q3 = wan_heads_seq(ctx, ggml_reshape_3d(ctx, q, hd, heads, n));  // [hd, n, heads]
        ggml_tensor* k3 = wan_heads_seq(ctx, ggml_reshape_3d(ctx, k, hd, heads, n));
        ggml_tensor* kq = ggml_mul_mat(ctx, k3, q3);                 // [n_k, n_q, heads]
        kq = ggml_add(ctx, kq, bias);
        kq = ggml_soft_max(ctx, kq);
        ggml_tensor* v3 = ggml_reshape_3d(ctx, v, hd, heads, n);
        ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v3, 1, 2, 0, 3));  // [n, hd, heads]
        ggml_tensor* kqv = ggml_mul_mat(ctx, vt, kq);                // [hd, n_q, heads]
        ggml_tensor* merged = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3));
        merged = ggml_reshape_2d(ctx, merged, dim, n);
        x = ggml_add(ctx, x, wan_mm(ctx, ow, merged, true));

        // gated-GELU FFN (T5DenseGatedActDense)
        ggml_tensor* h2 = wan_rms(ctx, x, fn, d->eps);
        ggml_tensor* hg = ggml_gelu(ctx, wan_mm(ctx, gw, h2, true));
        ggml_tensor* hu = wan_mm(ctx, uw, h2, true);
        x = ggml_add(ctx, x, wan_mm(ctx, dw, ggml_mul(ctx, hg, hu), true));
    }

    ggml_tensor* fnorm = wb.f32v(d->final_norm, dim);
    x = wan_rms(ctx, x, fnorm, d->eps);
    ggml_tensor* outc = ggml_cpy(ctx, x, outT);
    ggml_set_output(outc);

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
    ggml_build_forward_expand(graph, outc);

    ggml_set_input(tokens);
    ggml_set_input(bucket);
    if (mask != nullptr) ggml_set_input(mask);

    BufferHandle buffer(nullptr);
    if (!alloc_graph_reuse_gallocr(graph))
    {
        buffer.value = ggml_backend_alloc_ctx_tensors(ctx, g_backend);
        if (buffer.value == nullptr) { set_last_error("WanT5Encode: buffer alloc failed."); return false; }
    }

    host_read_barrier();
    for (auto& u : wb.uploads) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
    ggml_backend_tensor_set(tokens, d->tokens, 0, static_cast<std::size_t>(n) * sizeof(std::int32_t));
    ggml_backend_tensor_set(bucket, d->rel_bucket, 0, static_cast<std::size_t>(n) * n * sizeof(std::int32_t));
    if (mask != nullptr)
        ggml_backend_tensor_set(mask, d->attn_mask, 0, static_cast<std::size_t>(n) * sizeof(float));

    if (tsg::compute_graph(g_backend, graph) != GGML_STATUS_SUCCESS)
    { set_last_error("WanT5Encode: graph compute failed."); return false; }
    tsg::sync_backend(g_backend);
    ggml_backend_tensor_get(outT, d->out, 0, static_cast<std::size_t>(dim) * n * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// Wan DiT graph
// ---------------------------------------------------------------------------

struct WanDitGraph
{
    ggml_cgraph* graph = nullptr;
    ggml_tensor *xIn = nullptr, *ctxIn = nullptr, *tsinIn = nullptr, *tsin0In = nullptr;
    ggml_tensor *cosIn = nullptr, *sinIn = nullptr, *outT = nullptr;
    std::vector<WanUpload> uploads;
    // TS_WAN_DIT_TRACE: named intermediates flagged as graph outputs (so gallocr
    // cannot alias their buffers) whose stats are printed after compute.
    std::vector<std::pair<std::string, ggml_tensor*>> taps;
};

// Per-block device-side weight leaves.
struct WanDitBlockT
{
    ggml_tensor *mod;
    ggml_tensor *sqw, *sqb, *skw, *skb, *svw, *svb, *sow, *sob;
    ggml_tensor *snq, *snk;
    ggml_tensor *n3w, *n3b;
    ggml_tensor *xqw, *xqb, *xkw, *xkb, *xvw, *xvb, *xow, *xob;
    ggml_tensor *xnq, *xnk;
    ggml_tensor *f0w, *f0b, *f2w, *f2b;
};

bool wan_dit_build_graph(ggml_context* ctx, const TSGgmlWanDitDesc* d, WanDitGraph& g)
{
    // TS_WAN_DIT_MAX_LAYERS=k: build only the first k blocks (debug bisect aid).
    static const int maxLayers = []() {
        const char* e = std::getenv("TS_WAN_DIT_MAX_LAYERS");
        return e != nullptr ? std::atoi(e) : -1;
    }();
    const int dim = d->dim, heads = d->heads, hd = d->head_dim, seq = d->seq, cl = d->ctx_len;
    const int nl = maxLayers >= 0 && maxLayers < d->num_layers ? maxLayers : d->num_layers;
    const float eps = d->eps;
    const int in_tok = static_cast<int>(d->patch.ne0);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    WanBind wb; wb.ctx = ctx; wb.dev = ggml_backend_get_device(g_backend);
    const bool traceOn = std::getenv("TS_WAN_DIT_TRACE") != nullptr;
    auto tap = [&](const char* name, ggml_tensor* t) {
        if (!traceOn || t == nullptr) return;
        ggml_set_output(t);
        g.taps.emplace_back(name, t);
    };

    // seq0 > 0: the first seq0 tokens (the conditioning image's latent frame in
    // Wan 2.2 TI2V i2v) are AdaLN-modulated with tsin0's timestep instead of
    // tsin's. Attention and every matmul still run over the full joint sequence.
    const int s0 = (d->seq0 > 0 && d->tsin0 != nullptr && d->seq0 < seq) ? d->seq0 : 0;

    g.xIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_tok, seq);
    g.ctxIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d->text_dim, cl);
    g.tsinIn = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d->freq_dim);
    g.tsin0In = s0 > 0 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d->freq_dim) : nullptr;
    g.cosIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, seq);
    g.sinIn = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hd, seq);
    g.outT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, static_cast<int>(d->head.ne1), seq);

    // ---- prelude weights ----
    ggml_tensor *patchB, *t0B, *t2B, *ti0B, *ti2B, *tpB, *headB;
    ggml_tensor* patchW = wb.w2d(d->patch, &patchB);
    ggml_tensor* text0W = wb.w2d(d->text0, &t0B);
    ggml_tensor* text2W = wb.w2d(d->text2, &t2B);
    ggml_tensor* time0W = wb.w2d(d->time0, &ti0B);
    ggml_tensor* time2W = wb.w2d(d->time2, &ti2B);
    ggml_tensor* tprojW = wb.w2d(d->tproj, &tpB);
    ggml_tensor* headW = wb.w2d(d->head, &headB);
    ggml_tensor* headMod = wb.f32v(d->head_mod, 2LL * dim);

    // ---- per-block weights ----
    std::vector<WanDitBlockT> bt(nl);
    for (int l = 0; l < nl; l++)
    {
        const TSGWanDitBlockW& s = d->blocks[l];
        WanDitBlockT& b = bt[l];
        b.mod = wb.f32v(s.modulation, 6LL * dim);
        b.sqw = wb.w2d(s.sq, &b.sqb); b.skw = wb.w2d(s.sk, &b.skb);
        b.svw = wb.w2d(s.sv, &b.svb); b.sow = wb.w2d(s.so, &b.sob);
        b.snq = wb.f32v(s.s_norm_q, dim); b.snk = wb.f32v(s.s_norm_k, dim);
        b.n3w = wb.f32v(s.norm3_w, dim); b.n3b = wb.f32v(s.norm3_b, dim);
        b.xqw = wb.w2d(s.xq, &b.xqb); b.xkw = wb.w2d(s.xk, &b.xkb);
        b.xvw = wb.w2d(s.xv, &b.xvb); b.xow = wb.w2d(s.xo, &b.xob);
        b.xnq = wb.f32v(s.x_norm_q, dim); b.xnk = wb.f32v(s.x_norm_k, dim);
        b.f0w = wb.w2d(s.ffn0, &b.f0b); b.f2w = wb.w2d(s.ffn2, &b.f2b);
    }

    // ---- prelude ----
    // patch embedding (pre-flattened conv3d): [in_tok, seq] -> [dim, seq]
    ggml_tensor* x = wan_lin(ctx, patchW, g.xIn, patchB);
    tap("patch_embed", x);

    // time embedding: sinusoid [freq_dim] -> [dim]; e0 = time_projection(silu(e)) -> [6*dim]
    auto timeEmbed = [&](ggml_tensor* tsin, ggml_tensor** e2dOut) -> ggml_tensor*
    {
        ggml_tensor* e = wan_lin(ctx, time0W, tsin, ti0B);
        e = ggml_silu(ctx, e);
        e = wan_lin(ctx, time2W, e, ti2B);                            // [dim]
        ggml_tensor* e2d = ggml_reshape_2d(ctx, e, dim, 1);
        *e2dOut = e2d;
        return wan_lin(ctx, tprojW, ggml_silu(ctx, e2d), tpB);        // [6*dim, 1]
    };
    ggml_tensor* e2d = nullptr;
    ggml_tensor* e0 = timeEmbed(g.tsinIn, &e2d);
    tap("time_e", e2d); tap("time_e0", e0);
    ggml_tensor* e2dB = nullptr;
    ggml_tensor* e0B = s0 > 0 ? timeEmbed(g.tsin0In, &e2dB) : nullptr;

    // Apply f per token segment: rows [0, s0) with the B (tsin0) parameters, rows
    // [s0, seq) with the A (tsin) parameters; concat restores [dim', seq]. The row
    // ranges of a contiguous [dim', seq] tensor are themselves contiguous views.
    auto segRows = [&](ggml_tensor* x, std::int64_t from, std::int64_t count) {
        return ggml_view_2d(ctx, x, x->ne[0], count, x->nb[1],
                            static_cast<std::size_t>(from) * x->nb[1]);
    };
    auto segModulate = [&](ggml_tensor* n, ggml_tensor* shiftA, ggml_tensor* scaleA,
                           ggml_tensor* shiftB, ggml_tensor* scaleB) -> ggml_tensor*
    {
        if (s0 == 0)
            return ggml_add(ctx, ggml_add(ctx, n, ggml_mul(ctx, n, scaleA)), shiftA);
        ggml_tensor* nB = segRows(n, 0, s0);
        ggml_tensor* nA = segRows(n, s0, seq - s0);
        ggml_tensor* yB = ggml_add(ctx, ggml_add(ctx, nB, ggml_mul(ctx, nB, scaleB)), shiftB);
        ggml_tensor* yA = ggml_add(ctx, ggml_add(ctx, nA, ggml_mul(ctx, nA, scaleA)), shiftA);
        return ggml_concat(ctx, yB, yA, 1);
    };
    auto segGate = [&](ggml_tensor* v, ggml_tensor* gateA, ggml_tensor* gateB) -> ggml_tensor*
    {
        if (s0 == 0)
            return ggml_mul(ctx, v, gateA);
        ggml_tensor* vB = segRows(v, 0, s0);
        ggml_tensor* vA = segRows(v, s0, seq - s0);
        return ggml_concat(ctx, ggml_mul(ctx, vB, gateB), ggml_mul(ctx, vA, gateA), 1);
    };

    // text embedding: [text_dim, cl] -> [dim, cl] (GELU-tanh between the linears)
    ggml_tensor* txt = wan_lin(ctx, text0W, g.ctxIn, t0B);
    txt = ggml_gelu(ctx, txt);
    txt = wan_lin(ctx, text2W, txt, t2B);                             // [dim, cl]
    tap("text_embed", txt);

    // ---- blocks ----
    for (int l = 0; l < nl; l++)
    {
        const WanDitBlockT& b = bt[l];

        // e_b = e0 + per-block modulation -> six [dim, 1] chunks (per timestep segment)
        ggml_tensor* mod2d = ggml_reshape_2d(ctx, b.mod, 6LL * dim, 1);
        ggml_tensor* eb = ggml_add(ctx, e0, mod2d);
        ggml_tensor* ebB = s0 > 0 ? ggml_add(ctx, e0B, mod2d) : nullptr;
        auto chunk = [&](ggml_tensor* e6, int k) -> ggml_tensor* {
            if (e6 == nullptr) return nullptr;
            return ggml_view_2d(ctx, e6, dim, 1, e6->nb[1], static_cast<std::size_t>(k) * dim * sizeof(float));
        };
        ggml_tensor* eShiftA = chunk(eb, 0); ggml_tensor* eScaleA = chunk(eb, 1); ggml_tensor* eGateA = chunk(eb, 2);
        ggml_tensor* eShiftM = chunk(eb, 3); ggml_tensor* eScaleM = chunk(eb, 4); ggml_tensor* eGateM = chunk(eb, 5);
        ggml_tensor* eShiftAB = chunk(ebB, 0); ggml_tensor* eScaleAB = chunk(ebB, 1); ggml_tensor* eGateAB = chunk(ebB, 2);
        ggml_tensor* eShiftMB = chunk(ebB, 3); ggml_tensor* eScaleMB = chunk(ebB, 4); ggml_tensor* eGateMB = chunk(ebB, 5);

        // --- self-attention sub-layer ---
        ggml_tensor* n1 = ggml_norm(ctx, x, eps);                    // LayerNorm, no affine
        ggml_tensor* y = segModulate(n1, eShiftA, eScaleA, eShiftAB, eScaleAB);

        ggml_tensor* q = wan_rms(ctx, wan_lin(ctx, b.sqw, y, b.sqb), b.snq, eps);
        ggml_tensor* k = wan_rms(ctx, wan_lin(ctx, b.skw, y, b.skb), b.snk, eps);
        ggml_tensor* v = wan_lin(ctx, b.svw, y, b.svb);

        ggml_tensor* q3 = ggml_reshape_3d(ctx, q, hd, heads, seq);
        ggml_tensor* k3 = ggml_reshape_3d(ctx, k, hd, heads, seq);
        q3 = wan_rope(ctx, q3, g.cosIn, g.sinIn, hd, heads, seq);
        k3 = wan_rope(ctx, k3, g.cosIn, g.sinIn, hd, heads, seq);

        ggml_tensor* qa = wan_heads_seq(ctx, q3);                    // [hd, seq, heads]
        ggml_tensor* ka = wan_heads_seq_kv(ctx, k3);                 // F16 (see wan_kv_f16_enabled)
        ggml_tensor* va = wan_heads_seq_kv(ctx, ggml_reshape_3d(ctx, v, hd, heads, seq));
        ggml_tensor* attn = wan_attention(ctx, qa, ka, va, nullptr, dim, seq, scale);
        attn = wan_lin(ctx, b.sow, attn, b.sob);
        x = ggml_add(ctx, x, segGate(attn, eGateA, eGateAB));

        // --- cross-attention sub-layer (affine LayerNorm, no gating) ---
        ggml_tensor* n3 = ggml_norm(ctx, x, eps);
        n3 = ggml_add(ctx, ggml_mul(ctx, n3, b.n3w), b.n3b);
        ggml_tensor* xq = wan_rms(ctx, wan_lin(ctx, b.xqw, n3, b.xqb), b.xnq, eps);
        ggml_tensor* xk = wan_rms(ctx, wan_lin(ctx, b.xkw, txt, b.xkb), b.xnk, eps);
        ggml_tensor* xv = wan_lin(ctx, b.xvw, txt, b.xvb);
        ggml_tensor* xqa = wan_heads_seq(ctx, ggml_reshape_3d(ctx, xq, hd, heads, seq));
        ggml_tensor* xka = wan_heads_seq_kv(ctx, ggml_reshape_3d(ctx, xk, hd, heads, cl));
        ggml_tensor* xva = wan_heads_seq_kv(ctx, ggml_reshape_3d(ctx, xv, hd, heads, cl));
        // ctx_len is a multiple of the KV stride (512), so flash needs no mask here.
        ggml_tensor* xattn = wan_attention(ctx, xqa, xka, xva, nullptr, dim, seq, scale);
        x = ggml_add(ctx, x, wan_lin(ctx, b.xow, xattn, b.xob));

        // --- FFN sub-layer ---
        ggml_tensor* n2 = ggml_norm(ctx, x, eps);
        ggml_tensor* ym = segModulate(n2, eShiftM, eScaleM, eShiftMB, eScaleMB);
        ggml_tensor* mlp = wan_ffn(ctx, b.f0w, b.f0b, b.f2w, b.f2b, ym, dim, d->ff, seq);
        x = ggml_add(ctx, x, segGate(mlp, eGateM, eGateMB));
        if (l == 0 || l == nl - 1)
        {
            tap((std::string("block") + std::to_string(l) + "_out").c_str(), x);
        }
    }

    // ---- head: modulated norm + projection ----
    // es = head_mod([2, dim]) + e; x = norm(x) * (1 + es[1]) + es[0]
    ggml_tensor* hm = ggml_reshape_2d(ctx, headMod, dim, 2);
    auto headChunks = [&](ggml_tensor* eSeg, ggml_tensor** shift, ggml_tensor** scaleT)
    {
        ggml_tensor* he = ggml_add(ctx, hm, eSeg);                   // broadcast e over the 2 rows
        *shift = ggml_view_2d(ctx, he, dim, 1, he->nb[1], 0);
        *scaleT = ggml_view_2d(ctx, he, dim, 1, he->nb[1], he->nb[1]);
    };
    ggml_tensor *hShift, *hScale, *hShiftB = nullptr, *hScaleB = nullptr;
    headChunks(e2d, &hShift, &hScale);
    if (s0 > 0) headChunks(e2dB, &hShiftB, &hScaleB);
    ggml_tensor* hn = ggml_norm(ctx, x, eps);
    hn = segModulate(hn, hShift, hScale, hShiftB, hScaleB);
    tap("head_norm", hn);
    ggml_tensor* outv = wan_lin(ctx, headW, hn, headB);              // [out_tok, seq]

    ggml_tensor* outc = ggml_cpy(ctx, outv, g.outT);
    ggml_set_output(outc);

    const std::size_t nodes = wan_dit_nodes(nl, d->ff, d->seq);
    g.graph = ggml_new_graph_custom(ctx, nodes, false);
    if (g.graph == nullptr) return false;
    ggml_build_forward_expand(g.graph, outc);
    for (auto& [tname, tt] : g.taps) ggml_build_forward_expand(g.graph, tt);

    ggml_set_input(g.xIn); ggml_set_input(g.ctxIn); ggml_set_input(g.tsinIn);
    if (g.tsin0In != nullptr) ggml_set_input(g.tsin0In);
    ggml_set_input(g.cosIn); ggml_set_input(g.sinIn);

    g.uploads = std::move(wb.uploads);
    return true;
}

// Persistent per-shape entry (CUDA-graph capture; same pattern as QiForwardPersist).
struct WanDitPersist
{
    bool valid = false;
    ggml_context* ctx = nullptr;
    ggml_gallocr_t galloc = nullptr;
    WanDitGraph g{};
    int seq = 0, cl = 0, nl = 0;
    const void* wkey = nullptr;

    int seq0 = 0;

    bool matches(const TSGgmlWanDitDesc* d) const
    {
        return valid && seq == d->seq && cl == d->ctx_len && nl == d->num_layers &&
               seq0 == d->seq0 && wkey == d->blocks[0].sq.w;
    }
    void reset()
    {
        if (galloc) { ggml_gallocr_free(galloc); galloc = nullptr; }
        if (ctx) { ggml_free(ctx); ctx = nullptr; }
        g = WanDitGraph{}; valid = false;
        seq = cl = nl = seq0 = 0; wkey = nullptr;
    }
};

constexpr int kWanDitCacheMax = 2;
WanDitPersist g_wanDit[kWanDitCacheMax];
int g_wanDitRR = 0;

// Shapes whose persistent graph does not fit in device memory. Building one costs
// a full 30-block graph construction plus a multi-GB gallocr allocation that is
// then thrown away; without this memo a 121-frame request pays that on EVERY
// denoise pass and reprints the fallback notice each time.
struct WanDitTooBig { int seq, cl, nl, seq0; const void* wkey; };
std::vector<WanDitTooBig> g_wanDitTooBig;

bool wan_dit_too_big(const TSGgmlWanDitDesc* d)
{
    for (const auto& t : g_wanDitTooBig)
        if (t.seq == d->seq && t.cl == d->ctx_len && t.nl == d->num_layers
            && t.seq0 == d->seq0 && t.wkey == d->blocks[0].sq.w)
            return true;
    return false;
}

void wan_dit_mark_too_big(const TSGgmlWanDitDesc* d)
{
    if (!wan_dit_too_big(d))
        g_wanDitTooBig.push_back({ d->seq, d->ctx_len, d->num_layers, d->seq0, d->blocks[0].sq.w });
}

bool wan_dit_capture_enabled()
{
    // Read per call (once per forward): wan_dit_resident_cache_test needs to
    // disable and re-enable the persistent builder inside one process.
    const char* e = std::getenv("TS_WAN_DIT_CAPTURE");
    const int s = (e && e[0] == '0') ? 0 : 1;
    if (s == 0 || g_backend == nullptr) return false;
    const char* name = ggml_backend_name(g_backend);
    return name != nullptr && std::strncmp(name, "CUDA", 4) == 0;
}

void wan_dit_upload_inputs(const WanDitGraph& g, const TSGgmlWanDitDesc* d)
{
    if (g.xIn != nullptr && g.xIn->buffer != nullptr)
        ggml_backend_tensor_set(g.xIn, d->x, 0, static_cast<std::size_t>(d->patch.ne0) * d->seq * sizeof(float));
    if (g.ctxIn != nullptr && g.ctxIn->buffer != nullptr)
        ggml_backend_tensor_set(g.ctxIn, d->context, 0, static_cast<std::size_t>(d->text_dim) * d->ctx_len * sizeof(float));
    if (g.tsinIn != nullptr && g.tsinIn->buffer != nullptr)
        ggml_backend_tensor_set(g.tsinIn, d->tsin, 0, static_cast<std::size_t>(d->freq_dim) * sizeof(float));
    if (g.tsin0In != nullptr && g.tsin0In->buffer != nullptr && d->tsin0 != nullptr)
        ggml_backend_tensor_set(g.tsin0In, d->tsin0, 0, static_cast<std::size_t>(d->freq_dim) * sizeof(float));
    if (g.cosIn != nullptr && g.cosIn->buffer != nullptr)
        ggml_backend_tensor_set(g.cosIn, d->cosf, 0, static_cast<std::size_t>(d->head_dim) * d->seq * sizeof(float));
    if (g.sinIn != nullptr && g.sinIn->buffer != nullptr)
        ggml_backend_tensor_set(g.sinIn, d->sinf, 0, static_cast<std::size_t>(d->head_dim) * d->seq * sizeof(float));
}

// TS_WAN_DIT_TRACE=1: after compute, scan the graph nodes for the first one whose
// (possibly reused) buffer holds NaN and print its neighborhood. Debug aid only —
// gallocr reuses buffers, so downstream reads can alias, but the NaN onset node
// in execution order is reliable because NaN propagates forward.
void wan_dit_trace_taps(const WanDitGraph& g)
{
    const char* path = std::getenv("TS_WAN_DIT_TRACE");
    if (path == nullptr || g.taps.empty()) return;
    std::FILE* out = std::fopen(path, "a");
    if (out == nullptr) return;
    for (const auto& [name, t] : g.taps)
    {
        if (t->buffer == nullptr) { std::fprintf(out, "[wan-trace] %s: NO BUFFER\n", name.c_str()); continue; }
        const std::int64_t nel = ggml_nelements(t);
        std::vector<float> host(static_cast<std::size_t>(nel));
        ggml_backend_tensor_get(t, host.data(), 0, static_cast<std::size_t>(nel) * sizeof(float));
        double mn = 1e300, mx = -1e300, sum = 0; std::int64_t nan = 0;
        for (std::int64_t j = 0; j < nel; j++)
        {
            float v = host[static_cast<std::size_t>(j)];
            if (std::isnan(v) || std::isinf(v)) { nan++; continue; }
            mn = std::min<double>(mn, v); mx = std::max<double>(mx, v); sum += v;
        }
        std::fprintf(out, "[wan-trace] %-14s ne=[%lld,%lld] nan=%lld/%lld min=%g max=%g mean=%g\n",
                     name.c_str(), (long long)t->ne[0], (long long)t->ne[1],
                     (long long)nan, (long long)nel, mn, mx, sum / std::max<std::int64_t>(1, nel - nan));
    }
    std::fclose(out);
}

int wan_dit_run_persist(WanDitPersist* e, const TSGgmlWanDitDesc* d)
{
    WanDitGraph& g = e->g;
    if (e->galloc != nullptr && !ggml_gallocr_alloc_graph(e->galloc, g.graph))
    {
        e->reset();
        set_last_error("WanDitForward: gallocr realloc failed.");
        return 0;
    }
    // Re-upload the input-slot leaves living in the gallocr buffer: the buffer is not
    // cleared here because this graph never reads an intermediate before writing it,
    // but input-slot weights must be refreshed after a re-plan.
    host_read_barrier();
    if (g.outT != nullptr && g.outT->buffer != nullptr)
    {
        ggml_backend_buffer_t gb = g.outT->buffer;
        for (auto& u : g.uploads)
            if (u.t->buffer == gb) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
    }
    wan_dit_upload_inputs(g, d);
    if (tsg::compute_graph(g_backend, g.graph) != GGML_STATUS_SUCCESS)
    { set_last_error("WanDitForward: graph compute failed."); return 0; }
    tsg::sync_backend(g_backend);
    ggml_backend_tensor_get(g.outT, d->out, 0, static_cast<std::size_t>(d->head.ne1) * d->seq * sizeof(float));
    clear_last_error();
    return 1;
}

WanDitPersist* wan_dit_build_persist(const TSGgmlWanDitDesc* d)
{
    const int nl = d->num_layers;
    const std::size_t nodes = wan_dit_nodes(nl, d->ff, d->seq);
    const std::size_t meta = ggml_tensor_overhead() * (nodes + 1024)
                             + ggml_graph_overhead_custom(nodes, false) + (8u << 20);
    ggml_init_params ip{ meta, nullptr, true };
    ggml_context* ctx = ggml_init(ip);
    if (ctx == nullptr) return nullptr;

    WanDitGraph g;
    if (!wan_dit_build_graph(ctx, d, g)) { ggml_free(ctx); return nullptr; }

    // TS_WAN_DIT_CAPTURE=abandon: fault injection for wan_dit_resident_cache_test.
    // Drops the graph at the same point the VRAM spill guard below does — after
    // every weight has been bound (and, since the fix, filled) — so the test can
    // reproduce the 16 GB/27404-token path deterministically without needing a
    // card that is actually out of memory. Read per call, not cached in a static,
    // so a test can toggle it between phases.
    {
        const char* e = std::getenv("TS_WAN_DIT_CAPTURE");
        if (e != nullptr && std::strcmp(e, "abandon") == 0) { ggml_free(ctx); return nullptr; }
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
    if (galloc == nullptr) { ggml_free(ctx); return nullptr; }
    if (!ggml_gallocr_alloc_graph(galloc, g.graph)) { ggml_gallocr_free(galloc); ggml_free(ctx); return nullptr; }

    // VRAM spill guard: refuse the persistent graph when its allocation left the
    // device nearly full (the per-call path runs instead).
    {
        ggml_backend_dev_t mdev = ggml_backend_get_device(g_backend);
        std::size_t freeb = 0, totalb = 0;
        if (mdev) ggml_backend_dev_memory(mdev, &freeb, &totalb);
        if (totalb > 0 && freeb < static_cast<std::size_t>(384) * 1024 * 1024)
        {
            std::fprintf(stderr,
                "[wan] dit capture: free VRAM %zu MiB after alloc -> non-persistent fallback for %d tokens "
                "(this shape won't be retried; fewer frames or a smaller frame area keeps the graph resident)\n",
                freeb >> 20, d->seq);
            wan_dit_mark_too_big(d);   // skip the (multi-GB, discarded) capture attempt from now on
            ggml_gallocr_free(galloc); ggml_free(ctx); return nullptr;
        }
    }

    host_read_barrier();
    // Debug layer truncation can leave declared prelude weights unreachable
    // from the final graph. Gallocr intentionally does not allocate those
    // unused leaves, so do not try to upload them.
    for (auto& u : g.uploads)
        if (u.t->buffer != nullptr) ggml_backend_tensor_set(u.t, u.d, 0, u.b);

    WanDitPersist* e = nullptr;
    for (auto& c : g_wanDit) if (!c.valid) { e = &c; break; }
    if (e == nullptr) { e = &g_wanDit[g_wanDitRR]; g_wanDitRR = (g_wanDitRR + 1) % kWanDitCacheMax; e->reset(); }
    e->ctx = ctx; e->galloc = galloc; e->g = g;
    e->seq = d->seq; e->cl = d->ctx_len; e->nl = nl; e->seq0 = d->seq0; e->wkey = d->blocks[0].sq.w;
    e->valid = true;
    return e;
}

// ---------------------------------------------------------------------------
// Wan VAE decoder graph
// ---------------------------------------------------------------------------

// Feature layout throughout the decoder: [W, H, C, T] (frame-contiguous slabs so
// temporal windows are plain ne3 views and 2D convs batch over frames).
struct WanVaeBuild
{
    ggml_context* ctx = nullptr;
    WanBind* wb = nullptr;
    // Causal feature caches (CACHE_T = 2 trailing input frames per conv), carried
    // across the temporal chunks inside the one graph.
    std::vector<ggml_tensor*> cache;
    int cursor = 0;
    int chunk = 0;
    long long gemmMax = 384LL << 20;   // im2col scratch budget (see wan_vae_gemm_budget)
};

// im2col scratch budget for wan_vae_conv2d. TS_WAN_VAE_GEMM_MAX_MB pins it
// (0 = direct conv); otherwise it adapts to the device's free memory. Both
// extremes were measured slow at 720p on a 16 GB card: the fixed 384 MB
// default splits every full-res conv into ~11 bands whose strided CONT +
// CONCAT-chain copies through ggml's generic kernels were 72% of decode time,
// while unbanded (4.1 GB im2col) oversubscribed WDDM residency and ran
// IM2COL/MUL_MAT at PCIe speed (~6 GB/s). free/8 keeps the scratch a small
// slice of what must stay resident (activation planes, cross-chunk caches,
// the growing output) — a few bands on 16 GB, unbanded on large cards — and
// the 384 MB floor preserves the old behavior under memory pressure.
// True when the process-global GGML backend is ggml-metal ("MTL0", "MTL1", ...).
static bool wan_backend_is_metal()
{
    const char* name = g_backend != nullptr ? ggml_backend_name(g_backend) : nullptr;
    return name != nullptr && std::strncmp(name, "MTL", 3) == 0;
}

// Whether the Metal 4 tensor API is active for this process
// (ggml_ops_metal_flash_pad.h). Used to decide the VAE conv strategy; a rare
// false positive only costs speed, never correctness.
static bool wan_metal_tensor_api_likely()
{
    return tsg_metal_tensor_api_likely(g_backend);
}

static long long wan_vae_gemm_budget()
{
    const char* e = std::getenv("TS_WAN_VAE_GEMM_MAX_MB");
    if (e != nullptr) return std::strtoll(e, nullptr, 10) * 1024 * 1024;

    // ggml-metal's Metal 4 tensor-API mul_mm misreads operands inside the VAE
    // decode graph on M5 (first pass, buffer-layout dependent). When the tensor
    // API is active, route the VAE convs through ggml_conv_2d_direct (budget 0):
    // slower than im2col+GEMM, but correct.
    //
    // Do NOT "verify this is fixed" with an isolated decode. Tried 2026-08-17:
    // WanVideoBench decoded synthetic latents at five shapes — including the
    // 32x32 layout recorded as the original all-NaN repro — and tensor-API vs
    // non-tensor output agreed to 91-93 dB PSNR with no NaN anywhere. It looked
    // conclusively fixed. The very next full 1088x832x121f generation, with the
    // DiT loaded and released before the decode, rendered 121 uniformly BLACK
    // frames. The defect follows the allocation history, so only an end-to-end
    // video is evidence.
    if (wan_metal_tensor_api_likely())
        return 0;
    // Vulkan stays at the banded floor: its drivers reject the multi-GB single
    // gallocr arena an unbanded full-plane im2col produces (maxMemoryAllocationSize
    // is commonly 4 GB or less). Metal and CUDA size the budget from free device
    // memory — on Metal that measured 56.4s -> 49.7s for a 1088x832x9f decode,
    // because it turns many small banded GEMMs into few large ones.
    const char* name = g_backend != nullptr ? ggml_backend_name(g_backend) : nullptr;
    const bool isCuda = name != nullptr && std::strncmp(name, "CUDA", 4) == 0;
    const bool isMetal = name != nullptr && std::strncmp(name, "MTL", 3) == 0;
    if (!isCuda && !isMetal)
        return 384LL << 20;

    std::size_t freeB = 0, totalB = 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
    if (dev != nullptr) ggml_backend_dev_memory(dev, &freeB, &totalB);
    long long budget = static_cast<long long>(freeB / 8);
    const long long lo = 384LL << 20;
    long long hi = 8LL << 30;

    // CUDA: growing the scratch past ~384 MB is a LOSS, not a win, and a large one.
    // The im2col scratch is only one term in the graph's peak; the activation planes,
    // the cross-chunk causal caches and the accumulating output are the rest, and on
    // a discrete card the whole thing has to sit in real VRAM. Sizing the scratch
    // from free memory just spends the headroom the rest of the graph needs, and on
    // WDDM the overflow becomes host paging at PCIe speed.
    //
    // Measured on an RTX 3080 Laptop (16 GB), 1088x832x17f isolated decode
    // (benchmarks/WanVideoBench vae-decode 5 52 68 cuda), at the original code:
    //     free/8 (~1.9 GB) 122.3 s | 1536 MB 91.5 s | 768 MB 89.3 s
    //     384 MB  55.4 s  <-- optimum | 256 MB 61.2 s | 128 MB 73.9 s | 64 MB 79.3 s
    // i.e. the old free/8 sizing cost 2.2x. The curve rises on BOTH sides: below
    // ~256 MB the band count grows and the per-band CONT + join starts to dominate.
    // With the shared-im2col taps path and even banding the plateau is flatter and
    // lower — 384 MB 38.1 s | 448 MB 38.6 s | 512 MB 40.2 s | 640 MB 51.7 s — so
    // 384 MB stays the pick, now with the cliff a comfortable distance away.
    //
    // Metal keeps the free-memory sizing: it is unified memory with no PCIe cliff,
    // and it measured 56.4 s -> 49.7 s for a 1088x832x9f decode by turning many
    // small banded GEMMs into few large ones.
    if (isCuda)
        hi = 384LL << 20;

    if (budget > hi) budget = hi;
    if (budget < lo) budget = lo;
    return budget;
}

ggml_tensor* wan_vae_conv_w(WanVaeBuild& b, void* tap, const TSGWanVaeConv& c)
{
    const ggml_type tt = c.tap_type == 1 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor* t = ggml_new_tensor_4d(b.ctx, tt, c.k, c.k, c.ic, c.oc);
    b.wb->bind(t, tap, static_cast<std::size_t>(c.k) * c.k * c.ic * c.oc * ggml_type_size(tt));
    return t;
}

// One 2D convolution of the decoder. ggml_conv_2d_direct needs no scratch but its
// CUDA kernel is a naive direct convolution — the whole-video decode measured
// ~50x slower than the im2col+GEMM path at 480p. So convs run as im2col (F16) +
// mul_mat, with the LARGE convs split into horizontal bands so the materialized
// im2col stays bounded (TS_WAN_VAE_GEMM_MAX_MB, default 384): the full-resolution
// 3x3 convs would otherwise materialize ~2.8 GB each, and together with the
// cross-chunk feature caches that filled a 16 GB card into WDDM paging (decode
// 51 s -> 180 s depending on what else held VRAM). Banding is exact — each band
// gets its vertical context rows from a single pre-padded copy of the input.
// ggml_pad_ext with LEADING padding, built so it runs on every backend.
//
// ggml-metal implements only trailing padding: ggml_metal_device_supports_op
// returns false for GGML_OP_PAD whenever any of op_params 0/2/4/6 (the lo pads)
// is non-zero, and ggml_metal_op_encode_impl then ABORTS THE PROCESS with
// "unsupported op 'PAD'". The Wan VAE leans on leading pads throughout - the
// causal 3D convolutions zero-pad the FRONT of the temporal axis, and the banded
// 2D conv pre-pads the top row band - so every Wan video generation died in
// TSGgml_WanVaeDecode on Metal (SIGABRT) after a perfectly good denoise loop.
//
// Trailing pads still go through ggml_pad_ext. Each leading pad is prepended as a
// zeroed strip of the tensor itself (view -> cont -> scale 0 -> concat), which
// uses only ops every backend implements. The strips are 1-2 slices wide, so the
// extra work is negligible next to the convolutions around them.
static ggml_tensor* wan_pad_ext_compat(ggml_context* ctx, ggml_tensor* x,
                                       int lp0, int rp0, int lp1, int rp1,
                                       int lp2, int rp2, int lp3, int rp3)
{
    const int lead[4] = { lp0, lp1, lp2, lp3 };
    if (lead[0] <= 0 && lead[1] <= 0 && lead[2] <= 0 && lead[3] <= 0)
        return ggml_pad_ext(ctx, x, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3);

    ggml_tensor* t = (rp0 > 0 || rp1 > 0 || rp2 > 0 || rp3 > 0)
        ? ggml_pad_ext(ctx, x, 0, rp0, 0, rp1, 0, rp2, 0, rp3)
        : x;

    for (int d = 0; d < 4; d++)
    {
        int remaining = lead[d];
        while (remaining > 0)
        {
            // A prefix along any single axis is a valid strided view of t.
            const std::int64_t take = std::min<std::int64_t>(remaining, t->ne[d]);
            std::int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
            ne[d] = take;
            ggml_tensor* strip = ggml_cont(ctx, ggml_view_4d(ctx, t, ne[0], ne[1], ne[2], ne[3],
                                                             t->nb[1], t->nb[2], t->nb[3], 0));
            t = ggml_concat(ctx, ggml_scale(ctx, strip, 0.0f), t, d);
            remaining -= static_cast<int>(take);
        }
    }
    return t;
}

ggml_tensor* wan_vae_conv2d(WanVaeBuild& b, ggml_tensor* wt, ggml_tensor* x, int pad, int stride = 1)
{
    const long long gemmMax = b.gemmMax;
    ggml_context* ctx = b.ctx;
    // MPS executes the convolution whole, so emit the un-lowered node: this also
    // skips the horizontal banding and its leading-pad shim entirely.
    if (tsg::fast_conv_enabled())
        return ggml_conv_2d_direct(ctx, wt, x, stride, stride, pad, pad, 1, 1);
    if (gemmMax <= 0)
        return ggml_conv_2d_direct(ctx, wt, x, stride, stride, pad, pad, 1, 1);

    const long long KW = wt->ne[0], KH = wt->ne[1], IC = wt->ne[2];
    const long long OW = (x->ne[0] + 2 * pad - KW) / stride + 1;
    const long long OH = (x->ne[1] + 2 * pad - KH) / stride + 1;
    const long long T = x->ne[3];
    const long long rowBytes = KW * KH * IC * OW * T * 2;   // im2col bytes per output row
    ggml_tensor* wf16 = wt->type == GGML_TYPE_F16 ? wt : ggml_cast(ctx, wt, GGML_TYPE_F16);
    // Defensive on Metal: give a legacy in-graph weight cast a dedicated
    // (never gallocr-reused) slot. Weights marshalled as F16 (tap_type == 1)
    // skip the cast entirely, and tensor-API devices take the direct-conv
    // path (wan_vae_gemm_budget) anyway, so this only matters for F32-tap
    // callers on the GEMM path.
    if (wf16 != wt && wan_backend_is_metal())
        ggml_set_output(wf16);

    if (rowBytes * OH <= gemmMax)
        return ggml_conv_2d(ctx, wf16, x, stride, stride, pad, pad, 1, 1);

    // Banded path: pre-pad vertically once, then convolve horizontal bands whose
    // views carry their own context rows (horizontal padding stays in the conv).
    // Even bands (see wan_vae_conv2d_taps).
    const long long fitRows = std::max<long long>(1, gemmMax / std::max<long long>(1, rowBytes));
    const long long nBands = (OH + fitRows - 1) / fitRows;
    const long long bandRows = (OH + nBands - 1) / nBands;
    ggml_tensor* xp = pad > 0 ? wan_pad_ext_compat(ctx, x, 0, 0, pad, pad, 0, 0, 0, 0) : x;
    std::vector<ggml_tensor*> bands;
    bands.reserve(static_cast<std::size_t>((OH + bandRows - 1) / bandRows));
    for (long long y = 0; y < OH; y += bandRows)
    {
        const long long rows = std::min(bandRows, OH - y);
        const long long inRows = (rows - 1) * stride + KH;
        ggml_tensor* band = ggml_view_4d(ctx, xp, xp->ne[0], inRows, xp->ne[2], xp->ne[3],
                                         xp->nb[1], xp->nb[2], xp->nb[3], y * stride * xp->nb[1]);
        // CUDA CONV_2D requires contiguous input; the row-band view is strided.
        band = ggml_cont(ctx, band);
        bands.push_back(ggml_conv_2d(ctx, wf16, band, stride, stride, pad, 0, 1, 1));
    }
    return wan_concat_all(ctx, bands, 1);
}

// The kd temporal taps of one causal conv, sharing ONE im2col per spatial band.
//
// A causal conv3d is a sum of kd 2D convolutions, tap j reading input frames
// [j, j+T) of the front-padded input. Run tap-by-tap through ggml_conv_2d — as this
// did before — each tap re-lowers its own overlapping frame window, so the same
// pixels are expanded into an im2col matrix kd times over. im2col is 44% of a
// full-resolution decode (the single biggest cost in the graph, KW*KH = 9x the
// input), so that repetition is worth removing: ONE im2col over all T+kd-1 frames
// serves every tap, because frames are im2col's batch axis (ne[3]) and a frame
// range of the result is a plain contiguous view. 3 taps over T=4 frames — the
// widest, most expensive scales — go from 12 frame-lowerings to 6.
//
// Two smaller savings ride along: the per-band ggml_cont of the strided row view
// happens once instead of kd times, and ggml_conv_2d's trailing
// cont(permute [OW,OH,N,OC] -> [OW,OH,OC,N]) — another full copy of the output —
// is deferred until after the taps are summed, so it runs once rather than kd
// times. Everything here is data movement; the GEMMs and their accumulation order
// are unchanged, so the result is bit-identical.
ggml_tensor* wan_vae_conv2d_taps(WanVaeBuild& b, ggml_tensor** wts, int nTaps,
                                 ggml_tensor* xin, std::int64_t T, int pad)
{
    ggml_context* ctx = b.ctx;
    ggml_tensor* w0 = wts[0];
    const std::int64_t KW = w0->ne[0], KH = w0->ne[1], IC = w0->ne[2], OC = w0->ne[3];
    const std::int64_t OW = xin->ne[0] + 2 * pad - KW + 1;
    const std::int64_t OH = xin->ne[1] + 2 * pad - KH + 1;
    const std::int64_t NT = xin->ne[3];               // T + nTaps - 1

    std::vector<ggml_tensor*> wf16(static_cast<std::size_t>(nTaps));
    for (int j = 0; j < nTaps; j++)
        wf16[static_cast<std::size_t>(j)] =
            wts[j]->type == GGML_TYPE_F16 ? wts[j] : ggml_cast(ctx, wts[j], GGML_TYPE_F16);

    // One band's shared lowering + one GEMM per tap, returned in ggml_conv_2d's
    // [OW, rows, OC, T] layout.
    //
    // The permute+cont stays PER BAND on purpose. Deferring it to the joined
    // full-height tensor would run it once instead of once per band, but it then
    // needs the pre-permute and post-permute full-resolution tensors alive at the
    // same time — a whole extra copy of the widest activation (~2 GB at 1088x832).
    // Measured: that version cut profiled per-node time 17% and still ran 82 s vs
    // 49 s wall, because the extra residency pushed the decode into WDDM paging.
    // Per band it is still kd times fewer permutes than the old tap-by-tap loop.
    auto band_taps = [&](ggml_tensor* src, std::int64_t rows, int padV) -> ggml_tensor*
    {
        ggml_tensor* im = ggml_im2col(ctx, wf16[0], src, 1, 1, pad, padV, 1, 1, true, GGML_TYPE_F16);
        ggml_tensor* acc = nullptr;
        for (int j = 0; j < nTaps; j++)
        {
            // Frames [j, j+T) of [IC*KH*KW, OW, rows, NT]. ne[0..2] are unchanged,
            // so the view keeps the parent's strides and is itself contiguous.
            ggml_tensor* imv = (nTaps == 1)
                ? im
                : ggml_view_4d(ctx, im, im->ne[0], im->ne[1], im->ne[2], T,
                               im->nb[1], im->nb[2], im->nb[3],
                               static_cast<std::size_t>(j) * im->nb[3]);
            ggml_tensor* r = ggml_mul_mat(
                ctx,
                ggml_reshape_2d(ctx, imv, imv->ne[0], imv->ne[1] * imv->ne[2] * imv->ne[3]),
                ggml_reshape_2d(ctx, wf16[static_cast<std::size_t>(j)], KW * KH * IC, OC));
            r = ggml_reshape_4d(ctx, r, OW, rows, T, OC);
            acc = acc == nullptr ? r : ggml_add(ctx, acc, r);
        }
        return ggml_cont(ctx, ggml_permute(ctx, acc, 0, 1, 3, 2));   // -> [OW, rows, OC, T]
    };

    const long long rowBytes = KW * KH * IC * OW * NT * 2;   // shared im2col bytes per output row
    ggml_tensor* joined = nullptr;
    if (rowBytes * OH <= b.gemmMax)
    {
        joined = band_taps(xin, OH, pad);
    }
    else
    {
        // Even bands, not a fixed walk with a runt at the end: every band re-copies
        // KH-1 context rows, so the waste scales with the band COUNT, and an uneven
        // split pays for a band that does almost no useful work. Picking the count
        // from the budget and then dividing OH evenly also makes the cost a smooth
        // function of the budget instead of a sawtooth (measured 36 s at 448 MB vs
        // 51 s at 576 MB on the same decode, purely from where the split landed).
        // Same idea as WanVaeBase.PlanBands for the outer spatial tiling.
        const long long fit = std::max<long long>(1, b.gemmMax / std::max<long long>(1, rowBytes));
        const long long nBands = (OH + fit - 1) / fit;
        const long long bandRows = (OH + nBands - 1) / nBands;
        ggml_tensor* xp = pad > 0 ? wan_pad_ext_compat(ctx, xin, 0, 0, pad, pad, 0, 0, 0, 0) : xin;
        std::vector<ggml_tensor*> bands;
        bands.reserve(static_cast<std::size_t>((OH + bandRows - 1) / bandRows));
        for (std::int64_t y = 0; y < OH; y += bandRows)
        {
            const std::int64_t rows = std::min<std::int64_t>(bandRows, OH - y);
            ggml_tensor* band = ggml_view_4d(ctx, xp, xp->ne[0], rows + KH - 1, xp->ne[2], xp->ne[3],
                                             xp->nb[1], xp->nb[2], xp->nb[3],
                                             static_cast<std::size_t>(y) * xp->nb[1]);
            bands.push_back(band_taps(ggml_cont(ctx, band), rows, 0));
        }
        joined = wan_concat_all(ctx, bands, 1);
    }
    return joined;   // already [OW, OH, OC, T]
}

// Causal conv3d over x [W,H,IC,T] -> [W,H,OC,T]. Temporal context comes from the
// per-conv cache (previous chunk's trailing frames) or zero front-padding on the
// first chunk; the cache is updated to this chunk's trailing input frames.
ggml_tensor* wan_vae_causal_conv(WanVaeBuild& b, const TSGWanVaeConv& c, ggml_tensor* x)
{
    ggml_context* ctx = b.ctx;
    const std::int64_t T = x->ne[3];
    const int pad = c.k == 3 ? 1 : 0;

    ggml_tensor* xin = x;
    if (c.kd == 3)
    {
        const int idx = b.cursor++;
        // Caches live across every remaining chunk of the graph, so they dominate
        // the cross-chunk VRAM liveness; they are stored F16 (visually lossless
        // temporal context, half the resident bytes) and cast back on use.
        ggml_tensor* prev = b.cache[idx] != nullptr ? ggml_cast(ctx, b.cache[idx], GGML_TYPE_F32) : nullptr;
        if (prev == nullptr)
        {
            // first chunk: 2 zero frames in front (causal zero padding)
            xin = wan_pad_ext_compat(ctx, x, 0, 0, 0, 0, 0, 0, 2, 0);
        }
        else
        {
            if (prev->ne[3] < 2)
            {
                // previous chunk had a single frame: zero-pad its front once
                prev = wan_pad_ext_compat(ctx, prev, 0, 0, 0, 0, 0, 0, 2 - static_cast<int>(prev->ne[3]), 0);
            }
            xin = ggml_concat(ctx, prev, x, 3);
        }
        // new cache = trailing (up to 2) input frames of x, extended with the old
        // cache's last frame when x is a single frame (sd.cpp CACHE_T logic).
        ggml_tensor* nc;
        if (T >= 2)
            nc = ggml_view_4d(ctx, x, x->ne[0], x->ne[1], x->ne[2], 2,
                              x->nb[1], x->nb[2], x->nb[3], (T - 2) * x->nb[3]);
        else if (prev != nullptr)
        {
            ggml_tensor* lastPrev = ggml_view_4d(ctx, prev,
                prev->ne[0], prev->ne[1], prev->ne[2], 1,
                prev->nb[1], prev->nb[2], prev->nb[3],
                (prev->ne[3] - 1) * prev->nb[3]);
            nc = ggml_concat(ctx, lastPrev, x, 3);
        }
        else
            nc = x;
        b.cache[idx] = ggml_cast(ctx, nc, GGML_TYPE_F16);
    }

    // Sum of per-temporal-tap 2D convolutions; output frame t reads input frames
    // t .. t+kd-1 of the assembled (front-padded) input.
    void* taps[3] = { c.tap0, c.tap1, c.tap2 };
    ggml_tensor* acc = nullptr;
    // The shared-im2col path needs ggml's im2col lowering; the MPS/direct-conv
    // routes emit an un-lowered CONV_2D node instead, so they keep the tap loop.
    // TS_WAN_VAE_TAP_SHARE=0 forces the tap loop for A/B testing: with a budget
    // large enough that neither path bands, the two are bit-identical (the sharing
    // only removes duplicate lowering; the GEMMs and their order are unchanged).
    static const bool tapShare = []{
        const char* e = std::getenv("TS_WAN_VAE_TAP_SHARE"); return e == nullptr || e[0] != '0'; }();
    if (tapShare && c.kd > 1 && b.gemmMax > 0 && !tsg::fast_conv_enabled())
    {
        ggml_tensor* wt[3] = { nullptr, nullptr, nullptr };
        for (int j = 0; j < c.kd; j++) wt[j] = wan_vae_conv_w(b, taps[j], c);
        acc = wan_vae_conv2d_taps(b, wt, c.kd, xin, T, pad);
    }
    else
    for (int j = 0; j < c.kd; j++)
    {
        ggml_tensor* wt = wan_vae_conv_w(b, taps[j], c);
        ggml_tensor* win = xin;
        if (c.kd > 1)
            win = ggml_view_4d(b.ctx, xin, xin->ne[0], xin->ne[1], xin->ne[2], T,
                               xin->nb[1], xin->nb[2], xin->nb[3], j * xin->nb[3]);
        ggml_tensor* y = wan_vae_conv2d(b, wt, win, pad);
        acc = acc == nullptr ? y : ggml_add(ctx, acc, y);
    }
    if (c.bias != nullptr)
    {
        ggml_tensor* bt = b.wb->f32v(c.bias, c.oc);
        acc = ggml_add(ctx, acc, ggml_reshape_4d(ctx, bt, 1, 1, c.oc, 1));
    }
    return acc;
}

// Channel RMS norm (RMS_norm in wan_vae.hpp): normalize over C at every (w,h,t).
ggml_tensor* wan_vae_norm(WanVaeBuild& b, const TSGWanVaeNorm& n, ggml_tensor* x)
{
    ggml_context* ctx = b.ctx;
    ggml_tensor* g = b.wb->f32v(n.gamma, n.c);
    ggml_tensor* h = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));   // [C, W, H, T]
    h = ggml_mul(ctx, ggml_rms_norm(ctx, h, 1e-12f), g);
    return ggml_cont(ctx, ggml_permute(ctx, h, 2, 0, 1, 3));             // [W, H, C, T]
}

ggml_tensor* wan_vae_res_block(WanVaeBuild& b, const TSGWanVaeResBlockW& r, ggml_tensor* x)
{
    ggml_context* ctx = b.ctx;
    ggml_tensor* h = x;
    if (r.shortcut.tap0 != nullptr)
        h = wan_vae_causal_conv(b, r.shortcut, x);
    ggml_tensor* y = wan_vae_norm(b, r.n0, x);
    y = ggml_silu(ctx, y);
    y = wan_vae_causal_conv(b, r.c2, y);
    y = wan_vae_norm(b, r.n3, y);
    y = ggml_silu(ctx, y);
    y = wan_vae_causal_conv(b, r.c6, y);
    return ggml_add(ctx, y, h);
}

// Spatial single-head self-attention over each frame (mid block; T == 1 for the
// wan2.1 decoder chunks).
ggml_tensor* wan_vae_attn(WanVaeBuild& b, const TSGWanVaeAttnW& a, ggml_tensor* x)
{
    ggml_context* ctx = b.ctx;
    const std::int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], T = x->ne[3];
    const std::int64_t WH = W * H;

    ggml_tensor* qkvW = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, 3 * C);
    b.wb->bind(qkvW, a.qkv_w, static_cast<std::size_t>(C) * 3 * C * sizeof(float));
    ggml_tensor* qkvB = b.wb->f32v(a.qkv_b, 3 * C);
    ggml_tensor* projW = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, C);
    b.wb->bind(projW, a.proj_w, static_cast<std::size_t>(C) * C * sizeof(float));
    ggml_tensor* projB = b.wb->f32v(a.proj_b, C);

    // [W,H,C,T] -> [C, WH, T]
    ggml_tensor* xt = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));  // [C, W, H, T]
    xt = ggml_reshape_3d(ctx, xt, C, WH, T);
    ggml_tensor* identity = xt;

    ggml_tensor* n = ggml_mul(ctx, ggml_rms_norm(ctx, xt, 1e-12f),
                              b.wb->f32v(a.norm.gamma, C));
    ggml_tensor* qkv = ggml_add(ctx, ggml_mul_mat(ctx, qkvW, n),
                                ggml_reshape_3d(ctx, qkvB, 3 * C, 1, 1)); // [3C, WH, T]

    // q/k/v are ne0 slices of the fused [3C, WH, T] projection.
    ggml_tensor* q = ggml_cont(ctx, ggml_view_3d(ctx, qkv, C, WH, T, qkv->nb[1], qkv->nb[2], 0));
    ggml_tensor* k = ggml_cont(ctx, ggml_view_3d(ctx, qkv, C, WH, T, qkv->nb[1], qkv->nb[2], C * sizeof(float)));
    ggml_tensor* v = ggml_cont(ctx, ggml_view_3d(ctx, qkv, C, WH, T, qkv->nb[1], qkv->nb[2], 2 * C * sizeof(float)));

    ggml_tensor* kq = ggml_mul_mat(ctx, k, q);                            // [WH_k, WH_q, T]
    kq = ggml_scale(ctx, kq, 1.0f / std::sqrt(static_cast<float>(C)));
    kq = ggml_soft_max(ctx, kq);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));   // [WH, C, T]
    ggml_tensor* kqv = ggml_mul_mat(ctx, vt, kq);                         // [C, WH_q, T]
    ggml_tensor* o = ggml_add(ctx, ggml_mul_mat(ctx, projW, kqv),
                              ggml_reshape_3d(ctx, projB, C, 1, 1));
    o = ggml_add(ctx, o, identity);

    // back to [W,H,C,T]
    o = ggml_reshape_4d(ctx, o, C, W, H, T);
    return ggml_cont(ctx, ggml_permute(ctx, o, 2, 0, 1, 3));
}

// Resample: optional causal temporal doubling (time_conv, chunks >= 1) then
// nearest x2 spatial upsample + 3x3 conv.
ggml_tensor* wan_vae_upsample(WanVaeBuild& b, const TSGWanVaeUpsampleW& u, ggml_tensor* x)
{
    ggml_context* ctx = b.ctx;

    if (u.time_conv.tap0 != nullptr)
    {
        const int idx = b.cursor++;
        if (b.chunk == 0)
        {
            // first chunk passes through untouched; the cache stays empty and the
            // NEXT chunk zero-pads (sd.cpp Resample upsample3d, chunk_idx == 0).
        }
        else
        {
            const std::int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], T = x->ne[3];
            ggml_tensor* prev = b.cache[idx] != nullptr ? ggml_cast(ctx, b.cache[idx], GGML_TYPE_F32) : nullptr;
            ggml_tensor* xin;
            if (prev == nullptr)
                xin = wan_pad_ext_compat(ctx, x, 0, 0, 0, 0, 0, 0, 2, 0);
            else
                xin = ggml_concat(ctx, prev, x, 3);
            // cache = trailing 2 frames of x (zero-front-padded when T == 1 on the
            // first cached chunk; extended with the previous cache otherwise)
            ggml_tensor* nc;
            if (T >= 2)
                nc = ggml_view_4d(ctx, x, W, H, C, 2, x->nb[1], x->nb[2], x->nb[3], (T - 2) * x->nb[3]);
            else if (prev != nullptr)
            {
                ggml_tensor* lastPrev = ggml_view_4d(ctx, prev, W, H, C, 1,
                    prev->nb[1], prev->nb[2], prev->nb[3], (prev->ne[3] - 1) * prev->nb[3]);
                nc = ggml_concat(ctx, lastPrev, x, 3);
            }
            else
                nc = wan_pad_ext_compat(ctx, x, 0, 0, 0, 0, 0, 0, 1, 0);
            b.cache[idx] = ggml_cast(ctx, nc, GGML_TYPE_F16);

            // temporal conv (kd=3, 1x1 spatial): per-tap 1x1 conv over the window
            void* taps[3] = { u.time_conv.tap0, u.time_conv.tap1, u.time_conv.tap2 };
            ggml_tensor* acc = nullptr;
            for (int j = 0; j < 3; j++)
            {
                ggml_tensor* wt = wan_vae_conv_w(b, taps[j], u.time_conv);
                ggml_tensor* win = ggml_view_4d(ctx, xin, xin->ne[0], xin->ne[1], xin->ne[2], T,
                                                xin->nb[1], xin->nb[2], xin->nb[3], j * xin->nb[3]);
                ggml_tensor* y = wan_vae_conv2d(b, wt, win, 0);
                acc = acc == nullptr ? y : ggml_add(ctx, acc, y);
            }
            if (u.time_conv.bias != nullptr)
            {
                ggml_tensor* bt = b.wb->f32v(u.time_conv.bias, u.time_conv.oc);
                acc = ggml_add(ctx, acc, ggml_reshape_4d(ctx, bt, 1, 1, u.time_conv.oc, 1));
            }
            // [W,H,2C,T] -> interleave the channel halves as frame pairs -> [W,H,C,2T].
            // The conv output channel index is half*C + c (torch view(2, c, ...)), so a
            // contiguous reinterpret already yields frame index half + 2*t.
            x = ggml_reshape_4d(ctx, acc, W, H, C, 2 * T);
        }
        // Advance chunk-0 state: leave cache[idx] == nullptr so chunk 1 zero-pads.
    }

    // nearest x2 spatial upsample + 3x3 conv (pad 1)
    x = ggml_upscale(ctx, x, 2, GGML_SCALE_MODE_NEAREST);
    ggml_tensor* wt = wan_vae_conv_w(b, u.sconv.tap0, u.sconv);
    ggml_tensor* y = wan_vae_conv2d(b, wt, x, 1);
    if (u.sconv.bias != nullptr)
    {
        ggml_tensor* bt = b.wb->f32v(u.sconv.bias, u.sconv.oc);
        y = ggml_add(ctx, y, ggml_reshape_4d(ctx, bt, 1, 1, u.sconv.oc, 1));
    }
    return y;
}

// DupUp3D (wan2.2 Up_ResidualBlock shortcut): channel-grouped nearest upsample.
// diffusers semantics: out[co, t*ft+ift, y*fs+ys, x*fs+xs] = in[c2 / repeats]
// with c2 = co*(ft*fs*fs) + ift*fs*fs + ys*fs + xs. The wan2.2 decoder uses only
// (ft=2, fs=2, in==out) — exact nearest x2 in T/H/W — and (ft=1, fs=2, in==2*out)
// — output row parity selects the input channel pair half, columns duplicate.
ggml_tensor* wan_vae_dup_up(WanVaeBuild& b, ggml_tensor* x, int ft, int fs, bool first_chunk)
{
    ggml_context* ctx = b.ctx;
    const std::int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], T = x->ne[3];
    (void)fs;   // both decoder configurations have fs == 2

    if (ft == 2)
    {
        // temporal duplicate [W,H,C,T] -> [W,H,C,2T] (frame t -> t*2, t*2+1) ...
        ggml_tensor* r = ggml_reshape_3d(ctx, x, W * H * C, 1, T);
        r = ggml_concat(ctx, r, r, 1);                            // [WHC, 2, T]
        x = ggml_reshape_4d(ctx, r, W, H, C, 2 * T);
        // ... then spatial nearest x2
        x = ggml_interpolate(ctx, x, W * 2, H * 2, C, 2 * T, GGML_SCALE_MODE_NEAREST);
        if (first_chunk)
        {
            // causal first chunk keeps only the last of the ft duplicated frames
            x = ggml_view_4d(ctx, x, x->ne[0], x->ne[1], x->ne[2], 2 * T - 1,
                             x->nb[1], x->nb[2], x->nb[3], x->nb[3]);
        }
        return x;
    }

    // ft == 1, in == 2*out: out[co, t, 2y+ys, 2x+xs] = in[2*co + ys, t, y, x]
    const std::int64_t Co = C / 2;
    ggml_tensor* r = ggml_reshape_4d(ctx, x, W, H, 2, Co * T);       // isolate ys (channel low bit)
    r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));            // [W, 2, H, Co*T]
    r = ggml_reshape_4d(ctx, r, W, 2 * H, Co, T);                    // rows interleaved by parity
    return ggml_interpolate(ctx, r, W * 2, 2 * H, Co, T, GGML_SCALE_MODE_NEAREST);
}

// wan2.2 pixel unpatchify: [W, H, 4c, T] -> [2W, 2H, c, T] with channel index
// c*4 + xoff*2 + yoff (diffusers unpatchify order).
ggml_tensor* wan_vae_unpatchify2(WanVaeBuild& b, ggml_tensor* x)
{
    ggml_context* ctx = b.ctx;
    const std::int64_t W = x->ne[0], H = x->ne[1], C4 = x->ne[2], T = x->ne[3];
    const std::int64_t C = C4 / 4;
    ggml_tensor* t1 = ggml_reshape_4d(ctx, x, W, H, 2, 2 * C * T);   // isolate yoff
    t1 = ggml_cont(ctx, ggml_permute(ctx, t1, 0, 2, 1, 3));          // [W, 2, H, 2CT]
    t1 = ggml_reshape_4d(ctx, t1, W, 2 * H, 2, C * T);               // rows = h*2+yoff; isolate xoff
    t1 = ggml_cont(ctx, ggml_permute(ctx, t1, 1, 2, 0, 3));          // [2, W, 2H, CT]
    return ggml_reshape_4d(ctx, t1, 2 * W, 2 * H, C, T);             // cols = w*2+xoff
}

// One decoder pass over a single latent frame (chunk). Mirrors Decoder3d::forward:
// conv1, middle (res/attn/res), 4 scales x 3 res blocks with upsamples after
// scales 0/1 (temporal+spatial) and 2 (spatial), head. Wan 2.2 (version 2) adds
// the DupUp3D residual shortcut around each upsampled scale group and a final
// 2x2 pixel unpatchify (12 -> 3 channels).
ggml_tensor* wan_vae_decode_chunk(WanVaeBuild& b, const TSGgmlWanVaeDecodeDesc* d, ggml_tensor* z1)
{
    ggml_context* ctx = b.ctx;
    b.cursor = 0;
    const bool w22 = d->version == 2;

    ggml_tensor* x = wan_vae_causal_conv(b, d->conv1, z1);
    x = wan_vae_res_block(b, d->mid0, x);
    x = wan_vae_attn(b, d->mid1, x);
    x = wan_vae_res_block(b, d->mid2, x);

    for (int scale = 0; scale < 4; scale++)
    {
        ggml_tensor* xin = x;
        for (int r = 0; r < 3; r++)
            x = wan_vae_res_block(b, d->res[scale * 3 + r], x);
        if (scale < 3)
        {
            x = wan_vae_upsample(b, d->up[scale], x);
            if (w22)
            {
                // temperal_upsample = [true, true, false]
                const int ft = scale < 2 ? 2 : 1;
                ggml_tensor* sc = wan_vae_dup_up(b, xin, ft, 2, b.chunk == 0);
                x = ggml_add(ctx, x, sc);
            }
        }
    }

    x = wan_vae_norm(b, d->head_norm, x);
    x = ggml_silu(ctx, x);
    x = wan_vae_causal_conv(b, d->head_conv, x);
    if (w22 && d->patch == 2)
        x = wan_vae_unpatchify2(b, x);
    return x;   // [W*8*patch, H*8*patch, 3, t_out]
}

// Encoder Resample: right/bottom zero-pad + 3x3 stride-2 spatial conv, then
// (downsample3d) a stride-2 temporal conv whose 1-frame cache carries the causal
// context between chunks (diffusers WanResample downsample2d/3d).
ggml_tensor* wan_vae_enc_resample(WanVaeBuild& b, const TSGWanVaeDownW& u, ggml_tensor* x)
{
    ggml_context* ctx = b.ctx;

    ggml_tensor* xp = wan_pad_ext_compat(ctx, x, 0, 1, 0, 1, 0, 0, 0, 0);
    ggml_tensor* wt = wan_vae_conv_w(b, u.sconv.tap0, u.sconv);
    ggml_tensor* y = wan_vae_conv2d(b, wt, xp, 0, 2);
    if (u.sconv.bias != nullptr)
    {
        ggml_tensor* bt = b.wb->f32v(u.sconv.bias, u.sconv.oc);
        y = ggml_add(ctx, y, ggml_reshape_4d(ctx, bt, 1, 1, u.sconv.oc, 1));
    }

    if (u.tconv.tap0 == nullptr)
        return y;                                                   // downsample2d

    const int idx = b.cursor++;
    const std::int64_t T = y->ne[3];
    if (b.cache[idx] == nullptr)
    {
        // first chunk: store the (single) frame, temporal conv starts next chunk
        b.cache[idx] = ggml_cast(ctx, y, GGML_TYPE_F16);
        return y;
    }

    ggml_tensor* prev = ggml_cast(ctx, b.cache[idx], GGML_TYPE_F32);
    ggml_tensor* lastPrev = ggml_view_4d(ctx, prev, prev->ne[0], prev->ne[1], prev->ne[2], 1,
                                         prev->nb[1], prev->nb[2], prev->nb[3],
                                         (prev->ne[3] - 1) * prev->nb[3]);
    ggml_tensor* xin = ggml_concat(ctx, lastPrev, y, 3);            // [.., T + 1]
    b.cache[idx] = ggml_cast(ctx,
        ggml_view_4d(ctx, y, y->ne[0], y->ne[1], y->ne[2], 1,
                     y->nb[1], y->nb[2], y->nb[3], (T - 1) * y->nb[3]),
        GGML_TYPE_F16);

    // stride-2 temporal conv over the assembled window: out[j] = sum_tap w_tap * xin[2j + tap]
    const std::int64_t T2 = (xin->ne[3] - 3) / 2 + 1;
    void* taps[3] = { u.tconv.tap0, u.tconv.tap1, u.tconv.tap2 };
    ggml_tensor* acc = nullptr;
    for (int j = 0; j < 3; j++)
    {
        ggml_tensor* tw = wan_vae_conv_w(b, taps[j], u.tconv);
        ggml_tensor* win = ggml_view_4d(ctx, xin, xin->ne[0], xin->ne[1], xin->ne[2], T2,
                                        xin->nb[1], xin->nb[2], 2 * xin->nb[3], j * xin->nb[3]);
        win = ggml_cont(ctx, win);
        ggml_tensor* yj = wan_vae_conv2d(b, tw, win, 0);
        acc = acc == nullptr ? yj : ggml_add(ctx, acc, yj);
    }
    if (u.tconv.bias != nullptr)
    {
        ggml_tensor* bt = b.wb->f32v(u.tconv.bias, u.tconv.oc);
        acc = ggml_add(ctx, acc, ggml_reshape_4d(ctx, bt, 1, 1, u.tconv.oc, 1));
    }
    return acc;
}

// AvgDown3D (wan2.2 Down_ResidualBlock shortcut). With the wan2.2 encoder's
// group sizes this reduces to a spatial 2x2 average pool plus (ft == 2) a
// temporal-pair channel regroup: out channel c*2 + ift holds frame parity ift
// (front zero-padded to even length, matching diffusers' F.pad).
ggml_tensor* wan_vae_avg_down(WanVaeBuild& b, ggml_tensor* x, int ft, int fs)
{
    ggml_context* ctx = b.ctx;
    if (fs == 2)
        x = ggml_pool_2d(ctx, x, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
    if (ft == 2)
    {
        if (x->ne[3] % 2 == 1)
            x = wan_pad_ext_compat(ctx, x, 0, 0, 0, 0, 0, 0, 1, 0);       // front zero frame
        const std::int64_t W = x->ne[0], H = x->ne[1], C = x->ne[2], T = x->ne[3];
        ggml_tensor* r = ggml_reshape_4d(ctx, x, W * H, C, 2, T / 2);
        r = ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));       // [WH, 2, C, T/2]
        x = ggml_reshape_4d(ctx, r, W, H, 2 * C, T / 2);
    }
    return x;
}

// One encoder pass over one pixel chunk (1 frame, then 4-frame chunks). Mirrors
// WanEncoder3d: stem conv, 4 scales x 2 res blocks with downsamples after scales
// 0 (spatial) / 1,2 (temporal+spatial), middle (res/attn/res), head, quant conv;
// returns the posterior mean (first z_dim channels). Wan 2.2 adds the AvgDown3D
// residual shortcut around every scale group.
ggml_tensor* wan_vae_encode_chunk(WanVaeBuild& b, const TSGgmlWanVaeEncodeDesc* d, ggml_tensor* xc)
{
    ggml_context* ctx = b.ctx;
    b.cursor = 0;
    const bool w22 = d->version == 2;

    ggml_tensor* x = wan_vae_causal_conv(b, d->stem, xc);

    for (int scale = 0; scale < 4; scale++)
    {
        ggml_tensor* xin = x;
        for (int r = 0; r < 2; r++)
            x = wan_vae_res_block(b, d->res[scale * 2 + r], x);
        if (scale < 3)
            x = wan_vae_enc_resample(b, d->down[scale], x);
        if (w22)
        {
            // temperal_downsample = [false, true, true]; spatial down at scales 0..2
            const int ft = (scale == 1 || scale == 2) ? 2 : 1;
            const int fs = scale < 3 ? 2 : 1;
            ggml_tensor* sc = (ft == 1 && fs == 1) ? xin : wan_vae_avg_down(b, xin, ft, fs);
            x = ggml_add(ctx, x, sc);
        }
    }

    x = wan_vae_res_block(b, d->mid0, x);
    x = wan_vae_attn(b, d->mid1, x);
    x = wan_vae_res_block(b, d->mid2, x);

    x = wan_vae_norm(b, d->head_norm, x);
    x = ggml_silu(ctx, x);
    x = wan_vae_causal_conv(b, d->head_conv, x);                    // [lw, lh, 2z, t]
    x = wan_vae_causal_conv(b, d->quant, x);

    // posterior mean = first z_dim channels
    return ggml_cont(ctx, ggml_view_4d(ctx, x, x->ne[0], x->ne[1], d->z_dim, x->ne[3],
                                       x->nb[1], x->nb[2], x->nb[3], 0));
}

// Per-chunk graph-node budget for the VAE graphs. The 4096 base covers the
// conv/norm chains at the 832x480 recipe, but the banded im2col conv splits
// (wan_vae_conv2d) add nodes proportional to the full-resolution plane area —
// and inversely to TS_WAN_VAE_GEMM_MAX_MB — so the budget must scale with both
// (a 768x1152 decode overflowed the fixed budget AFTER the full denoise).
// Meta is transient host memory, so overestimating is cheap; underestimating
// is a GGML abort.
static std::size_t wan_vae_chunk_nodes(long long pixels)
{
    long long scale = (pixels + 832LL * 480 - 1) / (832LL * 480);
    const char* e = std::getenv("TS_WAN_VAE_GEMM_MAX_MB");
    const long long mb = e != nullptr ? std::strtoll(e, nullptr, 10) : 384;
    if (mb > 0 && mb < 384)
        scale *= (384 + mb - 1) / mb;
    if (scale < 1) scale = 1;
    if (scale > 64) scale = 64;   // RAM guard: 64x is already ~2 GB of graph meta
    return static_cast<std::size_t>(4096) * static_cast<std::size_t>(scale);
}

bool wan_vae_encode(const TSGgmlWanVaeEncodeDesc* d)
{
    const int pw = d->px_w, ph = d->px_h, pc = d->px_c, pt = d->px_t;
    const int chunks = 1 + (pt - 1 + 3) / 4;

    // pw/ph are the (pre-patchified for wan2.2) pixel grid: x4 restores pixels.
    const std::size_t nodes = static_cast<std::size_t>(chunks) *
        wan_vae_chunk_nodes(static_cast<long long>(pw) * ph * (d->version == 2 ? 4 : 1)) + 4096;
    const std::size_t meta = ggml_tensor_overhead() * (nodes + 4096)
                             + ggml_graph_overhead_custom(nodes, false) + (16u << 20);
    ggml_init_params ip{ meta, nullptr, true };
    ContextHandle context(ggml_init(ip));
    if (context.value == nullptr) { set_last_error("WanVaeEncode: ctx alloc failed."); return false; }
    ggml_context* ctx = context.value;

    WanBind wbind; wbind.ctx = ctx; wbind.dev = ggml_backend_get_device(g_backend);
    WanVaeBuild b; b.ctx = ctx; b.wb = &wbind;
    b.cache.assign(64, nullptr);
    b.gemmMax = wan_vae_gemm_budget();

    ggml_tensor* xIn = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, pw, ph, pc, pt);
    ggml_set_input(xIn);

    ggml_tensor* out = nullptr;
    for (int i = 0; i < chunks; i++)
    {
        b.chunk = i;
        const int from = i == 0 ? 0 : 1 + 4 * (i - 1);
        const int count = std::min(pt, 1 + 4 * i) - from;
        ggml_tensor* xc = ggml_view_4d(ctx, xIn, pw, ph, pc, count,
                                       xIn->nb[1], xIn->nb[2], xIn->nb[3],
                                       static_cast<std::size_t>(from) * xIn->nb[3]);
        ggml_tensor* mu = wan_vae_encode_chunk(b, d, xc);
        out = out == nullptr ? mu : ggml_concat(ctx, out, mu, 3);
    }

    ggml_tensor* outT = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, out->ne[0], out->ne[1], out->ne[2], out->ne[3]);
    ggml_tensor* outc = ggml_cpy(ctx, out, outT);
    ggml_set_output(outc);

    const std::int64_t outElems = out->ne[0] * out->ne[1] * out->ne[2] * out->ne[3];
    if (outElems != d->out_len)
    {
        std::fprintf(stderr, "[wan] vae encode: got [%lld, %lld, %lld, %lld] = %lld, expected %lld\n",
                     (long long)out->ne[0], (long long)out->ne[1], (long long)out->ne[2], (long long)out->ne[3],
                     (long long)outElems, (long long)d->out_len);
        set_last_error("WanVaeEncode: output shape mismatch.");
        return false;
    }

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
    ggml_build_forward_expand(graph, outc);
    for (ggml_tensor* c : b.cache)
        if (c != nullptr) ggml_build_forward_expand(graph, c);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
    if (galloc == nullptr) { set_last_error("WanVaeEncode: gallocr alloc failed."); return false; }
    struct GallocGuard { ggml_gallocr_t g; ~GallocGuard() { if (g) ggml_gallocr_free(g); } } guard{galloc};
    if (!ggml_gallocr_alloc_graph(galloc, graph))
    { set_last_error("WanVaeEncode: graph alloc failed (out of device memory?)."); return false; }

    host_read_barrier();
    for (auto& u : wbind.uploads) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
    ggml_backend_tensor_set(xIn, d->x, 0, static_cast<std::size_t>(pw) * ph * pc * pt * sizeof(float));

    // The encoder shares wan_vae_conv2d, so on the MPS path its graph also holds
    // un-lowered CONV_2D nodes and needs the same segmented runner.
    const ggml_status encSt = tsg::fast_conv_enabled()
        ? tsg::graph_compute_fast_conv(graph, "wan vae")
        : tsg::compute_graph(g_backend, graph);
    if (encSt != GGML_STATUS_SUCCESS)
    { set_last_error("WanVaeEncode: graph compute failed."); return false; }
    tsg::sync_backend(g_backend);
    ggml_backend_tensor_get(outT, d->out, 0, static_cast<std::size_t>(d->out_len) * sizeof(float));
    return true;
}

static bool wan_vae_op_writes(const ggml_tensor* t);
static bool wan_vae_node_live(ggml_cgraph* g, int i, int upto);

// TS_WAN_VAE_TRACE=<path>: after compute, walk the graph nodes in execution
// order and report the first ones whose buffer holds NaN/Inf. Only nodes whose
// slot has not been legally reused by a later writer are scanned (gallocr
// aliasing makes anything else a dtype-misread ghost).
void wan_vae_trace_nan(ggml_cgraph* graph)
{
    const char* path = std::getenv("TS_WAN_VAE_TRACE");
    if (path == nullptr) return;
    std::FILE* out = std::fopen(path, "a");
    if (out == nullptr) return;
    const int n = ggml_graph_n_nodes(graph);
    int printed = 0;
    for (int i = 0; i < n && printed < 12; i++)
    {
        ggml_tensor* t = ggml_graph_node(graph, i);
        if (t->buffer == nullptr || t->data == nullptr) continue;
        const std::int64_t nel = ggml_nelements(t);
        if (nel <= 0) continue;
        if (!wan_vae_op_writes(t)) continue;          // views mirror their src
        if (!wan_vae_node_live(graph, i, n)) continue; // slot legally reused
        std::vector<float> host;
        if (t->type == GGML_TYPE_F32)
        {
            host.resize(static_cast<std::size_t>(nel));
            ggml_backend_tensor_get(t, host.data(), 0, static_cast<std::size_t>(nel) * sizeof(float));
        }
        else if (t->type == GGML_TYPE_F16)
        {
            std::vector<ggml_fp16_t> h16(static_cast<std::size_t>(nel));
            ggml_backend_tensor_get(t, h16.data(), 0, static_cast<std::size_t>(nel) * sizeof(ggml_fp16_t));
            host.resize(static_cast<std::size_t>(nel));
            for (std::int64_t j = 0; j < nel; j++) host[j] = ggml_fp16_to_fp32(h16[j]);
        }
        else continue;
        std::int64_t nan = 0;
        for (std::int64_t j = 0; j < nel; j++)
            if (std::isnan(host[j]) || std::isinf(host[j])) nan++;
        if (nan == 0) continue;
        std::fprintf(out, "[wan-vae-trace] node %d %s '%s' %s ne=[%lld,%lld,%lld,%lld] nan=%lld/%lld\n",
                     i, ggml_op_name(t->op), t->name, ggml_type_name(t->type),
                     (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                     (long long)nan, (long long)nel);
        for (int s = 0; s < 2; s++)
        {
            ggml_tensor* src = t->src[s];
            if (src == nullptr) continue;
            std::fprintf(out, "[wan-vae-trace]   src%d %s '%s' %s ne=[%lld,%lld,%lld,%lld] buf=%s\n",
                         s, ggml_op_name(src->op), src->name, ggml_type_name(src->type),
                         (long long)src->ne[0], (long long)src->ne[1], (long long)src->ne[2], (long long)src->ne[3],
                         src->buffer ? ggml_backend_buffer_name(src->buffer) : "none");
        }
        printed++;
    }
    if (printed == 0)
        std::fprintf(out, "[wan-vae-trace] no NaN in any of %d nodes\n", n);
    std::fclose(out);
}

// A node's op writes memory unless it is a pure view. (NONE = leaf.)
static bool wan_vae_op_writes(const ggml_tensor* t)
{
    switch (t->op)
    {
        case GGML_OP_NONE: case GGML_OP_RESHAPE: case GGML_OP_VIEW:
        case GGML_OP_PERMUTE: case GGML_OP_TRANSPOSE:
            return false;
        default:
            return true;
    }
}

// Is node i's output still readable (not legally overwritten) after nodes
// (i, upto) have run? gallocr reuses freed slots, so a later writer whose dst
// range overlaps node i's range invalidates it.
static bool wan_vae_node_live(ggml_cgraph* g, int i, int upto)
{
    ggml_tensor* t = ggml_graph_node(g, i);
    const std::uintptr_t s0 = reinterpret_cast<std::uintptr_t>(t->data);
    const std::uintptr_t e0 = s0 + ggml_nbytes(t);
    for (int j = i + 1; j < upto; j++)
    {
        ggml_tensor* u = ggml_graph_node(g, j);
        if (!wan_vae_op_writes(u) || u->data == nullptr) continue;
        const std::uintptr_t s1 = reinterpret_cast<std::uintptr_t>(u->data);
        const std::uintptr_t e1 = s1 + ggml_nbytes(u);
        if (s1 < e0 && s0 < e1) return false;
    }
    return true;
}


bool wan_vae_decode(const TSGgmlWanVaeDecodeDesc* d)
{
    const int zw = d->zw, zh = d->zh, zt = d->zt, zc = d->zc;

    // 16x (wan2.2) / 8x (wan2.1) spatial VAE: latent -> pixel area for the budget.
    const std::size_t nodes = static_cast<std::size_t>(zt) *
        wan_vae_chunk_nodes(static_cast<long long>(zw) * zh * (d->version == 2 ? 256 : 64)) + 4096;
    const std::size_t meta = ggml_tensor_overhead() * (nodes + 4096)
                             + ggml_graph_overhead_custom(nodes, false) + (16u << 20);
    ggml_init_params ip{ meta, nullptr, true };
    ContextHandle context(ggml_init(ip));
    if (context.value == nullptr) { set_last_error("WanVaeDecode: ctx alloc failed."); return false; }
    ggml_context* ctx = context.value;

    WanBind wbind; wbind.ctx = ctx; wbind.dev = ggml_backend_get_device(g_backend);
    WanVaeBuild b; b.ctx = ctx; b.wb = &wbind;
    b.cache.assign(64, nullptr);
    b.gemmMax = wan_vae_gemm_budget();

    ggml_tensor* zIn = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, zw, zh, zc, zt);
    ggml_set_input(zIn);

    // post-quant conv (1x1x1) over the whole latent
    ggml_tensor* z2 = wan_vae_causal_conv(b, d->conv2, zIn);

    // Chunks must be BUILT in order (each one advances the causal feature caches),
    // but they are JOINED as a balanced tree — see wan_concat_all. The old
    // left-leaning chain re-copied the whole accumulated video once per chunk, which
    // at 121 frames is ~31x the output through ggml's generic concat kernel.
    std::vector<ggml_tensor*> chunks;
    chunks.reserve(static_cast<std::size_t>(zt));
    for (int i = 0; i < zt; i++)
    {
        b.chunk = i;
        ggml_tensor* z1 = ggml_view_4d(ctx, z2, zw, zh, zc, 1, z2->nb[1], z2->nb[2], z2->nb[3], i * z2->nb[3]);
        chunks.push_back(wan_vae_decode_chunk(b, d, z1));
    }
    ggml_tensor* out = wan_concat_all(ctx, chunks, 3);

    ggml_tensor* outT = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, out->ne[0], out->ne[1], out->ne[2], out->ne[3]);
    ggml_tensor* outc = ggml_cpy(ctx, out, outT);
    ggml_set_output(outc);

    const std::int64_t outElems = out->ne[0] * out->ne[1] * out->ne[2] * out->ne[3];
    if (outElems != d->out_len)
    { set_last_error("WanVaeDecode: output shape mismatch."); return false; }

    ggml_cgraph* graph = ggml_new_graph_custom(ctx, nodes, false);
    ggml_build_forward_expand(graph, outc);
    // Keep every cross-chunk cache tensor alive to the end of the graph (its last
    // reader may otherwise free the buffer slot a later chunk still reads).
    for (ggml_tensor* c : b.cache)
        if (c != nullptr) ggml_build_forward_expand(graph, c);

    // The VAE working set is huge (hundreds of MB per tensor at full resolution);
    // keep it in a dedicated gallocr freed at the end of the call rather than
    // growing the shared reuse gallocr to VAE size permanently.
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
    if (galloc == nullptr) { set_last_error("WanVaeDecode: gallocr alloc failed."); return false; }
    struct GallocGuard { ggml_gallocr_t g; ~GallocGuard() { if (g) ggml_gallocr_free(g); } } guard{galloc};
    if (!ggml_gallocr_alloc_graph(galloc, graph))
    { set_last_error("WanVaeDecode: graph alloc failed (out of device memory?)."); return false; }

    host_read_barrier();
    for (auto& u : wbind.uploads) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
    ggml_backend_tensor_set(zIn, d->z, 0, static_cast<std::size_t>(zw) * zh * zc * zt * sizeof(float));

    // TS_WAN_VAE_SLICE=<K>[,nosync] (debug): execute the graph as sequential
    // K-node sub-graphs (buffers stay as the full-graph gallocr assigned them),
    // synchronizing between slices unless ",nosync". K <= 64 keeps every slice
    // in a single main-thread command buffer.
    const char* sliceEnv = std::getenv("TS_WAN_VAE_SLICE");
    if (sliceEnv != nullptr && std::strtol(sliceEnv, nullptr, 10) > 0)
    {
        const int K = static_cast<int>(std::strtol(sliceEnv, nullptr, 10));
        const bool doSync = std::strstr(sliceEnv, "nosync") == nullptr;
        const int n = ggml_graph_n_nodes(graph);
        for (int a = 0; a < n; a += K)
        {
            const int e = std::min(n, a + K);
            const std::size_t smeta = ggml_graph_overhead_custom(static_cast<std::size_t>(K) + 8, false) + (1u << 20);
            ggml_init_params sip{ smeta, nullptr, true };
            ggml_context* sctx = ggml_init(sip);
            ggml_cgraph* sub = ggml_new_graph_custom(sctx, static_cast<std::size_t>(K) + 8, false);
            for (int i = a; i < e; i++)
                ggml_graph_add_node(sub, ggml_graph_node(graph, i));
            const ggml_status st = tsg::compute_graph(g_backend, sub);
            if (st != GGML_STATUS_SUCCESS)
            { ggml_free(sctx); set_last_error("WanVaeDecode: sliced graph compute failed."); return false; }
            if (doSync) tsg::sync_backend(g_backend);

            // TS_WAN_VAE_SLICE_SCAN=<path>: immediately after each synced slice,
            // scan the slice's own outputs (nothing later has run, so only
            // in-slice reuse can alias) and report the first corrupt node.
            static const char* scanPath = std::getenv("TS_WAN_VAE_SLICE_SCAN");
            if (scanPath != nullptr && doSync)
            {
                for (int i = 0; i < e - a; i++)
                {
                    ggml_tensor* t = ggml_graph_node(sub, i);
                    if (t->buffer == nullptr || t->data == nullptr) continue;
                    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) continue;
                    if (!wan_vae_op_writes(t)) continue;
                    if (!wan_vae_node_live(sub, i, e - a)) continue;
                    const std::int64_t nel = ggml_nelements(t);
                    std::vector<float> host(static_cast<std::size_t>(nel));
                    if (t->type == GGML_TYPE_F32)
                        ggml_backend_tensor_get(t, host.data(), 0, static_cast<std::size_t>(nel) * sizeof(float));
                    else
                    {
                        std::vector<ggml_fp16_t> h16(static_cast<std::size_t>(nel));
                        ggml_backend_tensor_get(t, h16.data(), 0, static_cast<std::size_t>(nel) * sizeof(ggml_fp16_t));
                        for (std::int64_t j = 0; j < nel; j++) host[j] = ggml_fp16_to_fp32(h16[j]);
                    }
                    std::int64_t nan = 0;
                    for (std::int64_t j = 0; j < nel; j++)
                        if (std::isnan(host[j]) || std::isinf(host[j])) nan++;
                    if (nan == 0) continue;
                    std::FILE* out = std::fopen(scanPath, "a");
                    if (out != nullptr)
                    {
                        ggml_tensor* g = ggml_graph_node(graph, a + i);
                        std::fprintf(out, "[wan-vae-slice] FIRST corrupt node %d (slice [%d,%d)) %s '%s' %s "
                                          "ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] data=%p buf=%s nan=%lld/%lld\n",
                                     a + i, a, e, ggml_op_name(g->op), g->name, ggml_type_name(g->type),
                                     (long long)g->ne[0], (long long)g->ne[1], (long long)g->ne[2], (long long)g->ne[3],
                                     g->nb[0], g->nb[1], g->nb[2], g->nb[3], g->data,
                                     g->buffer ? ggml_backend_buffer_name(g->buffer) : "none",
                                     (long long)nan, (long long)nel);
                        for (int s = 0; s < 3; s++)
                        {
                            ggml_tensor* src = g->src[s];
                            if (src == nullptr) continue;
                            ggml_tensor* vsrc = src->view_src != nullptr ? src->view_src : src;
                            std::fprintf(out, "[wan-vae-slice]   src%d %s '%s' %s ne=[%lld,%lld,%lld,%lld] "
                                              "nb=[%zu,%zu,%zu,%zu] data=%p buf=%s (view of %s '%s')\n",
                                         s, ggml_op_name(src->op), src->name, ggml_type_name(src->type),
                                         (long long)src->ne[0], (long long)src->ne[1], (long long)src->ne[2], (long long)src->ne[3],
                                         src->nb[0], src->nb[1], src->nb[2], src->nb[3], src->data,
                                         src->buffer ? ggml_backend_buffer_name(src->buffer) : "none",
                                         ggml_op_name(vsrc->op), vsrc->name);
                        }
                        std::fclose(out);
                    }
                    scanPath = nullptr;   // report only the first
                    break;
                }
            }
            ggml_free(sctx);
        }
    }
    else if (tsg::fast_conv_enabled())
    {
        if (tsg::graph_compute_fast_conv(graph, "wan vae") != GGML_STATUS_SUCCESS)
        { set_last_error("WanVaeDecode: graph compute failed (MPS conv path)."); return false; }
    }
    else if (tsg::graph_compute_profiled(g_backend, graph, "wan vae decode") != GGML_STATUS_SUCCESS)
    { set_last_error("WanVaeDecode: graph compute failed."); return false; }
    tsg::sync_backend(g_backend);
    wan_vae_trace_nan(graph);
    ggml_backend_tensor_get(outT, d->out, 0, static_cast<std::size_t>(d->out_len) * sizeof(float));
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

extern "C" {

TSG_EXPORT void TSGgml_WanResetForwardCache()
{
    for (auto& e : g_wanDit) e.reset();
    g_wanDitRR = 0;
    // The "too big to keep resident" verdicts were measured against the VRAM the
    // previous stage had committed. A reset means that residency is being handed
    // back (stage boundary, model reload), so re-probe rather than inherit them.
    g_wanDitTooBig.clear();
}

TSG_EXPORT int TSGgml_WanT5Encode(const TSGgmlWanT5Desc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlWanT5Desc)))
        { set_last_error("WanT5Encode: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;
        if (!wan_t5_build_and_run(d)) return 0;
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("WanT5Encode: unknown error."); return 0; }
}

TSG_EXPORT int TSGgml_WanDitForward(const TSGgmlWanDitDesc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlWanDitDesc)))
        { set_last_error("WanDitForward: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;

        if (wan_dit_capture_enabled())
        {
            WanDitPersist* e = nullptr;
            for (auto& c : g_wanDit) if (c.matches(d)) { e = &c; break; }
            if (e == nullptr && !wan_dit_too_big(d)) e = wan_dit_build_persist(d);
            if (e != nullptr) return wan_dit_run_persist(e, d);
            // build failed (VRAM?): fall through to the rebuild-per-forward path
        }

        const std::size_t nodes = wan_dit_nodes(d->num_layers, d->ff, d->seq);
        const std::size_t meta = ggml_tensor_overhead() * (nodes + 1024)
                                 + ggml_graph_overhead_custom(nodes, false) + (8u << 20);
        ggml_init_params ip{ meta, nullptr, true };
        ContextHandle context(ggml_init(ip));
        if (context.value == nullptr) { set_last_error("WanDitForward: ctx alloc failed."); return 0; }

        WanDitGraph g;
        if (!wan_dit_build_graph(context.value, d, g))
        { set_last_error("WanDitForward: graph build failed."); return 0; }

        // A whole-DiT context contains every intermediate from every block.
        // It must always use the graph allocator so mutually exclusive tensor
        // lifetimes share storage. Falling back to alloc_ctx_tensors after a
        // gallocr OOM sums all block temporaries (several TiB at long video
        // lengths) and can never improve on the lifetime-packed allocation.
        ggml_gallocr_t localGalloc = nullptr;
        struct LocalGallocGuard
        {
            ggml_gallocr_t& value;
            ~LocalGallocGuard() { if (value != nullptr) ggml_gallocr_free(value); }
        } localGallocGuard{localGalloc};
        if (!alloc_graph_reuse_gallocr(g.graph))
        {
            // The shared allocator can be disabled for diagnostics, so retain a
            // dedicated gallocr fallback. If the shared attempt really was an
            // OOM this may fail too, but it remains lifetime-packed and reports
            // the real requirement instead of attempting a context-wide buffer.
            localGalloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
            if (localGalloc == nullptr || !ggml_gallocr_alloc_graph(localGalloc, g.graph))
            {
                set_last_error(
                    "WanDitForward: lifetime-packed graph buffer allocation failed. "
                    "Reduce video frames/resolution or free device memory.");
                return 0;
            }
        }

        host_read_barrier();
        for (auto& u : g.uploads)
            if (u.t->buffer != nullptr) ggml_backend_tensor_set(u.t, u.d, 0, u.b);
        wan_dit_upload_inputs(g, d);

        if (tsg::compute_graph(g_backend, g.graph) != GGML_STATUS_SUCCESS)
        { set_last_error("WanDitForward: graph compute failed."); return 0; }
        tsg::sync_backend(g_backend);
        wan_dit_trace_taps(g);
        ggml_backend_tensor_get(g.outT, d->out, 0, static_cast<std::size_t>(d->head.ne1) * d->seq * sizeof(float));
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("WanDitForward: unknown error."); return 0; }
}

TSG_EXPORT int TSGgml_WanVaeDecode(const TSGgmlWanVaeDecodeDesc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlWanVaeDecodeDesc)))
        { set_last_error("WanVaeDecode: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;
        if (!wan_vae_decode(d)) return 0;
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("WanVaeDecode: unknown error."); return 0; }
}

TSG_EXPORT int TSGgml_WanVaeEncode(const TSGgmlWanVaeEncodeDesc* d)
{
    try
    {
        if (d == nullptr || d->struct_bytes != static_cast<std::int32_t>(sizeof(TSGgmlWanVaeEncodeDesc)))
        { set_last_error("WanVaeEncode: bad descriptor."); return 0; }
        if (!ensure_backend()) return 0;
        if (!wan_vae_encode(d)) return 0;
        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex) { set_last_error(ex.what()); return 0; }
    catch (...) { set_last_error("WanVaeEncode: unknown error."); return 0; }
}

} // extern "C"
