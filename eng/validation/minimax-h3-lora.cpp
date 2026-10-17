// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// MiniMax-H3 DiT LoRA graph helpers (TensorSharp.GGML.Native/ggml_h3_lora.h)
// against unchanged upstream ggml.
//
//   GgmlOpsMiniMaxH3LoraTest cpu|metal|cuda --test      (exit 77: backend unavailable)
//
// --test runs the native (L1) cases of the MiniMax-H3 Turbo plug-in design:
//   1  expand/shrink against a double reference: F32 and F16 factors over F32,
//      F16, Q8_0 and Q4_K bases, with h3_mm's quantized-activation guard on and off
//   2  a stacked q/k/v shrink is bit-equal to three separate ones
//   3  the folds are bit-equal: the slice add against cont + add, and strength *
//      alpha / rank (8 / 128) folded into up against scaling the product
//   4  the split fc1 against swiglu(gv + delta), for full, gate-only and
//      value-only updates (bit-equal on cpu)
//   5  zero up factors leave a refiner block and two trunk blocks bit-equal on
//      cpu and metal (CUDA records max |d|); real factors change them
//   6  an adapter whose updates are all rank 0 builds the same nodes and bits as none
//   7  one block's six updates add <= kH3LoraNodesPerBlock nodes, and leaves
//   8  compute buffer at seq 2048 and H3's widths: split fc1 <= base * 1.05 < naive
//   9  graph metadata of the full-depth graph with an adapter (50 trunk + 2
//      refiner blocks, the most segments a request can make) stays within 75% of
//      the 32 MiB context TSGgml_MiniMaxH3DitForward builds it in
//  10  every h3_lora_validate rejection, by its message
//
// The blocks here are a MIRROR of TSGgml_MiniMaxH3DitForward (ggml_ops_minimax_h3.cpp):
// that function lives in GgmlOps behind its process-wide backend, so this ggml-only
// test restates its attention, MLP and modulation, calling the ggml_h3_lora.h
// helpers exactly as it does. Case 9 counts the mirror's tensors, so a change to
// the DiT graph belongs here as well.
#include "ggml_h3_lora.h"
#include "ggml_h3_token_rows.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef TSG_GGML_USE_METAL
#include "ggml-metal.h"
#endif
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

// Bit-equality is required on cpu and metal; ggml-cuda's identity cases are recorded.
bool g_exact = true;
std::string g_report;

void report(const std::string & key, const std::string & value) {
    g_report += (g_report.empty() ? "" : ",") + ("\"" + key + "\":" + value);
}

std::string num(double v) {
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.6g", v);
    return buffer;
}

// Deterministic values in [-scale, scale).
std::vector<float> values(std::size_t n, std::uint32_t seed, float scale) {
    std::vector<float> v(n);
    std::uint32_t s = seed * 2654435761u + 0x9e3779b9u;
    for (float & x : v) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        x = (static_cast<float>(s >> 8) / 8388608.0f - 1.0f) * scale;
    }
    return v;
}

// Host bytes of a [ne0, rows] tensor in `type`, and the floats ggml decodes them to.
struct Blob {
    std::vector<std::uint8_t> bytes;
    std::vector<float> decoded;
    void * data() { return bytes.data(); }
};

Blob encode(ggml_type type, const std::vector<float> & v, std::int64_t ne0) {
    const std::int64_t rows = static_cast<std::int64_t>(v.size()) / ne0;
    Blob b;
    b.bytes.resize(ggml_row_size(type, ne0) * static_cast<std::size_t>(rows));
    if (type == GGML_TYPE_F32) std::memcpy(b.bytes.data(), v.data(), b.bytes.size());
    else if (type == GGML_TYPE_F16)
        ggml_fp32_to_fp16_row(v.data(), reinterpret_cast<ggml_fp16_t *>(b.bytes.data()),
                              static_cast<std::int64_t>(v.size()));
    else ggml_quantize_chunk(type, v.data(), b.bytes.data(), 0, rows, ne0, nullptr);
    b.decoded.resize(v.size());
    if (type == GGML_TYPE_F32) b.decoded = v;
    else ggml_get_type_traits(type)->to_float(b.bytes.data(), b.decoded.data(), static_cast<std::int64_t>(v.size()));
    return b;
}

// Stable storage for every buffer a graph binds.
std::deque<Blob> g_blobs;
Blob & keep(Blob b) { g_blobs.push_back(std::move(b)); return g_blobs.back(); }
Blob & keep(ggml_type type, const std::vector<float> & v, std::int64_t ne0) { return keep(encode(type, v, ne0)); }

// One graph on the test backend: binds become inputs uploaded before compute,
// exactly as the DiT's `bind` does for a weight the device cache declines.
struct Run {
    ggml_backend_t backend;
    ggml_context * ctx = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ggml_cgraph * graph = nullptr;
    struct Upload { ggml_tensor * t; const void * data; std::size_t bytes; };
    std::vector<Upload> uploads;
    tsg::H3Bind bind;

    explicit Run(ggml_backend_t b, std::size_t meta = 64u << 20) : backend(b) {
        ctx = ggml_init({meta, nullptr, true});
        require(ctx != nullptr, "context allocation failed");
        bind = [this](ggml_tensor * t, void * data, std::size_t bytes) {
            if (t == nullptr || data == nullptr) return;
            require(bytes == ggml_nbytes(t), "a bound byte count differs from its tensor");
            ggml_set_input(t);
            uploads.push_back({t, data, bytes});
        };
    }
    ~Run() { if (allocator) ggml_gallocr_free(allocator); if (ctx) ggml_free(ctx); }
    Run(const Run &) = delete;
    Run & operator=(const Run &) = delete;

    ggml_tensor * input(ggml_type type, std::int64_t ne0, std::int64_t ne1, void * data) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
        bind(t, data, ggml_nbytes(t));
        return t;
    }
    void build(const std::vector<ggml_tensor *> & outputs) {
        graph = ggml_new_graph_custom(ctx, 16384, false);
        for (ggml_tensor * o : outputs) { ggml_set_output(o); ggml_build_forward_expand(graph, o); }
    }
    void compute() {
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
            require(ggml_backend_supports_op(backend, ggml_graph_node(graph, i)),
                    std::string("backend does not support ") + ggml_op_desc(ggml_graph_node(graph, i)));
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(allocator && ggml_gallocr_alloc_graph(allocator, graph), "graph allocation failed");
        for (const Upload & u : uploads) ggml_backend_tensor_set(u.t, u.data, 0, u.bytes);
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute failed");
        ggml_backend_synchronize(backend);
    }
    std::vector<float> read(ggml_tensor * t) {
        require(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t), "outputs must be contiguous F32");
        std::vector<float> out(static_cast<std::size_t>(ggml_nelements(t)));
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
        for (float v : out) require(std::isfinite(v), "nonfinite output");
        return out;
    }
    int count(ggml_op op) const {
        int n = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) n += ggml_graph_node(graph, i)->op == op;
        return n;
    }
};

// The graph's leaves (ggml_cgraph is opaque): every distinct operand that is no op.
int count_leafs(ggml_cgraph * graph) {
    std::unordered_set<const ggml_tensor *> leafs;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        for (const ggml_tensor * src : ggml_graph_node(graph, i)->src)
            if (src != nullptr && src->op == GGML_OP_NONE) leafs.insert(src);
    return static_cast<int>(leafs.size());
}

// y[o, s] = sum_i w[i, o] x[i, s] in double; w is [in, out], x [in, seq] (ggml order).
std::vector<double> matmul(const std::vector<float> & w, std::int64_t in, std::int64_t out,
                           const std::vector<double> & x, std::int64_t seq) {
    std::vector<double> y(static_cast<std::size_t>(out * seq));
    for (std::int64_t s = 0; s < seq; ++s)
        for (std::int64_t o = 0; o < out; ++o) {
            double acc = 0;
            for (std::int64_t i = 0; i < in; ++i) acc += double(w[o * in + i]) * x[s * in + i];
            y[s * out + o] = acc;
        }
    return y;
}

std::vector<double> widen(const std::vector<float> & v) { return std::vector<double>(v.begin(), v.end()); }

// max |got - ref| / max |ref|
double rel_err(const std::vector<float> & got, const std::vector<double> & ref) {
    double worst = 0, peak = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        worst = std::max(worst, std::fabs(double(got[i]) - ref[i]));
        peak = std::max(peak, std::fabs(ref[i]));
    }
    return peak > 0 ? worst / peak : worst;
}

double rel_err(const std::vector<float> & got, const std::vector<float> & ref) { return rel_err(got, widen(ref)); }

bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Identity cases: bit-equal where required, else within `tolerance`; returns max |d| / max |ref|.
double check_identity(const std::vector<float> & got, const std::vector<float> & ref, const std::string & what,
                      double tolerance = 1e-3) {
    const double err = rel_err(got, ref);
    if (g_exact) require(same_bits(got, ref), what + " is not bit-equal (max rel " + num(err) + ")");
    else require(err <= tolerance, what + " differs by " + num(err));
    return err;
}

// ---- mirror of ggml_ops_minimax_h3.cpp ------------------------------------

constexpr float kGuard = 1024.0f;          // kH3MmScale
constexpr int kFlashKeyBudget = 256;       // kH3FlashKeyBudget

// h3_mm: a quantized weight's activation pre-scaled by 1 / kGuard and rescaled after.
ggml_tensor * mm_guarded(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, bool guard) {
    if (!guard || !ggml_is_quantized(w->type)) return ggml_mul_mat(ctx, w, x);
    ggml_tensor * xs = ggml_scale(ctx, x, 1.0f / kGuard);
    return ggml_scale(ctx, ggml_mul_mat(ctx, w, xs), kGuard);
}

float flash_v_scale(int keys) {
    float s = 1.0f;
    while (keys > kFlashKeyBudget * s) s *= 2.0f;
    return s;
}

struct Lin { void * w = nullptr; void * b = nullptr; std::int64_t ne0 = 0, ne1 = 0; ggml_type type = GGML_TYPE_F32; };
struct BlockW { void * norm1 = nullptr; void * norm2 = nullptr; void * q_norm = nullptr; void * k_norm = nullptr;
                Lin adaln, qkv, out, fc1, fc2; };
struct Seg { int start, end, col; };
struct Dims { int hidden, heads, hd, ffn, rot, ted; float eps = 1e-5f; int inner() const { return heads * hd; } };

struct Mirror {
    ggml_context * ctx;
    ggml_backend_t backend;
    const tsg::H3Bind & bind;
    Dims d;
    // ggml-metal's key-padded flash call (h3_attend_padded), the larger graph; built
    // for case 9's count only, never computed here.
    bool padded = false;
    struct Mask { int seq; ggml_tensor * mask; };
    std::vector<Mask> masks;

    Mirror(ggml_context * c, ggml_backend_t b, const tsg::H3Bind & bd, Dims dims) : ctx(c), backend(b), bind(bd), d(dims) {}

    ggml_tensor * vec(void * data, int n) {
        if (data == nullptr) return nullptr;
        ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        bind(t, data, static_cast<std::size_t>(n) * sizeof(float));
        return t;
    }
    void decl(const Lin & s, ggml_tensor *& w, ggml_tensor *& b) {
        w = ggml_new_tensor_2d(ctx, s.type, s.ne0, s.ne1);
        bind(w, s.w, ggml_nbytes(w));
        b = nullptr;
        if (s.b != nullptr) {
            b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.ne1);
            bind(b, s.b, static_cast<std::size_t>(s.ne1) * sizeof(float));
        }
    }
    ggml_tensor * lin(ggml_tensor * w, ggml_tensor * x, ggml_tensor * b, bool guard = true) {
        ggml_tensor * o = mm_guarded(ctx, w, x, guard);
        return b ? ggml_add(ctx, o, b) : o;
    }
    ggml_tensor * lora_delta(const TSGH3Lora & l, ggml_tensor * x) {
        return tsg::h3_lora_expand(ctx, bind, l, tsg::h3_lora_shrink(ctx, bind, {&l}, x)[0]);
    }

    // h3_vit_rope
    ggml_tensor * rope(ggml_tensor * x, ggml_tensor * cosf, ggml_tensor * sinf, int seq) {
        const int hd = d.hd, heads = d.heads, rot = d.rot;
        if (rot <= 0) return x;
        const std::size_t f = sizeof(float);
        ggml_tensor * xr = ggml_cont(ctx, ggml_view_3d(ctx, x, rot, heads, seq, x->nb[1], x->nb[2], 0));
        const int half = rot / 2;
        ggml_tensor * a = ggml_cont(ctx, ggml_view_3d(ctx, xr, half, heads, seq, xr->nb[1], xr->nb[2], 0));
        ggml_tensor * b = ggml_cont(ctx, ggml_view_3d(ctx, xr, half, heads, seq, xr->nb[1], xr->nb[2],
                                                    static_cast<std::size_t>(half) * f));
        ggml_tensor * rot_half = ggml_concat(ctx, ggml_neg(ctx, b), a, 0);
        ggml_tensor * c3 = ggml_reshape_3d(ctx, cosf, rot, 1, seq);
        ggml_tensor * s3 = ggml_reshape_3d(ctx, sinf, rot, 1, seq);
        ggml_tensor * rotated = ggml_add(ctx, ggml_mul(ctx, xr, c3), ggml_mul(ctx, rot_half, s3));
        if (rot == hd) return rotated;
        ggml_tensor * tail = ggml_cont(ctx, ggml_view_3d(ctx, x, hd - rot, heads, seq, x->nb[1], x->nb[2],
                                                       static_cast<std::size_t>(rot) * f));
        return ggml_concat(ctx, rotated, tail, 0);
    }

    // h3_attend (flash path, explicit fallback) and, when `padded`, h3_attend_padded.
    ggml_tensor * attend(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int seq) {
        const int hd = d.hd, heads = d.heads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        ggml_tensor * qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        const float vScale = flash_v_scale(seq);
        if (padded) {
            const int keys = (seq + 63) / 64 * 64;
            int chunk = 64;
            while (static_cast<long long>((seq + chunk - 1) / chunk) * heads > 65535) chunk *= 2;
            ggml_tensor * vs = vScale == 1.0f ? v : ggml_scale(ctx, v, 1.0f / vScale);
            ggml_tensor * kf = ggml_cast(ctx, ggml_permute(ctx, ggml_pad(ctx, k, 0, 0, keys - seq, 0), 0, 2, 1, 3), GGML_TYPE_F16);
            ggml_tensor * vf = ggml_cast(ctx, ggml_permute(ctx, ggml_pad(ctx, vs, 0, 0, keys - seq, 0), 0, 2, 1, 3), GGML_TYPE_F16);
            ggml_tensor * mask = nullptr;
            for (const Mask & m : masks) if (m.seq == seq) mask = m.mask;
            if (mask == nullptr) {
                ggml_tensor * index = ggml_arange(ctx, 0.0f, static_cast<float>(keys), 1.0f);
                ggml_tensor * is_pad = ggml_clamp(ctx, ggml_scale_bias(ctx, index, 1.0f, static_cast<float>(1 - seq)), 0.0f, 1.0f);
                ggml_tensor * row = ggml_cast(ctx, ggml_scale(ctx, is_pad, -65504.0f), GGML_TYPE_F16);
                mask = ggml_repeat_4d(ctx, ggml_reshape_2d(ctx, row, keys, 1), keys, chunk, 1, 1);
                masks.push_back({seq, mask});
            }
            const std::int64_t chunks = (seq + chunk - 1) / chunk;
            ggml_tensor * qr = ggml_reshape_3d(ctx, ggml_pad(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), 0,
                static_cast<int>(chunks * chunk - seq), 0, 0), hd, chunk, chunks * heads);
            ggml_tensor * out = ggml_flash_attn_ext(ctx, qr, kf, vf, mask, scale, 0.0f, 0.0f);
            ggml_prec_set_acc(out, GGML_PREC_F32);
            ggml_tensor * tokens = ggml_cont(ctx,
                ggml_permute(ctx, ggml_reshape_4d(ctx, out, hd, chunks, heads, chunk), 0, 3, 1, 2));
            ggml_tensor * merged = ggml_view_2d(ctx, tokens, static_cast<std::int64_t>(hd) * heads, seq,
                                                static_cast<std::size_t>(hd) * heads * sizeof(float), 0);
            return vScale == 1.0f ? merged : ggml_scale(ctx, merged, vScale);
        }
        ggml_tensor * vp = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        ggml_tensor * kf = ggml_cast(ctx, kp, GGML_TYPE_F16);
        ggml_tensor * vf = ggml_cast(ctx, vScale == 1.0f ? vp : ggml_scale(ctx, vp, 1.0f / vScale), GGML_TYPE_F16);
        ggml_tensor * out = ggml_flash_attn_ext(ctx, qp, kf, vf, nullptr, scale, 0.0f, 0.0f);
        if (out != nullptr && ggml_backend_supports_op(backend, out)) {
            ggml_prec_set_acc(out, GGML_PREC_F32);
            ggml_tensor * merged = ggml_reshape_2d(ctx, ggml_cont(ctx, out), hd * heads, seq);
            return vScale == 1.0f ? merged : ggml_scale(ctx, merged, vScale);
        }
        ggml_tensor * kq = ggml_mul_mat(ctx, kp, qp);
        ggml_tensor * probs = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor * kqv = ggml_mul_mat(ctx, vt, probs);
        return ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), hd * heads, seq);
    }

    // The DiT's `attention` lambda.
    ggml_tensor * attention(ggml_tensor * h, ggml_tensor * qkvw, ggml_tensor * qkvb, ggml_tensor * qn,
                            ggml_tensor * kn, ggml_tensor * ow, ggml_tensor * ob, int seq,
                            ggml_tensor * cosT, ggml_tensor * sinT, const TSGH3BlockLora * L) {
        const int hd = d.hd, heads = d.heads, inner = d.inner();
        const std::size_t f = sizeof(float);
        ggml_tensor * qkv = lin(qkvw, h, qkvb);
        ggml_tensor * adapted[3] = {nullptr, nullptr, nullptr};
        if (L != nullptr && (tsg::h3_lora_has(L->q) || tsg::h3_lora_has(L->k) || tsg::h3_lora_has(L->v))) {
            const TSGH3Lora * parts[3] = {&L->q, &L->k, &L->v};
            std::vector<ggml_tensor *> shrunk = tsg::h3_lora_shrink(ctx, bind, {&L->q, &L->k, &L->v}, h);
            for (int i = 0; i < 3; i++)
                if (shrunk[i] != nullptr)
                    adapted[i] = ggml_reshape_3d(ctx, tsg::h3_lora_add_slice(ctx,
                        tsg::h3_lora_expand(ctx, bind, *parts[i], shrunk[i]), qkv, inner,
                        static_cast<std::size_t>(i) * inner * f), hd, heads, seq);
        }
        ggml_tensor * q = adapted[0] != nullptr ? adapted[0] : ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
            static_cast<std::size_t>(hd) * f, qkv->nb[1], 0));
        ggml_tensor * k = adapted[1] != nullptr ? adapted[1] : ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
            static_cast<std::size_t>(hd) * f, qkv->nb[1], static_cast<std::size_t>(inner) * f));
        ggml_tensor * v = adapted[2] != nullptr ? adapted[2] : ggml_cont(ctx, ggml_view_3d(ctx, qkv, hd, heads, seq,
            static_cast<std::size_t>(hd) * f, qkv->nb[1], static_cast<std::size_t>(2) * inner * f));
        if (qn) q = ggml_mul(ctx, ggml_rms_norm(ctx, q, d.eps), qn);
        if (kn) k = ggml_mul(ctx, ggml_rms_norm(ctx, k, d.eps), kn);
        if (cosT != nullptr && sinT != nullptr) {
            q = rope(q, cosT, sinT, seq);
            k = rope(k, cosT, sinT, seq);
        }
        ggml_tensor * merged = attend(q, k, v, seq);
        if (L != nullptr && tsg::h3_lora_has(L->out))
            return ggml_add(ctx, lora_delta(L->out, merged), lin(ow, merged, ob));
        return lin(ow, merged, ob);
    }

    // The DiT's `mlp` lambda.
    ggml_tensor * mlp(ggml_tensor * h, ggml_tensor * w1, ggml_tensor * b1, ggml_tensor * w2, ggml_tensor * b2,
                      const TSGH3BlockLora * L) {
        ggml_tensor * act;
        if (L != nullptr && tsg::h3_lora_has(L->fc1)) {
            auto [gate, value] = tsg::h3_lora_fc1(ctx, bind, w1, h, L->fc1, d.ffn, kGuard);
            act = ggml_swiglu_split(ctx, gate, value);
            if (!ggml_backend_supports_op(backend, act)) act = ggml_mul(ctx, ggml_silu(ctx, gate), value);
        } else {
            ggml_tensor * gv = lin(w1, h, b1);
            require(gv->ne[0] == 2 * static_cast<std::int64_t>(d.ffn), "fc1 width is not 2 x ffn");
            act = ggml_swiglu(ctx, gv);
        }
        if (L != nullptr && tsg::h3_lora_has(L->fc2))
            return ggml_add(ctx, lora_delta(L->fc2, act), lin(w2, act, b2));
        return lin(w2, act, b2);
    }

    ggml_tensor * column(ggml_tensor * mods, int col) {
        return ggml_view_2d(ctx, mods, d.hidden, 1, mods->nb[1], static_cast<std::size_t>(col) * mods->nb[1]);
    }
    ggml_tensor * modulate(ggml_tensor * src, ggml_tensor * mods, const std::vector<Seg> & segs, int shiftIdx, int scaleIdx) {
        ggml_tensor * out = nullptr;
        for (const Seg & sg : segs) {
            const int n = sg.end - sg.start;
            if (n <= 0) continue;
            ggml_tensor * part = tsg::h3_token_rows(ctx, src, sg.start, n);
            ggml_tensor * sh = column(mods, sg.col + shiftIdx);
            ggml_tensor * sc = column(mods, sg.col + scaleIdx);
            part = ggml_add(ctx, ggml_add(ctx, part, ggml_mul(ctx, part, sc)), sh);
            out = out == nullptr ? part : ggml_concat(ctx, out, part, 1);
        }
        return out;
    }
    ggml_tensor * gated(ggml_tensor * base, ggml_tensor * upd, ggml_tensor * mods, const std::vector<Seg> & segs, int gateIdx) {
        ggml_tensor * out = nullptr;
        for (const Seg & sg : segs) {
            const int n = sg.end - sg.start;
            if (n <= 0) continue;
            ggml_tensor * b = tsg::h3_token_rows(ctx, base, sg.start, n);
            ggml_tensor * u = tsg::h3_token_rows(ctx, upd, sg.start, n);
            ggml_tensor * part = ggml_add(ctx, b, ggml_mul(ctx, u, column(mods, sg.col + gateIdx)));
            out = out == nullptr ? part : ggml_concat(ctx, out, part, 1);
        }
        return out;
    }

    // One token-refiner block (unmodulated, no RoPE).
    ggml_tensor * refiner(ggml_tensor * x, const BlockW & rw, int seq, const TSGH3BlockLora * L) {
        ggml_tensor * n1w = vec(rw.norm1, d.hidden);
        ggml_tensor * n2w = vec(rw.norm2, d.hidden);
        ggml_tensor * qn = vec(rw.q_norm, d.hd);
        ggml_tensor * kn = vec(rw.k_norm, d.hd);
        ggml_tensor *qkvw, *qkvb, *ow, *ob, *w1, *b1, *w2, *b2;
        decl(rw.qkv, qkvw, qkvb); decl(rw.out, ow, ob);
        decl(rw.fc1, w1, b1); decl(rw.fc2, w2, b2);
        ggml_tensor * n1 = ggml_mul(ctx, ggml_rms_norm(ctx, x, d.eps), n1w);
        x = ggml_add(ctx, x, attention(n1, qkvw, qkvb, qn, kn, ow, ob, seq, nullptr, nullptr, L));
        ggml_tensor * n2 = ggml_mul(ctx, ggml_rms_norm(ctx, x, d.eps), n2w);
        return ggml_add(ctx, x, mlp(n2, w1, b1, w2, b2, L));
    }

    // One modulated trunk block.
    ggml_tensor * trunk(ggml_tensor * x, ggml_tensor * temb, int nts, const BlockW & bw, const std::vector<Seg> & segs,
                        int ntok, ggml_tensor * cosf, ggml_tensor * sinf, const TSGH3BlockLora * L) {
        ggml_tensor * n1w = vec(bw.norm1, d.hidden);
        ggml_tensor * n2w = vec(bw.norm2, d.hidden);
        ggml_tensor * qn = vec(bw.q_norm, d.hd);
        ggml_tensor * kn = vec(bw.k_norm, d.hd);
        ggml_tensor *adw, *adb, *qkvw, *qkvb, *ow, *ob, *w1, *b1, *w2, *b2;
        decl(bw.adaln, adw, adb); decl(bw.qkv, qkvw, qkvb);
        decl(bw.out, ow, ob); decl(bw.fc1, w1, b1); decl(bw.fc2, w2, b2);
        ggml_tensor * mods = ggml_reshape_2d(ctx, lin(adw, temb, adb, false), d.hidden, 18 * nts);
        ggml_tensor * h = ggml_mul(ctx, ggml_rms_norm(ctx, x, d.eps), n1w);
        h = modulate(h, mods, segs, 0, 1);
        ggml_tensor * a = attention(h, qkvw, qkvb, qn, kn, ow, ob, ntok, cosf, sinf, L);
        x = gated(x, a, mods, segs, 2);
        h = ggml_mul(ctx, ggml_rms_norm(ctx, x, d.eps), n2w);
        h = modulate(h, mods, segs, 3, 4);
        ggml_tensor * m = mlp(h, w1, b1, w2, b2, L);
        return gated(x, m, mods, segs, 5);
    }
};

// The DiT's node budget (ggml_ops_minimax_h3.cpp), with the adapter's share.
std::size_t node_budget(int nl, int nref, int nseg, bool adapter) {
    const std::size_t lora = adapter ? static_cast<std::size_t>(tsg::kH3LoraNodesPerBlock) : 0;
    return static_cast<std::size_t>(nl) * (104 + static_cast<std::size_t>(nseg) * 32 + lora) +
           static_cast<std::size_t>(nref) * (104 + lora) + 4096;
}

// A LoRA update whose factors live in g_blobs. `down` may point into a shared blob.
TSGH3Lora make_lora(ggml_type type, int rank, std::int64_t in, std::int64_t out, void * down, void * up) {
    TSGH3Lora l{};
    l.down = down; l.up = up; l.type = type; l.rank = rank; l.in = in; l.out = out;
    return l;
}

// One block's six updates with q/k/v downs packed back to back (the loader's layout).
// upScale 0 gives zero up factors; rank 0 gives no updates at all.
TSGH3BlockLora make_block_lora(ggml_type type, int rank, const Dims & d, std::uint32_t seed, float upScale) {
    TSGH3BlockLora b{};
    if (rank == 0) return b;
    const std::int64_t hidden = d.hidden, inner = d.inner(), ffn = d.ffn;
    std::vector<float> qkvDown = values(static_cast<std::size_t>(3 * rank * hidden), seed, 0.5f / std::sqrt(float(hidden)));
    Blob & packed = keep(type, qkvDown, hidden);
    const std::size_t partBytes = ggml_row_size(type, hidden) * static_cast<std::size_t>(rank);
    auto up = [&](std::int64_t out, std::uint32_t s) -> void * {
        const std::size_t n = static_cast<std::size_t>(rank * out);
        return keep(type, upScale == 0.0f ? std::vector<float>(n, 0.0f)   // +0, not a mix of signed zeros
                                          : values(n, s, upScale / std::sqrt(float(rank))), rank).data();
    };
    auto down = [&](std::int64_t in, std::uint32_t s) -> void * {
        return keep(type, values(static_cast<std::size_t>(rank * in), s, 0.5f / std::sqrt(float(in))), in).data();
    };
    std::uint8_t * base = packed.bytes.data();
    b.q = make_lora(type, rank, hidden, inner, base, up(inner, seed + 1));
    b.k = make_lora(type, rank, hidden, inner, base + partBytes, up(inner, seed + 2));
    b.v = make_lora(type, rank, hidden, inner, base + 2 * partBytes, up(inner, seed + 3));
    b.out = make_lora(type, rank, inner, hidden, down(inner, seed + 4), up(hidden, seed + 5));
    b.fc1 = make_lora(type, rank, hidden, 2 * ffn, down(hidden, seed + 6), up(2 * ffn, seed + 7));
    b.fc2 = make_lora(type, rank, ffn, hidden, down(ffn, seed + 8), up(hidden, seed + 9));
    return b;
}

// Host weights of one block at small widths; projections in `wtype`.
BlockW make_block(const Dims & d, ggml_type wtype, std::uint32_t seed, bool modulated) {
    const std::int64_t hidden = d.hidden, inner = d.inner(), ffn = d.ffn;
    auto lin = [&](std::int64_t in, std::int64_t out, bool bias, std::uint32_t s, ggml_type t) {
        Lin l;
        l.ne0 = in; l.ne1 = out; l.type = t;
        l.w = keep(t, values(static_cast<std::size_t>(in * out), s, 1.0f / std::sqrt(float(in))), in).data();
        if (bias) l.b = keep(GGML_TYPE_F32, values(static_cast<std::size_t>(out), s + 100, 0.1f), out).data();
        return l;
    };
    auto norm = [&](std::int64_t n, std::uint32_t s) {
        std::vector<float> v = values(static_cast<std::size_t>(n), s, 0.2f);
        for (float & x : v) x += 1.0f;
        return keep(GGML_TYPE_F32, v, n).data();
    };
    BlockW w;
    w.norm1 = norm(hidden, seed); w.norm2 = norm(hidden, seed + 1);
    w.q_norm = norm(d.hd, seed + 2); w.k_norm = norm(d.hd, seed + 3);
    if (modulated) w.adaln = lin(d.ted, 18 * hidden, true, seed + 4, GGML_TYPE_F32);
    w.qkv = lin(hidden, 3 * inner, true, seed + 5, wtype);
    w.out = lin(inner, hidden, true, seed + 6, wtype);
    w.fc1 = lin(hidden, 2 * ffn, false, seed + 7, wtype);   // no released fc1 has a bias
    w.fc2 = lin(ffn, hidden, true, seed + 8, wtype);
    return w;
}

// ---- 1: expand/shrink against a double reference ---------------------------

void case_product(ggml_backend_t backend) {
    const std::int64_t in = 256, out = 96, rank = 16, seq = 5;
    const ggml_type bases[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K};
    double worstDeltaF32 = 0, worstDeltaF16 = 0, worstBase = 0;
    for (ggml_type ft : {GGML_TYPE_F32, GGML_TYPE_F16})
        for (ggml_type wt : bases)
            for (bool guard : {false, true}) {
                Blob & W = keep(wt, values(in * out, 11, 1.0f / std::sqrt(float(in))), in);
                Blob & A = keep(ft, values(in * rank, 12, 0.5f / std::sqrt(float(in))), in);
                Blob & B = keep(ft, values(rank * out, 13, 2.0f / std::sqrt(float(rank))), rank);
                std::vector<float> X = values(in * seq, 14, 4.0f);
                TSGH3Lora l = make_lora(ft, rank, in, out, A.data(), B.data());
                Run r(backend);
                ggml_tensor * x = r.input(GGML_TYPE_F32, in, seq, X.data());
                ggml_tensor * w = ggml_new_tensor_2d(r.ctx, wt, in, out);
                r.bind(w, W.data(), ggml_nbytes(w));
                ggml_tensor * base = mm_guarded(r.ctx, w, x, guard);
                ggml_tensor * delta = tsg::h3_lora_expand(r.ctx, r.bind, l, tsg::h3_lora_shrink(r.ctx, r.bind, {&l}, x)[0]);
                ggml_tensor * y = ggml_add(r.ctx, delta, base);
                r.build({base, y});
                r.compute();
                const std::vector<float> got0 = r.read(base), got = r.read(y);
                const std::vector<double> xs = widen(X);
                const std::vector<double> baseRef = matmul(W.decoded, in, out, xs, seq);
                const std::vector<double> deltaRef = matmul(B.decoded, rank, out, matmul(A.decoded, in, rank, xs, seq), seq);
                std::vector<float> d(got.size());
                for (std::size_t i = 0; i < got.size(); ++i) d[i] = got[i] - got0[i];
                const double eb = rel_err(got0, baseRef), ed = rel_err(d, deltaRef);
                const double tb = wt == GGML_TYPE_F32 ? 1e-4 : wt == GGML_TYPE_F16 ? 4e-3 : 3e-2;
                const double td = ft == GGML_TYPE_F32 ? 1e-4 : 5e-3;
                const std::string what = std::string(ggml_type_name(wt)) + " base, " + ggml_type_name(ft) +
                                         " factors, guard " + (guard ? "on" : "off");
                require(eb <= tb, what + ": base differs from the double reference by " + num(eb));
                require(ed <= td, what + ": LoRA term differs from the double reference by " + num(ed));
                (ft == GGML_TYPE_F32 ? worstDeltaF32 : worstDeltaF16) =
                    std::max(ft == GGML_TYPE_F32 ? worstDeltaF32 : worstDeltaF16, ed);
                if (wt == GGML_TYPE_F32 || wt == GGML_TYPE_F16) worstBase = std::max(worstBase, eb);
            }
    report("productDeltaRelF32", num(worstDeltaF32));
    report("productDeltaRelF16", num(worstDeltaF16));
    report("productBaseRelFloat", num(worstBase));
}

// ---- 2: stacked vs separate shrink ------------------------------------------

void case_stacked(ggml_backend_t backend) {
    const std::int64_t in = 256, inner = 64, seq = 9;
    const int ranks[3] = {16, 32, 16};
    double worst = 0;
    for (ggml_type ft : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        std::vector<float> all;
        std::vector<std::vector<float>> parts(3);
        for (int i = 0; i < 3; ++i) {
            parts[i] = values(static_cast<std::size_t>(ranks[i] * in), 21 + i, 0.1f);
            all.insert(all.end(), parts[i].begin(), parts[i].end());
        }
        Blob & stacked = keep(ft, all, in);
        void * separate[3];
        void * ups[3];
        for (int i = 0; i < 3; ++i) {
            separate[i] = keep(ft, parts[i], in).data();
            ups[i] = keep(ft, values(static_cast<std::size_t>(inner * ranks[i]), 31 + i, 0.3f), ranks[i]).data();
        }
        std::vector<float> X = values(in * seq, 41, 2.0f);
        std::vector<std::vector<float>> results[2];
        int shrinks[2];
        for (int variant = 0; variant < 2; ++variant) {
            TSGH3Lora l[3];
            std::size_t offset = 0;
            for (int i = 0; i < 3; ++i) {
                void * down = variant == 0 ? static_cast<void *>(stacked.bytes.data() + offset) : separate[i];
                l[i] = make_lora(ft, ranks[i], in, inner, down, ups[i]);
                offset += ggml_row_size(ft, in) * static_cast<std::size_t>(ranks[i]);
            }
            Run r(backend);
            ggml_tensor * x = r.input(GGML_TYPE_F32, in, seq, X.data());
            std::vector<ggml_tensor *> s = tsg::h3_lora_shrink(r.ctx, r.bind, {&l[0], &l[1], &l[2]}, x);
            std::vector<ggml_tensor *> outs;
            for (int i = 0; i < 3; ++i) outs.push_back(tsg::h3_lora_expand(r.ctx, r.bind, l[i], s[i]));
            r.build(outs);
            r.compute();
            shrinks[variant] = r.count(GGML_OP_MUL_MAT) - 3;
            for (ggml_tensor * o : outs) results[variant].push_back(r.read(o));
        }
        require(shrinks[0] == 1 && shrinks[1] == 3, "stacked downs did not run as one shrink (" +
                std::to_string(shrinks[0]) + " vs " + std::to_string(shrinks[1]) + ")");
        // ggml-cuda may run the wider stacked GEMM on another cuBLAS kernel: recorded there.
        for (int i = 0; i < 3; ++i)
            worst = std::max(worst, check_identity(results[1][i], results[0][i],
                std::string("stacked shrink, part ") + "qkv"[i] + " (" + ggml_type_name(ft) + ")", 1e-5));
    }

    // Skipped parts: null and rank 0 give no tensor; a rank-0 part breaks a run.
    {
        Run r(backend);
        std::vector<float> X = values(256 * 3, 43, 1.0f);
        Blob & down = keep(GGML_TYPE_F32, values(256 * 32, 44, 0.1f), 256);
        TSGH3Lora q = make_lora(GGML_TYPE_F32, 16, 256, 64, down.data(), nullptr);
        TSGH3Lora none = make_lora(GGML_TYPE_F32, 0, 256, 64, nullptr, nullptr);
        TSGH3Lora v = make_lora(GGML_TYPE_F32, 16, 256, 64, down.bytes.data() + 16 * 256 * sizeof(float), nullptr);
        ggml_tensor * x = r.input(GGML_TYPE_F32, 256, 3, X.data());
        std::vector<ggml_tensor *> s = tsg::h3_lora_shrink(r.ctx, r.bind, {&q, nullptr, &none}, x);
        require(s.size() == 3 && s[0] != nullptr && s[1] == nullptr && s[2] == nullptr, "skipped parts produced tensors");
        s = tsg::h3_lora_shrink(r.ctx, r.bind, {&q, &none, &v}, x);
        require(s[0] != nullptr && s[1] == nullptr && s[2] != nullptr && s[0] != s[2] &&
                s[0]->op == GGML_OP_MUL_MAT && s[2]->op == GGML_OP_MUL_MAT,
                "a rank-0 part between two updates did not split the run");
        bool threw = false;
        try { tsg::h3_lora_shrink(r.ctx, r.bind, {&q}, r.input(GGML_TYPE_F32, 128, 3, X.data())); }
        catch (const std::invalid_argument &) { threw = true; }
        require(threw, "a down factor of the wrong width was accepted");
    }
    report("stackedShrinkMaxRel", num(worst));
}

// ---- 3: the folds -------------------------------------------------------------

void case_folds(ggml_backend_t backend) {
    // (a) the slice add stands in for cont(view) + add.
    {
        const std::int64_t width = 64, seq = 7;
        std::vector<float> P = values(3 * width * seq, 51, 3.0f), D = values(width * seq, 52, 0.5f);
        Run r(backend);
        ggml_tensor * packed = r.input(GGML_TYPE_F32, 3 * width, seq, P.data());
        ggml_tensor * delta = r.input(GGML_TYPE_F32, width, seq, D.data());
        const std::size_t offset = static_cast<std::size_t>(width) * sizeof(float);
        ggml_tensor * folded = tsg::h3_lora_add_slice(r.ctx, delta, packed, width, offset);
        ggml_tensor * plain = ggml_add(r.ctx, delta,
            ggml_cont(r.ctx, ggml_view_2d(r.ctx, packed, width, seq, packed->nb[1], offset)));
        r.build({folded, plain});
        r.compute();
        require(same_bits(r.read(folded), r.read(plain)), "the slice add differs from cont + add");
    }
    // (b) strength * alpha / rank = 8 / 128 folded into up: a power of two, so the
    // folded factor's product equals the scaled product exactly.
    for (ggml_type ft : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        const std::int64_t in = 128, out = 64, rank = 128, seq = 5;
        const float fold = 8.0f / 128.0f;
        std::vector<float> up = values(rank * out, 53, 1.0f);
        for (float & u : up) u = std::copysign(0.05f + std::fabs(u), u);   // clear of F16 subnormals once folded
        std::vector<float> folded = up;
        for (float & u : folded) u *= fold;
        Blob & down = keep(ft, values(in * rank, 54, 0.2f), in);
        TSGH3Lora raw = make_lora(ft, static_cast<int>(rank), in, out, down.data(), keep(ft, up, rank).data());
        TSGH3Lora pre = make_lora(ft, static_cast<int>(rank), in, out, down.data(), keep(ft, folded, rank).data());
        std::vector<float> X = values(in * seq, 55, 1.0f);
        Run r(backend);
        ggml_tensor * x = r.input(GGML_TYPE_F32, in, seq, X.data());
        ggml_tensor * s = tsg::h3_lora_shrink(r.ctx, r.bind, {&raw}, x)[0];
        ggml_tensor * a = tsg::h3_lora_expand(r.ctx, r.bind, pre, s);
        ggml_tensor * b = ggml_scale(r.ctx, tsg::h3_lora_expand(r.ctx, r.bind, raw, s), fold);
        r.build({a, b});
        r.compute();
        require(same_bits(r.read(a), r.read(b)),
                std::string("folding alpha / rank into up changed the product (") + ggml_type_name(ft) + ")");
    }
    report("folds", "\"bit-equal\"");
}

// ---- 4: split fc1 vs swiglu(gv + delta) ----------------------------------------

void case_fc1(ggml_backend_t backend) {
    const std::int64_t hidden = 256, ffn = 128, rank = 16, seq = 7;
    double worst = 0, worstFallback = 0;
    for (ggml_type wt : {GGML_TYPE_Q8_0, GGML_TYPE_F16, GGML_TYPE_F32})
        for (ggml_type ft : {GGML_TYPE_F32, GGML_TYPE_F16})
            for (int which = 0; which < 3; ++which) {   // full, gate-only, value-only
                Blob & W1 = keep(wt, values(hidden * 2 * ffn, 61, 1.0f / std::sqrt(float(hidden))), hidden);
                Blob & A = keep(ft, values(hidden * rank, 62, 0.5f / std::sqrt(float(hidden))), hidden);
                std::vector<float> up = values(rank * 2 * ffn, 63, 1.0f / std::sqrt(float(rank)));
                // Rows [0, ffn) of up drive the gate, [ffn, 2 ffn) the value.
                if (which == 1) std::fill(up.begin() + rank * ffn, up.end(), 0.0f);
                if (which == 2) std::fill(up.begin(), up.begin() + rank * ffn, 0.0f);
                Blob & B = keep(ft, up, rank);
                TSGH3Lora l = make_lora(ft, static_cast<int>(rank), hidden, 2 * ffn, A.data(), B.data());
                std::vector<float> X = values(hidden * seq, 64, 3.0f);
                Run r(backend);
                ggml_tensor * x = r.input(GGML_TYPE_F32, hidden, seq, X.data());
                ggml_tensor * w1 = ggml_new_tensor_2d(r.ctx, wt, hidden, 2 * ffn);
                r.bind(w1, W1.data(), ggml_nbytes(w1));
                auto [gate, value] = tsg::h3_lora_fc1(r.ctx, r.bind, w1, x, l, ffn, kGuard);
                ggml_tensor * split = ggml_swiglu_split(r.ctx, gate, value);
                require(ggml_backend_supports_op(backend, split), "swiglu_split is unsupported");
                ggml_tensor * fallback = ggml_mul(r.ctx, ggml_silu(r.ctx, gate), value);
                ggml_tensor * gv = ggml_add(r.ctx, mm_guarded(r.ctx, w1, x, true),
                    tsg::h3_lora_expand(r.ctx, r.bind, l, tsg::h3_lora_shrink(r.ctx, r.bind, {&l}, x)[0]));
                ggml_tensor * naive = ggml_swiglu(r.ctx, gv);
                r.build({split, fallback, naive});
                r.compute();
                const std::vector<float> a = r.read(split), b = r.read(naive), c = r.read(fallback);
                const std::string what = std::string("fc1 split (") + ggml_type_name(wt) + " fc1, " + ggml_type_name(ft) +
                                         " factors, " + (which == 0 ? "full" : which == 1 ? "gate-only" : "value-only") + ")";
                worst = std::max(worst, check_identity(a, b, what, 1e-4));
                const double ef = rel_err(c, a);
                require(ef <= 1e-5, what + ": silu * value fallback differs by " + num(ef));
                worstFallback = std::max(worstFallback, ef);
            }
    report("fc1SplitVsNaiveRel", num(worst));
    report("fc1FallbackRel", num(worstFallback));
}

// ---- 5-7: whole blocks ----------------------------------------------------------

struct BlockRun {
    std::vector<float> refined, trunk;
    int nodes = 0, leafs = 0;
    std::vector<ggml_op> ops;
};

// A refiner block over `text` and two trunk blocks over `x`, as the DiT chains them.
BlockRun run_blocks(ggml_backend_t backend, const Dims & d, const BlockW & refW, const BlockW (&trunkW)[2],
                    const TSGH3DitAdapter * adapter, bool computeIt) {
    const int ntok = 37, ntext = 11, nts = 2;
    const std::vector<Seg> segs = {{0, 10, 0}, {10, 25, 12}, {25, 37, 24}};
    static std::vector<float> X = values(static_cast<std::size_t>(d.hidden) * ntok, 71, 1.0f);
    static std::vector<float> T = values(static_cast<std::size_t>(d.hidden) * ntext, 72, 1.0f);
    static std::vector<float> E = values(static_cast<std::size_t>(d.ted) * nts, 73, 1.0f);
    static std::vector<float> C, S;
    if (C.empty())
        for (int t = 0; t < ntok; ++t)
            for (int j = 0; j < d.rot; ++j) {
                const float angle = t * std::pow(10000.0f, -float(j % (d.rot / 2)) / float(d.rot / 2));
                C.push_back(std::cos(angle)); S.push_back(std::sin(angle));
            }
    Run r(backend);
    Mirror m(r.ctx, backend, r.bind, d);
    ggml_tensor * x = r.input(GGML_TYPE_F32, d.hidden, ntok, X.data());
    ggml_tensor * text = r.input(GGML_TYPE_F32, d.hidden, ntext, T.data());
    ggml_tensor * temb = r.input(GGML_TYPE_F32, d.ted, nts, E.data());
    ggml_tensor * cosf = r.input(GGML_TYPE_F32, d.rot, ntok, C.data());
    ggml_tensor * sinf = r.input(GGML_TYPE_F32, d.rot, ntok, S.data());
    ggml_tensor * refined = m.refiner(text, refW, ntext, adapter ? &adapter->refiner[0] : nullptr);
    for (int l = 0; l < 2; ++l)
        x = m.trunk(x, temb, nts, trunkW[l], segs, ntok, cosf, sinf, adapter ? &adapter->blocks[l] : nullptr);
    r.build({refined, x});
    BlockRun out;
    out.nodes = ggml_graph_n_nodes(r.graph);
    out.leafs = count_leafs(r.graph);
    for (int i = 0; i < out.nodes; ++i) out.ops.push_back(ggml_graph_node(r.graph, i)->op);
    if (computeIt) {
        r.compute();
        out.refined = r.read(refined);
        out.trunk = r.read(x);
    }
    return out;
}

void case_blocks(ggml_backend_t backend) {
    const Dims d{128, 2, 64, 256, 48, 8};
    const BlockW refW = make_block(d, GGML_TYPE_Q8_0, 100, false);
    const BlockW trunkW[2] = {make_block(d, GGML_TYPE_Q8_0, 200, true), make_block(d, GGML_TYPE_Q8_0, 300, true)};
    auto adapterOf = [](std::vector<TSGH3BlockLora> & blocks, std::vector<TSGH3BlockLora> & refiner) {
        TSGH3DitAdapter a{};
        a.struct_bytes = sizeof(TSGH3DitAdapter);
        a.num_blocks = static_cast<std::int32_t>(blocks.size());
        a.num_refiner_blocks = static_cast<std::int32_t>(refiner.size());
        a.blocks = blocks.data();
        a.refiner = refiner.data();
        return a;
    };
    const BlockRun none = run_blocks(backend, d, refW, trunkW, nullptr, true);

    // 5: zero up factors, F32 and F16.
    double worstZero = 0;
    for (ggml_type ft : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        std::vector<TSGH3BlockLora> blocks = {make_block_lora(ft, 16, d, 400, 0.0f), make_block_lora(ft, 16, d, 410, 0.0f)};
        std::vector<TSGH3BlockLora> refiner = {make_block_lora(ft, 16, d, 420, 0.0f)};
        const TSGH3DitAdapter a = adapterOf(blocks, refiner);
        require(tsg::h3_lora_validate(&a, 2, 1, d.hidden, d.inner(), d.ffn).empty(), "the test adapter is invalid");
        const BlockRun zero = run_blocks(backend, d, refW, trunkW, &a, true);
        const std::string what = std::string("zero-up adapter (") + ggml_type_name(ft) + ")";
        worstZero = std::max(worstZero, check_identity(zero.refined, none.refined, what + " refiner block"));
        worstZero = std::max(worstZero, check_identity(zero.trunk, none.trunk, what + " trunk blocks"));
    }

    // Real factors must change both outputs: the adapter is wired in at all.
    std::vector<TSGH3BlockLora> blocks = {make_block_lora(GGML_TYPE_F32, 16, d, 430, 1.0f),
                                          make_block_lora(GGML_TYPE_F32, 16, d, 440, 1.0f)};
    std::vector<TSGH3BlockLora> refiner = {make_block_lora(GGML_TYPE_F32, 16, d, 450, 1.0f)};
    const TSGH3DitAdapter full = adapterOf(blocks, refiner);
    const BlockRun applied = run_blocks(backend, d, refW, trunkW, &full, true);
    const double moved = std::min(rel_err(applied.refined, none.refined), rel_err(applied.trunk, none.trunk));
    require(moved > 1e-3, "an adapter with real factors left the blocks unchanged (" + num(moved) + ")");

    // 6: all ranks 0 -> the same graph and bits.
    std::vector<TSGH3BlockLora> rank0Blocks(2), rank0Refiner(1);
    for (TSGH3BlockLora * b : {&rank0Blocks[0], &rank0Blocks[1], &rank0Refiner[0]})
        for (TSGH3Lora * l : {&b->q, &b->k, &b->v, &b->out, &b->fc1, &b->fc2}) {
            l->type = 8; l->in = 1; l->out = 1;   // ignored at rank 0, as the validator ignores them
        }
    const TSGH3DitAdapter rank0 = adapterOf(rank0Blocks, rank0Refiner);
    require(tsg::h3_lora_validate(&rank0, 2, 1, d.hidden, d.inner(), d.ffn).empty(), "a rank-0 adapter was refused");
    const BlockRun r0 = run_blocks(backend, d, refW, trunkW, &rank0, true);
    require(r0.nodes == none.nodes && r0.leafs == none.leafs && r0.ops == none.ops,
            "a rank-0 adapter changed the graph (" + std::to_string(r0.nodes) + " vs " + std::to_string(none.nodes) + " nodes)");
    require(same_bits(r0.refined, none.refined) && same_bits(r0.trunk, none.trunk), "a rank-0 adapter changed the output bits");

    // 7: nodes and leaves one block's six updates add (3 blocks here: refiner + 2 trunk).
    const int addedNodes = applied.nodes - none.nodes, addedLeafs = applied.leafs - none.leafs;
    require(addedNodes <= 3 * tsg::kH3LoraNodesPerBlock && addedLeafs <= 3 * tsg::kH3LoraNodesPerBlock,
            "an adapter added " + std::to_string(addedNodes) + " nodes and " + std::to_string(addedLeafs) +
            " leaves over 3 blocks; the budget is " + std::to_string(tsg::kH3LoraNodesPerBlock) + " per block");
    report("zeroUpMaxRel", num(worstZero));
    report("zeroUpBitEqual", g_exact ? "true" : "false");
    report("rank0Identity", "true");
    report("loraNodesPerBlock", num(addedNodes / 3.0));
    report("loraLeavesPerBlock", num(addedLeafs / 3.0));
    report("loraNodesPerBlockBudget", std::to_string(tsg::kH3LoraNodesPerBlock));
}

// ---- measure-only graphs at H3's sizes ----------------------------------------------

const Dims kH3{5376, 56, 128, 14336, 96, 8};

// Distinct non-null addresses that are never read: graphs measured, not computed.
struct FakeAddresses {
    static std::uint8_t base[64];
    std::uintptr_t next = reinterpret_cast<std::uintptr_t>(base);
    void * take(std::size_t bytes) { void * p = reinterpret_cast<void *>(next); next += bytes; return p; }
};
std::uint8_t FakeAddresses::base[64];

// A block's six rank-`rank` F16 updates with q/k/v downs back to back, as packed.
TSGH3BlockLora fake_block_lora(FakeAddresses & a, const Dims & d, int rank) {
    auto lora = [&](std::int64_t in, std::int64_t out) {
        void * down = a.take(ggml_row_size(GGML_TYPE_F16, in) * rank);
        return make_lora(GGML_TYPE_F16, rank, in, out, down, a.take(ggml_row_size(GGML_TYPE_F16, rank) * out));
    };
    TSGH3BlockLora b{};
    const std::size_t part = ggml_row_size(GGML_TYPE_F16, d.hidden) * rank;
    void * qkv = a.take(3 * part);
    b.q = lora(d.hidden, d.inner()); b.k = lora(d.hidden, d.inner()); b.v = lora(d.hidden, d.inner());
    b.q.down = qkv;
    b.k.down = reinterpret_cast<void *>(reinterpret_cast<std::uintptr_t>(qkv) + part);
    b.v.down = reinterpret_cast<void *>(reinterpret_cast<std::uintptr_t>(qkv) + 2 * part);
    b.out = lora(d.inner(), d.hidden); b.fc1 = lora(d.hidden, 2 * d.ffn); b.fc2 = lora(d.ffn, d.hidden);
    return b;
}

// A block's weights as the released GGUFs carry them: quantized projections, so
// every h3_mm guard is built.
BlockW fake_block(FakeAddresses & a, const Dims & d) {
    auto lin = [&](std::int64_t in, std::int64_t out, ggml_type type, bool bias) {
        Lin l; l.w = a.take(1); l.b = bias ? a.take(1) : nullptr; l.ne0 = in; l.ne1 = out; l.type = type; return l;
    };
    BlockW bw;
    bw.norm1 = a.take(1); bw.norm2 = a.take(1); bw.q_norm = a.take(1); bw.k_norm = a.take(1);
    bw.adaln = lin(d.ted, 18 * d.hidden, GGML_TYPE_F32, true);
    bw.qkv = lin(d.hidden, 3 * d.inner(), GGML_TYPE_Q4_K, true);
    bw.out = lin(d.inner(), d.hidden, GGML_TYPE_Q4_K, true);
    bw.fc1 = lin(d.hidden, 2 * d.ffn, GGML_TYPE_Q4_K, false);
    bw.fc2 = lin(d.ffn, d.hidden, GGML_TYPE_Q4_K, true);
    return bw;
}

// gallocr's compute buffer for `graph`, without allocating it.
std::size_t buffer_bytes(ggml_backend_t backend, ggml_cgraph * graph) {
    ggml_gallocr_t allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    std::size_t size = 0;
    ggml_gallocr_reserve_n_size(allocator, graph, nullptr, nullptr, &size);
    ggml_gallocr_free(allocator);
    return size;
}

// Weights bound as resident (a data pointer, as the DiT's cached weights have), so
// gallocr's buffer holds activations only.
const tsg::H3Bind kResident = [](ggml_tensor * t, void * data, std::size_t) { t->data = data; };

// ---- 8: compute buffer at seq 2048 -----------------------------------------------

// The MLP at H3's widths with Q8_0 weights (so h3_mm's guard runs), measured by
// gallocr without allocating: weights are marked resident, as the DiT's cached
// weights are, so the buffer holds activations only.
std::size_t mlp_buffer(ggml_backend_t backend, int variant) {   // 0 base, 1 split, 2 naive
    const std::int64_t hidden = 5376, ffn = 14336, rank = 128, seq = 2048;
    ggml_context * ctx = ggml_init({8u << 20, nullptr, true});
    require(ctx != nullptr, "context allocation failed");
    FakeAddresses fake;
    const tsg::H3Bind & resident = kResident;
    auto weight = [&](ggml_type type, std::int64_t ne0, std::int64_t ne1) {
        ggml_tensor * t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
        resident(t, fake.take(1), ggml_nbytes(t));
        return t;
    };
    ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, seq);
    ggml_set_input(in);
    ggml_tensor * x = ggml_rms_norm(ctx, in, 1e-5f);   // the modulated norm the MLP reads
    ggml_tensor * w1 = weight(GGML_TYPE_Q8_0, hidden, 2 * ffn);
    ggml_tensor * w2 = weight(GGML_TYPE_Q8_0, ffn, hidden);
    TSGH3Lora l = make_lora(GGML_TYPE_F16, static_cast<int>(rank), hidden, 2 * ffn, fake.take(1), fake.take(1));
    ggml_tensor * act;
    if (variant == 1) {
        auto [gate, value] = tsg::h3_lora_fc1(ctx, resident, w1, x, l, ffn, kGuard);
        act = ggml_swiglu_split(ctx, gate, value);
    } else {
        ggml_tensor * gv = mm_guarded(ctx, w1, x, true);
        if (variant == 2)
            gv = ggml_add(ctx, gv, tsg::h3_lora_expand(ctx, resident, l, tsg::h3_lora_shrink(ctx, resident, {&l}, x)[0]));
        act = ggml_swiglu(ctx, gv);
    }
    ggml_tensor * out = mm_guarded(ctx, w2, act, true);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    const std::size_t size = buffer_bytes(backend, graph);
    ggml_free(ctx);
    return size;
}

// One whole trunk block at H3's widths: attention and MLP, with or without the
// six updates.
std::size_t block_buffer(ggml_backend_t backend, bool adapter) {
    const int seq = 2048, nts = 4;
    ggml_context * ctx = ggml_init({16u << 20, nullptr, true});
    require(ctx != nullptr, "context allocation failed");
    FakeAddresses fake;
    Mirror m(ctx, backend, kResident, kH3);
    const BlockW bw = fake_block(fake, kH3);
    const TSGH3BlockLora lora = fake_block_lora(fake, kH3, 128);
    const std::vector<Seg> segs = {{0, 200, 0}, {200, 520, 24}, {520, seq, 48}};
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kH3.hidden, seq);
    ggml_tensor * temb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kH3.ted, nts);
    ggml_tensor * cosf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kH3.rot, seq);
    ggml_tensor * sinf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kH3.rot, seq);
    for (ggml_tensor * t : {x, temb, cosf, sinf}) ggml_set_input(t);
    ggml_tensor * out = m.trunk(x, temb, nts, bw, segs, seq, cosf, sinf, adapter ? &lora : nullptr);
    ggml_set_output(out);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, out);
    const std::size_t size = buffer_bytes(backend, graph);
    ggml_free(ctx);
    return size;
}

void case_buffers(ggml_backend_t backend) {
    const std::size_t base = mlp_buffer(backend, 0), split = mlp_buffer(backend, 1), naive = mlp_buffer(backend, 2);
    require(static_cast<double>(split) <= static_cast<double>(base) * 1.05,
            "split fc1 needs " + std::to_string(split) + " bytes, over 1.05 x the fused " + std::to_string(base));
    require(static_cast<double>(base) * 1.05 < static_cast<double>(naive),
            "gv + delta (" + std::to_string(naive) + " bytes) is not above 1.05 x the fused path (" + std::to_string(base) + ")");
    report("mlpBufferBaseMiB", num(base / 1048576.0));
    report("mlpBufferSplitMiB", num(split / 1048576.0));
    report("mlpBufferNaiveMiB", num(naive / 1048576.0));
    // The whole block: the out and fc2 deltas and the q/k/v terms on top of the split fc1.
    const std::size_t blockBase = block_buffer(backend, false), blockLora = block_buffer(backend, true);
    require(static_cast<double>(blockLora) <= static_cast<double>(blockBase) * 1.05,
            "a block with all six updates needs " + std::to_string(blockLora) + " bytes, over 1.05 x " +
            std::to_string(blockBase));
    report("blockBufferBaseMiB", num(blockBase / 1048576.0));
    report("blockBufferLoraMiB", num(blockLora / 1048576.0));
}

// ---- 9: graph metadata of the full-depth graph -------------------------------------

struct ArenaUse { std::size_t used, nodes, leafs, budget; };

// TSGgml_MiniMaxH3DitForward's whole graph at H3's sizes in a no_alloc context of
// exactly the 32 MiB it gets, with the adapter on every block.
ArenaUse arena_use(ggml_backend_t backend, int nseg, bool padded) {
    constexpr std::size_t kArena = 32ull * 1024 * 1024;
    const Dims d = kH3;
    const int nl = 50, nref = 2, nts = 4, rank = 128;
    const int text = 301, cond = 2 * 1240, audio = 643, video = 34991 - text - cond - audio;
    const int ntok = text + cond + audio + video;
    const int vpd = 192, ach = 128, tdim = 5120;
    ggml_context * ctx = ggml_init({kArena, nullptr, true});
    require(ctx != nullptr, "context allocation failed");
    tsg::H3Bind none = [](ggml_tensor *, void *, std::size_t) {};
    Mirror m(ctx, backend, none, d);
    m.padded = padded;

    FakeAddresses fake;
    auto addr = [&](std::size_t bytes) { return fake.take(bytes); };
    std::vector<TSGH3BlockLora> blocks(nl), refiner(nref);
    for (auto & b : blocks) b = fake_block_lora(fake, d, rank);
    for (auto & b : refiner) b = fake_block_lora(fake, d, rank);
    TSGH3DitAdapter adapter{static_cast<std::int32_t>(sizeof(TSGH3DitAdapter)), nl, nref, 0, blocks.data(), refiner.data()};
    require(tsg::h3_lora_validate(&adapter, nl, nref, d.hidden, d.inner(), d.ffn).empty(), "the arena adapter is invalid");

    auto lin = [&](std::int64_t in, std::int64_t out, ggml_type type, bool bias) {
        Lin l; l.w = addr(1); l.b = bias ? addr(1) : nullptr; l.ne0 = in; l.ne1 = out; l.type = type; return l;
    };
    const BlockW bw = fake_block(fake, d);

    std::vector<Seg> segs;
    for (int i = 0; i < nseg; ++i) {
        const int start = static_cast<int>(static_cast<long long>(ntok) * i / nseg);
        const int end = static_cast<int>(static_cast<long long>(ntok) * (i + 1) / nseg);
        segs.push_back({start, end, ((i % nts) * 3 + i % 3) * 6});
    }

    // inputs
    ggml_tensor * vin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vpd, cond + video);
    ggml_tensor * ain = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ach, audio);
    ggml_tensor * temb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.ted, nts);
    ggml_tensor * cosf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.rot, ntok);
    ggml_tensor * sinf = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d.rot, ntok);
    ggml_tensor * tin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, tdim, text);
    ggml_tensor * voutT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vpd, video);
    ggml_tensor * aoutT = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ach, audio);
    for (ggml_tensor * t : {vin, ain, temb, cosf, sinf, tin}) ggml_set_input(t);

    // text: project, then refine
    ggml_tensor *w, *b;
    m.decl(lin(tdim, d.hidden, GGML_TYPE_F16, true), w, b);
    ggml_tensor * ctxTok = m.lin(w, tin, b);
    for (int i = 0; i < nref; ++i) ctxTok = m.refiner(ctxTok, bw, text, &adapter.refiner[i]);
    ctxTok = ggml_mul(ctx, ggml_rms_norm(ctx, ctxTok, d.eps), m.vec(addr(1), d.hidden));

    // patchify and pack: text, conditioning frames, audio, video
    ggml_tensor *vpw, *vpb, *apw, *apb;
    m.decl(lin(vpd, d.hidden, GGML_TYPE_F32, true), vpw, vpb);
    m.decl(lin(ach, d.hidden, GGML_TYPE_F32, true), apw, apb);
    ggml_tensor * allVideo = m.lin(vpw, vin, vpb);
    ggml_tensor * audioTok = m.lin(apw, ain, apb);
    ggml_tensor * condTok = tsg::h3_token_rows(ctx, allVideo, 0, cond);
    ggml_tensor * videoTok = tsg::h3_token_rows(ctx, allVideo, cond, video);
    ggml_tensor * x = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, ctxTok, condTok, 1), audioTok, 1), videoTok, 1);

    for (int l = 0; l < nl; ++l) x = m.trunk(x, temb, nts, bw, segs, ntok, cosf, sinf, &adapter.blocks[l]);

    // final layer
    ggml_tensor *faw, *fab;
    m.decl(lin(d.ted, 2 * d.hidden, GGML_TYPE_F32, true), faw, fab);
    ggml_tensor * fmods = ggml_reshape_2d(ctx, m.lin(faw, temb, fab, false), d.hidden, 2 * nts);
    ggml_tensor * fnw = m.vec(addr(1), d.hidden);
    auto finalSlice = [&](int start, int count, int col) {
        ggml_tensor * part = tsg::h3_token_rows(ctx, x, start, count);
        part = ggml_mul(ctx, ggml_rms_norm(ctx, part, d.eps), fnw);
        ggml_tensor * sh = ggml_view_2d(ctx, fmods, d.hidden, 1, fmods->nb[1], static_cast<std::size_t>(col) * fmods->nb[1]);
        ggml_tensor * sc = ggml_view_2d(ctx, fmods, d.hidden, 1, fmods->nb[1], static_cast<std::size_t>(col + 1) * fmods->nb[1]);
        return ggml_add(ctx, ggml_add(ctx, part, ggml_mul(ctx, part, sc)), sh);
    };
    ggml_tensor *fvw, *fvb, *fa2, *fb2;
    m.decl(lin(d.hidden, vpd, GGML_TYPE_F32, true), fvw, fvb);
    m.decl(lin(d.hidden, ach, GGML_TYPE_F32, true), fa2, fb2);
    ggml_tensor * videoOut = ggml_scale(ctx, m.lin(fvw, finalSlice(ntok - video, video, 0), fvb), -1.0f);
    ggml_tensor * audioOut = ggml_scale(ctx, m.lin(fa2, finalSlice(text + cond, audio, 2), fb2), 0.5f);
    ggml_tensor * vcopy = ggml_cpy(ctx, videoOut, voutT);
    ggml_tensor * acopy = ggml_cpy(ctx, audioOut, aoutT);
    ggml_set_output(vcopy); ggml_set_output(acopy);

    ArenaUse use{};
    use.budget = node_budget(nl, nref, nseg, true);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, use.budget, false);
    ggml_build_forward_expand(graph, vcopy);
    ggml_build_forward_expand(graph, acopy);
    // CUDA's stable-graph path also builds the audio scale's own 4-node graph here.
    ggml_scale_inplace(ctx, aoutT, 0.5f);
    ggml_new_graph_custom(ctx, 4, false);
    use.used = ggml_used_mem(ctx);
    use.nodes = static_cast<std::size_t>(ggml_graph_n_nodes(graph));
    use.leafs = static_cast<std::size_t>(count_leafs(graph));
    ggml_free(ctx);
    return use;
}

void case_arena(ggml_backend_t backend) {
    constexpr double kArena = 32.0 * 1024 * 1024;
    // FL2VA, the only partition the Turbo adapters apply to (Ref2VA refuses them),
    // makes at most 9 segments: the text runs around two keyframes' vision spans
    // (5), the 2 keyframes, the target audio and the target video.
    constexpr int kFl2vaSegments = 9;
    for (bool padded : {false, true}) {
        const ArenaUse use = arena_use(backend, kFl2vaSegments, padded);
        const double fraction = use.used / kArena;
        const std::string what = std::string("FL2VA, ") + std::to_string(kFl2vaSegments) + " segments" +
                                 (padded ? ", metal padded flash" : "");
        require(use.nodes <= use.budget && use.leafs <= use.budget, what + ": the graph outgrew its node budget");
        require(fraction <= 0.75, what + ": graph metadata uses " + num(fraction * 100) + "% of the 32 MiB arena");
        const std::string key = std::string("arenaFl2va") + (padded ? "Padded" : "");
        report(key + "UsedBytes", std::to_string(use.used));
        report(key + "UsedPct", num(fraction * 100));
        report(key + "Nodes", std::to_string(use.nodes));
        report(key + "Leafs", std::to_string(use.leafs));
        report(key + "NodeBudget", std::to_string(use.budget));
    }
    // Headroom: the most segments an adapter graph (the larger, padded attention)
    // could have and still stay within 75%. Each segment adds about 1.5%, so the
    // first count past 75% is still well inside the context.
    int segments = kFl2vaSegments;
    while (arena_use(backend, segments + 1, true).used / kArena <= 0.75) ++segments;
    report("arenaSegmentsWithin75Pct", std::to_string(segments));
}

// ---- 10: validator messages ----------------------------------------------------------

void case_validate() {
    const std::int64_t hidden = 5376, inner = 7168, ffn = 14336;
    static std::uint8_t bytes[16];
    auto good = [&](std::int64_t in, std::int64_t out) { return make_lora(GGML_TYPE_F16, 128, in, out, bytes, bytes + 8); };
    TSGH3BlockLora ok{};
    ok.q = good(hidden, inner); ok.k = good(hidden, inner); ok.v = good(hidden, inner);
    ok.out = good(inner, hidden); ok.fc1 = good(hidden, 2 * ffn); ok.fc2 = good(ffn, hidden);
    std::vector<TSGH3BlockLora> blocks(50, ok), refiner(2, ok);
    auto adapter = [&]() {
        TSGH3DitAdapter a{static_cast<std::int32_t>(sizeof(TSGH3DitAdapter)), 50, 2, 0, blocks.data(), refiner.data()};
        return a;
    };
    int rejections = 0;
    auto expect = [&](const TSGH3DitAdapter * a, const std::string & message) {
        const std::string got = tsg::h3_lora_validate(a, 50, 2, hidden, inner, ffn);
        require(got == message, "validator said \"" + got + "\", expected \"" + message + "\"");
        rejections += !message.empty();
    };
    expect(nullptr, "");
    TSGH3DitAdapter a = adapter();
    expect(&a, "");
    a.struct_bytes = 40;
    expect(&a, "adapter is 40 bytes, this GgmlOps expects 32; the managed TensorSharp build and the native library differ, rebuild GgmlOps.");
    a = adapter(); a.num_blocks = 49;
    expect(&a, "adapter covers 49 blocks, but the forward runs 50.");
    a = adapter(); a.num_refiner_blocks = 1;
    expect(&a, "adapter covers 1 refiner blocks, but the forward runs 2.");
    a = adapter(); a.blocks = nullptr;
    expect(&a, "adapter has no block array.");
    a = adapter(); a.refiner = nullptr;
    expect(&a, "adapter has no refiner block array.");
    a = adapter();
    blocks[17].v.rank = -3;
    expect(&a, "block 17 attn.to_v: rank -3 is negative.");
    blocks[17] = ok; blocks[17].q.type = GGML_TYPE_Q8_0;
    expect(&a, "block 17 attn.to_q: factor type 8 is neither F32 (0) nor F16 (1).");
    blocks[17] = ok; blocks[17].k.up = nullptr;
    expect(&a, "block 17 attn.to_k: a factor pointer is null.");
    blocks[17] = ok; blocks[17].out.in = hidden;
    expect(&a, "block 17 attn.to_out.0: factors are in 5376 x out 5376, but the layer is in 7168 x out 5376.");
    blocks[17] = ok; blocks[17].fc2.in = hidden;
    expect(&a, "block 17 ff.net.2: factors are in 5376 x out 5376, but the layer is in 14336 x out 5376.");
    blocks[17] = ok;
    refiner[1].fc1.out = ffn;
    expect(&a, "refiner block 1 ff.net.0.proj: factors are in 5376 x out 14336, but the layer is in 5376 x out 28672.");
    refiner[1] = ok;
    // Rank 0 is "no update": nothing else about the part is read.
    blocks[3].fc1 = TSGH3Lora{};
    blocks[3].fc1.type = 99;
    expect(&a, "");
    // An adapter longer than the forward is fine (a depth-limited probe runs fewer blocks).
    require(tsg::h3_lora_validate(&a, 10, 0, hidden, inner, ffn).empty(), "a forward of fewer blocks was refused");
    report("validatorRejections", std::to_string(rejections));
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3 || std::strcmp(argv[2], "--test") != 0) {
        std::fprintf(stderr, "usage: %s cpu|metal|cuda --test\n", argv[0]);
        return 2;
    }
    ggml_backend_t backend = nullptr;
    if (!std::strcmp(argv[1], "cpu")) {
        backend = ggml_backend_cpu_init();
        if (backend) ggml_backend_cpu_set_n_threads(backend, 4);
    }
#ifdef TSG_GGML_USE_METAL
    if (!std::strcmp(argv[1], "metal")) backend = ggml_backend_metal_init();
#endif
#ifdef TSG_GGML_USE_CUDA
    if (!std::strcmp(argv[1], "cuda")) { backend = ggml_backend_cuda_init(0); g_exact = false; }
#endif
    if (!backend) { std::fprintf(stderr, "requested backend unavailable\n"); return 77; }
    try {
        case_validate();
        case_product(backend);
        case_stacked(backend);
        case_folds(backend);
        case_fc1(backend);
        case_blocks(backend);
        case_buffers(backend);
        case_arena(backend);
        std::printf("{\"backend\":\"%s\",\"exactIdentity\":%s,%s}\n", ggml_backend_name(backend),
                    g_exact ? "true" : "false", g_report.c_str());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAILED: %s\n", e.what());
        ggml_backend_free(backend);
        return 1;
    }
    ggml_backend_free(backend);
    return 0;
}
