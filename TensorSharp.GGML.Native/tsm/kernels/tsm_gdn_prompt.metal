// A GDN layer's recurrence for a prompt chunk of T tokens, in three passes. Ported from Strata
// (github.com/Niko1221/Strata, metal/kernels/gdn_prompt.metal at 146df3c4: gdn_pre, gdn_scan_r<4, 16> and gdn_post).
// MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors; see ../LICENSE-STRATA.
//
// TensorSharp's changes: the recurrent state is read and written in ggml's gated-delta-net layout (per value head h a
// 128 x 128 matrix stored transposed: S[i][j] at (h * 128 + j) * 128 + i, i the key row, j the value column) instead of
// Strata's ((i * h_v + h) * 128 + j); the q / k L2 norm is ggml_l2_norm's x / max(|x|, eps) (Strata: x / sqrt(|x|^2 +
// eps)); the fused-input and probe function constants are dropped; gdn_post writes fp32 only; the conv history's new
// tail is staged (gdn_pre still reads the old one) and stored by tsm_gdn_hist_store.
//
//   tsm_gdn_pre    per (token, head): conv (4 taps; history oldest first) -> SiLU; q, k L2 norm; kq = k . q per key
//                  head; gate = exp(softplus(alpha + dt) * a), beta = sigmoid(beta); the chunk's last 3 conv inputs ->
//                  hist_out
//   tsm_gdn_scan   per (value head, 32 columns): the state column in registers (32 rows a simdgroup, 4 simdgroups);
//                  per token a = S^T k, b = S^T q (old state), delta = (v - g a) beta, o = (g b + delta kq) / sqrt(128)
//                  (= S_new^T q / sqrt(128)), S = g S + k delta^T
//   tsm_gdn_post   per (token, head): y = o * rsqrt(mean(o^2) + eps) * gamma * sigmoid(z)
//
// Layouts (ggml's tensors): qkv [T][channels] (q | k | v, channels = 2 * h_k * 128 + h_v * 128), z and y [T][h_v * 128],
// alpha and beta [T][h_v], the conv history [channels][3], the conv weights [channels][4]. Value head h reads q/k head
// h % h_k (ggml_repeat's tiling).

struct TsmGdnArgs {
    uint n_tok;
    uint h_k;
    uint h_v;
    uint channels;
    float eps;
    uint pad;
};

static inline float tsm_gdn_conv(device const float* qkv, device const float* hist, device const float* cw, int ch, int t,
                                 int channels) {
    float s = 0.0f;
    for (int k = 0; k < 4; ++k) {
        const int src = t - 3 + k;
        const float x = src >= 0 ? qkv[src * channels + ch] : hist[ch * 3 + (src + 3)];
        s = fma(x, cw[ch * 4 + k], s);
    }
    return s / (1.0f + exp(-s));
}

// grid (h_k + h_v / 2, T), 256 threads. x < h_k: q (threads 0..127) and k (128..255) of key head x; else value heads
// 2 (x - h_k) and + 1.
kernel void tsm_gdn_pre(device const float* qkv [[buffer(0)]], device const float* hist [[buffer(1)]],
                        device const float* conv_w [[buffer(2)]], device const float* alpha [[buffer(3)]],
                        device const float* beta [[buffer(4)]], device const float* dt [[buffer(5)]],
                        device const float* ssm_a [[buffer(6)]], device float* qn [[buffer(7)]], device float* kn [[buffer(8)]],
                        device float* vc [[buffer(9)]], device float2* gb [[buffer(10)]], device float* kq [[buffer(11)]],
                        device float* hist_out [[buffer(12)]], constant TsmGdnArgs& a [[buffer(13)]],
                        uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int S = 128;
    threadgroup float part[8];
    threadgroup float prod[S];
    const int x = int(tg.x), t = int(tg.y), hk_n = int(a.h_k), hv = int(a.h_v), C = int(a.channels), T = int(a.n_tok);
    int ch;
    if (x < hk_n) ch = tid < 128 ? x * S + tid : hk_n * S + x * S + (tid - 128);
    else ch = 2 * hk_n * S + (2 * (x - hk_n) + (tid >> 7)) * S + (tid & 127);
    const float cv = tsm_gdn_conv(qkv, hist, conv_w, ch, t, C);
    if (t == T - 1)
        for (int k = 0; k < 3; ++k) {
            const int src = T - 3 + k;
            hist_out[ch * 3 + k] = src >= 0 ? qkv[src * C + ch] : hist[ch * 3 + (src + 3)];
        }
    if (x < hk_n) {
        const float ssq = simd_sum(cv * cv);
        if (lane == 0) part[sg] = ssq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int base = tid < 128 ? 0 : 4;
        const float tot = (part[base] + part[base + 1]) + (part[base + 2] + part[base + 3]);
        const float nv = cv * (1.0f / max(sqrt(tot), a.eps));   // ggml_l2_norm's x / max(|x|, eps)
        if (tid < 128) {
            qn[(t * hk_n + x) * S + tid] = nv;
            prod[tid] = nv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid >= 128) {
            kn[(t * hk_n + x) * S + tid - 128] = nv;
            const float p = simd_sum(nv * prod[tid - 128]);
            if (lane == 0) part[sg] = p;   // simdgroups 4..7 (q's slots 0..3 are read already)
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) kq[t * hk_n + x] = (part[4] + part[5]) + (part[6] + part[7]);
    } else {
        const int h = 2 * (x - hk_n) + (tid >> 7);
        vc[(t * hv + h) * S + (tid & 127)] = cv;
        if ((tid & 127) == 0) {
            const float v = alpha[t * hv + h] + dt[h];
            const float e = exp(v), u = 1.0f + e;
            const float sp = v > 20.0f ? v : (u == 1.0f ? e : log(u) * e / (u - 1.0f));
            gb[t * hv + h] = float2(exp(sp * ssm_a[h]), 1.0f / (1.0f + exp(-beta[t * hv + h])));
        }
    }
}

// grid (h_v * 4), 128 threads: lane = column (32 columns a threadgroup), the 128 rows of a column split over 4
// simdgroups (32 rows a lane); per token the 4 partial dots are exchanged through threadgroup memory (double-buffered:
// one barrier a token). q and k staged 16 tokens at a time.
kernel void tsm_gdn_scan(device float* state [[buffer(0)]], device const float4* qn [[buffer(1)]],
                         device const float4* kn [[buffer(2)]], device const float* vc [[buffer(3)]],
                         device const float2* gb [[buffer(4)]], device const float* kq [[buffer(5)]],
                         device float* o [[buffer(6)]], constant TsmGdnArgs& a [[buffer(7)]],
                         uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int S = 128, IB = 4, TB = 16, R = S / IB, R4 = R / 4;
    threadgroup float4 qs[TB][32], ks[TB][32];
    threadgroup float2 red[2][IB][32];
    threadgroup float vs[TB][32];
    threadgroup float4 gk[TB];   // gate, beta, kq
    const int h = int(tg.x) / 4, hk = h % int(a.h_k), hv = int(a.h_v), hkn = int(a.h_k);
    const int j = (int(tg.x) % 4) * 32 + lane, i0 = sg * R;
    const int T = int(a.n_tok);
    const float scale = 1.0f / sqrt(float(S));
    // ggml's layout: S[i][j] at (h * 128 + j) * 128 + i - a lane's rows are contiguous
    device float* col = state + (h * S + j) * S + i0;
    float st[R];
#pragma clang loop unroll(full)
    for (int i = 0; i < R; ++i) st[i] = col[i];
    for (int t0 = 0; t0 < T; t0 += TB) {
        const int nb = min(TB, T - t0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = tid; idx < nb * 32; idx += IB * 32) {
            const int tt = idx >> 5, w = idx & 31;
            qs[tt][w] = qn[((t0 + tt) * hkn + hk) * 32 + w];
            ks[tt][w] = kn[((t0 + tt) * hkn + hk) * 32 + w];
            vs[tt][w] = vc[((t0 + tt) * hv + h) * S + (j - lane) + w];
            if (w == 0) {
                const float2 g_b = gb[(t0 + tt) * hv + h];
                gk[tt] = float4(g_b.x, g_b.y, kq[(t0 + tt) * hkn + hk], 0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int tt = 0; tt < nb; ++tt) {
            const int t = t0 + tt, slot = t & 1;
            float ka = 0.0f, qa = 0.0f;
#pragma clang loop unroll(full)
            for (int w = 0; w < R4; ++w) {
                const float4 k4 = ks[tt][i0 / 4 + w], q4 = qs[tt][i0 / 4 + w];
                ka = fma(st[4 * w], k4.x, ka); ka = fma(st[4 * w + 1], k4.y, ka);
                ka = fma(st[4 * w + 2], k4.z, ka); ka = fma(st[4 * w + 3], k4.w, ka);
                qa = fma(st[4 * w], q4.x, qa); qa = fma(st[4 * w + 1], q4.y, qa);
                qa = fma(st[4 * w + 2], q4.z, qa); qa = fma(st[4 * w + 3], q4.w, qa);
            }
            const float4 g_b = gk[tt];
            const float kqv = g_b.z;
            const float v = vs[tt][lane];
            red[slot][sg][lane] = float2(ka, qa);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float2 s = red[slot][0][lane];
#pragma clang loop unroll(full)
            for (int b = 1; b < IB; ++b) s += red[slot][b][lane];
            const float g = g_b.x;
            const float delta = (v - g * s.x) * g_b.y;
            if (sg == 0) o[(t * hv + h) * S + j] = fma(g, s.y, delta * kqv) * scale;
#pragma clang loop unroll(full)
            for (int w = 0; w < R4; ++w) {
                const float4 k4 = ks[tt][i0 / 4 + w];
                st[4 * w] = fma(g, st[4 * w], k4.x * delta);
                st[4 * w + 1] = fma(g, st[4 * w + 1], k4.y * delta);
                st[4 * w + 2] = fma(g, st[4 * w + 2], k4.z * delta);
                st[4 * w + 3] = fma(g, st[4 * w + 3], k4.w * delta);
            }
        }
    }
#pragma clang loop unroll(full)
    for (int i = 0; i < R; ++i) col[i] = st[i];
}

// grid (h_v, T), 128 threads
kernel void tsm_gdn_post(device const float* o [[buffer(0)]], device const float* z [[buffer(1)]],
                         device const float* gamma [[buffer(2)]], device float* y [[buffer(3)]],
                         constant TsmGdnArgs& a [[buffer(4)]],
                         uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float part[4];
    const int h = int(tg.x), t = int(tg.y), hv = int(a.h_v);
    const uint i = uint(t * hv + h) * 128 + tid;
    const float ov = o[i];
    const float ss = simd_sum(ov * ov);
    if (lane == 0) part[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float tot = (part[0] + part[1]) + (part[2] + part[3]);
    const float rs = rsqrt(tot / 128.0f + a.eps);
    y[i] = ((rs * ov) * gamma[tid]) * (1.0f / (1.0f + exp(-z[i])));
}

// The staged conv tail into the layer's history: grid ceil(channels * 3 / 256), 256 threads.
kernel void tsm_gdn_hist_store(device const float* staged [[buffer(0)]], device float* hist [[buffer(1)]],
                               constant TsmGdnArgs& a [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid < a.channels * 3) hist[gid] = staged[gid];
}
