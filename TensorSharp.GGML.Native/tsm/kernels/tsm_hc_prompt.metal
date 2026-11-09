// A prompt chunk's hyper-connection element-wise work around its GEMMs, one pass each instead of ggml's chains (after
// Strata's gr_epi_up and gr_norm_h, github.com/Niko1221/Strata metal/kernels/prompt.metal at 146df3c4, MIT; see
// ../LICENSE-STRATA). Both keep ggml's operation order (sigmoid = 1 / (1 + exp(-x)), the streams summed in order, then
// scaled), so they answer as the ops they replace up to the exponential's rounding.
//
//   tsm_hc_mix      mixed[t][d] = (sum over c of xn[t][c][d] * sigmoid(G[t][c][d])) * (1 / hc)
//                   (ggml: sigmoid, mul, a copy of stream 0, hc - 1 adds and a scale over [hc * n, T])
//   tsm_hc_scatter  out[t][c][d] = res[t][c][d] + b[t][d] * 2 * sigmoid(inject[t][c] * (1 / hc)); out may be res
//                   (ggml: scale, sigmoid, scale, a repeat of b to [n, hc, T], mul and add)
//
// Layouts (ggml's tensors): xn, G, res and out [T][hc][n]; mixed and b [T][n]; inject [T][hc].

struct TsmHcArgs {
    uint n_tok;
    uint n_embd;
    uint hc;
    uint pad;
};

// grid (ceil(n / 256), T), 256 threads
kernel void tsm_hc_mix(device const float* xn [[buffer(0)]], device const float* G [[buffer(1)]],
                       device float* mixed [[buffer(2)]], constant TsmHcArgs& a [[buffer(3)]],
                       uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint d = tg.x * 256 + tid, t = tg.y, n = a.n_embd, hc = a.hc;
    if (d >= n) return;
    const ulong row = ulong(t) * hc * n + d;
    float s = 0.0f;
    for (uint c = 0; c < hc; ++c) {
        const float g = 1.0f / (1.0f + exp(-G[row + c * n]));
        const float v = xn[row + c * n] * g;
        s = c == 0 ? v : s + v;
    }
    mixed[ulong(t) * n + d] = s * (1.0f / float(hc));
}

// grid (ceil(n / 256), T), 256 threads
kernel void tsm_hc_scatter(device const float* res [[buffer(0)]], device const float* b [[buffer(1)]],
                           device const float* inject [[buffer(2)]], device float* out [[buffer(3)]],
                           constant TsmHcArgs& a [[buffer(4)]],
                           uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint d = tg.x * 256 + tid, t = tg.y, n = a.n_embd, hc = a.hc;
    if (d >= n) return;
    const float bv = b[ulong(t) * n + d];
    const ulong row = ulong(t) * hc * n + d;
    for (uint c = 0; c < hc; ++c) {
        const float w = (1.0f / (1.0f + exp(-(inject[t * hc + c] * (1.0f / float(hc)))))) * 2.0f;
        out[row + c * n] = res[row + c * n] + bv * w;
    }
}
