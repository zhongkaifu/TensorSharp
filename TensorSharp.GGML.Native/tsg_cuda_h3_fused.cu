// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Fused elementwise stages of MiniMax-H3's DiT block on CUDA.
//
// With the attention on cuDNN and the projections on cuBLAS, what remained of a
// 480x864x158 denoising step (20,511 tokens) was ~10 s of elementwise passes over
// F32 activations of 0.4-1.8 GB each: per block, the q/k RMSNorm, ten RoPE passes,
// the permute copies and F16 casts in front of the attention, the per-segment
// AdaLN modulation rebuilt with concats, and the bias / LoRA / gate adds of the two
// residuals. Each kernel here does one of those stages in one read and one write,
// with the arithmetic in the graph's own order and rounding (__fmul_rn/__fadd_rn,
// so the compiler cannot contract a multiply-add the graph rounds twice):
//
//   tsg_h3_attn_prep      the qkv bias, q/k RMSNorm * weight, rotate-half RoPE over the leading
//                         `rot` dims, and Q/K/V narrowed to F16 in cuDNN's [H, S, D]
//                         (V divided by the graph's power-of-two V scale)
//   tsg_h3_modulate       RMSNorm(x) * w, then x + x*scale + shift per segment
//   tsg_h3_gated_residual base + (delta + (mm + bias)) * gate per segment
//   tsg_h3_widen_scaled   F16 [S, H, D] attention output to F32, times the V scale
//
// The reductions run in a different order than ggml's, so RMSNorm results can
// differ in the last bit; everything else matches the graph operation for
// operation. graph_compute_offload executes the GGML_OP_CUSTOM nodes that stand
// for these stages (see ggml_ops_minimax_h3.cpp); all launches go to the legacy
// default stream and the caller synchronizes.
//
// The kernels that read a projection's product (the attention prep its q, k and v, the
// two residuals their mm) also take it as the int32 sums an int8 GEMM left unscaled,
// given the peaks to scale them by as they load (tsg_cuda_int8_scale.h); null peaks
// mean F32, as before.
#if defined(TSG_GGML_USE_CUDA)

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "tsg_cuda_int8_scale.h"

#include <cstdint>

namespace {

constexpr int kMaxSegments = 8;

// Unused slots are empty ranges, and `last` is the final segment's column.
struct Segments
{
    int start[kMaxSegments], end[kMaxSegments], col[kMaxSegments];
    int last;
};

// Unrolled over every slot so each index is a constant: a kernel argument indexed
// by a runtime value is copied to every thread's local memory first, which costs
// more than the lookup. The segments are disjoint, so the order does not matter.
__device__ __forceinline__ int segment_col(const Segments& s, long long t)
{
    int col = s.last;
#pragma unroll
    for (int i = kMaxSegments - 1; i >= 0; i--)
        if (t >= s.start[i] && t < s.end[i]) col = s.col[i];
    return col;
}

__device__ __forceinline__ float warp_sum(float v)
{
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// One warp per (token, head); D/32 elements per lane. smem holds the normalized head
// so the rotate-half partner (d +- rot/2) can be read across lanes. S counts every
// token of the call; with several independent sequences of rope_seq tokens each (a
// batch of VAE tiles) token s takes the rotation of its position s % rope_seq.
template <int D>
__global__ void attn_prep(const float* __restrict__ q, long long q_sh, long long q_ss,
                          const float* __restrict__ k, long long k_sh, long long k_ss,
                          const float* __restrict__ v, long long v_sh, long long v_ss,
                          const float* __restrict__ qb, const float* __restrict__ kb, const float* __restrict__ vb,
                          long long bias_hs, const float* __restrict__ qn, const float* __restrict__ kn,
                          const float* __restrict__ cosT, const float* __restrict__ ssinT, int rot,
                          float eps, float vInv, int H, int S, int rope_seq,
                          __half* __restrict__ q16, __half* __restrict__ k16, __half* __restrict__ v16,
                          const uint32_t* __restrict__ rowpeak, const uint32_t* __restrict__ tokpeak,
                          long long q_off, long long k_off, long long v_off)
{
    constexpr int J = D / 32;
    __shared__ float smem[4][D];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long long pair = (long long) blockIdx.x * 4 + warp;
    if (pair >= (long long) S * H) return;
    const long long s = pair / H;
    const int h = (int) (pair - s * H);
    const long long outBase = ((long long) h * S + s) * D;
    const int half = rot / 2;
    const long long pos = rope_seq > 0 && rope_seq < S ? s % rope_seq : s;
    // With peaks, q, k and v are int32 sums; *_off is the product row q, k or v starts at.
    // The peaks are per token of the call, whichever sequence it belongs to.
    const float sx = tokpeak != nullptr ? tsg_int8_token_scale(tokpeak[s]) : 0.0f;

    for (int which = 0; which < 2; which++)
    {
        const float* src = which == 0 ? q + s * q_ss + h * q_sh : k + s * k_ss + h * k_sh;
        const long long first = which == 0 ? q_off + h * q_sh : k_off + h * k_sh;
        const float* w = which == 0 ? qn : kn;
        const float* bias = which == 0 ? qb : kb;
        __half* dst = which == 0 ? q16 : k16;
        float x[J];
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < J; j++)
        {
            x[j] = tsg_int8_product(src, lane + 32 * j, rowpeak, first, sx);
            if (bias != nullptr) x[j] = __fadd_rn(x[j], bias[h * bias_hs + lane + 32 * j]);
            ss += x[j] * x[j];
        }
        ss = warp_sum(ss);
        const float inv = rsqrtf(ss / D + eps);
#pragma unroll
        for (int j = 0; j < J; j++)
        {
            float y = __fmul_rn(x[j], inv);
            if (w != nullptr) y = __fmul_rn(y, w[lane + 32 * j]);
            x[j] = y;
            smem[warp][lane + 32 * j] = y;
        }
        __syncwarp();
#pragma unroll
        for (int j = 0; j < J; j++)
        {
            const int d = lane + 32 * j;
            float o = x[j];
            if (d < rot)
            {
                const float partner = smem[warp][d < half ? d + half : d - half];
                o = __fadd_rn(__fmul_rn(x[j], cosT[pos * rot + d]), __fmul_rn(partner, ssinT[pos * rot + d]));
            }
            dst[outBase + d] = __float2half_rn(o);
        }
        __syncwarp();
    }
    const float* vsrc = v + s * v_ss + h * v_sh;
    const long long vfirst = v_off + h * v_sh;
#pragma unroll
    for (int j = 0; j < J; j++)
    {
        float y = tsg_int8_product(vsrc, lane + 32 * j, rowpeak, vfirst, sx);
        if (vb != nullptr) y = __fadd_rn(y, vb[h * bias_hs + lane + 32 * j]);
        v16[outBase + lane + 32 * j] = __float2half_rn(vInv == 1.0f ? y : __fmul_rn(y, vInv));
    }
}

// One block per token: RMSNorm over `hidden`, times w, then the segment's
// x + x*scale + shift.
__global__ void modulate(const float* __restrict__ x, long long x_rs, const float* __restrict__ w,
                         const float* __restrict__ mods, long long m_rs, int shiftIdx, int scaleIdx,
                         Segments segs, float eps, int hidden, float* __restrict__ out)
{
    const long long t = blockIdx.x;
    const float* row = x + t * x_rs;
    float ss = 0.0f;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) { const float a = row[c]; ss += a * a; }
    __shared__ float part[32];
    ss = warp_sum(ss);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float v = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0.0f;
        v = warp_sum(v);
        if (threadIdx.x == 0) part[0] = v;
    }
    __syncthreads();
    const float inv = rsqrtf(part[0] / hidden + eps);
    const int col = segment_col(segs, t);
    const float* sh = mods + (long long) (col + shiftIdx) * m_rs;
    const float* sc = mods + (long long) (col + scaleIdx) * m_rs;
    float* o = out + t * hidden;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
    {
        float y = __fmul_rn(row[c], inv);
        if (w != nullptr) y = __fmul_rn(y, w[c]);
        y = __fadd_rn(y, __fmul_rn(y, sc[c]));
        o[c] = __fadd_rn(y, sh[c]);
    }
}

// One block per token, so the segment's gate row is looked up once per block.
__global__ void gated_residual(const float* __restrict__ base, long long b_rs,
                               const float* __restrict__ mm, long long mm_rs,
                               const float* __restrict__ bias,
                               const float* __restrict__ delta, long long d_rs,
                               const float* __restrict__ mods, long long m_rs, int gateIdx,
                               Segments segs, int hidden, float* __restrict__ out,
                               const uint32_t* __restrict__ mmRow, const uint32_t* __restrict__ mmTok)
{
    const long long t = blockIdx.x;
    const float* b = base + t * b_rs;
    const float* m = mm + t * mm_rs;
    const float sx = mmTok != nullptr ? tsg_int8_token_scale(mmTok[t]) : 0.0f;
    const float* d = delta != nullptr ? delta + t * d_rs : nullptr;
    const float* g = mods + (long long) (segment_col(segs, t) + gateIdx) * m_rs;
    float* o = out + t * hidden;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x)
    {
        float u = tsg_int8_product(m, c, mmRow, 0, sx);
        if (bias != nullptr) u = __fadd_rn(u, bias[c]);
        if (d != nullptr) u = __fadd_rn(d[c], u);
        o[c] = __fadd_rn(b[c], __fmul_rn(u, g[c]));
    }
}

// The video VAE's residual: base + (mm + bias) * scale, bias and scale per channel
// (each nullable), in the graph's order (add, mul, add).
__global__ void scaled_residual(const float* __restrict__ base, long long b_rs, const float* __restrict__ mm,
                                long long mm_rs, const float* __restrict__ bias, const float* __restrict__ scale,
                                int hidden, long long rows, float* __restrict__ out,
                                const uint32_t* __restrict__ mmRow, const uint32_t* __restrict__ mmTok)
{
    const long long n = (long long) hidden * rows;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x)
    {
        const long long t = i / hidden;
        const int c = (int) (i - t * hidden);
        float u = tsg_int8_product(mm + t * mm_rs, c, mmRow, 0, mmTok != nullptr ? tsg_int8_token_scale(mmTok[t]) : 0.0f);
        if (bias != nullptr) u = __fadd_rn(u, bias[c]);
        if (scale != nullptr) u = __fmul_rn(u, scale[c]);
        out[i] = __fadd_rn(base[t * b_rs + c], u);
    }
}

__global__ void widen_scaled(const __half* __restrict__ src, float* __restrict__ dst, long long n, float scale)
{
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x)
    {
        const float v = __half2float(src[i]);
        dst[i] = scale == 1.0f ? v : __fmul_rn(v, scale);
    }
}

int grid_for(long long n)
{
    const long long blocks = (n + 255) / 256;
    return (int) (blocks < 1048576 ? blocks : 1048576);
}

Segments make_segments(const int32_t* segs, int nseg)
{
    Segments s{};
    const int n = nseg < kMaxSegments ? nseg : kMaxSegments;
    for (int i = 0; i < n; i++)
    {
        s.start[i] = segs[4 * i + 0];
        s.end[i] = segs[4 * i + 1];
        s.col[i] = segs[4 * i + 2];
    }
    s.last = n > 0 ? s.col[n - 1] : 0;
    return s;
}

} // namespace

extern "C" {

int tsg_h3_fused_max_segments(void) { return kMaxSegments; }

// Strides are in floats: *_sh between heads, *_ss between tokens. qn/kn may be null.
// seq is every token of the call; rope_seq (0 = seq) is the length of each of the
// independent sequences they make up, which the rotation tables cover.
// With rowpeak and tokpeak, q, k and v are views of one product still held as int32
// sums, starting at its rows q_off, k_off and v_off (tsg_cuda_int8_scale.h).
bool tsg_h3_attn_prep(const float* q, int64_t q_sh, int64_t q_ss, const float* k, int64_t k_sh, int64_t k_ss,
                      const float* v, int64_t v_sh, int64_t v_ss, const float* qb, const float* kb, const float* vb,
                      int64_t bias_hs, const float* qn, const float* kn,
                      const float* cosT, const float* ssinT, int rot, float eps, float vInv,
                      int heads, int seq, int rope_seq, int head_dim, void* q16, void* k16, void* v16,
                      const uint32_t* rowpeak, const uint32_t* tokpeak, int64_t q_off, int64_t k_off, int64_t v_off)
{
    const long long pairs = (long long) seq * heads;
    const unsigned int blocks = (unsigned int) ((pairs + 3) / 4);
    switch (head_dim)
    {
    case 64:
        attn_prep<64><<<blocks, 128>>>(q, q_sh, q_ss, k, k_sh, k_ss, v, v_sh, v_ss, qb, kb, vb, bias_hs > 0 ? bias_hs : 64,
                                       qn, kn, cosT, ssinT, rot, eps, vInv,
                                       heads, seq, rope_seq, (__half*) q16, (__half*) k16, (__half*) v16,
                                       rowpeak, tokpeak, q_off, k_off, v_off);
        break;
    case 128:
        attn_prep<128><<<blocks, 128>>>(q, q_sh, q_ss, k, k_sh, k_ss, v, v_sh, v_ss, qb, kb, vb, bias_hs > 0 ? bias_hs : 128,
                                        qn, kn, cosT, ssinT, rot, eps, vInv,
                                        heads, seq, rope_seq, (__half*) q16, (__half*) k16, (__half*) v16,
                                        rowpeak, tokpeak, q_off, k_off, v_off);
        break;
    default:
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

// segs: nseg x {start, end, col, pad} int32, sorted and covering [0, rows).
bool tsg_h3_modulate(const float* x, int64_t x_rs, const float* w, const float* mods, int64_t m_rs,
                     int shiftIdx, int scaleIdx, const int32_t* segs, int nseg, float eps,
                     int hidden, int64_t rows, float* out)
{
    if (nseg <= 0 || nseg > kMaxSegments) return false;
    modulate<<<(unsigned int) rows, 256>>>(x, x_rs, w, mods, m_rs, shiftIdx, scaleIdx, make_segments(segs, nseg),
                                           eps, hidden, out);
    return cudaGetLastError() == cudaSuccess;
}

// mmRow and mmTok (both or neither): mm holds int32 sums (tsg_cuda_int8_scale.h).
bool tsg_h3_gated_residual(const float* base, int64_t b_rs, const float* mm, int64_t mm_rs, const float* bias,
                           const float* delta, int64_t d_rs, const float* mods, int64_t m_rs, int gateIdx,
                           const int32_t* segs, int nseg, int hidden, int64_t rows, float* out,
                           const uint32_t* mmRow, const uint32_t* mmTok)
{
    if (nseg <= 0 || nseg > kMaxSegments) return false;
    gated_residual<<<(unsigned int) rows, 256>>>(base, b_rs, mm, mm_rs, bias, delta, d_rs, mods, m_rs, gateIdx,
                                                 make_segments(segs, nseg), hidden, out, mmRow, mmTok);
    return cudaGetLastError() == cudaSuccess;
}

bool tsg_h3_scaled_residual(const float* base, int64_t b_rs, const float* mm, int64_t mm_rs, const float* bias,
                            const float* scale, int hidden, int64_t rows, float* out,
                            const uint32_t* mmRow, const uint32_t* mmTok)
{
    scaled_residual<<<grid_for((long long) hidden * rows), 256>>>(base, b_rs, mm, mm_rs, bias, scale, hidden, rows, out,
                                                                  mmRow, mmTok);
    return cudaGetLastError() == cudaSuccess;
}

bool tsg_h3_widen_scaled(const void* src16, float* dst, int64_t n, float scale)
{
    widen_scaled<<<grid_for(n), 256>>>((const __half*) src16, dst, n, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool tsg_h3_fused_sync(void) { return cudaStreamSynchronize(nullptr) == cudaSuccess; }

} // extern "C"

#endif // TSG_GGML_USE_CUDA
