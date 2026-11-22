// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Inkling 2's fused stages on ggml-cuda (see tsg_inkling_fused.h). Each kernel computes what the family's reference ops
// compute (Inkling2RuntimeModel.Graph.cs); reductions run in another order than ggml's, so results differ in the last bits.

#include "tsg_inkling_fused.h"
#include "tsg_cuda_graph_keys.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef TSG_GGML_USE_CUDA
#include "ggml_ops_dsv4_fused.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"   // ggml_backend_cuda_context (device, stream)

namespace
{
    __device__ __forceinline__ float warp_sum(float v)
    {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        return v;
    }

    __device__ __forceinline__ float warp_max(float v)
    {
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        return v;
    }

    __device__ __forceinline__ float sigmoidf_ref(float x) { return 1.0f / (1.0f + expf(-x)); }

    // Sum of `count` per-thread values over a block (blockDim.x a multiple of 32, at most 1024), every thread
    // receiving the sums. scratch holds 32 * count floats.
    __device__ void block_sums(float* values, int count, float* scratch)
    {
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
        for (int i = 0; i < count; i++)
        {
            const float v = warp_sum(values[i]);
            if (lane == 0) scratch[i * 32 + warp] = v;
        }
        __syncthreads();
        for (int i = 0; i < count; i++)
        {
            float v = lane < warps ? scratch[i * 32 + lane] : 0.0f;
            values[i] = warp_sum(v);
        }
        __syncthreads();
    }

    // ---- clip: one warp a vector of d
    __global__ void inkling_clip(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ y, int d,
                                 int heads, long long vectors, float threshold)
    {
        const long long v = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
        if (v >= vectors) return;
        const int lane = threadIdx.x & 31;
        const float* xv = x + v * d;
        const float* wv = w != nullptr ? w + (v % heads) * d : nullptr;
        float ss = 0.0f;
        for (int e = lane; e < d; e += 32)
        {
            float a = xv[e];
            if (wv != nullptr) a = __fmul_rn(a, wv[e]);
            ss += a * a;
        }
        ss = warp_sum(ss);
        const float norm = sqrtf(ss);
        const float factor = fminf(fmaxf(__fmul_rn(1.0f / __fadd_rn(norm, 1e-6f), threshold), 0.0f), 1.0f);
        float* yv = y + v * d;
        for (int e = lane; e < d; e += 32)
        {
            float a = xv[e];
            if (wv != nullptr) a = __fmul_rn(a, wv[e]);
            yv[e] = __fmul_rn(a, factor);
        }
    }

    // ---- short convolution: one thread a channel, every row in order
    constexpr int kMaxTaps = 8;

    __global__ void inkling_conv(const float* __restrict__ x, float* __restrict__ window, const float* __restrict__ kw,
                                 const float* __restrict__ dyn, float* __restrict__ y, int channels, int count, int taps,
                                 int heads, float scale)
    {
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= channels) return;
        const int hist = taps - 1;
        float win[kMaxTaps], wk[kMaxTaps];
        for (int j = 0; j < hist; j++) win[j] = window[(long long)c * hist + j];
        for (int i = 0; i < taps; i++) wk[i] = kw[(long long)c * taps + i];
        const int head = heads > 0 ? c / (channels / heads) : 0;
        const long long width = (long long)taps * heads;
        auto input = [&](int tau) -> float { return tau < hist ? win[tau] : x[(long long)(tau - hist) * channels + c]; };
        for (int t = 0; t < count; t++)
        {
            float conv = 0.0f;
            for (int i = 0; i < taps; i++) conv += input(t + i) * wk[i];
            if (dyn != nullptr)
                for (int lag = 0; lag < taps; lag++)
                    conv = __fadd_rn(conv, __fmul_rn(input(hist - lag + t), dyn[(long long)t * width + (long long)head * taps + lag]));
            const float xv = x[(long long)t * channels + c];
            y[(long long)t * channels + c] = __fmul_rn(__fadd_rn(xv, conv), scale);
        }
        float next[kMaxTaps];
        for (int j = 0; j < hist; j++) next[j] = input(count + j);
        for (int j = 0; j < hist; j++) window[(long long)c * hist + j] = next[j];
    }

    // ---- depth mix: one block a row
    constexpr int kMaxStates = 8, kMaxQueries = 4;

    __global__ void inkling_mix(const char* __restrict__ states, long long nb1, long long nb2, int n,
                                const float* __restrict__ query, int queries, float* __restrict__ out, int h, float eps,
                                float scale)
    {
        __shared__ float scratch[32 * (kMaxStates + kMaxStates * kMaxQueries)];
        __shared__ float rs[kMaxStates];
        __shared__ float probs[kMaxQueries * kMaxStates];
        const char* base = states + (long long)blockIdx.x * nb2;
        // One pass: each state's sum of squares and its raw dot with each query (the normed state's dot is rs times it).
        float acc[kMaxStates + kMaxStates * kMaxQueries];
        for (int i = 0; i < n + n * queries; i++) acc[i] = 0.0f;
        for (int e = threadIdx.x; e < h; e += blockDim.x)
            for (int s = 0; s < n; s++)
            {
                const float v = *(const float*)(base + s * nb1 + (long long)e * 4);
                acc[s] += v * v;
                for (int q = 0; q < queries; q++) acc[n + q * n + s] += query[(long long)q * h + e] * v;
            }
        block_sums(acc, n + n * queries, scratch);
        if (threadIdx.x < n) rs[threadIdx.x] = 1.0f / sqrtf(acc[threadIdx.x] / h + eps);
        __syncthreads();
        for (int i = 0; i < n * queries; i++) acc[i] = acc[n + i] * rs[i % n];
        if (threadIdx.x < queries)
        {
            const int q = threadIdx.x;
            float m = -INFINITY;
            for (int s = 0; s < n; s++) m = fmaxf(m, acc[q * n + s] * scale);
            float sum = 0.0f;
            for (int s = 0; s < n; s++)
            {
                const float p = expf(acc[q * n + s] * scale - m);
                probs[q * n + s] = p;
                sum += p;
            }
            for (int s = 0; s < n; s++) probs[q * n + s] = probs[q * n + s] / sum;
        }
        __syncthreads();
        for (int e = threadIdx.x; e < h; e += blockDim.x)
        {
            float v[kMaxStates];
            for (int s = 0; s < n; s++) v[s] = __fmul_rn(*(const float*)(base + s * nb1 + (long long)e * 4), rs[s]);
            for (int q = 0; q < queries; q++)
            {
                float o = 0.0f;
                for (int s = 0; s < n; s++) o += v[s] * probs[q * n + s];
                out[((long long)blockIdx.x * queries + q) * h + e] = o;
            }
        }
    }

    // ---- depth mix over several blocks a row: the partial sums (each state's squares and raw dots with the queries) of a
    // chunk of the hidden size, then every block finishes the norms and the softmax from all partials and writes its chunk.
    constexpr int kMixChunk = 256;

    __global__ void inkling_mix_partial(const char* __restrict__ states, long long nb1, long long nb2, int n,
                                        const float* __restrict__ query, int queries, float* __restrict__ partial, int h)
    {
        __shared__ float scratch[32 * (kMaxStates + kMaxStates * kMaxQueries)];
        const int chunks = gridDim.x, chunk = blockIdx.x;
        const long long row = blockIdx.y;
        const char* base = states + row * nb2;
        const int width = n + n * queries;
        float acc[kMaxStates + kMaxStates * kMaxQueries];
        for (int i = 0; i < width; i++) acc[i] = 0.0f;
        const int e = chunk * kMixChunk + threadIdx.x;
        if (e < h)
            for (int s = 0; s < n; s++)
            {
                const float v = *(const float*)(base + s * nb1 + (long long)e * 4);
                acc[s] = v * v;
                for (int q = 0; q < queries; q++) acc[n + q * n + s] = query[(long long)q * h + e] * v;
            }
        block_sums(acc, width, scratch);
        if (threadIdx.x < width) partial[(row * chunks + chunk) * width + threadIdx.x] = acc[threadIdx.x];
    }

    __global__ void inkling_mix_output(const char* __restrict__ states, long long nb1, long long nb2, int n, int queries,
                                       const float* __restrict__ partial, float* __restrict__ out, int h, float eps, float scale)
    {
        __shared__ float tot[kMaxStates + kMaxStates * kMaxQueries];
        __shared__ float rs[kMaxStates];
        __shared__ float probs[kMaxQueries * kMaxStates];
        const int chunks = gridDim.x, chunk = blockIdx.x;
        const long long row = blockIdx.y;
        const int width = n + n * queries;
        if (threadIdx.x < width)
        {
            float t = 0.0f;
            for (int c = 0; c < chunks; c++) t += partial[(row * chunks + c) * width + threadIdx.x];
            tot[threadIdx.x] = t;
        }
        __syncthreads();
        if (threadIdx.x < n) rs[threadIdx.x] = 1.0f / sqrtf(tot[threadIdx.x] / h + eps);
        __syncthreads();
        if (threadIdx.x < queries)
        {
            const int q = threadIdx.x;
            float m = -INFINITY;
            for (int s = 0; s < n; s++) m = fmaxf(m, tot[n + q * n + s] * rs[s] * scale);
            float sum = 0.0f;
            for (int s = 0; s < n; s++)
            {
                const float p = expf(tot[n + q * n + s] * rs[s] * scale - m);
                probs[q * n + s] = p;
                sum += p;
            }
            for (int s = 0; s < n; s++) probs[q * n + s] = probs[q * n + s] / sum;
        }
        __syncthreads();
        const int e = chunk * kMixChunk + threadIdx.x;
        if (e >= h) return;
        const char* base = states + row * nb2;
        float v[kMaxStates];
        for (int s = 0; s < n; s++) v[s] = __fmul_rn(*(const float*)(base + s * nb1 + (long long)e * 4), rs[s]);
        for (int q = 0; q < queries; q++)
        {
            float o = 0.0f;
            for (int s = 0; s < n; s++) o += v[s] * probs[q * n + s];
            out[(row * queries + q) * h + e] = o;
        }
    }

    // ---- loop controller: g = down [rms x; rms m] (one warp an output), then the blend (one thread a channel)
    __global__ void inkling_control_down(const float* __restrict__ x, const float* __restrict__ m,
                                         const float* __restrict__ down, float* __restrict__ g, int h, int rank, float eps)
    {
        __shared__ float scratch[64];
        __shared__ float factors[2];
        const long long row = blockIdx.y;
        const float* xr = x + row * h;
        const float* mr = m + row * h;
        float acc[2] = { 0.0f, 0.0f };
        for (int e = threadIdx.x; e < h; e += blockDim.x)
        {
            acc[0] += xr[e] * xr[e];
            acc[1] += mr[e] * mr[e];
        }
        block_sums(acc, 2, scratch);
        if (threadIdx.x == 0)
        {
            factors[0] = 1.0f / sqrtf(acc[0] / h + eps);
            factors[1] = 1.0f / sqrtf(acc[1] / h + eps);
        }
        __syncthreads();
        const int warps = blockDim.x >> 5, lane = threadIdx.x & 31;
        const int r = blockIdx.x * warps + (threadIdx.x >> 5);
        if (r >= rank) return;
        const float* wr = down + (long long)r * 2 * h;
        float sum = 0.0f;
        for (int i = lane; i < h; i += 32) sum += wr[i] * __fmul_rn(xr[i], factors[0]);
        for (int i = lane; i < h; i += 32) sum += wr[h + i] * __fmul_rn(mr[i], factors[1]);
        sum = warp_sum(sum);
        if (lane == 0) g[row * rank + r] = sum;
    }

    __global__ void inkling_control_blend(const float* __restrict__ x, const float* __restrict__ m,
                                          const float* __restrict__ gain, const float* __restrict__ forget,
                                          const float* __restrict__ g, float* __restrict__ y, int h, int rank)
    {
        // One warp a channel: its rank weights are read across the lanes (a thread a channel read them 256 bytes apart).
        const int lane = threadIdx.x & 31;
        const int c = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
        const long long row = blockIdx.y;
        if (c >= h) return;
        const float* gr = g + row * rank;
        const float* gw = gain + (long long)c * rank;
        const float* fw = forget + (long long)c * rank;
        float ga = 0.0f, fo = 0.0f;
        for (int r = lane; r < rank; r += 32)
        {
            ga += gw[r] * gr[r];
            fo += fw[r] * gr[r];
        }
        ga = warp_sum(ga);
        fo = warp_sum(fo);
        if (lane != 0) return;
        const float xv = x[row * h + c], mv = m[row * h + c];
        y[row * h + c] = __fadd_rn(__fsub_rn(xv, __fmul_rn(tanhf(fo), xv)), __fmul_rn(tanhf(ga), mv));
    }

    // ---- router: the top K of sigmoid(logits) + bias (a tie to the lower expert), in descending order
    __global__ void inkling_route_ids(const float* __restrict__ logits, const float* __restrict__ bias, int32_t* __restrict__ ids,
                                      int experts, int k)
    {
        __shared__ float sel[1024];
        __shared__ float best_v[32];
        __shared__ int best_i[32];
        const long long row = blockIdx.x;
        for (int e = threadIdx.x; e < experts; e += blockDim.x)
            sel[e] = __fadd_rn(sigmoidf_ref(logits[row * experts + e]), bias[e]);
        __syncthreads();
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
        for (int j = 0; j < k; j++)
        {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
            for (int e = threadIdx.x; e < experts; e += blockDim.x)
                if (sel[e] > bv || (sel[e] == bv && e < bi)) { bv = sel[e]; bi = e; }
            for (int o = 16; o > 0; o >>= 1)
            {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (lane == 0) { best_v[warp] = bv; best_i[warp] = bi; }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                float v = best_v[0];
                int idx = best_i[0];
                for (int w = 1; w < warps; w++)
                    if (best_v[w] > v || (best_v[w] == v && best_i[w] < idx)) { v = best_v[w]; idx = best_i[w]; }
                ids[row * k + j] = idx;
                sel[idx] = -INFINITY;
            }
            __syncthreads();
        }
    }

    __global__ void inkling_route_weights(const float* __restrict__ logits, const int32_t* __restrict__ ids,
                                          float* __restrict__ weights, int experts, int k, long long rows, float scale)
    {
        const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= rows) return;
        float picked[32];
        float sum = 0.0f;
        for (int j = 0; j < k; j++)
        {
            picked[j] = sigmoidf_ref(logits[row * experts + ids[row * k + j]]);
            sum += picked[j];
        }
        sum = __fadd_rn(sum, 1e-20f);
        for (int j = 0; j < k; j++) weights[row * k + j] = __fmul_rn(picked[j] / sum, scale);
    }

    // ---- SITU over gate|up
    __global__ void inkling_situ(const float* __restrict__ gu, float* __restrict__ out, long long ff, long long vectors,
                                 float beta, float lb)
    {
        const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= ff * vectors) return;
        const long long vec = i / ff, e = i - vec * ff;
        const float g = gu[vec * 2 * ff + e], u = gu[vec * 2 * ff + ff + e];
        float act = __fmul_rn(tanhf(__fmul_rn(g, 1.0f / beta)), beta);
        act = __fmul_rn(act, sigmoidf_ref(g));
        const float uu = lb > 0.0f ? __fmul_rn(tanhf(__fmul_rn(u, 1.0f / lb)), lb) : u;
        out[i] = __fmul_rn(act, uu);
    }

    // ---- the experts' weighted sum, in expert order (as the region's product and its additions)
    __global__ void inkling_expert_sum(const float* __restrict__ down, const float* __restrict__ weights, float* __restrict__ out,
                                       long long h, int k, long long rows)
    {
        const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= h * rows) return;
        const long long row = i / h, e = i - row * h;
        const float* d = down + row * k * h + e;
        const float* w = weights + row * k;
        float sum = __fmul_rn(d[0], w[0]);
        for (int j = 1; j < k; j++) sum = __fadd_rn(sum, __fmul_rn(d[(long long)j * h], w[j]));
        out[i] = sum;
    }

    unsigned int blocks(long long n, int per) { return (unsigned int)((n + per - 1) / per); }

    constexpr int kMaxHeadDim = 256, kMaxGroup = 8;

    template <typename KV>
    __device__ __forceinline__ float kv_load(const KV* p);
    template <> __device__ __forceinline__ float kv_load<float>(const float* p) { return *p; }
    template <> __device__ __forceinline__ float kv_load<half>(const half* p) { return __half2float(*p); }

    struct AttnArgs
    {
        const char* q; long long q_nb1, q_nb2;
        const char* k; long long k_nb1, k_nb2;
        const char* v; long long v_nb1, v_nb2;
        const float* T;              // [extent + 1, heads, rows]: B r - t by distance, the last entry - t
        const int32_t* positions;
        float* out;
        int d, dv, heads, kv_heads, keys, extent, rows;
        int ring, band, first_key, floor, row_offset;
        int kv_aligned;              // K and V rows start on 16 bytes (vector loads, cp.async)
        float scale;
    };

    __device__ __forceinline__ long long row_position(const AttnArgs& a, int i) { return a.positions[a.row_offset + i]; }

    // Key j's position: a ring cell holds the last position congruent to it the rows' last position reached; a linear
    // window's keys follow its first.
    __device__ __forceinline__ long long key_position(const AttnArgs& a, int j, long long last)
    {
        if (a.ring > 0) return j <= last ? (long long)j + (last - j) / a.ring * a.ring : -1;
        return (long long)a.first_key + j;
    }

    __device__ __forceinline__ bool sees(const AttnArgs& a, long long p, long long kp)
    {
        return kp >= 0 && kp >= a.floor && kp <= p && (a.band == 0 || p - kp < a.band);
    }

    // The lowest key position a row at p sees.
    __device__ __forceinline__ long long lowest_seen(const AttnArgs& a, long long p)
    {
        long long lo = a.floor > 0 ? a.floor : 0;
        if (a.band > 0 && p - a.band + 1 > lo) lo = p - a.band + 1;
        return lo;
    }

    // Row i of head h's bias by distance (the table's last entry past the extent).
    __device__ __forceinline__ const float* bias_row(const AttnArgs& a, int i, int h)
    {
        return a.T + ((long long)i * a.heads + h) * (a.extent + 1);
    }

    // ---- a few rows (decode, verify windows), split over the keys: one block takes a KV head's G query heads over
    // kSplitKeys keys of one row, a thread a key (its K row read once for the G heads), the block's softmax and its values,
    // and writes its partial (max, sum, values) per head; inkling_attention_combine merges the splits with the zero sink.
    // G (query heads a KV head) and the 128-wide heads are compile-time, so a thread's per-head sums stay in registers.
    constexpr int kSplitKeys = 256, kSplitThreads = 256, kSplitWidth = 128;

    template <typename KV, int G>
    __global__ void __launch_bounds__(kSplitThreads) inkling_attention_split(AttnArgs a, float* __restrict__ part, int splits)
    {
        constexpr int D = kSplitWidth, DV = kSplitWidth;
        __shared__ float qs[G * D];
        __shared__ float sc[G][kSplitKeys];
        __shared__ float red[G][32];
        __shared__ float mx[G], sm[G];
        const int g = blockIdx.x, split = blockIdx.y, i = blockIdx.z;
        const int k0 = split * kSplitKeys, k1 = min(a.keys, k0 + kSplitKeys);
        const long long p = row_position(a, i), last = row_position(a, a.rows - 1);
        float* out = part + (((long long)i * a.kv_heads + g) * splits + split) * G * (2 + DV);
        if (a.ring == 0)
        {
            // A linear window's split the row sees no key of: an empty partial.
            const long long lo = (long long)a.first_key + k0, hi = (long long)a.first_key + k1 - 1;
            if (k0 >= k1 || hi < lowest_seen(a, p) || lo > p)
            {
                for (int o = threadIdx.x; o < G * (2 + DV); o += blockDim.x) out[o] = o % (2 + DV) == 0 ? -INFINITY : 0.0f;
                return;
            }
        }
        for (int idx = threadIdx.x; idx < G * D; idx += blockDim.x)
        {
            const int hh = idx / D, e = idx - hh * D;
            qs[idx] = ((const float*)(a.q + (long long)i * a.q_nb1 + (long long)(g * G + hh) * a.q_nb2))[e];
        }
        __syncthreads();
        for (int j = k0 + threadIdx.x; j < k0 + kSplitKeys; j += blockDim.x)
        {
            const int jj = j - k0;
            const long long kp = j < k1 ? key_position(a, j, last) : -1;
            if (j >= k1 || !sees(a, p, kp))
            {
                for (int hh = 0; hh < G; hh++) sc[hh][jj] = -INFINITY;
                continue;
            }
            const long long dist = p - kp;
            const int bi = dist < a.extent ? (int)dist : a.extent;
            const KV* krow = (const KV*)(a.k + (long long)j * a.k_nb1 + (long long)g * a.k_nb2);
            float dot[G];
#pragma unroll
            for (int hh = 0; hh < G; hh++) dot[hh] = 0.0f;
            if (a.kv_aligned)
            {
                // The row in 16-byte loads (8 halves or 4 floats a load).
                constexpr int per = 16 / sizeof(KV);
                for (int e0 = 0; e0 < D; e0 += per)
                {
                    const uint4 raw = *reinterpret_cast<const uint4*>(krow + e0);
                    float kv[per];
                    if constexpr (sizeof(KV) == 2)
                    {
                        const half2* h2 = reinterpret_cast<const half2*>(&raw);
                        for (int u = 0; u < per / 2; u++) { const float2 f = __half22float2(h2[u]); kv[2 * u] = f.x; kv[2 * u + 1] = f.y; }
                    }
                    else
                    {
                        const float* f = reinterpret_cast<const float*>(&raw);
                        for (int u = 0; u < per; u++) kv[u] = f[u];
                    }
#pragma unroll
                    for (int u = 0; u < per; u++)
#pragma unroll
                        for (int hh = 0; hh < G; hh++) dot[hh] += qs[hh * D + e0 + u] * kv[u];
                }
            }
            else
            {
                for (int e = 0; e < D; e++)
                {
                    const float kv = kv_load<KV>(krow + e);
#pragma unroll
                    for (int hh = 0; hh < G; hh++) dot[hh] += qs[hh * D + e] * kv;
                }
            }
#pragma unroll
            for (int hh = 0; hh < G; hh++) sc[hh][jj] = dot[hh] * a.scale + bias_row(a, i, g * G + hh)[bi];
        }
        __syncthreads();
        // Each head's max and sum over the split's keys.
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
        for (int hh = 0; hh < G; hh++)
        {
            float v = -INFINITY;
            for (int jj = threadIdx.x; jj < kSplitKeys; jj += blockDim.x) v = fmaxf(v, sc[hh][jj]);
            v = warp_max(v);
            if (lane == 0) red[hh][warp] = v;
        }
        __syncthreads();
        if (threadIdx.x < G)
        {
            float v = -INFINITY;
            for (int w = 0; w < warps; w++) v = fmaxf(v, red[threadIdx.x][w]);
            mx[threadIdx.x] = v;
        }
        __syncthreads();
        for (int hh = 0; hh < G; hh++)
        {
            float v = 0.0f;
            for (int jj = threadIdx.x; jj < kSplitKeys; jj += blockDim.x)
            {
                const float e = sc[hh][jj] == -INFINITY ? 0.0f : expf(sc[hh][jj] - mx[hh]);
                sc[hh][jj] = e;
                v += e;
            }
            v = warp_sum(v);
            if (lane == 0) red[hh][warp] = v;
        }
        __syncthreads();
        if (threadIdx.x < G)
        {
            float v = 0.0f;
            for (int w = 0; w < warps; w++) v += red[threadIdx.x][w];
            sm[threadIdx.x] = v;
        }
        __syncthreads();
        // The split's values: each warp a run of keys, a lane `per` adjacent elements of every head of the group (a warp
        // reads a V row whole), the warps' sums added in shared memory (the qs buffer, no longer read).
        constexpr int per = DV / 32;
        const int run = (kSplitKeys + warps - 1) / warps;
        float vacc[G][per];
#pragma unroll
        for (int hh = 0; hh < G; hh++)
#pragma unroll
            for (int u = 0; u < per; u++) vacc[hh][u] = 0.0f;
        for (int j = k0 + warp * run; j < min(k1, k0 + (warp + 1) * run); j++)
        {
            float w[G];
            bool any = false;
#pragma unroll
            for (int hh = 0; hh < G; hh++) { w[hh] = sc[hh][j - k0]; any |= w[hh] != 0.0f; }
            if (!any) continue;   // uniform across the warp
            const KV* vrow = (const KV*)(a.v + (long long)j * a.v_nb1 + (long long)g * a.v_nb2);
#pragma unroll
            for (int u = 0; u < per; u++)
            {
                const float v = kv_load<KV>(vrow + lane * per + u);
#pragma unroll
                for (int hh = 0; hh < G; hh++) vacc[hh][u] += w[hh] * v;
            }
        }
        // The warps' sums into [G][DV] in shared memory (the qs buffer: D == DV).
        __syncthreads();
        for (int o = threadIdx.x; o < G * DV; o += blockDim.x) qs[o] = 0.0f;
        __syncthreads();
#pragma unroll
        for (int hh = 0; hh < G; hh++)
#pragma unroll
            for (int u = 0; u < per; u++) atomicAdd(&qs[hh * DV + lane * per + u], vacc[hh][u]);
        __syncthreads();
        for (int o = threadIdx.x; o < G * DV; o += blockDim.x)
        {
            const int hh = o / DV, ee = o - hh * DV;
            out[hh * (2 + DV) + 2 + ee] = qs[o];
        }
        if (threadIdx.x < G)
        {
            out[threadIdx.x * (2 + DV)] = mx[threadIdx.x];
            out[threadIdx.x * (2 + DV) + 1] = sm[threadIdx.x];
        }
    }

    // The splits of a (head, row) merged with the zero sink (a logit of 0 with a zero value).
    __global__ void inkling_attention_combine(const float* __restrict__ part, float* __restrict__ out, int heads, int kv_heads,
                                              int dv, int splits)
    {
        const int h = blockIdx.x, i = blockIdx.y;
        const int G = heads / kv_heads, g = h / G, hh = h - g * G;
        const float* base = part + (((long long)i * kv_heads + g) * splits) * G * (2 + dv) + hh * (2 + dv);
        const long long stride = (long long)G * (2 + dv);
        float m = 0.0f;
        for (int s = 0; s < splits; s++) m = fmaxf(m, base[s * stride]);
        float l = expf(0.0f - m);
        for (int s = 0; s < splits; s++)
        {
            const float ms = base[s * stride];
            if (ms != -INFINITY) l += base[s * stride + 1] * expf(ms - m);
        }
        for (int e = threadIdx.x; e < dv; e += blockDim.x)
        {
            float acc = 0.0f;
            for (int s = 0; s < splits; s++)
            {
                const float ms = base[s * stride];
                if (ms != -INFINITY) acc += base[s * stride + 2 + e] * expf(ms - m);
            }
            out[((long long)i * heads + h) * dv + e] = acc / l;
        }
    }

    // ---- a prompt chunk on tensor cores (128-wide heads), after TensorSharp's vision_mma_kernel: one block owns 64 rows of
    // one head (four warps of 16), Q rounded to F16 and held in registers as MMA fragments, the keys in 64-key K/V tiles in
    // shared memory (cp.async from an aligned F16 cache, converted from F32 otherwise); scores, the online softmax (the
    // zero sink starting it) and the values accumulate in F32 with each score's bias gathered from the row's table. A linear
    // window's block walks only the keys its rows see.
    __device__ __forceinline__ void mma_f16_16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
        __trap();
#endif
    }

    __device__ __forceinline__ uint32_t pack_half2(float low, float high)
    {
        const half2 value = __floats2half2_rn(low, high);
        return *reinterpret_cast<const uint32_t*>(&value);
    }

    __device__ __forceinline__ uint32_t load_half2(const half* p) { return *reinterpret_cast<const uint32_t*>(p); }

    __device__ __forceinline__ void cp_async_16(void* shared, const void* global)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(address), "l"(global));
#else
        *static_cast<uint4*>(shared) = *static_cast<const uint4*>(global);
#endif
    }

    __device__ __forceinline__ void cp_async_wait_all()
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
#endif
    }

    __device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& b0, uint32_t& b1, const half* row)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(row));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(address));
#else
        b0 = b1 = 0;
        __trap();
#endif
    }

    constexpr int kMmaWidth = 128, kMmaRow = kMmaWidth + 8, kMmaKeys = 64, kMmaWarps = 4, kMmaQueries = kMmaWarps * 16;
    // K and V tiles; Q is staged in the same memory before the key loop (kMmaQueries <= 2 * kMmaKeys rows).
    constexpr size_t kMmaSmemBytes = sizeof(half) * 2 * kMmaKeys * kMmaRow;

    template <typename KV>
    __global__ __launch_bounds__(kMmaWarps * 32) void inkling_attention_mma(AttnArgs a)
    {
        constexpr int W = kMmaWidth, ROW = kMmaRow, KEYS = kMmaKeys, THREADS = kMmaWarps * 32;
        extern __shared__ __align__(16) unsigned char smem_raw[];
        half* keys = reinterpret_cast<half*>(smem_raw);
        half* values = keys + KEYS * ROW;
        half* queries = keys;   // staged before the key loop
        const int head = blockIdx.y, kvh = head / (a.heads / a.kv_heads), tid = threadIdx.x;
        const int warp = tid >> 5, lane = tid & 31, group = lane >> 2, quad = lane & 3;
        const int first_row = blockIdx.x * kMmaQueries;
        for (int i = tid; i < kMmaQueries * W; i += THREADS)
        {
            const int r = i / W, x = i - r * W, row = first_row + r;
            queries[r * ROW + x] = __float2half(row < a.rows ? ((const float*)(a.q + (long long)row * a.q_nb1 + (long long)head * a.q_nb2))[x] : 0.0f);
        }
        __syncthreads();
        uint32_t qa[W / 16][4];
#pragma unroll
        for (int kk = 0; kk < W / 16; ++kk)
        {
            const half* base = queries + (warp * 16) * ROW + kk * 16 + quad * 2;
            qa[kk][0] = load_half2(base + group * ROW);
            qa[kk][1] = load_half2(base + (group + 8) * ROW);
            qa[kk][2] = load_half2(base + group * ROW + 8);
            qa[kk][3] = load_half2(base + (group + 8) * ROW + 8);
        }
        const int r0 = first_row + warp * 16 + group, r1 = r0 + 8;
        const long long p0 = r0 < a.rows ? row_position(a, r0) : -1, p1 = r1 < a.rows ? row_position(a, r1) : -1;
        const float* T0 = r0 < a.rows ? bias_row(a, r0, head) : nullptr;
        const float* T1 = r1 < a.rows ? bias_row(a, r1, head) : nullptr;
        const long long last = row_position(a, a.rows - 1);
        constexpr float LOG2E = 1.4426950408889634f;

        float acc[W / 8][4];
#pragma unroll
        for (int n = 0; n < W / 8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
        // The sink: a zero logit with a zero value; the running maxima start at it, and one lane of a row counts its weight.
        float max0 = 0.0f, max1 = 0.0f, sum0 = quad == 0 ? 1.0f : 0.0f, sum1 = quad == 0 ? 1.0f : 0.0f;

        int key_lo = 0, key_hi = a.keys;
        if (a.ring == 0)
        {
            const long long p_lo = row_position(a, first_row), p_hi = row_position(a, min(a.rows, first_row + kMmaQueries) - 1);
            const long long lo = lowest_seen(a, p_lo) - a.first_key, hi = p_hi - a.first_key + 1;
            key_lo = (int)(max(0LL, min((long long)a.keys, lo)) / KEYS * KEYS);
            key_hi = (int)max(0LL, min((long long)a.keys, hi));
        }
        for (int k0 = key_lo; k0 < key_hi; k0 += KEYS)
        {
            __syncthreads();
            if (sizeof(KV) == 2 && a.kv_aligned)
            {
                for (int i = tid; i < KEYS * (W / 8); i += THREADS)
                {
                    const int r = i / (W / 8), c = i - r * (W / 8), key = k0 + r;
                    half* kd = keys + r * ROW + c * 8;
                    half* vd = values + r * ROW + c * 8;
                    if (key < a.keys)
                    {
                        cp_async_16(kd, a.k + (long long)key * a.k_nb1 + (long long)kvh * a.k_nb2 + c * 16);
                        cp_async_16(vd, a.v + (long long)key * a.v_nb1 + (long long)kvh * a.v_nb2 + c * 16);
                    }
                    else
                    {
                        *reinterpret_cast<uint4*>(kd) = make_uint4(0, 0, 0, 0);
                        *reinterpret_cast<uint4*>(vd) = make_uint4(0, 0, 0, 0);
                    }
                }
                cp_async_wait_all();
            }
            else
            {
                for (int i = tid; i < KEYS * W; i += THREADS)
                {
                    const int r = i / W, x = i - r * W, key = k0 + r;
                    float kx = 0.0f, vx = 0.0f;
                    if (key < a.keys)
                    {
                        kx = kv_load<KV>((const KV*)(a.k + (long long)key * a.k_nb1 + (long long)kvh * a.k_nb2) + x);
                        vx = kv_load<KV>((const KV*)(a.v + (long long)key * a.v_nb1 + (long long)kvh * a.v_nb2) + x);
                    }
                    keys[r * ROW + x] = __float2half(kx);
                    values[r * ROW + x] = __float2half(vx);
                }
            }
            __syncthreads();

            float s[KEYS / 8][4];
#pragma unroll
            for (int j = 0; j < KEYS / 8; ++j)
            {
                s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
                for (int kk = 0; kk < W / 16; ++kk)
                {
                    const half* kb = keys + (j * 8 + group) * ROW + kk * 16 + quad * 2;
                    mma_f16_16816(s[j], qa[kk], load_half2(kb), load_half2(kb + 8));
                }
            }
            float tile0 = max0, tile1 = max1;
#pragma unroll
            for (int j = 0; j < KEYS / 8; ++j)
            {
#pragma unroll
                for (int c = 0; c < 4; ++c)
                {
                    const int key = k0 + j * 8 + quad * 2 + (c & 1);
                    const long long p = c < 2 ? p0 : p1;
                    float val = -CUDART_INF_F;
                    if (key < a.keys && p >= 0)
                    {
                        const long long kp = key_position(a, key, last);
                        if (sees(a, p, kp))
                        {
                            const long long dist = p - kp;
                            const float* T = c < 2 ? T0 : T1;
                            val = (s[j][c] * a.scale + T[dist < a.extent ? dist : a.extent]) * LOG2E;
                        }
                    }
                    s[j][c] = val;
                }
                tile0 = fmaxf(tile0, fmaxf(s[j][0], s[j][1]));
                tile1 = fmaxf(tile1, fmaxf(s[j][2], s[j][3]));
            }
#pragma unroll
            for (int offset = 1; offset < 4; offset <<= 1)
            {
                tile0 = fmaxf(tile0, __shfl_xor_sync(0xffffffffu, tile0, offset));
                tile1 = fmaxf(tile1, __shfl_xor_sync(0xffffffffu, tile1, offset));
            }
            const float correction0 = exp2f(max0 - tile0), correction1 = exp2f(max1 - tile1);
            max0 = tile0;
            max1 = tile1;
            sum0 *= correction0;
            sum1 *= correction1;
#pragma unroll
            for (int n = 0; n < W / 8; ++n)
            {
                acc[n][0] *= correction0;
                acc[n][1] *= correction0;
                acc[n][2] *= correction1;
                acc[n][3] *= correction1;
            }
#pragma unroll
            for (int j = 0; j < KEYS / 8; ++j)
            {
                s[j][0] = exp2f(s[j][0] - max0);
                s[j][1] = exp2f(s[j][1] - max0);
                s[j][2] = exp2f(s[j][2] - max1);
                s[j][3] = exp2f(s[j][3] - max1);
                sum0 += s[j][0] + s[j][1];
                sum1 += s[j][2] + s[j][3];
            }
#pragma unroll
            for (int kk = 0; kk < KEYS / 16; ++kk)
            {
                // The score accumulators of two adjacent 8-key tiles are the A fragment of the 16-key slice they cover.
                const uint32_t pa[4] = {
                    pack_half2(s[2 * kk][0], s[2 * kk][1]), pack_half2(s[2 * kk][2], s[2 * kk][3]),
                    pack_half2(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_half2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
                const half* vrow = values + (kk * 16 + (lane & 15)) * ROW;
#pragma unroll
                for (int n = 0; n < W / 8; ++n)
                {
                    uint32_t b0, b1;
                    ldmatrix_x2_trans(b0, b1, vrow + n * 8);
                    mma_f16_16816(acc[n], pa, b0, b1);
                }
            }
        }
#pragma unroll
        for (int offset = 1; offset < 4; offset <<= 1)
        {
            sum0 += __shfl_xor_sync(0xffffffffu, sum0, offset);
            sum1 += __shfl_xor_sync(0xffffffffu, sum1, offset);
        }
        const float inverse0 = 1.0f / sum0, inverse1 = 1.0f / sum1;
#pragma unroll
        for (int half_row = 0; half_row < 2; ++half_row)
        {
            const int row = half_row ? r1 : r0;
            if (row >= a.rows) continue;
            const float inverse = half_row ? inverse1 : inverse0;
            float* o = a.out + ((long long)row * a.heads + head) * a.dv;
#pragma unroll
            for (int n = 0; n < W / 8; ++n)
            {
                const int x = n * 8 + quad * 2;
                o[x] = acc[n][half_row * 2] * inverse;
                o[x + 1] = acc[n][half_row * 2 + 1] * inverse;
            }
        }
    }

    // Rows up to this many take the split kernel; more take the tensor-core one.
    constexpr int kSplitMaxRows = 16;

    template <typename KV>
    void launch_mma(const AttnArgs& a, cudaStream_t stream)
    {
        static bool attr = false;
        if (!attr)
        {
            CUDA_CHECK(cudaFuncSetAttribute(inkling_attention_mma<KV>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kMmaSmemBytes));
            attr = true;
        }
        const dim3 grid((unsigned int)((a.rows + kMmaQueries - 1) / kMmaQueries), (unsigned int)a.heads);
        inkling_attention_mma<KV><<<grid, kMmaWarps * 32, kMmaSmemBytes, stream>>>(a);
    }

    template <typename KV>
    void launch_split(const AttnArgs& a, float* part, int splits, dim3 grid, cudaStream_t stream)
    {
        switch (a.heads / a.kv_heads)
        {
            case 1: inkling_attention_split<KV, 1><<<grid, kSplitThreads, 0, stream>>>(a, part, splits); break;
            case 2: inkling_attention_split<KV, 2><<<grid, kSplitThreads, 0, stream>>>(a, part, splits); break;
            case 3: inkling_attention_split<KV, 3><<<grid, kSplitThreads, 0, stream>>>(a, part, splits); break;
            case 4: inkling_attention_split<KV, 4><<<grid, kSplitThreads, 0, stream>>>(a, part, splits); break;
            case 5: inkling_attention_split<KV, 5><<<grid, kSplitThreads, 0, stream>>>(a, part, splits); break;
            case 6: inkling_attention_split<KV, 6><<<grid, kSplitThreads, 0, stream>>>(a, part, splits); break;
            case 7: inkling_attention_split<KV, 7><<<grid, kSplitThreads, 0, stream>>>(a, part, splits); break;
            case 8: inkling_attention_split<KV, 8><<<grid, kSplitThreads, 0, stream>>>(a, part, splits); break;
            default: GGML_ABORT("inkling attention: %d query heads a KV head", a.heads / a.kv_heads);
        }
    }
}

int64_t tsg_inkling_attention_workspace_floats(int64_t rows, int64_t heads, int64_t kv_heads, int64_t keys, int64_t dv)
{
    if (rows > kSplitMaxRows || kv_heads <= 0) return 0;
    const int64_t splits = (keys + kSplitKeys - 1) / kSplitKeys;
    return rows * kv_heads * splits * (heads / kv_heads) * (2 + dv);
}

bool tsg_inkling_attention_supported(int d, int dv, int heads, int kv_heads)
{
    if (kv_heads < 1 || heads % kv_heads != 0) return false;
    const int g = heads / kv_heads;
    // The split kernel takes any width to 256; the tensor-core one 128-wide heads.
    return g <= kMaxGroup && d == kMmaWidth && dv == kMmaWidth;
}

void tsg_inkling_cuda_compute(ggml_tensor * dst, ggml_backend_t backend)
{
    auto * context = static_cast<ggml_backend_cuda_context *>(backend->context);
    CUDA_CHECK(cudaSetDevice(context->device));
    const cudaStream_t stream = context->stream(context->device, 0);
    ggml_custom_op_params params;
    memcpy(&params, dst->op_params, sizeof(params));
    const auto * d = static_cast<const tsg_dsv4_fused_desc *>(params.userdata);
    switch (d->kind)
    {
        case TSG_INKLING_CLIP:
        {
            const ggml_tensor * x = dst->src[0];
            const ggml_tensor * w = dst->src[1];
            const int dim = (int)x->ne[0], heads = (int)x->ne[1];
            const long long vectors = ggml_nelements(x) / dim;
            inkling_clip<<<blocks(vectors, 8), 256, 0, stream>>>((const float *)x->data, w ? (const float *)w->data : nullptr,
                (float *)dst->data, dim, heads, vectors, d->f0);
        } break;
        case TSG_INKLING_CONV:
        {
            const ggml_tensor * x = dst->src[0], * window = dst->src[1], * kernel = dst->src[2], * dyn = dst->src[3];
            const int channels = (int)x->ne[0], count = (int)x->ne[1], taps = (int)kernel->ne[0];
            inkling_conv<<<blocks(channels, 128), 128, 0, stream>>>((const float *)x->data, (float *)window->data,
                (const float *)kernel->data, dyn ? (const float *)dyn->data : nullptr, (float *)dst->data, channels, count, taps, d->i0,
                d->f0);
        } break;
        case TSG_INKLING_MIX:
        {
            const ggml_tensor * states = dst->src[0], * query = dst->src[1];
            const int h = (int)states->ne[0], n = (int)states->ne[1], rows = (int)states->ne[2], queries = (int)query->ne[1];
            // The partials live past the result, in the node's tail (the builder sizes it).
            const int chunks = (h + kMixChunk - 1) / kMixChunk;
            float * partial = (float *)dst->data + (long long)h * queries * rows;
            inkling_mix_partial<<<dim3(chunks, rows), kMixChunk, 0, stream>>>((const char *)states->data, (long long)states->nb[1],
                (long long)states->nb[2], n, (const float *)query->data, queries, partial, h);
            inkling_mix_output<<<dim3(chunks, rows), kMixChunk, 0, stream>>>((const char *)states->data, (long long)states->nb[1],
                (long long)states->nb[2], n, queries, partial, (float *)dst->data, h, d->f0, d->f1);
        } break;
        case TSG_INKLING_CONTROL:
        {
            const ggml_tensor * x = dst->src[0], * m = dst->src[1], * down = dst->src[2], * gain = dst->src[3], * forget = dst->src[4];
            const int h = (int)x->ne[0], rows = (int)x->ne[1], rank = (int)down->ne[1];
            float * y = (float *)dst->data;
            float * g = y + (long long)h * rows;
            inkling_control_down<<<dim3(blocks(rank, 8), rows), 256, 0, stream>>>((const float *)x->data, (const float *)m->data,
                (const float *)down->data, g, h, rank, d->f0);
            inkling_control_blend<<<dim3(blocks(h, 8), rows), 256, 0, stream>>>((const float *)x->data, (const float *)m->data,
                (const float *)gain->data, (const float *)forget->data, g, y, h, rank);
        } break;
        case TSG_INKLING_ATTENTION:
        {
            const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * table = dst->src[3], * positions = dst->src[4];
            AttnArgs a{};
            a.q = (const char *)q->data; a.q_nb1 = (long long)q->nb[1]; a.q_nb2 = (long long)q->nb[2];
            a.k = (const char *)k->data; a.k_nb1 = (long long)k->nb[1]; a.k_nb2 = (long long)k->nb[2];
            a.v = (const char *)v->data; a.v_nb1 = (long long)v->nb[1]; a.v_nb2 = (long long)v->nb[2];
            a.T = (const float *)table->data;
            a.positions = (const int32_t *)positions->data;
            a.out = (float *)dst->data;
            a.d = (int)q->ne[0]; a.dv = (int)v->ne[0]; a.heads = (int)q->ne[2]; a.kv_heads = (int)k->ne[2];
            a.keys = (int)k->ne[1]; a.rows = (int)q->ne[1];
            a.ring = d->i0; a.extent = d->i1; a.band = d->i2; a.first_key = d->i3; a.floor = d->i4; a.row_offset = d->i5;
            a.scale = d->f0;
            a.kv_aligned = ((uintptr_t)a.k % 16 == 0 && a.k_nb1 % 16 == 0 && a.k_nb2 % 16 == 0 && (uintptr_t)a.v % 16 == 0
                            && a.v_nb1 % 16 == 0 && a.v_nb2 % 16 == 0) ? 1 : 0;
            if (a.rows > kSplitMaxRows)
            {
                if (k->type == GGML_TYPE_F16) launch_mma<half>(a, stream);
                else launch_mma<float>(a, stream);
            }
            else
            {
                const int splits = (a.keys + kSplitKeys - 1) / kSplitKeys;
                float * part = a.out + (long long)a.rows * a.heads * a.dv;
                const dim3 sgrid((unsigned int)a.kv_heads, (unsigned int)splits, (unsigned int)a.rows);
                // TS_INKLING_ATTN_PROFILE=1 (with TS_INKLING_CAPTURE=0): the split and combine kernels' device time, waited on.
                static const bool profile = std::getenv("TS_INKLING_ATTN_PROFILE") != nullptr;
                static cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr;
                if (profile && e0 == nullptr) { cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); }
                if (profile) cudaEventRecord(e0, stream);
                if (k->type == GGML_TYPE_F16) launch_split<half>(a, part, splits, sgrid, stream);
                else launch_split<float>(a, part, splits, sgrid, stream);
                if (profile) cudaEventRecord(e1, stream);
                inkling_attention_combine<<<dim3((unsigned int)a.heads, (unsigned int)a.rows), 128, 0, stream>>>(part, a.out, a.heads,
                    a.kv_heads, a.dv, splits);
                if (profile)
                {
                    cudaEventRecord(e2, stream);
                    cudaEventSynchronize(e2);
                    float ms_split = 0, ms_combine = 0;
                    cudaEventElapsedTime(&ms_split, e0, e1);
                    cudaEventElapsedTime(&ms_combine, e1, e2);
                    static double sum_split[2] = {}, sum_combine[2] = {}, bytes[2] = {};
                    static long long n[2] = {};
                    const int big = a.keys > 4096 ? 1 : 0;
                    sum_split[big] += ms_split; sum_combine[big] += ms_combine; n[big]++;
                    bytes[big] += 2.0 * a.keys * a.kv_heads * a.d * ggml_type_size(k->type);
                    if (n[big] % 256 == 0)
                        std::fprintf(stderr, "[inkling-attn] %s: %lld launches, keys %d, rows %d: split %.1f us, combine %.1f us, K+V %.1f GB/s\n",
                            big ? "long" : "short", n[big], a.keys, a.rows, 1000 * sum_split[big] / n[big], 1000 * sum_combine[big] / n[big],
                            bytes[big] / (sum_split[big] * 1e-3) / 1e9);
                }
            }
            CUDA_CHECK(cudaGetLastError());
        } break;
        case TSG_INKLING_ROUTE_IDS:
        {
            const ggml_tensor * logits = dst->src[0], * bias = dst->src[1];
            const int experts = (int)logits->ne[0], rows = (int)logits->ne[1];
            inkling_route_ids<<<rows, 128, 0, stream>>>((const float *)logits->data, (const float *)bias->data, (int32_t *)dst->data,
                experts, d->i0);
        } break;
        case TSG_INKLING_ROUTE_WEIGHTS:
        {
            const ggml_tensor * logits = dst->src[0], * ids = dst->src[1];
            const int experts = (int)logits->ne[0];
            const long long rows = logits->ne[1];
            inkling_route_weights<<<blocks(rows, 64), 64, 0, stream>>>((const float *)logits->data, (const int32_t *)ids->data,
                (float *)dst->data, experts, d->i0, rows, d->f0);
        } break;
        case TSG_INKLING_SITU:
        {
            const ggml_tensor * gu = dst->src[0];
            const long long ff = gu->ne[0] / 2, vectors = gu->ne[1] * gu->ne[2] * gu->ne[3];
            inkling_situ<<<blocks(ff * vectors, 256), 256, 0, stream>>>((const float *)gu->data, (float *)dst->data, ff, vectors, d->f0, d->f1);
        } break;
        case TSG_INKLING_EXPERT_SUM:
        {
            const ggml_tensor * down = dst->src[0], * weights = dst->src[1];
            const long long h = down->ne[0], rows = down->ne[2];
            inkling_expert_sum<<<blocks(h * rows, 256), 256, 0, stream>>>((const float *)down->data, (const float *)weights->data,
                (float *)dst->data, h, (int)down->ne[1], rows);
        } break;
        default:
            GGML_ABORT("tsg_inkling_cuda_compute: unknown kind %d", d->kind);
    }
    CUDA_CHECK(cudaGetLastError());
}

namespace
{
    struct InklingCapture
    {
        cudaGraphExec_t exec = nullptr;
        int runs = 0;
        bool failed = false;
    };
}

void * tsg_inkling_capture_new() { return new InklingCapture(); }

void tsg_inkling_capture_free(void * capture)
{
    auto * c = static_cast<InklingCapture *>(capture);
    if (c == nullptr) return;
    if (c->exec != nullptr) cudaGraphExecDestroy(c->exec);
    delete c;
}

ggml_status tsg_inkling_compute(void * capture, ggml_backend_t fused_backend, ggml_backend_t cuda_backend, ggml_cgraph * graph,
                                int warmup)
{
    auto * c = static_cast<InklingCapture *>(capture);
    // A graph ggml-cuda would run with a stream synchronize in it (an expert product on its host-sorted fallback) is never
    // captured: the synchronize fails the capture inside ggml-cuda, which aborts the process rather than returning.
    if (c != nullptr && c->runs == 0 && !c->failed && !tsg_cuda_graph_capturable(cuda_backend, graph)) c->failed = true;
    if (c != nullptr && !c->failed && c->exec == nullptr)
    {
        // The runs ggml-cuda is handed start at the graph's first node and after each fused node: none captures itself, so
        // the capture below records their kernels (ggml-cuda would otherwise begin a capture on the capturing stream).
        std::vector<const void *> keys;
        bool after_fused = true;
        for (int i = 0; i < graph->n_nodes; i++)
        {
            const bool fused = graph->nodes[i]->op == GGML_OP_CUSTOM;
            if (!fused && after_fused) keys.push_back(graph->nodes[i]);
            after_fused = fused;
        }
        tsg_cuda_disable_graphs(cuda_backend, keys.data(), (int)keys.size());
    }
    if (c == nullptr || c->failed || c->runs < warmup)
    {
        if (c != nullptr) c->runs++;
        return ggml_backend_graph_compute_async(fused_backend, graph);
    }
    auto * context = static_cast<ggml_backend_cuda_context *>(cuda_backend->context);
    CUDA_CHECK(cudaSetDevice(context->device));
    const cudaStream_t stream = context->stream(context->device, 0);
    if (c->exec == nullptr)
    {
        cudaGraph_t g = nullptr;
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed) != cudaSuccess)
        {
            (void)cudaGetLastError();
            c->failed = true;
            return ggml_backend_graph_compute_async(fused_backend, graph);
        }
        const ggml_status st = ggml_backend_graph_compute_async(fused_backend, graph);
        const cudaError_t e = cudaStreamEndCapture(stream, &g);
        if (st != GGML_STATUS_SUCCESS || e != cudaSuccess || g == nullptr
            || cudaGraphInstantiate(&c->exec, g, 0) != cudaSuccess)
        {
            // Nothing ran while capturing: run the graph as it is, and do so from now on.
            (void)cudaGetLastError();
            if (g != nullptr) cudaGraphDestroy(g);
            c->exec = nullptr;
            c->failed = true;
            std::fprintf(stderr, "[inkling] whole-run CUDA graph capture failed (%s); running the graph as it is\n", cudaGetErrorString(e));
            return ggml_backend_graph_compute_async(fused_backend, graph);
        }
        static const bool log = std::getenv("TS_INKLING_CAPTURE_LOG") != nullptr;
        if (log)
        {
            size_t nodes = 0;
            cudaGraphGetNodes(g, nullptr, &nodes);
            std::fprintf(stderr, "[inkling] captured a run of %d graph nodes as %zu CUDA graph nodes\n", graph->n_nodes, nodes);
        }
        cudaGraphDestroy(g);
    }
    static const bool log_replays = std::getenv("TS_INKLING_CAPTURE_LOG") != nullptr;
    static long long replays = 0;
    if (log_replays && ++replays % 64 == 0) std::fprintf(stderr, "[inkling] %lld replays\n", replays);
    CUDA_CHECK(cudaGraphLaunch(c->exec, stream));
    return GGML_STATUS_SUCCESS;
}
#endif
