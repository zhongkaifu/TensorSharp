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
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <chrono>
#include <atomic>

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
                                 int heads, long long vectors, float threshold, long long x_row)
    {
        const long long v = (long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
        if (v >= vectors) return;
        const int lane = threadIdx.x & 31;
        // x's rows at x_row floats (a packed product's slice), its heads d apart; y contiguous.
        const float* xv = x + (v / heads) * x_row + (v % heads) * d;
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
                                 int heads, float scale, long long x_row, long long dyn_row)
    {
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= channels) return;
        const int hist = taps - 1;
        float win[kMaxTaps], wk[kMaxTaps];
        for (int j = 0; j < hist; j++) win[j] = window[(long long)c * hist + j];
        for (int i = 0; i < taps; i++) wk[i] = kw[(long long)c * taps + i];
        const int head = heads > 0 ? c / (channels / heads) : 0;
        const long long width = (long long)taps * heads;
        auto input = [&](int tau) -> float { return tau < hist ? win[tau] : x[(long long)(tau - hist) * x_row + c]; };
        for (int t = 0; t < count; t++)
        {
            float conv = 0.0f;
            for (int i = 0; i < taps; i++) conv += input(t + i) * wk[i];
            if (dyn != nullptr)
                for (int lag = 0; lag < taps; lag++)
                    conv = __fadd_rn(conv, __fmul_rn(input(hist - lag + t), dyn[(long long)t * dyn_row + (long long)head * taps + lag]));
            const float xv = x[(long long)t * x_row + c];
            y[(long long)t * channels + c] = __fmul_rn(__fadd_rn(xv, conv), scale);
        }
        float next[kMaxTaps];
        for (int j = 0; j < hist; j++) next[j] = input(count + j);
        for (int j = 0; j < hist; j++) window[(long long)c * hist + j] = next[j];
    }

    // ---- a prompt chunk's short convolution: a thread a (channel, row) reads its taps from the old window and the chunk,
    // then inkling_conv_window moves each channel's window to the chunk's last rows (after every row has read it).
    __global__ void inkling_conv_rows(const float* __restrict__ x, const float* __restrict__ window, const float* __restrict__ kw,
                                      const float* __restrict__ dyn, float* __restrict__ y, int channels, int count, int taps,
                                      int heads, float scale, long long x_row, long long dyn_row)
    {
        const int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
        if (c >= channels) return;
        const int hist = taps - 1;
        const int head = heads > 0 ? c / (channels / heads) : 0;
        auto input = [&](int tau) -> float { return tau < hist ? window[(long long)c * hist + tau] : x[(long long)(tau - hist) * x_row + c]; };
        float conv = 0.0f;
        for (int i = 0; i < taps; i++) conv += input(t + i) * kw[(long long)c * taps + i];
        if (dyn != nullptr)
            for (int lag = 0; lag < taps; lag++)
                conv = __fadd_rn(conv, __fmul_rn(input(hist - lag + t), dyn[(long long)t * dyn_row + (long long)head * taps + lag]));
        const float xv = x[(long long)t * x_row + c];
        y[(long long)t * channels + c] = __fmul_rn(__fadd_rn(xv, conv), scale);
    }

    __global__ void inkling_conv_window(const float* __restrict__ x, float* __restrict__ window, int channels, int count, int taps,
                                        long long x_row)
    {
        const int c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= channels) return;
        const int hist = taps - 1;
        float next[kMaxTaps];
        for (int j = 0; j < hist; j++)
        {
            const int tau = count + j;
            next[j] = tau < hist ? window[(long long)c * hist + tau] : x[(long long)(tau - hist) * x_row + c];
        }
        for (int j = 0; j < hist; j++) window[(long long)c * hist + j] = next[j];
    }

    // Rows from which a convolution runs parallel over its rows.
    constexpr int kConvRowsParallel = 16;

    // A verify window's history [taps - 1 + count, channels]: the live window, then the rows (the commit picks its window).
    __global__ void inkling_conv_history(const float* __restrict__ x, const float* __restrict__ window, float* __restrict__ history,
                                         int channels, int count, int hist, long long x_row)
    {
        const long long width = hist + count;
        const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= width * channels) return;
        const int c = (int)(i / width), j = (int)(i - (long long)c * width);
        history[i] = j < hist ? window[(long long)c * hist + j] : x[(long long)(j - hist) * x_row + c];
    }

    // ---- several sequences' short convolutions (TSG_INKLING_CONV_BATCH): a thread a channel of a row, its sequence's
    // window from the table; a verify writes each sequence's history [window | rows], else each window advances after.
    struct ConvBatchArgs
    {
        const float* x; long long x_row;          // [channels, rows] at x_row floats a row
        const float* kw; const float* dyn; long long dyn_row;
        float* y;
        int channels, taps, heads, n, rows;
        float scale;
        int first[TSG_INKLING_CONV_BATCH_MAX], count[TSG_INKLING_CONV_BATCH_MAX];
        float* window[TSG_INKLING_CONV_BATCH_MAX];
        float* history[TSG_INKLING_CONV_BATCH_MAX];
    };

    __global__ void inkling_conv_batch_rows(ConvBatchArgs a)
    {
        const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
        if (c >= a.channels) return;
        int s = 0;
        while (s + 1 < a.n && a.first[s + 1] <= r) s++;
        const int t = r - a.first[s], hist = a.taps - 1, base = a.first[s];
        const float* window = a.window[s];
        const int head = a.heads > 0 ? c / (a.channels / a.heads) : 0;
        auto input = [&](int tau) -> float { return tau < hist ? window[(long long)c * hist + tau] : a.x[(long long)(base + tau - hist) * a.x_row + c]; };
        float conv = 0.0f;
        for (int i = 0; i < a.taps; i++) conv += input(t + i) * a.kw[(long long)c * a.taps + i];
        if (a.dyn != nullptr)
            for (int lag = 0; lag < a.taps; lag++)
                conv = __fadd_rn(conv, __fmul_rn(input(hist - lag + t), a.dyn[(long long)r * a.dyn_row + (long long)head * a.taps + lag]));
        const float xv = a.x[(long long)r * a.x_row + c];
        a.y[(long long)r * a.channels + c] = __fmul_rn(__fadd_rn(xv, conv), a.scale);
    }

    // A verify's histories: sequence blockIdx.y's [window | rows] at its pitch (taps - 1 + rows).
    __global__ void inkling_conv_batch_history(ConvBatchArgs a)
    {
        const int s = blockIdx.y, hist = a.taps - 1, count = a.count[s];
        const long long width = hist + count;
        const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= width * a.channels) return;
        const int c = (int)(i / width), j = (int)(i - (long long)c * width);
        a.history[s][i] = j < hist ? a.window[s][(long long)c * hist + j] : a.x[(long long)(a.first[s] + j - hist) * a.x_row + c];
    }

    // Each sequence's window advanced past its rows (after every row read it).
    __global__ void inkling_conv_batch_window(ConvBatchArgs a)
    {
        const int s = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
        if (c >= a.channels) return;
        const int hist = a.taps - 1, count = a.count[s];
        float* window = a.window[s];
        float next[kMaxTaps];
        for (int j = 0; j < hist; j++)
        {
            const int tau = count + j;
            next[j] = tau < hist ? window[(long long)c * hist + tau] : a.x[(long long)(a.first[s] + tau - hist) * a.x_row + c];
        }
        for (int j = 0; j < hist; j++) window[(long long)c * hist + j] = next[j];
    }

    // ---- DSpark's pick: one block a lane, the argmax (the lower index on a tie) and the softcapped softmax's sum together
    // (online over the lane's logits), so conf = 1 / sum exp(capped - capped max).
    __global__ void __launch_bounds__(1024) inkling_pick(const float* __restrict__ base, long long base_col, const float* __restrict__ fold,
        const float* __restrict__ bias, int padded, int vocab, int lanes, float cap, float* __restrict__ out)
    {
        __shared__ float sm_m[32], sm_s[32];
        __shared__ int sm_i[32];
        const int j = blockIdx.x;
        const float* b = base + (long long)j * base_col;
        const float* fl = fold + (long long)j * padded;
        float mx = -INFINITY, sum = 0.0f;
        int idx = 0x7fffffff;
        const float inv = 1.0f / cap;
        for (int i = threadIdx.x; i < vocab; i += blockDim.x)
        {
            const float v = b[i] + (fl[i] + bias[i]);
            const float c = cap * tanhf(v * inv);
            if (c > mx) { sum = sum * __expf(mx - c) + 1.0f; mx = c; idx = i; }
            else sum += __expf(c - mx);
        }
        // Combine (max, sum, index) over the warp, then the warps.
        auto combine = [](float& m1, float& s1, int& i1, float m2, float s2, int i2)
        {
            if (m2 > m1 || (m2 == m1 && i2 < i1))
            {
                s1 = (m1 == -INFINITY ? 0.0f : s1 * __expf(m1 - m2)) + s2;
                m1 = m2;
                i1 = i2;
            }
            else s1 += (m2 == -INFINITY ? 0.0f : s2 * __expf(m2 - m1));
        };
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
        {
            const float m2 = __shfl_xor_sync(0xffffffffu, mx, o), s2 = __shfl_xor_sync(0xffffffffu, sum, o);
            const int i2 = __shfl_xor_sync(0xffffffffu, idx, o);
            combine(mx, sum, idx, m2, s2, i2);
        }
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        if (lane == 0) { sm_m[warp] = mx; sm_s[warp] = sum; sm_i[warp] = idx; }
        __syncthreads();
        if (warp == 0)
        {
            const int warps = blockDim.x >> 5;
            mx = lane < warps ? sm_m[lane] : -INFINITY;
            sum = lane < warps ? sm_s[lane] : 0.0f;
            idx = lane < warps ? sm_i[lane] : 0x7fffffff;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
            {
                const float m2 = __shfl_xor_sync(0xffffffffu, mx, o), s2 = __shfl_xor_sync(0xffffffffu, sum, o);
                const int i2 = __shfl_xor_sync(0xffffffffu, idx, o);
                combine(mx, sum, idx, m2, s2, i2);
            }
            if (lane == 0)
            {
                out[j] = (float)idx;
                out[lanes + j] = sum > 0.0f ? 1.0f / sum : 0.0f;
                out[2 * lanes + j] = (float)idx;
            }
        }
    }

    // The relative-bias table in F16: a block kRelPairs (head, row) pairs, their r in shared memory; a thread a distance at a
    // time, its W column read once for the block's pairs. REL (the relative width) is compile-time: a runtime bound let the
    // unrolled loads of the last column run past the table.
    constexpr int kRelPairs = 8, kRelThreads = 256, kRelWidth = 16;

    template <int REL>
    __global__ void __launch_bounds__(kRelThreads) inkling_rel_table(const float* __restrict__ w, const char* __restrict__ r,
        long long r_nb1, long long r_nb2, int heads, long long pairs, int width, __half* __restrict__ out)
    {
        __shared__ float rs[kRelPairs][REL];
        const long long p0 = (long long)blockIdx.x * kRelPairs;
        for (int i = threadIdx.x; i < kRelPairs * REL; i += blockDim.x)
        {
            const int pp = i / REL, k = i - pp * REL;
            const long long pair = p0 + pp;
            float v = 0.0f;
            if (pair < pairs)
            {
                const long long row = pair / heads, h = pair - row * heads;
                v = reinterpret_cast<const float*>(r + row * r_nb2 + h * r_nb1)[k];
            }
            rs[pp][k] = v;
        }
        __syncthreads();
        const int np = (int)min((long long)kRelPairs, pairs - p0);
        for (int dd = threadIdx.x; dd < width; dd += blockDim.x)
        {
            float wc[REL];
#pragma unroll
            for (int k = 0; k < REL; k++) wc[k] = w[(long long)dd * REL + k];
            for (int pp = 0; pp < np; pp++)
            {
                float acc = 0.0f;
#pragma unroll
                for (int k = 0; k < REL; k++) acc = fmaf(wc[k], rs[pp][k], acc);
                out[(p0 + pp) * width + dd] = __float2half(acc);
            }
        }
    }

    __global__ void inkling_pick_i32(const float* __restrict__ pick, int32_t* __restrict__ ids, int lanes)
    {
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j < lanes) ids[j] = (int32_t)pick[j];
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

    // ---- a prompt row's depth mix in one block: its states into shared memory once, then the norms and dots, the softmax
    // and the weighted sum from there.
    constexpr int kMixRowsFrom = 16;
    constexpr int kMixRowSmemMax = 80 * 1024;

    __global__ void inkling_mix_row(const char* __restrict__ states, long long nb1, long long nb2, int n,
                                    const float* __restrict__ query, int queries, float* __restrict__ out, int h, float eps,
                                    float scale)
    {
        extern __shared__ float held[];   // [n][h]
        __shared__ float scratch[32 * (kMaxStates + kMaxStates * kMaxQueries)];
        __shared__ float rs[kMaxStates];
        __shared__ float probs[kMaxQueries * kMaxStates];
        const long long row = blockIdx.x;
        const char* base = states + row * nb2;
        const int width = n + n * queries;
        float acc[kMaxStates + kMaxStates * kMaxQueries];
        for (int i = 0; i < width; i++) acc[i] = 0.0f;
        for (int e = threadIdx.x; e < h; e += blockDim.x)
            for (int s = 0; s < n; s++)
            {
                const float v = *(const float*)(base + s * nb1 + (long long)e * 4);
                held[s * h + e] = v;
                acc[s] += v * v;
                for (int q = 0; q < queries; q++) acc[n + q * n + s] += query[(long long)q * h + e] * v;
            }
        block_sums(acc, width, scratch);
        if (threadIdx.x < n) rs[threadIdx.x] = 1.0f / sqrtf(acc[threadIdx.x] / h + eps);
        __syncthreads();
        if (threadIdx.x < queries)
        {
            const int q = threadIdx.x;
            float m = -INFINITY;
            for (int s = 0; s < n; s++) m = fmaxf(m, acc[n + q * n + s] * rs[s] * scale);
            float sum = 0.0f;
            for (int s = 0; s < n; s++)
            {
                const float p = expf(acc[n + q * n + s] * rs[s] * scale - m);
                probs[q * n + s] = p;
                sum += p;
            }
            for (int s = 0; s < n; s++) probs[q * n + s] = probs[q * n + s] / sum;
        }
        __syncthreads();
        for (int e = threadIdx.x; e < h; e += blockDim.x)
        {
            float v[kMaxStates];
            for (int s = 0; s < n; s++) v[s] = __fmul_rn(held[s * h + e], rs[s]);
            for (int q = 0; q < queries; q++)
            {
                float o = 0.0f;
                for (int s = 0; s < n; s++) o += v[s] * probs[q * n + s];
                out[(row * queries + q) * h + e] = o;
            }
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

    // One block an output r of a row: the block's dot of the weight row with x and m, and both mean squares (every block
    // recomputes them; x and m are read from L2), then g = (w_x . x) / rms(x) + (w_m . m) / rms(m).
    __global__ void inkling_control_down_wide(const float* __restrict__ x, const float* __restrict__ m,
                                              const float* __restrict__ down, float* __restrict__ g, int h, int rank, float eps)
    {
        __shared__ float scratch[32 * 4];
        const int r = blockIdx.x;
        const long long row = blockIdx.y;
        const float* xr = x + row * h;
        const float* mr = m + row * h;
        const float* wr = down + (long long)r * 2 * h;
        float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int e = threadIdx.x; e < h; e += blockDim.x)
        {
            const float xv = xr[e], mv = mr[e];
            acc[0] += xv * xv;
            acc[1] += mv * mv;
            acc[2] += wr[e] * xv;
            acc[3] += wr[h + e] * mv;
        }
        block_sums(acc, 4, scratch);
        if (threadIdx.x == 0)
            g[row * rank + r] = acc[2] / sqrtf(acc[0] / h + eps) + acc[3] / sqrtf(acc[1] / h + eps);
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

    // ---- the loop controller of a prompt chunk. inkling_control_down_split: 16 rows x the 64 outputs over 256 columns a
    // block, its partial dots (x and m apart) and partial squares to part [split][row][2 * 64 + 2]; inkling_control_blend_rows
    // sums the splits into g, then 16 rows x 64 channels a block.
    constexpr int kCtlRows = 16, kCtlTile = 64, kCtlRank = 64, kCtlRowsFrom = 16, kCtlSplit = 256, kCtlPart = 2 * kCtlRank + 2;

    __global__ void __launch_bounds__(256) inkling_control_down_split(const float* __restrict__ x, const float* __restrict__ m,
        const float* __restrict__ down, float* __restrict__ part, int h, int rows)
    {
        __shared__ float xs[kCtlRows][kCtlTile + 1], ms[kCtlRows][kCtlTile + 1];
        __shared__ float wx[kCtlRank][kCtlTile + 1], wm[kCtlRank][kCtlTile + 1];
        const int tid = threadIdx.x, o = tid & (kCtlRank - 1), rg = tid >> 6, split = blockIdx.y;
        const long long row0 = (long long)blockIdx.x * kCtlRows;
        const int e_end = min(h, (split + 1) * kCtlSplit);
        float accx[4] = { 0.0f, 0.0f, 0.0f, 0.0f }, accm[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float own = 0.0f;   // threads 0 .. 31: the squares of x (0 .. 15) or m (16 .. 31) of one row
        for (int e0 = split * kCtlSplit; e0 < e_end; e0 += kCtlTile)
        {
            for (int i = tid; i < kCtlRows * kCtlTile; i += blockDim.x)
            {
                const int r = i / kCtlTile, k = i - r * kCtlTile;
                const long long row = row0 + r;
                const bool in = row < rows && e0 + k < e_end;
                xs[r][k] = in ? x[row * h + e0 + k] : 0.0f;
                ms[r][k] = in ? m[row * h + e0 + k] : 0.0f;
            }
            for (int i = tid; i < kCtlRank * kCtlTile; i += blockDim.x)
            {
                const int r = i / kCtlTile, k = i - r * kCtlTile;
                const bool in = e0 + k < e_end;
                wx[r][k] = in ? down[(long long)r * 2 * h + e0 + k] : 0.0f;
                wm[r][k] = in ? down[(long long)r * 2 * h + h + e0 + k] : 0.0f;
            }
            __syncthreads();
            if (tid < 2 * kCtlRows)
            {
                const float* v = tid < kCtlRows ? xs[tid] : ms[tid - kCtlRows];
                for (int k = 0; k < kCtlTile; k++) own += v[k] * v[k];
            }
#pragma unroll 8
            for (int k = 0; k < kCtlTile; k++)
            {
                const float wa = wx[o][k], wb = wm[o][k];
#pragma unroll
                for (int i = 0; i < 4; i++)
                {
                    accx[i] += wa * xs[rg * 4 + i][k];
                    accm[i] += wb * ms[rg * 4 + i][k];
                }
            }
            __syncthreads();
        }
        for (int i = 0; i < 4; i++)
        {
            const long long row = row0 + rg * 4 + i;
            if (row >= rows) continue;
            float* pr = part + ((long long)split * rows + row) * kCtlPart;
            pr[o] = accx[i];
            pr[kCtlRank + o] = accm[i];
        }
        if (tid < 2 * kCtlRows)
        {
            const long long row = row0 + (tid % kCtlRows);
            if (row < rows) part[((long long)split * rows + row) * kCtlPart + 2 * kCtlRank + tid / kCtlRows] = own;
        }
    }

    // g [rows][64] from the splits' partials: each row's mean squares, then its 64 outputs.
    __global__ void inkling_control_reduce(const float* __restrict__ part, int splits, float* __restrict__ g, int h, int rows,
                                           float eps)
    {
        const long long row = blockIdx.x;
        const int o = threadIdx.x;
        float sqx = 0.0f, sqm = 0.0f, ax = 0.0f, am = 0.0f;
        for (int s = 0; s < splits; s++)
        {
            const float* pr = part + ((long long)s * rows + row) * kCtlPart;
            sqx += pr[2 * kCtlRank];
            sqm += pr[2 * kCtlRank + 1];
            ax += pr[o];
            am += pr[kCtlRank + o];
        }
        g[row * kCtlRank + o] = ax / sqrtf(sqx / h + eps) + am / sqrtf(sqm / h + eps);
    }

    constexpr int kBlendRows = 32;

    __global__ void __launch_bounds__(256) inkling_control_blend_rows(const float* __restrict__ x, const float* __restrict__ m,
        const float* __restrict__ gain, const float* __restrict__ forget, const float* __restrict__ g, float* __restrict__ y,
        int h, int rows)
    {
        __shared__ float gs[kBlendRows][kCtlRank + 1];
        __shared__ float gw[kCtlTile][kCtlRank + 1], fw[kCtlTile][kCtlRank + 1];
        const int tid = threadIdx.x, cl = tid & (kCtlTile - 1), rg = tid >> 6;
        const int c0 = blockIdx.x * kCtlTile;
        const long long row0 = (long long)blockIdx.y * kBlendRows;
        for (int i = tid; i < kBlendRows * kCtlRank; i += blockDim.x)
        {
            const int r = i / kCtlRank, k = i - r * kCtlRank;
            gs[r][k] = row0 + r < rows ? g[(row0 + r) * kCtlRank + k] : 0.0f;
        }
        for (int i = tid; i < kCtlTile * kCtlRank; i += blockDim.x)
        {
            const int c = i / kCtlRank, k = i - c * kCtlRank;
            const bool in = c0 + c < h;
            gw[c][k] = in ? gain[(long long)(c0 + c) * kCtlRank + k] : 0.0f;
            fw[c][k] = in ? forget[(long long)(c0 + c) * kCtlRank + k] : 0.0f;
        }
        __syncthreads();
        const int c = c0 + cl;
        if (c >= h) return;
        constexpr int per = kBlendRows / 4;
        for (int i = 0; i < per; i++)
        {
            const int r = rg * per + i;
            const long long row = row0 + r;
            if (row >= rows) continue;
            float ga = 0.0f, fo = 0.0f;
#pragma unroll 8
            for (int k = 0; k < kCtlRank; k++)
            {
                ga += gw[cl][k] * gs[r][k];
                fo += fw[cl][k] * gs[r][k];
            }
            const float xv = x[row * h + c], mv = m[row * h + c];
            y[row * h + c] = __fadd_rn(__fsub_rn(xv, __fmul_rn(tanhf(fo), xv)), __fmul_rn(tanhf(ga), mv));
        }
    }


    // ---- the same on tensor cores (TF32 WMMA, fp32 sums). inkling_control_down_tc: a block 32 rows x the 64 outputs over a
    // kCtlSplit-column split, x and m staged in 64-column chunks (zero past the rows), the weights read from global; writes
    // inkling_control_down_split's partials. inkling_control_blend_tc: a block 64 rows x 64 channels, g staged (zero past the
    // rows), the gain and forget weights from global, then the tanh blend.
    constexpr int kTcRows = 32, kTcChunk = 64;

    // A loaded F32 fragment as its TF32 high part (in place) and the TF32 remainder (lo).
    template <typename Frag>
    __device__ __forceinline__ void tf32_split(Frag& hi, Frag& lo)
    {
        for (int t = 0; t < hi.num_elements; t++)
        {
            const float v = hi.x[t];
            const float h = nvcuda::wmma::__float_to_tf32(v);
            hi.x[t] = h;
            lo.x[t] = nvcuda::wmma::__float_to_tf32(v - h);
        }
    }

    __global__ void __launch_bounds__(128) inkling_control_down_tc(const float* __restrict__ x, const float* __restrict__ m,
        const float* __restrict__ down, float* __restrict__ part, int h, int rows)
    {
        using namespace nvcuda;
        __shared__ __align__(32) float xs[kTcRows][kTcChunk + 8];
        __shared__ __align__(32) float ms[kTcRows][kTcChunk + 8];
        __shared__ __align__(32) float outs[2][kTcRows][kCtlRank + 8];
        __shared__ float sq[2][kTcRows];
        const int tid = threadIdx.x, warp = tid >> 5, split = blockIdx.y;
        const long long row0 = (long long)blockIdx.x * kTcRows;
        const int e_begin = split * kCtlSplit, e_end = min(h, e_begin + kCtlSplit);
        const int mt = warp & 1, n0 = (warp >> 1) * 2;     // the warp's m tile and its two output tiles
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> ax[2], am[2];
        for (int t = 0; t < 2; t++) { wmma::fill_fragment(ax[t], 0.0f); wmma::fill_fragment(am[t], 0.0f); }
        if (tid < 2 * kTcRows) sq[tid / kTcRows][tid % kTcRows] = 0.0f;
        float own = 0.0f;   // threads 0 .. 63: the squares of x (0 .. 31) or m (32 .. 63) of one row
        for (int c0 = e_begin; c0 < e_end; c0 += kTcChunk)
        {
            for (int i = tid; i < kTcRows * kTcChunk; i += blockDim.x)
            {
                const int r = i / kTcChunk, k = i - r * kTcChunk;
                const long long row = row0 + r;
                const bool in = row < rows && c0 + k < e_end;
                xs[r][k] = in ? x[row * h + c0 + k] : 0.0f;
                ms[r][k] = in ? m[row * h + c0 + k] : 0.0f;
            }
            __syncthreads();
            if (tid < 2 * kTcRows)
            {
                const float* v = tid < kTcRows ? xs[tid] : ms[tid - kTcRows];
                for (int k = 0; k < kTcChunk; k++) own += v[k] * v[k];
            }
            const int width = min(kTcChunk, e_end - c0);
            for (int k = 0; k < width; k += 8)
            {
                // 3xTF32: each operand as a TF32 high part and the TF32 remainder, three products (near F32).
                wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fx, fm, fxl, fml;
                wmma::load_matrix_sync(fx, &xs[mt * 16][k], kTcChunk + 8);
                wmma::load_matrix_sync(fm, &ms[mt * 16][k], kTcChunk + 8);
                tf32_split(fx, fxl);
                tf32_split(fm, fml);
                for (int t = 0; t < 2; t++)
                {
                    // B[k][n] = W[n][k]: the weights' rows (outputs) as columns, 2 h apart.
                    const float* w = down + (long long)(n0 + t) * 16 * 2 * h + c0 + k;
                    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> bx, bm, bxl, bml;
                    wmma::load_matrix_sync(bx, w, 2 * h);
                    wmma::load_matrix_sync(bm, w + h, 2 * h);
                    tf32_split(bx, bxl);
                    tf32_split(bm, bml);
                    wmma::mma_sync(ax[t], fxl, bx, ax[t]);
                    wmma::mma_sync(ax[t], fx, bxl, ax[t]);
                    wmma::mma_sync(ax[t], fx, bx, ax[t]);
                    wmma::mma_sync(am[t], fml, bm, am[t]);
                    wmma::mma_sync(am[t], fm, bml, am[t]);
                    wmma::mma_sync(am[t], fm, bm, am[t]);
                }
            }
            __syncthreads();
        }
        for (int t = 0; t < 2; t++)
        {
            wmma::store_matrix_sync(&outs[0][mt * 16][(n0 + t) * 16], ax[t], kCtlRank + 8, wmma::mem_row_major);
            wmma::store_matrix_sync(&outs[1][mt * 16][(n0 + t) * 16], am[t], kCtlRank + 8, wmma::mem_row_major);
        }
        if (tid < 2 * kTcRows) sq[tid / kTcRows][tid % kTcRows] = own;
        __syncthreads();
        for (int i = tid; i < kTcRows * kCtlRank; i += blockDim.x)
        {
            const int r = i / kCtlRank, o = i - r * kCtlRank;
            const long long row = row0 + r;
            if (row >= rows) continue;
            float* pr = part + ((long long)split * rows + row) * kCtlPart;
            pr[o] = outs[0][r][o];
            pr[kCtlRank + o] = outs[1][r][o];
            if (o < 2) pr[2 * kCtlRank + o] = sq[o][r];
        }
    }

    constexpr int kTcBlendRows = 32, kTcBlendCh = 64;

    __global__ void __launch_bounds__(128) inkling_control_blend_tc(const float* __restrict__ x, const float* __restrict__ m,
        const float* __restrict__ gain, const float* __restrict__ forget, const float* __restrict__ g, float* __restrict__ y,
        int h, int rows)
    {
        using namespace nvcuda;
        __shared__ __align__(32) float gs[kTcBlendRows][kCtlRank + 8];
        __shared__ __align__(32) float ga[kTcBlendRows][kTcBlendCh + 8];
        __shared__ __align__(32) float fo[kTcBlendRows][kTcBlendCh + 8];
        const int tid = threadIdx.x, warp = tid >> 5;
        const int c0 = blockIdx.x * kTcBlendCh;
        const long long row0 = (long long)blockIdx.y * kTcBlendRows;
        for (int i = tid; i < kTcBlendRows * kCtlRank; i += blockDim.x)
        {
            const int r = i / kCtlRank, k = i - r * kCtlRank;
            gs[r][k] = row0 + r < rows ? g[(row0 + r) * kCtlRank + k] : 0.0f;
        }
        __syncthreads();
        // A warp: m tile (warp & 1) x two of the 4 channel tiles, for gain and forget.
        const int mt = warp & 1, n0 = (warp >> 1) * 2;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> aga[2], afo[2];
        for (int t = 0; t < 2; t++) { wmma::fill_fragment(aga[t], 0.0f); wmma::fill_fragment(afo[t], 0.0f); }
        for (int k = 0; k < kCtlRank; k += 8)
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fa, fal;
            wmma::load_matrix_sync(fa, &gs[mt * 16][k], kCtlRank + 8);
            tf32_split(fa, fal);
            for (int t = 0; t < 2; t++)
            {
                // B[k][c] = gain[c][k]: channel rows as columns, rank apart.
                const long long c = c0 + (n0 + t) * 16;
                wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> bg, bf, bgl, bfl;
                wmma::load_matrix_sync(bg, gain + c * kCtlRank + k, kCtlRank);
                wmma::load_matrix_sync(bf, forget + c * kCtlRank + k, kCtlRank);
                tf32_split(bg, bgl);
                tf32_split(bf, bfl);
                wmma::mma_sync(aga[t], fal, bg, aga[t]);
                wmma::mma_sync(aga[t], fa, bgl, aga[t]);
                wmma::mma_sync(aga[t], fa, bg, aga[t]);
                wmma::mma_sync(afo[t], fal, bf, afo[t]);
                wmma::mma_sync(afo[t], fa, bfl, afo[t]);
                wmma::mma_sync(afo[t], fa, bf, afo[t]);
            }
        }
        for (int t = 0; t < 2; t++)
        {
            wmma::store_matrix_sync(&ga[mt * 16][(n0 + t) * 16], aga[t], kTcBlendCh + 8, wmma::mem_row_major);
            wmma::store_matrix_sync(&fo[mt * 16][(n0 + t) * 16], afo[t], kTcBlendCh + 8, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < kTcBlendRows * kTcBlendCh; i += blockDim.x)
        {
            const int r = i / kTcBlendCh, cl = i - r * kTcBlendCh;
            const long long row = row0 + r;
            const int c = c0 + cl;
            if (row >= rows || c >= h) continue;
            const float xv = x[row * h + c], mv = m[row * h + c];
            y[row * h + c] = __fadd_rn(__fsub_rn(xv, __fmul_rn(tanhf(fo[r][cl]), xv)), __fmul_rn(tanhf(ga[r][cl]), mv));
        }
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
    __global__ void inkling_situ(const float* __restrict__ gu, const float* __restrict__ scale, float* __restrict__ out, long long ff,
                                 long long vectors, float beta, float lb)
    {
        const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= ff * vectors) return;
        const long long vec = i / ff, e = i - vec * ff;
        float g = gu[vec * 2 * ff + e], u = gu[vec * 2 * ff + ff + e];
        if (scale != nullptr)
        {
            g = __fmul_rn(g, scale[vec]);
            u = __fmul_rn(u, scale[vec]);
        }
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

#include "tsg_cuda_inkling_moe.cuh"
#include "tsg_cuda_inkling_fp4.cuh"

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
        const void* T;               // [extent + 1, heads, rows]: B r by distance, the last entry 0 (F32, or F16: t_half)
        int t_half;
        const float* sink;           // [heads, rows]: the sink logit t each score's bias takes away
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

    // A row and head's bias by distance, read as F32 from an F32 or F16 table.
    struct BiasRow
    {
        const void* p;
        int half;
        __device__ __forceinline__ float operator[](long long i) const
        {
            return half ? __half2float(reinterpret_cast<const __half*>(p)[i]) : reinterpret_cast<const float*>(p)[i];
        }
    };

    // Row i of head h's bias by distance (the table's last entry past the extent).
    __device__ __forceinline__ BiasRow bias_row(const AttnArgs& a, int i, int h)
    {
        const long long off = ((long long)i * a.heads + h) * (a.extent + 1);
        return { a.t_half ? (const void*)(reinterpret_cast<const __half*>(a.T) + off) : (const void*)(reinterpret_cast<const float*>(a.T) + off),
                 a.t_half };
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
        // The rows of a split run side by side (blockIdx.x), so a verify window reads each K/V tile from DRAM once.
        const int i = blockIdx.x, g = blockIdx.y, split = blockIdx.z;
        const int k0 = split * kSplitKeys, k1 = min(a.keys, k0 + kSplitKeys);
        const long long p = row_position(a, i), last = row_position(a, a.rows - 1);
        float* out = part + (((long long)i * a.kv_heads + g) * splits + split) * G * (2 + DV);
        {
            // A split the row sees no key of (a linear window's keys outside its window, a ring's cells holding positions before
            // it): an empty partial, its max alone (the combine reads no more of it).
            static_assert(kSplitThreads == kSplitKeys, "a thread a ring cell");
            bool seen;
            if (a.ring == 0)
            {
                const long long lo = (long long)a.first_key + k0, hi = (long long)a.first_key + k1 - 1;
                seen = !(k0 >= k1 || hi < lowest_seen(a, p) || lo > p);
            }
            else
            {
                const int j = k0 + (int)threadIdx.x;
                const long long kp = j < k1 ? key_position(a, j, last) : -1;
                seen = __syncthreads_or(kp >= 0 && kp >= lowest_seen(a, p) && kp <= p) != 0;
            }
            if (!seen)
            {
                for (int hh = threadIdx.x; hh < G; hh += blockDim.x) out[hh * (2 + DV)] = -INFINITY;
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
            for (int hh = 0; hh < G; hh++) sc[hh][jj] = dot[hh] * a.scale + bias_row(a, i, g * G + hh)[bi] - a.sink[(long long)i * a.heads + g * G + hh];
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


    // ---- a verify window's attention on tensor cores (2 .. 16 rows, F16 caches, 128-wide heads): one block a KV head and
    // kVmmaKeys keys for every row and query head of the group (m = row * G + head, padded to MT tiles of 16), S = Q K^T
    // and O = P V in WMMA (fp16 operands, fp32 sums) with the K and V fragments loaded from the cache itself, so a K or V row
    // is read once a step. The partials are inkling_attention_split's, over kVmmaKeys-key splits.
    constexpr int kVmmaKeys = 128, kVmmaThreads = 128;

    template <int MT>
    __global__ void __launch_bounds__(kVmmaThreads) inkling_attention_verify_mma(AttnArgs a, float* __restrict__ part, int splits)
    {
        using namespace nvcuda;
        constexpr int D = 128, M = MT * 16, SK = kVmmaKeys;
        extern __shared__ __align__(32) unsigned char vm_sm[];
        half* qs = reinterpret_cast<half*>(vm_sm);                                  // [M][D] queries, then [M][SK] probabilities
        float* ss = reinterpret_cast<float*>(vm_sm + M * D * 2);                    // [M][SK] scores, then [M][D] outputs
        half* ps = qs;                                                              // (SK == D: Q is dead once S is stored)
        static_assert(SK == D, "the probabilities take the queries' place");
        __shared__ float rmax[M], rsum[M];
        const int g = blockIdx.x, split = blockIdx.y;
        const int G = a.heads / a.kv_heads, R = a.rows, rowsM = R * G;
        const int k0 = split * SK;
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        const long long last = row_position(a, R - 1);
        {
            // A split no row sees a key of (a linear window's keys outside the rows' windows; a ring's cells holding positions
            // before them, most of a ring sized for long prompt steps): an empty partial, its max alone (the combine reads no
            // more of a split whose max is -inf).
            static_assert(kVmmaThreads == SK, "a thread a ring cell");
            const long long seen_lo = lowest_seen(a, row_position(a, 0));
            bool seen;
            if (a.ring == 0)
            {
                const long long lo = (long long)a.first_key + k0, hi = (long long)a.first_key + min(a.keys, k0 + SK) - 1;
                seen = !(lo > last || hi < seen_lo);
            }
            else
            {
                const int j = k0 + (int)threadIdx.x;
                const long long kp = j < a.keys ? key_position(a, j, last) : -1;
                seen = __syncthreads_or(kp >= 0 && kp >= seen_lo && kp <= last) != 0;
            }
            if (!seen)
            {
                for (int m = threadIdx.x; m < rowsM; m += blockDim.x)
                {
                    const int i = m / G, hh = m - i * G;
                    part[(((long long)i * a.kv_heads + g) * splits + split) * G * (2 + D) + hh * (2 + D)] = -INFINITY;
                }
                return;
            }
        }
        for (int idx = threadIdx.x; idx < M * D; idx += blockDim.x)
        {
            const int m = idx / D, e = idx - m * D;
            float v = 0.0f;
            if (m < rowsM)
            {
                const int i = m / G, hh = m - i * G;
                v = reinterpret_cast<const float*>(a.q + (long long)i * a.q_nb1 + (long long)(g * G + hh) * a.q_nb2)[e];
            }
            qs[idx] = __float2half(v);
        }
        __syncthreads();

        // S = Q K^T: a warp the key tiles warp, warp + 4 (of SK / 16), every m tile.
        const half* kbase = reinterpret_cast<const half*>(a.k + (long long)g * a.k_nb2);
        const int ldk = (int)(a.k_nb1 / 2);
        for (int n = warp; n < SK / 16; n += kVmmaThreads / 32)
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT];
#pragma unroll
            for (int mt = 0; mt < MT; mt++) wmma::fill_fragment(acc[mt], 0.0f);
            const int j0 = k0 + n * 16;
            if (j0 < a.keys)
            {
#pragma unroll
                for (int kk = 0; kk < D / 16; kk++)
                {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b;
                    wmma::load_matrix_sync(b, kbase + (long long)j0 * ldk + kk * 16, ldk);
#pragma unroll
                    for (int mt = 0; mt < MT; mt++)
                    {
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
                        wmma::load_matrix_sync(af, qs + mt * 16 * D + kk * 16, D);
                        wmma::mma_sync(acc[mt], af, b, acc[mt]);
                    }
                }
            }
#pragma unroll
            for (int mt = 0; mt < MT; mt++) wmma::store_matrix_sync(ss + mt * 16 * SK + n * 16, acc[mt], SK, wmma::mem_row_major);
        }
        __syncthreads();

        // Each (row, head)'s softmax over the split: a warp a row, a lane every 32nd key; the logit q.k scale + bias - sink.
        for (int m = warp; m < M; m += kVmmaThreads / 32)
        {
            const bool real = m < rowsM;
            const int i = real ? m / G : 0, hh = real ? m - i * G : 0;
            const long long p = real ? row_position(a, i) : 0;
            const BiasRow brow = real ? bias_row(a, i, g * G + hh) : BiasRow{ nullptr, 0 };
            const float sink = real ? a.sink[(long long)i * a.heads + g * G + hh] : 0.0f;
            float vals[SK / 32];
            float mx = -INFINITY;
#pragma unroll
            for (int t = 0; t < SK / 32; t++)
            {
                const int jj = lane + 32 * t, j = k0 + jj;
                float sv = -INFINITY;
                if (real && j < a.keys)
                {
                    const long long kp = key_position(a, j, last);
                    if (sees(a, p, kp))
                    {
                        const long long dist = p - kp;
                        const int bi = dist < a.extent ? (int)dist : a.extent;
                        sv = ss[m * SK + jj] * a.scale + brow[bi] - sink;
                    }
                }
                vals[t] = sv;
                mx = fmaxf(mx, sv);
            }
            mx = warp_max(mx);
            float sum = 0.0f;
#pragma unroll
            for (int t = 0; t < SK / 32; t++)
            {
                const float e = vals[t] == -INFINITY ? 0.0f : expf(vals[t] - mx);
                sum += e;
                ps[m * SK + lane + 32 * t] = __float2half(e);
            }
            sum = warp_sum(sum);
            if (lane == 0) { rmax[m] = mx; rsum[m] = sum; }
        }
        __syncthreads();

        // O = P V: a warp the value tiles 2 warp, 2 warp + 1 (of D / 16), every m tile, over the split's keys.
        const half* vbase = reinterpret_cast<const half*>(a.v + (long long)g * a.v_nb2);
        const int ldv = (int)(a.v_nb1 / 2);
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> oacc[MT][2];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int t = 0; t < 2; t++) wmma::fill_fragment(oacc[mt][t], 0.0f);
        for (int kk = 0; kk < SK / 16; kk++)
        {
            const int j0 = k0 + kk * 16;
            if (j0 >= a.keys) break;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf[MT];
#pragma unroll
            for (int mt = 0; mt < MT; mt++) wmma::load_matrix_sync(pf[mt], ps + mt * 16 * SK + kk * 16, SK);
#pragma unroll
            for (int t = 0; t < 2; t++)
            {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
                wmma::load_matrix_sync(vf, vbase + (long long)j0 * ldv + (warp * 2 + t) * 16, ldv);
#pragma unroll
                for (int mt = 0; mt < MT; mt++) wmma::mma_sync(oacc[mt][t], pf[mt], vf, oacc[mt][t]);
            }
        }
        // The outputs through the scores' region (SK == D), then each (row, head)'s partial.
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int t = 0; t < 2; t++) wmma::store_matrix_sync(ss + mt * 16 * D + (warp * 2 + t) * 16, oacc[mt][t], D, wmma::mem_row_major);
        __syncthreads();
        for (int idx = threadIdx.x; idx < rowsM * (2 + D); idx += blockDim.x)
        {
            const int m = idx / (2 + D), e = idx - m * (2 + D), i = m / G, hh = m - i * G;
            part[(((long long)i * a.kv_heads + g) * splits + split) * G * (2 + D) + hh * (2 + D) + e] =
                e == 0 ? rmax[m] : e == 1 ? rsum[m] : ss[m * D + e - 2];
        }
    }

    // The verify window's kernel where it takes the shapes (F16 K/V aligned for WMMA, 128-wide heads, G R <= 48 rows, at
    // least TS_INKLING_VMMA_MIN rows, default 2; TS_INKLING_VMMA=0 never); the splits it ran, or 0.
    int launch_verify_mma(const AttnArgs& a, const ggml_tensor* k, float* part, cudaStream_t stream)
    {
        static const bool off = []() { const char* e = std::getenv("TS_INKLING_VMMA"); return e != nullptr && e[0] == '0'; }();
        static const int fewest = []() { const char* e = std::getenv("TS_INKLING_VMMA_MIN"); return e != nullptr ? std::atoi(e) : 2; }();
        if (off || k->type != GGML_TYPE_F16 || a.d != 128 || a.dv != 128 || a.rows < (fewest > 2 ? fewest : 2) || a.keys % 16 != 0) return 0;
        if (a.kv_heads < 1 || a.heads % a.kv_heads != 0) return 0;
        const int G = a.heads / a.kv_heads, M = G * a.rows;
        if (M > 48) return 0;
        if ((uintptr_t)a.k % 32 != 0 || (uintptr_t)a.v % 32 != 0 || a.k_nb1 % 32 != 0 || a.k_nb2 % 32 != 0 || a.v_nb1 % 32 != 0 || a.v_nb2 % 32 != 0)
            return 0;
        const int splits = (a.keys + kVmmaKeys - 1) / kVmmaKeys;
        const dim3 grid((unsigned int)a.kv_heads, (unsigned int)splits);
        // Shared memory: M x (D halves + SK floats); the probabilities reuse the queries' place.
        if (M <= 16) inkling_attention_verify_mma<1><<<grid, kVmmaThreads, 16 * 768, stream>>>(a, part, splits);
        else if (M <= 32) inkling_attention_verify_mma<2><<<grid, kVmmaThreads, 32 * 768, stream>>>(a, part, splits);
        else inkling_attention_verify_mma<3><<<grid, kVmmaThreads, 48 * 768, stream>>>(a, part, splits);
        return splits;
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
    constexpr size_t kMmaSmemBytes = sizeof(half) * 2 * kMmaKeys * kMmaRow + sizeof(int) * kMmaKeys;

    template <typename KV>
    __global__ __launch_bounds__(kMmaWarps * 32) void inkling_attention_mma(AttnArgs a)
    {
        constexpr int W = kMmaWidth, ROW = kMmaRow, KEYS = kMmaKeys, THREADS = kMmaWarps * 32;
        extern __shared__ __align__(16) unsigned char smem_raw[];
        half* keys = reinterpret_cast<half*>(smem_raw);
        half* values = keys + KEYS * ROW;
        int* key_pos = reinterpret_cast<int*>(values + KEYS * ROW);   // the tile's key positions (-1: no key)
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
        const int p0 = r0 < a.rows ? (int)row_position(a, r0) : -1, p1 = r1 < a.rows ? (int)row_position(a, r1) : -1;
        const int floor_pos = a.floor, band = a.band, extent = a.extent;
        const BiasRow T0 = r0 < a.rows ? bias_row(a, r0, head) : BiasRow{ nullptr, 0 };
        const BiasRow T1 = r1 < a.rows ? bias_row(a, r1, head) : BiasRow{ nullptr, 0 };
        const float t0 = r0 < a.rows ? a.sink[(long long)r0 * a.heads + head] : 0.0f;
        const float t1 = r1 < a.rows ? a.sink[(long long)r1 * a.heads + head] : 0.0f;
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
            for (int r = tid; r < KEYS; r += THREADS)
                key_pos[r] = k0 + r < a.keys ? (int)key_position(a, k0 + r, last) : -1;
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
                    const int kp = key_pos[j * 8 + quad * 2 + (c & 1)];
                    const int p = c < 2 ? p0 : p1;
                    float val = -CUDART_INF_F;
                    if (p >= 0 && kp >= 0 && kp >= floor_pos && kp <= p && (band == 0 || p - kp < band))
                    {
                        const int dist = p - kp;
                        const BiasRow T = c < 2 ? T0 : T1;
                        val = (s[j][c] * a.scale + T[dist < extent ? dist : extent] - (c < 2 ? t0 : t1)) * LOG2E;
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


    // ---- v2: 64 rows x the G query heads of one KV head a block (4 warps a head), key positions walked
    // Splits a prompt chunk's attention takes over its keys: about four waves of blocks, each split 2,048 keys at least.
    __host__ __device__ inline int mma2_splits(long long rows, long long kv_heads, long long keys, int cap)
    {
        if (cap == 1 || keys < 4096 || rows <= 16) return 1;
        const long long blocks = (rows + 63) / 64 * kv_heads;
        long long sp = (384 + blocks - 1) / blocks;
        if (sp > keys / 2048) sp = keys / 2048;
        if (sp > 16) sp = 16;
        if (cap > 1 && sp > cap) sp = cap;
        return sp < 1 ? 1 : (int)sp;
    }

    template <typename KV, int G>
    __global__ __launch_bounds__(G * 128) void inkling_attention_mma2(AttnArgs a, float* __restrict__ part, int splits)
    {
        constexpr int W = kMmaWidth, ROW = kMmaRow, KEYS = kMmaKeys, THREADS = G * 128;
        extern __shared__ __align__(16) unsigned char smem_raw[];
        half* keys = reinterpret_cast<half*>(smem_raw);
        half* values = keys + KEYS * ROW;
        const int kvh = blockIdx.y, tid = threadIdx.x;
        const int warp = tid >> 5, lane = tid & 31, group = lane >> 2, quad = lane & 3;
        const int head = kvh * G + (warp >> 2);
        const int first_row = blockIdx.x * 64, wrow = first_row + (warp & 3) * 16;
        const int r0 = wrow + group, r1 = r0 + 8;
        const bool v0 = r0 < a.rows, v1 = r1 < a.rows;

        uint32_t qa[W / 16][4];
        {
            const float* q0 = v0 ? (const float*)(a.q + (long long)r0 * a.q_nb1 + (long long)head * a.q_nb2) : nullptr;
            const float* q1 = v1 ? (const float*)(a.q + (long long)r1 * a.q_nb1 + (long long)head * a.q_nb2) : nullptr;
#pragma unroll
            for (int kk = 0; kk < W / 16; ++kk)
            {
                const int c = kk * 16 + quad * 2;
                qa[kk][0] = pack_half2(v0 ? q0[c] : 0.0f, v0 ? q0[c + 1] : 0.0f);
                qa[kk][1] = pack_half2(v1 ? q1[c] : 0.0f, v1 ? q1[c + 1] : 0.0f);
                qa[kk][2] = pack_half2(v0 ? q0[c + 8] : 0.0f, v0 ? q0[c + 9] : 0.0f);
                qa[kk][3] = pack_half2(v1 ? q1[c + 8] : 0.0f, v1 ? q1[c + 9] : 0.0f);
            }
        }
        const int p0 = v0 ? (int)row_position(a, r0) : -1, p1 = v1 ? (int)row_position(a, r1) : -1;
        const BiasRow T0 = v0 ? bias_row(a, r0, head) : BiasRow{ nullptr, 0 };
        const BiasRow T1 = v1 ? bias_row(a, r1, head) : BiasRow{ nullptr, 0 };
        const float t0 = v0 ? a.sink[(long long)r0 * a.heads + head] : 0.0f;
        const float t1 = v1 ? a.sink[(long long)r1 * a.heads + head] : 0.0f;
        const int floor_pos = a.floor, band = a.band, extent = a.extent;
        const int last = (int)row_position(a, a.rows - 1);
        auto lowest = [&](int p) { int lo = floor_pos > 0 ? floor_pos : 0; if (band > 0 && p - band + 1 > lo) lo = p - band + 1; return lo; };

        // The block's key positions: what its rows see, within the cache's live keys.
        const int pb_lo = (int)row_position(a, first_row), pb_hi = (int)row_position(a, min(a.rows, first_row + 64) - 1);
        int lo = lowest(pb_lo), hi = pb_hi;
        if (a.ring > 0) lo = max(lo, last - a.ring + 1);
        else { lo = max(lo, a.first_key); hi = min(hi, a.first_key + a.keys - 1); }
        if (part != nullptr && hi >= lo)
        {
            // This split's share of the range, whole tiles.
            const int tiles = (hi - lo + KEYS) / KEYS, per = (tiles + splits - 1) / splits;
            const int slo = lo + blockIdx.z * per * KEYS;
            hi = min(hi, slo + per * KEYS - 1);
            lo = slo;
        }
        // The warp's rows.
        const bool warp_rows = wrow < a.rows;
        const int pw_lo = warp_rows ? (int)row_position(a, wrow) : 0;
        const int pw_hi = warp_rows ? (int)row_position(a, min(a.rows - 1, wrow + 15)) : -1;
        const int seen_all = lowest(pw_hi), seen_any = lowest(pw_lo);
        constexpr float LOG2E = 1.4426950408889634f;

        float acc[W / 8][4];
#pragma unroll
        for (int n = 0; n < W / 8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
        // The zero sink starts a whole row's softmax; a split's partial starts empty (the combine adds the sink once).
        float max0 = 0.0f, max1 = 0.0f, sum0 = quad == 0 ? 1.0f : 0.0f, sum1 = quad == 0 ? 1.0f : 0.0f;
        if (part != nullptr) { max0 = max1 = -1e30f; sum0 = sum1 = 0.0f; }

        for (int k0 = lo; k0 <= hi; k0 += KEYS)
        {
            __syncthreads();
            if (sizeof(KV) == 2 && a.kv_aligned)
            {
                for (int i = tid; i < KEYS * (W / 8); i += THREADS)
                {
                    const int r = i / (W / 8), c = i - r * (W / 8), pos = k0 + r;
                    half* kd = keys + r * ROW + c * 8;
                    half* vd = values + r * ROW + c * 8;
                    if (pos <= hi)
                    {
                        const int cell = a.ring > 0 ? pos % a.ring : pos - a.first_key;
                        cp_async_16(kd, a.k + (long long)cell * a.k_nb1 + (long long)kvh * a.k_nb2 + c * 16);
                        cp_async_16(vd, a.v + (long long)cell * a.v_nb1 + (long long)kvh * a.v_nb2 + c * 16);
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
                    const int r = i / W, x = i - r * W, pos = k0 + r;
                    float kx = 0.0f, vx = 0.0f;
                    if (pos <= hi)
                    {
                        const int cell = a.ring > 0 ? pos % a.ring : pos - a.first_key;
                        kx = kv_load<KV>((const KV*)(a.k + (long long)cell * a.k_nb1 + (long long)kvh * a.k_nb2) + x);
                        vx = kv_load<KV>((const KV*)(a.v + (long long)cell * a.v_nb1 + (long long)kvh * a.v_nb2) + x);
                    }
                    keys[r * ROW + x] = __float2half(kx);
                    values[r * ROW + x] = __float2half(vx);
                }
            }
            __syncthreads();
            const int kt_hi = min(hi, k0 + KEYS - 1);
            // None of the warp's rows sees the tile: no work (the block's loop and barriers go on).
            if (!warp_rows || k0 > pw_hi || kt_hi < seen_any) continue;
            const bool full = k0 >= seen_all && kt_hi <= pw_lo && kt_hi == k0 + KEYS - 1;
            const bool far = full && pw_lo - kt_hi >= extent;

            float sc[KEYS / 8][4];
#pragma unroll
            for (int j = 0; j < KEYS / 8; ++j)
            {
                sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.0f;
#pragma unroll
                for (int kk = 0; kk < W / 16; ++kk)
                {
                    const half* kb = keys + (j * 8 + group) * ROW + kk * 16 + quad * 2;
                    mma_f16_16816(sc[j], qa[kk], load_half2(kb), load_half2(kb + 8));
                }
            }
            float tile0 = max0, tile1 = max1;
#pragma unroll
            for (int j = 0; j < KEYS / 8; ++j)
            {
#pragma unroll
                for (int c = 0; c < 4; ++c)
                {
                    const int kp = k0 + j * 8 + quad * 2 + (c & 1);
                    const int p = c < 2 ? p0 : p1;
                    const float t = c < 2 ? t0 : t1;
                    float val = -CUDART_INF_F;
                    if (far)
                    {
                        if (p >= 0) val = (sc[j][c] * a.scale - t) * LOG2E;
                    }
                    else if (full)
                    {
                        if (p >= 0)
                        {
                            const int dist = p - kp;
                            val = (sc[j][c] * a.scale + (c < 2 ? T0 : T1)[dist < extent ? dist : extent] - t) * LOG2E;
                        }
                    }
                    else if (p >= 0 && kp <= kt_hi && kp >= floor_pos && kp <= p && (band == 0 || p - kp < band))
                    {
                        const int dist = p - kp;
                        val = (sc[j][c] * a.scale + (c < 2 ? T0 : T1)[dist < extent ? dist : extent] - t) * LOG2E;
                    }
                    sc[j][c] = val;
                }
                tile0 = fmaxf(tile0, fmaxf(sc[j][0], sc[j][1]));
                tile1 = fmaxf(tile1, fmaxf(sc[j][2], sc[j][3]));
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
                sc[j][0] = exp2f(sc[j][0] - max0);
                sc[j][1] = exp2f(sc[j][1] - max0);
                sc[j][2] = exp2f(sc[j][2] - max1);
                sc[j][3] = exp2f(sc[j][3] - max1);
                sum0 += sc[j][0] + sc[j][1];
                sum1 += sc[j][2] + sc[j][3];
            }
#pragma unroll
            for (int kk = 0; kk < KEYS / 16; ++kk)
            {
                const uint32_t pa[4] = {
                    pack_half2(sc[2 * kk][0], sc[2 * kk][1]), pack_half2(sc[2 * kk][2], sc[2 * kk][3]),
                    pack_half2(sc[2 * kk + 1][0], sc[2 * kk + 1][1]), pack_half2(sc[2 * kk + 1][2], sc[2 * kk + 1][3])};
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
        if (!warp_rows) return;
#pragma unroll
        for (int offset = 1; offset < 4; offset <<= 1)
        {
            sum0 += __shfl_xor_sync(0xffffffffu, sum0, offset);
            sum1 += __shfl_xor_sync(0xffffffffu, sum1, offset);
        }
        if (part != nullptr)
        {
            // The split's partial in the split kernel's layout (max in natural units, the values unnormalized).
            const int hh = warp >> 2;
#pragma unroll
            for (int half_row = 0; half_row < 2; ++half_row)
            {
                const int row = half_row ? r1 : r0;
                if (row >= a.rows) continue;
                const float mx = half_row ? max1 : max0, sm = half_row ? sum1 : sum0;
                float* o = part + (((long long)row * a.kv_heads + kvh) * splits + blockIdx.z) * G * (2 + a.dv) + hh * (2 + a.dv);
                if (quad == 0)
                {
                    o[0] = sm > 0.0f ? mx / LOG2E : -INFINITY;
                    o[1] = sm;
                }
#pragma unroll
                for (int n = 0; n < W / 8; ++n)
                {
                    const int x = n * 8 + quad * 2;
                    o[2 + x] = acc[n][half_row * 2];
                    o[2 + x + 1] = acc[n][half_row * 2 + 1];
                }
            }
            return;
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

    template <typename KV, int G>
    void launch_mma2_g(const AttnArgs& a, cudaStream_t stream, float* part, int splits)
    {
        static bool attr = false;
        if (!attr)
        {
            CUDA_CHECK(cudaFuncSetAttribute(inkling_attention_mma2<KV, G>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kMmaSmemBytes));
            attr = true;
        }
        const dim3 grid((unsigned int)((a.rows + 63) / 64), (unsigned int)a.kv_heads, (unsigned int)(part != nullptr ? splits : 1));
        inkling_attention_mma2<KV, G><<<grid, G * 128, kMmaSmemBytes, stream>>>(a, part, part != nullptr ? splits : 1);
    }

    // v2 where it takes the shape (up to 4 query heads a KV head, positions that fit 32 bits); false otherwise.
    template <typename KV>
    bool launch_mma2(const AttnArgs& a, cudaStream_t stream, float* part = nullptr, int splits = 1)
    {
        static const bool off = []() { const char* e = std::getenv("TS_INKLING_MMA2"); return e != nullptr && e[0] == '0'; }();
        if (off) return false;
        if (splits <= 1) part = nullptr;
        switch (a.heads / a.kv_heads)
        {
            case 1: launch_mma2_g<KV, 1>(a, stream, part, splits); return true;
            case 2: launch_mma2_g<KV, 2>(a, stream, part, splits); return true;
            case 3: launch_mma2_g<KV, 3>(a, stream, part, splits); return true;
            case 4: launch_mma2_g<KV, 4>(a, stream, part, splits); return true;
            default: return false;
        }
    }

    int mma2_split_cap()
    {
        static const int cap = []() { const char* e = std::getenv("TS_INKLING_MMA2_SPLITS"); return e != nullptr ? std::atoi(e) : 0; }();
        return cap;
    }

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


    // ---- a verify window (2 .. R rows) split over the keys: one block a KV head's G query heads over kSplitKeys keys for
    // every row, a thread a key (its K row read once for all rows and heads), each row-head's softmax over the split, then
    // the values in row groups of two (a warp a run of keys, a lane 4 adjacent elements: a V row read whole). The partials
    // are the per-row kernel's, so inkling_attention_combine finishes them.
    constexpr int kSplitRowsMax = 8;

    template <typename KV, int G>
    __global__ void __launch_bounds__(kSplitThreads) inkling_attention_split_rows(AttnArgs a, float* __restrict__ part, int splits)
    {
        constexpr int D = kSplitWidth, DV = kSplitWidth, R = kSplitRowsMax, P = R * G, WARPS = kSplitThreads / 32;
        __shared__ float qs[P * D];                 // the rows' queries, later the warps' value sums
        __shared__ float sc[P][kSplitKeys];
        __shared__ float red[P][WARPS];
        __shared__ float mx[P], sm[P];
        const int g = blockIdx.x, split = blockIdx.y, rows = a.rows;
        const int k0 = split * kSplitKeys, k1 = min(a.keys, k0 + kSplitKeys);
        const long long last = row_position(a, rows - 1);
        long long p[R];
#pragma unroll
        for (int r = 0; r < R; r++) p[r] = r < rows ? row_position(a, r) : -1;
        auto out_of = [&](int r) { return part + (((long long)r * a.kv_heads + g) * splits + split) * G * (2 + DV); };
        if (a.ring == 0)
        {
            // A linear window's split no row sees a key of: empty partials.
            const long long lo = (long long)a.first_key + k0, hi = (long long)a.first_key + k1 - 1;
            bool any = false;
            for (int r = 0; r < rows; r++) any |= !(k0 >= k1 || hi < lowest_seen(a, p[r]) || lo > p[r]);
            if (!any)
            {
                for (int r = 0; r < rows; r++)
                    for (int o = threadIdx.x; o < G * (2 + DV); o += blockDim.x) out_of(r)[o] = o % (2 + DV) == 0 ? -INFINITY : 0.0f;
                return;
            }
        }
        for (int idx = threadIdx.x; idx < P * D; idx += blockDim.x)
        {
            const int pr = idx / D, e = idx - pr * D, r = pr / G, hh = pr - r * G;
            qs[idx] = r < rows ? ((const float*)(a.q + (long long)r * a.q_nb1 + (long long)(g * G + hh) * a.q_nb2))[e] : 0.0f;
        }
        __syncthreads();
        for (int j = k0 + threadIdx.x; j < k0 + kSplitKeys; j += blockDim.x)
        {
            const int jj = j - k0;
            const long long kp = j < k1 ? key_position(a, j, last) : -1;
            bool seen[R];
            bool any = false;
#pragma unroll
            for (int r = 0; r < R; r++)
            {
                seen[r] = r < rows && kp >= 0 && sees(a, p[r], kp);
                any |= seen[r];
            }
            if (!any)
            {
                for (int pr = 0; pr < P; pr++) sc[pr][jj] = -INFINITY;
                continue;
            }
            const KV* krow = (const KV*)(a.k + (long long)j * a.k_nb1 + (long long)g * a.k_nb2);
            float dot[P];
#pragma unroll
            for (int pr = 0; pr < P; pr++) dot[pr] = 0.0f;
            constexpr int per = 16 / sizeof(KV);
            for (int e0 = 0; e0 < D; e0 += per)
            {
                float kv[per];
                if (a.kv_aligned)
                {
                    const uint4 raw = *reinterpret_cast<const uint4*>(krow + e0);
                    if constexpr (sizeof(KV) == 2)
                    {
                        const half2* h2 = reinterpret_cast<const half2*>(&raw);
#pragma unroll
                        for (int u = 0; u < per / 2; u++) { const float2 f = __half22float2(h2[u]); kv[2 * u] = f.x; kv[2 * u + 1] = f.y; }
                    }
                    else
                    {
                        const float* f = reinterpret_cast<const float*>(&raw);
#pragma unroll
                        for (int u = 0; u < per; u++) kv[u] = f[u];
                    }
                }
                else
                {
#pragma unroll
                    for (int u = 0; u < per; u++) kv[u] = kv_load<KV>(krow + e0 + u);
                }
#pragma unroll
                for (int u = 0; u < per; u++)
#pragma unroll
                    for (int pr = 0; pr < P; pr++) dot[pr] += qs[pr * D + e0 + u] * kv[u];
            }
#pragma unroll
            for (int r = 0; r < R; r++)
            {
                const long long dist = p[r] - kp;
                const int bi = dist < a.extent ? (int)dist : a.extent;
#pragma unroll
                for (int hh = 0; hh < G; hh++)
                    sc[r * G + hh][jj] = seen[r]
                        ? dot[r * G + hh] * a.scale + bias_row(a, r, g * G + hh)[bi] - a.sink[(long long)r * a.heads + g * G + hh]
                        : -INFINITY;
            }
        }
        __syncthreads();
        const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        for (int pr = 0; pr < P; pr++)
        {
            float v = -INFINITY;
            for (int jj = threadIdx.x; jj < kSplitKeys; jj += blockDim.x) v = fmaxf(v, sc[pr][jj]);
            v = warp_max(v);
            if (lane == 0) red[pr][warp] = v;
        }
        __syncthreads();
        if (threadIdx.x < P)
        {
            float v = -INFINITY;
            for (int w = 0; w < WARPS; w++) v = fmaxf(v, red[threadIdx.x][w]);
            mx[threadIdx.x] = v;
        }
        __syncthreads();
        for (int pr = 0; pr < P; pr++)
        {
            float v = 0.0f;
            for (int jj = threadIdx.x; jj < kSplitKeys; jj += blockDim.x)
            {
                const float e = sc[pr][jj] == -INFINITY ? 0.0f : expf(sc[pr][jj] - mx[pr]);
                sc[pr][jj] = e;
                v += e;
            }
            v = warp_sum(v);
            if (lane == 0) red[pr][warp] = v;
        }
        __syncthreads();
        if (threadIdx.x < P)
        {
            float v = 0.0f;
            for (int w = 0; w < WARPS; w++) v += red[threadIdx.x][w];
            sm[threadIdx.x] = v;
        }
        __syncthreads();
        // The values, two rows at a time: each warp a run of keys, each lane 4 elements of the V row for both rows' heads.
        constexpr int per = DV / 32;
        const int run = (kSplitKeys + WARPS - 1) / WARPS;
        float* sums = qs;   // [P][DV], the warps' partial sums (the queries are no longer read)
        for (int idx = threadIdx.x; idx < P * DV; idx += blockDim.x) sums[idx] = 0.0f;
        __syncthreads();
        for (int r0 = 0; r0 < rows; r0 += 2)
        {
            float vacc[2][G][per];
#pragma unroll
            for (int rr = 0; rr < 2; rr++)
#pragma unroll
                for (int hh = 0; hh < G; hh++)
#pragma unroll
                    for (int u = 0; u < per; u++) vacc[rr][hh][u] = 0.0f;
            const int jlo = warp * run, jhi = min(kSplitKeys, jlo + run);
            for (int jj = jlo; jj < jhi; jj++)
            {
                const int j = k0 + jj;
                if (j >= k1) break;
                bool any = false;
#pragma unroll
                for (int rr = 0; rr < 2; rr++)
#pragma unroll
                    for (int hh = 0; hh < G; hh++) any |= (r0 + rr < rows) && sc[(r0 + rr) * G + hh][jj] != 0.0f;
                if (!any) continue;
                const KV* vrow = (const KV*)(a.v + (long long)j * a.v_nb1 + (long long)g * a.v_nb2) + lane * per;
                float vv[per];
#pragma unroll
                for (int u = 0; u < per; u++) vv[u] = kv_load<KV>(vrow + u);
#pragma unroll
                for (int rr = 0; rr < 2; rr++)
                {
                    if (r0 + rr >= rows) break;
#pragma unroll
                    for (int hh = 0; hh < G; hh++)
                    {
                        const float w = sc[(r0 + rr) * G + hh][jj];
#pragma unroll
                        for (int u = 0; u < per; u++) vacc[rr][hh][u] += w * vv[u];
                    }
                }
            }
            // The warps' sums in warp order (deterministic).
            for (int w = 0; w < WARPS; w++)
            {
                if (warp == w)
#pragma unroll
                    for (int rr = 0; rr < 2; rr++)
                    {
                        if (r0 + rr >= rows) break;
#pragma unroll
                        for (int hh = 0; hh < G; hh++)
#pragma unroll
                            for (int u = 0; u < per; u++) sums[((r0 + rr) * G + hh) * DV + lane * per + u] += vacc[rr][hh][u];
                    }
                __syncthreads();
            }
        }
        __syncthreads();
        for (int r = 0; r < rows; r++)
        {
            float* out = out_of(r);
            for (int o = threadIdx.x; o < G * (2 + DV); o += blockDim.x)
            {
                const int hh = o / (2 + DV), e = o - hh * (2 + DV), pr = r * G + hh;
                out[o] = e == 0 ? mx[pr] : e == 1 ? sm[pr] : sums[pr * DV + e - 2];
            }
        }
    }

    template <typename KV>
    bool launch_split_rows(const AttnArgs& a, float* part, int splits, cudaStream_t stream)
    {
        // Opt-in (TS_INKLING_SPLIT_ROWS=1): measured slower than the per-row kernel, whose blocks share a split's keys in L2.
        static const bool off = []() { const char* e = std::getenv("TS_INKLING_SPLIT_ROWS"); return e == nullptr || e[0] != '1'; }();
        // Only where the window has splits enough to fill the GPU (a ring of 1536 keys is 6 splits: the per-row kernel's
        // blocks run faster there) and its reads matter.
        if (off || a.rows < 2 || a.rows > kSplitRowsMax || splits < 32) return false;
        const dim3 grid((unsigned int)a.kv_heads, (unsigned int)splits);
        switch (a.heads / a.kv_heads)
        {
            case 1: inkling_attention_split_rows<KV, 1><<<grid, kSplitThreads, 0, stream>>>(a, part, splits); return true;
            case 2: inkling_attention_split_rows<KV, 2><<<grid, kSplitThreads, 0, stream>>>(a, part, splits); return true;
            case 3: inkling_attention_split_rows<KV, 3><<<grid, kSplitThreads, 0, stream>>>(a, part, splits); return true;
            default: return false;
        }
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
    if (kv_heads <= 0) return 0;
    if (rows > kSplitMaxRows)
    {
        // A prompt chunk split over many keys keeps its splits' partials.
        const int sp = mma2_splits(rows, kv_heads, keys, mma2_split_cap());
        return sp > 1 ? rows * heads * sp * (2 + dv) : 0;
    }
    // The tensor-core verify kernel's splits (kVmmaKeys keys) where it may run, else the split kernel's.
    const int64_t per = rows >= 2 ? std::min(kSplitKeys, kVmmaKeys) : kSplitKeys;
    const int64_t splits = (keys + per - 1) / per;
    return rows * kv_heads * splits * (heads / kv_heads) * (2 + dv);
}

bool tsg_inkling_attention_supported(int d, int dv, int heads, int kv_heads)
{
    if (kv_heads < 1 || heads % kv_heads != 0) return false;
    const int g = heads / kv_heads;
    // The split kernel takes any width to 256; the tensor-core one 128-wide heads.
    return g <= kMaxGroup && d == kMmaWidth && dv == kMmaWidth;
}

int64_t tsg_inkling_experts_workspace_bytes(int64_t k, int64_t experts, int64_t used, int64_t rows, int64_t activations)
{
    const int64_t cap = std::min(experts, used * rows);
    const int64_t group_ints = 1 + cap + cap + 1 + used * rows;
    return (group_ints * 4 + 15) / 16 * 16 + (activations * k + 15) / 16 * 16 + activations * (k / 32) * 4;
}

bool tsg_inkling_experts_supported(enum ggml_type type)
{
    return type == GGML_TYPE_NVFP4 || type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL;
}

bool tsg_inkling_experts_fp4(enum ggml_type type, int64_t k, int64_t m)
{
    if (type != GGML_TYPE_NVFP4 || k % 128 != 0 || m % kF4Rows != 0) return false;
    static const bool on = []() {
        const char* e = std::getenv("TS_INKLING_EXPERTS_FP4");
        if (e != nullptr && e[0] == '0') return false;
        // Whether this library's kernels carry the FP4 MMA (built for an architecture-specific target): a probe, once, before any capture.
        int* d = nullptr;
        int h = 0;
        if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) return false;
        inkling_f4_probe<<<1, 1>>>(d);
        const bool ok = cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
        cudaFree(d);
        cudaGetLastError();
        return ok && h == 1;
    }();
    return on;
}

int64_t tsg_inkling_experts_fp4_workspace_bytes(int64_t k, int64_t experts, int64_t used, int64_t rows, int64_t activations)
{
    const int64_t pairs = used * rows, cap = std::min(experts, pairs);
    const int64_t group_ints = 1 + cap + cap + 1 + pairs;
    const int64_t tiles = pairs / kF4Pairs + experts;
    const int64_t a_row = (k / 64 * 36 + 15) / 16 * 16;
    return (group_ints * 4 + 15) / 16 * 16 + ((3 * tiles + 1) * 4 + 15) / 16 * 16 + (activations * 4 + 15) / 16 * 16 + activations * a_row;
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
                (float *)dst->data, dim, heads, vectors, d->f0, (long long)(x->nb[2] / sizeof(float)));
        } break;
        case TSG_INKLING_CONV:
        {
            const ggml_tensor * x = dst->src[0], * window = dst->src[1], * kernel = dst->src[2];
            const ggml_tensor * dyn = d->i0 > 0 ? dst->src[3] : nullptr;
            const ggml_tensor * history = d->i1 == 1 ? dst->src[d->i0 > 0 ? 4 : 3] : nullptr;
            const int channels = (int)x->ne[0], count = (int)x->ne[1], taps = (int)kernel->ne[0];
            // x and the dynamic taps at their row strides (a packed product's slices).
            const long long x_row = (long long)(x->nb[1] / sizeof(float)), dyn_row = dyn ? (long long)(dyn->nb[1] / sizeof(float)) : 0;
            if (history != nullptr)
            {
                // A verify window: every row from the live window, which stays as it is; the history keeps [window | rows].
                inkling_conv_rows<<<dim3(blocks(channels, 128), count), 128, 0, stream>>>((const float *)x->data,
                    (const float *)window->data, (const float *)kernel->data, dyn ? (const float *)dyn->data : nullptr,
                    (float *)dst->data, channels, count, taps, d->i0, d->f0, x_row, dyn_row);
                inkling_conv_history<<<blocks((long long)(taps - 1 + count) * channels, 256), 256, 0, stream>>>((const float *)x->data,
                    (const float *)window->data, (float *)history->data, channels, count, taps - 1, x_row);
            }
            else if (count >= kConvRowsParallel)
            {
                inkling_conv_rows<<<dim3(blocks(channels, 128), count), 128, 0, stream>>>((const float *)x->data,
                    (const float *)window->data, (const float *)kernel->data, dyn ? (const float *)dyn->data : nullptr,
                    (float *)dst->data, channels, count, taps, d->i0, d->f0, x_row, dyn_row);
                inkling_conv_window<<<blocks(channels, 128), 128, 0, stream>>>((const float *)x->data, (float *)window->data,
                    channels, count, taps, x_row);
            }
            else
                inkling_conv<<<blocks(channels, 128), 128, 0, stream>>>((const float *)x->data, (float *)window->data,
                    (const float *)kernel->data, dyn ? (const float *)dyn->data : nullptr, (float *)dst->data, channels, count, taps, d->i0,
                    d->f0, x_row, dyn_row);
        } break;
        case TSG_INKLING_PICK:
        {
            const ggml_tensor * base = dst->src[0], * fold = dst->src[1], * bias = dst->src[2];
            const int lanes = (int)base->ne[1], padded = (int)base->ne[0];
            inkling_pick<<<lanes, 1024, 0, stream>>>((const float *)base->data, (long long)(base->nb[1] / sizeof(float)),
                (const float *)fold->data, (const float *)bias->data, padded, d->i0, lanes, d->f0, (float *)dst->data);
        } break;
        case TSG_INKLING_REL_TABLE:
        {
            const ggml_tensor * w = dst->src[0], * r = dst->src[1];
            const int rel = (int)w->ne[0], width = (int)w->ne[1];
            const long long pairs = r->ne[1] * r->ne[2];
            GGML_ASSERT(rel == kRelWidth);
            inkling_rel_table<kRelWidth><<<blocks(pairs, kRelPairs), kRelThreads, 0, stream>>>((const float *)w->data, (const char *)r->data,
                (long long)r->nb[1], (long long)r->nb[2], (int)r->ne[1], pairs, width, (__half *)dst->data);
        } break;
        case TSG_INKLING_PICK_I32:
        {
            const ggml_tensor * pick = dst->src[0];
            const int lanes = (int)pick->ne[0];
            inkling_pick_i32<<<blocks(lanes, 32), 32, 0, stream>>>((const float *)pick->data, (int32_t *)dst->data, lanes);
        } break;
        case TSG_INKLING_CONV_BATCH:
        {
            const ggml_tensor * x = dst->src[0], * kernel = dst->src[1];
            const ggml_tensor * dyn = d->i0 > 0 ? dst->src[2] : nullptr;
            const auto * b = static_cast<const tsg_inkling_conv_batch *>(d->object);
            ConvBatchArgs a{};
            a.x = (const float *)x->data; a.x_row = (long long)(x->nb[1] / sizeof(float));
            a.kw = (const float *)kernel->data;
            a.dyn = dyn ? (const float *)dyn->data : nullptr; a.dyn_row = dyn ? (long long)(dyn->nb[1] / sizeof(float)) : 0;
            a.y = (float *)dst->data;
            a.channels = (int)x->ne[0]; a.rows = (int)x->ne[1]; a.taps = (int)kernel->ne[0]; a.heads = d->i0; a.n = b->n; a.scale = d->f0;
            int widest = 0;
            for (int s = 0; s < b->n; s++)
            {
                a.first[s] = b->first[s]; a.count[s] = b->count[s];
                widest = std::max(widest, b->count[s]);
                // The channels' place in the leaves: the window at taps - 1 a channel, the history at its pitch.
                a.window[s] = (float *)b->window[s]->data + (long long)b->offset * b->hist;
                a.history[s] = b->history[s] ? (float *)b->history[s]->data + (long long)b->offset * (b->hist + b->count[s]) : nullptr;
            }
            inkling_conv_batch_rows<<<dim3(blocks(a.channels, 128), a.rows), 128, 0, stream>>>(a);
            if (d->i1 == 1)
                inkling_conv_batch_history<<<dim3(blocks((long long)(b->hist + widest) * a.channels, 256), b->n), 256, 0, stream>>>(a);
            else
                inkling_conv_batch_window<<<dim3(blocks(a.channels, 128), b->n), 128, 0, stream>>>(a);
        } break;
        case TSG_INKLING_MIX:
        {
            const ggml_tensor * states = dst->src[0], * query = dst->src[1];
            const int h = (int)states->ne[0], n = (int)states->ne[1], rows = (int)states->ne[2], queries = (int)query->ne[1];
            // The partials live past the result, in the node's tail (the builder sizes it).
            const size_t held = (size_t)n * h * sizeof(float);
            if (rows >= kMixRowsFrom && held <= (size_t)kMixRowSmemMax)
            {
                static bool attr = false;
                if (!attr)
                {
                    CUDA_CHECK(cudaFuncSetAttribute(inkling_mix_row, cudaFuncAttributeMaxDynamicSharedMemorySize, kMixRowSmemMax));
                    attr = true;
                }
                inkling_mix_row<<<rows, 256, held, stream>>>((const char *)states->data, (long long)states->nb[1],
                    (long long)states->nb[2], n, (const float *)query->data, queries, (float *)dst->data, h, d->f0, d->f1);
                break;
            }
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
            static const bool narrow = std::getenv("TS_INKLING_CONTROL_NARROW") != nullptr;
            if (rows >= kCtlRowsFrom && rank == kCtlRank && !narrow)
            {
                // g's place in the node's tail holds the splits' partials (the builder sizes it).
                const int splits = (h + kCtlSplit - 1) / kCtlSplit;
                // TF32 tensor cores (TS_INKLING_CONTROL_TC=0: the CUDA-core kernels); their weight loads need 32-byte rows.
                static const bool tc_off = []() { const char* e = std::getenv("TS_INKLING_CONTROL_TC"); return e != nullptr && e[0] == '0'; }();
                const bool tc = !tc_off && h % kTcChunk == 0 && h % kTcBlendCh == 0 && (uintptr_t)down->data % 32 == 0
                    && (uintptr_t)gain->data % 32 == 0 && (uintptr_t)forget->data % 32 == 0;
                if (tc)
                    inkling_control_down_tc<<<dim3((rows + kTcRows - 1) / kTcRows, splits), 128, 0, stream>>>((const float *)x->data,
                        (const float *)m->data, (const float *)down->data, g, h, rows);
                else
                    inkling_control_down_split<<<dim3((rows + kCtlRows - 1) / kCtlRows, splits), 256, 0, stream>>>((const float *)x->data,
                        (const float *)m->data, (const float *)down->data, g, h, rows);
                float * gr = g + (long long)splits * rows * kCtlPart;
                inkling_control_reduce<<<rows, kCtlRank, 0, stream>>>(g, splits, gr, h, rows, d->f0);
                if (tc)
                    inkling_control_blend_tc<<<dim3(h / kTcBlendCh, (rows + kTcBlendRows - 1) / kTcBlendRows), 128, 0, stream>>>(
                        (const float *)x->data, (const float *)m->data, (const float *)gain->data, (const float *)forget->data, gr, y, h, rows);
                else
                    inkling_control_blend_rows<<<dim3((h + kCtlTile - 1) / kCtlTile, (rows + kBlendRows - 1) / kBlendRows), 256, 0, stream>>>(
                        (const float *)x->data, (const float *)m->data, (const float *)gain->data, (const float *)forget->data, gr, y, h, rows);
                break;
            }
            if (narrow)
                inkling_control_down<<<dim3(blocks(rank, 8), rows), 256, 0, stream>>>((const float *)x->data, (const float *)m->data,
                    (const float *)down->data, g, h, rank, d->f0);
            else
                inkling_control_down_wide<<<dim3(rank, rows), 256, 0, stream>>>((const float *)x->data, (const float *)m->data,
                    (const float *)down->data, g, h, rank, d->f0);
            inkling_control_blend<<<dim3(blocks(h, 8), rows), 256, 0, stream>>>((const float *)x->data, (const float *)m->data,
                (const float *)gain->data, (const float *)forget->data, g, y, h, rank);
        } break;
        case TSG_INKLING_ATTENTION:
        {
            const ggml_tensor * q = dst->src[0], * k = dst->src[1], * v = dst->src[2], * table = dst->src[3], * positions = dst->src[4];
            const ggml_tensor * sink = dst->src[5];
            AttnArgs a{};
            a.sink = (const float *)sink->data;
            a.q = (const char *)q->data; a.q_nb1 = (long long)q->nb[1]; a.q_nb2 = (long long)q->nb[2];
            a.k = (const char *)k->data; a.k_nb1 = (long long)k->nb[1]; a.k_nb2 = (long long)k->nb[2];
            a.v = (const char *)v->data; a.v_nb1 = (long long)v->nb[1]; a.v_nb2 = (long long)v->nb[2];
            a.T = table->data;
            a.t_half = table->type == GGML_TYPE_F16 ? 1 : 0;
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
                // Many keys: the chunk's attention split over them (partials in the node's tail), then combined.
                const int splits = mma2_splits(a.rows, a.kv_heads, a.keys, mma2_split_cap());
                float * part = splits > 1 ? a.out + (long long)a.rows * a.heads * a.dv : nullptr;
                bool split_ran = false;
                if (k->type == GGML_TYPE_F16) { if (launch_mma2<half>(a, stream, part, splits)) split_ran = part != nullptr; else launch_mma<half>(a, stream); }
                else if (launch_mma2<float>(a, stream, part, splits)) split_ran = part != nullptr; else launch_mma<float>(a, stream);
                if (split_ran)
                    inkling_attention_combine<<<dim3((unsigned int)a.heads, (unsigned int)a.rows), 128, 0, stream>>>(part, a.out, a.heads,
                        a.kv_heads, a.dv, splits);
            }
            else
            {
                const int splits = (a.keys + kSplitKeys - 1) / kSplitKeys;
                float * part = a.out + (long long)a.rows * a.heads * a.dv;
                const dim3 sgrid((unsigned int)a.rows, (unsigned int)a.kv_heads, (unsigned int)splits);
                // TS_INKLING_ATTN_PROFILE=1 (with TS_INKLING_CAPTURE=0): the split and combine kernels' device time, waited on.
                static const bool profile = std::getenv("TS_INKLING_ATTN_PROFILE") != nullptr;
                static cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr;
                if (profile && e0 == nullptr) { cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); }
                if (profile) cudaEventRecord(e0, stream);
                // A verify window over a long linear window: the prompt kernel (K/V tiles staged through shared memory, an online
                // softmax a warp's rows) in splits of ~2,048 keys, so the keys stream at bandwidth and the partials stay small
                // (TS_INKLING_VERIFY_MMA2: the fewest keys, default 8,192; 0 never). Else on tensor cores in kVmmaKeys-key
                // splits, else the split kernels.
                static const int vmma2_min = []() { const char* e = std::getenv("TS_INKLING_VERIFY_MMA2"); return e != nullptr ? std::atoi(e) : 8192; }();
                int ran = 0;
                if (vmma2_min > 0 && a.ring == 0 && a.keys >= vmma2_min && a.rows >= 2 && k->type == GGML_TYPE_F16 && a.kv_aligned)
                {
                    const int sp = std::min(32, std::max(2, a.keys / 2048));
                    if (launch_mma2<half>(a, stream, part, sp)) ran = sp;
                }
                if (ran == 0) ran = launch_verify_mma(a, k, part, stream);
                if (ran == 0)
                {
                    if (k->type == GGML_TYPE_F16) { if (!launch_split_rows<half>(a, part, splits, stream)) launch_split<half>(a, part, splits, sgrid, stream); }
                    else if (!launch_split_rows<float>(a, part, splits, stream)) launch_split<float>(a, part, splits, sgrid, stream);
                    ran = splits;
                }
                if (profile) cudaEventRecord(e1, stream);
                inkling_attention_combine<<<dim3((unsigned int)a.heads, (unsigned int)a.rows), 128, 0, stream>>>(part, a.out, a.heads,
                    a.kv_heads, a.dv, ran);
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
            const ggml_tensor * gu = dst->src[0], * scale = dst->src[1];
            const long long ff = gu->ne[0] / 2, vectors = gu->ne[1] * gu->ne[2] * gu->ne[3];
            inkling_situ<<<blocks(ff * vectors, 256), 256, 0, stream>>>((const float *)gu->data,
                scale != nullptr ? (const float *)scale->data : nullptr, (float *)dst->data, ff, vectors, d->f0, d->f1);
        } break;
        case TSG_INKLING_EXPERT_SUM:
        {
            const ggml_tensor * down = dst->src[0], * weights = dst->src[1];
            const long long h = down->ne[0], rows = down->ne[2];
            inkling_expert_sum<<<blocks(h * rows, 256), 256, 0, stream>>>((const float *)down->data, (const float *)weights->data,
                (float *)dst->data, h, (int)down->ne[1], rows);
        } break;
        case TSG_INKLING_EXPERTS:
        {
            const ggml_tensor * w = dst->src[0], * x = dst->src[1], * ids = dst->src[2];
            MoeArgs a{};
            a.w = (const char *)w->data; a.w_row = (long long)w->nb[1]; a.w_expert = (long long)w->nb[2];
            a.x = (const float *)x->data; a.x_nb1 = (long long)x->nb[1]; a.x_nb2 = (long long)x->nb[2];
            a.ids = (const int32_t *)ids->data; a.ids_nb1 = (long long)ids->nb[1];
            a.k = (int)w->ne[0]; a.m = (int)w->ne[1]; a.experts = (int)w->ne[2];
            a.used = (int)ids->ne[0]; a.rows = (int)ids->ne[1]; a.acts = (int)x->ne[1];
            a.cap = std::min(a.experts, a.used * a.rows);
            a.dst = (float *)dst->data;
            if (d->i0 == 1)
            {
                // A prompt chunk on FP4 tensor cores: the groups, their tiles of 64 pairs, the activations as NVFP4 rows, the GEMM.
                char * ws = (char *)dst->data + (size_t)a.m * a.used * a.rows * sizeof(float);
                const long long pairs = (long long)a.used * a.rows, acts = (long long)a.acts * a.rows;
                const long long group_ints = 1 + a.cap + a.cap + 1 + pairs, max_tiles = pairs / kF4Pairs + a.experts;
                const long long a_row = ((long long)a.k / 64 * 36 + 15) / 16 * 16;
                a.groups = (int *)ws;
                int * tiles = (int *)(ws + (group_ints * 4 + 15) / 16 * 16);
                int * ntiles = tiles + 3 * max_tiles;
                float * arow = (float *)((char *)tiles + ((3 * max_tiles + 1) * 4 + 15) / 16 * 16);
                char * aq = (char *)arow + (acts * 4 + 15) / 16 * 16;
                inkling_moe_group<<<1, 256, 2 * a.experts * sizeof(int), stream>>>(a);
                inkling_f4_tiles<<<1, 32, 0, stream>>>(a.groups, a.cap, tiles, ntiles);
                inkling_f4_quantize<<<blocks(acts, kF4QuantRows), 32 * kF4QuantRows, 0, stream>>>(a.x, a.x_nb1 / 4, a.x_nb2 / 4, a.acts, a.k,
                    aq, a_row, arow, acts);
                Fp4Args f{};
                f.w = a.w; f.w_row = a.w_row; f.w_expert = a.w_expert;
                f.aq = aq; f.a_row = a_row; f.arow = arow; f.tiles = tiles; f.ntiles = ntiles;
                f.list = a.groups + 1 + a.cap + a.cap + 1; f.used = a.used; f.acts = a.acts; f.k = a.k; f.n = a.m; f.y = a.dst;
                static bool attr = false;
                if (!attr) { cudaFuncSetAttribute(inkling_f4_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kF4Smem); attr = true; }
                inkling_f4_gemm<<<f4_grid((int)a.m, (unsigned int)max_tiles), kF4Threads, kF4Smem, stream>>>(f);
                break;
            }
            // The workspace past the result: the groups (ints), the int8 activations, their scales.
            char * ws = (char *)dst->data + (size_t)a.m * a.used * a.rows * sizeof(float);
            a.groups = (int *)ws;
            const long long group_ints = 1 + a.cap + a.cap + 1 + (long long)a.used * a.rows;
            const long long acts = (long long)a.acts * a.rows;
            a.q = (int8_t *)(ws + ((group_ints * 4 + 15) / 16) * 16);
            a.qd = (float *)((char *)a.q + ((acts * a.k + 15) / 16) * 16);
            inkling_moe_group<<<1, 256, 2 * a.experts * sizeof(int), stream>>>(a);
            inkling_moe_quantize<<<blocks(acts * (a.k / 32), 8), 256, 0, stream>>>(a);
            switch (w->type)
            {
                case GGML_TYPE_NVFP4: launch_moe_product<GGML_TYPE_NVFP4>(a, stream); break;
                case GGML_TYPE_IQ4_NL: launch_moe_product<GGML_TYPE_IQ4_NL>(a, stream); break;
                default: launch_moe_product<GGML_TYPE_Q8_0>(a, stream); break;
            }
        } break;
        default:
            GGML_ABORT("tsg_inkling_cuda_compute: unknown kind %d", d->kind);
    }
    CUDA_CHECK(cudaGetLastError());
}

namespace
{
    struct EventBucket
    {
        double ms = 0;
        long long calls = 0;
    };

    struct EventProfile
    {
        std::map<std::string, EventBucket> buckets;
        double total_ms = 0;
        long long graphs = 0;
    };

    struct InklingCapture
    {
        cudaGraphExec_t exec = nullptr;
        int runs = 0;
        bool failed = false;
        // TS_INKLING_EVENT_PROFILE: an event record node before each graph node and after the last, and each node's key.
        std::vector<cudaEvent_t> events;
        std::vector<std::string> keys;
    };

    std::string event_key(const ggml_tensor * node)
    {
        std::string key = ggml_op_name(node->op);
        if ((node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) && node->src[0] != nullptr && node->src[1] != nullptr)
        {
            const ggml_tensor * w = node->src[0];
            const ggml_tensor * x = node->src[1];
            key += std::string(":") + ggml_type_name(w->type) + ":" + std::to_string(w->ne[0]) + "x" + std::to_string(w->ne[1]);
            if (node->op == GGML_OP_MUL_MAT_ID) key += "x" + std::to_string(w->ne[2]) + ":ids" + std::to_string(node->src[2]->ne[0]);
            key += ":rows" + std::to_string(x->ne[1] * x->ne[2] * x->ne[3]);
        }
        else if (node->op == GGML_OP_CUSTOM)
        {
            ggml_custom_op_params params;
            memcpy(&params, node->op_params, sizeof(params));
            const auto * d = static_cast<const tsg_dsv4_fused_desc *>(params.userdata);
            static const char * names[] = { "clip", "conv", "mix", "control", "attention", "route_ids", "route_weights", "situ", "expert_sum", "experts" };
            if (d != nullptr && d->magic == TSG_DSV4_FUSED_MAGIC && d->kind >= 40 && d->kind <= 49)
            {
                key = std::string("fused:") + names[d->kind - 40];
                if (d->kind == TSG_INKLING_ATTENTION)
                    key += std::string(d->i0 > 0 ? ":ring" : ":linear") + ":extent" + std::to_string(d->i1) + ":q" + std::to_string(node->src[0]->ne[1]) + "x" + std::to_string(node->src[0]->ne[2]);
                if (d->kind == TSG_INKLING_EXPERTS)
                    key += std::string(":") + ggml_type_name(node->src[0]->type) + ":" + std::to_string(node->src[0]->ne[0]) + "x" + std::to_string(node->src[0]->ne[1]) + ":rows" + std::to_string(node->src[2]->ne[1]);
            }
            else
                key = std::string("fused:kind") + (d != nullptr ? std::to_string(d->kind) : std::string("?"));
        }
        else if (node->op == GGML_OP_UNARY)
            key += std::string(":") + ggml_unary_op_name(ggml_get_unary_op(node));
        else if (node->op == GGML_OP_GLU)
            key += std::string(":") + ggml_glu_op_name(ggml_get_glu_op(node));
        return key;
    }

    void event_profile_add(const InklingCapture & c)
    {
        static std::map<size_t, EventProfile> profiles;
        static const int every = []() { const char * e = std::getenv("TS_INKLING_EVENT_PROFILE_EVERY"); return e != nullptr ? std::max(1, std::atoi(e)) : 32; }();
        static const int top = []() { const char * e = std::getenv("TS_INKLING_EVENT_PROFILE_TOP"); return e != nullptr ? std::atoi(e) : 40; }();
        const size_t n = c.keys.size();
        EventProfile & p = profiles[n];
        double total = 0;
        for (size_t i = 0; i < n; i++)
        {
            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, c.events[i], c.events[i + 1]));
            EventBucket & b = p.buckets[c.keys[i]];
            b.ms += ms;
            b.calls++;
            total += ms;
        }
        p.total_ms += total;
        p.graphs++;
        if (p.graphs % every != 0) return;
        std::vector<std::pair<std::string, EventBucket>> sorted(p.buckets.begin(), p.buckets.end());
        std::sort(sorted.begin(), sorted.end(), [](const auto & a, const auto & b) { return a.second.ms > b.second.ms; });
        std::fprintf(stderr, "[event-profile] %zu nodes: %lld graphs, %.3f ms device/graph\n", n, p.graphs, p.total_ms / p.graphs);
        for (size_t i = 0; i < sorted.size() && (int)i < top; i++)
        {
            const EventBucket & b = sorted[i].second;
            std::fprintf(stderr, "    %8.3f ms %5.1f%% %6lld/graph %7.2f us each  %s\n", b.ms / p.graphs, 100.0 * b.ms / p.total_ms,
                         b.calls / p.graphs, 1000.0 * b.ms / b.calls, sorted[i].first.c_str());
        }
        std::fflush(stderr);
    }
}

bool tsg_inkling_event_profile_enabled()
{
    static const bool on = []() { const char * e = std::getenv("TS_INKLING_EVENT_PROFILE"); return e != nullptr && e[0] != '0'; }();
    return on;
}

void * tsg_inkling_capture_new() { return new InklingCapture(); }

void tsg_inkling_capture_free(void * capture)
{
    auto * c = static_cast<InklingCapture *>(capture);
    if (c == nullptr) return;
    if (c->exec != nullptr) cudaGraphExecDestroy(c->exec);
    for (auto e : c->events) cudaEventDestroy(e);
    delete c;
}

ggml_status tsg_inkling_compute(void * capture, ggml_backend_t fused_backend, ggml_backend_t cuda_backend, ggml_cgraph * graph,
                                int warmup)
{
    auto * c = static_cast<InklingCapture *>(capture);
    const bool profile = tsg_inkling_event_profile_enabled();
    // A graph ggml-cuda would run with a stream synchronize in it (an expert product on its host-sorted fallback) is never
    // captured: the synchronize fails the capture inside ggml-cuda, which aborts the process rather than returning.
    if (c != nullptr && c->runs == 0 && !c->failed && !tsg_cuda_graph_capturable(cuda_backend, graph)) c->failed = true;
    if (c != nullptr && !c->failed && c->exec == nullptr)
    {
        // The runs ggml-cuda is handed start at the graph's first node and after each fused node: none captures itself, so
        // the capture below records their kernels (ggml-cuda would otherwise begin a capture on the capturing stream).
        // Profiled, every node is a run of its own.
        std::vector<const void *> keys;
        bool after_fused = true;
        for (int i = 0; i < graph->n_nodes; i++)
        {
            const bool fused = graph->nodes[i]->op == GGML_OP_CUSTOM;
            if (profile || (!fused && after_fused)) keys.push_back(graph->nodes[i]);
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
        if (profile)
        {
            c->events.resize((size_t)graph->n_nodes + 1);
            for (auto & e : c->events) CUDA_CHECK(cudaEventCreate(&e));
            c->keys.clear();
            for (int i = 0; i < graph->n_nodes; i++) c->keys.push_back(event_key(graph->nodes[i]));
        }
        cudaGraph_t g = nullptr;
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed) != cudaSuccess)
        {
            (void)cudaGetLastError();
            c->failed = true;
            return ggml_backend_graph_compute_async(fused_backend, graph);
        }
        ggml_status st = GGML_STATUS_SUCCESS;
        if (profile)
        {
            for (int i = 0; i < graph->n_nodes && st == GGML_STATUS_SUCCESS; i++)
            {
                cudaEventRecordWithFlags(c->events[i], stream, cudaEventRecordExternal);
                ggml_cgraph view = ggml_graph_view(graph, i, i + 1);
                st = ggml_backend_graph_compute_async(fused_backend, &view);
            }
            cudaEventRecordWithFlags(c->events[graph->n_nodes], stream, cudaEventRecordExternal);
        }
        else
            st = ggml_backend_graph_compute_async(fused_backend, graph);
        const cudaError_t e = cudaStreamEndCapture(stream, &g);
        if (st != GGML_STATUS_SUCCESS || e != cudaSuccess || g == nullptr
            || cudaGraphInstantiate(&c->exec, g, 0) != cudaSuccess)
        {
            // Nothing ran while capturing: run the graph as it is, and do so from now on.
            (void)cudaGetLastError();
            if (g != nullptr) cudaGraphDestroy(g);
            c->exec = nullptr;
            c->failed = true;
            c->keys.clear();
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
    if (profile && !c->keys.empty())
    {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        event_profile_add(*c);
    }
    return GGML_STATUS_SUCCESS;
}

ggml_status tsg_inkling_event_profile(ggml_backend_t fused_backend, ggml_backend_t cuda_backend, ggml_cgraph * graph)
{
    // A graph that is not replayed: captured with its events, run once, and released.
    void * c = tsg_inkling_capture_new();
    const ggml_status st = tsg_inkling_compute(c, fused_backend, cuda_backend, graph, 0);
    auto * context = static_cast<ggml_backend_cuda_context *>(cuda_backend->context);
    CUDA_CHECK(cudaStreamSynchronize(context->stream(context->device, 0)));
    tsg_inkling_capture_free(c);
    return st;
}
#endif
