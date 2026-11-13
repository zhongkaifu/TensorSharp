// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Kernels of the direct CUDA backend's graph executor (CudaGraphExecutor): the program ops that
// have no generic tensor-op equivalent, with ggml's semantics so a program computes the same thing
// on every backend.

#define TS_GRAPH_KERNELS_VERSION 6

extern "C" __global__ void ts_graph_caps(int* out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    out[0] = TS_GRAPH_KERNELS_VERSION;
#if defined(__CUDA_ARCH__)
    out[1] = __CUDA_ARCH__;
#else
    out[1] = 0;
#endif
}

__device__ __forceinline__ float ts_graph_warp_sum(float v)
{
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ggml_gated_delta_net (ggml-cpu's loop, one warp per value row of a head's state):
//   q, k  [n_s, n_t, hk, S]   (value head h reads key head h % hk)
//   v     [n_s, n_t, h, S]
//   g     [n_s, n_t, h, 1] (decay exp(g) per head) or [n_s, n_t, h, S] (per key dim, kda)
//   beta  [n_s, n_t, h]
//   state [n_s, h, S, S]: row j of a head's state holds column j of S (s[j * S + i] = S[i][j])
//   out   [outputs n_s * n_t * h * S | snapshots slots x n_s * h * S * S], slot 0 the state after the
//         last row, slot s the state s rows before it (slots older than the rows are left as they are).
// Per row: S *= exp(g); delta = (v - S^T k) * beta; S += k delta^T; y = S^T q / sqrt(S).
// Launch: grid (n_s * h, ceil(S / 4)), block 128 (4 warps, one value row each); S % 32 == 0, S <= 256.
extern "C" __global__ void ts_graph_gated_delta_net_f32(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta, const float* __restrict__ state,
    float* __restrict__ out, int S, int h, int hk, int n_t, int n_s, int slots, int kda)
{
    const int seq = blockIdx.x / h;
    const int head = blockIdx.x % h;
    const int kh = head % hk;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.y * 4 + (threadIdx.x >> 5);
    if (seq >= n_s || row >= S)
        return;

    const int per = S >> 5;   // elements of the row a lane keeps (<= 8)
    float st[8];
    const float* s_in = state + ((size_t)(seq * h + head) * S + row) * S;
#pragma unroll
    for (int j = 0; j < 8; j++)
        st[j] = j < per ? s_in[lane + 32 * j] : 0.0f;

    const float scale = rsqrtf((float)S);
    const size_t attn = (size_t)n_s * n_t * h * S;
    const size_t per_snap = (size_t)n_s * h * S * S;
    for (int t = 0; t < n_t; t++)
    {
        const size_t qk_off = ((size_t)(seq * n_t + t) * hk + kh) * S;
        const size_t v_off = ((size_t)(seq * n_t + t) * h + head) * S;
        const size_t gb = (size_t)(seq * n_t + t) * h + head;
        float kv = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            if (j >= per) break;
            const int i = lane + 32 * j;
            st[j] *= expf(kda ? g[gb * S + i] : g[gb]);
            kv += st[j] * k[qk_off + i];
        }
        kv = ts_graph_warp_sum(kv);
        const float delta = (v[v_off + row] - kv) * beta[gb];
        float y = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            if (j >= per) break;
            const int i = lane + 32 * j;
            st[j] += k[qk_off + i] * delta;
            y += st[j] * q[qk_off + i];
        }
        y = ts_graph_warp_sum(y);
        if (lane == 0)
            out[v_off + row] = y * scale;

        const int slot = n_t - 1 - t;
        if (slot < slots)
        {
            float* s_out = out + attn + slot * per_snap + ((size_t)(seq * h + head) * S + row) * S;
#pragma unroll
            for (int j = 0; j < 8; j++)
                if (j < per) s_out[lane + 32 * j] = st[j];
        }
    }
}

// ts_graph_gated_delta_net_f32 with four state rows a warp: a token's q, k and gates are read once for the four rows
// (a prompt chunk re-read them for every row, 3.4x the legacy prefill scan's time on Qwen3.5-9B). Each state element
// goes through the same operations in the same order as in the one-row kernel, so the results are the same bits.
extern "C" __global__ void ts_graph_gated_delta_net_rows4_f32(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta, const float* __restrict__ state,
    float* __restrict__ out, int S, int h, int hk, int n_t, int n_s, int slots, int kda)
{
    const int seq = blockIdx.x / h;
    const int head = blockIdx.x % h;
    const int kh = head % hk;
    const int lane = threadIdx.x & 31;
    const int r0 = (blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 4;
    if (seq >= n_s || r0 >= S)
        return;

    const int per = S >> 5;   // elements of a row a lane keeps (<= 8)
    float st[4][8];
    const float* s_in = state + ((size_t)(seq * h + head) * S + r0) * S;
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int j = 0; j < 8; j++)
            st[r][j] = (j < per && r0 + r < S) ? s_in[(size_t)r * S + lane + 32 * j] : 0.0f;

    const float scale = rsqrtf((float)S);
    const size_t attn = (size_t)n_s * n_t * h * S;
    const size_t per_snap = (size_t)n_s * h * S * S;
    for (int t = 0; t < n_t; t++)
    {
        const size_t qk_off = ((size_t)(seq * n_t + t) * hk + kh) * S;
        const size_t v_off = ((size_t)(seq * n_t + t) * h + head) * S;
        const size_t gb = (size_t)(seq * n_t + t) * h + head;
        float kk[8], qq[8], e[8];
        const float eg = kda ? 0.0f : expf(g[gb]);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            if (j >= per) break;
            const int i = lane + 32 * j;
            kk[j] = k[qk_off + i];
            qq[j] = q[qk_off + i];
            e[j] = kda ? expf(g[gb * S + i]) : eg;
        }
        float kv[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
        for (int r = 0; r < 4; r++)
#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                if (j >= per) break;
                st[r][j] *= e[j];
                kv[r] += st[r][j] * kk[j];
            }
#pragma unroll
        for (int r = 0; r < 4; r++)
            kv[r] = ts_graph_warp_sum(kv[r]);
        const float b = beta[gb];
        float y[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll
        for (int r = 0; r < 4; r++)
        {
            const float delta = (r0 + r < S ? v[v_off + r0 + r] - kv[r] : 0.0f) * b;
#pragma unroll
            for (int j = 0; j < 8; j++)
            {
                if (j >= per) break;
                st[r][j] += kk[j] * delta;
                y[r] += st[r][j] * qq[j];
            }
        }
#pragma unroll
        for (int r = 0; r < 4; r++)
            y[r] = ts_graph_warp_sum(y[r]);
        if (lane == 0)
        {
#pragma unroll
            for (int r = 0; r < 4; r++)
                if (r0 + r < S) out[v_off + r0 + r] = y[r] * scale;
        }

        const int slot = n_t - 1 - t;
        if (slot < slots)
        {
            float* s_out = out + attn + slot * per_snap + ((size_t)(seq * h + head) * S + r0) * S;
#pragma unroll
            for (int r = 0; r < 4; r++)
            {
                if (r0 + r >= S) break;
#pragma unroll
                for (int j = 0; j < 8; j++)
                    if (j < per) s_out[(size_t)r * S + lane + 32 * j] = st[r][j];
            }
        }
    }
}

// ggml_softplus: x > 20 ? x : log(1 + exp(x)), elementwise over n contiguous floats.
extern "C" __global__ void ts_graph_softplus_f32(const float* __restrict__ x, float* __restrict__ y, long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const float v = x[i];
    y[i] = v > 20.0f ? v : logf(1.0f + expf(v));
}

// ggml_ssm_conv (ggml-cpu's order): x [n_s, channels, width - 1 + n_t] (time innermost), w [channels, width],
// out [n_s, n_t, channels]; out[s, t, c] = sum_k x[s, c, t + k] * w[c, k]. One thread per output.
extern "C" __global__ void ts_graph_ssm_conv_f32(const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ out, int channels, int length, int width, int n_t, int n_s)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)n_s * n_t * channels;
    if (i >= total)
        return;
    const int c = (int)(i % channels);
    const int t = (int)((i / channels) % n_t);
    const int s = (int)(i / ((long long)channels * n_t));
    const float* xr = x + ((size_t)s * channels + c) * length + t;
    const float* wr = w + (size_t)c * width;
    float acc = 0.0f;
    for (int k = 0; k < width; k++)
        acc += xr[k] * wr[k];
    out[i] = acc;
}

// GraphFusion.DeltaNetPrep for one sequence (the gated delta rule's inputs). Launch: grid (2 * k_heads + v_heads,
// rows), block d (d % 32 == 0, <= 1024): one block per (head, row), q heads then k heads then v heads, thread i
// owning channel head * d + i. window [channels, conv_rows] and conv_in [channels, conv_rows + rows] are time
// innermost; qkv [rows, channels] with rows qkv_pitch floats apart; conv_w [channels, width]; alpha, beta
// [rows, v_heads] with rows ab_pitch floats apart (views of a packed projection are read in place); dt, a [v_heads].
// Outputs: conv_in (window then rows); q, k [rows, heads, d] = silu(ssm_conv) L2-normed per head
// (x * rsqrt(sum x^2 + eps)); v [rows, v_heads, d] = silu(ssm_conv); g = softplus(alpha + dt) * a and
// beta = sigmoid(beta), [rows, v_heads].
extern "C" __global__ void ts_graph_deltanet_prep_f32(
    const float* __restrict__ window, const float* __restrict__ qkv, const float* __restrict__ conv_w,
    const float* __restrict__ alpha, const float* __restrict__ beta, const float* __restrict__ dt, const float* __restrict__ a,
    float* __restrict__ conv_in, float* __restrict__ q_out, float* __restrict__ k_out, float* __restrict__ v_out,
    float* __restrict__ g_out, float* __restrict__ beta_out,
    int channels, int conv_rows, int width, int d, int k_heads, int v_heads, int rows, float eps,
    long long qkv_pitch, long long ab_pitch)
{
    __shared__ float partial[32];
    const int i = threadIdx.x;
    const int head = blockIdx.x;
    const int t = blockIdx.y;
    const int span = conv_rows + rows;
    const size_t c = (size_t)head * d + i;
    float x = 0.0f;
    for (int j = 0; j < width; j++)
    {
        const int p = t + j;
        const float in = p < conv_rows ? window[c * conv_rows + p] : qkv[(size_t)(p - conv_rows) * qkv_pitch + c];
        x += in * conv_w[c * width + j];
    }
    conv_in[c * span + conv_rows + t] = qkv[(size_t)t * qkv_pitch + c];
    if (t == 0)
        for (int p = 0; p < conv_rows; p++) conv_in[c * span + p] = window[c * conv_rows + p];
    const float y = x / (1.0f + expf(-x));
    if (head < 2 * k_heads)
    {
        const float sq = ts_graph_warp_sum(y * y);
        if ((i & 31) == 0) partial[i >> 5] = sq;
        __syncthreads();
        float total = 0.0f;
        for (int s = 0; s < d / 32; s++) total += partial[s];
        const float yn = y * rsqrtf(total + eps);
        if (head < k_heads) q_out[((size_t)t * k_heads + head) * d + i] = yn;
        else k_out[((size_t)t * k_heads + head - k_heads) * d + i] = yn;
    }
    else
    {
        const int vh = head - 2 * k_heads;
        v_out[((size_t)t * v_heads + vh) * d + i] = y;
        if (i == 0)
        {
            const size_t gi = (size_t)t * v_heads + vh, ai = (size_t)t * ab_pitch + vh;
            const float s = alpha[ai] + dt[vh];
            g_out[gi] = (s > 20.0f ? s : logf(1.0f + expf(s))) * a[vh];
            beta_out[gi] = 1.0f / (1.0f + expf(-beta[ai]));
        }
    }
}

// The residual add and the next block's input norm in one pass: sum = a + b, normed = sum * rsqrt(mean(sum^2) + eps) * w
// over rows of cols (ggml_rms_norm(ggml_add(a, b)) * w). Launch: grid rows, block 512.
extern "C" __global__ void ts_graph_add_rms_norm_f32(const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ w, float* __restrict__ sum, float* __restrict__ normed, int cols, float eps)
{
    __shared__ float partial[32];
    const size_t r = blockIdx.x;
    const float* ar = a + r * cols;
    const float* br = b + r * cols;
    float* sr = sum + r * cols;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        const float v = ar[i] + br[i];
        sr[i] = v;
        ss += v * v;
    }
    ss = ts_graph_warp_sum(ss);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float t = threadIdx.x < (blockDim.x >> 5) ? partial[threadIdx.x] : 0.0f;
        t = ts_graph_warp_sum(t);
        if (threadIdx.x == 0) partial[0] = t;
    }
    __syncthreads();
    const float scale = rsqrtf(partial[0] / (float)cols + eps);
    float* nr = normed + r * cols;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        nr[i] = sr[i] * scale * w[i];
}

// GraphFusion.GatedRmsNorm: out = y * rsqrt(mean(y^2) + eps) * w * silu(z) over rows of d. Row r of y and out is head
// r % heads of token r / heads; z holds a token's heads * d values z_pitch floats after the previous token's (a view
// of a packed projection is read in place). Launch: grid rows, block d (d % 32 == 0, <= 1024).
extern "C" __global__ void ts_graph_gated_rms_norm_f32(const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ w, float* __restrict__ out, int d, float eps, int heads, long long z_pitch)
{
    __shared__ float partial[32];
    const int i = threadIdx.x;
    const size_t r = blockIdx.x;
    const float v = y[r * d + i];
    const float sq = ts_graph_warp_sum(v * v);
    if ((i & 31) == 0) partial[i >> 5] = sq;
    __syncthreads();
    float total = 0.0f;
    for (int s = 0; s < d / 32; s++) total += partial[s];
    const float zz = z[(r / heads) * z_pitch + (r % heads) * d + i];
    out[r * d + i] = v * rsqrtf(total / d + eps) * w[i] * (zz / (1.0f + expf(-zz)));
}
