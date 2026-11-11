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

#define TS_GRAPH_KERNELS_VERSION 2

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
