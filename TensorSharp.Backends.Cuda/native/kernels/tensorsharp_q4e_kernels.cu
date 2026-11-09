// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Qwen3.8-Flash-Next (qwen4exp) kernels for the direct-CUDA backend: the stages the
// DeepSeek V4 module (embedding, router selection, experts) and the GLM module (the gated
// delta recurrence) do not cover.
//
//   * low-rank hyper-connections over 4 residual streams: grouped RMS norm, a sigmoid gate
//     from a rank-R bottleneck, a mean collapse, and a 2 * sigmoid scatter per stream;
//   * Gated DeltaNet front end: one causal convolution over the joint q|k|v projection, L2
//     norms, K heads tiled over the V heads, and a per-head scalar decay (the recurrence is
//     GLM's ts_glm_kda_scan_f32, which takes a per-channel decay);
//   * gated GQA attention: per-head RMS norms, partial NEOX rotary, F16 caches, and attention
//     over every cell or over the cells Qwen Sparse Attention selects;
//   * QSA: raw indexer keys, pooled block keys, the relu-summed block scores and the cell
//     selection;
//   * the PLE block: the per-stream gate, the dilated depthwise convolution and its history.
//
// The semantics are those of the native executor ggml_cuda runs (ggml_ops_qwen4exp.cpp:
// q4e_nodes_gdn / q4e_nodes_attn / q4e_nodes_ffn / q4e_nodes_ple, ggml_ops_qwen4exp_qsa.inc).
//
// A launch covers nSeq sequences of rowsPerSeq consecutive rows: one sequence's ubatch
// (nSeq = 1) or one row of each of several sequences in a batched decode step
// (rowsPerSeq = 1), each with its own state passed through a device array of pointers.
// ---------------------------------------------------------------------------

#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float q4e_warp_sum(float v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off, 32);
    return v;
}

__device__ __forceinline__ float q4e_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float q4e_silu(float x)
{
    return x / (1.0f + expf(-x));
}

// ggml's softplus.
__device__ __forceinline__ float q4e_softplus(float x)
{
    return x > 20.0f ? x : logf(1.0f + expf(x));
}

// Sum over the block; every thread gets the total. `sh` holds 32 floats.
__device__ __forceinline__ float q4e_block_sum(float v, float* sh)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = q4e_warp_sum(v);
    if (lane == 0)
        sh[warp] = v;
    __syncthreads();
    if (warp == 0)
    {
        v = lane < ((blockDim.x + 31) >> 5) ? sh[lane] : 0.0f;
        v = q4e_warp_sum(v);
        if (lane == 0)
            sh[0] = v;
    }
    __syncthreads();
    const float total = sh[0];
    __syncthreads();
    return total;
}

// q4e_block_sum of N values with one barrier instead of three each: every value is summed exactly as q4e_block_sum
// sums it (its warp's butterfly, then warp 0's over the warps' totals), so the totals are bit for bit the same. Only
// thread 0 holds them afterwards. `sh` holds N x 32 floats.
template <int N>
__device__ __forceinline__ void q4e_block_sums_t0(float (&v)[N], float (*sh)[32])
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int k = 0; k < N; ++k)
        v[k] = q4e_warp_sum(v[k]);
    if (lane == 0)
    {
#pragma unroll
        for (int k = 0; k < N; ++k)
            sh[k][warp] = v[k];
    }
    __syncthreads();
    if (warp == 0)
    {
        const int warps = (blockDim.x + 31) >> 5;
#pragma unroll
        for (int k = 0; k < N; ++k)
            v[k] = q4e_warp_sum(lane < warps ? sh[k][lane] : 0.0f);
    }
}

// The NEOX rotation of pair (i, i + nRot/2) at `pos`, as ggml-cuda's rope_neox computes it
// (ext_factor 0, attn_factor 1).
__device__ __forceinline__ void q4e_rope_angle(int pos, int i, int nRot, float base, float freqScale,
    float* c, float* s)
{
    const float thetaScale = powf(base, -2.0f / (float)nRot);
    const float theta = freqScale * ((float)pos * powf(thetaScale, (float)i));
    *c = cosf(theta);
    *s = sinf(theta);
}

// ---------------------------------------------------------------------------
// hyper-connections
// ---------------------------------------------------------------------------

#define Q4E_HC_MAX 4

// xn[t] = each stream RMS-normed, times normW (hc*E wide); inject[t, c] = injW[c] . xn[t]
// (injW null: the output mixer, which has no scatter). One block per token.
extern "C" __global__ void ts_q4e_hc_norm_f32(
    const float* __restrict__ xs,        // [nt, HC, E]
    const float* __restrict__ normW,     // [HC * E]
    const float* __restrict__ injW,      // [HC, HC * E] or null
    float* __restrict__ xn,              // [nt, HC, E]
    float* __restrict__ inject,          // [nt, HC]
    const int E,
    const int HC,
    const float eps)
{
    __shared__ float sh[32];
    __shared__ float inv[Q4E_HC_MAX];
    const int t = blockIdx.x;
    const int flat = HC * E;
    const float* x = xs + (size_t)t * flat;
    float* o = xn + (size_t)t * flat;
    for (int c = 0; c < HC; ++c)
    {
        float ss = 0.0f;
        for (int i = threadIdx.x; i < E; i += blockDim.x)
        {
            const float v = x[c * E + i];
            ss += v * v;
        }
        ss = q4e_block_sum(ss, sh);
        if (threadIdx.x == 0)
            inv[c] = rsqrtf(ss / (float)E + eps);
    }
    __syncthreads();
    float acc[Q4E_HC_MAX] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int j = threadIdx.x; j < flat; j += blockDim.x)
    {
        const float v = x[j] * inv[j / E] * normW[j];
        o[j] = v;
        if (injW != nullptr)
        {
#pragma unroll
            for (int c = 0; c < Q4E_HC_MAX; ++c)
                if (c < HC)
                    acc[c] += v * injW[(size_t)c * flat + j];
        }
    }
    if (injW != nullptr)
    {
        for (int c = 0; c < HC; ++c)
        {
            const float s = q4e_block_sum(acc[c], sh);
            if (threadIdx.x == 0)
                inject[(size_t)t * HC + c] = s;
        }
    }
}

// q8_1 activation blocks, as the dp4a matvecs read them (tensorsharp_kernels.cu's ts_block_q8_1).
struct q4e_block_q8_1
{
    half d;
    half s;
    int8_t qs[32];
};

// One warp's 32 values quantized to q8_1 exactly as ts_quantize_q8_1_rows_warp_f32 rounds them.
__device__ __forceinline__ void q4e_quantize_block(float v, q4e_block_q8_1* dst, int lane)
{
    float amax = fmaxf(0.0f, fabsf(v));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(0xFFFFFFFF, amax, off));
    amax = __shfl_sync(0xFFFFFFFF, amax, 0);
    const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    int q = (int)rintf(v * id);
    q = max(-127, min(127, q));
    dst->qs[lane] = (int8_t)q;
    int sum = q;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
    if (lane == 0)
    {
        dst->d = __float2half_rn(d);
        dst->s = __float2half_rn(d * (float)sum);
    }
}

#define Q4E_HC_SLICE 256
#define Q4E_HC_PARTIAL (1 + Q4E_HC_MAX)

// A slice's partials from its value x at flat index j: the sum of squares and, with injW, the scatter-logit partials
// sum_j x_j * normW_j * injW[c][j], all reduced with one barrier (q4e_block_sums_t0); thread 0 writes them.
__device__ __forceinline__ void q4e_hc_partials_out(const float x, const float* __restrict__ normW,
    const float* __restrict__ injW, const int j, const int flat, const int HC, float (*sh)[32], float* __restrict__ out)
{
    float part[Q4E_HC_PARTIAL];
    part[0] = x * x;
    const float xw = injW != nullptr ? x * normW[j] : 0.0f;
#pragma unroll
    for (int c = 0; c < Q4E_HC_MAX; ++c)
        part[1 + c] = injW != nullptr && c < HC ? xw * injW[(size_t)c * flat + j] : 0.0f;
    q4e_block_sums_t0(part, sh);
    if (threadIdx.x == 0)
    {
        out[0] = part[0];
        if (injW != nullptr)
            for (int c = 0; c < HC; ++c)
                out[1 + c] = part[1 + c];
    }
}

// The decode-size mixer in two wide passes instead of one block per row. Pass 1, per (row, slice of
// 256 values of one stream): the slice's sum of squares and its scatter-logit partials
// sum_j x_j * normW_j * injW[c][j]. grid (HC * E / 256, rows), 256 threads.
extern "C" __global__ void ts_q4e_hc_partials_f32(
    const float* __restrict__ xs,        // [rows, HC, E]
    const float* __restrict__ normW,     // [HC * E]
    const float* __restrict__ injW,      // [HC, HC * E] or null
    float* __restrict__ partials,        // [rows, slices, Q4E_HC_PARTIAL]
    const int E,
    const int HC)
{
    __shared__ float sh[Q4E_HC_PARTIAL][32];
    const int slice = blockIdx.x;
    const int t = blockIdx.y;
    const int flat = HC * E;
    const int j = slice * Q4E_HC_SLICE + threadIdx.x;
    const float x = xs[(size_t)t * flat + j];
    q4e_hc_partials_out(x, normW, injW, j, flat, HC, sh, partials + ((size_t)t * gridDim.x + slice) * Q4E_HC_PARTIAL);
}

// One half's scatter (ts_q4e_hc_post_f32: xs[t, c, i] += out[t, i] * 2 * sigmoid(inject[t, c] / HC)) fused with
// the next mixer's first pass (ts_q4e_hc_partials_f32) over the updated streams, in the same arithmetic: a decode
// step's halves meet here with one launch instead of two. grid (HC * E / 256, rows), 256 threads.
extern "C" __global__ void ts_q4e_hc_post_partials_f32(
    float* __restrict__ xs,              // [rows, HC, E]
    const float* __restrict__ out,       // [rows, E]
    const float* __restrict__ inject,    // [rows, HC] this half's scatter logits
    const float* __restrict__ normW,     // [HC * E] the next mixer's
    const float* __restrict__ injW,      // [HC, HC * E] the next mixer's, or null
    float* __restrict__ partials,        // [rows, slices, Q4E_HC_PARTIAL]
    const int E,
    const int HC)
{
    __shared__ float sh[Q4E_HC_PARTIAL][32];
    const int slice = blockIdx.x;
    const int t = blockIdx.y;
    const int flat = HC * E;
    const int j = slice * Q4E_HC_SLICE + threadIdx.x;
    const int c = j / E, i = j - c * E;
    const float o = out[(size_t)t * E + i];
    const float w = q4e_sigmoid(inject[(size_t)t * HC + c] * (1.0f / (float)HC)) * 2.0f;
    float* xp = xs + (size_t)t * flat + j;
    float x = *xp;
    x += o * w;
    *xp = x;
    q4e_hc_partials_out(x, normW, injW, j, flat, HC, sh, partials + ((size_t)t * gridDim.x + slice) * Q4E_HC_PARTIAL);
}

// BF16 values widened to F32 (exact): a prompt's router GEMM over weights kept as the file's BF16.
extern "C" __global__ void ts_q4e_bf16_to_f32(const unsigned short* __restrict__ src, float* __restrict__ dst, const long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = __uint_as_float((unsigned int)src[i] << 16);
}

// Pass 2: xn = x * rsqrt(mean of its stream's squares + eps) * normW, also quantized to q8_1 for the
// bottleneck projection (a warp per 32-value block), and on the first slice's block the scatter
// logits inject[t, c] = sum over streams s of inv_s * (stream s's partials of c).
// grid (HC * E / 256, rows), 256 threads.
extern "C" __global__ void ts_q4e_hc_apply_f32(
    const float* __restrict__ xs,
    const float* __restrict__ normW,
    const float* __restrict__ partials,
    float* __restrict__ xn,              // [rows, HC, E]
    q4e_block_q8_1* __restrict__ xq,     // [rows, HC * E / 32]
    float* __restrict__ inject,          // [rows, HC] (null: the output mixer)
    const int E,
    const int HC,
    const float eps)
{
    __shared__ float inv[Q4E_HC_MAX];
    const int slice = blockIdx.x;
    const int t = blockIdx.y;
    const int flat = HC * E;
    const int per = E / Q4E_HC_SLICE;
    const float* p = partials + (size_t)t * gridDim.x * Q4E_HC_PARTIAL;
    if (threadIdx.x < HC)
    {
        float ss = 0.0f;
        for (int s = 0; s < per; ++s)
            ss += p[((size_t)threadIdx.x * per + s) * Q4E_HC_PARTIAL];
        inv[threadIdx.x] = rsqrtf(ss / (float)E + eps);
    }
    __syncthreads();
    const int j = slice * Q4E_HC_SLICE + threadIdx.x;
    const float v = xs[(size_t)t * flat + j] * inv[j / E] * normW[j];
    xn[(size_t)t * flat + j] = v;
    q4e_quantize_block(v, xq + (size_t)t * (flat / 32) + j / 32, threadIdx.x & 31);
    if (inject != nullptr && slice == 0 && threadIdx.x < HC)
    {
        float acc = 0.0f;
        for (int c = 0; c < HC; ++c)
        {
            float pc = 0.0f;
            for (int s = 0; s < per; ++s)
                pc += p[((size_t)c * per + s) * Q4E_HC_PARTIAL + 1 + threadIdx.x];
            acc += inv[c] * pc;
        }
        inject[(size_t)t * HC + threadIdx.x] = acc;
    }
}

// x = silu(x * scale) for whole 32-value blocks, written back and quantized to q8_1 for the next
// projection. One warp per block.
extern "C" __global__ void ts_q4e_silu_q81_f32(
    float* __restrict__ x,
    q4e_block_q8_1* __restrict__ xq,
    const long long blocks,
    const float scale)
{
    const int lane = threadIdx.x & 31;
    const long long b = (((long long)blockIdx.x * blockDim.x) + threadIdx.x) >> 5;
    if (b >= blocks)
        return;
    const float v = q4e_silu(x[b * 32 + lane] * scale);
    x[b * 32 + lane] = v;
    q4e_quantize_block(v, xq + b, lane);
}

// x = silu(x * scale) over n values: the bottleneck activation.
extern "C" __global__ void ts_q4e_silu_scale_f32(float* __restrict__ x, const long long n, const float scale)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = q4e_silu(x[i] * scale);
}

// cur[t, i] = mean over the streams of xn[t, c, i] * sigmoid(g[t, c, i]), summed in stream order.
// grid (ceil(E / 256), nt).
extern "C" __global__ void ts_q4e_hc_collapse_f32(
    const float* __restrict__ xn,        // [nt, HC, E]
    const float* __restrict__ g,         // [nt, HC, E] pre-sigmoid
    float* __restrict__ cur,             // [nt, E]
    const int E,
    const int HC)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= E)
        return;
    const float* xr = xn + (size_t)t * HC * E;
    const float* gr = g + (size_t)t * HC * E;
    float m = xr[i] * q4e_sigmoid(gr[i]);
    for (int c = 1; c < HC; ++c)
        m += xr[c * E + i] * q4e_sigmoid(gr[c * E + i]);
    cur[(size_t)t * E + i] = m * (1.0f / (float)HC);
}

// ts_q4e_hc_collapse_f32 that also quantizes cur to q8_1 for the projections that read it (as
// ts_quantize_q8_1_rows_warp_f32 rounds it, a warp per 32-value block): one launch instead of two. E % 32 == 0, so a
// warp is wholly inside or past the row. grid (ceil(E / 256), nt).
extern "C" __global__ void ts_q4e_hc_collapse_q81_f32(
    const float* __restrict__ xn,        // [nt, HC, E]
    const float* __restrict__ g,         // [nt, HC, E] pre-sigmoid
    float* __restrict__ cur,             // [nt, E]
    q4e_block_q8_1* __restrict__ xq,     // [nt, E / 32]
    const int E,
    const int HC)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= E)
        return;
    const float* xr = xn + (size_t)t * HC * E;
    const float* gr = g + (size_t)t * HC * E;
    float m = xr[i] * q4e_sigmoid(gr[i]);
    for (int c = 1; c < HC; ++c)
        m += xr[c * E + i] * q4e_sigmoid(gr[c * E + i]);
    const float v = m * (1.0f / (float)HC);
    cur[(size_t)t * E + i] = v;
    q4e_quantize_block(v, xq + (size_t)t * (E / 32) + i / 32, threadIdx.x & 31);
}

// xs[t, c, i] += out[t, i] * 2 * sigmoid(inject[t, c] / HC). grid (ceil(E / 256), nt).
extern "C" __global__ void ts_q4e_hc_post_f32(
    float* __restrict__ xs,              // [nt, HC, E]
    const float* __restrict__ out,       // [nt, E]
    const float* __restrict__ inject,    // [nt, HC]
    const int E,
    const int HC)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= E)
        return;
    const float o = out[(size_t)t * E + i];
    for (int c = 0; c < HC; ++c)
    {
        const float w = q4e_sigmoid(inject[(size_t)t * HC + c] * (1.0f / (float)HC)) * 2.0f;
        xs[((size_t)t * HC + c) * E + i] += o * w;
    }
}

// ---------------------------------------------------------------------------
// Gated DeltaNet front end
// ---------------------------------------------------------------------------

// Head width of the recurrence (the GLM scan's TS_GLM_KDA_HD) and its per-(row, head) scratch:
// q | k | v | log decay (128 each), then beta (TS_GLM_KDA_SCR).
#define Q4E_GDN_HD 128
#define Q4E_GDN_SCR (4 * Q4E_GDN_HD + 1)

// Channel `ch` of the causal convolution at row `local` of its sequence, before the SiLU: taps
// 0..dc-2 reach back into the previous rows (the state holds the dc-1 inputs before this
// ubatch, oldest first), the last tap is the row itself (ggml_ssm_conv's order).
__device__ __forceinline__ float q4e_conv(
    const float* __restrict__ x, const float* __restrict__ state, const float* __restrict__ w,
    int rowBase, int local, int ch, int C, int dc)
{
    float acc = 0.0f;
    for (int ki = 0; ki < dc; ++ki)
    {
        const int src = local - (dc - 1 - ki);
        const float v = src >= 0 ? x[(size_t)(rowBase + src) * C + ch] : state[(size_t)(dc - 1 + src) * C + ch];
        acc += v * w[(size_t)ch * dc + ki];
    }
    return acc;
}

// One warp per (row, V head): the convolution + SiLU of the head's q, k (from K head h % HK, the
// tiling ggml_repeat gives) and v channels, the L2 norms of q and k (ggml-cuda's
// rsqrt(max(sum, eps^2))), the scalar log decay softplus(alpha + dt_bias) * ssm_a (ssm_a ships
// negated) written for every channel, and beta = sigmoid.
extern "C" __global__ void ts_q4e_gdn_prep_f32(
    const float* __restrict__ qkv,                // [rows, C], C = 2 * HK * 128 + HV * 128
    const float* const* __restrict__ convStates,  // [nSeq] -> [dc-1, C]
    const float* __restrict__ convW,              // [C, dc]
    const float* __restrict__ alpha,              // [rows, HV]
    const float* __restrict__ beta,               // [rows, HV]
    const float* __restrict__ dtBias,             // [HV]
    const float* __restrict__ ssmA,               // [HV]
    float* __restrict__ scr,                      // [nSeq, HV, rowsPerSeq, Q4E_GDN_SCR]
    const int rowsPerSeq,
    const int nSeq,
    const int HK,
    const int HV,
    const int dc,
    const float eps)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rowsPerSeq * nSeq * HV)
        return;
    const int h = warp % HV;
    const int row = warp / HV;
    const int seq = row / rowsPerSeq;
    const int local = row - seq * rowsPerSeq;
    const int rowBase = seq * rowsPerSeq;
    const int keyDim = HK * Q4E_GDN_HD;
    const int C = 2 * keyDim + HV * Q4E_GDN_HD;
    const int kh = h % HK;
    const float* state = convStates[seq];
    float* dst = scr + (((size_t)seq * HV + h) * rowsPerSeq + local) * Q4E_GDN_SCR;

    float qv[4], kv[4];
    float qs = 0.0f, ks = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int d = lane + 32 * j;
        qv[j] = q4e_silu(q4e_conv(qkv, state, convW, rowBase, local, kh * Q4E_GDN_HD + d, C, dc));
        kv[j] = q4e_silu(q4e_conv(qkv, state, convW, rowBase, local, keyDim + kh * Q4E_GDN_HD + d, C, dc));
        qs += qv[j] * qv[j];
        ks += kv[j] * kv[j];
    }
    const float qScale = rsqrtf(fmaxf(q4e_warp_sum(qs), eps * eps));
    const float kScale = rsqrtf(fmaxf(q4e_warp_sum(ks), eps * eps));
    const float g = q4e_softplus(alpha[(size_t)row * HV + h] + dtBias[h]) * ssmA[h];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int d = lane + 32 * j;
        dst[d] = qv[j] * qScale;
        dst[Q4E_GDN_HD + d] = kv[j] * kScale;
        dst[2 * Q4E_GDN_HD + d] = q4e_silu(q4e_conv(qkv, state, convW, rowBase, local, 2 * keyDim + h * Q4E_GDN_HD + d, C, dc));
        dst[3 * Q4E_GDN_HD + d] = g;
    }
    if (lane == 0)
        dst[4 * Q4E_GDN_HD] = q4e_sigmoid(beta[(size_t)row * HV + h]);
}

// Carry each sequence's last dc-1 convolution inputs into its state, oldest first. One thread per
// (sequence, channel), walking the state rows upward so a row still to be read is never
// overwritten first.
extern "C" __global__ void ts_q4e_conv_update_f32(
    const float* __restrict__ x,           // [nSeq * rowsPerSeq, C]
    float* const* __restrict__ states,     // [nSeq] -> [hist, C]
    const int rowsPerSeq,
    const int nSeq,
    const int C,
    const int hist)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)nSeq * C)
        return;
    const int seq = (int)(i / C);
    const int ch = (int)(i - (long long)seq * C);
    float* st = states[seq];
    for (int r = 0; r < hist; ++r)
    {
        const int src = rowsPerSeq - hist + r;
        st[(size_t)r * C + ch] = src >= 0
            ? x[((size_t)seq * rowsPerSeq + src) * C + ch]
            : st[(size_t)(hist + src) * C + ch];
    }
}

// ---------------------------------------------------------------------------
// gated GQA attention
// ---------------------------------------------------------------------------

// Head width of the attention kernels (the model's attention.key_length).
#define Q4E_ATTN_HD 256

// KV cache formats. F16: [nCtx, NKV, HD] halves. Q8_0 (--kv-cache-dtype q8_0): ggml's Q8_0 arithmetic - per 32
// values d = amax / 127 and q = round(x / d) - with the codes and the scales split, so a lane reads its 8 codes in one
// load: int8 codes [nCtx, NKV, HD] and F16 scales [nCtx, NKV, HD / 32]. Half the bytes of F16 (+3% for the scales).
#define Q4E_KV_Q8_BLOCK 32

// One warp holds one 32-value block (threads 32w .. 32w + 31 of the head's 256).
// KV streaming (TS_Q4E_KV_RESIDENT): a streamed layer keeps its authoritative Q8_0 K/V in pinned, device-mapped host
// memory in the usual [nCtx, NKV, HD] layout and only a window of pages in VRAM. A page is Q4E_KV_PAGE consecutive cells
// (all KV heads); the page table maps a page to the VRAM slot holding it (-1: not resident). Readers go through the
// table (PAGED); ts_q4e_kv_resolve makes every page a call reads resident before it reads them.
#define Q4E_KV_PAGE 4

// The VRAM row holding cell c: the identity, or through the page table.
template <bool PAGED>
__device__ __forceinline__ int q4e_kv_row(const int32_t* __restrict__ table, const int c)
{
    if constexpr (PAGED)
        return table[c / Q4E_KV_PAGE] * Q4E_KV_PAGE + c % Q4E_KV_PAGE;
    else
        return c;
}

// q4e_store_kv_q8's value at two places: its VRAM row (vramOk) and the host copy.
__device__ __forceinline__ void q4e_store_kv_q8_2(signed char* __restrict__ codes, half* __restrict__ scales,
    const size_t at, const bool vramOk, signed char* __restrict__ hostCodes, half* __restrict__ hostScales,
    const size_t hostAt, const float x)
{
    float amax = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float dq = amax / 127.0f;
    const float id = dq != 0.0f ? 1.0f / dq : 0.0f;
    const signed char q = (signed char)(int)roundf(x * id);
    const half hs = __float2half(dq);
    if (vramOk)
    {
        codes[at] = q;
        if ((threadIdx.x & 31) == 0)
            scales[at / Q4E_KV_Q8_BLOCK] = hs;
    }
    hostCodes[hostAt] = q;
    if ((threadIdx.x & 31) == 0)
        hostScales[hostAt / Q4E_KV_Q8_BLOCK] = hs;
}

__device__ __forceinline__ void q4e_store_kv_q8(signed char* __restrict__ codes, half* __restrict__ scales,
    const size_t at, const float x)
{
    float amax = fabsf(x);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float dq = amax / 127.0f;
    const float id = dq != 0.0f ? 1.0f / dq : 0.0f;
    codes[at] = (signed char)(int)roundf(x * id);
    if ((threadIdx.x & 31) == 0)
        scales[at / Q4E_KV_Q8_BLOCK] = __float2half(dq);
}

// One block per (head, row): the query heads' RMS norm and rotary, their gates split out of the
// interleaved q|gate projection; the KV heads' norm and rotary, stored with the values into the
// caches at the row's position. Rows sit at p0, p0 + 1, ... or at positions[row].
// grid (NH + NKV, nt), Q4E_ATTN_HD threads.
template <bool Q8, bool PAGED = false>
__device__ __forceinline__ void q4e_attn_prep(
    const float* __restrict__ qg,        // [nt, NH, 2, HD]
    const float* __restrict__ k,         // [nt, NKV, HD]
    const float* __restrict__ v,         // [nt, NKV, HD]
    const float* __restrict__ qNorm,     // [HD]
    const float* __restrict__ kNorm,     // [HD]
    float* __restrict__ q,               // [nt, NH, HD]
    float* __restrict__ gate,            // [nt, NH, HD]
    void* __restrict__ kCache,           // [nCtx, NKV, HD] F16 or int8 codes
    void* __restrict__ vCache,
    half* __restrict__ kScale,           // Q8_0: [nCtx, NKV, HD / 32]
    half* __restrict__ vScale,
    const int32_t* __restrict__ positions,
    const int p0,
    const int NH,
    const int NKV,
    const int nRot,
    const float base,
    const float freqScale,
    const float eps,
    const int32_t* __restrict__ table = nullptr,     // PAGED: page -> VRAM slot, -1 not resident
    signed char* __restrict__ hostK = nullptr,       // PAGED: the host copies, [nCtx, NKV, HD] and their scales
    signed char* __restrict__ hostV = nullptr,
    half* __restrict__ hostKScale = nullptr,
    half* __restrict__ hostVScale = nullptr)
{
    __shared__ float sh[32];
    __shared__ float row[Q4E_ATTN_HD];
    const int head = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    const int pos = positions != nullptr ? positions[t] : p0 + t;
    const bool isQ = head < NH;
    const int kvh = head - NH;
    float x;
    if (isQ)
    {
        const float* src = qg + ((size_t)t * NH + head) * 2 * Q4E_ATTN_HD;
        x = src[d];
        gate[((size_t)t * NH + head) * Q4E_ATTN_HD + d] = src[Q4E_ATTN_HD + d];
    }
    else
    {
        x = k[((size_t)t * NKV + kvh) * Q4E_ATTN_HD + d];
    }
    const float ss = q4e_block_sum(x * x, sh);
    x = x * rsqrtf(ss / (float)Q4E_ATTN_HD + eps) * (isQ ? qNorm[d] : kNorm[d]);
    row[d] = x;
    __syncthreads();
    if (d < nRot)
    {
        const int half = nRot / 2;
        const int i = d < half ? d : d - half;
        float c, s;
        q4e_rope_angle(pos, i, nRot, base, freqScale, &c, &s);
        x = d < half ? row[d] * c - row[d + half] * s : row[d - half] * s + row[d] * c;
    }
    if (isQ)
    {
        q[((size_t)t * NH + head) * Q4E_ATTN_HD + d] = x;
    }
    else if constexpr (PAGED)
    {
        // The host copy always; the VRAM slot when the cell's page is resident (a page that is not gets the cell from
        // the host copy when a read makes it resident).
        const int slot = table[pos / Q4E_KV_PAGE];
        const size_t hostAt = ((size_t)pos * NKV + kvh) * Q4E_ATTN_HD + d;
        const size_t at = ((size_t)(slot < 0 ? 0 : slot * Q4E_KV_PAGE + pos % Q4E_KV_PAGE) * NKV + kvh) * Q4E_ATTN_HD + d;
        const float vx = v[((size_t)t * NKV + kvh) * Q4E_ATTN_HD + d];
        q4e_store_kv_q8_2(static_cast<signed char*>(kCache), kScale, at, slot >= 0, hostK, hostKScale, hostAt, x);
        q4e_store_kv_q8_2(static_cast<signed char*>(vCache), vScale, at, slot >= 0, hostV, hostVScale, hostAt, vx);
    }
    else
    {
        const size_t at = ((size_t)pos * NKV + kvh) * Q4E_ATTN_HD + d;
        const float vx = v[((size_t)t * NKV + kvh) * Q4E_ATTN_HD + d];
        if constexpr (Q8)
        {
            q4e_store_kv_q8(static_cast<signed char*>(kCache), kScale, at, x);
            q4e_store_kv_q8(static_cast<signed char*>(vCache), vScale, at, vx);
        }
        else
        {
            static_cast<half*>(kCache)[at] = __float2half(x);
            static_cast<half*>(vCache)[at] = __float2half(vx);
        }
    }
}

// The F16 kernel keeps its standalone body (see ts_q4e_attention_f32); the template serves the Q8_0 cache.
extern "C" __global__ void ts_q4e_attn_prep_f32(
    const float* __restrict__ qg, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ qNorm, const float* __restrict__ kNorm, float* __restrict__ q, float* __restrict__ gate,
    half* __restrict__ kCache, half* __restrict__ vCache, const int32_t* __restrict__ positions,
    const int p0, const int NH, const int NKV, const int nRot, const float base, const float freqScale, const float eps)
{
    __shared__ float sh[32];
    __shared__ float row[Q4E_ATTN_HD];
    const int head = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    const int pos = positions != nullptr ? positions[t] : p0 + t;
    const bool isQ = head < NH;
    const int kvh = head - NH;
    float x;
    if (isQ)
    {
        const float* src = qg + ((size_t)t * NH + head) * 2 * Q4E_ATTN_HD;
        x = src[d];
        gate[((size_t)t * NH + head) * Q4E_ATTN_HD + d] = src[Q4E_ATTN_HD + d];
    }
    else
    {
        x = k[((size_t)t * NKV + kvh) * Q4E_ATTN_HD + d];
    }
    const float ss = q4e_block_sum(x * x, sh);
    x = x * rsqrtf(ss / (float)Q4E_ATTN_HD + eps) * (isQ ? qNorm[d] : kNorm[d]);
    row[d] = x;
    __syncthreads();
    if (d < nRot)
    {
        const int half = nRot / 2;
        const int i = d < half ? d : d - half;
        float c, s;
        q4e_rope_angle(pos, i, nRot, base, freqScale, &c, &s);
        x = d < half ? row[d] * c - row[d + half] * s : row[d - half] * s + row[d] * c;
    }
    if (isQ)
    {
        q[((size_t)t * NH + head) * Q4E_ATTN_HD + d] = x;
    }
    else
    {
        const size_t at = ((size_t)pos * NKV + kvh) * Q4E_ATTN_HD + d;
        kCache[at] = __float2half(x);
        vCache[at] = __float2half(v[((size_t)t * NKV + kvh) * Q4E_ATTN_HD + d]);
    }
}

extern "C" __global__ void ts_q4e_attn_prep_q8_f32(
    const float* __restrict__ qg, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ qNorm, const float* __restrict__ kNorm, float* __restrict__ q, float* __restrict__ gate,
    signed char* __restrict__ kCache, signed char* __restrict__ vCache, half* __restrict__ kScale, half* __restrict__ vScale,
    const int32_t* __restrict__ positions,
    const int p0, const int NH, const int NKV, const int nRot, const float base, const float freqScale, const float eps)
{
    q4e_attn_prep<true>(qg, k, v, qNorm, kNorm, q, gate, kCache, vCache, kScale, vScale, positions,
        p0, NH, NKV, nRot, base, freqScale, eps);
}

// ts_q4e_attn_prep_q8_f32 for a streamed layer: K/V into the host copy, and into the VRAM slot of a resident page.
extern "C" __global__ void ts_q4e_attn_prep_q8_paged_f32(
    const float* __restrict__ qg, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ qNorm, const float* __restrict__ kNorm, float* __restrict__ q, float* __restrict__ gate,
    signed char* __restrict__ kCache, signed char* __restrict__ vCache, half* __restrict__ kScale, half* __restrict__ vScale,
    const int32_t* __restrict__ positions,
    const int p0, const int NH, const int NKV, const int nRot, const float base, const float freqScale, const float eps,
    const int32_t* __restrict__ table, signed char* __restrict__ hostK, signed char* __restrict__ hostV,
    half* __restrict__ hostKScale, half* __restrict__ hostVScale)
{
    q4e_attn_prep<true, true>(qg, k, v, qNorm, kNorm, q, gate, kCache, vCache, kScale, vScale, positions,
        p0, NH, NKV, nRot, base, freqScale, eps, table, hostK, hostV, hostKScale, hostVScale);
}

// One lane's 8 consecutive values of a cached K or V row (at = its first element).
template <bool Q8>
__device__ __forceinline__ void q4e_load_kv8(const void* __restrict__ cache, const half* __restrict__ scale,
    const size_t at, float* f)
{
    if constexpr (Q8)
    {
        const int2 w = *reinterpret_cast<const int2*>(static_cast<const signed char*>(cache) + at);
        const signed char* c = reinterpret_cast<const signed char*>(&w);
        const float s = __half2float(scale[at / Q4E_KV_Q8_BLOCK]);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            f[j] = s * (float)c[j];
    }
    else
    {
        const int4 w = *reinterpret_cast<const int4*>(static_cast<const half*>(cache) + at);
        const half2* h = reinterpret_cast<const half2*>(&w);
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const float2 a = __half22float2(h[j]);
            f[2 * j] = a.x; f[2 * j + 1] = a.y;
        }
    }
}

// Online-softmax attention of one query head over its KV head's cells, then the sigmoid gate.
// The cells are 0..p (cellCnt null or cellCnt[t] < 0) or the cellCnt[t] listed in cells.
// grid (NH, nt), 256 threads = 8 warps splitting the cells; lane owns 8 of the 256 dims.
template <bool Q8, bool PAGED = false>
__device__ __forceinline__ void q4e_attention(
    const float* __restrict__ q,         // [nt, NH, HD]
    const void* __restrict__ kCache,     // [nCtx, NKV, HD] F16 or int8 codes
    const void* __restrict__ vCache,
    const half* __restrict__ kScale,     // Q8_0: [nCtx, NKV, HD / 32]
    const half* __restrict__ vScale,
    const float* __restrict__ gate,      // [nt, NH, HD]
    const int32_t* __restrict__ cells,   // [nt, cellStride] or null
    const int32_t* __restrict__ cellCnt, // [nt] or null
    float* __restrict__ out,             // [nt, NH, HD]
    const int32_t* __restrict__ positions,
    const int p0,
    const int NH,
    const int NKV,
    const float scale,
    const int cellStride,
    const int32_t* __restrict__ table = nullptr)   // PAGED: page -> VRAM slot
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int kvh = h / (NH / NKV);
    const int p = positions != nullptr ? positions[t] : p0 + t;
    const int32_t* list = nullptr;
    int n = p + 1;
    if (cells != nullptr && cellCnt[t] >= 0)
    {
        n = cellCnt[t];
        list = cells + (size_t)t * cellStride;
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const float* qRow = q + ((size_t)t * NH + h) * Q4E_ATTN_HD + lane * 8;
    float qReg[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        qReg[i] = qRow[i];

    float m = -INFINITY, sum = 0.0f, acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        acc[i] = 0.0f;
    for (int i = warp; i < n; i += 8)
    {
        const int c = q4e_kv_row<PAGED>(table, list != nullptr ? list[i] : i);
        const size_t at = ((size_t)c * NKV + kvh) * Q4E_ATTN_HD + lane * 8;
        float kf[8], vf[8];
        q4e_load_kv8<Q8>(kCache, kScale, at, kf);
        q4e_load_kv8<Q8>(vCache, vScale, at, vf);
        float part = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            part += qReg[j] * kf[j];
        const float s = q4e_warp_sum(part) * scale;
        if (s > m)
        {
            const float r = expf(m - s);
            sum *= r;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[j] *= r;
            m = s;
        }
        const float w = expf(s - m);
        sum += w;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[j] += w * vf[j];
    }

    __shared__ float shAcc[8][Q4E_ATTN_HD];
    __shared__ float shM[8], shSum[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
        shAcc[warp][lane * 8 + j] = acc[j];
    if (lane == 0)
    {
        shM[warp] = m;
        shSum[warp] = sum;
    }
    __syncthreads();
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < 8; ++w)
        M = fmaxf(M, shM[w]);
    float S = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w)
        S += shSum[w] * (shM[w] == -INFINITY ? 0.0f : expf(shM[w] - M));
    const float invS = 1.0f / S;
    const size_t off = ((size_t)t * NH + h) * Q4E_ATTN_HD;
    for (int d = threadIdx.x; d < Q4E_ATTN_HD; d += blockDim.x)
    {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w)
            v += shM[w] == -INFINITY ? 0.0f : shAcc[w][d] * expf(shM[w] - M);
        out[off + d] = v * invS * q4e_sigmoid(gate[off + d]);
    }
}

// The F16 kernel is kept as its own standalone body (not the template above): built from the template, its
// rounding moved enough to flip a near-tie route in the adaptive-residency drift test.
extern "C" __global__ void ts_q4e_attention_f32(
    const float* __restrict__ q,         // [nt, NH, HD]
    const half* __restrict__ kCache,     // [nCtx, NKV, HD]
    const half* __restrict__ vCache,
    const float* __restrict__ gate,      // [nt, NH, HD]
    const int32_t* __restrict__ cells,   // [nt, cellStride] or null
    const int32_t* __restrict__ cellCnt, // [nt] or null
    float* __restrict__ out,             // [nt, NH, HD]
    const int32_t* __restrict__ positions,
    const int p0,
    const int NH,
    const int NKV,
    const float scale,
    const int cellStride)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int kvh = h / (NH / NKV);
    const int p = positions != nullptr ? positions[t] : p0 + t;
    const int32_t* list = nullptr;
    int n = p + 1;
    if (cells != nullptr && cellCnt[t] >= 0)
    {
        n = cellCnt[t];
        list = cells + (size_t)t * cellStride;
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const float* qRow = q + ((size_t)t * NH + h) * Q4E_ATTN_HD + lane * 8;
    float qReg[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        qReg[i] = qRow[i];

    float m = -INFINITY, sum = 0.0f, acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        acc[i] = 0.0f;
    for (int i = warp; i < n; i += 8)
    {
        const int c = list != nullptr ? list[i] : i;
        const size_t at = ((size_t)c * NKV + kvh) * Q4E_ATTN_HD + lane * 8;
        const int4 kw = *reinterpret_cast<const int4*>(kCache + at);
        const int4 vw = *reinterpret_cast<const int4*>(vCache + at);
        const half2* kh = reinterpret_cast<const half2*>(&kw);
        const half2* vh = reinterpret_cast<const half2*>(&vw);
        float kf[8], vf[8];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const float2 a = __half22float2(kh[j]);
            const float2 b = __half22float2(vh[j]);
            kf[2 * j] = a.x; kf[2 * j + 1] = a.y;
            vf[2 * j] = b.x; vf[2 * j + 1] = b.y;
        }
        float part = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            part += qReg[j] * kf[j];
        const float s = q4e_warp_sum(part) * scale;
        if (s > m)
        {
            const float r = expf(m - s);
            sum *= r;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[j] *= r;
            m = s;
        }
        const float w = expf(s - m);
        sum += w;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[j] += w * vf[j];
    }

    __shared__ float shAcc[8][Q4E_ATTN_HD];
    __shared__ float shM[8], shSum[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
        shAcc[warp][lane * 8 + j] = acc[j];
    if (lane == 0)
    {
        shM[warp] = m;
        shSum[warp] = sum;
    }
    __syncthreads();
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < 8; ++w)
        M = fmaxf(M, shM[w]);
    float S = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w)
        S += shSum[w] * (shM[w] == -INFINITY ? 0.0f : expf(shM[w] - M));
    const float invS = 1.0f / S;
    const size_t off = ((size_t)t * NH + h) * Q4E_ATTN_HD;
    for (int d = threadIdx.x; d < Q4E_ATTN_HD; d += blockDim.x)
    {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w)
            v += shM[w] == -INFINITY ? 0.0f : shAcc[w][d] * expf(shM[w] - M);
        out[off + d] = v * invS * q4e_sigmoid(gate[off + d]);
    }
}

extern "C" __global__ void ts_q4e_attention_q8_f32(
    const float* __restrict__ q, const signed char* __restrict__ kCache, const signed char* __restrict__ vCache,
    const half* __restrict__ kScale, const half* __restrict__ vScale,
    const float* __restrict__ gate, const int32_t* __restrict__ cells, const int32_t* __restrict__ cellCnt,
    float* __restrict__ out, const int32_t* __restrict__ positions,
    const int p0, const int NH, const int NKV, const float scale, const int cellStride)
{
    q4e_attention<true>(q, kCache, vCache, kScale, vScale, gate, cells, cellCnt, out, positions,
        p0, NH, NKV, scale, cellStride);
}

// ts_q4e_attention_q8_f32 of a streamed layer: every cell read through the page table.
extern "C" __global__ void ts_q4e_attention_q8_paged_f32(
    const float* __restrict__ q, const signed char* __restrict__ kCache, const signed char* __restrict__ vCache,
    const half* __restrict__ kScale, const half* __restrict__ vScale,
    const float* __restrict__ gate, const int32_t* __restrict__ cells, const int32_t* __restrict__ cellCnt,
    float* __restrict__ out, const int32_t* __restrict__ positions,
    const int p0, const int NH, const int NKV, const float scale, const int cellStride, const int32_t* __restrict__ table)
{
    q4e_attention<true, true>(q, kCache, vCache, kScale, vScale, gate, cells, cellCnt, out, positions,
        p0, NH, NKV, scale, cellStride, table);
}

// Decode-size attention split over the cells (flash decoding): grid (NH, nt, S). One block per (head, row) leaves
// all but 24 x rows SMs idle and walks up to the QSA width's 2,051 cells (or a dense row's whole context) one
// dependent load after another: 0.2 ms a layer at 4K context, 0.4 ms at 16K, for 4 rows. A row of more than
// Q4E_ATTN_SPLIT_MIN cells instead gives each of the S blocks a contiguous share of them; each block writes its
// (max, sum, unnormalized output) partial, and the block that arrives last merges the partials in split order and
// applies the gate (no second launch; it resets the row's arrival count for the next call). A row of at most
// Q4E_ATTN_SPLIT_MIN cells runs whole in block z = 0 with the arithmetic of q4e_attention (the other blocks exit),
// so short contexts answer bit for bit as before. The split depends on the row's cell count alone, never on how
// many rows the call holds: a verify window's row computes what that row's own decode step computes.
#define Q4E_ATTN_SPLIT_MIN 256
template <bool Q8, bool PAGED = false>
__device__ __forceinline__ void q4e_attention_split(
    const float* __restrict__ q, const void* __restrict__ kCache, const void* __restrict__ vCache,
    const half* __restrict__ kScale, const half* __restrict__ vScale,
    const float* __restrict__ gate, const int32_t* __restrict__ cells, const int32_t* __restrict__ cellCnt,
    float* __restrict__ out, const int32_t* __restrict__ positions,
    const int p0, const int NH, const int NKV, const float scale, const int cellStride,
    float* __restrict__ part,   // [nt, NH, S, 2 + HD]: max, sum, unnormalized output
    int* __restrict__ arrive,   // [nt, NH], zero between calls
    const int32_t* __restrict__ table = nullptr)   // PAGED: page -> VRAM slot
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int z = blockIdx.z;
    const int nSplit = gridDim.z;
    const int kvh = h / (NH / NKV);
    const int p = positions != nullptr ? positions[t] : p0 + t;
    const int32_t* list = nullptr;
    int n = p + 1;
    if (cells != nullptr && cellCnt[t] >= 0)
    {
        n = cellCnt[t];
        list = cells + (size_t)t * cellStride;
    }
    const bool split = n > Q4E_ATTN_SPLIT_MIN && nSplit > 1;
    if (!split && z > 0)
        return;
    int begin = 0, end = n;
    if (split)
    {
        const int chunk = (n + nSplit - 1) / nSplit;
        begin = min(n, z * chunk);
        end = min(n, begin + chunk);
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const float* qRow = q + ((size_t)t * NH + h) * Q4E_ATTN_HD + lane * 8;
    float qReg[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        qReg[i] = qRow[i];

    float m = -INFINITY, sum = 0.0f, acc[8];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        acc[i] = 0.0f;
    for (int i = begin + warp; i < end; i += 8)
    {
        const int c = q4e_kv_row<PAGED>(table, list != nullptr ? list[i] : i);
        const size_t at = ((size_t)c * NKV + kvh) * Q4E_ATTN_HD + lane * 8;
        float kf[8], vf[8];
        q4e_load_kv8<Q8>(kCache, kScale, at, kf);
        q4e_load_kv8<Q8>(vCache, vScale, at, vf);
        float partDot = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            partDot += qReg[j] * kf[j];
        const float s = q4e_warp_sum(partDot) * scale;
        if (s > m)
        {
            const float r = expf(m - s);
            sum *= r;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[j] *= r;
            m = s;
        }
        const float w = expf(s - m);
        sum += w;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc[j] += w * vf[j];
    }

    __shared__ float shAcc[8][Q4E_ATTN_HD];
    __shared__ float shM[8], shSum[8];
    __shared__ int isLast;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        shAcc[warp][lane * 8 + j] = acc[j];
    if (lane == 0)
    {
        shM[warp] = m;
        shSum[warp] = sum;
    }
    __syncthreads();
    float M = -INFINITY;
#pragma unroll
    for (int w = 0; w < 8; ++w)
        M = fmaxf(M, shM[w]);
    float S = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w)
        S += shSum[w] * (shM[w] == -INFINITY ? 0.0f : expf(shM[w] - M));
    const size_t off = ((size_t)t * NH + h) * Q4E_ATTN_HD;
    if (!split)
    {
        const float invS = 1.0f / S;
        for (int d = threadIdx.x; d < Q4E_ATTN_HD; d += blockDim.x)
        {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w)
                v += shM[w] == -INFINITY ? 0.0f : shAcc[w][d] * expf(shM[w] - M);
            out[off + d] = v * invS * q4e_sigmoid(gate[off + d]);
        }
        return;
    }

    const size_t stride = 2 + Q4E_ATTN_HD;
    float* rowPart = part + ((size_t)t * NH + h) * nSplit * stride;
    float* mine = rowPart + (size_t)z * stride;
    for (int d = threadIdx.x; d < Q4E_ATTN_HD; d += blockDim.x)
    {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w)
            v += shM[w] == -INFINITY ? 0.0f : shAcc[w][d] * expf(shM[w] - M);
        mine[2 + d] = v;
    }
    if (threadIdx.x == 0)
    {
        mine[0] = M;
        mine[1] = S;
    }
    // Publish the partial, then count this block in: the last block to arrive merges.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
        isLast = atomicAdd(&arrive[(size_t)t * NH + h], 1) == nSplit - 1;
    __syncthreads();
    if (!isLast)
        return;
    __threadfence();
    float MM = -INFINITY;
    for (int s = 0; s < nSplit; ++s)
        MM = fmaxf(MM, __ldcg(rowPart + (size_t)s * stride));
    float SS = 0.0f;
    for (int s = 0; s < nSplit; ++s)
    {
        const float ms = __ldcg(rowPart + (size_t)s * stride);
        SS += ms == -INFINITY ? 0.0f : __ldcg(rowPart + (size_t)s * stride + 1) * expf(ms - MM);
    }
    const float invS = 1.0f / SS;
    for (int d = threadIdx.x; d < Q4E_ATTN_HD; d += blockDim.x)
    {
        float v = 0.0f;
        for (int s = 0; s < nSplit; ++s)
        {
            const float ms = __ldcg(rowPart + (size_t)s * stride);
            v += ms == -INFINITY ? 0.0f : __ldcg(rowPart + (size_t)s * stride + 2 + d) * expf(ms - MM);
        }
        out[off + d] = v * invS * q4e_sigmoid(gate[off + d]);
    }
    if (threadIdx.x == 0)
        arrive[(size_t)t * NH + h] = 0;
}

extern "C" __global__ void ts_q4e_attention_split_f32(
    const float* __restrict__ q, const half* __restrict__ kCache, const half* __restrict__ vCache,
    const float* __restrict__ gate, const int32_t* __restrict__ cells, const int32_t* __restrict__ cellCnt,
    float* __restrict__ out, const int32_t* __restrict__ positions,
    const int p0, const int NH, const int NKV, const float scale, const int cellStride,
    float* __restrict__ part, int* __restrict__ arrive)
{
    q4e_attention_split<false>(q, kCache, vCache, nullptr, nullptr, gate, cells, cellCnt, out, positions,
        p0, NH, NKV, scale, cellStride, part, arrive);
}

extern "C" __global__ void ts_q4e_attention_split_q8_f32(
    const float* __restrict__ q, const signed char* __restrict__ kCache, const signed char* __restrict__ vCache,
    const half* __restrict__ kScale, const half* __restrict__ vScale,
    const float* __restrict__ gate, const int32_t* __restrict__ cells, const int32_t* __restrict__ cellCnt,
    float* __restrict__ out, const int32_t* __restrict__ positions,
    const int p0, const int NH, const int NKV, const float scale, const int cellStride,
    float* __restrict__ part, int* __restrict__ arrive)
{
    q4e_attention_split<true>(q, kCache, vCache, kScale, vScale, gate, cells, cellCnt, out, positions,
        p0, NH, NKV, scale, cellStride, part, arrive);
}

// ts_q4e_attention_split_q8_f32 of a streamed layer: every cell read through the page table.
extern "C" __global__ void ts_q4e_attention_split_q8_paged_f32(
    const float* __restrict__ q, const signed char* __restrict__ kCache, const signed char* __restrict__ vCache,
    const half* __restrict__ kScale, const half* __restrict__ vScale,
    const float* __restrict__ gate, const int32_t* __restrict__ cells, const int32_t* __restrict__ cellCnt,
    float* __restrict__ out, const int32_t* __restrict__ positions,
    const int p0, const int NH, const int NKV, const float scale, const int cellStride,
    float* __restrict__ part, int* __restrict__ arrive, const int32_t* __restrict__ table)
{
    q4e_attention_split<true, true>(q, kCache, vCache, kScale, vScale, gate, cells, cellCnt, out, positions,
        p0, NH, NKV, scale, cellStride, part, arrive, table);
}

// ---- KV streaming: residency of a streamed layer's pages ----
// Control words of a layer's page map (ctl): [0] the epoch of the last resolve, [1] the eviction hand, [2] the pages
// the last resolve missed (what ts_q4e_kv_page_copy copies), [3] calls that could not place every page (overflow),
// [4] misses and [5] calls, running totals (wrapping).
#define Q4E_KV_CTL 8

// Makes every page the call's rows read resident, inside a captured step: one block of 1,024 threads. A row reads its
// listed cells (cells / cellCnt[t] >= 0: the QSA selection) or cells 0..p. A resident page gets this call's epoch
// (slotStamp), a missing one is claimed once (table -1 -> -2) into the miss list; then each miss takes a slot no page of
// this call uses, sweeping from the hand (round robin), the slot's old page leaving the table. The page data follow in
// ts_q4e_kv_page_copy. The host keeps the window at least one call's pages (TS_Q4E_KV_RESIDENT's floor), so a call never
// runs out of slots; if it did, its unplaced pages would point at slot 0 and ctl[3] would count it.
extern "C" __global__ void __launch_bounds__(1024) ts_q4e_kv_resolve(
    const int32_t* __restrict__ cells,
    const int32_t* __restrict__ cellCnt,
    const int32_t* __restrict__ positions,
    const int p0,
    const int nt,
    const int cellStride,
    int32_t* __restrict__ table,         // [nPages]
    int32_t* __restrict__ slotPage,      // [nSlots]
    int32_t* __restrict__ slotStamp,     // [nSlots]
    int32_t* __restrict__ ctl,           // [Q4E_KV_CTL]
    int32_t* __restrict__ missPage,      // [maxMiss]
    int32_t* __restrict__ missSlot,      // [maxMiss]
    const int nSlots,
    const int maxMiss)
{
    __shared__ int shEpoch, shMiss, shHand, shTaken, shLast;
    __shared__ int warpCnt[32];
    if (threadIdx.x == 0)
    {
        shEpoch = ctl[0] + 1;
        shHand = ctl[1];
        shMiss = 0;
        shTaken = 0;
        shLast = -1;
    }
    __syncthreads();
    const int epoch = shEpoch;
    for (int t = 0; t < nt; ++t)
    {
        const int p = positions != nullptr ? positions[t] : p0 + t;
        const bool listed = cells != nullptr && cellCnt[t] >= 0;
        const int32_t* list = listed ? cells + (size_t)t * cellStride : nullptr;
        const int n = listed ? cellCnt[t] : p / Q4E_KV_PAGE + 1;   // listed: cells; dense: pages 0..p
        for (int i = threadIdx.x; i < n; i += blockDim.x)
        {
            const int pg = listed ? list[i] / Q4E_KV_PAGE : i;
            const int s = ((volatile int32_t*)table)[pg];
            if (s >= 0)
                slotStamp[s] = epoch;
            else if (s == -1 && atomicCAS(&table[pg], -1, -2) == -1)
            {
                const int k = atomicAdd(&shMiss, 1);
                if (k < maxMiss)
                    missPage[k] = pg;
            }
        }
    }
    __syncthreads();
    const int nMiss = min(shMiss, maxMiss);
    // Victims: slots this call does not use, in round-robin order from the hand; a block-wide ballot scan assigns them
    // to the misses in order.
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    const int hand = shHand;
    for (int sweep = 0; sweep < nSlots && shTaken < nMiss; sweep += blockDim.x)
    {
        const int off = sweep + threadIdx.x;
        const int s = (hand + off) % nSlots;
        const bool elig = off < nSlots && slotStamp[s] != epoch;
        const unsigned int ballot = __ballot_sync(0xffffffffu, elig);
        if (lane == 0)
            warpCnt[warp] = __popc(ballot);
        __syncthreads();
        int rank = shTaken + __popc(ballot & ((1u << lane) - 1u));
        for (int w = 0; w < warp; ++w)
            rank += warpCnt[w];
        if (elig && rank < nMiss)
        {
            const int pg = missPage[rank];
            const int old = slotPage[s];
            if (old >= 0)
                table[old] = -1;
            slotPage[s] = pg;
            slotStamp[s] = epoch;
            table[pg] = s;
            missSlot[rank] = s;
            if (rank == nMiss - 1)
                shLast = off;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            int total = 0;
            for (int w = 0; w < warps; ++w)
                total += warpCnt[w];
            shTaken += total;
        }
        __syncthreads();
    }
    // Pages no slot took (never, with the window's floor): read slot 0 rather than out of bounds, and counted.
    const int placed = min(shTaken, nMiss);
    for (int k = placed + threadIdx.x; k < nMiss; k += blockDim.x)
    {
        table[missPage[k]] = 0;
        missSlot[k] = -1;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        ctl[0] = epoch;
        if (shLast >= 0)
            ctl[1] = (hand + shLast + 1) % nSlots;
        ctl[2] = nMiss;
        if (placed < nMiss || shMiss > maxMiss)
            ctl[3] += 1;
        ctl[4] += shMiss;
        ctl[5] += 1;
    }
}

// The pages the last resolve placed (ctl[2] of them), copied from the host copy into their VRAM slots: Q8_0 codes and
// scales of K and V, all KV heads of the page's cells. Any grid; 16-byte units (a page's codes are 4 x NKV x 256 bytes,
// its scales 4 x NKV x 8 halves).
extern "C" __global__ void ts_q4e_kv_page_copy(
    const int32_t* __restrict__ ctl,
    const int32_t* __restrict__ missPage,
    const int32_t* __restrict__ missSlot,
    const signed char* __restrict__ hostK,
    const signed char* __restrict__ hostV,
    const half* __restrict__ hostKScale,
    const half* __restrict__ hostVScale,
    signed char* __restrict__ kCache,
    signed char* __restrict__ vCache,
    half* __restrict__ kScale,
    half* __restrict__ vScale,
    const int NKV)
{
    const int n = ctl[2];
    const int codes16 = Q4E_KV_PAGE * NKV * Q4E_ATTN_HD / 16;
    const int scales16 = Q4E_KV_PAGE * NKV * (Q4E_ATTN_HD / Q4E_KV_Q8_BLOCK) * 2 / 16;
    const int per = 2 * codes16 + 2 * scales16;
    for (int m = blockIdx.x; m < n; m += gridDim.x)
    {
        const int slot = missSlot[m];
        if (slot < 0)
            continue;
        const size_t src = (size_t)missPage[m], dst = (size_t)slot;
        for (int i = threadIdx.x; i < per; i += blockDim.x)
        {
            const int4* from;
            int4* to;
            int j = i;
            if (j < codes16)
            {
                from = reinterpret_cast<const int4*>(hostK) + src * codes16 + j;
                to = reinterpret_cast<int4*>(kCache) + dst * codes16 + j;
            }
            else if ((j -= codes16) < codes16)
            {
                from = reinterpret_cast<const int4*>(hostV) + src * codes16 + j;
                to = reinterpret_cast<int4*>(vCache) + dst * codes16 + j;
            }
            else if ((j -= codes16) < scales16)
            {
                from = reinterpret_cast<const int4*>(hostKScale) + src * scales16 + j;
                to = reinterpret_cast<int4*>(kScale) + dst * scales16 + j;
            }
            else
            {
                j -= scales16;
                from = reinterpret_cast<const int4*>(hostVScale) + src * scales16 + j;
                to = reinterpret_cast<int4*>(vScale) + dst * scales16 + j;
            }
            *to = *from;
        }
    }
}

// Pages [first, first + count) leave the window (their cells were rewritten through another path: a prompt's staging).
extern "C" __global__ void ts_q4e_kv_invalidate(int32_t* __restrict__ table, int32_t* __restrict__ slotPage,
    const int first, const int count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
    {
        const int pg = first + i;
        const int s = table[pg];
        if (s >= 0)
        {
            slotPage[s] = -1;
            table[pg] = -1;
        }
    }
}

// ---------------------------------------------------------------------------
// Prompt attention on tensor cores. q4e_attention gives each (query head, row) its own block, whose warps take the
// row's cells one warp-wide dot at a time; the G = NH / NKV query heads of a KV head each load and widen every cached
// key and value again: 0.18 s of a 4K prompt's 12 attention layers on the RTX 5090, and a 16K prompt's rows each walk
// 2,051 cells. Here a block of four warps takes one (row, KV head): the group's query heads are the rows of a 16-row
// mma tile (G <= 16, the rest zero), the row's cells come in chunks of 16 (warp w takes chunks w, w + 4, ...), each
// chunk's keys and then values are widened once to F16 in the warp's shared tile, and S = Q K^T and O += P V run on
// mma.sync.m16n8k16 (F16 operands, F32 accumulation) under a per-warp online softmax; the warps merge at the end.
// Queries, keys, values and probabilities are F16 operands where q4e_attention keeps F32, so a prompt row's attention
// rounds differently from that kernel's (decode steps and verify windows keep the split kernel). sm_80 and newer.
// ---------------------------------------------------------------------------

#define Q4E_MMA_ATTN_WARPS 4
#define Q4E_MMA_ATTN_LDS (Q4E_ATTN_HD + 8)   // halves a shared row holds: 528 B, so ldmatrix's eight rows hit distinct banks
#define Q4E_MMA_ATTN_OLD (Q4E_ATTN_HD + 8)   // floats a row of the merged output holds

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void q4e_ldsm_x4(uint32_t (&r)[4], const half* p)
{
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void q4e_ldsm_x4_trans(uint32_t (&r)[4], const half* p)
{
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void q4e_mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t q4e_pack_half2(float lo, float hi)
{
    const half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Cells base .. base + 15 of the list (cell i itself without one) of KV head kvh into sm[16][LDS] as F16, zero past
// n: lane & 15 takes a cell, lane >> 4 half of its dims. A Q8_0 value is widened exactly (code + 1152 assembled as an
// F16 1024 + (code + 128), minus 1152) and times its F16 scale rounded once, as q4e_load_kv8's F32 product rounds to F16.
template <bool Q8>
__device__ __forceinline__ void q4e_mma_attn_stage(half (*sm)[Q4E_MMA_ATTN_LDS], const void* __restrict__ cache,
    const half* __restrict__ scale, const int32_t* __restrict__ list, const int base, const int n, const int NKV,
    const int kvh, const int lane)
{
    const int j = lane & 15, part = lane >> 4;
    uint4* dst = reinterpret_cast<uint4*>(&sm[j][part * 128]);
    const int i = base + j;
    if (i >= n)
    {
#pragma unroll
        for (int s = 0; s < 16; ++s)
            dst[s] = make_uint4(0u, 0u, 0u, 0u);
        return;
    }
    const int c = list != nullptr ? list[i] : i;
    const size_t at = ((size_t)c * NKV + kvh) * Q4E_ATTN_HD + part * 128;
    if constexpr (Q8)
    {
        const int4* src = reinterpret_cast<const int4*>(static_cast<const signed char*>(cache) + at);
        int4 w[8];
#pragma unroll
        for (int s = 0; s < 8; ++s)
            w[s] = src[s];
        const uint2 sw = *reinterpret_cast<const uint2*>(scale + at / Q4E_KV_Q8_BLOCK);
        const half* sp = reinterpret_cast<const half*>(&sw);
        const half2 magic = __float2half2_rn(1152.0f);
#pragma unroll
        for (int s = 0; s < 8; ++s)
        {
            const half2 sc = __half2half2(sp[s >> 1]);
            const uint32_t words[4] = { (uint32_t)w[s].x, (uint32_t)w[s].y, (uint32_t)w[s].z, (uint32_t)w[s].w };
            uint32_t h[8];
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                const uint32_t u = words[k] ^ 0x80808080u;
                const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
                const uint32_t hi = __byte_perm(u, 0x64646464u, 0x4342);
                const half2 vl = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&lo), magic), sc);
                const half2 vh = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&hi), magic), sc);
                h[2 * k] = *reinterpret_cast<const uint32_t*>(&vl);
                h[2 * k + 1] = *reinterpret_cast<const uint32_t*>(&vh);
            }
            dst[2 * s] = make_uint4(h[0], h[1], h[2], h[3]);
            dst[2 * s + 1] = make_uint4(h[4], h[5], h[6], h[7]);
        }
    }
    else
    {
        const int4* src = reinterpret_cast<const int4*>(static_cast<const half*>(cache) + at);
        int4 w[16];
#pragma unroll
        for (int s = 0; s < 16; ++s)
            w[s] = src[s];
#pragma unroll
        for (int s = 0; s < 16; ++s)
            dst[s] = make_uint4((uint32_t)w[s].x, (uint32_t)w[s].y, (uint32_t)w[s].z, (uint32_t)w[s].w);
    }
}
#endif

// grid (NKV, nt), Q4E_MMA_ATTN_WARPS * 32 threads. Arguments as q4e_attention's.
template <bool Q8>
__device__ __forceinline__ void q4e_attention_mma(
    const float* __restrict__ q, const void* __restrict__ kCache, const void* __restrict__ vCache,
    const half* __restrict__ kScale, const half* __restrict__ vScale,
    const float* __restrict__ gate, const int32_t* __restrict__ cells, const int32_t* __restrict__ cellCnt,
    float* __restrict__ out, const int32_t* __restrict__ positions,
    const int p0, const int NH, const int NKV, const float scale, const int cellStride)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    __shared__ __align__(16) half smQ[16][Q4E_MMA_ATTN_LDS];
    __shared__ __align__(16) half smKV[Q4E_MMA_ATTN_WARPS][16][Q4E_MMA_ATTN_LDS];
    __shared__ float smM[Q4E_MMA_ATTN_WARPS][16], smL[Q4E_MMA_ATTN_WARPS][16], smInv[16];

    const int kvh = blockIdx.x, t = blockIdx.y;
    const int G = NH / NKV;
    const int p = positions != nullptr ? positions[t] : p0 + t;
    const int32_t* list = nullptr;
    int n = p + 1;
    if (cells != nullptr && cellCnt[t] >= 0)
    {
        n = cellCnt[t];
        list = cells + (size_t)t * cellStride;
    }
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;

    // The group's query heads as F16 rows, rows G..15 zero.
    for (int u = tid; u < 16 * (Q4E_ATTN_HD / 4); u += blockDim.x)
    {
        const int r = u / (Q4E_ATTN_HD / 4), c4 = (u % (Q4E_ATTN_HD / 4)) * 4;
        float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (r < G)
            f = *reinterpret_cast<const float4*>(q + ((size_t)t * NH + kvh * G + r) * Q4E_ATTN_HD + c4);
        *reinterpret_cast<uint2*>(&smQ[r][c4]) = make_uint2(q4e_pack_half2(f.x, f.y), q4e_pack_half2(f.z, f.w));
    }
    __syncthreads();

    float o[Q4E_ATTN_HD / 8][4];
#pragma unroll
    for (int dt = 0; dt < Q4E_ATTN_HD / 8; ++dt)
        o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.0f;
    float mRow[2] = { -INFINITY, -INFINITY }, lRow[2] = { 0.0f, 0.0f };
    const float sl2 = scale * 1.4426950408889634f;
    half (*sm)[Q4E_MMA_ATTN_LDS] = smKV[warp];
    const int nChunks = (n + 15) >> 4;
    for (int ch = warp; ch < nChunks; ch += Q4E_MMA_ATTN_WARPS)
    {
        const int base = ch << 4;
        q4e_mma_attn_stage<Q8>(sm, kCache, kScale, list, base, n, NKV, kvh, lane);
        __syncwarp();
        float s[2][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
#pragma unroll
        for (int kk = 0; kk < Q4E_ATTN_HD; kk += 16)
        {
            uint32_t a[4], b[4];
            q4e_ldsm_x4(a, &smQ[(lane & 7) + ((lane >> 3) & 1) * 8][kk + (lane >> 4) * 8]);
            q4e_ldsm_x4(b, &sm[(lane & 7) + (lane >> 4) * 8][kk + ((lane >> 3) & 1) * 8]);
            q4e_mma_f16(s[0], a, b[0], b[1]);
            q4e_mma_f16(s[1], a, b[2], b[3]);
        }
        // Column 2 tig + (e & 1) of tile h is cell base + 8 h + 2 tig + (e & 1); e < 2 is head g, else g + 8.
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                s[h][e] = base + 8 * h + 2 * tig + (e & 1) < n ? s[h][e] * sl2 : -INFINITY;
        float mx0 = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1]));
        float mx1 = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mN0 = fmaxf(mRow[0], mx0), mN1 = fmaxf(mRow[1], mx1);
        const float c0 = exp2f(mRow[0] - mN0), c1 = exp2f(mRow[1] - mN1);
        mRow[0] = mN0;
        mRow[1] = mN1;
        float pr[2][4];
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                pr[h][e] = exp2f(s[h][e] - (e < 2 ? mN0 : mN1));
        float sum0 = pr[0][0] + pr[0][1] + pr[1][0] + pr[1][1];
        float sum1 = pr[0][2] + pr[0][3] + pr[1][2] + pr[1][3];
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 1);
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, 2);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 1);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, 2);
        lRow[0] = lRow[0] * c0 + sum0;
        lRow[1] = lRow[1] * c1 + sum1;
#pragma unroll
        for (int dt = 0; dt < Q4E_ATTN_HD / 8; ++dt)
        {
            o[dt][0] *= c0; o[dt][1] *= c0;
            o[dt][2] *= c1; o[dt][3] *= c1;
        }
        const uint32_t pa[4] = { q4e_pack_half2(pr[0][0], pr[0][1]), q4e_pack_half2(pr[0][2], pr[0][3]),
                                 q4e_pack_half2(pr[1][0], pr[1][1]), q4e_pack_half2(pr[1][2], pr[1][3]) };
        __syncwarp();
        q4e_mma_attn_stage<Q8>(sm, vCache, vScale, list, base, n, NKV, kvh, lane);
        __syncwarp();
#pragma unroll
        for (int dt = 0; dt < Q4E_ATTN_HD / 8; dt += 2)
        {
            uint32_t b[4];
            q4e_ldsm_x4_trans(b, &sm[(lane & 7) + ((lane >> 3) & 1) * 8][dt * 8 + (lane >> 4) * 8]);
            q4e_mma_f16(o[dt], pa, b[0], b[1]);
            q4e_mma_f16(o[dt + 1], pa, b[2], b[3]);
        }
        __syncwarp();
    }

    // Merge the warps: each scales its output to the block's maximum and adds it in turn.
    __syncthreads();
    if (tig == 0)
    {
        smM[warp][g] = mRow[0];
        smM[warp][g + 8] = mRow[1];
        smL[warp][g] = lRow[0];
        smL[warp][g + 8] = lRow[1];
    }
    __syncthreads();
    float M0 = -INFINITY, M1 = -INFINITY;
#pragma unroll
    for (int w = 0; w < Q4E_MMA_ATTN_WARPS; ++w)
    {
        M0 = fmaxf(M0, smM[w][g]);
        M1 = fmaxf(M1, smM[w][g + 8]);
    }
    const float f0 = mRow[0] == -INFINITY ? 0.0f : exp2f(mRow[0] - M0);
    const float f1 = mRow[1] == -INFINITY ? 0.0f : exp2f(mRow[1] - M1);
    float* smO = reinterpret_cast<float*>(&smKV[0][0][0]);
#pragma unroll
    for (int w = 0; w < Q4E_MMA_ATTN_WARPS; ++w)
    {
        if (warp == w)
        {
#pragma unroll
            for (int dt = 0; dt < Q4E_ATTN_HD / 8; ++dt)
            {
                const int d = dt * 8 + 2 * tig;
                float2* r0 = reinterpret_cast<float2*>(&smO[g * Q4E_MMA_ATTN_OLD + d]);
                float2* r1 = reinterpret_cast<float2*>(&smO[(g + 8) * Q4E_MMA_ATTN_OLD + d]);
                float2 v0 = make_float2(o[dt][0] * f0, o[dt][1] * f0), v1 = make_float2(o[dt][2] * f1, o[dt][3] * f1);
                if (w > 0)
                {
                    const float2 p0v = *r0, p1v = *r1;
                    v0.x += p0v.x; v0.y += p0v.y;
                    v1.x += p1v.x; v1.y += p1v.y;
                }
                *r0 = v0;
                *r1 = v1;
            }
        }
        __syncthreads();
    }
    if (tid < 16)
    {
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < Q4E_MMA_ATTN_WARPS; ++w)
            M = fmaxf(M, smM[w][tid]);
        float L = 0.0f;
#pragma unroll
        for (int w = 0; w < Q4E_MMA_ATTN_WARPS; ++w)
            L += smM[w][tid] == -INFINITY ? 0.0f : smL[w][tid] * exp2f(smM[w][tid] - M);
        smInv[tid] = 1.0f / L;
    }
    __syncthreads();
    for (int u = tid; u < G * Q4E_ATTN_HD; u += blockDim.x)
    {
        const int r = u / Q4E_ATTN_HD, d = u % Q4E_ATTN_HD;
        const size_t off = ((size_t)t * NH + kvh * G + r) * Q4E_ATTN_HD + d;
        out[off] = smO[r * Q4E_MMA_ATTN_OLD + d] * smInv[r] * q4e_sigmoid(gate[off]);
    }
#endif
}

extern "C" __global__ void __launch_bounds__(Q4E_MMA_ATTN_WARPS * 32) ts_q4e_attention_mma_f32(
    const float* __restrict__ q, const half* __restrict__ kCache, const half* __restrict__ vCache,
    const float* __restrict__ gate, const int32_t* __restrict__ cells, const int32_t* __restrict__ cellCnt,
    float* __restrict__ out, const int32_t* __restrict__ positions,
    const int p0, const int NH, const int NKV, const float scale, const int cellStride)
{
    q4e_attention_mma<false>(q, kCache, vCache, nullptr, nullptr, gate, cells, cellCnt, out, positions,
        p0, NH, NKV, scale, cellStride);
}

extern "C" __global__ void __launch_bounds__(Q4E_MMA_ATTN_WARPS * 32) ts_q4e_attention_mma_q8_f32(
    const float* __restrict__ q, const signed char* __restrict__ kCache, const signed char* __restrict__ vCache,
    const half* __restrict__ kScale, const half* __restrict__ vScale,
    const float* __restrict__ gate, const int32_t* __restrict__ cells, const int32_t* __restrict__ cellCnt,
    float* __restrict__ out, const int32_t* __restrict__ positions,
    const int p0, const int NH, const int NKV, const float scale, const int cellStride)
{
    q4e_attention_mma<true>(q, kCache, vCache, kScale, vScale, gate, cells, cellCnt, out, positions,
        p0, NH, NKV, scale, cellStride);
}

// ---------------------------------------------------------------------------
// Qwen Sparse Attention
// ---------------------------------------------------------------------------

// Raw indexer keys into the F16 cache at the rows' positions. grid nt, D threads.
extern "C" __global__ void ts_q4e_qsa_store_f32(
    const float* __restrict__ raw,       // [nt, D]
    half* __restrict__ cache,            // [nCtx, D]
    const int32_t* __restrict__ positions,
    const int p0,
    const int D)
{
    const int t = blockIdx.x;
    const int pos = positions != nullptr ? positions[t] : p0 + t;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        cache[(size_t)pos * D + d] = __float2half(raw[(size_t)t * D + d]);
}

// ts_q4e_qsa_store_f32 into a ring of `ring` cells (cell p at row p % ring): only a block's members are ever read back
// (ts_q4e_qsa_pool_ring_f32), the latest of them at most a prompt ubatch before the block completes.
extern "C" __global__ void ts_q4e_qsa_store_ring_f32(
    const float* __restrict__ raw,       // [nt, D]
    half* __restrict__ cache,            // [ring, D]
    const int32_t* __restrict__ positions,
    const int p0,
    const int D,
    const int ring)
{
    const int t = blockIdx.x;
    const int pos = positions != nullptr ? positions[t] : p0 + t;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        cache[(size_t)(pos % ring) * D + d] = __float2half(raw[(size_t)t * D + d]);
}

// The keys of blocks [first, first + count): the mean of their R cached raw keys (read back
// through the cache's F16), RMS-normed times kNorm and rotated to the block's first position.
// grid count, D threads. With positions (a captured decode step, one row), the block the row at
// positions[0] completes, if it completes one.
extern "C" __global__ void ts_q4e_qsa_pool_f32(
    const half* __restrict__ cache,      // [nCtx, D]
    const float* __restrict__ kNorm,     // [D]
    float* __restrict__ pooled,          // [nCtx / R, D]
    const int32_t* __restrict__ positions,
    const int first,
    const int R,
    const int D,
    const int nRot,
    const float base,
    const float freqScale,
    const float eps)
{
    __shared__ float sh[32];
    __shared__ float row[512];
    int b = first + blockIdx.x;
    if (positions != nullptr)
    {
        const int p = positions[0];
        if ((p + 1) % R != 0)
            return;
        b = p / R;
    }
    const int d = threadIdx.x;
    float x = 0.0f;
    if (d < D)
    {
        x = __half2float(cache[(size_t)b * R * D + d]);
        for (int r = 1; r < R; ++r)
            x += __half2float(cache[((size_t)b * R + r) * D + d]);
        x *= 1.0f / (float)R;
    }
    const float ss = q4e_block_sum(x * x, sh);
    if (d < D)
    {
        x = x * rsqrtf(ss / (float)D + eps) * kNorm[d];
        row[d] = x;
    }
    __syncthreads();
    if (d < nRot)
    {
        const int half = nRot / 2;
        const int i = d < half ? d : d - half;
        float c, s;
        q4e_rope_angle(b * R, i, nRot, base, freqScale, &c, &s);
        x = d < half ? row[d] * c - row[d + half] * s : row[d - half] * s + row[d] * c;
    }
    if (d < D)
        pooled[(size_t)b * D + d] = x;
}

// ts_q4e_qsa_pool_f32 over the raw keys' ring (ts_q4e_qsa_store_ring_f32).
extern "C" __global__ void ts_q4e_qsa_pool_ring_f32(
    const half* __restrict__ cache,      // [ring, D]
    const float* __restrict__ kNorm,     // [D]
    float* __restrict__ pooled,          // [nCtx / R, D]
    const int32_t* __restrict__ positions,
    const int first,
    const int R,
    const int D,
    const int nRot,
    const float base,
    const float freqScale,
    const float eps,
    const int ring)
{
    __shared__ float sh[32];
    __shared__ float row[512];
    int b = first + blockIdx.x;
    if (positions != nullptr)
    {
        const int p = positions[0];
        if ((p + 1) % R != 0)
            return;
        b = p / R;
    }
    const int d = threadIdx.x;
    float x = 0.0f;
    if (d < D)
    {
        x = __half2float(cache[(size_t)((b * R) % ring) * D + d]);
        for (int r = 1; r < R; ++r)
            x += __half2float(cache[(size_t)((b * R + r) % ring) * D + d]);
        x *= 1.0f / (float)R;
    }
    const float ss = q4e_block_sum(x * x, sh);
    if (d < D)
    {
        x = x * rsqrtf(ss / (float)D + eps) * kNorm[d];
        row[d] = x;
    }
    __syncthreads();
    if (d < nRot)
    {
        const int half = nRot / 2;
        const int i = d < half ? d : d - half;
        float c, s;
        q4e_rope_angle(b * R, i, nRot, base, freqScale, &c, &s);
        x = d < half ? row[d] * c - row[d + half] * s : row[d - half] * s + row[d] * c;
    }
    if (d < D)
        pooled[(size_t)b * D + d] = x;
}

// The indexer queries in place: each head RMS-normed times qNorm and rotated to the row's
// position. grid (IH, nt), D threads.
extern "C" __global__ void ts_q4e_qsa_query_f32(
    float* __restrict__ q,               // [nt, IH, D]
    const float* __restrict__ qNorm,     // [D]
    const int32_t* __restrict__ positions,
    const int p0,
    const int IH,
    const int D,
    const int nRot,
    const float base,
    const float freqScale,
    const float eps)
{
    __shared__ float sh[32];
    __shared__ float row[512];
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int d = threadIdx.x;
    const int pos = positions != nullptr ? positions[t] : p0 + t;
    float* qr = q + ((size_t)t * IH + h) * D;
    float x = d < D ? qr[d] : 0.0f;
    const float ss = q4e_block_sum(x * x, sh);
    if (d < D)
    {
        x = x * rsqrtf(ss / (float)D + eps) * qNorm[d];
        row[d] = x;
    }
    __syncthreads();
    if (d < nRot)
    {
        const int half = nRot / 2;
        const int i = d < half ? d : d - half;
        float c, s;
        q4e_rope_angle(pos, i, nRot, base, freqScale, &c, &s);
        x = d < half ? row[d] * c - row[d + half] * s : row[d - half] * s + row[d] * c;
    }
    if (d < D)
        qr[d] = x;
}

// The score of every complete block a row's query competes over: sum over the indexer heads of
// relu(block key . query head), in head order. A row at p sees blocks b < (p + 1) / R. One warp
// per block, D / 32 dims per lane, striding over the blocks (a captured step's grid is sized for
// the whole context, not for the row's position). grid (x, nt), 256 threads.
extern "C" __global__ void ts_q4e_qsa_scores_f32(
    const float* __restrict__ q,         // [nt, IH, D]
    const float* __restrict__ pooled,    // [nCtx / R, D]
    float* __restrict__ scores,          // [nt, stride]
    const int32_t* __restrict__ positions,
    const int p0,
    const int IH,
    const int D,
    const int R,
    const int stride)
{
    const int t = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int p = positions != nullptr ? positions[t] : p0 + t;
    const int nb = (p + 1) / R;
    const int per = D / 32;
    for (int b = blockIdx.x * 8 + warp; b < nb; b += gridDim.x * 8)
    {
        const float* key = pooled + (size_t)b * D + lane * per;
        float kr[8];
        for (int i = 0; i < per; ++i)
            kr[i] = key[i];
        float s = 0.0f;
        for (int h = 0; h < IH; ++h)
        {
            const float* qh = q + ((size_t)t * IH + h) * D + lane * per;
            float part = 0.0f;
            for (int i = 0; i < per; ++i)
                part += kr[i] * qh[i];
            s += fmaxf(q4e_warp_sum(part), 0.0f);
        }
        if (lane == 0)
            scores[(size_t)t * stride + b] = s;
    }
}

// A row's attended cells. Below the width W = topK + R - 1 every cell is visible anyway
// (cellCnt = -1: attend 0..p). Past it, the cells of the query's own incomplete block
// (tail..p) first, then the best complete blocks by score until W cells are listed: whole
// blocks, and the lowest cells of the last one where W splits it. Ties rank the lower block
// first. That is the native cell-level top-W, whose order among equal scores is unspecified.
// One block of 256 threads per row.
extern "C" __global__ void ts_q4e_qsa_select_i32(
    const float* __restrict__ scores,    // [nt, stride]
    int32_t* __restrict__ cells,         // [nt, cellStride]
    int32_t* __restrict__ cellCnt,       // [nt]
    const int32_t* __restrict__ positions,
    const int p0,
    const int R,
    const int topK,
    const int stride,
    const int cellStride)
{
    const int t = blockIdx.x;
    const int p = positions != nullptr ? positions[t] : p0 + t;
    const int W = topK + R - 1;
    if (p + 1 <= W)
    {
        if (threadIdx.x == 0)
            cellCnt[t] = -1;
        return;
    }
    const int tail = (p + 1) / R * R;
    const int t0 = p + 1 - tail;
    const int nb = tail / R;
    const int need = W - t0;
    const int full = need / R;
    const int part = need - full * R;
    const int K = full + (part > 0 ? 1 : 0);
    const float* sc = scores + (size_t)t * stride;
    int32_t* out = cells + (size_t)t * cellStride;

    for (int i = threadIdx.x; i < t0; i += blockDim.x)
        out[i] = tail + i;

    // The K-th largest score, byte by byte from the most significant (order-preserving keys).
    __shared__ unsigned int hist[256];
    __shared__ unsigned int shPrefix;
    __shared__ int shNeed;
    __shared__ int scanWarp[8];
    if (threadIdx.x == 0)
    {
        shPrefix = 0;
        shNeed = K;
    }
    __syncthreads();
    for (int pass = 0; pass < 4; ++pass)
    {
        const int shift = 24 - 8 * pass;
        hist[threadIdx.x] = 0;
        __syncthreads();
        const unsigned int prefix = shPrefix;
        for (int b = threadIdx.x; b < nb; b += blockDim.x)
        {
            unsigned int key = __float_as_uint(sc[b]);
            key ^= (key & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
            if (pass > 0 && (key >> (shift + 8)) != prefix)
                continue;
            atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
        }
        __syncthreads();
        // The bin a walk from bin 255 down would stop at - the first whose count reaches the need left, the need then
        // dropping by the bins above it - as a block-wide scan (thread i holds bin 255 - i) instead of 256 serial steps
        // on one thread. No bin reaching it ends the walk at bin 0 with every count taken off, as the walk did.
        {
            const int i = threadIdx.x;
            const int c = (int)hist[255 - i];
            int incl = c;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1)
            {
                const int o = __shfl_up_sync(0xFFFFFFFFu, incl, off);
                if ((i & 31) >= off)
                    incl += o;
            }
            if ((i & 31) == 31)
                scanWarp[i >> 5] = incl;
            const int n = shNeed;
            __syncthreads();
            for (int w = 0; w < (i >> 5); ++w)
                incl += scanWarp[w];
            const int excl = incl - c;
            if (incl >= n && excl < n)
            {
                shNeed = n - excl;
                shPrefix = (shPrefix << 8) | (unsigned int)(255 - i);
            }
            else if (i == 255 && incl < n)
            {
                shNeed = n - incl;
                shPrefix = shPrefix << 8;
            }
        }
        __syncthreads();
    }

    // Slots in rank order: keys above the threshold, then ties in block order; slot `full` is
    // the partial block. A block-wide ballot scan assigns the slots deterministically.
    const unsigned int threshold = shPrefix;
    __shared__ int warpCounts[8];
    __shared__ int shBase;
    if (threadIdx.x == 0)
        shBase = 0;
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int pass = 0; pass < 2; ++pass)
    {
        for (int b0 = 0; b0 < nb; b0 += blockDim.x)
        {
            const int b = b0 + threadIdx.x;
            bool take = false;
            if (b < nb)
            {
                unsigned int key = __float_as_uint(sc[b]);
                key ^= (key & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
                take = pass == 0 ? key > threshold : key == threshold;
            }
            const unsigned int ballot = __ballot_sync(0xFFFFFFFFu, take);
            if (lane == 0)
                warpCounts[warp] = __popc(ballot);
            __syncthreads();
            int slot = shBase + __popc(ballot & ((1u << lane) - 1u));
            for (int w = 0; w < warp; ++w)
                slot += warpCounts[w];
            if (take && slot < K)
            {
                const int count = slot < full ? R : part;
                for (int r = 0; r < count; ++r)
                    out[t0 + slot * R + r] = b * R + r;
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                int total = 0;
                for (int w = 0; w < (int)(blockDim.x >> 5); ++w)
                    total += warpCounts[w];
                shBase += total;
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0)
        cellCnt[t] = W;
}

// ---------------------------------------------------------------------------
// PLE
// ---------------------------------------------------------------------------

// The PLE gate, one block per row: per stream c, s = <key_c, res_c> / sqrt(E) over the
// grouped-normed key and residual, g = sigmoid(sign(s) sqrt(clamp(|s|, 1e-6))), and
// gated_c = value * g.
extern "C" __global__ void ts_q4e_ple_gate_f32(
    const float* __restrict__ key,       // [nt, HC, E]
    const float* __restrict__ res,       // [nt, HC, E]
    const float* __restrict__ normKey,   // [HC * E]
    const float* __restrict__ normQuery, // [HC * E]
    const float* __restrict__ value,     // [nt, E]
    float* __restrict__ gated,           // [nt, HC, E]
    const int E,
    const int HC,
    const float eps)
{
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* kr = key + (size_t)t * HC * E;
    const float* rr = res + (size_t)t * HC * E;
    const float* vr = value + (size_t)t * E;
    float* gr = gated + (size_t)t * HC * E;
    for (int c = 0; c < HC; ++c)
    {
        float sk = 0.0f, sq = 0.0f;
        for (int i = threadIdx.x; i < E; i += blockDim.x)
        {
            const float a = kr[c * E + i], b = rr[c * E + i];
            sk += a * a;
            sq += b * b;
        }
        const float invK = rsqrtf(q4e_block_sum(sk, sh) / (float)E + eps);
        const float invQ = rsqrtf(q4e_block_sum(sq, sh) / (float)E + eps);
        float dot = 0.0f;
        for (int i = threadIdx.x; i < E; i += blockDim.x)
        {
            const int j = c * E + i;
            dot += (kr[j] * invK * normKey[j]) * (rr[j] * invQ * normQuery[j]);
        }
        const float s = q4e_block_sum(dot, sh) * (1.0f / sqrtf((float)E));
        const float sg = s > 0.0f ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
        const float g = q4e_sigmoid(sg * sqrtf(fminf(fmaxf(s * sg, 1e-6f), 3.0e38f)));
        for (int i = threadIdx.x; i < E; i += blockDim.x)
            gr[c * E + i] = vr[i] * g;
    }
}

// normc = each stream of `gated` RMS-normed, times normConv. One block per row.
extern "C" __global__ void ts_q4e_group_norm_f32(
    const float* __restrict__ x,         // [nt, HC, E]
    const float* __restrict__ w,         // [HC * E]
    float* __restrict__ y,               // [nt, HC, E]
    const int E,
    const int HC,
    const float eps)
{
    // grid (nt, HC): a block per row and group (a block per row walked the four groups in turn, 21 us a call on the
    // RTX 5090 for one row); each group's sum is the same block reduction as before, so the outputs are unchanged.
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* xr = x + (size_t)t * HC * E;
    float* yr = y + (size_t)t * HC * E;
    for (int c = blockIdx.y; c < HC; c += gridDim.y)
    {
        float ss = 0.0f;
        for (int i = threadIdx.x; i < E; i += blockDim.x)
        {
            const float v = xr[c * E + i];
            ss += v * v;
        }
        const float inv = rsqrtf(q4e_block_sum(ss, sh) / (float)E + eps);
        for (int i = threadIdx.x; i < E; i += blockDim.x)
            yr[c * E + i] = xr[c * E + i] * inv * w[c * E + i];
    }
}

// The dilated causal depthwise convolution of the normed gate output over each sequence's
// history (its hist = (kern-1) * dil previous rows, oldest first), SiLU, and
// res += gated + conv. Tap kk reads (kern-1-kk) * dil rows back; taps sum in order.
// One thread per (row, channel).
extern "C" __global__ void ts_q4e_ple_conv_f32(
    const float* __restrict__ gated,     // [rows, C]
    const float* __restrict__ normc,     // [rows, C]
    const float* const* __restrict__ hists, // [nSeq] -> [hist, C]
    const float* __restrict__ convWT,    // [kern, C]
    float* __restrict__ res,             // [rows, C]
    const int rowsPerSeq,
    const int nSeq,
    const int C,
    const int kern,
    const int dil)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)rowsPerSeq * nSeq * C)
        return;
    const int row = (int)(i / C);
    const int ch = (int)(i - (long long)row * C);
    const int seq = row / rowsPerSeq;
    const int local = row - seq * rowsPerSeq;
    const int rowBase = seq * rowsPerSeq;
    const int hist = (kern - 1) * dil;
    const float* h = hists[seq];
    float acc = 0.0f;
    for (int kk = 0; kk < kern; ++kk)
    {
        const int src = local - (kern - 1 - kk) * dil;
        const float v = src >= 0 ? normc[(size_t)(rowBase + src) * C + ch] : h[(size_t)(hist + src) * C + ch];
        const float term = v * convWT[(size_t)kk * C + ch];
        acc = kk == 0 ? term : acc + term;
    }
    const size_t at = (size_t)row * C + ch;
    res[at] = (res[at] + gated[at]) + q4e_silu(acc);
}

// ---------------------------------------------------------------------------
// MoE shared expert
// ---------------------------------------------------------------------------

// sh[t] *= sigmoid(cur[t] . w): the shared expert's scalar gate. One block per row.
extern "C" __global__ void ts_q4e_shared_gate_f32(
    const float* __restrict__ cur,       // [nt, E]
    const float* __restrict__ w,         // [E]
    float* __restrict__ sh,              // [nt, E]
    const int E)
{
    __shared__ float red[32];
    const int t = blockIdx.x;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x)
        dot += cur[(size_t)t * E + i] * w[i];
    const float g = q4e_sigmoid(q4e_block_sum(dot, red));
    for (int i = threadIdx.x; i < E; i += blockDim.x)
        sh[(size_t)t * E + i] *= g;
}

// x[t, :] of every row replicated into the HC streams of xs (the embedding scatter for rows
// the host supplies). grid (ceil(E / 256), nt).
extern "C" __global__ void ts_q4e_set_stream_rows_f32(
    const float* __restrict__ rows,      // [n, E]
    float* __restrict__ xs,              // [nt, HC, E]
    const int firstRow,
    const int E,
    const int HC)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (i >= E)
        return;
    const float v = rows[(size_t)r * E + i];
    for (int c = 0; c < HC; ++c)
        xs[((size_t)(firstRow + r) * HC + c) * E + i] = v;
}

// Each row's first-max index (ties to the lowest index, as a host scan with `>` finds it; a row with no finite
// maximum answers 0): a greedy verify returns the window's tokens instead of every row's logits. grid rows, 1024
// threads.
extern "C" __global__ void ts_q4e_argmax_rows_f32(
    const float* __restrict__ logits,    // [rows, V]
    int32_t* __restrict__ ids,           // [rows]
    const int V)
{
    __shared__ float shV[32];
    __shared__ int shI[32];
    const float* row = logits + (size_t)blockIdx.x * V;
    float best = -INFINITY;
    int bestIdx = -1;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
    {
        const float v = row[i];
        if (v > best)
        {
            best = v;
            bestIdx = i;
        }
    }
    // Warp then block: the larger value wins, a tie the lower index.
    for (int o = 16; o > 0; o >>= 1)
    {
        const float ov = __shfl_down_sync(0xffffffffu, best, o);
        const int oi = __shfl_down_sync(0xffffffffu, bestIdx, o);
        if (oi >= 0 && (bestIdx < 0 || ov > best || (ov == best && oi < bestIdx)))
        {
            best = ov;
            bestIdx = oi;
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0)
    {
        shV[warp] = best;
        shI[warp] = bestIdx;
    }
    __syncthreads();
    if (warp == 0)
    {
        const int nWarps = blockDim.x >> 5;
        best = lane < nWarps ? shV[lane] : -INFINITY;
        bestIdx = lane < nWarps ? shI[lane] : -1;
        for (int o = 16; o > 0; o >>= 1)
        {
            const float ov = __shfl_down_sync(0xffffffffu, best, o);
            const int oi = __shfl_down_sync(0xffffffffu, bestIdx, o);
            if (oi >= 0 && (bestIdx < 0 || ov > best || (ov == best && oi < bestIdx)))
            {
                best = ov;
                bestIdx = oi;
            }
        }
        if (lane == 0)
            ids[blockIdx.x] = bestIdx < 0 ? 0 : bestIdx;
    }
}

// Adaptive residency's selection counts from a prompt ubatch (decode steps count inside the doorbell kernel):
// counts[sel[i]] += 1 for each of the ubatch's n selections, counts being the layer's row of the table.
extern "C" __global__ void ts_q4e_count_selected(
    const int32_t* __restrict__ sel,
    const int n,
    unsigned int* __restrict__ counts)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const int e = sel[i];
    if (e >= 0)
        atomicAdd(counts + e, 1u);
}

// A draft step's pick on the device (the MTP draft chain): the row's first-max token, its confidence as the per-token
// draft gate measures it on the host (DraftHeadSpeculator.ArgmaxWithTopKConfidence: the top-1 probability of a
// softmax over the 10 largest logits), and the token again as the next step's input. One block of 1024 threads.
#define Q4E_PICK_K 10
extern "C" __global__ void ts_q4e_draft_pick_f32(
    const float* __restrict__ logits,    // [V]
    const int V,
    int32_t* __restrict__ ids,           // [steps]
    float* __restrict__ probs,           // [steps]
    const int step,
    int32_t* __restrict__ nextToken)     // [1]
{
    __shared__ float shV[32];
    __shared__ int shK[32], shT[32];
    __shared__ float top[Q4E_PICK_K];
    __shared__ int argIdx;
    // Each thread's own 10 largest values (descending; an equal value keeps the earlier index first) and the index of
    // its largest.
    float tv[Q4E_PICK_K];
#pragma unroll
    for (int j = 0; j < Q4E_PICK_K; ++j)
        tv[j] = -INFINITY;
    int headIdx = -1;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
    {
        const float v = logits[i];
        if (!(v > tv[Q4E_PICK_K - 1]))
            continue;
        int j = Q4E_PICK_K - 1;
        while (j > 0 && tv[j - 1] < v)
        {
            tv[j] = tv[j - 1];
            --j;
        }
        tv[j] = v;
        if (j == 0)
            headIdx = i;
    }
    // Ten rounds of a block maximum over the threads' next values; the first round's tie goes to the lower index.
    int ptr = 0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nWarps = blockDim.x >> 5;
    for (int r = 0; r < Q4E_PICK_K; ++r)
    {
        float v = ptr < Q4E_PICK_K ? tv[ptr] : -INFINITY;
        int key = ptr == 0 && headIdx >= 0 ? headIdx : 0x7fffffff;
        int who = threadIdx.x;
        for (int o = 16; o > 0; o >>= 1)
        {
            const float ov = __shfl_down_sync(0xffffffffu, v, o);
            const int ok = __shfl_down_sync(0xffffffffu, key, o);
            const int ow = __shfl_down_sync(0xffffffffu, who, o);
            if (ov > v || (ov == v && (ok < key || (ok == key && ow < who))))
            {
                v = ov; key = ok; who = ow;
            }
        }
        if (lane == 0)
        {
            shV[warp] = v; shK[warp] = key; shT[warp] = who;
        }
        __syncthreads();
        if (warp == 0)
        {
            v = lane < nWarps ? shV[lane] : -INFINITY;
            key = lane < nWarps ? shK[lane] : 0x7fffffff;
            who = lane < nWarps ? shT[lane] : 0x7fffffff;
            for (int o = 16; o > 0; o >>= 1)
            {
                const float ov = __shfl_down_sync(0xffffffffu, v, o);
                const int ok = __shfl_down_sync(0xffffffffu, key, o);
                const int ow = __shfl_down_sync(0xffffffffu, who, o);
                if (ov > v || (ov == v && (ok < key || (ok == key && ow < who))))
                {
                    v = ov; key = ok; who = ow;
                }
            }
            if (lane == 0)
            {
                top[r] = v;
                shT[0] = who;
                if (r == 0)
                    argIdx = key == 0x7fffffff ? 0 : key;
            }
        }
        __syncthreads();
        if (threadIdx.x == shT[0])
            ++ptr;
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        float denom = 0.0f;
        for (int j = 0; j < Q4E_PICK_K; ++j)
        {
            if (top[j] == -INFINITY)
                break;
            denom += expf(top[j] - top[0]);
        }
        ids[step] = argIdx;
        probs[step] = denom > 0.0f ? 1.0f / denom : 0.0f;
        *nextToken = argIdx;
    }
}

// ts_q4e_draft_pick_f32 in two stages. One block reading a 248,320-token row took 112 us a draft step (each of its
// 1,024 threads walking 243 logits); here every block keeps the ten largest logits of its slice and the first index of
// its largest, and one block merges those candidates by the same rules. The ten largest of the row are among the
// slices' ten largest, and its first maximum is the lowest first index among the slices holding the maximum (the slices
// are in index order, and a thread meets its candidates in that order), so the pick and its confidence are the
// one-block kernel's bit for bit.
// Ten rounds of a block maximum over each thread's ten largest values of vals[0, n) (strided), the first round's tie
// to the lower key: top[] gets the ten largest in descending order (-inf past the count), *head the key of the first
// (-1 if none). keys == null keys element i as base + i.
__device__ __forceinline__ void q4e_pick_topk_block(const float* __restrict__ vals, const int32_t* __restrict__ keys,
    const int base, const int n, float* top, int* head)
{
    __shared__ float shV[32];
    __shared__ int shK[32], shT[32];
    float tv[Q4E_PICK_K];
#pragma unroll
    for (int j = 0; j < Q4E_PICK_K; ++j)
        tv[j] = -INFINITY;
    int headIdx = -1;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        const float v = vals[i];
        if (!(v > tv[Q4E_PICK_K - 1]))
            continue;
        int j = Q4E_PICK_K - 1;
        while (j > 0 && tv[j - 1] < v)
        {
            tv[j] = tv[j - 1];
            --j;
        }
        tv[j] = v;
        if (j == 0)
            headIdx = keys != nullptr ? keys[i] : base + i;
    }
    int ptr = 0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, nWarps = blockDim.x >> 5;
    for (int r = 0; r < Q4E_PICK_K; ++r)
    {
        float v = ptr < Q4E_PICK_K ? tv[ptr] : -INFINITY;
        int key = ptr == 0 && headIdx >= 0 ? headIdx : 0x7fffffff;
        int who = threadIdx.x;
        for (int o = 16; o > 0; o >>= 1)
        {
            const float ov = __shfl_down_sync(0xffffffffu, v, o);
            const int ok = __shfl_down_sync(0xffffffffu, key, o);
            const int ow = __shfl_down_sync(0xffffffffu, who, o);
            if (ov > v || (ov == v && (ok < key || (ok == key && ow < who))))
            {
                v = ov; key = ok; who = ow;
            }
        }
        if (lane == 0)
        {
            shV[warp] = v; shK[warp] = key; shT[warp] = who;
        }
        __syncthreads();
        if (warp == 0)
        {
            v = lane < nWarps ? shV[lane] : -INFINITY;
            key = lane < nWarps ? shK[lane] : 0x7fffffff;
            who = lane < nWarps ? shT[lane] : 0x7fffffff;
            for (int o = 16; o > 0; o >>= 1)
            {
                const float ov = __shfl_down_sync(0xffffffffu, v, o);
                const int ok = __shfl_down_sync(0xffffffffu, key, o);
                const int ow = __shfl_down_sync(0xffffffffu, who, o);
                if (ov > v || (ov == v && (ok < key || (ok == key && ow < who))))
                {
                    v = ov; key = ok; who = ow;
                }
            }
            if (lane == 0)
            {
                top[r] = v;
                shT[0] = who;
                if (r == 0)
                    *head = key == 0x7fffffff ? -1 : key;
            }
        }
        __syncthreads();
        if (threadIdx.x == shT[0])
            ++ptr;
        __syncthreads();
    }
}

// Stage one: block b's slice [b * chunk, +chunk) of the row into candidates [b * 10, +10) (values, and the slice's
// first-max index beside its largest; the other keys are never compared first).
extern "C" __global__ void ts_q4e_draft_pick_part_f32(
    const float* __restrict__ logits,
    const int V,
    const int chunk,
    float* __restrict__ candVal,         // [blocks * 10]
    int32_t* __restrict__ candKey)       // [blocks * 10]
{
    __shared__ float top[Q4E_PICK_K];
    __shared__ int head;
    const int base = blockIdx.x * chunk;
    const int n = max(0, min(chunk, V - base));
    q4e_pick_topk_block(logits + base, nullptr, base, n, top, &head);
    if (threadIdx.x < Q4E_PICK_K)
    {
        candVal[blockIdx.x * Q4E_PICK_K + threadIdx.x] = top[threadIdx.x];
        candKey[blockIdx.x * Q4E_PICK_K + threadIdx.x] = threadIdx.x == 0 ? head : 0x7fffffff;
    }
}

// Stage two: one block over the n candidates, then ts_q4e_draft_pick_f32's outputs.
extern "C" __global__ void ts_q4e_draft_pick_merge_f32(
    const float* __restrict__ candVal,
    const int32_t* __restrict__ candKey,
    const int n,
    int32_t* __restrict__ ids,
    float* __restrict__ probs,
    const int step,
    int32_t* __restrict__ nextToken)
{
    __shared__ float top[Q4E_PICK_K];
    __shared__ int head;
    q4e_pick_topk_block(candVal, candKey, 0, n, top, &head);
    if (threadIdx.x == 0)
    {
        const int argIdx = head < 0 ? 0 : head;
        float denom = 0.0f;
        for (int j = 0; j < Q4E_PICK_K; ++j)
        {
            if (top[j] == -INFINITY)
                break;
            denom += expf(top[j] - top[0]);
        }
        ids[step] = argIdx;
        probs[step] = denom > 0.0f ? 1.0f / denom : 0.0f;
        *nextToken = argIdx;
    }
}
