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
    __shared__ float sh[32];
    const int slice = blockIdx.x;
    const int t = blockIdx.y;
    const int flat = HC * E;
    const int j = slice * Q4E_HC_SLICE + threadIdx.x;
    const float x = xs[(size_t)t * flat + j];
    float* out = partials + ((size_t)t * gridDim.x + slice) * Q4E_HC_PARTIAL;
    const float ss = q4e_block_sum(x * x, sh);
    if (threadIdx.x == 0)
        out[0] = ss;
    if (injW != nullptr)
    {
        const float xw = x * normW[j];
        for (int c = 0; c < HC; ++c)
        {
            const float p = q4e_block_sum(xw * injW[(size_t)c * flat + j], sh);
            if (threadIdx.x == 0)
                out[1 + c] = p;
        }
    }
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

// One block per (head, row): the query heads' RMS norm and rotary, their gates split out of the
// interleaved q|gate projection; the KV heads' norm and rotary, stored with the values into the
// F16 caches at the row's position. Rows sit at p0, p0 + 1, ... or at positions[row].
// grid (NH + NKV, nt), Q4E_ATTN_HD threads.
extern "C" __global__ void ts_q4e_attn_prep_f32(
    const float* __restrict__ qg,        // [nt, NH, 2, HD]
    const float* __restrict__ k,         // [nt, NKV, HD]
    const float* __restrict__ v,         // [nt, NKV, HD]
    const float* __restrict__ qNorm,     // [HD]
    const float* __restrict__ kNorm,     // [HD]
    float* __restrict__ q,               // [nt, NH, HD]
    float* __restrict__ gate,            // [nt, NH, HD]
    half* __restrict__ kCache,           // [nCtx, NKV, HD]
    half* __restrict__ vCache,           // [nCtx, NKV, HD]
    const int32_t* __restrict__ positions,
    const int p0,
    const int NH,
    const int NKV,
    const int nRot,
    const float base,
    const float freqScale,
    const float eps)
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

// Online-softmax attention of one query head over its KV head's cells, then the sigmoid gate.
// The cells are 0..p (cellCnt null or cellCnt[t] < 0) or the cellCnt[t] listed in cells.
// grid (NH, nt), 256 threads = 8 warps splitting the cells; lane owns 8 of the 256 dims.
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

// ts_q4e_attention_f32 for HG query heads of one KV head in a block: each cell's K and V rows load once for the
// group instead of once per head. Each head keeps that kernel's arithmetic exactly (the same warp per cell
// split, dot order, online softmax and cross-warp combine), so the output is bit-identical.
// grid (NH / HG, nt), 256 threads = 8 warps splitting the cells; lane owns 8 of the 256 dims.
template <int HG>
__device__ __forceinline__ void q4e_attention_group(
    const float* __restrict__ q, const half* __restrict__ kCache, const half* __restrict__ vCache,
    const float* __restrict__ gate, const int32_t* __restrict__ cells, const int32_t* __restrict__ cellCnt,
    float* __restrict__ out, const int32_t* __restrict__ positions, const int p0, const int NH, const int NKV,
    const float scale, const int cellStride)
{
    const int h0 = blockIdx.x * HG;
    const int t = blockIdx.y;
    const int kvh = h0 / (NH / NKV);
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
    float qReg[HG][8];
#pragma unroll
    for (int g = 0; g < HG; ++g)
    {
        const float* qRow = q + ((size_t)t * NH + h0 + g) * Q4E_ATTN_HD + lane * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            qReg[g][i] = qRow[i];
    }
    float m[HG], sum[HG], acc[HG][8];
#pragma unroll
    for (int g = 0; g < HG; ++g)
    {
        m[g] = -INFINITY;
        sum[g] = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            acc[g][i] = 0.0f;
    }
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
#pragma unroll
        for (int g = 0; g < HG; ++g)
        {
            float part = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                part += qReg[g][j] * kf[j];
            const float s = q4e_warp_sum(part) * scale;
            if (s > m[g])
            {
                const float r = expf(m[g] - s);
                sum[g] *= r;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[g][j] *= r;
                m[g] = s;
            }
            const float w = expf(s - m[g]);
            sum[g] += w;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                acc[g][j] += w * vf[j];
        }
    }

    __shared__ float shAcc[8][Q4E_ATTN_HD];
    __shared__ float shM[8], shSum[8];
#pragma unroll
    for (int g = 0; g < HG; ++g)
    {
#pragma unroll
        for (int j = 0; j < 8; ++j)
            shAcc[warp][lane * 8 + j] = acc[g][j];
        if (lane == 0)
        {
            shM[warp] = m[g];
            shSum[warp] = sum[g];
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
        const size_t off = ((size_t)t * NH + h0 + g) * Q4E_ATTN_HD;
        for (int d = threadIdx.x; d < Q4E_ATTN_HD; d += blockDim.x)
        {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w)
                v += shM[w] == -INFINITY ? 0.0f : shAcc[w][d] * expf(shM[w] - M);
            out[off + d] = v * invS * q4e_sigmoid(gate[off + d]);
        }
        __syncthreads();
    }
}

extern "C" __global__ void ts_q4e_attention_g4_f32(
    const float* __restrict__ q, const half* __restrict__ kCache, const half* __restrict__ vCache,
    const float* __restrict__ gate, const int32_t* __restrict__ cells, const int32_t* __restrict__ cellCnt,
    float* __restrict__ out, const int32_t* __restrict__ positions, const int p0, const int NH, const int NKV,
    const float scale, const int cellStride)
{ q4e_attention_group<4>(q, kCache, vCache, gate, cells, cellCnt, out, positions, p0, NH, NKV, scale, cellStride); }

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
        if (threadIdx.x == 0)
        {
            int n = shNeed;
            unsigned int bin = 0;
            for (int b2 = 255; b2 >= 0; --b2)
            {
                const int c = (int)hist[b2];
                if (c >= n)
                {
                    bin = (unsigned int)b2;
                    break;
                }
                n -= c;
            }
            shNeed = n;
            shPrefix = (shPrefix << 8) | bin;
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
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* xr = x + (size_t)t * HC * E;
    float* yr = y + (size_t)t * HC * E;
    for (int c = 0; c < HC; ++c)
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
