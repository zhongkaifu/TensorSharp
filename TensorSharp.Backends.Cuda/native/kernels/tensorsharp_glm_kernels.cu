// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// GLM-5.3-Flash (glm5next) kernels for the direct-CUDA backend: the stages the
// DeepSeek V4 module (tensorsharp_dsv4_kernels.cu, whose hyper-connection and
// MoE kernels this model shares) does not have.
//
//   * KDA (Kimi Delta Attention) linear attention on 34 of the 45 trunk layers:
//     the gated delta rule with a per-CHANNEL decay, q/k/v through one short
//     causal convolution with SiLU, L2-normed q and k, and a low-rank sigmoid
//     output gate after a per-head RMS norm.
//
// The semantics are those of build_kda in ggml_ops_glm_dsa.cpp (the native
// executor ggml_cuda runs) and of ggml's gated_delta_net, with
// GlmDsaModel.Glm5Next.cs as the managed reference.
//
// A launch covers nSeq sequences of rowsPerSeq consecutive rows each: one
// sequence's prefill ubatch (nSeq = 1) or one row of each of several sequences
// in a batched decode step (rowsPerSeq = 1). Each sequence brings its own
// convolution and recurrent state, passed as device arrays of pointers.
// ---------------------------------------------------------------------------

#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float ts_glm_warp_sum(float v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off, 32);
    return v;
}

__device__ __forceinline__ float ts_glm_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

// KDA head width the kernels are written for: 32 lanes x 4 channels.
#define TS_GLM_KDA_HD 128
// Per (sequence row, head) scratch: q | k | v | log-decay (128 each), then beta.
#define TS_GLM_KDA_SCR (4 * TS_GLM_KDA_HD + 1)

// Channel `ch` of one of the three convolved streams at row `local` of its sequence:
// taps 0..dc-2 reach back into the previous rows (the state holds the dc-1 inputs
// before this ubatch, oldest first), the last tap is the row itself.
__device__ __forceinline__ float ts_glm_kda_conv(
    const float* __restrict__ x, const float* __restrict__ state, const float* __restrict__ w,
    int rowBase, int local, int ch, int dInner, int stateStride, int dc)
{
    float acc = 0.0f;
    for (int ki = 0; ki < dc; ++ki)
    {
        const int back = dc - 1 - ki;            // rows before `local`
        const int src = local - back;
        const float v = src >= 0
            ? x[(size_t)(rowBase + src) * dInner + ch]
            : state[(size_t)(dc - 1 + src) * stateStride + ch];
        acc += v * w[(size_t)ch * dc + ki];
    }
    return acc / (1.0f + expf(-acc));            // SiLU on the convolution output
}

// One warp per (row, head). Short convolution + SiLU of q, k and v; L2 norm of q
// and k (ggml_l2_norm: x / max(|x|, 1e-6)); the per-channel log decay
// lb * sigmoid(-(f + dt_bias) * a[h]) (ssm_a holds -exp(A_log)); beta = sigmoid.
extern "C" __global__ void ts_glm_kda_prep_f32(
    const float* __restrict__ q,          // [rows, dInner] projections
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* const* __restrict__ convStates, // [nSeq] -> [dc-1, 3*dInner]
    const float* __restrict__ convQ,      // [dInner, dc]
    const float* __restrict__ convK,
    const float* __restrict__ convV,
    const float* __restrict__ f,          // [rows, dInner] f_b(f_a(x))
    const float* __restrict__ dtBias,     // [dInner]
    const float* __restrict__ aVec,       // [H]
    const float* __restrict__ betaRaw,    // [rows, H]
    float* __restrict__ scr,              // [nSeq, H, rowsPerSeq, TS_GLM_KDA_SCR]
    const int rowsPerSeq,
    const int nSeq,
    const int H,
    const int dc,
    const float gateLowerBound)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int rows = rowsPerSeq * nSeq;
    if (warp >= rows * H)
        return;
    const int h = warp % H;
    const int row = warp / H;
    const int seq = row / rowsPerSeq;
    const int local = row - seq * rowsPerSeq;
    const int rowBase = seq * rowsPerSeq;
    const int dInner = H * TS_GLM_KDA_HD;
    const float* state = convStates[seq];
    const int stateStride = 3 * dInner;

    float* dst = scr + (((size_t)seq * H + h) * rowsPerSeq + local) * TS_GLM_KDA_SCR;

    float qv[4], kv[4];
    float qs = 0.0f, ks = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int ch = h * TS_GLM_KDA_HD + lane + 32 * j;
        qv[j] = ts_glm_kda_conv(q, state, convQ, rowBase, local, ch, dInner, stateStride, dc);
        kv[j] = ts_glm_kda_conv(k, state + dInner, convK, rowBase, local, ch, dInner, stateStride, dc);
        qs += qv[j] * qv[j];
        ks += kv[j] * kv[j];
    }
    const float qScale = 1.0f / fmaxf(sqrtf(ts_glm_warp_sum(qs)), 1e-6f);
    const float kScale = 1.0f / fmaxf(sqrtf(ts_glm_warp_sum(ks)), 1e-6f);
    const float a = aVec[h];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int d = lane + 32 * j;
        const int ch = h * TS_GLM_KDA_HD + d;
        dst[d] = qv[j] * qScale;
        dst[TS_GLM_KDA_HD + d] = kv[j] * kScale;
        dst[2 * TS_GLM_KDA_HD + d] = ts_glm_kda_conv(v, state + 2 * dInner, convV, rowBase, local, ch, dInner, stateStride, dc);
        const float y = (f[(size_t)row * dInner + ch] + dtBias[ch]) * a;
        dst[3 * TS_GLM_KDA_HD + d] = gateLowerBound * ts_glm_sigmoid(-y);
    }
    if (lane == 0)
        dst[4 * TS_GLM_KDA_HD] = ts_glm_sigmoid(betaRaw[(size_t)row * H + h]);
}

// The recurrence. State S (per head, 128 x 128) is stored [v row][k col]; each warp
// keeps four v-rows in registers (lane owns k columns lane + 32j) across every row of
// its sequence:
//   S[:, c] *= exp(g[c]);  delta = (v - S k) * beta;  S += delta k^T;  o = (S q) / sqrt(hd)
// grid (H, 128 / (4 * warps), nSeq).
extern "C" __global__ void ts_glm_kda_scan_f32(
    const float* __restrict__ scr,
    float* const* __restrict__ ssmStates, // [nSeq] -> [H, 128, 128]
    float* __restrict__ core,             // [nSeq * rowsPerSeq, H * 128]
    const int rowsPerSeq,
    const int H)
{
    const int h = blockIdx.x;
    const int seq = blockIdx.z;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int r0 = (blockIdx.y * (blockDim.x >> 5) + warp) * 4;
    if (r0 >= TS_GLM_KDA_HD)
        return;

    const float* base = scr + ((size_t)seq * H + h) * rowsPerSeq * TS_GLM_KDA_SCR;
    float* S = ssmStates[seq] + (size_t)h * TS_GLM_KDA_HD * TS_GLM_KDA_HD;
    const float scale = rsqrtf((float)TS_GLM_KDA_HD);

    float st[4][4];
#pragma unroll
    for (int ri = 0; ri < 4; ++ri)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            st[ri][j] = S[(size_t)(r0 + ri) * TS_GLM_KDA_HD + lane + 32 * j];

    for (int s = 0; s < rowsPerSeq; ++s)
    {
        const float* row = base + (size_t)s * TS_GLM_KDA_SCR;
        float qj[4], kj[4], dj[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            qj[j] = row[lane + 32 * j];
            kj[j] = row[TS_GLM_KDA_HD + lane + 32 * j];
            dj[j] = expf(row[3 * TS_GLM_KDA_HD + lane + 32 * j]);
        }
        const float beta = row[4 * TS_GLM_KDA_HD];
        float* out = core + ((size_t)seq * rowsPerSeq + s) * H * TS_GLM_KDA_HD + h * TS_GLM_KDA_HD;
#pragma unroll
        for (int ri = 0; ri < 4; ++ri)
        {
            float kvDot = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                st[ri][j] *= dj[j];
                kvDot += st[ri][j] * kj[j];
            }
            kvDot = ts_glm_warp_sum(kvDot);
            const float delta = (row[2 * TS_GLM_KDA_HD + r0 + ri] - kvDot) * beta;
            float o = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                st[ri][j] += kj[j] * delta;
                o += st[ri][j] * qj[j];
            }
            o = ts_glm_warp_sum(o);
            if (lane == 0)
                out[r0 + ri] = o * scale;
        }
    }

#pragma unroll
    for (int ri = 0; ri < 4; ++ri)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            S[(size_t)(r0 + ri) * TS_GLM_KDA_HD + lane + 32 * j] = st[ri][j];
}

// The same recurrence with one state row per warp, and the next row's inputs loaded while this one computes:
// a prefill ubatch walks its rows in order, so each step waited on its loads and four rows a warp kept three
// quarters of the SMs' warp slots empty. Every state element computes exactly as ts_glm_kda_scan_f32 has it.
// grid (H, 128 / warps, nSeq).
extern "C" __global__ void ts_glm_kda_scan_r1_f32(
    const float* __restrict__ scr,
    float* const* __restrict__ ssmStates, // [nSeq] -> [H, 128, 128]
    float* __restrict__ core,             // [nSeq * rowsPerSeq, H * 128]
    const int rowsPerSeq,
    const int H)
{
    const int h = blockIdx.x;
    const int seq = blockIdx.z;
    const int lane = threadIdx.x & 31;
    const int r = blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= TS_GLM_KDA_HD)
        return;

    const float* base = scr + ((size_t)seq * H + h) * rowsPerSeq * TS_GLM_KDA_SCR;
    float* S = ssmStates[seq] + (size_t)h * TS_GLM_KDA_HD * TS_GLM_KDA_HD;
    const float scale = rsqrtf((float)TS_GLM_KDA_HD);

    float st[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        st[j] = S[(size_t)r * TS_GLM_KDA_HD + lane + 32 * j];

    float nq[4], nk[4], ng[4], nv = 0.0f, nb = 0.0f;
    if (rowsPerSeq > 0)
    {
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            nq[j] = base[lane + 32 * j];
            nk[j] = base[TS_GLM_KDA_HD + lane + 32 * j];
            ng[j] = base[3 * TS_GLM_KDA_HD + lane + 32 * j];
        }
        nv = base[2 * TS_GLM_KDA_HD + r];
        nb = base[4 * TS_GLM_KDA_HD];
    }
    for (int s = 0; s < rowsPerSeq; ++s)
    {
        float qj[4], kj[4], dj[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            qj[j] = nq[j];
            kj[j] = nk[j];
            dj[j] = expf(ng[j]);
        }
        const float v = nv, beta = nb;
        if (s + 1 < rowsPerSeq)
        {
            const float* next = base + (size_t)(s + 1) * TS_GLM_KDA_SCR;
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                nq[j] = next[lane + 32 * j];
                nk[j] = next[TS_GLM_KDA_HD + lane + 32 * j];
                ng[j] = next[3 * TS_GLM_KDA_HD + lane + 32 * j];
            }
            nv = next[2 * TS_GLM_KDA_HD + r];
            nb = next[4 * TS_GLM_KDA_HD];
        }
        float* out = core + ((size_t)seq * rowsPerSeq + s) * H * TS_GLM_KDA_HD + h * TS_GLM_KDA_HD;
        float kvDot = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            st[j] *= dj[j];
            kvDot += st[j] * kj[j];
        }
        kvDot = ts_glm_warp_sum(kvDot);
        const float delta = (v - kvDot) * beta;
        float o = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            st[j] += kj[j] * delta;
            o += st[j] * qj[j];
        }
        o = ts_glm_warp_sum(o);
        if (lane == 0)
            out[r] = o * scale;
    }

#pragma unroll
    for (int j = 0; j < 4; ++j)
        S[(size_t)r * TS_GLM_KDA_HD + lane + 32 * j] = st[j];
}

// One warp per (row, head): RMS norm over the head (one weight shared by every head),
// then the plain sigmoid output gate g_b(g_a(x)) - not SiLU.
extern "C" __global__ void ts_glm_kda_out_f32(
    const float* __restrict__ core,       // [rows, H * 128]
    const float* __restrict__ gate,       // [rows, H * 128]
    const float* __restrict__ normW,      // [128]
    float* __restrict__ out,              // [rows, H * 128]
    const int rows,
    const int H,
    const float eps)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows * H)
        return;
    const size_t off = (size_t)warp * TS_GLM_KDA_HD;   // row-major [row][h][d]
    float x[4];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        x[j] = core[off + lane + 32 * j];
        ss += x[j] * x[j];
    }
    const float inv = rsqrtf(ts_glm_warp_sum(ss) / (float)TS_GLM_KDA_HD + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int d = lane + 32 * j;
        out[off + d] = x[j] * inv * normW[d] * ts_glm_sigmoid(gate[off + d]);
    }
}

// Carry each sequence's last dc-1 convolution inputs into its state: after a ubatch of
// `rowsPerSeq` rows the state holds rows rowsPerSeq-dc+1 .. rowsPerSeq-1 of
// [old state; this ubatch], oldest first. One thread per (sequence, channel), walking
// the state rows upward so a row still to be read is never overwritten first.
extern "C" __global__ void ts_glm_kda_conv_update_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* const* __restrict__ convStates,
    const int rowsPerSeq,
    const int nSeq,
    const int dInner,
    const int dc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int C = 3 * dInner;
    if (i >= nSeq * C)
        return;
    const int seq = i / C;
    const int c = i - seq * C;
    const int which = c / dInner;
    const int ch = c - which * dInner;
    const float* x = which == 0 ? q : which == 1 ? k : v;
    float* state = convStates[seq];
    const int rowBase = seq * rowsPerSeq;
    for (int r = 0; r < dc - 1; ++r)
    {
        const int src = rowsPerSeq + r;          // index into [state (dc-1 rows); ubatch]
        state[(size_t)r * C + c] = src < dc - 1
            ? state[(size_t)src * C + c]
            : x[(size_t)(rowBase + src - (dc - 1)) * dInner + ch];
    }
}

// ---------------------------------------------------------------------------
// MLA (NoPE) and the pooled DSA indexer's cache writes
//
// The MLA layers keep one kv_lora-wide latent per token, which is both the key
// (after the query is absorbed through wk_b) and the value (decompressed after
// the attention through wv_b); nothing in it is rotated. The indexer caches each
// token's LayerNormed key beside its pooling gate, since a pool is only scored
// once its members have left the batch.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float ts_glm_block_sum(float v, float* sh)
{
    v = ts_glm_warp_sum(v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0)
        sh[warp] = v;
    __syncthreads();
    const int warps = blockDim.x >> 5;
    v = lane < warps ? sh[lane] : 0.0f;
    v = ts_glm_warp_sum(v);
    __syncthreads();
    return v;
}

// cache[pos(t)] = half(rms_norm(kv[t]) * w). One block per row; pos(t) = p0 + t, or
// positions[t] (a captured decode step reads its row's position from the device).
extern "C" __global__ void ts_glm_mla_kv_store_f32(
    const float* __restrict__ kv,         // [nt, width]
    const float* __restrict__ normW,      // [width]
    half* __restrict__ cache,             // [nCtx, width]
    const int p0,
    const int width,
    const float eps,
    const int32_t* __restrict__ positions)
{
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* x = kv + (size_t)t * width;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        ss += x[i] * x[i];
    const float inv = rsqrtf(ts_glm_block_sum(ss, sh) / (float)width + eps);
    const int pos = positions != nullptr ? positions[t] : p0 + t;
    half* dst = cache + (size_t)pos * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        dst[i] = __float2half(x[i] * inv * normW[i]);
}

// cache[pos(t)] = [ half(layer_norm(k[t]) * w + b) | half(gate[t]) ]: the indexer key
// (a genuine LayerNorm, weight and bias) and its pooling gate, D each.
extern "C" __global__ void ts_glm_indexer_store_f32(
    const float* __restrict__ key,        // [nt, D]
    const float* __restrict__ gate,       // [nt, D]
    const float* __restrict__ lnW,        // [D]
    const float* __restrict__ lnB,        // [D]
    half* __restrict__ cache,             // [nCtx, 2D]
    const int p0,
    const int D,
    const float eps,
    const int32_t* __restrict__ positions)
{
    __shared__ float sh[32];
    const int t = blockIdx.x;
    const float* x = key + (size_t)t * D;
    float s = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        s += x[i];
    const float mean = ts_glm_block_sum(s, sh) / (float)D;
    float v = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float d = x[i] - mean;
        v += d * d;
    }
    const float inv = rsqrtf(ts_glm_block_sum(v, sh) / (float)D + eps);
    const int pos = positions != nullptr ? positions[t] : p0 + t;
    half* dst = cache + (size_t)pos * 2 * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        dst[i] = __float2half((x[i] - mean) * inv * lnW[i] + lnB[i]);
        dst[D + i] = __float2half(gate[(size_t)t * D + i]);
    }
}

// ---------------------------------------------------------------------------
// Pooled DSA indexer (sparse attention past indexer_top_k cached tokens)
//
// Pools are position-aligned windows of kpool cells. A pool's key is, per channel, the
// softmax over its members' cached gates (plus a per-slot additive embedding) weighting
// their cached keys; it is computed once, when the pool's last member is cached. A query
// scores the pools whose last member it can see - sum over heads of w_h * relu(q_h . key) -
// keeps the best (the DeepSeek top-k kernel), and attends over their cells plus the cells
// of its own, still incomplete, pool. The reference is build_indexer_g5n and
// build_topk_mask_g5n in ggml_ops_glm_dsa.cpp.
// ---------------------------------------------------------------------------

#define TS_GLM_KPOOL 4

// poolKeys[b] for the pools b in [firstPool, firstPool + gridDim.x): one block per pool, one
// thread per channel. idxCache rows are [key (D) | gate (D)] F16; ape is [kpool, D] (slot-major,
// as the file stores [D, kpool] with D fastest). With positions (a captured decode step, one
// block), the pool the row at positions[0] completes, if it completes one.
extern "C" __global__ void ts_glm_pool_keys_f32(
    const half* __restrict__ idxCache,    // [nCtx, 2D]
    const float* __restrict__ ape,        // [kpool, D]
    float* __restrict__ poolKeys,         // [nCtx / kpool, D]
    const int firstPool,
    const int D,
    const int32_t* __restrict__ positions)
{
    int b = firstPool + blockIdx.x;
    if (positions != nullptr)
    {
        const int p = positions[0];
        if ((p + 1) % TS_GLM_KPOOL != 0)
            return;
        b = p / TS_GLM_KPOOL;
    }
    for (int d = threadIdx.x; d < D; d += blockDim.x)
    {
        float logit[TS_GLM_KPOOL], key[TS_GLM_KPOOL];
        float mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < TS_GLM_KPOOL; ++j)
        {
            const half* row = idxCache + ((size_t)b * TS_GLM_KPOOL + j) * 2 * D;
            key[j] = __half2float(row[d]);
            logit[j] = __half2float(row[D + d]) + ape[(size_t)j * D + d];
            mx = fmaxf(mx, logit[j]);
        }
        float sum = 0.0f;
#pragma unroll
        for (int j = 0; j < TS_GLM_KPOOL; ++j)
        {
            logit[j] = expf(logit[j] - mx);
            sum += logit[j];
        }
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < TS_GLM_KPOOL; ++j)
            acc += key[j] * (logit[j] / sum);
        poolKeys[(size_t)b * D + d] = acc;
    }
}

// scores[t, b] = sum_h w[t, h] * relu(q[t, h] . poolKeys[b]) for the pools query t sees (their
// last member at or before its position). One warp per (pool, query), striding over the pools
// (a captured step's grid is sized for the whole context); w carries 1/sqrt(D * H).
extern "C" __global__ void ts_glm_pool_scores_f32(
    const float* __restrict__ q,          // [nt, H, D]
    const float* __restrict__ w,          // [nt, H]
    const float* __restrict__ poolKeys,   // [pools, D]
    float* __restrict__ scores,           // [nt, stride]
    const int p0,
    const int H,
    const int D,
    const int stride,
    const int maxVis,
    const int32_t* __restrict__ positions)
{
    const int t = blockIdx.y;
    const long long pos = positions != nullptr ? (long long)positions[t] : (long long)p0 + t;
    int nVis = (int)((pos + 1) / TS_GLM_KPOOL);
    if (nVis > maxVis)
        nVis = maxVis;
    const float* qt = q + (size_t)t * H * D;
    const float* wt = w + (size_t)t * H;
    for (int b = blockIdx.x * blockDim.y + threadIdx.y; b < nVis; b += gridDim.x * blockDim.y)
    {
        const float* k = poolKeys + (size_t)b * D;
        float score = 0.0f;
        for (int h = 0; h < H; ++h)
        {
            float acc = 0.0f;
            for (int d = threadIdx.x; d < D; d += 32)
                acc += qt[(size_t)h * D + d] * k[d];
            acc = ts_glm_warp_sum(acc);
            if (acc > 0.0f)
                score += acc * wt[h];
        }
        if (threadIdx.x == 0)
            scores[(size_t)t * stride + b] = score;
    }
}

// The cells each query attends over: every cell of its selected pools, then its own pool's
// cells up to itself. One block per query.
extern "C" __global__ void ts_glm_expand_cells_i32(
    const int32_t* __restrict__ sel,      // [nt, K] pool indices
    const int32_t* __restrict__ selCnt,   // [nt]
    int32_t* __restrict__ cells,          // [nt, cellStride]
    int32_t* __restrict__ cellCnt,        // [nt]
    const int p0,
    const int K,
    const int cellStride,
    const int32_t* __restrict__ positions)
{
    const int t = blockIdx.x;
    const int n = selCnt[t];
    const int32_t* s = sel + (size_t)t * K;
    int32_t* out = cells + (size_t)t * cellStride;
    for (int i = threadIdx.x; i < n * TS_GLM_KPOOL; i += blockDim.x)
        out[i] = s[i / TS_GLM_KPOOL] * TS_GLM_KPOOL + i % TS_GLM_KPOOL;
    const long long qpos = positions != nullptr ? (long long)positions[t] : (long long)p0 + t;
    const int tailStart = (int)((qpos + 1) / TS_GLM_KPOOL * TS_GLM_KPOOL);
    const int tail = (int)(qpos + 1 - tailStart);
    if (threadIdx.x < tail)
        out[n * TS_GLM_KPOOL + threadIdx.x] = tailStart + threadIdx.x;
    if (threadIdx.x == 0)
        cellCnt[t] = n * TS_GLM_KPOOL + tail;
}

// ---------------------------------------------------------------------------
// MLA per-head projections for up to 16 rows (decode and batched decode)
//
// y[t, h, j] = sum_k W[h, j, k] * x[t, h, k] over NHead stacked F16 [outDim, inDim] matrices:
// the query absorption through wk_b and the value decompression through wv_b. One warp per
// (head, output), the weights read once and applied to every row; a row's sum is the same
// whatever the row count (prefill's wider batches go through cuBLAS instead).
// ---------------------------------------------------------------------------

template <int MAXR>
__device__ __forceinline__ void ts_glm_head_gemv_rows(
    const half* __restrict__ w,           // [heads, outDim, inDim]
    const float* __restrict__ x,          // [rows, heads, inDim]
    float* __restrict__ y,                // [rows, heads, outDim]
    const int heads,
    const int outDim,
    const int inDim,
    const int rows)
{
    const int h = blockIdx.y;
    const int j = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (j >= outDim)
        return;
    const half2* wr = (const half2*)(w + ((size_t)h * outDim + j) * inDim);
    float acc[MAXR];
#pragma unroll
    for (int r = 0; r < MAXR; ++r)
        acc[r] = 0.0f;
    for (int k2 = lane; k2 < inDim / 2; k2 += 32)
    {
        const float2 wf = __half22float2(wr[k2]);
#pragma unroll
        for (int r = 0; r < MAXR; ++r)
        {
            if (r >= rows)
                break;
            const float2 xv = ((const float2*)(x + ((size_t)r * heads + h) * inDim))[k2];
            acc[r] += wf.x * xv.x + wf.y * xv.y;
        }
    }
#pragma unroll
    for (int r = 0; r < MAXR; ++r)
    {
        if (r >= rows)
            break;
        const float v = ts_glm_warp_sum(acc[r]);
        if (lane == 0)
            y[((size_t)r * heads + h) * outDim + j] = v;
    }
}

extern "C" __global__ void ts_glm_head_gemv_1_f32(const half* w, const float* x, float* y, int heads, int outDim, int inDim, int rows)
{ ts_glm_head_gemv_rows<1>(w, x, y, heads, outDim, inDim, rows); }
extern "C" __global__ void ts_glm_head_gemv_4_f32(const half* w, const float* x, float* y, int heads, int outDim, int inDim, int rows)
{ ts_glm_head_gemv_rows<4>(w, x, y, heads, outDim, inDim, rows); }
extern "C" __global__ void ts_glm_head_gemv_8_f32(const half* w, const float* x, float* y, int heads, int outDim, int inDim, int rows)
{ ts_glm_head_gemv_rows<8>(w, x, y, heads, outDim, inDim, rows); }
extern "C" __global__ void ts_glm_head_gemv_16_f32(const half* w, const float* x, float* y, int heads, int outDim, int inDim, int rows)
{ ts_glm_head_gemv_rows<16>(w, x, y, heads, outDim, inDim, rows); }

// Projected vision rows replace the token embeddings of their positions in every stream:
// xs[firstRow + r, c, :] = rows[r, :] for each of the 4 hyper-connection streams.
extern "C" __global__ void ts_glm_set_stream_rows_f32(
    const float* __restrict__ rows,       // [nRows, E]
    float* __restrict__ xs,               // [nt, 4, E]
    const int firstRow,
    const int E)
{
    const int r = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E)
        return;
    const float v = rows[(size_t)r * E + e];
    float* dst = xs + (size_t)(firstRow + r) * 4 * E + e;
#pragma unroll
    for (int c = 0; c < 4; ++c)
        dst[(size_t)c * E] = v;
}
