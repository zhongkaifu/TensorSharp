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

#include <cuda_fp16.h>
#include <stdint.h>
#include <float.h>

#define TS_GRAPH_KERNELS_VERSION 33

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

// A verify window's scan inputs, kept for a later re-step of its kept rows (one sequence): planar regions of save_rows tokens
// each, q [t][hk][S] | k [t][hk][S] | v [t][h][S] | g [t][h][S or 1] | beta [t][h]. The head's first warp writes q and k
// (heads below hk: their key head) and its gates, every warp the v of its own rows.
__device__ __forceinline__ void ts_graph_gdn_save_token(float* __restrict__ in_save, int save_rows, const float* __restrict__ q,
    const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ g, const float* __restrict__ beta, int S,
    int h, int hk, int kda, int t, int head, int row, int rows, bool first, int lane)
{
    const size_t qk = (size_t)save_rows * hk * S, vs = (size_t)save_rows * h * S, gs = (size_t)save_rows * h * (kda ? S : 1);
    float* qs = in_save;
    float* ks = qs + qk;
    float* vv = ks + qk;
    float* gg = vv + vs;
    float* bb = gg + gs;
    const size_t gb = (size_t)t * h + head;
    if (first)
    {
        if (head < hk)
            for (int i = lane; i < S; i += 32)
            {
                qs[((size_t)t * hk + head) * S + i] = q[((size_t)t * hk + head) * S + i];
                ks[((size_t)t * hk + head) * S + i] = k[((size_t)t * hk + head) * S + i];
            }
        if (kda)
            for (int i = lane; i < S; i += 32) gg[gb * S + i] = g[gb * S + i];
        else if (lane == 0)
            gg[gb] = g[gb];
        if (lane == 0) bb[gb] = beta[gb];
    }
    if (lane == 0)
        for (int r = row; r < row + rows && r < S; r++) vv[gb * S + r] = v[gb * S + r];
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
// saved (null, or [h, S, S]): the state before the rows, written as it is read (one sequence); in_save (null, or the
// planar regions above): the scan inputs of the rows. state may be out's state part (one slot, one sequence): each warp
// reads its row before it writes it.
extern "C" __global__ void ts_graph_gated_delta_net_f32(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta, const float* state,
    float* out, int S, int h, int hk, int n_t, int n_s, int slots, int kda, float* __restrict__ saved,
    float* __restrict__ in_save, int save_rows)
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
    if (saved != nullptr)
    {
        float* s_sv = saved + ((size_t)(seq * h + head) * S + row) * S;
#pragma unroll
        for (int j = 0; j < 8; j++)
            if (j < per) s_sv[lane + 32 * j] = st[j];
    }

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
        if (in_save != nullptr)
            ts_graph_gdn_save_token(in_save, save_rows, q, k, v, g, beta, S, h, hk, kda, t, head, row, 1, row == 0, lane);

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
// The body takes the warp's sequence, head and first state row (r0); ts_graph_gated_delta_net_rows4_f32 and
// ts_graph_gdn_commit_f32 both run it.
__device__ __forceinline__ void ts_graph_gdn_rows4(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta, const float* state,
    float* out, int S, int h, int hk, int n_t, int n_s, int slots, int kda, float* __restrict__ saved,
    float* __restrict__ in_save, int save_rows, int seq, int head, int r0, int lane)
{
    const int kh = head % hk;
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
    if (saved != nullptr)
    {
        float* s_sv = saved + ((size_t)(seq * h + head) * S + r0) * S;
#pragma unroll
        for (int r = 0; r < 4; r++)
        {
            if (r0 + r >= S) break;
#pragma unroll
            for (int j = 0; j < 8; j++)
                if (j < per) s_sv[(size_t)r * S + lane + 32 * j] = st[r][j];
        }
    }

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
        if (in_save != nullptr)
            ts_graph_gdn_save_token(in_save, save_rows, q, k, v, g, beta, S, h, hk, kda, t, head, r0, 4, r0 == 0, lane);

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

extern "C" __global__ void ts_graph_gated_delta_net_rows4_f32(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta, const float* state,
    float* out, int S, int h, int hk, int n_t, int n_s, int slots, int kda, float* __restrict__ saved,
    float* __restrict__ in_save, int save_rows)
{
    ts_graph_gdn_rows4(q, k, v, g, beta, state, out, S, h, hk, n_t, n_s, slots, kda, saved, in_save, save_rows,
        blockIdx.x / h, blockIdx.x % h, (blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 4, threadIdx.x & 31);
}

// A verify window's commit over the gated delta-net layers of a program (GraphFusion.DeltaNetCommit), up to
// TS_GRAPH_GDN_COMMIT_LAYERS layers a launch, the layers' pointers by value. Per layer:
//   steps > 0:  the state after the first `steps` rows of the window's saved scan inputs (planar regions of save_rows
//               tokens: q | k [t][hk][S], v [t][h][S], g | beta [t][h]) from the state before the window (`state`):
//               ts_graph_gdn_rows4 as the decode rows' kernels ran the window (one sequence, one slot, scalar gates), its
//               outputs and the state after the rows written to out ([steps][h][S] | [h][S][S]);
//   steps == 0: the state at `state` copied to out;
//   steps < 0:  the state stays;
// and every layer's conv window conv_to [channels][conv_rows] takes conv_from[c * conv_pitch + j] (the history's columns
// after the kept rows). Sources and destinations are separate memory, or a re-step on the state in place (out's state
// part the state: each warp reads its state rows before it writes them). Launch: grid (h, ceil(S / 16), layers), block 128.
#define TS_GRAPH_GDN_COMMIT_LAYERS 32

struct ts_graph_gdn_commit
{
    const float* state[TS_GRAPH_GDN_COMMIT_LAYERS];
    float* out[TS_GRAPH_GDN_COMMIT_LAYERS];
    const float* inputs[TS_GRAPH_GDN_COMMIT_LAYERS];
    const float* conv_from[TS_GRAPH_GDN_COMMIT_LAYERS];
    float* conv_to[TS_GRAPH_GDN_COMMIT_LAYERS];
    int steps[TS_GRAPH_GDN_COMMIT_LAYERS];
    int layers, S, h, hk, save_rows, conv_rows, channels, conv_pitch;
};

// CudaGraphKernels.GdnCommitArgs lays the launch's argument out byte for byte.
static_assert(sizeof(ts_graph_gdn_commit) == 1440, "ts_graph_gdn_commit: CudaGraphKernels.GdnCommitArgs mirrors this layout");

extern "C" __global__ void __launch_bounds__(128) ts_graph_gdn_commit_f32(const ts_graph_gdn_commit c)
{
    const int layer = blockIdx.z;
    if (layer >= c.layers)
        return;
    // The conv window: every block of the layer copies its share.
    {
        const long long n = (long long)c.channels * c.conv_rows;
        const long long blocks = (long long)gridDim.x * gridDim.y, block = (long long)blockIdx.y * gridDim.x + blockIdx.x;
        const float* __restrict__ from = c.conv_from[layer];
        float* __restrict__ to = c.conv_to[layer];
        for (long long i = block * blockDim.x + threadIdx.x; i < n; i += blocks * blockDim.x)
        {
            const long long ch = i / c.conv_rows;
            to[i] = from[ch * c.conv_pitch + (i - ch * c.conv_rows)];
        }
    }
    const int steps = c.steps[layer], S = c.S, h = c.h, head = blockIdx.x, lane = threadIdx.x & 31;
    const int r0 = (blockIdx.y * (blockDim.x >> 5) + (threadIdx.x >> 5)) * 4;
    if (steps < 0 || r0 >= S)
        return;
    if (steps == 0)
    {
        const float* __restrict__ src = c.state[layer] + ((size_t)head * S + r0) * S;
        float* __restrict__ dst = c.out[layer] + ((size_t)head * S + r0) * S;
        for (int r = 0; r < 4 && r0 + r < S; r++)
            for (int i = lane; i < S; i += 32)
                dst[(size_t)r * S + i] = src[(size_t)r * S + i];
        return;
    }
    const float* in = c.inputs[layer];
    const size_t qk = (size_t)c.save_rows * c.hk * S, vs = (size_t)c.save_rows * h * S, gs = (size_t)c.save_rows * h;
    ts_graph_gdn_rows4(in, in + qk, in + 2 * qk, in + 2 * qk + vs, in + 2 * qk + vs + gs, c.state[layer], c.out[layer],
        S, h, c.hk, steps, 1, 1, 0, nullptr, nullptr, 0, 0, head, r0, lane);
}

// ggml_top_k: the indices of each row's k largest values, largest first, a tie going to the lower index (the order the
// host sort gives). x [rows, cols] (cols <= 4096), ids [rows, k] int32. Launch: grid rows, block 256, cols floats of
// dynamic shared memory.
extern "C" __global__ void ts_graph_topk_f32(const float* __restrict__ x, int* __restrict__ ids, int cols, int k)
{
    extern __shared__ float row[];
    __shared__ float bestV[8];
    __shared__ int bestI[8];
    const size_t r = blockIdx.x;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        row[c] = x[r * cols + c];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int j = 0; j < k; j++)
    {
        float v = -INFINITY;
        int idx = 0x7fffffff;
        for (int c = threadIdx.x; c < cols; c += blockDim.x)
        {
            const float t = row[c];
            if (t > v || (t == v && c < idx)) { v = t; idx = c; }
        }
        for (int off = 16; off > 0; off >>= 1)
        {
            const float ov = __shfl_xor_sync(0xffffffff, v, off);
            const int oi = __shfl_xor_sync(0xffffffff, idx, off);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0) { bestV[warp] = v; bestI[warp] = idx; }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            float bv = bestV[0];
            int bi = bestI[0];
            for (int w = 1; w < warps; w++)
                if (bestV[w] > bv || (bestV[w] == bv && bestI[w] < bi)) { bv = bestV[w]; bi = bestI[w]; }
            if (bi < 0 || bi >= cols) bi = 0;
            ids[r * k + j] = bi;
            row[bi] = -INFINITY;
        }
        __syncthreads();
    }
}

// Floats as unsigned keys in the same order (larger float, larger key); -0 is +0 (equal values tie, as the host sort has
// them).
__device__ __forceinline__ unsigned int ts_order_key(float v)
{
    const unsigned int u = v == 0.0f ? 0u : __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ts_graph_topk_wide_f32 at k = 1 (a greedy verify's token a row over a 248K vocabulary) in two stages, every SM busy where
// one block a row walked the whole row: each block keeps the largest of its slice as (order key << 32) | ~index - the
// encoding the wide kernel orders its picks by, so a larger value wins and an equal one goes to the lower index - and one
// warp a row merges the slices. The same pick bit for bit. Part: grid (slices, rows), block 256, part [rows, slices].
extern "C" __global__ void ts_graph_argmax_part_f32(const float* __restrict__ x, unsigned long long* __restrict__ part, int cols, int slice)
{
    __shared__ unsigned long long warpBest[8];
    const float* row = x + (size_t)blockIdx.y * cols;
    const int c0 = blockIdx.x * slice, c1 = min(cols, c0 + slice);
    unsigned long long best = 0;
    for (int c = c0 + threadIdx.x; c < c1; c += blockDim.x)
    {
        const unsigned long long v = ((unsigned long long)ts_order_key(row[c]) << 32) | (unsigned int)(~c);
        best = v > best ? v : best;
    }
    for (int off = 16; off > 0; off >>= 1)
    {
        const unsigned long long o = __shfl_xor_sync(0xffffffffu, best, off);
        best = o > best ? o : best;
    }
    if ((threadIdx.x & 31) == 0) warpBest[threadIdx.x >> 5] = best;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        for (int w = 1; w < (int)(blockDim.x >> 5); w++) best = warpBest[w] > best ? warpBest[w] : best;
        part[(size_t)blockIdx.y * gridDim.x + blockIdx.x] = best;
    }
}

// The merge: ids[r] = the index of row r's largest slice pick. Launch: grid rows, block 32.
extern "C" __global__ void ts_graph_argmax_merge(const unsigned long long* __restrict__ part, int* __restrict__ ids, int slices)
{
    const unsigned long long* p = part + (size_t)blockIdx.x * slices;
    unsigned long long best = 0;
    for (int s = threadIdx.x; s < slices; s += 32) best = p[s] > best ? p[s] : best;
    for (int off = 16; off > 0; off >>= 1)
    {
        const unsigned long long o = __shfl_xor_sync(0xffffffffu, best, off);
        best = o > best ? o : best;
    }
    if (threadIdx.x == 0) ids[blockIdx.x] = (int)(~(unsigned int)(best & 0xffffffffull));
}

// The block's largest of v over its threads (0 when none), on every thread.
__device__ __forceinline__ unsigned long long ts_graph_block_max_key(unsigned long long v, unsigned long long* warpBest)
{
    for (int off = 16; off > 0; off >>= 1)
    {
        const unsigned long long o = __shfl_xor_sync(0xffffffffu, v, off);
        v = o > v ? o : v;
    }
    if ((threadIdx.x & 31) == 0) warpBest[threadIdx.x >> 5] = v;
    __syncthreads();
    v = warpBest[0];
    for (int w = 1; w < (int)(blockDim.x >> 5); w++) v = warpBest[w] > v ? warpBest[w] : v;
    __syncthreads();
    return v;
}

// ts_graph_topk_wide_f32 for a few picks (k <= 16) over a wide row (a draft step's ten largest of a 151K vocabulary) in two
// stages over every SM, as ts_graph_argmax_part_f32 takes one: each block takes its slice's k largest keys, (order key << 32)
// | ~index - the order the wide kernel sorts its picks in, so a larger value comes first and an equal one goes to the lower
// index - and one block a row merges the slices' picks. The same picks in the same order, bit for bit. Part: grid (slices,
// rows), block 256, part [rows, slices, k] (0: no pick).
extern "C" __global__ void ts_graph_topk_few_part_f32(const float* __restrict__ x, unsigned long long* __restrict__ part, int cols,
    int slice, int k)
{
    __shared__ unsigned long long warpBest[8];
    const float* row = x + (size_t)blockIdx.y * cols;
    const int c0 = blockIdx.x * slice, c1 = min(cols, c0 + slice);
    unsigned long long* out = part + ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * k;
    unsigned long long below = ~0ull;
    for (int p = 0; p < k; p++)
    {
        unsigned long long best = 0;
        for (int c = c0 + threadIdx.x; c < c1; c += blockDim.x)
        {
            const unsigned long long v = ((unsigned long long)ts_order_key(row[c]) << 32) | (unsigned int)(~c);
            if (v < below && v > best) best = v;
        }
        best = ts_graph_block_max_key(best, warpBest);
        if (threadIdx.x == 0) out[p] = best;
        below = best;
    }
}

// The merge: ids[r] = row r's k largest slice picks, largest first. Launch: grid rows, block 256.
extern "C" __global__ void ts_graph_topk_few_merge(const unsigned long long* __restrict__ part, int* __restrict__ ids, int picks, int k)
{
    __shared__ unsigned long long warpBest[8];
    const unsigned long long* p = part + (size_t)blockIdx.x * picks;
    unsigned long long below = ~0ull;
    for (int j = 0; j < k; j++)
    {
        unsigned long long best = 0;
        for (int i = threadIdx.x; i < picks; i += blockDim.x)
        {
            const unsigned long long v = p[i];
            if (v < below && v > best) best = v;
        }
        best = ts_graph_block_max_key(best, warpBest);
        if (threadIdx.x == 0) ids[(size_t)blockIdx.x * k + j] = (int)(~(unsigned int)(best & 0xffffffffull));
        below = best;
    }
}

// ggml_top_k for many picks or wide rows (a sparse attention's blocks): the k-th largest value radix-selected (four 8-bit
// passes over the row), every larger value and the lowest-indexed equal ones taken, and the k picks sorted largest
// first, a tie to the lower index: ts_graph_topk_f32's order and ties. x [rows, cols], ids [rows, k], k <= n2 (a power of
// two). Launch: grid rows, block 1024, n2 * 8 bytes of dynamic shared memory.
extern "C" __global__ void ts_graph_topk_wide_f32(const float* __restrict__ x, int* __restrict__ ids, int cols, int k, int n2)
{
    extern __shared__ unsigned long long picks[];   // (key << 32) | ~index: larger value, then lower index, sorts first
    __shared__ unsigned int hist[256];
    __shared__ unsigned int warpCounts[32];
    __shared__ unsigned int s_prefix, s_remaining, s_taken, s_equalBase;
    const float* row = x + (size_t)blockIdx.x * cols;
    if (threadIdx.x == 0) { s_prefix = 0; s_remaining = (unsigned int)k; s_taken = 0; s_equalBase = 0; }
    unsigned int mask = 0;
    for (int shift = 24; shift >= 0; shift -= 8)
    {
        for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
        __syncthreads();
        const unsigned int prefix = s_prefix;
        for (int c = threadIdx.x; c < cols; c += blockDim.x)
        {
            const unsigned int key = ts_order_key(row[c]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            unsigned int remaining = s_remaining, b = 255;
            while (b > 0 && hist[b] < remaining) { remaining -= hist[b]; b--; }
            s_prefix = prefix | (b << shift);
            s_remaining = remaining;
        }
        mask |= 255u << shift;
        __syncthreads();
    }
    const unsigned int thr = s_prefix, need = s_remaining;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
    {
        const unsigned int key = ts_order_key(row[c]);
        if (key > thr) picks[atomicAdd(&s_taken, 1u)] = ((unsigned long long)key << 32) | (unsigned int)(~c);
    }
    __syncthreads();
    // The values equal to the k-th: the lowest-indexed ones, a block-wide scan over each stride of the row.
    const unsigned int greater = s_taken;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int base = 0; base < cols; base += blockDim.x)
    {
        const unsigned int done = s_equalBase;
        if (done >= need) break;   // uniform: read after the barrier below (or the one above)
        const int c = base + threadIdx.x;
        const bool eq = c < cols && ts_order_key(row[c]) == thr;
        const unsigned int ballot = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) warpCounts[warp] = __popc(ballot);
        __syncthreads();
        unsigned int before = done + __popc(ballot & ((1u << lane) - 1u));
        for (int w = 0; w < warp; w++) before += warpCounts[w];
        if (eq && before < need) picks[greater + before] = ((unsigned long long)thr << 32) | (unsigned int)(~c);
        __syncthreads();
        if (threadIdx.x == 0)
        {
            unsigned int total = 0;
            for (int w = 0; w < warps; w++) total += warpCounts[w];
            s_equalBase = done + total;
        }
        __syncthreads();
    }
    for (int i = k + threadIdx.x; i < n2; i += blockDim.x) picks[i] = 0ull;
    __syncthreads();
    // Bitonic sort, largest first.
    for (int size = 2; size <= n2; size <<= 1)
        for (int stride = size >> 1; stride > 0; stride >>= 1)
        {
            for (int i = threadIdx.x; i < n2; i += blockDim.x)
            {
                const int j = i ^ stride;
                if (j > i)
                {
                    const unsigned long long a = picks[i], b = picks[j];
                    if ((i & size) == 0 ? a < b : a > b) { picks[i] = b; picks[j] = a; }
                }
            }
            __syncthreads();
        }
    for (int i = threadIdx.x; i < k; i += blockDim.x)
        ids[(size_t)blockIdx.x * k + i] = (int)~(unsigned int)(picks[i] & 0xffffffffull);
}

// ggml_get_rows by device int32 indices into F32 rows: src [batches, rowsPerBatch, rowLen] (F32 or F16) with rows
// rowPitch elements apart (a strided view's; batches > 1 only for packed rows), ids [batches, n];
// out [batches, n, rowLen] = src[b, ids[b, i], :]. Launch: grid batches * n, block 256.
extern "C" __global__ void ts_graph_get_rows_f32(const float* __restrict__ src, const int* __restrict__ ids,
    float* __restrict__ out, int rowLen, int n, int rowsPerBatch, long long rowPitch)
{
    const size_t o = blockIdx.x;
    const size_t b = o / n;
    const size_t from = (b * rowsPerBatch + ids[o]) * (size_t)rowPitch;
    for (int c = threadIdx.x; c < rowLen; c += blockDim.x)
        out[o * rowLen + c] = src[from + c];
}

extern "C" __global__ void ts_graph_get_rows_f16(const __half* __restrict__ src, const int* __restrict__ ids,
    float* __restrict__ out, int rowLen, int n, int rowsPerBatch, long long rowPitch)
{
    const size_t o = blockIdx.x;
    const size_t b = o / n;
    const size_t from = (b * rowsPerBatch + ids[o]) * (size_t)rowPitch;
    for (int c = threadIdx.x; c < rowLen; c += blockDim.x)
        out[o * rowLen + c] = __half2float(src[from + c]);
}

// ggml_set_rows by device int32 indices: dst row ids[i] (F32 or F16 rows of rowLen, dstPitch elements apart, dstRows of
// them) takes src row i (contiguous F32; F16 rounds to nearest even); an index out of range writes nothing.
// Launch: grid count, block 256.
extern "C" __global__ void ts_graph_set_rows_f32(const float* __restrict__ src, const int* __restrict__ ids, void* __restrict__ dst,
    int rowLen, long long dstPitch, int dstRows, int half)
{
    const size_t i = blockIdx.x;
    const int row = ids[i];
    if (row < 0 || row >= dstRows) return;
    const size_t to = (size_t)row * (size_t)dstPitch;
    for (int c = threadIdx.x; c < rowLen; c += blockDim.x)
    {
        const float v = src[i * rowLen + c];
        if (half) ((__half*)dst)[to + c] = __float2half_rn(v);
        else ((float*)dst)[to + c] = v;
    }
}

// ggml_lightning_indexer: out[r, j] = sum over heads h of w[r, h] * relu(q[r, h, :] . keys[j, :]) + mask[r % maskRows, j].
// q [rows, heads, d], keys [n, d], w [rows, heads], mask [maskRows, n], all F32; d <= 256. Launch: grid (ceil(n / 8), rows),
// block 256 (a warp a key), heads * d floats of dynamic shared memory.
extern "C" __global__ void ts_graph_lightning_indexer_f32(const float* __restrict__ q, const float* __restrict__ keys,
    const float* __restrict__ w, const float* __restrict__ mask, float* __restrict__ out, int d, int heads, int n, int maskRows)
{
    extern __shared__ float sq[];
    const size_t r = blockIdx.y;
    for (int i = threadIdx.x; i < heads * d; i += blockDim.x)
        sq[i] = q[r * heads * d + i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (j >= n) return;
    float kr[8];
#pragma unroll
    for (int t = 0; t < 8; t++)
    {
        const int i = lane + 32 * t;
        kr[t] = i < d ? keys[(size_t)j * d + i] : 0.f;
    }
    float sum = 0.f;
    for (int h = 0; h < heads; h++)
    {
        float dot = 0.f;
#pragma unroll
        for (int t = 0; t < 8; t++)
        {
            const int i = lane + 32 * t;
            if (i < d) dot += sq[h * d + i] * kr[t];
        }
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xffffffffu, dot, off);
        if (dot > 0.f) sum += w[r * heads + h] * dot;
    }
    if (lane == 0)
        out[r * n + j] = sum + mask[(r % maskRows) * n + j];
}

// The cells each row of a mask keeps (above -inf), in order: cells [rows, n], count [rows] (a sparse attention's keys).
// Launch: grid rows, block 1024.
extern "C" __global__ void ts_graph_mask_cells(const float* __restrict__ mask, int* __restrict__ cells, int* __restrict__ count, int n)
{
    __shared__ unsigned int warpCounts[32];
    __shared__ unsigned int s_base;
    const float* m = mask + (size_t)blockIdx.x * n;
    int* out = cells + (size_t)blockIdx.x * n;
    if (threadIdx.x == 0) s_base = 0;
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int base = 0; base < n; base += blockDim.x)
    {
        const int c = base + threadIdx.x;
        const bool keep = c < n && m[c] != -INFINITY;
        const unsigned int ballot = __ballot_sync(0xffffffffu, keep);
        if (lane == 0) warpCounts[warp] = __popc(ballot);
        __syncthreads();
        unsigned int at = s_base + __popc(ballot & ((1u << lane) - 1u));
        for (int w = 0; w < warp; w++) at += warpCounts[w];
        if (keep) out[at] = c;
        __syncthreads();
        if (threadIdx.x == 0)
        {
            unsigned int total = 0;
            for (int w = 0; w < warps; w++) total += warpCounts[w];
            s_base += total;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) count[blockIdx.x] = (int)s_base;
}

// ggml_top_k_mask: out[r, c] = mask[r % maskRows, c] at the cells row r keeps, -inf elsewhere. Kept: each pick's cells
// (picks [rows, top]; a pick out of range, or whose score [rows, nPick] is -inf, keeps none; a pick's cells are its row of
// blocks [nBlocks, cells] when given, else the pick itself) and extra [rows, nExtra]; cells outside [0, nkv) are skipped.
// score, blocks and extra may be null. Launch: grid rows, block 256.
extern "C" __global__ void ts_graph_topk_mask_f32(const float* __restrict__ mask, const int* __restrict__ picks,
    const float* __restrict__ score, const int* __restrict__ blocks, const int* __restrict__ extra, float* __restrict__ out,
    int nkv, int maskRows, int top, int nPick, int cells, int nBlocks, int nExtra)
{
    const size_t r = blockIdx.x;
    float* o = out + r * nkv;
    const float* m = mask + (r % maskRows) * nkv;
    for (int c = threadIdx.x; c < nkv; c += blockDim.x)
        o[c] = -INFINITY;
    __syncthreads();
    const int perPick = blocks ? cells : 1;
    for (int i = threadIdx.x; i < top * perPick; i += blockDim.x)
    {
        const int pick = picks[r * top + i / perPick];
        if (score && (pick < 0 || pick >= nPick || score[r * nPick + pick] == -INFINITY)) continue;
        int c = pick;
        if (blocks)
        {
            if (pick < 0 || pick >= nBlocks) continue;
            c = blocks[(size_t)pick * cells + i % perPick];
        }
        if (c >= 0 && c < nkv) o[c] = m[c];
    }
    if (extra)
        for (int e = threadIdx.x; e < nExtra; e += blockDim.x)
        {
            const int c = extra[r * nExtra + e];
            if (c >= 0 && c < nkv) o[c] = m[c];
        }
}

// ggml_cpy between F16 and F32 over n contiguous elements (round to nearest even into F16).
extern "C" __global__ void ts_graph_f16_to_f32(const __half* __restrict__ x, float* __restrict__ y, long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = __half2float(x[i]);
}

extern "C" __global__ void ts_graph_f32_to_f16(const float* __restrict__ x, __half* __restrict__ y, long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = __float2half_rn(x[i]);
}

// ---- host experts on a recorded step (GraphOp.MoeHostExperts, the doorbell of the Qwen3.8 engines): a layer's routes
// to experts its device tier holds run on the device, the others on the host while the device works, without the step
// returning to the host. ts_graph_moe_route_split writes the device's routes (a host expert's id becomes -1, which the
// MoE kernels skip) and, when some route is the host's, publishes the layer's input and routes into host-mapped memory
// and advances the counter; ts_graph_moe_wait_add then waits for the host's flag and adds its sum. Both read the counter
// from memory, so a replayed graph carries no stale value; an all-device layer rings nothing (missDev = 0).
// hdr: [0] layer, [1] slots, [2] 1 (routes published). Launch route_split <<<1, 1024>>>.
// The route-tail skip: host route i (rank i % perRow) at tailRank or lower is left out when no row routes its expert
// higher (0: off).
__device__ __forceinline__ bool ts_graph_tail_dropped(const int* __restrict__ sel, int slots, int perRow, int i, int tailRank)
{
    if (tailRank <= 0 || i % perRow < tailRank) return false;
    const int e = sel[i];
    for (int j = 0; j < slots; j++)
        if (sel[j] == e && j % perRow < tailRank) return false;
    return true;
}

// onDevice[e]: 1 the device tier holds expert e, 2 the host does but its bytes are mapped into the device's address space,
// 0 the host alone can read it. Up to gpuMisses distinct experts of the 2s among the call's host routes (the first in slot
// order; the tail skip's leave none) are the device's too (the Qwen3.8 engines' hybrid misses): each gets staging slot k
// (stageList[k], *stageCount), its table entries point there, ts_graph_moe_stage_copy brings its bytes over the link with
// coalesced loads (the expert kernels' own row reads over the link are small scattered transactions), and
// ts_graph_moe_wait_add restores the entries; a layer whose misses they all are rings nothing.
#define TS_GRAPH_MAX_GPU_MISSES 16
extern "C" __global__ void ts_graph_moe_route_split(
    const float* __restrict__ x, const int* __restrict__ sel, const float* __restrict__ selW,
    const unsigned char* __restrict__ onDevice, int n, int rows, int perRow, int layer, int tailRank, int* __restrict__ devSel,
    unsigned int* __restrict__ counts, float* xOut, int* idsOut, float* wOut, int* hdr, unsigned int* seq, int* missDev,
    int gpuMisses, unsigned long long* __restrict__ tabGate, unsigned long long* __restrict__ tabUp,
    unsigned long long* __restrict__ tabDown, unsigned long long staging, long long stride, long long offUp, long long offDown,
    int* __restrict__ stageList, int* __restrict__ stageCount)
{
    // rows of perRow routes each, a route's rank its slot in its row; every route counted for the routing profile.
    const int slots = rows * perRow;
    __shared__ int taken[TS_GRAPH_MAX_GPU_MISSES];
    __shared__ int takenCount;
    if (threadIdx.x == 0)
    {
        int c = 0;
        const int cap = min(gpuMisses, TS_GRAPH_MAX_GPU_MISSES);
        for (int i = 0; i < slots && c < cap; i++)
        {
            const int e = sel[i];
            if (e < 0 || onDevice[e] != 2 || ts_graph_tail_dropped(sel, slots, perRow, i, tailRank)) continue;
            bool seen = false;
            for (int j = 0; j < c; j++) seen |= taken[j] == e;
            if (!seen) taken[c++] = e;
        }
        for (int j = 0; j < c; j++)
        {
            const int e = taken[j];
            const unsigned long long base = staging + (unsigned long long)j * (unsigned long long)stride;
            tabGate[e] = base;
            tabUp[e] = base + (unsigned long long)offUp;
            tabDown[e] = base + (unsigned long long)offDown;
            stageList[j] = e;
        }
        if (stageCount != nullptr) *stageCount = c;
        takenCount = c;
    }
    __syncthreads();
    int miss = 0;
    for (int i = threadIdx.x; i < slots; i += blockDim.x)
    {
        const int e = sel[i];
        if (e >= 0 && counts != nullptr) atomicAdd(&counts[e], 1u);
        bool device = e >= 0 && onDevice[e] == 1;
        if (e >= 0 && !device && onDevice[e] == 2)
            for (int j = 0; j < takenCount; j++) device |= taken[j] == e;
        devSel[i] = device ? e : -1;
        if (e >= 0 && !device && !ts_graph_tail_dropped(sel, slots, perRow, i, tailRank))
            miss = 1;
    }
    miss = __syncthreads_or(miss);
    if (threadIdx.x == 0)
        *missDev = miss;
    if (!miss)
        return;
    // A route the device runs, or the tail skip leaves out, is published as -(e + 2) (the host records it and skips it), so
    // the two sides split the routes by this table alone.
    for (int i = threadIdx.x; i < slots; i += blockDim.x)
    {
        const int e = sel[i];
        idsOut[i] = e >= 0 && (devSel[i] >= 0 || ts_graph_tail_dropped(sel, slots, perRow, i, tailRank)) ? -(e + 2) : e;
        wOut[i] = selW[i];
    }
    for (int i = threadIdx.x; i < rows * n; i += blockDim.x)
        xOut[i] = x[i];
    if (threadIdx.x == 0)
    {
        ((volatile int*)hdr)[0] = layer;
        ((volatile int*)hdr)[1] = slots;
        ((volatile int*)hdr)[2] = 1;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        *(volatile unsigned int*)seq = *(volatile unsigned int*)seq + 1u;
        __threadfence_system();
    }
}

// The experts ts_graph_moe_route_split staged: their bytes from the host's (device-mapped) memory into their slots with
// coalesced 16-byte loads. grid (capacity * chunks, 3 matrices); every byte count and address is 16-byte aligned.
extern "C" __global__ void ts_graph_moe_stage_copy(const int* __restrict__ stageList, const int* __restrict__ stageCount,
    const unsigned long long* __restrict__ srcGate, const unsigned long long* __restrict__ srcUp,
    const unsigned long long* __restrict__ srcDown, long long bytesGate, long long bytesUp, long long bytesDown,
    unsigned char* __restrict__ staging, long long stride, long long offUp, long long offDown, int chunks)
{
    const int k = blockIdx.x / chunks;
    if (k >= *stageCount) return;
    const int e = stageList[k], part = blockIdx.y;
    const long long bytes = part == 0 ? bytesGate : part == 1 ? bytesUp : bytesDown;
    const uint4* src = reinterpret_cast<const uint4*>(part == 0 ? srcGate[e] : part == 1 ? srcUp[e] : srcDown[e]);
    uint4* dst = reinterpret_cast<uint4*>(staging + (size_t)k * (size_t)stride + (part == 0 ? 0 : part == 1 ? offUp : offDown));
    const long long units = bytes / 16, per = (units + chunks - 1) / chunks;
    const long long begin = (long long)(blockIdx.x % chunks) * per, end = min(units, begin + per);
    for (long long u = begin + threadIdx.x; u < end; u += blockDim.x)
        dst[u] = src[u];
}

// Every block waits for the host's flag to reach the counter, then adds its part of the host's sum (missDev 0: nothing
// was rung; count 0: the host had nothing to add). With stageList, block 0 first points the staged experts' table entries
// back at their resting places (src*) and clears the count (the expert kernels that read the staging ran before).
extern "C" __global__ void ts_graph_moe_wait_add(const unsigned int* flag, const unsigned int* seq, const int* count,
    const float* hostOut, float* __restrict__ dst, int n, const int* missDev, const int* __restrict__ stageList,
    int* __restrict__ stageCount, unsigned long long* __restrict__ tabGate, unsigned long long* __restrict__ tabUp,
    unsigned long long* __restrict__ tabDown, const unsigned long long* __restrict__ srcGate,
    const unsigned long long* __restrict__ srcUp, const unsigned long long* __restrict__ srcDown)
{
    __shared__ int add;
    if (stageList != nullptr && blockIdx.x == 0)
    {
        const int staged = *stageCount;
        for (int i = threadIdx.x; i < staged; i += blockDim.x)
        {
            const int e = stageList[i];
            tabGate[e] = srcGate[e];
            tabUp[e] = srcUp[e];
            tabDown[e] = srcDown[e];
        }
        __syncthreads();
        if (threadIdx.x == 0) *stageCount = 0;
    }
    if (threadIdx.x == 0)
    {
        add = *missDev;
        if (add)
        {
            const unsigned int want = *(const volatile unsigned int*)seq;
            while (*(const volatile unsigned int*)flag != want)
            {
            }
            __threadfence_system();
            add = *(const volatile int*)count;
        }
    }
    __syncthreads();
    if (!add)
        return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        dst[i] += ((const volatile float*)hostOut)[i];
}

// ---- hyper-connections (GraphFusion.HcRead / HcScatter): the residual streams [rows][hc][n].
// The read's grouped norm: each stream of each row RMS-normed (eps inside the root), times the [hc * n] gammas.
// Launch: grid rows * hc, block a multiple of 32 up to 1024 (a decode row's four streams take 1024: four blocks of 256
// left the step's norm latency-bound).
// ggml's block_q8_1 (tensorsharp_kernels' ts_block_q8_1): the scale, the scale times the sum of the values, 32 int8 values.
struct ts_graph_block_q8_1
{
    half d;
    half s;
    int8_t qs[32];
};

// A warp's 32 values (lane i holding value i of a block) into q8_1 exactly as ts_quantize_q8_1_rows_warp_f32 rounds them, so
// a kernel that produces a product's input can write the quantized copy the product reads, and the product skips its own
// quantizing pass (the decode-size products read q8_1 activations).
__device__ __forceinline__ void ts_graph_quantize_q8_1(float x, ts_graph_block_q8_1* dst, int lane)
{
    float amax = fmaxf(0.0f, fabsf(x));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(0xFFFFFFFF, amax, offset));
    amax = __shfl_sync(0xFFFFFFFF, amax, 0);
    const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
    const float id = d > 0.0f ? 1.0f / d : 0.0f;
    int q = (int)rintf(x * id);
    q = max(-127, min(127, q));
    dst->qs[lane] = (int8_t)q;
    int sum = q;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    if (lane == 0)
    {
        dst->d = __float2half_rn(d);
        dst->s = __float2half_rn(d * (float)sum);
    }
}

// yq (null, or n % 32 == 0 and a block a multiple of 32 threads): y's rows also as q8_1.
// injW (null, or hc <= 8): the read's injection weights [hc, hc * n] as F32; injPart[s * hc + o] is then this stream's slice of
// the normed row dotted with row o's slice, which ts_graph_hc_gate_mix_f32 sums over the row's streams in order (the injection
// logits without a product of their own: the Qwen3.8 engines' HcPartials fold them into the norm's pass the same way).
extern "C" __global__ void ts_graph_hc_norm_f32(const float* __restrict__ x, const float* __restrict__ gamma, float* __restrict__ y,
    int n, int hc, float eps, ts_graph_block_q8_1* __restrict__ yq, const float* __restrict__ injW, float* __restrict__ injPart)
{
    __shared__ float part[32];
    __shared__ float ipart[32][8];
    const size_t s = blockIdx.x;               // row * hc + stream
    const float* xs = x + s * n;
    const float* g = gamma + (s % hc) * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        ss += xs[i] * xs[i];
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_xor_sync(0xffffffffu, ss, off);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float t = threadIdx.x < (int)(blockDim.x >> 5) ? part[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            t += __shfl_xor_sync(0xffffffffu, t, off);
        if (threadIdx.x == 0) part[0] = t;
    }
    __syncthreads();
    const float scale = rsqrtf(part[0] / n + eps);
    const size_t slice = (s % hc) * (size_t)n, rowWidth = (size_t)hc * n;
    float acc[8] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        const float v = xs[i] * scale * g[i];
        y[s * n + i] = v;
        if (yq != nullptr)
            ts_graph_quantize_q8_1(v, yq + (s * n + i) / 32, threadIdx.x & 31);
        if (injW != nullptr)
        {
#pragma unroll
            for (int o = 0; o < 8; o++)
                if (o < hc) acc[o] += v * injW[o * rowWidth + slice + i];
        }
    }
    if (injW == nullptr) return;
#pragma unroll
    for (int o = 0; o < 8; o++)
    {
        float t = acc[o];
        for (int off = 16; off > 0; off >>= 1)
            t += __shfl_xor_sync(0xffffffffu, t, off);
        if ((threadIdx.x & 31) == 0) ipart[threadIdx.x >> 5][o] = t;
    }
    __syncthreads();
    if (threadIdx.x < 32)
        for (int o = 0; o < hc; o++)
        {
            float t = threadIdx.x < (int)(blockDim.x >> 5) ? ipart[threadIdx.x][o] : 0.f;
            for (int off = 16; off > 0; off >>= 1)
                t += __shfl_xor_sync(0xffffffffu, t, off);
            if (threadIdx.x == 0) injPart[s * hc + o] = t;
        }
}

// The read's grouped norm at decode size in two passes over slices of 256 values (the Qwen3.8 engines' HcPartials and
// HcApply): one block a row's stream (ts_graph_hc_norm_f32) leaves all but rows * hc SMs idle. n % 256 == 0, hc <= 8,
// 256 threads, grid (hc * n / 256, rows); a row's sums run in one order whatever the row count.
#define TS_GRAPH_HC_SLICE 256
#define TS_GRAPH_HC_PARTIAL 9

// Pass 1, per (row, slice): the slice's sum of squares and, with injW, sum_j x_j * gamma_j * injW[o][j] for each output o,
// into partials [rows][slices][9].
// A slice's partials of the value v each thread holds (pass 1's arithmetic, shared by the kernels that compute v).
__device__ __forceinline__ void ts_graph_hc_partials_of(const float v, const float* __restrict__ gamma,
    const float* __restrict__ injW, float* __restrict__ partials, const int j, const int width, const int hc, const int slice,
    const int r, float (*sh)[8])
{
    float part[TS_GRAPH_HC_PARTIAL];
    part[0] = v * v;
    const float vg = injW != nullptr ? v * gamma[j] : 0.f;
#pragma unroll
    for (int o = 0; o < 8; o++)
        part[1 + o] = injW != nullptr && o < hc ? vg * injW[(size_t)o * width + j] : 0.f;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int p = 0; p < TS_GRAPH_HC_PARTIAL; p++)
    {
        const float t = ts_graph_warp_sum(part[p]);
        if (lane == 0) sh[p][warp] = t;
    }
    __syncthreads();
    if (threadIdx.x < TS_GRAPH_HC_PARTIAL)
    {
        float t = 0.f;
        for (int w = 0; w < TS_GRAPH_HC_SLICE / 32; w++)
            t += sh[threadIdx.x][w];
        partials[((size_t)r * gridDim.x + slice) * TS_GRAPH_HC_PARTIAL + threadIdx.x] = t;
    }
}

extern "C" __global__ void ts_graph_hc_partials_f32(const float* __restrict__ x, const float* __restrict__ gamma,
    const float* __restrict__ injW, float* __restrict__ partials, int n, int hc)
{
    __shared__ float sh[TS_GRAPH_HC_PARTIAL][8];
    const int slice = blockIdx.x, r = blockIdx.y, width = hc * n;
    const int j = slice * TS_GRAPH_HC_SLICE + threadIdx.x;
    ts_graph_hc_partials_of(x[(size_t)r * width + j], gamma, injW, partials, j, width, hc, slice, r, sh);
}

// A block's scatter (ts_graph_hc_scatter_f32's arithmetic, into out) fused with the next read's pass 1 over the scattered
// streams (the Qwen3.8 engines' hc_post_partials): a decode step's two halves of a layer meet in one launch instead of two.
// grid (hc * n / 256, rows), 256 threads.
extern "C" __global__ void ts_graph_hc_post_partials_f32(const float* __restrict__ stream, const float* __restrict__ block,
    const float* __restrict__ inject, float* __restrict__ out, const float* __restrict__ gamma, const float* __restrict__ injW,
    float* __restrict__ partials, int n, int hc)
{
    __shared__ float sh[TS_GRAPH_HC_PARTIAL][8];
    const int slice = blockIdx.x, r = blockIdx.y, width = hc * n;
    const int j = slice * TS_GRAPH_HC_SLICE + threadIdx.x;
    const int c = j / n, i = j - c * n;
    const float w = 2.f * (1.f / (1.f + expf(-inject[r * hc + c] * (1.f / hc))));
    const float v = stream[(size_t)r * width + j] + block[(size_t)r * n + i] * w;
    out[(size_t)r * width + j] = v;
    ts_graph_hc_partials_of(v, gamma, injW, partials, j, width, hc, slice, r, sh);
}

// Pass 2: y = x * rsqrt(mean of its stream's squares + eps) * gamma (and yq, its q8_1 copy, when not null); with inject (not
// null) the slice-0 block also writes inject[r][o] = sum over streams c, in order, of inv_c * (stream c's partials of o).
extern "C" __global__ void ts_graph_hc_apply_f32(const float* __restrict__ x, const float* __restrict__ gamma,
    const float* __restrict__ partials, float* __restrict__ y, ts_graph_block_q8_1* __restrict__ yq, float* __restrict__ inject,
    int n, int hc, float eps)
{
    __shared__ float inv[8];
    const int slice = blockIdx.x, r = blockIdx.y, width = hc * n, per = n / TS_GRAPH_HC_SLICE;
    const float* p = partials + (size_t)r * gridDim.x * TS_GRAPH_HC_PARTIAL;
    if (threadIdx.x < hc)
    {
        float ss = 0.f;
        for (int s = 0; s < per; s++)
            ss += p[((size_t)threadIdx.x * per + s) * TS_GRAPH_HC_PARTIAL];
        inv[threadIdx.x] = rsqrtf(ss / n + eps);
    }
    __syncthreads();
    const int j = slice * TS_GRAPH_HC_SLICE + threadIdx.x;
    const float v = x[(size_t)r * width + j] * inv[j / n] * gamma[j];
    y[(size_t)r * width + j] = v;
    if (yq != nullptr)
        ts_graph_quantize_q8_1(v, yq + ((size_t)r * width + j) / 32, threadIdx.x & 31);
    if (inject != nullptr && slice == 0 && threadIdx.x < hc)
    {
        float acc = 0.f;
        for (int c = 0; c < hc; c++)
        {
            float pc = 0.f;
            for (int s = 0; s < per; s++)
                pc += p[((size_t)c * per + s) * TS_GRAPH_HC_PARTIAL + 1 + threadIdx.x];
            acc += inv[c] * pc;
        }
        inject[(size_t)r * hc + threadIdx.x] = acc;
    }
}


// ---------------------------------------------------------------------------
// A Q6_K vocabulary head (an LM head) on the tiled layout of the Qwen3.5 CUDA engine's head kernels
// (ts_q35_repack_q6k_tiled, ts_q35_gemm_q6kt). A 210-byte superblock gives a row's pieces no 16-byte boundary, so the
// stored rows load 2 bytes at a time; the tiled layout keeps, for each tile of 64 rows and each superblock, one
// 13,440-byte chunk (the 64 superblocks' own bytes):
//   half 0: ql [64][64] | qh [64][32] | sc [64][8] | d [64] (half)    (6,784 bytes)
//   half 1: ql [64][64] | qh [64][32] | sc [64][8]                    (6,656 bytes)
// so a block streams each half superblock of its 64 rows as one contiguous, 16-byte-aligned run. 64 rows of 210 * (k / 256)
// bytes are exactly a tile's chunks, so a head of whole tiles is regrouped in place, tile by tile, through a copy.
// ---------------------------------------------------------------------------
#define TS_GRAPH_Q6KT_ROWS 64
#define TS_GRAPH_Q6KT_CHUNK 13440
#define TS_GRAPH_Q6KT_H1 6784
#define TS_GRAPH_Q6KT_QH 4096
#define TS_GRAPH_Q6KT_SC 6144
#define TS_GRAPH_Q6KT_D 6656

__device__ __forceinline__ int ts_graph_q6kt_ql(int h, int rr) { return h * TS_GRAPH_Q6KT_H1 + rr * 64; }
__device__ __forceinline__ int ts_graph_q6kt_qh(int h, int rr) { return h * TS_GRAPH_Q6KT_H1 + TS_GRAPH_Q6KT_QH + rr * 32; }
__device__ __forceinline__ int ts_graph_q6kt_sc(int h, int rr) { return h * TS_GRAPH_Q6KT_H1 + TS_GRAPH_Q6KT_SC + rr * 8; }

// `rows` stored Q6_K rows [rows][srcRowBytes] (from row rowBase of the head) into the tiled layout at dst (the head's base).
// One thread per (row, superblock). Launch: ceil(rows * k / 256 / 256), 256.
extern "C" __global__ void ts_graph_repack_q6k_tiled(
    const uint8_t* __restrict__ src, long long srcRowBytes, int rows, int k, int rowBase, uint8_t* __restrict__ dst)
{
    const int nsb = k / 256;
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)rows * nsb)
        return;
    const long long rs = i / nsb;
    const int b = (int)(i % nsb);
    const long long r = rowBase + rs;
    const int rr = (int)(r % TS_GRAPH_Q6KT_ROWS);
    const uint8_t* blk = src + rs * srcRowBytes + (long long)b * 210;
    uint8_t* chunk = dst + (r / TS_GRAPH_Q6KT_ROWS * nsb + b) * TS_GRAPH_Q6KT_CHUNK;
#pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        for (int j = 0; j < 64; ++j)
            chunk[ts_graph_q6kt_ql(h, rr) + j] = blk[64 * h + j];
        for (int j = 0; j < 32; ++j)
            chunk[ts_graph_q6kt_qh(h, rr) + j] = blk[128 + 32 * h + j];
        for (int j = 0; j < 8; ++j)
            chunk[ts_graph_q6kt_sc(h, rr) + j] = blk[192 + 8 * h + j];
    }
    chunk[TS_GRAPH_Q6KT_D + 2 * rr] = blk[208];
    chunk[TS_GRAPH_Q6KT_D + 2 * rr + 1] = blk[209];
}

// cp.async and the int8 tensor-core product need sm_80; below it (the compute_75 baseline) they trap, and the caps' image
// architecture keeps the head on its stored rows there.
__device__ __forceinline__ void ts_graph_cp16(void* dst, const void* src, bool valid)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned saddr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(saddr), "l"(src), "r"(valid ? 16 : 0));
#else
    __trap();
#endif
}

__device__ __forceinline__ void ts_graph_cp_commit()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template <int N>
__device__ __forceinline__ void ts_graph_cp_wait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

__device__ __forceinline__ void ts_graph_mma_s8(int (&c)[4], int a0, int a1, int b0)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(0), "r"(0), "r"(0), "r"(0));
#else
    __trap();
#endif
}

// (float)c for |c| < 2^22 without the quarter-rate I2F: the same value, exactly.
__device__ __forceinline__ float ts_graph_i2f_small(int c)
{
    return __int_as_float(c + 0x4B400000) - 12582912.0f;
}

// 64 weight rows (four warps of 16 each) against NT * 8 activation rows a block, one mma.m16n8k16.s8 per 16-value sub-block.
// A stage is half a superblock of the tile and of the activations, three deep; shared rows are padded (ql 80, qh 48,
// activations 144 bytes) so the fragment loads hit 32 banks. Each output's sums run over the head's superblocks in order
// whatever NT and the row count: a row's logits are the same alone or in a window of up to 64 (padding rows read zeros).
// The activations are q8_1 in the split layout (ts_quantize_q8_1_split_rows_f32): int8 [n][k], scales [n][k / 32].
template <int NT>
struct TsGraphQ6ktStage
{
    static constexpr int BR = TS_GRAPH_Q6KT_ROWS, NA = NT * 8, QL = 80, QH = 48, SC = 8, XA = 144;
    uint8_t ql[BR * QL];
    uint8_t qh[BR * QH];
    int8_t sc[BR * SC];
    half d[BR];
    int8_t x[NA * XA];
    float xd[NA * 4];
};

template <int NT>
__device__ __forceinline__ void ts_graph_gemm_q6kt(
    const uint8_t* __restrict__ w, int outDim, int k, const int8_t* __restrict__ xq, const float* __restrict__ xd, int n,
    float* __restrict__ y)
{
    constexpr int STAGES = 3;
    using Stage = TsGraphQ6ktStage<NT>;
    constexpr int BR = Stage::BR, NA = Stage::NA, QL = Stage::QL, QH = Stage::QH, SC = Stage::SC, XA = Stage::XA;
    constexpr int THREADS = 128;
    static_assert(sizeof(Stage) % 16 == 0, "16-byte stage slots");
    extern __shared__ __align__(16) uint8_t q6ktSmem[];
    Stage* st = reinterpret_cast<Stage*>(q6ktSmem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int tile0 = blockIdx.x, row0 = tile0 * TS_GRAPH_Q6KT_ROWS;
    const int nsb = k / 256, nH = 2 * nsb;
    const int tiles = (outDim + TS_GRAPH_Q6KT_ROWS - 1) / TS_GRAPH_Q6KT_ROWS;

    auto load = [&](int slot, int u) {
        Stage& s = st[slot];
        const int sbIdx = u >> 1, h = u & 1;
        const bool ok = tile0 < tiles;
        const uint8_t* chunk = w + ((long long)(ok ? tile0 : 0) * nsb + sbIdx) * TS_GRAPH_Q6KT_CHUNK;
        for (int i = tid; i < BR * 4; i += THREADS)          // ql: 4 x 16 bytes per row
        {
            const int r = i >> 2, c = i & 3;
            ts_graph_cp16(s.ql + r * QL + 16 * c, chunk + ts_graph_q6kt_ql(h, r) + 16 * c, ok);
        }
        for (int i = tid; i < BR * 2; i += THREADS)          // qh: 2 x 16 bytes per row
        {
            const int r = i >> 1, c = i & 1;
            ts_graph_cp16(s.qh + r * QH + 16 * c, chunk + ts_graph_q6kt_qh(h, r) + 16 * c, ok);
        }
        for (int i = tid; i < BR / 2; i += THREADS)          // sc: 8 bytes per row, 16 per pair
        {
            const int r = 2 * i;
            ts_graph_cp16(s.sc + r * SC, chunk + ts_graph_q6kt_sc(h, r), ok);
        }
        if (h == 0)
            for (int i = tid; i < BR / 8; i += THREADS)      // d: 2 bytes per row, half 0 only
            {
                const int r = 8 * i;
                ts_graph_cp16(s.d + r, chunk + TS_GRAPH_Q6KT_D + 2 * r, ok);
            }
        for (int i = tid; i < NA * 8; i += THREADS)
        {
            const int r = i >> 3, c = i & 7;
            const bool okr = r < n;
            ts_graph_cp16(s.x + r * XA + 16 * c, xq + (long long)(okr ? r : 0) * k + 128LL * u + 16 * c, okr);
        }
        for (int r = tid; r < NA; r += THREADS)
        {
            const bool okr = r < n;
            ts_graph_cp16(s.xd + 4 * r, xd + (long long)(okr ? r : 0) * (k >> 5) + 4LL * u, okr);
        }
    };

    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            acc[j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
    {
        if (s < nH)
            load(s, s);
        ts_graph_cp_commit();
    }
    const int wr = warp * 16 + g;   // this thread's first weight row in the tile (the second is wr + 8)
    float d0 = 0.0f, d1 = 0.0f;
    for (int it = 0; it < nH; ++it)
    {
        ts_graph_cp_wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < nH)
            load(next % STAGES, next);
        ts_graph_cp_commit();
        const Stage& s = st[it % STAGES];
        if ((it & 1) == 0)   // half 0 of a superblock carries its d
        {
            d0 = __half2float(s.d[wr]);
            d1 = __half2float(s.d[wr + 8]);
        }
        uint32_t qlw[2][2][2], qhw[2][2];
#pragma unroll
        for (int rr = 0; rr < 2; ++rr)
#pragma unroll
            for (int lh = 0; lh < 2; ++lh)
            {
                const int row = wr + 8 * rr;
                qlw[rr][lh][0] = *(const uint32_t*)(s.ql + row * QL + 16 * lh + 4 * t);
                qlw[rr][lh][1] = *(const uint32_t*)(s.ql + row * QL + 32 + 16 * lh + 4 * t);
                qhw[rr][lh] = *(const uint32_t*)(s.qh + row * QH + 16 * lh + 4 * t);
            }
#pragma unroll
        for (int m = 0; m < 4; ++m)
        {
            float x0[NT], x1[NT];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
            {
                x0[nt] = s.xd[(8 * nt + 2 * t) * 4 + m];
                x1[nt] = s.xd[(8 * nt + 2 * t + 1) * 4 + m];
            }
#pragma unroll
            for (int lh = 0; lh < 2; ++lh)
            {
                int a[2];
#pragma unroll
                for (int rr = 0; rr < 2; ++rr)
                    a[rr] = (int)__vsubss4(((qlw[rr][lh][m & 1] >> (4 * (m >> 1))) & 0x0F0F0F0Fu)
                                               | (((qhw[rr][lh] >> (2 * m)) & 0x03030303u) << 4), 0x20202020);
                const int jj = 2 * m + lh;
                const float s0 = d0 * (float)s.sc[wr * SC + jj], s1 = d1 * (float)s.sc[(wr + 8) * SC + jj];
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                {
                    int c[4];
                    ts_graph_mma_s8(c, a[0], a[1], *(const int*)(s.x + (8 * nt + g) * XA + 32 * m + 16 * lh + 4 * t));
                    acc[nt][0] += ts_graph_i2f_small(c[0]) * s0 * x0[nt];
                    acc[nt][1] += ts_graph_i2f_small(c[1]) * s0 * x1[nt];
                    acc[nt][2] += ts_graph_i2f_small(c[2]) * s1 * x0[nt];
                    acc[nt][3] += ts_graph_i2f_small(c[3]) * s1 * x1[nt];
                }
            }
        }
    }
    ts_graph_cp_wait<0>();
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const int r = row0 + wr + 8 * (e >> 1), col = 8 * nt + 2 * t + (e & 1);
            if (r < outDim && col < n)
                y[(long long)col * outDim + r] = acc[nt][e];
        }
}

// y [n][outDim] = xq [n] W^T for n <= NT * 8 rows. Launch: grid ceil(outDim / 64), block 128, dynamic shared memory
// ts_graph_gemm_q6kt_smem_bytes reports.
#define TS_GRAPH_GEMM_Q6KT(NT) \
    extern "C" __global__ void __launch_bounds__(128) ts_graph_gemm_q6kt_##NT( \
        const uint8_t* __restrict__ w, int outDim, int k, const int8_t* __restrict__ xq, const float* __restrict__ xd, int n, \
        float* __restrict__ y) \
    { ts_graph_gemm_q6kt<NT>(w, outDim, k, xq, xd, n, y); }
TS_GRAPH_GEMM_Q6KT(1)
TS_GRAPH_GEMM_Q6KT(2)
TS_GRAPH_GEMM_Q6KT(4)
TS_GRAPH_GEMM_Q6KT(8)

extern "C" __global__ void ts_graph_gemm_q6kt_smem_bytes(int* out)
{
    out[0] = 3 * (int)sizeof(TsGraphQ6ktStage<1>);
    out[1] = 3 * (int)sizeof(TsGraphQ6ktStage<2>);
    out[2] = 3 * (int)sizeof(TsGraphQ6ktStage<4>);
    out[3] = 3 * (int)sizeof(TsGraphQ6ktStage<8>);
}

// ---------------------------------------------------------------------------
// Q4_K x q8_1 on the int8 tensor cores: the direct engines' drafter product (ts_q35_gemm_q4k). Q4_K superblock (256
// values, 144 bytes): half d, half dmin, scales[12] (eight 6-bit scale / min pairs), qs[128]; value 64c + l (c < 4,
// l < 64) is nibble (l >= 32) of qs[32c + l % 32], in sub-block 2c + (l >= 32), and decodes to d * sc * q - dmin * m.
// One mma.m16n8k32.s8 per 32-value sub-block (exactly one q8_1 block):
//   sum_i (d sc q_i - dmin m) xd xq_i = d sc xd (sum q xq) - dmin m xd (sum xq).
// Block: 4 warps x 16 weight rows against NT * 8 activation rows; grid (row tiles, K splits). With more than one split
// every block adds its share atomically (the caller zeroes y first). A row's value never depends on how many rows the
// call holds, but the splits' additions land in any order: for products whose last bits may vary (a drafter's).
// ---------------------------------------------------------------------------
#define TS_GRAPH_Q4K_ROW (256 + 16)

__device__ __forceinline__ void ts_graph_q4k_scale_min(int j, const uint8_t* s, int& sc, int& m)
{
    if (j < 4)
    {
        sc = s[j] & 63;
        m = s[j + 4] & 63;
    }
    else
    {
        sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

__device__ __forceinline__ void ts_graph_mma_s8_k32(int (&c)[4], const int (&a)[4], int b0, int b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
#else
    __trap();
#endif
}

template <int NT>
__device__ __forceinline__ void ts_graph_gemm_q4k(
    const uint8_t* __restrict__ w, long long rowBytes, int outDim, int k,
    const int8_t* __restrict__ xq, const float* __restrict__ xd, int n, float* __restrict__ y)
{
    constexpr int NA = NT * 8;
    __shared__ __align__(16) int8_t xs[NA][TS_GRAPH_Q4K_ROW];
    __shared__ float xds[NA][8], xms[NA][8];
    __shared__ __align__(16) int8_t ws[4][16][TS_GRAPH_Q4K_ROW];
    __shared__ float wsd[4][16][8], wsm[4][16][8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * 64 + warp * 16;
    const int nsb = k / 256, per = (nsb + gridDim.y - 1) / gridDim.y;
    const int sb0 = blockIdx.y * per, sb1 = min(nsb, sb0 + per);
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            acc[j][e] = 0.0f;
    for (int sb = sb0; sb < sb1; ++sb)
    {
        __syncthreads();
        for (int i = tid; i < NA * 16; i += blockDim.x)
        {
            const int r = i >> 4, c = (i & 15) * 16;
            *(int4*)&xs[r][c] = r < n ? *(const int4*)(xq + (long long)r * k + sb * 256 + c) : make_int4(0, 0, 0, 0);
        }
        for (int i = tid; i < NA * 8; i += blockDim.x)
        {
            const int r = i >> 3;
            xds[r][i & 7] = r < n ? xd[(long long)r * (k >> 5) + sb * 8 + (i & 7)] : 0.0f;
        }
        // This warp's 16 weight rows: lane takes (row, chunk, half) items of 16 code bytes.
        for (int item = lane; item < 16 * 4 * 2; item += 32)
        {
            const int r = item >> 3, c = (item >> 1) & 3, hl = item & 1;
            const int gr = row0 + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (gr < outDim)
            {
                const uint8_t* blk = w + (long long)gr * rowBytes + (long long)sb * 144;
                v = *(const uint4*)(blk + 16 + 32 * c + 16 * hl);
                if (c == 0 && hl == 0)
                {
                    const float d = __half2float(*(const half*)blk), dmin = __half2float(*(const half*)(blk + 2));
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        int sc, m;
                        ts_graph_q4k_scale_min(j, blk + 4, sc, m);
                        wsd[warp][r][j] = d * (float)sc;
                        wsm[warp][r][j] = dmin * (float)m;
                    }
                }
            }
            else if (c == 0 && hl == 0)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    wsd[warp][r][j] = wsm[warp][r][j] = 0.0f;
            }
            int8_t* lo = &ws[warp][r][64 * c + 16 * hl];
            *(uint4*)lo = make_uint4(v.x & 0x0F0F0F0Fu, v.y & 0x0F0F0F0Fu, v.z & 0x0F0F0F0Fu, v.w & 0x0F0F0F0Fu);
            *(uint4*)(lo + 32) = make_uint4((v.x >> 4) & 0x0F0F0F0Fu, (v.y >> 4) & 0x0F0F0F0Fu,
                                            (v.z >> 4) & 0x0F0F0F0Fu, (v.w >> 4) & 0x0F0F0F0Fu);
        }
        __syncthreads();
        // Activation block sums, scaled: xd * sum(xq) per 32-value block.
        for (int i = tid; i < NA * 8; i += blockDim.x)
        {
            const int r = i >> 3, b = i & 7;
            const int* p = (const int*)&xs[r][32 * b];
            int sum = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                sum = __dp4a(p[j], 0x01010101, sum);
            xms[r][b] = xds[r][b] * (float)sum;
        }
        __syncthreads();
#pragma unroll 2
        for (int j = 0; j < 8; ++j)   // 32-value sub-blocks
        {
            const int a[4] = { *(const int*)&ws[warp][g][32 * j + 4 * t], *(const int*)&ws[warp][g + 8][32 * j + 4 * t],
                               *(const int*)&ws[warp][g][32 * j + 16 + 4 * t], *(const int*)&ws[warp][g + 8][32 * j + 16 + 4 * t] };
            const float d0 = wsd[warp][g][j], d1 = wsd[warp][g + 8][j];
            const float m0 = wsm[warp][g][j], m1 = wsm[warp][g + 8][j];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
            {
                int c[4];
                const int* xr = (const int*)&xs[8 * nt + g][32 * j];
                ts_graph_mma_s8_k32(c, a, xr[t], xr[4 + t]);
                const int c0 = 8 * nt + 2 * t;
                const float x0 = xds[c0][j], x1 = xds[c0 + 1][j], s0 = xms[c0][j], s1 = xms[c0 + 1][j];
                acc[nt][0] += (float)c[0] * d0 * x0 - m0 * s0;
                acc[nt][1] += (float)c[1] * d0 * x1 - m0 * s1;
                acc[nt][2] += (float)c[2] * d1 * x0 - m1 * s0;
                acc[nt][3] += (float)c[3] * d1 * x1 - m1 * s1;
            }
        }
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const int r = row0 + g + 8 * (e >> 1), col = 8 * nt + 2 * t + (e & 1);
            if (r < outDim && col < n)
            {
                float* dst = y + (long long)col * outDim + r;
                if (gridDim.y > 1)
                    atomicAdd(dst, acc[nt][e]);
                else
                    *dst = acc[nt][e];
            }
        }
}

// y [n][outDim] = xq [n] W^T for n <= NT * 8 rows of a Q4_K weight (rows rowBytes apart; k a multiple of 256). Launch:
// grid (ceil(outDim / 64), K splits), block 128; with more than one split y starts zeroed.
#define TS_GRAPH_GEMM_Q4K(NT) \
    extern "C" __global__ void __launch_bounds__(128) ts_graph_gemm_q4k_##NT( \
        const uint8_t* __restrict__ w, long long rowBytes, int outDim, int k, \
        const int8_t* __restrict__ xq, const float* __restrict__ xd, int n, float* __restrict__ y) \
    { ts_graph_gemm_q4k<NT>(w, rowBytes, outDim, k, xq, xd, n, y); }
TS_GRAPH_GEMM_Q4K(1)
TS_GRAPH_GEMM_Q4K(2)
TS_GRAPH_GEMM_Q4K(4)
TS_GRAPH_GEMM_Q4K(8)

// y = silu(x * scale) over n contiguous floats (the read's low-rank activation). Launch: ceil(n / 256), 256.
// yq (null, or n % 32 == 0): y also as q8_1.
extern "C" __global__ void ts_graph_scale_silu_f32(const float* __restrict__ x, float* __restrict__ y, long long n, float scale,
    ts_graph_block_q8_1* __restrict__ yq)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        const float v = x[i] * scale;
        const float o = v / (1.f + expf(-v));
        y[i] = o;
        if (yq != nullptr)
            ts_graph_quantize_q8_1(o, yq + i / 32, threadIdx.x & 31);
    }
}

// The read's mix: out[r][i] = (sum over c, in order, of xn[r][c][i] * sigmoid(gates[r][c][i])) / hc.
// Launch: ceil(rows * n / 256), 256.
// outq (null, or n % 32 == 0): out also as q8_1.
// injPart (null, or ts_graph_hc_norm_f32's partial injection dots of these rows): the first block also writes
// inject[r][o] = the sum over the row's streams c, in order, of injPart[(r * hc + c) * hc + o].
extern "C" __global__ void ts_graph_hc_gate_mix_f32(const float* __restrict__ xn, const float* __restrict__ gates,
    float* __restrict__ out, int rows, int n, int hc, ts_graph_block_q8_1* __restrict__ outq, const float* __restrict__ injPart,
    float* __restrict__ inject)
{
    if (injPart != nullptr && blockIdx.x == 0)
        for (int t = threadIdx.x; t < rows * hc; t += blockDim.x)
        {
            const int r = t / hc, o = t % hc;
            float sum = 0.f;
            for (int c = 0; c < hc; c++)
                sum += injPart[((size_t)r * hc + c) * hc + o];
            inject[t] = sum;
        }
    const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= (long long)rows * n) return;
    const long long r = j / n, i = j % n;
    const float* xr = xn + r * hc * n;
    const float* gr = gates + r * hc * n;
    float sum = 0.f;
    for (int c = 0; c < hc; c++)
    {
        const float g = gr[(long long)c * n + i];
        sum += xr[(long long)c * n + i] * (1.f / (1.f + expf(-g)));
    }
    const float m = sum * (1.f / hc);
    out[j] = m;
    if (outq != nullptr)
        ts_graph_quantize_q8_1(m, outq + j / 32, threadIdx.x & 31);
}

// The scatter: out[r][c][i] = stream[r][c][i] + block[r][i] * 2 * sigmoid(inject[r][c] / hc). Launch: ceil(rows * hc * n / 256), 256.
extern "C" __global__ void ts_graph_hc_scatter_f32(const float* __restrict__ stream, const float* __restrict__ block,
    const float* __restrict__ inject, float* __restrict__ out, int rows, int n, int hc)
{
    const long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long width = (long long)hc * n;
    if (j >= (long long)rows * width) return;
    const long long r = j / width, c = (j % width) / n, i = j % n;
    const float w = 2.f * (1.f / (1.f + expf(-inject[r * hc + c] * (1.f / hc))));
    out[j] = stream[j] + block[r * n + i] * w;
}

// ggml_sum_rows over contiguous rows of cols floats: y[r] = sum of x[r, :]. Launch: grid rows, block 256.
extern "C" __global__ void ts_graph_sum_rows_f32(const float* __restrict__ x, float* __restrict__ y, int cols)
{
    __shared__ float part[8];
    const float* row = x + (size_t)blockIdx.x * cols;
    float s = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        s += row[c];
    for (int off = 16; off > 0; off >>= 1)
        s += __shfl_xor_sync(0xffffffffu, s, off);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float t = 0.f;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += part[w];
        y[blockIdx.x] = t;
    }
}

// y[r][o] = sum over i of x[r][i] * w[o][i] for an F32 weight [outs][k] with few outputs and a few rows, where a GEMV
// library kernel spreads the outputs alone over the device (a hyper-connection's injection, 4 outputs of 10240, took
// 49 us on an A40 through cuBLAS). One block an output and up to 8 rows (blockIdx.y the row group), its threads
// across k; the sums run in a fixed order. x's rows are xPitch floats apart. Launch: grid (outs, ceil(rows / 8)),
// block a multiple of 32 up to 1024.
extern "C" __global__ void ts_graph_small_matvec_f32(const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ y, int rows, int k, int outs, long long xPitch)
{
    __shared__ float part[32][8];
    const int o = blockIdx.x, r0 = blockIdx.y * 8;
    const int nr = min(8, rows - r0);
    const float* wr = w + (size_t)o * k;
    const float* xr = x + (size_t)r0 * xPitch;
    float acc[8];
#pragma unroll
    for (int r = 0; r < 8; r++)
        acc[r] = 0.f;
    const bool vec = (k & 3) == 0 && (xPitch & 3) == 0 && ((size_t)wr & 15) == 0 && ((size_t)xr & 15) == 0;
    if (vec)
    {
        const float4* w4 = reinterpret_cast<const float4*>(wr);
        for (int i = threadIdx.x; i < (k >> 2); i += blockDim.x)
        {
            const float4 a = w4[i];
#pragma unroll
            for (int r = 0; r < 8; r++)
                if (r < nr)
                {
                    const float4 b = reinterpret_cast<const float4*>(xr + (size_t)r * xPitch)[i];
                    acc[r] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                }
        }
    }
    else
    {
        for (int i = threadIdx.x; i < k; i += blockDim.x)
        {
            const float a = wr[i];
#pragma unroll
            for (int r = 0; r < 8; r++)
                if (r < nr)
                    acc[r] += a * xr[(size_t)r * xPitch + i];
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
#pragma unroll
    for (int r = 0; r < 8; r++)
    {
        const float v = ts_graph_warp_sum(acc[r]);
        if (lane == 0)
            part[warp][r] = v;
    }
    __syncthreads();
    if (threadIdx.x < nr)
    {
        float t = 0.f;
        for (int i = 0; i < warps; i++)
            t += part[i][threadIdx.x];
        y[(size_t)(r0 + threadIdx.x) * outs + o] = t;
    }
}

// A prompt chunk's routes counted per expert on the device (the routing profile and the adaptive tiers read them later):
// counts[ids[i]] += 1 for every id in [0, experts). Launch: ceil(n / 256), 256.
extern "C" __global__ void ts_graph_count_routes(const int* __restrict__ ids, int n, unsigned int* __restrict__ counts, int experts)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        const int e = ids[i];
        if (e >= 0 && e < experts)
            atomicAdd(counts + e, 1u);
    }
}

// ggml's binary ops with b repeating over a (each of b's extents 1 or a's), both contiguous with ne0 innermost:
// out[i] = a[i] op b[i's coordinates, 0 where b's extent is 1]. op: 0 add, 1 sub, 2 mul, 3 div. Replaces a gather that
// materialized the repeated b first (a shared expert's [1, rows] gate, a hyper-connection's per-stream weights).
// Launch: ceil(n / 256), 256.
extern "C" __global__ void ts_graph_binary_bcast_f32(const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out, long long n, long long a0, long long a1, long long a2, long long b0, long long b1,
    long long b2, long long b3, int op)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    long long r = i;
    const long long i0 = r % a0; r /= a0;
    const long long i1 = r % a1; r /= a1;
    const long long i2 = r % a2;
    const long long i3 = r / a2;
    const long long j = (((b3 == 1 ? 0 : i3) * b2 + (b2 == 1 ? 0 : i2)) * b1 + (b1 == 1 ? 0 : i1)) * b0 + (b0 == 1 ? 0 : i0);
    const float x = a[i], y = b[j];
    out[i] = op == 0 ? x + y : op == 1 ? x - y : op == 2 ? x * y : x / y;
}

// ggml_sqrt, elementwise over n contiguous floats.
extern "C" __global__ void ts_graph_sqrt_f32(const float* __restrict__ x, float* __restrict__ y, long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = sqrtf(x[i]);
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

// ts_graph_deltanet_prep_f32 over a prompt chunk, the same outputs bit for bit, a block per (head, tile of
// TS_GRAPH_PREP_TILE tokens): the tile's conv inputs (positions t0 .. t0 + tokens + width - 2) staged in shared memory once
// (the per-row kernel read each input row width times), conv_in written along time through the stage (a warp writes 32
// consecutive tokens of one channel; the per-row kernel wrote one float per channel row, a 32-byte sector each), each
// thread's channel convolved and SiLU'd for every token of the tile, and each token's q/k norm reduced in the per-row
// kernel's order (a warp sum per 32 channels, then the warps' sums in order). conv_rows == width - 1, width <=
// TS_GRAPH_PREP_WIDTH (the conv weights in registers), d % 32 == 0, d <= 256. Launch: grid (2 * k_heads + v_heads,
// ceil(rows / TS_GRAPH_PREP_TILE)), block d, dynamic shared memory ((TS_GRAPH_PREP_TILE + width - 1) * (d + 1) +
// TS_GRAPH_PREP_TILE * d / 32) floats.
#define TS_GRAPH_PREP_TILE 32
#define TS_GRAPH_PREP_WIDTH 4
extern "C" __global__ void ts_graph_deltanet_prep_tiled_f32(
    const float* __restrict__ window, const float* __restrict__ qkv, const float* __restrict__ conv_w,
    const float* __restrict__ alpha, const float* __restrict__ beta, const float* __restrict__ dt, const float* __restrict__ a,
    float* __restrict__ conv_in, float* __restrict__ q_out, float* __restrict__ k_out, float* __restrict__ v_out,
    float* __restrict__ g_out, float* __restrict__ beta_out,
    int channels, int conv_rows, int width, int d, int k_heads, int v_heads, int rows, float eps,
    long long qkv_pitch, long long ab_pitch)
{
    extern __shared__ float prep_smem[];
    const int i = threadIdx.x;
    const int head = blockIdx.x;
    const int t0 = blockIdx.y * TS_GRAPH_PREP_TILE;
    const int tokens = min(TS_GRAPH_PREP_TILE, rows - t0);
    const int span = conv_rows + rows, pitch = d + 1, warps = d / 32;
    float* stage = prep_smem;                                                  // [tokens + width - 1][d + 1]
    float* partial = prep_smem + (TS_GRAPH_PREP_TILE + width - 1) * pitch;     // [tile][d / 32]
    const size_t c = (size_t)head * d + i;
    const int staged = tokens + width - 1;
    // Eight rows' loads in flight before any is stored (each store waits for its load).
    for (int s0 = 0; s0 < staged; s0 += 8)
    {
        float in[8];
#pragma unroll
        for (int u = 0; u < 8; u++)
        {
            const int p = t0 + s0 + u;
            if (s0 + u < staged)
                in[u] = p < conv_rows ? window[c * conv_rows + p] : qkv[(size_t)(p - conv_rows) * qkv_pitch + c];
        }
#pragma unroll
        for (int u = 0; u < 8; u++)
            if (s0 + u < staged) stage[(s0 + u) * pitch + i] = in[u];
    }
    float w[TS_GRAPH_PREP_WIDTH];
#pragma unroll
    for (int j = 0; j < TS_GRAPH_PREP_WIDTH; j++)
        if (j < width) w[j] = conv_w[c * width + j];
    if (t0 == 0)
        for (int p = 0; p < conv_rows; p++) conv_in[c * span + p] = window[c * conv_rows + p];
    __syncthreads();
    for (int e = i; e < d * tokens; e += d)
    {
        const int ch = e / tokens, tt = e % tokens;
        conv_in[((size_t)head * d + ch) * span + conv_rows + t0 + tt] = stage[(conv_rows + tt) * pitch + ch];
    }
    float y[TS_GRAPH_PREP_TILE];
#pragma unroll
    for (int tt = 0; tt < TS_GRAPH_PREP_TILE; tt++)
    {
        if (tt >= tokens) break;
        float x = 0.0f;
#pragma unroll
        for (int j = 0; j < TS_GRAPH_PREP_WIDTH; j++)
        {
            if (j >= width) break;
            x += stage[(tt + j) * pitch + i] * w[j];
        }
        y[tt] = x / (1.0f + expf(-x));
    }
    if (head < 2 * k_heads)
    {
#pragma unroll
        for (int tt = 0; tt < TS_GRAPH_PREP_TILE; tt++)
        {
            if (tt >= tokens) break;
            const float sq = ts_graph_warp_sum(y[tt] * y[tt]);
            if ((i & 31) == 0) partial[tt * warps + (i >> 5)] = sq;
        }
        __syncthreads();
#pragma unroll
        for (int tt = 0; tt < TS_GRAPH_PREP_TILE; tt++)
        {
            if (tt >= tokens) break;
            float total = 0.0f;
            for (int s = 0; s < warps; s++) total += partial[tt * warps + s];
            const float yn = y[tt] * rsqrtf(total + eps);
            const size_t t = (size_t)t0 + tt;
            if (head < k_heads) q_out[(t * k_heads + head) * d + i] = yn;
            else k_out[(t * k_heads + head - k_heads) * d + i] = yn;
        }
    }
    else
    {
        const int vh = head - 2 * k_heads;
#pragma unroll
        for (int tt = 0; tt < TS_GRAPH_PREP_TILE; tt++)
        {
            if (tt >= tokens) break;
            v_out[(((size_t)t0 + tt) * v_heads + vh) * d + i] = y[tt];
        }
        if (i < tokens)
        {
            const size_t t = (size_t)t0 + i;
            const size_t gi = t * v_heads + vh, ai = t * ab_pitch + vh;
            const float s = alpha[ai] + dt[vh];
            g_out[gi] = (s > 20.0f ? s : logf(1.0f + expf(s))) * a[vh];
            beta_out[gi] = 1.0f / (1.0f + expf(-beta[ai]));
        }
    }
}

// GraphFusion.SigmoidGate: out = attn * sigmoid(gate) over [rows, heads * d] (attn and out contiguous), gate read at its row
// and head strides where the query projection put it (its heads interleaved with the queries, rows a packed projection's
// pitch apart): the per-op path's sigmoid (ts_unary_f32) and product (ts_binary_f32) expressions, in one pass. With outq
// (not null; n % 32 == 0, so a warp's 32 values are one block) also out's q8_1 copy. Launch: grid ceil(n / 256), block 256.
extern "C" __global__ void ts_graph_sigmoid_gate_f32(const float* __restrict__ attn, const float* __restrict__ gate,
    float* __restrict__ out, int heads, int d, long long gate_row, long long gate_head, long long n,
    ts_graph_block_q8_1* __restrict__ outq)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const long long width = (long long)heads * d, r = i / width, rem = i - r * width;
    const long long h = rem / d, e = rem - h * d;
    const float g = gate[r * gate_row + h * gate_head + e];
    const float o = attn[i] * (1.0f / (1.0f + expf(-g)));
    out[i] = o;
    if (outq != nullptr)
        ts_graph_quantize_q8_1(o, outq + i / 32, (int)(i & 31));
}

// The residual add and the next block's input norm in one pass: sum = a + b, normed = sum * rsqrt(mean(sum^2) + eps) * w
// over rows of cols (ggml_rms_norm(ggml_add(a, b)) * w). b null: the sum is a (a residual add's norm whose sum was already
// made: the same reduction, so the same bits); sum null: not written. Launch: grid rows, block 512.
extern "C" __global__ void ts_graph_add_rms_norm_f32(const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ w, float* __restrict__ sum, float* __restrict__ normed, int cols, float eps)
{
    __shared__ float partial[32];
    const size_t r = blockIdx.x;
    const float* ar = a + r * cols;
    const float* br = b != nullptr ? b + r * cols : nullptr;
    float* sr = sum != nullptr ? sum + r * cols : nullptr;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        const float v = br != nullptr ? ar[i] + br[i] : ar[i];
        if (sr != nullptr) sr[i] = v;
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
        nr[i] = (sr != nullptr ? sr[i] : br != nullptr ? ar[i] + br[i] : ar[i]) * scale * w[i];
}

// ts_graph_add_rms_norm_f32 for rows of up to 16 * 512 = 8192 values at 512 threads, the row and its norm weights held in
// registers with every global load issued up front (the direct engine's ts_q35_add_rmsnorm_nvfp4_5120_f32 rule): the loop
// above waits one dependent round trip per element, and a verify window's rows are a block each. Thread t takes elements
// t, t + 512, ... in that order, so its sum, the reductions and the expressions are the loop's: the same bits. b and sum may
// be null as the loop's. Launch: grid rows, block 512.
#define TS_GRAPH_ADD_RMS_NORM_REG_PER 16
extern "C" __global__ void __launch_bounds__(512) ts_graph_add_rms_norm_reg_f32(const float* __restrict__ a,
    const float* __restrict__ b, const float* __restrict__ w, float* __restrict__ sum, float* __restrict__ normed, int cols,
    float eps)
{
    constexpr int T = 512, PER = TS_GRAPH_ADD_RMS_NORM_REG_PER;
    __shared__ float partial[32];
    const size_t r = blockIdx.x;
    const int tid = threadIdx.x;
    const float* ar = a + r * cols;
    const float* br = b != nullptr ? b + r * cols : nullptr;
    float v[PER], wv[PER];
#pragma unroll
    for (int p = 0; p < PER; p++)
    {
        const int i = tid + p * T;
        v[p] = i < cols ? (br != nullptr ? ar[i] + br[i] : ar[i]) : 0.0f;
    }
#pragma unroll
    for (int p = 0; p < PER; p++)
    {
        const int i = tid + p * T;
        wv[p] = i < cols ? w[i] : 0.0f;
    }
    float* sr = sum != nullptr ? sum + r * cols : nullptr;
    float ss = 0.0f;
#pragma unroll
    for (int p = 0; p < PER; p++)
    {
        const int i = tid + p * T;
        if (i < cols)
        {
            if (sr != nullptr) sr[i] = v[p];
            ss += v[p] * v[p];
        }
    }
    ss = ts_graph_warp_sum(ss);
    if ((tid & 31) == 0) partial[tid >> 5] = ss;
    __syncthreads();
    if (tid < 32)
    {
        float t = tid < (T >> 5) ? partial[tid] : 0.0f;
        t = ts_graph_warp_sum(t);
        if (tid == 0) partial[0] = t;
    }
    __syncthreads();
    const float scale = rsqrtf(partial[0] / (float)cols + eps);
    float* nr = normed + r * cols;
#pragma unroll
    for (int p = 0; p < PER; p++)
    {
        const int i = tid + p * T;
        if (i < cols)
            nr[i] = v[p] * scale * wv[p];
    }
}

// GraphFusion.GatedRmsNorm: out = y * rsqrt(mean(y^2) + eps) * w * silu(z) (sigmoid(z) with sigmoid_gate) over rows of
// d. Row r of y and out is head r % heads of token r / heads; z holds a token's heads * d values z_pitch floats after the
// previous token's (a view of a packed projection is read in place). With outq (not null) also out's q8_1 copy for the
// product that reads it (the quantizer's bytes). Launch: grid rows, block d (d % 32 == 0, <= 1024).
extern "C" __global__ void ts_graph_gated_rms_norm_f32(const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ w, float* __restrict__ out, int d, float eps, int heads, long long z_pitch, int sigmoid_gate,
    ts_graph_block_q8_1* __restrict__ outq)
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
    const float gate = sigmoid_gate ? 1.0f / (1.0f + expf(-zz)) : zz / (1.0f + expf(-zz));
    const float o = v * rsqrtf(total / d + eps) * w[i] * gate;
    out[r * d + i] = o;
    if (outq != nullptr)
        ts_graph_quantize_q8_1(o, outq + (r * d + i) / 32, i & 31);
}

// ts_rope_ex_f32 (tensorsharp_kernels.cu) over a strided source: row r is head r % heads of token r / heads, its values
// contiguous at input + token * token_stride + head * head_stride (a fused projection's q or k block read where it lies);
// the output is contiguous rows. The same expressions as ts_rope_ex_f32 and its YaRN helpers (copied), so a rotated value
// is that kernel's to the bit. Launch: as ts_rope_ex_f32.
__device__ __forceinline__ float ts_graph_yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float freq_base)
{
    return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * 3.14159265358979323846f)) / (2.0f * logf(freq_base));
}

__device__ __forceinline__ void ts_graph_yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float* low, float* high)
{
    if (beta_fast == 0.0f && beta_slow == 0.0f)
    {
        *low = FLT_MAX;
        *high = (float)(n_dims / 2 - 1);
    }
    else
    {
        *low = fmaxf(0.0f, floorf(ts_graph_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base)));
        *high = fminf((float)(n_dims / 2 - 1), ceilf(ts_graph_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base)));
    }
}

__device__ __forceinline__ void ts_graph_yarn_rope(float theta_extrap, float freq_scale, float corr_low, float corr_high, int i0, float ext_factor, float mscale, float* c, float* s)
{
    float theta_interp = freq_scale * theta_extrap;
    float ramp_y = ((float)i0 - corr_low) / fmaxf(0.001f, corr_high - corr_low);
    float ramp_mix = (1.0f - fminf(1.0f, fmaxf(0.0f, ramp_y))) * ext_factor;
    float theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
    *c = cosf(theta) * mscale;
    *s = sinf(theta) * mscale;
}

extern "C" __global__ void ts_graph_rope_ex_strided_f32(
    const float* input,
    long long token_stride,
    long long head_stride,
    int heads,
    const void* positions,
    float* output,
    int rows,
    int cols,
    int rope_dim,
    int mode,
    int positions_are_int32,
    int n_ctx_orig,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    int add_to_result)
{
    int active_dim = rope_dim < cols ? rope_dim : cols;
    int pair_count = active_dim / 2;
    int global = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_count <= 0)
    {
        int total = rows * cols;
        if (global < total && !add_to_result)
            output[global] = input[(long long)((global / cols) / heads) * token_stride + (long long)((global / cols) % heads) * head_stride + global % cols];
        return;
    }

    if (!add_to_result && global < rows * cols)
    {
        int col = global % cols;
        if (col >= pair_count * 2)
            output[global] = input[(long long)((global / cols) / heads) * token_stride + (long long)((global / cols) % heads) * head_stride + global % cols];
    }

    int idx = global;
    int total = rows * pair_count;
    if (idx >= total)
        return;

    const int GGML_ROPE_TYPE_NEOX = 2;
    bool neox = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    int row = idx / pair_count;
    int i = idx - row * pair_count;
    int pos = positions_are_int32
        ? reinterpret_cast<const int*>(positions)[row]
        : (int)reinterpret_cast<const float*>(positions)[row];

    float theta_extrap = (float)pos * powf(freq_base, -2.0f * (float)i / (float)active_dim);
    float c;
    float s;
    if (ext_factor != 0.0f)
    {
        float corr_low;
        float corr_high;
        ts_graph_yarn_corr_dims(active_dim, n_ctx_orig, freq_base, beta_fast, beta_slow, &corr_low, &corr_high);
        float mscale = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
        ts_graph_yarn_rope(theta_extrap, freq_scale, corr_low, corr_high, i, ext_factor, mscale, &c, &s);
    }
    else
    {
        float angle = theta_extrap * freq_scale;
        c = cosf(angle);
        s = sinf(angle);
    }

    const float* src = input + (long long)(row / heads) * token_stride + (long long)(row % heads) * head_stride;
    float* dst = output + (size_t)row * cols;

    int left_index;
    int right_index;
    if (neox)
    {
        left_index = i;
        right_index = i + pair_count;
    }
    else
    {
        left_index = i * 2;
        right_index = i * 2 + 1;
    }

    float left = src[left_index];
    float right = src[right_index];
    float out_left = left * c - right * s;
    float out_right = right * c + left * s;
    if (add_to_result)
    {
        dst[left_index] += out_left;
        dst[right_index] += out_right;
    }
    else
    {
        dst[left_index] = out_left;
        dst[right_index] = out_right;
    }
}

// GraphFusion.SharedExpertGate: out[r] = moe[r] + shared[r] * sigmoid(x[r] . w), the four launches it replaces in the same
// arithmetic: the gate's dot as ts_graph_small_matvec_f32 sums it (one output, up to 8 rows a block, the same threads and
// order), the per-op path's sigmoid expression, then the scale and the add each rounded on its own (as the separate
// broadcast product and add round them). Launch: grid (1, ceil(rows / 8)), block ts_graph_small_matvec_f32's for k.
extern "C" __global__ void ts_graph_shared_gate_add_f32(const float* __restrict__ moe, const float* __restrict__ shared,
    const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int rows, int k, long long xPitch, int n)
{
    __shared__ float part[32][8];
    __shared__ float gate[8];
    const int r0 = blockIdx.y * 8;
    const int nr = min(8, rows - r0);
    const float* wr = w;
    const float* xr = x + (size_t)r0 * xPitch;
    float acc[8];
#pragma unroll
    for (int r = 0; r < 8; r++)
        acc[r] = 0.f;
    const bool vec = (k & 3) == 0 && (xPitch & 3) == 0 && ((size_t)wr & 15) == 0 && ((size_t)xr & 15) == 0;
    if (vec)
    {
        const float4* w4 = reinterpret_cast<const float4*>(wr);
        for (int i = threadIdx.x; i < (k >> 2); i += blockDim.x)
        {
            const float4 a = w4[i];
#pragma unroll
            for (int r = 0; r < 8; r++)
                if (r < nr)
                {
                    const float4 b = reinterpret_cast<const float4*>(xr + (size_t)r * xPitch)[i];
                    acc[r] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                }
        }
    }
    else
    {
        for (int i = threadIdx.x; i < k; i += blockDim.x)
        {
            const float a = wr[i];
#pragma unroll
            for (int r = 0; r < 8; r++)
                if (r < nr)
                    acc[r] += a * xr[(size_t)r * xPitch + i];
        }
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
#pragma unroll
    for (int r = 0; r < 8; r++)
    {
        const float v = ts_graph_warp_sum(acc[r]);
        if (lane == 0)
            part[warp][r] = v;
    }
    __syncthreads();
    if (threadIdx.x < nr)
    {
        float t = 0.f;
        for (int i = 0; i < warps; i++)
            t += part[i][threadIdx.x];
        gate[threadIdx.x] = 1.0f / (1.0f + expf(-t));
    }
    __syncthreads();
    for (int r = 0; r < nr; r++)
    {
        const float s = gate[r];
        const size_t base = (size_t)(r0 + r) * n;
        for (int j = threadIdx.x; j < n; j += blockDim.x)
            out[base + j] = __fadd_rn(moe[base + j], __fmul_rn(shared[base + j], s));
    }
}

// ---------------------------------------------------------------------------
// GraphFusion.QsaAttention: a QSA layer's block selection and its attention over the selected cells, for one sequence's
// rows over its cache window. The direct engine's kernels (tensorsharp_q4e_kernels.cu: ts_q4e_qsa_scores_f32,
// ts_q4e_qsa_select_i32, ts_q4e_attention_mma_f32) on the runtime's operands and the runtime's selection: a row keeps its
// `picks` best visible blocks WHOLE and its own incomplete tail block (llama.cpp since #29751), where the engine kept
// top_k + ratio - 1 single cells (part of a further block where the tail is short). A row's position p is its KV index -
// the row the step wrote it at, read on the device, so a recorded step replays at every position its window covers; block b
// (cells b R .. b R + R - 1) is visible to it iff b < (p + 1) / R, and its pooled key is the pooled half of its last
// member's row of the indexer cache [cap][2 d] (F16).
// ---------------------------------------------------------------------------

// The region's block scores, ts_graph_lightning_indexer_f32's arithmetic in its order (a warp a block, lanes over the dims, the
// heads' rectified dots summed in head order, each times the weight w: the step's 1/sqrt(d) for every head), with each
// block's pooled key read where the cache holds it (F16, widened as ts_graph_get_rows_f16 widens the region's gather of it)
// and the visibility mask computed from the row's position: a visible block's score is the indexer's sum plus the mask's 0,
// an invisible one's -inf (its sum plus the mask's -inf; not computed). So the scores
// the region's GetRows and LightningIndexer ops give, bit for bit, for finite queries and keys. q [rows][heads][d] F32 (d <=
// 256), cache rows cachePitch halves apart, pos [rows], out [rows][pools]. Launch: grid (ceil(pools / 8), rows), block 256,
// heads * d floats of dynamic shared memory.
extern "C" __global__ void ts_graph_qsa_scores_f32(const float* __restrict__ q, const __half* __restrict__ cache, long long cachePitch,
    const int* __restrict__ pos, float* __restrict__ out, float w, int d, int heads, int pools, int ratio)
{
    extern __shared__ float sq[];
    const size_t r = blockIdx.y;
    for (int i = threadIdx.x; i < heads * d; i += blockDim.x)
        sq[i] = q[r * heads * d + i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int j = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (j >= pools) return;
    if (j >= (pos[r] + 1) / ratio)
    {
        if (lane == 0)
            out[r * pools + j] = -INFINITY;
        return;
    }
    const __half* key = cache + (size_t)(j * ratio + ratio - 1) * (size_t)cachePitch + d;
    float kr[8];
#pragma unroll
    for (int t = 0; t < 8; t++)
    {
        const int i = lane + 32 * t;
        kr[t] = i < d ? __half2float(key[i]) : 0.f;
    }
    float sum = 0.f;
    for (int h = 0; h < heads; h++)
    {
        float dot = 0.f;
#pragma unroll
        for (int t = 0; t < 8; t++)
        {
            const int i = lane + 32 * t;
            if (i < d) dot += sq[h * d + i] * kr[t];
        }
        for (int off = 16; off > 0; off >>= 1)
            dot += __shfl_xor_sync(0xffffffffu, dot, off);
        if (dot > 0.f) sum += w * dot;
    }
    if (lane == 0)
        out[r * pools + j] = sum + 0.f;
}

// The region's selection: the cells each row attends, ascending - its `picks` best visible blocks whole (by score, a tie to
// the lower block; every visible block when there are no more than picks), then its own incomplete tail block's cells up to
// p. The cell list ts_graph_topk_wide_f32 (or ts_graph_topk_f32) over the scores, ts_graph_topk_mask_f32 over the step's plan
// and ts_graph_mask_cells give, in their order. The picks' threshold is ts_q4e_qsa_select_i32's radix select (four 8-bit
// passes over the visible blocks' order keys, each pass's bin found by a block-wide scan) over ts_order_key, the wide top-k's
// order; one ascending ballot pass then takes every block above it and the lowest-indexed ones equal to it.
// scores [rows][pools], pos [rows], cells [rows][cellStride] (cellStride >= picks * ratio + ratio - 1), count [rows].
// Launch: grid rows, block 256.
extern "C" __global__ void __launch_bounds__(256) ts_graph_qsa_select_i32(const float* __restrict__ scores, const int* __restrict__ pos,
    int* __restrict__ cells, int* __restrict__ count, int pools, int ratio, int picks, int cellStride)
{
    __shared__ unsigned int hist[256];
    __shared__ unsigned int s_prefix;
    __shared__ int s_need, s_equal, s_taken;
    __shared__ int scanWarp[8], warpEqual[8], warpTaken[8];
    const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int p = pos[r];
    const int tail = (p + 1) / ratio * ratio, nb = tail / ratio;
    int* out = cells + (size_t)r * cellStride;
    if (nb <= picks)
    {
        // Every visible block and the tail: cells 0 .. p.
        for (int c = tid; c <= p; c += blockDim.x)
            out[c] = c;
        if (tid == 0)
            count[r] = p + 1;
        return;
    }
    const float* sc = scores + (size_t)r * pools;

    // The picks-th largest key, byte by byte from the most significant.
    if (tid == 0)
    {
        s_prefix = 0;
        s_need = picks;
    }
    __syncthreads();
    for (int pass = 0; pass < 4; ++pass)
    {
        const int shift = 24 - 8 * pass;
        hist[tid] = 0;
        __syncthreads();
        const unsigned int prefix = s_prefix;
        for (int b = tid; b < nb; b += blockDim.x)
        {
            const unsigned int key = ts_order_key(sc[b]);
            if (pass > 0 && (key >> (shift + 8)) != prefix)
                continue;
            atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        // The bin a walk from bin 255 down stops at (the first whose count reaches the need left, the need then dropping by
        // the bins above it), as a scan: thread i holds bin 255 - i. The bin always exists: the pass's keys number at least
        // the need (nb > picks, then the previous bin's count).
        {
            const int c = (int)hist[255 - tid];
            int incl = c;
#pragma unroll
            for (int off = 1; off < 32; off <<= 1)
            {
                const int o = __shfl_up_sync(0xffffffffu, incl, off);
                if (lane >= off)
                    incl += o;
            }
            if (lane == 31)
                scanWarp[warp] = incl;
            const int n = s_need;
            __syncthreads();
            for (int w = 0; w < warp; ++w)
                incl += scanWarp[w];
            const int excl = incl - c;
            if (incl >= n && excl < n)
            {
                s_need = n - excl;
                s_prefix = (prefix << 8) | (unsigned int)(255 - tid);
            }
        }
        __syncthreads();
    }

    // Ascending: every block above the threshold, and of those equal to it the lowest-indexed s_need.
    const unsigned int thr = s_prefix;
    const int needEqual = s_need;
    if (tid == 0)
    {
        s_equal = 0;
        s_taken = 0;
    }
    __syncthreads();
    for (int b0 = 0; b0 < nb; b0 += blockDim.x)
    {
        const int b = b0 + tid;
        const unsigned int key = b < nb ? ts_order_key(sc[b]) : 0u;
        const bool equal = b < nb && key == thr;
        const unsigned int equalBallot = __ballot_sync(0xffffffffu, equal);
        if (lane == 0)
            warpEqual[warp] = __popc(equalBallot);
        __syncthreads();
        int rank = s_equal + __popc(equalBallot & ((1u << lane) - 1u));
        for (int w = 0; w < warp; ++w)
            rank += warpEqual[w];
        const bool take = (b < nb && key > thr) || (equal && rank < needEqual);
        const unsigned int takeBallot = __ballot_sync(0xffffffffu, take);
        if (lane == 0)
            warpTaken[warp] = __popc(takeBallot);
        __syncthreads();
        int slot = s_taken + __popc(takeBallot & ((1u << lane) - 1u));
        for (int w = 0; w < warp; ++w)
            slot += warpTaken[w];
        if (take)
            for (int i = 0; i < ratio; ++i)
                out[slot * ratio + i] = b * ratio + i;
        __syncthreads();
        if (tid == 0)
        {
            int equalSeen = 0, taken = 0;
            for (int w = 0; w < (int)(blockDim.x >> 5); ++w)
            {
                equalSeen += warpEqual[w];
                taken += warpTaken[w];
            }
            s_equal += equalSeen;
            s_taken += taken;
        }
        __syncthreads();
    }
    const int kept = picks * ratio, t0 = p + 1 - tail;
    for (int i = tid; i < t0; i += blockDim.x)
        out[kept + i] = tail + i;
    if (tid == 0)
        count[r] = kept + t0;
}

// A prompt's rows over their selected cells on tensor cores: ts_q4e_attention_mma_f32 (the direct engine's prompt attention)
// on the head-first cache and without the output gate (the model's SigmoidGate region follows). A block of four warps takes
// one (KV head, row): the group's query heads are the rows of a 16-row mma tile (G <= 16, the rest zero), the row's cells come
// in chunks of 16 (warp w takes chunks w, w + 4, ...), each chunk's keys and then values are staged in the warp's shared tile,
// and S = Q K^T and O += P V run on mma.sync.m16n8k16 (F16 operands, F32 accumulation) under a per-warp online softmax; the
// warps merge at the end. Queries and probabilities are F16 operands, so a row rounds apart from the flash kernel's (which
// accumulates short runs in F16) within F16 P.V tolerance; decode rows and verify windows keep the flash kernel.
//   q, out  [rows][heads][256] F32;  k, v  [kvHeads][cap][256] F16;  cells [rows][cellStride], count [rows]
// sm_80 and newer (ldmatrix, mma.f16): the compute_75 image's body is empty; the host launches it only where
// CudaPrefillModule.ArchGuardedKernelsUsable holds. Launch: grid (kvHeads, rows), block 128.
#define TS_QSA_HD 256
#define TS_QSA_WARPS 4
#define TS_QSA_LDS (TS_QSA_HD + 8)   // halves a shared row holds: 528 B, so ldmatrix's eight rows hit distinct banks

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void ts_graph_qsa_ldsm_x4(uint32_t (&r)[4], const __half* p)
{
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void ts_graph_qsa_ldsm_x4_trans(uint32_t (&r)[4], const __half* p)
{
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void ts_graph_qsa_mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t ts_graph_qsa_pack(float lo, float hi)
{
    const half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&h);
}

// Listed cells base .. base + 15 of one KV head (head: its first row) into sm[16][LDS], zero past n: lane & 15 takes a cell,
// lane >> 4 half of its dims.
__device__ __forceinline__ void ts_graph_qsa_stage(__half (*sm)[TS_QSA_LDS], const __half* __restrict__ cache, size_t head,
    const int* __restrict__ list, int base, int n, int lane)
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
    const int4* src = reinterpret_cast<const int4*>(cache + (head + (size_t)list[i]) * TS_QSA_HD + part * 128);
    int4 w[16];
#pragma unroll
    for (int s = 0; s < 16; ++s)
        w[s] = src[s];
#pragma unroll
    for (int s = 0; s < 16; ++s)
        dst[s] = make_uint4((uint32_t)w[s].x, (uint32_t)w[s].y, (uint32_t)w[s].z, (uint32_t)w[s].w);
}
#endif

extern "C" __global__ void __launch_bounds__(TS_QSA_WARPS * 32) ts_graph_qsa_attn_mma_f16_d256(
    const float* __restrict__ q, const __half* __restrict__ kc, const __half* __restrict__ vc, const int* __restrict__ cells,
    const int* __restrict__ count, float* __restrict__ out, int cap, int heads, int kvHeads, float scale, int cellStride)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __shared__ __align__(16) __half smQ[16][TS_QSA_LDS];
    __shared__ __align__(16) __half smKV[TS_QSA_WARPS][16][TS_QSA_LDS];
    __shared__ float smM[TS_QSA_WARPS][16], smL[TS_QSA_WARPS][16], smInv[16];

    const int kvh = blockIdx.x, t = blockIdx.y;
    const int G = heads / kvHeads;
    const int n = count[t];
    const int* list = cells + (size_t)t * cellStride;
    const size_t head = (size_t)kvh * cap;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tig = lane & 3;

    // The group's query heads as F16 rows, rows G..15 zero.
    for (int u = tid; u < 16 * (TS_QSA_HD / 4); u += blockDim.x)
    {
        const int r = u / (TS_QSA_HD / 4), c4 = (u % (TS_QSA_HD / 4)) * 4;
        float4 f = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (r < G)
            f = *reinterpret_cast<const float4*>(q + ((size_t)t * heads + kvh * G + r) * TS_QSA_HD + c4);
        *reinterpret_cast<uint2*>(&smQ[r][c4]) = make_uint2(ts_graph_qsa_pack(f.x, f.y), ts_graph_qsa_pack(f.z, f.w));
    }
    __syncthreads();

    float o[TS_QSA_HD / 8][4];
#pragma unroll
    for (int dt = 0; dt < TS_QSA_HD / 8; ++dt)
        o[dt][0] = o[dt][1] = o[dt][2] = o[dt][3] = 0.0f;
    float mRow[2] = { -INFINITY, -INFINITY }, lRow[2] = { 0.0f, 0.0f };
    const float sl2 = scale * 1.4426950408889634f;
    __half (*sm)[TS_QSA_LDS] = smKV[warp];
    const int nChunks = (n + 15) >> 4;
    for (int ch = warp; ch < nChunks; ch += TS_QSA_WARPS)
    {
        const int base = ch << 4;
        ts_graph_qsa_stage(sm, kc, head, list, base, n, lane);
        __syncwarp();
        float s[2][4] = { { 0.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 0.0f, 0.0f } };
#pragma unroll
        for (int kk = 0; kk < TS_QSA_HD; kk += 16)
        {
            uint32_t a[4], b[4];
            ts_graph_qsa_ldsm_x4(a, &smQ[(lane & 7) + ((lane >> 3) & 1) * 8][kk + (lane >> 4) * 8]);
            ts_graph_qsa_ldsm_x4(b, &sm[(lane & 7) + (lane >> 4) * 8][kk + ((lane >> 3) & 1) * 8]);
            ts_graph_qsa_mma(s[0], a, b[0], b[1]);
            ts_graph_qsa_mma(s[1], a, b[2], b[3]);
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
        for (int dt = 0; dt < TS_QSA_HD / 8; ++dt)
        {
            o[dt][0] *= c0; o[dt][1] *= c0;
            o[dt][2] *= c1; o[dt][3] *= c1;
        }
        const uint32_t pa[4] = { ts_graph_qsa_pack(pr[0][0], pr[0][1]), ts_graph_qsa_pack(pr[0][2], pr[0][3]),
                                 ts_graph_qsa_pack(pr[1][0], pr[1][1]), ts_graph_qsa_pack(pr[1][2], pr[1][3]) };
        __syncwarp();
        ts_graph_qsa_stage(sm, vc, head, list, base, n, lane);
        __syncwarp();
#pragma unroll
        for (int dt = 0; dt < TS_QSA_HD / 8; dt += 2)
        {
            uint32_t b[4];
            ts_graph_qsa_ldsm_x4_trans(b, &sm[(lane & 7) + ((lane >> 3) & 1) * 8][dt * 8 + (lane >> 4) * 8]);
            ts_graph_qsa_mma(o[dt], pa, b[0], b[1]);
            ts_graph_qsa_mma(o[dt + 1], pa, b[2], b[3]);
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
    for (int w = 0; w < TS_QSA_WARPS; ++w)
    {
        M0 = fmaxf(M0, smM[w][g]);
        M1 = fmaxf(M1, smM[w][g + 8]);
    }
    const float f0 = mRow[0] == -INFINITY ? 0.0f : exp2f(mRow[0] - M0);
    const float f1 = mRow[1] == -INFINITY ? 0.0f : exp2f(mRow[1] - M1);
    float* smO = reinterpret_cast<float*>(&smKV[0][0][0]);   // [16][TS_QSA_LDS] floats over the staging tiles
#pragma unroll
    for (int w = 0; w < TS_QSA_WARPS; ++w)
    {
        if (warp == w)
        {
#pragma unroll
            for (int dt = 0; dt < TS_QSA_HD / 8; ++dt)
            {
                const int d = dt * 8 + 2 * tig;
                float2* r0 = reinterpret_cast<float2*>(&smO[g * TS_QSA_LDS + d]);
                float2* r1 = reinterpret_cast<float2*>(&smO[(g + 8) * TS_QSA_LDS + d]);
                float2 v0 = make_float2(o[dt][0] * f0, o[dt][1] * f0), v1 = make_float2(o[dt][2] * f1, o[dt][3] * f1);
                if (w > 0)
                {
                    const float2 p0 = *r0, p1 = *r1;
                    v0.x += p0.x; v0.y += p0.y;
                    v1.x += p1.x; v1.y += p1.y;
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
        for (int w = 0; w < TS_QSA_WARPS; ++w)
            M = fmaxf(M, smM[w][tid]);
        float L = 0.0f;
#pragma unroll
        for (int w = 0; w < TS_QSA_WARPS; ++w)
            L += smM[w][tid] == -INFINITY ? 0.0f : smL[w][tid] * exp2f(smM[w][tid] - M);
        smInv[tid] = L > 0.0f ? 1.0f / L : 0.0f;
    }
    __syncthreads();
    for (int u = tid; u < G * TS_QSA_HD; u += blockDim.x)
    {
        const int r = u / TS_QSA_HD, d = u % TS_QSA_HD;
        out[((size_t)t * heads + kvh * G + r) * TS_QSA_HD + d] = smO[r * TS_QSA_LDS + d] * smInv[r];
    }
#else
    (void)q; (void)kc; (void)vc; (void)cells; (void)count; (void)out; (void)cap; (void)heads; (void)kvHeads; (void)scale;
    (void)cellStride;
#endif
}

// ---------------------------------------------------------------------------
// GraphFusion.QsaKeys: a QSA layer's indexer keys for one sequence's rows in one launch, where the region's ops are about a
// dozen a layer and step (the packed [raw | 0] rows and their write; the members' gather, their sum, its scale, the
// weighted norm, the rotary, the [last raw | pooled] row and its write). The ops also pool again, to the values it holds,
// the last block complete before a short chunk; this kernel pools only the blocks its rows complete, so the cache's bytes
// are the ops' once the step has run.
// ---------------------------------------------------------------------------

// block_reduce_sum of tensorsharp_kernels.cu (ts_rmsnorm_f32's reduction, copied): each warp's sum by shuffles down, then the
// first warp's over the warps' sums; the block's total on thread 0.
__device__ __forceinline__ float ts_graph_block_sum_down(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, offset);

    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0)
        warp_sums[warp] = v;
    __syncthreads();

    if (warp == 0)
    {
        int num_warps = (blockDim.x + 31) >> 5;
        v = lane < num_warps ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, offset);
    }

    return v;
}

// Row r (KV index pos[r]; the rows consecutive, pos[0] the step's first) writes its raw key raw[r] into its [raw | pooled] F16
// cache row, rounded as the ops' row write rounds it, the pooled half 0 times the raw key unless the row completes its block
// ((p + 1) % ratio == 0). A completing row also pools the block into that half, in the ops' arithmetic and order: each
// member's raw key as the cache holds it (rows before the step read there, the step's own rounded to F16 here, as the ops'
// gather reads them back after the write), summed member by member, times 1 / ratio (the Scale); ts_rmsnorm_f32's weighted
// norm (its thread-strided squares over a 256-thread block, its reduction, rsqrtf(sum / d + eps), x * inv then * w); and
// ts_graph_rope_ex_strided_f32's rotation at the block's first cell's position - table[b - first], the ops' positions of the
// blocks they pool from `first` (PoolSpan: a short chunk's last n complete blocks, else the first block its rows complete,
// computed here from pos[0], so a recorded step replays). The rotated values rounded to F16 as the ops' pooled write rounds
// them. Launch: grid count, block 256 (the norm's), d floats of dynamic shared memory.
extern "C" __global__ void __launch_bounds__(256) ts_graph_qsa_store_pool_f32(
    const float* __restrict__ raw, __half* __restrict__ cache, const int* __restrict__ pos, const float* __restrict__ w,
    const int* __restrict__ table, int d, int ratio, int count, int shortChunk, int n, float eps, float inv_ratio,
    int rope_dim, int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor,
    float beta_fast, float beta_slow)
{
    extern __shared__ float pooled[];
    __shared__ float inv_rms;
    const int r = blockIdx.x, tid = threadIdx.x;
    const int start = pos[0], p = pos[r];
    const float* key = raw + (size_t)r * d;
    __half* row = cache + (size_t)p * 2 * d;
    // The block the row completes, and its index among the blocks whose positions the table holds.
    const int b = (p + 1) / ratio - 1, end = (start + count) / ratio;
    const int first = max(shortChunk ? end - (count + ratio - 1) / ratio : (start + ratio) / ratio - 1, 0);
    const bool completes = (p + 1) % ratio == 0 && b - first >= 0 && b - first < n;
    for (int i = tid; i < d; i += blockDim.x)
    {
        const float v = key[i];
        row[i] = __float2half_rn(v);
        if (!completes)
            row[d + i] = __float2half_rn(v * 0.0f);
    }
    if (!completes)
        return;

    const int base = b * ratio;
    float sum_sq = 0.0f;
    for (int i = tid; i < d; i += blockDim.x)
    {
        float acc = 0.0f;
        for (int k = 0; k < ratio; ++k)
        {
            const int c = base + k;
            const float m = c < start ? __half2float(cache[(size_t)c * 2 * d + i])
                                      : __half2float(__float2half_rn(raw[(size_t)(c - start) * d + i]));
            acc = k == 0 ? m : acc + m;
        }
        const float v = acc * inv_ratio;
        pooled[i] = v;
        sum_sq += v * v;
    }
    sum_sq = ts_graph_block_sum_down(sum_sq);
    if (tid == 0)
        inv_rms = rsqrtf(sum_sq / (float)d + eps);
    __syncthreads();
    for (int i = tid; i < d; i += blockDim.x)
    {
        float v = pooled[i] * inv_rms;
        v *= w[i];
        pooled[i] = v;
    }
    __syncthreads();

    const int position = table[b - first];
    const int active_dim = rope_dim < d ? rope_dim : d;
    const int pair_count = active_dim / 2;
    const bool neox = (mode & 2) != 0;   // GGML_ROPE_TYPE_NEOX
    for (int i = tid; i < pair_count; i += blockDim.x)
    {
        float theta_extrap = (float)position * powf(freq_base, -2.0f * (float)i / (float)active_dim);
        float c;
        float s;
        if (ext_factor != 0.0f)
        {
            float corr_low;
            float corr_high;
            ts_graph_yarn_corr_dims(active_dim, n_ctx_orig, freq_base, beta_fast, beta_slow, &corr_low, &corr_high);
            float mscale = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
            ts_graph_yarn_rope(theta_extrap, freq_scale, corr_low, corr_high, i, ext_factor, mscale, &c, &s);
        }
        else
        {
            float angle = theta_extrap * freq_scale;
            c = cosf(angle);
            s = sinf(angle);
        }
        const int left_index = neox ? i : i * 2;
        const int right_index = neox ? i + pair_count : i * 2 + 1;
        float left = pooled[left_index];
        float right = pooled[right_index];
        float out_left = left * c - right * s;
        float out_right = right * c + left * s;
        row[d + left_index] = __float2half_rn(out_left);
        row[d + right_index] = __float2half_rn(out_right);
    }
    for (int col = pair_count * 2 + tid; col < d; col += blockDim.x)
        row[d + col] = __float2half_rn(pooled[col]);
}

// ---------------------------------------------------------------------------
// GraphFusion.AttentionPrep: an attention layer's query/key prep and K/V cache writes for one sequence's rows in one launch,
// where the region's ops are about seven launches a layer and step: the queries' copy out of the gated projection, the two
// weighted norms, the two rotaries, and the two cache writes with the copies that regroup K and V head-first for them.
// ---------------------------------------------------------------------------

#define TS_GRAPH_ATTN_PREP_WARPS 8

// A value of unit (t, h) into its place: a query head's row of the output, or a KV head's key at `at` of the head-first
// cache, rounded to F16 as the ops' row writes round it (an F16 cache) or stored as it is (F32).
__device__ __forceinline__ void ts_graph_attn_prep_store(bool isQ, float* __restrict__ qDst, void* __restrict__ kCache, int f16Cache,
    size_t at, int col, float value)
{
    if (isQ) qDst[col] = value;
    else if (f16Cache) ((__half*)kCache)[at + col] = __float2half_rn(value);
    else ((float*)kCache)[at + col] = value;
}

// A warp per unit (t, h), as the direct engines' ts_q35_attn_prep_f32 / ts_q4e_attn_prep_f32 lay theirs out: units
// 0 .. heads - 1 of row t are its query heads, then its kvHeads key heads, then its kvHeads value heads. A query or key head
// is read where the projection put it (q + t * qRow + h * qHead; k likewise), normed with ts_rmsnorm_f32's weighted norm and
// rotated with ts_graph_rope_ex_strided_f32's expressions at the row's position pos[t] (the columns past the rotated pairs as
// they are), so every value is the ops' to the bit. The norm's sum is ts_rmsnorm_f32's 256-thread block sum replayed in one
// warp: lane l holds, for each of the block's 8 warps w, thread 32w + l's sum (its squares of columns 32w + l + 256k in k
// order), each warp's shuffle-down tree is run as that warp runs it, and the 8 warp sums go through block_reduce_sum's
// second tree from lanes 0..7 (the others 0) - the same additions in the same order. A query head goes to out (rows outRow,
// heads outHead floats apart: rows of heads, or heads of rows where the attention takes them so); a key head and a value head
// go to the head-first caches [kvHeads][cap][d] and [kvHeads][cap][dv] at the row's KV index cells[t] (skipped outside the
// cache, as ts_graph_set_rows_f32 skips it), rounded with __float2half_rn as the ops' row writes round them (F16 caches) or
// stored as they are (F32). The positions and KV indices are read on the device, so a recorded step replays. Launch: grid
// ceil(rows * (heads + 2 kvHeads) / 8), block 256, 8 * d floats of dynamic shared memory (a warp's row).
extern "C" __global__ void __launch_bounds__(TS_GRAPH_ATTN_PREP_WARPS * 32) ts_graph_attn_prep_f32(
    const float* __restrict__ q, long long qRow, long long qHead, const float* __restrict__ k, long long kRow, long long kHead,
    const float* __restrict__ v, long long vRow, long long vHead, const float* __restrict__ qNorm, const float* __restrict__ kNorm,
    float* __restrict__ out, long long outRow, long long outHead, void* __restrict__ kCache, void* __restrict__ vCache, int f16Cache,
    int cap, const int* __restrict__ pos, const int* __restrict__ cells, int rows, int heads, int kvHeads, int d, int dv, float eps,
    int rope_dim, int mode, int n_ctx_orig, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast,
    float beta_slow)
{
    extern __shared__ float rowbuf[];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int units = heads + 2 * kvHeads;
    const long long unit = (long long)blockIdx.x * TS_GRAPH_ATTN_PREP_WARPS + warp;
    // Every exit below is a whole warp's: the kernel has no block-wide barrier.
    if (unit >= (long long)rows * units)
        return;
    const int t = (int)(unit / units), h = (int)(unit % units);
    const bool isQ = h < heads, isV = h >= heads + kvHeads;
    const int kvh = isV ? h - heads - kvHeads : h - heads;
    const int cell = isQ ? 0 : cells[t];
    if (!isQ && (cell < 0 || cell >= cap))
        return;
    if (isV)
    {
        const float* src = v + t * vRow + (long long)kvh * vHead;
        const size_t at = ((size_t)kvh * cap + cell) * dv;
        for (int i = lane; i < dv; i += 32)
        {
            const float val = src[i];
            if (f16Cache) ((__half*)vCache)[at + i] = __float2half_rn(val);
            else ((float*)vCache)[at + i] = val;
        }
        return;
    }
    const float* x = isQ ? q + t * qRow + (long long)h * qHead : k + t * kRow + (long long)kvh * kHead;
    const float* w = isQ ? qNorm : kNorm;
    float* qDst = isQ ? out + t * outRow + (long long)h * outHead : nullptr;
    const size_t at = isQ ? 0 : ((size_t)kvh * cap + cell) * d;
    float* buf = rowbuf + (size_t)warp * d;

    // ts_rmsnorm_f32's thread sums (blockDim 256): thread 32m + lane squares columns 32m + lane + 256k, k ascending.
    float part[8];
#pragma unroll
    for (int m = 0; m < 8; ++m)
        part[m] = 0.0f;
    for (int base = 0; base < d; base += 256)
    {
#pragma unroll
        for (int m = 0; m < 8; ++m)
        {
            const int i = base + 32 * m + lane;
            if (i < d)
            {
                float val = x[i];
                buf[i] = val;
                part[m] += val * val;
            }
        }
    }
    // block_reduce_sum: each warp's shuffle-down tree (its sum on lane 0), then the first warp's over lanes 0..7.
    float sum_sq = 0.0f;
#pragma unroll
    for (int m = 0; m < 8; ++m)
    {
        float s = part[m];
        for (int offset = 16; offset > 0; offset >>= 1)
            s += __shfl_down_sync(0xFFFFFFFF, s, offset);
        s = __shfl_sync(0xFFFFFFFF, s, 0);
        if (lane == m)
            sum_sq = s;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    const float inv_rms = rsqrtf(__shfl_sync(0xFFFFFFFF, sum_sq, 0) / (float)d + eps);
    for (int i = lane; i < d; i += 32)
    {
        float val = buf[i] * inv_rms;
        val *= w[i];
        buf[i] = val;
    }
    __syncwarp();

    const int position = pos[t];
    const int active_dim = rope_dim < d ? rope_dim : d;
    const int pair_count = active_dim / 2;
    const bool neox = (mode & 2) != 0;   // GGML_ROPE_TYPE_NEOX
    for (int i = lane; i < pair_count; i += 32)
    {
        float theta_extrap = (float)position * powf(freq_base, -2.0f * (float)i / (float)active_dim);
        float c;
        float s;
        if (ext_factor != 0.0f)
        {
            float corr_low;
            float corr_high;
            ts_graph_yarn_corr_dims(active_dim, n_ctx_orig, freq_base, beta_fast, beta_slow, &corr_low, &corr_high);
            float mscale = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
            ts_graph_yarn_rope(theta_extrap, freq_scale, corr_low, corr_high, i, ext_factor, mscale, &c, &s);
        }
        else
        {
            float angle = theta_extrap * freq_scale;
            c = cosf(angle);
            s = sinf(angle);
        }
        const int left_index = neox ? i : i * 2;
        const int right_index = neox ? i + pair_count : i * 2 + 1;
        float left = buf[left_index];
        float right = buf[right_index];
        float out_left = left * c - right * s;
        float out_right = right * c + left * s;
        ts_graph_attn_prep_store(isQ, qDst, kCache, f16Cache, at, left_index, out_left);
        ts_graph_attn_prep_store(isQ, qDst, kCache, f16Cache, at, right_index, out_right);
    }
    for (int col = pair_count * 2 + lane; col < d; col += 32)
        ts_graph_attn_prep_store(isQ, qDst, kCache, f16Cache, at, col, buf[col]);
}

// ---------------------------------------------------------------------------
// A block drafter's attention over its KV ring (DFlash / DFlash2 draft steps): the direct engine's ts_q35_df_attn_f16 and
// ts_q35_df_combine_f32 over the runtime's ring (F32, the block's own rows already written into it) and its additive mask.
// Row r of a KV head's group is block row r / G, head kvh * G + r % G (G = heads / kvHeads); every row scores every slot,
// plus the mask's value for its block row and slot (0, or -inf outside the drafter's window). F16 operands (queries, K, V and
// the probabilities), F32 scores and accumulation, online softmax on mma.sync.m16n8k16. A block takes `chunk` keys of one KV
// head for 64 rows (4 warps x 16) and writes unnormalized partials (acc[HD], ln-domain max, sum); the combine merges a row's
// partials in key order, so a run's sums do not depend on timing.
//   q        F32, block row p head h at q[p * qRow + h * qHead + c]
//   k, v     F32 [kvHeads][len][HD] (16-byte aligned)
//   mask     [nq][len], block row p at mask[p * maskRow + slot]: F16 (maskHalf) or F32
//   partial  [nq][heads][splits][HD + 2];  out  [nq][heads][HD] F32
// sm_80 and newer (ldmatrix, mma.f16): the compute_75 image's body is empty; the host launches it only where
// CudaPrefillModule.ArchGuardedKernelsUsable holds. Launch: grid (splits, kvHeads, ceil(nq * G / 64)), block 128; the
// combine: grid nq * heads, block HD.
// ---------------------------------------------------------------------------
#define TS_RING_KT 32   // keys a tile

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void ts_graph_ldsm_x2_trans(uint32_t& b0, uint32_t& b1, const __half* row)
{
    const unsigned a = (unsigned)__cvta_generic_to_shared(row);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(a));
}
#endif

template <int HD>
__device__ __forceinline__ void ts_graph_ring_attn(
    const float* __restrict__ q, long long qRow, long long qHead, const float* __restrict__ k, const float* __restrict__ v, int len,
    const void* __restrict__ mask, int maskHalf, long long maskRow, int nq, int heads, int kvHeads, float scale, int chunk, int splits,
    float* __restrict__ partial)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int ROW = HD + 8;          // F16 row stride of the K / V tiles: ldmatrix's eight rows hit distinct banks
    constexpr int KK = HD / 16, NT = HD / 8;
    constexpr float LOG2E = 1.4426950408889634f;
    __shared__ __align__(16) __half ks[TS_RING_KT * ROW];
    __shared__ __align__(16) __half vs[TS_RING_KT * ROW];
    __shared__ float ms[64][TS_RING_KT + 1];   // the mask for the tile's block rows and keys, in log2 units
    const int split = blockIdx.x, kvh = blockIdx.y;
    const int G = heads / kvHeads, R = nq * G;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, qd = lane & 3;
    const int kLo = split * chunk, kHi = min(len, kLo + chunk);
    const float* kh = k + (long long)kvh * len * HD;
    const float* vh = v + (long long)kvh * len * HD;
    const int tile0 = blockIdx.z * 64, p0 = tile0 / G;
    const int pn = min(R - 1, tile0 + 63) / G - p0 + 1;   // block rows the tile's rows span (at most 64)

    const int r0 = tile0 + warp * 16 + g, r1 = r0 + 8;
    const bool live0 = r0 < R, live1 = r1 < R;
    const int pi0 = (live0 ? r0 : tile0) / G - p0, pi1 = (live1 ? r1 : tile0) / G - p0;
    const bool warpLive = tile0 + warp * 16 < R;
    auto qptr = [&](int r) { return q + (long long)(r / G) * qRow + (long long)(kvh * G + r % G) * qHead; };

    uint32_t qa[KK][4];
    {
        const float* q0 = live0 ? qptr(r0) : nullptr;
        const float* q1 = live1 ? qptr(r1) : nullptr;
#pragma unroll
        for (int kk = 0; kk < KK; ++kk)
        {
            const int c = kk * 16 + qd * 2;
            qa[kk][0] = q0 ? ts_graph_qsa_pack(q0[c], q0[c + 1]) : 0u;
            qa[kk][1] = q1 ? ts_graph_qsa_pack(q1[c], q1[c + 1]) : 0u;
            qa[kk][2] = q0 ? ts_graph_qsa_pack(q0[c + 8], q0[c + 9]) : 0u;
            qa[kk][3] = q1 ? ts_graph_qsa_pack(q1[c + 8], q1[c + 9]) : 0u;
        }
    }
    float acc[NT][4];
#pragma unroll
    for (int n = 0; n < NT; ++n)
        acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float log2Scale = scale * LOG2E;

    for (int first = kLo; first < kHi; first += TS_RING_KT)
    {
        __syncthreads();
        // The tile's K and V rows as F16 (zero past the range: finite, so a closed key's probability times V stays 0).
        for (int i = threadIdx.x; i < TS_RING_KT * (HD / 8); i += blockDim.x)
        {
            const int key = i / (HD / 8), c = (i % (HD / 8)) * 8, idx = first + key;
            float4 k0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f), k1 = k0, v0 = k0, v1 = k0;
            if (idx < kHi)
            {
                const float4* ksrc = reinterpret_cast<const float4*>(kh + (long long)idx * HD + c);
                const float4* vsrc = reinterpret_cast<const float4*>(vh + (long long)idx * HD + c);
                k0 = ksrc[0];
                k1 = ksrc[1];
                v0 = vsrc[0];
                v1 = vsrc[1];
            }
            *reinterpret_cast<uint4*>(ks + key * ROW + c) = make_uint4(ts_graph_qsa_pack(k0.x, k0.y), ts_graph_qsa_pack(k0.z, k0.w),
                ts_graph_qsa_pack(k1.x, k1.y), ts_graph_qsa_pack(k1.z, k1.w));
            *reinterpret_cast<uint4*>(vs + key * ROW + c) = make_uint4(ts_graph_qsa_pack(v0.x, v0.y), ts_graph_qsa_pack(v0.z, v0.w),
                ts_graph_qsa_pack(v1.x, v1.y), ts_graph_qsa_pack(v1.z, v1.w));
        }
        for (int i = threadIdx.x; i < pn * TS_RING_KT; i += blockDim.x)
        {
            const int pi = i / TS_RING_KT, key = i % TS_RING_KT, idx = first + key;
            const long long at = (long long)(p0 + pi) * maskRow + idx;
            const float m = idx >= kHi ? -INFINITY
                : maskHalf ? __half2float(reinterpret_cast<const __half*>(mask)[at]) : reinterpret_cast<const float*>(mask)[at];
            ms[pi][key] = m * LOG2E;
        }
        __syncthreads();
        if (!warpLive)
            continue;
        float s[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
            const __half* kb = ks + (j * 8 + g) * ROW + qd * 2;
#pragma unroll
            for (int kk = 0; kk < KK; ++kk)
                ts_graph_qsa_mma(s[j], qa[kk], *reinterpret_cast<const uint32_t*>(kb + kk * 16), *reinterpret_cast<const uint32_t*>(kb + kk * 16 + 8));
        }
        float t0 = m0, t1 = m1;
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int key = j * 8 + qd * 2;
            s[j][0] = fmaf(s[j][0], log2Scale, ms[pi0][key]);
            s[j][1] = fmaf(s[j][1], log2Scale, ms[pi0][key + 1]);
            s[j][2] = fmaf(s[j][2], log2Scale, ms[pi1][key]);
            s[j][3] = fmaf(s[j][3], log2Scale, ms[pi1][key + 1]);
            t0 = fmaxf(t0, fmaxf(s[j][0], s[j][1]));
            t1 = fmaxf(t1, fmaxf(s[j][2], s[j][3]));
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1)
        {
            t0 = fmaxf(t0, __shfl_xor_sync(0xffffffffu, t0, off));
            t1 = fmaxf(t1, __shfl_xor_sync(0xffffffffu, t1, off));
        }
        const float c0 = t0 == -INFINITY ? 1.0f : exp2f(m0 - t0);
        const float c1 = t1 == -INFINITY ? 1.0f : exp2f(m1 - t1);
        m0 = t0;
        m1 = t1;
        l0 *= c0;
        l1 *= c1;
#pragma unroll
        for (int n = 0; n < NT; ++n)
        {
            acc[n][0] *= c0;
            acc[n][1] *= c0;
            acc[n][2] *= c1;
            acc[n][3] *= c1;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            s[j][0] = t0 == -INFINITY ? 0.0f : exp2f(s[j][0] - t0);
            s[j][1] = t0 == -INFINITY ? 0.0f : exp2f(s[j][1] - t0);
            s[j][2] = t1 == -INFINITY ? 0.0f : exp2f(s[j][2] - t1);
            s[j][3] = t1 == -INFINITY ? 0.0f : exp2f(s[j][3] - t1);
            l0 += s[j][0] + s[j][1];
            l1 += s[j][2] + s[j][3];
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
        {
            const uint32_t pa[4] = {
                ts_graph_qsa_pack(s[2 * kk][0], s[2 * kk][1]), ts_graph_qsa_pack(s[2 * kk][2], s[2 * kk][3]),
                ts_graph_qsa_pack(s[2 * kk + 1][0], s[2 * kk + 1][1]), ts_graph_qsa_pack(s[2 * kk + 1][2], s[2 * kk + 1][3]) };
            const __half* vrow = vs + (kk * 16 + (lane & 15)) * ROW;
#pragma unroll
            for (int n = 0; n < NT; ++n)
            {
                uint32_t b0, b1;
                ts_graph_ldsm_x2_trans(b0, b1, vrow + n * 8);
                ts_graph_qsa_mma(acc[n], pa, b0, b1);
            }
        }
    }
    if (!warpLive)
        return;
#pragma unroll
    for (int off = 1; off < 4; off <<= 1)
    {
        l0 += __shfl_xor_sync(0xffffffffu, l0, off);
        l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
#pragma unroll
    for (int half_ = 0; half_ < 2; ++half_)
    {
        const int r = half_ ? r1 : r0;
        if (r >= R)
            continue;
        const long long base = (long long)(r / G) * heads + kvh * G + r % G;
        float* p = partial + (base * splits + split) * (HD + 2);
#pragma unroll
        for (int n = 0; n < NT; ++n)
        {
            const int x = n * 8 + qd * 2;
            p[x] = acc[n][half_ * 2];
            p[x + 1] = acc[n][half_ * 2 + 1];
        }
        if (qd == 0)
        {
            const float m = half_ ? m1 : m0;
            p[HD] = m == -INFINITY ? -INFINITY : m * 0.6931471805599453f;
            p[HD + 1] = half_ ? l1 : l0;
        }
    }
#endif
}

#define TS_GRAPH_RING_ATTN(HD)                                                                                                    \
    extern "C" __global__ void __launch_bounds__(128) ts_graph_ring_attn_f16_d##HD(                                                \
        const float* __restrict__ q, long long qRow, long long qHead, const float* __restrict__ k, const float* __restrict__ v,     \
        int len, const void* __restrict__ mask, int maskHalf, long long maskRow, int nq, int heads, int kvHeads, float scale,      \
        int chunk, int splits, float* __restrict__ partial)                                                                        \
    {                                                                                                                              \
        ts_graph_ring_attn<HD>(q, qRow, qHead, k, v, len, mask, maskHalf, maskRow, nq, heads, kvHeads, scale, chunk, splits,       \
            partial);                                                                                                              \
    }

TS_GRAPH_RING_ATTN(64)
TS_GRAPH_RING_ATTN(128)

// A row's partials merged in key order: out = sum_s w_s acc_s / sum_s w_s l_s, w_s = e^(m_s - max m). One block a (row, head).
extern "C" __global__ void ts_graph_ring_attn_combine_f32(const float* __restrict__ partial, int splits, int d, float* __restrict__ out)
{
    const long long base = blockIdx.x;
    const float* p = partial + base * splits * (d + 2);
    float mx = -INFINITY;
    for (int s = 0; s < splits; ++s)
        mx = fmaxf(mx, p[s * (d + 2) + d]);
    for (int x = threadIdx.x; x < d; x += blockDim.x)
    {
        float num = 0.0f, den = 0.0f;
        for (int s = 0; s < splits; ++s)
        {
            const float* ps = p + s * (d + 2);
            const float w = mx == -INFINITY ? 0.0f : expf(ps[d] - mx);
            num = fmaf(ps[x], w, num);
            den = fmaf(ps[d + 1], w, den);
        }
        out[base * d + x] = den > 0.0f ? num / den : 0.0f;
    }
}
