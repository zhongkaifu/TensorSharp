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

#define TS_GRAPH_KERNELS_VERSION 13

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

extern "C" __global__ void ts_graph_moe_route_split(
    const float* __restrict__ x, const int* __restrict__ sel, const float* __restrict__ selW,
    const unsigned char* __restrict__ onDevice, int n, int rows, int perRow, int layer, int tailRank, int* __restrict__ devSel,
    unsigned int* __restrict__ counts, float* xOut, int* idsOut, float* wOut, int* hdr, unsigned int* seq, int* missDev)
{
    // rows of perRow routes each, a route's rank its slot in its row; every route counted for the routing profile.
    const int slots = rows * perRow;
    int miss = 0;
    for (int i = threadIdx.x; i < slots; i += blockDim.x)
    {
        const int e = sel[i];
        if (e >= 0 && counts != nullptr) atomicAdd(&counts[e], 1u);
        const bool device = e >= 0 && onDevice[e];
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
        idsOut[i] = e >= 0 && (onDevice[e] || ts_graph_tail_dropped(sel, slots, perRow, i, tailRank)) ? -(e + 2) : e;
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

// Every block waits for the host's flag to reach the counter, then adds its part of the host's sum (missDev 0: nothing
// was rung; count 0: the host had nothing to add).
extern "C" __global__ void ts_graph_moe_wait_add(const unsigned int* flag, const unsigned int* seq, const int* count,
    const float* hostOut, float* __restrict__ dst, int n, const int* missDev)
{
    __shared__ int add;
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
extern "C" __global__ void ts_graph_hc_norm_f32(const float* __restrict__ x, const float* __restrict__ gamma, float* __restrict__ y,
    int n, int hc, float eps)
{
    __shared__ float part[32];
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
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        y[s * n + i] = xs[i] * scale * g[i];
}

// y = silu(x * scale) over n contiguous floats (the read's low-rank activation). Launch: ceil(n / 256), 256.
extern "C" __global__ void ts_graph_scale_silu_f32(const float* __restrict__ x, float* __restrict__ y, long long n, float scale)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        const float v = x[i] * scale;
        y[i] = v / (1.f + expf(-v));
    }
}

// The read's mix: out[r][i] = (sum over c, in order, of xn[r][c][i] * sigmoid(gates[r][c][i])) / hc.
// Launch: ceil(rows * n / 256), 256.
extern "C" __global__ void ts_graph_hc_gate_mix_f32(const float* __restrict__ xn, const float* __restrict__ gates,
    float* __restrict__ out, int rows, int n, int hc)
{
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
    out[j] = sum * (1.f / hc);
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
