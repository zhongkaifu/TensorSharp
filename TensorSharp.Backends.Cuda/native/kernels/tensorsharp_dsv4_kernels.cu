// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DeepSeek V4 (Flash) kernels for the direct-CUDA backend.
//
// These implement the model-specific stages of the DSV4 forward pass
// (4-stream hyper-connections, shared single-KV-head attention with sinks and
// inverse RoPE, CSA/HCA block compressors, the lightning indexer + top-k row
// selection, and the sqrt-softplus MoE with IQ3_S / MXFP4 expert weights) as
// standalone driver-API kernels, so the DeepSeek4 CUDA executor runs the
// whole model without ggml. Dense Q8_0 / Q6_K projections reuse the generic
// quantized-matmul kernels in tensorsharp_kernels.cu (a separate module
// loaded on the same stream); everything DSV4-specific lives here.
//
// The imperative executor knows p0/nt for every launch, so unlike the fused
// ggml-graph implementation none of these kernels needs metadata tensors:
// ring/scratch source resolution is plain index arithmetic (the CPU executor
// in DeepSeek4CpuExecutor.cs is the reference for the exact semantics).
//
// Caches are F16 (matching the native executor and llama.cpp); activations
// and compressor state rings are F32. RoPE uses precomputed per-position
// tables [n_ctx, n_rot] of interleaved (cos, sin) with the YaRN attention
// factor folded in, built host-side with the same math as the CPU executor.
// ---------------------------------------------------------------------------

#include <cuda_fp16.h>
#include <stdint.h>

#include "tensorsharp_dsv4_tables.cuh"
#include "tensorsharp_iq1s_tables.cuh"
#include "tensorsharp_iq2xxs_tables.cuh"

#define TS_DSV4_QK8_1 32

// Matches ts_block_q8_1 in tensorsharp_kernels.cu (and the layout produced by
// ts_quantize_q8_1_rows_warp_f32): half scale, half scale*sum, 32 int8.
struct ts_dsv4_block_q8_1
{
    half d;
    half s;
    int8_t qs[TS_DSV4_QK8_1];
};

// block_iq2_xxs: 66 bytes / 256 values (half d + 32 uint16 of packed grid
// indices, signs and scales).
struct ts_dsv4_block_iq2_xxs
{
    half d;
    uint16_t qs[32];
};

// block_iq2_xs: 74 bytes / 256 values (half d, 32 uint16 of 9-bit grid index +
// 7-bit sign index, then one byte of two 4-bit scales per 32 values).
struct ts_dsv4_block_iq2_xs
{
    half d;
    uint16_t qs[32];
    uint8_t scales[8];
};

// block_iq3_xxs: 98 bytes / 256 values (half d, 64 grid-index bytes, then one
// uint32 per 32 values: four 7-bit sign indices and a 4-bit scale).
struct ts_dsv4_block_iq3_xxs
{
    half d;
    uint8_t qs[96];
};

// block_iq2_s: 82 bytes / 256 values (half d, 32 low grid-index bytes then 32 sign
// bytes, one byte of high grid-index bits and one of two 4-bit scales per 32 values).
struct ts_dsv4_block_iq2_s
{
    half d;
    uint8_t qs[64];
    uint8_t qh[8];
    uint8_t scales[8];
};

// block_iq1_s: 50 bytes / 256 values (half d, 32 low grid-index bytes, then one uint16
// per 32 values: four 3-bit grid-index highs, a 3-bit scale and the delta's sign).
struct ts_dsv4_block_iq1_s
{
    half d;
    uint8_t qs[32];
    uint16_t qh[8];
};

static_assert(sizeof(ts_dsv4_block_iq2_s) == 82, "block_iq2_s is 82 bytes");
static_assert(sizeof(ts_dsv4_block_iq1_s) == 50, "block_iq1_s is 50 bytes");

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

// dp4a is native on every target CUDA 13 compiles (compute_75 and newer).
__device__ __forceinline__ int ts_dsv4_dp4a(int a, int b, int c)
{
    return __dp4a(a, b, c);
}

__device__ __forceinline__ float ts_dsv4_warp_sum(float v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off, 32);
    return v;
}

__device__ __forceinline__ float ts_dsv4_sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float ts_dsv4_softplus(float x)
{
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

// Weight ints are read as wide as the block layout allows: Q2_K, Q4_K and Q5_K blocks (84, 144,
// 176 bytes) keep their quants 4-byte aligned, Q3_K and Q6_K blocks (110, 210 bytes) 2-byte
// aligned, and only MXFP4's 17-byte blocks need byte loads. Four byte loads per int made the
// expert kernels load-instruction bound.
// int read from an aligned 4-byte lane (q8_1 activation quants, 4-byte-aligned block interiors).
__device__ __forceinline__ int ts_dsv4_get_int_b4(const void* x, int i)
{
    return ((const int*)x)[i];
}

__device__ __forceinline__ int ts_dsv4_get_int_b2(const void* x, int i)
{
    const uint16_t* x16 = (const uint16_t*)x;
    int v = x16[2*i + 0];
    v |= ((int)x16[2*i + 1]) << 16;
    return v;
}

// int assembled from an unaligned byte pointer (MXFP4 qs starts at offset 1).
__device__ __forceinline__ int ts_dsv4_get_int_b1(const uint8_t* x, int i)
{
    return (int)x[4*i + 0]
        | ((int)x[4*i + 1] << 8)
        | ((int)x[4*i + 2] << 16)
        | ((int)x[4*i + 3] << 24);
}

// Map 8 packed nibbles onto table values: .x = low nibbles, .y = high nibbles
// (ggml get_int_from_table_16).
__device__ __forceinline__ int2 ts_dsv4_int_from_table16(int q4, const int8_t* table)
{
    const int q0_32 = (q4 >> 0) & 0x0F0F0F0F;
    const int8_t* q0_8 = (const int8_t*)&q0_32;
    char4 v0 = make_char4(table[q0_8[0]], table[q0_8[1]], table[q0_8[2]], table[q0_8[3]]);

    const int q1_32 = (q4 >> 4) & 0x0F0F0F0F;
    const int8_t* q1_8 = (const int8_t*)&q1_32;
    char4 v1 = make_char4(table[q1_8[0]], table[q1_8[1]], table[q1_8[2]], table[q1_8[3]]);

    int2 r;
    r.x = *(const int*)&v0;
    r.y = *(const int*)&v1;
    return r;
}

// ---------------------------------------------------------------------------
// quantized dot helpers (weight superblock/block vs q8_1 activation blocks)
// ---------------------------------------------------------------------------

#define TS_DSV4_IQ3S_BLOCK_BYTES 110  // half d + qs[64] + qh[8] + signs[32] + scales[4], 256 values
#define TS_DSV4_MXFP4_BLOCK_BYTES 17  // uint8 e + qs[16], 32 values
#define TS_DSV4_Q8_0_BLOCK_BYTES 34   // half d + qs[32], 32 values

// Half of one IQ3_S superblock: iqs in [iqsBegin, iqsBegin+8), i.e. 128 of the
// 256 values against 4 consecutive q8_1 blocks. Splitting the superblock lets a
// 32-lane warp cover a 4096-wide row (only 16 superblocks) with every lane
// busy instead of half of them idle.
__device__ __forceinline__ float ts_dsv4_dot_iq3s_half(
    const uint8_t* blk, const ts_dsv4_block_q8_1* q8, int iqsBegin)
{
    const half d16 = *(const half*)blk;
    const uint8_t* qs = blk + 2;
    const uint8_t* qh = blk + 2 + 64;
    const uint8_t* signs = blk + 2 + 64 + 8;
    const uint8_t* scales = blk + 2 + 64 + 8 + 32;

    const float d = __half2float(d16);
    float sum = 0.0f;

#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int iqs = iqsBegin + 2 * j;
        int qs_pack0 = ts_dsv4_get_int_b2(qs, iqs + 0);
        int qs_pack1 = ts_dsv4_get_int_b2(qs, iqs + 1);
        const uint8_t* qsp0 = (const uint8_t*)&qs_pack0;
        const uint8_t* qsp1 = (const uint8_t*)&qs_pack1;
        uint8_t qsb[8];
        qsb[0] = qsp0[0]; qsb[1] = qsp0[1]; qsb[2] = qsp0[2]; qsb[3] = qsp0[3];
        qsb[4] = qsp1[0]; qsb[5] = qsp1[1]; qsb[6] = qsp1[2]; qsb[7] = qsp1[3];

        const int qhv = qh[iqs / 2];
        const int signs_packed_32 = ts_dsv4_get_int_b2(signs, iqs / 2);
        const uint8_t* sp8 = (const uint8_t*)&signs_packed_32;

        const ts_dsv4_block_q8_1* q8b = q8 + (iqs - iqsBegin) / 2;
        const int* q8i = (const int*)q8b->qs;

        int sumi = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2)
        {
            const uint32_t g0 = ts_iq3s_grid[qsb[l0 + 0] | ((qhv << (8 - l0)) & 0x100)];
            const uint32_t g1 = ts_iq3s_grid[qsb[l0 + 1] | ((qhv << (7 - l0)) & 0x100)];

            const int signs0 = __vcmpne4(((sp8[l0/2] & 0x03) << 7) | ((sp8[l0/2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((sp8[l0/2] & 0x30) << 3) | ((sp8[l0/2] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4((int)g0 ^ signs0, signs0);
            const int grid_h = __vsub4((int)g1 ^ signs1, signs1);

            sumi = ts_dsv4_dp4a(grid_l, q8i[l0 + 0], sumi);
            sumi = ts_dsv4_dp4a(grid_h, q8i[l0 + 1], sumi);
        }

        sumi *= 1 + 2 * ((scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
        sum += d * __half2float(q8b->d) * (float)sumi;
    }
    return sum;
}

// One IQ3_S superblock (256 values) against 8 consecutive q8_1 blocks.
// Port of ggml-cuda vec_dot_iq3_s_q8_1 (loop over iqs = 0,2,..,14).
__device__ __forceinline__ float ts_dsv4_dot_iq3s_superblock(
    const uint8_t* blk, const ts_dsv4_block_q8_1* q8)
{
    const half d16 = *(const half*)blk;
    const uint8_t* qs = blk + 2;
    const uint8_t* qh = blk + 2 + 64;
    const uint8_t* signs = blk + 2 + 64 + 8;
    const uint8_t* scales = blk + 2 + 64 + 8 + 32;

    const float d = __half2float(d16);
    float sum = 0.0f;

#pragma unroll
    for (int iqs = 0; iqs < 16; iqs += 2)
    {
        int qs_pack0 = ts_dsv4_get_int_b2(qs, iqs + 0);
        int qs_pack1 = ts_dsv4_get_int_b2(qs, iqs + 1);
        const uint8_t* qsp0 = (const uint8_t*)&qs_pack0;
        const uint8_t* qsp1 = (const uint8_t*)&qs_pack1;
        // byte j of the packed pair in ggml's flat view
        uint8_t qsb[8];
        qsb[0] = qsp0[0]; qsb[1] = qsp0[1]; qsb[2] = qsp0[2]; qsb[3] = qsp0[3];
        qsb[4] = qsp1[0]; qsb[5] = qsp1[1]; qsb[6] = qsp1[2]; qsb[7] = qsp1[3];

        const int qhv = qh[iqs / 2];
        const int signs_packed_32 = ts_dsv4_get_int_b2(signs, iqs / 2);
        const uint8_t* sp8 = (const uint8_t*)&signs_packed_32;

        const ts_dsv4_block_q8_1* q8b = q8 + iqs / 2;
        const int* q8i = (const int*)q8b->qs;

        int sumi = 0;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2)
        {
            const uint32_t g0 = ts_iq3s_grid[qsb[l0 + 0] | ((qhv << (8 - l0)) & 0x100)];
            const uint32_t g1 = ts_iq3s_grid[qsb[l0 + 1] | ((qhv << (7 - l0)) & 0x100)];

            const int signs0 = __vcmpne4(((sp8[l0/2] & 0x03) << 7) | ((sp8[l0/2] & 0x0C) << 21), 0x00000000);
            const int signs1 = __vcmpne4(((sp8[l0/2] & 0x30) << 3) | ((sp8[l0/2] & 0xC0) << 17), 0x00000000);

            const int grid_l = __vsub4((int)g0 ^ signs0, signs0);
            const int grid_h = __vsub4((int)g1 ^ signs1, signs1);

            sumi = ts_dsv4_dp4a(grid_l, q8i[l0 + 0], sumi);
            sumi = ts_dsv4_dp4a(grid_h, q8i[l0 + 1], sumi);
        }

        sumi *= 1 + 2 * ((scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
        sum += d * __half2float(q8b->d) * (float)sumi;
    }
    return sum;
}

// One MXFP4 block (32 values) against one q8_1 block.
// Port of ggml-cuda vec_dot_mxfp4_q8_1.
__device__ __forceinline__ float ts_dsv4_dot_mxfp4_block(
    const uint8_t* blk, const ts_dsv4_block_q8_1* q8)
{
    const uint8_t e = blk[0];
    const uint8_t* qs = blk + 1;
    const int* q8i = (const int*)q8->qs;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int aux = ts_dsv4_get_int_b1(qs, j);
        const int2 v = ts_dsv4_int_from_table16(aux, ts_kvalues_mxfp4);
        sumi = ts_dsv4_dp4a(v.x, q8i[j + 0], sumi);
        sumi = ts_dsv4_dp4a(v.y, q8i[j + 4], sumi);
    }

    const uint32_t ebits = ((uint32_t)e) << 23;
    const float escale = __uint_as_float(ebits);   // 2^(e-127)
    return escale * 0.5f * __half2float(q8->d) * (float)sumi;
}

// One Q8_0 block (32 values) against one q8_1 block (robustness for DSV4
// variants whose expert tensors are Q8_0).
__device__ __forceinline__ float ts_dsv4_dot_q8_0_block(
    const uint8_t* blk, const ts_dsv4_block_q8_1* q8)
{
    const half d16 = *(const half*)blk;
    const int8_t* qs = (const int8_t*)(blk + 2);
    const int* q8i = (const int*)q8->qs;

    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        int w = ts_dsv4_get_int_b2(qs, j);
        sumi = ts_dsv4_dp4a(w, q8i[j], sumi);
    }
    return __half2float(d16) * __half2float(q8->d) * (float)sumi;
}

#define TS_DSV4_Q5_1_BLOCK_BYTES 24   // half d, half m, uint32 qh, qs[16], 32 values

// One Q5_1 block's 32 five-bit codes as 8 packed ints: ints 0-3 are values 0-15 (the low nibbles
// of qs), ints 4-7 values 16-31 (the high nibbles), and value j's fifth bit is bit j of qh, placed
// as ggml-cuda's vec_dot_q5_1_q8_1 places it. A block decodes to d * code + m (ggml's
// dequantize_row_q5_1). The 24-byte blocks keep qh and qs 4-byte aligned.
__device__ __forceinline__ void ts_dsv4_q5_1_ints(const uint8_t* __restrict__ blk, int* __restrict__ w8)
{
    const int qh = ts_dsv4_get_int_b4(blk + 4, 0);
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        const int ql = ts_dsv4_get_int_b4(blk + 8, k);
        const int h0 = qh >> (4 * k);        // fifth bits of values 4k .. 4k+3
        const int h1 = qh >> (4 * k + 16);   // and of values 16+4k .. 16+4k+3
        int lo = ql & 0x0F0F0F0F;
        lo |= (h0 << 4) & 0x00000010;
        lo |= (h0 << 11) & 0x00001000;
        lo |= (h0 << 18) & 0x00100000;
        lo |= (h0 << 25) & 0x10000000;
        int hi = (ql >> 4) & 0x0F0F0F0F;
        hi |= (h1 << 4) & 0x00000010;
        hi |= (h1 << 11) & 0x00001000;
        hi |= (h1 << 18) & 0x00100000;
        hi |= (h1 << 25) & 0x10000000;
        w8[k] = lo;
        w8[k + 4] = hi;
    }
}

// One Q5_1 block against one q8_1 block: d5 * d8 * sum(code * q) + m5 * d8 * sum(q). ggml-cuda
// takes the second term from the q8_1 block's F16 sum; this file's other formats with an offset
// (Q2_K, Q4_K, Q5_K) sum the int8 activations with dp4a, so every expert route (this one, the
// register-staged kernels over the split q8_1 layout, which carries no sum) computes it the same way.
__device__ __forceinline__ float ts_dsv4_dot_q5_1_block(
    const uint8_t* __restrict__ blk, const ts_dsv4_block_q8_1* __restrict__ q8)
{
    int w8[8];
    ts_dsv4_q5_1_ints(blk, w8);
    const int* a = (const int*)q8->qs;
    int dot = 0, asum = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        dot = ts_dsv4_dp4a(w8[k], a[k], dot);
        asum = ts_dsv4_dp4a(0x01010101, a[k], asum);
    }
    const float d = __half2float(*(const half*)blk);
    const float m = __half2float(*(const half*)(blk + 2));
    return __half2float(q8->d) * (d * (float)dot + m * (float)asum);
}

// Present only in a module whose expert decoders read Q5_1. A module built from older source (a
// committed PTX baseline not yet regenerated, which machines without nvcc load) sends Q5_1 rows down
// the Q8_0 branch: wrong output and no error. Dsv4Kernels looks this name up, and the engines refuse
// Q5_1 experts on a module that lacks it. Never launched.
extern "C" __global__ void ts_dsv4_caps_q5_1_experts() {}

// Present only in a module whose expert decoders read IQ1_S and IQ2_S on every route, and IQ2_XXS on
// the grouped (register-staged and tensor-core) routes as well as the warp-per-row ones. A module
// built from older source decodes the first two as Q8_0, and IQ2_XXS too once it reaches a grouped
// kernel: Dsv4Kernels looks this name up, and CudaMoe keeps those layouts off a module that lacks it.
// Never launched.
extern "C" __global__ void ts_dsv4_caps_lowbit_iq_experts() {}

// ggml's legacy block formats that ISTA-DASLab's Qwen3.8-Flash-Next Q2_0 pack stores: Q2_0 (its routed
// experts and some dense projections), Q4_0 and Q5_0 (shared-expert down projections). Each against
// q8_1 activations with dp4a; the offset term sums the int8 activations with dp4a as Q5_1 does.

#define TS_DSV4_Q2_0_BLOCK_BYTES 18   // half d, qs[16]: 64 two-bit codes, value = (code - 1) * d
#define TS_DSV4_Q4_0_BLOCK_BYTES 18   // half d, qs[16]: 32 nibbles (values 0-15 low, 16-31 high), value = (q - 8) * d
#define TS_DSV4_Q5_0_BLOCK_BYTES 22   // half d, uint32 qh, qs[16]: value = (nibble | qh bit << 4) - 16, times d

// One Q2_0 block (64 values) against the two q8_1 blocks it spans. Byte i holds values 4i..4i+3 at bit
// offsets 0, 2, 4, 6 (ggml's dequantize_row_q2_0), which is exactly q8_1 int i's four activations.
__device__ __forceinline__ float ts_dsv4_dot_q2_0_block(
    const uint8_t* __restrict__ blk, const ts_dsv4_block_q8_1* __restrict__ q8)
{
    const uint8_t* qs = blk + 2;
    float acc = 0.0f;
#pragma unroll
    for (int b = 0; b < 2; ++b)
    {
        const int* a = (const int*)q8[b].qs;
        int dot = 0, asum = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const int c = qs[8 * b + i];
            const int w = (c & 0x03) | ((c & 0x0C) << 6) | ((c & 0x30) << 12) | ((c & 0xC0) << 18);
            dot = ts_dsv4_dp4a(w, a[i], dot);
            asum = ts_dsv4_dp4a(0x01010101, a[i], asum);
        }
        acc += __half2float(q8[b].d) * (float)(dot - asum);
    }
    return __half2float(*(const half*)blk) * acc;
}

// A quarter of a Q2_0 block (16 values, code bytes 4m..4m+3) against the q8_1 block holding them: bytes 4m..4m+3
// are values 16m..16m+15, the q8_1 block's ints 4(m & 1)..4(m & 1)+3. The block's codes start 2 bytes in.
__device__ __forceinline__ float ts_dsv4_dot_q2_0_quarter(
    const uint8_t* __restrict__ blk, int m, const ts_dsv4_block_q8_1* __restrict__ q8)
{
    const int c4 = ts_dsv4_get_int_b2(blk + 2, m);
    const int* a = (const int*)q8->qs + 4 * (m & 1);
    int dot = 0, asum = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const int c = (c4 >> (8 * i)) & 0xFF;
        const int w = (c & 0x03) | ((c & 0x0C) << 6) | ((c & 0x30) << 12) | ((c & 0xC0) << 18);
        dot = ts_dsv4_dp4a(w, a[i], dot);
        asum = ts_dsv4_dp4a(0x01010101, a[i], asum);
    }
    return __half2float(*(const half*)blk) * __half2float(q8->d) * (float)(dot - asum);
}

__device__ __forceinline__ float ts_dsv4_dot_q4_0_block(
    const uint8_t* __restrict__ blk, const ts_dsv4_block_q8_1* __restrict__ q8)
{
    const int* a = (const int*)q8->qs;
    int dot = 0, asum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        const int v = ts_dsv4_get_int_b2(blk + 2, k);
        dot = ts_dsv4_dp4a(v & 0x0F0F0F0F, a[k], dot);
        dot = ts_dsv4_dp4a((v >> 4) & 0x0F0F0F0F, a[k + 4], dot);
        asum = ts_dsv4_dp4a(0x01010101, a[k], asum);
        asum = ts_dsv4_dp4a(0x01010101, a[k + 4], asum);
    }
    return __half2float(*(const half*)blk) * __half2float(q8->d) * (float)(dot - 8 * asum);
}

// Q5_0's fifth bits sit as Q5_1's do (ts_dsv4_q5_1_ints), with qh at byte 2 and the nibbles at byte 6.
__device__ __forceinline__ float ts_dsv4_dot_q5_0_block(
    const uint8_t* __restrict__ blk, const ts_dsv4_block_q8_1* __restrict__ q8)
{
    const int qh = ts_dsv4_get_int_b2(blk + 2, 0);
    const int* a = (const int*)q8->qs;
    int dot = 0, asum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        const int ql = ts_dsv4_get_int_b2(blk + 6, k);
        const int h0 = qh >> (4 * k);
        const int h1 = qh >> (4 * k + 16);
        int lo = ql & 0x0F0F0F0F;
        lo |= (h0 << 4) & 0x00000010;
        lo |= (h0 << 11) & 0x00001000;
        lo |= (h0 << 18) & 0x00100000;
        lo |= (h0 << 25) & 0x10000000;
        int hi = (ql >> 4) & 0x0F0F0F0F;
        hi |= (h1 << 4) & 0x00000010;
        hi |= (h1 << 11) & 0x00001000;
        hi |= (h1 << 18) & 0x00100000;
        hi |= (h1 << 25) & 0x10000000;
        dot = ts_dsv4_dp4a(lo, a[k], dot);
        dot = ts_dsv4_dp4a(hi, a[k + 4], dot);
        asum = ts_dsv4_dp4a(0x01010101, a[k], asum);
        asum = ts_dsv4_dp4a(0x01010101, a[k + 4], asum);
    }
    return __half2float(*(const half*)blk) * __half2float(q8->d) * (float)(dot - 16 * asum);
}

// Present only in a module whose warp-per-row decoders (decode steps, dense rows, the warp expert route)
// read Q2_0, Q4_0 and Q5_0; an older module answers NaN for them. Never launched.
extern "C" __global__ void ts_dsv4_caps_legacy_quants() {}

#define TS_DSV4_WTYPE_Q5_1 7
#define TS_DSV4_WTYPE_Q8_0 8
#define TS_DSV4_WTYPE_Q6_K 14
#define TS_DSV4_WTYPE_IQ3S 21
#define TS_DSV4_WTYPE_MXFP4 39
#define TS_DSV4_WTYPE_Q2_K 10
#define TS_DSV4_WTYPE_Q3_K 11
#define TS_DSV4_WTYPE_Q4_K 12
#define TS_DSV4_WTYPE_Q5_K 13
#define TS_DSV4_WTYPE_IQ2XXS 16
#define TS_DSV4_WTYPE_IQ2XS 17
#define TS_DSV4_WTYPE_IQ3XXS 18
#define TS_DSV4_WTYPE_IQ1S 19
#define TS_DSV4_WTYPE_IQ4NL 20
#define TS_DSV4_WTYPE_IQ2S 22
#define TS_DSV4_WTYPE_IQ4XS 23
#define TS_DSV4_WTYPE_Q4_0 2
#define TS_DSV4_WTYPE_Q5_0 6
#define TS_DSV4_WTYPE_Q2_0 42
#define TS_DSV4_IQ4NL_BLOCK_BYTES 18  // half d + qs[16], 32 values
#define TS_DSV4_IQ4XS_BLOCK_BYTES 136 // half d, u16 scales_h, scales_l[4], qs[128], 256 values
#define TS_DSV4_Q2_K_BLOCK_BYTES 84
#define TS_DSV4_Q3_K_BLOCK_BYTES 110  // hmask[32] + qs[64] + scales[12] + half d, 256 values
#define TS_DSV4_Q4_K_BLOCK_BYTES 144  // half d, half dmin, scales[12], qs[128]
#define TS_DSV4_Q5_K_BLOCK_BYTES 176  // half d, half dmin, scales[12], qh[32], qs[128]
#define TS_DSV4_IQ2XXS_BLOCK_BYTES 66
#define TS_DSV4_IQ2XS_BLOCK_BYTES 74
#define TS_DSV4_IQ3XXS_BLOCK_BYTES 98
#define TS_DSV4_IQ2S_BLOCK_BYTES 82
#define TS_DSV4_IQ1S_BLOCK_BYTES 50
#define TS_DSV4_Q6_K_BLOCK_BYTES 210

// Full-row dot: one warp computes dot(weight row, activation row).
// inDim % 256 == 0 for every DSV4 projection, so each lane handles whole
// superblocks/blocks with no tail handling.
// One IQ2_XXS 32-value group dotted against one q8_1 activation block. Ported
// from the generic backend's dot_iq2_xxs_q8_1 (tensorsharp_kernels.cu), which
// follows ggml-cuda's vec_dot_iq2_xxs_q8_1: eight 8-value grid lookups with the
// sign permutation applied via __vsub4, accumulated with dp4a.
__device__ __forceinline__ float ts_dsv4_dot_iq2xxs_group(
    const uint8_t* iq_block, const ts_dsv4_block_q8_1* q8, int group)
{
    const ts_dsv4_block_iq2_xxs* bq2 = (const ts_dsv4_block_iq2_xxs*)iq_block;
    const int iqs = group * 2;
    const int q2 = ts_dsv4_get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = (const uint8_t*)&q2;
    const uint32_t aux32 = (uint32_t)ts_dsv4_get_int_b2(bq2->qs, iqs + 1);

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2)
    {
        const int* gridPos = (const int*)(iq2xxs_grid + aux8[k0 / 2]);
        const int signsPacked = ksigns_iq2xs[(aux32 >> (7 * k0 / 2)) & 0x7F];

        const int signs0 = __vcmpne4(((signsPacked & 0x03) << 7) | ((signsPacked & 0x0C) << 21), 0x00000000);
        const int grid0 = __vsub4(gridPos[0] ^ signs0, signs0);
        sumi = ts_dsv4_dp4a(grid0, ts_dsv4_get_int_b4(q8[group].qs, k0 + 0), sumi);

        const int signs1 = __vcmpne4(((signsPacked & 0x30) << 3) | ((signsPacked & 0xC0) << 17), 0x00000000);
        const int grid1 = __vsub4(gridPos[1] ^ signs1, signs1);
        sumi = ts_dsv4_dp4a(grid1, ts_dsv4_get_int_b4(q8[group].qs, k0 + 1), sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 4;
    return __half2float(bq2->d) * __half2float(q8[group].d) * (float)sumi;
}

// One IQ2_XS 32-value group dotted against one q8_1 activation block. Ported
// from ggml-cuda's vec_dot_iq2_xs_q8_1: four 8-value lookups into the 512-entry
// grid, each with its own sign byte (7 stored bits, the 8th the parity), and two
// 4-bit scales, one per 16 values.
__device__ __forceinline__ float ts_dsv4_dot_iq2xs_group(
    const uint8_t* iq_block, const ts_dsv4_block_q8_1* q8, int group)
{
    const ts_dsv4_block_iq2_xs* bq2 = (const ts_dsv4_block_iq2_xs*)iq_block;
    const int iqs = group * 2;
    const int qa = ts_dsv4_get_int_b2(bq2->qs, iqs + 0);
    const int qb = ts_dsv4_get_int_b2(bq2->qs, iqs + 1);
    const uint32_t q2[4] = { (uint32_t)qa & 0xFFFF, (uint32_t)qa >> 16, (uint32_t)qb & 0xFFFF, (uint32_t)qb >> 16 };
    const int ls0 = bq2->scales[group] & 0x0F;
    const int ls1 = bq2->scales[group] >> 4;

    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2)
    {
        const uint2 gridPos = ((const uint2*)iq2xs_grid)[q2[l0 / 2] & 0x1FF];
        const uint32_t v = q2[l0 / 2] >> 9;
        const uint32_t signs = (v ^ ((uint32_t)(__popc(v) & 1) << 7)) * 0x01010101u;

        const int signs0 = __vcmpne4(signs & 0x08040201u, 0);
        const int gridL = __vsub4((int)gridPos.x ^ signs0, signs0);
        const int signs1 = __vcmpne4(signs & 0x80402010u, 0);
        const int gridH = __vsub4((int)gridPos.y ^ signs1, signs1);

        const int u0 = ts_dsv4_get_int_b4(q8[group].qs, l0 + 0);
        const int u1 = ts_dsv4_get_int_b4(q8[group].qs, l0 + 1);
        if (l0 < 4)
        {
            sumi0 = ts_dsv4_dp4a(gridL, u0, sumi0);
            sumi0 = ts_dsv4_dp4a(gridH, u1, sumi0);
        }
        else
        {
            sumi1 = ts_dsv4_dp4a(gridL, u0, sumi1);
            sumi1 = ts_dsv4_dp4a(gridH, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return __half2float(bq2->d) * __half2float(q8[group].d) * (float)sumi;
}

// One IQ3_XXS 32-value group dotted against one q8_1 activation block. Ported
// from ggml-cuda's vec_dot_iq3_xxs_q8_1: eight 4-value lookups into the
// 256-entry grid; the group's uint32 carries four 7-bit sign indices (one per
// 8 values) and the 4-bit scale. Signs expand through ksigns_iq2xs exactly as in
// the IQ2_XXS dot above.
__device__ __forceinline__ float ts_dsv4_dot_iq3xxs_group(
    const uint8_t* iq_block, const ts_dsv4_block_q8_1* q8, int group)
{
    const ts_dsv4_block_iq3_xxs* bq3 = (const ts_dsv4_block_iq3_xxs*)iq_block;
    const int iqs = group * 2;
    const int qa = ts_dsv4_get_int_b2(bq3->qs, iqs + 0);
    const int qb = ts_dsv4_get_int_b2(bq3->qs, iqs + 1);
    const uint8_t* q3a = (const uint8_t*)&qa;
    const uint8_t* q3b = (const uint8_t*)&qb;
    // qs[0..63] are grid indices; the group's sign/scale word follows at byte 64 + 4 * group.
    const uint32_t aux32 = (uint32_t)ts_dsv4_get_int_b2(bq3->qs, 16 + iqs / 2);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2)
    {
        const uint8_t i0 = l0 < 4 ? q3a[l0] : q3b[l0 - 4];
        const uint8_t i1 = l0 < 4 ? q3a[l0 + 1] : q3b[l0 - 3];
        const int signsPacked = ksigns_iq2xs[(aux32 >> (7 * l0 / 2)) & 0x7F];
        const int signs0 = __vcmpne4(((signsPacked & 0x03) << 7) | ((signsPacked & 0x0C) << 21), 0x00000000);
        const int gridL = __vsub4((int)iq3xxs_grid[i0] ^ signs0, signs0);
        const int signs1 = __vcmpne4(((signsPacked & 0x30) << 3) | ((signsPacked & 0xC0) << 17), 0x00000000);
        const int gridH = __vsub4((int)iq3xxs_grid[i1] ^ signs1, signs1);

        sumi = ts_dsv4_dp4a(gridL, ts_dsv4_get_int_b4(q8[group].qs, l0 + 0), sumi);
        sumi = ts_dsv4_dp4a(gridH, ts_dsv4_get_int_b4(q8[group].qs, l0 + 1), sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    return __half2float(bq3->d) * __half2float(q8[group].d) * (float)sumi;
}

// One IQ2_S 32-value group as 8 packed signed ints (ggml's dequantize_row_iq2_s before the scale):
// four 8-value lookups into the 1024-entry grid, each index's low 8 bits from qs and its high 2 from
// the group's qh byte, with an explicit sign byte per 8 values. The low-bit IQ decoders read their
// blocks two bytes at a time: a 2304-wide expert row (V4.1 1017's down projection) is 738 bytes of
// IQ2_S, 594 of IQ2_XXS and 450 of IQ1_S, so every other row starts off a 4-byte boundary.
__device__ __forceinline__ void ts_dsv4_iq2s_group_ints(
    const ts_dsv4_block_iq2_s* b, int group, int* __restrict__ w8)
{
    const int qsPacked = ts_dsv4_get_int_b2(b->qs, group);
    const uint8_t* qs = (const uint8_t*)&qsPacked;
    const int signsPacked = ts_dsv4_get_int_b2(b->qs, 8 + group);
    const uint8_t* signs = (const uint8_t*)&signsPacked;
    const int qh = b->qh[group];
#pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        const uint2 gridPos = ((const uint2*)iq2s_grid)[qs[l] | ((qh << (8 - 2 * l)) & 0x300)];
        const int s0 = __vcmpne4(((signs[l] & 0x03) << 7) | ((signs[l] & 0x0C) << 21), 0x00000000);
        const int s1 = __vcmpne4(((signs[l] & 0x30) << 3) | ((signs[l] & 0xC0) << 17), 0x00000000);
        w8[2 * l + 0] = __vsub4((int)gridPos.x ^ s0, s0);
        w8[2 * l + 1] = __vsub4((int)gridPos.y ^ s1, s1);
    }
}

// One IQ2_S 32-value group dotted against one q8_1 activation block. Ported from ggml-cuda's
// vec_dot_iq2_s_q8_1 (as the generic backend's dot_iq2_s_q8_1 is), keeping its integer rounding of
// the two 4-bit scales, d * (0.5 + scale) / 4 per 16 values.
__device__ __forceinline__ float ts_dsv4_dot_iq2s_group(
    const uint8_t* iq_block, const ts_dsv4_block_q8_1* q8, int group)
{
    const ts_dsv4_block_iq2_s* bq2 = (const ts_dsv4_block_iq2_s*)iq_block;
    int w8[8];
    ts_dsv4_iq2s_group_ints(bq2, group, w8);
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        sumi0 = ts_dsv4_dp4a(w8[k], ts_dsv4_get_int_b4(q8[group].qs, k), sumi0);
        sumi1 = ts_dsv4_dp4a(w8[k + 4], ts_dsv4_get_int_b4(q8[group].qs, k + 4), sumi1);
    }
    const int ls0 = bq2->scales[group] & 0x0F;
    const int ls1 = bq2->scales[group] >> 4;
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    return __half2float(bq2->d) * __half2float(q8[group].d) * (float)sumi;
}

// One IQ1_S 32-value group as 8 packed signed ints, returning their scale. ggml's
// dequantize_row_iq1_s makes value j d * (2s + 1) * (g_j + delta), g_j in {-1, 0, 1} from the
// 2048-entry grid (whose GPU form stores g_j + 1) and delta = +/-1/8 by the top qh bit. Times 8 that
// is the integer 8 g_j +/- 1 (|.| <= 9), so the group decodes exactly into int8 under the scale
// d * (2s + 1) / 8 with no offset: no route needs the activation sum, which the register-staged
// kernels' split q8_1 layout does not carry (ggml-cuda's vec_dot_iq1_s_q8_1 takes the delta term
// from the q8_1 block's F16 sum instead).
__device__ __forceinline__ float ts_dsv4_iq1s_group_ints(
    const ts_dsv4_block_iq1_s* b, int group, int* __restrict__ w8)
{
    const int qsPacked = ts_dsv4_get_int_b2(b->qs, group);
    const uint8_t* qs = (const uint8_t*)&qsPacked;
    const int qh = b->qh[group];
    // 8 (g + 1) - 7 = 8 g + 1 for delta +1/8, 8 (g + 1) - 9 = 8 g - 1 for -1/8.
    const int bias = (qh & 0x8000) ? 0x09090909 : 0x07070707;
#pragma unroll
    for (int l = 0; l < 4; ++l)
    {
        const int grid = (int)iq1s_grid_gpu[qs[l] | (((qh >> (3 * l)) & 0x07) << 8)];
        w8[2 * l + 0] = __vsub4((grid & 0x0F0F0F0F) << 3, bias);
        w8[2 * l + 1] = __vsub4(((grid >> 4) & 0x0F0F0F0F) << 3, bias);
    }
    return __half2float(b->d) * (float)(2 * ((qh >> 12) & 0x07) + 1) * 0.125f;
}

// One IQ1_S 32-value group dotted against one q8_1 activation block (ggml-cuda's
// vec_dot_iq1_s_q8_1 over the exact integer form above).
__device__ __forceinline__ float ts_dsv4_dot_iq1s_group(
    const uint8_t* iq_block, const ts_dsv4_block_q8_1* q8, int group)
{
    int w8[8];
    const float sc = ts_dsv4_iq1s_group_ints((const ts_dsv4_block_iq1_s*)iq_block, group, w8);
    int sumi = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k)
        sumi = ts_dsv4_dp4a(w8[k], ts_dsv4_get_int_b4(q8[group].qs, k), sumi);
    return sc * __half2float(q8[group].d) * (float)sumi;
}

// One Q2_K 32-value group dotted against one q8_1 activation block.
//
// block_q2_K (84 bytes / 256 values): scales[16] (low nibble = value scale,
// high nibble = min scale), qs[64] (2 bits per value), d, dmin. Group g takes
// its 2-bit field from byte lane (g & 3) of the 32-byte half (g >> 2), and its
// two 16-value halves use consecutive scale bytes. The min term needs the
// activation sums per 16-value half, so it is accumulated with dp4a against
// 0x01010101 rather than read from the block's `s` field (which covers all 32).
__device__ __forceinline__ float ts_dsv4_dot_q2k_group(
    const uint8_t* block, const ts_dsv4_block_q8_1* q8, int group)
{
    const uint8_t* scales = block;
    const int gh = group >> 2;               // which 128-value half
    const int j = group & 3;                 // 2-bit field within the byte
    const uint8_t* qs = block + 16 + gh * 32;
    const int shift = 2 * j;
    const float d = __half2float(*(const half*)(block + 80));
    const float dmin = __half2float(*(const half*)(block + 82));
    const uint8_t sc0 = scales[gh * 8 + j * 2 + 0];
    const uint8_t sc1 = scales[gh * 8 + j * 2 + 1];

    int sumi0 = 0, sumi1 = 0, sa0 = 0, sa1 = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        const int w = (ts_dsv4_get_int_b4(qs, k) >> shift) & 0x03030303;
        const int u = ts_dsv4_get_int_b4(q8[group].qs, k);
        sumi0 = ts_dsv4_dp4a(w, u, sumi0);
        sa0 = ts_dsv4_dp4a(0x01010101, u, sa0);
    }
#pragma unroll
    for (int k = 4; k < 8; ++k)
    {
        const int w = (ts_dsv4_get_int_b4(qs, k) >> shift) & 0x03030303;
        const int u = ts_dsv4_get_int_b4(q8[group].qs, k);
        sumi1 = ts_dsv4_dp4a(w, u, sumi1);
        sa1 = ts_dsv4_dp4a(0x01010101, u, sa1);
    }

    const float da = __half2float(q8[group].d);
    const float val = d * ((float)(sc0 & 0xF) * (float)sumi0 + (float)(sc1 & 0xF) * (float)sumi1)
                    - dmin * ((float)(sc0 >> 4) * (float)sa0 + (float)(sc1 >> 4) * (float)sa1);
    return da * val;
}

// The signed 6-bit scale of 16-value sub-block `is` (0..15) of a Q3_K superblock: ggml packs the low
// four bits of scales 0-7 and 8-15 into the low and high nibbles of bytes 0-7, and their top two bits
// into bytes 8-11, two bits per sub-block (its kmask1/kmask2 unpack).
__device__ __forceinline__ int ts_dsv4_q3k_scale(const uint8_t* scales, int is)
{
    const int lo4 = is < 8 ? (scales[is] & 0xF) : (scales[is - 8] >> 4);
    const int hi2 = (scales[8 + (is & 3)] >> (2 * (is >> 2))) & 3;
    return (lo4 | (hi2 << 4)) - 32;
}

// One Q3_K 32-value group as 8 packed signed ints (-4..3) and the scales of its two 16-value halves.
// block_q3_K (110 bytes / 256 values): hmask[32], qs[64], scales[12], d. Group g sits in the 128-value
// half (g >> 2): its 2-bit field is bit pair (g & 3) of qs[half * 32 + l], and a clear hmask bit
// (half * 4 + (g & 3)) of hmask[l] subtracts 4.
__device__ __forceinline__ void ts_dsv4_q3k_group_ints(
    const uint8_t* block, int group, int* __restrict__ w8, float* __restrict__ scLo, float* __restrict__ scHi)
{
    const uint8_t* hmask = block;
    const uint8_t* q = block + 32 + (group >> 2) * 32;
    const int j = group & 3;
    const int bit = (group >> 2) * 4 + j;
#pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        const int q2 = (ts_dsv4_get_int_b2(q, k) >> (2 * j)) & 0x03030303;
        const int hb = (ts_dsv4_get_int_b2(hmask, k) >> bit) & 0x01010101;
        w8[k] = __vsubss4(q2, (hb ^ 0x01010101) << 2);
    }
    const float d = __half2float(*(const half*)(block + 108));
    const int is = (group >> 2) * 8 + j * 2;
    *scLo = d * (float)ts_dsv4_q3k_scale(block + 96, is);
    *scHi = d * (float)ts_dsv4_q3k_scale(block + 96, is + 1);
}

// One Q3_K 32-value group dotted against one q8_1 activation block.
__device__ __forceinline__ float ts_dsv4_dot_q3k_group(
    const uint8_t* block, const ts_dsv4_block_q8_1* q8, int group)
{
    int w[8];
    float scLo, scHi;
    ts_dsv4_q3k_group_ints(block, group, w, &scLo, &scHi);
    int lo = 0, hi = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
        lo = ts_dsv4_dp4a(w[k], ts_dsv4_get_int_b4(q8[group].qs, k), lo);
#pragma unroll
    for (int k = 4; k < 8; ++k)
        hi = ts_dsv4_dp4a(w[k], ts_dsv4_get_int_b4(q8[group].qs, k), hi);
    return __half2float(q8[group].d) * (scLo * (float)lo + scHi * (float)hi);
}

// One element of a K-quant row (Q2_K..Q6_K), for gathers such as the token embedding.
__device__ __forceinline__ float ts_dsv4_kquant_value(const uint8_t* row, int wtype, int e)
{
    const int i = e & 255;
    if (wtype == TS_DSV4_WTYPE_Q2_K)
    {
        const uint8_t* b = row + (size_t)(e >> 8) * TS_DSV4_Q2_K_BLOCK_BYTES;
        const int h = i >> 7, j = (i >> 5) & 3, l = i & 31;
        const int q = (b[16 + h * 32 + l] >> (2 * j)) & 3;
        const uint8_t sc = b[h * 8 + j * 2 + (l >> 4)];
        return __half2float(*(const half*)(b + 80)) * (float)(sc & 0xF) * (float)q
             - __half2float(*(const half*)(b + 82)) * (float)(sc >> 4);
    }
    if (wtype == TS_DSV4_WTYPE_Q3_K)
    {
        const uint8_t* b = row + (size_t)(e >> 8) * TS_DSV4_Q3_K_BLOCK_BYTES;
        const int h = i >> 7, j = (i >> 5) & 3, l = i & 31;
        const int q = ((b[32 + h * 32 + l] >> (2 * j)) & 3) - ((b[l] >> (h * 4 + j)) & 1 ? 0 : 4);
        return __half2float(*(const half*)(b + 108)) * (float)ts_dsv4_q3k_scale(b + 96, h * 8 + j * 2 + (l >> 4)) * (float)q;
    }
    if (wtype == TS_DSV4_WTYPE_Q4_K || wtype == TS_DSV4_WTYPE_Q5_K)
    {
        const bool q5 = wtype == TS_DSV4_WTYPE_Q5_K;
        const uint8_t* b = row + (size_t)(e >> 8) * (q5 ? TS_DSV4_Q5_K_BLOCK_BYTES : TS_DSV4_Q4_K_BLOCK_BYTES);
        const uint8_t* sc = b + 4;
        const int c = i >> 6, hi = (i >> 5) & 1, l = i & 31;
        const int is = 2 * c + hi;
        int scale, min;
        if (is < 4) { scale = sc[is] & 63; min = sc[is + 4] & 63; }
        else { scale = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); min = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
        const uint8_t* qs = b + (q5 ? 48 : 16);
        int q = hi ? qs[c * 32 + l] >> 4 : qs[c * 32 + l] & 0xF;
        if (q5 && ((b[16 + l] >> (2 * c + hi)) & 1)) q += 16;
        return __half2float(*(const half*)b) * (float)scale * (float)q - __half2float(*(const half*)(b + 2)) * (float)min;
    }
    // Q6_K
    const uint8_t* b = row + (size_t)(e >> 8) * TS_DSV4_Q6_K_BLOCK_BYTES;
    const int h = i >> 7, qq = (i >> 5) & 3, l = i & 31;
    const uint8_t ql = b[h * 64 + l + ((qq & 1) ? 32 : 0)];
    const int nib = qq >= 2 ? ql >> 4 : ql & 0xF;
    const int q = (nib | (((b[128 + h * 32 + l] >> (2 * qq)) & 3) << 4)) - 32;
    const int8_t scale = ((const int8_t*)(b + 192))[h * 8 + (l >> 4) + 2 * qq];
    return __half2float(*(const half*)(b + 208)) * (float)scale * (float)q;
}

__device__ __forceinline__ void ts_dsv4_decode_sub(
    const uint8_t* __restrict__ wRow, int wtype, int sub, int* __restrict__ w8,
    float* __restrict__ scLo, float* __restrict__ scHi,
    float* __restrict__ mnLo, float* __restrict__ mnHi);

// One 32-value sub-block of any type ts_dsv4_decode_sub reads, dotted against its q8_1
// activation block: the two 16-value halves' scales, minus their mins times the halves'
// activation sums.
__device__ __forceinline__ float ts_dsv4_dot_sub_q8_1(
    const uint8_t* wRow, int wtype, int sub, const ts_dsv4_block_q8_1* q8)
{
    int w8[8];
    float scLo, scHi, mnLo, mnHi;
    ts_dsv4_decode_sub(wRow, wtype, sub, w8, &scLo, &scHi, &mnLo, &mnHi);
    const int* a = (const int*)q8->qs;
    int lo = 0, hi = 0, sLo = 0, sHi = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        lo = ts_dsv4_dp4a(w8[k], a[k], lo);
        sLo = ts_dsv4_dp4a(0x01010101, a[k], sLo);
        hi = ts_dsv4_dp4a(w8[k + 4], a[k + 4], hi);
        sHi = ts_dsv4_dp4a(0x01010101, a[k + 4], sHi);
    }
    return __half2float(q8->d) * (scLo * (float)lo + scHi * (float)hi - mnLo * (float)sLo - mnHi * (float)sHi);
}

// A pair of Q4_K / Q5_K 32-value sub-blocks (2c, 2c+1 of a 256-value super-block) against their
// q8_1 activation blocks. The pair shares its 32 bytes of nibbles - the low half of each byte is
// the even sub-block, the high half the odd - and, for Q5_K, the super-block's 32 bytes of high
// bits; both are read as 16-byte words (the rows are 16-byte aligned), where one lane per
// sub-block read the same bytes twice as eight 4-byte words each.
__device__ __forceinline__ float ts_dsv4_dot_q45k_pair(
    const uint8_t* __restrict__ wRow, const bool q5, const int pair, const ts_dsv4_block_q8_1* __restrict__ act)
{
    const int sb = pair >> 2;
    const int c = pair & 3;
    const uint8_t* blk = wRow + (size_t)sb * (q5 ? TS_DSV4_Q5_K_BLOCK_BYTES : TS_DSV4_Q4_K_BLOCK_BYTES);
    const uint4* qs = reinterpret_cast<const uint4*>(blk + (q5 ? 48 : 16) + c * 32);
    const uint4 q0 = qs[0], q1 = qs[1];
    const int qw[8] = { (int)q0.x, (int)q0.y, (int)q0.z, (int)q0.w, (int)q1.x, (int)q1.y, (int)q1.z, (int)q1.w };
    int hw[8];
    if (q5)
    {
        const uint4* qh = reinterpret_cast<const uint4*>(blk + 16);
        const uint4 h0 = qh[0], h1 = qh[1];
        hw[0] = (int)h0.x; hw[1] = (int)h0.y; hw[2] = (int)h0.z; hw[3] = (int)h0.w;
        hw[4] = (int)h1.x; hw[5] = (int)h1.y; hw[6] = (int)h1.z; hw[7] = (int)h1.w;
    }
    const float d = __half2float(*reinterpret_cast<const half*>(blk));
    const float dmin = __half2float(*reinterpret_cast<const half*>(blk + 2));
    const uint8_t* sc = blk + 4;

    float sum = 0.0f;
#pragma unroll
    for (int hi = 0; hi < 2; ++hi)
    {
        const int is = 2 * c + hi;
        int scale, mn;
        if (is < 4) { scale = sc[is] & 63; mn = sc[is + 4] & 63; }
        else { scale = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); mn = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
        const ts_dsv4_block_q8_1* a8 = act + sb * 8 + is;
        const int* a = reinterpret_cast<const int*>(a8->qs);
        int dot = 0, asum = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            int v = hi ? ((qw[k] >> 4) & 0x0F0F0F0F) : (qw[k] & 0x0F0F0F0F);
            if (q5)
                v |= ((hw[k] >> is) & 0x01010101) << 4;
            dot = ts_dsv4_dp4a(v, a[k], dot);
            asum = ts_dsv4_dp4a(0x01010101, a[k], asum);
        }
        sum += __half2float(a8->d) * (d * (float)scale * (float)dot - dmin * (float)mn * (float)asum);
    }
    return sum;
}

// One thread's share of a row against q8_1 activations: the units first, first + step, ... of the row's
// decode units (blocks, groups or half-superblocks by format). A warp passes (lane, 32); a split-K kernel
// passes its block's thread index and thread count.
__device__ __forceinline__ float ts_dsv4_dot_row_part(
    const uint8_t* wRow, int wtype, const ts_dsv4_block_q8_1* act, int inDim, int first, int step)
{
    float sum = 0.0f;
    if (wtype == TS_DSV4_WTYPE_IQ3S)
    {
        // Two half-superblocks per lane-pair: a 4096-wide row is only 16
        // superblocks, so whole-superblock striding would idle half the warp.
        const int nHalves = inDim / 128;
        for (int h = first; h < nHalves; h += step)
        {
            const int sb = h >> 1;
            const int iqsBegin = (h & 1) ? 8 : 0;
            sum += ts_dsv4_dot_iq3s_half(
                wRow + (size_t)sb * TS_DSV4_IQ3S_BLOCK_BYTES, act + sb * 8 + (iqsBegin / 2), iqsBegin);
        }
    }
    else if (wtype == TS_DSV4_WTYPE_MXFP4)
    {
        const int nBlk = inDim / 32;
        for (int b = first; b < nBlk; b += step)
            sum += ts_dsv4_dot_mxfp4_block(wRow + (size_t)b * TS_DSV4_MXFP4_BLOCK_BYTES, act + b);
    }
    else if (wtype == TS_DSV4_WTYPE_IQ2XXS)
    {
        // 8 groups of 32 values per super-block: stride whole groups so a
        // 4096-wide row (16 super-blocks) still fills the warp.
        const int nGroups = inDim / 32;
        for (int g = first; g < nGroups; g += step)
            sum += ts_dsv4_dot_iq2xxs_group(wRow + (size_t)(g >> 3) * TS_DSV4_IQ2XXS_BLOCK_BYTES,
                                            act + (size_t)(g >> 3) * 8, g & 7);
    }
    else if (wtype == TS_DSV4_WTYPE_IQ2XS)
    {
        const int nGroups = inDim / 32;
        for (int g = first; g < nGroups; g += step)
            sum += ts_dsv4_dot_iq2xs_group(wRow + (size_t)(g >> 3) * TS_DSV4_IQ2XS_BLOCK_BYTES,
                                           act + (size_t)(g >> 3) * 8, g & 7);
    }
    else if (wtype == TS_DSV4_WTYPE_IQ3XXS)
    {
        const int nGroups = inDim / 32;
        for (int g = first; g < nGroups; g += step)
            sum += ts_dsv4_dot_iq3xxs_group(wRow + (size_t)(g >> 3) * TS_DSV4_IQ3XXS_BLOCK_BYTES,
                                            act + (size_t)(g >> 3) * 8, g & 7);
    }
    else if (wtype == TS_DSV4_WTYPE_Q2_K)
    {
        const int nGroups = inDim / 32;
        for (int g = first; g < nGroups; g += step)
            sum += ts_dsv4_dot_q2k_group(wRow + (size_t)(g >> 3) * TS_DSV4_Q2_K_BLOCK_BYTES,
                                         act + (size_t)(g >> 3) * 8, g & 7);
    }
    else if (wtype == TS_DSV4_WTYPE_Q3_K)
    {
        const int nGroups = inDim / 32;
        for (int g = first; g < nGroups; g += step)
            sum += ts_dsv4_dot_q3k_group(wRow + (size_t)(g >> 3) * TS_DSV4_Q3_K_BLOCK_BYTES,
                                         act + (size_t)(g >> 3) * 8, g & 7);
    }
    else if (wtype == TS_DSV4_WTYPE_Q4_K || wtype == TS_DSV4_WTYPE_Q5_K)
    {
        const bool q5 = wtype == TS_DSV4_WTYPE_Q5_K;
        const int nPairs = inDim / 64;
        for (int p = first; p < nPairs; p += step)
            sum += ts_dsv4_dot_q45k_pair(wRow, q5, p, act);
    }
    else if (wtype == TS_DSV4_WTYPE_Q6_K || wtype == TS_DSV4_WTYPE_IQ4NL || wtype == TS_DSV4_WTYPE_IQ4XS)
    {
        // One 32-value sub-block per lane: a whole 256-value super-block per lane would leave
        // 24 of 32 lanes idle on a 2048-wide expert row.
        const int nGroups = inDim / 32;
        for (int g = first; g < nGroups; g += step)
            sum += ts_dsv4_dot_sub_q8_1(wRow, wtype, g, act + g);
    }
    else if (wtype == TS_DSV4_WTYPE_Q5_1)
    {
        // Unsloth's UD-Q4_K_XL keeps most of Qwen3.8-Flash-Next's expert down projections in Q5_1.
        const int nBlk = inDim / 32;
        for (int b = first; b < nBlk; b += step)
            sum += ts_dsv4_dot_q5_1_block(wRow + (size_t)b * TS_DSV4_Q5_1_BLOCK_BYTES, act + b);
    }
    else if (wtype == TS_DSV4_WTYPE_IQ2S)
    {
        // Unsloth's DeepSeek V4.1 1017 UD-IQ2_XXS keeps its gate/up experts in IQ1_S, IQ2_S or
        // IQ2_XXS and most down projections in IQ2_S, IQ3_XXS or IQ2_XS.
        const int nGroups = inDim / 32;
        for (int g = first; g < nGroups; g += step)
            sum += ts_dsv4_dot_iq2s_group(wRow + (size_t)(g >> 3) * TS_DSV4_IQ2S_BLOCK_BYTES,
                                          act + (size_t)(g >> 3) * 8, g & 7);
    }
    else if (wtype == TS_DSV4_WTYPE_IQ1S)
    {
        const int nGroups = inDim / 32;
        for (int g = first; g < nGroups; g += step)
            sum += ts_dsv4_dot_iq1s_group(wRow + (size_t)(g >> 3) * TS_DSV4_IQ1S_BLOCK_BYTES,
                                          act + (size_t)(g >> 3) * 8, g & 7);
    }
    else if (wtype == TS_DSV4_WTYPE_Q8_0)
    {
        const int nBlk = inDim / 32;
        for (int b = first; b < nBlk; b += step)
            sum += ts_dsv4_dot_q8_0_block(wRow + (size_t)b * TS_DSV4_Q8_0_BLOCK_BYTES, act + b);
    }
    else if (wtype == TS_DSV4_WTYPE_Q2_0)
    {
        // 16-value units (4 code bytes): a 640-wide expert row is 10 blocks but 40 units, so the warp stays busy.
        const int nUnits = inDim / 16;
        for (int u = first; u < nUnits; u += step)
            sum += ts_dsv4_dot_q2_0_quarter(wRow + (size_t)(u >> 2) * TS_DSV4_Q2_0_BLOCK_BYTES, u & 3, act + 2 * (u >> 2) + ((u & 3) >> 1));
    }
    else if (wtype == TS_DSV4_WTYPE_Q4_0)
    {
        const int nBlk = inDim / 32;
        for (int b = first; b < nBlk; b += step)
            sum += ts_dsv4_dot_q4_0_block(wRow + (size_t)b * TS_DSV4_Q4_0_BLOCK_BYTES, act + b);
    }
    else if (wtype == TS_DSV4_WTYPE_Q5_0)
    {
        const int nBlk = inDim / 32;
        for (int b = first; b < nBlk; b += step)
            sum += ts_dsv4_dot_q5_0_block(wRow + (size_t)b * TS_DSV4_Q5_0_BLOCK_BYTES, act + b);
    }
    else
    {
        // A layout no branch decodes answers NaN: this chain once ended in a bare Q8_0 `else`, so a
        // gate admitting a type before its decoder read the rows as Q8_0, wrong with no error.
        sum = __int_as_float(0x7fc00000);
    }
    return sum;
}

__device__ __forceinline__ float ts_dsv4_dot_row_warp(
    const uint8_t* wRow, int wtype, const ts_dsv4_block_q8_1* act, int inDim, int lane)
{
    return ts_dsv4_warp_sum(ts_dsv4_dot_row_part(wRow, wtype, act, inDim, lane, 32));
}

// ---------------------------------------------------------------------------
// token embedding: dequantize token rows and replicate over the 4 HC streams
// ---------------------------------------------------------------------------

// One value of a stored row: Q8_0, the K-quants, F16, BF16 or F32.
__device__ __forceinline__ float ts_dsv4_row_value(const uint8_t* row, int wtype, int e)
{
    if (wtype == TS_DSV4_WTYPE_Q8_0)
    {
        const uint8_t* blk = row + (e / 32) * TS_DSV4_Q8_0_BLOCK_BYTES;
        return __half2float(*(const half*)blk) * (float)((const int8_t*)(blk + 2))[e % 32];
    }
    if (wtype == TS_DSV4_WTYPE_Q2_K || wtype == TS_DSV4_WTYPE_Q3_K || wtype == TS_DSV4_WTYPE_Q4_K
        || wtype == TS_DSV4_WTYPE_Q5_K || wtype == TS_DSV4_WTYPE_Q6_K)
        return ts_dsv4_kquant_value(row, wtype, e);
    if (wtype == 1) // F16
        return __half2float(((const half*)row)[e]);
    if (wtype == 30) // BF16
        return __uint_as_float((unsigned int)((const unsigned short*)row)[e] << 16);
    return ((const float*)row)[e]; // F32
}

// V4.1 Engram gather from a device-resident table: block b is (token b / columns, hash column
// b % columns), dequantizing row rows[b] into out[b * headDim ...].
extern "C" __global__ void ts_dsv41_engram_gather_f32(
    const uint8_t* __restrict__ table,
    const int wtype,
    const long long rowBytes,
    const int32_t* __restrict__ rows,   // [nt, columns]
    float* __restrict__ out,            // [nt, columns * headDim]
    const int headDim)
{
    const int b = blockIdx.x;
    const uint8_t* row = table + (size_t)rows[b] * rowBytes;
    float* dst = out + (size_t)b * headDim;
    for (int e = threadIdx.x; e < headDim; e += blockDim.x)
        dst[e] = ts_dsv4_row_value(row, wtype, e);
}

extern "C" __global__ void ts_dsv4_embed_f32(
    const uint8_t* __restrict__ w,   // token_embd rows
    const int32_t* __restrict__ tokens,
    float* __restrict__ xs,          // [nt, 4, E]
    const int wtype,                 // Q8_0 / K-quant / F16 / BF16 / F32
    const long long rowBytes,
    const int nt,
    const int E)
{
    const int t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt || e >= E)
        return;

    const float v = ts_dsv4_row_value(w + (size_t)tokens[t] * rowBytes, wtype, e);

    float* dst = xs + ((size_t)t * 4) * E + e;
#pragma unroll
    for (int s = 0; s < 4; ++s)
        dst[(size_t)s * E] = v;
}

// ---------------------------------------------------------------------------
// hyper-connections
// ---------------------------------------------------------------------------

// inv[t] = rsqrt(mean(flat^2) + eps) over the flattened 4*E streams.
extern "C" __global__ void ts_dsv4_hc_rms_f32(
    const float* __restrict__ xs,
    float* __restrict__ inv,
    const int flatDim,
    const float eps)
{
    const int t = blockIdx.x;
    const float* x = xs + (size_t)t * flatDim;

    __shared__ float red[256];
    float acc = 0.0f;
    for (int i = threadIdx.x; i < flatDim; i += blockDim.x)
    {
        const float v = x[i];
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        inv[t] = rsqrtf(red[0] / flatDim + eps);
}

#define TS_DSV4_GEMV_MAX_ROWS 16

// y[rows, outDim] = x[rows, inDim] . w^T for a small F32 weight (the hyper-connection mixes,
// the MoE router) and up to TS_DSV4_GEMV_MAX_ROWS rows. grid (outDim, splits): each CTA streams
// one output's weights over a 1/splits share of the input once and applies them to every
// activation row. The mixes have only 24 outputs over a 20480-wide input, so they are split
// along the input (the partial sums combined in split order by ts_dsv4_gemv_combine_f32); the
// router's 384 outputs fill the device unsplit. The split count depends on the shape only, so a
// row's sum is the same whatever the row count.
template <int MAXR>
__device__ __forceinline__ void ts_dsv4_gemv_rows(
    const float* __restrict__ x,      // [rows, inDim], inDim % 4 == 0
    const float* __restrict__ w,      // [outDim, inDim]
    float* __restrict__ y,            // [rows, outDim] (splits == 1)
    float* __restrict__ partials,     // [rows, outDim, splits] (splits > 1)
    const int inDim,
    const int outDim,
    const int rows,
    const int splits)
{
    __shared__ float sh[32];
    const int o = blockIdx.x;
    const int split = blockIdx.y;
    if (o >= outDim)
        return;
    const int n4 = inDim / 4;
    const int chunk = (n4 + splits - 1) / splits;
    const int begin = split * chunk;
    const int end = min(n4, begin + chunk);
    const float4* w4 = (const float4*)(w + (size_t)o * inDim);
    float acc[MAXR];
#pragma unroll
    for (int r = 0; r < MAXR; ++r)
        acc[r] = 0.0f;
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x)
    {
        const float4 b = w4[i];
#pragma unroll
        for (int r = 0; r < MAXR; ++r)
        {
            if (r >= rows)
                break;
            const float4 a = ((const float4*)(x + (size_t)r * inDim))[i];
            acc[r] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
#pragma unroll
    for (int r = 0; r < MAXR; ++r)
    {
        if (r >= rows)
            break;
        float v = ts_dsv4_warp_sum(acc[r]);
        if (lane == 0)
            sh[warp] = v;
        __syncthreads();
        if (warp == 0)
        {
            v = lane < warps ? sh[lane] : 0.0f;
            v = ts_dsv4_warp_sum(v);
            if (lane == 0)
            {
                if (splits == 1)
                    y[(size_t)r * outDim + o] = v;
                else
                    partials[((size_t)r * outDim + o) * splits + split] = v;
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void ts_dsv4_gemv_1_f32(const float* x, const float* w, float* y, float* p, int inDim, int outDim, int rows, int splits)
{ ts_dsv4_gemv_rows<1>(x, w, y, p, inDim, outDim, rows, splits); }
extern "C" __global__ void ts_dsv4_gemv_4_f32(const float* x, const float* w, float* y, float* p, int inDim, int outDim, int rows, int splits)
{ ts_dsv4_gemv_rows<4>(x, w, y, p, inDim, outDim, rows, splits); }
extern "C" __global__ void ts_dsv4_gemv_8_f32(const float* x, const float* w, float* y, float* p, int inDim, int outDim, int rows, int splits)
{ ts_dsv4_gemv_rows<8>(x, w, y, p, inDim, outDim, rows, splits); }
extern "C" __global__ void ts_dsv4_gemv_16_f32(const float* x, const float* w, float* y, float* p, int inDim, int outDim, int rows, int splits)
{ ts_dsv4_gemv_rows<16>(x, w, y, p, inDim, outDim, rows, splits); }

// The same product for a wide output (the MoE router's 256+ experts): a warp per output, four per
// CTA, lanes striding the whole input row in float4s. The CTA-per-output form gave each of its 128
// threads five float4s of a 2560-wide row and spent the launch on CTA scheduling. Each output's sum
// has one order whatever the row count.
#define TS_DSV4_GEMV_WARP_UNROLL 4
template <int MAXR>
__device__ __forceinline__ void ts_dsv4_gemv_warp_rows(
    const float* __restrict__ x,      // [rows, inDim], inDim % 4 == 0
    const float* __restrict__ w,      // [outDim, inDim]
    float* __restrict__ y,            // [rows, outDim]
    const int inDim,
    const int outDim,
    const int rows)
{
    const int lane = threadIdx.x & 31;
    const int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (o >= outDim)
        return;
    const int n4 = inDim / 4;
    const float4* w4 = (const float4*)(w + (size_t)o * inDim);
    float acc[MAXR];
#pragma unroll
    for (int r = 0; r < MAXR; ++r)
        acc[r] = 0.0f;
    // Several loads in flight per lane (see ts_dsv4_gemv_warp_bf16_rows); the sums keep their order.
    for (int i0 = lane; i0 < n4; i0 += 32 * TS_DSV4_GEMV_WARP_UNROLL)
    {
        float4 bw[TS_DSV4_GEMV_WARP_UNROLL];
#pragma unroll
        for (int u = 0; u < TS_DSV4_GEMV_WARP_UNROLL; ++u)
            bw[u] = i0 + 32 * u < n4 ? w4[i0 + 32 * u] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int u = 0; u < TS_DSV4_GEMV_WARP_UNROLL; ++u)
        {
            const int i = i0 + 32 * u;
            if (i >= n4)
                break;
            const float4 b = bw[u];
#pragma unroll
            for (int r = 0; r < MAXR; ++r)
            {
                if (r >= rows)
                    break;
                const float4 a = ((const float4*)(x + (size_t)r * inDim))[i];
                acc[r] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MAXR; ++r)
    {
        if (r >= rows)
            break;
        const float v = ts_dsv4_warp_sum(acc[r]);
        if (lane == 0)
            y[(size_t)r * outDim + o] = v;
    }
}

// ts_dsv4_gemv_warp_rows with the weights kept as the file's BF16 (a router that ships in BF16): each group of four
// is widened to F32 exactly, and the sums are that kernel's, in its order, so the outputs are the same bits at half
// the bytes read.
template <int MAXR>
__device__ __forceinline__ void ts_dsv4_gemv_warp_bf16_rows(
    const float* __restrict__ x,      // [rows, inDim], inDim % 4 == 0
    const unsigned short* __restrict__ w, // [outDim, inDim] BF16
    float* __restrict__ y,            // [rows, outDim]
    const int inDim,
    const int outDim,
    const int rows)
{
    const int lane = threadIdx.x & 31;
    const int o = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (o >= outDim)
        return;
    const int n4 = inDim / 4;
    const uint2* w4 = (const uint2*)(w + (size_t)o * inDim);
    float acc[MAXR];
#pragma unroll
    for (int r = 0; r < MAXR; ++r)
        acc[r] = 0.0f;
    // TS_DSV4_GEMV_WARP_UNROLL loads of the lane's row in flight before the first is summed: one at a time, a
    // 2560-wide router (512 warps over the whole GPU) waited on each load in turn, 14.2 us a 4-row verify layer
    // for 2.6 MB on the RTX 5090; 12.6 us this way. The sums keep their order.
    for (int i0 = lane; i0 < n4; i0 += 32 * TS_DSV4_GEMV_WARP_UNROLL)
    {
        uint2 pw[TS_DSV4_GEMV_WARP_UNROLL];
#pragma unroll
        for (int u = 0; u < TS_DSV4_GEMV_WARP_UNROLL; ++u)
            pw[u] = i0 + 32 * u < n4 ? w4[i0 + 32 * u] : make_uint2(0u, 0u);
#pragma unroll
        for (int u = 0; u < TS_DSV4_GEMV_WARP_UNROLL; ++u)
        {
            const int i = i0 + 32 * u;
            if (i >= n4)
                break;
            const uint2 p = pw[u];
            float4 b;
            b.x = __uint_as_float(p.x << 16);
            b.y = __uint_as_float(p.x & 0xffff0000u);
            b.z = __uint_as_float(p.y << 16);
            b.w = __uint_as_float(p.y & 0xffff0000u);
#pragma unroll
            for (int r = 0; r < MAXR; ++r)
            {
                if (r >= rows)
                    break;
                const float4 a = ((const float4*)(x + (size_t)r * inDim))[i];
                acc[r] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < MAXR; ++r)
    {
        if (r >= rows)
            break;
        const float v = ts_dsv4_warp_sum(acc[r]);
        if (lane == 0)
            y[(size_t)r * outDim + o] = v;
    }
}

// The wide-output GEMV with a block of TS_DSV4_GEMV_BLOCK_T threads per output (the router's 512 experts): a warp per
// output gave the whole GPU 512 warps, each walking 640 float4s of a 2560-wide row - 12.6 us a 4-row verify layer on the
// RTX 5090 for 2.6 MB, 4.4 us this way. Thread t sums elements t, t + T, ... in order, the warps' sums are added in warp order, so an
// output's sum has one order whatever the row count; the BF16 form widens each weight exactly, so it gives the F32
// form's bits.
#define TS_DSV4_GEMV_BLOCK_T 256
template <int MAXR, bool BF16>
__device__ __forceinline__ void ts_dsv4_gemv_block_rows(
    const float* __restrict__ x, const void* __restrict__ w, float* __restrict__ y, const int inDim, const int outDim,
    const int rows)
{
    const int o = blockIdx.x;
    const int n4 = inDim / 4;
    float acc[MAXR];
#pragma unroll
    for (int r = 0; r < MAXR; ++r)
        acc[r] = 0.0f;
    for (int i = threadIdx.x; i < n4; i += TS_DSV4_GEMV_BLOCK_T)
    {
        float4 b;
        if (BF16)
        {
            const uint2 p = ((const uint2*)((const unsigned short*)w + (size_t)o * inDim))[i];
            b.x = __uint_as_float(p.x << 16);
            b.y = __uint_as_float(p.x & 0xffff0000u);
            b.z = __uint_as_float(p.y << 16);
            b.w = __uint_as_float(p.y & 0xffff0000u);
        }
        else
            b = ((const float4*)((const float*)w + (size_t)o * inDim))[i];
#pragma unroll
        for (int r = 0; r < MAXR; ++r)
        {
            if (r >= rows)
                break;
            const float4 a = ((const float4*)(x + (size_t)r * inDim))[i];
            acc[r] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    }
    __shared__ float part[TS_DSV4_GEMV_BLOCK_T / 32][MAXR];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int r = 0; r < MAXR; ++r)
    {
        const float v = ts_dsv4_warp_sum(acc[r]);
        if (lane == 0)
            part[warp][r] = v;
    }
    __syncthreads();
    if (threadIdx.x < MAXR && threadIdx.x < rows)
    {
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < TS_DSV4_GEMV_BLOCK_T / 32; ++k)
            s += part[k][threadIdx.x];
        y[(size_t)threadIdx.x * outDim + o] = s;
    }
}

#define TS_DSV4_GEMV_BLOCK(R)                                                                                          \
    extern "C" __global__ void __launch_bounds__(TS_DSV4_GEMV_BLOCK_T) ts_dsv4_gemv_block_##R##_f32(                   \
        const float* x, const float* w, float* y, int inDim, int outDim, int rows)                                     \
    { ts_dsv4_gemv_block_rows<R, false>(x, w, y, inDim, outDim, rows); }                                               \
    extern "C" __global__ void __launch_bounds__(TS_DSV4_GEMV_BLOCK_T) ts_dsv4_gemv_block_bf16_##R##_f32(              \
        const float* x, const unsigned short* w, float* y, int inDim, int outDim, int rows)                            \
    { ts_dsv4_gemv_block_rows<R, true>(x, w, y, inDim, outDim, rows); }
TS_DSV4_GEMV_BLOCK(1)
TS_DSV4_GEMV_BLOCK(4)
TS_DSV4_GEMV_BLOCK(8)
TS_DSV4_GEMV_BLOCK(16)
#undef TS_DSV4_GEMV_BLOCK

extern "C" __global__ void ts_dsv4_gemv_warp_bf16_1_f32(const float* x, const unsigned short* w, float* y, int inDim, int outDim, int rows)
{ ts_dsv4_gemv_warp_bf16_rows<1>(x, w, y, inDim, outDim, rows); }
extern "C" __global__ void ts_dsv4_gemv_warp_bf16_4_f32(const float* x, const unsigned short* w, float* y, int inDim, int outDim, int rows)
{ ts_dsv4_gemv_warp_bf16_rows<4>(x, w, y, inDim, outDim, rows); }
extern "C" __global__ void ts_dsv4_gemv_warp_bf16_8_f32(const float* x, const unsigned short* w, float* y, int inDim, int outDim, int rows)
{ ts_dsv4_gemv_warp_bf16_rows<8>(x, w, y, inDim, outDim, rows); }
extern "C" __global__ void ts_dsv4_gemv_warp_bf16_16_f32(const float* x, const unsigned short* w, float* y, int inDim, int outDim, int rows)
{ ts_dsv4_gemv_warp_bf16_rows<16>(x, w, y, inDim, outDim, rows); }

extern "C" __global__ void ts_dsv4_gemv_warp_1_f32(const float* x, const float* w, float* y, int inDim, int outDim, int rows)
{ ts_dsv4_gemv_warp_rows<1>(x, w, y, inDim, outDim, rows); }
extern "C" __global__ void ts_dsv4_gemv_warp_4_f32(const float* x, const float* w, float* y, int inDim, int outDim, int rows)
{ ts_dsv4_gemv_warp_rows<4>(x, w, y, inDim, outDim, rows); }
extern "C" __global__ void ts_dsv4_gemv_warp_8_f32(const float* x, const float* w, float* y, int inDim, int outDim, int rows)
{ ts_dsv4_gemv_warp_rows<8>(x, w, y, inDim, outDim, rows); }
extern "C" __global__ void ts_dsv4_gemv_warp_16_f32(const float* x, const float* w, float* y, int inDim, int outDim, int rows)
{ ts_dsv4_gemv_warp_rows<16>(x, w, y, inDim, outDim, rows); }

// y[i] = sum over s of partials[i, s], in split order.
extern "C" __global__ void ts_dsv4_gemv_combine_f32(
    const float* __restrict__ partials, float* __restrict__ y, const int count, const int splits)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;
    const float* p = partials + (size_t)i * splits;
    float v = 0.0f;
    for (int s = 0; s < splits; ++s)
        v += p[s];
    y[i] = v;
}

// ---------------------------------------------------------------------------
// Hyper-connection pre-block: four residual streams, 24 mixes per token ((2 + 4) * 4: four pre
// gates, four post gates, a 4 x 4 comb matrix). The GLM-5.3 engine runs it as two kernels for
// every ubatch, the DeepSeek V4 engine for decode-size batches (its F32 mixes go through cuBLAS
// for prefill), where the stream RMS, the mix projection, the gates, the collapse and the norm
// were six:
//   ts_dsv4_hc_mix_partials_f32  per (slice of the flattened streams, token): each mix's partial
//                                dot product with hc_fn and the partial sum of squares;
//   ts_dsv4_hc_pre_finish_f32    per token: the mixes and the streams' inverse RMS from the
//                                partials (always summed in slice order), the gates, the streams
//                                collapsed, and the RMS norm of the result.
// Each token's arithmetic is the same whatever the row count, so a batched decode row matches
// the sequence's own step. (The DSpark drafter and the stage-debug dumps keep the separate
// kernels.)
// ---------------------------------------------------------------------------

#define TS_HC_STREAMS 4
#define TS_HC_MIX 24                    // (2 + TS_HC_STREAMS) * TS_HC_STREAMS
#define TS_HC_SLICE 512                 // flattened-stream values per partial CTA
#define TS_HC_PARTIAL (TS_HC_MIX + 1)   // the mixes, then the sum of squares
#define TS_HC_WTYPE_F32 0
#define TS_HC_WTYPE_Q8_0 8

// From one token's 24 mixes (already scaled by its streams' inverse RMS): the pre gates (sigmoid
// plus eps), the post gates (twice a sigmoid) and the Sinkhorn-normalized comb matrix
// [idst + 4 * isrc]: a softmax over each source's row plus eps, a column normalization, then
// iters - 1 rounds of row and column normalization.
__device__ __forceinline__ void ts_hc_gates_comb(
    const float* mixes, const float* __restrict__ scale, const float* __restrict__ baseW,
    float* pre, float* post, float* c, const int iters, const float eps)
{
#pragma unroll
    for (int s = 0; s < 4; ++s)
    {
        pre[s] = ts_dsv4_sigmoid(mixes[s] * scale[0] + baseW[s]) + eps;
        post[s] = 2.0f * ts_dsv4_sigmoid(mixes[4 + s] * scale[1] + baseW[4 + s]);
    }

#pragma unroll
    for (int isrc = 0; isrc < 4; ++isrc)
    {
        float m = -INFINITY;
#pragma unroll
        for (int idst = 0; idst < 4; ++idst)
        {
            const int idx = idst + 4 * isrc;
            const float v = mixes[8 + idx] * scale[2] + baseW[8 + idx];
            c[idx] = v;
            if (v > m) m = v;
        }
        float sum = 0.0f;
#pragma unroll
        for (int idst = 0; idst < 4; ++idst)
        {
            const int idx = idst + 4 * isrc;
            const float v = expf(c[idx] - m);
            c[idx] = v;
            sum += v;
        }
        const float invSum = 1.0f / sum;
#pragma unroll
        for (int idst = 0; idst < 4; ++idst)
            c[idst + 4 * isrc] = c[idst + 4 * isrc] * invSum + eps;
    }

    // Sinkhorn: col-norm, then (iters-1) x (row-norm, col-norm)
    for (int it = 0; it < iters; ++it)
    {
        if (it > 0)
        {
#pragma unroll
            for (int isrc = 0; isrc < 4; ++isrc)
            {
                float sum = eps;
#pragma unroll
                for (int idst = 0; idst < 4; ++idst) sum += c[idst + 4 * isrc];
                const float invSum = 1.0f / sum;
#pragma unroll
                for (int idst = 0; idst < 4; ++idst) c[idst + 4 * isrc] *= invSum;
            }
        }
#pragma unroll
        for (int idst = 0; idst < 4; ++idst)
        {
            float sum = eps;
#pragma unroll
            for (int isrc = 0; isrc < 4; ++isrc) sum += c[idst + 4 * isrc];
            const float invSum = 1.0f / sum;
#pragma unroll
            for (int isrc = 0; isrc < 4; ++isrc) c[idst + 4 * isrc] *= invSum;
        }
    }
}

__device__ __forceinline__ float ts_hc_warp_sum(float v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off, 32);
    return v;
}

// Partials of the mixes. grid (flat / TS_HC_SLICE, rows), 256 threads,
// two consecutive values each; partials[row][slice][0..23] are the mixes' partial dot products
// with hc_fn ([24, flat] rows, F32 or Q8_0, against the F32 streams) and [24] the partial sum of
// squares.
__device__ __forceinline__ void ts_hc_mix_partials(
    const float* __restrict__ xs, const uint8_t* __restrict__ fn, const int wtype,
    float* __restrict__ partials, const int flat)
{
    const int slice = blockIdx.x;
    const int t = blockIdx.y;
    const int i0 = slice * TS_HC_SLICE + threadIdx.x * 2;
    const float* x = xs + (size_t)t * flat;
    const float x0 = x[i0], x1 = x[i0 + 1];

    float acc[TS_HC_PARTIAL];
    if (wtype == TS_HC_WTYPE_Q8_0)
    {
        const size_t rowBytes = (size_t)(flat / 32) * 34;
        const int blk = i0 >> 5, within = i0 & 31;
#pragma unroll
        for (int o = 0; o < TS_HC_MIX; ++o)
        {
            const uint8_t* b = fn + (size_t)o * rowBytes + (size_t)blk * 34;
            const float d = __half2float(*reinterpret_cast<const half*>(b));
            const int8_t* q = reinterpret_cast<const int8_t*>(b + 2);
            acc[o] = d * ((float)q[within] * x0 + (float)q[within + 1] * x1);
        }
    }
    else
    {
        const float* w = reinterpret_cast<const float*>(fn);
#pragma unroll
        for (int o = 0; o < TS_HC_MIX; ++o)
        {
            const float2 wv = *reinterpret_cast<const float2*>(w + (size_t)o * flat + i0);
            acc[o] = wv.x * x0 + wv.y * x1;
        }
    }
    acc[TS_HC_MIX] = x0 * x0 + x1 * x1;

    __shared__ float red[8][TS_HC_PARTIAL];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 0; o < TS_HC_PARTIAL; ++o)
    {
        const float v = ts_hc_warp_sum(acc[o]);
        if (lane == 0)
            red[warp][o] = v;
    }
    __syncthreads();
    if (threadIdx.x < TS_HC_PARTIAL)
    {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w)
            v += red[w][threadIdx.x];
        partials[((size_t)t * gridDim.x + slice) * TS_HC_PARTIAL + threadIdx.x] = v;
    }
}

// The rest of the pre-block for one token (grid (rows), 256 threads): its mixes and inverse RMS
// from the partials, the gates into pre/post/comb, the streams collapsed into cur, and cur RMS
// normed with normW. The collapse weights are the token's own pre gates, or `collapseWith`
// (DeepSeek V4.1 collapses with the previous block's), or the plain stream mean when
// meanCollapse; `publish`, when set, receives the pre gates too.
__device__ __forceinline__ void ts_hc_pre_finish(
    const float* __restrict__ partials, const int nSlices, const float* __restrict__ xs,
    const float* __restrict__ scale, const float* __restrict__ baseW, const float* __restrict__ normW,
    float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb,
    const float* __restrict__ collapseWith, const int meanCollapse, float* __restrict__ publish,
    float* __restrict__ cur, const int e, const int iters, const float hcEps, const float rmsEps)
{
    const int t = blockIdx.x;
    __shared__ float sums[TS_HC_PARTIAL];
    __shared__ float weights[4];
    __shared__ float red[8];

    if (threadIdx.x < TS_HC_PARTIAL)
    {
        const float* p = partials + (size_t)t * nSlices * TS_HC_PARTIAL + threadIdx.x;
        float v = 0.0f;
        for (int s = 0; s < nSlices; ++s)
            v += p[(size_t)s * TS_HC_PARTIAL];
        sums[threadIdx.x] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        const float inv = rsqrtf(sums[TS_HC_MIX] / (float)(4 * e) + rmsEps);
        float mixes[TS_HC_MIX];
#pragma unroll
        for (int i = 0; i < TS_HC_MIX; ++i)
            mixes[i] = sums[i] * inv;
        float pr[4], po[4], c[16];
        ts_hc_gates_comb(mixes, scale, baseW, pr, po, c, iters, hcEps);
#pragma unroll
        for (int s = 0; s < 4; ++s)
        {
            pre[(size_t)t * 4 + s] = pr[s];
            post[(size_t)t * 4 + s] = po[s];
            if (publish != nullptr)
                publish[(size_t)t * 4 + s] = pr[s];
            weights[s] = meanCollapse ? 0.25f : collapseWith != nullptr ? collapseWith[(size_t)t * 4 + s] : pr[s];
        }
#pragma unroll
        for (int i = 0; i < 16; ++i)
            comb[(size_t)t * 16 + i] = c[i];
    }
    __syncthreads();

    const float* x = xs + (size_t)t * 4 * e;
    float* y = cur + (size_t)t * e;
    const float w0 = weights[0], w1 = weights[1], w2 = weights[2], w3 = weights[3];
    float ss = 0.0f;
    for (int d = threadIdx.x; d < e; d += blockDim.x)
    {
        float v = 0.0f;
        v += x[d] * w0;
        v += x[(size_t)e + d] * w1;
        v += x[(size_t)2 * e + d] * w2;
        v += x[(size_t)3 * e + d] * w3;
        y[d] = v;
        ss += v * v;
    }
    ss = ts_hc_warp_sum(ss);
    if ((threadIdx.x & 31) == 0)
        red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float v = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : 0.0f;
        v = ts_hc_warp_sum(v);
        if (threadIdx.x == 0)
            red[0] = rsqrtf(v / (float)e + rmsEps);
    }
    __syncthreads();
    const float r = red[0];
    for (int d = threadIdx.x; d < e; d += blockDim.x)
        y[d] = y[d] * r * normW[d];
}

// pre/post sigmoid gates + Sinkhorn-normalized 4x4 comb matrix, one thread per
// token (the whole computation is ~500 flops on 24 inputs).
extern "C" __global__ void ts_dsv4_hc_gates_comb_f32(
    const float* __restrict__ mixesRaw,  // [nt, 24] (un-scaled hc_fn output)
    const float* __restrict__ inv,       // [nt]
    const float* __restrict__ scale,     // [3]
    const float* __restrict__ baseW,     // [24]
    float* __restrict__ pre,             // [nt, 4]
    float* __restrict__ post,            // [nt, 4]
    float* __restrict__ comb,            // [nt, 16] (idst + 4*isrc)
    const int nt,
    const int iters,
    const float eps)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt)
        return;

    const float invT = inv[t];
    float mixes[TS_HC_MIX];
#pragma unroll
    for (int i = 0; i < TS_HC_MIX; ++i)
        mixes[i] = mixesRaw[(size_t)t * TS_HC_MIX + i] * invT;

    float pr[4], po[4], c[16];
    ts_hc_gates_comb(mixes, scale, baseW, pr, po, c, iters, eps);
#pragma unroll
    for (int s = 0; s < 4; ++s)
    {
        pre[(size_t)t * 4 + s] = pr[s];
        post[(size_t)t * 4 + s] = po[s];
    }
#pragma unroll
    for (int i = 0; i < 16; ++i)
        comb[(size_t)t * 16 + i] = c[i];
}

extern "C" __global__ void __launch_bounds__(256) ts_dsv4_hc_mix_partials_f32(
    const float* __restrict__ xs, const uint8_t* __restrict__ fn, const int wtype,
    float* __restrict__ partials, const int flat)
{
    ts_hc_mix_partials(xs, fn, wtype, partials, flat);
}

extern "C" __global__ void __launch_bounds__(256) ts_dsv4_hc_pre_finish_f32(
    const float* __restrict__ partials, const int nSlices, const float* __restrict__ xs,
    const float* __restrict__ scale, const float* __restrict__ baseW, const float* __restrict__ normW,
    float* __restrict__ pre, float* __restrict__ post, float* __restrict__ comb,
    const float* __restrict__ collapseWith, const int meanCollapse, float* __restrict__ publish,
    float* __restrict__ cur, const int e, const int iters, const float hcEps, const float rmsEps)
{
    ts_hc_pre_finish(partials, nSlices, xs, scale, baseW, normW, pre, post, comb,
        collapseWith, meanCollapse, publish, cur, e, iters, hcEps, rmsEps);
}

// cur[t,e] = sum_s xs[t,s,e] * pre[t,s]
extern "C" __global__ void ts_dsv4_hc_collapse_f32(
    const float* __restrict__ xs,
    const float* __restrict__ pre,
    float* __restrict__ cur,
    const int nt,
    const int E)
{
    const int t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt || e >= E)
        return;

    const float* x = xs + (size_t)t * 4 * E + e;
    const float* p = pre + (size_t)t * 4;
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < 4; ++s)
        acc += x[(size_t)s * E] * p[s];
    cur[(size_t)t * E + e] = acc;
}

// xsOut[t,idst,e] = o[t,e]*post[t,idst] + sum_isrc xs[t,isrc,e]*comb[t, idst+4*isrc]
extern "C" __global__ void ts_dsv4_hc_post_f32(
    const float* __restrict__ xs,
    const float* __restrict__ blockOut,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    float* __restrict__ xsOut,
    const int nt,
    const int E)
{
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt || i >= 4 * E)
        return;

    const int idst = i / E;
    const int e = i % E;

    const float* x = xs + (size_t)t * 4 * E + e;
    const float* cb = comb + (size_t)t * 16;
    float acc = blockOut[(size_t)t * E + e] * post[(size_t)t * 4 + idst];
#pragma unroll
    for (int isrc = 0; isrc < 4; ++isrc)
        acc += x[(size_t)isrc * E] * cb[idst + 4 * isrc];
    xsOut[(size_t)t * 4 * E + (size_t)idst * E + e] = acc;
}

// hc_head for the final token: collapse streams with sigmoid gates from
// output_hc_fn (F32 [4, flatDim]), written into cur [E].
// One block per row (grid.x == 1 for the target's single-token head, == the
// block size for the DSpark drafter's whole block).
extern "C" __global__ void ts_dsv4_hc_head_f32(
    const float* __restrict__ xAll,   // [rows, 4*E] (the selected tokens' streams)
    const float* __restrict__ fn,     // [4, 4*E] F32
    const float* __restrict__ scale,  // [>=1]
    const float* __restrict__ baseW,  // [>=1]
    float* __restrict__ curAll,       // [rows, E]
    const int E,
    const int scaleCount,
    const int baseCount,
    const float eps)
{
    const int flatDim = 4 * E;
    const float* __restrict__ x = xAll + (size_t)blockIdx.x * flatDim;
    float* __restrict__ cur = curAll + (size_t)blockIdx.x * E;
    __shared__ float red[256];
    __shared__ float gates[4];

    float acc = 0.0f;
    for (int i = threadIdx.x; i < flatDim; i += blockDim.x)
    {
        const float v = x[i];
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / flatDim + eps);
    __syncthreads();

    for (int s = 0; s < 4; ++s)
    {
        const float* fr = fn + (size_t)s * flatDim;
        float dot = 0.0f;
        for (int i = threadIdx.x; i < flatDim; i += blockDim.x)
            dot += fr[i] * x[i];
        red[threadIdx.x] = dot;
        __syncthreads();
        for (int r = blockDim.x / 2; r > 0; r >>= 1)
        {
            if (threadIdx.x < r)
                red[threadIdx.x] += red[threadIdx.x + r];
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            const float sc = scaleCount >= 4 ? scale[s] : scale[0];
            const float b = baseCount >= 4 ? baseW[s] : baseW[0];
            gates[s] = ts_dsv4_sigmoid(red[0] * inv * sc + b) + eps;
        }
        __syncthreads();
    }

    for (int e = threadIdx.x; e < E; e += blockDim.x)
    {
        float v = 0.0f;
#pragma unroll
        for (int s = 0; s < 4; ++s)
            v += x[(size_t)s * E + e] * gates[s];
        cur[e] = v;
    }
}

// ---------------------------------------------------------------------------
// norms / rope / attention prologue
// ---------------------------------------------------------------------------

// RMS norm now comes from the shared kernel set (ts_rmsnorm_f32, reached via
// Ops.RMSNorm): identical math, one implementation.

// blockIdx.x in [0, NH]: heads 0..NH-1 per-head-RMS + RoPE q in place; block
// NH RMS-norms (with weight) + RoPEs kv and commits it (F16) to the SWA ring.
extern "C" __global__ void ts_dsv4_attn_prep_f32(
    float* __restrict__ q,               // [nt, NH, HD] in-place
    const float* __restrict__ kvRaw,     // [nt, HD]
    const float* __restrict__ kvNormW,   // [HD]
    const float* __restrict__ ropeTab,   // [nCtx, nRot]
    half* __restrict__ ring,             // [ringRows, HD]
    const int p0,
    const int ringRows,
    const int NH,
    const int HD,
    const int nRot,
    const float eps)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const bool isKv = h == NH;

    __shared__ float sh[512];
    __shared__ float red[256];

    float* qDst = q + ((size_t)t * NH + h) * HD;
    const float* src = isKv ? kvRaw + (size_t)t * HD : qDst;

    float acc = 0.0f;
    for (int d = threadIdx.x; d < HD; d += blockDim.x)
    {
        const float v = src[d];
        sh[d] = v;
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / HD + eps);

    for (int d = threadIdx.x; d < HD; d += blockDim.x)
        sh[d] *= isKv ? inv * kvNormW[d] : inv;
    __syncthreads();

    const long long p = (long long)p0 + t;
    const float* tab = ropeTab + p * nRot;
    const int rbase = HD - nRot;

    if (isKv)
    {
        half* out = ring + (size_t)(p % ringRows) * HD;
        for (int d = threadIdx.x; d < HD; d += blockDim.x)
        {
            float v = sh[d];
            if (d >= rbase)
            {
                const int i = (d - rbase) >> 1;
                const float c = tab[2 * i + 0];
                const float s = tab[2 * i + 1];
                const float x0 = sh[rbase + 2 * i + 0];
                const float x1 = sh[rbase + 2 * i + 1];
                v = ((d - rbase) & 1) == 0 ? x0 * c - x1 * s : x0 * s + x1 * c;
            }
            out[d] = __float2half(v);
        }
    }
    else
    {
        for (int d = threadIdx.x; d < HD; d += blockDim.x)
        {
            float v = sh[d];
            if (d >= rbase)
            {
                const int i = (d - rbase) >> 1;
                const float c = tab[2 * i + 0];
                const float s = tab[2 * i + 1];
                const float x0 = sh[rbase + 2 * i + 0];
                const float x1 = sh[rbase + 2 * i + 1];
                v = ((d - rbase) & 1) == 0 ? x0 * c - x1 * s : x0 * s + x1 * c;
            }
            qDst[d] = v;
        }
    }
}

// stScore[t] += ape[(p0+t) % ratio]
extern "C" __global__ void ts_dsv4_ape_add_f32(
    float* __restrict__ stScore,
    const float* __restrict__ ape,
    const int p0,
    const int ratio,
    const int nt,
    const int cw)
{
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt || i >= cw)
        return;
    const int r = (int)(((long long)p0 + t) % ratio);
    stScore[(size_t)t * cw + i] += ape[(size_t)r * cw + i];
}

// ---------------------------------------------------------------------------
// block compressor (CSA coff=2 / HCA coff=1 / LID)
// ---------------------------------------------------------------------------

// One block per boundary in the ubatch. Sources are resolved by index math:
// window token tw reads the scratch projections when tw >= p0 and the state
// ring otherwise; tw < 0 contributes nothing (kv 0 / score -inf).
extern "C" __global__ void ts_dsv4_compress_f32(
    const float* __restrict__ stKv,      // [nt, cw]
    const float* __restrict__ stScore,   // [nt, cw]
    const float* __restrict__ histKv,    // [stateSize, cw]
    const float* __restrict__ histScore, // [stateSize, cw]
    const float* __restrict__ normW,     // [head]
    const float* __restrict__ ropeTab,   // [nCtx, nRot]
    half* __restrict__ cache,            // [rows, head]
    const long long firstBoundary,       // absolute position of first boundary
    const int nBlocks,
    const int p0,
    const int ratio,
    const int coff,
    const int head,
    const int stateSize,
    const int nRot,
    const float eps)
{
    const int b = blockIdx.x;
    if (b >= nBlocks)
        return;

    const int cw = coff * head;
    const int W = coff * ratio;
    const long long p = firstBoundary + (long long)b * ratio;
    const long long cacheRow = p / ratio;
    const long long blockStart = p + 1 - ratio;

    __shared__ float shOut[512];
    __shared__ float red[256];

    float acc2 = 0.0f;
    for (int d = threadIdx.x; d < head; d += blockDim.x)
    {
        float m = -INFINITY;
        float se = 0.0f;
        float sv = 0.0f;
        for (int w = 0; w < W; ++w)
        {
            const long long tw = p + 1 - W + w;
            if (tw < 0)
                continue;
            const int half_ = (coff == 2 && w >= ratio) ? head : 0;
            const float* kvSrc;
            const float* scSrc;
            if (tw >= p0)
            {
                const size_t off = (size_t)(tw - p0) * cw + half_ + d;
                kvSrc = stKv + off;
                scSrc = stScore + off;
            }
            else
            {
                const size_t off = (size_t)(tw % stateSize) * cw + half_ + d;
                kvSrc = histKv + off;
                scSrc = histScore + off;
            }
            const float sc = *scSrc;
            if (sc > m)
            {
                const float r = expf(m - sc); // 0 when m was -inf
                se *= r;
                sv *= r;
                m = sc;
            }
            const float e = sc == -INFINITY ? 0.0f : expf(sc - m);
            se += e;
            sv += e * (*kvSrc);
        }
        const float out = se > 0.0f ? sv / se : 0.0f;
        shOut[d] = out;
        acc2 += out * out;
    }

    red[threadIdx.x] = acc2;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float scale = rsqrtf(red[0] / head + eps);

    for (int d = threadIdx.x; d < head; d += blockDim.x)
        shOut[d] *= scale * normW[d];
    __syncthreads();

    const float* tab = ropeTab + blockStart * nRot;
    const int rbase = head - nRot;
    for (int d = threadIdx.x; d < head; d += blockDim.x)
    {
        float v = shOut[d];
        if (d >= rbase)
        {
            const int i = (d - rbase) >> 1;
            const float c = tab[2 * i + 0];
            const float s = tab[2 * i + 1];
            const float x0 = shOut[rbase + 2 * i + 0];
            const float x1 = shOut[rbase + 2 * i + 1];
            v = ((d - rbase) & 1) == 0 ? x0 * c - x1 * s : x0 * s + x1 * c;
        }
        cache[cacheRow * head + d] = __float2half(v);
    }
}

// Persist the tail of the ubatch's projections into the state ring.
extern "C" __global__ void ts_dsv4_persist_f32(
    const float* __restrict__ stKv,
    const float* __restrict__ stScore,
    float* __restrict__ histKv,
    float* __restrict__ histScore,
    const int p0,
    const int nt,
    const int stateSize,
    const int cw)
{
    const int np = nt < stateSize ? nt : stateSize;
    const long long total = (long long)np * cw;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    const int j = (int)(i / cw);
    const int d = (int)(i % cw);
    const int t = nt - np + j;
    const long long slot = ((long long)p0 + t) % stateSize;
    histKv[slot * cw + d] = stKv[(size_t)t * cw + d];
    histScore[slot * cw + d] = stScore[(size_t)t * cw + d];
}

// ---------------------------------------------------------------------------
// lightning indexer
// ---------------------------------------------------------------------------

// RoPE the tail dims of each indexer q head; scale the per-head weights.
extern "C" __global__ void ts_dsv4_idx_prep_f32(
    float* __restrict__ iq,            // [nt, IH, ID] in-place
    float* __restrict__ iw,            // [nt, IH]
    const float* __restrict__ ropeTab, // comp table
    const int p0,
    const int IH,
    const int ID,
    const int nRot,
    const float iwScale)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;

    __shared__ float sh[256];

    float* row = iq + ((size_t)t * IH + h) * ID;
    for (int d = threadIdx.x; d < ID; d += blockDim.x)
        sh[d] = row[d];
    __syncthreads();

    const long long p = (long long)p0 + t;
    const float* tab = ropeTab + p * nRot;
    const int rbase = ID - nRot;
    for (int d = threadIdx.x; d < ID; d += blockDim.x)
    {
        float v = sh[d];
        if (d >= rbase)
        {
            const int i = (d - rbase) >> 1;
            const float c = tab[2 * i + 0];
            const float s = tab[2 * i + 1];
            const float x0 = sh[rbase + 2 * i + 0];
            const float x1 = sh[rbase + 2 * i + 1];
            v = ((d - rbase) & 1) == 0 ? x0 * c - x1 * s : x0 * s + x1 * c;
        }
        row[d] = v;
    }

    if (h == 0 && threadIdx.x < IH)
        iw[(size_t)t * IH + threadIdx.x] *= iwScale;
}

// scores[t, r] = sum_h relu(q_th . k_r) * w_th for r < nVis(t).
// One warp per (row, token); q rows staged in shared per block.
#define TS_DSV4_IDX_ROWS_PER_BLOCK 8

extern "C" __global__ void ts_dsv4_idx_scores_f32(
    const float* __restrict__ iq,     // [nt, IH, ID]
    const float* __restrict__ iw,     // [nt, IH]
    const half* __restrict__ lidK,    // [rows, ID]
    float* __restrict__ scores,       // [nt, scoreStride]
    const int p0,
    const int ratio,
    const int IH,
    const int ID,
    const int nt,
    const int scoreStride)
{
    const int t = blockIdx.y;
    const int nVis = (int)(((long long)p0 + t + 1) / ratio);
    const int r0 = blockIdx.x * TS_DSV4_IDX_ROWS_PER_BLOCK;
    if (r0 >= nVis)
        return;

    // stage this token's q heads and weights (IH*ID <= 64*128 floats = 32KB)
    extern __shared__ float shQ[]; // [IH*ID + IH]
    float* shW = shQ + IH * ID;
    for (int i = threadIdx.x; i < IH * ID; i += blockDim.x)
        shQ[i] = iq[(size_t)t * IH * ID + i];
    for (int i = threadIdx.x; i < IH; i += blockDim.x)
        shW[i] = iw[(size_t)t * IH + i];
    __syncthreads();

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int r = r0 + warpId;
    if (r >= nVis)
        return;

    // each lane holds 4 consecutive k dims (ID = 128)
    const half* kRow = lidK + (size_t)r * ID;
    float k0 = __half2float(kRow[lane * 4 + 0]);
    float k1 = __half2float(kRow[lane * 4 + 1]);
    float k2 = __half2float(kRow[lane * 4 + 2]);
    float k3 = __half2float(kRow[lane * 4 + 3]);

    float score = 0.0f;
    for (int h = 0; h < IH; ++h)
    {
        const float* qh = shQ + (size_t)h * ID + lane * 4;
        float part = qh[0] * k0 + qh[1] * k1 + qh[2] * k2 + qh[3] * k3;
        part = ts_dsv4_warp_sum(part);
        if (lane == 0 && part > 0.0f)
            score += part * shW[h];
    }
    if (lane == 0)
        scores[(size_t)t * scoreStride + r] = score;
}

// Per-token top-K selection over scores[0, nVis) via 4-pass MSB radix select
// on flipped float keys. Emits an (unordered) index list; attention treats
// the selection as a set, so order does not matter. Row t sits at p0 + t, or at
// positions[t] (a captured decode step reads its position from the device).
extern "C" __global__ void ts_dsv4_topk_f32(
    const float* __restrict__ scores, // [nt, scoreStride]
    int32_t* __restrict__ topkIdx,    // [nt, K]
    int32_t* __restrict__ topkCnt,    // [nt]
    const int p0,
    const int ratio,
    const int K,
    const int scoreStride,
    const int32_t* __restrict__ positions)
{
    const int t = blockIdx.x;
    const long long pos = positions != nullptr ? (long long)positions[t] : (long long)p0 + t;
    const int nVis = (int)((pos + 1) / ratio);
    const float* sc = scores + (size_t)t * scoreStride;
    int32_t* outIdx = topkIdx + (size_t)t * K;

    const int k = nVis < K ? nVis : K;
    if (threadIdx.x == 0)
        topkCnt[t] = k;
    if (k <= 0)
        return;

    if (nVis <= K)
    {
        for (int r = threadIdx.x; r < nVis; r += blockDim.x)
            outIdx[r] = r;
        return;
    }

    __shared__ unsigned int hist[256];
    __shared__ unsigned int shPrefix;
    __shared__ int shNeed;
    if (threadIdx.x == 0)
    {
        shPrefix = 0;
        shNeed = k;
    }
    __syncthreads();

    // find the k-th largest key byte by byte from the MSB
    for (int pass = 0; pass < 4; ++pass)
    {
        const int shift = 24 - 8 * pass;
        if (threadIdx.x < 256)
            hist[threadIdx.x] = 0;
        __syncthreads();

        const unsigned int prefix = shPrefix;
        for (int r = threadIdx.x; r < nVis; r += blockDim.x)
        {
            const float v = sc[r];
            unsigned int key = __float_as_uint(v);
            key ^= (key & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
            if (pass > 0 && (key >> (shift + 8)) != prefix)
                continue;
            atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
        }
        __syncthreads();

        if (threadIdx.x == 0)
        {
            int need = shNeed;
            unsigned int bin = 255;
            for (int b2 = 255; b2 >= 0; --b2)
            {
                const int c = (int)hist[b2];
                if (c >= need)
                {
                    bin = (unsigned int)b2;
                    break;
                }
                need -= c;
            }
            shNeed = need;
            shPrefix = (shPrefix << 8) | bin;
        }
        __syncthreads();
    }

    // shPrefix is now the k-th largest key; emit keys > threshold, then pad
    // with ties (== threshold) up to k, both in ascending index order. A
    // block-wide scan over each blockDim-wide window assigns the slots, so the
    // selection is deterministic: the lowest indices win a tie, and the order
    // the attention sums the rows in never depends on atomic arrival.
    const unsigned int threshold = shPrefix;
    __shared__ int warpCounts[32];
    __shared__ int shBase;
    if (threadIdx.x == 0)
        shBase = 0;
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nWarps = (blockDim.x + 31) >> 5;
    for (int pass = 0; pass < 2; ++pass)
    {
        for (int r0 = 0; r0 < nVis; r0 += blockDim.x)
        {
            const int r = r0 + threadIdx.x;
            bool take = false;
            if (r < nVis)
            {
                unsigned int key = __float_as_uint(sc[r]);
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
            if (take && slot < k)
                outIdx[slot] = r;
            __syncthreads();
            if (threadIdx.x == 0)
            {
                int total = 0;
                for (int w = 0; w < nWarps; ++w)
                    total += warpCounts[w];
                shBase += total;
            }
            __syncthreads();
        }
    }
}

// ---------------------------------------------------------------------------
// attention core
// ---------------------------------------------------------------------------

// Online-softmax attention with sinks over [raw SWA ring rows | compressed
// rows]. K doubles as V. mode: 0 = raw rows only, 1 = compressed rows from
// the top-k list, 2 = all visible compressed rows ((p+1)/ratio).
// Row t sits at p0 + t, or at positions[t] (a captured decode step).
// grid (NH, nt), 256 threads = 8 warps; warps split the rows, each lane owns
// HD/32 = 16 dims of its warp's accumulator, partials merged through shared.
extern "C" __global__ void ts_dsv4_attention_f32(
    const float* __restrict__ q,        // [nt, NH, HD]
    const half* __restrict__ ring,      // [ringRows, HD]
    const half* __restrict__ comp,      // [rows, HD] (null for mode 0)
    const int32_t* __restrict__ topkIdx,// [nt, K] (mode 1)
    const int32_t* __restrict__ topkCnt,// [nt] (mode 1)
    const float* __restrict__ sinks,    // [NH]
    float* __restrict__ attnO,          // [nt, NH, HD]
    const int p0,
    const int nSwa,
    const int ringRows,
    const int NH,
    const int HD,
    const int mode,
    const int ratio,
    const int K,
    const float kqScale,
    const int32_t* __restrict__ positions)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const long long p = positions != nullptr ? (long long)positions[t] : (long long)p0 + t;

    // mode 3 (DSpark drafter): every block row sees the SAME committed history
    // -- the window ending at p0, the drafter's last committed position -- plus
    // all K rows of the block itself (passed in `comp`), non-causally.
    // mode 4 (GLM-5.3's pooled sparse attention): no window, only the listed cells of `comp`.
    const long long pHist = mode == 3 ? (long long)p0 : p;
    const long long rawStart = pHist - nSwa + 1 > 0 ? pHist - nSwa + 1 : 0;
    const int rawCnt = mode == 4 ? 0 : (int)(pHist - rawStart + 1);
    int compCnt = 0;
    const int32_t* sel = nullptr;
    if (mode == 1 || mode == 4)
    {
        compCnt = topkCnt[t];
        sel = topkIdx + (size_t)t * K;
    }
    else if (mode == 2)
    {
        compCnt = (int)((p + 1) / ratio);
    }
    else if (mode == 3)
    {
        compCnt = K;
    }
    const int n = rawCnt + compCnt;

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int dimsPerLane = HD / 32; // 16

    const float* qRow = q + ((size_t)t * NH + h) * HD;
    float qReg[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
        qReg[i] = qRow[lane * dimsPerLane + i];

    float m = -INFINITY;
    float sum = 0.0f;
    float acc[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
        acc[i] = 0.0f;

    for (int i = warpId; i < n; i += 8)
    {
        const half* kRow;
        if (i < rawCnt)
        {
            kRow = ring + (size_t)((rawStart + i) % ringRows) * HD;
        }
        else
        {
            const int cr = (mode == 1 || mode == 4) ? sel[i - rawCnt] : (i - rawCnt);
            kRow = comp + (size_t)cr * HD;
        }

        // 16 halves per lane = two 16-byte vector loads. Reading them as 16
        // scalar __half loads issues 8x the load instructions for the same
        // bytes, and K traffic is what this kernel is bound by.
        const half* kLane = kRow + lane * dimsPerLane;
        float kReg[16];
        {
            const int4 v0 = *(const int4*)kLane;
            const int4 v1 = *(const int4*)(kLane + 8);
            const half2* h0 = (const half2*)&v0;
            const half2* h1 = (const half2*)&v1;
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const float2 f0 = __half22float2(h0[j]);
                const float2 f1 = __half22float2(h1[j]);
                kReg[2*j + 0] = f0.x;
                kReg[2*j + 1] = f0.y;
                kReg[8 + 2*j + 0] = f1.x;
                kReg[8 + 2*j + 1] = f1.y;
            }
        }

        float part = 0.0f;
#pragma unroll
        for (int d2 = 0; d2 < 16; ++d2)
            part += qReg[d2] * kReg[d2];
        const float s = ts_dsv4_warp_sum(part) * kqScale;

        if (s > m)
        {
            const float r = expf(m - s); // 0 when m was -inf
            sum *= r;
#pragma unroll
            for (int d2 = 0; d2 < 16; ++d2)
                acc[d2] *= r;
            m = s;
        }
        const float w = expf(s - m);
        sum += w;
#pragma unroll
        for (int d2 = 0; d2 < 16; ++d2)
            acc[d2] += w * kReg[d2];
    }

    // merge the 8 warp partials
    __shared__ float shAcc[8][512];
    __shared__ float shM[8];
    __shared__ float shSum[8];

#pragma unroll
    for (int d2 = 0; d2 < 16; ++d2)
        shAcc[warpId][lane * dimsPerLane + d2] = acc[d2];
    if (lane == 0)
    {
        shM[warpId] = m;
        shSum[warpId] = sum;
    }
    __syncthreads();

    __shared__ float shTotM;
    __shared__ float shTotSum;
    if (threadIdx.x == 0)
    {
        float M = sinks[h];
        for (int w = 0; w < 8; ++w)
            if (shM[w] > M) M = shM[w];
        float S = expf(sinks[h] - M);
        for (int w = 0; w < 8; ++w)
            S += shSum[w] * expf(shM[w] - M);
        shTotM = M;
        shTotSum = S;
    }
    __syncthreads();

    const float M = shTotM;
    const float invSum = 1.0f / shTotSum;
    float* out = attnO + ((size_t)t * NH + h) * HD;
    for (int d = threadIdx.x; d < HD; d += blockDim.x)
    {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < 8; ++w)
            v += shAcc[w][d] * expf(shM[w] - M);
        out[d] = v * invSum;
    }
}

// Inverse RoPE on the tail dims of each attention output head, stored in the
// grouped layout [G, nt, headsPerGroup*HD] for the grouped LoRA out-proj.
// Row t sits at position p0 + t, or at pos[t] when a batched decode step
// carries one row per sequence.
extern "C" __global__ void ts_dsv4_attn_finish_f32(
    const float* __restrict__ attnO,   // [nt, NH, HD]
    const float* __restrict__ ropeTab,
    float* __restrict__ dst,           // [G, nt, headsPerGroup*HD]
    const int p0,
    const int NH,
    const int HD,
    const int nRot,
    const int headsPerGroup,
    const int nt,
    const int* __restrict__ pos)       // [nt] or nullptr
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;

    __shared__ float sh[512];
    const float* src = attnO + ((size_t)t * NH + h) * HD;
    for (int d = threadIdx.x; d < HD; d += blockDim.x)
        sh[d] = src[d];
    __syncthreads();

    const long long p = pos ? (long long)pos[t] : (long long)p0 + t;
    const float* tab = ropeTab + p * nRot;
    const int rbase = HD - nRot;
    const int g = h / headsPerGroup;
    const int hg = h % headsPerGroup;
    const size_t groupDim = (size_t)headsPerGroup * HD;
    float* out = dst + (size_t)g * nt * groupDim + (size_t)t * groupDim + (size_t)hg * HD;

    for (int d = threadIdx.x; d < HD; d += blockDim.x)
    {
        float v = sh[d];
        if (d >= rbase)
        {
            const int i = (d - rbase) >> 1;
            const float c = tab[2 * i + 0];
            const float s = tab[2 * i + 1];
            const float x0 = sh[rbase + 2 * i + 0];
            const float x1 = sh[rbase + 2 * i + 1];
            // transpose (inverse) of the forward rotation
            v = ((d - rbase) & 1) == 0 ? x0 * c + x1 * s : -x0 * s + x1 * c;
        }
        out[d] = v;
    }
}

// oG[t, g*R + r] = oGtmp[g, t, r] (regroup the per-group LoRA-A outputs into
// the concatenated wo_b input layout).
extern "C" __global__ void ts_dsv4_regroup_f32(
    const float* __restrict__ oGtmp, // [G, nt, R]
    float* __restrict__ oG,          // [nt, G*R]
    const int G,
    const int nt,
    const int R)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)G * nt * R;
    if (i >= total)
        return;
    const int g = (int)(i / ((long long)nt * R));
    const int rem = (int)(i % ((long long)nt * R));
    const int t = rem / R;
    const int r = rem % R;
    oG[(size_t)t * G * R + (size_t)g * R + r] = oGtmp[i];
}

// ---------------------------------------------------------------------------
// MoE
// ---------------------------------------------------------------------------

// Selection + weights. Hash layers route through the tid2eid LUT; the rest
// take the top-k of sqrt(softplus(logits)) + bias. Weights are the unbiased
// probs of the selected experts, optionally sum-normalized, then scaled.
// Router scores: DeepSeek V4's sqrt(softplus) (gating 0) or GLM's sigmoid (gating 1). The
// bias steers the selection only; the weights come from the unbiased scores. Gating 2, Qwen's
// softmax over every expert, needs the whole row and is computed in the select kernel's warp path.
__device__ __forceinline__ float ts_dsv4_router_score(float logit, int gating)
{
    return gating == 1 ? ts_dsv4_sigmoid(logit) : sqrtf(ts_dsv4_softplus(logit));
}

#define TS_DSV4_ROUTER_SOFTMAX 2

#define TS_DSV4_SELECT_PER_LANE 16   // register-held router scores per lane (512 experts)
#define TS_DSV4_SELECT_MAX_USED 16   // experts a token may select

extern "C" __global__ void ts_dsv4_moe_select_f32(
    const float* __restrict__ logits,   // [nt, nExpert]
    const float* __restrict__ bias,     // [nExpert] or null
    const int32_t* __restrict__ tid2eid,// [nVocab, nUsed] or null
    const int32_t* __restrict__ tokens, // [nt] (hash layers)
    int32_t* __restrict__ sel,          // [nt, nUsed]
    float* __restrict__ wOut,           // [nt, nUsed]
    const int nExpert,
    const int nUsed,
    const int norm,
    const float wScale,
    const int gating)
{
    const int t = blockIdx.x;
    const float* lg = logits + (size_t)t * nExpert;
    int32_t* selT = sel + (size_t)t * nUsed;
    float* wT = wOut + (size_t)t * nUsed;

    if (tid2eid != nullptr)
    {
        if (threadIdx.x < (unsigned)nUsed)
            selT[threadIdx.x] = tid2eid[(size_t)tokens[t] * nUsed + threadIdx.x];
    }
    else if (nExpert <= 32 * TS_DSV4_SELECT_PER_LANE)
    {
        // Up to 512 experts: warp 0 keeps the scores in registers and takes each pick as one
        // shuffle argmax (the block-wide reduction below spent eight barriered steps per pick).
        // Same order as that reduction: the higher score, then the lower valid index.
        // The weights come from the chosen experts' own router scores (without the bias), carried
        // through the argmax: reading them back from sel and the logits in global memory made the
        // weight step a chain of dependent round trips.
        if (threadIdx.x < 32)
        {
            const int lane = threadIdx.x;
            float v[TS_DSV4_SELECT_PER_LANE], raw[TS_DSV4_SELECT_PER_LANE];
            if (gating == TS_DSV4_ROUTER_SOFTMAX)
            {
                // ggml's soft_max: exp(l - max) times the reciprocal of their sum.
                float mx = -INFINITY;
#pragma unroll
                for (int i = 0; i < TS_DSV4_SELECT_PER_LANE; ++i)
                {
                    const int e = lane + 32 * i;
                    raw[i] = e < nExpert ? lg[e] : -INFINITY;
                    mx = fmaxf(mx, raw[i]);
                }
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
                float sum = 0.0f;
#pragma unroll
                for (int i = 0; i < TS_DSV4_SELECT_PER_LANE; ++i)
                {
                    raw[i] = lane + 32 * i < nExpert ? expf(raw[i] - mx) : 0.0f;
                    sum += raw[i];
                }
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    sum += __shfl_xor_sync(0xffffffffu, sum, off);
                const float inv = 1.0f / sum;
#pragma unroll
                for (int i = 0; i < TS_DSV4_SELECT_PER_LANE; ++i)
                {
                    const int e = lane + 32 * i;
                    raw[i] *= inv;
                    v[i] = e < nExpert ? raw[i] + (bias != nullptr ? bias[e] : 0.0f) : -INFINITY;
                }
            }
            else
            {
#pragma unroll
                for (int i = 0; i < TS_DSV4_SELECT_PER_LANE; ++i)
                {
                    const int e = lane + 32 * i;
                    raw[i] = e < nExpert ? ts_dsv4_router_score(lg[e], gating) : 0.0f;
                    v[i] = e < nExpert ? raw[i] + (bias != nullptr ? bias[e] : 0.0f) : -INFINITY;
                }
            }
            // Each lane keeps the best of its own scores (the higher score, then the lower index), so
            // a pick is one shuffle argmax over the lanes' bests and only the winning lane rescans.
            // Every register array is indexed by unrolled constants: the pick loop itself runs at
            // run time (unrolling it sixteen deep put the scores in local memory).
            float lbV = -INFINITY, lbRaw = 0.0f;
            int lbI = -1;
#pragma unroll
            for (int i = 0; i < TS_DSV4_SELECT_PER_LANE; ++i)
            {
                const int e = lane + 32 * i;
                if (e < nExpert && (v[i] > lbV || (v[i] == lbV && lbI >= 0 && e < lbI)))
                {
                    lbV = v[i];
                    lbRaw = raw[i];
                    lbI = e;
                }
            }
            const float raw0 = __shfl_sync(0xffffffffu, raw[0], 0);
            float wsum = 0.0f;
            for (int k = 0; k < nUsed; ++k)
            {
                float best = lbV, bestRaw = lbRaw;
                int bidx = lbI;
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                {
                    const float ov = __shfl_xor_sync(0xffffffffu, best, off);
                    const float orw = __shfl_xor_sync(0xffffffffu, bestRaw, off);
                    const int oi = __shfl_xor_sync(0xffffffffu, bidx, off);
                    if (ov > best || (ov == best && oi >= 0 && (bidx < 0 || oi < bidx)))
                    {
                        best = ov;
                        bestRaw = orw;
                        bidx = oi;
                    }
                }
                const int chosen = bidx < 0 ? 0 : bidx;
                // No valid pick (every score -inf or NaN): expert 0, weighed by its own score.
                const float chosenRaw = bidx < 0 ? raw0 : bestRaw;
                if (lane == 0)
                {
                    selT[k] = chosen;
                    wT[k] = chosenRaw;
                }
                wsum += chosenRaw;
                if (bidx >= 0 && (chosen & 31) == lane)
                {
                    const int slot = chosen >> 5;
                    lbV = -INFINITY;
                    lbRaw = 0.0f;
                    lbI = -1;
#pragma unroll
                    for (int i = 0; i < TS_DSV4_SELECT_PER_LANE; ++i)
                    {
                        if (i == slot)
                            v[i] = -INFINITY;
                        const int e = lane + 32 * i;
                        if (e < nExpert && (v[i] > lbV || (v[i] == lbV && lbI >= 0 && e < lbI)))
                        {
                            lbV = v[i];
                            lbRaw = raw[i];
                            lbI = e;
                        }
                    }
                }
            }
            if (lane == 0)
            {
                const float inv = 1.0f / fmaxf(wsum, 6.103515625e-5f);
                for (int k = 0; k < nUsed; ++k)
                    wT[k] = (norm ? wT[k] * inv : wT[k]) * wScale;
            }
        }
        return;
    }
    else
    {
        extern __shared__ float shSel[]; // [nExpert]
        for (int e = threadIdx.x; e < nExpert; e += blockDim.x)
            shSel[e] = ts_dsv4_router_score(lg[e], gating) + (bias != nullptr ? bias[e] : 0.0f);
        __syncthreads();

        __shared__ float shVal[256];
        __shared__ int shIdx[256];
        for (int k = 0; k < nUsed; ++k)
        {
            float best = -INFINITY;
            int bidx = -1;
            for (int e = threadIdx.x; e < nExpert; e += blockDim.x)
            {
                const float v = shSel[e];
                if (v > best || (v == best && e < bidx))
                {
                    best = v;
                    bidx = e;
                }
            }
            shVal[threadIdx.x] = best;
            shIdx[threadIdx.x] = bidx;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1)
            {
                if (threadIdx.x < s)
                {
                    const float ov = shVal[threadIdx.x + s];
                    const int oi = shIdx[threadIdx.x + s];
                    if (ov > shVal[threadIdx.x] || (ov == shVal[threadIdx.x] && oi >= 0 &&
                            (shIdx[threadIdx.x] < 0 || oi < shIdx[threadIdx.x])))
                    {
                        shVal[threadIdx.x] = ov;
                        shIdx[threadIdx.x] = oi;
                    }
                }
                __syncthreads();
            }
            if (threadIdx.x == 0)
            {
                const int chosen = shIdx[0] < 0 ? 0 : shIdx[0];
                selT[k] = chosen;
                shSel[chosen] = -INFINITY;
            }
            __syncthreads();
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        float w[16];
        float sum = 0.0f;
        for (int j = 0; j < nUsed; ++j)
        {
            w[j] = ts_dsv4_router_score(lg[selT[j]], gating);
            sum += w[j];
        }
        if (norm)
        {
            if (sum < 6.103515625e-5f)
                sum = 6.103515625e-5f;
            const float inv = 1.0f / sum;
            for (int j = 0; j < nUsed; ++j)
                w[j] *= inv;
        }
        for (int j = 0; j < nUsed; ++j)
            wT[j] = w[j] * wScale;
    }
}

// grouping plan: histogram / exclusive scan / slot scatter
extern "C" __global__ void ts_dsv4_moe_count_i32(
    const int32_t* __restrict__ sel, int32_t* __restrict__ counts, const int S)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S)
        return;
    atomicAdd(&counts[sel[i]], 1);
}

extern "C" __global__ void ts_dsv4_moe_scan_i32(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets,
    int32_t* __restrict__ cursors,
    const int nExpert)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    int off = 0;
    for (int e = 0; e < nExpert; ++e)
    {
        offsets[e] = off;
        cursors[e] = off;
        off += counts[e];
    }
}

extern "C" __global__ void ts_dsv4_moe_scatter_i32(
    const int32_t* __restrict__ sel,
    int32_t* __restrict__ cursors,
    int32_t* __restrict__ rowOfSlot,   // [S]
    int32_t* __restrict__ slotToken,   // [S]
    const int S,
    const int nUsed)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S)
        return;
    const int row = atomicAdd(&cursors[sel[i]], 1);
    rowOfSlot[i] = row;
    slotToken[row] = i / nUsed;
}

// ---------------------------------------------------------------------------
// Staged-weight expert projections (prefill).
//
// The per-token kernels below re-decode a quantized weight row for every
// member token: for IQ3_S that is ~32 codebook-grid lookups per 128 values,
// repeated m times, which dominates once an expert has more than a couple of
// routed tokens. These variants decode each row ONCE into shared memory as
// plain int8 plus a per-32-value float scale, then dot the staged row against
// every member token with nothing but dp4a. Weight-decode cost becomes O(1) in
// m instead of O(m).
//
// The staged row lives in REGISTERS, not shared memory: lane l owns the 32-value
// sub-blocks b = l, l+32, l+64, ... (four of them for a 4096-wide row), holding
// each as 8 packed ints plus its scale. That removes shared-memory bank
// conflicts entirely, needs no __syncthreads, and -- because a 36KB shared
// staging buffer capped occupancy at one block per SM -- lets many blocks run
// per SM.
//
// Scales fold the weight's block scale (and, for IQ3_S, the sub-block
// (1 + 2*ls) factor); the activation's own q8_1 scale is applied at use.
// Codebook values fit int8 for every supported type (IQ3_S grid |v| <= 15 after
// sign application, MXFP4 table |v| <= 12, Q6_K raw-32 is -32..31, Q5_1's codes
// are 0..31, the IQ2 grids |v| <= 43, IQ1_S's 8 g +/- 1 |v| <= 9, Q8_0 is already
// int8). Q6_K's two half-group scales are kept separately.
//
// Activations come from the DENSE split q8_1 layout (ts_quantize_q8_1_split_rows_f32):
// a contiguous int8 row plus a separate per-block float scale. The interleaved
// 36-byte ts_block_q8_1 is only 4-byte aligned, so reading it costs eight
// scattered 4-byte loads per sub-block; the dense form is read as two int4s.
// ---------------------------------------------------------------------------

#define TS_DSV4_MAX_SUBS_PER_LANE 5   // 32 lanes x 5 sub-blocks x 32 values = 5120 (DeepSeek V4.1's hidden width)
#define TS_DSV4_STAGE_ROWS 2          // weight rows staged per warp

// Decode one 32-value sub-block of a quantized weight row into 8 packed ints
// plus the scales of its two 16-value halves, and for Q2_K the mins those
// halves subtract (dmin * m, times the half's activation sum); 0 otherwise.
// Callers pass only the layouts it reads (ts_dsv4_dot_row_warp's Q6_K / IQ4
// branch, ts_dsv4_decode_sub_grouped's list), so its last branch, Q8_0, never
// sees a layout it does not decode.
__device__ __forceinline__ void ts_dsv4_decode_sub(
    const uint8_t* __restrict__ wRow, int wtype, int sub, int* __restrict__ w8,
    float* __restrict__ scLo, float* __restrict__ scHi,
    float* __restrict__ mnLo, float* __restrict__ mnHi)
{
    *mnLo = 0.0f;
    *mnHi = 0.0f;
    if (wtype == TS_DSV4_WTYPE_IQ3S)
    {
        const int sb = sub >> 3;
        const int iqs = (sub & 7) * 2;
        const uint8_t* blk = wRow + (size_t)sb * TS_DSV4_IQ3S_BLOCK_BYTES;
        const uint8_t* qs = blk + 2;
        const uint8_t* qh = blk + 2 + 64;
        const uint8_t* signs = blk + 2 + 64 + 8;
        const uint8_t* scales = blk + 2 + 64 + 8 + 32;

        int qs_pack0 = ts_dsv4_get_int_b2(qs, iqs + 0);
        int qs_pack1 = ts_dsv4_get_int_b2(qs, iqs + 1);
        const uint8_t* qsp0 = (const uint8_t*)&qs_pack0;
        const uint8_t* qsp1 = (const uint8_t*)&qs_pack1;
        uint8_t qsb[8];
        qsb[0] = qsp0[0]; qsb[1] = qsp0[1]; qsb[2] = qsp0[2]; qsb[3] = qsp0[3];
        qsb[4] = qsp1[0]; qsb[5] = qsp1[1]; qsb[6] = qsp1[2]; qsb[7] = qsp1[3];

        const int qhv = qh[iqs / 2];
        const int signs_packed_32 = ts_dsv4_get_int_b2(signs, iqs / 2);
        const uint8_t* sp8 = (const uint8_t*)&signs_packed_32;

#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2)
        {
            const uint32_t g0 = ts_iq3s_grid[qsb[l0 + 0] | ((qhv << (8 - l0)) & 0x100)];
            const uint32_t g1 = ts_iq3s_grid[qsb[l0 + 1] | ((qhv << (7 - l0)) & 0x100)];
            const int s0 = __vcmpne4(((sp8[l0/2] & 0x03) << 7) | ((sp8[l0/2] & 0x0C) << 21), 0x00000000);
            const int s1 = __vcmpne4(((sp8[l0/2] & 0x30) << 3) | ((sp8[l0/2] & 0xC0) << 17), 0x00000000);
            w8[l0 + 0] = __vsub4((int)g0 ^ s0, s0);
            w8[l0 + 1] = __vsub4((int)g1 ^ s1, s1);
        }
        *scLo = __half2float(*(const half*)blk)
              * (float)(1 + 2 * ((scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F));
        *scHi = *scLo;
    }
    else if (wtype == TS_DSV4_WTYPE_MXFP4)
    {
        const uint8_t* blk = wRow + (size_t)sub * TS_DSV4_MXFP4_BLOCK_BYTES;
        const uint8_t* qs = blk + 1;
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int2 v = ts_dsv4_int_from_table16(ts_dsv4_get_int_b1(qs, j), ts_kvalues_mxfp4);
            w8[j + 0] = v.x;
            w8[j + 4] = v.y;
        }
        *scLo = __uint_as_float(((uint32_t)blk[0]) << 23) * 0.5f;
        *scHi = *scLo;
    }
    else if (wtype == TS_DSV4_WTYPE_IQ4NL || wtype == TS_DSV4_WTYPE_IQ4XS)
    {
        // Both formats keep a 32-value sub-block's nibbles as 16 bytes: low nibbles are values
        // 0-15, high nibbles 16-31. IQ4_NL is one such block with its own f16 scale; IQ4_XS
        // packs eight under one f16 scale and a 6-bit sub-block scale biased by 32.
        const uint8_t* qs;
        float scale;
        if (wtype == TS_DSV4_WTYPE_IQ4NL)
        {
            const uint8_t* blk = wRow + (size_t)sub * TS_DSV4_IQ4NL_BLOCK_BYTES;
            qs = blk + 2;
            scale = __half2float(*(const half*)blk);
        }
        else
        {
            const uint8_t* blk = wRow + (size_t)(sub >> 3) * TS_DSV4_IQ4XS_BLOCK_BYTES;
            const int ib = sub & 7;
            const int scalesH = *(const uint16_t*)(blk + 2);
            const int ls = ((blk[4 + ib / 2] >> (4 * (ib & 1))) & 0xF) | (((scalesH >> (2 * ib)) & 3) << 4);
            qs = blk + 8 + 16 * ib;
            scale = __half2float(*(const half*)blk) * (float)(ls - 32);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int2 v = ts_dsv4_int_from_table16(ts_dsv4_get_int_b2(qs, j), ts_dsv4_kvalues_iq4nl);
            w8[j + 0] = v.x;
            w8[j + 4] = v.y;
        }
        *scLo = scale;
        *scHi = scale;
    }
    else if (wtype == TS_DSV4_WTYPE_Q6_K)
    {
        const int sb = sub >> 3;
        const int ls = sub & 7;
        const uint8_t* sblock = wRow + (size_t)sb * TS_DSV4_Q6_K_BLOCK_BYTES;
        const uint8_t* ql = sblock;
        const uint8_t* qh = sblock + 128;
        const int8_t* scales = (const int8_t*)(sblock + 192);
        const float dSb = __half2float(*(const half*)(sblock + 208));

        const int halfIdx = ls >> 2;
        const int group = ls & 3;
        const uint8_t* qlGroup = ql + halfIdx * 64 + ((group & 1) ? 32 : 0);
        const uint8_t* qhGroup = qh + halfIdx * 32;
        const int qlShift = group >= 2 ? 4 : 0;
        const int qhShift = group * 2;

#pragma unroll
        for (int lb = 0; lb < 4; ++lb)
        {
            const int g = lb * 2;
            const int raw0 = (((unsigned)ts_dsv4_get_int_b2(qlGroup, g) >> qlShift) & 0x0F0F0F0F)
                | ((((unsigned)ts_dsv4_get_int_b2(qhGroup, g) >> qhShift) & 0x03030303) << 4);
            const int raw1 = (((unsigned)ts_dsv4_get_int_b2(qlGroup, g + 1) >> qlShift) & 0x0F0F0F0F)
                | ((((unsigned)ts_dsv4_get_int_b2(qhGroup, g + 1) >> qhShift) & 0x03030303) << 4);
            w8[g + 0] = __vsubss4(raw0, 0x20202020);
            w8[g + 1] = __vsubss4(raw1, 0x20202020);
        }
        *scLo = dSb * (float)scales[halfIdx * 8 + group * 2 + 0];
        *scHi = dSb * (float)scales[halfIdx * 8 + group * 2 + 1];
    }
    else if (wtype == TS_DSV4_WTYPE_Q2_K)
    {
        const uint8_t* blk = wRow + (size_t)(sub >> 3) * TS_DSV4_Q2_K_BLOCK_BYTES;
        const int g = sub & 7;
        const int j = g & 3;
        const uint8_t* qs = blk + 16 + (g >> 2) * 32;
#pragma unroll
        for (int k = 0; k < 8; ++k)
            w8[k] = (ts_dsv4_get_int_b4(qs, k) >> (2 * j)) & 0x03030303;
        const float d = __half2float(*(const half*)(blk + 80));
        const float dmin = __half2float(*(const half*)(blk + 82));
        const uint8_t sc0 = blk[(g >> 2) * 8 + j * 2 + 0];
        const uint8_t sc1 = blk[(g >> 2) * 8 + j * 2 + 1];
        *scLo = d * (float)(sc0 & 0xF);
        *scHi = d * (float)(sc1 & 0xF);
        *mnLo = dmin * (float)(sc0 >> 4);
        *mnHi = dmin * (float)(sc1 >> 4);
    }
    else if (wtype == TS_DSV4_WTYPE_Q3_K)
    {
        ts_dsv4_q3k_group_ints(wRow + (size_t)(sub >> 3) * TS_DSV4_Q3_K_BLOCK_BYTES, sub & 7, w8, scLo, scHi);
    }
    else if (wtype == TS_DSV4_WTYPE_Q4_K || wtype == TS_DSV4_WTYPE_Q5_K)
    {
        // One 32-value sub-block of a 256-value super-block: nibbles (plus Q5_K's high bit)
        // times d * scale, minus dmin * min, both 6-bit and shared by the whole sub-block.
        const bool q5 = wtype == TS_DSV4_WTYPE_Q5_K;
        const uint8_t* blk = wRow + (size_t)(sub >> 3) * (q5 ? TS_DSV4_Q5_K_BLOCK_BYTES : TS_DSV4_Q4_K_BLOCK_BYTES);
        const int is = sub & 7, c = is >> 1, hiNib = is & 1;
        const uint8_t* sc = blk + 4;
        int scale, mn;
        if (is < 4) { scale = sc[is] & 63; mn = sc[is + 4] & 63; }
        else { scale = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); mn = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
        const uint8_t* qs = blk + (q5 ? 48 : 16) + c * 32;
#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            int v = ts_dsv4_get_int_b4(qs, k);
            v = hiNib ? ((v >> 4) & 0x0F0F0F0F) : (v & 0x0F0F0F0F);
            if (q5)
                v |= ((ts_dsv4_get_int_b4(blk + 16, k) >> is) & 0x01010101) << 4;
            w8[k] = v;
        }
        *scLo = __half2float(*(const half*)blk) * (float)scale;
        *scHi = *scLo;
        *mnLo = __half2float(*(const half*)(blk + 2)) * (float)mn;
        *mnHi = *mnLo;
    }
    else // Q8_0
    {
        const uint8_t* blk = wRow + (size_t)sub * TS_DSV4_Q8_0_BLOCK_BYTES;
        const int8_t* src = (const int8_t*)(blk + 2);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            w8[j] = ts_dsv4_get_int_b2(src, j);
        *scLo = __half2float(*(const half*)blk);
        *scHi = *scLo;
    }
}

// The grouped expert kernels' decoder (register-staged and tensor-core prefill): every layout
// ts_dsv4_decode_sub reads, plus IQ2_XS and IQ3_XXS, Qwen3.8-Flash-Next UD-Q2_K_XL's gate/up
// experts, Q5_1, its UD-Q4_K_XL's down experts, and IQ2_XXS, IQ2_S and IQ1_S, DeepSeek V4.1 1017
// UD-IQ2_XXS's. Kept out of ts_dsv4_decode_sub itself because the per-token decode kernels inline that
// one for their Q6_K / IQ4 rows: with the two branches there, Qwen3.8-Flash-Next's decode on two
// A40s fell from 56.9 to 55.0-55.6 tok/s (its down projection is IQ4_NL).
__device__ __forceinline__ void ts_dsv4_decode_sub_grouped(
    const uint8_t* __restrict__ wRow, int wtype, int sub, int* __restrict__ w8,
    float* __restrict__ scLo, float* __restrict__ scHi,
    float* __restrict__ mnLo, float* __restrict__ mnHi)
{
    if (wtype == TS_DSV4_WTYPE_IQ2XS)
    {
        // Four 16-bit words per 32 values, each a 9-bit index into the 512-entry grid (eight
        // values) and 7 sign bits whose parity is the 8th; one 4-bit scale per 16 values, as
        // ggml's dequantize_row_iq2_xs: d * (0.5 + scale) / 4.
        const ts_dsv4_block_iq2_xs* b =
            (const ts_dsv4_block_iq2_xs*)(wRow + (size_t)(sub >> 3) * TS_DSV4_IQ2XS_BLOCK_BYTES);
        const int group = sub & 7;
        const int qa = ts_dsv4_get_int_b2(b->qs, group * 2 + 0);
        const int qb = ts_dsv4_get_int_b2(b->qs, group * 2 + 1);
        const uint32_t q2[4] = { (uint32_t)qa & 0xFFFF, (uint32_t)qa >> 16, (uint32_t)qb & 0xFFFF, (uint32_t)qb >> 16 };
#pragma unroll
        for (int l = 0; l < 4; ++l)
        {
            const uint2 gridPos = ((const uint2*)iq2xs_grid)[q2[l] & 0x1FF];
            const uint32_t v = q2[l] >> 9;
            const uint32_t signs = (v ^ ((uint32_t)(__popc(v) & 1) << 7)) * 0x01010101u;
            const int s0 = __vcmpne4(signs & 0x08040201u, 0);
            const int s1 = __vcmpne4(signs & 0x80402010u, 0);
            w8[2 * l + 0] = __vsub4((int)gridPos.x ^ s0, s0);
            w8[2 * l + 1] = __vsub4((int)gridPos.y ^ s1, s1);
        }
        const float d = __half2float(b->d);
        *scLo = d * (0.5f + (float)(b->scales[group] & 0x0F)) * 0.25f;
        *scHi = d * (0.5f + (float)(b->scales[group] >> 4)) * 0.25f;
    }
    else if (wtype == TS_DSV4_WTYPE_IQ3XXS)
    {
        // Eight grid indices (four values each) per 32 values, then one word per 32 values: four
        // 7-bit sign indices into ksigns_iq2xs (one per eight values) under a 4-bit scale, as
        // ggml's dequantize_row_iq3_xxs: d * (0.5 + scale) / 2.
        const ts_dsv4_block_iq3_xxs* b =
            (const ts_dsv4_block_iq3_xxs*)(wRow + (size_t)(sub >> 3) * TS_DSV4_IQ3XXS_BLOCK_BYTES);
        const int group = sub & 7;
        const int qa = ts_dsv4_get_int_b2(b->qs, group * 2 + 0);
        const int qb = ts_dsv4_get_int_b2(b->qs, group * 2 + 1);
        const uint8_t* q3a = (const uint8_t*)&qa;
        const uint8_t* q3b = (const uint8_t*)&qb;
        const uint32_t aux32 = (uint32_t)ts_dsv4_get_int_b2(b->qs, 16 + group);
#pragma unroll
        for (int l = 0; l < 4; ++l)
        {
            const uint8_t i0 = l < 2 ? q3a[2 * l] : q3b[2 * l - 4];
            const uint8_t i1 = l < 2 ? q3a[2 * l + 1] : q3b[2 * l - 3];
            const int signsPacked = ksigns_iq2xs[(aux32 >> (7 * l)) & 0x7F];
            const int s0 = __vcmpne4(((signsPacked & 0x03) << 7) | ((signsPacked & 0x0C) << 21), 0x00000000);
            const int s1 = __vcmpne4(((signsPacked & 0x30) << 3) | ((signsPacked & 0xC0) << 17), 0x00000000);
            w8[2 * l + 0] = __vsub4((int)iq3xxs_grid[i0] ^ s0, s0);
            w8[2 * l + 1] = __vsub4((int)iq3xxs_grid[i1] ^ s1, s1);
        }
        const float sc = __half2float(b->d) * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        *scLo = sc;
        *scHi = sc;
    }
    else if (wtype == TS_DSV4_WTYPE_Q5_1)
    {
        // d * code + m: the grouped kernels subtract the min times the activation sum, so the
        // block's offset m goes in negated.
        const uint8_t* blk = wRow + (size_t)sub * TS_DSV4_Q5_1_BLOCK_BYTES;
        ts_dsv4_q5_1_ints(blk, w8);
        *scLo = __half2float(*(const half*)blk);
        *scHi = *scLo;
        *mnLo = -__half2float(*(const half*)(blk + 2));
        *mnHi = *mnLo;
        return;
    }
    else if (wtype == TS_DSV4_WTYPE_IQ2XXS)
    {
        // One word of four grid indices (eight values each) per 32 values, then one word of four
        // 7-bit sign indices into ksigns_iq2xs under a 4-bit scale, as ggml's dequantize_row_iq2_xxs:
        // d * (0.5 + scale) / 4 for the whole group.
        const ts_dsv4_block_iq2_xxs* b =
            (const ts_dsv4_block_iq2_xxs*)(wRow + (size_t)(sub >> 3) * TS_DSV4_IQ2XXS_BLOCK_BYTES);
        const int group = sub & 7;
        const int q2 = ts_dsv4_get_int_b2(b->qs, group * 2 + 0);
        const uint8_t* aux8 = (const uint8_t*)&q2;
        const uint32_t aux32 = (uint32_t)ts_dsv4_get_int_b2(b->qs, group * 2 + 1);
#pragma unroll
        for (int l = 0; l < 4; ++l)
        {
            const uint2 gridPos = ((const uint2*)iq2xxs_grid)[aux8[l]];
            const int signsPacked = ksigns_iq2xs[(aux32 >> (7 * l)) & 0x7F];
            const int s0 = __vcmpne4(((signsPacked & 0x03) << 7) | ((signsPacked & 0x0C) << 21), 0x00000000);
            const int s1 = __vcmpne4(((signsPacked & 0x30) << 3) | ((signsPacked & 0xC0) << 17), 0x00000000);
            w8[2 * l + 0] = __vsub4((int)gridPos.x ^ s0, s0);
            w8[2 * l + 1] = __vsub4((int)gridPos.y ^ s1, s1);
        }
        const float sc = __half2float(b->d) * (0.5f + (float)(aux32 >> 28)) * 0.25f;
        *scLo = sc;
        *scHi = sc;
    }
    else if (wtype == TS_DSV4_WTYPE_IQ2S)
    {
        // d * (0.5 + scale) / 4 per 16 values, as ggml's dequantize_row_iq2_s.
        const ts_dsv4_block_iq2_s* b =
            (const ts_dsv4_block_iq2_s*)(wRow + (size_t)(sub >> 3) * TS_DSV4_IQ2S_BLOCK_BYTES);
        const int group = sub & 7;
        ts_dsv4_iq2s_group_ints(b, group, w8);
        const float d = __half2float(b->d);
        *scLo = d * (0.5f + (float)(b->scales[group] & 0x0F)) * 0.25f;
        *scHi = d * (0.5f + (float)(b->scales[group] >> 4)) * 0.25f;
    }
    else if (wtype == TS_DSV4_WTYPE_IQ1S)
    {
        // 8 g +/- 1 under d * (2s + 1) / 8: the tensor-core tile's F16 scale * int is then exactly
        // ggml's value, and nothing is subtracted.
        const float sc = ts_dsv4_iq1s_group_ints(
            (const ts_dsv4_block_iq1_s*)(wRow + (size_t)(sub >> 3) * TS_DSV4_IQ1S_BLOCK_BYTES), sub & 7, w8);
        *scLo = sc;
        *scHi = sc;
    }
    else if (wtype == TS_DSV4_WTYPE_Q2_0)
    {
        // ISTA's Q2_0 experts: half of a 64-value block per sub-block. Code c decodes to (c - 1) * d, and c - 1
        // is already a signed int8 in {-1, 0, 1, 2}, so nothing is subtracted afterwards. Byte i of the half
        // holds values 4i..4i+3 at bit offsets 0, 2, 4, 6.
        const uint8_t* blk = wRow + (size_t)(sub >> 1) * TS_DSV4_Q2_0_BLOCK_BYTES;
        const uint8_t* qs = blk + 2 + 8 * (sub & 1);
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const int c = qs[i];
            const int w = (c & 0x03) | ((c & 0x0C) << 6) | ((c & 0x30) << 12) | ((c & 0xC0) << 18);
            w8[i] = __vsub4(w, 0x01010101);
        }
        *scLo = __half2float(*(const half*)blk);
        *scHi = *scLo;
    }
    else if (wtype == TS_DSV4_WTYPE_IQ3S || wtype == TS_DSV4_WTYPE_MXFP4 || wtype == TS_DSV4_WTYPE_IQ4NL
             || wtype == TS_DSV4_WTYPE_IQ4XS || wtype == TS_DSV4_WTYPE_Q6_K || wtype == TS_DSV4_WTYPE_Q2_K
             || wtype == TS_DSV4_WTYPE_Q3_K || wtype == TS_DSV4_WTYPE_Q4_K || wtype == TS_DSV4_WTYPE_Q5_K
             || wtype == TS_DSV4_WTYPE_Q8_0)
    {
        ts_dsv4_decode_sub(wRow, wtype, sub, w8, scLo, scHi, mnLo, mnHi);
        return;
    }
    else
    {
        // A layout no decoder reads: NaN scales, as ts_dsv4_dot_row_warp answers, never a Q8_0 read.
        // The test lives in this list and not in ts_dsv4_decode_sub's last branch: the per-token
        // decode kernels inline that function, and touching it moved their register allocation (Q8_0
        // decode 3% slower, IQ4_NL 2-5% faster on an RTX 3080), where this costs them nothing.
#pragma unroll
        for (int j = 0; j < 8; ++j)
            w8[j] = 0;
        *scLo = __int_as_float(0x7fc00000);
        *scHi = *scLo;
    }
    *mnLo = 0.0f;
    *mnHi = 0.0f;
}

// One warp: decode row `wRow` into registers, then dot it against `m` member
// tokens' dense split-q8_1 activations, writing one output per token.
// NK = sub-blocks per lane = inDim / 1024, a compile-time constant so the
// staged weights stay in registers.
// NR = weight rows staged per warp. Each activation load then feeds NR rows,
// which is what keeps this off the L2 bandwidth ceiling: with one row per warp
// every (row, token) pair pulls the whole activation row again.
template <int NK, int NR>
__device__ __forceinline__ void ts_dsv4_expert_row_tokens(
    const uint8_t* __restrict__ wBase, long long rowBytes, int wtype, int lane,
    const int8_t* __restrict__ actQs, const float* __restrict__ actD,
    const int32_t* __restrict__ slotToken, int off, int m,
    int inDim, int rowLimit,
    float* __restrict__ out, long long outStride, int outRow0)
{
    int w[NR][NK][8];
    float scLo[NR][NK], scHi[NR][NK], mnLo[NR][NK], mnHi[NR][NK];
    bool live[NR];
    // Only Q2_K, Q4_K, Q5_K and Q5_1 subtract mins (Q5_1 a negated offset), which needs each
    // half's activation sum; one branch per warp.
    const bool hasMin = wtype == TS_DSV4_WTYPE_Q2_K || wtype == TS_DSV4_WTYPE_Q4_K || wtype == TS_DSV4_WTYPE_Q5_K
        || wtype == TS_DSV4_WTYPE_Q5_1;
    // A width that is not a multiple of 1024 (V4.1's experts are 2304 wide) leaves the last
    // sub-blocks of the row to the first lanes only.
    const int nSub = inDim / 32;
#pragma unroll
    for (int rr = 0; rr < NR; ++rr)
    {
        live[rr] = (outRow0 + rr) < rowLimit;
        const uint8_t* wRow = wBase + (size_t)(outRow0 + rr) * rowBytes;
#pragma unroll
        for (int k = 0; k < NK; ++k)
        {
            if (live[rr] && lane + 32 * k < nSub)
                ts_dsv4_decode_sub_grouped(wRow, wtype, lane + 32 * k, w[rr][k], &scLo[rr][k], &scHi[rr][k],
                                   &mnLo[rr][k], &mnHi[rr][k]);
        }
    }

    for (int i = 0; i < m; ++i)
    {
        // slotToken == nullptr: activations are already packed in slot order
        const int arow = slotToken != nullptr ? slotToken[off + i] : (off + i);
        const int8_t* aQs = actQs + (size_t)arow * inDim;
        const float* aD = actD + (size_t)arow * nSub;

        float sum[NR];
#pragma unroll
        for (int rr = 0; rr < NR; ++rr)
            sum[rr] = 0.0f;

#pragma unroll
        for (int k = 0; k < NK; ++k)
        {
            const int b = lane + 32 * k;
            if (b >= nSub)
                break;
            const int4 a0 = *(const int4*)(aQs + (size_t)b * 32);
            const int4 a1 = *(const int4*)(aQs + (size_t)b * 32 + 16);
            const float dAct = aD[b];
            int sumLo = 0, sumHi = 0;
            if (hasMin)
            {
                sumLo = ts_dsv4_dp4a(0x01010101, a0.x, sumLo);
                sumLo = ts_dsv4_dp4a(0x01010101, a0.y, sumLo);
                sumLo = ts_dsv4_dp4a(0x01010101, a0.z, sumLo);
                sumLo = ts_dsv4_dp4a(0x01010101, a0.w, sumLo);
                sumHi = ts_dsv4_dp4a(0x01010101, a1.x, sumHi);
                sumHi = ts_dsv4_dp4a(0x01010101, a1.y, sumHi);
                sumHi = ts_dsv4_dp4a(0x01010101, a1.z, sumHi);
                sumHi = ts_dsv4_dp4a(0x01010101, a1.w, sumHi);
            }
#pragma unroll
            for (int rr = 0; rr < NR; ++rr)
            {
                int lo = 0, hi = 0;
                lo = ts_dsv4_dp4a(w[rr][k][0], a0.x, lo);
                lo = ts_dsv4_dp4a(w[rr][k][1], a0.y, lo);
                lo = ts_dsv4_dp4a(w[rr][k][2], a0.z, lo);
                lo = ts_dsv4_dp4a(w[rr][k][3], a0.w, lo);
                hi = ts_dsv4_dp4a(w[rr][k][4], a1.x, hi);
                hi = ts_dsv4_dp4a(w[rr][k][5], a1.y, hi);
                hi = ts_dsv4_dp4a(w[rr][k][6], a1.z, hi);
                hi = ts_dsv4_dp4a(w[rr][k][7], a1.w, hi);
                float v = scLo[rr][k] * (float)lo + scHi[rr][k] * (float)hi;
                if (hasMin)
                    v -= mnLo[rr][k] * (float)sumLo + mnHi[rr][k] * (float)sumHi;
                sum[rr] += dAct * v;
            }
        }

#pragma unroll
        for (int rr = 0; rr < NR; ++rr)
        {
            const float s = ts_dsv4_warp_sum(sum[rr]);
            if (lane == 0 && live[rr])
                out[(size_t)(off + i) * outStride + outRow0 + rr] = s;
        }
    }
}
// ---------------------------------------------------------------------------

#define TS_DSV4_STAGE_WARPS 8

// Staged gate/up. grid.x = row tiles, grid.y = expert, grid.z = 0 gate / 1 up.
// grid.y carries the expert because blockIdx.x varies fastest at launch, so a
// single expert's blocks stay co-resident and share its activations in cache.
// An expert's weight rows: the stacked tensor's expert e, or - when the engine keeps only some
// experts in VRAM - wherever its table says they are: a VRAM slot, or pinned host memory the GPU
// reads over the bus (the Qwen3.8-Flash-Next engine's expert residency). A null table is the
// stacked layout, so every caller that passes none runs exactly as before.
__device__ __forceinline__ const uint8_t* ts_dsv4_expert_rows(
    const uint8_t* stacked, const unsigned long long* __restrict__ table, int e, size_t expertBytes)
{
    return table != nullptr ? reinterpret_cast<const uint8_t*>(table[e]) : stacked + (size_t)e * expertBytes;
}

extern "C" __global__ void ts_dsv4_moe_gateup_staged_f32(
    const uint8_t* __restrict__ gateW,
    const uint8_t* __restrict__ upW,
    const int8_t* __restrict__ actQs,
    const float* __restrict__ actD,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ slotToken,
    float* __restrict__ gateOut,
    float* __restrict__ upOut,
    const int wtype,
    const int ff,
    const int inDim,
    const long long rowBytes,
    const unsigned long long* __restrict__ gateTab,
    const unsigned long long* __restrict__ upTab)
{
    const int e = blockIdx.y;
    const int m = counts[e];
    if (m == 0)
        return;

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int r = (blockIdx.x * TS_DSV4_STAGE_WARPS + warpId) * TS_DSV4_STAGE_ROWS;
    if (r >= ff)
        return;

    const bool isUp = blockIdx.z == 1;
    const uint8_t* wBase = isUp ? ts_dsv4_expert_rows(upW, upTab, e, (size_t)ff * rowBytes)
                                : ts_dsv4_expert_rows(gateW, gateTab, e, (size_t)ff * rowBytes);
    float* outBase = isUp ? upOut : gateOut;
    const int off = offsets[e];

    switch ((inDim + 1023) / 1024)
    {
        case 5:
            ts_dsv4_expert_row_tokens<5, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, slotToken, off, m, inDim, ff, outBase, ff, r);
            break;
        case 4:
            ts_dsv4_expert_row_tokens<4, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, slotToken, off, m, inDim, ff, outBase, ff, r);
            break;
        case 3:
            ts_dsv4_expert_row_tokens<3, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, slotToken, off, m, inDim, ff, outBase, ff, r);
            break;
        case 2:
            ts_dsv4_expert_row_tokens<2, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, slotToken, off, m, inDim, ff, outBase, ff, r);
            break;
        default:
            ts_dsv4_expert_row_tokens<1, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, slotToken, off, m, inDim, ff, outBase, ff, r);
            break;
    }
}

// Staged down projection over packed hidden rows (already in slot order, so no
// slotToken indirection).
extern "C" __global__ void ts_dsv4_moe_down_staged_f32(
    const uint8_t* __restrict__ downW,
    const int8_t* __restrict__ actQs,
    const float* __restrict__ actD,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    float* __restrict__ downOut,
    const int wtype,
    const int E,
    const int ff,
    const long long rowBytes,
    const unsigned long long* __restrict__ downTab)
{
    const int e = blockIdx.y;
    const int m = counts[e];
    if (m == 0)
        return;

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int r = (blockIdx.x * TS_DSV4_STAGE_WARPS + warpId) * TS_DSV4_STAGE_ROWS;
    if (r >= E)
        return;

    const uint8_t* wBase = ts_dsv4_expert_rows(downW, downTab, e, (size_t)E * rowBytes);
    const int off = offsets[e];

    switch ((ff + 1023) / 1024)
    {
        case 5:
            ts_dsv4_expert_row_tokens<5, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, nullptr, off, m, ff, E, downOut, E, r);
            break;
        case 4:
            ts_dsv4_expert_row_tokens<4, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, nullptr, off, m, ff, E, downOut, E, r);
            break;
        case 3:
            ts_dsv4_expert_row_tokens<3, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, nullptr, off, m, ff, E, downOut, E, r);
            break;
        case 2:
            ts_dsv4_expert_row_tokens<2, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, nullptr, off, m, ff, E, downOut, E, r);
            break;
        default:
            ts_dsv4_expert_row_tokens<1, TS_DSV4_STAGE_ROWS>(wBase, rowBytes, wtype, lane,
                actQs, actD, nullptr, off, m, ff, E, downOut, E, r);
            break;
    }
}

// ---------------------------------------------------------------------------
// Grouped expert projections on tensor cores (prefill).
//
// The staged kernels above dot one weight row against one member token at a
// time on the CUDA cores; a 512-token prefill of V4.1's 384-expert layers spent
// three quarters of its time there. Here a CTA owns 128 weight rows of one
// expert and every member token of it: each k-step decodes a 128 x 128 weight
// tile ONCE into shared memory as F16 (block scale and min folded in), stages
// the member tokens' F32 activations as F16 beside it, and eight warps run
// mma.m16n8k16 (F16 in, F32 accumulate) over 16 weight rows x 8 tokens each.
// Activations stay F16 rather than q8_1, so this path is also more exact than
// the one it replaces. Members beyond one pass of TS_MOE_MMA_N tokens take
// further passes over the same rows.
// ---------------------------------------------------------------------------

#define TS_MOE_MMA_M 128                          // weight rows per CTA (8 warps x 16)
#define TS_MOE_MMA_N 32                           // member tokens per pass (4 n8 fragments)
#define TS_MOE_MMA_NFRAG (TS_MOE_MMA_N / 8)
#define TS_MOE_MMA_KSUBS 4                        // 32-value sub-blocks per k-step
#define TS_MOE_MMA_K (TS_MOE_MMA_KSUBS * 32)      // 128 values per k-step
#define TS_MOE_MMA_LDS (TS_MOE_MMA_K + 8)         // row stride in halves: 272 B, conflict-free fragment loads
#define TS_MOE_MMA_THREADS 256

// out[off + i][row] = W_e[row] . act[member i] for the member tokens of expert blockIdx.y and
// rows [blockIdx.x * 128, +128). grid.z selects the weight/output pair (gate and up share a launch).
// slotToken maps a member to its token's activation row; null means the activations are already
// in member (slot) order, as the down projection's are. Requires inDim % 128 == 0.
extern "C" __global__ void __launch_bounds__(TS_MOE_MMA_THREADS, 2) ts_dsv4_moe_mma_f32(
    const uint8_t* __restrict__ w0,
    const uint8_t* __restrict__ w1,
    const float* __restrict__ act,
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ slotToken,
    float* __restrict__ out0,
    float* __restrict__ out1,
    const int wtype,
    const int rows,
    const int inDim,
    const long long rowBytes,
    const unsigned long long* __restrict__ tab0,
    const unsigned long long* __restrict__ tab1)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const int e = blockIdx.y;
    const int m = counts[e];
    const int r0 = blockIdx.x * TS_MOE_MMA_M;
    if (m == 0 || r0 >= rows)
        return;

    const uint8_t* wBase = blockIdx.z == 0 ? ts_dsv4_expert_rows(w0, tab0, e, (size_t)rows * rowBytes)
                                           : ts_dsv4_expert_rows(w1, tab1, e, (size_t)rows * rowBytes);
    float* out = blockIdx.z == 0 ? out0 : out1;
    const int off = offsets[e];
    const int nSub = inDim / 32;

    __shared__ __align__(16) half smW[TS_MOE_MMA_M][TS_MOE_MMA_LDS];
    __shared__ __align__(16) half smA[TS_MOE_MMA_N][TS_MOE_MMA_LDS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int tig = lane & 3;
    const int warpM = warp * 16;

    for (int p0 = 0; p0 < m; p0 += TS_MOE_MMA_N)
    {
        const int live = min(TS_MOE_MMA_N, m - p0);
        const int liveFrags = (live + 7) >> 3;

        float acc[TS_MOE_MMA_NFRAG][4];
#pragma unroll
        for (int nf = 0; nf < TS_MOE_MMA_NFRAG; ++nf)
        {
            acc[nf][0] = 0.0f; acc[nf][1] = 0.0f; acc[nf][2] = 0.0f; acc[nf][3] = 0.0f;
        }

        for (int k0 = 0; k0 < inDim; k0 += TS_MOE_MMA_K)
        {
            // Weights: 128 rows x 4 sub-blocks, two per thread; the four sub-blocks of one row
            // go to four neighbouring threads, which share its quant block through L1.
#pragma unroll
            for (int i = 0; i < (TS_MOE_MMA_M * TS_MOE_MMA_KSUBS) / TS_MOE_MMA_THREADS; ++i)
            {
                const int u = tid + i * TS_MOE_MMA_THREADS;
                const int r = u / TS_MOE_MMA_KSUBS;
                const int j = u % TS_MOE_MMA_KSUBS;
                const int sub = k0 / 32 + j;
                uint4* dst = reinterpret_cast<uint4*>(&smW[r][j * 32]);
                if (r0 + r < rows && sub < nSub)
                {
                    int w8[8];
                    float scLo, scHi, mnLo, mnHi;
                    ts_dsv4_decode_sub_grouped(wBase + (size_t)(r0 + r) * rowBytes, wtype, sub, w8, &scLo, &scHi, &mnLo, &mnHi);
                    uint32_t h[16];
#pragma unroll
                    for (int k = 0; k < 8; ++k)
                    {
                        const float sc = k < 4 ? scLo : scHi;
                        const float mn = k < 4 ? mnLo : mnHi;
                        const int v = w8[k];
                        const half2 p = __floats2half2_rn(sc * (float)(int8_t)(v & 0xFF) - mn,
                                                          sc * (float)(int8_t)((v >> 8) & 0xFF) - mn);
                        const half2 q = __floats2half2_rn(sc * (float)(int8_t)((v >> 16) & 0xFF) - mn,
                                                          sc * (float)(int8_t)((v >> 24) & 0xFF) - mn);
                        h[2 * k + 0] = *reinterpret_cast<const uint32_t*>(&p);
                        h[2 * k + 1] = *reinterpret_cast<const uint32_t*>(&q);
                    }
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        dst[k] = make_uint4(h[4 * k], h[4 * k + 1], h[4 * k + 2], h[4 * k + 3]);
                }
                else
                {
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        dst[k] = make_uint4(0, 0, 0, 0);
                }
            }

            // Activations: 32 tokens x 128 values, 16 per thread (4 float4 loads).
            {
                const int n = tid >> 3;
                const int c = (tid & 7) * 16;
                uint4* dst = reinterpret_cast<uint4*>(&smA[n][c]);
                if (n < live)
                {
                    const int arow = slotToken != nullptr ? slotToken[off + p0 + n] : off + p0 + n;
                    const float4* src = reinterpret_cast<const float4*>(act + (size_t)arow * inDim + k0 + c);
                    uint32_t h[8];
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                    {
                        const float4 f = src[k];
                        const half2 p = __floats2half2_rn(f.x, f.y);
                        const half2 q = __floats2half2_rn(f.z, f.w);
                        h[2 * k + 0] = *reinterpret_cast<const uint32_t*>(&p);
                        h[2 * k + 1] = *reinterpret_cast<const uint32_t*>(&q);
                    }
                    dst[0] = make_uint4(h[0], h[1], h[2], h[3]);
                    dst[1] = make_uint4(h[4], h[5], h[6], h[7]);
                }
                else
                {
                    dst[0] = make_uint4(0, 0, 0, 0);
                    dst[1] = make_uint4(0, 0, 0, 0);
                }
            }
            __syncthreads();

#pragma unroll
            for (int kk = 0; kk < TS_MOE_MMA_K; kk += 16)
            {
                const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&smW[warpM + g][kk + 2 * tig]);
                const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&smW[warpM + g + 8][kk + 2 * tig]);
                const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&smW[warpM + g][kk + 8 + 2 * tig]);
                const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&smW[warpM + g + 8][kk + 8 + 2 * tig]);
#pragma unroll
                for (int nf = 0; nf < TS_MOE_MMA_NFRAG; ++nf)
                {
                    if (nf >= liveFrags)
                        break;
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&smA[nf * 8 + g][kk + 2 * tig]);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&smA[nf * 8 + g][kk + 8 + 2 * tig]);
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                        : "+f"(acc[nf][0]), "+f"(acc[nf][1]), "+f"(acc[nf][2]), "+f"(acc[nf][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
                }
            }
            __syncthreads();
        }

        // c0/c1: weight row g, tokens 2*tig and 2*tig+1; c2/c3: row g+8.
        const int row0 = r0 + warpM + g;
        const int row1 = row0 + 8;
#pragma unroll
        for (int nf = 0; nf < TS_MOE_MMA_NFRAG; ++nf)
        {
            const int t0 = nf * 8 + 2 * tig;
            const int t1 = t0 + 1;
            if (t0 < live)
            {
                float* o = out + (size_t)(off + p0 + t0) * rows;
                if (row0 < rows) o[row0] = acc[nf][0];
                if (row1 < rows) o[row1] = acc[nf][2];
            }
            if (t1 < live)
            {
                float* o = out + (size_t)(off + p0 + t1) * rows;
                if (row0 < rows) o[row0] = acc[nf][1];
                if (row1 < rows) o[row1] = acc[nf][3];
            }
        }
    }
#else
    (void)w0; (void)w1; (void)act; (void)counts; (void)offsets; (void)slotToken;
    (void)out0; (void)out1; (void)wtype; (void)rows; (void)inDim; (void)rowBytes;
#endif
}

// Expert gate/up projections over grouped tokens. grid (nExpert, ffTiles, 2);
// z = 0 -> gate, z = 1 -> up. Each warp sweeps a weight row and dots it
// against every member token's q8_1 activation, so weight bytes are read once
// per chunk while activations come from L2.
#define TS_DSV4_MOE_TILE 32

extern "C" __global__ void ts_dsv4_moe_gateup_f32(
    const uint8_t* __restrict__ gateW,   // [nExpert, ff, inDim]
    const uint8_t* __restrict__ upW,
    const ts_dsv4_block_q8_1* __restrict__ act, // [nt] rows
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ slotToken,
    float* __restrict__ gateOut,         // [S, ff]
    float* __restrict__ upOut,           // [S, ff]
    const int wtype,
    const int ff,
    const int inDim,
    const long long rowBytes,
    const unsigned long long* __restrict__ gateTab,
    const unsigned long long* __restrict__ upTab)
{
    const int e = blockIdx.x;
    const int m = counts[e];
    if (m == 0)
        return;

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int rBegin = blockIdx.y * TS_DSV4_MOE_TILE + warpId * (TS_DSV4_MOE_TILE / 8);
    const int rEnd = blockIdx.y * TS_DSV4_MOE_TILE + (warpId + 1) * (TS_DSV4_MOE_TILE / 8);

    const bool isUp = blockIdx.z == 1;
    const uint8_t* wBase = isUp ? ts_dsv4_expert_rows(upW, upTab, e, (size_t)ff * rowBytes)
                                : ts_dsv4_expert_rows(gateW, gateTab, e, (size_t)ff * rowBytes);
    float* outBase = isUp ? upOut : gateOut;
    const int off = offsets[e];
    const int actBlocks = inDim / TS_DSV4_QK8_1;

    for (int rr = rBegin; rr < rEnd && rr < ff; ++rr)
    {
        const uint8_t* wRow = wBase + (size_t)rr * rowBytes;
        for (int i = 0; i < m; ++i)
        {
            const int tok = slotToken[off + i];
            const float dot = ts_dsv4_dot_row_warp(wRow, wtype, act + (size_t)tok * actBlocks, inDim, lane);
            if (lane == 0)
                outBase[(size_t)(off + i) * ff + rr] = dot;
        }
    }
}

// Expert down projection over grouped (already packed) hidden rows.
extern "C" __global__ void ts_dsv4_moe_down_f32(
    const uint8_t* __restrict__ downW,   // [nExpert, E, ff]
    const ts_dsv4_block_q8_1* __restrict__ act, // [S] packed rows
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    float* __restrict__ downOut,         // [S, E]
    const int wtype,
    const int E,
    const int ff,
    const long long rowBytes,
    const unsigned long long* __restrict__ downTab)
{
    const int e = blockIdx.x;
    const int m = counts[e];
    if (m == 0)
        return;

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int rBegin = blockIdx.y * TS_DSV4_MOE_TILE + warpId * (TS_DSV4_MOE_TILE / 8);
    const int rEnd = blockIdx.y * TS_DSV4_MOE_TILE + (warpId + 1) * (TS_DSV4_MOE_TILE / 8);

    const uint8_t* wBase = ts_dsv4_expert_rows(downW, downTab, e, (size_t)E * rowBytes);
    const int off = offsets[e];
    const int actBlocks = ff / TS_DSV4_QK8_1;

    for (int rr = rBegin; rr < rEnd && rr < E; ++rr)
    {
        const uint8_t* wRow = wBase + (size_t)rr * rowBytes;
        for (int i = 0; i < m; ++i)
        {
            const float dot = ts_dsv4_dot_row_warp(wRow, wtype, act + (size_t)(off + i) * actBlocks, ff, lane);
            if (lane == 0)
                downOut[(size_t)(off + i) * E + rr] = dot;
        }
    }
}

// Decode-specialized expert projections: grid maps directly over the selected
// experts (no grouping plan). z = 0 -> gate, z = 1 -> up.
// One block row per (token, selected expert) slot: a decode step's tokens each run exactly
// the arithmetic a single token's step runs, however many share the launch.
extern "C" __global__ void ts_dsv4_moe_gateup_decode_f32(
    const uint8_t* __restrict__ gateW,
    const uint8_t* __restrict__ upW,
    const ts_dsv4_block_q8_1* __restrict__ act, // [nt] rows
    const int32_t* __restrict__ sel,     // [nt * nUsed]
    float* __restrict__ gateOut,         // [nt * nUsed, ff]
    float* __restrict__ upOut,           // [nt * nUsed, ff]
    const int wtype,
    const int ff,
    const int inDim,
    const long long rowBytes,
    const int nUsed,
    const unsigned long long* __restrict__ gateTab,
    const unsigned long long* __restrict__ upTab)
{
    const int slot = blockIdx.x;
    const int e = sel[slot];
    act += (size_t)(slot / nUsed) * (inDim / TS_DSV4_QK8_1);

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int rBegin = blockIdx.y * TS_DSV4_MOE_TILE + warpId * (TS_DSV4_MOE_TILE / 8);
    const int rEnd = blockIdx.y * TS_DSV4_MOE_TILE + (warpId + 1) * (TS_DSV4_MOE_TILE / 8);

    const bool isUp = blockIdx.z == 1;
    const uint8_t* wBase = isUp ? ts_dsv4_expert_rows(upW, upTab, e, (size_t)ff * rowBytes)
                                : ts_dsv4_expert_rows(gateW, gateTab, e, (size_t)ff * rowBytes);
    float* outBase = isUp ? upOut : gateOut;

    for (int rr = rBegin; rr < rEnd && rr < ff; ++rr)
    {
        const float dot = ts_dsv4_dot_row_warp(wBase + (size_t)rr * rowBytes, wtype, act, inDim, lane);
        if (lane == 0)
            outBase[(size_t)slot * ff + rr] = dot;
    }
}

extern "C" __global__ void ts_dsv4_moe_down_decode_f32(
    const uint8_t* __restrict__ downW,
    const ts_dsv4_block_q8_1* __restrict__ act, // [nt * nUsed] rows
    const int32_t* __restrict__ sel,
    float* __restrict__ downOut,         // [nt * nUsed, E]
    const int wtype,
    const int E,
    const int ff,
    const long long rowBytes,
    const unsigned long long* __restrict__ downTab)
{
    const int slot = blockIdx.x;
    const int e = sel[slot];

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int rBegin = blockIdx.y * TS_DSV4_MOE_TILE + warpId * (TS_DSV4_MOE_TILE / 8);
    const int rEnd = blockIdx.y * TS_DSV4_MOE_TILE + (warpId + 1) * (TS_DSV4_MOE_TILE / 8);

    const uint8_t* wBase = ts_dsv4_expert_rows(downW, downTab, e, (size_t)E * rowBytes);
    const int actBlocks = ff / TS_DSV4_QK8_1;

    for (int rr = rBegin; rr < rEnd && rr < E; ++rr)
    {
        const float dot = ts_dsv4_dot_row_warp(wBase + (size_t)rr * rowBytes, wtype, act + (size_t)slot * actBlocks, ff, lane);
        if (lane == 0)
            downOut[(size_t)slot * E + rr] = dot;
    }
}

// ts_dsv4_moe_down_decode_f32 for short expert rows (ff <= 1024: Qwen3.8-Flash-Next's 640): eight lanes per output
// row, four rows a warp at once. A whole warp per 640-wide row left most lanes one unit each and spent the row on its
// five-step reduction; this kernel's down projection moved half the gate/up's bytes in more time. Every slot is
// computed the same way whatever the call's row count (a verify window's rows equal a decode step's).
extern "C" __global__ void ts_dsv4_moe_down_decode_short_f32(
    const uint8_t* __restrict__ downW,
    const ts_dsv4_block_q8_1* __restrict__ act, // [nt * nUsed] rows
    const int32_t* __restrict__ sel,
    float* __restrict__ downOut,         // [nt * nUsed, E]
    const int wtype,
    const int E,
    const int ff,
    const long long rowBytes,
    const unsigned long long* __restrict__ downTab)
{
    const int slot = blockIdx.x;
    const int e = sel[slot];
    const int lane = threadIdx.x % 32;
    const int group = lane >> 3, sub = lane & 7;
    // 32 rows a block: 8 warps x 4 groups.
    const int rr = blockIdx.y * 32 + (threadIdx.x / 32) * 4 + group;
    const uint8_t* wBase = ts_dsv4_expert_rows(downW, downTab, e, (size_t)E * rowBytes);
    const int actBlocks = ff / TS_DSV4_QK8_1;
    float v = rr < E ? ts_dsv4_dot_row_part(wBase + (size_t)rr * rowBytes, wtype, act + (size_t)slot * actBlocks, ff, sub, 8) : 0.0f;
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    if (sub == 0 && rr < E)
        downOut[(size_t)slot * E + rr] = v;
}

// Dense projection of decode-size inputs already quantized to q8_1: one warp per
// output row through ts_dsv4_dot_row_warp, grid (ceil(outDim / 32), rows). It
// reads only the caller's buffers, so a captured decode step can replay it - the
// shared matmul route quantizes into shared scratch and cannot be captured, and
// a checkpoint whose dense weights are K-quants the q8_1 decode kernels do not
// cover (UD-Q2_K_XL's Q5_K) otherwise lost decode capture altogether.
// out[r, o] = sum_i bf16(w[o, i]) * x[r, i] for up to 16 rows: one warp per output row, F32 activations
// (as ggml computes a BF16 matmul), every row's sum in its own register so the weights are read once.
// inDim is even. Reads only the buffers passed, so a decode step can capture it.
extern "C" __global__ void ts_dense_bf16_rows_f32(
    const unsigned short* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    const int outDim,
    const int inDim,
    const int rows)
{
    const int o = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (o >= outDim)
        return;
    const unsigned int* wr = (const unsigned int*)(w + (size_t)o * inDim);
    float acc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r)
        acc[r] = 0.0f;
    for (int i = lane; i < inDim / 2; i += 32)
    {
        const unsigned int p = wr[i];
        const float w0 = __uint_as_float(p << 16), w1 = __uint_as_float(p & 0xffff0000u);
#pragma unroll
        for (int r = 0; r < 16; ++r)
            if (r < rows)
            {
                const float2 xv = *(const float2*)(x + (size_t)r * inDim + 2 * i);
                acc[r] = fmaf(w1, xv.y, fmaf(w0, xv.x, acc[r]));
            }
    }
#pragma unroll
    for (int r = 0; r < 16; ++r)
        if (r < rows)
        {
            const float v = ts_dsv4_warp_sum(acc[r]);
            if (lane == 0)
                out[(size_t)r * outDim + o] = v;
        }
}

// Split-K form of ts_dsv4_dense_q81_rows_f32 for projections with few output rows and long inputs (the
// hyper-connections' 10240 -> 320 down projection): one 256-thread block per output row and token, its 8
// warps sharing the row's decode units, so the projection fills the GPU instead of 80 warps of it.
extern "C" __global__ void ts_dsv4_dense_q81_rows_splitk_f32(
    const uint8_t* __restrict__ w,
    const ts_dsv4_block_q8_1* __restrict__ act,
    float* __restrict__ out,
    const int wtype,
    const int outDim,
    const int inDim,
    const long long rowBytes)
{
    const int o = blockIdx.x, row = blockIdx.y;
    const int warpId = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __shared__ float part[8];
    const float v = ts_dsv4_warp_sum(ts_dsv4_dot_row_part(w + (size_t)o * rowBytes, wtype,
        act + (size_t)row * (inDim / TS_DSV4_QK8_1), inDim, threadIdx.x, blockDim.x));
    if (lane == 0)
        part[warpId] = v;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float s = 0.0f;
        for (int i = 0; i < (int)(blockDim.x >> 5); ++i)
            s += part[i];
        out[(size_t)row * outDim + o] = s;
    }
}

// Split-K form of ts_dense_bf16_rows_f32: one 256-thread block per output row, up to 16 tokens.
extern "C" __global__ void ts_dense_bf16_rows_splitk_f32(
    const unsigned short* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ out,
    const int outDim,
    const int inDim,
    const int rows)
{
    const int o = blockIdx.x;
    const int warpId = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned int* wr = (const unsigned int*)(w + (size_t)o * inDim);
    float acc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r)
        acc[r] = 0.0f;
    for (int i = threadIdx.x; i < inDim / 2; i += blockDim.x)
    {
        const unsigned int p = wr[i];
        const float w0 = __uint_as_float(p << 16), w1 = __uint_as_float(p & 0xffff0000u);
#pragma unroll
        for (int r = 0; r < 16; ++r)
            if (r < rows)
            {
                const float2 xv = *(const float2*)(x + (size_t)r * inDim + 2 * i);
                acc[r] = fmaf(w1, xv.y, fmaf(w0, xv.x, acc[r]));
            }
    }
    __shared__ float part[16][8];
#pragma unroll
    for (int r = 0; r < 16; ++r)
        if (r < rows)
        {
            const float v = ts_dsv4_warp_sum(acc[r]);
            if (lane == 0)
                part[r][warpId] = v;
        }
    __syncthreads();
    if (threadIdx.x < rows)
    {
        float s = 0.0f;
        for (int i = 0; i < (int)(blockDim.x >> 5); ++i)
            s += part[threadIdx.x][i];
        out[(size_t)threadIdx.x * outDim + o] = s;
    }
}

// ts_dsv4_dense_q81_rows_f32 for up to 16 activation rows in one block: a warp per output row, dotted against every
// activation row in turn, so the weight row comes from memory once and from L1 after it (a block per activation row read
// it once per row: a 4-row verify window's LM head took 4x one row's time). Eight output rows a block keep the grid as
// wide as four rows' blocks of the one-row kernel. Every row's dot is that kernel's (ts_dsv4_dot_row_warp), bit for bit.
extern "C" __global__ void ts_dsv4_dense_q81_rows_multi_f32(
    const uint8_t* __restrict__ w,               // [outDim, inDim] rows of rowBytes
    const ts_dsv4_block_q8_1* __restrict__ act,  // [rows, inDim / 32]
    float* __restrict__ out,                     // [rows, outDim]
    const int wtype,
    const int outDim,
    const int inDim,
    const long long rowBytes,
    const int rows)
{
    const int lane = threadIdx.x % 32;
    const int rr = blockIdx.x * 8 + threadIdx.x / 32;
    if (rr >= outDim)
        return;
    const uint8_t* wRow = w + (size_t)rr * rowBytes;
    for (int row = 0; row < rows; ++row)
    {
        const float dot = ts_dsv4_dot_row_warp(wRow, wtype, act + (size_t)row * (inDim / TS_DSV4_QK8_1), inDim, lane);
        if (lane == 0)
            out[(size_t)row * outDim + rr] = dot;
    }
}

extern "C" __global__ void ts_dsv4_dense_q81_rows_f32(
    const uint8_t* __restrict__ w,               // [outDim, inDim] rows of rowBytes
    const ts_dsv4_block_q8_1* __restrict__ act,  // [rows, inDim / 32]
    float* __restrict__ out,                     // [rows, outDim]
    const int wtype,
    const int outDim,
    const int inDim,
    const long long rowBytes)
{
    const int row = blockIdx.y;
    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int rBegin = blockIdx.x * TS_DSV4_MOE_TILE + warpId * (TS_DSV4_MOE_TILE / 8);
    const int rEnd = blockIdx.x * TS_DSV4_MOE_TILE + (warpId + 1) * (TS_DSV4_MOE_TILE / 8);
    const ts_dsv4_block_q8_1* a = act + (size_t)row * (inDim / TS_DSV4_QK8_1);

    for (int rr = rBegin; rr < rEnd && rr < outDim; ++rr)
    {
        const float dot = ts_dsv4_dot_row_warp(w + (size_t)rr * rowBytes, wtype, a, inDim, lane);
        if (lane == 0)
            out[(size_t)row * outDim + rr] = dot;
    }
}

// ---- decode-size dense projections, decode once and dot every row ----
// The kernels above walk a weight row once per activation row and decode each unit again for each: a 4-row verify
// window's 2560 -> 10240 IQ4_XS projection took 33 us (0.42 TB/s) on the RTX 5090 against 15 us for one row, the cost
// being the decode, not the bytes (an IQ4 nibble went through a byte load from the lookup table). Here each unit's
// weights are decoded to int8 lanes once - the IQ4 table held in registers and applied with __byte_perm, as llama.cpp's
// get_int_from_table_16 does - and dotted against every activation row (TS_DSV4_FAST_R at a time). W warps share an
// output row's units (few outputs of long rows need them to fill the GPU), 8 / W output rows a 256-thread block, the
// warps' partials added in warp order. A row's sum depends on the shape alone, never on how many rows the call holds,
// so a verify window's row is its own decode step's bit for bit. 11.5 us for that projection; the formats are the ones
// Qwen3.8 Flash Next's ISTA pack and DeepSeek V4.1 keep dense (Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL).

#define TS_DSV4_FAST_R_MAX 8

struct ts_dsv4_table16 { uint32_t t[4]; };

__device__ __forceinline__ ts_dsv4_table16 ts_dsv4_load_table16(const int8_t* table)
{
    ts_dsv4_table16 r;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        r.t[i] = (uint32_t)(uint8_t)table[4 * i] | ((uint32_t)(uint8_t)table[4 * i + 1] << 8)
               | ((uint32_t)(uint8_t)table[4 * i + 2] << 16) | ((uint32_t)(uint8_t)table[4 * i + 3] << 24);
    return r;
}

// Eight packed nibbles onto table values: .x the low nibbles', .y the high nibbles' (llama.cpp get_int_from_table_16).
__device__ __forceinline__ int2 ts_dsv4_table16_perm(const uint32_t q4, const ts_dsv4_table16& tb)
{
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210u | ((q4 & 0x88888888u) >> 1);
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        const uint32_t shift = 16 * i;
        const uint32_t lo = __byte_perm(tb.t[0], tb.t[1], q4 >> shift);
        const uint32_t hi = __byte_perm(tb.t[2], tb.t[3], q4 >> shift);
        tmp[i] = __byte_perm(lo, hi, sel >> shift);
    }
    return make_int2((int)__byte_perm(tmp[0], tmp[1], 0x6420), (int)__byte_perm(tmp[0], tmp[1], 0x7531));
}

__device__ __forceinline__ int ts_dsv4_q81_int(const ts_dsv4_block_q8_1* a, int k)
{
    return reinterpret_cast<const int*>(a->qs)[k];
}

// IQ4_XS: unit g is the row's 32-value sub-block g.
template <int R>
__device__ __forceinline__ void ts_dsv4_fast_iq4xs(const uint8_t* __restrict__ wRow, int g, const ts_dsv4_table16& tb,
    const ts_dsv4_block_q8_1* const (&a)[R], float (&sum)[R])
{
    const uint8_t* blk = wRow + (size_t)(g >> 3) * TS_DSV4_IQ4XS_BLOCK_BYTES;
    const int ib = g & 7;
    const uint2 q01 = *reinterpret_cast<const uint2*>(blk + 8 + 16 * ib);
    const uint2 q23 = *reinterpret_cast<const uint2*>(blk + 16 + 16 * ib);
    const uint32_t hdr = *reinterpret_cast<const uint32_t*>(blk);        // d, scales_h
    const uint32_t sl = *reinterpret_cast<const uint32_t*>(blk + 4);     // scales_l[4]
    const int ls = ((sl >> (8 * (ib >> 1) + 4 * (ib & 1))) & 0xF) | ((((int)(hdr >> 16) >> (2 * ib)) & 3) << 4);
    const float scale = __half2float(__ushort_as_half((unsigned short)(hdr & 0xFFFF))) * (float)(ls - 32);
    const uint32_t q[4] = { q01.x, q01.y, q23.x, q23.y };
    int w[8];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int2 v = ts_dsv4_table16_perm(q[j], tb);
        w[j] = v.x;
        w[j + 4] = v.y;
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
    {
        const ts_dsv4_block_q8_1* ab = a[r] + g;
        int dot = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k)
            dot = ts_dsv4_dp4a(w[k], ts_dsv4_q81_int(ab, k), dot);
        sum[r] += __half2float(ab->d) * (scale * (float)dot);
    }
}

// Q8_0: unit b is the row's 32-value block b.
template <int R>
__device__ __forceinline__ void ts_dsv4_fast_q8_0(const uint8_t* __restrict__ wRow, int b,
    const ts_dsv4_block_q8_1* const (&a)[R], float (&sum)[R])
{
    const uint8_t* blk = wRow + (size_t)b * TS_DSV4_Q8_0_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const half*>(blk));
    int w[8];
#pragma unroll
    for (int k = 0; k < 8; ++k)
        w[k] = ts_dsv4_get_int_b2(blk + 2, k);
#pragma unroll
    for (int r = 0; r < R; ++r)
    {
        const ts_dsv4_block_q8_1* ab = a[r] + b;
        int dot = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k)
            dot = ts_dsv4_dp4a(w[k], ts_dsv4_q81_int(ab, k), dot);
        sum[r] += __half2float(ab->d) * (d * (float)dot);
    }
}

// Q2_0 (ISTA): unit u is the row's 32-value half u of its 64-value blocks; value = (code - 1) * d.
__device__ __forceinline__ int ts_dsv4_q2_0_spread(uint32_t c)
{
    c &= 0xFF;
    return (int)((c & 0x03) | ((c & 0x0C) << 6) | ((c & 0x30) << 12) | ((c & 0xC0) << 18));
}

template <int R>
__device__ __forceinline__ void ts_dsv4_fast_q2_0(const uint8_t* __restrict__ wRow, int u,
    const ts_dsv4_block_q8_1* const (&a)[R], float (&sum)[R])
{
    const uint8_t* blk = wRow + (size_t)(u >> 1) * TS_DSV4_Q2_0_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const half*>(blk));
    const uint8_t* qs = blk + 2 + 8 * (u & 1);
    const uint32_t lo = (uint32_t)ts_dsv4_get_int_b2(qs, 0), hi = (uint32_t)ts_dsv4_get_int_b2(qs, 1);
    int w[8];
#pragma unroll
    for (int k = 0; k < 4; ++k)
    {
        w[k] = ts_dsv4_q2_0_spread(lo >> (8 * k));
        w[k + 4] = ts_dsv4_q2_0_spread(hi >> (8 * k));
    }
#pragma unroll
    for (int r = 0; r < R; ++r)
    {
        const ts_dsv4_block_q8_1* ab = a[r] + u;
        int dot = 0, asum = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            const int av = ts_dsv4_q81_int(ab, k);
            dot = ts_dsv4_dp4a(w[k], av, dot);
            asum = ts_dsv4_dp4a(0x01010101, av, asum);
        }
        sum[r] += __half2float(ab->d) * (d * (float)(dot - asum));
    }
}

// Q4_K / Q5_K: unit p is the pair of 32-value sub-blocks 2c, 2c + 1 of super-block p >> 2 (they share 32 nibble bytes).
template <int R, bool Q5>
__device__ __forceinline__ void ts_dsv4_fast_q45k(const uint8_t* __restrict__ wRow, int p,
    const ts_dsv4_block_q8_1* const (&a)[R], float (&sum)[R])
{
    const int sb = p >> 2, c = p & 3;
    const uint8_t* blk = wRow + (size_t)sb * (Q5 ? TS_DSV4_Q5_K_BLOCK_BYTES : TS_DSV4_Q4_K_BLOCK_BYTES);
    const uint4* qs = reinterpret_cast<const uint4*>(blk + (Q5 ? 48 : 16) + c * 32);
    const uint4 q0 = qs[0], q1 = qs[1];
    const int qw[8] = { (int)q0.x, (int)q0.y, (int)q0.z, (int)q0.w, (int)q1.x, (int)q1.y, (int)q1.z, (int)q1.w };
    int hw[8];
    if (Q5)
    {
        const uint4* qh = reinterpret_cast<const uint4*>(blk + 16);
        const uint4 h0 = qh[0], h1 = qh[1];
        hw[0] = (int)h0.x; hw[1] = (int)h0.y; hw[2] = (int)h0.z; hw[3] = (int)h0.w;
        hw[4] = (int)h1.x; hw[5] = (int)h1.y; hw[6] = (int)h1.z; hw[7] = (int)h1.w;
    }
    const uint32_t dd = *reinterpret_cast<const uint32_t*>(blk);
    const float d = __half2float(__ushort_as_half((unsigned short)(dd & 0xFFFF)));
    const float dmin = __half2float(__ushort_as_half((unsigned short)(dd >> 16)));
    const uint8_t* sc = blk + 4;
#pragma unroll
    for (int hi = 0; hi < 2; ++hi)
    {
        const int is = 2 * c + hi;
        int scale, mn;
        if (is < 4) { scale = sc[is] & 63; mn = sc[is + 4] & 63; }
        else { scale = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); mn = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
        int v[8];
#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            v[k] = hi ? ((qw[k] >> 4) & 0x0F0F0F0F) : (qw[k] & 0x0F0F0F0F);
            if (Q5)
                v[k] |= ((hw[k] >> is) & 0x01010101) << 4;
        }
        const float ds = d * (float)scale, dm = dmin * (float)mn;
#pragma unroll
        for (int r = 0; r < R; ++r)
        {
            const ts_dsv4_block_q8_1* a8 = a[r] + sb * 8 + is;
            int dot = 0, asum = 0;
#pragma unroll
            for (int k = 0; k < 8; ++k)
            {
                const int av = ts_dsv4_q81_int(a8, k);
                dot = ts_dsv4_dp4a(v[k], av, dot);
                asum = ts_dsv4_dp4a(0x01010101, av, asum);
            }
            sum[r] += __half2float(a8->d) * (ds * (float)dot - dm * (float)asum);
        }
    }
}

// Q3_K: unit g is the row's 32-value group g (two 16-value halves, a scale each).
template <int R>
__device__ __forceinline__ void ts_dsv4_fast_q3k(const uint8_t* __restrict__ wRow, int g,
    const ts_dsv4_block_q8_1* const (&a)[R], float (&sum)[R])
{
    int w[8];
    float scLo, scHi;
    ts_dsv4_q3k_group_ints(wRow + (size_t)(g >> 3) * TS_DSV4_Q3_K_BLOCK_BYTES, g & 7, w, &scLo, &scHi);
#pragma unroll
    for (int r = 0; r < R; ++r)
    {
        const ts_dsv4_block_q8_1* ab = a[r] + g;
        int lo = 0, hi = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            lo = ts_dsv4_dp4a(w[k], ts_dsv4_q81_int(ab, k), lo);
            hi = ts_dsv4_dp4a(w[k + 4], ts_dsv4_q81_int(ab, k + 4), hi);
        }
        sum[r] += __half2float(ab->d) * (scLo * (float)lo + scHi * (float)hi);
    }
}

// The symmetric 32-value sub-block layouts ts_dsv4_decode_sub reads (Q6_K, IQ4_NL): unit g is sub-block g.
template <int R>
__device__ __forceinline__ void ts_dsv4_fast_sub(const uint8_t* __restrict__ wRow, int wtype, int g,
    const ts_dsv4_block_q8_1* const (&a)[R], float (&sum)[R])
{
    int w[8];
    float scLo, scHi, mnLo, mnHi;
    ts_dsv4_decode_sub(wRow, wtype, g, w, &scLo, &scHi, &mnLo, &mnHi);
#pragma unroll
    for (int r = 0; r < R; ++r)
    {
        const ts_dsv4_block_q8_1* ab = a[r] + g;
        int lo = 0, hi = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k)
        {
            lo = ts_dsv4_dp4a(w[k], ts_dsv4_q81_int(ab, k), lo);
            hi = ts_dsv4_dp4a(w[k + 4], ts_dsv4_q81_int(ab, k + 4), hi);
        }
        sum[r] += __half2float(ab->d) * (scLo * (float)lo + scHi * (float)hi);
    }
}

// One thread's units first, first + step, ... of a row, a sum per activation row.
template <int R>
__device__ __forceinline__ void ts_dsv4_fast_row(const uint8_t* __restrict__ wRow, int wtype,
    const ts_dsv4_block_q8_1* const (&a)[R], int inDim, int first, int step, const ts_dsv4_table16& tb, float (&sum)[R])
{
#pragma unroll
    for (int r = 0; r < R; ++r)
        sum[r] = 0.0f;
    if (wtype == TS_DSV4_WTYPE_IQ4XS)
        for (int u = first; u < inDim / 32; u += step) ts_dsv4_fast_iq4xs<R>(wRow, u, tb, a, sum);
    else if (wtype == TS_DSV4_WTYPE_Q8_0)
        for (int u = first; u < inDim / 32; u += step) ts_dsv4_fast_q8_0<R>(wRow, u, a, sum);
    else if (wtype == TS_DSV4_WTYPE_Q2_0)
        for (int u = first; u < inDim / 32; u += step) ts_dsv4_fast_q2_0<R>(wRow, u, a, sum);
    else if (wtype == TS_DSV4_WTYPE_Q4_K)
        for (int u = first; u < inDim / 64; u += step) ts_dsv4_fast_q45k<R, false>(wRow, u, a, sum);
    else if (wtype == TS_DSV4_WTYPE_Q5_K)
        for (int u = first; u < inDim / 64; u += step) ts_dsv4_fast_q45k<R, true>(wRow, u, a, sum);
    else if (wtype == TS_DSV4_WTYPE_Q3_K)
        for (int u = first; u < inDim / 32; u += step) ts_dsv4_fast_q3k<R>(wRow, u, a, sum);
    else if (wtype == TS_DSV4_WTYPE_Q6_K || wtype == TS_DSV4_WTYPE_IQ4NL)
        for (int u = first; u < inDim / 32; u += step) ts_dsv4_fast_sub<R>(wRow, wtype, u, a, sum);
    else
    {
#pragma unroll
        for (int r = 0; r < R; ++r)
            sum[r] = __int_as_float(0x7fc00000);
    }
}

// out[rows, outDim] = act[rows] . w^T: W warps an output row, R activation rows a pass (rows past R take further passes,
// each row's arithmetic unchanged). grid ceil(outDim / (8 / W)), 256 threads.
template <int R, int W>
__device__ __forceinline__ void ts_dsv4_dense_fast(
    const uint8_t* __restrict__ w, const ts_dsv4_block_q8_1* __restrict__ act, float* __restrict__ out,
    const int wtype, const int outDim, const int inDim, const long long rowBytes, const int rows)
{
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int sub = warp % W;
    const int rr = blockIdx.x * (8 / W) + warp / W;
    ts_dsv4_table16 tb;
    if (wtype == TS_DSV4_WTYPE_IQ4XS)
        tb = ts_dsv4_load_table16(ts_dsv4_kvalues_iq4nl);
    const int actBlocks = inDim / TS_DSV4_QK8_1;
    __shared__ float part[8][R];
    for (int r0 = 0; r0 < rows; r0 += R)
    {
        const int n = min(R, rows - r0);
        const ts_dsv4_block_q8_1* a[R];
#pragma unroll
        for (int r = 0; r < R; ++r)
            a[r] = act + (size_t)(r0 + min(r, n - 1)) * actBlocks;
        float s[R];
        if (rr < outDim)
            ts_dsv4_fast_row<R>(w + (size_t)rr * rowBytes, wtype, a, inDim, sub * 32 + lane, 32 * W, tb, s);
        else
        {
#pragma unroll
            for (int r = 0; r < R; ++r)
                s[r] = 0.0f;
        }
        if (W == 1)
        {
#pragma unroll
            for (int r = 0; r < R; ++r)
            {
                const float v = ts_dsv4_warp_sum(s[r]);
                if (lane == 0 && r < n && rr < outDim)
                    out[(size_t)(r0 + r) * outDim + rr] = v;
            }
            continue;
        }
#pragma unroll
        for (int r = 0; r < R; ++r)
        {
            const float v = ts_dsv4_warp_sum(s[r]);
            if (lane == 0)
                part[warp][r] = v;
        }
        __syncthreads();
        if (sub == 0 && lane < n && rr < outDim)
        {
            float t = 0.0f;
            for (int k = 0; k < W; ++k)
                t += part[warp + k][lane];
            out[(size_t)(r0 + lane) * outDim + rr] = t;
        }
        __syncthreads();
    }
}

#define TS_DSV4_DENSE_FAST(R, W)                                                                                         \
    extern "C" __global__ void __launch_bounds__(256) ts_dsv4_dense_fast_r##R##_w##W##_f32(                              \
        const uint8_t* w, const ts_dsv4_block_q8_1* act, float* out, const int wtype, const int outDim, const int inDim, \
        const long long rowBytes, const int rows)                                                                         \
    { ts_dsv4_dense_fast<R, W>(w, act, out, wtype, outDim, inDim, rowBytes, rows); }
TS_DSV4_DENSE_FAST(1, 1)
TS_DSV4_DENSE_FAST(2, 1)
TS_DSV4_DENSE_FAST(4, 1)
TS_DSV4_DENSE_FAST(8, 1)
TS_DSV4_DENSE_FAST(1, 2)
TS_DSV4_DENSE_FAST(2, 2)
TS_DSV4_DENSE_FAST(4, 2)
TS_DSV4_DENSE_FAST(8, 2)
TS_DSV4_DENSE_FAST(1, 4)
TS_DSV4_DENSE_FAST(2, 4)
TS_DSV4_DENSE_FAST(4, 4)
TS_DSV4_DENSE_FAST(8, 4)
#undef TS_DSV4_DENSE_FAST

// ts_dsv4_moe_gateup_decode_f32 with sixteen lanes an output row and two rows a warp (the short down kernel's layout):
// a warp per row walked a 2560-wide Q2_0 row in 16-value quarters, five dependent steps a lane and four rows in turn,
// and a decode step's ten experts took 22.7 us on the RTX 5090 (412 GB/s) against 3.9 us for their down projection.
// Q2_0 rows go in 32-value units (ts_dsv4_fast_q2_0), the other layouts through ts_dsv4_dot_row_part; 4.4 us. Each slot
// is computed the same way whatever the call's row count. grid (slots, ceil(ff / 16), 2), 256 threads.
extern "C" __global__ void ts_dsv4_moe_gateup_decode_half_f32(
    const uint8_t* __restrict__ gateW,
    const uint8_t* __restrict__ upW,
    const ts_dsv4_block_q8_1* __restrict__ act, // [nt] rows
    const int32_t* __restrict__ sel,     // [nt * nUsed]
    float* __restrict__ gateOut,         // [nt * nUsed, ff]
    float* __restrict__ upOut,           // [nt * nUsed, ff]
    const int wtype,
    const int ff,
    const int inDim,
    const long long rowBytes,
    const int nUsed,
    const unsigned long long* __restrict__ gateTab,
    const unsigned long long* __restrict__ upTab)
{
    const int slot = blockIdx.x;
    const int e = sel[slot];
    const ts_dsv4_block_q8_1* a = act + (size_t)(slot / nUsed) * (inDim / TS_DSV4_QK8_1);
    const int lane = threadIdx.x & 31, group = lane >> 4, sub = lane & 15;
    const int rr = blockIdx.y * 16 + (threadIdx.x >> 5) * 2 + group;
    const bool isUp = blockIdx.z == 1;
    const uint8_t* wBase = isUp ? ts_dsv4_expert_rows(upW, upTab, e, (size_t)ff * rowBytes)
                                : ts_dsv4_expert_rows(gateW, gateTab, e, (size_t)ff * rowBytes);
    float v = 0.0f;
    if (rr < ff)
    {
        const uint8_t* wRow = wBase + (size_t)rr * rowBytes;
        if (wtype == TS_DSV4_WTYPE_Q2_0)
        {
            const ts_dsv4_block_q8_1* aa[1] = { a };
            float s[1] = { 0.0f };
            for (int u = sub; u < inDim / 32; u += 16)
                ts_dsv4_fast_q2_0<1>(wRow, u, aa, s);
            v = s[0];
        }
        else
            v = ts_dsv4_dot_row_part(wRow, wtype, a, inDim, sub, 16);
    }
    v += __shfl_xor_sync(0xffffffffu, v, 8);
    v += __shfl_xor_sync(0xffffffffu, v, 4);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    if (sub == 0 && rr < ff)
        (isUp ? upOut : gateOut)[(size_t)slot * ff + rr] = v;
}

// Clamped SwiGLU now comes from the shared kernel set
// (ts_silu_mul_clamp_f32, reached via Ops.SiLUMulClamp).

// ffnOut[t] = sum_j w[t,j] * downOut[row(t,j)] (+ shexp[t]); rowOfSlot null
// means identity (decode: slot j -> row j).
// ---- expert staging (the Qwen3.8-Flash-Next engine's expert residency) --------------------------------
// The experts a call selects that have no VRAM slot are copied once each, by coalesced 16-byte loads over
// the bus, from pinned host memory into VRAM staging, and their table entries point at the copy until
// ts_dsv4_moe_unstage puts them back. The expert kernels' own row reads straight from host memory are a
// stream of small uncoalesced bus transactions (UD-Q2_K_XL decoded at 0.4 tok/s that way). All three
// kernels read and write device state only, so a captured decode step replays them.

// Assigns staging slots to the distinct selected experts that live in host memory. stageOf[e] is -1
// between calls; over capacity an expert stays a read over the bus, in place.
extern "C" __global__ void ts_dsv4_moe_stage_plan(
    const int32_t* __restrict__ sel,
    const int slots,
    const uint8_t* __restrict__ isHost,
    int32_t* __restrict__ stageOf,
    int32_t* __restrict__ stageList,
    int32_t* __restrict__ stageCount,
    unsigned long long* __restrict__ tabGate,
    unsigned long long* __restrict__ tabUp,
    unsigned long long* __restrict__ tabDown,
    const unsigned long long staging,
    const long long stride,
    const long long offUp,
    const long long offDown,
    const int capacity)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < slots; i += gridDim.x * blockDim.x)
    {
        const int e = sel[i];
        if (e < 0 || !isHost[e] || atomicCAS(&stageOf[e], -1, -2) != -1)
            continue;
        const int k = atomicAdd(stageCount, 1);
        if (k >= capacity)
        {
            stageOf[e] = -1;
            continue;
        }
        stageList[k] = e;
        const unsigned long long base = staging + (unsigned long long)k * (unsigned long long)stride;
        tabGate[e] = base;
        tabUp[e] = base + (unsigned long long)offUp;
        tabDown[e] = base + (unsigned long long)offDown;
        stageOf[e] = k;
    }
}

// grid.x = staged-expert bound * chunks, grid.y = the three matrices (gate, up, down). Every byte count and
// base is 16-byte aligned (the engine checks before it enables staging).
extern "C" __global__ void ts_dsv4_moe_stage_copy(
    const int32_t* __restrict__ stageList,
    const int32_t* __restrict__ stageCount,
    const unsigned long long* __restrict__ hostTabGate,
    const unsigned long long* __restrict__ hostTabUp,
    const unsigned long long* __restrict__ hostTabDown,
    const long long bytesGate,
    const long long bytesUp,
    const long long bytesDown,
    uint8_t* __restrict__ staging,
    const long long stride,
    const long long offUp,
    const long long offDown,
    const int capacity,
    const int chunks)
{
    const int k = blockIdx.x / chunks;
    if (k >= min(*stageCount, capacity))
        return;
    const int e = stageList[k];
    const int part = blockIdx.y;
    const long long bytes = part == 0 ? bytesGate : part == 1 ? bytesUp : bytesDown;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(part == 0 ? hostTabGate[e] : part == 1 ? hostTabUp[e] : hostTabDown[e]);
    uint8_t* dst = staging + (size_t)k * (size_t)stride + (part == 0 ? 0 : part == 1 ? offUp : offDown);
    const long long units = bytes / 16;
    const long long per = (units + chunks - 1) / chunks;
    const long long begin = (long long)(blockIdx.x % chunks) * per;
    const long long end = min(units, begin + per);
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    for (long long u = begin + threadIdx.x; u < end; u += blockDim.x)
        d4[u] = s4[u];
}

// ts_dsv4_moe_stage_copy for a plan that keeps what earlier calls staged (a prompt read layer by layer: each
// layer's ubatches share one staging, so an expert crosses the bus once per layer, not once per ubatch): only
// the experts planned since *stageFrom (the count before this call's plan) are copied.
extern "C" __global__ void ts_dsv4_moe_stage_copy_from(
    const int32_t* __restrict__ stageList,
    const int32_t* __restrict__ stageCount,
    const int32_t* __restrict__ stageFrom,
    const unsigned long long* __restrict__ hostTabGate,
    const unsigned long long* __restrict__ hostTabUp,
    const unsigned long long* __restrict__ hostTabDown,
    const long long bytesGate,
    const long long bytesUp,
    const long long bytesDown,
    uint8_t* __restrict__ staging,
    const long long stride,
    const long long offUp,
    const long long offDown,
    const int capacity,
    const int chunks)
{
    const int k = *stageFrom + (int)(blockIdx.x / chunks);
    if (k >= min(*stageCount, capacity))
        return;
    const int e = stageList[k];
    const int part = blockIdx.y;
    const long long bytes = part == 0 ? bytesGate : part == 1 ? bytesUp : bytesDown;
    const uint8_t* src = reinterpret_cast<const uint8_t*>(part == 0 ? hostTabGate[e] : part == 1 ? hostTabUp[e] : hostTabDown[e]);
    uint8_t* dst = staging + (size_t)k * (size_t)stride + (part == 0 ? 0 : part == 1 ? offUp : offDown);
    const long long units = bytes / 16;
    const long long per = (units + chunks - 1) / chunks;
    const long long begin = (long long)(blockIdx.x % chunks) * per;
    const long long end = min(units, begin + per);
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    for (long long u = begin + threadIdx.x; u < end; u += blockDim.x)
        d4[u] = s4[u];
}

// One block: points the staged experts' table entries back at their resting addresses (hostTab*: per expert,
// its device-mapped host copy, or the zero expert when the host computes missed experts) and clears the plan.
extern "C" __global__ void ts_dsv4_moe_unstage(
    const int32_t* __restrict__ stageList,
    int32_t* __restrict__ stageCount,
    int32_t* __restrict__ stageOf,
    unsigned long long* __restrict__ tabGate,
    unsigned long long* __restrict__ tabUp,
    unsigned long long* __restrict__ tabDown,
    const unsigned long long* __restrict__ hostTabGate,
    const unsigned long long* __restrict__ hostTabUp,
    const unsigned long long* __restrict__ hostTabDown,
    const int capacity)
{
    const int n = min(*stageCount, capacity);
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        const int e = stageList[i];
        tabGate[e] = hostTabGate[e];
        tabUp[e] = hostTabUp[e];
        tabDown[e] = hostTabDown[e];
        stageOf[e] = -1;
    }
    __syncthreads();
    if (threadIdx.x == 0)
        *stageCount = 0;
}

// ---- host experts (expert residency, Strata's doorbell): the experts without a VRAM slot are computed by
// the host while the GPU computes the rest, inside one captured step. Per layer the GPU publishes the FFN
// input and the selection into host-mapped memory and advances a counter (the doorbell); a host thread
// sees it, computes the missed experts from pinned memory and sets a flag to the counter's value; the
// GPU's wait kernel then adds the host's sum into the layer's output. Each of these reads the counter
// from memory rather than taking it as an argument, so a replayed graph carries no stale value.
// hdr: [0] layer, [1] slots, [2] 1 when a selected expert has no slot (x was published).
extern "C" __global__ void ts_moe_doorbell_publish(
    const float* __restrict__ x,
    const int32_t* __restrict__ sel,
    const float* __restrict__ selW,
    const uint8_t* __restrict__ isHost,
    const int n,
    const int slots,
    const int layer,
    float* xOut,
    int32_t* idsOut,
    float* wOut,
    int32_t* hdr,
    unsigned int* seq)
{
    int miss = 0;
    for (int i = threadIdx.x; i < slots; i += blockDim.x)
    {
        const int e = sel[i];
        idsOut[i] = e;
        wOut[i] = selW[i];
        if (e >= 0 && isHost[e])
            miss = 1;
    }
    miss = __syncthreads_or(miss);
    if (miss)
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            xOut[i] = x[i];
    if (threadIdx.x == 0)
    {
        ((volatile int32_t*)hdr)[0] = layer;
        ((volatile int32_t*)hdr)[1] = slots;
        ((volatile int32_t*)hdr)[2] = miss;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        *(volatile unsigned int*)seq = *(volatile unsigned int*)seq + 1u;
        __threadfence_system();
    }
}

// Every block waits for the host's flag to reach the counter, then adds its part of the host's sum
// (count == 0: the host had nothing to add).
extern "C" __global__ void ts_moe_doorbell_wait_add(
    const unsigned int* flag,
    const unsigned int* seq,
    const int32_t* count,
    const float* hostOut,
    float* __restrict__ dst,
    const int n)
{
    __shared__ int add;
    if (threadIdx.x == 0)
    {
        const unsigned int want = *(const volatile unsigned int*)seq;
        while (*(const volatile unsigned int*)flag != want)
        {
        }
        __threadfence_system();
        add = *(const volatile int32_t*)count;
    }
    __syncthreads();
    if (!add)
        return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        dst[i] += ((const volatile float*)hostOut)[i];
}

// The doorbell for a layer whose selected experts may all have VRAM slots (Strata's device-planned groups): `missDev`
// (device memory) records whether any slot's expert lives on the host. Without one, the host is not rung and the wait
// below returns at once, so an all-resident layer costs no round trip over the bus; with one, as ts_moe_doorbell_publish.
extern "C" __global__ void ts_moe_doorbell_publish_skip(
    const float* __restrict__ x,
    const int32_t* __restrict__ sel,
    const float* __restrict__ selW,
    const uint8_t* __restrict__ isHost,
    const int n,
    const int slots,
    const int layer,
    float* xOut,
    int32_t* idsOut,
    float* wOut,
    int32_t* hdr,
    unsigned int* seq,
    int32_t* missDev,
    unsigned int* counts,                // [layers, nExpert] selections (adaptive residency), or null
    const int nExpert)
{
    int miss = 0;
    for (int i = threadIdx.x; i < slots; i += blockDim.x)
    {
        const int e = sel[i];
        if (e >= 0 && isHost[e])
            miss = 1;
        if (counts != nullptr && e >= 0)
            atomicAdd(counts + (size_t)layer * nExpert + e, 1u);
    }
    miss = __syncthreads_or(miss);
    if (threadIdx.x == 0)
        *missDev = miss;
    if (!miss)
        return;
    for (int i = threadIdx.x; i < slots; i += blockDim.x)
    {
        idsOut[i] = sel[i];
        wOut[i] = selW[i];
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        xOut[i] = x[i];
    if (threadIdx.x == 0)
    {
        ((volatile int32_t*)hdr)[0] = layer;
        ((volatile int32_t*)hdr)[1] = slots;
        ((volatile int32_t*)hdr)[2] = 1;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        *(volatile unsigned int*)seq = *(volatile unsigned int*)seq + 1u;
        __threadfence_system();
    }
}

// ts_moe_doorbell_publish_skip with Strata's route-tail skip (STRATA_ROUTE_TAIL_SKIP): a host expert that every row of
// the call routes at rank tailRank or lower (the row's selections are in descending order, slot i at rank i % nUsed) is
// not computed, so only a host expert some row routes above that rank rings the host. The host server applies the same
// rule to the slots of a layer that does ring (CudaHostExpertServer). tailRank <= 0 is ts_moe_doorbell_publish_skip.
extern "C" __global__ void ts_moe_doorbell_publish_tail(
    const float* __restrict__ x,
    const int32_t* __restrict__ sel,
    const float* __restrict__ selW,
    const uint8_t* __restrict__ isHost,
    const int n,
    const int slots,
    const int layer,
    float* xOut,
    int32_t* idsOut,
    float* wOut,
    int32_t* hdr,
    unsigned int* seq,
    int32_t* missDev,
    unsigned int* counts,                // [layers, nExpert] selections (adaptive residency), or null
    const int nExpert,
    const int nUsed,
    const int tailRank)
{
    int miss = 0;
    for (int i = threadIdx.x; i < slots; i += blockDim.x)
    {
        const int e = sel[i];
        if (e >= 0 && isHost[e] && (tailRank <= 0 || i % nUsed < tailRank))
            miss = 1;
        if (counts != nullptr && e >= 0)
            atomicAdd(counts + (size_t)layer * nExpert + e, 1u);
    }
    miss = __syncthreads_or(miss);
    if (threadIdx.x == 0)
        *missDev = miss;
    if (!miss)
        return;
    for (int i = threadIdx.x; i < slots; i += blockDim.x)
    {
        idsOut[i] = sel[i];
        wOut[i] = selW[i];
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        xOut[i] = x[i];
    if (threadIdx.x == 0)
    {
        ((volatile int32_t*)hdr)[0] = layer;
        ((volatile int32_t*)hdr)[1] = slots;
        ((volatile int32_t*)hdr)[2] = 1;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        *(volatile unsigned int*)seq = *(volatile unsigned int*)seq + 1u;
        __threadfence_system();
    }
}

// ts_moe_doorbell_wait_add after ts_moe_doorbell_publish_skip: nothing to wait for when the layer rang no doorbell.
extern "C" __global__ void ts_moe_doorbell_wait_add_skip(
    const unsigned int* flag,
    const unsigned int* seq,
    const int32_t* count,
    const float* hostOut,
    float* __restrict__ dst,
    const int n,
    const int32_t* __restrict__ missDev)
{
    if (*missDev == 0)
        return;
    __shared__ int add;
    if (threadIdx.x == 0)
    {
        const unsigned int want = *(const volatile unsigned int*)seq;
        while (*(const volatile unsigned int*)flag != want)
        {
        }
        __threadfence_system();
        add = *(const volatile int32_t*)count;
    }
    __syncthreads();
    if (!add)
        return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        dst[i] += ((const volatile float*)hostOut)[i];
}

extern "C" __global__ void ts_dsv4_moe_scatter_add_f32(
    const float* __restrict__ downOut,
    const int32_t* __restrict__ rowOfSlot, // [nt*nUsed] or null
    const float* __restrict__ wSel,        // [nt, nUsed]
    const float* __restrict__ shexp,       // [nt, E] or null
    float* __restrict__ ffnOut,            // [nt, E]
    const int nt,
    const int nUsed,
    const int E)
{
    const int t = blockIdx.y;
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt || e >= E)
        return;

    float acc = shexp != nullptr ? shexp[(size_t)t * E + e] : 0.0f;
    for (int j = 0; j < nUsed; ++j)
    {
        const int slot = t * nUsed + j;
        const int row = rowOfSlot != nullptr ? rowOfSlot[slot] : slot;
        acc += downOut[(size_t)row * E + e] * wSel[slot];
    }
    ffnOut[(size_t)t * E + e] = acc;
}

// ---------------------------------------------------------------------------
// DSpark speculative-decoding drafter
//
// The DSpark support module (DeepSeek's `mtp.*` stages) is three ordinary DSV4
// blocks with compress_ratio 0, driven off the target's hidden states instead
// of a token history of its own:
//
//   main_x       = main_norm(main_proj([h40; h41; h42]))     (per committed pos)
//   ring[pos]    = rope(kv_norm(wkv(main_x)))                (one row per pos)
//   block input  = embed([anchor, noise, noise, ...])        (block_size rows)
//   block attn   = softmax(q . [ring window | block kv]) with sinks, NON-causal
//                  inside the block (every block row sees the whole block)
//   logits(i)    = lm_head(norm(hc_head(x)))(i) + W2 . W1[prev(i)]
//   conf(i)      = sigmoid(proj . [x(i); W1[prev(i)]])
//
// Only the parts that differ from the target model live here: the block KV
// prep (rope position and ring slot are decoupled), and the Markov/confidence
// heads. The block-visible attention is mode 3 of ts_dsv4_attention_f32;
// everything else reuses the target's kernels.
// ---------------------------------------------------------------------------

// Per-row hyper-connection stream mean: dst[t, dstOff + d] = mean_c xs[t, c, d].
// Concatenating the three target layers into one [nt, 3*E] row is what the
// drafter's main_proj consumes.
extern "C" __global__ void ts_dsv4_hc_mean_f32(
    const float* __restrict__ xs,   // [nt, HC, E]
    float* __restrict__ dst,        // [nt, dstStride]
    const int nt,
    const int E,
    const int HC,
    const int dstStride,
    const int dstOff)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)nt * E;
    if (i >= total)
        return;
    const int t = (int)(i / E);
    const int d = (int)(i % E);

    const float* src = xs + (size_t)t * HC * E + d;
    float acc = 0.0f;
    for (int c = 0; c < HC; ++c)
        acc += src[(size_t)c * E];
    dst[(size_t)t * dstStride + dstOff + d] = acc / (float)HC;
}

// RMS-norm + RoPE for the drafter, with the rope position and the destination
// slot decoupled: the block's own KV goes to a dense [block, HD] buffer while
// the committed rows go to the SWA ring at (slot0 + t) % slotMod.
//
// grid.x == NH + 1 runs the q heads (h < NH, in place) and the single KV head
// (h == NH); grid.x == 1 with kvOnly runs the KV head alone (the catch-up pass
// has no queries).
extern "C" __global__ void ts_dsv4_dspark_prep_f32(
    float* __restrict__ q,               // [nt, NH, HD] in-place (unused when kvOnly)
    const float* __restrict__ kvRaw,     // [nt, HD]
    const float* __restrict__ kvNormW,   // [HD]
    const float* __restrict__ ropeTab,   // [nCtx, nRot]
    half* __restrict__ kvOut,            // [slotMod, HD]
    const int pos0,                      // rope position of row 0
    const int slot0,                     // kvOut slot of row 0
    const int slotMod,
    const int NH,
    const int HD,
    const int nRot,
    const float eps,
    const int kvOnly)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const bool isKv = kvOnly != 0 || h == NH;

    __shared__ float sh[512];
    __shared__ float red[256];

    float* qDst = q + ((size_t)t * NH + h) * HD;
    const float* src = isKv ? kvRaw + (size_t)t * HD : qDst;

    float acc = 0.0f;
    for (int d = threadIdx.x; d < HD; d += blockDim.x)
    {
        const float v = src[d];
        sh[d] = v;
        acc += v * v;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / HD + eps);

    for (int d = threadIdx.x; d < HD; d += blockDim.x)
        sh[d] *= isKv ? inv * kvNormW[d] : inv;
    __syncthreads();

    const float* tab = ropeTab + (long long)(pos0 + t) * nRot;
    const int rbase = HD - nRot;

    if (isKv)
    {
        half* out = kvOut + (size_t)((slot0 + t) % slotMod) * HD;
        for (int d = threadIdx.x; d < HD; d += blockDim.x)
        {
            float v = sh[d];
            if (d >= rbase)
            {
                const int i = (d - rbase) >> 1;
                const float c = tab[2 * i + 0];
                const float s = tab[2 * i + 1];
                const float x0 = sh[rbase + 2 * i + 0];
                const float x1 = sh[rbase + 2 * i + 1];
                v = ((d - rbase) & 1) == 0 ? x0 * c - x1 * s : x0 * s + x1 * c;
            }
            out[d] = __float2half(v);
        }
    }
    else
    {
        for (int d = threadIdx.x; d < HD; d += blockDim.x)
        {
            float v = sh[d];
            if (d >= rbase)
            {
                const int i = (d - rbase) >> 1;
                const float c = tab[2 * i + 0];
                const float s = tab[2 * i + 1];
                const float x0 = sh[rbase + 2 * i + 0];
                const float x1 = sh[rbase + 2 * i + 1];
                v = ((d - rbase) & 1) == 0 ? x0 * c - x1 * s : x0 * s + x1 * c;
            }
            qDst[d] = v;
        }
    }
}

// dst[0..R) = w1[tok * R ..], where tok is toks[tokSlot] (tokSlot < 0 selects
// the anchor token, which the host passes in directly).
extern "C" __global__ void ts_dsv4_dspark_gather_f32(
    const float* __restrict__ w1,   // [vocab, R]
    const int32_t* __restrict__ toks,
    const int tokSlot,
    const int anchorTok,
    float* __restrict__ dst,        // [R]
    const int R)
{
    const int tok = tokSlot < 0 ? anchorTok : toks[tokSlot];
    for (int i = threadIdx.x; i < R; i += blockDim.x)
        dst[i] = w1[(size_t)tok * R + i];
}

// toks[slot] = argmax_i (logits[i] + bias[i]). One block; ties resolve to the
// lowest index, matching the host-side argmax the verifier uses.
extern "C" __global__ void ts_dsv4_dspark_argmax_f32(
    const float* __restrict__ logits, // [V]
    const float* __restrict__ bias,   // [V] (may be null)
    int32_t* __restrict__ toks,
    const int slot,
    const int V)
{
    __shared__ float shV[256];
    __shared__ int shI[256];

    float best = -INFINITY;
    int bestIdx = 0;
    for (int i = threadIdx.x; i < V; i += blockDim.x)
    {
        const float v = bias != nullptr ? logits[i] + bias[i] : logits[i];
        if (v > best)
        {
            best = v;
            bestIdx = i;
        }
    }
    shV[threadIdx.x] = best;
    shI[threadIdx.x] = bestIdx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            const float o = shV[threadIdx.x + s];
            const int oi = shI[threadIdx.x + s];
            if (o > shV[threadIdx.x] || (o == shV[threadIdx.x] && oi < shI[threadIdx.x]))
            {
                shV[threadIdx.x] = o;
                shI[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        toks[slot] = shI[0];
}

// conf[t] = sigmoid(proj . [x[t]; w1Rows[t]]) -- the drafter's predicted
// acceptance probability for block position t.
extern "C" __global__ void ts_dsv4_dspark_conf_f32(
    const float* __restrict__ x,       // [B, E]
    const float* __restrict__ w1Rows,  // [B, R]
    const float* __restrict__ proj,    // [E + R]
    float* __restrict__ conf,          // [B]
    const int E,
    const int R)
{
    const int t = blockIdx.x;
    __shared__ float red[256];

    float acc = 0.0f;
    for (int i = threadIdx.x; i < E; i += blockDim.x)
        acc += x[(size_t)t * E + i] * proj[i];
    for (int i = threadIdx.x; i < R; i += blockDim.x)
        acc += w1Rows[(size_t)t * R + i] * proj[E + i];

    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        conf[t] = 1.0f / (1.0f + expf(-red[0]));
}

// ===========================================================================
// DeepSeek V4.1
//
// V4.1 keeps V4's LoRA-factored Q, shared K(=V) head, sinks and grouped output
// projection, and changes: no per-head query norm; compression ratios 1 and 2
// with a non-overlapping window and no absolute positional embedding; the
// indexer's K comes from the compressed latent; one layer per ratio group owns
// the caches and the sparse selection; and every cache commit reproduces the
// checkpoint's trained quantization.
//
// The quantization below is written out here rather than shared with any other
// backend, because this backend must stand alone. It has to agree to the bit
// with the reference, because the bins decide which compressed rows the sparse
// selection keeps -- not just the last digit of a value.
// ===========================================================================

// Round to BF16 and back, leaving NaN/Inf alone.
__device__ __forceinline__ float ts_dsv41_bf16(float v)
{
    unsigned int bits = __float_as_uint(v);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    return __uint_as_float(bits);
}

// Round half to even, for a non-negative value.
__device__ __forceinline__ float ts_dsv41_round_even(float v)
{
    const float base = floorf(v);
    const float frac = v - base;
    const bool up = (frac > 0.5f) || (frac == 0.5f && (((int)base) & 1) != 0);
    return base + (up ? 1.0f : 0.0f);
}

// Finite E4M3 (FP8), round-to-nearest-even, saturating at 448. frexp's exponent
// is ilogb + 1; the step 2^(exponent-4) leaves four significant bits and
// reproduces the format's subnormal step below 2^-6.
__device__ __forceinline__ float ts_dsv41_e4m3(float v)
{
    const float mag = fminf(fabsf(v), 448.0f);
    const int e = ilogbf(fmaxf(mag, 0.015625f)) + 1;
    const float step = ldexpf(1.0f, e - 4);
    return copysignf(ts_dsv41_round_even(__fdiv_rn(mag, step)) * step, v);
}

// E2M1 (FP4). Magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6; the inclusive boundaries
// alternate so a midpoint tie selects an even code.
__device__ __forceinline__ float ts_dsv41_e2m1(float v)
{
    const float m = fabsf(v);
    const float r = m <= 0.25f ? 0.0f : m < 0.75f ? 0.5f
                  : m <= 1.25f ? 1.0f : m < 1.75f ? 1.5f
                  : m <= 2.5f  ? 2.0f : m < 3.5f  ? 3.0f
                  : m <= 5.0f  ? 4.0f : 6.0f;
    return copysignf(r, v);
}

__device__ __forceinline__ float ts_dsv41_ceil_pow2(float v)
{
    const unsigned int bits = __float_as_uint(v);
    const int e = (int)((bits >> 23) & 255u) - 127 + ((bits & 0x7fffffu) != 0 ? 1 : 0);
    return ldexpf(1.0f, e);
}

// mode 0 = FP8 E4M3 (raw rows), 1 = MXFP4 (indexer), 2 = NVFP4 (compressed).
__device__ __forceinline__ float ts_dsv41_quant_scale(float amax, int mode)
{
    if (mode == 0)
        return ts_dsv41_ceil_pow2(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    if (mode == 1)
        return ts_dsv41_ceil_pow2(fmaxf(amax, 6.0f * 1.17549435e-38f) * (1.0f / 6.0f));
    return ts_dsv41_e4m3(__fdiv_rn(fmaxf(amax, 6.0f * 0.001953125f), 6.0f));
}

__device__ __forceinline__ int ts_dsv41_quant_block(int mode) { return mode == 2 ? 16 : 32; }

// Quantize a contiguous row already staged in shared memory. Divide, never
// multiply by a reciprocal: the NVFP4 scale is an E4M3 value rather than a
// power of two, and the reciprocal's rounding moves exact FP4 midpoints across
// a bin. One thread owns a whole group, which keeps the block amax private.
__device__ __forceinline__ void ts_dsv41_quant_row(float* row, int n, int mode, int tid, int nthreads)
{
    const int len = ts_dsv41_quant_block(mode);
    for (int g = tid; g < n / len; g += nthreads)
    {
        float* p = row + (size_t)g * len;
        float amax = 0.0f;
        for (int i = 0; i < len; ++i)
        {
            const float v = ts_dsv41_bf16(p[i]);
            p[i] = v;
            amax = fmaxf(amax, fabsf(v));
        }
        const float scale = ts_dsv41_quant_scale(amax, mode);
        for (int i = 0; i < len; ++i)
        {
            const float q = mode == 0 ? ts_dsv41_e4m3(__fdiv_rn(p[i], scale))
                                      : ts_dsv41_e2m1(__fdiv_rn(p[i], scale));
            p[i] = ts_dsv41_bf16(q * scale);
        }
    }
}

// Rotate the last nRot components of a staged row in place (interleaved pairs).
__device__ __forceinline__ void ts_dsv41_rope_row(float* row, const float* tab, int dim, int nRot, int tid, int nthreads)
{
    const int rbase = dim - nRot;
    for (int i = tid; i < nRot / 2; i += nthreads)
    {
        const float c = tab[2 * i + 0], s = tab[2 * i + 1];
        const float x0 = row[rbase + 2 * i + 0], x1 = row[rbase + 2 * i + 1];
        row[rbase + 2 * i + 0] = x0 * c - x1 * s;
        row[rbase + 2 * i + 1] = x0 * s + x1 * c;
    }
}

// ---------------------------------------------------------------------------
// attention preparation
//
// Same shape as the V4 kernel with two changes: the projected query heads are
// NOT re-normalized (their leading head_dim - n_rot components are MLA content
// dimensions, handed to attention as the projection produced them), and the
// shared K row is quantized the way the trained cache is before it is stored.
// ---------------------------------------------------------------------------
extern "C" __global__ void ts_dsv41_attn_prep_f32(
    float* __restrict__ q,               // [nt, NH, HD] in-place
    const float* __restrict__ kvRaw,     // [nt, HD]
    const float* __restrict__ kvNormW,   // [HD]
    const float* __restrict__ ropeTab,   // [nCtx, nRot]
    half* __restrict__ ring,             // [ringRows, HD]
    const int p0,
    const int ringRows,
    const int NH,
    const int HD,
    const int nRot,
    const float eps)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const bool isKv = h == NH;

    extern __shared__ float sh41[];
    __shared__ float red[256];

    float* qDst = q + ((size_t)t * NH + h) * HD;
    const float* src = isKv ? kvRaw + (size_t)t * HD : qDst;

    for (int d = threadIdx.x; d < HD; d += blockDim.x)
        sh41[d] = src[d];
    __syncthreads();

    if (isKv)
    {
        float acc = 0.0f;
        for (int d = threadIdx.x; d < HD; d += blockDim.x)
            acc += sh41[d] * sh41[d];
        red[threadIdx.x] = acc;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (threadIdx.x < s)
                red[threadIdx.x] += red[threadIdx.x + s];
            __syncthreads();
        }
        const float inv = rsqrtf(red[0] / HD + eps);
        for (int d = threadIdx.x; d < HD; d += blockDim.x)
            sh41[d] *= inv * kvNormW[d];
        __syncthreads();
    }

    const long long p = (long long)p0 + t;
    ts_dsv41_rope_row(sh41, ropeTab + p * nRot, HD, nRot, threadIdx.x, blockDim.x);
    __syncthreads();

    if (isKv)
    {
        ts_dsv41_quant_row(sh41, HD, 0, threadIdx.x, blockDim.x);
        __syncthreads();
        half* out = ring + (size_t)(p % ringRows) * HD;
        float* back = (float*) kvRaw + (size_t)t * HD;   // scratch, already consumed
        for (int d = threadIdx.x; d < HD; d += blockDim.x)
        {
            out[d] = __float2half(sh41[d]);
            // Write the committed value back so a trace of kvRaw shows what went
            // into the cache rather than the projection that preceded it.
            back[d] = sh41[d];
        }
    }
    else
    {
        for (int d = threadIdx.x; d < HD; d += blockDim.x)
            qDst[d] = sh41[d];
    }
}

// ---------------------------------------------------------------------------
// block compressor
//
// One block per compressed row. Ratio 1 compresses nothing: the latent is the
// normalized projection of that one token. Ratio 2 reduces each non-overlapping
// pair with a per-dimension softmax over the pair's gate projection. A window
// position before this ubatch is read from the state ring.
//
// The latent is left unrotated and unquantized: the indexer's K projection
// consumes this version, and ts_dsv41_commit_f32 finishes both caches after.
// ---------------------------------------------------------------------------
extern "C" __global__ void ts_dsv41_compress_f32(
    const float* __restrict__ stKv,      // [nt, HD]
    const float* __restrict__ stScore,   // [nt, HD]   (unused when ratio == 1)
    const float* __restrict__ histKv,    // [stateSize, HD]
    const float* __restrict__ histScore, // [stateSize, HD]
    const float* __restrict__ normW,     // [HD]
    float* __restrict__ latent,          // [nBlocks, HD]
    const long long firstBoundary,
    const int p0,
    const int ratio,
    const int HD,
    const float eps,
    const int stateSize)                 // ratio + the drafter's block: see ts_dsv41_persist_f32
{
    const int bi = blockIdx.x;
    const long long p = firstBoundary + (long long)bi * ratio;
    const long long start = p + 1 - ratio;

    extern __shared__ float shc[];
    __shared__ float red[256];

    for (int d = threadIdx.x; d < HD; d += blockDim.x)
    {
        float v;
        if (ratio == 1)
        {
            v = stKv[(size_t)(start - p0) * HD + d];
        }
        else
        {
            float m = -INFINITY;
            for (int w = 0; w < ratio; ++w)
            {
                const long long tw = start + w;
                const float sc = tw >= p0 ? stScore[(size_t)(tw - p0) * HD + d]
                                          : histScore[(size_t)(tw % stateSize) * HD + d];
                m = fmaxf(m, sc);
            }
            float se = 0.0f, sv = 0.0f;
            for (int w = 0; w < ratio; ++w)
            {
                const long long tw = start + w;
                const size_t off = (size_t)(tw >= p0 ? (tw - p0) : (tw % stateSize)) * HD + d;
                const float sc = tw >= p0 ? stScore[off] : histScore[off];
                const float kv = tw >= p0 ? stKv[off] : histKv[off];
                const float e = __expf(sc - m);
                se += e;
                sv += e * kv;
            }
            v = sv / se;
        }
        shc[d] = v;
    }
    __syncthreads();

    float acc = 0.0f;
    for (int d = threadIdx.x; d < HD; d += blockDim.x)
        acc += shc[d] * shc[d];
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / HD + eps);

    float* out = latent + (size_t)bi * HD;
    for (int d = threadIdx.x; d < HD; d += blockDim.x)
        out[d] = shc[d] * inv * normW[d];
}

// Rotate a produced row at its block-start position, quantize it, and store it
// F16 into the shared cache. Serves both the compressed cache (mode 2) and the
// indexer cache (mode 1); `dim` and `mode` are what differ between them.
extern "C" __global__ void ts_dsv41_commit_f32(
    const float* __restrict__ rows,      // [nBlocks, dim]
    const float* __restrict__ ropeTab,   // [nCtx, nRot]
    half* __restrict__ cache,            // [cacheRows, dim]
    const long long firstBoundary,
    const int ratio,
    const int dim,
    const int nRot,
    const int mode)
{
    const int bi = blockIdx.x;
    const long long p = firstBoundary + (long long)bi * ratio;
    const long long start = p + 1 - ratio;
    const long long row = p / ratio;

    extern __shared__ float shk[];
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        shk[d] = rows[(size_t)bi * dim + d];
    __syncthreads();

    ts_dsv41_rope_row(shk, ropeTab + start * nRot, dim, nRot, threadIdx.x, blockDim.x);
    __syncthreads();
    ts_dsv41_quant_row(shk, dim, mode, threadIdx.x, blockDim.x);
    __syncthreads();

    half* out = cache + (size_t)row * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x)
        out[d] = __float2half(shk[d]);
}

// Persist the last `ratio` token projections so a block straddling the next
// ubatch boundary still sees its earlier half.
// The state ring keeps the last stateSize positions at pos % stateSize. A
// speculative verify persists every row it forwarded, rejected ones included,
// so the ring is wider than the ratio by the drafter's block: a rejected tail
// then lands on rows no accepted position still needs, and a rewind has
// nothing to restore (the native executor's state_extra).
extern "C" __global__ void ts_dsv41_persist_f32(
    const float* __restrict__ stKv,
    const float* __restrict__ stScore,
    float* __restrict__ histKv,
    float* __restrict__ histScore,
    const int p0,
    const int nt,
    const int stateSize,
    const int HD)
{
    const int i = blockIdx.y;                 // 0..min(nt, stateSize)-1, newest last
    const int t = nt - min(nt, stateSize) + i;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= HD)
        return;
    const int slot = (int)(((long long)p0 + t) % stateSize);
    histKv[(size_t)slot * HD + d] = stKv[(size_t)t * HD + d];
    histScore[(size_t)slot * HD + d] = stScore[(size_t)t * HD + d];
}

// Rotate and quantize the indexer's query heads, and prescale its weights.
extern "C" __global__ void ts_dsv41_idx_prep_f32(
    float* __restrict__ iq,              // [nt, IH, ID] in-place
    float* __restrict__ iw,              // [nt, IH] in-place
    const float* __restrict__ ropeTab,
    const int p0,
    const int IH,
    const int ID,
    const int nRot,
    const float iwScale)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;

    extern __shared__ float shq[];
    float* dst = iq + ((size_t)t * IH + h) * ID;
    for (int d = threadIdx.x; d < ID; d += blockDim.x)
        shq[d] = dst[d];
    __syncthreads();

    ts_dsv41_rope_row(shq, ropeTab + ((long long)p0 + t) * nRot, ID, nRot, threadIdx.x, blockDim.x);
    __syncthreads();
    ts_dsv41_quant_row(shq, ID, 1, threadIdx.x, blockDim.x);
    __syncthreads();

    for (int d = threadIdx.x; d < ID; d += blockDim.x)
        dst[d] = shq[d];
    if (h == 0 && threadIdx.x == 0)
    {
        float* w = iw + (size_t)t * IH;
        for (int i = 0; i < IH; ++i)
            w[i] *= iwScale;
    }
}

// sum_h relu(q_h . k) * w_h over every visible compressed row, with rows the
// candidate layer pruned scored -inf so they rank last.
extern "C" __global__ void ts_dsv41_idx_scores_f32(
    const float* __restrict__ iq,
    const float* __restrict__ iw,
    const half* __restrict__ lidK,
    const unsigned char* __restrict__ candidates,   // [nt, rows] or null
    float* __restrict__ scores,                     // [nt, rows]
    const int p0,
    const int ratio,
    const int IH,
    const int ID,
    const int rows,
    const int maxVis)
{
    const int r = blockIdx.x * blockDim.y + threadIdx.y;
    const int t = blockIdx.y;
    if (r >= maxVis)
        return;
    const int nVis = (int)((((long long)p0 + t) + 1) / ratio);
    float* out = scores + (size_t)t * rows;
    if (r >= nVis)
        return;
    if (candidates && candidates[(size_t)t * rows + r] == 0)
    {
        if (threadIdx.x == 0)
            out[r] = -INFINITY;
        return;
    }

    const float* q = iq + (size_t)t * IH * ID;
    const float* w = iw + (size_t)t * IH;
    const half* k = lidK + (size_t)r * ID;

    float score = 0.0f;
    for (int h = 0; h < IH; ++h)
    {
        float acc = 0.0f;
        for (int d = threadIdx.x; d < ID; d += warpSize)
            acc += q[(size_t)h * ID + d] * __half2float(k[d]);
        acc = ts_dsv4_warp_sum(acc);
        if (acc > 0.0f)
            score += acc * w[h];
    }
    if (threadIdx.x == 0)
        out[r] = score;
}

// Pool the indexer scores into fixed row blocks, pin the block holding the
// query's own newest row, keep the best few, and publish the surviving rows as
// the mask every later indexer layer intersects with visibility.
extern "C" __global__ void ts_dsv41_candidate_f32(
    const float* __restrict__ scores,     // [nt, rows]
    unsigned char* __restrict__ mask,     // [nt, rows]
    const int p0,
    const int ratio,
    const int rows,
    const int blockLen,
    const int topk)
{
    const int t = blockIdx.x;
    const long long p = (long long)p0 + t;
    const int nVis = (int)((p + 1) / ratio);
    const int nBlocks = (nVis + blockLen - 1) / blockLen;
    const float* s = scores + (size_t)t * rows;
    unsigned char* m = mask + (size_t)t * rows;

    for (int r = threadIdx.x; r < nVis; r += blockDim.x)
        m[r] = 0;
    __syncthreads();

    if (nBlocks <= 0)
        return;

    // Rank blocks by pooled max, ties by lowest block id, with the pinned block
    // forced first. nBlocks can be large, so each thread ranks its own blocks by
    // counting how many beat them rather than materializing a sort.
    const long long pinned = p / blockLen;
    for (int b = threadIdx.x; b < nBlocks; b += blockDim.x)
    {
        float best = -INFINITY;
        const int end = min(nVis, (b + 1) * blockLen);
        for (int r = b * blockLen; r < end; ++r)
            best = fmaxf(best, s[r]);
        if (b == (int)pinned)
            best = INFINITY;
        if (best == -INFINITY)
            continue;

        int rank = 0;
        for (int o = 0; o < nBlocks && rank < topk; ++o)
        {
            if (o == b)
                continue;
            float other = -INFINITY;
            const int oend = min(nVis, (o + 1) * blockLen);
            for (int r = o * blockLen; r < oend; ++r)
                other = fmaxf(other, s[r]);
            if (o == (int)pinned)
                other = INFINITY;
            if (other > best || (other == best && o < b))
                ++rank;
        }
        if (rank >= topk)
            continue;
        for (int r = b * blockLen; r < end; ++r)
            m[r] = 1;
    }
}

// ---------------------------------------------------------------------------
// Engram
//
// The gathered rows are projected into one key per hyper-connection stream plus
// a single shared value; the key is scored against the residual and the value
// added back through a signed square-root sigmoid gate.
// ---------------------------------------------------------------------------
extern "C" __global__ void ts_dsv41_engram_gate_f32(
    float* __restrict__ xs,              // [nt, HC, E] in-place residual
    const float* __restrict__ kv,        // [nt, (HC+1)*E]
    const float* __restrict__ qw,        // [HC, E]
    const float* __restrict__ kw,        // [HC, E]
    const int E,
    const float eps)
{
    const int st = blockIdx.x;
    const int t = blockIdx.y;
    const int HC41 = gridDim.x;

    __shared__ float red[256];
    float* x = xs + ((size_t)t * HC41 + st) * E;
    const float* key = kv + (size_t)t * (HC41 + 1) * E + (size_t)st * E;
    const float* value = kv + (size_t)t * (HC41 + 1) * E + (size_t)HC41 * E;

    float sx = 0.0f, sk = 0.0f;
    for (int d = threadIdx.x; d < E; d += blockDim.x)
    {
        sx += x[d] * x[d];
        sk += key[d] * key[d];
    }
    red[threadIdx.x] = sx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float invX = rsqrtf(red[0] / E + eps);
    __syncthreads();
    red[threadIdx.x] = sk;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float invK = rsqrtf(red[0] / E + eps);
    __syncthreads();

    float dot = 0.0f;
    for (int d = threadIdx.x; d < E; d += blockDim.x)
        dot += (x[d] * invX * qw[(size_t)st * E + d]) * (key[d] * invK * kw[(size_t)st * E + d]);
    red[threadIdx.x] = dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    const float scaled = red[0] * rsqrtf((float)E);
    // Signed square root, floored away from zero, then a sigmoid.
    const float magnitude = sqrtf(fmaxf(fabsf(scaled), 1e-6f));
    const float gate = 1.0f / (1.0f + __expf(-(scaled >= 0.0f ? magnitude : -magnitude)));

    for (int d = threadIdx.x; d < E; d += blockDim.x)
        x[d] += gate * value[d];
}
