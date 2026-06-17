#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <mma.h>   // nvcuda::wmma int8 tensor-core MMA (sm_72+ / compute_86)

// IQ2_XXS dequant lookup tables, vendored from ggml-org/ggml so the CUDA backend
// builds without the upstream ggml checkout (see tensorsharp_iq2xxs_tables.cuh).
#include "tensorsharp_iq2xxs_tables.cuh"

#define GGML_Q4_0 2
#define GGML_Q4_1 3
#define GGML_Q5_0 6
#define GGML_Q5_1 7
#define GGML_Q8_0 8
#define GGML_Q8_1 9
#define GGML_Q2_K 10
#define GGML_Q3_K 11
#define GGML_Q4_K 12
#define GGML_Q5_K 13
#define GGML_Q6_K 14
#define GGML_IQ2_XXS 16
#define GGML_IQ3_XXS 18
#define GGML_IQ3_S 21
#define GGML_IQ2_S 22
#define TS_QK8_1 32

struct ts_block_q8_1
{
    half d;
    half s;
    int8_t qs[TS_QK8_1];
};

struct ts_block_iq2_xxs
{
    half d;
    uint16_t qs[32];
};

__device__ __forceinline__ unsigned int read_u32_unaligned(const uint8_t* p)
{
    return (unsigned int)p[0] | ((unsigned int)p[1] << 8) | ((unsigned int)p[2] << 16) | ((unsigned int)p[3] << 24);
}

__device__ __forceinline__ int get_int_b2(const void* x, int i32)
{
    const uint16_t* x16 = reinterpret_cast<const uint16_t*>(x);
    return (int)x16[2 * i32] | ((int)x16[2 * i32 + 1] << 16);
}

__device__ __forceinline__ int get_int_b4(const void* x, int i32)
{
    return reinterpret_cast<const int*>(x)[i32];
}

__device__ __forceinline__ int dp4a_i8(int a, int b, int c)
{
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const int8_t* a8 = reinterpret_cast<const int8_t*>(&a);
    const int8_t* b8 = reinterpret_cast<const int8_t*>(&b);
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
#endif
}

__device__ __forceinline__ int qrow_bytes(int type, int cols)
{
    switch (type)
    {
        case GGML_Q4_0: return (cols / 32) * 18;
        case GGML_Q4_1: return (cols / 32) * 20;
        case GGML_Q5_0: return (cols / 32) * 22;
        case GGML_Q5_1: return (cols / 32) * 24;
        case GGML_Q8_0: return (cols / 32) * 34;
        case GGML_Q8_1: return (cols / 32) * 36;
        case GGML_Q2_K: return (cols / 256) * 84;
        case GGML_Q3_K: return (cols / 256) * 110;
        case GGML_Q4_K: return (cols / 256) * 144;
        case GGML_Q5_K: return (cols / 256) * 176;
        case GGML_Q6_K: return (cols / 256) * 210;
        case GGML_IQ2_XXS: return (cols / 256) * 66;
        case GGML_IQ3_XXS: return (cols / 256) * 98;
        case GGML_IQ2_S: return (cols / 256) * 82;
        case GGML_IQ3_S: return (cols / 256) * 110;
        default: return 0;
    }
}

__device__ __forceinline__ int get_scale_min_k4(const uint8_t* s, int index)
{
    if (index < 4)
        return s[index] & 0x3F;
    return (s[index + 4] & 0x0F) | ((s[index - 4] >> 6) << 4);
}

__device__ __forceinline__ int get_min_k4(const uint8_t* s, int index)
{
    if (index < 4)
        return s[index + 4] & 0x3F;
    return (s[index + 4] >> 4) | ((s[index] >> 6) << 4);
}

__device__ __forceinline__ float qvalue_at(const uint8_t* row, int type, int col)
{
    if (type == GGML_Q4_0)
    {
        const uint8_t* block = row + (col / 32) * 18;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        int lane = col & 31;
        uint8_t packed = block[2 + (lane & 15)];
        int v = lane < 16 ? (packed & 0x0F) - 8 : (packed >> 4) - 8;
        return d * (float)v;
    }

    if (type == GGML_Q4_1)
    {
        const uint8_t* block = row + (col / 32) * 20;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float m = __half2float(*reinterpret_cast<const half*>(block + 2));
        int lane = col & 31;
        uint8_t packed = block[4 + (lane & 15)];
        int v = lane < 16 ? (packed & 0x0F) : (packed >> 4);
        return d * (float)v + m;
    }

    if (type == GGML_Q5_0)
    {
        const uint8_t* block = row + (col / 32) * 22;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        unsigned int qh = read_u32_unaligned(block + 2);
        int lane = col & 31;
        uint8_t packed = block[6 + (lane & 15)];
        int qhbit = (qh >> lane) & 1;
        int v = (lane < 16 ? (packed & 0x0F) : (packed >> 4)) | (qhbit << 4);
        return d * (float)(v - 16);
    }

    if (type == GGML_Q5_1)
    {
        const uint8_t* block = row + (col / 32) * 24;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float m = __half2float(*reinterpret_cast<const half*>(block + 2));
        unsigned int qh = read_u32_unaligned(block + 4);
        int lane = col & 31;
        uint8_t packed = block[8 + (lane & 15)];
        int qhbit = (qh >> lane) & 1;
        int v = (lane < 16 ? (packed & 0x0F) : (packed >> 4)) | (qhbit << 4);
        return d * (float)v + m;
    }

    if (type == GGML_Q8_0)
    {
        const uint8_t* block = row + (col / 32) * 34;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
        return d * (float)qs[col & 31];
    }

    if (type == GGML_Q8_1)
    {
        const uint8_t* block = row + (col / 32) * 36;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 4);
        return d * (float)qs[col & 31];
    }

    if (type == GGML_IQ2_XXS)
    {
        const uint8_t* block = row + (col / 256) * 66;
        int t = col & 255;
        int ib32 = t / 32;
        int l = (t & 31) / 8;
        int j = t & 7;

        float d = __half2float(*reinterpret_cast<const half*>(block));
        const uint8_t* qs = block + 2;
        const uint8_t grid_index = qs[ib32 * 8 + l];
        uint32_t signscale = read_u32_unaligned(qs + ib32 * 8 + 4);
        float db = d * (0.5f + (float)(signscale >> 28)) * 0.25f;

        uint32_t sign7 = (signscale >> (7 * l)) & 0x7F;
        uint32_t sign8 = sign7 | ((__popc(sign7) & 1) << 7);
        uint64_t grid = iq2xxs_grid[grid_index];
        int v = (int)((grid >> (8 * j)) & 0xFF);
        return (sign8 & (1u << j)) ? -db * (float)v : db * (float)v;
    }

    if (type == GGML_Q4_K)
    {
        const uint8_t* block = row + (col / 256) * 144;
        int t = col & 255;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales = block + 4;
        const uint8_t* qs = block + 16;
        int pair = t / 64;
        int pos = t & 63;
        int is_odd = pos / 32;
        int j = pos & 31;
        int sub = pair * 2 + is_odd;
        int sc = get_scale_min_k4(scales, sub);
        int m = get_min_k4(scales, sub);
        uint8_t packed = qs[pair * 32 + j];
        int v = is_odd ? (packed >> 4) : (packed & 0x0F);
        return d * (float)sc * (float)v - dmin * (float)m;
    }

    if (type == GGML_Q5_K)
    {
        const uint8_t* block = row + (col / 256) * 176;
        int t = col & 255;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales = block + 4;
        const uint8_t* qh = block + 16;
        const uint8_t* qs = block + 48;
        int sub = t / 32;
        int pos = t & 31;
        int pair = sub / 2;
        int sc = get_scale_min_k4(scales, sub);
        int m = get_min_k4(scales, sub);
        uint8_t packed = qs[pair * 32 + pos];
        int bit = (qh[pos] >> sub) & 1;
        int v = ((sub & 1) ? (packed >> 4) : (packed & 0x0F)) | (bit << 4);
        return d * (float)sc * (float)v - dmin * (float)m;
    }

    if (type == GGML_Q6_K)
    {
        const uint8_t* block = row + (col / 256) * 210;
        int t = col & 255;
        const uint8_t* ql = block;
        const uint8_t* qh = block + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
        float d = __half2float(*reinterpret_cast<const half*>(block + 208));
        int half_idx = t / 128;
        int pos = t & 127;
        const uint8_t* ql_half = ql + half_idx * 64;
        const uint8_t* qh_half = qh + half_idx * 32;
        const int8_t* sc_half = scales + half_idx * 8;
        int group = pos / 32;
        int l = pos & 31;
        int q;
        if (group == 0)
            q = ((ql_half[l] & 0x0F) | (((qh_half[l] >> 0) & 3) << 4)) - 32;
        else if (group == 1)
            q = ((ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 3) << 4)) - 32;
        else if (group == 2)
            q = ((ql_half[l] >> 4) | (((qh_half[l] >> 4) & 3) << 4)) - 32;
        else
            q = ((ql_half[l + 32] >> 4) | (((qh_half[l] >> 6) & 3) << 4)) - 32;
        int isc = l / 16;
        return d * (float)sc_half[isc + group * 2] * (float)q;
    }

    if (type == GGML_Q2_K)
    {
        // block_q2_K: scales[16], qs[64], d (half), dmin (half) = 84 bytes.
        const uint8_t* block = row + (col / 256) * 84;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        float d = __half2float(*reinterpret_cast<const half*>(block + 80));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 82));
        int t = col & 255;
        int group128 = t >> 7;       // 0 or 1 (128-element half)
        int within = t & 127;
        int j = within >> 5;         // 0..3 -> 2-bit shift = 2*j
        int pos = within & 31;       // 0..31 byte within the 32-byte chunk
        int half_sel = pos >> 4;     // 0 or 1 -> scale pair selector
        int sc_index = group128 * 8 + j * 2 + half_sel;
        uint8_t sc = scales[sc_index];
        float dl = d * (float)(sc & 0xF);
        float ml = dmin * (float)(sc >> 4);
        uint8_t q = qs[group128 * 32 + pos];
        int v = (q >> (2 * j)) & 3;
        return dl * (float)v - ml;
    }

    if (type == GGML_Q3_K)
    {
        // block_q3_K: hmask[32], qs[64], scales[12], d (half) = 110 bytes.
        const uint8_t* block = row + (col / 256) * 110;
        const uint8_t* hmask = block;
        const uint8_t* qs = block + 32;
        const uint8_t* scales = block + 96;
        float d_all = __half2float(*reinterpret_cast<const half*>(block + 108));
        int t = col & 255;
        int group128 = t >> 7;
        int within = t & 127;
        int j = within >> 5;         // 0..3
        int pos = within & 31;       // 0..31
        int half_sel = pos >> 4;
        int sc_index = group128 * 8 + j * 2 + half_sel;
        int global_j = group128 * 4 + j;
        int shift = 2 * j;

        // Unpack the 6-bit scale at sc_index from the 12-byte packed layout
        // (ggml dequantize_row_q3_K aux recombination, evaluated for one index).
        int aux_idx = sc_index >> 2;
        int b = sc_index & 3;
        int sc6;
        if (aux_idx == 0)
            sc6 = (scales[b] & 0x0F) | (((scales[8 + b] >> 0) & 3) << 4);
        else if (aux_idx == 1)
            sc6 = (scales[4 + b] & 0x0F) | (((scales[8 + b] >> 2) & 3) << 4);
        else if (aux_idx == 2)
            sc6 = ((scales[b] >> 4) & 0x0F) | (((scales[8 + b] >> 4) & 3) << 4);
        else
            sc6 = ((scales[4 + b] >> 4) & 0x0F) | (((scales[8 + b] >> 6) & 3) << 4);
        int sc = sc6 - 32;

        uint8_t q = qs[group128 * 32 + pos];
        int low2 = (q >> shift) & 3;
        int high = (hmask[pos] & (1 << global_j)) ? 0 : 4;
        return d_all * (float)sc * (float)(low2 - high);
    }

    if (type == GGML_IQ3_XXS)
    {
        // block_iq3_xxs: d (half), qs[96] = grid indices[64] + scales_and_signs[32].
        const uint8_t* block = row + (col / 256) * 98;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        const uint8_t* qs = block + 2;
        const uint8_t* sas = qs + 64;
        int t = col & 255;
        int ib32 = t >> 5;
        int within = t & 31;
        int l = within >> 3;         // 0..3
        int p = within & 7;          // 0..7 position in the 8-element group
        uint32_t aux32 = read_u32_unaligned(sas + 4 * ib32);
        float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        uint8_t grid_index = qs[8 * ib32 + 2 * l + (p >= 4 ? 1 : 0)];
        uint32_t grid = iq3xxs_grid[grid_index];
        int gv = (int)((grid >> (8 * (p & 3))) & 0xFF);
        uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
        float v = db * (float)gv;
        return (signs & (1u << p)) ? -v : v;
    }

    if (type == GGML_IQ2_S)
    {
        // block_iq2_s: d (half), qs[64], qh[8], scales[8] = 82 bytes.
        // qs[0..31] hold grid low bytes, qs[32..63] hold the per-group sign bytes.
        const uint8_t* block = row + (col / 256) * 82;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        const uint8_t* qs = block + 2;
        const uint8_t* qh = block + 66;
        const uint8_t* signs = qs + 32;
        const uint8_t* scales = block + 74;
        int t = col & 255;
        int ib32 = t >> 5;
        int within = t & 31;
        int l = within >> 3;     // 0..3
        int p = within & 7;      // 0..7
        int grid_index = qs[ib32 * 4 + l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
        uint64_t grid = iq2s_grid[grid_index];
        int gv = (int)((grid >> (8 * p)) & 0xFF);
        uint8_t sc = scales[ib32];
        float db = d * (0.5f + (float)((l < 2) ? (sc & 0xf) : (sc >> 4))) * 0.25f;
        uint8_t sign_byte = signs[ib32 * 4 + l];
        float v = db * (float)gv;
        return (sign_byte & (1u << p)) ? -v : v;
    }

    if (type == GGML_IQ3_S)
    {
        // block_iq3_s: d (half), qs[64], qh[8], signs[32], scales[4] = 110 bytes.
        const uint8_t* block = row + (col / 256) * 110;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        const uint8_t* qs = block + 2;
        const uint8_t* qh = block + 66;
        const uint8_t* signs = block + 74;
        const uint8_t* scales = block + 106;
        int t = col & 255;
        int ib32 = t >> 5;
        int within = t & 31;
        int l = within >> 3;     // 0..3
        int p = within & 7;      // 0..7 (pos 0..3 -> grid1, 4..7 -> grid2)
        uint8_t sc = scales[ib32 >> 1];
        float db = d * (float)(1 + 2 * ((ib32 & 1) ? (sc >> 4) : (sc & 0xf)));
        int qs_off = 8 * ib32;
        int grid_index = (p < 4)
            ? (qs[qs_off + 2 * l + 0] | ((qh[ib32] << (8 - 2 * l)) & 256))
            : (qs[qs_off + 2 * l + 1] | ((qh[ib32] << (7 - 2 * l)) & 256));
        uint32_t grid = iq3s_grid[grid_index];
        int gv = (int)((grid >> (8 * (p & 3))) & 0xFF);
        uint8_t sign_byte = signs[ib32 * 4 + l];
        float v = db * (float)gv;
        return (sign_byte & (1u << p)) ? -v : v;
    }

    return 0.0f;
}

__device__ __forceinline__ void quantize_q8_1_block(const float* x, ts_block_q8_1* dst)
{
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < TS_QK8_1; i++)
        amax = fmaxf(amax, fabsf(x[i]));

    float d = amax > 0.0f ? amax / 127.0f : 0.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    int sum = 0;
#pragma unroll
    for (int i = 0; i < TS_QK8_1; i++)
    {
        int q = (int)rintf(x[i] * id);
        q = max(-127, min(127, q));
        dst->qs[i] = (int8_t)q;
        sum += q;
    }

    dst->d = __float2half_rn(d);
    dst->s = __float2half_rn(d * (float)sum);
}

__device__ __forceinline__ float dot_iq2_xxs_q8_1(const uint8_t* iq_block, const ts_block_q8_1* q8_blocks, int group)
{
    const ts_block_iq2_xxs* bq2 = reinterpret_cast<const ts_block_iq2_xxs*>(iq_block);
    int iqs = group * 2;
    int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(&q2);
    uint32_t aux32 = (uint32_t)get_int_b2(bq2->qs, iqs + 1);

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2)
    {
        const int* grid_pos = reinterpret_cast<const int*>(iq2xxs_grid + aux8[k0 / 2]);
        int signs_packed = ksigns_iq2xs[(aux32 >> (7 * k0 / 2)) & 0x7F];

        int signs0 = __vcmpne4(((signs_packed & 0x03) << 7) | ((signs_packed & 0x0C) << 21), 0x00000000);
        int grid0 = __vsub4(grid_pos[0] ^ signs0, signs0);
        int u0 = get_int_b4(q8_blocks[group].qs, k0 + 0);
        sumi = dp4a_i8(grid0, u0, sumi);

        int signs1 = __vcmpne4(((signs_packed & 0x30) << 3) | ((signs_packed & 0xC0) << 17), 0x00000000);
        int grid1 = __vsub4(grid_pos[1] ^ signs1, signs1);
        int u1 = get_int_b4(q8_blocks[group].qs, k0 + 1);
        sumi = dp4a_i8(grid1, u1, sumi);
    }

    int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 4;
    float d = __half2float(bq2->d) * __half2float(q8_blocks[group].d);
    return d * (float)sumi;
}

__device__ __forceinline__ float block_reduce_sum(float v)
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

__device__ __forceinline__ float block_reduce_max(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xFFFFFFFF, v, offset));

    __shared__ float warp_vals[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0)
        warp_vals[warp] = v;
    __syncthreads();

    if (warp == 0)
    {
        int num_warps = (blockDim.x + 31) >> 5;
        v = lane < num_warps ? warp_vals[lane] : -FLT_MAX;
        for (int offset = 16; offset > 0; offset >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xFFFFFFFF, v, offset));
    }

    return v;
}

// All-lanes warp reduction (every lane receives the full sum) via butterfly shuffle.
// Used by the GatedDeltaNet kernel where each warp owns one delta-net row and all
// lanes need the reduced dot product to apply the rank-1 state update.
__device__ __forceinline__ float warp_allreduce_sum(float v)
{
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFF, v, offset);
    return v;
}

__device__ __forceinline__ float silu(float x)
{
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_f32(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float softplus_f32(float x)
{
    return x > 0.0f ? x + log1pf(expf(-x)) : log1pf(expf(x));
}

__device__ __forceinline__ float gelu(float x)
{
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

extern "C" __global__ void ts_fill_f32(float* output, int count, float value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = value;
}

extern "C" __global__ void ts_fill_f16(half* output, int count, float value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = __float2half_rn(value);
}

extern "C" __global__ void ts_unary_f32(const float* input, float* output, int count, int op)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    float x = input[i];
    float y = x;
    if (op == 0)
        y = fmaxf(x, 0.0f);
    else if (op == 1)
        y = 1.0f / (1.0f + expf(-x));
    else if (op == 2)
        y = silu(x);
    else if (op == 3)
        y = gelu(x);
    else if (op == 4)
        y = tanhf(x);
    output[i] = y;
}

extern "C" __global__ void ts_binary_f32(const float* lhs, const float* rhs, float* output, int count, int op)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    float x = lhs[i];
    float y = rhs[i];
    if (op == 0)
        output[i] = x + y;
    else if (op == 1)
        output[i] = x - y;
    else if (op == 2)
        output[i] = x * y;
    else
        output[i] = x / y;
}

extern "C" __global__ void ts_scalar_f32(const float* input, float* output, int count, float value, int op)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    float x = input[i];
    if (op == 0)
        output[i] = x + value;
    else if (op == 1)
        output[i] = x - value;
    else if (op == 2)
        output[i] = x * value;
    else if (op == 3)
        output[i] = x / value;
    else if (op == 4)
        output[i] = value - x;
    else
        output[i] = value / x;
}

extern "C" __global__ void ts_ternary_f32(const float* x, const float* y, const float* z, float* output, int count, int op)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    float a = x[i];
    float b = y[i];
    float c = z[i];
    if (op == 0)
        output[i] = a + b * c;
    else
        output[i] = a + b / c;
}

extern "C" __global__ void ts_addmul_scalar_f32(const float* x, const float* y, float* output, int count, float value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = x[i] + y[i] * value;
}

extern "C" __global__ void ts_mulmuladd_f32(const float* x, const float* y, const float* z, const float* w, float* output, int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        output[i] = x[i] * y[i] + z[i] * w[i];
}

extern "C" __global__ void ts_binary_activation_f32(const float* a, const float* b, float* output, int count, int op)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    float x = a[i];
    float y = b[i];
    if (op == 0)
        output[i] = silu(x) * y;
    else if (op == 1)
        output[i] = gelu(x) * y;
    else
        output[i] = x * (1.0f / (1.0f + expf(-y)));
}

extern "C" __global__ void ts_add_bias_rows_f32(float* tensor, const float* bias, int rows, int cols, int bias_cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int count = rows * cols;
    if (i >= count)
        return;

    int col = i - (i / cols) * cols;
    if (col < bias_cols)
        tensor[i] += bias[col];
}

extern "C" __global__ void ts_silu_mul_split_f32(const float* gate_up, float* output, int rows, int half_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * half_dim;
    if (idx >= total)
        return;

    int row = idx / half_dim;
    int col = idx - row * half_dim;
    const float* row_ptr = gate_up + (size_t)row * half_dim * 2;
    float gate = row_ptr[col];
    float up = row_ptr[col + half_dim];
    output[idx] = silu(gate) * up;
}

extern "C" __global__ void ts_gelu_mul_split_f32(const float* gate_up, float* output, int rows, int half_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * half_dim;
    if (idx >= total)
        return;

    int row = idx / half_dim;
    int col = idx - row * half_dim;
    const float* row_ptr = gate_up + (size_t)row * half_dim * 2;
    float gate = row_ptr[col];
    float up = row_ptr[col + half_dim];
    output[idx] = gelu(gate) * up;
}

extern "C" __global__ void ts_swiglu_oai_split_f32(const float* gate_up, float* output, int rows, int half_dim, float alpha, float limit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * half_dim;
    if (idx >= total)
        return;

    int row = idx / half_dim;
    int col = idx - row * half_dim;
    const float* row_ptr = gate_up + (size_t)row * half_dim * 2;
    float x = fminf(row_ptr[col], limit);
    float y = fminf(fmaxf(row_ptr[col + half_dim], -limit), limit);
    float sig = 1.0f / (1.0f + expf(-alpha * x));
    output[idx] = x * sig * (y + 1.0f);
}

__device__ __forceinline__ float qwen35_gdn_conv_channel(
    const float* packed,
    const float* conv_state,
    const float* conv_w,
    int s,
    int ch,
    int seq_len,
    int packed_dim,
    int qkv_dim,
    int conv_kernel,
    int conv_write_idx)
{
    int conv_dim = conv_kernel - 1;
    float acc = 0.0f;

    for (int ki = 0; ki < conv_kernel; ki++)
    {
        int logical = s + ki;
        float x;
        if (logical < conv_dim)
        {
            int slot = (conv_write_idx + logical) % conv_dim;
            x = conv_state[(size_t)slot * qkv_dim + ch];
        }
        else
        {
            int input_s = logical - conv_dim;
            input_s = input_s < seq_len ? input_s : seq_len - 1;
            x = packed[(size_t)input_s * packed_dim + ch];
        }
        acc += x * conv_w[(size_t)ch * conv_kernel + ki];
    }

    return silu(acc);
}

extern "C" __global__ void ts_qwen35_gdn_packed_f32(
    const float* packed,
    float* conv_state,
    float* ssm_state,
    const float* conv_w,
    const float* dt_bias,
    const float* a_log,
    const float* ssm_norm,
    float* output,
    int seq_len,
    int packed_dim,
    int qkv_dim,
    int qk_dim,
    int v_dim,
    int num_k_heads,
    int num_v_heads,
    int head_k_dim,
    int head_v_dim,
    int conv_kernel,
    int conv_write_idx,
    float eps)
{
    int h = blockIdx.x;
    if (h >= num_v_heads)
        return;

    // The delta-net rows of a head are mutually independent: row r reads and writes
    // only state_head[r, :]. The shared inputs (the normalized conv outputs q/k and
    // the per-head scalars) are computed once, then every row is processed by its own
    // warp in parallel. This replaces the old design that walked the rows one at a
    // time with a full block reduction per row (head_v_dim sequential sync points).
    extern __shared__ float scratch[];
    float* q = scratch;
    float* k = q + head_k_dim;
    float* core = k + head_k_dim;

    __shared__ float q_scale;
    __shared__ float k_scale;
    __shared__ float gate_h;
    __shared__ float beta_h;
    __shared__ float rms_inv;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = nthreads >> 5;

    int src_h = h % num_k_heads;
    int q_offset = src_h * head_k_dim;
    int k_offset = qk_dim + src_h * head_k_dim;
    int v_offset = 2 * qk_dim + h * head_v_dim;
    int z_offset = qkv_dim + h * head_v_dim;
    int beta_offset = qkv_dim + v_dim + h;
    int alpha_offset = qkv_dim + v_dim + num_v_heads + h;
    int state_per_head = head_v_dim * head_k_dim;
    float* state_head = ssm_state + (size_t)h * state_per_head;
    float q_head_scale = rsqrtf((float)head_v_dim);

    for (int s = 0; s < seq_len; s++)
    {
        const float* packed_row = packed + (size_t)s * packed_dim;

        // Causal conv + L2-norm accumulation for the shared q/k vectors.
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (int d = tid; d < head_k_dim; d += nthreads)
        {
            float qv = qwen35_gdn_conv_channel(
                packed, conv_state, conv_w, s, q_offset + d,
                seq_len, packed_dim, qkv_dim, conv_kernel, conv_write_idx);
            float kv = qwen35_gdn_conv_channel(
                packed, conv_state, conv_w, s, k_offset + d,
                seq_len, packed_dim, qkv_dim, conv_kernel, conv_write_idx);
            q[d] = qv;
            k[d] = kv;
            q_sum += qv * qv;
            k_sum += kv * kv;
        }

        q_sum = block_reduce_sum(q_sum);
        __syncthreads();
        k_sum = block_reduce_sum(k_sum);
        if (tid == 0)
        {
            q_scale = rsqrtf(q_sum + eps) * q_head_scale;
            k_scale = rsqrtf(k_sum + eps);
            gate_h = softplus_f32(packed_row[alpha_offset] + dt_bias[h]) * a_log[h];
            beta_h = sigmoid_f32(packed_row[beta_offset]);
        }
        __syncthreads();

        // Normalize the shared q/k and decay the recurrent state in place.
        float state_scale = expf(gate_h);
        for (int d = tid; d < head_k_dim; d += nthreads)
        {
            q[d] *= q_scale;
            k[d] *= k_scale;
        }
        for (int i = tid; i < state_per_head; i += nthreads)
            state_head[i] *= state_scale;
        __syncthreads();

        // One warp per row: kv = <state_row, k>, rank-1 update, core = <state_row, q>.
        // All lanes share the same row, so __shfl broadcasts/reductions are warp-safe.
        float beta = beta_h;
        for (int row = warp; row < head_v_dim; row += num_warps)
        {
            float* state_row = state_head + (size_t)row * head_k_dim;
            float kv_mem = 0.0f;
            for (int d = lane; d < head_k_dim; d += 32)
                kv_mem += state_row[d] * k[d];
            kv_mem = warp_allreduce_sum(kv_mem);

            float vrow;
            if (lane == 0)
                vrow = qwen35_gdn_conv_channel(
                    packed, conv_state, conv_w, s, v_offset + row,
                    seq_len, packed_dim, qkv_dim, conv_kernel, conv_write_idx);
            vrow = __shfl_sync(0xFFFFFFFF, vrow, 0);
            float delta = (vrow - kv_mem) * beta;

            // Fuse the state update with the core dot product to read state once.
            float core_v = 0.0f;
            for (int d = lane; d < head_k_dim; d += 32)
            {
                float sd = state_row[d] + k[d] * delta;
                state_row[d] = sd;
                core_v += sd * q[d];
            }
            core_v = warp_allreduce_sum(core_v);
            if (lane == 0)
                core[row] = core_v;
        }
        __syncthreads();

        float sum_sq = 0.0f;
        for (int row = tid; row < head_v_dim; row += nthreads)
            sum_sq += core[row] * core[row];
        sum_sq = block_reduce_sum(sum_sq);
        if (tid == 0)
            rms_inv = rsqrtf(sum_sq / (float)head_v_dim + eps);
        __syncthreads();

        float* out_row = output + (size_t)s * v_dim + h * head_v_dim;
        for (int row = tid; row < head_v_dim; row += nthreads)
        {
            float z = packed_row[z_offset + row];
            out_row[row] = core[row] * rms_inv * ssm_norm[row] * silu(z);
        }
        __syncthreads();
    }
}

extern "C" __global__ void ts_qwen35_gdn_update_conv_state_f32(
    const float* packed,
    float* conv_state,
    int seq_len,
    int packed_dim,
    int qkv_dim,
    int conv_dim,
    int conv_write_idx)
{
    int tail = seq_len < conv_dim ? seq_len : conv_dim;
    int total = tail * qkv_dim;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;

    int t = i / qkv_dim;
    int ch = i - t * qkv_dim;
    int s = seq_len - tail + t;
    int slot = (conv_write_idx + s) % conv_dim;
    conv_state[(size_t)slot * qkv_dim + ch] = packed[(size_t)s * packed_dim + ch];
}

extern "C" __global__ void ts_qwen35_gdn_pack_inputs_f32(
    const float* qkv,
    const float* z,
    const float* beta,
    const float* alpha,
    float* packed,
    int seq_len,
    int qkv_dim,
    int z_dim,
    int num_v_heads,
    int packed_dim)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * packed_dim;
    if (i >= total)
        return;

    int s = i / packed_dim;
    int col = i - s * packed_dim;
    if (col < qkv_dim)
    {
        packed[i] = qkv[(size_t)s * qkv_dim + col];
        return;
    }

    col -= qkv_dim;
    if (col < z_dim)
    {
        packed[i] = z[(size_t)s * z_dim + col];
        return;
    }

    col -= z_dim;
    if (col < num_v_heads)
    {
        packed[i] = beta[(size_t)s * num_v_heads + col];
        return;
    }

    col -= num_v_heads;
    packed[i] = alpha[(size_t)s * num_v_heads + col];
}

extern "C" __global__ void ts_rmsnorm_f32(
    const float* input,
    const float* alpha,
    const float* beta,
    float* output,
    int rows,
    int cols,
    float eps)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const float* x = input + (size_t)row * cols;
    float* y = output + (size_t)row * cols;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float v = x[i];
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float inv_rms;
    if (threadIdx.x == 0)
        inv_rms = rsqrtf(sum_sq / (float)cols + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float v = x[i] * inv_rms * alpha[i];
        if (beta != 0)
            v += beta[i];
        y[i] = v;
    }
}

// Fused RMSNorm + residual add (Gemma post-norm: residual += rms_norm(sublayer_out)).
// residual[row,i] += (input[row,i] * inv_rms(input[row])) * alpha[i], in place. Fuses
// the per-layer RMSNorm and the residual Add into one kernel (4 such pairs per Gemma 4
// layer) to cut the verify's per-op launch count.
extern "C" __global__ void ts_rmsnorm_residual_add_f32(
    const float* input,
    const float* alpha,
    float* residual,
    int rows,
    int cols,
    float eps)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const float* x = input + (size_t)row * cols;
    float* r = residual + (size_t)row * cols;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float v = x[i];
        sum_sq += v * v;
    }

    sum_sq = block_reduce_sum(sum_sq);
    __shared__ float inv_rms;
    if (threadIdx.x == 0)
        inv_rms = rsqrtf(sum_sq / (float)cols + eps);
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        r[i] += x[i] * inv_rms * alpha[i];
}

extern "C" __global__ void ts_softmax_f32(const float* input, float* output, int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    const float* x = input + (size_t)row * cols;
    float* y = output + (size_t)row * cols;

    float max_v = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        max_v = fmaxf(max_v, x[i]);

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
    {
        float e = expf(x[i] - shared_max);
        y[i] = e;
        sum += e;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = 1.0f / sum;
    __syncthreads();

    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        y[i] *= inv_sum;
}

extern "C" __global__ void ts_attention_softmax_sinks_f32(
    float* scores,
    const float* sinks,
    int num_heads,
    int seq_len,
    int kv_len,
    int mask_start,
    int window_size,
    float scale,
    int has_sinks)
{
    int row = blockIdx.x;
    int total_rows = num_heads * seq_len;
    if (row >= total_rows)
        return;

    int head = row / seq_len;
    int q = row - head * seq_len;
    int visible = mask_start + q;
    int min_visible = 0;
    if (window_size > 0)
        min_visible = max(0, visible - window_size + 1);

    float* row_ptr = scores + (size_t)row * kv_len;
    float max_v = has_sinks ? sinks[head] : -FLT_MAX;
    for (int k = threadIdx.x; k < kv_len; k += blockDim.x)
    {
        bool allowed = k <= visible && k >= min_visible;
        float v = allowed ? row_ptr[k] * scale : -FLT_MAX;
        row_ptr[k] = v;
        max_v = fmaxf(max_v, v);
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = (threadIdx.x == 0 && has_sinks) ? expf(sinks[head] - shared_max) : 0.0f;
    for (int k = threadIdx.x; k < kv_len; k += blockDim.x)
    {
        float v = row_ptr[k];
        float p = v == -FLT_MAX ? 0.0f : expf(v - shared_max);
        row_ptr[k] = p;
        sum += p;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    for (int k = threadIdx.x; k < kv_len; k += blockDim.x)
        row_ptr[k] *= inv_sum;
}

extern "C" __global__ void ts_scaled_dot_product_attention_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* mask,
    float* output,
    int batch,
    int seq_q,
    int seq_k,
    int heads,
    int key_dim,
    int value_dim,
    float scale,
    int has_mask)
{
    int batch_head = blockIdx.x;
    int q_pos = blockIdx.y;
    if (batch_head >= batch * heads || q_pos >= seq_q)
        return;

    int b = batch_head / heads;
    int h = batch_head - b * heads;
    const float* q = query + (((size_t)b * seq_q + q_pos) * heads + h) * key_dim;
    extern __shared__ float scores[];

    float max_v = -FLT_MAX;
    for (int k_pos = threadIdx.x; k_pos < seq_k; k_pos += blockDim.x)
    {
        const float* k = key + (((size_t)b * seq_k + k_pos) * heads + h) * key_dim;
        float dot = 0.0f;
        for (int d = 0; d < key_dim; d++)
            dot += q[d] * k[d];

        float score = dot * scale;
        if (has_mask)
            score += mask[(((size_t)b * heads + h) * seq_q + q_pos) * seq_k + k_pos];
        scores[k_pos] = score;
        max_v = fmaxf(max_v, score);
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = 0.0f;
    for (int k_pos = threadIdx.x; k_pos < seq_k; k_pos += blockDim.x)
    {
        float p = expf(scores[k_pos] - shared_max);
        scores[k_pos] = p;
        sum += p;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    float* out = output + (((size_t)b * seq_q + q_pos) * heads + h) * value_dim;
    for (int d = threadIdx.x; d < value_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int k_pos = 0; k_pos < seq_k; k_pos++)
        {
            const float* v = value + (((size_t)b * seq_k + k_pos) * heads + h) * value_dim;
            acc += scores[k_pos] * inv_sum * v[d];
        }
        out[d] = acc;
    }
}

// kv_stride is the per-kv-head element stride of key/value: it equals kv_len for a
// CONTIGUOUS [num_kv_heads, kv_len, head_dim] tensor (the seq-heads case), or the
// cache capacity for the LIVE cache [num_kv_heads, cache_size, head_dim] read in
// place (global full-attention verify — kv_len <= kv_stride logical positions).
extern "C" __global__ void ts_gqa_prefill_attention_f32(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int kv_len,
    int head_dim,
    int mask_start,
    int window_size,
    float scale,
    int kv_stride)
{
    int q_head = blockIdx.x;
    int q_pos = blockIdx.y;
    if (q_head >= num_q_heads || q_pos >= seq_len)
        return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;
    int visible = mask_start + q_pos;
    int min_visible = 0;
    if (window_size > 0)
        min_visible = max(0, visible - window_size + 1);
    int max_visible = min(visible, kv_len - 1);

    const float* q = query + ((size_t)q_head * seq_len + q_pos) * head_dim;
    extern __shared__ float scores[];

    float max_v = -FLT_MAX;
    for (int k_pos = min_visible + threadIdx.x; k_pos <= max_visible; k_pos += blockDim.x)
    {
        const float* k = key + ((size_t)kv_head * kv_stride + k_pos) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q[d] * k[d];
        float score = dot * scale;
        max_v = fmaxf(max_v, score);
        scores[k_pos] = score;
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = 0.0f;
    for (int k_pos = min_visible + threadIdx.x; k_pos <= max_visible; k_pos += blockDim.x)
    {
        float p = expf(scores[k_pos] - shared_max);
        scores[k_pos] = p;
        sum += p;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    float* out = output + ((size_t)q_pos * num_q_heads + q_head) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int k_pos = min_visible; k_pos <= max_visible; k_pos++)
        {
            float p = scores[k_pos];
            const float* v = value + ((size_t)kv_head * kv_stride + k_pos) * head_dim;
            acc += p * inv_sum * v[d];
        }
        out[d] = acc;
    }
}

extern "C" __global__ void ts_gqa_decode_attention_sinks_f32(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const float* sinks,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int attend_start,
    int attend_len,
    int cache_size,
    int circular,
    float scale,
    int has_sinks)
{
    int q_head = blockIdx.x;
    if (q_head >= num_q_heads)
        return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;
    const float* q = query + (size_t)q_head * head_dim;
    extern __shared__ float scores[];

    float max_v = has_sinks ? sinks[q_head] : -FLT_MAX;
    for (int t = threadIdx.x; t < attend_len; t += blockDim.x)
    {
        int logical_pos = attend_start + t;
        int cache_pos = circular ? (logical_pos % cache_size) : logical_pos;
        if (cache_pos < 0)
            cache_pos += cache_size;

        const float* k = key_cache + ((size_t)kv_head * cache_size + cache_pos) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q[d] * k[d];

        float score = dot * scale;
        scores[t] = score;
        max_v = fmaxf(max_v, score);
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = (threadIdx.x == 0 && has_sinks) ? expf(sinks[q_head] - shared_max) : 0.0f;
    for (int t = threadIdx.x; t < attend_len; t += blockDim.x)
    {
        float p = expf(scores[t] - shared_max);
        scores[t] = p;
        sum += p;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    float* out = output + (size_t)q_head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int t = 0; t < attend_len; t++)
        {
            int logical_pos = attend_start + t;
            int cache_pos = circular ? (logical_pos % cache_size) : logical_pos;
            if (cache_pos < 0)
                cache_pos += cache_size;

            const float* v = value_cache + ((size_t)kv_head * cache_size + cache_pos) * head_dim;
            acc += scores[t] * inv_sum * v[d];
        }
        out[d] = acc;
    }
}

extern "C" __global__ void ts_gqa_decode_attention_sinks_f16(
    const float* query,
    const half* key_cache,
    const half* value_cache,
    const float* sinks,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int attend_start,
    int attend_len,
    int cache_size,
    int circular,
    float scale,
    int has_sinks)
{
    int q_head = blockIdx.x;
    if (q_head >= num_q_heads)
        return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;
    const float* q = query + (size_t)q_head * head_dim;
    extern __shared__ float scores[];

    float max_v = has_sinks ? sinks[q_head] : -FLT_MAX;
    for (int t = threadIdx.x; t < attend_len; t += blockDim.x)
    {
        int logical_pos = attend_start + t;
        int cache_pos = circular ? (logical_pos % cache_size) : logical_pos;
        if (cache_pos < 0)
            cache_pos += cache_size;

        const half* k = key_cache + ((size_t)kv_head * cache_size + cache_pos) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q[d] * __half2float(k[d]);

        float score = dot * scale;
        scores[t] = score;
        max_v = fmaxf(max_v, score);
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = (threadIdx.x == 0 && has_sinks) ? expf(sinks[q_head] - shared_max) : 0.0f;
    for (int t = threadIdx.x; t < attend_len; t += blockDim.x)
    {
        float p = expf(scores[t] - shared_max);
        scores[t] = p;
        sum += p;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    float* out = output + (size_t)q_head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int t = 0; t < attend_len; t++)
        {
            int logical_pos = attend_start + t;
            int cache_pos = circular ? (logical_pos % cache_size) : logical_pos;
            if (cache_pos < 0)
                cache_pos += cache_size;

            const half* v = value_cache + ((size_t)kv_head * cache_size + cache_pos) * head_dim;
            acc += scores[t] * inv_sum * __half2float(v[d]);
        }
        out[d] = acc;
    }
}

extern "C" __global__ void ts_gqa_prefill_attention_f16(
    const float* query,
    const half* key,
    const half* value,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int kv_len,
    int head_dim,
    int mask_start,
    int window_size,
    float scale,
    int kv_stride)
{
    int q_head = blockIdx.x;
    int q_pos = blockIdx.y;
    if (q_head >= num_q_heads || q_pos >= seq_len)
        return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;
    int visible = mask_start + q_pos;
    int min_visible = 0;
    if (window_size > 0)
        min_visible = max(0, visible - window_size + 1);
    int max_visible = min(visible, kv_len - 1);

    const float* q = query + ((size_t)q_head * seq_len + q_pos) * head_dim;
    extern __shared__ float scores[];

    float max_v = -FLT_MAX;
    for (int k_pos = min_visible + threadIdx.x; k_pos <= max_visible; k_pos += blockDim.x)
    {
        const half* k = key + ((size_t)kv_head * kv_stride + k_pos) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q[d] * __half2float(k[d]);
        float score = dot * scale;
        max_v = fmaxf(max_v, score);
        scores[k_pos] = score;
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = 0.0f;
    for (int k_pos = min_visible + threadIdx.x; k_pos <= max_visible; k_pos += blockDim.x)
    {
        float p = expf(scores[k_pos] - shared_max);
        scores[k_pos] = p;
        sum += p;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    float* out = output + ((size_t)q_pos * num_q_heads + q_head) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int k_pos = min_visible; k_pos <= max_visible; k_pos++)
        {
            float p = scores[k_pos];
            const half* v = value + ((size_t)kv_head * kv_stride + k_pos) * head_dim;
            acc += p * inv_sum * __half2float(v[d]);
        }
        out[d] = acc;
    }
}

extern "C" __global__ void ts_gqa_prefill_attention_sinks_f32(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const float* sinks,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int kv_len,
    int cache_size,
    int head_dim,
    int mask_start,
    int window_size,
    float scale,
    int has_sinks)
{
    int q_head = blockIdx.x;
    int q_pos = blockIdx.y;
    if (q_head >= num_q_heads || q_pos >= seq_len)
        return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;
    int visible = mask_start + q_pos;
    int min_visible = 0;
    if (window_size > 0)
        min_visible = max(0, visible - window_size + 1);
    int max_visible = min(visible, kv_len - 1);

    const float* q = query + ((size_t)q_head * seq_len + q_pos) * head_dim;
    extern __shared__ float scores[];

    float max_v = has_sinks ? sinks[q_head] : -FLT_MAX;
    for (int k_pos = min_visible + threadIdx.x; k_pos <= max_visible; k_pos += blockDim.x)
    {
        const float* k = key_cache + ((size_t)kv_head * cache_size + k_pos) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q[d] * k[d];
        float score = dot * scale;
        max_v = fmaxf(max_v, score);
        scores[k_pos] = score;
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = (threadIdx.x == 0 && has_sinks) ? expf(sinks[q_head] - shared_max) : 0.0f;
    for (int k_pos = min_visible + threadIdx.x; k_pos <= max_visible; k_pos += blockDim.x)
    {
        float p = expf(scores[k_pos] - shared_max);
        scores[k_pos] = p;
        sum += p;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    float* out = output + ((size_t)q_pos * num_q_heads + q_head) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int k_pos = min_visible; k_pos <= max_visible; k_pos++)
        {
            float p = scores[k_pos];
            const float* v = value_cache + ((size_t)kv_head * cache_size + k_pos) * head_dim;
            acc += p * inv_sum * v[d];
        }
        out[d] = acc;
    }
}

extern "C" __global__ void ts_gqa_prefill_attention_sinks_f16(
    const float* query,
    const half* key_cache,
    const half* value_cache,
    const float* sinks,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int kv_len,
    int cache_size,
    int head_dim,
    int mask_start,
    int window_size,
    float scale,
    int has_sinks)
{
    int q_head = blockIdx.x;
    int q_pos = blockIdx.y;
    if (q_head >= num_q_heads || q_pos >= seq_len)
        return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;
    int visible = mask_start + q_pos;
    int min_visible = 0;
    if (window_size > 0)
        min_visible = max(0, visible - window_size + 1);
    int max_visible = min(visible, kv_len - 1);

    const float* q = query + ((size_t)q_head * seq_len + q_pos) * head_dim;
    extern __shared__ float scores[];

    float max_v = has_sinks ? sinks[q_head] : -FLT_MAX;
    for (int k_pos = min_visible + threadIdx.x; k_pos <= max_visible; k_pos += blockDim.x)
    {
        const half* k = key_cache + ((size_t)kv_head * cache_size + k_pos) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q[d] * __half2float(k[d]);
        float score = dot * scale;
        max_v = fmaxf(max_v, score);
        scores[k_pos] = score;
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = (threadIdx.x == 0 && has_sinks) ? expf(sinks[q_head] - shared_max) : 0.0f;
    for (int k_pos = min_visible + threadIdx.x; k_pos <= max_visible; k_pos += blockDim.x)
    {
        float p = expf(scores[k_pos] - shared_max);
        scores[k_pos] = p;
        sum += p;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    float* out = output + ((size_t)q_pos * num_q_heads + q_head) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int k_pos = min_visible; k_pos <= max_visible; k_pos++)
        {
            float p = scores[k_pos];
            const half* v = value_cache + ((size_t)kv_head * cache_size + k_pos) * head_dim;
            acc += p * inv_sum * __half2float(v[d]);
        }
        out[d] = acc;
    }
}

extern "C" __global__ void ts_gqa_decode_attention_f32(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int attend_start,
    int attend_len,
    int cache_size,
    int circular,
    float scale)
{
    int q_head = blockIdx.x;
    if (q_head >= num_q_heads)
        return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;
    const float* q = query + (size_t)q_head * head_dim;
    extern __shared__ float scores[];

    float max_v = -FLT_MAX;
    for (int t = threadIdx.x; t < attend_len; t += blockDim.x)
    {
        int logical_pos = attend_start + t;
        int cache_pos = circular ? (logical_pos % cache_size) : logical_pos;
        if (cache_pos < 0)
            cache_pos += cache_size;

        const float* k = key_cache + ((size_t)kv_head * cache_size + cache_pos) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q[d] * k[d];

        float score = dot * scale;
        scores[t] = score;
        max_v = fmaxf(max_v, score);
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = 0.0f;
    for (int t = threadIdx.x; t < attend_len; t += blockDim.x)
    {
        float p = expf(scores[t] - shared_max);
        scores[t] = p;
        sum += p;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    float* out = output + (size_t)q_head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int t = 0; t < attend_len; t++)
        {
            int logical_pos = attend_start + t;
            int cache_pos = circular ? (logical_pos % cache_size) : logical_pos;
            if (cache_pos < 0)
                cache_pos += cache_size;

            const float* v = value_cache + ((size_t)kv_head * cache_size + cache_pos) * head_dim;
            acc += scores[t] * inv_sum * v[d];
        }
        out[d] = acc;
    }
}

extern "C" __global__ void ts_gqa_decode_attention_f16(
    const float* query,
    const half* key_cache,
    const half* value_cache,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int attend_start,
    int attend_len,
    int cache_size,
    int circular,
    float scale)
{
    int q_head = blockIdx.x;
    if (q_head >= num_q_heads)
        return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;
    const float* q = query + (size_t)q_head * head_dim;
    extern __shared__ float scores[];

    float max_v = -FLT_MAX;
    for (int t = threadIdx.x; t < attend_len; t += blockDim.x)
    {
        int logical_pos = attend_start + t;
        int cache_pos = circular ? (logical_pos % cache_size) : logical_pos;
        if (cache_pos < 0)
            cache_pos += cache_size;

        const half* k = key_cache + ((size_t)kv_head * cache_size + cache_pos) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q[d] * __half2float(k[d]);

        float score = dot * scale;
        scores[t] = score;
        max_v = fmaxf(max_v, score);
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = 0.0f;
    for (int t = threadIdx.x; t < attend_len; t += blockDim.x)
    {
        float p = expf(scores[t] - shared_max);
        scores[t] = p;
        sum += p;
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    float* out = output + (size_t)q_head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int t = 0; t < attend_len; t++)
        {
            int logical_pos = attend_start + t;
            int cache_pos = circular ? (logical_pos % cache_size) : logical_pos;
            if (cache_pos < 0)
                cache_pos += cache_size;

            const half* v = value_cache + ((size_t)kv_head * cache_size + cache_pos) * head_dim;
            acc += scores[t] * inv_sum * __half2float(v[d]);
        }
        out[d] = acc;
    }
}

template <typename cache_t>
__device__ __forceinline__ float ts_cache_to_float(cache_t v)
{
    return (float)v;
}

template <>
__device__ __forceinline__ float ts_cache_to_float<half>(half v)
{
    return __half2float(v);
}

template <typename cache_t>
__device__ __forceinline__ void ts_gqa_decode_attention_partition_impl(
    const float* query,
    const cache_t* key_cache,
    const cache_t* value_cache,
    const float* sinks,
    float* partial,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int attend_start,
    int attend_len,
    int cache_size,
    int circular,
    float scale,
    int has_sinks,
    int num_partitions,
    int partition_size,
    float* scores)
{
    int q_head = blockIdx.x;
    int partition = blockIdx.y;
    if (q_head >= num_q_heads || partition >= num_partitions)
        return;

    int part_start = partition * partition_size;
    int part_len = attend_len - part_start;
    if (part_len > partition_size)
        part_len = partition_size;
    if (part_len < 0)
        part_len = 0;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;
    const float* q = query + (size_t)q_head * head_dim;
    int include_sink = has_sinks && partition == 0;

    float max_v = include_sink ? sinks[q_head] : -FLT_MAX;
    for (int i = threadIdx.x; i < part_len; i += blockDim.x)
    {
        int logical_pos = attend_start + part_start + i;
        int cache_pos = circular ? (logical_pos % cache_size) : logical_pos;
        if (cache_pos < 0)
            cache_pos += cache_size;

        const cache_t* k = key_cache + ((size_t)kv_head * cache_size + cache_pos) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q[d] * ts_cache_to_float<cache_t>(k[d]);

        float score = dot * scale;
        scores[i] = score;
        max_v = fmaxf(max_v, score);
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = (threadIdx.x == 0 && include_sink) ? expf(sinks[q_head] - shared_max) : 0.0f;
    for (int i = threadIdx.x; i < part_len; i += blockDim.x)
    {
        float p = expf(scores[i] - shared_max);
        scores[i] = p;
        sum += p;
    }

    sum = block_reduce_sum(sum);
    __shared__ float shared_sum;
    if (threadIdx.x == 0)
        shared_sum = sum;
    __syncthreads();

    float* partial_row = partial + ((size_t)q_head * num_partitions + partition) * (head_dim + 2);
    if (threadIdx.x == 0)
    {
        partial_row[0] = shared_max;
        partial_row[1] = shared_sum;
    }

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int i = 0; i < part_len; i++)
        {
            int logical_pos = attend_start + part_start + i;
            int cache_pos = circular ? (logical_pos % cache_size) : logical_pos;
            if (cache_pos < 0)
                cache_pos += cache_size;

            const cache_t* v = value_cache + ((size_t)kv_head * cache_size + cache_pos) * head_dim;
            acc += scores[i] * ts_cache_to_float<cache_t>(v[d]);
        }
        partial_row[2 + d] = acc;
    }
}

extern "C" __global__ void ts_gqa_decode_attention_partition_f32(
    const float* query,
    const float* key_cache,
    const float* value_cache,
    const float* sinks,
    float* partial,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int attend_start,
    int attend_len,
    int cache_size,
    int circular,
    float scale,
    int has_sinks,
    int num_partitions,
    int partition_size)
{
    extern __shared__ float scores[];
    ts_gqa_decode_attention_partition_impl<float>(
        query, key_cache, value_cache, sinks, partial,
        num_q_heads, num_kv_heads, head_dim,
        attend_start, attend_len, cache_size, circular, scale,
        has_sinks, num_partitions, partition_size, scores);
}

extern "C" __global__ void ts_gqa_decode_attention_partition_f16(
    const float* query,
    const half* key_cache,
    const half* value_cache,
    const float* sinks,
    float* partial,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int attend_start,
    int attend_len,
    int cache_size,
    int circular,
    float scale,
    int has_sinks,
    int num_partitions,
    int partition_size)
{
    extern __shared__ float scores[];
    ts_gqa_decode_attention_partition_impl<half>(
        query, key_cache, value_cache, sinks, partial,
        num_q_heads, num_kv_heads, head_dim,
        attend_start, attend_len, cache_size, circular, scale,
        has_sinks, num_partitions, partition_size, scores);
}

extern "C" __global__ void ts_gqa_decode_attention_partition_reduce_f32(
    const float* partial,
    float* output,
    int num_q_heads,
    int head_dim,
    int num_partitions)
{
    int q_head = blockIdx.x;
    if (q_head >= num_q_heads)
        return;

    int stride = head_dim + 2;
    const float* partial_head = partial + (size_t)q_head * num_partitions * stride;

    float max_v = -FLT_MAX;
    for (int p = threadIdx.x; p < num_partitions; p += blockDim.x)
    {
        const float* row = partial_head + (size_t)p * stride;
        if (row[1] > 0.0f)
            max_v = fmaxf(max_v, row[0]);
    }

    max_v = block_reduce_max(max_v);
    __shared__ float shared_max;
    if (threadIdx.x == 0)
        shared_max = max_v;
    __syncthreads();

    float sum = 0.0f;
    for (int p = threadIdx.x; p < num_partitions; p += blockDim.x)
    {
        const float* row = partial_head + (size_t)p * stride;
        if (row[1] > 0.0f)
            sum += expf(row[0] - shared_max) * row[1];
    }

    sum = block_reduce_sum(sum);
    __shared__ float inv_sum;
    if (threadIdx.x == 0)
        inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    __syncthreads();

    float* out = output + (size_t)q_head * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc = 0.0f;
        for (int p = 0; p < num_partitions; p++)
        {
            const float* row = partial_head + (size_t)p * stride;
            if (row[1] > 0.0f)
                acc += expf(row[0] - shared_max) * row[2 + d];
        }
        out[d] = acc * inv_sum;
    }
}

extern "C" __global__ void ts_slice_columns_f32(
    const float* source,
    float* output,
    int rows,
    int source_cols,
    int col_offset,
    int width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * width;
    if (idx >= total)
        return;

    int row = idx / width;
    int col = idx - row * width;
    output[idx] = source[(size_t)row * source_cols + col_offset + col];
}

extern "C" __global__ void ts_flat_to_head_first_f32(
    const float* source,
    float* output,
    int seq_len,
    int num_heads,
    int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads * head_dim;
    if (idx >= total)
        return;

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int seq = tmp % seq_len;
    int head = tmp / seq_len;
    output[idx] = source[((size_t)seq * num_heads + head) * head_dim + d];
}

extern "C" __global__ void ts_split_qkv_head_first_f32(
    const float* source,
    float* output,
    int seq_len,
    int source_cols,
    int col_offset,
    int num_heads,
    int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads * head_dim;
    if (idx >= total)
        return;

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int seq = tmp % seq_len;
    int head = tmp / seq_len;
    output[idx] = source[(size_t)seq * source_cols + col_offset + head * head_dim + d];
}

extern "C" __global__ void ts_copy_head_first_to_cache_f32(
    const float* source,
    float* cache,
    int num_heads,
    int seq_len,
    int head_dim,
    int start_pos,
    int cache_size,
    int circular)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * head_dim;
    if (idx >= total)
        return;

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int seq = tmp % seq_len;
    int head = tmp / seq_len;
    int cache_pos = circular ? ((start_pos + seq) % cache_size) : (start_pos + seq);
    cache[((size_t)head * cache_size + cache_pos) * head_dim + d] = source[idx];
}

extern "C" __global__ void ts_copy_head_first_to_cache_f16(
    const float* source,
    half* cache,
    int num_heads,
    int seq_len,
    int head_dim,
    int start_pos,
    int cache_size,
    int circular)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * head_dim;
    if (idx >= total)
        return;

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int seq = tmp % seq_len;
    int head = tmp / seq_len;
    int cache_pos = circular ? ((start_pos + seq) % cache_size) : (start_pos + seq);
    cache[((size_t)head * cache_size + cache_pos) * head_dim + d] = __float2half_rn(source[idx]);
}

extern "C" __global__ void ts_gather_circular_head_first_f32(
    const float* cache,
    float* output,
    int num_heads,
    int seq_len,
    int head_dim,
    int start_pos,
    int cache_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * head_dim;
    if (idx >= total)
        return;

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int seq = tmp % seq_len;
    int head = tmp / seq_len;
    int cache_pos = (start_pos + seq) % cache_size;
    if (cache_pos < 0)
        cache_pos += cache_size;
    output[idx] = cache[((size_t)head * cache_size + cache_pos) * head_dim + d];
}

extern "C" __global__ void ts_gather_circular_head_first_f16(
    const half* cache,
    float* output,
    int num_heads,
    int seq_len,
    int head_dim,
    int start_pos,
    int cache_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * head_dim;
    if (idx >= total)
        return;

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int seq = tmp % seq_len;
    int head = tmp / seq_len;
    int cache_pos = (start_pos + seq) % cache_size;
    if (cache_pos < 0)
        cache_pos += cache_size;
    output[idx] = __half2float(cache[((size_t)head * cache_size + cache_pos) * head_dim + d]);
}

extern "C" __global__ void ts_concat_head_first_f32(
    const float* a,
    const float* b,
    float* output,
    int num_heads,
    int len_a,
    int len_b,
    int head_dim)
{
    int total_len = len_a + len_b;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * total_len * head_dim;
    if (idx >= total)
        return;

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int seq = tmp % total_len;
    int head = tmp / total_len;
    if (seq < len_a)
        output[idx] = a[((size_t)head * len_a + seq) * head_dim + d];
    else
        output[idx] = b[((size_t)head * len_b + (seq - len_a)) * head_dim + d];
}

extern "C" __global__ void ts_neox_rope_head_first_f32(
    float* data,
    const float* cos_table,
    const float* sin_table,
    int num_heads,
    int seq_len,
    int head_dim,
    int rope_half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * rope_half;
    if (idx >= total)
        return;

    int j = idx % rope_half;
    int tmp = idx / rope_half;
    int seq = tmp % seq_len;
    int head = tmp / seq_len;
    size_t base = ((size_t)head * seq_len + seq) * head_dim;
    size_t table = (size_t)seq * rope_half + j;
    float cos_v = cos_table[table];
    float sin_v = sin_table[table];
    float x0 = data[base + j];
    float x1 = data[base + j + rope_half];
    data[base + j] = x0 * cos_v - x1 * sin_v;
    data[base + j + rope_half] = x0 * sin_v + x1 * cos_v;
}

// NeoX RoPE for the FLAT [seq_len, num_heads * head_dim] layout (element (s,h,j)
// at (s*num_heads + h)*head_dim + j) — the layout Gemma 4's q/k carry before
// ReshapeToHeads. Same rotation/table indexing as the head-first kernel; only the
// element address differs. cos/sin tables are [seq_len, rope_half] (rope_half =
// partial-rotary-dims/2, with per-frequency rope_freqs.weight already baked in),
// so this covers the partial-rotary + freq-factor global RoPE that Ops.RoPEEx
// cannot express. Replaces the CPU GetFloatPtr rotation (a per-global-layer DtoH
// stall) in ApplyNeoXRoPEDecode/Prefill on the pure-C# CUDA backend.
extern "C" __global__ void ts_neox_rope_flat_f32(
    float* data,
    const float* cos_table,
    const float* sin_table,
    int num_heads,
    int seq_len,
    int head_dim,
    int rope_half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * rope_half;
    if (idx >= total)
        return;

    int j = idx % rope_half;
    int tmp = idx / rope_half;
    int head = tmp % num_heads;
    int seq = tmp / num_heads;
    size_t base = ((size_t)seq * num_heads + head) * head_dim;
    size_t table = (size_t)seq * rope_half + j;
    float cos_v = cos_table[table];
    float sin_v = sin_table[table];
    float x0 = data[base + j];
    float x1 = data[base + j + rope_half];
    data[base + j] = x0 * cos_v - x1 * sin_v;
    data[base + j + rope_half] = x0 * sin_v + x1 * cos_v;
}

extern "C" __global__ void ts_index_select_f32(
    const float* source,
    const void* indices,
    float* output,
    int rows,
    int cols,
    int source_rows,
    int indices_are_int32,
    int is_add)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    int src_idx = indices_are_int32
        ? reinterpret_cast<const int*>(indices)[row]
        : (int)reinterpret_cast<const float*>(indices)[row];
    if (src_idx < 0 || src_idx >= source_rows)
        return;

    const float* src_row = source + (size_t)src_idx * cols;
    float* out_row = output + (size_t)row * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
    {
        float v = src_row[col];
        out_row[col] = is_add ? out_row[col] + v : v;
    }
}

extern "C" __global__ void ts_add_causal_mask_f32(float* tensor, int rows, int cols, int seq_len, int start_pos, float masked_value)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    int q = row % seq_len;
    int visible = start_pos + q;
    float* row_ptr = tensor + (size_t)row * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x)
    {
        if (col > visible)
            row_ptr[col] += masked_value;
    }
}

extern "C" __global__ void ts_rope_f32(const float* input, float* output, int rows, int cols, int seq_len, int row_offset)
{
    int pair_count = cols / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * pair_count;
    if (idx >= total)
        return;

    int row = idx / pair_count;
    int pair = idx - row * pair_count;
    int m = (row % seq_len) + row_offset;
    float theta = powf(500000.0f, -2.0f * (float)pair / (float)cols);
    float angle = theta * (float)m;
    float c = cosf(angle);
    float s = sinf(angle);

    const float* src = input + (size_t)row * cols;
    float* dst = output + (size_t)row * cols;
    float left = src[pair * 2];
    float right = src[pair * 2 + 1];
    dst[pair * 2] = left * c - right * s;
    dst[pair * 2 + 1] = right * c + left * s;
}

__device__ __forceinline__ float yarn_corr_dim(int n_dims, int n_ctx_orig, float n_rot, float freq_base)
{
    return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * 3.14159265358979323846f)) / (2.0f * logf(freq_base));
}

__device__ __forceinline__ void yarn_corr_dims(int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float* low, float* high)
{
    if (beta_fast == 0.0f && beta_slow == 0.0f)
    {
        *low = FLT_MAX;
        *high = (float)(n_dims / 2 - 1);
    }
    else
    {
        *low = fmaxf(0.0f, floorf(yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base)));
        *high = fminf((float)(n_dims / 2 - 1), ceilf(yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base)));
    }
}

__device__ __forceinline__ void yarn_rope(float theta_extrap, float freq_scale, float corr_low, float corr_high, int i0, float ext_factor, float mscale, float* c, float* s)
{
    float theta_interp = freq_scale * theta_extrap;
    float ramp_y = ((float)i0 - corr_low) / fmaxf(0.001f, corr_high - corr_low);
    float ramp_mix = (1.0f - fminf(1.0f, fmaxf(0.0f, ramp_y))) * ext_factor;
    float theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
    *c = cosf(theta) * mscale;
    *s = sinf(theta) * mscale;
}

extern "C" __global__ void ts_rope_ex_f32(
    const float* input,
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
            output[global] = input[global];
        return;
    }

    if (!add_to_result && global < rows * cols)
    {
        int col = global % cols;
        if (col >= pair_count * 2)
            output[global] = input[global];
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
        yarn_corr_dims(active_dim, n_ctx_orig, freq_base, beta_fast, beta_slow, &corr_low, &corr_high);
        float mscale = attn_factor * (1.0f + 0.1f * logf(1.0f / freq_scale));
        yarn_rope(theta_extrap, freq_scale, corr_low, corr_high, i, ext_factor, mscale, &c, &s);
    }
    else
    {
        float angle = theta_extrap * freq_scale;
        c = cosf(angle);
        s = sinf(angle);
    }

    const float* src = input + (size_t)row * cols;
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

extern "C" __global__ void ts_quant_matmul_f32(
    const uint8_t* weights,
    const float* input,
    float* output,
    int type,
    int in_dim,
    int out_dim,
    int rows)
{
    const int cols_per_block = 4;
    int out_col0 = blockIdx.x * cols_per_block;
    int row = blockIdx.y;
    if (out_col0 >= out_dim || row >= rows)
        return;

    int row_bytes = qrow_bytes(type, in_dim);
    const float* x_row = input + (size_t)row * in_dim;

    const uint8_t* w_row0 = weights + (size_t)(out_col0 + 0) * row_bytes;
    const uint8_t* w_row1 = out_col0 + 1 < out_dim ? weights + (size_t)(out_col0 + 1) * row_bytes : 0;
    const uint8_t* w_row2 = out_col0 + 2 < out_dim ? weights + (size_t)(out_col0 + 2) * row_bytes : 0;
    const uint8_t* w_row3 = out_col0 + 3 < out_dim ? weights + (size_t)(out_col0 + 3) * row_bytes : 0;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (int k = threadIdx.x; k < in_dim; k += blockDim.x)
    {
        float x = x_row[k];
        acc0 += qvalue_at(w_row0, type, k) * x;
        if (w_row1 != 0)
            acc1 += qvalue_at(w_row1, type, k) * x;
        if (w_row2 != 0)
            acc2 += qvalue_at(w_row2, type, k) * x;
        if (w_row3 != 0)
            acc3 += qvalue_at(w_row3, type, k) * x;
    }

    acc0 = block_reduce_sum(acc0);
    if (threadIdx.x == 0)
        output[(size_t)row * out_dim + out_col0] = acc0;
    __syncthreads();

    acc1 = block_reduce_sum(acc1);
    if (threadIdx.x == 0 && w_row1 != 0)
        output[(size_t)row * out_dim + out_col0 + 1] = acc1;
    __syncthreads();

    acc2 = block_reduce_sum(acc2);
    if (threadIdx.x == 0 && w_row2 != 0)
        output[(size_t)row * out_dim + out_col0 + 2] = acc2;
    __syncthreads();

    acc3 = block_reduce_sum(acc3);
    if (threadIdx.x == 0 && w_row3 != 0)
        output[(size_t)row * out_dim + out_col0 + 3] = acc3;
}

// Row tile height for the row-batched quantized matmul kernels below. Each
// block handles a contiguous tile of up to TS_QMM_ROW_TILE rows for one output
// column, decoding the weight ONCE and reusing it across the tile's rows;
// grid.y = ceil(rows/TILE) covers the rest. Kept small (matches the 4-row
// ts_quant_matmul_q8_0_f32 tiling) so the accumulators stay in registers.
// Weight memory traffic / dequant work drops from B x to ceil(B/TILE) x.
// (Q4_0 — the dominant dense quant — has its own row-tiled kernel,
// ts_quant_matmul_q4_0_batched_f32, that covers a full draft window in one pass.)
#define TS_QMM_ROW_TILE 4

// Row-batched quantized matmul for SMALL row counts (speculative MTP verify
// windows, short prefill chunks). The per-row kernels elsewhere re-read AND
// re-dequantize the whole weight row once per output row, so a B-row matmul
// costs B x the (memory-bound) weight traffic -- on a multi-GB quantized model
// that makes a B-token forward cost ~B single-token decodes and speculative
// verification can never amortize.
//
// One WARP per output column (warp-shuffle reduction, no block-wide sync); the
// warp streams the weight row a SINGLE time per tile and reuses each
// dequantized weight across the tile's rows (activations read from the
// L2-resident input). Numerically matches the generic ts_quant_matmul_f32 path
// (full-precision activations x dequantized weights), not the q8_1 dp4a path.
extern "C" __global__ void ts_quant_matmul_batched_f32(
    const uint8_t* weights,
    const float* input,
    float* output,
    int type,
    int in_dim,
    int out_dim,
    int rows)
{
    int warps_per_block = blockDim.x >> 5;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int out_col = blockIdx.x * warps_per_block + warp;
    int row0 = blockIdx.y * TS_QMM_ROW_TILE;
    if (out_col >= out_dim || row0 >= rows)
        return;
    int tile = min(TS_QMM_ROW_TILE, rows - row0);

    int row_bytes = qrow_bytes(type, in_dim);
    const uint8_t* w_row = weights + (size_t)out_col * row_bytes;
    const float* in0 = input + (size_t)row0 * in_dim;

    float acc[TS_QMM_ROW_TILE];
#pragma unroll
    for (int r = 0; r < TS_QMM_ROW_TILE; r++)
        acc[r] = 0.0f;

    for (int k = lane; k < in_dim; k += 32)
    {
        float wv = qvalue_at(w_row, type, k);
#pragma unroll
        for (int r = 0; r < TS_QMM_ROW_TILE; r++)
        {
            if (r < tile)
                acc[r] += wv * in0[(size_t)r * in_dim + k];
        }
    }

#pragma unroll
    for (int r = 0; r < TS_QMM_ROW_TILE; r++)
    {
        if (r >= tile)
            break;
        float a = acc[r];
        for (int offset = 16; offset > 0; offset >>= 1)
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        if (lane == 0)
            output[(size_t)(row0 + r) * out_dim + out_col] = a;
    }
}

// One warp per output column. For batch=1 decode the old design used the whole
// block to compute 4 outputs with four sequential block-wide reductions (most
// threads idle since dot_groups = in_dim/32 < blockDim). Giving each warp its own
// output replaces those block reductions with a single warp shuffle (no
// __syncthreads), keeps every lane busy, and halves the number of blocks that
// redundantly re-quantize the activation row.
extern "C" __global__ void ts_quant_matmul_iq2_xxs_q8_1_f32(
    const uint8_t* weights,
    const float* input,
    float* output,
    int in_dim,
    int out_dim,
    int rows)
{
    int row = blockIdx.y;
    if (row >= rows || (in_dim & 255) != 0)
        return;

    int warps_per_block = blockDim.x >> 5;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int out_col = blockIdx.x * warps_per_block + warp;

    int iq_blocks = in_dim / 256;
    int q8_blocks = in_dim / TS_QK8_1;
    int row_bytes = iq_blocks * (int)sizeof(ts_block_iq2_xxs);
    const float* x_row = input + (size_t)row * in_dim;

    // The whole block quantizes the activation row to q8_1 in shared memory once,
    // then every warp reuses it. All threads must reach this barrier, so the
    // out-of-range check happens afterwards.
    extern __shared__ __align__(16) unsigned char shared_q8_bytes[];
    ts_block_q8_1* xq = reinterpret_cast<ts_block_q8_1*>(shared_q8_bytes);

    for (int qb = threadIdx.x; qb < q8_blocks; qb += blockDim.x)
        quantize_q8_1_block(x_row + (size_t)qb * TS_QK8_1, xq + qb);
    __syncthreads();

    if (out_col >= out_dim)
        return;

    const uint8_t* w_row = weights + (size_t)out_col * row_bytes;
    int dot_groups = iq_blocks * 8;
    float acc = 0.0f;
    for (int g = lane; g < dot_groups; g += 32)
    {
        int ib = g >> 3;
        int group = g & 7;
        acc += dot_iq2_xxs_q8_1(w_row + ib * (int)sizeof(ts_block_iq2_xxs), xq + ib * 8, group);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    if (lane == 0)
        output[(size_t)row * out_dim + out_col] = acc;
}

extern "C" __global__ void ts_quant_matmul_q4_0_f32(
    const uint8_t* weights,
    const float* input,
    float* output,
    int in_dim,
    int out_dim,
    int rows)
{
    const int cols_per_block = 4;
    int out_col0 = blockIdx.x * cols_per_block;
    int row = blockIdx.y;
    if (out_col0 >= out_dim || row >= rows)
        return;

    int row_bytes = (in_dim / 32) * 18;
    const float* x_row = input + (size_t)row * in_dim;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (int k = threadIdx.x; k < in_dim; k += blockDim.x)
    {
        int block_offset = (k / 32) * 18;
        int lane = k & 31;
        int packed_index = lane & 15;
        int high = lane >> 4;
        float x = x_row[k];

        const uint8_t* w0 = weights + (size_t)(out_col0 + 0) * row_bytes + block_offset;
        float d0 = __half2float(*reinterpret_cast<const half*>(w0));
        uint8_t p0 = w0[2 + packed_index];
        int q0 = (high ? (p0 >> 4) : (p0 & 0x0F)) - 8;
        acc0 += d0 * (float)q0 * x;

        if (out_col0 + 1 < out_dim)
        {
            const uint8_t* w1 = weights + (size_t)(out_col0 + 1) * row_bytes + block_offset;
            float d1 = __half2float(*reinterpret_cast<const half*>(w1));
            uint8_t p1 = w1[2 + packed_index];
            int q1 = (high ? (p1 >> 4) : (p1 & 0x0F)) - 8;
            acc1 += d1 * (float)q1 * x;
        }

        if (out_col0 + 2 < out_dim)
        {
            const uint8_t* w2 = weights + (size_t)(out_col0 + 2) * row_bytes + block_offset;
            float d2 = __half2float(*reinterpret_cast<const half*>(w2));
            uint8_t p2 = w2[2 + packed_index];
            int q2 = (high ? (p2 >> 4) : (p2 & 0x0F)) - 8;
            acc2 += d2 * (float)q2 * x;
        }

        if (out_col0 + 3 < out_dim)
        {
            const uint8_t* w3 = weights + (size_t)(out_col0 + 3) * row_bytes + block_offset;
            float d3 = __half2float(*reinterpret_cast<const half*>(w3));
            uint8_t p3 = w3[2 + packed_index];
            int q3 = (high ? (p3 >> 4) : (p3 & 0x0F)) - 8;
            acc3 += d3 * (float)q3 * x;
        }
    }

    acc0 = block_reduce_sum(acc0);
    if (threadIdx.x == 0)
        output[(size_t)row * out_dim + out_col0] = acc0;
    __syncthreads();

    acc1 = block_reduce_sum(acc1);
    if (threadIdx.x == 0 && out_col0 + 1 < out_dim)
        output[(size_t)row * out_dim + out_col0 + 1] = acc1;
    __syncthreads();

    acc2 = block_reduce_sum(acc2);
    if (threadIdx.x == 0 && out_col0 + 2 < out_dim)
        output[(size_t)row * out_dim + out_col0 + 2] = acc2;
    __syncthreads();

    acc3 = block_reduce_sum(acc3);
    if (threadIdx.x == 0 && out_col0 + 3 < out_dim)
        output[(size_t)row * out_dim + out_col0 + 3] = acc3;
}

// Row-tiled Q4_0 matmul for the speculative-MTP verify window. Keeps the
// ts_quant_matmul_q4_0_f32 structure (every thread in the block cooperating on the
// dot product -> full column parallelism, unlike the warp-per-column generic
// scalar batched kernel which under-fills for Q4_0), but each block computes a
// TILE of consecutive rows for TS_Q40_COLS output columns: every weight nibble is
// unpacked ONCE and multiply-accumulated into all TILE rows. Weight read + dequant
// traffic drops from B x to ceil(B/TILE) x, so a B-row verify forward stops
// costing ~B single-token decodes (the reason MTP speculation was a net loss on
// the pure-C# CUDA backend for Q4_0 models). The tile covers a full draft window
// (n_max + 1 = 9 rows) in ONE pass so the most-confident drafts don't spill into a
// second weight-streaming tile; 2 columns/block keeps the accumulator file
// (TS_Q40_COLS * TS_Q40_ROW_TILE) in registers at full occupancy (4 cols x tile 12
// spilled and regressed). Numerically identical to the per-row kernel: same
// d*(q-8) dequant, same FP32 accumulation order over k.
#define TS_Q40_ROW_TILE 12
#define TS_Q40_COLS 2
extern "C" __global__ void ts_quant_matmul_q4_0_batched_f32(
    const uint8_t* weights,
    const float* input,
    float* output,
    int in_dim,
    int out_dim,
    int rows)
{
    int out_col0 = blockIdx.x * TS_Q40_COLS;
    int row0 = blockIdx.y * TS_Q40_ROW_TILE;
    if (out_col0 >= out_dim || row0 >= rows)
        return;
    int tile = min(TS_Q40_ROW_TILE, rows - row0);
    int ncols = min(TS_Q40_COLS, out_dim - out_col0);
    int row_bytes = (in_dim / 32) * 18;

    float acc[TS_Q40_COLS][TS_Q40_ROW_TILE];
#pragma unroll
    for (int c = 0; c < TS_Q40_COLS; c++)
#pragma unroll
        for (int r = 0; r < TS_Q40_ROW_TILE; r++)
            acc[c][r] = 0.0f;

    for (int k = threadIdx.x; k < in_dim; k += blockDim.x)
    {
        int block_offset = (k / 32) * 18;
        int lane = k & 31;
        int packed_index = lane & 15;
        int high = lane >> 4;

        // Unpack the columns' weight at element k ONCE.
        float wv[TS_Q40_COLS];
#pragma unroll
        for (int c = 0; c < TS_Q40_COLS; c++)
        {
            if (c < ncols)
            {
                const uint8_t* w = weights + (size_t)(out_col0 + c) * row_bytes + block_offset;
                float d = __half2float(*reinterpret_cast<const half*>(w));
                uint8_t packed = w[2 + packed_index];
                int q = (high ? (packed >> 4) : (packed & 0x0F)) - 8;
                wv[c] = d * (float)q;
            }
            else
                wv[c] = 0.0f;
        }

        // Reuse it across the tile's rows (activations are L2-resident).
#pragma unroll
        for (int r = 0; r < TS_Q40_ROW_TILE; r++)
        {
            if (r < tile)
            {
                float x = input[(size_t)(row0 + r) * in_dim + k];
#pragma unroll
                for (int c = 0; c < TS_Q40_COLS; c++)
                    acc[c][r] += wv[c] * x;
            }
        }
    }

    // tile / ncols are block-uniform, so every thread runs the same set of
    // block_reduce_sum calls (each has a __syncthreads); compile-time c/r keep
    // acc in registers.
#pragma unroll
    for (int c = 0; c < TS_Q40_COLS; c++)
    {
#pragma unroll
        for (int r = 0; r < TS_Q40_ROW_TILE; r++)
        {
            if (c < ncols && r < tile)
            {
                float s = block_reduce_sum(acc[c][r]);
                if (threadIdx.x == 0)
                    output[(size_t)(row0 + r) * out_dim + out_col0 + c] = s;
                __syncthreads();
            }
        }
    }
}

extern "C" __global__ void ts_quant_matmul_q8_0_single_f32(
    const uint8_t* weights,
    const float* input,
    float* output,
    int in_dim,
    int out_dim,
    int rows)
{
    const int cols_per_block = 4;
    int out_col0 = blockIdx.x * cols_per_block;
    int row = blockIdx.y;
    if (out_col0 >= out_dim || row >= rows)
        return;

    int row_bytes = (in_dim / 32) * 34;
    const float* x_row = input + (size_t)row * in_dim;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    for (int k = threadIdx.x; k < in_dim; k += blockDim.x)
    {
        int block_offset = (k / 32) * 34;
        int lane = k & 31;
        float x = x_row[k];

        const uint8_t* w0 = weights + (size_t)(out_col0 + 0) * row_bytes + block_offset;
        float d0 = __half2float(*reinterpret_cast<const half*>(w0));
        int8_t q0 = reinterpret_cast<const int8_t*>(w0 + 2)[lane];
        acc0 += d0 * (float)q0 * x;

        if (out_col0 + 1 < out_dim)
        {
            const uint8_t* w1 = weights + (size_t)(out_col0 + 1) * row_bytes + block_offset;
            float d1 = __half2float(*reinterpret_cast<const half*>(w1));
            int8_t q1 = reinterpret_cast<const int8_t*>(w1 + 2)[lane];
            acc1 += d1 * (float)q1 * x;
        }

        if (out_col0 + 2 < out_dim)
        {
            const uint8_t* w2 = weights + (size_t)(out_col0 + 2) * row_bytes + block_offset;
            float d2 = __half2float(*reinterpret_cast<const half*>(w2));
            int8_t q2 = reinterpret_cast<const int8_t*>(w2 + 2)[lane];
            acc2 += d2 * (float)q2 * x;
        }

        if (out_col0 + 3 < out_dim)
        {
            const uint8_t* w3 = weights + (size_t)(out_col0 + 3) * row_bytes + block_offset;
            float d3 = __half2float(*reinterpret_cast<const half*>(w3));
            int8_t q3 = reinterpret_cast<const int8_t*>(w3 + 2)[lane];
            acc3 += d3 * (float)q3 * x;
        }
    }

    acc0 = block_reduce_sum(acc0);
    if (threadIdx.x == 0)
        output[(size_t)row * out_dim + out_col0] = acc0;
    __syncthreads();

    acc1 = block_reduce_sum(acc1);
    if (threadIdx.x == 0 && out_col0 + 1 < out_dim)
        output[(size_t)row * out_dim + out_col0 + 1] = acc1;
    __syncthreads();

    acc2 = block_reduce_sum(acc2);
    if (threadIdx.x == 0 && out_col0 + 2 < out_dim)
        output[(size_t)row * out_dim + out_col0 + 2] = acc2;
    __syncthreads();

    acc3 = block_reduce_sum(acc3);
    if (threadIdx.x == 0 && out_col0 + 3 < out_dim)
        output[(size_t)row * out_dim + out_col0 + 3] = acc3;
}

extern "C" __global__ void ts_quant_matmul_q8_0_f32(
    const uint8_t* weights,
    const float* input,
    float* output,
    int in_dim,
    int out_dim,
    int rows)
{
    const int cols_per_block = 4;
    int out_col0 = blockIdx.x * cols_per_block;

    int row0 = blockIdx.y * 4;
    if (out_col0 >= out_dim || row0 >= rows)
        return;

    int row_bytes = (in_dim / 32) * 34;
    bool has_r1 = row0 + 1 < rows;
    bool has_r2 = row0 + 2 < rows;
    bool has_r3 = row0 + 3 < rows;

    float acc00 = 0.0f, acc01 = 0.0f, acc02 = 0.0f, acc03 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f, acc12 = 0.0f, acc13 = 0.0f;
    float acc20 = 0.0f, acc21 = 0.0f, acc22 = 0.0f, acc23 = 0.0f;
    float acc30 = 0.0f, acc31 = 0.0f, acc32 = 0.0f, acc33 = 0.0f;
    for (int k = threadIdx.x; k < in_dim; k += blockDim.x)
    {
        int block_offset = (k / 32) * 34;
        int lane = k & 31;
        float x0 = input[(size_t)(row0 + 0) * in_dim + k];
        float x1 = has_r1 ? input[(size_t)(row0 + 1) * in_dim + k] : 0.0f;
        float x2 = has_r2 ? input[(size_t)(row0 + 2) * in_dim + k] : 0.0f;
        float x3 = has_r3 ? input[(size_t)(row0 + 3) * in_dim + k] : 0.0f;

        const uint8_t* w0 = weights + (size_t)(out_col0 + 0) * row_bytes + block_offset;
        float d0 = __half2float(*reinterpret_cast<const half*>(w0));
        int8_t q0 = reinterpret_cast<const int8_t*>(w0 + 2)[lane];
        float wv0 = d0 * (float)q0;
        acc00 += wv0 * x0;
        acc10 += wv0 * x1;
        acc20 += wv0 * x2;
        acc30 += wv0 * x3;

        if (out_col0 + 1 < out_dim)
        {
            const uint8_t* w1 = weights + (size_t)(out_col0 + 1) * row_bytes + block_offset;
            float d1 = __half2float(*reinterpret_cast<const half*>(w1));
            int8_t q1 = reinterpret_cast<const int8_t*>(w1 + 2)[lane];
            float wv1 = d1 * (float)q1;
            acc01 += wv1 * x0;
            acc11 += wv1 * x1;
            acc21 += wv1 * x2;
            acc31 += wv1 * x3;
        }

        if (out_col0 + 2 < out_dim)
        {
            const uint8_t* w2 = weights + (size_t)(out_col0 + 2) * row_bytes + block_offset;
            float d2 = __half2float(*reinterpret_cast<const half*>(w2));
            int8_t q2 = reinterpret_cast<const int8_t*>(w2 + 2)[lane];
            float wv2 = d2 * (float)q2;
            acc02 += wv2 * x0;
            acc12 += wv2 * x1;
            acc22 += wv2 * x2;
            acc32 += wv2 * x3;
        }

        if (out_col0 + 3 < out_dim)
        {
            const uint8_t* w3 = weights + (size_t)(out_col0 + 3) * row_bytes + block_offset;
            float d3 = __half2float(*reinterpret_cast<const half*>(w3));
            int8_t q3 = reinterpret_cast<const int8_t*>(w3 + 2)[lane];
            float wv3 = d3 * (float)q3;
            acc03 += wv3 * x0;
            acc13 += wv3 * x1;
            acc23 += wv3 * x2;
            acc33 += wv3 * x3;
        }
    }

    acc00 = block_reduce_sum(acc00);
    if (threadIdx.x == 0)
        output[(size_t)(row0 + 0) * out_dim + out_col0] = acc00;
    __syncthreads();

    acc01 = block_reduce_sum(acc01);
    if (threadIdx.x == 0 && out_col0 + 1 < out_dim)
        output[(size_t)(row0 + 0) * out_dim + out_col0 + 1] = acc01;
    __syncthreads();

    acc02 = block_reduce_sum(acc02);
    if (threadIdx.x == 0 && out_col0 + 2 < out_dim)
        output[(size_t)(row0 + 0) * out_dim + out_col0 + 2] = acc02;
    __syncthreads();

    acc03 = block_reduce_sum(acc03);
    if (threadIdx.x == 0 && out_col0 + 3 < out_dim)
        output[(size_t)(row0 + 0) * out_dim + out_col0 + 3] = acc03;
    __syncthreads();

    acc10 = block_reduce_sum(acc10);
    if (threadIdx.x == 0 && has_r1)
        output[(size_t)(row0 + 1) * out_dim + out_col0] = acc10;
    __syncthreads();

    acc11 = block_reduce_sum(acc11);
    if (threadIdx.x == 0 && has_r1 && out_col0 + 1 < out_dim)
        output[(size_t)(row0 + 1) * out_dim + out_col0 + 1] = acc11;
    __syncthreads();

    acc12 = block_reduce_sum(acc12);
    if (threadIdx.x == 0 && has_r1 && out_col0 + 2 < out_dim)
        output[(size_t)(row0 + 1) * out_dim + out_col0 + 2] = acc12;
    __syncthreads();

    acc13 = block_reduce_sum(acc13);
    if (threadIdx.x == 0 && has_r1 && out_col0 + 3 < out_dim)
        output[(size_t)(row0 + 1) * out_dim + out_col0 + 3] = acc13;
    __syncthreads();

    acc20 = block_reduce_sum(acc20);
    if (threadIdx.x == 0 && has_r2)
        output[(size_t)(row0 + 2) * out_dim + out_col0] = acc20;
    __syncthreads();

    acc21 = block_reduce_sum(acc21);
    if (threadIdx.x == 0 && has_r2 && out_col0 + 1 < out_dim)
        output[(size_t)(row0 + 2) * out_dim + out_col0 + 1] = acc21;
    __syncthreads();

    acc22 = block_reduce_sum(acc22);
    if (threadIdx.x == 0 && has_r2 && out_col0 + 2 < out_dim)
        output[(size_t)(row0 + 2) * out_dim + out_col0 + 2] = acc22;
    __syncthreads();

    acc23 = block_reduce_sum(acc23);
    if (threadIdx.x == 0 && has_r2 && out_col0 + 3 < out_dim)
        output[(size_t)(row0 + 2) * out_dim + out_col0 + 3] = acc23;
    __syncthreads();

    acc30 = block_reduce_sum(acc30);
    if (threadIdx.x == 0 && has_r3)
        output[(size_t)(row0 + 3) * out_dim + out_col0] = acc30;
    __syncthreads();

    acc31 = block_reduce_sum(acc31);
    if (threadIdx.x == 0 && has_r3 && out_col0 + 1 < out_dim)
        output[(size_t)(row0 + 3) * out_dim + out_col0 + 1] = acc31;
    __syncthreads();

    acc32 = block_reduce_sum(acc32);
    if (threadIdx.x == 0 && has_r3 && out_col0 + 2 < out_dim)
        output[(size_t)(row0 + 3) * out_dim + out_col0 + 2] = acc32;
    __syncthreads();

    acc33 = block_reduce_sum(acc33);
    if (threadIdx.x == 0 && has_r3 && out_col0 + 3 < out_dim)
        output[(size_t)(row0 + 3) * out_dim + out_col0 + 3] = acc33;
}

// Quantize a batch of activation rows to q8_1 ONCE into a global scratch, so every
// output-tile block of the dp4a GEMM reads them (L2-cached) instead of re-quantizing.
extern "C" __global__ void ts_quantize_q8_1_rows_f32(
    const float* input,
    ts_block_q8_1* out,
    int in_dim,
    int rows)
{
    int q8_blocks = in_dim / TS_QK8_1;
    long total = (long)rows * q8_blocks;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    int r = (int)(idx / q8_blocks);
    int qb = (int)(idx - (long)r * q8_blocks);
    quantize_q8_1_block(input + (size_t)r * in_dim + (size_t)qb * TS_QK8_1, out + idx);
}

// Block-tile dp4a (int8-MMA) Q8_0 GEMM — the fast multi-row path for the MTP verify
// window (rows 2-8). The scalar block-reduce kernels above are compute-bound on the
// big FFN matmuls (measured ~78% of verify GPU time). This kernel:
//   * 256 threads compute a TS_Q8_DP4A_ROWS x TS_Q8_DP4A_COLS output tile;
//   * reads the pre-quantized q8_1 activations (xq) from global (L2-cached; quantized
//     once by ts_quantize_q8_1_rows_f32), weight read once per row-tile;
//   * each thread strides the dp4a-GROUPS (4 elements) of in_dim — full parallelism
//     even for small in_dim (gate_up). Q8_0 is symmetric so the per-32-block scale
//     d_w*d_act is constant within a block and can be applied per group (exact);
//   * a SINGLE fused block reduction combines all ROWS*COLS partials (one
//     __syncthreads, vs the scalar kernel's 16 sequential block reductions).
// dp4a does 4 int8 MACs/instruction. Numerically ~equal to the dequant-weight x
// f32-activation path; the only difference is the q8_1 round-trip of the activation
// (8-bit, same as ggml's mul_mat_q), well within FP noise.
#define TS_Q8_DP4A_ROWS 4
#define TS_Q8_DP4A_COLS 4
extern "C" __global__ void ts_quant_matmul_q8_0_dp4a_f32(
    const uint8_t* weights,
    const ts_block_q8_1* xq,
    float* output,
    int in_dim,
    int out_dim,
    int rows)
{
    int out_col0 = blockIdx.x * TS_Q8_DP4A_COLS;
    int row0 = blockIdx.y * TS_Q8_DP4A_ROWS;
    if (out_col0 >= out_dim || row0 >= rows || (in_dim & 31) != 0)
        return;

    int q8_blocks = in_dim / TS_QK8_1;
    int row_bytes = q8_blocks * 34;
    int tile_rows = min(TS_Q8_DP4A_ROWS, rows - row0);

    float partial[TS_Q8_DP4A_ROWS][TS_Q8_DP4A_COLS];
#pragma unroll
    for (int r = 0; r < TS_Q8_DP4A_ROWS; r++)
#pragma unroll
        for (int c = 0; c < TS_Q8_DP4A_COLS; c++)
            partial[r][c] = 0.0f;

    int total_groups = q8_blocks * 8;   // in_dim / 4 dp4a groups
    for (int g = threadIdx.x; g < total_groups; g += blockDim.x)
    {
        int ib = g >> 3;
        int gib = g & 7;

        // Load each row's activation group + scale ONCE (reused across all columns) —
        // the activation is identical for every output column of this tile.
        int   a4[TS_Q8_DP4A_ROWS];
        float dact[TS_Q8_DP4A_ROWS];
#pragma unroll
        for (int r = 0; r < TS_Q8_DP4A_ROWS; r++)
        {
            if (r >= tile_rows) continue;
            const ts_block_q8_1* ablk = &xq[(size_t)(row0 + r) * q8_blocks + ib];
            a4[r] = get_int_b4(ablk->qs, gib);
            dact[r] = __half2float(ablk->d);
        }

#pragma unroll
        for (int c = 0; c < TS_Q8_DP4A_COLS; c++)
        {
            int col = out_col0 + c;
            if (col >= out_dim)
                continue;
            const uint8_t* wblk = weights + (size_t)col * row_bytes + (size_t)ib * 34;
            float dw = __half2float(*reinterpret_cast<const half*>(wblk));
            // qs at wblk+2 is 2-byte aligned (block stride 34 is even) -> read as two
            // uint16 (get_int_b2) instead of 4 byte loads (read_u32_unaligned).
            int w4 = get_int_b2(wblk + 2, gib);
#pragma unroll
            for (int r = 0; r < TS_Q8_DP4A_ROWS; r++)
            {
                if (r >= tile_rows)
                    continue;
                int s = dp4a_i8(w4, a4[r], 0);
                partial[r][c] += dw * dact[r] * (float)s;
            }
        }
    }

    // Single fused block reduction of all TS_Q8_DP4A_ROWS*COLS partials: warp-reduce
    // each (no sync), stash per warp, ONE __syncthreads, warp 0 combines + writes.
    const int NRC = TS_Q8_DP4A_ROWS * TS_Q8_DP4A_COLS;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    __shared__ float red[(512 / 32) * NRC];
#pragma unroll
    for (int r = 0; r < TS_Q8_DP4A_ROWS; r++)
#pragma unroll
        for (int c = 0; c < TS_Q8_DP4A_COLS; c++)
        {
            float v = partial[r][c];
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xFFFFFFFF, v, off);
            if (lane == 0)
                red[warp * NRC + r * TS_Q8_DP4A_COLS + c] = v;
        }
    __syncthreads();
    if (warp == 0)
    {
#pragma unroll
        for (int rc = 0; rc < NRC; rc++)
        {
            float v = (lane < num_warps) ? red[lane * NRC + rc] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xFFFFFFFF, v, off);
            if (lane == 0)
            {
                int r = rc / TS_Q8_DP4A_COLS;
                int c = rc - r * TS_Q8_DP4A_COLS;
                int col = out_col0 + c;
                if (r < tile_rows && col < out_dim)
                    output[(size_t)(row0 + r) * out_dim + col] = v;
            }
        }
    }
}

// dp4a (int8) Q4_0 GEMM — the fast path for BOTH single-token decode (rows == 1)
// and the MTP verify window (rows 2-9) on the dominant dense quant. Mirrors the
// Q8_0 dp4a kernel above (256 threads compute a ROWS x COLS output tile from the
// pre-quantized q8_1 activations) but unpacks Q4_0 nibbles and carries the -8
// zero-point through the q8_1 block sum, exactly like ggml's vec_dot_q4_0_q8_1:
//   value_i = (nibble_i - 8) * d_w,  act_i = q8_i * d_act
//   sum_i value_i*act_i = d_w * ( d_act * dp4a(nibbles, q8) - 8 * s_act )
// where s_act = d_act * sum(q8) is the q8_1 block's stored 's'. Each block's 4
// weight ints carry the low (q8[0..15]) and high (q8[16..31]) nibble halves; the
// -8 correction is applied once per block (at the j==0 weight int). Replaces the
// scalar FP32 dequant matmul (which read Q4_0 weights at ~26 GB/s on the LM head);
// dp4a does 4 int8 MACs/instruction so this is ~memory-bound, matching ggml's
// mul_mat_vec_q. Numerically within FP noise of the dequant path (8-bit activation
// round-trip only).
#define TS_Q40_DP4A_ROWS 4
#define TS_Q40_DP4A_COLS 4
extern "C" __global__ void ts_quant_matmul_q4_0_dp4a_f32(
    const uint8_t* weights,
    const ts_block_q8_1* xq,
    float* output,
    int in_dim,
    int out_dim,
    int rows)
{
    int out_col0 = blockIdx.x * TS_Q40_DP4A_COLS;
    int row0 = blockIdx.y * TS_Q40_DP4A_ROWS;
    if (out_col0 >= out_dim || row0 >= rows || (in_dim & 31) != 0)
        return;

    int q8_blocks = in_dim / TS_QK8_1;   // 32 values per block
    int row_bytes = q8_blocks * 18;       // Q4_0 block = 2-byte d + 16-byte qs
    int tile_rows = min(TS_Q40_DP4A_ROWS, rows - row0);

    float partial[TS_Q40_DP4A_ROWS][TS_Q40_DP4A_COLS];
#pragma unroll
    for (int r = 0; r < TS_Q40_DP4A_ROWS; r++)
#pragma unroll
        for (int c = 0; c < TS_Q40_DP4A_COLS; c++)
            partial[r][c] = 0.0f;

    // One iteration per Q4_0 weight int (4 per 32-block) -> full thread occupancy
    // even for the small in_dim matmuls (qkv / gate_up / LM head, in_dim = 3840).
    int total_wints = q8_blocks * 4;
    for (int gw = threadIdx.x; gw < total_wints; gw += blockDim.x)
    {
        int ib = gw >> 2;     // 32-value block
        int j = gw & 3;       // weight int within the block (covers 4 low + 4 high values)

        // Activation halves for this block, per row: int j (q8 values [4j..4j+3], the
        // LOW nibbles) and int j+4 (q8 values [16+4j..], the HIGH nibbles), plus the
        // per-block scale and (once, at j==0) the sum term for the -8 correction.
        int alo[TS_Q40_DP4A_ROWS], ahi[TS_Q40_DP4A_ROWS];
        float dact[TS_Q40_DP4A_ROWS], sact[TS_Q40_DP4A_ROWS];
#pragma unroll
        for (int r = 0; r < TS_Q40_DP4A_ROWS; r++)
        {
            if (r >= tile_rows) continue;
            const ts_block_q8_1* ablk = &xq[(size_t)(row0 + r) * q8_blocks + ib];
            alo[r] = get_int_b4(ablk->qs, j);
            ahi[r] = get_int_b4(ablk->qs, j + 4);
            dact[r] = __half2float(ablk->d);
            if (j == 0) sact[r] = __half2float(ablk->s);
        }

#pragma unroll
        for (int c = 0; c < TS_Q40_DP4A_COLS; c++)
        {
            int col = out_col0 + c;
            if (col >= out_dim)
                continue;
            const uint8_t* wblk = weights + (size_t)col * row_bytes + (size_t)ib * 18;
            float dw = __half2float(*reinterpret_cast<const half*>(wblk));
            // qs starts at wblk+2 (2-byte aligned, block stride 18 is even); read its
            // j-th 4-byte int as two uint16 to stay aligned.
            int w = get_int_b2(wblk + 2, j);
            int wlo = (w >> 0) & 0x0F0F0F0F;
            int whi = (w >> 4) & 0x0F0F0F0F;
#pragma unroll
            for (int r = 0; r < TS_Q40_DP4A_ROWS; r++)
            {
                if (r >= tile_rows)
                    continue;
                int s = dp4a_i8(wlo, alo[r], 0);
                s = dp4a_i8(whi, ahi[r], s);
                partial[r][c] += dw * dact[r] * (float)s;
                if (j == 0)
                    partial[r][c] += -8.0f * dw * sact[r];
            }
        }
    }

    const int NRC = TS_Q40_DP4A_ROWS * TS_Q40_DP4A_COLS;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;
    __shared__ float red[(512 / 32) * NRC];
#pragma unroll
    for (int r = 0; r < TS_Q40_DP4A_ROWS; r++)
#pragma unroll
        for (int c = 0; c < TS_Q40_DP4A_COLS; c++)
        {
            float v = partial[r][c];
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xFFFFFFFF, v, off);
            if (lane == 0)
                red[warp * NRC + r * TS_Q40_DP4A_COLS + c] = v;
        }
    __syncthreads();
    if (warp == 0)
    {
#pragma unroll
        for (int rc = 0; rc < NRC; rc++)
        {
            float v = (lane < num_warps) ? red[lane * NRC + rc] : 0.0f;
            for (int off = 16; off > 0; off >>= 1)
                v += __shfl_down_sync(0xFFFFFFFF, v, off);
            if (lane == 0)
            {
                int r = rc / TS_Q40_DP4A_COLS;
                int c = rc - r * TS_Q40_DP4A_COLS;
                int col = out_col0 + c;
                if (r < tile_rows && col < out_dim)
                    output[(size_t)(row0 + r) * out_dim + col] = v;
            }
        }
    }
}

// Tensor-core (wmma int8 MMA) Q8_0 GEMM: output[M,N] = act[M,K] x weight[N,K]^T,
// weight Q8_0 (int8 + per-32-block f16 scale), act pre-quantized to q8_1 (xq, int8 +
// per-block scale). One WARP computes a 16x16 (M-tile x N-tile) output tile. M<16 is
// padded with zeros (verify window is small); the int8 m16n16k16 MMA does 16 rows
// regardless. Per Q8_0 32-block (= 2 k16 MMAs) the int32 dot is exact within the block
// (one scale), so we accumulate int32 for the block, then scale element (m,n) by
// d_w[n,block] * d_act[m,block] into a float accumulator (the scale is constant within
// the block). Numerically equals the dp4a path (same q8_1 quantization + int dot).
#define TS_MMA_TILE 16
extern "C" __global__ void ts_quant_matmul_q8_0_mma_f32(
    const uint8_t* weights,
    const ts_block_q8_1* xq,
    float* output,
    int in_dim,
    int out_dim,
    int rows)
{
    using namespace nvcuda;
    int n0 = blockIdx.x * TS_MMA_TILE;
    int m0 = blockIdx.y * TS_MMA_TILE;
    if (n0 >= out_dim || m0 >= rows)
        return;

    int q8_blocks = in_dim / TS_QK8_1;        // 32 elems / block
    int row_bytes = q8_blocks * 34;           // Q8_0 row stride
    int lane = threadIdx.x & 31;

    __shared__ int8_t smem_a[TS_MMA_TILE * TS_MMA_TILE];   // act tile [m][k]
    __shared__ int8_t smem_b[TS_MMA_TILE * TS_MMA_TILE];   // weight tile [n][k]
    __shared__ int    smem_i32[TS_MMA_TILE * TS_MMA_TILE]; // block int32 dot [m][n]
    __shared__ float  smem_facc[TS_MMA_TILE * TS_MMA_TILE];// float accumulator [m][n]
    __shared__ float  smem_dw[TS_MMA_TILE];                // weight block scales (per n)
    __shared__ float  smem_dact[TS_MMA_TILE];              // act block scales (per m)

    for (int i = lane; i < TS_MMA_TILE * TS_MMA_TILE; i += 32)
        smem_facc[i] = 0.0f;
    __syncwarp();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, int8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, int8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, int> acc_frag;

    for (int b = 0; b < q8_blocks; b++)
    {
        wmma::fill_fragment(acc_frag, 0);

        // 32 elems/block = 2 k16 MMA steps; accumulate int32 within the block.
        for (int k16 = 0; k16 < 2; k16++)
        {
            int koff = k16 * 16;
            for (int i = lane; i < TS_MMA_TILE * TS_MMA_TILE; i += 32)
            {
                int r = i >> 4;          // m for A, n for B
                int kk = i & 15;
                // A: act[m0+r][block b, koff+kk]
                int am = m0 + r;
                smem_a[i] = (am < rows)
                    ? xq[(size_t)am * q8_blocks + b].qs[koff + kk] : (int8_t)0;
                // B: weight[n0+r][block b, koff+kk]  (int8 at +2 in the 34B block)
                int bn = n0 + r;
                smem_b[i] = (bn < out_dim)
                    ? (int8_t)weights[(size_t)bn * row_bytes + (size_t)b * 34 + 2 + koff + kk] : (int8_t)0;
            }
            __syncwarp();
            wmma::load_matrix_sync(a_frag, smem_a, 16);
            wmma::load_matrix_sync(b_frag, smem_b, 16);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            __syncwarp();
        }

        wmma::store_matrix_sync(smem_i32, acc_frag, 16, wmma::mem_row_major);

        // Per-block scales: d_w[n] (weight f16 scale at the block start), d_act[m].
        for (int i = lane; i < TS_MMA_TILE; i += 32)
        {
            int bn = n0 + i;
            smem_dw[i] = (bn < out_dim)
                ? __half2float(*reinterpret_cast<const half*>(weights + (size_t)bn * row_bytes + (size_t)b * 34)) : 0.0f;
            int am = m0 + i;
            smem_dact[i] = (am < rows) ? __half2float(xq[(size_t)am * q8_blocks + b].d) : 0.0f;
        }
        __syncwarp();

        for (int i = lane; i < TS_MMA_TILE * TS_MMA_TILE; i += 32)
        {
            int m = i >> 4, n = i & 15;
            smem_facc[i] += (float)smem_i32[i] * smem_dw[n] * smem_dact[m];
        }
        __syncwarp();
    }

    for (int i = lane; i < TS_MMA_TILE * TS_MMA_TILE; i += 32)
    {
        int m = i >> 4, n = i & 15;
        if (m0 + m < rows && n0 + n < out_dim)
            output[(size_t)(m0 + m) * out_dim + (n0 + n)] = smem_facc[i];
    }
}

extern "C" __global__ void ts_quant_get_rows_f32(
    const uint8_t* weights,
    const void* indices,
    float* output,
    int type,
    int cols,
    int rows,
    int indices_are_int32)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    int src_row = indices_are_int32
        ? reinterpret_cast<const int*>(indices)[row]
        : (int)reinterpret_cast<const float*>(indices)[row];
    if (src_row < 0)
        return;

    int row_bytes = qrow_bytes(type, cols);
    const uint8_t* w_row = weights + (size_t)src_row * row_bytes;
    float* out_row = output + (size_t)row * cols;

    for (int col = threadIdx.x; col < cols; col += blockDim.x)
        out_row[col] = qvalue_at(w_row, type, col);
}
