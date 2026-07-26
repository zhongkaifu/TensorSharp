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
#define GGML_IQ4_XS 23
#define TS_QK8_1 32
#define TS_Q80_F16_CHUNK 2048
#define TS_Q80_BLOCK_BYTES 34

// Decode-graph dynamic parameter block (device int32[4], see CudaDecodeDynParams):
// dyn[0] = attend_len (kv length), dyn[1] = kv cache write position,
// dyn[2] = GDN conv ring write index, dyn[3] = RoPE position.
// A captured CUDA graph bakes scalar kernel arguments, so kernels on the
// per-token decode path re-read the position-dependent ones from this device
// block (refreshed each replay by a captured pinned-host->device memcpy node).
#define TS_DYN_ATTEND_LEN 0
#define TS_DYN_KV_WRITE_POS 1
#define TS_DYN_CONV_WRITE_IDX 2
#define TS_DYN_ROPE_POS 3

// IQ4_XS / IQ4_NL non-linear 4-bit codebook (ggml kvalues_iq4nl). Each 4-bit
// index maps to one of these 16 reconstruction levels; the per-sub-block scale
// multiplies the looked-up level. Matches ManagedQuantizedOps.Iq4NlValues so the
// device dequant is bit-for-bit consistent with the host reference.
__device__ static const int8_t ts_kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

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

struct ts_block_iq2_s
{
    half d;
    uint8_t qs[64];
    uint8_t qh[8];
    uint8_t scales[8];
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
        case GGML_IQ4_XS: return (cols / 256) * 136;
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

    if (type == GGML_IQ4_XS)
    {
        // block_iq4_xs: d (half), scales_h (uint16), scales_l[4], qs[128] = 136 bytes.
        // 8 sub-blocks of 32 elements each; per-sub-block 6-bit scale ls is split
        // between the low nibble in scales_l[ib/2] and 2 high bits in scales_h.
        // Within a sub-block, elements 0..15 read the low nibble of qs[j] and
        // 16..31 read the high nibble (ggml dequantize_row_iq4_xs).
        const uint8_t* block = row + (col / 256) * 136;
        float d = __half2float(*reinterpret_cast<const half*>(block));
        int scales_h = (int)block[2] | ((int)block[3] << 8);
        const uint8_t* scales_l = block + 4;
        const uint8_t* qs = block + 8;
        int t = col & 255;
        int ib = t >> 5;             // 0..7 sub-block
        int within = t & 31;         // 0..31 position in sub-block
        int j = within & 15;         // 0..15 byte index within the sub-block's 16 bytes
        int ls = ((scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        float dl = d * (float)(ls - 32);
        uint8_t packed = qs[ib * 16 + j];
        int nib = (within < 16) ? (packed & 0xF) : (packed >> 4);
        return dl * (float)ts_kvalues_iq4nl[nib];
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

// One IQ2_S 32-value group dotted against one q8_1 activation block. This is
// the direct-CUDA equivalent of ggml-cuda's vec_dot_iq2_s_q8_1. IQ2_S stores
// four 8-value grid indices/sign bytes per group and two 4-bit scales; doing
// the lookup once per 8 values and using dp4a avoids the scalar qvalue_at path
// re-reading the same 82-byte super-block metadata for every element.
__device__ __forceinline__ float dot_iq2_s_q8_1(
    const uint8_t* iq_block, const ts_block_q8_1* q8_blocks, int group)
{
    const ts_block_iq2_s* bq2 = reinterpret_cast<const ts_block_iq2_s*>(iq_block);

    // Four low grid-index bytes and four sign bytes for this 32-value group.
    const int qs_packed = get_int_b2(bq2->qs, group);
    const uint8_t* qs = reinterpret_cast<const uint8_t*>(&qs_packed);
    const int qh = bq2->qh[group];
    const int signs_packed = get_int_b2(bq2->qs, 8 + group);
    const uint8_t* signs = reinterpret_cast<const uint8_t*>(&signs_packed);

    const int ls0 = bq2->scales[group] & 0x0F;
    const int ls1 = bq2->scales[group] >> 4;
    int sumi0 = 0;
    int sumi1 = 0;

#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2)
    {
        int grid_index = qs[l0 / 2] | ((qh << (8 - l0)) & 0x300);
        const int* grid_pos = reinterpret_cast<const int*>(iq2s_grid + grid_index);
        uint8_t sign_byte = signs[l0 / 2];

        int signs0 = __vcmpne4(
            ((sign_byte & 0x03) << 7) | ((sign_byte & 0x0C) << 21),
            0x00000000);
        int grid0 = __vsub4(grid_pos[0] ^ signs0, signs0);
        int u0 = get_int_b4(q8_blocks[group].qs, l0 + 0);

        int signs1 = __vcmpne4(
            ((sign_byte & 0x30) << 3) | ((sign_byte & 0xC0) << 17),
            0x00000000);
        int grid1 = __vsub4(grid_pos[1] ^ signs1, signs1);
        int u1 = get_int_b4(q8_blocks[group].qs, l0 + 1);

        if (l0 < 4)
        {
            sumi0 = dp4a_i8(grid0, u0, sumi0);
            sumi0 = dp4a_i8(grid1, u1, sumi0);
        }
        else
        {
            sumi1 = dp4a_i8(grid0, u0, sumi1);
            sumi1 = dp4a_i8(grid1, u1, sumi1);
        }
    }

    // Algebraically (ls + 0.5)/4 for each 16-value half, with the exact
    // integer rounding order used by ggml's IQ2_S CUDA vec-dot.
    int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
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

// Generic 2D strided copy: `rows` rows of `innerBytes` bytes with independent
// src/dst row pitches. Replaces the per-row cuMemcpyDtoDAsync loop the strided
// tensor-copy fallback used to issue (one driver call per row; at prefill
// sizes that was tens of thousands of WDDM submissions per forward pass).
extern "C" __global__ void ts_copy2d_bytes(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    long long rows,
    long long innerBytes,
    long long srcPitch,
    long long dstPitch)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long r = blockIdx.y; r < rows; r += gridDim.y)
    {
        const unsigned char* s = src + r * srcPitch;
        unsigned char* d = dst + r * dstPitch;
        unsigned long long align =
            (unsigned long long)(size_t)s | (unsigned long long)(size_t)d | (unsigned long long)innerBytes;
        if ((align & 15ULL) == 0ULL)
        {
            const int4* s4 = (const int4*)s;
            int4* d4 = (int4*)d;
            long long n = innerBytes >> 4;
            for (long long j = i; j < n; j += stride)
                d4[j] = s4[j];
        }
        else if ((align & 3ULL) == 0ULL)
        {
            const int* s1 = (const int*)s;
            int* d1 = (int*)d;
            long long n = innerBytes >> 2;
            for (long long j = i; j < n; j += stride)
                d1[j] = s1[j];
        }
        else
        {
            for (long long j = i; j < innerBytes; j += stride)
                d[j] = s[j];
        }
    }
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

// Dequantize a whole quantized weight [out_dim, in_dim] to f16 row-major. Feeds
// the tensor-core cuBLAS GEMM used for large-row prefill matmuls: the weight is
// read ONCE here instead of rows/tile times by the block-tile quant kernels.
extern "C" __global__ void ts_dequant_weight_f16(
    const uint8_t* weights, half* output, int type, int in_dim, long long total)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    int row = (int)(i / in_dim);
    int col = (int)(i - (long long)row * in_dim);
    const uint8_t* wrow = weights + (size_t)row * qrow_bytes(type, in_dim);
    output[i] = __float2half(qvalue_at(wrow, type, col));
}

// Q8_0-specialized whole-weight conversion for the large-row cuBLAS path.
// Mirroring ggml_cuda's dequantize_block_q8_0_f16, one warp stages 2048
// quantized values (64 blocks, 2176 raw bytes) exactly once and emits packed
// half2 output. This removes the generic kernel's per-scalar 64-bit row
// division, runtime quant-type branch tree, and repeated block-scale loads.
//
// The multiply remains in FP32 before the F16 round so this is bit-identical to
// ts_dequant_weight_f16's Q8_0 branch, including subnormal scales. DeviceWeight
// allocations carry 16 slack bytes, making the final 4-byte staging load safe
// when an odd number of 34-byte blocks leaves a two-byte raw tail.
extern "C" __global__ void ts_dequant_weight_q8_0_f16(
    const uint8_t* weights, half* output, long long total)
{
    constexpr int q8_blocks_per_chunk = TS_Q80_F16_CHUNK / 32;
    constexpr int raw_bytes_per_chunk = q8_blocks_per_chunk * TS_Q80_BLOCK_BYTES;
    constexpr int raw_ints_per_chunk = raw_bytes_per_chunk / (int)sizeof(int);

    long long elem_base = (long long)blockIdx.x * TS_Q80_F16_CHUNK;
    long long raw_base = (elem_base / 32) * TS_Q80_BLOCK_BYTES;
    long long total_raw_bytes = (total / 32) * TS_Q80_BLOCK_BYTES;
    const int* src = reinterpret_cast<const int*>(weights + raw_base);

    __shared__ int packed[raw_ints_per_chunk];
#pragma unroll
    for (int i = threadIdx.x; i < raw_ints_per_chunk; i += 32)
    {
        long long byte_offset = raw_base + (long long)i * sizeof(int);
        packed[i] = byte_offset < total_raw_bytes ? src[i] : 0;
    }
    __syncthreads();

    const uint8_t* staged = reinterpret_cast<const uint8_t*>(packed);
#pragma unroll
    for (int local = 2 * threadIdx.x; local < TS_Q80_F16_CHUNK; local += 64)
    {
        long long out_idx = elem_base + local;
        if (out_idx >= total)
            break;

        const uint8_t* qblock =
            staged + (local / 32) * TS_Q80_BLOCK_BYTES;
        float d = __half2float(*reinterpret_cast<const half*>(qblock));
        char2 qs = *reinterpret_cast<const char2*>(qblock + 2 + (local & 31));
        half2 result = __floats2half2_rn(d * (float)qs.x, d * (float)qs.y);
        *reinterpret_cast<half2*>(output + out_idx) = result;
    }
}

extern "C" __global__ void ts_convert_f32_f16(const float* src, half* dst, long long count)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count)
        dst[i] = __float2half(src[i]);
}

// Deinterleave a fused Q+gate projection: src rows hold num_heads blocks of
// [q(head_dim) | gate(head_dim)]; q/gate receive the de-interleaved halves as
// dense [rows, num_heads*head_dim]. src_row_stride supports reading a Narrow'd
// column slice out of a wider fused QKV row (decode reuses the packed buffer).
extern "C" __global__ void ts_deinterleave_qgate_f32(
    const float* src, float* q, float* gate,
    int rows, int num_heads, int head_dim, long long src_row_stride)
{
    int per_row = num_heads * head_dim;
    int count = rows * per_row;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count)
        return;

    int row = i / per_row;
    int rem = i - row * per_row;
    int h = rem / head_dim;
    int d = rem - h * head_dim;
    const float* s = src + (size_t)row * src_row_stride + (size_t)h * head_dim * 2;
    q[i] = s[d];
    gate[i] = s[d + head_dim];
}

// Row-strided binary activation: same math as ts_binary_activation_f32 but each
// operand is a 2D row-major view whose rows may be padded (e.g. Narrow'd halves
// of a fused gate+up projection). Keeps strided SwiGLU/GeGLU activations on the
// device instead of falling back to the (element-wise, synchronizing) CPU path.
extern "C" __global__ void ts_binary_activation_strided_f32(
    const float* a, const float* b, float* output,
    int rows, int cols,
    long long a_row_stride, long long b_row_stride, long long out_row_stride,
    int op)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int count = rows * cols;
    if (i >= count)
        return;

    int row = i / cols;
    int col = i - row * cols;
    float x = a[(size_t)row * a_row_stride + col];
    float y = b[(size_t)row * b_row_stride + col];
    float value;
    if (op == 0)
        value = silu(x) * y;
    else if (op == 1)
        value = gelu(x) * y;
    else
        value = x * (1.0f / (1.0f + expf(-y)));
    output[(size_t)row * out_row_stride + col] = value;
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
    float eps,
    const int* dyn)
{
    int h = blockIdx.x;
    if (h >= num_v_heads)
        return;
    if (dyn)
        conv_write_idx = dyn[TS_DYN_CONV_WRITE_IDX];

    // ONE block per head. The head's recurrent state is staged into shared memory
    // once, the whole sequence recurs on-chip, and the final state is written back
    // once. This removes the per-token global read+write of the full state (the
    // dominant GDN prefill traffic) AND the earlier multi-block-per-head layout,
    // whose per-block whole-state decay and partially-filled core[] were racy for
    // head_v_dim > num_warps.
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = nthreads >> 5;

    // Dynamic shared layout: q | k | core | state[head_v_dim * head_k_dim].
    extern __shared__ float scratch[];
    float* q = scratch;
    float* k = q + head_k_dim;
    float* core = k + head_k_dim;
    float* state = core + head_v_dim;

    __shared__ float q_scale;
    __shared__ float k_scale;
    __shared__ float gate_h;
    __shared__ float beta_h;
    __shared__ float rms_inv;

    int src_h = h % num_k_heads;
    int q_offset = src_h * head_k_dim;
    int k_offset = qk_dim + src_h * head_k_dim;
    int v_offset = 2 * qk_dim + h * head_v_dim;
    int z_offset = qkv_dim + h * head_v_dim;
    int beta_offset = qkv_dim + v_dim + h;
    int alpha_offset = qkv_dim + v_dim + num_v_heads + h;
    int state_per_head = head_v_dim * head_k_dim;
    float* state_global = ssm_state + (size_t)h * state_per_head;
    float q_head_scale = rsqrtf((float)head_v_dim);

    for (int i = tid; i < state_per_head; i += nthreads)
        state[i] = state_global[i];
    __syncthreads();

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

        // Normalize the shared q/k and decay the (shared-resident) state in place.
        float state_scale = expf(gate_h);
        for (int d = tid; d < head_k_dim; d += nthreads)
        {
            q[d] *= q_scale;
            k[d] *= k_scale;
        }
        for (int i = tid; i < state_per_head; i += nthreads)
            state[i] *= state_scale;
        __syncthreads();

        // Rows are mutually independent; each warp walks rows warp, warp+num_warps, ...
        // kv = <state_row, k>, rank-1 update, core = <state_row, q>.
        float beta = beta_h;
        for (int row = warp; row < head_v_dim; row += num_warps)
        {
            float* state_row = state + (size_t)row * head_k_dim;
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

    for (int i = tid; i < state_per_head; i += nthreads)
        state_global[i] = state[i];
}

// ====================================================================
// Split GDN prefill (3 sync-free phases). The single-kernel path above walks
// the sequence with ~7 block-wide barriers per token and only num_v_heads
// blocks in flight, so a 255-token prefill is latency-bound (~5 ms/layer).
// The delta-net rows of a head are mutually independent across the WHOLE
// sequence; only the shared conv/norm inputs (parallel over tokens) and the
// output RMS (parallel over tokens) couple rows. Phases:
//   conv:  per (token, head) warp - causal conv + L2-normalized q/k, v,
//          gate/beta scalars -> scratch  [fully parallel]
//   scan:  per warp: 4 state rows register-resident across the whole window,
//          sequential over tokens but NO block synchronization; writes the
//          un-normalized core outputs  [num_v_heads*head_v_dim/4 warps]
//   out:   per (token, head) warp - RMS over core, ssm_norm & silu(z) gate
//          [fully parallel]
// Requires head_k_dim == 128 (state row = 4 regs/lane) and head_v_dim % 32
// == 0; the host falls back to the single-kernel path otherwise.
// ====================================================================
extern "C" __global__ void ts_qwen35_gdn_prefill_conv_f32(
    const float* packed,
    const float* conv_state,
    const float* conv_w,
    const float* dt_bias,
    const float* a_log,
    float* scr,           // [num_v_heads][win_len][2*head_k + head_v + 2]
    int win_start,
    int win_len,
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
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= win_len * num_v_heads)
        return;
    int h = wid % num_v_heads;
    int sl = wid / num_v_heads;
    int s = win_start + sl;

    int src_h = h % num_k_heads;
    int q_offset = src_h * head_k_dim;
    int k_offset = qk_dim + src_h * head_k_dim;
    int v_offset = 2 * qk_dim + h * head_v_dim;
    int beta_offset = qkv_dim + v_dim + h;
    int alpha_offset = qkv_dim + v_dim + num_v_heads + h;

    int stride = 2 * head_k_dim + head_v_dim + 2;
    float* dst = scr + ((size_t)h * win_len + sl) * stride;

    // Causal conv + L2 norm for q/k (head_k_dim <= 128 -> up to 4 dims/lane).
    float qv[4], kv[4];
    float q_sum = 0.0f, k_sum = 0.0f;
    int nj = head_k_dim >> 5;
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
        if (j < nj)
        {
            int d = lane + (j << 5);
            qv[j] = qwen35_gdn_conv_channel(
                packed, conv_state, conv_w, s, q_offset + d,
                seq_len, packed_dim, qkv_dim, conv_kernel, conv_write_idx);
            kv[j] = qwen35_gdn_conv_channel(
                packed, conv_state, conv_w, s, k_offset + d,
                seq_len, packed_dim, qkv_dim, conv_kernel, conv_write_idx);
            q_sum += qv[j] * qv[j];
            k_sum += kv[j] * kv[j];
        }
    }
    q_sum = warp_allreduce_sum(q_sum);
    k_sum = warp_allreduce_sum(k_sum);
    float q_scale = rsqrtf(q_sum + eps) * rsqrtf((float)head_v_dim);
    float k_scale = rsqrtf(k_sum + eps);
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
        if (j < nj)
        {
            int d = lane + (j << 5);
            dst[d] = qv[j] * q_scale;
            dst[head_k_dim + d] = kv[j] * k_scale;
        }
    }

    for (int d = lane; d < head_v_dim; d += 32)
        dst[2 * head_k_dim + d] = qwen35_gdn_conv_channel(
            packed, conv_state, conv_w, s, v_offset + d,
            seq_len, packed_dim, qkv_dim, conv_kernel, conv_write_idx);

    if (lane == 0)
    {
        const float* prow = packed + (size_t)s * packed_dim;
        dst[2 * head_k_dim + head_v_dim] =
            softplus_f32(prow[alpha_offset] + dt_bias[h]) * a_log[h];
        dst[2 * head_k_dim + head_v_dim + 1] = sigmoid_f32(prow[beta_offset]);
    }
}

extern "C" __global__ void ts_qwen35_gdn_prefill_scan_f32(
    const float* scr,     // phase-conv output
    float* ssm_state,     // [num_v_heads][head_v_dim][head_k_dim]
    float* core,          // [num_v_heads][win_len][head_v_dim]
    int win_len,
    int num_v_heads,
    int head_k_dim,       // must be 128
    int head_v_dim)
{
    int h = blockIdx.x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int rowsPerBlock = (blockDim.x >> 5) * 4;
    int r0 = blockIdx.y * rowsPerBlock + warp * 4;
    if (h >= num_v_heads || r0 >= head_v_dim)
        return;

    int stride = 2 * head_k_dim + head_v_dim + 2;
    const float* base = scr + (size_t)h * win_len * stride;
    float* state_head = ssm_state + (size_t)h * head_v_dim * head_k_dim;
    float* core_head = core + (size_t)h * win_len * head_v_dim;

    int nrows = head_v_dim - r0;
    if (nrows > 4) nrows = 4;

    // Register-resident state rows: st[ri][j] = state[r0+ri][lane + 32j].
    float st[4][4];
#pragma unroll
    for (int ri = 0; ri < 4; ri++)
#pragma unroll
        for (int j = 0; j < 4; j++)
            st[ri][j] = (ri < nrows)
                ? state_head[(size_t)(r0 + ri) * head_k_dim + lane + (j << 5)] : 0.0f;

    for (int s = 0; s < win_len; s++)
    {
        const float* row = base + (size_t)s * stride;
        float q0 = row[lane], q1 = row[lane + 32], q2 = row[lane + 64], q3 = row[lane + 96];
        float k0 = row[head_k_dim + lane], k1 = row[head_k_dim + lane + 32];
        float k2 = row[head_k_dim + lane + 64], k3 = row[head_k_dim + lane + 96];
        float decay = expf(row[2 * head_k_dim + head_v_dim]);
        float beta = row[2 * head_k_dim + head_v_dim + 1];

#pragma unroll
        for (int ri = 0; ri < 4; ri++)
        {
            if (ri >= nrows)
                break;
            st[ri][0] *= decay; st[ri][1] *= decay; st[ri][2] *= decay; st[ri][3] *= decay;
            float kvdot = st[ri][0] * k0 + st[ri][1] * k1 + st[ri][2] * k2 + st[ri][3] * k3;
            kvdot = warp_allreduce_sum(kvdot);
            float vr = row[2 * head_k_dim + r0 + ri];
            float delta = (vr - kvdot) * beta;
            st[ri][0] += k0 * delta; st[ri][1] += k1 * delta;
            st[ri][2] += k2 * delta; st[ri][3] += k3 * delta;
            float cv = st[ri][0] * q0 + st[ri][1] * q1 + st[ri][2] * q2 + st[ri][3] * q3;
            cv = warp_allreduce_sum(cv);
            if (lane == 0)
                core_head[(size_t)s * head_v_dim + r0 + ri] = cv;
        }
    }

#pragma unroll
    for (int ri = 0; ri < 4; ri++)
#pragma unroll
        for (int j = 0; j < 4; j++)
            if (ri < nrows)
                state_head[(size_t)(r0 + ri) * head_k_dim + lane + (j << 5)] = st[ri][j];
}

extern "C" __global__ void ts_qwen35_gdn_prefill_out_f32(
    const float* core,
    const float* packed,
    const float* ssm_norm,
    float* output,        // [seq_len][v_dim]
    int win_start,
    int win_len,
    int packed_dim,
    int qkv_dim,
    int v_dim,
    int num_v_heads,
    int head_v_dim,
    float eps)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= win_len * num_v_heads)
        return;
    int h = wid % num_v_heads;
    int sl = wid / num_v_heads;

    const float* ch = core + ((size_t)h * win_len + sl) * head_v_dim;
    float sum_sq = 0.0f;
    for (int r = lane; r < head_v_dim; r += 32)
    {
        float c = ch[r];
        sum_sq += c * c;
    }
    sum_sq = warp_allreduce_sum(sum_sq);
    float rms_inv = rsqrtf(sum_sq / (float)head_v_dim + eps);

    int s = win_start + sl;
    const float* prow = packed + (size_t)s * packed_dim;
    float* orow = output + (size_t)s * v_dim + h * head_v_dim;
    for (int r = lane; r < head_v_dim; r += 32)
        orow[r] = ch[r] * rms_inv * ssm_norm[r] * silu(prow[qkv_dim + h * head_v_dim + r]);
}

extern "C" __global__ void ts_qwen35_gdn_update_conv_state_f32(
    const float* packed,
    float* conv_state,
    int seq_len,
    int packed_dim,
    int qkv_dim,
    int conv_dim,
    int conv_write_idx,
    const int* dyn)
{
    int tail = seq_len < conv_dim ? seq_len : conv_dim;
    int total = tail * qkv_dim;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total)
        return;
    if (dyn)
        conv_write_idx = dyn[TS_DYN_CONV_WRITE_IDX];

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
// place (global full-attention verify ÔÇö kv_len <= kv_stride logical positions).
__device__ __forceinline__ float ts_gqa_prefill_cache_to_float(float v)
{
    return v;
}

__device__ __forceinline__ float ts_gqa_prefill_cache_to_float(half v)
{
    return __half2float(v);
}

template <typename cache_t>
__device__ __forceinline__ float ts_gqa_prefill_warp_dot(
    const float* q,
    const cache_t* k,
    int head_dim)
{
    int lane = threadIdx.x & 31;
    float dot = 0.0f;
    for (int d = lane; d < head_dim; d += 32)
        dot += q[d] * ts_gqa_prefill_cache_to_float(k[d]);

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
    return dot;
}

// Gemma 4 E4B's local-attention shape is fixed at four Q heads per KV head,
// d=256 and a <=512-token sliding window.  Tile four adjacent query positions
// into one CTA as well as all four heads in a GQA group.  The old specialization
// used one CTA per query position and consequently re-read the same 512 K/V rows
// four times.  This tile loads each K/V element once and applies it to eight
// query rows, which is the same reuse principle as a small flash-attention tile.
//
// The compact shared score matrix contains 4 heads * 2 queries and at most
// window+1 key positions.  At the production 512-token window that is 16.1 KiB,
// comfortably below the 48-KiB per-block limit on the oldest supported devices.
#define TS_GQA_GROUP4_Q_TILE 2
template <typename cache_t>
__device__ __forceinline__ void ts_gqa_prefill_attention_group4_d256_impl(
    const float* query,
    const cache_t* key,
    const cache_t* value,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int kv_len,
    int head_dim,
    int mask_start,
    int window_size,
    float scale,
    int kv_stride,
    float* scores)
{
    constexpr int group_size = 4;
    constexpr int fixed_head_dim = 256;

    int kv_head = blockIdx.x;
    int q_start = blockIdx.y * TS_GQA_GROUP4_Q_TILE;
    if (kv_head >= num_kv_heads || q_start >= seq_len ||
        num_q_heads != num_kv_heads * group_size ||
        head_dim != fixed_head_dim || window_size <= 0)
    {
        return;
    }

    int q_count = min(TS_GQA_GROUP4_Q_TILE, seq_len - q_start);
    int first_visible = mask_start + q_start;
    int last_visible = first_visible + q_count - 1;
    int min_visible = max(0, first_visible - window_size + 1);
    int max_visible = min(last_visible, kv_len - 1);
    if (min_visible > max_visible)
        return;

    int score_count = max_visible - min_visible + 1;
    int score_stride = min(kv_len, window_size + TS_GQA_GROUP4_Q_TILE - 1);
    int q_head_base = kv_head * group_size;
    int score_rows = q_count * group_size;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    // One warp owns one key at a time.  A lane loads K[d] once and applies it
    // to the 4-head x up-to-4-query tile before independent warp reductions.
    for (int t = warp; t < score_count; t += num_warps)
    {
        int k_pos = min_visible + t;
        const cache_t* k =
            key + ((size_t)kv_head * kv_stride + k_pos) * fixed_head_dim;
        float dots[TS_GQA_GROUP4_Q_TILE * group_size] = { 0.0f };
#pragma unroll
        for (int d = lane; d < fixed_head_dim; d += 32)
        {
            float kv = ts_gqa_prefill_cache_to_float(k[d]);
#pragma unroll
            for (int qi = 0; qi < TS_GQA_GROUP4_Q_TILE; qi++)
            {
                if (qi >= q_count)
                    continue;
#pragma unroll
                for (int h = 0; h < group_size; h++)
                {
                    const float* q =
                        query + ((size_t)(q_head_base + h) * seq_len + q_start + qi) * fixed_head_dim;
                    dots[qi * group_size + h] += q[d] * kv;
                }
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
#pragma unroll
            for (int r = 0; r < TS_GQA_GROUP4_Q_TILE * group_size; r++)
                dots[r] += __shfl_down_sync(0xFFFFFFFF, dots[r], offset);
        }
        if (lane == 0)
        {
#pragma unroll
            for (int qi = 0; qi < TS_GQA_GROUP4_Q_TILE; qi++)
            {
                if (qi >= q_count)
                    continue;
                int visible = first_visible + qi;
                int row_min = max(0, visible - window_size + 1);
                bool allowed = k_pos >= row_min && k_pos <= min(visible, kv_len - 1);
#pragma unroll
                for (int h = 0; h < group_size; h++)
                {
                    int r = qi * group_size + h;
                    scores[r * score_stride + t] =
                        allowed ? dots[r] * scale : -FLT_MAX;
                }
            }
        }
    }
    __syncthreads();

    // Normalize all score rows concurrently. Store normalized probabilities so
    // the V phase avoids reapplying inv_sum for every output dimension.
    for (int r = warp; r < score_rows; r += num_warps)
    {
        float* row = scores + r * score_stride;
        float max_v = -FLT_MAX;
        for (int t = lane; t < score_count; t += 32)
            max_v = fmaxf(max_v, row[t]);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            max_v = fmaxf(max_v, __shfl_down_sync(0xFFFFFFFF, max_v, offset));
        max_v = __shfl_sync(0xFFFFFFFF, max_v, 0);

        float sum = 0.0f;
        for (int t = lane; t < score_count; t += 32)
        {
            float p = expf(row[t] - max_v);
            row[t] = p;
            sum += p;
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        sum = __shfl_sync(0xFFFFFFFF, sum, 0);
        float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int t = lane; t < score_count; t += 32)
            row[t] *= inv_sum;
    }
    __syncthreads();

    // d=256 matches the 256-thread block, so every thread owns one V column.
    // It fetches V[d] once per key and updates the full query/head tile.
    int d = threadIdx.x;
    if (d < fixed_head_dim)
    {
        float acc[TS_GQA_GROUP4_Q_TILE * group_size] = { 0.0f };
        const cache_t* v =
            value + ((size_t)kv_head * kv_stride + min_visible) * fixed_head_dim + d;
        for (int t = 0; t < score_count; t++, v += fixed_head_dim)
        {
            float vv = ts_gqa_prefill_cache_to_float(*v);
#pragma unroll
            for (int r = 0; r < TS_GQA_GROUP4_Q_TILE * group_size; r++)
            {
                if (r < score_rows)
                    acc[r] += scores[r * score_stride + t] * vv;
            }
        }

#pragma unroll
        for (int qi = 0; qi < TS_GQA_GROUP4_Q_TILE; qi++)
        {
            if (qi >= q_count)
                continue;
            size_t out_base =
                ((size_t)(q_start + qi) * num_q_heads + q_head_base) * fixed_head_dim + d;
#pragma unroll
            for (int h = 0; h < group_size; h++)
                output[out_base + (size_t)h * fixed_head_dim] = acc[qi * group_size + h];
        }
    }
}

extern "C" __global__ void ts_gqa_prefill_attention_group4_d256_f32(
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
    extern __shared__ float scores[];
    ts_gqa_prefill_attention_group4_d256_impl(
        query, key, value, output, num_q_heads, num_kv_heads,
        seq_len, kv_len, head_dim, mask_start, window_size, scale, kv_stride,
        scores);
}

extern "C" __global__ void ts_gqa_prefill_attention_group4_d256_f16(
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
    extern __shared__ float scores[];
    ts_gqa_prefill_attention_group4_d256_impl(
        query, key, value, output, num_q_heads, num_kv_heads,
        seq_len, kv_len, head_dim, mask_start, window_size, scale, kv_stride,
        scores);
}

// Gemma 4 global attention uses the same four-query-head GQA group as local
// attention, but d=512 and no sliding window. One CTA computes all four query
// heads for a (query position, KV head) pair, so every K/V value is fetched once
// instead of four times by the generic one-CTA-per-query-head kernel. The score
// workspace is 4*kv_len floats; dispatch caps kv_len at 2048 (32 KiB).
template <typename cache_t>
__device__ __forceinline__ void ts_gqa_prefill_attention_group4_d512_impl(
    const float* query,
    const cache_t* key,
    const cache_t* value,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int kv_len,
    int head_dim,
    int mask_start,
    int window_size,
    float scale,
    int kv_stride,
    float* scores)
{
    constexpr int group_size = 4;
    constexpr int fixed_head_dim = 512;

    int kv_head = blockIdx.x;
    int q_pos = blockIdx.y;
    if (kv_head >= num_kv_heads || q_pos >= seq_len ||
        num_q_heads != num_kv_heads * group_size ||
        head_dim != fixed_head_dim || window_size != 0 || kv_len > 2048)
    {
        return;
    }

    int visible = min(mask_start + q_pos, kv_len - 1);
    int score_count = visible + 1;
    if (score_count <= 0)
        return;

    int q_head_base = kv_head * group_size;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    // One warp owns one key at a time. K[d] is shared across all four query
    // heads before the four independent warp reductions.
    for (int k_pos = warp; k_pos < score_count; k_pos += num_warps)
    {
        const cache_t* k =
            key + ((size_t)kv_head * kv_stride + k_pos) * fixed_head_dim;
        float dots[group_size] = { 0.0f };
#pragma unroll
        for (int d = lane; d < fixed_head_dim; d += 32)
        {
            float kv = ts_gqa_prefill_cache_to_float(k[d]);
#pragma unroll
            for (int h = 0; h < group_size; h++)
            {
                const float* q =
                    query + ((size_t)(q_head_base + h) * seq_len + q_pos) * fixed_head_dim;
                dots[h] += q[d] * kv;
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
#pragma unroll
            for (int h = 0; h < group_size; h++)
                dots[h] += __shfl_down_sync(0xFFFFFFFF, dots[h], offset);
        }
        if (lane == 0)
        {
#pragma unroll
            for (int h = 0; h < group_size; h++)
                scores[h * kv_len + k_pos] = dots[h] * scale;
        }
    }
    __syncthreads();

    // Four warps normalize the four query-head score rows concurrently.
    if (warp < group_size)
    {
        float* row = scores + warp * kv_len;
        float max_v = -FLT_MAX;
        for (int k_pos = lane; k_pos < score_count; k_pos += 32)
            max_v = fmaxf(max_v, row[k_pos]);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            max_v = fmaxf(max_v, __shfl_down_sync(0xFFFFFFFF, max_v, offset));
        max_v = __shfl_sync(0xFFFFFFFF, max_v, 0);

        float sum = 0.0f;
        for (int k_pos = lane; k_pos < score_count; k_pos += 32)
        {
            float p = expf(row[k_pos] - max_v);
            row[k_pos] = p;
            sum += p;
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        sum = __shfl_sync(0xFFFFFFFF, sum, 0);
        float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
        for (int k_pos = lane; k_pos < score_count; k_pos += 32)
            row[k_pos] *= inv_sum;
    }
    __syncthreads();

    // Each thread owns two output dimensions. V[d] is reused for all four
    // query heads, cutting the dominant global-memory traffic by 4x.
    for (int d = threadIdx.x; d < fixed_head_dim; d += blockDim.x)
    {
        float acc[group_size] = { 0.0f };
        const cache_t* v =
            value + (size_t)kv_head * kv_stride * fixed_head_dim + d;
        for (int k_pos = 0; k_pos < score_count; k_pos++, v += fixed_head_dim)
        {
            float vv = ts_gqa_prefill_cache_to_float(*v);
#pragma unroll
            for (int h = 0; h < group_size; h++)
                acc[h] += scores[h * kv_len + k_pos] * vv;
        }

        size_t out_base =
            ((size_t)q_pos * num_q_heads + q_head_base) * fixed_head_dim + d;
#pragma unroll
        for (int h = 0; h < group_size; h++)
            output[out_base + (size_t)h * fixed_head_dim] = acc[h];
    }
}

extern "C" __global__ void ts_gqa_prefill_attention_group4_d512_f32(
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
    extern __shared__ float scores[];
    ts_gqa_prefill_attention_group4_d512_impl(
        query, key, value, output, num_q_heads, num_kv_heads,
        seq_len, kv_len, head_dim, mask_start, window_size, scale, kv_stride,
        scores);
}

extern "C" __global__ void ts_gqa_prefill_attention_group4_d512_f16(
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
    extern __shared__ float scores[];
    ts_gqa_prefill_attention_group4_d512_impl(
        query, key, value, output, num_q_heads, num_kv_heads,
        seq_len, kv_len, head_dim, mask_start, window_size, scale, kv_stride,
        scores);
}

// Long-context d=512 variant. One warp owns one query head and maintains a
// numerically stable online softmax while walking the visible K/V rows. This
// removes the 4*kv_len score workspace (and its 2,048-token shared-memory
// ceiling) while keeping Q and the output accumulators resident in registers.
// The four warps read the same GQA K/V row together, so the duplicate loads hit
// the same cache lines even though each warp advances its own softmax state.
template <typename cache_t>
__device__ __forceinline__ void ts_gqa_prefill_attention_group4_online_d512_impl(
    const float* query,
    const cache_t* key,
    const cache_t* value,
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
    constexpr int group_size = 4;
    constexpr int fixed_head_dim = 512;
    constexpr int values_per_lane = fixed_head_dim / 32;

    const int kv_head = blockIdx.x;
    const int q_pos = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (kv_head >= num_kv_heads || q_pos >= seq_len ||
        warp >= group_size ||
        num_q_heads != num_kv_heads * group_size ||
        head_dim != fixed_head_dim || window_size != 0)
    {
        return;
    }

    const int visible = min(mask_start + q_pos, kv_len - 1);
    if (visible < 0)
        return;

    const int q_head = kv_head * group_size + warp;
    const float* q =
        query + ((size_t)q_head * seq_len + q_pos) * fixed_head_dim;
    float q_values[values_per_lane];
    float acc[values_per_lane] = { 0.0f };
#pragma unroll
    for (int i = 0; i < values_per_lane; i++)
        q_values[i] = q[lane + i * 32];

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;
    for (int k_pos = 0; k_pos <= visible; k_pos++)
    {
        const cache_t* k =
            key + ((size_t)kv_head * kv_stride + k_pos) * fixed_head_dim;
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < values_per_lane; i++)
            dot = fmaf(q_values[i],
                ts_gqa_prefill_cache_to_float(k[lane + i * 32]), dot);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
        dot = __shfl_sync(0xFFFFFFFF, dot, 0) * scale;

        const float next_max = fmaxf(running_max, dot);
        const float old_scale =
            running_max == -FLT_MAX ? 0.0f : expf(running_max - next_max);
        const float new_scale = expf(dot - next_max);
        const cache_t* v =
            value + ((size_t)kv_head * kv_stride + k_pos) * fixed_head_dim;
#pragma unroll
        for (int i = 0; i < values_per_lane; i++)
        {
            const float vv =
                ts_gqa_prefill_cache_to_float(v[lane + i * 32]);
            acc[i] = fmaf(new_scale, vv, old_scale * acc[i]);
        }
        running_sum = fmaf(old_scale, running_sum, new_scale);
        running_max = next_max;
    }

    const float inv_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    float* out =
        output + ((size_t)q_pos * num_q_heads + q_head) * fixed_head_dim;
#pragma unroll
    for (int i = 0; i < values_per_lane; i++)
        out[lane + i * 32] = acc[i] * inv_sum;
}

extern "C" __global__ void ts_gqa_prefill_attention_group4_online_d512_f32(
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
    ts_gqa_prefill_attention_group4_online_d512_impl(
        query, key, value, output, num_q_heads, num_kv_heads,
        seq_len, kv_len, head_dim, mask_start, window_size, scale, kv_stride);
}

extern "C" __global__ void ts_gqa_prefill_attention_group4_online_d512_f16(
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
    ts_gqa_prefill_attention_group4_online_d512_impl(
        query, key, value, output, num_q_heads, num_kv_heads,
        seq_len, kv_len, head_dim, mask_start, window_size, scale, kv_stride);
}

// ---------------------------------------------------------------------------
// Flash-style tiled GQA prefill attention (f16 K/V, group-of-4 query heads).
//
// The older group4 prefill kernels above process at most 4 queries per CTA and
// re-read the query vectors from GLOBAL memory for every visible key, so a
// 2048-token Gemma prefill spent 6.7 ms per SWA layer / 26 ms per global layer
// on this GPU. This kernel follows the ggml_cuda flash-attention structure
// instead:
//   * one CTA owns a (kv_head, QROWS-query) tile; its 4*QROWS score rows share
//     every K/V row the CTA reads;
//   * the Q tile is staged in shared memory ONCE (f32, no precision change);
//   * softmax is the numerically stable online variant (running max m and
//     normalizer l per row), so no kv_len-sized score workspace exists and the
//     kernel has no window-size or kv_len ceiling from shared memory;
//   * K is read with half2 loads by a warp per (key, query) task; V is read
//     once per CTA with thread-per-column coalesced rows.
// Numerics match the two-pass kernels to FP-reassociation order (same
// f16->f32 promotion of K/V, f32 accumulation, exp in f32).
//
// The causal/SWA mask matches ts_gqa_prefill_attention_group4_d256_impl:
// row qi attends k in [max(0, mask_start+q0+qi-window+1), min(mask_start+q0+qi,
// kv_len-1)] for window>0, and [0, min(mask_start+q0+qi, kv_len-1)] for
// window==0. Requires blockDim.x == 256.
#define TS_FLASH_KCHUNK 32

// Two consecutive cache values as floats (half2 load for f16, float2 for f32),
// so the flash kernel below can run on both the f16 KV cache and the current
// chunk's still-f32 K/V tensors with identical structure.
__device__ __forceinline__ float2 ts_flash_load2(const half* p)
{
    return __half22float2(*reinterpret_cast<const half2*>(p));
}

__device__ __forceinline__ float2 ts_flash_load2(const float* p)
{
    return *reinterpret_cast<const float2*>(p);
}

__device__ __forceinline__ float ts_flash_load1(const half* p)
{
    return __half2float(*p);
}

__device__ __forceinline__ float ts_flash_load1(const float* p)
{
    return *p;
}

template <int HEAD_DIM, int QROWS, typename cache_t>
__device__ __forceinline__ void ts_gqa_prefill_flash_group4_impl(
    const float* __restrict__ query,
    const cache_t* __restrict__ key,
    const cache_t* __restrict__ value,
    const float* __restrict__ sinks,
    float* __restrict__ output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int kv_len,
    int head_dim,
    int mask_start,
    int window_size,
    float scale,
    int kv_stride,
    int has_sinks)
{
    constexpr int group_size = 4;
    constexpr int ROWS = QROWS * group_size;
    constexpr int COLS_PER_THREAD = HEAD_DIM / 256;
    constexpr int half_dim = HEAD_DIM / 2;

    int kv_head = blockIdx.x;
    int q0 = blockIdx.y * QROWS;
    if (kv_head >= num_kv_heads || q0 >= seq_len ||
        num_q_heads != num_kv_heads * group_size || head_dim != HEAD_DIM)
    {
        return;
    }

    int q_count = min(QROWS, seq_len - q0);
    int q_head_base = kv_head * group_size;

    extern __shared__ float ws[];
    float* q_s = ws;                                   // ROWS * HEAD_DIM
    float* scores = q_s + ROWS * HEAD_DIM;             // ROWS * TS_FLASH_KCHUNK
    float* m_s = scores + ROWS * TS_FLASH_KCHUNK;      // ROWS
    float* l_s = m_s + ROWS;                           // ROWS
    float* alpha_s = l_s + ROWS;                       // ROWS

    for (int i = threadIdx.x; i < ROWS * HEAD_DIM; i += blockDim.x)
    {
        int r = i / HEAD_DIM;
        int qi = r >> 2;
        int h = r & 3;
        int d = i - r * HEAD_DIM;
        q_s[i] = qi < q_count
            ? query[((size_t)(q_head_base + h) * seq_len + q0 + qi) * HEAD_DIM + d]
            : 0.0f;
    }
    if (threadIdx.x < ROWS)
    {
        // An attention sink is one extra per-head logit in the softmax
        // denominator with no V row: seed the running max with it and the
        // normalizer with its exp(sink - m) == 1. Every later chunk's alpha
        // rescaling then carries exp(sink - m_final) forward exactly like the
        // two-pass sinks kernels' explicit term.
        float m0 = -FLT_MAX;
        float l0 = 0.0f;
        if (has_sinks)
        {
            m0 = sinks[q_head_base + (threadIdx.x & 3)];
            l0 = 1.0f;
        }
        m_s[threadIdx.x] = m0;
        l_s[threadIdx.x] = l0;
    }
    __syncthreads();

    int first_visible = mask_start + q0;
    int lo = window_size > 0 ? max(0, first_visible - window_size + 1) : 0;
    int hi = min(first_visible + q_count - 1, kv_len - 1);

    float o_acc[ROWS * COLS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ROWS * COLS_PER_THREAD; i++)
        o_acc[i] = 0.0f;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    for (int kc = lo; kc <= hi; kc += TS_FLASH_KCHUNK)
    {
        int chunk_len = min(TS_FLASH_KCHUNK, hi - kc + 1);

        // Phase A: one warp per (key, query) task; the query's 4 heads share
        // the half2 K row the warp streams in.
        int tasks = chunk_len * QROWS;
        for (int t = warp; t < tasks; t += num_warps)
        {
            int ki = t % chunk_len;
            int qi = t / chunk_len;
            int k_pos = kc + ki;
            const cache_t* krow =
                key + ((size_t)kv_head * kv_stride + k_pos) * HEAD_DIM;
            const float* qrow = q_s + (size_t)(qi * group_size) * HEAD_DIM;
            float dot0 = 0.0f;
            float dot1 = 0.0f;
            float dot2 = 0.0f;
            float dot3 = 0.0f;
#pragma unroll
            for (int d2 = lane; d2 < half_dim; d2 += 32)
            {
                float2 kf = ts_flash_load2(krow + d2 * 2);
                int d = d2 * 2;
                dot0 = fmaf(qrow[0 * HEAD_DIM + d], kf.x, dot0);
                dot0 = fmaf(qrow[0 * HEAD_DIM + d + 1], kf.y, dot0);
                dot1 = fmaf(qrow[1 * HEAD_DIM + d], kf.x, dot1);
                dot1 = fmaf(qrow[1 * HEAD_DIM + d + 1], kf.y, dot1);
                dot2 = fmaf(qrow[2 * HEAD_DIM + d], kf.x, dot2);
                dot2 = fmaf(qrow[2 * HEAD_DIM + d + 1], kf.y, dot2);
                dot3 = fmaf(qrow[3 * HEAD_DIM + d], kf.x, dot3);
                dot3 = fmaf(qrow[3 * HEAD_DIM + d + 1], kf.y, dot3);
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                dot0 += __shfl_down_sync(0xFFFFFFFF, dot0, offset);
                dot1 += __shfl_down_sync(0xFFFFFFFF, dot1, offset);
                dot2 += __shfl_down_sync(0xFFFFFFFF, dot2, offset);
                dot3 += __shfl_down_sync(0xFFFFFFFF, dot3, offset);
            }
            if (lane == 0)
            {
                int visible = first_visible + qi;
                int row_lo = window_size > 0
                    ? max(0, visible - window_size + 1)
                    : 0;
                bool allowed = qi < q_count &&
                    k_pos >= row_lo &&
                    k_pos <= min(visible, kv_len - 1);
                int r = qi * group_size;
                scores[(r + 0) * TS_FLASH_KCHUNK + ki] =
                    allowed ? dot0 * scale : -FLT_MAX;
                scores[(r + 1) * TS_FLASH_KCHUNK + ki] =
                    allowed ? dot1 * scale : -FLT_MAX;
                scores[(r + 2) * TS_FLASH_KCHUNK + ki] =
                    allowed ? dot2 * scale : -FLT_MAX;
                scores[(r + 3) * TS_FLASH_KCHUNK + ki] =
                    allowed ? dot3 * scale : -FLT_MAX;
            }
        }
        __syncthreads();

        // Phase B: per-row online-softmax state update; one warp per row, one
        // score lane each (TS_FLASH_KCHUNK == warp width).
        for (int r = warp; r < ROWS; r += num_warps)
        {
            float sc = lane < chunk_len
                ? scores[r * TS_FLASH_KCHUNK + lane]
                : -FLT_MAX;
            float mx = sc;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, offset));
            float m_old = m_s[r];
            float m_new = fmaxf(m_old, mx);
            // Masked lanes (sc == -FLT_MAX) underflow expf to exactly 0.
            float p = (lane < chunk_len && m_new > -FLT_MAX)
                ? expf(sc - m_new)
                : 0.0f;
            if (lane < chunk_len)
                scores[r * TS_FLASH_KCHUNK + lane] = p;
            float sum = p;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
            if (lane == 0)
            {
                float alpha = expf(m_old - m_new);
                alpha_s[r] = alpha;
                m_s[r] = m_new;
                l_s[r] = l_s[r] * alpha + sum;
            }
        }
        __syncthreads();

        // Phase C: rescale accumulators by this chunk's alpha, then add the
        // chunk's probability-weighted V rows. Thread t owns output columns
        // t, t+256, ... so each V row is one (or two) coalesced 512 B reads.
#pragma unroll
        for (int r = 0; r < ROWS; r++)
        {
            float a = alpha_s[r];
#pragma unroll
            for (int c = 0; c < COLS_PER_THREAD; c++)
                o_acc[r * COLS_PER_THREAD + c] *= a;
        }
        for (int ki = 0; ki < chunk_len; ki++)
        {
            const cache_t* vrow =
                value + ((size_t)kv_head * kv_stride + kc + ki) * HEAD_DIM;
            float vv[COLS_PER_THREAD];
#pragma unroll
            for (int c = 0; c < COLS_PER_THREAD; c++)
                vv[c] = ts_flash_load1(vrow + threadIdx.x + c * 256);
#pragma unroll
            for (int r = 0; r < ROWS; r++)
            {
                float p = scores[r * TS_FLASH_KCHUNK + ki];
#pragma unroll
                for (int c = 0; c < COLS_PER_THREAD; c++)
                    o_acc[r * COLS_PER_THREAD + c] =
                        fmaf(p, vv[c], o_acc[r * COLS_PER_THREAD + c]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++)
    {
        int qi = r >> 2;
        if (qi >= q_count)
            continue;
        int h = r & 3;
        float l = l_s[r];
        float inv = l > 0.0f ? 1.0f / l : 0.0f;
        size_t out_base =
            ((size_t)(q0 + qi) * num_q_heads + q_head_base + h) * HEAD_DIM;
#pragma unroll
        for (int c = 0; c < COLS_PER_THREAD; c++)
            output[out_base + threadIdx.x + c * 256] =
                o_acc[r * COLS_PER_THREAD + c] * inv;
    }
}

// ---------------------------------------------------------------------------
// flash2: flash1's exact warp-cooperative online-softmax structure, but with a
// LARGER K chunk (KC=64 vs 32). flash1 was neither FLOP- nor occupancy-bound
// (doubling occupancy via a smaller Q tile changed nothing); it is bound by the
// SERIAL per-chunk loop -- three __syncthreads plus a cross-chunk online-softmax
// dependency, repeated once per chunk. Halving the chunk count (KC 32->64) halves
// those sync rounds. Phase B folds KC/32 scores per lane so a warp still owns a
// full row. Numerics are byte-for-byte flash1 (f32 Q, f32 accumulation); only the
// score-reassociation width changes, which stays within the f16/f32 tolerances.
template <int HEAD_DIM, int QROWS, int KC, typename cache_t>
__device__ __forceinline__ void ts_gqa_prefill_flash2_group4_impl(
    const float* __restrict__ query,
    const cache_t* __restrict__ key,
    const cache_t* __restrict__ value,
    const float* __restrict__ sinks,
    float* __restrict__ output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int kv_len,
    int head_dim,
    int mask_start,
    int window_size,
    float scale,
    int kv_stride,
    int has_sinks)
{
    constexpr int group_size = 4;
    constexpr int ROWS = QROWS * group_size;
    constexpr int COLS_PER_THREAD = HEAD_DIM / 256;
    constexpr int half_dim = HEAD_DIM / 2;

    int kv_head = blockIdx.x;
    int q0 = blockIdx.y * QROWS;
    if (kv_head >= num_kv_heads || q0 >= seq_len ||
        num_q_heads != num_kv_heads * group_size || head_dim != HEAD_DIM)
    {
        return;
    }

    int q_count = min(QROWS, seq_len - q0);
    int q_head_base = kv_head * group_size;

    extern __shared__ float ws[];
    float* q_s = ws;                                   // ROWS * HEAD_DIM
    float* scores = q_s + ROWS * HEAD_DIM;             // ROWS * KC
    float* m_s = scores + ROWS * KC;                   // ROWS
    float* l_s = m_s + ROWS;                           // ROWS
    float* alpha_s = l_s + ROWS;                       // ROWS

    for (int i = threadIdx.x; i < ROWS * HEAD_DIM; i += blockDim.x)
    {
        int r = i / HEAD_DIM;
        int qi = r >> 2;
        int h = r & 3;
        int d = i - r * HEAD_DIM;
        q_s[i] = qi < q_count
            ? query[((size_t)(q_head_base + h) * seq_len + q0 + qi) * HEAD_DIM + d]
            : 0.0f;
    }
    if (threadIdx.x < ROWS)
    {
        float m0 = -FLT_MAX;
        float l0 = 0.0f;
        if (has_sinks)
        {
            m0 = sinks[q_head_base + (threadIdx.x & 3)];
            l0 = 1.0f;
        }
        m_s[threadIdx.x] = m0;
        l_s[threadIdx.x] = l0;
    }
    __syncthreads();

    int first_visible = mask_start + q0;
    int lo = window_size > 0 ? max(0, first_visible - window_size + 1) : 0;
    int hi = min(first_visible + q_count - 1, kv_len - 1);

    float o_acc[ROWS * COLS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ROWS * COLS_PER_THREAD; i++)
        o_acc[i] = 0.0f;

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    for (int kc = lo; kc <= hi; kc += KC)
    {
        int chunk_len = min(KC, hi - kc + 1);

        // Phase A: warp per (key, query) task; the query's 4 heads share the
        // half2/float2 K row the warp streams in — byte-for-byte flash1.
        int tasks = chunk_len * QROWS;
        for (int t = warp; t < tasks; t += num_warps)
        {
            int ki = t % chunk_len;
            int qi = t / chunk_len;
            int k_pos = kc + ki;
            const cache_t* krow =
                key + ((size_t)kv_head * kv_stride + k_pos) * HEAD_DIM;
            const float* qrow = q_s + (size_t)(qi * group_size) * HEAD_DIM;
            float dot0 = 0.0f, dot1 = 0.0f, dot2 = 0.0f, dot3 = 0.0f;
#pragma unroll
            for (int d2 = lane; d2 < half_dim; d2 += 32)
            {
                float2 kf = ts_flash_load2(krow + d2 * 2);
                int d = d2 * 2;
                dot0 = fmaf(qrow[0 * HEAD_DIM + d], kf.x, dot0);
                dot0 = fmaf(qrow[0 * HEAD_DIM + d + 1], kf.y, dot0);
                dot1 = fmaf(qrow[1 * HEAD_DIM + d], kf.x, dot1);
                dot1 = fmaf(qrow[1 * HEAD_DIM + d + 1], kf.y, dot1);
                dot2 = fmaf(qrow[2 * HEAD_DIM + d], kf.x, dot2);
                dot2 = fmaf(qrow[2 * HEAD_DIM + d + 1], kf.y, dot2);
                dot3 = fmaf(qrow[3 * HEAD_DIM + d], kf.x, dot3);
                dot3 = fmaf(qrow[3 * HEAD_DIM + d + 1], kf.y, dot3);
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                dot0 += __shfl_down_sync(0xFFFFFFFF, dot0, offset);
                dot1 += __shfl_down_sync(0xFFFFFFFF, dot1, offset);
                dot2 += __shfl_down_sync(0xFFFFFFFF, dot2, offset);
                dot3 += __shfl_down_sync(0xFFFFFFFF, dot3, offset);
            }
            if (lane == 0)
            {
                int visible = first_visible + qi;
                int row_lo = window_size > 0
                    ? max(0, visible - window_size + 1)
                    : 0;
                bool allowed = qi < q_count &&
                    k_pos >= row_lo &&
                    k_pos <= min(visible, kv_len - 1);
                int r = qi * group_size;
                scores[(r + 0) * KC + ki] = allowed ? dot0 * scale : -FLT_MAX;
                scores[(r + 1) * KC + ki] = allowed ? dot1 * scale : -FLT_MAX;
                scores[(r + 2) * KC + ki] = allowed ? dot2 * scale : -FLT_MAX;
                scores[(r + 3) * KC + ki] = allowed ? dot3 * scale : -FLT_MAX;
            }
        }
        __syncthreads();

        // Phase B: per-row online-softmax update. One warp owns a row; KC can
        // exceed the warp width (fewer, larger chunks = fewer sync rounds, which
        // is what this kernel is bound by), so each lane folds SLOTS scores.
        constexpr int SLOTS = KC / 32;
        for (int r = warp; r < ROWS; r += num_warps)
        {
            float sc[SLOTS];
            float mx = -FLT_MAX;
#pragma unroll
            for (int s = 0; s < SLOTS; s++)
            {
                int c = lane + s * 32;
                sc[s] = c < chunk_len ? scores[r * KC + c] : -FLT_MAX;
                mx = fmaxf(mx, sc[s]);
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFF, mx, offset));
            float m_old = m_s[r];
            float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < SLOTS; s++)
            {
                int c = lane + s * 32;
                float p = (c < chunk_len && m_new > -FLT_MAX) ? expf(sc[s] - m_new) : 0.0f;
                if (c < chunk_len)
                    scores[r * KC + c] = p;
                sum += p;
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
            if (lane == 0)
            {
                float alpha = expf(m_old - m_new);
                alpha_s[r] = alpha;
                m_s[r] = m_new;
                l_s[r] = l_s[r] * alpha + sum;
            }
        }
        __syncthreads();

        // Phase C: rescale + probability-weighted V — identical to flash1.
#pragma unroll
        for (int r = 0; r < ROWS; r++)
        {
            float a = alpha_s[r];
#pragma unroll
            for (int cc = 0; cc < COLS_PER_THREAD; cc++)
                o_acc[r * COLS_PER_THREAD + cc] *= a;
        }
        for (int ki = 0; ki < chunk_len; ki++)
        {
            const cache_t* vrow =
                value + ((size_t)kv_head * kv_stride + kc + ki) * HEAD_DIM;
            float vv[COLS_PER_THREAD];
#pragma unroll
            for (int cc = 0; cc < COLS_PER_THREAD; cc++)
                vv[cc] = ts_flash_load1(vrow + threadIdx.x + cc * 256);
#pragma unroll
            for (int r = 0; r < ROWS; r++)
            {
                float p = scores[r * KC + ki];
#pragma unroll
                for (int cc = 0; cc < COLS_PER_THREAD; cc++)
                    o_acc[r * COLS_PER_THREAD + cc] =
                        fmaf(p, vv[cc], o_acc[r * COLS_PER_THREAD + cc]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS; r++)
    {
        int qi = r >> 2;
        if (qi >= q_count)
            continue;
        int h = r & 3;
        float l = l_s[r];
        float inv = l > 0.0f ? 1.0f / l : 0.0f;
        size_t out_base =
            ((size_t)(q0 + qi) * num_q_heads + q_head_base + h) * HEAD_DIM;
#pragma unroll
        for (int cc = 0; cc < COLS_PER_THREAD; cc++)
            output[out_base + threadIdx.x + cc * 256] =
                o_acc[r * COLS_PER_THREAD + cc] * inv;
    }
}

extern "C" __global__ void __launch_bounds__(256)
ts_gqa_prefill_flash2_group4_d256_f16(
    const float* query, const half* key, const half* value, const float* sinks,
    float* output, int num_q_heads, int num_kv_heads, int seq_len, int kv_len,
    int head_dim, int mask_start, int window_size, float scale, int kv_stride, int has_sinks)
{
    ts_gqa_prefill_flash2_group4_impl<256, 8, 64, half>(
        query, key, value, sinks, output, num_q_heads, num_kv_heads,
        seq_len, kv_len, head_dim, mask_start, window_size, scale, kv_stride, has_sinks);
}

extern "C" __global__ void __launch_bounds__(256)
ts_gqa_prefill_flash2_group4_d256_f32(
    const float* query, const float* key, const float* value, const float* sinks,
    float* output, int num_q_heads, int num_kv_heads, int seq_len, int kv_len,
    int head_dim, int mask_start, int window_size, float scale, int kv_stride, int has_sinks)
{
    ts_gqa_prefill_flash2_group4_impl<256, 8, 64, float>(
        query, key, value, sinks, output, num_q_heads, num_kv_heads,
        seq_len, kv_len, head_dim, mask_start, window_size, scale, kv_stride, has_sinks);
}

extern "C" __global__ void __launch_bounds__(256)
ts_gqa_prefill_flash2_group4_d512_f16(
    const float* query, const half* key, const half* value, const float* sinks,
    float* output, int num_q_heads, int num_kv_heads, int seq_len, int kv_len,
    int head_dim, int mask_start, int window_size, float scale, int kv_stride, int has_sinks)
{
    ts_gqa_prefill_flash2_group4_impl<512, 4, 64, half>(
        query, key, value, sinks, output, num_q_heads, num_kv_heads,
        seq_len, kv_len, head_dim, mask_start, window_size, scale, kv_stride, has_sinks);
}

extern "C" __global__ void __launch_bounds__(256)
ts_gqa_prefill_flash2_group4_d512_f32(
    const float* query, const float* key, const float* value, const float* sinks,
    float* output, int num_q_heads, int num_kv_heads, int seq_len, int kv_len,
    int head_dim, int mask_start, int window_size, float scale, int kv_stride, int has_sinks)
{
    ts_gqa_prefill_flash2_group4_impl<512, 4, 64, float>(
        query, key, value, sinks, output, num_q_heads, num_kv_heads,
        seq_len, kv_len, head_dim, mask_start, window_size, scale, kv_stride, has_sinks);
}

extern "C" __global__ void __launch_bounds__(256)
ts_gqa_prefill_flash_group4_d256_f16(
    const float* query,
    const half* key,
    const half* value,
    const float* sinks,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int kv_len,
    int head_dim,
    int mask_start,
    int window_size,
    float scale,
    int kv_stride,
    int has_sinks)
{
    ts_gqa_prefill_flash_group4_impl<256, 8, half>(
        query, key, value, sinks, output, num_q_heads, num_kv_heads,
        seq_len, kv_len, head_dim, mask_start, window_size, scale, kv_stride,
        has_sinks);
}

extern "C" __global__ void __launch_bounds__(256)
ts_gqa_prefill_flash_group4_d512_f16(
    const float* query,
    const half* key,
    const half* value,
    const float* sinks,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int kv_len,
    int head_dim,
    int mask_start,
    int window_size,
    float scale,
    int kv_stride,
    int has_sinks)
{
    ts_gqa_prefill_flash_group4_impl<512, 4, half>(
        query, key, value, sinks, output, num_q_heads, num_kv_heads,
        seq_len, kv_len, head_dim, mask_start, window_size, scale, kv_stride,
        has_sinks);
}

extern "C" __global__ void __launch_bounds__(256)
ts_gqa_prefill_flash_group4_d256_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* sinks,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int kv_len,
    int head_dim,
    int mask_start,
    int window_size,
    float scale,
    int kv_stride,
    int has_sinks)
{
    ts_gqa_prefill_flash_group4_impl<256, 8, float>(
        query, key, value, sinks, output, num_q_heads, num_kv_heads,
        seq_len, kv_len, head_dim, mask_start, window_size, scale, kv_stride,
        has_sinks);
}

extern "C" __global__ void __launch_bounds__(256)
ts_gqa_prefill_flash_group4_d512_f32(
    const float* query,
    const float* key,
    const float* value,
    const float* sinks,
    float* output,
    int num_q_heads,
    int num_kv_heads,
    int seq_len,
    int kv_len,
    int head_dim,
    int mask_start,
    int window_size,
    float scale,
    int kv_stride,
    int has_sinks)
{
    ts_gqa_prefill_flash_group4_impl<512, 4, float>(
        query, key, value, sinks, output, num_q_heads, num_kv_heads,
        seq_len, kv_len, head_dim, mask_start, window_size, scale, kv_stride,
        has_sinks);
}

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
    int kv_stride,
    int warp_cooperative)
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
    if (warp_cooperative)
    {
        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        int num_warps = blockDim.x >> 5;
        for (int k_pos = min_visible + warp; k_pos <= max_visible; k_pos += num_warps)
        {
            const float* k = key + ((size_t)kv_head * kv_stride + k_pos) * head_dim;
            float dot = ts_gqa_prefill_warp_dot(q, k, head_dim);
            if (lane == 0)
            {
                float score = dot * scale;
                max_v = fmaxf(max_v, score);
                scores[k_pos] = score;
            }
        }
    }
    else
    {
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
    int kv_stride,
    int warp_cooperative)
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
    if (warp_cooperative)
    {
        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        int num_warps = blockDim.x >> 5;
        for (int k_pos = min_visible + warp; k_pos <= max_visible; k_pos += num_warps)
        {
            const half* k = key + ((size_t)kv_head * kv_stride + k_pos) * head_dim;
            float dot = ts_gqa_prefill_warp_dot(q, k, head_dim);
            if (lane == 0)
            {
                float score = dot * scale;
                max_v = fmaxf(max_v, score);
                scores[k_pos] = score;
            }
        }
    }
    else
    {
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
    int has_sinks,
    int warp_cooperative)
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
    if (warp_cooperative)
    {
        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        int num_warps = blockDim.x >> 5;
        for (int k_pos = min_visible + warp; k_pos <= max_visible; k_pos += num_warps)
        {
            const float* k = key_cache + ((size_t)kv_head * cache_size + k_pos) * head_dim;
            float dot = ts_gqa_prefill_warp_dot(q, k, head_dim);
            if (lane == 0)
            {
                float score = dot * scale;
                max_v = fmaxf(max_v, score);
                scores[k_pos] = score;
            }
        }
    }
    else
    {
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
    int has_sinks,
    int warp_cooperative)
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
    if (warp_cooperative)
    {
        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        int num_warps = blockDim.x >> 5;
        for (int k_pos = min_visible + warp; k_pos <= max_visible; k_pos += num_warps)
        {
            const half* k = key_cache + ((size_t)kv_head * cache_size + k_pos) * head_dim;
            float dot = ts_gqa_prefill_warp_dot(q, k, head_dim);
            if (lane == 0)
            {
                float score = dot * scale;
                max_v = fmaxf(max_v, score);
                scores[k_pos] = score;
            }
        }
    }
    else
    {
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
    float scale,
    const int* dyn)
{
    int q_head = blockIdx.x;
    if (q_head >= num_q_heads)
        return;
    if (dyn)
    {
        attend_len = dyn[TS_DYN_ATTEND_LEN];
        if (circular && attend_len > cache_size)
            attend_len = cache_size;
        if (circular)
            attend_start = max(0, dyn[TS_DYN_KV_WRITE_POS] + 1 - attend_len);
    }

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
    float scale,
    const int* dyn)
{
    int q_head = blockIdx.x;
    if (q_head >= num_q_heads)
        return;
    if (dyn)
    {
        attend_len = dyn[TS_DYN_ATTEND_LEN];
        if (circular && attend_len > cache_size)
            attend_len = cache_size;
        if (circular)
            attend_start = max(0, dyn[TS_DYN_KV_WRITE_POS] + 1 - attend_len);
    }

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

#define TS_GQA_DECODE_GROUP4_HEAD_DIM 512
#define TS_GQA_DECODE_GROUP4_HEADS 4

// Gemma 4 local/global attention has four query heads for each KV head. The
// generic decode kernel launches one CTA per query head, so it reads every K/V
// row four times and maps one thread to an entire d=256/d=512 QK dot product
// (strided loads across the warp). These specializations map one warp to a key
// row and share its coalesced K/V loads across all four query heads.
__device__ __forceinline__ void ts_gqa_decode_group4_scores_f16(
    const float* query,
    const half* key_cache,
    int kv_head,
    int cache_size,
    int logical_start,
    int token_count,
    float scale,
    int head_dim,
    int score_stride,
    float* scores,
    float* query_shared,
    float* shared_max,
    float* shared_sum)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warps = blockDim.x >> 5;
    const int q_head_base = kv_head * TS_GQA_DECODE_GROUP4_HEADS;

    for (int i = tid;
         i < TS_GQA_DECODE_GROUP4_HEADS * head_dim;
         i += blockDim.x)
    {
        query_shared[i] =
            query[(size_t)q_head_base * head_dim + i];
    }
    __syncthreads();

    const int half_dim = head_dim >> 1;
    for (int i = warp; i < token_count; i += warps)
    {
        // K rows are head_dim halves (512 B / 1 KiB) apart, so half2 loads stay
        // aligned; a warp pulls 128 B per instruction instead of 64 B.
        const half2* k2 = reinterpret_cast<const half2*>(key_cache +
            ((size_t)kv_head * cache_size + logical_start + i) *
                head_dim);
        float dot0 = 0.0f;
        float dot1 = 0.0f;
        float dot2 = 0.0f;
        float dot3 = 0.0f;
        for (int d2 = lane; d2 < half_dim; d2 += 32)
        {
            float2 kf = __half22float2(k2[d2]);
            int d = d2 * 2;
            dot0 = fmaf(query_shared[0 * head_dim + d], kf.x, dot0);
            dot0 = fmaf(query_shared[0 * head_dim + d + 1], kf.y, dot0);
            dot1 = fmaf(query_shared[1 * head_dim + d], kf.x, dot1);
            dot1 = fmaf(query_shared[1 * head_dim + d + 1], kf.y, dot1);
            dot2 = fmaf(query_shared[2 * head_dim + d], kf.x, dot2);
            dot2 = fmaf(query_shared[2 * head_dim + d + 1], kf.y, dot2);
            dot3 = fmaf(query_shared[3 * head_dim + d], kf.x, dot3);
            dot3 = fmaf(query_shared[3 * head_dim + d + 1], kf.y, dot3);
        }
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            dot0 += __shfl_down_sync(0xFFFFFFFF, dot0, offset);
            dot1 += __shfl_down_sync(0xFFFFFFFF, dot1, offset);
            dot2 += __shfl_down_sync(0xFFFFFFFF, dot2, offset);
            dot3 += __shfl_down_sync(0xFFFFFFFF, dot3, offset);
        }
        if (lane == 0)
        {
            scores[0 * score_stride + i] = dot0 * scale;
            scores[1 * score_stride + i] = dot1 * scale;
            scores[2 * score_stride + i] = dot2 * scale;
            scores[3 * score_stride + i] = dot3 * scale;
        }
    }
    __syncthreads();

#pragma unroll
    for (int h = 0; h < TS_GQA_DECODE_GROUP4_HEADS; h++)
    {
        float max_v = -FLT_MAX;
        for (int i = tid; i < token_count; i += blockDim.x)
            max_v = fmaxf(max_v, scores[h * score_stride + i]);
        max_v = block_reduce_max(max_v);
        if (tid == 0)
            shared_max[h] = max_v;
        __syncthreads();

        float sum = 0.0f;
        for (int i = tid; i < token_count; i += blockDim.x)
        {
            float p = expf(scores[h * score_stride + i] - shared_max[h]);
            scores[h * score_stride + i] = p;
            sum += p;
        }
        sum = block_reduce_sum(sum);
        if (tid == 0)
            shared_sum[h] = sum;
        __syncthreads();
    }
}

extern "C" __global__ void ts_gqa_decode_attention_group4_d512_f16(
    const float* query,
    const half* key_cache,
    const half* value_cache,
    float* output,
    int num_kv_heads,
    int attend_start,
    int attend_len,
    int cache_size,
    float scale,
    int score_capacity,
    const int* dyn)
{
    const int kv_head = blockIdx.x;
    if (kv_head >= num_kv_heads)
        return;
    if (dyn)
        attend_len = dyn[TS_DYN_ATTEND_LEN];
    attend_len = max(0, min(
        attend_len,
        min(score_capacity, cache_size - attend_start)));

    extern __shared__ float workspace[];
    float* scores = workspace;
    float* query_shared =
        scores + TS_GQA_DECODE_GROUP4_HEADS * score_capacity;
    __shared__ float shared_max[TS_GQA_DECODE_GROUP4_HEADS];
    __shared__ float shared_sum[TS_GQA_DECODE_GROUP4_HEADS];

    ts_gqa_decode_group4_scores_f16(
        query, key_cache, kv_head, cache_size, attend_start, attend_len,
        scale, TS_GQA_DECODE_GROUP4_HEAD_DIM, score_capacity,
        scores, query_shared, shared_max, shared_sum);

    const int q_head_base = kv_head * TS_GQA_DECODE_GROUP4_HEADS;
    for (int d = threadIdx.x; d < TS_GQA_DECODE_GROUP4_HEAD_DIM; d += blockDim.x)
    {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        for (int i = 0; i < attend_len; i++)
        {
            float vv = __half2float(value_cache[
                ((size_t)kv_head * cache_size + attend_start + i) *
                    TS_GQA_DECODE_GROUP4_HEAD_DIM + d]);
            acc0 = fmaf(scores[0 * score_capacity + i], vv, acc0);
            acc1 = fmaf(scores[1 * score_capacity + i], vv, acc1);
            acc2 = fmaf(scores[2 * score_capacity + i], vv, acc2);
            acc3 = fmaf(scores[3 * score_capacity + i], vv, acc3);
        }
        float* out = output +
            (size_t)q_head_base * TS_GQA_DECODE_GROUP4_HEAD_DIM + d;
        out[0 * TS_GQA_DECODE_GROUP4_HEAD_DIM] =
            shared_sum[0] > 0.0f ? acc0 / shared_sum[0] : 0.0f;
        out[1 * TS_GQA_DECODE_GROUP4_HEAD_DIM] =
            shared_sum[1] > 0.0f ? acc1 / shared_sum[1] : 0.0f;
        out[2 * TS_GQA_DECODE_GROUP4_HEAD_DIM] =
            shared_sum[2] > 0.0f ? acc2 / shared_sum[2] : 0.0f;
        out[3 * TS_GQA_DECODE_GROUP4_HEAD_DIM] =
            shared_sum[3] > 0.0f ? acc3 / shared_sum[3] : 0.0f;
    }
}

extern "C" __global__ void ts_gqa_decode_attention_group4_d256_f16(
    const float* query,
    const half* key_cache,
    const half* value_cache,
    float* output,
    int num_kv_heads,
    int attend_len,
    int cache_size,
    float scale,
    int score_capacity,
    const int* dyn)
{
    const int kv_head = blockIdx.x;
    if (kv_head >= num_kv_heads)
        return;
    if (dyn)
        attend_len = dyn[TS_DYN_ATTEND_LEN];
    // Before the first wrap the valid cache prefix is physically linear. Once
    // the SWA ring is full, chronological rotation is irrelevant: softmax
    // attention is invariant to applying the same permutation to K/V.
    attend_len = max(0, min(attend_len, min(cache_size, score_capacity)));

    const int head_dim = 256;
    extern __shared__ float workspace[];
    float* scores = workspace;
    float* query_shared =
        scores + TS_GQA_DECODE_GROUP4_HEADS * score_capacity;
    __shared__ float shared_max[TS_GQA_DECODE_GROUP4_HEADS];
    __shared__ float shared_sum[TS_GQA_DECODE_GROUP4_HEADS];

    ts_gqa_decode_group4_scores_f16(
        query, key_cache, kv_head, cache_size, 0, attend_len,
        scale, head_dim, score_capacity,
        scores, query_shared, shared_max, shared_sum);

    const int q_head_base = kv_head * TS_GQA_DECODE_GROUP4_HEADS;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        for (int i = 0; i < attend_len; i++)
        {
            float vv = __half2float(value_cache[
                ((size_t)kv_head * cache_size + i) * head_dim + d]);
            acc0 = fmaf(scores[0 * score_capacity + i], vv, acc0);
            acc1 = fmaf(scores[1 * score_capacity + i], vv, acc1);
            acc2 = fmaf(scores[2 * score_capacity + i], vv, acc2);
            acc3 = fmaf(scores[3 * score_capacity + i], vv, acc3);
        }
        float* out = output + (size_t)q_head_base * head_dim + d;
        out[0 * head_dim] = shared_sum[0] > 0.0f ? acc0 / shared_sum[0] : 0.0f;
        out[1 * head_dim] = shared_sum[1] > 0.0f ? acc1 / shared_sum[1] : 0.0f;
        out[2 * head_dim] = shared_sum[2] > 0.0f ? acc2 / shared_sum[2] : 0.0f;
        out[3 * head_dim] = shared_sum[3] > 0.0f ? acc3 / shared_sum[3] : 0.0f;
    }
}

extern "C" __global__ void ts_gqa_decode_attention_partition_group4_d512_f16(
    const float* query,
    const half* key_cache,
    const half* value_cache,
    float* partial,
    int num_kv_heads,
    int attend_start,
    int attend_len,
    int cache_size,
    float scale,
    int num_partitions,
    int partition_size,
    const int* dyn)
{
    const int kv_head = blockIdx.x;
    const int partition = blockIdx.y;
    if (kv_head >= num_kv_heads || partition >= num_partitions)
        return;
    if (dyn)
        attend_len = dyn[TS_DYN_ATTEND_LEN];
    attend_len = max(0, min(attend_len, cache_size - attend_start));

    const int part_start = partition * partition_size;
    int part_len = attend_len - part_start;
    if (part_len > partition_size)
        part_len = partition_size;
    if (part_len < 0)
        part_len = 0;

    extern __shared__ float workspace[];
    float* scores = workspace;
    float* query_shared =
        scores + TS_GQA_DECODE_GROUP4_HEADS * partition_size;
    __shared__ float shared_max[TS_GQA_DECODE_GROUP4_HEADS];
    __shared__ float shared_sum[TS_GQA_DECODE_GROUP4_HEADS];

    ts_gqa_decode_group4_scores_f16(
        query, key_cache, kv_head, cache_size,
        attend_start + part_start, part_len, scale,
        TS_GQA_DECODE_GROUP4_HEAD_DIM, partition_size,
        scores, query_shared, shared_max, shared_sum);

    const int q_head_base = kv_head * TS_GQA_DECODE_GROUP4_HEADS;
    const int partial_stride = TS_GQA_DECODE_GROUP4_HEAD_DIM + 2;
    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int h = 0; h < TS_GQA_DECODE_GROUP4_HEADS; h++)
        {
            float* row = partial +
                ((size_t)(q_head_base + h) * num_partitions + partition) *
                    partial_stride;
            row[0] = shared_max[h];
            row[1] = shared_sum[h];
        }
    }

    for (int d = threadIdx.x; d < TS_GQA_DECODE_GROUP4_HEAD_DIM; d += blockDim.x)
    {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        for (int i = 0; i < part_len; i++)
        {
            float vv = __half2float(value_cache[
                ((size_t)kv_head * cache_size + attend_start + part_start + i) *
                    TS_GQA_DECODE_GROUP4_HEAD_DIM + d]);
            acc0 = fmaf(scores[0 * partition_size + i], vv, acc0);
            acc1 = fmaf(scores[1 * partition_size + i], vv, acc1);
            acc2 = fmaf(scores[2 * partition_size + i], vv, acc2);
            acc3 = fmaf(scores[3 * partition_size + i], vv, acc3);
        }
        partial[
            ((size_t)(q_head_base + 0) * num_partitions + partition) *
                partial_stride + 2 + d] = acc0;
        partial[
            ((size_t)(q_head_base + 1) * num_partitions + partition) *
                partial_stride + 2 + d] = acc1;
        partial[
            ((size_t)(q_head_base + 2) * num_partitions + partition) *
                partial_stride + 2 + d] = acc2;
        partial[
            ((size_t)(q_head_base + 3) * num_partitions + partition) *
                partial_stride + 2 + d] = acc3;
    }
}

// Partitioned counterpart of ts_gqa_decode_attention_group4_d256_f16 for the
// circular SWA ring. The single-block kernel's grid is num_kv_heads CTAs --
// TWO on Gemma 4 -- which strands the rest of the GPU while every SWA decode
// layer serializes its whole window behind one CTA per KV head (~99 us/layer
// measured on a 48-SM RTX 3080). This kernel splits the physical ring
// [0, min(attend_len, cache_size)) across blockIdx.y partitions (softmax is
// invariant to the ring permutation, exactly as for the single-block kernel)
// and writes the same (max, sum, unnormalized acc) partial rows as the d512
// partition kernel above, combined by ts_gqa_decode_attention_partition_reduce_f32.
// Grid: (num_kv_heads, num_partitions).
extern "C" __global__ void ts_gqa_decode_attention_partition_group4_d256_f16(
    const float* query,
    const half* key_cache,
    const half* value_cache,
    const float* sinks,
    float* partial,
    int num_kv_heads,
    int attend_len,
    int cache_size,
    float scale,
    int has_sinks,
    int num_partitions,
    int partition_size,
    const int* dyn)
{
    const int kv_head = blockIdx.x;
    const int partition = blockIdx.y;
    if (kv_head >= num_kv_heads || partition >= num_partitions)
        return;
    if (dyn)
        attend_len = dyn[TS_DYN_ATTEND_LEN];
    attend_len = max(0, min(attend_len, cache_size));

    const int head_dim = 256;
    const int part_start = partition * partition_size;
    int part_len = attend_len - part_start;
    if (part_len > partition_size)
        part_len = partition_size;
    if (part_len < 0)
        part_len = 0;

    extern __shared__ float workspace[];
    float* scores = workspace;
    float* query_shared =
        scores + TS_GQA_DECODE_GROUP4_HEADS * partition_size;
    __shared__ float shared_max[TS_GQA_DECODE_GROUP4_HEADS];
    __shared__ float shared_sum[TS_GQA_DECODE_GROUP4_HEADS];
    __shared__ float sink_scale[TS_GQA_DECODE_GROUP4_HEADS];

    ts_gqa_decode_group4_scores_f16(
        query, key_cache, kv_head, cache_size,
        part_start, part_len, scale,
        head_dim, partition_size,
        scores, query_shared, shared_max, shared_sum);

    const int q_head_base = kv_head * TS_GQA_DECODE_GROUP4_HEADS;
    const int partial_stride = head_dim + 2;
    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int h = 0; h < TS_GQA_DECODE_GROUP4_HEADS; h++)
        {
            // Fold the per-head attention sink (an extra logit with no V row)
            // into partition 0's (max, sum) header. The scores stashed in
            // shared reference the pre-sink max, so the V accumulators below
            // are rescaled by exp(m_old - m_new) to match, exactly like an
            // online-softmax chunk step.
            float m = shared_max[h];
            float sum = shared_sum[h];
            float scl = 1.0f;
            if (has_sinks && partition == 0)
            {
                float s = sinks[q_head_base + h];
                float m_new = fmaxf(m, s);
                scl = expf(m - m_new);
                sum = sum * scl + expf(s - m_new);
                m = m_new;
            }
            sink_scale[h] = scl;
            float* row = partial +
                ((size_t)(q_head_base + h) * num_partitions + partition) *
                    partial_stride;
            row[0] = m;
            row[1] = sum;
        }
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        for (int i = 0; i < part_len; i++)
        {
            float vv = __half2float(value_cache[
                ((size_t)kv_head * cache_size + part_start + i) *
                    head_dim + d]);
            acc0 = fmaf(scores[0 * partition_size + i], vv, acc0);
            acc1 = fmaf(scores[1 * partition_size + i], vv, acc1);
            acc2 = fmaf(scores[2 * partition_size + i], vv, acc2);
            acc3 = fmaf(scores[3 * partition_size + i], vv, acc3);
        }
        partial[
            ((size_t)(q_head_base + 0) * num_partitions + partition) *
                partial_stride + 2 + d] = acc0 * sink_scale[0];
        partial[
            ((size_t)(q_head_base + 1) * num_partitions + partition) *
                partial_stride + 2 + d] = acc1 * sink_scale[1];
        partial[
            ((size_t)(q_head_base + 2) * num_partitions + partition) *
                partial_stride + 2 + d] = acc2 * sink_scale[2];
        partial[
            ((size_t)(q_head_base + 3) * num_partitions + partition) *
                partial_stride + 2 + d] = acc3 * sink_scale[3];
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
    const int* dyn,
    float* scores)
{
    int q_head = blockIdx.x;
    int partition = blockIdx.y;
    if (q_head >= num_q_heads || partition >= num_partitions)
        return;
    if (dyn)
    {
        attend_len = dyn[TS_DYN_ATTEND_LEN];
        if (circular && attend_len > cache_size)
            attend_len = cache_size;
        if (circular)
            attend_start = max(0, dyn[TS_DYN_KV_WRITE_POS] + 1 - attend_len);
    }

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
    int partition_size,
    const int* dyn)
{
    extern __shared__ float scores[];
    ts_gqa_decode_attention_partition_impl<float>(
        query, key_cache, value_cache, sinks, partial,
        num_q_heads, num_kv_heads, head_dim,
        attend_start, attend_len, cache_size, circular, scale,
        has_sinks, num_partitions, partition_size, dyn, scores);
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
    int partition_size,
    const int* dyn)
{
    extern __shared__ float scores[];
    ts_gqa_decode_attention_partition_impl<half>(
        query, key_cache, value_cache, sinks, partial,
        num_q_heads, num_kv_heads, head_dim,
        attend_start, attend_len, cache_size, circular, scale,
        has_sinks, num_partitions, partition_size, dyn, scores);
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
    int circular,
    const int* dyn)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * head_dim;
    if (idx >= total)
        return;
    if (dyn)
        start_pos = dyn[TS_DYN_KV_WRITE_POS];

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
    int circular,
    const int* dyn)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * seq_len * head_dim;
    if (idx >= total)
        return;
    if (dyn)
        start_pos = dyn[TS_DYN_KV_WRITE_POS];

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int seq = tmp % seq_len;
    int head = tmp / seq_len;
    int cache_pos = circular ? ((start_pos + seq) % cache_size) : (start_pos + seq);
    cache[((size_t)head * cache_size + cache_pos) * head_dim + d] = __float2half_rn(source[idx]);
}

// Refreshes the cached RoPE position tensors (one row per head, single-token
// decode) from the decode-graph dynamic parameter block. Captured at the top
// of a decode graph so every replay RoPEs with the current position.
extern "C" __global__ void ts_fill_rope_positions_i32(
    int* pos_q,
    int q_rows,
    int* pos_k,
    int k_rows,
    const int* dyn)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = dyn[TS_DYN_ROPE_POS];
    if (idx < q_rows)
        pos_q[idx] = pos;
    if (idx < k_rows)
        pos_k[idx] = pos;
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

// Expand a head-first GQA cache [kv_heads, cache_size, head_dim] into a
// contiguous F32 tensor [kv_heads * group_size, seq_len, head_dim].  Combining
// the active-window gather, F16 conversion and head broadcast in one pass keeps
// materialized cuBLAS attention entirely on the GPU.
template <typename cache_t>
__device__ __forceinline__ void ts_expand_kv_heads_impl(
    const cache_t* cache,
    float* output,
    int num_kv_heads,
    int seq_len,
    int cache_size,
    int head_dim,
    int group_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_heads = num_kv_heads * group_size;
    int total = out_heads * seq_len * head_dim;
    if (idx >= total)
        return;

    int d = idx % head_dim;
    int tmp = idx / head_dim;
    int seq = tmp % seq_len;
    int out_head = tmp / seq_len;
    int kv_head = out_head / group_size;
    output[idx] = (float)cache[((size_t)kv_head * cache_size + seq) * head_dim + d];
}

extern "C" __global__ void ts_expand_kv_heads_f32(
    const float* cache,
    float* output,
    int num_kv_heads,
    int seq_len,
    int cache_size,
    int head_dim,
    int group_size)
{
    ts_expand_kv_heads_impl(
        cache, output, num_kv_heads, seq_len, cache_size, head_dim, group_size);
}

extern "C" __global__ void ts_expand_kv_heads_f16(
    const half* cache,
    float* output,
    int num_kv_heads,
    int seq_len,
    int cache_size,
    int head_dim,
    int group_size)
{
    ts_expand_kv_heads_impl(
        cache, output, num_kv_heads, seq_len, cache_size, head_dim, group_size);
}

template <typename scalar_t>
__device__ __forceinline__ void ts_repeat_interleave_impl(
    const scalar_t* source,
    scalar_t* output,
    int outer_size,
    int dim_size,
    int repeats,
    int inner_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_dim_size = dim_size * repeats;
    int out_outer_stride = out_dim_size * inner_size;
    int total = outer_size * out_outer_stride;
    if (idx >= total)
        return;

    int outer = idx / out_outer_stride;
    int within_outer = idx - outer * out_outer_stride;
    int repeated_dim = within_outer / inner_size;
    int inner = within_outer - repeated_dim * inner_size;
    int source_dim = repeated_dim / repeats;
    output[idx] = source[((size_t)outer * dim_size + source_dim) * inner_size + inner];
}

extern "C" __global__ void ts_repeat_interleave_f32(
    const float* source,
    float* output,
    int outer_size,
    int dim_size,
    int repeats,
    int inner_size)
{
    ts_repeat_interleave_impl(
        source, output, outer_size, dim_size, repeats, inner_size);
}

extern "C" __global__ void ts_repeat_interleave_f16(
    const half* source,
    half* output,
    int outer_size,
    int dim_size,
    int repeats,
    int inner_size)
{
    ts_repeat_interleave_impl(
        source, output, outer_size, dim_size, repeats, inner_size);
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
// at (s*num_heads + h)*head_dim + j) ÔÇö the layout Gemma 4's q/k carry before
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

// Builds Gemma 4's local/global single-token NeoX RoPE lookup tables from the
// live CUDA-graph position.  The tables are produced once per replay and then
// reused by every attention layer, avoiding repeated sincosf work per head and
// layer.  Frequencies already include the learned global proportional-RoPE
// factors, which cannot be represented by a single base/scale pair.
extern "C" __global__ void ts_fill_neox_rope_tables_dyn_f32(
    float* local_cos,
    float* local_sin,
    const float* local_frequencies,
    int local_half,
    float* global_cos,
    float* global_sin,
    const float* global_frequencies,
    int global_half,
    const int* dyn)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = max(local_half, global_half);
    if (idx >= total || !dyn)
        return;

    float position = (float)dyn[TS_DYN_ROPE_POS];
    if (idx < local_half)
        sincosf(position * local_frequencies[idx], &local_sin[idx], &local_cos[idx]);
    if (idx < global_half)
        sincosf(position * global_frequencies[idx], &global_sin[idx], &global_cos[idx]);
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

// Single-token decode (rows==1) generic quant matmul: one BLOCK per output
// column instead of ts_quant_matmul_f32's four-columns-per-block split. Same
// blockDim.x thread budget per column as that kernel, but a single
// block_reduce_sum (no repeated __syncthreads()-separated reductions) and no
// row-tile machinery (nothing to amortize with only one row) -- see the
// TS_CUDA_QMM_VEC call site for why this beats both ts_quant_matmul_f32 (whose
// 4-way column split still costs 4 serialized block reductions) and
// ts_quant_matmul_batched_f32 (whose one-warp-per-column split under-uses the
// SM for a single row on wide tensors).
extern "C" __global__ void ts_quant_matmul_vec_f32(
    const uint8_t* weights,
    const float* input,
    float* output,
    int type,
    int in_dim,
    int out_dim)
{
    int out_col = blockIdx.x;
    if (out_col >= out_dim)
        return;

    int row_bytes = qrow_bytes(type, in_dim);
    const uint8_t* w_row = weights + (size_t)out_col * row_bytes;

    float acc = 0.0f;
    for (int k = threadIdx.x; k < in_dim; k += blockDim.x)
        acc += qvalue_at(w_row, type, k) * input[k];

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        output[out_col] = acc;
}

// Row tile height for the row-batched quantized matmul kernels below. Each
// block handles a contiguous tile of up to TS_QMM_ROW_TILE rows for one output
// column, decoding the weight ONCE and reusing it across the tile's rows;
// grid.y = ceil(rows/TILE) covers the rest. Kept small (matches the 4-row
// ts_quant_matmul_q8_0_f32 tiling) so the accumulators stay in registers.
// Weight memory traffic / dequant work drops from B x to ceil(B/TILE) x.
// (Q4_0 ÔÇö the dominant dense quant ÔÇö has its own row-tiled kernel,
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

// Decode-only IQ2 matvec over a globally pre-quantized q8_1 activation row.
// Unlike ts_quant_matmul_iq2_xxs_q8_1_f32, this kernel does not rebuild the
// q8_1 row in per-CTA shared memory: the caller quantizes it once, then four
// warp-owned output columns per CTA reuse that stable scratch. Supports the two
// IQ2 formats used most heavily by the Qwen3.6 dynamic quant (gate/up IQ2_XXS
// and expert-down IQ2_S). Multi-row matmul keeps the existing tiled paths.
extern "C" __global__ void ts_quant_matmul_iq2_vec_q8_1_f32(
    const uint8_t* weights,
    const ts_block_q8_1* xq,
    float* output,
    int type,
    int in_dim,
    int out_dim)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int out_col = blockIdx.x * warps_per_block + warp;
    if (out_col >= out_dim || (in_dim & 255) != 0)
        return;

    int iq_blocks = in_dim / 256;
    int dot_groups = iq_blocks * 8;
    int row_bytes = iq_blocks * (type == GGML_IQ2_XXS ? 66 : 82);
    const uint8_t* w_row = weights + (size_t)out_col * row_bytes;

    float acc = 0.0f;
    if (type == GGML_IQ2_XXS)
    {
        for (int g = lane; g < dot_groups; g += 32)
        {
            int ib = g >> 3;
            int group = g & 7;
            acc += dot_iq2_xxs_q8_1(w_row + (size_t)ib * 66, xq + (size_t)ib * 8, group);
        }
    }
    else if (type == GGML_IQ2_S)
    {
        for (int g = lane; g < dot_groups; g += 32)
        {
            int ib = g >> 3;
            int group = g & 7;
            acc += dot_iq2_s_q8_1(w_row + (size_t)ib * 82, xq + (size_t)ib * 8, group);
        }
    }
    else
    {
        return;
    }

    acc = warp_allreduce_sum(acc);
    if (lane == 0)
        output[out_col] = acc;
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

// Decode-oriented q8_1 activation quantizer: one warp cooperatively handles one
// 32-value block. The legacy kernel above assigns an entire block to one thread,
// serializing 32 loads and stores while neighboring lanes walk different,
// 128-byte-strided blocks. This mapping makes the input and q-byte accesses
// coalesced and exposes all 32 values to the SM at once.
//
// Keep the arithmetic/layout identical to quantize_q8_1_block: rintf rounding,
// int8 clamp, half-rounded d, and half-rounded s = d * sum(q). That lets callers
// switch between kernels without changing Q8_0/Q4_0 matmul results.
extern "C" __global__ void ts_quantize_q8_1_rows_warp_f32(
    const float* input,
    ts_block_q8_1* out,
    int in_dim,
    int rows)
{
    int lane = threadIdx.x & 31;
    long long warp_idx =
        (((long long)blockIdx.x * blockDim.x) + threadIdx.x) >> 5;
    long long total_blocks = (long long)rows * (in_dim / TS_QK8_1);
    if (warp_idx >= total_blocks)
        return;

    float x = input[(size_t)warp_idx * TS_QK8_1 + lane];
    // Starting from zero preserves the legacy fmaxf behavior for NaN inputs.
    float amax = fmaxf(0.0f, fabsf(x));
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(0xFFFFFFFF, amax, offset));
    amax = __shfl_sync(0xFFFFFFFF, amax, 0);

    float d = amax > 0.0f ? amax / 127.0f : 0.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;
    int q = (int)rintf(x * id);
    q = max(-127, min(127, q));

    ts_block_q8_1* dst = out + warp_idx;
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

// Split-layout q8_1 row quantization for the MMQ cp.async staging path: the qs
// bytes land in a dense [rows][in_dim] int8 array (rows are 16-byte aligned
// whenever in_dim % 16 == 0, so the GEMM can stage activation windows with
// single 16-byte copies instead of four 4-byte loads from the 36-byte-strided
// interleaved blocks) and the per-block scale in a separate float array.
// The scale is round-tripped through half precision so both the quantized
// values and the effective scales are bit-identical to
// ts_quantize_q8_1_rows_f32 (which stores d as half): the GEMM result must not
// depend on which scratch layout staged it.
extern "C" __global__ void ts_quantize_q8_1_split_rows_f32(
    const float* input,
    int8_t* qs_out,
    float* d_out,
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
    // The mmq2 route guarantees in_dim % 256 == 0, so every block's floats and
    // qs bytes are 16-byte aligned: read 8 float4s, write 2 int4s (the scalar
    // per-byte form of this kernel measured ~7x slower - 8% of prefill GPU).
    const float4* x4 = reinterpret_cast<const float4*>(
        input + (size_t)r * in_dim + (size_t)qb * TS_QK8_1);

    float4 v[8];
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        v[i] = x4[i];
        amax = fmaxf(amax, fabsf(v[i].x));
        amax = fmaxf(amax, fabsf(v[i].y));
        amax = fmaxf(amax, fabsf(v[i].z));
        amax = fmaxf(amax, fabsf(v[i].w));
    }

    float d = amax > 0.0f ? amax / 127.0f : 0.0f;
    float id = d > 0.0f ? 1.0f / d : 0.0f;

    int packed[8];
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        int q0 = max(-127, min(127, (int)rintf(v[i].x * id)));
        int q1 = max(-127, min(127, (int)rintf(v[i].y * id)));
        int q2 = max(-127, min(127, (int)rintf(v[i].z * id)));
        int q3 = max(-127, min(127, (int)rintf(v[i].w * id)));
        packed[i] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | ((q2 & 0xFF) << 16) | (q3 << 24);
    }

    int4* qs4 = reinterpret_cast<int4*>(qs_out + (size_t)r * in_dim + (size_t)qb * TS_QK8_1);
    qs4[0] = make_int4(packed[0], packed[1], packed[2], packed[3]);
    qs4[1] = make_int4(packed[4], packed[5], packed[6], packed[7]);
    d_out[idx] = __half2float(__float2half_rn(d));
}

// Block-tile dp4a (int8-MMA) Q8_0 GEMM ÔÇö the fast multi-row path for the MTP verify
// window (rows 2-8). The scalar block-reduce kernels above are compute-bound on the
// big FFN matmuls (measured ~78% of verify GPU time). This kernel:
//   * 256 threads compute a TS_Q8_DP4A_ROWS x TS_Q8_DP4A_COLS output tile;
//   * reads the pre-quantized q8_1 activations (xq) from global (L2-cached; quantized
//     once by ts_quantize_q8_1_rows_f32), weight read once per row-tile;
//   * each thread strides the dp4a-GROUPS (4 elements) of in_dim ÔÇö full parallelism
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

        // Load each row's activation group + scale ONCE (reused across all columns) ÔÇö
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

// MMQ-style Q8_0 GEMM for prefill-sized row counts: reads the quantized weight
// DIRECTLY (no f16 dequant round trip) with int8 tensor-core mma.m16n8k32.
// One CTA (8 warps) computes a TS_MMQ_M x TS_MMQ_N output tile; TS_MMQ_KSTEP
// Q8 blocks (k = 32 each) are staged in shared memory per iteration. Because a
// k32 mma step spans exactly one Q8_0/q8_1 block, the int32 dot per block is
// exact and the (d_w * d_act) scale is applied per block in registers - the
// same numerics as the dp4a path / ggml's MMQ. Weight DRAM traffic is
// ceil(rows / TS_MMQ_M) sweeps (vs ~3 effective sweeps for dequant+cuBLAS and
// rows/4 for the dp4a tile kernel).
#define TS_MMQ_M 128
#define TS_MMQ_N 64
#define TS_MMQ_NFRAG (TS_MMQ_N / 8)
#define TS_MMQ_KSTEP 4
#define TS_MMQ_KPAD 4   // smem row padding (bytes) to break bank conflicts
#define TS_MMQ_THREADS 256
// Best measured configuration: 2 CTAs/SM with the next k-step's WEIGHT bytes
// prefetched into registers during the mma phase. Variants tried and rejected:
// 256-row/512-thread CTA (single resident CTA exposes staging latency),
// KSTEP=2 cp.async double buffering (per-column window-offset math and 2x
// barrier count cost more than the async staging saved).
extern "C" __global__ void __launch_bounds__(TS_MMQ_THREADS, 2) ts_quant_matmul_q8_0_mmq_f32(
    const uint8_t* weights,
    const ts_block_q8_1* xq,
    float* output,
    int in_dim,
    int out_dim,
    int rows)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    int n0 = blockIdx.x * TS_MMQ_N;
    int m0 = blockIdx.y * TS_MMQ_M;
    if (n0 >= out_dim || m0 >= rows)
        return;

    int q8_blocks = in_dim / 32;
    int row_bytes = q8_blocks * 34;

    __shared__ int8_t smA[TS_MMQ_M][TS_MMQ_KSTEP * 32 + TS_MMQ_KPAD];
    __shared__ int8_t smB[TS_MMQ_N][TS_MMQ_KSTEP * 32 + TS_MMQ_KPAD];
    __shared__ float  smDa[TS_MMQ_M][TS_MMQ_KSTEP];
    __shared__ float  smDw[TS_MMQ_N][TS_MMQ_KSTEP];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;      // fragment group id (0..7)
    int tig = lane & 3;     // thread id in group (0..3)
    int warp_m = warp * 16; // this warp's row offset inside the CTA tile

    // Each warp owns a 16-row strip and sweeps all TS_MMQ_NFRAG n8 fragments,
    // so the staged A tile is reused across the CTA's 64 output columns
    // (an 8-wide CTA re-staged the whole activation slice out_dim/8 times,
    // which made L2 activation traffic dominate the weight traffic).
    float fc[TS_MMQ_NFRAG][4];
#pragma unroll
    for (int nf = 0; nf < TS_MMQ_NFRAG; nf++)
    {
        fc[nf][0] = 0.0f; fc[nf][1] = 0.0f; fc[nf][2] = 0.0f; fc[nf][3] = 0.0f;
    }

    // Software pipeline: each iteration PREFETCHES the next k-step's WEIGHT
    // bytes (the only DRAM-resident stream) into registers, runs the mma phase
    // on the smem staged by the previous iteration, then stages the next
    // activation tile directly (activations are small and L2-resident across
    // the many n-CTAs that share them) and commits the weight registers.
    // Without pipelining the staging loads' DRAM latency was fully exposed
    // between the __syncthreads pairs (measured ~55 GB/s effective).
    //
    // Per-thread staging slices (blockDim = 256):
    //   A: TS_MMQ_M*KSTEP*2 = 1024 16-byte units -> 4 units/thread
    //   B: TS_MMQ_N*KSTEP*2 = 512 units          -> 2 units/thread
    //   scales: 512 dAct (2/thread) + 256 dW (1/thread)
    int pfB[2][4];
    float pfDw;

    auto prefetchB = [&](int b0)
    {
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            int u = tid + (i << 8);
            int c = u >> 3;
            int rem = u & 7;
            int kb = rem >> 1;
            int hs = rem & 1;
            int col = n0 + c;
            int gb = b0 + kb;
            if (col < out_dim && gb < q8_blocks)
            {
                const uint8_t* wblk = weights + (size_t)col * row_bytes + (size_t)gb * 34;
                pfB[i][0] = get_int_b2(wblk + 2, hs * 4 + 0);
                pfB[i][1] = get_int_b2(wblk + 2, hs * 4 + 1);
                pfB[i][2] = get_int_b2(wblk + 2, hs * 4 + 2);
                pfB[i][3] = get_int_b2(wblk + 2, hs * 4 + 3);
            }
            else
            {
                pfB[i][0] = 0; pfB[i][1] = 0; pfB[i][2] = 0; pfB[i][3] = 0;
            }
        }
        {
            int c = tid >> 2;
            int kb = tid & 3;
            int col = n0 + c;
            int gb = b0 + kb;
            pfDw = (col < out_dim && gb < q8_blocks)
                ? __half2float(*reinterpret_cast<const half*>(
                      weights + (size_t)col * row_bytes + (size_t)gb * 34)) : 0.0f;
        }
    };

    auto stageA = [&](int b0)
    {
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            int u = tid + (i << 8);
            int r = u >> 3;
            int rem = u & 7;
            int kb = rem >> 1;
            int hs = rem & 1;
            int gr = m0 + r;
            int gb = b0 + kb;
            int* dst = reinterpret_cast<int*>(&smA[r][kb * 32 + hs * 16]);
            if (gr < rows && gb < q8_blocks)
            {
                // qs sits at +4 of the 36-byte q8_1 block: 4-byte aligned.
                const int* src = reinterpret_cast<const int*>(
                    xq[(size_t)gr * q8_blocks + gb].qs) + hs * 4;
                dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
            }
            else
            {
                dst[0] = 0; dst[1] = 0; dst[2] = 0; dst[3] = 0;
            }
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            int u = tid + (i << 8);
            int r = u >> 2;
            int kb = u & 3;
            int gr = m0 + r;
            int gb = b0 + kb;
            smDa[r][kb] = (gr < rows && gb < q8_blocks)
                ? __half2float(xq[(size_t)gr * q8_blocks + gb].d) : 0.0f;
        }
    };

    auto commitB = [&]()
    {
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            int u = tid + (i << 8);
            int c = u >> 3;
            int rem = u & 7;
            int kb = rem >> 1;
            int hs = rem & 1;
            int* dst = reinterpret_cast<int*>(&smB[c][kb * 32 + hs * 16]);
            dst[0] = pfB[i][0]; dst[1] = pfB[i][1]; dst[2] = pfB[i][2]; dst[3] = pfB[i][3];
        }
        smDw[tid >> 2][tid & 3] = pfDw;
    };

    prefetchB(0);
    stageA(0);
    commitB();
    __syncthreads();

    for (int b0 = 0; b0 < q8_blocks; b0 += TS_MMQ_KSTEP)
    {
        bool hasNext = b0 + TS_MMQ_KSTEP < q8_blocks;
        if (hasNext)
            prefetchB(b0 + TS_MMQ_KSTEP);

        // ---- per staged block: load A frags once, mma across all n8 frags ----
#pragma unroll
        for (int kb = 0; kb < TS_MMQ_KSTEP; kb++)
        {
            int a0 = *reinterpret_cast<const int*>(&smA[warp_m + g][kb * 32 + 4 * tig]);
            int a1 = *reinterpret_cast<const int*>(&smA[warp_m + g + 8][kb * 32 + 4 * tig]);
            int a2 = *reinterpret_cast<const int*>(&smA[warp_m + g][kb * 32 + 16 + 4 * tig]);
            int a3 = *reinterpret_cast<const int*>(&smA[warp_m + g + 8][kb * 32 + 16 + 4 * tig]);
            float daLo = smDa[warp_m + g][kb];
            float daHi = smDa[warp_m + g + 8][kb];

#pragma unroll
            for (int nf = 0; nf < TS_MMQ_NFRAG; nf++)
            {
                int b0r = *reinterpret_cast<const int*>(&smB[nf * 8 + g][kb * 32 + 4 * tig]);
                int b1r = *reinterpret_cast<const int*>(&smB[nf * 8 + g][kb * 32 + 16 + 4 * tig]);

                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0r), "r"(b1r));

                float dw0 = smDw[nf * 8 + 2 * tig][kb];
                float dw1 = smDw[nf * 8 + 2 * tig + 1][kb];
                fc[nf][0] += (float)c0 * dw0 * daLo;
                fc[nf][1] += (float)c1 * dw1 * daLo;
                fc[nf][2] += (float)c2 * dw0 * daHi;
                fc[nf][3] += (float)c3 * dw1 * daHi;
            }
        }
        __syncthreads();
        if (hasNext)
        {
            stageA(b0 + TS_MMQ_KSTEP);
            commitB();
            __syncthreads();
        }
    }

    // ---- bounds-checked store (c0/c1 -> row g, c2/c3 -> row g+8) ----
    int out_r0 = m0 + warp_m + g;
    int out_r1 = out_r0 + 8;
#pragma unroll
    for (int nf = 0; nf < TS_MMQ_NFRAG; nf++)
    {
        int out_c0 = n0 + nf * 8 + 2 * tig;
        int out_c1 = out_c0 + 1;
        if (out_r0 < rows)
        {
            if (out_c0 < out_dim) output[(size_t)out_r0 * out_dim + out_c0] = fc[nf][0];
            if (out_c1 < out_dim) output[(size_t)out_r0 * out_dim + out_c1] = fc[nf][1];
        }
        if (out_r1 < rows)
        {
            if (out_c0 < out_dim) output[(size_t)out_r1 * out_dim + out_c0] = fc[nf][2];
            if (out_c1 < out_dim) output[(size_t)out_r1 * out_dim + out_c1] = fc[nf][3];
        }
    }
#else
    (void)weights; (void)xq; (void)output; (void)in_dim; (void)out_dim; (void)rows;
#endif
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
// Ampere cp.async helpers (global -> shared without a register round trip).
__device__ __forceinline__ void ts_cp_async16(void* smem_dst, const void* gmem_src)
{
    unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(gmem_src));
}
__device__ __forceinline__ void ts_cp_async8(void* smem_dst, const void* gmem_src)
{
    unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(s), "l"(gmem_src));
}
__device__ __forceinline__ void ts_cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void ts_cp_async_wait_all()
{
    asm volatile("cp.async.wait_group 0;\n" ::);
}
#endif

// cp.async variant of the MMQ kernel, for in_dim % 256 == 0 (every real model
// dim). Identical math and accumulation order to ts_quant_matmul_q8_0_mmq_f32
// (bit-identical results); only the staging differs:
//   * WEIGHT windows are copied RAW into a double-buffered shared staging area
//     by 16-byte cp.async chunks issued BEFORE the mma phase, replacing the
//     register prefetch's ~24 two-byte loads per thread per k-step with ~2.25
//     async copies. in_dim % 256 == 0 makes row_bytes % 16 == 0, so a window's
//     16-byte alignment phase pa = (b0*34) & 15 is the SAME for every column
//     (the per-column offset math that sank the first cp.async attempt is gone)
//     and q8_blocks % TS_MMQ_KSTEP == 0 removes the k-tail entirely.
//   * ACTIVATIONS come from the SPLIT q8_1 scratch (dense qs rows + separate
//     float scales, see ts_quantize_q8_1_split_rows_f32), staged with 16-byte
//     loads instead of four 4-byte loads per unit.
// The mma phase reads weight qs straight from the raw staging bytes: qs of
// block kb sits at pa + kb*34 + 2, which is int-aligned for odd kb and
// half-aligned for even kb (34 = 2 mod 4), so even-kb fragments assemble from
// two u16 shared loads. dW is register-prefetched from global (one half per
// thread per window) and committed to shared floats, exactly like before.
#define TS_MMQ2_BSTRIDE (TS_MMQ_KSTEP * 34 + 8)      // 144: window bytes rounded up to 16
#define TS_MMQ2_BCHUNKS (TS_MMQ2_BSTRIDE / 16)       // 9 cp.async chunks per column window
extern "C" __global__ void __launch_bounds__(TS_MMQ_THREADS, 2) ts_quant_matmul_q8_0_mmq2_f32(
    const uint8_t* weights,
    const int8_t* xq_qs,
    const float* xq_d,
    float* output,
    int in_dim,
    int out_dim,
    int rows)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    int n0 = blockIdx.x * TS_MMQ_N;
    int m0 = blockIdx.y * TS_MMQ_M;
    if (n0 >= out_dim || m0 >= rows || (in_dim & 255) != 0)
        return;

    int q8_blocks = in_dim / 32;
    int row_bytes = q8_blocks * 34;   // % 16 == 0 because in_dim % 256 == 0

    // smA stride 144 (= KSTEP*32 + 16): 16-byte aligned rows for int4 staging
    // stores; mma-phase bank pattern (4g + tig) stays conflict-free like the
    // interleaved kernel's stride-132 layout.
    __shared__ __align__(16) int8_t  smA[TS_MMQ_M][TS_MMQ_KSTEP * 32 + 16];
    __shared__ __align__(16) uint8_t smBraw[2][TS_MMQ_N][TS_MMQ2_BSTRIDE];
    __shared__ float   smDa[TS_MMQ_M][TS_MMQ_KSTEP];
    __shared__ float   smDw[TS_MMQ_N][TS_MMQ_KSTEP];

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = lane >> 2;
    int tig = lane & 3;
    int warp_m = warp * 16;

    float fc[TS_MMQ_NFRAG][4];
#pragma unroll
    for (int nf = 0; nf < TS_MMQ_NFRAG; nf++)
    {
        fc[nf][0] = 0.0f; fc[nf][1] = 0.0f; fc[nf][2] = 0.0f; fc[nf][3] = 0.0f;
    }

    float pfDw;

    // Issue the raw weight window [col*row_bytes + b0*34 - pa, +TS_MMQ2_BSTRIDE)
    // for every CTA column as 16-byte cp.async chunks (the final chunk carries
    // up to 8 slack bytes; the weight allocation is padded by 16 bytes so it
    // never crosses the end), and register-prefetch this window's dW halves.
    auto issueB = [&](int buf, int b0)
    {
        int pa = (b0 * 34) & 15;
        const uint8_t* wbase = weights + (size_t)b0 * 34 - pa;
#pragma unroll
        for (int i = 0; i < 3; i++)
        {
            int u = tid + (i << 8);
            if (u < TS_MMQ_N * TS_MMQ2_BCHUNKS)
            {
                int c = u / TS_MMQ2_BCHUNKS;
                int ch = u - c * TS_MMQ2_BCHUNKS;
                int col = n0 + c;
                if (col < out_dim)
                    ts_cp_async16(&smBraw[buf][c][ch * 16], wbase + (size_t)col * row_bytes + (size_t)ch * 16);
            }
        }
        {
            int c = tid >> 2;
            int kb = tid & 3;
            int col = n0 + c;
            pfDw = col < out_dim
                ? __half2float(*reinterpret_cast<const half*>(
                      weights + (size_t)col * row_bytes + (size_t)(b0 + kb) * 34))
                : 0.0f;
        }
    };

    // Activations: 16-byte units straight out of the dense split-qs rows
    // (per thread: 4 int4 loads + stores vs the interleaved layout's 16+16).
    // Synchronous on purpose: an async A stage was measured SLOWER (extra
    // register spill and nothing to overlap the wait with; A is L2-resident).
    auto stageA = [&](int b0)
    {
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            int u = tid + (i << 8);
            int r = u >> 3;
            int off = (u & 7) << 4;
            int gr = m0 + r;
            int4* dst = reinterpret_cast<int4*>(&smA[r][off]);
            *dst = gr < rows
                ? *reinterpret_cast<const int4*>(xq_qs + (size_t)gr * in_dim + (size_t)b0 * 32 + off)
                : make_int4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < 2; i++)
        {
            int u = tid + (i << 8);
            int r = u >> 2;
            int kb = u & 3;
            int gr = m0 + r;
            smDa[r][kb] = gr < rows ? xq_d[(size_t)gr * q8_blocks + b0 + kb] : 0.0f;
        }
    };

    auto commitDw = [&]()
    {
        smDw[tid >> 2][tid & 3] = pfDw;
    };

    issueB(0, 0);
    stageA(0);
    commitDw();
    ts_cp_async_commit();
    ts_cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    for (int b0 = 0; b0 < q8_blocks; b0 += TS_MMQ_KSTEP)
    {
        bool hasNext = b0 + TS_MMQ_KSTEP < q8_blocks;
        if (hasNext)
        {
            issueB(cur ^ 1, b0 + TS_MMQ_KSTEP);   // overlaps the mma phase below
            ts_cp_async_commit();
        }

        int pa = (b0 * 34) & 15;                  // uniform across columns
#pragma unroll
        for (int kb = 0; kb < TS_MMQ_KSTEP; kb++)
        {
            int a0 = *reinterpret_cast<const int*>(&smA[warp_m + g][kb * 32 + 4 * tig]);
            int a1 = *reinterpret_cast<const int*>(&smA[warp_m + g + 8][kb * 32 + 4 * tig]);
            int a2 = *reinterpret_cast<const int*>(&smA[warp_m + g][kb * 32 + 16 + 4 * tig]);
            int a3 = *reinterpret_cast<const int*>(&smA[warp_m + g + 8][kb * 32 + 16 + 4 * tig]);
            float daLo = smDa[warp_m + g][kb];
            float daHi = smDa[warp_m + g + 8][kb];

#pragma unroll
            for (int nf = 0; nf < TS_MMQ_NFRAG; nf++)
            {
                const uint8_t* bp = &smBraw[cur][nf * 8 + g][pa + kb * 34 + 2 + 4 * tig];
                int b0r, b1r;
                if ((kb & 1) != 0)
                {
                    // qs offset pa + kb*34 + 2 is 0 mod 4 for odd kb.
                    b0r = *reinterpret_cast<const int*>(bp);
                    b1r = *reinterpret_cast<const int*>(bp + 16);
                }
                else
                {
                    // ... and 2 mod 4 for even kb: assemble from u16 pairs.
                    const uint16_t* hp = reinterpret_cast<const uint16_t*>(bp);
                    b0r = (int)hp[0] | ((int)hp[1] << 16);
                    b1r = (int)hp[8] | ((int)hp[9] << 16);
                }

                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0r), "r"(b1r));

                float dw0 = smDw[nf * 8 + 2 * tig][kb];
                float dw1 = smDw[nf * 8 + 2 * tig + 1][kb];
                fc[nf][0] += (float)c0 * dw0 * daLo;
                fc[nf][1] += (float)c1 * dw1 * daLo;
                fc[nf][2] += (float)c2 * dw0 * daHi;
                fc[nf][3] += (float)c3 * dw1 * daHi;
            }
        }
        __syncthreads();
        if (hasNext)
        {
            stageA(b0 + TS_MMQ_KSTEP);
            commitDw();
            ts_cp_async_commit();
            ts_cp_async_wait_all();
            __syncthreads();
            cur ^= 1;
        }
    }

    int out_r0 = m0 + warp_m + g;
    int out_r1 = out_r0 + 8;
#pragma unroll
    for (int nf = 0; nf < TS_MMQ_NFRAG; nf++)
    {
        int out_c0 = n0 + nf * 8 + 2 * tig;
        int out_c1 = out_c0 + 1;
        if (out_r0 < rows)
        {
            if (out_c0 < out_dim) output[(size_t)out_r0 * out_dim + out_c0] = fc[nf][0];
            if (out_c1 < out_dim) output[(size_t)out_r0 * out_dim + out_c1] = fc[nf][1];
        }
        if (out_r1 < rows)
        {
            if (out_c0 < out_dim) output[(size_t)out_r1 * out_dim + out_c0] = fc[nf][2];
            if (out_c1 < out_dim) output[(size_t)out_r1 * out_dim + out_c1] = fc[nf][3];
        }
    }
#else
    (void)weights; (void)xq_qs; (void)xq_d; (void)output; (void)in_dim; (void)out_dim; (void)rows;
#endif
}

// Single-row (decode) Q8_0 matvec: four warps cooperate on one output column,
// following ggml MMVQ's four-adjacent-lanes-per-Q8-block mapping. A 4-lane
// group owns one 32-element block and each lane evaluates two dp4a groups
// (8 values), so 128 threads issue coalesced loads across 32 consecutive
// blocks instead of 32 lanes each serializing all eight dp4a instructions for
// one block. The final CTA reduction is small relative to the improved weight
// bandwidth. The activation is pre-quantized once to q8_1, as in ggml.
extern "C" __global__ void ts_quant_matmul_q8_0_vec_f32(
    const uint8_t* weights,
    const ts_block_q8_1* xq,
    float* output,
    int in_dim,
    int out_dim)
{
    int col = blockIdx.x;
    if (col >= out_dim)
        return;

    int q8_blocks = in_dim / 32;
    const uint8_t* w_row = weights + (size_t)col * (size_t)q8_blocks * 34;

    float acc = 0.0f;
    int lane_in_block = threadIdx.x & 3;
    int block_group = threadIdx.x >> 2;
    int groups_per_cta = blockDim.x >> 2;
    for (int ib = block_group; ib < q8_blocks; ib += groups_per_cta)
    {
        const uint8_t* wblk = w_row + (size_t)ib * 34;
        float dw = __half2float(*reinterpret_cast<const half*>(wblk));
        const ts_block_q8_1* ablk = &xq[ib];
        float dact = __half2float(ablk->d);
        int g = lane_in_block * 2;
        int s = dp4a_i8(
            get_int_b2(wblk + 2, g),
            get_int_b4(ablk->qs, g),
            0);
        s = dp4a_i8(
            get_int_b2(wblk + 2, g + 1),
            get_int_b4(ablk->qs, g + 1),
            s);
        acc += dw * dact * (float)s;
    }

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        output[col] = acc;
}

// dp4a (int8) Q4_K single-token decode matvec. The generic scalar path
// (ts_quant_matmul_vec_f32 -> qvalue_at) re-parses the Q4_K super-block header
// (d/dmin + the 6-bit sub-block scale/min) for EVERY weight nibble, which
// dominates decode on a Q4_K-heavy model (all of the 26B-A4B's projections are
// Q4_K) and leaves it ~2x behind ggml. This mirrors ggml's vec_dot_q4_K_q8_1:
// quantize the activation to q8_1 (32-value blocks aligned to Q4_K's 32-value
// sub-blocks) ONCE, then per sub-block s of super-block sb:
//   sumi_s = dp4a(nibbles_s, q8_s)                        (int8 SIMD dot)
//   y += d_sb * sc_s * d8_s * sumi_s  -  dmin_sb * m_s * s8_s
// (d8_s / s8_s are the q8_1 block's stored scale / d*sum). The min term is
// independent of the 4-bit value, so it is added once per sub-block (lane 0).
// Layout mirrors ts_quant_matmul_q8_0_vec_f32: 4 threads cooperate on one
// 32-value block (8 ints, 2 per thread). Numerically within the 8-bit activation
// round-trip of the scalar dequant path (same tolerance as the Q4_0/Q8_0 dp4a
// paths); TS_CUDA_Q4K_DP4A=0 reverts to the exact scalar kernel.
extern "C" __global__ void ts_quant_matmul_q4k_dp4a_f32(
    const uint8_t* weights,
    const ts_block_q8_1* xq,
    float* output,
    int in_dim,
    int out_dim)
{
    int col = blockIdx.x;
    if (col >= out_dim)
        return;

    int n_super = in_dim / 256;   // Q4_K super-blocks (256 values, 144 B each)
    int n_sub = in_dim / 32;      // 32-value sub-blocks == q8_1 blocks
    const uint8_t* w_row = weights + (size_t)col * (size_t)n_super * 144;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

    int lane_in_block = threadIdx.x & 3;   // 4 threads cooperate on one sub-block
    int block_group = threadIdx.x >> 2;
    int groups_per_cta = blockDim.x >> 2;

    for (int ib = block_group; ib < n_sub; ib += groups_per_cta)
    {
        int sb = ib >> 3;          // super-block index
        int ls = ib & 7;           // sub-block within the super-block (0..7)
        const uint8_t* sblock = w_row + (size_t)sb * 144;
        float d_sb = __half2float(*reinterpret_cast<const half*>(sblock));
        float dmin_sb = __half2float(*reinterpret_cast<const half*>(sblock + 2));
        const uint8_t* scales = sblock + 4;
        const uint8_t* qs = sblock + 16;

        int pair = ls >> 1;
        int shift = (ls & 1) * 4;                 // low nibble (even ls) / high (odd)
        const uint8_t* w4 = qs + (size_t)pair * 32;   // 32 bytes = 8 ints

        const ts_block_q8_1* ablk = &xq[ib];
        int g = lane_in_block * 2;
        int w0 = (get_int_b4(w4, g)     >> shift) & 0x0F0F0F0F;
        int w1 = (get_int_b4(w4, g + 1) >> shift) & 0x0F0F0F0F;
        int sumi = dp4a_i8(w0, get_int_b4(ablk->qs, g), 0);
        sumi = dp4a_i8(w1, get_int_b4(ablk->qs, g + 1), sumi);

        int sc = get_scale_min_k4(scales, ls);
        float d8 = __half2float(ablk->d);
        sumf_d += d_sb * (float)sc * d8 * (float)sumi;

        if (lane_in_block == 0)
        {
            int m = get_min_k4(scales, ls);
            float s8 = __half2float(ablk->s);
            sumf_m += dmin_sb * (float)m * s8;
        }
    }

    float acc = sumf_d - sumf_m;
    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        output[col] = acc;
}

// Decode-only Q5_K matvec over one globally quantized q8_1 activation row.
// Q5_K uses the same eight 32-value sub-block scales/mins as Q4_K, plus one
// high bit per value. Four neighboring threads reconstruct and dot one
// sub-block with dp4a, matching ggml-cuda's vec_dot_q5_K_q8_1 layout.
extern "C" __global__ void ts_quant_matmul_q5k_dp4a_f32(
    const uint8_t* weights,
    const ts_block_q8_1* xq,
    float* output,
    int in_dim,
    int out_dim)
{
    int col = blockIdx.x;
    if (col >= out_dim)
        return;

    int n_super = in_dim / 256;
    int n_sub = in_dim / 32;
    const uint8_t* w_row = weights + (size_t)col * (size_t)n_super * 176;

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
    int lane_in_block = threadIdx.x & 3;
    int block_group = threadIdx.x >> 2;
    int groups_per_cta = blockDim.x >> 2;

    for (int ib = block_group; ib < n_sub; ib += groups_per_cta)
    {
        int sb = ib >> 3;
        int ls = ib & 7;
        const uint8_t* sblock = w_row + (size_t)sb * 176;
        float d_sb = __half2float(*reinterpret_cast<const half*>(sblock));
        float dmin_sb = __half2float(*reinterpret_cast<const half*>(sblock + 2));
        const uint8_t* scales = sblock + 4;
        const uint8_t* qh = sblock + 16;
        const uint8_t* qs = sblock + 48;

        int pair = ls >> 1;
        int shift = (ls & 1) * 4;
        const uint8_t* w4 = qs + (size_t)pair * 32;
        const ts_block_q8_1* ablk = &xq[ib];
        int g = lane_in_block * 2;

        int high0 = ((get_int_b4(qh, g) >> ls) & 0x01010101) << 4;
        int high1 = ((get_int_b4(qh, g + 1) >> ls) & 0x01010101) << 4;
        int w0 = ((get_int_b4(w4, g) >> shift) & 0x0F0F0F0F) | high0;
        int w1 = ((get_int_b4(w4, g + 1) >> shift) & 0x0F0F0F0F) | high1;
        int sumi = dp4a_i8(w0, get_int_b4(ablk->qs, g), 0);
        sumi = dp4a_i8(w1, get_int_b4(ablk->qs, g + 1), sumi);

        int sc = get_scale_min_k4(scales, ls);
        float d8 = __half2float(ablk->d);
        sumf_d += d_sb * (float)sc * d8 * (float)sumi;

        if (lane_in_block == 0)
        {
            int m = get_min_k4(scales, ls);
            sumf_m += dmin_sb * (float)m * __half2float(ablk->s);
        }
    }

    float acc = block_reduce_sum(sumf_d - sumf_m);
    if (threadIdx.x == 0)
        output[col] = acc;
}

// Decode-only Q6_K matvec. Each q8_1 block spans two independently scaled
// 16-value Q6_K groups. A four-thread group reconstructs the signed 6-bit
// values in packed bytes and executes two dp4a instructions per thread.
extern "C" __global__ void ts_quant_matmul_q6k_dp4a_f32(
    const uint8_t* weights,
    const ts_block_q8_1* xq,
    float* output,
    int in_dim,
    int out_dim)
{
    int col = blockIdx.x;
    if (col >= out_dim)
        return;

    int n_super = in_dim / 256;
    int n_sub = in_dim / 32;
    const uint8_t* w_row = weights + (size_t)col * (size_t)n_super * 210;

    float acc = 0.0f;
    int lane_in_block = threadIdx.x & 3;
    int block_group = threadIdx.x >> 2;
    int groups_per_cta = blockDim.x >> 2;

    for (int ib = block_group; ib < n_sub; ib += groups_per_cta)
    {
        int sb = ib >> 3;
        int ls = ib & 7;
        const uint8_t* sblock = w_row + (size_t)sb * 210;
        const uint8_t* ql = sblock;
        const uint8_t* qh = sblock + 128;
        const int8_t* scales = reinterpret_cast<const int8_t*>(sblock + 192);
        float d_sb = __half2float(*reinterpret_cast<const half*>(sblock + 208));

        int half_idx = ls >> 2;
        int group = ls & 3;
        const uint8_t* ql_group = ql + half_idx * 64 + ((group & 1) ? 32 : 0);
        const uint8_t* qh_group = qh + half_idx * 32;
        int ql_shift = group >= 2 ? 4 : 0;
        int qh_shift = group * 2;

        const ts_block_q8_1* ablk = &xq[ib];
        int g = lane_in_block * 2;
        // block_q6_K is 210 bytes, so odd super-blocks are only 2-byte
        // aligned. Assemble these packed words bytewise rather than issuing
        // potentially misaligned 32-bit loads.
        int raw0 = ((read_u32_unaligned(ql_group + 4 * g) >> ql_shift) & 0x0F0F0F0F)
                 | (((read_u32_unaligned(qh_group + 4 * g) >> qh_shift) & 0x03030303) << 4);
        int raw1 = ((read_u32_unaligned(ql_group + 4 * (g + 1)) >> ql_shift) & 0x0F0F0F0F)
                 | (((read_u32_unaligned(qh_group + 4 * (g + 1)) >> qh_shift) & 0x03030303) << 4);
        int w0 = __vsubss4(raw0, 0x20202020);
        int w1 = __vsubss4(raw1, 0x20202020);
        int sumi = dp4a_i8(w0, get_int_b4(ablk->qs, g), 0);
        sumi = dp4a_i8(w1, get_int_b4(ablk->qs, g + 1), sumi);

        int sc = scales[half_idx * 8 + group * 2 + (lane_in_block >= 2 ? 1 : 0)];
        acc += d_sb * (float)sc * __half2float(ablk->d) * (float)sumi;
    }

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        output[col] = acc;
}

// dp4a (int8) Q4_0 GEMM ÔÇö the fast path for BOTH single-token decode (rows == 1)
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 720)
    // int8 tensor-core (wmma m16n16k16) MMA requires sm_72+. For older targets
    // (e.g. the compute_61 fallback when GPU arch detection is unavailable),
    // nvcuda::wmma is not declared, so compile an empty stub: the symbol still
    // exists for module.GetFunction, and the host only ever launches this kernel
    // when the opt-in TS_CUDA_Q8_MMA path runs on tensor-core hardware.
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
#else
    (void)weights; (void)xq; (void)output; (void)in_dim; (void)out_dim; (void)rows;
#endif
}

// =====================================================================
// ts_qk_norm_rope_neox_f32 ÔÇö Fused QK-RMSNorm + NeoX RoPE
// =====================================================================
// Fuses per-head RMSNorm and NeoX rotary position embeddings into a
// single kernel pass.  Eliminates the intermediate global-memory write
// of the normalized Q/K tensor and the separate RoPE kernel launch.
//
// Grid:  (rows,)       ÔÇö one block per row (= seqLen * numHeads)
// Block: (BlockSize,)  ÔÇö 256 threads
// Shared: cols * sizeof(float)  ÔÇö for normalized values + RoPE rotation
//
// rows    = seqLen * numHeads  (or seqLen * kvHeads)
// cols    = headDim            (must match rope_dims for full rotation)
// rope_half = rope_dims / 2    (number of rotary pairs)
// eps     = RMSNorm epsilon
// rope_base, rope_freq_scale = RoPE frequency parameters
// positions = int32 [rows]     ÔÇö token position for each row
// =====================================================================
extern "C" __global__ void ts_qk_norm_rope_neox_f32(
    float* data,
    const float* alpha,
    const int* positions,
    int rows,
    int cols,
    int rope_half,
    float eps,
    float rope_base,
    float rope_freq_scale)
{
    int row = blockIdx.x;
    if (row >= rows)
        return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    float* x = data + (size_t)row * cols;

    // Phase 1: Compute sum of squares for RMSNorm
    float sum_sq = 0.0f;
    for (int i = tid; i < cols; i += num_threads)
    {
        float v = x[i];
        sum_sq += v * v;
    }
    sum_sq = block_reduce_sum(sum_sq);

    __shared__ float inv_rms;
    if (tid == 0)
        inv_rms = rsqrtf(sum_sq / (float)cols + eps);
    __syncthreads();

    // Phase 2: Normalize and store in shared memory
    // Layout: smem[0..cols-1] = normalized values, smem[cols..cols+rope_half-1] = cos table, smem[cols+rope_half..cols+2*rope_half-1] = sin table
    extern __shared__ float smem[];
    for (int i = tid; i < cols; i += num_threads)
        smem[i] = x[i] * inv_rms * alpha[i];
    __syncthreads();

    // Phase 2b: Pre-compute RoPE cos/sin lookup table in shared memory
    // Replaces on-the-fly powf/cosf/sinf with 2 global loads per pair
    float* cos_table = smem + cols;
    float* sin_table = smem + cols + rope_half;
    int pos = positions[row];
    // Frequencies follow the rotated span (rope_dims = 2 * rope_half), NOT the
    // full head width: models with partial RoPE (rope_dims < headDim, e.g.
    // Qwen3.5's hybrid attention heads) scale the exponent by rope_dims. Using
    // cols here compressed every frequency, which mis-rotated positions and
    // degraded generations progressively with context depth.
    for (int j = tid; j < rope_half; j += num_threads)
    {
        float theta = (float)pos * powf(rope_base, -2.0f * (float)j / (float)(2 * rope_half));
        float angle = theta * rope_freq_scale;
        cos_table[j] = cosf(angle);
        sin_table[j] = sinf(angle);
    }
    __syncthreads();

    // Phase 3: Apply NeoX RoPE rotation on pairs via lookup table
    // NeoX layout: pair j means (smem[j], smem[j + rope_half])
    for (int j = tid; j < rope_half; j += num_threads)
    {
        float c = cos_table[j];
        float s = sin_table[j];

        float x0 = smem[j];
        float x1 = smem[j + rope_half];
        smem[j]              = x0 * c - x1 * s;
        smem[j + rope_half]  = x0 * s + x1 * c;
    }
    __syncthreads();

    // Phase 4: Write back to global memory
    for (int i = tid; i < cols; i += num_threads)
        x[i] = smem[i];
}

// =====================================================================
// ts_qwen35_gdn_fused_f32 ÔÇö Fused pack + GDN kernel
// =====================================================================
// Reads directly from raw projection buffers (qkv, z, beta, alpha)
// instead of a pre-packed buffer.  Eliminates the separate
// ts_qwen35_gdn_pack_inputs_f32 kernel launch and the intermediate
// packed buffer allocation.
//
// Grid:  (numVHeads, ceil(headVDim / num_warps), 1)
// Block: (512, 1, 1)
// Shared: (2 * headKDim + headVDim) floats
// =====================================================================

__device__ __forceinline__ float gdn_read_raw(
    const float* qkv, const float* z, const float* beta, const float* alpha,
    int s, int ch, int qkv_dim, int z_dim, int num_v_heads)
{
    if (ch < qkv_dim)
        return qkv[(size_t)s * qkv_dim + ch];
    ch -= qkv_dim;
    if (ch < z_dim)
        return z[(size_t)s * z_dim + ch];
    ch -= z_dim;
    if (ch < num_v_heads)
        return beta[(size_t)s * num_v_heads + ch];
    return alpha[(size_t)s * num_v_heads + (ch - num_v_heads)];
}

__device__ __forceinline__ float gdn_conv_channel_raw(
    const float* qkv, const float* z, const float* beta, const float* alpha,
    const float* conv_state, const float* conv_w,
    int s, int ch, int seq_len, int qkv_dim, int z_dim, int num_v_heads,
    int conv_kernel, int conv_write_idx)
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
            x = gdn_read_raw(qkv, z, beta, alpha, input_s, ch, qkv_dim, z_dim, num_v_heads);
        }
        acc += x * conv_w[(size_t)ch * conv_kernel + ki];
    }
    return silu(acc);
}

extern "C" __global__ void ts_qwen35_gdn_fused_f32(
    const float* qkv,
    const float* z,
    const float* beta,
    const float* alpha,
    float* conv_state,
    float* ssm_state,
    const float* conv_w,
    const float* dt_bias,
    const float* a_log,
    const float* ssm_norm,
    float* output,
    int seq_len,
    int qkv_dim,
    int z_dim,
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

    // ONE block per head (launcher passes gridDim.y == 1): the per-block
    // whole-state decay below and the block-wide core[] reduction are only
    // correct when a single block owns the head, so rows are covered by
    // striding warps instead of blockIdx.y tiles (the old multi-block layout
    // raced the decay and read a partially-filled core[]).
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = nthreads >> 5;

    extern __shared__ float scratch[];
    float* q = scratch;
    float* k = q + head_k_dim;
    float* core = k + head_k_dim;

    __shared__ float q_scale;
    __shared__ float k_scale;
    __shared__ float gate_h;
    __shared__ float beta_h;
    __shared__ float rms_inv;

    int src_h = h % num_k_heads;
    int q_offset = src_h * head_k_dim;
    int k_offset = qk_dim + src_h * head_k_dim;
    int v_offset = 2 * qk_dim + h * head_v_dim;
    int state_per_head = head_v_dim * head_k_dim;
    float* state_head = ssm_state + (size_t)h * state_per_head;
    float q_head_scale = rsqrtf((float)head_v_dim);

    for (int s = 0; s < seq_len; s++)
    {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        for (int d = tid; d < head_k_dim; d += nthreads)
        {
            float qv = gdn_conv_channel_raw(
                qkv, z, beta, alpha, conv_state, conv_w,
                s, q_offset + d, seq_len, qkv_dim, z_dim, num_v_heads,
                conv_kernel, conv_write_idx);
            float kv = gdn_conv_channel_raw(
                qkv, z, beta, alpha, conv_state, conv_w,
                s, k_offset + d, seq_len, qkv_dim, z_dim, num_v_heads,
                conv_kernel, conv_write_idx);
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
            float alpha_val = gdn_read_raw(qkv, z, beta, alpha, s,
                qkv_dim + v_dim + num_v_heads + h, qkv_dim, z_dim, num_v_heads);
            float beta_val = gdn_read_raw(qkv, z, beta, alpha, s,
                qkv_dim + v_dim + h, qkv_dim, z_dim, num_v_heads);
            gate_h = softplus_f32(alpha_val + dt_bias[h]) * a_log[h];
            beta_h = sigmoid_f32(beta_val);
        }
        __syncthreads();

        float state_scale = expf(gate_h);
        for (int d = tid; d < head_k_dim; d += nthreads)
        {
            q[d] *= q_scale;
            k[d] *= k_scale;
        }
        for (int i = tid; i < state_per_head; i += nthreads)
            state_head[i] *= state_scale;
        __syncthreads();

        float bval = beta_h;
        for (int row = warp; row < head_v_dim; row += num_warps)
        {
            float* state_row = state_head + (size_t)row * head_k_dim;
            float kv_mem = 0.0f;
            for (int d = lane; d < head_k_dim; d += 32)
                kv_mem += state_row[d] * k[d];
            kv_mem = warp_allreduce_sum(kv_mem);

            float vrow;
            if (lane == 0)
                vrow = gdn_conv_channel_raw(
                    qkv, z, beta, alpha, conv_state, conv_w,
                    s, v_offset + row, seq_len, qkv_dim, z_dim, num_v_heads,
                    conv_kernel, conv_write_idx);
            vrow = __shfl_sync(0xFFFFFFFF, vrow, 0);
            float delta = (vrow - kv_mem) * bval;

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
            float z_val = gdn_read_raw(qkv, z, beta, alpha, s,
                qkv_dim + h * head_v_dim + row, qkv_dim, z_dim, num_v_heads);
            out_row[row] = core[row] * rms_inv * ssm_norm[row] * silu(z_val);
        }
        __syncthreads();
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

// ============================================================================
// On-device MoE decode (Gemma 4). Routing AND the expert FFN run entirely on the
// GPU so the whole decode layer loop is CUDA-graph capturable: no host readback
// of the router logits (the old MoERoute did a DtoH sync + CPU softmax/top-k)
// and no host gather/scatter of expert rows. The per-expert quantized weights
// stay device-resident exactly as preloaded; the FFN kernels pick the active
// expert's weight base pointer from a device pointer table indexed by the
// on-device top-k result, so a single captured graph replays for every token
// regardless of which experts it routes to.
// ============================================================================

// Router: top-k over the expert logits with the weights renormalized over the
// selected experts (== softmax over the selected logits) and the per-expert
// output scale folded in. Mirrors Gemma4Model.MoERoute's selected set + weights
// exactly (SelectTopKInPlace's strict-'>' first-seen-wins tie-break; softmax
// over selected == full-softmax-then-renormalize-over-selected). One block; the
// serial scan runs in thread 0 (num_experts is small, e.g. 128), which keeps the
// tie-break bit-identical to the CPU reference.
extern "C" __global__ void ts_moe_router_f32(
    const float* logits,            // [num_experts]
    const float* per_expert_scale,  // [num_experts] or nullptr
    int* selected_experts,          // [n_used]
    float* routing_weights,         // [n_used]
    int num_experts,
    int n_used)
{
    if (threadIdx.x != 0)
        return;

    const int MAX_K = 32;
    float top_val[MAX_K];
    int   top_idx[MAX_K];
    int k = n_used < MAX_K ? n_used : MAX_K;
    for (int i = 0; i < k; i++) { top_val[i] = -FLT_MAX; top_idx[i] = -1; }

    for (int e = 0; e < num_experts; e++)
    {
        float v = logits[e];
        int min_slot = 0;
        for (int j = 1; j < k; j++)
            if (top_val[j] < top_val[min_slot]) min_slot = j;
        if (v > top_val[min_slot]) { top_val[min_slot] = v; top_idx[min_slot] = e; }
    }

    float max_sel = -FLT_MAX;
    for (int i = 0; i < k; i++) max_sel = fmaxf(max_sel, top_val[i]);
    float sum = 0.0f;
    for (int i = 0; i < k; i++) { float ex = expf(top_val[i] - max_sel); top_val[i] = ex; sum += ex; }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    for (int i = 0; i < k; i++)
    {
        int e = top_idx[i];
        float w = top_val[i] * inv;
        if (per_expert_scale != nullptr && e >= 0)
            w *= per_expert_scale[e];
        selected_experts[i] = e;
        routing_weights[i] = w;
    }
}

// Gate/up projection for the selected experts. Each warp computes one output
// column and a CTA computes blockDim.x/32 adjacent columns. The old
// one-CTA-per-column layout used 128-256 threads for a dot with only 40-90
// quant blocks, leaving most lanes idle and creating hundreds of thousands of
// tiny CTAs per decoded token. Warp-owned rows match the organization of the
// mature mul_mat_vec kernels and keep several independent weight streams in
// flight per CTA. The expert weight base pointer is read from a device pointer
// table indexed by the on-device expert id, so nothing about the launch depends
// on a host-side router result.
extern "C" __global__ void ts_moe_expert_gate_up_vec_f32(
    const unsigned long long* expert_weight_ptrs, // [num_experts]
    const int* selected_experts,                  // [n_used]
    const float* input,                           // [in_dim] (RMSNorm'd MoE input row)
    float* gate_up_out,                           // [n_used * out_dim]
    int type,
    int in_dim,
    int out_dim)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int out_col = blockIdx.x * warps_per_block + warp;
    int slot = blockIdx.y;
    if (out_col >= out_dim)
        return;

    int e = selected_experts[slot];
    if (e < 0)
    {
        if (lane == 0)
            gate_up_out[(size_t)slot * out_dim + out_col] = 0.0f;
        return;
    }

    const uint8_t* w = reinterpret_cast<const uint8_t*>(expert_weight_ptrs[e]);
    int row_bytes = qrow_bytes(type, in_dim);
    const uint8_t* w_row = w + (size_t)out_col * row_bytes;

    float acc = 0.0f;
    for (int kk = lane; kk < in_dim; kk += 32)
        acc += qvalue_at(w_row, type, kk) * input[kk];

    acc = warp_allreduce_sum(acc);
    if (lane == 0)
        gate_up_out[(size_t)slot * out_dim + out_col] = acc;
}

// Down projection for the selected experts + weighted accumulation into the
// MoE output. Each warp owns one output element and loops the n_used experts,
// accumulating routing_weight[slot] * (W_down[e_slot] . h_slot).
// Looping the experts inside the warp (rather than scattering with atomics)
// keeps the accumulation deterministic and matches the CPU reference's order.
extern "C" __global__ void ts_moe_expert_down_accum_f32(
    const unsigned long long* expert_weight_ptrs, // [num_experts]
    const int* selected_experts,                  // [n_used]
    const float* routing_weights,                 // [n_used]
    const float* h_all,                           // [n_used * in_dim]
    float* output,                                // [out_dim]
    int type,
    int in_dim,
    int out_dim,
    int n_used)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int out_col = blockIdx.x * warps_per_block + warp;
    if (out_col >= out_dim)
        return;

    int row_bytes = qrow_bytes(type, in_dim);
    float acc = 0.0f;

    for (int slot = 0; slot < n_used; slot++)
    {
        int e = selected_experts[slot];
        if (e < 0)
            continue;
        float w = routing_weights[slot];
        const uint8_t* wp = reinterpret_cast<const uint8_t*>(expert_weight_ptrs[e]);
        const uint8_t* w_row = wp + (size_t)out_col * row_bytes;
        const float* h = h_all + (size_t)slot * in_dim;

        float partial = 0.0f;
        for (int kk = lane; kk < in_dim; kk += 32)
            partial += qvalue_at(w_row, type, kk) * h[kk];
        acc += w * partial;
    }

    acc = warp_allreduce_sum(acc);
    if (lane == 0)
        output[out_col] = acc;
}

// One Q4_K 32-value sub-block dot against a q8_1 activation block, returning the
// fully-scaled contribution d_sb*sc_ls*d8*sumi - dmin_sb*m_ls*s8 (see
// ts_quant_matmul_q4k_dp4a_f32 for the derivation). Shared by the on-device MoE
// dp4a kernels; `sblock` points at the 144-byte Q4_K super-block, `ls` is the
// 0..7 sub-block index within it.
__device__ __forceinline__ float q4k_subblock_dot_q8(
    const uint8_t* sblock, int ls, const ts_block_q8_1* ablk)
{
    float d_sb = __half2float(*reinterpret_cast<const half*>(sblock));
    float dmin_sb = __half2float(*reinterpret_cast<const half*>(sblock + 2));
    const uint8_t* scales = sblock + 4;
    const uint8_t* qs = sblock + 16;
    int pair = ls >> 1;
    int shift = (ls & 1) * 4;
    const uint8_t* w4 = qs + (size_t)pair * 32;

    int sumi = 0;
#pragma unroll
    for (int g = 0; g < 8; g++)
        sumi = dp4a_i8((get_int_b4(w4, g) >> shift) & 0x0F0F0F0F, get_int_b4(ablk->qs, g), sumi);

    int sc = get_scale_min_k4(scales, ls);
    int m = get_min_k4(scales, ls);
    float d8 = __half2float(ablk->d);
    float s8 = __half2float(ablk->s);
    return d_sb * (float)sc * d8 * (float)sumi - dmin_sb * (float)m * s8;
}

// One Q4_0 32-value block dot against a q8_1 activation block: d_w*(d8*dp4a - 8*s8)
// (the -8 zero-point carried through the q8_1 sum), mirroring
// ts_quant_matmul_q4_0_dp4a_f32 / ggml vec_dot_q4_0_q8_1. `block` is the 18-byte
// Q4_0 block (2-byte d + 16-byte qs).
__device__ __forceinline__ float q40_block_dot_q8(const uint8_t* block, const ts_block_q8_1* ablk)
{
    float dw = __half2float(*reinterpret_cast<const half*>(block));
    float dact = __half2float(ablk->d);
    float sact = __half2float(ablk->s);
    int s = 0;
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
        int w = get_int_b2(block + 2, j);            // 4 bytes = 8 nibbles
        int wlo = w & 0x0F0F0F0F;                     // low nibbles  -> q8[4j..4j+3]
        int whi = (w >> 4) & 0x0F0F0F0F;              // high nibbles -> q8[16+4j..]
        s = dp4a_i8(wlo, get_int_b4(ablk->qs, j), s);
        s = dp4a_i8(whi, get_int_b4(ablk->qs, j + 4), s);
    }
    return dw * (dact * (float)s - 8.0f * sact);
}

// One 32-value weight block dot against a q8_1 activation block, dispatching on
// the quant type (2 = Q4_0, 12 = Q4_K, 16 = IQ2_XXS, 22 = IQ2_S). `w_base` is the weight ROW
// base and `ib` the global 32-value sub-block index; `xq_row` is the ROW base of
// the q8_1 activation blocks (indexed by ib internally, so each type can also
// reach the enclosing super-block it belongs to). For Q4_0 each 32-value block is
// 18 bytes at ib. For Q4_K the 144-byte super-block holds 8 sub-blocks (ib & 7).
// For IQ2_XXS/IQ2_S the 66/82-byte super-block holds 8 groups (ib & 7); the
// vec-dots index the group's q8_1 block as
// xq_row[(ib>>3)*8 + group] == xq_row[ib].
__device__ __forceinline__ float moe_block_dot_q8(
    int type, const uint8_t* w_base, int ib, const ts_block_q8_1* xq_row)
{
    if (type == GGML_Q4_0)
        return q40_block_dot_q8(w_base + (size_t)ib * 18, &xq_row[ib]);
    if (type == GGML_IQ2_XXS)
        return dot_iq2_xxs_q8_1(w_base + (size_t)(ib >> 3) * 66, xq_row + (size_t)(ib >> 3) * 8, ib & 7);
    if (type == GGML_IQ2_S)
        return dot_iq2_s_q8_1(w_base + (size_t)(ib >> 3) * 82, xq_row + (size_t)(ib >> 3) * 8, ib & 7);
    // GGML_Q4_K
    return q4k_subblock_dot_q8(w_base + (size_t)(ib >> 3) * 144, ib & 7, &xq_row[ib]);
}

// dp4a gate/up projection for the selected experts (bulk of MoE decode cost).
// Each warp computes one (expert-slot, output-column) dot and a CTA computes
// blockDim.x/32 adjacent columns. The RMSNorm'd MoE input is pre-quantized to
// q8_1 once (xq_input, in_dim/32 blocks) and the expert's
// Q4_0/Q4_K/IQ2_XXS/IQ2_S
// weight is dp4a-dotted against it. in_dim is a multiple of the block size.
extern "C" __global__ void ts_moe_expert_gate_up_dp4a_f32(
    const unsigned long long* expert_weight_ptrs, // [num_experts]
    const int* selected_experts,                  // [n_used]
    const ts_block_q8_1* xq_input,                // [in_dim/32] q8_1 of the MoE input
    float* gate_up_out,                           // [n_used * out_dim]
    int type,
    int in_dim,
    int out_dim)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int out_col = blockIdx.x * warps_per_block + warp;
    int slot = blockIdx.y;
    if (out_col >= out_dim)
        return;

    int e = selected_experts[slot];
    if (e < 0)
    {
        if (lane == 0)
            gate_up_out[(size_t)slot * out_dim + out_col] = 0.0f;
        return;
    }

    const uint8_t* w_row = reinterpret_cast<const uint8_t*>(expert_weight_ptrs[e])
        + (size_t)out_col * (size_t)qrow_bytes(type, in_dim);
    int n_sub = in_dim / 32;

    float acc = 0.0f;
    for (int ib = lane; ib < n_sub; ib += 32)
        acc += moe_block_dot_q8(type, w_row, ib, xq_input);

    acc = warp_allreduce_sum(acc);
    if (lane == 0)
        gate_up_out[(size_t)slot * out_dim + out_col] = acc;
}

// dp4a down projection + routing-weighted accumulation. Each warp owns one
// output element and loops the selected experts, dp4a-dotting each expert's
// Q4_0/Q4_K/IQ2_XXS/IQ2_S down weight against that expert's q8_1-quantized activation
// (xq_h, laid out [n_used][in_dim/32]).
extern "C" __global__ void ts_moe_expert_down_dp4a_f32(
    const unsigned long long* expert_weight_ptrs, // [num_experts]
    const int* selected_experts,                  // [n_used]
    const float* routing_weights,                 // [n_used]
    const ts_block_q8_1* xq_h,                    // [n_used * (in_dim/32)]
    float* output,                                // [out_dim]
    int type,
    int in_dim,
    int out_dim,
    int n_used)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    int out_col = blockIdx.x * warps_per_block + warp;
    if (out_col >= out_dim)
        return;

    int n_sub = in_dim / 32;
    size_t row_bytes = (size_t)qrow_bytes(type, in_dim);
    float acc = 0.0f;

    for (int slot = 0; slot < n_used; slot++)
    {
        int e = selected_experts[slot];
        if (e < 0)
            continue;
        float rw = routing_weights[slot];
        const uint8_t* w_row = reinterpret_cast<const uint8_t*>(expert_weight_ptrs[e])
            + (size_t)out_col * row_bytes;
        const ts_block_q8_1* xqs = xq_h + (size_t)slot * n_sub;

        float partial = 0.0f;
        for (int ib = lane; ib < n_sub; ib += 32)
            partial += moe_block_dot_q8(type, w_row, ib, xqs);
        acc += rw * partial;
    }

    acc = warp_allreduce_sum(acc);
    if (lane == 0)
        output[out_col] = acc;
}

// Elementwise SwiGLU combine over two SEPARATE buffers (Qwen MoE keeps gate and
// up expert weights unfused): dst[i] = silu(a[i]) * b[i]. Used by the on-device
// SwiGLU MoE decode after the gate and up projections write distinct [n_used,n_ff]
// tensors. In-place safe when dst == a.
extern "C" __global__ void ts_silu_mul_f32(float* dst, const float* a, const float* b, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float x = a[i];
    dst[i] = silu(x) * b[i];
}

// Shared-expert gated accumulate for the on-device SwiGLU MoE decode. Computes the
// per-token gate scalar g = sigmoid(sum_i input[i]*gate_vec[i]) entirely on device
// (one block, block_reduce_sum over gate_dim) and folds output[j] += g*shared_down[j].
// gate_vec == nullptr => ungated shared expert (g = 1). Mirrors the host
// SigmoidScalar(VecDot(...)) + VecScaleAdd path in Qwen35Model.MoEForward without a
// host round-trip, so the decode layer loop stays CUDA-graph capturable.
extern "C" __global__ void ts_moe_shared_gated_add(
    float* output,             // [hidden] accumulate into
    const float* shared_down,  // [hidden]
    const float* input,        // [gate_dim] MoE input row (nullptr if ungated)
    const float* gate_vec,     // [gate_dim] shared-gate weights (nullptr if ungated)
    int hidden,
    int gate_dim)
{
    __shared__ float g_shared;
    if (gate_vec != nullptr && input != nullptr)
    {
        float s = 0.0f;
        for (int i = threadIdx.x; i < gate_dim; i += blockDim.x)
            s += input[i] * gate_vec[i];
        s = block_reduce_sum(s);
        if (threadIdx.x == 0)
            g_shared = 1.0f / (1.0f + __expf(-s));
    }
    else if (threadIdx.x == 0)
    {
        g_shared = 1.0f;
    }
    __syncthreads();

    float g = g_shared;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x)
        output[j] += g * shared_down[j];
}

// ============================================================================
// Batched (multi-token) on-device SwiGLU MoE for PREFILL. Same math as the
// single-token decode kernels above, with a token dimension added so the whole
// prefill MoE (routing + gate/up + SwiGLU + down + shared) runs on device with no
// host gather/scatter/routing round-trip. Each token independently routes its
// top-k experts; weights are re-read per (token, expert) — no cross-token batched
// reuse yet, but the elimination of the per-expert host readbacks (which were 82%
// of MoE time) and the resulting graph-capturability are the win.
// ============================================================================

// Per-token router: grid.x = num_tokens, one block per token (serial top-k in
// thread 0, matching SelectTopKInPlace's strict-'>' first-seen-wins).
extern "C" __global__ void ts_moe_router_batched_f32(
    const float* logits,            // [num_tokens * num_experts]
    const float* per_expert_scale,  // [num_experts] or nullptr
    int* selected_experts,          // [num_tokens * n_used]
    float* routing_weights,         // [num_tokens * n_used]
    int num_experts,
    int n_used,
    int num_tokens)
{
    int t = blockIdx.x;
    if (t >= num_tokens || threadIdx.x != 0)
        return;

    const float* tlogits = logits + (size_t)t * num_experts;
    int* tsel = selected_experts + (size_t)t * n_used;
    float* trw = routing_weights + (size_t)t * n_used;

    const int MAX_K = 32;
    float top_val[MAX_K];
    int   top_idx[MAX_K];
    int k = n_used < MAX_K ? n_used : MAX_K;
    for (int i = 0; i < k; i++) { top_val[i] = -FLT_MAX; top_idx[i] = -1; }

    for (int e = 0; e < num_experts; e++)
    {
        float v = tlogits[e];
        int min_slot = 0;
        for (int j = 1; j < k; j++)
            if (top_val[j] < top_val[min_slot]) min_slot = j;
        if (v > top_val[min_slot]) { top_val[min_slot] = v; top_idx[min_slot] = e; }
    }

    float max_sel = -FLT_MAX;
    for (int i = 0; i < k; i++) max_sel = fmaxf(max_sel, top_val[i]);
    float sum = 0.0f;
    for (int i = 0; i < k; i++) { float ex = expf(top_val[i] - max_sel); top_val[i] = ex; sum += ex; }
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;

    for (int i = 0; i < k; i++)
    {
        int e = top_idx[i];
        float w = top_val[i] * inv;
        if (per_expert_scale != nullptr && e >= 0)
            w *= per_expert_scale[e];
        tsel[i] = e;
        trw[i] = w;
    }
}

// Gate/up projection, dp4a. grid = (out_dim, n_used, num_tokens).
extern "C" __global__ void ts_moe_expert_gate_up_batched_dp4a_f32(
    const unsigned long long* expert_weight_ptrs,
    const int* selected_experts,   // [num_tokens * n_used]
    const ts_block_q8_1* xq_input, // [num_tokens * (in_dim/32)]
    float* gate_up_out,            // [num_tokens * n_used * out_dim]
    int type, int in_dim, int out_dim, int n_used)
{
    int out_col = blockIdx.x;
    int slot = blockIdx.y;
    int t = blockIdx.z;
    if (out_col >= out_dim)
        return;

    int e = selected_experts[(size_t)t * n_used + slot];
    size_t out_off = ((size_t)t * n_used + slot) * out_dim + out_col;
    if (e < 0)
    {
        if (threadIdx.x == 0) gate_up_out[out_off] = 0.0f;
        return;
    }

    const uint8_t* w_row = reinterpret_cast<const uint8_t*>(expert_weight_ptrs[e])
        + (size_t)out_col * (size_t)qrow_bytes(type, in_dim);
    const ts_block_q8_1* xq_row = xq_input + (size_t)t * (in_dim / 32);
    int n_sub = in_dim / 32;

    float acc = 0.0f;
    for (int ib = threadIdx.x; ib < n_sub; ib += blockDim.x)
        acc += moe_block_dot_q8(type, w_row, ib, xq_row);

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        gate_up_out[out_off] = acc;
}

// Gate/up projection, scalar dequant. grid = (out_dim, n_used, num_tokens).
extern "C" __global__ void ts_moe_expert_gate_up_batched_vec_f32(
    const unsigned long long* expert_weight_ptrs,
    const int* selected_experts,   // [num_tokens * n_used]
    const float* input,            // [num_tokens * in_dim]
    float* gate_up_out,            // [num_tokens * n_used * out_dim]
    int type, int in_dim, int out_dim, int n_used)
{
    int out_col = blockIdx.x;
    int slot = blockIdx.y;
    int t = blockIdx.z;
    if (out_col >= out_dim)
        return;

    int e = selected_experts[(size_t)t * n_used + slot];
    size_t out_off = ((size_t)t * n_used + slot) * out_dim + out_col;
    if (e < 0)
    {
        if (threadIdx.x == 0) gate_up_out[out_off] = 0.0f;
        return;
    }

    const uint8_t* w_row = reinterpret_cast<const uint8_t*>(expert_weight_ptrs[e])
        + (size_t)out_col * (size_t)qrow_bytes(type, in_dim);
    const float* in_row = input + (size_t)t * in_dim;

    float acc = 0.0f;
    for (int kk = threadIdx.x; kk < in_dim; kk += blockDim.x)
        acc += qvalue_at(w_row, type, kk) * in_row[kk];

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        gate_up_out[out_off] = acc;
}

// Down projection + routing-weighted accumulate, dp4a. grid = (out_dim, num_tokens).
extern "C" __global__ void ts_moe_expert_down_batched_dp4a_f32(
    const unsigned long long* expert_weight_ptrs,
    const int* selected_experts,   // [num_tokens * n_used]
    const float* routing_weights,  // [num_tokens * n_used]
    const ts_block_q8_1* xq_h,     // [num_tokens * n_used * (in_dim/32)]
    float* output,                 // [num_tokens * out_dim]
    int type, int in_dim, int out_dim, int n_used)
{
    int out_col = blockIdx.x;
    int t = blockIdx.y;
    if (out_col >= out_dim)
        return;

    int n_sub = in_dim / 32;
    size_t row_bytes = (size_t)qrow_bytes(type, in_dim);
    const int* sel = selected_experts + (size_t)t * n_used;
    const float* rw = routing_weights + (size_t)t * n_used;
    float acc = 0.0f;

    for (int slot = 0; slot < n_used; slot++)
    {
        int e = sel[slot];
        if (e < 0) continue;
        const uint8_t* w_row = reinterpret_cast<const uint8_t*>(expert_weight_ptrs[e]) + (size_t)out_col * row_bytes;
        const ts_block_q8_1* xqs = xq_h + ((size_t)t * n_used + slot) * n_sub;
        float partial = 0.0f;
        for (int ib = threadIdx.x; ib < n_sub; ib += blockDim.x)
            partial += moe_block_dot_q8(type, w_row, ib, xqs);
        acc += rw[slot] * partial;
    }

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        output[(size_t)t * out_dim + out_col] = acc;
}

// Down projection + routing-weighted accumulate, scalar. grid = (out_dim, num_tokens).
extern "C" __global__ void ts_moe_expert_down_batched_accum_f32(
    const unsigned long long* expert_weight_ptrs,
    const int* selected_experts,   // [num_tokens * n_used]
    const float* routing_weights,  // [num_tokens * n_used]
    const float* h_all,            // [num_tokens * n_used * in_dim]
    float* output,                 // [num_tokens * out_dim]
    int type, int in_dim, int out_dim, int n_used)
{
    int out_col = blockIdx.x;
    int t = blockIdx.y;
    if (out_col >= out_dim)
        return;

    size_t row_bytes = (size_t)qrow_bytes(type, in_dim);
    const int* sel = selected_experts + (size_t)t * n_used;
    const float* rw = routing_weights + (size_t)t * n_used;
    float acc = 0.0f;

    for (int slot = 0; slot < n_used; slot++)
    {
        int e = sel[slot];
        if (e < 0) continue;
        const uint8_t* w_row = reinterpret_cast<const uint8_t*>(expert_weight_ptrs[e]) + (size_t)out_col * row_bytes;
        const float* h = h_all + ((size_t)t * n_used + slot) * in_dim;
        float partial = 0.0f;
        for (int kk = threadIdx.x; kk < in_dim; kk += blockDim.x)
            partial += qvalue_at(w_row, type, kk) * h[kk];
        acc += rw[slot] * partial;
    }

    acc = block_reduce_sum(acc);
    if (threadIdx.x == 0)
        output[(size_t)t * out_dim + out_col] = acc;
}

// Batched shared-expert gated accumulate. grid.x = num_tokens, one block per token.
// input/output/shared_down rows are hidden-wide; the gate dots the first gate_dim.
extern "C" __global__ void ts_moe_shared_gated_add_batched(
    float* output,             // [num_tokens * hidden]
    const float* shared_down,  // [num_tokens * hidden]
    const float* input,        // [num_tokens * hidden] (nullptr if ungated)
    const float* gate_vec,     // [gate_dim] (nullptr if ungated)
    int hidden,
    int gate_dim,
    int num_tokens)
{
    int t = blockIdx.x;
    if (t >= num_tokens)
        return;

    float* out_row = output + (size_t)t * hidden;
    const float* sd_row = shared_down + (size_t)t * hidden;

    __shared__ float g_shared;
    if (gate_vec != nullptr && input != nullptr)
    {
        const float* in_row = input + (size_t)t * hidden;
        float s = 0.0f;
        for (int i = threadIdx.x; i < gate_dim; i += blockDim.x)
            s += in_row[i] * gate_vec[i];
        s = block_reduce_sum(s);
        if (threadIdx.x == 0)
            g_shared = 1.0f / (1.0f + __expf(-s));
    }
    else if (threadIdx.x == 0)
    {
        g_shared = 1.0f;
    }
    __syncthreads();

    float g = g_shared;
    for (int j = threadIdx.x; j < hidden; j += blockDim.x)
        out_row[j] += g * sd_row[j];
}

// Scatter one expert's grouped output rows back into the token-major MoE
// accumulator. row_indices are unique within one expert batch (top-k routing
// selects an expert at most once for a token), and expert batches are launched
// serially on one stream, so the non-atomic add is both safe and substantially
// cheaper than an atomic scatter.
extern "C" __global__ void ts_moe_scatter_add_weighted_rows_f32(
    float* output,                // [num_tokens * hidden]
    const float* expert_output,   // [batch_size * hidden]
    const int* row_indices,       // [batch_size]
    const float* routing_weights, // [batch_size]
    int batch_size,
    int num_tokens,
    int hidden)
{
    int row = blockIdx.x;
    if (row >= batch_size)
        return;

    int token = row_indices[row];
    if (token < 0 || token >= num_tokens)
        return;

    float* dst = output + (size_t)token * hidden;
    const float* src = expert_output + (size_t)row * hidden;
    float weight = routing_weights[row];
    for (int col = threadIdx.x; col < hidden; col += blockDim.x)
        dst[col] += weight * src[col];
}
