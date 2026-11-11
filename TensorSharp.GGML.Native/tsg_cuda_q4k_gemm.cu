// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Large Q4_K projections on cuBLAS BF16 tensor cores.
//
// MiniMax-H3's DiT multiplies 20,511 tokens (480x864x158) by 19.3 B Q4_K weights
// every denoising step. ggml-cuda runs those on MMQ (int8 tensor cores over q8_1
// activations) at about 52 TOPS where cuBLAS ran a BF16 GEMM of the same shape at
// about 89 TFLOP/s on the same GPU. For a matmul this wide, re-expanding the weights
// on every call costs little: the fused qkv projection (21,504 x 5,376) dequantizes
// in about 1 ms against a ~55 ms GEMM. So the weights stay Q4_K in memory, and each
// call
//   1. dequantizes the Q4_K rows to BF16 into scratch (exactly ggml's
//      dequantize_row_q4_K arithmetic in F32, rounded to BF16),
//   2. narrows the F32 activations to BF16 (BF16 keeps F32's exponent range, so
//      H3's unbounded activations cannot overflow as they could in F16), and
//   3. runs one cublasGemmEx with F32 accumulation and an F32 result.
//
// Numerics differ from MMQ, which rounds activations to 8-bit q8_1 blocks: BF16
// keeps 8 significant bits per element instead of per 32-element block. The
// caller (graph_compute_offload) decides which nodes take this route;
// TS_H3_CUBLAS_MATMUL=0 keeps them on ggml.
//
// TS_H3_FP8_MATMUL=1 multiplies FP8 (E4M3) operands through cuBLASLt instead, on GPUs
// that have such a kernel: one byte per element in place of two, at about twice the
// BF16 GEMM's rate. It is opt-in because it changes the result: E4M3 keeps 4
// significant bits, and on this model's projections the product came out 27 to 37 dB
// (31 on average) above its own rounding error. See fp8_gemm.
//
// TS_H3_INT8_MATMUL=1 multiplies 8-bit integers through cuBLASLt instead, which a GPU
// without FP8 kernels (sm_80: A100, RTX 30) has tensor cores for as well: each weight
// row and each token's activations are rounded to int8 against their own peak, the
// product accumulates exactly in int32, and one pass scales it back to F32. It is
// opt-in for the same reason: the result changes. See int8_gemm.
#if defined(TSG_GGML_USE_CUDA)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

// cuBLASLt has taken FP8 operands (CUDA_R_8F_E4M3) since CUDA 11.8.
#if CUBLAS_VER_MAJOR >= 12
#include <cublasLt.h>
#define TSG_CUDA_FP8_GEMM 1
#endif

#include "tsg_cuda_stage_profile.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

// ggml's block_q4_K (ggml-common.h): 256 weights in 144 bytes.
struct BlockQ4K
{
    __half d;               // super-block scale for the 6-bit scales
    __half dmin;            // super-block scale for the 6-bit mins
    uint8_t scales[12];     // 8 (scale, min) pairs, 6 bits each
    uint8_t qs[128];        // 256 4-bit quants
};
static_assert(sizeof(BlockQ4K) == 144, "block_q4_K layout");

// The two weights of quant byte b (0..127) of block blk, in F32: byte b holds weight l
// of 64-chunk j in its low nibble and weight 32 + l in its high nibble (b = 32 j + l).
// `lo` belongs at element `at` of the row-major weight and `hi` at `at + 32`. With
// `delta` (an F32 [N, K] LoRA update, up * down), both are the merged weight.
__device__ __forceinline__ void q4k_pair(const BlockQ4K* __restrict__ w, const float* __restrict__ delta,
                                         long long blk, int b, long long& at, float& lo, float& hi)
{
    const int j = b >> 5;               // 64-weight chunk
    const int l = b & 31;
    const BlockQ4K& x = w[blk];
    const float d = __half2float(x.d);
    const float dmin = __half2float(x.dmin);
    const uint8_t* q = x.scales;
    // get_scale_min_k4 for sub-blocks 2j (low nibbles) and 2j + 1 (high nibbles).
    auto scale_min = [&](int s, uint8_t& sc, uint8_t& m) {
        if (s < 4) { sc = q[s] & 63; m = q[s + 4] & 63; }
        else { sc = (q[s + 4] & 0xF) | ((q[s - 4] >> 6) << 4); m = (q[s + 4] >> 4) | ((q[s] >> 6) << 4); }
    };
    uint8_t sc0, m0, sc1, m1;
    scale_min(2 * j, sc0, m0);
    scale_min(2 * j + 1, sc1, m1);
    const uint8_t v = x.qs[b];
    const float d1 = d * sc0, mm1 = dmin * m0;
    const float d2 = d * sc1, mm2 = dmin * m1;
    const long long base = blk * 256 + j * 64;
    lo = d1 * (v & 0xF) - mm1;
    hi = d2 * (v >> 4) - mm2;
    if (delta != nullptr) { lo += delta[base + l]; hi += delta[base + 32 + l]; }
    at = base + l;
}

// One thread per quant byte; the merged weight is rounded to BF16 once.
__global__ void dequant_q4k_bf16(const BlockQ4K* __restrict__ w, const float* __restrict__ delta,
                                 __nv_bfloat16* __restrict__ out, long long blocks)
{
    const long long blk = blockIdx.x;
    if (blk >= blocks) return;
    long long at;
    float lo, hi;
    q4k_pair(w, delta, blk, threadIdx.x, at, lo, hi);
    out[at] = __float2bfloat16_rn(lo);
    out[at + 32] = __float2bfloat16_rn(hi);
}

// ggml's block_q6_K (ggml-common.h): 256 weights in 210 bytes, unaligned: the low
// four bits of each quant (128 bytes), the high two (64), sixteen int8 scales and the
// F16 super-block scale. Read as bytes.
enum : int { kQ6KBytes = 210, kQ6KHigh = 128, kQ6KScales = 192, kQ6KD = 208 };

// ggml's dequantize_row_q6_K in F32, rounded to BF16 once. One thread per (half, l)
// of a block writes that position's four weights.
__global__ void dequant_q6k_bf16(const uint8_t* __restrict__ w, __nv_bfloat16* __restrict__ out, long long blocks)
{
    const long long blk = blockIdx.x;
    if (blk >= blocks) return;
    const uint8_t* __restrict__ b = w + blk * kQ6KBytes;
    const int half = threadIdx.x >> 5;      // 128 weights each
    const int l = threadIdx.x & 31;
    const uint8_t* ql = b + 64 * half;
    const uint8_t* qh = b + kQ6KHigh + 32 * half;
    const int8_t* sc = reinterpret_cast<const int8_t*>(b + kQ6KScales) + 8 * half;
    const float d = __half2float(__ushort_as_half((unsigned short) (b[kQ6KD] | (b[kQ6KD + 1] << 8))));
    const int is = l >> 4;
    const int h = qh[l];
    const int q1 = (int) ((ql[l] & 0xF) | ((h & 3) << 4)) - 32;
    const int q2 = (int) ((ql[l + 32] & 0xF) | (((h >> 2) & 3) << 4)) - 32;
    const int q3 = (int) ((ql[l] >> 4) | (((h >> 4) & 3) << 4)) - 32;
    const int q4 = (int) ((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) - 32;
    __nv_bfloat16* __restrict__ y = out + blk * 256 + 128 * half + l;
    y[0] = __float2bfloat16_rn(d * sc[is] * q1);
    y[32] = __float2bfloat16_rn(d * sc[is + 2] * q2);
    y[64] = __float2bfloat16_rn(d * sc[is + 4] * q3);
    y[96] = __float2bfloat16_rn(d * sc[is + 6] * q4);
}

__global__ void narrow_rows_bf16(const float* __restrict__ x, long long row_stride, __nv_bfloat16* __restrict__ out,
                                 long long cols, long long rows)
{
    const long long n = cols * rows;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x)
    {
        const long long r = i / cols, c = i - r * cols;
        out[i] = __float2bfloat16_rn(x[r * row_stride + c]);
    }
}

// fc2's input straight from fc1's output: each row holds [gate | value], `cols`
// each, and the activation is ggml-cuda's SwiGLU, silu(gate) * value with
// silu(x) = x / (1 + expf(-x)), computed in F32 and rounded to BF16 once. This
// replaces the graph's GGML_OP_GLU node and its F32 [cols, rows] result.
__global__ void narrow_swiglu_bf16(const float* __restrict__ gv, long long row_stride, __nv_bfloat16* __restrict__ out,
                                   long long cols, long long rows)
{
    const long long n = cols * rows;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x)
    {
        const long long r = i / cols, c = i - r * cols;
        const float g = gv[r * row_stride + c];
        const float v = gv[r * row_stride + cols + c];
        out[i] = __float2bfloat16_rn((g / (1.0f + expf(-g))) * v);
    }
}

// F16 narrowing for the video VAE's F16 projections, optionally adding a bias, or
// (swiglu) SwiGLU of [gate | value] rows with their bias: the F16 input ggml-cuda's
// own cuBLAS path for an F16 weight converts its F32 activations to.
__global__ void narrow_rows_f16(const float* __restrict__ x, long long row_stride, const float* __restrict__ bias,
                                int swiglu, __half* __restrict__ out, long long cols, long long rows)
{
    const long long n = cols * rows;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x)
    {
        const long long r = i / cols, c = i - r * cols;
        float v;
        if (swiglu)
        {
            float g = x[r * row_stride + c];
            float u = x[r * row_stride + cols + c];
            if (bias != nullptr) { g = __fadd_rn(g, bias[c]); u = __fadd_rn(u, bias[cols + c]); }
            v = (g / (1.0f + expf(-g))) * u;
        }
        else
        {
            v = x[r * row_stride + c];
            if (bias != nullptr) v = __fadd_rn(v, bias[c]);
        }
        out[i] = __float2half_rn(v);
    }
}

#if defined(TSG_CUDA_FP8_GEMM)

// F32 to FP8 E4M3 (sign, 4 exponent bits biased by 7, 3 mantissa bits; no infinities,
// largest value 448), round to nearest even, saturating. An F32's top 12 bits below
// the sign are its exponent and 3 mantissa bits, so a normal code is those bits less
// 8 * (127 - 7) after rounding at bit 20; what rounds below the smallest normal
// (2^-6) is a subnormal, a multiple of 2^-9.
__device__ __forceinline__ uint8_t e4m3_rn(float v)
{
    const uint32_t bits = __float_as_uint(v), u = bits & 0x7FFFFFFFu;
    uint32_t code;
    if (u >= 0x43E00000u) code = 0x7E;      // 448 and above
    else
    {
        const uint32_t r = (u + 0x7FFFFu + ((u >> 20) & 1u)) >> 20;
        code = r >= 968u ? r - 960u : (uint32_t) __float2int_rn(__uint_as_float(u) * 512.0f);
    }
    return (uint8_t) (code | ((bits >> 31) << 7));
}

// Magnitudes order like their bit patterns, and Inf or NaN sorts above every finite
// value, so the largest pattern seen says both how to scale an operand and whether it
// can be scaled at all.
__device__ __forceinline__ uint32_t magnitude_bits(float v) { return __float_as_uint(v) & 0x7FFFFFFFu; }

__device__ __forceinline__ void raise_peak(uint32_t* peak, uint32_t u)
{
    if (peak != nullptr && u > *peak) atomicMax(peak, u);
}

// dequant_q4k_bf16 with the merged weight times `scale` rounded to E4M3. `peak`
// (optional) collects the largest unscaled magnitude.
__global__ void dequant_q4k_e4m3(const BlockQ4K* __restrict__ w, const float* __restrict__ delta,
                                 uint8_t* __restrict__ out, long long blocks, float scale, uint32_t* peak)
{
    const long long blk = blockIdx.x;
    if (blk >= blocks) return;
    long long at;
    float lo, hi;
    q4k_pair(w, delta, blk, threadIdx.x, at, lo, hi);
    out[at] = e4m3_rn(lo * scale);
    out[at + 32] = e4m3_rn(hi * scale);
    const uint32_t a = magnitude_bits(lo), b = magnitude_bits(hi);
    raise_peak(peak, a > b ? a : b);
}

// narrow_rows_bf16 or (swiglu) narrow_swiglu_bf16 with the activation times `scale`
// rounded to E4M3, `bias` (optional) added first as narrow_rows_f16 adds it. `peak`
// (optional) collects the largest unscaled magnitude.
__global__ void narrow_rows_e4m3(const float* __restrict__ x, long long row_stride, const float* __restrict__ bias,
                                 int swiglu, uint8_t* __restrict__ out, long long cols, long long rows, float scale,
                                 uint32_t* peak)
{
    const long long n = cols * rows;
    uint32_t top = 0;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x)
    {
        const long long r = i / cols, c = i - r * cols;
        float v = x[r * row_stride + c];
        if (swiglu)
        {
            float u = x[r * row_stride + cols + c];
            if (bias != nullptr) { v = __fadd_rn(v, bias[c]); u = __fadd_rn(u, bias[cols + c]); }
            v = (v / (1.0f + expf(-v))) * u;
        }
        else if (bias != nullptr) v = __fadd_rn(v, bias[c]);
        out[i] = e4m3_rn(v * scale);
        const uint32_t a = magnitude_bits(v);
        if (a > top) top = a;
    }
    raise_peak(peak, top);
}

// An F16 weight times `scale` rounded to E4M3. `peak` (optional) collects the largest
// unscaled magnitude.
__global__ void encode_f16_e4m3(const __half* __restrict__ w, uint8_t* __restrict__ out, long long n, float scale,
                                uint32_t* peak)
{
    uint32_t top = 0;
    for (long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long) gridDim.x * blockDim.x)
    {
        const float v = __half2float(w[i]);
        out[i] = e4m3_rn(v * scale);
        const uint32_t a = magnitude_bits(v);
        if (a > top) top = a;
    }
    raise_peak(peak, top);
}

// ---- INT8 (TS_H3_INT8_MATMUL) ----

// v, already scaled so that the operand's peak is 127, rounded to int8.
__device__ __forceinline__ int8_t i8_rn(float v)
{
    return (int8_t) (int) fminf(fmaxf(rintf(v), -127.0f), 127.0f);
}

// Row n of the merged weight rounded to int8 against its largest magnitude, which
// rowpeak[n] receives as F32 bits. One block per row: its threads find the peak over
// the row's quant bytes (q4k_pair, as dequant_q4k_bf16 reads them), then round.
__global__ void dequant_q4k_rows_i8(const BlockQ4K* __restrict__ w, const float* __restrict__ delta,
                                    int8_t* __restrict__ out, uint32_t* __restrict__ rowpeak, long long rows,
                                    int blocksPerRow)
{
    __shared__ float top[256];
    const long long n = blockIdx.x;
    if (n >= rows) return;
    const long long first = n * blocksPerRow;
    const int pairs = blocksPerRow * 128;
    long long at;
    float lo, hi;
    float m = 0.0f;
    for (int i = threadIdx.x; i < pairs; i += blockDim.x)
    {
        q4k_pair(w, delta, first + (i >> 7), i & 127, at, lo, hi);
        m = fmaxf(m, fmaxf(fabsf(lo), fabsf(hi)));
    }
    top[threadIdx.x] = m;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half >>= 1)
    {
        if (threadIdx.x < half) top[threadIdx.x] = fmaxf(top[threadIdx.x], top[threadIdx.x + half]);
        __syncthreads();
    }
    const float peak = top[0];
    if (threadIdx.x == 0) rowpeak[n] = __float_as_uint(peak);
    const float inv = peak > 0.0f && peak <= 3.0e38f ? 127.0f / peak : 0.0f;
    for (int i = threadIdx.x; i < pairs; i += blockDim.x)
    {
        q4k_pair(w, delta, first + (i >> 7), i & 127, at, lo, hi);
        out[at] = i8_rn(lo * inv);
        out[at + 32] = i8_rn(hi * inv);
    }
}

// narrow_rows_bf16 or (swiglu) narrow_swiglu_bf16 with each token's activations rounded
// to int8 against the token's own peak, which tokpeak[r] receives as F32 bits. One
// block per token: its threads find the peak, then round.
__global__ void quant_rows_i8(const float* __restrict__ x, long long row_stride, int swiglu,
                              int8_t* __restrict__ out, uint32_t* __restrict__ tokpeak, long long cols, long long rows)
{
    __shared__ float top[256];
    const long long r = blockIdx.x;
    if (r >= rows) return;
    const float* __restrict__ row = x + r * row_stride;
    auto value = [&](long long c) {
        const float g = row[c];
        return swiglu ? (g / (1.0f + expf(-g))) * row[cols + c] : g;
    };
    float m = 0.0f;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, fabsf(value(c)));
    top[threadIdx.x] = m;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half >>= 1)
    {
        if (threadIdx.x < half) top[threadIdx.x] = fmaxf(top[threadIdx.x], top[threadIdx.x + half]);
        __syncthreads();
    }
    const float peak = top[0];
    if (threadIdx.x == 0) tokpeak[r] = __float_as_uint(peak);
    const float inv = peak > 0.0f && peak <= 3.0e38f ? 127.0f / peak : 0.0f;
    int8_t* __restrict__ o = out + r * cols;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) o[c] = i8_rn(value(c) * inv);
}

// The int32 products in dst, in place, to F32: element (n, t) times the two scales its
// operands were divided by. One block per token.
__global__ void scale_i32_f32(float* __restrict__ dst, long long ld, const uint32_t* __restrict__ rowpeak,
                              const uint32_t* __restrict__ tokpeak, long long N, long long T)
{
    const long long t = blockIdx.x;
    if (t >= T) return;
    float* __restrict__ col = dst + t * ld;
    const int32_t* __restrict__ acc = reinterpret_cast<const int32_t*>(col);
    const float sx = __uint_as_float(tokpeak[t]) * (1.0f / (127.0f * 127.0f));
    for (long long n = threadIdx.x; n < N; n += blockDim.x)
        col[n] = (float) acc[n] * (__uint_as_float(rowpeak[n]) * sx);
}

// One multiply as cuBLASLt takes it: dst (N x T, ld) = W (K x N) transposed times
// x (K x T), both operands E4M3 with F32 accumulation and result, or (int8) both int8
// with an int32 one.
struct Fp8Plan
{
    bool int8 = false;
    int64_t K = 0, N = 0, T = 0, ld = 0;
    bool usable = false;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulHeuristicResult_t pick = {};
};

// The power-of-two exponents an operand pair was last encoded with.
struct Fp8Exponents
{
    int w = 0, x = 0;
};

// An F16 weight's E4M3 codes on the device and the exponent they carry. `data` is null
// for a weight that does not encode.
struct Fp8Codes
{
    void* data = nullptr;
    int64_t K = 0, N = 0;
    int e = 0;
};

#endif // TSG_CUDA_FP8_GEMM

struct GemmState
{
    std::mutex mu;
    cublasHandle_t handle = nullptr;
    bool probed = false, usable = false;
    void* w = nullptr; std::size_t wBytes = 0;
    void* x = nullptr; std::size_t xBytes = 0;
    void* d = nullptr; std::size_t dBytes = 0;     // F32 LoRA delta
#if defined(TSG_CUDA_FP8_GEMM)
    cublasLtHandle_t lt = nullptr;
    bool ltProbed = false;
    std::vector<Fp8Plan> plans;
    void* work = nullptr; std::size_t workBytes = 0;    // cuBLASLt workspace
    void* peak = nullptr; std::size_t peakBytes = 0;    // the two operands' peak magnitudes
    std::unordered_map<const void*, Fp8Exponents> guesses;     // by weight
    std::unordered_map<const void*, Fp8Codes> codes;           // by F16 weight
#endif
};

GemmState& state()
{
    static GemmState s;
    return s;
}

void* ensure(void*& p, std::size_t& have, std::size_t need)
{
    if (p != nullptr && have >= need) return p;
    if (p != nullptr) cudaFree(p);
    p = nullptr;
    have = 0;
    if (cudaMalloc(&p, need) != cudaSuccess) { p = nullptr; return nullptr; }
    have = need;
    return p;
}

bool ensure_handle(GemmState& s)
{
    if (s.probed) return s.usable;
    s.probed = true;
    if (cublasCreate(&s.handle) != CUBLAS_STATUS_SUCCESS) { s.handle = nullptr; return s.usable = false; }
    // The legacy default stream; the caller orders it against ggml's stream.
    if (cublasSetStream(s.handle, nullptr) != CUBLAS_STATUS_SUCCESS) return s.usable = false;
    return s.usable = true;
}

int grid_for(long long n)
{
    const long long blocks = (n + 255) / 256;
    return (int) (blocks < 1048576 ? blocks : 1048576);
}

#if defined(TSG_CUDA_FP8_GEMM)

// TS_H3_FP8_MATMUL: 1 for the DiT's projections, vae for the video VAE's, all for both.
enum Fp8Use { FP8_DIT = 1, FP8_VAE = 2 };

int fp8_use()
{
    static const int use = [] {
        const char* v = std::getenv("TS_H3_FP8_MATMUL");
        if (v == nullptr) return 0;
        if (std::strcmp(v, "1") == 0) return (int) FP8_DIT;
        if (std::strcmp(v, "vae") == 0) return (int) FP8_VAE;
        if (std::strcmp(v, "all") == 0) return FP8_DIT | FP8_VAE;
        return 0;
    }();
    return use;
}

void fp8_forget_codes(GemmState& s)
{
    for (auto& kv : s.codes)
        if (kv.second.data != nullptr) cudaFree(kv.second.data);
    s.codes.clear();
}

void fp8_destroy(Fp8Plan& p)
{
    if (p.a != nullptr) cublasLtMatrixLayoutDestroy(p.a);
    if (p.b != nullptr) cublasLtMatrixLayoutDestroy(p.b);
    if (p.c != nullptr) cublasLtMatrixLayoutDestroy(p.c);
    if (p.op != nullptr) cublasLtMatmulDescDestroy(p.op);
    p.a = p.b = p.c = nullptr;
    p.op = nullptr;
    p.usable = false;
}

void fp8_release(GemmState& s)
{
    fp8_forget_codes(s);
    for (Fp8Plan& p : s.plans) fp8_destroy(p);
    s.plans.clear();
    if (s.lt != nullptr) cublasLtDestroy(s.lt);
    s.lt = nullptr;
    s.ltProbed = false;
    if (s.work != nullptr) { cudaFree(s.work); s.work = nullptr; s.workBytes = 0; }
    if (s.peak != nullptr) { cudaFree(s.peak); s.peak = nullptr; s.peakBytes = 0; }
}

// The plan for a shape, or nullptr where cuBLASLt has no FP8 kernel for it: a GPU
// without one, or a shape it declines. That verdict is kept with the plan, so a
// declined shape costs one lookup per call.
Fp8Plan* fp8_plan(GemmState& s, int64_t K, int64_t N, int64_t T, int64_t ld, bool int8 = false)
{
    for (Fp8Plan& p : s.plans)
        if (p.int8 == int8 && p.K == K && p.N == N && p.T == T && p.ld == ld) return p.usable ? &p : nullptr;
    if (!s.ltProbed)
    {
        s.ltProbed = true;
        if (cublasLtCreate(&s.lt) != CUBLAS_STATUS_SUCCESS) s.lt = nullptr;
    }
    if (s.plans.size() >= 64)
    {
        for (Fp8Plan& p : s.plans) fp8_destroy(p);
        s.plans.clear();
    }
    Fp8Plan p;
    p.int8 = int8;
    p.K = K; p.N = N; p.T = T; p.ld = ld;
    const cudaDataType_t operand = int8 ? CUDA_R_8I : CUDA_R_8F_E4M3;
    const cudaDataType_t result = int8 ? CUDA_R_32I : CUDA_R_32F;
    if (s.lt != nullptr)
    {
        // FP8 kernels take the first operand transposed and the second as it is.
        const cublasOperation_t transposed = CUBLAS_OP_T;
        const uint64_t workLimit = 64ull << 20;
        cublasLtMatmulPreference_t pref = nullptr;
        int found = 0;
        const cublasStatus_t ok = CUBLAS_STATUS_SUCCESS;
        p.usable = cublasLtMatmulDescCreate(&p.op, int8 ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F, result) == ok
            && cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &transposed, sizeof(transposed)) == ok
            && cublasLtMatrixLayoutCreate(&p.a, operand, (uint64_t) K, (uint64_t) N, K) == ok
            && cublasLtMatrixLayoutCreate(&p.b, operand, (uint64_t) K, (uint64_t) T, K) == ok
            && cublasLtMatrixLayoutCreate(&p.c, result, (uint64_t) N, (uint64_t) T, ld) == ok
            && cublasLtMatmulPreferenceCreate(&pref) == ok
            && cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workLimit,
                                                    sizeof(workLimit)) == ok
            && cublasLtMatmulAlgoGetHeuristic(s.lt, p.op, p.a, p.b, p.c, p.c, pref, 1, &p.pick, &found) == ok
            && found > 0 && p.pick.state == ok;
        if (pref != nullptr) cublasLtMatmulPreferenceDestroy(pref);
        if (!p.usable) fp8_destroy(p);
    }
    s.plans.push_back(p);
    return s.plans.back().usable ? &s.plans.back() : nullptr;
}

// The exponent e that puts an operand's peak magnitude (F32 bits) at peak * 2^e in
// (224, 448], the top of E4M3's range, so that nothing saturates and as little as
// possible falls into the subnormals. False for an operand that cannot be encoded:
// one holding Inf or NaN, or one so large or small that the product's scale would
// leave F32. An all-zero operand encodes the same under any exponent and keeps `e`.
bool fp8_exponent(uint32_t peakBits, int& e)
{
    if (peakBits >= 0x7F800000u) return false;
    if (peakBits == 0) return true;
    float peak;
    std::memcpy(&peak, &peakBits, sizeof(peak));
    int ex = 0;
    const float m = std::frexp(peak, &ex);      // peak = m 2^ex, m in [0.5, 1); 448 = 0.875 * 2^9
    e = 9 - ex - (m > 0.875f ? 1 : 0);
    return e >= -60 && e <= 60;
}

// The product on FP8 tensor cores: dst = 2^-(we + xe) * E4M3(2^we W)^T E4M3(2^xe x),
// with the weight merged with `delta` and the activation taken through SwiGLU as on
// the BF16 route. Returns true when dst holds it. False leaves the BF16 route to
// compute it: no FP8 kernel for this shape on the device, or an operand that does not
// encode.
//
// One power-of-two scale per operand is all E4M3 needs. It is a floating-point
// format, so its relative precision is the same in every binade and a finer scale
// (per token, per row, or equalized across channels) measured no better; and a power
// of two moves no mantissa bit, so the scale itself adds no error.
//
// The exponents are a function of the operands alone (fp8_exponent of each peak), so
// a request computes the same result whatever ran before it. Finding a peak takes a
// pass over the operand, which is what encoding it costs too, so the two are one
// kernel: it encodes with the exponent this weight's operands had last time while it
// collects the peak, and only an operand whose peak asks for another exponent is
// encoded again. The merged weight's never changes and the activations' seldom does.
template <typename Lap>
bool fp8_gemm(GemmState& s, const void* w_q4k, const float* delta, int64_t K, int64_t N, const float* x,
              int64_t x_row_stride, int x_swiglu, int64_t T, float* dst, int64_t dst_ld, Lap&& lap)
{
    Fp8Plan* plan = fp8_plan(s, K, N, T, dst_ld);
    if (plan == nullptr) return false;
    if (ensure(s.peak, s.peakBytes, 2 * sizeof(uint32_t)) == nullptr) return false;
    if (plan->pick.workspaceSize > 0 && ensure(s.work, s.workBytes, plan->pick.workspaceSize) == nullptr)
        return false;
    const auto* wq = static_cast<const BlockQ4K*>(w_q4k);
    auto* wc = static_cast<uint8_t*>(s.w);       // sized for BF16: twice what the codes need
    auto* xc = static_cast<uint8_t*>(s.x);
    auto* peak = static_cast<uint32_t*>(s.peak);
    const long long blocks = (long long) N * (K / 256);
    const int xgrid = grid_for((long long) T * K);

    if (s.guesses.size() > 8192) s.guesses.clear();
    Fp8Exponents& guess = s.guesses[w_q4k];
    if (cudaMemsetAsync(peak, 0, 2 * sizeof(uint32_t), nullptr) != cudaSuccess) return false;
    dequant_q4k_e4m3<<<(unsigned int) blocks, 128>>>(wq, delta, wc, blocks, std::ldexp(1.0f, guess.w), peak);
    lap("dequant");
    narrow_rows_e4m3<<<xgrid, 256>>>(x, x_row_stride, nullptr, x_swiglu, xc, K, T, std::ldexp(1.0f, guess.x),
                                     peak + 1);
    lap("narrow");
    uint32_t seen[2] = { 0, 0 };
    if (cudaGetLastError() != cudaSuccess
        || cudaMemcpy(seen, peak, sizeof(seen), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    int we = guess.w, xe = guess.x;
    if (!fp8_exponent(seen[0], we) || !fp8_exponent(seen[1], xe)) return false;
    if (we != guess.w)
    {
        dequant_q4k_e4m3<<<(unsigned int) blocks, 128>>>(wq, delta, wc, blocks, std::ldexp(1.0f, we), nullptr);
        lap("dequant again");
    }
    if (xe != guess.x)
    {
        narrow_rows_e4m3<<<xgrid, 256>>>(x, x_row_stride, nullptr, x_swiglu, xc, K, T, std::ldexp(1.0f, xe), nullptr);
        lap("narrow again");
    }
    if (cudaGetLastError() != cudaSuccess) return false;
    guess.w = we;
    guess.x = xe;

    const float alpha = std::ldexp(1.0f, -(we + xe)), beta = 0.0f;
    if (cublasLtMatmul(s.lt, plan->op, &alpha, wc, plan->a, xc, plan->b, &beta, dst, plan->c, dst, plan->c,
                       &plan->pick.algo, s.work, plan->pick.workspaceSize, nullptr) != CUBLAS_STATUS_SUCCESS)
    {
        plan->usable = false;       // BF16 for this shape from here on
        return false;
    }
    const bool ok = cudaStreamSynchronize(nullptr) == cudaSuccess;
    lap("multiply");
    return ok;
}

// TS_H3_INT8_MATMUL=1: the DiT's projections on int8 tensor cores.
bool int8_use()
{
    static const bool use = [] {
        const char* v = std::getenv("TS_H3_INT8_MATMUL");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return use;
}

// The product on int8 tensor cores: dst[n, t] = sw[n] sx[t] * sum_k Wq[n, k] xq[t, k],
// where Wq is row n of the merged weight divided by sw[n] = peak / 127 and rounded to
// int8, and xq token t's activations divided by sx[t] likewise. The sum is exact in
// int32 (K * 127^2 stays below 2^31 up to K = 133,000), so the only error is the two
// roundings: each operand keeps 7 bits against its own row's peak, where BF16 keeps
// 8 significant bits per element. Returns true when dst holds the product; false
// leaves the BF16 route to compute it: no int8 kernel for this shape on the device, or
// scratch that could not be had.
//
// A scale per weight row and per token is what a GEMM lets through, and it is what
// this model needs: its tokens' magnitudes differ by orders, and a weight's rows by
// less but enough. Rounding the weight per call (each row read twice from its Q4_K
// blocks, the LoRA update merged in) keeps nothing resident, as the BF16 route keeps
// nothing.
template <typename Lap>
bool int8_gemm(GemmState& s, const void* w_q4k, const float* delta, int64_t K, int64_t N, const float* x,
               int64_t x_row_stride, int x_swiglu, int64_t T, float* dst, int64_t dst_ld, Lap&& lap)
{
    if (K > 131072 || T > 0x7fffffffLL || N > 0x7fffffffLL) return false;
    Fp8Plan* plan = fp8_plan(s, K, N, T, dst_ld, true);
    if (plan == nullptr)
    {
        static bool said = false;
        if (!said)
        {
            said = true;
            std::fprintf(stderr, "[h3-int8] cuBLASLt has no int8 kernel for %lld x %lld x %lld here; such products "
                                 "run on BF16.\n", (long long) K, (long long) N, (long long) T);
        }
        return false;
    }
    if (ensure(s.peak, s.peakBytes, (std::size_t) (N + T) * sizeof(uint32_t)) == nullptr) return false;
    if (plan->pick.workspaceSize > 0 && ensure(s.work, s.workBytes, plan->pick.workspaceSize) == nullptr)
        return false;
    const auto* wq = static_cast<const BlockQ4K*>(w_q4k);
    auto* wc = static_cast<int8_t*>(s.w);        // sized for BF16: twice what int8 needs
    auto* xc = static_cast<int8_t*>(s.x);
    auto* rowpeak = static_cast<uint32_t*>(s.peak);
    uint32_t* tokpeak = rowpeak + N;
    const int perRow = (int) (K / 256);

    dequant_q4k_rows_i8<<<(unsigned int) N, 256>>>(wq, delta, wc, rowpeak, N, perRow);
    lap("dequant");
    quant_rows_i8<<<(unsigned int) T, 256>>>(x, x_row_stride, x_swiglu, xc, tokpeak, K, T);
    if (cudaGetLastError() != cudaSuccess) return false;
    lap("narrow");

    const int32_t alpha = 1, beta = 0;
    if (cublasLtMatmul(s.lt, plan->op, &alpha, wc, plan->a, xc, plan->b, &beta, dst, plan->c, dst, plan->c,
                       &plan->pick.algo, s.work, plan->pick.workspaceSize, nullptr) != CUBLAS_STATUS_SUCCESS)
    {
        plan->usable = false;       // BF16 for this shape from here on
        return false;
    }
    lap("multiply");
    scale_i32_f32<<<(unsigned int) T, 256>>>(dst, dst_ld, rowpeak, tokpeak, N, T);
    if (cudaGetLastError() != cudaSuccess) return false;
    const bool ok = cudaStreamSynchronize(nullptr) == cudaSuccess;
    lap("scale");
    return ok;
}

// The E4M3 codes of an F16 weight and their exponent, made on the weight's first use
// and kept by its address until the weights are forgotten or the scratch is released.
// A tiled decoder multiplies by each weight once per tile, and encoding it every time
// would give back a third of what the FP8 kernel saves. Null for a weight that does
// not encode, or with no memory for its codes.
const uint8_t* fp8_codes(GemmState& s, const void* w_f16, int64_t K, int64_t N, int& e)
{
    auto it = s.codes.find(w_f16);
    if (it != s.codes.end())
    {
        if (it->second.K == K && it->second.N == N)
        {
            e = it->second.e;
            return static_cast<const uint8_t*>(it->second.data);
        }
        if (it->second.data != nullptr) cudaFree(it->second.data);
        s.codes.erase(it);
    }
    Fp8Codes made;
    made.K = K;
    made.N = N;
    const long long n = (long long) K * N;
    const auto* w = static_cast<const __half*>(w_f16);
    auto* peak = static_cast<uint32_t*>(s.peak);
    int& guess = s.guesses[w_f16].w;
    int we = guess;
    if (cudaMalloc(&made.data, (std::size_t) n) != cudaSuccess)
    {
        (void) cudaGetLastError();
        made.data = nullptr;
    }
    else
    {
        auto* out = static_cast<uint8_t*>(made.data);
        uint32_t seen = 0;
        bool ok = cudaMemsetAsync(peak, 0, sizeof(uint32_t), nullptr) == cudaSuccess;
        if (ok)
        {
            encode_f16_e4m3<<<grid_for(n), 256>>>(w, out, n, std::ldexp(1.0f, guess), peak);
            ok = cudaGetLastError() == cudaSuccess
                && cudaMemcpy(&seen, peak, sizeof(seen), cudaMemcpyDeviceToHost) == cudaSuccess
                && fp8_exponent(seen, we);
        }
        if (ok && we != guess)
        {
            encode_f16_e4m3<<<grid_for(n), 256>>>(w, out, n, std::ldexp(1.0f, we), nullptr);
            ok = cudaGetLastError() == cudaSuccess;
        }
        if (ok) guess = we;
        else
        {
            cudaFree(made.data);
            made.data = nullptr;
        }
    }
    made.e = we;
    s.codes[w_f16] = made;
    e = we;
    return static_cast<const uint8_t*>(made.data);
}

// tsg_cuda_f16_gemm's product on FP8 tensor cores: dst = 2^-(we + xe) *
// E4M3(2^we W)^T E4M3(2^xe act(x)), with the weight's codes kept between calls
// (fp8_codes). Returns true when dst holds it; false leaves the F16 route to compute it.
//
// As in fp8_gemm, the activation's exponent follows from its peak alone, which the
// pass that encodes it finds. These products are small enough that waiting for the
// peak before the multiply costs a round trip worth having back, so the multiply is
// issued on the exponent this weight's activations had last time, and both are redone
// only when the peak then asks for another.
bool fp8_f16_gemm(GemmState& s, const void* w_f16, int64_t K, int64_t N, const float* x, int64_t x_row_stride,
                  const float* x_bias, int swiglu, int64_t T, float* dst, int64_t dst_ld)
{
    // Too small to gain from it, and the caller keeps only larger weights at an address
    // of their own.
    if ((long long) K * N < 65536) return false;
    Fp8Plan* plan = fp8_plan(s, K, N, T, dst_ld);
    if (plan == nullptr) return false;
    if (ensure(s.peak, s.peakBytes, 2 * sizeof(uint32_t)) == nullptr) return false;
    if (plan->pick.workspaceSize > 0 && ensure(s.work, s.workBytes, plan->pick.workspaceSize) == nullptr)
        return false;
    if (s.guesses.size() > 8192) s.guesses.clear();
    int we = 0;
    const uint8_t* wc = fp8_codes(s, w_f16, K, N, we);
    if (wc == nullptr) return false;
    auto* xc = static_cast<uint8_t*>(s.x);       // sized for F16: twice what the codes need
    auto* peak = static_cast<uint32_t*>(s.peak) + 1;
    const int xgrid = grid_for((long long) T * K);
    int& guess = s.guesses[w_f16].x;
    const float beta = 0.0f;

    for (int pass = 0; pass < 2; pass++)
    {
        if (pass == 0 && cudaMemsetAsync(peak, 0, sizeof(uint32_t), nullptr) != cudaSuccess) return false;
        narrow_rows_e4m3<<<xgrid, 256>>>(x, x_row_stride, x_bias, swiglu, xc, K, T, std::ldexp(1.0f, guess),
                                         pass == 0 ? peak : nullptr);
        if (cudaGetLastError() != cudaSuccess) return false;
        const float alpha = std::ldexp(1.0f, -(we + guess));
        if (cublasLtMatmul(s.lt, plan->op, &alpha, wc, plan->a, xc, plan->b, &beta, dst, plan->c, dst, plan->c,
                           &plan->pick.algo, s.work, plan->pick.workspaceSize, nullptr) != CUBLAS_STATUS_SUCCESS)
        {
            plan->usable = false;       // F16 for this shape from here on
            return false;
        }
        if (pass == 1) return cudaStreamSynchronize(nullptr) == cudaSuccess;
        uint32_t seen = 0;
        int xe = guess;
        // The copy waits for the stream, so for the multiply too.
        if (cudaMemcpy(&seen, peak, sizeof(seen), cudaMemcpyDeviceToHost) != cudaSuccess || !fp8_exponent(seen, xe))
            return false;
        if (xe == guess) return true;
        guess = xe;
    }
    return false;
}

#endif // TSG_CUDA_FP8_GEMM

} // namespace

extern "C" {

// BF16 tensor-core GEMMs need sm_80 or newer. The graph decides at build time to
// drop the q8_1 guard for matmuls this route takes (and to merge their LoRA into a
// node only this route runs), so it must not be offered where cuBLAS would decline.
bool tsg_cuda_q4k_gemm_supported(void)
{
    static const bool ok = [] {
        int device = 0, major = 0;
        if (cudaGetDevice(&device) != cudaSuccess) return false;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) return false;
        return major >= 8;
    }();
    return ok;
}

// dst[n, t] = sum_k W[n, k] * x[t, k] for a contiguous Q4_K W of N rows by K
// (K a multiple of 256), F32 x with `x_row_stride` elements between tokens, and a
// contiguous F32 dst of `dst_ld` elements per token. Returns false (writing
// nothing) when scratch or cuBLAS is unavailable. Synchronizes the default stream.
bool tsg_cuda_q4k_lora_gemm_bf16(const void* w_q4k, int64_t K, int64_t N, const float* x, int64_t x_row_stride,
                                 int64_t T, float* dst, int64_t dst_ld, int nparts, const void* const* downs,
                                 const void* const* ups, const int64_t* ranks, const int64_t* outs,
                                 const int64_t* offsets, int factor_f32, int x_swiglu);

bool tsg_cuda_q4k_gemm_bf16(const void* w_q4k, int64_t K, int64_t N, const float* x, int64_t x_row_stride,
                            int64_t T, float* dst, int64_t dst_ld)
{
    return tsg_cuda_q4k_lora_gemm_bf16(w_q4k, K, N, x, x_row_stride, T, dst, dst_ld, 0, nullptr, nullptr,
                                       nullptr, nullptr, nullptr, 0, 0);
}

// As tsg_cuda_q4k_gemm_bf16, with LoRA updates merged into the weight first:
// rows [offsets[i], offsets[i] + outs[i]) of W gain up_i * down_i, where down_i is
// [ranks[i], K] and up_i [outs[i], ranks[i]] row-major (ggml [K, rank] and
// [rank, out]), F16 (or F32 with factor_f32). This is the y = W x + up (down x) of
// the unmerged graph with the sum moved onto the weight: the update is formed in
// F32 and added to the dequantized row before its single BF16 rounding. With
// x_swiglu, each x row holds fc1's [gate | value] (2K floats) and the GEMM reads
// SwiGLU of it (narrow_swiglu_bf16).
bool tsg_cuda_q4k_lora_gemm_bf16(const void* w_q4k, int64_t K, int64_t N, const float* x, int64_t x_row_stride,
                                 int64_t T, float* dst, int64_t dst_ld, int nparts, const void* const* downs,
                                 const void* const* ups, const int64_t* ranks, const int64_t* outs,
                                 const int64_t* offsets, int factor_f32, int x_swiglu)
{
    if (K % 256 != 0 || K <= 0 || N <= 0 || T <= 0) return false;
    GemmState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_handle(s)) return false;
    const std::size_t wNeed = (std::size_t) N * K * sizeof(__nv_bfloat16);
    const std::size_t xNeed = (std::size_t) T * K * sizeof(__nv_bfloat16);
    if (ensure(s.w, s.wBytes, wNeed) == nullptr || ensure(s.x, s.xBytes, xNeed) == nullptr) return false;
    auto* wb = static_cast<__nv_bfloat16*>(s.w);
    auto* xb = static_cast<__nv_bfloat16*>(s.x);

    tsg_stage::Clock clock;
    auto lap = [&](const char* what) {
        if (!tsg_stage::enabled()) return;
        char stage[64];
        std::snprintf(stage, sizeof(stage), "gemm %lldx%lld%s %s", (long long) K, (long long) N,
                      x_swiglu ? " swiglu" : "", what);
        clock.lap(stage);
    };

    float* delta = nullptr;
    if (nparts > 0)
    {
        if (ensure(s.d, s.dBytes, (std::size_t) N * K * sizeof(float)) == nullptr) return false;
        delta = static_cast<float*>(s.d);
        int64_t covered = 0;
        for (int i = 0; i < nparts; i++) covered += outs[i];
        if (covered != N && cudaMemsetAsync(delta, 0, (std::size_t) N * K * sizeof(float), nullptr) != cudaSuccess)
            return false;
        const float one = 1.0f, zero = 0.0f;
        const cudaDataType_t ft = factor_f32 ? CUDA_R_32F : CUDA_R_16F;
        for (int i = 0; i < nparts; i++)
        {
            // Column-major: delta' (K x out, ld K) = down' (K x rank, ld K) * up' (rank x out, ld rank).
            if (cublasGemmEx(s.handle, CUBLAS_OP_N, CUBLAS_OP_N, (int) K, (int) outs[i], (int) ranks[i], &one,
                             downs[i], ft, (int) K, ups[i], ft, (int) ranks[i], &zero,
                             delta + offsets[i] * K, CUDA_R_32F, (int) K,
                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS)
                return false;
        }
        lap("lora");
    }

#if defined(TSG_CUDA_FP8_GEMM)
    if (int8_use() && int8_gemm(s, w_q4k, delta, K, N, x, x_row_stride, x_swiglu, T, dst, dst_ld, lap))
        return true;
    if ((fp8_use() & FP8_DIT) != 0
        && fp8_gemm(s, w_q4k, delta, K, N, x, x_row_stride, x_swiglu, T, dst, dst_ld, lap))
        return true;
#endif

    const long long blocks = (long long) N * (K / 256);
    dequant_q4k_bf16<<<(unsigned int) blocks, 128>>>(static_cast<const BlockQ4K*>(w_q4k), delta, wb, blocks);
    lap("dequant");
    if (x_swiglu)
        narrow_swiglu_bf16<<<grid_for((long long) T * K), 256>>>(x, x_row_stride, xb, K, T);
    else
        narrow_rows_bf16<<<grid_for((long long) T * K), 256>>>(x, x_row_stride, xb, K, T);
    if (cudaGetLastError() != cudaSuccess) return false;
    lap("narrow");

    const float alpha = 1.0f, beta = 0.0f;
    // Column-major: dst (N x T, ld dst_ld) = W^T' (K x N, ld K)^T * x' (K x T, ld K).
    const cublasStatus_t st = cublasGemmEx(s.handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                           (int) N, (int) T, (int) K, &alpha,
                                           wb, CUDA_R_16BF, (int) K,
                                           xb, CUDA_R_16BF, (int) K, &beta,
                                           dst, CUDA_R_32F, (int) dst_ld,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) return false;
    const bool ok = cudaStreamSynchronize(nullptr) == cudaSuccess;
    lap("multiply");
    return ok;
}

// tsg_cuda_q4k_gemm_bf16 for a contiguous Q6_K W (the tensors a Q4_K_M file keeps at
// six bits): dequantized to BF16 per call, one cuBLAS GEMM with F32 accumulation.
bool tsg_cuda_q6k_gemm_bf16(const void* w_q6k, int64_t K, int64_t N, const float* x, int64_t x_row_stride,
                            int64_t T, float* dst, int64_t dst_ld)
{
    if (K % 256 != 0 || K <= 0 || N <= 0 || T <= 0) return false;
    GemmState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_handle(s)) return false;
    const std::size_t wNeed = (std::size_t) N * K * sizeof(__nv_bfloat16);
    const std::size_t xNeed = (std::size_t) T * K * sizeof(__nv_bfloat16);
    if (ensure(s.w, s.wBytes, wNeed) == nullptr || ensure(s.x, s.xBytes, xNeed) == nullptr) return false;
    auto* wb = static_cast<__nv_bfloat16*>(s.w);
    auto* xb = static_cast<__nv_bfloat16*>(s.x);
    const long long blocks = (long long) N * (K / 256);
    dequant_q6k_bf16<<<(unsigned int) blocks, 64>>>(static_cast<const uint8_t*>(w_q6k), wb, blocks);
    narrow_rows_bf16<<<grid_for((long long) T * K), 256>>>(x, x_row_stride, xb, K, T);
    if (cudaGetLastError() != cudaSuccess) return false;
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t st = cublasGemmEx(s.handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                           (int) N, (int) T, (int) K, &alpha,
                                           wb, CUDA_R_16BF, (int) K,
                                           xb, CUDA_R_16BF, (int) K, &beta,
                                           dst, CUDA_R_32F, (int) dst_ld,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) return false;
    return cudaStreamSynchronize(nullptr) == cudaSuccess;
}

// dst[n, t] = sum_k W[n, k] * act(x)[t, k] for a contiguous F16 W of N rows by K: the
// video VAE's projections. act is x (plus x_bias when given) or, with swiglu, SwiGLU
// of x's [gate | value] rows (2K floats each, x_bias then 2K long), narrowed to F16;
// the GEMM accumulates in F32 into an F32 dst. Synchronizes the default stream.
bool tsg_cuda_f16_gemm(const void* w_f16, int64_t K, int64_t N, const float* x, int64_t x_row_stride,
                       const float* x_bias, int swiglu, int64_t T, float* dst, int64_t dst_ld)
{
    if (K <= 0 || N <= 0 || T <= 0) return false;
    GemmState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!ensure_handle(s)) return false;
    if (ensure(s.x, s.xBytes, (std::size_t) T * K * sizeof(__half)) == nullptr) return false;
#if defined(TSG_CUDA_FP8_GEMM)
    if ((fp8_use() & FP8_VAE) != 0
        && fp8_f16_gemm(s, w_f16, K, N, x, x_row_stride, x_bias, swiglu, T, dst, dst_ld))
        return true;
#endif
    auto* xh = static_cast<__half*>(s.x);
    narrow_rows_f16<<<grid_for((long long) T * K), 256>>>(x, x_row_stride, x_bias, swiglu, xh, K, T);
    if (cudaGetLastError() != cudaSuccess) return false;
    const float alpha = 1.0f, beta = 0.0f;
    const cublasStatus_t st = cublasGemmEx(s.handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                           (int) N, (int) T, (int) K, &alpha,
                                           w_f16, CUDA_R_16F, (int) K,
                                           xh, CUDA_R_16F, (int) K, &beta,
                                           dst, CUDA_R_32F, (int) dst_ld,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) return false;
    return cudaStreamSynchronize(nullptr) == cudaSuccess;
}

// A weight tsg_cuda_f16_gemm is given has a new device copy, so another weight may now
// sit where one used to: drop what is kept by their addresses.
void tsg_cuda_f16_gemm_forget_weights(void)
{
#if defined(TSG_CUDA_FP8_GEMM)
    GemmState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    fp8_forget_codes(s);
#endif
}

void tsg_cuda_q4k_gemm_release(void)
{
    tsg_stage::dump("release");
    GemmState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (s.w != nullptr) { cudaFree(s.w); s.w = nullptr; s.wBytes = 0; }
    if (s.x != nullptr) { cudaFree(s.x); s.x = nullptr; s.xBytes = 0; }
    if (s.d != nullptr) { cudaFree(s.d); s.d = nullptr; s.dBytes = 0; }
#if defined(TSG_CUDA_FP8_GEMM)
    fp8_release(s);     // the exponent guesses are host memory and stay
#endif
    if (s.handle != nullptr) cublasDestroy(s.handle);
    s.handle = nullptr;
    s.probed = false;
    s.usable = false;
}

} // extern "C"

#endif // TSG_GGML_USE_CUDA
