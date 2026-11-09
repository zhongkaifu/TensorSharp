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
#if defined(TSG_GGML_USE_CUDA)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

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

// One thread per quant byte: byte b of a block holds weight l of 64-chunk j in its
// low nibble and weight 32 + l in its high nibble (b = 32 j + l). With `delta`
// (an F32 [N, K] LoRA update, up * down), the merged weight is rounded once.
__global__ void dequant_q4k_bf16(const BlockQ4K* __restrict__ w, const float* __restrict__ delta,
                                 __nv_bfloat16* __restrict__ out, long long blocks)
{
    const long long blk = blockIdx.x;
    if (blk >= blocks) return;
    const int b = threadIdx.x;          // 0..127
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
    float lo = d1 * (v & 0xF) - mm1;
    float hi = d2 * (v >> 4) - mm2;
    if (delta != nullptr) { lo += delta[base + l]; hi += delta[base + 32 + l]; }
    out[base + l] = __float2bfloat16_rn(lo);
    out[base + 32 + l] = __float2bfloat16_rn(hi);
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

struct GemmState
{
    std::mutex mu;
    cublasHandle_t handle = nullptr;
    bool probed = false, usable = false;
    void* w = nullptr; std::size_t wBytes = 0;
    void* x = nullptr; std::size_t xBytes = 0;
    void* d = nullptr; std::size_t dBytes = 0;     // F32 LoRA delta
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
                                 const int64_t* offsets, int factor_f32);

bool tsg_cuda_q4k_gemm_bf16(const void* w_q4k, int64_t K, int64_t N, const float* x, int64_t x_row_stride,
                            int64_t T, float* dst, int64_t dst_ld)
{
    return tsg_cuda_q4k_lora_gemm_bf16(w_q4k, K, N, x, x_row_stride, T, dst, dst_ld, 0, nullptr, nullptr,
                                       nullptr, nullptr, nullptr, 0);
}

// As tsg_cuda_q4k_gemm_bf16, with LoRA updates merged into the weight first:
// rows [offsets[i], offsets[i] + outs[i]) of W gain up_i * down_i, where down_i is
// [ranks[i], K] and up_i [outs[i], ranks[i]] row-major (ggml [K, rank] and
// [rank, out]), F16 (or F32 with factor_f32). This is the y = W x + up (down x) of
// the unmerged graph with the sum moved onto the weight: the update is formed in
// F32 and added to the dequantized row before its single BF16 rounding.
bool tsg_cuda_q4k_lora_gemm_bf16(const void* w_q4k, int64_t K, int64_t N, const float* x, int64_t x_row_stride,
                                 int64_t T, float* dst, int64_t dst_ld, int nparts, const void* const* downs,
                                 const void* const* ups, const int64_t* ranks, const int64_t* outs,
                                 const int64_t* offsets, int factor_f32)
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
    }

    const long long blocks = (long long) N * (K / 256);
    dequant_q4k_bf16<<<(unsigned int) blocks, 128>>>(static_cast<const BlockQ4K*>(w_q4k), delta, wb, blocks);
    narrow_rows_bf16<<<grid_for((long long) T * K), 256>>>(x, x_row_stride, xb, K, T);
    if (cudaGetLastError() != cudaSuccess) return false;

    const float alpha = 1.0f, beta = 0.0f;
    // Column-major: dst (N x T, ld dst_ld) = W^T' (K x N, ld K)^T * x' (K x T, ld K).
    const cublasStatus_t st = cublasGemmEx(s.handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                           (int) N, (int) T, (int) K, &alpha,
                                           wb, CUDA_R_16BF, (int) K,
                                           xb, CUDA_R_16BF, (int) K, &beta,
                                           dst, CUDA_R_32F, (int) dst_ld,
                                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) return false;
    return cudaStreamSynchronize(nullptr) == cudaSuccess;
}

void tsg_cuda_q4k_gemm_release(void)
{
    GemmState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (s.w != nullptr) { cudaFree(s.w); s.w = nullptr; s.wBytes = 0; }
    if (s.x != nullptr) { cudaFree(s.x); s.x = nullptr; s.xBytes = 0; }
    if (s.d != nullptr) { cudaFree(s.d); s.d = nullptr; s.dBytes = 0; }
    if (s.handle != nullptr) cublasDestroy(s.handle);
    s.handle = nullptr;
    s.probed = false;
    s.usable = false;
}

} // extern "C"

#endif // TSG_GGML_USE_CUDA
