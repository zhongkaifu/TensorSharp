// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// MiniMax-H3's DiT attention on FlashAttention-2's forward kernel
// (https://github.com/Dao-AILab/flash-attention, BSD-3-Clause), the kernel PyTorch's
// scaled_dot_product_attention runs. It takes the place of cuDNN's SDPA engine inside
// tsg_cudnn_h3_attention, between the same prep and widen passes and on the same F16
// operands: at 56 heads x 128 over 39,875 tokens on an A100 it ran in 245 ms where
// cuDNN's engine took 278 ms.
//
// Built only with -DTENSORSHARP_FLASH_ATTENTION=ON, which fetches the pinned
// flash-attention and CUTLASS sources at configure time and compiles the one kernel
// this model uses (head size 128, F16, no mask). The kernel's launch template includes
// two PyTorch headers; flash_attn_shim/ stands in for them. TS_H3_FLASH_ATTENTION=0
// leaves the attention on cuDNN, which is also where it stays on a GPU newer than sm_8x
// unless TS_H3_FLASH_ATTENTION=1 (see usable).
#if defined(TSG_GGML_USE_CUDA) && defined(TSG_HAVE_FLASH_ATTN)

#include "flash.h"

#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

// Set by the launch checks in flash_attn_shim/c10/cuda/CUDAException.h.
bool tsg_flash_attn_failed = false;

namespace {

struct FlashState
{
    std::mutex mu;
    bool probed = false, usable = false;
    void* lse = nullptr;            // the kernel's log-sum-exp per (head, query), unused here
    std::size_t lseBytes = 0;
};

FlashState& state()
{
    static FlashState s;
    return s;
}

bool usable(FlashState& s)
{
    if (s.probed) return s.usable;
    s.probed = true;
    const char* e = std::getenv("TS_H3_FLASH_ATTENTION");
    if (e != nullptr && e[0] == '0') return s.usable = false;
    // The kernel is tensor-core MMA code for sm_80 and newer; a build for an older
    // architecture holds only a stub.
    int device = 0, major = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return s.usable = false;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) return s.usable = false;
    // It is Ampere code, and it wins only there: at 39,875 tokens it takes 251 ms a block
    // on an A100 where cuDNN's SDPA takes 278, and 165 ms on an H100 where cuDNN's takes
    // 114 (cuDNN has a Hopper engine). So it is the default on sm_8x alone;
    // TS_H3_FLASH_ATTENTION=1 asks for it on a newer GPU too.
    const bool asked = e != nullptr && e[0] == '1';
    return s.usable = asked ? major >= 8 : major == 8;
}

} // namespace

extern "C" {

// O = softmax(scale * Q K^T) V for one unmasked sequence, all F16:
//   q, k, v: element (d, s, h) at d + s*D + h*D*S      (head-major, contiguous)
//   o:       element (d, h, s) at d + h*D + s*D*H      (ggml [D, H, S])
// as tsg_cudnn_h3_attention hands them to cuDNN. Returns false, with o undefined, when
// this build, device or shape has no kernel or the launch failed; the caller then runs
// cuDNN. Does not synchronize: the kernel is queued on the default stream.
bool tsg_flash_attn_h3(const void* q, const void* k, const void* v, void* o, int64_t heads, int64_t seq,
                       int64_t head_dim, float scale)
{
    FlashState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (!usable(s)) return false;
    if (head_dim != 128 || heads <= 0 || heads > 65535 || seq <= 0 || seq > 0x7fffff00LL) return false;
    const std::size_t lseNeed = (std::size_t) heads * seq * sizeof(float);
    if (s.lse == nullptr || s.lseBytes < lseNeed)
    {
        if (s.lse != nullptr) cudaFree(s.lse);
        s.lse = nullptr;
        s.lseBytes = 0;
        if (cudaMalloc(&s.lse, lseNeed) != cudaSuccess)
        {
            (void) cudaGetLastError();
            s.lse = nullptr;
            return false;
        }
        s.lseBytes = lseNeed;
    }

    using namespace FLASH_NAMESPACE;
    Flash_fwd_params p;
    std::memset(&p, 0, sizeof(p));
    p.q_ptr = const_cast<void*>(q);
    p.k_ptr = const_cast<void*>(k);
    p.v_ptr = const_cast<void*>(v);
    p.o_ptr = o;
    p.softmax_lse_ptr = s.lse;
    p.q_row_stride = p.k_row_stride = p.v_row_stride = head_dim;
    p.q_head_stride = p.k_head_stride = p.v_head_stride = seq * head_dim;
    p.q_batch_stride = p.k_batch_stride = p.v_batch_stride = heads * seq * head_dim;
    p.o_row_stride = heads * head_dim;
    p.o_head_stride = head_dim;
    p.o_batch_stride = heads * seq * head_dim;
    p.b = 1;
    p.h = p.h_k = (int) heads;
    p.h_h_k_ratio = 1;
    p.seqlen_q = p.seqlen_k = (int) seq;
    p.seqlen_q_rounded = p.seqlen_k_rounded = (int) ((seq + 127) / 128 * 128);
    p.d = p.d_rounded = (int) head_dim;
    p.scale_softmax = scale;
    p.scale_softmax_log2 = scale * 1.4426950408889634f;
    p.p_dropout = 1.0f;             // the probability of keeping: no dropout
    p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.0f;
    p.scale_softmax_rp_dropout = scale;
    p.window_size_left = p.window_size_right = -1;
    p.is_seqlens_k_cumulative = true;

    tsg_flash_attn_failed = false;
    run_mha_fwd_<cutlass::half_t, 128, false>(p, nullptr);
    if (tsg_flash_attn_failed)
    {
        (void) cudaGetLastError();
        s.usable = false;           // cuDNN from here on
        return false;
    }
    static bool said = false;
    if (!said)
    {
        said = true;
        if (const char* e = std::getenv("TS_H3_CUDNN_ATTENTION_VERBOSE"); e != nullptr && e[0] == '1')
            std::fprintf(stderr, "[flash-attn] H=%lld S=%lld D=%lld runs on FlashAttention-2\n", (long long) heads,
                         (long long) seq, (long long) head_dim);
    }
    return true;
}

// Frees the scratch; the next call re-creates it. Called with tsg_cudnn_sdpa_release.
void tsg_flash_attn_release(void)
{
    FlashState& s = state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (s.lse != nullptr) { cudaFree(s.lse); s.lse = nullptr; s.lseBytes = 0; }
    s.probed = false;
    s.usable = false;
}

} // extern "C"

#endif // TSG_GGML_USE_CUDA && TSG_HAVE_FLASH_ATTN
