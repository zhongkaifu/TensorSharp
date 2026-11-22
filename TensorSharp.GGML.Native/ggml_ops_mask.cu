// On-device causal/windowed attention-mask generation for the Gemma 4 fused
// verify (prefill) kernel. Replaces the host-side per-chunk mask fill
// (O(N * kvLen) CPU work that blocks the GPU) + the H2D upload with a tiny CUDA
// kernel that writes the F16 mask straight into its device buffer.
//
// The pattern EXACTLY mirrors the host fill in ggml_ops_gemma4_verify.cpp's
// get_causal_mask (non-bidirectional / text path):
//   threshold = nPast + qi
//   lo = max(window > 0 ? threshold - window + 1 : 0, 0)
//   hi = min(threshold, validLen - 1)
//   mask[qi][ki] = 0 for ki in [lo, hi], else -inf   (ki in [0, kvLen))
// fp16 0.0 (0x0000) and -inf (0xFC00) match ggml_fp32_to_fp16, so the produced
// mask is bit-identical to the host path.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

__global__ void tsg_fill_causal_mask_f16_kernel(
    __half* __restrict__ mask, int kvLen, int N, int nPast, int window, int validLen)
{
    const int qi = blockIdx.x;
    if (qi >= N) return;

    const int threshold = nPast + qi;
    int lo = (window > 0) ? (threshold - window + 1) : 0;
    if (lo < 0) lo = 0;
    int hi = (threshold < validLen - 1) ? threshold : (validLen - 1);

    __half* row = mask + (size_t) qi * kvLen;
    const __half zero = __float2half(0.0f);
    const __half ninf = __float2half(-INFINITY);
    for (int ki = threadIdx.x; ki < kvLen; ki += blockDim.x)
        row[ki] = (ki >= lo && ki <= hi) ? zero : ninf;
}

// Fill one [kvLen, N] F16 causal(+windowed) mask on stream 0. Not synchronized;
// the caller batches several fills then calls tsg_cuda_sync_stream0 once before
// enqueuing the graph (which reads the masks on the backend stream).
extern "C" bool tsg_cuda_fill_causal_mask_f16(
    void* mask_dev, int kvLen, int N, int nPast, int window, int validLen)
{
    if (mask_dev == nullptr || kvLen <= 0 || N <= 0) return false;
    const int threads = 256;
    tsg_fill_causal_mask_f16_kernel<<<(unsigned) N, threads, 0, 0>>>(
        (__half*) mask_dev, kvLen, N, nPast, window, validLen);
    return cudaGetLastError() == cudaSuccess;
}

// Block the host until the stream-0 mask fills complete, so the subsequent
// backend-stream graph compute sees the finished masks (host-ordered).
extern "C" bool tsg_cuda_sync_stream0(void)
{
    return cudaStreamSynchronize(0) == cudaSuccess;
}

// ---------------------------------------------------------------------------
// Highest NVIDIA compute capability among the visible devices, encoded exactly
// the way ggml-cuda encodes it (major*100 + minor*10; 8.6 -> 860, 12.0 -> 1200,
// GGML_CUDA_CC_ADA_LOVELACE == 890). ggml_cuda_get_best_fattn_kernel's choice
// between the VEC and the MMA flash-attention kernel is device-dependent, and
// kv_window_needs_cuda_flash_attn_copy has to mirror that choice to know
// whether a truncated K/V window must be materialised. ggml-cuda keeps its
// device table private (common.cuh), so ask the driver directly.
//
// The MAXIMUM is deliberate: on a heterogeneous host the copy has to be taken
// if ANY device could run the VEC kernel, so erring high errs safe.
extern "C" int tsg_cuda_max_compute_capability(void)
{
    static const int cc = [] {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
            return 0;
        int best = 0;
        for (int i = 0; i < count; i++)
        {
            int major = 0, minor = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, i) != cudaSuccess)
                continue;
            if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, i) != cudaSuccess)
                continue;
            const int v = major * 100 + minor * 10;
            if (v > best) best = v;
        }
        return best;
    }();
    return cc;
}
