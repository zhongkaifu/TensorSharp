// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// ============================================================================
// On-device DiffusionGemma sampler (CUDA).
//
// The DiffusionGemma denoising loop produces a [vocab, C] logits tensor every
// step (C = canvas length, vocab ~= 262K), then on the host computes, per canvas
// position: the argmax (emitted token), the Shannon entropy of softmax(logits *
// inv_temp) (the accept-ordering key), one multinomial sample (the accepted
// token), and the top-K tokens + their softmax weights (the next step's
// self-conditioning soft-embedding). That host pass forces a ~268 MB device->host
// logits download AND two full-vocab CPU sweeps every step.
//
// This kernel runs ALL of that on the GPU directly on the device logits tensor,
// so only C ints/floats (+ C*K for the top-K) come back instead of the full
// [vocab, C] block. It is a port of llama.cpp's diffusion-sampling.cu
// (argmax / entropy / multinomial) extended with a per-position top-K for
// TensorSharp's top-K soft-embedding self-conditioning, and with the final-logit
// softcap folded in so the lm_head graph can emit RAW logits.
//
// One CUDA block per canvas position; 256 threads. Reductions run in shared
// memory. The math mirrors the host worker (DiffusionGemmaSampler.DenoiseStep /
// ComputeSelfConditioning) so the denoising trajectory is unchanged to FP
// reduction tolerance.
// ============================================================================
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <map>
#include <mutex>

namespace {

constexpr int kThreads = 1024;

// Fast tanh via the SFU exp intrinsic (~10 cycles) instead of the accurate library tanhf (~80
// instructions, which dominated the per-element softmax/entropy pass — ~500 ms over the full vocab).
// Sampling only needs FP-reduction-level accuracy, so this approximation is well within tolerance.
__device__ __forceinline__ float fast_tanh(float x) {
    const float e = __expf(-2.0f * x);
    return (1.0f - e) / (1.0f + e);
}

// scaled, softcapped logit: x = inv_temp * (softcap>0 ? softcap*tanh(raw/softcap) : raw).
// softcap is monotonic increasing so argmax(x) == argmax(raw); inv_temp>0 keeps the order too.
__device__ __forceinline__ float scaled_logit(float raw, float inv_temp, float softcap) {
    float x = raw;
    if (softcap > 0.0f) {
        x = softcap * fast_tanh(x / softcap);
    }
    return x * inv_temp;
}

// One block per canvas position (row). Computes argmax, entropy, a multinomial draw
// (inverse-CDF with the row's pre-drawn u), and the top-K tokens + softmax-over-top-K
// weights. K is bounded by kMaxTopK.
//
// softcap (final-logit softcapping) and inv_temp are both strictly monotonic in the raw logit, so
// argmax and the top-K SELECTION are done on the RAW logits (cheap float compares, no transcendental)
// and the expensive scaled_logit (tanhf when softcap>0) is evaluated only where the softmax math truly
// needs it: once per element for the entropy/Z reduction, on one slice for the multinomial walk, and on
// the K selected logits. This keeps the kernel a handful of full-vocab passes (as in llama.cpp's
// diffusion-sampling.cu) instead of K+ tanhf sweeps.
constexpr int kMaxTopK = 256;

__global__ void diffusion_sample_kernel(
        const float * __restrict__ logits,   // [n_vocab, n_rows] contiguous, row-major over rows
        const float * __restrict__ u,         // [n_rows] uniform(0,1) per row
        int   * __restrict__ argmax,          // [n_rows]
        float * __restrict__ entropy,         // [n_rows]
        int   * __restrict__ sampled,         // [n_rows]
        int   * __restrict__ top_tokens,      // [n_rows * K] (may be null)
        float * __restrict__ top_probs,       // [n_rows * K] (may be null)
        const int   n_vocab,
        const float inv_temp,
        const float softcap,
        const int   K) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float s_val[kThreads];
    __shared__ float s_sum[kThreads];   // per-thread contiguous-slice exp-sum (kept for the multinomial scan)
    __shared__ int   s_idx[kThreads];

    const float * row_logits = logits + (size_t) row * n_vocab;

    // ---- pass 1: max + argmax over RAW logits (strided) ----
    float local_max = -FLT_MAX;
    int   local_idx = 0;
    for (int v = tid; v < n_vocab; v += blockDim.x) {
        const float x = row_logits[v];
        if (x > local_max) { local_max = x; local_idx = v; }
    }
    s_val[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            // first-index-wins tie break to match host IndexOfMax
            const bool take = (s_val[tid + stride] > s_val[tid]) ||
                              (s_val[tid + stride] == s_val[tid] && s_idx[tid + stride] < s_idx[tid]);
            if (take) { s_val[tid] = s_val[tid + stride]; s_idx[tid] = s_idx[tid + stride]; }
        }
        __syncthreads();
    }
    const float raw_max = s_val[0];
    const int   amax    = s_idx[0];
    const float max_l   = scaled_logit(raw_max, inv_temp, softcap);   // scaled max (argmax is order-preserving)

    // ---- pass 2 (contiguous slices): slice exp-sum + T contribution (entropy + multinomial scan) ----
    const int chunk = (n_vocab + blockDim.x - 1) / blockDim.x;
    const int beg   = tid * chunk;
    const int end   = min(beg + chunk, n_vocab);
    float slice_z = 0.0f;
    float slice_t = 0.0f;
    for (int v = beg; v < end; ++v) {
        const float d = scaled_logit(row_logits[v], inv_temp, softcap) - max_l;
        const float e = expf(d);
        slice_z += e;
        slice_t += d * e;
    }
    s_sum[tid] = slice_z;   // kept intact for the multinomial exclusive scan below
    s_val[tid] = slice_t;
    __syncthreads();

    __shared__ float s_z;
    __shared__ int   s_tok;
    if (tid == 0) {
        float z = 0.0f, t = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) { z += s_sum[i]; t += s_val[i]; }
        s_z = z;
        argmax[row]  = amax;
        entropy[row] = logf(z) - t / z;   // == log Z + max - (sum x*e)/Z (host formula), FP-equivalent

        // multinomial: exclusive-scan the slice sums, locate the slice that spans r = u*Z
        const float r = u[row] * z;
        s_tok    = n_vocab - 1;   // FP guard (host default if cum never reaches r)
        s_idx[0] = -1;            // crossing slice (none -> default stands)
        float pref = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) {
            const float next = pref + s_sum[i];
            if (next >= r) { s_idx[0] = i; s_val[0] = pref; break; }
            pref = next;
        }
    }
    __syncthreads();

    // only the crossing thread re-walks its (single) slice — one slice of scaled_logit, not the full vocab
    if (tid == s_idx[0]) {
        const float r = u[row] * s_z;
        float cum = s_val[0];
        for (int v = beg; v < end; ++v) {
            cum += expf(scaled_logit(row_logits[v], inv_temp, softcap) - max_l);
            if (cum >= r) { s_tok = v; break; }
        }
    }
    __syncthreads();
    if (tid == 0) { sampled[row] = s_tok; }

    // ---- top-K tokens + softmax-over-top-K weights (for self-conditioning) ----
    // Two-stage so the full vocab is read only ONCE (not K times): each thread keeps a local top-L over
    // its strided (coalesced) slice; the L*blockDim candidates are then reduced to the global top-K in
    // shared memory. With blockDim threads >> K, L=4 captures the true top-K with overwhelming probability
    // (a thread would need >L of the global top-K to fall in its residue class). Selection on RAW logits
    // (monotonic == scaled order); only the K winners are scaled + softmaxed.
    if (K <= 0 || top_tokens == nullptr || top_probs == nullptr) {
        return;
    }
    constexpr int L = 4;
    __shared__ float s_cand_val[kThreads * L];
    __shared__ int   s_cand_idx[kThreads * L];
    __shared__ float s_topx[kMaxTopK];   // RAW logit of the i-th largest
    __shared__ int   s_toptk[kMaxTopK];

    // stage 1: per-thread local top-L (descending), one coalesced pass over the vocab
    float lv[L]; int li[L];
    #pragma unroll
    for (int j = 0; j < L; ++j) { lv[j] = -FLT_MAX; li[j] = -1; }
    for (int v = tid; v < n_vocab; v += blockDim.x) {
        const float x = row_logits[v];
        if (x > lv[L - 1]) {
            lv[L - 1] = x; li[L - 1] = v;
            #pragma unroll
            for (int p = L - 1; p > 0; --p) {
                if (lv[p] > lv[p - 1]) {
                    float tv = lv[p]; lv[p] = lv[p - 1]; lv[p - 1] = tv;
                    int   ti = li[p]; li[p] = li[p - 1]; li[p - 1] = ti;
                }
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < L; ++j) { s_cand_val[tid * L + j] = lv[j]; s_cand_idx[tid * L + j] = li[j]; }
    __syncthreads();

    // stage 2: global top-K over the (blockDim*L) candidates via K eligibility-bounded max reductions
    const int nCand = blockDim.x * L;
    const int kk = min(K, kMaxTopK);
    __shared__ float s_prevVal;
    __shared__ int   s_prevIdx;
    if (tid == 0) { s_prevVal = FLT_MAX; s_prevIdx = -1; }
    __syncthreads();
    for (int i = 0; i < kk; ++i) {
        const float prevVal = s_prevVal;
        const int   prevIdx = s_prevIdx;
        float bVal = -FLT_MAX; int bIdx = -1;
        for (int c = tid; c < nCand; c += blockDim.x) {
            const int   vi = s_cand_idx[c];
            if (vi < 0) continue;
            const float x = s_cand_val[c];
            const bool eligible = (x < prevVal) || (x == prevVal && vi > prevIdx);
            if (!eligible) continue;
            if (bIdx == -1 || x > bVal || (x == bVal && vi < bIdx)) { bVal = x; bIdx = vi; }
        }
        s_val[tid] = bVal;
        s_idx[tid] = bIdx;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float ov = s_val[tid + stride];
                const int   oi = s_idx[tid + stride];
                const bool take = oi != -1 && (s_idx[tid] == -1 || ov > s_val[tid] ||
                                  (ov == s_val[tid] && oi < s_idx[tid]));
                if (take) { s_val[tid] = ov; s_idx[tid] = oi; }
            }
            __syncthreads();
        }
        if (tid == 0) {
            s_topx[i]  = s_val[0];
            s_toptk[i] = s_idx[0];
            s_prevVal  = s_val[0];
            s_prevIdx  = s_idx[0];
        }
        __syncthreads();
    }

    // softmax over the K selected logits, scaled (thread 0; K is small)
    if (tid == 0) {
        const float m = scaled_logit(s_topx[0], inv_temp, softcap);   // largest scaled logit
        float sum = 0.0f;
        for (int i = 0; i < kk; ++i) { const float e = expf(scaled_logit(s_topx[i], inv_temp, softcap) - m); s_topx[i] = e; sum += e; }
        const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
        int * tt = top_tokens + (size_t) row * K;
        float * tp = top_probs + (size_t) row * K;
        for (int i = 0; i < kk; ++i) { tt[i] = s_toptk[i]; tp[i] = s_topx[i] * inv; }
        for (int i = kk; i < K; ++i) { tt[i] = s_toptk[kk - 1]; tp[i] = 0.0f; }   // pad (vocab < K)
    }
}

// Per-device grow-only scratch so the steady state has no cudaMalloc.
struct dg_sample_scratch {
    float * u        = nullptr;
    int   * argmax   = nullptr;
    float * entropy  = nullptr;
    int   * sampled  = nullptr;
    int   * toptk    = nullptr;
    float * topp     = nullptr;
    int     cap_rows = 0;
    int     cap_k    = 0;
};

std::mutex g_dg_sample_mutex;
std::map<int, dg_sample_scratch> g_dg_sample;

bool reserve(dg_sample_scratch & s, int rows, int K) {
    if (s.cap_rows >= rows && s.cap_k >= K) return true;
    const int newRows = s.cap_rows >= rows ? s.cap_rows : rows;
    const int newK    = s.cap_k >= K ? s.cap_k : K;
    if (s.u)       cudaFree(s.u);
    if (s.argmax)  cudaFree(s.argmax);
    if (s.entropy) cudaFree(s.entropy);
    if (s.sampled) cudaFree(s.sampled);
    if (s.toptk)   cudaFree(s.toptk);
    if (s.topp)    cudaFree(s.topp);
    s = dg_sample_scratch{};
    bool ok = true;
    ok = ok && cudaMalloc((void **) &s.u,       (size_t) newRows * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc((void **) &s.argmax,  (size_t) newRows * sizeof(int))   == cudaSuccess;
    ok = ok && cudaMalloc((void **) &s.entropy, (size_t) newRows * sizeof(float)) == cudaSuccess;
    ok = ok && cudaMalloc((void **) &s.sampled, (size_t) newRows * sizeof(int))   == cudaSuccess;
    ok = ok && cudaMalloc((void **) &s.toptk,   (size_t) newRows * newK * sizeof(int))   == cudaSuccess;
    ok = ok && cudaMalloc((void **) &s.topp,    (size_t) newRows * newK * sizeof(float)) == cudaSuccess;
    if (!ok) {
        if (s.u) cudaFree(s.u); if (s.argmax) cudaFree(s.argmax); if (s.entropy) cudaFree(s.entropy);
        if (s.sampled) cudaFree(s.sampled); if (s.toptk) cudaFree(s.toptk); if (s.topp) cudaFree(s.topp);
        s = dg_sample_scratch{};
        return false;
    }
    s.cap_rows = newRows;
    s.cap_k = newK;
    return true;
}

} // anonymous namespace

// Returns true on success. logits_dev must be a contiguous [n_vocab, n_rows] F32 device buffer
// (the lm_head graph output, raw / pre-softcap). The caller must have synchronized the backend so
// the logits are ready. Runs on the default stream; downloads only the small per-row outputs.
extern "C" bool tsg_cuda_diffusion_sample(
        const void * logits_dev,
        int n_vocab, int n_rows,
        float inv_temp, float softcap,
        const float * u_host,
        int K,
        int * argmax_host,
        float * entropy_host,
        int * sampled_host,
        int * top_tokens_host,
        float * top_probs_host) {
    if (logits_dev == nullptr || u_host == nullptr || argmax_host == nullptr ||
        entropy_host == nullptr || sampled_host == nullptr || n_vocab <= 0 || n_rows <= 0) {
        return false;
    }
    // verify logits_dev is a device pointer (else the caller is not on a CUDA backend -> fall back)
    cudaPointerAttributes attr{};
    if (cudaPointerGetAttributes(&attr, logits_dev) != cudaSuccess ||
        (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged)) {
        cudaGetLastError();
        return false;
    }
    const bool wantTopK = K > 0 && top_tokens_host != nullptr && top_probs_host != nullptr;
    const int kEff = wantTopK ? K : 0;

    int device = 0;
    cudaGetDevice(&device);

    std::lock_guard<std::mutex> lock(g_dg_sample_mutex);
    dg_sample_scratch & s = g_dg_sample[device];
    if (!reserve(s, n_rows, kEff > 0 ? kEff : 1)) return false;

    if (cudaMemcpyAsync(s.u, u_host, (size_t) n_rows * sizeof(float), cudaMemcpyHostToDevice, 0) != cudaSuccess)
        return false;

    const bool timing = getenv("DIFFUSION_SAMPLE_TIMING") != nullptr;
    cudaEvent_t t0{}, t1{};
    if (timing) { cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventRecord(t0, 0); }

    diffusion_sample_kernel<<<n_rows, kThreads, 0, 0>>>(
        (const float *) logits_dev, s.u, s.argmax, s.entropy, s.sampled,
        kEff > 0 ? s.toptk : nullptr, kEff > 0 ? s.topp : nullptr,
        n_vocab, inv_temp, softcap, kEff);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (timing) {
        cudaEventRecord(t1, 0);
        cudaEventSynchronize(t1);
        float ms = 0.0f; cudaEventElapsedTime(&ms, t0, t1);
        fprintf(stderr, "[dg-sample-kernel] %.2f ms (rows=%d vocab=%d K=%d)\n", ms, n_rows, n_vocab, kEff);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

    cudaMemcpyAsync(argmax_host,  s.argmax,  (size_t) n_rows * sizeof(int),   cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(entropy_host, s.entropy, (size_t) n_rows * sizeof(float), cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(sampled_host, s.sampled, (size_t) n_rows * sizeof(int),   cudaMemcpyDeviceToHost, 0);
    if (kEff > 0) {
        cudaMemcpyAsync(top_tokens_host, s.toptk, (size_t) n_rows * kEff * sizeof(int),   cudaMemcpyDeviceToHost, 0);
        cudaMemcpyAsync(top_probs_host,  s.topp,  (size_t) n_rows * kEff * sizeof(float), cudaMemcpyDeviceToHost, 0);
    }
    return cudaStreamSynchronize(0) == cudaSuccess;
}
