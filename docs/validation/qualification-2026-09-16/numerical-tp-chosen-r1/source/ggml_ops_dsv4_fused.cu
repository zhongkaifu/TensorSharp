// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Fused DeepSeek V4 (Flash) CUDA kernels + the TensorSharp fused-op backend.
//
// The DSV4 decode graph is launch-bound: ~7.8K ggml nodes per token, most of
// them tiny elementwise/view chains. These kernels collapse the hot chains
// (compressors, attention prologue/epilogue, MoE routing/reduction, clamped
// SwiGLU, hyper-connection gates, top-k visibility masks) into single
// launches.
//
// Injection works without touching the vendored ggml: the graph builder emits
// GGML_OP_CUSTOM nodes (public ggml_custom_4d API) whose userdata points at a
// tsg_dsv4_fused_desc, and the ggml-backend implemented here runs them.
//
// That backend is registered with ggml_backend_sched INSTEAD OF the CUDA
// backend it wraps, not alongside it, and claims the CUDA device's ops and
// buffer types as well as its own fused nodes. Ordinary nodes are forwarded to
// the CUDA backend as graph views; fused nodes launch here. Everything goes to
// the CUDA backend's own stream, asynchronously, so one device's whole subgraph
// is a single ordered submission.
//
// Registering both backends is what the earlier design did, and it is why this
// matters: the scheduler splits the graph wherever the backend changes, which a
// DeepSeek V4.1 layer does about 14 times. That cost 565 splits per decode
// token, each ~5.6 nodes of work, and a blocking host synchronization at every
// one of the 564 boundaries.
// ---------------------------------------------------------------------------

#include "ggml_ops_dsv4_fused.h"
#include "ggml_ops_matmul_precision.h"
#include "ggml_ops_attention_precision.h"
#include "ggml_ops_q8_precision.h"
#include "dsv41_quant.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"   // ggml_backend_cuda_context (stream access)

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>

bool tsg_dsv4_cuda_supports_native_bf16(ggml_backend_t backend) {
    if (!ggml_backend_is_cuda(backend)) return false;
    const auto * context = static_cast<const ggml_backend_cuda_context *>(backend->context);
    const int cc = ggml_cuda_info().devices[context->device].cc;
    return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float tsg_dsv4_softplus(float x) {
    return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

static __device__ __forceinline__ float tsg_dsv4_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// per-element softmax-weighted window compression from [state ring | scratch]
// sources, RMS norm, table-driven RoPE, F16 cache commit — one thread block
// per compressed row — followed by the state-ring persist update.
static __global__ void tsg_dsv4_compress_f32(
        const float * __restrict__ kv_scr,
        const float * __restrict__ sc_scr,
        const float * __restrict__ ring_kv,
        const float * __restrict__ ring_sc,
        const float * __restrict__ norm_w,
        const float * __restrict__ rope_tab,
        const int32_t * __restrict__ read_idxs,
        const int32_t * __restrict__ write_meta,
        half * __restrict__ cache,
        const int n_blocks,
        const int ratio,
        const int coff,
        const int head,
        const int n_rope,
        const float eps,
        const int ss) {
    const int b = blockIdx.x;
    if (b >= n_blocks) {
        return;
    }

    const int64_t cw = (int64_t) coff * head;
    const int W = coff * ratio;

    const int64_t cache_row = write_meta[2*b + 0];
    const int64_t pos       = write_meta[2*b + 1];

    __shared__ float sh_out[512];
    __shared__ float sh_red[256];

    float acc2 = 0.0f;

    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        float m  = -INFINITY;
        float se = 0.0f;
        float sv = 0.0f;

        for (int w = 0; w < W; ++w) {
            const bool prev = coff == 2 && w < ratio;
            const int32_t idx = prev ? read_idxs[(int64_t) b*ratio + w]
                : coff == 2 ? read_idxs[(int64_t) ratio*n_blocks + (int64_t) b*ratio + (w - ratio)]
                            : read_idxs[(int64_t) b*ratio + w];
            const int64_t off = (coff == 2 && !prev) ? head : 0;

            const bool from_ring = idx <= ss;
            const int64_t src_off = from_ring ? (int64_t) idx * cw + off + d
                                              : (int64_t) (idx - ss - 1) * cw + off + d;
            const float sc = from_ring ? ring_sc[src_off] : sc_scr[src_off];
            const float kv = from_ring ? ring_kv[src_off] : kv_scr[src_off];

            if (sc > m) {
                const float r = expf(m - sc); // 0 when m == -inf
                se *= r;
                sv *= r;
                m = sc;
            }
            // guard the -inf boundary rows: expf(-inf - -inf) would be NaN
            const float e = sc == -INFINITY ? 0.0f : expf(sc - m);
            se += e;
            sv += e * kv;
        }

        const float out = se > 0.0f ? sv/se : 0.0f;
        sh_out[d] = out;
        acc2 += out*out;
    }

    // block-wide sum of squares for the RMS norm
    sh_red[threadIdx.x] = acc2;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh_red[threadIdx.x] += sh_red[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(sh_red[0]/head + eps);

    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        sh_out[d] *= scale * norm_w[d];
    }
    __syncthreads();

    // RoPE (normal/interleaved pairs) on the last n_rope dims + F16 store
    const int base = head - n_rope;
    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        float v = sh_out[d];
        if (d >= base) {
            const int i = (d - base) >> 1;
            const float c = rope_tab[pos*n_rope + 2*i + 0];
            const float s = rope_tab[pos*n_rope + 2*i + 1];
            const float x0 = sh_out[base + 2*i + 0];
            const float x1 = sh_out[base + 2*i + 1];
            v = ((d - base) & 1) == 0 ? x0*c - x1*s : x0*s + x1*c;
        }
        cache[cache_row*head + d] = __float2half(v);
    }
}

static __global__ void tsg_dsv4_persist_f32(
        const float * __restrict__ kv_scr,
        const float * __restrict__ sc_scr,
        float * __restrict__ ring_kv,
        float * __restrict__ ring_sc,
        const int32_t * __restrict__ persist_meta,
        const int np,
        const int64_t cw) {
    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int64_t) np * cw) {
        return;
    }

    const int p = (int) (t / cw);
    const int64_t d = t % cw;

    const int64_t src  = persist_meta[2*p + 0];
    const int64_t drow = persist_meta[2*p + 1];

    ring_kv[drow*cw + d] = kv_scr[src*cw + d];
    ring_sc[drow*cw + d] = sc_scr[src*cw + d];
}

// blockIdx.x in [0, n_head]: heads 0..n_head-1 normalize+rope q into dst,
// block n_head normalizes+ropes kv and commits it (F16) into the SWA ring.
static __global__ void tsg_dsv4_attn_prep_f32(
        const float * __restrict__ q_raw,     // [head*n_head, nt]
        const float * __restrict__ kv_raw,    // [head, nt]
        const float * __restrict__ kv_norm_w, // [head]
        const float * __restrict__ rope_tab,  // [n_rope, n_ctx]
        const int32_t * __restrict__ pos,     // [nt]
        half * __restrict__ ring,             // [head, ring_rows]
        const int64_t * __restrict__ raw_idxs,// [nt]
        float * __restrict__ dst,             // [head, n_head, nt]
        const int n_head,
        const int head,
        const int n_rope,
        const float eps) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;

    __shared__ float sh[512];
    __shared__ float sh_red[256];

    const bool is_kv = h == n_head;
    const float * src = is_kv ? kv_raw + (int64_t) t * head
                              : q_raw + (int64_t) t * head * n_head + (int64_t) h * head;

    float acc2 = 0.0f;
    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        const float v = src[d];
        sh[d] = v;
        acc2 += v*v;
    }
    sh_red[threadIdx.x] = acc2;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sh_red[threadIdx.x] += sh_red[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(sh_red[0]/head + eps);

    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        sh[d] *= is_kv ? scale * kv_norm_w[d] : scale;
    }
    __syncthreads();

    const int rbase = head - n_rope;
    const int64_t p = pos[t];

    if (is_kv) {
        half * out = ring + raw_idxs[t] * head;
        for (int d = threadIdx.x; d < head; d += blockDim.x) {
            float v = sh[d];
            if (d >= rbase) {
                const int i = (d - rbase) >> 1;
                const float c = rope_tab[p*n_rope + 2*i + 0];
                const float s = rope_tab[p*n_rope + 2*i + 1];
                const float x0 = sh[rbase + 2*i + 0];
                const float x1 = sh[rbase + 2*i + 1];
                v = ((d - rbase) & 1) == 0 ? x0*c - x1*s : x0*s + x1*c;
            }
            out[d] = __float2half(v);
        }
    } else {
        float * out = dst + (int64_t) t * head * n_head + (int64_t) h * head;
        for (int d = threadIdx.x; d < head; d += blockDim.x) {
            float v = sh[d];
            if (d >= rbase) {
                const int i = (d - rbase) >> 1;
                const float c = rope_tab[p*n_rope + 2*i + 0];
                const float s = rope_tab[p*n_rope + 2*i + 1];
                const float x0 = sh[rbase + 2*i + 0];
                const float x1 = sh[rbase + 2*i + 1];
                v = ((d - rbase) & 1) == 0 ? x0*c - x1*s : x0*s + x1*c;
            }
            out[d] = v;
        }
    }
}

// inverse RoPE on the tail dims of each attention output head + store in the
// grouped output layout [group_dim, nt, n_groups].
static __global__ void tsg_dsv4_attn_finish_f32(
        const float * __restrict__ attn,     // [head*n_head, nt]
        const float * __restrict__ rope_tab, // [n_rope, n_ctx]
        const int32_t * __restrict__ pos,    // [nt]
        float * __restrict__ dst,            // [group_dim, nt, n_groups]
        const int n_head,
        const int head,
        const int n_rope,
        const int heads_per_group,
        const int nt) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;

    __shared__ float sh[512];

    const float * src = attn + (int64_t) t * head * n_head + (int64_t) h * head;
    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        sh[d] = src[d];
    }
    __syncthreads();

    const int rbase = head - n_rope;
    const int64_t p = pos[t];
    const int g  = h / heads_per_group;
    const int hg = h % heads_per_group;
    const int64_t group_dim = (int64_t) heads_per_group * head;

    float * out = dst + (int64_t) g * group_dim * nt + (int64_t) t * group_dim + (int64_t) hg * head;

    for (int d = threadIdx.x; d < head; d += blockDim.x) {
        float v = sh[d];
        if (d >= rbase) {
            // inverse (transpose) rotation of the forward RoPE
            const int i = (d - rbase) >> 1;
            const float c = rope_tab[p*n_rope + 2*i + 0];
            const float s = rope_tab[p*n_rope + 2*i + 1];
            const float x0 = sh[rbase + 2*i + 0];
            const float x1 = sh[rbase + 2*i + 1];
            v = ((d - rbase) & 1) == 0 ? x0*c + x1*s : -x0*s + x1*c;
        }
        out[d] = v;
    }
}

// sel = sqrt(softplus(logits)) + bias; ids = top-k (descending, ties to the
// lower index). One block of 256 threads per token, n_expert <= 1024.
static __global__ void tsg_dsv4_moe_topk_f32(
        const float * __restrict__ logits, // [n_expert, nt]
        const float * __restrict__ bias,   // [n_expert]
        int32_t * __restrict__ dst,        // [n_used, nt]
        const int n_expert,
        const int n_used) {
    const int t = blockIdx.x;

    extern __shared__ float sh_sel[]; // [n_expert]
    __shared__ float sh_val[256];
    __shared__ int   sh_idx[256];

    const float * lg = logits + (int64_t) t * n_expert;
    for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
        sh_sel[e] = sqrtf(tsg_dsv4_softplus(lg[e])) + bias[e];
    }
    __syncthreads();

    for (int k = 0; k < n_used; ++k) {
        float best = -INFINITY;
        int bidx = -1;
        for (int e = threadIdx.x; e < n_expert; e += blockDim.x) {
            const float v = sh_sel[e];
            if (v > best || (v == best && e < bidx)) {
                best = v;
                bidx = e;
            }
        }
        sh_val[threadIdx.x] = best;
        sh_idx[threadIdx.x] = bidx;
        __syncthreads();
        for (int s = blockDim.x/2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                const float ov = sh_val[threadIdx.x + s];
                const int   oi = sh_idx[threadIdx.x + s];
                if (ov > sh_val[threadIdx.x] || (ov == sh_val[threadIdx.x] && oi >= 0 &&
                        (sh_idx[threadIdx.x] < 0 || oi < sh_idx[threadIdx.x]))) {
                    sh_val[threadIdx.x] = ov;
                    sh_idx[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const int sel = sh_idx[0] < 0 ? 0 : sh_idx[0];
            dst[(int64_t) t * n_used + k] = sel;
            sh_sel[sel] = -INFINITY;
        }
        __syncthreads();
    }
}

// w_e = sqrt(softplus(logits[ids_e])); optional sum-normalization; scale.
static __global__ void tsg_dsv4_moe_weights_f32(
        const float * __restrict__ logits, // [n_expert, nt]
        const int32_t * __restrict__ ids,  // [n_used, nt]
        float * __restrict__ dst,          // [1, n_used, nt]
        const int n_expert,
        const int n_used,
        const int norm,
        const float scale) {
    const int t = blockIdx.x;
    const int e = threadIdx.x;

    float w = 0.0f;
    if (e < n_used) {
        const int32_t id = ids[(int64_t) t * n_used + e];
        w = sqrtf(tsg_dsv4_softplus(logits[(int64_t) t * n_expert + id]));
    }

    if (norm) {
        float sum = w;
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, off, 32);
        }
        sum = fmaxf(sum, 6.103515625e-5f);
        w /= sum;
    }

    if (e < n_used) {
        dst[(int64_t) t * n_used + e] = w * scale;
    }
}

// dst = sum_e experts[:,e,t]*w[e,t] + shexp[:,t]
static __global__ void tsg_dsv4_expert_reduce_f32(
        const float * __restrict__ experts, // [n_embd, n_used, nt]
        const float * __restrict__ weights, // [1, n_used, nt]
        const float * __restrict__ shexp,   // [n_embd, nt]
        float * __restrict__ dst,           // [n_embd, nt]
        const int n_embd,
        const int n_used) {
    const int t = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n_embd) {
        return;
    }

    const float * ex = experts + (int64_t) t * n_embd * n_used + d;
    const float * w  = weights + (int64_t) t * n_used;

    float acc = 0.0f;
    for (int e = 0; e < n_used; ++e) {
        acc += ex[(int64_t) e * n_embd] * w[e];
    }
    acc += shexp[(int64_t) t * n_embd + d];
    dst[(int64_t) t * n_embd + d] = acc;
}

static __global__ void tsg_dsv4_swiglu_clamp_f32(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        float * __restrict__ dst,
        const int64_t n,
        const float limit) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float g = fminf(gate[i], limit);
    const float u = fminf(fmaxf(up[i], -limit), limit);
    dst[i] = u * (g / (1.0f + expf(-g)));
}

// dst rows [0,hc): sigmoid(mixes_pre*scale0 + base_pre) + eps
// dst rows [hc,2hc): 2*sigmoid(mixes_post*scale1 + base_post)
static __global__ void tsg_dsv4_hc_gates_f32(
        const float * __restrict__ mixes, // [(2+hc)*hc, nt]
        const float * __restrict__ scale, // [3]
        const float * __restrict__ base,  // [(2+hc)*hc]
        float * __restrict__ dst,         // [2*hc, nt]
        const int64_t n,
        const int hc,
        const int64_t sm1,
        const int64_t ss0,
        const int64_t sb0,
        const float eps) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int64_t t = i / (2*hc);
    const int     r = (int) (i % (2*hc));
    const bool  pre = r < hc;

    const float sc = scale[(pre ? 0 : 1) * ss0];
    const float b  = base[(int64_t) r * sb0];
    const float v  = tsg_dsv4_sigmoid(mixes[t*sm1 + r] * sc + b);

    dst[t*2*hc + r] = pre ? v + eps : 2.0f*v;
}

// dst[r,t] = base[r,t] for r < offset (raw window region);
//            base[r,t] where (r-offset) in top_k[:,t], else -inf.
static __global__ void tsg_dsv4_topk_mask_f16(
        const half * __restrict__ base,   // [W, nt]
        const int32_t * __restrict__ topk,// [k, nt]
        half * __restrict__ dst,          // [W, nt]
        const int W,
        const int k,
        const int offset) {
    const int t = blockIdx.x;

    const half * bs = base + (int64_t) t * W;
    half * out      = dst + (int64_t) t * W;

    const half neg_inf = __float2half(-INFINITY);
    for (int r = threadIdx.x; r < W; r += blockDim.x) {
        out[r] = r < offset ? bs[r] : neg_inf;
    }
    __syncthreads();

    const int32_t * tk = topk + (int64_t) t * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const int r = offset + tk[i];
        if (r >= offset && r < W) {
            out[r] = bs[r];
        }
    }
}

// Compact sparse-attention K for decode: dst row r < ring_rows copies the raw
// SWA ring row r, row ring_rows+j gathers compressed cache row topk[j]. K
// doubles as V, so one gather serves the whole attention read. Rows are
// contiguous F16; copy as int (half2 pairs).
static __global__ void tsg_dsv4_kgather_f16(
        const half * __restrict__ ring,    // [head, ring_rows]
        const half * __restrict__ comp,    // [head, n_comp_rows]
        const int32_t * __restrict__ topk, // [n_rows - ring_rows]
        half * __restrict__ dst,           // [head, n_rows]
        const int head,
        const int ring_rows) {
    const int r = blockIdx.x;
    const half * src = r < ring_rows ? ring + (int64_t) r * head
                                     : comp + (int64_t) topk[r - ring_rows] * head;
    const int * s = (const int *) src;
    int * o = (int *) (dst + (int64_t) r * head);
    for (int i = threadIdx.x; i < head/2; i += blockDim.x) {
        o[i] = s[i];
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

// One warp owns a block of 32 (MX) or 16 (NV) activations. Four groups per
// CTA amortize launch overhead while keeping maxima entirely in registers.
static __global__ void tsg_dsv41_quant_f32(const float * input, float * output,
                                         int64_t groups, int mode)
{
    const int lane = threadIdx.x & 31;
    const int block = mode == 2 ? 16 : 32;
    const int64_t group = (int64_t) blockIdx.x * 4 + threadIdx.x / 32;
    if (group >= groups) return;
    const float value = lane < block ? tsg_dsv41_bf16(input[group * block + lane]) : 0.0f;
    float amax = fabsf(value);
    for (int delta = 16; delta > 0; delta /= 2)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, delta));
    const float scale = tsg_dsv41_quant_scale(amax, mode);
    if (lane < block)
        output[group * block + lane] = tsg_dsv41_quant_value(value, scale, mode);
}

static __global__ void tsg_dsv41_candidate_scores_f32(const float * scores, const int32_t * pos,
        float * output, int64_t count, int width, int blocks, int block_size)
{
    const int64_t index = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int64_t query = index / blocks;
    const int block = int(index % blocks);
    float best = -INFINITY;
    for (int key = block * block_size; key < min(width, (block + 1) * block_size); ++key)
        best = fmaxf(best, scores[query * width + key]);
    output[index] = pos[query] >= 0 && block == pos[query] / block_size ? INFINITY : best;
}

static __global__ void tsg_dsv41_candidate_mask_f16(const float * pooled, const int32_t * topk,
        half * output, int width, int blocks, int k, int block_size)
{
    const int query = blockIdx.x;
    half * out = output + (int64_t) query * width;
    for (int key = threadIdx.x; key < width; key += blockDim.x)
        out[key] = __float2half(-INFINITY);
    __syncthreads();
    for (int i = threadIdx.x; i < k; i += blockDim.x)
    {
        const int block = topk[(int64_t) query * k + i];
        if (block < 0 || block >= blocks || !(pooled[(int64_t) query * blocks + block] > -INFINITY)) continue;
        for (int key = block * block_size; key < min(width, (block + 1) * block_size); ++key)
            out[key] = __float2half(0.0f);
    }
}

static void tsg_dsv4_fused_launch(const tsg_dsv4_fused_desc * d, ggml_tensor * dst, cudaStream_t stream)
{
    switch (d->kind)
    {
        case TSG_DSV41_FUSED_QUANT:
        {
            const ggml_tensor * src = dst->src[0];
            const int block = d->i0 == 2 ? 16 : 32;
            GGML_ASSERT(d->i0 >= 0 && d->i0 <= 2);
            GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
            GGML_ASSERT(src->ne[0] % block == 0 && ggml_nelements(src) == ggml_nelements(dst));
            const int64_t groups = ggml_nelements(src) / block;
            tsg_dsv41_quant_f32<<<(unsigned) ((groups + 3) / 4), 128, 0, stream>>>(
                (const float *) src->data, (float *) dst->data, groups, d->i0);
        } break;
        case TSG_DSV41_CANDIDATE_SCORES:
        {
            const ggml_tensor * src = dst->src[0];
            const int64_t count = ggml_nelements(dst);
            GGML_ASSERT(d->i0 > 0 && dst->ne[0] == (src->ne[0] + d->i0 - 1) / d->i0);
            GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
            tsg_dsv41_candidate_scores_f32<<<(unsigned) ((count + 255) / 256), 256, 0, stream>>>(
                (const float *) src->data, (const int32_t *) dst->src[1]->data,
                (float *) dst->data, count, (int) src->ne[0], (int) dst->ne[0], d->i0);
        } break;
        case TSG_DSV41_CANDIDATE_MASK:
        {
            const ggml_tensor * scores = dst->src[0], * topk = dst->src[1];
            GGML_ASSERT(d->i0 > 0 && dst->type == GGML_TYPE_F16);
            GGML_ASSERT(scores->type == GGML_TYPE_F32 && topk->type == GGML_TYPE_I32);
            GGML_ASSERT(ggml_is_contiguous(scores) && ggml_is_contiguous(topk) && ggml_is_contiguous(dst));
            tsg_dsv41_candidate_mask_f16<<<(unsigned) dst->ne[1], 256, 0, stream>>>(
                (const float *) scores->data, (const int32_t *) topk->data, (half *) dst->data,
                (int) dst->ne[0], (int) scores->ne[0], (int) topk->ne[0], d->i0);
        } break;
        case TSG_DSV4_FUSED_COMPRESS:
        {
            const ggml_tensor * st_kv        = dst->src[0];
            const ggml_tensor * st_score     = dst->src[1];
            const ggml_tensor * state_kv     = dst->src[2];
            const ggml_tensor * state_score  = dst->src[3];
            const ggml_tensor * norm_w       = dst->src[4];
            const ggml_tensor * rope_tab     = dst->src[5];
            const ggml_tensor * comp_meta    = dst->src[6];  // I32 [read | write_meta | persist_meta]
            const ggml_tensor * cache        = dst->src[7];

            const int n_blocks = d->i0;
            const int ratio    = d->i1;
            const int coff     = d->i2;
            const int n_rope   = d->i3;
            const float eps    = d->f0;
            const int np       = (int) (d->f1);   // persist count

            const int64_t head = cache->ne[0];
            const int64_t cw   = (int64_t) coff * head;
            const int     ss   = (int) (state_kv->ne[1] - 1);

            const int32_t * meta = (const int32_t *) comp_meta->data;
            const int32_t * read_idxs    = meta;
            const int32_t * write_meta   = meta + (int64_t) coff * ratio * n_blocks;
            const int32_t * persist_meta = write_meta + 2 * n_blocks;

            if (n_blocks > 0) {
                // the compression must read the ring before the persist below updates it
                tsg_dsv4_compress_f32<<<n_blocks, 256, 0, stream>>>(
                    (const float *) st_kv->data, (const float *) st_score->data,
                    (const float *) state_kv->data, (const float *) state_score->data,
                    (const float *) norm_w->data, (const float *) rope_tab->data,
                    read_idxs, write_meta,
                    (half *) cache->data,
                    n_blocks, ratio, coff, (int) head, n_rope, eps, ss);
            }
            if (np > 0) {
                const int64_t total = (int64_t) np * cw;
                const int nblk = (int) ((total + 255) / 256);
                tsg_dsv4_persist_f32<<<nblk, 256, 0, stream>>>(
                    (const float *) st_kv->data, (const float *) st_score->data,
                    (float *) state_kv->data, (float *) state_score->data,
                    persist_meta, np, cw);
            }
        } break;

        case TSG_DSV4_FUSED_ATTN_PREP:
        {
            const ggml_tensor * q_raw     = dst->src[0];
            const ggml_tensor * kv_raw    = dst->src[1];
            const ggml_tensor * kv_norm_w = dst->src[2];
            const ggml_tensor * rope_tab  = dst->src[3];
            const ggml_tensor * pos       = dst->src[4];
            const ggml_tensor * ring      = dst->src[5];
            const ggml_tensor * raw_idxs  = dst->src[6];

            const int head   = (int) dst->ne[0];
            const int n_head = (int) dst->ne[1];
            const int nt     = (int) dst->ne[2];

            const dim3 grid(n_head + 1, nt, 1);
            tsg_dsv4_attn_prep_f32<<<grid, 256, 0, stream>>>(
                (const float *) q_raw->data, (const float *) kv_raw->data,
                (const float *) kv_norm_w->data, (const float *) rope_tab->data,
                (const int32_t *) pos->data,
                (half *) ring->data, (const int64_t *) raw_idxs->data,
                (float *) dst->data,
                n_head, head, d->i0 /*n_rope*/, d->f0 /*eps*/);
        } break;

        case TSG_DSV4_FUSED_ATTN_FINISH:
        {
            const ggml_tensor * attn     = dst->src[0];
            const ggml_tensor * rope_tab = dst->src[1];
            const ggml_tensor * pos      = dst->src[2];

            const int n_rope   = d->i0;
            const int n_groups = d->i1;
            const int head     = d->i2;

            const int nt    = (int) dst->ne[1];
            const int heads = (int) (attn->ne[0] / head);
            const int heads_per_group = heads / n_groups;

            const dim3 grid(heads, nt, 1);
            tsg_dsv4_attn_finish_f32<<<grid, 256, 0, stream>>>(
                (const float *) attn->data, (const float *) rope_tab->data, (const int32_t *) pos->data,
                (float *) dst->data,
                heads, head, n_rope, heads_per_group, nt);
        } break;

        case TSG_DSV4_FUSED_MOE_TOPK:
        {
            const ggml_tensor * logits = dst->src[0];
            const ggml_tensor * bias   = dst->src[1];

            const int n_expert = (int) logits->ne[0];
            const int n_used   = (int) dst->ne[0];
            const int nt       = (int) dst->ne[1];

            tsg_dsv4_moe_topk_f32<<<nt, 256, n_expert*sizeof(float), stream>>>(
                (const float *) logits->data, (const float *) bias->data,
                (int32_t *) dst->data, n_expert, n_used);
        } break;

        case TSG_DSV4_FUSED_MOE_WEIGHTS:
        {
            const ggml_tensor * logits = dst->src[0];
            const ggml_tensor * ids    = dst->src[1];

            const int n_expert = (int) logits->ne[0];
            const int n_used   = (int) dst->ne[1];
            const int nt       = (int) dst->ne[2];

            tsg_dsv4_moe_weights_f32<<<nt, 32, 0, stream>>>(
                (const float *) logits->data, (const int32_t *) ids->data,
                (float *) dst->data, n_expert, n_used, d->i0 /*norm*/, d->f0 /*scale*/);
        } break;

        case TSG_DSV4_FUSED_EXPERT_REDUCE:
        {
            const ggml_tensor * experts = dst->src[0];
            const ggml_tensor * weights = dst->src[1];
            const ggml_tensor * shexp   = dst->src[2];

            const int n_embd = (int) dst->ne[0];
            const int n_used = (int) experts->ne[1];
            const int nt     = (int) dst->ne[1];

            const dim3 grid((n_embd + 255) / 256, nt, 1);
            tsg_dsv4_expert_reduce_f32<<<grid, 256, 0, stream>>>(
                (const float *) experts->data, (const float *) weights->data,
                (const float *) shexp->data, (float *) dst->data, n_embd, n_used);
        } break;

        case TSG_DSV4_FUSED_SWIGLU_CLAMP:
        {
            const ggml_tensor * gate = dst->src[0];
            const ggml_tensor * up   = dst->src[1];

            const int64_t n = ggml_nelements(dst);
            const int nblk = (int) ((n + 255) / 256);
            tsg_dsv4_swiglu_clamp_f32<<<nblk, 256, 0, stream>>>(
                (const float *) gate->data, (const float *) up->data, (float *) dst->data, n, d->f0 /*limit*/);
        } break;

        case TSG_DSV4_FUSED_HC_GATES:
        {
            const ggml_tensor * mixes = dst->src[0];
            const ggml_tensor * scale = dst->src[1];
            const ggml_tensor * base  = dst->src[2];

            const int hc = (int) (dst->ne[0] / 2);
            const int64_t nt = dst->ne[1];
            const int64_t n = 2*hc*nt;

            tsg_dsv4_hc_gates_f32<<<(int) ((n + 255)/256), 256, 0, stream>>>(
                (const float *) mixes->data, (const float *) scale->data, (const float *) base->data,
                (float *) dst->data, n, hc,
                (int64_t) (mixes->nb[1]/sizeof(float)),
                (int64_t) (scale->nb[0]/sizeof(float)),
                (int64_t) (base->nb[0]/sizeof(float)),
                d->f0 /*eps*/);
        } break;

        case TSG_DSV4_FUSED_TOPK_MASK:
        {
            const ggml_tensor * base = dst->src[0];
            const ggml_tensor * topk = dst->src[1];

            const int W  = (int) dst->ne[0];
            const int nt = (int) dst->ne[1];
            const int k  = (int) topk->ne[0];

            tsg_dsv4_topk_mask_f16<<<nt, 256, 0, stream>>>(
                (const half *) base->data, (const int32_t *) topk->data,
                (half *) dst->data, W, k, d->i0 /*offset*/);
        } break;

        case TSG_DSV4_FUSED_KGATHER:
        {
            const ggml_tensor * ring = dst->src[0];
            const ggml_tensor * comp = dst->src[1];
            const ggml_tensor * topk = dst->src[2];

            const int head      = (int) dst->ne[0];
            const int n_rows    = (int) dst->ne[2];
            const int ring_rows = d->i0;

            tsg_dsv4_kgather_f16<<<n_rows, 256, 0, stream>>>(
                (const half *) ring->data, (const half *) comp->data,
                (const int32_t *) topk->data, (half *) dst->data, head, ring_rows);
        } break;

        default:
            GGML_ABORT("tsg_dsv4_fused: unknown kind %d", d->kind);
    }
}

// ---------------------------------------------------------------------------
// Fused-op backend
// ---------------------------------------------------------------------------

struct tsg_dsv4_backend_ctx
{
    int device = 0;
    ggml_backend_t cuda_backend = nullptr;           // paired instance (not owned)
    ggml_backend_buffer_type_t cuda_buft = nullptr;
    char name[32] = {};
    char desc[64] = {};
    ggml_backend_dev_t cuda_dev = nullptr;
    tsg_matmul_cuda_state * matmul = nullptr;
};

static const tsg_dsv4_fused_desc * tsg_dsv4_node_desc(const ggml_tensor * node)
{
    if (node->op != GGML_OP_CUSTOM) return nullptr;
    ggml_custom_op_params p;
    memcpy(&p, node->op_params, sizeof(p));
    const tsg_dsv4_fused_desc * d = (const tsg_dsv4_fused_desc *) p.userdata;
    if (!d || d->magic != TSG_DSV4_FUSED_MAGIC) return nullptr;
    return d;
}

static cudaStream_t tsg_dsv4_backend_stream(tsg_dsv4_backend_ctx * c)
{
    auto * cc = (ggml_backend_cuda_context *) c->cuda_backend->context;
    return cc->stream(c->device, 0);
}

// ---- backend iface ----

static const char * tsg_dsv4_backend_get_name(ggml_backend_t backend)
{
    return ((tsg_dsv4_backend_ctx *) backend->context)->name;
}

static void tsg_dsv4_backend_free(ggml_backend_t backend)
{
    // The device record is this backend's own (see tsg_dsv4_fused_backend_init)
    // and shares the context, so free it here and only here.
    tsg_matmul_cuda_free(((tsg_dsv4_backend_ctx *) backend->context)->matmul);
    delete backend->device;
    delete (tsg_dsv4_backend_ctx *) backend->context;
    delete backend;
}

static ggml_guid_t tsg_dsv4_backend_guid();

// The CUDA backend behind `backend`, or `backend` itself when it is not one of
// ours. Delegated calls must name the CUDA backend, because ggml-cuda checks
// ggml_backend_is_cuda on the arguments it is handed.
static ggml_backend_t tsg_dsv4_unwrap(ggml_backend_t backend)
{
    if (!backend || !backend->guid || !ggml_guid_matches(backend->guid, tsg_dsv4_backend_guid()))
        return backend;
    return ((tsg_dsv4_backend_ctx *) backend->context)->cuda_backend;
}

static void tsg_dsv4_backend_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size)
{
    ggml_backend_t cuda = tsg_dsv4_unwrap(backend);
    cuda->iface.set_tensor_async(cuda, tensor, data, offset, size);
}

static void tsg_dsv4_backend_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size)
{
    ggml_backend_t cuda = tsg_dsv4_unwrap(backend);
    cuda->iface.get_tensor_async(cuda, tensor, data, offset, size);
}

// Peer-to-peer device copies at the layer-split boundaries. Without this the
// scheduler falls back to a host round trip for every cross-device tensor.
static bool tsg_dsv4_backend_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst,
        const ggml_tensor * src, ggml_tensor * dst)
{
    ggml_backend_t cuda_src = tsg_dsv4_unwrap(backend_src);
    ggml_backend_t cuda_dst = tsg_dsv4_unwrap(backend_dst);
    if (!cuda_dst || !cuda_dst->iface.cpy_tensor_async) return false;
    return cuda_dst->iface.cpy_tensor_async(cuda_src, cuda_dst, src, dst);
}

// Dropping these would silently disable ggml-cuda's graph optimization and
// ggml_backend_sched's host-weight offload for every GPU this backend replaces.
static void tsg_dsv4_backend_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph,
        ggml_backend_graph_optimize_params * params)
{
    ggml_backend_t cuda = tsg_dsv4_unwrap(backend);
    if (cuda->iface.graph_optimize) cuda->iface.graph_optimize(cuda, cgraph, params);
}

static void tsg_dsv4_backend_event_record(ggml_backend_t backend, ggml_backend_event_t event)
{
    ggml_backend_t cuda = tsg_dsv4_unwrap(backend);
    cuda->iface.event_record(cuda, event);
}

static void tsg_dsv4_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event)
{
    ggml_backend_t cuda = tsg_dsv4_unwrap(backend);
    cuda->iface.event_wait(cuda, event);
}

static void tsg_dsv4_backend_synchronize(ggml_backend_t backend)
{
    auto * c = (tsg_dsv4_backend_ctx *) backend->context;
    CUDA_CHECK(cudaSetDevice(c->device));
    // A successful submission does not guarantee successful execution. Surface
    // asynchronous kernel errors here instead of losing them until a later
    // peer copy or request touches this CUDA context.
    CUDA_CHECK(cudaStreamSynchronize(tsg_dsv4_backend_stream(c)));
}

// Submission diagnostics (TS_DSV4_PERF>=3). Plain relaxed atomics: they are
// incremented on whichever thread submits a device subgraph and read once the
// forward has finished.
static std::atomic<unsigned long long> tsg_dsv4_stat_calls{0}, tsg_dsv4_stat_views{0},
    tsg_dsv4_stat_fused{0}, tsg_dsv4_stat_nodes{0}, tsg_dsv4_stat_submit_ns{0};

void tsg_dsv4_fused_counters_reset()
{
    tsg_dsv4_stat_calls.store(0, std::memory_order_relaxed);
    tsg_dsv4_stat_views.store(0, std::memory_order_relaxed);
    tsg_dsv4_stat_fused.store(0, std::memory_order_relaxed);
    tsg_dsv4_stat_nodes.store(0, std::memory_order_relaxed);
    tsg_dsv4_stat_submit_ns.store(0, std::memory_order_relaxed);
}

tsg_dsv4_fused_counters tsg_dsv4_fused_counters_read()
{
    tsg_dsv4_fused_counters out;
    out.calls = tsg_dsv4_stat_calls.load(std::memory_order_relaxed);
    out.views = tsg_dsv4_stat_views.load(std::memory_order_relaxed);
    out.fused = tsg_dsv4_stat_fused.load(std::memory_order_relaxed);
    out.nodes = tsg_dsv4_stat_nodes.load(std::memory_order_relaxed);
    out.submit_ms = tsg_dsv4_stat_submit_ns.load(std::memory_order_relaxed) / 1.0e6;
    return out;
}

// This backend owns every node the scheduler gives it: fused nodes launch here,
// and each maximal run of ordinary nodes is handed to the CUDA backend as a
// graph view. Submission is asynchronous throughout, so the whole device
// subgraph is one ordered stream with no host round trip inside it.
static enum ggml_status tsg_dsv4_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph)
{
    auto * c = (tsg_dsv4_backend_ctx *) backend->context;
    CUDA_CHECK(cudaSetDevice(c->device));
    cudaStream_t stream = tsg_dsv4_backend_stream(c);
    static const bool stats = []() { const char * e = getenv("TS_DSV4_PERF"); return e && atoi(e) >= 3; }();
    const auto submit_t0 = std::chrono::steady_clock::now();

    int run_start = -1;     // first node of the pending run of ordinary nodes
    bool run_has_work = false;  // ... and whether any of them computes anything
    auto flush = [&](int end) -> enum ggml_status
    {
        const int start = run_start;
        const bool work = run_has_work;
        run_start = -1;
        run_has_work = false;
        // A run of nothing but views and reshapes has no kernels to launch, and
        // handing it to ggml-cuda would still cost a CUDA-graph capture cycle.
        if (start < 0 || !work) return GGML_STATUS_SUCCESS;
        if (stats) tsg_dsv4_stat_views.fetch_add(1, std::memory_order_relaxed);
        ggml_cgraph view = ggml_graph_view(cgraph, start, end);
        return ggml_backend_graph_compute_async(c->cuda_backend, &view);
    };

    for (int i = 0; i < cgraph->n_nodes; i++)
    {
        ggml_tensor * node = cgraph->nodes[i];
        const tsg_dsv4_fused_desc * d = tsg_dsv4_node_desc(node);
        if (!d)
        {
            if (run_start < 0) run_start = i;
            switch (node->op)
            {
                case GGML_OP_NONE:
                case GGML_OP_VIEW:
                case GGML_OP_RESHAPE:
                case GGML_OP_PERMUTE:
                case GGML_OP_TRANSPOSE:
                    break;
                default:
                    run_has_work = true;
                    break;
            }
            continue;
        }
        // A fused node depends on the run before it, so that run has to be
        // submitted first; both go to the same stream, which orders them.
        const enum ggml_status status = flush(i);
        if (status != GGML_STATUS_SUCCESS)
        {
            // Whatever we already launched is still in flight on this stream.
            // Drain it before returning, so the caller's error handling does
            // not race kernels that are still reading the graph's tensors.
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return status;
        }
        if (stats) tsg_dsv4_stat_fused.fetch_add(1, std::memory_order_relaxed);
        if (d->kind == TSG_MATMUL_F32 || d->kind == TSG_MATMUL_ID_F32 ||
            d->kind == TSG_MATMUL_ID_QUANT_STRIP || d->kind == TSG_MATMUL_ID_QUANT_PAIR)
            tsg_matmul_cuda_compute(c->matmul, node);
        else if (d->kind == TSG_MATMUL_Q8_F32)
            tsg_matmul_q8_cuda_compute(node, c->cuda_backend);
        else if (d->kind == TSG_ATTN_F32_PARTIAL || d->kind == TSG_ATTN_F32_FINISH ||
                 d->kind == TSG_ATTN_F32_SOFTMAX || d->kind == TSG_ATTN_MASK_COMPACT)
            tsg_attention_cuda_compute(node, c->cuda_backend);
        else
        {
            tsg_dsv4_fused_launch(d, node, stream);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    const enum ggml_status status = flush(cgraph->n_nodes);
    if (status != GGML_STATUS_SUCCESS) CUDA_CHECK(cudaStreamSynchronize(stream));
    if (stats)
    {
        tsg_dsv4_stat_calls.fetch_add(1, std::memory_order_relaxed);
        tsg_dsv4_stat_nodes.fetch_add((unsigned long long) cgraph->n_nodes, std::memory_order_relaxed);
        tsg_dsv4_stat_submit_ns.fetch_add((unsigned long long)
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - submit_t0).count(), std::memory_order_relaxed);
    }
    return status;
}

static const ggml_backend_i tsg_dsv4_backend_iface = {
    /* .get_name            = */ tsg_dsv4_backend_get_name,
    /* .free                = */ tsg_dsv4_backend_free,
    /* .set_tensor_async    = */ tsg_dsv4_backend_set_tensor_async,
    /* .get_tensor_async    = */ tsg_dsv4_backend_get_tensor_async,
    /* .set_tensor_2d_async = */ nullptr,
    /* .get_tensor_2d_async = */ nullptr,
    /* .cpy_tensor_async    = */ tsg_dsv4_backend_cpy_tensor_async,
    /* .synchronize         = */ tsg_dsv4_backend_synchronize,
    /* .graph_plan_create   = */ nullptr,
    /* .graph_plan_free     = */ nullptr,
    /* .graph_plan_update   = */ nullptr,
    /* .graph_plan_compute  = */ nullptr,
    /* .graph_compute       = */ tsg_dsv4_backend_graph_compute,
    /* .event_record        = */ tsg_dsv4_backend_event_record,
    /* .event_wait          = */ tsg_dsv4_backend_event_wait,
    /* .graph_optimize      = */ tsg_dsv4_backend_graph_optimize,
};

// ---- device iface ----

static const char * tsg_dsv4_dev_get_name(ggml_backend_dev_t dev)
{
    return ((tsg_dsv4_backend_ctx *) dev->context)->name;
}

static const char * tsg_dsv4_dev_get_description(ggml_backend_dev_t dev)
{
    return ((tsg_dsv4_backend_ctx *) dev->context)->desc;
}

static void tsg_dsv4_dev_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total)
{
    auto * c = (tsg_dsv4_backend_ctx *) dev->context;
    ggml_backend_dev_memory(c->cuda_dev, free, total);
}

static enum ggml_backend_dev_type tsg_dsv4_dev_get_type(ggml_backend_dev_t dev)
{
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void tsg_dsv4_dev_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props)
{
    memset(props, 0, sizeof(*props));
    props->name = tsg_dsv4_dev_get_name(dev);
    props->description = tsg_dsv4_dev_get_description(dev);
    props->type = tsg_dsv4_dev_get_type(dev);
    tsg_dsv4_dev_get_memory(dev, &props->memory_free, &props->memory_total);
}

static ggml_backend_t tsg_dsv4_dev_init_backend(ggml_backend_dev_t dev, const char * params)
{
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return nullptr; // instances are created via tsg_dsv4_fused_backend_init
}

static ggml_backend_buffer_type_t tsg_dsv4_dev_get_buffer_type(ggml_backend_dev_t dev)
{
    return ((tsg_dsv4_backend_ctx *) dev->context)->cuda_buft;
}

static bool tsg_dsv4_dev_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op)
{
    auto * c = (tsg_dsv4_backend_ctx *) dev->context;
    if (tsg_dsv4_node_desc(op) != nullptr) return true;
    // Claiming the CUDA device's ops too is what keeps this device's subgraph in
    // ONE scheduler split. Registering both backends instead splits the graph at
    // every alternation between them -- roughly 14 times per DeepSeek V4.1 layer.
    return ggml_backend_dev_supports_op(c->cuda_dev, op);
}

static bool tsg_dsv4_dev_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft)
{
    auto * c = (tsg_dsv4_backend_ctx *) dev->context;
    return buft == c->cuda_buft || ggml_backend_dev_supports_buft(c->cuda_dev, buft);
}

// Events belong to the paired CUDA device, which is also where they are
// recorded and waited on, so they behave exactly as an unwrapped CUDA event.
static ggml_backend_event_t tsg_dsv4_dev_event_new(ggml_backend_dev_t dev)
{
    auto * c = (tsg_dsv4_backend_ctx *) dev->context;
    return c->cuda_dev->iface.event_new ? c->cuda_dev->iface.event_new(c->cuda_dev) : nullptr;
}

static void tsg_dsv4_dev_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event)
{
    auto * c = (tsg_dsv4_backend_ctx *) dev->context;
    if (c->cuda_dev->iface.event_free) c->cuda_dev->iface.event_free(c->cuda_dev, event);
}

static void tsg_dsv4_dev_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event)
{
    auto * c = (tsg_dsv4_backend_ctx *) dev->context;
    if (c->cuda_dev->iface.event_synchronize) c->cuda_dev->iface.event_synchronize(c->cuda_dev, event);
}

static bool tsg_dsv4_dev_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op)
{
    auto * c = (tsg_dsv4_backend_ctx *) dev->context;
    return c->cuda_dev->iface.offload_op && c->cuda_dev->iface.offload_op(c->cuda_dev, op);
}

static const ggml_backend_device_i tsg_dsv4_device_iface = {
    /* .get_name             = */ tsg_dsv4_dev_get_name,
    /* .get_description      = */ tsg_dsv4_dev_get_description,
    /* .get_memory           = */ tsg_dsv4_dev_get_memory,
    /* .get_type             = */ tsg_dsv4_dev_get_type,
    /* .get_props            = */ tsg_dsv4_dev_get_props,
    /* .init_backend         = */ tsg_dsv4_dev_init_backend,
    /* .get_buffer_type      = */ tsg_dsv4_dev_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ tsg_dsv4_dev_supports_op,
    /* .supports_buft        = */ tsg_dsv4_dev_supports_buft,
    /* .offload_op           = */ tsg_dsv4_dev_offload_op,
    /* .event_new            = */ tsg_dsv4_dev_event_new,
    /* .event_free           = */ tsg_dsv4_dev_event_free,
    /* .event_synchronize    = */ tsg_dsv4_dev_event_synchronize,
};

static ggml_guid_t tsg_dsv4_backend_guid()
{
    static ggml_guid guid = { 0x7d, 0x54, 0x53, 0x44, 0x53, 0x56, 0x34, 0x46, 0x55, 0x53, 0x45, 0x44, 0x42, 0x4b, 0x4e, 0x44 };
    return &guid;
}

ggml_backend_t tsg_dsv4_fused_backend_init(ggml_backend_t cuda_backend)
{
    if (!cuda_backend || !ggml_backend_is_cuda(cuda_backend))
    {
        return nullptr;
    }

    auto * cc = (ggml_backend_cuda_context *) cuda_backend->context;

    auto * ctx = new tsg_dsv4_backend_ctx();
    ctx->device = cc->device;
    ctx->cuda_backend = cuda_backend;
    ctx->cuda_dev = ggml_backend_get_device(cuda_backend);
    ctx->cuda_buft = ggml_backend_get_default_buffer_type(cuda_backend);
    ctx->matmul = tsg_matmul_cuda_init(cuda_backend);
    snprintf(ctx->name, sizeof(ctx->name), "TSDSV4-%d", ctx->device);
    snprintf(ctx->desc, sizeof(ctx->desc), "TensorSharp DSV4 fused ops (CUDA%d)", ctx->device);

    // One device record per backend instance, not one per CUDA device index: a
    // shared static would be re-pointed at each new model's context, and freeing
    // any one model would leave the others' device record dangling. The record
    // is reached on every supports_op call, so this has to be per instance.
    auto * dev = new ggml_backend_device();
    dev->iface = tsg_dsv4_device_iface;
    dev->reg = nullptr;
    dev->context = ctx;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ tsg_dsv4_backend_guid(),
        /* .iface   = */ tsg_dsv4_backend_iface,
        /* .device  = */ dev,
        /* .context = */ ctx,
    };
    return backend;
}
