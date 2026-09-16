// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_attention_precision.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include <climits>
#include <cstdio>
#include <cstring>
#include <math_constants.h>

namespace {
struct view {
    const char * data;
    int64_t ne[4];
    size_t nb[4];
    int type;
};
view tensor_view(const ggml_tensor * t) {
    view result;
    result.data = static_cast<const char *>(t->data);
    result.type = t->type;
    for (int i = 0; i < 4; ++i) {
        result.ne[i] = t->ne[i];
        result.nb[i] = t->nb[i];
    }
    return result;
}
__device__ __forceinline__ float read(view t, int64_t x, int64_t y = 0, int64_t h = 0, int64_t b = 0) {
    const char * p = t.data + x * t.nb[0] + y * t.nb[1] + h * t.nb[2] + b * t.nb[3];
    if (t.type == GGML_TYPE_F32)
        return *reinterpret_cast<const float *>(p);
    if (t.type == GGML_TYPE_F16)
        return __half2float(*reinterpret_cast<const half *>(p));
    return __uint_as_float(uint32_t(*reinterpret_cast<const uint16_t *>(p)) << 16);
}

// A block owns one query/head and one key partition. Sixteen scores at a
// time share the cached F32 query. Both online softmax and the V weighting
// remain F32; no score matrix or host synchronization is needed.
template <int MAX_VALUE, bool SPARSE>
__global__ void partial_kernel(view q, view k, view v, view mask, view compact, bool masked, float scale, int chunk, int splits,
                               float * output) {
    constexpr int TILE = 16, THREADS = 128;
    extern __shared__ float query_values[];
    __shared__ float scores[TILE], weights[TILE], maximum, denominator, correction;
    __shared__ int64_t selected_keys[TILE];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int64_t row = blockIdx.x, split = row % splits, h = (row / splits) % q.ne[2];
    const int64_t query = (row / (splits * q.ne[2])) % q.ne[1], batch = row / (splits * q.ne[2] * q.ne[1]);
    const int64_t kh = h / (q.ne[2] / k.ne[2]), kb = batch / (q.ne[3] / k.ne[3]);
    const int64_t vh = h / (q.ne[2] / v.ne[2]), vb = batch / (q.ne[3] / v.ne[3]);
    const int32_t * indices = nullptr;
    int64_t count = k.ne[1];
    if constexpr (SPARSE) {
        const int64_t index_row = query + q.ne[1] *
            (h % compact.ne[2] + compact.ne[2] * (batch % compact.ne[3]));
        const auto * candidate = reinterpret_cast<const int32_t *>(compact.data) + index_row * compact.ne[0];
        if (candidate[0] >= 0 && candidate[0] < compact.ne[0]) {
            count = candidate[0];
            indices = candidate + 1;
        }
        chunk = int(max(int64_t(16), ((count + splits - 1) / splits + 15) / 16 * 16));
    }
    const int64_t begin = split * chunk, end = min(count, begin + chunk);
    float values[(MAX_VALUE + THREADS - 1) / THREADS] = {};
    for (int x = tid; x < q.ne[0]; x += THREADS)
        query_values[x] = read(q, x, query, h, batch);
    if (tid == 0) {
        maximum = -CUDART_INF_F;
        denominator = 0;
    }
    __syncthreads();
    for (int64_t first = begin; first < end; first += TILE) {
        for (int j = warp; j < TILE; j += 4) {
            const int64_t entry = first + j;
            const int64_t key = entry < end && indices ? indices[entry] : entry;
            const float bias =
                entry < end ? (masked ? read(mask, key, query, h % mask.ne[2], batch % mask.ne[3]) : 0) : -CUDART_INF_F;
            float dot = 0;
            if (bias != -CUDART_INF_F) {
                for (int x = 2 * lane; x < q.ne[0]; x += 64) {
                    dot = fmaf(query_values[x], read(k, x, key, kh, kb), dot);
                    if (x + 1 < q.ne[0])
                        dot = fmaf(query_values[x + 1], read(k, x + 1, key, kh, kb), dot);
                }
                for (int shift = 16; shift; shift /= 2)
                    dot += __shfl_down_sync(0xffffffff, dot, shift);
            }
            if (lane == 0) {
                if constexpr (SPARSE) selected_keys[j] = key;
                scores[j] = bias == -CUDART_INF_F ? -CUDART_INF_F : __fmul_rn(dot, scale) + bias;
            }
        }
        __syncthreads();
        if (tid == 0) {
            float next = maximum;
            for (int j = 0; j < TILE; ++j)
                next = fmaxf(next, scores[j]);
            correction = next == -CUDART_INF_F ? 1 : expf(maximum - next);
            float sum = denominator * correction;
            for (int j = 0; j < TILE; ++j) {
                weights[j] = scores[j] == -CUDART_INF_F ? 0 : expf(scores[j] - next);
                sum += weights[j];
            }
            maximum = next;
            denominator = sum;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < (MAX_VALUE + THREADS - 1) / THREADS; ++i) {
            const int x = tid + i * THREADS;
            if (x < v.ne[0]) {
                float sum = values[i] * correction;
#pragma unroll
                for (int j = 0; j < TILE; ++j)
                    if (weights[j] != 0)
                        sum = fmaf(weights[j], read(v, x, SPARSE ? selected_keys[j] : first + j, vh, vb), sum);
                values[i] = sum;
            }
        }
        __syncthreads();
    }
    float * result = output + row * (v.ne[0] + 2);
#pragma unroll
    for (int i = 0; i < (MAX_VALUE + THREADS - 1) / THREADS; ++i) {
        const int x = tid + i * THREADS;
        if (x < v.ne[0])
            result[x] = values[i];
    }
    if (tid == 0) {
        result[v.ne[0]] = maximum;
        result[v.ne[0] + 1] = denominator;
    }
}

// A stable, block-local prefix sum emits selected positions in key order.
// Count every visible entry even after capacity is exhausted; consumers then
// take the full-mask fallback for that row. No host readback is required.
__global__ void compact_kernel(view mask, int keys, int queries, int capacity, int32_t * output) {
    __shared__ int count, exceptional, warp_counts[8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int64_t row = blockIdx.x, query = row % queries;
    const int64_t h = row / queries % mask.ne[2], batch = row / (queries * mask.ne[2]);
    int32_t * result = output + row * (int64_t(capacity) + 1);
    if (tid == 0) { count = 0; exceptional = 0; }
    __syncthreads();
    for (int64_t first = 0; first < keys; first += 256) {
        const int64_t key = first + tid;
        const float bias = key < keys ? read(mask, key, query, h, batch) : -CUDART_INF_F;
        const bool selected = bias != -CUDART_INF_F;
        if (selected && !isfinite(bias)) atomicExch(&exceptional, 1);
        const unsigned bits = __ballot_sync(0xffffffff, selected);
        const unsigned before = lane == 0 ? 0 : (0xffffffffu >> (32 - lane));
        int offset = __popc(bits & before);
        if (lane == 0) warp_counts[warp] = __popc(bits);
        __syncthreads();
        for (int i = 0; i < warp; ++i) offset += warp_counts[i];
        offset += count;
        if (selected && offset < capacity) result[offset + 1] = int(key);
        __syncthreads();
        if (tid == 0)
            for (int i = 0; i < 8; ++i) count += warp_counts[i];
        __syncthreads();
    }
    if (tid == 0) result[0] = exceptional ? -1 : count;
}

__global__ void finish_kernel(const float * partial, view sinks, bool has_sinks, int width, int heads, int splits,
                              float * output) {
    __shared__ float scales[16], reciprocal;
    const int64_t row = blockIdx.x;
    const float * p = partial + row * splits * (width + 2);
    if (threadIdx.x == 0) {
        const float sink = has_sinks ? read(sinks, row % heads) : -CUDART_INF_F;
        float maximum = sink;
        for (int s = 0; s < splits; ++s)
            maximum = fmaxf(maximum, p[s * (width + 2) + width]);
        float sum = has_sinks && isfinite(maximum) ? expf(sink - maximum) : 0;
        for (int s = 0; s < splits; ++s) {
            scales[s] = isfinite(maximum) ? expf(p[s * (width + 2) + width] - maximum) : 0;
            sum = fmaf(p[s * (width + 2) + width + 1], scales[s], sum);
        }
        reciprocal = sum > 0 ? 1.0f / sum : 0;
    }
    __syncthreads();
    for (int x = threadIdx.x; x < width; x += blockDim.x) {
        float value = 0;
        for (int s = 0; s < splits; ++s)
            value = fmaf(p[s * (width + 2) + x], scales[s], value);
        output[row * width + x] = value * reciprocal;
    }
}

__global__ void softmax_kernel(view scores, view mask, view sinks, bool masked, bool has_sinks, float scale,
                               float * output) {
    constexpr int THREADS = 256;
    __shared__ float reduction[THREADS];
    const int tid = threadIdx.x;
    const int64_t row = blockIdx.x, query = row % scores.ne[1];
    const int64_t head = (row / scores.ne[1]) % scores.ne[2];
    const int64_t batch = row / (scores.ne[1] * scores.ne[2]);
    float * result = output + row * scores.ne[0];
    float maximum = has_sinks ? read(sinks, head) : -CUDART_INF_F;
    for (int64_t key = tid; key < scores.ne[0]; key += THREADS) {
        const float bias = masked ? read(mask, key, query, head % mask.ne[2], batch % mask.ne[3]) : 0;
        const float value = __fmul_rn(read(scores, key, query, head, batch), scale) + bias;
        result[key] = value;
        maximum = fmaxf(maximum, value);
    }
    reduction[tid] = maximum;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
        if (tid < stride)
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
        __syncthreads();
    }
    maximum = reduction[0];
    float sum = tid == 0 && has_sinks && isfinite(maximum) ? expf(read(sinks, head) - maximum) : 0;
    for (int64_t key = tid; key < scores.ne[0]; key += THREADS) {
        const float value = isfinite(maximum) ? expf(result[key] - maximum) : 0;
        result[key] = value;
        sum += value;
    }
    __syncthreads();
    reduction[tid] = sum;
    __syncthreads();
    for (int stride = THREADS / 2; stride; stride /= 2) {
        if (tid < stride)
            reduction[tid] += reduction[tid + stride];
        __syncthreads();
    }
    const float reciprocal = reduction[0] > 0 ? 1.0f / reduction[0] : 0;
    for (int64_t key = tid; key < scores.ne[0]; key += THREADS)
        result[key] *= reciprocal;
}
} // namespace

void tsg_attention_cuda_compute(ggml_tensor * dst, ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_cuda_context *>(backend->context);
    const cudaStream_t stream = context->stream(context->device, 0);
    ggml_custom_op_params params;
    std::memcpy(&params, dst->op_params, sizeof(params));
    const auto & desc = *static_cast<const tsg_dsv4_fused_desc *>(params.userdata);
    CUDA_CHECK(cudaSetDevice(context->device));
    if (desc.kind == TSG_ATTN_MASK_COMPACT) {
        const int64_t rows = ggml_nelements(dst) / dst->ne[0];
        GGML_ASSERT(rows <= INT_MAX);
        compact_kernel<<<unsigned(rows), 256, 0, stream>>>(tensor_view(dst->src[0]), desc.i0,
            int(dst->ne[1]), int(dst->ne[0] - 1), static_cast<int32_t *>(dst->data));
    } else if (desc.kind == TSG_ATTN_F32_SOFTMAX) {
        const int64_t rows = ggml_nelements(dst) / dst->ne[0];
        GGML_ASSERT(rows <= INT_MAX);
        softmax_kernel<<<unsigned(rows), 256, 0, stream>>>(tensor_view(dst->src[0]), tensor_view(dst->src[1]),
                                                           tensor_view(dst->src[2]), desc.i0 != 0, desc.i1 != 0,
                                                           desc.f0, static_cast<float *>(dst->data));
    } else if (desc.kind == TSG_ATTN_F32_PARTIAL) {
        const auto q = tensor_view(dst->src[0]), k = tensor_view(dst->src[1]), v = tensor_view(dst->src[2]);
        const auto mask = tensor_view(dst->src[3]);
        const auto compact = dst->src[4] ? tensor_view(dst->src[4]) : view{};
        const int64_t rows = dst->ne[1] * dst->ne[2] * dst->ne[3];
        GGML_ASSERT(rows <= INT_MAX);
        const unsigned blocks = unsigned(rows);
        const size_t shared = q.ne[0] * sizeof(float);
        auto * out = static_cast<float *>(dst->data);
#define TSG_LAUNCH_ATTN(W)                                                                                             \
    if (compact.data)                                                                                                  \
        partial_kernel<W, true><<<blocks, 128, shared, stream>>>(q, k, v, mask, compact, desc.i0 != 0, desc.f0, desc.i2, int(dst->ne[1]), out); \
    else                                                                                                              \
        partial_kernel<W, false><<<blocks, 128, shared, stream>>>(q, k, v, mask, compact, desc.i0 != 0, desc.f0, desc.i2, int(dst->ne[1]), out)
        if (v.ne[0] <= 128) {
            TSG_LAUNCH_ATTN(128);
        } else if (v.ne[0] <= 256) {
            TSG_LAUNCH_ATTN(256);
        } else if (v.ne[0] <= 512) {
            TSG_LAUNCH_ATTN(512);
        } else if (v.ne[0] <= 1024) {
            TSG_LAUNCH_ATTN(1024);
        } else {
            TSG_LAUNCH_ATTN(2048);
        }
#undef TSG_LAUNCH_ATTN
    } else {
        GGML_ASSERT(desc.kind == TSG_ATTN_F32_FINISH);
        const int64_t rows = ggml_nelements(dst) / dst->ne[0];
        GGML_ASSERT(rows <= INT_MAX && dst->src[0]->ne[1] <= 16);
        finish_kernel<<<unsigned(rows), 128, 0, stream>>>(
            static_cast<const float *>(dst->src[0]->data), tensor_view(dst->src[1]), desc.i1 != 0, int(dst->ne[0]),
            int(dst->ne[1]), int(dst->src[0]->ne[1]), static_cast<float *>(dst->data));
    }
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
        // Preserve the failing owned operation's geometry even after CUDA has
        // entered an error state and pointer-attribute queries are unavailable.
        std::fprintf(stderr, "TensorSharp F32 attention failure: kind=%d device=%d flags=%d,%d,%d scale=%g\n",
                     int(desc.kind), context->device, desc.i0, desc.i1, desc.i2, double(desc.f0));
        for (int i = -1; i < GGML_MAX_SRC; ++i) {
            const ggml_tensor * tensor = i < 0 ? dst : dst->src[i];
            if (!tensor) continue;
            std::fprintf(stderr,
                "  %s%d name=%s type=%d data=%p ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] view=%p offset=%zu buffer=%s\n",
                i < 0 ? "dst" : "src", i < 0 ? 0 : i, tensor->name, int(tensor->type), tensor->data,
                (long long)tensor->ne[0], (long long)tensor->ne[1], (long long)tensor->ne[2], (long long)tensor->ne[3],
                tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3],
                tensor->view_src ? tensor->view_src->data : nullptr, tensor->view_offs,
                tensor->buffer ? ggml_backend_buffer_name(tensor->buffer) : "none");
        }
        std::fflush(stderr);
    }
    CUDA_CHECK(launch_status);
}
