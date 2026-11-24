// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_attention_precision.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda/common.cuh"
#include <cublasLt.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <mutex>
#include <unordered_map>
#include <cstdint>
#include <cstdlib>
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

// One thread owns one query vector, while the block shares each K/V tile.
// All operands, dot products, online-softmax state and weighted sums stay F32.
// Unlike upstream's generic 72-wide flash kernel, this never rounds Q/K/V or
// accumulators to half; unlike decomposed attention, scratch is O(N * D).
template <int WIDTH>
__global__ __launch_bounds__(128) void vision_kernel(view q, view k, view v, float scale, float * output) {
    constexpr int QUERIES = 128, KEYS = 16;
    __shared__ float keys[KEYS * WIDTH], values[KEYS * WIDTH];
    // Query-major register arrays for all scores add register pressure and make
    // the compiler emit an enormous unrolled instruction stream. Keep this
    // small score tile key-major in shared memory: adjacent query threads use
    // adjacent banks, and the two vector loops below keep their independent
    // per-component accumulators in registers.
    __shared__ float probabilities[KEYS * QUERIES];
    const int head = blockIdx.y, batch = blockIdx.z;
    const int64_t query = int64_t(blockIdx.x) * QUERIES + threadIdx.x;
    const bool active = query < q.ne[1];
    float query_values[WIDTH], accumulators[WIDTH];
#pragma unroll
    for (int x = 0; x < WIDTH; ++x) {
        query_values[x] = active ? read(q, x, query, head, batch) : 0;
        accumulators[x] = 0;
    }
    float maximum = -CUDART_INF_F, denominator = 0;
    for (int64_t first = 0; first < k.ne[1]; first += KEYS) {
        const int count = int(min(int64_t(KEYS), k.ne[1] - first));
        for (int i = threadIdx.x; i < count * WIDTH; i += QUERIES) {
            keys[i] = read(k, i % WIDTH, first + i / WIDTH, head, batch);
            values[i] = read(v, i % WIDTH, first + i / WIDTH, head, batch);
        }
        __syncthreads();
        if (active) {
            float next = maximum;
#pragma unroll 1
            for (int j = 0; j < KEYS; ++j) {
                float dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;
                if (j < count) {
#pragma unroll
                    for (int x = 0; x < WIDTH; x += 4) {
                        dot0 = fmaf(query_values[x], keys[j * WIDTH + x], dot0);
                        dot1 = fmaf(query_values[x + 1], keys[j * WIDTH + x + 1], dot1);
                        dot2 = fmaf(query_values[x + 2], keys[j * WIDTH + x + 2], dot2);
                        dot3 = fmaf(query_values[x + 3], keys[j * WIDTH + x + 3], dot3);
                    }
                }
                const float dot = (dot0 + dot1) + (dot2 + dot3);
                const float score = j < count ? __fmul_rn(dot, scale) : -CUDART_INF_F;
                probabilities[j * QUERIES + threadIdx.x] = score;
                next = fmaxf(next, score);
            }
            const float correction = expf(maximum - next);
            denominator *= correction;
#pragma unroll 1
            for (int j = 0; j < KEYS; ++j) {
                const float probability = j < count ? expf(probabilities[j * QUERIES + threadIdx.x] - next) : 0;
                probabilities[j * QUERIES + threadIdx.x] = probability;
                denominator += probability;
            }
#pragma unroll
            for (int x = 0; x < WIDTH; ++x) accumulators[x] *= correction;
#pragma unroll 1
            for (int j = 0; j < count; ++j) {
                const float probability = probabilities[j * QUERIES + threadIdx.x];
#pragma unroll
                for (int x = 0; x < WIDTH; ++x)
                    accumulators[x] = fmaf(probability, values[j * WIDTH + x], accumulators[x]);
            }
            maximum = next;
        }
        // An inactive tail query participates in both barriers; no tile can be
        // overwritten until all active queries have consumed it.
        __syncthreads();
    }
    if (active) {
        const int64_t row = head + q.ne[2] * (query + q.ne[1] * batch);
#pragma unroll
        for (int x = 0; x < WIDTH; ++x)
            output[row * WIDTH + x] = accumulators[x] / denominator;
    }
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

// Decode-class attention whose heads all share one F16 K=V head of width 512:
// DeepSeek V4.1's MLA decode and DSpark verify. partial_kernel gives every head
// its own block, so each one re-reads the same keys (64x the K/V traffic) and
// serializes softmax on one thread. Here a block owns one query, one key
// partition and four heads: each K/V tile is loaded into shared memory once for
// all four, and only the partition's visible keys are loaded at all.
//
// A decode step sees few visible keys, packed into few partitions, so a block
// also splits its own keys over four streams of four warps (one per head):
// stream s takes every fourth 16-key tile of the visible list, keeps its own
// online-softmax state, and the streams merge in stream order at the end. A
// key is scored by a lane pair, each half over 256 dimensions with four
// interleaved F32 accumulators combined in a fixed order; softmax and the V sum
// stay F32. Partition, tile/stream assignment and every reduction order depend
// only on the key extent and the query's own mask, never on the launch width,
// so verify rows equal decode rows bit for bit. Output rows use
// partial_kernel's layout; finish_kernel is shared.
constexpr int MQA_WIDTH = 512, MQA_HEADS = 4, MQA_STREAMS = 4, MQA_TILE = 16, MQA_WINDOW = 128;
constexpr int MQA_STREAM_THREADS = MQA_HEADS * 32, MQA_THREADS = MQA_STREAMS * MQA_STREAM_THREADS;
constexpr int MQA_ROW_BYTES = MQA_WIDTH * 2 + 16;   // 1040: 16-byte loads hit distinct banks
constexpr size_t MQA_TILE_BYTES = size_t(MQA_TILE) * MQA_ROW_BYTES;
constexpr size_t MQA_SHARED = size_t(MQA_HEADS) * MQA_WIDTH * sizeof(float) + MQA_STREAMS * MQA_TILE_BYTES;
static_assert(MQA_TILE_BYTES >= size_t(MQA_HEADS) * (MQA_WIDTH + 2) * sizeof(float), "stream state must fit its tile");

__device__ __forceinline__ void mqa_stream_barrier(int stream) {
    asm volatile("bar.sync %0, %1;" :: "r"(stream + 1), "r"(MQA_STREAM_THREADS) : "memory");
}

// SPARSE: the keys are each query's compacted visible list (TSG_ATTN_MASK_COMPACT),
// as partial_kernel's sparse prefill reads them; an overflowing or exceptional
// row takes the full key range instead, exactly like partial_kernel.
template <bool SPARSE>
__global__ __launch_bounds__(MQA_THREADS) void mqa_partial_kernel(view q, view kv, view mask, view compact, bool masked,
                                                                  float scale, int chunk, int splits, float * output) {
    extern __shared__ __align__(16) unsigned char mqa_shared[];
    float * qs = reinterpret_cast<float *>(mqa_shared);
    __shared__ int keys[MQA_WINDOW];
    __shared__ float biases[MQA_WINDOW];
    __shared__ int warp_counts[MQA_WINDOW / 32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head_slot = warp % MQA_HEADS, stream = warp / MQA_HEADS, stream_tid = tid % MQA_STREAM_THREADS;
    unsigned char * tile = mqa_shared + size_t(MQA_HEADS) * MQA_WIDTH * sizeof(float) + stream * MQA_TILE_BYTES;
    const int key_lane = lane & (MQA_TILE - 1), half_index = lane / MQA_TILE;
    const int64_t split = blockIdx.x, h = int64_t(blockIdx.y) * MQA_HEADS + head_slot;
    const int64_t query = blockIdx.z % q.ne[1], batch = blockIdx.z / q.ne[1];
    const int64_t kb = batch / (q.ne[3] / kv.ne[3]);
    const char * base = kv.data + kb * kv.nb[3];
    for (int i = tid; i < MQA_HEADS * MQA_WIDTH; i += MQA_THREADS)
        qs[i] = read(q, i % MQA_WIDTH, query, int64_t(blockIdx.y) * MQA_HEADS + i / MQA_WIDTH, batch);
    const int32_t * list = nullptr;
    int64_t extent = kv.ne[1];
    if constexpr (SPARSE) {
        const auto * candidate = reinterpret_cast<const int32_t *>(compact.data) +
            (query + q.ne[1] * (batch % compact.ne[3])) * compact.ne[0];
        if (candidate[0] >= 0 && candidate[0] < compact.ne[0]) {
            extent = candidate[0];
            list = candidate + 1;
            chunk = int(max(int64_t(16), ((extent + splits - 1) / splits + 15) / 16 * 16));
        }
    }
    // Dense: a key range. Sparse: a range of positions in the query's list.
    const int64_t begin = split * chunk, end = min(extent, begin + chunk);
    float maximum = -CUDART_INF_F, denominator = 0;
    float values[16] = {};
    int tile_index = 0;   // across windows, so stream assignment follows the visible-key order
    for (int64_t window = begin; window < end; window += MQA_WINDOW) {
        __syncthreads();   // the previous window's list is no longer read
        if (warp < MQA_WINDOW / 32) {
            const int64_t entry = window + tid;
            const int64_t key = entry < end && list ? int64_t(list[entry]) : entry;
            const float bias = entry < end ? (masked ? read(mask, key, query, 0, batch % mask.ne[3]) : 0) : -CUDART_INF_F;
            const bool visible = bias != -CUDART_INF_F;
            const unsigned bits = __ballot_sync(0xffffffffu, visible);
            if (lane == 0) warp_counts[warp] = __popc(bits);
            // Each warp stores its own run after the counts are visible.
            asm volatile("bar.sync %0, %1;" :: "r"(MQA_STREAMS + 1), "r"(MQA_WINDOW) : "memory");
            int offset = __popc(bits & ((1u << lane) - 1u));
            for (int i = 0; i < warp; ++i) offset += warp_counts[i];
            if (visible) { keys[offset] = int(key); biases[offset] = bias; }
        }
        __syncthreads();
        int count = 0;
        for (int i = 0; i < MQA_WINDOW / 32; ++i) count += warp_counts[i];
        const int tiles = (count + MQA_TILE - 1) / MQA_TILE;
        for (int t = 0; t < tiles; ++t, ++tile_index) {
            if (tile_index % MQA_STREAMS != stream) continue;
            const int first = t * MQA_TILE, n = min(MQA_TILE, count - first);
            // 64 uint4 per 1 KiB row; the stream's 128 threads fill two rows per pass.
            for (int i = stream_tid; i < MQA_TILE * (MQA_WIDTH / 8); i += MQA_STREAM_THREADS) {
                const int r = i / (MQA_WIDTH / 8), c = i % (MQA_WIDTH / 8);
                uint4 chunk_data = make_uint4(0, 0, 0, 0);
                if (r < n) chunk_data = *reinterpret_cast<const uint4 *>(base + int64_t(keys[first + r]) * kv.nb[1] + c * 16);
                *reinterpret_cast<uint4 *>(tile + r * MQA_ROW_BYTES + c * 16) = chunk_data;
            }
            mqa_stream_barrier(stream);
            float score = -CUDART_INF_F;
            {
                // Lane pair (j, j + 16) scores key j: halves [0, 256) and [256, 512).
                const float * qh = qs + head_slot * MQA_WIDTH + half_index * (MQA_WIDTH / 2);
                const unsigned char * row = tile + key_lane * MQA_ROW_BYTES + half_index * MQA_WIDTH;
                float acc[4] = {};
#pragma unroll 4
                for (int c = 0; c < MQA_WIDTH / 16; ++c) {
                    const uint4 packed = *reinterpret_cast<const uint4 *>(row + c * 16);
                    const half2 * k2 = reinterpret_cast<const half2 *>(&packed);
                    const float4 qa = *reinterpret_cast<const float4 *>(qh + c * 8);
                    const float4 qb = *reinterpret_cast<const float4 *>(qh + c * 8 + 4);
                    const float2 k0 = __half22float2(k2[0]), k1 = __half22float2(k2[1]);
                    const float2 k2v = __half22float2(k2[2]), k3 = __half22float2(k2[3]);
                    acc[0] = fmaf(qa.x, k0.x, acc[0]); acc[0] = fmaf(qa.y, k0.y, acc[0]);
                    acc[1] = fmaf(qa.z, k1.x, acc[1]); acc[1] = fmaf(qa.w, k1.y, acc[1]);
                    acc[2] = fmaf(qb.x, k2v.x, acc[2]); acc[2] = fmaf(qb.y, k2v.y, acc[2]);
                    acc[3] = fmaf(qb.z, k3.x, acc[3]); acc[3] = fmaf(qb.w, k3.y, acc[3]);
                }
                const float part = (acc[0] + acc[1]) + (acc[2] + acc[3]);
                const float other = __shfl_xor_sync(0xffffffffu, part, MQA_TILE);
                const float dot = half_index == 0 ? part + other : other + part;
                if (key_lane < n) score = __fmul_rn(dot, scale) + biases[first + key_lane];
            }
            float tile_max = score;
            for (int shift = MQA_TILE / 2; shift; shift /= 2) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, shift));
            const float next = fmaxf(maximum, tile_max);
            const float correction = next == -CUDART_INF_F ? 1 : expf(maximum - next);
            const float weight = score == -CUDART_INF_F ? 0 : expf(score - next);
            float tile_sum = weight;
            for (int shift = MQA_TILE / 2; shift; shift /= 2) tile_sum += __shfl_xor_sync(0xffffffffu, tile_sum, shift);
            denominator = fmaf(denominator, correction, tile_sum);
            maximum = next;
#pragma unroll
            for (int i = 0; i < 16; ++i) values[i] *= correction;
            for (int j = 0; j < n; ++j) {
                const float w = __shfl_sync(0xffffffffu, weight, j);
                const unsigned char * row = tile + j * MQA_ROW_BYTES;
#pragma unroll
                for (int part = 0; part < 2; ++part) {
                    const uint4 packed = *reinterpret_cast<const uint4 *>(row + (lane + 32 * part) * 16);
                    const half2 * v2 = reinterpret_cast<const half2 *>(&packed);
#pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const float2 v = __half22float2(v2[p]);
                        values[part * 8 + 2 * p] = fmaf(w, v.x, values[part * 8 + 2 * p]);
                        values[part * 8 + 2 * p + 1] = fmaf(w, v.y, values[part * 8 + 2 * p + 1]);
                    }
                }
            }
            mqa_stream_barrier(stream);   // the stream's next tile overwrites its rows
        }
    }
    // Merge the streams in stream order. Each stream parks its state in its
    // own (now idle) tile buffer; stream 0 combines and writes the partial row.
    __syncthreads();
    float * state = reinterpret_cast<float *>(tile) + head_slot * (MQA_WIDTH + 2);
#pragma unroll
    for (int part = 0; part < 2; ++part)
#pragma unroll
        for (int p = 0; p < 8; ++p) state[(lane + 32 * part) * 8 + p] = values[part * 8 + p];
    if (lane == 0) { state[MQA_WIDTH] = maximum; state[MQA_WIDTH + 1] = denominator; }
    __syncthreads();
    if (stream != 0) return;
    const unsigned char * tiles_base = mqa_shared + size_t(MQA_HEADS) * MQA_WIDTH * sizeof(float);
    float merged_max = -CUDART_INF_F;
    for (int s2 = 0; s2 < MQA_STREAMS; ++s2)
        merged_max = fmaxf(merged_max, reinterpret_cast<const float *>(tiles_base + s2 * MQA_TILE_BYTES)[head_slot * (MQA_WIDTH + 2) + MQA_WIDTH]);
    float merged_den = 0, merged[16] = {};
    for (int s2 = 0; s2 < MQA_STREAMS; ++s2) {
        const float * other = reinterpret_cast<const float *>(tiles_base + s2 * MQA_TILE_BYTES) + head_slot * (MQA_WIDTH + 2);
        const float factor = other[MQA_WIDTH] == -CUDART_INF_F ? 0 : expf(other[MQA_WIDTH] - merged_max);
        merged_den = fmaf(other[MQA_WIDTH + 1], factor, merged_den);
#pragma unroll
        for (int part = 0; part < 2; ++part)
#pragma unroll
            for (int p = 0; p < 8; ++p)
                merged[part * 8 + p] = fmaf(other[(lane + 32 * part) * 8 + p], factor, merged[part * 8 + p]);
    }
    float * result = output + (split + splits * (h + q.ne[2] * (query + q.ne[1] * batch))) * (MQA_WIDTH + 2);
#pragma unroll
    for (int part = 0; part < 2; ++part)
#pragma unroll
        for (int p = 0; p < 8; ++p) result[(lane + 32 * part) * 8 + p] = merged[part * 8 + p];
    if (lane == 0) {
        result[MQA_WIDTH] = merged_max;
        result[MQA_WIDTH + 1] = merged_den;
    }
}

// DeepSeek V4.1's MLA shape: every head shares one F16 K=V head of width 512.
bool mqa_shape(const view & q, const view & k, const view & v) {
    return k.data == v.data && k.type == GGML_TYPE_F16 && k.ne[0] == MQA_WIDTH && v.ne[0] == MQA_WIDTH &&
        q.ne[0] == MQA_WIDTH && k.ne[2] == 1 && v.ne[2] == 1;
}

bool mqa_enabled() {
    // TS_DSV41_MQA_DECODE=0 keeps partial_kernel for A/B and diagnosis.
    static const bool enabled = [] { const char * e = std::getenv("TS_DSV41_MQA_DECODE"); return !e || std::atoi(e) != 0; }();
    return enabled;
}

// The MQA kernel's preconditions; anything else keeps partial_kernel.
bool mqa_eligible(const view & q, const view & k, const view & v, const view & mask, const view & compact, bool masked) {
    return mqa_enabled() && mqa_shape(q, k, v) && k.nb[0] == 2 && k.nb[1] % 16 == 0 &&
        reinterpret_cast<uintptr_t>(k.data) % 16 == 0 && q.ne[2] % MQA_HEADS == 0 && q.ne[2] / MQA_HEADS <= 65535 &&
        q.ne[3] % k.ne[3] == 0 && q.ne[1] * q.ne[3] <= 65535 && (!masked || mask.ne[2] == 1) &&
        (!compact.data || (masked && compact.ne[2] == 1 && compact.ne[1] == q.ne[1]));
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
    __shared__ float scales[64], reciprocal;
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

// m16n8k16 tensor-core MMA, F16 operands, F32 accumulators (sm_80+).
__device__ __forceinline__ void mma_f16_16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

__device__ __forceinline__ uint32_t pack_half2(float low, float high) {
    const half2 value = __floats2half2_rn(low, high);
    return *reinterpret_cast<const uint32_t *>(&value);
}

__device__ __forceinline__ uint32_t load_half2(const half * p) {
    return *reinterpret_cast<const uint32_t *>(p);
}

// Copy 16 bytes global -> shared without staging through registers (sm_80+).
__device__ __forceinline__ void cp_async_16(void * shared, const void * global) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(address), "l"(global));
#else
    *static_cast<uint4 *>(shared) = *static_cast<const uint4 *>(global);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
#endif
}

// B fragments of two 8x8 F16 tiles stored row-major (rows = the MMA's k), transposed
// on the way into registers.
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t & b0, uint32_t & b1, const half * row) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(row));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1) : "r"(address));
#else
    b0 = b1 = 0;
    __trap();
#endif
}

// Tensor-core flash attention for the vision towers' 64/72-wide heads, without
// masks. Q arrives F32 and is rounded to F16 here; K and V arrive as F16 (ggml
// casts them once per layer) - the same 10-bit mantissa the TF32 GEMMs of the
// decomposed path round all three to. Scores, the online softmax and the output
// accumulate in F32. One block owns 128 queries of one head (eight warps of 16
// rows) and copies 64-key K/V tiles to shared memory asynchronously; the quadratic
// score matrix is never materialized. Head widths pad to a multiple of 16 with zeros.
template <int WIDTH>
__global__ __launch_bounds__(256) void vision_mma_kernel(view q, view k, view v, float scale, float * output) {
    constexpr int PADDED = (WIDTH + 15) / 16 * 16, WARPS = 8, QUERIES = WARPS * 16, KEYS = 64;
    constexpr int CHUNKS = WIDTH / 8;     // 16-byte chunks per F16 key row
    // Row stride padded so fragment and ldmatrix accesses of a warp hit distinct banks.
    constexpr int ROW = PADDED + 8;
    static_assert(WIDTH % 8 == 0, "K/V rows are copied in 16-byte chunks");
    __shared__ __align__(16) half queries[QUERIES * ROW];
    __shared__ __align__(16) half keys[KEYS * ROW];
    __shared__ __align__(16) half values[KEYS * ROW];
    const int head = blockIdx.y, batch = blockIdx.z, tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31, group = lane >> 2, quad = lane & 3;
    const int64_t first_query = int64_t(blockIdx.x) * QUERIES, nq = q.ne[1], nk = k.ne[1];
    const char * key_base = k.data + head * k.nb[2] + batch * k.nb[3];
    const char * value_base = v.data + head * v.nb[2] + batch * v.nb[3];

    for (int i = tid; i < QUERIES * PADDED; i += WARPS * 32) {
        const int r = i / PADDED, x = i % PADDED;
        const int64_t query = first_query + r;
        queries[r * ROW + x] = __float2half(x < WIDTH && query < nq ? read(q, x, query, head, batch) : 0.0f);
    }
    // Padding columns are never copied over: zero them once.
    if constexpr (PADDED > WIDTH) {
        for (int i = tid; i < KEYS * (PADDED - WIDTH); i += WARPS * 32) {
            const int r = i / (PADDED - WIDTH), x = WIDTH + i % (PADDED - WIDTH);
            keys[r * ROW + x] = values[r * ROW + x] = __float2half(0.0f);
        }
    }
    __syncthreads();
    uint32_t qa[PADDED / 16][4];
#pragma unroll
    for (int kk = 0; kk < PADDED / 16; ++kk) {
        const half * base = queries + (warp * 16) * ROW + kk * 16 + quad * 2;
        qa[kk][0] = load_half2(base + group * ROW);
        qa[kk][1] = load_half2(base + (group + 8) * ROW);
        qa[kk][2] = load_half2(base + group * ROW + 8);
        qa[kk][3] = load_half2(base + (group + 8) * ROW + 8);
    }

    // Rows `group` and `group + 8` of this warp's 16; scores live in the log2 domain.
    float acc[PADDED / 8][4];
#pragma unroll
    for (int n = 0; n < PADDED / 8; ++n)
        acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    float max0 = -CUDART_INF_F, max1 = -CUDART_INF_F, sum0 = 0.0f, sum1 = 0.0f;
    const float log2_scale = scale * 1.4426950408889634f;

    for (int64_t first_key = 0; first_key < nk; first_key += KEYS) {
        __syncthreads();
        for (int i = tid; i < KEYS * CHUNKS; i += WARPS * 32) {
            const int r = i / CHUNKS, c = i % CHUNKS;
            const int64_t key = first_key + r;
            half * kd = keys + r * ROW + c * 8;
            half * vd = values + r * ROW + c * 8;
            if (key < nk) {
                cp_async_16(kd, key_base + key * k.nb[1] + c * 16);
                cp_async_16(vd, value_base + key * v.nb[1] + c * 16);
            } else {
                // Past the last key: finite zeros, so masked probabilities times
                // V stay exactly zero.
                *reinterpret_cast<uint4 *>(kd) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4 *>(vd) = make_uint4(0, 0, 0, 0);
            }
        }
        cp_async_wait_all();
        __syncthreads();

        float s[KEYS / 8][4];
#pragma unroll
        for (int j = 0; j < KEYS / 8; ++j) {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
            for (int kk = 0; kk < PADDED / 16; ++kk) {
                const half * kb = keys + (j * 8 + group) * ROW + kk * 16 + quad * 2;
                mma_f16_16816(s[j], qa[kk], load_half2(kb), load_half2(kb + 8));
            }
        }
        float tile0 = max0, tile1 = max1;
#pragma unroll
        for (int j = 0; j < KEYS / 8; ++j) {
            const int64_t key = first_key + j * 8 + quad * 2;
#pragma unroll
            for (int c = 0; c < 4; ++c)
                s[j][c] = key + (c & 1) < nk ? s[j][c] * log2_scale : -CUDART_INF_F;
            tile0 = fmaxf(tile0, fmaxf(s[j][0], s[j][1]));
            tile1 = fmaxf(tile1, fmaxf(s[j][2], s[j][3]));
        }
#pragma unroll
        for (int offset = 1; offset < 4; offset <<= 1) {
            tile0 = fmaxf(tile0, __shfl_xor_sync(0xffffffffu, tile0, offset));
            tile1 = fmaxf(tile1, __shfl_xor_sync(0xffffffffu, tile1, offset));
        }
        // The first tile always holds a key, so tile maxima are finite.
        const float correction0 = exp2f(max0 - tile0), correction1 = exp2f(max1 - tile1);
        max0 = tile0;
        max1 = tile1;
        sum0 *= correction0;
        sum1 *= correction1;
#pragma unroll
        for (int n = 0; n < PADDED / 8; ++n) {
            acc[n][0] *= correction0;
            acc[n][1] *= correction0;
            acc[n][2] *= correction1;
            acc[n][3] *= correction1;
        }
#pragma unroll
        for (int j = 0; j < KEYS / 8; ++j) {
            s[j][0] = exp2f(s[j][0] - max0);
            s[j][1] = exp2f(s[j][1] - max0);
            s[j][2] = exp2f(s[j][2] - max1);
            s[j][3] = exp2f(s[j][3] - max1);
            sum0 += s[j][0] + s[j][1];
            sum1 += s[j][2] + s[j][3];
        }
#pragma unroll
        for (int kk = 0; kk < KEYS / 16; ++kk) {
            // The score accumulators of two adjacent 8-key tiles are exactly the
            // A-operand fragment of the 16-key slice they cover.
            const uint32_t pa[4] = {
                pack_half2(s[2 * kk][0], s[2 * kk][1]), pack_half2(s[2 * kk][2], s[2 * kk][3]),
                pack_half2(s[2 * kk + 1][0], s[2 * kk + 1][1]), pack_half2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
            const half * vrow = values + (kk * 16 + (lane & 15)) * ROW;
#pragma unroll
            for (int n = 0; n < PADDED / 8; ++n) {
                uint32_t b0, b1;
                ldmatrix_x2_trans(b0, b1, vrow + n * 8);
                mma_f16_16816(acc[n], pa, b0, b1);
            }
        }
    }
#pragma unroll
    for (int offset = 1; offset < 4; offset <<= 1) {
        sum0 += __shfl_xor_sync(0xffffffffu, sum0, offset);
        sum1 += __shfl_xor_sync(0xffffffffu, sum1, offset);
    }
    const float inverse0 = 1.0f / sum0, inverse1 = 1.0f / sum1;
#pragma unroll
    for (int half_row = 0; half_row < 2; ++half_row) {
        const int64_t query = first_query + warp * 16 + group + half_row * 8;
        if (query >= nq)
            continue;
        const float inverse = half_row ? inverse1 : inverse0;
        float * out = output + (head + q.ne[2] * (query + q.ne[1] * batch)) * WIDTH;
#pragma unroll
        for (int n = 0; n < PADDED / 8; ++n) {
            const int x = n * 8 + quad * 2;
            if (x < WIDTH)
                out[x] = acc[n][half_row * 2] * inverse;
            if (x + 1 < WIDTH)
                out[x + 1] = acc[n][half_row * 2 + 1] * inverse;
        }
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
    if (desc.kind == TSG_ATTN_VISION_F32) {
        const auto q = tensor_view(dst->src[0]), k = tensor_view(dst->src[1]), v = tensor_view(dst->src[2]);
        GGML_ASSERT((q.ne[0] == 64 || q.ne[0] == 72) && (q.ne[1] + 127) / 128 <= INT_MAX && q.ne[2] <= 65535 && q.ne[3] <= 65535);
        const dim3 blocks(unsigned((q.ne[1] + 127) / 128), unsigned(q.ne[2]), unsigned(q.ne[3]));
        if (q.ne[0] == 64)
            vision_kernel<64><<<blocks, 128, 0, stream>>>(q, k, v, desc.f0, static_cast<float *>(dst->data));
        else
            vision_kernel<72><<<blocks, 128, 0, stream>>>(q, k, v, desc.f0, static_cast<float *>(dst->data));
    } else if (desc.kind == TSG_ATTN_VISION_MMA) {
        const auto q = tensor_view(dst->src[0]), k = tensor_view(dst->src[1]), v = tensor_view(dst->src[2]);
        GGML_ASSERT((q.ne[0] == 64 || q.ne[0] == 72) && (q.ne[1] + 127) / 128 <= INT_MAX && q.ne[2] <= 65535 && q.ne[3] <= 65535);
        // K/V rows are copied in 16-byte chunks straight from their F16 tensors.
        GGML_ASSERT(k.type == GGML_TYPE_F16 && v.type == GGML_TYPE_F16 && k.nb[0] == 2 && v.nb[0] == 2);
        GGML_ASSERT(((uintptr_t(k.data) | uintptr_t(v.data) | k.nb[1] | v.nb[1] | k.nb[2] | v.nb[2] | k.nb[3] | v.nb[3]) & 15) == 0);
        const dim3 blocks(unsigned((q.ne[1] + 127) / 128), unsigned(q.ne[2]), unsigned(q.ne[3]));
        if (q.ne[0] == 64)
            vision_mma_kernel<64><<<blocks, 256, 0, stream>>>(q, k, v, desc.f0, static_cast<float *>(dst->data));
        else
            vision_mma_kernel<72><<<blocks, 256, 0, stream>>>(q, k, v, desc.f0, static_cast<float *>(dst->data));
        CUDA_CHECK(cudaGetLastError());
    } else if (desc.kind == TSG_ATTN_MASK_COMPACT) {
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
        const bool mqa = mqa_eligible(q, k, v, mask, compact, desc.i0 != 0);
        if (!mqa && mqa_enabled() && mqa_shape(q, k, v)) {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true))
                std::fprintf(stderr, "[dsv41] an MLA attention launch cannot use the MQA kernel (layout: alignment, "
                    "head count or per-head mask); it runs the per-head kernel, several times slower\n");
        }
        if (mqa) {
            // Over 48 KiB of dynamic shared memory needs a per-device opt-in.
            static std::atomic<uint64_t> configured{0};
            const uint64_t bit = uint64_t(1) << (context->device & 63);
            if (!(configured.load() & bit)) {
                CUDA_CHECK(cudaFuncSetAttribute(mqa_partial_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    int(MQA_SHARED)));
                CUDA_CHECK(cudaFuncSetAttribute(mqa_partial_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    int(MQA_SHARED)));
                configured.fetch_or(bit);
            }
            const dim3 grid(unsigned(dst->ne[1]), unsigned(q.ne[2] / MQA_HEADS), unsigned(q.ne[1] * q.ne[3]));
            if (compact.data)
                mqa_partial_kernel<true><<<grid, MQA_THREADS, MQA_SHARED, stream>>>(q, k, mask, compact, desc.i0 != 0,
                    desc.f0, desc.i2, int(dst->ne[1]), out);
            else
                mqa_partial_kernel<false><<<grid, MQA_THREADS, MQA_SHARED, stream>>>(q, k, mask, compact, desc.i0 != 0,
                    desc.f0, desc.i2, int(dst->ne[1]), out);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
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
        GGML_ASSERT(rows <= INT_MAX && dst->src[0]->ne[1] <= 64);
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

bool tsg_vision_mma_supported(ggml_backend_t backend) {
    if (backend == nullptr || !ggml_backend_is_cuda(backend))
        return false;
    const auto * context = static_cast<const ggml_backend_cuda_context *>(backend->context);
    int major = 0;
    return cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, context->device) == cudaSuccess
        && major >= 8;
}

namespace {
__global__ void narrow_rows_f16(const char * x, size_t row_bytes, int64_t k_dim, int64_t count, __half * out) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(blockDim.x) * gridDim.x) {
        const int64_t t = i / k_dim, k = i - t * k_dim;
        out[i] = __float2half_rn(reinterpret_cast<const float *>(x + t * row_bytes)[k]);
    }
}

__device__ __forceinline__ float linear_gelu_tanh(float v) {
    return 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

// The fallback's epilogues, for a GPU whose cuBLASLt has no kernel with the bias / GELU epilogue.
__global__ void linear_epilogue_f32(float * y, const float * bias, const float * residual, int64_t n_dim, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(blockDim.x) * gridDim.x) {
        float v = y[i];
        if (bias != nullptr) v += bias[i % n_dim];
        if (residual != nullptr) v += residual[i];
        y[i] = v;
    }
}

__global__ void linear_epilogue_gelu_f16(__half * y, const __half * bias, int64_t n_dim, int64_t count) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += int64_t(blockDim.x) * gridDim.x) {
        float v = __half2float(y[i]);
        if (bias != nullptr) v += __half2float(bias[i % n_dim]);
        y[i] = __float2half_rn(linear_gelu_tanh(v));
    }
}

int linear_blocks(int64_t count) { return int(std::min<int64_t>((count + 255) / 256, 65535)); }

// One cuBLAS / cuBLASLt handle pair and a 32 MiB workspace per device (with only cuBLAS's default workspace the
// fastest kernels are left out), and one cuBLASLt plan per shape. ggml-cuda's own handles sit past members of its context
// that this translation unit cannot address (see tsg_msa_cuda_compute), so the linear keeps its own.
constexpr size_t kLinearF16Workspace = size_t(32) << 20;

struct LinearPlanKey {
    int device; int64_t k, n, t, ldx; int flags;
    bool operator==(const LinearPlanKey & o) const {
        return device == o.device && k == o.k && n == o.n && t == o.t && ldx == o.ldx && flags == o.flags;
    }
};
struct LinearPlanHash {
    size_t operator()(const LinearPlanKey & key) const {
        size_t h = std::hash<int64_t>()(key.k);
        const int64_t rest[] = {key.n, key.t, key.ldx, int64_t(key.flags), int64_t(key.device)};
        for (int64_t v : rest) h = h * 1000003u ^ std::hash<int64_t>()(v);
        return h;
    }
};
struct LinearPlan {
    bool lt = false;   // false: the plain-cuBLAS fallback
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
};
struct LinearState {
    std::mutex mu;
    cublasHandle_t blas[GGML_CUDA_MAX_DEVICES] = {};
    cublasLtHandle_t lt[GGML_CUDA_MAX_DEVICES] = {};
    void * workspace[GGML_CUDA_MAX_DEVICES] = {};
    std::unordered_map<LinearPlanKey, LinearPlan, LinearPlanHash> plans;
};
LinearState & linear_state() { static LinearState s; return s; }

enum : int { kLinearBias = 1, kLinearResidual = 2, kLinearGelu = 4 };

// Caller holds the state's lock.
LinearPlan & linear_plan(LinearState & s, const LinearPlanKey & key) {
    auto found = s.plans.find(key);
    if (found != s.plans.end()) return found->second;
    LinearPlan plan;
    const bool gelu = (key.flags & kLinearGelu) != 0, bias = (key.flags & kLinearBias) != 0;
    const cudaDataType_t out_type = gelu ? CUDA_R_16F : CUDA_R_32F;
    bool ok = cublasLtMatmulDescCreate(&plan.op, CUBLAS_COMPUTE_32F, CUDA_R_32F) == CUBLAS_STATUS_SUCCESS;
    const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    const cublasLtEpilogue_t epilogue = gelu ? (bias ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_GELU)
                                             : bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
    // A GELU epilogue writing F16 takes an F16 bias; the F32 results take the F32 bias as it is.
    const cudaDataType_t bias_type = gelu ? CUDA_R_16F : CUDA_R_32F;
    ok = ok && cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)) == CUBLAS_STATUS_SUCCESS
            && cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)) == CUBLAS_STATUS_SUCCESS
            && cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)) == CUBLAS_STATUS_SUCCESS
            && (!bias || cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                                        sizeof(bias_type)) == CUBLAS_STATUS_SUCCESS)
            && cublasLtMatrixLayoutCreate(&plan.a, CUDA_R_16F, key.k, key.n, key.k) == CUBLAS_STATUS_SUCCESS
            && cublasLtMatrixLayoutCreate(&plan.b, CUDA_R_16F, key.k, key.t, key.ldx) == CUBLAS_STATUS_SUCCESS
            && cublasLtMatrixLayoutCreate(&plan.c, out_type, key.n, key.t, key.n) == CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulPreference_t preference = nullptr;
    ok = ok && cublasLtMatmulPreferenceCreate(&preference) == CUBLAS_STATUS_SUCCESS
            && cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                    &kLinearF16Workspace, sizeof(kLinearF16Workspace)) == CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulHeuristicResult_t result{};
    int found_algos = 0;
    ok = ok && cublasLtMatmulAlgoGetHeuristic(s.lt[key.device], plan.op, plan.a, plan.b, plan.c, plan.c, preference, 1,
                                              &result, &found_algos) == CUBLAS_STATUS_SUCCESS && found_algos > 0;
    if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
    plan.lt = ok;
    if (ok) plan.algo = result.algo;
    return s.plans.emplace(key, plan).first->second;
}
} // namespace

void tsg_linear_f16_cuda_compute(ggml_tensor * dst, ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_cuda_context *>(backend->context);
    const int device = context->device;
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);
    CUDA_CHECK(cudaSetDevice(device));
    const cudaStream_t stream = context->stream(device, 0);
    ggml_custom_op_params params;
    std::memcpy(&params, dst->op_params, sizeof(params));
    const auto & d = *static_cast<const tsg_dsv4_fused_desc *>(params.userdata);
    const bool has_bias = d.i0 != 0, has_residual = d.i1 != 0, gelu = d.i2 != 0;
    const ggml_tensor * w = dst->src[0], * x = dst->src[1];
    int arg = 2;
    const ggml_tensor * bias = has_bias ? dst->src[arg++] : nullptr;
    const ggml_tensor * residual = has_residual ? dst->src[arg++] : nullptr;
    const int64_t k_dim = w->ne[0], n_dim = w->ne[1], tokens = x->ne[1];
    const bool x_f32 = x->type == GGML_TYPE_F32;
    const auto sizes = tsg_linear_f16_sizes(k_dim, n_dim, tokens, x_f32, gelu);
    GGML_ASSERT(ggml_nbytes(dst) >= sizes.out_bytes + sizes.x16_bytes + sizes.bias16_bytes &&
                "tsg_linear_f16: node was not built by it");
    char * scratch = static_cast<char *>(dst->data) + sizes.out_bytes;

    // Operands: x as F16 rows (narrowed into the node's scratch when it is F32), and with gelu the bias as F16.
    const void * xh = x->data;
    int64_t ldx = int64_t(x->nb[1] / sizeof(__half));
    if (x_f32) {
        auto * narrowed = reinterpret_cast<__half *>(scratch);
        const int64_t count = tokens * k_dim;
        narrow_rows_f16<<<linear_blocks(count), 256, 0, stream>>>(static_cast<const char *>(x->data), x->nb[1], k_dim,
                                                                  count, narrowed);
        CUDA_CHECK(cudaGetLastError());
        xh = narrowed;
        ldx = k_dim;
        scratch += sizes.x16_bytes;
    }
    const void * bias_ptr = bias != nullptr ? bias->data : nullptr;
    if (gelu && bias != nullptr) {
        auto * bias16 = reinterpret_cast<__half *>(scratch);
        narrow_rows_f16<<<linear_blocks(n_dim), 256, 0, stream>>>(static_cast<const char *>(bias->data), 0, n_dim, n_dim,
                                                                  bias16);
        CUDA_CHECK(cudaGetLastError());
        bias_ptr = bias16;
    }

    LinearState & s = linear_state();
    std::lock_guard<std::mutex> lock(s.mu);
    if (s.blas[device] == nullptr) {
        CUBLAS_CHECK(cublasCreate(&s.blas[device]));
        CUBLAS_CHECK(cublasLtCreate(&s.lt[device]));
        CUDA_CHECK(cudaMalloc(&s.workspace[device], kLinearF16Workspace));
    }
    const int flags = (has_bias ? kLinearBias : 0) | (has_residual ? kLinearResidual : 0) | (gelu ? kLinearGelu : 0);
    LinearPlan & plan = linear_plan(s, {device, k_dim, n_dim, tokens, ldx, flags});
    const float one = 1.0f, zero = 0.0f;
    const float beta = has_residual ? 1.0f : 0.0f;
    const void * c = has_residual ? residual->data : dst->data;
    if (plan.lt) {
        if (has_bias)
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)));
        CUBLAS_CHECK(cublasLtMatmul(s.lt[device], plan.op, &one, w->data, plan.a, xh, plan.b, &beta, c, plan.c, dst->data,
                                    plan.c, &plan.algo, s.workspace[device], kLinearF16Workspace, stream));
        return;
    }
    // Plain cuBLAS (F32 or, with gelu, F16 results) and the epilogue as its own kernel.
    CUBLAS_CHECK(cublasSetStream(s.blas[device], stream));
    // cublasSetStream resets the workspace to cuBLAS's default pool, so the workspace is set after it.
    CUBLAS_CHECK(cublasSetWorkspace(s.blas[device], s.workspace[device], kLinearF16Workspace));
    CUBLAS_CHECK(cublasGemmEx(s.blas[device], CUBLAS_OP_T, CUBLAS_OP_N, int(n_dim), int(tokens), int(k_dim), &one,
        w->data, CUDA_R_16F, int(k_dim), xh, CUDA_R_16F, int(ldx), &zero,
        dst->data, gelu ? CUDA_R_16F : CUDA_R_32F, int(n_dim), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    const int64_t count = n_dim * tokens;
    if (gelu)
        linear_epilogue_gelu_f16<<<linear_blocks(count), 256, 0, stream>>>(static_cast<__half *>(dst->data),
            static_cast<const __half *>(bias_ptr), n_dim, count);
    else if (has_bias || has_residual)
        linear_epilogue_f32<<<linear_blocks(count), 256, 0, stream>>>(static_cast<float *>(dst->data),
            static_cast<const float *>(bias_ptr), has_residual ? static_cast<const float *>(residual->data) : nullptr,
            n_dim, count);
    CUDA_CHECK(cudaGetLastError());
}
