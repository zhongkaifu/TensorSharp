#pragma once

// Shared Qwen3.6 GQA dimensions and leaf PTX helpers used by the independently tuned
// BF16 and INT8 prompt kernels. This file deliberately owns no staging policy,
// shared-memory arena, warp schedule, or kernel body.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaPrefillHeadDim = 256;

inline constexpr int kGqaPrefillBr        = 64;
inline constexpr int kGqaPrefillBc        = 64;
inline constexpr int kGqaPrefillThreads   = 128;
inline constexpr int kGqaPrefillSmemBytes = (kGqaPrefillBr + 2 * kGqaPrefillBc) *
                                            kGqaPrefillHeadDim *
                                            static_cast<int>(sizeof(__nv_bfloat16));

struct GqaPrefillDirectMetadata {
    const std::int32_t* table;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const { return width; }

    __device__ __forceinline__ const std::int32_t* block_table() const { return table; }

    // Residual side planes in a layer view are pre-sliced to the sequence's slot row.
    __device__ __forceinline__ std::int32_t residual_slot() const { return 0; }
};

template <bool Masked>
struct GqaPrefillBatchMetadata {
    const std::int32_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    std::int32_t table_stride;

    __device__ __forceinline__ std::int32_t valid_tokens(std::int32_t width) const {
        if constexpr (Masked) {
            const std::int32_t valid = valid_columns[0];
            return valid <= 0 ? 0 : (valid < width ? valid : width);
        }
        return width;
    }

    __device__ __forceinline__ const std::int32_t* block_table() const {
        return tables + static_cast<std::int64_t>(table_rows[0]) * table_stride;
    }

    // The prompt route is single-sequence; the batch view carries every slot row.
    __device__ __forceinline__ std::int32_t residual_slot() const { return table_rows[0]; }
};

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_prefill_q_index(int q_head, int d, int token) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaPrefillHeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ void gqa_prefill_zero_output_rows(__nv_bfloat16* out, int q_head,
                                                             int row_begin, int row_end, int tid,
                                                             int threads) {
    if (row_begin >= row_end) { return; }
    const int elements = (row_end - row_begin) * kGqaPrefillHeadDim;
    for (int element = tid; element < elements; element += threads) {
        const int row = row_begin + element / kGqaPrefillHeadDim;
        const int d   = element - (row - row_begin) * kGqaPrefillHeadDim;
        out[gqa_prefill_q_index<Geometry>(q_head, d, row)] = __float2bfloat16(0.0f);
    }
}

// XOR-swizzled b16 element address. INT8 operands use the same layout by packing
// two consecutive signed bytes into each b16 lane before ldmatrix.
__device__ __forceinline__ int gqa_prefill_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ unsigned gqa_prefill_swz_addr(unsigned lane_base, unsigned ck,
                                                         unsigned as, unsigned r) {
    return lane_base + ((ck | as) ^ r);
}

// Key-split partial layout (ROADMAP WI-K1a): the prompt kernels with split_count > 1 store
// per-(head, token, split) online-softmax state in the same tensor shapes the small-T route
// allocates ({kHeadDim, q_heads, width, splits} BF16 and {q_heads, width, splits} FP32). The
// index formulas mirror the decode-side gqa_partial_{acc,stat}_index; they are kept local so
// prefill TUs do not pull the decode kernel header.
template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_prefill_partial_acc_index(int q_head, int d, int token,
                                                                      int split, int width) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaPrefillHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(Geometry::QHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(width) * split));
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_prefill_partial_stat_index(int q_head, int token,
                                                                       int split, int width) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(Geometry::QHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(width) * split);
}

// Merges the fixed-S key-split partials back into normalized bf16 rows. m = -inf marks a
// causally-empty split (or a dead output row); such rows emit zeros without reading acc.
template <typename Geometry>
__global__ __launch_bounds__(256) void gqa_attention_prefill_reduce_kernel(
    const float* __restrict__ partial_acc, const float* __restrict__ partial_m,
    const float* __restrict__ partial_l, float scale, std::int32_t width,
    std::int32_t split_count, __nv_bfloat16* __restrict__ out) {
    constexpr float kLog2E = 1.4426950408889634074f;
    const int token        = static_cast<int>(blockIdx.x);
    const int q_head       = static_cast<int>(blockIdx.y);
    if (token >= width || q_head >= Geometry::QHeads) { return; }
    const int d = static_cast<int>(threadIdx.x);

    float m = -CUDART_INF_F;
    for (int split = 0; split < split_count; ++split) {
        m = fmaxf(m, partial_m[gqa_prefill_partial_stat_index<Geometry>(q_head, token, split, width)]);
    }
    if (m == -CUDART_INF_F) {
        out[gqa_prefill_q_index<Geometry>(q_head, d, token)] = __float2bfloat16(0.0f);
        return;
    }
    const float scale_l2 = scale * kLog2E;
    float weights[4];   // l_s * exp2((m_s - m*) * scale * log2e): the split's share of the
                        // final normalization mass (its partial acc is stored pre-normalized
                        // by its own l_s, so the merge re-weights by exactly this).
    float l = 0.0f;
    for (int split = 0; split < split_count; ++split) {
        const float ms = partial_m[gqa_prefill_partial_stat_index<Geometry>(q_head, token, split, width)];
        const float ls = partial_l[gqa_prefill_partial_stat_index<Geometry>(q_head, token, split, width)];
        const float share = ms == -CUDART_INF_F ? 0.0f : ls * exp2f((ms - m) * scale_l2);
        weights[split] = share;
        l += share;
    }
    float acc = 0.0f;
    for (int split = 0; split < split_count; ++split) {
        acc += partial_acc[gqa_prefill_partial_acc_index<Geometry>(q_head, d, token, split,
                                                                   width)] *
               weights[split];
    }
    out[gqa_prefill_q_index<Geometry>(q_head, d, token)] =
        __float2bfloat16(l > 0.0f ? acc / l : 0.0f);
}

} // namespace ninfer::ops
