// Copyright (c) Zhongkai Fu. All rights reserved.
// Derived portions are licensed under Apache-2.0; see a16-source-provenance.json
// and the unchanged upstream LICENSE/NOTICE files in ../w8/upstream/.
//
// TensorSharp-owned storage adaptation of the published NInfer A16 FMA schedule:
// gpillon/ignis e600a7d9e0d5921d767a6cb464373583b3cdcaf4,
// kernel/vendor/src/ops/linear/nvfp4/{nvfp4_gemv.cuh,nvfp4_small_t.cuh,nvfp4_config.h}.
// The reference fork pinned by that revision is gpillon/ninfer a00648cb828457986cf5b4b4f712b4cbcd7af0d1.
//
// Changes from that source: a standalone CUDA-driver entry; TS low8/high8 code
// bytes and row-major scales; F32 storage of already BF16-rounded activations;
// explicit strides and row tails. There is no PDL launch contract here. T=1 uses
// the original four accumulator chains, T=2/3 its one-chain small-T schedule.
// This differs numerically from the existing ts_ni_project_a16 helper and must
// pass independent component and model checks before any serving adoption.
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kRowsPerWarp = 2;
constexpr int kValuesPerLane = 16;
constexpr int kPhaseValues = 32 * kValuesPerLane;

__device__ __forceinline__ float decode_scale(uint8_t bits) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<uint16_t>(bits) | (static_cast<uint16_t>(bits) << 8);
    return static_cast<float2>(value).x;
}

// Original arithmetic consumes adjacent logical K pairs. Reconstruct just one
// such packed pair from the two TS uint32 words; encoded values stay unchanged.
__device__ __forceinline__ float2 decode_pair(uint2 codes, int pair) {
    const uint32_t word = (pair & 3) < 2 ? codes.x : codes.y;
    const int byte_shift = (pair & 1) * 16;
    const int nibble_shift = pair < 4 ? 0 : 4;
    const unsigned first = (word >> (byte_shift + nibble_shift)) & 15u;
    const unsigned second = (word >> (byte_shift + 8 + nibble_shift)) & 15u;
    __nv_fp4x2_e2m1 value;
    value.__x = static_cast<uint8_t>(first | (second << 4));
    return static_cast<float2>(value);
}

__device__ __forceinline__ float warp_reduce(float value) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, delta);
    return value;
}

template<int Columns, int Tokens, int Chains, bool StageActivation>
__device__ __forceinline__ void project(const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ scales, const float* __restrict__ input,
    float* __restrict__ output, int n, int k, int rows, int input_stride,
    int output_stride, float inverse_weight_divisor, float* staged) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int first_row = static_cast<int>(blockIdx.x) * kWarps * kRowsPerWarp + warp * kRowsPerWarp;
    // Every CTA takes the same path before a possible shared-memory barrier.
    if (rows != Tokens || k != Columns || blockDim.x != kThreads) {
        asm("trap;");
        return;
    }
    float accumulator[kRowsPerWarp][Tokens][Chains] = {};
    constexpr int groups_per_row = Columns / 16;
    constexpr int code_bytes_per_row = Columns / 2;
    constexpr int phase_unroll = Tokens == 1 ? Columns / kPhaseValues : 1;

#pragma unroll phase_unroll
    for (int phase = 0; phase < Columns / kPhaseValues; ++phase) {
        if constexpr (StageActivation) {
            for (int index = threadIdx.x; index < Tokens * kPhaseValues; index += kThreads) {
                const int token = index / kPhaseValues;
                const int column = index % kPhaseValues;
                staged[index] = input[static_cast<long long>(token) * input_stride + phase * kPhaseValues + column];
            }
            __syncthreads();
        }

        const int group = phase * 32 + lane;
        float coefficient[kRowsPerWarp];
        uint2 row_codes[kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < kRowsPerWarp; ++local_row) {
            const int row = first_row + local_row;
            coefficient[local_row] = 0.0f;
            row_codes[local_row] = make_uint2(0, 0);
            if (row < n) {
                // Keep scale * inverseD before the code multiplication, exactly
                // as the original A16 schedule does; do not scale the final dot.
                coefficient[local_row] = decode_scale(scales[static_cast<long long>(row) * groups_per_row + group]) * inverse_weight_divisor;
                row_codes[local_row] = *reinterpret_cast<const uint2*>(codes + static_cast<long long>(row) * code_bytes_per_row + group * 8);
            }
        }

#pragma unroll
        for (int pair = 0; pair < 8; ++pair) {
            float2 weight[kRowsPerWarp];
#pragma unroll
            for (int local_row = 0; local_row < kRowsPerWarp; ++local_row) {
                const float2 code = decode_pair(row_codes[local_row], pair);
                weight[local_row] = make_float2(code.x * coefficient[local_row], code.y * coefficient[local_row]);
            }
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                const int column = lane * 16 + pair * 2;
                float x0, x1;
                if constexpr (StageActivation) {
                    x0 = staged[token * kPhaseValues + column];
                    x1 = staged[token * kPhaseValues + column + 1];
                } else {
                    const long long at = static_cast<long long>(token) * input_stride + phase * kPhaseValues + column;
                    x0 = input[at];
                    x1 = input[at + 1];
                }
#pragma unroll
                for (int local_row = 0; local_row < kRowsPerWarp; ++local_row) {
                    accumulator[local_row][token][(2 * pair) & (Chains - 1)] =
                        fmaf(weight[local_row].x, x0, accumulator[local_row][token][(2 * pair) & (Chains - 1)]);
                    accumulator[local_row][token][(2 * pair + 1) & (Chains - 1)] =
                        fmaf(weight[local_row].y, x1, accumulator[local_row][token][(2 * pair + 1) & (Chains - 1)]);
                }
            }
        }
        if constexpr (StageActivation) __syncthreads();
    }

#pragma unroll
    for (int local_row = 0; local_row < kRowsPerWarp; ++local_row) {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            float total = 0.0f;
#pragma unroll
            for (int chain = 0; chain < Chains; ++chain) total += accumulator[local_row][token][chain];
            total = warp_reduce(total);
            if (lane == 0 && first_row + local_row < n)
                output[static_cast<long long>(token) * output_stride + first_row + local_row] = total;
        }
    }
}
}

#define TS_NINFER_A16_ENTRY(Name, Columns, Tokens, Chains, Stage, MinBlocks) \
extern "C" __global__ __launch_bounds__(256, MinBlocks) void Name( \
    const uint8_t* codes, const uint8_t* scales, const float* input, float* output, \
    int n, int k, int rows, int input_stride, int output_stride, float inverse_weight_divisor) { \
    __shared__ float staged[(Stage) ? (Tokens) * kPhaseValues : 1]; \
    project<Columns, Tokens, Chains, Stage>(codes, scales, input, output, n, k, rows, input_stride, output_stride, inverse_weight_divisor, staged); \
}

#define TS_NINFER_A16_FOR_K(Columns) \
    TS_NINFER_A16_ENTRY(ts_ni_project_a16_candidate_k##Columns##_t1, Columns, 1, 4, false, 2) \
    TS_NINFER_A16_ENTRY(ts_ni_project_a16_candidate_k##Columns##_t2, Columns, 2, 1, true, 1) \
    TS_NINFER_A16_ENTRY(ts_ni_project_a16_candidate_k##Columns##_t3, Columns, 3, 1, true, 1)

TS_NINFER_A16_FOR_K(5120)
TS_NINFER_A16_FOR_K(6144)
TS_NINFER_A16_FOR_K(17408)
