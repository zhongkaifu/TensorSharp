// Copyright (c) Zhongkai Fu. All rights reserved.
// Derived scheduling portions are licensed under Apache-2.0; see
// a16-batch-exact-source-provenance.json and unchanged ../w8/upstream/
// LICENSE/NOTICE files. TensorSharp modifications retain this attribution.
// Scheduling adapted from the separately attributed original NInfer A16
// candidate: gpillon/ignis e600a7d9e0d5921d767a6cb464373583b3cdcaf4,
// kernel/vendor/src/ops/linear/nvfp4/{nvfp4_gemv.cuh,nvfp4_small_t.cuh}.
// TensorSharp-owned scheduling optimization of ts_ni_project_a16. This keeps
// its arithmetic order, including low-K/high-K alternation and XOR reduction.
// No arithmetic from the separate original-Ignis-schedule candidate is used.
// This separate image replaces only FP4 table lookups with exact FP4x2 decode.
// This separate optional image extends the exact per-token policy through T=8.
// The already qualified T1..3 images remain unchanged. Shared-memory swizzling
// changes addresses only, preserving every F32 activation bit.
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int kThreads = 256;
constexpr int kRowsPerWarp = 2;
constexpr int kRowsPerBlock = kThreads / 32 * kRowsPerWarp;
constexpr int kPhaseValues = 512;

__device__ __forceinline__ int stage_index(int column) {
    // A permutation within each512-value phase. Both contiguous warp stores
    // and lane*16+j warp reads address distinct 32-bit shared-memory banks.
    return column ^ ((column >> 4) & 31);
}

__device__ __forceinline__ float2 decode_codes(uint8_t codes) {
    // The TS byte is (logical K=j, logical K=j+8), exactly the legacy
    // alternating FMA order. Every E2M1 value, including negative zero,
    // is exactly representable in F32; no weight multiplication is moved.
    __nv_fp4x2_e2m1 pair;
    pair.__x = codes;
    return static_cast<float2>(pair);
}

__device__ __forceinline__ float decode_scale(uint8_t code) {
    __nv_fp8_e4m3 value;
    value.__x = code;
    return static_cast<float>(value);
}

template<int Columns, int Tokens, bool StageActivation>
__device__ __forceinline__ void project(const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ scales, const float* __restrict__ input,
    float* __restrict__ output, int n, int k, int rows, int input_stride,
    int output_stride, float inverse_weight_divisor, float* staged) {
    const int lane = threadIdx.x & 31;
    const int first_row = static_cast<int>(blockIdx.x) * kRowsPerBlock
        + (threadIdx.x >> 5) * kRowsPerWarp;
    if (rows != Tokens || k != Columns || blockDim.x != kThreads) {
        asm("trap;");
        return;
    }
    // One accumulator per row/token, initialized once. Never introduce the
    // extra 0 + accumulator used to merge the other candidate's chains.
    float accumulator[kRowsPerWarp][Tokens] = {};
    constexpr int groups_per_row = Columns / 16;
    constexpr int code_bytes_per_row = Columns / 2;
    constexpr int phase_unroll = Tokens == 1 ? Columns / kPhaseValues : 1;
#pragma unroll phase_unroll
    for (int phase = 0; phase < Columns / kPhaseValues; ++phase) {
        if constexpr (StageActivation) {
            for (int i = threadIdx.x; i < Tokens * kPhaseValues; i += kThreads) {
                int token = i / kPhaseValues;
                staged[token * kPhaseValues + stage_index(i % kPhaseValues)] = input[static_cast<long long>(token) * input_stride
                    + phase * kPhaseValues + i % kPhaseValues];
            }
            __syncthreads();
        }
        const int group = phase * 32 + lane;
        float coefficient[kRowsPerWarp];
        uint2 packed[kRowsPerWarp];
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            coefficient[r] = 0;
            packed[r] = make_uint2(0, 0);
            if (first_row + r < n) {
                coefficient[r] = decode_scale(scales[static_cast<long long>(first_row + r)
                    * groups_per_row + group]) * inverse_weight_divisor;
                packed[r] = *reinterpret_cast<const uint2*>(codes
                    + static_cast<long long>(first_row + r) * code_bytes_per_row + group * 8);
            }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float low[kRowsPerWarp], high[kRowsPerWarp];
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                const uint32_t word = j < 4 ? packed[r].x : packed[r].y;
                const uint8_t code = static_cast<uint8_t>(word >> ((j & 3) * 8));
                const float2 decoded = decode_codes(code);
                low[r] = decoded.x * coefficient[r];
                high[r] = decoded.y * coefficient[r];
            }
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                float x_low, x_high;
                if constexpr (StageActivation) {
                    x_low = staged[token * kPhaseValues + stage_index(lane * 16 + j)];
                    x_high = staged[token * kPhaseValues + stage_index(lane * 16 + j + 8)];
                } else {
                    const long long at = static_cast<long long>(token) * input_stride
                        + phase * kPhaseValues + lane * 16 + j;
                    x_low = input[at];
                    x_high = input[at + 8];
                }
#pragma unroll
                for (int r = 0; r < kRowsPerWarp; ++r) {
                    accumulator[r][token] = fmaf(low[r], x_low, accumulator[r][token]);
                    accumulator[r][token] = fmaf(high[r], x_high, accumulator[r][token]);
                }
            }
        }
        if constexpr (StageActivation) __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            float sum = accumulator[r][token];
#pragma unroll
            for (int distance = 16; distance; distance >>= 1)
                sum += __shfl_xor_sync(0xffffffffu, sum, distance);
            if (!lane && first_row + r < n)
                output[static_cast<long long>(token) * output_stride + first_row + r] = sum;
        }
    }
}
}

#define TS_NINFER_LEGACY_EXACT_ENTRY(Columns, Tokens, Stage, MinBlocks) \
extern "C" __global__ __launch_bounds__(256, MinBlocks) \
void ts_ni_project_a16_batch_exact_k##Columns##_t##Tokens( \
    const uint8_t* codes, const uint8_t* scales, const float* input, float* output, \
    int n, int k, int rows, int input_stride, int output_stride, float inverse_weight_divisor) { \
    __shared__ float staged[(Stage) ? (Tokens) * kPhaseValues : 1]; \
    project<Columns, Tokens, Stage>(codes, scales, input, output, n, k, rows, input_stride, \
        output_stride, inverse_weight_divisor, staged); \
}

#define TS_NINFER_LEGACY_EXACT_FOR_K(Columns) \
    TS_NINFER_LEGACY_EXACT_ENTRY(Columns, 1, false, 2) \
    TS_NINFER_LEGACY_EXACT_ENTRY(Columns, 2, true, 1) \
    TS_NINFER_LEGACY_EXACT_ENTRY(Columns, 3, true, 1) \
    TS_NINFER_LEGACY_EXACT_ENTRY(Columns, 4, true, 1) \
    TS_NINFER_LEGACY_EXACT_ENTRY(Columns, 5, true, 1) \
    TS_NINFER_LEGACY_EXACT_ENTRY(Columns, 6, true, 1) \
    TS_NINFER_LEGACY_EXACT_ENTRY(Columns, 7, true, 1) \
    TS_NINFER_LEGACY_EXACT_ENTRY(Columns, 8, true, 1)

TS_NINFER_LEGACY_EXACT_FOR_K(5120)
TS_NINFER_LEGACY_EXACT_FOR_K(6144)
TS_NINFER_LEGACY_EXACT_FOR_K(17408)
