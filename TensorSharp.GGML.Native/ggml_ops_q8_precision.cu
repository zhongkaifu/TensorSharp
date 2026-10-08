// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_q8_precision.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace {
// Diagnostic opt-in while model/prefill/decode qualification is in progress.
// Fix the policy at first use: changing an environment variable must not mix
// arithmetic inside already captured CUDA graphs. The qualified default below
// retains its K-increasing sum and cross-column bitwise contract.
bool parallel_vector_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("TS_GGML_Q8_PARALLEL_VECTOR");
        const bool selected = value && std::strcmp(value, "1") == 0;
        if (selected)
            std::fprintf(stderr, "[q8-f32] Experimental parallel-K F32 vector selected (N=1).\n");
        return selected;
    }();
    return enabled;
}

bool parallel_small_batch_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("TS_GGML_Q8_PARALLEL_SMALL_BATCH");
        const bool selected = value && std::strcmp(value, "1") == 0;
        if (selected)
            std::fprintf(stderr, "[q8-f32] Experimental parallel-K F32 small batch selected (2<=N<=8).\n");
        return selected;
    }();
    return enabled;
}

// One full warp per output row. The decoded Q8 product and F32 activation are
// unchanged; the lane sums and five shuffle additions deliberately use a new
// reduction order. No atomics, staging allocation or input narrowing. A final
// partial CTA may contain unused whole warps, never a partial active warp.
template<bool Batched>
__global__ void q8_f32_vector_parallel(const char * weights, const char * input, float * output,
        int inner, int rows, size_t weight_stride, size_t input_inner_stride, size_t input_column_stride) {
    const int lane = int(threadIdx.x) & 31;
    const int row = int(blockIdx.x) * 4 + int(threadIdx.x) / 32;
    if (row >= rows) return;
    // Independent columns use the same lane/FMA/shuffle order as N=1. No
    // staging, duplicated weight ownership or extra device allocation.
    if constexpr (Batched) {
        input += size_t(blockIdx.y) * input_column_stride;
        output += size_t(blockIdx.y) * rows;
    }
    float sum = 0.0f;
    for (int block = 0; block < inner / 32; ++block) {
        const char * source = weights + size_t(row) * weight_stride + size_t(block) * 34;
        const float scale = __half2float(*reinterpret_cast<const half *>(source));
        const float weight = scale * float(*reinterpret_cast<const int8_t *>(source + 2 + lane));
        const float activation = *reinterpret_cast<const float *>(input + size_t(block * 32 + lane) * input_inner_stride);
        sum = fmaf(weight, activation, sum);
    }
    for (int shift = 16; shift > 0; shift /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, shift);
    if (lane == 0) output[row] = sum;
}

// Decode has only one activation column. The matrix kernel below would still
// compute eight columns (seven zero-filled), leaving only 16 of 128 threads
// contributing useful results. Threads cooperatively decode 64 weight rows,
// then one half warp owns four independent row accumulators per lane. All
// other threads skip the otherwise redundant seven columns while helping the
// coalesced weight loads. Shared-memory transposition
// keeps both the Q8 loads and the per-K row reads coalesced/bank-conflict free.
// Keep exactly the existing scale multiplication followed by K-increasing
// fmaf: no parallel-K reduction, activation narrowing or extra payload buffer.
__global__ void q8_f32_vector(const char * weights, const char * input, float * output,
        int inner, int rows, size_t weight_stride, size_t input_inner_stride) {
    constexpr int TileRows = 64, TileInner = 32;
    __shared__ float ws[TileInner][TileRows + 1];
    __shared__ float xs[TileInner];
    const int lane = int(threadIdx.x);
    const int row_base = int(blockIdx.x) * TileRows;
    float sums[4] = {};
    for (int base = 0; base < inner; base += TileInner) {
        for (int index = lane; index < TileRows * TileInner; index += 128) {
            const int row = index / TileInner, k = index % TileInner;
            float value = 0.0f;
            if (row_base + row < rows) {
                const char * block = weights + size_t(row_base + row) * weight_stride + size_t(base / 32) * 34;
                const float scale = __half2float(*reinterpret_cast<const half *>(block));
                value = scale * float(*reinterpret_cast<const int8_t *>(block + 2 + k));
            }
            ws[k][row] = value;
        }
        if (lane < TileInner)
            xs[lane] = *reinterpret_cast<const float *>(input + size_t(base + lane) * input_inner_stride);
        __syncthreads();
        if (lane < 16) {
#pragma unroll
            for (int k = 0; k < TileInner; ++k) {
                const float x = xs[k];
#pragma unroll
                for (int row = 0; row < 4; ++row)
                    sums[row] = fmaf(ws[k][lane + row * 16], x, sums[row]);
            }
        }
        __syncthreads();
    }
    if (lane < 16) {
#pragma unroll
        for (int row = 0; row < 4; ++row)
            if (row_base + lane + row * 16 < rows) output[row_base + lane + row * 16] = sums[row];
    }
}

// Each CTA owns a 64-row output tile. Its 128 threads each accumulate four
// rows and Columns/8 columns. Weight blocks widen once into shared memory and
// are reused across the entire column tile, preserving quantized residency.
// Padding the shared-memory leading dimensions avoids transposed-load bank
// conflicts. Every output traverses K in the same order for every N dispatch;
// changing companions or moving a column to another tile cannot change its
// reduction order. No split-K, atomics, source narrowing, or activation quants.
template<int Columns>
__global__ void q8_f32_tiled(const char * weights, const char * input, float * output,
        int inner, int rows, int columns, size_t weight_stride, size_t input_inner_stride,
        size_t input_column_stride) {
    constexpr int TileRows = 64, TileInner = 32, ColumnParts = Columns / 8;
    __shared__ float ws[TileInner][TileRows + 1];
    __shared__ float xs[TileInner][Columns + 1];
    const int row_base = int(blockIdx.x) * TileRows;
    const int column_base = int(blockIdx.y) * Columns;
    const int row_lane = threadIdx.x % 16, column_lane = threadIdx.x / 16;
    float sums[4][ColumnParts] = {};
    for (int base = 0; base < inner; base += TileInner) {
        for (int index = threadIdx.x; index < TileRows * TileInner; index += 128) {
            const int row = index / TileInner, k = index % TileInner;
            float value = 0;
            if (row_base + row < rows) {
                // Q8_0 blocks contain a two-byte scale and 32 signed bytes.
                // The 34-byte block stride does not permit aligned int4 loads.
                const char * block = weights + size_t(row_base + row) * weight_stride + size_t(base / 32) * 34;
                const float scale = __half2float(*reinterpret_cast<const half *>(block));
                value = scale * float(*reinterpret_cast<const int8_t *>(block + 2 + k));
            }
            ws[k][row] = value;
        }
        for (int index = threadIdx.x; index < Columns * TileInner; index += 128) {
            const int column = index / TileInner, k = index % TileInner;
            xs[k][column] = column_base + column < columns
                ? *reinterpret_cast<const float *>(input + size_t(column_base + column) * input_column_stride
                    + size_t(base + k) * input_inner_stride) : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TileInner; ++k) {
            float w[4], x[ColumnParts];
#pragma unroll
            for (int r = 0; r < 4; ++r) w[r] = ws[k][row_lane + 16 * r];
#pragma unroll
            for (int c = 0; c < ColumnParts; ++c) x[c] = xs[k][column_lane + 8 * c];
#pragma unroll
            for (int r = 0; r < 4; ++r)
#pragma unroll
            for (int c = 0; c < ColumnParts; ++c) sums[r][c] = fmaf(w[r], x[c], sums[r][c]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int c = 0; c < ColumnParts; ++c) {
        const int row = row_base + row_lane + 16 * r;
        const int column = column_base + column_lane + 8 * c;
        if (row < rows && column < columns) output[size_t(column) * rows + row] = sums[r][c];
    }
}

template<int Columns>
void launch(const void * weights, const void * input, float * output,
        int inner, int rows, int columns, size_t weight_stride, size_t input_inner_stride,
        size_t input_column_stride, cudaStream_t stream) {
    const dim3 grid(unsigned((int64_t(rows) + 63) / 64), unsigned((columns + Columns - 1) / Columns));
    q8_f32_tiled<Columns><<<grid, 128, 0, stream>>>(static_cast<const char *>(weights),
        static_cast<const char *>(input), output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride);
}
} // namespace

int tsg_matmul_q8_cuda_launch(const void * weights, const void * input, float * output,
        int inner, int rows, int columns, size_t weight_stride, size_t input_inner_stride,
        size_t input_column_stride, void * stream_pointer) {
    const auto stream = static_cast<cudaStream_t>(stream_pointer);
    if (columns == 1 && parallel_vector_enabled()) {
        const unsigned blocks = unsigned((int64_t(rows) + 3) / 4);
        q8_f32_vector_parallel<false><<<blocks, 128, 0, stream>>>(static_cast<const char *>(weights),
            static_cast<const char *>(input), output, inner, rows, weight_stride, input_inner_stride, input_column_stride);
    }
    else if (columns == 1) {
        const unsigned blocks = unsigned((int64_t(rows) + 63) / 64);
        q8_f32_vector<<<blocks, 128, 0, stream>>>(static_cast<const char *>(weights),
            static_cast<const char *>(input), output, inner, rows, weight_stride, input_inner_stride);
    }
    else if (columns <= 8 && parallel_small_batch_enabled()) {
        const dim3 grid(unsigned((int64_t(rows) + 3) / 4), unsigned(columns));
        q8_f32_vector_parallel<true><<<grid, 128, 0, stream>>>(static_cast<const char *>(weights),
            static_cast<const char *>(input), output, inner, rows, weight_stride, input_inner_stride, input_column_stride);
    }
    else if (columns <= 8) launch<8>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    else if (columns <= 16) launch<16>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    else launch<32>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    return int(cudaGetLastError());
}

#if defined(TSG_GGML_TEST_HOOKS)
// Keep the previous implementation available only to native correctness tests;
// no runtime switch or additional public C ABI changes production dispatch.
int tsg_matmul_q8_cuda_launch_reference(const void * weights, const void * input, float * output,
        int inner, int rows, int columns, size_t weight_stride, size_t input_inner_stride,
        size_t input_column_stride, void * stream_pointer) {
    const auto stream = static_cast<cudaStream_t>(stream_pointer);
    if (columns <= 8) launch<8>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    else if (columns <= 16) launch<16>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    else launch<32>(weights, input, output, inner, rows, columns,
        weight_stride, input_inner_stride, input_column_stride, stream);
    return int(cudaGetLastError());
}
#endif

void tsg_matmul_q8_cuda_compute(ggml_tensor * dst, ggml_backend_t cuda_backend) {
    GGML_ASSERT(ggml_backend_is_cuda(cuda_backend) && ggml_is_contiguous(dst));
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q8_0 && dst->src[1]->type == GGML_TYPE_F32);
    // CUDA y-grid is bounded even on modern devices. The encoder's maximum
    // 65536-token microbatch is well below this limit for every tile size.
    GGML_ASSERT(dst->ne[1] <= int64_t(65535) * 8);
    auto * context = static_cast<ggml_backend_cuda_context *>(cuda_backend->context);
    CUDA_CHECK(cudaSetDevice(context->device));
    const cudaStream_t stream = context->stream(context->device, 0);
    const auto * w = dst->src[0];
    const auto * x = dst->src[1];
    CUDA_CHECK(static_cast<cudaError_t>(tsg_matmul_q8_cuda_launch(w->data, x->data,
        static_cast<float *>(dst->data), int(w->ne[0]), int(w->ne[1]), int(x->ne[1]),
        w->nb[1], x->nb[0], x->nb[1], stream)));
}
