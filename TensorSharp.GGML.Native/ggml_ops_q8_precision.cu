// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_q8_precision.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cuda/common.cuh"
#include <cstdint>

namespace {
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
void launch(ggml_tensor * dst, cudaStream_t stream) {
    const auto * w = dst->src[0];
    const auto * x = dst->src[1];
    const dim3 grid(unsigned((w->ne[1] + 63) / 64), unsigned((x->ne[1] + Columns - 1) / Columns));
    q8_f32_tiled<Columns><<<grid, 128, 0, stream>>>(static_cast<const char *>(w->data),
        static_cast<const char *>(x->data), static_cast<float *>(dst->data), int(w->ne[0]),
        int(w->ne[1]), int(x->ne[1]), w->nb[1], x->nb[0], x->nb[1]);
}
} // namespace

void tsg_matmul_q8_cuda_compute(ggml_tensor * dst, ggml_backend_t cuda_backend) {
    GGML_ASSERT(ggml_backend_is_cuda(cuda_backend) && ggml_is_contiguous(dst));
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_Q8_0 && dst->src[1]->type == GGML_TYPE_F32);
    // CUDA y-grid is bounded even on modern devices. The encoder's maximum
    // 65536-token microbatch is well below this limit for every tile size.
    GGML_ASSERT(dst->ne[1] <= int64_t(65535) * 8);
    auto * context = static_cast<ggml_backend_cuda_context *>(cuda_backend->context);
    CUDA_CHECK(cudaSetDevice(context->device));
    const cudaStream_t stream = context->stream(context->device, 0);
    if (dst->ne[1] <= 8) launch<8>(dst, stream);
    else if (dst->ne[1] <= 16) launch<16>(dst, stream);
    else launch<32>(dst, stream);
    CUDA_CHECK(cudaGetLastError());
}
