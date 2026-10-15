// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_f16_streaming.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace {
// No activation narrowing or split-K reduction. Every output visits K in the
// same F32 FMA order, independent of token/output tiling. Odd K is supported.
template<int Columns>
__global__ void f16_f32_tiled(const half* weights, const float* input, float* output,
        int inner, int rows, int columns) {
    constexpr int TileRows = 64, TileInner = 32, ColumnParts = Columns / 8;
    __shared__ float ws[TileInner][TileRows + 1];
    __shared__ float xs[TileInner][Columns + 1];
    const int row_base = int(blockIdx.x) * TileRows;
    const int column_base = int(blockIdx.y) * Columns;
    const int row_lane = threadIdx.x % 16, column_lane = threadIdx.x / 16;
    float sums[4][ColumnParts] = {};
    for (std::int64_t base = 0; base < inner; base += TileInner) {
        for (int index = threadIdx.x; index < TileRows * TileInner; index += 128) {
            const int row = index / TileInner, k = index % TileInner;
            ws[k][row] = row_base + row < rows && base + k < inner
                ? __half2float(weights[std::size_t(row_base + row) * inner + std::size_t(base + k)]) : 0.0f;
        }
        for (int index = threadIdx.x; index < Columns * TileInner; index += 128) {
            const int column = index / TileInner, k = index % TileInner;
            xs[k][column] = column_base + column < columns && base + k < inner
                ? input[std::size_t(column_base + column) * inner + std::size_t(base + k)] : 0.0f;
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
        const int row = row_base + row_lane + 16 * r, column = column_base + column_lane + 8 * c;
        if (row < rows && column < columns) output[std::size_t(column) * rows + row] = sums[r][c];
    }
}
template<int Columns>
void launch(const void* weights, const void* input, float* output, int inner, int rows, int columns, cudaStream_t stream) {
    const dim3 grid(unsigned((std::int64_t(rows) + 63) / 64), unsigned((columns + Columns - 1) / Columns));
    f16_f32_tiled<Columns><<<grid, 128, 0, stream>>>(static_cast<const half*>(weights),
        static_cast<const float*>(input), output, inner, rows, columns);
}
}
int tsg_matmul_f16_cuda_launch(const void* weights, const void* input, float* output,
        int inner, int rows, int columns, void* stream_pointer) {
    const auto stream = static_cast<cudaStream_t>(stream_pointer);
    if (columns <= 8) launch<8>(weights, input, output, inner, rows, columns, stream);
    else if (columns <= 16) launch<16>(weights, input, output, inner, rows, columns, stream);
    else launch<32>(weights, input, output, inner, rows, columns, stream);
    return int(cudaGetLastError());
}
