// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// TensorSharp-owned placement adapters. No upstream kernel or weight bytes change.
#include <cuda_bf16.h>
#include <cstdint>

extern "C" __global__ void ts_ninfer_fi_cast_bf16(
    const float* __restrict__ input, __nv_bfloat16* __restrict__ output,
    unsigned long long count)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2bfloat16_rn(input[i]);
}

// Preserve every F32 bit, including signed zero and NaN payloads. Rounding is the
// existing target caller's epilogue, never part of this row-placement operation.
extern "C" __global__ void ts_ninfer_fi_scatter_f32(
    const std::uint32_t* __restrict__ input, std::uint32_t* __restrict__ output,
    int outputRows, int tokens, int mapping)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long count = (unsigned long long)outputRows * tokens;
    if (i >= count) return;
    const int row = (int)(i % outputRows);
    int sourceRow = row;
    if (mapping == 1)
        sourceRow = (row / 16) * 8 + row % 8 + ((row % 16 >= 8) ? outputRows / 2 : 0);
    else if (mapping == 2)
    {
        if (row < 12288)
            sourceRow = (row / 512) * 256 + row % 256 + ((row % 512 >= 256) ? 7168 : 0);
        else if (row < 13312)
            sourceRow = 6144 + row - 12288;
    }
    output[i] = input[(i / outputRows) * outputRows + sourceRow];
}
