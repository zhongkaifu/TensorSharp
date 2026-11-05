// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Exact, partitioned top-k for DFlash proposal logits. Only comparisons and copies:
// no logit rounding, approximate shortlist, or changed candidate tie policy.

#pragma once

__device__ __forceinline__ bool ts_q35_topk_before(float value, int index, float other, int otherIndex)
{
    return value > other || (value == other && index < otherIndex);
}

// The original one-CTA selector omits NaNs and -infinity, padding exhausted rows
// with (-infinity, INT_MAX). Preserve those semantics, including signed-zero bits.
__device__ __forceinline__ void ts_q35_topk_partition(
    const float* row, const int* indices, int count, int indexBase, int k,
    int* idxOut, float* valOut)
{
    __shared__ float values[16][257];
    __shared__ int indicesShared[16][257];
    __shared__ float warpValues[8];
    __shared__ int warpIndices[8], warpSlots[8], heads[256];
    float bestValues[16];
    int bestIndices[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
    {
        bestValues[j] = -INFINITY;
        bestIndices[j] = 0x7fffffff;
    }
    for (int i = threadIdx.x; i < count; i += blockDim.x)
    {
        const float value = row[i];
        const int index = indices ? indices[i] : indexBase + i;
        if (value > -INFINITY && ts_q35_topk_before(value, index, bestValues[15], bestIndices[15]))
        {
            bestValues[15] = value;
            bestIndices[15] = index;
#pragma unroll
            for (int j = 15; j > 0; --j)
                if (ts_q35_topk_before(bestValues[j], bestIndices[j], bestValues[j - 1], bestIndices[j - 1]))
                {
                    const float v = bestValues[j]; bestValues[j] = bestValues[j - 1]; bestValues[j - 1] = v;
                    const int ix = bestIndices[j]; bestIndices[j] = bestIndices[j - 1]; bestIndices[j - 1] = ix;
                }
        }
    }
#pragma unroll
    for (int j = 0; j < 16; ++j)
    {
        values[j][threadIdx.x] = bestValues[j];
        indicesShared[j][threadIdx.x] = bestIndices[j];
    }
    heads[threadIdx.x] = 0;
    __syncthreads();
    for (int out = 0; out < k; ++out)
    {
        const int head = heads[threadIdx.x];
        float value = head < 16 ? values[head][threadIdx.x] : -INFINITY;
        int index = head < 16 ? indicesShared[head][threadIdx.x] : 0x7fffffff;
        int slot = threadIdx.x;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            const float otherValue = __shfl_xor_sync(0xffffffff, value, offset);
            const int otherIndex = __shfl_xor_sync(0xffffffff, index, offset);
            const int otherSlot = __shfl_xor_sync(0xffffffff, slot, offset);
            if (ts_q35_topk_before(otherValue, otherIndex, value, index))
            {
                value = otherValue;
                index = otherIndex;
                slot = otherSlot;
            }
        }
        if ((threadIdx.x & 31) == 0)
        {
            warpValues[threadIdx.x >> 5] = value;
            warpIndices[threadIdx.x >> 5] = index;
            warpSlots[threadIdx.x >> 5] = slot;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            int winner = 0;
            for (int i = 1; i < 8; ++i)
                if (ts_q35_topk_before(warpValues[i], warpIndices[i], warpValues[winner], warpIndices[winner]))
                    winner = i;
            idxOut[out] = warpIndices[winner];
            valOut[out] = warpValues[winner];
            heads[warpSlots[winner]]++;
        }
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(256) ts_q35_topk_partial_f32(
    const float* x, int n, int k, int partitions, int* idxOut, float* valOut)
{
    const int row = blockIdx.y, partition = blockIdx.x;
    const int start = (long long)n * partition / partitions;
    const int end = (long long)n * (partition + 1) / partitions;
    const long long output = ((long long)row * partitions + partition) * k;
    ts_q35_topk_partition(x + (long long)row * n + start, nullptr, end - start, start, k,
        idxOut + output, valOut + output);
}

extern "C" __global__ void __launch_bounds__(256) ts_q35_topk_merge_f32(
    const int* partialIdx, const float* partialVal, int k, int partitions, int* idxOut, float* valOut)
{
    const int row = blockIdx.x;
    const long long start = (long long)row * partitions * k;
    ts_q35_topk_partition(partialVal + start, partialIdx + start, partitions * k, 0, k,
        idxOut + (long long)row * k, valOut + (long long)row * k);
}
