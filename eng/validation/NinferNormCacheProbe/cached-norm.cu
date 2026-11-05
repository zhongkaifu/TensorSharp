// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Independent TensorSharp candidate derived from its own ts_ni_add_norm helper.
// Register-reuse idea reference: gpillon/ignis
// e600a7d9e0d5921d767a6cb464373583b3cdcaf4,
// kernel/vendor/src/ops/kernel/rmsnorm.cuh:157-191 (Apache-2.0).
// No Ignis source is copied or included; the original helper and ggml are unchanged.
#include <cuda_bf16.h>

namespace {
__device__ __forceinline__ float ts_ni_bf(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}
__device__ __forceinline__ float ts_ni_sum(float x) {
#pragma unroll
    for (int d=16;d;d>>=1) x+=__shfl_xor_sync(0xffffffffu,x,d);
    return x;
}
}

// Caller contract: grid=(8,1,1), block=(256,1,1), width=5120, positive finite eps.
// x and y do not overlap. delta may be null; no new restrict/alias assumptions.
// effectiveWeight is F32(stored BF16 + 1), not BF16-rounded again.
// Keep width as a runtime ABI value in the division and row stride. In particular,
// a width==5120 guard here could allow constant-folding of the surviving division.
extern "C" __global__ void ts_ni_add_norm_t8_cached(float* x,const float* delta,
    const float* weight,float* y,int width,float eps) {
    int row=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    float values[20];
    float ss=0;
#pragma unroll
    for (int k=0;k<20;k++) {
        int i=threadIdx.x+k*blockDim.x;
        long long j=(long long)row*width+i;
        float v=ts_ni_bf(x[j]+(delta?delta[j]:0.0f)); x[j]=v; ss+=v*v;
        values[k]=v;
    }
    ss=ts_ni_sum(ss);
    __shared__ float partial[32], inv;
    if(!lane) partial[warp]=ss;
    __syncthreads();
    if(!warp) {
        ss=lane<(blockDim.x>>5)?partial[lane]:0; ss=ts_ni_sum(ss);
        if(!lane) inv=rsqrtf(ss/(float)width+eps);
    }
    __syncthreads();
#pragma unroll
    for (int k=0;k<20;k++) {
        int i=threadIdx.x+k*blockDim.x;
        long long j=(long long)row*width+i;
        y[j]=ts_ni_bf(values[k]*inv*weight[i]);
    }
}
