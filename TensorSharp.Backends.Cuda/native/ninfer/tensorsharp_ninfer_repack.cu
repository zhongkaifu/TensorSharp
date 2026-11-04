// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Pure encoded-byte permutation. No FP conversion, requantization, calibration
// arithmetic, ggml source, or changes to existing inference kernels.
#include <stdint.h>

__device__ __forceinline__ int ts_ni_repack_source_row(int row,int rows,int mapping) {
    if(mapping==1)return (row/16)*8+row%8+(row%16>=8?rows/2:0);
    if(mapping==2) {
        if(row<12288)return (row/512)*256+row%256+(row%512>=256?7168:0);
        if(row<13312)return 6144+row-12288;
    }
    return row;
}
extern "C" __global__ void ts_ni_repack_nvfp4(
    const uint8_t* __restrict__ original,uint8_t* __restrict__ destination,int rows,int columns,int mapping) {
    const long long groups=columns/16;
    const long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=(long long)rows*groups)return;
    const int row=(int)(index/groups),group=(int)(index%groups);
    const int sourceRow=ts_ni_repack_source_row(row,rows,mapping);
    const long long codeBytes=(long long)rows*columns/2;
    const uint2 packed=*reinterpret_cast<const uint2*>(original+(long long)sourceRow*(columns/2)+(long long)group*8);
    // Original bytes hold adjacent-K nibbles. Form the even and odd output
    // bytes, then interleave them into TS low[0..7]/high[8..15] storage.
    const uint32_t even=(packed.x&0x0f0f0f0fu)|((packed.y&0x0f0f0f0fu)<<4);
    const uint32_t odd=((packed.x>>4)&0x0f0f0f0fu)|(packed.y&0xf0f0f0f0u);
    uint2 output;
    output.x=__byte_perm(even,odd,0x5140);
    output.y=__byte_perm(even,odd,0x7362);
    *reinterpret_cast<uint2*>(destination+(long long)row*(columns/2)+(long long)group*8)=output;
    const int inner=sourceRow%128;
    const long long sourceScale=((long long)(sourceRow/128)*(columns/64)+group/4)*512
        +(inner%32)*16+(inner/32)*4+group%4;
    destination[codeBytes+index]=original[codeBytes+sourceScale];
    // The original FP32 divisor trailer is deliberately not copied or consumed.
    // Its validated bits remain the projection descriptor's arithmetic policy.
}
