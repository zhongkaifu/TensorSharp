// Copyright (c) Zhongkai Fu. All rights reserved.
// TensorSharp-owned instantiation and representation bridge. The unchanged
// upstream/ files retain their original Apache-2.0 licenses and notices.
// Standalone candidate: no engine/default selects this module.
#include "upstream/kernel/vendor/src/ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cuh"
#include "upstream/kernel/vendor/src/ops/linear_attention/gated_delta_net/chunked/state_passing.cuh"
#include "upstream/kernel/vendor/src/ops/linear_attention/gated_delta_net/chunked/output.cuh"
#include <cstddef>
#include <type_traits>

using ninfer::ops::detail::gated_delta_net::head_map;
static_assert(sizeof(void*)==8);
static_assert(sizeof(head_map)==20&&alignof(head_map)==4);
static_assert(offsetof(head_map,H_qk)==0&&offsetof(head_map,H_v)==4&&offsetof(head_map,group_magic)==8);
static_assert(std::is_trivially_copyable_v<head_map>);
namespace chunk = ninfer::ops::detail::gated_delta_net::chunked;
static_assert(chunk::prepare_wy_wu::kernel_dims<64,32>::SMEM_FLOATS*sizeof(float)==29440);
static_assert(chunk::state_passing::smem_layout<16>::SMEM_FLOATS*sizeof(float)==45312);
static_assert(chunk::output::kernel_dims::SMEM_BYTES==24576);

// Instantiate the actual original global kernels. No private copy, wrapper
// body, arithmetic edit or alternate compilation of upstream source is used.
namespace ninfer::ops::detail::gated_delta_net::chunked::prepare_wy_wu {
template __global__ void prepare_wy_wu_kernel<64,32,8>(
    const __nv_bfloat16*,const __nv_bfloat16*,const float*,const float*,
    __nv_bfloat16*,__nv_bfloat16*,float*,head_map);
}
namespace ninfer::ops::detail::gated_delta_net::chunked::state_passing {
template __global__ void state_passing_kernel<16>(
    const __nv_bfloat16*,const __nv_bfloat16*,const __nv_bfloat16*,const float*,const float*,
    __nv_bfloat16*,__nv_bfloat16*,float*,head_map,int);
}
namespace ninfer::ops::detail::gated_delta_net::chunked::output {
template __global__ void output_kernel<false>(
    const __nv_bfloat16*,const __nv_bfloat16*,const __nv_bfloat16*,const float*,
    const __nv_bfloat16*,__nv_bfloat16*,head_map,float,int);
template __global__ void output_kernel<true>(
    const __nv_bfloat16*,const __nv_bfloat16*,const __nv_bfloat16*,const float*,
    const __nv_bfloat16*,__nv_bfloat16*,head_map,float,int);
}

// The incoming packed Q/K are already L2-normalized F32 values. Chunked NInfer
// materializes normalized Q/K in BF16 before the three original stages. V was
// stored BF16 by the convolution. This bridge applies only that representation
// boundary and splits the three contiguous ranges; it does not renormalize.
extern "C" __global__ void ts_ni_chunk_pack(
    const float* qkv,__nv_bfloat16* q,__nv_bfloat16* k,__nv_bfloat16* v,int rows,int stride) {
    constexpr int qk=16*128,value=48*128,total=2*qk+value;
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(long long)rows*total)return;
    int row=(int)(i/total),column=(int)(i%total);
    __nv_bfloat16 x=__float2bfloat16_rn(qkv[(long long)row*stride+column]);
    if(column<qk)q[(long long)row*qk+column]=x;
    else if(column<2*qk)k[(long long)row*qk+column-qk]=x;
    else v[(long long)row*value+column-2*qk]=x;
}
extern "C" __global__ void ts_ni_chunk_unpack(const __nv_bfloat16* input,float* output,long long count) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)output[i]=__bfloat162float(input[i]);
}
