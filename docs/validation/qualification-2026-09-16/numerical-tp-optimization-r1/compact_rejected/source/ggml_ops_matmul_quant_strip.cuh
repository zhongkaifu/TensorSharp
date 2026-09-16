// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Included only by ggml_ops_matmul_precision.cu. Weight strips retain the
// unsplit MMQ stream-K partition and reduction order. Upstream tile arithmetic
// is reused unchanged; unowned virtual row tiles do not read weights or write
// output. This prevents a shape-dependent gate/up rounding difference from
// crossing the down projection's activation quantization boundary.
#pragma once
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmid.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml_ops_matmul_quant_tile.cuh"
namespace {
struct quant_strip_args {
    const char * weights;
    const int * input;
    const int32_t * ids;
    const int32_t * bounds;
    float * out;
    float * partial;
    int full_rows, first, rows, row_stride, expert_stride;
    int selected_rows, tokens, experts;
    uint3 blocks, ntx;
    const char * weights2 = nullptr;
    float * out2 = nullptr, * partial2 = nullptr;
};

template<ggml_type type, int J,bool Compact>
__launch_bounds__(Compact?64:ggml_cuda_mmq_get_nthreads(type, J, false), Compact?2:ggml_cuda_mmq_get_occupancy(type, J, false))
__global__ void quant_strip_main(quant_strip_args a) {
    constexpr int I=ggml_cuda_mmq_get_I(type,J,false);
    constexpr int warp=ggml_cuda_get_physical_warp_size();
    constexpr int nwarps=Compact?2:ggml_cuda_mmq_get_nthreads(type,J,false)/warp;
    constexpr int R=Compact?32:I,groups=I/R;
    if (blockIdx.y/groups) { a.weights=a.weights2; a.out=a.out2; a.partial=a.partial2; }
    const int subgroup=blockIdx.y%groups;
    a.partial+=size_t(subgroup)*gridDim.x*J*R;
    constexpr int per_iter=ggml_cuda_mmq_get_K_vram(type,J,false)/ggml_cuda_type_traits<type>::qk;
    extern __shared__ int selected[];
    const int tiles=(a.full_rows/I)*a.ntx.z*a.experts;
    int begin=int64_t(blockIdx.x)*tiles*a.blocks.z/gridDim.x;
    int end=int64_t(blockIdx.x+1)*tiles*a.blocks.z/gridDim.x;
    begin-=fastmodulo(begin,a.blocks)%per_iter;
    end-=fastmodulo(end,a.blocks)%per_iter;
    while(begin<end) {
        const int kb=fastmodulo(begin,a.blocks);
        const int stop=min(a.blocks.z,uint32_t(kb+end-begin));
        const int tile=fastdiv(begin,a.blocks);
        const int jt=fastmodulo(tile,a.ntx);
        const int expert=fastdiv(tile,a.ntx)%a.experts;
        const int global_row=fastdiv(tile,a.ntx)/a.experts*I;
        const int local_row=global_row-a.first+subgroup*R;
        const int low=a.bounds[expert], columns=a.bounds[expert+1]-low;
        if(local_row>=0 && local_row<a.rows && jt*J<columns) {
            const bool partial=stop!=int(a.blocks.z);
            __syncthreads();
            for(int j=threadIdx.y*warp+threadIdx.x;j<J;j+=nwarps*warp)
                selected[j]=partial?j:(jt*J+j<columns?a.ids[low+jt*J+j]:0);
            __syncthreads();
            const int xo=expert*a.expert_stride+local_row*a.row_stride;
            const int yo=(low+jt*J)*(sizeof(block_q8_1_mmq)/sizeof(int));
            if constexpr(Compact) {
                if(partial)tsg_quant_strip_compact_tile<type,J,true>(a.weights,xo,a.input+yo,selected,a.out+local_row,a.partial,
                    a.row_stride,a.selected_rows,a.rows,columns-jt*J-1,kb,stop);
                else tsg_quant_strip_compact_tile<type,J,false>(a.weights,xo,a.input+yo,selected,a.out+local_row,a.partial,
                    a.row_stride,a.selected_rows,a.rows,columns-jt*J-1,kb,stop);
            } else {
            if(partial)
                mul_mat_q_process_tile<type,J,false,true>(a.weights,xo,a.input+yo,selected,a.out+local_row,a.partial,nullptr,
                    a.row_stride,a.selected_rows,a.rows,a.rows-local_row-1,columns-jt*J-1,kb,stop);
            else
                mul_mat_q_process_tile<type,J,false,false>(a.weights,xo,a.input+yo,selected,a.out+local_row,a.partial,nullptr,
                    a.row_stride,a.selected_rows,a.rows,a.rows-local_row-1,columns-jt*J-1,kb,stop);
            }
        }
        begin+=stop-kb;
    }
}

template<ggml_type type,int J,bool Compact>
__launch_bounds__(ggml_cuda_mmq_get_nthreads(type,J,false)/2,1)
__global__ void quant_strip_fixup(quant_strip_args a) {
    constexpr int I=ggml_cuda_mmq_get_I(type,J,false);
    constexpr int warp=ggml_cuda_get_physical_warp_size();
    constexpr int nwarps=(ggml_cuda_mmq_get_nthreads(type,J,false)/2)/warp;
    constexpr int R=Compact?32:I,groups=I/R;
    if (blockIdx.z/groups) { a.out=a.out2; a.partial=a.partial2; }
    const int subgroup=blockIdx.z%groups;
    a.partial+=size_t(subgroup)*gridDim.x*J*R;
    constexpr int per_iter=ggml_cuda_mmq_get_K_vram(type,J,false)/ggml_cuda_type_traits<type>::qk;
    const int tiles=(a.full_rows/I)*a.ntx.z*a.experts;
    int begin=int64_t(blockIdx.x)*tiles*a.blocks.z/gridDim.x;
    int end=int64_t(blockIdx.x+1)*tiles*a.blocks.z/gridDim.x;
    begin-=fastmodulo(begin,a.blocks)%per_iter;
    end-=fastmodulo(end,a.blocks)%per_iter;
    if(begin==end || fastmodulo(begin,a.blocks)==0 ||
        (fastdiv(begin,a.blocks)==fastdiv(end,a.blocks)&&fastmodulo(end,a.blocks)!=0))return;
    const int tile=fastdiv(begin,a.blocks);
    const int jt=fastmodulo(tile,a.ntx);
    const int expert=fastdiv(tile,a.ntx)%a.experts;
    const int row=fastdiv(tile,a.ntx)/a.experts*I-a.first+subgroup*R;
    const int low=a.bounds[expert],columns=a.bounds[expert+1]-low;
    if(row<0 || row>=a.rows || jt*J>=columns)return;
    const int i=blockIdx.y*warp+threadIdx.x;
    float sum[J/nwarps]={};
    int previous=blockIdx.x-1, previous_end=begin;
    while(true) {
        int previous_begin=int64_t(previous)*tiles*a.blocks.z/gridDim.x;
        previous_begin-=fastmodulo(previous_begin,a.blocks)%per_iter;
        if(previous_begin==previous_end){--previous;previous_end=previous_begin;continue;}
        for(int j0=0;j0<J;j0+=nwarps)
            sum[j0/nwarps]+=a.partial[previous*(J*R)+(j0+threadIdx.y)*R+i];
        if(fastmodulo(previous_begin,a.blocks)==0 || fastdiv(previous_begin,a.blocks)<fastdiv(begin,a.blocks))break;
        --previous;previous_end=previous_begin;
    }
    for(int j0=0;j0<J;j0+=nwarps) {
        const int j=j0+threadIdx.y;
        if(jt*J+j>=columns)return;
        a.out[a.ids[low+jt*J+j]*a.rows+row+i]+=sum[j0/nwarps];
    }
}


int quant_strip_width(ggml_type type, int tokens, int device) {
    const auto & info = ggml_cuda_info().devices[device];
    int width = 0, tiles = INT_MAX;
    for (int j = 8; j <= 128 && tiles > 1; j += 8) {
        const auto config = ggml_cuda_mmq_get_config(type, j, false, info.cc);
        if (config.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(config, info.cc) > info.smpbo) continue;
        const int count = (tokens + j - 1) / j;
        if (count < tiles) { width = j; tiles = count; }
    }
    return width;
}
int quant_strip_blocks(const quant_strip_args & a, const ggml_cuda_mmq_config & config, int device) {
    const int nsm = ggml_cuda_info().devices[device].nsm;
    const int tiles = (a.full_rows / config.I) * a.ntx.z * a.experts;
    const int waves = (tiles + nsm - 1) / nsm;
    return int64_t(100) * tiles / (int64_t(nsm) * waves) >= 90 ? tiles : nsm;
}
template<ggml_type type, int J,bool Compact>
void launch_quant_strip(int device, cudaStream_t stream, quant_strip_args a) {
    const int cc = ggml_cuda_info().devices[device].cc;
    const auto config = ggml_cuda_mmq_get_config(type, J, false, cc);
    GGML_ASSERT(config.type != GGML_TYPE_COUNT && config.stream_k);
    const int blocks = quant_strip_blocks(a, config, device);
    const int tiles = (a.full_rows / config.I) * a.ntx.z * a.experts;
    auto local=config;
    if constexpr(Compact){local.I=32;local.nthreads=64;}
    const int groups=config.I/local.I;
    const int shared = mmq_get_nbytes_shared(local, cc);
    CUDA_SET_SHARED_MEMORY_LIMIT((quant_strip_main<type, J,Compact>), shared);
    quant_strip_main<type, J,Compact><<<dim3(blocks,groups*(a.weights2?2:1)), dim3(32, local.nthreads/32), shared, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
    if (tiles % blocks)
        quant_strip_fixup<type, J,Compact><<<dim3(blocks, local.I/32,groups*(a.weights2?2:1)), dim3(32, config.nthreads/64), 0, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());
}
template<ggml_type type>
void dispatch_quant_strip(int device, cudaStream_t stream, int width, quant_strip_args a) {
    const int cc=ggml_cuda_info().devices[device].cc;
    const bool compact=a.rows<=512 && cc>=GGML_CUDA_CC_TURING && cc<1000 &&
        ggml_cuda_mmq_get_config(type,width,false,cc).I==128;
    switch (width) {
#define TSG_QUANT_STRIP_CASE(J) case J: if(compact)launch_quant_strip<type,J,true>(device,stream,a);else launch_quant_strip<type,J,false>(device,stream,a); break;
        TSG_QUANT_STRIP_CASE(8)
        TSG_QUANT_STRIP_CASE(16)
        TSG_QUANT_STRIP_CASE(24)
        TSG_QUANT_STRIP_CASE(32)
        TSG_QUANT_STRIP_CASE(40)
        TSG_QUANT_STRIP_CASE(48)
        TSG_QUANT_STRIP_CASE(56)
        TSG_QUANT_STRIP_CASE(64)
        TSG_QUANT_STRIP_CASE(72)
        TSG_QUANT_STRIP_CASE(80)
        TSG_QUANT_STRIP_CASE(88)
        TSG_QUANT_STRIP_CASE(96)
        TSG_QUANT_STRIP_CASE(104)
        TSG_QUANT_STRIP_CASE(112)
        TSG_QUANT_STRIP_CASE(120)
        TSG_QUANT_STRIP_CASE(128)

#undef TSG_QUANT_STRIP_CASE
        default: GGML_ABORT("Unsupported TensorSharp quantized strip tile width");
    }
}
} // namespace
