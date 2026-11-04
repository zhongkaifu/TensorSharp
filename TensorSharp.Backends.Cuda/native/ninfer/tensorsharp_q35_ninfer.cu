// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// TensorSharp-owned NInfer numerical adaptation. No ggml source is included or modified.
// F32 buffers carry BF16-rounded values at the declared storage boundaries. Recurrent
// state and gate projections retain FP32. Arithmetic follows the published NInfer
// representation; independent component and model validation remain required.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

__device__ __forceinline__ float ts_ni_bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float ts_ni_sum(float x) {
#pragma unroll
    for (int d=16;d;d>>=1) x+=__shfl_xor_sync(0xffffffffu,x,d);
    return x;
}
__device__ __forceinline__ float ts_ni_silu(float x) { return x/(1.0f+expf(-x)); }
__device__ __forceinline__ float ts_ni_sigmoid(float x) { return 1.0f/(1.0f+expf(-x)); }
__device__ __forceinline__ float ts_ni_softplus(float x) { return x>20.0f?x:log1pf(expf(x)); }
__device__ __forceinline__ float ts_ni_fp8(uint8_t b) {
    __nv_fp8_e4m3 q; q.__x=b; return (float)q;
}
__device__ __forceinline__ float ts_ni_fp4(uint8_t b) {
    const float table[8]={0,0.5f,1,1.5f,2,3,4,6};
    float x=table[b&7]; return b&8?-x:x;
}
extern "C" __global__ void ts_ni_round(float* x, long long count) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) x[i]=ts_ni_bf(x[i]);
}
extern "C" __global__ void ts_ni_f32_bf16(const float* x,__nv_bfloat16* y,long long count) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) y[i]=__float2bfloat16_rn(x[i]);
}
extern "C" __global__ void ts_ni_embed_w8(const int8_t* codes,const __half* scales,
    const int* ids,float* out,int width,int vocab) {
    int row=blockIdx.x,id=ids[row];
    if(id<0||id>=vocab) { asm("trap;"); return; }
    for(int i=threadIdx.x;i<width;i+=blockDim.x)
        out[(long long)row*width+i]=ts_ni_bf((float)codes[(long long)id*width+i]*
            __half2float(scales[(long long)id*(width/32)+i/32]));
}
// effectiveWeight contains F32(stored BF16 + 1), or plain GDN weights.
// x and y may not alias: x receives the rounded residual and y its rounded norm.
extern "C" __global__ void ts_ni_add_norm(float* x,const float* delta,const float* weight,
    float* y,int width,float eps) {
    int row=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    float ss=0;
    for(int i=threadIdx.x;i<width;i+=blockDim.x) {
        long long j=(long long)row*width+i;
        float v=ts_ni_bf(x[j]+(delta?delta[j]:0.0f)); x[j]=v; ss+=v*v;
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
    for(int i=threadIdx.x;i<width;i+=blockDim.x) {
        long long j=(long long)row*width+i; y[j]=ts_ni_bf(x[j]*inv*weight[i]);
    }
}
// One warp owns two independent K16 groups. Each half-warp handles a group;
// paired lanes write TS's low8/high8 nibble convention, without requantizing weights.
extern "C" __global__ void ts_ni_quant_nvfp4(const float* x,uint8_t* codes,uint8_t* scales,
    float* rowScale,int k,int rows,float inputD) {
    long long group=((long long)blockIdx.x*blockDim.x+threadIdx.x)/16;
    int lane=threadIdx.x&15,groups=k/16;
    if(group>=(long long)rows*groups) return;
    int row=(int)(group/groups),g=(int)(group%groups);
    float v=x[(long long)row*k+g*16+lane],a=fabsf(v);
#pragma unroll
    for(int off=8;off;off>>=1) a=fmaxf(a,__shfl_xor_sync(0xffffffffu,a,off,16));
    uint8_t s=__nv_cvt_float_to_fp8(__fdiv_rn(inputD*a,6.0f),__NV_SATFINITE,__NV_E4M3);
    float q=s?__fdiv_rn(v*inputD,ts_ni_fp8(s)):0;
    float high=__shfl_xor_sync(0xffffffffu,q,8,16);
    if(lane<8) {
        // sm_120a lacks cvt.e2m1x2.f32. Compare exact binary midpoints
        // directly: no F16 conversion/double rounding and ties choose even codes.
        float a0=fabsf(q),a1=fabsf(high);
        unsigned lo=a0<=0.25f?0:a0<0.75f?1:a0<=1.25f?2:a0<1.75f?3:a0<=2.5f?4:a0<3.5f?5:a0<=5.0f?6:7;
        unsigned hi=a1<=0.25f?0:a1<0.75f?1:a1<=1.25f?2:a1<1.75f?3:a1<=2.5f?4:a1<3.5f?5:a1<=5.0f?6:7;
        lo|=(__float_as_uint(q)>>28)&8u; hi|=(__float_as_uint(high)>>28)&8u;
        codes[(long long)row*(k/2)+g*8+lane]=(uint8_t)(lo|(hi<<4));
    }
    if(lane==0) { scales[group]=s; if(g==0) rowScale[row]=1.0f; }
}
extern "C" __global__ void ts_ni_swiglu(const float* packed,float* out,int ff) {
    int i=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y;
    if(i>=ff) return;
    int gate=(i/8)*16+i%8;
    out[(long long)row*ff+i]=ts_ni_bf(ts_ni_silu(packed[(long long)row*2*ff+gate])*
        packed[(long long)row*2*ff+gate+8]);
}
// A16 route: weights are decoded to F32 with the source's scale*inverseD
// coefficient before the dot; no activation quantization or BF16 weight rounding.
// Four output rows per CTA; one warp computes each dot in 16-value groups.
extern "C" __global__ void ts_ni_project_a16(const uint8_t* codes,const uint8_t* scales,
    const float* input,float* output,int n,int k,int rows,int inputStride,int outputStride,float inverseD) {
    int lane=threadIdx.x&31,r=blockIdx.x*4+(threadIdx.x>>5),t=blockIdx.y;
    if(r>=n||t>=rows) return;
    float sum=0;
    for(int g=lane;g<k/16;g+=32) {
        float coeff=ts_ni_fp8(scales[(long long)r*(k/16)+g])*inverseD;
        const uint8_t* c=codes+(long long)r*(k/2)+g*8;
        const float* x=input+(long long)t*inputStride+g*16;
#pragma unroll
        for(int j=0;j<8;j++) {
            uint8_t b=c[j];
            sum=fmaf(ts_ni_fp4(b&15)*coeff,x[j],sum);
            sum=fmaf(ts_ni_fp4(b>>4)*coeff,x[j+8],sum);
        }
    }
    sum=ts_ni_sum(sum);
    if(!lane) output[(long long)t*outputStride+r]=sum;
}
extern "C" __global__ void ts_ni_gdn_conv(const float* packed,int packedWidth,int rowsPerSeq,
    int convDim,int dConv,const float* weights,float* const* states,float* qkv,float* raw,const int* count) {
    int ch=blockIdx.x*blockDim.x+threadIdx.x,t=blockIdx.y,seq=blockIdx.z;
    if(ch>=convDim) return;
    long long row=(long long)seq*rowsPerSeq+t;
    if(count&&t>=count[seq]) { qkv[row*convDim+ch]=0; if(raw)raw[row*convDim+ch]=0; return; }
    float acc=0;
    for(int j=0;j<dConv;j++) {
        int src=t-(dConv-1)+j;
        float v=src>=0?packed[((long long)seq*rowsPerSeq+src)*packedWidth+ch]
            :states[seq][(long long)(dConv-1+src)*convDim+ch];
        // Model builder transposes original tap-major weights into channel-major.
        acc=fmaf(weights[(long long)ch*dConv+j],v,acc);
    }
    qkv[row*convDim+ch]=ts_ni_bf(ts_ni_silu(acc));
    if(raw)raw[row*convDim+ch]=packed[row*packedWidth+ch];
}
extern "C" __global__ void ts_ni_gdn_conv_state(const float* src,int srcWidth,long long layerStride,
    int rowsPerSeq,const int* count,int convDim,int dConv,float* const* states,int ptrStride) {
    int ch=blockIdx.x*blockDim.x+threadIdx.x,seq=blockIdx.y,layer=blockIdx.z;
    float* state=states[(long long)layer*ptrStride+seq];
    if(ch>=convDim||!state) return;
    int rows=count?count[seq]:rowsPerSeq;
    const float* input=src+layer*layerStride+(long long)seq*rowsPerSeq*srcWidth;
    for(int i=0;i<dConv-1;i++) {
        int r=rows-(dConv-1)+i;
        state[(long long)i*convDim+ch]=r>=0?input[(long long)r*srcWidth+ch]
            :state[(long long)(dConv-1+r)*convDim+ch];
    }
}
// Separate AB projection is [alpha|beta], unlike GGUF's fused [beta|alpha].
extern "C" __global__ void ts_ni_gdn_prep(float* qkv,const float* ab,int abWidth,int rows,
    int hk,int hv,int convDim,const float* dtBias,const float* aLog,float* g,float* beta,
    const int* count,int rowsPerSeq) {
    long long unit=((long long)blockIdx.x*blockDim.x+threadIdx.x)>>5;
    int lane=threadIdx.x&31,units=2*hk+hv;
    if(unit>=(long long)rows*units) return;
    int row=(int)(unit/units),h=(int)(unit%units);
    bool pad=count&&row%rowsPerSeq>=count[row/rowsPerSeq];
    if(h<2*hk) {
        float* p=qkv+(long long)row*convDim+h*128;
        float v[4],ss=0;
#pragma unroll
        for(int j=0;j<4;j++) { v[j]=p[lane*4+j]; ss+=v[j]*v[j]; }
        ss=ts_ni_sum(ss); float inv=rsqrtf(ss+1.0e-6f);
#pragma unroll
        for(int j=0;j<4;j++)p[lane*4+j]=pad?0:v[j]*inv;
    } else if(!lane) {
        h-=2*hk; const float* p=ab+(long long)row*abWidth;
        g[(long long)row*hv+h]=pad?0:-expf(aLog[h])*ts_ni_softplus(p[h]+dtBias[h]);
        beta[(long long)row*hv+h]=pad?0:ts_ni_sigmoid(p[hv+h]);
    }
}
// State is [valueHead][valueCoordinate][keyCoordinate], matching Q35's slot arena.
// Four warps own four value coordinates each, with contiguous four-key lanes.
template<bool WriteOut,bool WriteState>
__device__ void ts_ni_scan(const float* qkv,int convDim,int hk,int hv,int rowsPerSeq,int rows,
    const float* g,const float* beta,float* state,float* output,int valueBlock) {
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,h=blockIdx.x,seq=blockIdx.y;
    int value0=valueBlock*16+warp*4,key0=lane*4,kh=h/(hv/hk);
    float s[4][4];
#pragma unroll
    for(int v=0;v<4;v++)
#pragma unroll
        for(int j=0;j<4;j++)s[v][j]=state[((long long)h*128+value0+v)*128+key0+j];
    for(int t=0;t<rows;t++) {
        long long row=(long long)seq*rowsPerSeq+t;
        const float* k=qkv+row*convDim+(hk+kh)*128;
        const float* q=qkv+row*convDim+kh*128;
        const float* value=qkv+row*convDim+2*hk*128+h*128;
        float decay=expf(g[row*hv+h]),b=beta[row*hv+h];
        float vlocal=lane<4?value[value0+lane]:0;
#pragma unroll
        for(int v=0;v<4;v++) {
            float dot=0;
#pragma unroll
            for(int j=0;j<4;j++)dot+=s[v][j]*k[key0+j];
            dot=ts_ni_sum(dot);
            float delta=b*(__shfl_sync(0xffffffffu,vlocal,v)-decay*dot);
#pragma unroll
            for(int j=0;j<4;j++)s[v][j]=decay*s[v][j]+delta*k[key0+j];
        }
        if constexpr(WriteOut) {
            float y=0;
#pragma unroll
            for(int v=0;v<4;v++) {
                float dot=0;
#pragma unroll
                for(int j=0;j<4;j++)dot+=s[v][j]*q[key0+j];
                dot=ts_ni_sum(dot); if(lane==v)y=dot;
            }
            if(lane<4)output[(row*hv+h)*128+value0+lane]=ts_ni_bf(y*0.08838834764831845f);
        }
    }
    if constexpr(WriteState) {
#pragma unroll
        for(int v=0;v<4;v++)
#pragma unroll
            for(int j=0;j<4;j++)state[((long long)h*128+value0+v)*128+key0+j]=s[v][j];
    }
}
extern "C" __global__ __launch_bounds__(128) void ts_ni_gdn_scan(const float* qkv,int convDim,
    int hk,int hv,int rowsPerSeq,const float* g,const float* beta,float* const* states,float* out,
    const int* count,int verify) {
    int rows=count?count[blockIdx.y]:rowsPerSeq;
    if(verify)ts_ni_scan<true,false>(qkv,convDim,hk,hv,rowsPerSeq,rows,g,beta,states[blockIdx.y],out,blockIdx.z);
    else ts_ni_scan<true,true>(qkv,convDim,hk,hv,rowsPerSeq,rows,g,beta,states[blockIdx.y],out,blockIdx.z);
}
extern "C" __global__ __launch_bounds__(128) void ts_ni_gdn_replay(const float* qkv,long long qkvStride,
    int convDim,int hk,int hv,int rowsPerSeq,const int* kept,const float* g,const float* beta,
    long long gStride,float* const* states,int ptrStride) {
    int layer=blockIdx.z/8; float* state=states[(long long)layer*ptrStride+blockIdx.y];
    if(!state)return;
    ts_ni_scan<false,true>(qkv+layer*qkvStride,convDim,hk,hv,rowsPerSeq,kept[blockIdx.y],
        g+layer*gStride,beta+layer*gStride,state,nullptr,blockIdx.z%8);
}
extern "C" __global__ void ts_ni_gdn_out(const float* o,const float* packed,int packedWidth,
    int convDim,int rows,int hv,const float* norm,float eps,float* y) {
    long long warp=((long long)blockIdx.x*blockDim.x+threadIdx.x)>>5;
    int lane=threadIdx.x&31;
    if(warp>=(long long)rows*hv)return;
    long long row=warp/hv; int h=(int)(warp%hv);
    const float* input=o+(row*hv+h)*128;
    float v[4],ss=0;
#pragma unroll
    for(int j=0;j<4;j++) { v[j]=input[lane*4+j]; ss+=v[j]*v[j]; }
    ss=ts_ni_sum(ss); float inv=rsqrtf(ss/128.0f+eps);
    const float* z=packed+row*packedWidth+convDim+h*128;
#pragma unroll
    for(int j=0;j<4;j++) {
        int i=lane*4+j;
        y[(row*hv+h)*128+i]=ts_ni_bf(v[j]*inv*norm[i]*ts_ni_silu(z[i]));
    }
}
