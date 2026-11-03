// Copyright (c) Zhongkai Fu. All rights reserved.
// Independently authored TensorSharp W8G32 kernels. No Ignis source copied.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
static __device__ __forceinline__ uint32_t PairBf16(int a,int b){
    // Every signed-int8 value is exactly representable in BF16. Taking the
    // high FP32 half avoids an SDK-version-dependent pair reinterpret intrinsic.
    return (__float_as_uint(float(a))>>16)|(__float_as_uint(float(b))&0xffff0000u);
}
static __device__ __forceinline__ uint32_t LoadX(const __nv_bfloat16* x,int t,int k,int c,int tokens){
    if(t>=tokens)return 0;
    return reinterpret_cast<const uint32_t*>(x+(int64_t)t*k+c)[0];
}
static __device__ __forceinline__ void Mma(float* d,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
       "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
       : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
       : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
extern "C" __global__ void ts_w8_cast(const float* x,__nv_bfloat16* y,int64_t n){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=__float2bfloat16_rn(x[i]);
}
// Independent double precision literal reference. One thread owns one output;
// mathematical products have only8+11+8 significant bits before summation.
extern "C" __global__ void ts_w8_reference(const int8_t* w,const __half* scales,const __nv_bfloat16* x,
    double* y,double* sumabs,int rows,int k,int kp,int tokens){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(int64_t)rows*tokens)return;
    int r=int(i%rows),t=int(i/rows);double z=0,a=0;
    for(int c=0;c<k;c++){
        double p=double(w[(int64_t)r*kp+c])*double(__half2float(scales[(int64_t)r*(kp/32)+c/32]))*double(__bfloat162float(x[(int64_t)t*k+c]));
        z+=p;a+=fabs(p);
    }
    y[i]=z;sumabs[i]=a;
}
// Functional same-dtype SIMT reference, not an optimized leader baseline.
extern "C" __global__ void ts_w8_simt(const int8_t* w,const __half* scales,const __nv_bfloat16* x,
    __nv_bfloat16* y,int rows,int k,int kp,int tokens){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(int64_t)rows*tokens)return;
    int r=int(i%rows),t=int(i/rows);float z=0;
    for(int g=0;g<k/32;g++){
        float dot=0;
        for(int j=0;j<32;j++)dot=fmaf(float(w[(int64_t)r*kp+g*32+j]),__bfloat162float(x[(int64_t)t*k+g*32+j]),dot);
        z=fmaf(dot,__half2float(scales[(int64_t)r*(kp/32)+g]),z);
    }
    y[i]=__float2bfloat16_rn(z);
}
template<int T>
__device__ __forceinline__ void W8Tc(const int8_t* w,const __half* scales,const __nv_bfloat16* x,__nv_bfloat16* y,int rows,int k,int kp){
    constexpr int S=4,NT=(T+7)/8;
    int lane=threadIdx.x&31,warp=threadIdx.x>>5,g=lane>>2,q=lane&3,row0=int(blockIdx.x)*16;
    int ra=row0+g,rb=ra+8;
    float acc[NT][4];
#pragma unroll
    for(int n=0;n<NT;n++)for(int e=0;e<4;e++)acc[n][e]=0;
    for(int block=warp;block<k/32;block+=S){
        float part[NT][4];
#pragma unroll
        for(int n=0;n<NT;n++)for(int e=0;e<4;e++)part[n][e]=0;
#pragma unroll
        for(int h=0;h<2;h++){
            int col=block*32+h*16+q*2;
            uint32_t a0=ra<rows?PairBf16(w[(int64_t)ra*kp+col],w[(int64_t)ra*kp+col+1]):0;
            uint32_t a1=rb<rows?PairBf16(w[(int64_t)rb*kp+col],w[(int64_t)rb*kp+col+1]):0;
            uint32_t a2=ra<rows?PairBf16(w[(int64_t)ra*kp+col+8],w[(int64_t)ra*kp+col+9]):0;
            uint32_t a3=rb<rows?PairBf16(w[(int64_t)rb*kp+col+8],w[(int64_t)rb*kp+col+9]):0;
#pragma unroll
            for(int n=0;n<NT;n++)Mma(part[n],a0,a1,a2,a3,LoadX(x,n*8+g,k,col,T),LoadX(x,n*8+g,k,col+8,T));
        }
        float sa=ra<rows?__half2float(scales[(int64_t)ra*(kp/32)+block]):0;
        float sb=rb<rows?__half2float(scales[(int64_t)rb*(kp/32)+block]):0;
#pragma unroll
        for(int n=0;n<NT;n++){
            acc[n][0]=fmaf(part[n][0],sa,acc[n][0]);acc[n][1]=fmaf(part[n][1],sa,acc[n][1]);
            acc[n][2]=fmaf(part[n][2],sb,acc[n][2]);acc[n][3]=fmaf(part[n][3],sb,acc[n][3]);
        }
    }
    __shared__ float partial[S][NT][32][4];
#pragma unroll
    for(int n=0;n<NT;n++)for(int e=0;e<4;e++)partial[warp][n][lane][e]=acc[n][e];
    __syncthreads();
    if(warp==0){
#pragma unroll
        for(int n=0;n<NT;n++){
#pragma unroll
            for(int s=1;s<S;s++)for(int e=0;e<4;e++)acc[n][e]+=partial[s][n][lane][e];
            int ta=n*8+q*2,tb=ta+1;
            if(ra<rows&&ta<T)y[(int64_t)ta*rows+ra]=__float2bfloat16_rn(acc[n][0]);
            if(ra<rows&&tb<T)y[(int64_t)tb*rows+ra]=__float2bfloat16_rn(acc[n][1]);
            if(rb<rows&&ta<T)y[(int64_t)ta*rows+rb]=__float2bfloat16_rn(acc[n][2]);
            if(rb<rows&&tb<T)y[(int64_t)tb*rows+rb]=__float2bfloat16_rn(acc[n][3]);
        }
    }
}
#define TS_W8(T) extern "C" __global__ void ts_w8_tc_##T(const int8_t* w,const __half* s,const __nv_bfloat16* x,__nv_bfloat16* y,int r,int k,int kp){ W8Tc<T>(w,s,x,y,r,k,kp); }
// Device entry wrappers are generated by build.py, directly instantiating a
// device implementation: a __global__ is never called as a device function.
TS_W8(1) TS_W8(7) TS_W8(8) TS_W8(14) TS_W8(16) TS_W8(28) TS_W8(32) TS_W8(33)

// Each block reports both a checksum and number of words read. This L2 hint is
// an eviction control for all data, not a guarantee of physically cold weights.
extern "C" __global__ void ts_w8_sweep(const uint32_t* data,uint64_t words,uint64_t* result){
    uint64_t sum=0,count=0;
    for(uint64_t i=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;i<words;i+=(uint64_t)gridDim.x*blockDim.x){
        uint32_t v;asm volatile("ld.global.cg.u32 %0,[%1];":"=r"(v):"l"(data+i):"memory");sum+=v;++count;
    }
    __shared__ uint64_t sums[256],counts[256];sums[threadIdx.x]=sum;counts[threadIdx.x]=count;__syncthreads();
    for(int d=128;d;d>>=1){if(threadIdx.x<d){sums[threadIdx.x]+=sums[threadIdx.x+d];counts[threadIdx.x]+=counts[threadIdx.x+d];}__syncthreads();}
    if(!threadIdx.x){result[2*blockIdx.x]=sums[0];result[2*blockIdx.x+1]=counts[0];}
}
