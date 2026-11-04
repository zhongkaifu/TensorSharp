// Copyright (c) Zhongkai Fu. All rights reserved.
// Derived portions licensed under Apache-2.0; see draft/upstream/{LICENSE,NOTICE,manifest.json}.
// TensorSharp-owned standalone entrypoints adapted from gpillon/ignis
// e600a7d9e0d5921d767a6cb464373583b3cdcaf4. Original headers are unchanged.
// Uses ORIGINAL adjacent-K NVFP4 codes, m128x4-swizzled scales and actual BF16 buffers.
// No GGML source, A4 activation quantization, weight conversion or serving hook.
#include "draft/upstream/kernel/vendor/src/ops/linear/nvfp4/nvfp4_small_t.cuh"
#include "draft/upstream/kernel/vendor/src/ops/kernel/bidirectional_gqa_attention.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace ts_nidf {
using namespace ninfer::ops;
using namespace ninfer::ops::detail;
template<int N,int K,int T> struct LinearConfig {
 using Geometry=Nvfp4GemvGeometry<N,K>;
 using Schedule=typename Nvfp4LinearSmallTProductionSchedule<Geometry,T>::Type;
 static constexpr int Threads=Schedule::kThreads, MinBlocks=Schedule::kMinBlocksPerSm;
};
template<int N,int K> struct LinearConfig<N,K,1> {
 using Geometry=Nvfp4GemvGeometry<N,K>;
 using Schedule=typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
 static constexpr int Threads=Schedule::kThreads, MinBlocks=Schedule::kMinBlocksPerSm;
};
template<int N,int K,int T>
__device__ __forceinline__ void linear(const uint8_t* codes,const uint8_t* scales,
 const __nv_bfloat16* input,__nv_bfloat16* output,float inverseD) {
 using C=LinearConfig<N,K,T>; using G=typename C::Geometry; using S=typename C::Schedule;
 constexpr int CtasPerTile=128/S::kRowsPerCta;
 int m_tile=(int)blockIdx.x/CtasPerTile, cta=(int)blockIdx.x-m_tile*CtasPerTile;
 int rmod_base=cta*(S::kRowsPerCta/4),lane=(int)threadIdx.x&31,warp=(int)threadIdx.x>>5;
 int parent[S::kRowsPerWarp];
#pragma unroll
 for(int r=0;r<S::kRowsPerWarp;r++) {
  int flat=warp*S::kRowsPerWarp+r;
  parent[r]=m_tile*128+rmod_base+flat/4+(flat&3)*32;
 }
 if constexpr(T==1) {
  __shared__ Nvfp4GemvSharedStorage<G,S> shared;
  stage_nvfp4_scales<G,S>(scales,shared,m_tile,rmod_base);
  float a[S::kRowsPerWarp][S::kAccumulatorChains]={};
  compute_nvfp4_rows<G,S>(input,codes,scales,shared,inverseD,parent,warp*S::kRowsPerWarp,lane,a);
#pragma unroll
  for(int r=0;r<S::kRowsPerWarp;r++) {
   float total=0;
#pragma unroll
   for(int c=0;c<S::kAccumulatorChains;c++)total+=a[r][c];
   total=warp_reduce_sum(total);
   if(!lane)output[parent[r]]=__float2bfloat16_rn(total);
  }
 } else {
  static_assert(S::kWarpsPerRow==1 && S::kTokenTile==T);
  __shared__ Nvfp4SmallTSharedStorage<G,T,S> shared;
  stage_nvfp4_scales<G,S>(scales,shared.gemv,m_tile,rmod_base);
  float a[S::kRowsPerWarp][T][S::kAccumulatorChains]={};
  compute_nvfp4_small_t_rows<G,T,S>(input,codes,scales,shared,inverseD,parent,warp*S::kRowsPerWarp,0,0,lane,a);
#pragma unroll
  for(int r=0;r<S::kRowsPerWarp;r++) {
#pragma unroll
   for(int t=0;t<T;t++) {
    float total=0;
#pragma unroll
    for(int c=0;c<S::kAccumulatorChains;c++)total+=a[r][t][c];
    total=warp_reduce_sum(total);
    if(!lane)output[(long long)t*N+parent[r]]=__float2bfloat16_rn(total);
   }
  }
 }
}
template<int T> struct SwiConfig {
 static constexpr int Warps=T>=13?16:(T>=5?4:8);
 static constexpr auto Access=T<=4?Nvfp4SmallTActivationAccess::SharedPhase:Nvfp4SmallTActivationAccess::TokenPacked;
 using Schedule=Nvfp4SmallTSchedule<Warps,1,2,16,T,1,Access,Nvfp4ScaleAccess::Direct,Nvfp4CodeCache::Default,1,Nvfp4SmallTBlockOrder::RowsContiguous,1>;
 static constexpr int Threads=Schedule::kThreads,MinBlocks=1;
};
template<> struct SwiConfig<1> {
 using Schedule=Nvfp4GemvSchedule<8,2,16,4,Nvfp4ScaleAccess::Direct,Nvfp4CodeCache::Default,2>;
 static constexpr int Threads=256,MinBlocks=2;
};
template<int T>
__device__ __forceinline__ void swiglu(const uint8_t* codes,const uint8_t* scales,
 const __nv_bfloat16* input,__nv_bfloat16* output,float inverseD) {
 using G=Nvfp4MlpGateUpGeometry;using S=typename SwiConfig<T>::Schedule;
 constexpr int FF=17408,CtasPerTile=128/S::kWarpsPerCta;
 int m=(int)blockIdx.x/CtasPerTile,cta=(int)blockIdx.x-m*CtasPerTile;
 int lane=(int)threadIdx.x&31,warp=(int)threadIdx.x>>5,flat=cta*S::kWarpsPerCta+warp;
 int gate=m*128+(flat>>2)+(flat&3)*32,parent[2]={gate,gate+FF};
 if constexpr(T==1) {
  __shared__ Nvfp4GemvSharedStorage<G,S> shared;
  float a[2][S::kAccumulatorChains]={};
  compute_nvfp4_rows<G,S>(input,codes,scales,shared,inverseD,parent,warp*2,lane,a);
  float g=0,u=0;
#pragma unroll
  for(int c=0;c<S::kAccumulatorChains;c++){g+=a[0][c];u+=a[1][c];}
  g=warp_reduce_sum(g);u=warp_reduce_sum(u);
  if(!lane)output[gate]=__float2bfloat16_rn(silu(g)*u);
 } else {
  __shared__ Nvfp4SmallTSharedStorage<G,T,S> shared;
  float a[2][T][S::kAccumulatorChains]={};
  compute_nvfp4_small_t_rows<G,T,S>(input,codes,scales,shared,inverseD,parent,warp*2,0,0,lane,a);
#pragma unroll
  for(int t=0;t<T;t++) {
   float g=0,u=0;
#pragma unroll
   for(int c=0;c<S::kAccumulatorChains;c++){g+=a[0][t][c];u+=a[1][t][c];}
   g=warp_reduce_sum(g);u=warp_reduce_sum(u);
   if(!lane)output[(long long)t*FF+gate]=__float2bfloat16_rn(silu(g)*u);
  }
 }
}
} // namespace ts_nidf
#define TS_NIDF_LINEAR(N,K,T) \
extern "C" __global__ __launch_bounds__(ts_nidf::LinearConfig<N,K,T>::Threads,ts_nidf::LinearConfig<N,K,T>::MinBlocks) \
void ts_nidf_linear_n##N##_k##K##_t##T(const uint8_t* c,const uint8_t* s,const __nv_bfloat16* x,__nv_bfloat16* y,float inv) { ts_nidf::linear<N,K,T>(c,s,x,y,inv); }
#define TS_NIDF_SWIGLU(T) \
extern "C" __global__ __launch_bounds__(ts_nidf::SwiConfig<T>::Threads,ts_nidf::SwiConfig<T>::MinBlocks) \
void ts_nidf_swiglu_t##T(const uint8_t* c,const uint8_t* s,const __nv_bfloat16* x,__nv_bfloat16* y,float inv) { ts_nidf::swiglu<T>(c,s,x,y,inv); }
TS_NIDF_LINEAR(5120,25600,1)
TS_NIDF_LINEAR(5120,25600,2)
TS_NIDF_LINEAR(5120,25600,3)
TS_NIDF_LINEAR(5120,25600,4)
TS_NIDF_LINEAR(5120,25600,5)
TS_NIDF_LINEAR(5120,25600,6)
TS_NIDF_LINEAR(5120,25600,7)
TS_NIDF_LINEAR(5120,25600,8)
TS_NIDF_LINEAR(5120,25600,9)
TS_NIDF_LINEAR(5120,25600,10)
TS_NIDF_LINEAR(5120,25600,11)
TS_NIDF_LINEAR(5120,25600,12)
TS_NIDF_LINEAR(5120,25600,13)
TS_NIDF_LINEAR(5120,25600,14)
TS_NIDF_LINEAR(5120,25600,15)
TS_NIDF_LINEAR(5120,25600,16)
TS_NIDF_LINEAR(5120,25600,17)
TS_NIDF_LINEAR(5120,25600,18)
TS_NIDF_LINEAR(5120,25600,19)
TS_NIDF_LINEAR(5120,25600,20)
TS_NIDF_LINEAR(5120,25600,21)
TS_NIDF_LINEAR(5120,25600,22)
TS_NIDF_LINEAR(5120,25600,23)
TS_NIDF_LINEAR(5120,25600,24)
TS_NIDF_LINEAR(5120,25600,25)
TS_NIDF_LINEAR(5120,25600,26)
TS_NIDF_LINEAR(5120,25600,27)
TS_NIDF_LINEAR(5120,25600,28)
TS_NIDF_LINEAR(5120,25600,29)
TS_NIDF_LINEAR(5120,25600,30)
TS_NIDF_LINEAR(5120,25600,31)
TS_NIDF_LINEAR(5120,25600,32)
TS_NIDF_LINEAR(6144,5120,1)
TS_NIDF_LINEAR(6144,5120,2)
TS_NIDF_LINEAR(6144,5120,3)
TS_NIDF_LINEAR(6144,5120,4)
TS_NIDF_LINEAR(6144,5120,5)
TS_NIDF_LINEAR(6144,5120,6)
TS_NIDF_LINEAR(6144,5120,7)
TS_NIDF_LINEAR(6144,5120,8)
TS_NIDF_LINEAR(6144,5120,9)
TS_NIDF_LINEAR(6144,5120,10)
TS_NIDF_LINEAR(6144,5120,11)
TS_NIDF_LINEAR(6144,5120,12)
TS_NIDF_LINEAR(6144,5120,13)
TS_NIDF_LINEAR(6144,5120,14)
TS_NIDF_LINEAR(6144,5120,15)
TS_NIDF_LINEAR(6144,5120,16)
TS_NIDF_LINEAR(6144,5120,17)
TS_NIDF_LINEAR(6144,5120,18)
TS_NIDF_LINEAR(6144,5120,19)
TS_NIDF_LINEAR(6144,5120,20)
TS_NIDF_LINEAR(6144,5120,21)
TS_NIDF_LINEAR(6144,5120,22)
TS_NIDF_LINEAR(6144,5120,23)
TS_NIDF_LINEAR(6144,5120,24)
TS_NIDF_LINEAR(6144,5120,25)
TS_NIDF_LINEAR(6144,5120,26)
TS_NIDF_LINEAR(6144,5120,27)
TS_NIDF_LINEAR(6144,5120,28)
TS_NIDF_LINEAR(6144,5120,29)
TS_NIDF_LINEAR(6144,5120,30)
TS_NIDF_LINEAR(6144,5120,31)
TS_NIDF_LINEAR(6144,5120,32)
TS_NIDF_LINEAR(5120,4096,1)
TS_NIDF_LINEAR(5120,4096,2)
TS_NIDF_LINEAR(5120,4096,3)
TS_NIDF_LINEAR(5120,4096,4)
TS_NIDF_LINEAR(5120,4096,5)
TS_NIDF_LINEAR(5120,4096,6)
TS_NIDF_LINEAR(5120,4096,7)
TS_NIDF_LINEAR(5120,4096,8)
TS_NIDF_LINEAR(5120,4096,9)
TS_NIDF_LINEAR(5120,4096,10)
TS_NIDF_LINEAR(5120,4096,11)
TS_NIDF_LINEAR(5120,4096,12)
TS_NIDF_LINEAR(5120,4096,13)
TS_NIDF_LINEAR(5120,4096,14)
TS_NIDF_LINEAR(5120,4096,15)
TS_NIDF_LINEAR(5120,4096,16)
TS_NIDF_LINEAR(5120,4096,17)
TS_NIDF_LINEAR(5120,4096,18)
TS_NIDF_LINEAR(5120,4096,19)
TS_NIDF_LINEAR(5120,4096,20)
TS_NIDF_LINEAR(5120,4096,21)
TS_NIDF_LINEAR(5120,4096,22)
TS_NIDF_LINEAR(5120,4096,23)
TS_NIDF_LINEAR(5120,4096,24)
TS_NIDF_LINEAR(5120,4096,25)
TS_NIDF_LINEAR(5120,4096,26)
TS_NIDF_LINEAR(5120,4096,27)
TS_NIDF_LINEAR(5120,4096,28)
TS_NIDF_LINEAR(5120,4096,29)
TS_NIDF_LINEAR(5120,4096,30)
TS_NIDF_LINEAR(5120,4096,31)
TS_NIDF_LINEAR(5120,4096,32)
TS_NIDF_LINEAR(1280,5120,1)
TS_NIDF_LINEAR(1280,5120,2)
TS_NIDF_LINEAR(1280,5120,3)
TS_NIDF_LINEAR(1280,5120,4)
TS_NIDF_LINEAR(1280,5120,5)
TS_NIDF_LINEAR(1280,5120,6)
TS_NIDF_LINEAR(1280,5120,7)
TS_NIDF_LINEAR(1280,5120,8)
TS_NIDF_LINEAR(1280,5120,9)
TS_NIDF_LINEAR(1280,5120,10)
TS_NIDF_LINEAR(1280,5120,11)
TS_NIDF_LINEAR(1280,5120,12)
TS_NIDF_LINEAR(1280,5120,13)
TS_NIDF_LINEAR(1280,5120,14)
TS_NIDF_LINEAR(1280,5120,15)
TS_NIDF_LINEAR(1280,5120,16)
TS_NIDF_LINEAR(1280,5120,17)
TS_NIDF_LINEAR(1280,5120,18)
TS_NIDF_LINEAR(1280,5120,19)
TS_NIDF_LINEAR(1280,5120,20)
TS_NIDF_LINEAR(1280,5120,21)
TS_NIDF_LINEAR(1280,5120,22)
TS_NIDF_LINEAR(1280,5120,23)
TS_NIDF_LINEAR(1280,5120,24)
TS_NIDF_LINEAR(1280,5120,25)
TS_NIDF_LINEAR(1280,5120,26)
TS_NIDF_LINEAR(1280,5120,27)
TS_NIDF_LINEAR(1280,5120,28)
TS_NIDF_LINEAR(1280,5120,29)
TS_NIDF_LINEAR(1280,5120,30)
TS_NIDF_LINEAR(1280,5120,31)
TS_NIDF_LINEAR(1280,5120,32)
TS_NIDF_LINEAR(256,5120,1)
TS_NIDF_LINEAR(256,5120,2)
TS_NIDF_LINEAR(256,5120,3)
TS_NIDF_LINEAR(256,5120,4)
TS_NIDF_LINEAR(256,5120,5)
TS_NIDF_LINEAR(256,5120,6)
TS_NIDF_LINEAR(256,5120,7)
TS_NIDF_LINEAR(256,5120,8)
TS_NIDF_LINEAR(256,5120,9)
TS_NIDF_LINEAR(256,5120,10)
TS_NIDF_LINEAR(256,5120,11)
TS_NIDF_LINEAR(256,5120,12)
TS_NIDF_LINEAR(256,5120,13)
TS_NIDF_LINEAR(256,5120,14)
TS_NIDF_LINEAR(256,5120,15)
TS_NIDF_LINEAR(256,5120,16)
TS_NIDF_LINEAR(256,5120,17)
TS_NIDF_LINEAR(256,5120,18)
TS_NIDF_LINEAR(256,5120,19)
TS_NIDF_LINEAR(256,5120,20)
TS_NIDF_LINEAR(256,5120,21)
TS_NIDF_LINEAR(256,5120,22)
TS_NIDF_LINEAR(256,5120,23)
TS_NIDF_LINEAR(256,5120,24)
TS_NIDF_LINEAR(256,5120,25)
TS_NIDF_LINEAR(256,5120,26)
TS_NIDF_LINEAR(256,5120,27)
TS_NIDF_LINEAR(256,5120,28)
TS_NIDF_LINEAR(256,5120,29)
TS_NIDF_LINEAR(256,5120,30)
TS_NIDF_LINEAR(256,5120,31)
TS_NIDF_LINEAR(256,5120,32)
TS_NIDF_LINEAR(5120,17408,1)
TS_NIDF_LINEAR(5120,17408,2)
TS_NIDF_LINEAR(5120,17408,3)
TS_NIDF_LINEAR(5120,17408,4)
TS_NIDF_LINEAR(5120,17408,5)
TS_NIDF_LINEAR(5120,17408,6)
TS_NIDF_LINEAR(5120,17408,7)
TS_NIDF_LINEAR(5120,17408,8)
TS_NIDF_LINEAR(5120,17408,9)
TS_NIDF_LINEAR(5120,17408,10)
TS_NIDF_LINEAR(5120,17408,11)
TS_NIDF_LINEAR(5120,17408,12)
TS_NIDF_LINEAR(5120,17408,13)
TS_NIDF_LINEAR(5120,17408,14)
TS_NIDF_LINEAR(5120,17408,15)
TS_NIDF_LINEAR(5120,17408,16)
TS_NIDF_LINEAR(5120,17408,17)
TS_NIDF_LINEAR(5120,17408,18)
TS_NIDF_LINEAR(5120,17408,19)
TS_NIDF_LINEAR(5120,17408,20)
TS_NIDF_LINEAR(5120,17408,21)
TS_NIDF_LINEAR(5120,17408,22)
TS_NIDF_LINEAR(5120,17408,23)
TS_NIDF_LINEAR(5120,17408,24)
TS_NIDF_LINEAR(5120,17408,25)
TS_NIDF_LINEAR(5120,17408,26)
TS_NIDF_LINEAR(5120,17408,27)
TS_NIDF_LINEAR(5120,17408,28)
TS_NIDF_LINEAR(5120,17408,29)
TS_NIDF_LINEAR(5120,17408,30)
TS_NIDF_LINEAR(5120,17408,31)
TS_NIDF_LINEAR(5120,17408,32)
TS_NIDF_SWIGLU(1)
TS_NIDF_SWIGLU(2)
TS_NIDF_SWIGLU(3)
TS_NIDF_SWIGLU(4)
TS_NIDF_SWIGLU(5)
TS_NIDF_SWIGLU(6)
TS_NIDF_SWIGLU(7)
TS_NIDF_SWIGLU(8)
TS_NIDF_SWIGLU(9)
TS_NIDF_SWIGLU(10)
TS_NIDF_SWIGLU(11)
TS_NIDF_SWIGLU(12)
TS_NIDF_SWIGLU(13)
TS_NIDF_SWIGLU(14)
TS_NIDF_SWIGLU(15)
TS_NIDF_SWIGLU(16)

// The following scalar/selection adapters preserve the pinned original operation order.
// Their source functions and adaptations are enumerated in draft/source-provenance.json.
using namespace ninfer::ops;
using namespace ninfer::ops::detail;
constexpr int kDflash2ConvGroupSize=16,kDflash2SelectorThreads=256;
extern "C" __global__ void ts_nidf_dynamic_conv(
    const __nv_bfloat16* __restrict__ hidden, const __nv_bfloat16* __restrict__ dynamic,
    const __nv_bfloat16* __restrict__ base, std::int32_t hidden_size, std::int32_t columns,
    std::int32_t block_size, std::int32_t side, __nv_bfloat16* __restrict__ out) {
    const int groups = hidden_size / kDflash2ConvGroupSize;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < hidden_size * columns;
         idx += blockDim.x * gridDim.x) {
        const int c   = idx % hidden_size;
        const int t   = idx / hidden_size;
        const int p   = t % block_size;
        const int g   = c / kDflash2ConvGroupSize;
        float acc     = 0.0F;
        const int dy  = groups * 2 * side;
        // Fork layout: [hidden, columns] tensors store channel-fastest, so
        // element (c, t) lives at c + H*t; dynamic rows stride the same way.
        const std::int64_t dynamic_rows = static_cast<std::int64_t>(groups) * 4;
        for (int k = 0; k < 2; ++k) {
            const float w =
                __bfloat162float(dynamic[static_cast<std::int64_t>(dy + groups * k + g) +
                                            dynamic_rows * t]) +
                __bfloat162float(base[(static_cast<std::int64_t>(side) * 2 + k) * hidden_size + c]);
            float x = 0.0F;
            if (p - k >= 0) {
                x = __bfloat162float(
                    hidden[static_cast<std::int64_t>(c) + static_cast<std::int64_t>(hidden_size) *
                                                                 (t - k)]);
            }
            acc += w * x;
        }
        out[idx] = __float2bfloat16(acc);
    }
}
extern "C" __global__ void ts_nidf_selector_scores(
    const std::int32_t* __restrict__ candidates, const std::int32_t* __restrict__ pred_ids,
    const float* __restrict__ unary, const float* __restrict__ hidden_proj,
    const std::uint8_t* __restrict__ successor_codes,
    const std::uint8_t* __restrict__ successor_scales, float successor_inverse_divisor,
    const std::uint8_t* __restrict__ predecessor_codes,
    const std::uint8_t* __restrict__ predecessor_scales, float predecessor_inverse_divisor,
    std::int32_t rank, std::int32_t top_k, std::int32_t positions, std::int32_t lanes,
    float* __restrict__ out) {
    // One block per (i, j, s, l), enumerated so the block index is the output element index.
    const int idx = blockIdx.x;
    const int i   = idx % top_k;
    const int j   = (idx / top_k) % top_k;
    const int s   = (idx / (top_k * top_k)) % positions;
    const int l   = idx / (top_k * top_k * positions);

    // [K, P, L] is contiguous with the draft position ahead of the lane, so the per-position
    // vector (x, s, l) sits at x + K * (s + P * l). Putting the lane first here is the whole
    // batch>1 defect: it is the identity at L = 1 and a transpose at every wider batch.
    const std::int64_t slot = static_cast<std::int64_t>(l) * positions + s;
    const std::int32_t succ_id = candidates[static_cast<std::int64_t>(i) + top_k * slot];
    const std::int32_t pred_id = pred_ids[static_cast<std::int64_t>(j) + top_k * slot];
    const std::int64_t succ_row  = static_cast<std::int64_t>(succ_id) * rank;
    const std::int64_t pred_row  = static_cast<std::int64_t>(pred_id) * rank;
    const float* h               = hidden_proj + slot * rank;

    // The codebooks are NVFP4 weights in QuantLayout::BlockScaleK16M128x4. Only the
    // E4M3FN scale plane is swizzled by the artifact writer (swizzle_nvfp4_scales in
    // tools/artifact/layouts.py); the E2M1 code plane stays row-major, so the code
    // gathers below keep their plain `row / 2` offsets. Scales must go through the
    // shared nvfp4_scale_offset (ops/linear/nvfp4/nvfp4_codec.cuh) - the same
    // definition nvfp4_gemv.cuh's nvfp4_scale_offset<Geometry> wraps - addressed by
    // the token id as the parent row, not by the flattened `id * rank` row base.
    // This kernel is not templated on a Geometry, so kScaleTilesPerRow = K / 64 is
    // supplied from the runtime rank.
    const int scale_tiles_per_row = rank / 64;

    float local = 0.0F;
    for (int r = threadIdx.x; r < rank; r += blockDim.x) {
        const float2 succ_pair = decode_nvfp4_e2m1x2(
            successor_codes[succ_row / 2 + r / 2]);
        const float2 pred_pair = decode_nvfp4_e2m1x2(
            predecessor_codes[pred_row / 2 + r / 2]);
        const float succ = (r & 1) == 0 ? succ_pair.x : succ_pair.y;
        const float pred = (r & 1) == 0 ? pred_pair.x : pred_pair.y;
        const float succ_scale = decode_nvfp4_e4m3(
            successor_scales[nvfp4_scale_offset(succ_id, r / 16, scale_tiles_per_row)]);
        const float pred_scale = decode_nvfp4_e4m3(
            predecessor_scales[nvfp4_scale_offset(pred_id, r / 16, scale_tiles_per_row)]);
        local += succ * succ_scale * successor_inverse_divisor * pred * pred_scale *
                 predecessor_inverse_divisor * h[r];
    }
    __shared__ float warp_sums[kDflash2SelectorThreads / 32];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset /= 2) {
        local += __shfl_down_sync(0xffffffffu, local, offset);
    }
    if (lane == 0) { warp_sums[warp] = local; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0F;
        for (int w = 0; w < blockDim.x / 32; ++w) {
            total += warp_sums[w];
        }
        out[static_cast<std::int64_t>(i) +
            top_k * (static_cast<std::int64_t>(j) + top_k * slot)] =
            total + unary[static_cast<std::int64_t>(i) + top_k * slot];
    }
}
extern "C" __global__ void ts_nidf_selector_walk(const float* __restrict__ scores,
                                             const std::int32_t* __restrict__ candidates,
                                             std::int32_t top_k, std::int32_t positions,
                                             std::int32_t lanes, std::int32_t* __restrict__ out) {
    const int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l >= lanes) { return; }
    int pred = 0;
    for (int s = 0; s < positions; ++s) {
        const std::int64_t slot = static_cast<std::int64_t>(l) * positions + s;
        int best_i   = 0;
        float best_v = -INFINITY;
        for (int i = 0; i < top_k; ++i) {
            const float v = scores[static_cast<std::int64_t>(i) +
                                   top_k * (static_cast<std::int64_t>(pred) + top_k * slot)];
            if (v > best_v) {
                best_v = v;
                best_i = i;
            }
        }
        out[slot] = candidates[static_cast<std::int64_t>(best_i) + top_k * slot];
        pred = best_i;
    }
}
extern "C" __global__ void ts_nidf_selector_predecessors(const std::int32_t* __restrict__ candidates,
                                                     const std::int32_t* __restrict__ anchors,
                                                     std::int32_t top_k, std::int32_t positions,
                                                     std::int32_t lanes,
                                                     std::int32_t* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= top_k * positions * lanes) { return; }
    const int j = idx % top_k;
    const int s = (idx / top_k) % positions;
    const int l = idx / (top_k * positions);
    const std::int64_t slot = static_cast<std::int64_t>(l) * positions + s;
    const std::int32_t value =
        s == 0 ? anchors[l]
               : candidates[static_cast<std::int64_t>(j) + top_k * (slot - 1)];
    out[static_cast<std::int64_t>(j) + top_k * slot] = value;
}

// Original plain RMSNorm reductions: D128 warp BF16x2; D5120 512-thread CTA BF16x2.
extern "C" __global__ void ts_nidf_rms128(const __nv_bfloat16* x,const __nv_bfloat16* weight,
 __nv_bfloat16* out,int rows,float eps) {
 int lane=(int)threadIdx.x&31,warp=(int)threadIdx.x>>5,row=(int)blockIdx.x*4+warp;
 if(row>=rows)return;
 int p=lane*2;long long b=(long long)row*128;
 float a=__bfloat162float(x[b+p]),c=__bfloat162float(x[b+p+1]);
 float d=__bfloat162float(x[b+p+64]),e=__bfloat162float(x[b+p+65]);
 float sum=a*a+c*c+d*d+e*e;sum=warp_reduce_sum(sum);
 float inv=lane==0?rsqrtf(sum*(1.f/128.f)+eps):0;inv=__shfl_sync(0xffffffffu,inv,0);
 out[b+p]=__float2bfloat16_rn(a*inv*__bfloat162float(weight[p]));
 out[b+p+1]=__float2bfloat16_rn(c*inv*__bfloat162float(weight[p+1]));
 out[b+p+64]=__float2bfloat16_rn(d*inv*__bfloat162float(weight[p+64]));
 out[b+p+65]=__float2bfloat16_rn(e*inv*__bfloat162float(weight[p+65]));
}
extern "C" __global__ void ts_nidf_rms5120(const __nv_bfloat16* x,const __nv_bfloat16* weight,
 __nv_bfloat16* out,int rows,float eps) {
 int row=(int)blockIdx.x;if(row>=rows)return;
 float values[10],sum=0;long long b=(long long)row*5120;
#pragma unroll
 for(int i=0;i<5;i++){int p=((int)threadIdx.x+i*512)*2;
  float a=__bfloat162float(x[b+p]),c=__bfloat162float(x[b+p+1]);
  values[2*i]=a;values[2*i+1]=c;sum+=a*a+c*c;}
 __shared__ float sums[16],inv;
 float total=block_reduce_sum<512>(sum,sums);
 if(!threadIdx.x)inv=rsqrtf(total/5120.f+eps);__syncthreads();
#pragma unroll
 for(int i=0;i<5;i++){int p=((int)threadIdx.x+i*512)*2;
  out[b+p]=__float2bfloat16_rn(values[2*i]*inv*__bfloat162float(weight[p]));
  out[b+p+1]=__float2bfloat16_rn(values[2*i+1]*inv*__bfloat162float(weight[p+1]));}
}
extern "C" __global__ void ts_nidf_residual_add(const __nv_bfloat16* add,__nv_bfloat16* residual,long long count) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 if(i<count)residual[i]=__float2bfloat16_rn(__bfloat162float(add[i])+__bfloat162float(residual[i]));
}
extern "C" __global__ void ts_nidf_copy_target_tap(const float* x,__nv_bfloat16* concatenated,int rows,int tap,int inputStride,int outputStride) {
 int c=(int)blockIdx.x*blockDim.x+threadIdx.x,t=(int)blockIdx.y;
 if(c<5120&&t<rows)concatenated[(long long)t*outputStride+tap*5120+c]=__float2bfloat16_rn(x[(long long)t*inputStride+c]);
}
extern "C" __global__ void ts_nidf_pack_hidden(const __nv_bfloat16* x,__nv_bfloat16* packed,int block,int lanes) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,count=(long long)lanes*(block-1)*5120;
 if(i>=count)return;long long row=i/5120,c=i%5120,lane=row/(block-1),t=row%(block-1);
 packed[i]=x[((lane*block+t+1)*5120)+c];
}
extern "C" __global__ void ts_nidf_cast_bf16_f32(const __nv_bfloat16* x,float* y,long long count) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[i]=__bfloat162float(x[i]);
}
extern "C" __global__ void ts_nidf_prepare_block(const int* anchors,const int* lengths,const int* valid,int* ids,int* positions,int block,int vocab) {
 int i=(int)threadIdx.x,b=(int)blockIdx.x;if(i>=block)return;
 int count=valid[b],base=lengths[b],anchor=anchors[b];if(count<1||count>block||base<0||base>0x7fffffff-block||anchor<0||anchor>=vocab){asm("trap;");return;}
 int at=i+block*b;ids[at]=i==0?anchor:248070;positions[at]=base+min(i,count-1);
}
// Original DflashText1D: double frequency and range reduction; full128 split-half NeoX.
// q may be null for context append. Every norm is a separate BF16 boundary before this op.
extern "C" __global__ void ts_nidf_rope(__nv_bfloat16* q,__nv_bfloat16* k,const int* positions,const double* frequencies,int tokens) {
 int token=(int)blockIdx.x,combined=(int)blockIdx.y,pair=(int)threadIdx.x;
 if(token>=tokens||pair>=64)return;
 __nv_bfloat16* data=combined<32?q:k;int heads=combined<32?32:8,head=combined<32?combined:combined-32;
 if(!data)return;
 constexpr double InvTwoPi=1.59154943091895336e-01,TwoPi=6.28318530717958648e+00;
 double angle=(double)positions[token]*frequencies[pair],turns=angle*InvTwoPi;
 float reduced=(float)(angle-nearbyint(turns)*TwoPi),s,c;sincosf(reduced,&s,&c);
 long long at=((long long)token*heads+head)*128+pair;
 float a=__bfloat162float(data[at]),b=__bfloat162float(data[at+64]);
 data[at]=__float2bfloat16_rn(a*c-b*s);data[at+64]=__float2bfloat16_rn(b*c+a*s);
}
// Owned ring copy. Width<=2048 plus contiguous valid positions ensures unique destinations.
extern "C" __global__ void ts_nidf_ring_append(const __nv_bfloat16* k,const __nv_bfloat16* v,const int* positions,const int* counts,const int* lanes,
 __nv_bfloat16* ringK,__nv_bfloat16* ringV,int width,int batches,int slots) {
 long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x,total=(long long)batches*width*1024;if(i>=total)return;
 int c=i%128,h=(i/128)%8,t=(i/1024)%width,b=i/((long long)width*1024),count=counts[b],slot=lanes[b];
 if(count<0||count>width||slot<0||slot>=slots){asm("trap;");return;}if(t>=count)return;
 int position=positions[b*width+t];if(position<0){asm("trap;");return;}
 long long at=(((long long)slot*8+h)*2048+(position&2047))*128+c;ringK[at]=k[i];ringV[at]=v[i];
}

namespace ts_nidf_topk {


// The one shape this engine calls with (kDflash2SelectorTopK,
// kernel/src/model_internal.h). Any other k is the vendored op's.
constexpr std::int32_t kSpecializedK = 16;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
// Rows per block, picked so a 248k-row column spreads over ~120 blocks and
// the grid fills a 170-SM card even at one column. Smaller slices would add
// merge work for no more parallelism.
constexpr std::int64_t kRowsPerSplit = 2048;
constexpr std::int32_t kMaxSplits = 1024;

struct Entry {
  float value;
  int row;
};

// The vendored op's order, verbatim in intent: better means larger value, and
// on a tie the smaller row id. `-0.0f == 0.0f` in IEEE compare, so the two
// zeroes tie and the row decides, which is what the contract says.
__device__ __forceinline__ bool better(const Entry &a, const Entry &b) {
  return a.value > b.value || (a.value == b.value && a.row < b.row);
}

__device__ __forceinline__ Entry sentinel() {
  return Entry{-INFINITY, 0x7fffffff};
}

// Insert `e` into the sorted-best-first list, dropping the worst entry. Fully
// unrolled with static indices so `list` stays in registers: the vendored
// op's data-dependent `while` loop is what forces its list to local memory.
template <int K>
__device__ __forceinline__ void insert(Entry (&list)[K], Entry e) {
#pragma unroll
  for (int i = 0; i < K; ++i) {
    const Entry held = list[i];
    const bool take = better(e, held);
    list[i] = take ? e : held;
    e = take ? held : e;
  }
}

// K rounds of warp argmax over the lanes' list heads. Lane 0 writes the
// column's k winners, best first, to `out` (stride 1). Every lane must call
// this: it is warp-collective, and it consumes `list`.
//
// The head is always `list[0]` and the winning lane shifts its list down by
// one, rather than the vendored op's cursor into the list. That is the whole
// reason this implementation is fast: `list[cursor]` is a dynamic index, and
// a register array indexed dynamically is not a register array -- it is local
// memory, which is what makes the vendored kernel's 20-register frame spill.
// Every index here is a constant under `#pragma unroll`.
template <int K>
__device__ __forceinline__ void warp_drain_topk(Entry (&list)[K], int lane, Entry *out) {
#pragma unroll 1
  for (int slot = 0; slot < K; ++slot) {
    Entry head = list[0];
    for (int offset = 16; offset > 0; offset /= 2) {
      Entry other;
      other.value = __shfl_down_sync(0xffffffffu, head.value, offset);
      other.row = __shfl_down_sync(0xffffffffu, head.row, offset);
      if (better(other, head)) { head = other; }
    }
    head.value = __shfl_sync(0xffffffffu, head.value, 0);
    head.row = __shfl_sync(0xffffffffu, head.row, 0);
    if (lane == 0) { out[slot] = head; }
    // Rows are unique within a column, so exactly one lane holds the winner
    // and exactly one list advances. The sentinel guard matters only for a
    // column with fewer than K rows, where every remaining slot is a
    // sentinel whichever lane yields it.
    if (head.row != 0x7fffffff && list[0].row == head.row) {
#pragma unroll
      for (int i = 0; i < K - 1; ++i) { list[i] = list[i + 1]; }
      list[K - 1] = sentinel();
    }
  }
}

// The block's own k, from the lanes' private lists: each warp to k in shared
// memory, then warp 0 over the 4 * k survivors.
template <int K>
__device__ __forceinline__ void block_drain_topk(Entry (&list)[K], Entry *shared, Entry *out) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  warp_drain_topk<K>(list, lane, shared + warp * K);
  __syncthreads();
  if (warp == 0) {
    Entry merged[K];
#pragma unroll
    for (int i = 0; i < K; ++i) { merged[i] = sentinel(); }
    for (int i = lane; i < kWarps * K; i += 32) { insert<K>(merged, shared[i]); }
    warp_drain_topk<K>(merged, lane, out);
  }
}

extern "C" __global__ void __launch_bounds__(kThreads) ts_nidf_top16_partial(
    const __nv_bfloat16 *__restrict__ logits, std::int32_t rows, std::int32_t splits,
    Entry *__restrict__ partial) {
  constexpr int K=16;
  const int split = static_cast<int>(blockIdx.x);
  const int column = static_cast<int>(blockIdx.y);
  // Contiguous slices, balanced to within one row, so the read is coalesced
  // and every row belongs to exactly one block.
  const std::int64_t begin = static_cast<std::int64_t>(rows) * split / splits;
  const std::int64_t end = static_cast<std::int64_t>(rows) * (split + 1) / splits;
  const std::int64_t base = static_cast<std::int64_t>(rows) * column;

  Entry list[K];
#pragma unroll
  for (int i = 0; i < K; ++i) { list[i] = sentinel(); }
  for (std::int64_t row = begin + threadIdx.x; row < end; row += kThreads) {
    insert<K>(list, Entry{__bfloat162float(logits[base + row]), static_cast<int>(row)});
  }

  __shared__ Entry shared[kWarps * K];
  block_drain_topk<K>(list, shared, partial + (static_cast<std::int64_t>(column) * splits + split) * K);
}

extern "C" __global__ void __launch_bounds__(kThreads) ts_nidf_top16_merge(
    const Entry *__restrict__ partial, const __nv_bfloat16 *__restrict__ logits, std::int32_t rows,
    std::int32_t splits, std::int32_t *__restrict__ ids, float *__restrict__ values) {
  constexpr int K=16;
  const int column = static_cast<int>(blockIdx.x);
  const Entry *column_partial = partial + static_cast<std::int64_t>(column) * splits * K;
  const int count = splits * K;

  Entry list[K];
#pragma unroll
  for (int i = 0; i < K; ++i) { list[i] = sentinel(); }
  for (int i = threadIdx.x; i < count; i += kThreads) { insert<K>(list, column_partial[i]); }

  __shared__ Entry shared[kWarps * K];
  __shared__ Entry winners[K];
  block_drain_topk<K>(list, shared, winners);
  __syncthreads();

  // `values` carries the logits entry bit-exactly. Re-reading the winning row
  // is what makes that true by construction rather than by trusting a
  // float round trip -- the BF16 the column holds is copied, never rebuilt.
  if (threadIdx.x < K) {
    const Entry winner = winners[threadIdx.x];
    const std::int64_t slot = static_cast<std::int64_t>(threadIdx.x) + static_cast<std::int64_t>(K) * column;
    ids[slot] = winner.row;
    values[slot] = winner.row == 0x7fffffff
        ? winner.value
        : __bfloat162float(logits[static_cast<std::int64_t>(rows) * column + winner.row]);
  }
}


} // namespace ts_nidf_topk

namespace ts_nidf {
template<int Tokens,bool Direct>
__device__ __forceinline__ void swa_partial(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v,
 const int* positions,const int* valid,const int* lanes,const __nv_bfloat16* ringK,const __nv_bfloat16* ringV,
 int maxContext,int splits,__nv_bfloat16* accum,float* m,float* l,__nv_bfloat16* output) {
 noncausal_gqa_split_partial_body<true,Tokens,(Tokens+3)/4,32,Direct>(q,k,v,positions,valid,lanes,ringK,ringV,nullptr,
 2048,2048,0,maxContext,splits,0.08838834613561630f,accum,m,l,output);
}
// Copied operation order from pinned swa_reduce_kernel<Tokens,32,1>, with only exported entry adaptation.
template<int Tokens>
__device__ __forceinline__ void swa_reduce(const __nv_bfloat16* partial_acc,const float* partial_m,const float* partial_l,
 const int* positions,const int* valid_columns,int window,int max_context,int split_capacity,__nv_bfloat16* out)
{
    constexpr int WarpsPerBlock=1,KeyBlock=32;
    constexpr int MaxSplits = 128;
    constexpr unsigned Mask = 0xffffffffu;
    __shared__ float weights[WarpsPerBlock][MaxSplits];

    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int batch      = static_cast<int>(blockIdx.z);
    const int output_row = static_cast<int>(blockIdx.x) * WarpsPerBlock + warp;
    const int token      = output_row / kBidirectionalGqaQHeads;
    const int q_head     = output_row - token * kBidirectionalGqaQHeads;
    if (warp >= WarpsPerBlock || token >= Tokens) return;

    constexpr std::int64_t QueryElements =
        static_cast<std::int64_t>(kBidirectionalGqaHeadDim) * kBidirectionalGqaQHeads * Tokens;
    constexpr std::int64_t StatElements =
        static_cast<std::int64_t>(kBidirectionalGqaQHeads) * Tokens;
    partial_acc += QueryElements * split_capacity * batch;
    partial_m += StatElements * split_capacity * batch;
    partial_l += StatElements * split_capacity * batch;
    positions += static_cast<std::int64_t>(Tokens) * batch;
    out += QueryElements * batch;

    const int length = positions[0];
    if (length < 0 || length > max_context || token >= valid_columns[batch]) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d                                      = lane + item * 32;
            out[bidirectional_gqa_q_index(q_head, d, token)] = __float2bfloat16(0.0f);
        }
        return;
    }
    const int context_count = min(length, window - 1);
    const int context_tiles = (context_count + KeyBlock - 1) / KeyBlock;
    const int active_splits = context_tiles > 0 ? min(context_tiles, split_capacity) : 1;

    float local_m = -CUDART_INF_F;
    for (int split = lane; split < active_splits; split += 32) {
        local_m =
            fmaxf(local_m, partial_m[bidirectional_gqa_stat_index<Tokens>(q_head, token, split)]);
    }
    const float global_m = warp_max<32>(local_m, Mask);

    float local_l = 0.0f;
    for (int split = lane; split < active_splits; split += 32) {
        const auto stat      = bidirectional_gqa_stat_index<Tokens>(q_head, token, split);
        const float weight   = expf(partial_m[stat] - global_m);
        weights[warp][split] = weight;
        local_l += partial_l[stat] * weight;
    }
    const float global_l = warp_sum<32>(local_l, Mask);
    __syncwarp(Mask);

#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d     = lane + item * 32;
        float numerator = 0.0f;
        for (int split = 0; split < active_splits; ++split) {
            numerator +=
                __bfloat162float(
                    partial_acc[bidirectional_gqa_partial_index<Tokens>(q_head, d, token, split)]) *
                weights[warp][split];
        }
        const float value = global_l > 0.0f ? numerator / global_l : 0.0f;
        out[bidirectional_gqa_q_index(q_head, d, token)] = __float2bfloat16(value);
    }
}
}
#define TS_NIDF_SWA(T) \
extern "C" __global__ __launch_bounds__(((T+3)/4)*32,2) void ts_nidf_swa_partial_t##T(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v, const int* p,const int* c,const int* lanes,const __nv_bfloat16* rk,const __nv_bfloat16* rv, int maxContext,int splits,__nv_bfloat16* a,float* m,float* l,__nv_bfloat16* out) { ts_nidf::swa_partial<T,false>(q,k,v,p,c,lanes,rk,rv,maxContext,splits,a,m,l,out); } \
extern "C" __global__ __launch_bounds__(((T+3)/4)*32,2) void ts_nidf_swa_direct_t##T(const __nv_bfloat16* q,const __nv_bfloat16* k,const __nv_bfloat16* v, const int* p,const int* c,const int* lanes,const __nv_bfloat16* rk,const __nv_bfloat16* rv, int maxContext,int splits,__nv_bfloat16* a,float* m,float* l,__nv_bfloat16* out) { ts_nidf::swa_partial<T,true>(q,k,v,p,c,lanes,rk,rv,maxContext,splits,a,m,l,out); } \
extern "C" __global__ __launch_bounds__(32,2) void ts_nidf_swa_reduce_t##T(const __nv_bfloat16* a,const float* m,const float* l, const int* p,const int* c,int maxContext,int splits,__nv_bfloat16* out) { ts_nidf::swa_reduce<T>(a,m,l,p,c,2048,maxContext,splits,out); }
TS_NIDF_SWA(1)
TS_NIDF_SWA(2)
TS_NIDF_SWA(3)
TS_NIDF_SWA(4)
TS_NIDF_SWA(5)
TS_NIDF_SWA(6)
TS_NIDF_SWA(7)
TS_NIDF_SWA(8)



