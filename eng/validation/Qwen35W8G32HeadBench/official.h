// Copyright (c) Zhongkai Fu. All rights reserved.
// TensorSharp-owned host adapter. The upstream/ subtree is copied byte-for-byte
// from Ignis e600a7d9e0d5921d767a6cb464373583b3cdcaf4; see its LICENSE/NOTICE.
// Build with -I<tool>/upstream/kernel/vendor/src. This is a freshly compiled
// official-source baseline, not the historical Ignis executable or TS Q6_K.
#pragma once

#include "upstream/kernel/vendor/src/ops/linear/w8/w8_small_t_mma.cuh"

namespace ts_w8_official {
using Geometry = ninfer::ops::detail::W8VocabularyProjectionGeometry;
static_assert(Geometry::kOutputRows == 248320 && Geometry::kInputRows == 5120);

template<int T>
inline auto Kernel() {
    using namespace ninfer::ops::detail;
    using Schedule = typename W8LinearSmallTProductionSchedule<Geometry,T>::Type;
    return &w8_small_t_mma_kernel<Geometry,T,Schedule,W8ContiguousOutput,
        W8SmallTMmaStoreEpilogue,W8SmallTMmaIdentityRows>;
}

template<int T>
inline cudaError_t Launch(const __nv_bfloat16* x,const int8_t* codes,
    const __half* scales,__nv_bfloat16* out,cudaStream_t stream) {
    using namespace ninfer::ops::detail;
    using Schedule = typename W8LinearSmallTProductionSchedule<Geometry,T>::Type;
    W8ContiguousOutput output{out,Geometry::kOutputRows};
    // Direct original __global__ template launch; no device wrapper, math,
    // bounds, output epilogue, launch bounds or synchronization is substituted.
    return ninfer::pdl::launch_dependent(
        {dim3(Geometry::kOutputRows/Schedule::kRowsPerCta),dim3(Schedule::kThreads),0,stream},
        Kernel<T>(),x,reinterpret_cast<const std::uint8_t*>(codes),
        reinterpret_cast<const std::uint8_t*>(scales),output,
        W8SmallTMmaStoreEpilogue{},W8SmallTMmaIdentityRows{});
}

template<int T>
inline cudaError_t Attributes(cudaFuncAttributes* attr,int* active,int* threads) {
    using Schedule = typename ninfer::ops::detail::W8LinearSmallTProductionSchedule<Geometry,T>::Type;
    const auto kernel = Kernel<T>();
    cudaError_t status = cudaFuncGetAttributes(attr,kernel);
    if(status != cudaSuccess) return status;
    *threads = Schedule::kThreads;
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(active,kernel,Schedule::kThreads,0);
}
} // namespace ts_w8_official

// Supported only for real [248320,5120] vocabulary geometry. Small/tail fixtures
// use the independent owned/SIMT paths; they are not official-source coverage.
inline cudaError_t LaunchOfficial(int tokens,const __nv_bfloat16* x,
    const int8_t* codes,const __half* scales,__nv_bfloat16* out,cudaStream_t stream=nullptr) {
    if(!x || !codes || !scales || !out) return cudaErrorInvalidValue;
    if((reinterpret_cast<std::uintptr_t>(x) & 15) ||
       (reinterpret_cast<std::uintptr_t>(codes) & 15) ||
       (reinterpret_cast<std::uintptr_t>(scales) & 15) ||
       (reinterpret_cast<std::uintptr_t>(out) & 1)) return cudaErrorInvalidValue;
#define TS_OFFICIAL_LAUNCH(T) case T: return ts_w8_official::Launch<T>(x,codes,scales,out,stream)
    switch(tokens) {
        TS_OFFICIAL_LAUNCH(1); TS_OFFICIAL_LAUNCH(7); TS_OFFICIAL_LAUNCH(8); TS_OFFICIAL_LAUNCH(14);
        TS_OFFICIAL_LAUNCH(16); TS_OFFICIAL_LAUNCH(28); TS_OFFICIAL_LAUNCH(32); TS_OFFICIAL_LAUNCH(33);
        default: return cudaErrorInvalidValue;
    }
#undef TS_OFFICIAL_LAUNCH
}

inline cudaError_t OfficialAttributes(int tokens,cudaFuncAttributes* attr,int* active,int* block_threads) {
    if(!attr || !active || !block_threads) return cudaErrorInvalidValue;
    *attr = {}; *active = 0; *block_threads = 0;
#define TS_OFFICIAL_ATTRIBUTES(T) case T: return ts_w8_official::Attributes<T>(attr,active,block_threads)
    switch(tokens) {
        TS_OFFICIAL_ATTRIBUTES(1); TS_OFFICIAL_ATTRIBUTES(7); TS_OFFICIAL_ATTRIBUTES(8); TS_OFFICIAL_ATTRIBUTES(14);
        TS_OFFICIAL_ATTRIBUTES(16); TS_OFFICIAL_ATTRIBUTES(28); TS_OFFICIAL_ATTRIBUTES(32); TS_OFFICIAL_ATTRIBUTES(33);
        default: return cudaErrorInvalidValue;
    }
#undef TS_OFFICIAL_ATTRIBUTES
}
