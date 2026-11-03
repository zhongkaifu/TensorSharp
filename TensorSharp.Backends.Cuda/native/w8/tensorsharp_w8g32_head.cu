// Copyright (c) Zhongkai Fu. All rights reserved.
// TensorSharp-owned instantiation/cast adapter. The unchanged upstream/ subtree
// is Apache-2.0; retain all its LICENSE/NOTICE/provenance files when distributing.
// No Q35 engine or model default selects this standalone module.
#include "upstream/kernel/vendor/src/ops/linear/w8/w8_small_t_mma.cuh"
#include <cuda.h>
#include <cstddef>
#include <type_traits>

using namespace ninfer::ops::detail;
static_assert(sizeof(void*) == 8, "W8 driver adapter requires a 64-bit ABI");
static_assert(sizeof(W8ContiguousOutput) == 16 && alignof(W8ContiguousOutput) == 8);
static_assert(offsetof(W8ContiguousOutput, data) == 0 && offsetof(W8ContiguousOutput, leading_dim) == 8);
static_assert(sizeof(W8SmallTMmaStoreEpilogue) == 1 && sizeof(W8SmallTMmaIdentityRows) == 1);
static_assert(std::is_trivially_copyable_v<W8ContiguousOutput>);
// These assertions are compiled by the actual installed SDK, independently of
// the managed Marshal-size tests. Driver parameter offsets are checked at load.
static_assert(sizeof(CUlaunchAttributeValue) == 64 && sizeof(CUlaunchAttribute) == 72);
static_assert(offsetof(CUlaunchAttribute, value) == 8);
static_assert(sizeof(CUlaunchConfig) == 56 && offsetof(CUlaunchConfig, hStream) == 32);
static_assert(offsetof(CUlaunchConfig, attrs) == 40 && offsetof(CUlaunchConfig, numAttrs) == 48);
static_assert(CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION == 6);

// Explicitly instantiate the original global template, not a device wrapper or
// private kernel clone. Original launch bounds, schedule, math and PDL remain.
namespace ninfer::ops::detail {
#define TS_W8_INSTANTIATE(T) \
 template __global__ void w8_small_t_mma_kernel<W8VocabularyProjectionGeometry,T, \
   typename W8LinearSmallTProductionSchedule<W8VocabularyProjectionGeometry,T>::Type, \
   W8ContiguousOutput,W8SmallTMmaStoreEpilogue,W8SmallTMmaIdentityRows,false>( \
   const __nv_bfloat16*,const std::uint8_t*,const std::uint8_t*, \
   W8ContiguousOutput,W8SmallTMmaStoreEpilogue,W8SmallTMmaIdentityRows);
TS_W8_INSTANTIATE(1) TS_W8_INSTANTIATE(2) TS_W8_INSTANTIATE(3)
TS_W8_INSTANTIATE(4) TS_W8_INSTANTIATE(5) TS_W8_INSTANTIATE(6)
TS_W8_INSTANTIATE(7) TS_W8_INSTANTIATE(8) TS_W8_INSTANTIATE(9)
TS_W8_INSTANTIATE(10) TS_W8_INSTANTIATE(11) TS_W8_INSTANTIATE(12)
TS_W8_INSTANTIATE(13) TS_W8_INSTANTIATE(14) TS_W8_INSTANTIATE(15)
TS_W8_INSTANTIATE(16) TS_W8_INSTANTIATE(17) TS_W8_INSTANTIATE(18)
TS_W8_INSTANTIATE(19) TS_W8_INSTANTIATE(20) TS_W8_INSTANTIATE(21)
TS_W8_INSTANTIATE(22) TS_W8_INSTANTIATE(23) TS_W8_INSTANTIATE(24)
TS_W8_INSTANTIATE(25) TS_W8_INSTANTIATE(26) TS_W8_INSTANTIATE(27)
TS_W8_INSTANTIATE(28) TS_W8_INSTANTIATE(29) TS_W8_INSTANTIATE(30)
TS_W8_INSTANTIATE(31) TS_W8_INSTANTIATE(32) TS_W8_INSTANTIATE(33)
#undef TS_W8_INSTANTIATE
}

// The exact RNE cast expression used by the successful ELF comparison. The
// caller owns its BF16 workspace and validates the numeric input domain.
extern "C" __global__ void ts_w8g32_cast_f32_bf16(
    const float* input,__nv_bfloat16* output,unsigned long long count) {
    unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)output[i]=__float2bfloat16_rn(input[i]);
}
