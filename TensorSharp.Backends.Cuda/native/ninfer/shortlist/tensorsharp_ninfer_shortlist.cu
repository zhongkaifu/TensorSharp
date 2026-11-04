// Copyright (c) Zhongkai Fu. All rights reserved.
// TensorSharp-owned template instantiation/remap bridge. Derived portions are
// Apache-2.0; retain upstream/{LICENSE,NOTICE,manifest.json} and vendor notices.
// Original Ignis e600a7d9e0d5921d767a6cb464373583b3cdcaf4 Q4G64_F16S
// shortlist:131072x5120, actual BF16 activations/output, raw original planes.
// No engine hook/default, weight conversion, activation quantization or GGML patch.
#include "upstream/kernel/vendor/src/ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "upstream/kernel/vendor/src/ops/linear/q4/q4_small_t_mma.cuh"
#include "upstream/kernel/vendor/src/ops/linear/q4/q4_rowsplit_gemm_mma.cuh"
#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {
static_assert(sizeof(void*) == 8);
static_assert(sizeof(Q4GemvStoreEpilogue) == 1);
static_assert(sizeof(Q4SmallTMmaStoreEpilogue) == 1 && sizeof(Q4SmallTMmaIdentityRows) == 1);
static_assert(Q4DraftHeadGeometry<5120>::kOutputRows == 131072);
static_assert(Q4GemvR4W1DirectSchedule::kThreads == 128 && Q4GemvR4W1DirectSchedule::kRowsPerCta == 4);
// The ORIGINAL global templates are instantiated unchanged, not cloned into
// a private numerical kernel. Schedule selections are q4_dispatch.cpp's K5120
// shortlist routes: T1 GEMV, T2..8 exact small-T, T>=9 r64_c128 MMA.
template __global__ void q4_rowsplit_gemv_kernel<Q4GemvR4W1DirectSchedule>(
    const __nv_bfloat16*, const std::uint8_t*, const std::uint8_t*,
    __nv_bfloat16*, __nv_bfloat16*, std::int32_t, std::int32_t, Q4GemvStoreEpilogue);

#define TS_NINFER_SHORTLIST_SMALL(T) \
template __global__ void q4_small_t_mma_kernel<Q4DraftHeadGeometry<5120>,8,T>( \
    const __nv_bfloat16*, const std::uint8_t*, const std::uint8_t*, \
    __nv_bfloat16*, Q4SmallTMmaStoreEpilogue, Q4SmallTMmaIdentityRows);
TS_NINFER_SHORTLIST_SMALL(2)
TS_NINFER_SHORTLIST_SMALL(3)
TS_NINFER_SHORTLIST_SMALL(4)
TS_NINFER_SHORTLIST_SMALL(5)
TS_NINFER_SHORTLIST_SMALL(6)
TS_NINFER_SHORTLIST_SMALL(7)
TS_NINFER_SHORTLIST_SMALL(8)
#undef TS_NINFER_SHORTLIST_SMALL

using TsNinferShortlistMmaSchedule = Q4RowSplitMmaGemmSchedule<64,128,64,64,32,2,1,
    Q4FragmentPipeline::Serial, Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
static_assert(TsNinferShortlistMmaSchedule::kThreads == 128);
static_assert(TsNinferShortlistMmaSchedule::kSharedBytes == 45568);
// T9..33 always uses the original boundary variant (Full=false): no token
// count in this module is divisible by the original128-column physical tile.
template __global__ void q4_rowsplit_gemm_mma_kernel<TsNinferShortlistMmaSchedule,false>(
    const __nv_bfloat16*, const std::uint8_t*, const std::uint8_t*,
    __nv_bfloat16*, std::int32_t, std::int32_t, std::int32_t, std::int32_t);
}

// Original speculative_round.cuh442..448 range behavior, exported separately.
// Invalid shortlist row ids remain unchanged; no clamp, dedup or unary rewrite.
extern "C" __global__ void ts_ninfer_shortlist_remap(std::int32_t* tokens,
    std::int32_t count, const std::int32_t* id_map, std::int32_t rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int row = tokens[i];
    if (row >= 0 && row < rows) tokens[i] = id_map[row];
}
