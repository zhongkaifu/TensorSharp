// TensorSharp-owned fixed instantiations and F32 store policy only.
// Ignis e600 original MMA core and Apache-2.0 license are pristine in upstream/.
// The observed Ignis binary uses this 32x32x256/S3 schedule; the e600 source's
// production alias uses a different schedule. Neither core nor alias is edited.
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma.cuh"

#include <cstddef>
#include <type_traits>

namespace tensorsharp::ninfer_bf16_mma {

// Raw F32 preserves TensorSharp's caller-owned rounding/residual boundary.
// No alpha, beta, BF16 conversion, or residual operation is added here.
template <int Rows>
struct TsNinferBf16MmaF32Output {
    float* data;

    __device__ __forceinline__ TsNinferBf16MmaF32Output tile(std::int32_t) const {
        return *this;
    }

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token,
                                         float value) const {
        data[static_cast<std::int64_t>(token) * Rows + row] = value;
    }
};

static_assert(sizeof(void*) == 8);
static_assert(sizeof(TsNinferBf16MmaF32Output<14336>) == 8);
static_assert(alignof(TsNinferBf16MmaF32Output<14336>) == 8);
static_assert(std::is_standard_layout_v<TsNinferBf16MmaF32Output<14336>>);
static_assert(offsetof(TsNinferBf16MmaF32Output<14336>, data) == 0);
static_assert(sizeof(ninfer::ops::detail::Bf16MmaContiguousOutput) == 16);
static_assert(alignof(ninfer::ops::detail::Bf16MmaContiguousOutput) == 8);
static_assert(offsetof(ninfer::ops::detail::Bf16MmaContiguousOutput, leading_dim) == 8);

} // namespace tensorsharp::ninfer_bf16_mma

namespace ninfer::ops::detail {

using TsGqaGeometry = Bf16GemvGeometry<14336, 5120>;
using TsOutputGeometry = Bf16GemvGeometry<5120, 6144>;
using TsObservedMmaSchedule = Bf16MmaSchedule<
    32, 32, 256, 16, 8, 3, 1, Cache::cg, Cache::cg,
    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast,
    Bf16MmaSwizzle::Xor64, 1>;
static_assert(TsObservedMmaSchedule::kThreads == 256);
static_assert(TsObservedMmaSchedule::kSharedBytes == 98304);
static_assert(TsObservedMmaSchedule::kMmaK == 16);
static_assert(TsGqaGeometry::kInputRows % TsObservedMmaSchedule::kBlockK == 0);
static_assert(TsOutputGeometry::kInputRows % TsObservedMmaSchedule::kBlockK == 0);

// FullTokens=false is required: eight live tokens occupy a 32-token tile.
// Instantiate the unmodified original global, not a copied arithmetic body.
template __global__ void bf16_gemm_mma_kernel<
    TsGqaGeometry, TsObservedMmaSchedule, false,
    tensorsharp::ninfer_bf16_mma::TsNinferBf16MmaF32Output<14336>>(
    const __nv_bfloat16*, const __nv_bfloat16*,
    tensorsharp::ninfer_bf16_mma::TsNinferBf16MmaF32Output<14336>, int);
template __global__ void bf16_gemm_mma_kernel<
    TsOutputGeometry, TsObservedMmaSchedule, false,
    tensorsharp::ninfer_bf16_mma::TsNinferBf16MmaF32Output<5120>>(
    const __nv_bfloat16*, const __nv_bfloat16*,
    tensorsharp::ninfer_bf16_mma::TsNinferBf16MmaF32Output<5120>, int);

// Original BF16 epilogue is a component reference only; it must not replace
// TensorSharp's F32 output followed by its existing caller-owned consumer.
template __global__ void bf16_gemm_mma_kernel<
    TsGqaGeometry, TsObservedMmaSchedule, false, Bf16MmaContiguousOutput>(
    const __nv_bfloat16*, const __nv_bfloat16*, Bf16MmaContiguousOutput, int);
template __global__ void bf16_gemm_mma_kernel<
    TsOutputGeometry, TsObservedMmaSchedule, false, Bf16MmaContiguousOutput>(
    const __nv_bfloat16*, const __nv_bfloat16*, Bf16MmaContiguousOutput, int);

} // namespace ninfer::ops::detail
