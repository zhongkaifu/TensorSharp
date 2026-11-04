// TensorSharp-owned instantiations of the pristine Ignis e600 BF16 small-T core.
// Upstream sources and license are retained unchanged under upstream/.
#include "ops/linear/bf16/bf16_small_t.cuh"

#include <cstddef>
#include <type_traits>

namespace tensorsharp::ninfer_bf16 {

// Preserve the current TensorSharp BF16 projection contract: the projection
// writes raw F32. Its existing caller, rather than this epilogue, rounds GQA
// output to BF16 or adds the output projection to the residual before rounding.
template <int Rows>
struct TsNinferBf16F32Output {
    float* data;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token,
                                         float value) const {
        data[static_cast<std::int64_t>(token) * Rows + row] = value;
    }
};

static_assert(sizeof(void*) == 8);
static_assert(sizeof(TsNinferBf16F32Output<14336>) == 8);
static_assert(alignof(TsNinferBf16F32Output<14336>) == 8);
static_assert(std::is_standard_layout_v<TsNinferBf16F32Output<14336>>);
static_assert(offsetof(TsNinferBf16F32Output<14336>, data) == 0);
static_assert(sizeof(ninfer::ops::detail::Bf16SmallTContiguousOutput) == 16);
static_assert(offsetof(ninfer::ops::detail::Bf16SmallTContiguousOutput, rows) == 8);

} // namespace tensorsharp::ninfer_bf16

namespace ninfer::ops::detail {

using TsGqaGeometry = Bf16GemvGeometry<14336, 5120>;
using TsOutputGeometry = Bf16GemvGeometry<5120, 6144>;
using TsGqaSchedule = Bf16LinearSmallTProductionSchedule<TsGqaGeometry, 8>::Type;
using TsOutputSchedule = Bf16LinearSmallTProductionSchedule<TsOutputGeometry, 8>::Type;
static_assert(TsGqaSchedule::kThreads == 128 && TsGqaSchedule::kRowsPerCta == 16);
static_assert(TsOutputSchedule::kThreads == 128 && TsOutputSchedule::kRowsPerCta == 16);
static_assert(TsGqaSchedule::kValuesPerLane == 8 && TsOutputSchedule::kValuesPerLane == 16);
static_assert(TsGqaSchedule::kWarpsPerRow == 1 && TsOutputSchedule::kWarpsPerRow == 1);
static_assert(TsGqaSchedule::kAccumulatorChains == 1 && TsOutputSchedule::kAccumulatorChains == 1);

// Instantiate the original global kernel itself. No computation body, phase
// order, accumulator, reduction, PDL boundary, or BF16 input decode is copied.
template __global__ void bf16_small_t_inner_kernel<
    TsGqaGeometry, 8, TsGqaSchedule,
    tensorsharp::ninfer_bf16::TsNinferBf16F32Output<14336>>(
    const __nv_bfloat16*, const __nv_bfloat16*,
    tensorsharp::ninfer_bf16::TsNinferBf16F32Output<14336>);
template __global__ void bf16_small_t_inner_kernel<
    TsOutputGeometry, 8, TsOutputSchedule,
    tensorsharp::ninfer_bf16::TsNinferBf16F32Output<5120>>(
    const __nv_bfloat16*, const __nv_bfloat16*,
    tensorsharp::ninfer_bf16::TsNinferBf16F32Output<5120>);

// Original BF16-output reference for a separate component arm. It is not a
// substitute for TensorSharp's raw F32 projection output.
template __global__ void bf16_small_t_inner_kernel<
    TsGqaGeometry, 8, TsGqaSchedule, Bf16SmallTContiguousOutput>(
    const __nv_bfloat16*, const __nv_bfloat16*, Bf16SmallTContiguousOutput);
template __global__ void bf16_small_t_inner_kernel<
    TsOutputGeometry, 8, TsOutputSchedule, Bf16SmallTContiguousOutput>(
    const __nv_bfloat16*, const __nv_bfloat16*, Bf16SmallTContiguousOutput);

} // namespace ninfer::ops::detail
