// Copyright (c) Zhongkai Fu. TensorSharp adapters under Apache-2.0.
// Original core/reference sources retain their unchanged licenses and notices.
#include "upstream/kernel/vendor/src/ops/kernel/gqa_attention_prefill_bf16.cuh"
#include "tensorsharp_ninfer_prefill_bf16.cuh"
#include <cstddef>

using ninfer::ops::Gqa27Geometry;
using ninfer::ops::GqaPrefillDirectMetadata;
using ninfer::ops::TsNinferPrefillPlane;
static_assert(sizeof(void*) == 8);
static_assert(sizeof(GqaPrefillDirectMetadata) == 8);
static_assert(offsetof(TsNinferPrefillPlane, blocks) == 0 && offsetof(TsNinferPrefillPlane, offset) == 8);

namespace ninfer::ops {
#define TS_PREFILL_ARGUMENTS \
    const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, GqaPrefillDirectMetadata, \
    const std::int32_t*, float, __nv_bfloat16*, std::int32_t, std::int32_t, std::int32_t, std::int32_t, \
    __nv_bfloat16*, float*, float*, std::int32_t, float*, float*, float*, std::int32_t
template __global__ void ts_ni_prefill_bf16_kernel<Gqa27Geometry, GqaPrefillDirectMetadata, false, false>(TS_PREFILL_ARGUMENTS);
// The independent component oracle runs the exact original page64 entry on
// separately serialized contiguous original cache planes and I32 page tables.
template __global__ void gqa_attention_prefill_bf16_kernel<Gqa27Geometry, GqaPrefillDirectMetadata, false, false>(TS_PREFILL_ARGUMENTS);
template __global__ void gqa_attention_prefill_reduce_kernel<Gqa27Geometry>(
    const float*, const float*, const float*, float, std::int32_t, std::int32_t, __nv_bfloat16*);
#undef TS_PREFILL_ARGUMENTS
}

extern "C" __global__ void ts_ni_prefill_pack(
    const float* input, __nv_bfloat16* output, unsigned long long count,
    TsNinferPrefillPlane* planes, const std::uint8_t* const* blocks,
    long long k_offset, long long v_offset, std::int32_t* positions,
    std::int32_t base_position, std::int32_t* identity, std::int32_t pages)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = __float2bfloat16_rn(input[i]);
    if (i < (unsigned)pages) identity[i] = static_cast<std::int32_t>(i);
    if (i == 0) {
        planes[0] = {blocks, k_offset}; planes[1] = {blocks, v_offset}; positions[0] = base_position;
    }
}

extern "C" __global__ void ts_ni_prefill_gate(
    const __nv_bfloat16* attention, const float* gate, float* output,
    unsigned long long count)
{
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        // The core already rounded normalized attention toBF16. Preserve the
        // exact existing target boundary BF16(BF16(attention)*sigmoid(F32gate)).
        const float value = __bfloat162float(attention[i]);
        output[i] = __bfloat162float(__float2bfloat16_rn(value * (1.0f / (1.0f + expf(-gate[i])))));
    }
}
