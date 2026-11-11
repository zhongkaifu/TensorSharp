// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <tuple>

namespace at { namespace cuda { namespace philox {
__host__ __device__ inline std::tuple<uint64_t, uint64_t> unpack(at::PhiloxCudaState state)
{
    return std::make_tuple(state.seed_, state.offset_);
}
}}} // namespace at::cuda::philox
