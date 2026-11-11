// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The dropout seed FlashAttention's forward kernel unpacks. TensorSharp never asks for
// dropout, so this only has to be there (see tsg_cuda_flash_attn.cu).
#pragma once

#include <cstdint>

namespace at {
struct PhiloxCudaState
{
    uint64_t seed_ = 0;
    uint64_t offset_ = 0;
};
} // namespace at
