// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// FlashAttention's launch template checks its launches with two PyTorch macros. This
// stands in for PyTorch's header (see tsg_cuda_flash_attn.cu): a failed launch is
// recorded for the caller, which then runs the attention elsewhere, instead of thrown.
#pragma once

#include <cuda_runtime.h>

// hardware_info.h reports through these and gets them from the header this replaces.
#include <cstdio>
#include <cstdlib>

extern bool tsg_flash_attn_failed;

#define C10_CUDA_CHECK(expr)                                         \
    do {                                                             \
        if ((expr) != cudaSuccess) ::tsg_flash_attn_failed = true;   \
    } while (0)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() C10_CUDA_CHECK(cudaGetLastError())
