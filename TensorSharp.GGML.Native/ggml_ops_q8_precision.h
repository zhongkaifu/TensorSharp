// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"

// Two-dimensional Q8_0 [K,M] weights times F32 [K,N] activations, producing
// contiguous F32 [M,N]. K must be a multiple of 32. Weight rows may be padded;
// F32 activations may have arbitrary positive, element-aligned strides.
// Activations are never quantized or narrowed. CPU fallback uses a double
// accumulator; the owned CUDA implementation uses F32 multiply-adds.
ggml_tensor * tsg_matmul_q8_f32(ggml_context *, ggml_tensor *, ggml_tensor *);

#ifdef TSG_GGML_USE_CUDA
void tsg_matmul_q8_cuda_compute(ggml_tensor *, ggml_backend_t cuda_backend);
#endif
