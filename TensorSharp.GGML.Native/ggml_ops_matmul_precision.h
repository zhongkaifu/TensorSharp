// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"

// TensorSharp-owned matmuls preserve every F32 activation bit. Their custom
// CPU implementation is also the scheduler fallback on non-CUDA devices.
// Floating-point weights (F32/F16/BF16), strided inputs, batch broadcasting,
// and indexed expert/input-slot selection have the usual ggml semantics.
extern "C" ggml_tensor * tsg_matmul_f32(ggml_context *, ggml_tensor *, ggml_tensor *);
extern "C" ggml_tensor * tsg_matmul_id_f32(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *);
// Convert a graph node before scheduling, preserving its identity, sources,
// name, flags and all downstream references. Only MUL_MAT(_ID) is accepted.
extern "C" void tsg_matmul_require_f32(ggml_context *, ggml_tensor *);

#ifdef TSG_GGML_USE_CUDA
struct tsg_matmul_cuda_state;
tsg_matmul_cuda_state * tsg_matmul_cuda_init(ggml_backend_t);
void tsg_matmul_cuda_free(tsg_matmul_cuda_state *);
void tsg_matmul_cuda_compute(tsg_matmul_cuda_state *, ggml_tensor *);
#endif
