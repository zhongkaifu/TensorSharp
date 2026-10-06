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

struct tsg_dsv4_fused_desc;
// Keep construction and device qualification on the same quantization set.
// Every type here has upstream MMQ tile arithmetic; the owned strip changes
// row ownership only and consumes the checkpoint's original quantized bytes.
// IQ1_M has MMVQ support but no MMQ tiles and is deliberately absent.
constexpr bool tsg_matmul_id_quant_strip_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}
// The descriptor must outlive the graph. CUDA support is checked before
// construction; the CPU callback provides a portable quantized fallback.
ggml_tensor * tsg_matmul_id_quant_strip(ggml_context *, ggml_tensor *, ggml_tensor *,
    ggml_tensor *, tsg_dsv4_fused_desc *);

ggml_tensor * tsg_matmul_id_quant_pair(ggml_context *, ggml_tensor *, ggml_tensor *, ggml_tensor *,
    ggml_tensor *, tsg_dsv4_fused_desc *);

#ifdef TSG_GGML_USE_CUDA
struct tsg_matmul_cuda_state;
tsg_matmul_cuda_state * tsg_matmul_cuda_init(ggml_backend_t);
void tsg_matmul_cuda_free(tsg_matmul_cuda_state *);
void tsg_matmul_cuda_compute(tsg_matmul_cuda_state *, ggml_tensor *);
bool tsg_matmul_id_quant_strip_supported(ggml_backend_t, const ggml_tensor *,
    int64_t tokens, int64_t full_rows, int64_t first_row);
#endif
