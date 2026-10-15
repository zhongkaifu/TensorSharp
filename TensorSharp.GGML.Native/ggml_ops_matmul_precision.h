// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml_ops_precision_policy.h"
#include "ggml_ops_q8_precision.h"

// Optional specialized construction, before a caller reshapes activation
// columns for ordinary backend batching. Q8_0 bytes remain resident in their
// original format; only the sensitive projection's activations retain F32.
// Returning null leaves every other type/role on the caller's existing path.
inline ggml_tensor* tsg_projection_q8_f32_if_required(ggml_context* ctx,
    ggml_tensor* weight, ggml_tensor* input, tsg_projection_role role) {
    return weight && tsg_projection_requires_f32(role) && weight->type == GGML_TYPE_Q8_0
        ? tsg_matmul_q8_f32(ctx, weight, input) : nullptr;
}

inline ggml_tensor* tsg_projection_quant_f32_if_required(ggml_context* ctx,
    ggml_tensor* weight, ggml_tensor* input, tsg_projection_role role) {
    return weight && tsg_projection_requires_f32(role) && tsg_matmul_quant_f32_type_supported(weight->type)
        ? tsg_matmul_quant_f32(ctx, weight, input) : nullptr;
}

inline ggml_tensor* tsg_projection_weight(ggml_context* ctx, ggml_tensor* weight, tsg_projection_role role) {
    // CPU's BF16/F16 dot traits narrow activations regardless of an accumulator
    // tag. Widening the small sensitive weight projection instead makes both
    // sources F32 on every backend. This public graph node has ordinary
    // allocator liveness and ownership, unlike a private vendor scratch copy.
    return weight && tsg_projection_requires_f32(role)
        && (weight->type == GGML_TYPE_F16 || weight->type == GGML_TYPE_BF16)
        ? ggml_cast(ctx, weight, GGML_TYPE_F32) : weight;
}

// Defined below; the strict CUDA path has its own pedantic math handle.
extern "C" void tsg_matmul_require_f32(ggml_context *, ggml_tensor *);

inline void tsg_require_projection_precision(ggml_context* ctx, ggml_tensor* node,
    tsg_projection_role role, bool owned_cuda = false) {
    if (!node || !tsg_projection_requires_f32(role)) return;
    while (node->view_src) node = node->view_src;
    if (node->op == GGML_OP_CONCAT) {
        tsg_require_projection_precision(ctx, node->src[0], role, owned_cuda);
        tsg_require_projection_precision(ctx, node->src[1], role, owned_cuda);
    } else if (node->op == GGML_OP_MUL_MAT && node->src[0]
        && (node->src[0]->type == GGML_TYPE_F32 || node->src[0]->type == GGML_TYPE_F16
            || node->src[0]->type == GGML_TYPE_BF16)) {
        if (owned_cuda) {
            // The sensitive graph cast and normal F32 activation layouts must
            // need no backend-private packing allocation. Their payloads are
            // already owned by the graph allocator, including wide prefill.
            const auto* a = node->src[0]; const auto* b = node->src[1];
            GGML_ASSERT(a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32
                && a->nb[0] == sizeof(float) && a->nb[1] % sizeof(float) == 0
                && b->nb[0] == sizeof(float) && b->nb[1] % sizeof(float) == 0);
            tsg_matmul_require_f32(ctx, node);
        } else ggml_prec_set_acc(node, GGML_PREC_F32);
    }
}

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
