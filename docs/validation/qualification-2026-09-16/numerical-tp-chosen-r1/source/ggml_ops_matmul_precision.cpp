// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_matmul_precision.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml-cpu.h"
#include <climits>
#include <cstring>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace {
#if defined(__aarch64__)
float paired_f32_dot(const float * a, const float * b, int count) {
    // Keep neighboring products in the same accumulator, as the owned CUDA
    // kernel does. Separating alternating signs into SIMD lanes can build
    // large same-sign partials and discard small F32 source residuals before
    // the final cancellation (e.g. [3 + 2^-16, -3] repeated 2048 times).
    float32x4_t sums[4] = {vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0)};
    int i = 0;
    for (; i + 31 < count; i += 32) {
        for (int j = 0; j < 4; ++j) {
            const auto x = vmulq_f32(vld1q_f32(a + i + 8*j), vld1q_f32(b + i + 8*j));
            const auto y = vmulq_f32(vld1q_f32(a + i + 8*j + 4), vld1q_f32(b + i + 8*j + 4));
            sums[j] = vaddq_f32(sums[j], vpaddq_f32(x, y));
        }
    }
    double total = vaddvq_f32(vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3])));
    for (; i < count; ++i) total += double(a[i]) * b[i];
    return float(total);
}
#endif

float read_weight(const ggml_tensor * t, size_t offset) {
    const auto * p = static_cast<const char *>(t->data) + offset;
    switch (t->type) {
        case GGML_TYPE_F32: { float v; std::memcpy(&v, p, sizeof(v)); return v; }
        case GGML_TYPE_F16: { ggml_fp16_t v; std::memcpy(&v, p, sizeof(v)); return ggml_fp16_to_fp32(v); }
        case GGML_TYPE_BF16: { ggml_bf16_t v; std::memcpy(&v, p, sizeof(v)); return ggml_bf16_to_fp32(v); }
        default: GGML_ABORT("TensorSharp precise matmul requires floating-point weights");
    }
}

void compute_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto * desc = static_cast<const tsg_dsv4_fused_desc *>(userdata);
    const auto * a = dst->src[0];
    const auto * b = dst->src[1];
    const bool paired = desc->kind == TSG_MATMUL_ID_QUANT_PAIR;
    const bool quantized = desc->kind == TSG_MATMUL_ID_QUANT_STRIP || paired;
    const bool indexed = desc->kind == TSG_MATMUL_ID_F32 || quantized;
    const auto * quant_traits = quantized ? ggml_get_type_traits_cpu(a->type) : nullptr;
    const auto * activation_traits = quantized ? ggml_get_type_traits_cpu(quant_traits->vec_dot_type) : nullptr;
    std::vector<char> quant_input(quantized ? ggml_row_size(quant_traits->vec_dot_type, a->ne[0]) : 0);
    size_t last_input = SIZE_MAX;
    const int64_t count = ggml_nelements(dst);
    const bool vector_dot = a->type == GGML_TYPE_F32 && a->nb[0] == sizeof(float)
        && b->nb[0] == sizeof(float) && a->ne[0] <= INT_MAX;
#if !defined(__aarch64__)
    const auto dot = ggml_get_type_traits_cpu(GGML_TYPE_F32)->vec_dot;
#endif
    // Keep each worker's rows adjacent. Interleaving individual outputs makes
    // workers repeatedly invalidate one another's output cache lines.
    const int64_t first = count * ith / nth, end = count * (ith + 1) / nth;
    for (int64_t index = first; index < end; ++index) {
        int64_t rest = index;
        const int64_t row = rest % dst->ne[0]; rest /= dst->ne[0];
        const int64_t column = rest % dst->ne[1]; rest /= dst->ne[1];
        const int64_t i2 = rest % dst->ne[2], i3 = rest / dst->ne[2];
        if (paired) a = dst->src[i3 ? 3 : 0];
        size_t ao, bo;
        if (indexed) {
            const auto * ids = dst->src[2];
            int32_t expert;
            std::memcpy(&expert, static_cast<const char *>(ids->data) + column * ids->nb[0] + i2 * ids->nb[1], sizeof(expert));
            GGML_ASSERT(expert >= 0 && expert < a->ne[2]);
            ao = row * a->nb[1] + expert * a->nb[2];
            bo = (column % b->ne[1]) * b->nb[1] + i2 * b->nb[2];
        } else {
            ao = row * a->nb[1] + (i2 / (b->ne[2] / a->ne[2])) * a->nb[2]
                + (i3 / (b->ne[3] / a->ne[3])) * a->nb[3];
            bo = column * b->nb[1] + i2 * b->nb[2] + i3 * b->nb[3];
        }
        float value;
        if (quantized) {
            if (bo != last_input) {
                activation_traits->from_float(reinterpret_cast<const float *>(static_cast<const char *>(b->data) + bo),
                    quant_input.data(), a->ne[0]);
                last_input = bo;
            }
            quant_traits->vec_dot(int(a->ne[0]), &value, 0, static_cast<const char *>(a->data) + ao, 0,
                quant_input.data(), 0, 1);
        } else if (vector_dot) {
#if defined(__aarch64__)
            value = paired_f32_dot(reinterpret_cast<const float *>(static_cast<const char *>(a->data) + ao),
                reinterpret_cast<const float *>(static_cast<const char *>(b->data) + bo), int(a->ne[0]));
#else
            // Reuse ggml's public CPU SIMD dot primitive without narrowing
            // either source. Its architecture dispatch stays upstream-owned.
            dot(int(a->ne[0]), &value, 0, static_cast<const char *>(a->data) + ao, 0,
                static_cast<const char *>(b->data) + bo, 0, 1);
#endif
        } else {
            // Unusual strides and mixed weight formats use the portable
            // reference path with an extended accumulator.
            double sum = 0;
            for (int64_t k = 0; k < a->ne[0]; ++k)
                sum += double(read_weight(a, ao + k * a->nb[0])) * read_weight(b, bo + k * b->nb[0]);
            value = float(sum);
        }
        std::memcpy(static_cast<char *>(dst->data) + row * dst->nb[0] + column * dst->nb[1]
            + i2 * dst->nb[2] + i3 * dst->nb[3], &value, sizeof(value));
    }
}

ggml_tensor * create(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, ggml_tensor * ids) {
    GGML_ASSERT(a && b && b->type == GGML_TYPE_F32 && a->ne[0] == b->ne[0]);
    GGML_ASSERT(a->type == GGML_TYPE_F32 || a->type == GGML_TYPE_F16 || a->type == GGML_TYPE_BF16);
    static tsg_dsv4_fused_desc ordinary = [] { tsg_dsv4_fused_desc d; d.kind = TSG_MATMUL_F32; return d; }();
    static tsg_dsv4_fused_desc indexed = [] { tsg_dsv4_fused_desc d; d.kind = TSG_MATMUL_ID_F32; return d; }();
    ggml_tensor * args[] = {a, b, ids};
    if (ids) {
        GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[2] == 1 && ids->ne[3] == 1);
        GGML_ASSERT(a->ne[3] == 1 && b->ne[3] == 1 && ids->ne[1] == b->ne[2]);
        GGML_ASSERT(ids->ne[0] % b->ne[1] == 0);
        return ggml_custom_4d(ctx, GGML_TYPE_F32, a->ne[1], ids->ne[0], b->ne[2], 1,
            args, 3, compute_cpu, GGML_N_TASKS_MAX, &indexed);
    }
    GGML_ASSERT(b->ne[2] % a->ne[2] == 0 && b->ne[3] % a->ne[3] == 0);
    return ggml_custom_4d(ctx, GGML_TYPE_F32, a->ne[1], b->ne[1], b->ne[2], b->ne[3],
        args, 2, compute_cpu, GGML_N_TASKS_MAX, &ordinary);
}
} // namespace

extern "C" ggml_tensor * tsg_matmul_f32(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    return create(ctx, a, b, nullptr);
}
extern "C" ggml_tensor * tsg_matmul_id_f32(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b, ggml_tensor * ids) {
    return create(ctx, a, b, ids);
}
extern "C" void tsg_matmul_require_f32(ggml_context * ctx, ggml_tensor * node) {
    GGML_ASSERT(node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID);
    auto * custom = create(ctx, node->src[0], node->src[1], node->op == GGML_OP_MUL_MAT_ID ? node->src[2] : nullptr);
    GGML_ASSERT(ggml_are_same_shape(node, custom));
    node->op = custom->op;
    // Obtain the callback parameter representation from ggml's public custom
    // constructor instead of writing private precision slots in op_params.
    std::memcpy(node->op_params, custom->op_params, sizeof(node->op_params));
}

ggml_tensor * tsg_matmul_id_quant_strip(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b,
        ggml_tensor * ids, tsg_dsv4_fused_desc * desc) {
    GGML_ASSERT(desc && desc->kind == TSG_MATMUL_ID_QUANT_STRIP);
    GGML_ASSERT(a && b && ids && (a->type == GGML_TYPE_Q2_K || a->type == GGML_TYPE_Q4_K));
    GGML_ASSERT(b->type == GGML_TYPE_F32 && b->ne[0] == a->ne[0] && b->ne[1] == 1);
    GGML_ASSERT(a->ne[3] == 1 && b->ne[3] == 1 && b->nb[0] == sizeof(float));
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->ne[1] == b->ne[2] && ids->ne[2] == 1 && ids->ne[3] == 1);
    GGML_ASSERT(desc->i0 > 0 && desc->i1 >= 0 && int64_t(desc->i1) + a->ne[1] <= desc->i0);
    GGML_ASSERT(a->ne[0] <= INT_MAX);
    ggml_tensor * args[] = {a, b, ids};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, a->ne[1], ids->ne[0], b->ne[2], 1,
        args, 3, compute_cpu, GGML_N_TASKS_MAX, desc);
}

ggml_tensor * tsg_matmul_id_quant_pair(ggml_context * ctx, ggml_tensor * gate, ggml_tensor * up,
        ggml_tensor * input, ggml_tensor * ids, tsg_dsv4_fused_desc * desc) {
    GGML_ASSERT(desc && desc->kind == TSG_MATMUL_ID_QUANT_PAIR);
    GGML_ASSERT(gate && up && input && ids);
    GGML_ASSERT(gate->type == up->type && ggml_are_same_shape(gate, up));
    GGML_ASSERT(gate->type == GGML_TYPE_Q2_K || gate->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(ggml_is_contiguous(gate) && ggml_is_contiguous(up));
    GGML_ASSERT(input->type == GGML_TYPE_F32 && input->ne[0] == gate->ne[0] && input->ne[1] == 1);
    GGML_ASSERT(gate->ne[3] == 1 && input->ne[3] == 1 && input->nb[0] == sizeof(float));
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));
    GGML_ASSERT(ids->ne[1] == input->ne[2] && ids->ne[2] == 1 && ids->ne[3] == 1);
    GGML_ASSERT(desc->i0 > 0 && desc->i1 >= 0 && int64_t(desc->i1) + gate->ne[1] <= desc->i0);
    GGML_ASSERT(gate->ne[0] <= INT_MAX);
    ggml_tensor * args[] = {gate,input,ids,up};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,gate->ne[1],ids->ne[0],input->ne[2],2,
        args,4,compute_cpu,GGML_N_TASKS_MAX,desc);
}
