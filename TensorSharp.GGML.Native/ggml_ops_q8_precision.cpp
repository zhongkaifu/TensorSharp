// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_q8_precision.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml-impl.h"
#include <climits>
#include <cstdint>
#include <cstring>

namespace {
constexpr int block_elements = 32;
constexpr size_t block_bytes = sizeof(ggml_fp16_t) + block_elements;
constexpr int q5_block_elements = 256;
constexpr size_t q5_block_bytes = 176;

void q5_scale_min(const unsigned char* scales, int group, int& scale, int& minimum) {
    if (group < 4) {
        scale = scales[group] & 63;
        minimum = scales[group + 4] & 63;
    } else {
        scale = (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
        minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
}

void compute_cpu(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * weights = dst->src[0];
    const auto * input = dst->src[1];
    const int64_t count = ggml_nelements(dst);
    for (int64_t index = count * ith / nth; index < count * (ith + 1) / nth; ++index) {
        const int64_t row = index % dst->ne[0], column = index / dst->ne[0];
        const auto * w = static_cast<const char *>(weights->data) + row * weights->nb[1];
        const auto * x = static_cast<const char *>(input->data) + column * input->nb[1];
        double sum = 0;
        if (weights->type == GGML_TYPE_Q8_0) for (int64_t block = 0; block < weights->ne[0] / block_elements; ++block) {
            ggml_fp16_t scale_bits;
            std::memcpy(&scale_bits, w + block * block_bytes, sizeof(scale_bits));
            const float scale = ggml_fp16_to_fp32(scale_bits);
            for (int k = 0; k < block_elements; ++k) {
                int8_t quant;
                float activation;
                std::memcpy(&quant, w + block * block_bytes + sizeof(scale_bits) + k, sizeof(quant));
                std::memcpy(&activation, x + (block * block_elements + k) * input->nb[0], sizeof(activation));
                sum += double(scale * float(quant)) * activation;
            }
        // CPU Q5 keeps the independent double accumulator. CUDA's narrow
        // adjacent-pair/warp reduction and wide K-ordered FMA tiles each have
        // their own F32 error bounds; cross-backend/class bit identity is not
        // promised, while each CUDA width class preserves column invariance.
        } else for (int64_t block = 0; block < weights->ne[0] / q5_block_elements; ++block) {
            const auto* source = reinterpret_cast<const unsigned char*>(w + block * q5_block_bytes);
            ggml_fp16_t d_bits, minimum_bits;
            std::memcpy(&d_bits, source, sizeof(d_bits));
            std::memcpy(&minimum_bits, source + sizeof(d_bits), sizeof(minimum_bits));
            const float d = ggml_fp16_to_fp32(d_bits), d_minimum = ggml_fp16_to_fp32(minimum_bits);
            for (int group = 0; group < 8; ++group) {
                int scale, minimum;
                q5_scale_min(source + 4, group, scale, minimum);
                const float d_scale = d * float(scale), scaled_minimum = d_minimum * float(minimum);
                for (int k = 0; k < 32; ++k) {
                    const auto low = source[48 + (group / 2) * 32 + k];
                    const int quant = ((group & 1) ? low >> 4 : low & 15)
                        | (((source[16 + k] >> group) & 1) << 4);
                    const float weight = d_scale * float(quant) - scaled_minimum;
                    float activation;
                    std::memcpy(&activation, x + (block * q5_block_elements + group * 32 + k) * input->nb[0], sizeof(activation));
                    sum += double(weight) * activation;
                }
            }
        }
        static_cast<float *>(dst->data)[index] = float(sum);
    }
}
} // namespace

bool tsg_is_matmul_quant_f32(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->op != GGML_OP_CUSTOM || !tensor->src[0]
        || !tsg_matmul_quant_f32_type_supported(tensor->src[0]->type)) return false;
    ggml_custom_op_params parameters;
    std::memcpy(&parameters, tensor->op_params, sizeof(parameters));
    // Identify our callback without interpreting another custom op's userdata.
    return parameters.fun == compute_cpu;
}

bool tsg_is_matmul_q8_f32(const ggml_tensor * tensor) {
    return tsg_is_matmul_quant_f32(tensor) && tensor->src[0]->type == GGML_TYPE_Q8_0;
}

ggml_tensor * tsg_matmul_quant_f32(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * input) {
    GGML_ASSERT(weights && input && tsg_matmul_quant_f32_type_supported(weights->type) && input->type == GGML_TYPE_F32);
    const bool q5 = weights->type == GGML_TYPE_Q5_K;
    const int elements = q5 ? q5_block_elements : block_elements;
    const size_t bytes = q5 ? q5_block_bytes : block_bytes;
    GGML_ASSERT(weights->ne[0] > 0 && weights->ne[0] % elements == 0);
    GGML_ASSERT(weights->ne[0] == input->ne[0] && weights->ne[1] > 0 && input->ne[1] > 0);
    GGML_ASSERT(weights->ne[2] == 1 && weights->ne[3] == 1 && input->ne[2] == 1 && input->ne[3] == 1);
    GGML_ASSERT(weights->ne[0] <= INT_MAX && weights->ne[1] <= INT_MAX && input->ne[1] <= INT_MAX);
    GGML_ASSERT(ggml_blck_size(weights->type) == elements && ggml_type_size(weights->type) == bytes);
    GGML_ASSERT(weights->nb[0] == bytes && weights->nb[1] >= size_t(weights->ne[0] / elements) * bytes);
    GGML_ASSERT(weights->nb[1] % alignof(ggml_fp16_t) == 0);
    GGML_ASSERT(input->nb[0] >= sizeof(float) && input->nb[1] >= sizeof(float));
    GGML_ASSERT(input->nb[0] % sizeof(float) == 0 && input->nb[1] % sizeof(float) == 0);
    static tsg_dsv4_fused_desc description = [] {
        tsg_dsv4_fused_desc d;
        // Preserve the existing internal descriptor slot; its executor checks
        // the original weight type before selecting qualified Q8/Q5 kernels.
        d.kind = TSG_MATMUL_Q8_F32;
        return d;
    }();
    ggml_tensor * args[] = {weights, input};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, weights->ne[1], input->ne[1], 1, 1,
        args, 2, compute_cpu, GGML_N_TASKS_MAX, &description);
}

ggml_tensor * tsg_matmul_q8_f32(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * input) {
    GGML_ASSERT(weights && weights->type == GGML_TYPE_Q8_0);
    return tsg_matmul_quant_f32(ctx, weights, input);
}
