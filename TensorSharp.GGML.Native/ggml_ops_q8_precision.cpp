// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_q8_precision.h"
#include "ggml_ops_dsv4_fused.h"
#include <climits>
#include <cstdint>
#include <cstring>

namespace {
constexpr int block_elements = 32;
constexpr size_t block_bytes = sizeof(ggml_fp16_t) + block_elements;

void compute_cpu(ggml_tensor * dst, int ith, int nth, void *) {
    const auto * weights = dst->src[0];
    const auto * input = dst->src[1];
    const int64_t count = ggml_nelements(dst);
    for (int64_t index = count * ith / nth; index < count * (ith + 1) / nth; ++index) {
        const int64_t row = index % dst->ne[0], column = index / dst->ne[0];
        const auto * w = static_cast<const char *>(weights->data) + row * weights->nb[1];
        const auto * x = static_cast<const char *>(input->data) + column * input->nb[1];
        double sum = 0;
        for (int64_t block = 0; block < weights->ne[0] / block_elements; ++block) {
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
        }
        static_cast<float *>(dst->data)[index] = float(sum);
    }
}
} // namespace

ggml_tensor * tsg_matmul_q8_f32(ggml_context * ctx, ggml_tensor * weights, ggml_tensor * input) {
    GGML_ASSERT(weights && input && weights->type == GGML_TYPE_Q8_0 && input->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->ne[0] > 0 && weights->ne[0] % block_elements == 0);
    GGML_ASSERT(weights->ne[0] == input->ne[0] && weights->ne[1] > 0 && input->ne[1] > 0);
    GGML_ASSERT(weights->ne[2] == 1 && weights->ne[3] == 1 && input->ne[2] == 1 && input->ne[3] == 1);
    GGML_ASSERT(weights->ne[0] <= INT_MAX && weights->ne[1] <= INT_MAX && input->ne[1] <= INT_MAX);
    GGML_ASSERT(ggml_blck_size(GGML_TYPE_Q8_0) == block_elements && ggml_type_size(GGML_TYPE_Q8_0) == block_bytes);
    GGML_ASSERT(weights->nb[0] == block_bytes && weights->nb[1] >= size_t(weights->ne[0] / block_elements) * block_bytes);
    GGML_ASSERT(weights->nb[1] % alignof(ggml_fp16_t) == 0);
    GGML_ASSERT(input->nb[0] >= sizeof(float) && input->nb[1] >= sizeof(float));
    GGML_ASSERT(input->nb[0] % sizeof(float) == 0 && input->nb[1] % sizeof(float) == 0);
    static tsg_dsv4_fused_desc description = [] {
        tsg_dsv4_fused_desc d;
        d.kind = TSG_MATMUL_Q8_F32;
        return d;
    }();
    ggml_tensor * args[] = {weights, input};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, weights->ne[1], input->ne[1], 1, 1,
        args, 2, compute_cpu, GGML_N_TASKS_MAX, &description);
}
