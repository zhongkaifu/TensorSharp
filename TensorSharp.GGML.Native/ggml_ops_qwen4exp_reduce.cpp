// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_qwen4exp_reduce.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml_ops_precision_policy.h"
#include <cstdint>

// Volatile intermediates enforce the same two rounding boundaries as separate
// ggml MUL and ADD kernels, including on compilers that contract a*b+c by default.
static void q4e_reduce_cpu(ggml_tensor * dst, int ith, int nth, void *)
{
    const ggml_tensor * experts = dst->src[0], * routes = dst->src[1];
    auto read = [](const ggml_tensor * tensor, int64_t col, int64_t expert, int64_t token) {
        return *reinterpret_cast<const float *>(static_cast<const uint8_t *>(tensor->data)
            + col * tensor->nb[0] + expert * tensor->nb[1] + token * tensor->nb[2]);
    };
    auto * out = static_cast<float *>(dst->data);
    for (int64_t i = ith; i < ggml_nelements(dst); i += nth)
    {
        const int64_t col = i % dst->ne[0], token = i / dst->ne[0];
        volatile float sum = read(experts, col, 0, token) * read(routes, 0, 0, token);
        for (int64_t k = 1; k < experts->ne[1]; ++k)
        {
            volatile float product = read(experts, col, k, token) * read(routes, 0, k, token);
            sum = sum + product;
        }
        out[i] = sum;
    }
}

ggml_tensor * tsg_q4e_prefill_expert_reduce(ggml_context * ctx,
    ggml_tensor * experts, ggml_tensor * routes)
{
    if (!ctx || !experts || !routes || experts->type != GGML_TYPE_F32 || routes->type != GGML_TYPE_F32
        || experts->ne[0] <= 0 || experts->ne[1] < 2 || experts->ne[1] > 64
        // The CUDA kernel maps tokens to grid.y, whose maximum is 65535.
        || experts->ne[2] <= TSG_PRECISION_DECODE_COLUMNS || experts->ne[2] > 65535 || experts->ne[3] != 1
        || routes->ne[0] != 1 || routes->ne[1] != experts->ne[1]
        || routes->ne[2] != experts->ne[2] || routes->ne[3] != 1
        || experts->nb[0] != sizeof(float) || routes->nb[0] != sizeof(float)
        || experts->nb[1] % sizeof(float) || experts->nb[2] % sizeof(float)
        || routes->nb[1] % sizeof(float) || routes->nb[2] % sizeof(float))
        return nullptr;
    static tsg_dsv4_fused_desc descriptor = [] {
        tsg_dsv4_fused_desc d;
        d.kind = TSG_Q4E_EXPERT_REDUCE;
        return d;
    }();
    ggml_tensor * inputs[] = {experts, routes};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, experts->ne[0], experts->ne[2], 1, 1,
        inputs, 2, q4e_reduce_cpu, GGML_N_TASKS_MAX, &descriptor);
}
