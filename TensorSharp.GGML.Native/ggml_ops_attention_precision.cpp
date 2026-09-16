// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_attention_precision.h"
#include "ggml_ops_matmul_precision.h"
#include "ggml_ops_dsv4_fused.h"
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstring>
#include <new>
#include <unordered_set>
#include <vector>

namespace {
float read(const ggml_tensor * t, int64_t x, int64_t y = 0, int64_t h = 0, int64_t b = 0) {
    const char * p = static_cast<const char *>(t->data) + x * t->nb[0] + y * t->nb[1] + h * t->nb[2] + b * t->nb[3];
    if (t->type == GGML_TYPE_F32) {
        float value;
        std::memcpy(&value, p, 4);
        return value;
    }
    if (t->type == GGML_TYPE_F16) {
        ggml_fp16_t value;
        std::memcpy(&value, p, 2);
        return ggml_fp16_to_fp32(value);
    }
    ggml_bf16_t value;
    std::memcpy(&value, p, 2);
    return ggml_bf16_to_fp32(value);
}
void compact_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto & d = *static_cast<tsg_dsv4_fused_desc *>(userdata);
    const auto * mask = dst->src[0];
    const int64_t rows = ggml_nelements(dst) / dst->ne[0], capacity = dst->ne[0] - 1;
    for (int64_t row = rows * ith / nth; row < rows * (ith + 1) / nth; ++row) {
        auto * output = static_cast<int32_t *>(dst->data) + row * dst->ne[0];
        const int64_t query = row % dst->ne[1], h = row / dst->ne[1] % dst->ne[2];
        const int64_t batch = row / (dst->ne[1] * dst->ne[2]);
        int count = 0;
        bool exceptional = false;
        for (int key = 0; key < d.i0; ++key) {
            const float bias = read(mask, key, query, h, batch);
            if (bias == -INFINITY) continue;
            exceptional |= !std::isfinite(bias);
            if (count < capacity) output[count + 1] = key;
            ++count;
        }
        output[0] = exceptional ? -1 : count;
    }
}
void partial_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto & d = *static_cast<tsg_dsv4_fused_desc *>(userdata);
    const auto *q = dst->src[0], *k = dst->src[1], *v = dst->src[2], *mask = dst->src[3];
    const int64_t splits = dst->ne[1], heads = q->ne[2], queries = q->ne[1], width = v->ne[0];
    const int64_t rows = splits * heads * queries * q->ne[3];
    std::vector<double> out(width);
    for (int64_t row = rows * ith / nth; row < rows * (ith + 1) / nth; ++row) {
        const int64_t split = row % splits, h = (row / splits) % heads, query = (row / (splits * heads)) % queries;
        const int64_t batch = row / (splits * heads * queries);
        const auto * compact = dst->src[4];
        const int32_t * indices = nullptr;
        int64_t count = k->ne[1], chunk = d.i2;
        if (compact) {
            const int64_t index_row = query + queries *
                (h % compact->ne[2] + compact->ne[2] * (batch % compact->ne[3]));
            const auto * candidate = static_cast<const int32_t *>(compact->data) + index_row * compact->ne[0];
            if (candidate[0] >= 0 && candidate[0] < compact->ne[0]) {
                count = candidate[0];
                indices = candidate + 1;
            }
            chunk = std::max<int64_t>(16, ((count + splits - 1) / splits + 15) / 16 * 16);
        }
        const int64_t first = split * chunk, end = std::min(count, first + chunk);
        double maximum = -INFINITY, sum = 0;
        std::fill(out.begin(), out.end(), 0);
        for (int64_t entry = first; entry < end; ++entry) {
            const int64_t key = indices ? indices[entry] : entry;
            const float bias = d.i0 ? read(mask, key, query, h % mask->ne[2], batch % mask->ne[3]) : 0;
            if (bias == -INFINITY)
                continue;
            double dot = 0;
            for (int64_t x = 0; x < q->ne[0]; ++x)
                dot += double(read(q, x, query, h, batch)) *
                       read(k, x, key, h / (heads / k->ne[2]), batch / (q->ne[3] / k->ne[3]));
            const double score = dot * d.f0 + bias, next = std::max(maximum, score);
            const double correction = std::exp(maximum - next), weight = std::exp(score - next);
            sum = sum * correction + weight;
            for (int64_t x = 0; x < width; ++x)
                out[x] = out[x] * correction +
                         weight * read(v, x, key, h / (heads / v->ne[2]), batch / (q->ne[3] / v->ne[3]));
            maximum = next;
        }
        float * result = static_cast<float *>(dst->data) + row * (width + 2);
        for (int64_t x = 0; x < width; ++x)
            result[x] = float(out[x]);
        result[width] = float(maximum);
        result[width + 1] = float(sum);
    }
}
void finish_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto & d = *static_cast<tsg_dsv4_fused_desc *>(userdata);
    const auto *partial = dst->src[0], *sinks = dst->src[1];
    const int64_t width = dst->ne[0], splits = partial->ne[1], rows = ggml_nelements(dst) / width;
    for (int64_t row = rows * ith / nth; row < rows * (ith + 1) / nth; ++row) {
        const float * p = static_cast<const float *>(partial->data) + row * splits * (width + 2);
        double maximum = d.i1 ? read(sinks, row % dst->ne[1]) : -INFINITY;
        for (int64_t s = 0; s < splits; ++s)
            maximum = std::max(maximum, double(p[s * (width + 2) + width]));
        float * out = static_cast<float *>(dst->data) + row * width;
        if (maximum == -INFINITY || maximum == INFINITY) {
            std::fill(out, out + width, 0);
            continue;
        }
        double sum = d.i1 ? std::exp(read(sinks, row % dst->ne[1]) - maximum) : 0;
        for (int64_t s = 0; s < splits; ++s)
            sum += p[s * (width + 2) + width + 1] * std::exp(p[s * (width + 2) + width] - maximum);
        for (int64_t x = 0; x < width; ++x) {
            double value = 0;
            for (int64_t s = 0; s < splits; ++s)
                value += p[s * (width + 2) + x] * std::exp(p[s * (width + 2) + width] - maximum);
            out[x] = sum > 0 ? float(value / sum) : 0;
        }
    }
}
bool floating(const ggml_tensor * t) {
    return t && (t->type == GGML_TYPE_F32 || t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16);
}

void softmax_cpu(ggml_tensor * dst, int ith, int nth, void * userdata) {
    const auto & d = *static_cast<tsg_dsv4_fused_desc *>(userdata);
    const auto *scores = dst->src[0], *mask = dst->src[1], *sinks = dst->src[2];
    const int64_t keys = dst->ne[0], rows = ggml_nelements(dst) / keys;
    for (int64_t row = rows * ith / nth; row < rows * (ith + 1) / nth; ++row) {
        const int64_t query = row % dst->ne[1], h = (row / dst->ne[1]) % dst->ne[2];
        const int64_t batch = row / (dst->ne[1] * dst->ne[2]);
        float * output = static_cast<float *>(dst->data) + row * keys;
        double maximum = d.i1 ? read(sinks, h) : -INFINITY;
        for (int64_t key = 0; key < keys; ++key) {
            const float bias = d.i0 ? read(mask, key, query, h % mask->ne[2], batch % mask->ne[3]) : 0;
            output[key] = read(scores, key, query, h, batch) * d.f0 + bias;
            maximum = std::max(maximum, double(output[key]));
        }
        double sum = d.i1 && std::isfinite(maximum) ? std::exp(read(sinks, h) - maximum) : 0;
        for (int64_t key = 0; key < keys; ++key) {
            output[key] = std::isfinite(maximum) ? float(std::exp(output[key] - maximum)) : 0;
            sum += output[key];
        }
        for (int64_t key = 0; key < keys; ++key)
            output[key] = sum > 0 ? float(output[key] / sum) : 0;
    }
}

ggml_tensor * tiled_attention(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask,
                              ggml_tensor * sinks, float scale) {
    // Short-key prefill benefits from larger SGEMMs; a 512-MiB score budget
    // avoids repeated small query tiles there. Keep the 96-MiB bound at long
    // contexts (or one query row), and cap every tile at 512 queries. Chained
    // in-place writes let the allocator reuse temporary tiles across prefill.
    const int64_t score_row_bytes = k->ne[1] * q->ne[2] * q->ne[3] * sizeof(float);
    const int64_t score_budget = (k->ne[1] < 8192 ? int64_t(512) : int64_t(96)) * 1024 * 1024;
    const int64_t query_limit = std::max<int64_t>(1, std::min<int64_t>(512, score_budget / score_row_bytes));
    const int64_t tile_queries = query_limit >= 64 ? query_limit / 64 * 64 : query_limit;
    auto * values = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    auto * output = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, v->ne[0], q->ne[2], q->ne[1], q->ne[3]);
    for (int64_t first = 0; first < q->ne[1]; first += tile_queries) {
        const int64_t count = std::min(tile_queries, q->ne[1] - first);
        auto * query =
            ggml_view_4d(ctx, q, q->ne[0], count, q->ne[2], q->ne[3], q->nb[1], q->nb[2], q->nb[3], first * q->nb[1]);
        query->nb[0] = q->nb[0];
        auto * bias = mask ? ggml_view_4d(ctx, mask, mask->ne[0], count, mask->ne[2], mask->ne[3], mask->nb[1],
                                          mask->nb[2], mask->nb[3], first * mask->nb[1])
                           : nullptr;
        if (bias) bias->nb[0] = mask->nb[0];
        auto * scores = tsg_matmul_f32(ctx, k, query);
        auto * desc = new (ggml_new_buffer(ctx, sizeof(tsg_dsv4_fused_desc))) tsg_dsv4_fused_desc;
        desc->kind = TSG_ATTN_F32_SOFTMAX;
        desc->i0 = mask != nullptr;
        desc->i1 = sinks != nullptr;
        desc->f0 = scale;
        ggml_tensor * args[] = {bias ? bias : scores, sinks ? sinks : scores};
        auto * probabilities = ggml_custom_inplace(ctx, scores, args, 2, softmax_cpu, GGML_N_TASKS_MAX, desc);
        auto * weighted = tsg_matmul_f32(ctx, values, probabilities);
        auto * tile = ggml_cont(ctx, ggml_permute(ctx, weighted, 0, 2, 1, 3));
        output =
            ggml_set_inplace(ctx, output, tile, output->nb[1], output->nb[2], output->nb[3], first * output->nb[2]);
    }
    return output;
}
} // namespace

static ggml_tensor * attention_impl(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                    ggml_tensor * mask, ggml_tensor * sinks, float scale, int capacity) {
    GGML_ASSERT(q && q->type == GGML_TYPE_F32 && floating(k) && floating(v) && std::isfinite(scale));
    GGML_ASSERT(q->ne[0] == k->ne[0] && k->ne[1] == v->ne[1] && q->ne[0] <= 2048 && v->ne[0] <= 2048);
    GGML_ASSERT(q->ne[2] % k->ne[2] == 0 && q->ne[2] % v->ne[2] == 0);
    GGML_ASSERT(q->ne[3] % k->ne[3] == 0 && q->ne[3] % v->ne[3] == 0);
    if (mask)
        GGML_ASSERT(floating(mask) && mask->ne[0] >= k->ne[1] && mask->ne[1] >= q->ne[1] &&
                    q->ne[2] % mask->ne[2] == 0 && q->ne[3] % mask->ne[3] == 0);
    if (sinks)
        GGML_ASSERT(sinks->type == GGML_TYPE_F32 && ggml_nelements(sinks) == q->ne[2] && ggml_is_vector(sinks));
    GGML_ASSERT(capacity >= 0 && capacity < INT_MAX && k->ne[1] <= INT_MAX);
    if (capacity == 0 && q->ne[1] > 4)
        return tiled_attention(ctx, q, k, v, mask, sinks, scale);
    ggml_tensor * compact = nullptr;
    if (capacity > 0) {
        GGML_ASSERT(mask);
        auto * desc = new (ggml_new_buffer(ctx, sizeof(tsg_dsv4_fused_desc))) tsg_dsv4_fused_desc;
        desc->kind = TSG_ATTN_MASK_COMPACT;
        desc->i0 = int(k->ne[1]);
        ggml_tensor * args[] = {mask};
        compact = ggml_custom_4d(ctx, GGML_TYPE_I32, int64_t(capacity) + 1, q->ne[1], mask->ne[2], mask->ne[3],
            args, 1, compact_cpu, GGML_N_TASKS_MAX, desc);
    }
    // Bound scratch independently of context length, with enough split-key
    // parallelism for decode and fewer partitions when queries fill the GPU.
    const int64_t query_rows = q->ne[1] * q->ne[2] * q->ne[3];
    const int max_splits = query_rows >= 512 ? 1 : q->ne[1] <= 4 ? 16 : 4;
    const int64_t extent = capacity > 0 ? std::min<int64_t>(capacity, k->ne[1]) : k->ne[1];
    const int64_t chunk = std::max<int64_t>(128, ((extent + max_splits - 1) / max_splits + 15) / 16 * 16);
    const int64_t splits = (extent + chunk - 1) / chunk;
    auto * first = new (ggml_new_buffer(ctx, sizeof(tsg_dsv4_fused_desc))) tsg_dsv4_fused_desc;
    first->kind = TSG_ATTN_F32_PARTIAL;
    first->i0 = mask != nullptr;
    first->i2 = int(chunk);
    first->f0 = scale;
    ggml_tensor * args[] = {q, k, v, mask ? mask : q, compact};
    auto * partial = ggml_custom_4d(ctx, GGML_TYPE_F32, v->ne[0] + 2, splits, q->ne[2], q->ne[1] * q->ne[3], args, compact ? 5 : 4,
                                    partial_cpu, GGML_N_TASKS_MAX, first);
    auto * last = new (ggml_new_buffer(ctx, sizeof(tsg_dsv4_fused_desc))) tsg_dsv4_fused_desc;
    last->kind = TSG_ATTN_F32_FINISH;
    last->i1 = sinks != nullptr;
    ggml_tensor * finish_args[] = {partial, sinks ? sinks : q};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, v->ne[0], q->ne[2], q->ne[1], q->ne[3], finish_args, 2, finish_cpu,
                          GGML_N_TASKS_MAX, last);
}

extern "C" ggml_tensor * tsg_attention_f32(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
    ggml_tensor * mask, ggml_tensor * sinks, float scale) {
    return attention_impl(ctx, q, k, v, mask, sinks, scale, 0);
}

extern "C" ggml_tensor * tsg_attention_f32_sparse(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
    ggml_tensor * mask, ggml_tensor * sinks, float scale, int capacity) {
    return attention_impl(ctx, q, k, v, mask, sinks, scale, capacity);
}

extern "C" ggml_tensor * tsg_attention_f32_on_backend(ggml_context * ctx, ggml_backend_sched_t scheduler,
    ggml_backend_t backend, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
    ggml_tensor * mask, ggml_tensor * sinks, float scale, int sparse_capacity) noexcept(false) {
    GGML_ASSERT(scheduler && backend);
    auto * output = attention_impl(ctx, q, k, v, mask, sinks, scale, sparse_capacity);
    std::vector<ggml_tensor *> pending = {output};
    std::unordered_set<ggml_tensor *> visited = {q, k, v, mask, sinks, nullptr};
    std::unordered_set<ggml_tensor *> owned;
    while (!pending.empty()) {
        auto * node = pending.back();
        pending.pop_back();
        if (!visited.insert(node).second) continue;
        owned.insert(node);
        for (auto * source : node->src) pending.push_back(source);
    }
    for (auto * node : owned) {
        // A view of an existing input must keep that input's buffer/device;
        // the scheduler copies the view when an owned compute node reads it.
        // Do not follow a flattened view_src through the input boundary.
        if (node->view_src && !owned.count(node->view_src)) {
            GGML_ASSERT(node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE);
            continue;
        }
        GGML_ASSERT(node->op == GGML_OP_NONE || ggml_backend_supports_op(backend, node));
        ggml_backend_sched_set_tensor_backend(scheduler, node, backend);
        // SET/custom-in-place tensors share their root allocation. Pinning
        // only their last writer lets the scheduler allocate that root on a
        // different device from earlier tiles and their later writers.
    }
    return output;
}
