// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_attention_precision.h"
#include "ggml_ops_precision_policy.h"
#include "ggml-alloc.h"
#include "precision_test_utils.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml_ops_dsv4_fused.h"
#include "ggml_ops_matmul_precision.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"
#else
#include "ggml-cpu.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>

enum class attention_data { random, source_bits, probability_bits, large_values, exact_uniform };

struct attention_case
{
    int head = 512, value_head = 512, queries = 5, keys = 129;
    int heads = 4, kv_heads = 1, batches = 1, kv_batches = 1;
    int value_heads = 0, value_batches = 0;
    int mask_heads = 1, mask_batches = 1;
    int sparse_capacity = 0;
    bool mask = true, sink = true, padded = false, interleaved = false, shared_kv = false;
    ggml_type kv_type = GGML_TYPE_F16, mask_type = GGML_TYPE_F16;
    ggml_type value_type = GGML_TYPE_COUNT;
    attention_data data = attention_data::random;
};

static double timed_compute(ggml_backend_t backend, ggml_cgraph * graph, int repeats)
{
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i)
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Precision attention compute failed");
    ggml_backend_synchronize(backend);
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / repeats;
}

static void check(ggml_backend_t allocator, ggml_backend_t backend, const attention_case & c)
{
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Cannot initialize attention test context");
    input_tensor q(ctx, GGML_TYPE_F32, {c.head, c.queries, c.heads, c.batches}, c.padded, c.interleaved);
    input_tensor k(ctx, c.kv_type, {c.head, c.keys, c.kv_heads, c.kv_batches}, c.padded, c.interleaved);
    input_tensor v(ctx, c.value_type == GGML_TYPE_COUNT ? c.kv_type : c.value_type,
        {c.value_head, c.keys, c.value_heads ? c.value_heads : c.kv_heads,
         c.value_batches ? c.value_batches : c.kv_batches}, c.padded, c.interleaved);
    input_tensor mask(ctx, c.mask_type, {c.keys, c.queries + 3, c.mask_heads, c.mask_batches}, c.padded, c.interleaved);
    input_tensor sink(ctx, GGML_TYPE_F32, {c.heads, 1, 1, 1});
    auto & values = c.shared_kv ? k : v;
    const float scale = c.data == attention_data::random ? 1.0f / std::sqrt(float(c.head)) : 1.0f;
    auto * output = tsg_attention_f32_sparse(ctx, q.tensor, k.tensor, values.tensor,
        c.mask ? mask.tensor : nullptr, c.sink ? sink.tensor : nullptr, scale, c.sparse_capacity);
    require(output != nullptr, "Cannot create TensorSharp attention graph");
    require(output->ne[0] == c.value_head && output->ne[1] == c.heads &&
        output->ne[2] == c.queries && output->ne[3] == c.batches, "Precision attention output shape changed");
    auto * graph = ggml_new_graph(ctx);
    // Exercise an ordinary ggml node after the owned attention nodes.
    auto * result_tensor = ggml_scale(ctx, output, 0.5f);
    ggml_build_forward_expand(graph, result_tensor);
    bool has_owned_precision = false;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
    {
        const auto * node = ggml_graph_node(graph, i);
        require(node->op != GGML_OP_FLASH_ATTN_EXT, "Owned F32 attention cannot fall back to reduced-precision flash attention");
        has_owned_precision = has_owned_precision || node->op == GGML_OP_CUSTOM;
    }
    require(has_owned_precision, "Attention must contain TensorSharp's F32 implementation");
    ggml_tensor * baseline = nullptr;
    ggml_cgraph * baseline_graph = nullptr;
    if (c.data == attention_data::exact_uniform)
    {
        baseline = ggml_flash_attn_ext(ctx, q.tensor, k.tensor, values.tensor,
            c.mask ? mask.tensor : nullptr, scale, 0.0f, 0.0f);
        if (c.sink) ggml_flash_attn_ext_add_sinks(baseline, sink.tensor);
        require(ggml_prec_set_acc(baseline, GGML_PREC_F32), "Cannot set baseline attention accumulator");
        if (ggml_backend_supports_op(allocator, baseline))
        {
            baseline_graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(baseline_graph, baseline);
        }
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, allocator);
    require(buffer != nullptr, "Cannot allocate precision attention graph");
    std::mt19937 random(94151 + c.queries + c.keys);
    std::uniform_real_distribution<float> uniform(-0.5f, 0.5f);
    for (float & value : q.values) value = uniform(random);
    for (float & value : k.values) value = uniform(random);
    for (float & value : v.values) value = uniform(random);
    for (int h = 0; h < c.heads; ++h) sink.values[h] = float(h % 5) - 2.0f;
    const float negative_infinity = -std::numeric_limits<float>::infinity();
    for (int batch = 0; batch < c.mask_batches; ++batch)
    for (int head = 0; head < c.mask_heads; ++head)
    for (int query = 0; query < mask.tensor->ne[1]; ++query)
    for (int key = 0; key < c.keys; ++key)
    {
        // Empty rows, one-key rows, a causal boundary and noncontiguous holes.
        const int visible = query == 0 ? 0 : query == 1 ? 1 :
            std::max(1, c.keys - c.queries + query + 1);
        const bool selected = key < visible && (key < 2 || (key + head + batch) % 7 != 0);
        mask.values[mask.logical(key, query, head, batch)] = selected
            ? -0.125f * ((key + 2 * head + batch) % 5) : negative_infinity;
    }
    if (c.data != attention_data::random)
    {
        std::fill(q.values.begin(), q.values.end(), 0.0f);
        std::fill(k.values.begin(), k.values.end(), 0.0f);
        std::fill(v.values.begin(), v.values.end(), 0.0f);
        std::fill(mask.values.begin(), mask.values.end(), 0.0f);
        std::fill(sink.values.begin(), sink.values.end(), 0.0f);
        for (int batch = 0; batch < c.batches; ++batch)
        for (int head = 0; head < c.heads; ++head)
        for (int query = 0; query < c.queries; ++query)
            q.values[q.logical(0, query, head, batch)] = c.data == attention_data::source_bits ? 1.0f + 0x1p-12f
                : c.data == attention_data::probability_bits ? 1.0f : 0.0f;
        for (int batch = 0; batch < c.kv_batches; ++batch)
        for (int head = 0; head < c.kv_heads; ++head)
        for (int key = 0; key < c.keys; ++key)
        {
            k.values[k.logical(0, key, head, batch)] = c.data == attention_data::source_bits
                ? (key == 0 ? 1.0f : -1.0f) : c.data == attention_data::probability_bits ? 0.125f * key : 0.0f;
            for (int d = 0; d < c.value_head; ++d)
                v.values[v.logical(d, key, head, batch)] = c.data == attention_data::large_values ? 70000.125f
                    : c.data == attention_data::exact_uniform ? float((d + 3 * key + head + batch) % 31 - 15) / 32.0f
                    : key == 0 ? 1.0f : -1.0f;
        }
        // There are 127/511 visible keys and one zero-valued sink, making each
        // probability exactly 1/128 or 1/512 in BOTH F16 and F32 arithmetic.
        if (c.data == attention_data::exact_uniform && c.mask)
            for (int query = 0; query < mask.tensor->ne[1]; ++query)
                mask.values[mask.logical(c.keys - 1, query, 0, 0)] = negative_infinity;
    }
    q.upload(); k.upload(); v.upload(); mask.upload(); sink.upload();

    auto validate_compaction = [&]() {
        if (c.sparse_capacity == 0) return;
        auto * compact = output->src[0]->src[4];
        require(compact && compact->type == GGML_TYPE_I32, "Missing ordered mask compaction");
        std::vector<int32_t> actual(ggml_nelements(compact));
        ggml_backend_tensor_get(compact, actual.data(), 0, actual.size() * sizeof(int32_t));
        for (int batch = 0; batch < c.mask_batches; ++batch)
        for (int head = 0; head < c.mask_heads; ++head)
        for (int query = 0; query < c.queries; ++query) {
            size_t row = query + c.queries * (head + c.mask_heads * batch);
            const auto * indices = actual.data() + row * (c.sparse_capacity + 1);
            int selected = 0;
            for (int key = 0; key < c.keys; ++key)
                if (mask.at(key, query, head, batch) != -INFINITY) {
                    if (selected < c.sparse_capacity)
                        require(indices[selected + 1] == key, "Mask compaction lost or reordered a visible key");
                    ++selected;
                }
            require(indices[0] == selected, "Compaction count must include overflow entries and exclude stale entries");
        }
    };

    auto validate = [&](ggml_tensor * actual_tensor, double output_scale)
    {
        std::vector<float> actual(ggml_nelements(actual_tensor));
        ggml_backend_tensor_get(actual_tensor, actual.data(), 0, actual.size() * sizeof(float));
        std::vector<double> scores(c.keys), probabilities(c.keys), reference(c.value_head);
        double maximum = 0, squared_error = 0, squared_reference = 0;
        size_t failures = 0;
        for (int batch = 0; batch < c.batches; ++batch)
        for (int query = 0; query < c.queries; ++query)
        for (int head = 0; head < c.heads; ++head)
        {
            const int kv_head = head / (c.heads / c.kv_heads), kv_batch = batch / (c.batches / c.kv_batches);
            double maximum_score = c.sink ? sink.at(head, 0) : -INFINITY;
            for (int key = 0; key < c.keys; ++key)
            {
                double dot = 0;
                for (int d = 0; d < c.head; ++d) dot += double(q.at(d, query, head, batch)) * k.at(d, key, kv_head, kv_batch);
                scores[key] = double(scale) * dot + (c.mask ? mask.at(key, query, head % c.mask_heads, batch % c.mask_batches) : 0.0);
                maximum_score = std::max(maximum_score, scores[key]);
            }
            double denominator = c.sink && std::isfinite(maximum_score) ? std::exp(sink.at(head, 0) - maximum_score) : 0.0;
            for (int key = 0; key < c.keys; ++key)
            {
                probabilities[key] = std::isfinite(scores[key]) ? std::exp(scores[key] - maximum_score) : 0.0;
                denominator += probabilities[key];
            }
            for (int d = 0; d < c.value_head; ++d)
            {
                double value = 0;
                for (int key = 0; key < c.keys; ++key)
                    value += probabilities[key] * values.at(d, key,
                        head / (c.heads / values.tensor->ne[2]), batch / (c.batches / values.tensor->ne[3]));
                reference[d] = denominator > 0 ? output_scale * value / denominator : 0.0;
                const size_t index = d + c.value_head * (head + c.heads * (query + c.queries * batch));
                require(std::isfinite(actual[index]) && std::isfinite(reference[d]), "Non-finite precision attention result");
                const double error = std::abs(actual[index] - reference[d]);
                maximum = std::max(maximum, error);
                squared_error += error * error;
                squared_reference += reference[d] * reference[d];
                const double tolerance = 3e-6 + 3e-6 * std::abs(reference[d]);
                if (error > tolerance && failures++ == 0)
                    std::fprintf(stderr, "Attention mismatch index=%zu reference=%.12g actual=%.12g error=%.8g tolerance=%.8g\n",
                        index, reference[d], actual[index], error, tolerance);
            }
        }
        std::printf(" max_abs=%.8g rel_l2=%.8g", maximum, std::sqrt(squared_error / std::max(1e-30, squared_reference)));
        require(failures == 0, "F32 attention changed source precision, softmax probabilities, or tensor indexing");
    };

    std::printf("%s F32_ATTN head=%d value_head=%d queries=%d keys=%d heads=%d/%d batches=%d/%d mask=%d:%s/%d/%d sink=%d padded=%d stride0=%d alias=%d pattern=%d",
        ggml_backend_name(backend), c.head, c.value_head, c.queries, c.keys, c.heads, c.kv_heads,
        c.batches, c.kv_batches, c.mask, ggml_type_name(c.mask_type), c.mask_heads, c.mask_batches,
        c.sink, c.padded, c.interleaved, c.shared_kv, int(c.data));
    std::printf(" sparse_capacity=%d k_type=%s v_type=%s v_heads=%lld v_batches=%lld", c.sparse_capacity, ggml_type_name(k.tensor->type),
        ggml_type_name(values.tensor->type), (long long) values.tensor->ne[2], (long long) values.tensor->ne[3]);
    timed_compute(backend, graph, 1);
    validate_compaction();
    validate(result_tensor, 0.5);
    std::printf(" mean_us=%.1f", timed_compute(backend, graph, 3));
    if (baseline_graph)
    {
        timed_compute(allocator, baseline_graph, 1);
        validate(baseline, 1.0);
        std::printf(" upstream_exact_contract_us=%.1f", timed_compute(allocator, baseline_graph, 3));
    }
    else if (baseline) std::printf(" upstream_exact_contract=unsupported");
    for (float & value : q.values) value *= 0.75f;
    for (float & value : v.values) value *= -0.5f;
    for (float & value : sink.values) value += 0.25f;
    if (c.mask) mask.values[0] = 0.0f;
    if (c.sparse_capacity > 0)
        for (int batch = 0; batch < c.mask_batches; ++batch)
        for (int head = 0; head < c.mask_heads; ++head)
        for (int query = 0; query < c.queries; ++query)
        for (int key = 0; key < c.keys; ++key)
            mask.values[mask.logical(key, query, head, batch)] = query != 1 &&
                (key == c.keys - 1 || (key + query * 17 + head * 3 + batch) % 37 == 0)
                ? -0.125f * ((key + query) % 5) : negative_infinity;
    q.upload(); v.upload(); mask.upload(); sink.upload();
    timed_compute(backend, graph, 1);
    validate_compaction();
    validate(result_tensor, 0.5);
    std::printf("\n");
    q.check_unchanged(); k.check_unchanged(); v.check_unchanged(); mask.check_unchanged(); sink.check_unchanged();
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

// Test exceptional-mask dispatch independently of attention's nonfinite-input
// arithmetic. NaN and +infinity must force a full-row scan, never pruning.
static void check_compaction_guards(ggml_backend_t allocator, ggml_backend_t backend, ggml_type type)
{
    constexpr int keys = 257, queries = 7, capacity = 16;
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Cannot initialize compaction guard test");
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 8, queries, 4, 2);
    auto * kv = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 8, keys, 1, 1);
    input_tensor mask(ctx, type, {keys, queries + 3, 2, 2}, true, true);
    auto * output = tsg_attention_f32_sparse(ctx, q, kv, kv, mask.tensor, nullptr, 1.0f, capacity);
    auto * compact = output->src[0]->src[4];
    require(compact && compact->type == GGML_TYPE_I32, "Missing compaction guard output");
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, compact);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, allocator);
    require(buffer != nullptr, "Cannot allocate compaction guard test");
    for (int repeat = 0; repeat < 3; ++repeat) {
        std::fill(mask.values.begin(), mask.values.end(), -INFINITY);
        for (int b = 0; b < 2; ++b) for (int h = 0; h < 2; ++h) for (int query = 0; query < queries; ++query) {
            const int pattern = (query + repeat + h + b) % queries;
            const int visible = pattern == 0 ? 0 : pattern == 1 ? capacity : pattern == 2 ? capacity + 1 :
                pattern == 6 ? keys : 1;
            for (int entry = 0; entry < visible; ++entry) {
                const int key = keys - visible + entry;
                mask.values[mask.logical(key, query, h, b)] = pattern == 3 ? std::numeric_limits<float>::quiet_NaN() :
                    pattern == 4 ? INFINITY : -0.125f * (key % 5);
            }
        }
        mask.upload();
        timed_compute(backend, graph, 1);
        std::vector<int32_t> actual(ggml_nelements(compact));
        ggml_backend_tensor_get(compact, actual.data(), 0, actual.size() * sizeof(int32_t));
        for (int b = 0; b < 2; ++b) for (int h = 0; h < 2; ++h) for (int query = 0; query < queries; ++query) {
            const auto * indices = actual.data() + (query + queries * (h + 2 * b)) * (capacity + 1);
            int count = 0;
            bool exceptional = false;
            for (int key = 0; key < keys; ++key) {
                const float value = mask.at(key, query, h, b);
                if (value == -INFINITY) continue;
                exceptional |= !std::isfinite(value);
                if (count < capacity) require(indices[count + 1] == key, "Guard compaction reordered or lost a key");
                ++count;
            }
            require(indices[0] == (exceptional ? -1 : count), "Compaction failed overflow, exceptional-mask or reuse guard");
        }
        mask.check_unchanged();
    }
    std::printf("%s COMPACTION_GUARDS mask=%s exact_capacity=%d overflow_capacity=%d exceptional=nan,+inf repeats=3 passed\n",
        ggml_backend_name(backend), ggml_type_name(type), capacity, capacity + 1);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

// A query of a decode-class launch must be bit-identical to the same query
// computed alone: a V4.1 speculative verify (block_size + 1 queries) commits
// quantized cache rows that single-token decode would otherwise have written.
// 64 heads make the widest launch cross the 512-row mark and 300 keys span
// several key partitions, so neither may change the per-query arithmetic.
static void check_query_invariance(ggml_backend_t allocator, ggml_backend_t backend)
{
    const int head = 64, keys = 300, heads = 64;
    const int widest = int(TSG_PRECISION_DECODE_COLUMNS);
    std::vector<float> reference;
    for (int queries = widest; queries >= 1; --queries)
    {
        auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        require(ctx != nullptr, "Cannot create query invariance context");
        input_tensor q(ctx, GGML_TYPE_F32, {head, queries, heads, 1});
        input_tensor k(ctx, GGML_TYPE_F16, {head, keys, 1, 1});
        input_tensor v(ctx, GGML_TYPE_F16, {head, keys, 1, 1});
        input_tensor mask(ctx, GGML_TYPE_F16, {keys, queries, 1, 1});
        input_tensor sink(ctx, GGML_TYPE_F32, {heads, 1, 1, 1});
        auto * output = tsg_attention_f32(ctx, q.tensor, k.tensor, v.tensor, mask.tensor, sink.tensor, 1.0f / 8.0f);
        require(output != nullptr, "Cannot create query invariance graph");
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, allocator);
        require(buffer != nullptr, "Query invariance allocation failed");
        std::mt19937 shared_random(50311);
        std::uniform_real_distribution<float> uniform(-0.5f, 0.5f);
        for (float & value : k.values) value = uniform(shared_random);
        for (float & value : v.values) value = uniform(shared_random);
        for (int h = 0; h < heads; ++h) sink.values[h] = float(h % 5) - 2.0f;
        const float negative_infinity = -std::numeric_limits<float>::infinity();
        // Query j carries the same data and sees the same keys in every launch.
        for (int query = 0; query < queries; ++query)
        {
            std::mt19937 query_random(2000 + query);
            for (int h = 0; h < heads; ++h) for (int x = 0; x < head; ++x)
                q.values[q.logical(x, query, h, 0)] = uniform(query_random);
            for (int key = 0; key < keys; ++key)
            {
                const bool visible = key < 130 + 20 * query && (key + query) % 9 != 0;
                mask.values[mask.logical(key, query, 0, 0)] = visible ? -0.125f * ((key + query) % 5) : negative_infinity;
            }
        }
        q.upload(); k.upload(); v.upload(); mask.upload(); sink.upload();
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Query invariance compute failed");
        std::vector<float> result(ggml_nelements(output));
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        for (float value : result) require(std::isfinite(value), "Query invariance produced a non-finite value");
        // Output is [Dv, H, N, B]: the first `queries` query slabs are comparable.
        const size_t per_query = size_t(head) * heads;
        if (queries == widest) reference = result;
        else
        {
            size_t differing = 0;
            for (size_t i = 0; i < per_query * size_t(queries); ++i)
                differing += std::memcmp(&result[i], &reference[i], sizeof(float)) != 0;
            if (differing)
                std::fprintf(stderr, "%s queries=%d differs from queries=%d in %zu of %zu outputs\n",
                    ggml_backend_name(backend), queries, widest, differing, per_query * size_t(queries));
            require(differing == 0, "A decode-class query must not depend on the other queries in its launch");
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    std::printf("%s QUERY_INVARIANCE head=%d keys=%d heads=%d widths=1..%d bit_identical\n",
        ggml_backend_name(backend), head, keys, heads, widest);
}

static void run(ggml_backend_t allocator, ggml_backend_t backend)
{
    check_query_invariance(allocator, backend);
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16})
        check_compaction_guards(allocator, backend, type);
    for (int keys : {15, 16, 17, 255, 256, 257})
    for (int capacity : {1, 16, 257}) {
        attention_case c;
        c.head = 64; c.value_head = 33; c.queries = 5; c.keys = keys;
        c.heads = 8; c.kv_heads = 2; c.batches = 4; c.kv_batches = 2;
        c.value_heads = 4; c.value_batches = 1; c.mask_heads = 2; c.mask_batches = 2;
        c.sparse_capacity = capacity; c.padded = true; c.interleaved = true;
        c.kv_type = capacity == 1 ? GGML_TYPE_F32 : capacity == 16 ? GGML_TYPE_BF16 : GGML_TYPE_F16;
        c.value_type = GGML_TYPE_F32; c.mask_type = GGML_TYPE_F32; c.sink = capacity != 16;
        check(allocator, backend, c);
    }
    for (attention_data data : {attention_data::source_bits, attention_data::probability_bits, attention_data::large_values}) {
        attention_case c;
        c.keys = 2; c.sparse_capacity = 2; c.sink = false; c.kv_type = GGML_TYPE_F32; c.data = data;
        check(allocator, backend, c);
    }
    {
        attention_case c;
        c.head = 8; c.value_head = 7; c.queries = 3; c.keys = 8193; c.heads = 2; c.sparse_capacity = 256;
        check(allocator, backend, c);
    }
    for (int queries : {1, 5, 31}) for (int keys : {1, 7, 129, 513})
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_F32})
    {
        attention_case c;
        c.queries = queries; c.keys = keys; c.kv_type = type;
        check(allocator, backend, c);
    }
    for (attention_data data : {attention_data::source_bits, attention_data::probability_bits, attention_data::large_values})
    {
        attention_case c;
        c.keys = 2; c.mask = false; c.sink = false; c.kv_type = GGML_TYPE_F32; c.data = data;
        check(allocator, backend, c);
        if (data != attention_data::large_values) { c.kv_type = GGML_TYPE_F16; check(allocator, backend, c); }
    }
    for (int queries : {4, 5}) for (bool interleaved : {false, true}) for (bool sink : {false, true})
    {
        attention_case c;
        c.head = 64; c.value_head = 33; c.queries = queries; c.keys = 129; c.padded = true; c.interleaved = interleaved;
        c.heads = 8; c.kv_heads = 2; c.batches = 4; c.kv_batches = 2;
        c.value_heads = 4; c.value_batches = 1; c.value_type = GGML_TYPE_F32;
        c.mask_heads = 2; c.mask_batches = 2; c.mask_type = GGML_TYPE_F32; c.sink = sink;
        check(allocator, backend, c);
    }
    {
        attention_case c;
        c.keys = 513; c.queries = 5; c.heads = 64; c.shared_kv = true;
        check(allocator, backend, c);
        c.mask = false; c.sink = false; c.heads = 4;
        check(allocator, backend, c);
    }
    for (int queries : {1, 5, 31})
    {
        attention_case c;
        c.queries = queries; c.keys = 128; c.heads = 8; c.data = attention_data::exact_uniform;
        check(allocator, backend, c);
    }
    for (bool sink : {false, true})
    {
        attention_case c;
        c.queries = 513; c.keys = 129; c.head = 64; c.value_head = 33;
        c.heads = 4; c.padded = true; c.mask_type = GGML_TYPE_F32; c.sink = sink;
        check(allocator, backend, c);
    }
}

#ifdef TSG_GGML_USE_CUDA
// Run separately from CTest: realistic prefill sizes use a decomposed F32 GPU
// reference, and never allocate both large graphs on the device at once.
static void benchmark(ggml_backend_t allocator, ggml_backend_t backend,
                      int queries, int keys, int heads, int repeats, int sparse_capacity = 0)
{
    constexpr int width = 512;
    require(queries > 0 && keys > 0 && heads > 0 && repeats > 0, "Invalid attention benchmark dimensions");
    std::vector<float> reference;
    double baseline_us = 0;
    for (int mode = 0; mode < (sparse_capacity > 0 ? 3 : 2); ++mode)
    {
        const bool owned = mode != 0;
        auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        require(ctx != nullptr, "Cannot create attention benchmark context");
        auto * input_ctx = ggml_init({1024 * 1024, nullptr, true});
        require(input_ctx != nullptr, "Cannot create attention input context");
        auto * q = ggml_new_tensor_3d(input_ctx, GGML_TYPE_F32, width, queries, heads);
        auto * kv = ggml_new_tensor_3d(input_ctx, GGML_TYPE_F16, width, keys, 1);
        auto * mask = ggml_new_tensor_2d(input_ctx, GGML_TYPE_F16, keys, queries);
        auto * sinks = ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, heads);
        const float scale = 1.0f / std::sqrt(float(width));
        ggml_tensor * output;
        if (owned) output = tsg_attention_f32_sparse(ctx, q, kv, kv, mask, sinks, scale, mode == 2 ? sparse_capacity : 0);
        else
        {
            auto * scores = tsg_matmul_f32(ctx, kv, q);
            auto * probabilities = ggml_soft_max_ext(ctx, scores, mask, scale, 0.0f);
            ggml_soft_max_add_sinks(probabilities, sinks);
            auto * weighted = tsg_matmul_f32(ctx, ggml_transpose(ctx, kv), probabilities);
            output = ggml_cont(ctx, ggml_permute(ctx, weighted, 0, 2, 1, 3));
        }
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        ggml_set_output(output);
        auto * buffer = ggml_backend_alloc_ctx_tensors(input_ctx, allocator);
        require(buffer != nullptr, "Cannot allocate attention benchmark inputs");
        auto * graph_allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(allocator));
        require(graph_allocator && ggml_gallocr_alloc_graph(graph_allocator, graph), "Cannot allocate attention benchmark graph");
        const size_t buffer_bytes = ggml_backend_buffer_get_size(buffer) + ggml_gallocr_get_buffer_size(graph_allocator, 0);
        std::mt19937 random(78471);
        std::uniform_real_distribution<float> uniform(-0.5f, 0.5f);
        {
            std::vector<float> data(ggml_nelements(q));
            for (float & value : data) value = uniform(random);
            ggml_backend_tensor_set(q, data.data(), 0, data.size() * sizeof(float));
        }
        {
            std::vector<ggml_fp16_t> data(ggml_nelements(kv));
            for (auto & value : data) value = ggml_fp32_to_fp16(uniform(random));
            ggml_backend_tensor_set(kv, data.data(), 0, data.size() * sizeof(ggml_fp16_t));
        }
        {
            std::vector<ggml_fp16_t> data(ggml_nelements(mask));
            for (int query = 0; query < queries; ++query) for (int key = 0; key < keys; ++key)
                data[size_t(query) * keys + key] = ggml_fp32_to_fp16(
                    (sparse_capacity > 0 ? (int64_t(key) + int64_t(query) * 17) % keys >= sparse_capacity
                        : key > keys - 32 + query % 32) ? -INFINITY : -0.125f * ((key + query) % 5));
            ggml_backend_tensor_set(mask, data.data(), 0, data.size() * sizeof(ggml_fp16_t));
        }
        {
            std::vector<float> data(heads);
            for (int head = 0; head < heads; ++head) data[head] = (head % 5) - 2.0f;
            ggml_backend_tensor_set(sinks, data.data(), 0, data.size() * sizeof(float));
        }
        timed_compute(backend, graph, 3);
        std::vector<double> samples(repeats);
        double total = 0;
        for (double & sample : samples) { sample = timed_compute(backend, graph, 1); total += sample; }
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const double elapsed = (sorted[(repeats - 1) / 2] + sorted[repeats / 2]) / 2;
        std::vector<float> actual(ggml_nelements(output));
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
        std::printf("ATTENTION_BENCH implementation=%s queries=%d keys=%d heads=%d head=512 visible_capacity=%d median_us=%.1f mean_us=%.1f min_us=%.1f max_us=%.1f buffer_bytes=%zu samples_us=[",
            mode == 2 ? "owned-sparse-f32" : owned ? "owned-f32" : "decomposed-f32", queries, keys, heads,
            sparse_capacity, elapsed, total / repeats, sorted.front(), sorted.back(), buffer_bytes);
        for (int i = 0; i < repeats; ++i) std::printf("%s%.1f", i ? "," : "", samples[i]);
        std::printf("]");
        if (owned)
        {
            require(actual.size() == reference.size(), "Attention benchmark output shape mismatch");
            double maximum = 0, squared_error = 0, squared_reference = 0;
            size_t failures = 0;
            for (size_t i = 0; i < actual.size(); ++i)
            {
                require(std::isfinite(actual[i]) && std::isfinite(reference[i]), "Nonfinite attention benchmark output");
                const double error = std::abs(double(actual[i]) - reference[i]);
                maximum = std::max(maximum, error);
                squared_error += error * error;
                squared_reference += double(reference[i]) * reference[i];
                if (error > 6e-6 + 6e-6 * std::abs(reference[i])) ++failures;
            }
            std::printf(" max_abs=%.8g rel_l2=%.8g speedup=%.4f failures=%zu", maximum,
                std::sqrt(squared_error / std::max(1e-30, squared_reference)), baseline_us / elapsed, failures);
            require(failures == 0, "Attention benchmark differs from decomposed F32 reference");
        }
        else { reference = std::move(actual); baseline_us = elapsed; }
        std::printf("\n");
        ggml_gallocr_free(graph_allocator);
        ggml_backend_buffer_free(buffer);
        ggml_free(input_ctx);
        ggml_free(ctx);
    }
}
#endif

#ifdef TSG_GGML_USE_CUDA
#include "attention_scheduler_stress.h"
#endif

int main(int argc, char ** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#ifdef TSG_GGML_USE_CUDA
    const int devices = ggml_backend_cuda_get_device_count();
    if (devices == 0) return 77;
    const bool sparse_scheduler = argc == 7 && (std::strcmp(argv[1], "--scheduler-stress-sparse") == 0 ||
        std::strcmp(argv[1], "--scheduler-shape-stress-sparse") == 0);
    const bool changing_shapes = argc == 6 && (std::strcmp(argv[1], "--scheduler-shape-stress") == 0 ||
        std::strcmp(argv[1], "--scheduler-shape-stress-legacy") == 0);
    const bool scheduler_legacy = argc == 6 && (std::strcmp(argv[1], "--scheduler-stress-legacy") == 0 ||
        std::strcmp(argv[1], "--scheduler-shape-stress-legacy") == 0);
    const bool scheduler_complete = argc == 6 && (std::strcmp(argv[1], "--scheduler-stress") == 0 ||
        std::strcmp(argv[1], "--scheduler-shape-stress") == 0);
    if (scheduler_legacy || scheduler_complete || sparse_scheduler)
        return scheduler_stress(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]),
            scheduler_complete || sparse_scheduler, changing_shapes ||
                (sparse_scheduler && std::strcmp(argv[1], "--scheduler-shape-stress-sparse") == 0),
            sparse_scheduler ? std::atoi(argv[6]) : 0);
    const bool sparse_bench = argc == 7 && std::strcmp(argv[1], "--benchmark-sparse") == 0;
    const bool bench = sparse_bench || (argc == 6 && std::strcmp(argv[1], "--benchmark") == 0);
    require(argc == 1 || bench, "Usage: attention_precision_test [--benchmark|--scheduler-stress queries keys heads repeats] or --benchmark-sparse queries keys heads repeats capacity");
    for (int device = 0; device < devices; ++device)
    {
        auto * cuda = ggml_backend_cuda_init(device);
        require(cuda != nullptr, "Cannot initialize visible CUDA device");
        auto * backend = tsg_dsv4_fused_backend_init(cuda);
        require(backend != nullptr, "Cannot initialize TensorSharp attention backend");
        if (bench) benchmark(cuda, backend, std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), std::atoi(argv[5]),
            sparse_bench ? std::atoi(argv[6]) : 0);
        else run(cuda, backend);
        ggml_backend_free(backend);
        ggml_backend_free(cuda);
        if (bench) break;
    }
#else
    require(argc == 1, "Large attention benchmark mode requires CUDA");
    auto * backend = ggml_backend_cpu_init();
    require(backend != nullptr, "Cannot initialize CPU attention backend");
    ggml_backend_cpu_set_n_threads(backend, 4);
    run(backend, backend);
    ggml_backend_free(backend);
#endif
    return 0;
}
