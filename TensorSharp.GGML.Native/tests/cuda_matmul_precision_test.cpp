// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_matmul_precision.h"
#include "ggml_ops_precision_policy.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#else
#include "ggml-cpu.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "precision_test_utils.h"

// Quantize the source once, then retain complete source rows for each shard.
// Tile-specific data catches a wrong global row offset; expert-specific data
// catches a wrong indexed weight stride. Repeating sixteen distinct rows per
// tile keeps the IQ quantizer setup small without making different tiles equal.
struct quantized_fixture
{
    ggml_type type;
    int inner, rows, experts;
    size_t row_bytes;
    std::vector<unsigned char> gate, up;

    quantized_fixture(ggml_type kind, int width, int height, int count)
        : type(kind), inner(width), rows(height), experts(count), row_bytes(ggml_row_size(kind, width))
    {
        ggml_quantize_init(type);
        std::vector<float> importance(inner, 1.0f);
        auto generate = [&](std::vector<unsigned char> & packed, uint32_t seed)
        {
            packed.resize(row_bytes * rows * experts);
            std::mt19937 random(seed);
            std::normal_distribution<float> normal(0.0f, 1.0f / std::sqrt(float(inner)));
            std::vector<float> values(size_t(inner) * 16);
            std::vector<unsigned char> tile(row_bytes * 16);
            for (int expert = 0; expert < experts; ++expert)
            for (int first = 0; first < rows; first += 128)
            {
                for (float & value : values) value = normal(random);
                require(ggml_quantize_chunk(type, values.data(), tile.data(), 0, 16, inner,
                    ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr) == tile.size(),
                    "Cannot quantize original FFN fixture");
                for (int row = first; row < std::min(first + 128, rows); ++row)
                    std::memcpy(packed.data() + (size_t(expert) * rows + row) * row_bytes,
                        tile.data() + ((row - first) % 16) * row_bytes, row_bytes);
            }
        };
        generate(gate, 73918); generate(up, 94713);
    }

    std::vector<unsigned char> slice(const std::vector<unsigned char> & source, int first, int count) const
    {
        std::vector<unsigned char> result(row_bytes * count * experts);
        for (int expert = 0; expert < experts; ++expert)
            std::memcpy(result.data() + size_t(expert) * count * row_bytes,
                source.data() + (size_t(expert) * rows + first) * row_bytes, size_t(count) * row_bytes);
        return result;
    }
};

static void check_quantized_strip(ggml_backend_t allocator, ggml_backend_t backend,
    const quantized_fixture & data, int tokens, bool down, int selected_experts = 4)
{
    const int used = std::min(selected_experts, data.experts), slots = down ? used : 1;
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Cannot create quantized FFN context");
    auto * gate = ggml_new_tensor_3d(ctx, data.type, data.inner, data.rows, data.experts);
    auto * up = ggml_new_tensor_3d(ctx, data.type, data.inner, data.rows, data.experts);
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, data.inner, slots, tokens);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
    auto * plain_gate = ggml_mul_mat_id(ctx, gate, input, ids);
    auto * plain_up = ggml_mul_mat_id(ctx, up, input, ids);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, plain_gate); ggml_build_forward_expand(graph, plain_up);
    struct shard
    {
        int first, count;
        ggml_tensor * gate, * up, * gate_output, * up_output, * pair = nullptr;
        std::vector<unsigned char> gate_bytes, up_bytes;
        tsg_dsv4_fused_desc gate_desc, up_desc, pair_desc;
        bool owned = false;
    };
    std::array<shard, 2> shards;
    for (int rank = 0; rank < 2; ++rank)
    {
        auto & s = shards[rank];
        // 640-wide gate/up slices retain overlapping MMQ tiles around the
        // logical 320-row boundary, exactly as Qwen4Exp's loader does.
        s.first = down ? rank * data.rows / 2 : rank * 256;
        s.count = down ? data.rows / 2 : 384;
        s.gate = ggml_new_tensor_3d(ctx, data.type, data.inner, s.count, data.experts);
        s.up = ggml_new_tensor_3d(ctx, data.type, data.inner, s.count, data.experts);
        s.gate_bytes = data.slice(data.gate, s.first, s.count);
        s.up_bytes = data.slice(data.up, s.first, s.count);
#ifdef TSG_GGML_USE_CUDA
        s.owned = tsg_matmul_id_quant_strip_supported(allocator, s.gate, tokens, data.rows, s.first);
        require(tokens < 129 || s.owned, "Prefill fixture did not engage the owned CUDA strip kernel");
#else
        s.owned = true;
#endif
        if (s.owned)
        {
            s.gate_desc.kind = s.up_desc.kind = TSG_MATMUL_ID_QUANT_STRIP;
            s.gate_desc.i0 = s.up_desc.i0 = data.rows;
            s.gate_desc.i1 = s.up_desc.i1 = s.first;
            s.gate_output = tsg_matmul_id_quant_strip(ctx, s.gate, input, ids, &s.gate_desc);
            s.up_output = tsg_matmul_id_quant_strip(ctx, s.up, input, ids, &s.up_desc);
            if (!down)
            {
                s.pair_desc.kind = TSG_MATMUL_ID_QUANT_PAIR;
                s.pair_desc.i0 = data.rows; s.pair_desc.i1 = s.first;
                s.pair = tsg_matmul_id_quant_pair(ctx, s.gate, s.up, input, ids, &s.pair_desc);
            }
        }
        else
        {
            s.gate_output = ggml_mul_mat_id(ctx, s.gate, input, ids);
            s.up_output = ggml_mul_mat_id(ctx, s.up, input, ids);
        }
        ggml_build_forward_expand(graph, s.gate_output); ggml_build_forward_expand(graph, s.up_output);
        if (s.pair) ggml_build_forward_expand(graph, s.pair);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, allocator);
    require(buffer != nullptr, "Cannot allocate quantized FFN graph");
    ggml_backend_tensor_set(gate, data.gate.data(), 0, data.gate.size());
    ggml_backend_tensor_set(up, data.up.data(), 0, data.up.size());
    for (const auto & s : shards)
    {
        ggml_backend_tensor_set(s.gate, s.gate_bytes.data(), 0, s.gate_bytes.size());
        ggml_backend_tensor_set(s.up, s.up_bytes.data(), 0, s.up_bytes.size());
    }
    std::mt19937 random(38910);
    std::normal_distribution<float> normal(0.0f, 0.4f);
    std::vector<float> activations(ggml_nelements(input));
    std::vector<int32_t> selected(used * tokens);
    auto get = [](ggml_tensor * tensor)
    {
        std::vector<float> result(ggml_nelements(tensor));
        ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
        for (float value : result) require(std::isfinite(value), "Nonfinite quantized FFN projection");
        return result;
    };
    for (int reuse = 0; reuse < 2; ++reuse)
    {
        for (float & value : activations) value = normal(random);
        for (int token = 0; token < tokens; ++token)
        for (int slot = 0; slot < used; ++slot)
            selected[token * used + slot] = (3 * token + 2 * slot + reuse) % data.experts;
        ggml_backend_tensor_set(input, activations.data(), 0, activations.size() * sizeof(float));
        ggml_backend_tensor_set(ids, selected.data(), 0, selected.size() * sizeof(int32_t));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Quantized FFN graph compute failed");
        const auto expected_gate = get(plain_gate), expected_up = get(plain_up);
        size_t compared = 0;
        for (const auto & s : shards)
        {
            const auto actual_gate = get(s.gate_output), actual_up = get(s.up_output);
            for (int column = 0; column < tokens * used; ++column)
            {
                const size_t full = size_t(column) * data.rows + s.first, part = size_t(column) * s.count;
                require(std::memcmp(actual_gate.data() + part, expected_gate.data() + full, s.count * sizeof(float)) == 0,
                    "Quantized gate/down row slice differs from untouched upstream projection");
                require(std::memcmp(actual_up.data() + part, expected_up.data() + full, s.count * sizeof(float)) == 0,
                    "Quantized up/down row slice differs from untouched upstream projection");
                compared += 2 * s.count;
            }
            if (s.pair)
            {
                const auto paired = get(s.pair);
                require(std::memcmp(paired.data(), actual_gate.data(), actual_gate.size() * sizeof(float)) == 0,
                    "Paired quantized gate differs from independent gate strip");
                require(std::memcmp(paired.data() + actual_gate.size(), actual_up.data(), actual_up.size() * sizeof(float)) == 0,
                    "Paired quantized up differs from independent up strip");
            }
        }
        // Independent CPU dequantization + FP64 products expose arithmetic
        // error without reproducing the CUDA row/expert address calculation.
        // Backend activation quantization is intentionally reported separately
        // from the exact upstream-versus-shard acceptance check above.
        std::vector<float> decoded(data.inner);
        double max_abs = 0, squared_error = 0, squared_reference = 0;
        size_t samples = 0;
        for (int token : {0, tokens - 1}) for (int slot = 0; slot < used; ++slot)
        for (int row : {0, 255, data.rows / 2, data.rows - 1})
        {
            const int expert = selected[token * used + slot];
            const float * x = activations.data() + (size_t(token) * slots + (down ? slot : 0)) * data.inner;
            for (int matrix = 0; matrix < 2; ++matrix)
            {
                const auto & packed = matrix ? data.up : data.gate;
                const auto & expected = matrix ? expected_up : expected_gate;
                ggml_get_type_traits(data.type)->to_float(packed.data() + (size_t(expert) * data.rows + row) * data.row_bytes,
                    decoded.data(), data.inner);
                double reference = 0;
                for (int k = 0; k < data.inner; ++k) reference += double(decoded[k]) * x[k];
                const double actual = expected[(size_t(token) * used + slot) * data.rows + row];
                require(std::isfinite(reference), "Nonfinite independent FP64 quantized projection");
                const double error = actual - reference;
                max_abs = std::max(max_abs, std::abs(error));
                squared_error += error * error; squared_reference += reference * reference; ++samples;
            }
        }
        std::printf("%s QUANT_FFN role=%s weights=%s inner=%d rows=%d tokens=%d used=%d reuse=%d owned=%d bit_identical_values=%zu f64_samples=%zu f64_max_abs=%.8g f64_rel_l2=%.8g\n",
            ggml_backend_name(backend), down ? "down" : "gate/up/pair", ggml_type_name(data.type),
            data.inner, data.rows, tokens, used, reuse, shards[0].owned, compared, samples, max_abs,
            std::sqrt(squared_error / std::max(1e-30, squared_reference)));
    }
    auto unchanged = [](ggml_tensor * tensor, const std::vector<unsigned char> & expected)
    {
        std::vector<unsigned char> actual(expected.size());
        ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
        require(actual == expected, "Quantized FFN execution changed original weight bytes");
    };
    unchanged(gate, data.gate); unchanged(up, data.up);
    for (const auto & s : shards) { unchanged(s.gate, s.gate_bytes); unchanged(s.up, s.up_bytes); }
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

#if defined(TSG_GGML_USE_CUDA) && defined(TSG_GGML_TEST_HOOKS)
// Qwen3.8-Flash-Next's image-prefill shape: 512 experts, 10 selected,
// 2560 embedding channels and a 640-channel FFN. An unconditional partial
// buffer formerly reserved gigabytes even though complete-tile launches do
// not use it. The ceiling checks the actual CUDA allocation request, while
// every gate/up/down/pair output remains bit-identical to upstream MMQ.
static void run_wide_quantized(ggml_backend_t allocator, ggml_backend_t backend)
{
#if defined(_WIN32)
    _putenv_s("TS_TP_TEST_MAX_SCRATCH", "67108864");
#else
    setenv("TS_TP_TEST_MAX_SCRATCH", "67108864", 1);
#endif
    for (ggml_type type : {GGML_TYPE_IQ4_XS, GGML_TYPE_Q6_K})
    {
        {
            quantized_fixture gate_up(type, 2560, 640, 512);
            for (int tokens : {129, 2048})
                check_quantized_strip(allocator, backend, gate_up, tokens, false, 10);
        }
        {
            quantized_fixture down(type, 768, 2560, 512);
            for (int tokens : {129, 2048})
                check_quantized_strip(allocator, backend, down, tokens, true, 10);
        }
    }
#if defined(_WIN32)
    _putenv_s("TS_TP_TEST_MAX_SCRATCH", "");
#else
    unsetenv("TS_TP_TEST_MAX_SCRATCH");
#endif
    std::printf("%s QUANT_FFN_WIDE experts=512 used=10 tokens=129,2048 scratch_ceiling=67108864 passed\n",
        ggml_backend_name(backend));
}
#endif

static void run_quantized(ggml_backend_t allocator, ggml_backend_t backend, const std::string & only_type, int inner)
{
    const auto lowercase = [](std::string value) {
        for (char & c : value) if (c >= 'A' && c <= 'Z') c = char(c + ('a' - 'A'));
        return value;
    };
    bool selected = false;
    for (ggml_type type : {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS})
    {
        if (!only_type.empty() && lowercase(only_type) != lowercase(ggml_type_name(type))) continue;
        selected = true;
        quantized_fixture gate_up(type, inner, 640, 8);
        quantized_fixture down(type, type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL ? 640 : 768, 2560, 8);
        for (int tokens : {1, 4, 8, 9, 17, 31, 129})
        {
            check_quantized_strip(allocator, backend, gate_up, tokens, false);
            check_quantized_strip(allocator, backend, down, tokens, true);
        }
        if (type == GGML_TYPE_Q5_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q8_0)
        {
            // Shared FFNs are one-expert graphs. Incomplete J tiles at9/129
            // columns exercise the MMQ input padding before partial scratch.
            quantized_fixture shared(type, inner, 640, 1);
            for (int tokens : {9, 17, 129}) check_quantized_strip(allocator, backend, shared, tokens, false);
        }
    }
    require(selected, "Unknown --type (use the upstream ggml lowercase type name)");
}

enum class pattern { random, activation_residual, large_activation, weight_residual };

struct test_case
{
    int tokens = 5, inner = 256, rows = 64;
    bool indexed = false, precise = true, padded = false, interleaved = false, convert = false, pipeline = false;
    int experts = 4, used = 4, slots = 1;
    int a2 = 1, a3 = 1, b2 = 1, b3 = 1;
    ggml_type weights = GGML_TYPE_F32;
    pattern data = pattern::random;
};

static void check(ggml_backend_t allocator, ggml_backend_t backend, const test_case & c)
{
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Cannot create precision test context");
    input_tensor a(ctx, c.weights, {c.inner, c.rows, c.indexed ? c.experts : c.a2, c.indexed ? 1 : c.a3},
                   c.padded, c.interleaved);
    input_tensor b(ctx, GGML_TYPE_F32,
                   {c.inner, c.indexed ? c.slots : c.tokens, c.indexed ? c.tokens : c.b2, c.indexed ? 1 : c.b3},
                   c.padded, c.interleaved);
    input_tensor ids(ctx, GGML_TYPE_I32, {c.indexed ? c.used : 1, c.indexed ? c.tokens : 1, 1, 1}, c.padded,
                     c.indexed && c.interleaved);
    auto * activation = c.pipeline ? ggml_scale(ctx, b.tensor, 2.0f) : b.tensor;
    ggml_tensor * output;
    if (!c.precise || c.convert)
    {
        output = c.indexed ? ggml_mul_mat_id(ctx, a.tensor, activation, ids.tensor) : ggml_mul_mat(ctx, a.tensor, activation);
        if (c.precise) tsg_matmul_require_f32(ctx, output);
    }
    else output = c.indexed ? tsg_matmul_id_f32(ctx, a.tensor, activation, ids.tensor) : tsg_matmul_f32(ctx, a.tensor, activation);
    require(output != nullptr, "Cannot construct precision matmul");
    if (c.precise) require(output->op == GGML_OP_CUSTOM, "Explicit F32 must use the TensorSharp implementation");
    auto * graph_output = ggml_scale(ctx, output, 0.5f);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, graph_output);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, allocator);
    require(buffer != nullptr, "Precision test allocation failed");
    std::mt19937 random(41091);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (size_t i = 0; i < a.values.size(); ++i)
    {
        const int k = i % c.inner, row = (i / c.inner) % c.rows;
        a.values[i] = c.data == pattern::random ? uniform(random)
            : (k % 2 == 0 ? 1.0f : -1.0f) * (1 + row % 4);
        if (c.data == pattern::weight_residual && k % 2 == 0) a.values[i] += 0x1p-16f;
    }
    for (size_t i = 0; i < b.values.size(); ++i)
    {
        const int k = i % c.inner;
        const float residue = (1 + (i / c.inner) % 5) * 0x1p-16f;
        // One large pair isolates F16 overflow/TF32 rounding from the legal
        // F32 error of summing hundreds of large positive partial products.
        b.values[i] = c.data == pattern::random ? uniform(random)
            : c.data == pattern::large_activation ? (k < 2 ? 70000.0f + (k == 0 ? 0.125f : 0.0f) : 0.0f)
            : c.data == pattern::activation_residual ? 1.0f + (k % 2 == 0 ? residue : 0.0f) : 1.0f;
    }
    for (int t = 0; t < ids.tensor->ne[1]; ++t)
        for (int e = 0; e < ids.tensor->ne[0]; ++e)
            // Upstream's sorted expert path requires unique IDs, as top-k
            // produces. The owned-only padded cases also stress duplicate IDs.
            ids.values[ids.logical(e, t, 0, 0)] = (3 * t + (c.padded ? e / 2 : e)) % c.experts;
    a.upload(); b.upload(); ids.upload();

    auto validate = [&]()
    {
        std::vector<float> result(ggml_nelements(graph_output));
        ggml_backend_tensor_get(graph_output, result.data(), 0, result.size() * sizeof(float));
        double squared_error = 0, squared_reference = 0, maximum = 0;
        size_t failures = 0;
        for (int64_t w = 0; w < output->ne[3]; ++w)
        for (int64_t z = 0; z < output->ne[2]; ++z)
        for (int64_t y = 0; y < output->ne[1]; ++y)
        for (int64_t row = 0; row < output->ne[0]; ++row)
        {
            const int64_t az = c.indexed ? static_cast<int64_t>(ids.at(y, z)) : z / (c.b2 / c.a2);
            const int64_t aw = c.indexed ? 0 : w / (c.b3 / c.a3);
            const int64_t by = c.indexed ? y % c.slots : y;
            double reference = 0;
            for (int64_t k = 0; k < c.inner; ++k) reference += double(a.at(k, row, az, aw)) * b.at(k, by, z, w);
            reference *= c.pipeline ? 1.0 : 0.5;
            const size_t index = row + output->ne[0] * (y + output->ne[1] * (z + output->ne[2] * w));
            require(std::isfinite(result[index]) && std::isfinite(reference), "Non-finite precision matmul result");
            const double error = std::abs(result[index] - reference);
            maximum = std::max(maximum, error);
            squared_error += error * error;
            squared_reference += reference * reference;
            const double tolerance = 3e-5 * std::max(1.0, std::sqrt(c.inner / 256.0)) + 3e-6 * std::abs(reference);
            if (c.precise && error > tolerance)
            {
                if (failures++ == 0) std::fprintf(stderr, "Mismatch index=%zu expected=%.12g actual=%.12g error=%.8g tolerance=%.8g\n",
                    index, reference, result[index], error, tolerance);
            }
        }
        const double relative = std::sqrt(squared_error / std::max(1e-30, squared_reference));
        std::printf(" max_abs=%.8g rel_l2=%.8g", maximum, relative);
        require(failures == 0, "Explicit F32 matmul lost precision or used the wrong input");
        if (!c.precise)
            require(relative < (c.weights == GGML_TYPE_BF16 ? 0.02 : 0.005),
                "Default matmul exceeds expected source-rounding error");
    };

    std::printf("%s %s nt=%d weights=%s inner=%d rows=%d slots=%d batches=%d,%d/%d,%d padded=%d stride0=%d pattern=%d convert=%d",
        ggml_backend_name(backend), c.indexed ? "MUL_MAT_ID" : "MUL_MAT", c.tokens, ggml_type_name(c.weights),
        c.inner, c.rows, c.slots, c.a2, c.a3, c.b2, c.b3, c.padded, c.interleaved, int(c.data), c.convert);
    std::printf(" precision=%s pipeline=%d", c.precise ? "F32" : "default", c.pipeline);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Precision matmul failed");
    validate();
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < 5; ++repeat)
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Repeated precision matmul failed");
    ggml_backend_synchronize(backend);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 5;
    std::printf(" mean_us=%.1f", us);
    // Reuse the graph with changed activations and expert IDs, as decode does.
    // Exact cancellation fixtures keep exact products; random perturbation here
    // would test reduction-order error instead of lost source precision.
    for (float & value : b.values)
        value = c.data == pattern::large_activation || c.data == pattern::weight_residual
            ? 2.0f * value : value + 0.125f * uniform(random);
    for (float & value : ids.values) value = float((int(value) + 1) % c.experts);
    b.upload(); ids.upload();
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Precision graph reuse failed");
    validate();
    std::printf("\n");
    a.check_unchanged(); b.check_unchanged(); ids.check_unchanged();
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

// A column of a decode-class launch must be bit-identical to the same column
// computed alone: V4.1 quantizes these outputs into its caches, and a
// speculative verify (block_size + 1 columns) stands in for single-token
// decode steps. The widest decode-class launch is the reference; every
// narrower prefix of the same columns must reproduce it exactly.
static void check_column_invariance(ggml_backend_t allocator, ggml_backend_t backend, ggml_type weights, bool indexed)
{
    const int inner = 4096, rows = 96, experts = 4, used = 2;
    const int widest = int(TSG_PRECISION_DECODE_COLUMNS);
    std::vector<float> reference;
    for (int tokens = widest; tokens >= 1; --tokens)
    {
        auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
        require(ctx != nullptr, "Cannot create column invariance context");
        input_tensor a(ctx, weights, {inner, rows, indexed ? experts : 1, 1});
        input_tensor b(ctx, GGML_TYPE_F32, indexed ? std::array<int64_t, 4>{inner, 1, tokens, 1}
                                                   : std::array<int64_t, 4>{inner, tokens, 1, 1});
        input_tensor ids(ctx, GGML_TYPE_I32, {indexed ? used : 1, indexed ? tokens : 1, 1, 1});
        auto * output = indexed ? tsg_matmul_id_f32(ctx, a.tensor, b.tensor, ids.tensor)
                                : tsg_matmul_f32(ctx, a.tensor, b.tensor);
        require(output != nullptr && output->op == GGML_OP_CUSTOM, "Column invariance needs the owned matmul");
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, allocator);
        require(buffer != nullptr, "Column invariance allocation failed");
        std::mt19937 weight_random(77031);
        std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
        for (float & value : a.values) value = uniform(weight_random);
        // Column j carries the same data in every launch, whatever its width.
        for (int column = 0; column < tokens; ++column)
        {
            std::mt19937 column_random(1000 + column);
            for (int k = 0; k < inner; ++k)
                b.values[indexed ? b.logical(k, 0, column, 0) : b.logical(k, column, 0, 0)] = uniform(column_random);
            for (int e = 0; e < used && indexed; ++e) ids.values[ids.logical(e, column, 0, 0)] = float((column + e) % experts);
        }
        a.upload(); b.upload(); ids.upload();
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Column invariance compute failed");
        std::vector<float> result(ggml_nelements(output));
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        for (float value : result) require(std::isfinite(value), "Column invariance produced a non-finite value");
        const size_t per_column = size_t(rows) * (indexed ? used : 1);
        if (tokens == widest) reference = result;
        else
        {
            size_t differing = 0;
            for (size_t i = 0; i < per_column * size_t(tokens); ++i)
                differing += std::memcmp(&result[i], &reference[i], sizeof(float)) != 0;
            if (differing)
                std::fprintf(stderr, "%s columns=%d differs from columns=%d in %zu of %zu outputs (weights=%s indexed=%d)\n",
                    ggml_backend_name(backend), tokens, widest, differing, per_column * size_t(tokens),
                    ggml_type_name(weights), indexed);
            require(differing == 0, "A decode-class column must not depend on the other columns in its launch");
        }
        a.check_unchanged(); b.check_unchanged(); ids.check_unchanged();
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    std::printf("%s COLUMN_INVARIANCE %s weights=%s inner=%d rows=%d widths=1..%d bit_identical\n", ggml_backend_name(backend),
        indexed ? "MUL_MAT_ID" : "MUL_MAT", ggml_type_name(weights), inner, rows, widest);
}

static void run(ggml_backend_t allocator, ggml_backend_t backend)
{
    for (bool indexed : {false, true}) for (ggml_type weights : {GGML_TYPE_F32, GGML_TYPE_F16})
        check_column_invariance(allocator, backend, weights, indexed);
    for (bool indexed : {false, true}) for (int tokens : {1, 4, 5, 16, 31})
    for (ggml_type weights : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16})
    for (bool precise : {false, true})
    {
        test_case c;
        c.indexed = indexed; c.tokens = tokens; c.weights = weights; c.precise = precise;
        check(allocator, backend, c);
    }
    for (ggml_type weights : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16})
    {
        test_case batch;
        batch.weights = weights; batch.b2 = 3; batch.b3 = 2;
        check(allocator, backend, batch);
        batch.a2 = batch.b2; batch.a3 = batch.b3;
        check(allocator, backend, batch);
        for (bool indexed : {false, true}) for (pattern data : {pattern::activation_residual, pattern::large_activation})
        {
            test_case c;
            c.indexed = indexed; c.tokens = 16; c.weights = weights; c.data = data;
            check(allocator, backend, c);
        }
        for (bool interleaved : {false, true})
        {
            test_case c;
            c.weights = weights; c.padded = true; c.interleaved = interleaved; c.convert = true;
            c.inner = 257; c.rows = 37; c.a2 = 2; c.a3 = 2; c.b2 = 4; c.b3 = 6;
            check(allocator, backend, c);
            c.indexed = true; c.slots = 2;
            check(allocator, backend, c);
            c.slots = c.used;
            check(allocator, backend, c);
        }
    }
    for (int tokens : {16, 31}) for (bool precise : {false, true})
    {
        test_case c;
        c.tokens = tokens; c.inner = 128; c.rows = 1280; c.precise = precise;
        check(allocator, backend, c);
    }
    for (bool indexed : {false, true})
    {
        test_case c;
        c.indexed = indexed; c.inner = 4096; c.rows = 17; c.data = pattern::weight_residual;
        check(allocator, backend, c);
        c.inner = 256; c.rows = 64; c.data = pattern::random; c.pipeline = true;
        check(allocator, backend, c);
    }
}

int main(int argc, char ** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    bool quantized = false, wide_quantized = false;
    std::string only_type;
    int inner = 256;
    for (int arg = 1; arg < argc; ++arg)
    {
        const std::string option(argv[arg]);
        if (option == "--quant-strip-only") quantized = true;
        else if (option == "--quant-strip-wide-only") wide_quantized = true;
        else if (option == "--type" && arg + 1 < argc) only_type = argv[++arg];
        else if (option == "--inner" && arg + 1 < argc) inner = std::atoi(argv[++arg]);
        else require(false, "Usage: [--quant-strip-only [--type iq2_xs] [--inner 2560]] | --quant-strip-wide-only");
    }
    require(inner > 0 && inner % 256 == 0, "Quantized --inner must be a positive multiple of256");
    require(quantized || (only_type.empty() && inner == 256), "--type/--inner require --quant-strip-only");
    require(!(quantized && wide_quantized), "Select one quantized test mode");
#if !defined(TSG_GGML_USE_CUDA) || !defined(TSG_GGML_TEST_HOOKS)
    if (wide_quantized) return 77;
#endif
#ifdef TSG_GGML_USE_CUDA
    const int devices = ggml_backend_cuda_get_device_count();
    if (devices == 0) return 77;
    for (int device = 0; device < devices; ++device)
    {
        auto * cuda = ggml_backend_cuda_init(device);
        require(cuda != nullptr, "Cannot initialize a visible CUDA device");
        auto * backend = tsg_dsv4_fused_backend_init(cuda);
        require(backend != nullptr, "Cannot initialize TensorSharp CUDA precision backend");
        if (wide_quantized) {
#if defined(TSG_GGML_TEST_HOOKS)
            run_wide_quantized(cuda, backend);
#endif
        }
        else if (quantized) run_quantized(cuda, backend, only_type, inner);
        else run(cuda, backend);
        ggml_backend_free(backend);
        ggml_backend_free(cuda);
    }
#else
    auto * backend = ggml_backend_cpu_init();
    require(backend != nullptr, "Cannot initialize CPU precision backend");
    ggml_backend_cpu_set_n_threads(backend, 4);
    if (quantized) run_quantized(backend, backend, only_type, inner);
    else run(backend, backend);
    ggml_backend_free(backend);
#endif
    return 0;
}
