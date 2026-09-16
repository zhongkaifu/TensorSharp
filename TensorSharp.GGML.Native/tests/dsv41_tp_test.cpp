// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_deepseek41_tp.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#if defined(TSG_GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <random>
#include <stdexcept>

namespace
{
void require(bool value, const char * message)
{
    if (!value) throw std::runtime_error(message);
}
struct fixture
{
    ggml_type type;
    int embedding, hidden, experts = 8, used = 6;
    std::filesystem::path path;
    tsg_dsv41_tp::source gate, up, down;
    std::vector<char> gate_data, up_data, down_data;
    fixture(ggml_type format, ggml_type down_format = GGML_TYPE_COUNT, int full_embedding = 0)
        : type(format), embedding(full_embedding > 0 ? full_embedding : format == GGML_TYPE_F32 ? 16 : 256),
          hidden(format == GGML_TYPE_F32 ? 18 : 2304)
    {
        std::mt19937 random(410912);
        path = std::filesystem::temp_directory_path() /
            ("dsv41-tp-" + std::to_string(std::random_device{}()) + ".weights");
        std::ofstream file(path, std::ios::binary);
        require((bool) file, "Cannot create TP fixture file");
        file << "unaligned-offset-fixture";
        auto write = [&](tsg_dsv41_tp::source & src, std::vector<char> & bytes, int inner, int rows, ggml_type storage) {
            src.path = path.string();
            src.offset = (size_t) file.tellp();
            src.type = storage;
            src.ne = {inner, rows, experts, 1};
            std::uniform_real_distribution<float> uniform(-.2f, .2f);
            std::vector<float> values((size_t) inner * rows * experts);
            for (float & x : values) x = uniform(random) / std::sqrt((float) inner);
            bytes.resize(ggml_row_size(storage, inner) * rows * experts);
            require(ggml_quantize_chunk(storage, values.data(), bytes.data(), 0, rows * experts, inner, nullptr) == bytes.size(),
                    "Cannot quantize TP test weights");
            file.write(bytes.data(), bytes.size());
        };
        write(gate, gate_data, embedding, hidden, format);
        write(up, up_data, embedding, hidden, format);
        write(down, down_data, hidden, embedding, down_format == GGML_TYPE_COUNT ? format : down_format);
    }
    // Independently materialize a column/row partition for diagnostic graphs,
    // without using executor upload_strip or writing another weight file.
    fixture(const fixture & full, tsg_dsv41_tp::strip part)
        : type(full.type), embedding(full.embedding), hidden((int) part.count), experts(full.experts), used(full.used)
    {
        auto copy = [&](const tsg_dsv41_tp::source & src, const std::vector<char> & source,
                        tsg_dsv41_tp::source & dst, std::vector<char> & bytes, bool inner) {
            dst = src;
            dst.ne[inner ? 0 : 1] = part.count;
            const size_t full_row = ggml_row_size(src.type, src.ne[0]);
            const size_t count = inner ? src.ne[1] * src.ne[2] : src.ne[2];
            const size_t width = inner ? ggml_row_size(src.type, part.count) : full_row * part.count;
            const size_t offset = inner ? ggml_row_size(src.type, part.first) : full_row * part.first;
            const size_t stride = inner ? full_row : full_row * src.ne[1];
            bytes.resize(count * width);
            for (size_t i = 0; i < count; ++i)
                std::memcpy(bytes.data() + i * width, source.data() + offset + i * stride, width);
        };
        copy(full.gate, full.gate_data, gate, gate_data, false);
        copy(full.up, full.up_data, up, up_data, false);
        copy(full.down, full.down_data, down, down_data, true);
    }
    ~fixture() { std::error_code error; std::filesystem::remove(path, error); }
};

struct evaluation
{
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * x = nullptr, * ids = nullptr, * weights = nullptr, * out = nullptr;
    ggml_tensor * gate_out = nullptr, * up_out = nullptr, * hidden_out = nullptr;
    evaluation(const fixture & data, int tokens, tsg_dsv41_tp::executor * tp, int layer,
               ggml_backend_dev_t reference_device = nullptr, bool preserve_taps = false, bool shared_once = false)
    {
        backend = !tp && reference_device ? ggml_backend_dev_init(reference_device, nullptr) : ggml_backend_cpu_init();
        if (ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_n_threads(backend, 1);
        ctx = ggml_init({1024 * 1024, nullptr, true});
        x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, data.embedding, tokens);
        if (tp && layer == 1)
        {
            // Match argsort_top_k: selected IDs retain the full router's
            // original column stride and must be materialized by the TP hook.
            auto * storage = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, data.used + 3, tokens);
            ids = ggml_view_2d(ctx, storage, data.used, tokens, storage->nb[1], 0);
        }
        else ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, data.used, tokens);
        weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, data.used, tokens);
        ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
        if (tp) out = tp->build(ctx, layer, x, weights, ids);
        else
        {
            gate = ggml_new_tensor_3d(ctx, data.gate.type, data.embedding, data.hidden, data.experts);
            up = ggml_new_tensor_3d(ctx, data.up.type, data.embedding, data.hidden, data.experts);
            down = ggml_new_tensor_3d(ctx, data.down.type, data.hidden, data.embedding, data.experts);
            auto * input = ggml_reshape_3d(ctx, x, data.embedding, 1, tokens);
            auto * g = ggml_clamp(ctx, ggml_mul_mat_id(ctx, gate, input, ids), -INFINITY, .1f);
            auto * u = ggml_clamp(ctx, ggml_mul_mat_id(ctx, up, input, ids), -.1f, .1f);
            auto * h = ggml_swiglu_split(ctx, g, u);
            gate_out = g; up_out = u; hidden_out = h;
            if (preserve_taps) { ggml_set_output(g); ggml_set_output(u); ggml_set_output(h); }
            auto * e = ggml_mul(ctx, ggml_mul_mat_id(ctx, down, h, ids), weights);
            for (int expert = 0; expert < data.used; ++expert)
            {
                auto * view = ggml_view_2d(ctx, e, data.embedding, tokens, e->nb[2], expert * e->nb[1]);
                out = out ? ggml_add(ctx, out, view) : view;
            }
        }
        // The production caller adds the unsharded shared-expert output after
        // the routed TP reduction. Exercise that boundary with a known tensor;
        // the independent scalar check below rejects rank-count multiplication.
        ggml_tensor * shared = nullptr;
        if (shared_once)
        {
            shared = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, data.embedding, tokens);
            out = ggml_add(ctx, out, shared);
        }
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        {
            auto * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_MUL_MAT_ID && node->src[0]->type == GGML_TYPE_F32)
            {
                ggml_prec_set_acc(node, GGML_PREC_F32);
                ggml_prec_set_src(node, GGML_PREC_F32, 1);
            }
        }
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "Cannot allocate TP test graph");
        if (!tp)
        {
            ggml_backend_tensor_set(gate, data.gate_data.data(), 0, data.gate_data.size());
            ggml_backend_tensor_set(up, data.up_data.data(), 0, data.up_data.size());
            ggml_backend_tensor_set(down, data.down_data.data(), 0, data.down_data.size());
        }
        std::vector<float> input(data.embedding * tokens), routing(data.used * tokens);
        std::vector<int> selected(data.used * tokens);
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin((float) (i + 1) * .13f);
        for (int token = 0; token < tokens; ++token) for (int e = 0; e < data.used; ++e)
        {
            selected[token * data.used + e] = (token + e) % data.experts;
            routing[token * data.used + e] = float(e + 1) / (data.used * (data.used + 1) / 2);
        }
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        for (int token = 0; token < tokens; ++token)
            ggml_backend_tensor_set(ids, selected.data() + token * data.used,
                                    token * ids->nb[1], data.used * sizeof(int));
        ggml_backend_tensor_set(weights, routing.data(), 0, routing.size() * sizeof(float));
        if (shared)
        {
            std::vector<float> values(data.embedding * tokens);
            for (size_t i = 0; i < values.size(); ++i) values[i] = (i % 2 ? -.125f : .25f);
            ggml_backend_tensor_set(shared, values.data(), 0, values.size() * sizeof(float));
        }
    }
    ~evaluation()
    {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
    }
    std::vector<float> run()
    {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "TP test forward failed");
        std::vector<float> result(ggml_nelements(out));
        ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));
        return result;
    }
};

// Small F32 oracle: no ggml graph, TP split/upload helper or sliced reference
// participates. Direct logical tensor indexing and double dot products make
// this independent of both the rank executor and the full-weight graph oracle.
std::vector<float> scalar_f32_reference(const fixture & data, int tokens, bool shared_once)
{
    require(data.gate.type == GGML_TYPE_F32 && data.down.type == GGML_TYPE_F32,
            "Scalar TP oracle requires F32 weights");
    auto weight = [](const std::vector<char> & bytes, size_t index) {
        float value;
        std::memcpy(&value, bytes.data() + index * sizeof(float), sizeof(float));
        return value;
    };
    std::vector<float> result(data.embedding * tokens);
    for (int token = 0; token < tokens; ++token)
    {
        for (int selected = 0; selected < data.used; ++selected)
        {
            const int expert = (token + selected) % data.experts;
            const float routing = float(selected + 1) / (data.used * (data.used + 1) / 2);
            std::vector<float> hidden(data.hidden);
            for (int row = 0; row < data.hidden; ++row)
            {
                double gate = 0, up = 0;
                for (int k = 0; k < data.embedding; ++k)
                {
                    const float x = std::sin(float(token * data.embedding + k + 1) * .13f);
                    const size_t index = ((size_t) expert * data.hidden + row) * data.embedding + k;
                    gate += double(weight(data.gate_data, index)) * x;
                    up += double(weight(data.up_data, index)) * x;
                }
                const float g = std::min(.1f, (float) gate), u = std::clamp((float) up, -.1f, .1f);
                hidden[row] = g / (1.0f + std::exp(-g)) * u;
            }
            for (int row = 0; row < data.embedding; ++row)
            {
                double dot = 0;
                for (int k = 0; k < data.hidden; ++k)
                    dot += double(weight(data.down_data, ((size_t) expert * data.embedding + row) * data.hidden + k)) * hidden[k];
                result[token * data.embedding + row] += (float) dot * routing;
            }
        }
    }
    if (shared_once)
        for (size_t i = 0; i < result.size(); ++i) result[i] += (i % 2 ? -.125f : .25f);
    return result;
}

void check_partitions()
{
    for (int ranks : {2, 4, 7, 8})
    {
        std::vector<int64_t> aggregate(ranks);
        for (int layer = 0; layer < ranks; ++layer)
        {
            const auto strips = tsg_dsv41_tp::split(2304, 256, ranks, layer);
            int64_t end = 0;
            int wide = 0;
            for (int rank = 0; rank < ranks; ++rank)
            {
                const auto & part = strips[rank];
                require(part.first == end && part.first % 256 == 0 && part.count % 256 == 0 && part.count > 0,
                        "TP split is incomplete, overlapping, or unaligned");
                require(part.count == (9 / ranks) * 256 || part.count == (9 / ranks + 1) * 256,
                        "TP strip is not balanced by quantization blocks");
                wide += part.count > (9 / ranks) * 256;
                aggregate[rank] += part.count;
                end += part.count;
            }
            require(end == 2304 && wide == 9 % ranks, "TP split lost rows or assigned the wrong number of wider strips");
            if (ranks == 7 && layer == 6)
                require(strips[0].count == 512 && strips[6].count == 512 &&
                        std::all_of(strips.begin() + 1, strips.begin() + 6, [](auto part) { return part.count == 256; }),
                        "Seven-rank wider strips did not rotate across the last/first rank boundary");
        }
        require(std::all_of(aggregate.begin(), aggregate.end(), [](int64_t count) { return count == 2304; }),
                "A complete layer rotation leaves imbalanced expert strips");
    }
    for (auto type : {GGML_TYPE_BF16, GGML_TYPE_F16})
    for (int64_t width : {int64_t(2304), int64_t(2368)})
    for (int ranks : {2, 4, 7, 8})
    {
        std::vector<int64_t> aggregate(ranks);
        for (int layer = 0; layer < ranks; ++layer)
        {
            const auto parts = tsg_dsv41_tp::split_weights(width, type, ranks, layer);
            int64_t end = 0;
            int wide = 0;
            for (int rank = 0; rank < ranks; ++rank)
            {
                const auto part = parts[rank];
                require(part.first == end && part.first % 64 == 0 && part.count > 0 && part.count % 64 == 0,
                        "Floating TP weights lost full coverage or matrix alignment");
                require(part.count == (width / 64 / ranks) * 64 || part.count == (width / 64 / ranks + 1) * 64,
                        "Floating TP strips are not balanced by 64-channel blocks");
                wide += part.count > (width / 64 / ranks) * 64;
                end += part.count;
                aggregate[rank] += part.count;
            }
            require(end == width && wide == width / 64 % ranks, "Floating TP split lost blocks");
            if (width == 2368 && ranks == 7 && layer == 6)
                require(parts[6].count == 384 && parts[0].count == 384 && parts[1].count == 320,
                        "Floating TP wider strips did not wrap across the last/first rank boundary");
        }
        require(std::all_of(aggregate.begin(), aggregate.end(), [=](int64_t count) { return count == width; }),
                "Floating TP complete layer rotation is unbalanced");
    }
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K})
    for (int ranks : {2, 4, 7, 8})
    {
        // F32/quantized layouts must not change. Floating small/non-64 widths
        // must still form valid nonempty strips through the original fallback.
        const int64_t width = type == GGML_TYPE_BF16 ? 128 : type == GGML_TYPE_F16 ? 130 : 2304;
        if ((type == GGML_TYPE_BF16 || type == GGML_TYPE_F16) && width % 64 == 0 && width / 64 >= ranks) continue;
        const auto before = tsg_dsv41_tp::split(width, ggml_blck_size(type), ranks, 1);
        const auto after = tsg_dsv41_tp::split_weights(width, type, ranks, 1);
        for (int rank = 0; rank < ranks; ++rank)
            require(before[rank].first == after[rank].first && before[rank].count == after[rank].count,
                    "Typed split changed F32/quantized layout or small floating fallback");
    }
}

std::vector<float> values(ggml_tensor * tensor)
{
    std::vector<float> output(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, output.data(), 0, output.size() * sizeof(float));
    return output;
}

void report_difference(const char * label, const std::vector<float> & reference, const std::vector<float> & actual)
{
    double maximum = 0, error = 0, scale = 0;
    for (size_t i = 0; i < reference.size(); ++i)
    {
        const double d = actual[i] - reference[i];
        maximum = std::max(maximum, std::abs(d)); error += d * d; scale += double(reference[i]) * reference[i];
    }
    std::cout << "DIAGNOSTIC " << label << " max_abs=" << maximum << " rel_l2="
              << std::sqrt(error / std::max(1e-30, scale)) << std::endl;
}

void diagnose(const fixture & data, int tokens, int ranks, int layer, ggml_backend_dev_t device,
              const std::vector<float> & tp_output)
{
    evaluation full(data, tokens, nullptr, layer, device, true);
    const auto full_output = full.run();
    const auto full_gate = values(full.gate_out), full_up = values(full.up_out), full_hidden = values(full.hidden_out);
    std::vector<float> gate(full_gate.size()), up(full_up.size()), hidden(full_hidden.size()), sum(full_output.size());
    for (auto part : tsg_dsv41_tp::split_weights(data.hidden, data.down.type, ranks, layer))
    {
        fixture subset(data, part);
        evaluation sliced(subset, tokens, nullptr, layer, device, true);
        const auto output = sliced.run();
        const auto g = values(sliced.gate_out), u = values(sliced.up_out), h = values(sliced.hidden_out);
        for (int row = 0; row < tokens * data.used; ++row)
        {
            std::copy_n(g.data() + row * part.count, part.count, gate.data() + row * data.hidden + part.first);
            std::copy_n(u.data() + row * part.count, part.count, up.data() + row * data.hidden + part.first);
            std::copy_n(h.data() + row * part.count, part.count, hidden.data() + row * data.hidden + part.first);
        }
        for (size_t i = 0; i < sum.size(); ++i) sum[i] += output[i];
    }
    report_difference("gate-column-strips", full_gate, gate);
    report_difference("up-column-strips", full_up, up);
    report_difference("swiglu-column-strips", full_hidden, hidden);
    report_difference("same-device-manual-strips", full_output, sum);
    report_difference("manual-strips-versus-tp", sum, tp_output);
    // CUDA's Q3_K MMQ path uses one scale per32 activations and roundf(x*127/max).
    // This host calculation estimates the number of changed downstream bins;
    // raw projection errors above are measured directly on the executing device.
    size_t bins = 0;
    for (size_t first = 0; first < hidden.size(); first += 32)
    {
        float a = 0, b = 0;
        for (size_t i = first; i < first + 32; ++i) { a = std::max(a, std::abs(full_hidden[i])); b = std::max(b, std::abs(hidden[i])); }
        const float inv_a = a > 0 ? 127.0f / a : 0, inv_b = b > 0 ? 127.0f / b : 0;
        for (size_t i = first; i < first + 32; ++i)
            bins += std::round(full_hidden[i] * inv_a) != std::round(hidden[i] * inv_b);
    }
    std::cout << "DIAGNOSTIC host-estimated-Q8-bin-changes=" << bins << " of=" << hidden.size() << std::endl;
}

void set_fault(const char * stage)
{
#if defined(_WIN32)
    _putenv_s("TS_DSV41_TEST_FAIL_STAGE", stage ? stage : "");
#else
    if (stage) setenv("TS_DSV41_TEST_FAIL_STAGE", stage, 1);
    else unsetenv("TS_DSV41_TEST_FAIL_STAGE");
#endif
}

void failure_recovery(const std::vector<ggml_backend_dev_t> & devices)
{
    fixture data(GGML_TYPE_F32);
    const auto expected = evaluation(data, 3, nullptr, 0).run();
    int checks = 0;
    auto check = [&](bool condition, const char * message) { require(condition, message); ++checks; };
    auto compare = [&](const std::vector<float> & output) {
        double error = 0, norm = 0;
        for (size_t i = 0; i < output.size(); ++i)
        {
            require(std::isfinite(output[i]), "Recovered TP graph produced nonfinite output");
            error += std::pow(double(output[i]) - expected[i], 2);
            norm += double(expected[i]) * expected[i];
        }
        check(std::sqrt(error / norm) < 1e-5, "Recovered TP graph differs from independent full reference");
    };
    for (bool single : {false, true})
    for (const char * stage : {"tp-graph", "tp-rank", "tp-sync", "tp-rank-and-sync"})
    {
        // The candidate additionally preserves a submission exception when
        // draining that same rank also reports a failure.
        if (!single && std::strcmp(stage, "tp-rank-and-sync") == 0) continue;
        tsg_dsv41_tp::executor tp(devices, data.used);
        tp.test_single_fanout(single);
        for (int layer : {0, 1}) tp.add_layer(layer, data.gate, data.up, data.down, .1f);
        tp.begin_forward();
        tp.test_set_position(0);
        set_fault(stage);
        const auto failed = evaluation(data, 3, &tp, 0).run();
        set_fault(nullptr);
        const char * expected_error = std::strcmp(stage, "tp-rank-and-sync") == 0 ? "tp-rank" : stage;
        check(tp.error().find(expected_error) != std::string::npos, "Rank fault did not reach the TP error latch");
        check(std::all_of(failed.begin(), failed.end(), [](float x) { return std::isnan(x); }),
              "Failed TP callback did not mark its output invalid");
        const auto first_error = tp.error();
        // A healthy later layer must preserve the earlier request error.
        compare(evaluation(data, 3, &tp, 1).run());
        check(tp.error() == first_error, "A later healthy layer concealed a prior rank fault");
        tp.begin_forward();
        check(tp.error().empty(), "A new request retained a stale TP error");
        // Reuse the failed shape, including the genuinely incomplete graph
        // produced by tp-graph. A prematurely published context would crash.
        compare(evaluation(data, 3, &tp, 0).run());
        check(tp.error().empty(), "Same-shape recovery retained an error");
        compare(evaluation(data, 3, &tp, 1).run());
        check(tp.error().empty(), "Recovered TP executor damaged another layer");
    }
    std::cout << "Passed " << checks << " TP failure lifecycle checks\n";
}

void paired_fanout(const std::vector<ggml_backend_dev_t> & devices, int pairs)
{
    fixture data(GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, 5120);
    tsg_dsv41_tp::executor tp(devices, data.used);
    tp.add_layer(0, data.gate, data.up, data.down, .1f);
    for (int tokens : {1, 16})
    {
        evaluation model(data, tokens, &tp, 0);
        auto run = [&](bool single) {
            tp.test_single_fanout(single);
            tp.begin_forward();
            const auto start = std::chrono::steady_clock::now();
            auto output = model.run();
            const double milliseconds = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            require(tp.error().empty(), tp.error().c_str());
            return std::make_pair(std::move(output), milliseconds);
        };
        for (int warm = 0; warm < 10; ++warm) { run(false); run(true); }
        std::vector<double> separate, combined;
        for (int pair = 0; pair < pairs; ++pair)
        {
            // Alternate order to keep consistent order bias out of the pair.
            auto first = run(pair % 2 != 0), second = run(pair % 2 == 0);
            require(first.first.size() == second.first.size() &&
                std::memcmp(first.first.data(), second.first.data(), first.first.size() * sizeof(float)) == 0,
                "Single/two-fanout TP outputs are not bitwise identical");
            require(std::all_of(first.first.begin(), first.first.end(), [](float x) { return std::isfinite(x); }),
                    "Paired TP output contains nonfinite values");
            separate.push_back(pair % 2 ? second.second : first.second);
            combined.push_back(pair % 2 ? first.second : second.second);
        }
        auto array = [](const std::vector<double> & values) {
            std::cout << '[';
            for (size_t i = 0; i < values.size(); ++i) { if (i) std::cout << ','; std::cout << values[i]; }
            std::cout << ']';
        };
        std::cout << std::setprecision(17) << "FANOUT_BENCH {\"ranks\":" << devices.size()
                  << ",\"tokens\":" << tokens << ",\"embedding\":5120,\"hidden\":2304,\"experts\":8,\"top_k\":6"
                  << ",\"gate_up\":\"q2_k\",\"down\":\"q3_k\",\"pairs\":" << pairs
                  << ",\"bitwise_equal\":true,\"two_fanout_ms\":";
        array(separate);
        std::cout << ",\"single_fanout_ms\":"; array(combined);
        std::cout << "}" << std::endl;
    }
}
}

int main(int argc, char ** argv)
{
    try
    {
        bool cuda = false, checkpoint_shape = false, diagnostic = false, down_f32 = false, failure_only = false;
        int cuda_ranks = 0, fanout_pairs = 0;
        for (int arg = 1; arg < argc; ++arg)
        {
            const std::string option = argv[arg];
            if (option == "--cuda" && arg + 1 < argc) { cuda = true; cuda_ranks = std::stoi(argv[++arg]); }
            else if (option == "--checkpoint-shape") checkpoint_shape = true;
            else if (option == "--diagnose") diagnostic = true;
            else if (option == "--diagnostic-down-f32") down_f32 = true;
            else if (option == "--failure-only") failure_only = true;
            else if (option == "--fanout-pairs" && arg + 1 < argc) fanout_pairs = std::stoi(argv[++arg]);
            else throw std::runtime_error("Usage: GgmlOpsDsv41TpTest [--cuda 2|4|7|8] [--checkpoint-shape] [--diagnose] [--diagnostic-down-f32] [--failure-only|--fanout-pairs N]");
        }
        if (cuda && cuda_ranks != 2 && cuda_ranks != 4 && cuda_ranks != 7 && cuda_ranks != 8) return 1;
        require(!down_f32 || checkpoint_shape, "The F32-down control requires --checkpoint-shape");
        require(fanout_pairs >= 0 && fanout_pairs <= 10000 && !(fanout_pairs && failure_only), "Invalid fanout comparison count");
        std::vector<ggml_backend_dev_t> devices;
#if defined(TSG_GGML_USE_CUDA)
        if (cuda)
        {
            if (ggml_backend_cuda_get_device_count() < cuda_ranks) return 77;
            auto reg = ggml_backend_cuda_reg();
            for (int i = 0; i < cuda_ranks; ++i) devices.push_back(ggml_backend_reg_dev_get(reg, i));
        }
#else
        if (cuda) return 77;
#endif
        const auto cpu = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
        if (failure_only)
        {
            failure_recovery(cuda ? devices : std::vector<ggml_backend_dev_t>(2, cpu));
            return 0;
        }
        if (fanout_pairs)
        {
            if (cuda) paired_fanout(devices, fanout_pairs);
            else for (int ranks : {2, 4, 8}) paired_fanout(std::vector<ggml_backend_dev_t>(ranks, cpu), fanout_pairs);
            return 0;
        }
        // Quantized down rows have nine blocks, including seven unequal ranks.
        check_partitions();
        bool rejected = false;
        try { tsg_dsv41_tp::split(256, 256, 2, 0); } catch (const std::exception &) { rejected = true; }
        require(rejected, "TP accepted empty rank strips");
        int checks = 0;
        // The published Q2_K checkpoint stores routed gate/up in Q2_K and
        // down in Q3_K. Its nine down blocks must remain independently typed
        // and aligned when every rank receives a different hidden-width strip.
        const std::vector<std::pair<ggml_type, ggml_type>> formats = checkpoint_shape
            ? (down_f32
                ? std::vector<std::pair<ggml_type, ggml_type>>{{GGML_TYPE_Q2_K, GGML_TYPE_F32}}
                : std::vector<std::pair<ggml_type, ggml_type>>{{GGML_TYPE_Q2_K, GGML_TYPE_Q3_K}, {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}})
            : std::vector<std::pair<ggml_type, ggml_type>>{{GGML_TYPE_F32, GGML_TYPE_F32},
                {GGML_TYPE_BF16, GGML_TYPE_BF16}, {GGML_TYPE_F16, GGML_TYPE_F16}, {GGML_TYPE_Q2_K, GGML_TYPE_Q2_K},
                {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K}, {GGML_TYPE_Q4_K, GGML_TYPE_Q4_K},
                {GGML_TYPE_Q6_K, GGML_TYPE_Q6_K}, {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}};
        for (auto [type, down_type] : formats)
        {
            fixture data(type, down_type, checkpoint_shape ? 5120 : 0);
            std::cout << "Fixture gate=" << ggml_type_name(type) << " up=" << ggml_type_name(type)
                      << " down=" << ggml_type_name(down_type) << " embedding=" << data.embedding
                      << " hidden=" << data.hidden << " experts=" << data.experts << " top_k=" << data.used << std::endl;
            std::map<int, std::vector<float>> expected;
            for (int tokens : {1, 5, 16})
                expected[tokens] = evaluation(data, tokens, nullptr, 0, cuda ? devices[0] : nullptr).run();
            for (int ranks : {2, 4, 7, 8})
            {
                if (cuda && ranks != cuda_ranks) continue;
                auto rank_devices = cuda ? devices : std::vector<ggml_backend_dev_t>(ranks, cpu);
                tsg_dsv41_tp::executor tp(rank_devices, data.used);
                // Reuse scratch across layers and shapes, including seven-rank
                // wider strips wrapping from rank six back to rank zero.
                const std::vector<int> layers = ranks == 7 ? std::vector<int>{0, 1, 6} : std::vector<int>{0, 1};
                for (int layer : layers) tp.add_layer(layer, data.gate, data.up, data.down, .1f);
                size_t total = 0;
                for (int rank = 0; rank < ranks; ++rank) total += tp.rank_weight_bytes(rank);
                require(total == layers.size() * (data.gate_data.size() + data.up_data.size() + data.down_data.size()),
                        "TP duplicated or dropped weight bytes");
                if (type == GGML_TYPE_F32)
                {
                    for (bool shared_once : {false, true})
                    {
                        const auto independent = scalar_f32_reference(data, 3, shared_once);
                        const auto actual = evaluation(data, 3, &tp, layers.back(), nullptr, false, shared_once).run();
                        require(tp.error().empty(), tp.error().c_str());
                        for (size_t i = 0; i < actual.size(); ++i)
                            require(std::isfinite(actual[i]) && std::abs(actual[i] - independent[i]) < 1e-7,
                                    "TP differs from scalar F32 oracle or counts shared output more than once");
                        std::cout << "Scalar F32 oracle ranks=" << ranks << " shared_output=" << shared_once << " passed\n";
                        ++checks;
                    }
                }
                for (int tokens : {1, 5, 16, 1}) for (int layer : layers)
                {
                    auto actual = evaluation(data, tokens, &tp, layer).run();
                    require(tp.error().empty(), tp.error().c_str());
                    double maximum = 0, squared_error = 0, squared_reference = 0;
                    for (size_t i = 0; i < actual.size(); ++i)
                    {
                        const double error = std::abs(actual[i] - expected[tokens][i]);
                        maximum = std::max(maximum, error);
                        squared_error += error * error;
                        squared_reference += double(expected[tokens][i]) * expected[tokens][i];
                        require(std::isfinite(actual[i]), "TP produced nonfinite output");
                    }
                    const double relative = std::sqrt(squared_error / std::max(1e-30, squared_reference));
                    std::cout << ggml_type_name(type) << "/" << ggml_type_name(down_type) << " ranks=" << ranks << " layer=" << layer
                              << " tokens=" << tokens << " max_abs=" << maximum << " rel_l2=" << relative << std::endl;
                    if (diagnostic && relative >= 1e-5)
                        diagnose(data, tokens, ranks, layer, cuda ? devices[0] : cpu, actual);
                    require(maximum < (type == GGML_TYPE_F32 ? 1e-7 : 2e-6),
                            "TP output differs from the full-weight reference");
                    require(relative < 1e-5,
                            "TP relative error exceeds the full-weight reference tolerance");
                    ++checks;
                }
            }
        }
        require(checks > 0, "No TP numerical comparisons executed");
        std::cout << "Passed " << checks << " true MoE tensor-parallel comparisons\n";
        return 0;
    }
    catch (const std::exception & error)
    {
        set_fault(nullptr);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
