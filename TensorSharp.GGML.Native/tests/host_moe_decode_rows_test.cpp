// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Standalone CPU regression: link the production ggml_ops_host_moe_decode.cpp
// with unchanged ggml-cpu/ggml-base. No accelerator, model checkpoint, or seam
// implementation is substituted. The only TensorSharp dependency stub is the
// configurable default worker count below. --bench adds an optional hot-memory
// kernel diagnostic, not an end-to-end or checkpoint throughput benchmark.
#include "ggml_ops_internal.h"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace tsg {
    bool host_moe_decode_experts_rows(const HostMoeSegment&, const float*, const std::int32_t*, const float*, float*);
    static int test_threads = 1;
    int host_moe_default_thread_count() { return test_threads; }
}

namespace {
    constexpr float sentinel = -123456.75f;
    constexpr int guard = 16;

    void require(bool condition, const std::string& detail)
    {
        if (!condition) { std::fprintf(stderr, "FAIL: %s\n", detail.c_str()); std::exit(1); }
    }

    std::vector<float> values(std::size_t count, std::uint32_t seed, float scale)
    {
        std::vector<float> result(count);
        for (float& x : result)
        {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            x = float(int(seed & 0xffffu) - 32768) * (scale / 32768.f);
        }
        return result;
    }

    std::vector<std::uint8_t> quantized(ggml_type type, int inner, int outer, int experts, std::uint32_t seed)
    {
        ggml_quantize_init(type);
        const std::size_t row_bytes = ggml_row_size(type, inner);
        std::vector<std::uint8_t> result(row_bytes * outer * experts);
        std::vector<float> importance(inner, 1.f);
        for (int expert = 0; expert < experts; ++expert)
        {
            auto input = values(std::size_t(inner) * outer, seed + 7919u * expert, .035f);
            const auto written = ggml_quantize_chunk(type, input.data(), result.data() + std::size_t(expert) * outer * row_bytes,
                0, outer, inner, ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr);
            require(written == row_bytes * outer, "Quantization produced an incomplete expert matrix.");
        }
        return result;
    }

    struct Fixture
    {
        tsg::HostMoeSegment hm;
        std::array<std::vector<std::uint8_t>, 3> weights;

        Fixture(ggml_type type, int hidden = 256, int ff = 256, int experts = 12, int used = 5)
            : Fixture(std::array<ggml_type, 3>{type, type, type}, hidden, ff, experts, used) {}

        Fixture(const std::array<ggml_type, 3>& types, int hidden = 256, int ff = 256, int experts = 12, int used = 5)
        {
            weights[0] = quantized(types[0], hidden, ff, experts, 101);
            weights[1] = quantized(types[1], hidden, ff, experts, 211);
            weights[2] = quantized(types[2], ff, hidden, experts, 307);
            hm.hidden = hidden; hm.n_ff = ff; hm.num_experts = experts; hm.n_used = used;
            hm.independent_decode_rows = true;
            hm.gate_data = weights[0].data(); hm.gate_type = types[0];
            hm.gate_ne0 = hidden; hm.gate_ne1 = ff; hm.gate_bytes = weights[0].size();
            hm.up_data = weights[1].data(); hm.up_type = types[1];
            hm.up_ne0 = hidden; hm.up_ne1 = ff; hm.up_bytes = weights[1].size();
            hm.down_data = weights[2].data(); hm.down_type = types[2];
            hm.down_ne0 = ff; hm.down_ne1 = hidden; hm.down_bytes = weights[2].size();
        }

        std::string name() const
        {
            return std::string(ggml_type_name((ggml_type)hm.gate_type)) + "/"
                + ggml_type_name((ggml_type)hm.up_type) + "/" + ggml_type_name((ggml_type)hm.down_type);
        }
    };

    struct Inputs
    {
        std::vector<float> x, route;
        std::vector<std::int32_t> ids;

        Inputs(const tsg::HostMoeSegment& hm, int width, int iteration)
        {
            x = values(std::size_t(width) * hm.hidden, 104729u + 37u * iteration, .8f);
            ids.resize(std::size_t(width) * hm.n_used); route.resize(ids.size());
            for (int row = 0; row < width; ++row)
                for (int slot = 0; slot < hm.n_used; ++slot)
                {
                    const std::size_t i = std::size_t(row) * hm.n_used + slot;
                    // Duplicate slots, reversed order, distinct rows, and
                    // positive/negative coefficients must preserve slot order.
                    const int source = slot == 2 ? 0 : slot;
                    ids[i] = (3 * row + hm.n_used - source + iteration) % hm.num_experts;
                    route[i] = (slot % 3 == 1 ? -1.f : 1.f) * (.13f + float((row + slot + iteration) % 11) / 32.f);
                }
        }
    };

    std::vector<float> solo(const tsg::HostMoeSegment& hm, const Inputs& in, int width)
    {
        std::vector<float> output(std::size_t(width) * hm.hidden);
        auto one = hm; one.seq_len = 1;
        for (int row = 0; row < width; ++row)
            require(tsg::host_moe_decode_experts(one, in.x.data() + std::size_t(row) * hm.hidden,
                in.ids.data() + std::size_t(row) * hm.n_used, in.route.data() + std::size_t(row) * hm.n_used,
                output.data() + std::size_t(row) * hm.hidden), "Solo production kernel declined a supported fixture.");
        return output;
    }

    void check(const Fixture& fixture, int width, int iteration)
    {
        auto hm = fixture.hm; hm.seq_len = width;
        Inputs in(hm, width, iteration);
        auto expected = solo(hm, in, width);
        std::vector<float> output(expected.size() + 2 * guard, sentinel);
        require(tsg::host_moe_decode_experts_rows(hm, in.x.data(), in.ids.data(), in.route.data(), output.data() + guard),
            "Batched production kernel declined a supported fixture.");
        int changed = 0; double max_error = 0;
        for (std::size_t i = 0; i < expected.size(); ++i)
        {
            const float actual = output[guard + i];
            require(std::isfinite(actual) && std::isfinite(expected[i]), "Non-finite expert output.");
            if (std::memcmp(&expected[i], &actual, sizeof(float))) ++changed;
            max_error = std::max(max_error, std::abs(double(expected[i]) - actual));
        }
        require(changed == 0, fixture.name() + ": width=" + std::to_string(width)
            + ", iteration=" + std::to_string(iteration) + ", bit differences=" + std::to_string(changed)
            + ", maximum error=" + std::to_string(max_error));
        require(std::all_of(output.begin(), output.begin() + guard, [](float x) { return x == sentinel; })
            && std::all_of(output.end() - guard, output.end(), [](float x) { return x == sentinel; }), "Output guard overwritten.");
    }

    void declines_untouched(const tsg::HostMoeSegment& hm, const Inputs& in, const char* reason)
    {
        std::vector<float> out(std::size_t(9) * std::max(1, hm.hidden) + 2 * guard, sentinel);
        const auto before = out;
        require(!tsg::host_moe_decode_experts_rows(hm, in.x.data(), in.ids.data(), in.route.data(), out.data() + guard), reason);
        require(std::memcmp(before.data(), out.data(), out.size() * sizeof(float)) == 0, std::string(reason) + ": output changed before decline.");
    }

    void rejection_cases(const Fixture& fixture)
    {
        auto base = fixture.hm; base.seq_len = 4;
        Inputs in(base, 9, 17);
        for (int width : {0, -1, 9}) { auto bad = base; bad.seq_len = width; declines_untouched(bad, in, "Unsupported width accepted."); }
        for (int id : {-1, base.num_experts})
        {
            auto bad_ids = in; bad_ids.ids[std::size_t(base.seq_len) * base.n_used - 1] = id;
            declines_untouched(base, bad_ids, "Invalid ID in final row accepted.");
        }
        const float bias = 1.f;
        for (int part = 0; part < 3; ++part)
        {
            auto bad = base;
            if (part == 0) bad.gate_bias = &bias;
            if (part == 1) bad.up_bias = &bias;
            if (part == 2) bad.down_bias = &bias;
            declines_untouched(bad, in, "Unsupported bias accepted.");
        }
        for (int part = 0; part < 3; ++part)
        {
            auto bad = base;
            if (part == 0) bad.gate_type = GGML_TYPE_COUNT;
            if (part == 1) bad.up_type = -1;
            if (part == 2) bad.down_type = GGML_TYPE_COUNT;
            declines_untouched(bad, in, "Invalid weight type accepted.");
        }
        auto bad = base; --bad.gate_bytes; declines_untouched(bad, in, "Truncated gate bytes accepted.");
        bad = base; --bad.up_bytes; declines_untouched(bad, in, "Truncated up bytes accepted.");
        bad = base; --bad.down_bytes; declines_untouched(bad, in, "Truncated down bytes accepted.");
        bad = base; --bad.gate_ne0; declines_untouched(bad, in, "Wrong gate inner dimension accepted.");
        bad = base; --bad.down_ne1; declines_untouched(bad, in, "Wrong down outer dimension accepted.");
        bad = base; bad.up_data = nullptr; declines_untouched(bad, in, "Unsupported fused gate/up accepted.");
        bad = base; bad.activation = 1; declines_untouched(bad, in, "Unsupported activation accepted.");
        bad = base; bad.n_used = 0; declines_untouched(bad, in, "No selected experts accepted.");
    }

    void benchmark()
    {
        constexpr int steps = 128;
        Fixture fixture(GGML_TYPE_IQ2_XXS, 1024, 512, 16, 10);
        tsg::test_threads = 4; tsg::host_moe_decode_release();
        for (int width : {3, 4})
        {
            auto hm = fixture.hm; hm.seq_len = width;
            Inputs in(hm, width, 19);
            std::vector<float> out(std::size_t(width) * hm.hidden);
            auto one = hm; one.seq_len = 1;
            check(fixture, width, 19);
            for (int warm = 0; warm < 3; ++warm)
                require(tsg::host_moe_decode_experts_rows(hm, in.x.data(), in.ids.data(), in.route.data(), out.data()), "Benchmark warmup declined.");
            auto started = std::chrono::steady_clock::now();
            for (int step = 0; step < steps; ++step)
                for (int row = 0; row < width; ++row)
                    require(tsg::host_moe_decode_experts(one, in.x.data() + std::size_t(row) * hm.hidden,
                        in.ids.data() + std::size_t(row) * hm.n_used, in.route.data() + std::size_t(row) * hm.n_used,
                        out.data() + std::size_t(row) * hm.hidden), "Benchmark solo declined.");
            const double serial = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            started = std::chrono::steady_clock::now();
            for (int step = 0; step < steps; ++step)
                require(tsg::host_moe_decode_experts_rows(hm, in.x.data(), in.ids.data(), in.route.data(), out.data()), "Benchmark batch declined.");
            const double batched = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            std::printf("diagnostic hot-memory IQ2_XXS H=1024 FF=512 experts=16 used=10 threads=4 width=%d steps=%d serial=%.6fs batch=%.6fs ratio=%.3f; no performance gate\n",
                width, steps, serial, batched, serial / batched);
        }
        tsg::host_moe_decode_release();
    }
}

int main(int argc, char** argv)
{
    bool bench = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--bench") == 0) bench = true;
        else { std::fprintf(stderr, "usage: %s [--bench]\n", argv[0]); return 2; }
    }
    int checks = 0;
    for (ggml_type type : {GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_S})
    {
        Fixture fixture(type);
        for (int threads : {1, 4})
        {
            tsg::test_threads = threads; tsg::host_moe_decode_release();
            for (int iteration = 0; iteration < 3; ++iteration)
            {
                for (int width : {1, 2, 3, 4, 8, 2, 8, 3}) { check(fixture, width, iteration); ++checks; }
                tsg::host_moe_decode_release(); // Recreate team and reuse all scratch shapes.
            }
            rejection_cases(fixture);
            tsg::host_moe_decode_release();
            std::printf("PASS %s threads=%d: exact independent rows, changing widths, duplicate/reordered IDs, negative coefficients, invalid-input preflight\n", ggml_type_name(type), threads);
        }
    }
    // Deliberately give gate/up different vec_dot activation types and row
    // sizes. Sharing one quantized input stride would corrupt rows after zero.
    // Different down types also exercise each selected expert's hidden stride
    // and the original-slot-order reduction after weight-locality sorting.
    const std::array<std::array<ggml_type, 3>, 6> mixed = {{
        {GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_IQ2_XXS},
        {GGML_TYPE_Q2_K, GGML_TYPE_Q8_0, GGML_TYPE_IQ2_S},
        {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_Q8_0},
        {GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_K},
        {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_IQ2_S},
        {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q8_0},
    }};
    for (const auto& types : mixed)
    {
        Fixture fixture(types);
        for (int threads : {1, 4})
        {
            tsg::test_threads = threads; tsg::host_moe_decode_release();
            for (int iteration = 0; iteration < 3; ++iteration)
            {
                for (int width : {3, 4, 8, 3, 8, 4}) { check(fixture, width, iteration); ++checks; }
                tsg::host_moe_decode_release();
            }
            rejection_cases(fixture);
            tsg::host_moe_decode_release();
            std::printf("PASS mixed %s threads=%d: exact independent rows and activation strides, changing widths, duplicate/reordered IDs, negative coefficients, invalid-input preflight\n",
                fixture.name().c_str(), threads);
        }
    }
    if (bench) benchmark();
    std::printf("PASS %d bitwise batch cases; CPU only, no unavailable-device scenarios\n", checks);
    ggml_quantize_free();
    return 0;
}
