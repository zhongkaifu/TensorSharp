// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Flash attention on shapes a backend has no kernel for.
//
// ggml-cuda aborts the process ("ggml-cuda/fattn.cu:730: fatal error") when a
// GGML_OP_FLASH_ATTN_EXT node it has no kernel for reaches graph compute: a head
// size outside its list (the synthetic test models use 16), or a 512-dim head
// (Gemma 4 global layers) whose KV length is not a multiple of 256 - a Gemma 4
// global cache grown from TS_KV_INITIAL_TOKENS=8 to 16 rows was exactly that.
// tsg_flash_attn_ext_guarded must turn those shapes into explicit attention
// with the same result, and leave the shapes that do have a kernel on it.
//
// Runs on the CPU backend (explicit path against a double-precision oracle,
// including sinks, logit softcap, ALiBi and query chunking) and on the first
// GPU device the build has (CUDA on Linux, Metal on macOS).
//
//   GgmlOpsFlashAttnGuardTest              the test
//   GgmlOpsFlashAttnGuardTest --unguarded  build the BARE flash node for the
//                                          head-16 case on the GPU and compute it:
//                                          on CUDA this reproduces the abort the
//                                          guard exists for (never run by ctest)

#include "ggml_ops_flash_attn_guard.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace
{
    int g_failures = 0;

    void check(bool ok, const std::string& what)
    {
        if (!ok)
        {
            std::fprintf(stderr, "FAIL: %s\n", what.c_str());
            ++g_failures;
        }
    }

    struct attention_case
    {
        const char* name;
        int head = 16, value_head = 16;
        int queries = 5, keys = 12, heads = 4, kv_heads = 2;
        int cache_rows = 0;             // > keys: K/V are strided sub-views of a larger cache
        ggml_type kv_type = GGML_TYPE_F16;
        bool mask = true;
        bool sinks = false;
        float softcap = 0.0f;
        float max_bias = 0.0f;
        std::int64_t score_budget = 0;  // explicit path only: forces query chunking
    };

    struct case_data
    {
        std::vector<float> q, k, v, mask, sinks; // logical, row-major in ggml index order
    };

    case_data make_data(const attention_case& c, std::mt19937& rng)
    {
        std::normal_distribution<float> normal(0.0f, 0.6f);
        case_data d;
        const int cache = c.cache_rows > 0 ? c.cache_rows : c.keys;
        d.q.resize(static_cast<std::size_t>(c.head) * c.queries * c.heads);
        d.k.resize(static_cast<std::size_t>(c.head) * cache * c.kv_heads);
        d.v.resize(static_cast<std::size_t>(c.value_head) * cache * c.kv_heads);
        for (auto& x : d.q) x = normal(rng);
        for (auto& x : d.k) x = normal(rng);
        for (auto& x : d.v) x = normal(rng);
        if (c.kv_type == GGML_TYPE_F16)
        {
            // Store what F16 can represent so the oracle sees the same values.
            for (auto& x : d.k) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
            for (auto& x : d.v) x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x));
        }
        // Causal mask over the last `queries` positions of `keys`, plus an
        // extra fully-allowed padding row count the flash layout permits.
        const int mask_rows = c.queries + 2;
        d.mask.assign(static_cast<std::size_t>(c.keys) * mask_rows, 0.0f);
        const int first = c.keys - c.queries;
        for (int r = 0; r < mask_rows; ++r)
            for (int s = 0; s < c.keys; ++s)
                if (r < c.queries && s > first + r)
                    d.mask[static_cast<std::size_t>(r) * c.keys + s] = -std::numeric_limits<float>::infinity();
        if (c.sinks)
        {
            d.sinks.resize(c.heads);
            for (auto& x : d.sinks) x = normal(rng);
        }
        return d;
    }

    // Double-precision reference. K/V index [x, s, h, 0] of the (possibly
    // larger) cache; the window reads rows [offset, offset + keys).
    std::vector<float> reference(const attention_case& c, const case_data& d, int offset)
    {
        const int cache = c.cache_rows > 0 ? c.cache_rows : c.keys;
        const float scale = 1.0f / std::sqrt(static_cast<float>(c.head));
        const int ratio = c.heads / c.kv_heads;
        const int n_head_log2 = 1 << static_cast<int>(std::floor(std::log2(static_cast<double>(c.heads))));
        const double m0 = std::pow(2.0, -(c.max_bias) / n_head_log2);
        const double m1 = std::pow(2.0, -(c.max_bias / 2.0) / n_head_log2);
        std::vector<float> out(static_cast<std::size_t>(c.value_head) * c.heads * c.queries, 0.0f);
        std::vector<double> w(c.keys);
        for (int n = 0; n < c.queries; ++n)
        for (int h = 0; h < c.heads; ++h)
        {
            const int hk = h / ratio;
            const double slope = c.max_bias > 0.0f
                ? (h < n_head_log2 ? std::pow(m0, h + 1) : std::pow(m1, 2 * (h - n_head_log2) + 1))
                : 1.0;
            double maximum = -std::numeric_limits<double>::infinity();
            for (int s = 0; s < c.keys; ++s)
            {
                double dot = 0.0;
                for (int x = 0; x < c.head; ++x)
                    dot += static_cast<double>(d.q[x + c.head * (n + c.queries * h)]) *
                           d.k[x + c.head * ((offset + s) + cache * hk)];
                double score = dot * scale;
                if (c.softcap != 0.0f)
                    score = c.softcap * std::tanh(score / c.softcap);
                const double bias = c.mask ? d.mask[static_cast<std::size_t>(n) * c.keys + s] : 0.0;
                w[s] = score + slope * bias;
                maximum = std::max(maximum, w[s]);
            }
            double sum = 0.0;
            if (c.sinks) { maximum = std::max(maximum, static_cast<double>(d.sinks[h])); sum += std::exp(d.sinks[h] - maximum); }
            for (int s = 0; s < c.keys; ++s) { w[s] = std::exp(w[s] - maximum); sum += w[s]; }
            for (int s = 0; s < c.keys; ++s)
            {
                const double p = w[s] / sum;
                for (int x = 0; x < c.value_head; ++x)
                    out[x + c.value_head * (h + c.heads * n)] +=
                        static_cast<float>(p * d.v[x + c.value_head * ((offset + s) + cache * hk)]);
            }
        }
        return out;
    }

    enum class build_mode { guarded, explicit_only, bare };

    struct run_result
    {
        std::vector<float> out;
        bool took_flash = false;
        bool bare_supported = false;
        std::uint64_t fallbacks = 0;
    };

    void set_kv(ggml_tensor* t, const std::vector<float>& values)
    {
        if (t->type == GGML_TYPE_F32)
        {
            ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(float));
            return;
        }
        std::vector<ggml_fp16_t> half(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) half[i] = ggml_fp32_to_fp16(values[i]);
        ggml_backend_tensor_set(t, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
    }

    run_result run_case(ggml_backend_t backend, const attention_case& c, const case_data& d, int offset, build_mode mode)
    {
        const int cache = c.cache_rows > 0 ? c.cache_rows : c.keys;
        const int mask_rows = c.queries + 2;
        ggml_init_params params = { 64 * ggml_tensor_overhead() + ggml_graph_overhead() + 1024 * 1024, nullptr, true };
        ggml_context* ctx = ggml_init(params);

        ggml_tensor* q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.head, c.queries, c.heads, 1);
        ggml_tensor* k_cache = ggml_new_tensor_4d(ctx, c.kv_type, c.head, cache, c.kv_heads, 1);
        ggml_tensor* v_cache = ggml_new_tensor_4d(ctx, c.kv_type, c.value_head, cache, c.kv_heads, 1);
        ggml_tensor* mask = c.mask ? ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.keys, mask_rows, 1, 1) : nullptr;
        ggml_tensor* sinks = c.sinks ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.heads) : nullptr;

        // The window a decode kernel hands flash attention: rows [offset, offset + keys)
        // of a larger cache, strided by the cache length (view_kv_cache_window's shape).
        ggml_tensor* k = k_cache;
        ggml_tensor* v = v_cache;
        if (cache != c.keys)
        {
            k = ggml_view_4d(ctx, k_cache, c.head, c.keys, c.kv_heads, 1,
                             k_cache->nb[1], k_cache->nb[2], k_cache->nb[3], static_cast<std::size_t>(offset) * k_cache->nb[1]);
            v = ggml_view_4d(ctx, v_cache, c.value_head, c.keys, c.kv_heads, 1,
                             v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], static_cast<std::size_t>(offset) * v_cache->nb[1]);
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(c.head));
        run_result r;
        {
            ggml_tensor* probe = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, c.max_bias, c.softcap);
            if (sinks != nullptr) ggml_flash_attn_ext_add_sinks(probe, sinks);
            r.bare_supported = ggml_backend_supports_op(backend, probe);
        }

        const std::uint64_t before = tsg_flash_attn_fallback_count();
        ggml_tensor* out = nullptr;
        switch (mode)
        {
            case build_mode::guarded:
                out = tsg_flash_attn_ext_guarded(ctx, backend, c.name, q, k, v, mask, scale, c.max_bias, c.softcap,
                                                 sinks, GGML_PREC_F32);
                break;
            case build_mode::explicit_only:
                out = tsg_attention_explicit(ctx, q, k, v, mask, scale, c.max_bias, c.softcap, sinks, c.score_budget);
                break;
            case build_mode::bare:
                out = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, c.max_bias, c.softcap);
                if (sinks != nullptr) ggml_flash_attn_ext_add_sinks(out, sinks);
                break;
        }
        r.fallbacks = tsg_flash_attn_fallback_count() - before;
        r.took_flash = out->op == GGML_OP_FLASH_ATTN_EXT;
        ggml_set_output(out);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 1024, false);
        ggml_build_forward_expand(graph, out);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (buffer == nullptr)
        {
            std::fprintf(stderr, "FAIL: %s: cannot allocate on %s\n", c.name, ggml_backend_name(backend));
            std::exit(1);
        }

        ggml_backend_tensor_set(q, d.q.data(), 0, d.q.size() * sizeof(float));
        set_kv(k_cache, d.k);
        set_kv(v_cache, d.v);
        if (mask != nullptr)
        {
            std::vector<ggml_fp16_t> half(d.mask.size());
            for (std::size_t i = 0; i < half.size(); ++i) half[i] = ggml_fp32_to_fp16(d.mask[i]);
            ggml_backend_tensor_set(mask, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
        }
        if (sinks != nullptr)
            ggml_backend_tensor_set(sinks, d.sinks.data(), 0, d.sinks.size() * sizeof(float));

        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        {
            std::fprintf(stderr, "FAIL: %s: graph compute failed on %s\n", c.name, ggml_backend_name(backend));
            std::exit(1);
        }
        r.out.resize(ggml_nelements(out));
        ggml_backend_tensor_get(out, r.out.data(), 0, r.out.size() * sizeof(float));

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return r;
    }

    double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
    {
        if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
        double m = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            const double diff = std::fabs(static_cast<double>(a[i]) - b[i]);
            if (!(diff <= m)) m = std::isnan(diff) ? std::numeric_limits<double>::infinity() : std::max(m, diff);
        }
        return m;
    }

    const double kTolerance = 2e-3;
}

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const bool unguarded = argc == 2 && std::strcmp(argv[1], "--unguarded") == 0;
    if (argc != 1 && !unguarded)
    {
        std::fprintf(stderr, "usage: %s [--unguarded]\n", argv[0]);
        return 2;
    }

    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    std::mt19937 rng(20260917);

    if (unguarded)
    {
        if (gpu == nullptr) { std::printf("no GPU backend\n"); return 77; }
        attention_case c; c.name = "bare head 16";
        const case_data d = make_data(c, rng);
        std::printf("computing a bare head-16 flash attention on %s (supports_op=%d)\n", ggml_backend_name(gpu),
                    run_case(gpu, c, d, 0, build_mode::explicit_only).bare_supported ? 1 : 0);
        run_result r = run_case(gpu, c, d, 0, build_mode::bare);
        std::printf("bare flash attention returned (max |diff| %.3g)\n", max_abs_diff(r.out, reference(c, d, 0)));
        return 0;
    }

    // ---- CPU: the explicit path itself against the oracle ----
    ggml_backend_t cpu = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(cpu, 4);
    std::vector<attention_case> explicit_cases;
    {
        attention_case c; c.name = "explicit head 16 gqa strided f16"; c.cache_rows = 20; explicit_cases.push_back(c);
        c = {}; c.name = "explicit sinks softcap"; c.sinks = true; c.softcap = 3.0f; c.kv_type = GGML_TYPE_F32; explicit_cases.push_back(c);
        c = {}; c.name = "explicit alibi"; c.max_bias = 8.0f; c.heads = 6; c.kv_heads = 3; explicit_cases.push_back(c);
        c = {}; c.name = "explicit no mask"; c.mask = false; c.head = 32; c.value_head = 24; explicit_cases.push_back(c);
        c = {}; c.name = "explicit chunked queries"; c.queries = 9; c.keys = 17; c.cache_rows = 23; c.sinks = true;
        c.score_budget = static_cast<std::int64_t>(17) * 4 * 4 * 2; explicit_cases.push_back(c);   // two query rows per chunk
    }
    for (const attention_case& c : explicit_cases)
    {
        const case_data d = make_data(c, rng);
        const int offset = c.cache_rows > 0 ? 2 : 0;
        const run_result r = run_case(cpu, c, d, offset, build_mode::explicit_only);
        const double diff = max_abs_diff(r.out, reference(c, d, offset));
        std::printf("cpu  %-36s max|diff| %.3g\n", c.name, diff);
        check(diff < kTolerance, std::string(c.name) + ": explicit attention differs from the reference");
        check(!r.took_flash && r.fallbacks == 0, std::string(c.name) + ": explicit path built a flash node or counted a fallback");
    }

    // The guard never falls back on the CPU backend, which has every kernel.
    {
        attention_case c; c.name = "cpu guarded head 16";
        const case_data d = make_data(c, rng);
        const run_result r = run_case(cpu, c, d, 0, build_mode::guarded);
        check(r.bare_supported && r.took_flash && r.fallbacks == 0, "cpu: the guard left the flash kernel for a supported shape");
        check(max_abs_diff(r.out, reference(c, d, 0)) < kTolerance, "cpu: guarded flash attention differs from the reference");
    }
    ggml_backend_free(cpu);

    // ---- GPU: unsupported shapes must fall back, supported ones must not ----
    if (gpu == nullptr)
    {
        std::printf("no GPU backend in this build/host: GPU cases skipped\n");
        return g_failures == 0 ? 0 : 1;
    }
    const std::string gpu_name = ggml_backend_name(gpu);
    const bool is_cuda = gpu_name.rfind("CUDA", 0) == 0;
    std::printf("gpu backend: %s\n", gpu_name.c_str());

    struct gpu_case { attention_case c; bool cuda_has_kernel; };
    std::vector<gpu_case> gpu_cases;
    {
        attention_case c;
        c.name = "synthetic head 16 prefill"; c.cache_rows = 20;
        gpu_cases.push_back({ c, false });
        c = {}; c.name = "synthetic head 16 decode sinks"; c.queries = 1; c.sinks = true; c.kv_type = GGML_TYPE_F32;
        gpu_cases.push_back({ c, false });
        c = {}; c.name = "gemma4 global head 512 cache 16"; c.head = c.value_head = 512; c.queries = 1; c.keys = 16;
        c.cache_rows = 16; c.heads = 8; c.kv_heads = 2;
        gpu_cases.push_back({ c, false });
        c = {}; c.name = "gemma4 global head 512 window 300 of 320"; c.head = c.value_head = 512; c.queries = 3; c.keys = 300;
        c.cache_rows = 320; c.heads = 8; c.kv_heads = 2;
        gpu_cases.push_back({ c, false });
        c = {}; c.name = "head 512 window 256 (kernel)"; c.head = c.value_head = 512; c.queries = 1; c.keys = 256;
        c.cache_rows = 512; c.heads = 8; c.kv_heads = 2;
        gpu_cases.push_back({ c, true });
        c = {}; c.name = "head 128 window 256 (kernel)"; c.head = c.value_head = 128; c.queries = 4; c.keys = 256;
        c.cache_rows = 300; c.heads = 8; c.kv_heads = 2;
        gpu_cases.push_back({ c, true });
    }
    for (const gpu_case& g : gpu_cases)
    {
        const attention_case& c = g.c;
        const case_data d = make_data(c, rng);
        const int offset = c.cache_rows > c.keys ? 2 : 0;
        const run_result r = run_case(gpu, c, d, offset, build_mode::guarded);
        const double diff = max_abs_diff(r.out, reference(c, d, offset));
        std::printf("gpu  %-42s kernel=%d path=%s max|diff| %.3g\n", c.name, r.bare_supported ? 1 : 0,
                    r.took_flash ? "flash" : "explicit", diff);
        check(diff < kTolerance, std::string(c.name) + ": result differs from the reference");
        check(r.took_flash == r.bare_supported, std::string(c.name) + ": guard chose the wrong path");
        check(r.fallbacks == (r.bare_supported ? 0u : 1u), std::string(c.name) + ": fallback count is wrong");
        if (is_cuda)
            check(r.bare_supported == g.cuda_has_kernel,
                  std::string(c.name) + ": ggml-cuda kernel availability changed (update this test and the guard notes)");
    }
    ggml_backend_free(gpu);

    if (g_failures != 0)
    {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("flash-attention guard: all checks passed\n");
    return 0;
}
