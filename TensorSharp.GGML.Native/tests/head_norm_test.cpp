// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// tsg::rms_norm_heads and tsg::gdn_l2_norm_heads (ggml_ops_head_norm.h), the
// per-head norms the Qwen 3.5 prefill graphs use. Their flattened
// [head_dim, n_heads * N] norms aborted ggml-vulkan on an Intel Arc 140V once a
// prefill step exceeded 2048 tokens (and, with that fixed, 4096):
//
//   ggml-vulkan-common.h:285: GGML_ASSERT(wg0 <= ...maxComputeWorkGroupCount[0] ...)
//
// ggml-vulkan dispatches RMS_NORM as one workgroup per ne01 row on the X axis and
// clamps only Y and Z, and the Arc reports maxComputeWorkGroupCount = 65536. The
// gated-delta-net output norm (128-wide, 32 value heads) asked for 32 * 2049 =
// 65568 X workgroups; the attention q norm and the gated-delta-net q/k L2 norm
// (an RMS_NORM underneath, 16 heads each) for 16 * 4097.
//
//   1. Structure (any backend, no device): for every column count up to and past
//      the largest prefill step, the norm's X axis (ne01) stays at n_heads. The
//      flattened form is asserted to cross the Arc's limit exactly where the
//      aborts started, so the check really tells the two apart.
//   2. Values: the result matches a double-precision reference and, on the CPU,
//      the flattened norm bit for bit (same memory, normed per row either way).
//   3. `vulkan`: the shapes that aborted run on the first Vulkan device and match
//      the reference. Exits 77 (skip) when no Vulkan device is present.
#include "ggml_ops_head_norm.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef TSG_GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace
{

enum class Norm
{
    Rms,          // rms_norm_heads with a weight
    RmsUnweighted,// rms_norm_heads without one
    GdnL2,        // gdn_l2_norm_heads
};

struct Case
{
    const char* name;
    std::int64_t head_dim;
    std::int64_t n_heads;
    Norm norm;
};

// Qwen 3.5 4B / 9B: gated-delta-net output (head_v_dim x value heads), attention
// q and k (head_dim x heads / kv heads), and the gated-delta-net q/k L2 norm
// (head_k_dim x key heads).
constexpr Case kGdnOut = { "gdn-out", 128, 32, Norm::Rms };
constexpr Case kQ = { "attn-q", 256, 16, Norm::Rms };
constexpr Case kK = { "attn-k", 256, 4, Norm::Rms };
constexpr Case kUnweighted = { "unweighted", 256, 4, Norm::RmsUnweighted };
constexpr Case kGdnQk = { "gdn-qk-l2", 128, 16, Norm::GdnL2 };
constexpr Case kAll[] = { kGdnOut, kQ, kK, kUnweighted, kGdnQk };
constexpr float kEps = 1e-6f;
// maxComputeWorkGroupCount[0] on the Intel Arc 140V (driver 101.9028).
constexpr std::int64_t kArcMaxWorkGroupCountX = 65536;

int g_checks = 0;

void require(bool condition, const char* what)
{
    ++g_checks;
    if (!condition)
    {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

ggml_tensor* build_new(ggml_context* ctx, ggml_tensor* x, const Case& c, std::int64_t n_cols, ggml_tensor* w)
{
    if (c.norm == Norm::GdnL2)
        return tsg::gdn_l2_norm_heads(ctx, x, c.head_dim, c.n_heads, n_cols, kEps);
    return tsg::rms_norm_heads(ctx, x, c.head_dim, c.n_heads, n_cols, w, kEps);
}

// The flattened form the Qwen 3.5 prefill graphs used before the fix.
ggml_tensor* build_legacy(ggml_context* ctx, ggml_tensor* x, const Case& c, std::int64_t n_cols, ggml_tensor* w)
{
    ggml_tensor* flat = ggml_reshape_2d(ctx, x, c.head_dim, c.n_heads * n_cols);
    ggml_tensor* y = c.norm == Norm::GdnL2 ? tsg::build_gdn_l2_norm(ctx, flat, kEps) : ggml_rms_norm(ctx, flat, kEps);
    if (w != nullptr) y = ggml_mul(ctx, y, w);
    return ggml_reshape_3d(ctx, y, c.head_dim, c.n_heads, n_cols);
}

using Builder = ggml_tensor* (*)(ggml_context*, ggml_tensor*, const Case&, std::int64_t, ggml_tensor*);

const ggml_tensor* find_rms_norm(ggml_tensor* out)
{
    const ggml_tensor* t = out;
    while (t != nullptr && t->op != GGML_OP_RMS_NORM)
        t = t->src[0];
    return t;
}

// X-axis row count (ne01) of the RMS_NORM the builder asks a backend to dispatch.
std::int64_t norm_rows(Builder build, const Case& c, std::int64_t n_cols)
{
    ggml_init_params params = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(params);
    require(ctx != nullptr, "metadata context");
    // The callers hand in a [head_dim * n_heads, n_cols] projection slice.
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.head_dim * c.n_heads, n_cols);
    ggml_tensor* w = c.norm == Norm::Rms ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.head_dim) : nullptr;
    ggml_tensor* out = build(ctx, x, c, n_cols, w);
    require(out->ne[0] == c.head_dim && out->ne[1] == c.n_heads && out->ne[2] == n_cols,
            "result is [head_dim, n_heads, n_cols]");
    const ggml_tensor* norm = find_rms_norm(out);
    require(norm != nullptr && norm->ne[0] == c.head_dim, "norm runs over head_dim");
    const std::int64_t rows = norm->ne[1];
    ggml_free(ctx);
    return rows;
}

void check_structure()
{
    // 2048 / 2049 bracket the first abort; 4096 / 4097 the 16-head ones; 8192 is
    // the solo prefill step a raised TS_SCHED_MAX_BATCHED_TOKENS allows.
    for (const Case& c : kAll)
        for (std::int64_t n : { 1, 2048, 2049, 4096, 4097, 8192, 65536 })
            require(norm_rows(build_new, c, n) == c.n_heads, "norm's X axis is n_heads at every N");

    require(norm_rows(build_legacy, kGdnOut, 2048) <= kArcMaxWorkGroupCountX, "flattened gdn-out norm fits at N = 2048");
    require(norm_rows(build_legacy, kGdnOut, 2049) > kArcMaxWorkGroupCountX,
            "flattened gdn-out norm overflows at N = 2049, so the structural check distinguishes the fix");
    for (const Case& c : { kQ, kGdnQk })
    {
        require(norm_rows(build_legacy, c, 4096) <= kArcMaxWorkGroupCountX, "flattened 16-head norm fits at N = 4096");
        require(norm_rows(build_legacy, c, 4097) > kArcMaxWorkGroupCountX, "flattened 16-head norm overflows at N = 4097");
    }
}

std::vector<float> random_values(std::size_t count, unsigned seed, float lo, float hi)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(count);
    for (float& x : v) x = dist(rng);
    return v;
}

std::vector<float> reference(const std::vector<float>& x, const std::vector<float>* w, const Case& c)
{
    std::vector<float> y(x.size());
    const std::size_t d = static_cast<std::size_t>(c.head_dim);
    for (std::size_t r = 0; r < x.size() / d; ++r)
    {
        const float* in = &x[r * d];
        double sum = 0.0;
        for (std::size_t i = 0; i < d; ++i) sum += static_cast<double>(in[i]) * in[i];
        // RMSNorm: x / sqrt(mean(x^2) + eps). GDN L2: x / sqrt(sum(x^2) + eps).
        const double scale = c.norm == Norm::GdnL2 ? 1.0 / std::sqrt(sum + kEps)
                                                   : 1.0 / std::sqrt(sum / static_cast<double>(d) + kEps);
        for (std::size_t i = 0; i < d; ++i)
            y[r * d + i] = static_cast<float>(in[i] * scale * (w != nullptr ? (*w)[i] : 1.0f));
    }
    return y;
}

std::vector<float> run(ggml_backend_t backend, Builder build, const Case& c, std::int64_t n_cols,
                       const std::vector<float>& x, const std::vector<float>* w)
{
    ggml_init_params params = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(params);
    require(ctx != nullptr, "graph context");
    ggml_tensor* in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.head_dim * c.n_heads, n_cols);
    ggml_tensor* norm_w = w != nullptr ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, c.head_dim) : nullptr;
    ggml_tensor* out = build(ctx, in, c, n_cols, norm_w);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "allocate graph tensors");
    ggml_backend_tensor_set(in, x.data(), 0, x.size() * sizeof(float));
    if (norm_w != nullptr) ggml_backend_tensor_set(norm_w, w->data(), 0, w->size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute");

    std::vector<float> y(x.size());
    ggml_backend_tensor_get(out, y.data(), 0, y.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return y;
}

void check_values(ggml_backend_t backend, const Case& c, std::int64_t n_cols, bool compare_legacy)
{
    const std::size_t count = static_cast<std::size_t>(c.head_dim * c.n_heads) * static_cast<std::size_t>(n_cols);
    const std::vector<float> x = random_values(count, 1234u + static_cast<unsigned>(n_cols * c.n_heads), -3.0f, 3.0f);
    const std::vector<float> w_values = random_values(static_cast<std::size_t>(c.head_dim), 99u, 0.25f, 1.75f);
    const std::vector<float>* w = c.norm == Norm::Rms ? &w_values : nullptr;
    const std::vector<float> expected = reference(x, w, c);
    const std::vector<float> actual = run(backend, build_new, c, n_cols, x, w);

    double worst = 0.0;
    for (std::size_t i = 0; i < count; ++i)
        worst = std::fmax(worst, std::fabs(static_cast<double>(actual[i]) - expected[i]));
    std::printf("  %s %s N=%lld: max |err| vs double reference = %.3g\n", ggml_backend_name(backend), c.name,
                static_cast<long long>(n_cols), worst);
    require(worst < 1e-4, "3-D norm matches the double-precision reference");

    if (compare_legacy)
    {
        const std::vector<float> legacy = run(backend, build_legacy, c, n_cols, x, w);
        require(std::memcmp(legacy.data(), actual.data(), count * sizeof(float)) == 0,
                "3-D norm is bit-identical to the flattened norm it replaces");
    }
}

} // namespace

int main(int argc, char** argv)
{
    const char* mode = argc > 1 ? argv[1] : "cpu";

    check_structure();

    if (std::strcmp(mode, "cpu") == 0)
    {
        ggml_backend_t cpu = ggml_backend_cpu_init();
        require(cpu != nullptr, "CPU backend");
        for (const Case& c : kAll)
            for (std::int64_t n : { 1, 37, 129 })
                check_values(cpu, c, n, /*compare_legacy=*/true);
        ggml_backend_free(cpu);
    }
    else if (std::strcmp(mode, "vulkan") == 0)
    {
#ifdef TSG_GGML_USE_VULKAN
        if (ggml_backend_vk_get_device_count() < 1)
        {
            std::printf("SKIP: no Vulkan device\n");
            return 77;
        }
        char description[256] = {};
        ggml_backend_vk_get_device_description(0, description, sizeof(description));
        std::printf("  Vulkan device 0: %s\n", description);
        ggml_backend_t vk = ggml_backend_vk_init(0);
        require(vk != nullptr, "Vulkan backend");
        // The step sizes whose flattened norms needed more than 65536 X workgroups.
        // The flattened form is not run here: on such a device it aborts the process.
        check_values(vk, kGdnOut, 2049, /*compare_legacy=*/false);
        check_values(vk, kGdnOut, 8192, /*compare_legacy=*/false);
        check_values(vk, kQ, 4097, /*compare_legacy=*/false);
        check_values(vk, kK, 8192, /*compare_legacy=*/false);
        check_values(vk, kGdnQk, 4097, /*compare_legacy=*/false);
        check_values(vk, kGdnQk, 8192, /*compare_legacy=*/false);
        ggml_backend_free(vk);
#else
        std::printf("SKIP: built without ggml-vulkan\n");
        return 77;
#endif
    }
    else
    {
        std::fprintf(stderr, "usage: %s [cpu|vulkan]\n", argv[0]);
        return 2;
    }

    std::printf("head norm: %d checks passed\n", g_checks);
    return 0;
}
