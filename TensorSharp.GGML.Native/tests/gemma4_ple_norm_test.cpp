// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Gemma 4's per-layer-embedding projection norm (gemma4_ple_norm.h), the step of
// TSGgml_Gemma4ModelVerify that aborted ggml-vulkan on an Intel Arc 140V once a
// gemma-4-E4B prefill step exceeded 1536 tokens:
//
//   ggml-vulkan-common.h:285: GGML_ASSERT(wg0 <= ...maxComputeWorkGroupCount[0] ...)
//
// ggml-vulkan dispatches RMS_NORM as one workgroup per ne01 row on the X axis and
// clamps only Y and Z, and the Arc reports maxComputeWorkGroupCount = 65536. The
// old graph normed a flattened [ple_dim, num_layers * N] view, so 42 layers asked
// for 42 * N X workgroups (67200 at the Vulkan graph's 64-padded N = 1600).
//
//   1. Structure (any backend, no device): for every column count up to and past
//      the largest prefill step, the norm's X axis (ne01) stays at num_layers,
//      under the 65535 every Vulkan device must support. The same check on the
//      old flattened builder is asserted to FAIL, so it really tells them apart.
//   2. Values: the result matches a double-precision reference and, on the CPU,
//      the old flattened builder bit for bit (it is the same memory, normed per
//      row either way).
//   3. `vulkan`: the shapes that aborted run on the first Vulkan device and match
//      the reference. Exits 77 (skip) when no Vulkan device is present.
#include "gemma4_ple_norm.h"

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

// gemma-4-E4B: 42 layers of 256-wide per-layer embeddings.
constexpr int kPleDim = 256;
constexpr int kLayers = 42;
constexpr float kEps = 1e-6f;
// The smallest maxComputeWorkGroupCount any Vulkan implementation may report.
constexpr std::int64_t kVulkanMinWorkGroupCount = 65535;

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

// The builder TSGgml_Gemma4ModelVerify / ModelDecodeBatched used before the fix.
ggml_tensor* legacy_proj_norm(ggml_context* ctx, ggml_tensor* proj, int ple_dim, int num_layers,
                              ggml_tensor* norm_w, float eps)
{
    const std::int64_t n_cols = proj->ne[1];
    ggml_tensor* rows = ggml_reshape_2d(ctx, ggml_cont(ctx, proj), ple_dim, static_cast<std::int64_t>(num_layers) * n_cols);
    rows = ggml_mul(ctx, ggml_rms_norm(ctx, rows, eps), norm_w);
    return ggml_reshape_2d(ctx, rows, static_cast<std::int64_t>(ple_dim) * num_layers, n_cols);
}

using Builder = ggml_tensor* (*)(ggml_context*, ggml_tensor*, int, int, ggml_tensor*, float);

// The single RMS_NORM node `out` depends on.
const ggml_tensor* find_rms_norm(ggml_tensor* out)
{
    const ggml_tensor* t = out;
    while (t != nullptr && t->op != GGML_OP_RMS_NORM)
        t = t->src[0];
    return t;
}

// Largest X-axis row count the norm asks a backend to dispatch.
std::int64_t norm_rows(Builder build, std::int64_t n_cols)
{
    ggml_init_params params = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(params);
    require(ctx != nullptr, "metadata context");
    ggml_tensor* proj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, static_cast<std::int64_t>(kPleDim) * kLayers, n_cols);
    ggml_tensor* w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kPleDim);
    ggml_tensor* out = build(ctx, proj, kPleDim, kLayers, w, kEps);
    require(out->ne[0] == static_cast<std::int64_t>(kPleDim) * kLayers && out->ne[1] == n_cols && out->ne[2] == 1,
            "result keeps the [num_layers * ple_dim, n_cols] shape");
    const ggml_tensor* norm = find_rms_norm(out);
    require(norm != nullptr && norm->ne[0] == kPleDim, "norm runs over ple_dim");
    const std::int64_t rows = norm->ne[1];
    ggml_free(ctx);
    return rows;
}

void check_structure()
{
    // 1536 passed and 1537 (padded to 1600 on Vulkan) aborted; 1561 is the first
    // unpadded count past 65536 / 42; 8192 is the default solo prefill step.
    for (std::int64_t n : { 1, 64, 1536, 1561, 1600, 8192, 65536 })
        require(norm_rows(tsg_gemma4_ple::proj_norm, n) == kLayers, "norm's X axis is num_layers at every N");

    require(norm_rows(legacy_proj_norm, 1536) <= kVulkanMinWorkGroupCount, "legacy builder fits at N = 1536");
    require(norm_rows(legacy_proj_norm, 1561) > kVulkanMinWorkGroupCount,
            "legacy builder overflows at N = 1561, so the structural check distinguishes the fix");
}

std::vector<float> random_values(std::size_t count, unsigned seed, float lo, float hi)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> v(count);
    for (float& x : v) x = dist(rng);
    return v;
}

std::vector<float> reference(const std::vector<float>& x, const std::vector<float>& w, std::int64_t n_cols)
{
    std::vector<float> y(x.size());
    const std::int64_t rows = static_cast<std::int64_t>(kLayers) * n_cols;
    for (std::int64_t r = 0; r < rows; ++r)
    {
        const float* in = &x[static_cast<std::size_t>(r) * kPleDim];
        double sum = 0.0;
        for (int i = 0; i < kPleDim; ++i) sum += static_cast<double>(in[i]) * in[i];
        const double scale = 1.0 / std::sqrt(sum / kPleDim + kEps);
        for (int i = 0; i < kPleDim; ++i)
            y[static_cast<std::size_t>(r) * kPleDim + i] = static_cast<float>(in[i] * scale * w[i]);
    }
    return y;
}

std::vector<float> run(ggml_backend_t backend, Builder build, std::int64_t n_cols,
                       const std::vector<float>& x, const std::vector<float>& w)
{
    ggml_init_params params = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context* ctx = ggml_init(params);
    require(ctx != nullptr, "graph context");
    ggml_tensor* proj = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, static_cast<std::int64_t>(kPleDim) * kLayers, n_cols);
    ggml_tensor* norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kPleDim);
    ggml_tensor* out = build(ctx, proj, kPleDim, kLayers, norm_w, kEps);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "allocate graph tensors");
    ggml_backend_tensor_set(proj, x.data(), 0, x.size() * sizeof(float));
    ggml_backend_tensor_set(norm_w, w.data(), 0, w.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute");

    std::vector<float> y(x.size());
    ggml_backend_tensor_get(out, y.data(), 0, y.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return y;
}

void check_values(ggml_backend_t backend, std::int64_t n_cols, bool compare_legacy)
{
    const std::size_t count = static_cast<std::size_t>(kPleDim) * kLayers * static_cast<std::size_t>(n_cols);
    const std::vector<float> x = random_values(count, 1234u + static_cast<unsigned>(n_cols), -3.0f, 3.0f);
    const std::vector<float> w = random_values(kPleDim, 99u, 0.25f, 1.75f);
    const std::vector<float> expected = reference(x, w, n_cols);
    const std::vector<float> actual = run(backend, tsg_gemma4_ple::proj_norm, n_cols, x, w);

    double worst = 0.0;
    for (std::size_t i = 0; i < count; ++i)
        worst = std::fmax(worst, std::fabs(static_cast<double>(actual[i]) - expected[i]));
    std::printf("  %s N=%lld: max |err| vs double reference = %.3g\n",
                ggml_backend_name(backend), static_cast<long long>(n_cols), worst);
    require(worst < 1e-4, "3-D norm matches the double-precision reference");

    if (compare_legacy)
    {
        const std::vector<float> legacy = run(backend, legacy_proj_norm, n_cols, x, w);
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
        for (std::int64_t n : { 1, 37, 129 })
            check_values(cpu, n, /*compare_legacy=*/true);
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
        // 1561 is the first N whose flattened norm needed more than 65536 X
        // workgroups; 1600 is the padded step that aborted on the Arc 140V. The
        // flattened builder is not run here: on such a device it aborts the process.
        for (std::int64_t n : { 64, 1561, 1600 })
            check_values(vk, n, /*compare_legacy=*/false);
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

    std::printf("gemma4 PLE norm: %d checks passed\n", g_checks);
    return 0;
}
