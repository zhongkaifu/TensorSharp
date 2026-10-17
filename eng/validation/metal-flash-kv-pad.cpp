// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Flash attention with keys padded to a multiple of 64, the pads masked and the
// queries chunked (ggml_ops_metal_flash_pad.h), against the same attention
// unpadded.
//
//   GgmlOpsMetalFlashKvPadTest <cpu|metal> --test
//       Correctness: the device-built mask holds 0 on every real key and -65504 on
//       every pad, and the chunked padded attention matches unpadded attention --
//       exactly on CPU (a masked key's weight underflows to 0 and the chunks map
//       back to the same rows), to FP16 rounding on Metal, where the padded call
//       runs the tensor-API kernel and the unpadded one the simdgroup kernel.
//       Exit 77 when the backend is not available.
//
//   GgmlOpsMetalFlashKvPadTest metal [keys] [heads] [repeats]
//       Timing, defaults 34991 keys (MiniMax-H3 at 1280x768x107) and 56 heads of
//       128, F16 K/V, F32 Q: the attention alone, no projections, no model.
#include "ggml_ops_metal_flash_pad.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef TSG_GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

int round_up_64(int n) { return (n + 63) / 64 * 64; }

struct Case
{
    ggml_context* ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor *q = nullptr, *k = nullptr, *v = nullptr;
    ggml_tensor *mask = nullptr, *unpadded = nullptr, *padded = nullptr;
    ggml_cgraph *graph_unpadded = nullptr, *graph_padded = nullptr;
    int keys = 0, padded_keys = 0, heads = 0, chunk = 0, head_dim = 128;

    Case(ggml_backend_t backend, int n, int h, int query_chunk, unsigned seed)
        : keys(n), padded_keys(round_up_64(n)), heads(h), chunk(query_chunk)
    {
        ctx = ggml_init({64u << 20, nullptr, true});
        require(ctx != nullptr, "context allocation failed");
        q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, keys, heads);
        // K/V hold the padded length; the unpadded attention reads a view of the first
        // `keys` rows, so both calls see byte-identical real keys.
        k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_dim, padded_keys, heads);
        v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_dim, padded_keys, heads);
        ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        ggml_tensor* kv = ggml_view_3d(ctx, k, head_dim, keys, heads, k->nb[1], k->nb[2], 0);
        ggml_tensor* vv = ggml_view_3d(ctx, v, head_dim, keys, heads, v->nb[1], v->nb[2], 0);
        unpadded = ggml_flash_attn_ext(ctx, q, kv, vv, nullptr, scale, 0.0f, 0.0f);
        require(ggml_backend_supports_op(backend, unpadded), "flash attention unsupported for this shape");
        mask = tsg_metal_flash_kv_pad_mask(ctx, keys, padded_keys, chunk);
        padded = tsg_metal_flash_attn_chunked(ctx, backend, q, k, v, mask, chunk, scale);
        require(padded != nullptr, "chunked padded flash attention unsupported for this shape");
        ggml_set_output(unpadded); ggml_set_output(padded); ggml_set_output(mask);
        graph_unpadded = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph_unpadded, unpadded);
        graph_padded = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph_padded, padded);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "buffer allocation failed");

        std::mt19937 rng(seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> host(static_cast<std::size_t>(head_dim) * keys * heads);
        for (float& x : host) x = normal(rng) * 1.5f;
        ggml_backend_tensor_set(q, host.data(), 0, ggml_nbytes(q));
        for (ggml_tensor* t : {k, v})
        {
            // Real rows random, pad rows zero, as ggml_pad leaves them in h3_attend.
            std::vector<ggml_fp16_t> half(static_cast<std::size_t>(head_dim) * padded_keys * heads);
            for (int hh = 0; hh < heads; hh++)
                for (int j = 0; j < padded_keys; j++)
                    for (int c = 0; c < head_dim; c++)
                        half[(static_cast<std::size_t>(hh) * padded_keys + j) * head_dim + c] =
                            ggml_fp32_to_fp16(j < keys ? normal(rng) * 1.5f : 0.0f);
            ggml_backend_tensor_set(t, half.data(), 0, ggml_nbytes(t));
        }
    }

    ~Case()
    {
        if (buffer != nullptr) ggml_backend_buffer_free(buffer);
        if (ctx != nullptr) ggml_free(ctx);
    }

    static double run(ggml_backend_t backend, ggml_cgraph* graph)
    {
        const auto t0 = std::chrono::steady_clock::now();
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
        ggml_backend_synchronize(backend);
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    std::vector<float> read(ggml_tensor* t) const
    {
        std::vector<float> out(static_cast<std::size_t>(ggml_nelements(t)));
        ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
        return out;
    }
};

ggml_backend_t open_backend(const std::string& name)
{
    if (name == "cpu") return ggml_backend_cpu_init();
#ifdef TSG_GGML_USE_METAL
    if (name == "metal") return ggml_backend_metal_init();
#endif
    return nullptr;
}

void check_mask(Case& c)
{
    std::vector<ggml_fp16_t> mask(static_cast<std::size_t>(c.padded_keys) * c.chunk);
    ggml_backend_tensor_get(c.mask, mask.data(), 0, ggml_nbytes(c.mask));
    for (int row = 0; row < c.chunk; row++)
        for (int j = 0; j < c.padded_keys; j++)
        {
            const float value = ggml_fp16_to_fp32(mask[static_cast<std::size_t>(row) * c.padded_keys + j]);
            const float want = j < c.keys ? 0.0f : -65504.0f;
            require(value == want, "mask row " + std::to_string(row) + " key " + std::to_string(j) +
                                       " is " + std::to_string(value) + ", want " + std::to_string(want));
        }
}

int test(ggml_backend_t backend, bool metal)
{
    struct Shape { int keys, heads, chunk; };
    // 425 and 252 are the H3 VAE-decoder and vision-tower lengths the oracle tests hit;
    // 756 is a 384x256x22 DiT; 2999 crosses many 64-key blocks; 2560 needs no padding.
    // Chunks of 64 and 128 rows, the latter ragged against 425 and 756.
    const Shape shapes[] = {{425, 4, 64}, {425, 4, 128}, {252, 8, 64}, {756, 6, 128},
                            {2999, 4, 64}, {64 * 40, 2, 64}};
    for (const Shape& shape : shapes)
    {
        Case c(backend, shape.keys, shape.heads, shape.chunk, 1234u + static_cast<unsigned>(shape.keys));
        if (c.padded_keys == c.keys)
        {
            // Already a multiple of 64: the mask is all zero and must change nothing.
            require(tsg_metal_flash_kv_length(backend, c.keys, c.keys) == c.keys, "aligned length was padded");
        }
        Case::run(backend, c.graph_unpadded);
        Case::run(backend, c.graph_padded);
        check_mask(c);
        const std::vector<float> a = c.read(c.unpadded), b = c.read(c.padded);
        double max_diff = 0, err = 0, ref = 0;
        for (std::size_t i = 0; i < a.size(); i++)
        {
            require(std::isfinite(b[i]), "padded output is not finite");
            const double d = static_cast<double>(a[i]) - b[i];
            max_diff = std::fmax(max_diff, std::fabs(d));
            err += d * d;
            ref += static_cast<double>(a[i]) * a[i];
        }
        const double rel = std::sqrt(err / ref);
        std::printf("keys %5d -> %5d, %d heads, chunks of %3d: max |diff| %.3e, rel-rms %.3e\n",
                    c.keys, c.padded_keys, c.heads, c.chunk, max_diff, rel);
        // CPU walks the same keys in the same order; a pad's weight is exactly 0.
        // Metal swaps kernels, so only FP16 rounding of the probabilities may differ.
        require(metal ? rel < 2e-3 : max_diff == 0.0, "padded attention differs from unpadded");
    }
    std::printf("predicted tensor API: %s; tsg_metal_flash_kv_length(34991) = %d\n",
                tsg_metal_tensor_api_likely(backend) ? "yes" : "no",
                tsg_metal_flash_kv_length(backend, 34991, 34991));
    return 0;
}

int bench(ggml_backend_t backend, int keys, int heads, int repeats)
{
    Case c(backend, keys, heads, tsg_metal_flash_query_chunk(keys, heads), 42u);
    Case::run(backend, c.graph_unpadded); // pipeline compilation
    Case::run(backend, c.graph_padded);
    double best_unpadded = 1e30, best_padded = 1e30;
    for (int r = 0; r < repeats; r++)
    {
        best_unpadded = std::fmin(best_unpadded, Case::run(backend, c.graph_unpadded));
        best_padded = std::fmin(best_padded, Case::run(backend, c.graph_padded));
    }
    const double flop = 4.0 * keys * static_cast<double>(keys) * c.head_dim * heads;
    std::printf("keys %d -> %d, %d heads x %d, queries in chunks of %d, best of %d "
                "(padded includes building its mask and the query pad/permute copies)\n",
                keys, c.padded_keys, heads, c.head_dim, c.chunk, repeats);
    std::printf("  unpadded %8.3f s  %6.2f TFLOP/s\n", best_unpadded, flop / best_unpadded / 1e12);
    std::printf("  padded   %8.3f s  %6.2f TFLOP/s  %.2fx\n", best_padded, flop / best_padded / 1e12,
                best_unpadded / best_padded);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const std::string name = argc > 1 ? argv[1] : "metal";
        ggml_backend_t backend = open_backend(name);
        if (backend == nullptr)
        {
            std::printf("backend '%s' unavailable; skipping\n", name.c_str());
            return 77;
        }
        int rc;
        if (argc > 2 && std::strcmp(argv[2], "--test") == 0)
            rc = test(backend, name == "metal");
        else
            rc = bench(backend, argc > 2 ? std::atoi(argv[2]) : 34991, argc > 3 ? std::atoi(argv[3]) : 56,
                       argc > 4 ? std::atoi(argv[4]) : 2);
        ggml_backend_free(backend);
        return rc;
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "FAILED: %s\n", ex.what());
        return 1;
    }
}
