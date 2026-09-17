// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Explicit performance control, never a CTest: what one DeepSeek V4.1 routed-
// expert layer costs per prefill chunk and per token at each ubatch width.
//
// The automatic prefill width (dsv4_ubatch_plan.h) rests on this: a resident
// routed-expert layer's chunk cost grows far slower than its token count, so a
// wider chunk is cheaper per token. Shapes are the Q4_K_M checkpoint's (384
// experts, 6 used, n_embd 5120, n_ff 2304, Q4_K gate/up/down). The expert
// blocks hold random bytes with bounded super-block scales: the kernels'
// arithmetic cost does not depend on the values, and no output is checked.
// Routing is uniform top-6, which touches more distinct experts per chunk than
// a trained router, so it errs towards a higher chunk cost at small widths.
//
//   GgmlOpsDsv4MoeWidthBench [--host THREADS] [--widths 256,512,1024] [--repeats N]
//
// Without --host it runs on CUDA device 0 (set CUDA_VISIBLE_DEVICES); with it,
// on the CPU backend with that many threads (a host-offloaded layer).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static constexpr int64_t N_EMBD = 5120, N_FF = 2304, N_EXPERT = 384, N_USED = 6;

static void fill_q4_k(ggml_tensor * t, std::mt19937_64 & rng)
{
    // block_q4_K: fp16 d, fp16 dmin, 12 bytes of 6-bit scales, 128 nibbles.
    const size_t block = ggml_type_size(GGML_TYPE_Q4_K);
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    for (auto & b : bytes) b = uint8_t(rng());
    const ggml_fp16_t d = ggml_fp32_to_fp16(1e-3f), dmin = ggml_fp32_to_fp16(5e-4f);
    for (size_t off = 0; off + block <= bytes.size(); off += block)
    {
        std::memcpy(&bytes[off], &d, 2);
        std::memcpy(&bytes[off + 2], &dmin, 2);
    }
    ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
}

int main(int argc, char ** argv)
{
    int host_threads = 0, repeats = 5;
    std::vector<int> widths = {256, 512, 1024};
    for (int i = 1; i < argc; ++i)
    {
        if (!std::strcmp(argv[i], "--host") && i + 1 < argc) host_threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = std::max(1, std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--widths") && i + 1 < argc)
        {
            widths.clear();
            std::string list = argv[++i];
            for (size_t p = 0; p < list.size();)
            {
                size_t q = list.find(',', p);
                widths.push_back(std::atoi(list.substr(p, q == std::string::npos ? q : q - p).c_str()));
                p = q == std::string::npos ? list.size() : q + 1;
            }
        }
        else { std::fprintf(stderr, "usage: %s [--host THREADS] [--widths a,b,c] [--repeats N]\n", argv[0]); return 2; }
    }

    ggml_backend_t backend = nullptr;
    if (host_threads > 0)
    {
        backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, host_threads);
    }
    else
    {
#ifdef TSG_GGML_USE_CUDA
        if (ggml_backend_cuda_get_device_count() == 0) { std::fprintf(stderr, "no CUDA device\n"); return 77; }
        backend = ggml_backend_cuda_init(0);
#else
        std::fprintf(stderr, "built without CUDA: pass --host THREADS\n");
        return 77;
#endif
    }
    if (!backend) { std::fprintf(stderr, "backend init failed\n"); return 1; }

    // Weights: one routed-expert layer.
    ggml_init_params wp = { 8 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * up = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_K, N_EMBD, N_FF, N_EXPERT);
    ggml_tensor * gate = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_K, N_EMBD, N_FF, N_EXPERT);
    ggml_tensor * down = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_K, N_FF, N_EMBD, N_EXPERT);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    if (!wbuf) { std::fprintf(stderr, "weight allocation failed\n"); return 1; }
    std::mt19937_64 rng(4109);
    for (ggml_tensor * t : { up, gate, down }) fill_q4_k(t, rng);

    std::printf("MOE_WIDTH_BENCH backend=%s threads=%d experts=%lld used=%lld n_embd=%lld n_ff=%lld type=q4_K weights_gib=%.2f\n",
        ggml_backend_name(backend), host_threads, (long long) N_EXPERT, (long long) N_USED, (long long) N_EMBD,
        (long long) N_FF, ggml_backend_buffer_get_size(wbuf) / double(1 << 30));

    for (int nt : widths)
    {
        ggml_init_params gp = { 64 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(gp);
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N_EMBD, 1, nt);
        ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, nt);
        ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, N_USED, nt);
        for (ggml_tensor * t : { x, ids, weights }) ggml_set_input(t);
        ggml_tensor * u = ggml_mul_mat_id(ctx, up, x, ids);
        ggml_tensor * g = ggml_mul_mat_id(ctx, gate, x, ids);
        u = ggml_clamp(ctx, u, -10.0f, 10.0f);
        g = ggml_clamp(ctx, g, -INFINITY, 10.0f);
        ggml_tensor * h = ggml_swiglu_split(ctx, g, u);
        ggml_tensor * experts = ggml_mul(ctx, ggml_mul_mat_id(ctx, down, h, ids), weights);
        ggml_tensor * out = nullptr;
        for (int64_t e = 0; e < N_USED; ++e)
        {
            ggml_tensor * v = ggml_view_2d(ctx, experts, N_EMBD, nt, experts->nb[2], e * experts->nb[1]);
            out = out ? ggml_add(ctx, out, v) : v;
        }
        ggml_set_output(out);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) { std::fprintf(stderr, "graph allocation failed at %d\n", nt); return 1; }

        std::normal_distribution<float> normal(0.0f, 1.0f);
        std::vector<float> xv((size_t) (N_EMBD * nt));
        for (auto & v : xv) v = normal(rng);
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        std::vector<int32_t> iv((size_t) (N_USED * nt));
        std::vector<int32_t> pool(N_EXPERT);
        for (int64_t i = 0; i < N_EXPERT; ++i) pool[(size_t) i] = int32_t(i);
        for (int t = 0; t < nt; ++t)
        {
            for (int64_t k = 0; k < N_USED; ++k)
                std::swap(pool[(size_t) k], pool[(size_t) (k + (int64_t) (rng() % uint64_t(N_EXPERT - k)))]);
            std::copy(pool.begin(), pool.begin() + N_USED, iv.begin() + t * N_USED);
        }
        ggml_backend_tensor_set(ids, iv.data(), 0, ggml_nbytes(ids));
        std::vector<float> wv((size_t) (N_USED * nt), 1.0f / N_USED);
        ggml_backend_tensor_set(weights, wv.data(), 0, ggml_nbytes(weights));

        for (int w = 0; w < 2; ++w) ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        std::vector<double> samples;
        for (int r = 0; r < repeats; ++r)
        {
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const double median = (sorted[(sorted.size() - 1) / 2] + sorted[sorted.size() / 2]) / 2;
        std::printf("MOE_WIDTH nt=%d median_ms=%.2f per_token_ms=%.4f compute_buffer_mib=%.1f samples_ms=[",
            nt, median, median / nt, ggml_gallocr_get_buffer_size(galloc, 0) / double(1 << 20));
        for (size_t i = 0; i < samples.size(); ++i) std::printf("%s%.2f", i ? "," : "", samples[i]);
        std::printf("]\n");
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(backend);
    return 0;
}
