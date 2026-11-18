// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Per-op time of llama.cpp's graphs, the llama.cpp side of TensorSharp's TS_GGML_NODE_PROFILE (ggml_ops_core.cpp's
// graph_compute_profiled): the scheduler's eval callback asks for every node, so each node runs as its own compute and
// is timed from the ask to its result (a synchronize per node, as TensorSharp's profiler does). One prompt of PROMPT
// pseudo-random tokens (the ids Lfm3Probe bench feeds), then DECODE single-token steps; first unprofiled (tokens/s),
// then profiled, the time per op summed per phase ("prompt": graphs of more than one row, "decode": one row), MUL_MAT
// keyed by the weight's type and shape. TensorSharp's per-op timings compare with these op for op.
//
// Build against a llama.cpp build tree (the sources stay unchanged), e.g. the CPU-only one:
//   L=~/work/llama.cpp-lfm3; B=$L/build-cpu-noblas
//   c++ -std=c++17 -O2 -Wno-deprecated-declarations eng/lfm3/bench/llamacpp-op-profile.cpp -I$L/include -I$L/ggml/include \
//       -L$B/bin -lllama -lggml -lggml-base -Wl,-rpath,$B/bin -o artifacts/lfm3/llamacpp-op-profile-noblas
// Run:
//   llamacpp-op-profile MODEL THREADS FA PROMPT DECODE [NGL]
//   (FA 0/1 is flash attention; NGL defaults to 0; n_batch 2048 and n_ubatch 512 as llama-bench)
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct Bucket { double us = 0; long long calls = 0; };

struct Profile {
    bool on = false;
    bool decode = false;
    clock_type::time_point asked;
    std::map<std::string, Bucket> prompt, decode_ops;
    double prompt_us = 0, decode_us = 0;
    long long prompt_graphs = 0, decode_graphs = 0;
};

Profile g;

std::string key_of(const ggml_tensor * t) {
    std::string key = ggml_op_desc(t);
    if ((t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) && t->src[0] != nullptr) {
        key += std::string(":") + ggml_type_name(t->src[0]->type) + ":" + std::to_string(t->src[0]->ne[0]) + "x" +
               std::to_string(t->src[0]->ne[1]);
        if (t->src[0]->buffer != nullptr) key += std::string("@") + ggml_backend_buffer_name(t->src[0]->buffer);
    }
    return key;
}

bool eval_cb(ggml_tensor * t, bool ask, void *) {
    if (!g.on) return false;
    if (ask) { g.asked = clock_type::now(); return true; }
    const double us = std::chrono::duration<double, std::micro>(clock_type::now() - g.asked).count();
    Bucket & b = (g.decode ? g.decode_ops : g.prompt)[key_of(t)];
    b.us += us;
    b.calls++;
    (g.decode ? g.decode_us : g.prompt_us) += us;
    return true;
}

std::vector<llama_token> tokens(int n, int seed) {
    std::vector<llama_token> t(n);
    for (int i = 0; i < n; i++) t[i] = 1000 + (llama_token) ((uint32_t) (i * 2654435761u + seed) % 60000u);
    return t;
}

void print(const char * phase, const std::map<std::string, Bucket> & ops, double total_us, long long steps, const char * unit) {
    std::vector<std::pair<std::string, Bucket>> v(ops.begin(), ops.end());
    std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.second.us > b.second.us; });
    printf("[op-profile] %s: %.3f ms per %s over %lld (profiled, a synchronize per node)\n", phase,
           total_us / 1000.0 / std::max(1LL, steps), unit, steps);
    for (const auto & [k, b] : v) {
        printf("    %-48s %9.3f ms/%s %6.1f%% %6lld nodes/%s\n", k.c_str(), b.us / 1000.0 / std::max(1LL, steps), unit,
               100.0 * b.us / std::max(1.0, total_us), b.calls / std::max(1LL, steps), unit);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s MODEL THREADS FA PROMPT DECODE [NGL]\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];
    const int threads = atoi(argv[2]), fa = atoi(argv[3]), prompt = atoi(argv[4]), decode = atoi(argv[5]);
    const int ngl = argc > 6 ? atoi(argv[6]) : 0;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", path); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = prompt + decode + 16;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.flash_attn_type = fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    // The sliding-window layers in their own small cache, as llama-bench and llama-server run them (the library default
    // keeps a full-size SWA cache, whose attention reads every key).
    cp.swa_full = false;
    cp.cb_eval = eval_cb;
    cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "cannot create a context\n"); return 1; }

    auto run = [&](int seed, double & prompt_s, double & decode_s) {
        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<llama_token> p = tokens(prompt, seed), d = tokens(decode, seed + 11);
        g.decode = false;
        auto t0 = clock_type::now();
        for (int at = 0; at < prompt; at += 2048) {
            const int n = std::min(2048, prompt - at);
            if (llama_decode(ctx, llama_batch_get_one(p.data() + at, n)) != 0) { fprintf(stderr, "prompt decode failed\n"); exit(1); }
        }
        llama_synchronize(ctx);
        auto t1 = clock_type::now();
        g.decode = true;
        for (int i = 0; i < decode; i++) {
            if (llama_decode(ctx, llama_batch_get_one(&d[i], 1)) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
            llama_get_logits_ith(ctx, -1);
        }
        llama_synchronize(ctx);
        auto t2 = clock_type::now();
        prompt_s = std::chrono::duration<double>(t1 - t0).count();
        decode_s = std::chrono::duration<double>(t2 - t1).count();
    };

    double ps, ds;
    run(1, ps, ds);   // warm
    run(7, ps, ds);
    printf("unprofiled: prompt %d tokens %.1f tok/s, decode %d tokens %.2f tok/s (threads %d, fa %d)\n", prompt,
           prompt / ps, decode, decode / ds, threads, fa);
    g.on = true;
    run(7, ps, ds);
    g.on = false;
    printf("profiled:   prompt %.3f s, decode %.3f s\n", ps, ds);
    const long long ubatches = (prompt + 511) / 512;
    print("prompt", g.prompt, g.prompt_us, ubatches, "ubatch");
    print("decode", g.decode_ops, g.decode_us, decode, "token");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
