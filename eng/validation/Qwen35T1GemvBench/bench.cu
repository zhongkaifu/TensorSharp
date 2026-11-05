// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// One-row (T = 1) projection microbenchmark for the NInfer Qwen3.8 shapes: the legacy A16 NVFP4
// kernel against ts_ni_gemv_nvfp4_*, the 8-row FP8 MMA tile against the ts_q35_fp8_gemv
// schedules, and ts_ni_add_norm against ts_ni_add_norm_5120 (bit-identity checked). Weights are
// random and rotated over 4 copies so no projection is served from the 96 MB L2. See README.md.
#include "../../../TensorSharp.Backends.Cuda/native/kernels/tensorsharp_small_t_mma.cuh"
#include "../../../TensorSharp.Backends.Cuda/native/ninfer/tensorsharp_q35_ninfer.cu"
#include "../../../TensorSharp.Backends.Cuda/native/ninfer/tensorsharp_q35_ninfer_a16_legacy_exact_vector.cu"

// The production instances (tensorsharp_q35_kernels.cu TS_Q35_FP8_SMALL / TS_Q35_FP8_GEMV), rebuilt
// here from the same templates so this file does not compile the whole Qwen3.5 kernel image.
#define BENCH_FP8_SMALL(NAME, OUT) \
    __global__ __launch_bounds__(128) void NAME(const uint8_t* __restrict__ w, const uint16_t* __restrict__ s, \
        const void* __restrict__ x, void* __restrict__ y, int n, int k, int m, int ldy) \
    { ts_smt::project<ts_smt::kFp8Row, false, false, OUT, 8, 4, 3, 1>(w, s, x, y, n, k, m, ldy); }
BENCH_FP8_SMALL(ts_q35_fp8_f32_t8_plain, 0)
BENCH_FP8_SMALL(ts_q35_fp8_f32_t8_round, 1)
BENCH_FP8_SMALL(ts_q35_fp8_f32_t8_swiglu, 2)
#define BENCH_FP8_GEMV(NAME, WARPS, RPW, VPL, OUT, MINB) \
    __global__ __launch_bounds__(WARPS * 32, MINB) void NAME(const uint8_t* __restrict__ w, \
        const uint16_t* __restrict__ s, const float* __restrict__ x, float* __restrict__ y, int k) \
    { ts_smt::gemv<WARPS, RPW, VPL, OUT>(w, s, x, y, k); }
BENCH_FP8_GEMV(ts_q35_fp8_gemv_b_plain, 4, 4, 16, 0, 1)
BENCH_FP8_GEMV(ts_q35_fp8_gemv_b_round, 4, 4, 16, 1, 1)
BENCH_FP8_GEMV(ts_q35_fp8_gemv_b_swiglu, 4, 4, 16, 2, 1)
BENCH_FP8_GEMV(ts_q35_fp8_gemv_c_plain, 8, 2, 16, 0, 2)
BENCH_FP8_GEMV(ts_q35_fp8_gemv_c_round, 8, 2, 16, 1, 2)
BENCH_FP8_GEMV(ts_q35_fp8_gemv_c_swiglu, 8, 2, 16, 2, 2)
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
static const int COPIES = 4;
template <class F> float timeit(F f, int iters = 100) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 10; ++i) f(i);
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i) f(i);
    cudaEventRecord(b); CK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b); return ms * 1000.0f / iters;
}
static float bfr(float x) { unsigned u; memcpy(&u, &x, 4); u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u; memcpy(&x, &u, 4); return x; }
static void* upload(const std::vector<uint8_t>& h) { void* d; CK(cudaMalloc(&d, h.size())); CK(cudaMemcpy(d, h.data(), h.size(), cudaMemcpyHostToDevice)); return d; }
static std::vector<float> fetch(const float* d, int n) { std::vector<float> h(n); CK(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost)); return h; }
static double maxrel(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0, s = 0; for (size_t i = 0; i < a.size(); ++i) s = fmax(s, fabs(a[i]));
    for (size_t i = 0; i < a.size(); ++i) m = fmax(m, fabs(a[i] - b[i]) / (s + 1e-30)); return m; }
int main() {
    srand(1);
    float* x; CK(cudaMalloc(&x, 17408 * 4));
    { std::vector<float> h(17408); for (auto& v : h) v = bfr((rand() / (float)RAND_MAX - 0.5f) * 2.0f); CK(cudaMemcpy(x, h.data(), h.size() * 4, cudaMemcpyHostToDevice)); }
    float *y1, *y2; CK(cudaMalloc(&y1, 248320 * 4)); CK(cudaMalloc(&y2, 248320 * 4));
    // NVFP4 shapes
    struct S4 { int n, k; } s4[] = { {34816, 5120}, {5120, 17408} };
    for (auto sh : s4) {
        std::vector<uint8_t> hc((size_t)sh.n * sh.k / 2), hs((size_t)sh.n * sh.k / 16);
        for (auto& b : hc) b = rand(); for (auto& b : hs) b = 0x30 + rand() % 16;
        uint8_t *c[COPIES], *s[COPIES];
        for (int i = 0; i < COPIES; ++i) { c[i] = (uint8_t*)upload(hc); s[i] = (uint8_t*)upload(hs); }
        double bytes = hc.size() + hs.size();
        float inv = 1.0f / 836.0f;
        auto legacy = [&](int i) { if (sh.k == 5120) ts_ni_project_a16_legacy_exact_k5120_t1<<<sh.n / 16, 256>>>(c[i % COPIES], s[i % COPIES], x, y1, sh.n, sh.k, 1, sh.k, sh.n, inv);
                                   else ts_ni_project_a16_legacy_exact_k17408_t1<<<sh.n / 16, 256>>>(c[i % COPIES], s[i % COPIES], x, y1, sh.n, sh.k, 1, sh.k, sh.n, inv); };
        auto gnew = [&](int i) { if (sh.k == 5120) ts_ni_gemv_nvfp4_k5120<<<sh.n / 16, 256>>>(c[i % COPIES], s[i % COPIES], x, y2, inv);
                                 else ts_ni_gemv_nvfp4_k17408<<<sh.n / 16, 256>>>(c[i % COPIES], s[i % COPIES], x, y2, inv); };
        float tl = timeit(legacy), tn = timeit(gnew);
        legacy(0); gnew(0); CK(cudaDeviceSynchronize());
        printf("nvfp4 n=%6d k=%5d legacy %7.1f us (%.2f TB/s)  new %7.1f us (%.2f TB/s)  maxrel %.2e\n", sh.n, sh.k, tl, bytes / tl / 1e6, tn, bytes / tn / 1e6, maxrel(fetch(y1, sh.n), fetch(y2, sh.n)));
        if (sh.k == 5120) { float t1 = timeit([&](int i) { ts_ni_gemv_nvfp4_k5120_swiglu<<<sh.n / 16, 256>>>(c[i % COPIES], s[i % COPIES], x, y2, inv); });
            printf("   swiglu %7.1f us (%.2f TB/s)\n", t1, bytes / t1 / 1e6); }
        for (int i = 0; i < COPIES; ++i) { cudaFree(c[i]); cudaFree(s[i]); }
    }
    // FP8 shapes
    struct S8 { int n, k, out; } s8[] = { {16384, 5120, 1}, {14336, 5120, 1}, {5120, 6144, 0}, {34816, 5120, 2}, {5120, 17408, 0}, {248320, 5120, 1} };
    for (auto sh : s8) {
        int copies = sh.n == 248320 ? 1 : COPIES;
        std::vector<uint8_t> hc((size_t)sh.n * sh.k), hs((size_t)sh.n * 2);
        for (auto& b : hc) { b = rand(); if ((b & 0x7f) == 0x7f) b = 0x30; }
        for (size_t i = 0; i < hs.size(); i += 2) { hs[i] = 0; hs[i + 1] = 0x3b; }
        uint8_t* c[COPIES]; uint16_t* s[COPIES];
        for (int i = 0; i < copies; ++i) { c[i] = (uint8_t*)upload(hc); s[i] = (uint16_t*)upload(hs); }
        double bytes = hc.size() + hs.size();
        int on = sh.out == 2 ? sh.n / 2 : sh.n;
        auto mma = [&](int i) { auto f = sh.out == 0 ? ts_q35_fp8_f32_t8_plain : sh.out == 1 ? ts_q35_fp8_f32_t8_round : ts_q35_fp8_f32_t8_swiglu;
            f<<<sh.n / 16, 128>>>(c[i % copies], s[i % copies], x, y1, sh.n, sh.k, 1, on); };
        auto gb = [&](int i) { auto f = sh.out == 0 ? ts_q35_fp8_gemv_b_plain : sh.out == 1 ? ts_q35_fp8_gemv_b_round : ts_q35_fp8_gemv_b_swiglu;
            f<<<sh.n / 16, 128>>>(c[i % copies], s[i % copies], x, y2, sh.k); };
        auto gc = [&](int i) { auto f = sh.out == 0 ? ts_q35_fp8_gemv_c_plain : sh.out == 1 ? ts_q35_fp8_gemv_c_round : ts_q35_fp8_gemv_c_swiglu;
            f<<<sh.n / 16, 256>>>(c[i % copies], s[i % copies], x, y2, sh.k); };
        float tm = timeit(mma), tb = timeit(gb), tc = timeit(gc);
        mma(0); gb(0); CK(cudaDeviceSynchronize()); double rb = maxrel(fetch(y1, on), fetch(y2, on));
        gc(0); CK(cudaDeviceSynchronize()); double rc = maxrel(fetch(y1, on), fetch(y2, on));
        printf("fp8 n=%6d k=%5d out%d mma %6.1f us (%.2f TB/s)  b %6.1f (%.2f) maxrel %.1e  c %6.1f (%.2f) maxrel %.1e\n", sh.n, sh.k, sh.out,
            tm, bytes / tm / 1e6, tb, bytes / tb / 1e6, rb, tc, bytes / tc / 1e6, rc);
        for (int i = 0; i < copies; ++i) { cudaFree(c[i]); cudaFree(s[i]); }
    }
    // add_norm
    { float *r, *d, *w, *o1, *o2, *r2; CK(cudaMalloc(&r, 5120 * 4)); CK(cudaMalloc(&r2, 5120 * 4)); CK(cudaMalloc(&d, 5120 * 4)); CK(cudaMalloc(&w, 5120 * 4)); CK(cudaMalloc(&o1, 5120 * 4)); CK(cudaMalloc(&o2, 5120 * 4));
      std::vector<float> h(5120); for (auto& v : h) v = bfr(rand() / (float)RAND_MAX); CK(cudaMemcpy(r, h.data(), 5120 * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(r2, h.data(), 5120 * 4, cudaMemcpyHostToDevice));
      for (auto& v : h) v = bfr(rand() / (float)RAND_MAX * 0.01f); CK(cudaMemcpy(d, h.data(), 5120 * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(w, h.data(), 5120 * 4, cudaMemcpyHostToDevice));
      ts_ni_add_norm<<<1, 256>>>(r, d, w, o1, 5120, 1e-6f); ts_ni_add_norm_5120<<<1, 256>>>(r2, d, w, o2, 5120, 1e-6f); CK(cudaDeviceSynchronize());
      auto a = fetch(o1, 5120), b = fetch(o2, 5120), ra = fetch(r, 5120), rb = fetch(r2, 5120); int diff = 0;
      for (int i = 0; i < 5120; ++i) diff += memcmp(&a[i], &b[i], 4) != 0 || memcmp(&ra[i], &rb[i], 4) != 0;
      float t1 = timeit([&](int) { ts_ni_add_norm<<<1, 256>>>(r, d, w, o1, 5120, 1e-6f); }, 1000);
      float t2 = timeit([&](int) { ts_ni_add_norm_5120<<<1, 256>>>(r2, d, w, o2, 5120, 1e-6f); }, 1000);
      printf("add_norm 5120: old %.2f us  new %.2f us  differing values %d\n", t1, t2, diff); }
    return 0;
}
