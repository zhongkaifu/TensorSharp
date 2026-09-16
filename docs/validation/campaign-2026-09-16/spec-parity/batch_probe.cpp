// Batch-invariance probe for ggml-cuda: does a row computed in an N-row launch
// equal the same row computed alone? mul_mat (Q8_0, F16, BF16) and flash_attn_ext.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <cstdlib>

static ggml_backend_t g_backend;
static std::mt19937 rng(1234);

static std::vector<float> randn(size_t n, float scale = 1.0f) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto & x : v) x = d(rng);
    return v;
}

struct Graph {
    ggml_context * ctx;
    ggml_backend_buffer_t buf = nullptr;
    Graph() { ctx = ggml_init({256 * 1024 * 1024, nullptr, true}); }
    ~Graph() { if (buf) ggml_backend_buffer_free(buf); ggml_free(ctx); }
    void alloc() { buf = ggml_backend_alloc_ctx_tensors(ctx, g_backend); }
    std::vector<float> run(ggml_tensor * out) {
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
        ggml_gallocr_alloc_graph(ga, gf);
        ggml_backend_graph_compute(g_backend, gf);
        std::vector<float> r(ggml_nelements(out));
        ggml_backend_tensor_get(out, r.data(), 0, ggml_nbytes(out));
        ggml_gallocr_free(ga);
        return r;
    }
};

static std::vector<uint8_t> quantize(ggml_type t, const std::vector<float> & w, int64_t ne0, int64_t ne1) {
    std::vector<uint8_t> q(ggml_row_size(t, ne0) * ne1);
    if (t == GGML_TYPE_F16) { for (size_t i = 0; i < w.size(); i++) { ggml_fp16_t h = ggml_fp32_to_fp16(w[i]); memcpy(q.data() + 2 * i, &h, 2); } return q; }
    if (t == GGML_TYPE_BF16) { ggml_fp32_to_bf16_row(w.data(), (ggml_bf16_t *) q.data(), (int64_t) w.size()); return q; }
    ggml_quantize_chunk(t, w.data(), q.data(), 0, ne1, ne0, nullptr);
    return q;
}

static std::vector<float> mm_run(ggml_type t, const std::vector<uint8_t> & qw, int64_t in, int64_t out, const float * x, int N) {
    Graph g;
    ggml_tensor * W = ggml_new_tensor_2d(g.ctx, t, in, out);
    ggml_tensor * X = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, in, N);
    ggml_tensor * Y = ggml_mul_mat(g.ctx, W, X);
    g.alloc();
    ggml_backend_tensor_set(W, qw.data(), 0, qw.size());
    ggml_backend_tensor_set(X, x, 0, in * N * sizeof(float));
    return g.run(Y);
}

static void probe_mul_mat(ggml_type t, int64_t in, int64_t out) {
    auto w = randn(in * out, 0.02f);
    std::vector<uint8_t> qw;
    if (t == GGML_TYPE_F32) { qw.resize(w.size() * 4); memcpy(qw.data(), w.data(), qw.size()); }
    else qw = quantize(t, w, in, out);
    int maxN = 8;
    auto x = randn(in * maxN, 1.0f);
    std::vector<std::vector<float>> single(maxN);
    for (int r = 0; r < maxN; r++) single[r] = mm_run(t, qw, in, out, &x[r * in], 1);
    for (int N : {2, 3, 4, 5, 8}) {
        auto y = mm_run(t, qw, in, out, x.data(), N);
        printf("mul_mat %s %lldx%lld rows=%d: per-row max|diff| vs 1-row:", ggml_type_name(t), (long long) in, (long long) out, N);
        for (int r = 0; r < N; r++) {
            double m = 0;
            for (int64_t i = 0; i < out; i++) m = std::max(m, (double) std::fabs(y[r * out + i] - single[r][i]));
            printf(" %.2g", m);
        }
        printf("\n");
    }
}

static void fill_mask(std::vector<float> & m, int L, int NQ, int valid0) {
    m.assign((size_t) L * NQ, -INFINITY);
    for (int q = 0; q < NQ; q++) for (int k = 0; k < std::min(L, valid0 + q); k++) m[(size_t) q * L + k] = 0.0f;
}

static void probe_fa(int hd, int heads, int kvh, int L, int valid0, bool f32prec) {
    int maxN = 8;
    auto q = randn(hd * heads * maxN, 1.0f);
    auto k = randn(hd * kvh * L, 1.0f);
    auto v = randn(hd * kvh * L, 1.0f);
    std::vector<ggml_fp16_t> k16(k.size()), v16(v.size());
    for (size_t i = 0; i < k.size(); i++) { k16[i] = ggml_fp32_to_fp16(k[i]); v16[i] = ggml_fp32_to_fp16(v[i]); }
    std::vector<std::vector<float>> single(maxN);
    auto runN = [&](int N, int qoff, int Ly) {
        Graph g;
        ggml_tensor * Q = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, hd, N, heads);
        ggml_tensor * K = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, hd, Ly, kvh);
        ggml_tensor * V = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, hd, Ly, kvh);
        ggml_tensor * M = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, Ly, N);
        ggml_tensor * A = ggml_flash_attn_ext(g.ctx, Q, K, V, M, 1.0f, 0.0f, 0.0f);
        if (f32prec) ggml_flash_attn_ext_set_prec(A, GGML_PREC_F32);
        g.alloc();
        // Q laid out [hd, N, heads]: take query rows qoff..qoff+N-1
        std::vector<float> qs((size_t) hd * N * heads);
        for (int h = 0; h < heads; h++) for (int n = 0; n < N; n++)
            memcpy(&qs[(size_t) (h * N + n) * hd], &q[(size_t) (h * maxN + qoff + n) * hd], hd * sizeof(float));
        ggml_backend_tensor_set(Q, qs.data(), 0, qs.size() * sizeof(float));
        std::vector<ggml_fp16_t> ks((size_t) hd * Ly * kvh), vs(ks.size());
        for (int h = 0; h < kvh; h++) { memcpy(&ks[(size_t) h * Ly * hd], &k16[(size_t) h * L * hd], (size_t) Ly * hd * 2); memcpy(&vs[(size_t) h * Ly * hd], &v16[(size_t) h * L * hd], (size_t) Ly * hd * 2); }
        ggml_backend_tensor_set(K, ks.data(), 0, ks.size() * 2);
        ggml_backend_tensor_set(V, vs.data(), 0, vs.size() * 2);
        std::vector<float> mf; fill_mask(mf, Ly, N, valid0 + qoff);
        std::vector<ggml_fp16_t> m16(mf.size());
        for (size_t i = 0; i < mf.size(); i++) m16[i] = ggml_fp32_to_fp16(mf[i]);
        ggml_backend_tensor_set(M, m16.data(), 0, m16.size() * 2);
        return g.run(A);   // [hd, heads, N]
    };
    for (int i = 0; i < maxN; i++) single[i] = runN(1, i, L);
    for (int N : {2, 4, 8}) {
        auto r = runN(N, 0, L);
        double m = 0;
        for (int n = 0; n < N; n++) for (int j = 0; j < hd * heads; j++)
            m = std::max(m, (double) std::fabs(r[(size_t) n * hd * heads + j] - single[n][j]));
        printf("flash_attn hd=%d heads=%d kv=%d L=%d prec=%s queries=%d: vs 1-query max|diff|=%.3g\n", hd, heads, kvh, L, f32prec ? "f32" : "def", N, m);
    }
    // Same single query over an unpadded key length (valid0 keys only).
    if (valid0 < L) {
        auto r = runN(1, 0, valid0);
        double m = 0;
        for (int j = 0; j < hd * heads; j++) m = std::max(m, (double) std::fabs(r[j] - single[0][j]));
        printf("flash_attn hd=%d L=%d vs L=%d (1 query, same valid keys): max|diff|=%.3g\n", hd, valid0, L, m);
    }
}

int main() {
    g_backend = ggml_backend_cuda_init(0);
    if (!g_backend) { fprintf(stderr, "no cuda\n"); return 1; }
    probe_mul_mat(GGML_TYPE_Q8_0, 2560, 10240);
    probe_mul_mat(GGML_TYPE_Q8_0, 2560, 256);
    probe_mul_mat(GGML_TYPE_Q8_0, 256, 2560);
    probe_mul_mat(GGML_TYPE_F32, 2560, 10752);
    probe_mul_mat(GGML_TYPE_BF16, 2560, 10752);
    probe_mul_mat(GGML_TYPE_Q8_0, 2560, 262144);
    probe_fa(256, 8, 2, 256, 70, true);
    probe_fa(512, 8, 2, 256, 70, true);
    ggml_backend_free(g_backend);
    return 0;
}
