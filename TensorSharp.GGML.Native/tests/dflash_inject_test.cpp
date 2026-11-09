// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The DFlash ring inject (TSGgml_DFlashInject*, shared by DFlash and DSpine) under the failures that crashed a
// Qwen3.8-27B DSpine server on ggml_cuda:
//   - a long inject in chunks (TSGgml_DFlashTestInjectChunk) against the same rows in one graph: the ring rows match,
//     bit for bit or within the GEMM kernel's rounding (the row count picks the kernel), and the difference is printed;
//   - a graph-storage allocation that fails (TSGgml_DFlashTestFailAllocs) declines the call with an error, and the
//     next call builds and writes the rows;
//   - the ring's cached device copy freed under a persisted inject graph (TSGgml_InvalidateHostBuffer, as a per-op ring
//     write did) and the host ring edited: the next inject rebuilds its graph instead of replaying it over the released
//     copy (an illegal memory access on CUDA), and the ring holds the edit, the earlier rows and the new row;
//   - a ring the backend cannot write in place (on CPU, a host pointer below the buffer type's alignment) declines the
//     inject with an error instead of writing the rows into a graph-local copy and dropping them; CUDA writes such a
//     ring through its cached device copy.
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
int TSGgml_IsBackendAvailable(int);
const char* TSGgml_GetLastError();
int TSGgml_SyncHostBuffer(void*, size_t);
void TSGgml_InvalidateHostBuffer(void*);
void TSGgml_DFlashResetCaches();
void TSGgml_ClearHostBufferCache();
void TSGgml_DFlashTestFailAllocs(int);
void TSGgml_DFlashTestInjectChunk(int);
int TSGgml_DFlashInjectScaled(
    const float* feat_rows, int feature_size, int n_rows, const std::int64_t* ring_rows_idx, const int* positions,
    int num_layers, int hidden_size, int head_dim, int num_kv_heads, int ring_rows,
    float eps, float rope_base, float rope_freq_scale,
    const void* fc_data, int fc_type, std::int64_t fc_ne0, std::int64_t fc_ne1, std::int64_t fc_bytes,
    const void* enc_norm_data,
    void** k_arr, int* k_type_arr, std::int64_t* k_ne0_arr, std::int64_t* k_ne1_arr, std::int64_t* k_bytes_arr,
    void** v_arr, int* v_type_arr, std::int64_t* v_ne0_arr, std::int64_t* v_ne1_arr, std::int64_t* v_bytes_arr,
    void** k_norm_arr, void** ring_k_arr, void** ring_v_arr, int ring_dtype,
    const float* scales, int scale_count);
}

namespace {
constexpr int N = 2, H = 64, HD = 32, NKV = 2, FEAT = 192, RING = 96;

std::uint32_t random_state = 4242;
std::vector<float> data(std::size_t n, float magnitude = .1f) {
    std::vector<float> result(n);
    for (float& v : result) {
        random_state = random_state * 1664525u + 1013904223u;
        v = float(int(random_state >> 16) - 32768) * (magnitude / 32768.f);
    }
    return result;
}

struct Weights {
    std::vector<float> fc = data(std::size_t(FEAT) * H), norm = std::vector<float>(H, 1.f), hnorm = std::vector<float>(HD, 1.f);
    std::vector<std::vector<float>> k, v;
    std::vector<void*> kp, vp, hn;
    std::vector<int> type = std::vector<int>(N, GGML_TYPE_F32);
    std::vector<std::int64_t> ne0 = std::vector<std::int64_t>(N, H), ne1 = std::vector<std::int64_t>(N, NKV * HD),
        bytes = std::vector<std::int64_t>(N, std::int64_t(H) * NKV * HD * sizeof(float));
    Weights() {
        for (int l = 0; l < N; ++l) { k.push_back(data(std::size_t(H) * NKV * HD)); v.push_back(data(std::size_t(H) * NKV * HD)); }
        for (int l = 0; l < N; ++l) { kp.push_back(k[l].data()); vp.push_back(v[l].data()); hn.push_back(hnorm.data()); }
    }
};

// Zeroed floats aligned to 16 KB (a Metal page), as the allocator's tensors are: the ring binds its host pointer only
// at the buffer type's alignment (64 on CPU), which std::vector's 16 does not meet, and an unbound ring would be
// written into a graph copy instead of the host rows.
struct Aligned {
    std::vector<float> store;
    float* p = nullptr;
    std::size_t n = 0;
    explicit Aligned(std::size_t count, std::size_t skew = 0) : store(count + 16384 / sizeof(float) + skew), n(count) {
        p = reinterpret_cast<float*>((reinterpret_cast<std::uintptr_t>(store.data()) + 16383) & ~std::uintptr_t(16383)) + skew;
    }
    Aligned(const Aligned&) = delete;
    Aligned& operator=(const Aligned&) = delete;
    Aligned(Aligned&&) = default;   // the moved store keeps its buffer, so p stays valid
    float* data() { return p; }
    const float* data() const { return p; }
    std::size_t size() const { return n; }
    float& operator[](std::size_t i) { return p[i]; }
    float operator[](std::size_t i) const { return p[i]; }
};

struct Ring {
    std::vector<Aligned> k, v;
    std::vector<void*> kp, vp;
    // `skew` floats past a 16 KB boundary: 0 binds in place on every backend, 1 is below the CPU's alignment.
    explicit Ring(std::size_t skew = 0) {
        for (int l = 0; l < N; ++l) { k.emplace_back(std::size_t(RING) * HD * NKV, skew); v.emplace_back(std::size_t(RING) * HD * NKV, skew); }
        for (int l = 0; l < N; ++l) { kp.push_back(k[l].data()); vp.push_back(v[l].data()); }
    }
    void sync() {
        for (int l = 0; l < N; ++l) {
            TSGgml_SyncHostBuffer(k[l].data(), k[l].size() * sizeof(float));
            TSGgml_SyncHostBuffer(v[l].data(), v[l].size() * sizeof(float));
        }
    }
    // The ring rows of `slot` (every layer, K then V, every head).
    std::vector<float> rows(int slot) const {
        std::vector<float> r;
        for (int l = 0; l < N; ++l)
            for (const auto* t : { &k[l], &v[l] })
                for (int h = 0; h < NKV; ++h) {
                    const float* p = t->data() + (std::size_t(h) * RING + slot) * HD;
                    r.insert(r.end(), p, p + HD);
                }
        return r;
    }
};

int inject(const Weights& w, Ring& ring, int start, int n) {
    std::vector<float> feats;
    std::vector<std::int64_t> idx(n);
    std::vector<int> pos(n);
    for (int i = 0; i < n; ++i) {
        random_state = 7919u + std::uint32_t(start + i) * 104729u;   // features are a function of the position
        std::vector<float> f = data(FEAT, 1.f);
        feats.insert(feats.end(), f.begin(), f.end());
        idx[i] = (start + i) % RING;
        pos[i] = start + i;
    }
    return TSGgml_DFlashInjectScaled(feats.data(), FEAT, n, idx.data(), pos.data(), N, H, HD, NKV, RING, 1e-6f, 10000.f, 1.f,
        w.fc.data(), GGML_TYPE_F32, FEAT, H, std::int64_t(FEAT) * H * sizeof(float), w.norm.data(),
        const_cast<void**>(w.kp.data()), const_cast<int*>(w.type.data()), const_cast<std::int64_t*>(w.ne0.data()),
        const_cast<std::int64_t*>(w.ne1.data()), const_cast<std::int64_t*>(w.bytes.data()),
        const_cast<void**>(w.vp.data()), const_cast<int*>(w.type.data()), const_cast<std::int64_t*>(w.ne0.data()),
        const_cast<std::int64_t*>(w.ne1.data()), const_cast<std::int64_t*>(w.bytes.data()),
        const_cast<void**>(w.hn.data()), ring.kp.data(), ring.vp.data(), GGML_TYPE_F32, nullptr, 0);
}

double max_rel(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 1e-30;
    for (std::size_t i = 0; i < a.size(); ++i) { num = std::max(num, double(std::fabs(a[i] - b[i]))); den = std::max(den, double(std::fabs(b[i]))); }
    return num / den;
}

bool fail(const char* what) { std::fprintf(stderr, "FAIL %s (%s)\n", what, TSGgml_GetLastError()); return false; }

void reset() { TSGgml_DFlashTestInjectChunk(0); TSGgml_DFlashTestFailAllocs(0); TSGgml_DFlashResetCaches(); TSGgml_ClearHostBufferCache(); }
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const int backend = !std::strcmp(argv[1], "cuda") ? 3 : !std::strcmp(argv[1], "metal") ? 1 : 2;
    if (!TSGgml_IsBackendAvailable(backend)) return 77;
    const Weights w;
    bool ok = true;

    // 1. Chunked against one graph: 40 rows, chunks of 16 (the last one padded by repeating its last row).
    {
        Ring single, chunked;
        if (inject(w, single, 0, 40) != 1) ok = fail("single-shot inject");
        TSGgml_DFlashTestInjectChunk(16);
        if (inject(w, chunked, 0, 40) != 1) ok = fail("chunked inject");
        TSGgml_DFlashTestInjectChunk(0);
        single.sync(); chunked.sync();
        double worst = 0;
        bool identical = true;
        for (int s = 0; s < 40; ++s) {
            const auto a = chunked.rows(s), b = single.rows(s);
            worst = std::max(worst, max_rel(a, b));
            identical = identical && a == b;
        }
        // Slots past the injected rows stay untouched (the padding wrote slot 39 only).
        bool untouched = true;
        for (int s = 40; s < RING; ++s)
            for (float x : chunked.rows(s)) untouched = untouched && x == 0.f;
        std::printf("chunked (16) vs single-shot (40 rows): %s, max relative difference %.3g; slots past the rows untouched %d\n",
                    identical ? "bit-identical" : "not bit-identical", worst, int(untouched));
        if (!(worst < 2e-2) || !untouched) ok = fail("chunked inject rows");
        reset();
    }

    // 2. A failed graph-storage allocation declines with an error; the next call builds and writes.
    {
        Ring got, want;
        if (inject(w, want, 0, 7) != 1) ok = fail("reference inject");
        want.sync();   // before reset(): clearing the buffer cache drops the device copy, unsynced rows with it
        reset();
        TSGgml_DFlashTestFailAllocs(1);
        const int declined = inject(w, got, 0, 7);
        const std::string err = TSGgml_GetLastError() ? TSGgml_GetLastError() : "";
        const bool said = err.find("allocate") != std::string::npos;
        if (declined != 0 || !said) ok = fail("a failed allocation must decline with an allocation error");
        if (inject(w, got, 0, 7) != 1) ok = fail("the inject after a declined one");
        got.sync();
        bool same = true;
        for (int s = 0; s < 7; ++s) same = same && got.rows(s) == want.rows(s);
        std::printf("failed allocation: declined %d (\"%s\"), next inject wrote the rows %d\n", int(declined == 0), err.c_str(), int(same));
        if (!same) ok = fail("rows after a declined inject");
        reset();
    }

    // 3. The ring's device copy released under a persisted graph, the host ring edited, then the cached size again.
    {
        Ring got, want;
        for (int p = 0; p < 3; ++p) if (inject(w, want, p, 1) != 1) ok = fail("reference injects");
        want.sync();
        reset();
        if (inject(w, got, 0, 1) != 1 || inject(w, got, 1, 1) != 1) ok = fail("injects before the release");   // graph cached
        got.sync();
        for (int l = 0; l < N; ++l) { TSGgml_InvalidateHostBuffer(got.k[l].data()); TSGgml_InvalidateHostBuffer(got.v[l].data()); }
        const int edited = 50;
        for (int l = 0; l < N; ++l) got.k[l][(std::size_t(0) * RING + edited) * HD] = 3.25f;   // as a per-op ring write would
        if (inject(w, got, 2, 1) != 1) ok = fail("inject after the ring's device copy was released");
        got.sync();
        bool rows = true;
        for (int s = 0; s < 3; ++s) rows = rows && got.rows(s) == want.rows(s);
        bool edit = true;
        for (int l = 0; l < N; ++l) edit = edit && got.k[l][(std::size_t(0) * RING + edited) * HD] == 3.25f;
        std::printf("ring copy released under a cached graph: earlier and new rows %d, host edit kept %d\n", int(rows), int(edit));
        if (!rows || !edit) ok = fail("ring after its device copy was released");
        reset();
    }

    // 4. A misaligned ring: the CPU cannot wrap it, so the inject declines and names the ring; a GPU writes it through
    //    its cached device copy. Either way no inject reports success while the ring misses the rows.
    {
        Ring want, skewed(1);
        if (inject(w, want, 0, 5) != 1) ok = fail("reference inject");
        want.sync();
        reset();
        const int status = inject(w, skewed, 0, 5);
        const std::string err = TSGgml_GetLastError() ? TSGgml_GetLastError() : "";
        skewed.sync();
        bool same = true;
        for (int s = 0; s < 5; ++s) same = same && skewed.rows(s) == want.rows(s);
        const bool declined = status == 0 && err.find("in place") != std::string::npos;
        std::printf("misaligned ring: %s\n", status == 1 ? (same ? "written in place" : "reported success, rows missing")
                                                         : declined ? ("declined (\"" + err + "\")").c_str() : "failed otherwise");
        if (status == 1 ? !same : !declined) ok = fail("a misaligned ring must be written or declined");
        if (backend == 3 && status != 1) ok = fail("CUDA writes a misaligned ring through its device copy");
        reset();
    }

    std::printf("%s DFlash inject robustness (%s)\n", ok ? "ok  " : "FAIL", argv[1]);
    return ok ? 0 : 1;
}
