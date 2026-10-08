// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
struct TensorView2DDesc { void* data; int dim0, dim1, stride0, stride1; std::int64_t raw_bytes; };
struct TensorView3DDesc { void* data; int dim0, dim1, dim2, stride0, stride1, stride2; std::int64_t raw_bytes; };
const char* TSGgml_GetLastError();
int TSGgml_GetGpuDeviceCount(int backend_type);
int TSGgml_IsBackendAvailable(int backend_type);
int TSGgml_GatedDeltaNetChunkedF32(TensorView3DDesc q, TensorView3DDesc k,
    TensorView3DDesc v, TensorView3DDesc z, TensorView2DDesc alpha, TensorView2DDesc beta,
    TensorView3DDesc state, TensorView3DDesc output, void* dt_bias, void* a_log,
    void* norm, int chunk_size, float eps, int gate_mode);
std::int64_t TSGgml_TestGdnChunkedCacheBytes();
std::int64_t TSGgml_TestQwen35RecurrentPrefillCacheBytes();
void TSGgml_ClearHostBufferCache();
void TSGgml_Shutdown();
}
static void require(bool condition, const char* message) {
    if (!condition) { std::fprintf(stderr, "%s: %s\n", message, TSGgml_GetLastError()); std::exit(1); }
}
static TensorView3DDesc three(std::vector<float>& values, int a, int b, int c) {
    return {values.data(), a, b, c, b * c, c, 1, std::int64_t(values.size() * sizeof(float))};
}
static TensorView2DDesc two(std::vector<float>& values, int a, int b) {
    return {values.data(), a, b, b, 1, std::int64_t(values.size() * sizeof(float))};
}
static void populate() {
    constexpr int tokens = 32, heads = 1, width = 128;
    std::vector<float> q(tokens * heads * width, 0), k(q), v(q.size(), 1), z(q.size(), 1),
        output(q.size(), -1), alpha(tokens * heads, -.1f), beta(tokens * heads, .5f),
        state(heads * width * width, 0), norm(width, 1);
    require(TSGgml_GatedDeltaNetChunkedF32(three(q, tokens, heads, width), three(k, tokens, heads, width),
        three(v, tokens, heads, width), three(z, tokens, heads, width), two(alpha, tokens, heads),
        two(beta, tokens, heads), three(state, heads, width, width), three(output, tokens, heads, width),
        nullptr, nullptr, norm.data(), 64, 1e-6f, 0) == 1, "populate real GDN cached graph");
    require(std::all_of(output.begin(), output.end(), [](float x) { return std::isfinite(x) && x == 0; }),
        "zero-query GDN fixture output");
    require(TSGgml_TestGdnChunkedCacheBytes() > 0, "graph must own a physical backend buffer before cleanup");
}
int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (TSGgml_GetGpuDeviceCount(3) < 1) return 77;
    require(TSGgml_IsBackendAvailable(3) == 1, "initialize CUDA");
    for (int cycle = 0; cycle < 2; ++cycle) {
        populate();
        std::printf("cycle %d cached GDN buffer: %lld bytes\n", cycle,
            static_cast<long long>(TSGgml_TestGdnChunkedCacheBytes()));
        TSGgml_ClearHostBufferCache();
        require(TSGgml_TestGdnChunkedCacheBytes() == 0 && TSGgml_TestQwen35RecurrentPrefillCacheBytes() == 0,
            "model cache clear must physically release recurrent graphs");
    }
    populate();
    TSGgml_Shutdown();
    require(TSGgml_TestGdnChunkedCacheBytes() == 0 && TSGgml_TestQwen35RecurrentPrefillCacheBytes() == 0,
        "shutdown must physically release recurrent graphs");
    std::puts("Repeated GDN cache creation/clear, shutdown, and ordinary process exit passed.");
    return 0; // Static/atexit destruction must now find no device allocation.
}
