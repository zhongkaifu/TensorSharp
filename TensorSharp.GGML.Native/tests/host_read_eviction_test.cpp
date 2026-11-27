// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// A cached buffer that an op's queued download may still be reading must not be
// freed until host_read_barrier has drained that download. The releases that
// used to skip it: LRU eviction in the paged- and prefill-attention session
// caches (destroy() ran before the call's barrier), and growth of the shared
// reuse compute buffer and reuse gallocr (the old buffer is freed before the
// next op's barrier). Only Metal async queues such a download, so this drives
// the CPU backend, marks the barrier through the test diagnostics as such a
// download does (TSGgml_TestMarkPendingGpuWork), and asks them whether a release
// ran while that mark was still pending.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
extern "C" {
int TSGgml_IsBackendAvailable(int);
const char* TSGgml_GetLastError();
int TSGgml_HostReadBarrier();
int TSGgml_PagedAttentionSessionCount();
int TSGgml_PrefillAttentionSessionCount();
int TSGgml_TestUndrainedCacheReleases();
int TSGgml_TestMarkPendingGpuWork();
std::int64_t TSGgml_TestReuseAllocate(std::int64_t, int);
void TSGgml_Shutdown();
int TSGgml_PagedAttentionForward(const float*,const float*,const float*,float*,const int*,const int*,
    const int*,const int*,const int*,int,int,int,int,int,int,int,float);
int TSGgml_FusedPrefillAttentionF32(const float*,const float*,const float*,float*,
    int,int,int,int,int,int,int,float,int);
}

namespace {
constexpr int kHeads = 4, kKvHeads = 2, kDim = 64, kBlock = 4, kKeys = 8, kRows = 2;
constexpr int kRowFloats = kRows * kHeads * kDim;
constexpr int kKvFloats = kKeys * kKvHeads * kDim;
constexpr int kCacheSize = 16;   // kPagedAttnCacheSize == kPrefillAttnCacheSize

int failures = 0;
void expect(bool condition, const char* what)
{
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
}

// Distinct scales key distinct cached sessions without changing any shape.
float scale_of(int i) { return (1.0f + i / 64.0f) / std::sqrt(float(kDim)); }

struct Inputs
{
    std::vector<float> q, k, v;
    Inputs() : q(kRowFloats), k(kKvFloats), v(kKvFloats)
    {
        std::uint32_t state = 0x2545f491u;
        auto next = [&] { state = state * 1664525u + 1013904223u; return (state >> 8) / float(1u << 24) - 0.5f; };
        for (float& x : q) x = next();
        for (float& x : k) x = next();
        for (float& x : v) x = next();
    }
};

const int kStarts[] = {0, kRows}, kLengths[] = {kKeys}, kPositions[] = {kKeys - 2, kKeys - 1};
const int kTable[] = {0, 1}, kOffsets[] = {0};

bool finite(const float* x, int n)
{
    for (int i = 0; i < n; i++) if (!std::isfinite(x[i])) return false;
    return true;
}

// The host-array path: one cached paged-attention session per scale.
bool paged(const Inputs& in, int i)
{
    std::vector<float> out(kRowFloats, NAN);
    if (!TSGgml_PagedAttentionForward(in.q.data(), in.k.data(), in.v.data(), out.data(), kStarts, kLengths,
            kPositions, kTable, kOffsets, 1, kRows, kHeads, kKvHeads, kDim, kBlock, 0, scale_of(i)))
    {
        std::fprintf(stderr, "paged attention failed: %s\n", TSGgml_GetLastError());
        return false;
    }
    return finite(out.data(), kRowFloats);
}

// Head-first Q [heads, rows, dim] and K/V [kv heads, keys, dim]: the cached path.
bool prefill(const Inputs& in, int i)
{
    std::vector<float> out(kRowFloats, NAN);
    if (!TSGgml_FusedPrefillAttentionF32(in.q.data(), in.k.data(), in.v.data(), out.data(),
            kHeads, kKvHeads, kDim, kRows, kKeys, kKeys - kRows, 0, scale_of(i), 0))
    {
        std::fprintf(stderr, "prefill attention failed: %s\n", TSGgml_GetLastError());
        return false;
    }
    return finite(out.data(), kRowFloats);
}

// Work an op left queued for the barrier, as Metal async compute leaves it.
bool mark()
{
    if (!TSGgml_TestMarkPendingGpuWork())
    {
        std::fprintf(stderr, "marking pending GPU work failed: %s\n", TSGgml_GetLastError());
        return false;
    }
    return true;
}
}

int main()
{
    if (!TSGgml_IsBackendAvailable(2)) return 77;
    const Inputs in;

    // Control: the mark really is pending, or nothing below is tested.
    if (!mark()) return 1;
    expect(TSGgml_HostReadBarrier() == 1, "the mark left work pending");
    expect(TSGgml_HostReadBarrier() == 0, "one barrier drained the mark");

    // Paged attention: fill the cache, mark, then a new scale evicts the LRU session.
    int before = TSGgml_TestUndrainedCacheReleases();
    for (int i = 0; i < kCacheSize; i++)
        if (!paged(in, i)) return 1;
    expect(TSGgml_PagedAttentionSessionCount() == kCacheSize, "the paged session cache is full");
    if (!mark() || !paged(in, kCacheSize)) return 1;
    expect(TSGgml_PagedAttentionSessionCount() == kCacheSize, "the new paged session replaced an evicted one");
    expect(TSGgml_TestUndrainedCacheReleases() == before,
           "paged attention evicted a session while marked work was pending");

    // Prefill attention: the same, through its own per-thread session cache.
    before = TSGgml_TestUndrainedCacheReleases();
    for (int i = 0; i < kCacheSize; i++)
        if (!prefill(in, i)) return 1;
    expect(TSGgml_PrefillAttentionSessionCount() == kCacheSize, "the prefill session cache is full");
    if (!mark() || !prefill(in, kCacheSize)) return 1;
    expect(TSGgml_PrefillAttentionSessionCount() == kCacheSize, "the new prefill session replaced an evicted one");
    expect(TSGgml_TestUndrainedCacheReleases() == before,
           "prefill attention evicted a session while marked work was pending");

    // The shared reuse compute buffer, grown past its 64 MiB slab while marked.
    before = TSGgml_TestUndrainedCacheReleases();
    const std::int64_t small_buffer = TSGgml_TestReuseAllocate(1024, 0);
    if (small_buffer <= 0 || !mark()) return 1;
    const std::int64_t grown_buffer = TSGgml_TestReuseAllocate(std::int64_t(20) << 20, 0);
    expect(grown_buffer > small_buffer, "the reuse compute buffer grew");
    expect(TSGgml_TestUndrainedCacheReleases() == before,
           "the reuse compute buffer was freed for growth while marked work was pending");

    // The shared reuse gallocr, handed a larger graph while marked.
    before = TSGgml_TestUndrainedCacheReleases();
    const std::int64_t small_gallocr = TSGgml_TestReuseAllocate(1024, 1);
    if (small_gallocr <= 0 || !mark()) return 1;
    const std::int64_t grown_gallocr = TSGgml_TestReuseAllocate(std::int64_t(1) << 20, 1);
    expect(grown_gallocr > small_gallocr, "the reuse gallocr grew");
    expect(TSGgml_TestUndrainedCacheReleases() == before,
           "the reuse gallocr was given a graph to grow into while marked work was pending");

    TSGgml_Shutdown();
    if (failures != 0)
    {
        std::fprintf(stderr, "%d host read eviction check(s) failed\n", failures);
        return 1;
    }
    std::printf("host read eviction: all checks passed\n");
    return 0;
}
