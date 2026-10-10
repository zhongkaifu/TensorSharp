// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_expert_eviction.h"
#include <cstdio>
#include <cstdlib>

static void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

int main() {
    tsg::ExpertEvictionPolicy lru, lfu;
    lru.configure(512, 116, false);
    lfu.configure(512, 116, true);
    require(lru.epoch() == 0 && lfu.epoch() == 256, "capacity-dependent epoch is wrong");
    const std::int32_t hot[] = {0}, rare[] = {1};
    for (int i = 0; i < 20; ++i) { lru.observe(hot, 1); lfu.observe(hot, 1); }
    lru.observe(rare, 1); lfu.observe(rare, 1);
    require(lru.prefer(0, 1, 1, 20), "LRU override lost its recency ordering");
    require(lfu.prefer(1, 20, 0, 1), "frequent old expert was evicted before rare new expert");
    require(lfu.prefer(-1, 999, 1, 20), "invalid slot was retained before a valid expert");
    require(lfu.prefer(2, 10, 3, 20) && !lfu.prefer(2, 20, 3, 10), "LFU ties lost LRU ordering");
    // A shift in the request's routing must overcome old popularity.
    for (int i = 0; i < 256 * 8; ++i) lfu.observe(rare, 1);
    require(lfu.prefer(0, 1000, 1, 1), "frequency aging did not forget old routing");
    // Saturation, not wraparound, when repeated IDs are supplied in one row.
    std::vector<std::int32_t> repeated(65536, 2);
    lfu.observe(repeated.data(), static_cast<int>(repeated.size()));
    require(lfu.prefer(1, 0, 2, 100), "frequency counter overflowed");
    lfu.configure(512, 24, true);
    require(lfu.epoch() == 64 && lfu.prefer(2, 0, 1, 1), "reconfiguration kept stale metadata");
    lfu.configure(4096, 4096, true);
    require(lfu.epoch() == 1024, "large cache epoch exceeded its bound");
    bool rejected = false;
    try { lfu.configure(4, 5, true); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid geometry accepted");
    std::puts("PASS: bounded decaying frequencies, LRU ties, empty slots, saturation and routing shifts");
}
