// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The production rewind predicate is tested directly. The cache simulation is
// an independent absolute-position/value oracle: it proves the proposed ring
// capacity contract, not that a CUDA tensor graph implements this simulation.
// Real graph/slot/Engram integration must additionally pass numerical fixtures.
#include "dsv41_dspark.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>

namespace {
uint64_t checks = 0, branches = 0, rejected_guards = 0;
void require(bool condition, const char * message) {
    ++checks;
    if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

struct row {
    int64_t position = -1;
    int64_t value = -1;
};

// The value distinguishes two histories that assign different tokens to the
// same absolute position. Checking positions alone would miss stale branches.
int64_t token(int64_t p, int64_t branch) { return branch * 1000000 + p * 17 + 3; }

struct state {
    int64_t ratio, block, window, head = 0, begin = -1, end = -1, checkpoint = -1;
    std::vector<row> compressor, raw, draft;
    std::map<int64_t, std::vector<int64_t>> compressed;
    std::vector<int64_t> history;

    state(int64_t r, int64_t b, int64_t w, int64_t raw_rows, int64_t extra)
        : ratio(r), block(b), window(w), compressor((size_t)(r + extra)),
          raw((size_t)raw_rows), draft((size_t)raw_rows) {}

    // Sequential construction does not reuse the candidate's scratch/persist
    // addressing. It provides an independent complete reference prefix.
    void seed(int64_t n) {
        for (int64_t p = 0; p < n; ++p) {
            row value{p, token(p, 0)};
            compressor[(size_t)(p % compressor.size())] = value;
            raw[(size_t)(p % raw.size())] = draft[(size_t)(p % draft.size())] = value;
            history.push_back(value.value);
            if ((p + 1) % ratio == 0)
                compressed[p / ratio] = std::vector<int64_t>(history.end()-ratio, history.end());
        }
        head = n;
    }

    // Read old incomplete-block rows BEFORE persisting the new ubatch. Values
    // from this call are read from scratch, so even a call wider than the
    // compressor ring must work. No full cache copy is used for rewind.
    bool forward(int64_t count, int64_t branch, bool verify, bool early_persist = false) {
        const int64_t p0 = head;
        begin = end = -1;  // an ordinary/failed write invalidates the old proof
        std::vector<int64_t> expected = history;
        for (int64_t p = p0; p < p0 + count; ++p) expected.push_back(token(p, branch));
        auto persist = [&]() {
            for (int64_t p = p0; p < p0 + count; ++p)
                compressor[(size_t)(p % compressor.size())] = {p, expected[(size_t)p]};
        };
        if (early_persist) persist();
        for (int64_t p = p0; p < p0 + count; ++p) {
            if ((p + 1) % ratio) continue;
            std::vector<int64_t> actual;
            for (int64_t source = p + 1 - ratio; source <= p; ++source) {
                row value = source >= p0 ? row{source, expected[(size_t)source]}
                    : compressor[(size_t)(source % compressor.size())];
                if (value.position != source || value.value != expected[(size_t)source]) return false;
                actual.push_back(value.value);
            }
            compressed[p / ratio] = actual;
        }
        persist();
        // Graph raw/draft writes happen before attention. Verify every causal
        // position in every query window, including an abandoned tail whose
        // physical slot would otherwise be misattributed to an older position.
        for (int64_t p = p0; p < p0 + count; ++p)
            raw[(size_t)(p % raw.size())] = draft[(size_t)(p % draft.size())] = {p, expected[(size_t)p]};
        for (int64_t q = p0; q < p0 + count; ++q) {
            for (int64_t source = std::max<int64_t>(0, q-window+1); source <= q; ++source) {
                for (const auto * ring : {&raw, &draft}) {
                    const auto & value = (*ring)[(size_t)(source % ring->size())];
                    if (value.position != source || value.value != expected[(size_t)source]) return false;
                }
            }
            // Only completed compressed rows are visible. Inspect the newest
            // three (including the partial/replacement boundary) and row zero.
            const int64_t visible = (q + 1) / ratio;
            std::set<int64_t> ids;
            if (visible) ids.insert(0);
            for (int64_t id = std::max<int64_t>(0, visible-3); id < visible; ++id) ids.insert(id);
            for (auto id : ids) {
                auto found = compressed.find(id);
                if (found == compressed.end()) return false;
                const std::vector<int64_t> correct(expected.begin()+id*ratio, expected.begin()+(id+1)*ratio);
                if (found->second != correct) return false;
            }
        }
        history = std::move(expected);
        head += count;
        if (verify) { begin = p0; end = head; }
        return true;
    }

    bool rewind(int64_t target) {
        const int64_t raw_span = (int64_t)raw.size() - window + 1;
        if (!dsv41_dspark_can_rewind(begin, end, head, target, block, raw_span)) return false;
        head = target;
        history.resize((size_t)target); // Engram cannot retain rejected tokens
        if (checkpoint > target) checkpoint = -1;
        return true;
    }
};

void guard_grid() {
    // Independent retained-prefix enumeration: accepted anchor plus0..drafts
    // tokens. No copy of the implementation's boolean expression is used.
    for (int64_t b : {1, 2, 5, 8})
    for (int64_t start : {0, 1, 2, 7, 127, 128, 513})
    for (int64_t width = 1; width <= b+1; ++width) {
        const int64_t end = start+width;
        std::set<int64_t> prefixes;
        for (int64_t accepted = 0; accepted < width; ++accepted) prefixes.insert(start+1+accepted);
        for (int64_t current = start+1; current <= end; ++current)
        for (int64_t target = start-1; target <= end+1; ++target) {
            bool expected = prefixes.count(target) && target <= current;
            require(dsv41_dspark_can_rewind(start,end,current,target,b,b) == expected, "retained anchor/accepted prefix guard");
        }
        for (const auto & bad : std::vector<std::vector<int64_t>>{
            {-1,end,end,start+1,b,b}, {start,start,end,start+1,b,b},
            {start,start+b+2,start+b+2,start+1,b,b}, {start,end,start,start,b,b},
            {start,end,start-1,start-1,b,b}, {start,end,0,0,b,b},
            {start,end,end+1,start+1,b,b}, {start,end,end,start,b,b},
            {start,end,end,start+1,0,b}, {start,end,end,start+1,b,b-1},
            {start,end,end,start+1,b,-1},
            {-1,-1,end,start+1,b,b}}) {
            require(!dsv41_dspark_can_rewind(bad[0],bad[1],bad[2],bad[3],bad[4],bad[5]), "unsafe/stale verification interval refused");
            ++rejected_guards;
        }
    }
}

void branches_grid() {
    for (int64_t b : {1, 5, 8})
    for (int64_t ratio : {1, 2, 4, 8})
    for (int64_t window : {8, 128}) {
        const int64_t chunk = ratio+b+3;
        const int64_t raw_rows = window+std::max(chunk,b+1);
        // Every raw-ring and compressor-ring alignment across multiple wraps.
        for (int64_t prefix = 0; prefix < 2*raw_rows+2*(ratio+b); ++prefix) {
            state original(ratio,b,window,raw_rows,b);
            original.seed(prefix);
            require(original.forward(b+1,1,true), "verify writes and visibility");
            original.checkpoint = original.head; // speculative checkpoint is invalid after partial acceptance
            for (int64_t accepted = 0; accepted <= b; ++accepted) {
                auto changed = original;
                const int64_t target = prefix+1+accepted;
                require(changed.rewind(target), "all accepted prefix lengths are legal");
                require((int64_t)changed.history.size() == target, "Engram tail removed");
                require(changed.checkpoint == (target == original.end ? target : -1), "future checkpoint invalidated");
                // Repeated rewind uses the ORIGINAL last-written end, not the
                // progressively reduced head. It must not renew its budget.
                auto repeated = changed;
                require(repeated.rewind(prefix+1), "repeated rewind remains inside original verify");
                require(!repeated.rewind(prefix), "repeated rewind cannot discard anchor");
                require(repeated.forward(chunk,2,false), "replacement after repeated rewind");
                require(changed.forward(chunk,2,false), "replacement partial blocks and wrapped scratch persist");
                require(!changed.rewind(changed.head-1), "ordinary forward invalidates verify authorization");
                require(changed.forward(b+1,3,true), "next independent verify");
                require(changed.rewind(changed.begin+1+accepted), "next verify interval and accepted tail");
                require(changed.forward(1,4,false), "single-token replacement after second verify");
                ++branches;
            }
        }
    }
}

void counterexamples_and_checkpoint() {
    // Missing B padding corrupts the pre-existing half block even though its
    // absolute compressed output is hidden until the replacement boundary.
    state short_ring(2,5,8,32,0); short_ring.seed(8);
    require(short_ring.forward(6,1,true), "negative control initial verify");
    require(short_ring.rewind(9), "negative control legal metadata rewind");
    require(!short_ring.forward(1,2,false), "unpadded compressor corruption detected");

    state ordering(2,5,8,32,5); ordering.seed(9);
    require(!ordering.forward(10,2,false,true), "persist-before-read corruption detected");

    state short_raw(2,5,8,12,5); short_raw.seed(64);
    require(!short_raw.forward(6,1,true), "raw ring too short for full verify ubatch detected");

    state live(2,5,8,32,5); live.seed(64);
    const state shadow = live;
    for (int i = 0; i < 4; ++i) require(live.forward(20,1,false), "long forward for checkpoint setup");
    // A full conversational checkpoint restores draft rings too. Omitting only
    // this ring is a deliberate control that must fail the first continuation.
    auto missing_draft_shadow = shadow;
    missing_draft_shadow.draft = live.draft;
    require(!missing_draft_shadow.forward(1,2,false), "missing draft checkpoint shadow detected");
    auto restored = shadow;
    require(restored.forward(1,2,false), "complete conversational shadow restores continuation");
}
} // namespace

int main() {
    guard_grid();
    branches_grid();
    counterexamples_and_checkpoint();
    std::printf("PASS: DSpark guard/ring contract checks=%llu branches=%llu refused=%llu\n",
        (unsigned long long)checks, (unsigned long long)branches, (unsigned long long)rejected_guards);
    std::puts("Scope: direct production rewind predicate plus independent ring simulation; no model/CUDA/cache tensor execution.");
    return 0;
}
