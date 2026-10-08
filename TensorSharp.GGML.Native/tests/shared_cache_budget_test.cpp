// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_shared_cache_budget.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

static void require(bool value, const char* message)
{
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
struct Ledger {
    struct Allocation { int rank; std::int64_t bytes; bool committed = false; bool physical = false; };
    std::int64_t capacity[3] = {128, 64, 64}; // Shared RAM plus independent GPU constraints.
    std::int64_t reserved[3] = {}, committed[3] = {};
    std::uint64_t next = 0;
    std::map<std::uint64_t, Allocation> allocations;
    bool fail_commit = false;
    static std::uint64_t reserve(void* context, int rank, int kind, std::int64_t bytes)
    {
        auto& l = *static_cast<Ledger*>(context);
        require(kind == 0 || kind == 1, "unknown cache allocation kind");
        if (rank < 0 || rank > 1) return 0;
        for (int pool : {0, rank + 1})
            if (bytes > l.capacity[pool] - l.reserved[pool] - l.committed[pool]) return 0;
        const auto token = ++l.next;
        l.allocations.emplace(token, Allocation{rank, bytes});
        for (int pool : {0, rank + 1}) l.reserved[pool] += bytes;
        return token;
    }
    static int commit(void* context, std::uint64_t token)
    {
        auto& l = *static_cast<Ledger*>(context);
        if (l.fail_commit) return 0;
        auto& a = l.allocations.at(token);
        require(!a.committed, "double commit");
        for (int pool : {0, a.rank + 1}) { l.reserved[pool] -= a.bytes; l.committed[pool] += a.bytes; }
        a.committed = true;
        return 1;
    }
    static void release(void* context, std::uint64_t token)
    {
        auto& l = *static_cast<Ledger*>(context);
        auto a = l.allocations.at(token);
        require(!a.physical, "shared credits released before physical memory");
        for (int pool : {0, a.rank + 1}) (a.committed ? l.committed[pool] : l.reserved[pool]) -= a.bytes;
        l.allocations.erase(token);
    }
};
static bool attach(Ledger& ledger)
{
    return tsg::SharedCacheCharge::attach(&ledger, Ledger::reserve, Ledger::commit, Ledger::release);
}

int main()
{
    using tsg::SharedCacheCharge;
    Ledger ledger;
    // An allocation admitted before configuration cannot silently escape a new budget.
    auto unconfigured = SharedCacheCharge::reserve(0, 0, 64);
    require(bool(unconfigured) && !attach(ledger), "attach adopted unbudgeted in-flight allocation");
    require(unconfigured->commit(64) && !attach(ledger), "attach adopted an unbudgeted resident allocation");
    unconfigured.reset();
    require(attach(ledger), "cannot attach an empty cache budget");
    require(!attach(ledger), "double registration accepted");
    require(!SharedCacheCharge::detach(nullptr), "wrong owner detached callbacks");

    auto first = SharedCacheCharge::reserve(0, 0, 64);
    auto second = SharedCacheCharge::reserve(1, 1, 64);
    require(first && second && ledger.reserved[0] == 128, "rank/UMA reservation is not atomic");
    require(!SharedCacheCharge::reserve(1, 0, 1), "shared quota exceeded");
    require(!SharedCacheCharge::detach(&ledger), "detached pending callbacks");
    require(first->commit(64) && second->commit(64), "commit failed");
    require(ledger.committed[0] == 128 && ledger.committed[1] == 64 && ledger.committed[2] == 64,
        "committed multi-pool accounting is incorrect");
    require(!SharedCacheCharge::detach(&ledger), "detached resident callbacks");
    first.reset(); second.reset();

    // A native allocation/upload failure unwinds physical storage before its
    // ticket; both pending and committed failures obey the same ownership order.
    for (bool committed : {false, true}) {
        try {
            auto ticket = SharedCacheCharge::reserve(0, 0, 64);
            require(bool(ticket), "cannot reserve rollback case");
            auto& allocation = ledger.allocations.at(ledger.next);
            allocation.physical = true;
            struct Physical { bool& live; ~Physical() { live = false; } } physical{allocation.physical};
            if (committed) require(ticket->commit(64), "cannot commit rollback case");
            throw std::runtime_error("injected native allocation/upload failure");
        } catch (const std::runtime_error&) { }
        require(ledger.allocations.empty() && ledger.reserved[0] == 0 && ledger.committed[0] == 0,
            "rollback leaked shared credit");
    }
    auto too_large = SharedCacheCharge::reserve(0, 0, 64);
    require(!too_large->commit(65), "under-reserved allocation published");
    too_large.reset();
    ledger.fail_commit = true;
    auto refused_commit = SharedCacheCharge::reserve(0, 0, 64);
    require(!refused_commit->commit(64), "failed callback commit accepted");
    refused_commit.reset();
    ledger.fail_commit = false;

    std::atomic<int> attempted{0}, admitted{0};
    std::atomic<bool> release{false};
    std::vector<std::thread> workers;
    for (int i = 0; i < 32; ++i) workers.emplace_back([&, i] {
        auto ticket = SharedCacheCharge::reserve(i % 2, i % 2, 16);
        if (ticket) admitted.fetch_add(1);
        attempted.fetch_add(1);
        while (!release.load()) std::this_thread::yield();
        if (ticket) require(ticket->commit(16), "concurrent reservation could not commit");
    });
    while (attempted.load() != 32) std::this_thread::yield();
    require(admitted == 8 && !SharedCacheCharge::detach(&ledger), "concurrent admission or detach escaped live ownership");
    release = true;
    for (auto& worker : workers) worker.join();
    require(ledger.allocations.empty() && ledger.committed[0] == 0 && ledger.reserved[0] == 0,
        "concurrent release leaked shared charges");
    require(SharedCacheCharge::detach(&ledger), "cannot detach fully released callbacks");
    require(attach(ledger) && SharedCacheCharge::detach(&ledger), "detach cannot be followed by clean registration");
    std::puts("PASS shared cache bridge: atomic attach/detach, multi-pool/rank quota, physical-free order, rollback and concurrency");
}
