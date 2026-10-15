// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../glm_upload_buffer.h"
#include "../glm_file_strip.h"
#include "../glm_joined_workers.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

static std::atomic<bool> fail_next_new{false};
void * operator new(std::size_t size) {
    if (fail_next_new.exchange(false)) throw std::bad_alloc();
    if (void * p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }

namespace {
std::atomic<int> checks{0};
void require(bool value, const char * message) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL %s\n", message); std::exit(1); }
}
template<class Error, class Action> void reject(Action action, const char * message) {
    bool caught = false;
    try { action(); } catch (const Error &) { caught = true; }
    require(caught, message);
}
struct job { size_t len, src_row = 0, row_len = 0; };
struct ledger {
    std::mutex mutex;
    size_t limit = 0, owned = 0, physical = 0, peak = 0, allocations = 0, refused = 0;
    uint64_t next = 0;
    bool fail_allocate = false, null_allocate = false, fail_commit = false;
    struct ticket { size_t bytes, physical = 0; bool committed = false; std::thread::id thread; };
    std::map<uint64_t, ticket> tickets;
    std::map<uint8_t *, uint64_t> pointers;
    static ledger * current;
    static uint64_t reserve(void * c, int rank, int kind, int64_t bytes) {
        auto & l = *static_cast<ledger *>(c); std::lock_guard<std::mutex> lock(l.mutex);
        require(rank == 0 && kind == 3 && bytes > 0, "host-only aggregate class");
        if (uint64_t(bytes) > l.limit - l.owned) { ++l.refused; return 0; }
        l.tickets.emplace(++l.next, ticket{size_t(bytes), 0, false, std::this_thread::get_id()});
        l.owned += size_t(bytes); l.peak = (std::max)(l.peak, l.owned);
        return l.next;
    }
    static int commit(void * c, uint64_t id) {
        auto & l = *static_cast<ledger *>(c); std::lock_guard<std::mutex> lock(l.mutex);
        auto & t = l.tickets.at(id);
        require(!t.committed && t.physical == t.bytes, "aggregate commit follows exact payload allocation");
        if (l.fail_commit) return 0;
        t.committed = true; return 1;
    }
    static void release(void * c, uint64_t id) {
        auto & l = *static_cast<ledger *>(c); std::lock_guard<std::mutex> lock(l.mutex);
        const auto t = l.tickets.at(id);
        require(!t.physical, "all physical payload freed before credit");
        l.owned -= t.bytes; l.tickets.erase(id);
    }
    bool attach() { return tsg::SharedCacheCharge::attach(this, reserve, commit, release, false, true); }
    void empty() {
        require(owned == 0 && physical == 0 && tickets.empty() && pointers.empty(), "no payload or owner leaks");
    }
};
ledger * ledger::current = nullptr;
struct allocator {
    static uint8_t * allocate(size_t bytes) {
        auto & l = *ledger::current; std::lock_guard<std::mutex> lock(l.mutex);
        if (l.fail_allocate) throw std::bad_alloc();
        if (l.null_allocate) return nullptr;
        for (auto & [id, t] : l.tickets) {
            if (t.thread != std::this_thread::get_id() || t.physical || t.bytes != bytes) continue;
            auto * p = new uint8_t[bytes];
            l.pointers.emplace(p, id); t.physical = bytes; l.physical += bytes; ++l.allocations;
            return p;
        }
        throw std::runtime_error("test allocation without exact preceding reservation");
    }
    static void release(uint8_t * p) noexcept {
        auto & l = *ledger::current; std::lock_guard<std::mutex> lock(l.mutex);
        auto & t = l.tickets.at(l.pointers.at(p));
        delete[] p; l.physical -= t.physical; t.physical = 0; l.pointers.erase(p);
    }
};
using arena = tsg_glm::upload_arena<allocator>;
}

int main() {
    using namespace tsg_glm;
    ledger l; ledger::current = &l;
    const std::vector<job> jobs(16, job{64 * 1024 * 1024});
    {
        auto plan = plan_upload(jobs, 16, 64 * 1024 * 1024);
        require(!plan.accounted && plan.workers == 16 && plan.chunk == 64 * 1024 * 1024,
                "unbudgeted plan retains original worker/chunk controls");
        require(!l.attach(), "unaccounted plan blocks mid-upload host attach");
        require(l.allocations == 0, "normal preplan never allocates payload eagerly");
    }
    {
        auto plan = plan_upload(std::vector<job>{{64, 16, 8}}, 1, 64);
        require(!plan.accounted && plan.staging == 64 && plan.slab == 120 && plan.aggregate == 184,
                "unbudgeted TP report retains the actual legacy lazy slab upper bound");
    }
    require(l.attach(), "host scope attaches after normal plan ends");
    {
        auto empty = plan_upload(std::vector<job>{}, 0, 0);
        require(!empty.reservation && empty.workers == 0 && empty.attempts == 0, "no jobs require no quota or workers");
    }
    l.limit = 8 * 1024 * 1024;
    {
        auto plan = plan_upload(jobs, 16, 64 * 1024 * 1024);
        require(plan.accounted && plan.workers == 1 && plan.chunk == 8 * 1024 * 1024,
                "default sixteen-by-64MiB adapts to the current 8MiB quota");
        require(plan.aggregate == l.limit && plan.refusals > 0 && l.physical == 0,
                "one final atomic reservation before any physical allocation");
        arena workspace(plan);
        require(l.physical == l.limit && l.owned == l.limit && l.allocations == 1,
                "one exact aggregate allocation and commit");
        auto stage = workspace.staging(0);
        reject<std::runtime_error>([&] { stage.resize(stage.size() + 1); }, "worker cannot grow its borrowed slice");
        reject<std::out_of_range>([&] { workspace.staging(1); }, "worker cannot borrow another/out-of-range slice");
        require(!tsg::SharedCacheCharge::reserve(0, 3, 1), "later contenders cannot consume a committed arena's credit");
        require(!tsg::SharedCacheCharge::detach(&l), "live arena prevents budget detach");
    }
    l.empty();
    l.limit = 1;
    const size_t before = l.allocations;
    reject<upload_budget_refused>([&] { plan_upload(jobs, 16, 64 * 1024 * 1024); },
        "one-byte quota refuses before payload allocation or model upload");
    require(l.allocations == before, "failed negotiation never allocates payload"); l.empty();
    l.limit = 65536;
    const std::vector<job> small{{16384}};
    for (int failure = 0; failure != 3; ++failure) {
        auto plan = plan_upload(small, 1, 16384);
        l.fail_allocate = failure == 0; l.null_allocate = failure == 1; l.fail_commit = failure == 2;
        if (failure == 2) reject<std::runtime_error>([&] { arena bad(plan); }, "commit failure cleans allocation");
        else reject<std::bad_alloc>([&] { arena bad(plan); }, "throw/null allocation cleans reservation");
        l.empty(); l.fail_allocate = l.null_allocate = l.fail_commit = false;
        auto retry = plan_upload(small, 1, 16384); { arena valid(retry); } l.empty();
    }
    fail_next_new = true;
    reject<std::bad_alloc>([&] { plan_upload(small, 1, 16384); }, "reservation object allocation failure has no physical work");
    l.empty();
    {
        auto plan = plan_upload(std::vector<job>(4, job{8192}), 4, 8192);
        arena workspace(plan);
        std::atomic<size_t> slots{0}; std::atomic<bool> stop{false};
        const size_t allocations = l.allocations;
        run_joined_workers(plan.workers, stop, [&] {
            const size_t slot = slots++;
            auto view = workspace.staging(slot);
            std::memset(view.data(), int(slot + 1), view.size());
        });
        for (size_t slot = 0; slot < plan.workers; ++slot) {
            auto view = workspace.staging(slot);
            require(view.data()[0] == slot + 1 && view.data()[view.size() - 1] == slot + 1,
                    "concurrent worker slices do not overlap");
        }
        require(l.allocations == allocations, "workers never reserve/allocate extra payload");
        reject<std::runtime_error>([&] {
            std::atomic<bool> fail{false};
            run_joined_workers(plan.workers, fail, [&] { throw std::runtime_error("injected upload/read failure"); });
        }, "all failed workers join while arena remains live");
        require(l.physical == plan.aggregate, "worker exception cannot release an active arena");
    }
    l.empty();
    {
        // Quota negotiation sees already-live owners, not configured capacity.
        auto other = tsg::SharedCacheCharge::reserve(0, 3, 57344);
        auto plan = plan_upload(small, 1, 16384);
        require(plan.aggregate == 8192 && l.owned == 65536, "existing reservations reduce current available quota");
        arena workspace(plan);
    }
    l.empty();
    l.limit = 20000;
    {
        auto plan = plan_upload(std::vector<job>{{16384, 128, 64}}, 1, 16384);
        require(plan.chunk == 8192 && plan.staging == 8192 && plan.slab == 8128 && plan.aggregate == 16320,
                "TP source slab and output staging shrink together under quota");
        arena workspace(plan);
    }
    l.empty();
    l.limit = 128;
    {
        auto plan = plan_upload(std::vector<job>{{96, 16, 8}}, 1, 64);
        require(plan.chunk == 64 && plan.staging == 64 && plan.slab == 56 && plan.aggregate == 120,
                "TP source slab and staged output are accounted together");
        arena workspace(plan); auto stage = workspace.staging(0); auto scratch = workspace.slab(0);
        unique_file file(std::tmpfile()); require(bool(file), "create owned strided read fixture");
        uint8_t source[184], actual[96];
        for (size_t i = 0; i < sizeof(source); ++i) source[i] = uint8_t(i);
        require(std::fwrite(source, 1, sizeof(source), file.get()) == sizeof(source) && std::fflush(file.get()) == 0,
                "write finite source fixture ending at final TP strip");
        for (size_t done = 0; done < sizeof(actual);) {
            const size_t bytes = upload_segment_bytes(sizeof(actual) - done, plan.chunk, 16, 8);
            require(read_file_strips(file.get(), (done / 8) * 16, 16, 8, bytes, stage.data(), scratch, plan.slab_limit),
                    "gather through preallocated TP slices");
            std::memcpy(actual + done, stage.data(), bytes); done += bytes;
        }
        for (size_t i = 0; i < sizeof(actual); ++i)
            require(actual[i] == source[(i / 8) * 16 + i % 8], "TP gathered bytes exact through every segment");
    }
    l.empty();
    l.limit = 16383;
    reject<upload_budget_refused>([&] { plan_upload(std::vector<job>{{16384, 16384, 8192}}, 1, 4096); },
        "TP quota below one complete strip plus its staging safely refuses"); l.empty();
    l.limit = 16384;
    { auto plan = plan_upload(std::vector<job>{{16384, 16384, 8192}}, 1, 4096);
      require(plan.staging == 8192 && plan.slab == 8192, "TP row larger than requested chunk stays whole"); arena workspace(plan); }
    l.empty();
    reject<std::invalid_argument>([&] { plan_upload(std::vector<job>{{7, 16, 4}}, 1, 4); }, "partial TP row rejected");
    reject<std::invalid_argument>([&] { plan_upload(std::vector<job>{{8, 2, 4}}, 1, 4); }, "overlapping TP strip rejected");
    reject<std::invalid_argument>([&] { plan_upload(std::vector<job>{{0}}, 1, 4); }, "zero-length job rejected");
    reject<std::overflow_error>([&] { plan_upload(std::vector<job>(2, job{(std::numeric_limits<size_t>::max)()}), 2,
        (std::numeric_limits<size_t>::max)()); }, "aggregate size_t overflow rejected before reservation");
    reject<std::overflow_error>([&] { upload_add((std::numeric_limits<size_t>::max)(), 1); }, "payload addition overflow rejected");
    const size_t maximum = (size_t) (std::numeric_limits<int64_t>::max)();
    validate_upload_range(maximum - 4, 4, 0, 0);
    reject<std::overflow_error>([&] { validate_upload_range(maximum - 4, 5, 0, 0); }, "contiguous file range overflow rejected");
    reject<std::overflow_error>([&] { validate_upload_range(maximum - 7, 8, 8, 4); }, "TP file range overflow rejected");
    require(upload_option(nullptr, 16, 1, 1024) == 16 && upload_option("0", 16, 1, 1024) == 16,
            "absent/zero options retain legacy defaults");
    require(upload_option("8", 16, 1024, 65536) == 8192, "valid option unit conversion exact");
    reject<std::overflow_error>([&] { upload_option("65", 16, 1024, 65536); }, "option multiplication overflow rejected");
    reject<std::invalid_argument>([&] { upload_option("8bad", 16, 1, 1024); }, "malformed option rejected");
    l.empty();
    l.limit = 8192;
    {
        std::atomic<int> attempted{0}, admitted{0}; std::atomic<bool> release{false};
        std::vector<std::thread> threads;
        for (int i = 0; i < 8; ++i) threads.emplace_back([&] {
            try {
                auto plan = plan_upload(std::vector<job>{{4096}}, 1, 4096); arena workspace(plan);
                ++admitted; ++attempted;
                while (!release.load()) std::this_thread::yield();
            } catch (const upload_budget_refused &) { ++attempted; }
        });
        while (attempted.load() != 8) std::this_thread::yield();
        require(admitted == 2 && l.physical == 8192 && l.owned == 8192, "concurrent planners cannot overbook payload");
        release = true; for (auto & thread : threads) thread.join();
    }
    l.empty(); require(tsg::SharedCacheCharge::detach(&l), "final detach after all arenas and plans end");
    std::printf("PASS %d GLM upload preplanning checks\n", checks.load());
}
