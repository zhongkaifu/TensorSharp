// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../dsv4_file_warm.h"
#include <condition_variable>
#include <map>
#include <thread>

namespace {
std::atomic<int> checks{0};
void check(bool value, const char * text) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL %s\n", text); std::abort(); }
}
template<class Error, class Action> void throws(Action action, const char * text) {
    bool caught = false;
    try { action(); } catch (const Error &) { caught = true; }
    check(caught, text);
}
struct ledger {
    std::mutex mutex;
    size_t limit = 0, owned = 0, physical = 0, allocations = 0, peak = 0;
    bool fail_allocate = false, null_allocate = false, fail_commit = false;
    uint64_t next = 0;
    struct entry { size_t bytes, physical = 0; std::thread::id thread; };
    std::map<uint64_t, entry> entries;
    std::map<unsigned char *, uint64_t> pointers;
    static ledger * active;
    static uint64_t reserve(void * opaque, int rank, int kind, int64_t bytes) {
        auto & l = *static_cast<ledger *>(opaque); std::lock_guard<std::mutex> lock(l.mutex);
        check(rank == 0 && kind == 3 && bytes > 0, "warm host ticket kind/rank");
        if (uint64_t(bytes) > l.limit - l.owned) return 0;
        l.entries.emplace(++l.next, entry{size_t(bytes), 0, std::this_thread::get_id()});
        l.owned += size_t(bytes); l.peak = (std::max)(l.peak, l.owned);
        return l.next;
    }
    static int commit(void * opaque, uint64_t id) {
        auto & l = *static_cast<ledger *>(opaque); std::lock_guard<std::mutex> lock(l.mutex);
        const auto & t = l.entries.at(id);
        check(t.bytes == t.physical, "exact aggregate allocated before commit");
        return !l.fail_commit;
    }
    static void release(void * opaque, uint64_t id) {
        auto & l = *static_cast<ledger *>(opaque); std::lock_guard<std::mutex> lock(l.mutex);
        const auto t = l.entries.at(id);
        check(t.physical == 0, "physical payload freed before credit release");
        l.owned -= t.bytes; l.entries.erase(id);
    }
    bool attach() { return tsg::SharedCacheCharge::attach(this, reserve, commit, release, false, true); }
    void empty() { check(owned == 0 && physical == 0 && entries.empty() && pointers.empty(), "no warm ownership leak"); }
};
ledger * ledger::active = nullptr;
struct allocator {
    static unsigned char * allocate(size_t bytes) {
        auto & l = *ledger::active; std::lock_guard<std::mutex> lock(l.mutex);
        if (l.fail_allocate) throw std::bad_alloc();
        if (l.null_allocate) return nullptr;
        for (auto & [id, t] : l.entries) {
            if (t.thread != std::this_thread::get_id() || t.physical || t.bytes != bytes) continue;
            auto p = std::make_unique<unsigned char[]>(bytes);
            l.pointers.emplace(p.get(), id);
            t.physical = bytes; l.physical += bytes; ++l.allocations;
            return p.release();
        }
        throw std::runtime_error("payload allocated without preceding reservation");
    }
    static void release(unsigned char * p) noexcept {
        auto & l = *ledger::active; std::lock_guard<std::mutex> lock(l.mutex);
        auto & t = l.entries.at(l.pointers.at(p));
        delete[] p; l.physical -= t.physical; t.physical = 0; l.pointers.erase(p);
    }
};
using namespace tsg_dsv4;
using arena = file_warm_arena<allocator>;

void portable(ledger & l) {
    {
        auto p = plan_file_warm_budget(uint64_t(1) << 30, 16, 64 << 20, 4096, true);
        check(!p.accounted && !p.skipped && p.workers == 16 && p.block == 64 << 20,
            "unscoped geometry remains legacy16x64MiB");
        arena a(p);
        check(a.payload(0) == nullptr && l.allocations == 0, "unscoped payload stays lazy");
        check(!l.attach(), "active unscoped warming blocks scope attachment");
    }
    check(l.attach(), "scope attaches after unscoped plan release");
    {
        auto p = plan_file_warm_budget(0, 0, 0, 0, true);
        check(!p.reservation && !p.workers, "empty ranges need no ticket");
    }
    l.limit = 8 << 20;
    {
        auto p = plan_file_warm_budget(uint64_t(1) << 30, 16, 64 << 20, 4096, true);
        check(p.accounted && p.workers == 1 && p.block == 4 << 20,
            "8MiB quota includes residency scratch, so8MiB payload alone cannot fit");
        check(p.residency == 1025 && p.aggregate == (4 << 20) + 1025 && !l.physical,
            "exact maximum unaligned mincore bytes included before allocate");
        arena a(p);
        auto s = a.scratch(0); s.resize(p.residency); s[p.residency - 1] = 7;
        check(a.payload(0) + p.block == s.data() && l.physical == p.aggregate, "fixed scratch is in same charged arena");
        throws<std::out_of_range>([&] { s.resize(p.residency + 1); }, "fixed scratch cannot grow");
        throws<std::out_of_range>([&] { a.payload(1); }, "worker cannot escape its slice");
        check(!tsg::SharedCacheCharge::detach(&l), "live arena blocks scope detach");
    }
    l.empty();
    l.limit = 4096;
    {
        auto p = plan_file_warm_budget(8192, 1, 8192, 4096, true);
        check(p.skipped && !p.reservation && p.refusals == 2, "residency scratch prevents hidden2-byte overage");
    }
    {
        auto p = plan_file_warm_budget(8192, 1, 8192, 4096, false);
        check(!p.skipped && p.aggregate == 4096, "unmapped reads need no mincore scratch");
        arena a(p);
    }
    l.empty();
    l.limit = 1;
    {
        auto p = plan_file_warm_budget(8192, 16, 64 << 20, 4096, false);
        check(p.skipped && !p.reservation, "too-small quota skips optional warming");
    }
    l.limit = 32768;
    for (int failure = 0; failure < 3; ++failure) {
        l.fail_allocate = failure == 0; l.null_allocate = failure == 1; l.fail_commit = failure == 2;
        if (failure < 2) throws<std::bad_alloc>([&] { auto p = plan_file_warm_budget(8192, 1, 8192, 4096, true); arena a(p); }, "allocation failure releases ticket");
        else throws<std::runtime_error>([&] { auto p = plan_file_warm_budget(8192, 1, 8192, 4096, true); arena a(p); }, "commit failure frees payload then ticket");
        l.empty();
    }
    l.fail_allocate = l.null_allocate = l.fail_commit = false;
    {
        auto p = plan_file_warm_budget(8192, 1, 8192, 4096, true); arena a(p);
        check(l.physical == 8195, "retry succeeds after every failure");
    }
    l.empty();
    throws<std::invalid_argument>([] { plan_file_warm_budget(1, 0, 1, 4096, false); }, "zero workers rejected");
    throws<std::overflow_error>([] { plan_file_warm_budget(UINT64_MAX, 16, (std::numeric_limits<size_t>::max)(), 4096, true); }, "overflow cannot bypass quota");
    throws<std::overflow_error>([] { merge_file_warm_ranges({{0, UINT64_MAX - 7, 16, nullptr}}); }, "range endpoint overflow rejected");
    throws<std::overflow_error>([] { plan_file_warm({{0, 0, UINT64_MAX, nullptr}, {1, 0, 1, nullptr}}, 1, 4096); }, "aggregate range overflow rejected");
    // Concurrent reservation is atomic: another warmer cannot borrow a held
    // ticket's bytes or detach the active scope. No timing/sleep assumptions.
    l.limit = 8192;
    std::mutex gate; std::condition_variable changed; bool ready = false, release = false;
    std::thread owner([&] {
        auto p = plan_file_warm_budget(8192, 1, 8192, 4096, false); arena a(p);
        std::unique_lock<std::mutex> lock(gate); ready = true; changed.notify_one();
        changed.wait(lock, [&] { return release; });
    });
    {
        std::unique_lock<std::mutex> lock(gate); changed.wait(lock, [&] { return ready; });
        auto p = plan_file_warm_budget(8192, 1, 8192, 4096, false);
        check(p.skipped, "concurrent warmer refuses held capacity before allocating");
        const auto tickets = l.entries.size(), held = l.owned;
        auto competing_payload = tsg::SharedCacheCharge::reserve(0, 3, 1);
        check(!competing_payload && l.entries.size() == tickets && l.owned == held,
            "other host payload refuses immediately without an orphan reservation");
        check(!tsg::SharedCacheCharge::detach(&l), "concurrent holder protects callback lifetime");
        release = true; changed.notify_one();
    }
    owner.join(); l.empty();
    check(tsg::SharedCacheCharge::detach(&l), "scope detaches after all work/free");
}

#if !defined(_WIN32)
struct fake_io {
    std::atomic<int> opens{0}, closes{0}, reads{0}, resident_calls{0};
    bool throw_read = false;
    int open_file(const std::string &) { return opens.fetch_add(1); }
    void close_file(int) noexcept { ++closes; }
    bool resident(const void *, uint64_t bytes, file_warm_scratch & scratch) {
        ++resident_calls;
        const size_t pages = (size_t) ((bytes - 1) / file_warm_detail::page_size() + 2);
        scratch.resize(pages); std::memset(scratch.data(), 0, pages);
        return false;
    }
    file_read_attempt read_at(int, uint64_t offset, void * p, size_t bytes) {
        ++reads;
        if (throw_read) throw std::runtime_error("injected warming read exception");
        std::memset(p, int(offset & 255), bytes); return { bytes };
    }
};
struct partial_launcher {
    static std::atomic<int> live;
    template<class Worker> static std::thread start(Worker & worker, int index) {
        if (index == 2) throw std::runtime_error("injected second thread creation failure");
        return std::thread([&worker, index] { ++live; worker(index); --live; });
    }
};
std::atomic<int> partial_launcher::live{0};

void posix(ledger & l) {
    check(l.attach(), "POSIX helper scope attach");
    file_warm_options o; o.threads = 4; o.block_bytes = 8192;
    const std::vector<std::string> paths{"fake-file"};
    std::vector<char> mapping(65536); // Mock mincore; populate is disabled.
    const std::vector<file_warm_range> ranges{{0, 0, mapping.size(), mapping.data()}};
    {
        fake_io io; std::atomic<bool> stop{true}; o.stop = &stop;
        const auto before = l.allocations, tickets = l.next;
        const auto r = file_warm_detail::warm_file_ranges_with_io<fake_io, allocator>(paths, ranges, o, io);
        check(r.ok && r.stopped && !r.threads && io.opens == 0 && io.reads == 0,
            "pre-cancelled warmer starts no file work");
        check(l.allocations == before && l.next == tickets && !l.owned,
            "pre-cancelled warmer reserves no ticket and allocates no payload (metadata excluded)");
        o.stop = nullptr;
    }
    {
        fake_io io;
        throws<std::overflow_error>([&] {
            file_warm_detail::warm_file_ranges_with_io<fake_io, allocator>(paths,
                {{0, UINT64_MAX - 7, 16, nullptr}}, o, io);
        }, "helper rejects wrapped endpoint before I/O");
        throws<std::overflow_error>([&] {
            file_warm_detail::warm_file_ranges_with_io<fake_io, allocator>({"a", "b"},
                {{0, 0, UINT64_MAX, nullptr}, {1, 0, 1, nullptr}}, o, io);
        }, "helper rejects wrapped total before I/O");
        check(!io.opens && !io.reads && !l.owned, "invalid range has no I/O or owner");
    }
    l.limit = 1;
    {
        fake_io io;
        auto r = file_warm_detail::warm_file_ranges_with_io<fake_io, allocator>(paths, ranges, o, io);
        check(r.ok && r.budget_skipped && r.threads == 0 && io.opens == 0 && io.reads == 0,
            "budget skip precedes all file opens, mincore, reads and workers");
        check(!r.bytes_read && !r.bytes_resident && r.workspace_capacity == 0, "skip never claims warmed bytes");
    }
    l.limit = 32768;
    for (int failure = 0; failure < 3; ++failure) {
        fake_io io;
        l.fail_allocate = failure == 0; l.null_allocate = failure == 1; l.fail_commit = failure == 2;
        auto warm = [&] {
            file_warm_detail::warm_file_ranges_with_io<fake_io, allocator>(paths, ranges, o, io);
        };
        // Planning exceptions are part of this helper's contract. GLM and
        // DSV4/TP load boundaries catch std::exception; prove these failures
        // happen before any file/worker activity and release all ownership.
        if (failure < 2) throws<std::bad_alloc>(warm, "helper allocation failure propagates before file work");
        else throws<std::runtime_error>(warm, "helper commit failure propagates before file work");
        check(io.opens == 0 && io.closes == 0 && io.reads == 0 && io.resident_calls == 0,
            "arena planning failure performs no file warming or worker I/O");
        l.empty();
    }
    l.fail_allocate = l.null_allocate = l.fail_commit = false;
    {
        fake_io io;
        auto r = file_warm_detail::warm_file_ranges_with_io<fake_io, allocator>(paths, ranges, o, io);
        check(r.ok && !r.budget_skipped && r.budget_accounted && r.threads == 2,
            "aggregate helper negotiates threads including mincore scratch");
        check(r.bytes_read == 65536 && io.opens == io.closes && io.resident_calls > 0,
            "all demand bytes warm and descriptors close");
    }
    l.empty();
    for (int failure = 0; failure < 3; ++failure) {
        fake_io io; std::atomic<bool> stop{false};
        io.throw_read = failure == 0;
        o.stop = &stop;
        o.on_block = [&](int, int, uint64_t, uint64_t, bool) {
            if (failure == 1) throw std::bad_alloc();
            if (failure == 2) stop.store(true);
        };
        const auto r = file_warm_detail::warm_file_ranges_with_io<fake_io, allocator>(paths, ranges, o, io);
        check(failure == 2 ? r.ok && r.stopped : !r.ok, "read/callback/cancel failure propagated");
        check(io.opens == io.closes, "exception/cancel closes every descriptor"); l.empty();
    }
    o.stop = nullptr; o.on_block = {};
    l.limit = 65536;
    {
        fake_io io;
        const auto r = file_warm_detail::warm_file_ranges_with_io<fake_io, allocator, partial_launcher>(paths, ranges, o, io);
        check(!r.ok && partial_launcher::live == 0, "partially started workers all joined before return");
        check(io.opens == io.closes, "partial startup closes descriptors");
    }
    l.empty();
    {
        fake_io io;
        const auto r = file_warm_detail::warm_file_ranges_with_io<fake_io, allocator>(paths, ranges, o, io);
        check(r.ok && r.bytes_read == 65536 && io.opens == io.closes, "same helper retries after faults");
    }
    l.empty();
    {
        // Actual POSIX mincore/pread/populate with an unaligned mapped range.
        // Cache temperature is uncontrolled: both resident-skip and read are
        // valid, but every byte must be covered under the same exact arena.
        struct temporary_mapping {
            char path[64] = "/tmp/tsg-file-warm-budget-XXXXXX";
            int fd = -1;
            void * data = MAP_FAILED;
            ~temporary_mapping() {
                if (data != MAP_FAILED) munmap(data, 65536);
                if (fd >= 0) close(fd);
                unlink(path);
            }
        } file;
        file.fd = mkstemp(file.path);
        check(file.fd >= 0, "real POSIX temporary file created");
        std::vector<unsigned char> expected(65536);
        for (size_t i = 0; i < expected.size(); ++i) expected[i] = (unsigned char) (i % 251);
        size_t written = 0;
        while (written < expected.size()) {
            const ssize_t n = write(file.fd, expected.data() + written, expected.size() - written);
            check(n > 0, "real fixture data write");
            written += (size_t) n;
        }
        file.data = mmap(nullptr, expected.size(), PROT_READ, MAP_SHARED, file.fd, 0);
        check(file.data != MAP_FAILED, "real POSIX read-only mapping");
        file_warm_detail::system_file_ops io;
        o.populate = true;
        const auto r = file_warm_detail::warm_file_ranges_with_io<file_warm_detail::system_file_ops, allocator>(
            {file.path}, {{0, 1, 65534, (const char *) file.data + 1}}, o, io);
        check(r.ok && r.budget_accounted && !r.budget_skipped && r.workspace_capacity <= l.limit,
            "actual POSIX mincore/pread/populate uses reserved fixed slices");
        check(r.bytes_read + r.bytes_resident == 65534 && std::memcmp(file.data, expected.data(), expected.size()) == 0,
            "actual unaligned mapped data stays exact, regardless of page-cache warmth");
        l.empty();
        // The freshly written file may be entirely resident. Force a second
        // pass through actual pread so resident skip cannot substitute for
        // exercising the accounted payload. This is hot-cache correctness,
        // not a storage-throughput or cold-cache measurement.
        o.skip_resident = false;
        const auto read = file_warm_detail::warm_file_ranges_with_io<file_warm_detail::system_file_ops, allocator>(
            {file.path}, {{0, 1, 65534, (const char *) file.data + 1}}, o, io);
        check(read.ok && read.budget_accounted && !read.budget_skipped && read.workspace_capacity == 32768,
            "actual forced pread uses only the four charged payload slices");
        check(read.bytes_read == 65534 && read.bytes_resident == 0 &&
            std::memcmp(file.data, expected.data(), expected.size()) == 0,
            "actual pread covers every requested byte without resident-skip substitution");
        l.empty();
    }
    l.empty(); check(tsg::SharedCacheCharge::detach(&l), "POSIX helper scope detaches");
}
#endif
} // namespace

int main() {
    ledger l; ledger::active = &l;
    portable(l);
#if !defined(_WIN32)
    posix(l);
    std::printf("file warm budget: %d checks passed (portable and POSIX helper)\n", checks.load());
#else
    // The public Windows path returns before touching the shared registry or
    // constructing payloads. Real Windows mapped-page prefault is unchanged.
    check(l.attach(), "Windows scope attaches"); l.limit = 1;
    const auto allocations = l.allocations;
    const auto result = warm_file_ranges({"unused"}, {{0, 0, 4096, nullptr}}, {});
    check(!result.ok && !result.budget_skipped && l.allocations == allocations && l.owned == 0,
        "Windows unavailable pread path does not reserve or allocate");
    check(tsg::SharedCacheCharge::detach(&l), "Windows scope detaches");
    std::printf("file warm budget: %d checks passed (portable); POSIX helper explicitly untested\n", checks.load());
#endif
}
