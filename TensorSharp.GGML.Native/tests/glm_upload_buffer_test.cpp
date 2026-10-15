// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../glm_upload_buffer.h"
#include "../glm_file_strip.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <new>
#include <string>
#include <chrono>
#include <filesystem>

namespace {
int checks = 0;
void require(bool ok, const char * message) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}
struct ledger {
    size_t limit = 128, owned = 0, peak = 0, physical = 0;
    uint64_t next = 0;
    bool fail_commit = false, fail_allocate = false;
    struct ticket { size_t bytes; bool committed = false; size_t physical = 0; };
    std::map<uint64_t, ticket> tickets;
    std::map<uint8_t *, uint64_t> pointers;
    static ledger * current;
    static uint64_t reserve(void * context, int rank, int kind, int64_t bytes) {
        auto & l = *static_cast<ledger *>(context);
        require(rank == 0 && kind == 3 && bytes > 0, "host payload allocation class");
        if (uint64_t(bytes) > l.limit - l.owned) return 0;
        l.tickets.emplace(++l.next, ticket{size_t(bytes)});
        l.owned += size_t(bytes);
        l.peak = (std::max)(l.peak, l.owned);
        return l.next;
    }
    static int commit(void * context, uint64_t id) {
        auto & l = *static_cast<ledger *>(context);
        if (l.fail_commit) return 0;
        auto & t = l.tickets.at(id);
        require(!t.committed && t.physical == t.bytes, "commit follows physical allocation");
        t.committed = true;
        return 1;
    }
    static void release(void * context, uint64_t id) {
        auto & l = *static_cast<ledger *>(context);
        auto & t = l.tickets.at(id);
        require(t.physical == 0, "credit returned only after physical free");
        l.owned -= t.bytes;
        l.tickets.erase(id);
    }
    bool attach() { return tsg::SharedCacheCharge::attach(this, reserve, commit, release, false, true); }
    void empty() {
        require(owned == 0 && physical == 0 && tickets.empty() && pointers.empty(), "no leaked payload or charge");
    }
};
ledger * ledger::current = nullptr;
struct allocator {
    static uint8_t * allocate(size_t bytes) {
        auto & l = *ledger::current;
        auto & t = l.tickets.at(l.next);
        require(t.bytes == bytes && !t.committed && t.physical == 0, "reservation precedes physical allocation");
        if (l.fail_allocate) throw std::bad_alloc();
        uint8_t * p = new uint8_t[bytes];
        l.pointers.emplace(p, l.next);
        t.physical = bytes;
        l.physical += bytes;
        return p;
    }
    static void release(uint8_t * p) noexcept {
        auto & l = *ledger::current;
        auto & t = l.tickets.at(l.pointers.at(p));
        delete[] p;
        l.physical -= t.physical;
        t.physical = 0;
        l.pointers.erase(p);
    }
};
using buffer = tsg_glm::upload_buffer<allocator>;
}

int main() {
    ledger l;
    ledger::current = &l;
    {
        tsg_glm::upload_buffer<> existing;
        existing.resize(8);
        require(!l.attach(), "cannot attach a quota over already-live staging");
        existing.clear();
    }
    require(l.attach(), "attach after old staging released");
    {
        buffer a, b;
        a.resize(48); b.resize(64);
        require(l.owned == 112 && l.physical == 112, "two workers share the same host capacity");
        auto * original = a.data();
        a.resize(12); a.resize(48);
        require(a.data() == original && a.size() == 48 && l.owned == 112, "warm worker reuses exact capacity");
        require(!tsg::SharedCacheCharge::detach(&l), "live upload prevents budget detach");
        bool refused = false;
        try { buffer c; c.resize(17); } catch (const std::runtime_error &) { refused = true; }
        require(refused && l.owned == 112 && l.physical == 112, "over-capacity worker does not allocate");
        b.clear();
        a.resize(96);
        require(l.owned == 96 && l.physical == 96 && l.peak == 112, "growth frees old capacity before reserving replacement");
    }
    l.empty();
    for (bool commit_failure : {false, true}) {
        buffer a;
        a.resize(32);
        l.fail_allocate = !commit_failure;
        l.fail_commit = commit_failure;
        bool refused = false;
        try { a.resize(64); }
        catch (const std::bad_alloc &) { refused = !commit_failure; }
        catch (const std::runtime_error &) { refused = commit_failure; }
        require(refused && a.size() == 0 && a.data() == nullptr, "failed growth leaves an empty reusable workspace");
        l.empty();
        l.fail_allocate = l.fail_commit = false;
        a.resize(64);
        require(a.size() == 64 && l.owned == 64, "retry after allocation/commit failure");
    }
    l.empty();
    {
        buffer a;
        a.resize(32);
        bool refused = false;
        try { a.resize(129); } catch (const std::runtime_error &) { refused = true; }
        require(refused && a.size() == 0, "budget refusal after growth releases obsolete staging");
        l.empty();
    }
    {
        // Compile and exercise the TP gather's real scratch interface. An I/O
        // validation failure must leave any retained workspace charged.
        buffer scratch;
        scratch.resize(16);
        uint8_t output[8] = {};
        require(!tsg_glm::read_file_strips(nullptr, 0, 8, 4, 8, output, scratch), "invalid file rejected without uncharging scratch");
        require(l.owned == 16, "failed strip read retains physical scratch charge");
    }
    l.empty();
    {
        const auto directory = std::filesystem::temp_directory_path() /
            ("glm-upload-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        require(std::filesystem::create_directory(directory), "create isolated strip fixture");
        const auto path = directory / "strips.bin";
        struct cleanup {
            std::filesystem::path path, directory;
            ~cleanup() { std::filesystem::remove(path); std::filesystem::remove(directory); }
        } cleanup_file{path, directory};
        std::unique_ptr<FILE, decltype(&std::fclose)> file(std::fopen(path.string().c_str(), "w+b"), &std::fclose);
        require(bool(file), "open strip fixture");
        uint8_t source[64], output[12] = {};
        for (size_t i = 0; i < sizeof(source); ++i) source[i] = uint8_t(i);
        require(std::fwrite(source, 1, sizeof(source), file.get()) == sizeof(source) &&
                std::fflush(file.get()) == 0, "write strip fixture");
        buffer scratch;
        scratch.resize(16);
        require(tsg_glm::read_file_strips(file.get(), 5, 9, 3, sizeof(output), output, scratch, 32), "read TP strips with accounted workspace");
        require(scratch.size() == 30 && l.owned == 30 && l.physical == 30, "strip growth keeps exact payload charge");
        for (size_t i = 0; i < sizeof(output); ++i)
            require(output[i] == uint8_t(5 + (i / 3) * 9 + i % 3), "gathered source bytes unchanged");
        require(!tsg_glm::read_file_strips(file.get(), 61, 9, 3, sizeof(output), output, scratch, 32), "truncated strips refuse partial data");
        require(l.owned == 30 && l.physical == 30, "read failure retains workspace owner until destruction");
    }
    l.empty();
    require(tsg::SharedCacheCharge::detach(&l), "detach after all uploads ended");
    std::printf("PASS: %d upload workspace checks\n", checks);
}
