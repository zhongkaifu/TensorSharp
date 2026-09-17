// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// deepseek41-host-file-warm: the pread page-cache warm used for the DeepSeek
// V4 / V4.1 host-mapped experts and Engram tables, on temporary files, plus the
// pure load policies next to it (TS_DSV4_WARM_PREAD, TS_HOST_MOE_PIN for DSV4,
// TS_DSV4_LOAD_DROP_CACHE's automatic rule).
#include "../dsv4_file_warm.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace tsg_dsv4;

static int g_failures = 0;
static void check(bool ok, const char * fmt, ...)
{
    if (ok) return;
    ++g_failures;
    std::va_list args;
    va_start(args, fmt);
    std::fprintf(stderr, "FAIL: ");
    std::vfprintf(stderr, fmt, args);
    std::fprintf(stderr, "\n");
    va_end(args);
}

constexpr uint64_t GiB = uint64_t(1) << 30;
constexpr uint64_t MiB = uint64_t(1) << 20;

static uint64_t gib_tenths(double gib) { return (uint64_t) (gib * double(GiB)); }

static void test_policies()
{
    bool invalid = true;
    check(resolve_warm_pread(nullptr, &invalid) && !invalid, "unset TS_DSV4_WARM_PREAD must read with pread");
    check(resolve_warm_pread("", &invalid) && !invalid, "empty TS_DSV4_WARM_PREAD must read with pread");
    check(resolve_warm_pread("1", &invalid) && !invalid, "TS_DSV4_WARM_PREAD=1 must read with pread");
    check(!resolve_warm_pread("0", &invalid) && !invalid, "TS_DSV4_WARM_PREAD=0 must restore the touch walks");
    check(resolve_warm_pread("yes", &invalid) && invalid, "an invalid TS_DSV4_WARM_PREAD must be reported and use the default");

    check(!dsv4_host_expert_pin_requested(nullptr), "unset TS_HOST_MOE_PIN must not pin DSV4 experts");
    check(!dsv4_host_expert_pin_requested(""), "empty TS_HOST_MOE_PIN must not pin DSV4 experts");
    check(!dsv4_host_expert_pin_requested("0"), "TS_HOST_MOE_PIN=0 must not pin");
    check(dsv4_host_expert_pin_requested("1"), "TS_HOST_MOE_PIN=1 must pin DSV4 experts");

    // The A40 lane: 263 GiB uploaded, 48.2 GiB experts + 103 GiB Engram mapped,
    // a 326.9 GiB cgroup.
    const uint64_t upload = gib_tenths(263.0), mapped = gib_tenths(151.2), allowance = gib_tenths(326.9);
    drop_cache_decision d = decide_drop_cache(nullptr, upload, mapped, allowance);
    check(d.drop && d.automatic && d.allowance_known, "263 + 151.2 + 8 GiB against 326.9 GiB must drop");
    const std::string text = describe_drop_cache(d, upload, mapped, allowance);
    check(text.find("263.0 GiB") != std::string::npos && text.find("151.2 GiB") != std::string::npos &&
          text.find("326.9 GiB") != std::string::npos && text.find("8.0 GiB") != std::string::npos,
          "the automatic drop-cache line must name every number: %s", text.c_str());
    check(text.find("dropping") != std::string::npos, "the drop line must say it drops: %s", text.c_str());

    d = decide_drop_cache(nullptr, gib_tenths(80.0), 0, allowance);
    check(!d.drop && d.automatic, "80 + 0 + 8 GiB against 326.9 GiB must keep the cache");
    const std::string keep = describe_drop_cache(d, gib_tenths(80.0), 0, allowance);
    check(keep.find("keeping") != std::string::npos && keep.find("80.0 GiB") != std::string::npos &&
          keep.find("326.9 GiB") != std::string::npos, "the keep line must name the numbers: %s", keep.c_str());

    d = decide_drop_cache(nullptr, upload, mapped, 0);
    check(!d.drop && d.automatic && !d.allowance_known, "an unknown allowance must keep the cache");
    check(describe_drop_cache(d, upload, mapped, 0).find("unknown") != std::string::npos,
          "the unknown-allowance line must say so");

    d = decide_drop_cache("0", upload, mapped, allowance);
    check(!d.drop && !d.automatic, "TS_DSV4_LOAD_DROP_CACHE=0 must keep the cache");
    d = decide_drop_cache("1", gib_tenths(80.0), 0, allowance);
    check(d.drop && !d.automatic, "TS_DSV4_LOAD_DROP_CACHE=1 must drop");
    d = decide_drop_cache("", gib_tenths(80.0), 0, allowance);
    check(!d.drop && d.automatic, "an empty TS_DSV4_LOAD_DROP_CACHE is the automatic rule");
    // Exactly at the boundary the load still fits.
    d = decide_drop_cache(nullptr, 100 * GiB, 18 * GiB, 126 * GiB);
    check(!d.drop, "upload + mapped + headroom == allowance fits");
    d = decide_drop_cache(nullptr, 100 * GiB, 18 * GiB + 1, 126 * GiB);
    check(d.drop, "one byte over the allowance drops");
}

static void test_planning()
{
    static const char arena[8192] = { 0 };
    const char * base = arena;
    const char * other = arena + 7000;
    // Out of order, overlapping, touching, a gap, a mapping disagreement and an empty range.
    std::vector<file_warm_range> ranges = {
        { 1, 4096, 100, nullptr },
        { 0, 192, 808, base + 192 },
        { 0, 1000, 24, base + 1000 },       // touches the first file-0 range
        { 0, 900, 200, base + 900 },        // overlaps both
        { 0, 5000, 10, base + 5000 },       // gap: stays separate
        { 1, 4196, 50, other },             // touches but the mapping disagrees
        { 1, 9000, 0, nullptr },            // empty
    };
    const auto merged = merge_file_warm_ranges(ranges);
    check(merged.size() == 4, "merge produced %zu ranges, expected 4", merged.size());
    if (merged.size() == 4)
    {
        check(merged[0].file == 0 && merged[0].offset == 192 && merged[0].bytes == 1100 - 192 &&
              merged[0].mapped == base + 192, "file 0 [192, 1100) must merge into one range");
        check(merged[1].file == 0 && merged[1].offset == 5000 && merged[1].bytes == 10, "the gapped range stays apart");
        check(merged[2].file == 1 && merged[2].offset == 4096 && merged[2].bytes == 100, "file 1 first range");
        check(merged[3].file == 1 && merged[3].offset == 4196 && merged[3].bytes == 50 &&
              merged[3].mapped == other, "a mapping disagreement must not merge");
    }

    // Byte-balanced contiguous split, crossing range and file boundaries.
    const std::vector<file_warm_range> big = {
        { 0, 192, 10 * MiB + 7, nullptr }, { 0, 20 * MiB, 3 * MiB, nullptr }, { 2, 0, 5 * MiB + 1, nullptr } };
    uint64_t total = 0;
    for (const auto & r : big) total += r.bytes;
    for (int threads : { 1, 2, 3, 4, 7, 16, 64 })
    {
        const auto plan = plan_file_warm(big, threads, MiB);
        const uint64_t n = std::min<uint64_t>((uint64_t) threads, (total + MiB - 1) / MiB);
        check(plan.size() == n, "threads=%d: %zu runs, expected %llu", threads, plan.size(), (unsigned long long) n);
        uint64_t covered = 0, expect_off = big[0].offset;
        size_t ri = 0;
        for (size_t k = 0; k < plan.size(); ++k)
        {
            uint64_t bytes = 0;
            for (const auto & piece : plan[k])
            {
                // Pieces continue exactly where the previous one ended.
                check(piece.range == ri && piece.offset == expect_off, "threads=%d run %zu is not contiguous", threads, k);
                bytes += piece.bytes;
                expect_off = piece.offset + piece.bytes;
                if (expect_off == big[ri].offset + big[ri].bytes && ri + 1 < big.size())
                {
                    ++ri;
                    expect_off = big[ri].offset;
                }
            }
            covered += bytes;
            const uint64_t lo = total / n, hi = lo + 1;
            check(bytes == lo || bytes == hi, "threads=%d run %zu holds %llu bytes, expected %llu or %llu",
                  threads, k, (unsigned long long) bytes, (unsigned long long) lo, (unsigned long long) hi);
        }
        check(covered == total, "threads=%d covered %llu of %llu bytes", threads,
              (unsigned long long) covered, (unsigned long long) total);
    }
    check(plan_file_warm({}, 16, MiB).empty(), "nothing to read plans no threads");
}

#if defined(_WIN32)
int main()
{
    test_policies();
    test_planning();
    if (g_failures) return 1;
    std::puts("dsv4 file warm: policies and planning passed (file I/O checks need POSIX)");
    return 0;
}
#else

struct temp_dir
{
    std::string path;
    std::vector<std::string> files;
    temp_dir()
    {
        const char * tmp = std::getenv("TMPDIR");
        std::string pattern = std::string(tmp && *tmp ? tmp : "/tmp") + "/dsv4-file-warm-XXXXXX";
        std::vector<char> buf(pattern.begin(), pattern.end());
        buf.push_back(0);
        if (!mkdtemp(buf.data())) { std::perror("mkdtemp"); std::exit(2); }
        path = buf.data();
    }
    ~temp_dir()
    {
        for (const auto & f : files) unlink(f.c_str());
        rmdir(path.c_str());
    }
    std::string create(const char * name, uint64_t bytes)
    {
        std::string file = path + "/" + name;
        const int fd = open(file.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
        if (fd < 0) { std::perror("open"); std::exit(2); }
        std::vector<char> chunk(MiB);
        for (uint64_t off = 0; off < bytes; )
        {
            const size_t n = (size_t) std::min<uint64_t>(chunk.size(), bytes - off);
            for (size_t i = 0; i < n; ++i) chunk[i] = (char) ((off + i) * 131 + name[0]);
            if (write(fd, chunk.data(), n) != (ssize_t) n) { std::perror("write"); std::exit(2); }
            off += n;
        }
        fsync(fd);
        close(fd);
        files.push_back(file);
        return file;
    }
};

struct mapping
{
    void * addr = MAP_FAILED;
    size_t bytes = 0;
    explicit mapping(const std::string & file)
    {
        const int fd = open(file.c_str(), O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0) { std::perror("map"); std::exit(2); }
        bytes = (size_t) st.st_size;
        addr = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (addr == MAP_FAILED) { std::perror("mmap"); std::exit(2); }
    }
    ~mapping() { if (addr != MAP_FAILED) munmap(addr, bytes); }
    const char * at(uint64_t off) const { return (const char *) addr + off; }
};

static double residency(const char * p, uint64_t bytes)
{
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const uintptr_t start = (uintptr_t) p / page * page;
    const size_t len = (size_t) ((uintptr_t) p + bytes - start);
    std::vector<unsigned char> vec((len + page - 1) / page);
#if defined(__APPLE__)
    if (mincore((void *) start, len, (char *) vec.data()) != 0) return -1;
#else
    if (mincore((void *) start, len, vec.data()) != 0) return -1;
#endif
    size_t in = 0;
    for (unsigned char v : vec) in += v & 1;
    return vec.empty() ? 1.0 : double(in) / double(vec.size());
}

static bool evict(const std::string & file)
{
#if defined(__linux__)
    const int fd = open(file.c_str(), O_RDONLY);
    if (fd < 0) return false;
    const bool ok = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) == 0;
    close(fd);
    return ok;
#else
    (void) file;
    return false;
#endif
}

static void test_file_warm()
{
    temp_dir dir;
    // Shard-like files: a table from offset 192 to the end (not page aligned), and
    // a file whose ranges merge, gap and end mid-block.
    const uint64_t a_size = 9 * MiB + 12345, b_size = 7 * MiB + 99;
    const std::string a = dir.create("a.gguf", a_size), b = dir.create("b.gguf", b_size);
    const std::vector<std::string> paths = { a, b };
    const mapping ma(a), mb(b);
    const std::vector<file_warm_range> ranges = {
        { 0, 192, a_size - 192, ma.at(192) },
        { 1, 4 * MiB + 3, 2 * MiB, mb.at(4 * MiB + 3) },
        { 1, 5, 2 * MiB, mb.at(5) },
        { 1, 2 * MiB + 5, MiB + 1, mb.at(2 * MiB + 5) },   // touches the previous one
    };
    const auto merged = merge_file_warm_ranges(ranges);
    uint64_t total = 0;
    for (const auto & r : merged) total += r.bytes;

    // Pass 1: every byte read exactly once, across 5 threads and 1 MiB blocks.
    std::mutex mu;
    std::map<int, std::vector<std::pair<uint64_t, uint64_t>>> blocks;
    std::set<int> threads_seen;
    file_warm_options o;
    o.threads = 5;
    o.block_bytes = MiB;
    o.skip_resident = false;
    o.on_block = [&](int thread, int file, uint64_t off, uint64_t len, bool read)
    {
        std::lock_guard<std::mutex> lock(mu);
        check(read, "a block was skipped with skip_resident=false");
        blocks[file].push_back({ off, len });
        threads_seen.insert(thread);
    };
    file_warm_result r = warm_file_ranges(paths, ranges, o);
    check(r.ok && !r.stopped, "pass 1 failed: %s", r.error.c_str());
    check(r.bytes_total == total && r.bytes_read == total && r.bytes_resident == 0,
          "pass 1 read %llu of %llu bytes (%llu resident)", (unsigned long long) r.bytes_read,
          (unsigned long long) total, (unsigned long long) r.bytes_resident);
    check(r.threads == 5 && threads_seen.size() == 5, "pass 1 used %d threads, %zu reported", r.threads, threads_seen.size());
    for (const auto & range : merged)
    {
        std::vector<std::pair<uint64_t, uint64_t>> seen;
        for (const auto & blk : blocks[range.file])
            if (blk.first >= range.offset && blk.first < range.offset + range.bytes) seen.push_back(blk);
        std::sort(seen.begin(), seen.end());
        uint64_t at = range.offset;
        for (const auto & blk : seen)
        {
            check(blk.first == at, "file %d: block at %llu, expected %llu (a byte read twice or never)",
                  range.file, (unsigned long long) blk.first, (unsigned long long) at);
            at = blk.first + blk.second;
        }
        check(at == range.offset + range.bytes, "file %d range [%llu, +%llu) ends coverage at %llu", range.file,
              (unsigned long long) range.offset, (unsigned long long) range.bytes, (unsigned long long) at);
    }
    uint64_t block_bytes = 0;
    for (const auto & f : blocks) for (const auto & blk : f.second) block_bytes += blk.second;
    check(block_bytes == total, "blocks cover %llu bytes, ranges %llu", (unsigned long long) block_bytes,
          (unsigned long long) total);

    // Residency. Linux reports page-cache residency for a file this process owns.
    const double res_a = residency(ma.at(192), a_size - 192), res_b = residency(mb.at(5), 2 * MiB);
#if defined(__linux__)
    const bool strict = true;
#else
    const bool strict = res_a == 1.0 && res_b == 1.0;
    if (!strict)
        std::printf("note: mincore on this platform reported %.3f/%.3f after pread; skip-path checks are informational\n",
                    res_a, res_b);
#endif
    if (strict) check(res_a == 1.0 && res_b == 1.0, "mincore residency after pass 1 is %.3f / %.3f, expected 1.0", res_a, res_b);

    // Pass 2: everything resident, nothing read.
    blocks.clear();
    o.skip_resident = true;
    o.on_block = nullptr;
    r = warm_file_ranges(paths, ranges, o);
    check(r.ok, "pass 2 failed: %s", r.error.c_str());
    if (strict)
        check(r.bytes_read == 0 && r.bytes_resident == total, "pass 2 read %llu bytes, expected 0 (skip path)",
              (unsigned long long) r.bytes_read);
    std::printf("resident re-warm: %llu bytes skipped, %llu read, %.4fs\n", (unsigned long long) r.bytes_resident,
                (unsigned long long) r.bytes_read, r.seconds);

    // Pass 3: evicted, the skip path reads again (where eviction works here).
    const bool evicted = evict(a) && evict(b);
    double left = 0;
    for (const auto & range : merged)
        left = std::max(left, residency((const char *) range.mapped, range.bytes));
    r = warm_file_ranges(paths, ranges, o);
    check(r.ok && r.bytes_read + r.bytes_resident == total, "pass 3 accounted %llu + %llu of %llu bytes",
          (unsigned long long) r.bytes_read, (unsigned long long) r.bytes_resident, (unsigned long long) total);
    // Not necessarily every byte: the kernel's own readahead behind one block's
    // pread can make the next (short) block resident before it is checked, and
    // skipping it then is correct.
    std::printf("after eviction (%.3f left resident): pass 3 read %llu of %llu bytes, %llu already resident\n",
                left, (unsigned long long) r.bytes_read, (unsigned long long) total, (unsigned long long) r.bytes_resident);
    if (evicted && left < 1.0) check(r.bytes_read > 0, "pass 3 after eviction read nothing");
    for (const auto & range : merged)
        if (strict) check(residency((const char *) range.mapped, range.bytes) == 1.0, "pass 3 left file %d cold", range.file);

    // Populate maps every block (resident here, so skipped and then walked).
    o.populate = true;
    r = warm_file_ranges(paths, ranges, o);
    check(r.ok && r.bytes_read + r.bytes_resident == total, "populate pass failed: %s", r.error.c_str());
    o.populate = false;

    // The stop flag: one thread stops after the block in flight.
    {
        std::atomic<bool> stop(false);
        std::atomic<int> after(0);
        file_warm_options s;
        s.threads = 1;
        s.block_bytes = MiB;
        s.skip_resident = false;
        s.stop = &stop;
        s.on_block = [&](int, int, uint64_t, uint64_t, bool) { if (stop.exchange(true)) ++after; };
        r = warm_file_ranges(paths, ranges, s);
        check(r.ok && r.stopped && after == 0 && r.bytes_read == MiB,
              "1 thread: stop read %llu bytes and %d more blocks, expected exactly one block",
              (unsigned long long) r.bytes_read, after.load());
        stop = false;
        std::atomic<int> blocks_after(0);
        s.threads = 4;
        s.on_block = [&](int, int, uint64_t, uint64_t, bool) { if (stop.exchange(true)) ++blocks_after; };
        r = warm_file_ranges(paths, ranges, s);
        check(r.ok && r.stopped && blocks_after <= 3 && r.bytes_read <= 4 * MiB,
              "4 threads: %d blocks after stop (at most the 3 others in flight), %llu bytes",
              blocks_after.load(), (unsigned long long) r.bytes_read);
        stop = true;
        r = warm_file_ranges(paths, ranges, s);
        check(r.ok && r.stopped && r.bytes_read == 0, "a stop set beforehand must read nothing");
    }

    // A truncated file fails with its path and the offset where the data ends.
    {
        const std::string c = dir.create("truncated.gguf", 5 * MiB + 17);
        file_warm_options t;
        t.threads = 2;
        t.block_bytes = MiB;
        r = warm_file_ranges({ c }, { { 0, 192, 8 * MiB, nullptr } }, t);
        char offset[64];
        std::snprintf(offset, sizeof(offset), "offset %llu", (unsigned long long) (5 * MiB + 17));
        check(!r.ok && r.error.find(c) != std::string::npos && r.error.find(offset) != std::string::npos,
              "truncated file error must name the path and %s: '%s'", offset, r.error.c_str());
        std::printf("truncated: %s\n", r.error.c_str());
        r = warm_file_ranges({ dir.path + "/missing.gguf" }, { { 0, 0, MiB, nullptr } }, t);
        check(!r.ok && r.error.find("missing.gguf") != std::string::npos, "a missing file must be named: '%s'",
              r.error.c_str());
        r = warm_file_ranges({ c }, { { 3, 0, MiB, nullptr } }, t);
        check(!r.ok, "a range naming an unknown file must fail");
    }

    // The TS_DSV4_WARM_PREAD=0 control walk leaves the range resident too.
    evict(a);
    touch_prefault_spans({ { (const volatile char *) ma.at(192), (size_t) (a_size - 192) } }, 3, (size_t) (2 * MiB));
    const double touched = residency(ma.at(192), a_size - 192);
    if (strict) check(touched == 1.0, "touch_prefault_spans left residency %.3f", touched);
}

int main()
{
    test_policies();
    test_planning();
    test_file_warm();
    if (g_failures)
    {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::puts("dsv4 file warm: policies, planning, pread coverage, residency skip, stop flag and truncation passed");
    return 0;
}
#endif
