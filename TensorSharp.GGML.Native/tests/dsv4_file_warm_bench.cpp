// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Host-only storage benchmark for the DeepSeek V4 / V4.1 load-time warm passes,
// run against a byte range of a real checkpoint shard. Linux only. It EVICTS the
// named range from the page cache (posix_fadvise DONTNEED) before every arm, so
// run it only while nothing else is reading that file.
//
//   GgmlOpsDsv4FileWarmBench FILE OFFSET BYTES [--threads N] [--repeats N]
//       [--modes pread,touch,engram] [--rows N] [--row-bytes N] [--row-origin N]
//       [--drop-cost] [--hint-wait SECONDS]
//
// Modes (each preceded by eviction with mincore()=0 verified):
//   pread   tsg_dsv4::warm_file_ranges, the default warm (TS_DSV4_WARM_PREAD=1)
//   touch   tsg_dsv4::touch_prefault_spans, the TS_DSV4_WARM_PREAD=0 expert prefault
//   engram  tsg_dsv41::engram_io_pool::warm, the TS_DSV4_WARM_PREAD=0 Engram warm
//   madv_willneed / fadv_willneed / readahead
//           the read-ahead hints instead of reads: resident_after is measured
//           --hint-wait seconds (default 10) after the call
// After each pread arm, on the now-resident range:
//   first_touch   a fresh mapping's one-byte-per-page walk (minor faults only):
//                 what the first prefill pays when the page tables are empty
//   populate      warm_file_ranges(populate) on a fresh mapping: every block skipped
//                 as resident and mapped by the per-page walk the prefault uses
//   madv_populate the same range mapped with madvise(MADV_POPULATE_READ) instead
//   rows          N random row reads through a fresh MADV_RANDOM mapping
//   rewarm        warm_file_ranges again: every block skipped as resident
// --drop-cost times POSIX_FADV_DONTNEED on resident 64 MiB chunks (the cost the
// upload pays per chunk when TS_DSV4_LOAD_DROP_CACHE drops).
#include "../dsv4_file_warm.h"
#include "../dsv41_engram_io.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#if !defined(MADV_POPULATE_READ)
#define MADV_POPULATE_READ 22 // Linux 5.14 uapi value; older headers lack the name
#endif

using clock_type = std::chrono::steady_clock;
constexpr double GiB = 1024.0 * 1024.0 * 1024.0;

static double seconds_since(clock_type::time_point t0)
{
    return std::chrono::duration<double>(clock_type::now() - t0).count();
}

static std::string loadavg()
{
    char buf[64] = { 0 };
    if (FILE * f = std::fopen("/proc/loadavg", "r"))
    {
        if (!std::fgets(buf, sizeof(buf), f)) buf[0] = 0;
        std::fclose(f);
    }
    std::string s(buf);
    const size_t sp = s.find(' ');
    return sp == std::string::npos ? s : s.substr(0, sp);
}

struct faults { long major = 0, minor = 0; };
static faults fault_counts(int who)
{
    rusage ru;
    getrusage(who, &ru);
    return { ru.ru_majflt, ru.ru_minflt };
}

struct map_view
{
    void * base = MAP_FAILED;
    size_t size = 0;
    explicit map_view(const char * path)
    {
        const int fd = open(path, O_RDONLY);
        struct stat st;
        if (fd < 0 || fstat(fd, &st) != 0) { std::perror(path); std::exit(2); }
        size = (size_t) st.st_size;
        base = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        close(fd);
        if (base == MAP_FAILED) { std::perror("mmap"); std::exit(2); }
    }
    ~map_view() { munmap(base, size); }
    const char * at(uint64_t off) const { return (const char *) base + off; }
};

static double residency(const char * p, uint64_t bytes)
{
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const uintptr_t start = (uintptr_t) p / page * page;
    const size_t len = (size_t) ((uintptr_t) p + bytes - start);
    std::vector<unsigned char> vec((len + page - 1) / page);
    if (mincore((void *) start, len, vec.data()) != 0) return -1;
    size_t in = 0;
    for (unsigned char v : vec) in += v & 1;
    return double(in) / double(vec.size());
}

static double evict(const char * path, uint64_t off, uint64_t bytes, const map_view & view)
{
    double res = 1;
    for (int attempt = 0; attempt < 5 && res > 0; ++attempt)
    {
        const int fd = open(path, O_RDONLY);
        if (fd >= 0)
        {
            posix_fadvise(fd, (off_t) off, (off_t) bytes, POSIX_FADV_DONTNEED);
            close(fd);
        }
        res = residency(view.at(off), bytes);
        if (res > 0) std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return res;
}

int main(int argc, char ** argv)
{
    if (argc < 4)
    {
        std::fprintf(stderr, "usage: %s FILE OFFSET BYTES [--threads N] [--repeats N] [--modes pread,touch,engram] "
                             "[--rows N] [--row-bytes N] [--row-origin N] [--drop-cost] [--hint-wait S]\n", argv[0]);
        return 2;
    }
    const char * path = argv[1];
    const uint64_t off = std::strtoull(argv[2], nullptr, 10), bytes = std::strtoull(argv[3], nullptr, 10);
    int threads = 16, repeats = 3;
    size_t rows = 2000;
    uint64_t row_bytes = 144, row_origin = 192;
    bool drop_cost = false;
    double hint_wait = 10.0;
    std::vector<std::string> modes = { "pread", "touch", "engram" };
    for (int i = 4; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--repeats") repeats = std::atoi(next().c_str());
        else if (a == "--rows") rows = (size_t) std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--row-bytes") row_bytes = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--row-origin") row_origin = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--drop-cost") drop_cost = true;
        else if (a == "--hint-wait") hint_wait = std::atof(next().c_str());
        else if (a == "--modes")
        {
            modes.clear();
            std::string list = next();
            for (size_t p = 0; p <= list.size(); )
            {
                size_t q = list.find(',', p);
                if (q == std::string::npos) q = list.size();
                if (q > p) modes.push_back(list.substr(p, q - p));
                p = q + 1;
            }
        }
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    {
        map_view probe(path);
        if (off + bytes > probe.size) { std::fprintf(stderr, "range beyond the end of %s\n", path); return 2; }
    }
    const std::vector<std::string> paths = { path };
    std::mt19937_64 rng(4101);

    for (int rep = 0; rep < repeats; ++rep)
    {
        std::vector<std::string> order = modes;
        if (rep % 2) std::reverse(order.begin(), order.end());
        for (const std::string & mode : order)
        {
            map_view view(path);
            const double before = evict(path, off, bytes, view);
            const faults f0 = fault_counts(RUSAGE_SELF);
            const std::string load0 = loadavg();
            const auto t0 = clock_type::now();
            uint64_t read = 0, resident = 0;
            bool ok = true;
            std::string error;
            if (mode == "pread")
            {
                tsg_dsv4::file_warm_options o;
                o.threads = threads;
                const auto r = tsg_dsv4::warm_file_ranges(paths, { { 0, off, bytes, view.at(off) } }, o);
                ok = r.ok; error = r.error; read = r.bytes_read; resident = r.bytes_resident;
            }
            else if (mode == "touch")
            {
                tsg_dsv4::touch_prefault_spans({ { (const volatile char *) view.at(off), (size_t) bytes } }, threads);
            }
            else if (mode == "engram")
            {
                tsg_dsv41::engram_io_pool pool((unsigned) std::min(threads, 32));
                pool.warm(view.at(off), (size_t) bytes);
            }
            else if (mode == "madv_willneed" || mode == "fadv_willneed" || mode == "readahead")
            {
                // Read-ahead hints instead of reads: issue the hint over the whole
                // range, then give the kernel `hint_wait` seconds before measuring
                // what became resident.
                const size_t page = (size_t) sysconf(_SC_PAGESIZE);
                const uintptr_t start = (uintptr_t) view.at(off) / page * page;
                const int fd = open(path, O_RDONLY);
                int rc = 0;
                if (mode == "madv_willneed")
                    rc = madvise((void *) start, (size_t) ((uintptr_t) view.at(off) + bytes - start), MADV_WILLNEED);
                else if (mode == "fadv_willneed")
                    rc = posix_fadvise(fd, (off_t) off, (off_t) bytes, POSIX_FADV_WILLNEED);
                else
                    rc = (int) readahead(fd, (off64_t) off, (size_t) bytes);
                close(fd);
                ok = rc == 0;
                if (!ok) error = std::strerror(errno);
                std::this_thread::sleep_for(std::chrono::duration<double>(hint_wait));
            }
            else { std::fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 2; }
            const double secs = seconds_since(t0);
            const faults f1 = fault_counts(RUSAGE_SELF);
            const double after = residency(view.at(off), bytes);
            std::printf("{\"file\":\"%s\",\"offset\":%" PRIu64 ",\"gib\":%.3f,\"mode\":\"%s\",\"repeat\":%d,\"threads\":%d,"
                        "\"resident_before\":%.6f,\"seconds\":%.3f,\"gib_per_s\":%.3f,\"resident_after\":%.6f,"
                        "\"bytes_read\":%" PRIu64 ",\"bytes_resident\":%" PRIu64 ",\"major_faults\":%ld,\"minor_faults\":%ld,"
                        "\"load_start\":\"%s\",\"load_end\":\"%s\",\"ok\":%s,\"error\":\"%s\"}\n",
                        path, off, bytes / GiB, mode.c_str(), rep, threads, before, secs, bytes / GiB / secs, after,
                        read, resident, f1.major - f0.major, f1.minor - f0.minor, load0.c_str(), loadavg().c_str(),
                        ok ? "true" : "false", error.c_str());
            std::fflush(stdout);
            if (mode != "pread" || !ok) continue;

            // Page tables: the first access after pread, through a fresh mapping.
            {
                map_view fresh(path);
                const faults g0 = fault_counts(RUSAGE_SELF);
                const auto t1 = clock_type::now();
                tsg_dsv4::touch_prefault_spans({ { (const volatile char *) fresh.at(off), (size_t) bytes } }, threads);
                const double s = seconds_since(t1);
                const faults g1 = fault_counts(RUSAGE_SELF);
                std::printf("{\"mode\":\"first_touch\",\"repeat\":%d,\"threads\":%d,\"seconds\":%.3f,\"s_per_gib\":%.4f,"
                            "\"major_faults\":%ld,\"minor_faults\":%ld}\n", rep, threads, s, s / (bytes / GiB),
                            g1.major - g0.major, g1.minor - g0.minor);
            }
            {
                map_view fresh(path);
                tsg_dsv4::file_warm_options o;
                o.threads = threads;
                o.populate = true;
                const faults g0 = fault_counts(RUSAGE_SELF);
                const auto r = tsg_dsv4::warm_file_ranges(paths, { { 0, off, bytes, fresh.at(off) } }, o);
                const faults g1 = fault_counts(RUSAGE_SELF);
                std::printf("{\"mode\":\"populate\",\"repeat\":%d,\"threads\":%d,\"seconds\":%.3f,\"s_per_gib\":%.4f,"
                            "\"bytes_read\":%" PRIu64 ",\"major_faults\":%ld,\"minor_faults\":%ld,\"ok\":%s}\n",
                            rep, r.threads, r.seconds, r.seconds / (bytes / GiB),
                            r.bytes_read, g1.major - g0.major, g1.minor - g0.minor, r.ok ? "true" : "false");
            }
            {
                map_view fresh(path);
                constexpr uint64_t block = 64ull << 20;
                const faults g0 = fault_counts(RUSAGE_SELF);
                const auto t1 = clock_type::now();
                std::atomic<uint64_t> cursor(0);
                std::atomic<int> errors(0);
                std::vector<std::thread> pool;
                for (int t = 0; t < threads; ++t)
                    pool.emplace_back([&]() {
                        for (;;)
                        {
                            const uint64_t o = cursor.fetch_add(block);
                            if (o >= bytes) break;
                            const size_t page = (size_t) sysconf(_SC_PAGESIZE);
                            const uintptr_t p = (uintptr_t) fresh.at(off + o), start = p / page * page;
                            const size_t len = (size_t) (p + std::min(block, bytes - o) - start);
                            if (madvise((void *) start, len, MADV_POPULATE_READ) != 0) ++errors;
                        }
                    });
                for (auto & th : pool) th.join();
                const double s = seconds_since(t1);
                const faults g1 = fault_counts(RUSAGE_SELF);
                std::printf("{\"mode\":\"madv_populate\",\"repeat\":%d,\"threads\":%d,\"seconds\":%.3f,\"s_per_gib\":%.4f,"
                            "\"errors\":%d,\"major_faults\":%ld,\"minor_faults\":%ld}\n",
                            rep, threads, s, s / (bytes / GiB), errors.load(), g1.major - g0.major, g1.minor - g0.minor);
            }
            // Sparse rows through a MADV_RANDOM mapping, as the Engram lookup reads them.
            {
                map_view fresh(path);
                const size_t page = (size_t) sysconf(_SC_PAGESIZE);
                const uintptr_t start = (uintptr_t) fresh.at(off) / page * page;
                posix_madvise((void *) start, (size_t) ((uintptr_t) fresh.at(off) + bytes - start), POSIX_MADV_RANDOM);
                const uint64_t first_row = off <= row_origin ? 0 : (off - row_origin + row_bytes - 1) / row_bytes;
                const uint64_t end_row = (off + bytes - row_origin) / row_bytes;
                std::uniform_int_distribution<uint64_t> pick(first_row, end_row - 1);
                std::vector<char> row((size_t) row_bytes);
                double total = 0, worst = 0;
                uint64_t sum = 0;
                const faults g0 = fault_counts(RUSAGE_THREAD);
                for (size_t i = 0; i < rows; ++i)
                {
                    const uint64_t at = row_origin + pick(rng) * row_bytes;
                    const auto t1 = clock_type::now();
                    std::memcpy(row.data(), fresh.at(at), (size_t) row_bytes);
                    const double s = seconds_since(t1);
                    total += s;
                    worst = std::max(worst, s);
                    sum += (unsigned char) row[0];
                }
                const faults g1 = fault_counts(RUSAGE_THREAD);
                std::printf("{\"mode\":\"rows\",\"repeat\":%d,\"rows\":%zu,\"row_bytes\":%" PRIu64 ",\"avg_ms\":%.5f,"
                            "\"max_ms\":%.4f,\"major_faults\":%ld,\"minor_faults\":%ld,\"checksum\":%" PRIu64 "}\n",
                            rep, rows, row_bytes, total / rows * 1e3, worst * 1e3, g1.major - g0.major, g1.minor - g0.minor, sum);
            }
            {
                tsg_dsv4::file_warm_options o;
                o.threads = threads;
                const auto r = tsg_dsv4::warm_file_ranges(paths, { { 0, off, bytes, view.at(off) } }, o);
                std::printf("{\"mode\":\"rewarm\",\"repeat\":%d,\"threads\":%d,\"seconds\":%.4f,\"gib_per_s\":%.1f,"
                            "\"bytes_read\":%" PRIu64 ",\"bytes_resident\":%" PRIu64 ",\"ok\":%s}\n",
                            rep, r.threads, r.seconds, bytes / GiB / r.seconds, r.bytes_read, r.bytes_resident,
                            r.ok ? "true" : "false");
            }
            if (drop_cost)
            {
                constexpr uint64_t chunk = 64ull << 20;
                const int fd = open(path, O_RDONLY);
                const auto t1 = clock_type::now();
                uint64_t n = 0;
                for (uint64_t o = 0; o + chunk <= bytes; o += chunk, ++n)
                    posix_fadvise(fd, (off_t) (off + o), (off_t) chunk, POSIX_FADV_DONTNEED);
                const double s = seconds_since(t1);
                close(fd);
                std::printf("{\"mode\":\"drop_cost\",\"repeat\":%d,\"chunks\":%" PRIu64 ",\"ms_per_chunk\":%.3f,"
                            "\"resident_after\":%.6f}\n", rep, n, s / std::max<uint64_t>(n, 1) * 1e3,
                            residency(view.at(off), bytes));
            }
            std::fflush(stdout);
        }
    }
    return 0;
}
