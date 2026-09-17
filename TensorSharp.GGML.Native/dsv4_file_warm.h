// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Reading checkpoint byte ranges into the page cache at load time, and the
// small pure policies the DeepSeek V4 / V4.1 loader applies around that.
//
// WHY PREAD. The host-resident experts (--n-cpu-moe) and the V4.1 Engram tables
// are served straight from the GGUF mapping, and the loader warms them so the
// first requests do not fault them in from storage. It used to warm them by
// touching the mapping one byte per 4 KiB page. On a network filesystem every
// such fault is a SYNCHRONOUS read capped at the mount's read_ahead_kb (128 KiB
// on the MooseFS/FUSE mount of the A40 VMs), so the walk is thousands of tiny
// round trips no matter how many threads share it. Reading the same bytes with
// pread() in 64 MiB blocks - one contiguous range per thread, one descriptor per
// thread - fills the same page cache in large requests. Measured on the seven-A40
// VM (tests/dsv4_file_warm_bench.cpp), 16 threads, 8 GiB of an Engram table
// (shard 00002 @20 GiB) and 8 GiB of experts (shard 00003 @9,002,135,936), each
// arm evicted with mincore()=0 verified, 3-5 repeats: pread 2.24-2.54 GiB/s,
// against 0.62-0.74 GiB/s for the 256 MiB-span prefault walk and 0.63-0.69 for
// the 8 MiB-chunk Engram walk. A resident range re-warms (every block skipped)
// at 64-187 GiB/s.
//
// DO NOT RETRY THE HINTS. MADV_WILLNEED, posix_fadvise(POSIX_FADV_WILLNEED) and
// readahead(2) were each measured on that mount (the bench's madv_willneed,
// fadv_willneed and readahead modes): all three stop at the 128 KiB readahead
// window, leaving 128 KiB (0.0015%) of an evicted 8 GiB range resident ten
// seconds after the call.
//
// PREAD FILLS THE PAGE CACHE, NOT THIS PROCESS'S PAGE TABLES. The touch walk did
// both. The first access through the mapping afterwards is a minor fault (no
// I/O). For a range that is read densely right away (the experts), `populate`
// maps each block once it is cached by reading one byte per page through the
// mapping. On resident pages that walk measured 0.004-0.006 s/GiB at 16 threads,
// against 0.019-0.023 s/GiB for madvise(MADV_POPULATE_READ) on the same ranges
// (the same ~131k fault-around faults per 8 GiB; the madvise path walks every
// page through get_user_pages), and it needs no particular kernel.
//
// MINCORE SKIP. Before reading a block the helper asks mincore() whether the
// block is already fully resident (a warm reload) and skips it if so. Since
// Linux 5.2 mincore() only reports page-cache residency for files the process
// owns or could write; otherwise it reports pages mapped in this process, which
// makes a resident block look cold. That fails safe: the block is re-read from
// the page cache at memory speed.
#pragma once

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace tsg_dsv4 {

constexpr uint64_t file_warm_gib = uint64_t(1) << 30;

// ---------------------------------------------------------------------------
// Load policies (pure; unit-tested in tests/dsv4_file_warm_test.cpp)
// ---------------------------------------------------------------------------

// TS_DSV4_WARM_PREAD: unset or "1" reads warm ranges with pread (the default);
// "0" restores the page-touch walks exactly. Any other value is reported via
// `invalid` and resolves to the default, so a typo is visible but not fatal
// half-way through a load.
inline bool resolve_warm_pread(const char * value, bool * invalid = nullptr)
{
    if (invalid) *invalid = false;
    if (!value || !*value || std::strcmp(value, "1") == 0) return true;
    if (std::strcmp(value, "0") == 0) return false;
    if (invalid) *invalid = true;
    return true;
}

// Whether DSV4 / V4.1 page-locks its host-resident (cpu_moe) experts.
//
// The DeepSeek graph builder assigns every node of an offloaded layer's routed
// experts to the CPU backend (build_moe_host pins mul_mat_id, clamp, swiglu, mul
// and the expert adds). ggml_backend_sched never overrides a user assignment, so
// the op_offload rule that would stream expert WEIGHTS to the GPU never fires:
// only [n_embd, n_tokens] activations cross the bus. Page-locking the 48 GiB of
// experts therefore speeds nothing up here, costs ~0.42 s/GiB at load, and makes
// those pages unevictable. Off unless TS_HOST_MOE_PIN is set to a value that
// does not start with '0' (the shared reader in ggml_ops_host_pin.cu treats the
// same values as "on"). ggml_ops_moe.cpp, which really streams experts for the
// other MoE architectures, keeps its own default (on).
inline bool dsv4_host_expert_pin_requested(const char * value)
{
    return value && value[0] != '\0' && value[0] != '0';
}

// TS_DSV4_LOAD_DROP_CACHE policy for the weight upload: whether each consumed
// chunk's page cache is released once the chunk is on the device.
//   "0"   never drop (the behaviour before the automatic rule)
//   other set value: atoi() != 0 drops, like before
//   unset: drop only when the load cannot fit in the host allowance - the upload
//          bytes, plus the host-mapped weights the later stages read, plus
//          `headroom` - and keep the cache whenever the allowance is unknown (0).
// Dropping unconditionally would make every reload of a GPU-resident checkpoint
// cold, so the automatic rule only fires when the page cache would overflow.
struct drop_cache_decision
{
    bool drop = false;
    bool automatic = false;      // decided by the rule rather than by the variable
    bool allowance_known = false;
};

inline drop_cache_decision decide_drop_cache(const char * value, uint64_t upload_bytes, uint64_t mapped_bytes,
                                             uint64_t allowance, uint64_t headroom = 8 * file_warm_gib)
{
    drop_cache_decision d;
    d.allowance_known = allowance > 0;
    if (value && *value)
    {
        d.drop = std::atoi(value) != 0;
        return d;
    }
    d.automatic = true;
    d.drop = allowance > 0 && upload_bytes + mapped_bytes + headroom > allowance;
    return d;
}

inline std::string describe_drop_cache(const drop_cache_decision & d, uint64_t upload_bytes, uint64_t mapped_bytes,
                                       uint64_t allowance, uint64_t headroom = 8 * file_warm_gib)
{
    char buf[512];
    const double g = double(file_warm_gib);
    if (!d.automatic)
        std::snprintf(buf, sizeof(buf), "%s each uploaded chunk's page cache (TS_DSV4_LOAD_DROP_CACHE set; "
                      "%.1f GiB upload + %.1f GiB host-mapped + %.1f GiB headroom against a %.1f GiB allowance)",
                      d.drop ? "dropping" : "keeping", upload_bytes / g, mapped_bytes / g, headroom / g, allowance / g);
    else if (!d.allowance_known)
        std::snprintf(buf, sizeof(buf), "keeping each uploaded chunk's page cache (automatic: host memory allowance "
                      "unknown; %.1f GiB upload + %.1f GiB host-mapped; TS_DSV4_LOAD_DROP_CACHE=1 drops)",
                      upload_bytes / g, mapped_bytes / g);
    else
        std::snprintf(buf, sizeof(buf), "%s each uploaded chunk's page cache (automatic: %.1f GiB upload + %.1f GiB "
                      "host-mapped + %.1f GiB headroom %s the %.1f GiB allowance; TS_DSV4_LOAD_DROP_CACHE=%s overrides)",
                      d.drop ? "dropping" : "keeping", upload_bytes / g, mapped_bytes / g, headroom / g,
                      d.drop ? "exceeds" : "fits", allowance / g, d.drop ? "0" : "1");
    return buf;
}

// ---------------------------------------------------------------------------
// Range planning (pure)
// ---------------------------------------------------------------------------

// One byte range of one file that should end up in the page cache.
struct file_warm_range
{
    int file = 0;                   // index into the caller's path list
    uint64_t offset = 0;            // file offset of the first byte
    uint64_t bytes = 0;
    const void * mapped = nullptr;  // where `offset` lives in an existing mapping of
                                    // the file, for the residency check; null = none
};

// Sort by (file, offset) and merge ranges that touch or overlap. Two ranges
// merge only when their mapping addresses agree (or neither has one), so the
// merged range's `mapped` stays valid for every byte it covers. GGUF tensors
// are laid out back to back, so the tensors of a run (a layer's three expert
// tensors, a whole Engram table) become one range; a run of other tensors in
// between (the next layer's attention, ~80 MiB) keeps two ranges apart.
inline std::vector<file_warm_range> merge_file_warm_ranges(std::vector<file_warm_range> ranges)
{
    ranges.erase(std::remove_if(ranges.begin(), ranges.end(),
        [](const file_warm_range & r) { return r.bytes == 0; }), ranges.end());
    std::sort(ranges.begin(), ranges.end(), [](const file_warm_range & a, const file_warm_range & b)
    {
        if (a.file != b.file) return a.file < b.file;
        return a.offset < b.offset;
    });
    std::vector<file_warm_range> out;
    for (const file_warm_range & r : ranges)
    {
        if (!out.empty())
        {
            file_warm_range & cur = out.back();
            const uint64_t cur_end = cur.offset + cur.bytes;
            bool same_map = (cur.mapped == nullptr) == (r.mapped == nullptr);
            if (same_map && cur.mapped)
                same_map = (const char *) cur.mapped + (r.offset - cur.offset) == (const char *) r.mapped;
            if (cur.file == r.file && r.offset <= cur_end && same_map)
            {
                cur.bytes = std::max(cur_end, r.offset + r.bytes) - cur.offset;
                continue;
            }
        }
        out.push_back(r);
    }
    return out;
}

// A contiguous slice of one merged range, assigned to one thread.
struct file_warm_piece
{
    size_t range = 0;       // index into the merged ranges
    uint64_t offset = 0;    // absolute file offset
    uint64_t bytes = 0;
};

// Split the merged ranges, in order, into at most `threads` byte-balanced runs:
// thread k reads bytes [k*T/n, (k+1)*T/n) of the concatenation. Each thread's
// run is contiguous in file order (it may step over the gap between two ranges
// or continue into the next file), which is what per-descriptor readahead and
// the storage's own sequential detection want. Uses no more threads than there
// are `block_bytes` blocks to read.
inline std::vector<std::vector<file_warm_piece>> plan_file_warm(const std::vector<file_warm_range> & merged,
                                                                int threads, uint64_t block_bytes)
{
    uint64_t total = 0;
    for (const file_warm_range & r : merged) total += r.bytes;
    std::vector<std::vector<file_warm_piece>> plan;
    if (total == 0) return plan;
    if (block_bytes == 0) block_bytes = 1;
    uint64_t n = (uint64_t) std::max(1, threads);
    n = std::min<uint64_t>(n, (total + block_bytes - 1) / block_bytes);
    plan.resize((size_t) n);
    size_t ri = 0;
    uint64_t in_range = 0; // bytes of merged[ri] already assigned
    for (uint64_t k = 0; k < n; ++k)
    {
        const uint64_t lo = total / n * k + total % n * k / n;
        const uint64_t hi = k + 1 == n ? total : total / n * (k + 1) + total % n * (k + 1) / n;
        uint64_t want = hi - lo;
        while (want > 0 && ri < merged.size())
        {
            const uint64_t left = merged[ri].bytes - in_range;
            const uint64_t take = std::min(left, want);
            if (take > 0)
            {
                plan[(size_t) k].push_back({ ri, merged[ri].offset + in_range, take });
                in_range += take;
                want -= take;
            }
            if (in_range == merged[ri].bytes) { ++ri; in_range = 0; }
        }
    }
    return plan;
}

// ---------------------------------------------------------------------------
// Warming
// ---------------------------------------------------------------------------

struct file_warm_options
{
    int threads = 16;                                   // TS_DSV4_LOAD_THREADS
    uint64_t block_bytes = uint64_t(64) * 1024 * 1024;  // one pread and one residency check
    bool skip_resident = true;                          // mincore skip
    bool populate = false;                              // map each block (one read per page) once cached
    const std::atomic<bool> * stop = nullptr;           // checked before every block
    // Test and benchmark hook: called after every block with whether it was read
    // (false = skipped as resident). Called concurrently from the reader threads.
    std::function<void(int thread, int file, uint64_t offset, uint64_t bytes, bool read)> on_block;
};

struct file_warm_result
{
    bool ok = true;
    bool stopped = false;
    int threads = 0;
    uint64_t bytes_total = 0;
    uint64_t bytes_read = 0;
    uint64_t bytes_resident = 0;       // skipped because every page was already cached
    double seconds = 0;
    std::string error;                 // first failure, naming the file and offset
};

#if !defined(_WIN32)
namespace file_warm_detail {

inline size_t page_size()
{
    static const size_t page = []() { const long p = sysconf(_SC_PAGESIZE); return p > 0 ? (size_t) p : (size_t) 4096; }();
    return page;
}

// [aligned, aligned + len) is the page-aligned span covering [p, p + bytes).
inline void page_span(const void * p, uint64_t bytes, void *& aligned, size_t & len)
{
    const uintptr_t a = (uintptr_t) p, page = page_size();
    const uintptr_t start = a / page * page;
    aligned = (void *) start;
    len = (size_t) (a + bytes - start);
}

// Whether every page of [p, p + bytes) is resident. False when unknown.
inline bool fully_resident(const void * p, uint64_t bytes, std::vector<unsigned char> & scratch)
{
    if (!p || bytes == 0) return false;
    void * aligned; size_t len;
    page_span(p, bytes, aligned, len);
    const size_t pages = (len + page_size() - 1) / page_size();
    scratch.resize(pages);
#if defined(__APPLE__)
    if (mincore(aligned, len, (char *) scratch.data()) != 0) return false;
#else
    if (mincore(aligned, len, scratch.data()) != 0) return false;
#endif
    for (size_t i = 0; i < pages; ++i)
        if ((scratch[i] & 1) == 0) return false;
    return true;
}

// Map [p, p + bytes) into this process by reading one byte of every page. On a
// block that was just read (or found resident) every access is a minor fault.
inline void populate(const void * p, uint64_t bytes)
{
    if (!p || bytes == 0) return;
    void * aligned; size_t len;
    page_span(p, bytes, aligned, len);
    const volatile char * base = (const volatile char *) p;
    const uintptr_t first = (uintptr_t) aligned + page_size() - (uintptr_t) p; // next page start, relative
    (void) base[0];
    for (uint64_t off = first; off < bytes; off += page_size())
        (void) base[off];
}

} // namespace file_warm_detail
#endif

// Read every byte of `ranges` into the page cache (see the header comment).
// Never throws; a failure is reported in the result with the path and offset.
inline file_warm_result warm_file_ranges(const std::vector<std::string> & paths,
                                         const std::vector<file_warm_range> & ranges,
                                         const file_warm_options & options)
{
    file_warm_result result;
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<file_warm_range> merged = merge_file_warm_ranges(ranges);
    for (const file_warm_range & r : merged)
    {
        result.bytes_total += r.bytes;
        if (r.file < 0 || (size_t) r.file >= paths.size())
        {
            result.ok = false;
            result.error = "warm range names file index " + std::to_string(r.file) + " of " +
                           std::to_string(paths.size());
            return result;
        }
    }
#if defined(_WIN32)
    (void) options;
    if (result.bytes_total == 0) return result; // nothing mapped, nothing to warm
    result.ok = false;
    result.error = "pread warming is not available on Windows";
    return result;
#else
    const uint64_t block = std::max<uint64_t>(options.block_bytes, 1);
    const auto plan = plan_file_warm(merged, options.threads, block);
    result.threads = (int) plan.size();
    if (plan.empty()) return result;

    std::atomic<bool> failed(false), stopped(false);
    std::atomic<uint64_t> bytes_read(0), bytes_resident(0);
    std::mutex error_mu;
    auto fail = [&](const std::string & message)
    {
        std::lock_guard<std::mutex> lock(error_mu);
        if (!failed.exchange(true)) result.error = message;
    };

    auto worker = [&](int k)
    {
        std::vector<int> fds(paths.size(), -1);
        std::unique_ptr<char[]> buffer;
        std::vector<unsigned char> residency;
        // Returns at the first failure or stop; the descriptors close below.
        auto run = [&]()
        {
            for (const file_warm_piece & piece : plan[(size_t) k])
            {
                const file_warm_range & range = merged[piece.range];
                const std::string & path = paths[(size_t) range.file];
                for (uint64_t done = 0; done < piece.bytes; )
                {
                    if (failed.load(std::memory_order_relaxed)) return;
                    if (options.stop && options.stop->load(std::memory_order_relaxed))
                    {
                        stopped.store(true, std::memory_order_relaxed);
                        return;
                    }
                    const uint64_t off = piece.offset + done;
                    const uint64_t len = std::min(block, piece.bytes - done);
                    const void * mapped = range.mapped
                        ? (const void *) ((const char *) range.mapped + (off - range.offset)) : nullptr;
                    const bool resident = options.skip_resident && mapped &&
                                          file_warm_detail::fully_resident(mapped, len, residency);
                    if (resident)
                    {
                        bytes_resident.fetch_add(len, std::memory_order_relaxed);
                    }
                    else
                    {
                        int & fd = fds[(size_t) range.file];
                        if (fd < 0)
                        {
                            fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
                            if (fd < 0)
                            {
                                fail("cannot open " + path + " for warming: " + std::strerror(errno));
                                return;
                            }
                        }
                        if (!buffer)
                        {
                            buffer.reset(new (std::nothrow) char[(size_t) block]);
                            if (!buffer)
                            {
                                fail("cannot allocate a " + std::to_string(block) + "-byte warm buffer");
                                return;
                            }
                        }
                        for (uint64_t got_total = 0; got_total < len; )
                        {
                            const ssize_t got = ::pread(fd, buffer.get(), (size_t) (len - got_total),
                                                        (off_t) (off + got_total));
                            if (got < 0 && errno == EINTR) continue;
                            if (got <= 0)
                            {
                                const int err = errno;
                                char msg[256];
                                if (got == 0)
                                    std::snprintf(msg, sizeof(msg), "short read at offset %llu (%llu of %llu bytes missing; "
                                                  "the file ends early)", (unsigned long long) (off + got_total),
                                                  (unsigned long long) (len - got_total), (unsigned long long) len);
                                else
                                    std::snprintf(msg, sizeof(msg), "read error at offset %llu: %s",
                                                  (unsigned long long) (off + got_total), std::strerror(err));
                                fail(path + ": " + msg);
                                return;
                            }
                            got_total += (uint64_t) got;
                        }
                        bytes_read.fetch_add(len, std::memory_order_relaxed);
                    }
                    if (options.populate && mapped) file_warm_detail::populate(mapped, len);
                    if (options.on_block) options.on_block(k, range.file, off, len, !resident);
                    done += len;
                }
            }
        };
        run();
        for (int fd : fds) if (fd >= 0) ::close(fd);
    };

    std::vector<std::thread> pool;
    pool.reserve(plan.size());
    try
    {
        for (int k = 1; k < (int) plan.size(); ++k) pool.emplace_back(worker, k);
    }
    catch (const std::exception & e)
    {
        fail(std::string("cannot start warm threads: ") + e.what());
    }
    worker(0);
    for (auto & th : pool) th.join();

    result.ok = !failed.load();
    result.stopped = stopped.load();
    result.bytes_read = bytes_read.load();
    result.bytes_resident = bytes_resident.load();
    result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return result;
#endif
}

// ---------------------------------------------------------------------------
// The page-touch prefault (TS_DSV4_WARM_PREAD=0)
// ---------------------------------------------------------------------------

// The host-expert prefault as it was before the pread helper: each range is cut
// into 256 MiB spans, the spans are handed out from one shared cursor, and each
// thread reads one byte of every 4 KiB page of its span through the mapping.
// Kept byte for byte as the escape hatch and as the benchmark's control.
inline void touch_prefault_spans(const std::vector<std::pair<const volatile char *, size_t>> & ranges,
                                 int load_threads, size_t warm_span = (size_t) 256 * 1024 * 1024)
{
    // Split each tensor into spans so the pool is sized by BYTES, not by
    // tensor count: --n-cpu-moe 2 leaves 6 tensors, which capped the pool
    // at 6 threads and made this 31 s for 16.4 GiB. Each thread still walks
    // one contiguous span, which is what readahead wants.
    std::vector<std::pair<const volatile char *, size_t>> spans;
    for (const auto & r : ranges)
        for (size_t off = 0; off < r.second; off += warm_span)
            spans.emplace_back(r.first + off, std::min(warm_span, r.second - off));
    std::atomic<size_t> r_cursor(0);
    auto warm_worker = [&]()
    {
        for (;;)
        {
            const size_t i = r_cursor.fetch_add(1, std::memory_order_relaxed);
            if (i >= spans.size()) break;
            const volatile char * p = spans[i].first;
            // MADV_WILLNEED here was measured and did NOT help on the MooseFS
            // mount (29.7 s against 27.7 s for the plain walk, i.e. inside the
            // run-to-run spread), so this stays a plain fault-in walk.
            for (size_t off = 0; off < spans[i].second; off += 4096)
                (void) p[off];
        }
    };
    std::vector<std::thread> warm_pool;
    for (int i = 0; i < std::min<int>(load_threads, (int) spans.size()); i++) warm_pool.emplace_back(warm_worker);
    for (auto & th : warm_pool) th.join();
}

} // namespace tsg_dsv4
