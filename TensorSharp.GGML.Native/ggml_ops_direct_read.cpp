// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Weight uploads that read the file instead of the mapping.
//
// A resident weight is bound as a pointer into the mapped GGUF, so its upload
// copies out of the mapping, and every page the process has not seen yet is a
// fault served through the page cache from inside the host-to-device copy. For a
// network that is uploaded for one request and released again that is the whole
// cost of the stage. MiniMax-H3's text encoder is the case this was written for:
// its file is too large for the cache to keep between requests next to the
// denoiser's, a read-ahead through the same cache tops out well below the disk's
// own rate, and the pages it places push the other networks' files out.
//
// h3_direct_read_upload reads the same byte ranges with unbuffered, sector-aligned
// reads on a few threads into a small pool of page-locked buffers and uploads each
// buffer as it arrives. Nothing goes through the page cache, so nothing is evicted
// and the disk is the limit: the text encoder's upload measured 0.7 GB/s through
// the mapping and 2.8 GB/s this way.
//
// The mapping itself is never touched. TSGgml_DirectReadRegister records which
// file a mapping is, so a weight pointer resolves to (file, offset). Any failure -
// a file that cannot be opened unbuffered, a short read, no memory for the pool -
// leaves the tensors it did not finish to the caller's mapped upload, which
// rewrites them whole.
//
// Reads are issued on worker threads; every backend call stays on the caller's
// thread, as ggml_backend_tensor_set requires.
//
// TS_H3_DIRECT_READ=0 turns it off and TS_H3_DIRECT_READ_THREADS sets the reader
// count (default 8). Implemented for Windows, where the mapped upload has no page
// warming of its own; elsewhere it reports unavailable and nothing changes.
#include "ggml_ops_internal.h"
#include "ggml_h3_direct_read.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#if defined(TSG_GGML_USE_CUDA)
#include <cuda_runtime.h>
#endif
#endif

namespace {

bool direct_read_enabled()
{
    const char* e = std::getenv("TS_H3_DIRECT_READ");
    return !(e != nullptr && e[0] == '0');
}

#if defined(_WIN32)

using tsg::H3DirectReadItem;

// Unbuffered reads need the offset, the length and the buffer aligned to the
// volume's sector size; 4096 covers 512-byte and 4K sectors alike.
constexpr std::size_t kAlign = 4096;
constexpr std::size_t kChunk = std::size_t{16} << 20;

struct Region
{
    std::uintptr_t base;
    std::size_t bytes;
    std::wstring path;
};

std::mutex g_region_mutex;
std::vector<Region> g_regions;

std::wstring widen_path(const char* utf8)
{
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, nullptr, 0);
    if (n <= 1) return {};
    std::wstring wide(static_cast<std::size_t>(n), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, &wide[0], n) != n) return {};
    wide.resize(static_cast<std::size_t>(n) - 1);
    // A drive-letter path past MAX_PATH needs the extended-length prefix.
    if (wide.size() >= 248 && wide.size() > 2 && wide[1] == L':' && wide.compare(0, 4, L"\\\\?\\") != 0)
        wide.insert(0, L"\\\\?\\");
    return wide;
}

// The pool the reads land in. Page-locked when CUDA can provide it, so that the
// upload is a DMA transfer rather than a second copy through the driver's own
// staging buffer.
struct Staging
{
    unsigned char* base = nullptr;
    bool pinned = false;

    bool allocate(std::size_t bytes)
    {
#if defined(TSG_GGML_USE_CUDA)
        void* p = nullptr;
        if (cudaHostAlloc(&p, bytes, cudaHostAllocPortable) == cudaSuccess && p != nullptr)
        {
            if ((reinterpret_cast<std::uintptr_t>(p) & (kAlign - 1)) == 0)
            {
                base = static_cast<unsigned char*>(p);
                pinned = true;
                return true;
            }
            cudaFreeHost(p);
        }
        else
        {
            cudaGetLastError();   // a refused allocation must not surface in a later CUDA call
        }
#endif
        base = static_cast<unsigned char*>(VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        return base != nullptr;
    }

    ~Staging()
    {
        if (base == nullptr) return;
#if defined(TSG_GGML_USE_CUDA)
        if (pinned) { cudaFreeHost(base); return; }
#endif
        VirtualFree(base, 0, MEM_RELEASE);
    }
};

// A run of one tensor's bytes inside one chunk.
struct Segment
{
    std::uint32_t item;
    std::uint32_t chunk_offset;
    std::uint32_t bytes;
    std::size_t tensor_offset;
};

// One aligned read. `needed` is how far into it the last segment reaches, which is
// all a read at the end of the file has to return.
struct Chunk
{
    std::uint32_t file = 0;
    std::uint32_t needed = 0;
    std::uint64_t offset = 0;
    std::size_t first_segment = 0;
    std::size_t segments = 0;
};

struct Planned
{
    std::uint32_t item;
    std::uint32_t file;
    std::uint64_t offset;
};

struct Ready
{
    std::size_t chunk;
    std::size_t buffer;
    bool ok;
};

struct Shared
{
    std::mutex mutex;
    std::condition_variable free_cv, ready_cv;
    std::vector<std::size_t> free_buffers;
    std::deque<Ready> ready;
    std::atomic<std::size_t> next{0};
    std::atomic<bool> stop{false};
};

// Joins the readers on every way out, an exception included.
struct Readers
{
    Shared& shared;
    std::vector<std::thread> threads;

    explicit Readers(Shared& s) : shared(s) {}
    ~Readers()
    {
        shared.stop.store(true, std::memory_order_release);
        { std::lock_guard<std::mutex> lock(shared.mutex); }
        shared.free_cv.notify_all();
        for (auto& t : threads)
            if (t.joinable()) t.join();
    }
};

std::size_t reader_count()
{
    if (const char* e = std::getenv("TS_H3_DIRECT_READ_THREADS"))
    {
        char* end = nullptr;
        const long n = std::strtol(e, &end, 10);
        if (end != e && *end == '\0' && n >= 1 && n <= 32) return static_cast<std::size_t>(n);
    }
    return 8;
}

std::size_t upload(H3DirectReadItem* items, std::size_t count, const char* tag)
{
    using Clock = std::chrono::steady_clock;
    const auto begin = Clock::now();

    // Resolve each binding to (file, offset).
    std::vector<std::wstring> files;
    std::vector<Planned> planned;
    {
        std::lock_guard<std::mutex> lock(g_region_mutex);
        if (g_regions.empty()) return 0;
        std::vector<std::int32_t> file_of_region(g_regions.size(), -1);
        planned.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            H3DirectReadItem& it = items[i];
            it.done = false;
            if (it.tensor == nullptr || it.data == nullptr || it.bytes == 0) continue;
            const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(it.data);
            for (std::size_t r = 0; r < g_regions.size(); ++r)
            {
                const Region& region = g_regions[r];
                if (addr < region.base || it.bytes > region.bytes || addr - region.base > region.bytes - it.bytes)
                    continue;
                if (file_of_region[r] < 0)
                {
                    file_of_region[r] = static_cast<std::int32_t>(files.size());
                    files.push_back(region.path);
                }
                planned.push_back({ static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(file_of_region[r]),
                                    static_cast<std::uint64_t>(addr - region.base) });
                break;
            }
        }
    }
    if (planned.empty()) return 0;

    // Cut the ranges, in file order, into aligned chunks. Neighbouring tensors share
    // a chunk; the few bytes of padding between them are read and ignored.
    std::sort(planned.begin(), planned.end(), [](const Planned& a, const Planned& b) {
        return a.file != b.file ? a.file < b.file : a.offset < b.offset;
    });
    std::vector<Chunk> chunks;
    std::vector<Segment> segments;
    std::vector<std::size_t> remaining(count, 0);
    std::vector<char> in_plan(count, 0);
    std::size_t planned_bytes = 0, planned_tensors = 0;
    {
        bool open = false;
        Chunk cur;
        const auto close = [&] {
            if (!open) return;
            cur.segments = segments.size() - cur.first_segment;
            chunks.push_back(cur);
            open = false;
        };
        for (const Planned& p : planned)
        {
            if (open && p.file != cur.file) close();
            // Bytes shared with a tensor already split across earlier chunks: the
            // mapped upload takes this one.
            if (open && p.offset < cur.offset) continue;
            const std::size_t bytes = items[p.item].bytes;
            std::uint64_t pos = p.offset;
            std::size_t left = bytes;
            in_plan[p.item] = 1;
            remaining[p.item] = bytes;
            planned_bytes += bytes;
            ++planned_tensors;
            while (left > 0)
            {
                if (!open || pos >= cur.offset + kChunk)
                {
                    close();
                    cur = Chunk{};
                    cur.file = p.file;
                    cur.offset = pos & ~static_cast<std::uint64_t>(kAlign - 1);
                    cur.first_segment = segments.size();
                    open = true;
                }
                const std::size_t in_chunk = static_cast<std::size_t>(pos - cur.offset);
                const std::size_t take = std::min(left, kChunk - in_chunk);
                segments.push_back({ p.item, static_cast<std::uint32_t>(in_chunk), static_cast<std::uint32_t>(take),
                                     bytes - left });
                cur.needed = std::max(cur.needed, static_cast<std::uint32_t>(in_chunk + take));
                pos += take;
                left -= take;
            }
        }
        close();
    }

    const std::size_t threads = std::min(reader_count(), chunks.size());
    const std::size_t buffers = std::min(threads + 4, chunks.size());
    Staging staging;
    if (!staging.allocate(buffers * kChunk))
    {
        std::fprintf(stderr, "[h3-direct-read] tag=%s no staging memory; using the mapped upload\n", tag);
        return 0;
    }
    const double alloc_seconds = std::chrono::duration<double>(Clock::now() - begin).count();

    Shared shared;
    shared.free_buffers.reserve(buffers);
    for (std::size_t b = 0; b < buffers; ++b) shared.free_buffers.push_back(b);
    std::atomic<unsigned long> first_error{0};

    const auto read_chunks = [&] {
        std::vector<HANDLE> handles(files.size(), INVALID_HANDLE_VALUE);
        for (;;)
        {
            if (shared.stop.load(std::memory_order_acquire)) break;
            const std::size_t index = shared.next.fetch_add(1, std::memory_order_relaxed);
            if (index >= chunks.size()) break;
            std::size_t buffer = 0;
            {
                std::unique_lock<std::mutex> lock(shared.mutex);
                shared.free_cv.wait(lock, [&] {
                    return shared.stop.load(std::memory_order_acquire) || !shared.free_buffers.empty();
                });
                if (shared.stop.load(std::memory_order_acquire)) break;
                buffer = shared.free_buffers.back();
                shared.free_buffers.pop_back();
            }
            const Chunk& chunk = chunks[index];
            HANDLE& handle = handles[chunk.file];
            if (handle == INVALID_HANDLE_VALUE)
                handle = CreateFileW(files[chunk.file].c_str(), GENERIC_READ,
                                     FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                     FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
            bool ok = handle != INVALID_HANDLE_VALUE;
            if (ok)
            {
                // A read that runs past the end of the file returns what is there.
                const DWORD length = static_cast<DWORD>((chunk.needed + kAlign - 1) & ~(kAlign - 1));
                OVERLAPPED at{};
                at.Offset = static_cast<DWORD>(chunk.offset & 0xffffffffu);
                at.OffsetHigh = static_cast<DWORD>(chunk.offset >> 32);
                DWORD got = 0;
                ok = ReadFile(handle, staging.base + buffer * kChunk, length, &got, &at) != 0 && got >= chunk.needed;
            }
            if (!ok)
            {
                unsigned long expected = 0;
                const unsigned long error = GetLastError();
                first_error.compare_exchange_strong(expected, error != 0 ? error : ERROR_HANDLE_EOF);
            }
            {
                std::lock_guard<std::mutex> lock(shared.mutex);
                shared.ready.push_back({ index, buffer, ok });
            }
            shared.ready_cv.notify_one();
            if (!ok) break;
        }
        for (HANDLE handle : handles)
            if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
    };

    bool failed = false;
    {
        Readers readers(shared);
        readers.threads.reserve(threads);
        for (std::size_t t = 0; t < threads; ++t) readers.threads.emplace_back(read_chunks);

        for (std::size_t processed = 0; processed < chunks.size(); ++processed)
        {
            Ready ready{};
            {
                std::unique_lock<std::mutex> lock(shared.mutex);
                shared.ready_cv.wait(lock, [&] { return !shared.ready.empty(); });
                ready = shared.ready.front();
                shared.ready.pop_front();
            }
            if (!ready.ok) { failed = true; break; }
            const Chunk& chunk = chunks[ready.chunk];
            const unsigned char* source = staging.base + ready.buffer * kChunk;
            for (std::size_t s = chunk.first_segment; s < chunk.first_segment + chunk.segments; ++s)
            {
                const Segment& segment = segments[s];
                ggml_backend_tensor_set(items[segment.item].tensor, source + segment.chunk_offset,
                                        segment.tensor_offset, segment.bytes);
                remaining[segment.item] -= segment.bytes;
            }
            {
                std::lock_guard<std::mutex> lock(shared.mutex);
                shared.free_buffers.push_back(ready.buffer);
            }
            shared.free_cv.notify_one();
        }
    }   // joins the readers

    std::size_t done = 0, done_bytes = 0;
    for (std::size_t i = 0; i < count; ++i)
    {
        if (!in_plan[i] || remaining[i] != 0) continue;
        items[i].done = true;
        done_bytes += items[i].bytes;
        ++done;
    }
    if (failed)
        std::fprintf(stderr, "[h3-direct-read] tag=%s read failed (error %lu) after %zu of %zu tensors; "
                             "the mapped upload takes the rest\n", tag, first_error.load(), done, planned_tensors);
    const char* profile = std::getenv("TS_H3_UPLOAD_PROFILE");
    if (profile != nullptr && std::strcmp(profile, "1") == 0)
    {
        const double seconds = std::chrono::duration<double>(Clock::now() - begin).count();
        std::fprintf(stderr,
            "[h3-direct-read] tag=%s tensors=%zu bytes=%zu planned_bytes=%zu chunks=%zu threads=%zu buffers=%zu "
            "pinned=%d alloc_seconds=%.6f seconds=%.6f gb_per_second=%.2f\n",
            tag, done, done_bytes, planned_bytes, chunks.size(), threads, buffers, staging.pinned ? 1 : 0,
            alloc_seconds, seconds, seconds > 0 ? static_cast<double>(done_bytes) / seconds / 1e9 : 0.0);
    }
    if (failed || (profile != nullptr && std::strcmp(profile, "1") == 0)) std::fflush(stderr);
    return done;
}

#endif // _WIN32

} // namespace

namespace tsg {

std::size_t h3_direct_read_upload(ggml_backend_t backend, H3DirectReadItem* items, std::size_t count, const char* tag)
{
#if defined(_WIN32)
    if (backend == nullptr || items == nullptr || count == 0 || !direct_read_enabled()) return 0;
    const char* name = ggml_backend_name(backend);
    if (name == nullptr || std::strncmp(name, "CUDA", 4) != 0) return 0;
    try
    {
        return upload(items, count, tag != nullptr ? tag : "");
    }
    catch (...)
    {
        // Out of memory while planning or starting a reader. `done` is only ever set
        // once a tensor is complete, so whatever is unset is still the caller's.
        std::size_t done = 0;
        for (std::size_t i = 0; i < count; ++i) done += items[i].done ? 1 : 0;
        return done;
    }
#else
    (void) backend; (void) items; (void) count; (void) tag;
    return 0;
#endif
}

} // namespace tsg

// 1 when uploads can read a registered file directly on this platform, so a caller
// can leave out the page-cache read-ahead it would otherwise start for the mapping.
TSG_EXPORT int TSGgml_DirectReadSupported(void)
{
#if defined(_WIN32)
    return direct_read_enabled() ? 1 : 0;
#else
    return 0;
#endif
}

// Records that [base, base + length) maps `path` from file offset 0. Returns 1 when
// registered, 0 when direct reads are unavailable.
TSG_EXPORT int TSGgml_DirectReadRegister(const void* base, long long length, const char* path_utf8)
{
#if defined(_WIN32)
    if (base == nullptr || length <= 0 || path_utf8 == nullptr || !direct_read_enabled()) return 0;
    try
    {
        std::wstring path = widen_path(path_utf8);
        if (path.empty()) return 0;
        const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(base);
        std::lock_guard<std::mutex> lock(g_region_mutex);
        // An address registered twice describes the newer mapping.
        for (Region& region : g_regions)
        {
            if (region.base != address) continue;
            region.bytes = static_cast<std::size_t>(length);
            region.path = std::move(path);
            return 1;
        }
        g_regions.push_back({ address, static_cast<std::size_t>(length), std::move(path) });
        return 1;
    }
    catch (...)
    {
        return 0;
    }
#else
    (void) base; (void) length; (void) path_utf8;
    return 0;
#endif
}

// Forgets a mapping. Call before unmapping it, so that a later mapping at the same
// address cannot resolve to the old file.
TSG_EXPORT void TSGgml_DirectReadUnregister(const void* base)
{
#if defined(_WIN32)
    if (base == nullptr) return;
    const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(base);
    std::lock_guard<std::mutex> lock(g_region_mutex);
    g_regions.erase(std::remove_if(g_regions.begin(), g_regions.end(),
                                   [address](const Region& region) { return region.base == address; }),
                    g_regions.end());
#else
    (void) base;
#endif
}
