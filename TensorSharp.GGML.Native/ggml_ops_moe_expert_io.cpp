// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

// ============================================================================
// Explicitly read host expert cache for MoE CPU offload.
//
// A host-offloaded expert used to be read straight from the GGUF mapping, so a
// cold expert arrived through page faults taken inside the compute team: one
// synchronous fault per page, a handful of threads deep. On an M5 Pro SSD that
// ran at 0.5-1 GB/s, while pread of the same 0.6 MB expert slices reaches
// 2.8 GB/s on one thread and 13-14 GB/s at 8-16 in flight. When a model is
// larger than RAM (Qwen3.8-Flash-Next UD-Q2_K_XL: 79 GB on 48 GB), those
// faults were half of every decode step.
//
// So a registered mapping's experts are served from a bounded cache of
// anonymous slots instead. Misses of one call (every routed expert of every
// row, all three parts) are read together on a small sleeping I/O team, the
// caller reading too, then the compute team reads the slots. The file is read
// with F_NOCACHE on Apple so the slots are the only copy kept. Weights are
// immutable, so a slot never needs writing back; slots live until evicted
// (least recently used, byte budget) or until their mapping is unregistered.
// NInfer reads its disk-backed PLE table the same way: by row, into a bounded
// row cache, rather than through mmap paging.
//
// A pointer outside every registered mapping (or with the cache off) resolves
// to itself, so callers need no fallback path.
// ============================================================================

#include "ggml_ops_internal.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <list>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <mach/mach.h>
#include <pthread/qos.h>
#endif

namespace tsg
{
    namespace
    {
#if !defined(_WIN32)
        struct MappedFile
        {
            const std::uint8_t* base = nullptr;
            std::size_t length = 0;
            int fd = -1;
        };

        struct Slot
        {
            std::uint8_t* data = nullptr;
            std::size_t bytes = 0;
            const void* key = nullptr;
            std::uint64_t pin = 0;          // call epoch that last used it
            bool prefetched = false;        // read ahead and not yet used
            std::list<Slot*>::iterator lru;
        };

        struct IoRequest
        {
            int fd;
            std::uint8_t* dst;
            std::size_t bytes;
            off_t offset;
        };

        // Sleeps between calls: a spinning host thread takes the package's shared
        // power budget and the accelerator clocks down (see the decode team).
        class IoTeam
        {
        public:
            explicit IoTeam(int workers)
            {
                for (int i = 0; i < workers; ++i) m_threads.emplace_back([this] { loop(); });
            }
            ~IoTeam()
            {
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_stop = true;
                }
                m_cv.notify_all();
                for (auto& t : m_threads) t.join();
            }
            // Returns false when any read came back short or failed.
            bool run(const std::vector<IoRequest>& requests)
            {
                if (requests.empty()) return true;
                // One job at a time: a second caller (a prefetch still loading while
                // the next is queued) waits for the first to finish.
                std::lock_guard<std::mutex> one_caller(m_run_mutex);
                {
                    // A worker still inside the previous call's work() would take
                    // an index from this call's counter (and, for another request
                    // list, drop or misread it): reset only once all have left.
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_idle_cv.wait(lock, [&] { return m_active == 0; });
                    m_requests = &requests;
                    m_next.store(0, std::memory_order_relaxed);
                    m_done.store(0, std::memory_order_relaxed);
                    m_failed.store(false, std::memory_order_relaxed);
                    ++m_generation;
                }
                if (requests.size() > 1) m_cv.notify_all();
                work(requests);
                std::unique_lock<std::mutex> lock(m_mutex);
                m_done_cv.wait(lock, [&] { return m_done.load(std::memory_order_acquire) == requests.size(); });
                m_requests = nullptr;
                return !m_failed.load(std::memory_order_relaxed);
            }
            // Background batch: the workers alone read `requests` (kept here) while
            // the caller goes on; finish() waits for them. Never mixed with run():
            // a team is used one way or the other.
            void start(std::vector<IoRequest>&& requests)
            {
                finish();
                if (requests.empty()) return;
                {
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_idle_cv.wait(lock, [&] { return m_active == 0; });
                    m_own = std::move(requests);
                    m_requests = &m_own;
                    m_next.store(0, std::memory_order_relaxed);
                    m_done.store(0, std::memory_order_relaxed);
                    m_failed.store(false, std::memory_order_relaxed);
                    ++m_generation;
                    m_pending = true;
                }
                m_cv.notify_all();
            }
            bool pending() const { return m_pending; }
            bool finish()
            {
                if (!m_pending) return true;
                std::unique_lock<std::mutex> lock(m_mutex);
                m_done_cv.wait(lock, [&] { return m_done.load(std::memory_order_acquire) == m_own.size(); });
                m_requests = nullptr;
                m_pending = false;
                return !m_failed.load(std::memory_order_relaxed);
            }
        private:
            void work(const std::vector<IoRequest>& requests)
            {
                for (;;)
                {
                    const std::size_t i = m_next.fetch_add(1, std::memory_order_relaxed);
                    if (i >= requests.size()) return;
                    const IoRequest& r = requests[i];
                    std::size_t got = 0;
                    while (got < r.bytes)
                    {
                        const ssize_t n = pread(r.fd, r.dst + got, r.bytes - got, r.offset + static_cast<off_t>(got));
                        if (n <= 0) { if (n < 0 && errno == EINTR) continue; break; }
                        got += static_cast<std::size_t>(n);
                    }
                    if (got != r.bytes) m_failed.store(true, std::memory_order_relaxed);
                    if (m_done.fetch_add(1, std::memory_order_acq_rel) + 1 == requests.size())
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        m_done_cv.notify_all();
                    }
                }
            }
            void loop()
            {
#if defined(__APPLE__)
                pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif
                std::uint64_t seen = 0;
                for (;;)
                {
                    const std::vector<IoRequest>* requests = nullptr;
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_cv.wait(lock, [&] { return m_stop || (m_generation != seen && m_requests != nullptr); });
                        if (m_stop) return;
                        seen = m_generation;
                        requests = m_requests;
                        ++m_active;
                    }
                    work(*requests);
                    {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        --m_active;
                    }
                    m_idle_cv.notify_all();
                }
            }

            std::vector<std::thread> m_threads;
            std::mutex m_mutex;
            std::mutex m_run_mutex;
            std::condition_variable m_cv, m_done_cv, m_idle_cv;
            int m_active = 0;                     // workers inside work()
            const std::vector<IoRequest>* m_requests = nullptr;
            std::uint64_t m_generation = 0;
            std::atomic<std::size_t> m_next{0}, m_done{0};
            std::atomic<bool> m_failed{false};
            bool m_stop = false;
            std::vector<IoRequest> m_own;         // start()'s batch
            bool m_pending = false;
        };

        std::mutex g_mutex;
        std::vector<MappedFile> g_files;
        std::unordered_map<const void*, Slot*> g_slots;
        std::list<Slot*> g_lru;                 // front = most recent
        std::size_t g_budget = 0, g_used = 0;
        std::uint64_t g_epoch = 0;
        std::uint64_t g_hits = 0, g_misses = 0, g_read_bytes = 0;
        std::uint64_t g_wire_failures = 0;   // slots mlock refused (they stay pageable)
        IoTeam* g_team = nullptr;
        // host_expert_io_read_slices_async: its own team (a seam's synchronous reads
        // may run while a batch is in flight) and the batch, for the copy fallback.
        IoTeam* g_async_team = nullptr;
        std::vector<const void*> g_async_src;
        std::vector<std::size_t> g_async_len;
        std::vector<std::uint8_t*> g_async_dst;
        // host_expert_io_prefetch: slices read ahead into cache slots on a team of their
        // own; the next resolve settles the batch before it reads any slot.
        IoTeam* g_prefetch_team = nullptr;
        std::vector<const void*> g_prefetch_keys;
        std::uint64_t g_prefetch_reads = 0, g_prefetch_used = 0;

        const MappedFile* find_file(const void* p, std::size_t bytes)
        {
            const auto* q = static_cast<const std::uint8_t*>(p);
            for (const auto& f : g_files)
                if (q >= f.base && bytes <= f.length && static_cast<std::size_t>(q - f.base) <= f.length - bytes)
                    return &f;
            return nullptr;
        }

        void free_slot(Slot* s)
        {
            g_lru.erase(s->lru);
            g_slots.erase(s->key);
            g_used -= s->bytes;
            munmap(s->data, s->bytes);
            delete s;
        }

        // A slot of exactly `bytes`, reusing the least recently used one of that
        // size when the budget is full; null when everything left is pinned.
        // A model's slices come in a few sizes (DeepSeek V4.1 UD-IQ2_XXS: IQ1_S
        // gate/up, IQ2_XXS or IQ2_S down), so the oldest unpinned slot is often
        // the wrong size: a same-size one among the next oldest is reused instead
        // of unmapping it and mapping (and zero-filling) a fresh one.
        Slot* take_slot(std::size_t bytes)
        {
            constexpr int kSizeMatchWindow = 32;
            while (g_used + bytes > g_budget && !g_lru.empty())
            {
                Slot* victim = nullptr;
                Slot* same = nullptr;
                int seen = 0;
                for (auto it = g_lru.rbegin(); it != g_lru.rend() && seen < kSizeMatchWindow; ++it)
                {
                    if ((*it)->pin == g_epoch) continue;
                    if (victim == nullptr) victim = *it;
                    if ((*it)->bytes == bytes) { same = *it; break; }
                    ++seen;
                }
                if (victim == nullptr) return nullptr;
                if (same != nullptr)
                {
                    g_slots.erase(same->key);
                    g_lru.erase(same->lru);
                    g_lru.push_front(same);
                    same->lru = g_lru.begin();
                    same->prefetched = false;
                    return same;
                }
                free_slot(victim);
            }
            if (g_used + bytes > g_budget) return nullptr;
            void* data = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
            if (data == MAP_FAILED) return nullptr;
#if defined(__APPLE__) && TARGET_OS_OSX
            // Wired (macOS; iOS answers memory pressure with jetsam and is unmeasured
            // here): expert weights barely compress (1.2:1 measured), so a slot macOS
            // compresses frees a sixth of itself and turns every later hit into a
            // decompression; overcommitted by 3 GB, a 23 GB cache had 15-17 GB in the
            // compressor (DeepSeek V4.1, 48 GB Mac). Wired, macOS takes the room from
            // the file cache and other programs' compressible memory instead - as
            // Strata wires its arena (metal/src/engine/stream.cpp). munmap unwires.
            static const bool wire = [] {
                const char* e = std::getenv("TS_HOST_MOE_IO_WIRE");
                return e == nullptr || e[0] != '0';
            }();
            if (wire && mlock(data, bytes) != 0 && g_wire_failures++ == 0)
                std::fprintf(stderr, "[expert-io] mlock refused a %.1f MB cache slot (%s): slots that cannot be wired stay "
                    "pageable, and macOS may compress them (TS_HOST_MOE_IO_WIRE=0 stops trying)\n", bytes / 1e6,
                    std::strerror(errno));
#endif
            auto* s = new Slot();
            s->data = static_cast<std::uint8_t*>(data);
            s->bytes = bytes;
            g_lru.push_front(s);
            s->lru = g_lru.begin();
            g_used += bytes;
            return s;
        }

        void clear_locked()
        {
            while (!g_lru.empty()) free_slot(g_lru.back());
        }

        // Under g_mutex: wait for a read-ahead batch still in flight (its workers
        // never take g_mutex) and drop its slots if any read came back short.
        void prefetch_settle_locked()
        {
            if (g_prefetch_team == nullptr || !g_prefetch_team->pending()) return;
            if (!g_prefetch_team->finish())
                for (const void* key : g_prefetch_keys)
                {
                    auto it = g_slots.find(key);
                    if (it != g_slots.end() && it->second->prefetched) free_slot(it->second);
                }
            g_prefetch_keys.clear();
        }
#endif
    }

    void host_expert_io_resolve(const void* const* sources, const std::size_t* bytes, int count,
        const std::uint8_t** resolved)
    {
        for (int i = 0; i < count; ++i) resolved[i] = static_cast<const std::uint8_t*>(sources[i]);
#if !defined(_WIN32)
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_budget == 0 || g_files.empty()) return;
        prefetch_settle_locked();
        ++g_epoch;
        static thread_local std::vector<IoRequest> s_requests;
        s_requests.clear();
        for (int i = 0; i < count; ++i)
        {
            auto found = g_slots.find(sources[i]);
            if (found != g_slots.end() && found->second->bytes == bytes[i])
            {
                Slot* s = found->second;
                s->pin = g_epoch;
                g_lru.erase(s->lru);
                g_lru.push_front(s);
                s->lru = g_lru.begin();
                resolved[i] = s->data;
                ++g_hits;
                if (s->prefetched) { s->prefetched = false; ++g_prefetch_used; }
                continue;
            }
            const MappedFile* file = find_file(sources[i], bytes[i]);
            if (file == nullptr) continue;
            Slot* s = take_slot(bytes[i]);
            if (s == nullptr) continue;     // everything pinned: read the mapping instead
            s->key = sources[i];
            s->pin = g_epoch;
            g_slots[sources[i]] = s;
            s_requests.push_back({file->fd, s->data, bytes[i],
                static_cast<off_t>(static_cast<const std::uint8_t*>(sources[i]) - file->base)});
            resolved[i] = s->data;
            ++g_misses;
            g_read_bytes += bytes[i];
        }
        if (s_requests.empty()) return;
        if (g_team == nullptr)
        {
            static const int workers = [] {
                const char* e = std::getenv("TS_HOST_MOE_IO_THREADS");
                const int n = e != nullptr ? std::atoi(e) : 0;
                return n > 0 && n <= 64 ? n - 1 : 11;
            }();
            g_team = new IoTeam(workers);
        }
        if (!g_team->run(s_requests))
        {
            // A short read leaves a slot half filled: drop every slot of this call
            // and fall back to the mapping for it.
            for (const auto& r : s_requests)
                for (int i = 0; i < count; ++i)
                    if (resolved[i] == r.dst)
                    {
                        resolved[i] = static_cast<const std::uint8_t*>(sources[i]);
                        auto it = g_slots.find(sources[i]);
                        if (it != g_slots.end()) free_slot(it->second);
                    }
        }
#endif
    }

    void host_expert_io_prefetch(const void* const* sources, const std::size_t* bytes, int count)
    {
#if !defined(_WIN32)
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_budget == 0 || g_files.empty() || count <= 0) return;
        prefetch_settle_locked();
        std::vector<IoRequest> requests;
        for (int i = 0; i < count; ++i)
        {
            if (g_slots.count(sources[i]) != 0) continue;
            const MappedFile* file = find_file(sources[i], bytes[i]);
            if (file == nullptr) continue;
            // Pinned with the epoch of the call that just ran, so this batch neither
            // evicts that call's slots nor its own; the next resolve unpins them.
            Slot* s = take_slot(bytes[i]);
            if (s == nullptr) break;
            s->key = sources[i];
            s->pin = g_epoch;
            s->prefetched = true;
            g_slots[sources[i]] = s;
            g_prefetch_keys.push_back(sources[i]);
            requests.push_back({file->fd, s->data, bytes[i],
                static_cast<off_t>(static_cast<const std::uint8_t*>(sources[i]) - file->base)});
            ++g_prefetch_reads;
            g_read_bytes += bytes[i];
        }
        if (requests.empty()) return;
        if (g_prefetch_team == nullptr)
        {
            const char* e = std::getenv("TS_HOST_MOE_IO_THREADS");
            const int n = e != nullptr ? std::atoi(e) : 0;
            g_prefetch_team = new IoTeam(n > 0 && n <= 64 ? n : 12);
        }
        g_prefetch_team->start(std::move(requests));
#else
        (void)sources; (void)bytes; (void)count;
#endif
    }

    void host_expert_io_prefetch_stats(std::uint64_t& reads, std::uint64_t& used)
    {
#if !defined(_WIN32)
        std::lock_guard<std::mutex> lock(g_mutex);
        reads = g_prefetch_reads;
        used = g_prefetch_used;
#else
        reads = used = 0;
#endif
    }

    bool host_expert_io_read_slices(const void* const* sources, const std::size_t* bytes,
        std::uint8_t* const* destinations, int count)
    {
#if !defined(_WIN32)
        std::lock_guard<std::mutex> lock(g_mutex);
        static thread_local std::vector<IoRequest> s_requests;
        s_requests.clear();
        for (int i = 0; i < count; ++i)
        {
            const MappedFile* file = g_files.empty() ? nullptr : find_file(sources[i], bytes[i]);
            if (file == nullptr)
            {
                std::memcpy(destinations[i], sources[i], bytes[i]);
                continue;
            }
            s_requests.push_back({file->fd, destinations[i], bytes[i],
                static_cast<off_t>(static_cast<const std::uint8_t*>(sources[i]) - file->base)});
            g_read_bytes += bytes[i];
        }
        if (s_requests.empty()) return true;
        if (g_team == nullptr)
        {
            const char* e = std::getenv("TS_HOST_MOE_IO_THREADS");
            const int n = e != nullptr ? std::atoi(e) : 0;
            g_team = new IoTeam(n > 0 && n <= 64 ? n - 1 : 11);
        }
        return g_team->run(s_requests);
#else
        for (int i = 0; i < count; ++i) std::memcpy(destinations[i], sources[i], bytes[i]);
        return true;
#endif
    }

    bool host_expert_io_read_rows(const std::uint8_t* table, std::size_t row_bytes, const std::int32_t* rows, int count,
        std::uint8_t* dst)
    {
#if !defined(_WIN32)
        std::lock_guard<std::mutex> lock(g_mutex);
        static thread_local std::vector<IoRequest> s_requests;
        s_requests.clear();
        s_requests.reserve(static_cast<std::size_t>(count));
        const MappedFile* last = nullptr;
        for (int i = 0; i < count; ++i)
        {
            if (rows[i] < 0) return false;
            const std::uint8_t* src = table + static_cast<std::size_t>(rows[i]) * row_bytes;
            if (last == nullptr || src < last->base || src + row_bytes > last->base + last->length)
                last = g_files.empty() ? nullptr : find_file(src, row_bytes);
            if (last == nullptr) return false;
            s_requests.push_back({last->fd, dst + static_cast<std::size_t>(i) * row_bytes, row_bytes,
                static_cast<off_t>(src - last->base)});
        }
        g_read_bytes += static_cast<std::uint64_t>(count) * row_bytes;
        if (g_team == nullptr)
        {
            const char* e = std::getenv("TS_HOST_MOE_IO_THREADS");
            const int n = e != nullptr ? std::atoi(e) : 0;
            g_team = new IoTeam(n > 0 && n <= 64 ? n - 1 : 11);
        }
        return g_team->run(s_requests);
#else
        (void)table; (void)row_bytes; (void)rows; (void)count; (void)dst;
        return false;
#endif
    }

    void host_expert_io_read_slices_async(const void* const* sources, const std::size_t* bytes,
        std::uint8_t* const* destinations, int count)
    {
        host_expert_io_wait_async();
#if !defined(_WIN32)
        std::lock_guard<std::mutex> lock(g_mutex);
        std::vector<IoRequest> requests;
        g_async_src.clear(); g_async_len.clear(); g_async_dst.clear();
        for (int i = 0; i < count; ++i)
        {
            const MappedFile* file = g_files.empty() ? nullptr : find_file(sources[i], bytes[i]);
            if (file == nullptr)
            {
                std::memcpy(destinations[i], sources[i], bytes[i]);
                continue;
            }
            requests.push_back({file->fd, destinations[i], bytes[i],
                static_cast<off_t>(static_cast<const std::uint8_t*>(sources[i]) - file->base)});
            g_async_src.push_back(sources[i]); g_async_len.push_back(bytes[i]); g_async_dst.push_back(destinations[i]);
            g_read_bytes += bytes[i];
        }
        if (requests.empty()) return;
        if (g_async_team == nullptr)
        {
            const char* e = std::getenv("TS_HOST_MOE_IO_THREADS");
            const int n = e != nullptr ? std::atoi(e) : 0;
            g_async_team = new IoTeam(n > 0 && n <= 64 ? n : 12);
        }
        g_async_team->start(std::move(requests));
#else
        for (int i = 0; i < count; ++i) std::memcpy(destinations[i], sources[i], bytes[i]);
#endif
    }

    bool host_expert_io_async_pending()
    {
#if !defined(_WIN32)
        return g_async_team != nullptr && g_async_team->pending();
#else
        return false;
#endif
    }

    bool host_expert_io_wait_async()
    {
#if !defined(_WIN32)
        if (g_async_team == nullptr || !g_async_team->pending()) return true;
        if (g_async_team->finish()) return true;
        // A short or failed read left its slot undefined: copy the batch from the
        // mapping instead (page faults, but every slot is valid before it is read).
        for (std::size_t i = 0; i < g_async_src.size(); ++i)
            std::memcpy(g_async_dst[i], g_async_src[i], g_async_len[i]);
        return false;
#else
        return true;
#endif
    }

    bool host_expert_io_read_ranges(const void* source_base,
        const std::vector<std::pair<std::size_t, std::size_t>>& ranges, std::uint8_t* destination_base)
    {
#if !defined(_WIN32)
        if (source_base == nullptr || destination_base == nullptr || ranges.empty()) return false;
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_files.empty()) return false;
        std::size_t last = 0;
        for (const auto& r : ranges) last = std::max(last, r.first + r.second);
        const MappedFile* file = find_file(source_base, last);
        if (file == nullptr) return false;
        const off_t base = static_cast<off_t>(static_cast<const std::uint8_t*>(source_base) - file->base);
        // 4 MB requests keep every reader busy without one range serializing them.
        constexpr std::size_t kChunk = std::size_t(4) << 20;
        static thread_local std::vector<IoRequest> s_requests;
        s_requests.clear();
        for (const auto& r : ranges)
            for (std::size_t o = 0; o < r.second; o += kChunk)
                s_requests.push_back({file->fd, destination_base + r.first + o, std::min(kChunk, r.second - o),
                    base + static_cast<off_t>(r.first + o)});
        if (g_team == nullptr)
        {
            const char* e = std::getenv("TS_HOST_MOE_IO_THREADS");
            const int n = e != nullptr ? std::atoi(e) : 0;
            g_team = new IoTeam(n > 0 && n <= 64 ? n - 1 : 11);
        }
        g_read_bytes += last;
        return g_team->run(s_requests);
#else
        (void)source_base; (void)ranges; (void)destination_base;
        return false;
#endif
    }

    bool host_expert_io_active()
    {
#if !defined(_WIN32)
        std::lock_guard<std::mutex> lock(g_mutex);
        return !g_files.empty();
#else
        return false;
#endif
    }

    void host_expert_io_on_drop(const void* ptr)
    {
#if !defined(_WIN32)
        if (ptr == nullptr) return;
        host_expert_io_wait_async();   // a batch in flight still reads these files
        std::lock_guard<std::mutex> lock(g_mutex);
        prefetch_settle_locked();      // and a read-ahead one writes into slots
        for (auto it = g_files.begin(); it != g_files.end(); ++it)
        {
            const auto* q = static_cast<const std::uint8_t*>(ptr);
            if (q < it->base || q >= it->base + it->length) continue;
            // The mapping is going away: its cached experts must not outlive it,
            // since a later model can be mapped at the same address.
            for (auto s = g_lru.begin(); s != g_lru.end();)
            {
                Slot* slot = *s++;
                const auto* k = static_cast<const std::uint8_t*>(slot->key);
                if (k >= it->base && k < it->base + it->length) free_slot(slot);
            }
            close(it->fd);
            g_files.erase(it);
            return;
        }
#else
        (void)ptr;
#endif
    }

    void host_expert_io_release()
    {
#if !defined(_WIN32)
        host_expert_io_wait_async();
        IoTeam* team = nullptr;
        IoTeam* async_team = nullptr;
        IoTeam* prefetch_team = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            prefetch_settle_locked();
            clear_locked();
            for (auto& f : g_files) close(f.fd);
            g_files.clear();
            team = g_team;
            g_team = nullptr;
            async_team = g_async_team;
            g_async_team = nullptr;
            prefetch_team = g_prefetch_team;
            g_prefetch_team = nullptr;
        }
        delete team;
        delete async_team;
        delete prefetch_team;
#endif
    }
}

// Serve host-offloaded experts inside [base, base + length) - a read-only
// mapping of `path` from offset 0 - from the explicitly read cache. Returns 1
// when registered (or already registered).
// Scattered fixed-size rows of a registered mapping - a prompt chunk's PLE n-gram rows, 16 a token from a 29 GB
// table - read with one pread each on the I/O team, packed into dst in list order. A fault per row through the mapping
// moved 21-59K rows/s (M5 Pro, a 4096-token chunk's 65K rows in 1.1-3.1 s); parallel reads do not queue on the VM
// object. Returns 1, or 0 (nothing read) when a row lies outside every registered mapping: the caller reads the
// mapping instead.
TSG_EXPORT int TSGgml_HostReadRows(const void* table, long long row_bytes, const std::int32_t* rows, int count, void* dst)
{
    if (table == nullptr || row_bytes <= 0 || rows == nullptr || count <= 0 || dst == nullptr) return 0;
    return tsg::host_expert_io_read_rows(static_cast<const std::uint8_t*>(table), static_cast<std::size_t>(row_bytes),
        rows, count, static_cast<std::uint8_t*>(dst)) ? 1 : 0;
}

TSG_EXPORT int TSGgml_HostExpertIoRegister(const void* base, long long length, const char* path)
{
#if !defined(_WIN32)
    if (base == nullptr || length <= 0 || path == nullptr) return 0;
    std::lock_guard<std::mutex> lock(tsg::g_mutex);
    for (const auto& f : tsg::g_files)
        if (f.base == base) return 1;
    const int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) { tsg::set_last_error(std::string("host expert I/O: cannot open ") + path); return 0; }
#if defined(__APPLE__)
    // The slots are the copy kept; the unified buffer cache need not hold another.
    // F_NOCACHE alone does not do that: the kernel's read-ahead still brings the
    // file's next pages into the cache, ~1.5 GB for every 3 GB of 3 MB slices read
    // (vm_stat file-backed pages, M5 Pro), and ~0 with read-ahead off at the same
    // speed. A DeepSeek V4.1 run reads 150-380 GiB this way: the cache filled with
    // those pages, and macOS compressed anonymous memory to keep them - the slots
    // included, most of them even at a 1 GB budget, so a hit was a decompression.
    fcntl(fd, F_NOCACHE, 1);
    fcntl(fd, F_RDAHEAD, 0);
#elif defined(__linux__)
    // The same on Linux: the slices are read whole into the cache, so the kernel's read-ahead around
    // each one only fills the page cache, which a container's memory limit counts.
    posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
#endif
    tsg::g_files.push_back({static_cast<const std::uint8_t*>(base), static_cast<std::size_t>(length), fd});
    return 1;
#else
    (void)base; (void)length; (void)path;
    return 0;
#endif
}

// Stop serving the mapping that starts at `base` and free its cached experts.
TSG_EXPORT void TSGgml_HostExpertIoUnregister(const void* base)
{
    tsg::host_expert_io_on_drop(base);
}

// What macOS can give now without compressing or swapping anything: free and purgeable
// pages plus the clean file cache (inactive and speculative pages, at most the file-backed
// ones). Other programs' inactive anonymous memory is left out - macOS frees it only by
// compressing it. Strata's arena_avail (metal/src/engine/arena_budget.h). -1 elsewhere.
TSG_EXPORT long long TSGgml_HostAvailableMemoryBytes()
{
#if defined(__APPLE__)
    vm_statistics64_data_t vs{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vs), &count) != KERN_SUCCESS)
        return -1;
    const unsigned long long clean = std::min<unsigned long long>(
        static_cast<unsigned long long>(vs.inactive_count) + vs.speculative_count, vs.external_page_count);
    return static_cast<long long>((static_cast<unsigned long long>(vs.free_count) + vs.purgeable_count + clean)
        * static_cast<unsigned long long>(vm_page_size));
#else
    return -1;
#endif
}

// Byte budget of the cache (0 disables it and frees every slot).
TSG_EXPORT void TSGgml_HostExpertIoSetBudget(long long bytes)
{
#if !defined(_WIN32)
    std::lock_guard<std::mutex> lock(tsg::g_mutex);
    tsg::prefetch_settle_locked();
    tsg::g_budget = bytes > 0 ? static_cast<std::size_t>(bytes) : 0;
    while (tsg::g_used > tsg::g_budget && !tsg::g_lru.empty()) tsg::free_slot(tsg::g_lru.back());
#else
    (void)bytes;
#endif
}

// hits, misses, bytes read, bytes held.
TSG_EXPORT void TSGgml_HostExpertIoStats(long long* stats4)
{
    if (stats4 == nullptr) return;
#if !defined(_WIN32)
    std::lock_guard<std::mutex> lock(tsg::g_mutex);
    stats4[0] = static_cast<long long>(tsg::g_hits);
    stats4[1] = static_cast<long long>(tsg::g_misses);
    stats4[2] = static_cast<long long>(tsg::g_read_bytes);
    stats4[3] = static_cast<long long>(tsg::g_used);
#else
    stats4[0] = stats4[1] = stats4[2] = stats4[3] = 0;
#endif
}
