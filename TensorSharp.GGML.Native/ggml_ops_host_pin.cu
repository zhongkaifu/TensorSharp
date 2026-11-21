// Page-locking (pinning) of host weight ranges for MoE CPU offload.
//
// `--n-cpu-moe N` keeps a layer's experts in system RAM and, for a
// prefill-sized batch, streams them to the accelerator for one graph
// (moe_offload_stream_batch_threshold). That stream is a pure PCIe transfer of
// hundreds of MB per layer per ubatch, so its bandwidth IS the offloaded
// prefill throughput.
//
// The experts live in the GGUF mmap, which is PAGEABLE memory. A cudaMemcpy
// from pageable memory cannot DMA: the driver copies through its own small
// pinned staging buffer, and the measured cost is brutal. On the benchmark host
// (RTX PRO 6000 Blackwell, PCIe 5.0 x16, 1 GiB transfers):
//
//   mmap, pageable          9.3 GB/s
//   mmap, cudaHostRegister  55.6 GB/s     (6.0x)
//   cudaHostAlloc           55.6 GB/s     (the ceiling for this link)
//
// Registering the pages the experts already occupy therefore buys the full link
// speed with NO extra host memory (unlike staging through a pinned mirror,
// which would need a second copy of every offloaded expert) and costs ~65 ms
// per GiB, once, on the first ubatch that streams the layer.
//
// llama.cpp does not do this — `ggml_backend_cuda_register_host_buffer` exists
// in ggml but nothing in llama.cpp calls it, so its `-ncmoe` prefill pays the
// pageable rate. This is where TensorSharp's offloaded prefill overtakes it.
//
// Safety rails, because pinned pages are unswappable and unevictable:
//   * a budget (default 60% of the cgroup/host memory limit) that the
//     registered total may not exceed;
//   * interval bookkeeping, since cudaHostRegister REJECTS a range that
//     overlaps an existing registration and adjacent expert tensors in one
//     mmap share a page after alignment;
//   * every failure is non-fatal — the caller just gets the pageable copy.
//
// TS_HOST_MOE_PIN=0 disables it, TS_HOST_MOE_PIN_MAX_MB overrides the budget.

#include "ggml_ops_internal.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <mutex>

namespace
{
    constexpr std::size_t kPageSize = 4096;

    std::mutex g_pin_mutex;
    // base -> end of every range known to be locked, kept disjoint and sorted
    // so a new request can be split into the gaps between them. g_owned is the
    // subset THIS module registered and may therefore unregister.
    std::map<std::uintptr_t, std::uintptr_t> g_pinned;
    std::map<std::uintptr_t, std::uintptr_t> g_owned;
    std::size_t g_pinned_bytes = 0;
    bool g_pin_disabled = false;

    bool pin_enabled()
    {
        static const bool s_on = []() {
            const char* e = std::getenv("TS_HOST_MOE_PIN");
            return !(e != nullptr && e[0] == '0');
        }();
        return s_on;
    }

    // Bytes this process may hold pinned. Defaults to 60% of the memory the
    // process is actually allowed (cgroup limit when there is one, MemTotal
    // otherwise): pinning is unswappable, so overshooting it does not slow the
    // machine down, it OOM-kills it.
    std::size_t pin_budget_bytes()
    {
        static const std::size_t s_budget = []() -> std::size_t {
            if (const char* e = std::getenv("TS_HOST_MOE_PIN_MAX_MB"))
            {
                const long long mb = std::atoll(e);
                if (mb >= 0)
                    return static_cast<std::size_t>(mb) * 1024ull * 1024ull;
            }

            std::size_t limit = 0;
#ifdef __linux__
            if (FILE* f = std::fopen("/sys/fs/cgroup/memory.max", "r"))   // cgroup v2
            {
                char v[64] = { 0 };
                if (std::fscanf(f, "%63s", v) == 1 && std::strcmp(v, "max") != 0)
                    limit = static_cast<std::size_t>(std::atoll(v));
                std::fclose(f);
            }
            if (limit == 0)
            {
                if (FILE* f = std::fopen("/sys/fs/cgroup/memory/memory.limit_in_bytes", "r"))
                {
                    long long v = 0;
                    if (std::fscanf(f, "%lld", &v) == 1 && v > 0 && v < (1ll << 62))
                        limit = static_cast<std::size_t>(v);
                    std::fclose(f);
                }
            }
            if (limit == 0)
            {
                if (FILE* f = std::fopen("/proc/meminfo", "r"))
                {
                    long long kb = 0;
                    if (std::fscanf(f, "MemTotal: %lld kB", &kb) == 1 && kb > 0)
                        limit = static_cast<std::size_t>(kb) * 1024ull;
                    std::fclose(f);
                }
            }
#endif
            if (limit == 0)
                return 0;   // unknown: refuse to guess, stay pageable
            return limit / 10 * 6;
        }();
        return s_budget;
    }

    // Free memory right now, as the kernel sees it (0 = unknown). Locking pages
    // takes them out of reclaim, so a range worth pinning on a 1.5 TB host can
    // be the difference between "slower prefill" and "OOM killer" on a laptop
    // that is already tight. The budget above bounds the TOTAL; this bounds any
    // single registration against what is actually free at the time.
    std::size_t available_bytes()
    {
#ifdef __linux__
        if (FILE* f = std::fopen("/proc/meminfo", "r"))
        {
            char line[256];
            while (std::fgets(line, sizeof(line), f) != nullptr)
            {
                long long kb = 0;
                if (std::sscanf(line, "MemAvailable: %lld kB", &kb) == 1 && kb > 0)
                {
                    std::fclose(f);
                    return static_cast<std::size_t>(kb) * 1024ull;
                }
            }
            std::fclose(f);
        }
#endif
        return 0;
    }

    // Register [base, end) after the caller has established it overlaps nothing
    // already registered. Read-only + portable when the driver takes it (it
    // lets the driver skip write-tracking for a mapping the device only reads);
    // plain registration otherwise.
    bool register_locked(std::uintptr_t base, std::uintptr_t end)
    {
        void* p = reinterpret_cast<void*>(base);
        const std::size_t len = static_cast<std::size_t>(end - base);

        cudaError_t err = cudaHostRegister(p, len, cudaHostRegisterReadOnly | cudaHostRegisterPortable);
        if (err == cudaErrorHostMemoryAlreadyRegistered)
        {
            // Someone else (ggml, or a previous load in this process) already
            // owns these pages. Record the range so we stop asking, but not as
            // OURS: teardown must not unregister what it did not register.
            cudaGetLastError();
            g_pinned.emplace(base, end);
            return true;
        }
        if (err != cudaSuccess)
        {
            // ReadOnly needs a recent driver and a mapping it accepts; fall back
            // rather than give up the bandwidth.
            cudaGetLastError();
            err = cudaHostRegister(p, len, cudaHostRegisterPortable);
        }
        if (err != cudaSuccess)
        {
            cudaGetLastError();
            err = cudaHostRegister(p, len, cudaHostRegisterDefault);
        }
        if (err != cudaSuccess)
        {
            cudaGetLastError();
            return false;
        }

        g_pinned.emplace(base, end);
        g_owned.emplace(base, end);
        g_pinned_bytes += len;
        return true;
    }
}

namespace tsg
{
    bool host_pin_range(const void* ptr, std::size_t bytes)
    {
        if (ptr == nullptr || bytes == 0 || !pin_enabled())
            return false;

        std::lock_guard<std::mutex> lock(g_pin_mutex);
        if (g_pin_disabled)
            return false;

        const std::uintptr_t raw = reinterpret_cast<std::uintptr_t>(ptr);
        const std::uintptr_t want_base = raw & ~static_cast<std::uintptr_t>(kPageSize - 1);
        const std::uintptr_t want_end = (raw + bytes + kPageSize - 1) & ~static_cast<std::uintptr_t>(kPageSize - 1);

        const std::size_t budget = pin_budget_bytes();
        if (budget == 0)
            return false;

        // Never lock a range the host cannot spare right now. Half of
        // MemAvailable leaves room for the page cache the rest of the model is
        // still being read through.
        const std::size_t avail = available_bytes();
        if (avail != 0 && (want_end - want_base) > avail / 2)
            return false;

        // Walk the gaps between the ranges already registered. cudaHostRegister
        // fails on ANY overlap, and after page alignment the expert tensors of
        // one mmap routinely share their boundary pages.
        bool all = true;
        std::uintptr_t cursor = want_base;
        while (cursor < want_end)
        {
            auto it = g_pinned.upper_bound(cursor);
            std::uintptr_t next_base = want_end;
            if (it != g_pinned.end() && it->first < want_end)
                next_base = it->first;

            // Skip over a range that already covers the cursor.
            if (it != g_pinned.begin())
            {
                auto prev = std::prev(it);
                if (prev->second > cursor)
                {
                    cursor = prev->second;
                    continue;
                }
            }

            if (next_base > cursor)
            {
                const std::size_t len = static_cast<std::size_t>(next_base - cursor);
                if (g_pinned_bytes + len > budget)
                {
                    // Out of budget: stop trying for the rest of the run rather
                    // than repeating a failing syscall on every ubatch.
                    g_pin_disabled = true;
                    std::fprintf(stderr,
                        "[moe-offload] host pinning stopped at %.1f GiB (budget %.1f GiB); "
                        "the remaining expert streams use pageable copies.\n",
                        g_pinned_bytes / 1073741824.0, budget / 1073741824.0);
                    std::fflush(stderr);
                    return false;
                }
                if (!register_locked(cursor, next_base))
                    all = false;
                cursor = next_base;
            }
        }
        return all;
    }

    void host_pin_split(const void* ptr, std::size_t bytes,
                        std::vector<std::pair<std::size_t, std::size_t>>& pieces)
    {
        pieces.clear();
        const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(ptr), end = start + bytes;
        std::lock_guard<std::mutex> lock(g_pin_mutex);
        std::uintptr_t cursor = start;
        while (cursor < end)
        {
            // Inside a registration: the piece ends where it does. In a gap: the
            // piece ends where the next registration begins.
            std::uintptr_t piece_end = end;
            auto it = g_pinned.upper_bound(cursor);
            if (it != g_pinned.begin() && std::prev(it)->second > cursor)
                piece_end = std::min(end, std::prev(it)->second);
            else if (it != g_pinned.end() && it->first < end)
                piece_end = it->first;
            pieces.emplace_back(static_cast<std::size_t>(cursor - start), static_cast<std::size_t>(piece_end - cursor));
            cursor = piece_end;
        }
    }

    std::size_t host_pinned_bytes()
    {
        std::lock_guard<std::mutex> lock(g_pin_mutex);
        return g_pinned_bytes;
    }

    void host_pin_release_ranges(const std::vector<std::pair<std::uintptr_t, std::uintptr_t>>& ranges)
    {
        std::lock_guard<std::mutex> lock(g_pin_mutex);
        auto inside = [&](std::uintptr_t a)
        {
            auto s = std::upper_bound(ranges.begin(), ranges.end(), std::make_pair(a, UINTPTR_MAX));
            return s != ranges.begin() && a < std::prev(s)->second;
        };
        for (auto it = g_owned.begin(); it != g_owned.end();)
        {
            if (!inside(it->first)) { ++it; continue; }
            cudaHostUnregister(reinterpret_cast<void*>(it->first));
            cudaGetLastError();
            it = g_owned.erase(it);
        }
        for (auto it = g_pinned.begin(); it != g_pinned.end();)
        {
            if (!inside(it->first)) { ++it; continue; }
            g_pinned_bytes -= std::min<std::size_t>(g_pinned_bytes, static_cast<std::size_t>(it->second - it->first));
            it = g_pinned.erase(it);
        }
    }

    void host_pin_release_all()
    {
        std::lock_guard<std::mutex> lock(g_pin_mutex);
        for (const auto& kv : g_owned)
        {
            cudaHostUnregister(reinterpret_cast<void*>(kv.first));
            cudaGetLastError();
        }
        g_owned.clear();
        g_pinned.clear();
        g_pinned_bytes = 0;
        g_pin_disabled = false;
    }
}
