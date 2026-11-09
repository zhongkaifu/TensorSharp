// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// A graph slot's arena kept across its rebuilds (tsg::ReuseBufferType, ggml_ops_internal.h).
//
// A prompt rebuilds its span graph for every chunk, because the attention window grows, and each rebuild freed the
// graph allocator's arena and allocated a new one of a few GB. On Metal a new buffer is mapped and made resident
// (residency sets) before the graph runs: ~75 ms a GB, 100-400 ms a chunk on Flash Next (M5 Pro), the "gallocr"
// phase of TS_Q4E_PHASE=1. This buffer type sits over the backend's own and keeps the one buffer its owner freed last,
// handing it to the owner's next allocation when it is large enough: any prompt-sized request (512 MB and up; a 16K
// prompt's arenas grow 3.1 -> 3.5 -> 4.2 GB and its 1187-token tail needs 1.3 GB), and a smaller one only when the
// kept buffer is not wastefully large, so a decode graph does not hold a prompt's arena idle. A new buffer gets a
// quarter of headroom, so a window that grows by a chunk usually still fits. What it hands out are the backend's own
// buffers (same type, same context, same interface but for the free), so every backend path sees an ordinary buffer.
//
// On Metal it also refuses a prompt-sized buffer that would not fit. Metal allocates past its working set without
// complaint and fails the command buffer that touches the memory instead (kIOGPUCommandBufferCallbackErrorOutOfMemory),
// which no later graph recovers from. Flash Next on a 48 GB M5 Pro left ~14 GB of a 38.3 GB working set beside its
// expert tier; Strata's benchmark (16K-token prompts that think for 512 tokens) brought that to 2.2 GB, and the next
// 16K prompt's arena (4-5 GB) took the server down. So before a new buffer of 512 MB or more is made, the working set
// must hold it and a headroom: the kept arenas of every reusing type go first, then the quarter of headroom, and when
// it still does not fit the allocation fails. The span builders report that as "failed to allocate graph tensors" and
// prefill the prompt again in halves (Qwen4ExpModel.ForwardInSpans), as a failed allocation on CUDA always did.
#include "ggml_ops_internal.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace tsg
{
    struct ReuseBufferType
    {
        ggml_backend_buffer_type buft{};          // what the allocator is given; context = this
        ggml_backend_buffer_type_t inner = nullptr;
        ggml_backend_buffer_t kept = nullptr;     // the inner buffer freed last, or null
        int live = 0;                             // buffers handed out and not yet freed
        bool released = false;                    // the owner is gone: free what comes back
    };

    namespace
    {
        constexpr std::size_t kLargeRequest = std::size_t(512) << 20;
        std::mutex g_reuse_mu;
        // Handed-out buffer (its context, the inner buffer's) -> the inner buffer and its type.
        struct Handed { ggml_backend_buffer_t inner; ReuseBufferType* type; };
        std::unordered_map<void*, Handed> g_handed;
        // Every live type, so a buffer that does not fit can take the others' kept arenas first.
        std::unordered_set<ReuseBufferType*> g_types;

        // Free working set on a Metal device, or false elsewhere (a CUDA allocation that does not fit fails by itself).
        bool metal_free_bytes(ggml_backend_buffer_type_t inner, std::size_t& free_bytes, std::size_t& total_bytes)
        {
            ggml_backend_dev_t dev = inner != nullptr ? inner->device : nullptr;
            if (dev == nullptr) return false;
            const char* name = ggml_backend_dev_name(dev);
            if (name == nullptr || std::strncmp(name, "MTL", 3) != 0) return false;
            free_bytes = 0; total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            return total_bytes > 0;
        }

        // Frees every reusing type's kept arena (the next allocation of a type makes its own again); the bytes freed.
        long long release_kept_arenas()
        {
            std::vector<ggml_backend_buffer_t> idle;
            long long bytes = 0;
            {
                std::lock_guard<std::mutex> lock(g_reuse_mu);
                for (ReuseBufferType* t : g_types)
                    if (t->kept != nullptr)
                    {
                        bytes += (long long)ggml_backend_buffer_get_size(t->kept);
                        idle.push_back(t->kept);
                        t->kept = nullptr;
                    }
            }
            for (ggml_backend_buffer_t b : idle) ggml_backend_buffer_free(b);
            return bytes;
        }

        // What a new buffer must leave free: command buffers and the driver allocate beside the graphs.
        std::size_t metal_headroom(std::size_t total_bytes)
        {
            return std::max<std::size_t>(std::size_t(1) << 30, total_bytes / 32);
        }

        void reuse_free_locked(ReuseBufferType* type)
        {
            if (type->released && type->live == 0 && type->kept == nullptr) { g_types.erase(type); delete type; }
        }

        void reuse_buffer_free(ggml_backend_buffer_t buffer)
        {
            ggml_backend_buffer_t drop = nullptr;
            {
                std::lock_guard<std::mutex> lock(g_reuse_mu);
                auto it = g_handed.find(buffer->context);
                if (it == g_handed.end()) return;   // not ours (cannot happen)
                const Handed h = it->second;
                g_handed.erase(it);
                ReuseBufferType* type = h.type;
                --type->live;
                if (type->released) drop = h.inner;
                else
                {
                    drop = type->kept;   // one kept: the newest
                    type->kept = h.inner;
                }
                reuse_free_locked(type);
            }
            if (drop != nullptr) ggml_backend_buffer_free(drop);
        }

        ggml_backend_buffer_t hand_out(ReuseBufferType* type, ggml_backend_buffer_t inner)
        {
            ggml_backend_buffer_i iface = inner->iface;
            iface.free_buffer = reuse_buffer_free;
            ggml_backend_buffer_t outer = ggml_backend_buffer_init(inner->buft, iface, inner->context, inner->size);
            if (outer == nullptr) return nullptr;
            outer->usage = inner->usage;
            g_handed[inner->context] = Handed{inner, type};
            ++type->live;
            return outer;
        }

        ReuseBufferType* type_of(ggml_backend_buffer_type_t buft) { return static_cast<ReuseBufferType*>(buft->context); }

        const char* reuse_get_name(ggml_backend_buffer_type_t buft) { return ggml_backend_buft_name(type_of(buft)->inner); }

        ggml_backend_buffer_t reuse_alloc_buffer(ggml_backend_buffer_type_t buft, std::size_t size)
        {
            ReuseBufferType* type = type_of(buft);
            ggml_backend_buffer_t drop = nullptr;
            {
                std::lock_guard<std::mutex> lock(g_reuse_mu);
                if (type->kept != nullptr)
                {
                    // A prompt-sized request takes any larger kept arena (the slot's next chunk, or the next
                    // prompt, needs it again); a small one (a decode graph) does not hold a large one idle.
                    const std::size_t have = ggml_backend_buffer_get_size(type->kept);
                    if (have >= size && (size >= kLargeRequest || have <= size + size / 2 + (64u << 20)))
                    {
                        ggml_backend_buffer_t inner = type->kept;
                        type->kept = nullptr;
                        if (ggml_backend_buffer_t outer = hand_out(type, inner)) return outer;
                        type->kept = inner;
                    }
                    drop = type->kept;   // too small or too large: it goes before the new one comes
                    type->kept = nullptr;
                }
            }
            if (drop != nullptr) ggml_backend_buffer_free(drop);
            const std::size_t max = ggml_backend_buft_get_max_size(type->inner);
            std::size_t want = size + size / 4;
            if (want > max) want = size;
            std::size_t free_bytes = 0, total_bytes = 0;
            if (size >= kLargeRequest && metal_free_bytes(type->inner, free_bytes, total_bytes))
            {
                const std::size_t headroom = metal_headroom(total_bytes);
                if (free_bytes < want + headroom)
                {
                    // Every idle kept arena goes before this one is refused.
                    release_kept_arenas();
                    metal_free_bytes(type->inner, free_bytes, total_bytes);
                    if (free_bytes < want + headroom) want = size;
                    if (free_bytes < size + headroom)
                    {
                        std::fprintf(stderr, "[ts] a %.2f GB graph arena does not fit the Metal working set (%.2f GB free of "
                                     "%.2f GB, %.2f GB kept free); the graph is refused so it runs in smaller pieces.\n",
                                     size / 1e9, free_bytes / 1e9, total_bytes / 1e9, headroom / 1e9);
                        return nullptr;
                    }
                }
            }
            ggml_backend_buffer_t inner = ggml_backend_buft_alloc_buffer(type->inner, want);
            if (inner == nullptr && want != size) inner = ggml_backend_buft_alloc_buffer(type->inner, size);
            if (inner == nullptr) return nullptr;
            if (vram_log_enabled())
            {
                std::size_t f = 0, t = 0, kept_total = 0;
                metal_free_bytes(type->inner, f, t);
                {
                    std::lock_guard<std::mutex> lock(g_reuse_mu);
                    for (ReuseBufferType* k : g_types) if (k->kept) kept_total += ggml_backend_buffer_get_size(k->kept);
                }
                std::fprintf(stderr, "[TSVRAM] reuse-arena %p %10.1f MB (asked %.1f) | kept %.1f MB | dev free %.1f / %.1f MB\n",
                             (void*)type, ggml_backend_buffer_get_size(inner) / 1048576.0, size / 1048576.0,
                             kept_total / 1048576.0, f / 1048576.0, t / 1048576.0);
            }
            std::lock_guard<std::mutex> lock(g_reuse_mu);
            ggml_backend_buffer_t outer = hand_out(type, inner);
            if (outer == nullptr) ggml_backend_buffer_free(inner);
            return outer;
        }

        std::size_t reuse_get_alignment(ggml_backend_buffer_type_t buft) { return ggml_backend_buft_get_alignment(type_of(buft)->inner); }
        std::size_t reuse_get_max_size(ggml_backend_buffer_type_t buft) { return ggml_backend_buft_get_max_size(type_of(buft)->inner); }
        std::size_t reuse_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor* tensor)
        {
            return ggml_backend_buft_get_alloc_size(type_of(buft)->inner, tensor);
        }
        bool reuse_is_host(ggml_backend_buffer_type_t buft) { return ggml_backend_buft_is_host(type_of(buft)->inner); }
    }

    ReuseBufferType* reuse_buffer_type_new(ggml_backend_buffer_type_t inner)
    {
        if (inner == nullptr) return nullptr;
        auto* type = new ReuseBufferType();
        type->inner = inner;
        {
            std::lock_guard<std::mutex> lock(g_reuse_mu);
            g_types.insert(type);
        }
        type->buft.iface = {};
        type->buft.iface.get_name = reuse_get_name;
        type->buft.iface.alloc_buffer = reuse_alloc_buffer;
        type->buft.iface.get_alignment = reuse_get_alignment;
        type->buft.iface.get_max_size = reuse_get_max_size;
        type->buft.iface.get_alloc_size = reuse_get_alloc_size;
        type->buft.iface.is_host = reuse_is_host;
        type->buft.device = inner->device;
        type->buft.context = type;
        return type;
    }

    ggml_backend_buffer_type_t reuse_buffer_type_get(ReuseBufferType* type) { return type ? &type->buft : nullptr; }

    ggml_backend_buffer_type_t reuse_buffer_type_inner(const ReuseBufferType* type) { return type ? type->inner : nullptr; }

    void reuse_buffer_type_trim(ReuseBufferType* type)
    {
        if (type == nullptr) return;
        ggml_backend_buffer_t drop = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_reuse_mu);
            drop = type->kept;
            type->kept = nullptr;
        }
        if (drop != nullptr) ggml_backend_buffer_free(drop);
    }

    void reuse_buffer_type_release(ReuseBufferType* type)
    {
        if (type == nullptr) return;
        reuse_buffer_type_trim(type);
        std::lock_guard<std::mutex> lock(g_reuse_mu);
        type->released = true;
        reuse_free_locked(type);
    }
}

// Frees the idle arenas every graph slot keeps for its next build (tsg::ReuseBufferType) and returns the bytes freed:
// what a caller that is about to need device memory (a concurrent request's state on Metal) can get back first.
TSG_EXPORT long long TSGgml_ReleaseKeptArenas()
{
    return tsg::release_kept_arenas();
}
