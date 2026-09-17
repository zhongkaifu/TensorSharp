// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#if defined(TSG_GGML_USE_METAL)
#include "ggml-metal.h"
#endif
#include "ggml-cpu.h"
#include "ggml-quants.h"

#if defined(_WIN32)
#define TSG_EXPORT extern "C" __declspec(dllexport)
#elif defined(__clang__) || defined(__GNUC__)
#define TSG_EXPORT extern "C" __attribute__((visibility("default")))
#else
#define TSG_EXPORT extern "C"
#endif

// ---------------------------------------------------------------------------
// Memory pool: reuse GGML context buffers to avoid per-op allocation overhead
// ---------------------------------------------------------------------------
namespace ggml_pool
{
    constexpr std::size_t k_pool_buffer_size = 32 * 1024 * 1024;
    constexpr int k_pool_initial_count = 8;
    constexpr int k_pool_max_count = 32;

    struct PoolEntry
    {
        void* ptr = nullptr;
        std::size_t size = 0;
    };

    PoolEntry acquire(std::size_t required_size);
    void release(PoolEntry e);
    void ensure_initial_pool();
}

// ---------------------------------------------------------------------------
// Internal shared types, globals, and helpers
// ---------------------------------------------------------------------------
// Qwen3.5 arena batched-decode coherence hooks (ggml_ops_qwen35_batched_arena.cpp).
// on_external_touch: a non-arena path is about to use this host cache/state
// pointer — flush the arena slot's rows/state to the host bytes, invalidate the
// resident copies, and retire the slot (no-op for unregistered pointers).
// on_drop: the host bytes were rewritten/freed — discard without flushing.
namespace tsg_q35arena
{
    void on_external_touch(const void* host_ptr);
    void on_drop(const void* host_ptr);
    void on_drop_all();
}

// qwen4exp arena batched-decode coherence hooks (ggml_ops_qwen4exp_arena.cpp).
// Same contract as tsg_q35arena, plus a per-device sweep: the qwen4exp pools
// are per-rank, and the touch/drop caller does not always know which rank a
// holder's slot lives on.
namespace tsg_q4earena
{
    void on_external_touch(const void* host_ptr);
    void on_drop(const void* host_ptr);
    void on_drop_all();
}

namespace tsg
{
    // Qwen3.5 GatedDeltaNet normalizes each Q/K head with
    //
    //     x * rsqrt(sum(x^2) + eps)
    //
    // ggml_l2_norm uses a different epsilon convention. Express the exact
    // operation through RMSNorm instead: RMSNorm(x, eps / n) / sqrt(n), where
    // n is the head dimension (ne[0]). This mirrors llama.cpp's GDN helper.
    inline ggml_tensor* build_gdn_l2_norm(ggml_context* ctx, ggml_tensor* x, float eps)
    {
        const float n = static_cast<float>(x->ne[0]);
        return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps / n), 1.0f / std::sqrt(n));
    }

    // --- Tensor descriptor structs ---

    struct TensorView2DDesc
    {
        void* data;
        int dim0;
        int dim1;
        int stride0;
        int stride1;
        std::int64_t raw_bytes;
    };

    struct TensorView3DDesc
    {
        void* data;
        int dim0;
        int dim1;
        int dim2;
        int stride0;
        int stride1;
        int stride2;
        std::int64_t raw_bytes;
    };

    struct TensorView4DDesc
    {
        void* data;
        int ne0;
        int ne1;
        int ne2;
        int ne3;
        std::int64_t nb1;
        std::int64_t nb2;
        std::int64_t nb3;
        std::int64_t raw_bytes;
    };

    struct ContiguousTensorDesc
    {
        void* data;
        std::int64_t element_count;
        int element_type;
    };

    struct QuantizedWeightDesc
    {
        void* data;
        int ggml_type;
        std::int64_t ne0;
        std::int64_t ne1;
        std::int64_t raw_bytes;
    };

    // --- Data type constants ---

    constexpr int TSG_DTYPE_F32 = 0;
    constexpr int TSG_DTYPE_I32 = 3;

    // --- Op-code enumerations ---

    enum class UnaryOpCode : int
    {
        Neg = 1, Exp = 2, Log = 3, Sqrt = 4, Relu = 5,
        Sigmoid = 6, Tanh = 7, SiLU = 8, Step = 9,
        Abs = 10, Sign = 11, GELU = 12,
    };

    enum class FusedActMulOpCode : int
    {
        SiLUMul = 1, GELUMul = 2, SigmoidMul = 3,
    };

    enum class BinaryTensorOpCode : int
    {
        Add = 1, Sub = 2, Mul = 3, Div = 4,
    };

    enum class BinaryScalarOpCode : int
    {
        Add = 1, Sub = 2, ReverseSub = 3,
        Mul = 4, Div = 5, ReverseDiv = 6,
    };

    enum class ActivationGradOpCode : int
    {
        Relu = 1, Sigmoid = 2, Tanh = 3, SiLU = 4,
    };

    enum class NormOpCode : int
    {
        LayerNorm = 1, RmsNorm = 2,
    };

    enum class ReductionOpCode : int
    {
        Sum = 1, Mean = 2,
    };

    enum class IndexReductionOpCode : int
    {
        Argmin = 1, Argmax = 2,
    };

    // --- Tensor binding helpers ---

    struct TensorBinding
    {
        ggml_tensor* storage = nullptr;
        ggml_tensor* tensor = nullptr;
        std::size_t raw_bytes = 0;
    };

    struct HostPtrBinding
    {
        TensorBinding binding;
        ggml_backend_buffer_t buffer = nullptr;
    };

    // --- Global state ---

    extern thread_local std::string g_last_error;
    // A one-shot that can be UN-shot. This was a std::once_flag, which is exactly
    // right for "initialise the backend once" and exactly wrong for the one thing
    // that turned out to be needed: on Metal a refused command buffer latches
    // has_error inside ggml-metal, and the ONLY way to clear it is to build the
    // backend again. A once_flag cannot be reset, so recovery was impossible in
    // process and an iPhone that had been sent to the background at the wrong
    // moment stayed broken until it was force-quit. See TSGgml_RecreateBackend.
    extern std::mutex g_backend_init_mutex;
    extern bool g_backend_initialized;
    extern int g_backend_type;

    enum class CachedBufferMode
    {
        HostPtr,
        DeviceCopy,
    };

    struct CachedHostBuffer {
        ggml_backend_buffer_t buffer = nullptr;
        std::size_t bytes = 0;
        std::size_t buffer_size = 0;
        CachedBufferMode mode = CachedBufferMode::HostPtr;
        // Allocation size (0 = never) that has been through the backend's
        // init_tensor for this buffer. Every later graph binds a FRESH
        // ggml_tensor to the same buffer and address, and re-running
        // init_tensor is not free: ggml-cuda zeroes a quantized tensor's
        // padding with a cudaMemset, so a 600-weight prefill graph issued 600
        // of them re-zeroing padding that has been zero since the weight was
        // first uploaded.
        //
        // Matched on the ALLOC size rather than a bare flag: that is what
        // decides how much padding init_tensor would zero, so skipping it for
        // a larger tensor would leave the extra padding holding whatever the
        // allocator last put there - which a quantized matmul reads as NaN.
        std::size_t initialized_alloc_size = 0;
        // Address the first successful bind resolved to (the device copy's base,
        // or the host pointer for a zero-copy wrap). Recording it lets a repeat
        // bind attach with two assignments and no backend calls at all.
        void* bound_addr = nullptr;
    };

    // --- Multi-device (tensor-parallel) state -------------------------------
    //
    // The bridge historically owned a single ggml backend (`g_backend`). Tensor
    // parallelism needs one backend per GPU, each with its own buffer caches:
    // a *replicated* weight (an RMSNorm alpha, say) has one host pointer but
    // must have a distinct device-resident copy per rank, so a single
    // pointer-keyed cache would hand rank 1 a buffer that lives on rank 0's GPU.
    //
    // Every rank's state therefore lives in its own DeviceState slot, and the
    // active slot is selected per *thread* (`g_active_rank`). Single-device runs
    // never leave rank 0, so the layout is identical to the old singleton.
    //
    // `g_backend` is kept as a macro that resolves to the active slot's backend
    // (see below) so the ~500 existing use sites keep working unchanged.
    constexpr int TSG_MAX_DEVICES = 16;

    struct DeviceState
    {
        ggml_backend_t backend = nullptr;
        // Index into the ggml device list this slot was initialized from
        // (== the CUDA/Vulkan ordinal). -1 until initialized.
        int device_index = -1;

        std::mutex host_buffer_cache_mutex;
        std::unordered_map<void*, CachedHostBuffer> host_buffer_cache;
        std::mutex preloaded_buffer_cache_mutex;
        std::unordered_map<void*, CachedHostBuffer> preloaded_buffer_cache;

        std::unordered_set<void*> offloadable_keys;
        std::list<void*> offloadable_lru;
        std::unordered_map<void*, std::list<void*>::iterator> offloadable_lru_map;
        std::int64_t offloadable_resident_bytes = 0;
        std::int64_t offloadable_budget = 0;

        std::int64_t device_copy_resident_bytes = 0;
        std::int64_t device_copy_budget_bytes = 0;
    };

    extern DeviceState g_device_states[TSG_MAX_DEVICES];
    // Number of initialized device slots. 1 for the classic single-device path.
    extern std::atomic<int> g_device_count;
    // Active rank for the calling thread. Per-thread so a rank worker pool can
    // drive several GPUs concurrently without stepping on each other.
    extern thread_local int g_active_rank;

    // Cluster-wide tensor-parallel geometry, for a run split across NODES.
    // g_device_count is this process's share; these two describe the whole
    // group. Both stay at their defaults for a single-node run, where the
    // helpers below then reduce to the local values exactly.
    //
    // These exist because expert parallelism is sharded GLOBALLY: the managed
    // side slices the expert stack by the cluster degree, so a kernel that
    // derived its geometry from the LOCAL degree declared a tensor with more
    // experts than the bytes bound to it and let the router address the
    // difference - uninitialised VRAM, silently, on every multi-node MoE token.
    extern std::atomic<int> g_tp_global_degree;   // 0 = single node
    extern std::atomic<int> g_tp_rank_offset;     // this node's first global rank

    // Cluster degree, falling back to the local one when nothing set it.
    inline int tp_global_degree(int local_degree)
    {
        const int g = g_tp_global_degree.load(std::memory_order_relaxed);
        return g > 0 ? g : local_degree;
    }

    // This thread's rank within the whole group.
    inline int tp_global_rank()
    {
        return g_tp_rank_offset.load(std::memory_order_relaxed) + g_active_rank;
    }

    inline DeviceState& dev() { return g_device_states[g_active_rank]; }
    inline DeviceState& dev(int rank) { return g_device_states[rank]; }
    inline ggml_backend_t& active_backend() { return g_device_states[g_active_rank].backend; }

    // Scoped rank switch — restores the previous rank on destruction so an early
    // return or a thrown exception cannot leave the thread pointed at the wrong GPU.
    struct ScopedRank
    {
        int previous;
        explicit ScopedRank(int rank) : previous(g_active_rank) { g_active_rank = rank; }
        ~ScopedRank() { g_active_rank = previous; }
        ScopedRank(const ScopedRank&) = delete;
        ScopedRank& operator=(const ScopedRank&) = delete;
    };

    // --- MoE expert weight offload state ---
    //
    // When a host data pointer is registered as offloadable (via
    // TSGgml_RegisterOffloadable, typically driven by C# for MoE expert
    // weights), its CachedHostBuffer entry participates in an LRU that is
    // bounded by g_offloadable_budget (set via TSGgml_SetOffloadableBudget).
    // Entries past the budget are evicted from the tail of the LRU at the
    // next cache-miss insert, which frees the MTLBuffer wrapper and releases
    // Metal's claim on the underlying mmap pages so the OS may reclaim them.
    // All of these are guarded by g_host_buffer_cache_mutex.
    // (per-rank; see DeviceState)

    // --- Device-copy VRAM budget (discrete-GPU backends, e.g. CUDA) ---
    //
    // When g_device_copy_budget_bytes > 0, try_get_cacheable_tensor_buffer
    // refuses to create a NEW device-local copy buffer once the resident
    // device-copy total would exceed the budget; the caller's bind falls back
    // to the per-graph upload path (the tensor streams to the GPU each graph
    // instead of becoming resident). This prevents VRAM oversubscription: on
    // Windows WDDM an oversubscribed working set is transparently paged
    // between VRAM and system RAM on every command submission, which measured
    // far slower than explicit per-step streaming for diffusion decode.
    // 0 = unlimited (legacy behaviour, correct when everything fits).
    // Both guarded by g_host_buffer_cache_mutex.
    // (per-rank; see DeviceState)

    // --- Backend constants ---

    // Compatibility shims: the pre-TP bridge referenced these as plain globals.
    // Routing them through the active DeviceState keeps every existing use site
    // (~500 across the op files) correct on both the single-device and the
    // tensor-parallel paths without touching them.
#define g_backend                       (::tsg::active_backend())
#define g_host_buffer_cache_mutex       (::tsg::dev().host_buffer_cache_mutex)
#define g_host_buffer_cache             (::tsg::dev().host_buffer_cache)
#define g_preloaded_buffer_cache_mutex  (::tsg::dev().preloaded_buffer_cache_mutex)
#define g_preloaded_buffer_cache        (::tsg::dev().preloaded_buffer_cache)
#define g_offloadable_keys              (::tsg::dev().offloadable_keys)
#define g_offloadable_lru               (::tsg::dev().offloadable_lru)
#define g_offloadable_lru_map           (::tsg::dev().offloadable_lru_map)
#define g_offloadable_resident_bytes    (::tsg::dev().offloadable_resident_bytes)
#define g_offloadable_budget            (::tsg::dev().offloadable_budget)
#define g_device_copy_resident_bytes    (::tsg::dev().device_copy_resident_bytes)
#define g_device_copy_budget_bytes      (::tsg::dev().device_copy_budget_bytes)

    constexpr int BACKEND_TYPE_METAL = 1;
    constexpr int BACKEND_TYPE_CPU = 2;
    constexpr int BACKEND_TYPE_CUDA = 3;
    constexpr int BACKEND_TYPE_VULKAN = 4;

    // --- Async dispatch state ---
    //
    // When async compute is enabled (TSGgml_SetAsyncCompute(1)) per-op kernels can
    // skip the trailing ggml_backend_synchronize and ggml_backend_tensor_get *iff*
    // they used a host-mapped (zero-copy) result binding on the Metal backend.
    //
    // Skipping the sync lets the next op submit its command buffer while the previous
    // one is still running on the GPU. Metal serialises command buffers in queue
    // order, so subsequent ops still observe correct data, and host code that needs
    // to read the result must call ts_metal_host_read_barrier() (exposed to C# via
    // TSGgml_HostReadBarrier and invoked automatically from
    // TensorComputePrimitives.GetFloatPointer).
    //
    // This is the same pattern llama.cpp uses on Metal: ggml_metal_graph_compute
    // returns immediately after committing its command buffer and only blocks on
    // an explicit ggml_backend_synchronize call (see ggml-metal-context.m).

    extern std::atomic<bool> g_async_compute_enabled;
    extern std::atomic<bool> g_pending_gpu_work;

    // True iff we're allowed to defer the sync after this op:
    //   - async mode is on,
    //   - the result was bound zero-copy (host memory directly mapped to a Metal
    //     buffer, so the GPU writes are visible to the host once the command
    //     buffer retires),
    //   - the active backend is Metal (other backends are not exercised under
    //     this lazy-sync model yet).
    inline bool can_defer_sync(bool result_used_zero_copy)
    {
        if (!result_used_zero_copy) return false;
        if (!g_async_compute_enabled.load(std::memory_order_acquire)) return false;
        return g_backend_type == BACKEND_TYPE_METAL;
    }

    inline void mark_pending_gpu_work()
    {
        g_pending_gpu_work.store(true, std::memory_order_release);
    }

    // Bumped once per GGML_LOG_LEVEL_ERROR line ggml emits (see
    // filtered_ggml_log), and latched when one of those lines shows up while we
    // are draining an op's GPU work.
    extern std::atomic<std::uint64_t> g_ggml_error_count;
    extern std::atomic<bool> g_backend_compute_failed;

    // A command buffer that dies on the GPU — Metal's
    // kIOGPUCommandBufferCallbackErrorOutOfMemory, say — is reported to the ggml
    // log and NOWHERE else, because the call that discovers it,
    // ggml_backend_synchronize, returns void. Worse, ggml_backend_graph_compute is
    // compute_async + synchronize and returns the status captured BEFORE that
    // synchronize (ggml-backend.cpp:455-459), while ggml_metal_graph_compute
    // commits its command buffers and returns GGML_STATUS_SUCCESS without waiting
    // for them. So the graph whose command buffer died still reports SUCCESS, its
    // op hands back undefined results, ggml-metal latches its own has_error, and
    // the first thing to actually fail is the NEXT graph — which is why the
    // exception used to name an innocent bystander (the embedding get_rows that
    // opens the following forward).
    //
    // These two drop-ins keep ggml's signatures and return values and add the one
    // thing missing: any error ggml logs while a compute or a drain is in flight
    // belongs to that command buffer, so latch it. The flag is sticky because the
    // backend is — ggml-metal clears has_error only by being recreated. That is what
    // TSGgml_RecreateBackend exists for, and it is the only thing that clears this.
    inline ggml_status compute_graph(ggml_backend_t backend, ggml_cgraph* graph)
    {
        const std::uint64_t before = g_ggml_error_count.load(std::memory_order_acquire);
        const ggml_status status = ggml_backend_graph_compute(backend, graph);
        if (g_ggml_error_count.load(std::memory_order_acquire) != before)
            g_backend_compute_failed.store(true, std::memory_order_release);
        return status;
    }

    // Same, for the paths that submit asynchronously and drain separately.
    inline void sync_backend(ggml_backend_t backend)
    {
        const std::uint64_t before = g_ggml_error_count.load(std::memory_order_acquire);
        ggml_backend_synchronize(backend);
        if (g_ggml_error_count.load(std::memory_order_acquire) != before)
            g_backend_compute_failed.store(true, std::memory_order_release);
    }

    // Standard end-of-op finalisation. Either records that the GPU still has work in
    // flight (lazy-sync path) or syncs and copies the result back to the caller's
    // buffer (eager path, used when the result is not host-mapped or when async
    // mode is disabled).
    inline void finalize_compute(
        bool result_used_zero_copy,
        ggml_tensor* result_storage,
        void* result_data,
        std::size_t result_bytes)
    {
        if (can_defer_sync(result_used_zero_copy))
        {
            mark_pending_gpu_work();
            return;
        }
        sync_backend(g_backend);
        if (!result_used_zero_copy &&
            result_storage != nullptr &&
            result_data != nullptr &&
            result_bytes > 0)
        {
            ggml_backend_tensor_get(result_storage, result_data, 0, result_bytes);
        }
    }

    // Same as finalize_compute() but for ops that already drained their own data
    // back to the host inside the impl (e.g. a partial argmax download). They just
    // need to remember to either sync or mark-pending depending on async mode.
    inline void finalize_compute_no_download()
    {
        if (g_async_compute_enabled.load(std::memory_order_acquire) && g_backend_type == BACKEND_TYPE_METAL)
        {
            mark_pending_gpu_work();
            return;
        }
        sync_backend(g_backend);
    }

    // Variant for ops whose result lives in a ggml-owned backend buffer (no
    // zero-copy host binding) and therefore *must* be copied back to the caller's
    // host buffer. In async mode we queue an async blit-download on the Metal
    // command queue, which returns immediately; the next op (or an explicit host
    // read barrier) will observe the data once the queued blit retires. In eager
    // mode we sync and copy back synchronously.
    inline void finalize_compute_with_download(
        ggml_tensor* result_storage,
        void* result_data,
        std::size_t result_bytes)
    {
        if (g_async_compute_enabled.load(std::memory_order_acquire) && g_backend_type == BACKEND_TYPE_METAL)
        {
            if (result_storage != nullptr && result_data != nullptr && result_bytes > 0)
            {
                ggml_backend_tensor_get_async(g_backend, result_storage, result_data, 0, result_bytes);
            }
            mark_pending_gpu_work();
            return;
        }
        sync_backend(g_backend);
        if (result_storage != nullptr && result_data != nullptr && result_bytes > 0)
        {
            ggml_backend_tensor_get(result_storage, result_data, 0, result_bytes);
        }
    }

    // Drains pending GPU work if any. Called from C# right before host code wants
    // to read tensor data (via Storage.EnsureHostReadable). Returns true when a
    // sync actually happened.
    inline bool host_read_barrier()
    {
        if (g_pending_gpu_work.exchange(false, std::memory_order_acq_rel))
        {
            sync_backend(g_backend);
            return true;
        }
        return false;
    }

    // --- RAII handles ---

    struct ContextHandle
    {
        ggml_context* value = nullptr;

        explicit ContextHandle(ggml_context* ctx) : value(ctx) {}
        ~ContextHandle() { if (value) ggml_free(value); }

        ContextHandle(const ContextHandle&) = delete;
        ContextHandle& operator=(const ContextHandle&) = delete;
    };

    struct PooledContextHandle
    {
        ggml_context* value = nullptr;
        ggml_pool::PoolEntry pool_entry;

        PooledContextHandle() = default;

        bool init(std::size_t required_size)
        {
            pool_entry = ggml_pool::acquire(required_size);
            if (pool_entry.ptr == nullptr)
                return false;
            ggml_init_params params = {};
            params.mem_size = pool_entry.size;
            params.mem_buffer = pool_entry.ptr;
            params.no_alloc = true;
            value = ggml_init(params);
            if (value == nullptr)
            {
                ggml_pool::release(pool_entry);
                pool_entry = {};
                return false;
            }
            return true;
        }

        ~PooledContextHandle()
        {
            if (value != nullptr)
            {
                ggml_free(value);
                value = nullptr;
            }
            if (pool_entry.ptr != nullptr)
            {
                ggml_pool::release(pool_entry);
                pool_entry = {};
            }
        }

        PooledContextHandle(const PooledContextHandle&) = delete;
        PooledContextHandle& operator=(const PooledContextHandle&) = delete;

        // Movable so a caller can hand ownership to something that outlives the
        // call — the tensor-parallel driver runs a graph after the function that
        // built it has returned, and the graph's context has to survive that.
        PooledContextHandle(PooledContextHandle&& other) noexcept
            : value(other.value), pool_entry(other.pool_entry)
        {
            other.value = nullptr;
            other.pool_entry = {};
        }

        PooledContextHandle& operator=(PooledContextHandle&& other) noexcept
        {
            if (this != &other)
            {
                if (value != nullptr) ggml_free(value);
                if (pool_entry.ptr != nullptr) ggml_pool::release(pool_entry);
                value = other.value;
                pool_entry = other.pool_entry;
                other.value = nullptr;
                other.pool_entry = {};
            }
            return *this;
        }
    };

    struct BufferHandle
    {
        ggml_backend_buffer_t value = nullptr;

        explicit BufferHandle(ggml_backend_buffer_t buffer) : value(buffer) {}

        ~BufferHandle()
        {
            if (value != nullptr)
                ggml_backend_buffer_free(value);
        }

        BufferHandle(const BufferHandle&) = delete;
        BufferHandle& operator=(const BufferHandle&) = delete;

        BufferHandle(BufferHandle&& other) noexcept : value(other.value)
        {
            other.value = nullptr;
        }

        BufferHandle& operator=(BufferHandle&& other) noexcept
        {
            if (this != &other)
            {
                if (value != nullptr)
                    ggml_backend_buffer_free(value);
                value = other.value;
                other.value = nullptr;
            }
            return *this;
        }
    };

    // --- Template helpers (must be in header) ---

    template <std::size_t N>
    bool is_non_overlapping_fast_to_slow(const std::array<int, N>& sizes, const std::array<int, N>& strides)
    {
        std::int64_t required_stride = 1;
        for (std::size_t i = 0; i < N; ++i)
        {
            if (sizes[i] <= 0 || strides[i] < 0)
                return false;
            if (sizes[i] == 1)
                continue;
            if (strides[i] < required_stride)
                return false;
            required_stride = static_cast<std::int64_t>(strides[i]) * sizes[i];
        }
        return true;
    }

    // --- Error helpers ---

    void set_last_error(const std::string& message);
    void clear_last_error();

    // A whole-model loader declining a load (not enough VRAM, a --tp layout the
    // devices cannot hold, a missing shard): print the line to stderr exactly as
    // before AND keep it as the thread's last error, so the managed side can put
    // the reason in the exception it throws instead of "see stderr". The hosts
    // turn that exception into one error line and a documented exit code; a
    // pointer at scrollback is useless once the process has exited.
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((format(printf, 1, 2)))
#endif
    void report_load_refusal(const char* format, ...);

    // --- VRAM allocation diagnostics (TS_GGML_LOG_VRAM=1) ---
    //
    // Logs each device-buffer allocation with a tag plus the device's current
    // free/total memory so operators (and CI) can attribute VRAM growth to the
    // subsystem that caused it: weight preload, per-tensor device copies (KV
    // cache / activations), the reuse gallocr, persistent decode/verify graph
    // buffers, and graph-compute pool growth (visible as the free-memory delta
    // between the *-begin / *-end markers). Zero overhead when the env var is
    // unset (single static bool check).
    bool vram_log_enabled();
    void vram_log(const char* tag, std::int64_t bytes);

    // TS_GGML_LOG_VRAM=2 additionally breaks a persistent graph buffer down by
    // tensor. A whole-model graph allocated with ggml_backend_alloc_ctx_tensors
    // gives every intermediate its own slot, so a single "500 MB" line hides
    // which tensor shape actually owns the buffer; this aggregates the still
    // unbound (= about to be allocated) tensors by name, largest first, so the
    // per-layer offender is named rather than guessed at. Call it on the graph's
    // context immediately BEFORE the allocation.
    bool vram_log_verbose();
    void vram_log_ctx_breakdown(const char* tag, ggml_context* ctx, int top_n);

    // --- Backend management ---

    ggml_backend_t create_backend_instance(int backend_type);
    // Create a backend bound to a specific GPU ordinal (tensor parallelism).
    // device_index < 0 keeps the legacy "first available device" behaviour.
    ggml_backend_t create_backend_instance_on_device(int backend_type, int device_index);
    // Number of GPUs the backend type can address (1 for CPU/Metal).
    int gpu_device_count(int backend_type);
    void initialize_backend();
    bool ensure_backend(int backend_type);
    bool ensure_backend();
    bool can_initialize_backend(int backend_type);
    bool backend_supports_op(ggml_tensor* op);

    // --- Tensor parallelism (ggml_ops_tensor_parallel.cpp) ---

    // Resolve the backend's collective-communication entry points for the
    // initialized rank set. False when the backend provides none.
    bool tp_comm_ensure();
    // Behavioural pre-flight of the NCCL device collective on the given CUDA
    // devices (ggml_ops_tp_probe.cu, CUDA builds only): runs one small
    // AllReduce end to end on private streams with a bounded wait and checks
    // the sums that come back. Some hosts (virtualized cloud instances)
    // advertise P2P that never delivers, and NCCL's first collective then
    // spins forever. Returns 1 when the collective works, 0 when it hangs or
    // corrupts (the caller reroutes to the pinned-host "internal" AllReduce),
    // -1 when the probe does not apply (Windows, NCCL absent, disabled).
    int tp_probe_cuda_collective(const int* device_indices, int count);

    // Behavioural check that peer copies between the given devices actually
    // deliver data (some hosts advertise P2P that never completes). 1 = ok,
    // 0 = advertised but broken, -1 = not applicable. Must be called before the
    // process creates its first NCCL communicator for its verdict to be
    // actionable — NCCL caches NCCL_P2P_DISABLE at that point.
    int tp_probe_cuda_peer_access(const int* device_indices, int count);
    // Release the collective context and every rank's AllReduce scratch buffer.
    void tp_comm_free();
    // In-place device AllReduce (sum) over one contiguous F32 tensor per rank.
    bool tp_device_allreduce(ggml_tensor** tensors);

    // True when the backend's collective really reduces (verified once with a
    // one-element AllReduce), false when every call would fall through to the
    // host reduction.
    bool tp_device_allreduce_usable();
    // In-place host AllReduce (sum) over one contiguous F32 buffer per rank.
    void tp_host_allreduce(float** buffers, int n, std::int64_t count);
    void tp_host_allreduce_mt(float** buffers, int n, std::int64_t count);

    // --- Fused tensor-parallel graph execution --------------------------------
    //
    // A fused single-GPU kernel builds one ggml graph for the whole model (or a
    // whole layer). Tensor parallelism needs the same graph per rank, cut at the
    // points where a row-parallel projection produces a *partial* sum that the
    // ranks must add together before anything non-linear touches it.
    //
    // TpRankPlan is that cut: one rank's already-built graph plus the node
    // indices where it must pause. The driver (tp_execute_plans) then runs the
    // schedule llama.cpp's meta backend uses — submit segment k on every rank
    // asynchronously, AllReduce the segment's partials on-device, move to k+1 —
    // so the activations never leave VRAM and the host issues 2*L graph launches
    // per token instead of ~L*30 op round trips.
    //
    // HostMoeSegment (MoE CPU offload, --n-cpu-moe) is the second kind of pause
    // the same graph can need, so it is declared here rather than down in the
    // MoE section: a TpRankPlan carries the offloaded layers' seams and the two
    // schedules merge into one seg_end list. Its own contract is documented
    // under "Whole-model decode graph segmentation for MoE CPU offload" below.
    struct HostMoeSegment
    {
        int layer = 0;

        // --- graph tensors ---
        // Read back from the accelerator after the segment ends.
        ggml_tensor* moe_in = nullptr;     // F32 [hidden, seq]
        ggml_tensor* sel_ids = nullptr;    // I32 [n_used, seq]
        ggml_tensor* weights = nullptr;    // F32 [n_used, seq] (or [1, n_used, seq])
        // Written by the host before the next segment runs. Must be flagged BOTH
        // input and output: ggml-alloc pre-allocates inputs but still frees them
        // after their last consumer, and this one is written from outside the
        // graph, so it has to stay pinned for the whole pass.
        ggml_tensor* moe_out = nullptr;    // F32 [hidden, seq]
        // Optional on-GPU reference chain (TS_HOST_MOE_VERIFY=1).
        ggml_tensor* verify_gpu = nullptr;

        // --- host expert weights (never uploaded) ---
        void* gate_data = nullptr;  int gate_type = 0;  std::int64_t gate_ne0 = 0, gate_ne1 = 0, gate_bytes = 0;
        void* up_data = nullptr;    int up_type = 0;    std::int64_t up_ne0 = 0,   up_ne1 = 0,   up_bytes = 0;
        void* down_data = nullptr;  int down_type = 0;  std::int64_t down_ne0 = 0, down_ne1 = 0, down_bytes = 0;
        const float* gate_bias = nullptr;
        const float* up_bias = nullptr;
        const float* down_bias = nullptr;

        int activation = 0;          // tsg MoE activation enum (see moe_ffn_host_experts)
        float oai_alpha = 1.702f;
        float oai_limit = 7.0f;

        int num_experts = 0;
        int n_used = 0;
        int seq_len = 1;             // 1 for autoregressive decode, C for DiffusionGemma blocks
        int hidden = 0;
        int n_ff = 0;

        // --- tensor parallelism ---
        // Under TP the offloaded layer is computed ONCE, on the host, from the
        // unsharded expert weights: the host backend is a single mutex-guarded
        // thread pool, so splitting the matmul across ranks would serialize
        // anyway while costing an extra download per rank. What differs is
        // where the one result goes, and that depends on whether the graph
        // still reduces it downstream:
        //
        //   tp_reduced = 1  the layer's MoE output feeds a later AllReduce
        //                   (Qwen3.5: moe_out + Megatron shared expert). Rank 0
        //                   gets the full value, every other rank gets zeros, so
        //                   the collective sums to exactly the single-GPU value.
        //   tp_reduced = 0  nothing reduces it (Gemma 4 skips the MoE partial
        //                   when the layer is offloaded). Every rank gets the
        //                   full value.
        //
        // Ignored outside tp_execute_plans.
        int tp_reduced = 0;
    };

    struct TpRankPlan
    {
        ggml_cgraph* graph = nullptr;
        // Exclusive end node index of each segment; the last entry is n_nodes.
        std::vector<int> seg_end;
        // Partial tensor to reduce after segment k (size == seg_end.size()-1).
        // May be null at a boundary that exists only to hand a layer's MoE to
        // the host — see host_moe below.
        std::vector<ggml_tensor*> ar_tensor;
        // MoE CPU offload under tensor parallelism. The offloaded layers' seams
        // are boundaries too, so the two segment schedules are MERGED into the
        // one seg_end above rather than nested: host_moe_at[k] is the index into
        // host_moe of the seam that runs after segment k, or -1 when segment k
        // ends in an AllReduce only. Filled by tp_plan_segments from host_moe,
        // which the kernel sets before planning.
        std::vector<HostMoeSegment> host_moe;
        std::vector<int> host_moe_at;
        // Optional result download, performed after the final synchronize.
        ggml_tensor* out_tensor = nullptr;
        void* out_host = nullptr;
        std::size_t out_bytes = 0;
        // Additional post-execution downloads (e.g. the per-layer GDN states a
        // whole-model prefill graph leaves behind), performed with out_tensor.
        struct ExtraDownload { ggml_tensor* tensor; void* host; std::size_t bytes; };
        std::vector<ExtraDownload> extra_out;

        void clear()
        {
            graph = nullptr;
            seg_end.clear();
            ar_tensor.clear();
            host_moe.clear();
            host_moe_at.clear();
            out_tensor = nullptr;
            out_host = nullptr;
            out_bytes = 0;
            extra_out.clear();
        }
        bool valid() const { return graph != nullptr && !seg_end.empty(); }
    };

    // Fill plan.seg_end from the graph positions of plan.ar_tensor's nodes (or
    // of `boundary`, when the node that ends a segment is a no-op view of the
    // partial rather than the partial itself). Returns false when a boundary
    // tensor is absent from the graph, which means the caller recorded a node
    // the builder did not emit.
    bool tp_plan_segments(TpRankPlan& plan, const std::vector<ggml_tensor*>& boundary);

    // Run one segmented graph per rank. `plans[r]` must describe rank r and all
    // ranks must agree on the segment count. Returns false (with the last error
    // set) when a submission or a collective fails.
    /// Cross-node AllReduce hook: sums `data` element-wise across every node
    /// of the cluster and leaves the result in place on all of them. Returns
    /// false to abort the forward pass. Null for a single-node run.
    using TpDistributedAllReduce = bool (*)(void* user, float* data, int count);

    bool tp_execute_plans(TpRankPlan** plans, int rank_count,
                          TpDistributedAllReduce cross_node = nullptr,
                          void* cross_node_user = nullptr);

    // True when the fused TP path is usable: more than one rank and a device
    // collective to reduce with. Without a collective every segment boundary
    // would cost a host round trip, which is exactly what this path exists to
    // remove, so the caller should fall back to the generic per-op forward.
    bool tp_fused_available(int rank_count, bool distributed = false);

    // ------------------------------------------------------------------
    // MoE CPU offload (--n-cpu-moe / --cpu-moe)
    // ------------------------------------------------------------------
    // Routed-expert FFN evaluated on the host ggml CPU backend, reading the
    // stacked per-expert GGUF blocks zero-copy out of their mmap. Implemented
    // in ggml_ops_moe.cpp next to the accelerator version so both share one
    // graph builder and stay numerically identical.
    //
    // The whole-model decode graphs call this between segments: the GPU graph
    // pauses after producing the layer's post-attention norm and its router
    // decision, the host multiplies the selected experts, and the result is
    // uploaded into the graph's moe_out input before the next segment runs.
    //
    //   hidden_in         [seq_len * hidden_dim] F32, host
    //   hidden_out        [seq_len * hidden_dim] F32, host (written)
    //   selected_experts  [seq_len * n_used]     I32, host
    //   routing_weights   [seq_len * n_used]     F32, host
    //   gate/up/down      stacked [ne0, ne1, num_experts] quantized blocks.
    //                     up_data == nullptr selects the fused gate_up layout
    //                     (Gemma 4 / DiffusionGemma) or, for ReLU^2, the single
    //                     projection Nemotron-H uses.
    //   *_bias            optional per-expert bias, host F32; null to skip.
    //   activation_type   0 swiglu split, 1 swiglu OAI, 2 geglu split,
    //                     3 relu(up)^2 — same enum the fused MoE kernel takes.
    //
    // Returns false with the last error set on any failure; callers fall back
    // to the accelerator path rather than producing wrong output.
    bool moe_ffn_host_experts(
        const float* hidden_in,
        float* hidden_out,
        int seq_len,
        int hidden_dim,
        int n_ff,
        int num_experts,
        int n_used,
        const std::int32_t* selected_experts,
        const float* routing_weights,
        void* gate_data, int gate_type, std::int64_t gate_ne0, std::int64_t gate_ne1, std::int64_t gate_bytes,
        void* up_data,   int up_type,   std::int64_t up_ne0,   std::int64_t up_ne1,   std::int64_t up_bytes,
        void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_bytes,
        const float* gate_bias, const float* up_bias, const float* down_bias,
        int activation_type, float oai_alpha, float oai_limit,
        // Whether a prefill-sized batch may be evaluated on the accelerator with
        // the experts streamed in for that one graph instead of on the host (the
        // experts are never left resident either way). False under tensor
        // parallelism, which needs the single host evaluation over the unsharded
        // stack. See moe_offload_stream_batch_threshold.
        bool allow_device_stream = true);

    // ------------------------------------------------------------------
    // Host page-locking for the streamed-expert path (see ggml_ops_host_pin.cu)
    // ------------------------------------------------------------------
    // Page-locks the host range holding an offloaded expert stack so the H2D
    // stream DMAs instead of trickling through the driver's staging buffer:
    // measured 9.3 -> 55.6 GB/s on PCIe 5.0 x16 with the GGUF mmap. Costs no
    // extra host memory (the mmap's own pages are locked) and ~65 ms/GiB once.
    // Returns false when pinning is disabled, over budget, or unsupported, in
    // which case the caller simply gets the slower pageable copy.
#if defined(TSG_GGML_USE_CUDA)
    bool host_pin_range(const void* ptr, std::size_t bytes);
    std::size_t host_pinned_bytes();
    void host_pin_release_all();
#else
    inline bool host_pin_range(const void*, std::size_t) { return false; }
    inline std::size_t host_pinned_bytes() { return 0; }
    inline void host_pin_release_all() {}
#endif

    // Worker-thread count for the host MoE matmul (--cpu-moe-threads). Zero
    // clears the explicit override. Must be an
    // explicit call rather than an environment variable: .NET's
    // Environment.SetEnvironmentVariable does not write the native environment
    // on Linux, so std::getenv here never observed the flag.
    void moe_set_host_thread_count(int threads);

    // Only the explicit native/CLI override, or zero when unset. Model-specific
    // backends can apply their own automatic count and environment precedence
    // without accidentally inheriting the generic host-MoE default policy.
    int host_moe_explicit_thread_count();

    // Default worker count for a host-side expert matmul: half the usable CPU
    // parallelism, capped at 64 (the decode-side matmul is one token wide, so
    // beyond a few dozen workers each extra thread is a barrier participant,
    // not bandwidth). Shared so every host-MoE caller picks the same number.
    int host_moe_default_thread_count();

    // Free the persistent host MoE backend (thread pool). Called from the
    // backend teardown path so a model reload does not leak worker threads.
    void moe_ffn_host_release();

    // CPU parallelism this process can ACTUALLY use: hardware_concurrency
    // clamped by the scheduler affinity mask and by the cgroup CPU quota.
    //
    // Containers routinely report every host thread (96) while granting a
    // fraction of them (a 23.8-CPU quota). ggml's worker pool spins at its
    // per-node barriers, so oversubscribing a quota does not slow down in
    // proportion — it collapses: every barrier waits for threads the scheduler
    // will not run until the cgroup refills, which costs whole timeslices. On
    // the 4x-oversubscribed case measured here that turned a 7.5 ms offloaded
    // DeepSeek V4 layer into a 330 ms one (4.5 s per token, 25x slower than the
    // same run with a quota-sized pool).
    int available_cpu_parallelism();

    // ------------------------------------------------------------------
    // Whole-model decode graph segmentation for MoE CPU offload
    // ------------------------------------------------------------------
    // Every whole-model decode kernel (Qwen3.5/3.6, Gemma 4 MoE, GPT-OSS,
    // DiffusionGemma) runs a token as ONE accelerator graph. That is worth
    // roughly 3x over per-layer dispatch, so an offloaded layer must not cost
    // the model its fused graph.
    //
    // The shape below keeps it. For each offloaded layer the builder omits the
    // routed-expert mul_mat_id chain and instead emits three outputs — the MoE
    // input, the router's top-k ids and its final routing weights — plus one
    // `moe_out` tensor the host fills. Execution then alternates:
    //
    //   [accelerator segment] -> read moe_in/ids/weights -> [host expert matmul]
    //   -> upload moe_out -> [accelerator segment] -> ...
    //
    // Attention, norms, the router, any shared/dense expert and the LM head all
    // stay in the accelerator graph. Each boundary costs one synchronize plus
    // (seq*hidden + 2*seq*n_used) values over the bus, which is noise next to
    // the host matmul it exists to feed.
    //
    // This is the same segmented-replay mechanism tp_execute_plans uses for
    // tensor parallelism, factored out so all four kernels share one
    // implementation (and one place where the seam can be got wrong).
    // HostMoeSegment itself is declared above, next to TpRankPlan: a tensor-
    // parallel plan carries the offloaded layers' seams, so the two segment
    // schedules merge into one and the type has to precede both.

    // TS_HOST_MOE_VERIFY=1: build the on-GPU expert chain alongside the host one
    // and print their divergence per layer. Only for A/B validating the seam on
    // a model that also fits in VRAM; it defeats the memory saving by design.
    bool host_moe_verify_enabled();

    // Node index of `t` in `graph`, or -1 when the builder never emitted it.
    int host_moe_graph_node_index(ggml_cgraph* graph, ggml_tensor* t);

    // ------------------------------------------------------------------
    // Whole-model graph profiler (TS_GGML_NODE_PROFILE)
    // ------------------------------------------------------------------
    // Drop-in for ggml_backend_graph_compute in the whole-model kernels. Unset,
    // it IS ggml_backend_graph_compute — same call, no branch worth measuring.
    // Set to 1 it runs the graph one node at a time with a synchronize around
    // each, accumulates the cost per op type (and per (op, shape) for the worst
    // offenders) and dumps a table every TS_GGML_NODE_PROFILE_EVERY calls
    // (default 64).
    //
    // The per-node synchronize is itself ~5-10 us, so the profiled total runs
    // well above the real graph time; the dump prints both so the launch tax is
    // visible rather than hidden. What it is for is the SHAPE of the cost — which
    // op type owns the graph — which is exactly what a "we are 0.7x llama.cpp"
    // gap needs before anything is rewritten.
    bool graph_node_profile_enabled();
    ggml_status graph_compute_profiled(ggml_backend_t backend, ggml_cgraph* graph, const char* tag);

    // Platform convolution offload for the VAEs (MPSGraph on Metal, cuDNN on
    // CUDA). ggml lowers conv2d to im2col + mul_mat, which dominates a VAE graph
    // and materialises 9x the input for a 3x3 kernel; the vendor libraries
    // convolve directly. fast_conv_enabled() says whether to emit un-lowered
    // CONV_2D nodes, and graph_compute_fast_conv() executes such a graph.
    bool fast_conv_enabled();
    ggml_status graph_compute_fast_conv(ggml_cgraph* graph, const char* tag);

    // ------------------------------------------------------------------
    // Whole-model kernel phase timer (TS_GGML_PHASE_TIMING)
    // ------------------------------------------------------------------
    // A prefill kernel is graph build -> weight bind -> buffer alloc -> input
    // upload -> compute -> result download, and only the compute part scales
    // with the token count. When a prefill is fast at 2048 tokens and slow at
    // 512 the cost is in the other five, and this says which. Zero cost when the
    // env var is unset (one relaxed atomic load per mark).
    bool phase_timing_enabled();
    void phase_timing_report(const char* tag, const char* const* phases, const double* ms, int count);

    struct PhaseTimer
    {
        explicit PhaseTimer(const char* tag) : tag_(tag), on_(phase_timing_enabled())
        {
            if (on_) last_ = now();
        }

        void mark(const char* phase)
        {
            if (!on_ || count_ >= kMaxPhases) return;
            const double t = now();
            phases_[count_] = phase;
            ms_[count_] = t - last_;
            last_ = t;
            ++count_;
        }

        ~PhaseTimer()
        {
            if (on_ && count_ > 0) phase_timing_report(tag_, phases_, ms_, count_);
        }

        PhaseTimer(const PhaseTimer&) = delete;
        PhaseTimer& operator=(const PhaseTimer&) = delete;

    private:
        static double now();
        static constexpr int kMaxPhases = 12;
        const char* tag_;
        bool on_;
        double last_ = 0.0;
        int count_ = 0;
        const char* phases_[kMaxPhases] = {};
        double ms_[kMaxPhases] = {};
    };

    // Turn recorded segments into node cut points. The cut for a segment goes
    // after the last of its boundary tensors, whichever order the expander
    // emitted them in; `seg_end` ends with the graph's node count so the tail
    // after the final offloaded layer runs too. Returns false (last error set)
    // when a boundary tensor is missing or the segments are not in graph order —
    // either would silently feed the host stale values.
    bool host_moe_build_segment_ends(
        ggml_cgraph* graph,
        const std::vector<HostMoeSegment>& segments,
        std::vector<int>& seg_end,
        const char* kernel_name);

    // Execute `graph` as the alternating accelerator/host sequence described
    // above. `seg_end` must come from host_moe_build_segment_ends.
    bool host_moe_execute_segments(
        ggml_cgraph* graph,
        const std::vector<HostMoeSegment>& segments,
        const std::vector<int>& seg_end,
        const char* kernel_name);

    // One seam, split so tensor parallelism can reuse it. `compute` downloads
    // the segment's inputs from the ACTIVE rank and leaves the expert result in
    // `out` (resized to hidden*seq_len) without uploading; `upload` pushes a
    // result into the active rank's moe_out, and `zero` fills it with zeros for
    // the ranks that must not contribute (see HostMoeSegment::tp_reduced).
    // Staged copies of a seam's inputs, valid until the next
    // host_moe_compute_segment call on the same thread. Only the diagnostics
    // want them; pass nullptr otherwise.
    struct HostMoeStagedInputs
    {
        const float* moe_in = nullptr;
        const std::int32_t* sel_ids = nullptr;
        const float* weights = nullptr;
    };
    bool host_moe_compute_segment(const HostMoeSegment& hm, std::vector<float>& out, const char* kernel_name,
                                  HostMoeStagedInputs* staged = nullptr,
                                  bool allow_device_stream = true);
    void host_moe_upload_segment(const HostMoeSegment& hm, const float* data);
    void host_moe_zero_segment(const HostMoeSegment& hm);

    // Memoized static device properties. ggml_backend_dev_get_props also
    // refreshes free/total device memory, which some backends answer with an
    // expensive query — ggml-vulkan on Windows re-enumerates the physical
    // devices and reads the WDDM memory budget, ~4 ms per call. The binding
    // predicates (prefers_device_local_cache / host_ptr_buffer_capable) run
    // hundreds of times per decode-graph build, which turned that into
    // ~1.5 s/token on Vulkan. They only need the type/caps fields, which are
    // immutable per device, so cache those once.
    struct DeviceStaticProps
    {
        enum ggml_backend_dev_type type;
        bool buffer_from_host_ptr;
    };
    DeviceStaticProps get_device_static_props(ggml_backend_dev_t dev);

    // Allocate the unallocated (compute/intermediate) tensors of `ctx` into a
    // persistent, reusable backend buffer instead of freshly allocating a new
    // backend buffer on every call. Mirrors ggml_backend_alloc_ctx_tensors but
    // keeps the underlying buffer cached (grown on demand) across calls, which
    // avoids the ~20 ms/call Metal buffer allocation that dominates per-layer
    // prefill. Returns true on success. The caller must NOT free the buffer;
    // it is owned by the cache and released in TSGgml_Shutdown. Safe only when
    // the previous graph that used the buffer has been synchronized before the
    // next graph_compute (the per-layer prefill host_read_barrier guarantees
    // this). Returns false (caller should fall back to the stock allocator) if
    // the required size exceeds a single backend buffer's maximum.
    // Supplying the final graph lets Metal share only attention workspaces;
    // ordinary activations/state retain unique slots in the same cached slab.
    bool alloc_ctx_tensors_reuse(ggml_context* ctx, ggml_cgraph* graph = nullptr);
    // Free the cached reuse buffer (called from TSGgml_Shutdown).
    void free_reuse_compute_buffer();

    // Allocate a graph's intermediates into a persistent, reused gallocr (grown on
    // demand, never freed per call) — for large multi-token fused graphs (e.g. the
    // MTP MoE verify) whose per-call gallocr alloc/free would fragment Metal VRAM.
    // Returns false if unavailable (caller falls back to its own gallocr).
    bool alloc_graph_reuse_gallocr(ggml_cgraph* graph);

    // Same, but backed by a SEPARATE persistent allocator reserved for the
    // MoE-offload streaming graphs. They run nested inside a partially executed
    // outer graph, so they must not re-plan the allocator that outer graph's
    // tensors are placed in. See alloc_graph_in_gallocr_slot.
    bool alloc_graph_moe_stream_gallocr(ggml_cgraph* graph);
    // Run Metal's backend graph optimizer before any graph allocation. Direct
    // backend graph_compute calls do not invoke this hook themselves.
    //
    // The shared allocators (alloc_graph_reuse_gallocr / alloc_graph_moe_stream_gallocr)
    // call this for every graph they place, so a whole-model kernel gets it for
    // free. Paths that allocate with ggml_backend_alloc_ctx_tensors directly (the
    // persistent captured graphs, the small-N bump-allocated ones) still have to
    // call it themselves, before that allocation.
    void optimize_graph_for_metal(ggml_cgraph* graph);

    // Some whole-model graphs are executed as ORDERED SLICES of their node array
    // (ggml_graph_view): the host-MoE seams stop at a node INDEX to run an
    // offloaded expert on the CPU, the tensor-parallel driver reduces at recorded
    // cut points, and the vendor-conv runner pulls out CONV_2D nodes one at a
    // time. Metal's graph_optimize PERMUTES gf->nodes[], so on such a graph it
    // would move work across a seam whose position is an index — the host would
    // then read an activation the GPU has not produced yet, which shows up as
    // wrong numbers rather than a failure.
    //
    // A builder that will slice its graph holds one of these across the
    // allocation (and across any explicit optimize_graph_for_metal call);
    // the optimizer then leaves that graph's order alone. Construct with the
    // condition itself — `SuppressGraphReorder keep_order(tp_mode || !host_moe.empty());`
    // — so the guard reads as the reason it exists.
    struct SuppressGraphReorder
    {
        explicit SuppressGraphReorder(bool active);
        ~SuppressGraphReorder();
        SuppressGraphReorder(const SuppressGraphReorder&) = delete;
        SuppressGraphReorder& operator=(const SuppressGraphReorder&) = delete;

    private:
        bool active_;
    };
    // Free the cached reuse gallocr (called from TSGgml_Shutdown / backend reset).
    void free_reuse_gallocr();
    // Free the calling thread's cached prefill-attention sessions (defined in
    // ggml_ops_norm_attn.cpp; called from TSGgml_Shutdown so their CUDA
    // buffers are released before driver teardown).
    void free_prefill_attn_sessions();

    // --- Size / layout queries ---

    std::size_t required_raw_bytes(const TensorView2DDesc& desc);
    std::size_t required_raw_bytes(const TensorView3DDesc& desc);
    std::size_t required_raw_bytes(const TensorView4DDesc& desc);
    std::size_t logical_bytes(const TensorView2DDesc& desc);
    std::size_t logical_bytes(const TensorView3DDesc& desc);
    std::size_t logical_bytes(const TensorView4DDesc& desc);
    std::size_t logical_row_bytes(const TensorView2DDesc& desc);
    std::size_t raw_row_bytes(const TensorView2DDesc& desc);

    constexpr std::size_t k_ggml_cuda_max_copy_bytes = static_cast<std::size_t>(std::numeric_limits<int>::max());

    TensorView2DDesc slice_rows_2d(const TensorView2DDesc& desc, int row_start, int row_count);
    int limit_rows_for_cuda_copy(int current_limit, const TensorView2DDesc& desc);

    // --- Validation ---

    bool validate_desc(const TensorView2DDesc& desc, const char* name);
    bool validate_desc(const TensorView3DDesc& desc, const char* name);
    bool validate_desc(const TensorView4DDesc& desc, const char* name);
    bool validate_desc(const ContiguousTensorDesc& desc, const char* name);
    bool read_i32_values(std::vector<std::int32_t>& output, const ContiguousTensorDesc& desc, const char* name);

    // --- Layout queries ---

    bool can_map_standard_view(const TensorView2DDesc& desc);
    bool can_map_standard_view(const TensorView3DDesc& desc);
    bool can_map_standard_view(const TensorView4DDesc& desc);
    bool can_map_m2_direct(const TensorView2DDesc& desc);
    bool can_map_m2_direct(const TensorView3DDesc& desc);

    // --- Pointer / buffer utilities ---

    bool is_pointer_aligned(const void* ptr, std::size_t alignment);
    std::size_t get_host_ptr_alignment(ggml_backend_t backend, ggml_backend_dev_t dev);
    bool prefers_device_local_cache(ggml_backend_dev_t dev);
    bool can_use_host_ptr_buffer(ggml_backend_t backend, ggml_backend_dev_t dev, const void* ptr, std::size_t size);
    bool host_ptr_buffer_capable(ggml_backend_t backend, ggml_backend_dev_t dev, const void* ptr, std::size_t size);
    // Returns true when a cached backend buffer was actually freed. Callers that
    // must invalidate downstream state keyed on that buffer use the result to skip
    // the work when the pointer had no device copy to begin with.
    bool invalidate_cached_buffer(void* data);

    // allow_unified_weight: when true the prefers_device_local_cache gate is
    // bypassed and only raw capability (host_ptr_buffer_capable) is required.
    // Set by the read-only-weight path on unified-memory Metal so weights are
    // wrapped zero-copy instead of duplicated into a device-local copy.
    bool try_get_host_ptr_buffer(
        ggml_backend_t backend, ggml_backend_dev_t dev,
        void* data, std::size_t bytes, bool cacheable,
        ggml_backend_buffer_t& out_buffer,
        bool allow_unified_weight = false);

    bool try_peek_cached_device_copy(const void* data, std::size_t bytes,
                                     ggml_backend_buffer_t& out_buffer, void*& out_addr);
    bool try_get_cacheable_tensor_buffer(
        ggml_backend_t backend, ggml_backend_dev_t dev,
        ggml_tensor* tensor, void* data, std::size_t bytes,
        ggml_backend_buffer_t& out_buffer, void*& out_addr, bool& out_needs_upload,
        enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // Bind `tensor` to the cached device buffer for `data`, doing the lookup and
    // the attach in one step. Returns false when there is no usable cached
    // buffer (the caller falls back to a host-pointer wrap or an upload).
    //
    // The reason this exists rather than callers pairing
    // try_get_cacheable_tensor_buffer with ggml_backend_tensor_alloc: the second
    // half of that pair runs the backend's init_tensor every time, and for a
    // quantized weight ggml-cuda's init_tensor issues a cudaMemset to zero the
    // block padding. A whole-model prefill graph binds ~600 weights per call, so
    // that was ~600 memsets per prefill (7.5 ms on a 30-layer Gemma 4 — half the
    // gap to llama.cpp at 512 tokens) re-zeroing padding that has been zero since
    // the weight was first uploaded. The cache entry remembers that it has been
    // initialized, and repeat binds attach directly. The flag lives in the entry,
    // so freeing/evicting the buffer clears it with the rest.
    bool try_bind_cached_tensor(
        ggml_backend_t backend, ggml_backend_dev_t dev,
        ggml_tensor* tensor, void* data, std::size_t bytes, bool& out_needs_upload,
        enum ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    bool sync_cached_buffer_to_host(void* data, std::size_t bytes);

    // --- Tensor binding creation ---

    TensorBinding create_standard_binding(ggml_context* ctx, const TensorView2DDesc& desc);
    TensorBinding create_standard_binding(ggml_context* ctx, const TensorView3DDesc& desc);
    TensorBinding create_standard_binding(ggml_context* ctx, const TensorView4DDesc& desc);
    TensorBinding create_contiguous_binding(ggml_context* ctx, const ContiguousTensorDesc& desc);
    TensorBinding create_direct_m2_binding(ggml_context* ctx, const TensorView2DDesc& desc);
    TensorBinding create_direct_m2_binding(ggml_context* ctx, const TensorView3DDesc& desc);

    std::vector<float> pack_m2(const TensorView2DDesc& desc);
    std::vector<float> pack_m2(const TensorView3DDesc& desc);
    std::vector<float> pack_standard(const TensorView2DDesc& desc);
    std::vector<float> pack_standard(const TensorView3DDesc& desc);

    TensorBinding create_packed_m2_binding(ggml_context* ctx, const TensorView2DDesc& desc, std::vector<float>& packed);
    TensorBinding create_packed_m2_binding(ggml_context* ctx, const TensorView3DDesc& desc, std::vector<float>& packed);
    TensorBinding create_packed_standard_binding(ggml_context* ctx, const TensorView2DDesc& desc, std::vector<float>& packed);
    TensorBinding create_packed_standard_binding(ggml_context* ctx, const TensorView3DDesc& desc, std::vector<float>& packed);

    void upload_binding(const TensorBinding& binding, const void* data, std::size_t size);

    // Map a preloaded weight's opaque cache key back to the host bytes it stands
    // for. Uploading straight from a cache key would dereference a GCHandle;
    // every upload path that may be handed one must go through this first.
    const void* resolve_upload_source(const void* data);

    // --- Zero-copy host-pointer bindings ---

    bool create_binding_from_host_ptr_2d(
        ggml_context* ctx, ggml_backend_t backend, const TensorView2DDesc& desc,
        TensorBinding& out_binding, ggml_backend_buffer_t& out_buffer);

    bool create_binding_from_host_ptr_direct_m2_2d(
        ggml_context* ctx, ggml_backend_t backend, const TensorView2DDesc& desc,
        TensorBinding& out_binding, ggml_backend_buffer_t& out_buffer);

    bool create_binding_from_host_ptr_3d(
        ggml_context* ctx, ggml_backend_t backend, const TensorView3DDesc& desc,
        TensorBinding& out_binding, ggml_backend_buffer_t& out_buffer);

    bool create_binding_from_host_ptr_direct_m2_3d(
        ggml_context* ctx, ggml_backend_t backend, const TensorView3DDesc& desc,
        TensorBinding& out_binding, ggml_backend_buffer_t& out_buffer);

    bool create_binding_from_host_ptr_4d(
        ggml_context* ctx, ggml_backend_t backend, const TensorView4DDesc& desc,
        TensorBinding& out_binding, ggml_backend_buffer_t& out_buffer);

    bool create_binding_from_host_ptr_contiguous(
        ggml_context* ctx, ggml_backend_t backend, const ContiguousTensorDesc& desc,
        TensorBinding& out_binding, ggml_backend_buffer_t& out_buffer);

    // --- Shape queries ---

    inline bool same_shape(const TensorView4DDesc& lhs, const TensorView4DDesc& rhs)
    {
        return lhs.ne0 == rhs.ne0 && lhs.ne1 == rhs.ne1 &&
            lhs.ne2 == rhs.ne2 && lhs.ne3 == rhs.ne3;
    }

    inline bool same_shape_with_last_dim_reduced(const TensorView4DDesc& result, const TensorView4DDesc& src)
    {
        return result.ne0 == 1 && result.ne1 == src.ne1 &&
            result.ne2 == src.ne2 && result.ne3 == src.ne3;
    }

    inline bool can_repeat(const TensorView4DDesc& repeated, const TensorView4DDesc& target)
    {
        return (target.ne0 % repeated.ne0) == 0 &&
            (target.ne1 % repeated.ne1) == 0 &&
            (target.ne2 % repeated.ne2) == 0 &&
            (target.ne3 % repeated.ne3) == 0;
    }

    inline bool is_vector_like(const TensorView4DDesc& desc, std::int64_t width)
    {
        return desc.ne0 == width && desc.ne1 == 1 && desc.ne2 == 1 && desc.ne3 == 1;
    }

    inline std::int64_t flat_row_count(const TensorView4DDesc& desc)
    {
        return static_cast<std::int64_t>(desc.ne1) * desc.ne2 * desc.ne3;
    }

    // --- Small binding helpers ---

    inline TensorBinding create_scalar_binding(ggml_context* ctx)
    {
        ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        return { tensor, tensor, sizeof(float) };
    }

    inline TensorBinding create_matrix_binding(ggml_context* ctx, std::int64_t cols, std::int64_t rows)
    {
        ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        return { tensor, tensor, static_cast<std::size_t>(cols * rows * static_cast<std::int64_t>(sizeof(float))) };
    }

    // --- Tensor reshape helpers ---

    inline ggml_tensor* flatten_to_rows(ggml_context* ctx, ggml_tensor* tensor, std::int64_t cols, std::int64_t rows)
    {
        return ggml_reshape_2d(ctx, tensor, cols, rows);
    }

    ggml_tensor* sum_rows_to_feature_vector(ggml_context* ctx, ggml_tensor* tensor);

    // --- Op-code dispatch helpers ---

    ggml_tensor* make_unary_tensor(ggml_context* ctx, UnaryOpCode op, ggml_tensor* src);
    ggml_tensor* make_fused_act_mul_tensor(ggml_context* ctx, FusedActMulOpCode op, ggml_tensor* a, ggml_tensor* b);
    ggml_tensor* make_binary_tensor(ggml_context* ctx, BinaryTensorOpCode op, ggml_tensor* lhs, ggml_tensor* rhs);
    ggml_tensor* make_norm_tensor(ggml_context* ctx, NormOpCode op, ggml_tensor* src, float eps);
    ggml_tensor* make_reduction_tensor(ggml_context* ctx, ReductionOpCode op, ggml_tensor* src);

    // --- Cross-entropy label buffer ---

    bool build_cross_entropy_label_buffer(
        std::vector<float>& labels,
        const ContiguousTensorDesc& target_indices_desc,
        std::int64_t rows, std::int64_t cols, float label_smooth);

} // namespace tsg

// ---------------------------------------------------------------------------
// qwen4exp (Qwen3.8-Flash-Next) shared executor pieces.
//
// The per-layer descriptor structs, the weight binder and the node builders
// are defined/implemented in ggml_ops_qwen4exp.cpp and shared with the arena
// batched-decode kernel in ggml_ops_qwen4exp_arena.cpp, so the solo span and
// the arena compose their layer math from ONE source of truth.
// ---------------------------------------------------------------------------

// Per-layer weights. Pointers first, then int64, then int32 - the layout the
// C# side mirrors; append within a run rather than reordering.
struct TSGgmlQwen4ExpFfnArgs
{
    // hyper-connection mixer
    void* hc_norm;          // f32 [hc_dim], gamma folded to (1 + w)
    void* hc_down;          // [hc_dim, hc_low_rank]
    void* hc_up;            // [hc_low_rank, hc_dim]
    void* hc_inject;        // [hc_dim, hc]
    // MoE
    void* router;           // [n_embd, n_expert]
    void* gate_exps;        // [n_embd, n_ff, n_expert]
    void* up_exps;          // [n_embd, n_ff, n_expert]
    void* down_exps;        // [n_ff, n_embd, n_expert]
    // shared expert
    void* sh_gate_inp;      // f32 [n_embd] - one sigmoid scalar per token
    void* sh_gate;          // [n_embd, n_ff_sh]
    void* sh_up;            // [n_embd, n_ff_sh]
    void* sh_down;          // [n_ff_sh, n_embd]

    long long hc_down_bytes, hc_up_bytes, hc_inject_bytes;
    long long router_bytes, gate_exps_bytes, up_exps_bytes, down_exps_bytes;
    long long sh_gate_bytes, sh_up_bytes, sh_down_bytes;

    int hc_down_type, hc_up_type, hc_inject_type;
    int router_type, gate_exps_type, up_exps_type, down_exps_type;
    int sh_gate_type, sh_up_type, sh_down_type;
};

// Per-layer weights for the recurrent (Gated DeltaNet) half of a layer.
struct TSGgmlQwen4ExpGdnArgs
{
    // hyper-connection mixer
    void* hc_norm;
    void* hc_down;
    void* hc_up;
    void* hc_inject;
    // delta net
    void* qkv;              // [n_embd, conv_dim]
    void* gate;             // [n_embd, value_dim]
    void* beta;             // [n_embd, n_v_heads]
    void* alpha;            // [n_embd, n_v_heads]
    void* conv1d;           // f32 [d_conv, conv_dim]
    void* ssm_dt;           // f32 [n_v_heads]
    void* ssm_a;            // f32 [n_v_heads], pre-negated
    void* ssm_norm;         // f32 [head_v_dim]
    void* out_proj;         // [value_dim, n_embd]
    // state, updated in place
    void* conv_state;       // f32 [d_conv-1, conv_dim]
    void* ssm_state;        // f32 [head_v_dim, head_v_dim, n_v_heads]

    long long hc_down_bytes, hc_up_bytes, hc_inject_bytes;
    long long qkv_bytes, gate_bytes, beta_bytes, alpha_bytes, out_proj_bytes;

    int hc_down_type, hc_up_type, hc_inject_type;
    int qkv_type, gate_type, beta_type, alpha_type, out_proj_type;
};

// Per-layer weights for the full-attention half of a layer.
struct TSGgmlQwen4ExpAttnArgs
{
    void* hc_norm;
    void* hc_down;
    void* hc_up;
    void* hc_inject;
    void* wq;               // [n_embd, head_dim * n_head * 2]  (query|gate interleaved)
    void* wk;               // [n_embd, head_dim * n_head_kv]
    void* wv;
    void* wo;               // [head_dim * n_head, n_embd]
    void* q_norm;           // f32 [head_dim]
    void* k_norm;           // f32 [head_dim]
    void* k_cache;          // f16/f32 [head_dim, capacity, n_head_kv]
    void* v_cache;

    long long hc_down_bytes, hc_up_bytes, hc_inject_bytes;
    long long wq_bytes, wk_bytes, wv_bytes, wo_bytes;
    long long kv_bytes;     // per cache

    int hc_down_type, hc_up_type, hc_inject_type;
    int wq_type, wk_type, wv_type, wo_type;
    int kv_type;
};

// The PLE block riding inside the span. Only the n-gram hash and the gather
// from the ~320M-row host table stay on the CPU; the gathered rows arrive as a
// graph input and the projections, norms, gating, dilated depthwise conv and
// the residual add all run on the device. The conv history is persistent
// device state, written in place like the GDN state.
struct TSGgmlQwen4ExpPleArgs
{
    void* key_w;            // [n_embd, hc_dim]
    void* value_w;          // [n_embd, n_embd]
    void* norm_key;         // f32 [hc_dim]
    void* norm_query;       // f32 [hc_dim]
    void* norm_conv;        // f32 [hc_dim]
    void* conv1d_t;         // f32 [hc_dim, kern] - tap-major transpose of ple_conv1d
    void* conv_state;       // f32 [hc_dim, hist] seed (host layout matches)

    long long key_bytes, value_bytes;

    int key_type, value_type;
    int kern;               // conv kernel taps
    int dil;                // dilation (the n-gram size)
};

// The output stage: the final hyper-connection mixer (which IS the output norm -
// qwen4exp ships no separate one) and the LM head, riding the tail of the last
// span. The mixer runs on the LAST token only - at prefill the managed path used
// to mix every position and throw all but one away.
struct TSGgmlQwen4ExpHeadArgs
{
    void* hc_norm;          // f32 [hc_dim]
    void* hc_down;          // [hc_dim, hc_low_rank]
    void* hc_up;            // [hc_low_rank, hc_dim]
    void* head;             // [n_embd, vocab]

    long long hc_down_bytes, hc_up_bytes, head_bytes;

    int hc_down_type, hc_up_type, head_type;
    int vocab;
};

struct TSGgmlQwen4ExpMtpConfig
{
    void* enorm;
    void* hnorm;
    void* eh_proj;
    long long eh_bytes;
    int eh_type;
    int n_embd, hc, hc_low_rank;
    int head_dim, n_head, n_head_kv, n_rot;
    int n_expert, n_expert_used, n_ff, n_ff_sh;
    int capacity, device;
    float eps, rope_base, rope_scale, attn_scale;
    int rope_sections[4];
};

// Additive QSA descriptor. Existing attention/span descriptor ABIs stay intact.
struct TSGgmlQwen4ExpQsaArgs
{
    void* k_proj;
    void* q_proj;
    void* k_norm;
    void* q_norm;
    void* cache;
    long long k_bytes, q_bytes, cache_bytes;
    int k_type, q_type, cache_type;
    int head_dim, heads, ratio, top_k;
    int rope_sections[4];
};

struct Q4eQsaInputs
{
    int ratio = 0;
    ggml_tensor* cell_blocks = nullptr;
    ggml_tensor* block_cells = nullptr;
    ggml_tensor* block_positions = nullptr;
    ggml_tensor* query_positions = nullptr;
    ggml_tensor* bias = nullptr;
};

struct Q4eQsaGraph
{
    const TSGgmlQwen4ExpQsaArgs* args = nullptr;
    const Q4eQsaInputs* inputs = nullptr;
    ggml_tensor* cache = nullptr;
    // Optional test-only outputs, never retained by production graphs.
    std::vector<ggml_tensor*>* probes = nullptr;
};

// A weight binding resolved through the resident cache, remembered so a
// REPLAY can re-resolve it. The cache can move or re-create a device copy
// (large allocations elsewhere churn it), and a persisted graph would keep
// reading the old address forever - the graph-uid stamp even tells ggml-cuda
// to skip the staleness walk that might have noticed.
struct Q4eCachedBind
{
    ggml_tensor* tensor;
    void* data;
    std::size_t bytes;
    ggml_backend_buffer_usage usage;
};

// One place for the weight-binding policy the block builders share. Collecting
// the uploads rather than binding immediately is what lets several layers
// build into one graph: everything is bound before a single allocation pass
// runs over the lot. Methods are defined in ggml_ops_qwen4exp.cpp.
struct Q4eHostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };

struct Q4eBinder
{
    ggml_backend_dev_t dev = nullptr;
    std::vector<Q4eHostBinding> upload_list;
    std::vector<Q4eCachedBind> cached;

    void add(ggml_tensor* tgt, void* data, std::size_t bytes,
             ggml_backend_buffer_usage usage = GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    void flush();
};

// GDN write-back sources: the caller decides how they reach the persistent
// state (copied out after the compute, or cpy-expanded in graph after the
// reads).
struct Q4eGdnWriteback { ggml_tensor* tail; ggml_tensor* new_state; };

// Arena override for q4e_nodes_attn: the KV lives in a slot-stable 2D arena
// [head_dim, (n_slots+1) * n_head_kv * cap] instead of the holder's resident
// cache copy, written by ONE absolute-row ggml_set_rows per K/V and read back
// as a 4D view of the set_rows RESULT (a real src edge, so node order is not
// load-bearing for the write->read hazard). The mask passed alongside must be
// the 4D per-slot mask [cap, 1, 1, n_slots] and `pos` the per-slot ROPE
// positions.
struct Q4eAttnArenaIO
{
    ggml_tensor* k_arena = nullptr;     // [head_dim, (n_slots+1)*kvH*cap], kv_type
    ggml_tensor* v_arena = nullptr;
    ggml_tensor* kv_idx_abs = nullptr;  // I64 [kvH * n_slots], absolute arena rows
    std::int64_t cap = 0;               // rows per head plane
    std::int64_t rows_per_slot = 0;     // kvH * cap
    // out: the set_rows nodes, for the caller's OUTPUT-flag + expand pass.
    ggml_tensor* k_set = nullptr;
    ggml_tensor* v_set = nullptr;
    ggml_tensor* fa = nullptr;          // out: the flash node, for support probing
};

// Per-sequence recurrent-state entry: GDN conv+ssm state (packed conv @0,
// ssm @ (conv_bytes + 255) & ~255) or the PLE conv history, in a device buffer
// keyed by the HOST SEED POINTER the descriptors carry. `ready` gates the
// one-time seed upload; after the first forward the device copy is
// authoritative and the host seed permanently stale.
struct Q4eSeqStateEntry
{
    ggml_backend_buffer_t buf = nullptr;
    std::size_t bytes = 0;
    bool ready = false;
};

// Create-or-grow the active rank's entry for `key` (the span/per-layer path).
Q4eSeqStateEntry* q4e_seq_state(const void* key, std::size_t bytes);
// Lookup WITHOUT creating (the arena join declines on a missing entry rather
// than fabricating state for a sequence that never ran through the span).
Q4eSeqStateEntry* q4e_seq_state_find(const void* key);
// Clamp a caller-supplied device index to an initialized rank (-1 = active).
int q4e_resolve_device(int device);
// ggml-cuda flash-attention eligibility for this family's KV type/head size.
bool q4e_flash_attn_ok(int kv_type, int head_dim);
// Drop (graph only - state kept) every cached per-layer/span graph built from
// the given holder descriptor arrays, on every device. The arena flush calls
// this after invalidating a holder's resident KV copies: captured solo graphs
// bake those buffers, and q4e_refresh_bindings only SAMPLES every 32nd replay,
// which is no protection against a freed pointer.
void q4e_drop_holder_graphs(const void* attn_base, const void* gdn_base, const void* ple_base);

// ---------------------------------------------------------------------------
// Node builders (defined in ggml_ops_qwen4exp.cpp). Each appends one
// half-layer to ctx and returns the new residual; the per-layer entry points,
// the token span and the arena share them, so the graph a half builds has a
// single source of truth.
// ---------------------------------------------------------------------------
ggml_tensor* q4e_nodes_ffn(
    ggml_context* ctx, Q4eBinder& bnd,
    const TSGgmlQwen4ExpFfnArgs* a, ggml_tensor* res_in,
    int n_embd, int hc, int hc_low_rank, int T,
    int n_expert, int n_expert_used, int n_ff, int n_ff_sh, float eps);

ggml_tensor* q4e_nodes_gdn(
    ggml_context* ctx, Q4eBinder& bnd,
    const TSGgmlQwen4ExpGdnArgs* a, ggml_tensor* res_in,
    ggml_tensor* conv_state, ggml_tensor* ssm_state,
    int n_embd, int hc, int hc_low_rank, int T,
    int head_k_dim, int head_v_dim, int n_k_heads, int n_v_heads, int d_conv,
    float eps, Q4eGdnWriteback* wb,
    std::vector<ggml_tensor*>* probe = nullptr);

ggml_tensor* q4e_nodes_attn(
    ggml_context* ctx, ggml_cgraph* graph, Q4eBinder& bnd,
    const TSGgmlQwen4ExpAttnArgs* a, ggml_tensor* res_in,
    ggml_tensor* mask, ggml_tensor* pos, ggml_tensor* kv_idx,
    int n_embd, int hc, int hc_low_rank, int T,
    int head_dim, int n_head, int n_head_kv, int kv_capacity, int n_kv_pad,
    int n_rot, float rope_base, float rope_freq_scale, float attn_scale,
    float eps, bool use_flash,
    std::vector<ggml_tensor*>* kv_out = nullptr,
    std::vector<ggml_tensor*>* probe = nullptr,
    const std::int32_t* mrope_sections = nullptr,
    Q4eAttnArenaIO* arena = nullptr,
    ggml_tensor* owned_k = nullptr, ggml_tensor* owned_v = nullptr,
    const Q4eQsaGraph* qsa = nullptr);

// PLE half. `conv_hist` is the persistent history - [hc_dim, hist] for the
// span (n_streams == 1), a [hc_dim, hist, n_streams] arena view for the arena
// (T == 1 per stream). When `writes` is null the write-back cpy is expanded
// into `graph` right after the residual (span semantics); otherwise it is
// appended to *writes for the caller's OUTPUT-flag + expand pass.
ggml_tensor* q4e_nodes_ple(
    ggml_context* ctx, ggml_cgraph* graph, Q4eBinder& bnd,
    const TSGgmlQwen4ExpPleArgs* a, ggml_tensor* res_in, ggml_tensor* ple_emb_in,
    ggml_tensor* conv_hist,
    int n_embd, int hc, int T, int n_streams, float eps,
    std::vector<ggml_tensor*>* writes = nullptr);

// Head: final hyper-connection mixer + LM head over `res_last` [hc_dim, T]
// (must be contiguous - the span passes a cont of the last column, the arena
// its whole [hc_dim, n_slots] residual since every slot's token is "last").
// Returns logits [vocab, T]; the caller flags/expands it.
ggml_tensor* q4e_nodes_head(
    ggml_context* ctx, Q4eBinder& bnd,
    const TSGgmlQwen4ExpHeadArgs* a, ggml_tensor* res_last,
    int n_embd, int hc, int hc_low_rank, int T, float eps);
