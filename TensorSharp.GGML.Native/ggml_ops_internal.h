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
#include <cstdint>
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
namespace tsg
{
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
    extern std::once_flag g_backend_init_once;
    extern ggml_backend_t g_backend;
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
    };

    extern std::mutex g_host_buffer_cache_mutex;
    extern std::unordered_map<void*, CachedHostBuffer> g_host_buffer_cache;
    extern std::mutex g_preloaded_buffer_cache_mutex;
    extern std::unordered_map<void*, CachedHostBuffer> g_preloaded_buffer_cache;

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
    extern std::unordered_set<void*> g_offloadable_keys;
    extern std::list<void*> g_offloadable_lru;
    extern std::unordered_map<void*, std::list<void*>::iterator> g_offloadable_lru_map;
    extern std::int64_t g_offloadable_resident_bytes;
    extern std::int64_t g_offloadable_budget;

    // --- Backend constants ---

    constexpr int BACKEND_TYPE_METAL = 1;
    constexpr int BACKEND_TYPE_CPU = 2;
    constexpr int BACKEND_TYPE_CUDA = 3;

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
        ggml_backend_synchronize(g_backend);
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
        ggml_backend_synchronize(g_backend);
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
        ggml_backend_synchronize(g_backend);
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
            ggml_backend_synchronize(g_backend);
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

    // --- Backend management ---

    ggml_backend_t create_backend_instance(int backend_type);
    void initialize_backend();
    bool ensure_backend(int backend_type);
    bool ensure_backend();
    bool can_initialize_backend(int backend_type);
    bool backend_supports_op(ggml_tensor* op);

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
    bool alloc_ctx_tensors_reuse(ggml_context* ctx);
    // Free the cached reuse buffer (called from TSGgml_Shutdown).
    void free_reuse_compute_buffer();

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
    void invalidate_cached_buffer(void* data);

    // allow_unified_weight: when true the prefers_device_local_cache gate is
    // bypassed and only raw capability (host_ptr_buffer_capable) is required.
    // Set by the read-only-weight path on unified-memory Metal so weights are
    // wrapped zero-copy instead of duplicated into a device-local copy.
    bool try_get_host_ptr_buffer(
        ggml_backend_t backend, ggml_backend_dev_t dev,
        void* data, std::size_t bytes, bool cacheable,
        ggml_backend_buffer_t& out_buffer,
        bool allow_unified_weight = false);

    bool try_get_cacheable_tensor_buffer(
        ggml_backend_t backend, ggml_backend_dev_t dev,
        ggml_tensor* tensor, void* data, std::size_t bytes,
        ggml_backend_buffer_t& out_buffer, void*& out_addr, bool& out_needs_upload,
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
