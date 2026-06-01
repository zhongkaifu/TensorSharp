// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
#include "ggml_ops_internal.h"

#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <cstdio>

// ============================================================================
// ggml_pool implementation
// ============================================================================
namespace ggml_pool
{
    static std::mutex g_pool_mutex;
    static std::vector<PoolEntry> g_pool;

    static void* pool_alloc(std::size_t size)
    {
        if (size == 0 || size > k_pool_buffer_size)
            return nullptr;
        void* ptr = std::malloc(size);
        return ptr;
    }

    static void pool_free(void* ptr)
    {
        if (ptr != nullptr)
            std::free(ptr);
    }

    PoolEntry acquire(std::size_t required_size)
    {
        if (required_size == 0 || required_size > k_pool_buffer_size)
            return {};
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        for (auto it = g_pool.begin(); it != g_pool.end(); ++it)
        {
            if (it->size >= required_size)
            {
                PoolEntry e = *it;
                g_pool.erase(it);
                return e;
            }
        }
        void* ptr = pool_alloc(k_pool_buffer_size);
        if (ptr == nullptr)
            return {};
        return { ptr, k_pool_buffer_size };
    }

    void release(PoolEntry e)
    {
        if (e.ptr == nullptr)
            return;
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        if (static_cast<int>(g_pool.size()) < k_pool_max_count)
        {
            g_pool.push_back(e);
        }
        else
        {
            pool_free(e.ptr);
        }
    }

    void ensure_initial_pool()
    {
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        for (int i = static_cast<int>(g_pool.size()); i < k_pool_initial_count; ++i)
        {
            void* ptr = pool_alloc(k_pool_buffer_size);
            if (ptr == nullptr)
                break;
            g_pool.push_back({ ptr, k_pool_buffer_size });
        }
    }
}

// ============================================================================
// tsg namespace: global state definitions and helper implementations
// ============================================================================
namespace tsg
{
    // --- Global state definitions ---

    thread_local std::string g_last_error;
    std::once_flag g_backend_init_once;
    ggml_backend_t g_backend = nullptr;
    int g_backend_type = 0;

    std::mutex g_host_buffer_cache_mutex;
    std::unordered_map<void*, CachedHostBuffer> g_host_buffer_cache;
    std::mutex g_preloaded_buffer_cache_mutex;
    std::unordered_map<void*, CachedHostBuffer> g_preloaded_buffer_cache;

    // MoE expert weight offload state — see ggml_ops_internal.h for the contract.
    std::unordered_set<void*> g_offloadable_keys;
    std::list<void*> g_offloadable_lru;
    std::unordered_map<void*, std::list<void*>::iterator> g_offloadable_lru_map;
    std::int64_t g_offloadable_resident_bytes = 0;
    std::int64_t g_offloadable_budget = 0;

    // Async dispatch state. The defaults keep the legacy (eager-sync) behaviour;
    // C# enables async at backend init time via TSGgml_SetAsyncCompute(1).
    std::atomic<bool> g_async_compute_enabled{false};
    std::atomic<bool> g_pending_gpu_work{false};

    static bool is_truthy_env(const char* value)
    {
        return value != nullptr &&
            (std::strcmp(value, "1") == 0 ||
             std::strcmp(value, "true") == 0 ||
             std::strcmp(value, "TRUE") == 0 ||
             std::strcmp(value, "True") == 0 ||
             std::strcmp(value, "yes") == 0 ||
             std::strcmp(value, "YES") == 0 ||
             std::strcmp(value, "on") == 0 ||
             std::strcmp(value, "ON") == 0);
    }

    static bool ggml_debug_logging_enabled()
    {
        return is_truthy_env(std::getenv("TENSORSHARP_GGML_DEBUG"));
    }

    static void filtered_ggml_log(enum ggml_log_level level, const char* text, void* user_data)
    {
        (void) user_data;
        if (level == GGML_LOG_LEVEL_DEBUG && !ggml_debug_logging_enabled())
            return;
        std::fputs(text, stderr);
        std::fflush(stderr);
    }

    static void configure_ggml_logging()
    {
        ggml_log_set(filtered_ggml_log, nullptr);
    }

    // --- Error helpers ---

    void set_last_error(const std::string& message)
    {
        g_last_error = message;
    }

    void clear_last_error()
    {
        g_last_error.clear();
    }

    // --- Backend management ---

    ggml_backend_t create_backend_instance(int backend_type)
    {
        if (backend_type == BACKEND_TYPE_METAL)
        {
#if defined(TSG_GGML_USE_METAL)
            ggml_backend_t backend = ggml_backend_metal_init();
            if (backend == nullptr)
                set_last_error("ggml-metal backend initialization failed.");
            return backend;
#else
            set_last_error("The ggml-metal backend is not available in this build.");
            return nullptr;
#endif
        }

        if (backend_type == BACKEND_TYPE_CPU)
        {
            ggml_backend_t backend = ggml_backend_cpu_init();
            if (backend == nullptr)
                set_last_error("ggml-cpu backend initialization failed.");
            return backend;
        }

        if (backend_type == BACKEND_TYPE_CUDA)
        {
#if defined(GGML_USE_CUDA)
            ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
            if (device == nullptr)
            {
                set_last_error("No GGML GPU device is available for ggml-cuda.");
                return nullptr;
            }

            ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
            if (backend == nullptr)
                set_last_error("ggml-cuda backend initialization failed.");
            return backend;
#else
            set_last_error("The ggml-cuda backend is not available in this build.");
            return nullptr;
#endif
        }

        set_last_error("Unknown GGML backend type requested.");
        return nullptr;
    }

    void initialize_backend()
    {
        clear_last_error();
        configure_ggml_logging();
        g_backend = create_backend_instance(g_backend_type);
        if (g_backend == nullptr)
            return;
        ggml_pool::ensure_initial_pool();
    }

    bool ensure_backend(int backend_type)
    {
        if (backend_type != BACKEND_TYPE_METAL &&
            backend_type != BACKEND_TYPE_CPU &&
            backend_type != BACKEND_TYPE_CUDA)
        {
            set_last_error("Invalid GGML backend type.");
            return false;
        }

        if (g_backend_type == 0)
            g_backend_type = backend_type;
        else if (g_backend_type != backend_type)
        {
            set_last_error("A different GGML backend was already initialized in this process.");
            return false;
        }

        std::call_once(g_backend_init_once, initialize_backend);
        return g_backend != nullptr;
    }

    bool ensure_backend()
    {
        const int backend_type = (g_backend_type == 0) ? BACKEND_TYPE_METAL : g_backend_type;
        return ensure_backend(backend_type);
    }

    bool can_initialize_backend(int backend_type)
    {
        // Lightweight availability check: report only compile-time support so we
        // don't spin up the actual GGML device (Metal MTLDevice / CUDA driver) at
        // process start — important when a non-GGML backend (MLX, direct CUDA) is
        // selected, otherwise the unrelated GGML init logs leak into that run.
        // Real init still happens lazily via ensure_backend when a GGML backend
        // is actually selected, and surfaces a clear error then if it fails.
        clear_last_error();
        if (backend_type == BACKEND_TYPE_CPU)
            return true;

        if (backend_type == BACKEND_TYPE_METAL)
        {
#if defined(TSG_GGML_USE_METAL)
            return true;
#else
            set_last_error("The ggml-metal backend is not available in this build.");
            return false;
#endif
        }

        if (backend_type == BACKEND_TYPE_CUDA)
        {
#if defined(GGML_USE_CUDA)
            return true;
#else
            set_last_error("The ggml-cuda backend is not available in this build.");
            return false;
#endif
        }

        set_last_error("Invalid GGML backend type.");
        return false;
    }

    bool backend_supports_op(ggml_tensor* op)
    {
        return op != nullptr && g_backend != nullptr && ggml_backend_supports_op(g_backend, op);
    }

    // --- Size / layout queries ---

    std::size_t required_raw_bytes(const TensorView2DDesc& desc)
    {
        const std::int64_t max_offset =
            (static_cast<std::int64_t>(desc.dim0) - 1) * desc.stride0 +
            (static_cast<std::int64_t>(desc.dim1) - 1) * desc.stride1;
        return static_cast<std::size_t>((max_offset + 1) * sizeof(float));
    }

    std::size_t required_raw_bytes(const TensorView3DDesc& desc)
    {
        const std::int64_t max_offset =
            (static_cast<std::int64_t>(desc.dim0) - 1) * desc.stride0 +
            (static_cast<std::int64_t>(desc.dim1) - 1) * desc.stride1 +
            (static_cast<std::int64_t>(desc.dim2) - 1) * desc.stride2;
        return static_cast<std::size_t>((max_offset + 1) * sizeof(float));
    }

    std::size_t required_raw_bytes(const TensorView4DDesc& desc)
    {
        const std::int64_t max_offset =
            (static_cast<std::int64_t>(desc.ne0) - 1) +
            (static_cast<std::int64_t>(desc.ne1) - 1) * (desc.nb1 / static_cast<std::int64_t>(sizeof(float))) +
            (static_cast<std::int64_t>(desc.ne2) - 1) * (desc.nb2 / static_cast<std::int64_t>(sizeof(float))) +
            (static_cast<std::int64_t>(desc.ne3) - 1) * (desc.nb3 / static_cast<std::int64_t>(sizeof(float)));
        return static_cast<std::size_t>((max_offset + 1) * sizeof(float));
    }

    std::size_t logical_bytes(const TensorView2DDesc& desc)
    {
        return static_cast<std::size_t>(desc.dim0) * desc.dim1 * sizeof(float);
    }

    std::size_t logical_row_bytes(const TensorView2DDesc& desc)
    {
        return static_cast<std::size_t>(desc.dim1) * sizeof(float);
    }

    std::size_t logical_bytes(const TensorView3DDesc& desc)
    {
        return static_cast<std::size_t>(desc.dim0) * desc.dim1 * desc.dim2 * sizeof(float);
    }

    std::size_t logical_bytes(const TensorView4DDesc& desc)
    {
        return static_cast<std::size_t>(desc.ne0) * desc.ne1 * desc.ne2 * desc.ne3 * sizeof(float);
    }

    std::size_t raw_row_bytes(const TensorView2DDesc& desc)
    {
        TensorView2DDesc row_desc = desc;
        row_desc.dim0 = 1;
        return required_raw_bytes(row_desc);
    }

    TensorView2DDesc slice_rows_2d(const TensorView2DDesc& desc, int row_start, int row_count)
    {
        TensorView2DDesc slice = desc;
        slice.data = static_cast<char*>(desc.data) +
            static_cast<std::size_t>(row_start) *
            static_cast<std::size_t>(desc.stride0) *
            sizeof(float);
        slice.dim0 = row_count;
        slice.raw_bytes = static_cast<std::int64_t>(required_raw_bytes(slice));
        return slice;
    }

    int limit_rows_for_cuda_copy(int current_limit, const TensorView2DDesc& desc)
    {
        if (current_limit <= 0)
            return 0;
        const std::size_t per_row_bytes = std::max(logical_row_bytes(desc), raw_row_bytes(desc));
        if (per_row_bytes == 0 || per_row_bytes > k_ggml_cuda_max_copy_bytes)
            return 0;
        const int limit = static_cast<int>(k_ggml_cuda_max_copy_bytes / per_row_bytes);
        return std::min(current_limit, std::max(1, limit));
    }

    // --- Validation ---

    bool validate_desc(const TensorView2DDesc& desc, const char* name)
    {
        if (desc.data == nullptr)
        {
            set_last_error(std::string("Null pointer passed for ") + name + '.');
            return false;
        }
        if (desc.dim0 <= 0 || desc.dim1 <= 0)
        {
            set_last_error(std::string("Invalid tensor shape passed for ") + name + '.');
            return false;
        }
        if (desc.stride0 < 0 || desc.stride1 < 0)
        {
            set_last_error(std::string("Negative tensor strides are not supported for ") + name + '.');
            return false;
        }
        if (desc.raw_bytes <= 0 || (desc.raw_bytes % static_cast<std::int64_t>(sizeof(float))) != 0)
        {
            set_last_error(std::string("Invalid raw byte size passed for ") + name + '.');
            return false;
        }
        if (static_cast<std::size_t>(desc.raw_bytes) < required_raw_bytes(desc))
        {
            set_last_error(std::string("Raw byte span is too small for ") + name + '.');
            return false;
        }
        return true;
    }

    bool validate_desc(const TensorView3DDesc& desc, const char* name)
    {
        if (desc.data == nullptr)
        {
            set_last_error(std::string("Null pointer passed for ") + name + '.');
            return false;
        }
        if (desc.dim0 <= 0 || desc.dim1 <= 0 || desc.dim2 <= 0)
        {
            set_last_error(std::string("Invalid tensor shape passed for ") + name + '.');
            return false;
        }
        if (desc.stride0 < 0 || desc.stride1 < 0 || desc.stride2 < 0)
        {
            set_last_error(std::string("Negative tensor strides are not supported for ") + name + '.');
            return false;
        }
        if (desc.raw_bytes <= 0 || (desc.raw_bytes % static_cast<std::int64_t>(sizeof(float))) != 0)
        {
            set_last_error(std::string("Invalid raw byte size passed for ") + name + '.');
            return false;
        }
        if (static_cast<std::size_t>(desc.raw_bytes) < required_raw_bytes(desc))
        {
            set_last_error(std::string("Raw byte span is too small for ") + name + '.');
            return false;
        }
        return true;
    }

    bool validate_desc(const TensorView4DDesc& desc, const char* name)
    {
        if (desc.data == nullptr)
        {
            set_last_error(std::string("Null pointer passed for ") + name + '.');
            return false;
        }
        if (desc.ne0 <= 0 || desc.ne1 <= 0 || desc.ne2 <= 0 || desc.ne3 <= 0)
        {
            set_last_error(std::string("Invalid tensor shape passed for ") + name + '.');
            return false;
        }
        if (desc.nb1 <= 0 || desc.nb2 <= 0 || desc.nb3 <= 0)
        {
            set_last_error(std::string("Invalid tensor strides passed for ") + name + '.');
            return false;
        }
        if ((desc.nb1 % static_cast<std::int64_t>(sizeof(float))) != 0
            || (desc.nb2 % static_cast<std::int64_t>(sizeof(float))) != 0
            || (desc.nb3 % static_cast<std::int64_t>(sizeof(float))) != 0)
        {
            set_last_error(std::string("Tensor byte strides must be multiples of sizeof(float) for ") + name + '.');
            return false;
        }
        if (desc.raw_bytes <= 0 || (desc.raw_bytes % static_cast<std::int64_t>(sizeof(float))) != 0)
        {
            set_last_error(std::string("Invalid raw byte size passed for ") + name + '.');
            return false;
        }
        if (static_cast<std::size_t>(desc.raw_bytes) < required_raw_bytes(desc))
        {
            set_last_error(std::string("Raw byte span is too small for ") + name + '.');
            return false;
        }
        return true;
    }

    bool validate_desc(const ContiguousTensorDesc& desc, const char* name)
    {
        if (desc.data == nullptr)
        {
            set_last_error(std::string("Null pointer passed for ") + name + '.');
            return false;
        }
        if (desc.element_count <= 0)
        {
            set_last_error(std::string("Invalid element count passed for ") + name + '.');
            return false;
        }
        if (desc.element_type != TSG_DTYPE_F32 && desc.element_type != TSG_DTYPE_I32)
        {
            set_last_error(std::string("Unsupported contiguous tensor element type passed for ") + name + '.');
            return false;
        }
        return true;
    }

    bool read_i32_values(std::vector<std::int32_t>& output, const ContiguousTensorDesc& desc, const char* name)
    {
        output.resize(static_cast<std::size_t>(desc.element_count));
        if (desc.element_type == TSG_DTYPE_I32)
        {
            const std::int32_t* raw = static_cast<const std::int32_t*>(desc.data);
            std::copy(raw, raw + output.size(), output.begin());
            return true;
        }
        if (desc.element_type == TSG_DTYPE_F32)
        {
            const float* raw = static_cast<const float*>(desc.data);
            for (std::size_t i = 0; i < output.size(); ++i)
                output[i] = static_cast<std::int32_t>(raw[i]);
            return true;
        }
        set_last_error(std::string("Unsupported element type for ") + name + '.');
        return false;
    }

    // --- Layout queries ---

    bool can_map_standard_view(const TensorView2DDesc& desc)
    {
        return desc.stride1 == 1 &&
            is_non_overlapping_fast_to_slow<2>({ desc.dim1, desc.dim0 }, { desc.stride1, desc.stride0 });
    }

    bool can_map_standard_view(const TensorView3DDesc& desc)
    {
        return desc.stride2 == 1 &&
            is_non_overlapping_fast_to_slow<3>({ desc.dim2, desc.dim1, desc.dim0 }, { desc.stride2, desc.stride1, desc.stride0 });
    }

    bool can_map_standard_view(const TensorView4DDesc& desc)
    {
        const auto stride1 = static_cast<int>(desc.nb1 / static_cast<std::int64_t>(sizeof(float)));
        const auto stride2 = static_cast<int>(desc.nb2 / static_cast<std::int64_t>(sizeof(float)));
        const auto stride3 = static_cast<int>(desc.nb3 / static_cast<std::int64_t>(sizeof(float)));
        return is_non_overlapping_fast_to_slow<4>({ desc.ne0, desc.ne1, desc.ne2, desc.ne3 }, { 1, stride1, stride2, stride3 });
    }

    bool can_map_m2_direct(const TensorView2DDesc& desc)
    {
        return desc.stride0 == 1 &&
            desc.stride1 >= desc.dim0 &&
            is_non_overlapping_fast_to_slow<2>({ desc.dim0, desc.dim1 }, { desc.stride0, desc.stride1 });
    }

    bool can_map_m2_direct(const TensorView3DDesc& desc)
    {
        return desc.stride1 == 1 &&
            desc.stride2 >= desc.dim1 &&
            is_non_overlapping_fast_to_slow<3>({ desc.dim1, desc.dim2, desc.dim0 }, { desc.stride1, desc.stride2, desc.stride0 });
    }

    // --- Pointer / buffer utilities ---

    bool is_pointer_aligned(const void* ptr, std::size_t alignment)
    {
        return ptr != nullptr && (alignment <= 1 || (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0);
    }

    std::size_t get_host_ptr_alignment(ggml_backend_t backend, ggml_backend_dev_t dev)
    {
        if (dev != nullptr)
        {
            if (ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev))
                return ggml_backend_buft_get_alignment(buft);
        }
        return 16384;
    }

    bool prefers_device_local_cache(ggml_backend_dev_t dev)
    {
        if (dev == nullptr)
            return false;
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        // Upstream ggml's ggml_backend_dev_props has no `integrated` field (that was an
        // ollama-fork extension). On the backends we use the field was effectively always
        // 0 anyway -- the Metal backend reports type=GPU and never set it -- so the
        // discrete-GPU test reduces to "is this a GPU device".
        return props.type == GGML_BACKEND_DEVICE_TYPE_GPU;
    }

    bool can_use_host_ptr_buffer(ggml_backend_t backend, ggml_backend_dev_t dev, const void* ptr, std::size_t size)
    {
        if (dev == nullptr || ptr == nullptr || size == 0)
            return false;
        if (prefers_device_local_cache(dev))
            return false;
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.buffer_from_host_ptr)
            return false;
        const std::size_t alignment = get_host_ptr_alignment(backend, dev);
        return is_pointer_aligned(ptr, alignment);
    }

    // Hint to the OS that the given file-backed mmap region is no longer
    // needed. Pairs with offloadable LRU eviction: once Metal's MTLBuffer
    // wrapper has been freed, calling MADV_DONTNEED tells the kernel it
    // may immediately reclaim those pages without waiting for memory
    // pressure. On the next access the pages page-fault back in from SSD.
    // The range is rounded outward to whole page boundaries; for our use
    // case (GGUF tensors aligned on 32-byte block boundaries in a file
    // mmap'd read-only) the rounding may overlap adjacent tensors, which
    // is fine — they're also file-backed and will page back in on next
    // touch. Safe on Apple Silicon (16 KB pages) and Linux.
    void advise_pages_dont_need(void* data, std::size_t bytes)
    {
#if defined(__APPLE__) || defined(__linux__)
        if (data == nullptr || bytes == 0)
            return;
        const long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0)
            return;
        const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(data);
        const std::uintptr_t aligned_addr = addr & ~(static_cast<std::uintptr_t>(page_size) - 1);
        const std::size_t prefix = static_cast<std::size_t>(addr - aligned_addr);
        const std::size_t total = bytes + prefix;
        const std::size_t mask = static_cast<std::size_t>(page_size) - 1;
        const std::size_t rounded = (total + mask) & ~mask;
        (void)madvise(reinterpret_cast<void*>(aligned_addr), rounded, MADV_DONTNEED);
#else
        (void)data;
        (void)bytes;
#endif
    }

    // --- Offloadable LRU helpers (caller holds g_host_buffer_cache_mutex) ---

    void offloadable_lru_remove_locked(void* key)
    {
        auto it = g_offloadable_lru_map.find(key);
        if (it == g_offloadable_lru_map.end())
            return;
        g_offloadable_lru.erase(it->second);
        g_offloadable_lru_map.erase(it);
    }

    void offloadable_lru_touch_locked(void* key)
    {
        auto it = g_offloadable_lru_map.find(key);
        if (it == g_offloadable_lru_map.end())
            return;
        g_offloadable_lru.erase(it->second);
        g_offloadable_lru.push_front(key);
        it->second = g_offloadable_lru.begin();
    }

    void offloadable_lru_insert_front_locked(void* key)
    {
        offloadable_lru_remove_locked(key);
        g_offloadable_lru.push_front(key);
        g_offloadable_lru_map[key] = g_offloadable_lru.begin();
    }

    // Drop an offloadable LRU entry: removes the cache entry, frees the
    // backend buffer wrapper (releasing Metal's claim on the underlying
    // host pages), and hints the OS that the pages can be reclaimed now.
    // Returns the number of bytes freed.
    std::size_t offloadable_evict_one_locked()
    {
        if (g_offloadable_lru.empty())
            return 0;
        void* key = g_offloadable_lru.back();
        g_offloadable_lru.pop_back();
        g_offloadable_lru_map.erase(key);

        auto cit = g_host_buffer_cache.find(key);
        if (cit == g_host_buffer_cache.end())
            return 0;
        std::size_t freed = cit->second.bytes;
        ggml_backend_buffer_free(cit->second.buffer);
        g_host_buffer_cache.erase(cit);
        advise_pages_dont_need(key, freed);
        if (g_offloadable_resident_bytes >= static_cast<std::int64_t>(freed))
            g_offloadable_resident_bytes -= static_cast<std::int64_t>(freed);
        else
            g_offloadable_resident_bytes = 0;
        return freed;
    }

    void offloadable_evict_to_budget_locked()
    {
        if (g_offloadable_budget <= 0)
            return;
        while (g_offloadable_resident_bytes > g_offloadable_budget && !g_offloadable_lru.empty())
        {
            if (offloadable_evict_one_locked() == 0)
                break;
        }
    }

    void invalidate_cached_buffer(void* data)
    {
        if (data == nullptr)
            return;

        {
            std::lock_guard<std::mutex> lock(g_preloaded_buffer_cache_mutex);
            auto it = g_preloaded_buffer_cache.find(data);
            if (it != g_preloaded_buffer_cache.end())
            {
                ggml_backend_buffer_free(it->second.buffer);
                g_preloaded_buffer_cache.erase(it);
                return;
            }
        }

        {
            std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
            auto it = g_host_buffer_cache.find(data);
            if (it == g_host_buffer_cache.end())
                return;
            offloadable_lru_remove_locked(data);
            if (g_offloadable_keys.count(data))
            {
                if (g_offloadable_resident_bytes >= static_cast<std::int64_t>(it->second.bytes))
                    g_offloadable_resident_bytes -= static_cast<std::int64_t>(it->second.bytes);
                else
                    g_offloadable_resident_bytes = 0;
            }
            ggml_backend_buffer_free(it->second.buffer);
            g_host_buffer_cache.erase(it);
        }
    }

    bool try_get_host_ptr_buffer(
        ggml_backend_t backend, ggml_backend_dev_t dev,
        void* data, std::size_t bytes, bool cacheable,
        ggml_backend_buffer_t& out_buffer)
    {
        out_buffer = nullptr;
        if (!can_use_host_ptr_buffer(backend, dev, data, bytes))
            return false;

        if (cacheable)
        {
            std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
            auto it = g_host_buffer_cache.find(data);
            if (it != g_host_buffer_cache.end() &&
                it->second.bytes == bytes &&
                it->second.mode == CachedBufferMode::HostPtr)
            {
                out_buffer = it->second.buffer;
                if (g_offloadable_keys.count(data))
                    offloadable_lru_touch_locked(data);
                return true;
            }
        }

        out_buffer = ggml_backend_dev_buffer_from_host_ptr(dev, data, bytes, bytes);
        if (out_buffer == nullptr)
            return false;

        if (cacheable)
        {
            std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
            g_host_buffer_cache[data] = {
                out_buffer, bytes,
                ggml_backend_buffer_get_size(out_buffer),
                CachedBufferMode::HostPtr
            };
            if (g_offloadable_keys.count(data))
            {
                offloadable_lru_insert_front_locked(data);
                g_offloadable_resident_bytes += static_cast<std::int64_t>(bytes);
                // Evict from the tail of the LRU; the just-inserted entry is
                // at the front and is safe (it's the one the caller will use
                // for the in-progress graph build). Eviction of other tail
                // entries frees their MTLBuffer wrappers; any kernel whose
                // graph computed earlier has already released the references
                // it captured at build time.
                offloadable_evict_to_budget_locked();
            }
        }

        return true;
    }

    bool try_get_cacheable_tensor_buffer(
        ggml_backend_t backend, ggml_backend_dev_t dev,
        ggml_tensor* tensor, void* data, std::size_t bytes,
        ggml_backend_buffer_t& out_buffer, void*& out_addr, bool& out_needs_upload,
        enum ggml_backend_buffer_usage usage)
    {
        out_buffer = nullptr;
        out_addr = nullptr;
        out_needs_upload = false;

        if (backend == nullptr || dev == nullptr || tensor == nullptr || data == nullptr || bytes == 0)
            return false;

        const bool use_device_copy = prefers_device_local_cache(dev);

        {
            std::lock_guard<std::mutex> lock(g_preloaded_buffer_cache_mutex);
            auto it = g_preloaded_buffer_cache.find(data);
            if (it != g_preloaded_buffer_cache.end())
            {
                const std::size_t required_size = ggml_backend_buffer_get_alloc_size(it->second.buffer, tensor);
                if (it->second.bytes == bytes &&
                    required_size <= it->second.buffer_size)
                {
                    out_buffer = it->second.buffer;
                    out_addr = ggml_backend_buffer_get_base(out_buffer);
                    return true;
                }
                ggml_backend_buffer_free(it->second.buffer);
                g_preloaded_buffer_cache.erase(it);
            }
        }

        {
            std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
            auto it = g_host_buffer_cache.find(data);
            if (it != g_host_buffer_cache.end())
            {
                const bool mode_matches =
                    (use_device_copy && it->second.mode == CachedBufferMode::DeviceCopy) ||
                    (!use_device_copy && it->second.mode == CachedBufferMode::HostPtr);
                const std::size_t required_size = ggml_backend_buffer_get_alloc_size(it->second.buffer, tensor);

                if (mode_matches &&
                    it->second.bytes == bytes &&
                    required_size <= it->second.buffer_size)
                {
                    out_buffer = it->second.buffer;
                    out_addr = use_device_copy ? ggml_backend_buffer_get_base(out_buffer) : data;
                    return true;
                }
                ggml_backend_buffer_free(it->second.buffer);
                g_host_buffer_cache.erase(it);
            }
        }

        if (use_device_copy)
        {
            ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
            if (buft == nullptr)
                return false;
            const std::size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);
            out_buffer = ggml_backend_buft_alloc_buffer(buft, alloc_size);
            if (out_buffer == nullptr)
                return false;
            ggml_backend_buffer_set_usage(out_buffer, usage);
            out_addr = ggml_backend_buffer_get_base(out_buffer);
            out_needs_upload = true;

            std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
            g_host_buffer_cache[data] = {
                out_buffer, bytes,
                ggml_backend_buffer_get_size(out_buffer),
                CachedBufferMode::DeviceCopy
            };
            return true;
        }

        if (!try_get_host_ptr_buffer(backend, dev, data, bytes, true, out_buffer))
            return false;

        out_addr = data;
        return true;
    }

    bool sync_cached_buffer_to_host(void* data, std::size_t bytes)
    {
        if (data == nullptr || bytes == 0)
            return true;

        ggml_backend_buffer_t buffer = nullptr;
        CachedBufferMode mode = CachedBufferMode::HostPtr;
        {
            std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
            auto it = g_host_buffer_cache.find(data);
            if (it == g_host_buffer_cache.end())
                return true;
            // Size mismatch means the C# pool recycled this host pointer for a
            // larger tensor (typical: KV-cache resize). The cached Metal buffer
            // belongs to the previous, smaller occupant — its contents are
            // stale relative to the new tensor's host memory, so syncing it
            // back would corrupt freshly-initialized data. Treat it as
            // "nothing to sync"; try_get_cacheable_tensor_buffer rebuilds the
            // binding when the next kernel uses this address.
            //
            // We do not eagerly ggml_backend_buffer_free the stale buffer here:
            // pending Metal command buffers may still hold references under
            // async compute, and freeing would race with their completion.
            // try_get_cacheable_tensor_buffer evicts on demand (after the size
            // check there) when it next encounters this address.
            if (bytes > it->second.bytes)
                return true;
            buffer = it->second.buffer;
            mode = it->second.mode;
        }

        if (mode != CachedBufferMode::DeviceCopy || buffer == nullptr)
            return true;

        PooledContextHandle context;
        if (!context.init(64 * 1024))
            return false;

        ggml_tensor* tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_I8, static_cast<std::int64_t>(bytes));
        if (tensor == nullptr)
            return false;

        void* addr = ggml_backend_buffer_get_base(buffer);
        if (addr == nullptr)
            return false;

        ggml_status status = ggml_backend_tensor_alloc(buffer, tensor, addr);
        if (status != GGML_STATUS_SUCCESS)
            return false;

        ggml_backend_tensor_get(tensor, data, 0, bytes);
        ggml_backend_synchronize(g_backend);
        return true;
    }

    // --- Tensor binding creation ---

    TensorBinding create_standard_binding(ggml_context* ctx, const TensorView2DDesc& desc)
    {
        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        ggml_tensor* view = ggml_view_2d(ctx, base, desc.dim1, desc.dim0, static_cast<std::size_t>(desc.stride0) * sizeof(float), 0);
        return { base, view, static_cast<std::size_t>(desc.raw_bytes) };
    }

    TensorBinding create_standard_binding(ggml_context* ctx, const TensorView3DDesc& desc)
    {
        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        ggml_tensor* view = ggml_view_3d(ctx, base, desc.dim2, desc.dim1, desc.dim0,
            static_cast<std::size_t>(desc.stride1) * sizeof(float),
            static_cast<std::size_t>(desc.stride0) * sizeof(float), 0);
        return { base, view, static_cast<std::size_t>(desc.raw_bytes) };
    }

    TensorBinding create_standard_binding(ggml_context* ctx, const TensorView4DDesc& desc)
    {
        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        ggml_tensor* view = ggml_view_4d(ctx, base, desc.ne0, desc.ne1, desc.ne2, desc.ne3,
            static_cast<std::size_t>(desc.nb1),
            static_cast<std::size_t>(desc.nb2),
            static_cast<std::size_t>(desc.nb3), 0);
        return { base, view, static_cast<std::size_t>(desc.raw_bytes) };
    }

    TensorBinding create_contiguous_binding(ggml_context* ctx, const ContiguousTensorDesc& desc)
    {
        ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.element_count);
        return { tensor, tensor, static_cast<std::size_t>(desc.element_count * static_cast<std::int64_t>(sizeof(float))) };
    }

    TensorBinding create_direct_m2_binding(ggml_context* ctx, const TensorView2DDesc& desc)
    {
        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        ggml_tensor* view = ggml_view_2d(ctx, base, desc.dim0, desc.dim1, static_cast<std::size_t>(desc.stride1) * sizeof(float), 0);
        return { base, view, static_cast<std::size_t>(desc.raw_bytes) };
    }

    TensorBinding create_direct_m2_binding(ggml_context* ctx, const TensorView3DDesc& desc)
    {
        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        ggml_tensor* view = ggml_view_3d(ctx, base, desc.dim1, desc.dim2, desc.dim0,
            static_cast<std::size_t>(desc.stride2) * sizeof(float),
            static_cast<std::size_t>(desc.stride0) * sizeof(float), 0);
        return { base, view, static_cast<std::size_t>(desc.raw_bytes) };
    }

    std::vector<float> pack_m2(const TensorView2DDesc& desc)
    {
        const float* data = static_cast<const float*>(desc.data);
        std::vector<float> packed(static_cast<std::size_t>(desc.dim0) * desc.dim1);
        for (int row = 0; row < desc.dim0; ++row)
            for (int col = 0; col < desc.dim1; ++col)
                packed[(static_cast<std::size_t>(col) * desc.dim0) + row] =
                    data[(static_cast<std::size_t>(row) * desc.stride0) + (static_cast<std::size_t>(col) * desc.stride1)];
        return packed;
    }

    std::vector<float> pack_m2(const TensorView3DDesc& desc)
    {
        const float* data = static_cast<const float*>(desc.data);
        std::vector<float> packed(static_cast<std::size_t>(desc.dim0) * desc.dim1 * desc.dim2);
        for (int batch = 0; batch < desc.dim0; ++batch)
            for (int row = 0; row < desc.dim1; ++row)
                for (int col = 0; col < desc.dim2; ++col)
                    packed[((static_cast<std::size_t>(batch) * desc.dim2 + col) * desc.dim1) + row] =
                        data[(static_cast<std::size_t>(batch) * desc.stride0) +
                             (static_cast<std::size_t>(row) * desc.stride1) +
                             (static_cast<std::size_t>(col) * desc.stride2)];
        return packed;
    }

    std::vector<float> pack_standard(const TensorView2DDesc& desc)
    {
        const float* data = static_cast<const float*>(desc.data);
        std::vector<float> packed(static_cast<std::size_t>(desc.dim0) * desc.dim1);
        for (int row = 0; row < desc.dim0; ++row)
            for (int col = 0; col < desc.dim1; ++col)
                packed[(static_cast<std::size_t>(row) * desc.dim1) + col] =
                    data[(static_cast<std::size_t>(row) * desc.stride0) + (static_cast<std::size_t>(col) * desc.stride1)];
        return packed;
    }

    std::vector<float> pack_standard(const TensorView3DDesc& desc)
    {
        const float* data = static_cast<const float*>(desc.data);
        std::vector<float> packed(static_cast<std::size_t>(desc.dim0) * desc.dim1 * desc.dim2);
        for (int batch = 0; batch < desc.dim0; ++batch)
            for (int row = 0; row < desc.dim1; ++row)
                for (int col = 0; col < desc.dim2; ++col)
                    packed[((static_cast<std::size_t>(batch) * desc.dim1 + row) * desc.dim2) + col] =
                        data[(static_cast<std::size_t>(batch) * desc.stride0) +
                             (static_cast<std::size_t>(row) * desc.stride1) +
                             (static_cast<std::size_t>(col) * desc.stride2)];
        return packed;
    }

    TensorBinding create_packed_m2_binding(ggml_context* ctx, const TensorView2DDesc& desc, std::vector<float>& packed)
    {
        packed = pack_m2(desc);
        ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, desc.dim0, desc.dim1);
        return { tensor, tensor, packed.size() * sizeof(float) };
    }

    TensorBinding create_packed_m2_binding(ggml_context* ctx, const TensorView3DDesc& desc, std::vector<float>& packed)
    {
        packed = pack_m2(desc);
        ggml_tensor* tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, desc.dim1, desc.dim2, desc.dim0);
        return { tensor, tensor, packed.size() * sizeof(float) };
    }

    TensorBinding create_packed_standard_binding(ggml_context* ctx, const TensorView2DDesc& desc, std::vector<float>& packed)
    {
        packed = pack_standard(desc);
        ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, desc.dim1, desc.dim0);
        return { tensor, tensor, packed.size() * sizeof(float) };
    }

    TensorBinding create_packed_standard_binding(ggml_context* ctx, const TensorView3DDesc& desc, std::vector<float>& packed)
    {
        packed = pack_standard(desc);
        ggml_tensor* tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, desc.dim2, desc.dim1, desc.dim0);
        return { tensor, tensor, packed.size() * sizeof(float) };
    }

    void upload_binding(const TensorBinding& binding, const void* data, std::size_t size)
    {
        // Async mode safety: ggml_backend_tensor_set on a shared (host-mapped)
        // backend buffer is a CPU memcpy. If the source `data` is host memory that
        // a previously-committed-but-not-yet-completed Metal command buffer is
        // still writing to (e.g. the output of a prior zero-copy op), the memcpy
        // races with the GPU write and reads partial data.
        //
        // Draining pending work here is conservative — it converts every upload
        // into a sync point. Ops that bind their inputs zero-copy don't reach
        // this path, so they still chain freely; only ops that actually copy
        // host data into a backend buffer pay the sync. For prefill on Metal the
        // common path (matmul / addmm_quant / elementwise ops) is zero-copy, so
        // this is rarely hit in steady state.
        host_read_barrier();
        ggml_backend_tensor_set(binding.storage, data, 0, size);
    }

    // --- Zero-copy host-pointer bindings ---

    bool create_binding_from_host_ptr_2d(
        ggml_context* ctx, ggml_backend_t backend, const TensorView2DDesc& desc,
        TensorBinding& out_binding, ggml_backend_buffer_t& out_buffer)
    {
        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (!can_use_host_ptr_buffer(backend, dev, desc.data, raw_bytes)) return false;
        out_buffer = ggml_backend_dev_buffer_from_host_ptr(dev, desc.data, raw_bytes, raw_bytes);
        if (out_buffer == nullptr) return false;

        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        if (base == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_status st = ggml_backend_tensor_alloc(out_buffer, base, const_cast<void*>(desc.data));
        if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_tensor* view = ggml_view_2d(ctx, base, desc.dim1, desc.dim0, static_cast<std::size_t>(desc.stride0) * sizeof(float), 0);
        if (view == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        out_binding = { base, view, raw_bytes };
        return true;
    }

    bool create_binding_from_host_ptr_direct_m2_2d(
        ggml_context* ctx, ggml_backend_t backend, const TensorView2DDesc& desc,
        TensorBinding& out_binding, ggml_backend_buffer_t& out_buffer)
    {
        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (!can_use_host_ptr_buffer(backend, dev, desc.data, raw_bytes)) return false;
        out_buffer = ggml_backend_dev_buffer_from_host_ptr(dev, desc.data, raw_bytes, raw_bytes);
        if (out_buffer == nullptr) return false;

        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        if (base == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_status st = ggml_backend_tensor_alloc(out_buffer, base, const_cast<void*>(desc.data));
        if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_tensor* view = ggml_view_2d(ctx, base, desc.dim0, desc.dim1, static_cast<std::size_t>(desc.stride1) * sizeof(float), 0);
        if (view == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        out_binding = { base, view, raw_bytes };
        return true;
    }

    bool create_binding_from_host_ptr_3d(
        ggml_context* ctx, ggml_backend_t backend, const TensorView3DDesc& desc,
        TensorBinding& out_binding, ggml_backend_buffer_t& out_buffer)
    {
        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (!can_use_host_ptr_buffer(backend, dev, desc.data, raw_bytes)) return false;
        out_buffer = ggml_backend_dev_buffer_from_host_ptr(dev, desc.data, raw_bytes, raw_bytes);
        if (out_buffer == nullptr) return false;

        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        if (base == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_status st = ggml_backend_tensor_alloc(out_buffer, base, const_cast<void*>(desc.data));
        if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_tensor* view = ggml_view_3d(ctx, base, desc.dim2, desc.dim1, desc.dim0,
            static_cast<std::size_t>(desc.stride1) * sizeof(float),
            static_cast<std::size_t>(desc.stride0) * sizeof(float), 0);
        if (view == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        out_binding = { base, view, raw_bytes };
        return true;
    }

    bool create_binding_from_host_ptr_direct_m2_3d(
        ggml_context* ctx, ggml_backend_t backend, const TensorView3DDesc& desc,
        TensorBinding& out_binding, ggml_backend_buffer_t& out_buffer)
    {
        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (!can_use_host_ptr_buffer(backend, dev, desc.data, raw_bytes)) return false;
        out_buffer = ggml_backend_dev_buffer_from_host_ptr(dev, desc.data, raw_bytes, raw_bytes);
        if (out_buffer == nullptr) return false;

        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        if (base == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_status st = ggml_backend_tensor_alloc(out_buffer, base, const_cast<void*>(desc.data));
        if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_tensor* view = ggml_view_3d(ctx, base, desc.dim1, desc.dim2, desc.dim0,
            static_cast<std::size_t>(desc.stride2) * sizeof(float),
            static_cast<std::size_t>(desc.stride0) * sizeof(float), 0);
        if (view == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        out_binding = { base, view, raw_bytes };
        return true;
    }

    bool create_binding_from_host_ptr_4d(
        ggml_context* ctx, ggml_backend_t backend, const TensorView4DDesc& desc,
        TensorBinding& out_binding, ggml_backend_buffer_t& out_buffer)
    {
        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (!can_use_host_ptr_buffer(backend, dev, desc.data, raw_bytes)) return false;
        out_buffer = ggml_backend_dev_buffer_from_host_ptr(dev, desc.data, raw_bytes, raw_bytes);
        if (out_buffer == nullptr) return false;

        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        if (base == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_status st = ggml_backend_tensor_alloc(out_buffer, base, const_cast<void*>(desc.data));
        if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_tensor* view = ggml_view_4d(ctx, base, desc.ne0, desc.ne1, desc.ne2, desc.ne3,
            static_cast<std::size_t>(desc.nb1),
            static_cast<std::size_t>(desc.nb2),
            static_cast<std::size_t>(desc.nb3), 0);
        if (view == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        out_binding = { base, view, raw_bytes };
        return true;
    }

    bool create_binding_from_host_ptr_contiguous(
        ggml_context* ctx, ggml_backend_t backend, const ContiguousTensorDesc& desc,
        TensorBinding& out_binding, ggml_backend_buffer_t& out_buffer)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        std::size_t raw_bytes = static_cast<std::size_t>(desc.element_count) * sizeof(float);
        if (!can_use_host_ptr_buffer(backend, dev, desc.data, raw_bytes)) return false;

        out_buffer = ggml_backend_dev_buffer_from_host_ptr(dev, desc.data, raw_bytes, raw_bytes);
        if (out_buffer == nullptr) return false;

        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.element_count);
        if (base == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_status st = ggml_backend_tensor_alloc(out_buffer, base, const_cast<void*>(desc.data));
        if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        out_binding = { base, base, raw_bytes };
        return true;
    }

    // --- Tensor reshape helpers ---

    ggml_tensor* sum_rows_to_feature_vector(ggml_context* ctx, ggml_tensor* tensor)
    {
        ggml_tensor* transposed = ggml_transpose(ctx, tensor);
        ggml_tensor* transposed_contiguous = transposed == nullptr ? nullptr : ggml_cont(ctx, transposed);
        ggml_tensor* summed = transposed_contiguous == nullptr ? nullptr : ggml_sum_rows(ctx, transposed_contiguous);
        ggml_tensor* restored = summed == nullptr ? nullptr : ggml_transpose(ctx, summed);
        return restored == nullptr ? nullptr : ggml_cont(ctx, restored);
    }

    // --- Op-code dispatch helpers ---

    ggml_tensor* make_unary_tensor(ggml_context* ctx, UnaryOpCode op, ggml_tensor* src)
    {
        switch (op)
        {
        case UnaryOpCode::Neg:     return ggml_neg(ctx, src);
        case UnaryOpCode::Exp:     return ggml_exp(ctx, src);
        case UnaryOpCode::Log:     return ggml_log(ctx, src);
        case UnaryOpCode::Sqrt:    return ggml_sqrt(ctx, src);
        case UnaryOpCode::Relu:    return ggml_relu(ctx, src);
        case UnaryOpCode::Sigmoid: return ggml_sigmoid(ctx, src);
        case UnaryOpCode::Tanh:    return ggml_tanh(ctx, src);
        case UnaryOpCode::SiLU:    return ggml_silu(ctx, src);
        case UnaryOpCode::Step:    return ggml_step(ctx, src);
        case UnaryOpCode::Abs:     return ggml_abs(ctx, src);
        case UnaryOpCode::Sign:    return ggml_sgn(ctx, src);
        case UnaryOpCode::GELU:    return ggml_gelu(ctx, src);
        default:
            set_last_error("Unsupported unary ggml op code.");
            return nullptr;
        }
    }

    ggml_tensor* make_fused_act_mul_tensor(ggml_context* ctx, FusedActMulOpCode op, ggml_tensor* a, ggml_tensor* b)
    {
        switch (op)
        {
        case FusedActMulOpCode::SiLUMul:    return ggml_mul(ctx, ggml_silu(ctx, a), b);
        case FusedActMulOpCode::GELUMul:    return ggml_mul(ctx, ggml_gelu(ctx, a), b);
        case FusedActMulOpCode::SigmoidMul: return ggml_mul(ctx, a, ggml_sigmoid(ctx, b));
        default:
            set_last_error("Unsupported fused activation-multiply ggml op code.");
            return nullptr;
        }
    }

    ggml_tensor* make_binary_tensor(ggml_context* ctx, BinaryTensorOpCode op, ggml_tensor* lhs, ggml_tensor* rhs)
    {
        switch (op)
        {
        case BinaryTensorOpCode::Add: return ggml_add(ctx, lhs, rhs);
        case BinaryTensorOpCode::Sub: return ggml_sub(ctx, lhs, rhs);
        case BinaryTensorOpCode::Mul: return ggml_mul(ctx, lhs, rhs);
        case BinaryTensorOpCode::Div: return ggml_div(ctx, lhs, rhs);
        default:
            set_last_error("Unsupported binary ggml op code.");
            return nullptr;
        }
    }

    ggml_tensor* make_norm_tensor(ggml_context* ctx, NormOpCode op, ggml_tensor* src, float eps)
    {
        switch (op)
        {
        case NormOpCode::LayerNorm: return ggml_norm(ctx, src, eps);
        case NormOpCode::RmsNorm:   return ggml_rms_norm(ctx, src, eps);
        default:
            set_last_error("Unsupported norm ggml op code.");
            return nullptr;
        }
    }

    ggml_tensor* make_reduction_tensor(ggml_context* ctx, ReductionOpCode op, ggml_tensor* src)
    {
        switch (op)
        {
        case ReductionOpCode::Sum:  return ggml_sum_rows(ctx, src);
        case ReductionOpCode::Mean: return ggml_mean(ctx, src);
        default:
            set_last_error("Unsupported reduction ggml op code.");
            return nullptr;
        }
    }

    // --- Cross-entropy label buffer ---

    bool build_cross_entropy_label_buffer(
        std::vector<float>& labels,
        const ContiguousTensorDesc& target_indices_desc,
        std::int64_t rows, std::int64_t cols, float label_smooth)
    {
        if (target_indices_desc.element_count != rows)
        {
            set_last_error("Target index count must match the number of probability rows for ggml crossentropyloss.");
            return false;
        }

        const float base_value = label_smooth > 0.0f
            ? (label_smooth / static_cast<float>(cols))
            : 0.0f;
        const float target_value = 1.0f - label_smooth + (label_smooth / static_cast<float>(cols));

        labels.assign(static_cast<std::size_t>(rows * cols), base_value);

        std::vector<std::int32_t> target_indices;
        if (!read_i32_values(target_indices, target_indices_desc, "targetIndices"))
            return false;

        for (std::int64_t row = 0; row < rows; ++row)
        {
            const std::int64_t target_index = static_cast<std::int64_t>(target_indices[static_cast<std::size_t>(row)]);
            if (target_index < 0 || target_index >= cols)
            {
                set_last_error("Target index out of range for ggml crossentropyloss.");
                return false;
            }
            labels[static_cast<std::size_t>(row * cols + target_index)] = target_value;
        }

        return true;
    }

} // namespace tsg

// ============================================================================
// Exported utility functions
// ============================================================================

using namespace tsg;

TSG_EXPORT const char* TSGgml_GetLastError()
{
    return g_last_error.c_str();
}

TSG_EXPORT int TSGgml_IsMetalAvailable()
{
    clear_last_error();
    return can_initialize_backend(BACKEND_TYPE_METAL) ? 1 : 0;
}

TSG_EXPORT int TSGgml_CanInitializeBackend(int backendType)
{
    clear_last_error();
    return can_initialize_backend(backendType) ? 1 : 0;
}

TSG_EXPORT int TSGgml_IsBackendAvailable(int backendType)
{
    clear_last_error();
    return ensure_backend(backendType) ? 1 : 0;
}

TSG_EXPORT void* TSGgml_AlignedAlloc(size_t size)
{
    if (size == 0)
        return nullptr;
    const size_t alignment = 16384;
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(size, alignment);
#else
    if (posix_memalign(&ptr, alignment, size) != 0)
        return nullptr;
#endif
    return ptr;
}

TSG_EXPORT void TSGgml_AlignedFree(void* ptr)
{
    if (ptr == nullptr)
        return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

TSG_EXPORT void TSGgml_ClearHostBufferCache()
{
    {
        std::lock_guard<std::mutex> lock(g_preloaded_buffer_cache_mutex);
        for (auto& [ptr, cached] : g_preloaded_buffer_cache)
            ggml_backend_buffer_free(cached.buffer);
        g_preloaded_buffer_cache.clear();
    }

    {
        std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
        for (auto& [ptr, cached] : g_host_buffer_cache)
            ggml_backend_buffer_free(cached.buffer);
        g_host_buffer_cache.clear();
        g_offloadable_lru.clear();
        g_offloadable_lru_map.clear();
        g_offloadable_resident_bytes = 0;
    }
}

// Tear down the process-global GGML backend and any state that holds device
// resource references. Must be called before the process's C runtime
// finalisers run. The .NET host wires this onto AppDomain.ProcessExit /
// IHostApplicationLifetime.ApplicationStopped so SIGINT-driven shutdowns
// reach it.
//
// Why this exists: on macOS the ggml-metal backend's device singleton is a
// C++ static unique_ptr whose deleter asserts that the device's resource set
// is empty (ggml-metal-device.m:608: GGML_ASSERT([rsets->data count] == 0)).
// Without an explicit free, g_backend (and the MTLBuffer wrappers it holds
// via g_host_buffer_cache / g_preloaded_buffer_cache) outlives the .NET host
// and the assertion fires inside __cxa_finalize_ranges, aborting the
// process. Freeing the backend here drains every Metal command buffer and
// releases the resource-set entries before the device deleter runs.
TSG_EXPORT void TSGgml_Shutdown()
{
    {
        std::lock_guard<std::mutex> lock(g_preloaded_buffer_cache_mutex);
        for (auto& [ptr, cached] : g_preloaded_buffer_cache)
            ggml_backend_buffer_free(cached.buffer);
        g_preloaded_buffer_cache.clear();
    }

    {
        std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
        for (auto& [ptr, cached] : g_host_buffer_cache)
            ggml_backend_buffer_free(cached.buffer);
        g_host_buffer_cache.clear();
        g_offloadable_keys.clear();
        g_offloadable_lru.clear();
        g_offloadable_lru_map.clear();
        g_offloadable_resident_bytes = 0;
        g_offloadable_budget = 0;
    }

    if (g_backend != nullptr)
    {
        ggml_backend_synchronize(g_backend);
        ggml_backend_free(g_backend);
        g_backend = nullptr;
    }
    g_pending_gpu_work.store(false, std::memory_order_release);
}

// Mark a host data pointer as eligible for the MoE expert offload LRU.
// Once registered, subsequent cache lookups for that pointer update an LRU,
// and cache misses that grow the resident byte total beyond the configured
// budget trigger eviction from the LRU tail. Registration is sticky — call
// TSGgml_ClearOffloadableState to reset (typically on model unload).
TSG_EXPORT void TSGgml_RegisterOffloadable(void* key)
{
    if (key == nullptr)
        return;
    std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
    g_offloadable_keys.insert(key);
}

// Set the byte ceiling for offloadable cache residency. Zero (or negative)
// disables eviction (registered entries still participate in the LRU but
// nothing is freed).
TSG_EXPORT void TSGgml_SetOffloadableBudget(int64_t bytes)
{
    std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
    g_offloadable_budget = bytes > 0 ? bytes : 0;
    offloadable_evict_to_budget_locked();
}

// Clear the offloadable registry, LRU, and byte accounting. Does NOT touch
// the underlying CachedHostBuffer entries — they remain reachable via
// g_host_buffer_cache and will be freed by TSGgml_ClearHostBufferCache or
// when the process exits.
TSG_EXPORT void TSGgml_ClearOffloadableState()
{
    std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
    g_offloadable_keys.clear();
    g_offloadable_lru.clear();
    g_offloadable_lru_map.clear();
    g_offloadable_resident_bytes = 0;
    g_offloadable_budget = 0;
}

TSG_EXPORT void TSGgml_InvalidateHostBuffer(void* ptr)
{
    invalidate_cached_buffer(ptr);
}

TSG_EXPORT int TSGgml_SyncHostBuffer(void* ptr, size_t size)
{
    if (sync_cached_buffer_to_host(ptr, size))
    {
        clear_last_error();
        return 1;
    }
    set_last_error("Failed to synchronize cached GGML device buffer back to host memory.");
    return 0;
}

// Toggle the lazy-sync code path on the per-op kernels. When enabled, ops that
// wrote their result to host-mapped memory (zero-copy) on the Metal backend skip
// the trailing ggml_backend_synchronize so the next op's command buffer can be
// queued while the previous one is still running on the GPU.
//
// C# enables this once at backend init (see GgmlBasicOps.SetAsyncCompute) and
// pairs it with a barrier in TensorComputePrimitives.GetFloatPointer so that
// host-side reads always see fully-flushed data.
TSG_EXPORT void TSGgml_SetAsyncCompute(int enabled)
{
    bool desired = enabled != 0;
    bool previous = g_async_compute_enabled.exchange(desired, std::memory_order_acq_rel);

    // When async is being turned off, drain any pending GPU work so subsequent
    // host reads don't see stale data.
    if (previous && !desired)
    {
        if (g_pending_gpu_work.exchange(false, std::memory_order_acq_rel) && g_backend != nullptr)
        {
            ggml_backend_synchronize(g_backend);
        }
    }
}

TSG_EXPORT int TSGgml_GetAsyncCompute()
{
    return g_async_compute_enabled.load(std::memory_order_acquire) ? 1 : 0;
}

// Drain pending GPU work iff any was deferred. Returns 1 when it actually
// blocked on the backend, 0 when there was nothing to do. Safe to call from
// any thread; cheap when there's no pending work (single atomic exchange).
TSG_EXPORT int TSGgml_HostReadBarrier()
{
    return host_read_barrier() ? 1 : 0;
}

TSG_EXPORT int TSGgml_PreloadQuantizedWeight(
    void* cache_key, void* host_data, int ggml_type,
    int64_t ne0, int64_t ne1, int64_t raw_bytes)
{
    try
    {
        if (!ensure_backend())
            return 0;

        if (cache_key == nullptr || host_data == nullptr || ne0 <= 0 || ne1 <= 0 || raw_bytes <= 0)
        {
            set_last_error("Invalid arguments for quantized weight preload.");
            return 0;
        }

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        if (dev == nullptr)
        {
            set_last_error("No GGML backend device is available for quantized weight preload.");
            return 0;
        }

        if (!prefers_device_local_cache(dev))
        {
            clear_last_error();
            return 1;
        }

        const std::size_t bytes = static_cast<std::size_t>(raw_bytes);
        const enum ggml_type qtype = static_cast<enum ggml_type>(ggml_type);

        PooledContextHandle context;
        if (!context.init(64 * 1024))
        {
            set_last_error("Failed to create GGML context for quantized weight preload.");
            return 0;
        }

        ggml_tensor* tensor = ggml_new_tensor_2d(context.value, qtype, ne0, ne1);
        if (tensor == nullptr)
        {
            set_last_error("Failed to create GGML tensor for quantized weight preload.");
            return 0;
        }

        {
            std::lock_guard<std::mutex> lock(g_preloaded_buffer_cache_mutex);
            auto it = g_preloaded_buffer_cache.find(cache_key);
            if (it != g_preloaded_buffer_cache.end())
            {
                const std::size_t required_size = ggml_backend_buffer_get_alloc_size(it->second.buffer, tensor);
                if (it->second.bytes == bytes &&
                    required_size <= it->second.buffer_size)
                {
                    clear_last_error();
                    return 1;
                }
                ggml_backend_buffer_free(it->second.buffer);
                g_preloaded_buffer_cache.erase(it);
            }
        }

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(g_backend);
        if (buft == nullptr)
        {
            set_last_error("Failed to get GGML backend buffer type for quantized weight preload.");
            return 0;
        }

        const std::size_t alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, alloc_size);
        if (buffer == nullptr)
        {
            set_last_error("Failed to allocate GGML backend buffer for quantized weight preload.");
            return 0;
        }

        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        void* addr = ggml_backend_buffer_get_base(buffer);
        if (addr == nullptr)
        {
            ggml_backend_buffer_free(buffer);
            set_last_error("Failed to get GGML backend buffer base for quantized weight preload.");
            return 0;
        }

        const ggml_status alloc_status = ggml_backend_tensor_alloc(buffer, tensor, addr);
        if (alloc_status != GGML_STATUS_SUCCESS)
        {
            ggml_backend_buffer_free(buffer);
            set_last_error("Failed to bind GGML tensor to backend buffer during quantized weight preload.");
            return 0;
        }

        ggml_backend_tensor_set(tensor, host_data, 0, bytes);
        ggml_backend_synchronize(g_backend);

        {
            std::lock_guard<std::mutex> lock(g_preloaded_buffer_cache_mutex);
            g_preloaded_buffer_cache[cache_key] = {
                buffer, bytes,
                ggml_backend_buffer_get_size(buffer),
                CachedBufferMode::DeviceCopy
            };
        }

        clear_last_error();
        return 1;
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error while preloading quantized weight.");
        return 0;
    }
}

TSG_EXPORT size_t TSGgml_RowSize(int ggml_type, int64_t ne)
{
    if (ggml_type < 0 || ggml_type >= GGML_TYPE_COUNT || ne <= 0)
        return 0;
    const enum ggml_type t = static_cast<enum ggml_type>(ggml_type);
    const int64_t bs = ggml_blck_size(t);
    if (bs <= 0 || ne % bs != 0)
        return 0;
    return ggml_row_size(t, ne);
}

TSG_EXPORT int TSGgml_DequantizeToF32(int ggml_type, const void* src, int64_t num_elements, float* dst)
{
    if (src == nullptr || dst == nullptr || num_elements < 0)
        return -1;
    if (num_elements == 0)
        return 0;
    if (ggml_type == GGML_TYPE_F32)
    {
        std::memcpy(dst, src, static_cast<size_t>(num_elements) * sizeof(float));
        return 0;
    }
    const struct ggml_type_traits* traits = ggml_get_type_traits(static_cast<enum ggml_type>(ggml_type));
    if (traits != nullptr && traits->to_float != nullptr)
    {
        traits->to_float(src, dst, num_elements);
        return 0;
    }
    if (ggml_type == GGML_TYPE_Q8_K)
    {
        dequantize_row_q8_K(static_cast<const block_q8_K*>(src), dst, num_elements);
        return 0;
    }
    return -2;
}
