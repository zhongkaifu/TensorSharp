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

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#if defined(GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

#if defined(GGML_USE_VULKAN)
#include "ggml-vulkan.h"
#endif

#include <cstdio>
#include <thread>

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
    // Vulkan device index requested via TSGgml_SetVulkanDeviceIndex. Must be set
    // before the first backend init (create_backend_instance runs once under
    // g_backend_init_once); later calls with a different index fail. Indices are
    // positions in ggml-vulkan's enumeration order (after any
    // GGML_VK_VISIBLE_DEVICES filtering applied at process launch).
    std::atomic<int> g_vulkan_device_index{0};

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

    // Device-copy VRAM budget — see ggml_ops_internal.h for the contract.
    std::int64_t g_device_copy_resident_bytes = 0;
    std::int64_t g_device_copy_budget_bytes = 0;

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

#if defined(_WIN32)
    // Crash triage aid: with TS_GGML_VEH_DEBUG=1 a vectored exception handler
    // prints the faulting module + offset of any access violation to stderr
    // before the process dies. .NET's managed stack trace only shows the
    // TSGgml_* entry point; this tells you whether the fault is inside
    // GgmlOps.dll, a GPU driver DLL, or the runtime. First-chance handler,
    // EXCEPTION_CONTINUE_SEARCH: the CLR's own AV-based null-ref machinery
    // still works (those prints are recognizable by their coreclr/jit module).
    static LONG WINAPI tsg_veh_log_access_violation(PEXCEPTION_POINTERS info)
    {
        if (info != nullptr && info->ExceptionRecord != nullptr &&
            info->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION)
        {
            void* ip = info->ExceptionRecord->ExceptionAddress;
            const ULONG_PTR is_write = info->ExceptionRecord->NumberParameters > 1
                ? info->ExceptionRecord->ExceptionInformation[0] : 0;
            const ULONG_PTR target = info->ExceptionRecord->NumberParameters > 1
                ? info->ExceptionRecord->ExceptionInformation[1] : 0;
            HMODULE mod = nullptr;
            char mod_name[MAX_PATH] = { 0 };
            if (GetModuleHandleExA(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    reinterpret_cast<LPCSTR>(ip), &mod) && mod != nullptr)
            {
                GetModuleFileNameA(mod, mod_name, MAX_PATH);
            }
            std::fprintf(stderr, "[tsg-veh] ACCESS_VIOLATION ip=%p (%s+0x%llx) %s addr=%p\n",
                ip,
                mod_name[0] != '\0' ? mod_name : "(jit/unknown)",
                mod != nullptr
                    ? static_cast<unsigned long long>(
                        reinterpret_cast<char*>(ip) - reinterpret_cast<char*>(mod))
                    : 0ULL,
                is_write ? "writing" : "reading",
                reinterpret_cast<void*>(target));

            // Poor-man's backtrace: scan the faulting thread's stack for
            // quadwords that land inside a loaded module and print module+offset.
            // No frame pointers / PDBs needed; noisy but enough to identify the
            // native caller chain of a memcpy fault.
            if (info->ContextRecord != nullptr)
            {
                const ULONG_PTR* sp = reinterpret_cast<const ULONG_PTR*>(info->ContextRecord->Rsp);
                int printed = 0;
                for (int slot = 0; slot < 256 && printed < 12; ++slot)
                {
                    ULONG_PTR value = 0;
                    __try { value = sp[slot]; }
                    __except (EXCEPTION_EXECUTE_HANDLER) { break; }
                    if (value < 0x10000)
                        continue;
                    HMODULE frame_mod = nullptr;
                    if (!GetModuleHandleExA(
                            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCSTR>(value), &frame_mod) || frame_mod == nullptr)
                        continue;
                    char frame_name[MAX_PATH] = { 0 };
                    GetModuleFileNameA(frame_mod, frame_name, MAX_PATH);
                    std::fprintf(stderr, "[tsg-veh]   stack[%3d] %s+0x%llx\n", slot,
                        frame_name,
                        static_cast<unsigned long long>(
                            value - reinterpret_cast<ULONG_PTR>(frame_mod)));
                    ++printed;
                }
            }
            std::fflush(stderr);
        }
        return EXCEPTION_CONTINUE_SEARCH;
    }
#endif

    static void install_crash_triage_handler_if_requested()
    {
#if defined(_WIN32)
        if (is_truthy_env(std::getenv("TS_GGML_VEH_DEBUG")))
            AddVectoredExceptionHandler(1, tsg_veh_log_access_violation);
#endif
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

        if (backend_type == BACKEND_TYPE_VULKAN)
        {
#if defined(GGML_USE_VULKAN)
            // Init by the Vulkan-specific API rather than dev_by_type(GPU): when
            // several GPU backends are compiled into one binary (CUDA + Vulkan),
            // dev_by_type returns the first registered GPU device, which is
            // ggml-cuda's. The CUDA branch above keeps that behaviour; here the
            // Vulkan device must be picked explicitly.
            const int device_count = ggml_backend_vk_get_device_count();
            if (device_count <= 0)
            {
                set_last_error("No Vulkan device is available for ggml-vulkan.");
                return nullptr;
            }

            const int device_index = g_vulkan_device_index.load(std::memory_order_acquire);
            if (device_index < 0 || device_index >= device_count)
            {
                set_last_error("Vulkan device index " + std::to_string(device_index) +
                    " is out of range: " + std::to_string(device_count) + " Vulkan device(s) available.");
                return nullptr;
            }

            ggml_backend_t backend = ggml_backend_vk_init(static_cast<size_t>(device_index));
            if (backend == nullptr)
                set_last_error("ggml-vulkan backend initialization failed.");
            return backend;
#else
            set_last_error("The ggml-vulkan backend is not available in this build.");
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
        install_crash_triage_handler_if_requested();
        g_backend = create_backend_instance(g_backend_type);
        if (g_backend == nullptr)
            return;
        ggml_pool::ensure_initial_pool();
    }

    bool ensure_backend(int backend_type)
    {
        if (backend_type != BACKEND_TYPE_METAL &&
            backend_type != BACKEND_TYPE_CPU &&
            backend_type != BACKEND_TYPE_CUDA &&
            backend_type != BACKEND_TYPE_VULKAN)
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

        if (backend_type == BACKEND_TYPE_VULKAN)
        {
#if defined(GGML_USE_VULKAN)
            return true;
#else
            set_last_error("The ggml-vulkan backend is not available in this build.");
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

    DeviceStaticProps get_device_static_props(ggml_backend_dev_t dev)
    {
        static std::mutex s_mutex;
        static std::unordered_map<ggml_backend_dev_t, DeviceStaticProps> s_cache;
        std::lock_guard<std::mutex> lock(s_mutex);
        auto it = s_cache.find(dev);
        if (it != s_cache.end())
            return it->second;
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        DeviceStaticProps s{ props.type, props.caps.buffer_from_host_ptr };
        s_cache.emplace(dev, s);
        return s;
    }

    bool prefers_device_local_cache(ggml_backend_dev_t dev)
    {
        if (dev == nullptr)
            return false;
        // Upstream ggml's ggml_backend_dev_props has no `integrated` field (that was an
        // ollama-fork extension). On the backends we use the field was effectively always
        // 0 anyway -- the Metal backend reports type=GPU and never set it -- so the
        // discrete-GPU test reduces to "is this a GPU device".
        //
        // NOTE: This governs the binding policy for *read-write* tensors
        // (activations, KV cache). For those, even on unified-memory Metal we
        // keep the device-local + explicit upload/download path because the
        // zero-copy host-ptr path for read-write tensors is not exercised on
        // Metal (it relies on a lazy-sync model that the per-op activation
        // bindings here don't fully honour). Large *read-only weights* are
        // handled separately and ARE wrapped zero-copy on Metal -- see the
        // unified-memory weight branch in try_get_cacheable_tensor_buffer,
        // which is where the model-weight memory duplication is avoided.
        //
        // Integrated GPUs count as GPUs here. Upstream ggml now reports them as
        // GGML_BACKEND_DEVICE_TYPE_IGPU (ggml-vulkan for iGPUs behind e.g.
        // --gpu-device, ggml-cuda for Tegra). Excluding IGPU broke the preload
        // contract: TSGgml_PreloadQuantizedWeight early-returns success when
        // this predicate is false WITHOUT caching anything, the managed side
        // then releases the host weight copies, and the first forward's cache
        // miss dereferenced the opaque GCHandle cache key as if it were weight
        // bytes -> access violation on Intel iGPUs (their UMA device buffers
        // work exactly like discrete ones for our binding purposes).
        const enum ggml_backend_dev_type type = get_device_static_props(dev).type;
        return type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU;
    }

    // Capability-only test: can this host pointer be wrapped as a device-visible
    // buffer at all (backend supports buffer_from_host_ptr and the pointer meets
    // the buffer-type alignment)? Unlike can_use_host_ptr_buffer this does NOT
    // consult prefers_device_local_cache, so it returns true on unified-memory
    // Metal. Used by the read-only-weight zero-copy path; read-write activation
    // bindings continue to gate on can_use_host_ptr_buffer.
    bool host_ptr_buffer_capable(ggml_backend_t backend, ggml_backend_dev_t dev, const void* ptr, std::size_t size)
    {
        if (dev == nullptr || ptr == nullptr || size == 0)
            return false;
        if (!get_device_static_props(dev).buffer_from_host_ptr)
            return false;
        const std::size_t alignment = get_host_ptr_alignment(backend, dev);
        return is_pointer_aligned(ptr, alignment);
    }

    bool can_use_host_ptr_buffer(ggml_backend_t backend, ggml_backend_dev_t dev, const void* ptr, std::size_t size)
    {
        if (prefers_device_local_cache(dev))
            return false;
        return host_ptr_buffer_capable(backend, dev, ptr, size);
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

    // --- Device-copy budget accounting (caller holds g_host_buffer_cache_mutex) ---

    static void device_copy_account_remove_locked(const CachedHostBuffer& entry)
    {
        if (entry.mode != CachedBufferMode::DeviceCopy)
            return;
        const std::int64_t sz = static_cast<std::int64_t>(entry.buffer_size);
        g_device_copy_resident_bytes = g_device_copy_resident_bytes >= sz
            ? g_device_copy_resident_bytes - sz : 0;
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
        device_copy_account_remove_locked(cit->second);
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
            device_copy_account_remove_locked(it->second);
            ggml_backend_buffer_free(it->second.buffer);
            g_host_buffer_cache.erase(it);
        }
    }

    bool try_get_host_ptr_buffer(
        ggml_backend_t backend, ggml_backend_dev_t dev,
        void* data, std::size_t bytes, bool cacheable,
        ggml_backend_buffer_t& out_buffer,
        bool allow_unified_weight)
    {
        out_buffer = nullptr;
        const bool capable = allow_unified_weight
            ? host_ptr_buffer_capable(backend, dev, data, bytes)
            : can_use_host_ptr_buffer(backend, dev, data, bytes);
        if (!capable)
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

        // Read-only model weights on a unified-memory backend (Metal on Apple
        // Silicon) are wrapped zero-copy around their host/mmap pointer rather
        // than copied into a device-local buffer. This is THE fix for model
        // weight memory blow-up: a 12 GB Q8_0 model otherwise pays ~12 GB of
        // dirty anonymous device copies ON TOP of the 12 GB GGUF mmap (~24 GB,
        // swapping on a 24 GB box). The weight bytes are read-only and the
        // GGUF mmap stays alive for the model's lifetime, so the wrap is safe.
        //
        // Restricted to USAGE_WEIGHTS: small read-write tensors (KV cache,
        // activations) are bound with USAGE_COMPUTE and keep the device-local
        // copy path, whose explicit upload/download is what the Metal kernels
        // here rely on for correctness.
        const bool unified_weight =
            usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
            g_backend_type == BACKEND_TYPE_METAL &&
            host_ptr_buffer_capable(backend, dev, data, bytes);

        const bool use_device_copy = prefers_device_local_cache(dev) && !unified_weight;

        static const bool s_bind_debug = is_truthy_env(std::getenv("TS_GGML_BIND_DEBUG"));

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
                if (s_bind_debug)
                {
                    std::fprintf(stderr,
                        "[bind-debug] preload entry REJECTED key=%p bytes=%zu(entry %zu) required=%zu buffer=%zu type=%s ne=[%lld,%lld]\n",
                        data, bytes, it->second.bytes, required_size, it->second.buffer_size,
                        ggml_type_name(tensor->type),
                        (long long)tensor->ne[0], (long long)tensor->ne[1]);
                    std::fflush(stderr);
                }
                ggml_backend_buffer_free(it->second.buffer);
                g_preloaded_buffer_cache.erase(it);
            }
            else if (s_bind_debug && bytes >= 4096)
            {
                std::fprintf(stderr,
                    "[bind-debug] preload MISS key=%p bytes=%zu type=%s ne=[%lld,%lld] map_size=%zu\n",
                    data, bytes, ggml_type_name(tensor->type),
                    (long long)tensor->ne[0], (long long)tensor->ne[1],
                    g_preloaded_buffer_cache.size());
                std::fflush(stderr);
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
                device_copy_account_remove_locked(it->second);
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

            // Device-copy budget: refuse to create a NEW resident copy past the
            // budget so VRAM is never oversubscribed (the caller streams the
            // tensor through the per-graph upload path instead). Existing cache
            // hits returned above are unaffected.
            {
                std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
                if (g_device_copy_budget_bytes > 0 &&
                    g_device_copy_resident_bytes + static_cast<std::int64_t>(alloc_size) > g_device_copy_budget_bytes)
                {
                    return false;
                }
            }

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
            g_device_copy_resident_bytes += static_cast<std::int64_t>(ggml_backend_buffer_get_size(out_buffer));
            return true;
        }

        if (!try_get_host_ptr_buffer(backend, dev, data, bytes, true, out_buffer, unified_weight))
            return false;

        out_addr = data;
        return true;
    }

    // --- Reusable compute buffer for per-graph intermediate tensors ---
    //
    // Per-layer Gemma4 prefill builds a fresh ggml graph each layer and used to
    // allocate a fresh Metal backend buffer for its intermediate activations on
    // every call. That allocation (ggml_backend_alloc_ctx_tensors -> Metal
    // newBufferWithLength of ~100-150 MB for a 512-token chunk) costs ~20 ms and
    // ran 42x per chunk, dominating prefill wall time. The buffer's contents are
    // fully overwritten by each graph_compute (every intermediate is produced
    // before it is consumed), and the per-layer host_read_barrier drains the
    // previous layer's GPU work before the next graph runs, so a single buffer
    // can be safely reused (re-packed via ggml_tallocr) across calls.
    static std::mutex g_reuse_compute_mutex;
    static ggml_backend_buffer_t g_reuse_compute_buf = nullptr;
    static std::size_t g_reuse_compute_size = 0;
    static ggml_backend_t g_reuse_compute_backend = nullptr;

    // Persistent graph allocator for the large multi-token fused graphs (e.g. the
    // MTP MoE verify). Those used to ggml_gallocr_new()/ggml_gallocr_free() a
    // ~400 MB device buffer on EVERY call; on Metal that per-call alloc+free of a
    // large shared (vm_allocate) buffer fragments the device VM over hundreds of
    // verify steps until a contiguous allocation fails (OOM). A single gallocr
    // reused across calls grows its buffer once and keeps it, eliminating the
    // churn (and the per-call ~20 ms Metal allocation). Reset on backend swap.
    static std::mutex g_reuse_gallocr_mutex;
    static ggml_gallocr_t g_reuse_gallocr = nullptr;
    static ggml_backend_t g_reuse_gallocr_backend = nullptr;

    void free_reuse_compute_buffer()
    {
        std::lock_guard<std::mutex> lock(g_reuse_compute_mutex);
        if (g_reuse_compute_buf != nullptr)
        {
            ggml_backend_buffer_free(g_reuse_compute_buf);
            g_reuse_compute_buf = nullptr;
        }
        g_reuse_compute_size = 0;
        g_reuse_compute_backend = nullptr;
    }

    void free_reuse_gallocr()
    {
        std::lock_guard<std::mutex> lock(g_reuse_gallocr_mutex);
        if (g_reuse_gallocr != nullptr)
        {
            ggml_gallocr_free(g_reuse_gallocr);
            g_reuse_gallocr = nullptr;
        }
        g_reuse_gallocr_backend = nullptr;
    }

    // Allocate `graph`'s intermediates into a persistent, reused gallocr (grown on
    // demand). Returns false if the gallocr could not be created/allocated, in
    // which case the caller should fall back to its own gallocr or per-op path.
    // The caller must NOT free the gallocr; it lives for the backend's lifetime.
    bool alloc_graph_reuse_gallocr(ggml_cgraph* graph)
    {
        // Escape hatch (shares the reuse-buffer toggle): TS_GGML_REUSE_COMPUTE_BUF=0
        // disables both so A/B testing can isolate the persistent allocators.
        static const bool s_disabled = []() {
            const char* e = std::getenv("TS_GGML_REUSE_COMPUTE_BUF");
            return e != nullptr && e[0] == '0';
        }();
        if (s_disabled || g_backend == nullptr || graph == nullptr)
            return false;

        std::lock_guard<std::mutex> lock(g_reuse_gallocr_mutex);
        if (g_reuse_gallocr_backend != g_backend)
        {
            // Backend swapped (model reload). The old backend freed its buffers on
            // teardown, so drop the stale handle rather than freeing through it.
            g_reuse_gallocr = nullptr;
            g_reuse_gallocr_backend = g_backend;
        }
        if (g_reuse_gallocr == nullptr)
        {
            g_reuse_gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
            if (g_reuse_gallocr == nullptr)
                return false;
        }
        // ggml_gallocr_alloc_graph reuses the existing buffer when the new graph
        // fits and grows (reallocates) it only when a larger graph appears.
        bool ok = ggml_gallocr_alloc_graph(g_reuse_gallocr, graph);
        static const bool s_memlog = std::getenv("TS_GMTP_MEMLOG") != nullptr;
        if (ok && s_memlog)
        {
            static std::size_t s_last = 0;
            std::size_t sz = ggml_gallocr_get_buffer_size(g_reuse_gallocr, 0);
            if (sz != s_last)
            {
                fprintf(stderr, "[memlog] reuse_gallocr size %zu -> %zu MB\n",
                        s_last / 1024 / 1024, sz / 1024 / 1024);
                s_last = sz;
            }
        }
        return ok;
    }

    bool alloc_ctx_tensors_reuse(ggml_context* ctx)
    {
        // Escape hatch for A/B testing / regression isolation.
        static const bool s_disabled = []() {
            const char* e = std::getenv("TS_GGML_REUSE_COMPUTE_BUF");
            return e != nullptr && e[0] == '0';
        }();
        if (s_disabled)
            return false;

        if (g_backend == nullptr || ctx == nullptr)
            return false;

        ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(g_backend);
        if (buft == nullptr)
            return false;

        const std::size_t needed = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
        if (needed == 0)
            return true; // every tensor already has a buffer (all inputs pre-bound)

        const std::size_t max_size = ggml_backend_buft_get_max_size(buft);
        if (needed > max_size)
            return false; // would require splitting across buffers; caller falls back

        std::lock_guard<std::mutex> lock(g_reuse_compute_mutex);

        // A backend swap (model reload) invalidates the cached buffer. The old
        // backend already freed its buffers on teardown, so just drop the stale
        // handle rather than freeing through the dead backend.
        if (g_reuse_compute_backend != g_backend)
        {
            g_reuse_compute_buf = nullptr;
            g_reuse_compute_size = 0;
            g_reuse_compute_backend = g_backend;
        }

        if (g_reuse_compute_buf == nullptr || g_reuse_compute_size < needed)
        {
            // Grow with slack rounded up to a 64 MiB boundary. The graph's
            // intermediate footprint creeps up by sub-MB amounts every decode step
            // (the attention scratch scales with the growing context), so allocating
            // exactly `needed` reallocates the buffer on EVERY step. On Metal each
            // realloc frees+allocs a multi-hundred-MB shared (vm_allocate) buffer;
            // doing that hundreds of times fragments the device VM until a large
            // contiguous allocation (e.g. the MTP verify graph) can no longer be
            // satisfied -> kIOGPUCommandBufferCallbackErrorOutOfMemory even though
            // total free bytes remain. Rounding to 64 MiB makes the buffer grow in
            // rare, big steps and be reused unchanged across thousands of decodes.
            std::size_t alloc_size = needed;
            const std::size_t slab = static_cast<std::size_t>(64) * 1024 * 1024;
            alloc_size = ((alloc_size + slab - 1) / slab) * slab;
            if (alloc_size > max_size) alloc_size = max_size; // never exceed a single buffer
            if (alloc_size < needed) alloc_size = needed;
            if (std::getenv("TS_GMTP_MEMLOG") != nullptr)
                fprintf(stderr, "[memlog] reuse_compute_buf grow %zu -> %zu MB (needed %zu MB)\n",
                        g_reuse_compute_size / 1024 / 1024, alloc_size / 1024 / 1024, needed / 1024 / 1024);
            if (g_reuse_compute_buf != nullptr)
                ggml_backend_buffer_free(g_reuse_compute_buf);
            g_reuse_compute_buf = ggml_backend_buft_alloc_buffer(buft, alloc_size);
            if (g_reuse_compute_buf == nullptr)
            {
                g_reuse_compute_size = 0;
                return false;
            }
            g_reuse_compute_size = alloc_size;
            ggml_backend_buffer_set_usage(g_reuse_compute_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        }

        // Re-pack this graph's unallocated tensors into the cached buffer. Mirrors
        // ggml-alloc.c's alloc_tensor_range exactly (the size query above used the
        // identical iteration, so everything fits a single buffer here).
        ggml_tallocr tallocr = ggml_tallocr_new(g_reuse_compute_buf);
        for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t))
        {
            ggml_status status = GGML_STATUS_SUCCESS;
            if (t->data == nullptr)
            {
                if (t->view_src == nullptr)
                    status = ggml_tallocr_alloc(&tallocr, t);
                else if (t->buffer == nullptr)
                    status = ggml_backend_view_init(t);
            }
            else if (t->view_src != nullptr && t->buffer == nullptr)
            {
                status = ggml_backend_view_init(t);
            }
            if (status != GGML_STATUS_SUCCESS)
                return false;
        }
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

// Selects which Vulkan device ggml-vulkan initializes on (multi-GPU hosts, e.g.
// an iGPU next to a discrete GPU). Must be called before the first GGML op /
// TSGgml_IsBackendAvailable; once the backend singleton exists the device can
// no longer change, so a differing late call fails instead of silently
// binding to the wrong GPU.
TSG_EXPORT int TSGgml_SetVulkanDeviceIndex(int deviceIndex)
{
    clear_last_error();
    if (deviceIndex < 0)
    {
        set_last_error("Vulkan device index must be non-negative.");
        return 0;
    }

    if (g_backend != nullptr &&
        g_backend_type == BACKEND_TYPE_VULKAN &&
        g_vulkan_device_index.load(std::memory_order_acquire) != deviceIndex)
    {
        set_last_error("The ggml-vulkan backend was already initialized on a different device.");
        return 0;
    }

    g_vulkan_device_index.store(deviceIndex, std::memory_order_release);
    return 1;
}

TSG_EXPORT int TSGgml_GetVulkanDeviceCount()
{
    clear_last_error();
#if defined(GGML_USE_VULKAN)
    return ggml_backend_vk_get_device_count();
#else
    set_last_error("The ggml-vulkan backend is not available in this build.");
    return 0;
#endif
}

TSG_EXPORT int TSGgml_GetVulkanDeviceDescription(int deviceIndex, char* description, int descriptionSize)
{
    clear_last_error();
    if (description == nullptr || descriptionSize <= 0)
    {
        set_last_error("Invalid description buffer.");
        return 0;
    }
#if defined(GGML_USE_VULKAN)
    if (deviceIndex < 0 || deviceIndex >= ggml_backend_vk_get_device_count())
    {
        set_last_error("Vulkan device index " + std::to_string(deviceIndex) + " is out of range.");
        return 0;
    }
    ggml_backend_vk_get_device_description(deviceIndex, description, static_cast<size_t>(descriptionSize));
    return 1;
#else
    (void) deviceIndex;
    set_last_error("The ggml-vulkan backend is not available in this build.");
    return 0;
#endif
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

// Defined in ggml_ops_qwen_image.cpp; drops the persistent whole-model graphs whose
// resident weights live in the caches cleared below.
extern "C" void TSGgml_QwenImageResetForwardCache();

TSG_EXPORT void TSGgml_ClearHostBufferCache()
{
    // Drop any persistent whole-model graphs first: they bind weights resident by
    // GGUF pointer (shared via these caches), so freeing the caches below would leave
    // their captured graphs pointing at freed device memory.
    TSGgml_QwenImageResetForwardCache();

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
        g_device_copy_resident_bytes = 0;
        // The budget belongs to the model whose load configured it (model
        // unload clears this cache); don't let it leak onto the next model.
        g_device_copy_budget_bytes = 0;
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
        g_device_copy_resident_bytes = 0;
        g_device_copy_budget_bytes = 0;
    }

    // Release the reusable per-graph compute buffer + gallocr before the backend
    // they were allocated from is torn down.
    free_reuse_compute_buffer();
    free_reuse_gallocr();

    if (g_backend != nullptr)
    {
        ggml_backend_synchronize(g_backend);
        ggml_backend_free(g_backend);
        g_backend = nullptr;
    }
    g_pending_gpu_work.store(false, std::memory_order_release);
}

// Release the reusable per-graph compute buffer + gallocr WITHOUT tearing down the
// backend. The Qwen-Image denoise loop packs every DiT block into the persistent reuse
// gallocr; at high resolution that buffer grows to a few GB and would otherwise stay
// resident through the final VAE decode, competing with its im2col scratch for the
// (19 GB) Metal working set. The pipeline calls this after the denoise loop, before
// Vae.Decode, to hand that scratch back; the next graph re-creates the gallocr on demand.
TSG_EXPORT void TSGgml_ReleaseReuseComputeBuffers()
{
    free_reuse_compute_buffer();
    free_reuse_gallocr();
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

// Page-lock (cudaHostRegister) a host memory region so device uploads from it
// use the fast DMA path (~2x pageable copy throughput). Used for weight regions
// that stream to the GPU every step because they did not fit the residency
// budget. CUDA-only; returns 0 (no-op) on other backends or failure. The caller
// MUST unregister before the memory is unmapped/freed (mmap'd GGUF regions!).
TSG_EXPORT int TSGgml_RegisterPinnedHostBuffer(void* ptr, int64_t bytes)
{
#if defined(GGML_USE_CUDA)
    if (g_backend_type == BACKEND_TYPE_CUDA && g_backend != nullptr && ptr != nullptr && bytes > 0)
    {
        // ggml-cuda gates cudaHostRegister behind this env var (returns false
        // without it); our callers opt in explicitly, so satisfy the gate here.
        static const int s_env_once = []() {
#if defined(_WIN32)
            _putenv_s("GGML_CUDA_REGISTER_HOST", "1");
#else
            setenv("GGML_CUDA_REGISTER_HOST", "1", 0);
#endif
            return 0;
        }();
        (void)s_env_once;
        // NOTE: ggml registers with cudaHostRegisterReadOnly, so this is for
        // host->device upload sources (streamed weights) only — do not use it
        // for buffers the device writes back to.
        return ggml_backend_cuda_register_host_buffer(ptr, static_cast<std::size_t>(bytes)) ? 1 : 0;
    }
#else
    (void)ptr; (void)bytes;
#endif
    return 0;
}

TSG_EXPORT void TSGgml_UnregisterPinnedHostBuffer(void* ptr)
{
#if defined(GGML_USE_CUDA)
    if (g_backend_type == BACKEND_TYPE_CUDA && g_backend != nullptr && ptr != nullptr)
        ggml_backend_cuda_unregister_host_buffer(ptr);
#else
    (void)ptr;
#endif
}

// Set the byte ceiling for device-local copy residency (discrete-GPU weight
// caching). Zero (or negative) disables the cap. See ggml_ops_internal.h.
TSG_EXPORT void TSGgml_SetDeviceCopyBudget(int64_t bytes)
{
    std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
    g_device_copy_budget_bytes = bytes > 0 ? bytes : 0;
}

// Current free/total memory of the active backend's device, in bytes. For the
// CUDA backend this is physical VRAM; C# uses it to size weight preloading and
// the device-copy budget so VRAM is never oversubscribed. Returns 0 on failure
// (e.g. CPU backend), leaving free/total untouched.
TSG_EXPORT int TSGgml_DeviceMemoryInfo(int64_t* free_bytes, int64_t* total_bytes)
{
    if (!ensure_backend() || free_bytes == nullptr || total_bytes == nullptr)
        return 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
    if (dev == nullptr)
        return 0;
    std::size_t free_sz = 0, total_sz = 0;
    ggml_backend_dev_memory(dev, &free_sz, &total_sz);
    if (total_sz == 0)
        return 0;
    *free_bytes = static_cast<int64_t>(free_sz);
    *total_bytes = static_cast<int64_t>(total_sz);
    return 1;
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

// Diagnostic: total bytes of device-local COPY buffers currently resident in the
// host-buffer cache (CachedBufferMode::DeviceCopy). On Metal these are the
// activation/KV buffers that the per-op and fused kernels duplicate on-device
// (read-only weights are wrapped zero-copy as HostPtr and are excluded). Used by
// the diffusion multi-turn regression test to assert the prompt K/V device copies
// are reclaimed across blocks/turns instead of accumulating (the OOM regression).
TSG_EXPORT int64_t TSGgml_DeviceCopyCacheResidentBytes()
{
    std::int64_t total = 0;
    std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
    for (const auto& kv : g_host_buffer_cache)
    {
        if (kv.second.mode == CachedBufferMode::DeviceCopy)
            total += static_cast<std::int64_t>(kv.second.buffer_size);
    }
    return total;
}

// Diagnostic: the active backend device's memory accounting. On Metal `total`
// is recommendedMaxWorkingSetSize and `free` is total - currentAllocatedSize, so
// (total - free) is the bytes Metal currently has resident (weights + KV + every
// live compute/graph buffer). Lets a test see how close a run is to the working-
// set ceiling and which fix actually moves the needle. Returns 0 on success.
TSG_EXPORT int TSGgml_GetBackendMemory(int64_t* free_bytes, int64_t* total_bytes)
{
    if (free_bytes != nullptr) *free_bytes = 0;
    if (total_bytes != nullptr) *total_bytes = 0;
    if (!ensure_backend() || g_backend == nullptr)
        return 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
    if (dev == nullptr)
        return 0;
    std::size_t f = 0, t = 0;
    ggml_backend_dev_memory(dev, &f, &t);
    if (free_bytes != nullptr) *free_bytes = static_cast<std::int64_t>(f);
    if (total_bytes != nullptr) *total_bytes = static_cast<std::int64_t>(t);
    return 1;
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

// Merge a LoRA delta into a (possibly quantized) weight IN PLACE:
//   W[r, :] += scale * sum_k up[r, k] * down[k, :]      (r = 0..ne1-1 output rows)
// following stable-diffusion.cpp's apply path for quantized weights (lora.hpp
// build_lora_graph): dequantize to F32, add the delta, requantize back to the SAME
// type via ggml_quantize_chunk. `w` layout is the ggml row-major weight
// [ne1 rows x ne0 elements]; `up` is [ne1, rank] row-major, `down` is [rank, ne0]
// row-major (the safetensors lora_up / lora_down layouts).
// Returns 0 on success; <0 on validation/type errors (weight left untouched).
TSG_EXPORT int TSGgml_ApplyLoraDelta(void* w, int ggml_type, int64_t ne0, int64_t ne1,
                                     const float* up, const float* down, int32_t rank,
                                     float scale, int32_t n_threads)
{
    if (w == nullptr || up == nullptr || down == nullptr || ne0 <= 0 || ne1 <= 0 || rank <= 0)
        return -1;
    if (ggml_type < 0 || ggml_type >= GGML_TYPE_COUNT)
        return -1;
    const enum ggml_type t = static_cast<enum ggml_type>(ggml_type);
    const int64_t blck = ggml_blck_size(t);
    if (blck <= 0 || ne0 % blck != 0)
        return -2;
    const bool is_f32 = (t == GGML_TYPE_F32);
    const struct ggml_type_traits* traits = ggml_get_type_traits(t);
    if (!is_f32)
    {
        if (traits == nullptr || traits->to_float == nullptr)
            return -3;                                   // no dequant path for this type
        if (ggml_quantize_requires_imatrix(t))
            return -4;                                   // can't requantize without an imatrix
        ggml_quantize_init(t);                           // thread-safe to call up front
    }
    const size_t row_bytes = ggml_row_size(t, ne0);

    int nt = n_threads > 0 ? n_threads : (int)std::thread::hardware_concurrency();
    if (nt < 1) nt = 1;
    if ((int64_t)nt > ne1) nt = (int)ne1;

    std::atomic<int> err{0};
    auto worker = [&](int64_t r0, int64_t r1)
    {
        std::vector<float> buf((size_t)ne0);
        for (int64_t r = r0; r < r1 && err.load(std::memory_order_relaxed) == 0; r++)
        {
            uint8_t* wrow = static_cast<uint8_t*>(w) + (size_t)r * row_bytes;
            float* frow;
            if (is_f32)
                frow = reinterpret_cast<float*>(wrow);
            else
            {
                traits->to_float(wrow, buf.data(), ne0);
                frow = buf.data();
            }
            const float* uprow = up + (size_t)r * rank;
            for (int32_t k = 0; k < rank; k++)
            {
                const float a = scale * uprow[k];
                if (a == 0.0f) continue;
                const float* drow = down + (size_t)k * ne0;
                for (int64_t i = 0; i < ne0; i++)
                    frow[i] += a * drow[i];
            }
            if (!is_f32)
            {
                const size_t written = ggml_quantize_chunk(t, frow, wrow, 0, 1, ne0, nullptr);
                if (written != row_bytes)
                    err.store(-5, std::memory_order_relaxed);
            }
        }
    };

    if (nt == 1)
    {
        worker(0, ne1);
    }
    else
    {
        std::vector<std::thread> threads;
        threads.reserve(nt);
        const int64_t chunk = (ne1 + nt - 1) / nt;
        for (int i = 0; i < nt; i++)
        {
            int64_t r0 = (int64_t)i * chunk;
            int64_t r1 = std::min(ne1, r0 + chunk);
            if (r0 >= r1) break;
            threads.emplace_back(worker, r0, r1);
        }
        for (auto& th : threads) th.join();
    }
    return err.load();
}
