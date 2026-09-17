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
#include "ggml_ops_attention_alloc.h"

#if defined(TSG_GGML_USE_METAL)
#include "ggml-backend-impl.h"
#endif

#if defined(__APPLE__) || defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <sys/sysctl.h>
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

#include "ggml-impl.h"   // ggml_graph_view, for the node profiler

#include <chrono>
#include <cstdio>
#include <cstring>
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
    std::mutex g_backend_init_mutex;
    bool g_backend_initialized = false;
    int g_backend_type = 0;
    // The CPU backend's thread pools, one per backend instance ever created. They
    // used to be a function-local "process-lifetime" static, which was true while a
    // backend was created exactly once; TSGgml_RecreateBackend creates another, so
    // TSGgml_Shutdown frees them after the backend that used them is gone.
    std::vector<ggml_threadpool_t> g_cpu_pools;

    DeviceState g_device_states[TSG_MAX_DEVICES];
    std::atomic<int> g_device_count{1};
    thread_local int g_active_rank = 0;
    // Vulkan device index requested via TSGgml_SetVulkanDeviceIndex. Must be set
    // before the first backend init (create_backend_instance runs once under
    // g_backend_init_mutex); later calls with a different index fail. Indices are
    // positions in ggml-vulkan's enumeration order (after any
    // GGML_VK_VISIBLE_DEVICES filtering applied at process launch).
    std::atomic<int> g_vulkan_device_index{0};

    // The host-buffer / preload / offload / device-copy-budget state now lives
    // per rank in DeviceState (see ggml_ops_internal.h); the old global names
    // are macros onto the active slot.

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

    // ggml's DEBUG channel is where the CUDA backend reports whether a graph is
    // being CUDA-graph-captured ("CUDA graph warmup complete" / "... reset"),
    // which is not otherwise observable and is worth ~19 ms per replay on a
    // 3765-node graph under WDDM. Off by default because it is chatty.
    static bool ggml_debug_log_enabled()
    {
        static const bool v = []{
            const char* e = std::getenv("TS_GGML_LOG_DEBUG");
            return is_truthy_env(e);
        }();
        return v;
    }

    // ggml reports the interesting failures — a Metal command buffer that died
    // with kIOGPUCommandBufferCallbackErrorOutOfMemory, a backend that has
    // latched its sticky error state — only through this log callback, which used
    // to go to stderr and nowhere else. The op that then returns 0 says nothing
    // more than "graph execution failed", so the .NET exception named whichever
    // op happened to run next (an embedding get_rows, say) and the real cause was
    // visible only to whoever was watching the console. Keep the most recent
    // error text so a failing op can hand it to the caller.
    //
    // Process-global rather than thread_local like g_last_error: ggml logs from
    // whichever thread encodes or synchronizes the command buffer, so scoping the
    // capture to the failing thread would usually capture nothing.
    std::mutex g_ggml_error_log_mutex;
    // Errors since the last op succeeded — what set_last_error() appends.
    std::string g_ggml_error_log;
    // The same text, but never cleared: once the backend has failed, the op that
    // saw it has long since returned and its window is gone.
    std::string g_ggml_failure_log;
    constexpr std::size_t kGgmlErrorLogCap = 1024;
    std::atomic<std::uint64_t> g_ggml_error_count{0};
    std::atomic<bool> g_backend_compute_failed{false};
    // clear_last_error() runs on every successful op — hundreds per forward — and
    // the capture is empty on all but the failing path, so the lock is worth
    // skipping. Set only by capture_ggml_error, cleared only under the lock.
    std::atomic<bool> g_ggml_error_log_dirty{false};

    static void append_capped(std::string& target, const std::string& line)
    {
        if (target.size() >= kGgmlErrorLogCap)
            return;
        if (!target.empty())
            target += " | ";
        target += line;
    }

    static void capture_ggml_error(const char* text)
    {
        std::string line(text);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
            line.pop_back();
        if (line.empty())
            return;

        {
            std::lock_guard<std::mutex> lock(g_ggml_error_log_mutex);
            append_capped(g_ggml_error_log, line);
            append_capped(g_ggml_failure_log, line);
        }
        g_ggml_error_log_dirty.store(true, std::memory_order_release);

        // Bumped last so compute_graph()/sync_backend()'s before/after comparison
        // never observes a count for text it cannot yet read.
        g_ggml_error_count.fetch_add(1, std::memory_order_release);
    }

    static void filtered_ggml_log(enum ggml_log_level level, const char* text, void* user_data)
    {
        (void) user_data;
        if (level == GGML_LOG_LEVEL_DEBUG && !ggml_debug_log_enabled())
            return;
        if (level == GGML_LOG_LEVEL_ERROR && text != nullptr)
            capture_ggml_error(text);
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

        // Whatever ggml logged since the last op succeeded IS the cause of this
        // failure — append it rather than making the operator go find the console.
        if (!g_ggml_error_log_dirty.load(std::memory_order_acquire))
            return;

        std::lock_guard<std::mutex> lock(g_ggml_error_log_mutex);
        if (!g_ggml_error_log.empty())
        {
            g_last_error += " ggml: ";
            g_last_error += g_ggml_error_log;
        }
    }

    void report_load_refusal(const char* format, ...)
    {
        char stack_buffer[1024];
        va_list args;
        va_start(args, format);
        va_list copy;
        va_copy(copy, args);
        int needed = std::vsnprintf(stack_buffer, sizeof(stack_buffer), format, args);
        va_end(args);

        std::string message;
        if (needed < 0)
        {
            message = format;
        }
        else if ((size_t) needed < sizeof(stack_buffer))
        {
            message.assign(stack_buffer, (size_t) needed);
        }
        else
        {
            message.resize((size_t) needed + 1);
            std::vsnprintf(&message[0], message.size(), format, copy);
            message.resize((size_t) needed);
        }
        va_end(copy);

        std::fputs(message.c_str(), stderr);
        if (message.empty() || message.back() != '\n')
            std::fputc('\n', stderr);
        std::fflush(stderr);

        while (!message.empty() && (message.back() == '\n' || message.back() == '\r'))
            message.pop_back();
        set_last_error(message);
    }

    void clear_last_error()
    {
        g_last_error.clear();

        // An op that succeeded is the point past which older ggml chatter can no
        // longer explain a failure, so the capture window starts again here.
        if (!g_ggml_error_log_dirty.load(std::memory_order_acquire))
            return;

        std::lock_guard<std::mutex> lock(g_ggml_error_log_mutex);
        g_ggml_error_log.clear();
        g_ggml_error_log_dirty.store(false, std::memory_order_release);
    }

#if defined(GGML_USE_VULKAN)
    // ggml-vulkan's device query calls ggml_vk_instance_init(), which throws a
    // vk::SystemError when no usable driver is present. Only ggml_backend_vk_reg()
    // catches that; every other entry point (get_device_count, vk_init,
    // get_device_description) lets it escape. Escaping a C++ exception through our
    // extern "C" boundary into the .NET runtime means std::terminate and a core
    // dump, so a host that merely lacks a working ICD looked like a TensorSharp
    // crash. Funnel all Vulkan probing through here and turn the failure into an
    // ordinary "no devices" answer plus a diagnostic the managed side can surface.
    //
    // The most common cause on GPU containers is a missing libEGL.so.1: NVIDIA's
    // Vulkan ICD dlopens it from vk_icdGetInstanceProcAddr and reports no driver
    // when it is absent, which is why the hint names it.
    int vk_device_count_guarded()
    {
        try
        {
            return ggml_backend_vk_get_device_count();
        }
        catch (const std::exception& e)
        {
            set_last_error(
                std::string("Vulkan initialization failed: ") + e.what() +
                ". No usable Vulkan driver was found. On NVIDIA containers this is "
                "usually a missing GLVND EGL library -- install libegl1 (and libgl1 / "
                "libglvnd0) so the NVIDIA Vulkan ICD can load. Diagnose with "
                "VK_LOADER_DEBUG=all.");
            return 0;
        }
        catch (...)
        {
            set_last_error(
                "Vulkan initialization failed with an unknown error. No usable Vulkan "
                "driver was found. On NVIDIA containers this is usually a missing GLVND "
                "EGL library -- install libegl1 so the NVIDIA Vulkan ICD can load.");
            return 0;
        }
    }

    ggml_backend_t vk_init_guarded(int device_index)
    {
        try
        {
            return ggml_backend_vk_init(static_cast<size_t>(device_index));
        }
        catch (const std::exception& e)
        {
            set_last_error(std::string("ggml-vulkan backend initialization failed for device ") +
                std::to_string(device_index) + ": " + e.what());
            return nullptr;
        }
        catch (...)
        {
            set_last_error("ggml-vulkan backend initialization failed for device " +
                std::to_string(device_index) + " with an unknown error.");
            return nullptr;
        }
    }
#endif

    // --- VRAM allocation diagnostics (TS_GGML_LOG_VRAM=1) ---

    bool vram_log_enabled()
    {
        static const bool enabled = []{
            const char* e = std::getenv("TS_GGML_LOG_VRAM");
            return e != nullptr && (e[0] == '1' || e[0] == '2');
        }();
        return enabled;
    }

    bool vram_log_verbose()
    {
        static const bool enabled = []{
            const char* e = std::getenv("TS_GGML_LOG_VRAM");
            return e != nullptr && e[0] == '2';
        }();
        return enabled;
    }

    void vram_log_ctx_breakdown(const char* tag, ggml_context* ctx, int top_n)
    {
        if (!vram_log_verbose() || ctx == nullptr)
            return;

        // Only tensors that are still unbound get a slot from
        // ggml_backend_alloc_ctx_tensors; views alias their source and weights
        // already point at their cached device buffer.
        std::unordered_map<std::string, std::pair<int, std::size_t>> by_name;
        std::size_t total = 0;
        int count = 0;
        for (ggml_tensor* t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t))
        {
            if (t->data != nullptr || t->view_src != nullptr)
                continue;
            const std::size_t bytes = ggml_nbytes(t);
            total += bytes;
            ++count;
            const char* raw = ggml_get_name(t);
            std::string name = (raw != nullptr && raw[0] != '\0') ? raw : "(unnamed)";
            // Per-layer tensors are usually named "<role>_<layer>"; strip a
            // trailing numeric suffix so all 65 layers aggregate into one row.
            std::size_t cut = name.find_last_not_of("0123456789");
            if (cut != std::string::npos && cut + 1 < name.size() &&
                (name[cut] == '_' || name[cut] == '.' || name[cut] == '-'))
                name.resize(cut);
            auto& slot = by_name[name];
            slot.first += 1;
            slot.second += bytes;
        }

        std::vector<std::pair<std::string, std::pair<int, std::size_t>>> rows(by_name.begin(), by_name.end());
        std::sort(rows.begin(), rows.end(),
            [](const auto& a, const auto& b) { return a.second.second > b.second.second; });

        std::fprintf(stderr, "[TSVRAM] %s breakdown: %d unbound tensors, %.1f MB\n",
            tag, count, total / (1024.0 * 1024.0));
        const int limit = (top_n > 0 && top_n < static_cast<int>(rows.size())) ? top_n : static_cast<int>(rows.size());
        for (int i = 0; i < limit; i++)
            std::fprintf(stderr, "[TSVRAM]     %-40s x%-5d %9.1f MB\n",
                rows[i].first.c_str(), rows[i].second.first,
                rows[i].second.second / (1024.0 * 1024.0));
        std::fflush(stderr);
    }

    void vram_log(const char* tag, std::int64_t bytes)
    {
        if (!vram_log_enabled())
            return;
        std::size_t free_b = 0, total_b = 0;
        if (g_backend != nullptr)
        {
            ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
            if (dev != nullptr)
                ggml_backend_dev_memory(dev, &free_b, &total_b);
        }
        std::fprintf(stderr, "[TSVRAM] %-32s %9.1f MB | dev free %9.1f / %9.1f MB\n",
            tag, bytes / (1024.0 * 1024.0),
            free_b / (1024.0 * 1024.0), total_b / (1024.0 * 1024.0));
        std::fflush(stderr);
    }

    // --- Backend management ---

    // Enumerate the GPU devices a backend type can use, in ggml registration
    // order. Used both by TSGgml_GetGpuDeviceCount (so C# can size a TP group)
    // and by create_backend_instance_on_device.
    static std::vector<ggml_backend_dev_t> enumerate_gpu_devices(int backend_type)
    {
        std::vector<ggml_backend_dev_t> out;
#if defined(GGML_USE_CUDA)
        if (backend_type == BACKEND_TYPE_CUDA)
        {
            const size_t n = ggml_backend_dev_count();
            for (size_t i = 0; i < n; ++i)
            {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (d == nullptr)
                    continue;
                // GPU or IGPU: ggml-cuda classifies a device by
                // cudaDeviceProp.integrated, and some virtualized hosts
                // (observed: RunPod RTX PRO 6000, driver 595.91) report
                // integrated=1 for a discrete datacenter card. Filtering on
                // GPU alone then silently drops every CUDA device - and the
                // caller falls back to whatever the registry lists next
                // (ggml-vulkan), so a server asked for ggml_cuda ran on
                // Vulkan. The registry-name check below is the real gate.
                const enum ggml_backend_dev_type dt = ggml_backend_dev_type(d);
                if (dt != GGML_BACKEND_DEVICE_TYPE_GPU && dt != GGML_BACKEND_DEVICE_TYPE_IGPU)
                    continue;
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d);
                // Only ggml-cuda devices: a mixed CUDA+Vulkan build registers both.
                if (reg != nullptr && std::strcmp(ggml_backend_reg_name(reg), GGML_CUDA_NAME) != 0)
                    continue;
                out.push_back(d);
            }
        }
#endif
        (void) backend_type;
        return out;
    }

    int gpu_device_count(int backend_type)
    {
#if defined(GGML_USE_CUDA)
        if (backend_type == BACKEND_TYPE_CUDA)
            return static_cast<int>(enumerate_gpu_devices(backend_type).size());
#endif
#if defined(GGML_USE_VULKAN)
        if (backend_type == BACKEND_TYPE_VULKAN)
            return vk_device_count_guarded();
#endif
        (void) backend_type;
        return 1;
    }

    // Create a backend bound to a specific GPU ordinal. device_index < 0 keeps
    // the legacy "first available device" behaviour.
    ggml_backend_t create_backend_instance_on_device(int backend_type, int device_index)
    {
        if (device_index < 0)
            return create_backend_instance(backend_type);

        if (backend_type == BACKEND_TYPE_CUDA)
        {
#if defined(GGML_USE_CUDA)
            const auto devices = enumerate_gpu_devices(backend_type);
            if (device_index >= static_cast<int>(devices.size()))
            {
                set_last_error("CUDA device index " + std::to_string(device_index) +
                    " is out of range: " + std::to_string(devices.size()) + " device(s) available.");
                return nullptr;
            }
            ggml_backend_t backend = ggml_backend_dev_init(devices[device_index], nullptr);
            if (backend == nullptr)
                set_last_error("ggml-cuda backend initialization failed for device " + std::to_string(device_index) + ".");
            return backend;
#else
            set_last_error("The ggml-cuda backend is not available in this build.");
            return nullptr;
#endif
        }

        if (backend_type == BACKEND_TYPE_VULKAN)
        {
#if defined(GGML_USE_VULKAN)
            if (device_index >= vk_device_count_guarded())
            {
                set_last_error("Vulkan device index " + std::to_string(device_index) + " is out of range.");
                return nullptr;
            }
            return vk_init_guarded(device_index);
#else
            set_last_error("The ggml-vulkan backend is not available in this build.");
            return nullptr;
#endif
        }

        // CPU / Metal have a single logical device; anything past rank 0 would
        // just be a second view of the same hardware.
        if (device_index != 0)
        {
            set_last_error("The selected GGML backend exposes a single device; rank " +
                std::to_string(device_index) + " cannot be initialized.");
            return nullptr;
        }
        return create_backend_instance(backend_type);
    }

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
            {
                set_last_error("ggml-cpu backend initialization failed.");
                return backend;
            }
            // A bare ggml_backend_cpu_init() runs GGML_DEFAULT_N_THREADS (4) and
            // spawns a DISPOSABLE thread pool per graph_compute - on the per-op
            // path that is ~940 pool spawn/join cycles per decoded token, on 4 of
            // however many cores the machine has. Default to ALL physical cores
            // plus one persistent thread pool for the life of the backend.
            // llama.cpp defaults to P-cores only on Apple Silicon; measured on an
            // M5 Pro (6P+12E) that leaves 2x on the table for this workload -
            // Muse-Glimmer 30B IQ2_XXS decode is 3.8 tok/s at 6 threads and
            // 8.2 at 18 (llama.cpp itself moves 3.7 -> 7.9 when given -t 18,
            // and its prompt throughput DROPS with E-cores while ours rises).
            // TS_GGML_CPU_THREADS overrides the count.
            int cpu_threads = 0;
            if (const char* e = std::getenv("TS_GGML_CPU_THREADS"))
            {
                const int v = std::atoi(e);
                if (v > 0) cpu_threads = v;
            }
#if defined(__APPLE__)
            if (cpu_threads <= 0)
            {
                std::uint32_t cores = 0;
                std::size_t len = sizeof(cores);
                if (sysctlbyname("hw.physicalcpu", &cores, &len, nullptr, 0) == 0 && cores > 0)
                    cpu_threads = static_cast<int>(cores);
            }
#endif
            if (cpu_threads <= 0)
                cpu_threads = available_cpu_parallelism();
            ggml_backend_cpu_set_n_threads(backend, cpu_threads);
            ggml_threadpool_params tpp = ggml_threadpool_params_default(cpu_threads);
            if (ggml_threadpool_t pool = ggml_threadpool_new(&tpp))
            {
                g_cpu_pools.push_back(pool);
                ggml_backend_cpu_set_threadpool(backend, pool);
            }
            return backend;
        }

        if (backend_type == BACKEND_TYPE_CUDA)
        {
#if defined(GGML_USE_CUDA)
            // ggml_backend_dev_by_type(GPU) returns the registry's FIRST GPU
            // device of ANY backend. In a CUDA+Vulkan build that has picked
            // the ggml-vulkan device, so a server asked for ggml_cuda ran
            // every graph on Vulkan (observed on RTX PRO 6000: the CUDA
            // banner printed, the managed label said GgmlCuda, and the
            // executing backend name was Vulkan0). Use the same
            // registry-name-filtered enumeration the multi-GPU path uses.
            const auto cuda_devices = enumerate_gpu_devices(BACKEND_TYPE_CUDA);
            ggml_backend_dev_t device = cuda_devices.empty() ? nullptr : cuda_devices.front();
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
            const int device_count = vk_device_count_guarded();
            if (device_count <= 0)
            {
                // Keep the guard's diagnostic when it explained *why* the driver
                // is unusable; a bare "no device" would throw that away.
                if (g_last_error.empty())
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

            return vk_init_guarded(device_index);
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

        {
            // Once, exactly as std::call_once did -- including the part where an
            // initialize_backend that RETURNS having failed is not tried again, so a
            // machine with no usable device does not pay for the attempt on every op.
            // TSGgml_RecreateBackend is the only thing that puts this back.
            std::lock_guard<std::mutex> guard(g_backend_init_mutex);
            if (!g_backend_initialized)
            {
                initialize_backend();
                g_backend_initialized = true;
            }
        }
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

    bool invalidate_cached_buffer(void* data)
    {
        if (data == nullptr)
            return false;

        {
            std::lock_guard<std::mutex> lock(g_preloaded_buffer_cache_mutex);
            auto it = g_preloaded_buffer_cache.find(data);
            if (it != g_preloaded_buffer_cache.end())
            {
                ggml_backend_buffer_free(it->second.buffer);
                g_preloaded_buffer_cache.erase(it);
                return true;
            }
        }

        {
            std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
            auto it = g_host_buffer_cache.find(data);
            if (it == g_host_buffer_cache.end())
                return false;
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
        return true;
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

    // Probe-only lookup: an EXISTING DeviceCopy-mode cached buffer for `data`
    // covering exactly `bytes`. Never allocates, uploads, or evicts — callers
    // that only want to READ the current resident bytes (the qwen35 arena join)
    // use this so probing absent entries cannot create device copies.
    bool try_peek_cached_device_copy(const void* data, std::size_t bytes,
                                     ggml_backend_buffer_t& out_buffer, void*& out_addr)
    {
        out_buffer = nullptr;
        out_addr = nullptr;
        if (data == nullptr || bytes == 0)
            return false;
        std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
        auto it = g_host_buffer_cache.find(const_cast<void*>(data));
        if (it == g_host_buffer_cache.end() ||
            it->second.mode != CachedBufferMode::DeviceCopy ||
            it->second.bytes != bytes)
            return false;
        out_buffer = it->second.buffer;
        out_addr = ggml_backend_buffer_get_base(it->second.buffer);
        return out_addr != nullptr;
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
                // A preloaded weight losing its device copy means every later use
                // re-uploads it — a silent multi-hundred-MB per-call cost on a big
                // LM head. Make it visible rather than merely slow.
                if (vram_log_enabled())
                {
                    std::fprintf(stderr,
                        "[TSVRAM] preloaded weight dropped: cached %zu B / buffer %zu B, requested %zu B, need %zu B\n",
                        it->second.bytes, it->second.buffer_size, bytes, required_size);
                    std::fflush(stderr);
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

            // CONTRACT: out_needs_upload == true means "this buffer has never been
            // written". The entry is published in g_host_buffer_cache below BEFORE
            // any bytes reach the device, and CachedHostBuffer carries no "was
            // filled" bit — a later cache HIT therefore reports needs_upload ==
            // false unconditionally. So a caller that takes this branch MUST upload
            // before it can abandon the graph it is building: bailing out in between
            // (a VRAM guard, a gallocr failure) leaves a hot entry backing
            // uninitialised device memory that every later graph accepts as valid,
            // and reads of it are silent — freshly mapped VRAM is zeros, so the model
            // computes to a plausible finite answer that is simply wrong. Bind sites
            // that can abandon a built graph (WanBind::bind in ggml_ops_wan.cpp,
            // qi_fwd_build_graph's bind in ggml_ops_qwen_image.cpp) therefore fill
            // the tensor inline here rather than queueing it for a later loop.
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
            if (vram_log_enabled())
            {
                char tag[96];
                std::snprintf(tag, sizeof(tag), "devcopy(total=%.1fMB)",
                    g_device_copy_resident_bytes / (1024.0 * 1024.0));
                vram_log(tag, static_cast<std::int64_t>(ggml_backend_buffer_get_size(out_buffer)));
            }
            return true;
        }

        if (!try_get_host_ptr_buffer(backend, dev, data, bytes, true, out_buffer, unified_weight))
            return false;

        out_addr = data;
        return true;
    }

    // Mark a cached entry as having been through the backend's init_tensor, so
    // the next graph that binds the same weight can attach directly. Looks in
    // both caches because a weight can be in either (preloaded vs. bound on
    // first use); the flag rides along with the entry, so an eviction clears it.
    namespace
    {
        void mark_cached_buffer_initialized(void* data, ggml_backend_buffer_t buffer, std::size_t alloc_size)
        {
            {
                std::lock_guard<std::mutex> lock(g_preloaded_buffer_cache_mutex);
                auto it = g_preloaded_buffer_cache.find(data);
                if (it != g_preloaded_buffer_cache.end() && it->second.buffer == buffer)
                {
                    it->second.initialized_alloc_size = alloc_size;
                    return;
                }
            }
            std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
            auto it = g_host_buffer_cache.find(data);
            if (it != g_host_buffer_cache.end() && it->second.buffer == buffer)
                it->second.initialized_alloc_size = alloc_size;
        }

        bool cached_buffer_is_initialized(void* data, ggml_backend_buffer_t buffer, std::size_t alloc_size)
        {
            {
                std::lock_guard<std::mutex> lock(g_preloaded_buffer_cache_mutex);
                auto it = g_preloaded_buffer_cache.find(data);
                if (it != g_preloaded_buffer_cache.end() && it->second.buffer == buffer)
                    return it->second.initialized_alloc_size == alloc_size;
            }
            std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
            auto it = g_host_buffer_cache.find(data);
            return it != g_host_buffer_cache.end() && it->second.buffer == buffer
                && it->second.initialized_alloc_size == alloc_size;
        }
    }

    namespace
    {
        // Repeat bind: the weight is already resident, already initialised, and
        // the address it resolved to last time is recorded. Attaching is then two
        // assignments — which is what llama.cpp effectively does by binding its
        // weights once at load and never touching them again.
        //
        // This exists because the slow path is not slow for any one reason, it is
        // slow ~450 times: try_get_cacheable_tensor_buffer takes two mutexes and
        // asks the backend for an allocation size, then ggml_backend_tensor_alloc
        // runs the backend's init_tensor (a cudaMemset over the quant padding on
        // ggml-cuda). At ~17 us a weight that is 8 ms per prefill on a 30-layer
        // Gemma 4 — 13% of the call, and pure repetition.
        bool try_attach_bound_tensor(ggml_tensor* tensor, void* data, std::size_t bytes)
        {
            auto attach = [&](std::unordered_map<void*, CachedHostBuffer>& cache) {
                auto it = cache.find(data);
                if (it == cache.end() || it->second.bytes != bytes ||
                    it->second.bound_addr == nullptr || it->second.initialized_alloc_size == 0)
                    return false;
                tensor->buffer = it->second.buffer;
                tensor->data = it->second.bound_addr;
                return true;
            };
            {
                std::lock_guard<std::mutex> lock(g_preloaded_buffer_cache_mutex);
                if (attach(g_preloaded_buffer_cache)) return true;
            }
            std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
            return attach(g_host_buffer_cache);
        }

        void record_bound_addr(void* data, ggml_backend_buffer_t buffer, void* addr)
        {
            {
                std::lock_guard<std::mutex> lock(g_preloaded_buffer_cache_mutex);
                auto it = g_preloaded_buffer_cache.find(data);
                if (it != g_preloaded_buffer_cache.end() && it->second.buffer == buffer)
                {
                    it->second.bound_addr = addr;
                    return;
                }
            }
            std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
            auto it = g_host_buffer_cache.find(data);
            if (it != g_host_buffer_cache.end() && it->second.buffer == buffer)
                it->second.bound_addr = addr;
        }
    }

    bool try_bind_cached_tensor(
        ggml_backend_t backend, ggml_backend_dev_t dev,
        ggml_tensor* tensor, void* data, std::size_t bytes, bool& out_needs_upload,
        enum ggml_backend_buffer_usage usage)
    {
        out_needs_upload = false;

        // The overwhelmingly common case: this weight was bound by an earlier
        // graph and nothing has evicted it since.
        if (try_attach_bound_tensor(tensor, data, bytes))
            return true;

        ggml_backend_buffer_t buf = nullptr;
        void* addr = nullptr;
        if (!try_get_cacheable_tensor_buffer(backend, dev, tensor, data, bytes, buf, addr, out_needs_upload, usage))
            return false;

        const std::size_t alloc_size = ggml_backend_buffer_get_alloc_size(buf, tensor);

        // Already initialised at this alloc size (the entry just had no recorded
        // address yet, e.g. it was created by the weight preload): attach without
        // running init_tensor again.
        if (!out_needs_upload && cached_buffer_is_initialized(data, buf, alloc_size))
        {
            tensor->buffer = buf;
            tensor->data = addr;
            record_bound_addr(data, buf, addr);
            return true;
        }

        if (ggml_backend_tensor_alloc(buf, tensor, addr) != GGML_STATUS_SUCCESS)
        {
            invalidate_cached_buffer(data);
            return false;
        }
        mark_cached_buffer_initialized(data, buf, alloc_size);
        // Only a bind that needed no upload is safe to short-circuit next time;
        // one that did needs the upload_list entry the caller builds from
        // out_needs_upload.
        if (!out_needs_upload)
            record_bound_addr(data, buf, addr);
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
    //
    // Under tensor parallelism each rank drives its own backend, so the cached
    // buffer is per rank: a single shared slot would see the backend change on
    // every rank switch and free/realloc the buffer twice per layer.
    struct ReuseComputeSlot
    {
        std::mutex mutex;
        ggml_backend_buffer_t buf = nullptr;
        std::size_t size = 0;
        ggml_backend_t backend = nullptr;
    };
    static ReuseComputeSlot g_reuse_compute_slots[TSG_MAX_DEVICES];
#define g_reuse_compute_mutex   (g_reuse_compute_slots[::tsg::g_active_rank].mutex)
#define g_reuse_compute_buf     (g_reuse_compute_slots[::tsg::g_active_rank].buf)
#define g_reuse_compute_size    (g_reuse_compute_slots[::tsg::g_active_rank].size)
#define g_reuse_compute_backend (g_reuse_compute_slots[::tsg::g_active_rank].backend)

    // Persistent graph allocator for the large multi-token fused graphs (e.g. the
    // MTP MoE verify). Those used to ggml_gallocr_new()/ggml_gallocr_free() a
    // ~400 MB device buffer on EVERY call; on Metal that per-call alloc+free of a
    // large shared (vm_allocate) buffer fragments the device VM over hundreds of
    // verify steps until a contiguous allocation fails (OOM). A single gallocr
    // reused across calls grows its buffer once and keeps it, eliminating the
    // churn (and the per-call ~20 ms Metal allocation). Reset on backend swap.
    struct ReuseGallocrSlot
    {
        std::mutex mutex;
        ggml_gallocr_t gallocr = nullptr;
        ggml_backend_t backend = nullptr;
        std::size_t last_logged_size = 0;
    };
    static ReuseGallocrSlot g_reuse_gallocr_slots[TSG_MAX_DEVICES];
#define g_reuse_gallocr_mutex   (g_reuse_gallocr_slots[::tsg::g_active_rank].mutex)
#define g_reuse_gallocr         (g_reuse_gallocr_slots[::tsg::g_active_rank].gallocr)
#define g_reuse_gallocr_backend (g_reuse_gallocr_slots[::tsg::g_active_rank].backend)

    // A SECOND, independent set of slots for the MoE-offload streaming graphs.
    // Those are built and run *inside* a partially executed outer graph (the
    // host-MoE seam cuts the whole-model graph and evaluates the offloaded layer
    // between two slices), and ggml_gallocr_alloc_graph re-plans the allocator
    // it is handed for whatever graph it is given -- re-planning the outer
    // graph's allocator mid-pass moves the tensors that graph's remaining nodes
    // still point at. Symptom is not a wrong number but a hard
    // GGML_ASSERT(device >= 0 && device < info.device_count) once a relocated
    // tensor is read back through a stale buffer.
    static ReuseGallocrSlot g_moe_stream_gallocr_slots[TSG_MAX_DEVICES];

    bool alloc_graph_in_gallocr_slot(ggml_cgraph* graph, ReuseGallocrSlot* slots, const char* log_tag);

    void free_reuse_compute_buffer()
    {
        for (int r = 0; r < TSG_MAX_DEVICES; ++r)
        {
            auto& slot = g_reuse_compute_slots[r];
            std::lock_guard<std::mutex> lock(slot.mutex);
            if (slot.buf != nullptr)
            {
                ggml_backend_buffer_free(slot.buf);
                slot.buf = nullptr;
            }
            slot.size = 0;
            slot.backend = nullptr;
        }
    }

    void free_reuse_gallocr()
    {
        ReuseGallocrSlot* const all[] = { g_reuse_gallocr_slots, g_moe_stream_gallocr_slots };
        for (ReuseGallocrSlot* slots : all)
        {
            for (int r = 0; r < TSG_MAX_DEVICES; ++r)
            {
                auto& slot = slots[r];
                std::lock_guard<std::mutex> lock(slot.mutex);
                if (slot.gallocr != nullptr)
                {
                    ggml_gallocr_free(slot.gallocr);
                    slot.gallocr = nullptr;
                }
                slot.backend = nullptr;
                slot.last_logged_size = 0;
            }
        }
    }

    // Allocate `graph`'s intermediates into a persistent, reused gallocr (grown on
    // demand). Returns false if the gallocr could not be created/allocated, in
    // which case the caller should fall back to its own gallocr or per-op path.
    // The caller must NOT free the gallocr; it lives for the backend's lifetime.
    bool alloc_graph_reuse_gallocr(ggml_cgraph* graph)
    {
        return alloc_graph_in_gallocr_slot(graph, g_reuse_gallocr_slots, "reuse-gallocr");
    }

    bool alloc_graph_moe_stream_gallocr(ggml_cgraph* graph)
    {
        return alloc_graph_in_gallocr_slot(graph, g_moe_stream_gallocr_slots, "moe-stream-gallocr");
    }

    bool alloc_graph_in_gallocr_slot(ggml_cgraph* graph, ReuseGallocrSlot* slots, const char* log_tag)
    {
        // Escape hatch (shares the reuse-buffer toggle): TS_GGML_REUSE_COMPUTE_BUF=0
        // disables both so A/B testing can isolate the persistent allocators.
        static const bool s_disabled = []() {
            const char* e = std::getenv("TS_GGML_REUSE_COMPUTE_BUF");
            return e != nullptr && e[0] == '0';
        }();
        if (s_disabled || g_backend == nullptr || graph == nullptr || slots == nullptr)
            return false;

        // Deliberately does NOT reorder for Metal here, even though this is the one
        // place that would cover every whole-model kernel at once. It measured
        // slower on Gemma 4; see optimize_graph_for_metal below for the numbers.
        ReuseGallocrSlot& slot = slots[::tsg::g_active_rank];
        std::lock_guard<std::mutex> lock(slot.mutex);
        if (slot.backend != g_backend)
        {
            // Backend swapped (model reload). The old backend freed its buffers on
            // teardown, so drop the stale handle rather than freeing through it.
            slot.gallocr = nullptr;
            slot.backend = g_backend;
        }
        if (slot.gallocr == nullptr)
        {
            slot.gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));
            if (slot.gallocr == nullptr)
                return false;
        }
        // ggml_gallocr_alloc_graph reuses the existing buffer when the new graph
        // fits and grows (reallocates) it only when a larger graph appears.
        bool ok = ggml_gallocr_alloc_graph(slot.gallocr, graph);
        if (!ok)
        {
            // A FAILED alloc leaves the allocator POISONED, and because this one
            // is persistent the poison outlives the call. ggml_gallocr_reserve_n
            // first re-plans node_allocs/leaf_allocs for the new (larger) graph
            // and frees the old buffer, and only then tries to allocate the new
            // one; on OOM it returns false with galloc->buffers[i] == NULL while
            // the plan still claims every tensor is placed.
            //
            // The next call is the trap: model graphs have a FIXED topology
            // (node/leaf counts depend on the layer stack, not the token count),
            // so a shorter prompt yields the same node count with no-larger
            // tensors -> ggml_gallocr_needs_realloc() returns false -> the
            // reserve is skipped -> ggml_gallocr_init_tensor() dereferences the
            // NULL buffer. That is the 0xC0000005 / SIGSEGV of issue #113: one
            // OOM on a long prompt crashed the process on the NEXT request.
            //
            // There is no ggml API to inspect or repair that state, so drop the
            // allocator; the next call builds a fresh one and re-reserves from
            // scratch. The failed reserve already freed the buffer this owned,
            // so nothing extra is lost by throwing away the bookkeeping.
            ggml_gallocr_free(slot.gallocr);
            slot.gallocr = nullptr;
            return false;
        }
        if (vram_log_enabled())
        {
            const std::size_t size = ggml_gallocr_get_buffer_size(slot.gallocr, 0);
            if (size != slot.last_logged_size)
            {
                slot.last_logged_size = size;
                vram_log(log_tag, static_cast<std::int64_t>(size));
            }
        }
        return true;
    }

    // Depth rather than a bool: a builder can hold the guard across a nested
    // allocation (the MoE-offload streaming graph is built inside one).
    static thread_local int g_graph_reorder_suppress_depth = 0;

    SuppressGraphReorder::SuppressGraphReorder(bool active)
        : active_(active)
    {
        if (active_)
            ++g_graph_reorder_suppress_depth;
    }

    SuppressGraphReorder::~SuppressGraphReorder()
    {
        if (active_)
            --g_graph_reorder_suppress_depth;
    }

    // Kept an explicit per-kernel call rather than folded into the shared allocator,
    // because the reorder is a per-architecture trade rather than a free win. Wiring
    // it in for everyone does raise concurrency exactly as advertised — gemma-4-E4B
    // pp512 went from 44 of 1061 nodes encoded concurrently to 127, past llama.cpp's
    // 120 — and still measured SLOWER, reproducibly, with the A/B order alternated so
    // thermal drift could not fake it:
    //
    //   gemma-4-E4B  pp512  2281 -> 2254 t/s   tg128  46.0 -> 44.8 tok/s
    //   Qwen3.5-9B   pp512  1280 -> 1305 t/s   tg128  31.4 -> 31.8 tok/s
    //   Qwen3.6-A3B  pp512  1740 -> 1802 t/s   tg128  77.5 -> 83.9 tok/s
    //
    // Removing barriers is not free: the reordered schedule interleaves ops that were
    // adjacent, widening the live set the caches have to hold. So each kernel opts in
    // where it measures faster, instead of Gemma 4 paying to buy Qwen3.6 its 8%.
    void optimize_graph_for_metal(ggml_cgraph* graph)
    {
#if defined(TSG_GGML_USE_METAL)
        // A/B escape hatch. TS_METAL_GRAPH_OPTIMIZE=0 restores the unreordered order
        // for isolating a regression to it.
        static const bool s_enabled = []() {
            const char* e = std::getenv("TS_METAL_GRAPH_OPTIMIZE");
            return !(e != nullptr && e[0] == '0');
        }();
        if (!s_enabled)
            return;

        // This graph will be executed as ordered slices; see SuppressGraphReorder.
        if (g_graph_reorder_suppress_depth > 0)
            return;
        // Direct tsg::compute_graph() calls do not run the backend
        // optimizer. Match ggml's scheduler path for Metal, where this hook
        // reorders alias-aware nodes and applies supported graph fusions.
        // This must run before gallocr/context allocation because reordering
        // after lifetime planning can invalidate the allocator's alias plan.
        if (g_backend_type == BACKEND_TYPE_METAL &&
            g_backend != nullptr &&
            graph != nullptr &&
            g_backend->iface.graph_optimize != nullptr)
        {
            // graph_optimize also takes an allocation-dependency sink. A backend
            // that reorders across concurrent streams uses it to say "keep TENSOR
            // allocated until UNTIL has been computed", and ggml_backend_sched
            // honours that by inserting GGML_OP_NONE nodes into its graph copy
            // before it allocates. This path allocates with gallocr directly and
            // has nowhere to put such a node, so it cannot honour one.
            //
            // Calling from here is sound only because Metal's implementation is
            // GGML_UNUSED(params) and adds none. That is ggml's to change, and a
            // dropped dependency would surface as a tensor freed while still live —
            // wrong numbers rather than a failure. So the sink is real and it says
            // so, rather than being the null pointer that would turn the same
            // change into a crash inside the backend.
            ggml_backend_graph_optimize_params opt_params = {
                /* .add_alloc_dep = */ [](void*, ggml_tensor*, ggml_tensor*) {
                    static std::once_flag once;
                    std::call_once(once, []() {
                        std::fprintf(stderr,
                            "[TSGGML] Metal's graph_optimize asked for an allocation dependency, "
                            "which the direct-compute path cannot honour — reordered graphs on this "
                            "path are no longer trustworthy. Please report this.\n");
                        std::fflush(stderr);
                    });
                },
                /* .user_data     = */ nullptr,
            };
            g_backend->iface.graph_optimize(g_backend, graph, &opt_params);
        }
#else
        (void) graph;
#endif
    }

    bool alloc_ctx_tensors_reuse(ggml_context* ctx, ggml_cgraph* graph)
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

        const bool reuse_attention = graph != nullptr && g_backend_type == BACKEND_TYPE_METAL;
        const std::size_t needed = reuse_attention ? 0 : ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft);
        if (!reuse_attention && needed == 0)
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

        // The optional graph-aware planner supplies its actual packed size to
        // this same retained-buffer policy. Ordinary tensors still have unique
        // slots, and only completed attention workspaces may share addresses.
        const auto acquire_buffer = [&](std::size_t required_bytes) -> ggml_backend_buffer_t
        {
            if (required_bytes > max_size)
                return nullptr;
            if (g_reuse_compute_buf == nullptr || g_reuse_compute_size < required_bytes)
            {
                // Grow with slack rounded up to a 64 MiB boundary. The graph's
                // intermediate footprint creeps up by sub-MB amounts every decode step
                // (the attention scratch scales with the growing context), so allocating
                // exactly `required_bytes` reallocates the buffer on EVERY step. On Metal each
                // realloc frees+allocs a multi-hundred-MB shared (vm_allocate) buffer;
                // doing that hundreds of times fragments the device VM until a large
                // contiguous allocation (e.g. the MTP verify graph) can no longer be
                // satisfied -> kIOGPUCommandBufferCallbackErrorOutOfMemory even though
                // total free bytes remain. Rounding to 64 MiB makes the buffer grow in
                // rare, big steps and be reused unchanged across thousands of decodes.
                std::size_t alloc_size = required_bytes;
                const std::size_t slab = static_cast<std::size_t>(64) * 1024 * 1024;
                alloc_size = ((alloc_size + slab - 1) / slab) * slab;
                if (alloc_size > max_size) alloc_size = max_size; // never exceed a single buffer
                if (alloc_size < required_bytes) alloc_size = required_bytes;
                if (g_reuse_compute_buf != nullptr)
                    ggml_backend_buffer_free(g_reuse_compute_buf);
                g_reuse_compute_buf = ggml_backend_buft_alloc_buffer(buft, alloc_size);
                if (g_reuse_compute_buf == nullptr)
                {
                    g_reuse_compute_size = 0;
                    return nullptr;
                }
                g_reuse_compute_size = alloc_size;
                ggml_backend_buffer_set_usage(g_reuse_compute_buf, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
                if (vram_log_enabled())
                    vram_log("reuse-compute-buf(grew)", static_cast<std::int64_t>(alloc_size));
            }
            return g_reuse_compute_buf;
        };
        if (reuse_attention)
            return alloc_ctx_tensors_with_attention_reuse(ctx, graph, g_backend, acquire_buffer) != nullptr;
        if (acquire_buffer(needed) == nullptr)
            return false;

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
        tsg::sync_backend(g_backend);
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

    // --- Quantized-weight cache keys ---
    //
    // A preloaded quantized weight is identified to the bridge by its *cache
    // key*, not by its host pointer: once C# has pinned the weight it hands us
    // an opaque GCHandle so the device copy survives the host buffer being
    // released. Every lookup path is keyed on it.
    //
    // The hazard is the miss path. If a lookup ever fails — the entry was
    // evicted, or the buffer was rebuilt at a different size — the caller falls
    // back to "upload the weight from `data`", and `data` is the GCHandle. That
    // dereferences an address that is not weight memory and segfaults inside
    // cudaMemcpyAsync.
    //
    // So record what each key actually stands for at preload time and resolve
    // it back before any upload. Registration is the only thing that makes an
    // upload from a key safe, and it makes the miss path merely slow (a
    // re-upload) instead of fatal.
    std::mutex g_cache_key_mutex;
    std::unordered_map<const void*, const void*> g_cache_key_host_data;

    void register_cache_key(const void* cache_key, const void* host_data)
    {
        if (cache_key == nullptr || host_data == nullptr || cache_key == host_data)
            return;
        std::lock_guard<std::mutex> lock(g_cache_key_mutex);
        g_cache_key_host_data[cache_key] = host_data;
    }

    void forget_cache_keys()
    {
        std::lock_guard<std::mutex> lock(g_cache_key_mutex);
        g_cache_key_host_data.clear();
    }

    // Number of uploads that had to be redirected — i.e. preloaded weights whose
    // device copy was not found. Each one is a silent re-upload of a whole
    // weight, so a non-zero count is a performance signal as well as the thing
    // that used to be a crash. Reported under TS_GGML_LOG_VRAM=1.
    std::atomic<std::int64_t> g_cache_key_redirects{0};

    const void* resolve_upload_source(const void* data)
    {
        {
            std::lock_guard<std::mutex> lock(g_cache_key_mutex);
            // Fast path: nothing registered (no preloads yet, CPU backend, ...).
            if (g_cache_key_host_data.empty())
                return data;
            auto it = g_cache_key_host_data.find(data);
            if (it == g_cache_key_host_data.end())
                return data;
            data = it->second;
        }

        const std::int64_t n = g_cache_key_redirects.fetch_add(1) + 1;
        if (vram_log_enabled() && (n == 1 || (n % 1000) == 0))
        {
            std::fprintf(stderr,
                "[TSVRAM] quantized weight upload fell back to host bytes (%lld so far) — "
                "its preloaded device copy was not found\n", static_cast<long long>(n));
            std::fflush(stderr);
        }
        return data;
    }

    void upload_binding(const TensorBinding& binding, const void* data, std::size_t size)
    {
        // A quantized weight arrives here as its cache key when its device copy
        // was expected to be resident; map it back to the real bytes.
        data = resolve_upload_source(data);

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

    // ------------------------------------------------------------------
    // Whole-model graph profiler (TS_GGML_NODE_PROFILE) — see the header.
    // ------------------------------------------------------------------
    bool graph_node_profile_enabled()
    {
        static const bool on = []{
            const char* e = std::getenv("TS_GGML_NODE_PROFILE");
            return e != nullptr && e[0] != '0';
        }();
        return on;
    }

    namespace
    {
        struct NodeProfileBucket
        {
            double us = 0.0;
            std::int64_t calls = 0;
            std::string shape;   // widest shape seen, for the per-op detail line
            double shape_us = 0.0;
        };

        struct NodeProfileState
        {
            std::mutex mu;
            std::unordered_map<std::string, NodeProfileBucket> buckets;
            double total_us = 0.0;
            std::int64_t graphs = 0;
            int nodes = 0;
        };

        NodeProfileState& node_profile_state()
        {
            static NodeProfileState s;
            return s;
        }

        int node_profile_every()
        {
            static const int every = []{
                const char* e = std::getenv("TS_GGML_NODE_PROFILE_EVERY");
                int v = (e != nullptr) ? std::atoi(e) : 0;
                return v > 0 ? v : 64;
            }();
            return every;
        }

        std::string node_shape_label(const ggml_tensor* t)
        {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "[%lld,%lld,%lld]",
                          static_cast<long long>(t->ne[0]),
                          static_cast<long long>(t->ne[1]),
                          static_cast<long long>(t->ne[2]));
            return std::string(buf);
        }
    }


    // True when the process-global GGML backend is ggml-metal ("MTL0", "MTL1", ...).
    static bool backend_is_metal()
    {
        const char* name = g_backend != nullptr ? ggml_backend_name(g_backend) : nullptr;
        return name != nullptr && std::strncmp(name, "MTL", 3) == 0;
    }

    // Wan VAE convolutions on MPSGraph (tsg_metal_mps_conv.mm). ggml lowers conv2d
    // to im2col + mul_mat, which is 74.6% of a VAE decode and moves 9x the input
    // before the GEMM starts; Apple's tuned convolution runs the same shapes 6-14x
    // faster and reaches the matrix units WITHOUT ggml's mul_mm kernel, whose Metal 4
    // tensor path corrupts this graph. TS_VAE_MPS_CONV=0 opts out.
    #if defined(TSG_GGML_USE_METAL)
    extern "C" bool tsg_mps_conv2d_available(void);
    extern "C" bool tsg_mps_conv2d(const void* w, int wIsF16, int kw, int kh, int ic, int oc,
                                   const float* x, int W, int H, int T,
                                   int stride, int pad,
                                   float* dst, int OW, int OH);
    extern "C" void tsg_mps_conv2d_release(void);
    #endif

    #if defined(TSG_HAVE_CUDNN)
    // The CUDA twin (tsg_cuda_cudnn_conv.cu). On an L4 the im2col lowering is 46% of
    // a VAE decode against only 24.5% for the GEMM, so cuBLAS speed does not help --
    // the materialisation itself is the cost. TS_VAE_CUDNN_CONV=0 opts out.
    extern "C" bool tsg_cudnn_conv2d_available(void);
    extern "C" bool tsg_cudnn_conv2d(const void* w, int wIsF16, int kw, int kh, int ic, int oc,
                                     const void* x, int W, int H, int T,
                                     int stride, int pad,
                                     void* dst, int OW, int OH);
    extern "C" void tsg_cudnn_conv2d_release(void);
    #endif

    // True when the process-global GGML backend is ggml-cuda.
    static bool backend_is_cuda()
    {
        const char* name = g_backend != nullptr ? ggml_backend_name(g_backend) : nullptr;
        return name != nullptr && std::strncmp(name, "CUDA", 4) == 0;
    }

    // True when a VAE should emit single CONV_2D nodes for a vendor
    // convolution library to execute instead of ggml's im2col + mul_mat lowering.
    bool fast_conv_enabled()
    {
    #if defined(TSG_GGML_USE_METAL)
        if (backend_is_metal())
        {
            static const bool on = []{
                const char* e = std::getenv("TS_VAE_MPS_CONV");
                if (e != nullptr && e[0] == '0') return false;
                return tsg_mps_conv2d_available();
            }();
            return on;
        }
    #endif
    #if defined(TSG_HAVE_CUDNN)
        if (backend_is_cuda())
        {
            // OPT-IN on CUDA (TS_VAE_CUDNN_CONV=1), unlike the Metal/MPS route above.
            //
            // cuDNN convolves directly and so removes ggml's im2col lowering AND the
            // band tiling that exists to bound its scratch. That is a win on a short
            // clip and a large loss on a long one, because this route executes each
            // convolution OUTSIDE the graph: run_conv_fast is bracketed by a
            // ggml_backend_synchronize and a stream sync, so the cost is
            // (convolutions x host round trip), and the convolution count scales with
            // the temporal chunk count while the graph submission does not.
            //
            // Measured, RTX 3080 Laptop 16 GB, 1088x832 isolated decode
            // (benchmarks/WanVideoBench vae-decode ... cuda):
            //     17 frames  (5 chunks):   33.8 s cuDNN  vs  39.8 s ggml   (1.18x WIN)
            //     121 frames (31 chunks): 1327.9 s cuDNN vs 294.4 s ggml   (4.51x LOSS)
            // Outputs agree (mean 0.435643 vs 0.435646; the diffusers oracle passes at
            // 80.2 dB either way) -- this is purely a scheduling cost. ggml's path
            // scales linearly with frames; this one does not, because ~197
            // convolutions per temporal chunk become ~12 200 host synchronisations at
            // 121 frames, on a device that is already paging at that size.
            //
            // Turn it on for short clips, for the Qwen-Image VAE (single frame), or on
            // a card with enough headroom that the decode is not memory bound.
            static const bool on = []{
                const char* e = std::getenv("TS_VAE_CUDNN_CONV");
                if (e == nullptr || e[0] == '0') return false;
                return tsg_cudnn_conv2d_available();
            }();
            return on;
        }
    #endif
        return false;
    }


    // Run one CONV_2D node through the backend's convolution library (MPSGraph on
    // Metal, cuDNN on CUDA). ggml's kernel is [KW,KH,IC,OC] F16 and
    // the activation [W,H,IC,T] F32, which are MPS OIHW / NCHW byte for byte, so the
    // operands go across as they lie. Returns false for anything unsupported, and
    // the caller then lets ggml execute the node normally.
    static bool run_conv_fast(ggml_tensor* node)
    {
        ggml_tensor* kern = node->src[0];
        ggml_tensor* act  = node->src[1];
        if (kern == nullptr || act == nullptr) return false;
        const bool kernF16 = kern->type == GGML_TYPE_F16;
        if ((!kernF16 && kern->type != GGML_TYPE_F32) || act->type != GGML_TYPE_F32 || node->type != GGML_TYPE_F32)
            return false;
        if (!ggml_is_contiguous(kern) || !ggml_is_contiguous(act) || !ggml_is_contiguous(node))
            return false;

        const int32_t* op = (const int32_t*) node->op_params;
        const int s0 = op[0], s1 = op[1], p0 = op[2], p1 = op[3], d0 = op[4], d1 = op[5];
        if (s0 != s1 || p0 != p1 || d0 != 1 || d1 != 1) return false;   // square stride/pad only

        const int kw = (int) kern->ne[0], kh = (int) kern->ne[1];
        const int ic = (int) kern->ne[2], oc = (int) kern->ne[3];
        const int W  = (int) act->ne[0],  H  = (int) act->ne[1], T = (int) act->ne[3];
        const int OW = (int) node->ne[0], OH = (int) node->ne[1];
        if ((int) act->ne[2] != ic || (int) node->ne[2] != oc || (int) node->ne[3] != T) return false;

    #if defined(TSG_HAVE_CUDNN)
        if (backend_is_cuda())
        {
            // ggml tensor data is already a device pointer here, so cuDNN reads and
            // writes ggml's own buffers -- no staging, no copies.
            return tsg_cudnn_conv2d(kern->data, kernF16 ? 1 : 0, kw, kh, ic, oc,
                                    act->data, W, H, T, s0, p0,
                                    node->data, OW, OH);
        }
    #endif
    #if defined(TSG_GGML_USE_METAL)
        if (backend_is_metal())
        {
            std::vector<std::uint8_t> kbuf((std::size_t) ggml_nbytes(kern));
            std::vector<float> ah((std::size_t) ggml_nelements(act));
            std::vector<float> oh((std::size_t) ggml_nelements(node));
            ggml_backend_tensor_get(kern, kbuf.data(), 0, kbuf.size());
            ggml_backend_tensor_get(act,  ah.data(), 0, ah.size() * sizeof(float));
            if (!tsg_mps_conv2d(kbuf.data(), kernF16 ? 1 : 0, kw, kh, ic, oc,
                                ah.data(), W, H, T, s0, p0,
                                oh.data(), OW, OH))
                return false;
            ggml_backend_tensor_set(node, oh.data(), 0, oh.size() * sizeof(float));
            return true;
        }
    #endif
        (void) kw; (void) kh; (void) ic; (void) oc; (void) W; (void) H; (void) T; (void) OW; (void) OH;
        return false;
    }


    // Execute a graph with every CONV_2D node handed to the platform convolution
    // library (MPSGraph on Metal, cuDNN on CUDA) and the stretches between them
    // left to ggml. Running node ranges through ggml_graph_view is the same
    // mechanism graph_compute_profiled uses, so it is safe against gallocr's
    // buffer reuse.
    ggml_status graph_compute_fast_conv(ggml_cgraph* graph, const char* tag)
    {
        const int n = ggml_graph_n_nodes(graph);
        // TS_GGML_NODE_PROFILE also covers this path: it does not go through
        // graph_compute_profiled (the convolutions are executed outside the graph),
        // so without this the vendor-conv route is the one shape that cannot be
        // profiled — exactly the one whose split between library and graph time
        // decides whether the vendor library is worth it.
        const bool prof = graph_node_profile_enabled();
        double convUs = 0.0, graphUs = 0.0;
        int convs = 0, fallbacks = 0;
        auto now = [] { return std::chrono::steady_clock::now(); };
        const auto tAll = now();

        int from = 0;
        for (int i = 0; i < n; i++)
        {
            ggml_tensor* node = ggml_graph_node(graph, i);
            if (node->op != GGML_OP_CONV_2D) continue;

            if (i > from)
            {
                ggml_cgraph view = ggml_graph_view(graph, from, i);
                const auto t0 = now();
                const ggml_status st = tsg::compute_graph(g_backend, &view);
                if (st != GGML_STATUS_SUCCESS) return st;
                if (prof) { tsg::sync_backend(g_backend); graphUs += std::chrono::duration<double, std::micro>(now() - t0).count(); }
            }
            tsg::sync_backend(g_backend);
            const auto t1 = now();
            if (!run_conv_fast(node))
            {
                // Unsupported shape: let ggml run just this node.
                ggml_cgraph one = ggml_graph_view(graph, i, i + 1);
                const ggml_status st = tsg::compute_graph(g_backend, &one);
                if (st != GGML_STATUS_SUCCESS) return st;
                if (prof) { tsg::sync_backend(g_backend); fallbacks++; }
            }
            if (prof) { convUs += std::chrono::duration<double, std::micro>(now() - t1).count(); convs++; }
            from = i + 1;
        }
        if (from < n)
        {
            ggml_cgraph view = ggml_graph_view(graph, from, n);
            const auto t0 = now();
            const ggml_status st = tsg::compute_graph(g_backend, &view);
            if (st != GGML_STATUS_SUCCESS) return st;
            if (prof) { tsg::sync_backend(g_backend); graphUs += std::chrono::duration<double, std::micro>(now() - t0).count(); }
        }
        if (prof)
        {
            const double totalUs = std::chrono::duration<double, std::micro>(now() - tAll).count();
            std::printf("[fast-conv] %s: %d nodes, %d convolutions (%d fell back to ggml) | "
                        "vendor conv %.0f ms (%.1f%%) | rest of graph %.0f ms (%.1f%%) | total %.0f ms\n",
                        tag != nullptr ? tag : "graph", n, convs, fallbacks,
                        convUs / 1000.0, 100.0 * convUs / totalUs,
                        graphUs / 1000.0, 100.0 * graphUs / totalUs,
                        totalUs / 1000.0);
            std::fflush(stdout);
        }
        return GGML_STATUS_SUCCESS;
    }

    ggml_status graph_compute_profiled(ggml_backend_t backend, ggml_cgraph* graph, const char* tag)
    {
        // TS_GGML_NODE_PROFILE_TAG=<substring> profiles only the graphs whose tag
        // contains it (e.g. "verify"), so a decode-dominated run does not bury
        // the graph of interest; the others run unprofiled.
        {
            static const char* want = std::getenv("TS_GGML_NODE_PROFILE_TAG");
            if (want != nullptr && want[0] != '\0' && graph_node_profile_enabled()
                && (tag == nullptr || std::strstr(tag, want) == nullptr))
                return tsg::compute_graph(backend, graph);
        }
        // TSG debug: identify the backend object actually executing graphs.
        { static int tsg_dbg_n = 0;
          if (tsg_dbg_n < 4) { tsg_dbg_n++;
              FILE* f = fopen("/tmp/tsg_backend_id.log", "a");
              if (f) { fprintf(f, "backend=%p name=%s compute=%p tag=%s n=%d\n",
                               (void*)backend, ggml_backend_name(backend),
                               (void*)nullptr, /* iface opaque here */
                               tag ? tag : "?", graph ? ggml_graph_n_nodes(graph) : -1);
                       fclose(f); } } }
        if (!graph_node_profile_enabled())
            return tsg::compute_graph(backend, graph);

        const int n = ggml_graph_n_nodes(graph);
        std::vector<double> per_node(static_cast<std::size_t>(n), 0.0);

        tsg::sync_backend(backend);
        const auto t_start = std::chrono::steady_clock::now();
        for (int i = 0; i < n; i++)
        {
            ggml_cgraph view = ggml_graph_view(graph, i, i + 1);
            const auto t0 = std::chrono::steady_clock::now();
            ggml_status st = tsg::compute_graph(backend, &view);
            tsg::sync_backend(backend);
            const auto t1 = std::chrono::steady_clock::now();
            if (st != GGML_STATUS_SUCCESS)
                return st;
            per_node[static_cast<std::size_t>(i)] =
                std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        const auto t_end = std::chrono::steady_clock::now();

        NodeProfileState& s = node_profile_state();
        std::lock_guard<std::mutex> lock(s.mu);
        for (int i = 0; i < n; i++)
        {
            ggml_tensor* node = ggml_graph_node(graph, i);
            const double us = per_node[static_cast<std::size_t>(i)];
            NodeProfileBucket& b = s.buckets[ggml_op_name(node->op)];
            b.us += us;
            b.calls++;
            if (us > b.shape_us) { b.shape_us = us; b.shape = node_shape_label(node); }
        }
        s.total_us += std::chrono::duration<double, std::micro>(t_end - t_start).count();
        s.graphs++;
        s.nodes = n;

        if (s.graphs % node_profile_every() == 0)
        {
            std::vector<std::pair<std::string, NodeProfileBucket>> sorted(s.buckets.begin(), s.buckets.end());
            std::sort(sorted.begin(), sorted.end(),
                      [](const auto& a, const auto& b) { return a.second.us > b.second.us; });
            std::fprintf(stderr, "[node-profile] %s: %lld graphs x %d nodes, %.3f ms/graph (profiled, includes a synchronize per node)\n",
                        tag != nullptr ? tag : "graph",
                        static_cast<long long>(s.graphs), s.nodes,
                        s.total_us / 1000.0 / static_cast<double>(s.graphs));
            for (std::size_t i = 0; i < sorted.size() && i < 16; i++)
            {
                const NodeProfileBucket& b = sorted[i].second;
                std::fprintf(stderr, "    %-18s %8.3f ms/graph  %6.1f%%  %6lld nodes/graph  worst %s\n",
                            sorted[i].first.c_str(),
                            b.us / 1000.0 / static_cast<double>(s.graphs),
                            100.0 * b.us / s.total_us,
                            static_cast<long long>(b.calls / s.graphs),
                            b.shape.c_str());
            }
            std::fflush(stdout);
        }
        return GGML_STATUS_SUCCESS;
    }

    // ------------------------------------------------------------------
    // Whole-model kernel phase timer (TS_GGML_PHASE_TIMING) - see the header.
    // ------------------------------------------------------------------
    bool phase_timing_enabled()
    {
        static const bool on = []{
            const char* e = std::getenv("TS_GGML_PHASE_TIMING");
            return e != nullptr && e[0] != '0';
        }();
        return on;
    }

    double PhaseTimer::now()
    {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void phase_timing_report(const char* tag, const char* const* phases, const double* ms, int count)
    {
        double total = 0.0;
        for (int i = 0; i < count; i++) total += ms[i];
        std::string line = "[phase] ";
        line += (tag != nullptr ? tag : "kernel");
        char buf[64];
        std::snprintf(buf, sizeof(buf), " total=%.2fms", total);
        line += buf;
        for (int i = 0; i < count; i++)
        {
            std::snprintf(buf, sizeof(buf), " %s=%.2f", phases[i], ms[i]);
            line += buf;
        }
        std::printf("%s\n", line.c_str());
        std::fflush(stdout);
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

// Whether a GPU command buffer has failed at any point in this process, together
// with what ggml said about it. Latched by sync_backend() (ggml_ops_internal.h)
// rather than reported by the op that hit it, because the op that drains a dead
// command buffer returns SUCCESS — ggml_backend_synchronize has no way to say
// otherwise — and only the NEXT graph fails.
//
// It does not clear by itself. On Metal the backend latches its own has_error and
// recovers only by being recreated -- which TSGgml_RecreateBackend now does, and is
// the only thing that clears this flag. Callers that cannot rebuild (every host
// except the iOS app, so far) should still read it as terminal.
TSG_EXPORT int TSGgml_HasBackendFailure()
{
    return g_backend_compute_failed.load(std::memory_order_acquire) ? 1 : 0;
}

TSG_EXPORT const char* TSGgml_GetBackendFailureText()
{
    // Copied into a thread_local so the caller reads a stable buffer while the
    // log keeps growing behind the mutex.
    static thread_local std::string snapshot;
    {
        std::lock_guard<std::mutex> lock(g_ggml_error_log_mutex);
        snapshot = g_ggml_failure_log;
    }
    return snapshot.c_str();
}

// Graph builds that ran an attention as explicit ops because the backend has no
// flash-attention kernel for its shape (ggml_ops_flash_attn_guard.h). Lets a
// test prove the shape it drives really took the fallback.
TSG_EXPORT int64_t TSGgml_FlashAttnFallbackCount()
{
    return static_cast<int64_t>(tsg_flash_attn_fallback_count());
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
    return vk_device_count_guarded();
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
    if (deviceIndex < 0 || deviceIndex >= vk_device_count_guarded())
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
// Defined in ggml_ops_wan.cpp; same contract for the Wan DiT persistent graphs.
extern "C" void TSGgml_WanResetForwardCache();

// Tensor-parallel graphs held across calls, defined in their own kernels.
// TSGgml_Shutdown releases them while the backends are still alive.
extern "C" void TSGgml_Gemma4ReleaseVerifyTpGraphs();
extern "C" void TSGgml_Qwen35ReleaseAttentionTpGraphs();
extern "C" void TSGgml_Qwen35GdnDropTpGraphs();
extern "C" void TSGgml_ReleaseFusedFfnTpGraphs();
extern "C" void TSGgml_ReleaseFusedMatmulAddTpGraphs();
extern "C" void TSGgml_Gemma4MoEReleaseVerifyTpGraphs();
extern "C" void TSGgml_Gemma4MoEResetDecodeCache();
extern "C" void TSGgml_Gemma4ResetDecodeCache();
extern "C" void TSGgml_Qwen35ReleaseVerifyTpGraphs();
extern "C" void TSGgml_Qwen35ReleaseVerifyGraphsPreserveState();
extern "C" void TSGgml_Qwen35ResetDecodeCache();
extern "C" void TSGgml_Qwen35ResetBatchedDecodeCache();
extern "C" void TSGgml_Qwen35ResetVerifyCache();
extern "C" void TSGgml_Qwen35ResetVerifyCacheForHostPointer(const void* host_ptr);
extern "C" void TSGgml_Qwen3ResetDecodeCache();
extern "C" void TSGgml_Gemma4ResetBatchedDecodeCache();
extern "C" void TSGgml_Gemma4ResetMoEBatchedDecodeCache();
extern "C" void TSGgml_GptOssResetDecodeCache();
extern "C" void TSGgml_GptOssResetBatchedDecodeCache();
extern "C" void TSGgml_Qwen35ArenaResetBatchedDecodeCache();
extern "C" void TSGgml_Qwen4ExpArenaResetBatchedDecodeCache();
namespace tsg_q35arena { void on_drop(const void* host_ptr); }
extern "C" void TSGgml_GptOssInvalidateKvCache(const void* kCacheData, const void* vCacheData);
extern "C" void TSGgml_MuseGlimmerResetDecodeCache();
extern "C" void TSGgml_DFlashResetCaches();
extern "C" void TSGgml_QwenImageResetForwardCache();
extern "C" void TSGgml_WanResetForwardCache();

TSG_EXPORT void TSGgml_ClearHostBufferCache()
{
    // The slot-stable arena pools bind resident weight buffers this wipe is
    // about to free; their captured graphs must not survive it.
    TSGgml_GptOssResetBatchedDecodeCache();
    TSGgml_Qwen35ArenaResetBatchedDecodeCache();
    TSGgml_Qwen4ExpArenaResetBatchedDecodeCache();
    // Drop any persistent whole-model graphs first: they bind weights resident by
    // GGUF pointer (shared via these caches), so freeing the caches below would leave
    // their captured graphs pointing at freed device memory.
    TSGgml_QwenImageResetForwardCache();
    TSGgml_WanResetForwardCache();
    TSGgml_Qwen35ResetDecodeCache();
    TSGgml_Qwen3ResetDecodeCache();
    TSGgml_Qwen35ResetBatchedDecodeCache();
    // A process-global host-weight eviction must retire every verify graph/TP
    // plan, but another live Qwen35 model may still own the only current copy of
    // its recurrent state. Preserve those owner-private state buffers so its next
    // call can rebuild safely.
    TSGgml_Qwen35ReleaseVerifyGraphsPreserveState();
    TSGgml_Qwen35ReleaseAttentionTpGraphs();
    TSGgml_Qwen35GdnDropTpGraphs();
    // The vendor convolution library holds a handle, its engine tables and a
    // workspace on the device. The VAE ENCODER runs before the DiT, so leaving them
    // resident charges the whole denoise for memory only the decode needs.
#if defined(TSG_HAVE_CUDNN)
    tsg_cudnn_conv2d_release();
#endif
    forget_cache_keys();

    // Every rank owns its own device copies; clear all of them.
    for (int r = 0; r < tsg::g_device_count.load(std::memory_order_acquire); ++r)
    {
        tsg::ScopedRank rank(r);
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
// One teardown at a time. TSGgml_Shutdown is reached from three places that can
// overlap: AgentAppHost.Dispose, the ProcessExit hook the app installs, and
// TSGgml_RecreateBackend on a recovery. Two of them inside ggml_backend_free on the
// same context at once is a double free, and "the user quit the app while it was
// rebuilding the engine" is an ordinary way to get there. Recursive, because the
// recreate holds it across its call to the shutdown.
static std::recursive_mutex g_teardown_mutex;

TSG_EXPORT void TSGgml_Shutdown()
{
    std::lock_guard<std::recursive_mutex> teardown(g_teardown_mutex);
    // Tear the TP communicator down first: it holds NCCL communicators and
    // pinned staging buffers that reference every rank's backend.
    tp_comm_free();
    TSGgml_Qwen35ResetDecodeCache();
    TSGgml_Qwen3ResetDecodeCache();
    TSGgml_Qwen35ResetBatchedDecodeCache();
    TSGgml_Qwen35ReleaseVerifyTpGraphs();
    forget_cache_keys();

    const int ranks = tsg::g_device_count.load(std::memory_order_acquire);
    for (int r = 0; r < ranks; ++r)
    {
        tsg::ScopedRank rank(r);
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
    }

    // Release the reusable per-graph compute buffer + gallocr before the backend
    // they were allocated from is torn down. Both free every rank's slot.
    free_reuse_compute_buffer();
    free_reuse_gallocr();
    // Tensor-parallel graphs parked between "build" and "execute", plus the
    // persistent per-rank decode graphs. They own backend buffers, and letting
    // static destructors free them after the CUDA driver has shut down aborts
    // the process ("CUDA error: driver shutting down").
    TSGgml_Gemma4ReleaseVerifyTpGraphs();
    TSGgml_Qwen35ReleaseAttentionTpGraphs();
    TSGgml_Qwen35GdnDropTpGraphs();
    TSGgml_ReleaseFusedFfnTpGraphs();
    TSGgml_ReleaseFusedMatmulAddTpGraphs();
    TSGgml_Gemma4MoEReleaseVerifyTpGraphs();
    TSGgml_Gemma4MoEResetDecodeCache();
    TSGgml_Gemma4ResetDecodeCache();
    TSGgml_Gemma4ResetBatchedDecodeCache();
    TSGgml_Gemma4ResetMoEBatchedDecodeCache();
    TSGgml_Qwen35ResetVerifyCache();
    // GPT-OSS keeps per-layer device-resident K/V windows (tsg_gptoss::kv_*)
    // alive for the whole session; nothing else drops them, so on Metal their
    // MTLBuffers were still registered in the device residency set when the
    // ggml_metal_device static destructor ran, tripping
    // GGML_ASSERT([rsets->data count] == 0) and aborting the process at exit
    // (SIGABRT / exit code 134) after a perfectly good generation. Passing
    // (null, null) drops every window.
    TSGgml_GptOssResetDecodeCache();
    TSGgml_GptOssResetBatchedDecodeCache();
    TSGgml_Qwen35ArenaResetBatchedDecodeCache();
    TSGgml_Qwen4ExpArenaResetBatchedDecodeCache();
    TSGgml_GptOssInvalidateKvCache(nullptr, nullptr);
    // Same contract for the other whole-model graph caches: each parks a ggml
    // context + backend buffer that must be released before the backend is.
    TSGgml_MuseGlimmerResetDecodeCache();
    TSGgml_DFlashResetCaches();
    TSGgml_QwenImageResetForwardCache();
    TSGgml_WanResetForwardCache();
    // Release the calling thread's cached prefill-attention sessions while the
    // CUDA driver is still alive; leaving them to thread_local destructors
    // aborts the process on exit ("CUDA error: driver shutting down").
    free_prefill_attn_sessions();
    // MoE CPU offload keeps a persistent ggml CPU backend (and its worker thread
    // pool) for the host-side expert matmuls. It is independent of g_backend, so
    // it has to be released explicitly or the threads outlive the shutdown.
    tsg::moe_ffn_host_release();

    for (int r = 0; r < ranks; ++r)
    {
        tsg::ScopedRank rank(r);
        if (g_backend != nullptr)
        {
            tsg::sync_backend(g_backend);
            ggml_backend_free(g_backend);
            g_backend = nullptr;
        }
        tsg::dev(r).device_index = -1;
    }
    // After the backend: a CPU backend still references its pool until it is freed.
    for (ggml_threadpool_t pool : tsg::g_cpu_pools)
        ggml_threadpool_free(pool);
    tsg::g_cpu_pools.clear();
    tsg::g_device_count.store(1, std::memory_order_release);
    g_pending_gpu_work.store(false, std::memory_order_release);
}

// Throw the GPU backend away and build it again, in this process.
//
// The failure this exists for belongs to iOS. An app that is not frontmost may not
// submit work to the GPU, and a command buffer committed a moment after the user
// swipes away comes back with
// kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted ("Insufficient
// Permission (to submit GPU work from background)"). ggml-metal's reaction is to
// latch has_error, and its own comment says the backend must be recreated to clear
// it -- so one badly-timed submission did not cost a token, it cost the model for the
// rest of the process. Every message after it failed too, and the only cure was
// force-quitting the app. On a phone that is the whole app.
//
// Reloading the weights was not enough and it took a device to show why: g_backend is
// a process global that a model load never touches, so the "repaired" engine was the
// same poisoned one with new weights in it. This is the missing half.
//
// The caller must release the loaded model FIRST. Its tensors live in buffers this
// frees, and freeing them twice is a crash rather than an error. The order that works
// is: unload the model, recreate the backend, load the model again.
//
// Returns 1 when a working backend is standing afterwards.
TSG_EXPORT int TSGgml_RecreateBackend()
{
    std::lock_guard<std::recursive_mutex> teardown(g_teardown_mutex);
    clear_last_error();

    const int backend_type = tsg::g_backend_type;
    if (backend_type == 0)
    {
        // Nothing was ever initialised, so there is nothing to repair and the next op
        // will build one anyway.
        return 1;
    }

    // The full teardown, reused rather than reimplemented: it drains every command
    // buffer, releases the caches that hold MTLBuffer wrappers, and frees the backend
    // in the order the Metal device's deleter asserts on.
    TSGgml_Shutdown();

    {
        std::lock_guard<std::mutex> guard(tsg::g_backend_init_mutex);
        tsg::g_backend_initialized = false;
    }
    // Shutdown leaves the TYPE alone on purpose; the app is not switching backends,
    // it is replacing the one it has with an identical, healthy one.
    tsg::g_backend_type = backend_type;

    // Cleared only here. Anything latched belonged to the backend that has just been
    // freed, and keeping it would make the new one look broken from its first graph.
    tsg::g_backend_compute_failed.store(false, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(tsg::g_ggml_error_log_mutex);
        tsg::g_ggml_failure_log.clear();
    }

    if (!tsg::ensure_backend(backend_type))
        return 0;
    return 1;
}

// Release the reusable per-graph compute buffer + gallocr WITHOUT tearing down the
// backend. The Qwen-Image denoise loop packs every DiT block into the persistent reuse
// gallocr; at high resolution that buffer grows to a few GB and would otherwise stay
// resident through the final VAE decode, competing with its im2col scratch for the
// (19 GB) Metal working set. The pipeline calls this after the denoise loop, before
// Vae.Decode, to hand that scratch back; the next graph re-creates the gallocr on demand.
TSG_EXPORT void TSGgml_ReleaseReuseComputeBuffers()
{
    // Drain first. Under async compute the last graph's command buffer is still
    // reading the very scratch this frees, and Metal commits that read whenever
    // it gets round to it - long after the free returned. The barrier is a
    // single atomic when nothing was deferred.
    host_read_barrier();
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
    for (int r = 0; r < tsg::g_device_count.load(std::memory_order_acquire); ++r)
    {
        tsg::ScopedRank rank(r);
        std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
        g_offloadable_keys.insert(key);
    }
}

// Set the byte ceiling for offloadable cache residency. Zero (or negative)
// disables eviction (registered entries still participate in the LRU but
// nothing is freed).
TSG_EXPORT void TSGgml_SetOffloadableBudget(int64_t bytes)
{
    // Per-rank budget: each GPU holds its own slice of the expert weights, so
    // the caller's ceiling applies to every rank independently.
    for (int r = 0; r < tsg::g_device_count.load(std::memory_order_acquire); ++r)
    {
        tsg::ScopedRank rank(r);
        std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
        g_offloadable_budget = bytes > 0 ? bytes : 0;
        offloadable_evict_to_budget_locked();
    }
}

// Clear the offloadable registry, LRU, and byte accounting. Does NOT touch
// the underlying CachedHostBuffer entries — they remain reachable via
// g_host_buffer_cache and will be freed by TSGgml_ClearHostBufferCache or
// when the process exits.
TSG_EXPORT void TSGgml_ClearOffloadableState()
{
    for (int r = 0; r < tsg::g_device_count.load(std::memory_order_acquire); ++r)
    {
        tsg::ScopedRank rank(r);
        std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
        g_offloadable_keys.clear();
        g_offloadable_lru.clear();
        g_offloadable_lru_map.clear();
        g_offloadable_resident_bytes = 0;
        g_offloadable_budget = 0;
    }
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
    for (int r = 0; r < tsg::g_device_count.load(std::memory_order_acquire); ++r)
    {
        tsg::ScopedRank rank(r);
        std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
        g_device_copy_budget_bytes = bytes > 0 ? bytes : 0;
    }
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
    // The same host pointer can be resident on several ranks (a replicated
    // weight, or an activation the TP forward round-robins); drop all of them.
    bool freedDeviceCopy = false;
    for (int r = 0; r < tsg::g_device_count.load(std::memory_order_acquire); ++r)
    {
        tsg::ScopedRank rank(r);
        freedDeviceCopy |= invalidate_cached_buffer(ptr);
    }

    // The GPT-OSS attention kernel keeps its own device-resident copy of the KV
    // cache keyed by the same host pointer. Every caller that invalidates a
    // tensor's device copy (KV truncate, snapshot inject, cache reset) means
    // "the host bytes changed underneath you", which is exactly when that copy
    // has to go too — hooking it here keeps the two caches from disagreeing
    // without every call site having to know about both.
    TSGgml_GptOssInvalidateKvCache(ptr, nullptr);

    // Same contract for the qwen35 arena: this host pointer's bytes changed (or
    // are being freed) behind the kernels — its arena slot, if any, is stale.
    tsg_q35arena::on_drop(ptr);
    tsg_q4earena::on_drop(ptr);

    // Same argument, one level up: the persistent whole-model graphs bake the
    // buffer we just freed into their nodes, so replaying one after this point is
    // a use-after-free. ggml-vulkan catches it as
    // "GGML_ASSERT(buffer != nullptr)" inside ggml_vk_tensor_subbuffer (the freed
    // ggml_backend_vk_buffer_context reads back a null dev_buffer) and aborts;
    // ggml-cuda reads the stale allocation instead and silently computes on
    // freed memory. Reproduced by any multi-turn chat: turn 2 truncates the KV
    // cache, which invalidates its host buffers, and the very next decode
    // replays the turn-1 graph.
    //
    // Gated on freedDeviceCopy: hot paths call this for pointers that have no
    // device copy at all (Qwen3.5's gated-delta-net invalidates its conv/delta
    // state every decode step), and dropping every graph there would rebuild the
    // whole model graph per token — measured -13% to -48% decode before this
    // guard. When a buffer really was freed, invalidation happens at most once per
    // turn (KV truncate / reset / snapshot inject) and each cache simply rebuilds
    // on its next call.
    if (!freedDeviceCopy)
        return;

    TSGgml_Gemma4ResetDecodeCache();
    TSGgml_Gemma4ResetBatchedDecodeCache();
    TSGgml_Gemma4ResetMoEBatchedDecodeCache();
    TSGgml_Gemma4MoEResetDecodeCache();
    TSGgml_GptOssResetDecodeCache();
    TSGgml_GptOssResetBatchedDecodeCache();
    TSGgml_Qwen35ArenaResetBatchedDecodeCache();
    TSGgml_Qwen4ExpArenaResetBatchedDecodeCache();
    TSGgml_Qwen35ResetDecodeCache();
    TSGgml_Qwen35ResetBatchedDecodeCache();
    TSGgml_Qwen35ResetVerifyCacheForHostPointer(ptr);
    TSGgml_Qwen3ResetDecodeCache();
}

TSG_EXPORT int TSGgml_SyncHostBuffer(void* ptr, size_t size)
{
    bool any = false;
    for (int r = 0; r < tsg::g_device_count.load(std::memory_order_acquire); ++r)
    {
        tsg::ScopedRank rank(r);
        any |= sync_cached_buffer_to_host(ptr, size);
    }
    if (any)
    {
        clear_last_error();
        return 1;
    }
    set_last_error("Failed to synchronize cached GGML device buffer back to host memory.");
    return 0;
}

// Range variants of TSGgml_SyncHostBuffer / re-upload, for a caller that edits a
// FEW rows of a large cached tensor on the host - Gemma 4 restoring the sliding-
// window rows a rejected speculative verify evicted. The whole-buffer pair
// (sync everything down, invalidate, re-upload everything on the next bind)
// moved the entire K/V cache for a 16 KB edit; these move only the ranges. A
// tensor with no device copy (a zero-copy host wrap, or a CPU backend) needs
// nothing: its host memory IS what the device reads.
static bool cached_buffer_ranges(void* base, const std::int64_t* offsets, const std::int64_t* lengths, int count, bool to_host)
{
    if (base == nullptr || count <= 0)
        return true;

    ggml_backend_buffer_t buffer = nullptr;
    tsg::CachedBufferMode mode = tsg::CachedBufferMode::HostPtr;
    std::size_t cached_bytes = 0;
    {
        std::lock_guard<std::mutex> lock(g_host_buffer_cache_mutex);
        auto it = g_host_buffer_cache.find(base);
        if (it == g_host_buffer_cache.end())
            return true;
        buffer = it->second.buffer;
        mode = it->second.mode;
        cached_bytes = it->second.bytes;
    }
    if (mode != tsg::CachedBufferMode::DeviceCopy || buffer == nullptr)
        return true;

    for (int i = 0; i < count; ++i)
    {
        if (offsets[i] < 0 || lengths[i] < 0 ||
            static_cast<std::size_t>(offsets[i] + lengths[i]) > cached_bytes)
        {
            // The cached copy belongs to a previous, smaller occupant of this host
            // address (a K/V resize); the next bind rebuilds it from host memory.
            return true;
        }
    }

    tsg::PooledContextHandle context;
    if (!context.init(64 * 1024))
        return false;
    ggml_tensor* tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_I8, static_cast<std::int64_t>(cached_bytes));
    if (tensor == nullptr)
        return false;
    void* addr = ggml_backend_buffer_get_base(buffer);
    if (addr == nullptr)
        return false;
    if (ggml_backend_tensor_alloc(buffer, tensor, addr) != GGML_STATUS_SUCCESS)
        return false;

    // Pending GPU work may still be writing (or reading) this buffer; drain it
    // before touching the bytes either way.
    tsg::host_read_barrier();
    for (int i = 0; i < count; ++i)
    {
        if (lengths[i] == 0)
            continue;
        char* host = static_cast<char*>(base) + offsets[i];
        if (to_host)
            ggml_backend_tensor_get(tensor, host, static_cast<std::size_t>(offsets[i]), static_cast<std::size_t>(lengths[i]));
        else
            ggml_backend_tensor_set(tensor, host, static_cast<std::size_t>(offsets[i]), static_cast<std::size_t>(lengths[i]));
    }
    tsg::sync_backend(g_backend);
    return true;
}

TSG_EXPORT int TSGgml_SyncHostBufferRanges(void* ptr, const std::int64_t* offsets, const std::int64_t* lengths, int count)
{
    bool ok = true;
    for (int r = 0; r < tsg::g_device_count.load(std::memory_order_acquire); ++r)
    {
        tsg::ScopedRank rank(r);
        ok &= cached_buffer_ranges(ptr, offsets, lengths, count, /*to_host=*/true);
    }
    if (ok)
    {
        clear_last_error();
        return 1;
    }
    set_last_error("Failed to synchronize a range of a cached GGML device buffer back to host memory.");
    return 0;
}

TSG_EXPORT int TSGgml_UploadHostBufferRanges(void* ptr, const std::int64_t* offsets, const std::int64_t* lengths, int count)
{
    bool ok = true;
    for (int r = 0; r < tsg::g_device_count.load(std::memory_order_acquire); ++r)
    {
        tsg::ScopedRank rank(r);
        ok &= cached_buffer_ranges(ptr, offsets, lengths, count, /*to_host=*/false);
    }
    if (ok)
    {
        clear_last_error();
        return 1;
    }
    set_last_error("Failed to upload a range of host memory into a cached GGML device buffer.");
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

// Returns 1 if the active backend's device is an integrated GPU (unified-memory
// iGPU: Intel UHD / AMD APU via ggml-vulkan, Tegra via ggml-cuda), else 0. ggml
// reports these as GGML_BACKEND_DEVICE_TYPE_IGPU (see ggml-vulkan device_type()).
// The managed startup warmup uses this to skip the heavy multi-token prefill
// warmup on memory-bandwidth-bound integrated GPUs, where a 2048-token fused
// prefill takes minutes and makes the server look hung during initialization.
TSG_EXPORT int TSGgml_IsActiveDeviceIntegrated()
{
    if (!ensure_backend() || g_backend == nullptr)
        return 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
    if (dev == nullptr)
        return 0;
    return get_device_static_props(dev).type == GGML_BACKEND_DEVICE_TYPE_IGPU ? 1 : 0;
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
            tsg::sync_backend(g_backend);
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

        // Register before any early return: from here on the managed side may
        // hand this key to an op, and every path that could upload from it must
        // be able to resolve it back to the real weight bytes.
        register_cache_key(cache_key, host_data);

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
            // A ggml tensor must live in ONE backend buffer, and some devices cap
            // a single buffer well below total VRAM (ggml-vulkan rejects anything
            // above the driver's maxBufferSize; WSL's dzn layer caps it under
            // 3 GB, which e.g. Gemma E4B's Q8_0 per_layer_token_embd exceeds).
            // When the failed request is larger than the buffer type's advertised
            // max size, report "too large to preload" (2) so the managed side
            // keeps the host copy and serves the weight through its host-gather
            // fallback instead of failing the whole model load. Failures at or
            // below the advertised max stay hard errors (genuine OOM).
            if (alloc_size > ggml_backend_buft_get_max_size(buft))
            {
                clear_last_error();
                return 2;
            }
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
        tsg::sync_backend(g_backend);

        {
            std::lock_guard<std::mutex> lock(g_preloaded_buffer_cache_mutex);
            g_preloaded_buffer_cache[cache_key] = {
                buffer, bytes,
                ggml_backend_buffer_get_size(buffer),
                CachedBufferMode::DeviceCopy
            };
        }


        if (vram_log_enabled())
        {
            static std::atomic<std::int64_t> s_preload_total{0};
            const std::int64_t total = s_preload_total.fetch_add(
                static_cast<std::int64_t>(alloc_size)) + static_cast<std::int64_t>(alloc_size);
            char tag[96];
            std::snprintf(tag, sizeof(tag), "preload-weight(total=%.1fMB)", total / (1024.0 * 1024.0));
            vram_log(tag, static_cast<std::int64_t>(alloc_size));
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
