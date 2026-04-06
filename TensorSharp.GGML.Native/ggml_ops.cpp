// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-metal.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

// GGML context memory pool: reuse mem_buffers to avoid per-op allocation overhead
namespace ggml_pool
{
    constexpr std::size_t k_pool_buffer_size = 32 * 1024 * 1024;  // 32 MB, covers larger ops and batched matmul
    constexpr int k_pool_initial_count = 8;
    constexpr int k_pool_max_count = 32;

    struct PoolEntry
    {
        void* ptr = nullptr;
        std::size_t size = 0;
    };

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

    static PoolEntry acquire(std::size_t required_size)
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

    static void release(PoolEntry e)
    {
        if (e.ptr == nullptr)
            return;
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        if (g_pool.size() < static_cast<std::size_t>(k_pool_max_count))
            g_pool.push_back(e);
        else
            pool_free(e.ptr);
    }

    static void ensure_initial_pool()
    {
        std::lock_guard<std::mutex> lock(g_pool_mutex);
        while (g_pool.size() < static_cast<std::size_t>(k_pool_initial_count))
        {
            void* ptr = pool_alloc(k_pool_buffer_size);
            if (ptr == nullptr)
                break;
            g_pool.push_back({ ptr, k_pool_buffer_size });
        }
    }
}

#if defined(__clang__) || defined(__GNUC__)
#define TSG_EXPORT extern "C" __attribute__((visibility("default")))
#else
#define TSG_EXPORT extern "C"
#endif

namespace
{
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

    constexpr int TSG_DTYPE_F32 = 0;
    constexpr int TSG_DTYPE_I32 = 3;

    enum class UnaryOpCode : int
    {
        Neg = 1,
        Exp = 2,
        Log = 3,
        Sqrt = 4,
        Relu = 5,
        Sigmoid = 6,
        Tanh = 7,
        SiLU = 8,
        Step = 9,
        Abs = 10,
        Sign = 11,
        GELU = 12,
    };

    enum class FusedActMulOpCode : int
    {
        SiLUMul = 1,
        GELUMul = 2,
        SigmoidMul = 3,
    };

    enum class BinaryTensorOpCode : int
    {
        Add = 1,
        Sub = 2,
        Mul = 3,
        Div = 4,
    };

    enum class BinaryScalarOpCode : int
    {
        Add = 1,
        Sub = 2,
        ReverseSub = 3,
        Mul = 4,
        Div = 5,
        ReverseDiv = 6,
    };

    enum class ActivationGradOpCode : int
    {
        Relu = 1,
        Sigmoid = 2,
        Tanh = 3,
        SiLU = 4,
    };

    enum class NormOpCode : int
    {
        LayerNorm = 1,
        RmsNorm = 2,
    };

    enum class ReductionOpCode : int
    {
        Sum = 1,
        Mean = 2,
    };

    enum class IndexReductionOpCode : int
    {
        Argmin = 1,
        Argmax = 2,
    };

    struct TensorBinding
    {
        ggml_tensor* storage = nullptr;
        ggml_tensor* tensor = nullptr;
        std::size_t raw_bytes = 0;
    };

    // For zero-copy path: binding + buffer that must stay alive
    struct HostPtrBinding
    {
        TensorBinding binding;
        ggml_backend_buffer_t buffer = nullptr;
    };

    thread_local std::string g_last_error;
    std::once_flag g_backend_init_once;
    ggml_backend_t g_backend = nullptr;
    int g_backend_type = 0;

    struct CachedHostBuffer {
        ggml_backend_buffer_t buffer;
        std::size_t bytes;
    };
    std::unordered_map<void*, CachedHostBuffer> g_host_buffer_cache;

    void set_last_error(const std::string& message)
    {
        g_last_error = message;
    }

    void clear_last_error()
    {
        g_last_error.clear();
    }

    constexpr int BACKEND_TYPE_METAL = 1;
    constexpr int BACKEND_TYPE_CPU = 2;

    void initialize_backend()
    {
        clear_last_error();

        if (g_backend_type == BACKEND_TYPE_METAL)
        {
            g_backend = ggml_backend_metal_init();
            if (g_backend == nullptr)
            {
                set_last_error("ggml-metal backend initialization failed.");
                return;
            }
        }
        else if (g_backend_type == BACKEND_TYPE_CPU)
        {
            g_backend = ggml_backend_cpu_init();
            if (g_backend == nullptr)
            {
                set_last_error("ggml-cpu backend initialization failed.");
                return;
            }
        }
        else
        {
            set_last_error("Unknown GGML backend type requested.");
            return;
        }

        ggml_pool::ensure_initial_pool();
    }

    bool ensure_backend(int backend_type)
    {
        if (backend_type != BACKEND_TYPE_METAL && backend_type != BACKEND_TYPE_CPU)
        {
            set_last_error("Invalid GGML backend type.");
            return false;
        }

        if (g_backend_type == 0)
        {
            g_backend_type = backend_type;
        }
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

    bool backend_supports_op(ggml_tensor* op)
    {
        return op != nullptr && g_backend != nullptr && ggml_backend_supports_op(g_backend, op);
    }

    struct ContextHandle
    {
        ggml_context* value = nullptr;

        explicit ContextHandle(ggml_context* ctx)
            : value(ctx)
        {
        }

        ~ContextHandle()
        {
            if (value != nullptr)
            {
                ggml_free(value);
            }
        }

        ContextHandle(const ContextHandle&) = delete;
        ContextHandle& operator=(const ContextHandle&) = delete;
    };

    // Pooled context: uses memory pool for ggml context buffer, returns buffer to pool on destruction
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

        explicit BufferHandle(ggml_backend_buffer_t buffer)
            : value(buffer)
        {
        }

        ~BufferHandle()
        {
            if (value != nullptr)
            {
                ggml_backend_buffer_free(value);
            }
        }

        BufferHandle(const BufferHandle&) = delete;
        BufferHandle& operator=(const BufferHandle&) = delete;

        BufferHandle(BufferHandle&& other) noexcept
            : value(other.value)
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

    template <std::size_t N>
    bool is_non_overlapping_fast_to_slow(const std::array<int, N>& sizes, const std::array<int, N>& strides)
    {
        std::int64_t required_stride = 1;
        for (std::size_t i = 0; i < N; ++i)
        {
            if (sizes[i] <= 0 || strides[i] < 0)
            {
                return false;
            }

            if (sizes[i] == 1)
            {
                continue;
            }

            if (strides[i] < required_stride)
            {
                return false;
            }

            required_stride = static_cast<std::int64_t>(strides[i]) * sizes[i];
        }

        return true;
    }

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

    std::size_t logical_bytes(const TensorView2DDesc& desc)
    {
        return static_cast<std::size_t>(desc.dim0) * desc.dim1 * sizeof(float);
    }

    std::size_t logical_bytes(const TensorView3DDesc& desc)
    {
        return static_cast<std::size_t>(desc.dim0) * desc.dim1 * desc.dim2 * sizeof(float);
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

    std::size_t logical_bytes(const TensorView4DDesc& desc)
    {
        return static_cast<std::size_t>(desc.ne0) * desc.ne1 * desc.ne2 * desc.ne3 * sizeof(float);
    }

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
            {
                output[i] = static_cast<std::int32_t>(raw[i]);
            }
            return true;
        }

        set_last_error(std::string("Unsupported element type for ") + name + '.');
        return false;
    }

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
            is_non_overlapping_fast_to_slow<2>({ desc.dim0, desc.dim1 }, { desc.stride0, desc.stride1 });
    }

    bool can_map_m2_direct(const TensorView3DDesc& desc)
    {
        return desc.stride1 == 1 &&
            is_non_overlapping_fast_to_slow<3>({ desc.dim1, desc.dim2, desc.dim0 }, { desc.stride1, desc.stride2, desc.stride0 });
    }

    TensorBinding create_standard_binding(ggml_context* ctx, const TensorView2DDesc& desc)
    {
        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        ggml_tensor* view = ggml_view_2d(ctx, base, desc.dim1, desc.dim0, static_cast<std::size_t>(desc.stride0) * sizeof(float), 0);
        return { base, view, static_cast<std::size_t>(desc.raw_bytes) };
    }

    TensorBinding create_standard_binding(ggml_context* ctx, const TensorView3DDesc& desc)
    {
        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        ggml_tensor* view = ggml_view_3d(
            ctx,
            base,
            desc.dim2,
            desc.dim1,
            desc.dim0,
            static_cast<std::size_t>(desc.stride1) * sizeof(float),
            static_cast<std::size_t>(desc.stride0) * sizeof(float),
            0);
        return { base, view, static_cast<std::size_t>(desc.raw_bytes) };
    }

    TensorBinding create_standard_binding(ggml_context* ctx, const TensorView4DDesc& desc)
    {
        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        ggml_tensor* view = ggml_view_4d(
            ctx,
            base,
            desc.ne0,
            desc.ne1,
            desc.ne2,
            desc.ne3,
            static_cast<std::size_t>(desc.nb1),
            static_cast<std::size_t>(desc.nb2),
            static_cast<std::size_t>(desc.nb3),
            0);
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
        ggml_tensor* view = ggml_view_3d(
            ctx,
            base,
            desc.dim1,
            desc.dim2,
            desc.dim0,
            static_cast<std::size_t>(desc.stride2) * sizeof(float),
            static_cast<std::size_t>(desc.stride0) * sizeof(float),
            0);
        return { base, view, static_cast<std::size_t>(desc.raw_bytes) };
    }

    std::vector<float> pack_m2(const TensorView2DDesc& desc)
    {
        const float* data = static_cast<const float*>(desc.data);
        std::vector<float> packed(static_cast<std::size_t>(desc.dim0) * desc.dim1);

        for (int row = 0; row < desc.dim0; ++row)
        {
            for (int col = 0; col < desc.dim1; ++col)
            {
                packed[(static_cast<std::size_t>(col) * desc.dim0) + row] =
                    data[(static_cast<std::size_t>(row) * desc.stride0) + (static_cast<std::size_t>(col) * desc.stride1)];
            }
        }

        return packed;
    }

    std::vector<float> pack_m2(const TensorView3DDesc& desc)
    {
        const float* data = static_cast<const float*>(desc.data);
        std::vector<float> packed(static_cast<std::size_t>(desc.dim0) * desc.dim1 * desc.dim2);

        for (int batch = 0; batch < desc.dim0; ++batch)
        {
            for (int row = 0; row < desc.dim1; ++row)
            {
                for (int col = 0; col < desc.dim2; ++col)
                {
                    packed[((static_cast<std::size_t>(batch) * desc.dim2 + col) * desc.dim1) + row] =
                        data[(static_cast<std::size_t>(batch) * desc.stride0) +
                             (static_cast<std::size_t>(row) * desc.stride1) +
                             (static_cast<std::size_t>(col) * desc.stride2)];
                }
            }
        }

        return packed;
    }

    std::vector<float> pack_standard(const TensorView2DDesc& desc)
    {
        const float* data = static_cast<const float*>(desc.data);
        std::vector<float> packed(static_cast<std::size_t>(desc.dim0) * desc.dim1);

        for (int row = 0; row < desc.dim0; ++row)
        {
            for (int col = 0; col < desc.dim1; ++col)
            {
                packed[(static_cast<std::size_t>(row) * desc.dim1) + col] =
                    data[(static_cast<std::size_t>(row) * desc.stride0) + (static_cast<std::size_t>(col) * desc.stride1)];
            }
        }

        return packed;
    }

    std::vector<float> pack_standard(const TensorView3DDesc& desc)
    {
        const float* data = static_cast<const float*>(desc.data);
        std::vector<float> packed(static_cast<std::size_t>(desc.dim0) * desc.dim1 * desc.dim2);

        for (int batch = 0; batch < desc.dim0; ++batch)
        {
            for (int row = 0; row < desc.dim1; ++row)
            {
                for (int col = 0; col < desc.dim2; ++col)
                {
                    packed[((static_cast<std::size_t>(batch) * desc.dim1 + row) * desc.dim2) + col] =
                        data[(static_cast<std::size_t>(batch) * desc.stride0) +
                             (static_cast<std::size_t>(row) * desc.stride1) +
                             (static_cast<std::size_t>(col) * desc.stride2)];
                }
            }
        }

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
        ggml_backend_tensor_set(binding.storage, data, 0, size);
    }

    bool is_pointer_aligned_for_backend(ggml_backend_t backend, const void* ptr)
    {
        if (backend == nullptr || ptr == nullptr)
            return false;
        std::size_t alignment = ggml_backend_get_alignment(backend);
        if (alignment == 0)
            alignment = GGML_MEM_ALIGN;
        return (reinterpret_cast<std::uintptr_t>(ptr) % alignment) == 0;
    }

    bool try_create_host_ptr_buffer(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        void* data,
        std::size_t raw_bytes,
        ggml_backend_buffer_t& out_buffer)
    {
        out_buffer = nullptr;
        if (backend == nullptr || dev == nullptr || data == nullptr || raw_bytes == 0)
            return false;
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (!props.caps.buffer_from_host_ptr)
            return false;
        if (!is_pointer_aligned_for_backend(backend, data))
            return false;
        out_buffer = ggml_backend_dev_buffer_from_host_ptr(dev, data, raw_bytes, raw_bytes);
        return out_buffer != nullptr;
    }

    // Create a binding that uses host ptr directly as Metal shared memory (zero host-device copies on Apple Silicon).
    // Returns empty binding on failure. Caller must keep buffer_handle alive until compute completes.
    bool create_binding_from_host_ptr_2d(
        ggml_context* ctx,
        ggml_backend_t backend,
        const TensorView2DDesc& desc,
        TensorBinding& out_binding,
        ggml_backend_buffer_t& out_buffer)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) return false;

        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        if (!try_create_host_ptr_buffer(backend, dev, desc.data, raw_bytes, out_buffer)) return false;

        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, raw_bytes / static_cast<std::int64_t>(sizeof(float)));
        if (base == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_status st = ggml_backend_tensor_alloc(out_buffer, base, const_cast<void*>(desc.data));
        if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_tensor* view = ggml_view_2d(ctx, base, desc.dim1, desc.dim0, static_cast<std::size_t>(desc.stride0) * sizeof(float), 0);
        if (view == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        out_binding = { base, view, raw_bytes };
        return true;
    }

    // Zero-copy for m2 (direct layout: stride0==1, column-major)
    bool create_binding_from_host_ptr_direct_m2_2d(
        ggml_context* ctx,
        ggml_backend_t backend,
        const TensorView2DDesc& desc,
        TensorBinding& out_binding,
        ggml_backend_buffer_t& out_buffer)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) return false;

        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        if (!try_create_host_ptr_buffer(backend, dev, desc.data, raw_bytes, out_buffer)) return false;

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
        ggml_context* ctx,
        ggml_backend_t backend,
        const TensorView3DDesc& desc,
        TensorBinding& out_binding,
        ggml_backend_buffer_t& out_buffer)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) return false;

        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        if (!try_create_host_ptr_buffer(backend, dev, desc.data, raw_bytes, out_buffer)) return false;

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
        ggml_context* ctx,
        ggml_backend_t backend,
        const TensorView3DDesc& desc,
        TensorBinding& out_binding,
        ggml_backend_buffer_t& out_buffer)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) return false;

        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        if (!try_create_host_ptr_buffer(backend, dev, desc.data, raw_bytes, out_buffer)) return false;

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
        ggml_context* ctx,
        ggml_backend_t backend,
        const TensorView4DDesc& desc,
        TensorBinding& out_binding,
        ggml_backend_buffer_t& out_buffer)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) return false;

        std::size_t raw_bytes = static_cast<std::size_t>(desc.raw_bytes);
        if (!try_create_host_ptr_buffer(backend, dev, desc.data, raw_bytes, out_buffer)) return false;

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
        ggml_context* ctx,
        ggml_backend_t backend,
        const ContiguousTensorDesc& desc,
        TensorBinding& out_binding,
        ggml_backend_buffer_t& out_buffer)
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(backend);
        if (dev == nullptr) return false;

        std::size_t raw_bytes = static_cast<std::size_t>(desc.element_count) * sizeof(float);
        if (raw_bytes == 0) return false;

        if (!try_create_host_ptr_buffer(backend, dev, desc.data, raw_bytes, out_buffer)) return false;

        ggml_tensor* base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, desc.element_count);
        if (base == nullptr) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        ggml_status st = ggml_backend_tensor_alloc(out_buffer, base, const_cast<void*>(desc.data));
        if (st != GGML_STATUS_SUCCESS) { ggml_backend_buffer_free(out_buffer); out_buffer = nullptr; return false; }

        out_binding = { base, base, raw_bytes };
        return true;
    }

    int addmm_f32_impl(
        const TensorView2DDesc& result_desc,
        const TensorView2DDesc& src_desc,
        const TensorView2DDesc& m1_desc,
        const TensorView2DDesc& m2_desc,
        float beta,
        float alpha)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(m1_desc, "m1") || !validate_desc(m2_desc, "m2"))
        {
            return 0;
        }

        if (beta != 0.0f && !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        const int rows = result_desc.dim0;
        const int cols = result_desc.dim1;
        const int shared = m1_desc.dim1;

        if (m1_desc.dim0 != rows || m2_desc.dim0 != shared || m2_desc.dim1 != cols)
        {
            set_last_error("Size mismatch passed to ggml addmm.");
            return 0;
        }

        if (beta != 0.0f && ((rows % src_desc.dim0) != 0 || (cols % src_desc.dim1) != 0))
        {
            set_last_error("Source tensor shape cannot be broadcast to result shape for ggml addmm.");
            return 0;
        }

        if (!can_map_standard_view(result_desc))
        {
            set_last_error("Result tensor layout is not supported by the ggml addmm Metal path.");
            return 0;
        }

        if (beta != 0.0f && !can_map_standard_view(src_desc))
        {
            set_last_error("Source tensor layout is not supported by the ggml addmm Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(m1_desc) && can_map_m2_direct(m2_desc);

        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        std::vector<BufferHandle> host_ptr_buffers;
        TensorBinding result_binding;
        TensorBinding m1_binding;
        TensorBinding src_binding;
        std::vector<float> packed_m1;
        std::vector<float> packed_m2;

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            result_binding = create_standard_binding(context.value, result_desc);

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, m1_desc, m1_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        else
            m1_binding = can_map_standard_view(m1_desc)
                ? create_standard_binding(context.value, m1_desc)
                : create_packed_standard_binding(context.value, m1_desc, packed_m1);

        if (use_zero_copy && beta != 0.0f)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        else if (beta != 0.0f)
            src_binding = create_standard_binding(context.value, src_desc);

        TensorBinding m2_binding;
        bool m2_zero_copy = false;
        if (use_zero_copy && can_map_m2_direct(m2_desc))
        {
            ggml_backend_buffer_t buf = nullptr;
            if (create_binding_from_host_ptr_direct_m2_2d(context.value, g_backend, m2_desc, m2_binding, buf))
            {
                m2_zero_copy = true;
                host_ptr_buffers.emplace_back(buf);
            }
        }
        if (!m2_zero_copy)
            m2_binding = can_map_m2_direct(m2_desc)
                ? create_direct_m2_binding(context.value, m2_desc)
                : create_packed_m2_binding(context.value, m2_desc, packed_m2);

        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            m1_binding.storage == nullptr || m1_binding.tensor == nullptr ||
            m2_binding.storage == nullptr || m2_binding.tensor == nullptr ||
            (beta != 0.0f && (src_binding.storage == nullptr || src_binding.tensor == nullptr)))
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* mm_tensor = ggml_mul_mat(context.value, m2_binding.tensor, m1_binding.tensor);
        if (mm_tensor == nullptr)
        {
            set_last_error("Failed to create ggml matmul node.");
            return 0;
        }

        ggml_tensor* combined_tensor = mm_tensor;
        if (alpha != 1.0f)
        {
            combined_tensor = ggml_scale(context.value, combined_tensor, alpha);
            if (combined_tensor == nullptr)
            {
                set_last_error("Failed to scale ggml matmul output.");
                return 0;
            }
        }

        if (beta != 0.0f)
        {
            ggml_tensor* scaled_src = src_binding.tensor;
            if (beta != 1.0f)
            {
                scaled_src = ggml_scale(context.value, src_binding.tensor, beta);
                if (scaled_src == nullptr)
                {
                    set_last_error("Failed to scale ggml source tensor.");
                    return 0;
                }
            }

            combined_tensor = ggml_add(context.value, combined_tensor, scaled_src);
            if (combined_tensor == nullptr)
            {
                set_last_error("Failed to create ggml add node.");
                return 0;
            }
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, combined_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            if (beta != 0.0f)
                upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (packed_m1.empty())
                upload_binding(m1_binding, m1_desc.data, m1_binding.raw_bytes);
            else
                upload_binding(m1_binding, packed_m1.data(), m1_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }
        if (!m2_zero_copy)
        {
            if (packed_m2.empty())
                upload_binding(m2_binding, m2_desc.data, m2_binding.raw_bytes);
            else
                upload_binding(m2_binding, packed_m2.data(), m2_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int addmm_quant_f32_impl(
        const TensorView2DDesc& result_desc,
        const TensorView2DDesc& m1_desc,
        const QuantizedWeightDesc& m2_quant)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(m1_desc, "m1"))
        {
            return 0;
        }

        if (m2_quant.data == nullptr || m2_quant.ne0 <= 0 || m2_quant.ne1 <= 0 || m2_quant.raw_bytes <= 0)
        {
            set_last_error("Invalid quantized weight descriptor.");
            return 0;
        }

        const int rows = result_desc.dim0;   // seqLen
        const int cols = result_desc.dim1;   // outDim
        const int shared = m1_desc.dim1;     // inDim

        if (m1_desc.dim0 != rows)
        {
            set_last_error("Size mismatch: m1.dim0 != result.dim0 in addmm_quant.");
            return 0;
        }

        // m2_quant: ne0 = inDim (shared), ne1 = outDim
        if (m2_quant.ne0 != shared || m2_quant.ne1 != cols)
        {
            set_last_error("Size mismatch: quantized weight dims don't match in addmm_quant.");
            return 0;
        }

        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        std::vector<BufferHandle> host_ptr_buffers;
        bool use_zero_copy = can_map_standard_view(m1_desc);

        // Result binding
        TensorBinding result_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            result_binding = create_standard_binding(context.value, result_desc);

        // m1 (input) binding
        TensorBinding m1_binding;
        std::vector<float> packed_m1;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, m1_desc, m1_binding, buf))
            {
                // Zero-copy requires both result and m1 bindings to succeed.
                // If m1 cannot be host-mapped (e.g., alignment constraints), fall back both tensors
                // to regular backend-managed buffers to keep upload/download logic consistent.
                use_zero_copy = false;
                result_binding = create_standard_binding(context.value, result_desc);
                m1_binding = can_map_standard_view(m1_desc)
                    ? create_standard_binding(context.value, m1_desc)
                    : create_packed_standard_binding(context.value, m1_desc, packed_m1);
            }
            else
                host_ptr_buffers.emplace_back(buf);
        }
        else
            m1_binding = can_map_standard_view(m1_desc)
                ? create_standard_binding(context.value, m1_desc)
                : create_packed_standard_binding(context.value, m1_desc, packed_m1);

        // m2 (quantized weight) binding: create ggml tensor with actual quantized type
        ggml_type qtype = static_cast<ggml_type>(m2_quant.ggml_type);
        ggml_tensor* m2_tensor = ggml_new_tensor_2d(context.value, qtype, m2_quant.ne0, m2_quant.ne1);
        TensorBinding m2_binding = { m2_tensor, m2_tensor, static_cast<std::size_t>(m2_quant.raw_bytes) };

        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            m1_binding.storage == nullptr || m1_binding.tensor == nullptr ||
            m2_tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views for addmm_quant.");
            return 0;
        }

        // Try cached host_ptr binding for quantized weight (stable pointer across calls)
        bool m2_bound = false;
        {
            ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
            if (dev != nullptr && m2_quant.raw_bytes >= 4096)
            {
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (props.caps.buffer_from_host_ptr)
                {
                    ggml_backend_buffer_t buf = nullptr;
                    auto it = g_host_buffer_cache.find(m2_quant.data);
                    if (it != g_host_buffer_cache.end() && it->second.bytes == static_cast<std::size_t>(m2_quant.raw_bytes))
                    {
                        buf = it->second.buffer;
                    }
                    else
                    {
                        (void)try_create_host_ptr_buffer(
                            g_backend,
                            dev,
                            m2_quant.data,
                            static_cast<std::size_t>(m2_quant.raw_bytes),
                            buf);
                        if (buf != nullptr)
                            g_host_buffer_cache[m2_quant.data] = {buf, static_cast<std::size_t>(m2_quant.raw_bytes)};
                    }
                    if (buf != nullptr)
                    {
                        ggml_status st = ggml_backend_tensor_alloc(buf, m2_tensor, m2_quant.data);
                        m2_bound = (st == GGML_STATUS_SUCCESS);
                    }
                }
            }
        }

        ggml_tensor* mm_tensor = ggml_mul_mat(context.value, m2_binding.tensor, m1_binding.tensor);
        if (mm_tensor == nullptr)
        {
            set_last_error("Failed to create ggml matmul node for addmm_quant.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, mm_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml output copy node for addmm_quant.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph for addmm_quant.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer for addmm_quant.");
            return 0;
        }

        // Upload data
        if (!use_zero_copy)
        {
            if (packed_m1.empty())
                upload_binding(m1_binding, m1_desc.data, m1_binding.raw_bytes);
            else
                upload_binding(m1_binding, packed_m1.data(), m1_binding.raw_bytes);
        }

        if (!m2_bound)
            upload_binding(m2_binding, m2_quant.data, m2_binding.raw_bytes);

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for addmm_quant.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    // get_rows from a quantized source tensor: result[i] = dequant(src[indices[i]])
    int get_rows_quant_f32_impl(
        const TensorView2DDesc& result_desc,
        const QuantizedWeightDesc& src_quant,
        const ContiguousTensorDesc& indices_desc)
    {
        if (!ensure_backend())
            return 0;

        if (!validate_desc(result_desc, "result") || !validate_desc(indices_desc, "indices"))
            return 0;

        if (src_quant.data == nullptr || src_quant.ne0 <= 0 || src_quant.ne1 <= 0 || src_quant.raw_bytes <= 0)
        {
            set_last_error("Invalid quantized weight descriptor for get_rows_quant.");
            return 0;
        }

        const int num_indices = static_cast<int>(indices_desc.element_count);
        const int embedding_dim = static_cast<int>(src_quant.ne0);

        if (result_desc.dim0 != num_indices || result_desc.dim1 != embedding_dim)
        {
            set_last_error("Shape mismatch in get_rows_quant: result must be [num_indices, ne0].");
            return 0;
        }

        const std::size_t ctx_size = 2 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for get_rows_quant.");
            return 0;
        }

        std::vector<BufferHandle> host_ptr_buffers;
        bool use_zero_copy = can_map_standard_view(result_desc);

        // Result binding (F32 output)
        TensorBinding result_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            result_binding = create_standard_binding(context.value, result_desc);

        // Source tensor: quantized type
        ggml_type qtype = static_cast<ggml_type>(src_quant.ggml_type);
        ggml_tensor* src_tensor = ggml_new_tensor_2d(context.value, qtype, src_quant.ne0, src_quant.ne1);

        // Index tensor: I32
        ggml_tensor* index_tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_I32, num_indices);

        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_tensor == nullptr || index_tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors for get_rows_quant.");
            return 0;
        }

        TensorBinding src_binding = { src_tensor, src_tensor, static_cast<std::size_t>(src_quant.raw_bytes) };

        // Cache quantized source buffer (same as addmm_quant)
        bool src_bound = false;
        {
            ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
            if (dev != nullptr && src_quant.raw_bytes >= 4096)
            {
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (props.caps.buffer_from_host_ptr)
                {
                    ggml_backend_buffer_t buf = nullptr;
                    auto it = g_host_buffer_cache.find(src_quant.data);
                    if (it != g_host_buffer_cache.end() && it->second.bytes == static_cast<std::size_t>(src_quant.raw_bytes))
                    {
                        buf = it->second.buffer;
                    }
                    else
                    {
                        (void)try_create_host_ptr_buffer(
                            g_backend,
                            dev,
                            src_quant.data,
                            static_cast<std::size_t>(src_quant.raw_bytes),
                            buf);
                        if (buf != nullptr)
                            g_host_buffer_cache[src_quant.data] = {buf, static_cast<std::size_t>(src_quant.raw_bytes)};
                    }
                    if (buf != nullptr)
                    {
                        ggml_status st = ggml_backend_tensor_alloc(buf, src_tensor, src_quant.data);
                        src_bound = (st == GGML_STATUS_SUCCESS);
                    }
                }
            }
        }

        // Build graph: get_rows(src, indices) -> copy -> result
        ggml_tensor* rows_tensor = ggml_get_rows(context.value, src_tensor, index_tensor);
        if (rows_tensor == nullptr)
        {
            set_last_error("Failed to create ggml get_rows node for get_rows_quant.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, rows_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml output copy node for get_rows_quant.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph for get_rows_quant.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer for get_rows_quant.");
            return 0;
        }

        // Upload quantized source if not zero-copy bound
        if (!src_bound)
            upload_binding(src_binding, src_quant.data, src_binding.raw_bytes);

        // Upload indices
        std::vector<std::int32_t> indices;
        if (!read_i32_values(indices, indices_desc, "indices"))
            return 0;
        ggml_backend_tensor_set(index_tensor, indices.data(), 0, indices.size() * sizeof(std::int32_t));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for get_rows_quant.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    // Batched quantized matmul: result[b] = input[b] * quantWeights[b]^T
    // Each batch uses a separate quantized weight at offset b*per_weight_bytes
    int addmm_quant_batch_f32_impl(
        const TensorView2DDesc& result_desc,
        const TensorView2DDesc& m1_desc,
        const QuantizedWeightDesc& m2_quant,
        int batch_count,
        const std::int64_t* weight_offsets,
        const std::int64_t* weight_ne1_arr)
    {
        if (!ensure_backend())
            return 0;

        if (!validate_desc(result_desc, "result") || !validate_desc(m1_desc, "m1"))
            return 0;

        if (m2_quant.data == nullptr || batch_count <= 0)
        {
            set_last_error("Invalid arguments for addmm_quant_batch.");
            return 0;
        }

        // Process each batch sequentially using the existing single-batch impl
        const int in_dim = m1_desc.dim1;
        int result_row = 0;
        int m1_row = 0;

        for (int b = 0; b < batch_count; b++)
        {
            std::int64_t ne1_b = weight_ne1_arr[b];
            std::int64_t offset_b = weight_offsets[b];

            TensorView2DDesc r_desc = result_desc;
            r_desc.dim0 = 1;
            r_desc.data = static_cast<char*>(result_desc.data) + static_cast<std::size_t>(result_row) * result_desc.stride0 * sizeof(float);
            r_desc.raw_bytes = static_cast<std::int64_t>(r_desc.dim1) * sizeof(float);

            TensorView2DDesc input_desc = m1_desc;
            input_desc.dim0 = 1;
            input_desc.data = static_cast<char*>(m1_desc.data) + static_cast<std::size_t>(m1_row) * m1_desc.stride0 * sizeof(float);
            input_desc.raw_bytes = static_cast<std::int64_t>(input_desc.dim1) * sizeof(float);

            QuantizedWeightDesc w_desc;
            w_desc.data = static_cast<char*>(m2_quant.data) + offset_b;
            w_desc.ggml_type = m2_quant.ggml_type;
            w_desc.ne0 = m2_quant.ne0;
            w_desc.ne1 = ne1_b;
            long rowSize = ggml_row_size(static_cast<ggml_type>(m2_quant.ggml_type), m2_quant.ne0);
            w_desc.raw_bytes = ne1_b * rowSize;

            int ok = addmm_quant_f32_impl(r_desc, input_desc, w_desc);
            if (!ok) return 0;

            result_row++;
            m1_row++;
        }

        clear_last_error();
        return 1;
    }

    int addmm_batch_f32_impl(
        const TensorView3DDesc& result_desc,
        const TensorView3DDesc& src_desc,
        const TensorView3DDesc& m1_desc,
        const TensorView3DDesc& m2_desc,
        float beta,
        float alpha)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(m1_desc, "m1") || !validate_desc(m2_desc, "m2"))
        {
            return 0;
        }

        if (beta != 0.0f && !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        const int batches = result_desc.dim0;
        const int rows = result_desc.dim1;
        const int cols = result_desc.dim2;
        const int shared = m1_desc.dim2;

        if (m1_desc.dim0 != batches || m2_desc.dim0 != batches || m1_desc.dim1 != rows || m2_desc.dim1 != shared || m2_desc.dim2 != cols)
        {
            set_last_error("Size mismatch passed to ggml addmmbatch.");
            return 0;
        }

        if (beta != 0.0f && ((batches % src_desc.dim0) != 0 || (rows % src_desc.dim1) != 0 || (cols % src_desc.dim2) != 0))
        {
            set_last_error("Source tensor shape cannot be broadcast to result shape for ggml addmmbatch.");
            return 0;
        }

        if (!can_map_standard_view(result_desc))
        {
            set_last_error("Result tensor layout is not supported by the ggml addmmbatch Metal path.");
            return 0;
        }

        if (beta != 0.0f && !can_map_standard_view(src_desc))
        {
            set_last_error("Source tensor layout is not supported by the ggml addmmbatch Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(m1_desc) && can_map_m2_direct(m2_desc);

        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        std::vector<BufferHandle> host_ptr_buffers;
        TensorBinding result_binding;
        TensorBinding m1_binding;
        TensorBinding src_binding;
        std::vector<float> packed_m1;
        std::vector<float> packed_m2;
        TensorBinding m2_binding;
        bool m2_zero_copy = false;

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_3d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            result_binding = create_standard_binding(context.value, result_desc);

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_3d(context.value, g_backend, m1_desc, m1_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        else
            m1_binding = can_map_standard_view(m1_desc)
                ? create_standard_binding(context.value, m1_desc)
                : create_packed_standard_binding(context.value, m1_desc, packed_m1);

        if (use_zero_copy && beta != 0.0f)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_3d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        else if (beta != 0.0f)
            src_binding = create_standard_binding(context.value, src_desc);

        if (use_zero_copy && can_map_m2_direct(m2_desc))
        {
            ggml_backend_buffer_t buf = nullptr;
            if (create_binding_from_host_ptr_direct_m2_3d(context.value, g_backend, m2_desc, m2_binding, buf))
            {
                m2_zero_copy = true;
                host_ptr_buffers.emplace_back(buf);
            }
        }
        if (!m2_zero_copy)
            m2_binding = can_map_m2_direct(m2_desc)
                ? create_direct_m2_binding(context.value, m2_desc)
                : create_packed_m2_binding(context.value, m2_desc, packed_m2);

        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            m1_binding.storage == nullptr || m1_binding.tensor == nullptr ||
            m2_binding.storage == nullptr || m2_binding.tensor == nullptr ||
            (beta != 0.0f && (src_binding.storage == nullptr || src_binding.tensor == nullptr)))
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* mm_tensor = ggml_mul_mat(context.value, m2_binding.tensor, m1_binding.tensor);
        if (mm_tensor == nullptr)
        {
            set_last_error("Failed to create ggml batched matmul node.");
            return 0;
        }

        ggml_tensor* combined_tensor = mm_tensor;
        if (alpha != 1.0f)
        {
            combined_tensor = ggml_scale(context.value, combined_tensor, alpha);
            if (combined_tensor == nullptr)
            {
                set_last_error("Failed to scale ggml batched matmul output.");
                return 0;
            }
        }

        if (beta != 0.0f)
        {
            ggml_tensor* scaled_src = src_binding.tensor;
            if (beta != 1.0f)
            {
                scaled_src = ggml_scale(context.value, src_binding.tensor, beta);
                if (scaled_src == nullptr)
                {
                    set_last_error("Failed to scale ggml batched source tensor.");
                    return 0;
                }
            }

            combined_tensor = ggml_add(context.value, combined_tensor, scaled_src);
            if (combined_tensor == nullptr)
            {
                set_last_error("Failed to create ggml batched add node.");
                return 0;
            }
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, combined_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml batched output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            if (beta != 0.0f)
                upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (packed_m1.empty())
                upload_binding(m1_binding, m1_desc.data, m1_binding.raw_bytes);
            else
                upload_binding(m1_binding, packed_m1.data(), m1_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }
        if (!m2_zero_copy)
        {
            if (packed_m2.empty())
                upload_binding(m2_binding, m2_desc.data, m2_binding.raw_bytes);
            else
                upload_binding(m2_binding, packed_m2.data(), m2_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    bool same_shape(const TensorView4DDesc& lhs, const TensorView4DDesc& rhs)
    {
        return lhs.ne0 == rhs.ne0 &&
            lhs.ne1 == rhs.ne1 &&
            lhs.ne2 == rhs.ne2 &&
            lhs.ne3 == rhs.ne3;
    }

    bool same_shape_with_last_dim_reduced(const TensorView4DDesc& result, const TensorView4DDesc& src)
    {
        return result.ne0 == 1 &&
            result.ne1 == src.ne1 &&
            result.ne2 == src.ne2 &&
            result.ne3 == src.ne3;
    }

    bool can_repeat(const TensorView4DDesc& repeated, const TensorView4DDesc& target)
    {
        return (target.ne0 % repeated.ne0) == 0 &&
            (target.ne1 % repeated.ne1) == 0 &&
            (target.ne2 % repeated.ne2) == 0 &&
            (target.ne3 % repeated.ne3) == 0;
    }

    TensorBinding create_scalar_binding(ggml_context* ctx)
    {
        ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
        return { tensor, tensor, sizeof(float) };
    }

    TensorBinding create_matrix_binding(ggml_context* ctx, std::int64_t cols, std::int64_t rows)
    {
        ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
        return { tensor, tensor, static_cast<std::size_t>(cols * rows * static_cast<std::int64_t>(sizeof(float))) };
    }

    bool build_cross_entropy_label_buffer(
        std::vector<float>& labels,
        const ContiguousTensorDesc& target_indices_desc,
        std::int64_t rows,
        std::int64_t cols,
        float label_smooth)
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
        {
            return false;
        }
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

    ggml_tensor* make_unary_tensor(ggml_context* ctx, UnaryOpCode op, ggml_tensor* src)
    {
        switch (op)
        {
        case UnaryOpCode::Neg:
            return ggml_neg(ctx, src);
        case UnaryOpCode::Exp:
            return ggml_exp(ctx, src);
        case UnaryOpCode::Log:
            return ggml_log(ctx, src);
        case UnaryOpCode::Sqrt:
            return ggml_sqrt(ctx, src);
        case UnaryOpCode::Relu:
            return ggml_relu(ctx, src);
        case UnaryOpCode::Sigmoid:
            return ggml_sigmoid(ctx, src);
        case UnaryOpCode::Tanh:
            return ggml_tanh(ctx, src);
        case UnaryOpCode::SiLU:
            return ggml_silu(ctx, src);
        case UnaryOpCode::Step:
            return ggml_step(ctx, src);
        case UnaryOpCode::Abs:
            return ggml_abs(ctx, src);
        case UnaryOpCode::Sign:
            return ggml_sgn(ctx, src);
        case UnaryOpCode::GELU:
            return ggml_gelu(ctx, src);
        default:
            set_last_error("Unsupported unary ggml op code.");
            return nullptr;
        }
    }

    ggml_tensor* make_fused_act_mul_tensor(ggml_context* ctx, FusedActMulOpCode op, ggml_tensor* a, ggml_tensor* b)
    {
        switch (op)
        {
        case FusedActMulOpCode::SiLUMul:
            return ggml_mul(ctx, ggml_silu(ctx, a), b);
        case FusedActMulOpCode::GELUMul:
            return ggml_mul(ctx, ggml_gelu(ctx, a), b);
        case FusedActMulOpCode::SigmoidMul:
            return ggml_mul(ctx, a, ggml_sigmoid(ctx, b));
        default:
            set_last_error("Unsupported fused activation-multiply ggml op code.");
            return nullptr;
        }
    }

    ggml_tensor* make_binary_tensor(ggml_context* ctx, BinaryTensorOpCode op, ggml_tensor* lhs, ggml_tensor* rhs)
    {
        switch (op)
        {
        case BinaryTensorOpCode::Add:
            return ggml_add(ctx, lhs, rhs);
        case BinaryTensorOpCode::Sub:
            return ggml_sub(ctx, lhs, rhs);
        case BinaryTensorOpCode::Mul:
            return ggml_mul(ctx, lhs, rhs);
        case BinaryTensorOpCode::Div:
            return ggml_div(ctx, lhs, rhs);
        default:
            set_last_error("Unsupported binary ggml op code.");
            return nullptr;
        }
    }

    ggml_tensor* make_norm_tensor(ggml_context* ctx, NormOpCode op, ggml_tensor* src, float eps)
    {
        switch (op)
        {
        case NormOpCode::LayerNorm:
            return ggml_norm(ctx, src, eps);
        case NormOpCode::RmsNorm:
            return ggml_rms_norm(ctx, src, eps);
        default:
            set_last_error("Unsupported norm ggml op code.");
            return nullptr;
        }
    }

    ggml_tensor* make_reduction_tensor(ggml_context* ctx, ReductionOpCode op, ggml_tensor* src)
    {
        switch (op)
        {
        case ReductionOpCode::Sum:
            return ggml_sum_rows(ctx, src);
        case ReductionOpCode::Mean:
            return ggml_mean(ctx, src);
        default:
            set_last_error("Unsupported reduction ggml op code.");
            return nullptr;
        }
    }

    bool is_vector_like(const TensorView4DDesc& desc, std::int64_t width)
    {
        return desc.ne0 == width && desc.ne1 == 1 && desc.ne2 == 1 && desc.ne3 == 1;
    }

    std::int64_t flat_row_count(const TensorView4DDesc& desc)
    {
        return static_cast<std::int64_t>(desc.ne1) * desc.ne2 * desc.ne3;
    }

    ggml_tensor* flatten_to_rows(ggml_context* ctx, ggml_tensor* tensor, std::int64_t cols, std::int64_t rows)
    {
        return ggml_reshape_2d(ctx, tensor, cols, rows);
    }

    ggml_tensor* sum_rows_to_feature_vector(ggml_context* ctx, ggml_tensor* tensor)
    {
        ggml_tensor* transposed = ggml_transpose(ctx, tensor);
        ggml_tensor* transposed_contiguous = transposed == nullptr ? nullptr : ggml_cont(ctx, transposed);
        ggml_tensor* summed = transposed_contiguous == nullptr ? nullptr : ggml_sum_rows(ctx, transposed_contiguous);
        ggml_tensor* restored = summed == nullptr ? nullptr : ggml_transpose(ctx, summed);
        return restored == nullptr ? nullptr : ggml_cont(ctx, restored);
    }

    int reduce_last_dim_f32_impl(
        ReductionOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        if (!same_shape_with_last_dim_reduced(result_desc, src_desc))
        {
            set_last_error("Result tensor shape must match source shape with the last dimension reduced to 1.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml reduction Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
        }
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to create ggml contiguous reduction input.");
            return 0;
        }

        ggml_tensor* reduced_tensor = make_reduction_tensor(context.value, op, contiguous_src);
        if (reduced_tensor == nullptr)
        {
            if (g_last_error.empty())
            {
                set_last_error("Failed to create ggml reduction node.");
            }
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, reduced_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml reduction output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int index_reduction_f32_impl(
        IndexReductionOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        if (!same_shape_with_last_dim_reduced(result_desc, src_desc))
        {
            set_last_error("Result tensor shape must match source shape with the last dimension reduced to 1.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml index-reduction Metal path.");
            return 0;
        }

        bool src_zero_copy = can_map_standard_view(src_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding src_binding;
        if (src_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                src_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!src_zero_copy)
            src_binding = create_standard_binding(context.value, src_desc);
        if (src_binding.storage == nullptr || src_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to create ggml contiguous index-reduction input.");
            return 0;
        }

        ggml_tensor* reduction_input = contiguous_src;
        if (op == IndexReductionOpCode::Argmin)
        {
            reduction_input = ggml_neg(context.value, contiguous_src);
            if (reduction_input == nullptr)
            {
                set_last_error("Failed to create ggml argmin preprocessing node.");
                return 0;
            }
        }
        else if (op != IndexReductionOpCode::Argmax)
        {
            set_last_error("Unsupported index-reduction ggml op code.");
            return 0;
        }

        const std::int64_t rows = flat_row_count(src_desc);
        ggml_tensor* flat_input = flatten_to_rows(context.value, reduction_input, src_desc.ne0, rows);
        ggml_tensor* arg_tensor = flat_input == nullptr ? nullptr : ggml_argmax(context.value, flat_input);
        if (flat_input == nullptr || arg_tensor == nullptr)
        {
            set_last_error("Failed to create ggml index-reduction node.");
            return 0;
        }

        ggml_set_output(arg_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, arg_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!src_zero_copy)
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        std::vector<std::int32_t> host_indices(static_cast<std::size_t>(rows));
        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(arg_tensor, host_indices.data(), 0, host_indices.size() * sizeof(std::int32_t));

        float* result_data = static_cast<float*>(result_desc.data);
        for (std::size_t i = 0; i < host_indices.size(); ++i)
        {
            result_data[i] = static_cast<float>(host_indices[i]);
        }

        clear_last_error();
        return 1;
    }

    int copy_f32_impl(
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        if (!same_shape(result_desc, src_desc))
        {
            set_last_error("Source tensor shape does not match result shape for ggml copy.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml copy Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
        }
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to create ggml contiguous copy input.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, contiguous_src, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int mul_mat_id_f32_impl(
        const TensorView3DDesc& result_desc,
        const TensorView3DDesc& expert_desc,
        const TensorView3DDesc& input_desc,
        const ContiguousTensorDesc& ids_desc,
        int ids_rows,
        int ids_cols)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(expert_desc, "expertWeights") || !validate_desc(input_desc, "input") || !validate_desc(ids_desc, "ids"))
        {
            return 0;
        }

        if (ids_rows <= 0 || ids_cols <= 0)
        {
            set_last_error("mulmatid requires positive id matrix dimensions.");
            return 0;
        }

        if (expert_desc.dim2 != input_desc.dim2)
        {
            set_last_error("mulmatid expects expert weights and input to share the inner dimension.");
            return 0;
        }

        if (input_desc.dim0 != ids_rows || (ids_cols % input_desc.dim1) != 0)
        {
            set_last_error("mulmatid expects ids rows to match input tokens and ids cols to broadcast over input expert slots.");
            return 0;
        }

        if (result_desc.dim0 != input_desc.dim0 || result_desc.dim1 != ids_cols || result_desc.dim2 != expert_desc.dim1)
        {
            set_last_error("mulmatid expects result shape [tokens, expert_used, rows].");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(expert_desc) || !can_map_standard_view(input_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml mulmatid path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(expert_desc) && can_map_standard_view(input_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 2 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding expert_binding;
        TensorBinding input_binding;

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_3d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            result_binding = create_standard_binding(context.value, result_desc);

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_3d(context.value, g_backend, expert_desc, expert_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            expert_binding = create_standard_binding(context.value, expert_desc);

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_3d(context.value, g_backend, input_desc, input_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            input_binding = create_standard_binding(context.value, input_desc);

        ggml_tensor* ids_tensor = ggml_new_tensor_2d(context.value, GGML_TYPE_I32, ids_cols, ids_rows);
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            expert_binding.storage == nullptr || expert_binding.tensor == nullptr ||
            input_binding.storage == nullptr || input_binding.tensor == nullptr ||
            ids_tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml mulmatid tensors.");
            return 0;
        }

        ggml_tensor* value_tensor = ggml_mul_mat_id(context.value, expert_binding.tensor, input_binding.tensor, ids_tensor);
        if (value_tensor == nullptr)
        {
            set_last_error("Failed to create ggml mul_mat_id node.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml mulmatid output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(expert_binding, expert_desc.data, expert_binding.raw_bytes);
            upload_binding(input_binding, input_desc.data, input_binding.raw_bytes);
        }

        std::vector<std::int32_t> ids;
        if (!read_i32_values(ids, ids_desc, "ids"))
        {
            return 0;
        }
        ggml_backend_tensor_set(ids_tensor, ids.data(), 0, ids.size() * sizeof(std::int32_t));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int add_id_f32_impl(
        const TensorView3DDesc& result_desc,
        const TensorView3DDesc& src_desc,
        const TensorView2DDesc& bias_desc,
        const ContiguousTensorDesc& ids_desc,
        int ids_rows,
        int ids_cols)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src") || !validate_desc(bias_desc, "bias") || !validate_desc(ids_desc, "ids"))
        {
            return 0;
        }

        if (ids_rows <= 0 || ids_cols <= 0)
        {
            set_last_error("addid requires positive id matrix dimensions.");
            return 0;
        }

        if (result_desc.dim0 != src_desc.dim0 || result_desc.dim1 != src_desc.dim1 || result_desc.dim2 != src_desc.dim2)
        {
            set_last_error("addid expects result and src to have the same shape.");
            return 0;
        }

        if (src_desc.dim0 != ids_rows || src_desc.dim1 != ids_cols || src_desc.dim2 != bias_desc.dim1)
        {
            set_last_error("addid expects src shape [tokens, expert_used, rows], bias shape [experts, rows], and ids shape [tokens, expert_used].");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc) || !can_map_standard_view(bias_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml addid path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc) && can_map_standard_view(bias_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 2 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        TensorBinding bias_binding;

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_3d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            result_binding = create_standard_binding(context.value, result_desc);

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_3d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            src_binding = create_standard_binding(context.value, src_desc);

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, bias_desc, bias_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            bias_binding = create_standard_binding(context.value, bias_desc);

        ggml_tensor* ids_tensor = ggml_new_tensor_2d(context.value, GGML_TYPE_I32, ids_cols, ids_rows);
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr ||
            bias_binding.storage == nullptr || bias_binding.tensor == nullptr ||
            ids_tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml addid tensors.");
            return 0;
        }

        ggml_tensor* value_tensor = ggml_add_id(context.value, src_binding.tensor, bias_binding.tensor, ids_tensor);
        if (value_tensor == nullptr)
        {
            set_last_error("Failed to create ggml add_id node.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml addid output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            upload_binding(bias_binding, bias_desc.data, bias_binding.raw_bytes);
        }

        std::vector<std::int32_t> ids;
        if (!read_i32_values(ids, ids_desc, "ids"))
        {
            return 0;
        }
        ggml_backend_tensor_set(ids_tensor, ids.data(), 0, ids.size() * sizeof(std::int32_t));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int unary_f32_impl(
        UnaryOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        if (!same_shape(result_desc, src_desc))
        {
            set_last_error("Source tensor shape does not match result shape for unary ggml op.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the unary ggml Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
        }
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to create ggml contiguous unary input.");
            return 0;
        }

        ggml_tensor* value_tensor = make_unary_tensor(context.value, op, contiguous_src);
        if (value_tensor == nullptr)
        {
            if (g_last_error.empty())
            {
                set_last_error("Failed to create ggml unary node.");
            }
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml unary output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int binary_tensor_f32_impl(
        BinaryTensorOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& lhs_desc,
        const TensorView4DDesc& rhs_desc)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(lhs_desc, "lhs") || !validate_desc(rhs_desc, "rhs"))
        {
            return 0;
        }

        if (!same_shape(result_desc, lhs_desc))
        {
            set_last_error("Result tensor shape does not match lhs tensor shape.");
            return 0;
        }

        if (!can_repeat(rhs_desc, lhs_desc))
        {
            set_last_error("rhs tensor shape cannot be broadcast to lhs for ggml binary op.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(lhs_desc) || !can_map_standard_view(rhs_desc))
        {
            set_last_error("Tensor layout is not supported by the binary ggml Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(lhs_desc) && can_map_standard_view(rhs_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 2 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding lhs_binding;
        TensorBinding rhs_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, lhs_desc, lhs_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, rhs_desc, rhs_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            lhs_binding = create_standard_binding(context.value, lhs_desc);
            rhs_binding = create_standard_binding(context.value, rhs_desc);
        }
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            lhs_binding.storage == nullptr || lhs_binding.tensor == nullptr ||
            rhs_binding.storage == nullptr || rhs_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* value_tensor = make_binary_tensor(context.value, op, lhs_binding.tensor, rhs_binding.tensor);
        if (value_tensor == nullptr)
        {
            if (g_last_error.empty())
            {
                set_last_error("Failed to create ggml binary node.");
            }
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml binary output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(lhs_binding, lhs_desc.data, lhs_binding.raw_bytes);
            upload_binding(rhs_binding, rhs_desc.data, rhs_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int fused_act_mul_f32_impl(
        FusedActMulOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& a_desc,
        const TensorView4DDesc& b_desc)
    {
        if (!ensure_backend())
            return 0;

        if (!validate_desc(result_desc, "result") || !validate_desc(a_desc, "a") || !validate_desc(b_desc, "b"))
            return 0;

        if (!same_shape(result_desc, a_desc) || !same_shape(result_desc, b_desc))
        {
            set_last_error("All tensor shapes must match for fused activation-multiply op.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(a_desc) || !can_map_standard_view(b_desc))
        {
            set_last_error("Tensor layout is not supported by the fused activation-multiply ggml path.");
            return 0;
        }

        bool use_zero_copy = true;
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 2 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding, a_binding, b_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, a_desc, a_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, b_desc, b_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            a_binding = create_standard_binding(context.value, a_desc);
            b_binding = create_standard_binding(context.value, b_desc);
        }
        if (result_binding.storage == nullptr || a_binding.storage == nullptr || b_binding.storage == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views for fused op.");
            return 0;
        }

        ggml_tensor* value_tensor = make_fused_act_mul_tensor(context.value, op, a_binding.tensor, b_binding.tensor);
        if (value_tensor == nullptr)
        {
            if (g_last_error.empty())
                set_last_error("Failed to create ggml fused activation-multiply node.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml fused output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(a_binding, a_desc.data, a_binding.raw_bytes);
            upload_binding(b_binding, b_desc.data, b_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int binary_scalar_f32_impl(
        BinaryScalarOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc,
        float scalar)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        if (!same_shape(result_desc, src_desc))
        {
            set_last_error("Result tensor shape does not match source tensor shape for scalar ggml op.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the scalar ggml Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 2 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
        }
        TensorBinding scalar_binding = create_scalar_binding(context.value);
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr ||
            scalar_binding.storage == nullptr || scalar_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to create ggml contiguous scalar-op input.");
            return 0;
        }

        ggml_tensor* value_tensor = nullptr;
        if (op == BinaryScalarOpCode::Mul)
        {
            value_tensor = ggml_scale(context.value, contiguous_src, scalar);
        }
        else
        {
            ggml_tensor* repeated_scalar = ggml_repeat(context.value, scalar_binding.tensor, contiguous_src);
            if (repeated_scalar == nullptr)
            {
                set_last_error("Failed to create repeated scalar tensor.");
                return 0;
            }

            switch (op)
            {
            case BinaryScalarOpCode::Add:
                value_tensor = ggml_add(context.value, contiguous_src, repeated_scalar);
                break;
            case BinaryScalarOpCode::Sub:
                value_tensor = ggml_sub(context.value, contiguous_src, repeated_scalar);
                break;
            case BinaryScalarOpCode::ReverseSub:
                value_tensor = ggml_sub(context.value, repeated_scalar, contiguous_src);
                break;
            case BinaryScalarOpCode::Div:
                value_tensor = ggml_div(context.value, contiguous_src, repeated_scalar);
                break;
            case BinaryScalarOpCode::ReverseDiv:
                value_tensor = ggml_div(context.value, repeated_scalar, contiguous_src);
                break;
            default:
                set_last_error("Unsupported scalar ggml op code.");
                return 0;
            }
        }

        if (value_tensor == nullptr)
        {
            if (g_last_error.empty())
            {
                set_last_error("Failed to create ggml scalar op node.");
            }
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml scalar-op output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }
        if (op != BinaryScalarOpCode::Mul)
            ggml_backend_tensor_set(scalar_binding.storage, &scalar, 0, sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int activation_grad_f32_impl(
        ActivationGradOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc,
        const TensorView4DDesc& grad_desc,
        const TensorView4DDesc& accumulation_desc,
        bool has_accumulation)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src") || !validate_desc(grad_desc, "grad"))
        {
            return 0;
        }

        if (has_accumulation && !validate_desc(accumulation_desc, "accumulation"))
        {
            return 0;
        }

        if (!same_shape(result_desc, src_desc) || !same_shape(src_desc, grad_desc) ||
            (has_accumulation && !same_shape(src_desc, accumulation_desc)))
        {
            set_last_error("Tensor shape mismatch passed to ggml activation grad.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc) || !can_map_standard_view(grad_desc) ||
            (has_accumulation && !can_map_standard_view(accumulation_desc)))
        {
            set_last_error("Tensor layout is not supported by the ggml activation-grad Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc) && can_map_standard_view(grad_desc) &&
            (!has_accumulation || can_map_standard_view(accumulation_desc));
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        TensorBinding grad_binding;
        TensorBinding accumulation_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, grad_desc, grad_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy && has_accumulation)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, accumulation_desc, accumulation_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
            grad_binding = create_standard_binding(context.value, grad_desc);
            if (has_accumulation)
                accumulation_binding = create_standard_binding(context.value, accumulation_desc);
        }
        TensorBinding one_binding = create_scalar_binding(context.value);
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr ||
            grad_binding.storage == nullptr || grad_binding.tensor == nullptr ||
            one_binding.storage == nullptr || one_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        if (has_accumulation && (accumulation_binding.storage == nullptr || accumulation_binding.tensor == nullptr))
        {
            set_last_error("Failed to allocate ggml accumulation tensor.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        ggml_tensor* contiguous_grad = ggml_cont(context.value, grad_binding.tensor);
        if (contiguous_src == nullptr || contiguous_grad == nullptr)
        {
            set_last_error("Failed to create ggml contiguous activation-grad inputs.");
            return 0;
        }

        ggml_tensor* value_tensor = nullptr;
        switch (op)
        {
        case ActivationGradOpCode::Relu:
        {
            ggml_tensor* step_tensor = ggml_step(context.value, contiguous_src);
            if (step_tensor != nullptr)
            {
                value_tensor = ggml_mul(context.value, step_tensor, contiguous_grad);
            }
        } break;
        case ActivationGradOpCode::Sigmoid:
        {
            ggml_tensor* one_tensor = ggml_repeat(context.value, one_binding.tensor, contiguous_src);
            ggml_tensor* one_minus = one_tensor == nullptr ? nullptr : ggml_sub(context.value, one_tensor, contiguous_src);
            ggml_tensor* deriv_tensor = one_minus == nullptr ? nullptr : ggml_mul(context.value, contiguous_src, one_minus);
            value_tensor = deriv_tensor == nullptr ? nullptr : ggml_mul(context.value, deriv_tensor, contiguous_grad);
        } break;
        case ActivationGradOpCode::Tanh:
        {
            ggml_tensor* one_tensor = ggml_repeat(context.value, one_binding.tensor, contiguous_src);
            ggml_tensor* sq_tensor = ggml_mul(context.value, contiguous_src, contiguous_src);
            ggml_tensor* one_minus = (one_tensor == nullptr || sq_tensor == nullptr) ? nullptr : ggml_sub(context.value, one_tensor, sq_tensor);
            value_tensor = one_minus == nullptr ? nullptr : ggml_mul(context.value, one_minus, contiguous_grad);
        } break;
        case ActivationGradOpCode::SiLU:
        {
            value_tensor = ggml_silu_back(context.value, contiguous_grad, contiguous_src);
            if (!backend_supports_op(value_tensor))
            {
                ggml_tensor* one_tensor = ggml_repeat(context.value, one_binding.tensor, contiguous_src);
                ggml_tensor* sig_tensor = ggml_sigmoid(context.value, contiguous_src);
                ggml_tensor* one_minus_sig = (one_tensor == nullptr || sig_tensor == nullptr) ? nullptr : ggml_sub(context.value, one_tensor, sig_tensor);
                ggml_tensor* weighted_tensor = one_minus_sig == nullptr ? nullptr : ggml_mul(context.value, contiguous_src, one_minus_sig);
                ggml_tensor* inner_tensor = (one_tensor == nullptr || weighted_tensor == nullptr) ? nullptr : ggml_add(context.value, one_tensor, weighted_tensor);
                ggml_tensor* deriv_tensor = (sig_tensor == nullptr || inner_tensor == nullptr) ? nullptr : ggml_mul(context.value, sig_tensor, inner_tensor);
                value_tensor = deriv_tensor == nullptr ? nullptr : ggml_mul(context.value, deriv_tensor, contiguous_grad);
            }
        } break;
        default:
            set_last_error("Unsupported activation-grad ggml op code.");
            return 0;
        }

        if (value_tensor == nullptr)
        {
            set_last_error("Failed to create ggml activation-grad node.");
            return 0;
        }

        if (has_accumulation)
        {
            ggml_tensor* contiguous_accumulation = ggml_cont(context.value, accumulation_binding.tensor);
            if (contiguous_accumulation == nullptr)
            {
                set_last_error("Failed to create ggml contiguous accumulation input.");
                return 0;
            }

            value_tensor = ggml_add(context.value, contiguous_accumulation, value_tensor);
            if (value_tensor == nullptr)
            {
                set_last_error("Failed to create ggml activation-grad accumulation node.");
                return 0;
            }
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml activation-grad output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            upload_binding(grad_binding, grad_desc.data, grad_binding.raw_bytes);
            if (has_accumulation)
                upload_binding(accumulation_binding, accumulation_desc.data, accumulation_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }
        const float one_value = 1.0f;
        ggml_backend_tensor_set(one_binding.storage, &one_value, 0, sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int norm_f32_impl(
        NormOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc,
        const TensorView4DDesc& gamma_desc,
        const TensorView4DDesc& beta_desc,
        bool has_beta,
        float eps)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src") || !validate_desc(gamma_desc, "gamma"))
        {
            return 0;
        }

        if (has_beta && !validate_desc(beta_desc, "beta"))
        {
            return 0;
        }

        if (!same_shape(result_desc, src_desc))
        {
            set_last_error("Result tensor shape does not match source tensor shape for ggml norm op.");
            return 0;
        }

        if (!can_repeat(gamma_desc, src_desc) || (has_beta && !can_repeat(beta_desc, src_desc)))
        {
            set_last_error("gamma/beta tensor shape cannot be broadcast to source tensor for ggml norm op.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc) || !can_map_standard_view(gamma_desc) ||
            (has_beta && !can_map_standard_view(beta_desc)))
        {
            set_last_error("Tensor layout is not supported by the ggml norm Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc) &&
            can_map_standard_view(gamma_desc) && (!has_beta || can_map_standard_view(beta_desc));
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 3 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        TensorBinding gamma_binding;
        TensorBinding beta_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, gamma_desc, gamma_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy && has_beta)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, beta_desc, beta_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
            gamma_binding = create_standard_binding(context.value, gamma_desc);
            if (has_beta)
                beta_binding = create_standard_binding(context.value, beta_desc);
        }
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr ||
            gamma_binding.storage == nullptr || gamma_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        if (has_beta && (beta_binding.storage == nullptr || beta_binding.tensor == nullptr))
        {
            set_last_error("Failed to allocate ggml beta tensor.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        ggml_tensor* contiguous_gamma = ggml_cont(context.value, gamma_binding.tensor);
        if (contiguous_src == nullptr || contiguous_gamma == nullptr)
        {
            set_last_error("Failed to create ggml contiguous norm inputs.");
            return 0;
        }

        ggml_tensor* value_tensor = make_norm_tensor(context.value, op, contiguous_src, eps);
        if (value_tensor == nullptr)
        {
            if (g_last_error.empty())
            {
                set_last_error("Failed to create ggml norm node.");
            }
            return 0;
        }

        value_tensor = ggml_mul(context.value, value_tensor, contiguous_gamma);
        if (value_tensor == nullptr)
        {
            set_last_error("Failed to create ggml norm scale node.");
            return 0;
        }

        if (has_beta)
        {
            ggml_tensor* contiguous_beta = ggml_cont(context.value, beta_binding.tensor);
            if (contiguous_beta == nullptr)
            {
                set_last_error("Failed to create ggml contiguous beta tensor.");
                return 0;
            }

            value_tensor = ggml_add(context.value, value_tensor, contiguous_beta);
            if (value_tensor == nullptr)
            {
                set_last_error("Failed to create ggml norm bias node.");
                return 0;
            }
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml norm output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            upload_binding(gamma_binding, gamma_desc.data, gamma_binding.raw_bytes);
            if (has_beta)
                upload_binding(beta_binding, beta_desc.data, beta_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int norm_grad_f32_impl(
        NormOpCode op,
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& grad_gamma_desc,
        const TensorView4DDesc& grad_beta_desc,
        const TensorView4DDesc& adj_desc,
        const TensorView4DDesc& x_desc,
        const TensorView4DDesc& gamma_desc,
        bool has_grad_beta,
        float eps)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result")
            || !validate_desc(grad_gamma_desc, "gradGamma")
            || !validate_desc(adj_desc, "adj")
            || !validate_desc(x_desc, "x")
            || !validate_desc(gamma_desc, "gamma"))
        {
            return 0;
        }

        if (has_grad_beta && !validate_desc(grad_beta_desc, "gradBeta"))
        {
            return 0;
        }

        if (!same_shape(result_desc, adj_desc) || !same_shape(adj_desc, x_desc))
        {
            set_last_error("Tensor shape mismatch passed to ggml norm grad.");
            return 0;
        }

        if (!is_vector_like(gamma_desc, x_desc.ne0) || !is_vector_like(grad_gamma_desc, x_desc.ne0) || (has_grad_beta && !is_vector_like(grad_beta_desc, x_desc.ne0)))
        {
            set_last_error("gamma/gradGamma/gradBeta must match the last source dimension for ggml norm grad.");
            return 0;
        }

        if (!can_map_standard_view(result_desc)
            || !can_map_standard_view(grad_gamma_desc)
            || !can_map_standard_view(adj_desc)
            || !can_map_standard_view(x_desc)
            || !can_map_standard_view(gamma_desc)
            || (has_grad_beta && !can_map_standard_view(grad_beta_desc)))
        {
            set_last_error("Tensor layout is not supported by the ggml norm-grad Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(grad_gamma_desc) &&
            can_map_standard_view(adj_desc) && can_map_standard_view(x_desc) && can_map_standard_view(gamma_desc) &&
            (!has_grad_beta || can_map_standard_view(grad_beta_desc));
        std::vector<BufferHandle> host_ptr_buffers;
        constexpr size_t graph_capacity = 512;
        const std::size_t ctx_size = 16 * 1024 * 1024 + ggml_graph_overhead_custom(graph_capacity, true);

        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding grad_gamma_binding;
        TensorBinding adj_binding;
        TensorBinding x_binding;
        TensorBinding gamma_binding;
        TensorBinding grad_beta_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, grad_gamma_desc, grad_gamma_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, adj_desc, adj_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, x_desc, x_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, gamma_desc, gamma_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy && has_grad_beta)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, grad_beta_desc, grad_beta_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            grad_gamma_binding = create_standard_binding(context.value, grad_gamma_desc);
            adj_binding = create_standard_binding(context.value, adj_desc);
            x_binding = create_standard_binding(context.value, x_desc);
            gamma_binding = create_standard_binding(context.value, gamma_desc);
            if (has_grad_beta)
                grad_beta_binding = create_standard_binding(context.value, grad_beta_desc);
        }
        TensorBinding eps_binding = create_scalar_binding(context.value);

        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            grad_gamma_binding.storage == nullptr || grad_gamma_binding.tensor == nullptr ||
            adj_binding.storage == nullptr || adj_binding.tensor == nullptr ||
            x_binding.storage == nullptr || x_binding.tensor == nullptr ||
            gamma_binding.storage == nullptr || gamma_binding.tensor == nullptr ||
            eps_binding.storage == nullptr || eps_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        if (has_grad_beta && (grad_beta_binding.storage == nullptr || grad_beta_binding.tensor == nullptr))
        {
            set_last_error("Failed to allocate ggml gradBeta tensor.");
            return 0;
        }

        ggml_tensor* contiguous_result = ggml_cont(context.value, result_binding.tensor);
        ggml_tensor* contiguous_grad_gamma = ggml_cont(context.value, grad_gamma_binding.tensor);
        ggml_tensor* contiguous_adj = ggml_cont(context.value, adj_binding.tensor);
        ggml_tensor* contiguous_x = ggml_cont(context.value, x_binding.tensor);
        ggml_tensor* contiguous_gamma = ggml_cont(context.value, gamma_binding.tensor);
        ggml_tensor* contiguous_grad_beta = nullptr;
        if (has_grad_beta)
        {
            contiguous_grad_beta = ggml_cont(context.value, grad_beta_binding.tensor);
        }

        if (contiguous_result == nullptr || contiguous_grad_gamma == nullptr || contiguous_adj == nullptr || contiguous_x == nullptr || contiguous_gamma == nullptr ||
            (has_grad_beta && contiguous_grad_beta == nullptr))
        {
            set_last_error("Failed to create ggml contiguous norm-grad inputs.");
            return 0;
        }

        if (op == NormOpCode::LayerNorm)
        {
            ggml_set_param(x_binding.storage);

            ggml_tensor* norm_value = ggml_norm(context.value, contiguous_x, eps);
            ggml_tensor* scaled_value = norm_value == nullptr ? nullptr : ggml_mul(context.value, norm_value, contiguous_gamma);
            ggml_tensor* weighted_value = scaled_value == nullptr ? nullptr : ggml_mul(context.value, scaled_value, contiguous_adj);
            ggml_tensor* loss_tensor = weighted_value == nullptr ? nullptr : ggml_sum(context.value, weighted_value);
            if (loss_tensor == nullptr)
            {
                set_last_error("Failed to create ggml layernorm backward loss graph.");
                return 0;
            }
            ggml_set_loss(loss_tensor);

            ggml_cgraph* graph = ggml_new_graph_custom(context.value, graph_capacity, true);
            if (graph == nullptr)
            {
                set_last_error("Failed to create ggml backward graph.");
                return 0;
            }

            ggml_build_forward_expand(graph, loss_tensor);
            ggml_build_backward_expand(context.value, graph, nullptr);

            ggml_tensor* dx_delta = ggml_graph_get_grad(graph, contiguous_x);
            if (dx_delta == nullptr)
            {
                set_last_error("Failed to obtain ggml layernorm input gradient.");
                return 0;
            }

            const std::int64_t rows = flat_row_count(x_desc);
            ggml_tensor* flat_adj = flatten_to_rows(context.value, contiguous_adj, x_desc.ne0, rows);
            ggml_tensor* flat_norm = norm_value == nullptr ? nullptr : flatten_to_rows(context.value, norm_value, x_desc.ne0, rows);
            ggml_tensor* flat_grad_gamma = flatten_to_rows(context.value, contiguous_grad_gamma, x_desc.ne0, 1);
            ggml_tensor* flat_grad_beta = has_grad_beta ? flatten_to_rows(context.value, contiguous_grad_beta, x_desc.ne0, 1) : nullptr;
            if (flat_adj == nullptr || flat_norm == nullptr || flat_grad_gamma == nullptr || (has_grad_beta && flat_grad_beta == nullptr))
            {
                set_last_error("Failed to reshape ggml layernorm gradient tensors.");
                return 0;
            }

            ggml_tensor* adj_norm = ggml_mul(context.value, flat_adj, flat_norm);
            ggml_tensor* grad_gamma_delta = adj_norm == nullptr ? nullptr : sum_rows_to_feature_vector(context.value, adj_norm);
            ggml_tensor* grad_beta_delta = has_grad_beta ? sum_rows_to_feature_vector(context.value, flat_adj) : nullptr;
            if (grad_gamma_delta == nullptr || (has_grad_beta && grad_beta_delta == nullptr))
            {
                set_last_error("Failed to create ggml layernorm parameter gradients.");
                return 0;
            }

            ggml_tensor* dx_value = ggml_add(context.value, contiguous_result, dx_delta);
            ggml_tensor* grad_gamma_value = ggml_add(context.value, flat_grad_gamma, grad_gamma_delta);
            ggml_tensor* grad_gamma_view = grad_gamma_value == nullptr ? nullptr : ggml_reshape_4d(context.value, grad_gamma_value, grad_gamma_desc.ne0, grad_gamma_desc.ne1, grad_gamma_desc.ne2, grad_gamma_desc.ne3);
            ggml_tensor* grad_beta_value = has_grad_beta ? ggml_add(context.value, flat_grad_beta, grad_beta_delta) : nullptr;
            ggml_tensor* grad_beta_view = has_grad_beta && grad_beta_value != nullptr
                ? ggml_reshape_4d(context.value, grad_beta_value, grad_beta_desc.ne0, grad_beta_desc.ne1, grad_beta_desc.ne2, grad_beta_desc.ne3)
                : nullptr;
            ggml_tensor* dx_output = dx_value == nullptr ? nullptr : ggml_cpy(context.value, dx_value, result_binding.tensor);
            ggml_tensor* grad_gamma_output = grad_gamma_view == nullptr ? nullptr : ggml_cpy(context.value, grad_gamma_view, grad_gamma_binding.tensor);
            ggml_tensor* grad_beta_output = has_grad_beta
                ? (grad_beta_view == nullptr ? nullptr : ggml_cpy(context.value, grad_beta_view, grad_beta_binding.tensor))
                : nullptr;
            if (dx_output == nullptr || grad_gamma_output == nullptr || (has_grad_beta && grad_beta_output == nullptr))
            {
                set_last_error("Failed to create ggml layernorm output copy nodes.");
                return 0;
            }

            ggml_set_output(dx_output);
            ggml_set_output(grad_gamma_output);
            if (has_grad_beta)
            {
                ggml_set_output(grad_beta_output);
            }

            ggml_build_forward_expand(graph, dx_output);
            ggml_build_forward_expand(graph, grad_gamma_output);
            if (has_grad_beta)
            {
                ggml_build_forward_expand(graph, grad_beta_output);
            }

            BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
            if (buffer.value == nullptr)
            {
                set_last_error("Failed to allocate ggml backend buffer.");
                return 0;
            }

            if (!use_zero_copy)
            {
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
                upload_binding(grad_gamma_binding, grad_gamma_desc.data, grad_gamma_binding.raw_bytes);
                upload_binding(adj_binding, adj_desc.data, adj_binding.raw_bytes);
                upload_binding(x_binding, x_desc.data, x_binding.raw_bytes);
                upload_binding(gamma_binding, gamma_desc.data, gamma_binding.raw_bytes);
                if (has_grad_beta)
                    upload_binding(grad_beta_binding, grad_beta_desc.data, grad_beta_binding.raw_bytes);
            }

            ggml_graph_reset(graph);

            ggml_status status = ggml_backend_graph_compute(g_backend, graph);
            if (status != GGML_STATUS_SUCCESS)
            {
                set_last_error("ggml backend graph execution failed.");
                return 0;
            }

            ggml_backend_synchronize(g_backend);
            if (!use_zero_copy)
            {
                ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);
                ggml_backend_tensor_get(grad_gamma_binding.storage, grad_gamma_desc.data, 0, grad_gamma_binding.raw_bytes);
                if (has_grad_beta)
                    ggml_backend_tensor_get(grad_beta_binding.storage, grad_beta_desc.data, 0, grad_beta_binding.raw_bytes);
            }

            clear_last_error();
            return 1;
        }

        const std::int64_t rows = flat_row_count(x_desc);
        const float inv_cols = 1.0f / static_cast<float>(x_desc.ne0);
        const float cols_value = static_cast<float>(x_desc.ne0);

        ggml_tensor* flat_adj = flatten_to_rows(context.value, contiguous_adj, x_desc.ne0, rows);
        ggml_tensor* flat_x = flatten_to_rows(context.value, contiguous_x, x_desc.ne0, rows);
        ggml_tensor* flat_gamma = flatten_to_rows(context.value, contiguous_gamma, x_desc.ne0, 1);
        ggml_tensor* flat_grad_gamma = flatten_to_rows(context.value, contiguous_grad_gamma, x_desc.ne0, 1);
        ggml_tensor* flat_grad_beta = has_grad_beta ? flatten_to_rows(context.value, contiguous_grad_beta, x_desc.ne0, 1) : nullptr;
        if (flat_adj == nullptr || flat_x == nullptr || flat_gamma == nullptr || flat_grad_gamma == nullptr || (has_grad_beta && flat_grad_beta == nullptr))
        {
            set_last_error("Failed to reshape ggml norm-grad tensors.");
            return 0;
        }

        ggml_tensor* dx_delta_flat = nullptr;
        ggml_tensor* grad_gamma_delta = nullptr;
        ggml_tensor* grad_beta_delta = nullptr;

        switch (op)
        {
        case NormOpCode::RmsNorm:
        {
            ggml_tensor* native_adj = ggml_mul(context.value, contiguous_adj, contiguous_gamma);
            ggml_tensor* native_dx = native_adj == nullptr ? nullptr : ggml_rms_norm_back(context.value, native_adj, contiguous_x, eps);
            if (backend_supports_op(native_dx))
            {
                dx_delta_flat = flatten_to_rows(context.value, native_dx, x_desc.ne0, rows);
            }

            ggml_tensor* sq = ggml_mul(context.value, flat_x, flat_x);
            ggml_tensor* sq_sum = sq == nullptr ? nullptr : ggml_sum_rows(context.value, sq);
            ggml_tensor* mean_sq = sq_sum == nullptr ? nullptr : ggml_scale(context.value, sq_sum, inv_cols);
            ggml_tensor* eps_full = mean_sq == nullptr ? nullptr : ggml_repeat(context.value, eps_binding.tensor, mean_sq);
            ggml_tensor* rms_sq = (mean_sq == nullptr || eps_full == nullptr) ? nullptr : ggml_add(context.value, mean_sq, eps_full);
            ggml_tensor* rms = rms_sq == nullptr ? nullptr : ggml_sqrt(context.value, rms_sq);
            ggml_tensor* rms_full = rms == nullptr ? nullptr : ggml_repeat(context.value, rms, flat_x);
            ggml_tensor* rms_norm = rms_full == nullptr ? nullptr : ggml_div(context.value, flat_x, rms_full);
            ggml_tensor* adj_rms_norm = rms_norm == nullptr ? nullptr : ggml_mul(context.value, flat_adj, rms_norm);
            ggml_tensor* sum_adj_rms_norm = adj_rms_norm == nullptr ? nullptr : ggml_sum_rows(context.value, adj_rms_norm);
            ggml_tensor* sum_adj_rms_norm_full = sum_adj_rms_norm == nullptr ? nullptr : ggml_repeat(context.value, sum_adj_rms_norm, flat_x);
            ggml_tensor* weighted = (rms_norm == nullptr || sum_adj_rms_norm_full == nullptr) ? nullptr : ggml_mul(context.value, rms_norm, sum_adj_rms_norm_full);
            ggml_tensor* scaled_adj = ggml_scale(context.value, flat_adj, cols_value);
            ggml_tensor* dx_numerator = (scaled_adj == nullptr || weighted == nullptr) ? nullptr : ggml_sub(context.value, scaled_adj, weighted);
            ggml_tensor* dx_denominator = rms_full == nullptr ? nullptr : ggml_scale(context.value, rms_full, cols_value);
            ggml_tensor* dx_core = (dx_numerator == nullptr || dx_denominator == nullptr) ? nullptr : ggml_div(context.value, dx_numerator, dx_denominator);
            ggml_tensor* unclamped = (dx_core == nullptr) ? nullptr : ggml_mul(context.value, dx_core, flat_gamma);

            if (dx_delta_flat == nullptr)
            {
                dx_delta_flat = unclamped == nullptr ? nullptr : ggml_clamp(context.value, unclamped, -1000.0f, 1000.0f);
            }
            grad_gamma_delta = adj_rms_norm == nullptr ? nullptr : sum_rows_to_feature_vector(context.value, adj_rms_norm);
            if (has_grad_beta)
            {
                grad_beta_delta = sum_rows_to_feature_vector(context.value, flat_adj);
            }
        } break;
        default:
            set_last_error("Unsupported norm-grad ggml op code.");
            return 0;
        }

        if (dx_delta_flat == nullptr || grad_gamma_delta == nullptr || (has_grad_beta && grad_beta_delta == nullptr))
        {
            set_last_error("Failed to create ggml norm-grad intermediate tensors.");
            return 0;
        }

        ggml_tensor* dx_delta = ggml_reshape_4d(context.value, dx_delta_flat, result_desc.ne0, result_desc.ne1, result_desc.ne2, result_desc.ne3);
        ggml_tensor* dx_value = dx_delta == nullptr ? nullptr : ggml_add(context.value, contiguous_result, dx_delta);
        ggml_tensor* grad_gamma_value = ggml_add(context.value, flat_grad_gamma, grad_gamma_delta);
        ggml_tensor* grad_gamma_view = grad_gamma_value == nullptr ? nullptr : ggml_reshape_4d(context.value, grad_gamma_value, grad_gamma_desc.ne0, grad_gamma_desc.ne1, grad_gamma_desc.ne2, grad_gamma_desc.ne3);
        ggml_tensor* grad_beta_value = nullptr;
        ggml_tensor* grad_beta_view = nullptr;
        if (has_grad_beta)
        {
            grad_beta_value = ggml_add(context.value, flat_grad_beta, grad_beta_delta);
            grad_beta_view = grad_beta_value == nullptr ? nullptr : ggml_reshape_4d(context.value, grad_beta_value, grad_beta_desc.ne0, grad_beta_desc.ne1, grad_beta_desc.ne2, grad_beta_desc.ne3);
        }

        if (dx_value == nullptr || grad_gamma_view == nullptr || (has_grad_beta && grad_beta_view == nullptr))
        {
            set_last_error("Failed to create ggml norm-grad accumulation tensors.");
            return 0;
        }

        ggml_tensor* dx_output = ggml_cpy(context.value, dx_value, result_binding.tensor);
        ggml_tensor* grad_gamma_output = ggml_cpy(context.value, grad_gamma_view, grad_gamma_binding.tensor);
        ggml_tensor* grad_beta_output = has_grad_beta ? ggml_cpy(context.value, grad_beta_view, grad_beta_binding.tensor) : nullptr;
        if (dx_output == nullptr || grad_gamma_output == nullptr || (has_grad_beta && grad_beta_output == nullptr))
        {
            set_last_error("Failed to create ggml norm-grad output copy nodes.");
            return 0;
        }

        ggml_set_output(dx_output);
        ggml_set_output(grad_gamma_output);
        if (has_grad_beta)
        {
            ggml_set_output(grad_beta_output);
        }

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, dx_output);
        ggml_build_forward_expand(graph, grad_gamma_output);
        if (has_grad_beta)
        {
            ggml_build_forward_expand(graph, grad_beta_output);
        }

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
            upload_binding(grad_gamma_binding, grad_gamma_desc.data, grad_gamma_binding.raw_bytes);
            upload_binding(adj_binding, adj_desc.data, adj_binding.raw_bytes);
            upload_binding(x_binding, x_desc.data, x_binding.raw_bytes);
            upload_binding(gamma_binding, gamma_desc.data, gamma_binding.raw_bytes);
            if (has_grad_beta)
                upload_binding(grad_beta_binding, grad_beta_desc.data, grad_beta_binding.raw_bytes);
        }
        ggml_backend_tensor_set(eps_binding.storage, &eps, 0, sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
        {
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);
            ggml_backend_tensor_get(grad_gamma_binding.storage, grad_gamma_desc.data, 0, grad_gamma_binding.raw_bytes);
            if (has_grad_beta)
                ggml_backend_tensor_get(grad_beta_binding.storage, grad_beta_desc.data, 0, grad_beta_binding.raw_bytes);
        }

        clear_last_error();
        return 1;
    }

    int index_select_f32_impl(
        const TensorView2DDesc& result_desc,
        const TensorView2DDesc& src_desc,
        const ContiguousTensorDesc& indices_desc,
        bool add_to_result)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src") || !validate_desc(indices_desc, "indices"))
        {
            return 0;
        }

        if (result_desc.dim1 != src_desc.dim1 || indices_desc.element_count != result_desc.dim0)
        {
            set_last_error("Tensor shape mismatch passed to ggml indexselect.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml indexselect Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 2 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
        }
        ggml_tensor* index_tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_I32, indices_desc.element_count);
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr ||
            index_tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to create ggml contiguous indexselect input.");
            return 0;
        }

        ggml_tensor* value_tensor = ggml_get_rows(context.value, contiguous_src, index_tensor);
        if (value_tensor == nullptr)
        {
            set_last_error("Failed to create ggml get_rows node.");
            return 0;
        }

        if (add_to_result)
        {
            ggml_tensor* contiguous_result = ggml_cont(context.value, result_binding.tensor);
            if (contiguous_result == nullptr)
            {
                set_last_error("Failed to create ggml contiguous indexselect accumulation input.");
                return 0;
            }

            value_tensor = ggml_add(context.value, value_tensor, contiguous_result);
            if (value_tensor == nullptr)
            {
                set_last_error("Failed to create ggml indexselect accumulation node.");
                return 0;
            }
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml indexselect output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (add_to_result || result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        std::vector<std::int32_t> indices;
        if (!read_i32_values(indices, indices_desc, "indices"))
        {
            return 0;
        }
        ggml_backend_tensor_set(index_tensor, indices.data(), 0, indices.size() * sizeof(std::int32_t));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int index_select_grad_f32_impl(
        const TensorView2DDesc& grad_desc,
        const TensorView2DDesc& adj_desc,
        const ContiguousTensorDesc& indices_desc)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(grad_desc, "grad") || !validate_desc(adj_desc, "adj") || !validate_desc(indices_desc, "indices"))
        {
            return 0;
        }

        if (adj_desc.dim0 != indices_desc.element_count || grad_desc.dim1 != adj_desc.dim1)
        {
            set_last_error("Tensor shape mismatch passed to ggml indexselectgrad.");
            return 0;
        }

        if (!can_map_standard_view(grad_desc) || !can_map_standard_view(adj_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml indexselectgrad Metal path.");
            return 0;
        }

        std::vector<std::int32_t> indices;
        std::size_t active_row_count = 0;
        if (!read_i32_values(indices, indices_desc, "indices"))
        {
            return 0;
        }

        for (std::size_t i = 0; i < indices.size(); ++i)
        {
            if (indices[i] >= 0)
            {
                ++active_row_count;
            }
        }

        bool use_zero_copy = can_map_standard_view(grad_desc) && can_map_standard_view(adj_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t min_graph_capacity = static_cast<std::size_t>(GGML_DEFAULT_GRAPH_SIZE) * 8;
        const std::size_t estimated_graph_capacity = active_row_count * 6 + 64;
        const std::size_t graph_capacity = estimated_graph_capacity > min_graph_capacity ? estimated_graph_capacity : min_graph_capacity;

        const std::size_t ctx_size = 16 * 1024 * 1024 + ggml_graph_overhead_custom(graph_capacity, false);

        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding grad_binding;
        TensorBinding adj_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, grad_desc, grad_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_2d(context.value, g_backend, adj_desc, adj_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            grad_binding = create_standard_binding(context.value, grad_desc);
            adj_binding = create_standard_binding(context.value, adj_desc);
        }
        if (grad_binding.storage == nullptr || grad_binding.tensor == nullptr ||
            adj_binding.storage == nullptr || adj_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        ggml_tensor* working_grad = ggml_cont(context.value, grad_binding.tensor);
        ggml_tensor* contiguous_adj = ggml_cont(context.value, adj_binding.tensor);
        if (working_grad == nullptr || contiguous_adj == nullptr)
        {
            set_last_error("Failed to create ggml contiguous indexselectgrad inputs.");
            return 0;
        }

        struct PendingIndexUpload
        {
            ggml_tensor* tensor;
            std::int32_t value;
        };

        std::vector<PendingIndexUpload> pending_index_uploads;
        pending_index_uploads.reserve(indices.size());

        const std::size_t row_bytes = static_cast<std::size_t>(adj_desc.dim1) * sizeof(float);
        for (std::size_t row = 0; row < indices.size(); ++row)
        {
            if (indices[row] < 0)
            {
                continue;
            }

            ggml_tensor* index_tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_I32, 1);
            ggml_tensor* current_row = index_tensor == nullptr ? nullptr : ggml_get_rows(context.value, working_grad, index_tensor);
            ggml_tensor* adj_row = current_row == nullptr ? nullptr : ggml_view_2d(
                context.value,
                contiguous_adj,
                adj_desc.dim1,
                1,
                row_bytes,
                row * row_bytes);
            ggml_tensor* updated_row = (current_row == nullptr || adj_row == nullptr) ? nullptr : ggml_add(context.value, current_row, adj_row);
            ggml_tensor* updated_grad = (updated_row == nullptr) ? nullptr : ggml_set_rows(context.value, working_grad, updated_row, index_tensor);

            if (index_tensor == nullptr || current_row == nullptr || adj_row == nullptr || updated_row == nullptr || updated_grad == nullptr)
            {
                set_last_error("Failed to create ggml indexselectgrad scatter-add node.");
                return 0;
            }

            pending_index_uploads.push_back({ index_tensor, indices[row] });
            working_grad = updated_grad;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, working_grad, grad_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml indexselectgrad output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph_custom(context.value, graph_capacity, false);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(grad_binding, grad_desc.data, grad_binding.raw_bytes);
            upload_binding(adj_binding, adj_desc.data, adj_binding.raw_bytes);
        }
        for (const PendingIndexUpload& upload : pending_index_uploads)
        {
            ggml_backend_tensor_set(upload.tensor, &upload.value, 0, sizeof(upload.value));
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(grad_binding.storage, grad_desc.data, 0, grad_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int rope_f32_impl(
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc,
        int seq_len,
        int row_offset,
        bool add_to_result,
        bool invert_positions)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        if (seq_len <= 0)
        {
            set_last_error("seqLen must be positive for ggml rope.");
            return 0;
        }

        if (!same_shape(result_desc, src_desc))
        {
            set_last_error("Source tensor shape does not match result shape for ggml rope.");
            return 0;
        }

        if ((src_desc.ne0 % 2) != 0)
        {
            set_last_error("ggml rope requires an even embedding dimension.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml rope Metal path.");
            return 0;
        }

        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding = create_standard_binding(context.value, result_desc);
        TensorBinding src_binding = create_standard_binding(context.value, src_desc);
        ggml_tensor* position_tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_I32, flat_row_count(src_desc));
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr ||
            position_tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        ggml_tensor* contiguous_result = add_to_result ? ggml_cont(context.value, result_binding.tensor) : nullptr;
        if (contiguous_src == nullptr || (add_to_result && contiguous_result == nullptr))
        {
            set_last_error("Failed to create ggml contiguous rope inputs.");
            return 0;
        }

        const std::int64_t rows = flat_row_count(src_desc);
        ggml_tensor* rope_input = ggml_reshape_4d(context.value, contiguous_src, src_desc.ne0, 1, rows, 1);
        ggml_tensor* rope_tensor = nullptr;
        bool use_native_backward = false;
        if (rope_input != nullptr && invert_positions)
        {
            ggml_tensor* native_backward = ggml_rope_ext_back(
                context.value,
                rope_input,
                position_tensor,
                nullptr,
                src_desc.ne0,
                0,
                0,
                500000.0f,
                1.0f,
                0.0f,
                1.0f,
                0.0f,
                0.0f);
            if (backend_supports_op(native_backward))
            {
                rope_tensor = native_backward;
                use_native_backward = true;
            }
        }

        if (rope_tensor == nullptr)
        {
            rope_tensor = rope_input == nullptr ? nullptr : ggml_rope_ext(
                context.value,
                rope_input,
                position_tensor,
                nullptr,
                src_desc.ne0,
                0,
                0,
                500000.0f,
                1.0f,
                0.0f,
                1.0f,
                0.0f,
                0.0f);
        }
        ggml_tensor* restored = rope_tensor == nullptr ? nullptr : ggml_reshape_4d(context.value, rope_tensor, result_desc.ne0, result_desc.ne1, result_desc.ne2, result_desc.ne3);
        ggml_tensor* value_tensor = restored;
        if (add_to_result)
        {
            value_tensor = restored == nullptr ? nullptr : ggml_add(context.value, contiguous_result, restored);
        }

        if (rope_input == nullptr || rope_tensor == nullptr || value_tensor == nullptr)
        {
            set_last_error("Failed to create ggml rope node.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml rope output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);

        std::vector<std::int32_t> positions(static_cast<std::size_t>(rows));
        for (std::size_t i = 0; i < positions.size(); ++i)
        {
            std::int32_t position = static_cast<std::int32_t>(row_offset + static_cast<int>(i % static_cast<std::size_t>(seq_len)));
            positions[i] = (invert_positions && !use_native_backward) ? -position : position;
        }
        ggml_backend_tensor_set(position_tensor, positions.data(), 0, positions.size() * sizeof(std::int32_t));

        if (add_to_result || result_binding.raw_bytes > logical_bytes(result_desc))
        {
            upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int rope_ex_f32_impl(
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc,
        const ContiguousTensorDesc& positions_desc,
        int rope_dim,
        int mode,
        int original_context_length,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        bool add_to_result,
        bool invert_positions)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src") || !validate_desc(positions_desc, "positions"))
        {
            return 0;
        }

        if (!same_shape(result_desc, src_desc))
        {
            set_last_error("Source tensor shape does not match result shape for ggml rope_ex.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml rope_ex Metal path.");
            return 0;
        }

        if (rope_dim <= 0 || rope_dim > src_desc.ne0 || (rope_dim % 2) != 0)
        {
            set_last_error("rope_dim must be positive, even, and within the source embedding dimension.");
            return 0;
        }

        const std::int64_t rows = flat_row_count(src_desc);
        if (positions_desc.element_count != rows)
        {
            set_last_error("rope_ex expects one position per logical row.");
            return 0;
        }

        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding = create_standard_binding(context.value, result_desc);
        TensorBinding src_binding = create_standard_binding(context.value, src_desc);
        ggml_tensor* position_tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_I32, rows);
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr ||
            position_tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        ggml_tensor* contiguous_result = add_to_result ? ggml_cont(context.value, result_binding.tensor) : nullptr;
        if (contiguous_src == nullptr || (add_to_result && contiguous_result == nullptr))
        {
            set_last_error("Failed to create ggml contiguous rope_ex inputs.");
            return 0;
        }

        ggml_tensor* rope_input = ggml_reshape_4d(context.value, contiguous_src, src_desc.ne0, 1, rows, 1);
        if (rope_input == nullptr)
        {
            set_last_error("Failed to reshape ggml rope_ex input.");
            return 0;
        }

        std::vector<std::int32_t> positions;
        if (!read_i32_values(positions, positions_desc, "positions"))
        {
            return 0;
        }

        if (invert_positions)
        {
            for (std::int32_t& position : positions)
            {
                position = -position;
            }
        }

        ggml_tensor* rope_tensor = ggml_rope_ext(
            context.value,
            rope_input,
            position_tensor,
            nullptr,
            rope_dim,
            mode,
            original_context_length,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow);
        ggml_tensor* restored = rope_tensor == nullptr ? nullptr : ggml_reshape_4d(context.value, rope_tensor, result_desc.ne0, result_desc.ne1, result_desc.ne2, result_desc.ne3);
        ggml_tensor* value_tensor = restored;
        if (add_to_result)
        {
            value_tensor = restored == nullptr ? nullptr : ggml_add(context.value, contiguous_result, restored);
        }

        if (rope_tensor == nullptr || restored == nullptr || value_tensor == nullptr)
        {
            set_last_error("Failed to create ggml rope_ex node.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, value_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml rope_ex output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
        ggml_backend_tensor_set(position_tensor, positions.data(), 0, positions.size() * sizeof(std::int32_t));

        if (add_to_result || result_binding.raw_bytes > logical_bytes(result_desc))
        {
            upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int scaled_dot_product_attention_f32_impl(
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& query_desc,
        const TensorView4DDesc& key_desc,
        const TensorView4DDesc& value_desc,
        const TensorView4DDesc& mask_desc,
        bool has_mask,
        float scale)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result")
            || !validate_desc(query_desc, "query")
            || !validate_desc(key_desc, "key")
            || !validate_desc(value_desc, "value")
            || (has_mask && !validate_desc(mask_desc, "mask")))
        {
            return 0;
        }

        if (!can_map_standard_view(result_desc)
            || !can_map_standard_view(query_desc)
            || !can_map_standard_view(key_desc)
            || !can_map_standard_view(value_desc)
            || (has_mask && !can_map_standard_view(mask_desc)))
        {
            set_last_error("Tensor layout is not supported by the ggml scaled_dot_product_attention path.");
            return 0;
        }

        if (query_desc.ne3 != key_desc.ne3 || query_desc.ne3 != value_desc.ne3)
        {
            set_last_error("scaled_dot_product_attention expects matching batch dimensions.");
            return 0;
        }

        if (query_desc.ne2 != key_desc.ne2 || query_desc.ne2 != value_desc.ne2)
        {
            set_last_error("scaled_dot_product_attention expects matching head dimensions.");
            return 0;
        }

        if (query_desc.ne0 != key_desc.ne0)
        {
            set_last_error("scaled_dot_product_attention expects query and key to share the key dimension.");
            return 0;
        }

        if (result_desc.ne3 != query_desc.ne3 || result_desc.ne1 != query_desc.ne1 || result_desc.ne2 != query_desc.ne2 || result_desc.ne0 != value_desc.ne0)
        {
            set_last_error("scaled_dot_product_attention expects result shape [value_dim, heads, seq_q, batch].");
            return 0;
        }

        if (has_mask)
        {
            if (mask_desc.ne3 != query_desc.ne3 || mask_desc.ne2 != query_desc.ne1 || mask_desc.ne1 != query_desc.ne2 || mask_desc.ne0 != key_desc.ne2)
            {
                set_last_error("scaled_dot_product_attention expects mask shape [seq_k, seq_q, heads, batch].");
                return 0;
            }
        }

        bool use_zero_copy = true;
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding query_binding;
        TensorBinding key_binding;
        TensorBinding value_binding;
        TensorBinding mask_binding;

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            result_binding = create_standard_binding(context.value, result_desc);

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, query_desc, query_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            query_binding = create_standard_binding(context.value, query_desc);

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, key_desc, key_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            key_binding = create_standard_binding(context.value, key_desc);

        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, value_desc, value_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
            value_binding = create_standard_binding(context.value, value_desc);

        if (has_mask && use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, mask_desc, mask_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (has_mask && !use_zero_copy)
            mask_binding = create_standard_binding(context.value, mask_desc);

        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            query_binding.storage == nullptr || query_binding.tensor == nullptr ||
            key_binding.storage == nullptr || key_binding.tensor == nullptr ||
            value_binding.storage == nullptr || value_binding.tensor == nullptr ||
            (has_mask && (mask_binding.storage == nullptr || mask_binding.tensor == nullptr)))
        {
            set_last_error("Failed to allocate ggml scaled_dot_product_attention tensors.");
            return 0;
        }

        ggml_tensor* query_perm = ggml_permute(context.value, query_binding.tensor, 0, 2, 1, 3);
        ggml_tensor* key_perm = ggml_permute(context.value, key_binding.tensor, 0, 2, 1, 3);
        ggml_tensor* value_perm = ggml_permute(context.value, value_binding.tensor, 1, 2, 0, 3);
        value_perm = value_perm == nullptr ? nullptr : ggml_cont(context.value, value_perm);
        if (query_perm == nullptr || key_perm == nullptr || value_perm == nullptr)
        {
            set_last_error("Failed to create ggml attention permutation nodes.");
            return 0;
        }

        ggml_tensor* scores = ggml_mul_mat(context.value, key_perm, query_perm);
        if (scores == nullptr)
        {
            set_last_error("Failed to create ggml attention score node.");
            return 0;
        }
        ggml_mul_mat_set_prec(scores, GGML_PREC_F32);

        ggml_tensor* probs = ggml_soft_max_ext(context.value, scores, has_mask ? mask_binding.tensor : nullptr, scale, 0.0f);
        if (probs == nullptr)
        {
            set_last_error("Failed to create ggml soft_max_ext node.");
            return 0;
        }

        ggml_tensor* context_tensor = ggml_mul_mat(context.value, value_perm, probs);
        context_tensor = context_tensor == nullptr ? nullptr : ggml_permute(context.value, context_tensor, 0, 2, 1, 3);
        context_tensor = context_tensor == nullptr ? nullptr : ggml_cont(context.value, context_tensor);
        if (context_tensor == nullptr)
        {
            set_last_error("Failed to create ggml attention output node.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, context_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml attention output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(query_binding, query_desc.data, query_binding.raw_bytes);
            upload_binding(key_binding, key_desc.data, key_binding.raw_bytes);
            upload_binding(value_binding, value_desc.data, value_binding.raw_bytes);
            if (has_mask)
            {
                upload_binding(mask_binding, mask_desc.data, mask_binding.raw_bytes);
            }
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int softmax_f32_impl(
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& src_desc)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(src_desc, "src"))
        {
            return 0;
        }

        if (!same_shape(result_desc, src_desc))
        {
            set_last_error("Source tensor shape does not match result shape for ggml softmax.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(src_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml softmax Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(src_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding src_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, src_desc, src_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            src_binding = create_standard_binding(context.value, src_desc);
        }
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            src_binding.storage == nullptr || src_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* contiguous_src = ggml_cont(context.value, src_binding.tensor);
        if (contiguous_src == nullptr)
        {
            set_last_error("Failed to create ggml contiguous softmax input.");
            return 0;
        }

        ggml_tensor* softmax_tensor = ggml_soft_max(context.value, contiguous_src);
        if (softmax_tensor == nullptr)
        {
            set_last_error("Failed to create ggml softmax node.");
            return 0;
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, softmax_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(src_binding, src_desc.data, src_binding.raw_bytes);
            if (result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int softmax_grad_f32_impl(
        const TensorView4DDesc& result_desc,
        const TensorView4DDesc& adj_desc,
        const TensorView4DDesc& val_desc,
        bool add_grad)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(result_desc, "result") || !validate_desc(adj_desc, "adj") || !validate_desc(val_desc, "val"))
        {
            return 0;
        }

        if (!same_shape(result_desc, adj_desc) || !same_shape(result_desc, val_desc))
        {
            set_last_error("Tensor shape mismatch passed to ggml softmaxgrad.");
            return 0;
        }

        if (!can_map_standard_view(result_desc) || !can_map_standard_view(adj_desc) || !can_map_standard_view(val_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml softmaxgrad Metal path.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(result_desc) && can_map_standard_view(adj_desc) && can_map_standard_view(val_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 2 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding result_binding;
        TensorBinding adj_binding;
        TensorBinding val_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, result_desc, result_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, adj_desc, adj_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, val_desc, val_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            result_binding = create_standard_binding(context.value, result_desc);
            adj_binding = create_standard_binding(context.value, adj_desc);
            val_binding = create_standard_binding(context.value, val_desc);
        }
        if (result_binding.storage == nullptr || result_binding.tensor == nullptr ||
            adj_binding.storage == nullptr || adj_binding.tensor == nullptr ||
            val_binding.storage == nullptr || val_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensor views.");
            return 0;
        }

        ggml_tensor* contiguous_adj = ggml_cont(context.value, adj_binding.tensor);
        ggml_tensor* contiguous_val = ggml_cont(context.value, val_binding.tensor);
        if (contiguous_adj == nullptr || contiguous_val == nullptr)
        {
            set_last_error("Failed to create ggml contiguous softmaxgrad inputs.");
            return 0;
        }

        ggml_tensor* grad_tensor = ggml_soft_max_ext_back(context.value, contiguous_adj, contiguous_val, 1.0f, 0.0f);
        if (!backend_supports_op(grad_tensor))
        {
            ggml_tensor* weighted_adj = ggml_mul(context.value, contiguous_val, contiguous_adj);
            if (weighted_adj == nullptr)
            {
                set_last_error("Failed to create ggml softmaxgrad mul node.");
                return 0;
            }

            ggml_tensor* row_sum = ggml_sum_rows(context.value, weighted_adj);
            if (row_sum == nullptr)
            {
                set_last_error("Failed to create ggml softmaxgrad sum_rows node.");
                return 0;
            }

            ggml_tensor* centered_adj = ggml_sub(context.value, contiguous_adj, row_sum);
            if (centered_adj == nullptr)
            {
                set_last_error("Failed to create ggml softmaxgrad subtract node.");
                return 0;
            }

            grad_tensor = ggml_mul(context.value, contiguous_val, centered_adj);
        }

        if (grad_tensor == nullptr)
        {
            set_last_error("Failed to create ggml softmaxgrad output node.");
            return 0;
        }

        if (add_grad)
        {
            ggml_tensor* contiguous_result = ggml_cont(context.value, result_binding.tensor);
            if (contiguous_result == nullptr)
            {
                set_last_error("Failed to create ggml contiguous softmaxgrad accumulation input.");
                return 0;
            }

            grad_tensor = ggml_add(context.value, grad_tensor, contiguous_result);
            if (grad_tensor == nullptr)
            {
                set_last_error("Failed to create ggml softmaxgrad accumulation node.");
                return 0;
            }
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, grad_tensor, result_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml softmaxgrad output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(adj_binding, adj_desc.data, adj_binding.raw_bytes);
            upload_binding(val_binding, val_desc.data, val_binding.raw_bytes);
            if (add_grad || result_binding.raw_bytes > logical_bytes(result_desc))
                upload_binding(result_binding, result_desc.data, result_binding.raw_bytes);
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(result_binding.storage, result_desc.data, 0, result_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int cross_entropy_loss_f32_impl(
        float* loss_value,
        const TensorView4DDesc& probs_desc,
        const ContiguousTensorDesc& target_indices_desc,
        float smooth,
        float label_smooth)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (loss_value == nullptr)
        {
            set_last_error("Null pointer passed for lossValue.");
            return 0;
        }

        if (!validate_desc(probs_desc, "probs") || !validate_desc(target_indices_desc, "targetIndices"))
        {
            return 0;
        }

        if (!can_map_standard_view(probs_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml crossentropyloss Metal path.");
            return 0;
        }

        if (label_smooth < 0.0f || label_smooth > 1.0f)
        {
            set_last_error("labelSmooth must be in [0, 1] for ggml crossentropyloss.");
            return 0;
        }

        const std::int64_t rows = flat_row_count(probs_desc);
        const std::int64_t cols = probs_desc.ne0;
        if (target_indices_desc.element_count != rows)
        {
            set_last_error("Target index count must match the number of probability rows for ggml crossentropyloss.");
            return 0;
        }

        bool probs_zero_copy = can_map_standard_view(probs_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding probs_binding;
        if (probs_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, probs_desc, probs_binding, buf))
                probs_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!probs_zero_copy)
            probs_binding = create_standard_binding(context.value, probs_desc);
        TensorBinding labels_binding = create_matrix_binding(context.value, cols, rows);

        if (probs_binding.storage == nullptr || probs_binding.tensor == nullptr ||
            labels_binding.storage == nullptr || labels_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors for crossentropyloss.");
            return 0;
        }

        ggml_tensor* contiguous_probs = ggml_cont(context.value, probs_binding.tensor);
        ggml_tensor* flat_probs = contiguous_probs == nullptr ? nullptr : flatten_to_rows(context.value, contiguous_probs, cols, rows);
        ggml_tensor* logits_tensor = flat_probs == nullptr ? nullptr : ggml_log(context.value, flat_probs);
        if (contiguous_probs == nullptr || flat_probs == nullptr || logits_tensor == nullptr)
        {
            set_last_error("Failed to create ggml crossentropyloss logits tensor.");
            return 0;
        }

        ggml_tensor* loss_tensor = ggml_cross_entropy_loss(context.value, logits_tensor, labels_binding.tensor);
        if (loss_tensor == nullptr)
        {
            set_last_error("Failed to create ggml_cross_entropy_loss node.");
            return 0;
        }

        if (!backend_supports_op(loss_tensor))
        {
            set_last_error("ggml_cross_entropy_loss is not supported by the active backend.");
            return 0;
        }

        ggml_set_output(loss_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, loss_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        std::vector<float> labels;
        if (!build_cross_entropy_label_buffer(labels, target_indices_desc, rows, cols, label_smooth))
        {
            return 0;
        }

        if (!probs_zero_copy)
            upload_binding(probs_binding, probs_desc.data, probs_binding.raw_bytes);
        upload_binding(labels_binding, labels.data(), labels_binding.raw_bytes);

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_get(loss_tensor, loss_value, 0, sizeof(float));

        clear_last_error();
        return 1;
    }

    int cross_entropy_loss_backward_f32_impl(
        const TensorView4DDesc& grad_desc,
        const TensorView4DDesc& probs_desc,
        const ContiguousTensorDesc& target_indices_desc,
        float loss_gradient,
        float smooth,
        float label_smooth,
        bool add_grad)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(grad_desc, "grad") || !validate_desc(probs_desc, "probs") || !validate_desc(target_indices_desc, "targetIndices"))
        {
            return 0;
        }

        if (!same_shape(grad_desc, probs_desc))
        {
            set_last_error("Gradient tensor shape must match probability tensor shape for ggml crossentropyloss backward.");
            return 0;
        }

        if (!can_map_standard_view(grad_desc) || !can_map_standard_view(probs_desc))
        {
            set_last_error("Tensor layout is not supported by the ggml crossentropyloss backward Metal path.");
            return 0;
        }

        if (label_smooth < 0.0f || label_smooth > 1.0f)
        {
            set_last_error("labelSmooth must be in [0, 1] for ggml crossentropyloss backward.");
            return 0;
        }

        const std::int64_t rows = flat_row_count(probs_desc);
        const std::int64_t cols = probs_desc.ne0;
        if (target_indices_desc.element_count != rows)
        {
            set_last_error("Target index count must match the number of probability rows for ggml crossentropyloss backward.");
            return 0;
        }

        bool use_zero_copy = can_map_standard_view(grad_desc) && can_map_standard_view(probs_desc);
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 6 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding grad_binding;
        TensorBinding probs_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, grad_desc, grad_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_4d(context.value, g_backend, probs_desc, probs_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            grad_binding = create_standard_binding(context.value, grad_desc);
            probs_binding = create_standard_binding(context.value, probs_desc);
        }
        TensorBinding labels_binding = create_matrix_binding(context.value, cols, rows);
        TensorBinding loss_grad_binding = create_scalar_binding(context.value);

        if (grad_binding.storage == nullptr || grad_binding.tensor == nullptr ||
            probs_binding.storage == nullptr || probs_binding.tensor == nullptr ||
            labels_binding.storage == nullptr || labels_binding.tensor == nullptr ||
            loss_grad_binding.storage == nullptr || loss_grad_binding.tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors for crossentropyloss backward.");
            return 0;
        }

        ggml_tensor* contiguous_probs = ggml_cont(context.value, probs_binding.tensor);
        ggml_tensor* flat_probs = contiguous_probs == nullptr ? nullptr : flatten_to_rows(context.value, contiguous_probs, cols, rows);
        ggml_tensor* logits_tensor = flat_probs == nullptr ? nullptr : ggml_log(context.value, flat_probs);
        if (contiguous_probs == nullptr || flat_probs == nullptr || logits_tensor == nullptr)
        {
            set_last_error("Failed to create ggml crossentropyloss backward logits tensor.");
            return 0;
        }

        ggml_tensor* grad_tensor = ggml_cross_entropy_loss_back(context.value, loss_grad_binding.tensor, logits_tensor, labels_binding.tensor);
        if (grad_tensor == nullptr)
        {
            set_last_error("Failed to create ggml_cross_entropy_loss_back node.");
            return 0;
        }

        if (!backend_supports_op(grad_tensor))
        {
            set_last_error("ggml_cross_entropy_loss_back is not supported by the active backend.");
            return 0;
        }

        ggml_tensor* reshaped_grad = ggml_reshape_4d(context.value, grad_tensor, grad_desc.ne0, grad_desc.ne1, grad_desc.ne2, grad_desc.ne3);
        if (reshaped_grad == nullptr)
        {
            set_last_error("Failed to reshape ggml crossentropyloss backward tensor.");
            return 0;
        }

        if (add_grad)
        {
            ggml_tensor* contiguous_grad = ggml_cont(context.value, grad_binding.tensor);
            reshaped_grad = contiguous_grad == nullptr ? nullptr : ggml_add(context.value, contiguous_grad, reshaped_grad);
            if (reshaped_grad == nullptr)
            {
                set_last_error("Failed to create ggml crossentropyloss backward accumulation node.");
                return 0;
            }
        }

        ggml_tensor* output_tensor = ggml_cpy(context.value, reshaped_grad, grad_binding.tensor);
        if (output_tensor == nullptr)
        {
            set_last_error("Failed to create ggml crossentropyloss backward output copy node.");
            return 0;
        }

        ggml_set_output(output_tensor);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, output_tensor);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        std::vector<float> labels;
        if (!build_cross_entropy_label_buffer(labels, target_indices_desc, rows, cols, label_smooth))
        {
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(probs_binding, probs_desc.data, probs_binding.raw_bytes);
            if (add_grad || grad_binding.raw_bytes > logical_bytes(grad_desc))
                upload_binding(grad_binding, grad_desc.data, grad_binding.raw_bytes);
        }
        upload_binding(labels_binding, labels.data(), labels_binding.raw_bytes);
        ggml_backend_tensor_set(loss_grad_binding.storage, &loss_gradient, 0, sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        if (!use_zero_copy)
            ggml_backend_tensor_get(grad_binding.storage, grad_desc.data, 0, grad_binding.raw_bytes);

        clear_last_error();
        return 1;
    }

    int adam_f32_impl(
        const ContiguousTensorDesc& weight_desc,
        const ContiguousTensorDesc& gradient_desc,
        const ContiguousTensorDesc& v_desc,
        const ContiguousTensorDesc& m_desc,
        float grad_norm_factor,
        float step_size,
        float clip_value,
        float regc,
        float decay_rate_v,
        float decay_rate_m,
        int iter,
        float eps)
    {
        if (!ensure_backend())
        {
            return 0;
        }

        if (!validate_desc(weight_desc, "weight")
            || !validate_desc(gradient_desc, "gradient")
            || !validate_desc(v_desc, "v")
            || !validate_desc(m_desc, "m"))
        {
            return 0;
        }

        if (weight_desc.element_count != gradient_desc.element_count
            || weight_desc.element_count != v_desc.element_count
            || weight_desc.element_count != m_desc.element_count)
        {
            set_last_error("Tensor shape mismatch passed to ggml adam.");
            return 0;
        }

        bool use_zero_copy = true;
        std::vector<BufferHandle> host_ptr_buffers;
        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context.");
            return 0;
        }

        TensorBinding weight_binding;
        TensorBinding gradient_binding;
        TensorBinding v_binding;
        TensorBinding m_binding;
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_contiguous(context.value, g_backend, weight_desc, weight_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_contiguous(context.value, g_backend, gradient_desc, gradient_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_contiguous(context.value, g_backend, v_desc, v_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (use_zero_copy)
        {
            ggml_backend_buffer_t buf = nullptr;
            if (!create_binding_from_host_ptr_contiguous(context.value, g_backend, m_desc, m_binding, buf))
                use_zero_copy = false;
            else
                host_ptr_buffers.emplace_back(buf);
        }
        if (!use_zero_copy)
        {
            weight_binding = create_contiguous_binding(context.value, weight_desc);
            gradient_binding = create_contiguous_binding(context.value, gradient_desc);
            v_binding = create_contiguous_binding(context.value, v_desc);
            m_binding = create_contiguous_binding(context.value, m_desc);
        }
        ggml_tensor* adamw_params_tensor = ggml_new_tensor_1d(context.value, GGML_TYPE_F32, 7);

        if (weight_binding.storage == nullptr || weight_binding.tensor == nullptr ||
            gradient_binding.storage == nullptr || gradient_binding.tensor == nullptr ||
            v_binding.storage == nullptr || v_binding.tensor == nullptr ||
            m_binding.storage == nullptr || m_binding.tensor == nullptr ||
            adamw_params_tensor == nullptr)
        {
            set_last_error("Failed to allocate ggml tensors for adam.");
            return 0;
        }

        ggml_tensor* grad_tensor = gradient_binding.tensor;
        if (grad_norm_factor != 1.0f)
        {
            grad_tensor = ggml_scale(context.value, grad_tensor, grad_norm_factor);
            if (grad_tensor == nullptr)
            {
                set_last_error("Failed to create ggml adam grad scaling node.");
                return 0;
            }
        }

        ggml_tensor* clipped_grad = ggml_clamp(context.value, grad_tensor, -clip_value, clip_value);
        if (clipped_grad == nullptr)
        {
            set_last_error("Failed to create ggml adam clamp node.");
            return 0;
        }

        const float bias_correction_m = static_cast<float>(1.0 / (1.0 - std::pow(decay_rate_m, iter)));
        const float bias_correction_v = static_cast<float>(1.0 / (1.0 - std::pow(decay_rate_v, iter)));
        const std::array<float, 7> adamw_params = {
            step_size,
            decay_rate_m,
            decay_rate_v,
            eps,
            regc,
            bias_correction_m,
            bias_correction_v
        };

        ggml_set_param(weight_binding.tensor);

        ggml_tensor* adamw_step = ggml_opt_step_adamw(
            context.value,
            weight_binding.tensor,
            clipped_grad,
            m_binding.tensor,
            v_binding.tensor,
            adamw_params_tensor);
        if (adamw_step == nullptr)
        {
            set_last_error("Failed to create ggml adamw optimizer node.");
            return 0;
        }

        ggml_set_output(adamw_step);

        ggml_cgraph* graph = ggml_new_graph(context.value);
        if (graph == nullptr)
        {
            set_last_error("Failed to create ggml graph.");
            return 0;
        }

        ggml_build_forward_expand(graph, adamw_step);

        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(context.value, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate ggml backend buffer.");
            return 0;
        }

        if (!use_zero_copy)
        {
            upload_binding(weight_binding, weight_desc.data, weight_binding.raw_bytes);
            upload_binding(gradient_binding, gradient_desc.data, gradient_binding.raw_bytes);
            upload_binding(v_binding, v_desc.data, v_binding.raw_bytes);
            upload_binding(m_binding, m_desc.data, m_binding.raw_bytes);
        }
        ggml_backend_tensor_set(adamw_params_tensor, adamw_params.data(), 0, adamw_params.size() * sizeof(float));

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed.");
            return 0;
        }

        ggml_backend_synchronize(g_backend);
        ggml_backend_tensor_memset(gradient_binding.storage, 0, 0, gradient_binding.raw_bytes);
        ggml_backend_synchronize(g_backend);

        if (!use_zero_copy)
        {
            ggml_backend_tensor_get(weight_binding.storage, weight_desc.data, 0, weight_binding.raw_bytes);
            ggml_backend_tensor_get(m_binding.storage, m_desc.data, 0, m_binding.raw_bytes);
            ggml_backend_tensor_get(v_binding.storage, v_desc.data, 0, v_binding.raw_bytes);
            ggml_backend_tensor_get(gradient_binding.storage, gradient_desc.data, 0, gradient_binding.raw_bytes);
        }

        clear_last_error();
        return 1;
    }
}

TSG_EXPORT const char* TSGgml_GetLastError()
{
    return g_last_error.c_str();
}

TSG_EXPORT int TSGgml_IsMetalAvailable()
{
    clear_last_error();
    return ensure_backend(BACKEND_TYPE_METAL) ? 1 : 0;
}

TSG_EXPORT int TSGgml_IsBackendAvailable(int backendType)
{
    clear_last_error();
    return ensure_backend(backendType) ? 1 : 0;
}

TSG_EXPORT int TSGgml_AddmmF32(
    TensorView2DDesc result,
    TensorView2DDesc src,
    TensorView2DDesc m1,
    TensorView2DDesc m2,
    float beta,
    float alpha)
{
    try
    {
        return addmm_f32_impl(result, src, m1, m2, beta, alpha);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml addmm failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_AddmmQuantF32(
    TensorView2DDesc result,
    TensorView2DDesc m1,
    void* m2_data,
    int m2_ggml_type,
    std::int64_t m2_ne0,
    std::int64_t m2_ne1,
    std::int64_t m2_raw_bytes)
{
    try
    {
        QuantizedWeightDesc m2_quant;
        m2_quant.data = m2_data;
        m2_quant.ggml_type = m2_ggml_type;
        m2_quant.ne0 = m2_ne0;
        m2_quant.ne1 = m2_ne1;
        m2_quant.raw_bytes = m2_raw_bytes;
        return addmm_quant_f32_impl(result, m1, m2_quant);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml addmm_quant failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_GetRowsQuantF32(
    TensorView2DDesc result,
    void* src_data,
    int src_ggml_type,
    std::int64_t src_ne0,
    std::int64_t src_ne1,
    std::int64_t src_raw_bytes,
    ContiguousTensorDesc indices)
{
    try
    {
        QuantizedWeightDesc src_quant;
        src_quant.data = src_data;
        src_quant.ggml_type = src_ggml_type;
        src_quant.ne0 = src_ne0;
        src_quant.ne1 = src_ne1;
        src_quant.raw_bytes = src_raw_bytes;
        return get_rows_quant_f32_impl(result, src_quant, indices);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml get_rows_quant failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_AddmmQuantBatchF32(
    TensorView2DDesc result,
    TensorView2DDesc m1,
    void* m2_data,
    int m2_ggml_type,
    std::int64_t m2_ne0,
    std::int64_t m2_raw_bytes,
    int batch_count,
    std::int64_t* weight_offsets,
    std::int64_t* weight_ne1_arr)
{
    try
    {
        QuantizedWeightDesc m2_quant;
        m2_quant.data = m2_data;
        m2_quant.ggml_type = m2_ggml_type;
        m2_quant.ne0 = m2_ne0;
        m2_quant.ne1 = 0;
        m2_quant.raw_bytes = m2_raw_bytes;
        return addmm_quant_batch_f32_impl(result, m1, m2_quant, batch_count, weight_offsets, weight_ne1_arr);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml addmm_quant_batch failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_AddmmBatchF32(
    TensorView3DDesc result,
    TensorView3DDesc src,
    TensorView3DDesc m1,
    TensorView3DDesc m2,
    float beta,
    float alpha)
{
    try
    {
        return addmm_batch_f32_impl(result, src, m1, m2, beta, alpha);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml addmmbatch failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_ReduceLastDimF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc src)
{
    try
    {
        return reduce_last_dim_f32_impl(static_cast<ReductionOpCode>(op), result, src);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml reduction failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_IndexReductionF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc src)
{
    try
    {
        return index_reduction_f32_impl(static_cast<IndexReductionOpCode>(op), result, src);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml index-reduction failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_CopyF32(
    TensorView4DDesc result,
    TensorView4DDesc src)
{
    try
    {
        return copy_f32_impl(result, src);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml copy failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_UnaryF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc src)
{
    try
    {
        return unary_f32_impl(static_cast<UnaryOpCode>(op), result, src);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml unary failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_BinaryTensorF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc lhs,
    TensorView4DDesc rhs)
{
    try
    {
        return binary_tensor_f32_impl(static_cast<BinaryTensorOpCode>(op), result, lhs, rhs);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml binary-tensor failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_FusedActMulF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc a,
    TensorView4DDesc b)
{
    try
    {
        return fused_act_mul_f32_impl(static_cast<FusedActMulOpCode>(op), result, a, b);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml fused activation-multiply failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_BinaryScalarF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc src,
    float scalar)
{
    try
    {
        return binary_scalar_f32_impl(static_cast<BinaryScalarOpCode>(op), result, src, scalar);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml binary-scalar failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_ActivationGradF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc src,
    TensorView4DDesc grad,
    TensorView4DDesc accumulation,
    int has_accumulation)
{
    try
    {
        return activation_grad_f32_impl(static_cast<ActivationGradOpCode>(op), result, src, grad, accumulation, has_accumulation != 0);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml activation-grad failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_NormF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc src,
    TensorView4DDesc gamma,
    TensorView4DDesc beta,
    int has_beta,
    float eps)
{
    try
    {
        return norm_f32_impl(static_cast<NormOpCode>(op), result, src, gamma, beta, has_beta != 0, eps);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml norm failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_NormGradF32(
    int op,
    TensorView4DDesc result,
    TensorView4DDesc grad_gamma,
    TensorView4DDesc grad_beta,
    TensorView4DDesc adj,
    TensorView4DDesc x,
    TensorView4DDesc gamma,
    int has_grad_beta,
    float eps)
{
    try
    {
        return norm_grad_f32_impl(static_cast<NormOpCode>(op), result, grad_gamma, grad_beta, adj, x, gamma, has_grad_beta != 0, eps);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml norm-grad failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_MulMatIdF32(
    TensorView3DDesc result,
    TensorView3DDesc expert_weights,
    TensorView3DDesc input,
    ContiguousTensorDesc ids,
    int ids_rows,
    int ids_cols)
{
    try
    {
        return mul_mat_id_f32_impl(result, expert_weights, input, ids, ids_rows, ids_cols);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml mulmatid failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_AddIdF32(
    TensorView3DDesc result,
    TensorView3DDesc src,
    TensorView2DDesc bias,
    ContiguousTensorDesc ids,
    int ids_rows,
    int ids_cols)
{
    try
    {
        return add_id_f32_impl(result, src, bias, ids, ids_rows, ids_cols);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml addid failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_IndexSelectF32(
    TensorView2DDesc result,
    TensorView2DDesc src,
    ContiguousTensorDesc indices,
    int add_to_result)
{
    try
    {
        return index_select_f32_impl(result, src, indices, add_to_result != 0);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml indexselect failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_IndexSelectGradF32(
    TensorView2DDesc grad,
    TensorView2DDesc adj,
    ContiguousTensorDesc indices)
{
    try
    {
        return index_select_grad_f32_impl(grad, adj, indices);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml indexselectgrad failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_RoPEF32(
    TensorView4DDesc result,
    TensorView4DDesc src,
    int seq_len,
    int row_offset,
    int add_to_result,
    int invert_positions)
{
    try
    {
        return rope_f32_impl(result, src, seq_len, row_offset, add_to_result != 0, invert_positions != 0);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml rope failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_RoPEExF32(
    TensorView4DDesc result,
    TensorView4DDesc src,
    ContiguousTensorDesc positions,
    int rope_dim,
    int mode,
    int original_context_length,
    float freq_base,
    float freq_scale,
    float ext_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    int add_to_result,
    int invert_positions)
{
    try
    {
        return rope_ex_f32_impl(
            result,
            src,
            positions,
            rope_dim,
            mode,
            original_context_length,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            add_to_result != 0,
            invert_positions != 0);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml rope_ex failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_ScaledDotProductAttentionF32(
    TensorView4DDesc result,
    TensorView4DDesc query,
    TensorView4DDesc key,
    TensorView4DDesc value,
    TensorView4DDesc mask,
    int has_mask,
    float scale)
{
    try
    {
        return scaled_dot_product_attention_f32_impl(result, query, key, value, mask, has_mask != 0, scale);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml scaled_dot_product_attention failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_SoftmaxF32(
    TensorView4DDesc result,
    TensorView4DDesc src)
{
    try
    {
        return softmax_f32_impl(result, src);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml softmax failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_SoftmaxGradF32(
    TensorView4DDesc result,
    TensorView4DDesc adj,
    TensorView4DDesc val,
    int add_grad)
{
    try
    {
        return softmax_grad_f32_impl(result, adj, val, add_grad != 0);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml softmaxgrad failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_CrossEntropyLossF32(
    float* loss_value,
    TensorView4DDesc probs,
    ContiguousTensorDesc target_indices,
    float smooth,
    float label_smooth)
{
    try
    {
        return cross_entropy_loss_f32_impl(loss_value, probs, target_indices, smooth, label_smooth);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml crossentropyloss failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_CrossEntropyLossBackwardF32(
    TensorView4DDesc grad,
    TensorView4DDesc probs,
    ContiguousTensorDesc target_indices,
    float loss_gradient,
    float smooth,
    float label_smooth,
    int add_grad)
{
    try
    {
        return cross_entropy_loss_backward_f32_impl(grad, probs, target_indices, loss_gradient, smooth, label_smooth, add_grad != 0);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml crossentropyloss backward failure.");
        return 0;
    }
}

TSG_EXPORT int TSGgml_AdamF32(
    ContiguousTensorDesc weight,
    ContiguousTensorDesc gradient,
    ContiguousTensorDesc v,
    ContiguousTensorDesc m,
    float grad_norm_factor,
    float step_size,
    float clip_value,
    float regc,
    float decay_rate_v,
    float decay_rate_m,
    int iter,
    float eps)
{
    try
    {
        return adam_f32_impl(weight, gradient, v, m, grad_norm_factor, step_size, clip_value, regc, decay_rate_v, decay_rate_m, iter, eps);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown ggml adam failure.");
        return 0;
    }
}

// --- GGUF: tensor row bytes and dequantize to F32 (InferenceEngine / GgufStreamingReader) ---

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
    for (auto& [ptr, cached] : g_host_buffer_cache)
        ggml_backend_buffer_free(cached.buffer);
    g_host_buffer_cache.clear();
}

TSG_EXPORT size_t TSGgml_RowSize(int ggml_type, int64_t ne)
{
    if (ggml_type < 0 || ggml_type >= GGML_TYPE_COUNT || ne <= 0)
    {
        return 0;
    }
    const enum ggml_type t = static_cast<enum ggml_type>(ggml_type);
    const int64_t bs = ggml_blck_size(t);
    if (bs <= 0 || ne % bs != 0)
    {
        return 0;
    }
    return ggml_row_size(t, ne);
}

TSG_EXPORT int TSGgml_DequantizeToF32(int ggml_type, const void* src, int64_t num_elements, float* dst)
{
    if (src == nullptr || dst == nullptr || num_elements < 0)
    {
        return -1;
    }
    if (num_elements == 0)
    {
        return 0;
    }
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
    // Q8_K has no to_float in ggml type_traits in some versions
    if (ggml_type == GGML_TYPE_Q8_K)
    {
        dequantize_row_q8_K(static_cast<const block_q8_K*>(src), dst, num_elements);
        return 0;
    }
    return -2;
}

// ============================================================================
// Batched transformer layer decode: full layer in a single GGML graph.
// Handles: attn_norm → QKV matmul → QK norm → RoPE → flash attention →
//          O projection → residual → FFN norm → GateUp matmul → SiLU*Mul →
//          Down matmul → residual.
// Updates hidden state in-place and writes new K/V to the KV cache.
// ============================================================================
namespace
{
    int transformer_layer_decode_impl(
        float* hidden_data, int hidden_size,
        float* attn_norm_data,
        void* qkv_data, int qkv_type, std::int64_t qkv_ne0, std::int64_t qkv_ne1, std::int64_t qkv_bytes,
        float* q_norm_data, float* k_norm_data, int head_dim,
        void* o_data, int o_type, std::int64_t o_ne0, std::int64_t o_ne1, std::int64_t o_bytes,
        float* ffn_norm_data,
        void* gu_data, int gu_type, std::int64_t gu_ne0, std::int64_t gu_ne1, std::int64_t gu_bytes,
        void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_bytes,
        float* k_cache_data, float* v_cache_data,
        int num_heads, int num_kv_heads,
        int max_seq_len, int position,
        float eps, float rope_base, float rope_freq_scale,
        int intermediate_size, int rope_mode)
    {
        if (!ensure_backend())
            return 0;

        const int qDim = num_heads * head_dim;
        const int kDim = num_kv_heads * head_dim;
        const int totalSeqLen = position + 1;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Create contiguous copies of cached KV (strided cache → contiguous buffer)
        std::vector<float> k_cached_buf, v_cached_buf;
        if (position > 0)
        {
            k_cached_buf.resize(static_cast<std::size_t>(position) * kDim);
            v_cached_buf.resize(static_cast<std::size_t>(position) * kDim);
            for (int h = 0; h < num_kv_heads; h++)
            {
                std::memcpy(k_cached_buf.data() + h * position * head_dim,
                            k_cache_data + h * max_seq_len * head_dim,
                            static_cast<std::size_t>(position) * head_dim * sizeof(float));
                std::memcpy(v_cached_buf.data() + h * position * head_dim,
                            v_cache_data + h * max_seq_len * head_dim,
                            static_cast<std::size_t>(position) * head_dim * sizeof(float));
            }
        }

        PooledContextHandle context;
        if (!context.init(2 * 1024 * 1024))
        {
            set_last_error("Failed to create ggml context for transformer layer decode.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // === Input / weight tensors ===
        ggml_tensor* input        = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* attn_norm_w  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* q_norm_w     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
        ggml_tensor* k_norm_w     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
        ggml_tensor* ffn_norm_w   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

        ggml_tensor* qkv_w  = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkv_type), qkv_ne0, qkv_ne1);
        ggml_tensor* o_w     = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type), o_ne0, o_ne1);
        ggml_tensor* gu_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gu_type), gu_ne0, gu_ne1);
        ggml_tensor* down_w  = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type), down_ne0, down_ne1);

        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

        ggml_tensor* k_cached_t = nullptr;
        ggml_tensor* v_cached_t = nullptr;
        if (position > 0)
        {
            k_cached_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, position, num_kv_heads);
            v_cached_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, position, num_kv_heads);
        }

        // Output download targets
        ggml_tensor* hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* k_new_out  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDim);
        ggml_tensor* v_new_out  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDim);

        // === Build computation graph ===

        // 1. Attention norm: RMSNorm + element-wise scale
        ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, input, eps), attn_norm_w);

        // 2. Fused QKV projection (quantized matmul)
        ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, hidden_size, 1);
        ggml_tensor* qkv_flat  = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, qkv_w, normed_2d), qDim + 2 * kDim);

        // 3. Split Q, K, V
        ggml_tensor* q_raw = ggml_view_1d(ctx, qkv_flat, qDim, 0);
        ggml_tensor* k_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim)  * sizeof(float));
        ggml_tensor* v_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim + kDim) * sizeof(float));

        // 4. Per-head QK norm
        ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_raw, head_dim, num_heads);
        ggml_tensor* k_2d = ggml_reshape_2d(ctx, k_raw, head_dim, num_kv_heads);

        ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), q_norm_w);
        ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d, eps), k_norm_w);

        // 5. RoPE (NeoX mode)
        // ggml_rope_ext expects: ne[0]=head_dim, ne[1]=n_heads, ne[2]=seqLen
        // positions tensor ne[0] must equal ne[2]
        ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, head_dim, num_heads, 1);
        ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, head_dim, num_kv_heads, 1);

        ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, nullptr,
            head_dim, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
        ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, nullptr,
            head_dim, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);

        // 6. Build full KV for attention: concat cached + new
        // After RoPE: q_rope=[head_dim, num_heads, 1], k_rope=[head_dim, num_kv_heads, 1]
        // flash_attn_ext expects: q=[head_dim, n_batch, n_head], k/v=[head_dim, n_kv, n_head_kv]
        // Need to permute dims 1,2: [head_dim, n_heads, 1] → [head_dim, 1, n_heads]
        ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);

        ggml_tensor* k_rope_perm = ggml_permute(ctx, k_rope, 0, 2, 1, 3);
        ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_raw, head_dim, num_kv_heads, 1);
        ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);

        ggml_tensor* k_full;
        ggml_tensor* v_full;
        if (position > 0)
        {
            k_full = ggml_concat(ctx, k_cached_t, ggml_cont(ctx, k_rope_perm), 1);
            v_full = ggml_concat(ctx, v_cached_t, ggml_cont(ctx, v_perm),       1);
        }
        else
        {
            k_full = ggml_cont(ctx, k_rope_perm);
            v_full = ggml_cont(ctx, v_perm);
        }

        // 7. Flash attention (handles GQA broadcasting automatically)
        // q: [head_dim, 1, num_heads], k/v: [head_dim, totalSeqLen, num_kv_heads]
        ggml_tensor* attn_out = ggml_flash_attn_ext(ctx,
            q_attn, k_full, v_full, nullptr, scale, 0.0f, 0.0f);

        // 8. O projection
        ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, qDim, 1);
        ggml_tensor* o_flat    = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, o_w, attn_flat), hidden_size);

        // 9. First residual
        ggml_tensor* residual1 = ggml_add(ctx, input, o_flat);

        // 10. FFN norm
        ggml_tensor* normed2 = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), ffn_norm_w);

        // 11. Fused GateUp projection
        ggml_tensor* normed2_2d = ggml_reshape_2d(ctx, normed2, hidden_size, 1);
        ggml_tensor* gu_flat    = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, gu_w, normed2_2d), 2 * intermediate_size);

        // 12. Split gate / up, SiLU(gate) * up
        ggml_tensor* gate = ggml_view_1d(ctx, gu_flat, intermediate_size, 0);
        ggml_tensor* up   = ggml_view_1d(ctx, gu_flat, intermediate_size,
                                          static_cast<std::size_t>(intermediate_size) * sizeof(float));
        ggml_tensor* ffn_hidden = ggml_mul(ctx, ggml_silu(ctx, gate), up);

        // 13. Down projection
        ggml_tensor* ffn_2d   = ggml_reshape_2d(ctx, ffn_hidden, intermediate_size, 1);
        ggml_tensor* down_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, down_w, ffn_2d), hidden_size);

        // 14. Second residual
        ggml_tensor* result = ggml_add(ctx, residual1, down_flat);

        // Mark graph outputs: updated hidden, new K (after RoPE), new V
        ggml_tensor* out_hidden = ggml_cpy(ctx, result, hidden_out);
        ggml_set_output(out_hidden);

        ggml_tensor* k_rope_flat = ggml_reshape_1d(ctx, k_rope, kDim);
        ggml_tensor* out_k = ggml_cpy(ctx, k_rope_flat, k_new_out);
        ggml_set_output(out_k);

        ggml_tensor* v_flat = ggml_reshape_1d(ctx, v_raw, kDim);
        ggml_tensor* out_v  = ggml_cpy(ctx, v_flat, v_new_out);
        ggml_set_output(out_v);

        // Build graph
        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out_hidden);
        ggml_build_forward_expand(graph, out_k);
        ggml_build_forward_expand(graph, out_v);

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        bool can_host_ptr = false;
        if (dev != nullptr)
        {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev, &props);
            can_host_ptr = props.caps.buffer_from_host_ptr;
        }

        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable) {
            if (can_host_ptr && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (cacheable)
                {
                    auto it = g_host_buffer_cache.find(data);
                    if (it != g_host_buffer_cache.end() && it->second.bytes == bytes)
                    {
                        buf = it->second.buffer;
                    }
                    else
                    {
                        (void)try_create_host_ptr_buffer(g_backend, dev, data, bytes, buf);
                        if (buf != nullptr)
                            g_host_buffer_cache[data] = {buf, bytes};
                    }
                }
                else
                {
                    (void)try_create_host_ptr_buffer(g_backend, dev, data, bytes, buf);
                    if (buf != nullptr)
                        ephemeral_bufs.emplace_back(buf);
                }
                if (buf != nullptr)
                {
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, data);
                    if (st == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        bind_or_mark(qkv_w,  qkv_data,  static_cast<std::size_t>(qkv_bytes), true);
        bind_or_mark(o_w,    o_data,    static_cast<std::size_t>(o_bytes), true);
        bind_or_mark(gu_w,   gu_data,   static_cast<std::size_t>(gu_bytes), true);
        bind_or_mark(down_w, down_data, static_cast<std::size_t>(down_bytes), true);

        bind_or_mark(attn_norm_w, attn_norm_data, static_cast<std::size_t>(hidden_size) * sizeof(float), true);
        bind_or_mark(ffn_norm_w,  ffn_norm_data,  static_cast<std::size_t>(hidden_size) * sizeof(float), true);
        bind_or_mark(q_norm_w,    q_norm_data,    static_cast<std::size_t>(head_dim) * sizeof(float), true);
        bind_or_mark(k_norm_w,    k_norm_data,    static_cast<std::size_t>(head_dim) * sizeof(float), true);

        if (position > 0)
        {
            bind_or_mark(k_cached_t, k_cached_buf.data(), k_cached_buf.size() * sizeof(float), false);
            bind_or_mark(v_cached_t, v_cached_buf.data(), v_cached_buf.size() * sizeof(float), false);
        }

        // Allocate backend buffer for remaining tensors (intermediates + non-host-ptr tensors)
        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate backend buffer for transformer layer decode.");
            return 0;
        }

        // Upload non-host-ptr tensors
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, u.bytes > 0 ? 0 : 0, u.bytes);

        ggml_backend_tensor_set(input, hidden_data, 0, static_cast<std::size_t>(hidden_size) * sizeof(float));

        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));

        // Execute
        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for transformer layer decode.");
            return 0;
        }
        ggml_backend_synchronize(g_backend);

        // Download updated hidden state
        ggml_backend_tensor_get(hidden_out, hidden_data, 0, static_cast<std::size_t>(hidden_size) * sizeof(float));

        // Download new K/V and write to cache at `position`
        std::vector<float> k_new_buf(static_cast<std::size_t>(kDim));
        std::vector<float> v_new_buf(static_cast<std::size_t>(kDim));
        ggml_backend_tensor_get(k_new_out, k_new_buf.data(), 0, static_cast<std::size_t>(kDim) * sizeof(float));
        ggml_backend_tensor_get(v_new_out, v_new_buf.data(), 0, static_cast<std::size_t>(kDim) * sizeof(float));

        for (int h = 0; h < num_kv_heads; h++)
        {
            std::memcpy(k_cache_data + h * max_seq_len * head_dim + position * head_dim,
                        k_new_buf.data() + h * head_dim,
                        static_cast<std::size_t>(head_dim) * sizeof(float));
            std::memcpy(v_cache_data + h * max_seq_len * head_dim + position * head_dim,
                        v_new_buf.data() + h * head_dim,
                        static_cast<std::size_t>(head_dim) * sizeof(float));
        }

        clear_last_error();
        return 1;
    }
}

TSG_EXPORT int TSGgml_TransformerLayerDecode(
    float* hidden_data, int hidden_size,
    float* attn_norm_data,
    void* qkv_data, int qkv_type, std::int64_t qkv_ne0, std::int64_t qkv_ne1, std::int64_t qkv_bytes,
    float* q_norm_data, float* k_norm_data, int head_dim,
    void* o_data, int o_type, std::int64_t o_ne0, std::int64_t o_ne1, std::int64_t o_bytes,
    float* ffn_norm_data,
    void* gu_data, int gu_type, std::int64_t gu_ne0, std::int64_t gu_ne1, std::int64_t gu_bytes,
    void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_bytes,
    float* k_cache_data, float* v_cache_data,
    int num_heads, int num_kv_heads,
    int max_seq_len, int position,
    float eps, float rope_base, float rope_freq_scale,
    int intermediate_size, int rope_mode)
{
    try
    {
        return transformer_layer_decode_impl(
            hidden_data, hidden_size,
            attn_norm_data,
            qkv_data, qkv_type, qkv_ne0, qkv_ne1, qkv_bytes,
            q_norm_data, k_norm_data, head_dim,
            o_data, o_type, o_ne0, o_ne1, o_bytes,
            ffn_norm_data,
            gu_data, gu_type, gu_ne0, gu_ne1, gu_bytes,
            down_data, down_type, down_ne0, down_ne1, down_bytes,
            k_cache_data, v_cache_data,
            num_heads, num_kv_heads,
            max_seq_len, position,
            eps, rope_base, rope_freq_scale,
            intermediate_size, rope_mode);
    }
    catch (const std::exception& ex)
    {
        set_last_error(ex.what());
        return 0;
    }
    catch (...)
    {
        set_last_error("Unknown error in transformer layer decode.");
        return 0;
    }
}

// ============================================================================
// Full-model decode: ALL transformer layers in a single GGML graph.
// Eliminates per-layer Metal synchronization overhead.
// ============================================================================

TSG_EXPORT int TSGgml_TransformerModelDecode(
    float* hidden_data, int hidden_size, int num_layers,
    void** attn_norm_arr, void** qkv_arr, void** q_norm_arr, void** k_norm_arr,
    void** o_arr, void** ffn_norm_arr, void** gu_arr, void** down_arr,
    void** k_cache_arr, void** v_cache_arr,
    int qkv_type, std::int64_t qkv_ne0, std::int64_t qkv_ne1, std::int64_t qkv_bytes,
    int o_type, std::int64_t o_ne0, std::int64_t o_ne1, std::int64_t o_bytes,
    int gu_type, std::int64_t gu_ne0, std::int64_t gu_ne1, std::int64_t gu_bytes,
    int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_bytes,
    int head_dim, int num_heads, int num_kv_heads,
    int max_seq_len, int position,
    float eps, float rope_base, float rope_freq_scale,
    int intermediate_size, int rope_mode)
{
    try
    {
        if (!ensure_backend())
            return 0;

        const int qDim = num_heads * head_dim;
        const int kDim = num_kv_heads * head_dim;
        const int totalSeqLen = position + 1;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Pre-copy cached KV for all layers
        struct LayerKVCache {
            std::vector<float> k_buf;
            std::vector<float> v_buf;
        };
        std::vector<LayerKVCache> kv_caches(num_layers);
        if (position > 0)
        {
            for (int l = 0; l < num_layers; l++)
            {
                auto& cache = kv_caches[l];
                cache.k_buf.resize(static_cast<std::size_t>(position) * kDim);
                cache.v_buf.resize(static_cast<std::size_t>(position) * kDim);
                float* kc = static_cast<float*>(k_cache_arr[l]);
                float* vc = static_cast<float*>(v_cache_arr[l]);
                for (int h = 0; h < num_kv_heads; h++)
                {
                    std::memcpy(cache.k_buf.data() + h * position * head_dim,
                                kc + h * max_seq_len * head_dim,
                                static_cast<std::size_t>(position) * head_dim * sizeof(float));
                    std::memcpy(cache.v_buf.data() + h * position * head_dim,
                                vc + h * max_seq_len * head_dim,
                                static_cast<std::size_t>(position) * head_dim * sizeof(float));
                }
            }
        }

        // Large context for all layers
        const std::size_t ctx_size = 16 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for model decode.");
            return 0;
        }
        ggml_context* ctx = context.value;

        // Input tensor (shared across graph)
        ggml_tensor* current = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

        // Per-layer weight tensors and KV cache tensors
        struct LayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* q_norm_w;
            ggml_tensor* k_norm_w;
            ggml_tensor* o_w;
            ggml_tensor* ffn_norm_w;
            ggml_tensor* gu_w;
            ggml_tensor* down_w;
            ggml_tensor* k_cached_t;
            ggml_tensor* v_cached_t;
            ggml_tensor* k_new_out;
            ggml_tensor* v_new_out;
            ggml_tensor* out_k_cpy;
            ggml_tensor* out_v_cpy;
        };
        std::vector<LayerTensors> layers(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            lt.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.qkv_w  = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkv_type), qkv_ne0, qkv_ne1);
            lt.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            lt.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_dim);
            lt.o_w    = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type), o_ne0, o_ne1);
            lt.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.gu_w   = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gu_type), gu_ne0, gu_ne1);
            lt.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type), down_ne0, down_ne1);
            lt.k_new_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDim);
            lt.v_new_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kDim);

            if (position > 0)
            {
                lt.k_cached_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, position, num_kv_heads);
                lt.v_cached_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, position, num_kv_heads);
            }
            else
            {
                lt.k_cached_t = nullptr;
                lt.v_cached_t = nullptr;
            }
        }

        // Build computation graph: chain all layers
        ggml_tensor* hidden = current;

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];

            // Attention norm
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), lt.attn_norm_w);

            // Fused QKV projection
            ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, hidden_size, 1);
            ggml_tensor* qkv_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.qkv_w, normed_2d), qDim + 2 * kDim);

            // Split Q, K, V
            ggml_tensor* q_raw = ggml_view_1d(ctx, qkv_flat, qDim, 0);
            ggml_tensor* k_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim) * sizeof(float));
            ggml_tensor* v_raw = ggml_view_1d(ctx, qkv_flat, kDim, static_cast<std::size_t>(qDim + kDim) * sizeof(float));

            // Per-head QK norm
            ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_raw, head_dim, num_heads);
            ggml_tensor* k_2d = ggml_reshape_2d(ctx, k_raw, head_dim, num_kv_heads);

            ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), lt.q_norm_w);
            ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d, eps), lt.k_norm_w);

            // RoPE
            ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, head_dim, num_heads, 1);
            ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, head_dim, num_kv_heads, 1);

            ggml_tensor* q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, nullptr,
                head_dim, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);
            ggml_tensor* k_rope = ggml_rope_ext(ctx, k_3d, pos_tensor, nullptr,
                head_dim, rope_mode, 0, rope_base, rope_freq_scale, 0, 1, 0, 0);

            // Build full KV sequence
            ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
            ggml_tensor* k_rope_perm = ggml_permute(ctx, k_rope, 0, 2, 1, 3);
            ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_raw, head_dim, num_kv_heads, 1);
            ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);

            ggml_tensor* k_full;
            ggml_tensor* v_full;
            if (position > 0)
            {
                k_full = ggml_concat(ctx, lt.k_cached_t, ggml_cont(ctx, k_rope_perm), 1);
                v_full = ggml_concat(ctx, lt.v_cached_t, ggml_cont(ctx, v_perm), 1);
            }
            else
            {
                k_full = ggml_cont(ctx, k_rope_perm);
                v_full = ggml_cont(ctx, v_perm);
            }

            // Flash attention
            ggml_tensor* attn_out = ggml_flash_attn_ext(ctx,
                q_attn, k_full, v_full, nullptr, scale, 0.0f, 0.0f);

            // O projection + residual
            ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, qDim, 1);
            ggml_tensor* o_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.o_w, attn_flat), hidden_size);
            ggml_tensor* residual1 = ggml_add(ctx, hidden, o_flat);

            // FFN
            ggml_tensor* normed2 = ggml_mul(ctx, ggml_rms_norm(ctx, residual1, eps), lt.ffn_norm_w);
            ggml_tensor* normed2_2d = ggml_reshape_2d(ctx, normed2, hidden_size, 1);
            ggml_tensor* gu_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.gu_w, normed2_2d), 2 * intermediate_size);

            ggml_tensor* gate = ggml_view_1d(ctx, gu_flat, intermediate_size, 0);
            ggml_tensor* up = ggml_view_1d(ctx, gu_flat, intermediate_size,
                                           static_cast<std::size_t>(intermediate_size) * sizeof(float));
            ggml_tensor* ffn_hidden = ggml_mul(ctx, ggml_silu(ctx, gate), up);

            ggml_tensor* ffn_2d = ggml_reshape_2d(ctx, ffn_hidden, intermediate_size, 1);
            ggml_tensor* down_flat = ggml_reshape_1d(ctx, ggml_mul_mat(ctx, lt.down_w, ffn_2d), hidden_size);

            // Second residual - this becomes 'hidden' for the next layer
            hidden = ggml_add(ctx, residual1, down_flat);

            // Mark KV outputs for this layer
            ggml_tensor* k_rope_flat = ggml_reshape_1d(ctx, k_rope, kDim);
            lt.out_k_cpy = ggml_cpy(ctx, k_rope_flat, lt.k_new_out);
            ggml_set_output(lt.out_k_cpy);

            ggml_tensor* v_flat = ggml_reshape_1d(ctx, v_raw, kDim);
            lt.out_v_cpy = ggml_cpy(ctx, v_flat, lt.v_new_out);
            ggml_set_output(lt.out_v_cpy);
        }

        // Output: copy hidden state so we can download it
        ggml_tensor* hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* out_hidden = ggml_cpy(ctx, hidden, hidden_out);
        ggml_set_output(out_hidden);

        // Build graph
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * 64 + 256;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(graph, out_hidden);
        for (int l = 0; l < num_layers; l++)
        {
            ggml_build_forward_expand(graph, layers[l].out_k_cpy);
            ggml_build_forward_expand(graph, layers[l].out_v_cpy);
        }

        // Bind weights via cached host_ptr
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        bool can_host_ptr = false;
        if (dev != nullptr)
        {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev, &props);
            can_host_ptr = props.caps.buffer_from_host_ptr;
        }

        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable) {
            if (can_host_ptr && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (cacheable)
                {
                    auto it = g_host_buffer_cache.find(data);
                    if (it != g_host_buffer_cache.end() && it->second.bytes == bytes)
                        buf = it->second.buffer;
                    else
                    {
                        (void)try_create_host_ptr_buffer(g_backend, dev, data, bytes, buf);
                        if (buf != nullptr)
                            g_host_buffer_cache[data] = {buf, bytes};
                    }
                }
                else
                {
                    (void)try_create_host_ptr_buffer(g_backend, dev, data, bytes, buf);
                    if (buf != nullptr)
                        ephemeral_bufs.emplace_back(buf);
                }
                if (buf != nullptr)
                {
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, data);
                    if (st == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            bind_or_mark(lt.qkv_w,  qkv_arr[l],  static_cast<std::size_t>(qkv_bytes), true);
            bind_or_mark(lt.o_w,    o_arr[l],     static_cast<std::size_t>(o_bytes), true);
            bind_or_mark(lt.gu_w,   gu_arr[l],    static_cast<std::size_t>(gu_bytes), true);
            bind_or_mark(lt.down_w, down_arr[l],  static_cast<std::size_t>(down_bytes), true);

            bind_or_mark(lt.attn_norm_w, attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.ffn_norm_w,  ffn_norm_arr[l],  static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.q_norm_w,    q_norm_arr[l],    static_cast<std::size_t>(head_dim) * sizeof(float), true);
            bind_or_mark(lt.k_norm_w,    k_norm_arr[l],    static_cast<std::size_t>(head_dim) * sizeof(float), true);

            if (position > 0)
            {
                bind_or_mark(lt.k_cached_t, kv_caches[l].k_buf.data(), kv_caches[l].k_buf.size() * sizeof(float), false);
                bind_or_mark(lt.v_cached_t, kv_caches[l].v_buf.data(), kv_caches[l].v_buf.size() * sizeof(float), false);
            }
        }

        // Allocate backend buffer for intermediates
        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate backend buffer for model decode.");
            return 0;
        }

        // Upload non-bound tensors
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);

        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(hidden_size) * sizeof(float));

        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));

        // Execute single graph
        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for model decode.");
            return 0;
        }
        ggml_backend_synchronize(g_backend);

        // Download hidden state back to caller
        ggml_backend_tensor_get(hidden_out, hidden_data, 0, static_cast<std::size_t>(hidden_size) * sizeof(float));

        // Download new KV and write to caches
        for (int l = 0; l < num_layers; l++)
        {
            std::vector<float> k_new_buf(static_cast<std::size_t>(kDim));
            std::vector<float> v_new_buf(static_cast<std::size_t>(kDim));
            ggml_backend_tensor_get(layers[l].k_new_out, k_new_buf.data(), 0, static_cast<std::size_t>(kDim) * sizeof(float));
            ggml_backend_tensor_get(layers[l].v_new_out, v_new_buf.data(), 0, static_cast<std::size_t>(kDim) * sizeof(float));

            float* kc = static_cast<float*>(k_cache_arr[l]);
            float* vc = static_cast<float*>(v_cache_arr[l]);
            for (int h = 0; h < num_kv_heads; h++)
            {
                std::memcpy(kc + h * max_seq_len * head_dim + position * head_dim,
                            k_new_buf.data() + h * head_dim,
                            static_cast<std::size_t>(head_dim) * sizeof(float));
                std::memcpy(vc + h * max_seq_len * head_dim + position * head_dim,
                            v_new_buf.data() + h * head_dim,
                            static_cast<std::size_t>(head_dim) * sizeof(float));
            }
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
        set_last_error("Unknown error in transformer model decode.");
        return 0;
    }
}

// ============================================================================
// Gemma4 full-model decode: ALL dense transformer layers in a single GGML graph.
// Handles Gemma4-specific features: GELU activation, V norm, post-attn/FFN norms,
// layer scalars, different head dims per layer type, sliding window, softcap.
// ============================================================================

TSG_EXPORT int TSGgml_Gemma4ModelDecode(
    float* hidden_data, int hidden_size, int num_layers,
    // Per-layer weight pointers (arrays of size num_layers)
    void** attn_norm_arr,
    void** qkv_arr,
    void** q_norm_arr, void** k_norm_arr,
    void** o_arr,
    void** post_attn_norm_arr,
    void** ffn_norm_arr,
    void** gu_arr, void** down_arr,
    void** post_ffn_norm_arr,
    // Per-layer KV caches
    void** k_cache_arr, void** v_cache_arr,
    // Per-layer metadata (arrays of size num_layers)
    int* head_dim_arr,
    int* kv_heads_arr,
    int* cache_size_arr,
    int* is_local_arr,
    int* kv_source_arr,
    float* rope_base_arr,
    float* layer_scalar_arr,
    // Per-layer weight shapes
    int* qkv_type_arr, std::int64_t* qkv_ne0_arr, std::int64_t* qkv_ne1_arr, std::int64_t* qkv_bytes_arr,
    int* o_type_arr, std::int64_t* o_ne0_arr, std::int64_t* o_ne1_arr, std::int64_t* o_bytes_arr,
    int* gu_type_arr, std::int64_t* gu_ne0_arr, std::int64_t* gu_ne1_arr, std::int64_t* gu_bytes_arr,
    int* down_type_arr, std::int64_t* down_ne0_arr, std::int64_t* down_ne1_arr, std::int64_t* down_bytes_arr,
    // Global params
    int num_heads, int position,
    float eps, int sliding_window,
    // RoPE freq_factors (nullable, for global layers with proportional RoPE)
    float* rope_freq_factors, int rope_freq_factors_len,
    int* rope_n_dims_arr,
    // PLE data (nullable)
    float* ple_data, int ple_dim,
    void** ple_gate_arr, int* ple_gate_type_arr, std::int64_t* ple_gate_ne0_arr, std::int64_t* ple_gate_ne1_arr, std::int64_t* ple_gate_bytes_arr,
    void** ple_proj_arr, int* ple_proj_type_arr, std::int64_t* ple_proj_ne0_arr, std::int64_t* ple_proj_ne1_arr, std::int64_t* ple_proj_bytes_arr,
    void** ple_post_norm_arr)
{
    try
    {
        if (!ensure_backend())
            return 0;

        const int totalSeqLen = position + 1;

        // Compute max head dim for context sizing
        int maxHd = 0;
        for (int l = 0; l < num_layers; l++)
            if (head_dim_arr[l] > maxHd) maxHd = head_dim_arr[l];

        // Prepare per-layer contiguous KV cache copies
        struct LayerInfo {
            int hd;
            int kvHeads;
            int qDim;
            int kDim;
            int cacheSize;
            bool isLocal;
            bool isShared;
            int kvSource;
            int attendLen;
            std::vector<float> k_buf;
            std::vector<float> v_buf;
        };
        std::vector<LayerInfo> li(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            auto& info = li[l];
            info.hd = head_dim_arr[l];
            info.kvHeads = kv_heads_arr[l];
            info.qDim = num_heads * info.hd;
            info.kDim = info.kvHeads * info.hd;
            info.kvSource = kv_source_arr[l];
            info.isShared = (info.kvSource != l);

            // For shared layers, use the donor's cache size/local flag
            int kvSrc = info.kvSource;
            info.cacheSize = cache_size_arr[kvSrc];
            info.isLocal = is_local_arr[kvSrc] != 0;
            info.attendLen = info.isLocal ? std::min(totalSeqLen, sliding_window) : totalSeqLen;
        }

        // Extract KV cache data: only for unique KV source layers (avoid duplicate copies)
        std::unordered_map<int, int> kvSrcDone;
        for (int l = 0; l < num_layers; l++)
        {
            auto& info = li[l];
            int kvSrc = info.kvSource;
            if (kvSrcDone.count(kvSrc)) continue;
            kvSrcDone[kvSrc] = 1;

            int windowLen = info.attendLen - 1;
            if (windowLen <= 0) continue;

            auto& srcInfo = li[kvSrc];
            srcInfo.k_buf.resize(static_cast<std::size_t>(windowLen) * info.kDim);
            srcInfo.v_buf.resize(static_cast<std::size_t>(windowLen) * info.kDim);
            float* kc = static_cast<float*>(k_cache_arr[kvSrc]);
            float* vc = static_cast<float*>(v_cache_arr[kvSrc]);

            if (info.isLocal)
            {
                int start = (totalSeqLen > sliding_window) ? totalSeqLen - sliding_window : 0;
                for (int h = 0; h < info.kvHeads; h++)
                {
                    float* kHead = kc + h * info.cacheSize * info.hd;
                    float* vHead = vc + h * info.cacheSize * info.hd;
                    for (int p = 0; p < windowLen; p++)
                    {
                        int cacheIdx = (start + p) % info.cacheSize;
                        std::memcpy(srcInfo.k_buf.data() + (h * windowLen + p) * info.hd,
                                   kHead + cacheIdx * info.hd, info.hd * sizeof(float));
                        std::memcpy(srcInfo.v_buf.data() + (h * windowLen + p) * info.hd,
                                   vHead + cacheIdx * info.hd, info.hd * sizeof(float));
                    }
                }
            }
            else
            {
                for (int h = 0; h < info.kvHeads; h++)
                {
                    std::memcpy(srcInfo.k_buf.data() + h * windowLen * info.hd,
                               kc + h * info.cacheSize * info.hd,
                               static_cast<std::size_t>(windowLen) * info.hd * sizeof(float));
                    std::memcpy(srcInfo.v_buf.data() + h * windowLen * info.hd,
                               vc + h * info.cacheSize * info.hd,
                               static_cast<std::size_t>(windowLen) * info.hd * sizeof(float));
                }
            }
        }

        // Create GGML context
        const std::size_t ctx_size = 32 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("Failed to create ggml context for Gemma4 model decode.");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_tensor* current = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* pos_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

        ggml_tensor* freq_factors_t = nullptr;
        if (rope_freq_factors != nullptr && rope_freq_factors_len > 0)
            freq_factors_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, rope_freq_factors_len);

        // PLE input
        ggml_tensor* ple_input = nullptr;
        if (ple_data != nullptr && ple_dim > 0)
            ple_input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, num_layers * ple_dim);

        struct LayerTensors {
            ggml_tensor* attn_norm_w;
            ggml_tensor* qkv_w;
            ggml_tensor* q_norm_w;
            ggml_tensor* k_norm_w;
            ggml_tensor* o_w;
            ggml_tensor* post_attn_norm_w;
            ggml_tensor* ffn_norm_w;
            ggml_tensor* gu_w;
            ggml_tensor* down_w;
            ggml_tensor* post_ffn_norm_w;
            ggml_tensor* k_cached_t;
            ggml_tensor* v_cached_t;
            ggml_tensor* k_new_out;
            ggml_tensor* v_new_out;
            // PLE
            ggml_tensor* ple_gate_w;
            ggml_tensor* ple_proj_w;
            ggml_tensor* ple_post_norm_w;
        };
        std::vector<LayerTensors> layers(num_layers);

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];

            lt.attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.qkv_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(qkv_type_arr[l]), qkv_ne0_arr[l], qkv_ne1_arr[l]);
            lt.q_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.k_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.hd);
            lt.o_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(o_type_arr[l]), o_ne0_arr[l], o_ne1_arr[l]);
            lt.post_attn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            lt.gu_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(gu_type_arr[l]), gu_ne0_arr[l], gu_ne1_arr[l]);
            lt.down_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(down_type_arr[l]), down_ne0_arr[l], down_ne1_arr[l]);
            lt.post_ffn_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);

            if (!info.isShared)
            {
                lt.k_new_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.kDim);
                lt.v_new_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.kDim);
            }
            else
            {
                lt.k_new_out = nullptr;
                lt.v_new_out = nullptr;
            }

            int windowLen = info.attendLen - 1;
            // For shared layers, reuse donor's cached_t (set below after all layers created)
            if (!info.isShared && windowLen > 0)
            {
                lt.k_cached_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, info.hd, windowLen, info.kvHeads);
                lt.v_cached_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, info.hd, windowLen, info.kvHeads);
            }
            else
            {
                lt.k_cached_t = nullptr;
                lt.v_cached_t = nullptr;
            }

            lt.ple_gate_w = nullptr;
            lt.ple_proj_w = nullptr;
            lt.ple_post_norm_w = nullptr;
            if (ple_data != nullptr && ple_gate_arr != nullptr && ple_gate_arr[l] != nullptr)
            {
                lt.ple_gate_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_gate_type_arr[l]),
                    ple_gate_ne0_arr[l], ple_gate_ne1_arr[l]);
                lt.ple_proj_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(ple_proj_type_arr[l]),
                    ple_proj_ne0_arr[l], ple_proj_ne1_arr[l]);
                lt.ple_post_norm_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
            }
        }

        // Link shared layers to donor KV tensors
        for (int l = 0; l < num_layers; l++)
        {
            auto& info = li[l];
            if (info.isShared)
            {
                layers[l].k_cached_t = layers[info.kvSource].k_cached_t;
                layers[l].v_cached_t = layers[info.kvSource].v_cached_t;
            }
        }

        // Build compute graph
        ggml_tensor* hidden = current;

        // Track new K/V tensors produced by non-shared layers for concat with cached
        std::vector<ggml_tensor*> layer_k_new(num_layers, nullptr);
        std::vector<ggml_tensor*> layer_v_new(num_layers, nullptr);

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];
            float rope_base = rope_base_arr[l];

            // 1. Attn norm
            ggml_tensor* normed = ggml_mul(ctx, ggml_rms_norm(ctx, hidden, eps), lt.attn_norm_w);

            ggml_tensor* normed_2d = ggml_reshape_2d(ctx, normed, hidden_size, 1);
            ggml_tensor* q_rope;
            ggml_tensor* k_full;
            ggml_tensor* v_full;

            if (!info.isShared)
            {
                // 2. Fused QKV projection
                ggml_tensor* qkv_flat = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, lt.qkv_w, normed_2d), info.qDim + 2 * info.kDim);
                ggml_tensor* q_raw = ggml_view_1d(ctx, qkv_flat, info.qDim, 0);
                ggml_tensor* k_raw = ggml_view_1d(ctx, qkv_flat, info.kDim,
                    static_cast<std::size_t>(info.qDim) * sizeof(float));
                ggml_tensor* v_raw = ggml_view_1d(ctx, qkv_flat, info.kDim,
                    static_cast<std::size_t>(info.qDim + info.kDim) * sizeof(float));

                // Per-head Q/K norm
                ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_raw, info.hd, num_heads);
                ggml_tensor* k_2d = ggml_reshape_2d(ctx, k_raw, info.hd, info.kvHeads);
                ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), lt.q_norm_w);
                ggml_tensor* k_normed = ggml_mul(ctx, ggml_rms_norm(ctx, k_2d, eps), lt.k_norm_w);

                // V norm (unweighted RMSNorm)
                ggml_tensor* v_2d = ggml_reshape_2d(ctx, v_raw, info.hd, info.kvHeads);
                ggml_tensor* v_normed = ggml_rms_norm(ctx, v_2d, eps);

                // RoPE (use per-layer n_dims and optional freq_factors)
                int rope_dims = rope_n_dims_arr[l];
                ggml_tensor* rope_ff = info.isLocal ? nullptr : freq_factors_t;
                ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, info.hd, num_heads, 1);
                ggml_tensor* k_3d = ggml_reshape_3d(ctx, k_normed, info.hd, info.kvHeads, 1);
                q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff,
                    rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);
                ggml_tensor* k_rope_t = ggml_rope_ext(ctx, k_3d, pos_tensor, rope_ff,
                    rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);

                ggml_tensor* k_rope_perm = ggml_permute(ctx, k_rope_t, 0, 2, 1, 3);
                ggml_tensor* v_3d = ggml_reshape_3d(ctx, v_normed, info.hd, info.kvHeads, 1);
                ggml_tensor* v_perm = ggml_permute(ctx, v_3d, 0, 2, 1, 3);

                int windowLen = info.attendLen - 1;
                if (windowLen > 0)
                {
                    k_full = ggml_concat(ctx, lt.k_cached_t, ggml_cont(ctx, k_rope_perm), 1);
                    v_full = ggml_concat(ctx, lt.v_cached_t, ggml_cont(ctx, v_perm), 1);
                }
                else
                {
                    k_full = ggml_cont(ctx, k_rope_perm);
                    v_full = ggml_cont(ctx, v_perm);
                }

                // Store new K/V refs for KV output
                layer_k_new[l] = k_rope_t;
                layer_v_new[l] = ggml_reshape_1d(ctx, v_normed, info.kDim);
            }
            else
            {
                // Shared layer: Q-only projection (qkv_w is just Q weight)
                ggml_tensor* q_flat = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, lt.qkv_w, normed_2d), info.qDim);
                ggml_tensor* q_2d = ggml_reshape_2d(ctx, q_flat, info.hd, num_heads);
                ggml_tensor* q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, q_2d, eps), lt.q_norm_w);
                int rope_dims = rope_n_dims_arr[l];
                ggml_tensor* rope_ff = info.isLocal ? nullptr : freq_factors_t;
                ggml_tensor* q_3d = ggml_reshape_3d(ctx, q_normed, info.hd, num_heads, 1);
                q_rope = ggml_rope_ext(ctx, q_3d, pos_tensor, rope_ff,
                    rope_dims, 2, 0, rope_base, 1.0f, 0, 1, 0, 0);

                // Use the donor layer's K/V (already computed earlier in the graph)
                int donor = info.kvSource;
                auto& donorInfo = li[donor];
                int windowLen = info.attendLen - 1;

                if (layer_k_new[donor] != nullptr && windowLen > 0)
                {
                    // Donor's new K/V were produced - concat with cached
                    ggml_tensor* dk_perm = ggml_permute(ctx, layer_k_new[donor], 0, 2, 1, 3);
                    ggml_tensor* dv_1d = layer_v_new[donor];
                    ggml_tensor* dv_3d = ggml_reshape_3d(ctx, dv_1d, donorInfo.hd, donorInfo.kvHeads, 1);
                    ggml_tensor* dv_perm = ggml_permute(ctx, dv_3d, 0, 2, 1, 3);
                    k_full = ggml_concat(ctx, lt.k_cached_t, ggml_cont(ctx, dk_perm), 1);
                    v_full = ggml_concat(ctx, lt.v_cached_t, ggml_cont(ctx, dv_perm), 1);
                }
                else if (layer_k_new[donor] != nullptr)
                {
                    ggml_tensor* dk_perm = ggml_permute(ctx, layer_k_new[donor], 0, 2, 1, 3);
                    ggml_tensor* dv_1d = layer_v_new[donor];
                    ggml_tensor* dv_3d = ggml_reshape_3d(ctx, dv_1d, donorInfo.hd, donorInfo.kvHeads, 1);
                    ggml_tensor* dv_perm = ggml_permute(ctx, dv_3d, 0, 2, 1, 3);
                    k_full = ggml_cont(ctx, dk_perm);
                    v_full = ggml_cont(ctx, dv_perm);
                }
                else if (windowLen > 0)
                {
                    k_full = lt.k_cached_t;
                    v_full = lt.v_cached_t;
                }
                else
                {
                    // No cached data and no new data - should not happen
                    set_last_error("Shared layer has no KV data available.");
                    return 0;
                }
            }

            // Manual attention: scores = softmax(K^T @ Q), output = V_T @ scores
            // Gemma4 uses QK-Norm (per-head RMSNorm on Q/K), so no 1/sqrt(d) scaling
            ggml_tensor* q_attn = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
            ggml_tensor* scores = ggml_mul_mat(ctx, k_full, q_attn);
            scores = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f, 0.0f);
            ggml_tensor* v_t = ggml_cont(ctx, ggml_transpose(ctx, v_full));
            ggml_tensor* attn_out = ggml_mul_mat(ctx, v_t, scores);

            // 8. O projection
            ggml_tensor* attn_flat = ggml_reshape_2d(ctx, attn_out, info.qDim, 1);
            ggml_tensor* o_flat = ggml_reshape_1d(ctx,
                ggml_mul_mat(ctx, lt.o_w, attn_flat), hidden_size);

            // 9. Post-attn norm + residual
            ggml_tensor* post_attn_normed = ggml_mul(ctx,
                ggml_rms_norm(ctx, o_flat, eps), lt.post_attn_norm_w);
            ggml_tensor* residual1 = ggml_add(ctx, hidden, post_attn_normed);

            // 10. FFN: norm → gate_up → GELU*up → down → post_ffn_norm
            ggml_tensor* ffn_normed = ggml_mul(ctx,
                ggml_rms_norm(ctx, residual1, eps), lt.ffn_norm_w);
            ggml_tensor* ffn_normed_2d = ggml_reshape_2d(ctx, ffn_normed, hidden_size, 1);

            std::int64_t intermediate_size = gu_ne1_arr[l] / 2;
            ggml_tensor* gu_flat = ggml_reshape_1d(ctx,
                ggml_mul_mat(ctx, lt.gu_w, ffn_normed_2d), 2 * intermediate_size);
            ggml_tensor* gate = ggml_view_1d(ctx, gu_flat, intermediate_size, 0);
            ggml_tensor* up = ggml_view_1d(ctx, gu_flat, intermediate_size,
                static_cast<std::size_t>(intermediate_size) * sizeof(float));
            ggml_tensor* ffn_hidden = ggml_mul(ctx, ggml_gelu(ctx, gate), up);

            ggml_tensor* ffn_2d = ggml_reshape_2d(ctx, ffn_hidden, intermediate_size, 1);
            ggml_tensor* down_flat = ggml_reshape_1d(ctx,
                ggml_mul_mat(ctx, lt.down_w, ffn_2d), hidden_size);

            // 11. Post-FFN norm + residual
            ggml_tensor* post_ffn_normed = ggml_mul(ctx,
                ggml_rms_norm(ctx, down_flat, eps), lt.post_ffn_norm_w);
            ggml_tensor* residual2 = ggml_add(ctx, residual1, post_ffn_normed);

            // 12. PLE injection (if present)
            if (lt.ple_gate_w != nullptr && ple_input != nullptr)
            {
                ggml_tensor* ple_slice = ggml_view_1d(ctx, ple_input, ple_dim,
                    static_cast<std::size_t>(l) * ple_dim * sizeof(float));
                ggml_tensor* ple_slice_2d = ggml_reshape_2d(ctx, residual2, hidden_size, 1);
                ggml_tensor* ple_gate_proj = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, lt.ple_gate_w, ple_slice_2d), ple_dim);
                ggml_tensor* ple_gated = ggml_mul(ctx, ggml_gelu(ctx, ple_gate_proj), ple_slice);
                ggml_tensor* ple_gated_2d = ggml_reshape_2d(ctx, ple_gated, ple_dim, 1);
                ggml_tensor* ple_proj = ggml_reshape_1d(ctx,
                    ggml_mul_mat(ctx, lt.ple_proj_w, ple_gated_2d), hidden_size);
                ggml_tensor* ple_normed = ggml_mul(ctx,
                    ggml_rms_norm(ctx, ple_proj, eps), lt.ple_post_norm_w);
                residual2 = ggml_add(ctx, residual2, ple_normed);
            }

            // 13. Layer scalar
            float scalar = layer_scalar_arr[l];
            if (std::fabs(scalar - 1.0f) > 1e-6f)
                residual2 = ggml_scale(ctx, residual2, scalar);

            hidden = residual2;

            // Mark KV outputs (only for non-shared layers)
            if (!info.isShared && lt.k_new_out != nullptr && layer_k_new[l] != nullptr)
            {
                ggml_tensor* k_flat = ggml_reshape_1d(ctx, layer_k_new[l], info.kDim);
                lt.k_new_out = ggml_cpy(ctx, k_flat, lt.k_new_out);
                ggml_set_output(lt.k_new_out);

                lt.v_new_out = ggml_cpy(ctx, layer_v_new[l], lt.v_new_out);
                ggml_set_output(lt.v_new_out);
            }
        }

        // Output: copy hidden state
        ggml_tensor* hidden_out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_size);
        ggml_tensor* out_hidden = ggml_cpy(ctx, hidden, hidden_out);
        ggml_set_output(out_hidden);

        // Build graph
        const std::size_t graph_size = static_cast<std::size_t>(num_layers) * 128 + 512;
        ggml_cgraph* graph = ggml_new_graph_custom(ctx, graph_size, false);
        ggml_build_forward_expand(graph, out_hidden);
        for (int l = 0; l < num_layers; l++)
        {
            if (!li[l].isShared && layers[l].k_new_out != nullptr)
            {
                ggml_build_forward_expand(graph, layers[l].k_new_out);
                ggml_build_forward_expand(graph, layers[l].v_new_out);
            }
        }

        // Bind weight data
        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);
        bool can_host_ptr = false;
        if (dev != nullptr)
        {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev, &props);
            can_host_ptr = props.caps.buffer_from_host_ptr;
        }

        struct HostBinding { ggml_tensor* tensor; void* data; std::size_t bytes; };
        std::vector<HostBinding> upload_list;
        std::vector<BufferHandle> ephemeral_bufs;

        auto bind_or_mark = [&](ggml_tensor* t, void* data, std::size_t bytes, bool cacheable) {
            if (t == nullptr || data == nullptr) return;
            if (can_host_ptr && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (cacheable)
                {
                    auto it = g_host_buffer_cache.find(data);
                    if (it != g_host_buffer_cache.end() && it->second.bytes == bytes)
                        buf = it->second.buffer;
                    else
                    {
                        (void)try_create_host_ptr_buffer(g_backend, dev, data, bytes, buf);
                        if (buf != nullptr)
                            g_host_buffer_cache[data] = {buf, bytes};
                    }
                }
                else
                {
                    (void)try_create_host_ptr_buffer(g_backend, dev, data, bytes, buf);
                    if (buf != nullptr)
                        ephemeral_bufs.emplace_back(buf);
                }
                if (buf != nullptr)
                {
                    ggml_status st = ggml_backend_tensor_alloc(buf, t, data);
                    if (st == GGML_STATUS_SUCCESS)
                        return;
                }
            }
            upload_list.push_back({t, data, bytes});
        };

        for (int l = 0; l < num_layers; l++)
        {
            auto& lt = layers[l];
            auto& info = li[l];

            bind_or_mark(lt.qkv_w, qkv_arr[l], static_cast<std::size_t>(qkv_bytes_arr[l]), true);
            bind_or_mark(lt.o_w, o_arr[l], static_cast<std::size_t>(o_bytes_arr[l]), true);
            bind_or_mark(lt.gu_w, gu_arr[l], static_cast<std::size_t>(gu_bytes_arr[l]), true);
            bind_or_mark(lt.down_w, down_arr[l], static_cast<std::size_t>(down_bytes_arr[l]), true);

            bind_or_mark(lt.attn_norm_w, attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.post_attn_norm_w, post_attn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.ffn_norm_w, ffn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.post_ffn_norm_w, post_ffn_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            bind_or_mark(lt.q_norm_w, q_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);
            if (!info.isShared)
                bind_or_mark(lt.k_norm_w, k_norm_arr[l], static_cast<std::size_t>(info.hd) * sizeof(float), true);

            if (!info.isShared)
            {
                int windowLen = info.attendLen - 1;
                if (windowLen > 0)
                {
                    auto& srcInfo = li[info.kvSource];
                    bind_or_mark(lt.k_cached_t, srcInfo.k_buf.data(), srcInfo.k_buf.size() * sizeof(float), false);
                    bind_or_mark(lt.v_cached_t, srcInfo.v_buf.data(), srcInfo.v_buf.size() * sizeof(float), false);
                }
            }

            if (lt.ple_gate_w != nullptr)
            {
                bind_or_mark(lt.ple_gate_w, ple_gate_arr[l], static_cast<std::size_t>(ple_gate_bytes_arr[l]), true);
                bind_or_mark(lt.ple_proj_w, ple_proj_arr[l], static_cast<std::size_t>(ple_proj_bytes_arr[l]), true);
                bind_or_mark(lt.ple_post_norm_w, ple_post_norm_arr[l], static_cast<std::size_t>(hidden_size) * sizeof(float), true);
            }
        }

        // Allocate backend buffer
        BufferHandle buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (buffer.value == nullptr)
        {
            set_last_error("Failed to allocate backend buffer for Gemma4 model decode.");
            return 0;
        }

        // Upload data
        for (auto& u : upload_list)
            ggml_backend_tensor_set(u.tensor, u.data, 0, u.bytes);

        ggml_backend_tensor_set(current, hidden_data, 0, static_cast<std::size_t>(hidden_size) * sizeof(float));

        std::int32_t pos_val = position;
        ggml_backend_tensor_set(pos_tensor, &pos_val, 0, sizeof(std::int32_t));

        if (freq_factors_t != nullptr)
            ggml_backend_tensor_set(freq_factors_t, rope_freq_factors, 0,
                static_cast<std::size_t>(rope_freq_factors_len) * sizeof(float));

        if (ple_input != nullptr && ple_data != nullptr)
            ggml_backend_tensor_set(ple_input, ple_data, 0,
                static_cast<std::size_t>(num_layers) * ple_dim * sizeof(float));

        // Execute single graph
        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("ggml backend graph execution failed for Gemma4 model decode.");
            return 0;
        }
        ggml_backend_synchronize(g_backend);

        // Download hidden state
        ggml_backend_tensor_get(hidden_out, hidden_data, 0, static_cast<std::size_t>(hidden_size) * sizeof(float));

        // Download new KV and write to caches (non-shared layers only)
        for (int l = 0; l < num_layers; l++)
        {
            auto& info = li[l];
            if (info.isShared || layers[l].k_new_out == nullptr)
                continue;

            std::vector<float> k_new_buf(static_cast<std::size_t>(info.kDim));
            std::vector<float> v_new_buf(static_cast<std::size_t>(info.kDim));
            ggml_backend_tensor_get(layers[l].k_new_out, k_new_buf.data(), 0,
                static_cast<std::size_t>(info.kDim) * sizeof(float));
            ggml_backend_tensor_get(layers[l].v_new_out, v_new_buf.data(), 0,
                static_cast<std::size_t>(info.kDim) * sizeof(float));

            float* kc = static_cast<float*>(k_cache_arr[l]);
            float* vc = static_cast<float*>(v_cache_arr[l]);

            int cachePos;
            if (info.isLocal)
                cachePos = position % info.cacheSize;
            else
                cachePos = position;

            for (int h = 0; h < info.kvHeads; h++)
            {
                std::memcpy(kc + h * info.cacheSize * info.hd + cachePos * info.hd,
                           k_new_buf.data() + h * info.hd,
                           static_cast<std::size_t>(info.hd) * sizeof(float));
                std::memcpy(vc + h * info.cacheSize * info.hd + cachePos * info.hd,
                           v_new_buf.data() + h * info.hd,
                           static_cast<std::size_t>(info.hd) * sizeof(float));
            }
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
        set_last_error("Unknown error in Gemma4 model decode.");
        return 0;
    }
}
